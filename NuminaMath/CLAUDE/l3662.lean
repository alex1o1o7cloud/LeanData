import Mathlib

namespace NUMINAMATH_CALUDE_fathers_age_is_32_l3662_366250

/-- The son's current age -/
def sons_age : ℕ := 16

/-- The father's current age -/
def fathers_age : ℕ := 32

/-- Theorem stating that the father's age is 32 -/
theorem fathers_age_is_32 :
  (fathers_age - sons_age = sons_age) ∧ 
  (sons_age = 11 + 5) →
  fathers_age = 32 := by sorry

end NUMINAMATH_CALUDE_fathers_age_is_32_l3662_366250


namespace NUMINAMATH_CALUDE_two_digit_swap_l3662_366215

/-- 
Given a two-digit number with 1 in the tens place and x in the ones place,
if swapping these digits results in a number 18 greater than the original,
then the equation 10x + 1 - (10 + x) = 18 holds.
-/
theorem two_digit_swap (x : ℕ) : 
  (x < 10) →  -- Ensure x is a single digit
  (10 * x + 1) - (10 + x) = 18 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_swap_l3662_366215


namespace NUMINAMATH_CALUDE_mask_digit_assignment_l3662_366270

/-- Represents the four masks in the problem -/
inductive Mask
| elephant
| mouse
| pig
| panda

/-- Assigns a digit to each mask -/
def digit_assignment : Mask → Nat
| Mask.elephant => 6
| Mask.mouse => 4
| Mask.pig => 8
| Mask.panda => 1

/-- Checks if a number is two digits -/
def is_two_digit (n : Nat) : Prop := n ≥ 10 ∧ n < 100

/-- The main theorem statement -/
theorem mask_digit_assignment :
  (∀ m : Mask, digit_assignment m ≤ 9) ∧ 
  (∀ m1 m2 : Mask, m1 ≠ m2 → digit_assignment m1 ≠ digit_assignment m2) ∧
  (∀ m : Mask, is_two_digit ((digit_assignment m) * (digit_assignment m))) ∧
  (∀ m : Mask, (digit_assignment m) * (digit_assignment m) % 10 ≠ digit_assignment m) ∧
  ((digit_assignment Mask.mouse) * (digit_assignment Mask.mouse) % 10 = digit_assignment Mask.elephant) :=
by sorry

end NUMINAMATH_CALUDE_mask_digit_assignment_l3662_366270


namespace NUMINAMATH_CALUDE_remuneration_problem_l3662_366251

/-- Represents the remuneration problem -/
theorem remuneration_problem (annual_clothing : ℕ) (annual_coins : ℕ) 
  (months_worked : ℕ) (received_clothing : ℕ) (received_coins : ℕ) :
  annual_clothing = 1 →
  annual_coins = 10 →
  months_worked = 7 →
  received_clothing = 1 →
  received_coins = 2 →
  ∃ (clothing_value : ℚ),
    clothing_value = 46 / 5 ∧
    (clothing_value + annual_coins : ℚ) / 12 = (clothing_value + received_coins) / months_worked :=
by sorry

end NUMINAMATH_CALUDE_remuneration_problem_l3662_366251


namespace NUMINAMATH_CALUDE_roses_remaining_l3662_366246

/-- Given 3 dozen roses, prove that after giving half away and removing one-third of the remaining flowers, 12 flowers are left. -/
theorem roses_remaining (initial_roses : ℕ) (dozen : ℕ) (half : ℕ → ℕ) (third : ℕ → ℕ) : 
  initial_roses = 3 * dozen → 
  dozen = 12 →
  half n = n / 2 →
  third n = n / 3 →
  third (initial_roses - half initial_roses) = 12 := by
sorry

end NUMINAMATH_CALUDE_roses_remaining_l3662_366246


namespace NUMINAMATH_CALUDE_equilibrium_shift_without_K_change_l3662_366272

-- Define the type for factors that can influence chemical equilibrium
inductive EquilibriumFactor
  | Temperature
  | Concentration
  | Pressure
  | Catalyst

-- Define a function to represent if a factor changes the equilibrium constant
def changesK (factor : EquilibriumFactor) : Prop :=
  match factor with
  | EquilibriumFactor.Temperature => True
  | _ => False

-- Define a function to represent if a factor can shift the equilibrium
def canShiftEquilibrium (factor : EquilibriumFactor) : Prop :=
  match factor with
  | EquilibriumFactor.Temperature => True
  | EquilibriumFactor.Concentration => True
  | EquilibriumFactor.Pressure => True
  | EquilibriumFactor.Catalyst => True

-- Theorem stating that there exists a factor that can shift equilibrium without changing K
theorem equilibrium_shift_without_K_change :
  ∃ (factor : EquilibriumFactor), canShiftEquilibrium factor ∧ ¬changesK factor :=
by
  sorry


end NUMINAMATH_CALUDE_equilibrium_shift_without_K_change_l3662_366272


namespace NUMINAMATH_CALUDE_min_value_fraction_l3662_366294

theorem min_value_fraction (x y : ℝ) (h : x^2 + y^2 = 1) :
  ∃ m : ℝ, m = -1 - Real.sqrt 2 ∧ ∀ z, z = (2*x*y)/(x+y+1) → m ≤ z :=
sorry

end NUMINAMATH_CALUDE_min_value_fraction_l3662_366294


namespace NUMINAMATH_CALUDE_factor_difference_of_squares_l3662_366213

theorem factor_difference_of_squares (x : ℝ) : x^2 - 144 = (x - 12) * (x + 12) := by
  sorry

end NUMINAMATH_CALUDE_factor_difference_of_squares_l3662_366213


namespace NUMINAMATH_CALUDE_problem_solution_l3662_366211

def factorial (n : ℕ) : ℕ := Nat.factorial n

theorem problem_solution (n : ℕ) (h : n * factorial n + 2 * factorial n = 5040) : n = 5 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3662_366211


namespace NUMINAMATH_CALUDE_anna_coins_value_l3662_366221

/-- Represents the number and value of coins Anna has. -/
structure Coins where
  pennies : ℕ
  nickels : ℕ
  total : ℕ
  penny_nickel_relation : pennies = 2 * (nickels + 1) + 1
  total_coins : pennies + nickels = total

/-- The value of Anna's coins in cents -/
def coin_value (c : Coins) : ℕ := c.pennies + 5 * c.nickels

/-- Theorem stating that Anna's coins are worth 31 cents -/
theorem anna_coins_value :
  ∃ c : Coins, c.total = 15 ∧ coin_value c = 31 := by
  sorry


end NUMINAMATH_CALUDE_anna_coins_value_l3662_366221


namespace NUMINAMATH_CALUDE_r_amount_l3662_366266

theorem r_amount (total : ℝ) (r_fraction : ℝ) (h1 : total = 5000) (h2 : r_fraction = 2/3) :
  r_fraction * (total / (1 + r_fraction)) = 2000 := by
  sorry

end NUMINAMATH_CALUDE_r_amount_l3662_366266


namespace NUMINAMATH_CALUDE_max_sum_distances_l3662_366284

-- Define the points A, B, and O
def A : ℝ × ℝ := (3, 0)
def B : ℝ × ℝ := (0, 4)
def O : ℝ × ℝ := (0, 0)

-- Define the incircle of triangle AOB
def incircle (P : ℝ × ℝ) : Prop :=
  ∃ θ : ℝ, P.1 = 1 + Real.cos θ ∧ P.2 = 4/3 + Real.sin θ

-- Define the distance function
def dist_squared (P Q : ℝ × ℝ) : ℝ :=
  (P.1 - Q.1)^2 + (P.2 - Q.2)^2

-- State the theorem
theorem max_sum_distances :
  ∀ P : ℝ × ℝ, incircle P →
    dist_squared P A + dist_squared P B + dist_squared P O ≤ 22 ∧
    ∃ P : ℝ × ℝ, incircle P ∧
      dist_squared P A + dist_squared P B + dist_squared P O = 22 := by
  sorry


end NUMINAMATH_CALUDE_max_sum_distances_l3662_366284


namespace NUMINAMATH_CALUDE_man_rowing_speed_l3662_366229

/-- The speed of a man rowing a boat against the stream, given his speed with the stream and his rate in still water. -/
def speed_against_stream (speed_with_stream : ℝ) (rate_still_water : ℝ) : ℝ :=
  abs (2 * rate_still_water - speed_with_stream)

/-- Theorem: Given a man's speed with the stream of 22 km/h and his rate in still water of 6 km/h, his speed against the stream is 10 km/h. -/
theorem man_rowing_speed 
  (h1 : speed_with_stream = 22)
  (h2 : rate_still_water = 6) :
  speed_against_stream speed_with_stream rate_still_water = 10 := by
  sorry

#eval speed_against_stream 22 6

end NUMINAMATH_CALUDE_man_rowing_speed_l3662_366229


namespace NUMINAMATH_CALUDE_max_sphere_radius_in_glass_l3662_366256

theorem max_sphere_radius_in_glass (x : ℝ) :
  let r := (3 * 2^(1/3)) / 4
  let glass_curve := fun x => x^4
  let sphere_equation := fun (x y : ℝ) => x^2 + (y - r)^2 = r^2
  (∃ y, y = glass_curve x ∧ sphere_equation x y) ∧
  (∀ r' > r, ∃ x y, y < glass_curve x ∧ x^2 + (y - r')^2 = r'^2) ∧
  sphere_equation 0 0 :=
by sorry

end NUMINAMATH_CALUDE_max_sphere_radius_in_glass_l3662_366256


namespace NUMINAMATH_CALUDE_expression_simplification_l3662_366236

theorem expression_simplification (x : ℝ) : 
  3*x*(3*x^2 - 2*x + 1) - 2*x^2 + x = 9*x^3 - 8*x^2 + 4*x := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3662_366236


namespace NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_35_proof_l3662_366271

/-- The smallest four-digit number divisible by 35 -/
def smallest_four_digit_divisible_by_35 : Nat := 1170

/-- A number is four digits if it's between 1000 and 9999 -/
def is_four_digit (n : Nat) : Prop := 1000 ≤ n ∧ n ≤ 9999

theorem smallest_four_digit_divisible_by_35_proof :
  (is_four_digit smallest_four_digit_divisible_by_35) ∧ 
  (smallest_four_digit_divisible_by_35 % 35 = 0) ∧
  (∀ n : Nat, is_four_digit n → n % 35 = 0 → n ≥ smallest_four_digit_divisible_by_35) := by
  sorry

#eval smallest_four_digit_divisible_by_35

end NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_35_proof_l3662_366271


namespace NUMINAMATH_CALUDE_light_travel_distance_l3662_366227

/-- The distance light travels in one year (in miles) -/
def light_year : ℕ := 5870000000000

/-- The number of years we're calculating for -/
def years : ℕ := 500

/-- The expected distance light travels in 500 years (in miles) -/
def expected_distance : ℕ := 2935 * (10^12)

theorem light_travel_distance :
  (light_year * years : ℕ) = expected_distance := by
  sorry

end NUMINAMATH_CALUDE_light_travel_distance_l3662_366227


namespace NUMINAMATH_CALUDE_cubic_root_problem_l3662_366216

theorem cubic_root_problem (a b r s : ℤ) : 
  a ≠ 0 → b ≠ 0 → 
  (∀ x : ℤ, x^3 + a*x^2 + b*x + 16*a = (x - r)^2 * (x - s)) →
  (r = s ∨ r = -2 ∨ s = -2) →
  (|a*b| = 272) :=
sorry

end NUMINAMATH_CALUDE_cubic_root_problem_l3662_366216


namespace NUMINAMATH_CALUDE_occupation_combinations_eq_636_l3662_366220

/-- The number of Earth-like planets -/
def earth_like_planets : ℕ := 5

/-- The number of Mars-like planets -/
def mars_like_planets : ℕ := 9

/-- The number of colonization units required for an Earth-like planet -/
def earth_units : ℕ := 2

/-- The number of colonization units required for a Mars-like planet -/
def mars_units : ℕ := 1

/-- The total number of available colonization units -/
def total_units : ℕ := 14

/-- Function to calculate the number of combinations of occupying planets -/
def occupation_combinations : ℕ := sorry

theorem occupation_combinations_eq_636 : occupation_combinations = 636 := by sorry

end NUMINAMATH_CALUDE_occupation_combinations_eq_636_l3662_366220


namespace NUMINAMATH_CALUDE_cable_intersections_6_8_l3662_366218

/-- The number of pairwise intersections of cables connecting houses across a street -/
def cable_intersections (n m : ℕ) : ℕ :=
  Nat.choose n 2 * Nat.choose m 2

/-- Theorem stating the number of cable intersections for 6 houses on one side and 8 on the other -/
theorem cable_intersections_6_8 :
  cable_intersections 6 8 = 420 := by
  sorry

end NUMINAMATH_CALUDE_cable_intersections_6_8_l3662_366218


namespace NUMINAMATH_CALUDE_largest_negative_integer_and_abs_property_l3662_366249

theorem largest_negative_integer_and_abs_property :
  (∀ n : ℤ, n < 0 → n ≤ -1) ∧
  (∀ x : ℝ, |x| = x → x ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_largest_negative_integer_and_abs_property_l3662_366249


namespace NUMINAMATH_CALUDE_sum_three_numbers_l3662_366281

theorem sum_three_numbers (x y z M : ℝ) : 
  x + y + z = 90 ∧ 
  x - 5 = M ∧ 
  y + 5 = M ∧ 
  5 * z = M → 
  M = 450 / 11 := by
sorry

end NUMINAMATH_CALUDE_sum_three_numbers_l3662_366281


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_div_i_l3662_366257

-- Define the complex number z
def z : ℂ := Complex.mk 1 (-3)

-- Theorem statement
theorem imaginary_part_of_z_div_i : (z / Complex.I).im = -1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_div_i_l3662_366257


namespace NUMINAMATH_CALUDE_reebok_cost_is_35_l3662_366279

/-- The cost of a pair of Reebok shoes -/
def reebok_cost : ℚ := 35

/-- Alice's sales quota -/
def quota : ℚ := 1000

/-- The cost of a pair of Adidas shoes -/
def adidas_cost : ℚ := 45

/-- The cost of a pair of Nike shoes -/
def nike_cost : ℚ := 60

/-- The number of Nike shoes sold -/
def nike_sold : ℕ := 8

/-- The number of Adidas shoes sold -/
def adidas_sold : ℕ := 6

/-- The number of Reebok shoes sold -/
def reebok_sold : ℕ := 9

/-- The amount by which Alice exceeded her quota -/
def excess : ℚ := 65

theorem reebok_cost_is_35 :
  reebok_cost * reebok_sold + nike_cost * nike_sold + adidas_cost * adidas_sold = quota + excess :=
by sorry

end NUMINAMATH_CALUDE_reebok_cost_is_35_l3662_366279


namespace NUMINAMATH_CALUDE_f_max_at_neg_two_l3662_366260

-- Define the function
def f (x : ℝ) : ℝ := -2 * x^2 - 8 * x + 16

-- State the theorem
theorem f_max_at_neg_two :
  ∀ x : ℝ, f x ≤ f (-2) :=
by
  sorry

end NUMINAMATH_CALUDE_f_max_at_neg_two_l3662_366260


namespace NUMINAMATH_CALUDE_gcd_linear_combination_l3662_366214

theorem gcd_linear_combination (a b : ℤ) : Int.gcd (5*a + 3*b) (13*a + 8*b) = Int.gcd a b := by
  sorry

end NUMINAMATH_CALUDE_gcd_linear_combination_l3662_366214


namespace NUMINAMATH_CALUDE_farm_tax_calculation_l3662_366274

/-- Represents the farm tax calculation for a village and an individual landowner. -/
theorem farm_tax_calculation 
  (total_tax : ℝ) 
  (individual_land_ratio : ℝ) 
  (h1 : total_tax = 3840) 
  (h2 : individual_land_ratio = 0.5) : 
  individual_land_ratio * total_tax = 1920 := by
  sorry


end NUMINAMATH_CALUDE_farm_tax_calculation_l3662_366274


namespace NUMINAMATH_CALUDE_puzzle_missing_pieces_l3662_366293

theorem puzzle_missing_pieces 
  (total_pieces : ℕ) 
  (border_pieces : ℕ) 
  (trevor_pieces : ℕ) 
  (joe_multiplier : ℕ) : 
  total_pieces = 500 →
  border_pieces = 75 →
  trevor_pieces = 105 →
  joe_multiplier = 3 →
  total_pieces - border_pieces - (trevor_pieces + joe_multiplier * trevor_pieces) = 5 :=
by
  sorry

#check puzzle_missing_pieces

end NUMINAMATH_CALUDE_puzzle_missing_pieces_l3662_366293


namespace NUMINAMATH_CALUDE_smallest_k_multiple_of_144_l3662_366247

def sum_of_squares (k : ℕ+) : ℕ := k.val * (k.val + 1) * (2 * k.val + 1) / 6

theorem smallest_k_multiple_of_144 :
  ∀ k : ℕ+, k.val < 26 → ¬(144 ∣ sum_of_squares k) ∧
  144 ∣ sum_of_squares 26 :=
sorry

end NUMINAMATH_CALUDE_smallest_k_multiple_of_144_l3662_366247


namespace NUMINAMATH_CALUDE_right_angled_triangle_unique_k_l3662_366248

/-- A triangle with side lengths 13, 17, and k is right-angled if and only if k = 21 -/
theorem right_angled_triangle_unique_k : ∃! (k : ℕ), k > 0 ∧ 
  (13^2 + 17^2 = k^2 ∨ 13^2 + k^2 = 17^2 ∨ 17^2 + k^2 = 13^2) := by
  sorry

end NUMINAMATH_CALUDE_right_angled_triangle_unique_k_l3662_366248


namespace NUMINAMATH_CALUDE_simplify_expression_l3662_366282

theorem simplify_expression (x y : ℝ) (h : x * y ≠ 0) :
  ((x^3 + 2) / x) * ((y^3 + 2) / y) + ((x^3 - 2) / y) * ((y^3 - 2) / x) = 2 * x^2 * y^2 + 8 / (x * y) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3662_366282


namespace NUMINAMATH_CALUDE_soda_price_ratio_l3662_366252

theorem soda_price_ratio (v : ℝ) (p : ℝ) (hv : v > 0) (hp : p > 0) :
  let x_volume := 1.3 * v
  let x_price := 0.85 * p
  let x_unit_price := x_price / x_volume
  let y_unit_price := p / v
  x_unit_price / y_unit_price = 17 / 26 := by
sorry

end NUMINAMATH_CALUDE_soda_price_ratio_l3662_366252


namespace NUMINAMATH_CALUDE_anthony_final_pet_count_l3662_366222

/-- The number of pets Anthony has after a series of events --/
def final_pet_count (initial_pets : ℕ) : ℕ :=
  let pets_after_loss := initial_pets - (initial_pets * 12 / 100)
  let pets_after_contest := pets_after_loss + 7
  let pets_giving_birth := pets_after_contest / 4
  let new_offspring := pets_giving_birth * 2
  let pets_before_deaths := pets_after_contest + new_offspring
  pets_before_deaths - (pets_before_deaths / 10)

/-- Theorem stating that Anthony ends up with 62 pets --/
theorem anthony_final_pet_count :
  final_pet_count 45 = 62 := by
  sorry

end NUMINAMATH_CALUDE_anthony_final_pet_count_l3662_366222


namespace NUMINAMATH_CALUDE_triangle_area_formulas_l3662_366288

/-- Given a triangle with area t, semiperimeter s, angles α, β, γ, and sides a, b, c,
    prove two formulas for the area. -/
theorem triangle_area_formulas (t s a b c α β γ : ℝ) 
  (h_area : t > 0)
  (h_semiperimeter : s > 0)
  (h_sides : a > 0 ∧ b > 0 ∧ c > 0)
  (h_angles : α > 0 ∧ β > 0 ∧ γ > 0)
  (h_sum_angles : α + β + γ = π)
  (h_semiperimeter_def : s = (a + b + c) / 2) :
  (t = s^2 * Real.tan (α/2) * Real.tan (β/2) * Real.tan (γ/2)) ∧
  (t = (a*b*c/s) * Real.cos (α/2) * Real.cos (β/2) * Real.cos (γ/2)) := by
  sorry


end NUMINAMATH_CALUDE_triangle_area_formulas_l3662_366288


namespace NUMINAMATH_CALUDE_inequality_proof_l3662_366217

theorem inequality_proof (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h4 : x + y + z = 1/2) :
  (Real.sqrt x) / (4 * x + 1) + (Real.sqrt y) / (4 * y + 1) + (Real.sqrt z) / (4 * z + 1) ≤ 3 * Real.sqrt 6 / 10 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l3662_366217


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_problem_l3662_366245

theorem arithmetic_geometric_sequence_problem :
  ∃ (a b c : ℝ),
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    b - a = c - b ∧
    a + b + c = 15 ∧
    (a + 2) * (c + 13) = (b + 5)^2 ∧
    a = 3 ∧ b = 5 ∧ c = 7 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_problem_l3662_366245


namespace NUMINAMATH_CALUDE_total_strikes_is_180_l3662_366296

/-- Calculates the total number of strikes made by a clock in a 24-hour period. -/
def total_strikes : ℕ :=
  let hourly_strikes := 12 * 13 / 2 * 2  -- Sum of 1 to 12, twice
  let half_hour_strikes := 24            -- One strike every half hour (excluding full hours)
  hourly_strikes + half_hour_strikes

/-- Theorem stating that the total number of strikes in a 24-hour period is 180. -/
theorem total_strikes_is_180 : total_strikes = 180 := by
  sorry

end NUMINAMATH_CALUDE_total_strikes_is_180_l3662_366296


namespace NUMINAMATH_CALUDE_last_two_digits_of_1032_power_1032_l3662_366212

theorem last_two_digits_of_1032_power_1032 : ∃ n : ℕ, 1032^1032 ≡ 76 [ZMOD 100] :=
sorry

end NUMINAMATH_CALUDE_last_two_digits_of_1032_power_1032_l3662_366212


namespace NUMINAMATH_CALUDE_g_difference_l3662_366219

/-- The function g(x) = 2x^3 + 5x^2 - 2x - 1 -/
def g (x : ℝ) : ℝ := 2 * x^3 + 5 * x^2 - 2 * x - 1

/-- Theorem stating that g(x+h) - g(x) = h(6x^2 + 6xh + 2h^2 + 10x + 5h - 2) for all x and h -/
theorem g_difference (x h : ℝ) : g (x + h) - g x = h * (6 * x^2 + 6 * x * h + 2 * h^2 + 10 * x + 5 * h - 2) := by
  sorry

end NUMINAMATH_CALUDE_g_difference_l3662_366219


namespace NUMINAMATH_CALUDE_different_tens_digit_probability_l3662_366255

def lower_bound : ℕ := 30
def upper_bound : ℕ := 89
def num_integers : ℕ := 7

def favorable_outcomes : ℕ := 27000000
def total_outcomes : ℕ := Nat.choose (upper_bound - lower_bound + 1) num_integers

theorem different_tens_digit_probability :
  (favorable_outcomes : ℚ) / total_outcomes = 6750 / 9655173 := by
  sorry

end NUMINAMATH_CALUDE_different_tens_digit_probability_l3662_366255


namespace NUMINAMATH_CALUDE_transaction_fraction_l3662_366200

theorem transaction_fraction :
  let mabel_transactions : ℕ := 90
  let anthony_transactions : ℕ := mabel_transactions + mabel_transactions / 10
  let jade_transactions : ℕ := 82
  let cal_transactions : ℕ := jade_transactions - 16
  (cal_transactions : ℚ) / (anthony_transactions : ℚ) = 2 / 3 := by
sorry

end NUMINAMATH_CALUDE_transaction_fraction_l3662_366200


namespace NUMINAMATH_CALUDE_inequality_theorem_l3662_366280

theorem inequality_theorem (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : a + b + c ≥ a * b * c) :
  (2 / a + 3 / b + 6 / c ≥ 6 ∧ 2 / b + 3 / c + 6 / a ≥ 6) ∨
  (2 / a + 3 / b + 6 / c ≥ 6 ∧ 2 / c + 3 / a + 6 / b ≥ 6) ∨
  (2 / b + 3 / c + 6 / a ≥ 6 ∧ 2 / c + 3 / a + 6 / b ≥ 6) :=
by sorry

end NUMINAMATH_CALUDE_inequality_theorem_l3662_366280


namespace NUMINAMATH_CALUDE_sum_divisible_by_addends_l3662_366292

theorem sum_divisible_by_addends : 
  ∃ (a b c : ℕ), 
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
    (a + b + c) % a = 0 ∧ 
    (a + b + c) % b = 0 ∧ 
    (a + b + c) % c = 0 :=
by sorry

end NUMINAMATH_CALUDE_sum_divisible_by_addends_l3662_366292


namespace NUMINAMATH_CALUDE_equation_solutions_l3662_366204

theorem equation_solutions : 
  let f (x : ℝ) := (x - 1) * (x - 2) * (x - 3) * (x - 4) * (x - 3) * (x - 2) * (x - 1) * (x - 5)
  let g (x : ℝ) := (x - 2) * (x - 4) * (x - 2) * (x - 5)
  ∀ x : ℝ, (g x ≠ 0 ∧ f x / g x = 1) ↔ (x = 2 + Real.sqrt 2 ∨ x = 2 - Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l3662_366204


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3662_366254

/-- Given an arithmetic sequence {a_n} with the specified conditions, 
    prove that a₅ + a₈ + a₁₁ = 15 -/
theorem arithmetic_sequence_sum (a : ℕ → ℤ) 
  (h1 : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1))  -- arithmetic sequence
  (h2 : a 1 + a 4 + a 7 = 39)
  (h3 : a 2 + a 5 + a 8 = 33) :
  a 5 + a 8 + a 11 = 15 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3662_366254


namespace NUMINAMATH_CALUDE_units_digit_sum_l3662_366225

theorem units_digit_sum (n : ℕ) : (35^87 + 3^45) % 10 = 8 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_sum_l3662_366225


namespace NUMINAMATH_CALUDE_solve_for_a_l3662_366291

theorem solve_for_a (a : ℝ) (h : 2 * 2^2 * a = 2^6) : a = 8 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_a_l3662_366291


namespace NUMINAMATH_CALUDE_max_sum_squares_l3662_366234

theorem max_sum_squares (a b c d : ℝ) 
  (h1 : a + b = 18)
  (h2 : a * b + c + d = 91)
  (h3 : a * d + b * c = 195)
  (h4 : c * d = 120) :
  a^2 + b^2 + c^2 + d^2 ≤ 82 ∧ 
  ∃ (a' b' c' d' : ℝ), 
    a' + b' = 18 ∧
    a' * b' + c' + d' = 91 ∧
    a' * d' + b' * c' = 195 ∧
    c' * d' = 120 ∧
    a'^2 + b'^2 + c'^2 + d'^2 = 82 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_squares_l3662_366234


namespace NUMINAMATH_CALUDE_rain_is_random_event_l3662_366231

/-- An event is random if its probability is strictly between 0 and 1 -/
def is_random_event (p : ℝ) : Prop := 0 < p ∧ p < 1

/-- The probability of rain in Xiangyang tomorrow -/
def rain_probability : ℝ := 0.75

theorem rain_is_random_event : is_random_event rain_probability := by
  sorry

end NUMINAMATH_CALUDE_rain_is_random_event_l3662_366231


namespace NUMINAMATH_CALUDE_real_part_of_z_l3662_366240

theorem real_part_of_z (z : ℂ) (h : (1 + Complex.I) * z = 2 * Complex.I) : 
  Complex.re z = 1 := by sorry

end NUMINAMATH_CALUDE_real_part_of_z_l3662_366240


namespace NUMINAMATH_CALUDE_solve_for_n_l3662_366265

/-- The number of balls labeled '2' -/
def n : ℕ := sorry

/-- The total number of balls in the bag -/
def total_balls : ℕ := n + 2

/-- The probability of drawing a ball labeled '2' -/
def prob_2 : ℚ := n / total_balls

theorem solve_for_n : 
  (prob_2 = 1/3) → n = 1 :=
by sorry

end NUMINAMATH_CALUDE_solve_for_n_l3662_366265


namespace NUMINAMATH_CALUDE_mary_coins_l3662_366209

theorem mary_coins (dimes quarters : ℕ) 
  (h1 : quarters = 2 * dimes + 7)
  (h2 : (0.10 : ℚ) * dimes + (0.25 : ℚ) * quarters = 10.15) : quarters = 35 := by
  sorry

end NUMINAMATH_CALUDE_mary_coins_l3662_366209


namespace NUMINAMATH_CALUDE_red_snapper_cost_l3662_366205

/-- The cost of a Red snapper given the fisherman's daily catch and earnings -/
theorem red_snapper_cost (red_snappers : ℕ) (tunas : ℕ) (tuna_cost : ℚ) (daily_earnings : ℚ) : 
  red_snappers = 8 → tunas = 14 → tuna_cost = 2 → daily_earnings = 52 → 
  (daily_earnings - (tunas * tuna_cost)) / red_snappers = 3 := by
sorry

end NUMINAMATH_CALUDE_red_snapper_cost_l3662_366205


namespace NUMINAMATH_CALUDE_geometric_sequence_a_value_l3662_366202

theorem geometric_sequence_a_value (a : ℝ) :
  (1 / (a - 1)) * (a + 1) = (a + 1) * (a^2 - 1) →
  a = 0 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_a_value_l3662_366202


namespace NUMINAMATH_CALUDE_right_triangle_among_sets_l3662_366228

def is_right_triangle (a b c : ℕ) : Prop :=
  a * a + b * b = c * c ∨ a * a + c * c = b * b ∨ b * b + c * c = a * a

theorem right_triangle_among_sets :
  ¬ is_right_triangle 1 2 3 ∧
  ¬ is_right_triangle 2 3 4 ∧
  is_right_triangle 3 4 5 ∧
  ¬ is_right_triangle 4 5 6 :=
sorry

end NUMINAMATH_CALUDE_right_triangle_among_sets_l3662_366228


namespace NUMINAMATH_CALUDE_equilateral_iff_sum_zero_l3662_366235

-- Define j as a complex number representing a rotation by 120°
noncomputable def j : ℂ := Complex.exp (2 * Real.pi * Complex.I / 3)

-- Define the property of j
axiom j_cube : j ^ 3 = 1
axiom j_sum : 1 + j + j^2 = 0

-- Define a triangle in the complex plane
structure Triangle :=
  (A B C : ℂ)

-- Define the property of being equilateral
def is_equilateral (t : Triangle) : Prop :=
  Complex.abs (t.B - t.A) = Complex.abs (t.C - t.B) ∧
  Complex.abs (t.C - t.B) = Complex.abs (t.A - t.C)

-- State the theorem
theorem equilateral_iff_sum_zero (t : Triangle) :
  is_equilateral t ↔ t.A + j * t.B + j^2 * t.C = 0 :=
sorry

end NUMINAMATH_CALUDE_equilateral_iff_sum_zero_l3662_366235


namespace NUMINAMATH_CALUDE_robin_gum_packages_l3662_366207

/-- The number of pieces of gum in each package -/
def pieces_per_package : ℕ := 15

/-- The total number of pieces of gum Robin has -/
def total_pieces : ℕ := 135

/-- The number of packages Robin has -/
def num_packages : ℕ := total_pieces / pieces_per_package

theorem robin_gum_packages : num_packages = 9 := by
  sorry

end NUMINAMATH_CALUDE_robin_gum_packages_l3662_366207


namespace NUMINAMATH_CALUDE_equal_division_possible_l3662_366264

/-- Represents the state of the three vessels -/
structure VesselState :=
  (v1 v2 v3 : ℕ)

/-- Represents a pouring action between two vessels -/
inductive PourAction
  | from1to2 | from1to3 | from2to1 | from2to3 | from3to1 | from3to2

/-- Applies a pouring action to a vessel state -/
def applyPour (state : VesselState) (action : PourAction) : VesselState :=
  sorry

/-- Checks if a vessel state is valid (respects capacities) -/
def isValidState (state : VesselState) : Prop :=
  state.v1 ≤ 3 ∧ state.v2 ≤ 5 ∧ state.v3 ≤ 8

/-- Checks if a sequence of pours is valid -/
def isValidPourSequence (initialState : VesselState) (pours : List PourAction) : Prop :=
  sorry

/-- The theorem stating that it's possible to divide the liquid equally -/
theorem equal_division_possible : ∃ (pours : List PourAction),
  isValidPourSequence ⟨0, 0, 8⟩ pours ∧
  let finalState := pours.foldl applyPour ⟨0, 0, 8⟩
  finalState.v2 = 4 ∧ finalState.v3 = 4 :=
  sorry

end NUMINAMATH_CALUDE_equal_division_possible_l3662_366264


namespace NUMINAMATH_CALUDE_trig_expression_value_l3662_366253

theorem trig_expression_value (α : Real) (h : Real.tan α = 3) :
  (6 * Real.sin α - 2 * Real.cos α) / (5 * Real.cos α + 3 * Real.sin α) = 8 / 7 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_value_l3662_366253


namespace NUMINAMATH_CALUDE_rectangular_plot_area_l3662_366244

/-- A rectangular plot with length thrice its width and width of 12 meters has an area of 432 square meters. -/
theorem rectangular_plot_area : 
  ∀ (width length area : ℝ),
  width = 12 →
  length = 3 * width →
  area = length * width →
  area = 432 := by
sorry

end NUMINAMATH_CALUDE_rectangular_plot_area_l3662_366244


namespace NUMINAMATH_CALUDE_function_sum_equals_four_l3662_366224

/-- Given a function f(x) = ax^7 - bx^5 + cx^3 + 2, prove that f(5) + f(-5) = 4 -/
theorem function_sum_equals_four (a b c m : ℝ) :
  let f : ℝ → ℝ := λ x ↦ a * x^7 - b * x^5 + c * x^3 + 2
  f (-5) = m →
  f 5 + f (-5) = 4 := by
  sorry

end NUMINAMATH_CALUDE_function_sum_equals_four_l3662_366224


namespace NUMINAMATH_CALUDE_cylinder_surface_area_l3662_366273

/-- The surface area of a cylinder with diameter and height both equal to 4 is 24π. -/
theorem cylinder_surface_area : 
  ∀ (d h : ℝ), d = 4 → h = 4 → 2 * π * (d / 2) * (d / 2 + h) = 24 * π :=
by
  sorry

end NUMINAMATH_CALUDE_cylinder_surface_area_l3662_366273


namespace NUMINAMATH_CALUDE_donation_to_third_home_l3662_366203

/-- Proves that the donation to the third home is $230.00 -/
theorem donation_to_third_home 
  (total_donation : ℝ) 
  (first_home_donation : ℝ) 
  (second_home_donation : ℝ)
  (h1 : total_donation = 700)
  (h2 : first_home_donation = 245)
  (h3 : second_home_donation = 225) :
  total_donation - first_home_donation - second_home_donation = 230 := by
  sorry

end NUMINAMATH_CALUDE_donation_to_third_home_l3662_366203


namespace NUMINAMATH_CALUDE_perpendicular_line_through_point_l3662_366258

/-- Given a line L1 with equation 3x - 6y = 9 and a point P (-2, -3),
    prove that the line L2 with equation y = -2x - 7 is perpendicular to L1
    and passes through P. -/
theorem perpendicular_line_through_point 
  (L1 : Real → Real → Prop) 
  (P : Real × Real) 
  (L2 : Real → Real → Prop) : 
  (∀ x y, L1 x y ↔ 3 * x - 6 * y = 9) →
  P = (-2, -3) →
  (∀ x y, L2 x y ↔ y = -2 * x - 7) →
  (∀ x₁ y₁ x₂ y₂, L1 x₁ y₁ → L1 x₂ y₂ → (x₂ - x₁) * (x₂ - x₁) + (y₂ - y₁) * (y₂ - y₁) ≠ 0 →
    (x₂ - x₁) * (P.1 - x₁) + (y₂ - y₁) * (P.2 - y₁) = 0) →
  L2 P.1 P.2 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_line_through_point_l3662_366258


namespace NUMINAMATH_CALUDE_ellipse_equation_l3662_366268

/-- The equation of an ellipse with given parameters -/
theorem ellipse_equation (ε x₀ y₀ α : ℝ) (ε_pos : 0 < ε) (ε_lt_one : ε < 1) :
  let c : ℝ := (y₀ - x₀ * Real.tan α) / Real.tan α
  let a : ℝ := c / ε
  let b : ℝ := Real.sqrt (a^2 - c^2)
  ∀ x y : ℝ, (x^2 / a^2 + y^2 / b^2 = 1) ↔
    (x^2 / (c^2 / ε^2) + y^2 / ((c^2 / ε^2) - c^2) = 1) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_equation_l3662_366268


namespace NUMINAMATH_CALUDE_prob_red_blue_black_l3662_366277

/-- Represents the color of a marble -/
inductive MarbleColor
  | Red
  | Green
  | Blue
  | White
  | Black
  | Yellow

/-- Represents a bag of marbles -/
structure MarbleBag where
  total : ℕ
  colors : List MarbleColor
  probs : MarbleColor → ℚ

/-- The probability of drawing a marble of a specific color or set of colors -/
def prob (bag : MarbleBag) (colors : List MarbleColor) : ℚ :=
  colors.map bag.probs |>.sum

/-- Theorem stating the probability of drawing a red, blue, or black marble -/
theorem prob_red_blue_black (bag : MarbleBag) :
  bag.total = 120 ∧
  bag.colors = [MarbleColor.Red, MarbleColor.Green, MarbleColor.Blue,
                MarbleColor.White, MarbleColor.Black, MarbleColor.Yellow] ∧
  bag.probs MarbleColor.White = 1/5 ∧
  bag.probs MarbleColor.Green = 3/10 ∧
  bag.probs MarbleColor.Yellow = 1/6 →
  prob bag [MarbleColor.Red, MarbleColor.Blue, MarbleColor.Black] = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_prob_red_blue_black_l3662_366277


namespace NUMINAMATH_CALUDE_largest_number_l3662_366290

theorem largest_number : 
  let a := 0.938
  let b := 0.9389
  let c := 0.93809
  let d := 0.839
  let e := 0.8909
  b > a ∧ b > c ∧ b > d ∧ b > e := by
  sorry

end NUMINAMATH_CALUDE_largest_number_l3662_366290


namespace NUMINAMATH_CALUDE_system_two_solutions_l3662_366278

/-- The system of equations has exactly two solutions if and only if a = 49 or a = 169 -/
theorem system_two_solutions (a : ℝ) :
  (∃! (s : Set (ℝ × ℝ)), s.ncard = 2 ∧ 
    (∀ (x y : ℝ), (x, y) ∈ s ↔ 
      (|x + y + 5| + |y - x + 5| = 10 ∧
       (|x| - 12)^2 + (|y| - 5)^2 = a))) ↔
  (a = 49 ∨ a = 169) :=
sorry

end NUMINAMATH_CALUDE_system_two_solutions_l3662_366278


namespace NUMINAMATH_CALUDE_sets_theorem_l3662_366263

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 4 = 0}
def B : Set ℝ := {y | ∃ x, y = x^2 - 4}

-- Statement to prove
theorem sets_theorem :
  (A ∩ B = A) ∧ (A ∪ B = B) := by sorry

end NUMINAMATH_CALUDE_sets_theorem_l3662_366263


namespace NUMINAMATH_CALUDE_apple_cost_price_l3662_366299

/-- The cost price of an apple, given its selling price and loss ratio. -/
def cost_price (selling_price : ℚ) (loss_ratio : ℚ) : ℚ :=
  selling_price / (1 - loss_ratio)

/-- Theorem: The cost price of an apple is 20.4 when sold for 17 with a 1/6 loss. -/
theorem apple_cost_price :
  cost_price 17 (1/6) = 20.4 := by
sorry

end NUMINAMATH_CALUDE_apple_cost_price_l3662_366299


namespace NUMINAMATH_CALUDE_emily_age_l3662_366286

theorem emily_age :
  ∀ (e g : ℕ),
  g = 15 * e →
  g - e = 70 →
  e = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_emily_age_l3662_366286


namespace NUMINAMATH_CALUDE_exists_special_sequence_l3662_366223

/-- A sequence of natural numbers satisfying specific conditions -/
def SpecialSequence (F : ℕ → ℕ) : Prop :=
  (∀ k, ∃ n, F n = k) ∧
  (∀ k, Set.Infinite {n | F n = k}) ∧
  (∀ n ≥ 2, F (F (n^163)) = F (F n) + F (F 361))

/-- There exists a sequence satisfying the SpecialSequence conditions -/
theorem exists_special_sequence : ∃ F, SpecialSequence F := by
  sorry

end NUMINAMATH_CALUDE_exists_special_sequence_l3662_366223


namespace NUMINAMATH_CALUDE_average_sale_is_7500_l3662_366242

def monthly_sales : List ℕ := [7435, 7920, 7855, 8230, 7560, 6000]

def total_sales : ℕ := monthly_sales.sum

def num_months : ℕ := monthly_sales.length

def average_sale : ℚ := (total_sales : ℚ) / (num_months : ℚ)

theorem average_sale_is_7500 : average_sale = 7500 := by
  sorry

end NUMINAMATH_CALUDE_average_sale_is_7500_l3662_366242


namespace NUMINAMATH_CALUDE_simplify_expression_evaluate_expression_complex_expression_l3662_366267

-- Problem 1
theorem simplify_expression (a b : ℝ) :
  10 * (a - b)^2 - 12 * (a - b)^2 + 9 * (a - b)^2 = 7 * (a - b)^2 := by
  sorry

-- Problem 2
theorem evaluate_expression (x y : ℝ) (h : x^2 - 2*y = -5) :
  4 * x^2 - 8 * y + 24 = 4 := by
  sorry

-- Problem 3
theorem complex_expression (a b c d : ℝ) 
  (h1 : a - 2*b = 1009 + 1/2)
  (h2 : 2*b - c = -2024 - 2/3)
  (h3 : c - d = 1013 + 1/6) :
  (a - c) + (2*b - d) - (2*b - c) = -2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_evaluate_expression_complex_expression_l3662_366267


namespace NUMINAMATH_CALUDE_additional_men_problem_l3662_366241

/-- Calculates the number of additional men given initial conditions and new duration -/
def additional_men (initial_men : ℕ) (initial_days : ℚ) (new_days : ℚ) : ℚ :=
  (initial_men * initial_days / new_days) - initial_men

theorem additional_men_problem :
  let initial_men : ℕ := 1000
  let initial_days : ℚ := 17
  let new_days : ℚ := 11.333333333333334
  additional_men initial_men initial_days new_days = 500 := by
sorry

end NUMINAMATH_CALUDE_additional_men_problem_l3662_366241


namespace NUMINAMATH_CALUDE_evaluate_expression_l3662_366259

theorem evaluate_expression : Real.sqrt ((4 / 3) * (1 / 15 + 1 / 25)) = 4 * Real.sqrt 2 / 15 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3662_366259


namespace NUMINAMATH_CALUDE_right_triangle_existence_l3662_366233

/-- A right triangle with hypotenuse c and angle bisector f of the right angle -/
structure RightTriangle where
  c : ℝ  -- Length of hypotenuse
  f : ℝ  -- Length of angle bisector of right angle
  c_pos : c > 0
  f_pos : f > 0

/-- The condition for the existence of a right triangle given its hypotenuse and angle bisector -/
def constructible (t : RightTriangle) : Prop :=
  t.f < t.c / 2

/-- Theorem stating the condition for the existence of a right triangle -/
theorem right_triangle_existence (t : RightTriangle) :
  constructible t ↔ ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a^2 + b^2 = t.c^2 ∧ 
    t.f = (a * b) / (a + b) :=
sorry

end NUMINAMATH_CALUDE_right_triangle_existence_l3662_366233


namespace NUMINAMATH_CALUDE_triangle_properties_l3662_366262

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.a + t.b = 5 ∧
  t.c = Real.sqrt 7 ∧
  4 * (Real.sin ((t.A + t.B) / 2))^2 - Real.cos (2 * t.C) = 7/2 ∧
  t.A + t.B + t.C = Real.pi

-- Theorem statement
theorem triangle_properties (t : Triangle) (h : triangle_conditions t) :
  t.C = Real.pi / 3 ∧ (1/2 * t.a * t.b * Real.sin t.C) = (3 * Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l3662_366262


namespace NUMINAMATH_CALUDE_four_students_three_communities_l3662_366298

/-- The number of ways to distribute students among communities -/
def distribute_students (num_students : ℕ) (num_communities : ℕ) : ℕ :=
  sorry

/-- Theorem stating the correct number of arrangements for 4 students and 3 communities -/
theorem four_students_three_communities :
  distribute_students 4 3 = 36 := by sorry

end NUMINAMATH_CALUDE_four_students_three_communities_l3662_366298


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3662_366276

theorem complex_equation_solution (z : ℂ) (h : z * Complex.I + z = 2) : z = 1 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3662_366276


namespace NUMINAMATH_CALUDE_fifth_row_sum_in_spiral_grid_l3662_366261

/-- Represents a spiral arrangement of numbers in a square grid -/
def SpiralGrid (n : ℕ) := Matrix (Fin n) (Fin n) ℕ

/-- Creates a spiral grid of size n × n with numbers from 1 to n^2 -/
def createSpiralGrid (n : ℕ) : SpiralGrid n :=
  sorry

/-- Returns the numbers in a specific row of the spiral grid -/
def getRowNumbers (grid : SpiralGrid 20) (row : Fin 20) : List ℕ :=
  sorry

/-- Theorem: In a 20x20 spiral grid, the sum of the greatest and least numbers 
    in the fifth row is 565 -/
theorem fifth_row_sum_in_spiral_grid :
  let grid := createSpiralGrid 20
  let fifthRowNumbers := getRowNumbers grid 4
  (List.maximum fifthRowNumbers).getD 0 + (List.minimum fifthRowNumbers).getD 0 = 565 := by
  sorry

end NUMINAMATH_CALUDE_fifth_row_sum_in_spiral_grid_l3662_366261


namespace NUMINAMATH_CALUDE_fraction_equality_l3662_366239

theorem fraction_equality (q r s u : ℚ) 
  (h1 : q / r = 8)
  (h2 : s / r = 4)
  (h3 : s / u = 1 / 3) :
  u / q = 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_fraction_equality_l3662_366239


namespace NUMINAMATH_CALUDE_probability_A_B_same_group_l3662_366232

-- Define the score ranges and their frequencies
def score_ranges : List (ℕ × ℕ × ℕ) := [
  (60, 75, 2),
  (75, 90, 3),
  (90, 105, 14),
  (105, 120, 15),
  (120, 135, 12),
  (135, 150, 4)
]

-- Define the total number of students
def total_students : ℕ := 50

-- Define student A's score
def score_A : ℕ := 62

-- Define student B's score
def score_B : ℕ := 140

-- Define the "two-help-one" group formation rule
def two_help_one (s1 s2 s3 : ℕ) : Prop :=
  (s1 ≥ 135 ∧ s1 ≤ 150) ∧ (s2 ≥ 135 ∧ s2 ≤ 150) ∧ (s3 ≥ 60 ∧ s3 < 75)

-- Theorem to prove
theorem probability_A_B_same_group :
  ∃ (p : ℚ), p = 1/4 ∧ 
  (p = (number_of_groups_with_A_and_B : ℚ) / (total_number_of_possible_groups : ℚ)) :=
sorry

end NUMINAMATH_CALUDE_probability_A_B_same_group_l3662_366232


namespace NUMINAMATH_CALUDE_union_A_B_when_a_4_intersection_A_B_equals_A_iff_l3662_366275

open Set
open Real

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | |x - a| ≤ 1}
def B : Set ℝ := {x | (4 - x) * (x - 1) ≤ 0}

-- Theorem 1: When a = 4, A ∪ B = {x | x ≥ 3 ∨ x ≤ 1}
theorem union_A_B_when_a_4 : 
  A 4 ∪ B = {x : ℝ | x ≥ 3 ∨ x ≤ 1} := by sorry

-- Theorem 2: A ∩ B = A if and only if a ≥ 5 or a ≤ 0
theorem intersection_A_B_equals_A_iff (a : ℝ) : 
  A a ∩ B = A a ↔ a ≥ 5 ∨ a ≤ 0 := by sorry

end NUMINAMATH_CALUDE_union_A_B_when_a_4_intersection_A_B_equals_A_iff_l3662_366275


namespace NUMINAMATH_CALUDE_shortest_distance_point_l3662_366297

/-- Given points A and B, find the point P on the y-axis that minimizes AP + BP -/
theorem shortest_distance_point (A B P : ℝ × ℝ) : 
  A = (3, 2) →
  B = (1, -2) →
  P.1 = 0 →
  P = (0, -1) →
  ∀ Q : ℝ × ℝ, Q.1 = 0 → 
    Real.sqrt ((A.1 - P.1)^2 + (A.2 - P.2)^2) + Real.sqrt ((B.1 - P.1)^2 + (B.2 - P.2)^2) ≤ 
    Real.sqrt ((A.1 - Q.1)^2 + (A.2 - Q.2)^2) + Real.sqrt ((B.1 - Q.1)^2 + (B.2 - Q.2)^2) :=
by sorry


end NUMINAMATH_CALUDE_shortest_distance_point_l3662_366297


namespace NUMINAMATH_CALUDE_rv_parking_probability_l3662_366283

/-- The number of parking spaces -/
def total_spaces : ℕ := 20

/-- The number of cars that have already parked -/
def parked_cars : ℕ := 15

/-- The number of adjacent spaces required for the RV -/
def required_adjacent_spaces : ℕ := 3

/-- The probability of being able to park the RV -/
def parking_probability : ℚ := 232 / 323

theorem rv_parking_probability :
  let empty_spaces := total_spaces - parked_cars
  let total_arrangements := Nat.choose total_spaces parked_cars
  let valid_arrangements := total_arrangements - Nat.choose (empty_spaces + parked_cars - required_adjacent_spaces + 1) empty_spaces
  (valid_arrangements : ℚ) / total_arrangements = parking_probability := by
  sorry

end NUMINAMATH_CALUDE_rv_parking_probability_l3662_366283


namespace NUMINAMATH_CALUDE_phika_inequality_l3662_366226

/-- A sextuple of positive real numbers is phika if the sum of a's equals the sum of b's equals 1 -/
def IsPhika (a₁ a₂ a₃ b₁ b₂ b₃ : ℝ) : Prop :=
  a₁ > 0 ∧ a₂ > 0 ∧ a₃ > 0 ∧ b₁ > 0 ∧ b₂ > 0 ∧ b₃ > 0 ∧
  a₁ + a₂ + a₃ = 1 ∧ b₁ + b₂ + b₃ = 1

theorem phika_inequality :
  (∃ a₁ a₂ a₃ b₁ b₂ b₃ : ℝ, IsPhika a₁ a₂ a₃ b₁ b₂ b₃ ∧
    a₁ * (Real.sqrt b₁ + a₂) + a₂ * (Real.sqrt b₂ + a₃) + a₃ * (Real.sqrt b₃ + a₁) > 1 - 1 / (2022^2022)) ∧
  (∀ a₁ a₂ a₃ b₁ b₂ b₃ : ℝ, IsPhika a₁ a₂ a₃ b₁ b₂ b₃ →
    a₁ * (Real.sqrt b₁ + a₂) + a₂ * (Real.sqrt b₂ + a₃) + a₃ * (Real.sqrt b₃ + a₁) < 1) := by
  sorry

end NUMINAMATH_CALUDE_phika_inequality_l3662_366226


namespace NUMINAMATH_CALUDE_at_least_one_passes_l3662_366237

/-- The probability that at least one of three independent events occurs, given their individual probabilities -/
theorem at_least_one_passes (pA pB pC : ℝ) (hA : pA = 0.8) (hB : pB = 0.6) (hC : pC = 0.5) :
  1 - (1 - pA) * (1 - pB) * (1 - pC) = 0.96 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_passes_l3662_366237


namespace NUMINAMATH_CALUDE_place_mat_length_l3662_366287

-- Define the table and place mat properties
def table_radius : ℝ := 5
def num_mats : ℕ := 8
def mat_width : ℝ := 1.5

-- Define the length of the place mat
def mat_length (y : ℝ) : Prop :=
  y = table_radius * Real.sqrt (2 - Real.sqrt 2)

-- Define the arrangement of the place mats
def mats_arrangement (y : ℝ) : Prop :=
  ∃ (chord_length : ℝ),
    chord_length = 2 * table_radius * Real.sin (Real.pi / (2 * num_mats)) ∧
    y = chord_length

-- Theorem statement
theorem place_mat_length :
  ∃ y : ℝ, mat_length y ∧ mats_arrangement y :=
sorry

end NUMINAMATH_CALUDE_place_mat_length_l3662_366287


namespace NUMINAMATH_CALUDE_opera_ticket_price_increase_l3662_366285

theorem opera_ticket_price_increase (initial_price new_price : ℝ) 
  (h1 : initial_price = 85)
  (h2 : new_price = 102) :
  (new_price - initial_price) / initial_price * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_opera_ticket_price_increase_l3662_366285


namespace NUMINAMATH_CALUDE_share_change_l3662_366243

theorem share_change (total money : ℝ) (ostap_share kisa_share : ℝ) 
  (h1 : ostap_share + kisa_share = total)
  (h2 : ostap_share = 1.5 * kisa_share) :
  let new_ostap_share := 1.5 * ostap_share
  let new_kisa_share := total - new_ostap_share
  new_kisa_share = 0.25 * kisa_share := by
sorry

end NUMINAMATH_CALUDE_share_change_l3662_366243


namespace NUMINAMATH_CALUDE_industrial_park_investment_l3662_366201

theorem industrial_park_investment
  (total_investment : ℝ)
  (return_rate_A : ℝ)
  (return_rate_B : ℝ)
  (total_return : ℝ)
  (h1 : total_investment = 2000)
  (h2 : return_rate_A = 0.054)
  (h3 : return_rate_B = 0.0828)
  (h4 : total_return = 122.4)
  : ∃ (investment_A investment_B : ℝ),
    investment_A + investment_B = total_investment ∧
    investment_A * return_rate_A + investment_B * return_rate_B = total_return ∧
    investment_A = 1500 ∧
    investment_B = 500 := by
  sorry

end NUMINAMATH_CALUDE_industrial_park_investment_l3662_366201


namespace NUMINAMATH_CALUDE_box_third_side_l3662_366208

/-- A rectangular box with known properties -/
structure Box where
  cubes : ℕ  -- Number of cubes that fit in the box
  cube_volume : ℕ  -- Volume of each cube in cubic centimetres
  side1 : ℕ  -- Length of first known side in centimetres
  side2 : ℕ  -- Length of second known side in centimetres

/-- The length of the third side of the box -/
def third_side (b : Box) : ℚ :=
  (b.cubes * b.cube_volume : ℚ) / (b.side1 * b.side2)

/-- Theorem stating that the third side of the given box is 6 centimetres -/
theorem box_third_side :
  let b : Box := { cubes := 24, cube_volume := 27, side1 := 9, side2 := 12 }
  third_side b = 6 := by sorry

end NUMINAMATH_CALUDE_box_third_side_l3662_366208


namespace NUMINAMATH_CALUDE_solve_nested_equation_l3662_366295

theorem solve_nested_equation : 
  ∃ x : ℤ, 45 - (28 - (x - (15 - 16))) = 55 ∧ x = 37 :=
by sorry

end NUMINAMATH_CALUDE_solve_nested_equation_l3662_366295


namespace NUMINAMATH_CALUDE_modular_arithmetic_expression_l3662_366206

theorem modular_arithmetic_expression : (240 * 15 - 33 * 8 + 6) % 18 = 12 := by
  sorry

end NUMINAMATH_CALUDE_modular_arithmetic_expression_l3662_366206


namespace NUMINAMATH_CALUDE_carla_counting_theorem_l3662_366289

/-- The number of times Carla counted tiles on Tuesday -/
def tile_counts : ℕ := 2

/-- The number of times Carla counted books on Tuesday -/
def book_counts : ℕ := 3

/-- The total number of times Carla counted something on Tuesday -/
def total_counts : ℕ := tile_counts + book_counts

theorem carla_counting_theorem : total_counts = 5 := by
  sorry

end NUMINAMATH_CALUDE_carla_counting_theorem_l3662_366289


namespace NUMINAMATH_CALUDE_a_range_l3662_366210

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (2 - a) * x + 1 else a^x

theorem a_range (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → f a x₁ < f a x₂) →
  (3/2 ≤ a ∧ a < 2) :=
sorry

end NUMINAMATH_CALUDE_a_range_l3662_366210


namespace NUMINAMATH_CALUDE_golf_distance_ratio_l3662_366230

/-- Proves that the ratio of the distance traveled on the second turn to the distance traveled on the first turn is 1/2 in a golf scenario. -/
theorem golf_distance_ratio
  (total_distance : ℝ)
  (first_turn_distance : ℝ)
  (overshoot_distance : ℝ)
  (h1 : total_distance = 250)
  (h2 : first_turn_distance = 180)
  (h3 : overshoot_distance = 20)
  : (total_distance - first_turn_distance + overshoot_distance) / first_turn_distance = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_golf_distance_ratio_l3662_366230


namespace NUMINAMATH_CALUDE_student_ticket_price_l3662_366238

theorem student_ticket_price 
  (total_tickets : ℕ) 
  (student_tickets : ℕ) 
  (non_student_tickets : ℕ) 
  (non_student_price : ℚ) 
  (total_revenue : ℚ) :
  total_tickets = 150 →
  student_tickets = 90 →
  non_student_tickets = 60 →
  non_student_price = 8 →
  total_revenue = 930 →
  ∃ (student_price : ℚ), 
    student_price * student_tickets + non_student_price * non_student_tickets = total_revenue ∧
    student_price = 5 := by
  sorry

end NUMINAMATH_CALUDE_student_ticket_price_l3662_366238


namespace NUMINAMATH_CALUDE_max_gcd_of_sum_1111_l3662_366269

theorem max_gcd_of_sum_1111 :
  ∃ (a b : ℕ+), a + b = 1111 ∧ 
  ∀ (c d : ℕ+), c + d = 1111 → Nat.gcd c.val d.val ≤ Nat.gcd a.val b.val ∧
  Nat.gcd a.val b.val = 101 :=
sorry

end NUMINAMATH_CALUDE_max_gcd_of_sum_1111_l3662_366269
