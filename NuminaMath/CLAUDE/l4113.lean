import Mathlib

namespace NUMINAMATH_CALUDE_candy_distribution_theorem_l4113_411379

/-- The number of ways to distribute candy among children with restrictions -/
def distribute_candy (total_candy : ℕ) (num_children : ℕ) (min_candy : ℕ) (max_candy : ℕ) : ℕ :=
  sorry

/-- Theorem stating the specific case of candy distribution -/
theorem candy_distribution_theorem :
  distribute_candy 40 3 2 19 = 171 :=
by sorry

end NUMINAMATH_CALUDE_candy_distribution_theorem_l4113_411379


namespace NUMINAMATH_CALUDE_additional_interest_percentage_l4113_411302

theorem additional_interest_percentage
  (initial_deposit : ℝ)
  (amount_after_3_years : ℝ)
  (target_amount : ℝ)
  (time_period : ℝ)
  (h1 : initial_deposit = 8000)
  (h2 : amount_after_3_years = 11200)
  (h3 : target_amount = 11680)
  (h4 : time_period = 3) :
  let original_interest := amount_after_3_years - initial_deposit
  let target_interest := target_amount - initial_deposit
  let additional_interest := target_interest - original_interest
  let additional_rate := (additional_interest * 100) / (initial_deposit * time_period)
  additional_rate = 2 := by
sorry

end NUMINAMATH_CALUDE_additional_interest_percentage_l4113_411302


namespace NUMINAMATH_CALUDE_no_rain_percentage_l4113_411373

theorem no_rain_percentage (p_monday : ℝ) (p_tuesday : ℝ) (p_both : ℝ) 
  (h_monday : p_monday = 0.7)
  (h_tuesday : p_tuesday = 0.55)
  (h_both : p_both = 0.6) :
  1 - (p_monday + p_tuesday - p_both) = 0.35 := by
  sorry

end NUMINAMATH_CALUDE_no_rain_percentage_l4113_411373


namespace NUMINAMATH_CALUDE_move_point_theorem_l4113_411362

/-- A point in 2D space represented by its x and y coordinates. -/
structure Point where
  x : ℝ
  y : ℝ

/-- Moves a point left by a given distance. -/
def moveLeft (p : Point) (distance : ℝ) : Point :=
  { x := p.x - distance, y := p.y }

/-- Moves a point up by a given distance. -/
def moveUp (p : Point) (distance : ℝ) : Point :=
  { x := p.x, y := p.y + distance }

/-- Theorem stating that moving point P(0, 3) left by 2 units and up by 1 unit results in P₁(-2, 4). -/
theorem move_point_theorem : 
  let P : Point := { x := 0, y := 3 }
  let P₁ : Point := moveUp (moveLeft P 2) 1
  P₁.x = -2 ∧ P₁.y = 4 := by
  sorry

end NUMINAMATH_CALUDE_move_point_theorem_l4113_411362


namespace NUMINAMATH_CALUDE_haleys_trees_l4113_411346

theorem haleys_trees (initial_trees : ℕ) : 
  (initial_trees - 4 + 5 = 10) → initial_trees = 9 := by
  sorry

end NUMINAMATH_CALUDE_haleys_trees_l4113_411346


namespace NUMINAMATH_CALUDE_fence_cost_l4113_411343

theorem fence_cost (area : ℝ) (price_per_foot : ℝ) (h1 : area = 289) (h2 : price_per_foot = 58) :
  let side_length := Real.sqrt area
  let perimeter := 4 * side_length
  let total_cost := perimeter * price_per_foot
  total_cost = 3944 :=
by sorry

end NUMINAMATH_CALUDE_fence_cost_l4113_411343


namespace NUMINAMATH_CALUDE_farmer_profit_is_960_l4113_411301

/-- Represents the farmer's pig business -/
structure PigBusiness where
  num_piglets : ℕ
  sale_price : ℕ
  min_growth_months : ℕ
  feed_cost_per_month : ℕ
  pigs_sold_12_months : ℕ
  pigs_sold_16_months : ℕ

/-- Calculates the total profit for the pig business -/
def calculate_profit (business : PigBusiness) : ℕ :=
  let revenue := business.sale_price * (business.pigs_sold_12_months + business.pigs_sold_16_months)
  let feed_cost_12_months := business.feed_cost_per_month * business.min_growth_months * business.pigs_sold_12_months
  let feed_cost_16_months := business.feed_cost_per_month * 16 * business.pigs_sold_16_months
  let total_feed_cost := feed_cost_12_months + feed_cost_16_months
  revenue - total_feed_cost

/-- The farmer's profit is $960 -/
theorem farmer_profit_is_960 (business : PigBusiness) 
    (h1 : business.num_piglets = 6)
    (h2 : business.sale_price = 300)
    (h3 : business.min_growth_months = 12)
    (h4 : business.feed_cost_per_month = 10)
    (h5 : business.pigs_sold_12_months = 3)
    (h6 : business.pigs_sold_16_months = 3) :
    calculate_profit business = 960 := by
  sorry

end NUMINAMATH_CALUDE_farmer_profit_is_960_l4113_411301


namespace NUMINAMATH_CALUDE_min_value_theorem_l4113_411345

theorem min_value_theorem (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x * y * z = 64) :
  ∃ (m : ℝ), m = 24 ∧ ∀ (a b c : ℝ), a > 0 → b > 0 → c > 0 → a * b * c = 64 → x + 4*y + 8*z ≤ a + 4*b + 8*c ∧
  (x + 4*y + 8*z = m ∨ a + 4*b + 8*c > m) := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l4113_411345


namespace NUMINAMATH_CALUDE_magic_square_sum_l4113_411342

theorem magic_square_sum (a b c d e f g : ℕ) : 
  (a + 13 + 12 + 1 = 34) →
  (g + 13 + 2 + 16 = 34) →
  (f + 16 + 9 + 4 = 34) →
  (c + 1 + 15 + 4 = 34) →
  (b + 12 + 7 + 9 = 34) →
  (d + 15 + 6 + 3 = 34) →
  (e + 2 + 7 + 14 = 34) →
  a - b - c + d + e + f - g = 11 := by
  sorry

end NUMINAMATH_CALUDE_magic_square_sum_l4113_411342


namespace NUMINAMATH_CALUDE_tea_mixture_price_l4113_411374

/-- Given two types of tea mixed in equal proportions, this theorem proves
    the price of the second tea given the price of the first tea and the mixture. -/
theorem tea_mixture_price
  (price_tea1 : ℝ)
  (price_mixture : ℝ)
  (h1 : price_tea1 = 64)
  (h2 : price_mixture = 69) :
  ∃ (price_tea2 : ℝ),
    price_tea2 = 74 ∧
    (price_tea1 + price_tea2) / 2 = price_mixture :=
by
  sorry

end NUMINAMATH_CALUDE_tea_mixture_price_l4113_411374


namespace NUMINAMATH_CALUDE_digit_subtraction_reaches_zero_l4113_411337

/-- Sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

/-- The sequence obtained by repeatedly subtracting the sum of digits -/
def digitSubtractionSequence (n : ℕ) : ℕ → ℕ
  | 0 => n
  | k + 1 => digitSubtractionSequence n k - sumOfDigits (digitSubtractionSequence n k)

/-- The theorem stating that the digit subtraction sequence always reaches 0 -/
theorem digit_subtraction_reaches_zero (n : ℕ) :
  ∃ k : ℕ, digitSubtractionSequence n k = 0 :=
sorry

end NUMINAMATH_CALUDE_digit_subtraction_reaches_zero_l4113_411337


namespace NUMINAMATH_CALUDE_initial_concentration_proof_l4113_411349

/-- Proves that the initial concentration of an acidic liquid is 40% given the problem conditions --/
theorem initial_concentration_proof (initial_volume : ℝ) (water_removed : ℝ) (final_concentration : ℝ) :
  initial_volume = 12 →
  water_removed = 4 →
  final_concentration = 60 →
  (initial_volume - water_removed) * final_concentration / 100 = initial_volume * 40 / 100 := by
  sorry

#check initial_concentration_proof

end NUMINAMATH_CALUDE_initial_concentration_proof_l4113_411349


namespace NUMINAMATH_CALUDE_f_deriv_at_one_l4113_411358

-- Define a differentiable function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the condition f(x) = 2xf'(1) + ln x
axiom f_condition (x : ℝ) : f x = 2 * x * (deriv f 1) + Real.log x

-- Theorem statement
theorem f_deriv_at_one : deriv f 1 = -1 := by sorry

end NUMINAMATH_CALUDE_f_deriv_at_one_l4113_411358


namespace NUMINAMATH_CALUDE_inequality_proof_l4113_411384

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : 1/a + 1/b = 1) :
  ∀ n : ℕ, (a + b)^n - a^n - b^n ≥ 2^(2*n) - 2^(n+1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l4113_411384


namespace NUMINAMATH_CALUDE_roots_equation_relation_l4113_411313

theorem roots_equation_relation (p q a b c : ℝ) : 
  (a^2 + p*a + 1 = 0) → 
  (b^2 + p*b + 1 = 0) → 
  (b^2 + q*b + 2 = 0) → 
  (c^2 + q*c + 2 = 0) → 
  (b-a)*(b-c) = p*q - 6 := by
  sorry

end NUMINAMATH_CALUDE_roots_equation_relation_l4113_411313


namespace NUMINAMATH_CALUDE_fraction_sum_simplification_l4113_411388

theorem fraction_sum_simplification :
  8 / 24 - 5 / 72 + 3 / 8 = 23 / 36 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_simplification_l4113_411388


namespace NUMINAMATH_CALUDE_max_digit_sum_l4113_411305

theorem max_digit_sum (a b c x y : ℕ) : 
  (a < 10 ∧ b < 10 ∧ c < 10) →  -- a, b, c are digits
  (1000 * (1 : ℚ) / (100 * a + 10 * b + c) = y) →  -- 0.abc = 1/y
  (0 < y ∧ y ≤ 50) →  -- y is an integer and 0 < y ≤ 50
  (1000 * (1 : ℚ) / (100 * y + 10 * y + y) = x) →  -- 0.yyy = 1/x
  (0 < x ∧ x ≤ 9) →  -- x is an integer and 0 < x ≤ 9
  (∀ a' b' c' : ℕ, 
    (a' < 10 ∧ b' < 10 ∧ c' < 10) →
    (∃ x' y' : ℕ, 
      (1000 * (1 : ℚ) / (100 * a' + 10 * b' + c') = y') ∧
      (0 < y' ∧ y' ≤ 50) ∧
      (1000 * (1 : ℚ) / (100 * y' + 10 * y' + y') = x') ∧
      (0 < x' ∧ x' ≤ 9)) →
    (a + b + c ≥ a' + b' + c')) →
  a + b + c = 8 := by
sorry

end NUMINAMATH_CALUDE_max_digit_sum_l4113_411305


namespace NUMINAMATH_CALUDE_acute_angle_trig_inequality_l4113_411381

theorem acute_angle_trig_inequality (α : Real) (h : 0 < α ∧ α < Real.pi / 2) :
  1/2 < Real.sqrt 3 / 2 * Real.sin α + 1/2 * Real.cos α ∧
  Real.sqrt 3 / 2 * Real.sin α + 1/2 * Real.cos α ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_acute_angle_trig_inequality_l4113_411381


namespace NUMINAMATH_CALUDE_whole_number_between_36_and_40_l4113_411375

theorem whole_number_between_36_and_40 (M : ℤ) :
  (9 < M / 4 ∧ M / 4 < 10) → (M = 37 ∨ M = 38 ∨ M = 39) := by
  sorry

end NUMINAMATH_CALUDE_whole_number_between_36_and_40_l4113_411375


namespace NUMINAMATH_CALUDE_carpenter_problem_solution_l4113_411322

/-- Represents the carpenter problem -/
def CarpenterProblem (x : ℝ) : Prop :=
  let first_carpenter_rate := 1 / (x + 4)
  let second_carpenter_rate := 1 / 5
  let combined_rate := first_carpenter_rate + second_carpenter_rate
  2 * combined_rate = 4 * first_carpenter_rate

/-- The solution to the carpenter problem is 1 day -/
theorem carpenter_problem_solution :
  ∃ (x : ℝ), CarpenterProblem x ∧ x = 1 :=
sorry

end NUMINAMATH_CALUDE_carpenter_problem_solution_l4113_411322


namespace NUMINAMATH_CALUDE_product_of_roots_l4113_411399

theorem product_of_roots (x : ℝ) : (x + 3) * (x - 4) = 17 → ∃ y : ℝ, (x + 3) * (x - 4) = 17 ∧ (x * y = -29) := by
  sorry

end NUMINAMATH_CALUDE_product_of_roots_l4113_411399


namespace NUMINAMATH_CALUDE_original_number_is_107_l4113_411314

def is_three_digit_number (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

def increase_digits (n : ℕ) : ℕ :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let units := n % 10
  (hundreds + 3) * 100 + (tens + 2) * 10 + (units + 1)

theorem original_number_is_107 :
  is_three_digit_number 107 ∧ increase_digits 107 = 4 * 107 :=
sorry

end NUMINAMATH_CALUDE_original_number_is_107_l4113_411314


namespace NUMINAMATH_CALUDE_henry_trays_capacity_l4113_411364

/-- The number of trays Henry picked up from the first table -/
def trays_table1 : ℕ := 29

/-- The number of trays Henry picked up from the second table -/
def trays_table2 : ℕ := 52

/-- The total number of trips Henry made -/
def total_trips : ℕ := 9

/-- The number of trays Henry could carry at a time -/
def trays_per_trip : ℕ := (trays_table1 + trays_table2) / total_trips

theorem henry_trays_capacity : trays_per_trip = 9 := by
  sorry

end NUMINAMATH_CALUDE_henry_trays_capacity_l4113_411364


namespace NUMINAMATH_CALUDE_floor_equation_solution_l4113_411378

theorem floor_equation_solution (a b : ℕ+) :
  (Int.floor (a^2 / b : ℚ) + Int.floor (b^2 / a : ℚ) = 
   Int.floor ((a^2 + b^2) / (a * b) : ℚ) + a * b) ↔ 
  (a = b^2 + 1) :=
by sorry

end NUMINAMATH_CALUDE_floor_equation_solution_l4113_411378


namespace NUMINAMATH_CALUDE_binomial_prob_example_l4113_411311

/-- The probability mass function of a binomial distribution -/
def binomial_pmf (n : ℕ) (p : ℝ) (k : ℕ) : ℝ :=
  (Nat.choose n k : ℝ) * p^k * (1 - p)^(n - k)

/-- Theorem: For X ~ B(4, 1/3), P(X = 1) = 32/81 -/
theorem binomial_prob_example :
  let n : ℕ := 4
  let p : ℝ := 1/3
  let k : ℕ := 1
  binomial_pmf n p k = 32/81 := by
sorry

end NUMINAMATH_CALUDE_binomial_prob_example_l4113_411311


namespace NUMINAMATH_CALUDE_max_blue_points_l4113_411331

/-- The maximum number of blue points when 2016 spheres are colored red or green -/
theorem max_blue_points (total_spheres : Nat) (h : total_spheres = 2016) :
  ∃ (red_spheres : Nat),
    red_spheres ≤ total_spheres ∧
    red_spheres * (total_spheres - red_spheres) = 1016064 ∧
    ∀ (x : Nat), x ≤ total_spheres →
      x * (total_spheres - x) ≤ 1016064 := by
  sorry

end NUMINAMATH_CALUDE_max_blue_points_l4113_411331


namespace NUMINAMATH_CALUDE_roots_have_different_signs_l4113_411365

/-- A quadratic polynomial f(x) = ax^2 + bx + c -/
def quadraticPolynomial (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem roots_have_different_signs (a b c : ℝ) (ha : a ≠ 0) :
  (quadraticPolynomial a b c (1/a)) * (quadraticPolynomial a b c c) < 0 →
  ∃ x₁ x₂ : ℝ, x₁ * x₂ < 0 ∧ ∀ x, quadraticPolynomial a b c x = 0 ↔ x = x₁ ∨ x = x₂ :=
by sorry

end NUMINAMATH_CALUDE_roots_have_different_signs_l4113_411365


namespace NUMINAMATH_CALUDE_cube_sum_equals_diff_implies_square_sum_less_than_one_l4113_411355

theorem cube_sum_equals_diff_implies_square_sum_less_than_one 
  (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x^3 + y^3 = x - y) : 
  x^2 + y^2 < 1 := by
sorry

end NUMINAMATH_CALUDE_cube_sum_equals_diff_implies_square_sum_less_than_one_l4113_411355


namespace NUMINAMATH_CALUDE_total_legs_calculation_l4113_411317

/-- The number of legs a spider has -/
def spider_legs : ℕ := 8

/-- The number of legs a centipede has -/
def centipede_legs : ℕ := 100

/-- The number of spiders in the room -/
def num_spiders : ℕ := 4

/-- The number of centipedes in the room -/
def num_centipedes : ℕ := 3

/-- The total number of legs for all spiders and centipedes -/
def total_legs : ℕ := num_spiders * spider_legs + num_centipedes * centipede_legs

theorem total_legs_calculation :
  total_legs = 332 := by sorry

end NUMINAMATH_CALUDE_total_legs_calculation_l4113_411317


namespace NUMINAMATH_CALUDE_line_through_point_l4113_411347

theorem line_through_point (k : ℚ) : 
  (2 - 3 * k * (-3) = 5 * 1) → k = 1/3 := by sorry

end NUMINAMATH_CALUDE_line_through_point_l4113_411347


namespace NUMINAMATH_CALUDE_equation_solution_l4113_411306

theorem equation_solution : ∃! x : ℝ, (x + 1) / 2 - 1 = (2 - 3 * x) / 3 ∧ x = 7 / 9 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l4113_411306


namespace NUMINAMATH_CALUDE_sphere_volume_circumscribing_cube_l4113_411387

/-- The volume of a sphere circumscribing a cube with edge length 2 cm -/
theorem sphere_volume_circumscribing_cube : 
  let cube_edge : ℝ := 2
  let sphere_radius : ℝ := cube_edge * Real.sqrt 3 / 2
  let sphere_volume : ℝ := (4 / 3) * Real.pi * sphere_radius ^ 3
  sphere_volume = 4 * Real.sqrt 3 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_circumscribing_cube_l4113_411387


namespace NUMINAMATH_CALUDE_steak_weight_for_tommy_family_l4113_411341

/-- Given a family where each member wants one pound of steak, 
    this function calculates the weight of each steak needed to be purchased. -/
def steak_weight (family_size : ℕ) (num_steaks : ℕ) : ℚ :=
  (family_size : ℚ) / (num_steaks : ℚ)

/-- Proves that for a family of 5 members, each wanting one pound of steak,
    and needing to buy 4 steaks, the weight of each steak is 1.25 pounds. -/
theorem steak_weight_for_tommy_family : 
  steak_weight 5 4 = 5/4 := by sorry

end NUMINAMATH_CALUDE_steak_weight_for_tommy_family_l4113_411341


namespace NUMINAMATH_CALUDE_cost_difference_l4113_411395

-- Define the parameters
def batches : ℕ := 4
def ounces_per_batch : ℕ := 12
def blueberry_carton_size : ℕ := 6
def raspberry_carton_size : ℕ := 8
def blueberry_price : ℚ := 5
def raspberry_price : ℚ := 3

-- Define the total ounces needed
def total_ounces : ℕ := batches * ounces_per_batch

-- Define the number of cartons needed for each fruit
def blueberry_cartons : ℕ := (total_ounces + blueberry_carton_size - 1) / blueberry_carton_size
def raspberry_cartons : ℕ := (total_ounces + raspberry_carton_size - 1) / raspberry_carton_size

-- Define the total cost for each fruit
def blueberry_cost : ℚ := blueberry_price * blueberry_cartons
def raspberry_cost : ℚ := raspberry_price * raspberry_cartons

-- Theorem to prove
theorem cost_difference : blueberry_cost - raspberry_cost = 22 := by
  sorry

end NUMINAMATH_CALUDE_cost_difference_l4113_411395


namespace NUMINAMATH_CALUDE_other_x_intercept_of_quadratic_l4113_411353

/-- Given a quadratic function with vertex (6, -2) and one x-intercept at (1, 0),
    the x-coordinate of the other x-intercept is 11. -/
theorem other_x_intercept_of_quadratic (a b c : ℝ) :
  (∀ x, a * x^2 + b * x + c = -2 ↔ x = 6) →  -- vertex condition
  a * 1^2 + b * 1 + c = 0 →                  -- x-intercept condition
  ∃ x, x ≠ 1 ∧ a * x^2 + b * x + c = 0 ∧ x = 11 := by
  sorry

end NUMINAMATH_CALUDE_other_x_intercept_of_quadratic_l4113_411353


namespace NUMINAMATH_CALUDE_pen_price_theorem_l4113_411323

/-- Given the conditions of the pen and pencil purchase, prove the average price of a pen. -/
theorem pen_price_theorem (total_pens : ℕ) (total_pencils : ℕ) (total_cost : ℚ) (avg_pencil_price : ℚ) :
  total_pens = 30 →
  total_pencils = 75 →
  total_cost = 510 →
  avg_pencil_price = 2 →
  (total_cost - total_pencils * avg_pencil_price) / total_pens = 12 :=
by sorry

end NUMINAMATH_CALUDE_pen_price_theorem_l4113_411323


namespace NUMINAMATH_CALUDE_opposite_of_2023_l4113_411376

theorem opposite_of_2023 : 
  ∀ (x : ℤ), (x + 2023 = 0) → x = -2023 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_2023_l4113_411376


namespace NUMINAMATH_CALUDE_prime_iff_k_t_greater_n_div_4_l4113_411304

theorem prime_iff_k_t_greater_n_div_4 (n : ℕ) (k t : ℕ) : 
  Odd n → n > 3 →
  (∀ k' < k, ¬ ∃ m : ℕ, k' * n + 1 = m * m) →
  (∀ t' < t, ¬ ∃ m : ℕ, t' * n = m * m) →
  (∃ m : ℕ, k * n + 1 = m * m) →
  (∃ m : ℕ, t * n = m * m) →
  (Nat.Prime n ↔ (k > n / 4 ∧ t > n / 4)) :=
by sorry

end NUMINAMATH_CALUDE_prime_iff_k_t_greater_n_div_4_l4113_411304


namespace NUMINAMATH_CALUDE_certain_number_exists_l4113_411338

theorem certain_number_exists : ∃ x : ℝ, 
  5.4 * x - (0.6 * 10) / 1.2 = 31.000000000000004 ∧ 
  abs (x - 6.666666666666667) < 1e-15 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_exists_l4113_411338


namespace NUMINAMATH_CALUDE_radiator_water_fraction_l4113_411398

/-- Calculates the fraction of water remaining in a radiator after a given number of replacements -/
def waterFraction (initialVolume : ℚ) (replacementVolume : ℚ) (numReplacements : ℕ) : ℚ :=
  (1 - replacementVolume / initialVolume) ^ numReplacements

theorem radiator_water_fraction :
  let initialVolume : ℚ := 25
  let replacementVolume : ℚ := 5
  let numReplacements : ℕ := 5
  waterFraction initialVolume replacementVolume numReplacements = 1024 / 3125 := by
  sorry

#eval waterFraction 25 5 5

end NUMINAMATH_CALUDE_radiator_water_fraction_l4113_411398


namespace NUMINAMATH_CALUDE_caras_cat_catch_proof_l4113_411380

/-- The number of animals Cara's cat catches given Martha's cat's catch -/
def caras_cat_catch (marthas_rats : ℕ) (marthas_birds : ℕ) : ℕ :=
  5 * (marthas_rats + marthas_birds) - 3

theorem caras_cat_catch_proof :
  caras_cat_catch 3 7 = 47 := by
  sorry

end NUMINAMATH_CALUDE_caras_cat_catch_proof_l4113_411380


namespace NUMINAMATH_CALUDE_paulines_potato_count_l4113_411316

/-- Represents Pauline's vegetable garden --/
structure Garden where
  rows : Nat
  spacesPerRow : Nat
  tomatoKinds : Nat
  tomatoesPerKind : Nat
  cucumberKinds : Nat
  cucumbersPerKind : Nat
  availableSpaces : Nat

/-- Calculates the number of potatoes in the garden --/
def potatoCount (g : Garden) : Nat :=
  g.rows * g.spacesPerRow - 
  (g.tomatoKinds * g.tomatoesPerKind + g.cucumberKinds * g.cucumbersPerKind) - 
  g.availableSpaces

/-- Theorem stating the number of potatoes in Pauline's garden --/
theorem paulines_potato_count :
  let g : Garden := {
    rows := 10,
    spacesPerRow := 15,
    tomatoKinds := 3,
    tomatoesPerKind := 5,
    cucumberKinds := 5,
    cucumbersPerKind := 4,
    availableSpaces := 85
  }
  potatoCount g = 30 := by
  sorry

end NUMINAMATH_CALUDE_paulines_potato_count_l4113_411316


namespace NUMINAMATH_CALUDE_geometric_sequence_value_l4113_411386

theorem geometric_sequence_value (b : ℝ) (h1 : b > 0) : 
  (∃ r : ℝ, r > 0 ∧ b = 30 * r ∧ 15/4 = b * r) → b = 15 * Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_value_l4113_411386


namespace NUMINAMATH_CALUDE_trigonometric_expression_equality_trigonometric_identities_l4113_411315

-- Part 1
theorem trigonometric_expression_equality :
  (Real.sqrt 3 * Real.sin (-20/3 * π)) / Real.tan (11/3 * π) - 
  Real.cos (13/4 * π) * Real.tan (-37/4 * π) = 
  (Real.sqrt 3 - Real.sqrt 2) / 2 := by sorry

-- Part 2
theorem trigonometric_identities (a : Real) (h : Real.tan a = 4/3) :
  (Real.sin a ^ 2 + 2 * Real.sin a * Real.cos a) / (2 * Real.cos a ^ 2 - Real.sin a ^ 2) = 20 ∧
  Real.sin a * Real.cos a = 12/25 := by sorry

end NUMINAMATH_CALUDE_trigonometric_expression_equality_trigonometric_identities_l4113_411315


namespace NUMINAMATH_CALUDE_point_P_in_quadrant_III_l4113_411363

def point_in_quadrant_III (x y : ℝ) : Prop :=
  x < 0 ∧ y < 0

theorem point_P_in_quadrant_III :
  point_in_quadrant_III (-1 : ℝ) (-2 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_point_P_in_quadrant_III_l4113_411363


namespace NUMINAMATH_CALUDE_museum_entrance_cost_l4113_411354

/-- The total cost of entrance tickets for a group of students and teachers -/
def total_cost (num_students : ℕ) (num_teachers : ℕ) (ticket_price : ℕ) : ℕ :=
  (num_students + num_teachers) * ticket_price

/-- Theorem: The total cost for 20 students and 3 teachers with $5 tickets is $115 -/
theorem museum_entrance_cost : total_cost 20 3 5 = 115 := by
  sorry

end NUMINAMATH_CALUDE_museum_entrance_cost_l4113_411354


namespace NUMINAMATH_CALUDE_basketball_court_length_l4113_411320

theorem basketball_court_length :
  ∀ (width length : ℝ),
  length = width + 14 →
  2 * length + 2 * width = 96 →
  length = 31 :=
by
  sorry

end NUMINAMATH_CALUDE_basketball_court_length_l4113_411320


namespace NUMINAMATH_CALUDE_no_five_digit_flippy_divisible_by_11_l4113_411327

/-- A flippy number is a number whose digits alternate between two distinct digits. -/
def is_flippy (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a ≠ b ∧ 
  (∃ (d1 d2 d3 d4 d5 : ℕ), 
    n = d1 * 10000 + d2 * 1000 + d3 * 100 + d4 * 10 + d5 ∧
    ((d1 = a ∧ d2 = b ∧ d3 = a ∧ d4 = b ∧ d5 = a) ∨
     (d1 = b ∧ d2 = a ∧ d3 = b ∧ d4 = a ∧ d5 = b)))

/-- A number is five digits long if it's between 10000 and 99999, inclusive. -/
def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n ≤ 99999

theorem no_five_digit_flippy_divisible_by_11 : 
  ¬∃ (n : ℕ), is_flippy n ∧ is_five_digit n ∧ n % 11 = 0 :=
sorry

end NUMINAMATH_CALUDE_no_five_digit_flippy_divisible_by_11_l4113_411327


namespace NUMINAMATH_CALUDE_football_team_girls_l4113_411356

theorem football_team_girls (total : ℕ) (attended : ℕ) (girls : ℕ) (boys : ℕ) : 
  total = 30 →
  attended = 18 →
  girls + boys = total →
  attended = boys + (girls / 3) →
  girls = 18 :=
by sorry

end NUMINAMATH_CALUDE_football_team_girls_l4113_411356


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_four_l4113_411383

theorem reciprocal_of_negative_four :
  (1 : ℚ) / (-4 : ℚ) = -1/4 := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_four_l4113_411383


namespace NUMINAMATH_CALUDE_rationalize_denominator_l4113_411382

theorem rationalize_denominator :
  ∃ (A B C : ℤ), (2 + Real.sqrt 5) / (2 - Real.sqrt 5) = A + B * Real.sqrt C :=
by sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l4113_411382


namespace NUMINAMATH_CALUDE_function_and_intersection_points_l4113_411357

noncomputable def f (b c d : ℝ) (x : ℝ) : ℝ := x^3 + b*x^2 + c*x + d

theorem function_and_intersection_points 
  (b c d : ℝ) 
  (h1 : f b c d 0 = 2) 
  (h2 : (6 : ℝ) * (-1) - f b c d (-1) + 7 = 0) 
  (h3 : (6 : ℝ) = (3 * (-1)^2 + 2*b*(-1) + c)) :
  (∀ x, f b c d x = x^3 - 3*x^2 - 3*x + 2) ∧
  (∀ a, (∃ x₁ x₂ x₃, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
    f b c d x₁ = (3/2)*x₁^2 - 9*x₁ + a + 2 ∧
    f b c d x₂ = (3/2)*x₂^2 - 9*x₂ + a + 2 ∧
    f b c d x₃ = (3/2)*x₃^2 - 9*x₃ + a + 2) →
  2 < a ∧ a < 5/2) :=
by sorry

end NUMINAMATH_CALUDE_function_and_intersection_points_l4113_411357


namespace NUMINAMATH_CALUDE_no_prime_for_expression_l4113_411307

theorem no_prime_for_expression (p : ℕ) (hp : Nat.Prime p) : ¬ Nat.Prime (22 * p^2 + 23) := by
  sorry

end NUMINAMATH_CALUDE_no_prime_for_expression_l4113_411307


namespace NUMINAMATH_CALUDE_average_height_students_count_l4113_411351

/-- Represents the number of students in different height categories --/
structure HeightDistribution where
  total : ℕ
  short : ℕ
  tall : ℕ
  extremelyTall : ℕ

/-- Calculates the number of students with average height --/
def averageHeightStudents (h : HeightDistribution) : ℕ :=
  h.total - (h.short + h.tall + h.extremelyTall)

/-- Theorem: The number of students with average height in the given class is 110 --/
theorem average_height_students_count (h : HeightDistribution) 
  (h_total : h.total = 400)
  (h_short : h.short = 2 * h.total / 5)
  (h_extremelyTall : h.extremelyTall = h.total / 10)
  (h_tall : h.tall = 90) :
  averageHeightStudents h = 110 := by
  sorry

#eval averageHeightStudents ⟨400, 160, 90, 40⟩

end NUMINAMATH_CALUDE_average_height_students_count_l4113_411351


namespace NUMINAMATH_CALUDE_xiaopang_problem_l4113_411350

theorem xiaopang_problem (a : ℕ) (d : ℕ) (n : ℕ) : 
  a = 1 → d = 2 → n = 8 → (n / 2) * (2 * a + (n - 1) * d) = 64 := by
  sorry

end NUMINAMATH_CALUDE_xiaopang_problem_l4113_411350


namespace NUMINAMATH_CALUDE_daleyza_project_units_l4113_411329

/-- Calculates the total number of units in a three-building construction project -/
def total_units (first_building : ℕ) : ℕ :=
  let second_building := (2 : ℕ) * first_building / 5
  let third_building := (6 : ℕ) * second_building / 5
  first_building + second_building + third_building

/-- Theorem stating that given the specific conditions of Daleyza's project, 
    the total number of units is 7520 -/
theorem daleyza_project_units : total_units 4000 = 7520 := by
  sorry

end NUMINAMATH_CALUDE_daleyza_project_units_l4113_411329


namespace NUMINAMATH_CALUDE_emily_productivity_l4113_411385

/-- Emily's work productivity over two days -/
theorem emily_productivity (p h : ℕ) : 
  p = 3 * h →                           -- Condition: p = 3h
  (p - 3) * (h + 3) - p * h = 6 * h - 9 -- Prove: difference in pages is 6h - 9
  := by sorry

end NUMINAMATH_CALUDE_emily_productivity_l4113_411385


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l4113_411309

theorem imaginary_part_of_complex_fraction : 
  Complex.im (2 * Complex.I / (1 + Complex.I)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l4113_411309


namespace NUMINAMATH_CALUDE_soccer_team_subjects_l4113_411324

theorem soccer_team_subjects (total : ℕ) (history : ℕ) (both : ℕ) (geography : ℕ) : 
  total = 18 → 
  history = 10 → 
  both = 6 → 
  geography = total - (history - both) → 
  geography = 14 := by
sorry

end NUMINAMATH_CALUDE_soccer_team_subjects_l4113_411324


namespace NUMINAMATH_CALUDE_appointment_ways_l4113_411310

def dedicated_fitters : ℕ := 5
def dedicated_turners : ℕ := 4
def versatile_workers : ℕ := 2
def total_workers : ℕ := dedicated_fitters + dedicated_turners + versatile_workers
def required_fitters : ℕ := 4
def required_turners : ℕ := 4

theorem appointment_ways : 
  (Nat.choose dedicated_fitters required_fitters * Nat.choose dedicated_turners (required_turners - 1) * Nat.choose versatile_workers 1) +
  (Nat.choose dedicated_fitters (required_fitters - 1) * Nat.choose dedicated_turners required_turners * Nat.choose versatile_workers 1) +
  (Nat.choose dedicated_fitters required_fitters * Nat.choose dedicated_turners (required_turners - 2) * Nat.choose versatile_workers 2) +
  (Nat.choose dedicated_fitters (required_fitters - 1) * Nat.choose dedicated_turners (required_turners - 1) * Nat.choose versatile_workers 2) +
  (Nat.choose dedicated_fitters (required_fitters - 1) * Nat.choose dedicated_turners (required_turners - 2) * Nat.choose versatile_workers 2) = 190 := by
  sorry

end NUMINAMATH_CALUDE_appointment_ways_l4113_411310


namespace NUMINAMATH_CALUDE_value_of_expression_l4113_411335

theorem value_of_expression (x : ℝ) (h : x = -2) : (3 * x + 4)^2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_value_of_expression_l4113_411335


namespace NUMINAMATH_CALUDE_dart_partitions_l4113_411344

def partition_count (n : ℕ) (k : ℕ) : ℕ :=
  sorry

theorem dart_partitions :
  partition_count 5 3 = 5 := by
  sorry

end NUMINAMATH_CALUDE_dart_partitions_l4113_411344


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l4113_411339

theorem triangle_abc_properties (a b c : ℝ) (A B C : ℝ) :
  -- Triangle ABC with sides a, b, c opposite to angles A, B, C
  (0 < a ∧ 0 < b ∧ 0 < c) →
  (0 < A ∧ 0 < B ∧ 0 < C) →
  (A + B + C = π) →
  -- Given condition
  (2 * b * Real.cos C = 2 * a + c) →
  -- Additional condition for part 2
  (2 * Real.sqrt 3 * Real.sin (A / 2 + π / 6) * Real.cos (A / 2 + π / 6) - 
   2 * Real.sin (A / 2 + π / 6) ^ 2 = 11 / 13) →
  -- Conclusions to prove
  (B = 2 * π / 3 ∧ 
   Real.cos C = (12 + 5 * Real.sqrt 3) / 26) := by
sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l4113_411339


namespace NUMINAMATH_CALUDE_sons_age_is_fourteen_l4113_411308

/-- Proves that given the conditions, the son's present age is 14 years -/
theorem sons_age_is_fourteen (son_age father_age : ℕ) : 
  father_age = son_age + 16 →
  father_age + 2 = 2 * (son_age + 2) →
  son_age = 14 := by
  sorry

end NUMINAMATH_CALUDE_sons_age_is_fourteen_l4113_411308


namespace NUMINAMATH_CALUDE_oil_distance_theorem_l4113_411321

/-- Represents the relationship between remaining oil and distance traveled --/
def oil_distance_relation (x : ℝ) : ℝ := 62 - 0.12 * x

theorem oil_distance_theorem :
  let initial_oil : ℝ := 62
  let data_points : List (ℝ × ℝ) := [(100, 50), (200, 38), (300, 26), (400, 14)]
  ∀ (x y : ℝ), (x, y) ∈ data_points → y = oil_distance_relation x :=
by sorry

end NUMINAMATH_CALUDE_oil_distance_theorem_l4113_411321


namespace NUMINAMATH_CALUDE_inverse_g_sum_l4113_411303

-- Define the function g
def g (x : ℝ) : ℝ := x * |x|

-- State the theorem
theorem inverse_g_sum : ∃ (a b : ℝ), g a = 9 ∧ g b = -64 ∧ a + b = -5 := by
  sorry

end NUMINAMATH_CALUDE_inverse_g_sum_l4113_411303


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l4113_411326

/-- The equation of a line perpendicular to x - y = 0 and passing through (1, 0) -/
theorem perpendicular_line_equation :
  let l₁ : Set (ℝ × ℝ) := {p | p.1 - p.2 = 0}  -- The line x - y = 0
  let p : ℝ × ℝ := (1, 0)  -- The point (1, 0)
  let l₂ : Set (ℝ × ℝ) := {q | q.1 + q.2 - 1 = 0}  -- The line we want to prove
  (∀ x y, (x, y) ∈ l₂ ↔ x + y - 1 = 0) ∧  -- l₂ is indeed x + y - 1 = 0
  p ∈ l₂ ∧  -- l₂ passes through (1, 0)
  (∀ x₁ y₁ x₂ y₂, (x₁, y₁) ∈ l₁ ∧ (x₂, y₂) ∈ l₁ ∧ x₁ ≠ x₂ →
    (x₁ - x₂) * ((1 - x) / (0 - y)) = -1)  -- l₂ is perpendicular to l₁
  := by sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l4113_411326


namespace NUMINAMATH_CALUDE_least_n_mod_1000_l4113_411369

/-- Sum of digits in base 4 representation -/
def f (n : ℕ) : ℕ :=
  sorry

/-- Sum of digits in base 8 representation of f(n) -/
def g (n : ℕ) : ℕ :=
  sorry

/-- The least value of n such that g(n) ≥ 10 -/
def N : ℕ :=
  sorry

theorem least_n_mod_1000 : N % 1000 = 151 := by
  sorry

end NUMINAMATH_CALUDE_least_n_mod_1000_l4113_411369


namespace NUMINAMATH_CALUDE_fish_cost_theorem_l4113_411348

theorem fish_cost_theorem (dog_fish : ℕ) (fish_price : ℕ) :
  dog_fish = 40 →
  fish_price = 4 →
  (dog_fish + dog_fish / 2) * fish_price = 240 :=
by
  sorry

end NUMINAMATH_CALUDE_fish_cost_theorem_l4113_411348


namespace NUMINAMATH_CALUDE_stating_final_number_lower_bound_l4113_411389

/-- 
Given a sequence of n ones, we repeatedly replace two numbers a and b 
with (a+b)/4 for n-1 steps. This function represents the final number 
after these operations.
-/
noncomputable def final_number (n : ℕ) : ℝ :=
  sorry

/-- 
Theorem stating that the final number after n-1 steps of the described 
operation, starting with n ones, is greater than or equal to 1/n.
-/
theorem final_number_lower_bound (n : ℕ) (h : n > 0) : 
  final_number n ≥ 1 / n :=
sorry

end NUMINAMATH_CALUDE_stating_final_number_lower_bound_l4113_411389


namespace NUMINAMATH_CALUDE_product_trailing_zeros_l4113_411312

def trailing_zeros (n : ℕ) : ℕ := sorry

theorem product_trailing_zeros : 
  trailing_zeros (45 * 320 * 125) = 5 := by sorry

end NUMINAMATH_CALUDE_product_trailing_zeros_l4113_411312


namespace NUMINAMATH_CALUDE_rent_increase_percentage_l4113_411340

/-- Proves that the rent increase percentage is 25% given the specified conditions -/
theorem rent_increase_percentage 
  (num_friends : ℕ) 
  (initial_avg_rent : ℝ) 
  (new_avg_rent : ℝ) 
  (original_rent : ℝ) : 
  num_friends = 4 → 
  initial_avg_rent = 800 → 
  new_avg_rent = 850 → 
  original_rent = 800 → 
  (new_avg_rent * num_friends - initial_avg_rent * num_friends) / original_rent * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_rent_increase_percentage_l4113_411340


namespace NUMINAMATH_CALUDE_sunflower_height_l4113_411328

/-- Converts feet and inches to total inches -/
def feet_inches_to_inches (feet : ℕ) (inches : ℕ) : ℕ :=
  feet * 12 + inches

/-- Converts inches to feet, rounding down -/
def inches_to_feet (inches : ℕ) : ℕ :=
  inches / 12

theorem sunflower_height
  (sister_height_feet : ℕ)
  (sister_height_inches : ℕ)
  (sunflower_diff : ℕ)
  (h1 : sister_height_feet = 4)
  (h2 : sister_height_inches = 3)
  (h3 : sunflower_diff = 21) :
  inches_to_feet (feet_inches_to_inches sister_height_feet sister_height_inches + sunflower_diff) = 6 := by
  sorry

end NUMINAMATH_CALUDE_sunflower_height_l4113_411328


namespace NUMINAMATH_CALUDE_truncated_cube_vertex_edge_count_l4113_411392

/-- A polyhedron with 8 triangular faces and 6 heptagonal faces -/
structure TruncatedCube where
  triangularFaces : ℕ
  heptagonalFaces : ℕ
  triangularFaces_eq : triangularFaces = 8
  heptagonalFaces_eq : heptagonalFaces = 6

/-- The number of vertices in a TruncatedCube -/
def vertexCount (cube : TruncatedCube) : ℕ := 21

/-- The number of edges in a TruncatedCube -/
def edgeCount (cube : TruncatedCube) : ℕ := 33

/-- Theorem stating that a TruncatedCube has 21 vertices and 33 edges -/
theorem truncated_cube_vertex_edge_count (cube : TruncatedCube) : 
  vertexCount cube = 21 ∧ edgeCount cube = 33 := by
  sorry


end NUMINAMATH_CALUDE_truncated_cube_vertex_edge_count_l4113_411392


namespace NUMINAMATH_CALUDE_decimal_to_fraction_l4113_411361

theorem decimal_to_fraction : 
  (3.36 : ℚ) = 84 / 25 := by sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_l4113_411361


namespace NUMINAMATH_CALUDE_female_rabbits_count_l4113_411366

theorem female_rabbits_count (white_rabbits black_rabbits male_rabbits : ℕ) 
  (h1 : white_rabbits = 11)
  (h2 : black_rabbits = 13)
  (h3 : male_rabbits = 15) : 
  white_rabbits + black_rabbits - male_rabbits = 9 := by
  sorry

end NUMINAMATH_CALUDE_female_rabbits_count_l4113_411366


namespace NUMINAMATH_CALUDE_sum_fractions_equals_eight_l4113_411390

/-- Given real numbers a, b, and c satisfying specific conditions, 
    prove that b/(a + b) + c/(b + c) + a/(c + a) = 8 -/
theorem sum_fractions_equals_eight 
  (a b c : ℝ) 
  (h1 : a * c / (a + b) + b * a / (b + c) + c * b / (c + a) = -6)
  (h2 : b * c / (a + b) + c * a / (b + c) + a * b / (c + a) = 7) :
  b / (a + b) + c / (b + c) + a / (c + a) = 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_fractions_equals_eight_l4113_411390


namespace NUMINAMATH_CALUDE_ellipse_hyperbola_tangency_l4113_411396

/-- The value of m that makes the ellipse x^2 + 9y^2 = 9 tangent to the hyperbola x^2 - m(y+3)^2 = 4 -/
def tangency_value : ℚ := 5/54

/-- Definition of the ellipse equation -/
def is_on_ellipse (x y : ℝ) : Prop := x^2 + 9*y^2 = 9

/-- Definition of the hyperbola equation -/
def is_on_hyperbola (x y m : ℝ) : Prop := x^2 - m*(y+3)^2 = 4

/-- Theorem stating that 5/54 is the value of m that makes the ellipse tangent to the hyperbola -/
theorem ellipse_hyperbola_tangency :
  ∃! (m : ℝ), m = tangency_value ∧ 
  (∃! (x y : ℝ), is_on_ellipse x y ∧ is_on_hyperbola x y m) :=
sorry

end NUMINAMATH_CALUDE_ellipse_hyperbola_tangency_l4113_411396


namespace NUMINAMATH_CALUDE_point_on_x_axis_l4113_411370

theorem point_on_x_axis (x : ℝ) : 
  (x^2 + 2 + 9 = 12) → (x = 1 ∨ x = -1) := by
  sorry

#check point_on_x_axis

end NUMINAMATH_CALUDE_point_on_x_axis_l4113_411370


namespace NUMINAMATH_CALUDE_digit_cancellation_fractions_l4113_411372

theorem digit_cancellation_fractions :
  let valid_fractions : List (ℕ × ℕ) := [(26, 65), (16, 64), (19, 95), (49, 98)]
  ∀ a b c : ℕ,
    a < 10 ∧ b < 10 ∧ c < 10 →
    (10 * a + b) * c = (10 * b + c) * a →
    a < c →
    (10 * a + b, 10 * b + c) ∈ valid_fractions :=
by sorry

end NUMINAMATH_CALUDE_digit_cancellation_fractions_l4113_411372


namespace NUMINAMATH_CALUDE_binary_to_quinary_conversion_l4113_411330

/-- Converts a natural number from base 2 to base 10 -/
def base2_to_base10 (n : List Bool) : ℕ :=
  n.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- Converts a natural number from base 10 to base 5 -/
def base10_to_base5 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) :=
      if m = 0 then acc else aux (m / 5) ((m % 5) :: acc)
    aux n []

theorem binary_to_quinary_conversion :
  base10_to_base5 (base2_to_base10 [true, false, true, true, true]) = [1, 0, 4] := by
  sorry

end NUMINAMATH_CALUDE_binary_to_quinary_conversion_l4113_411330


namespace NUMINAMATH_CALUDE_island_perimeter_calculation_l4113_411371

/-- The perimeter of a rectangular island -/
def island_perimeter (width : ℝ) (length : ℝ) : ℝ :=
  2 * (width + length)

/-- Theorem: The perimeter of a rectangular island with width 4 miles and length 7 miles is 22 miles -/
theorem island_perimeter_calculation :
  island_perimeter 4 7 = 22 := by
  sorry

end NUMINAMATH_CALUDE_island_perimeter_calculation_l4113_411371


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l4113_411360

/-- The sum of an arithmetic sequence with first term 5, common difference 3, and 15 terms -/
def arithmetic_sum : ℕ := 
  let a₁ : ℕ := 5  -- first term
  let d : ℕ := 3   -- common difference
  let n : ℕ := 15  -- number of terms
  n * (2 * a₁ + (n - 1) * d) / 2

theorem arithmetic_sequence_sum : arithmetic_sum = 390 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l4113_411360


namespace NUMINAMATH_CALUDE_abc_remainder_mod_7_l4113_411318

theorem abc_remainder_mod_7 (a b c : ℕ) 
  (h_a : a < 7) (h_b : b < 7) (h_c : c < 7)
  (h1 : (a + 2*b + 3*c) % 7 = 0)
  (h2 : (2*a + 3*b + c) % 7 = 2)
  (h3 : (3*a + b + 2*c) % 7 = 4) :
  (a * b * c) % 7 = 0 := by
sorry

end NUMINAMATH_CALUDE_abc_remainder_mod_7_l4113_411318


namespace NUMINAMATH_CALUDE_smallest_prime_ten_less_square_l4113_411336

theorem smallest_prime_ten_less_square : ∃ (n : ℕ), 
  (∀ m : ℕ, m < n → ¬(Nat.Prime m ∧ ∃ k : ℕ, m = k^2 - 10)) ∧ 
  (Nat.Prime n ∧ ∃ k : ℕ, n = k^2 - 10) :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_ten_less_square_l4113_411336


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l4113_411393

theorem quadratic_inequality_range (a : ℝ) : 
  (∃ x : ℝ, x^2 + a*x + 1 < 0) → (a < -2 ∨ a > 2) := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l4113_411393


namespace NUMINAMATH_CALUDE_coefficient_x4_proof_l4113_411359

/-- The expression to be simplified -/
def expression (x : ℝ) : ℝ := 2 * (x^4 - 2*x^3) + 3 * (2*x^2 - 3*x^4 + x^6) - (5*x^6 - 2*x^4)

/-- The coefficient of x^4 in the simplified expression -/
def coefficient_x4 : ℝ := -5

theorem coefficient_x4_proof : ∃ (f : ℝ → ℝ), ∀ x, expression x = f x + coefficient_x4 * x^4 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x4_proof_l4113_411359


namespace NUMINAMATH_CALUDE_largest_number_with_conditions_l4113_411377

def is_valid_number (n : ℕ) : Prop :=
  let digits := n.digits 10
  (∀ i j, i ≠ j → digits.get i ≠ digits.get j) ∧
  (0 ∉ digits) ∧
  (digits.sum = 18)

theorem largest_number_with_conditions :
  ∀ n : ℕ, is_valid_number n → n ≤ 6543 :=
by sorry

end NUMINAMATH_CALUDE_largest_number_with_conditions_l4113_411377


namespace NUMINAMATH_CALUDE_halfway_fraction_l4113_411397

theorem halfway_fraction :
  ∃ (n d : ℕ), d ≠ 0 ∧ (n : ℚ) / d > 1 / 2 ∧
  (n : ℚ) / d = (3 / 4 + 5 / 7) / 2 ∧
  n = 41 ∧ d = 56 := by
  sorry

end NUMINAMATH_CALUDE_halfway_fraction_l4113_411397


namespace NUMINAMATH_CALUDE_horse_speed_problem_l4113_411325

/-- A problem from "Nine Chapters on the Mathematical Art" about horse speeds and travel times. -/
theorem horse_speed_problem (x : ℝ) (h_x : x > 3) : 
  let distance : ℝ := 900
  let slow_horse_time : ℝ := x + 1
  let fast_horse_time : ℝ := x - 3
  let slow_horse_speed : ℝ := distance / slow_horse_time
  let fast_horse_speed : ℝ := distance / fast_horse_time
  2 * slow_horse_speed = fast_horse_speed :=
by sorry

end NUMINAMATH_CALUDE_horse_speed_problem_l4113_411325


namespace NUMINAMATH_CALUDE_fraction_inequality_l4113_411332

theorem fraction_inequality (x : ℝ) :
  -3 ≤ x ∧ x ≤ 3 →
  (8 * x - 3 < 9 + 5 * x ↔ -3 ≤ x ∧ x < 3) :=
by sorry

end NUMINAMATH_CALUDE_fraction_inequality_l4113_411332


namespace NUMINAMATH_CALUDE_initial_birds_count_l4113_411300

theorem initial_birds_count (initial_storks : ℕ) (additional_birds : ℕ) (total_after : ℕ) :
  initial_storks = 2 →
  additional_birds = 5 →
  total_after = 10 →
  ∃ initial_birds : ℕ, initial_birds + initial_storks + additional_birds = total_after ∧ initial_birds = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_initial_birds_count_l4113_411300


namespace NUMINAMATH_CALUDE_min_value_of_function_l4113_411334

theorem min_value_of_function (x y : ℝ) : 2*x^2 + 3*y^2 + 8*x - 6*y + 5*x*y + 36 ≥ 10 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_function_l4113_411334


namespace NUMINAMATH_CALUDE_smallest_covering_l4113_411391

/-- A rectangle with integer dimensions -/
structure Rectangle where
  length : ℕ
  width : ℕ

/-- The area of a rectangle -/
def area (r : Rectangle) : ℕ := r.length * r.width

/-- A covering of rectangles -/
structure Covering where
  target : Rectangle
  tiles : List Rectangle

/-- Whether a covering is valid (complete and non-overlapping) -/
def is_valid_covering (c : Covering) : Prop :=
  (area c.target = (c.tiles.map area).sum) ∧
  (∀ r ∈ c.tiles, r.length = 3 ∧ r.width = 4)

/-- The main theorem -/
theorem smallest_covering :
  ∃ (c : Covering),
    is_valid_covering c ∧
    c.tiles.length = 2 ∧
    (∀ (c' : Covering), is_valid_covering c' → c'.tiles.length ≥ 2) :=
sorry

end NUMINAMATH_CALUDE_smallest_covering_l4113_411391


namespace NUMINAMATH_CALUDE_prime_sequence_l4113_411394

theorem prime_sequence (n : ℕ) (h1 : n ≥ 2) 
  (h2 : ∀ k : ℕ, 0 ≤ k ∧ k ≤ Real.sqrt (n / 3) → Nat.Prime (k^2 + k + n)) :
  ∀ k : ℕ, 0 ≤ k ∧ k ≤ n - 2 → Nat.Prime (k^2 + k + n) := by
sorry

end NUMINAMATH_CALUDE_prime_sequence_l4113_411394


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l4113_411368

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- The theorem to be proved -/
theorem geometric_sequence_sum (a : ℕ → ℝ) :
  geometric_sequence a →
  (a 5 + a 6 = 4) →
  (a 15 + a 16 = 16) →
  (a 25 + a 26 = 64) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l4113_411368


namespace NUMINAMATH_CALUDE_second_drive_speed_l4113_411352

def same_distance_drives (v : ℝ) : Prop :=
  let d := 180 / 3  -- distance for each drive
  (d / 4 + d / v + d / 6 = 37) ∧ (d / 4 + d / v + d / 6 > 0)

theorem second_drive_speed : ∃ v : ℝ, same_distance_drives v ∧ v = 5 := by
  sorry

end NUMINAMATH_CALUDE_second_drive_speed_l4113_411352


namespace NUMINAMATH_CALUDE_M_has_three_elements_l4113_411319

def M : Set ℝ :=
  {m | ∃ x y z : ℝ, x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧
    m = x / |x| + y / |y| + z / |z| + (x * y * z) / |x * y * z|}

theorem M_has_three_elements :
  ∃ a b c : ℝ, M = {a, b, c} :=
sorry

end NUMINAMATH_CALUDE_M_has_three_elements_l4113_411319


namespace NUMINAMATH_CALUDE_percentage_decrease_in_z_l4113_411333

/-- Given positive real numbers x and z, and a real number q, if x and (z+10) are inversely 
    proportional, and x increases by q%, then the percentage decrease in z is q(z+10)/(100+q)% -/
theorem percentage_decrease_in_z (x z q : ℝ) (hx : x > 0) (hz : z > 0) (hq : q ≠ -100) :
  (∃ k : ℝ, k > 0 ∧ x * (z + 10) = k) →
  let x' := x * (1 + q / 100)
  let z' := (100 / (100 + q)) * (z + 10) - 10
  (z - z') / z * 100 = q * (z + 10) / (100 + q) := by
  sorry

end NUMINAMATH_CALUDE_percentage_decrease_in_z_l4113_411333


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l4113_411367

/-- Triangle ABC with given properties -/
structure TriangleABC where
  /-- Point A coordinates -/
  A : ℝ × ℝ
  /-- Equation of median CM: 5√3x + 9y - 18 = 0 -/
  CM : ℝ → ℝ → Prop
  /-- Equation of angle bisector BT: y = 1 -/
  BT : ℝ → ℝ → Prop

/-- Properties of the given triangle -/
def triangle_properties (t : TriangleABC) : Prop :=
  t.A = (Real.sqrt 3, 3) ∧
  (∀ x y, t.CM x y ↔ 5 * Real.sqrt 3 * x + 9 * y - 18 = 0) ∧
  (∀ x y, t.BT x y ↔ y = 1)

/-- Theorem stating the properties of vertex B and the area of the triangle -/
theorem triangle_abc_properties (t : TriangleABC) 
  (h : triangle_properties t) : 
  (∃ B : ℝ × ℝ, B = (-Real.sqrt 3, 1)) ∧ 
  (∃ area : ℝ, area = 8 * Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l4113_411367
