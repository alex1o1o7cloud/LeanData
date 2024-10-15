import Mathlib

namespace NUMINAMATH_CALUDE_sixteen_seats_painting_ways_l2079_207901

def paintingWays (n : ℕ) : ℕ := 
  let rec a : ℕ → ℕ
    | 0 => 1
    | 1 => 1
    | i + 1 => (List.range ((i + 1) / 2 + 1)).foldl (λ sum j => sum + a (i - 2 * j)) 0
  2 * a n

theorem sixteen_seats_painting_ways :
  paintingWays 16 = 1686 := by sorry

end NUMINAMATH_CALUDE_sixteen_seats_painting_ways_l2079_207901


namespace NUMINAMATH_CALUDE_peach_count_l2079_207967

theorem peach_count (initial : ℕ) (picked : ℕ) (total : ℕ) : 
  initial = 34 → picked = 52 → total = initial + picked → total = 86 := by
sorry

end NUMINAMATH_CALUDE_peach_count_l2079_207967


namespace NUMINAMATH_CALUDE_find_d_l2079_207968

theorem find_d (a b c d : ℝ) 
  (h : a^2 + b^2 + c^2 + 4 = 2*d + Real.sqrt (a + b + c + d)) : 
  d = (-7 + Real.sqrt 33) / 8 := by
sorry

end NUMINAMATH_CALUDE_find_d_l2079_207968


namespace NUMINAMATH_CALUDE_rotated_angle_measure_l2079_207917

/-- Given an initial angle of 60 degrees and a clockwise rotation of 600 degrees,
    the resulting new acute angle is 60 degrees. -/
theorem rotated_angle_measure (initial_angle rotation : ℝ) : 
  initial_angle = 60 →
  rotation = 600 →
  let effective_rotation := rotation % 360
  let new_angle := (effective_rotation - initial_angle) % 180
  new_angle = 60 :=
by sorry

end NUMINAMATH_CALUDE_rotated_angle_measure_l2079_207917


namespace NUMINAMATH_CALUDE_cans_per_bag_l2079_207954

/-- Given that Paul filled 6 bags on Saturday, 3 bags on Sunday, and collected a total of 72 cans,
    prove that the number of cans in each bag is 8. -/
theorem cans_per_bag (saturday_bags : Nat) (sunday_bags : Nat) (total_cans : Nat) :
  saturday_bags = 6 →
  sunday_bags = 3 →
  total_cans = 72 →
  total_cans / (saturday_bags + sunday_bags) = 8 := by
  sorry

end NUMINAMATH_CALUDE_cans_per_bag_l2079_207954


namespace NUMINAMATH_CALUDE_least_positive_angle_theorem_l2079_207971

theorem least_positive_angle_theorem : 
  ∃ θ : Real, θ > 0 ∧ θ = 15 * π / 180 ∧
  (∀ φ : Real, φ > 0 ∧ Real.cos (15 * π / 180) = Real.sin (45 * π / 180) + Real.sin φ → θ ≤ φ) :=
sorry

end NUMINAMATH_CALUDE_least_positive_angle_theorem_l2079_207971


namespace NUMINAMATH_CALUDE_total_tires_is_101_l2079_207969

/-- The number of tires on a car -/
def car_tires : ℕ := 4

/-- The number of tires on a bicycle -/
def bicycle_tires : ℕ := 2

/-- The number of tires on a pickup truck -/
def pickup_truck_tires : ℕ := 4

/-- The number of tires on a tricycle -/
def tricycle_tires : ℕ := 3

/-- The number of cars Juan saw -/
def cars_seen : ℕ := 15

/-- The number of bicycles Juan saw -/
def bicycles_seen : ℕ := 3

/-- The number of pickup trucks Juan saw -/
def pickup_trucks_seen : ℕ := 8

/-- The number of tricycles Juan saw -/
def tricycles_seen : ℕ := 1

/-- The total number of tires on all vehicles Juan saw -/
def total_tires : ℕ := 
  car_tires * cars_seen + 
  bicycle_tires * bicycles_seen + 
  pickup_truck_tires * pickup_trucks_seen + 
  tricycle_tires * tricycles_seen

theorem total_tires_is_101 : total_tires = 101 := by
  sorry

end NUMINAMATH_CALUDE_total_tires_is_101_l2079_207969


namespace NUMINAMATH_CALUDE_complex_power_36_135_deg_l2079_207910

theorem complex_power_36_135_deg :
  (Complex.exp (Complex.I * Real.pi * (3 / 4)))^36 = Complex.ofReal (1 / 2) - Complex.I * Complex.ofReal (Real.sqrt 3 / 2) := by
  sorry

end NUMINAMATH_CALUDE_complex_power_36_135_deg_l2079_207910


namespace NUMINAMATH_CALUDE_descending_order_original_statement_l2079_207942

theorem descending_order : 0.38 > 0.373 ∧ 0.373 > 0.37 := by
  sorry

-- Define 37% as 0.37
def thirty_seven_percent : ℝ := 0.37

-- Prove that the original statement holds
theorem original_statement : 0.38 > 0.373 ∧ 0.373 > thirty_seven_percent := by
  sorry

end NUMINAMATH_CALUDE_descending_order_original_statement_l2079_207942


namespace NUMINAMATH_CALUDE_perfect_square_binomial_condition_l2079_207944

/-- A quadratic expression is a perfect square binomial if it can be written as (px + q)^2 for some real p and q -/
def IsPerfectSquareBinomial (f : ℝ → ℝ) : Prop :=
  ∃ p q : ℝ, ∀ x, f x = (p * x + q)^2

/-- Given that 9x^2 - 27x + a is a perfect square binomial, prove that a = 20.25 -/
theorem perfect_square_binomial_condition (a : ℝ) 
  (h : IsPerfectSquareBinomial (fun x ↦ 9*x^2 - 27*x + a)) : 
  a = 20.25 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_binomial_condition_l2079_207944


namespace NUMINAMATH_CALUDE_polynomial_inequality_l2079_207976

theorem polynomial_inequality (m : ℚ) : 5 * m^2 - 8 * m + 1 > 4 * m^2 - 8 * m - 1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_inequality_l2079_207976


namespace NUMINAMATH_CALUDE_identical_pairs_imply_x_equals_four_l2079_207920

-- Define the binary operation ★
def star (a b c d : ℤ) : ℤ × ℤ := (a - 2*c, b + 2*d)

-- Theorem statement
theorem identical_pairs_imply_x_equals_four :
  ∀ x y : ℤ, star 2 (-4) 1 (-3) = star x y 2 1 → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_identical_pairs_imply_x_equals_four_l2079_207920


namespace NUMINAMATH_CALUDE_bus_intersection_percentages_l2079_207981

theorem bus_intersection_percentages : 
  let first_intersection_entrants : ℝ := 12
  let second_intersection_entrants : ℝ := 18
  let third_intersection_entrants : ℝ := 15
  (0.3 * first_intersection_entrants + 
   0.5 * second_intersection_entrants + 
   0.2 * third_intersection_entrants) = 15.6 := by
  sorry

end NUMINAMATH_CALUDE_bus_intersection_percentages_l2079_207981


namespace NUMINAMATH_CALUDE_fraction_inequality_l2079_207964

theorem fraction_inequality (a b m : ℝ) (h1 : b > a) (h2 : m > 0) :
  b / a > (b + m) / (a + m) := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_l2079_207964


namespace NUMINAMATH_CALUDE_aaron_used_three_boxes_l2079_207979

/-- Given the initial number of can lids, final number of can lids, and lids per box,
    calculate the number of boxes used. -/
def boxes_used (initial_lids : ℕ) (final_lids : ℕ) (lids_per_box : ℕ) : ℕ :=
  (final_lids - initial_lids) / lids_per_box

/-- Theorem stating that Aaron used 3 boxes of canned tomatoes. -/
theorem aaron_used_three_boxes :
  boxes_used 14 53 13 = 3 := by
  sorry

end NUMINAMATH_CALUDE_aaron_used_three_boxes_l2079_207979


namespace NUMINAMATH_CALUDE_mean_equals_n_l2079_207952

theorem mean_equals_n (n : ℝ) : 
  (17 + 98 + 39 + 54 + n) / 5 = n → n = 52 := by
  sorry

end NUMINAMATH_CALUDE_mean_equals_n_l2079_207952


namespace NUMINAMATH_CALUDE_max_profit_is_45_6_l2079_207933

-- Define the profit functions
def profit_A (t : ℕ) : ℚ := 5.06 * t - 0.15 * t^2
def profit_B (t : ℕ) : ℚ := 2 * t

-- Define the total profit function
def total_profit (x : ℕ) : ℚ := profit_A x + profit_B (15 - x)

-- Theorem statement
theorem max_profit_is_45_6 :
  ∃ (x : ℕ), x ≤ 15 ∧ total_profit x = 45.6 ∧
  ∀ (y : ℕ), y ≤ 15 → total_profit y ≤ 45.6 := by
  sorry


end NUMINAMATH_CALUDE_max_profit_is_45_6_l2079_207933


namespace NUMINAMATH_CALUDE_ladder_slide_l2079_207902

theorem ladder_slide (ladder_length : ℝ) (initial_distance : ℝ) (slip_distance : ℝ) :
  ladder_length = 30 →
  initial_distance = 8 →
  slip_distance = 6 →
  let initial_height : ℝ := Real.sqrt (ladder_length^2 - initial_distance^2)
  let new_height : ℝ := initial_height - slip_distance
  let new_distance : ℝ := Real.sqrt (ladder_length^2 - new_height^2)
  new_distance - initial_distance = 10 := by
  sorry

end NUMINAMATH_CALUDE_ladder_slide_l2079_207902


namespace NUMINAMATH_CALUDE_percentage_problem_l2079_207911

theorem percentage_problem (x : ℝ) : (27 / x = 45 / 100) → x = 60 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l2079_207911


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l2079_207996

open Set

def U : Set Nat := {1, 2, 3, 4, 5}
def A : Set Nat := {1, 3, 5}
def B : Set Nat := {3, 4}

theorem complement_intersection_theorem :
  (U \ A) ∩ B = {4} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l2079_207996


namespace NUMINAMATH_CALUDE_negation_of_ln_positive_l2079_207929

theorem negation_of_ln_positive :
  (¬ ∀ x : ℝ, x > 0 → Real.log x > 0) ↔ (∃ x : ℝ, x > 0 ∧ Real.log x ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_ln_positive_l2079_207929


namespace NUMINAMATH_CALUDE_largest_consecutive_sum_35_l2079_207990

def sum_of_consecutive_integers (start : ℕ) (count : ℕ) : ℕ :=
  count * (2 * start + count - 1) / 2

theorem largest_consecutive_sum_35 :
  (∃ (start : ℕ), sum_of_consecutive_integers start 7 = 35) ∧
  (∀ (start count : ℕ), count > 7 → sum_of_consecutive_integers start count ≠ 35) :=
sorry

end NUMINAMATH_CALUDE_largest_consecutive_sum_35_l2079_207990


namespace NUMINAMATH_CALUDE_hawks_score_l2079_207984

def total_points : ℕ := 82
def margin : ℕ := 22

theorem hawks_score (eagles_score hawks_score : ℕ) 
  (h1 : eagles_score + hawks_score = total_points)
  (h2 : eagles_score = hawks_score + margin) : 
  hawks_score = 30 := by
  sorry

end NUMINAMATH_CALUDE_hawks_score_l2079_207984


namespace NUMINAMATH_CALUDE_sum_of_tens_and_units_digits_of_9_pow_2004_l2079_207965

/-- The sum of the tens digit and the units digit in the decimal representation of 9^2004 is 7. -/
theorem sum_of_tens_and_units_digits_of_9_pow_2004 : ∃ n : ℕ, 9^2004 = 100 * n + 61 :=
sorry

end NUMINAMATH_CALUDE_sum_of_tens_and_units_digits_of_9_pow_2004_l2079_207965


namespace NUMINAMATH_CALUDE_average_rounds_played_l2079_207912

/-- Represents the distribution of golf rounds played by members -/
def golf_distribution : List (Nat × Nat) := [(1, 3), (2, 4), (3, 6), (4, 3), (5, 2)]

/-- Calculates the total number of rounds played -/
def total_rounds (dist : List (Nat × Nat)) : Nat :=
  dist.foldr (fun p acc => p.1 * p.2 + acc) 0

/-- Calculates the total number of golfers -/
def total_golfers (dist : List (Nat × Nat)) : Nat :=
  dist.foldr (fun p acc => p.2 + acc) 0

/-- Rounds a rational number to the nearest integer -/
def round_to_nearest (x : Rat) : Int :=
  if x - x.floor < 1/2 then x.floor else x.ceil

theorem average_rounds_played : 
  round_to_nearest ((total_rounds golf_distribution : Rat) / total_golfers golf_distribution) = 3 := by
  sorry

end NUMINAMATH_CALUDE_average_rounds_played_l2079_207912


namespace NUMINAMATH_CALUDE_liz_shopping_cost_l2079_207928

def problem (recipe_book_cost baking_dish_cost ingredient_cost apron_cost mixer_cost measuring_cups_cost spice_cost discount : ℝ) : Prop :=
  let total_cost := 
    recipe_book_cost + 
    baking_dish_cost + 
    (5 * ingredient_cost) + 
    apron_cost + 
    mixer_cost + 
    measuring_cups_cost + 
    (4 * spice_cost) - 
    discount
  total_cost = 84.5 ∧
  recipe_book_cost = 6 ∧
  baking_dish_cost = 2 * recipe_book_cost ∧
  ingredient_cost = 3 ∧
  apron_cost = recipe_book_cost + 1 ∧
  mixer_cost = 3 * baking_dish_cost ∧
  measuring_cups_cost = apron_cost / 2 ∧
  spice_cost = 2 ∧
  discount = 3

theorem liz_shopping_cost : ∃ (recipe_book_cost baking_dish_cost ingredient_cost apron_cost mixer_cost measuring_cups_cost spice_cost discount : ℝ),
  problem recipe_book_cost baking_dish_cost ingredient_cost apron_cost mixer_cost measuring_cups_cost spice_cost discount :=
by sorry

end NUMINAMATH_CALUDE_liz_shopping_cost_l2079_207928


namespace NUMINAMATH_CALUDE_ronaldo_age_l2079_207927

theorem ronaldo_age (ronnie_age_last_year : ℕ) (ronaldo_age_last_year : ℕ) 
  (h1 : ronnie_age_last_year * 7 = ronaldo_age_last_year * 6)
  (h2 : (ronnie_age_last_year + 5) * 8 = (ronaldo_age_last_year + 5) * 7) :
  ronaldo_age_last_year + 1 = 36 := by
  sorry

end NUMINAMATH_CALUDE_ronaldo_age_l2079_207927


namespace NUMINAMATH_CALUDE_andrew_kept_130_stickers_l2079_207980

def total_stickers : ℕ := 750
def daniel_stickers : ℕ := 250
def fred_extra_stickers : ℕ := 120

def andrew_kept_stickers : ℕ := total_stickers - (daniel_stickers + (daniel_stickers + fred_extra_stickers))

theorem andrew_kept_130_stickers : andrew_kept_stickers = 130 := by
  sorry

end NUMINAMATH_CALUDE_andrew_kept_130_stickers_l2079_207980


namespace NUMINAMATH_CALUDE_coin_grid_probability_l2079_207993

/-- Represents a square grid -/
structure Grid where
  size : ℕ  -- number of squares on each side
  square_size : ℝ  -- side length of each square
  
/-- Represents a circular coin -/
structure Coin where
  diameter : ℝ

/-- Calculates the probability of a coin landing in a winning position on a grid -/
def winning_probability (g : Grid) (c : Coin) : ℝ :=
  sorry

/-- The main theorem to be proved -/
theorem coin_grid_probability :
  let g : Grid := { size := 5, square_size := 10 }
  let c : Coin := { diameter := 8 }
  winning_probability g c = 25 / 441 := by
  sorry

end NUMINAMATH_CALUDE_coin_grid_probability_l2079_207993


namespace NUMINAMATH_CALUDE_state_tax_rate_is_4_percent_l2079_207946

/-- Calculates the state tax rate given the following conditions:
  * The taxpayer was a resident for 9 months out of the year
  * The taxpayer's taxable income for the year
  * The prorated tax amount paid for the time of residency
-/
def calculate_state_tax_rate (months_resident : ℕ) (taxable_income : ℚ) (tax_paid : ℚ) : ℚ :=
  let full_year_months : ℕ := 12
  let residence_ratio : ℚ := months_resident / full_year_months
  let full_year_tax : ℚ := tax_paid / residence_ratio
  (full_year_tax / taxable_income) * 100

theorem state_tax_rate_is_4_percent :
  let months_resident : ℕ := 9
  let taxable_income : ℚ := 42500
  let tax_paid : ℚ := 1275
  calculate_state_tax_rate months_resident taxable_income tax_paid = 4 := by
  sorry

end NUMINAMATH_CALUDE_state_tax_rate_is_4_percent_l2079_207946


namespace NUMINAMATH_CALUDE_division_problem_l2079_207931

theorem division_problem (dividend : Nat) (divisor : Nat) (quotient : Nat) (remainder : Nat) :
  dividend = 95 →
  divisor = 15 →
  remainder = 5 →
  dividend = divisor * quotient + remainder →
  quotient = 6 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l2079_207931


namespace NUMINAMATH_CALUDE_alloy_mixture_theorem_l2079_207948

/-- The percentage of chromium in the first alloy -/
def chromium_percent_1 : ℝ := 12

/-- The percentage of chromium in the second alloy -/
def chromium_percent_2 : ℝ := 8

/-- The amount of the first alloy used (in kg) -/
def amount_1 : ℝ := 20

/-- The percentage of chromium in the new alloy -/
def new_chromium_percent : ℝ := 9.454545454545453

/-- The amount of the second alloy used (in kg) -/
def amount_2 : ℝ := 35

theorem alloy_mixture_theorem :
  chromium_percent_1 * amount_1 / 100 + chromium_percent_2 * amount_2 / 100 =
  new_chromium_percent * (amount_1 + amount_2) / 100 := by
  sorry

end NUMINAMATH_CALUDE_alloy_mixture_theorem_l2079_207948


namespace NUMINAMATH_CALUDE_negation_of_existence_real_roots_l2079_207974

theorem negation_of_existence_real_roots :
  (¬ ∃ m : ℝ, ∃ x : ℝ, x^2 + m*x + 1 = 0) ↔
  (∀ m : ℝ, ∀ x : ℝ, x^2 + m*x + 1 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_real_roots_l2079_207974


namespace NUMINAMATH_CALUDE_divisibility_by_441_l2079_207958

theorem divisibility_by_441 (a b : ℕ) (h : 21 ∣ (a^2 + b^2)) : 441 ∣ (a^2 + b^2) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_441_l2079_207958


namespace NUMINAMATH_CALUDE_optimal_washing_effect_l2079_207907

/-- Represents the laundry scenario with given parameters -/
structure LaundryScenario where
  tub_capacity : ℝ
  clothes_weight : ℝ
  initial_detergent_scoops : ℕ
  scoop_weight : ℝ
  optimal_ratio : ℝ

/-- Calculates the optimal amount of detergent and water to add -/
def optimal_addition (scenario : LaundryScenario) : ℝ × ℝ :=
  sorry

/-- Theorem stating that the calculated optimal addition achieves the desired washing effect -/
theorem optimal_washing_effect (scenario : LaundryScenario) 
  (h1 : scenario.tub_capacity = 20)
  (h2 : scenario.clothes_weight = 5)
  (h3 : scenario.initial_detergent_scoops = 2)
  (h4 : scenario.scoop_weight = 0.02)
  (h5 : scenario.optimal_ratio = 0.004) :
  let (added_detergent, added_water) := optimal_addition scenario
  (added_detergent = 0.02) ∧ 
  (added_water = 14.94) ∧
  (scenario.tub_capacity = scenario.clothes_weight + added_water + added_detergent + scenario.initial_detergent_scoops * scenario.scoop_weight) ∧
  (added_detergent + scenario.initial_detergent_scoops * scenario.scoop_weight = scenario.optimal_ratio * (added_water + scenario.initial_detergent_scoops * scenario.scoop_weight)) :=
by
  sorry


end NUMINAMATH_CALUDE_optimal_washing_effect_l2079_207907


namespace NUMINAMATH_CALUDE_triangle_inradius_l2079_207923

/-- The inradius of a triangle with side lengths 7, 24, and 25 is 3 -/
theorem triangle_inradius (a b c : ℝ) (h1 : a = 7) (h2 : b = 24) (h3 : c = 25) :
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  area / s = 3 := by sorry

end NUMINAMATH_CALUDE_triangle_inradius_l2079_207923


namespace NUMINAMATH_CALUDE_books_read_by_three_l2079_207987

/-- The number of different books read by three people given their individual book counts and overlap -/
def total_different_books (tony_books dean_books breanna_books tony_dean_overlap all_overlap : ℕ) : ℕ :=
  tony_books + dean_books + breanna_books - tony_dean_overlap - 2 * all_overlap

/-- Theorem stating the total number of different books read by Tony, Dean, and Breanna -/
theorem books_read_by_three :
  total_different_books 23 12 17 3 1 = 47 := by
  sorry

end NUMINAMATH_CALUDE_books_read_by_three_l2079_207987


namespace NUMINAMATH_CALUDE_compound_has_six_hydrogen_atoms_l2079_207936

/-- The atomic weight of carbon in g/mol -/
def carbon_weight : ℝ := 12.01

/-- The atomic weight of oxygen in g/mol -/
def oxygen_weight : ℝ := 16.00

/-- The atomic weight of hydrogen in g/mol -/
def hydrogen_weight : ℝ := 1.008

/-- The total molecular weight of the compound in g/mol -/
def total_weight : ℝ := 122

/-- The number of carbon atoms in the compound -/
def carbon_count : ℕ := 7

/-- The number of oxygen atoms in the compound -/
def oxygen_count : ℕ := 2

/-- Calculate the molecular weight of the compound given the number of hydrogen atoms -/
def molecular_weight (hydrogen_count : ℕ) : ℝ :=
  carbon_weight * carbon_count + oxygen_weight * oxygen_count + hydrogen_weight * hydrogen_count

/-- Theorem stating that the compound has 6 hydrogen atoms -/
theorem compound_has_six_hydrogen_atoms :
  ∃ (n : ℕ), molecular_weight n = total_weight ∧ n = 6 := by
  sorry

end NUMINAMATH_CALUDE_compound_has_six_hydrogen_atoms_l2079_207936


namespace NUMINAMATH_CALUDE_bonus_pool_ratio_l2079_207904

theorem bonus_pool_ratio (P : ℕ) (k : ℕ) (h1 : P % 5 = 2) (h2 : (k * P) % 5 = 1) :
  k = 3 :=
sorry

end NUMINAMATH_CALUDE_bonus_pool_ratio_l2079_207904


namespace NUMINAMATH_CALUDE_last_number_proof_l2079_207957

theorem last_number_proof (a b c d : ℝ) 
  (h1 : (a + b + c) / 3 = 6)
  (h2 : a + d = 13)
  (h3 : (b + c + d) / 3 = 3) : 
  d = 2 := by
sorry

end NUMINAMATH_CALUDE_last_number_proof_l2079_207957


namespace NUMINAMATH_CALUDE_pascal_diagonal_sum_equals_fibonacci_l2079_207925

/-- Sum of numbers in the n-th diagonal of Pascal's triangle -/
def b (n : ℕ) : ℕ := sorry

/-- n-th term of the Fibonacci sequence -/
def a : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => a (n + 1) + a n

/-- Theorem stating that b_n equals a_n for all n -/
theorem pascal_diagonal_sum_equals_fibonacci (n : ℕ) : b n = a n := by
  sorry

end NUMINAMATH_CALUDE_pascal_diagonal_sum_equals_fibonacci_l2079_207925


namespace NUMINAMATH_CALUDE_ratio_HB_JD_l2079_207988

-- Define the points
variable (A B C D E F G H J : ℝ × ℝ)

-- Define the conditions
axiom on_line : ∃ (t : ℝ), B = A + t • (F - A) ∧
                           C = A + (t + 1) • (F - A) ∧
                           D = A + (t + 3) • (F - A) ∧
                           E = A + (t + 4) • (F - A) ∧
                           F = A + (t + 5) • (F - A)

axiom segment_lengths : 
  dist A B = 1 ∧ dist B C = 2 ∧ dist C D = 1 ∧ dist D E = 2 ∧ dist E F = 1

axiom G_not_on_line : ∀ (t : ℝ), G ≠ A + t • (F - A)

axiom H_on_GD : ∃ (t : ℝ), H = G + t • (D - G)

axiom J_on_GE : ∃ (t : ℝ), J = G + t • (E - G)

axiom parallel_lines : 
  (H.2 - B.2) / (H.1 - B.1) = (J.2 - D.2) / (J.1 - D.1) ∧
  (J.2 - D.2) / (J.1 - D.1) = (G.2 - A.2) / (G.1 - A.1)

-- Theorem to prove
theorem ratio_HB_JD : dist H B / dist J D = 5 / 4 :=
sorry

end NUMINAMATH_CALUDE_ratio_HB_JD_l2079_207988


namespace NUMINAMATH_CALUDE_james_container_capacity_l2079_207914

/-- Represents the capacity of different container types and their quantities --/
structure ContainerInventory where
  largeCaskCapacity : ℕ
  barrelCapacity : ℕ
  smallCaskCapacity : ℕ
  glassBottleCapacity : ℕ
  clayJugCapacity : ℕ
  barrelCount : ℕ
  largeCaskCount : ℕ
  smallCaskCount : ℕ
  glassBottleCount : ℕ
  clayJugCount : ℕ

/-- Calculates the total capacity of all containers --/
def totalCapacity (inv : ContainerInventory) : ℕ :=
  inv.barrelCapacity * inv.barrelCount +
  inv.largeCaskCapacity * inv.largeCaskCount +
  inv.smallCaskCapacity * inv.smallCaskCount +
  inv.glassBottleCapacity * inv.glassBottleCount +
  inv.clayJugCapacity * inv.clayJugCount

/-- Theorem stating that James' total container capacity is 318 gallons --/
theorem james_container_capacity :
  ∀ (inv : ContainerInventory),
    inv.largeCaskCapacity = 20 →
    inv.barrelCapacity = 2 * inv.largeCaskCapacity + 3 →
    inv.smallCaskCapacity = inv.largeCaskCapacity / 2 →
    inv.glassBottleCapacity = inv.smallCaskCapacity / 10 →
    inv.clayJugCapacity = 3 * inv.glassBottleCapacity →
    inv.barrelCount = 4 →
    inv.largeCaskCount = 3 →
    inv.smallCaskCount = 5 →
    inv.glassBottleCount = 12 →
    inv.clayJugCount = 8 →
    totalCapacity inv = 318 :=
by
  sorry


end NUMINAMATH_CALUDE_james_container_capacity_l2079_207914


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l2079_207949

/-- An arithmetic sequence with its sum satisfying a specific quadratic equation -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Sum of first n terms
  A : ℝ      -- Coefficient of n^2
  B : ℝ      -- Coefficient of n
  h1 : A ≠ 0
  h2 : ∀ n : ℕ, a n + S n = A * n^2 + B * n + 1

/-- The main theorem: if an arithmetic sequence satisfies the given condition, then (B-1)/A = 3 -/
theorem arithmetic_sequence_property (seq : ArithmeticSequence) : (seq.B - 1) / seq.A = 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l2079_207949


namespace NUMINAMATH_CALUDE_subset_M_N_l2079_207977

theorem subset_M_N : ∀ (x y : ℝ), (|x| + |y| ≤ 1) → (x^2 + y^2 ≤ |x| + |y|) := by
  sorry

end NUMINAMATH_CALUDE_subset_M_N_l2079_207977


namespace NUMINAMATH_CALUDE_expression_is_equation_l2079_207924

/-- Definition of an equation -/
def is_equation (e : Prop) : Prop :=
  ∃ (x : ℝ) (f g : ℝ → ℝ), e = (f x = g x)

/-- The expression 2x - 1 = 3 -/
def expression : Prop :=
  ∃ x : ℝ, 2 * x - 1 = 3

/-- Theorem: The given expression is an equation -/
theorem expression_is_equation : is_equation expression :=
sorry

end NUMINAMATH_CALUDE_expression_is_equation_l2079_207924


namespace NUMINAMATH_CALUDE_trains_passing_time_l2079_207915

theorem trains_passing_time (train_length : ℝ) (train_speed : ℝ) : 
  train_length = 500 →
  train_speed = 30 →
  (2 * train_length) / (2 * train_speed * (5/18)) = 60 :=
by
  sorry

#check trains_passing_time

end NUMINAMATH_CALUDE_trains_passing_time_l2079_207915


namespace NUMINAMATH_CALUDE_three_planes_division_l2079_207926

/-- A plane in 3D space -/
structure Plane3D where
  -- Add necessary fields or axioms here
  dummy : Unit

/-- The number of parts that a set of planes divides 3D space into -/
def num_parts (planes : Set Plane3D) : ℕ := sorry

theorem three_planes_division :
  ∀ (p1 p2 p3 : Plane3D),
  4 ≤ num_parts {p1, p2, p3} ∧ num_parts {p1, p2, p3} ≤ 8 := by
  sorry

end NUMINAMATH_CALUDE_three_planes_division_l2079_207926


namespace NUMINAMATH_CALUDE_chocolate_division_l2079_207992

theorem chocolate_division (total_chocolate : ℚ) (num_piles : ℕ) (piles_given : ℕ) :
  total_chocolate = 72 / 7 →
  num_piles = 6 →
  piles_given = 2 →
  piles_given * (total_chocolate / num_piles) = 24 / 7 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_division_l2079_207992


namespace NUMINAMATH_CALUDE_calculation_one_l2079_207935

theorem calculation_one : 6.8 - (-4.2) + (-4) * (-3) = 23 := by
  sorry

end NUMINAMATH_CALUDE_calculation_one_l2079_207935


namespace NUMINAMATH_CALUDE_perimeter_is_24_l2079_207999

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/36 + y^2/25 = 1

-- Define the foci F₁ and F₂
def F₁ : ℝ × ℝ := sorry
def F₂ : ℝ × ℝ := sorry

-- Define points A and B on the ellipse
def A : ℝ × ℝ := sorry
def B : ℝ × ℝ := sorry

-- State that A and B are on the ellipse
axiom A_on_ellipse : ellipse A.1 A.2
axiom B_on_ellipse : ellipse B.1 B.2

-- State that A, B, and F₁ are collinear
axiom A_B_F₁_collinear : ∃ (t : ℝ), A = F₁ + t • (B - F₁) ∨ B = F₁ + t • (A - F₁)

-- Define the perimeter of triangle ABF₂
def perimeter_ABF₂ : ℝ := sorry

-- Theorem to prove
theorem perimeter_is_24 : perimeter_ABF₂ = 24 := by sorry

end NUMINAMATH_CALUDE_perimeter_is_24_l2079_207999


namespace NUMINAMATH_CALUDE_optimal_price_is_160_l2079_207956

/-- Represents the price and occupancy data for a hotel room --/
structure PriceOccupancy where
  price : ℝ
  occupancy : ℝ

/-- Calculates the daily income for a given price and occupancy --/
def dailyIncome (po : PriceOccupancy) (totalRooms : ℝ) : ℝ :=
  po.price * po.occupancy * totalRooms

/-- Theorem: The optimal price for maximizing daily income is 160 yuan --/
theorem optimal_price_is_160 (totalRooms : ℝ) 
  (priceOccupancyData : List PriceOccupancy) 
  (h1 : totalRooms = 100)
  (h2 : priceOccupancyData = [
    ⟨200, 0.65⟩, 
    ⟨180, 0.75⟩, 
    ⟨160, 0.85⟩, 
    ⟨140, 0.95⟩
  ]) : 
  ∃ (optimalPO : PriceOccupancy), 
    optimalPO ∈ priceOccupancyData ∧ 
    optimalPO.price = 160 ∧
    ∀ (po : PriceOccupancy), 
      po ∈ priceOccupancyData → 
      dailyIncome optimalPO totalRooms ≥ dailyIncome po totalRooms :=
by sorry

end NUMINAMATH_CALUDE_optimal_price_is_160_l2079_207956


namespace NUMINAMATH_CALUDE_lucky_point_properties_l2079_207940

/-- Definition of a lucky point -/
def is_lucky_point (m n x y : ℝ) : Prop :=
  2 * m = 4 + n ∧ x = m - 1 ∧ y = (n + 2) / 2

theorem lucky_point_properties :
  -- Part 1: When m = 2, the lucky point is (1, 1)
  (∃ n : ℝ, is_lucky_point 2 n 1 1) ∧
  -- Part 2: Point (3, 3) is a lucky point
  (∃ m n : ℝ, is_lucky_point m n 3 3) ∧
  -- Part 3: If (a, 2a-1) is a lucky point, then it's in the first quadrant
  (∀ a m n : ℝ, is_lucky_point m n a (2*a-1) → a > 0 ∧ 2*a-1 > 0) :=
by sorry

end NUMINAMATH_CALUDE_lucky_point_properties_l2079_207940


namespace NUMINAMATH_CALUDE_time_spent_on_other_subjects_l2079_207937

def total_time : ℝ := 150

def math_percent : ℝ := 0.20
def science_percent : ℝ := 0.25
def history_percent : ℝ := 0.10
def english_percent : ℝ := 0.15

def min_time_remaining_subject : ℝ := 30

theorem time_spent_on_other_subjects :
  let math_time := total_time * math_percent
  let science_time := total_time * science_percent
  let history_time := total_time * history_percent
  let english_time := total_time * english_percent
  let known_subjects_time := math_time + science_time + history_time + english_time
  let remaining_time := total_time - known_subjects_time
  remaining_time - min_time_remaining_subject = 15 := by
  sorry

end NUMINAMATH_CALUDE_time_spent_on_other_subjects_l2079_207937


namespace NUMINAMATH_CALUDE_second_box_clay_capacity_l2079_207982

/-- Represents the dimensions and clay capacity of a box -/
structure Box where
  height : ℝ
  width : ℝ
  length : ℝ
  clayCapacity : ℝ

/-- Calculates the volume of a box -/
def boxVolume (b : Box) : ℝ := b.height * b.width * b.length

/-- Theorem stating the clay capacity of the second box -/
theorem second_box_clay_capacity 
  (box1 : Box)
  (box2 : Box)
  (h1 : box1.height = 4)
  (h2 : box1.width = 3)
  (h3 : box1.length = 7)
  (h4 : box1.clayCapacity = 84)
  (h5 : box2.height = box1.height / 2)
  (h6 : box2.width = box1.width * 4)
  (h7 : box2.length = box1.length)
  (h8 : boxVolume box1 * box1.clayCapacity = boxVolume box2 * box2.clayCapacity) :
  box2.clayCapacity = 168 := by
  sorry


end NUMINAMATH_CALUDE_second_box_clay_capacity_l2079_207982


namespace NUMINAMATH_CALUDE_identify_geometric_bodies_l2079_207903

/-- Represents the possible geometric bodies --/
inductive GeometricBody
  | TriangularPrism
  | QuadrangularPyramid
  | Cone
  | Frustum
  | TriangularFrustum
  | TriangularPyramid

/-- Represents a view of a geometric body --/
structure View where
  -- We'll assume some properties of the view, but won't define them explicitly
  dummy : Unit

/-- A function that determines if a set of views corresponds to a specific geometric body --/
def viewsMatchBody (views : List View) (body : GeometricBody) : Bool :=
  sorry -- The actual implementation would depend on how we define views

/-- The theorem stating that given the correct views, we can identify the four bodies --/
theorem identify_geometric_bodies 
  (views1 views2 views3 views4 : List View) 
  (h1 : viewsMatchBody views1 GeometricBody.TriangularPrism)
  (h2 : viewsMatchBody views2 GeometricBody.QuadrangularPyramid)
  (h3 : viewsMatchBody views3 GeometricBody.Cone)
  (h4 : viewsMatchBody views4 GeometricBody.Frustum) :
  ∃ (bodies : List GeometricBody), 
    bodies = [GeometricBody.TriangularPrism, 
              GeometricBody.QuadrangularPyramid, 
              GeometricBody.Cone, 
              GeometricBody.Frustum] ∧
    (∀ (views : List View), 
      views ∈ [views1, views2, views3, views4] → 
      ∃ (body : GeometricBody), body ∈ bodies ∧ viewsMatchBody views body) :=
by
  sorry


end NUMINAMATH_CALUDE_identify_geometric_bodies_l2079_207903


namespace NUMINAMATH_CALUDE_barbaras_coin_collection_l2079_207991

/-- The total number of coins Barbara has -/
def total_coins : ℕ := 18

/-- The number of type A coins Barbara has -/
def type_A_coins : ℕ := 12

/-- The value of 8 type A coins in dollars -/
def value_8_type_A : ℕ := 24

/-- The value of 6 type B coins in dollars -/
def value_6_type_B : ℕ := 21

/-- The total worth of Barbara's entire collection in dollars -/
def total_worth : ℕ := 57

theorem barbaras_coin_collection :
  total_coins = type_A_coins + (total_coins - type_A_coins) ∧
  value_8_type_A / 8 * type_A_coins + value_6_type_B / 6 * (total_coins - type_A_coins) = total_worth :=
by sorry

end NUMINAMATH_CALUDE_barbaras_coin_collection_l2079_207991


namespace NUMINAMATH_CALUDE_root_equation_value_l2079_207922

theorem root_equation_value (m : ℝ) : m^2 + m - 1 = 0 → 2023 - m^2 - m = 2022 := by
  sorry

end NUMINAMATH_CALUDE_root_equation_value_l2079_207922


namespace NUMINAMATH_CALUDE_cone_base_diameter_l2079_207994

/-- A cone with surface area 3π and lateral surface unfolding to a semicircle has base diameter 2 -/
theorem cone_base_diameter (r : ℝ) (l : ℝ) : 
  r > 0 → l > 0 → 
  (π * r^2 + π * r * l = 3 * π) → 
  (π * l = 2 * π * r) → 
  (2 * r = 2) := by
  sorry

end NUMINAMATH_CALUDE_cone_base_diameter_l2079_207994


namespace NUMINAMATH_CALUDE_total_pages_is_281_l2079_207941

/-- Calculates the total number of pages read over two months given Janine's reading habits --/
def total_pages_read : ℕ :=
  let last_month_books := 3 + 2
  let last_month_pages := 3 * 12 + 2 * 15
  let this_month_books := 2 * last_month_books
  let this_month_pages := 1 * 20 + 4 * 25 + 2 * 30 + 1 * 35
  last_month_pages + this_month_pages

/-- Proves that the total number of pages read over two months is 281 --/
theorem total_pages_is_281 : total_pages_read = 281 := by
  sorry

end NUMINAMATH_CALUDE_total_pages_is_281_l2079_207941


namespace NUMINAMATH_CALUDE_beaus_sons_correct_number_of_sons_l2079_207930

theorem beaus_sons (sons_age_today : ℕ) (beaus_age_today : ℕ) : ℕ :=
  let sons_age_three_years_ago := sons_age_today - 3
  let beaus_age_three_years_ago := beaus_age_today - 3
  let num_sons := beaus_age_three_years_ago / sons_age_three_years_ago
  num_sons

theorem correct_number_of_sons : beaus_sons 16 42 = 3 := by
  sorry

end NUMINAMATH_CALUDE_beaus_sons_correct_number_of_sons_l2079_207930


namespace NUMINAMATH_CALUDE_participants_with_three_points_l2079_207973

/-- Represents the number of participants in a tennis tournament with a specific score -/
def participantsWithScore (n : ℕ) (k : ℕ) : ℕ := Nat.choose n k

/-- Represents the total number of participants in the tournament -/
def totalParticipants (n : ℕ) : ℕ := 2^n + 4

/-- Theorem stating the number of participants with exactly 3 points at the end of the tournament -/
theorem participants_with_three_points (n : ℕ) (h : n > 4) :
  ∃ (winner : ℕ), winner = participantsWithScore n 3 + 1 ∧
  winner ≤ totalParticipants n :=
by sorry

end NUMINAMATH_CALUDE_participants_with_three_points_l2079_207973


namespace NUMINAMATH_CALUDE_midpoint_triangle_area_for_specific_configuration_l2079_207959

/-- Configuration of three congruent circles -/
structure CircleConfiguration where
  radius : ℝ
  passes_through_centers : Prop

/-- Triangle formed by midpoints of arcs -/
structure MidpointTriangle where
  config : CircleConfiguration
  area : ℝ

/-- The main theorem -/
theorem midpoint_triangle_area_for_specific_configuration :
  ∀ (config : CircleConfiguration) (triangle : MidpointTriangle),
    config.radius = 2 ∧
    config.passes_through_centers ∧
    triangle.config = config →
    ∃ (a b : ℕ),
      triangle.area = Real.sqrt 3 ∧
      triangle.area = Real.sqrt a - b ∧
      100 * a + b = 300 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_triangle_area_for_specific_configuration_l2079_207959


namespace NUMINAMATH_CALUDE_triangle_perimeter_not_85_l2079_207975

theorem triangle_perimeter_not_85 (a b c : ℝ) : 
  a = 24 → b = 18 → a + b + c > a + b → a + c > b → b + c > a → a + b + c ≠ 85 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_not_85_l2079_207975


namespace NUMINAMATH_CALUDE_probability_of_event_A_l2079_207955

theorem probability_of_event_A (P_B P_AB P_AUB : ℝ) 
  (hB : P_B = 0.4)
  (hAB : P_AB = 0.25)
  (hAUB : P_AUB = 0.6)
  : ∃ P_A : ℝ, P_A = 0.45 ∧ P_AUB = P_A + P_B - P_AB :=
by
  sorry

end NUMINAMATH_CALUDE_probability_of_event_A_l2079_207955


namespace NUMINAMATH_CALUDE_point_on_extension_line_l2079_207986

/-- Given two points in a 2D plane and a third point on their extension line,
    prove that the third point has specific coordinates. -/
theorem point_on_extension_line (P₁ P₂ P : ℝ × ℝ) : 
  P₁ = (2, -1) →
  P₂ = (0, 5) →
  (∃ t : ℝ, t > 1 ∧ P = P₁ + t • (P₂ - P₁)) →
  ‖P - P₁‖ = 2 * ‖P - P₂‖ →
  P = (-2, 11) := by
  sorry


end NUMINAMATH_CALUDE_point_on_extension_line_l2079_207986


namespace NUMINAMATH_CALUDE_cyrus_additional_bites_l2079_207909

/-- The number of mosquito bites Cyrus initially counted on his arms and legs -/
def initial_bites : ℕ := 14

/-- The number of people in Cyrus's family, excluding Cyrus -/
def family_members : ℕ := 6

/-- The number of additional mosquito bites on Cyrus's body -/
def additional_bites : ℕ := 14

/-- The total number of mosquito bites Cyrus got -/
def total_cyrus_bites : ℕ := initial_bites + additional_bites

/-- The total number of mosquito bites Cyrus's family got -/
def family_bites : ℕ := total_cyrus_bites / 2

/-- The number of mosquito bites each family member got -/
def bites_per_family_member : ℚ := family_bites / family_members

theorem cyrus_additional_bites :
  bites_per_family_member = additional_bites / family_members :=
by sorry

end NUMINAMATH_CALUDE_cyrus_additional_bites_l2079_207909


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l2079_207966

theorem arithmetic_calculation : 4 * 6 * 8 - 10 / 2 = 187 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l2079_207966


namespace NUMINAMATH_CALUDE_min_abs_sum_with_constraints_l2079_207950

theorem min_abs_sum_with_constraints (α β γ : ℝ) 
  (sum_constraint : α + β + γ = 2)
  (product_constraint : α * β * γ = 4) :
  ∃ v : ℝ, v = 6 ∧ ∀ α' β' γ' : ℝ, 
    α' + β' + γ' = 2 → α' * β' * γ' = 4 → 
    |α'| + |β'| + |γ'| ≥ v :=
sorry

end NUMINAMATH_CALUDE_min_abs_sum_with_constraints_l2079_207950


namespace NUMINAMATH_CALUDE_employment_calculation_l2079_207989

/-- The percentage of employed people in the population of town X -/
def employed_percentage : ℝ := sorry

/-- The percentage of the population that are employed males -/
def employed_males_percentage : ℝ := 15

/-- The percentage of employed people who are females -/
def employed_females_percentage : ℝ := 75

theorem employment_calculation :
  employed_percentage = 60 :=
by
  sorry

end NUMINAMATH_CALUDE_employment_calculation_l2079_207989


namespace NUMINAMATH_CALUDE_regions_divisible_by_six_l2079_207938

/-- Represents a triangle with sides divided into congruent segments --/
structure DividedTriangle where
  segments : ℕ
  segments_pos : segments > 0

/-- Calculates the number of regions formed in a divided triangle --/
def num_regions (t : DividedTriangle) : ℕ :=
  t.segments^2 + (2*t.segments - 1) * (t.segments - 1) - r t
where
  /-- Number of points where three lines intersect (excluding vertices) --/
  r (t : DividedTriangle) : ℕ := sorry

/-- The main theorem stating that the number of regions is divisible by 6 --/
theorem regions_divisible_by_six (t : DividedTriangle) (h : t.segments = 2002) :
  6 ∣ num_regions t := by sorry

end NUMINAMATH_CALUDE_regions_divisible_by_six_l2079_207938


namespace NUMINAMATH_CALUDE_cake_price_calculation_l2079_207978

theorem cake_price_calculation (smoothie_price : ℝ) (smoothie_count : ℕ) (cake_count : ℕ) (total_revenue : ℝ) :
  smoothie_price = 3 →
  smoothie_count = 40 →
  cake_count = 18 →
  total_revenue = 156 →
  ∃ (cake_price : ℝ), cake_price = 2 ∧ smoothie_price * smoothie_count + cake_price * cake_count = total_revenue :=
by
  sorry

#check cake_price_calculation

end NUMINAMATH_CALUDE_cake_price_calculation_l2079_207978


namespace NUMINAMATH_CALUDE_smallest_w_l2079_207919

def is_factor (a b : ℕ) : Prop := ∃ k : ℕ, b = a * k

theorem smallest_w (w : ℕ) (hw : w > 0) 
  (h1 : is_factor (2^5) (936 * w))
  (h2 : is_factor (3^3) (936 * w))
  (h3 : is_factor (13^2) (936 * w)) :
  w ≥ 156 ∧ ∃ w', w' = 156 ∧ w' > 0 ∧ 
    is_factor (2^5) (936 * w') ∧ 
    is_factor (3^3) (936 * w') ∧ 
    is_factor (13^2) (936 * w') :=
sorry

end NUMINAMATH_CALUDE_smallest_w_l2079_207919


namespace NUMINAMATH_CALUDE_binary_to_quaternary_conversion_l2079_207972

theorem binary_to_quaternary_conversion : 
  (1 * 2^4 + 1 * 2^3 + 1 * 2^2 + 0 * 2^1 + 0 * 2^0) = 
  (1 * 4^2 + 3 * 4^1 + 0 * 4^0) := by
  sorry

end NUMINAMATH_CALUDE_binary_to_quaternary_conversion_l2079_207972


namespace NUMINAMATH_CALUDE_pipe_A_rate_l2079_207997

/-- The rate at which Pipe A fills the tank -/
def rate_A : ℝ := 40

/-- The capacity of the tank in liters -/
def tank_capacity : ℝ := 800

/-- The rate at which Pipe B fills the tank in liters per minute -/
def rate_B : ℝ := 30

/-- The rate at which Pipe C drains the tank in liters per minute -/
def rate_C : ℝ := 20

/-- The time in minutes it takes to fill the tank -/
def fill_time : ℝ := 48

/-- The duration of one cycle in minutes -/
def cycle_duration : ℝ := 3

theorem pipe_A_rate : 
  rate_A = 40 ∧ 
  (fill_time / cycle_duration) * (rate_A + rate_B - rate_C) = tank_capacity := by
  sorry

end NUMINAMATH_CALUDE_pipe_A_rate_l2079_207997


namespace NUMINAMATH_CALUDE_existence_of_1000_consecutive_with_five_primes_l2079_207916

theorem existence_of_1000_consecutive_with_five_primes :
  (∃ n : ℕ, ∀ k ∈ Finset.range 1000, ¬ Nat.Prime (n + k + 2)) →
  (∃ m : ℕ, (Finset.filter (λ k => Nat.Prime (m + k + 1)) (Finset.range 1000)).card = 5) :=
by sorry

end NUMINAMATH_CALUDE_existence_of_1000_consecutive_with_five_primes_l2079_207916


namespace NUMINAMATH_CALUDE_rectangle_to_square_l2079_207963

theorem rectangle_to_square (k : ℕ) (h1 : k > 7) :
  (∃ (n : ℕ), k * (k - 7) = n^2) → (∃ (n : ℕ), k * (k - 7) = n^2 ∧ n = 24) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_to_square_l2079_207963


namespace NUMINAMATH_CALUDE_smallest_sum_of_reciprocals_l2079_207934

theorem smallest_sum_of_reciprocals (x y : ℕ+) (h1 : x ≠ y) (h2 : (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 12) :
  ∃ (a b : ℕ+), a ≠ b ∧ (1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / 12 ∧ a + b = 49 ∧ ∀ (c d : ℕ+), c ≠ d → (1 : ℚ) / c + (1 : ℚ) / d = (1 : ℚ) / 12 → c + d ≥ 49 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_of_reciprocals_l2079_207934


namespace NUMINAMATH_CALUDE_quadratic_inequality_empty_set_l2079_207906

theorem quadratic_inequality_empty_set (a : ℝ) : 
  (∀ x : ℝ, a * x^2 - a * x + 1 ≥ 0) ↔ 0 ≤ a ∧ a ≤ 4 := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_empty_set_l2079_207906


namespace NUMINAMATH_CALUDE_quadratic_root_in_unit_interval_l2079_207932

theorem quadratic_root_in_unit_interval (a b c : ℝ) (h : 2*a + 3*b + 6*c = 0) :
  ∃ x : ℝ, 0 < x ∧ x < 1 ∧ a*x^2 + b*x + c = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_in_unit_interval_l2079_207932


namespace NUMINAMATH_CALUDE_triangle_area_increase_l2079_207905

theorem triangle_area_increase (a b θ : ℝ) (ha : a > 0) (hb : b > 0) (hθ : 0 < θ ∧ θ < π) :
  let original_area := (1/2) * a * b * Real.sin θ
  let new_area := (1/2) * (3*a) * (2*b) * Real.sin θ
  new_area = 6 * original_area := by
sorry

end NUMINAMATH_CALUDE_triangle_area_increase_l2079_207905


namespace NUMINAMATH_CALUDE_solve_business_partnership_l2079_207918

/-- Represents the problem of determining when Hari joined Praveen's business --/
def business_partnership_problem (praveen_investment : ℕ) (hari_investment : ℕ) (profit_ratio_praveen : ℕ) (profit_ratio_hari : ℕ) (total_months : ℕ) : Prop :=
  ∃ (x : ℕ), 
    x ≤ total_months ∧
    (praveen_investment * total_months) * profit_ratio_hari = 
    (hari_investment * (total_months - x)) * profit_ratio_praveen

/-- Theorem stating the solution to the business partnership problem --/
theorem solve_business_partnership : 
  business_partnership_problem 3360 8640 2 3 12 → 
  ∃ (x : ℕ), x = 5 := by
  sorry

end NUMINAMATH_CALUDE_solve_business_partnership_l2079_207918


namespace NUMINAMATH_CALUDE_initial_amount_satisfies_equation_l2079_207913

/-- The initial amount of money the man has --/
def initial_amount : ℝ := 6.25

/-- The amount spent at each shop --/
def amount_spent : ℝ := 10

/-- The equation representing the man's transactions --/
def transaction_equation (x : ℝ) : Prop :=
  2 * (2 * (2 * x - amount_spent) - amount_spent) - amount_spent = 0

/-- Theorem stating that the initial amount satisfies the transaction equation --/
theorem initial_amount_satisfies_equation : 
  transaction_equation initial_amount := by sorry

end NUMINAMATH_CALUDE_initial_amount_satisfies_equation_l2079_207913


namespace NUMINAMATH_CALUDE_carla_marbles_l2079_207900

/-- The number of marbles Carla bought -/
def marbles_bought : ℕ := 134

/-- The number of marbles Carla has now -/
def marbles_now : ℕ := 187

/-- The number of marbles Carla started with -/
def marbles_start : ℕ := marbles_now - marbles_bought

theorem carla_marbles : marbles_start = 53 := by
  sorry

end NUMINAMATH_CALUDE_carla_marbles_l2079_207900


namespace NUMINAMATH_CALUDE_divisibility_and_primality_l2079_207985

def ten_eights_base_nine : ℕ := 8 * 9^9 + 8 * 9^8 + 8 * 9^7 + 8 * 9^6 + 8 * 9^5 + 8 * 9^4 + 8 * 9^3 + 8 * 9^2 + 8 * 9^1 + 8 * 9^0

def twelve_eights_base_nine : ℕ := 8 * 9^11 + 8 * 9^10 + 8 * 9^9 + 8 * 9^8 + 8 * 9^7 + 8 * 9^6 + 8 * 9^5 + 8 * 9^4 + 8 * 9^3 + 8 * 9^2 + 8 * 9^1 + 8 * 9^0

def divisor1 : ℕ := 9^4 - 9^3 + 9^2 - 9 + 1
def divisor2 : ℕ := 9^4 - 9^2 + 1

theorem divisibility_and_primality :
  (ten_eights_base_nine % divisor1 = 0) ∧
  (twelve_eights_base_nine % divisor2 = 0) ∧
  (¬ Nat.Prime divisor1) ∧
  (Nat.Prime divisor2) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_and_primality_l2079_207985


namespace NUMINAMATH_CALUDE_nilpotent_is_zero_fourth_power_eq_self_l2079_207995

class SpecialRing (A : Type*) extends Ring A where
  special_property : ∀ x : A, x + x^2 + x^3 = x^4 + x^5 + x^6

variable {A : Type*} [SpecialRing A]

theorem nilpotent_is_zero (x : A) (n : ℕ) (hn : n ≥ 2) (hx : x^n = 0) : x = 0 := by
  sorry

theorem fourth_power_eq_self (x : A) : x^4 = x := by
  sorry

end NUMINAMATH_CALUDE_nilpotent_is_zero_fourth_power_eq_self_l2079_207995


namespace NUMINAMATH_CALUDE_cookie_price_calculation_l2079_207998

/-- Represents a neighborhood with homes and boxes sold per home -/
structure Neighborhood where
  homes : ℕ
  boxesPerHome : ℕ

/-- Calculates the total boxes sold in a neighborhood -/
def totalBoxesSold (n : Neighborhood) : ℕ :=
  n.homes * n.boxesPerHome

/-- The price per box of cookies -/
def pricePerBox : ℚ := 2

theorem cookie_price_calculation 
  (neighborhoodA neighborhoodB : Neighborhood)
  (hA : neighborhoodA = ⟨10, 2⟩)
  (hB : neighborhoodB = ⟨5, 5⟩)
  (hRevenue : 50 = pricePerBox * max (totalBoxesSold neighborhoodA) (totalBoxesSold neighborhoodB)) :
  pricePerBox = 2 := by
sorry

end NUMINAMATH_CALUDE_cookie_price_calculation_l2079_207998


namespace NUMINAMATH_CALUDE_parallel_planes_normal_vectors_l2079_207961

/-- Given two planes α and β with normal vectors, prove that if they are parallel, then k = 4 -/
theorem parallel_planes_normal_vectors (k : ℝ) : 
  let n_alpha : ℝ × ℝ × ℝ := (1, 2, -2)
  let n_beta : ℝ × ℝ × ℝ := (-2, -4, k)
  (∃ (c : ℝ), c ≠ 0 ∧ n_alpha = c • n_beta) → k = 4 := by
  sorry

end NUMINAMATH_CALUDE_parallel_planes_normal_vectors_l2079_207961


namespace NUMINAMATH_CALUDE_common_divisors_product_l2079_207943

theorem common_divisors_product (list : List Int) : 
  list = [48, 64, -18, 162, 144] →
  ∃ (a b c d e : Nat), 
    a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧
    (∀ x ∈ list, a ∣ x.natAbs) ∧
    (∀ x ∈ list, b ∣ x.natAbs) ∧
    (∀ x ∈ list, c ∣ x.natAbs) ∧
    (∀ x ∈ list, d ∣ x.natAbs) ∧
    (∀ x ∈ list, e ∣ x.natAbs) ∧
    a * b * c * d * e = 108 :=
by sorry

end NUMINAMATH_CALUDE_common_divisors_product_l2079_207943


namespace NUMINAMATH_CALUDE_digit_57_is_5_l2079_207962

/-- The decimal expansion of 21/22 has a repeating pattern of "54" -/
def repeating_pattern : ℕ → ℕ
  | n => if n % 2 = 0 then 4 else 5

/-- The 57th digit after the decimal point in the expansion of 21/22 -/
def digit_57 : ℕ := repeating_pattern 56

theorem digit_57_is_5 : digit_57 = 5 := by
  sorry

end NUMINAMATH_CALUDE_digit_57_is_5_l2079_207962


namespace NUMINAMATH_CALUDE_lakeside_club_overlap_l2079_207960

/-- The number of students in both the theater and robotics clubs at Lakeside High School -/
def students_in_both_clubs (total_students theater_members robotics_members either_or_both : ℕ) : ℕ :=
  theater_members + robotics_members - either_or_both

/-- Theorem: Given the conditions from Lakeside High School, 
    the number of students in both the theater and robotics clubs is 25 -/
theorem lakeside_club_overlap : 
  let total_students : ℕ := 250
  let theater_members : ℕ := 85
  let robotics_members : ℕ := 120
  let either_or_both : ℕ := 180
  students_in_both_clubs total_students theater_members robotics_members either_or_both = 25 := by
  sorry

end NUMINAMATH_CALUDE_lakeside_club_overlap_l2079_207960


namespace NUMINAMATH_CALUDE_calculate_swimming_speed_triathlete_swimming_speed_l2079_207947

/-- Calculates the swimming speed given the total distance, running speed, and average speed -/
theorem calculate_swimming_speed 
  (total_distance : ℝ) 
  (running_distance : ℝ) 
  (running_speed : ℝ) 
  (average_speed : ℝ) : ℝ :=
  let swimming_distance := total_distance - running_distance
  let total_time := total_distance / average_speed
  let running_time := running_distance / running_speed
  let swimming_time := total_time - running_time
  swimming_distance / swimming_time

/-- Proves that the swimming speed is 6 miles per hour given the problem conditions -/
theorem triathlete_swimming_speed : 
  calculate_swimming_speed 6 3 10 7.5 = 6 := by
  sorry

end NUMINAMATH_CALUDE_calculate_swimming_speed_triathlete_swimming_speed_l2079_207947


namespace NUMINAMATH_CALUDE_sector_central_angle_l2079_207939

/-- Given a sector with perimeter 4 and area 1, its central angle is 2 radians -/
theorem sector_central_angle (r l : ℝ) (h1 : l + 2*r = 4) (h2 : (1/2)*l*r = 1) :
  l / r = 2 := by sorry

end NUMINAMATH_CALUDE_sector_central_angle_l2079_207939


namespace NUMINAMATH_CALUDE_gcd_of_three_numbers_l2079_207945

theorem gcd_of_three_numbers : Nat.gcd 17420 (Nat.gcd 23826 36654) = 2 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_three_numbers_l2079_207945


namespace NUMINAMATH_CALUDE_equation_system_solution_l2079_207921

theorem equation_system_solution (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (eq1 : 1 / (x * y) = x / z + 1)
  (eq2 : 1 / (y * z) = y / x + 1)
  (eq3 : 1 / (z * x) = z / y + 1) :
  x = 1 / Real.sqrt 2 ∧ y = 1 / Real.sqrt 2 ∧ z = 1 / Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_equation_system_solution_l2079_207921


namespace NUMINAMATH_CALUDE_group_purchase_equation_l2079_207970

theorem group_purchase_equation (x : ℕ) : 
  (∀ (required : ℕ), 8 * x = required + 3 ∧ 7 * x = required - 4) → 
  8 * x - 3 = 7 * x + 4 := by
sorry

end NUMINAMATH_CALUDE_group_purchase_equation_l2079_207970


namespace NUMINAMATH_CALUDE_min_value_expression_l2079_207983

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  b / (3 * a) + 3 / b ≥ 5 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + b₀ = 1 ∧ b₀ / (3 * a₀) + 3 / b₀ = 5 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l2079_207983


namespace NUMINAMATH_CALUDE_number_exceeding_fraction_l2079_207908

theorem number_exceeding_fraction : 
  ∀ x : ℚ, x = (3/8) * x + 20 → x = 32 := by
sorry

end NUMINAMATH_CALUDE_number_exceeding_fraction_l2079_207908


namespace NUMINAMATH_CALUDE_triangle_acute_from_sine_ratio_l2079_207951

theorem triangle_acute_from_sine_ratio (A B C : ℝ) (h_triangle : A + B + C = Real.pi)
  (h_positive : 0 < A ∧ 0 < B ∧ 0 < C) (h_sine_ratio : ∃ k : ℝ, k > 0 ∧ Real.sin A = 5 * k ∧ Real.sin B = 11 * k ∧ Real.sin C = 13 * k) :
  A < Real.pi / 2 ∧ B < Real.pi / 2 ∧ C < Real.pi / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_acute_from_sine_ratio_l2079_207951


namespace NUMINAMATH_CALUDE_lcm_16_24_45_l2079_207953

theorem lcm_16_24_45 : Nat.lcm (Nat.lcm 16 24) 45 = 720 := by
  sorry

end NUMINAMATH_CALUDE_lcm_16_24_45_l2079_207953
