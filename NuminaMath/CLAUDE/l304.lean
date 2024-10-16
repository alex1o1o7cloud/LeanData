import Mathlib

namespace NUMINAMATH_CALUDE_fifth_minus_fourth_rectangles_l304_30422

def rectangle_tiles (n : ℕ) : ℕ := (2 * n - 1) ^ 2

theorem fifth_minus_fourth_rectangles : rectangle_tiles 5 - rectangle_tiles 4 = 32 := by
  sorry

end NUMINAMATH_CALUDE_fifth_minus_fourth_rectangles_l304_30422


namespace NUMINAMATH_CALUDE_sum_of_integers_l304_30409

theorem sum_of_integers (x y : ℕ) (h1 : x > y) (h2 : x - y = 8) (h3 : x * y = 240) : x + y = 32 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_integers_l304_30409


namespace NUMINAMATH_CALUDE_watermelon_puree_volume_watermelon_puree_volume_proof_l304_30474

/-- Given the conditions of Carla's smoothie recipe, prove that she uses 500 ml of watermelon puree. -/
theorem watermelon_puree_volume : ℝ → Prop :=
  fun watermelon_puree : ℝ =>
    let total_volume : ℝ := 4 * 150
    let cream_volume : ℝ := 100
    (total_volume = watermelon_puree + cream_volume) → (watermelon_puree = 500)

/-- Proof of the watermelon puree volume theorem -/
theorem watermelon_puree_volume_proof : watermelon_puree_volume 500 := by
  sorry

#check watermelon_puree_volume
#check watermelon_puree_volume_proof

end NUMINAMATH_CALUDE_watermelon_puree_volume_watermelon_puree_volume_proof_l304_30474


namespace NUMINAMATH_CALUDE_star_two_three_l304_30493

/-- The star operation defined as a * b = a * b^3 - 2 * b + 2 -/
def star (a b : ℝ) : ℝ := a * b^3 - 2 * b + 2

/-- Theorem: The value of 2 ★ 3 is 50 -/
theorem star_two_three : star 2 3 = 50 := by
  sorry

end NUMINAMATH_CALUDE_star_two_three_l304_30493


namespace NUMINAMATH_CALUDE_coefficient_x_squared_is_120_l304_30434

/-- The coefficient of x^2 in the expansion of (1+x)^2 + (1+x)^3 + ... + (1+x)^9 -/
def coefficient_x_squared : ℕ :=
  (Finset.range 8).sum (λ n => Nat.choose (n + 2) 2)

/-- Theorem stating that the coefficient of x^2 in the expansion is 120 -/
theorem coefficient_x_squared_is_120 : coefficient_x_squared = 120 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x_squared_is_120_l304_30434


namespace NUMINAMATH_CALUDE_crab_fishing_income_l304_30496

theorem crab_fishing_income 
  (num_buckets : ℕ) 
  (crabs_per_bucket : ℕ) 
  (price_per_crab : ℕ) 
  (days_per_week : ℕ) 
  (h1 : num_buckets = 8) 
  (h2 : crabs_per_bucket = 12) 
  (h3 : price_per_crab = 5) 
  (h4 : days_per_week = 7) : 
  num_buckets * crabs_per_bucket * price_per_crab * days_per_week = 3360 := by
sorry

end NUMINAMATH_CALUDE_crab_fishing_income_l304_30496


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l304_30488

theorem quadratic_equation_roots (k : ℝ) : 
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
  (x₁^2 - (k-3)*x₁ - k + 1 = 0) ∧ 
  (x₂^2 - (k-3)*x₂ - k + 1 = 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l304_30488


namespace NUMINAMATH_CALUDE_kylie_jewelry_beads_l304_30438

/-- The number of beads Kylie uses in total to make her jewelry over the week -/
def total_beads : ℕ :=
  let necklace_beads := 20
  let bracelet_beads := 10
  let earring_beads := 5
  let anklet_beads := 8
  let ring_beads := 7
  let monday_necklaces := 10
  let tuesday_necklaces := 2
  let wednesday_bracelets := 5
  let thursday_earrings := 3
  let friday_anklets := 4
  let friday_rings := 6
  (necklace_beads * (monday_necklaces + tuesday_necklaces)) +
  (bracelet_beads * wednesday_bracelets) +
  (earring_beads * thursday_earrings) +
  (anklet_beads * friday_anklets) +
  (ring_beads * friday_rings)

theorem kylie_jewelry_beads : total_beads = 379 := by
  sorry

end NUMINAMATH_CALUDE_kylie_jewelry_beads_l304_30438


namespace NUMINAMATH_CALUDE_max_area_triangle_l304_30481

/-- Given points A and B, and a circle with two symmetric points M and N,
    prove that the maximum area of triangle PAB is 3 + √2 --/
theorem max_area_triangle (k : ℝ) : 
  let A : ℝ × ℝ := (-2, 0)
  let B : ℝ × ℝ := (0, 2)
  let circle := {(x, y) : ℝ × ℝ | x^2 + y^2 + k*x = 0}
  let symmetry_line := {(x, y) : ℝ × ℝ | x - y - 1 = 0}
  ∃ (M N : ℝ × ℝ), M ∈ circle ∧ N ∈ circle ∧ M ≠ N ∧
    (∃ (c : ℝ × ℝ), c ∈ symmetry_line ∧ 
      (M.1 - c.1)^2 + (M.2 - c.2)^2 = (N.1 - c.1)^2 + (N.2 - c.2)^2) →
  (⨆ (P : ℝ × ℝ) (h : P ∈ circle), 
    abs ((P.1 - A.1) * (B.2 - A.2) - (P.2 - A.2) * (B.1 - A.1)) / 2) = 3 + Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_max_area_triangle_l304_30481


namespace NUMINAMATH_CALUDE_vasilyev_car_loan_payment_l304_30408

/-- Calculates the maximum monthly car loan payment for the Vasilyev family --/
def max_car_loan_payment (total_income : ℝ) (total_expenses : ℝ) (emergency_fund_rate : ℝ) : ℝ :=
  let remaining_income := total_income - total_expenses
  let emergency_fund := emergency_fund_rate * remaining_income
  total_income - total_expenses - emergency_fund

/-- Theorem stating the maximum monthly car loan payment for the Vasilyev family --/
theorem vasilyev_car_loan_payment :
  max_car_loan_payment 84600 49800 0.1 = 31320 := by
  sorry

end NUMINAMATH_CALUDE_vasilyev_car_loan_payment_l304_30408


namespace NUMINAMATH_CALUDE_first_day_over_500_l304_30407

def marbles (k : ℕ) : ℕ := 5 * 3^k

theorem first_day_over_500 : (∃ k : ℕ, marbles k > 500) ∧ 
  (∀ j : ℕ, j < 5 → marbles j ≤ 500) ∧ 
  marbles 5 > 500 := by
  sorry

end NUMINAMATH_CALUDE_first_day_over_500_l304_30407


namespace NUMINAMATH_CALUDE_dave_tickets_left_l304_30433

/-- Given that Dave had 13 tickets initially and used 6 tickets,
    prove that the number of tickets he has left is 7. -/
theorem dave_tickets_left (initial_tickets : ℕ) (used_tickets : ℕ) 
  (h1 : initial_tickets = 13) (h2 : used_tickets = 6) : 
  initial_tickets - used_tickets = 7 := by
  sorry

end NUMINAMATH_CALUDE_dave_tickets_left_l304_30433


namespace NUMINAMATH_CALUDE_intersection_of_three_lines_l304_30443

/-- Given three lines that intersect at a single point, prove the value of k -/
theorem intersection_of_three_lines (k : ℚ) : 
  (∃! p : ℚ × ℚ, 
    (p.2 = 4 * p.1 + 2) ∧ 
    (p.2 = -2 * p.1 - 8) ∧ 
    (p.2 = 2 * p.1 + k)) → 
  k = -4/3 := by
sorry

end NUMINAMATH_CALUDE_intersection_of_three_lines_l304_30443


namespace NUMINAMATH_CALUDE_flags_left_proof_l304_30446

/-- Calculates the number of flags left after installation -/
def flags_left (circumference : ℕ) (interval : ℕ) (available_flags : ℕ) : ℕ :=
  available_flags - (circumference / interval)

/-- Theorem: Given the specific conditions, the number of flags left is 2 -/
theorem flags_left_proof :
  flags_left 200 20 12 = 2 := by
  sorry

end NUMINAMATH_CALUDE_flags_left_proof_l304_30446


namespace NUMINAMATH_CALUDE_tangent_perpendicular_max_derivative_decreasing_function_range_l304_30442

noncomputable section

variables (a : ℝ)

def f (x : ℝ) : ℝ := a * x^2 - Real.exp x

def f_derivative (x : ℝ) : ℝ := 2 * a * x - Real.exp x

theorem tangent_perpendicular_max_derivative :
  f_derivative a 1 = 0 →
  ∀ x, f_derivative a x ≤ 0 :=
sorry

theorem decreasing_function_range :
  (∀ x₁ x₂, 0 ≤ x₁ → x₁ < x₂ → 
    f a x₂ + x₂ * (2 - 2 * Real.log 2) < f a x₁ + x₁ * (2 - 2 * Real.log 2)) →
  a ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_tangent_perpendicular_max_derivative_decreasing_function_range_l304_30442


namespace NUMINAMATH_CALUDE_fixed_charge_is_45_l304_30466

/-- Represents Chris's internet bill structure and usage -/
structure InternetBill where
  fixed_charge : ℝ  -- Fixed monthly charge for 100 GB
  over_charge_per_gb : ℝ  -- Charge per GB over 100 GB limit
  total_bill : ℝ  -- Total bill amount
  gb_over_limit : ℝ  -- Number of GB over the 100 GB limit

/-- Theorem stating that given the conditions, the fixed monthly charge is $45 -/
theorem fixed_charge_is_45 (bill : InternetBill) 
  (h1 : bill.over_charge_per_gb = 0.25)
  (h2 : bill.total_bill = 65)
  (h3 : bill.gb_over_limit = 80) : 
  bill.fixed_charge = 45 := by
  sorry

#check fixed_charge_is_45

end NUMINAMATH_CALUDE_fixed_charge_is_45_l304_30466


namespace NUMINAMATH_CALUDE_negative_one_minus_two_times_negative_two_l304_30445

theorem negative_one_minus_two_times_negative_two : -1 - 2 * (-2) = 3 := by
  sorry

end NUMINAMATH_CALUDE_negative_one_minus_two_times_negative_two_l304_30445


namespace NUMINAMATH_CALUDE_mike_remaining_cards_l304_30497

/-- Given Mike's initial number of baseball cards and the number of cards sold to Sam and Alex,
    calculate the number of cards Mike has left. -/
def remaining_cards (initial : ℕ) (sold_to_sam : ℕ) (sold_to_alex : ℕ) : ℕ :=
  initial - (sold_to_sam + sold_to_alex)

/-- Theorem stating that Mike has 59 baseball cards left after selling to Sam and Alex. -/
theorem mike_remaining_cards :
  remaining_cards 87 13 15 = 59 := by
  sorry

#eval remaining_cards 87 13 15

end NUMINAMATH_CALUDE_mike_remaining_cards_l304_30497


namespace NUMINAMATH_CALUDE_coin_bill_combinations_l304_30455

theorem coin_bill_combinations : 
  ∃! n : ℕ, n = (Finset.filter (fun p : ℕ × ℕ => 2 * p.1 + 5 * p.2 = 207) (Finset.product (Finset.range 104) (Finset.range 42))).card :=
by
  sorry

end NUMINAMATH_CALUDE_coin_bill_combinations_l304_30455


namespace NUMINAMATH_CALUDE_greatest_partition_size_l304_30453

theorem greatest_partition_size (m n p : ℕ) (h_m : m > 0) (h_n : n > 0) (h_p : Nat.Prime p) :
  ∃ (s : ℕ), s > 0 ∧ s ≤ m ∧
  ∀ (t : ℕ), t > s →
    ¬∃ (partition : Fin (t * n * p) → Fin t),
      ∀ (i : Fin t),
        ∃ (r : ℕ),
          ∀ (j k : Fin (t * n * p)),
            partition j = i → partition k = i →
              (j.val + k.val) % p = r :=
by sorry

end NUMINAMATH_CALUDE_greatest_partition_size_l304_30453


namespace NUMINAMATH_CALUDE_trees_in_gray_areas_trees_in_gray_areas_proof_l304_30462

/-- Given three pictures with an equal number of trees, where the white areas
contain 82, 82, and 100 trees respectively, the total number of trees in the
gray areas is 26. -/
theorem trees_in_gray_areas : ℕ → ℕ → ℕ → Prop :=
  fun (total : ℕ) (x : ℕ) (y : ℕ) =>
    (total = 82 + x) ∧
    (total = 82 + y) ∧
    (total = 100) →
    x + y = 26

/-- Proof of the theorem -/
theorem trees_in_gray_areas_proof : ∃ (total : ℕ), trees_in_gray_areas total 18 8 := by
  sorry

end NUMINAMATH_CALUDE_trees_in_gray_areas_trees_in_gray_areas_proof_l304_30462


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l304_30470

theorem quadratic_inequality_solution (a b k : ℝ) : 
  (∀ x, a * x^2 - 3 * x + 2 > 0 ↔ (x < 1 ∨ x > b)) →
  b > 1 →
  (∀ x y, x > 0 → y > 0 → a / x + b / y = 1 → 2 * x + y ≥ k^2 + k + 2) →
  a = 1 ∧ b = 2 ∧ -3 ≤ k ∧ k ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l304_30470


namespace NUMINAMATH_CALUDE_max_gcd_value_l304_30411

def a (n : ℕ+) : ℕ := 121 + n^2

def d (n : ℕ+) : ℕ := Nat.gcd (a n) (a (n + 1))

theorem max_gcd_value :
  (∃ (k : ℕ+), d k = 99) ∧ (∀ (n : ℕ+), d n ≤ 99) := by
  sorry

end NUMINAMATH_CALUDE_max_gcd_value_l304_30411


namespace NUMINAMATH_CALUDE_smallest_stair_count_l304_30418

theorem smallest_stair_count : ∃ n : ℕ, n = 71 ∧ n > 15 ∧ 
  n % 3 = 2 ∧ n % 7 = 1 ∧ n % 4 = 3 ∧
  ∀ m : ℕ, m > 15 → m % 3 = 2 → m % 7 = 1 → m % 4 = 3 → m ≥ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_stair_count_l304_30418


namespace NUMINAMATH_CALUDE_jude_matchbox_vehicles_l304_30431

/-- Calculates the total number of matchbox vehicles Jude buys given the specified conditions -/
theorem jude_matchbox_vehicles :
  let car_cost : ℕ := 10
  let truck_cost : ℕ := 15
  let helicopter_cost : ℕ := 20
  let total_caps : ℕ := 250
  let trucks_bought : ℕ := 5
  let caps_spent_on_trucks : ℕ := trucks_bought * truck_cost
  let remaining_caps : ℕ := total_caps - caps_spent_on_trucks
  let caps_for_cars : ℕ := (remaining_caps * 60) / 100
  let cars_bought : ℕ := caps_for_cars / car_cost
  let caps_left : ℕ := remaining_caps - (cars_bought * car_cost)
  let helicopters_bought : ℕ := caps_left / helicopter_cost
  trucks_bought + cars_bought + helicopters_bought = 18 :=
by sorry

end NUMINAMATH_CALUDE_jude_matchbox_vehicles_l304_30431


namespace NUMINAMATH_CALUDE_unique_solution_condition_l304_30478

theorem unique_solution_condition (c d : ℝ) :
  (∃! x : ℝ, 4 * x - 7 + c = d * x + 4) ↔ d ≠ 4 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l304_30478


namespace NUMINAMATH_CALUDE_min_value_3a_3b_l304_30454

theorem min_value_3a_3b (a b : ℝ) (h : a * b = 2) : 3 * a + 3 * b ≥ 6 := by
  sorry

end NUMINAMATH_CALUDE_min_value_3a_3b_l304_30454


namespace NUMINAMATH_CALUDE_goldfish_sales_l304_30417

theorem goldfish_sales (buy_price sell_price tank_cost shortfall_percent : ℚ) 
  (h1 : buy_price = 25 / 100)
  (h2 : sell_price = 75 / 100)
  (h3 : tank_cost = 100)
  (h4 : shortfall_percent = 45 / 100) :
  (tank_cost * (1 - shortfall_percent)) / (sell_price - buy_price) = 110 := by
sorry

end NUMINAMATH_CALUDE_goldfish_sales_l304_30417


namespace NUMINAMATH_CALUDE_remainder_of_87_pow_88_plus_7_mod_88_l304_30430

theorem remainder_of_87_pow_88_plus_7_mod_88 : 87^88 + 7 ≡ 8 [MOD 88] := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_87_pow_88_plus_7_mod_88_l304_30430


namespace NUMINAMATH_CALUDE_line_parametrization_l304_30484

/-- The slope of the line -/
def m : ℚ := 3/4

/-- The y-intercept of the line -/
def b : ℚ := -5

/-- The x-coordinate of the point on the line -/
def x₀ : ℚ := -8

/-- The y-component of the direction vector -/
def v : ℚ := 7

/-- The equation of the line -/
def line_eq (x y : ℚ) : Prop := y = m * x + b

/-- The parametric form of the line -/
def parametric_eq (s l t x y : ℚ) : Prop :=
  x = x₀ + t * l ∧ y = s + t * v

theorem line_parametrization (s l : ℚ) :
  (∀ t x y, parametric_eq s l t x y → line_eq x y) →
  s = -11 ∧ l = 28/3 := by sorry

end NUMINAMATH_CALUDE_line_parametrization_l304_30484


namespace NUMINAMATH_CALUDE_intense_goblet_point_difference_l304_30413

/-- The number of teams in the tournament -/
def num_teams : ℕ := 10

/-- Points awarded for a win -/
def win_points : ℕ := 4

/-- Points awarded for a tie -/
def tie_points : ℕ := 2

/-- Points awarded for a loss -/
def loss_points : ℕ := 1

/-- The total number of games in a round-robin tournament -/
def total_games (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The maximum total points possible in the tournament -/
def max_total_points : ℕ := total_games num_teams * win_points

/-- The minimum total points possible in the tournament -/
def min_total_points : ℕ := num_teams * (num_teams - 1) * loss_points

theorem intense_goblet_point_difference :
  max_total_points - min_total_points = 90 := by
  sorry

end NUMINAMATH_CALUDE_intense_goblet_point_difference_l304_30413


namespace NUMINAMATH_CALUDE_function_characterization_l304_30491

def f (x a b : ℝ) : ℝ := (x + a) * (b * x + 2 * a)

theorem function_characterization (a b : ℝ) :
  (∀ x, f x a b = f (-x) a b) →  -- f is even
  (∀ y, y ∈ Set.Iic 4 → ∃ x, f x a b = y) →  -- range is (-∞, 4]
  (∀ y, ∃ x, f x a b = y → y ≤ 4) →  -- range is (-∞, 4]
  (∀ x, f x a b = -2 * x^2 + 4) :=
by sorry

end NUMINAMATH_CALUDE_function_characterization_l304_30491


namespace NUMINAMATH_CALUDE_suzie_reading_rate_l304_30467

/-- The number of pages Liza reads in an hour -/
def liza_pages_per_hour : ℕ := 20

/-- The number of additional pages Liza reads compared to Suzie in 3 hours -/
def liza_additional_pages : ℕ := 15

/-- The number of hours considered -/
def hours : ℕ := 3

/-- The number of pages Suzie reads in an hour -/
def suzie_pages_per_hour : ℕ := 15

theorem suzie_reading_rate :
  suzie_pages_per_hour = (liza_pages_per_hour * hours - liza_additional_pages) / hours :=
by sorry

end NUMINAMATH_CALUDE_suzie_reading_rate_l304_30467


namespace NUMINAMATH_CALUDE_ndfl_calculation_l304_30468

/-- Calculates the total NDFL (personal income tax) on securities income -/
def calculate_ndfl (dividend_income : ℕ) (ofz_income : ℕ) (corporate_bond_income : ℕ) 
                   (shares_sold : ℕ) (sale_price_per_share : ℕ) (purchase_price_per_share : ℕ) : ℕ :=
  let dividend_tax := dividend_income * 13 / 100
  let corporate_bond_tax := corporate_bond_income * 13 / 100
  let capital_gain := shares_sold * (sale_price_per_share - purchase_price_per_share)
  let capital_gain_tax := capital_gain * 13 / 100
  dividend_tax + corporate_bond_tax + capital_gain_tax

/-- The total NDFL on securities income is 11,050 rubles -/
theorem ndfl_calculation : 
  calculate_ndfl 50000 40000 30000 100 200 150 = 11050 := by
  sorry

end NUMINAMATH_CALUDE_ndfl_calculation_l304_30468


namespace NUMINAMATH_CALUDE_unique_solution_range_l304_30490

theorem unique_solution_range (x a : ℝ) : 
  (∃! x, Real.log (4 * x^2 + 4 * a * x) - Real.log (4 * x - a + 1) = 0) ↔ 
  (1/5 ≤ a ∧ a < 1) :=
sorry

end NUMINAMATH_CALUDE_unique_solution_range_l304_30490


namespace NUMINAMATH_CALUDE_find_divisor_l304_30423

theorem find_divisor (dividend quotient remainder : ℕ) 
  (h1 : dividend = 166) 
  (h2 : quotient = 8) 
  (h3 : remainder = 6) 
  (h4 : dividend = quotient * (dividend / quotient) + remainder) : 
  dividend / quotient = 20 := by
sorry

end NUMINAMATH_CALUDE_find_divisor_l304_30423


namespace NUMINAMATH_CALUDE_tom_needs_163_blue_tickets_l304_30426

/-- Represents the number of tickets Tom has -/
structure Tickets :=
  (yellow : ℕ)
  (red : ℕ)
  (blue : ℕ)

/-- Calculates the total number of blue tickets equivalent to a given number of tickets -/
def blueEquivalent (t : Tickets) : ℕ :=
  t.yellow * 100 + t.red * 10 + t.blue

/-- The number of blue tickets needed to win a Bible -/
def bibleRequirement : ℕ := 1000

/-- Tom's current tickets -/
def tomsTickets : Tickets := ⟨8, 3, 7⟩

/-- Theorem stating how many more blue tickets Tom needs -/
theorem tom_needs_163_blue_tickets :
  bibleRequirement - blueEquivalent tomsTickets = 163 := by
  sorry

end NUMINAMATH_CALUDE_tom_needs_163_blue_tickets_l304_30426


namespace NUMINAMATH_CALUDE_fifth_largest_divisor_of_n_l304_30482

def n : ℕ := 2025000000

-- Define a function to get the kth largest divisor
def kth_largest_divisor (k : ℕ) (n : ℕ) : ℕ :=
  sorry

-- Theorem statement
theorem fifth_largest_divisor_of_n :
  kth_largest_divisor 5 n = 126562500 :=
sorry

end NUMINAMATH_CALUDE_fifth_largest_divisor_of_n_l304_30482


namespace NUMINAMATH_CALUDE_wire_service_reporters_l304_30472

theorem wire_service_reporters (total : ℝ) (h1 : total > 0) : 
  let local_politics := 0.35 * total
  let not_politics := 0.5 * total
  let politics := total - not_politics
  let not_local_politics := politics - local_politics
  (not_local_politics / politics) * 100 = 30 := by
sorry

end NUMINAMATH_CALUDE_wire_service_reporters_l304_30472


namespace NUMINAMATH_CALUDE_perfect_square_power_of_two_plus_33_l304_30479

theorem perfect_square_power_of_two_plus_33 :
  ∀ n : ℕ, (∃ m : ℕ, 2^n + 33 = m^2) ↔ n = 4 ∨ n = 8 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_power_of_two_plus_33_l304_30479


namespace NUMINAMATH_CALUDE_cubic_fraction_equals_fifteen_l304_30477

theorem cubic_fraction_equals_fifteen :
  let a : ℤ := 7
  let b : ℤ := 5
  let c : ℤ := 3
  (a^3 + b^3 + c^3) / (a^2 - a*b + b^2 - b*c + c^2) = 15 := by
  sorry

end NUMINAMATH_CALUDE_cubic_fraction_equals_fifteen_l304_30477


namespace NUMINAMATH_CALUDE_red_ball_probability_l304_30405

-- Define the containers and their contents
structure Container where
  red : ℕ
  green : ℕ

def containerA : Container := ⟨10, 5⟩
def containerB : Container := ⟨6, 6⟩
def containerC : Container := ⟨3, 9⟩
def containerD : Container := ⟨4, 8⟩

-- Define the list of containers
def containers : List Container := [containerA, containerB, containerC, containerD]

-- Function to calculate the probability of selecting a red ball from a container
def redProbability (c : Container) : ℚ :=
  c.red / (c.red + c.green)

-- Theorem stating the probability of selecting a red ball
theorem red_ball_probability :
  (1 / (containers.length : ℚ)) * (containers.map redProbability).sum = 25 / 48 := by
  sorry

end NUMINAMATH_CALUDE_red_ball_probability_l304_30405


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l304_30489

theorem quadratic_equation_roots (c : ℝ) : 
  c = 0 → ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + 2*x₁ + 4*c = 0 ∧ x₂^2 + 2*x₂ + 4*c = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l304_30489


namespace NUMINAMATH_CALUDE_parabola_unique_coefficients_l304_30483

/-- Represents a parabola of the form y = ax² + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The y-coordinate of a point on the parabola given its x-coordinate -/
def Parabola.y (p : Parabola) (x : ℝ) : ℝ :=
  p.a * x^2 + p.b * x + p.c

/-- The slope of the tangent line to the parabola at a given x-coordinate -/
def Parabola.slope (p : Parabola) (x : ℝ) : ℝ :=
  2 * p.a * x + p.b

theorem parabola_unique_coefficients (p : Parabola) :
  p.y 1 = 1 →
  p.y 2 = -1 →
  p.slope 2 = 1 →
  p.a = 3 ∧ p.b = -11 ∧ p.c = 9 := by
  sorry

end NUMINAMATH_CALUDE_parabola_unique_coefficients_l304_30483


namespace NUMINAMATH_CALUDE_inequality_solution_l304_30425

theorem inequality_solution : ∃! (x y z : ℤ),
  (1 / Real.sqrt (x - 2*y + z + 1 : ℝ) +
   2 / Real.sqrt (2*x - y + 3*z - 1 : ℝ) +
   3 / Real.sqrt (3*y - 3*x - 4*z + 3 : ℝ) >
   x^2 - 4*x + 3) ∧
  (x = 3 ∧ y = 1 ∧ z = -1) := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_l304_30425


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l304_30441

theorem sqrt_equation_solution (y : ℝ) : 
  Real.sqrt (1 + Real.sqrt (2 * y - 5)) = Real.sqrt 7 → y = 20.5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l304_30441


namespace NUMINAMATH_CALUDE_fake_coin_identification_l304_30463

/-- Represents a weighing strategy for identifying a fake coin. -/
structure WeighingStrategy where
  /-- The number of weighings performed. -/
  num_weighings : ℕ
  /-- The maximum number of times any single coin is weighed. -/
  max_weighs_per_coin : ℕ

/-- Represents the problem of identifying a fake coin among a set of coins. -/
structure FakeCoinProblem where
  /-- The total number of coins. -/
  total_coins : ℕ
  /-- The number of fake coins. -/
  num_fake_coins : ℕ
  /-- Indicates whether the fake coin is lighter than the genuine coins. -/
  fake_is_lighter : Bool

/-- Theorem stating that the fake coin can be identified within the given constraints. -/
theorem fake_coin_identification
  (problem : FakeCoinProblem)
  (strategy : WeighingStrategy) :
  problem.total_coins = 99 →
  problem.num_fake_coins = 1 →
  problem.fake_is_lighter = true →
  strategy.num_weighings ≤ 7 →
  strategy.max_weighs_per_coin ≤ 2 →
  ∃ (identification_method : Unit), True :=
by
  sorry

end NUMINAMATH_CALUDE_fake_coin_identification_l304_30463


namespace NUMINAMATH_CALUDE_connie_tickets_l304_30460

def ticket_distribution (total_tickets : ℕ) : Prop :=
  let koala := total_tickets * 20 / 100
  let earbuds := 30
  let car := earbuds * 2
  let bracelets := total_tickets * 15 / 100
  let remaining := total_tickets - (koala + earbuds + car + bracelets)
  let poster := (remaining * 4) / 7
  let keychain := (remaining * 3) / 7
  koala = 100 ∧ 
  earbuds = 30 ∧ 
  car = 60 ∧ 
  bracelets = 75 ∧ 
  poster = 135 ∧ 
  keychain = 100 ∧
  koala + earbuds + car + bracelets + poster + keychain = total_tickets

theorem connie_tickets : ticket_distribution 500 := by
  sorry

end NUMINAMATH_CALUDE_connie_tickets_l304_30460


namespace NUMINAMATH_CALUDE_tan_105_minus_one_over_tan_105_plus_one_equals_sqrt_three_l304_30416

theorem tan_105_minus_one_over_tan_105_plus_one_equals_sqrt_three :
  (Real.tan (105 * π / 180) - 1) / (Real.tan (105 * π / 180) + 1) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_105_minus_one_over_tan_105_plus_one_equals_sqrt_three_l304_30416


namespace NUMINAMATH_CALUDE_max_product_sum_constraint_l304_30435

theorem max_product_sum_constraint (w x y z : ℝ) : 
  w ≥ 0 → x ≥ 0 → y ≥ 0 → z ≥ 0 → w + x + y + z = 200 → 
  (w + x) * (y + z) ≤ 10000 := by
sorry

end NUMINAMATH_CALUDE_max_product_sum_constraint_l304_30435


namespace NUMINAMATH_CALUDE_percentage_problem_l304_30420

theorem percentage_problem (x : ℝ) : 0.25 * x = 0.15 * 1500 - 20 → x = 820 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l304_30420


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l304_30427

def U : Set ℕ := {x | 1 < x ∧ x < 5}
def A : Set ℕ := {2, 3}

theorem complement_of_A_in_U : (U \ A) = {4} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l304_30427


namespace NUMINAMATH_CALUDE_jordyn_total_cost_l304_30476

/-- The total amount Jordyn would pay for the fruits with discounts, sales tax, and service charge -/
def total_cost (cherry_price olives_price grapes_price : ℚ)
               (cherry_quantity olives_quantity grapes_quantity : ℕ)
               (cherry_discount olives_discount grapes_discount : ℚ)
               (sales_tax service_charge : ℚ) : ℚ :=
  let cherry_total := cherry_price * cherry_quantity
  let olives_total := olives_price * olives_quantity
  let grapes_total := grapes_price * grapes_quantity
  let cherry_discounted := cherry_total * (1 - cherry_discount)
  let olives_discounted := olives_total * (1 - olives_discount)
  let grapes_discounted := grapes_total * (1 - grapes_discount)
  let subtotal := cherry_discounted + olives_discounted + grapes_discounted
  let with_tax := subtotal * (1 + sales_tax)
  with_tax * (1 + service_charge)

/-- The theorem stating the total cost Jordyn would pay -/
theorem jordyn_total_cost :
  total_cost 5 7 11 50 75 25 (12/100) (8/100) (15/100) (5/100) (2/100) = 1002.32 := by
  sorry

end NUMINAMATH_CALUDE_jordyn_total_cost_l304_30476


namespace NUMINAMATH_CALUDE_wire_cutting_problem_l304_30457

theorem wire_cutting_problem (wire_length : ℕ) (num_pieces : ℕ) (piece_length : ℕ) :
  wire_length = 1040 ∧ 
  num_pieces = 15 ∧ 
  wire_length = num_pieces * piece_length ∧
  piece_length > 0 →
  piece_length = 66 :=
by sorry

end NUMINAMATH_CALUDE_wire_cutting_problem_l304_30457


namespace NUMINAMATH_CALUDE_binomial_coefficient_sequence_periodic_l304_30440

/-- 
Given positive integers k and m, the sequence of binomial coefficients (n choose k) mod m,
where n ≥ k, is periodic.
-/
theorem binomial_coefficient_sequence_periodic (k m : ℕ+) :
  ∃ (p : ℕ+), ∀ (n : ℕ), n ≥ k →
    (n.choose k : ZMod m) = ((n + p) : ℕ).choose k := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_sequence_periodic_l304_30440


namespace NUMINAMATH_CALUDE_sequence_equation_l304_30452

theorem sequence_equation (n : ℕ+) : 9 * (n - 1) + n = (n - 1) * 10 + 1 := by
  sorry

end NUMINAMATH_CALUDE_sequence_equation_l304_30452


namespace NUMINAMATH_CALUDE_base5_sum_theorem_l304_30464

/-- Converts a base-10 number to its base-5 representation -/
def toBase5 (n : ℕ) : List ℕ :=
  sorry

/-- Converts a base-5 representation to base-10 -/
def fromBase5 (l : List ℕ) : ℕ :=
  sorry

/-- Adds two base-5 numbers represented as lists -/
def addBase5 (a b : List ℕ) : List ℕ :=
  sorry

theorem base5_sum_theorem :
  let a := toBase5 259
  let b := toBase5 63
  addBase5 a b = [2, 2, 4, 2] := by sorry

end NUMINAMATH_CALUDE_base5_sum_theorem_l304_30464


namespace NUMINAMATH_CALUDE_first_month_sale_is_5921_l304_30424

/-- Calculates the sale in the first month given the sales for months 2 to 6 and the average sale -/
def first_month_sale (sales_2_to_5 : List ℕ) (sale_6 : ℕ) (average : ℕ) : ℕ :=
  6 * average - (sales_2_to_5.sum + sale_6)

/-- Theorem stating that the sale in the first month is 5921 -/
theorem first_month_sale_is_5921 :
  first_month_sale [5468, 5568, 6088, 6433] 5922 5900 = 5921 := by
  sorry

end NUMINAMATH_CALUDE_first_month_sale_is_5921_l304_30424


namespace NUMINAMATH_CALUDE_largest_non_sum_of_composite_odd_l304_30401

/-- A function that checks if a number is composite -/
def isComposite (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ a * b = n

/-- A function that checks if a number is odd -/
def isOdd (n : ℕ) : Prop :=
  ∃ (k : ℕ), n = 2 * k + 1

/-- A function that checks if a number can be expressed as the sum of two composite odd positive integers -/
def isSumOfTwoCompositeOdd (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ isOdd a ∧ isOdd b ∧ isComposite a ∧ isComposite b ∧ a + b = n

/-- The main theorem stating that 38 is the largest even positive integer that cannot be expressed as the sum of two composite odd positive integers -/
theorem largest_non_sum_of_composite_odd :
  (∀ (n : ℕ), n > 38 → n % 2 = 0 → isSumOfTwoCompositeOdd n) ∧
  ¬(isSumOfTwoCompositeOdd 38) :=
sorry

end NUMINAMATH_CALUDE_largest_non_sum_of_composite_odd_l304_30401


namespace NUMINAMATH_CALUDE_max_value_of_sum_of_roots_l304_30439

theorem max_value_of_sum_of_roots (x : ℝ) (h : 3 < x ∧ x < 6) :
  ∃ (k : ℝ), k = Real.sqrt 6 ∧ ∀ y : ℝ, (Real.sqrt (x - 3) + Real.sqrt (6 - x) ≤ y) → y ≥ k :=
sorry

end NUMINAMATH_CALUDE_max_value_of_sum_of_roots_l304_30439


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_l304_30428

theorem quadratic_roots_sum (m n : ℝ) : 
  (m^2 + m - 12 = 0) → (n^2 + n - 12 = 0) → m^2 + 2*m + n = 11 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_l304_30428


namespace NUMINAMATH_CALUDE_rental_company_properties_l304_30461

/-- Represents the rental company's car rental scenario. -/
structure RentalCompany where
  totalCars : ℕ := 100
  initialRent : ℕ := 3000
  rentIncrement : ℕ := 50
  rentedCarMaintenance : ℕ := 150
  nonRentedCarMaintenance : ℕ := 50

/-- Calculates the number of cars rented given a specific rent. -/
def carsRented (company : RentalCompany) (rent : ℕ) : ℕ :=
  company.totalCars - (rent - company.initialRent) / company.rentIncrement

/-- Calculates the monthly revenue given a specific rent. -/
def monthlyRevenue (company : RentalCompany) (rent : ℕ) : ℕ :=
  let rented := carsRented company rent
  rent * rented - company.rentedCarMaintenance * rented - 
    company.nonRentedCarMaintenance * (company.totalCars - rented)

/-- Theorem stating the properties of the rental company scenario. -/
theorem rental_company_properties (company : RentalCompany) : 
  carsRented company 3600 = 88 ∧ 
  (∃ (maxRent : ℕ), maxRent = 4050 ∧ 
    (∀ (rent : ℕ), monthlyRevenue company rent ≤ monthlyRevenue company maxRent) ∧
    monthlyRevenue company maxRent = 307050) := by
  sorry


end NUMINAMATH_CALUDE_rental_company_properties_l304_30461


namespace NUMINAMATH_CALUDE_cooking_gear_final_cost_l304_30451

def cookingGearCost (mitts apron utensils recipients discount tax : ℝ) : ℝ :=
  let knife := 2 * utensils
  let setPrice := mitts + apron + utensils + knife
  let discountedPrice := setPrice * (1 - discount)
  let totalBeforeTax := discountedPrice * recipients
  totalBeforeTax * (1 + tax)

theorem cooking_gear_final_cost :
  cookingGearCost 14 16 10 8 0.25 0.08 = 388.80 := by
  sorry

end NUMINAMATH_CALUDE_cooking_gear_final_cost_l304_30451


namespace NUMINAMATH_CALUDE_max_product_of_externally_tangent_circles_l304_30447

/-- Circle C₁ with center (a, -2) and radius 2 -/
def C₁ (a : ℝ) (x y : ℝ) : Prop := (x - a)^2 + (y + 2)^2 = 4

/-- Circle C₂ with center (-b, -2) and radius 1 -/
def C₂ (b : ℝ) (x y : ℝ) : Prop := (x + b)^2 + (y + 2)^2 = 1

/-- Circles C₁ and C₂ are externally tangent -/
def externally_tangent (a b : ℝ) : Prop := (a + b)^2 = 3^2

theorem max_product_of_externally_tangent_circles (a b : ℝ) 
  (h : externally_tangent a b) : 
  a * b ≤ 9/4 := by sorry

end NUMINAMATH_CALUDE_max_product_of_externally_tangent_circles_l304_30447


namespace NUMINAMATH_CALUDE_purely_imaginary_iff_x_eq_one_l304_30402

theorem purely_imaginary_iff_x_eq_one (x : ℝ) : 
  let z : ℂ := (x^2 - 1) + (x + 1)*I
  (∃ y : ℝ, z = y*I) ↔ x = 1 := by sorry

end NUMINAMATH_CALUDE_purely_imaginary_iff_x_eq_one_l304_30402


namespace NUMINAMATH_CALUDE_friends_game_sales_l304_30485

/-- The amount of money received by Zachary -/
def zachary_amount : ℚ := 40 * 5

/-- The amount of money received by Jason -/
def jason_amount : ℚ := zachary_amount * (1 + 30 / 100)

/-- The amount of money received by Ryan -/
def ryan_amount : ℚ := jason_amount + 50

/-- The amount of money received by Emily -/
def emily_amount : ℚ := ryan_amount * (1 - 20 / 100)

/-- The amount of money received by Lily -/
def lily_amount : ℚ := emily_amount + 70

/-- The total amount of money received by all five friends -/
def total_amount : ℚ := zachary_amount + jason_amount + ryan_amount + emily_amount + lily_amount

theorem friends_game_sales : total_amount = 1336 := by
  sorry

end NUMINAMATH_CALUDE_friends_game_sales_l304_30485


namespace NUMINAMATH_CALUDE_max_term_is_9_8_l304_30403

/-- The sequence defined by n^2 / 2^n for n ≥ 1 -/
def a (n : ℕ) : ℚ := (n^2 : ℚ) / 2^n

/-- The maximum term of the sequence occurs at n = 3 -/
def max_term_index : ℕ := 3

/-- The maximum value of the sequence -/
def max_term_value : ℚ := 9/8

/-- Theorem stating that the maximum term of the sequence a(n) is 9/8 -/
theorem max_term_is_9_8 :
  (∀ n : ℕ, n ≥ 1 → a n ≤ max_term_value) ∧ 
  (∃ n : ℕ, n ≥ 1 ∧ a n = max_term_value) :=
sorry

end NUMINAMATH_CALUDE_max_term_is_9_8_l304_30403


namespace NUMINAMATH_CALUDE_exponent_calculation_l304_30495

theorem exponent_calculation : (-4)^6 / 4^4 + 2^5 - 7^2 = -1 := by sorry

end NUMINAMATH_CALUDE_exponent_calculation_l304_30495


namespace NUMINAMATH_CALUDE_find_x_l304_30499

theorem find_x : 
  ∀ x : ℝ, 
  (x * 0.48 * 2.50) / (0.12 * 0.09 * 0.5) = 2400.0000000000005 → 
  x = 10.8 := by
sorry

end NUMINAMATH_CALUDE_find_x_l304_30499


namespace NUMINAMATH_CALUDE_fouad_ahmed_age_multiple_l304_30475

theorem fouad_ahmed_age_multiple : ∃ x : ℕ, (26 + x) % 11 = 0 ∧ (26 + x) / 11 = 3 := by
  sorry

end NUMINAMATH_CALUDE_fouad_ahmed_age_multiple_l304_30475


namespace NUMINAMATH_CALUDE_cos_50_tan_40_equals_sqrt_3_l304_30487

theorem cos_50_tan_40_equals_sqrt_3 : 
  4 * Real.cos (50 * π / 180) - Real.tan (40 * π / 180) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_cos_50_tan_40_equals_sqrt_3_l304_30487


namespace NUMINAMATH_CALUDE_bus_capacity_l304_30473

/-- The number of students that can be accommodated by a given number of buses,
    each with a specified number of columns and rows of seats. -/
def total_students (buses : ℕ) (columns : ℕ) (rows : ℕ) : ℕ :=
  buses * columns * rows

/-- Theorem stating that 6 buses with 4 columns and 10 rows each can accommodate 240 students. -/
theorem bus_capacity : total_students 6 4 10 = 240 := by
  sorry

end NUMINAMATH_CALUDE_bus_capacity_l304_30473


namespace NUMINAMATH_CALUDE_jackies_free_time_l304_30400

def hours_in_day : ℕ := 24

def working_hours : ℕ := 8
def exercise_hours : ℕ := 3
def sleep_hours : ℕ := 8

def scheduled_hours : ℕ := working_hours + exercise_hours + sleep_hours

theorem jackies_free_time : hours_in_day - scheduled_hours = 5 := by
  sorry

end NUMINAMATH_CALUDE_jackies_free_time_l304_30400


namespace NUMINAMATH_CALUDE_original_denominator_problem_l304_30456

theorem original_denominator_problem (d : ℤ) : 
  (3 : ℚ) / d ≠ 0 →
  (9 : ℚ) / (d + 7) = (1 : ℚ) / 3 →
  d = 20 := by
sorry

end NUMINAMATH_CALUDE_original_denominator_problem_l304_30456


namespace NUMINAMATH_CALUDE_eliza_says_500_l304_30406

-- Define the upper bound of the counting range
def upper_bound : ℕ := 500

-- Define the skipping pattern for each student
def alice_skip (n : ℕ) : Bool := n % 4 = 0
def barbara_skip (n : ℕ) : Bool := n % 12 = 4
def candice_skip (n : ℕ) : Bool := n % 16 = 0
def debbie_skip (n : ℕ) : Bool := n % 64 = 0

-- Define a function to check if a number is said by any of the first four students
def is_said_by_first_four (n : ℕ) : Bool :=
  ¬(alice_skip n) ∨ ¬(barbara_skip n) ∨ ¬(candice_skip n) ∨ ¬(debbie_skip n)

-- Theorem statement
theorem eliza_says_500 : 
  ∀ n : ℕ, n ≤ upper_bound → (n ≠ upper_bound → is_said_by_first_four n) ∧ ¬(is_said_by_first_four upper_bound) :=
by sorry

end NUMINAMATH_CALUDE_eliza_says_500_l304_30406


namespace NUMINAMATH_CALUDE_leading_coefficient_of_P_l304_30419

/-- The polynomial in question -/
def P (x : ℝ) : ℝ := 5*(x^5 - 2*x^4 + 3*x^3) - 6*(x^5 + x^3 + x) + 3*(3*x^5 - x^4 + 4*x^2 + 2)

/-- The leading coefficient of a polynomial -/
def leading_coefficient (p : ℝ → ℝ) : ℝ := 
  sorry

theorem leading_coefficient_of_P : leading_coefficient P = 8 := by
  sorry

end NUMINAMATH_CALUDE_leading_coefficient_of_P_l304_30419


namespace NUMINAMATH_CALUDE_problem_solution_l304_30459

noncomputable def f (x : ℝ) : ℝ := Real.exp x * Real.log x + (2 / x) * Real.exp (x - 1)

noncomputable def g (x : ℝ) : ℝ := x * Real.exp (-x) - 2 / Real.exp 1

theorem problem_solution (x : ℝ) (hx : x > 0) :
  f 1 = 2 ∧ 
  (deriv f) 1 = Real.exp 1 ∧
  (∀ y > 0, g y ≤ -1 / Real.exp 1) ∧
  (∀ y > 0, g y = -1 / Real.exp 1 → y = 1) ∧
  f x > 1 :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l304_30459


namespace NUMINAMATH_CALUDE_triangle_problem_l304_30444

theorem triangle_problem (A B C : Real) (a b c : Real) :
  (0 < A) ∧ (A < Real.pi) ∧
  (0 < B) ∧ (B < Real.pi) ∧
  (0 < C) ∧ (C < Real.pi) ∧
  (A + B + C = Real.pi) ∧
  (Real.sin A)^2 - (Real.sin B)^2 - (Real.sin C)^2 = Real.sin B * Real.sin C ∧
  a = 3 ∧
  a / Real.sin A = b / Real.sin B ∧
  b / Real.sin B = c / Real.sin C →
  A = 2 * Real.pi / 3 ∧
  (a + b + c) ≤ 3 + 2 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_problem_l304_30444


namespace NUMINAMATH_CALUDE_total_pears_picked_l304_30458

theorem total_pears_picked (sara tim emily max : ℕ) 
  (h_sara : sara = 6)
  (h_tim : tim = 5)
  (h_emily : emily = 9)
  (h_max : max = 12) :
  sara + tim + emily + max = 32 := by
  sorry

end NUMINAMATH_CALUDE_total_pears_picked_l304_30458


namespace NUMINAMATH_CALUDE_single_elimination_tournament_games_l304_30471

/-- The number of games in a single-elimination tournament -/
def num_games (n : ℕ) : ℕ := n - 1

/-- Theorem: In a single-elimination tournament with 17 teams and no ties, 
    the number of games played is 16 -/
theorem single_elimination_tournament_games : 
  num_games 17 = 16 := by sorry

end NUMINAMATH_CALUDE_single_elimination_tournament_games_l304_30471


namespace NUMINAMATH_CALUDE_sum_of_squares_representation_l304_30404

theorem sum_of_squares_representation (n : ℕ) :
  ∃ (m : ℤ), (∃ (representations : Finset (ℤ × ℤ)), 
    (∀ (pair : ℤ × ℤ), pair ∈ representations → m = pair.1^2 + pair.2^2) ∧
    representations.card ≥ n) ∧ 
  m = 5^(2*n) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_squares_representation_l304_30404


namespace NUMINAMATH_CALUDE_equipment_cannot_fit_l304_30498

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The L-shaped corridor width -/
def corridorWidth : ℝ := 3

/-- The center of the unit circle representing the corner of the L-shaped corridor -/
def circleCenter : Point := ⟨corridorWidth, corridorWidth⟩

/-- The radius of the circle representing the corner of the L-shaped corridor -/
def circleRadius : ℝ := 1

/-- The origin point -/
def origin : Point := ⟨0, 0⟩

/-- The maximum length of the equipment -/
def maxEquipmentLength : ℝ := 7

/-- A line passing through the origin -/
structure Line where
  a : ℝ
  b : ℝ

/-- The length of the line segment from the origin to its intersection with the circle -/
def lineSegmentLength (l : Line) : ℝ := sorry

/-- The minimum length of a line segment intersecting the circle and passing through the origin -/
def minLineSegmentLength : ℝ := sorry

/-- Theorem stating that the equipment cannot fit through the L-shaped corridor -/
theorem equipment_cannot_fit : minLineSegmentLength > maxEquipmentLength := by sorry

end NUMINAMATH_CALUDE_equipment_cannot_fit_l304_30498


namespace NUMINAMATH_CALUDE_triangle_problem_l304_30448

noncomputable def f (x : ℝ) := Real.sqrt 3 * Real.sin (2 * x) + 2 * (Real.cos x) ^ 2 - 1

theorem triangle_problem (A B C a b c : ℝ) :
  c = Real.sqrt 3 →
  f C = 1 →
  Real.sin B = 2 * Real.sin A →
  (∃ (T : ℝ), T > 0 ∧ ∀ (x : ℝ), f (x + T) = f x ∧ ∀ (T' : ℝ), T' > 0 ∧ (∀ (x : ℝ), f (x + T') = f x) → T ≤ T') →
  a = 1 ∧ b = 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_problem_l304_30448


namespace NUMINAMATH_CALUDE_triangle_problem_l304_30450

open Real

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  3 * b = 4 * c →
  B = 2 * C →
  sin B = (4 * Real.sqrt 5) / 9 ∧
  (b = 4 → (1/2) * b * c * sin A = (14 * Real.sqrt 5) / 9) :=
by sorry

end NUMINAMATH_CALUDE_triangle_problem_l304_30450


namespace NUMINAMATH_CALUDE_triangle_area_proof_l304_30436

theorem triangle_area_proof (a b c : ℝ) (h1 : a + b + c = 10 + 2 * Real.sqrt 7) 
  (h2 : a / 2 = b / 3) (h3 : a / 2 = c / Real.sqrt 7) : 
  Real.sqrt ((1 / 4) * (c^2 * a^2 - ((c^2 + a^2 - b^2) / 2)^2)) = 6 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_proof_l304_30436


namespace NUMINAMATH_CALUDE_constant_water_level_l304_30437

/-- Represents a water pipe that can fill or empty a tank -/
structure Pipe where
  rate : ℚ  -- Rate of fill/empty (positive for fill, negative for empty)

/-- Represents a water tank system with multiple pipes -/
structure TankSystem where
  pipes : List Pipe

def TankSystem.netRate (system : TankSystem) : ℚ :=
  system.pipes.map (λ p => p.rate) |>.sum

theorem constant_water_level (pipeA pipeB pipeC : Pipe) 
  (hA : pipeA.rate = 1 / 15)
  (hB : pipeB.rate = -1 / 6)  -- Negative because it empties the tank
  (hC : pipeC.rate = 1 / 10) :
  TankSystem.netRate { pipes := [pipeA, pipeB, pipeC] } = 0 := by
  sorry

#check constant_water_level

end NUMINAMATH_CALUDE_constant_water_level_l304_30437


namespace NUMINAMATH_CALUDE_divisors_multiple_of_five_l304_30421

def n : ℕ := 7560

-- Define a function that counts positive divisors of n that are multiples of 5
def count_divisors_multiple_of_five (n : ℕ) : ℕ :=
  (Finset.filter (λ d => d ∣ n ∧ 5 ∣ d) (Finset.range (n + 1))).card

-- State the theorem
theorem divisors_multiple_of_five :
  count_divisors_multiple_of_five n = 32 := by
  sorry

end NUMINAMATH_CALUDE_divisors_multiple_of_five_l304_30421


namespace NUMINAMATH_CALUDE_jills_total_earnings_l304_30432

/-- Calculates Jill's earnings over three months of online work --/
def jills_earnings (first_month_daily_rate : ℕ) (days_per_month : ℕ) : ℕ :=
  let first_month := first_month_daily_rate * days_per_month
  let second_month := 2 * first_month_daily_rate * days_per_month
  let third_month := 2 * first_month_daily_rate * (days_per_month / 2)
  first_month + second_month + third_month

/-- Proves that Jill's earnings over three months equals $1200 --/
theorem jills_total_earnings : 
  jills_earnings 10 30 = 1200 := by
  sorry

end NUMINAMATH_CALUDE_jills_total_earnings_l304_30432


namespace NUMINAMATH_CALUDE_triangle_237_not_exists_triangle_555_exists_l304_30429

/-- Triangle inequality theorem checker -/
def satisfies_triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Theorem: A triangle with sides 2, 3, and 7 does not satisfy the triangle inequality -/
theorem triangle_237_not_exists : ¬ (satisfies_triangle_inequality 2 3 7) :=
sorry

/-- Theorem: A triangle with sides 5, 5, and 5 satisfies the triangle inequality -/
theorem triangle_555_exists : satisfies_triangle_inequality 5 5 5 :=
sorry

end NUMINAMATH_CALUDE_triangle_237_not_exists_triangle_555_exists_l304_30429


namespace NUMINAMATH_CALUDE_pure_imaginary_condition_l304_30480

theorem pure_imaginary_condition (a : ℝ) : 
  (Complex.I : ℂ).im ≠ 0 →
  (Complex.ofReal (a - 1) * Complex.ofReal (a + 1) + Complex.I) = 
    Complex.ofReal (a^2 - 1) + Complex.I * Complex.ofReal (a - 1) →
  (Complex.ofReal (a - 1) * Complex.ofReal (a + 1) + Complex.I).re = 0 →
  a = 1 := by
sorry

end NUMINAMATH_CALUDE_pure_imaginary_condition_l304_30480


namespace NUMINAMATH_CALUDE_parabola_line_intersection_l304_30412

/-- Given a parabola and a line intersecting it, prove the value of m -/
theorem parabola_line_intersection (x₁ x₂ y₁ y₂ m : ℝ) : 
  (x₁^2 = 4*y₁) →  -- Point A on parabola
  (x₂^2 = 4*y₂) →  -- Point B on parabola
  (∃ k, y₁ = k*x₁ + m ∧ y₂ = k*x₂ + m) →  -- Line equation
  (x₁ * x₂ = -4) →  -- Product of x-coordinates
  (m = 1) := by
sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_l304_30412


namespace NUMINAMATH_CALUDE_team_win_percentage_l304_30415

theorem team_win_percentage (first_games : ℕ) (first_wins : ℕ) (remaining_games : ℕ) (remaining_wins : ℕ) :
  first_games = 50 →
  first_wins = 40 →
  remaining_games = 40 →
  remaining_wins = 23 →
  (first_wins + remaining_wins : ℚ) / (first_games + remaining_games : ℚ) = 7 / 10 := by
  sorry

end NUMINAMATH_CALUDE_team_win_percentage_l304_30415


namespace NUMINAMATH_CALUDE_stone_game_ratio_bound_l304_30494

/-- The stone game process -/
structure StoneGame where
  n : ℕ
  s : ℕ
  t : ℕ
  board : Multiset ℕ

/-- The rules of the stone game -/
def stone_game_step (game : StoneGame) (a b : ℕ) : StoneGame :=
  { n := game.n
  , s := game.s + 1
  , t := game.t + Nat.gcd a b
  , board := game.board - {a, b} + {1, a + b}
  }

/-- The theorem to prove -/
theorem stone_game_ratio_bound (game : StoneGame) (h_n : game.n ≥ 3) 
    (h_init : game.board = Multiset.replicate game.n 1) 
    (h_s_pos : game.s > 0) : 
    1 ≤ (game.t : ℚ) / game.s ∧ (game.t : ℚ) / game.s < game.n - 1 := by
  sorry

end NUMINAMATH_CALUDE_stone_game_ratio_bound_l304_30494


namespace NUMINAMATH_CALUDE_tobacco_acreage_change_l304_30465

/-- Calculates the change in tobacco acreage when crop ratios are adjusted -/
theorem tobacco_acreage_change 
  (total_land : ℝ) 
  (initial_ratio_corn initial_ratio_sugarcane initial_ratio_tobacco : ℕ)
  (new_ratio_corn new_ratio_sugarcane new_ratio_tobacco : ℕ) : 
  total_land = 1350 ∧ 
  initial_ratio_corn = 5 ∧ 
  initial_ratio_sugarcane = 2 ∧ 
  initial_ratio_tobacco = 2 ∧
  new_ratio_corn = 2 ∧ 
  new_ratio_sugarcane = 2 ∧ 
  new_ratio_tobacco = 5 →
  (new_ratio_tobacco * total_land / (new_ratio_corn + new_ratio_sugarcane + new_ratio_tobacco) -
   initial_ratio_tobacco * total_land / (initial_ratio_corn + initial_ratio_sugarcane + initial_ratio_tobacco)) = 450 :=
by sorry

end NUMINAMATH_CALUDE_tobacco_acreage_change_l304_30465


namespace NUMINAMATH_CALUDE_evelyn_bottle_caps_l304_30492

/-- The number of bottle caps Evelyn ends with after losing some -/
def bottle_caps_remaining (initial : Float) (lost : Float) : Float :=
  initial - lost

/-- Theorem: If Evelyn starts with 63.0 bottle caps and loses 18.0, she ends with 45.0 -/
theorem evelyn_bottle_caps : bottle_caps_remaining 63.0 18.0 = 45.0 := by
  sorry

end NUMINAMATH_CALUDE_evelyn_bottle_caps_l304_30492


namespace NUMINAMATH_CALUDE_exactly_one_positive_l304_30414

theorem exactly_one_positive (a b c : ℝ) 
  (sum_zero : a + b + c = 0) 
  (product_one : a * b * c = 1) : 
  (a > 0 ∧ b ≤ 0 ∧ c ≤ 0) ∨ 
  (a ≤ 0 ∧ b > 0 ∧ c ≤ 0) ∨ 
  (a ≤ 0 ∧ b ≤ 0 ∧ c > 0) :=
sorry

end NUMINAMATH_CALUDE_exactly_one_positive_l304_30414


namespace NUMINAMATH_CALUDE_jeff_games_won_l304_30449

def minutes_per_hour : ℕ := 60
def play_duration_hours : ℕ := 2
def minutes_per_point : ℕ := 5
def points_per_match : ℕ := 8

def games_won : ℕ :=
  (play_duration_hours * minutes_per_hour) / minutes_per_point / points_per_match

theorem jeff_games_won : games_won = 3 := by
  sorry

end NUMINAMATH_CALUDE_jeff_games_won_l304_30449


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l304_30486

/-- The eccentricity of a hyperbola with equation x²/a² - y²/b² = 1 is √(1 + b²/a²) -/
theorem hyperbola_eccentricity (a b : ℝ) (h : 0 < a ∧ 0 < b) :
  let e := Real.sqrt (1 + b^2 / a^2)
  ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 → e = 5/3 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l304_30486


namespace NUMINAMATH_CALUDE_statement_I_statement_II_statement_III_l304_30469

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

-- Statement I
theorem statement_I : ∀ x : ℝ, floor (x + 1) = floor x + 1 := by sorry

-- Statement II (negation)
theorem statement_II : ∃ x y : ℝ, ∃ k : ℤ, floor (x + y + k) ≠ floor x + floor y + k := by sorry

-- Statement III (negation)
theorem statement_III : ∃ x y : ℝ, floor (x * y) ≠ floor x * floor y := by sorry

end NUMINAMATH_CALUDE_statement_I_statement_II_statement_III_l304_30469


namespace NUMINAMATH_CALUDE_fish_ratio_l304_30410

def fish_problem (O B R : ℕ) : Prop :=
  O = B + 25 ∧
  B = 75 ∧
  (O + B + R) / 3 = 75

theorem fish_ratio : ∀ O B R : ℕ, fish_problem O B R → R * 2 = O :=
sorry

end NUMINAMATH_CALUDE_fish_ratio_l304_30410
