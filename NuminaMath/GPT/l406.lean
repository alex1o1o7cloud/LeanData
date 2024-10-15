import Mathlib

namespace NUMINAMATH_GPT_sculpture_height_correct_l406_40629

/-- Define the conditions --/
def base_height_in_inches : ℝ := 4
def total_height_in_feet : ℝ := 3.1666666666666665
def inches_per_foot : ℝ := 12

/-- Define the conversion from feet to inches for the total height --/
def total_height_in_inches : ℝ := total_height_in_feet * inches_per_foot

/-- Define the height of the sculpture in inches --/
def sculpture_height_in_inches : ℝ := total_height_in_inches - base_height_in_inches

/-- The proof problem in Lean 4 statement --/
theorem sculpture_height_correct :
  sculpture_height_in_inches = 34 := by
  sorry

end NUMINAMATH_GPT_sculpture_height_correct_l406_40629


namespace NUMINAMATH_GPT_net_rate_of_pay_l406_40680

/-- The net rate of pay in dollars per hour for a truck driver after deducting gasoline expenses. -/
theorem net_rate_of_pay
  (hrs : ℕ) (speed : ℕ) (miles_per_gallon : ℕ) (pay_per_mile : ℚ) (cost_per_gallon : ℚ) 
  (H1 : hrs = 3)
  (H2 : speed = 50)
  (H3 : miles_per_gallon = 25)
  (H4 : pay_per_mile = 0.6)
  (H5 : cost_per_gallon = 2.50) :
  pay_per_mile * (hrs * speed) - cost_per_gallon * ((hrs * speed) / miles_per_gallon) = 25 * hrs :=
by sorry

end NUMINAMATH_GPT_net_rate_of_pay_l406_40680


namespace NUMINAMATH_GPT_find_percentage_l406_40668

variable (P : ℝ)
variable (num : ℝ := 70)
variable (result : ℝ := 25)

theorem find_percentage (h : ((P / 100) * num) - 10 = result) : P = 50 := by
  sorry

end NUMINAMATH_GPT_find_percentage_l406_40668


namespace NUMINAMATH_GPT_calc_pow_product_l406_40673

theorem calc_pow_product : (0.25 ^ 2023) * (4 ^ 2023) = 1 := 
  by 
  sorry

end NUMINAMATH_GPT_calc_pow_product_l406_40673


namespace NUMINAMATH_GPT_part1_part2_l406_40679

open Set

variable {U : Type} [TopologicalSpace U]

def universal_set : Set ℝ := univ
def set_A : Set ℝ := {x | -1 ≤ x ∧ x < 3}
def set_B (k : ℝ) : Set ℝ := {x | x ≤ k}

theorem part1 (k : ℝ) (hk : k = 1) :
  A ∩ (univ \ set_B k) = {x | 1 < x ∧ x < 3} :=
by
  sorry

theorem part2 (k : ℝ) (h : set_A ∩ set_B k ≠ ∅) :
  k ≥ -1 :=
by
  sorry

end NUMINAMATH_GPT_part1_part2_l406_40679


namespace NUMINAMATH_GPT_race_course_length_l406_40642

variable (v d : ℝ)

theorem race_course_length (h1 : 4 * v > 0) (h2 : ∀ t : ℝ, t > 0 → (d / (4 * v)) = ((d - 72) / v)) : d = 96 := by
  sorry

end NUMINAMATH_GPT_race_course_length_l406_40642


namespace NUMINAMATH_GPT_cuboid_layers_l406_40623

theorem cuboid_layers (V : ℕ) (n_blocks : ℕ) (volume_per_block : ℕ) (blocks_per_layer : ℕ)
  (hV : V = 252) (hvol : volume_per_block = 1) (hblocks : n_blocks = V / volume_per_block) (hlayer : blocks_per_layer = 36) :
  (n_blocks / blocks_per_layer) = 7 :=
by
  sorry

end NUMINAMATH_GPT_cuboid_layers_l406_40623


namespace NUMINAMATH_GPT_seashells_calculation_l406_40657

theorem seashells_calculation :
  let mimi_seashells := 24
  let kyle_seashells := 2 * mimi_seashells
  let leigh_seashells := kyle_seashells / 3
  leigh_seashells = 16 :=
by
  let mimi_seashells := 24
  let kyle_seashells := 2 * mimi_seashells
  let leigh_seashells := kyle_seashells / 3
  show leigh_seashells = 16
  sorry

end NUMINAMATH_GPT_seashells_calculation_l406_40657


namespace NUMINAMATH_GPT_sector_area_l406_40650

theorem sector_area (r : ℝ) (h1 : r = 2) (h2 : 2 * r + r * ((2 * π * r - 2) / r) = 4 * π) :
  (1 / 2) * r^2 * ((4 * π - 2) / r) = 4 * π - 2 :=
by
  sorry

end NUMINAMATH_GPT_sector_area_l406_40650


namespace NUMINAMATH_GPT_first_player_guaranteed_win_l406_40663

def is_power_of_two (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 2 ^ k

theorem first_player_guaranteed_win (n : ℕ) (h : n > 1) : 
  ¬ is_power_of_two n ↔ ∃ m : ℕ, 1 ≤ m ∧ m < n ∧ (∀ k : ℕ, m ≤ k + 1 → ∀ t, t ≤ m → ∃ r, r = k + 1 ∧ r <= m) → 
                                (∃ l : ℕ, (l = 1) → true) :=
sorry

end NUMINAMATH_GPT_first_player_guaranteed_win_l406_40663


namespace NUMINAMATH_GPT_difference_between_number_and_its_3_5_l406_40662

theorem difference_between_number_and_its_3_5 (x : ℕ) (h : x = 155) :
  x - (3 / 5 : ℚ) * x = 62 := by
  sorry

end NUMINAMATH_GPT_difference_between_number_and_its_3_5_l406_40662


namespace NUMINAMATH_GPT_total_pupils_correct_l406_40688

def number_of_girls : ℕ := 868
def difference_girls_boys : ℕ := 281
def number_of_boys : ℕ := number_of_girls - difference_girls_boys
def total_pupils : ℕ := number_of_girls + number_of_boys

theorem total_pupils_correct : total_pupils = 1455 := by
  sorry

end NUMINAMATH_GPT_total_pupils_correct_l406_40688


namespace NUMINAMATH_GPT_radius_squared_of_intersection_circle_l406_40699

def parabola1 (x y : ℝ) := y = (x - 2) ^ 2
def parabola2 (x y : ℝ) := x + 6 = (y - 5) ^ 2

theorem radius_squared_of_intersection_circle
    (x y : ℝ)
    (h₁ : parabola1 x y)
    (h₂ : parabola2 x y) :
    ∃ r, r ^ 2 = 83 / 4 :=
sorry

end NUMINAMATH_GPT_radius_squared_of_intersection_circle_l406_40699


namespace NUMINAMATH_GPT_relationship_between_f_l406_40652

-- Given definitions
def quadratic_parabola (a b c x : ℝ) : ℝ := a * x^2 + b * x + c
def axis_of_symmetry (f : ℝ → ℝ) (α : ℝ) : Prop :=
  ∀ x y : ℝ, f x = f y ↔ x + y = 2 * α

-- The problem statement to prove in Lean 4
theorem relationship_between_f (a b c x : ℝ) (hpos : x > 0) (apos : a > 0) :
  axis_of_symmetry (quadratic_parabola a b c) 1 →
  quadratic_parabola a b c (3^x) > quadratic_parabola a b c (2^x) :=
by
  sorry

end NUMINAMATH_GPT_relationship_between_f_l406_40652


namespace NUMINAMATH_GPT_sin_add_cos_l406_40644

theorem sin_add_cos (s72 c18 c72 s18 : ℝ) (h1 : s72 = Real.sin (72 * Real.pi / 180)) (h2 : c18 = Real.cos (18 * Real.pi / 180)) (h3 : c72 = Real.cos (72 * Real.pi / 180)) (h4 : s18 = Real.sin (18 * Real.pi / 180)) :
  s72 * c18 + c72 * s18 = 1 :=
by 
  sorry

end NUMINAMATH_GPT_sin_add_cos_l406_40644


namespace NUMINAMATH_GPT_jovana_initial_shells_l406_40695

theorem jovana_initial_shells (x : ℕ) (h₁ : x + 12 = 17) : x = 5 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_jovana_initial_shells_l406_40695


namespace NUMINAMATH_GPT_johns_money_left_l406_40676

def dog_walking_days_in_april := 26
def earnings_per_day := 10
def money_spent_on_books := 50
def money_given_to_sister := 50

theorem johns_money_left : (dog_walking_days_in_april * earnings_per_day) - (money_spent_on_books + money_given_to_sister) = 160 := 
by
  sorry

end NUMINAMATH_GPT_johns_money_left_l406_40676


namespace NUMINAMATH_GPT_ticket_price_l406_40628

theorem ticket_price (P : ℝ) (h_capacity : 50 * P - 24 * P = 208) :
  P = 8 :=
sorry

end NUMINAMATH_GPT_ticket_price_l406_40628


namespace NUMINAMATH_GPT_solve_equation_l406_40677

theorem solve_equation (x y : ℤ) (h : 3 * (y - 2) = 5 * (x - 1)) :
  (x = 1 ∧ y = 2) ∨ (x = 4 ∧ y = 7) :=
sorry

end NUMINAMATH_GPT_solve_equation_l406_40677


namespace NUMINAMATH_GPT_tangent_circles_locus_l406_40611

noncomputable def locus_condition (a b r : ℝ) : Prop :=
  (a^2 + b^2 = (r + 2)^2) ∧ ((a - 3)^2 + b^2 = (5 - r)^2)

theorem tangent_circles_locus (a b : ℝ) (r : ℝ) (h : locus_condition a b r) :
  a^2 + 7 * b^2 - 34 * a - 57 = 0 :=
sorry

end NUMINAMATH_GPT_tangent_circles_locus_l406_40611


namespace NUMINAMATH_GPT_sunzi_wood_problem_l406_40649

theorem sunzi_wood_problem (x : ℝ) :
  (∃ (length_of_rope : ℝ), length_of_rope = x + 4.5 ∧
    ∃ (half_length_of_rope : ℝ), half_length_of_rope = length_of_rope / 2 ∧ 
      (half_length_of_rope + 1 = x)) ↔ 
  (1 / 2 * (x + 4.5) = x - 1) :=
by
  sorry

end NUMINAMATH_GPT_sunzi_wood_problem_l406_40649


namespace NUMINAMATH_GPT_largest_reciprocal_l406_40655

-- Definitions of the given numbers
def num1 := 1 / 6
def num2 := 2 / 7
def num3 := (2 : ℝ)
def num4 := (8 : ℝ)
def num5 := (1000 : ℝ)

-- The main problem: prove that the reciprocal of 1/6 is the largest
theorem largest_reciprocal :
  (1 / num1 > 1 / num2) ∧ (1 / num1 > 1 / num3) ∧ (1 / num1 > 1 / num4) ∧ (1 / num1 > 1 / num5) :=
by
  sorry

end NUMINAMATH_GPT_largest_reciprocal_l406_40655


namespace NUMINAMATH_GPT_largest_of_given_numbers_l406_40654

theorem largest_of_given_numbers :
  (0.99 > 0.9099) ∧
  (0.99 > 0.9) ∧
  (0.99 > 0.909) ∧
  (0.99 > 0.9009) →
  ∀ (x : ℝ), (x = 0.99 ∨ x = 0.9099 ∨ x = 0.9 ∨ x = 0.909 ∨ x = 0.9009) → 
  x ≤ 0.99 :=
by
  sorry

end NUMINAMATH_GPT_largest_of_given_numbers_l406_40654


namespace NUMINAMATH_GPT_freds_sister_borrowed_3_dimes_l406_40638

-- Define the conditions
def original_dimes := 7
def remaining_dimes := 4

-- Define the question and answer
def borrowed_dimes := original_dimes - remaining_dimes

-- Statement to prove
theorem freds_sister_borrowed_3_dimes : borrowed_dimes = 3 := by
  sorry

end NUMINAMATH_GPT_freds_sister_borrowed_3_dimes_l406_40638


namespace NUMINAMATH_GPT_distance_between_A_B_is_16_l406_40613

-- The given conditions are translated as definitions
def polar_line (ρ θ : ℝ) : Prop := ρ * Real.cos θ = 4

def curve (t : ℝ) : ℝ × ℝ := (t^2, t^3)

-- The theorem stating the proof problem
theorem distance_between_A_B_is_16 :
  let A : ℝ × ℝ := (4, 8)
  let B : ℝ × ℝ := (4, -8)
  let d : ℝ := Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)
  d = 16 :=
by
  sorry

end NUMINAMATH_GPT_distance_between_A_B_is_16_l406_40613


namespace NUMINAMATH_GPT_angle_MON_l406_40656

theorem angle_MON (O M N : ℝ × ℝ) (D : ℝ) :
  (O = (0, 0)) →
  (M = (-2, 2)) →
  (N = (2, 2)) →
  (x^2 + y^2 + D * x - 4 * y = 0) →
  (D = 0) →
  ∃ θ : ℝ, θ = 90 :=
by
  sorry

end NUMINAMATH_GPT_angle_MON_l406_40656


namespace NUMINAMATH_GPT_sum_of_two_coprimes_l406_40603

theorem sum_of_two_coprimes (n : ℤ) (h : n ≥ 7) : 
  ∃ a b : ℤ, a + b = n ∧ Int.gcd a b = 1 ∧ a > 1 ∧ b > 1 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_two_coprimes_l406_40603


namespace NUMINAMATH_GPT_car_speed_l406_40645

theorem car_speed {vp vc : ℚ} (h1 : vp = 7 / 2) (h2 : vc = 6 * vp) : 
  vc = 21 := 
by 
  sorry

end NUMINAMATH_GPT_car_speed_l406_40645


namespace NUMINAMATH_GPT_not_a_cube_l406_40633

theorem not_a_cube (a b : ℤ) : ¬ ∃ c : ℤ, a^3 + b^3 + 4 = c^3 := 
sorry

end NUMINAMATH_GPT_not_a_cube_l406_40633


namespace NUMINAMATH_GPT_bridge_length_l406_40648

/-- The length of the bridge that a train 110 meters long and traveling at 45 km/hr can cross in 30 seconds is 265 meters. -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (cross_time_sec : ℝ) (bridge_length : ℝ) :
  train_length = 110 ∧ train_speed_kmh = 45 ∧ cross_time_sec = 30 ∧ bridge_length = 265 → 
  (train_speed_kmh * (1000 / 3600) * cross_time_sec - train_length = bridge_length) :=
by
  sorry

end NUMINAMATH_GPT_bridge_length_l406_40648


namespace NUMINAMATH_GPT_expected_profit_correct_l406_40615

-- Define the conditions
def ticket_cost : ℝ := 2
def winning_probability : ℝ := 0.01
def prize : ℝ := 50

-- Define the expected profit calculation
def expected_profit : ℝ := (winning_probability * prize) - ticket_cost

-- The theorem we want to prove
theorem expected_profit_correct : expected_profit = -1.5 := by
  sorry

end NUMINAMATH_GPT_expected_profit_correct_l406_40615


namespace NUMINAMATH_GPT_smallest_b_exists_l406_40686

theorem smallest_b_exists :
  ∃ b : ℕ, (∀ r s : ℤ, r * s = 4032 ∧ r + s = b) ∧
    (∀ b' : ℕ, (∀ r' s' : ℤ, r' * s' = 4032 ∧ r' + s' = b') → b ≤ b') :=
sorry

end NUMINAMATH_GPT_smallest_b_exists_l406_40686


namespace NUMINAMATH_GPT_line_equation_passes_through_and_has_normal_l406_40631

theorem line_equation_passes_through_and_has_normal (x y : ℝ) 
    (H1 : ∃ l : ℝ → ℝ, l 3 = 4)
    (H2 : ∃ n : ℝ × ℝ, n = (1, 2)) : 
    x + 2 * y - 11 = 0 :=
sorry

end NUMINAMATH_GPT_line_equation_passes_through_and_has_normal_l406_40631


namespace NUMINAMATH_GPT_num_terminating_decimals_l406_40602

-- Define the problem conditions and statement
def is_terminating_decimal (n : ℕ) : Prop :=
  n % 3 = 0

theorem num_terminating_decimals : 
  ∃ (k : ℕ), k = 220 ∧ (∀ n : ℕ, 1 ≤ n ∧ n ≤ 660 → is_terminating_decimal n ↔ n % 3 = 0) := 
by
  sorry

end NUMINAMATH_GPT_num_terminating_decimals_l406_40602


namespace NUMINAMATH_GPT_ian_lottery_win_l406_40625

theorem ian_lottery_win 
  (amount_paid_to_colin : ℕ)
  (amount_left : ℕ)
  (amount_paid_to_helen : ℕ := 2 * amount_paid_to_colin)
  (amount_paid_to_benedict : ℕ := amount_paid_to_helen / 2)
  (total_debts_paid : ℕ := amount_paid_to_colin + amount_paid_to_helen + amount_paid_to_benedict)
  (total_money_won : ℕ := total_debts_paid + amount_left)
  (h1 : amount_paid_to_colin = 20)
  (h2 : amount_left = 20) :
  total_money_won = 100 := 
sorry

end NUMINAMATH_GPT_ian_lottery_win_l406_40625


namespace NUMINAMATH_GPT_fraction_meaningful_iff_l406_40690

theorem fraction_meaningful_iff (m : ℝ) : 
  (∃ (x : ℝ), x = 3 / (m - 4)) ↔ m ≠ 4 :=
by 
  sorry

end NUMINAMATH_GPT_fraction_meaningful_iff_l406_40690


namespace NUMINAMATH_GPT_remainders_inequalities_l406_40661

theorem remainders_inequalities
  (X Y M A B s t u : ℕ)
  (h1 : X > Y)
  (h2 : X = Y + 8)
  (h3 : X % M = A)
  (h4 : Y % M = B)
  (h5 : s = (X^2) % M)
  (h6 : t = (Y^2) % M)
  (h7 : u = (A * B)^2 % M) :
  s ≠ t ∧ t ≠ u ∧ s ≠ u :=
sorry

end NUMINAMATH_GPT_remainders_inequalities_l406_40661


namespace NUMINAMATH_GPT_gas_volume_at_25_degrees_l406_40643

theorem gas_volume_at_25_degrees :
  (∀ (T V : ℕ), (T = 40 → V = 30) →
  (∀ (k : ℕ), T = 40 - 5 * k → V = 30 - 6 * k) → 
  (25 = 40 - 5 * 3) → 
  (V = 30 - 6 * 3) → 
  V = 12) := 
by
  sorry

end NUMINAMATH_GPT_gas_volume_at_25_degrees_l406_40643


namespace NUMINAMATH_GPT_bank_check_problem_l406_40609

theorem bank_check_problem :
  ∃ (x y : ℕ), (0 ≤ y ∧ y ≤ 99) ∧ (y + (x : ℚ) / 100 - 0.05 = 2 * (x + (y : ℚ) / 100)) ∧ x = 31 ∧ y = 63 :=
by
  -- Definitions and Conditions
  sorry

end NUMINAMATH_GPT_bank_check_problem_l406_40609


namespace NUMINAMATH_GPT_range_of_m_l406_40627

noncomputable def f (x : ℝ) : ℝ := 3 * x + Real.sin x

theorem range_of_m (m : ℝ) 
  (h_odd : ∀ x, f (-x) = -f x) 
  (h_incr : ∀ x y, x < y → f x < f y) : 
  f (2 * m - 1) + f (3 - m) > 0 ↔ m > -2 := 
by 
  sorry

end NUMINAMATH_GPT_range_of_m_l406_40627


namespace NUMINAMATH_GPT_find_g_at_1_l406_40687

noncomputable def g (x : ℝ) : ℝ := x^2 - 2*x + 4

theorem find_g_at_1 : 
  (∀ x : ℝ, g (2*x + 3) = x^2 - 2*x + 4) → 
  g 1 = 7 := 
by
  intro h
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_find_g_at_1_l406_40687


namespace NUMINAMATH_GPT_not_eq_positive_integers_l406_40658

theorem not_eq_positive_integers (a b : ℤ) (ha : a > 0) (hb : b > 0) : 
  a^3 + (a + b)^2 + b ≠ b^3 + a + 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_not_eq_positive_integers_l406_40658


namespace NUMINAMATH_GPT_arithmetic_geometric_sequences_l406_40653

theorem arithmetic_geometric_sequences :
  ∃ (A B C D : ℤ), A < B ∧ B > 0 ∧ C > 0 ∧ -- Ensure A, B, C are positive
  (B - A) = (C - B) ∧  -- Arithmetic sequence condition
  B * (49 : ℚ) = C * (49 / 9 : ℚ) ∧ -- Geometric sequence condition written using fractional equality
  A + B + C + D = 76 := 
by {
  sorry -- Placeholder for actual proof
}

end NUMINAMATH_GPT_arithmetic_geometric_sequences_l406_40653


namespace NUMINAMATH_GPT_repeating_six_as_fraction_l406_40639

theorem repeating_six_as_fraction : (∑' n : ℕ, 6 / (10 * (10 : ℝ)^n)) = (2 / 3) :=
by
  sorry

end NUMINAMATH_GPT_repeating_six_as_fraction_l406_40639


namespace NUMINAMATH_GPT_min_days_to_plant_trees_l406_40697

theorem min_days_to_plant_trees (n : ℕ) (h : 2 ≤ n) :
  (2 ^ (n + 1) - 2 ≥ 1000) ↔ (n ≥ 9) :=
by sorry

end NUMINAMATH_GPT_min_days_to_plant_trees_l406_40697


namespace NUMINAMATH_GPT_find_b_value_l406_40641

theorem find_b_value 
  (point1 : ℝ × ℝ) (point2 : ℝ × ℝ) (b : ℝ) 
  (h1 : point1 = (0, -2))
  (h2 : point2 = (1, 0))
  (h3 : (∃ m c, ∀ x y, y = m * x + c ↔ (x, y) = point1 ∨ (x, y) = point2))
  (h4 : ∀ x y, y = 2 * x - 2 → (x, y) = (7, b)) :
  b = 12 :=
sorry

end NUMINAMATH_GPT_find_b_value_l406_40641


namespace NUMINAMATH_GPT_car_speed_l406_40607

-- Definitions based on the conditions
def distance : ℕ := 375
def time : ℕ := 5

-- Mathematically equivalent proof statement
theorem car_speed : distance / time = 75 := 
  by
  -- The actual proof will be placed here, but we'll skip it for now.
  sorry

end NUMINAMATH_GPT_car_speed_l406_40607


namespace NUMINAMATH_GPT_harvest_season_weeks_l406_40693

-- Definitions based on given conditions
def weekly_earnings : ℕ := 491
def weekly_rent : ℕ := 216
def total_savings : ℕ := 324775

-- Definition to calculate net earnings per week
def net_earnings_per_week (earnings rent : ℕ) : ℕ :=
  earnings - rent

-- Definition to calculate number of weeks
def number_of_weeks (savings net_earnings : ℕ) : ℕ :=
  savings / net_earnings

theorem harvest_season_weeks :
  number_of_weeks total_savings (net_earnings_per_week weekly_earnings weekly_rent) = 1181 :=
by
  sorry

end NUMINAMATH_GPT_harvest_season_weeks_l406_40693


namespace NUMINAMATH_GPT_one_third_of_1206_is_201_percent_of_200_l406_40647

theorem one_third_of_1206_is_201_percent_of_200 : 
  (1 / 3) * 1206 = 402 ∧ 402 / 200 = 201 / 100 :=
by
  sorry

end NUMINAMATH_GPT_one_third_of_1206_is_201_percent_of_200_l406_40647


namespace NUMINAMATH_GPT_isabella_hair_length_l406_40671

theorem isabella_hair_length (original : ℝ) (increase_percent : ℝ) (new_length : ℝ) 
    (h1 : original = 18) (h2 : increase_percent = 0.75) 
    (h3 : new_length = original + increase_percent * original) : 
    new_length = 31.5 := by sorry

end NUMINAMATH_GPT_isabella_hair_length_l406_40671


namespace NUMINAMATH_GPT_brick_width_l406_40635

theorem brick_width (length_courtyard : ℕ) (width_courtyard : ℕ) (num_bricks : ℕ) (brick_length : ℕ) (total_area : ℕ) (brick_area : ℕ) (w : ℕ)
  (h1 : length_courtyard = 1800)
  (h2 : width_courtyard = 1200)
  (h3 : num_bricks = 30000)
  (h4 : brick_length = 12)
  (h5 : total_area = length_courtyard * width_courtyard)
  (h6 : total_area = num_bricks * brick_area)
  (h7 : brick_area = brick_length * w) :
  w = 6 :=
by
  sorry

end NUMINAMATH_GPT_brick_width_l406_40635


namespace NUMINAMATH_GPT_average_speed_first_girl_l406_40626

theorem average_speed_first_girl (v : ℝ) 
  (start_same_point : True)
  (opp_directions : True)
  (avg_speed_second_girl : ℝ := 3)
  (distance_after_12_hours : (v + avg_speed_second_girl) * 12 = 120) :
  v = 7 :=
by
  sorry

end NUMINAMATH_GPT_average_speed_first_girl_l406_40626


namespace NUMINAMATH_GPT_music_tool_cost_l406_40636

noncomputable def flute_cost : ℝ := 142.46
noncomputable def song_book_cost : ℝ := 7
noncomputable def total_spent : ℝ := 158.35

theorem music_tool_cost :
    total_spent - (flute_cost + song_book_cost) = 8.89 :=
by
  sorry

end NUMINAMATH_GPT_music_tool_cost_l406_40636


namespace NUMINAMATH_GPT_find_b_value_l406_40660

theorem find_b_value (x : ℝ) (h_neg : x < 0) (h_eq : 1 / (x + 1 / (x + 2)) = 2) : 
  x + 7 / 2 = 2 :=
sorry

end NUMINAMATH_GPT_find_b_value_l406_40660


namespace NUMINAMATH_GPT_xy_square_sum_l406_40637

theorem xy_square_sum (x y : ℝ) (h1 : (x + y) / 2 = 20) (h2 : Real.sqrt (x * y) = Real.sqrt 132) : x^2 + y^2 = 1336 :=
by
  sorry

end NUMINAMATH_GPT_xy_square_sum_l406_40637


namespace NUMINAMATH_GPT_no_integer_solutions_l406_40670

theorem no_integer_solutions (x y : ℤ) : ¬ (3 * x^2 + 2 = y^2) :=
sorry

end NUMINAMATH_GPT_no_integer_solutions_l406_40670


namespace NUMINAMATH_GPT_like_terms_calc_l406_40651

theorem like_terms_calc {m n : ℕ} (h1 : m + 2 = 6) (h2 : n + 1 = 3) : (- (m : ℤ))^3 + (n : ℤ)^2 = -60 :=
  sorry

end NUMINAMATH_GPT_like_terms_calc_l406_40651


namespace NUMINAMATH_GPT_joe_eggs_club_house_l406_40665

theorem joe_eggs_club_house (C : ℕ) (h : C + 5 + 3 = 20) : C = 12 :=
by 
  sorry

end NUMINAMATH_GPT_joe_eggs_club_house_l406_40665


namespace NUMINAMATH_GPT_incenter_coordinates_l406_40621

theorem incenter_coordinates (p q r : ℝ) (h₁ : p = 8) (h₂ : q = 6) (h₃ : r = 10) :
  ∃ x y z : ℝ, x + y + z = 1 ∧ x = p / (p + q + r) ∧ y = q / (p + q + r) ∧ z = r / (p + q + r) ∧
  x = 1 / 3 ∧ y = 1 / 4 ∧ z = 5 / 12 :=
by
  sorry

end NUMINAMATH_GPT_incenter_coordinates_l406_40621


namespace NUMINAMATH_GPT_find_width_l406_40684

-- Definitions and Conditions
def length : ℝ := 6
def depth : ℝ := 2
def total_surface_area : ℝ := 104

-- Statement to prove the width
theorem find_width (width : ℝ) (h : 12 * width + 4 * width + 24 = total_surface_area) : width = 5 := 
by { 
  -- lean 4 statement only, proof omitted
  sorry 
}

end NUMINAMATH_GPT_find_width_l406_40684


namespace NUMINAMATH_GPT_min_value_x_add_2y_l406_40666

theorem min_value_x_add_2y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 2 * y = x * y) : x + 2 * y ≥ 8 :=
sorry

end NUMINAMATH_GPT_min_value_x_add_2y_l406_40666


namespace NUMINAMATH_GPT_parking_lot_wheels_l406_40698

-- definitions for the conditions
def num_cars : ℕ := 10
def num_bikes : ℕ := 2
def wheels_per_car : ℕ := 4
def wheels_per_bike : ℕ := 2

-- statement of the theorem
theorem parking_lot_wheels : (num_cars * wheels_per_car) + (num_bikes * wheels_per_bike) = 44 := by
  sorry

end NUMINAMATH_GPT_parking_lot_wheels_l406_40698


namespace NUMINAMATH_GPT_profit_at_15_percent_off_l406_40696

theorem profit_at_15_percent_off 
    (cost_price marked_price : ℝ) 
    (cost_price_eq : cost_price = 2000)
    (marked_price_eq : marked_price = (200 + cost_price) / 0.8) :
    (0.85 * marked_price - cost_price) = 337.5 := by
  sorry

end NUMINAMATH_GPT_profit_at_15_percent_off_l406_40696


namespace NUMINAMATH_GPT_geometric_sequence_fourth_term_l406_40616

theorem geometric_sequence_fourth_term (x : ℝ) (h1 : (2 * x + 2) ^ 2 = x * (3 * x + 3))
  (h2 : x ≠ -1) : (3*x + 3) * (3/2) = -27/2 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_fourth_term_l406_40616


namespace NUMINAMATH_GPT_simplify_division_l406_40619

theorem simplify_division :
  (2 * 10^12) / (4 * 10^5 - 1 * 10^4) = 5.1282 * 10^6 :=
by
  -- problem statement
  sorry

end NUMINAMATH_GPT_simplify_division_l406_40619


namespace NUMINAMATH_GPT_interval_of_increase_monotone_increasing_monotonically_increasing_decreasing_l406_40620

noncomputable def f (x a : ℝ) : ℝ := Real.exp x - a * x - 1

theorem interval_of_increase (a : ℝ) : 
  (∀ x : ℝ, 0 < a → (Real.exp x - a ≥ 0 ↔ x ≥ Real.log a)) ∧ 
  (∀ x : ℝ, a ≤ 0 → (Real.exp x - a ≥ 0)) :=
by sorry

theorem monotone_increasing (a : ℝ) (h : ∀ x : ℝ, Real.exp x - a ≥ 0) : 
  a ≤ 0 :=
by sorry

theorem monotonically_increasing_decreasing : 
  ∃ a : ℝ, (∀ x ≤ 0, Real.exp x - a ≤ 0) ∧ 
           (∀ x ≥ 0, Real.exp x - a ≥ 0) ↔ a = 1 :=
by sorry

end NUMINAMATH_GPT_interval_of_increase_monotone_increasing_monotonically_increasing_decreasing_l406_40620


namespace NUMINAMATH_GPT_birgit_hiking_time_l406_40634

def hiking_conditions
  (hours_hiked : ℝ)
  (distance_km : ℝ)
  (time_faster : ℝ)
  (distance_target_km : ℝ) : Prop :=
  ∃ (average_speed_time : ℝ) (birgit_speed_time : ℝ) (total_minutes_hiked : ℝ),
    total_minutes_hiked = hours_hiked * 60 ∧
    average_speed_time = total_minutes_hiked / distance_km ∧
    birgit_speed_time = average_speed_time - time_faster ∧
    (birgit_speed_time * distance_target_km = 48)

theorem birgit_hiking_time
  (hours_hiked : ℝ)
  (distance_km : ℝ)
  (time_faster : ℝ)
  (distance_target_km : ℝ)
  : hiking_conditions hours_hiked distance_km time_faster distance_target_km :=
by
  use 10, 6, 210
  sorry

end NUMINAMATH_GPT_birgit_hiking_time_l406_40634


namespace NUMINAMATH_GPT_sum_largest_smallest_prime_factors_1155_l406_40685

theorem sum_largest_smallest_prime_factors_1155 : 
  ∃ smallest largest : ℕ, 
  smallest ∣ 1155 ∧ largest ∣ 1155 ∧ 
  Prime smallest ∧ Prime largest ∧ 
  smallest <= largest ∧ 
  (∀ p : ℕ, p ∣ 1155 → Prime p → (smallest ≤ p ∧ p ≤ largest)) ∧ 
  (smallest + largest = 14) := 
by {
  sorry
}

end NUMINAMATH_GPT_sum_largest_smallest_prime_factors_1155_l406_40685


namespace NUMINAMATH_GPT_sum_of_roots_eq_six_l406_40630

variable (a b : ℝ)

theorem sum_of_roots_eq_six (h1 : a * (a - 6) = 7) (h2 : b * (b - 6) = 7) (h3 : a ≠ b) : a + b = 6 :=
sorry

end NUMINAMATH_GPT_sum_of_roots_eq_six_l406_40630


namespace NUMINAMATH_GPT_inversely_proportional_x_y_l406_40632

theorem inversely_proportional_x_y {x y k : ℝ}
    (h_inv_proportional : x * y = k)
    (h_k : k = 75)
    (h_y : y = 45) :
    x = 5 / 3 :=
by
  sorry

end NUMINAMATH_GPT_inversely_proportional_x_y_l406_40632


namespace NUMINAMATH_GPT_triangle_is_right_triangle_l406_40646

theorem triangle_is_right_triangle (A B C : ℝ) (hC_eq_A_plus_B : C = A + B) (h_angle_sum : A + B + C = 180) : C = 90 :=
by
  sorry

end NUMINAMATH_GPT_triangle_is_right_triangle_l406_40646


namespace NUMINAMATH_GPT_find_valid_pairs_l406_40664

theorem find_valid_pairs (x y : ℤ) : 
  (x^3 + y) % (x^2 + y^2) = 0 ∧ 
  (x + y^3) % (x^2 + y^2) = 0 ↔ 
  (x, y) = (1, 1) ∨ (x, y) = (1, 0) ∨ (x, y) = (1, -1) ∨ 
  (x, y) = (0, 1) ∨ (x, y) = (0, -1) ∨ (x, y) = (-1, 1) ∨ 
  (x, y) = (-1, 0) ∨ (x, y) = (-1, -1) :=
sorry

end NUMINAMATH_GPT_find_valid_pairs_l406_40664


namespace NUMINAMATH_GPT_circle_radius_of_equal_area_l406_40612

theorem circle_radius_of_equal_area (A B C D : Type) (r : ℝ) (π : ℝ) 
  (h_rect_area : 8 * 9 = 72)
  (h_circle_area : π * r ^ 2 = 36) :
  r = 6 / Real.sqrt π :=
by
  sorry

end NUMINAMATH_GPT_circle_radius_of_equal_area_l406_40612


namespace NUMINAMATH_GPT_range_of_absolute_difference_l406_40614

theorem range_of_absolute_difference : (∃ x : ℝ, y = |x + 4| - |x - 5|) → y ∈ [-9, 9] :=
sorry

end NUMINAMATH_GPT_range_of_absolute_difference_l406_40614


namespace NUMINAMATH_GPT_eval_expression_l406_40640

theorem eval_expression : 3 * 4^2 - (8 / 2) = 44 := by
  sorry

end NUMINAMATH_GPT_eval_expression_l406_40640


namespace NUMINAMATH_GPT_Papi_Calot_has_to_buy_141_plants_l406_40659

noncomputable def calc_number_of_plants : Nat :=
  let initial_plants := 7 * 18
  let additional_plants := 15
  initial_plants + additional_plants

theorem Papi_Calot_has_to_buy_141_plants :
  calc_number_of_plants = 141 :=
by
  sorry

end NUMINAMATH_GPT_Papi_Calot_has_to_buy_141_plants_l406_40659


namespace NUMINAMATH_GPT_total_settings_weight_l406_40675

/-- 
Each piece of silverware weighs 4 ounces and there are three pieces of silverware per setting.
Each plate weighs 12 ounces and there are two plates per setting.
Mason needs enough settings for 15 tables with 8 settings each, plus 20 backup settings in case of breakage.
Prove the total weight of all the settings equals 5040 ounces.
-/
theorem total_settings_weight
    (silverware_weight : ℝ := 4) (pieces_per_setting : ℕ := 3)
    (plate_weight : ℝ := 12) (plates_per_setting : ℕ := 2)
    (tables : ℕ := 15) (settings_per_table : ℕ := 8) (backup_settings : ℕ := 20) :
    let settings_needed := (tables * settings_per_table) + backup_settings
    let weight_per_setting := (silverware_weight * pieces_per_setting) + (plate_weight * plates_per_setting)
    settings_needed * weight_per_setting = 5040 :=
by
  sorry

end NUMINAMATH_GPT_total_settings_weight_l406_40675


namespace NUMINAMATH_GPT_katya_female_classmates_l406_40678

theorem katya_female_classmates (g b : ℕ) (h1 : b = 2 * g) (h2 : b = g + 7) :
  g - 1 = 6 :=
by
  sorry

end NUMINAMATH_GPT_katya_female_classmates_l406_40678


namespace NUMINAMATH_GPT_combined_weight_l406_40601

noncomputable def Jake_weight : ℕ := 196
noncomputable def Kendra_weight : ℕ := 94

-- Condition: If Jake loses 8 pounds, he will weigh twice as much as Kendra
axiom lose_8_pounds (j k : ℕ) : (j - 8 = 2 * k) → j = Jake_weight → k = Kendra_weight

-- To Prove: The combined weight of Jake and Kendra is 290 pounds
theorem combined_weight (j k : ℕ) (h₁ : j = Jake_weight) (h₂ : k = Kendra_weight) : j + k = 290 := 
by  sorry

end NUMINAMATH_GPT_combined_weight_l406_40601


namespace NUMINAMATH_GPT_complement_intersection_l406_40681

open Finset

def U : Finset ℕ := {1, 2, 3, 4, 5}
def A : Finset ℕ := {1, 3, 4}
def B : Finset ℕ := {3, 5}

theorem complement_intersection :
  (U \ (A ∩ B)) = {1, 2, 4, 5} :=
by sorry

end NUMINAMATH_GPT_complement_intersection_l406_40681


namespace NUMINAMATH_GPT_xiao_li_first_three_l406_40691

def q1_proba_correct (p1 p2 p3 : ℚ) : ℚ :=
  p1 * p2 * p3 + 
  (1 - p1) * p2 * p3 + 
  p1 * (1 - p2) * p3 + 
  p1 * p2 * (1 - p3)

theorem xiao_li_first_three (p1 p2 p3 : ℚ) (h1 : p1 = 3/4) (h2 : p2 = 1/2) (h3 : p3 = 5/6) :
  q1_proba_correct p1 p2 p3 = 11 / 24 := by
  rw [h1, h2, h3]
  sorry

end NUMINAMATH_GPT_xiao_li_first_three_l406_40691


namespace NUMINAMATH_GPT_difference_abs_eq_200_l406_40669

theorem difference_abs_eq_200 (x y : ℤ) (h1 : x + y = 250) (h2 : y = 225) : |x - y| = 200 := sorry

end NUMINAMATH_GPT_difference_abs_eq_200_l406_40669


namespace NUMINAMATH_GPT_half_day_division_l406_40622

theorem half_day_division : 
  ∃ (n m : ℕ), n * m = 43200 ∧ (∃! (k : ℕ), k = 60) := sorry

end NUMINAMATH_GPT_half_day_division_l406_40622


namespace NUMINAMATH_GPT_joan_total_seashells_l406_40689

def seashells_given_to_Sam : ℕ := 43
def seashells_left_with_Joan : ℕ := 27
def total_seashells_found := seashells_given_to_Sam + seashells_left_with_Joan

theorem joan_total_seashells : total_seashells_found = 70 := by
  -- proof goes here, but for now we will use sorry
  sorry

end NUMINAMATH_GPT_joan_total_seashells_l406_40689


namespace NUMINAMATH_GPT_area_of_rectangle_l406_40606

-- Given conditions
def shadedSquareArea : ℝ := 4
def nonShadedSquareArea : ℝ := shadedSquareArea
def largerSquareArea : ℝ := 4 * 4  -- Since the side length is twice the previous squares

-- Problem statement
theorem area_of_rectangle (shadedSquareArea nonShadedSquareArea largerSquareArea : ℝ) :
  shadedSquareArea + nonShadedSquareArea + largerSquareArea = 24 :=
sorry

end NUMINAMATH_GPT_area_of_rectangle_l406_40606


namespace NUMINAMATH_GPT_problem_distribution_count_l406_40674

theorem problem_distribution_count : 12^6 = 2985984 := 
by
  sorry

end NUMINAMATH_GPT_problem_distribution_count_l406_40674


namespace NUMINAMATH_GPT_petals_vs_wings_and_unvisited_leaves_l406_40624

def flowers_petals_leaves := 5
def petals_per_flower := 2
def bees_wings := 3
def wings_per_bee := 4
def leaves_per_flower := 3
def visits_per_bee := 2
def total_flowers := flowers_petals_leaves
def total_bees := bees_wings

def total_petals : ℕ := total_flowers * petals_per_flower
def total_wings : ℕ := total_bees * wings_per_bee
def more_wings_than_petals := total_wings - total_petals

def total_leaves : ℕ := total_flowers * leaves_per_flower
def total_visits : ℕ := total_bees * visits_per_bee
def leaves_per_visit := leaves_per_flower
def visited_leaves : ℕ := min total_leaves (total_visits * leaves_per_visit)
def unvisited_leaves : ℕ := total_leaves - visited_leaves

theorem petals_vs_wings_and_unvisited_leaves :
  more_wings_than_petals = 2 ∧ unvisited_leaves = 0 :=
by
  sorry

end NUMINAMATH_GPT_petals_vs_wings_and_unvisited_leaves_l406_40624


namespace NUMINAMATH_GPT_calc_neg_half_times_neg_two_pow_l406_40605

theorem calc_neg_half_times_neg_two_pow :
  - (0.5 ^ 20) * ((-2) ^ 26) = -64 := by
  sorry

end NUMINAMATH_GPT_calc_neg_half_times_neg_two_pow_l406_40605


namespace NUMINAMATH_GPT_solve_equation_l406_40600

theorem solve_equation : ∀ x : ℝ, (3 * (x - 2) + 1 = x - (2 * x - 1)) → x = 3 / 2 :=
by
  intro x
  intro h
  sorry

end NUMINAMATH_GPT_solve_equation_l406_40600


namespace NUMINAMATH_GPT_number_of_positive_integer_pairs_l406_40610

theorem number_of_positive_integer_pairs (x y : ℕ) : 
  (x^2 - y^2 = 77) → (0 < x) → (0 < y) → (∃ x1 y1 x2 y2, (x1, y1) ≠ (x2, y2) ∧ 
  x1^2 - y1^2 = 77 ∧ x2^2 - y2^2 = 77 ∧ 0 < x1 ∧ 0 < y1 ∧ 0 < x2 ∧ 0 < y2 ∧
  ∀ a b, (a^2 - b^2 = 77 → a = x1 ∧ b = y1) ∨ (a = x2 ∧ b = y2)) :=
sorry

end NUMINAMATH_GPT_number_of_positive_integer_pairs_l406_40610


namespace NUMINAMATH_GPT_irrational_of_sqrt_3_l406_40692

noncomputable def is_irritational (x : ℝ) : Prop :=
  ¬ (∃ p q : ℤ, q ≠ 0 ∧ x = p / q)

theorem irrational_of_sqrt_3 :
  is_irritational 0 = false ∧
  is_irritational 3.14 = false ∧
  is_irritational (-1) = false ∧
  is_irritational (Real.sqrt 3) = true := 
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_irrational_of_sqrt_3_l406_40692


namespace NUMINAMATH_GPT_number_of_girls_is_eleven_l406_40694

-- Conditions transformation
def boys_wear_red_hats : Prop := true
def girls_wear_yellow_hats : Prop := true
def teachers_wear_blue_hats : Prop := true
def cannot_see_own_hat : Prop := true
def little_qiang_sees_hats (x k : ℕ) : Prop := (x + 2) = (x + 2)
def little_hua_sees_hats (x k : ℕ) : Prop := x = 2 * k
def teacher_sees_hats (x k : ℕ) : Prop := k + 2 = (x + 2) + k - 11

-- Proof Statement
theorem number_of_girls_is_eleven (x k : ℕ) (h1 : boys_wear_red_hats)
  (h2 : girls_wear_yellow_hats) (h3 : teachers_wear_blue_hats)
  (h4 : cannot_see_own_hat) (hq : little_qiang_sees_hats x k)
  (hh : little_hua_sees_hats x k) (ht : teacher_sees_hats x k) : x = 11 :=
sorry

end NUMINAMATH_GPT_number_of_girls_is_eleven_l406_40694


namespace NUMINAMATH_GPT_bus_driver_regular_rate_l406_40667

theorem bus_driver_regular_rate (R : ℝ) (h1 : 976 = (40 * R) + (14.32 * (1.75 * R))) : 
  R = 15 := 
by
  sorry

end NUMINAMATH_GPT_bus_driver_regular_rate_l406_40667


namespace NUMINAMATH_GPT_evaluate_expression_l406_40604

-- Define x as given in the condition
def x : ℤ := 5

-- State the theorem we need to prove
theorem evaluate_expression : x^3 - 3 * x = 110 :=
by
  -- Proof will be provided here
  sorry

end NUMINAMATH_GPT_evaluate_expression_l406_40604


namespace NUMINAMATH_GPT_trapezoid_area_l406_40618

/-- Given that the area of the outer square is 36 square units and the area of the inner square is 
4 square units, the area of one of the four congruent trapezoids formed between the squares is 8 
square units. -/
theorem trapezoid_area (outer_square_area inner_square_area : ℕ) 
  (h_outer : outer_square_area = 36)
  (h_inner : inner_square_area = 4) : 
  (outer_square_area - inner_square_area) / 4 = 8 :=
by sorry

end NUMINAMATH_GPT_trapezoid_area_l406_40618


namespace NUMINAMATH_GPT_expansion_sum_l406_40683

theorem expansion_sum (A B C : ℤ) (h1 : A = (2 - 1)^10) (h2 : B = (2 + 0)^10) (h3 : C = -5120) : 
A + B + C = -4095 :=
by 
  sorry

end NUMINAMATH_GPT_expansion_sum_l406_40683


namespace NUMINAMATH_GPT_find_b_l406_40682

theorem find_b (a b : ℕ) (h1 : (a + b) % 10 = 5) (h2 : (a + b) % 7 = 4) : b = 2 := 
sorry

end NUMINAMATH_GPT_find_b_l406_40682


namespace NUMINAMATH_GPT_arithmetic_sequence_c_d_sum_l406_40608

theorem arithmetic_sequence_c_d_sum :
  let c := 19 + (11 - 3)
  let d := c + (11 - 3)
  c + d = 62 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_c_d_sum_l406_40608


namespace NUMINAMATH_GPT_Amith_current_age_l406_40672

variable (A D : ℕ)

theorem Amith_current_age
  (h1 : A - 5 = 3 * (D - 5))
  (h2 : A + 10 = 2 * (D + 10)) :
  A = 50 := by
  sorry

end NUMINAMATH_GPT_Amith_current_age_l406_40672


namespace NUMINAMATH_GPT_inequality_neg_3_l406_40617

theorem inequality_neg_3 (a b : ℝ) : a < b → -3 * a > -3 * b :=
by
  sorry

end NUMINAMATH_GPT_inequality_neg_3_l406_40617
