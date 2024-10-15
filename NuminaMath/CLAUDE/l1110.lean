import Mathlib

namespace NUMINAMATH_CALUDE_cupboard_cost_price_l1110_111014

/-- The cost price of the cupboard -/
def cost_price : ℝ := sorry

/-- The selling price of the cupboard -/
def selling_price : ℝ := 0.84 * cost_price

/-- The increased selling price -/
def increased_selling_price : ℝ := 1.16 * cost_price

theorem cupboard_cost_price : cost_price = 3750 := by
  have h1 : selling_price = 0.84 * cost_price := rfl
  have h2 : increased_selling_price = 1.16 * cost_price := rfl
  have h3 : increased_selling_price - selling_price = 1200 := sorry
  sorry

end NUMINAMATH_CALUDE_cupboard_cost_price_l1110_111014


namespace NUMINAMATH_CALUDE_stone_piles_total_l1110_111080

/-- Represents the number of stones in each pile -/
structure StonePiles where
  pile1 : ℕ
  pile2 : ℕ
  pile3 : ℕ
  pile4 : ℕ
  pile5 : ℕ

/-- The conditions of the stone pile problem -/
def satisfiesConditions (piles : StonePiles) : Prop :=
  piles.pile5 = 6 * piles.pile3 ∧
  piles.pile2 = 2 * (piles.pile3 + piles.pile5) ∧
  piles.pile1 = piles.pile5 / 3 ∧
  piles.pile1 = piles.pile4 - 10 ∧
  piles.pile4 = piles.pile2 / 2

/-- The theorem stating that any StonePiles satisfying the conditions will have a total of 60 stones -/
theorem stone_piles_total (piles : StonePiles) :
  satisfiesConditions piles →
  piles.pile1 + piles.pile2 + piles.pile3 + piles.pile4 + piles.pile5 = 60 := by
  sorry

end NUMINAMATH_CALUDE_stone_piles_total_l1110_111080


namespace NUMINAMATH_CALUDE_purely_imaginary_implies_m_eq_two_third_quadrant_implies_m_range_l1110_111084

-- Define the complex number z as a function of m
def z (m : ℝ) : ℂ := Complex.mk (m^2 + 2*m - 8) (m^2 + 2*m - 3)

-- Define the condition for z - m + 2 being purely imaginary
def is_purely_imaginary (m : ℝ) : Prop :=
  (z m - m + 2).re = 0 ∧ (z m - m + 2).im ≠ 0

-- Define the condition for point A being in the third quadrant
def in_third_quadrant (m : ℝ) : Prop :=
  (z m).re < 0 ∧ (z m).im < 0

-- Theorem 1: If z - m + 2 is purely imaginary, then m = 2
theorem purely_imaginary_implies_m_eq_two (m : ℝ) :
  is_purely_imaginary m → m = 2 :=
sorry

-- Theorem 2: If point A is in the third quadrant, then -3 < m < 1
theorem third_quadrant_implies_m_range (m : ℝ) :
  in_third_quadrant m → -3 < m ∧ m < 1 :=
sorry

end NUMINAMATH_CALUDE_purely_imaginary_implies_m_eq_two_third_quadrant_implies_m_range_l1110_111084


namespace NUMINAMATH_CALUDE_final_pen_count_l1110_111001

theorem final_pen_count (x : ℝ) (x_pos : x > 0) : 
  let after_mike := x + 0.5 * x
  let after_cindy := 2 * after_mike
  let given_to_sharon := 0.25 * after_cindy
  after_cindy - given_to_sharon = 2.25 * x :=
by sorry

end NUMINAMATH_CALUDE_final_pen_count_l1110_111001


namespace NUMINAMATH_CALUDE_sin_product_equals_one_sixteenth_l1110_111053

theorem sin_product_equals_one_sixteenth :
  Real.sin (12 * π / 180) * Real.sin (36 * π / 180) * Real.sin (72 * π / 180) * Real.sin (84 * π / 180) = 1 / 16 :=
by sorry

end NUMINAMATH_CALUDE_sin_product_equals_one_sixteenth_l1110_111053


namespace NUMINAMATH_CALUDE_rectangle_area_l1110_111028

theorem rectangle_area (b : ℝ) : 
  let square_area : ℝ := 2025
  let square_side : ℝ := Real.sqrt square_area
  let circle_radius : ℝ := square_side
  let rectangle_length : ℝ := (2 / 5) * circle_radius
  let rectangle_breadth : ℝ := b
  let rectangle_area : ℝ := rectangle_length * rectangle_breadth
  rectangle_area = 18 * b :=
by sorry

end NUMINAMATH_CALUDE_rectangle_area_l1110_111028


namespace NUMINAMATH_CALUDE_opposite_sign_sum_zero_l1110_111031

theorem opposite_sign_sum_zero (a b : ℝ) : 
  (|a - 2| + (b + 1)^2 = 0) → (a - b = 3) := by
  sorry

end NUMINAMATH_CALUDE_opposite_sign_sum_zero_l1110_111031


namespace NUMINAMATH_CALUDE_percentage_equation_l1110_111023

theorem percentage_equation (x : ℝ) : 0.65 * x = 0.20 * 552.50 → x = 170 := by
  sorry

end NUMINAMATH_CALUDE_percentage_equation_l1110_111023


namespace NUMINAMATH_CALUDE_cube_root_equation_solution_l1110_111086

theorem cube_root_equation_solution :
  ∃! x : ℝ, (5 - x / 3) ^ (1/3 : ℝ) = -4 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_cube_root_equation_solution_l1110_111086


namespace NUMINAMATH_CALUDE_leftover_grass_seed_coverage_l1110_111078

/-- Proves the leftover grass seed coverage for Drew's lawn -/
theorem leftover_grass_seed_coverage 
  (lawn_length : ℕ) 
  (lawn_width : ℕ) 
  (seed_bags : ℕ) 
  (coverage_per_bag : ℕ) :
  lawn_length = 22 →
  lawn_width = 36 →
  seed_bags = 4 →
  coverage_per_bag = 250 →
  seed_bags * coverage_per_bag - lawn_length * lawn_width = 208 :=
by sorry

end NUMINAMATH_CALUDE_leftover_grass_seed_coverage_l1110_111078


namespace NUMINAMATH_CALUDE_line_inclination_angle_l1110_111060

theorem line_inclination_angle (x1 y1 x2 y2 : ℝ) :
  x1 = 1 →
  y1 = 1 →
  x2 = 2 →
  y2 = 1 + Real.sqrt 3 →
  ∃ θ : ℝ, θ * (π / 180) = π / 3 ∧ Real.tan θ = (y2 - y1) / (x2 - x1) := by
  sorry

end NUMINAMATH_CALUDE_line_inclination_angle_l1110_111060


namespace NUMINAMATH_CALUDE_min_value_expression_l1110_111085

theorem min_value_expression (x y : ℝ) 
  (h1 : |x| < 1) 
  (h2 : |y| < 2) 
  (h3 : x * y = 1) : 
  (1 / (1 - x^2)) + (4 / (4 - y^2)) ≥ 4 ∧ 
  ∃ (x₀ y₀ : ℝ), |x₀| < 1 ∧ |y₀| < 2 ∧ x₀ * y₀ = 1 ∧ 
    (1 / (1 - x₀^2)) + (4 / (4 - y₀^2)) = 4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l1110_111085


namespace NUMINAMATH_CALUDE_liquid_film_radius_l1110_111092

theorem liquid_film_radius (volume : ℝ) (thickness : ℝ) (radius : ℝ) : 
  volume = 320 →
  thickness = 0.05 →
  volume = π * radius^2 * thickness →
  radius = Real.sqrt (6400 / π) := by
sorry

end NUMINAMATH_CALUDE_liquid_film_radius_l1110_111092


namespace NUMINAMATH_CALUDE_trigonometric_equation_solutions_l1110_111055

theorem trigonometric_equation_solutions (a : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x ∈ Set.Icc 0 (Real.pi / 2) ∧ y ∈ Set.Icc 0 (Real.pi / 2) ∧ 
   Real.cos (2 * x) + Real.sqrt 3 * Real.sin (2 * x) = a + 1 ∧
   Real.cos (2 * y) + Real.sqrt 3 * Real.sin (2 * y) = a + 1) →
  0 ≤ a ∧ a < 1 :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solutions_l1110_111055


namespace NUMINAMATH_CALUDE_a_equals_one_necessary_not_sufficient_l1110_111000

-- Define sets A and B
def A : Set ℝ := {x : ℝ | x ≤ 1}
def B (a : ℝ) : Set ℝ := {x : ℝ | x ≥ a}

-- Statement of the theorem
theorem a_equals_one_necessary_not_sufficient :
  (∃ a : ℝ, a ≠ 1 ∧ A ∪ B a = Set.univ) ∧
  (∀ a : ℝ, a = 1 → A ∪ B a = Set.univ) :=
by sorry

end NUMINAMATH_CALUDE_a_equals_one_necessary_not_sufficient_l1110_111000


namespace NUMINAMATH_CALUDE_circle_fraction_l1110_111051

theorem circle_fraction (n : ℕ) (m : ℕ) (h1 : n > 0) (h2 : m ≤ n) :
  (m : ℚ) / n = m * (1 / n) :=
by sorry

#check circle_fraction

end NUMINAMATH_CALUDE_circle_fraction_l1110_111051


namespace NUMINAMATH_CALUDE_second_sibling_age_difference_l1110_111057

theorem second_sibling_age_difference (Y x : ℕ) : 
  Y = 17 → 
  (Y + (Y + x) + (Y + 4) + (Y + 7)) / 4 = 21 → 
  x = 5 := by
sorry

end NUMINAMATH_CALUDE_second_sibling_age_difference_l1110_111057


namespace NUMINAMATH_CALUDE_fraction_evaluation_l1110_111064

theorem fraction_evaluation (a b c : ℝ) (ha : a = 4) (hb : b = -4) (hc : c = 3) :
  3 / (a + b + c) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l1110_111064


namespace NUMINAMATH_CALUDE_unique_solution_exponential_equation_l1110_111072

theorem unique_solution_exponential_equation :
  ∃! x : ℝ, (2 : ℝ) ^ (6 * x + 3) * (4 : ℝ) ^ (3 * x + 6) = (8 : ℝ) ^ (-4 * x + 5) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_exponential_equation_l1110_111072


namespace NUMINAMATH_CALUDE_move_right_four_units_l1110_111077

/-- A point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Move a point horizontally in a Cartesian coordinate system -/
def moveHorizontal (p : Point) (units : ℝ) : Point :=
  { x := p.x + units, y := p.y }

/-- The theorem stating that moving (-2, 3) 4 units right results in (2, 3) -/
theorem move_right_four_units :
  let initial : Point := { x := -2, y := 3 }
  let final : Point := moveHorizontal initial 4
  final.x = 2 ∧ final.y = 3 := by
  sorry

end NUMINAMATH_CALUDE_move_right_four_units_l1110_111077


namespace NUMINAMATH_CALUDE_quadratic_decreasing_threshold_l1110_111047

/-- Represents a quadratic function of the form ax^2 - 2ax + 1 -/
def QuadraticFunction (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 2 * a * x + 1

/-- Proves that for a quadratic function f(x) = ax^2 - 2ax + 1 where a < 0,
    the minimum value of m for which f(x) is decreasing for all x > m is 1 -/
theorem quadratic_decreasing_threshold (a : ℝ) (h : a < 0) :
  ∃ m : ℝ, m = 1 ∧ ∀ x > m, ∀ y > x,
    QuadraticFunction a y < QuadraticFunction a x :=
by sorry

end NUMINAMATH_CALUDE_quadratic_decreasing_threshold_l1110_111047


namespace NUMINAMATH_CALUDE_largest_angle_in_special_triangle_l1110_111082

/-- Given a triangle with angles in the ratio 3:4:9 and an external angle equal to the smallest 
    internal angle attached at the largest angle, prove that the largest internal angle is 101.25°. -/
theorem largest_angle_in_special_triangle :
  ∀ (a b c : ℝ), 
    a > 0 ∧ b > 0 ∧ c > 0 →  -- Angles are positive
    b = (4/3) * a ∧ c = 3 * a →  -- Ratio of angles is 3:4:9
    a + b + c = 180 →  -- Sum of internal angles is 180°
    c + a = 12 * a →  -- External angle equals smallest internal angle
    c = 101.25 := by sorry

end NUMINAMATH_CALUDE_largest_angle_in_special_triangle_l1110_111082


namespace NUMINAMATH_CALUDE_annual_income_proof_l1110_111096

/-- Calculates the yearly simple interest income given principal and rate -/
def simple_interest (principal : ℝ) (rate : ℝ) : ℝ :=
  principal * rate

theorem annual_income_proof (total_amount : ℝ) (part1 : ℝ) (rate1 : ℝ) (rate2 : ℝ) 
  (h1 : total_amount = 2500)
  (h2 : part1 = 1000)
  (h3 : rate1 = 0.05)
  (h4 : rate2 = 0.06) :
  simple_interest part1 rate1 + simple_interest (total_amount - part1) rate2 = 140 := by
  sorry

end NUMINAMATH_CALUDE_annual_income_proof_l1110_111096


namespace NUMINAMATH_CALUDE_carly_running_schedule_l1110_111059

def running_schedule (week1 : ℚ) (week2_multiplier : ℚ) (week2_extra : ℚ) (week3_multiplier : ℚ) (week4_reduction : ℚ) : ℚ → ℚ
  | 1 => week1
  | 2 => week1 * week2_multiplier + week2_extra
  | 3 => (week1 * week2_multiplier + week2_extra) * week3_multiplier
  | 4 => (week1 * week2_multiplier + week2_extra) * week3_multiplier - week4_reduction
  | _ => 0

theorem carly_running_schedule :
  running_schedule 2 2 3 (9/7) 5 4 = 4 := by sorry

end NUMINAMATH_CALUDE_carly_running_schedule_l1110_111059


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1110_111008

theorem complex_equation_solution (z : ℂ) : (3 - 4*I)*z = 5*I → z = 4/5 + 3/5*I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1110_111008


namespace NUMINAMATH_CALUDE_union_of_sets_l1110_111097

theorem union_of_sets (M N : Set ℕ) : 
  M = {0, 2, 3} → N = {1, 3} → M ∪ N = {0, 1, 2, 3} := by
  sorry

end NUMINAMATH_CALUDE_union_of_sets_l1110_111097


namespace NUMINAMATH_CALUDE_mean_of_sequence_mean_of_sequence_is_17_75_l1110_111048

theorem mean_of_sequence : Real → Prop :=
  fun mean =>
    let sequence := [1^2, 2^2, 3^2, 4^2, 5^2, 6^2, 7^2, 2]
    mean = (sequence.sum : Real) / sequence.length ∧ mean = 17.75

-- The proof is omitted
theorem mean_of_sequence_is_17_75 : ∃ mean, mean_of_sequence mean :=
  sorry

end NUMINAMATH_CALUDE_mean_of_sequence_mean_of_sequence_is_17_75_l1110_111048


namespace NUMINAMATH_CALUDE_log_inequality_l1110_111010

theorem log_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  Real.log ((Real.sqrt a + Real.sqrt b) / 2) > Real.log (Real.sqrt (a + b) / 2) := by
  sorry

end NUMINAMATH_CALUDE_log_inequality_l1110_111010


namespace NUMINAMATH_CALUDE_josh_siblings_count_josh_problem_l1110_111058

theorem josh_siblings_count (initial_candies : ℕ) 
  (candies_per_sibling : ℕ) (eat_himself : ℕ) (remaining_candies : ℕ) : ℕ :=
  let siblings_count := (initial_candies - 2 * (remaining_candies + eat_himself)) / (2 * candies_per_sibling)
  siblings_count

theorem josh_problem :
  josh_siblings_count 100 10 16 19 = 3 := by
  sorry

end NUMINAMATH_CALUDE_josh_siblings_count_josh_problem_l1110_111058


namespace NUMINAMATH_CALUDE_six_by_six_tiling_impossible_l1110_111069

/-- Represents a chessboard -/
structure Chessboard :=
  (rows : Nat)
  (cols : Nat)

/-- Represents a tile -/
structure Tile :=
  (length : Nat)
  (width : Nat)

/-- Represents a tiling configuration -/
structure TilingConfig :=
  (board : Chessboard)
  (tile : Tile)
  (num_tiles : Nat)

/-- Predicate to check if a tiling configuration is valid -/
def is_valid_tiling (config : TilingConfig) : Prop :=
  config.board.rows * config.board.cols = config.tile.length * config.tile.width * config.num_tiles

/-- Theorem stating that a 6x6 chessboard cannot be tiled with nine 1x4 tiles -/
theorem six_by_six_tiling_impossible :
  ¬ is_valid_tiling { board := { rows := 6, cols := 6 },
                      tile := { length := 1, width := 4 },
                      num_tiles := 9 } :=
by sorry

end NUMINAMATH_CALUDE_six_by_six_tiling_impossible_l1110_111069


namespace NUMINAMATH_CALUDE_sum_abc_equals_42_l1110_111029

theorem sum_abc_equals_42 
  (a b c : ℕ+) 
  (h1 : a * b + c = 41)
  (h2 : b * c + a = 41)
  (h3 : a * c + b = 41) : 
  a + b + c = 42 := by
  sorry

end NUMINAMATH_CALUDE_sum_abc_equals_42_l1110_111029


namespace NUMINAMATH_CALUDE_correct_operation_l1110_111021

theorem correct_operation (a b : ℝ) : 2 * a^2 * b * (4 * a * b^3) = 8 * a^3 * b^4 := by
  sorry

end NUMINAMATH_CALUDE_correct_operation_l1110_111021


namespace NUMINAMATH_CALUDE_cube_divided_by_self_l1110_111067

theorem cube_divided_by_self (a : ℝ) (h : a ≠ 0) : a^3 / a = a^2 := by
  sorry

end NUMINAMATH_CALUDE_cube_divided_by_self_l1110_111067


namespace NUMINAMATH_CALUDE_soccer_league_games_l1110_111093

/-- Proves that in a soccer league with 11 teams and 55 total games, each team plays others 2 times -/
theorem soccer_league_games (num_teams : ℕ) (total_games : ℕ) (games_per_pair : ℕ) : 
  num_teams = 11 → 
  total_games = 55 → 
  total_games = (num_teams * (num_teams - 1) * games_per_pair) / 2 → 
  games_per_pair = 2 := by
sorry

end NUMINAMATH_CALUDE_soccer_league_games_l1110_111093


namespace NUMINAMATH_CALUDE_product_of_negative_real_part_solutions_l1110_111004

theorem product_of_negative_real_part_solutions :
  let solutions : List (ℂ) := [2 * (Complex.exp (Complex.I * Real.pi / 4)),
                               2 * (Complex.exp (Complex.I * 3 * Real.pi / 4)),
                               2 * (Complex.exp (Complex.I * 5 * Real.pi / 4)),
                               2 * (Complex.exp (Complex.I * 7 * Real.pi / 4))]
  let negative_real_part_solutions := solutions.filter (fun z => z.re < 0)
  ∀ z ∈ solutions, z^4 = -16 →
  negative_real_part_solutions.prod = 4 := by
sorry

end NUMINAMATH_CALUDE_product_of_negative_real_part_solutions_l1110_111004


namespace NUMINAMATH_CALUDE_austin_hourly_rate_l1110_111071

def hours_per_week : ℕ := 6
def weeks_worked : ℕ := 6
def bicycle_cost : ℕ := 180

theorem austin_hourly_rate :
  ∃ (rate : ℚ), rate * (hours_per_week * weeks_worked : ℚ) = bicycle_cost ∧ rate = 5 := by
  sorry

end NUMINAMATH_CALUDE_austin_hourly_rate_l1110_111071


namespace NUMINAMATH_CALUDE_intersection_irrationality_l1110_111025

theorem intersection_irrationality (p q : ℤ) (hp : Odd p) (hq : Odd q) :
  ∀ x : ℚ, x^2 - 2*p*x + 2*q ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_intersection_irrationality_l1110_111025


namespace NUMINAMATH_CALUDE_tom_bricks_count_l1110_111088

/-- The number of bricks Tom needs to buy -/
def num_bricks : ℕ := 1000

/-- The cost of a brick at full price -/
def full_price : ℚ := 1/2

/-- The total amount Tom spends -/
def total_spent : ℚ := 375

theorem tom_bricks_count :
  (num_bricks / 2 : ℚ) * (full_price / 2) + (num_bricks / 2 : ℚ) * full_price = total_spent :=
sorry

end NUMINAMATH_CALUDE_tom_bricks_count_l1110_111088


namespace NUMINAMATH_CALUDE_sum_of_squares_l1110_111054

theorem sum_of_squares (x y z a b c k : ℝ) 
  (h1 : x * y = k * a)
  (h2 : x * z = b)
  (h3 : y * z = c)
  (h4 : k ≠ 0)
  (h5 : x ≠ 0)
  (h6 : y ≠ 0)
  (h7 : z ≠ 0)
  (h8 : a ≠ 0)
  (h9 : b ≠ 0)
  (h10 : c ≠ 0) :
  x^2 + y^2 + z^2 = (k * (a * b + a * c + b * c)) / (a * b * c) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l1110_111054


namespace NUMINAMATH_CALUDE_children_neither_happy_nor_sad_l1110_111030

theorem children_neither_happy_nor_sad 
  (total_children : ℕ)
  (happy_children : ℕ)
  (sad_children : ℕ)
  (boys : ℕ)
  (girls : ℕ)
  (happy_boys : ℕ)
  (sad_girls : ℕ)
  (neither_boys : ℕ)
  (h1 : total_children = 60)
  (h2 : happy_children = 30)
  (h3 : sad_children = 10)
  (h4 : boys = 19)
  (h5 : girls = 41)
  (h6 : happy_boys = 6)
  (h7 : sad_girls = 4)
  (h8 : neither_boys = 7)
  : total_children - happy_children - sad_children = 20 := by
  sorry

end NUMINAMATH_CALUDE_children_neither_happy_nor_sad_l1110_111030


namespace NUMINAMATH_CALUDE_mary_fruit_cost_l1110_111042

/-- Calculates the total cost of fruits with a discount applied -/
def fruitCost (applePrice orangePrice bananaPrice : ℚ) 
              (appleCount orangeCount bananaCount : ℕ) 
              (fruitPerDiscount : ℕ) (discountAmount : ℚ) : ℚ :=
  let totalFruits := appleCount + orangeCount + bananaCount
  let subtotal := applePrice * appleCount + orangePrice * orangeCount + bananaPrice * bananaCount
  let discountCount := totalFruits / fruitPerDiscount
  subtotal - (discountCount * discountAmount)

/-- Theorem stating that Mary will pay $15 for her fruits -/
theorem mary_fruit_cost : 
  fruitCost 1 2 3 5 3 2 5 1 = 15 := by
  sorry

end NUMINAMATH_CALUDE_mary_fruit_cost_l1110_111042


namespace NUMINAMATH_CALUDE_flood_damage_conversion_l1110_111075

/-- Converts Australian dollars to US dollars given an exchange rate -/
def aud_to_usd (aud_amount : ℝ) (exchange_rate : ℝ) : ℝ :=
  aud_amount * exchange_rate

/-- Theorem stating the conversion of flood damage from AUD to USD -/
theorem flood_damage_conversion (damage_aud : ℝ) (exchange_rate : ℝ) 
  (h1 : damage_aud = 45000000)
  (h2 : exchange_rate = 0.7) :
  aud_to_usd damage_aud exchange_rate = 31500000 :=
by sorry

end NUMINAMATH_CALUDE_flood_damage_conversion_l1110_111075


namespace NUMINAMATH_CALUDE_correct_calculation_l1110_111043

theorem correct_calculation (m n : ℝ) : 4*m + 2*n - (n - m) = 5*m + n := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l1110_111043


namespace NUMINAMATH_CALUDE_pants_cost_is_250_l1110_111040

/-- The cost of each pair of pants given the total cost of t-shirts and pants, 
    the number of t-shirts and pants, and the cost of each t-shirt. -/
def cost_of_pants (total_cost : ℕ) (num_tshirts : ℕ) (num_pants : ℕ) (tshirt_cost : ℕ) : ℕ :=
  (total_cost - num_tshirts * tshirt_cost) / num_pants

/-- Theorem stating that the cost of each pair of pants is 250 
    given the conditions in the problem. -/
theorem pants_cost_is_250 :
  cost_of_pants 1500 5 4 100 = 250 := by
  sorry

end NUMINAMATH_CALUDE_pants_cost_is_250_l1110_111040


namespace NUMINAMATH_CALUDE_stratified_sample_small_supermarkets_l1110_111033

/-- Calculates the number of small supermarkets in a stratified sample -/
def smallSupermarketsInSample (totalSupermarkets : ℕ) (smallSupermarkets : ℕ) (sampleSize : ℕ) : ℕ :=
  (smallSupermarkets * sampleSize) / totalSupermarkets

theorem stratified_sample_small_supermarkets :
  smallSupermarketsInSample 3000 2100 100 = 70 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sample_small_supermarkets_l1110_111033


namespace NUMINAMATH_CALUDE_min_abs_z_l1110_111007

theorem min_abs_z (z : ℂ) (h : Complex.abs (z - 8) + Complex.abs (z - Complex.I * 7) = 15) :
  ∃ (w : ℂ), Complex.abs w ≤ Complex.abs z ∧ Complex.abs (w - 8) + Complex.abs (w - Complex.I * 7) = 15 ∧ Complex.abs w = 56 / 15 :=
by
  sorry

end NUMINAMATH_CALUDE_min_abs_z_l1110_111007


namespace NUMINAMATH_CALUDE_lcm_hcf_problem_l1110_111090

theorem lcm_hcf_problem (a b : ℕ+) : 
  Nat.lcm a b = 2310 → 
  Nat.gcd a b = 47 → 
  b = 517 → 
  a = 210 := by sorry

end NUMINAMATH_CALUDE_lcm_hcf_problem_l1110_111090


namespace NUMINAMATH_CALUDE_dog_food_theorem_l1110_111099

/-- The number of cups of dog food in a bag that lasts 16 days -/
def cups_in_bag (morning_cups : ℕ) (evening_cups : ℕ) (days : ℕ) : ℕ :=
  (morning_cups + evening_cups) * days

/-- Theorem stating that a bag lasting 16 days contains 32 cups of dog food -/
theorem dog_food_theorem :
  cups_in_bag 1 1 16 = 32 := by
  sorry

end NUMINAMATH_CALUDE_dog_food_theorem_l1110_111099


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l1110_111083

theorem imaginary_part_of_z (i : ℂ) (h : i * i = -1) : 
  Complex.im ((1 - 2*i) / (2 + i)) = -1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l1110_111083


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l1110_111091

theorem absolute_value_equation_solution :
  ∀ x : ℝ, |1/2 * x| = 2 ↔ x = 4 ∨ x = -4 := by
sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l1110_111091


namespace NUMINAMATH_CALUDE_inserted_sequence_theorem_l1110_111052

/-- Given a sequence, insert_between inserts n elements between each pair of adjacent elements -/
def insert_between (seq : ℕ → α) (n : ℕ) : ℕ → α :=
  λ k => if k % (n + 1) = 0 then seq (k / (n + 1) + 1) else seq (k / (n + 1) + 1)

theorem inserted_sequence_theorem (original_seq : ℕ → α) :
  (insert_between original_seq 3) 69 = original_seq 18 := by
  sorry

end NUMINAMATH_CALUDE_inserted_sequence_theorem_l1110_111052


namespace NUMINAMATH_CALUDE_cherry_soda_count_l1110_111012

theorem cherry_soda_count (total_cans : ℕ) (orange_ratio : ℕ) (cherry_count : ℕ) : 
  total_cans = 24 →
  orange_ratio = 2 →
  total_cans = cherry_count + orange_ratio * cherry_count →
  cherry_count = 8 := by
  sorry

end NUMINAMATH_CALUDE_cherry_soda_count_l1110_111012


namespace NUMINAMATH_CALUDE_square_sum_factorization_l1110_111017

theorem square_sum_factorization (x y : ℝ) : x^2 + 2*x*y + y^2 = (x + y)^2 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_factorization_l1110_111017


namespace NUMINAMATH_CALUDE_pears_left_l1110_111039

theorem pears_left (keith_picked mike_picked keith_gave_away : ℕ) 
  (h1 : keith_picked = 47)
  (h2 : mike_picked = 12)
  (h3 : keith_gave_away = 46) :
  keith_picked - keith_gave_away + mike_picked = 13 := by
sorry

end NUMINAMATH_CALUDE_pears_left_l1110_111039


namespace NUMINAMATH_CALUDE_light_flash_interval_l1110_111011

/-- Given a light that flashes 180 times in ¾ of an hour, 
    prove that the time between flashes is 15 seconds. -/
theorem light_flash_interval (flashes : ℕ) (time : ℚ) 
  (h1 : flashes = 180) 
  (h2 : time = 3/4) : 
  (time * 3600) / flashes = 15 := by
  sorry

end NUMINAMATH_CALUDE_light_flash_interval_l1110_111011


namespace NUMINAMATH_CALUDE_worker_completion_time_l1110_111081

/-- Given workers A and B, where A can complete a job in 15 days,
    A works for 5 days, and B finishes the remaining work in 12 days,
    prove that B alone can complete the entire job in 18 days. -/
theorem worker_completion_time (a_total_days b_remaining_days : ℕ) 
    (h1 : a_total_days = 15)
    (h2 : b_remaining_days = 12) : 
  (18 : ℚ) = (b_remaining_days : ℚ) / ((a_total_days - 5 : ℚ) / a_total_days) := by
  sorry

end NUMINAMATH_CALUDE_worker_completion_time_l1110_111081


namespace NUMINAMATH_CALUDE_negation_of_exists_leq_negation_of_proposition_l1110_111034

theorem negation_of_exists_leq (p : ℝ → Prop) :
  (¬ ∃ x, p x) ↔ ∀ x, ¬ p x :=
by sorry

theorem negation_of_proposition :
  (¬ ∃ x : ℝ, Real.exp x - x - 1 ≤ 0) ↔ (∀ x : ℝ, Real.exp x - x - 1 > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_exists_leq_negation_of_proposition_l1110_111034


namespace NUMINAMATH_CALUDE_f_divisible_by_8_l1110_111095

def f (n : ℕ) : ℕ := 5^n + 2 * 3^(n-1) + 1

theorem f_divisible_by_8 (n : ℕ) (h : n > 0) : 
  ∃ k : ℕ, f n = 8 * k := by
  sorry

end NUMINAMATH_CALUDE_f_divisible_by_8_l1110_111095


namespace NUMINAMATH_CALUDE_congruence_solution_l1110_111066

theorem congruence_solution (n : ℤ) : 
  -20 ≤ n ∧ n ≤ 20 ∧ n ≡ -127 [ZMOD 7] → n = -13 ∨ n = 1 ∨ n = 15 := by
  sorry

end NUMINAMATH_CALUDE_congruence_solution_l1110_111066


namespace NUMINAMATH_CALUDE_optimal_truck_loading_l1110_111015

theorem optimal_truck_loading (total_load : ℕ) (large_capacity : ℕ) (small_capacity : ℕ)
  (h_total : total_load = 134)
  (h_large : large_capacity = 15)
  (h_small : small_capacity = 7) :
  ∃ (large_count small_count : ℕ),
    large_count * large_capacity + small_count * small_capacity = total_load ∧
    large_count = 8 ∧
    small_count = 2 ∧
    ∀ (l s : ℕ), l * large_capacity + s * small_capacity = total_load →
      l + s ≥ large_count + small_count :=
by sorry

end NUMINAMATH_CALUDE_optimal_truck_loading_l1110_111015


namespace NUMINAMATH_CALUDE_rotated_solid_properties_l1110_111073

/-- A right-angled triangle with sides 3, 4, and 5 -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  right_angled : a^2 + b^2 = c^2
  side_a : a = 3
  side_b : b = 4
  side_c : c = 5

/-- The solid formed by rotating the triangle around its hypotenuse -/
def RotatedSolid (t : RightTriangle) : Prop :=
  ∃ (surface_area volume : ℝ),
    surface_area = 84/5 * Real.pi ∧
    volume = 48/5 * Real.pi

/-- Theorem stating the surface area and volume of the rotated solid -/
theorem rotated_solid_properties (t : RightTriangle) :
  RotatedSolid t := by sorry

end NUMINAMATH_CALUDE_rotated_solid_properties_l1110_111073


namespace NUMINAMATH_CALUDE_victory_points_value_l1110_111063

/-- Represents the number of points awarded for different match outcomes -/
structure PointSystem where
  victory : ℕ
  draw : ℕ
  defeat : ℕ

/-- Represents the state of a team's performance in the tournament -/
structure TeamPerformance where
  totalMatches : ℕ
  playedMatches : ℕ
  currentPoints : ℕ
  pointsNeeded : ℕ
  minWinsNeeded : ℕ

/-- The theorem stating the point value for a victory -/
theorem victory_points_value (ps : PointSystem) (tp : TeamPerformance) : 
  ps.draw = 1 ∧ 
  ps.defeat = 0 ∧
  tp.totalMatches = 20 ∧
  tp.playedMatches = 5 ∧
  tp.currentPoints = 12 ∧
  tp.pointsNeeded = 40 ∧
  tp.minWinsNeeded = 7 →
  ps.victory = 4 := by
  sorry

end NUMINAMATH_CALUDE_victory_points_value_l1110_111063


namespace NUMINAMATH_CALUDE_modulo_equivalence_56234_l1110_111009

theorem modulo_equivalence_56234 :
  ∃! n : ℤ, 0 ≤ n ∧ n < 23 ∧ 56234 ≡ n [ZMOD 23] ∧ n = 1 := by
  sorry

end NUMINAMATH_CALUDE_modulo_equivalence_56234_l1110_111009


namespace NUMINAMATH_CALUDE_orthocenter_locus_l1110_111070

noncomputable section

/-- A circle in a 2D plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A triangle in a 2D plane --/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The orthocenter of a triangle --/
def orthocenter (t : Triangle) : ℝ × ℝ := sorry

/-- Check if a point is inside a circle --/
def is_inside (p : ℝ × ℝ) (c : Circle) : Prop := sorry

/-- Check if a triangle is inscribed in a circle --/
def is_inscribed (t : Triangle) (c : Circle) : Prop := sorry

/-- The theorem stating the locus of orthocenters --/
theorem orthocenter_locus (c : Circle) :
  ∀ t : Triangle, is_inscribed t c →
    is_inside (orthocenter t) { center := c.center, radius := 3 * c.radius } :=
sorry

end NUMINAMATH_CALUDE_orthocenter_locus_l1110_111070


namespace NUMINAMATH_CALUDE_sum_of_seventh_row_l1110_111068

-- Define the sum function for the triangular array
def f : ℕ → ℕ
  | 0 => 0  -- Base case: f(0) = 0 (not used in the problem, but needed for recursion)
  | 1 => 2  -- Base case: f(1) = 2
  | n + 1 => 2 * f n + 4  -- Recursive case: f(n+1) = 2f(n) + 4

-- Theorem statement
theorem sum_of_seventh_row : f 7 = 284 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_seventh_row_l1110_111068


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1110_111044

theorem quadratic_equation_solution : ∃ (x₁ x₂ : ℝ), 
  (x₁ = -3/2 ∧ 2*x₁^2 - 4*x₁ = 6 - 3*x₁) ∧
  (x₂ = 2 ∧ 2*x₂^2 - 4*x₂ = 6 - 3*x₂) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1110_111044


namespace NUMINAMATH_CALUDE_grape_juice_mixture_problem_l1110_111022

theorem grape_juice_mixture_problem (initial_volume : ℝ) (added_pure_juice : ℝ) (final_percentage : ℝ) :
  initial_volume = 30 →
  added_pure_juice = 10 →
  final_percentage = 0.325 →
  ∃ initial_percentage : ℝ,
    initial_percentage * initial_volume + added_pure_juice = 
    (initial_volume + added_pure_juice) * final_percentage ∧
    initial_percentage = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_grape_juice_mixture_problem_l1110_111022


namespace NUMINAMATH_CALUDE_ball_count_proof_l1110_111038

theorem ball_count_proof (a : ℕ) (h1 : a > 0) (h2 : 3 ≤ a) : 
  (3 : ℚ) / a = 1 / 4 → a = 12 := by
sorry

end NUMINAMATH_CALUDE_ball_count_proof_l1110_111038


namespace NUMINAMATH_CALUDE_plane_equation_proof_l1110_111062

/-- A plane in 3D space represented by the equation Ax + By + Cz + D = 0 --/
structure Plane where
  A : ℤ
  B : ℤ
  C : ℤ
  D : ℤ
  A_pos : A > 0
  gcd_one : Nat.gcd (Int.natAbs A) (Nat.gcd (Int.natAbs B) (Nat.gcd (Int.natAbs C) (Int.natAbs D))) = 1

/-- A point in 3D space --/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def parallelPlanes (p1 p2 : Plane) : Prop :=
  ∃ (k : ℚ), k ≠ 0 ∧ p1.A = k * p2.A ∧ p1.B = k * p2.B ∧ p1.C = k * p2.C

def pointOnPlane (point : Point3D) (plane : Plane) : Prop :=
  plane.A * point.x + plane.B * point.y + plane.C * point.z + plane.D = 0

theorem plane_equation_proof (given_plane : Plane) (point : Point3D) :
  given_plane.A = 2 ∧ given_plane.B = -1 ∧ given_plane.C = 3 ∧ given_plane.D = 5 →
  point.x = 2 ∧ point.y = 3 ∧ point.z = -4 →
  ∃ (result_plane : Plane),
    parallelPlanes result_plane given_plane ∧
    pointOnPlane point result_plane ∧
    result_plane.A = 2 ∧ result_plane.B = -1 ∧ result_plane.C = 3 ∧ result_plane.D = 11 :=
sorry

end NUMINAMATH_CALUDE_plane_equation_proof_l1110_111062


namespace NUMINAMATH_CALUDE_fraction_difference_l1110_111035

theorem fraction_difference (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x - y = x / y) :
  1 / x - 1 / y = -(1 / y^2) := by
sorry

end NUMINAMATH_CALUDE_fraction_difference_l1110_111035


namespace NUMINAMATH_CALUDE_set_operations_and_intersection_l1110_111024

open Set

-- Define the sets A, B, and C
def A : Set ℝ := {x | 1 ≤ x ∧ x < 7}
def B : Set ℝ := {x | 2 < x ∧ x < 10}
def C (a : ℝ) : Set ℝ := {x | x < a}

-- Theorem statement
theorem set_operations_and_intersection :
  (A ∪ B = {x | 1 ≤ x ∧ x < 10}) ∧
  (Bᶜ = {x | x ≤ 2 ∨ x ≥ 10}) ∧
  (∀ a : ℝ, (A ∩ C a).Nonempty → a > 1) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_and_intersection_l1110_111024


namespace NUMINAMATH_CALUDE_tiling_symmetry_l1110_111050

/-- A rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Tiling relation between rectangles -/
def CanTile (A B : Rectangle) : Prop :=
  ∃ (n m : ℕ), n * A.width = m * B.width ∧ n * A.height = m * B.height

/-- Similarity relation between rectangles -/
def IsSimilarTo (A B : Rectangle) : Prop :=
  ∃ (k : ℝ), k > 0 ∧ A.width = k * B.width ∧ A.height = k * B.height

/-- Main theorem: If a rectangle similar to A can be tiled with B, 
    then a rectangle similar to B can be tiled with A -/
theorem tiling_symmetry (A B : Rectangle) :
  (∃ (C : Rectangle), IsSimilarTo C A ∧ CanTile C B) →
  (∃ (D : Rectangle), IsSimilarTo D B ∧ CanTile D A) :=
by
  sorry


end NUMINAMATH_CALUDE_tiling_symmetry_l1110_111050


namespace NUMINAMATH_CALUDE_wall_height_is_600_l1110_111018

/-- Represents the dimensions of a rectangular object -/
structure Dimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculate the volume of a rectangular object given its dimensions -/
def volume (d : Dimensions) : ℝ := d.length * d.width * d.height

/-- The dimensions of a single brick -/
def brick_dim : Dimensions := ⟨80, 11.25, 6⟩

/-- The known dimensions of the wall (length and width) -/
def wall_dim (h : ℝ) : Dimensions := ⟨800, 22.5, h⟩

/-- The number of bricks required to build the wall -/
def num_bricks : ℕ := 2000

/-- Theorem stating that if 2000 bricks of given dimensions are required to build a wall
    with known length and width, then the height of the wall is 600 cm -/
theorem wall_height_is_600 :
  volume (wall_dim 600) = (volume brick_dim) * num_bricks := by sorry

end NUMINAMATH_CALUDE_wall_height_is_600_l1110_111018


namespace NUMINAMATH_CALUDE_right_triangle_exterior_angles_sum_l1110_111019

theorem right_triangle_exterior_angles_sum (α β γ δ ε : Real) : 
  α + β + γ = 180 →  -- Sum of angles in a triangle
  γ = 90 →           -- Right angle in the triangle
  α + δ = 180 →      -- Linear pair for first non-right angle
  β + ε = 180 →      -- Linear pair for second non-right angle
  δ + ε = 270 :=     -- Sum of exterior angles
by sorry

end NUMINAMATH_CALUDE_right_triangle_exterior_angles_sum_l1110_111019


namespace NUMINAMATH_CALUDE_john_arcade_spending_l1110_111036

/-- The fraction of John's allowance spent at the arcade -/
def arcade_fraction (allowance arcade_spent : ℚ) : ℚ :=
  arcade_spent / allowance

/-- The amount remaining after spending at the arcade and toy store -/
def remaining_after_toy_store (allowance arcade_spent : ℚ) : ℚ :=
  allowance - arcade_spent - (1/3) * (allowance - arcade_spent)

theorem john_arcade_spending :
  ∃ (arcade_spent : ℚ),
    arcade_fraction 3.30 arcade_spent = 3/5 ∧
    remaining_after_toy_store 3.30 arcade_spent = 0.88 := by
  sorry

end NUMINAMATH_CALUDE_john_arcade_spending_l1110_111036


namespace NUMINAMATH_CALUDE_probability_of_three_in_18_23_l1110_111065

/-- The decimal representation of a rational number -/
def decimalRepresentation (n d : ℕ) : List ℕ :=
  sorry

/-- Count the occurrences of a digit in a list of digits -/
def countOccurrences (digit : ℕ) (digits : List ℕ) : ℕ :=
  sorry

/-- The probability of selecting a specific digit from a decimal representation -/
def probabilityOfDigit (n d digit : ℕ) : ℚ :=
  let digits := decimalRepresentation n d
  (countOccurrences digit digits : ℚ) / (digits.length : ℚ)

theorem probability_of_three_in_18_23 :
  probabilityOfDigit 18 23 3 = 3 / 26 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_three_in_18_23_l1110_111065


namespace NUMINAMATH_CALUDE_certain_inning_is_19th_l1110_111094

/-- Represents the statistics of a cricketer before and after a certain inning -/
structure CricketerStats where
  prevInnings : ℕ
  prevAverage : ℚ
  runsScored : ℕ
  newAverage : ℚ

/-- Theorem stating that given the conditions, the certain inning was the 19th inning -/
theorem certain_inning_is_19th (stats : CricketerStats)
  (h1 : stats.runsScored = 97)
  (h2 : stats.newAverage = stats.prevAverage + 4)
  (h3 : stats.newAverage = 25) :
  stats.prevInnings + 1 = 19 := by
  sorry

end NUMINAMATH_CALUDE_certain_inning_is_19th_l1110_111094


namespace NUMINAMATH_CALUDE_sum_of_three_digit_permutations_not_2018_l1110_111046

theorem sum_of_three_digit_permutations_not_2018 (a b c : ℕ) : 
  (0 < a ∧ a ≤ 9) → (0 < b ∧ b ≤ 9) → (0 < c ∧ c ≤ 9) → 
  a ≠ b → b ≠ c → a ≠ c →
  (100*a + 10*b + c) + (100*a + 10*c + b) + (100*b + 10*a + c) + 
  (100*b + 10*c + a) + (100*c + 10*a + b) + (100*c + 10*b + a) ≠ 2018 :=
sorry

end NUMINAMATH_CALUDE_sum_of_three_digit_permutations_not_2018_l1110_111046


namespace NUMINAMATH_CALUDE_shifted_roots_polynomial_l1110_111016

theorem shifted_roots_polynomial (a b c : ℂ) : 
  (∀ x : ℂ, x^3 - 5*x + 7 = 0 ↔ x = a ∨ x = b ∨ x = c) →
  (∀ x : ℂ, x^3 + 9*x^2 + 22*x + 19 = 0 ↔ x = a - 3 ∨ x = b - 3 ∨ x = c - 3) := by
sorry

end NUMINAMATH_CALUDE_shifted_roots_polynomial_l1110_111016


namespace NUMINAMATH_CALUDE_consecutive_integers_sqrt_17_l1110_111079

theorem consecutive_integers_sqrt_17 (a b : ℕ) :
  a > 0 ∧ b > 0 ∧ b = a + 1 ∧ (a : ℝ) < Real.sqrt 17 ∧ Real.sqrt 17 < (b : ℝ) → a + b = 9 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_sqrt_17_l1110_111079


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l1110_111003

/-- If x^3 - 2x^2 + px + q is divisible by x + 2, then q = 16 + 2p -/
theorem polynomial_divisibility (p q : ℝ) : 
  (∃ k : ℝ, ∀ x : ℝ, x^3 - 2*x^2 + p*x + q = (x + 2) * k) → 
  q = 16 + 2*p := by
sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l1110_111003


namespace NUMINAMATH_CALUDE_parallelogram_area_32_15_l1110_111074

/-- The area of a parallelogram with given base and height -/
def parallelogram_area (base height : ℝ) : ℝ := base * height

/-- Theorem: The area of a parallelogram with base 32 cm and height 15 cm is 480 square centimeters -/
theorem parallelogram_area_32_15 :
  parallelogram_area 32 15 = 480 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_32_15_l1110_111074


namespace NUMINAMATH_CALUDE_arithmetic_geometric_ratio_l1110_111041

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Three terms form a geometric sequence -/
def geometric_sequence (x y z : ℝ) : Prop :=
  ∃ q : ℝ, y = x * q ∧ z = y * q

theorem arithmetic_geometric_ratio (a : ℕ → ℝ) :
  arithmetic_sequence a →
  geometric_sequence (a 1 + 2) (a 5 + 5) (a 9 + 8) →
  ∃ q : ℝ, geometric_sequence (a 1 + 2) (a 5 + 5) (a 9 + 8) ∧ q = 1 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_ratio_l1110_111041


namespace NUMINAMATH_CALUDE_common_root_is_neg_half_l1110_111089

/-- Definition of the first polynomial -/
def p (a b c : ℝ) (x : ℝ) : ℝ := 50 * x^4 + a * x^3 + b * x^2 + c * x + 16

/-- Definition of the second polynomial -/
def q (d e f g : ℝ) (x : ℝ) : ℝ := 16 * x^5 + d * x^4 + e * x^3 + f * x^2 + g * x + 50

/-- Theorem stating that if p and q have a common negative rational root, it must be -1/2 -/
theorem common_root_is_neg_half (a b c d e f g : ℝ) :
  (∃ (k : ℚ), k < 0 ∧ p a b c k = 0 ∧ q d e f g k = 0) →
  (p a b c (-1/2 : ℚ) = 0 ∧ q d e f g (-1/2 : ℚ) = 0) :=
by sorry

end NUMINAMATH_CALUDE_common_root_is_neg_half_l1110_111089


namespace NUMINAMATH_CALUDE_diameter_scientific_notation_l1110_111026

-- Define the original diameter value
def original_diameter : ℝ := 0.000000103

-- Define the scientific notation components
def coefficient : ℝ := 1.03
def exponent : ℤ := -7

-- Theorem to prove the equality
theorem diameter_scientific_notation :
  original_diameter = coefficient * (10 : ℝ) ^ exponent :=
by
  sorry

end NUMINAMATH_CALUDE_diameter_scientific_notation_l1110_111026


namespace NUMINAMATH_CALUDE_cube_stacking_height_l1110_111049

/-- The edge length of the large cube in meters -/
def large_cube_edge : ℝ := 1

/-- The edge length of the small cubes in millimeters -/
def small_cube_edge : ℝ := 1

/-- Conversion factor from meters to millimeters -/
def m_to_mm : ℝ := 1000

/-- Conversion factor from kilometers to millimeters -/
def km_to_mm : ℝ := 1000000

/-- The height of the column formed by stacking all small cubes in kilometers -/
def column_height : ℝ := 1000

theorem cube_stacking_height :
  (large_cube_edge * m_to_mm)^3 / small_cube_edge^3 * small_cube_edge / km_to_mm = column_height := by
  sorry

end NUMINAMATH_CALUDE_cube_stacking_height_l1110_111049


namespace NUMINAMATH_CALUDE_min_value_problem_l1110_111032

theorem min_value_problem (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x + 1/y) * (x + 1/y - 2023) + (y + 1/x) * (y + 1/x - 2023) ≥ -2048113 := by
  sorry

end NUMINAMATH_CALUDE_min_value_problem_l1110_111032


namespace NUMINAMATH_CALUDE_emerie_nickels_l1110_111056

/-- The number of coin types -/
def num_coin_types : ℕ := 3

/-- The number of coins Zain has -/
def zain_coins : ℕ := 48

/-- The number of quarters Emerie has -/
def emerie_quarters : ℕ := 6

/-- The number of dimes Emerie has -/
def emerie_dimes : ℕ := 7

/-- The number of extra coins Zain has for each type -/
def extra_coins_per_type : ℕ := 10

theorem emerie_nickels : 
  (zain_coins - num_coin_types * extra_coins_per_type) - (emerie_quarters + emerie_dimes) = 5 := by
  sorry

end NUMINAMATH_CALUDE_emerie_nickels_l1110_111056


namespace NUMINAMATH_CALUDE_A_necessary_not_sufficient_l1110_111061

-- Define the logarithm base 10
noncomputable def log10 (x : ℝ) := Real.log x / Real.log 10

-- Define propositions A and B
def proposition_A (x : ℝ) : Prop := log10 (x^2) = 0
def proposition_B (x : ℝ) : Prop := x = 1

-- Theorem stating A is necessary but not sufficient for B
theorem A_necessary_not_sufficient :
  (∀ x : ℝ, proposition_B x → proposition_A x) ∧
  (∃ x : ℝ, proposition_A x ∧ ¬proposition_B x) :=
sorry

end NUMINAMATH_CALUDE_A_necessary_not_sufficient_l1110_111061


namespace NUMINAMATH_CALUDE_range_of_m_min_value_sum_squares_equality_condition_l1110_111002

-- Define the function f
def f (x : ℝ) : ℝ := |x + 1| - |x - 4|

-- Theorem 1: Range of m
theorem range_of_m (m : ℝ) :
  (∀ x, f x ≤ -m^2 + 6*m) → 1 ≤ m ∧ m ≤ 5 :=
sorry

-- Theorem 2: Minimum value of a^2 + b^2 + c^2
theorem min_value_sum_squares (a b c : ℝ) :
  a > 0 → b > 0 → c > 0 → 3*a + 4*b + 5*c = 5 →
  a^2 + b^2 + c^2 ≥ 1/2 :=
sorry

-- Theorem 3: Equality condition
theorem equality_condition (a b c : ℝ) :
  a > 0 → b > 0 → c > 0 → 3*a + 4*b + 5*c = 5 →
  a^2 + b^2 + c^2 = 1/2 ↔ a = 3/10 ∧ b = 4/10 ∧ c = 5/10 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_min_value_sum_squares_equality_condition_l1110_111002


namespace NUMINAMATH_CALUDE_deck_problem_l1110_111006

theorem deck_problem (r b : ℕ) : 
  r / (r + b) = 1 / 5 →
  r / (r + (b + 6)) = 1 / 7 →
  r = 3 :=
by sorry

end NUMINAMATH_CALUDE_deck_problem_l1110_111006


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_m_values_l1110_111027

/-- A perfect square trinomial in the form ax^2 + bx + c -/
structure PerfectSquareTrinomial (a b c : ℝ) : Prop where
  is_perfect_square : ∃ (p q : ℝ), a * x^2 + b * x + c = (p * x + q)^2

/-- The main theorem -/
theorem perfect_square_trinomial_m_values (m : ℝ) :
  PerfectSquareTrinomial 1 (m - 1) 9 → m = -5 ∨ m = 7 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_m_values_l1110_111027


namespace NUMINAMATH_CALUDE_point_constraints_l1110_111037

theorem point_constraints (x y : ℝ) :
  x^2 + y^2 ≤ 2 →
  -1 ≤ x / (x + y) →
  x / (x + y) ≤ 1 →
  0 ≤ y ∧ -2*x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_point_constraints_l1110_111037


namespace NUMINAMATH_CALUDE_panthers_score_l1110_111020

theorem panthers_score (total_points margin : ℕ) 
  (h1 : total_points = 34)
  (h2 : margin = 14) : 
  total_points - (total_points + margin) / 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_panthers_score_l1110_111020


namespace NUMINAMATH_CALUDE_cube_preserves_order_for_negative_numbers_l1110_111098

theorem cube_preserves_order_for_negative_numbers (a b : ℝ) (h1 : a < b) (h2 : b < 0) : a^3 < b^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_preserves_order_for_negative_numbers_l1110_111098


namespace NUMINAMATH_CALUDE_minimize_S_l1110_111005

/-- The sum of squared differences function -/
def S (x y z : ℝ) : ℝ :=
  (x + y + z - 10)^2 + (x + y - z - 7)^2 + (x - y + z - 6)^2 + (-x + y + z - 5)^2

/-- Theorem stating that (4.5, 4, 3.5) minimizes S -/
theorem minimize_S :
  ∀ x y z : ℝ, S x y z ≥ S 4.5 4 3.5 := by sorry

end NUMINAMATH_CALUDE_minimize_S_l1110_111005


namespace NUMINAMATH_CALUDE_primitive_root_mod_p_squared_l1110_111087

theorem primitive_root_mod_p_squared (p : Nat) (x : Nat) 
  (h_p : Nat.Prime p) 
  (h_p_odd : Odd p) 
  (h_x_prim_root : IsPrimitiveRoot x p) : 
  IsPrimitiveRoot x (p^2) ∨ IsPrimitiveRoot (x + p) (p^2) := by
  sorry

end NUMINAMATH_CALUDE_primitive_root_mod_p_squared_l1110_111087


namespace NUMINAMATH_CALUDE_square_perimeter_l1110_111013

theorem square_perimeter (area : ℝ) (side : ℝ) (perimeter : ℝ) : 
  area = 500 / 3 → 
  area = side^2 → 
  perimeter = 4 * side → 
  perimeter = 40 * Real.sqrt 15 / 3 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_l1110_111013


namespace NUMINAMATH_CALUDE_prob_sum_32_four_eight_sided_dice_prob_sum_32_four_eight_sided_dice_eq_frac_l1110_111045

/-- The probability of rolling a sum of 32 with four fair eight-sided dice -/
theorem prob_sum_32_four_eight_sided_dice : ℝ :=
  let num_faces : ℕ := 8
  let num_dice : ℕ := 4
  let target_sum : ℕ := 32
  let prob_max_face : ℝ := 1 / num_faces
  (prob_max_face ^ num_dice : ℝ)

#check prob_sum_32_four_eight_sided_dice

theorem prob_sum_32_four_eight_sided_dice_eq_frac :
  prob_sum_32_four_eight_sided_dice = 1 / 4096 := by sorry

end NUMINAMATH_CALUDE_prob_sum_32_four_eight_sided_dice_prob_sum_32_four_eight_sided_dice_eq_frac_l1110_111045


namespace NUMINAMATH_CALUDE_max_sum_of_squares_l1110_111076

theorem max_sum_of_squares (a b c : ℝ) 
  (h1 : a + b = c - 1) 
  (h2 : a * b = c^2 - 7*c + 14) : 
  ∃ (m : ℝ), (∀ (x y z : ℝ), x + y = z - 1 → x * y = z^2 - 7*z + 14 → x^2 + y^2 ≤ m) ∧ a^2 + b^2 = m :=
sorry

end NUMINAMATH_CALUDE_max_sum_of_squares_l1110_111076
