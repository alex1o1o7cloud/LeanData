import Mathlib

namespace NUMINAMATH_CALUDE_differential_at_zero_l986_98671

noncomputable def f (x : ℝ) : ℝ := Real.exp x * (x^2 + 3)

theorem differential_at_zero (x : ℝ) : 
  deriv f 0 = 3 := by sorry

end NUMINAMATH_CALUDE_differential_at_zero_l986_98671


namespace NUMINAMATH_CALUDE_pairwise_sums_distinct_digits_impossible_l986_98644

theorem pairwise_sums_distinct_digits_impossible :
  ¬ ∃ (a b c d e : ℕ),
    let sums := [a + b, a + c, a + d, a + e, b + c, b + d, b + e, c + d, c + e, d + e]
    ∀ (i j : Fin 10), i ≠ j → sums[i] % 10 ≠ sums[j] % 10 := by
  sorry

#check pairwise_sums_distinct_digits_impossible

end NUMINAMATH_CALUDE_pairwise_sums_distinct_digits_impossible_l986_98644


namespace NUMINAMATH_CALUDE_evaluate_expression_l986_98665

theorem evaluate_expression : 7^3 - 4 * 7^2 + 6 * 7 - 1 = 188 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l986_98665


namespace NUMINAMATH_CALUDE_gratuity_percentage_is_twenty_percent_l986_98662

def number_of_people : ℕ := 6
def total_bill : ℚ := 720
def average_cost_before_gratuity : ℚ := 100

theorem gratuity_percentage_is_twenty_percent :
  let total_before_gratuity : ℚ := number_of_people * average_cost_before_gratuity
  let gratuity_amount : ℚ := total_bill - total_before_gratuity
  gratuity_amount / total_before_gratuity = 1/5 := by
sorry

end NUMINAMATH_CALUDE_gratuity_percentage_is_twenty_percent_l986_98662


namespace NUMINAMATH_CALUDE_daves_diner_cost_l986_98623

/-- Represents the pricing and discount structure at Dave's Diner -/
structure DavesDiner where
  burger_price : ℕ
  fries_price : ℕ
  discount_amount : ℕ
  discount_threshold : ℕ

/-- Calculates the total cost of a purchase at Dave's Diner -/
def calculate_total_cost (d : DavesDiner) (num_burgers : ℕ) (num_fries : ℕ) : ℕ :=
  let burger_cost := if num_burgers ≥ d.discount_threshold
    then (d.burger_price - d.discount_amount) * num_burgers
    else d.burger_price * num_burgers
  let fries_cost := d.fries_price * num_fries
  burger_cost + fries_cost

/-- Theorem stating that the total cost of 6 burgers and 5 fries at Dave's Diner is 27 -/
theorem daves_diner_cost : 
  let d : DavesDiner := { 
    burger_price := 4, 
    fries_price := 3, 
    discount_amount := 2, 
    discount_threshold := 4 
  }
  calculate_total_cost d 6 5 = 27 := by
  sorry

end NUMINAMATH_CALUDE_daves_diner_cost_l986_98623


namespace NUMINAMATH_CALUDE_equation_solution_l986_98638

theorem equation_solution : 
  let x : ℚ := -43/8
  7 * (4 * x + 3) - 5 = -3 * (2 - 8 * x) + 1/2 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l986_98638


namespace NUMINAMATH_CALUDE_parabola_transformation_l986_98636

-- Define the original function
def original_function (x : ℝ) : ℝ := (x - 1)^2 + 2

-- Define the transformation
def transform (f : ℝ → ℝ) : ℝ → ℝ := λ x => f (x + 1) - 1

-- State the theorem
theorem parabola_transformation :
  ∀ x : ℝ, transform original_function x = x^2 + 1 :=
by sorry

end NUMINAMATH_CALUDE_parabola_transformation_l986_98636


namespace NUMINAMATH_CALUDE_union_when_k_neg_one_intersection_condition_l986_98645

-- Define sets A and B
def A : Set ℝ := {x | (x - 2) / (x + 1) < 0}
def B (k : ℝ) : Set ℝ := {x | k < x ∧ x < 2 - k}

-- Theorem 1: When k = -1, A ∪ B = (-1, 3)
theorem union_when_k_neg_one :
  A ∪ B (-1) = Set.Ioo (-1) 3 := by sorry

-- Theorem 2: A ∩ B = B if and only if k ∈ [0, +∞)
theorem intersection_condition (k : ℝ) :
  A ∩ B k = B k ↔ k ∈ Set.Ici 0 := by sorry

end NUMINAMATH_CALUDE_union_when_k_neg_one_intersection_condition_l986_98645


namespace NUMINAMATH_CALUDE_certain_number_proof_l986_98657

/-- Given that 213 * 16 = 3408, prove that the number x satisfying x * 2.13 = 0.03408 is equal to 0.016 -/
theorem certain_number_proof (h : 213 * 16 = 3408) : 
  ∃ x : ℝ, x * 2.13 = 0.03408 ∧ x = 0.016 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l986_98657


namespace NUMINAMATH_CALUDE_janet_song_time_l986_98606

theorem janet_song_time (original_time : ℝ) (speed_increase : ℝ) (new_time : ℝ) : 
  original_time = 200 →
  speed_increase = 0.25 →
  new_time = original_time / (1 + speed_increase) →
  new_time = 160 := by
sorry

end NUMINAMATH_CALUDE_janet_song_time_l986_98606


namespace NUMINAMATH_CALUDE_sons_age_l986_98631

theorem sons_age (father_age son_age : ℕ) : 
  father_age = son_age + 25 →
  father_age + 2 = 2 * (son_age + 2) →
  son_age = 23 := by
sorry

end NUMINAMATH_CALUDE_sons_age_l986_98631


namespace NUMINAMATH_CALUDE_watch_sale_loss_percentage_l986_98686

/-- Proves that the loss percentage is 10% given the conditions of the watch sale problem -/
theorem watch_sale_loss_percentage (cost_price : ℝ) (additional_amount : ℝ) (gain_percentage : ℝ) :
  cost_price = 3000 →
  additional_amount = 540 →
  gain_percentage = 8 →
  ∃ (loss_percentage : ℝ),
    loss_percentage = 10 ∧
    cost_price * (1 + gain_percentage / 100) = 
    cost_price * (1 - loss_percentage / 100) + additional_amount :=
by
  sorry

end NUMINAMATH_CALUDE_watch_sale_loss_percentage_l986_98686


namespace NUMINAMATH_CALUDE_binomial_minus_five_l986_98608

theorem binomial_minus_five : Nat.choose 10 3 - 5 = 115 := by
  sorry

end NUMINAMATH_CALUDE_binomial_minus_five_l986_98608


namespace NUMINAMATH_CALUDE_jacoby_trip_savings_l986_98688

-- Define the problem parameters
def trip_cost : ℕ := 5000
def hourly_wage : ℕ := 20
def hours_worked : ℕ := 10
def cookie_price : ℕ := 4
def cookies_sold : ℕ := 24
def lottery_ticket_cost : ℕ := 10
def lottery_winnings : ℕ := 500
def sister_gift : ℕ := 500
def num_sisters : ℕ := 2

-- Define the theorem
theorem jacoby_trip_savings : 
  trip_cost - (hourly_wage * hours_worked + cookie_price * cookies_sold - lottery_ticket_cost + lottery_winnings + sister_gift * num_sisters) = 3214 := by
  sorry

end NUMINAMATH_CALUDE_jacoby_trip_savings_l986_98688


namespace NUMINAMATH_CALUDE_smallest_sum_reciprocals_l986_98627

theorem smallest_sum_reciprocals (x y : ℕ+) (h1 : x ≠ y) (h2 : (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 10) :
  ∃ (a b : ℕ+), a ≠ b ∧ (1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / 10 ∧ (a : ℕ) + (b : ℕ) = 49 ∧
  ∀ (c d : ℕ+), c ≠ d → (1 : ℚ) / c + (1 : ℚ) / d = (1 : ℚ) / 10 → (c : ℕ) + (d : ℕ) ≥ 49 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_reciprocals_l986_98627


namespace NUMINAMATH_CALUDE_repeating_decimal_fraction_sum_l986_98653

theorem repeating_decimal_fraction_sum : ∃ (n d : ℕ), 
  (n.gcd d = 1) ∧ 
  (n : ℚ) / (d : ℚ) = 3.17171717 ∧ 
  n + d = 413 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_fraction_sum_l986_98653


namespace NUMINAMATH_CALUDE_can_display_total_l986_98617

def triangle_display (n : ℕ) (first_row : ℕ) (increment : ℕ) : ℕ :=
  (n * (2 * first_row + (n - 1) * increment)) / 2

theorem can_display_total :
  let n := 9  -- number of rows
  let seventh_row := 19  -- number of cans in the seventh row
  let increment := 3  -- difference in cans between adjacent rows
  let first_row := seventh_row - 6 * increment  -- number of cans in the first row
  triangle_display n first_row increment = 117 :=
by
  sorry

end NUMINAMATH_CALUDE_can_display_total_l986_98617


namespace NUMINAMATH_CALUDE_book_sale_revenue_l986_98615

theorem book_sale_revenue (total_books : ℕ) (sold_fraction : ℚ) (price_per_book : ℚ) (remaining_books : ℕ) : 
  sold_fraction = 2/3 →
  price_per_book = 2 →
  remaining_books = 36 →
  (1 - sold_fraction) * total_books = remaining_books →
  sold_fraction * total_books * price_per_book = 144 :=
by
  sorry

end NUMINAMATH_CALUDE_book_sale_revenue_l986_98615


namespace NUMINAMATH_CALUDE_lottery_smallest_number_l986_98689

def lottery_problem (max_num : ℕ) : Prop :=
  let prob_1_to_15 : ℚ := 1/3
  let prob_greater_than_N : ℚ := 2/3
  let prob_less_equal_15 : ℚ := 2/3
  ∃ (N : ℕ),
    N = 16 ∧
    N > 15 ∧
    prob_1_to_15 + prob_greater_than_N = 1 ∧
    prob_1_to_15 + (prob_less_equal_15 - prob_1_to_15) = prob_greater_than_N ∧
    ∀ (M : ℕ), M < N → (↑M : ℚ)/max_num ≤ prob_less_equal_15

theorem lottery_smallest_number :
  ∃ (max_num : ℕ), lottery_problem max_num :=
sorry

end NUMINAMATH_CALUDE_lottery_smallest_number_l986_98689


namespace NUMINAMATH_CALUDE_smallest_k_for_error_bound_l986_98660

def u : ℕ → ℚ
  | 0 => 1/3
  | n + 1 => 2 * u n - 2 * (u n)^2

def L : ℚ := 1/2

theorem smallest_k_for_error_bound :
  ∃ (k : ℕ), (∀ (n : ℕ), n < k → |u n - L| > 1/2^1000) ∧
             |u k - L| ≤ 1/2^1000 ∧
             k = 9 := by
  sorry

end NUMINAMATH_CALUDE_smallest_k_for_error_bound_l986_98660


namespace NUMINAMATH_CALUDE_snow_on_monday_l986_98605

theorem snow_on_monday (total_snow : ℝ) (tuesday_snow : ℝ) 
  (h1 : total_snow = 0.53)
  (h2 : tuesday_snow = 0.21) :
  total_snow - tuesday_snow = 0.53 - 0.21 := by
sorry

end NUMINAMATH_CALUDE_snow_on_monday_l986_98605


namespace NUMINAMATH_CALUDE_min_value_cubic_function_l986_98695

theorem min_value_cubic_function (y : ℝ) (h : y > 0) :
  y^2 + 10*y + 100/y^3 ≥ 50^(2/3) + 10 * 50^(1/3) + 2 ∧
  (y^2 + 10*y + 100/y^3 = 50^(2/3) + 10 * 50^(1/3) + 2 ↔ y = 50^(1/3)) :=
by sorry

end NUMINAMATH_CALUDE_min_value_cubic_function_l986_98695


namespace NUMINAMATH_CALUDE_probability_ella_zoe_same_team_l986_98635

/-- The number of cards in the deck -/
def deck_size : ℕ := 52

/-- The card number chosen by Ella -/
def b : ℕ := 11

/-- The probability that Ella and Zoe are on the same team -/
def p (b : ℕ) : ℚ :=
  let remaining_cards := deck_size - 2
  let total_combinations := remaining_cards.choose 2
  let lower_team_combinations := (b - 1).choose 2
  let higher_team_combinations := (deck_size - b - 11).choose 2
  (lower_team_combinations + higher_team_combinations : ℚ) / total_combinations

theorem probability_ella_zoe_same_team :
  p b = 857 / 1225 :=
sorry

end NUMINAMATH_CALUDE_probability_ella_zoe_same_team_l986_98635


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_specific_roots_l986_98666

/-- The quadratic equation x^2 - (k+2)x + 2k - 1 = 0 has two distinct real roots for any real k -/
theorem quadratic_two_distinct_roots (k : ℝ) : 
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    x₁^2 - (k+2)*x₁ + 2*k - 1 = 0 ∧ 
    x₂^2 - (k+2)*x₂ + 2*k - 1 = 0 :=
sorry

/-- When one root of the equation x^2 - (k+2)x + 2k - 1 = 0 is 3, k = 2 and the other root is 1 -/
theorem specific_roots : 
  ∃ k : ℝ, 3^2 - (k+2)*3 + 2*k - 1 = 0 ∧ 
    k = 2 ∧
    1^2 - (k+2)*1 + 2*k - 1 = 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_specific_roots_l986_98666


namespace NUMINAMATH_CALUDE_distance_difference_around_block_l986_98676

/-- The difference in distance run around a square block -/
theorem distance_difference_around_block (block_side_length street_width : ℝ) :
  block_side_length = 500 →
  street_width = 25 →
  (4 * (block_side_length + 2 * street_width)) - (4 * block_side_length) = 200 := by
  sorry

end NUMINAMATH_CALUDE_distance_difference_around_block_l986_98676


namespace NUMINAMATH_CALUDE_perpendicular_lines_parallel_l986_98637

-- Define the types for lines and planes in 3D space
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perp : Line → Plane → Prop)
variable (para : Line → Line → Prop)

-- State the theorem
theorem perpendicular_lines_parallel (m n : Line) (α : Plane) :
  perp m α → perp n α → para m n :=
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_parallel_l986_98637


namespace NUMINAMATH_CALUDE_min_distance_circle_line_l986_98677

/-- The minimum distance between a point on the circle (x + 1)² + y² = 1 
    and a point on the line 3x + 4y + 13 = 0 is equal to 1. -/
theorem min_distance_circle_line : 
  ∃ (d : ℝ), d = 1 ∧ 
  ∀ (x₁ y₁ x₂ y₂ : ℝ), 
    ((x₁ + 1)^2 + y₁^2 = 1) →
    (3*x₂ + 4*y₂ + 13 = 0) →
    ((x₁ - x₂)^2 + (y₁ - y₂)^2)^(1/2) ≥ d :=
by sorry

end NUMINAMATH_CALUDE_min_distance_circle_line_l986_98677


namespace NUMINAMATH_CALUDE_molecular_weight_is_265_21_l986_98626

/-- Atomic weight of Aluminium in amu -/
def Al_weight : ℝ := 26.98

/-- Atomic weight of Oxygen in amu -/
def O_weight : ℝ := 16.00

/-- Atomic weight of Hydrogen in amu -/
def H_weight : ℝ := 1.01

/-- Atomic weight of Silicon in amu -/
def Si_weight : ℝ := 28.09

/-- Atomic weight of Nitrogen in amu -/
def N_weight : ℝ := 14.01

/-- Number of Aluminium atoms in the compound -/
def Al_count : ℕ := 2

/-- Number of Oxygen atoms in the compound -/
def O_count : ℕ := 6

/-- Number of Hydrogen atoms in the compound -/
def H_count : ℕ := 3

/-- Number of Silicon atoms in the compound -/
def Si_count : ℕ := 2

/-- Number of Nitrogen atoms in the compound -/
def N_count : ℕ := 4

/-- The molecular weight of the compound -/
def molecular_weight : ℝ :=
  Al_count * Al_weight + O_count * O_weight + H_count * H_weight +
  Si_count * Si_weight + N_count * N_weight

theorem molecular_weight_is_265_21 : molecular_weight = 265.21 := by
  sorry

end NUMINAMATH_CALUDE_molecular_weight_is_265_21_l986_98626


namespace NUMINAMATH_CALUDE_course_selection_theorem_l986_98683

def type_a_courses : ℕ := 3
def type_b_courses : ℕ := 4
def total_courses_to_choose : ℕ := 3

/-- The number of ways to select courses from two types of electives -/
def number_of_selections : ℕ :=
  Nat.choose type_a_courses 1 * Nat.choose type_b_courses 2 +
  Nat.choose type_a_courses 2 * Nat.choose type_b_courses 1

theorem course_selection_theorem :
  number_of_selections = 30 := by sorry

end NUMINAMATH_CALUDE_course_selection_theorem_l986_98683


namespace NUMINAMATH_CALUDE_trig_identity_l986_98600

theorem trig_identity : 
  4 * Real.sin (15 * π / 180) + Real.tan (75 * π / 180) = 
  (4 - 3 * (Real.cos (15 * π / 180))^2 + Real.cos (15 * π / 180)) / Real.sin (15 * π / 180) := by
sorry

end NUMINAMATH_CALUDE_trig_identity_l986_98600


namespace NUMINAMATH_CALUDE_no_extreme_points_iff_m_range_l986_98675

/-- The function f(x) = x ln x + mx² - m has no extreme points in its domain
    if and only if m ∈ (-∞, -1/2] --/
theorem no_extreme_points_iff_m_range (m : ℝ) :
  (∀ x : ℝ, x > 0 → ∃ ε > 0, ∀ y ∈ Set.Ioo (x - ε) (x + ε),
    (y * Real.log y + m * y^2 - m ≠ x * Real.log x + m * x^2 - m)) ↔
  m ≤ -1/2 :=
sorry

end NUMINAMATH_CALUDE_no_extreme_points_iff_m_range_l986_98675


namespace NUMINAMATH_CALUDE_water_level_unchanged_l986_98690

-- Define the densities of water and ice
variable (ρ_water ρ_ice : ℝ)

-- Define the initial volume of water taken for freezing
variable (V : ℝ)

-- Hypothesis: density of ice is less than density of water
axiom h1 : ρ_ice < ρ_water

-- Hypothesis: mass is conserved when water freezes
axiom h2 : V * ρ_water = (V * ρ_water / ρ_ice) * ρ_ice

-- Hypothesis: Archimedes' principle applies to floating ice
axiom h3 : ∀ W : ℝ, W * ρ_ice = (W * ρ_ice / ρ_water) * ρ_water

-- Theorem: The volume of water displaced by the ice is equal to the original volume of water
theorem water_level_unchanged (V : ℝ) (h1 : ρ_ice < ρ_water) 
  (h2 : V * ρ_water = (V * ρ_water / ρ_ice) * ρ_ice) 
  (h3 : ∀ W : ℝ, W * ρ_ice = (W * ρ_ice / ρ_water) * ρ_water) :
  (V * ρ_water / ρ_ice) * ρ_ice / ρ_water = V :=
by sorry

end NUMINAMATH_CALUDE_water_level_unchanged_l986_98690


namespace NUMINAMATH_CALUDE_belinda_passed_twenty_percent_l986_98669

/-- The percentage of flyers Belinda passed out -/
def belinda_percentage (total flyers : ℕ) (ryan alyssa scott : ℕ) : ℚ :=
  (total - (ryan + alyssa + scott)) / total * 100

/-- Theorem stating that Belinda passed out 20% of the flyers -/
theorem belinda_passed_twenty_percent :
  belinda_percentage 200 42 67 51 = 20 := by
  sorry

end NUMINAMATH_CALUDE_belinda_passed_twenty_percent_l986_98669


namespace NUMINAMATH_CALUDE_ex_factory_price_decrease_selling_price_for_profit_l986_98625

/-- Ex-factory price in 2019 -/
def price_2019 : ℝ := 144

/-- Ex-factory price in 2021 -/
def price_2021 : ℝ := 100

/-- Current selling price -/
def current_price : ℝ := 140

/-- Current daily sales -/
def current_sales : ℝ := 20

/-- Sales increase per price reduction -/
def sales_increase : ℝ := 10

/-- Price reduction step -/
def price_reduction : ℝ := 5

/-- Target daily profit -/
def target_profit : ℝ := 1250

/-- Average yearly percentage decrease in ex-factory price -/
def avg_decrease : ℝ := 16.67

/-- Selling price for desired profit -/
def desired_price : ℝ := 125

theorem ex_factory_price_decrease :
  ∃ (x : ℝ), price_2019 * (1 - x / 100)^2 = price_2021 ∧ x = avg_decrease :=
sorry

theorem selling_price_for_profit :
  ∃ (y : ℝ),
    (y - price_2021) * (current_sales + sales_increase * (current_price - y) / price_reduction) = target_profit ∧
    y = desired_price :=
sorry

end NUMINAMATH_CALUDE_ex_factory_price_decrease_selling_price_for_profit_l986_98625


namespace NUMINAMATH_CALUDE_calculation_proof_l986_98694

theorem calculation_proof :
  (- (1 : ℤ) ^ 2023 + 8 * (-(1/2 : ℚ))^3 + |(-3 : ℤ)| = 1) ∧
  ((-25 : ℤ) * (3/2 : ℚ) - (-25 : ℤ) * (5/8 : ℚ) + (-25 : ℤ) / 8 = -25) := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l986_98694


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l986_98602

theorem geometric_sequence_sum (a : ℕ → ℚ) (r : ℚ) :
  (∀ n : ℕ, a (n + 1) = a n * r) →
  a 3 = 256 →
  a 5 = 4 →
  a 3 + a 4 = 80 :=
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l986_98602


namespace NUMINAMATH_CALUDE_reggie_lost_games_l986_98654

theorem reggie_lost_games (initial_marbles : ℕ) (bet_per_game : ℕ) (total_games : ℕ) (final_marbles : ℕ)
  (h1 : initial_marbles = 100)
  (h2 : bet_per_game = 10)
  (h3 : total_games = 9)
  (h4 : final_marbles = 90) :
  (initial_marbles - final_marbles) / bet_per_game = 1 := by
  sorry

end NUMINAMATH_CALUDE_reggie_lost_games_l986_98654


namespace NUMINAMATH_CALUDE_xyz_value_l986_98652

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 35)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 11) : 
  x * y * z = 8 := by
sorry

end NUMINAMATH_CALUDE_xyz_value_l986_98652


namespace NUMINAMATH_CALUDE_complex_equation_solution_l986_98672

theorem complex_equation_solution :
  ∃ z : ℂ, (4 : ℂ) - 2 * Complex.I * z = 3 + 5 * Complex.I * z ∧ z = (1 / 7 : ℂ) * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l986_98672


namespace NUMINAMATH_CALUDE_bakery_baguettes_l986_98687

theorem bakery_baguettes (baguettes_per_batch : ℕ) : 
  (3 * baguettes_per_batch - 37 - 52 - 49 = 6) → baguettes_per_batch = 48 := by
  sorry

end NUMINAMATH_CALUDE_bakery_baguettes_l986_98687


namespace NUMINAMATH_CALUDE_paul_sunday_bags_l986_98642

/-- The number of bags Paul filled on Saturday -/
def saturday_bags : ℕ := 6

/-- The number of cans in each bag -/
def cans_per_bag : ℕ := 8

/-- The total number of cans collected -/
def total_cans : ℕ := 72

/-- The number of bags Paul filled on Sunday -/
def sunday_bags : ℕ := (total_cans - saturday_bags * cans_per_bag) / cans_per_bag

theorem paul_sunday_bags :
  sunday_bags = 3 := by sorry

end NUMINAMATH_CALUDE_paul_sunday_bags_l986_98642


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l986_98624

/-- Hyperbola structure -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  c : ℝ
  h : c^2 = a^2 + b^2

/-- Equilateral triangle structure -/
structure EquilateralTriangle where
  side : ℝ

/-- Theorem: Eccentricity of a hyperbola with specific conditions -/
theorem hyperbola_eccentricity (C : Hyperbola) (T : EquilateralTriangle)
  (h1 : T.side = 2 * C.c) -- AF₁ = AF₂ = F₁F₂ = 2c
  (h2 : ∃ (B : ℝ × ℝ), B.1^2 / C.a^2 - B.2^2 / C.b^2 = 1 ∧ 
    (B.1 + C.c)^2 + B.2^2 = (5/4 * T.side)^2) -- B is on the hyperbola and AB = 5/4 * AF₁
  : C.c / C.a = (Real.sqrt 13 + 1) / 3 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l986_98624


namespace NUMINAMATH_CALUDE_ammonium_iodide_required_l986_98684

-- Define the molecules and their molar quantities
structure Reaction where
  nh4i : ℝ  -- Ammonium iodide
  koh : ℝ   -- Potassium hydroxide
  nh3 : ℝ   -- Ammonia
  ki : ℝ    -- Potassium iodide
  h2o : ℝ   -- Water

-- Define the balanced chemical equation
def balanced_equation (r : Reaction) : Prop :=
  r.nh4i = r.koh ∧ r.nh4i = r.nh3 ∧ r.nh4i = r.ki ∧ r.nh4i = r.h2o

-- Define the given conditions
def given_conditions (r : Reaction) : Prop :=
  r.koh = 3 ∧ r.nh3 = 3 ∧ r.ki = 3 ∧ r.h2o = 3

-- Theorem statement
theorem ammonium_iodide_required (r : Reaction) 
  (h1 : balanced_equation r) (h2 : given_conditions r) : 
  r.nh4i = 3 :=
sorry

end NUMINAMATH_CALUDE_ammonium_iodide_required_l986_98684


namespace NUMINAMATH_CALUDE_circle_tangent_to_x_axis_l986_98655

-- Define the center of the circle
def center : ℝ × ℝ := (-3, 4)

-- Define the equation of the circle
def circle_equation (x y : ℝ) : Prop :=
  (x + 3)^2 + (y - 4)^2 = 16

-- Theorem statement
theorem circle_tangent_to_x_axis :
  -- The circle has center at (-3, 4)
  ∃ (x y : ℝ), circle_equation x y ∧ (x, y) = center ∧
  -- The circle is tangent to the x-axis
  ∃ (x : ℝ), circle_equation x 0 ∧
  -- The equation represents a circle
  ∀ (p : ℝ × ℝ), p ∈ {p | circle_equation p.1 p.2} ↔ 
    (p.1 - center.1)^2 + (p.2 - center.2)^2 = 4^2 :=
sorry

end NUMINAMATH_CALUDE_circle_tangent_to_x_axis_l986_98655


namespace NUMINAMATH_CALUDE_olivia_car_rental_cost_l986_98691

/-- Calculates the total cost of renting a car given the daily rate, per-mile rate, number of days, and miles driven. -/
def carRentalCost (dailyRate perMileRate : ℚ) (days miles : ℕ) : ℚ :=
  dailyRate * days + perMileRate * miles

/-- Proves that Olivia's car rental costs $215 given the specified conditions. -/
theorem olivia_car_rental_cost :
  carRentalCost 30 (1/4) 3 500 = 215 := by
  sorry

end NUMINAMATH_CALUDE_olivia_car_rental_cost_l986_98691


namespace NUMINAMATH_CALUDE_calculation_equality_l986_98674

theorem calculation_equality : 
  |3 - Real.sqrt 12| + (1/3)⁻¹ - 4 * Real.sin (60 * π / 180) + (Real.sqrt 2)^2 = 2 := by sorry

end NUMINAMATH_CALUDE_calculation_equality_l986_98674


namespace NUMINAMATH_CALUDE_min_value_one_iff_k_eq_two_ninths_l986_98699

/-- The expression as a function of x, y, and k -/
def f (x y k : ℝ) : ℝ := 9*x^2 - 8*k*x*y + (4*k^2 + 3)*y^2 - 6*x - 6*y + 9

/-- The theorem stating the minimum value of f is 1 iff k = 2/9 -/
theorem min_value_one_iff_k_eq_two_ninths :
  (∀ x y : ℝ, f x y (2/9 : ℝ) ≥ 1) ∧ (∃ x y : ℝ, f x y (2/9 : ℝ) = 1) ↔
  ∀ k : ℝ, (∀ x y : ℝ, f x y k ≥ 1) ∧ (∃ x y : ℝ, f x y k = 1) → k = 2/9 :=
sorry

end NUMINAMATH_CALUDE_min_value_one_iff_k_eq_two_ninths_l986_98699


namespace NUMINAMATH_CALUDE_pie_arrangement_rows_l986_98678

/-- Given the number of pecan and apple pies, calculates the number of complete rows when arranged with a fixed number of pies per row. -/
def calculate_rows (pecan_pies apple_pies pies_per_row : ℕ) : ℕ :=
  (pecan_pies + apple_pies) / pies_per_row

/-- Proves that 16 pecan pies and 14 apple pies, when arranged in rows of 5 pies each, result in 6 complete rows. -/
theorem pie_arrangement_rows : calculate_rows 16 14 5 = 6 := by
  sorry

end NUMINAMATH_CALUDE_pie_arrangement_rows_l986_98678


namespace NUMINAMATH_CALUDE_valid_triples_l986_98679

def isPrime (n : Nat) : Prop := n > 1 ∧ ∀ m : Nat, m > 1 → m < n → ¬(n % m = 0)

def isGeometricSequence (x y z : Nat) : Prop := ∃ r : ℚ, y = x * r ∧ z = y * r

def validTriple (a b c : Nat) : Prop :=
  isPrime a ∧ isPrime b ∧ isPrime c ∧
  a < b ∧ b < c ∧ c < 100 ∧
  isGeometricSequence (a + 1) (b + 1) (c + 1)

theorem valid_triples :
  {t : Nat × Nat × Nat | validTriple t.1 t.2.1 t.2.2} =
  {(2, 5, 11), (2, 11, 47), (5, 11, 23), (5, 17, 53), (7, 23, 71), (11, 23, 47)} :=
by sorry

end NUMINAMATH_CALUDE_valid_triples_l986_98679


namespace NUMINAMATH_CALUDE_last_two_digits_sum_factorials_l986_98630

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def last_two_digits (n : ℕ) : ℕ := n % 100

def sum_factorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem last_two_digits_sum_factorials :
  last_two_digits (sum_factorials 50) = last_two_digits (sum_factorials 9) := by
sorry

end NUMINAMATH_CALUDE_last_two_digits_sum_factorials_l986_98630


namespace NUMINAMATH_CALUDE_problem_statement_l986_98607

theorem problem_statement (a b x : ℝ) 
  (h1 : a ≠ b) 
  (h2 : a^3 - b^3 = 27*x^3) 
  (h3 : a - b = 3*x) : 
  a = 3*x := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l986_98607


namespace NUMINAMATH_CALUDE_spade_nested_calculation_l986_98601

def spade (x y : ℝ) : ℝ := (x + y) * (x - y)

theorem spade_nested_calculation : spade 3 (spade 4 5) = -72 := by
  sorry

end NUMINAMATH_CALUDE_spade_nested_calculation_l986_98601


namespace NUMINAMATH_CALUDE_cubic_root_magnitude_l986_98614

theorem cubic_root_magnitude (q : ℝ) (r₁ r₂ r₃ : ℝ) : 
  r₁ ≠ r₂ ∧ r₂ ≠ r₃ ∧ r₁ ≠ r₃ →
  r₁^3 + q*r₁^2 + 6*r₁ + 9 = 0 →
  r₂^3 + q*r₂^2 + 6*r₂ + 9 = 0 →
  r₃^3 + q*r₃^2 + 6*r₃ + 9 = 0 →
  (q^2 * 6^2 - 4 * 6^3 - 4*q^3 * 9 - 27 * 9^2 + 18 * q * 6 * (-9)) ≠ 0 →
  max (|r₁|) (max (|r₂|) (|r₃|)) > 2 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_magnitude_l986_98614


namespace NUMINAMATH_CALUDE_change_calculation_l986_98692

def flour_cost : ℕ := 5
def cake_stand_cost : ℕ := 28
def bills_given : ℕ := 20 * 2
def coins_given : ℕ := 3

def total_cost : ℕ := flour_cost + cake_stand_cost
def total_paid : ℕ := bills_given + coins_given

theorem change_calculation (change : ℕ) : 
  change = total_paid - total_cost := by sorry

end NUMINAMATH_CALUDE_change_calculation_l986_98692


namespace NUMINAMATH_CALUDE_li_elevator_journey_l986_98680

def floor_movements : List Int := [5, -3, 10, -8, 12, -6, -10]
def floor_height : ℝ := 2.8
def electricity_per_meter : ℝ := 0.1

theorem li_elevator_journey :
  (List.sum floor_movements = 0) ∧
  (List.sum (List.map (λ x => floor_height * electricity_per_meter * |x|) floor_movements) = 15.12) := by
  sorry

end NUMINAMATH_CALUDE_li_elevator_journey_l986_98680


namespace NUMINAMATH_CALUDE_rectangle_shorter_side_length_l986_98685

theorem rectangle_shorter_side_length 
  (perimeter : ℝ) 
  (longer_side : ℝ) 
  (h1 : perimeter = 100) 
  (h2 : longer_side = 28) : 
  (perimeter - 2 * longer_side) / 2 = 22 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_shorter_side_length_l986_98685


namespace NUMINAMATH_CALUDE_bakers_cakes_l986_98639

/-- Baker's pastry and cake problem -/
theorem bakers_cakes (pastries_made : ℕ) (cakes_sold : ℕ) (pastries_sold : ℕ) (cakes_left : ℕ) :
  pastries_made = 61 →
  cakes_sold = 108 →
  pastries_sold = 44 →
  cakes_left = 59 →
  cakes_sold + cakes_left = 167 := by
  sorry


end NUMINAMATH_CALUDE_bakers_cakes_l986_98639


namespace NUMINAMATH_CALUDE_root_implies_m_value_l986_98633

theorem root_implies_m_value (m : ℝ) : (3^2 - 4*3 + m = 0) → m = 3 := by
  sorry

end NUMINAMATH_CALUDE_root_implies_m_value_l986_98633


namespace NUMINAMATH_CALUDE_fence_cost_l986_98650

/-- The cost of building a fence around a square plot -/
theorem fence_cost (area : ℝ) (price_per_foot : ℝ) (cost : ℝ) : 
  area = 289 →
  price_per_foot = 60 →
  cost = 4 * Real.sqrt area * price_per_foot →
  cost = 4080 := by
  sorry


end NUMINAMATH_CALUDE_fence_cost_l986_98650


namespace NUMINAMATH_CALUDE_point_in_second_quadrant_l986_98668

/-- A point is in the second quadrant if its x-coordinate is negative and its y-coordinate is positive -/
def is_in_second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

/-- The point P(-1, m^2+1) is in the second quadrant for any real number m -/
theorem point_in_second_quadrant (m : ℝ) : is_in_second_quadrant (-1) (m^2 + 1) := by
  sorry

end NUMINAMATH_CALUDE_point_in_second_quadrant_l986_98668


namespace NUMINAMATH_CALUDE_negative_p_exponent_product_l986_98696

theorem negative_p_exponent_product (p : ℝ) : (-p)^2 * (-p)^3 = -p^5 := by sorry

end NUMINAMATH_CALUDE_negative_p_exponent_product_l986_98696


namespace NUMINAMATH_CALUDE_barney_towel_shortage_l986_98647

/-- Represents the number of days in a week -/
def daysInWeek : ℕ := 7

/-- Represents Barney's towel situation -/
structure TowelSituation where
  totalTowels : ℕ
  towelsPerDay : ℕ
  extraTowelsUsed : ℕ
  expectedGuests : ℕ

/-- Calculates the number of days without clean towels -/
def daysWithoutCleanTowels (s : TowelSituation) : ℕ :=
  daysInWeek

/-- Theorem stating that Barney will not have clean towels for 7 days -/
theorem barney_towel_shortage (s : TowelSituation)
  (h1 : s.totalTowels = 18)
  (h2 : s.towelsPerDay = 2)
  (h3 : s.extraTowelsUsed = 5)
  (h4 : s.expectedGuests = 3) :
  daysWithoutCleanTowels s = daysInWeek :=
by sorry

#check barney_towel_shortage

end NUMINAMATH_CALUDE_barney_towel_shortage_l986_98647


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l986_98628

theorem sqrt_equation_solution :
  ∀ x : ℝ, x ≥ 0 → x + 4 ≥ 0 → Real.sqrt x + Real.sqrt (x + 4) = 12 → x = 1225 / 36 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l986_98628


namespace NUMINAMATH_CALUDE_great_wall_scientific_notation_l986_98649

theorem great_wall_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), 21200000 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 2.12 ∧ n = 7 := by
  sorry

end NUMINAMATH_CALUDE_great_wall_scientific_notation_l986_98649


namespace NUMINAMATH_CALUDE_fraction_product_l986_98670

theorem fraction_product : (2 : ℚ) / 9 * 5 / 14 = 5 / 63 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_l986_98670


namespace NUMINAMATH_CALUDE_soccer_goals_proof_l986_98610

theorem soccer_goals_proof (total_goals : ℕ) : 
  (total_goals / 3 : ℚ) + (total_goals / 5 : ℚ) + 8 + 20 = total_goals →
  20 ≤ 27 →
  ∃ (individual_goals : List ℕ), 
    individual_goals.length = 9 ∧ 
    individual_goals.sum = 20 ∧
    ∀ g ∈ individual_goals, g ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_soccer_goals_proof_l986_98610


namespace NUMINAMATH_CALUDE_triangle_area_l986_98616

theorem triangle_area (a b c : ℝ) (h_right_angle : a^2 + b^2 = c^2)
  (h_angle : a / c = 1 / 2) (h_hypotenuse : c = 40) :
  (1 / 2) * a * b = 200 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l986_98616


namespace NUMINAMATH_CALUDE_sum_of_zeros_greater_than_one_l986_98643

noncomputable def g (x m : ℝ) : ℝ := Real.log x - x + x + 1 / (2 * x) - m

theorem sum_of_zeros_greater_than_one (m : ℝ) (x₁ x₂ : ℝ) 
  (h₁ : 0 < x₁) (h₂ : 0 < x₂) (h₃ : x₁ < x₂) 
  (hz₁ : g x₁ m = 0) (hz₂ : g x₂ m = 0) : 
  x₁ + x₂ > 1 := by
sorry

end NUMINAMATH_CALUDE_sum_of_zeros_greater_than_one_l986_98643


namespace NUMINAMATH_CALUDE_smallest_purple_balls_l986_98611

theorem smallest_purple_balls (x : ℕ) (y : ℕ) : 
  x > 0 ∧ 
  x % 10 = 0 ∧ 
  x % 8 = 0 ∧ 
  x % 3 = 0 ∧
  x / 8 + 10 = 8 ∧
  y = x / 10 + x / 8 + x / 3 + (x / 10 + 9) + 8 + x / 8 - x ∧
  y > 0 →
  y ≥ 25 :=
sorry

end NUMINAMATH_CALUDE_smallest_purple_balls_l986_98611


namespace NUMINAMATH_CALUDE_inequality_holds_for_p_greater_than_three_largest_interval_l986_98632

theorem inequality_holds_for_p_greater_than_three (p q : ℝ) (hp : p > 3) (hq : q > 0) :
  (7 * (p * q^2 + p^2 * q + 3 * q^2 + 3 * p * q)) / (p + q) > 3 * p^2 * q :=
sorry

theorem largest_interval (p q : ℝ) (hq : q > 0) :
  (∀ q > 0, (7 * (p * q^2 + p^2 * q + 3 * q^2 + 3 * p * q)) / (p + q) > 3 * p^2 * q) ↔ p > 3 :=
sorry

end NUMINAMATH_CALUDE_inequality_holds_for_p_greater_than_three_largest_interval_l986_98632


namespace NUMINAMATH_CALUDE_F_range_l986_98646

noncomputable def F (x : ℝ) : ℝ := |2 * x + 4| - |x - 2|

theorem F_range :
  Set.range F = Set.Ici (-4) :=
sorry

end NUMINAMATH_CALUDE_F_range_l986_98646


namespace NUMINAMATH_CALUDE_isosceles_triangle_altitude_l986_98618

theorem isosceles_triangle_altitude (a : ℝ) : 
  let r : ℝ := 7
  let circle_x_circumference : ℝ := 14 * Real.pi
  let circle_y_radius : ℝ := 2 * a
  (circle_x_circumference = 2 * Real.pi * r) →
  (circle_y_radius = r) →
  let h : ℝ := Real.sqrt 3 * a
  (h^2 + a^2 = r^2) :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_altitude_l986_98618


namespace NUMINAMATH_CALUDE_exists_158_consecutive_not_div_17_exists_div_17_in_159_consecutive_l986_98648

-- Define a function to calculate the sum of digits
def sum_of_digits (n : ℕ) : ℕ := sorry

-- Theorem 1: There exists a sequence of 158 consecutive integers where the sum of digits of each number is not divisible by 17
theorem exists_158_consecutive_not_div_17 : 
  ∃ (start : ℕ), ∀ (i : ℕ), i < 158 → ¬(17 ∣ sum_of_digits (start + i)) :=
sorry

-- Theorem 2: For any sequence of 159 consecutive integers, there exists at least one integer in the sequence whose sum of digits is divisible by 17
theorem exists_div_17_in_159_consecutive (start : ℕ) : 
  ∃ (i : ℕ), i < 159 ∧ (17 ∣ sum_of_digits (start + i)) :=
sorry

end NUMINAMATH_CALUDE_exists_158_consecutive_not_div_17_exists_div_17_in_159_consecutive_l986_98648


namespace NUMINAMATH_CALUDE_inequality_proof_l986_98622

theorem inequality_proof (a b c : ℝ) (h : (a + b + c) * c < 0) : b^2 > 4*a*c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l986_98622


namespace NUMINAMATH_CALUDE_min_dot_product_on_ellipse_l986_98664

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 9 + y^2 / 8 = 1

-- Define the center and left focus
def O : ℝ × ℝ := (0, 0)
def F : ℝ × ℝ := (-1, 0)

-- Define the dot product of OP and FP
def dot_product (x y : ℝ) : ℝ := x^2 + x + y^2

theorem min_dot_product_on_ellipse :
  ∀ x y : ℝ, is_on_ellipse x y →
  dot_product x y ≥ 6 :=
sorry

end NUMINAMATH_CALUDE_min_dot_product_on_ellipse_l986_98664


namespace NUMINAMATH_CALUDE_kolya_mistake_l986_98658

structure Box where
  blue : ℕ
  green : ℕ

def vasya_correct (b : Box) : Prop := b.blue ≥ 4
def kolya_correct (b : Box) : Prop := b.green ≥ 5
def petya_correct (b : Box) : Prop := b.blue ≥ 3 ∧ b.green ≥ 4
def misha_correct (b : Box) : Prop := b.blue ≥ 4 ∧ b.green ≥ 4

theorem kolya_mistake (b : Box) :
  (vasya_correct b ∧ petya_correct b ∧ misha_correct b ∧ ¬kolya_correct b) ∨
  (vasya_correct b ∧ petya_correct b ∧ misha_correct b ∧ kolya_correct b) :=
by sorry

end NUMINAMATH_CALUDE_kolya_mistake_l986_98658


namespace NUMINAMATH_CALUDE_no_tangent_point_largest_integer_a_l986_98641

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x - 1) * Real.exp x - (a / 2) * x^2

theorem no_tangent_point (a : ℝ) : ¬∃ x, f a x = 0 ∧ (deriv (f a)) x = 0 := sorry

theorem largest_integer_a :
  ∃ a : ℤ, (∀ x₁ x₂ : ℝ, x₂ > 0 → f a (x₁ + x₂) - f a (x₁ - x₂) > -2 * x₂) ∧
  (∀ b : ℤ, b > a → ∃ x₁ x₂ : ℝ, x₂ > 0 ∧ f b (x₁ + x₂) - f b (x₁ - x₂) ≤ -2 * x₂) ∧
  a = 3 := sorry

end NUMINAMATH_CALUDE_no_tangent_point_largest_integer_a_l986_98641


namespace NUMINAMATH_CALUDE_dog_adoptions_l986_98673

theorem dog_adoptions (dog_fee cat_fee : ℕ) (cat_adoptions : ℕ) (donation_fraction : ℚ) (donation_amount : ℕ) : 
  dog_fee = 15 →
  cat_fee = 13 →
  cat_adoptions = 3 →
  donation_fraction = 1/3 →
  donation_amount = 53 →
  ∃ (dog_adoptions : ℕ), 
    dog_adoptions = 8 ∧ 
    (↑donation_amount : ℚ) = donation_fraction * (↑dog_fee * ↑dog_adoptions + ↑cat_fee * ↑cat_adoptions) :=
by sorry

end NUMINAMATH_CALUDE_dog_adoptions_l986_98673


namespace NUMINAMATH_CALUDE_hexagon_y_coordinate_l986_98651

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a hexagon with six vertices -/
structure Hexagon where
  A : Point
  B : Point
  C : Point
  D : Point
  E : Point
  F : Point

/-- Calculates the area of a hexagon -/
def hexagonArea (h : Hexagon) : ℝ := sorry

/-- Checks if a hexagon has a vertical line of symmetry -/
def hasVerticalSymmetry (h : Hexagon) : Prop := sorry

theorem hexagon_y_coordinate 
  (h : Hexagon)
  (symm : hasVerticalSymmetry h)
  (aCoord : h.A = ⟨0, 0⟩)
  (bCoord : h.B = ⟨0, 6⟩)
  (eCoord : h.E = ⟨4, 0⟩)
  (dCoord : h.D.x = 4)
  (area : hexagonArea h = 58) :
  h.D.y = 14.5 := by sorry

end NUMINAMATH_CALUDE_hexagon_y_coordinate_l986_98651


namespace NUMINAMATH_CALUDE_min_distance_squared_l986_98640

theorem min_distance_squared (a b c d : ℝ) :
  (a - 2 * Real.exp a) / b = 1 →
  (2 - c) / d = 1 →
  ∃ (m : ℝ), ∀ (x y : ℝ),
    (x - 2 * Real.exp x) / y = 1 →
    (2 - x) / y = 1 →
    (a - x)^2 + (b - y)^2 ≥ m ∧
    m = 8 :=
sorry

end NUMINAMATH_CALUDE_min_distance_squared_l986_98640


namespace NUMINAMATH_CALUDE_product_QED_l986_98681

theorem product_QED (Q E D : ℂ) (hQ : Q = 6 + 3*I) (hE : E = -I) (hD : D = 6 - 3*I) :
  Q * E * D = -45 * I :=
by sorry

end NUMINAMATH_CALUDE_product_QED_l986_98681


namespace NUMINAMATH_CALUDE_percentage_difference_l986_98613

theorem percentage_difference (w q y z P : ℝ) : 
  w = q * (1 - P / 100) →
  q = y * 0.6 →
  z = y * 0.54 →
  z = w * 1.5 →
  P = 78.4 := by
sorry

end NUMINAMATH_CALUDE_percentage_difference_l986_98613


namespace NUMINAMATH_CALUDE_function_transformation_l986_98603

/-- Given a function f such that f(x-1) = 2x^2 + 3x for all x,
    prove that f(x) = 2x^2 + 7x + 5 for all x. -/
theorem function_transformation (f : ℝ → ℝ) 
    (h : ∀ x, f (x - 1) = 2 * x^2 + 3 * x) : 
    ∀ x, f x = 2 * x^2 + 7 * x + 5 := by
  sorry

end NUMINAMATH_CALUDE_function_transformation_l986_98603


namespace NUMINAMATH_CALUDE_restaurant_bill_proof_l986_98634

/-- The total bill for a group of friends dining at a restaurant -/
def total_bill : ℕ := 270

/-- The number of friends dining at the restaurant -/
def num_friends : ℕ := 10

/-- The extra amount each paying friend contributes to cover the non-paying friend -/
def extra_contribution : ℕ := 3

/-- The number of friends who pay the bill -/
def num_paying_friends : ℕ := num_friends - 1

theorem restaurant_bill_proof :
  total_bill = num_paying_friends * (total_bill / num_friends + extra_contribution) :=
sorry

end NUMINAMATH_CALUDE_restaurant_bill_proof_l986_98634


namespace NUMINAMATH_CALUDE_whale_length_from_relative_speed_l986_98667

/-- The length of a whale can be determined by the relative speed of two whales
    and the time taken for one to cross the other. -/
theorem whale_length_from_relative_speed (v_fast v_slow t : ℝ) (h1 : v_fast > v_slow) :
  (v_fast - v_slow) * t = (v_fast - v_slow) * 15 → v_fast = 18 → v_slow = 15 → (v_fast - v_slow) * 15 = 45 := by
  sorry

#check whale_length_from_relative_speed

end NUMINAMATH_CALUDE_whale_length_from_relative_speed_l986_98667


namespace NUMINAMATH_CALUDE_problem_solution_l986_98682

theorem problem_solution (t : ℝ) (x y : ℝ) 
  (h1 : x = 3 - 2*t) 
  (h2 : y = 3*t + 6) 
  (h3 : x = -6) : 
  y = 19.5 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l986_98682


namespace NUMINAMATH_CALUDE_sphere_in_cube_surface_area_l986_98659

/-- The surface area of a sphere inscribed in a cube with edge length 2 is 8π. -/
theorem sphere_in_cube_surface_area :
  let cube_edge : ℝ := 2
  let sphere_diameter : ℝ := cube_edge
  let sphere_radius : ℝ := sphere_diameter / 2
  let sphere_surface_area : ℝ := 4 * Real.pi * sphere_radius ^ 2
  sphere_surface_area = 8 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sphere_in_cube_surface_area_l986_98659


namespace NUMINAMATH_CALUDE_investment_problem_l986_98621

theorem investment_problem (total_interest desired_interest fixed_investment fixed_rate variable_rate : ℝ) :
  desired_interest = 980 →
  fixed_investment = 6000 →
  fixed_rate = 0.09 →
  variable_rate = 0.11 →
  total_interest = fixed_rate * fixed_investment + variable_rate * (total_interest - fixed_investment) →
  total_interest = 10000 := by
  sorry

end NUMINAMATH_CALUDE_investment_problem_l986_98621


namespace NUMINAMATH_CALUDE_number_plus_273_l986_98656

theorem number_plus_273 (x : ℤ) : x - 477 = 273 → x + 273 = 1023 := by
  sorry

end NUMINAMATH_CALUDE_number_plus_273_l986_98656


namespace NUMINAMATH_CALUDE_museum_visitors_l986_98620

theorem museum_visitors (yesterday : ℕ) (today_increase : ℕ) : 
  yesterday = 247 → today_increase = 131 → 
  yesterday + (yesterday + today_increase) = 625 := by
sorry

end NUMINAMATH_CALUDE_museum_visitors_l986_98620


namespace NUMINAMATH_CALUDE_g_pi_third_equals_one_l986_98629

/-- Given a function f and a constant w, φ, prove that g(π/3) = 1 -/
theorem g_pi_third_equals_one 
  (f : ℝ → ℝ) 
  (w φ : ℝ) 
  (h1 : ∀ x, f x = 5 * Real.cos (w * x + φ))
  (h2 : ∀ x, f (π/3 + x) = f (π/3 - x))
  (g : ℝ → ℝ)
  (h3 : ∀ x, g x = 4 * Real.sin (w * x + φ) + 1) :
  g (π/3) = 1 := by
sorry

end NUMINAMATH_CALUDE_g_pi_third_equals_one_l986_98629


namespace NUMINAMATH_CALUDE_engineer_designer_ratio_l986_98619

theorem engineer_designer_ratio (e d : ℕ) (h_total : (40 * e + 55 * d) / (e + d) = 45) :
  e = 2 * d := by
  sorry

end NUMINAMATH_CALUDE_engineer_designer_ratio_l986_98619


namespace NUMINAMATH_CALUDE_min_minutes_for_plan_b_cheaper_l986_98697

/-- Cost function for Plan A -/
def costA (x : ℕ) : ℕ :=
  if x ≤ 300 then 8 * x else 2400 + 7 * (x - 300)

/-- Cost function for Plan B -/
def costB (x : ℕ) : ℕ := 2500 + 4 * x

/-- Theorem stating the minimum number of minutes for Plan B to be cheaper -/
theorem min_minutes_for_plan_b_cheaper :
  ∀ x : ℕ, x < 301 → costA x ≤ costB x ∧
  ∀ y : ℕ, y ≥ 301 → costB y < costA y := by
  sorry

#check min_minutes_for_plan_b_cheaper

end NUMINAMATH_CALUDE_min_minutes_for_plan_b_cheaper_l986_98697


namespace NUMINAMATH_CALUDE_bess_frisbee_throws_l986_98612

/-- The problem of determining how many times Bess throws the Frisbee -/
theorem bess_frisbee_throws :
  ∀ (bess_throw_distance : ℕ) 
    (holly_throw_distance : ℕ) 
    (holly_throw_count : ℕ) 
    (total_distance : ℕ),
  bess_throw_distance = 20 →
  holly_throw_distance = 8 →
  holly_throw_count = 5 →
  total_distance = 200 →
  ∃ (bess_throw_count : ℕ),
    bess_throw_count * (2 * bess_throw_distance) + 
    holly_throw_count * holly_throw_distance = total_distance ∧
    bess_throw_count = 4 := by
  sorry

end NUMINAMATH_CALUDE_bess_frisbee_throws_l986_98612


namespace NUMINAMATH_CALUDE_cheddar_package_size_l986_98604

/-- The number of slices in a package of Swiss cheese -/
def swiss_slices_per_package : ℕ := 28

/-- The total number of slices bought for each type of cheese -/
def total_slices_per_type : ℕ := 84

/-- The number of slices in a package of cheddar cheese -/
def cheddar_slices_per_package : ℕ := sorry

theorem cheddar_package_size :
  cheddar_slices_per_package = swiss_slices_per_package :=
by
  sorry

#check cheddar_package_size

end NUMINAMATH_CALUDE_cheddar_package_size_l986_98604


namespace NUMINAMATH_CALUDE_johns_labor_cost_johns_specific_labor_cost_l986_98661

/-- Represents the problem of calculating labor costs for John's table-making business --/
theorem johns_labor_cost (trees : ℕ) (planks_per_tree : ℕ) (planks_per_table : ℕ) 
  (price_per_table : ℕ) (total_profit : ℕ) : ℕ :=
  let total_planks := trees * planks_per_tree
  let total_tables := total_planks / planks_per_table
  let total_revenue := total_tables * price_per_table
  let labor_cost := total_revenue - total_profit
  labor_cost

/-- The specific instance of John's labor cost calculation --/
theorem johns_specific_labor_cost : 
  johns_labor_cost 30 25 15 300 12000 = 3000 := by
  sorry

end NUMINAMATH_CALUDE_johns_labor_cost_johns_specific_labor_cost_l986_98661


namespace NUMINAMATH_CALUDE_equation_solutions_l986_98698

theorem equation_solutions : ∃ (x₁ x₂ : ℝ), 
  (x₁ = 3 ∧ x₂ = 5) ∧ 
  (∀ x : ℝ, (x - 2)^6 + (x - 6)^6 = 64 ↔ (x = x₁ ∨ x = x₂)) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l986_98698


namespace NUMINAMATH_CALUDE_function_equality_l986_98609

theorem function_equality (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (f (f x + f y)) = f x + y) →
  (∀ x : ℝ, f x = x) :=
by sorry

end NUMINAMATH_CALUDE_function_equality_l986_98609


namespace NUMINAMATH_CALUDE_smallest_valid_seating_l986_98663

/-- Represents a circular seating arrangement -/
structure CircularSeating where
  total_chairs : ℕ
  seated_people : ℕ

/-- Checks if the seating arrangement satisfies the condition -/
def is_valid_seating (seating : CircularSeating) : Prop :=
  seating.seated_people > 0 ∧
  seating.seated_people ≤ seating.total_chairs ∧
  ∀ (n : ℕ), n < seating.total_chairs → 
    ∃ (m : ℕ), m < seating.seated_people ∧ 
      (n = m * (seating.total_chairs / seating.seated_people) ∨ 
       n = m * (seating.total_chairs / seating.seated_people) + 1)

/-- The theorem to be proved -/
theorem smallest_valid_seating : 
  let seating := CircularSeating.mk 72 18
  is_valid_seating seating ∧ 
  ∀ (n : ℕ), n < 18 → ¬is_valid_seating (CircularSeating.mk 72 n) :=
by sorry

end NUMINAMATH_CALUDE_smallest_valid_seating_l986_98663


namespace NUMINAMATH_CALUDE_triangle_probability_l986_98693

def stick_lengths : List ℕ := [1, 4, 6, 8, 9, 10, 12, 15]

def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

def has_perimeter_gt_20 (a b c : ℕ) : Prop :=
  a + b + c > 20

def valid_triangle_count : ℕ := 16

def total_combinations : ℕ := Nat.choose 8 3

theorem triangle_probability : 
  (valid_triangle_count : ℚ) / total_combinations = 2 / 7 := by sorry

end NUMINAMATH_CALUDE_triangle_probability_l986_98693
