import Mathlib

namespace y_power_neg_x_value_l3890_389071

theorem y_power_neg_x_value (x y : ℝ) (h : |y - 2*x| + (x + y - 3)^2 = 0) : y^(-x) = 1/2 := by
  sorry

end y_power_neg_x_value_l3890_389071


namespace angle_relationship_l3890_389035

theorem angle_relationship (angle1 angle2 angle3 angle4 : Real) :
  (angle1 + angle2 = 90) →  -- angle1 and angle2 are complementary
  (angle3 + angle4 = 180) →  -- angle3 and angle4 are supplementary
  (angle1 = angle3) →  -- angle1 equals angle3
  (angle2 + 90 = angle4) :=  -- conclusion to prove
by
  sorry

end angle_relationship_l3890_389035


namespace no_valid_chessboard_config_l3890_389093

/-- A chessboard configuration is a function from (Fin 8 × Fin 8) to Fin 64 -/
def ChessboardConfig := Fin 8 × Fin 8 → Fin 64

/-- A 2x2 square on the chessboard -/
structure Square (config : ChessboardConfig) where
  row : Fin 7
  col : Fin 7

/-- The sum of numbers in a 2x2 square -/
def squareSum (config : ChessboardConfig) (square : Square config) : ℕ :=
  (config (square.row, square.col)).val + 1 +
  (config (square.row, square.col.succ)).val + 1 +
  (config (square.row.succ, square.col)).val + 1 +
  (config (square.row.succ, square.col.succ)).val + 1

/-- A valid configuration satisfies the divisibility condition for all 2x2 squares -/
def isValidConfig (config : ChessboardConfig) : Prop :=
  (∀ square : Square config, (squareSum config square) % 5 = 0) ∧
  Function.Injective config

theorem no_valid_chessboard_config : ¬ ∃ config : ChessboardConfig, isValidConfig config := by
  sorry

end no_valid_chessboard_config_l3890_389093


namespace some_number_value_l3890_389017

theorem some_number_value (a : ℕ) (some_number : ℕ) 
  (h1 : a = 105)
  (h2 : a^3 = 21 * 25 * some_number * 63) :
  some_number = 35 := by
  sorry

end some_number_value_l3890_389017


namespace specific_cube_unpainted_count_l3890_389011

/-- Represents a cube with painted strips on its faces -/
structure PaintedCube where
  size : Nat
  totalUnitCubes : Nat
  verticalStripWidth : Nat
  horizontalStripHeight : Nat

/-- Calculates the number of unpainted unit cubes in the painted cube -/
def unpaintedUnitCubes (cube : PaintedCube) : Nat :=
  sorry

/-- Theorem stating that a 6x6x6 cube with specific painted strips has 160 unpainted unit cubes -/
theorem specific_cube_unpainted_count :
  let cube : PaintedCube := {
    size := 6,
    totalUnitCubes := 216,
    verticalStripWidth := 2,
    horizontalStripHeight := 2
  }
  unpaintedUnitCubes cube = 160 := by
  sorry

end specific_cube_unpainted_count_l3890_389011


namespace simplified_fraction_sum_l3890_389018

theorem simplified_fraction_sum (a b : ℕ) (h : a = 54 ∧ b = 81) :
  let g := Nat.gcd a b
  (a / g) + (b / g) = 5 := by
sorry

end simplified_fraction_sum_l3890_389018


namespace area_of_ABCM_l3890_389043

structure Polygon where
  sides : ℕ
  sideLength : ℝ
  rightAngles : Bool

def intersectionPoint (p1 p2 p3 p4 : Point) : Point := sorry

def quadrilateralArea (p1 p2 p3 p4 : Point) : ℝ := sorry

theorem area_of_ABCM (poly : Polygon) (A B C G J M : Point) :
  poly.sides = 14 ∧
  poly.sideLength = 3 ∧
  poly.rightAngles = true ∧
  M = intersectionPoint A G C J →
  quadrilateralArea A B C M = 24.75 := by
  sorry

end area_of_ABCM_l3890_389043


namespace sheila_work_hours_l3890_389030

/-- Represents Sheila's work schedule and earnings --/
structure WorkSchedule where
  mwf_hours : ℝ  -- Hours worked on Monday, Wednesday, and Friday combined
  tt_hours : ℝ   -- Hours worked on Tuesday and Thursday combined
  hourly_rate : ℝ -- Hourly rate in dollars
  weekly_earnings : ℝ -- Total weekly earnings in dollars

/-- Theorem stating Sheila's work hours on Monday, Wednesday, and Friday --/
theorem sheila_work_hours (s : WorkSchedule) 
  (h1 : s.tt_hours = 12)  -- 6 hours each on Tuesday and Thursday
  (h2 : s.hourly_rate = 14)  -- $14 per hour
  (h3 : s.weekly_earnings = 504)  -- $504 per week
  : s.mwf_hours = 24 := by
  sorry


end sheila_work_hours_l3890_389030


namespace six_houses_configurations_l3890_389060

/-- Represents the material of a house -/
inductive Material
  | Brick
  | Wood

/-- A configuration of houses is a list of their materials -/
def Configuration := List Material

/-- Checks if a configuration is valid (no adjacent wooden houses) -/
def isValidConfiguration (config : Configuration) : Bool :=
  match config with
  | [] => true
  | [_] => true
  | Material.Wood :: Material.Wood :: _ => false
  | _ :: rest => isValidConfiguration rest

/-- Generates all possible configurations of n houses -/
def allConfigurations (n : Nat) : List Configuration :=
  match n with
  | 0 => [[]]
  | m + 1 => 
    let prev := allConfigurations m
    (prev.map (λ c => Material.Brick :: c)) ++ (prev.map (λ c => Material.Wood :: c))

/-- Counts the number of valid configurations for n houses -/
def countValidConfigurations (n : Nat) : Nat :=
  (allConfigurations n).filter isValidConfiguration |>.length

/-- The main theorem: there are 21 valid configurations for 6 houses -/
theorem six_houses_configurations :
  countValidConfigurations 6 = 21 := by
  sorry


end six_houses_configurations_l3890_389060


namespace square_diff_inequality_l3890_389050

theorem square_diff_inequality (a b : ℝ) (h1 : a < b) (h2 : b < 0) :
  (a^2 + b^2) * (a - b) > (a^2 - b^2) * (a + b) := by
  sorry

end square_diff_inequality_l3890_389050


namespace no_real_roots_third_polynomial_l3890_389023

/-- Given two quadratic polynomials with integer roots, prove the third has no real roots -/
theorem no_real_roots_third_polynomial (a b : ℝ) :
  (∃ x : ℤ, x^2 + a*x + b = 0) →
  (∃ y : ℤ, y^2 + a*y + (b+1) = 0) →
  ¬∃ z : ℝ, z^2 + a*z + (b+2) = 0 :=
by sorry

end no_real_roots_third_polynomial_l3890_389023


namespace bobs_age_l3890_389000

/-- Represents the ages of four siblings -/
structure SiblingAges where
  susan : ℕ
  arthur : ℕ
  tom : ℕ
  bob : ℕ

/-- Theorem stating Bob's age given the problem conditions -/
theorem bobs_age (ages : SiblingAges) : 
  ages.susan = 15 ∧ 
  ages.arthur = ages.susan + 2 ∧ 
  ages.tom = ages.bob - 3 ∧ 
  ages.susan + ages.arthur + ages.tom + ages.bob = 51 →
  ages.bob = 11 := by
sorry

end bobs_age_l3890_389000


namespace water_distribution_l3890_389074

theorem water_distribution (total_water : ℕ) (size_8oz : ℕ) (size_5oz : ℕ) (size_4oz : ℕ) 
  (num_8oz : ℕ) (num_5oz : ℕ) :
  total_water = 122 →
  size_8oz = 8 →
  size_5oz = 5 →
  size_4oz = 4 →
  num_8oz = 4 →
  num_5oz = 6 →
  (total_water - (num_8oz * size_8oz + num_5oz * size_5oz)) / size_4oz = 15 := by
sorry

end water_distribution_l3890_389074


namespace floor_plus_self_unique_solution_l3890_389010

theorem floor_plus_self_unique_solution (r : ℝ) : 
  (⌊r⌋ : ℝ) + r = 18.2 ↔ r = 9.2 := by sorry

end floor_plus_self_unique_solution_l3890_389010


namespace equal_savings_l3890_389056

theorem equal_savings (your_initial : ℕ) (your_weekly : ℕ) (friend_initial : ℕ) (friend_weekly : ℕ) (weeks : ℕ) :
  your_initial = 160 →
  your_weekly = 7 →
  friend_initial = 210 →
  friend_weekly = 5 →
  weeks = 25 →
  your_initial + your_weekly * weeks = friend_initial + friend_weekly * weeks :=
by sorry

end equal_savings_l3890_389056


namespace negation_of_universal_proposition_l3890_389009

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, |x| + x^2 ≥ 0) ↔ (∃ x₀ : ℝ, |x₀| + x₀^2 < 0) := by
  sorry

end negation_of_universal_proposition_l3890_389009


namespace divisible_by_24_l3890_389070

theorem divisible_by_24 (n : ℕ+) : ∃ k : ℤ, (n + 7)^2 - (n - 5)^2 = 24 * k := by
  sorry

end divisible_by_24_l3890_389070


namespace last_two_digits_of_factorial_sum_l3890_389059

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def last_two_digits (n : ℕ) : ℕ := n % 100

def factorial_sum : ℕ := (List.range 20).foldl (fun acc i => acc + factorial ((i + 1) * 5)) 0

theorem last_two_digits_of_factorial_sum :
  last_two_digits factorial_sum = 20 := by sorry

end last_two_digits_of_factorial_sum_l3890_389059


namespace b_is_positive_l3890_389084

theorem b_is_positive (x a : ℤ) (h1 : x < a) (h2 : a < 0) (b : ℤ) (h3 : b = x^2 - a^2) : b > 0 := by
  sorry

end b_is_positive_l3890_389084


namespace cans_collected_l3890_389014

theorem cans_collected (monday_cans tuesday_cans : ℕ) 
  (h1 : monday_cans = 71) 
  (h2 : tuesday_cans = 27) : 
  monday_cans + tuesday_cans = 98 := by
  sorry

end cans_collected_l3890_389014


namespace doughnut_profit_l3890_389047

/-- Calculate the profit from selling doughnuts -/
theorem doughnut_profit 
  (expenses : ℕ) 
  (num_doughnuts : ℕ) 
  (price_per_doughnut : ℕ) 
  (h1 : expenses = 53)
  (h2 : num_doughnuts = 25)
  (h3 : price_per_doughnut = 3) : 
  num_doughnuts * price_per_doughnut - expenses = 22 := by
  sorry

end doughnut_profit_l3890_389047


namespace find_a_empty_solution_set_l3890_389097

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

-- Theorem for part (1)
theorem find_a : 
  ∀ a : ℝ, (∀ x : ℝ, f a (2*x) ≤ 4 ↔ 0 ≤ x ∧ x ≤ 4) → a = 4 := by sorry

-- Theorem for part (2)
theorem empty_solution_set (m : ℝ) : 
  (∀ x : ℝ, ¬(f 4 x + f 4 (x + m) < 2)) ↔ (m ≥ 2 ∨ m ≤ -2) := by sorry

end find_a_empty_solution_set_l3890_389097


namespace geometry_number_theory_arrangement_l3890_389024

theorem geometry_number_theory_arrangement (n_geometry : ℕ) (n_number_theory : ℕ) :
  n_geometry = 4 →
  n_number_theory = 5 →
  (number_of_arrangements : ℕ) =
    Nat.choose (n_number_theory + 1) n_geometry :=
by sorry

end geometry_number_theory_arrangement_l3890_389024


namespace quadratic_one_solution_sum_sum_of_b_values_l3890_389052

theorem quadratic_one_solution_sum (b : ℝ) : 
  (∃! x, 3 * x^2 + b * x + 6 * x + 1 = 0) ↔ 
  (b = -6 + 2 * Real.sqrt 3 ∨ b = -6 - 2 * Real.sqrt 3) :=
by sorry

theorem sum_of_b_values : 
  (-6 + 2 * Real.sqrt 3) + (-6 - 2 * Real.sqrt 3) = -12 :=
by sorry

end quadratic_one_solution_sum_sum_of_b_values_l3890_389052


namespace vlad_sister_height_l3890_389028

/-- Converts feet and inches to total inches -/
def height_to_inches (feet : ℕ) (inches : ℕ) : ℕ := feet * 12 + inches

/-- Converts total inches to feet (discarding remaining inches) -/
def inches_to_feet (inches : ℕ) : ℕ := inches / 12

theorem vlad_sister_height :
  let vlad_height := height_to_inches 6 3
  let height_diff := 41
  let sister_inches := vlad_height - height_diff
  inches_to_feet sister_inches = 2 := by sorry

end vlad_sister_height_l3890_389028


namespace binary_sum_equality_l3890_389022

/-- Prove that the binary sum 1111₂ + 110₂ - 1001₂ + 1110₂ equals 11100₂ --/
theorem binary_sum_equality : 
  (0b1111 : Nat) + 0b110 - 0b1001 + 0b1110 = 0b11100 := by
  sorry

end binary_sum_equality_l3890_389022


namespace solve_for_m_l3890_389057

/-- Given that x = -2, y = 1, and mx + 3y = 7, prove that m = -2 -/
theorem solve_for_m (x y m : ℝ) 
  (hx : x = -2) 
  (hy : y = 1) 
  (heq : m * x + 3 * y = 7) : 
  m = -2 := by
sorry

end solve_for_m_l3890_389057


namespace new_sequence_69th_is_original_18th_l3890_389091

/-- Given a sequence, insert_between n seq inserts n elements between each pair of adjacent elements in seq -/
def insert_between (n : ℕ) (seq : ℕ → ℕ) : ℕ → ℕ := sorry

/-- The original sequence -/
def original_sequence : ℕ → ℕ := sorry

/-- The new sequence with 3 elements inserted between each pair of adjacent elements -/
def new_sequence : ℕ → ℕ := insert_between 3 original_sequence

theorem new_sequence_69th_is_original_18th :
  new_sequence 69 = original_sequence 18 := by sorry

end new_sequence_69th_is_original_18th_l3890_389091


namespace sphere_radius_ratio_l3890_389055

theorem sphere_radius_ratio (V_L V_S r_L r_S : ℝ) : 
  V_L = 432 * Real.pi ∧ 
  V_S = 0.08 * V_L ∧ 
  V_L = (4/3) * Real.pi * r_L^3 ∧ 
  V_S = (4/3) * Real.pi * r_S^3 →
  r_S / r_L = 1/2 := by
sorry

end sphere_radius_ratio_l3890_389055


namespace positive_difference_of_roots_l3890_389073

theorem positive_difference_of_roots (x : ℝ) : 
  let f : ℝ → ℝ := λ x => 2*x^2 - 10*x + 18 - (2*x + 34)
  let roots := {x : ℝ | f x = 0}
  ∃ (r₁ r₂ : ℝ), r₁ ∈ roots ∧ r₂ ∈ roots ∧ r₁ ≠ r₂ ∧ |r₁ - r₂| = 2 * Real.sqrt 17 :=
sorry

end positive_difference_of_roots_l3890_389073


namespace snickers_bought_l3890_389089

-- Define the cost of a single Snickers
def snickers_cost : ℚ := 3/2

-- Define the number of M&M packs bought
def mm_packs : ℕ := 3

-- Define the total amount paid
def total_paid : ℚ := 20

-- Define the change received
def change : ℚ := 8

-- Define the relationship between M&M pack cost and Snickers cost
def mm_pack_cost (s : ℚ) : ℚ := 2 * s

-- Theorem to prove
theorem snickers_bought :
  ∃ (n : ℕ), (n : ℚ) * snickers_cost + mm_packs * mm_pack_cost snickers_cost = total_paid - change ∧ n = 2 := by
  sorry


end snickers_bought_l3890_389089


namespace parallel_line_with_chord_l3890_389094

/-- Given a line parallel to 3x + 3y + 5 = 0 and intercepted by the circle x² + y² = 20
    with a chord length of 6√2, prove that the equation of the line is x + y ± 2 = 0 -/
theorem parallel_line_with_chord (a b c : ℝ) : 
  (∃ k : ℝ, a = 3 * k ∧ b = 3 * k) → -- Line is parallel to 3x + 3y + 5 = 0
  (∀ x y : ℝ, a * x + b * y + c = 0 → x^2 + y^2 ≤ 20) → -- Line intersects the circle
  (∃ x₁ y₁ x₂ y₂ : ℝ, 
    a * x₁ + b * y₁ + c = 0 ∧
    a * x₂ + b * y₂ + c = 0 ∧
    x₁^2 + y₁^2 = 20 ∧
    x₂^2 + y₂^2 = 20 ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 72) → -- Chord length is 6√2
  ∃ s : ℝ, (s = 1 ∨ s = -1) ∧ a * x + b * y + c = 0 ↔ x + y + 2 * s = 0 :=
by sorry

end parallel_line_with_chord_l3890_389094


namespace f_min_at_neg_seven_l3890_389077

/-- The quadratic function we're minimizing -/
def f (x : ℝ) := x^2 + 14*x - 20

/-- The theorem stating that f attains its minimum at x = -7 -/
theorem f_min_at_neg_seven :
  ∃ (x_min : ℝ), ∀ (x : ℝ), f x_min ≤ f x ∧ x_min = -7 :=
sorry

end f_min_at_neg_seven_l3890_389077


namespace area_triangle_DEF_is_seven_l3890_389067

/-- The area of triangle DEF in the given configuration --/
def area_triangle_DEF (side_length_PQRS : ℝ) (side_length_small_square : ℝ) : ℝ :=
  sorry

/-- The theorem stating the area of triangle DEF is 7 cm² --/
theorem area_triangle_DEF_is_seven
  (h1 : area_triangle_DEF 6 2 = 7) :
  ∃ (side_length_PQRS side_length_small_square : ℝ),
    side_length_PQRS^2 = 36 ∧
    side_length_small_square = 2 ∧
    area_triangle_DEF side_length_PQRS side_length_small_square = 7 :=
  sorry

end area_triangle_DEF_is_seven_l3890_389067


namespace gathering_gift_equation_l3890_389075

/-- Represents a gathering where gifts are exchanged -/
structure Gathering where
  attendees : ℕ
  gifts_exchanged : ℕ
  gift_exchange_rule : attendees > 0 → gifts_exchanged = attendees * (attendees - 1)

/-- Theorem: In a gathering where each pair of attendees exchanges a different small gift,
    if the total number of gifts exchanged is 56 and the number of attendees is x,
    then x(x-1) = 56 -/
theorem gathering_gift_equation (g : Gathering) (h1 : g.gifts_exchanged = 56) :
  g.attendees * (g.attendees - 1) = 56 := by
  sorry

end gathering_gift_equation_l3890_389075


namespace sale_increase_percentage_l3890_389046

theorem sale_increase_percentage
  (original_fee : ℝ)
  (fee_reduction_percentage : ℝ)
  (visitor_increase_percentage : ℝ)
  (h1 : original_fee = 1)
  (h2 : fee_reduction_percentage = 25)
  (h3 : visitor_increase_percentage = 60) :
  let new_fee := original_fee * (1 - fee_reduction_percentage / 100)
  let visitor_multiplier := 1 + visitor_increase_percentage / 100
  let sale_increase_percentage := (new_fee * visitor_multiplier - 1) * 100
  sale_increase_percentage = 20 :=
by sorry

end sale_increase_percentage_l3890_389046


namespace sin_alpha_for_given_point_l3890_389036

theorem sin_alpha_for_given_point : ∀ α : Real,
  let x : Real := -2
  let y : Real := 2 * Real.sqrt 3
  let r : Real := Real.sqrt (x^2 + y^2)
  (∃ A : ℝ × ℝ, A = (x, y) ∧ A.1 = r * Real.cos α ∧ A.2 = r * Real.sin α) →
  Real.sin α = Real.sqrt 3 / 2 := by
sorry

end sin_alpha_for_given_point_l3890_389036


namespace simplify_expression_l3890_389061

variable (R : Type*) [Ring R]
variable (a b : R)

theorem simplify_expression : (a - b) - (a + b) = -2 * b := by
  sorry

end simplify_expression_l3890_389061


namespace square_rectangle_area_relation_l3890_389083

theorem square_rectangle_area_relation (x : ℝ) :
  let square_side := x - 4
  let rect_length := x - 5
  let rect_width := x + 6
  let square_area := square_side ^ 2
  let rect_area := rect_length * rect_width
  rect_area = 3 * square_area →
  ∃ x₁ x₂ : ℝ, (x = x₁ ∨ x = x₂) ∧ x₁ + x₂ = 12.5 :=
by sorry

end square_rectangle_area_relation_l3890_389083


namespace commodity_sales_profit_l3890_389004

/-- Profit function for a commodity sale --/
def profit_function (x : ℝ) : ℝ := -10 * x^2 + 500 * x - 4000

/-- Sales quantity function --/
def sales_quantity (x : ℝ) : ℝ := -10 * x + 400

theorem commodity_sales_profit 
  (cost_price : ℝ) 
  (h_cost : cost_price = 10) 
  (h_domain : ∀ x, 0 < x → x ≤ 40 → sales_quantity x ≥ 0) :
  /- 1. Profit function is correct for the given domain -/
  (∀ x, 0 < x → x ≤ 40 → 
    profit_function x = (sales_quantity x) * (x - cost_price)) ∧
  /- 2. Selling price for $1250 profit that maximizes sales is $15 -/
  (∃ x, profit_function x = 1250 ∧ 
    sales_quantity x = (sales_quantity 15) ∧
    x = 15) ∧
  /- 3. Maximum profit when x ≥ 28 and y ≥ 50 is $2160 -/
  (∀ x, x ≥ 28 → sales_quantity x ≥ 50 → 
    profit_function x ≤ 2160) ∧
  (∃ x, x ≥ 28 ∧ sales_quantity x ≥ 50 ∧ 
    profit_function x = 2160) := by
  sorry

end commodity_sales_profit_l3890_389004


namespace cab_driver_income_l3890_389003

/-- Proves that given the incomes for days 1, 3, 4, and 5, and the average income for all 5 days, the income for day 2 must be $50. -/
theorem cab_driver_income
  (income_day1 : ℕ)
  (income_day3 : ℕ)
  (income_day4 : ℕ)
  (income_day5 : ℕ)
  (average_income : ℕ)
  (h1 : income_day1 = 45)
  (h3 : income_day3 = 60)
  (h4 : income_day4 = 65)
  (h5 : income_day5 = 70)
  (h_avg : average_income = 58)
  : ∃ (income_day2 : ℕ), income_day2 = 50 ∧ 
    (income_day1 + income_day2 + income_day3 + income_day4 + income_day5) / 5 = average_income :=
by
  sorry

end cab_driver_income_l3890_389003


namespace original_lines_per_sheet_l3890_389034

/-- Represents the number of lines on each sheet in the original report -/
def L : ℕ := 56

/-- The number of sheets in the original report -/
def original_sheets : ℕ := 20

/-- The number of characters per line in the original report -/
def original_chars_per_line : ℕ := 65

/-- The number of lines per sheet in the retyped report -/
def new_lines_per_sheet : ℕ := 65

/-- The number of characters per line in the retyped report -/
def new_chars_per_line : ℕ := 70

/-- The percentage reduction in the number of sheets -/
def reduction_percentage : ℚ := 20 / 100

theorem original_lines_per_sheet :
  L = 56 ∧
  original_sheets * L * original_chars_per_line = 
    (original_sheets * (1 - reduction_percentage)).floor * new_lines_per_sheet * new_chars_per_line :=
by sorry

end original_lines_per_sheet_l3890_389034


namespace not_all_vertices_on_same_branch_coordinates_of_Q_and_R_l3890_389015

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x * y = 1

-- Define the branches of the hyperbola
def branch1 (x y : ℝ) : Prop := hyperbola x y ∧ x > 0 ∧ y > 0
def branch2 (x y : ℝ) : Prop := hyperbola x y ∧ x < 0 ∧ y < 0

-- Define an equilateral triangle
def is_equilateral_triangle (P Q R : ℝ × ℝ) : Prop :=
  let (px, py) := P
  let (qx, qy) := Q
  let (rx, ry) := R
  (px - qx)^2 + (py - qy)^2 = (qx - rx)^2 + (qy - ry)^2 ∧
  (qx - rx)^2 + (qy - ry)^2 = (rx - px)^2 + (ry - py)^2

-- Theorem 1: Not all vertices can lie on the same branch
theorem not_all_vertices_on_same_branch 
  (P Q R : ℝ × ℝ) 
  (h_triangle : is_equilateral_triangle P Q R)
  (h_on_hyperbola : hyperbola P.1 P.2 ∧ hyperbola Q.1 Q.2 ∧ hyperbola R.1 R.2) :
  ¬(branch1 P.1 P.2 ∧ branch1 Q.1 Q.2 ∧ branch1 R.1 R.2) ∧
  ¬(branch2 P.1 P.2 ∧ branch2 Q.1 Q.2 ∧ branch2 R.1 R.2) :=
sorry

-- Theorem 2: Coordinates of Q and R given P(-1, -1)
theorem coordinates_of_Q_and_R
  (P Q R : ℝ × ℝ)
  (h_triangle : is_equilateral_triangle P Q R)
  (h_on_hyperbola : hyperbola P.1 P.2 ∧ hyperbola Q.1 Q.2 ∧ hyperbola R.1 R.2)
  (h_P : P = (-1, -1))
  (h_Q_R_branch1 : branch1 Q.1 Q.2 ∧ branch1 R.1 R.2) :
  (Q = (2 - Real.sqrt 3, 2 + Real.sqrt 3) ∧ R = (2 + Real.sqrt 3, 2 - Real.sqrt 3)) ∨
  (Q = (2 + Real.sqrt 3, 2 - Real.sqrt 3) ∧ R = (2 - Real.sqrt 3, 2 + Real.sqrt 3)) :=
sorry

end not_all_vertices_on_same_branch_coordinates_of_Q_and_R_l3890_389015


namespace greatest_four_digit_divisible_by_63_and_11_l3890_389049

/-- Reverses the digits of a positive integer -/
def reverseDigits (n : ℕ) : ℕ := sorry

/-- Checks if a number is a four-digit integer -/
def isFourDigit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

theorem greatest_four_digit_divisible_by_63_and_11 :
  ∃ (p : ℕ),
    isFourDigit p ∧
    p % 63 = 0 ∧
    (reverseDigits p) % 63 = 0 ∧
    p % 11 = 0 ∧
    ∀ (q : ℕ),
      isFourDigit q ∧
      q % 63 = 0 ∧
      (reverseDigits q) % 63 = 0 ∧
      q % 11 = 0 →
      q ≤ p ∧
    p = 9779 :=
by sorry

end greatest_four_digit_divisible_by_63_and_11_l3890_389049


namespace gloria_cypress_trees_l3890_389016

def cabin_price : ℕ := 129000
def initial_cash : ℕ := 150
def final_cash : ℕ := 350
def pine_trees : ℕ := 600
def maple_trees : ℕ := 24
def pine_price : ℕ := 200
def maple_price : ℕ := 300
def cypress_price : ℕ := 100

theorem gloria_cypress_trees :
  ∃ (cypress_trees : ℕ),
    cypress_trees * cypress_price + 
    pine_trees * pine_price + 
    maple_trees * maple_price = 
    cabin_price + final_cash - initial_cash ∧
    cypress_trees = 20 := by
  sorry

end gloria_cypress_trees_l3890_389016


namespace one_third_point_coordinates_l3890_389087

/-- 
Given two points (x₁, y₁) and (x₂, y₂) in a 2D plane, and a rational number t between 0 and 1,
this function returns the coordinates of a point that is t of the way from (x₁, y₁) to (x₂, y₂).
-/
def pointOnLine (x₁ y₁ x₂ y₂ t : ℚ) : ℚ × ℚ :=
  ((1 - t) * x₁ + t * x₂, (1 - t) * y₁ + t * y₂)

theorem one_third_point_coordinates :
  let p := pointOnLine 2 6 8 (-2) (1/3)
  p.1 = 4 ∧ p.2 = 10/3 := by sorry

end one_third_point_coordinates_l3890_389087


namespace special_rectangle_area_l3890_389039

/-- A rectangle with specific properties -/
structure SpecialRectangle where
  /-- The length of the rectangle -/
  length : ℝ
  /-- The width of the rectangle -/
  width : ℝ
  /-- The distance from the intersection of diagonals to the shorter side -/
  diag_dist : ℝ
  /-- Condition: The distance from the intersection of diagonals to the longer side is 2 cm more than to the shorter side -/
  diag_dist_diff : diag_dist + 2 = length / 2
  /-- Condition: The perimeter of the rectangle is 56 cm -/
  perimeter_cond : 2 * (length + width) = 56

/-- The area of a SpecialRectangle is 192 cm² -/
theorem special_rectangle_area (r : SpecialRectangle) : r.length * r.width = 192 := by
  sorry

end special_rectangle_area_l3890_389039


namespace quarterback_passes_l3890_389048

theorem quarterback_passes (total : ℕ) (left : ℕ) : 
  total = 50 → 
  left + 2 * left + (left + 2) = total → 
  left = 12 := by
sorry

end quarterback_passes_l3890_389048


namespace star_inequality_l3890_389019

-- Define the * operation
def star (m n : Int) : Int := (m + 2) * 3 - n

-- Theorem statement
theorem star_inequality : star 2 (-2) > star (-2) 2 := by
  sorry

end star_inequality_l3890_389019


namespace power_product_rule_l3890_389002

theorem power_product_rule (a : ℝ) : a^3 * a^4 = a^7 := by
  sorry

end power_product_rule_l3890_389002


namespace repeating_decimal_subtraction_l3890_389076

/-- Represents a repeating decimal with a three-digit repetend -/
def RepeatingDecimal (a b c : ℕ) : ℚ :=
  (a * 100 + b * 10 + c) / 999

theorem repeating_decimal_subtraction :
  RepeatingDecimal 8 6 4 - RepeatingDecimal 5 7 9 - RepeatingDecimal 1 3 5 = 50 / 333 := by
  sorry

end repeating_decimal_subtraction_l3890_389076


namespace cinema_seat_removal_l3890_389095

/-- The number of seats that should be removed from a cinema with
    total_seats arranged in rows of seats_per_row, given expected_attendees,
    to minimize unoccupied seats while ensuring full rows. -/
def seats_to_remove (total_seats seats_per_row expected_attendees : ℕ) : ℕ :=
  total_seats - (((expected_attendees + seats_per_row - 1) / seats_per_row) * seats_per_row)

/-- Theorem stating that for the given cinema setup, 88 seats should be removed. -/
theorem cinema_seat_removal :
  seats_to_remove 240 8 150 = 88 := by
  sorry

end cinema_seat_removal_l3890_389095


namespace barbaras_candies_l3890_389001

/-- Barbara's candy counting problem -/
theorem barbaras_candies (initial : ℕ) (bought : ℕ) (total : ℕ) : 
  initial = 9 → bought = 18 → total = initial + bought → total = 27 := by
  sorry

end barbaras_candies_l3890_389001


namespace log_two_x_equals_neg_two_l3890_389021

theorem log_two_x_equals_neg_two (x : ℝ) : 
  x = (Real.log 4 / Real.log 16) ^ (Real.log 16 / Real.log 4) → Real.log x / Real.log 2 = -2 :=
by sorry

end log_two_x_equals_neg_two_l3890_389021


namespace square_figure_division_l3890_389037

/-- Represents a rectangular figure composed of squares -/
structure SquareFigure where
  width : ℕ
  height : ℕ
  pattern : List (List Bool)

/-- Represents a cut in the figure -/
inductive Cut
  | Vertical : ℕ → Cut
  | Horizontal : ℕ → Cut

/-- Checks if a cut follows the sides of the squares -/
def isValidCut (figure : SquareFigure) (cut : Cut) : Prop :=
  match cut with
  | Cut.Vertical n => n > 0 ∧ n < figure.width
  | Cut.Horizontal n => n > 0 ∧ n < figure.height

/-- Checks if two cuts divide the figure into four parts -/
def dividesFourParts (figure : SquareFigure) (cut1 cut2 : Cut) : Prop :=
  isValidCut figure cut1 ∧ isValidCut figure cut2 ∧
  ((∃ n m, cut1 = Cut.Vertical n ∧ cut2 = Cut.Horizontal m) ∨
   (∃ n m, cut1 = Cut.Horizontal n ∧ cut2 = Cut.Vertical m))

/-- Checks if all parts are identical after cuts -/
def partsAreIdentical (figure : SquareFigure) (cut1 cut2 : Cut) : Prop :=
  sorry  -- Definition of identical parts

/-- Main theorem: The figure can be divided into four identical parts -/
theorem square_figure_division (figure : SquareFigure) :
  ∃ cut1 cut2, dividesFourParts figure cut1 cut2 ∧ partsAreIdentical figure cut1 cut2 :=
sorry


end square_figure_division_l3890_389037


namespace stratified_sampling_school_l3890_389078

/-- Proves that in a stratified sampling of a school, given the total number of students,
    the number of second-year students, and the number of second-year students selected,
    we can determine the total number of students selected. -/
theorem stratified_sampling_school (total : ℕ) (second_year : ℕ) (selected_second_year : ℕ) 
    (h1 : total = 1800) 
    (h2 : second_year = 600) 
    (h3 : selected_second_year = 21) :
    ∃ n : ℕ, n * second_year = selected_second_year * total ∧ n = 63 := by
  sorry

end stratified_sampling_school_l3890_389078


namespace nina_weekend_sales_l3890_389099

/-- Calculates the total money Nina made from jewelry sales over the weekend -/
def weekend_sales (necklace_price bracelet_price earring_price ensemble_price : ℚ)
                  (necklaces_sold bracelets_sold earrings_sold ensembles_sold : ℕ) : ℚ :=
  necklace_price * necklaces_sold +
  bracelet_price * bracelets_sold +
  earring_price * earrings_sold +
  ensemble_price * ensembles_sold

/-- Proves that Nina's weekend sales totaled $565.00 -/
theorem nina_weekend_sales :
  weekend_sales 25 15 10 45 5 10 20 2 = 565 := by
  sorry

end nina_weekend_sales_l3890_389099


namespace total_oranges_picked_l3890_389098

theorem total_oranges_picked (mary_oranges jason_oranges amanda_oranges : ℕ)
  (h1 : mary_oranges = 14)
  (h2 : jason_oranges = 41)
  (h3 : amanda_oranges = 56) :
  mary_oranges + jason_oranges + amanda_oranges = 111 := by
  sorry

end total_oranges_picked_l3890_389098


namespace compound_interest_rate_l3890_389013

/-- The compound interest rate that satisfies the given conditions -/
def interest_rate : ℝ := 20

/-- The principal amount (initial deposit) -/
noncomputable def principal : ℝ := 
  3000 / (1 + interest_rate / 100) ^ 3

theorem compound_interest_rate : 
  (principal * (1 + interest_rate / 100) ^ 3 = 3000) ∧ 
  (principal * (1 + interest_rate / 100) ^ 4 = 3600) := by
  sorry

#check compound_interest_rate

end compound_interest_rate_l3890_389013


namespace die_probability_l3890_389040

/-- A fair 8-sided die -/
def Die : Finset ℕ := Finset.range 8

/-- Perfect squares from 1 to 8 -/
def PerfectSquares : Finset ℕ := {1, 4}

/-- Even numbers from 1 to 8 -/
def EvenNumbers : Finset ℕ := {2, 4, 6, 8}

/-- The probability of rolling a number that is either a perfect square or an even number -/
theorem die_probability : 
  (Finset.card (PerfectSquares ∪ EvenNumbers) : ℚ) / Finset.card Die = 5 / 8 :=
sorry

end die_probability_l3890_389040


namespace kelly_apples_l3890_389008

/-- The number of apples Kelly needs to pick -/
def apples_to_pick : ℕ := 49

/-- The total number of apples Kelly will have after picking -/
def total_apples : ℕ := 105

/-- The number of apples Kelly has now -/
def current_apples : ℕ := total_apples - apples_to_pick

theorem kelly_apples : current_apples = 56 := by
  sorry

end kelly_apples_l3890_389008


namespace largest_angle_is_right_angle_l3890_389006

/-- Given a triangle ABC with sides a, b, c and corresponding altitudes ha, hb, hc -/
theorem largest_angle_is_right_angle 
  (a b c ha hb hc : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ ha > 0 ∧ hb > 0 ∧ hc > 0) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) 
  (h_volumes : (1 : ℝ) / (ha^2 * a)^2 = 1 / (hb^2 * b)^2 + 1 / (hc^2 * c)^2) :
  ∃ θ : ℝ, θ = Real.pi / 2 ∧ 
    θ = max (Real.arccos ((b^2 + c^2 - a^2) / (2 * b * c))) 
            (max (Real.arccos ((a^2 + c^2 - b^2) / (2 * a * c))) 
                 (Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b)))) := by
  sorry

end largest_angle_is_right_angle_l3890_389006


namespace austin_to_dallas_passes_three_buses_l3890_389005

/-- Represents the time in hours since midnight -/
def Time := ℝ

/-- Represents the distance between Dallas and Austin in arbitrary units -/
def Distance := ℝ

/-- Represents the schedule and movement of buses -/
structure BusSchedule where
  departure_interval : ℝ
  departure_offset : ℝ
  trip_duration : ℝ

/-- Calculates the number of buses passed during a trip -/
def buses_passed (austin_schedule dallas_schedule : BusSchedule) : ℕ :=
  sorry

theorem austin_to_dallas_passes_three_buses 
  (austin_schedule : BusSchedule) 
  (dallas_schedule : BusSchedule) : 
  austin_schedule.departure_interval = 2 ∧ 
  austin_schedule.departure_offset = 0.5 ∧
  austin_schedule.trip_duration = 6 ∧
  dallas_schedule.departure_interval = 2 ∧
  dallas_schedule.departure_offset = 0 ∧
  dallas_schedule.trip_duration = 6 →
  buses_passed austin_schedule dallas_schedule = 3 :=
sorry

end austin_to_dallas_passes_three_buses_l3890_389005


namespace value_of_a_l3890_389042

/-- Proves that if 0.5% of a equals 95 paise, then a equals 190 rupees -/
theorem value_of_a (a : ℚ) : (0.5 / 100) * a = 95 / 100 → a = 190 := by
  sorry

end value_of_a_l3890_389042


namespace hotel_profit_maximized_l3890_389033

/-- Represents a hotel with pricing and occupancy information -/
structure Hotel where
  totalRooms : ℕ
  basePrice : ℕ
  priceIncrement : ℕ
  occupancyDecrease : ℕ
  expensePerRoom : ℕ

/-- Calculates the profit for a given price increase -/
def profit (h : Hotel) (priceIncrease : ℕ) : ℤ :=
  let price := h.basePrice + priceIncrease * h.priceIncrement
  let occupiedRooms := h.totalRooms - priceIncrease * h.occupancyDecrease
  (price - h.expensePerRoom) * occupiedRooms

/-- Theorem stating that the profit is maximized at a specific price -/
theorem hotel_profit_maximized (h : Hotel) :
  h.totalRooms = 50 ∧
  h.basePrice = 180 ∧
  h.priceIncrement = 10 ∧
  h.occupancyDecrease = 1 ∧
  h.expensePerRoom = 20 →
  ∃ (maxPriceIncrease : ℕ),
    (∀ (x : ℕ), profit h x ≤ profit h maxPriceIncrease) ∧
    h.basePrice + maxPriceIncrease * h.priceIncrement = 350 :=
sorry

end hotel_profit_maximized_l3890_389033


namespace sqrt_expression_equality_l3890_389029

theorem sqrt_expression_equality : (Real.sqrt 3 + 1) * (Real.sqrt 3 - 1) + Real.sqrt 20 = 2 + 2 * Real.sqrt 5 := by
  sorry

end sqrt_expression_equality_l3890_389029


namespace xiaozhao_journey_l3890_389072

def movements : List Int := [1000, -900, 700, -1200, 1200, 100, -1100, -200]

def calorie_per_km : Nat := 7000

def final_position (moves : List Int) : Int :=
  moves.sum

def total_distance (moves : List Int) : Nat :=
  moves.map (Int.natAbs) |>.sum

theorem xiaozhao_journey :
  let pos := final_position movements
  let dist := total_distance movements
  (pos < 0 ∧ pos.natAbs = 400) ∧
  (dist * calorie_per_km / 1000 = 44800) := by
  sorry

end xiaozhao_journey_l3890_389072


namespace correct_stool_height_l3890_389066

/-- Calculates the height of a stool needed to reach a light bulb. -/
def stool_height (ceiling_height room_height alice_height alice_reach book_thickness : ℝ) : ℝ :=
  ceiling_height - room_height - (alice_height + alice_reach + book_thickness)

/-- Theorem stating the correct height of the stool needed. -/
theorem correct_stool_height :
  let ceiling_height : ℝ := 300
  let light_bulb_below_ceiling : ℝ := 15
  let alice_height : ℝ := 160
  let alice_reach : ℝ := 50
  let book_thickness : ℝ := 5
  stool_height ceiling_height light_bulb_below_ceiling alice_height alice_reach book_thickness = 70 := by
  sorry

#eval stool_height 300 15 160 50 5

end correct_stool_height_l3890_389066


namespace linear_function_inverse_sum_l3890_389063

/-- Given a linear function f and its inverse f⁻¹, prove that a + b + c = 0 --/
theorem linear_function_inverse_sum (a b c : ℝ) 
  (f : ℝ → ℝ) (f_inv : ℝ → ℝ)
  (h1 : ∀ x, f x = a * x + b)
  (h2 : ∀ x, f_inv x = b * x + a + c)
  (h3 : ∀ x, f (f_inv x) = x) :
  a + b + c = 0 := by
  sorry

end linear_function_inverse_sum_l3890_389063


namespace right_triangle_trig_l3890_389026

theorem right_triangle_trig (D E F : ℝ) (h1 : D = 90) (h2 : E = 8) (h3 : F = 17) :
  let cosF := E / F
  let sinF := Real.sqrt (F^2 - E^2) / F
  cosF = 8 / 17 ∧ sinF = 15 / 17 := by
sorry

end right_triangle_trig_l3890_389026


namespace ball_ground_hit_time_l3890_389044

/-- The time at which a ball hits the ground when thrown downward -/
theorem ball_ground_hit_time :
  let h (t : ℝ) := -16 * t^2 - 30 * t + 200
  ∃ t : ℝ, h t = 0 ∧ t = (-15 + Real.sqrt 3425) / 16 :=
by sorry

end ball_ground_hit_time_l3890_389044


namespace bobs_current_time_l3890_389020

/-- Given that Bob's sister runs a mile in 320 seconds, and Bob needs to improve his time by 50% to match his sister's time, prove that Bob's current time is 480 seconds. -/
theorem bobs_current_time (sister_time : ℝ) (improvement_rate : ℝ) (bob_time : ℝ) 
  (h1 : sister_time = 320)
  (h2 : improvement_rate = 0.5)
  (h3 : bob_time = sister_time + sister_time * improvement_rate) :
  bob_time = 480 := by
  sorry

end bobs_current_time_l3890_389020


namespace sqrt_56_58_fraction_existence_l3890_389058

theorem sqrt_56_58_fraction_existence (q : ℕ+) :
  q ≠ 1 → q ≠ 3 → ∃ p : ℤ, Real.sqrt 56 < (p : ℚ) / q ∧ (p : ℚ) / q < Real.sqrt 58 :=
by sorry

end sqrt_56_58_fraction_existence_l3890_389058


namespace unique_original_message_exists_l3890_389085

/-- Represents a cryptogram as a list of characters -/
def Cryptogram := List Char

/-- Represents a bijective letter substitution -/
def Substitution := Char → Char

/-- The first cryptogram -/
def cryptogram1 : Cryptogram := 
  ['М', 'И', 'М', 'О', 'П', 'Р', 'А', 'С', 'Т', 'Е', 'Т', 'И', 'Р', 'А', 'С', 'И', 'С', 'П', 'Д', 'А', 'И', 'С', 'А', 'Ф', 'Е', 'И', 'И', 'Б', 'О', 'Е', 'Т', 'К', 'Ж', 'Р', 'Г', 'Л', 'Е', 'О', 'Л', 'О', 'И', 'Ш', 'И', 'С', 'А', 'Н', 'Н', 'С', 'Й', 'С', 'А', 'О', 'О', 'Л', 'Т', 'Л', 'Е', 'Я', 'Т', 'У', 'И', 'Ц', 'В', 'Ы', 'И', 'П', 'И', 'Я', 'Д', 'П', 'И', 'Щ', 'П', 'Ь', 'П', 'С', 'Е', 'Ю', 'Я', 'Я']

/-- The second cryptogram -/
def cryptogram2 : Cryptogram := 
  ['У', 'Щ', 'Ф', 'М', 'Ш', 'П', 'Д', 'Р', 'Е', 'Ц', 'Ч', 'Е', 'Ш', 'Ю', 'Ч', 'Д', 'А', 'К', 'Е', 'Ч', 'М', 'Д', 'В', 'К', 'Ш', 'Б', 'Е', 'Е', 'Ч', 'Д', 'Ф', 'Э', 'П', 'Й', 'Щ', 'Г', 'Ш', 'Ф', 'Щ', 'Ц', 'Е', 'Ю', 'Щ', 'Ф', 'П', 'М', 'Е', 'Ч', 'П', 'М', 'Р', 'Р', 'М', 'Е', 'О', 'Ч', 'Х', 'Е', 'Ш', 'Р', 'Т', 'Г', 'И', 'Ф', 'Р', 'С', 'Я', 'Ы', 'Л', 'К', 'Д', 'Ф', 'Ф', 'Е', 'Е']

/-- The original message -/
def original_message : Cryptogram := 
  ['Ш', 'Е', 'С', 'Т', 'А', 'Я', 'О', 'Л', 'И', 'М', 'П', 'И', 'А', 'Д', 'А', 'П', 'О', 'К', 'Р', 'И', 'П', 'Т', 'О', 'Г', 'Р', 'А', 'Ф', 'И', 'И', 'П', 'О', 'С', 'В', 'Я', 'Щ', 'Е', 'Н', 'А', 'С', 'Е', 'М', 'И', 'Д', 'Е', 'С', 'Я', 'Т', 'И', 'П', 'Я', 'Т', 'И', 'Л', 'Е', 'Т', 'И', 'Ю', 'С', 'П', 'Е', 'Ц', 'И', 'А', 'Л', 'Ь', 'Н', 'О', 'Й', 'С', 'Л', 'У', 'Ж', 'Б', 'Ы', 'Р', 'О', 'С', 'С', 'И', 'И']

/-- Predicate to check if a list is a permutation of another list -/
def is_permutation (l1 l2 : List α) : Prop := sorry

/-- Predicate to check if a function is bijective on a given list -/
def is_bijective_on (f : α → β) (l : List α) : Prop := sorry

/-- Main theorem: There exists a unique original message that satisfies the cryptogram conditions -/
theorem unique_original_message_exists : 
  ∃! (msg : Cryptogram), 
    (is_permutation msg cryptogram1) ∧ 
    (∃ (subst : Substitution), 
      (is_bijective_on subst msg) ∧ 
      (cryptogram2 = msg.map subst)) :=
sorry

end unique_original_message_exists_l3890_389085


namespace betty_savings_ratio_l3890_389082

theorem betty_savings_ratio (wallet_cost parents_gift grandparents_gift needed_more initial_savings : ℚ) :
  wallet_cost = 100 →
  parents_gift = 15 →
  grandparents_gift = 2 * parents_gift →
  needed_more = 5 →
  initial_savings + parents_gift + grandparents_gift = wallet_cost - needed_more →
  initial_savings / wallet_cost = 1 / 2 := by
sorry

end betty_savings_ratio_l3890_389082


namespace sum_of_coefficients_l3890_389032

theorem sum_of_coefficients (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℝ) :
  (∀ x : ℝ, (1 - 3*x + x^2)^5 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + 
                               a₆*x^6 + a₇*x^7 + a₈*x^8 + a₉*x^9 + a₁₀*x^10) →
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ + a₁₀ = -2 :=
by sorry

end sum_of_coefficients_l3890_389032


namespace quadratic_inequality_and_constraint_l3890_389012

theorem quadratic_inequality_and_constraint (a b : ℝ) : 
  (∀ x, x < 1 ∨ x > b ↔ a * x^2 - 3 * x + 2 > 0) →
  b > 1 →
  (a = 1 ∧ b = 2) ∧
  (∀ x y, x > 0 → y > 0 → a / x + b / y = 1 → 2 * x + y ≥ 8) ∧
  (∀ k, (∀ x y, x > 0 → y > 0 → a / x + b / y = 1 → 2 * x + y ≥ k^2 + k + 2) ↔ -3 ≤ k ∧ k ≤ 2) :=
by sorry

end quadratic_inequality_and_constraint_l3890_389012


namespace billy_laundry_loads_l3890_389068

/-- Represents the time taken for each chore in minutes -/
structure ChoreTime where
  sweeping : ℕ  -- time to sweep one room
  dishwashing : ℕ  -- time to wash one dish
  laundry : ℕ  -- time to do one load of laundry

/-- Represents the chores done by each child -/
structure Chores where
  rooms_swept : ℕ
  dishes_washed : ℕ
  laundry_loads : ℕ

def total_time (ct : ChoreTime) (c : Chores) : ℕ :=
  ct.sweeping * c.rooms_swept + ct.dishwashing * c.dishes_washed + ct.laundry * c.laundry_loads

theorem billy_laundry_loads (ct : ChoreTime) (anna billy : Chores) :
  ct.sweeping = 3 →
  ct.dishwashing = 2 →
  ct.laundry = 9 →
  anna.rooms_swept = 10 →
  billy.dishes_washed = 6 →
  anna.dishes_washed = 0 →
  anna.laundry_loads = 0 →
  billy.rooms_swept = 0 →
  total_time ct anna = total_time ct billy →
  billy.laundry_loads = 2 := by
  sorry

end billy_laundry_loads_l3890_389068


namespace weight_of_six_meter_rod_l3890_389062

/-- Given a uniform steel rod with specified properties, this theorem proves
    the weight of a 6 m piece of the same rod. -/
theorem weight_of_six_meter_rod (r : ℝ) (ρ : ℝ) : 
  let rod_length : ℝ := 11.25
  let rod_weight : ℝ := 42.75
  let piece_length : ℝ := 6
  let rod_volume := π * r^2 * rod_length
  let piece_volume := π * r^2 * piece_length
  let density := rod_weight / rod_volume
  piece_volume * density = 22.8 := by
  sorry

end weight_of_six_meter_rod_l3890_389062


namespace fraction_equality_l3890_389096

theorem fraction_equality : (8 : ℚ) / (4 * 25) = 0.8 / (0.4 * 25) := by
  sorry

end fraction_equality_l3890_389096


namespace total_cats_l3890_389054

theorem total_cats (white : ℕ) (black : ℕ) (gray : ℕ) 
  (h_white : white = 2) 
  (h_black : black = 10) 
  (h_gray : gray = 3) : 
  white + black + gray = 15 := by
  sorry

end total_cats_l3890_389054


namespace two_a_plus_b_value_l3890_389069

theorem two_a_plus_b_value (a b : ℚ) 
  (eq1 : 3 * a - b = 8) 
  (eq2 : 4 * b + 7 * a = 13) : 
  2 * a + b = 73 / 19 := by
sorry

end two_a_plus_b_value_l3890_389069


namespace tan_product_equals_two_l3890_389027

theorem tan_product_equals_two : 
  (1 + Real.tan (23 * π / 180)) * (1 + Real.tan (22 * π / 180)) = 2 := by
  sorry

end tan_product_equals_two_l3890_389027


namespace chairs_arrangement_l3890_389090

/-- Given a total number of chairs and chairs per row, calculates the number of rows -/
def calculate_rows (total_chairs : ℕ) (chairs_per_row : ℕ) : ℕ :=
  total_chairs / chairs_per_row

/-- Theorem: For 432 chairs arranged in rows of 16, there are 27 rows -/
theorem chairs_arrangement :
  calculate_rows 432 16 = 27 := by
  sorry

end chairs_arrangement_l3890_389090


namespace intersection_points_vary_l3890_389092

theorem intersection_points_vary (A B C : ℝ) (hA : A > 0) (hC : C > 0) (hB : B ≥ 0) :
  ∃ x y : ℝ, y = A * x^2 + B * x + C ∧ y^2 + 2 * x = x^2 + 4 * y + C ∧
  ∃ A' B' C' : ℝ, A' > 0 ∧ C' > 0 ∧ B' ≥ 0 ∧
    (∃ x1 y1 x2 y2 : ℝ, x1 ≠ x2 ∧
      y1 = A' * x1^2 + B' * x1 + C' ∧ y1^2 + 2 * x1 = x1^2 + 4 * y1 + C' ∧
      y2 = A' * x2^2 + B' * x2 + C' ∧ y2^2 + 2 * x2 = x2^2 + 4 * y2 + C') :=
by sorry

end intersection_points_vary_l3890_389092


namespace polygon_sides_l3890_389081

theorem polygon_sides (n : ℕ) : 
  (n - 2) * 180 = 3 * 360 → n = 8 :=
by sorry

end polygon_sides_l3890_389081


namespace min_value_expression_l3890_389064

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x + 1 / y^2) * (x + 1 / y^2 - 500) + (y + 1 / x^2) * (y + 1 / x^2 - 500) ≥ -125000 := by
  sorry

end min_value_expression_l3890_389064


namespace first_purchase_correct_max_profit_correct_l3890_389045

/-- Represents the types of dolls -/
inductive DollType
| A
| B

/-- Represents the purchase and selling prices of dolls -/
def price (t : DollType) : ℕ × ℕ :=
  match t with
  | DollType.A => (20, 25)
  | DollType.B => (15, 18)

/-- The total number of dolls purchased -/
def total_dolls : ℕ := 100

/-- The total cost of the first purchase -/
def total_cost : ℕ := 1650

/-- Calculates the number of each type of doll in the first purchase -/
def first_purchase : ℕ × ℕ := sorry

/-- Calculates the profit for a given number of A dolls in the second purchase -/
def profit (x : ℕ) : ℕ := sorry

/-- Finds the maximum profit and corresponding number of dolls for the second purchase -/
def max_profit : ℕ × ℕ × ℕ := sorry

theorem first_purchase_correct :
  first_purchase = (30, 70) := by sorry

theorem max_profit_correct :
  max_profit = (366, 33, 67) := by sorry

end first_purchase_correct_max_profit_correct_l3890_389045


namespace sufficient_not_necessary_l3890_389080

theorem sufficient_not_necessary (a : ℝ) : 
  (∀ a, a > 4 → a^2 > 16) ∧ 
  (∃ a, a^2 > 16 ∧ ¬(a > 4)) := by
  sorry

end sufficient_not_necessary_l3890_389080


namespace harmonic_mean_leq_geometric_mean_l3890_389038

theorem harmonic_mean_leq_geometric_mean (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  2 / (1/a + 1/b) ≤ Real.sqrt (a * b) := by
  sorry

end harmonic_mean_leq_geometric_mean_l3890_389038


namespace teacher_instruction_l3890_389041

theorem teacher_instruction (x : ℝ) : ((x - 2) * 3 + 3) * 3 = 63 ↔ x = 8 := by sorry

end teacher_instruction_l3890_389041


namespace square_side_length_l3890_389051

theorem square_side_length (s : ℝ) (h : s > 0) : s^2 = 2 * (4 * s) → s = 8 := by
  sorry

end square_side_length_l3890_389051


namespace line_intersection_area_ratio_l3890_389079

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in the form y = mx + b -/
structure Line where
  m : ℝ
  b : ℝ

/-- Calculates the area of a triangle given three points -/
def triangleArea (A B C : Point) : ℝ :=
  sorry

/-- Theorem: Given a line y = c - x where 0 < c < 6, intersecting the y-axis at P
    and the line x = 6 at S, if the ratio of the area of triangle QRS to the area
    of triangle QOP is 4:16, then c = 4 -/
theorem line_intersection_area_ratio (c : ℝ) 
  (h1 : 0 < c) (h2 : c < 6) : 
  let l : Line := { m := -1, b := c }
  let P : Point := { x := 0, y := c }
  let S : Point := { x := 6, y := c - 6 }
  let Q : Point := { x := c, y := 0 }
  let R : Point := { x := 6, y := 0 }
  let O : Point := { x := 0, y := 0 }
  triangleArea Q R S / triangleArea Q O P = 4 / 16 →
  c = 4 := by
  sorry

end line_intersection_area_ratio_l3890_389079


namespace relay_race_total_time_l3890_389007

/-- The time taken by four athletes to complete a relay race -/
def relay_race_time (athlete1_time : ℕ) : ℕ :=
  let athlete2_time := athlete1_time + 10
  let athlete3_time := athlete2_time - 15
  let athlete4_time := athlete1_time - 25
  athlete1_time + athlete2_time + athlete3_time + athlete4_time

/-- Theorem stating that the total time for the relay race is 200 seconds -/
theorem relay_race_total_time : relay_race_time 55 = 200 := by
  sorry

end relay_race_total_time_l3890_389007


namespace inequality_equivalence_l3890_389086

theorem inequality_equivalence (x : ℝ) : x + 1 < (4 + 3 * x) / 2 ↔ x > -2 := by
  sorry

end inequality_equivalence_l3890_389086


namespace sixteen_solutions_l3890_389065

/-- The function f(x) = x^2 - 3x --/
def f (x : ℝ) : ℝ := x^2 - 3*x

/-- The fourth composition of f --/
def f_4 (x : ℝ) : ℝ := f (f (f (f x)))

/-- There are exactly 16 distinct real solutions to f(f(f(f(c)))) = 6 --/
theorem sixteen_solutions : ∃! (s : Finset ℝ), s.card = 16 ∧ ∀ c, c ∈ s ↔ f_4 c = 6 := by sorry

end sixteen_solutions_l3890_389065


namespace range_of_a_l3890_389053

theorem range_of_a (a : ℝ) : (∃ x : ℝ, |x - a| + |x - 1| ≤ 3) → -2 ≤ a ∧ a ≤ 4 := by
  sorry

end range_of_a_l3890_389053


namespace geometric_sequence_common_ratio_l3890_389031

theorem geometric_sequence_common_ratio : 
  let a : ℕ → ℝ := fun n => (4 : ℝ) ^ (2 * n + 1)
  ∀ n : ℕ, a (n + 1) / a n = 16 := by
  sorry

end geometric_sequence_common_ratio_l3890_389031


namespace bobbys_remaining_candy_l3890_389088

/-- Given Bobby's initial candy count and the amounts eaten, prove that the remaining candy count is 8. -/
theorem bobbys_remaining_candy (initial_candy : ℕ) (first_eaten : ℕ) (second_eaten : ℕ)
  (h1 : initial_candy = 22)
  (h2 : first_eaten = 9)
  (h3 : second_eaten = 5) :
  initial_candy - first_eaten - second_eaten = 8 := by
  sorry

end bobbys_remaining_candy_l3890_389088


namespace number_raised_to_fourth_l3890_389025

theorem number_raised_to_fourth : ∃ x : ℝ, 121 * x^4 = 75625 ∧ x = 5 := by
  sorry

end number_raised_to_fourth_l3890_389025
