import Mathlib

namespace NUMINAMATH_CALUDE_shaded_region_perimeter_l3943_394302

theorem shaded_region_perimeter (r : ℝ) (h : r = 7) :
  let circle_fraction : ℝ := 3 / 4
  let arc_length := circle_fraction * (2 * π * r)
  let radii_length := 2 * r
  radii_length + arc_length = 14 + (21 / 2) * π := by
  sorry

end NUMINAMATH_CALUDE_shaded_region_perimeter_l3943_394302


namespace NUMINAMATH_CALUDE_inequality_solution_exists_implies_a_leq_4_l3943_394327

theorem inequality_solution_exists_implies_a_leq_4 :
  (∃ x : ℝ, |x - 2| - |x + 2| ≥ a) → a ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_exists_implies_a_leq_4_l3943_394327


namespace NUMINAMATH_CALUDE_pump_fill_time_l3943_394379

/-- The time it takes for a pump to fill a tank without a leak, given information about filling and emptying rates with a leak present. -/
theorem pump_fill_time (fill_with_leak : ℝ) (empty_time : ℝ) (h1 : fill_with_leak = 10) (h2 : empty_time = 10) :
  ∃ T : ℝ, T = 5 ∧ (1 / T - 1 / empty_time = 1 / fill_with_leak) := by
  sorry

end NUMINAMATH_CALUDE_pump_fill_time_l3943_394379


namespace NUMINAMATH_CALUDE_people_who_left_train_l3943_394376

/-- The number of people who left a train given initial, boarding, and final passenger counts. -/
theorem people_who_left_train (initial : ℕ) (boarded : ℕ) (final : ℕ) : 
  initial = 82 → boarded = 17 → final = 73 → initial + boarded - final = 26 := by
  sorry

end NUMINAMATH_CALUDE_people_who_left_train_l3943_394376


namespace NUMINAMATH_CALUDE_initial_men_correct_l3943_394398

/-- Represents the initial number of men working on the project -/
def initial_men : ℕ := 27

/-- Represents the number of days to complete the project with the initial group -/
def initial_days : ℕ := 40

/-- Represents the number of days worked before some men leave -/
def days_before_leaving : ℕ := 18

/-- Represents the number of men who leave the project -/
def men_leaving : ℕ := 12

/-- Represents the number of days to complete the project after some men leave -/
def remaining_days : ℕ := 40

/-- Theorem stating that the initial number of men is correct given the conditions -/
theorem initial_men_correct :
  (initial_men : ℚ) * (days_before_leaving : ℚ) / initial_days +
  (initial_men - men_leaving : ℚ) * remaining_days / initial_days = 1 :=
sorry

end NUMINAMATH_CALUDE_initial_men_correct_l3943_394398


namespace NUMINAMATH_CALUDE_last_bead_is_white_l3943_394389

/-- Represents the color of a bead -/
inductive BeadColor
| White
| Black
| Red

/-- Returns the color of the nth bead in the pattern -/
def nthBeadColor (n : ℕ) : BeadColor :=
  match n % 6 with
  | 1 => BeadColor.White
  | 2 | 3 => BeadColor.Black
  | _ => BeadColor.Red

/-- The total number of beads in the necklace -/
def totalBeads : ℕ := 85

theorem last_bead_is_white :
  nthBeadColor totalBeads = BeadColor.White := by
  sorry

end NUMINAMATH_CALUDE_last_bead_is_white_l3943_394389


namespace NUMINAMATH_CALUDE_choose_and_assign_officers_l3943_394375

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The factorial of a natural number -/
def factorial (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to choose 3 people from 5 and assign them to 3 distinct roles -/
def waysToChooseAndAssign : ℕ := choose 5 3 * factorial 3

theorem choose_and_assign_officers :
  waysToChooseAndAssign = 60 := by
  sorry

end NUMINAMATH_CALUDE_choose_and_assign_officers_l3943_394375


namespace NUMINAMATH_CALUDE_tan_theta_for_pure_imaginary_l3943_394382

theorem tan_theta_for_pure_imaginary (θ : Real) :
  let z : ℂ := Complex.mk (Real.sin θ - 3/5) (Real.cos θ - 4/5)
  (z.re = 0 ∧ z.im ≠ 0) → Real.tan θ = -3/4 := by
  sorry

end NUMINAMATH_CALUDE_tan_theta_for_pure_imaginary_l3943_394382


namespace NUMINAMATH_CALUDE_bob_hair_growth_time_l3943_394320

/-- Represents the growth of Bob's hair over time -/
def hair_growth (initial_length : ℝ) (growth_rate : ℝ) (time : ℝ) : ℝ :=
  initial_length + growth_rate * time

/-- Theorem stating the time it takes for Bob's hair to grow from 6 inches to 36 inches -/
theorem bob_hair_growth_time :
  let initial_length : ℝ := 6
  let final_length : ℝ := 36
  let monthly_growth_rate : ℝ := 0.5
  let years : ℝ := 5
  hair_growth initial_length (monthly_growth_rate * 12) years = final_length := by
  sorry

#check bob_hair_growth_time

end NUMINAMATH_CALUDE_bob_hair_growth_time_l3943_394320


namespace NUMINAMATH_CALUDE_exponential_growth_dominates_power_growth_l3943_394344

theorem exponential_growth_dominates_power_growth 
  (a : ℝ) (α : ℝ) (ha : a > 1) (hα : α > 0) :
  ∃ x₀ : ℝ, x₀ > 0 ∧ ∀ x > x₀, 
    (deriv (fun x => a^x) x) / a^x > (deriv (fun x => x^α) x) / x^α :=
sorry

end NUMINAMATH_CALUDE_exponential_growth_dominates_power_growth_l3943_394344


namespace NUMINAMATH_CALUDE_track_extension_calculation_l3943_394346

/-- Theorem: Track Extension Calculation
Given a train track with an elevation gain of 600 meters,
changing the gradient from 3% to 2% results in a track extension of 10 km. -/
theorem track_extension_calculation (elevation_gain : ℝ) (initial_gradient : ℝ) (final_gradient : ℝ) :
  elevation_gain = 600 →
  initial_gradient = 0.03 →
  final_gradient = 0.02 →
  (elevation_gain / final_gradient - elevation_gain / initial_gradient) / 1000 = 10 := by
  sorry

#check track_extension_calculation

end NUMINAMATH_CALUDE_track_extension_calculation_l3943_394346


namespace NUMINAMATH_CALUDE_bromine_extraction_l3943_394378

-- Define the solubility of a substance in a solvent
def solubility (substance solvent : Type) : ℝ := sorry

-- Define the property of being immiscible
def immiscible (solvent1 solvent2 : Type) : Prop := sorry

-- Define the extraction process
def can_extract (substance from_solvent to_solvent : Type) : Prop := sorry

-- Define the substances and solvents
def bromine : Type := sorry
def water : Type := sorry
def benzene : Type := sorry
def soybean_oil : Type := sorry

-- Theorem statement
theorem bromine_extraction :
  (solubility bromine benzene > solubility bromine water) →
  (solubility bromine soybean_oil > solubility bromine water) →
  immiscible benzene water →
  immiscible soybean_oil water →
  (can_extract bromine water benzene ∨ can_extract bromine water soybean_oil) :=
by sorry

end NUMINAMATH_CALUDE_bromine_extraction_l3943_394378


namespace NUMINAMATH_CALUDE_root_product_theorem_l3943_394383

theorem root_product_theorem (n r : ℝ) (c d : ℝ) : 
  c^2 - n*c + 3 = 0 → 
  d^2 - n*d + 3 = 0 → 
  (c + 2/d)^2 - r*(c + 2/d) + s = 0 → 
  (d + 2/c)^2 - r*(d + 2/c) + s = 0 → 
  s = 25/3 := by sorry

end NUMINAMATH_CALUDE_root_product_theorem_l3943_394383


namespace NUMINAMATH_CALUDE_yogurt_combinations_count_l3943_394377

/-- The number of combinations of one item from a set of 4 and two different items from a set of 6 -/
def yogurt_combinations (flavors : Nat) (toppings : Nat) : Nat :=
  flavors * (toppings.choose 2)

/-- Theorem stating that the number of combinations is 60 -/
theorem yogurt_combinations_count :
  yogurt_combinations 4 6 = 60 := by
  sorry

end NUMINAMATH_CALUDE_yogurt_combinations_count_l3943_394377


namespace NUMINAMATH_CALUDE_range_of_a_l3943_394392

theorem range_of_a (p : ∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0) 
                   (q : ∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0) :
  a ≤ -2 ∨ a = 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3943_394392


namespace NUMINAMATH_CALUDE_dividend_calculation_l3943_394394

theorem dividend_calculation (remainder quotient divisor : ℕ) 
  (h_remainder : remainder = 8)
  (h_quotient : quotient = 43)
  (h_divisor : divisor = 23) :
  divisor * quotient + remainder = 997 := by
  sorry

end NUMINAMATH_CALUDE_dividend_calculation_l3943_394394


namespace NUMINAMATH_CALUDE_sum_of_vertices_l3943_394380

/-- Properties of a single cuboid -/
def cuboid_properties : Nat × Nat × Nat := (12, 6, 8)

/-- The sum of edges and faces of all cuboids -/
def total_edges_and_faces : Nat := 216

/-- Theorem: Given the sum of edges and faces of all cuboids is 216, 
    the sum of vertices of all cuboids is 96 -/
theorem sum_of_vertices (n : Nat) : 
  n * (cuboid_properties.1 + cuboid_properties.2.1) = total_edges_and_faces → 
  n * cuboid_properties.2.2 = 96 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_vertices_l3943_394380


namespace NUMINAMATH_CALUDE_probability_x_plus_y_less_than_4_l3943_394324

/-- A square in the 2D plane -/
structure Square where
  bottomLeft : ℝ × ℝ
  sideLength : ℝ

/-- The probability that a randomly chosen point in the square satisfies a condition -/
def probabilityInSquare (s : Square) (condition : ℝ × ℝ → Prop) : ℝ :=
  sorry

theorem probability_x_plus_y_less_than_4 :
  let s : Square := { bottomLeft := (0, 0), sideLength := 3 }
  probabilityInSquare s (fun (x, y) ↦ x + y < 4) = 7 / 9 := by
  sorry

end NUMINAMATH_CALUDE_probability_x_plus_y_less_than_4_l3943_394324


namespace NUMINAMATH_CALUDE_largest_three_digit_product_l3943_394335

def is_composite (n : ℕ) : Prop := ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

theorem largest_three_digit_product (n x y : ℕ) : 
  100 ≤ n ∧ n < 1000 ∧                  -- n is a three-digit number
  n = x * y * (5 * x + 2 * y) ∧         -- n is the product of x, y, and (5x+2y)
  x < 10 ∧ y < 10 ∧                     -- x and y are less than 10
  is_composite (5 * x + 2 * y) →        -- (5x+2y) is composite
  n ≤ 336 :=                            -- The largest possible value of n is 336
by sorry

end NUMINAMATH_CALUDE_largest_three_digit_product_l3943_394335


namespace NUMINAMATH_CALUDE_arithmetic_mean_difference_l3943_394323

/-- Given that the arithmetic mean of p and q is 10 and the arithmetic mean of q and r is 25,
    prove that r - p = 30 -/
theorem arithmetic_mean_difference (p q r : ℝ) 
  (h1 : (p + q) / 2 = 10) 
  (h2 : (q + r) / 2 = 25) : 
  r - p = 30 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_difference_l3943_394323


namespace NUMINAMATH_CALUDE_expression_simplification_l3943_394322

theorem expression_simplification (m : ℝ) (h1 : m^2 - 4 = 0) (h2 : m ≠ 2) :
  (m^2 + 6*m + 9) / (m - 2) / (m + 2 + (3*m + 4) / (m - 2)) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3943_394322


namespace NUMINAMATH_CALUDE_transport_tax_calculation_l3943_394339

/-- Calculate the transport tax for a vehicle -/
def calculate_transport_tax (horsepower : ℕ) (tax_rate : ℕ) (months_owned : ℕ) : ℕ :=
  (horsepower * tax_rate * months_owned) / 12

theorem transport_tax_calculation (horsepower tax_rate months_owned : ℕ) 
  (h1 : horsepower = 150)
  (h2 : tax_rate = 20)
  (h3 : months_owned = 8) :
  calculate_transport_tax horsepower tax_rate months_owned = 2000 := by
  sorry

#eval calculate_transport_tax 150 20 8

end NUMINAMATH_CALUDE_transport_tax_calculation_l3943_394339


namespace NUMINAMATH_CALUDE_money_sharing_l3943_394332

theorem money_sharing (total : ℝ) (debby_share : ℝ) (maggie_share : ℝ) : 
  debby_share = 0.25 * total →
  maggie_share = total - debby_share →
  maggie_share = 4500 →
  total = 6000 := by
sorry

end NUMINAMATH_CALUDE_money_sharing_l3943_394332


namespace NUMINAMATH_CALUDE_rectangle_width_l3943_394343

/-- Proves that the width of a rectangle is 5 cm, given the specified conditions -/
theorem rectangle_width (length width : ℝ) : 
  (2 * length + 2 * width = 16) →  -- Perimeter is 16 cm
  (width = length + 2) →           -- Width is 2 cm longer than length
  width = 5 := by
sorry


end NUMINAMATH_CALUDE_rectangle_width_l3943_394343


namespace NUMINAMATH_CALUDE_sock_selection_l3943_394345

theorem sock_selection (n k : ℕ) (h1 : n = 6) (h2 : k = 4) : 
  Nat.choose n k = 15 := by
  sorry

end NUMINAMATH_CALUDE_sock_selection_l3943_394345


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l3943_394337

theorem inequality_and_equality_condition (a b c d : ℝ) 
  (pos_a : a > 0) (pos_b : b > 0) (pos_c : c > 0) (pos_d : d > 0)
  (h : a * b * c * d = 1) : 
  a^2 + b^2 + c^2 + d^2 + a*b + a*c + a*d + b*c + b*d + c*d ≥ 10 ∧ 
  (a^2 + b^2 + c^2 + d^2 + a*b + a*c + a*d + b*c + b*d + c*d = 10 ↔ a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l3943_394337


namespace NUMINAMATH_CALUDE_triangle_arctan_sum_l3943_394369

theorem triangle_arctan_sum (a b c : ℝ) (h : c = a + b) :
  Real.arctan (a / (b + c)) + Real.arctan (b / (a + c)) = Real.arctan (1 / 2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_arctan_sum_l3943_394369


namespace NUMINAMATH_CALUDE_tangent_lines_to_circle_l3943_394334

/-- A line in 2D space represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A circle in 2D space represented by the equation (x - h)² + (y - k)² = r² -/
structure Circle where
  h : ℝ
  k : ℝ
  r : ℝ

/-- Check if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- Check if a line is tangent to a circle -/
def tangent (l : Line) (c : Circle) : Prop :=
  (c.h * l.a + c.k * l.b + l.c)^2 = (l.a^2 + l.b^2) * c.r^2

theorem tangent_lines_to_circle (given_line : Line) (c : Circle) :
  given_line = Line.mk 2 (-1) 1 ∧ c = Circle.mk 0 0 (Real.sqrt 5) →
  ∃ (l1 l2 : Line),
    l1 = Line.mk 2 (-1) 5 ∧
    l2 = Line.mk 2 (-1) (-5) ∧
    parallel l1 given_line ∧
    parallel l2 given_line ∧
    tangent l1 c ∧
    tangent l2 c ∧
    ∀ (l : Line), parallel l given_line ∧ tangent l c → l = l1 ∨ l = l2 :=
by sorry

end NUMINAMATH_CALUDE_tangent_lines_to_circle_l3943_394334


namespace NUMINAMATH_CALUDE_circle_proof_l3943_394360

/-- The equation of the given circle -/
def given_circle (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*x - 6*y + 5 = 0

/-- The equation of the circle we want to prove -/
def our_circle (x y : ℝ) : Prop :=
  (x - 3)^2 + (y - 1)^2 = 5

/-- Point A -/
def point_A : ℝ × ℝ := (4, -1)

/-- Point B -/
def point_B : ℝ × ℝ := (1, 2)

/-- Two circles are tangent if they intersect at exactly one point -/
def tangent (c1 c2 : ℝ → ℝ → Prop) (p : ℝ × ℝ) : Prop :=
  c1 p.1 p.2 ∧ c2 p.1 p.2 ∧ ∀ x y, c1 x y ∧ c2 x y → (x, y) = p

theorem circle_proof :
  our_circle point_A.1 point_A.2 ∧
  tangent given_circle our_circle point_B :=
sorry

end NUMINAMATH_CALUDE_circle_proof_l3943_394360


namespace NUMINAMATH_CALUDE_c_investment_value_l3943_394393

/-- Represents the investment and profit distribution in a partnership business. -/
structure Partnership where
  a_investment : ℕ
  b_investment : ℕ
  c_investment : ℕ
  total_profit : ℕ
  c_profit : ℕ

/-- Theorem stating that given the conditions of the partnership,
    c's investment is 50,000. -/
theorem c_investment_value (p : Partnership)
  (h1 : p.a_investment = 30000)
  (h2 : p.b_investment = 45000)
  (h3 : p.total_profit = 90000)
  (h4 : p.c_profit = 36000)
  (h5 : p.c_investment * p.total_profit = p.c_profit * (p.a_investment + p.b_investment + p.c_investment)) :
  p.c_investment = 50000 := by
  sorry


end NUMINAMATH_CALUDE_c_investment_value_l3943_394393


namespace NUMINAMATH_CALUDE_common_number_in_list_l3943_394381

theorem common_number_in_list (list : List ℝ) : 
  list.length = 7 →
  (list.take 4).sum / 4 = 7 →
  (list.drop 3).sum / 4 = 10 →
  list.sum / 7 = 8 →
  ∃ x ∈ list.take 4 ∩ list.drop 3, x = 12 :=
by sorry

end NUMINAMATH_CALUDE_common_number_in_list_l3943_394381


namespace NUMINAMATH_CALUDE_chicken_increase_l3943_394326

/-- The increase in chickens is the sum of chickens bought on two days -/
theorem chicken_increase (initial : ℕ) (day1 : ℕ) (day2 : ℕ) :
  day1 + day2 = (initial + day1 + day2) - initial :=
by sorry

end NUMINAMATH_CALUDE_chicken_increase_l3943_394326


namespace NUMINAMATH_CALUDE_trigonometric_identity_l3943_394396

theorem trigonometric_identity : 
  let sin30 : ℝ := 1/2
  let cos45 : ℝ := Real.sqrt 2 / 2
  let cos60 : ℝ := 1/2
  2 * sin30 - cos45^2 + cos60 = 1 := by sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l3943_394396


namespace NUMINAMATH_CALUDE_craft_item_pricing_problem_l3943_394314

/-- Represents the daily profit function for a craft item store -/
def daily_profit (initial_sales : ℕ) (initial_profit : ℝ) (price_reduction : ℝ) : ℝ :=
  (initial_profit - price_reduction) * (initial_sales + 2 * price_reduction)

theorem craft_item_pricing_problem 
  (initial_sales : ℕ) 
  (initial_profit : ℝ) 
  (price_reduction_1050 : ℝ) 
  (h1 : initial_sales = 20)
  (h2 : initial_profit = 40)
  (h3 : price_reduction_1050 < 40)
  (h4 : daily_profit initial_sales initial_profit price_reduction_1050 = 1050) :
  price_reduction_1050 = 25 ∧ 
  ∀ (price_reduction : ℝ), price_reduction < 40 → 
    daily_profit initial_sales initial_profit price_reduction ≠ 1600 := by
  sorry


end NUMINAMATH_CALUDE_craft_item_pricing_problem_l3943_394314


namespace NUMINAMATH_CALUDE_max_candy_leftover_l3943_394397

theorem max_candy_leftover (x : ℕ+) : ∃ (q r : ℕ), x = 7 * q + r ∧ r ≤ 6 ∧ ∀ (r' : ℕ), x = 7 * q + r' → r' ≤ r :=
sorry

end NUMINAMATH_CALUDE_max_candy_leftover_l3943_394397


namespace NUMINAMATH_CALUDE_mod_37_5_l3943_394395

theorem mod_37_5 : 37 % 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_mod_37_5_l3943_394395


namespace NUMINAMATH_CALUDE_sqrt_sum_inequality_l3943_394306

theorem sqrt_sum_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  Real.sqrt (a / (a + 3 * b)) + Real.sqrt (b / (b + 3 * a)) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_inequality_l3943_394306


namespace NUMINAMATH_CALUDE_completing_square_transformation_l3943_394341

theorem completing_square_transformation (x : ℝ) :
  (x^2 - 2*x - 5 = 0) ↔ ((x - 1)^2 = 6) := by
  sorry

end NUMINAMATH_CALUDE_completing_square_transformation_l3943_394341


namespace NUMINAMATH_CALUDE_not_consecutive_odd_beautiful_l3943_394301

def IsBeautiful (g : ℤ → ℤ) (a : ℤ) : Prop :=
  ∀ x : ℤ, g x = g (a - x)

theorem not_consecutive_odd_beautiful
  (g : ℤ → ℤ)
  (h1 : ∀ x : ℤ, g x ≠ x)
  : ¬∃ a : ℤ, IsBeautiful g a ∧ IsBeautiful g (a + 2) ∧ Odd a :=
by sorry

end NUMINAMATH_CALUDE_not_consecutive_odd_beautiful_l3943_394301


namespace NUMINAMATH_CALUDE_intersection_M_N_l3943_394336

def M : Set ℝ := {x | x^2 > 4}
def N : Set ℝ := {x | x^2 - 3*x ≤ 0}

theorem intersection_M_N : N ∩ M = {x | 2 < x ∧ x ≤ 3} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l3943_394336


namespace NUMINAMATH_CALUDE_complex_expression_simplification_l3943_394385

theorem complex_expression_simplification :
  3 * (4 - 2 * Complex.I) - 2 * (2 * Complex.I - 3) = 18 - 10 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_expression_simplification_l3943_394385


namespace NUMINAMATH_CALUDE_shopping_money_l3943_394308

theorem shopping_money (initial_amount remaining_amount : ℝ) : 
  remaining_amount = initial_amount * (1 - 0.3) ∧ remaining_amount = 840 →
  initial_amount = 1200 := by
sorry

end NUMINAMATH_CALUDE_shopping_money_l3943_394308


namespace NUMINAMATH_CALUDE_value_of_x_l3943_394328

theorem value_of_x (x y z : ℚ) 
  (h1 : x = y / 2) 
  (h2 : y = z / 3) 
  (h3 : z = 100) : 
  x = 50 / 3 := by
  sorry

end NUMINAMATH_CALUDE_value_of_x_l3943_394328


namespace NUMINAMATH_CALUDE_sum_of_digits_253_l3943_394347

/-- Given a three-digit number with specific properties, prove that the sum of its digits is 10 -/
theorem sum_of_digits_253 (a b c : ℕ) : 
  -- The number is 253
  100 * a + 10 * b + c = 253 →
  -- The middle digit is the sum of the other two
  b = a + c →
  -- Reversing the digits increases the number by 99
  100 * c + 10 * b + a = 100 * a + 10 * b + c + 99 →
  -- The sum of the digits is 10
  a + b + c = 10 := by
sorry


end NUMINAMATH_CALUDE_sum_of_digits_253_l3943_394347


namespace NUMINAMATH_CALUDE_cube_sum_theorem_l3943_394373

theorem cube_sum_theorem (p q r : ℝ) 
  (h1 : p + q + r = 4)
  (h2 : p * q + q * r + r * p = 6)
  (h3 : p * q * r = -8) :
  p^3 + q^3 + r^3 = 64 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_theorem_l3943_394373


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l3943_394305

theorem cubic_equation_solution :
  ∀ x y z : ℤ, x^3 + y^3 + z^3 - 3*x*y*z = 2003 ↔
    ((x = 668 ∧ y = 668 ∧ z = 667) ∨
     (x = 668 ∧ y = 667 ∧ z = 668) ∨
     (x = 667 ∧ y = 668 ∧ z = 668)) :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l3943_394305


namespace NUMINAMATH_CALUDE_min_value_theorem_l3943_394312

theorem min_value_theorem (x y z : ℝ) (h : x^3 + y^3 + z^3 - 3*x*y*z = 8) :
  ∃ (m : ℝ), m = 40/3 ∧ ∀ (a b c : ℝ), a^3 + b^3 + c^3 - 3*a*b*c = 8 → 
    a^2 + b^2 + c^2 + 2*a*c ≥ m ∧ 
    (∃ (p q r : ℝ), p^3 + q^3 + r^3 - 3*p*q*r = 8 ∧ p^2 + q^2 + r^2 + 2*p*r = m) :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3943_394312


namespace NUMINAMATH_CALUDE_zoe_recycled_pounds_l3943_394359

/-- The number of pounds that earn one point -/
def pounds_per_point : ℕ := 8

/-- The number of pounds Zoe's friends recycled -/
def friends_pounds : ℕ := 23

/-- The total number of points earned -/
def total_points : ℕ := 6

/-- The number of pounds Zoe recycled -/
def zoe_pounds : ℕ := 25

theorem zoe_recycled_pounds :
  zoe_pounds + friends_pounds = pounds_per_point * total_points :=
sorry

end NUMINAMATH_CALUDE_zoe_recycled_pounds_l3943_394359


namespace NUMINAMATH_CALUDE_f_odd_and_decreasing_l3943_394387

-- Define the function f(x) = -x|x|
def f (x : ℝ) : ℝ := -x * abs x

-- Theorem stating that f is both odd and decreasing
theorem f_odd_and_decreasing :
  (∀ x : ℝ, f (-x) = -f x) ∧ 
  (∀ x y : ℝ, x < y → f y < f x) :=
sorry

end NUMINAMATH_CALUDE_f_odd_and_decreasing_l3943_394387


namespace NUMINAMATH_CALUDE_smaller_omelette_has_three_eggs_l3943_394350

/-- Represents the number of eggs in a smaller omelette -/
def smaller_omelette_eggs : ℕ := sorry

/-- Represents the number of eggs in a larger omelette -/
def larger_omelette_eggs : ℕ := 4

/-- Represents the number of smaller omelettes ordered in the first hour -/
def first_hour_smaller : ℕ := 5

/-- Represents the number of larger omelettes ordered in the second hour -/
def second_hour_larger : ℕ := 7

/-- Represents the number of smaller omelettes ordered in the third hour -/
def third_hour_smaller : ℕ := 3

/-- Represents the number of larger omelettes ordered in the fourth hour -/
def fourth_hour_larger : ℕ := 8

/-- Represents the total number of eggs used -/
def total_eggs : ℕ := 84

/-- Theorem stating that the number of eggs in a smaller omelette is 3 -/
theorem smaller_omelette_has_three_eggs :
  smaller_omelette_eggs = 3 :=
by
  have h1 : first_hour_smaller * smaller_omelette_eggs +
            second_hour_larger * larger_omelette_eggs +
            third_hour_smaller * smaller_omelette_eggs +
            fourth_hour_larger * larger_omelette_eggs = total_eggs := sorry
  sorry

end NUMINAMATH_CALUDE_smaller_omelette_has_three_eggs_l3943_394350


namespace NUMINAMATH_CALUDE_parking_lot_width_l3943_394304

/-- Calculates the width of a parking lot given its specifications -/
theorem parking_lot_width
  (total_length : ℝ)
  (usable_percentage : ℝ)
  (area_per_car : ℝ)
  (total_cars : ℕ)
  (h1 : total_length = 500)
  (h2 : usable_percentage = 0.8)
  (h3 : area_per_car = 10)
  (h4 : total_cars = 16000) :
  (total_length * usable_percentage * (total_cars : ℝ) * area_per_car) / (total_length * usable_percentage) = 400 := by
  sorry

#check parking_lot_width

end NUMINAMATH_CALUDE_parking_lot_width_l3943_394304


namespace NUMINAMATH_CALUDE_shaded_area_square_circles_l3943_394364

/-- The shaded area between a square and four circles --/
theorem shaded_area_square_circles (s : ℝ) (r : ℝ) (h1 : s = 10) (h2 : r = 3 * Real.sqrt 3) :
  s^2 - 4 * (π * r^2 / 4) - 8 * (s / 2 * Real.sqrt ((3 * Real.sqrt 3)^2 - (s / 2)^2) / 2) = 
    100 - 27 * π - 20 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_shaded_area_square_circles_l3943_394364


namespace NUMINAMATH_CALUDE_old_toilet_water_usage_l3943_394370

/-- The amount of water saved by switching to a new toilet in June -/
def water_saved : ℝ := 1800

/-- The number of times the toilet is flushed per day -/
def flushes_per_day : ℕ := 15

/-- The number of days in June -/
def days_in_june : ℕ := 30

/-- The percentage of water saved by the new toilet compared to the old one -/
def water_saving_percentage : ℝ := 0.8

theorem old_toilet_water_usage : ℝ :=
  let total_flushes : ℕ := flushes_per_day * days_in_june
  let water_saved_per_flush : ℝ := water_saved / total_flushes
  water_saved_per_flush / water_saving_percentage

#check @old_toilet_water_usage

end NUMINAMATH_CALUDE_old_toilet_water_usage_l3943_394370


namespace NUMINAMATH_CALUDE_BaSO4_molecular_weight_l3943_394351

/-- The atomic weight of Barium in g/mol -/
def Ba_weight : ℝ := 137.327

/-- The atomic weight of Sulfur in g/mol -/
def S_weight : ℝ := 32.065

/-- The atomic weight of Oxygen in g/mol -/
def O_weight : ℝ := 15.999

/-- The number of Oxygen atoms in BaSO4 -/
def O_count : ℕ := 4

/-- The molecular weight of BaSO4 in g/mol -/
def BaSO4_weight : ℝ := Ba_weight + S_weight + O_count * O_weight

theorem BaSO4_molecular_weight : BaSO4_weight = 233.388 := by
  sorry

end NUMINAMATH_CALUDE_BaSO4_molecular_weight_l3943_394351


namespace NUMINAMATH_CALUDE_half_x_is_32_implies_2x_is_128_l3943_394300

theorem half_x_is_32_implies_2x_is_128 (x : ℝ) (h : x / 2 = 32) : 2 * x = 128 := by
  sorry

end NUMINAMATH_CALUDE_half_x_is_32_implies_2x_is_128_l3943_394300


namespace NUMINAMATH_CALUDE_f_2023_of_2_eq_one_seventh_l3943_394349

-- Define the function f
def f (x : ℚ) : ℚ := (1 + x) / (1 - 3*x)

-- Define f_n recursively
def f_n : ℕ → (ℚ → ℚ)
  | 0 => f
  | 1 => λ x => f (f x)
  | (n+2) => λ x => f (f_n (n+1) x)

-- Theorem statement
theorem f_2023_of_2_eq_one_seventh : f_n 2023 2 = 1/7 := by sorry

end NUMINAMATH_CALUDE_f_2023_of_2_eq_one_seventh_l3943_394349


namespace NUMINAMATH_CALUDE_sequence_shorter_than_25_l3943_394333

/-- The sequence of digits formed by writing consecutive integers from 10 to 1 -/
def descendingSequence : List Nat := [1, 0, 9, 8, 7, 6, 5, 4, 3, 2, 1]

/-- The length of the sequence -/
def sequenceLength : Nat := descendingSequence.length

theorem sequence_shorter_than_25 : sequenceLength < 25 := by
  sorry

end NUMINAMATH_CALUDE_sequence_shorter_than_25_l3943_394333


namespace NUMINAMATH_CALUDE_chocolate_bar_difference_l3943_394384

theorem chocolate_bar_difference :
  let first_friend_portion : ℚ := 5 / 6
  let second_friend_portion : ℚ := 2 / 3
  first_friend_portion - second_friend_portion = 1 / 6 := by
sorry

end NUMINAMATH_CALUDE_chocolate_bar_difference_l3943_394384


namespace NUMINAMATH_CALUDE_candy_distribution_l3943_394372

theorem candy_distribution (n : ℕ) : 
  n > 0 → 
  (∃ k : ℕ, n * k + 1 = 120) → 
  n = 7 ∨ n = 17 :=
sorry

end NUMINAMATH_CALUDE_candy_distribution_l3943_394372


namespace NUMINAMATH_CALUDE_election_winner_percentage_l3943_394390

theorem election_winner_percentage (total_votes : ℕ) (majority : ℕ) : 
  total_votes = 470 → majority = 188 → 
  (70 : ℚ) * total_votes / 100 - ((100 : ℚ) - 70) * total_votes / 100 = majority := by
  sorry

end NUMINAMATH_CALUDE_election_winner_percentage_l3943_394390


namespace NUMINAMATH_CALUDE_exact_blue_marbles_probability_l3943_394361

def total_marbles : ℕ := 12
def blue_marbles : ℕ := 8
def red_marbles : ℕ := 4
def num_draws : ℕ := 7
def target_blue : ℕ := 4

theorem exact_blue_marbles_probability :
  (Nat.choose num_draws target_blue : ℚ) * (blue_marbles ^ target_blue * red_marbles ^ (num_draws - target_blue)) / (total_marbles ^ num_draws) = 35 * (16 : ℚ) / 2187 := by
  sorry

end NUMINAMATH_CALUDE_exact_blue_marbles_probability_l3943_394361


namespace NUMINAMATH_CALUDE_external_tangent_circle_l3943_394309

/-- Given circle C with equation (x-2)^2 + (y+1)^2 = 4 and point A(4, -1) on C,
    prove that the circle with equation (x-5)^2 + (y+1)^2 = 1 is externally
    tangent to C at A and has radius 1. -/
theorem external_tangent_circle
  (C : Set (ℝ × ℝ))
  (A : ℝ × ℝ)
  (hC : C = {p : ℝ × ℝ | (p.1 - 2)^2 + (p.2 + 1)^2 = 4})
  (hA : A = (4, -1))
  (hA_on_C : A ∈ C)
  : ∃ (M : Set (ℝ × ℝ)),
    M = {p : ℝ × ℝ | (p.1 - 5)^2 + (p.2 + 1)^2 = 1} ∧
    (∀ p ∈ M, ∃ q ∈ C, (p.1 - q.1)^2 + (p.2 - q.2)^2 = 1) ∧
    A ∈ M ∧
    (∀ p ∈ M, (p.1 - 5)^2 + (p.2 + 1)^2 = 1) :=
sorry

end NUMINAMATH_CALUDE_external_tangent_circle_l3943_394309


namespace NUMINAMATH_CALUDE_geometric_sequence_third_term_l3943_394331

theorem geometric_sequence_third_term
  (a : ℕ → ℝ)
  (h_geometric : ∀ n, a (n + 1) / a n = a (n + 2) / a (n + 1))
  (h_a1 : a 1 = 2)
  (h_a5 : a 5 = 8) :
  a 3 = 4 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_third_term_l3943_394331


namespace NUMINAMATH_CALUDE_intersection_with_complement_l3943_394330

-- Define the sets A and B
def A : Set ℝ := {x | -1 ≤ x ∧ x < 3}
def B : Set ℝ := {x | 2 ≤ x ∧ x < 3}

-- State the theorem
theorem intersection_with_complement :
  A ∩ (Set.univ \ B) = {x : ℝ | -1 ≤ x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_intersection_with_complement_l3943_394330


namespace NUMINAMATH_CALUDE_speed_conversion_correct_l3943_394365

/-- Converts meters per second to kilometers per hour -/
def mps_to_kmh (speed_mps : ℚ) : ℚ :=
  speed_mps * 3.6

theorem speed_conversion_correct : 
  mps_to_kmh (13/48) = 39/40 :=
by sorry

end NUMINAMATH_CALUDE_speed_conversion_correct_l3943_394365


namespace NUMINAMATH_CALUDE_hexagonal_table_dice_probability_l3943_394315

/-- The number of people seated around the hexagonal table -/
def num_people : ℕ := 6

/-- The number of sides on the standard die -/
def die_sides : ℕ := 6

/-- A function to calculate the number of valid options for each person's roll -/
def valid_options (person : ℕ) : ℕ :=
  match person with
  | 1 => 6  -- Person A
  | 2 => 5  -- Person B
  | 3 => 4  -- Person C
  | 4 => 5  -- Person D
  | 5 => 3  -- Person E
  | 6 => 3  -- Person F
  | _ => 0  -- Invalid person number

/-- The probability of no two adjacent or opposite people rolling the same number -/
def probability : ℚ :=
  (valid_options 1 * valid_options 2 * valid_options 3 * valid_options 4 * valid_options 5 * valid_options 6) /
  (die_sides ^ num_people)

theorem hexagonal_table_dice_probability :
  probability = 25 / 648 := by
  sorry

end NUMINAMATH_CALUDE_hexagonal_table_dice_probability_l3943_394315


namespace NUMINAMATH_CALUDE_inverse_variation_cube_square_l3943_394340

/-- Given that a³ varies inversely with b², prove that a³ = 125/16 when b = 8,
    given that a = 5 when b = 2. -/
theorem inverse_variation_cube_square (a b : ℝ) (k : ℝ) : 
  (∀ x y : ℝ, x^3 * y^2 = k) →  -- a³ varies inversely with b²
  (5^3 * 2^2 = k) →             -- a = 5 when b = 2
  (a^3 * 8^2 = k) →             -- condition for b = 8
  a^3 = 125/16 := by
sorry

end NUMINAMATH_CALUDE_inverse_variation_cube_square_l3943_394340


namespace NUMINAMATH_CALUDE_floor_ceiling_sum_l3943_394371

theorem floor_ceiling_sum : ⌊(-3.67 : ℝ)⌋ + ⌈(34.2 : ℝ)⌉ = 31 := by sorry

end NUMINAMATH_CALUDE_floor_ceiling_sum_l3943_394371


namespace NUMINAMATH_CALUDE_freshmen_psych_liberal_arts_percentage_l3943_394317

/-- Represents the percentage of students that are freshmen -/
def freshman_percentage : ℝ := 80

/-- Represents the percentage of freshmen enrolled in liberal arts -/
def liberal_arts_percentage : ℝ := 60

/-- Represents the percentage of liberal arts freshmen who are psychology majors -/
def psychology_percentage : ℝ := 50

/-- Theorem stating the percentage of students who are freshmen psychology majors in liberal arts -/
theorem freshmen_psych_liberal_arts_percentage :
  (freshman_percentage / 100) * (liberal_arts_percentage / 100) * (psychology_percentage / 100) * 100 = 24 := by
  sorry


end NUMINAMATH_CALUDE_freshmen_psych_liberal_arts_percentage_l3943_394317


namespace NUMINAMATH_CALUDE_journey_time_calculation_l3943_394329

theorem journey_time_calculation (total_distance : ℝ) (initial_fraction : ℝ) (initial_time : ℝ) (lunch_time : ℝ) :
  total_distance = 200 →
  initial_fraction = 1/4 →
  initial_time = 1 →
  lunch_time = 1 →
  ∃ (total_time : ℝ), total_time = 5 := by
  sorry

end NUMINAMATH_CALUDE_journey_time_calculation_l3943_394329


namespace NUMINAMATH_CALUDE_complex_magnitude_one_l3943_394319

theorem complex_magnitude_one (z : ℂ) (h : 11 * z^10 + 10*Complex.I * z^9 + 10*Complex.I * z - 11 = 0) : Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_one_l3943_394319


namespace NUMINAMATH_CALUDE_area_2018_correct_l3943_394348

/-- Calculates the area to be converted after a given number of years -/
def area_to_convert (initial_area : ℝ) (annual_increase : ℝ) (years : ℕ) : ℝ :=
  initial_area * (1 + annual_increase) ^ years

/-- Proves that the area to be converted in 2018 is correct -/
theorem area_2018_correct (initial_area : ℝ) (annual_increase : ℝ) :
  initial_area = 8 →
  annual_increase = 0.1 →
  area_to_convert initial_area annual_increase 5 = 8 * 1.1^5 := by
  sorry

#check area_2018_correct

end NUMINAMATH_CALUDE_area_2018_correct_l3943_394348


namespace NUMINAMATH_CALUDE_press_conference_seating_l3943_394352

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def seating_arrangements (team_sizes : List ℕ) : ℕ :=
  (factorial team_sizes.length) * (team_sizes.map factorial).prod

theorem press_conference_seating :
  seating_arrangements [3, 3, 2, 2] = 3456 := by
  sorry

end NUMINAMATH_CALUDE_press_conference_seating_l3943_394352


namespace NUMINAMATH_CALUDE_darias_current_money_l3943_394368

/-- Calculates Daria's current money for concert tickets -/
theorem darias_current_money
  (ticket_cost : ℕ)  -- Cost of one ticket
  (num_tickets : ℕ)  -- Number of tickets Daria needs to buy
  (money_needed : ℕ) -- Additional money Daria needs to earn
  (h1 : ticket_cost = 90)
  (h2 : num_tickets = 4)
  (h3 : money_needed = 171) :
  ticket_cost * num_tickets - money_needed = 189 :=
by sorry

end NUMINAMATH_CALUDE_darias_current_money_l3943_394368


namespace NUMINAMATH_CALUDE_parabola_triangle_area_l3943_394388

/-- Given a parabola y² = 4x with focus F(1,0), and points A and B on the parabola
    such that FA = 2BF, the area of triangle OAB is 3√2/2. -/
theorem parabola_triangle_area (A B : ℝ × ℝ) :
  let C : ℝ × ℝ → Prop := λ p => p.2^2 = 4 * p.1
  let F : ℝ × ℝ := (1, 0)
  let O : ℝ × ℝ := (0, 0)
  C A ∧ C B ∧ 
  (∃ (t : ℝ), A = F + t • (A - F) ∧ B = F + t • (B - F)) ∧
  (A - F) = 2 • (F - B) →
  abs ((A.1 * B.2 - A.2 * B.1) / 2) = 3 * Real.sqrt 2 / 2 :=
by sorry


end NUMINAMATH_CALUDE_parabola_triangle_area_l3943_394388


namespace NUMINAMATH_CALUDE_cube_sum_reciprocal_l3943_394303

theorem cube_sum_reciprocal (x : ℝ) (h : x + 1/x = 4) : x^3 + 1/x^3 = 52 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_reciprocal_l3943_394303


namespace NUMINAMATH_CALUDE_book_club_monthly_books_l3943_394356

def prove_book_club_monthly_books (initial_books final_books bookstore_purchase yard_sale_purchase 
  daughter_gift mother_gift donated_books sold_books : ℕ) : Prop :=
  let total_acquired := bookstore_purchase + yard_sale_purchase + daughter_gift + mother_gift
  let total_removed := donated_books + sold_books
  let net_change := final_books - initial_books
  let book_club_total := net_change + total_removed - total_acquired
  (book_club_total % 12 = 0) ∧ (book_club_total / 12 = 1)

theorem book_club_monthly_books :
  prove_book_club_monthly_books 72 81 5 2 1 4 12 3 :=
sorry

end NUMINAMATH_CALUDE_book_club_monthly_books_l3943_394356


namespace NUMINAMATH_CALUDE_tangent_circle_radius_l3943_394358

/-- A circle tangent to the x-axis, y-axis, and hypotenuse of a 45°-45°-90° triangle --/
structure TangentCircle where
  /-- The radius of the circle --/
  radius : ℝ
  /-- The circle is tangent to the x-axis --/
  tangent_x : True
  /-- The circle is tangent to the y-axis --/
  tangent_y : True
  /-- The circle is tangent to the hypotenuse of a 45°-45°-90° triangle --/
  tangent_hypotenuse : True
  /-- The length of a leg of the 45°-45°-90° triangle is 2 --/
  triangle_leg : ℝ := 2

/-- The radius of the TangentCircle is equal to 2 + √2 --/
theorem tangent_circle_radius (c : TangentCircle) : c.radius = 2 + Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_circle_radius_l3943_394358


namespace NUMINAMATH_CALUDE_set_B_is_empty_l3943_394338

def set_B : Set ℝ := {x | x > 8 ∧ x < 5}

theorem set_B_is_empty : set_B = ∅ := by sorry

end NUMINAMATH_CALUDE_set_B_is_empty_l3943_394338


namespace NUMINAMATH_CALUDE_find_y_l3943_394374

theorem find_y : ∃ y : ℕ, (12^3 * 6^4) / y = 5184 ∧ y = 432 := by
  sorry

end NUMINAMATH_CALUDE_find_y_l3943_394374


namespace NUMINAMATH_CALUDE_vector_equality_l3943_394391

/-- Given vectors a, b, and c in R², prove that c = a - 3b -/
theorem vector_equality (a b c : Fin 2 → ℝ) 
  (ha : a = ![1, 1]) 
  (hb : b = ![1, -1]) 
  (hc : c = ![-2, 4]) : 
  c = a - 3 • b := by sorry

end NUMINAMATH_CALUDE_vector_equality_l3943_394391


namespace NUMINAMATH_CALUDE_smallest_variable_l3943_394366

theorem smallest_variable (p q r s : ℝ) 
  (h : p + 3 = q - 1 ∧ p + 3 = r + 5 ∧ p + 3 = s - 2) : 
  r ≤ p ∧ r ≤ q ∧ r ≤ s := by
  sorry

end NUMINAMATH_CALUDE_smallest_variable_l3943_394366


namespace NUMINAMATH_CALUDE_contractor_absent_days_l3943_394325

/-- Represents the contract details and calculates the number of absent days -/
def calculate_absent_days (total_days : ℕ) (pay_per_day : ℚ) (fine_per_day : ℚ) (total_pay : ℚ) : ℚ :=
  let worked_days := total_days - (total_pay - total_days * pay_per_day) / (pay_per_day + fine_per_day)
  total_days - worked_days

/-- Theorem stating that given the specific contract conditions, the number of absent days is 12 -/
theorem contractor_absent_days :
  let total_days : ℕ := 30
  let pay_per_day : ℚ := 25
  let fine_per_day : ℚ := 7.5
  let total_pay : ℚ := 360
  calculate_absent_days total_days pay_per_day fine_per_day total_pay = 12 := by
  sorry

#eval calculate_absent_days 30 25 7.5 360

end NUMINAMATH_CALUDE_contractor_absent_days_l3943_394325


namespace NUMINAMATH_CALUDE_burger_cost_is_100_cents_l3943_394399

/-- The cost of items in cents -/
structure ItemCosts where
  burger : ℕ
  soda : ℕ
  fries : ℕ

/-- Alice's purchase -/
def alice_purchase (costs : ItemCosts) : ℕ :=
  4 * costs.burger + 3 * costs.soda + costs.fries

/-- Bob's purchase -/
def bob_purchase (costs : ItemCosts) : ℕ :=
  3 * costs.burger + 2 * costs.soda + 2 * costs.fries

/-- Theorem stating that the cost of a burger is 100 cents -/
theorem burger_cost_is_100_cents :
  ∃ (costs : ItemCosts),
    alice_purchase costs = 540 ∧
    bob_purchase costs = 580 ∧
    costs.burger = 100 := by
  sorry

end NUMINAMATH_CALUDE_burger_cost_is_100_cents_l3943_394399


namespace NUMINAMATH_CALUDE_equation_roots_l3943_394363

theorem equation_roots (a b : ℝ) :
  -- Part 1
  (∃ x : ℂ, x = 1 - Complex.I * Real.sqrt 3 ∧ x / a + b / x = 1) →
  a = 2 ∧ b = 2
  ∧
  -- Part 2
  (b / a > 1 / 4 ∧ a > 0) →
  ¬∃ x : ℝ, x / a + b / x = 1 :=
by sorry

end NUMINAMATH_CALUDE_equation_roots_l3943_394363


namespace NUMINAMATH_CALUDE_snow_volume_calculation_l3943_394367

/-- The volume of snow to be shoveled from a walkway -/
def snow_volume (total_length width depth no_shovel_length : ℝ) : ℝ :=
  (total_length - no_shovel_length) * width * depth

/-- Proof that the volume of snow to be shoveled is 46.875 cubic feet -/
theorem snow_volume_calculation : 
  snow_volume 30 2.5 0.75 5 = 46.875 := by
  sorry

end NUMINAMATH_CALUDE_snow_volume_calculation_l3943_394367


namespace NUMINAMATH_CALUDE_min_black_edges_four_black_edges_possible_l3943_394386

structure Cube :=
  (edges : Finset (Fin 12))
  (faces : Finset (Fin 6))
  (edge_coloring : Fin 12 → Bool)
  (face_edges : Fin 6 → Finset (Fin 12))
  (edge_faces : Fin 12 → Finset (Fin 2))

def is_valid_coloring (c : Cube) : Prop :=
  ∀ f : Fin 6, 
    (∃ e ∈ c.face_edges f, c.edge_coloring e = true) ∧ 
    (∃ e ∈ c.face_edges f, c.edge_coloring e = false)

def num_black_edges (c : Cube) : Nat :=
  (c.edges.filter (λ e => c.edge_coloring e = true)).card

theorem min_black_edges (c : Cube) :
  is_valid_coloring c → num_black_edges c ≥ 4 :=
sorry

theorem four_black_edges_possible : 
  ∃ c : Cube, is_valid_coloring c ∧ num_black_edges c = 4 :=
sorry

end NUMINAMATH_CALUDE_min_black_edges_four_black_edges_possible_l3943_394386


namespace NUMINAMATH_CALUDE_coffee_stock_decaf_percentage_l3943_394342

/-- Calculates the percentage of decaffeinated coffee in the total stock -/
theorem coffee_stock_decaf_percentage
  (initial_stock : ℝ)
  (initial_decaf_percent : ℝ)
  (additional_stock : ℝ)
  (additional_decaf_percent : ℝ)
  (h1 : initial_stock = 400)
  (h2 : initial_decaf_percent = 20)
  (h3 : additional_stock = 100)
  (h4 : additional_decaf_percent = 70) :
  let total_stock := initial_stock + additional_stock
  let total_decaf := (initial_stock * initial_decaf_percent / 100) +
                     (additional_stock * additional_decaf_percent / 100)
  total_decaf / total_stock * 100 = 30 := by
sorry

end NUMINAMATH_CALUDE_coffee_stock_decaf_percentage_l3943_394342


namespace NUMINAMATH_CALUDE_rectangle_width_l3943_394353

/-- Given a rectangle with perimeter 48 cm and width 2 cm shorter than length, prove width is 11 cm -/
theorem rectangle_width (length width : ℝ) : 
  (2 * length + 2 * width = 48) →  -- Perimeter condition
  (width = length - 2) →           -- Width-length relation
  (width = 11) :=                  -- Conclusion to prove
by
  sorry

end NUMINAMATH_CALUDE_rectangle_width_l3943_394353


namespace NUMINAMATH_CALUDE_solution_set_inequality_l3943_394321

theorem solution_set_inequality (x : ℝ) :
  (x - 5) * (x + 1) > 0 ↔ x ∈ Set.Iio (-1) ∪ Set.Ioi 5 :=
by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l3943_394321


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3943_394311

theorem min_value_reciprocal_sum (m n : ℝ) (hm : m > 0) (hn : n > 0) (h_sum : m + n = 1) :
  (∀ m' n' : ℝ, m' > 0 → n' > 0 → m' + n' = 1 → 1/m' + 2/n' ≥ 1/m + 2/n) →
  1/m + 2/n = 3 + 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3943_394311


namespace NUMINAMATH_CALUDE_basketball_lineups_l3943_394362

/-- The number of players in the basketball team -/
def team_size : ℕ := 12

/-- The number of positions in the starting lineup -/
def lineup_size : ℕ := 5

/-- The number of different starting lineups that can be chosen -/
def num_lineups : ℕ := 95040

/-- Theorem: The number of different starting lineups that can be chosen
    from a team of 12 players for 5 distinct positions is 95,040 -/
theorem basketball_lineups :
  (team_size.factorial) / ((team_size - lineup_size).factorial) = num_lineups := by
  sorry

end NUMINAMATH_CALUDE_basketball_lineups_l3943_394362


namespace NUMINAMATH_CALUDE_compared_same_type_as_reference_l3943_394307

/-- Two expressions are of the same type if they have the same variables with the same exponents -/
def same_type (e1 e2 : ℕ → ℕ → ℚ) : Prop :=
  ∀ a b, ∃ k : ℚ, e1 a b = k * e2 a b

/-- The reference expression a^2 * b -/
def reference (a b : ℕ) : ℚ := (a^2 : ℚ) * b

/-- The expression to be compared: -2/5 * b * a^2 -/
def compared (a b : ℕ) : ℚ := -(2/5 : ℚ) * b * (a^2 : ℚ)

/-- Theorem stating that the compared expression is of the same type as the reference -/
theorem compared_same_type_as_reference : same_type compared reference := by
  sorry

end NUMINAMATH_CALUDE_compared_same_type_as_reference_l3943_394307


namespace NUMINAMATH_CALUDE_tangent_slope_at_one_l3943_394355

-- Define the function
def f (x : ℝ) : ℝ := x^2 + 2

-- State the theorem
theorem tangent_slope_at_one :
  (deriv f) 1 = 2 :=
sorry

end NUMINAMATH_CALUDE_tangent_slope_at_one_l3943_394355


namespace NUMINAMATH_CALUDE_central_cell_removed_theorem_corner_cell_removed_theorem_l3943_394318

-- Define a 7x7 grid
def Grid := Fin 7 → Fin 7 → Bool

-- Define a domino placement
structure Domino where
  x : Fin 7
  y : Fin 7
  horizontal : Bool

-- Define a tiling of the grid
def Tiling := List Domino

-- Function to check if a tiling is valid for a given grid
def is_valid_tiling (g : Grid) (t : Tiling) : Prop := sorry

-- Function to count horizontal dominoes in a tiling
def count_horizontal (t : Tiling) : Nat := sorry

-- Function to count vertical dominoes in a tiling
def count_vertical (t : Tiling) : Nat := sorry

-- Define a grid with the central cell removed
def central_removed_grid : Grid := sorry

-- Define a grid with a corner cell removed
def corner_removed_grid : Grid := sorry

theorem central_cell_removed_theorem :
  ∃ t : Tiling, is_valid_tiling central_removed_grid t ∧
    count_horizontal t = count_vertical t := sorry

theorem corner_cell_removed_theorem :
  ¬∃ t : Tiling, is_valid_tiling corner_removed_grid t ∧
    count_horizontal t = count_vertical t := sorry

end NUMINAMATH_CALUDE_central_cell_removed_theorem_corner_cell_removed_theorem_l3943_394318


namespace NUMINAMATH_CALUDE_division_problem_additional_condition_l3943_394316

theorem division_problem (x : ℝ) : 2994 / x = 175 → x = 17.1 := by
  sorry

-- Additional theorem to include the unused condition
theorem additional_condition : 29.94 / 1.45 = 17.5 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_additional_condition_l3943_394316


namespace NUMINAMATH_CALUDE_dales_peppers_total_l3943_394310

/-- The weight of green peppers in pounds -/
def green_peppers : ℝ := 3.25

/-- The weight of red peppers in pounds -/
def red_peppers : ℝ := 2.5

/-- The weight of yellow peppers in pounds -/
def yellow_peppers : ℝ := 1.75

/-- The weight of orange peppers in pounds -/
def orange_peppers : ℝ := 4.6

/-- The total weight of all peppers bought by Dale's Vegetarian Restaurant -/
def total_peppers : ℝ := green_peppers + red_peppers + yellow_peppers + orange_peppers

theorem dales_peppers_total :
  total_peppers = 12.1 := by sorry

end NUMINAMATH_CALUDE_dales_peppers_total_l3943_394310


namespace NUMINAMATH_CALUDE_squirrel_nuts_theorem_l3943_394357

/-- The number of nuts found by Pizizubka -/
def pizizubka_nuts : ℕ := 48

/-- The number of nuts found by Zrzečka -/
def zrzecka_nuts : ℕ := 96

/-- The number of nuts found by Ouška -/
def ouska_nuts : ℕ := 144

/-- The fraction of nuts Pizizubka ate -/
def pizizubka_ate : ℚ := 1/2

/-- The fraction of nuts Zrzečka ate -/
def zrzecka_ate : ℚ := 1/3

/-- The fraction of nuts Ouška ate -/
def ouska_ate : ℚ := 1/4

/-- The total number of nuts left -/
def total_nuts_left : ℕ := 196

theorem squirrel_nuts_theorem :
  zrzecka_nuts = 2 * pizizubka_nuts ∧
  ouska_nuts = 3 * pizizubka_nuts ∧
  (1 - pizizubka_ate) * pizizubka_nuts +
  (1 - zrzecka_ate) * zrzecka_nuts +
  (1 - ouska_ate) * ouska_nuts = total_nuts_left :=
by sorry

end NUMINAMATH_CALUDE_squirrel_nuts_theorem_l3943_394357


namespace NUMINAMATH_CALUDE_largest_divisor_of_sequence_l3943_394354

theorem largest_divisor_of_sequence :
  ∃ (x : ℕ), x = 18 ∧ 
  (∀ y : ℕ, x ∣ (7^y + 12*y - 1)) ∧
  (∀ z : ℕ, z > x → ∃ w : ℕ, ¬(z ∣ (7^w + 12*w - 1))) := by
  sorry

end NUMINAMATH_CALUDE_largest_divisor_of_sequence_l3943_394354


namespace NUMINAMATH_CALUDE_swap_7_and_9_breaks_equality_l3943_394313

def original_number : ℕ := 271828
def swapped_number : ℕ := 291828
def target_sum : ℕ := 314159

def swap_digits (n : ℕ) (d1 d2 : ℕ) : ℕ := sorry

theorem swap_7_and_9_breaks_equality :
  swap_digits original_number 7 9 = swapped_number ∧
  swapped_number + original_number ≠ 2 * target_sum :=
sorry

end NUMINAMATH_CALUDE_swap_7_and_9_breaks_equality_l3943_394313
