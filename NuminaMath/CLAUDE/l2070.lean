import Mathlib

namespace NUMINAMATH_CALUDE_inequality_reversal_l2070_207089

theorem inequality_reversal (a b : ℝ) (h : a < b) : -2 * a > -2 * b := by
  sorry

end NUMINAMATH_CALUDE_inequality_reversal_l2070_207089


namespace NUMINAMATH_CALUDE_f_properties_l2070_207040

noncomputable def f (x φ : ℝ) : ℝ := Real.sqrt 3 * Real.sin (2 * x + φ) + Real.cos (2 * x + φ)

theorem f_properties (φ : ℝ) :
  (∀ x, f x φ = f (-x) φ) →  -- f is an even function
  (∀ x ∈ Set.Icc 0 (π / 4), ∀ y ∈ Set.Icc 0 (π / 4), x < y → f x φ < f y φ) →  -- f is increasing in [0, π/4]
  φ = 4 * π / 3 := by
sorry

end NUMINAMATH_CALUDE_f_properties_l2070_207040


namespace NUMINAMATH_CALUDE_toothpicks_in_45x25_grid_with_gaps_l2070_207071

/-- Calculates the number of effective lines in a grid with gaps every fifth line -/
def effectiveLines (total : ℕ) : ℕ :=
  total + 1 - (total + 1) / 5

/-- Calculates the total number of toothpicks in a rectangular grid with gaps -/
def toothpicksInGrid (length width : ℕ) : ℕ :=
  let verticalLines := effectiveLines length
  let horizontalLines := effectiveLines width
  verticalLines * width + horizontalLines * length

/-- Theorem: A 45x25 grid with every fifth row and column missing uses 1722 toothpicks -/
theorem toothpicks_in_45x25_grid_with_gaps :
  toothpicksInGrid 45 25 = 1722 := by
  sorry

#eval toothpicksInGrid 45 25

end NUMINAMATH_CALUDE_toothpicks_in_45x25_grid_with_gaps_l2070_207071


namespace NUMINAMATH_CALUDE_fifth_coaster_speed_l2070_207079

def rollercoaster_problem (S₁ S₂ S₃ S₄ S₅ : ℝ) : Prop :=
  S₁ = 50 ∧ S₂ = 62 ∧ S₃ = 73 ∧ S₄ = 70 ∧ (S₁ + S₂ + S₃ + S₄ + S₅) / 5 = 59

theorem fifth_coaster_speed :
  ∀ S₁ S₂ S₃ S₄ S₅ : ℝ,
  rollercoaster_problem S₁ S₂ S₃ S₄ S₅ →
  S₅ = 40 := by
  sorry


end NUMINAMATH_CALUDE_fifth_coaster_speed_l2070_207079


namespace NUMINAMATH_CALUDE_popcorn_profit_30_bags_l2070_207008

/-- Calculates the profit from selling popcorn bags -/
def popcorn_profit (buy_price sell_price : ℕ) (num_bags : ℕ) : ℕ :=
  (sell_price - buy_price) * num_bags

theorem popcorn_profit_30_bags :
  popcorn_profit 4 8 30 = 120 := by
  sorry

end NUMINAMATH_CALUDE_popcorn_profit_30_bags_l2070_207008


namespace NUMINAMATH_CALUDE_function_range_l2070_207093

/-- Given a function f(x) = x + 4a/x - a, where a < 0, 
    if f(x) < 0 for all x in (0, 1], then a ≤ -1/3 -/
theorem function_range (a : ℝ) (h1 : a < 0) :
  (∀ x ∈ Set.Ioo 0 1, x + 4 * a / x - a < 0) →
  a ≤ -1/3 := by
  sorry

end NUMINAMATH_CALUDE_function_range_l2070_207093


namespace NUMINAMATH_CALUDE_cube_side_length_l2070_207097

/-- Proves that given the cost of paint, coverage, and total cost to paint a cube,
    the side length of the cube is 8 feet. -/
theorem cube_side_length 
  (paint_cost : ℝ) 
  (paint_coverage : ℝ) 
  (total_cost : ℝ) 
  (h1 : paint_cost = 36.50)
  (h2 : paint_coverage = 16)
  (h3 : total_cost = 876) :
  ∃ (s : ℝ), s = 8 ∧ 
  total_cost = (6 * s^2 / paint_coverage) * paint_cost :=
sorry

end NUMINAMATH_CALUDE_cube_side_length_l2070_207097


namespace NUMINAMATH_CALUDE_remainder_of_196c_pow_2008_mod_97_l2070_207070

theorem remainder_of_196c_pow_2008_mod_97 (c : ℤ) : (196 * c)^2008 % 97 = 44 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_196c_pow_2008_mod_97_l2070_207070


namespace NUMINAMATH_CALUDE_hcf_from_lcm_and_product_l2070_207006

/-- Given two positive integers with LCM 750 and product 18750, prove their HCF is 25 -/
theorem hcf_from_lcm_and_product (a b : ℕ+) 
  (h_lcm : Nat.lcm a b = 750)
  (h_product : a * b = 18750) : 
  Nat.gcd a b = 25 := by
  sorry

end NUMINAMATH_CALUDE_hcf_from_lcm_and_product_l2070_207006


namespace NUMINAMATH_CALUDE_T_perimeter_is_20_l2070_207077

/-- The perimeter of a T shape formed by two 2-inch × 4-inch rectangles -/
def T_perimeter : ℝ :=
  let rectangle_width : ℝ := 2
  let rectangle_length : ℝ := 4
  let rectangle_perimeter : ℝ := 2 * (rectangle_width + rectangle_length)
  let overlap : ℝ := 2 * rectangle_width
  2 * rectangle_perimeter - overlap

/-- Theorem stating that the perimeter of the T shape is 20 inches -/
theorem T_perimeter_is_20 : T_perimeter = 20 := by
  sorry

end NUMINAMATH_CALUDE_T_perimeter_is_20_l2070_207077


namespace NUMINAMATH_CALUDE_square_roots_of_25_l2070_207061

theorem square_roots_of_25 : Set ℝ := by
  -- Define the set of square roots of 25
  let roots : Set ℝ := {x : ℝ | x^2 = 25}
  
  -- Prove that this set is equal to {-5, 5}
  have h : roots = {-5, 5} := by sorry
  
  -- Return the set of square roots
  exact roots

end NUMINAMATH_CALUDE_square_roots_of_25_l2070_207061


namespace NUMINAMATH_CALUDE_triangle_inequality_l2070_207066

theorem triangle_inequality (a b c : ℝ) (A B C : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = π →
  (a ≥ b ∧ b ≥ c → A ≥ B ∧ B ≥ C) →
  (a * A + b * B + c * C) / (a + b + c) ≥ π / 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l2070_207066


namespace NUMINAMATH_CALUDE_quad_pyramid_volume_l2070_207049

noncomputable section

/-- A quadrilateral pyramid with a square base -/
structure QuadPyramid where
  /-- Side length of the square base -/
  a : ℝ
  /-- Dihedral angle at edge SA -/
  α : ℝ
  /-- The side length is positive -/
  a_pos : 0 < a
  /-- The dihedral angle is within the valid range -/
  α_range : π / 2 < α ∧ α ≤ 2 * π / 3
  /-- Angles between opposite lateral faces are right angles -/
  opposite_faces_right : True

/-- Volume of the quadrilateral pyramid -/
def volume (p : QuadPyramid) : ℝ := (p.a ^ 3 * |Real.cos p.α|) / 3

/-- Theorem stating the volume of the quadrilateral pyramid -/
theorem quad_pyramid_volume (p : QuadPyramid) : 
  volume p = (p.a ^ 3 * |Real.cos p.α|) / 3 := by sorry

end

end NUMINAMATH_CALUDE_quad_pyramid_volume_l2070_207049


namespace NUMINAMATH_CALUDE_bird_families_left_l2070_207009

theorem bird_families_left (initial_families : ℕ) (families_flown_away : ℕ) : 
  initial_families = 67 → families_flown_away = 32 → initial_families - families_flown_away = 35 :=
by sorry

end NUMINAMATH_CALUDE_bird_families_left_l2070_207009


namespace NUMINAMATH_CALUDE_log_division_simplification_l2070_207064

theorem log_division_simplification :
  Real.log 16 / Real.log (1/16) = -1 := by
  sorry

end NUMINAMATH_CALUDE_log_division_simplification_l2070_207064


namespace NUMINAMATH_CALUDE_sqrt_sum_eq_sqrt_prime_l2070_207038

theorem sqrt_sum_eq_sqrt_prime (p : ℕ) (hp : Prime p) :
  ∀ x y : ℕ, Real.sqrt x + Real.sqrt y = Real.sqrt p ↔ (x = 0 ∧ y = p) ∨ (x = p ∧ y = 0) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_sum_eq_sqrt_prime_l2070_207038


namespace NUMINAMATH_CALUDE_min_sum_of_coefficients_tangent_line_l2070_207090

/-- Given a circle and a line tangent to it, find the minimum sum of coefficients -/
theorem min_sum_of_coefficients_tangent_line (a b : ℝ) : 
  a > 0 → 
  b > 0 → 
  (∃ x y : ℝ, x^2 + y^2 = 1 ∧ (a-1)*x + (b-1)*y + a + b = 0) → 
  (∀ x y : ℝ, x^2 + y^2 = 1 → (a-1)*x + (b-1)*y + a + b ≠ 0 ∨ 
    ∃ t : ℝ, (a-1)*(-2*x*t) + (b-1)*(-2*y*t) = 2*x*(a-1) + 2*y*(b-1)) →
  a + b ≥ 2*Real.sqrt 2 - 2 :=
sorry

end NUMINAMATH_CALUDE_min_sum_of_coefficients_tangent_line_l2070_207090


namespace NUMINAMATH_CALUDE_cannot_form_triangle_l2070_207054

/-- Triangle inequality theorem: The sum of the lengths of any two sides of a triangle
    must be greater than the length of the remaining side. -/
axiom triangle_inequality (a b c : ℝ) : a > 0 ∧ b > 0 ∧ c > 0 → a + b > c ∧ b + c > a ∧ c + a > b

/-- A function that determines if three line segments can form a triangle -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

/-- Theorem: The set of line segments (2, 2, 6) cannot form a triangle -/
theorem cannot_form_triangle : ¬ can_form_triangle 2 2 6 := by
  sorry


end NUMINAMATH_CALUDE_cannot_form_triangle_l2070_207054


namespace NUMINAMATH_CALUDE_tangent_circles_bound_l2070_207042

/-- The maximum number of pairs of tangent circles for n circles -/
def l (n : ℕ) : ℕ :=
  match n with
  | 3 => 3
  | 4 => 5
  | 5 => 7
  | 7 => 12
  | 8 => 14
  | 9 => 16
  | 10 => 19
  | _ => 3 * n - 11

/-- Theorem: For n ≥ 9, the maximum number of pairs of tangent circles is at most 3n - 11 -/
theorem tangent_circles_bound (n : ℕ) (h : n ≥ 9) : l n ≤ 3 * n - 11 := by
  sorry

end NUMINAMATH_CALUDE_tangent_circles_bound_l2070_207042


namespace NUMINAMATH_CALUDE_partnership_gain_l2070_207075

/-- Represents the investment and profit share of a partner in the partnership. -/
structure Partner where
  investment : ℕ  -- Amount invested
  duration : ℕ    -- Duration of investment in months
  share : ℕ       -- Share of profit

/-- Represents the partnership with three partners. -/
structure Partnership where
  a : Partner
  b : Partner
  c : Partner

/-- Calculates the total annual gain of the partnership. -/
def totalAnnualGain (p : Partnership) : ℕ :=
  p.a.share + p.b.share + p.c.share

/-- Theorem stating the total annual gain of the partnership. -/
theorem partnership_gain (p : Partnership) 
  (h1 : p.a.investment > 0)
  (h2 : p.b.investment = 2 * p.a.investment)
  (h3 : p.c.investment = 3 * p.a.investment)
  (h4 : p.a.duration = 12)
  (h5 : p.b.duration = 6)
  (h6 : p.c.duration = 4)
  (h7 : p.a.share = 6100)
  (h8 : p.a.share = p.b.share)
  (h9 : p.b.share = p.c.share) :
  totalAnnualGain p = 18300 := by
  sorry

end NUMINAMATH_CALUDE_partnership_gain_l2070_207075


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_sets_l2070_207068

theorem quadratic_inequality_solution_sets
  (a b c : ℝ)
  (h : Set.Ioo (-1/3 : ℝ) 2 = {x : ℝ | a * x^2 + b * x + c > 0}) :
  {x : ℝ | c * x^2 + b * x + a < 0} = Set.Ioo (-3 : ℝ) (1/2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_sets_l2070_207068


namespace NUMINAMATH_CALUDE_base_8_to_10_fraction_l2070_207021

theorem base_8_to_10_fraction (c d : ℕ) : 
  (c < 10 ∧ d < 10) →  -- c and d are base-10 digits
  (5 * 8^2 + 6 * 8 + 3 = 3 * 100 + c * 10 + d) →  -- 563_8 = 3cd_10
  (c * d) / 12 = 7 / 4 := by
sorry

end NUMINAMATH_CALUDE_base_8_to_10_fraction_l2070_207021


namespace NUMINAMATH_CALUDE_smallest_of_five_consecutive_even_numbers_l2070_207094

theorem smallest_of_five_consecutive_even_numbers (a b c d e : ℕ) : 
  (∀ n : ℕ, a = 2*n ∧ b = 2*n + 2 ∧ c = 2*n + 4 ∧ d = 2*n + 6 ∧ e = 2*n + 8) → 
  a + b + c + d + e = 320 → 
  a = 60 :=
by sorry

end NUMINAMATH_CALUDE_smallest_of_five_consecutive_even_numbers_l2070_207094


namespace NUMINAMATH_CALUDE_a_range_for_region_above_l2070_207072

/-- The inequality represents the region above the line -/
def represents_region_above (a : ℝ) : Prop :=
  ∀ x y : ℝ, 3 * a * x + (a^2 - 3 * a + 2) * y - 9 < 0 ↔ 
    y > (9 - 3 * a * x) / (a^2 - 3 * a + 2)

/-- The theorem stating the range of a -/
theorem a_range_for_region_above : 
  ∀ a : ℝ, represents_region_above a ↔ 1 < a ∧ a < 2 := by sorry

end NUMINAMATH_CALUDE_a_range_for_region_above_l2070_207072


namespace NUMINAMATH_CALUDE_food_price_increase_l2070_207034

theorem food_price_increase 
  (initial_students : ℝ) 
  (initial_food_price : ℝ) 
  (initial_food_consumption : ℝ) 
  (h_students_positive : initial_students > 0) 
  (h_price_positive : initial_food_price > 0) 
  (h_consumption_positive : initial_food_consumption > 0) :
  let new_students := 0.9 * initial_students
  let new_food_consumption := 0.9259259259259259 * initial_food_consumption
  let new_food_price := x * initial_food_price
  x = 1.2 ↔ 
    new_students * new_food_consumption * new_food_price = 
    initial_students * initial_food_consumption * initial_food_price := by
sorry

end NUMINAMATH_CALUDE_food_price_increase_l2070_207034


namespace NUMINAMATH_CALUDE_range_of_f_l2070_207098

noncomputable def f (x : ℝ) : ℝ := (1/2) ^ (-x^2 + 2*x)

theorem range_of_f :
  Set.range f = Set.Ioo (1/2) (Real.pi) :=
sorry

end NUMINAMATH_CALUDE_range_of_f_l2070_207098


namespace NUMINAMATH_CALUDE_other_bill_denomination_l2070_207082

-- Define the total amount spent
def total_spent : ℕ := 80

-- Define the number of $10 bills used
def num_ten_bills : ℕ := 2

-- Define the function to calculate the number of other bills
def num_other_bills (n : ℕ) : ℕ := n + 1

-- Define the theorem
theorem other_bill_denomination :
  ∃ (x : ℕ), 
    x * num_other_bills num_ten_bills + 10 * num_ten_bills = total_spent ∧
    x = 20 := by
  sorry

end NUMINAMATH_CALUDE_other_bill_denomination_l2070_207082


namespace NUMINAMATH_CALUDE_sufficient_condition_absolute_value_l2070_207059

theorem sufficient_condition_absolute_value (a : ℝ) : 
  (∀ x : ℝ, 0 < x ∧ x < 4 → |x - 1| < a) → a ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_sufficient_condition_absolute_value_l2070_207059


namespace NUMINAMATH_CALUDE_stratified_sampling_survey_l2070_207056

theorem stratified_sampling_survey (young_population middle_aged_population elderly_population : ℕ)
  (elderly_sampled : ℕ) (young_sampled : ℕ) :
  young_population = 800 →
  middle_aged_population = 1600 →
  elderly_population = 1400 →
  elderly_sampled = 70 →
  (elderly_sampled : ℚ) / elderly_population = (young_sampled : ℚ) / young_population →
  young_sampled = 40 :=
by sorry

end NUMINAMATH_CALUDE_stratified_sampling_survey_l2070_207056


namespace NUMINAMATH_CALUDE_relay_race_tables_l2070_207045

/-- The number of tables required for a relay race with given conditions -/
def num_tables (race_distance : ℕ) (distance_between_1_and_3 : ℕ) : ℕ :=
  (race_distance / (distance_between_1_and_3 / 2)) + 1

theorem relay_race_tables :
  num_tables 1200 400 = 7 :=
by sorry

end NUMINAMATH_CALUDE_relay_race_tables_l2070_207045


namespace NUMINAMATH_CALUDE_largest_integer_x_l2070_207007

theorem largest_integer_x : ∃ x : ℤ, 
  (∀ y : ℤ, (7 - 3 * y > 20 ∧ y ≥ -10) → y ≤ x) ∧ 
  (7 - 3 * x > 20 ∧ x ≥ -10) ∧ 
  x = -5 := by
sorry

end NUMINAMATH_CALUDE_largest_integer_x_l2070_207007


namespace NUMINAMATH_CALUDE_boris_neighbors_l2070_207078

/-- Represents the six people in the circle -/
inductive Person : Type
  | Arkady : Person
  | Boris : Person
  | Vera : Person
  | Galya : Person
  | Danya : Person
  | Egor : Person

/-- Represents the circular arrangement of people -/
def Circle := List Person

/-- Check if two people are standing next to each other in the circle -/
def are_adjacent (c : Circle) (p1 p2 : Person) : Prop :=
  ∃ i : Nat, (c.get? i = some p1 ∧ c.get? ((i + 1) % c.length) = some p2) ∨
             (c.get? i = some p2 ∧ c.get? ((i + 1) % c.length) = some p1)

/-- Check if two people are standing opposite each other in the circle -/
def are_opposite (c : Circle) (p1 p2 : Person) : Prop :=
  ∃ i : Nat, c.get? i = some p1 ∧ c.get? ((i + c.length / 2) % c.length) = some p2

theorem boris_neighbors (c : Circle) :
  c.length = 6 →
  are_adjacent c Person.Danya Person.Vera →
  are_adjacent c Person.Danya Person.Egor →
  are_opposite c Person.Galya Person.Egor →
  ¬ are_adjacent c Person.Arkady Person.Galya →
  (are_adjacent c Person.Boris Person.Arkady ∧ are_adjacent c Person.Boris Person.Galya) :=
by sorry

end NUMINAMATH_CALUDE_boris_neighbors_l2070_207078


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l2070_207055

theorem diophantine_equation_solutions (k : ℤ) :
  (k > 7 → ∃ (x y : ℕ), 5 * x + 3 * y = k) ∧
  (k > 15 → ∃ (x y : ℕ+), 5 * x + 3 * y = k) ∧
  (∀ N : ℤ, (∀ k > N, ∃ (x y : ℕ+), 5 * x + 3 * y = k) → N ≥ 15) :=
by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l2070_207055


namespace NUMINAMATH_CALUDE_quadratic_real_roots_condition_l2070_207084

/-- If the quadratic equation x^2 - 3x + 2m = 0 has real roots, then m ≤ 9/8 -/
theorem quadratic_real_roots_condition (m : ℝ) : 
  (∃ x : ℝ, x^2 - 3*x + 2*m = 0) → m ≤ 9/8 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_condition_l2070_207084


namespace NUMINAMATH_CALUDE_stock_dividend_rate_l2070_207062

/-- Given a stock with a certain yield and price, calculate its dividend rate. -/
def dividend_rate (yield : ℝ) (price : ℝ) : ℝ :=
  yield * price

/-- Theorem: The dividend rate of a stock yielding 8% quoted at 150 is 12. -/
theorem stock_dividend_rate :
  let yield : ℝ := 0.08
  let price : ℝ := 150
  dividend_rate yield price = 12 := by
  sorry

end NUMINAMATH_CALUDE_stock_dividend_rate_l2070_207062


namespace NUMINAMATH_CALUDE_smallest_n_divisibility_l2070_207096

theorem smallest_n_divisibility : ∃ n : ℕ+, 
  (∀ m : ℕ+, m < n → (¬(24 ∣ m^2) ∨ ¬(540 ∣ m^3))) ∧ 
  24 ∣ n^2 ∧ 
  540 ∣ n^3 ∧ 
  n = 60 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_divisibility_l2070_207096


namespace NUMINAMATH_CALUDE_andy_diana_weight_l2070_207086

theorem andy_diana_weight (a b c d : ℝ) 
  (h1 : a + b = 300)
  (h2 : b + c = 280)
  (h3 : c + d = 310) :
  a + d = 330 := by
  sorry

end NUMINAMATH_CALUDE_andy_diana_weight_l2070_207086


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l2070_207029

-- Problem 1
theorem problem_1 (a b : ℝ) (h1 : (a + b)^2 = 6) (h2 : (a - b)^2 = 2) : 
  a^2 + b^2 = 4 ∧ a * b = 1 := by sorry

-- Problem 2
theorem problem_2 (x : ℝ) (h : x + 1/x = 3) : 
  x^2 + 1/x^2 = 7 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l2070_207029


namespace NUMINAMATH_CALUDE_solution_sets_union_l2070_207044

-- Define the solution sets M and N
def M (p : ℝ) : Set ℝ := {x | x^2 - p*x + 6 = 0}
def N (q : ℝ) : Set ℝ := {x | x^2 + 6*x - q = 0}

-- State the theorem
theorem solution_sets_union (p q : ℝ) :
  (∃ (x : ℝ), x ∈ M p ∧ x ∈ N q) ∧ (M p ∩ N q = {2}) →
  M p ∪ N q = {2, 3, -8} :=
by sorry

end NUMINAMATH_CALUDE_solution_sets_union_l2070_207044


namespace NUMINAMATH_CALUDE_b_51_equals_5151_l2070_207002

def a (n : ℕ) : ℕ := n * (n + 1) / 2

def is_not_even (n : ℕ) : Prop := ¬(2 ∣ n)

def b : ℕ → ℕ := sorry

theorem b_51_equals_5151 : b 51 = 5151 := by sorry

end NUMINAMATH_CALUDE_b_51_equals_5151_l2070_207002


namespace NUMINAMATH_CALUDE_valid_a_values_l2070_207022

/-- Set A defined by the quadratic equation x^2 - 2x - 3 = 0 -/
def A : Set ℝ := {x | x^2 - 2*x - 3 = 0}

/-- Set B defined by the linear equation ax - 1 = 0, parameterized by a -/
def B (a : ℝ) : Set ℝ := {x | a*x - 1 = 0}

/-- The set of values for a such that B is a subset of A -/
def valid_a : Set ℝ := {a | B a ⊆ A}

/-- Theorem stating that the set of valid a values is {-1, 0, 1/3} -/
theorem valid_a_values : valid_a = {-1, 0, 1/3} := by sorry

end NUMINAMATH_CALUDE_valid_a_values_l2070_207022


namespace NUMINAMATH_CALUDE_largest_c_value_l2070_207025

theorem largest_c_value (c : ℝ) : (3 * c + 4) * (c - 2) = 9 * c → c ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_largest_c_value_l2070_207025


namespace NUMINAMATH_CALUDE_remainder_1234567_div_256_l2070_207036

theorem remainder_1234567_div_256 : 1234567 % 256 = 45 := by
  sorry

end NUMINAMATH_CALUDE_remainder_1234567_div_256_l2070_207036


namespace NUMINAMATH_CALUDE_quadratic_trinomial_characterization_l2070_207048

/-- A quadratic trinomial with real coefficients -/
structure QuadraticTrinomial where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Condition: Replacing any coefficient with 1 results in a trinomial with exactly one root -/
def has_one_root_when_replaced (qt : QuadraticTrinomial) : Prop :=
  (qt.b^2 - 4*qt.c = 0) ∧ 
  (1 - 4*qt.a*qt.c = 0) ∧ 
  (qt.b^2 - 4*qt.a = 0)

/-- Theorem: Characterization of quadratic trinomials satisfying the condition -/
theorem quadratic_trinomial_characterization (qt : QuadraticTrinomial) :
  has_one_root_when_replaced qt →
  (qt.a = 1/2 ∧ qt.c = 1/2 ∧ (qt.b = Real.sqrt 2 ∨ qt.b = -Real.sqrt 2)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_trinomial_characterization_l2070_207048


namespace NUMINAMATH_CALUDE_cos_squared_alpha_minus_pi_fourth_l2070_207013

theorem cos_squared_alpha_minus_pi_fourth (α : ℝ) (h : Real.sin (2 * α) = 1 / 3) :
  Real.cos (α - Real.pi / 4) ^ 2 = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_cos_squared_alpha_minus_pi_fourth_l2070_207013


namespace NUMINAMATH_CALUDE_marcos_salary_calculation_l2070_207024

theorem marcos_salary_calculation (initial_salary : ℝ) : 
  initial_salary = 2500 →
  let salary_after_first_raise := initial_salary * 1.15
  let salary_after_second_raise := salary_after_first_raise * 1.10
  let final_salary := salary_after_second_raise * 0.85
  final_salary = 2688.125 := by
sorry

end NUMINAMATH_CALUDE_marcos_salary_calculation_l2070_207024


namespace NUMINAMATH_CALUDE_cupcake_difference_l2070_207030

/-- Betty's cupcake production rate per hour -/
def betty_rate : ℕ := 10

/-- Dora's cupcake production rate per hour -/
def dora_rate : ℕ := 8

/-- Duration of Betty's break in hours -/
def betty_break : ℕ := 2

/-- Total time elapsed in hours -/
def total_time : ℕ := 5

/-- Theorem stating that after 5 hours, the difference between Betty's and Dora's cupcake counts is 10 -/
theorem cupcake_difference : 
  (dora_rate * total_time) - (betty_rate * (total_time - betty_break)) = 10 := by
  sorry

end NUMINAMATH_CALUDE_cupcake_difference_l2070_207030


namespace NUMINAMATH_CALUDE_units_digit_of_27_times_46_l2070_207004

theorem units_digit_of_27_times_46 : (27 * 46) % 10 = 2 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_27_times_46_l2070_207004


namespace NUMINAMATH_CALUDE_charlotte_tuesday_poodles_l2070_207053

/-- Represents the schedule and constraints for Charlotte's dog walking --/
structure DogWalkingSchedule where
  monday_poodles : Nat
  monday_chihuahuas : Nat
  wednesday_labradors : Nat
  poodle_time : Nat
  chihuahua_time : Nat
  labrador_time : Nat
  total_time : Nat

/-- Calculates the number of poodles Charlotte can walk on Tuesday --/
def tuesday_poodles (schedule : DogWalkingSchedule) : Nat :=
  let monday_time := schedule.monday_poodles * schedule.poodle_time + 
                     schedule.monday_chihuahuas * schedule.chihuahua_time
  let tuesday_chihuahua_time := schedule.monday_chihuahuas * schedule.chihuahua_time
  let wednesday_time := schedule.wednesday_labradors * schedule.labrador_time
  let available_time := schedule.total_time - monday_time - tuesday_chihuahua_time - wednesday_time
  available_time / schedule.poodle_time

/-- Theorem stating that given the schedule constraints, Charlotte can walk 4 poodles on Tuesday --/
theorem charlotte_tuesday_poodles : 
  ∀ (schedule : DogWalkingSchedule), 
  schedule.monday_poodles = 4 ∧ 
  schedule.monday_chihuahuas = 2 ∧ 
  schedule.wednesday_labradors = 4 ∧ 
  schedule.poodle_time = 2 ∧ 
  schedule.chihuahua_time = 1 ∧ 
  schedule.labrador_time = 3 ∧ 
  schedule.total_time = 32 → 
  tuesday_poodles schedule = 4 := by
  sorry


end NUMINAMATH_CALUDE_charlotte_tuesday_poodles_l2070_207053


namespace NUMINAMATH_CALUDE_bowTie_equation_solution_l2070_207026

-- Define the bow tie operation
noncomputable def bowTie (a b : ℝ) : ℝ :=
  a + Real.sqrt (b + Real.sqrt (b + Real.sqrt (b + Real.sqrt b)))

-- Theorem statement
theorem bowTie_equation_solution (h : ℝ) :
  bowTie 5 h = 7 → h = 2 := by sorry

end NUMINAMATH_CALUDE_bowTie_equation_solution_l2070_207026


namespace NUMINAMATH_CALUDE_count_solutions_for_a_main_result_l2070_207033

theorem count_solutions_for_a (max_a : Nat) : Nat :=
  let count_pairs (a : Nat) : Nat :=
    (Finset.filter (fun p : Nat × Nat =>
      let m := p.1
      let n := p.2
      n * (1 - m) + a * (1 + m) = 0 ∧ 
      m > 0 ∧ n > 0
    ) (Finset.product (Finset.range (max_a + 1)) (Finset.range (max_a + 1)))).card

  (Finset.filter (fun a : Nat =>
    a > 0 ∧ a ≤ max_a ∧ count_pairs a = 6
  ) (Finset.range (max_a + 1))).card

theorem main_result : count_solutions_for_a 50 = 12 := by
  sorry

end NUMINAMATH_CALUDE_count_solutions_for_a_main_result_l2070_207033


namespace NUMINAMATH_CALUDE_inequality_solution_l2070_207020

theorem inequality_solution (x : ℝ) : -3 * x^2 + 5 * x + 4 < 0 ∧ x > 0 → x ∈ Set.Ioo 0 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l2070_207020


namespace NUMINAMATH_CALUDE_train_speed_conversion_l2070_207017

theorem train_speed_conversion (speed_kmph : ℝ) (speed_ms : ℝ) : 
  speed_kmph = 216 → speed_ms = 60 → speed_kmph * (1000 / 3600) = speed_ms :=
by
  sorry

end NUMINAMATH_CALUDE_train_speed_conversion_l2070_207017


namespace NUMINAMATH_CALUDE_regression_line_not_always_through_point_l2070_207028

/-- A sample data point in a regression analysis -/
structure DataPoint where
  x : ℝ
  y : ℝ

/-- Linear regression equation -/
structure LinearRegression where
  a : ℝ  -- intercept
  b : ℝ  -- slope

/-- Check if a point lies on a line defined by a linear regression equation -/
def pointOnLine (p : DataPoint) (reg : LinearRegression) : Prop :=
  p.y = reg.b * p.x + reg.a

/-- Theorem stating that it's not necessarily true that a linear regression line passes through at least one sample point -/
theorem regression_line_not_always_through_point :
  ∃ (n : ℕ) (data : Fin n → DataPoint) (reg : LinearRegression),
    ∀ i : Fin n, ¬(pointOnLine (data i) reg) :=
sorry

end NUMINAMATH_CALUDE_regression_line_not_always_through_point_l2070_207028


namespace NUMINAMATH_CALUDE_intersection_M_N_l2070_207092

def M : Set ℤ := {-2, -1, 0, 1}
def N : Set ℤ := {n : ℤ | -1 ≤ n ∧ n ≤ 3}

theorem intersection_M_N : M ∩ N = {-1, 0, 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2070_207092


namespace NUMINAMATH_CALUDE_polygon_sides_count_l2070_207069

/-- A polygon is a closed plane figure with straight sides. -/
structure Polygon where
  sides : ℕ

/-- Represents the number of triangles formed by connecting a point on a side to all vertices. -/
def triangles_formed (p : Polygon) : ℕ := p.sides - 1

/-- The polygon in our problem. -/
def our_polygon : Polygon :=
  { sides := 2024 }

/-- The theorem stating our problem. -/
theorem polygon_sides_count : triangles_formed our_polygon = 2023 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_count_l2070_207069


namespace NUMINAMATH_CALUDE_martha_improvement_l2070_207091

/-- Represents Martha's running performance at a given time --/
structure Performance where
  laps : ℕ
  time : ℕ
  
/-- Calculates the lap time in seconds given a Performance --/
def lapTime (p : Performance) : ℚ :=
  (p.time * 60) / p.laps

/-- Martha's initial performance --/
def initialPerformance : Performance := ⟨15, 30⟩

/-- Martha's performance after two months --/
def finalPerformance : Performance := ⟨20, 27⟩

/-- Theorem stating the improvement in Martha's lap time --/
theorem martha_improvement :
  lapTime initialPerformance - lapTime finalPerformance = 39 := by
  sorry

end NUMINAMATH_CALUDE_martha_improvement_l2070_207091


namespace NUMINAMATH_CALUDE_complex_sum_powers_l2070_207037

theorem complex_sum_powers (z : ℂ) (h : z^2 - z + 1 = 0) :
  z^99 + z^100 + z^101 + z^102 + z^103 = 2 + Complex.I * Real.sqrt 3 ∨
  z^99 + z^100 + z^101 + z^102 + z^103 = 2 - Complex.I * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_complex_sum_powers_l2070_207037


namespace NUMINAMATH_CALUDE_triangle_movement_path_length_l2070_207035

/-- Represents the movement of a triangle inside a square -/
structure TriangleMovement where
  square_side : ℝ
  triangle_side : ℝ
  initial_rotation_radius : ℝ
  final_rotation_radius : ℝ
  initial_rotation_angle : ℝ
  final_rotation_angle : ℝ

/-- Calculates the total path traversed by vertex P -/
def total_path_length (m : TriangleMovement) : ℝ :=
  m.initial_rotation_radius * m.initial_rotation_angle +
  m.final_rotation_radius * m.final_rotation_angle

/-- The theorem to be proved -/
theorem triangle_movement_path_length :
  ∀ (m : TriangleMovement),
  m.square_side = 6 ∧
  m.triangle_side = 3 ∧
  m.initial_rotation_radius = m.triangle_side ∧
  m.final_rotation_radius = (m.square_side / 2 + m.triangle_side / 2) ∧
  m.initial_rotation_angle = Real.pi ∧
  m.final_rotation_angle = 2 * Real.pi →
  total_path_length m = 12 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_triangle_movement_path_length_l2070_207035


namespace NUMINAMATH_CALUDE_correct_average_after_error_l2070_207047

theorem correct_average_after_error (n : ℕ) (initial_avg : ℚ) (wrong_mark correct_mark : ℚ) :
  n = 10 →
  initial_avg = 100 →
  wrong_mark = 50 →
  correct_mark = 10 →
  (n : ℚ) * initial_avg - wrong_mark + correct_mark = (n : ℚ) * 96 :=
by
  sorry

end NUMINAMATH_CALUDE_correct_average_after_error_l2070_207047


namespace NUMINAMATH_CALUDE_calorie_allowance_for_longevity_l2070_207018

/-- Calculates the weekly calorie allowance for a person in their 60s aiming to live to 100 years old -/
def weeklyCalorieAllowance (averageDailyAllowance : ℕ) (reduction : ℕ) (daysInWeek : ℕ) : ℕ :=
  (averageDailyAllowance - reduction) * daysInWeek

/-- Theorem stating the weekly calorie allowance for a person in their 60s aiming to live to 100 years old -/
theorem calorie_allowance_for_longevity :
  weeklyCalorieAllowance 2000 500 7 = 10500 := by
  sorry

#eval weeklyCalorieAllowance 2000 500 7

end NUMINAMATH_CALUDE_calorie_allowance_for_longevity_l2070_207018


namespace NUMINAMATH_CALUDE_sum_of_specific_repeating_decimals_l2070_207052

/-- Definition of a repeating decimal with a 3-digit repetend -/
def repeating_decimal (a b c : ℕ) : ℚ := (a * 100 + b * 10 + c) / 999

/-- The sum of two specific repeating decimals equals 161/999 -/
theorem sum_of_specific_repeating_decimals : 
  repeating_decimal 1 3 7 + repeating_decimal 0 2 4 = 161 / 999 := by sorry

end NUMINAMATH_CALUDE_sum_of_specific_repeating_decimals_l2070_207052


namespace NUMINAMATH_CALUDE_chord_length_theorem_l2070_207051

-- Define the line equation
def line_equation (x y : ℝ) : Prop := x + 2*y = 0

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := (x - 3)^2 + (y - 1)^2 = 25

-- Theorem statement
theorem chord_length_theorem :
  ∃ (chord_length : ℝ),
    (∀ (x y : ℝ), line_equation x y → circle_equation x y → 
      chord_length = 4 * Real.sqrt 5) :=
sorry

end NUMINAMATH_CALUDE_chord_length_theorem_l2070_207051


namespace NUMINAMATH_CALUDE_cos_alpha_plus_20_eq_neg_alpha_l2070_207032

theorem cos_alpha_plus_20_eq_neg_alpha (α : ℝ) (h : Real.sin (α - 70 * Real.pi / 180) = α) :
  Real.cos (α + 20 * Real.pi / 180) = -α := by
  sorry

end NUMINAMATH_CALUDE_cos_alpha_plus_20_eq_neg_alpha_l2070_207032


namespace NUMINAMATH_CALUDE_monomial_sum_l2070_207043

/-- Given two monomials that form a monomial when added together, prove that m + n = 4 -/
theorem monomial_sum (m n : ℕ) : 
  (∃ (a : ℝ), ∀ (x y : ℝ), 2 * x^(m-1) * y^2 + (1/3) * x^2 * y^(n+1) = a * x^2 * y^2) → 
  m + n = 4 := by
  sorry

end NUMINAMATH_CALUDE_monomial_sum_l2070_207043


namespace NUMINAMATH_CALUDE_income_ratio_is_seven_to_six_l2070_207050

/-- Represents the income and expenditure of a person -/
structure Person where
  income : ℕ
  expenditure : ℕ

/-- Given the conditions of the problem, prove that the ratio of Rajan's income to Balan's income is 7:6 -/
theorem income_ratio_is_seven_to_six 
  (rajan balan : Person)
  (h1 : rajan.expenditure * 5 = balan.expenditure * 6)
  (h2 : rajan.income - rajan.expenditure = 1000)
  (h3 : balan.income - balan.expenditure = 1000)
  (h4 : rajan.income = 7000) :
  7 * balan.income = 6 * rajan.income := by
  sorry

#check income_ratio_is_seven_to_six

end NUMINAMATH_CALUDE_income_ratio_is_seven_to_six_l2070_207050


namespace NUMINAMATH_CALUDE_square_difference_symmetry_l2070_207046

theorem square_difference_symmetry (x y : ℝ) : (x - y)^2 = (y - x)^2 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_symmetry_l2070_207046


namespace NUMINAMATH_CALUDE_investment_partnership_profit_share_l2070_207074

/-- Investment partnership problem -/
theorem investment_partnership_profit_share
  (investment_B : ℝ)
  (investment_A : ℝ)
  (investment_C : ℝ)
  (investment_D : ℝ)
  (time_A : ℝ)
  (time_B : ℝ)
  (time_C : ℝ)
  (time_D : ℝ)
  (total_profit : ℝ)
  (h1 : investment_A = 3 * investment_B)
  (h2 : investment_B = 2 / 3 * investment_C)
  (h3 : investment_D = 1 / 2 * (investment_A + investment_B + investment_C))
  (h4 : time_A = 6)
  (h5 : time_B = 9)
  (h6 : time_C = 12)
  (h7 : time_D = 4)
  (h8 : total_profit = 22000) :
  (investment_B * time_B) / (investment_A * time_A + investment_B * time_B + investment_C * time_C + investment_D * time_D) * total_profit = 3666.67 := by
  sorry

end NUMINAMATH_CALUDE_investment_partnership_profit_share_l2070_207074


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l2070_207063

theorem expression_simplification_and_evaluation (a : ℚ) (h : a = 1/2) :
  (a - 1) / (a - 2) * ((a^2 - 4) / (a^2 - 2*a + 1)) - 2 / (a - 1) = -1 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l2070_207063


namespace NUMINAMATH_CALUDE_tylers_scissors_l2070_207081

/-- Proves the number of scissors Tyler bought given his initial amount, costs, and remaining amount -/
theorem tylers_scissors (initial_amount : ℕ) (scissors_cost : ℕ) (erasers_cost : ℕ) (remaining_amount : ℕ) :
  initial_amount = 100 →
  scissors_cost = 5 →
  erasers_cost = 40 →
  remaining_amount = 20 →
  ∃ (num_scissors : ℕ), num_scissors * scissors_cost + erasers_cost = initial_amount - remaining_amount ∧ num_scissors = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_tylers_scissors_l2070_207081


namespace NUMINAMATH_CALUDE_tricycle_count_l2070_207076

/-- The number of tricycles in a group of children -/
def num_tricycles (total_children : ℕ) (total_wheels : ℕ) : ℕ :=
  total_children - (total_wheels - 3 * total_children) / 1

/-- Theorem stating that given 10 children and 26 wheels, there are 6 tricycles -/
theorem tricycle_count : num_tricycles 10 26 = 6 := by
  sorry

end NUMINAMATH_CALUDE_tricycle_count_l2070_207076


namespace NUMINAMATH_CALUDE_unit_vectors_collinear_with_vector_l2070_207003

def vector : ℝ × ℝ × ℝ := (-3, -4, 5)

theorem unit_vectors_collinear_with_vector :
  let norm := Real.sqrt ((-3)^2 + (-4)^2 + 5^2)
  let unit_vector₁ : ℝ × ℝ × ℝ := (3 * Real.sqrt 2 / 10, 2 * Real.sqrt 2 / 5, -Real.sqrt 2 / 2)
  let unit_vector₂ : ℝ × ℝ × ℝ := (-3 * Real.sqrt 2 / 10, -2 * Real.sqrt 2 / 5, Real.sqrt 2 / 2)
  (∃ (k : ℝ), vector = (k • unit_vector₁)) ∧
  (∃ (k : ℝ), vector = (k • unit_vector₂)) ∧
  (norm * norm = (-3)^2 + (-4)^2 + 5^2) ∧
  (Real.sqrt 2 * Real.sqrt 2 = 2) ∧
  (∀ (v : ℝ × ℝ × ℝ), (∃ (k : ℝ), vector = (k • v)) → (v = unit_vector₁ ∨ v = unit_vector₂)) :=
by sorry

end NUMINAMATH_CALUDE_unit_vectors_collinear_with_vector_l2070_207003


namespace NUMINAMATH_CALUDE_tenth_student_score_l2070_207065

/-- Represents a valid arithmetic sequence of exam scores -/
structure ExamScores where
  scores : Fin 10 → ℕ
  is_arithmetic : ∀ i j k : Fin 10, i.val + k.val = j.val + j.val → scores i + scores k = scores j + scores j
  max_score : ∀ i : Fin 10, scores i ≤ 100
  sum_middle : scores 2 + scores 3 + scores 4 + scores 5 = 354
  contains_96 : ∃ i : Fin 10, scores i = 96

/-- The theorem stating the possible scores for the 10th student -/
theorem tenth_student_score (e : ExamScores) : e.scores 0 = 61 ∨ e.scores 0 = 72 := by
  sorry

end NUMINAMATH_CALUDE_tenth_student_score_l2070_207065


namespace NUMINAMATH_CALUDE_race_end_people_count_l2070_207019

/-- The number of people in cars at the end of a race -/
def people_at_race_end (num_cars : ℕ) (initial_people_per_car : ℕ) (additional_passengers : ℕ) : ℕ :=
  num_cars * (initial_people_per_car + additional_passengers)

/-- Theorem stating the number of people at the end of the race -/
theorem race_end_people_count : 
  people_at_race_end 20 3 1 = 80 := by sorry

end NUMINAMATH_CALUDE_race_end_people_count_l2070_207019


namespace NUMINAMATH_CALUDE_f_divisible_by_36_l2070_207005

def f (n : ℕ) : ℕ := (2 * n + 7) * 3^n + 9

theorem f_divisible_by_36 : ∀ n : ℕ, 36 ∣ f n := by sorry

end NUMINAMATH_CALUDE_f_divisible_by_36_l2070_207005


namespace NUMINAMATH_CALUDE_expansion_theorem_l2070_207011

theorem expansion_theorem (x : ℝ) (n : ℕ) :
  (∃ k : ℕ, (Nat.choose n 2) / (Nat.choose n 4) = 3 / 14) →
  (n = 10 ∧ 
   ∃ m : ℕ, m = 8 ∧ 
   (Nat.choose n m) = 45 ∧ 
   20 - 2 * m - (1/2) * m = 0) :=
by sorry

end NUMINAMATH_CALUDE_expansion_theorem_l2070_207011


namespace NUMINAMATH_CALUDE_sum_squares_five_consecutive_integers_l2070_207060

theorem sum_squares_five_consecutive_integers (n : ℤ) :
  ∃ k : ℤ, (n - 2)^2 + (n - 1)^2 + n^2 + (n + 1)^2 + (n + 2)^2 = 5 * k :=
by sorry

end NUMINAMATH_CALUDE_sum_squares_five_consecutive_integers_l2070_207060


namespace NUMINAMATH_CALUDE_midpoint_rectangle_area_l2070_207000

/-- Given a rectangle with area 48 and length-to-width ratio 3:2, 
    the area of the rectangle formed by connecting its side midpoints is 12. -/
theorem midpoint_rectangle_area (length width : ℝ) : 
  length * width = 48 →
  length / width = 3 / 2 →
  (length / 2) * (width / 2) = 12 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_rectangle_area_l2070_207000


namespace NUMINAMATH_CALUDE_quadratic_equation_pairs_l2070_207085

theorem quadratic_equation_pairs : 
  let count := Finset.filter (fun p : ℕ × ℕ => 
    let b := p.1
    let c := p.2
    b > 0 ∧ c > 0 ∧ 
    b + c ≤ 10 ∧
    b^2 - 4*c = 0 ∧
    c^2 - 4*b ≤ 0) (Finset.product (Finset.range 11) (Finset.range 11))
  count.card = 2 := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_pairs_l2070_207085


namespace NUMINAMATH_CALUDE_max_quartets_correct_max_quartets_5x5_l2070_207073

/-- Represents a rectangle on a grid --/
structure Rectangle where
  m : ℕ
  n : ℕ

/-- Calculates the maximum number of quartets in a rectangle --/
def max_quartets (rect : Rectangle) : ℕ :=
  if rect.m % 2 = 0 ∧ rect.n % 2 = 1 then
    (rect.m * (rect.n - 1)) / 4
  else if rect.m % 2 = 1 ∧ rect.n % 2 = 0 then
    (rect.n * (rect.m - 1)) / 4
  else if rect.m % 2 = 1 ∧ rect.n % 2 = 1 then
    if (rect.n - 1) % 4 = 0 then
      (rect.m * (rect.n - 1)) / 4
    else
      (rect.m * (rect.n - 1) - 2) / 4
  else
    (rect.m * rect.n) / 4

theorem max_quartets_correct (rect : Rectangle) :
  max_quartets rect =
    if rect.m % 2 = 0 ∧ rect.n % 2 = 1 then
      (rect.m * (rect.n - 1)) / 4
    else if rect.m % 2 = 1 ∧ rect.n % 2 = 0 then
      (rect.n * (rect.m - 1)) / 4
    else if rect.m % 2 = 1 ∧ rect.n % 2 = 1 then
      if (rect.n - 1) % 4 = 0 then
        (rect.m * (rect.n - 1)) / 4
      else
        (rect.m * (rect.n - 1) - 2) / 4
    else
      (rect.m * rect.n) / 4 :=
by sorry

/-- Specific case for 5x5 square --/
def square_5x5 : Rectangle := { m := 5, n := 5 }

theorem max_quartets_5x5 :
  max_quartets square_5x5 = 5 :=
by sorry

end NUMINAMATH_CALUDE_max_quartets_correct_max_quartets_5x5_l2070_207073


namespace NUMINAMATH_CALUDE_umbrella_arrangement_count_l2070_207087

/-- The number of ways to arrange n people with distinct heights in an umbrella shape -/
def umbrella_arrangements (n : ℕ) : ℕ :=
  sorry

/-- There are 7 actors with distinct heights to be arranged -/
def num_actors : ℕ := 7

theorem umbrella_arrangement_count :
  umbrella_arrangements num_actors = 20 := by sorry

end NUMINAMATH_CALUDE_umbrella_arrangement_count_l2070_207087


namespace NUMINAMATH_CALUDE_infinite_solutions_in_interval_l2070_207010

theorem infinite_solutions_in_interval (x : Real) (h : x ∈ Set.Icc 0 (2 * Real.pi)) :
  Real.cos ((Real.pi / 2) * Real.cos x + (Real.pi / 2) * Real.sin x) =
  Real.sin ((Real.pi / 2) * Real.cos x - (Real.pi / 2) * Real.sin x) := by
  sorry

end NUMINAMATH_CALUDE_infinite_solutions_in_interval_l2070_207010


namespace NUMINAMATH_CALUDE_union_equals_universe_l2070_207027

def U : Finset ℕ := {2, 3, 4, 5, 6}
def M : Finset ℕ := {3, 4, 5}
def N : Finset ℕ := {2, 4, 5, 6}

theorem union_equals_universe : M ∪ N = U := by
  sorry

end NUMINAMATH_CALUDE_union_equals_universe_l2070_207027


namespace NUMINAMATH_CALUDE_expected_rolls_non_leap_year_l2070_207080

/-- The expected number of times Alice rolls her die in a non-leap year -/
def expected_rolls (days : ℕ) (die_sides : ℕ) (reroll_value : ℕ) : ℚ :=
  (die_sides / (die_sides - 1 : ℚ)) * days

/-- Theorem: Expected number of rolls in a non-leap year -/
theorem expected_rolls_non_leap_year :
  expected_rolls 365 8 8 = 417.14285714285714285714285714285714285714 := by
  sorry

end NUMINAMATH_CALUDE_expected_rolls_non_leap_year_l2070_207080


namespace NUMINAMATH_CALUDE_max_value_ad_minus_bc_l2070_207039

theorem max_value_ad_minus_bc :
  ∃ (a b c d : ℤ),
    a ∈ ({-1, 1, 2} : Set ℤ) ∧
    b ∈ ({-1, 1, 2} : Set ℤ) ∧
    c ∈ ({-1, 1, 2} : Set ℤ) ∧
    d ∈ ({-1, 1, 2} : Set ℤ) ∧
    a * d - b * c = 6 ∧
    ∀ (x y z w : ℤ),
      x ∈ ({-1, 1, 2} : Set ℤ) →
      y ∈ ({-1, 1, 2} : Set ℤ) →
      z ∈ ({-1, 1, 2} : Set ℤ) →
      w ∈ ({-1, 1, 2} : Set ℤ) →
      x * w - y * z ≤ 6 :=
by
  sorry

end NUMINAMATH_CALUDE_max_value_ad_minus_bc_l2070_207039


namespace NUMINAMATH_CALUDE_average_equation_solution_l2070_207095

theorem average_equation_solution (x : ℝ) : 
  ((2*x + 4) + (5*x + 3) + (3*x + 8)) / 3 = 3*x - 5 → x = -30 := by
  sorry

end NUMINAMATH_CALUDE_average_equation_solution_l2070_207095


namespace NUMINAMATH_CALUDE_irrational_floor_inequality_l2070_207016

theorem irrational_floor_inequality : ∃ (a b : ℝ), 
  Irrational a ∧ Irrational b ∧ a > 1 ∧ b > 1 ∧
  ∀ (m n : ℕ), m > 0 → n > 0 → ⌊a^m⌋ ≠ ⌊b^n⌋ := by
  sorry

end NUMINAMATH_CALUDE_irrational_floor_inequality_l2070_207016


namespace NUMINAMATH_CALUDE_linear_functions_coefficient_difference_l2070_207031

/-- Given linear functions f and g, and their composition h with a known inverse,
    prove that the difference of coefficients of f is 5. -/
theorem linear_functions_coefficient_difference (a b : ℝ) : 
  (∃ (f g h : ℝ → ℝ), 
    (∀ x, f x = a * x + b) ∧ 
    (∀ x, g x = -2 * x + 7) ∧ 
    (∀ x, h x = f (g x)) ∧ 
    (∀ x, Function.invFun h x = x + 9)) → 
  a - b = 5 := by
sorry

end NUMINAMATH_CALUDE_linear_functions_coefficient_difference_l2070_207031


namespace NUMINAMATH_CALUDE_students_in_all_classes_l2070_207041

/-- Proves that 8 students are registered for all 3 classes given the problem conditions -/
theorem students_in_all_classes (total_students : ℕ) (history_students : ℕ) (math_students : ℕ) 
  (english_students : ℕ) (two_classes_students : ℕ) : ℕ :=
by
  sorry

#check students_in_all_classes 68 19 14 26 7

end NUMINAMATH_CALUDE_students_in_all_classes_l2070_207041


namespace NUMINAMATH_CALUDE_triangle_cookie_cutters_l2070_207001

theorem triangle_cookie_cutters (total_sides : ℕ) (square_cutters : ℕ) (hexagon_cutters : ℕ) 
  (h1 : total_sides = 46)
  (h2 : square_cutters = 4)
  (h3 : hexagon_cutters = 2) :
  ∃ (triangle_cutters : ℕ), 
    triangle_cutters * 3 + square_cutters * 4 + hexagon_cutters * 6 = total_sides ∧ 
    triangle_cutters = 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_cookie_cutters_l2070_207001


namespace NUMINAMATH_CALUDE_carries_revenue_l2070_207023

/-- Represents the harvest quantities of vegetables -/
structure Harvest where
  tomatoes : ℕ
  carrots : ℕ
  eggplants : ℕ
  cucumbers : ℕ

/-- Represents the selling prices of vegetables -/
structure Prices where
  tomato : ℚ
  carrot : ℚ
  eggplant : ℚ
  cucumber : ℚ

/-- Calculates the total revenue from selling all vegetables -/
def totalRevenue (h : Harvest) (p : Prices) : ℚ :=
  h.tomatoes * p.tomato +
  h.carrots * p.carrot +
  h.eggplants * p.eggplant +
  h.cucumbers * p.cucumber

/-- Theorem stating that Carrie's total revenue is $1156.25 -/
theorem carries_revenue :
  let h : Harvest := { tomatoes := 200, carrots := 350, eggplants := 120, cucumbers := 75 }
  let p : Prices := { tomato := 1, carrot := 3/2, eggplant := 5/2, cucumber := 7/4 }
  totalRevenue h p = 4625/4 := by
  sorry

#eval (4625/4 : ℚ)  -- This should evaluate to 1156.25

end NUMINAMATH_CALUDE_carries_revenue_l2070_207023


namespace NUMINAMATH_CALUDE_line_point_x_coordinate_l2070_207099

/-- Theorem: For a line passing through points (x₁, -4) and (5, 0.8) with slope 0.8, x₁ = -1 -/
theorem line_point_x_coordinate (x₁ : ℝ) : 
  let y₁ : ℝ := -4
  let x₂ : ℝ := 5
  let y₂ : ℝ := 0.8
  let k : ℝ := 0.8
  (y₂ - y₁) / (x₂ - x₁) = k → x₁ = -1 := by
  sorry

end NUMINAMATH_CALUDE_line_point_x_coordinate_l2070_207099


namespace NUMINAMATH_CALUDE_opposite_of_2xyz_l2070_207015

theorem opposite_of_2xyz (x y z : ℝ) : 
  Real.sqrt (2 * x - 1) + Real.sqrt (1 - 2 * x) + |x - 2 * y| + |z + 4 * y| = 0 → 
  -(2 * x * y * z) = (1 : ℝ) / 4 := by
sorry

end NUMINAMATH_CALUDE_opposite_of_2xyz_l2070_207015


namespace NUMINAMATH_CALUDE_smallest_max_sum_l2070_207088

theorem smallest_max_sum (a b c d e f : ℕ+) 
  (sum_eq : a + b + c + d + e + f = 2512) : 
  (∃ (M : ℕ), M = max (a + b) (max (b + c) (max (c + d) (max (d + e) (e + f)))) ∧ 
   (∀ (M' : ℕ), M' = max (a + b) (max (b + c) (max (c + d) (max (d + e) (e + f)))) → M ≤ M') ∧
   M = 1005) := by
  sorry

end NUMINAMATH_CALUDE_smallest_max_sum_l2070_207088


namespace NUMINAMATH_CALUDE_difference_15x_x_squared_l2070_207012

theorem difference_15x_x_squared (x : ℕ) (h : x = 8) : 15 * x - x^2 = 56 := by
  sorry

end NUMINAMATH_CALUDE_difference_15x_x_squared_l2070_207012


namespace NUMINAMATH_CALUDE_black_white_difference_l2070_207083

/-- Represents a chessboard square color -/
inductive Color
| Black
| White

/-- Represents a chessboard -/
structure Chessboard :=
  (rows : Nat)
  (cols : Nat)
  (startColor : Color)

/-- Counts the number of squares of a given color on the chessboard -/
def countSquares (board : Chessboard) (color : Color) : Nat :=
  sorry

theorem black_white_difference (board : Chessboard) :
  board.rows = 7 ∧ board.cols = 9 ∧ board.startColor = Color.Black →
  countSquares board Color.Black = countSquares board Color.White + 1 := by
  sorry

end NUMINAMATH_CALUDE_black_white_difference_l2070_207083


namespace NUMINAMATH_CALUDE_joe_fruit_probability_l2070_207014

def num_fruits : ℕ := 4
def num_meals : ℕ := 3

theorem joe_fruit_probability :
  let p_same := (1 / num_fruits : ℚ) ^ num_meals * num_fruits
  1 - p_same = 15 / 16 := by sorry

end NUMINAMATH_CALUDE_joe_fruit_probability_l2070_207014


namespace NUMINAMATH_CALUDE_cube_edge_length_range_l2070_207057

theorem cube_edge_length_range (volume : ℝ) (h : volume = 100) :
  ∃ (edge : ℝ), edge ^ 3 = volume ∧ 4 < edge ∧ edge < 5 := by
  sorry

end NUMINAMATH_CALUDE_cube_edge_length_range_l2070_207057


namespace NUMINAMATH_CALUDE_diamond_equation_solution_l2070_207058

-- Define the diamond operation
def diamond (A B : ℝ) : ℝ := 4 * A - 3 * B + 7

-- State the theorem
theorem diamond_equation_solution :
  ∃! A : ℝ, diamond A 10 = 57 ∧ A = 20 := by sorry

end NUMINAMATH_CALUDE_diamond_equation_solution_l2070_207058


namespace NUMINAMATH_CALUDE_gcd_from_lcm_and_ratio_l2070_207067

theorem gcd_from_lcm_and_ratio (C D : ℕ+) 
  (h_lcm : Nat.lcm C D = 250)
  (h_ratio : C * 5 = D * 2) :
  Nat.gcd C D = 5 := by
sorry

end NUMINAMATH_CALUDE_gcd_from_lcm_and_ratio_l2070_207067
