import Mathlib

namespace NUMINAMATH_CALUDE_sqrt_3_simplest_l1108_110866

def is_simplest_sqrt (x : ℝ) (options : List ℝ) : Prop :=
  ∀ y ∈ options, (∃ z : ℝ, z ^ 2 = x) → (∃ w : ℝ, w ^ 2 = y) → x ≤ y

theorem sqrt_3_simplest : 
  is_simplest_sqrt 3 [0.1, 8, (abs a), 3] := by sorry

end NUMINAMATH_CALUDE_sqrt_3_simplest_l1108_110866


namespace NUMINAMATH_CALUDE_john_initial_money_l1108_110883

/-- Represents John's financial transactions and final balance --/
def john_money (initial spent allowance final : ℕ) : Prop :=
  initial - spent + allowance = final

/-- Proves that John's initial money was $5 --/
theorem john_initial_money : 
  ∃ (initial : ℕ), john_money initial 2 26 29 ∧ initial = 5 := by
  sorry

end NUMINAMATH_CALUDE_john_initial_money_l1108_110883


namespace NUMINAMATH_CALUDE_birds_remaining_count_l1108_110874

/-- The number of grey birds initially in the cage -/
def grey_birds : ℕ := 40

/-- The number of white birds next to the cage -/
def white_birds : ℕ := grey_birds + 6

/-- The number of grey birds remaining after half are freed -/
def remaining_grey_birds : ℕ := grey_birds / 2

/-- The total number of birds remaining after ten minutes -/
def total_remaining_birds : ℕ := remaining_grey_birds + white_birds

theorem birds_remaining_count : total_remaining_birds = 66 := by
  sorry

end NUMINAMATH_CALUDE_birds_remaining_count_l1108_110874


namespace NUMINAMATH_CALUDE_rain_on_monday_l1108_110890

theorem rain_on_monday (tuesday_rain : Real) (no_rain : Real) (both_rain : Real) 
  (h1 : tuesday_rain = 0.55)
  (h2 : no_rain = 0.35)
  (h3 : both_rain = 0.60) : 
  ∃ monday_rain : Real, monday_rain = 0.70 := by
  sorry

end NUMINAMATH_CALUDE_rain_on_monday_l1108_110890


namespace NUMINAMATH_CALUDE_cassidy_grounding_period_l1108_110850

/-- Calculates the total grounding period based on initial days, extra days per grade below B, and number of grades below B. -/
def total_grounding_period (initial_days : ℕ) (extra_days_per_grade : ℕ) (grades_below_b : ℕ) : ℕ :=
  initial_days + extra_days_per_grade * grades_below_b

/-- Proves that given the specified conditions, the total grounding period is 26 days. -/
theorem cassidy_grounding_period :
  total_grounding_period 14 3 4 = 26 := by
  sorry

end NUMINAMATH_CALUDE_cassidy_grounding_period_l1108_110850


namespace NUMINAMATH_CALUDE_min_value_of_complex_expression_l1108_110804

theorem min_value_of_complex_expression (z : ℂ) (h : Complex.abs z = 2) :
  ∃ (min_u : ℝ), min_u = (3/2) * Real.sqrt 3 ∧
  ∀ (u : ℝ), u = Complex.abs (z^2 - z + 1) → u ≥ min_u :=
sorry

end NUMINAMATH_CALUDE_min_value_of_complex_expression_l1108_110804


namespace NUMINAMATH_CALUDE_complex_equation_square_sum_l1108_110863

theorem complex_equation_square_sum (a b : ℝ) (i : ℂ) 
  (h1 : i * i = -1) 
  (h2 : (a - 2 * i) * i = b - i) : 
  a^2 + b^2 = 5 := by sorry

end NUMINAMATH_CALUDE_complex_equation_square_sum_l1108_110863


namespace NUMINAMATH_CALUDE_car_wash_earnings_l1108_110889

theorem car_wash_earnings (friday_earnings : ℕ) (x : ℚ) : 
  friday_earnings = 147 →
  friday_earnings + (friday_earnings * x + 7) + (friday_earnings + 78) = 673 →
  x = 2 := by
sorry

end NUMINAMATH_CALUDE_car_wash_earnings_l1108_110889


namespace NUMINAMATH_CALUDE_probability_x_plus_y_le_five_l1108_110842

/-- The probability of randomly selecting a point (x,y) from the rectangle [0,4] × [0,7] such that x + y ≤ 5 is equal to 5/14. -/
theorem probability_x_plus_y_le_five : 
  let total_area : ℝ := 4 * 7
  let favorable_area : ℝ := (1 / 2) * 5 * 4
  favorable_area / total_area = 5 / 14 := by sorry

end NUMINAMATH_CALUDE_probability_x_plus_y_le_five_l1108_110842


namespace NUMINAMATH_CALUDE_hot_dog_stand_mayo_bottles_l1108_110894

/-- Given a ratio of ketchup : mustard : mayo bottles and the number of ketchup bottles,
    calculate the number of mayo bottles -/
def mayo_bottles (ketchup_ratio mustard_ratio mayo_ratio ketchup_bottles : ℕ) : ℕ :=
  (mayo_ratio * ketchup_bottles) / ketchup_ratio

/-- Theorem: Given the ratio 3:3:2 for ketchup:mustard:mayo and 6 ketchup bottles,
    there are 4 mayo bottles -/
theorem hot_dog_stand_mayo_bottles :
  mayo_bottles 3 3 2 6 = 4 := by
  sorry

end NUMINAMATH_CALUDE_hot_dog_stand_mayo_bottles_l1108_110894


namespace NUMINAMATH_CALUDE_sqrt_twelve_minus_sqrt_three_equals_sqrt_three_l1108_110877

theorem sqrt_twelve_minus_sqrt_three_equals_sqrt_three :
  Real.sqrt 12 - Real.sqrt 3 = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_twelve_minus_sqrt_three_equals_sqrt_three_l1108_110877


namespace NUMINAMATH_CALUDE_long_division_problem_l1108_110857

theorem long_division_problem :
  let divisor : ℕ := 12
  let quotient : ℕ := 909809
  let dividend : ℕ := divisor * quotient
  dividend = 10917708 := by
sorry

end NUMINAMATH_CALUDE_long_division_problem_l1108_110857


namespace NUMINAMATH_CALUDE_larry_dog_time_l1108_110833

/-- The number of minutes Larry spends on his dog each day -/
def time_spent_on_dog (walking_playing_time : ℕ) (feeding_time : ℕ) : ℕ :=
  walking_playing_time * 2 + feeding_time

theorem larry_dog_time :
  let walking_playing_time : ℕ := 30 -- half an hour in minutes
  let feeding_time : ℕ := 12 -- a fifth of an hour in minutes
  time_spent_on_dog walking_playing_time feeding_time = 72 := by
sorry

end NUMINAMATH_CALUDE_larry_dog_time_l1108_110833


namespace NUMINAMATH_CALUDE_min_black_edges_four_black_edges_possible_l1108_110851

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

end NUMINAMATH_CALUDE_min_black_edges_four_black_edges_possible_l1108_110851


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l1108_110800

theorem cubic_equation_solution :
  ∀ x y z : ℤ, x^3 + y^3 + z^3 - 3*x*y*z = 2003 ↔
    ((x = 668 ∧ y = 668 ∧ z = 667) ∨
     (x = 668 ∧ y = 667 ∧ z = 668) ∨
     (x = 667 ∧ y = 668 ∧ z = 668)) :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l1108_110800


namespace NUMINAMATH_CALUDE_increasing_linear_function_not_in_fourth_quadrant_l1108_110878

/-- A linear function that passes through (-2, 0) and increases with x -/
structure IncreasingLinearFunction where
  k : ℝ
  b : ℝ
  k_neq_zero : k ≠ 0
  passes_through_neg_two_zero : 0 = -2 * k + b
  increasing : k > 0

/-- The fourth quadrant of the Cartesian plane -/
def fourth_quadrant : Set (ℝ × ℝ) :=
  {p | p.1 > 0 ∧ p.2 < 0}

/-- The graph of a linear function -/
def graph (f : IncreasingLinearFunction) : Set (ℝ × ℝ) :=
  {p | p.2 = f.k * p.1 + f.b}

theorem increasing_linear_function_not_in_fourth_quadrant (f : IncreasingLinearFunction) :
  graph f ∩ fourth_quadrant = ∅ :=
sorry

end NUMINAMATH_CALUDE_increasing_linear_function_not_in_fourth_quadrant_l1108_110878


namespace NUMINAMATH_CALUDE_equal_money_distribution_l1108_110840

/-- Represents the money distribution problem with Carmela and her cousins -/
def money_distribution (carmela_initial : ℕ) (cousin_initial : ℕ) (num_cousins : ℕ) (amount_given : ℕ) : Prop :=
  let total_money := carmela_initial + num_cousins * cousin_initial
  let people_count := num_cousins + 1
  let carmela_final := carmela_initial - num_cousins * amount_given
  let cousin_final := cousin_initial + amount_given
  (carmela_final = cousin_final) ∧ (total_money = people_count * carmela_final)

/-- Theorem stating that giving $1 to each cousin results in equal distribution -/
theorem equal_money_distribution :
  money_distribution 7 2 4 1 := by
  sorry

end NUMINAMATH_CALUDE_equal_money_distribution_l1108_110840


namespace NUMINAMATH_CALUDE_fraction_equality_l1108_110802

/-- Given two amounts a and b, prove that the fraction of b that equals 2/3 of a is 2/3 -/
theorem fraction_equality (a b : ℚ) (h1 : a + b = 1210) (h2 : b = 484) : 
  ∃ x : ℚ, x * b = 2/3 * a ∧ x = 2/3 := by
sorry

end NUMINAMATH_CALUDE_fraction_equality_l1108_110802


namespace NUMINAMATH_CALUDE_square_plus_one_geq_two_abs_l1108_110860

theorem square_plus_one_geq_two_abs (x : ℝ) : x^2 + 1 ≥ 2 * |x| := by
  sorry

end NUMINAMATH_CALUDE_square_plus_one_geq_two_abs_l1108_110860


namespace NUMINAMATH_CALUDE_salary_savings_percentage_l1108_110814

theorem salary_savings_percentage 
  (salary : ℝ) 
  (savings_after_increase : ℝ) 
  (expense_increase_percentage : ℝ) :
  salary = 5750 →
  savings_after_increase = 230 →
  expense_increase_percentage = 20 →
  ∃ (savings_percentage : ℝ),
    savings_percentage = 20 ∧
    savings_after_increase = salary - (1 + expense_increase_percentage / 100) * ((100 - savings_percentage) / 100 * salary) :=
by sorry

end NUMINAMATH_CALUDE_salary_savings_percentage_l1108_110814


namespace NUMINAMATH_CALUDE_x_neg_one_is_local_minimum_l1108_110880

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x

theorem x_neg_one_is_local_minimum :
  ∃ δ > 0, ∀ x : ℝ, x ≠ -1 → |x - (-1)| < δ → f x ≥ f (-1) :=
sorry

end NUMINAMATH_CALUDE_x_neg_one_is_local_minimum_l1108_110880


namespace NUMINAMATH_CALUDE_toothpicks_200th_stage_l1108_110824

def toothpicks (n : ℕ) : ℕ :=
  if n ≤ 49 then
    4 + 4 * (n - 1)
  else if n ≤ 99 then
    toothpicks 49 + 5 * (n - 49)
  else if n ≤ 149 then
    toothpicks 99 + 6 * (n - 99)
  else
    toothpicks 149 + 7 * (n - 149)

theorem toothpicks_200th_stage :
  toothpicks 200 = 1082 := by sorry

end NUMINAMATH_CALUDE_toothpicks_200th_stage_l1108_110824


namespace NUMINAMATH_CALUDE_ab_greater_ac_l1108_110843

theorem ab_greater_ac (a b c : ℝ) (h1 : c < b) (h2 : b < a) (h3 : a * c < 0) : a * b > a * c := by
  sorry

end NUMINAMATH_CALUDE_ab_greater_ac_l1108_110843


namespace NUMINAMATH_CALUDE_investment_difference_l1108_110847

def initial_investment : ℝ := 500

def jackson_growth_rate : ℝ := 4

def brandon_growth_rate : ℝ := 0.2

def jackson_final_value : ℝ := initial_investment * jackson_growth_rate

def brandon_final_value : ℝ := initial_investment * brandon_growth_rate

theorem investment_difference :
  jackson_final_value - brandon_final_value = 1900 := by sorry

end NUMINAMATH_CALUDE_investment_difference_l1108_110847


namespace NUMINAMATH_CALUDE_correct_costs_l1108_110819

/-- Represents the costs of a pen, pencil, and ink refill -/
structure ItemCosts where
  pen : ℚ
  pencil : ℚ
  ink_refill : ℚ

/-- Checks if the given costs satisfy the problem conditions -/
def satisfies_conditions (costs : ItemCosts) : Prop :=
  costs.pen + costs.pencil + costs.ink_refill = 2.4 ∧
  costs.pen = costs.ink_refill + 1.5 ∧
  costs.pencil = costs.ink_refill - 0.4

/-- Theorem stating the correct costs for the items -/
theorem correct_costs :
  ∃ (costs : ItemCosts),
    satisfies_conditions costs ∧
    costs.pen = 1.93 ∧
    costs.pencil = 0.03 ∧
    costs.ink_refill = 0.43 :=
by
  sorry

end NUMINAMATH_CALUDE_correct_costs_l1108_110819


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_when_m_3_A_subset_B_iff_m_in_range_l1108_110806

-- Define sets A and B
def A : Set ℝ := {x | x^2 - 3*x - 18 ≤ 0}
def B (m : ℝ) : Set ℝ := {x | m - 8 ≤ x ∧ x ≤ m + 4}

-- Statement 1
theorem complement_A_intersect_B_when_m_3 : 
  (Set.univ \ A) ∩ B 3 = {x | -5 ≤ x ∧ x < -3 ∨ 6 < x ∧ x ≤ 7} := by sorry

-- Statement 2
theorem A_subset_B_iff_m_in_range : 
  ∀ m, A ∩ B m = A ↔ 2 ≤ m ∧ m ≤ 5 := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_when_m_3_A_subset_B_iff_m_in_range_l1108_110806


namespace NUMINAMATH_CALUDE_gary_stickers_left_l1108_110808

/-- The number of stickers Gary had initially -/
def initial_stickers : ℕ := 99

/-- The number of stickers Gary gave to Lucy -/
def stickers_to_lucy : ℕ := 42

/-- The number of stickers Gary gave to Alex -/
def stickers_to_alex : ℕ := 26

/-- The number of stickers Gary had left after giving stickers to Lucy and Alex -/
def stickers_left : ℕ := initial_stickers - (stickers_to_lucy + stickers_to_alex)

theorem gary_stickers_left : stickers_left = 31 := by
  sorry

end NUMINAMATH_CALUDE_gary_stickers_left_l1108_110808


namespace NUMINAMATH_CALUDE_collinear_points_xy_value_l1108_110882

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Check if three points are collinear -/
def collinear (p q r : Point3D) : Prop :=
  ∃ t : ℝ, (r.x - p.x) = t * (q.x - p.x) ∧
            (r.y - p.y) = t * (q.y - p.y) ∧
            (r.z - p.z) = t * (q.z - p.z)

/-- The main theorem -/
theorem collinear_points_xy_value :
  ∀ (x y : ℝ),
  let A : Point3D := ⟨1, -2, 11⟩
  let B : Point3D := ⟨4, 2, 3⟩
  let C : Point3D := ⟨x, y, 15⟩
  collinear A B C → x * y = 2 := by
  sorry

end NUMINAMATH_CALUDE_collinear_points_xy_value_l1108_110882


namespace NUMINAMATH_CALUDE_smallest_k_for_multiple_of_180_k_1080_is_multiple_of_180_k_1080_is_smallest_l1108_110852

def sum_of_squares (k : ℕ) : ℕ := k * (k + 1) * (2 * k + 1) / 6

theorem smallest_k_for_multiple_of_180 :
  ∀ k : ℕ, k > 0 → sum_of_squares k % 180 = 0 → k ≥ 1080 :=
by sorry

theorem k_1080_is_multiple_of_180 :
  sum_of_squares 1080 % 180 = 0 :=
by sorry

theorem k_1080_is_smallest :
  ∀ k : ℕ, k > 0 → sum_of_squares k % 180 = 0 → k = 1080 :=
by sorry

end NUMINAMATH_CALUDE_smallest_k_for_multiple_of_180_k_1080_is_multiple_of_180_k_1080_is_smallest_l1108_110852


namespace NUMINAMATH_CALUDE_sum_of_valid_b_is_six_l1108_110820

/-- A function that checks if a number is a perfect square -/
def is_perfect_square (n : ℤ) : Prop := ∃ m : ℤ, n = m * m

/-- The sum of all positive integer values of b for which the quadratic equation 3x^2 + 7x + b = 0 has rational roots -/
def sum_of_valid_b : ℕ := sorry

/-- The main theorem stating that the sum of valid b values is 6 -/
theorem sum_of_valid_b_is_six : sum_of_valid_b = 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_valid_b_is_six_l1108_110820


namespace NUMINAMATH_CALUDE_c_value_is_198_l1108_110859

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the equation for c
def c_equation (a b : ℕ+) : ℂ := (a + b * i)^3 - 107 * i

-- State the theorem
theorem c_value_is_198 :
  ∀ a b c : ℕ+, c_equation a b = c → c = 198 := by
  sorry

end NUMINAMATH_CALUDE_c_value_is_198_l1108_110859


namespace NUMINAMATH_CALUDE_equation1_solutions_equation2_solutions_l1108_110810

-- Define the quadratic equations
def equation1 (x : ℝ) : Prop := x^2 - 4*x + 1 = 0
def equation2 (x : ℝ) : Prop := 2*x^2 - 3*x + 1 = 0

-- Theorem for equation 1
theorem equation1_solutions :
  ∃ x₁ x₂ : ℝ, x₁ = 2 + Real.sqrt 3 ∧ x₂ = 2 - Real.sqrt 3 ∧
  equation1 x₁ ∧ equation1 x₂ ∧
  ∀ x : ℝ, equation1 x → x = x₁ ∨ x = x₂ :=
sorry

-- Theorem for equation 2
theorem equation2_solutions :
  ∃ x₁ x₂ : ℝ, x₁ = 1 ∧ x₂ = 1/2 ∧
  equation2 x₁ ∧ equation2 x₂ ∧
  ∀ x : ℝ, equation2 x → x = x₁ ∨ x = x₂ :=
sorry

end NUMINAMATH_CALUDE_equation1_solutions_equation2_solutions_l1108_110810


namespace NUMINAMATH_CALUDE_first_number_is_1841_l1108_110831

/-- Represents one operation of replacing the first number with the average of the other two -/
def operation (x y z : ℤ) : ℤ × ℤ × ℤ := (y, z, (y + z) / 2)

/-- Applies the operation n times -/
def apply_operations (n : ℕ) (x y z : ℤ) : ℤ × ℤ × ℤ :=
  match n with
  | 0 => (x, y, z)
  | n + 1 => 
    let (a, b, c) := apply_operations n x y z
    operation a b c

theorem first_number_is_1841 (a b c : ℤ) : 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ -- all numbers are positive
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ -- all numbers are different
  a + b + c = 2013 ∧ -- initial sum
  (let (x, y, z) := apply_operations 7 a b c; x + y + z = 195) → -- sum after 7 operations
  a = 1841 := by sorry

end NUMINAMATH_CALUDE_first_number_is_1841_l1108_110831


namespace NUMINAMATH_CALUDE_wire_poles_problem_l1108_110835

theorem wire_poles_problem (wire_length : ℝ) (distance_increase : ℝ) : 
  wire_length = 5000 →
  distance_increase = 1.25 →
  ∃ (n : ℕ), 
    n > 1 ∧
    wire_length / (n - 1 : ℝ) + distance_increase = wire_length / (n - 2 : ℝ) ∧
    n = 65 := by
  sorry

end NUMINAMATH_CALUDE_wire_poles_problem_l1108_110835


namespace NUMINAMATH_CALUDE_total_blue_balloons_l1108_110886

theorem total_blue_balloons (joan_balloons melanie_balloons john_balloons : ℕ) 
  (h1 : joan_balloons = 40)
  (h2 : melanie_balloons = 41)
  (h3 : john_balloons = 55) :
  joan_balloons + melanie_balloons + john_balloons = 136 := by
sorry

end NUMINAMATH_CALUDE_total_blue_balloons_l1108_110886


namespace NUMINAMATH_CALUDE_sector_area_l1108_110895

/-- Given a circular sector with arc length 3π and central angle 135°, prove its area is 6π. -/
theorem sector_area (r : ℝ) (θ : ℝ) : 
  r * θ = 3 * Real.pi → 
  θ = 135 * Real.pi / 180 →
  (1 / 2) * r^2 * θ = 6 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_sector_area_l1108_110895


namespace NUMINAMATH_CALUDE_f_of_f_3_l1108_110888

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then x^2 + 1 else 2/x

-- Theorem statement
theorem f_of_f_3 : f (f 3) = 13/9 := by
  sorry

end NUMINAMATH_CALUDE_f_of_f_3_l1108_110888


namespace NUMINAMATH_CALUDE_toy_set_pricing_l1108_110816

/-- Represents the cost and sales data for Asian Games mascot plush toy sets -/
structure ToySetData where
  cost_price : ℝ
  batch1_quantity : ℕ
  batch1_price : ℝ
  batch2_quantity : ℕ
  batch2_price : ℝ
  total_profit : ℝ
  batch3_quantity : ℕ
  batch3_initial_price : ℝ
  day1_sales : ℕ
  day2_sales : ℕ
  sales_increase_per_reduction : ℝ
  reduction_step : ℝ
  day3_profit : ℝ

/-- Theorem stating the cost price and required price reduction for the toy sets -/
theorem toy_set_pricing (data : ToySetData) :
  data.cost_price = 60 ∧
  (∃ (price_reduction : ℝ), price_reduction = 10 ∧
    (data.batch1_quantity * (data.batch1_price - data.cost_price) +
     data.batch2_quantity * (data.batch2_price - data.cost_price) = data.total_profit) ∧
    (data.day1_sales * (data.batch3_initial_price - data.cost_price) +
     data.day2_sales * (data.batch3_initial_price - data.cost_price) +
     (data.day2_sales + price_reduction / data.reduction_step * data.sales_increase_per_reduction) *
       (data.batch3_initial_price - price_reduction - data.cost_price) = data.day3_profit)) :=
by sorry

end NUMINAMATH_CALUDE_toy_set_pricing_l1108_110816


namespace NUMINAMATH_CALUDE_emily_team_score_l1108_110856

theorem emily_team_score (total_players : ℕ) (emily_score : ℕ) (other_player_score : ℕ) : 
  total_players = 8 →
  emily_score = 23 →
  other_player_score = 2 →
  emily_score + (total_players - 1) * other_player_score = 37 := by
  sorry

end NUMINAMATH_CALUDE_emily_team_score_l1108_110856


namespace NUMINAMATH_CALUDE_distance_sum_is_18_l1108_110834

/-- Given three points A, B, and D in a plane, prove that the sum of distances AD and BD is 18 -/
theorem distance_sum_is_18 (A B D : ℝ × ℝ) : 
  A = (16, 0) → B = (1, 1) → D = (4, 5) → 
  Real.sqrt ((A.1 - D.1)^2 + (A.2 - D.2)^2) + Real.sqrt ((B.1 - D.1)^2 + (B.2 - D.2)^2) = 18 := by
  sorry

end NUMINAMATH_CALUDE_distance_sum_is_18_l1108_110834


namespace NUMINAMATH_CALUDE_donut_selections_l1108_110867

theorem donut_selections (n k : ℕ) (hn : n = 5) (hk : k = 4) : 
  Nat.choose (n + k - 1) (k - 1) = 56 := by
  sorry

end NUMINAMATH_CALUDE_donut_selections_l1108_110867


namespace NUMINAMATH_CALUDE_unique_solution_condition_l1108_110862

theorem unique_solution_condition (c d : ℝ) :
  (∃! x : ℝ, 4 * x - 7 + c = d * x + 2) ↔ d ≠ 4 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l1108_110862


namespace NUMINAMATH_CALUDE_pokemon_cards_distribution_l1108_110879

theorem pokemon_cards_distribution (total_cards : ℕ) (num_friends : ℕ) 
  (h1 : total_cards = 56) (h2 : num_friends = 4) :
  total_cards / num_friends = 14 := by
  sorry

end NUMINAMATH_CALUDE_pokemon_cards_distribution_l1108_110879


namespace NUMINAMATH_CALUDE_min_value_quadratic_expression_l1108_110855

theorem min_value_quadratic_expression :
  ∀ x y : ℝ, 3 * x^2 + 4 * x * y + 2 * y^2 - 6 * x + 4 * y + 5 ≥ -1 :=
by sorry

end NUMINAMATH_CALUDE_min_value_quadratic_expression_l1108_110855


namespace NUMINAMATH_CALUDE_car_profit_percentage_l1108_110873

/-- Calculates the profit percentage on the original price of a car
    given the discount percentage on purchase and markup percentage on sale. -/
theorem car_profit_percentage
  (P : ℝ)                    -- Original price of the car
  (discount : ℝ)             -- Discount percentage on purchase
  (markup : ℝ)               -- Markup percentage on sale
  (h_discount : discount = 20)
  (h_markup : markup = 45)
  : (((1 - discount / 100) * (1 + markup / 100) - 1) * 100 = 16) := by
  sorry

end NUMINAMATH_CALUDE_car_profit_percentage_l1108_110873


namespace NUMINAMATH_CALUDE_base_seven_digits_of_2401_l1108_110832

theorem base_seven_digits_of_2401 : ∃ n : ℕ, n > 0 ∧ 7^(n-1) ≤ 2401 ∧ 2401 < 7^n ∧ n = 5 := by
  sorry

end NUMINAMATH_CALUDE_base_seven_digits_of_2401_l1108_110832


namespace NUMINAMATH_CALUDE_product_of_four_consecutive_integers_divisible_by_24_l1108_110892

theorem product_of_four_consecutive_integers_divisible_by_24 (n : ℕ) :
  ∃ k : ℕ, n * (n + 1) * (n + 2) * (n + 3) = 24 * k ∧
  ∀ m : ℕ, m > 24 → ¬(∀ n : ℕ, ∃ k : ℕ, n * (n + 1) * (n + 2) * (n + 3) = m * k) :=
by sorry

end NUMINAMATH_CALUDE_product_of_four_consecutive_integers_divisible_by_24_l1108_110892


namespace NUMINAMATH_CALUDE_min_balls_for_target_color_l1108_110822

def red_balls : ℕ := 35
def green_balls : ℕ := 25
def yellow_balls : ℕ := 22
def blue_balls : ℕ := 15
def white_balls : ℕ := 12
def black_balls : ℕ := 11

def total_balls : ℕ := red_balls + green_balls + yellow_balls + blue_balls + white_balls + black_balls

def target_color_count : ℕ := 18

theorem min_balls_for_target_color :
  ∃ (n : ℕ), n = 89 ∧
  (∀ (m : ℕ), m < n → ∃ (r g y bl w bk : ℕ),
    r + g + y + bl + w + bk = m ∧
    r ≤ red_balls ∧ g ≤ green_balls ∧ y ≤ yellow_balls ∧
    bl ≤ blue_balls ∧ w ≤ white_balls ∧ bk ≤ black_balls ∧
    r < target_color_count ∧ g < target_color_count ∧ y < target_color_count ∧
    bl < target_color_count ∧ w < target_color_count ∧ bk < target_color_count) ∧
  (∀ (r g y bl w bk : ℕ),
    r + g + y + bl + w + bk = n →
    r ≤ red_balls → g ≤ green_balls → y ≤ yellow_balls →
    bl ≤ blue_balls → w ≤ white_balls → bk ≤ black_balls →
    r ≥ target_color_count ∨ g ≥ target_color_count ∨ y ≥ target_color_count ∨
    bl ≥ target_color_count ∨ w ≥ target_color_count ∨ bk ≥ target_color_count) :=
by sorry

#check min_balls_for_target_color

end NUMINAMATH_CALUDE_min_balls_for_target_color_l1108_110822


namespace NUMINAMATH_CALUDE_sum_of_fractions_l1108_110836

theorem sum_of_fractions : (1/2 : ℚ) + 2/4 + 4/8 + 8/16 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l1108_110836


namespace NUMINAMATH_CALUDE_difference_sum_of_T_l1108_110815

def T : Finset ℕ := Finset.range 11

def difference_sum (s : Finset ℕ) : ℕ :=
  s.sum (fun i => s.sum (fun j => if i > j then (3^i - 3^j) else 0))

theorem difference_sum_of_T : difference_sum T = 793168 := by
  sorry

end NUMINAMATH_CALUDE_difference_sum_of_T_l1108_110815


namespace NUMINAMATH_CALUDE_factorization_xy_minus_8y_l1108_110849

theorem factorization_xy_minus_8y (x y : ℝ) : x * y - 8 * y = y * (x - 8) := by
  sorry

end NUMINAMATH_CALUDE_factorization_xy_minus_8y_l1108_110849


namespace NUMINAMATH_CALUDE_evaluate_expression_l1108_110829

theorem evaluate_expression (x : ℝ) (h : x = -3) :
  (5 + x * (5 + x) - 5^2) / (x - 5 + x^2) = -26 := by
sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1108_110829


namespace NUMINAMATH_CALUDE_chocolate_fraction_is_11_24_l1108_110809

/-- The fraction of students who chose chocolate ice cream -/
def chocolate_fraction (chocolate strawberry vanilla : ℕ) : ℚ :=
  chocolate / (chocolate + strawberry + vanilla)

/-- Theorem stating that the fraction of students who chose chocolate ice cream is 11/24 -/
theorem chocolate_fraction_is_11_24 :
  chocolate_fraction 11 5 8 = 11 / 24 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_fraction_is_11_24_l1108_110809


namespace NUMINAMATH_CALUDE_solve_average_age_problem_l1108_110839

def average_age_problem (T : ℝ) (original_size : ℕ) (replaced_age : ℝ) (age_decrease : ℝ) : Prop :=
  let new_size : ℕ := original_size
  let new_average : ℝ := (T - replaced_age + (T / original_size - age_decrease)) / new_size
  (T / original_size) - age_decrease = new_average

theorem solve_average_age_problem :
  ∀ (T : ℝ) (original_size : ℕ) (replaced_age : ℝ) (age_decrease : ℝ),
  original_size = 20 →
  replaced_age = 60 →
  age_decrease = 4 →
  average_age_problem T original_size replaced_age age_decrease →
  (T / original_size - age_decrease) = 40 :=
sorry

end NUMINAMATH_CALUDE_solve_average_age_problem_l1108_110839


namespace NUMINAMATH_CALUDE_mukesh_travel_distance_l1108_110868

theorem mukesh_travel_distance : ∀ x : ℝ,
  (x / 90 - x / 120 = 4 / 15) →
  x = 96 := by
  sorry

end NUMINAMATH_CALUDE_mukesh_travel_distance_l1108_110868


namespace NUMINAMATH_CALUDE_fraction_meaningful_l1108_110848

theorem fraction_meaningful (x : ℝ) : 
  (∃ y : ℝ, y = 4 / (x - 3)) ↔ x ≠ 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_meaningful_l1108_110848


namespace NUMINAMATH_CALUDE_discount_comparison_l1108_110813

def original_price : ℝ := 15000

def apply_discount (price : ℝ) (discount : ℝ) : ℝ :=
  price * (1 - discount)

def scheme1_discounts : List ℝ := [0.25, 0.15, 0.10]
def scheme2_discounts : List ℝ := [0.30, 0.10, 0.05]

def apply_scheme (price : ℝ) (discounts : List ℝ) : ℝ :=
  discounts.foldl apply_discount price

theorem discount_comparison :
  apply_scheme original_price scheme1_discounts - apply_scheme original_price scheme2_discounts = 371.25 := by
  sorry

end NUMINAMATH_CALUDE_discount_comparison_l1108_110813


namespace NUMINAMATH_CALUDE_larger_number_proof_l1108_110864

theorem larger_number_proof (a b : ℕ) (h1 : a > 0) (h2 : b > 0) : 
  Nat.gcd a b = 84 → Nat.lcm a b = 21 → 4 * a = b → b = 84 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l1108_110864


namespace NUMINAMATH_CALUDE_coordinates_wrt_x_axis_l1108_110875

/-- Reflects a point across the x-axis in a 2D Cartesian coordinate system -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

/-- The coordinates of point P -/
def P : ℝ × ℝ := (-2, 3)

/-- Theorem: The coordinates of P(-2, 3) with respect to the x-axis are (-2, -3) -/
theorem coordinates_wrt_x_axis : reflect_x P = (-2, -3) := by
  sorry

end NUMINAMATH_CALUDE_coordinates_wrt_x_axis_l1108_110875


namespace NUMINAMATH_CALUDE_probability_is_two_over_155_l1108_110844

/-- Represents a 5x5x5 cube with two adjacent faces painted red -/
structure PaintedCube :=
  (size : Nat)
  (painted_faces : Nat)

/-- Calculates the number of unit cubes with exactly three painted faces -/
def count_three_painted_faces (cube : PaintedCube) : Nat :=
  1

/-- Calculates the number of unit cubes with no painted faces -/
def count_unpainted_faces (cube : PaintedCube) : Nat :=
  cube.size^3 - (cube.size^2 * 2 - cube.size)

/-- Calculates the total number of ways to choose two unit cubes -/
def total_combinations (cube : PaintedCube) : Nat :=
  (cube.size^3 * (cube.size^3 - 1)) / 2

/-- Calculates the probability of selecting one cube with three painted faces
    and one cube with no painted faces -/
def probability_three_and_none (cube : PaintedCube) : Rat :=
  (count_three_painted_faces cube * count_unpainted_faces cube) / total_combinations cube

/-- The main theorem stating the probability is 2/155 -/
theorem probability_is_two_over_155 (cube : PaintedCube) 
  (h1 : cube.size = 5) 
  (h2 : cube.painted_faces = 2) :
  probability_three_and_none cube = 2 / 155 := by
  sorry

end NUMINAMATH_CALUDE_probability_is_two_over_155_l1108_110844


namespace NUMINAMATH_CALUDE_consumption_ranking_l1108_110821

-- Define the regions
inductive Region
| West
| NonWest
| Russia

-- Define the consumption function
def consumption : Region → ℝ
| Region.West => 21428
| Region.NonWest => 26848.55
| Region.Russia => 302790.13

-- Define the ranking function
def ranking (r : Region) : ℕ :=
  match r with
  | Region.West => 3
  | Region.NonWest => 2
  | Region.Russia => 1

-- Theorem statement
theorem consumption_ranking :
  ∀ r1 r2 : Region, ranking r1 < ranking r2 ↔ consumption r1 > consumption r2 :=
by sorry

end NUMINAMATH_CALUDE_consumption_ranking_l1108_110821


namespace NUMINAMATH_CALUDE_largest_difference_theorem_l1108_110803

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

def digit_constraints (a b c d e f g h i : ℕ) : Prop :=
  a ∈ ({3, 5, 9} : Set ℕ) ∧
  b ∈ ({2, 3, 7} : Set ℕ) ∧
  c ∈ ({3, 4, 8, 9} : Set ℕ) ∧
  d ∈ ({2, 3, 7} : Set ℕ) ∧
  e ∈ ({3, 5, 9} : Set ℕ) ∧
  f ∈ ({1, 4, 7} : Set ℕ) ∧
  g ∈ ({4, 5, 9} : Set ℕ) ∧
  h = 2 ∧
  i ∈ ({4, 5, 9} : Set ℕ)

def number_from_digits (a b c : ℕ) : ℕ := 100 * a + 10 * b + c

theorem largest_difference_theorem (a b c d e f g h i : ℕ) :
  digit_constraints a b c d e f g h i →
  is_three_digit (number_from_digits a b c) →
  is_three_digit (number_from_digits d e f) →
  is_three_digit (number_from_digits g h i) →
  number_from_digits a b c - number_from_digits d e f = number_from_digits g h i →
  ∀ (x y z u v w : ℕ),
    digit_constraints x y z u v w g h i →
    is_three_digit (number_from_digits x y z) →
    is_three_digit (number_from_digits u v w) →
    number_from_digits x y z - number_from_digits u v w = number_from_digits g h i →
    number_from_digits g h i ≤ 529 →
  (a = 9 ∧ b = 2 ∧ c = 3 ∧ d = 3 ∧ e = 9 ∧ f = 4) :=
sorry

end NUMINAMATH_CALUDE_largest_difference_theorem_l1108_110803


namespace NUMINAMATH_CALUDE_range_of_x_range_of_m_l1108_110871

-- Problem 1
theorem range_of_x (x : ℝ) : (4*x - 3)^2 ≤ 1 → 1/2 ≤ x ∧ x ≤ 1 := by sorry

-- Problem 2
theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, 4*x + m < 0 → x^2 - x - 2 > 0) → m ≥ 4 := by sorry

end NUMINAMATH_CALUDE_range_of_x_range_of_m_l1108_110871


namespace NUMINAMATH_CALUDE_triangle_squares_area_sum_l1108_110830

/-- Given a right triangle EAB with BE = 12 and another right triangle EAH with AH = 5,
    the sum of the areas of squares ABCD, AEFG, and AHIJ is equal to 169 square units. -/
theorem triangle_squares_area_sum : 
  ∀ (A B C D E F G H I J : ℝ × ℝ),
  let ab := dist A B
  let ae := dist A E
  let ah := dist A H
  let be := dist B E
  -- Angle EAB is a right angle
  (ab ^ 2 + ae ^ 2 = be ^ 2) →
  -- BE = 12 units
  (be = 12) →
  -- Triangle EAH is a right triangle
  (ae ^ 2 + ah ^ 2 = (dist E H) ^ 2) →
  -- AH = 5 units
  (ah = 5) →
  -- The sum of the areas of squares ABCD, AEFG, and AHIJ is 169
  (ab ^ 2 + ae ^ 2 + (dist E H) ^ 2 = 169) := by
  sorry


end NUMINAMATH_CALUDE_triangle_squares_area_sum_l1108_110830


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l1108_110845

theorem algebraic_expression_value (a : ℝ) (h : a^2 + a - 1 = 0) :
  2*a^2 + 2*a + 2021 = 2023 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l1108_110845


namespace NUMINAMATH_CALUDE_rectangle_circle_square_area_l1108_110865

theorem rectangle_circle_square_area : 
  ∀ (r l b : ℝ), 
    l = (2/5) * r → 
    b = 10 → 
    l * b = 220 → 
    r^2 = 3025 :=
by
  sorry

end NUMINAMATH_CALUDE_rectangle_circle_square_area_l1108_110865


namespace NUMINAMATH_CALUDE_length_AE_l1108_110807

-- Define the circle
def Circle := {c : ℝ × ℝ | c.1^2 + c.2^2 = 4}

-- Define points A, B, C, D, E
variable (A B C D E : ℝ × ℝ)

-- AB is a diameter of the circle
axiom diam : A ∈ Circle ∧ B ∈ Circle ∧ (A.1 - B.1)^2 + (A.2 - B.2)^2 = 16

-- ABC is an equilateral triangle
axiom equilateral : (A.1 - B.1)^2 + (A.2 - B.2)^2 = (B.1 - C.1)^2 + (B.2 - C.2)^2 ∧
                    (B.1 - C.1)^2 + (B.2 - C.2)^2 = (C.1 - A.1)^2 + (C.2 - A.2)^2

-- D is the intersection of the circle and AC
axiom D_on_circle : D ∈ Circle
axiom D_on_AC : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ D = (t * A.1 + (1 - t) * C.1, t * A.2 + (1 - t) * C.2)

-- E is the intersection of the circle and BC
axiom E_on_circle : E ∈ Circle
axiom E_on_BC : ∃ s : ℝ, 0 ≤ s ∧ s ≤ 1 ∧ E = (s * B.1 + (1 - s) * C.1, s * B.2 + (1 - s) * C.2)

-- Theorem: The length of AE is 2√3
theorem length_AE : (A.1 - E.1)^2 + (A.2 - E.2)^2 = 12 := by sorry

end NUMINAMATH_CALUDE_length_AE_l1108_110807


namespace NUMINAMATH_CALUDE_miss_adamson_paper_usage_l1108_110818

/-- Calculates the total number of sheets of paper used by a teacher for all students --/
def total_sheets_of_paper (num_classes : ℕ) (students_per_class : ℕ) (sheets_per_student : ℕ) : ℕ :=
  num_classes * students_per_class * sheets_per_student

/-- Proves that Miss Adamson will use 400 sheets of paper for all her students --/
theorem miss_adamson_paper_usage :
  total_sheets_of_paper 4 20 5 = 400 := by
  sorry

#eval total_sheets_of_paper 4 20 5

end NUMINAMATH_CALUDE_miss_adamson_paper_usage_l1108_110818


namespace NUMINAMATH_CALUDE_factor_calculation_l1108_110884

theorem factor_calculation (initial_number : ℕ) (factor : ℚ) : 
  initial_number = 8 →
  factor * (2 * initial_number + 9) = 75 →
  factor = 3 := by
sorry

end NUMINAMATH_CALUDE_factor_calculation_l1108_110884


namespace NUMINAMATH_CALUDE_log_equation_solution_l1108_110841

theorem log_equation_solution :
  ∀ x : ℝ, x > 0 → (Real.log x / Real.log 2 = -1/2 ↔ x = Real.sqrt 2 / 2) := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l1108_110841


namespace NUMINAMATH_CALUDE_matrix_multiplication_example_l1108_110870

theorem matrix_multiplication_example :
  let A : Matrix (Fin 2) (Fin 2) ℤ := !![3, 1; 4, -2]
  let B : Matrix (Fin 2) (Fin 2) ℤ := !![5, -3; 2, 2]
  A * B = !![17, -7; 16, -16] := by
  sorry

end NUMINAMATH_CALUDE_matrix_multiplication_example_l1108_110870


namespace NUMINAMATH_CALUDE_division_problem_additional_condition_l1108_110801

theorem division_problem (x : ℝ) : 2994 / x = 175 → x = 17.1 := by
  sorry

-- Additional theorem to include the unused condition
theorem additional_condition : 29.94 / 1.45 = 17.5 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_additional_condition_l1108_110801


namespace NUMINAMATH_CALUDE_triangle_area_l1108_110854

/-- In triangle ABC, prove that given specific side lengths and an angle relation, 
    the area of the triangle is √3/2. -/
theorem triangle_area (A B C : ℝ) (a b c : ℝ) :
  b = 1 →
  c = 2 →
  (2 * c - b) * Real.cos A = a * Real.cos B →
  (1/2) * b * c * Real.sin A = Real.sqrt 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_l1108_110854


namespace NUMINAMATH_CALUDE_binomial_expected_value_l1108_110869

/-- A random variable following a binomial distribution -/
structure BinomialRV where
  n : ℕ    -- number of trials
  p : ℝ    -- probability of success
  h1 : 0 ≤ p ∧ p ≤ 1  -- probability is between 0 and 1

/-- Expected value of a binomial random variable -/
def expected_value (ξ : BinomialRV) : ℝ := ξ.n * ξ.p

/-- Theorem: The expected value of ξ ~ B(6, 1/3) is 2 -/
theorem binomial_expected_value :
  ∀ ξ : BinomialRV, ξ.n = 6 ∧ ξ.p = 1/3 → expected_value ξ = 2 :=
by sorry

end NUMINAMATH_CALUDE_binomial_expected_value_l1108_110869


namespace NUMINAMATH_CALUDE_inequality_not_always_hold_l1108_110817

theorem inequality_not_always_hold (a b : ℝ) (h : a > b ∧ b > 0) : 
  ¬ ∀ c : ℝ, a * c > b * c :=
sorry

end NUMINAMATH_CALUDE_inequality_not_always_hold_l1108_110817


namespace NUMINAMATH_CALUDE_u_2023_equals_3_l1108_110893

-- Define the function g
def g : ℕ → ℕ
| 1 => 5
| 2 => 3
| 3 => 1
| 4 => 2
| 5 => 4
| _ => 0  -- For completeness, though not used in the problem

-- Define the sequence u
def u : ℕ → ℕ
| 0 => 5
| (n + 1) => g (u n)

-- Theorem statement
theorem u_2023_equals_3 : u 2023 = 3 := by
  sorry

end NUMINAMATH_CALUDE_u_2023_equals_3_l1108_110893


namespace NUMINAMATH_CALUDE_parallelogram_area_l1108_110861

/-- The area of a parallelogram with given base, slant height, and angle --/
theorem parallelogram_area (base slant_height : ℝ) (angle : ℝ) : 
  base = 24 → slant_height = 26 → angle = 40 * π / 180 →
  abs (base * (slant_height * Real.cos angle) - 478) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_l1108_110861


namespace NUMINAMATH_CALUDE_sum_1423_9_and_711_9_in_base3_l1108_110826

/-- Converts a number from base 9 to base 10 -/
def base9To10 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 10 to base 3 -/
def base10To3 (n : ℕ) : ℕ := sorry

/-- The sum of 1423 in base 9 and 711 in base 9, converted to base 3 -/
def sumInBase3 : ℕ := base10To3 (base9To10 1423 + base9To10 711)

theorem sum_1423_9_and_711_9_in_base3 :
  sumInBase3 = 2001011 := by sorry

end NUMINAMATH_CALUDE_sum_1423_9_and_711_9_in_base3_l1108_110826


namespace NUMINAMATH_CALUDE_least_common_period_is_36_l1108_110837

/-- A function satisfying the given functional equation -/
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x + 6) + f (x - 6) = f x

/-- The period of a function -/
def IsPeriod (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x : ℝ, f (x + p) = f x

/-- The least common positive period for all functions satisfying the equation -/
def LeastCommonPeriod (p : ℝ) : Prop :=
  p > 0 ∧
  (∀ f : ℝ → ℝ, SatisfiesEquation f → IsPeriod f p) ∧
  (∀ q : ℝ, q > 0 → (∀ f : ℝ → ℝ, SatisfiesEquation f → IsPeriod f q) → p ≤ q)

theorem least_common_period_is_36 :
  LeastCommonPeriod 36 := by sorry

end NUMINAMATH_CALUDE_least_common_period_is_36_l1108_110837


namespace NUMINAMATH_CALUDE_quarters_percentage_is_65_22_l1108_110805

/-- The number of dimes -/
def num_dimes : ℕ := 40

/-- The number of quarters -/
def num_quarters : ℕ := 30

/-- The value of a dime in cents -/
def dime_value : ℕ := 10

/-- The value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- The total value of all coins in cents -/
def total_value : ℕ := num_dimes * dime_value + num_quarters * quarter_value

/-- The value of all quarters in cents -/
def quarters_value : ℕ := num_quarters * quarter_value

/-- The percentage of the total value that is in quarters -/
def quarters_percentage : ℚ := (quarters_value : ℚ) / (total_value : ℚ) * 100

theorem quarters_percentage_is_65_22 : 
  ∀ ε > 0, |quarters_percentage - 65.22| < ε :=
sorry

end NUMINAMATH_CALUDE_quarters_percentage_is_65_22_l1108_110805


namespace NUMINAMATH_CALUDE_product_of_two_numbers_l1108_110812

theorem product_of_two_numbers (x y : ℝ) 
  (h1 : x^2 + y^2 = 289) 
  (h2 : x + y = 23) : 
  x * y = 120 := by
sorry

end NUMINAMATH_CALUDE_product_of_two_numbers_l1108_110812


namespace NUMINAMATH_CALUDE_interior_exterior_angle_ratio_octagon_l1108_110876

/-- The ratio of an interior angle to an exterior angle in a regular octagon is 3:1 -/
theorem interior_exterior_angle_ratio_octagon : 
  ∀ (interior_angle exterior_angle : ℝ),
  interior_angle > 0 → 
  exterior_angle > 0 →
  (∀ (n : ℕ), n = 8 → interior_angle = (n - 2) * 180 / n) →
  (∀ (n : ℕ), n = 8 → exterior_angle = 360 / n) →
  interior_angle / exterior_angle = 3 := by
sorry

end NUMINAMATH_CALUDE_interior_exterior_angle_ratio_octagon_l1108_110876


namespace NUMINAMATH_CALUDE_quadratic_inequality_condition_l1108_110898

theorem quadratic_inequality_condition (a : ℝ) : 
  (∀ x : ℝ, x^2 - 2*a*x + a > 0) → (0 < a ∧ a ≤ 1) ∧ 
  ¬(0 < a ∧ a ≤ 1 → ∀ x : ℝ, x^2 - 2*a*x + a > 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_condition_l1108_110898


namespace NUMINAMATH_CALUDE_b_paisa_per_a_rupee_l1108_110881

-- Define the total sum of money in rupees
def total_sum : ℚ := 164

-- Define C's share in rupees
def c_share : ℚ := 32

-- Define the ratio of C's paisa to A's rupees
def c_to_a_ratio : ℚ := 40 / 100

-- Define A's share in rupees
def a_share : ℚ := c_share / c_to_a_ratio

-- Define B's share in paisa
def b_share : ℚ := (total_sum - a_share - c_share) * 100

-- Theorem to prove
theorem b_paisa_per_a_rupee : b_share / a_share = 65 := by
  sorry

end NUMINAMATH_CALUDE_b_paisa_per_a_rupee_l1108_110881


namespace NUMINAMATH_CALUDE_four_number_inequality_equality_condition_l1108_110858

theorem four_number_inequality (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  ((a - b) * (a - c)) / (a + b + c) +
  ((b - c) * (b - d)) / (b + c + d) +
  ((c - d) * (c - a)) / (c + d + a) +
  ((d - a) * (d - b)) / (d + a + b) ≥ 0 :=
by sorry

theorem equality_condition (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  ((a - b) * (a - c)) / (a + b + c) +
  ((b - c) * (b - d)) / (b + c + d) +
  ((c - d) * (c - a)) / (c + d + a) +
  ((d - a) * (d - b)) / (d + a + b) = 0 ↔
  a = b ∧ b = c ∧ c = d :=
by sorry

end NUMINAMATH_CALUDE_four_number_inequality_equality_condition_l1108_110858


namespace NUMINAMATH_CALUDE_speaker_discount_savings_l1108_110811

/-- Calculates the savings from a discount given the initial price and discounted price. -/
def savings (initial_price discounted_price : ℝ) : ℝ :=
  initial_price - discounted_price

/-- Theorem stating that the savings from a discount on speakers priced at $475.00 and sold for $199.00 is equal to $276.00. -/
theorem speaker_discount_savings :
  savings 475 199 = 276 := by
  sorry

end NUMINAMATH_CALUDE_speaker_discount_savings_l1108_110811


namespace NUMINAMATH_CALUDE_soda_crates_count_l1108_110899

def bridge_weight_limit : ℕ := 20000
def empty_truck_weight : ℕ := 12000
def soda_crate_weight : ℕ := 50
def num_dryers : ℕ := 3
def dryer_weight : ℕ := 3000
def loaded_truck_weight : ℕ := 24000

def calculate_soda_crates (bridge_weight_limit empty_truck_weight soda_crate_weight 
                           num_dryers dryer_weight loaded_truck_weight : ℕ) : ℕ := 
  let total_dryer_weight := num_dryers * dryer_weight
  let remaining_weight := loaded_truck_weight - empty_truck_weight - total_dryer_weight
  let soda_weight := remaining_weight / 3
  soda_weight / soda_crate_weight

theorem soda_crates_count : 
  calculate_soda_crates bridge_weight_limit empty_truck_weight soda_crate_weight 
                         num_dryers dryer_weight loaded_truck_weight = 20 := by
  sorry

end NUMINAMATH_CALUDE_soda_crates_count_l1108_110899


namespace NUMINAMATH_CALUDE_decreasing_interval_of_f_l1108_110896

noncomputable def f (x : ℝ) : ℝ := Real.sin x - Real.sqrt 3 * Real.cos x

def monotonically_decreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f y ≤ f x

theorem decreasing_interval_of_f :
  ∃ (a b : ℝ), a = 5 * Real.pi / 6 ∧ b = Real.pi ∧
  monotonically_decreasing f a b ∧
  ∀ c d, 0 ≤ c ∧ d ≤ Real.pi ∧ c < d ∧ monotonically_decreasing f c d →
    a ≤ c ∧ d ≤ b :=
by sorry

end NUMINAMATH_CALUDE_decreasing_interval_of_f_l1108_110896


namespace NUMINAMATH_CALUDE_positive_integers_satisfying_condition_l1108_110823

theorem positive_integers_satisfying_condition :
  ∀ n : ℕ+, (25 - 3 * n.val ≥ 4) ↔ n.val ∈ ({1, 2, 3, 4, 5, 6, 7} : Set ℕ) := by
  sorry

end NUMINAMATH_CALUDE_positive_integers_satisfying_condition_l1108_110823


namespace NUMINAMATH_CALUDE_smallest_result_l1108_110891

def S : Finset ℕ := {2, 5, 8, 11, 14}

def process (a b c : ℕ) : ℕ := (a + b) * c

theorem smallest_result :
  ∃ (a b c : ℕ), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  process a b c = 26 ∧
  ∀ (x y z : ℕ), x ∈ S → y ∈ S → z ∈ S → x ≠ y → y ≠ z → x ≠ z →
  process x y z ≥ 26 :=
by sorry

end NUMINAMATH_CALUDE_smallest_result_l1108_110891


namespace NUMINAMATH_CALUDE_flight_750_male_first_class_fraction_l1108_110825

theorem flight_750_male_first_class_fraction 
  (total_passengers : ℕ) 
  (female_percentage : ℚ) 
  (first_class_percentage : ℚ) 
  (female_coach : ℕ) :
  total_passengers = 120 →
  female_percentage = 45/100 →
  first_class_percentage = 10/100 →
  female_coach = 46 →
  (total_passengers * first_class_percentage * (1 - female_percentage / (1 - first_class_percentage)) / 
   (total_passengers * first_class_percentage) : ℚ) = 1/3 := by
sorry

end NUMINAMATH_CALUDE_flight_750_male_first_class_fraction_l1108_110825


namespace NUMINAMATH_CALUDE_sector_area_for_unit_radian_l1108_110887

/-- Given a circle where the arc length corresponding to a central angle of 1 radian is 2,
    prove that the area of the sector corresponding to this central angle is 2. -/
theorem sector_area_for_unit_radian (r : ℝ) (l : ℝ) (α : ℝ) : 
  α = 1 → l = 2 → α = l / r → (1 / 2) * r * l = 2 := by
  sorry

end NUMINAMATH_CALUDE_sector_area_for_unit_radian_l1108_110887


namespace NUMINAMATH_CALUDE_sports_club_overlapping_members_l1108_110897

theorem sports_club_overlapping_members 
  (total_members : ℕ) 
  (badminton_players : ℕ) 
  (tennis_players : ℕ) 
  (neither_players : ℕ) 
  (h1 : total_members = 30)
  (h2 : badminton_players = 17)
  (h3 : tennis_players = 21)
  (h4 : neither_players = 2) :
  badminton_players + tennis_players - total_members + neither_players = 10 := by
  sorry

end NUMINAMATH_CALUDE_sports_club_overlapping_members_l1108_110897


namespace NUMINAMATH_CALUDE_b_5_times_b_9_equals_16_l1108_110872

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- A geometric sequence -/
def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = b n * r

theorem b_5_times_b_9_equals_16 
  (a : ℕ → ℝ) 
  (b : ℕ → ℝ) 
  (h1 : arithmetic_sequence a)
  (h2 : 2 * a 2 - (a 7)^2 + 2 * a 12 = 0)
  (h3 : geometric_sequence b)
  (h4 : b 7 = a 7) :
  b 5 * b 9 = 16 := by
sorry

end NUMINAMATH_CALUDE_b_5_times_b_9_equals_16_l1108_110872


namespace NUMINAMATH_CALUDE_back_seat_tickets_sold_l1108_110827

/-- Proves the number of back seat tickets sold at a concert --/
theorem back_seat_tickets_sold (total_seats : ℕ) (main_price back_price : ℕ) (total_revenue : ℕ) :
  total_seats = 20000 →
  main_price = 55 →
  back_price = 45 →
  total_revenue = 955000 →
  ∃ (main_seats back_seats : ℕ),
    main_seats + back_seats = total_seats ∧
    main_price * main_seats + back_price * back_seats = total_revenue ∧
    back_seats = 14500 := by
  sorry

#check back_seat_tickets_sold

end NUMINAMATH_CALUDE_back_seat_tickets_sold_l1108_110827


namespace NUMINAMATH_CALUDE_slant_height_neq_base_side_l1108_110838

/-- Represents a regular hexagonal pyramid --/
structure RegularHexagonalPyramid where
  r : ℝ  -- side length of each equilateral triangle in the base
  h : ℝ  -- height of the pyramid
  l : ℝ  -- slant height (lateral edge) of the pyramid
  r_pos : r > 0
  h_pos : h > 0
  l_pos : l > 0
  pythagorean : h^2 + r^2 = l^2

/-- Theorem: In a regular hexagonal pyramid, the slant height cannot be equal to the side length of the base hexagon --/
theorem slant_height_neq_base_side (p : RegularHexagonalPyramid) : p.l ≠ p.r := by
  sorry


end NUMINAMATH_CALUDE_slant_height_neq_base_side_l1108_110838


namespace NUMINAMATH_CALUDE_coin_difference_l1108_110828

/-- Represents the denominations of coins available -/
inductive Coin
  | fiveCent
  | twentyCent
  | fiftyCent

/-- The value of each coin in cents -/
def coinValue : Coin → Nat
  | Coin.fiveCent => 5
  | Coin.twentyCent => 20
  | Coin.fiftyCent => 50

/-- The amount to be paid in cents -/
def amountToPay : Nat := 50

/-- A function that calculates the minimum number of coins needed -/
def minCoins : Nat := sorry

/-- A function that calculates the maximum number of coins needed -/
def maxCoins : Nat := sorry

/-- Theorem stating the difference between max and min number of coins -/
theorem coin_difference : maxCoins - minCoins = 9 := by sorry

end NUMINAMATH_CALUDE_coin_difference_l1108_110828


namespace NUMINAMATH_CALUDE_commuting_matrices_ratio_l1108_110846

/-- Given two 2x2 matrices A and B that commute, prove that (2a - 3d) / (4b - 3c) = -3 --/
theorem commuting_matrices_ratio (a b c d : ℝ) : 
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![2, 3; 4, 5]
  let B : Matrix (Fin 2) (Fin 2) ℝ := !![a, b; c, d]
  (A * B = B * A) → (4 * b ≠ 3 * c) → ((2 * a - 3 * d) / (4 * b - 3 * c) = -3) := by
  sorry


end NUMINAMATH_CALUDE_commuting_matrices_ratio_l1108_110846


namespace NUMINAMATH_CALUDE_cubic_root_sum_squares_l1108_110885

theorem cubic_root_sum_squares (p q r : ℝ) : 
  (p^3 - 18*p^2 + 40*p - 15 = 0) →
  (q^3 - 18*q^2 + 40*q - 15 = 0) →
  (r^3 - 18*r^2 + 40*r - 15 = 0) →
  (p + q + r = 18) →
  (p*q + q*r + r*p = 40) →
  (p+q)^2 + (q+r)^2 + (r+p)^2 = 568 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_squares_l1108_110885


namespace NUMINAMATH_CALUDE_integral_equals_ten_implies_k_equals_one_l1108_110853

theorem integral_equals_ten_implies_k_equals_one :
  (∫ x in (0:ℝ)..2, (3 * x^2 + k)) = 10 → k = 1 := by
  sorry

end NUMINAMATH_CALUDE_integral_equals_ten_implies_k_equals_one_l1108_110853
