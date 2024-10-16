import Mathlib

namespace NUMINAMATH_CALUDE_division_of_terms_l500_50024

theorem division_of_terms (a b : ℝ) (h : b ≠ 0) : 3 * a^2 * b / b = 3 * a^2 := by
  sorry

end NUMINAMATH_CALUDE_division_of_terms_l500_50024


namespace NUMINAMATH_CALUDE_prob_even_diagonals_eq_one_over_101_l500_50091

/-- Represents a 3x3 grid filled with numbers 1 to 9 --/
def Grid := Fin 9 → Fin 9

/-- Checks if a given grid has even sums on both diagonals --/
def has_even_diagonal_sums (g : Grid) : Prop :=
  (g 0 + g 4 + g 8) % 2 = 0 ∧ (g 2 + g 4 + g 6) % 2 = 0

/-- The set of all valid grids --/
def all_grids : Finset Grid :=
  sorry

/-- The set of grids with even diagonal sums --/
def even_sum_grids : Finset Grid :=
  sorry

/-- The probability of having even sums on both diagonals --/
def prob_even_diagonals : ℚ :=
  (Finset.card even_sum_grids : ℚ) / (Finset.card all_grids : ℚ)

theorem prob_even_diagonals_eq_one_over_101 : 
  prob_even_diagonals = 1 / 101 :=
sorry

end NUMINAMATH_CALUDE_prob_even_diagonals_eq_one_over_101_l500_50091


namespace NUMINAMATH_CALUDE_small_boxes_count_l500_50058

theorem small_boxes_count (total_chocolates : ℕ) (chocolates_per_box : ℕ) 
  (h1 : total_chocolates = 504) 
  (h2 : chocolates_per_box = 28) : 
  total_chocolates / chocolates_per_box = 18 := by
  sorry

#check small_boxes_count

end NUMINAMATH_CALUDE_small_boxes_count_l500_50058


namespace NUMINAMATH_CALUDE_series_convergence_power_l500_50095

theorem series_convergence_power (a : ℕ → ℝ) 
  (h_pos : ∀ n, a n > 0) 
  (h_conv : Summable a) :
  Summable (fun n => (a n) ^ (n / (n + 1))) := by
sorry

end NUMINAMATH_CALUDE_series_convergence_power_l500_50095


namespace NUMINAMATH_CALUDE_intersection_M_N_l500_50053

def M : Set ℝ := {x | x^2 > 1}
def N : Set ℝ := {-2, -1, 0, 1, 2}

theorem intersection_M_N : M ∩ N = {-2, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l500_50053


namespace NUMINAMATH_CALUDE_apple_distribution_l500_50020

theorem apple_distribution (total_apples : ℕ) (apples_per_student : ℕ) : 
  total_apples = 120 →
  apples_per_student = 2 →
  (∃ (num_students : ℕ), 
    num_students * apples_per_student = total_apples - 1 ∧
    num_students > 0) →
  ∃ (num_students : ℕ), num_students = 59 := by
sorry

end NUMINAMATH_CALUDE_apple_distribution_l500_50020


namespace NUMINAMATH_CALUDE_matthew_crackers_l500_50021

theorem matthew_crackers (total_crackers : ℕ) (crackers_per_friend : ℕ) (num_friends : ℕ) :
  total_crackers = 8 →
  crackers_per_friend = 2 →
  total_crackers = num_friends * crackers_per_friend →
  num_friends = 4 := by
sorry

end NUMINAMATH_CALUDE_matthew_crackers_l500_50021


namespace NUMINAMATH_CALUDE_complex_sum_squared_l500_50025

noncomputable def i : ℂ := Complex.I

theorem complex_sum_squared 
  (a b c : ℂ) 
  (eq1 : a^2 + a*b + b^2 = 1 + i)
  (eq2 : b^2 + b*c + c^2 = -2)
  (eq3 : c^2 + c*a + a^2 = 1) :
  (a*b + b*c + c*a)^2 = (-11 - 4*i) / 3 := by
sorry

end NUMINAMATH_CALUDE_complex_sum_squared_l500_50025


namespace NUMINAMATH_CALUDE_income_ratio_is_5_to_4_l500_50029

-- Define the incomes and expenditures
def income_A : ℕ := 4000
def income_B : ℕ := 3200
def expenditure_A : ℕ := 2400
def expenditure_B : ℕ := 1600

-- Define the savings
def savings : ℕ := 1600

-- Theorem to prove
theorem income_ratio_is_5_to_4 :
  -- Conditions
  (expenditure_A / expenditure_B = 3 / 2) ∧
  (income_A - expenditure_A = savings) ∧
  (income_B - expenditure_B = savings) ∧
  (income_A = 4000) →
  -- Conclusion
  (income_A : ℚ) / (income_B : ℚ) = 5 / 4 :=
by sorry

end NUMINAMATH_CALUDE_income_ratio_is_5_to_4_l500_50029


namespace NUMINAMATH_CALUDE_dans_balloons_l500_50084

theorem dans_balloons (dans_balloons : ℕ) (tims_balloons : ℕ) : 
  tims_balloons = 203 → 
  tims_balloons = 7 * dans_balloons → 
  dans_balloons = 29 := by
sorry

end NUMINAMATH_CALUDE_dans_balloons_l500_50084


namespace NUMINAMATH_CALUDE_farmer_land_calculation_l500_50099

/-- Represents the total land owned by the farmer in acres -/
def total_land : ℝ := 7000

/-- Represents the proportion of land that was cleared for planting -/
def cleared_proportion : ℝ := 0.90

/-- Represents the proportion of cleared land planted with potato -/
def potato_proportion : ℝ := 0.20

/-- Represents the proportion of cleared land planted with tomato -/
def tomato_proportion : ℝ := 0.70

/-- Represents the amount of cleared land planted with corn in acres -/
def corn_land : ℝ := 630

theorem farmer_land_calculation :
  total_land * cleared_proportion * (potato_proportion + tomato_proportion) + corn_land = 
  total_land * cleared_proportion := by sorry

end NUMINAMATH_CALUDE_farmer_land_calculation_l500_50099


namespace NUMINAMATH_CALUDE_tony_total_cost_l500_50093

/-- Represents the total cost of Tony's purchases at the toy store -/
def total_cost (lego_price toy_sword_price play_dough_price : ℝ)
               (lego_sets toy_swords play_doughs : ℕ)
               (first_day_discount second_day_discount sales_tax : ℝ) : ℝ :=
  let first_day_cost := (2 * lego_price + 3 * toy_sword_price) * (1 - first_day_discount) * (1 + sales_tax)
  let second_day_cost := ((lego_sets - 2) * lego_price + (toy_swords - 3) * toy_sword_price + play_doughs * play_dough_price) * (1 - second_day_discount) * (1 + sales_tax)
  first_day_cost + second_day_cost

/-- Theorem stating that Tony's total cost matches the calculated amount -/
theorem tony_total_cost :
  total_cost 250 120 35 3 5 10 0.2 0.1 0.05 = 1516.20 := by
  sorry

end NUMINAMATH_CALUDE_tony_total_cost_l500_50093


namespace NUMINAMATH_CALUDE_rogers_trays_l500_50019

/-- Roger's tray-carrying problem -/
theorem rogers_trays (trays_per_trip : ℕ) (trips : ℕ) (trays_second_table : ℕ) : 
  trays_per_trip = 4 → trips = 3 → trays_second_table = 2 →
  trays_per_trip * trips - trays_second_table = 10 := by
  sorry

end NUMINAMATH_CALUDE_rogers_trays_l500_50019


namespace NUMINAMATH_CALUDE_cube_root_equation_solution_l500_50062

theorem cube_root_equation_solution (y : ℝ) :
  (5 - 2 / y) ^ (1/3 : ℝ) = -3 → y = 1/16 :=
by sorry

end NUMINAMATH_CALUDE_cube_root_equation_solution_l500_50062


namespace NUMINAMATH_CALUDE_total_cost_is_49_27_l500_50067

/-- Represents the cost of tickets for a family outing to a theme park -/
def theme_park_tickets : ℝ → Prop :=
  λ total_cost : ℝ =>
    ∃ (regular_price : ℝ),
      -- A senior ticket (30% discount) costs $7.50
      0.7 * regular_price = 7.5 ∧
      -- Total cost calculation
      total_cost = 2 * 7.5 + -- Two senior tickets
                   2 * regular_price + -- Two regular tickets
                   2 * (0.6 * regular_price) -- Two children tickets (40% discount)

/-- The total cost for all tickets is $49.27 -/
theorem total_cost_is_49_27 : theme_park_tickets 49.27 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_49_27_l500_50067


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l500_50071

theorem arithmetic_mean_of_fractions : 
  (1/3 : ℚ) * ((3/4 : ℚ) + (5/6 : ℚ) + (9/10 : ℚ)) = 149/180 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l500_50071


namespace NUMINAMATH_CALUDE_circle_M_equation_l500_50087

-- Define the circle M
def circle_M (a r : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - a)^2 + p.2^2 = r^2}

-- Define the line l₁: x = -2
def line_l₁ : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 = -2}

-- Define the line l₂: 2x - √5y - 4 = 0
def line_l₂ : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 2 * p.1 - Real.sqrt 5 * p.2 - 4 = 0}

theorem circle_M_equation 
  (a : ℝ) 
  (h1 : a > -2)
  (h2 : ∃ r : ℝ, 
    -- The chord formed by the intersection of M and l₁ has length 2√3
    (3 : ℝ) + (a + 2)^2 = r^2 ∧ 
    -- M is tangent to l₂
    r = |2 * a - 4| / 3) :
  circle_M a 2 = circle_M 1 2 := by sorry

end NUMINAMATH_CALUDE_circle_M_equation_l500_50087


namespace NUMINAMATH_CALUDE_debbys_flour_amount_l500_50043

/-- Calculates the final amount of flour Debby has -/
def final_flour_amount (initial : ℕ) (used : ℕ) (given : ℕ) (bought : ℕ) : ℕ :=
  initial - used - given + bought

/-- Proves that Debby's final amount of flour is 11 pounds -/
theorem debbys_flour_amount :
  final_flour_amount 12 3 2 4 = 11 := by
  sorry

end NUMINAMATH_CALUDE_debbys_flour_amount_l500_50043


namespace NUMINAMATH_CALUDE_largest_when_first_digit_changed_l500_50050

def original_number : ℚ := 0.123456

def change_digit (n : ℕ) (d : ℕ) : ℚ :=
  if n = 1 then 0.8 + (original_number - 0.1)
  else if n = 2 then 0.1 + 0.08 + (original_number - 0.12)
  else if n = 3 then 0.12 + 0.008 + (original_number - 0.123)
  else if n = 4 then 0.123 + 0.0008 + (original_number - 0.1234)
  else if n = 5 then 0.1234 + 0.00008 + (original_number - 0.12345)
  else 0.12345 + 0.000008 + (original_number - 0.123456)

theorem largest_when_first_digit_changed :
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 6 → change_digit 1 8 ≥ change_digit n 8 :=
by sorry

end NUMINAMATH_CALUDE_largest_when_first_digit_changed_l500_50050


namespace NUMINAMATH_CALUDE_translate_line_2x_minus_1_l500_50073

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Translates a line vertically -/
def translate_line (l : Line) (units : ℝ) : Line :=
  { slope := l.slope, intercept := l.intercept + units }

/-- The theorem stating that translating y = 2x - 1 by 2 units 
    upward results in y = 2x + 1 -/
theorem translate_line_2x_minus_1 :
  let original_line : Line := { slope := 2, intercept := -1 }
  let translated_line := translate_line original_line 2
  translated_line = { slope := 2, intercept := 1 } := by
  sorry

end NUMINAMATH_CALUDE_translate_line_2x_minus_1_l500_50073


namespace NUMINAMATH_CALUDE_system_equation_ratio_l500_50027

theorem system_equation_ratio (x y c d : ℝ) (h1 : 4 * x - 3 * y = c) (h2 : 2 * y - 8 * x = d) (h3 : d ≠ 0) :
  c / d = -1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_system_equation_ratio_l500_50027


namespace NUMINAMATH_CALUDE_paint_leftover_l500_50077

/-- Given the following conditions:
    1. The total number of paint containers is 16
    2. There are 4 equally-sized walls
    3. One wall is not painted
    4. One container is used for the ceiling
    Prove that the number of leftover paint containers is 3. -/
theorem paint_leftover (total_containers : ℕ) (num_walls : ℕ) (unpainted_walls : ℕ) (ceiling_containers : ℕ) :
  total_containers = 16 →
  num_walls = 4 →
  unpainted_walls = 1 →
  ceiling_containers = 1 →
  total_containers - (num_walls - unpainted_walls) * (total_containers / num_walls) - ceiling_containers = 3 :=
by sorry

end NUMINAMATH_CALUDE_paint_leftover_l500_50077


namespace NUMINAMATH_CALUDE_max_container_weight_l500_50081

def can_transport (k : ℕ) : Prop :=
  ∀ (distribution : List ℕ),
    (distribution.sum = 1500) →
    (∀ x ∈ distribution, x ≤ k ∧ x > 0) →
    ∃ (platform_loads : List ℕ),
      (platform_loads.length = 25) ∧
      (∀ load ∈ platform_loads, load ≤ 80) ∧
      (platform_loads.sum = 1500)

theorem max_container_weight :
  (can_transport 26) ∧ ¬(can_transport 27) := by sorry

end NUMINAMATH_CALUDE_max_container_weight_l500_50081


namespace NUMINAMATH_CALUDE_triangle_theorem_l500_50082

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem to be proved -/
theorem triangle_theorem (t : Triangle) 
  (h1 : t.b * (1 + Real.cos t.C) = t.c * (2 - Real.cos t.B))
  (h2 : t.C = π / 3)
  (h3 : t.a * t.b * Real.sin t.C / 2 = 4 * Real.sqrt 3) :
  (2 * t.c = t.a + t.b) ∧ (t.c = 4) := by
  sorry

end NUMINAMATH_CALUDE_triangle_theorem_l500_50082


namespace NUMINAMATH_CALUDE_ratio_transitivity_l500_50054

theorem ratio_transitivity (a b c : ℚ) 
  (hab : a / b = 8 / 3) 
  (hbc : b / c = 1 / 5) : 
  a / c = 8 / 15 := by
  sorry

end NUMINAMATH_CALUDE_ratio_transitivity_l500_50054


namespace NUMINAMATH_CALUDE_inequality_solution_set_l500_50076

theorem inequality_solution_set : 
  {x : ℤ | (x + 3)^3 ≤ 8} = {x : ℤ | x ≤ -1} := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l500_50076


namespace NUMINAMATH_CALUDE_vasyas_number_l500_50049

theorem vasyas_number : ∃! n : ℕ, 
  10 ≤ n ∧ n < 100 ∧ 
  1008 + 10 * n = 28 * n :=
by
  sorry

end NUMINAMATH_CALUDE_vasyas_number_l500_50049


namespace NUMINAMATH_CALUDE_amelia_monday_sales_l500_50009

/-- Represents the number of Jet Bars Amelia sold on Monday -/
def monday_sales : ℕ := sorry

/-- Represents the number of Jet Bars Amelia sold on Tuesday -/
def tuesday_sales : ℕ := sorry

/-- The weekly goal for Jet Bar sales -/
def weekly_goal : ℕ := 90

/-- The number of Jet Bars remaining to be sold -/
def remaining_sales : ℕ := 16

theorem amelia_monday_sales :
  monday_sales = 45 ∧
  tuesday_sales = monday_sales - 16 ∧
  monday_sales + tuesday_sales + remaining_sales = weekly_goal :=
by sorry

end NUMINAMATH_CALUDE_amelia_monday_sales_l500_50009


namespace NUMINAMATH_CALUDE_geometric_sequence_minimum_value_l500_50080

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_minimum_value
  (a : ℕ → ℝ)
  (h_pos : ∀ n, a n > 0)
  (h_geom : GeometricSequence a)
  (h_third : a 3 = 2 * a 1 + a 2)
  (h_exist : ∃ m n : ℕ, Real.sqrt (a m * a n) = 4 * a 1) :
  (∃ m n : ℕ, 1 / m + 4 / n = 3 / 2) ∧
  (∀ m n : ℕ, 1 / m + 4 / n ≥ 3 / 2) := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_minimum_value_l500_50080


namespace NUMINAMATH_CALUDE_average_first_21_multiples_of_5_l500_50003

theorem average_first_21_multiples_of_5 : 
  let n : ℕ := 21
  let multiples : List ℕ := List.range n |>.map (fun i => (i + 1) * 5)
  (multiples.sum : ℚ) / n = 55 := by
  sorry

end NUMINAMATH_CALUDE_average_first_21_multiples_of_5_l500_50003


namespace NUMINAMATH_CALUDE_part_one_part_two_l500_50075

-- Define the function f
def f (a x : ℝ) : ℝ := |x - a|

-- Part I
theorem part_one (x : ℝ) :
  let a := 2
  f a x ≥ 7 - |x - 1| ↔ x ∈ Set.Iic (-2) ∪ Set.Ici 5 := by sorry

-- Part II
theorem part_two (m n : ℝ) (h1 : m > 0) (h2 : n > 0) :
  (∀ x, f 1 x ≤ 1 ↔ x ∈ Set.Icc 0 2) →
  m^2 + 2*n^2 = 1 →
  m + 4*n ≤ 3 ∧ ∃ m n, m > 0 ∧ n > 0 ∧ m^2 + 2*n^2 = 1 ∧ m + 4*n = 3 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l500_50075


namespace NUMINAMATH_CALUDE_average_blanket_price_l500_50074

/-- The average price of blankets given specific purchase conditions -/
theorem average_blanket_price : 
  let blanket_group1 := (3, 100)  -- (quantity, price)
  let blanket_group2 := (5, 150)
  let blanket_group3 := (2, 275)  -- 550 / 2 = 275
  let total_blankets := blanket_group1.1 + blanket_group2.1 + blanket_group3.1
  let total_cost := blanket_group1.1 * blanket_group1.2 + 
                    blanket_group2.1 * blanket_group2.2 + 
                    blanket_group3.1 * blanket_group3.2
  (total_cost / total_blankets : ℚ) = 160 := by
  sorry

end NUMINAMATH_CALUDE_average_blanket_price_l500_50074


namespace NUMINAMATH_CALUDE_number_equation_solution_l500_50066

theorem number_equation_solution : 
  ∃ x : ℝ, x^2 + 95 = (x - 20)^2 ∧ x = 7.625 := by
  sorry

end NUMINAMATH_CALUDE_number_equation_solution_l500_50066


namespace NUMINAMATH_CALUDE_retreat_speed_l500_50038

theorem retreat_speed (total_distance : ℝ) (total_time : ℝ) (return_speed : ℝ) :
  total_distance = 600 →
  total_time = 10 →
  return_speed = 75 →
  ∃ outbound_speed : ℝ,
    outbound_speed = 50 ∧
    total_time = (total_distance / 2) / outbound_speed + (total_distance / 2) / return_speed :=
by sorry

end NUMINAMATH_CALUDE_retreat_speed_l500_50038


namespace NUMINAMATH_CALUDE_trigonometric_equality_l500_50079

theorem trigonometric_equality : 
  3.427 * Real.cos (50 * π / 180) + 
  8 * Real.cos (200 * π / 180) * Real.cos (220 * π / 180) * Real.cos (80 * π / 180) = 
  2 * Real.sin (65 * π / 180) ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_equality_l500_50079


namespace NUMINAMATH_CALUDE_quadrilateral_to_square_l500_50022

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a quadrilateral -/
structure Quadrilateral where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Represents a trapezoid -/
structure Trapezoid where
  base1 : ℝ
  base2 : ℝ
  height : ℝ

/-- Function to cut a quadrilateral into two trapezoids -/
def cutQuadrilateral (q : Quadrilateral) : (Trapezoid × Trapezoid) :=
  sorry

/-- Function to check if two trapezoids can form a square -/
def canFormSquare (t1 t2 : Trapezoid) : Prop :=
  sorry

/-- Theorem stating that the quadrilateral can be cut and rearranged into a square -/
theorem quadrilateral_to_square (q : Quadrilateral) :
  ∃ (t1 t2 : Trapezoid), 
    (t1, t2) = cutQuadrilateral q ∧ 
    canFormSquare t1 t2 ∧
    ∃ (side : ℝ), side = t1.height ∧ side * side = t1.base1 * t1.height + t2.base1 * t2.height :=
  sorry

end NUMINAMATH_CALUDE_quadrilateral_to_square_l500_50022


namespace NUMINAMATH_CALUDE_symmetric_points_difference_l500_50006

/-- Given two points A(a, 3) and B(-4, b) that are symmetric with respect to the origin,
    prove that a - b = 7 -/
theorem symmetric_points_difference (a b : ℝ) : 
  (∃ (A B : ℝ × ℝ), A = (a, 3) ∧ B = (-4, b) ∧ A = (-B.1, -B.2)) →
  a - b = 7 := by
sorry

end NUMINAMATH_CALUDE_symmetric_points_difference_l500_50006


namespace NUMINAMATH_CALUDE_unique_solution_ABC_l500_50015

/-- Converts a two-digit base-5 number to its decimal representation -/
def base5ToDecimal (tens : Nat) (ones : Nat) : Nat :=
  5 * tens + ones

/-- Theorem stating the unique solution for A, B, and C -/
theorem unique_solution_ABC :
  ∀ A B C : Nat,
  (A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0) →
  (A < 5 ∧ B < 5 ∧ C < 5) →
  (A ≠ B ∧ B ≠ C ∧ A ≠ C) →
  (base5ToDecimal A B + C = base5ToDecimal C 0) →
  (base5ToDecimal A B + base5ToDecimal B A = base5ToDecimal C C) →
  (A = 3 ∧ B = 2 ∧ C = 3) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_ABC_l500_50015


namespace NUMINAMATH_CALUDE_second_even_integer_is_78_l500_50039

/-- Given three consecutive even integers where the sum of the first and third is 156,
    prove that the second integer is 78. -/
theorem second_even_integer_is_78 :
  ∀ (a b c : ℤ),
  (b = a + 2) →  -- b is the next consecutive even integer after a
  (c = b + 2) →  -- c is the next consecutive even integer after b
  (a % 2 = 0) →  -- a is even
  (a + c = 156) →  -- sum of first and third is 156
  b = 78 := by
sorry

end NUMINAMATH_CALUDE_second_even_integer_is_78_l500_50039


namespace NUMINAMATH_CALUDE_find_A_l500_50045

theorem find_A : ∃ A : ℕ, 
  (1047 % A = 23) ∧ 
  (1047 % (A + 1) = 7) ∧ 
  (A = 64) := by
sorry

end NUMINAMATH_CALUDE_find_A_l500_50045


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_l500_50094

theorem isosceles_triangle_base (t α : ℝ) (h_t : t > 0) (h_α : 0 < α ∧ α < π) :
  ∃ a : ℝ, a > 0 ∧ a = 2 * Real.sqrt (t * Real.tan (α / 2)) ∧
    ∃ b : ℝ, b > 0 ∧
      let m := b * Real.cos (α / 2)
      t = (1 / 2) * a * m ∧
      α = 2 * Real.arccos (m / b) :=
by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_l500_50094


namespace NUMINAMATH_CALUDE_quadratic_properties_l500_50051

def quadratic_function (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_properties (a b c : ℝ) (h1 : a < 0) 
  (h2 : quadratic_function a b c (-1) = 0) 
  (h3 : -b / (2 * a) = 1) :
  (a - b + c = 0) ∧ 
  (∀ m : ℝ, quadratic_function a b c m ≤ -4 * a) ∧
  (∀ x1 x2 : ℝ, x1 < x2 → quadratic_function a b c x1 = -1 → 
    quadratic_function a b c x2 = -1 → x1 < -1 ∧ x2 > 3) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_properties_l500_50051


namespace NUMINAMATH_CALUDE_decreasing_condition_l500_50090

/-- The quadratic function f(x) = 2(x-1)^2 - 3 -/
def f (x : ℝ) : ℝ := 2 * (x - 1)^2 - 3

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 4 * (x - 1)

theorem decreasing_condition (x : ℝ) : 
  x < 1 → f' x < 0 :=
sorry

end NUMINAMATH_CALUDE_decreasing_condition_l500_50090


namespace NUMINAMATH_CALUDE_min_red_cells_for_win_thirteen_red_cells_win_l500_50072

/-- Represents an 8x8 grid where some cells are colored red -/
def Grid := Fin 8 → Fin 8 → Bool

/-- Returns true if the given cell is covered by the selected rows and columns -/
def isCovered (rows columns : Finset (Fin 8)) (i j : Fin 8) : Prop :=
  i ∈ rows ∨ j ∈ columns

/-- Returns the number of red cells in the grid -/
def redCount (g : Grid) : Nat :=
  (Finset.univ.filter (λ i => Finset.univ.filter (λ j => g i j) ≠ ∅)).card

/-- Returns true if there exists an uncovered red cell -/
def hasUncoveredRed (g : Grid) (rows columns : Finset (Fin 8)) : Prop :=
  ∃ i j, g i j ∧ ¬isCovered rows columns i j

theorem min_red_cells_for_win :
  ∀ n : Nat, n < 13 →
    ∃ g : Grid, redCount g = n ∧
      ∃ rows columns : Finset (Fin 8),
        rows.card = 4 ∧ columns.card = 4 ∧ ¬hasUncoveredRed g rows columns :=
by sorry

theorem thirteen_red_cells_win :
  ∃ g : Grid, redCount g = 13 ∧
    ∀ rows columns : Finset (Fin 8),
      rows.card = 4 ∧ columns.card = 4 → hasUncoveredRed g rows columns :=
by sorry

end NUMINAMATH_CALUDE_min_red_cells_for_win_thirteen_red_cells_win_l500_50072


namespace NUMINAMATH_CALUDE_total_shirts_made_l500_50061

-- Define the rate of shirt production
def shirts_per_minute : ℕ := 3

-- Define the working time
def working_time : ℕ := 2

-- Theorem to prove
theorem total_shirts_made : shirts_per_minute * working_time = 6 := by
  sorry

end NUMINAMATH_CALUDE_total_shirts_made_l500_50061


namespace NUMINAMATH_CALUDE_constant_dot_product_l500_50059

open Real

/-- Definition of the ellipse C -/
def ellipse (x y : ℝ) : Prop := x^2 / 27 + y^2 / 18 = 1

/-- Right focus of the ellipse -/
def F : ℝ × ℝ := (3, 0)

/-- The fixed point P -/
def P : ℝ × ℝ := (4, 0)

/-- A line passing through F -/
def line_through_F (k : ℝ) (x : ℝ) : ℝ := k * (x - F.1)

/-- Intersection points of the line with the ellipse -/
def intersection_points (k : ℝ) : Set (ℝ × ℝ) :=
  {p | ellipse p.1 p.2 ∧ p.2 = line_through_F k p.1}

/-- Dot product of vectors PA and PB -/
def dot_product (A B : ℝ × ℝ) : ℝ :=
  (A.1 - P.1) * (B.1 - P.1) + (A.2 - P.2) * (B.2 - P.2)

/-- Theorem: The dot product PA · PB is constant for any line through F -/
theorem constant_dot_product :
  ∃ (c : ℝ), ∀ (k : ℝ) (A B : ℝ × ℝ),
    A ∈ intersection_points k → B ∈ intersection_points k →
    A ≠ B → dot_product A B = c :=
sorry

end NUMINAMATH_CALUDE_constant_dot_product_l500_50059


namespace NUMINAMATH_CALUDE_mac_total_loss_l500_50030

/-- Represents the value of a coin in cents -/
def coin_value (coin : String) : ℕ :=
  match coin with
  | "penny" => 1
  | "nickel" => 5
  | "dime" => 10
  | "quarter" => 25
  | "half-dollar" => 50
  | _ => 0

/-- Calculates the expected loss for a single trade -/
def expected_loss (given_coins : List String) (probability : ℚ) : ℚ :=
  let given_value : ℚ := (given_coins.map coin_value).sum
  let quarter_value : ℚ := coin_value "quarter"
  (given_value - quarter_value) * probability

/-- Represents Mac's trading scenario -/
def mac_trades : List (List String × ℚ × ℕ) := [
  (["dime", "dime", "dime", "dime", "penny", "penny"], 1/20, 20),
  (["nickel", "nickel", "nickel", "nickel", "nickel", "nickel", "nickel", "nickel", "nickel", "penny"], 1/10, 20),
  (["half-dollar", "penny", "penny", "penny"], 17/20, 20)
]

/-- Theorem stating the total expected loss for Mac's trades -/
theorem mac_total_loss :
  (mac_trades.map (λ (coins, prob, repeats) => expected_loss coins prob * repeats)).sum = 535/100 := by
  sorry


end NUMINAMATH_CALUDE_mac_total_loss_l500_50030


namespace NUMINAMATH_CALUDE_pie_eating_contest_l500_50070

theorem pie_eating_contest :
  let student1 : ℚ := 8 / 9
  let student2 : ℚ := 5 / 6
  let student3 : ℚ := 2 / 3
  student1 + student2 + student3 = 43 / 18 :=
by sorry

end NUMINAMATH_CALUDE_pie_eating_contest_l500_50070


namespace NUMINAMATH_CALUDE_expression_equality_l500_50012

theorem expression_equality (x : ℝ) : x*(x*(x*(3-2*x)-4)+8)+3*x^2 = -2*x^4 + 3*x^3 - x^2 + 8*x := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l500_50012


namespace NUMINAMATH_CALUDE_min_pencils_in_box_l500_50037

theorem min_pencils_in_box (total_boxes : Nat) (total_pencils : Nat) (max_capacity : Nat)
  (h1 : total_boxes = 13)
  (h2 : total_pencils = 74)
  (h3 : max_capacity = 6) :
  ∃ (min_pencils : Nat), min_pencils = 2 ∧
    (∀ (box : Nat), box ≤ total_boxes → ∃ (pencils_in_box : Nat),
      pencils_in_box ≥ min_pencils ∧ pencils_in_box ≤ max_capacity) ∧
    (∃ (box : Nat), box ≤ total_boxes ∧ ∃ (pencils_in_box : Nat), pencils_in_box = min_pencils) :=
by
  sorry

end NUMINAMATH_CALUDE_min_pencils_in_box_l500_50037


namespace NUMINAMATH_CALUDE_quadratic_sum_l500_50031

/-- Given a quadratic polynomial 6x^2 + 36x + 216, when expressed in the form a(x + b)^2 + c,
    where a, b, and c are constants, prove that a + b + c = 171. -/
theorem quadratic_sum (a b c : ℝ) : 
  (∀ x, 6 * x^2 + 36 * x + 216 = a * (x + b)^2 + c) → a + b + c = 171 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_l500_50031


namespace NUMINAMATH_CALUDE_negative_square_root_operations_l500_50078

theorem negative_square_root_operations :
  (-Real.sqrt (2^2) < 0) ∧
  ((Real.sqrt 2)^2 ≥ 0) ∧
  (Real.sqrt (2^2) ≥ 0) ∧
  (Real.sqrt ((-2)^2) ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_negative_square_root_operations_l500_50078


namespace NUMINAMATH_CALUDE_gcd_factorial_eight_and_factorial_six_squared_l500_50005

theorem gcd_factorial_eight_and_factorial_six_squared : 
  Nat.gcd (Nat.factorial 8) ((Nat.factorial 6)^2) = 1440 := by
  sorry

end NUMINAMATH_CALUDE_gcd_factorial_eight_and_factorial_six_squared_l500_50005


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l500_50092

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m

theorem arithmetic_sequence_sum (a : ℕ → ℚ) :
  arithmetic_sequence a → a 5 = 21 → a 4 + a 5 + a 6 = 63 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l500_50092


namespace NUMINAMATH_CALUDE_no_valid_arrangement_l500_50044

def is_valid_arrangement (perm : List Nat) : Prop :=
  perm.length = 8 ∧
  (∀ n, n ∈ perm → n ∈ [1, 2, 3, 4, 5, 6, 8, 9]) ∧
  (∀ i, i < perm.length - 1 → (10 * perm[i]! + perm[i+1]!) % 7 = 0)

theorem no_valid_arrangement : ¬∃ perm : List Nat, is_valid_arrangement perm := by
  sorry

end NUMINAMATH_CALUDE_no_valid_arrangement_l500_50044


namespace NUMINAMATH_CALUDE_cell_population_after_ten_days_l500_50097

/-- Represents the growth of a cell population over time -/
def cellGrowth (initialPopulation : ℕ) (growthFactor : ℕ) (intervalDays : ℕ) (totalDays : ℕ) : ℕ :=
  initialPopulation * growthFactor ^ (totalDays / intervalDays)

/-- Theorem stating the cell population after 10 days -/
theorem cell_population_after_ten_days :
  cellGrowth 5 3 2 10 = 1215 := by
  sorry

#eval cellGrowth 5 3 2 10

end NUMINAMATH_CALUDE_cell_population_after_ten_days_l500_50097


namespace NUMINAMATH_CALUDE_inequality_proof_l500_50017

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h_sum_squares : a^2 + b^2 + c^2 = 1) :
  1/a^2 + 1/b^2 + 1/c^2 ≥ 2*(a^3 + b^3 + c^3)/(a*b*c) + 3 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l500_50017


namespace NUMINAMATH_CALUDE_max_correct_answers_l500_50034

theorem max_correct_answers (total_questions : ℕ) (correct_points : ℤ) (incorrect_points : ℤ) (total_score : ℤ) :
  total_questions = 60 →
  correct_points = 5 →
  incorrect_points = -2 →
  total_score = 150 →
  ∃ (correct blank incorrect : ℕ),
    correct + blank + incorrect = total_questions ∧
    correct_points * correct + incorrect_points * incorrect = total_score ∧
    correct ≤ 38 ∧
    ∀ (c : ℕ), c > 38 →
      ¬(∃ (b i : ℕ), c + b + i = total_questions ∧
        correct_points * c + incorrect_points * i = total_score) :=
by sorry

end NUMINAMATH_CALUDE_max_correct_answers_l500_50034


namespace NUMINAMATH_CALUDE_total_net_buried_bones_l500_50010

/-- Represents the types of bones Barkley receives --/
inductive BoneType
  | A
  | B
  | C

/-- Represents Barkley's bone statistics over 5 months --/
structure BoneStats where
  received : Nat
  buried : Nat
  eaten : Nat

/-- Calculates the net buried bones for a given BoneStats --/
def netBuried (stats : BoneStats) : Nat :=
  stats.buried - stats.eaten

/-- Defines Barkley's bone statistics for each type over 5 months --/
def barkleyStats : BoneType → BoneStats
  | BoneType.A => { received := 50, buried := 30, eaten := 3 }
  | BoneType.B => { received := 30, buried := 16, eaten := 2 }
  | BoneType.C => { received := 20, buried := 10, eaten := 2 }

/-- Theorem: The total net number of buried bones after 5 months is 49 --/
theorem total_net_buried_bones :
  (netBuried (barkleyStats BoneType.A) +
   netBuried (barkleyStats BoneType.B) +
   netBuried (barkleyStats BoneType.C)) = 49 := by
  sorry


end NUMINAMATH_CALUDE_total_net_buried_bones_l500_50010


namespace NUMINAMATH_CALUDE_min_value_theorem_equality_condition_l500_50098

theorem min_value_theorem (a : ℝ) (h : a > 0) : 
  2 * a + 1 / a ≥ 2 * Real.sqrt 2 :=
by sorry

theorem equality_condition (a : ℝ) (h : a > 0) : 
  (2 * a + 1 / a = 2 * Real.sqrt 2) ↔ (a = Real.sqrt 2 / 2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_equality_condition_l500_50098


namespace NUMINAMATH_CALUDE_function_difference_equals_nine_minimum_value_minus_four_l500_50065

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*a*x - 3

-- Theorem 1
theorem function_difference_equals_nine (a : ℝ) :
  f a (a + 1) - f a a = 9 → a = 2 :=
sorry

-- Theorem 2
theorem minimum_value_minus_four (a : ℝ) :
  (∃ x, f a x = -4 ∧ ∀ y, f a y ≥ -4) → (a = 1 ∨ a = -1) :=
sorry

end NUMINAMATH_CALUDE_function_difference_equals_nine_minimum_value_minus_four_l500_50065


namespace NUMINAMATH_CALUDE_quadratic_max_value_change_l500_50026

theorem quadratic_max_value_change (a b c : ℝ) (h_a : a < 0) :
  let f (x : ℝ) := a * x^2 + b * x + c
  let max_value (a' : ℝ) := -b^2 / (4 * a') + c
  (max_value (a + 1) = max_value a + 27 / 2) →
  (max_value (a - 4) = max_value a - 9) →
  (max_value (a - 2) = max_value a - 27 / 4) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_max_value_change_l500_50026


namespace NUMINAMATH_CALUDE_harvey_sam_race_l500_50063

theorem harvey_sam_race (sam_miles harvey_miles : ℕ) : 
  sam_miles = 12 → 
  harvey_miles > sam_miles → 
  sam_miles + harvey_miles = 32 → 
  harvey_miles - sam_miles = 8 := by
sorry

end NUMINAMATH_CALUDE_harvey_sam_race_l500_50063


namespace NUMINAMATH_CALUDE_clark_bought_seven_parts_l500_50001

/-- The number of parts Clark bought -/
def n : ℕ := sorry

/-- The original price of each part in dollars -/
def original_price : ℕ := 80

/-- The total amount Clark paid in dollars -/
def total_paid : ℕ := 439

/-- The total discount in dollars -/
def total_discount : ℕ := 121

theorem clark_bought_seven_parts : n = 7 := by
  sorry

end NUMINAMATH_CALUDE_clark_bought_seven_parts_l500_50001


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l500_50096

theorem sufficient_not_necessary (a : ℝ) : 
  (∀ a, a > 1 → a^2 > 1) ∧ (∃ a, a ≤ 1 ∧ a^2 > 1) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l500_50096


namespace NUMINAMATH_CALUDE_impossible_tiling_l500_50057

/-- Represents a tile type -/
inductive TileType
| TwoByTwo
| OneByFour

/-- Represents a set of tiles -/
structure TileSet where
  twoByTwo : Nat
  oneByFour : Nat

/-- Represents a rectangular box -/
structure Box where
  length : Nat
  width : Nat

/-- Checks if a box can be tiled with a given tile set -/
def canTile (box : Box) (tiles : TileSet) : Prop :=
  sorry

/-- The main theorem -/
theorem impossible_tiling (box : Box) (initialTiles : TileSet) :
  canTile box initialTiles →
  ¬canTile box { twoByTwo := initialTiles.twoByTwo - 1, oneByFour := initialTiles.oneByFour + 1 } :=
sorry

end NUMINAMATH_CALUDE_impossible_tiling_l500_50057


namespace NUMINAMATH_CALUDE_smallest_number_with_remainders_l500_50064

theorem smallest_number_with_remainders : ∃ (b : ℕ), b = 87 ∧
  b % 5 = 2 ∧ b % 4 = 3 ∧ b % 7 = 1 ∧
  ∀ (n : ℕ), n % 5 = 2 ∧ n % 4 = 3 ∧ n % 7 = 1 → b ≤ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_with_remainders_l500_50064


namespace NUMINAMATH_CALUDE_paris_saturday_study_hours_l500_50013

/-- The number of hours Paris studies on Saturdays during the semester -/
def saturday_study_hours (
  semester_weeks : ℕ)
  (weekday_study_hours : ℕ)
  (sunday_study_hours : ℕ)
  (total_study_hours : ℕ) : ℕ :=
  total_study_hours - (semester_weeks * 5 * weekday_study_hours) - (semester_weeks * sunday_study_hours)

/-- Theorem stating that Paris studies 60 hours on Saturdays during the semester -/
theorem paris_saturday_study_hours :
  saturday_study_hours 15 3 5 360 = 60 := by
  sorry


end NUMINAMATH_CALUDE_paris_saturday_study_hours_l500_50013


namespace NUMINAMATH_CALUDE_min_value_on_interval_l500_50000

-- Define the function
def f (x : ℝ) : ℝ := x^2 - 4*x + 1

-- Define the interval
def interval : Set ℝ := Set.Icc 0 3

-- Theorem statement
theorem min_value_on_interval :
  ∃ (x : ℝ), x ∈ interval ∧ f x = -3 ∧ ∀ (y : ℝ), y ∈ interval → f y ≥ f x :=
sorry

end NUMINAMATH_CALUDE_min_value_on_interval_l500_50000


namespace NUMINAMATH_CALUDE_stamps_needed_tara_stamps_problem_l500_50011

theorem stamps_needed (current_stamps : ℕ) (stamps_per_sheet : ℕ) : ℕ :=
  stamps_per_sheet - (current_stamps % stamps_per_sheet)

theorem tara_stamps_problem : stamps_needed 38 9 = 7 := by
  sorry

end NUMINAMATH_CALUDE_stamps_needed_tara_stamps_problem_l500_50011


namespace NUMINAMATH_CALUDE_sum_of_odd_integers_between_400_and_700_l500_50040

def first_term : ℕ := 401
def last_term : ℕ := 699
def common_difference : ℕ := 2

def number_of_terms : ℕ := (last_term - first_term) / common_difference + 1

theorem sum_of_odd_integers_between_400_and_700 :
  (number_of_terms : ℝ) / 2 * (first_term + last_term : ℝ) = 82500 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_odd_integers_between_400_and_700_l500_50040


namespace NUMINAMATH_CALUDE_matrix_equation_proof_l500_50089

def N : Matrix (Fin 2) (Fin 2) ℝ := !![1, -10; 0, 1]

theorem matrix_equation_proof :
  N^3 - 3 * N^2 + 2 * N = !![5, 10; 0, 5] := by sorry

end NUMINAMATH_CALUDE_matrix_equation_proof_l500_50089


namespace NUMINAMATH_CALUDE_polynomial_coefficients_l500_50041

/-- The polynomial f(x) = ax^4 - 7x^3 + bx^2 - 12x - 8 -/
def f (a b x : ℝ) : ℝ := a * x^4 - 7 * x^3 + b * x^2 - 12 * x - 8

/-- Theorem stating that if f(2) = -7 and f(-3) = -80, then a = -9/4 and b = 29.25 -/
theorem polynomial_coefficients (a b : ℝ) :
  f a b 2 = -7 ∧ f a b (-3) = -80 → a = -9/4 ∧ b = 29.25 := by
  sorry

#check polynomial_coefficients

end NUMINAMATH_CALUDE_polynomial_coefficients_l500_50041


namespace NUMINAMATH_CALUDE_cosine_sine_sum_zero_l500_50052

theorem cosine_sine_sum_zero (x : ℝ) 
  (h : Real.cos (π / 6 - x) = -Real.sqrt 3 / 3) : 
  Real.cos (5 * π / 6 + x) + Real.sin (2 * π / 3 - x) = 0 := by
  sorry

end NUMINAMATH_CALUDE_cosine_sine_sum_zero_l500_50052


namespace NUMINAMATH_CALUDE_paper_tray_height_l500_50048

theorem paper_tray_height (side_length : ℝ) (cut_distance : ℝ) (cut_angle : ℝ) :
  side_length = 120 →
  cut_distance = 6 →
  cut_angle = 45 →
  let tray_height := cut_distance
  tray_height = 6 := by sorry

end NUMINAMATH_CALUDE_paper_tray_height_l500_50048


namespace NUMINAMATH_CALUDE_extra_postage_count_l500_50047

/-- Represents an envelope with its dimensions --/
structure Envelope where
  length : Float
  height : Float
  thickness : Float

/-- Checks if an envelope requires extra postage --/
def requiresExtraPostage (e : Envelope) : Bool :=
  let ratio := e.length / e.height
  ratio < 1.2 || ratio > 2.8 || e.thickness > 0.25

/-- The set of envelopes given in the problem --/
def envelopes : List Envelope := [
  { length := 7, height := 5, thickness := 0.2 },
  { length := 10, height := 2, thickness := 0.3 },
  { length := 7, height := 7, thickness := 0.1 },
  { length := 12, height := 4, thickness := 0.26 }
]

/-- The main theorem to prove --/
theorem extra_postage_count :
  (envelopes.filter requiresExtraPostage).length = 3 := by
  sorry

end NUMINAMATH_CALUDE_extra_postage_count_l500_50047


namespace NUMINAMATH_CALUDE_one_divides_six_digit_number_l500_50035

/-- Represents a 6-digit number of the form abacab -/
def SixDigitNumber (a b c : ℕ) : ℕ :=
  100000 * a + 10000 * b + 1000 * a + 100 * c + 10 * a + b

/-- Theorem stating that 1 is a factor of any SixDigitNumber -/
theorem one_divides_six_digit_number (a b c : ℕ) (h1 : a ≠ 0) (h2 : a < 10) (h3 : b < 10) (h4 : c < 10) :
  1 ∣ SixDigitNumber a b c := by
  sorry


end NUMINAMATH_CALUDE_one_divides_six_digit_number_l500_50035


namespace NUMINAMATH_CALUDE_problem_solution_l500_50018

def problem (A B X : ℕ) : Prop :=
  A > 0 ∧ B > 0 ∧
  Nat.gcd A B = 20 ∧
  A = 300 ∧
  Nat.lcm A B = 20 * X * 15

theorem problem_solution :
  ∀ A B X, problem A B X → X = 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l500_50018


namespace NUMINAMATH_CALUDE_product_minus_sum_probability_l500_50004

def valid_pair (a b : ℕ) : Prop :=
  a ≤ 10 ∧ b ≤ 10 ∧ a * b - (a + b) > 4

def total_pairs : ℕ := 100

def valid_pairs : ℕ := 44

theorem product_minus_sum_probability :
  (valid_pairs : ℚ) / total_pairs = 11 / 25 := by sorry

end NUMINAMATH_CALUDE_product_minus_sum_probability_l500_50004


namespace NUMINAMATH_CALUDE_house_of_cards_impossible_l500_50046

theorem house_of_cards_impossible (decks : ℕ) (cards_per_deck : ℕ) (layers : ℕ) : 
  decks = 36 → cards_per_deck = 104 → layers = 64 → 
  ¬ ∃ (cards_per_layer : ℕ), (decks * cards_per_deck) = (layers * cards_per_layer) :=
by
  sorry

end NUMINAMATH_CALUDE_house_of_cards_impossible_l500_50046


namespace NUMINAMATH_CALUDE_basil_daytime_cookies_l500_50016

/-- Represents the number of cookies Basil gets per day -/
structure BasilCookies where
  morning : ℚ
  evening : ℚ
  daytime : ℕ

/-- Represents the cookie box information -/
structure CookieBox where
  cookies_per_box : ℕ
  boxes_needed : ℕ
  days_lasting : ℕ

theorem basil_daytime_cookies 
  (basil_cookies : BasilCookies)
  (cookie_box : CookieBox)
  (h1 : basil_cookies.morning = 1/2)
  (h2 : basil_cookies.evening = 1/2)
  (h3 : cookie_box.cookies_per_box = 45)
  (h4 : cookie_box.boxes_needed = 2)
  (h5 : cookie_box.days_lasting = 30) :
  basil_cookies.daytime = 2 :=
sorry

end NUMINAMATH_CALUDE_basil_daytime_cookies_l500_50016


namespace NUMINAMATH_CALUDE_cube_side_length_l500_50002

theorem cube_side_length (surface_area : ℝ) (h : surface_area = 600) :
  ∃ (side_length : ℝ), side_length > 0 ∧ 6 * side_length^2 = surface_area ∧ side_length = 10 := by
  sorry

end NUMINAMATH_CALUDE_cube_side_length_l500_50002


namespace NUMINAMATH_CALUDE_unique_solution_absolute_value_equation_l500_50086

theorem unique_solution_absolute_value_equation :
  ∃! x : ℝ, |x - 2| = |x - 4| + |x - 6| := by
sorry

end NUMINAMATH_CALUDE_unique_solution_absolute_value_equation_l500_50086


namespace NUMINAMATH_CALUDE_sons_age_l500_50008

/-- Proves that given the conditions, the son's age is 35 years. -/
theorem sons_age (son_age father_age : ℕ) : 
  father_age = son_age + 37 →
  father_age + 2 = 2 * (son_age + 2) →
  son_age = 35 := by
sorry

end NUMINAMATH_CALUDE_sons_age_l500_50008


namespace NUMINAMATH_CALUDE_partnership_problem_l500_50056

/-- Partnership problem -/
theorem partnership_problem (a_months b_months : ℕ) (b_contribution total_profit a_share : ℝ) 
  (h1 : a_months = 8)
  (h2 : b_months = 5)
  (h3 : b_contribution = 6000)
  (h4 : total_profit = 8400)
  (h5 : a_share = 4800) :
  ∃ (a_contribution : ℝ),
    a_contribution * a_months * (total_profit - a_share) = 
    b_contribution * b_months * a_share ∧ 
    a_contribution = 5000 := by
  sorry

end NUMINAMATH_CALUDE_partnership_problem_l500_50056


namespace NUMINAMATH_CALUDE_proportional_function_decreasing_l500_50060

/-- A proportional function passing through (2, -4) has a decreasing y as x increases -/
theorem proportional_function_decreasing (k : ℝ) (h1 : k ≠ 0) (h2 : k * 2 = -4) :
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → k * x₁ > k * x₂ := by
  sorry

end NUMINAMATH_CALUDE_proportional_function_decreasing_l500_50060


namespace NUMINAMATH_CALUDE_sphere_surface_area_l500_50032

theorem sphere_surface_area (V : Real) (r : Real) : 
  V = (4 / 3) * Real.pi * r^3 → 
  V = 36 * Real.pi → 
  4 * Real.pi * r^2 = 36 * Real.pi := by
sorry

end NUMINAMATH_CALUDE_sphere_surface_area_l500_50032


namespace NUMINAMATH_CALUDE_city_council_vote_change_l500_50033

theorem city_council_vote_change :
  ∀ (x y x' y' : ℕ),
    x + y = 500 →
    y > x →
    x' + y' = 500 →
    x' - y' = (3 * (y - x)) / 2 →
    x' = (13 * y) / 12 →
    x' - x = 125 :=
by sorry

end NUMINAMATH_CALUDE_city_council_vote_change_l500_50033


namespace NUMINAMATH_CALUDE_fourth_player_win_probability_prove_fourth_player_win_probability_l500_50028

/-- The probability of the fourth player winning in a coin-flipping game -/
theorem fourth_player_win_probability : Real → Prop :=
  fun p =>
    -- Define the game setup
    let n_players : ℕ := 4
    let coin_prob : Real := 1 / 2
    -- Define the probability of the fourth player winning on their nth turn
    let prob_win_nth_turn : ℕ → Real := fun n => coin_prob ^ (n_players * n)
    -- Define the sum of the infinite geometric series
    let total_prob : Real := (prob_win_nth_turn 1) / (1 - prob_win_nth_turn 1)
    -- The theorem statement
    p = total_prob ∧ p = 1 / 31

/-- Proof of the theorem -/
theorem prove_fourth_player_win_probability : 
  ∃ p : Real, fourth_player_win_probability p :=
sorry

end NUMINAMATH_CALUDE_fourth_player_win_probability_prove_fourth_player_win_probability_l500_50028


namespace NUMINAMATH_CALUDE_line_slope_point_value_l500_50068

theorem line_slope_point_value (m : ℝ) : 
  m > 0 → 
  (((m - 5) / (2 - m)) = Real.sqrt 2) → 
  m = 2 + 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_line_slope_point_value_l500_50068


namespace NUMINAMATH_CALUDE_puzzle_solution_l500_50083

theorem puzzle_solution (c o u n t s : ℕ) 
  (h1 : c + o = u)
  (h2 : u + n = t)
  (h3 : t + c = s)
  (h4 : o + n + s = 12)
  (h5 : c ≠ 0 ∧ o ≠ 0 ∧ u ≠ 0 ∧ n ≠ 0 ∧ t ≠ 0 ∧ s ≠ 0) :
  t = 6 := by
  sorry


end NUMINAMATH_CALUDE_puzzle_solution_l500_50083


namespace NUMINAMATH_CALUDE_circle_points_speeds_l500_50036

/-- Two points moving along a unit circle -/
structure CirclePoints where
  v₁ : ℝ  -- Speed of the first point
  v₂ : ℝ  -- Speed of the second point

/-- Conditions for the circle points -/
def satisfies_conditions (cp : CirclePoints) : Prop :=
  cp.v₁ > 0 ∧ cp.v₂ > 0 ∧  -- Positive speeds
  cp.v₁ - cp.v₂ = 1 / 720 ∧  -- Meet every 12 minutes (720 seconds)
  1 / cp.v₂ - 1 / cp.v₁ = 10  -- First point is 10 seconds faster

/-- The theorem to be proved -/
theorem circle_points_speeds (cp : CirclePoints) 
  (h : satisfies_conditions cp) : cp.v₁ = 1/80 ∧ cp.v₂ = 1/90 := by
  sorry

end NUMINAMATH_CALUDE_circle_points_speeds_l500_50036


namespace NUMINAMATH_CALUDE_hyperbola_probability_l500_50023

-- Define the set of possible values for m and n
def S : Set ℕ := {1, 2, 3}

-- Define the condition for (m, n) to be on the hyperbola
def on_hyperbola (m n : ℕ) : Prop := n = 6 / m

-- Define the probability space
def total_outcomes : ℕ := 6

-- Define the favorable outcomes
def favorable_outcomes : ℕ := 2

-- State the theorem
theorem hyperbola_probability :
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 3 := by sorry

end NUMINAMATH_CALUDE_hyperbola_probability_l500_50023


namespace NUMINAMATH_CALUDE_sequence_existence_and_extension_l500_50042

theorem sequence_existence_and_extension (m : ℕ) (hm : m ≥ 2) :
  (∃ x : ℕ → ℕ, ∀ i : ℕ, 1 ≤ i ∧ i ≤ m → x i * x (m + i) = x (i + 1) * x (m + i - 1) + 1) ∧
  (∀ x : ℕ → ℕ, (∀ i : ℕ, 1 ≤ i ∧ i ≤ m → x i * x (m + i) = x (i + 1) * x (m + i - 1) + 1) →
    ∃ y : ℤ → ℕ, (∀ k : ℤ, y k * y (m + k) = y (k + 1) * y (m + k - 1) + 1) ∧
               (∀ i : ℕ, 1 ≤ i ∧ i ≤ 2 * m → y i = x i)) :=
by sorry

end NUMINAMATH_CALUDE_sequence_existence_and_extension_l500_50042


namespace NUMINAMATH_CALUDE_brick_height_l500_50055

/-- The surface area of a rectangular prism given its length, width, and height. -/
def surface_area (l w h : ℝ) : ℝ := 2 * (l * w + l * h + w * h)

/-- Theorem stating that a rectangular prism with length 10, width 4, and surface area 136 has height 2. -/
theorem brick_height : ∃ (h : ℝ), h > 0 ∧ surface_area 10 4 h = 136 → h = 2 := by
  sorry

end NUMINAMATH_CALUDE_brick_height_l500_50055


namespace NUMINAMATH_CALUDE_smallest_common_flock_size_l500_50007

theorem smallest_common_flock_size : ∃ n : ℕ, 
  n > 0 ∧ 
  n % 13 = 0 ∧ 
  n % 14 = 0 ∧ 
  (∀ m : ℕ, m > 0 → m % 13 = 0 → m % 14 = 0 → m ≥ n) ∧
  n = 182 := by
sorry

end NUMINAMATH_CALUDE_smallest_common_flock_size_l500_50007


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l500_50088

theorem arithmetic_geometric_sequence (d : ℝ) (a : ℕ → ℝ) :
  d ≠ 0 ∧
  (∀ n, a (n + 1) = a n + d) ∧
  a 1 = 1 ∧
  (a 3) ^ 2 = a 1 * a 13 →
  d = 2 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l500_50088


namespace NUMINAMATH_CALUDE_min_additional_candies_for_equal_distribution_l500_50014

/-- Given 25 candies and 4 friends, the minimum number of additional candies
    needed for equal distribution is 1. -/
theorem min_additional_candies_for_equal_distribution :
  let initial_candies : ℕ := 25
  let num_friends : ℕ := 4
  let additional_candies : ℕ := 1
  (initial_candies + additional_candies) % num_friends = 0 ∧
  ∀ n : ℕ, n < additional_candies → (initial_candies + n) % num_friends ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_min_additional_candies_for_equal_distribution_l500_50014


namespace NUMINAMATH_CALUDE_not_solution_one_l500_50085

theorem not_solution_one (x : ℂ) (h1 : x^2 + x + 1 = 0) (h2 : x ≠ 0) : x ≠ 1 := by
  sorry

end NUMINAMATH_CALUDE_not_solution_one_l500_50085


namespace NUMINAMATH_CALUDE_complex_equality_l500_50069

theorem complex_equality (u v : ℂ) 
  (h1 : 3 * Complex.abs (u + 1) * Complex.abs (v + 1) ≥ Complex.abs (u * v + 5 * u + 5 * v + 1))
  (h2 : Complex.abs (u + v) = Complex.abs (u * v + 1)) :
  u = 1 ∨ v = 1 :=
sorry

end NUMINAMATH_CALUDE_complex_equality_l500_50069
