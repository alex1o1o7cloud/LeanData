import Mathlib

namespace NUMINAMATH_CALUDE_amount_after_two_years_l3541_354174

theorem amount_after_two_years
  (initial_amount : ℝ)
  (annual_rate : ℝ)
  (years : ℕ)
  (h1 : initial_amount = 51200)
  (h2 : annual_rate = 1 / 8)
  (h3 : years = 2) :
  initial_amount * (1 + annual_rate) ^ years = 64800 :=
by sorry

end NUMINAMATH_CALUDE_amount_after_two_years_l3541_354174


namespace NUMINAMATH_CALUDE_bathroom_area_is_eight_l3541_354156

/-- The area of a rectangular bathroom -/
def bathroom_area (length width : ℝ) : ℝ := length * width

/-- Theorem: The area of a bathroom with length 4 feet and width 2 feet is 8 square feet -/
theorem bathroom_area_is_eight :
  bathroom_area 4 2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_bathroom_area_is_eight_l3541_354156


namespace NUMINAMATH_CALUDE_bird_speed_theorem_l3541_354125

theorem bird_speed_theorem (d t : ℝ) (h1 : d = 50 * (t + 1/12)) (h2 : d = 70 * (t - 1/12)) :
  let r := d / t
  ∃ ε > 0, abs (r - 58) < ε :=
sorry

end NUMINAMATH_CALUDE_bird_speed_theorem_l3541_354125


namespace NUMINAMATH_CALUDE_first_term_of_constant_ratio_l3541_354187

def arithmetic_sum (a : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  (n : ℚ) * (2 * a + (n - 1 : ℚ) * d) / 2

theorem first_term_of_constant_ratio (d : ℚ) (h : d = 5) :
  (∃ k : ℚ, ∀ n : ℕ, n > 0 → 
    arithmetic_sum a d (4*n) / arithmetic_sum a d n = k) →
  a = -5/2 :=
sorry

end NUMINAMATH_CALUDE_first_term_of_constant_ratio_l3541_354187


namespace NUMINAMATH_CALUDE_nicky_pace_is_3_l3541_354153

/-- Nicky's pace in meters per second -/
def nicky_pace : ℝ := 3

/-- Cristina's pace in meters per second -/
def cristina_pace : ℝ := 5

/-- Head start given to Nicky in meters -/
def head_start : ℝ := 48

/-- Time it takes Cristina to catch up to Nicky in seconds -/
def catch_up_time : ℝ := 24

/-- Theorem stating that Nicky's pace is 3 meters per second given the conditions -/
theorem nicky_pace_is_3 :
  cristina_pace > nicky_pace ∧
  cristina_pace * catch_up_time = nicky_pace * catch_up_time + head_start →
  nicky_pace = 3 := by
  sorry


end NUMINAMATH_CALUDE_nicky_pace_is_3_l3541_354153


namespace NUMINAMATH_CALUDE_susan_spending_ratio_l3541_354166

/-- Proves that the ratio of the amount spent on books to the amount left after buying clothes is 1:2 --/
theorem susan_spending_ratio (
  total_earned : ℝ)
  (spent_on_clothes : ℝ)
  (left_after_books : ℝ)
  (h1 : total_earned = 600)
  (h2 : spent_on_clothes = total_earned / 2)
  (h3 : left_after_books = 150) :
  (total_earned - spent_on_clothes - left_after_books) / (total_earned - spent_on_clothes) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_susan_spending_ratio_l3541_354166


namespace NUMINAMATH_CALUDE_two_digit_numbers_product_5681_sum_154_l3541_354123

theorem two_digit_numbers_product_5681_sum_154 : 
  ∃ (a b : ℕ), 10 ≤ a ∧ a < 100 ∧ 10 ≤ b ∧ b < 100 ∧ a * b = 5681 ∧ a + b = 154 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_numbers_product_5681_sum_154_l3541_354123


namespace NUMINAMATH_CALUDE_reflection_line_sum_l3541_354128

/-- Given a line y = mx + c, if the reflection of point (-2, 0) across this line is (6, 4), then m + c = 4 -/
theorem reflection_line_sum (m c : ℝ) : 
  (∀ (x y : ℝ), y = m * x + c → 
    (x + 2) * (x - 6) + (y - 4) * (y - 0) = 0 ∧ 
    (x - 2) = m * (y - 2)) → 
  m + c = 4 := by sorry

end NUMINAMATH_CALUDE_reflection_line_sum_l3541_354128


namespace NUMINAMATH_CALUDE_number_fraction_relation_l3541_354193

theorem number_fraction_relation (x : ℝ) (h : (2 / 5) * x = 20) : (1 / 3) * x = 50 / 3 := by
  sorry

end NUMINAMATH_CALUDE_number_fraction_relation_l3541_354193


namespace NUMINAMATH_CALUDE_positive_solution_square_root_form_l3541_354132

theorem positive_solution_square_root_form :
  ∃ (a' b' : ℕ+), 
    (∃ (x : ℝ), x^2 + 14*x = 96 ∧ x > 0 ∧ x = Real.sqrt a' - b') ∧
    a' = 145 ∧ 
    b' = 7 ∧
    (a' : ℕ) + (b' : ℕ) = 152 := by
  sorry

end NUMINAMATH_CALUDE_positive_solution_square_root_form_l3541_354132


namespace NUMINAMATH_CALUDE_sequence_max_value_l3541_354161

theorem sequence_max_value (n : ℤ) : -n^2 + 15*n + 3 ≤ 59 := by
  sorry

end NUMINAMATH_CALUDE_sequence_max_value_l3541_354161


namespace NUMINAMATH_CALUDE_equation_holds_iff_base_ten_l3541_354101

/-- Represents a digit in base k --/
def Digit (k : ℕ) := Fin k

/-- Converts a natural number to its representation in base k --/
def toBaseK (n : ℕ) (k : ℕ) : List (Digit k) :=
  sorry

/-- Adds two numbers represented in base k --/
def addBaseK (a b : List (Digit k)) : List (Digit k) :=
  sorry

/-- Converts a list of digits in base k to a natural number --/
def fromBaseK (digits : List (Digit k)) (k : ℕ) : ℕ :=
  sorry

/-- The main theorem stating that the equation holds iff k = 10 --/
theorem equation_holds_iff_base_ten (k : ℕ) :
  (fromBaseK (addBaseK (toBaseK 5342 k) (toBaseK 6421 k)) k = fromBaseK (toBaseK 14163 k) k) ↔ k = 10 :=
sorry

end NUMINAMATH_CALUDE_equation_holds_iff_base_ten_l3541_354101


namespace NUMINAMATH_CALUDE_nina_age_l3541_354105

/-- Given the ages of Max, Leah, Alex, and Nina, prove Nina's age --/
theorem nina_age (max_age leah_age alex_age nina_age : ℕ) 
  (h1 : max_age = leah_age - 5)
  (h2 : leah_age = alex_age + 6)
  (h3 : nina_age = alex_age + 2)
  (h4 : max_age = 16) : 
  nina_age = 17 := by
  sorry

#check nina_age

end NUMINAMATH_CALUDE_nina_age_l3541_354105


namespace NUMINAMATH_CALUDE_race_lead_calculation_l3541_354173

theorem race_lead_calculation (total_length max_remaining : ℕ) 
  (initial_together first_lead second_lead : ℕ) : 
  total_length = 5000 →
  max_remaining = 3890 →
  initial_together = 200 →
  first_lead = 300 →
  second_lead = 170 →
  (total_length - max_remaining - initial_together) - (first_lead - second_lead) = 780 :=
by sorry

end NUMINAMATH_CALUDE_race_lead_calculation_l3541_354173


namespace NUMINAMATH_CALUDE_roadwork_pitch_barrels_l3541_354137

/-- Roadwork problem -/
theorem roadwork_pitch_barrels (total_length day1_paving : ℕ)
  (gravel_per_truckload : ℕ) (gravel_pitch_ratio : ℕ) (truckloads_per_mile : ℕ) :
  total_length = 16 →
  day1_paving = 4 →
  gravel_per_truckload = 2 →
  gravel_pitch_ratio = 5 →
  truckloads_per_mile = 3 →
  (total_length - (day1_paving + (2 * day1_paving - 1))) * truckloads_per_mile * 
    (gravel_per_truckload / gravel_pitch_ratio : ℚ) = 6 :=
by sorry

end NUMINAMATH_CALUDE_roadwork_pitch_barrels_l3541_354137


namespace NUMINAMATH_CALUDE_inequality_solution_l3541_354124

theorem inequality_solution (x : ℕ+) : 
  (x.val - 3) / 3 < 7 - (5 / 3) * x.val ↔ x.val = 1 ∨ x.val = 2 ∨ x.val = 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l3541_354124


namespace NUMINAMATH_CALUDE_binomial_coefficient_10_3_l3541_354136

theorem binomial_coefficient_10_3 : Nat.choose 10 3 = 120 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_10_3_l3541_354136


namespace NUMINAMATH_CALUDE_x_power_plus_reciprocal_l3541_354116

theorem x_power_plus_reciprocal (θ : ℝ) (x : ℝ) (n : ℕ+) 
  (h1 : 0 < θ) (h2 : θ < π / 2) (h3 : x + 1 / x = 2 * Real.sin θ) : 
  x^(n : ℝ) + 1 / x^(n : ℝ) = 2 * Real.cos (n * (π / 2 - θ)) := by
  sorry

end NUMINAMATH_CALUDE_x_power_plus_reciprocal_l3541_354116


namespace NUMINAMATH_CALUDE_grape_sales_properties_l3541_354102

/-- Represents the properties of the grape sales scenario -/
structure GrapeSales where
  initial_price : ℝ
  initial_volume : ℝ
  cost_price : ℝ
  price_reduction_effect : ℝ

/-- Calculates the daily sales profit for a given price reduction -/
def daily_profit (g : GrapeSales) (price_reduction : ℝ) : ℝ :=
  let new_price := g.initial_price - price_reduction
  let new_volume := g.initial_volume + price_reduction * g.price_reduction_effect
  (new_price - g.cost_price) * new_volume

/-- Calculates the profit as a function of selling price -/
def profit_function (g : GrapeSales) (x : ℝ) : ℝ :=
  (x - g.cost_price) * (g.initial_volume + (g.initial_price - x) * g.price_reduction_effect)

/-- Theorem stating the properties of the grape sales scenario -/
theorem grape_sales_properties (g : GrapeSales) 
  (h1 : g.initial_price = 30)
  (h2 : g.initial_volume = 60)
  (h3 : g.cost_price = 15)
  (h4 : g.price_reduction_effect = 10) :
  daily_profit g 2 = 1040 ∧ 
  (∃ (x : ℝ), x = 51/2 ∧ ∀ (y : ℝ), profit_function g y ≤ profit_function g x) ∧
  (∃ (max_profit : ℝ), max_profit = 1102.5 ∧ 
    ∀ (y : ℝ), profit_function g y ≤ max_profit) := by
  sorry

end NUMINAMATH_CALUDE_grape_sales_properties_l3541_354102


namespace NUMINAMATH_CALUDE_smaller_number_in_ratio_l3541_354103

theorem smaller_number_in_ratio (a b : ℝ) : 
  a > 0 ∧ b > 0 ∧  -- Two positive numbers
  b / a = 11 / 7 ∧  -- In the ratio 7:11
  b - a = 16  -- Larger number exceeds smaller by 16
  → a = 28 := by  -- The smaller number is 28
sorry

end NUMINAMATH_CALUDE_smaller_number_in_ratio_l3541_354103


namespace NUMINAMATH_CALUDE_seventh_root_unity_product_l3541_354131

theorem seventh_root_unity_product (r : ℂ) (h1 : r^7 = 1) (h2 : r ≠ 1) :
  (r - 1) * (r^2 - 1) * (r^3 - 1) * (r^4 - 1) * (r^5 - 1) * (r^6 - 1) = 14 := by
  sorry

end NUMINAMATH_CALUDE_seventh_root_unity_product_l3541_354131


namespace NUMINAMATH_CALUDE_binomial_coefficient_16_4_l3541_354134

theorem binomial_coefficient_16_4 : Nat.choose 16 4 = 1820 := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_16_4_l3541_354134


namespace NUMINAMATH_CALUDE_multiples_of_15_between_21_and_205_l3541_354148

theorem multiples_of_15_between_21_and_205 : 
  (Finset.filter (fun n => n % 15 = 0 ∧ n > 21 ∧ n < 205) (Finset.range 205)).card = 12 := by
  sorry

end NUMINAMATH_CALUDE_multiples_of_15_between_21_and_205_l3541_354148


namespace NUMINAMATH_CALUDE_marathon_total_distance_l3541_354160

/-- Represents the length of a marathon in miles and yards -/
structure Marathon where
  miles : ℕ
  yards : ℕ

/-- Represents a total distance in miles and yards -/
structure TotalDistance where
  miles : ℕ
  yards : ℕ
  h : yards < 1760

def marathon_length : Marathon := { miles := 26, yards := 385 }

def yards_per_mile : ℕ := 1760

def num_marathons : ℕ := 10

theorem marathon_total_distance :
  ∃ (m : ℕ) (y : ℕ) (h : y < 1760),
    (m * yards_per_mile + y) = 
      (num_marathons * marathon_length.miles * yards_per_mile + 
       num_marathons * marathon_length.yards) ∧
    y = 330 := by sorry

end NUMINAMATH_CALUDE_marathon_total_distance_l3541_354160


namespace NUMINAMATH_CALUDE_fraction_zero_implies_x_equals_one_l3541_354165

theorem fraction_zero_implies_x_equals_one (x : ℝ) :
  (x^2 - 1) / ((x - 2) * (x + 1)) = 0 → x = 1 :=
by sorry

end NUMINAMATH_CALUDE_fraction_zero_implies_x_equals_one_l3541_354165


namespace NUMINAMATH_CALUDE_rectangle_area_l3541_354152

theorem rectangle_area (p : ℝ) (h : p > 0) : ∃ (l w : ℝ),
  l > 0 ∧ w > 0 ∧
  l / w = 5 / 2 ∧
  2 * (l + w) = p ∧
  l * w = (5 / 98) * p^2 :=
sorry

end NUMINAMATH_CALUDE_rectangle_area_l3541_354152


namespace NUMINAMATH_CALUDE_root_in_smaller_interval_l3541_354141

-- Define the function
def f (x : ℝ) := x^3 - 6*x^2 + 4

-- State the theorem
theorem root_in_smaller_interval :
  (∃ x ∈ Set.Ioo 0 1, f x = 0) →
  (∃ x ∈ Set.Ioo (1/2) 1, f x = 0) :=
by sorry

end NUMINAMATH_CALUDE_root_in_smaller_interval_l3541_354141


namespace NUMINAMATH_CALUDE_amy_math_problems_l3541_354112

/-- The number of math problems Amy had to solve -/
def math_problems : ℕ := sorry

/-- The number of spelling problems Amy had to solve -/
def spelling_problems : ℕ := 6

/-- The number of problems Amy can finish in an hour -/
def problems_per_hour : ℕ := 4

/-- The number of hours it took Amy to finish all problems -/
def total_hours : ℕ := 6

/-- Theorem stating that Amy had 18 math problems -/
theorem amy_math_problems : 
  math_problems = 18 := by sorry

end NUMINAMATH_CALUDE_amy_math_problems_l3541_354112


namespace NUMINAMATH_CALUDE_composite_sequences_exist_l3541_354190

def is_composite (n : ℕ) : Prop := ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

def consecutive_composites (start : ℕ) (len : ℕ) : Prop :=
  ∀ i, i ∈ Finset.range len → is_composite (start + i)

theorem composite_sequences_exist :
  (∃ start : ℕ, start ≤ 500 - 9 + 1 ∧ consecutive_composites start 9) ∧
  (∃ start : ℕ, start ≤ 500 - 11 + 1 ∧ consecutive_composites start 11) :=
sorry

end NUMINAMATH_CALUDE_composite_sequences_exist_l3541_354190


namespace NUMINAMATH_CALUDE_f_geq_m_range_l3541_354178

/-- The function f(x) = x^2 - 2mx + 2 -/
def f (x m : ℝ) : ℝ := x^2 - 2*m*x + 2

/-- The theorem stating the range of m for which f(x) ≥ m holds for all x ∈ [-1, +∞) -/
theorem f_geq_m_range (m : ℝ) :
  (∀ x : ℝ, x ≥ -1 → f x m ≥ m) ↔ -3 ≤ m ∧ m ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_f_geq_m_range_l3541_354178


namespace NUMINAMATH_CALUDE_least_integer_with_divisibility_l3541_354108

def n : ℕ := 2329089562800

theorem least_integer_with_divisibility (k : ℕ) (hk : k < n) : 
  (∀ i ∈ Finset.range 18, n % (i + 1) = 0) ∧ 
  (∀ i ∈ Finset.range 10, n % (i + 21) = 0) ∧ 
  n % 19 ≠ 0 ∧ 
  n % 20 ≠ 0 → 
  ¬(∀ i ∈ Finset.range 18, k % (i + 1) = 0) ∨ 
  ¬(∀ i ∈ Finset.range 10, k % (i + 21) = 0) ∨ 
  k % 19 = 0 ∨ 
  k % 20 = 0 :=
by sorry

#check least_integer_with_divisibility

end NUMINAMATH_CALUDE_least_integer_with_divisibility_l3541_354108


namespace NUMINAMATH_CALUDE_arithmetic_equality_l3541_354179

theorem arithmetic_equality : 4 * 5 + 5 * 4 = 40 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_equality_l3541_354179


namespace NUMINAMATH_CALUDE_drilled_solid_surface_area_l3541_354159

noncomputable section

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cube with given edge length -/
structure Cube where
  edgeLength : ℝ

/-- Represents the drilled solid S -/
structure DrilledSolid where
  cube : Cube
  tunnelStart : Point3D
  tunnelEnd : Point3D

/-- Calculate the surface area of the drilled solid S -/
def surfaceArea (s : DrilledSolid) : ℝ := sorry

/-- The main theorem stating the surface area of the drilled solid -/
theorem drilled_solid_surface_area 
  (e f g h c d b a i j k : Point3D)
  (cube : Cube)
  (s : DrilledSolid)
  (h1 : cube.edgeLength = 10)
  (h2 : e.x = 10 ∧ e.y = 10 ∧ e.z = 10)
  (h3 : i.x = 7 ∧ i.y = 10 ∧ i.z = 10)
  (h4 : j.x = 10 ∧ j.y = 7 ∧ j.z = 10)
  (h5 : k.x = 10 ∧ k.y = 10 ∧ k.z = 7)
  (h6 : s.cube = cube)
  (h7 : s.tunnelStart = i)
  (h8 : s.tunnelEnd = k) :
  surfaceArea s = 582 + 13.5 * Real.sqrt 6 := by sorry

end NUMINAMATH_CALUDE_drilled_solid_surface_area_l3541_354159


namespace NUMINAMATH_CALUDE_line_intersects_circle_r_range_l3541_354117

/-- The range of r for a line intersecting a circle -/
theorem line_intersects_circle_r_range (α : Real) (r : Real) :
  (∃ x y : Real, x * Real.cos α + y * Real.sin α = 1 ∧ x^2 + y^2 = r^2) →
  r > 0 →
  r > 1 := by
  sorry

end NUMINAMATH_CALUDE_line_intersects_circle_r_range_l3541_354117


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_4_range_of_a_for_all_x_geq_4_l3541_354127

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - 1| + |x - a|

-- Theorem for part (1)
theorem solution_set_when_a_is_4 :
  {x : ℝ | f 4 x ≥ 5} = {x : ℝ | x ≤ 0} ∪ {x : ℝ | x ≥ 5} := by sorry

-- Theorem for part (2)
theorem range_of_a_for_all_x_geq_4 :
  (∀ x : ℝ, f a x ≥ 4) ↔ (a ≤ -3 ∨ a ≥ 5) := by sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_4_range_of_a_for_all_x_geq_4_l3541_354127


namespace NUMINAMATH_CALUDE_max_value_theorem_l3541_354130

theorem max_value_theorem (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hsum : x + y + z = 2) :
  (∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → a + b + c = 2 → (a + b) / (a * b * c) ≤ (x + y) / (x * y * z)) →
  (x + y) / (x * y * z) = 13.5 :=
sorry

end NUMINAMATH_CALUDE_max_value_theorem_l3541_354130


namespace NUMINAMATH_CALUDE_polynomial_expansion_l3541_354164

theorem polynomial_expansion (x : ℝ) : 
  (3*x^2 + 4*x + 8)*(x - 2) - (x - 2)*(2*x^2 + 5*x - 72) + (4*x - 21)*(x - 2)*(x - 3) = 
  5*x^3 - 23*x^2 + 43*x - 34 := by
sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l3541_354164


namespace NUMINAMATH_CALUDE_division_problem_l3541_354168

theorem division_problem (divisor : ℕ) : 
  (265 / divisor = 12) ∧ (265 % divisor = 1) → divisor = 22 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l3541_354168


namespace NUMINAMATH_CALUDE_two_valid_colorings_l3541_354181

/-- Represents the three possible colors for a hexagon -/
inductive Color
| Red
| Yellow
| Green

/-- Represents a column of hexagons -/
structure Column where
  hexagons : List Color
  size : Nat
  size_eq : hexagons.length = size

/-- Represents the entire figure of hexagons -/
structure HexagonFigure where
  column1 : Column
  column2 : Column
  column3 : Column
  column4 : Column
  col1_size : column1.size = 3
  col2_size : column2.size = 4
  col3_size : column3.size = 4
  col4_size : column4.size = 3
  bottom_red : column1.hexagons.head? = some Color.Red

/-- Predicate to check if two colors are different -/
def differentColors (c1 c2 : Color) : Prop :=
  c1 ≠ c2

/-- Predicate to check if a coloring is valid (no adjacent hexagons have the same color) -/
def validColoring (figure : HexagonFigure) : Prop :=
  -- Add conditions to check adjacent hexagons in each column and between columns
  sorry

/-- The number of valid colorings for the hexagon figure -/
def numValidColorings : Nat :=
  -- Count the number of valid colorings
  sorry

/-- Theorem stating that there are exactly 2 valid colorings -/
theorem two_valid_colorings : numValidColorings = 2 := by
  sorry

end NUMINAMATH_CALUDE_two_valid_colorings_l3541_354181


namespace NUMINAMATH_CALUDE_no_roots_implies_non_integer_difference_l3541_354185

theorem no_roots_implies_non_integer_difference (a b : ℝ) : 
  a ≠ b → 
  (∀ x : ℝ, (x^2 + 20*a*x + 10*b) * (x^2 + 20*b*x + 10*a) ≠ 0) → 
  ¬(∃ n : ℤ, 20*(b - a) = n) :=
by sorry

end NUMINAMATH_CALUDE_no_roots_implies_non_integer_difference_l3541_354185


namespace NUMINAMATH_CALUDE_other_number_proof_l3541_354139

theorem other_number_proof (a b : ℕ+) : 
  Nat.gcd a b = 12 → 
  Nat.lcm a b = 396 → 
  a = 36 → 
  b = 132 := by
sorry

end NUMINAMATH_CALUDE_other_number_proof_l3541_354139


namespace NUMINAMATH_CALUDE_martha_and_john_money_l3541_354146

theorem martha_and_john_money : (5 / 8 : ℚ) + (2 / 5 : ℚ) = 1.025 := by sorry

end NUMINAMATH_CALUDE_martha_and_john_money_l3541_354146


namespace NUMINAMATH_CALUDE_sum_product_inequality_cubic_inequality_l3541_354194

-- Part 1
theorem sum_product_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (sum_eq_one : a + b + c = 1) : 
  a * b + b * c + c * a ≤ 1 / 3 := by
sorry

-- Part 2
theorem cubic_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a ≥ b) :
  2 * a^3 - b^3 ≥ 2 * a * b^2 - a^2 * b := by
sorry

end NUMINAMATH_CALUDE_sum_product_inequality_cubic_inequality_l3541_354194


namespace NUMINAMATH_CALUDE_max_value_of_b_plus_c_l3541_354172

/-- A cubic function f(x) = x³ + bx² + cx + d -/
def f (b c d : ℝ) (x : ℝ) : ℝ := x^3 + b*x^2 + c*x + d

/-- The derivative of f(x) -/
def f_deriv (b c : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*b*x + c

/-- f(x) is decreasing on the interval [-2, 2] -/
def is_decreasing_on_interval (b c d : ℝ) : Prop :=
  ∀ x ∈ Set.Icc (-2) 2, f_deriv b c x ≤ 0

theorem max_value_of_b_plus_c (b c d : ℝ) 
  (h : is_decreasing_on_interval b c d) : 
  b + c ≤ -12 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_b_plus_c_l3541_354172


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l3541_354162

/-- Given two vectors a and b in R^3, if a is parallel to b, then m = -2 -/
theorem parallel_vectors_m_value (m : ℝ) :
  let a : Fin 3 → ℝ := ![2*m+1, 3, m-1]
  let b : Fin 3 → ℝ := ![2, m, -m]
  (∃ (k : ℝ), a = k • b) → m = -2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_value_l3541_354162


namespace NUMINAMATH_CALUDE_box_height_rounding_equivalence_l3541_354189

def round_to_nearest_ten (n : ℕ) : ℕ :=
  10 * ((n + 5) / 10)

theorem box_height_rounding_equivalence :
  let height1 : ℕ := 53
  let height2 : ℕ := 78
  let correct_sum := height1 + height2
  let alice_sum := height1 + round_to_nearest_ten height2
  round_to_nearest_ten correct_sum = round_to_nearest_ten alice_sum :=
by
  sorry

end NUMINAMATH_CALUDE_box_height_rounding_equivalence_l3541_354189


namespace NUMINAMATH_CALUDE_shaded_area_approx_l3541_354138

-- Define the circle and rectangle
def circle_radius : ℝ := 3
def rectangle_side_OA : ℝ := 2
def rectangle_side_AB : ℝ := 1

-- Define the points
def O : ℝ × ℝ := (0, 0)
def A : ℝ × ℝ := (rectangle_side_OA, 0)
def B : ℝ × ℝ := (rectangle_side_OA, rectangle_side_AB)
def C : ℝ × ℝ := (0, rectangle_side_AB)

-- Define the function to calculate the area of the shaded region
def shaded_area : ℝ := sorry

-- Theorem statement
theorem shaded_area_approx :
  abs (shaded_area - 6.23) < 0.01 := by sorry

end NUMINAMATH_CALUDE_shaded_area_approx_l3541_354138


namespace NUMINAMATH_CALUDE_pony_price_is_18_l3541_354129

/-- The regular price of Fox jeans in dollars -/
def fox_price : ℝ := 15

/-- The discount rate for Pony jeans as a decimal -/
def pony_discount : ℝ := 0.1

/-- The sum of discount rates for Fox and Pony jeans as a decimal -/
def total_discount : ℝ := 0.22

/-- The number of Fox jeans purchased -/
def fox_quantity : ℕ := 3

/-- The number of Pony jeans purchased -/
def pony_quantity : ℕ := 2

/-- The total savings in dollars from purchasing both types of jeans -/
def total_savings : ℝ := 9

/-- Theorem stating that the regular price of Pony jeans is $18 -/
theorem pony_price_is_18 : 
  ∃ (pony_price : ℝ), 
    pony_price = 18 ∧ 
    (fox_quantity * fox_price * (total_discount - pony_discount) + 
     pony_quantity * pony_price * pony_discount = total_savings) :=
by sorry

end NUMINAMATH_CALUDE_pony_price_is_18_l3541_354129


namespace NUMINAMATH_CALUDE_class_average_proof_l3541_354199

/-- Given a class with boys and girls, their average scores, and the ratio of boys to girls,
    prove that the overall class average is 94 points. -/
theorem class_average_proof (boys_avg : ℝ) (girls_avg : ℝ) (ratio : ℝ) :
  boys_avg = 90 →
  girls_avg = 96 →
  ratio = 0.5 →
  (ratio * girls_avg + girls_avg) / (ratio + 1) = 94 := by
  sorry

end NUMINAMATH_CALUDE_class_average_proof_l3541_354199


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l3541_354188

theorem imaginary_part_of_z (z : ℂ) (h : (1 - Complex.I) * z = 3 + 7 * Complex.I) : 
  z.im = 5 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l3541_354188


namespace NUMINAMATH_CALUDE_sheila_mon_wed_fri_hours_l3541_354196

/-- Represents Sheila's work schedule and earnings --/
structure WorkSchedule where
  hours_mon_wed_fri : ℕ
  hours_tue_thu : ℕ
  weekly_earnings : ℕ
  hourly_rate : ℕ

/-- Theorem stating the number of hours Sheila works on Monday, Wednesday, and Friday --/
theorem sheila_mon_wed_fri_hours (schedule : WorkSchedule) 
  (h1 : schedule.hours_tue_thu = 6 * 2)
  (h2 : schedule.weekly_earnings = 360)
  (h3 : schedule.hourly_rate = 10)
  (h4 : schedule.weekly_earnings = schedule.hourly_rate * (schedule.hours_mon_wed_fri + schedule.hours_tue_thu)) :
  schedule.hours_mon_wed_fri = 24 := by
  sorry

#check sheila_mon_wed_fri_hours

end NUMINAMATH_CALUDE_sheila_mon_wed_fri_hours_l3541_354196


namespace NUMINAMATH_CALUDE_part_one_part_two_l3541_354170

-- Define the function f
def f (x a : ℝ) : ℝ := |x - 1| + |x - a|

-- Part 1
theorem part_one : 
  let a := -1
  {x : ℝ | f x a ≥ 3} = {x : ℝ | x ≤ -1.5 ∨ x ≥ 1.5} := by sorry

-- Part 2
theorem part_two :
  (∀ x : ℝ, f x a ≥ 2) → (a = 3 ∨ a = -1) := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l3541_354170


namespace NUMINAMATH_CALUDE_distance_traveled_downstream_l3541_354142

/-- The distance traveled downstream by a boat -/
def distance_downstream (boat_speed : ℝ) (stream_speed : ℝ) (time : ℝ) : ℝ :=
  (boat_speed + stream_speed) * time

/-- Theorem: The distance traveled downstream is 68 km -/
theorem distance_traveled_downstream : 
  distance_downstream 13 4 4 = 68 := by
  sorry

end NUMINAMATH_CALUDE_distance_traveled_downstream_l3541_354142


namespace NUMINAMATH_CALUDE_average_glasses_per_box_l3541_354151

/-- Prove that the average number of glasses per box is 15, given the following conditions:
  - There are two types of boxes: small (12 glasses) and large (16 glasses)
  - There are 16 more large boxes than small boxes
  - The total number of glasses is 480
-/
theorem average_glasses_per_box (small_box : ℕ) (large_box : ℕ) :
  small_box * 12 + large_box * 16 = 480 →
  large_box = small_box + 16 →
  (480 : ℚ) / (small_box + large_box) = 15 := by
sorry


end NUMINAMATH_CALUDE_average_glasses_per_box_l3541_354151


namespace NUMINAMATH_CALUDE_power_of_three_plus_five_mod_eight_l3541_354133

theorem power_of_three_plus_five_mod_eight :
  (3^100 + 5) % 8 = 6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_three_plus_five_mod_eight_l3541_354133


namespace NUMINAMATH_CALUDE_square_field_area_proof_l3541_354122

def square_field_area (wire_cost_per_meter : ℚ) (total_cost : ℚ) (gate_width : ℚ) (num_gates : ℕ) : ℚ :=
  let side_length := ((total_cost / wire_cost_per_meter + 2 * gate_width * num_gates) / 4 : ℚ)
  side_length * side_length

theorem square_field_area_proof (wire_cost_per_meter : ℚ) (total_cost : ℚ) (gate_width : ℚ) (num_gates : ℕ) :
  wire_cost_per_meter = 3/2 ∧ total_cost = 999 ∧ gate_width = 1 ∧ num_gates = 2 →
  square_field_area wire_cost_per_meter total_cost gate_width num_gates = 27889 := by
  sorry

#eval square_field_area (3/2) 999 1 2

end NUMINAMATH_CALUDE_square_field_area_proof_l3541_354122


namespace NUMINAMATH_CALUDE_tulip_lilac_cost_comparison_l3541_354158

/-- Given that 4 tulips and 5 lilacs cost less than 22 yuan, and 6 tulips and 3 lilacs cost more than 24 yuan, prove that 2 tulips cost more than 3 lilacs. -/
theorem tulip_lilac_cost_comparison (x y : ℝ) 
  (h1 : 4 * x + 5 * y < 22) 
  (h2 : 6 * x + 3 * y > 24) : 
  2 * x > 3 * y := by
  sorry

end NUMINAMATH_CALUDE_tulip_lilac_cost_comparison_l3541_354158


namespace NUMINAMATH_CALUDE_arithmetic_sequence_count_l3541_354197

theorem arithmetic_sequence_count (a₁ a_n d : ℤ) (h1 : a₁ = 156) (h2 : a_n = 36) (h3 : d = -4) :
  (a₁ - a_n) / d + 1 = 31 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_count_l3541_354197


namespace NUMINAMATH_CALUDE_exam_average_l3541_354155

theorem exam_average (students_group1 : ℕ) (average1 : ℚ) 
  (students_group2 : ℕ) (average2 : ℚ) : 
  students_group1 = 15 → 
  average1 = 70/100 → 
  students_group2 = 10 → 
  average2 = 90/100 → 
  (students_group1 * average1 + students_group2 * average2) / (students_group1 + students_group2) = 78/100 := by
  sorry

end NUMINAMATH_CALUDE_exam_average_l3541_354155


namespace NUMINAMATH_CALUDE_minimum_a_value_l3541_354110

theorem minimum_a_value (a : ℝ) : 
  (∀ (x y : ℝ), x > 0 → y > 0 → Real.sqrt x + Real.sqrt y ≤ a * Real.sqrt (x + y)) ↔ 
  a ≥ Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_minimum_a_value_l3541_354110


namespace NUMINAMATH_CALUDE_ratio_problem_l3541_354109

theorem ratio_problem (x y z w : ℝ) 
  (h1 : x / y = 5)
  (h2 : y / z = 1 / 4)
  (h3 : z / w = 7)
  : w / x = 4 / 35 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l3541_354109


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3541_354198

def A : Set ℤ := {-2, -1}
def B : Set ℤ := {-1, 2, 3}

theorem intersection_of_A_and_B : A ∩ B = {-1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3541_354198


namespace NUMINAMATH_CALUDE_negative_two_power_sum_l3541_354184

theorem negative_two_power_sum : (-2)^2004 + (-2)^2005 = -2^2004 := by sorry

end NUMINAMATH_CALUDE_negative_two_power_sum_l3541_354184


namespace NUMINAMATH_CALUDE_total_packs_eq_sum_l3541_354157

/-- The number of glue stick packs Emily's mom bought -/
def total_packs : ℕ := sorry

/-- The number of glue stick packs Emily received -/
def emily_packs : ℕ := 6

/-- The number of glue stick packs Emily's sister received -/
def sister_packs : ℕ := 7

/-- Theorem: The total number of glue stick packs is the sum of packs given to Emily and her sister -/
theorem total_packs_eq_sum : total_packs = emily_packs + sister_packs := by sorry

end NUMINAMATH_CALUDE_total_packs_eq_sum_l3541_354157


namespace NUMINAMATH_CALUDE_pure_imaginary_ratio_l3541_354195

theorem pure_imaginary_ratio (c d : ℝ) (hc : c ≠ 0) (hd : d ≠ 0) 
  (h : ∃ (y : ℝ), (3 - 4 * Complex.I) * (c + d * Complex.I) = y * Complex.I) : 
  c / d = -4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_ratio_l3541_354195


namespace NUMINAMATH_CALUDE_binomial_coefficient_congruence_l3541_354183

theorem binomial_coefficient_congruence (p n : ℕ) (hp : Prime p) :
  (Nat.choose n p) ≡ (n / p : ℕ) [MOD p] := by sorry

end NUMINAMATH_CALUDE_binomial_coefficient_congruence_l3541_354183


namespace NUMINAMATH_CALUDE_stating_equal_cost_guests_proof_l3541_354120

/-- The number of guests for which the costs of renting Caesar's and Venus Hall are equal -/
def equal_cost_guests : ℕ := 60

/-- The room rental cost for Caesar's -/
def caesars_rental : ℕ := 800

/-- The per-meal cost for Caesar's -/
def caesars_per_meal : ℕ := 30

/-- The room rental cost for Venus Hall -/
def venus_rental : ℕ := 500

/-- The per-meal cost for Venus Hall -/
def venus_per_meal : ℕ := 35

/-- 
Theorem stating that the number of guests for which the costs of renting 
Caesar's and Venus Hall are equal is 60, given the rental and per-meal costs for each venue.
-/
theorem equal_cost_guests_proof :
  caesars_rental + caesars_per_meal * equal_cost_guests = 
  venus_rental + venus_per_meal * equal_cost_guests :=
by sorry

end NUMINAMATH_CALUDE_stating_equal_cost_guests_proof_l3541_354120


namespace NUMINAMATH_CALUDE_arrival_time_difference_l3541_354119

def distance : ℝ := 1.2
def jill_speed : ℝ := 8
def jack_speed : ℝ := 5

theorem arrival_time_difference : 
  let jill_time := distance / jill_speed
  let jack_time := distance / jack_speed
  (jack_time - jill_time) * 60 = 5.4 := by sorry

end NUMINAMATH_CALUDE_arrival_time_difference_l3541_354119


namespace NUMINAMATH_CALUDE_range_of_H_l3541_354118

-- Define the function H
def H (x : ℝ) : ℝ := |x + 2| - |x - 3|

-- State the theorem about the range of H
theorem range_of_H :
  ∀ y : ℝ, (∃ x : ℝ, H x = y) ↔ 3 ≤ y ∧ y ≤ 5 :=
by sorry

end NUMINAMATH_CALUDE_range_of_H_l3541_354118


namespace NUMINAMATH_CALUDE_car_profit_percentage_l3541_354180

theorem car_profit_percentage (original_price : ℝ) (h1 : original_price > 0) : 
  let discount_rate : ℝ := 0.2
  let increase_rate : ℝ := 1
  let buying_price : ℝ := original_price * (1 - discount_rate)
  let selling_price : ℝ := buying_price * (1 + increase_rate)
  let profit : ℝ := selling_price - original_price
  let profit_percentage : ℝ := (profit / original_price) * 100
  profit_percentage = 60 := by sorry

end NUMINAMATH_CALUDE_car_profit_percentage_l3541_354180


namespace NUMINAMATH_CALUDE_shepherd_sheep_problem_l3541_354177

/-- The number of sheep with the shepherd boy on the mountain -/
def x : ℕ := 20

/-- The number of sheep with the shepherd boy at the foot of the mountain -/
def y : ℕ := 12

/-- Theorem stating that the given numbers of sheep satisfy the problem conditions -/
theorem shepherd_sheep_problem :
  (x - 4 = y + 4) ∧ (x + 4 = 3 * (y - 4)) := by
  sorry

#check shepherd_sheep_problem

end NUMINAMATH_CALUDE_shepherd_sheep_problem_l3541_354177


namespace NUMINAMATH_CALUDE_three_primes_product_sum_l3541_354150

theorem three_primes_product_sum : 
  ∃! (p q r : ℕ), 
    Prime p ∧ Prime q ∧ Prime r ∧ 
    p < q ∧ q < r ∧
    p * q * r = 5 * (p + q + r) ∧
    p = 2 ∧ q = 5 ∧ r = 7 := by
  sorry

end NUMINAMATH_CALUDE_three_primes_product_sum_l3541_354150


namespace NUMINAMATH_CALUDE_nonagon_diagonals_l3541_354163

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A nonagon is a polygon with 9 sides -/
def nonagon_sides : ℕ := 9

theorem nonagon_diagonals : num_diagonals nonagon_sides = 27 := by
  sorry

end NUMINAMATH_CALUDE_nonagon_diagonals_l3541_354163


namespace NUMINAMATH_CALUDE_repeating_decimal_fraction_sum_l3541_354176

-- Define the repeating decimal
def repeating_decimal : ℚ := 7 + 17 / 990

-- Theorem statement
theorem repeating_decimal_fraction_sum :
  (repeating_decimal = 710 / 99) ∧
  (710 + 99 = 809) :=
by sorry

end NUMINAMATH_CALUDE_repeating_decimal_fraction_sum_l3541_354176


namespace NUMINAMATH_CALUDE_trick_success_iff_edge_position_l3541_354135

/-- Represents a deck of cards with a specific card of interest -/
structure Deck :=
  (size : ℕ)
  (special_card_pos : ℕ)
  (h_pos : special_card_pos > 0 ∧ special_card_pos ≤ size)

/-- Represents the result of the card trick -/
inductive TrickResult
  | Success
  | Failure

/-- Function that simulates the card trick process -/
def perform_trick (d : Deck) : TrickResult :=
  sorry

/-- Theorem stating that the trick succeeds if and only if the special card is at an edge -/
theorem trick_success_iff_edge_position (d : Deck) :
  perform_trick d = TrickResult.Success ↔ d.special_card_pos = 1 ∨ d.special_card_pos = d.size :=
sorry

end NUMINAMATH_CALUDE_trick_success_iff_edge_position_l3541_354135


namespace NUMINAMATH_CALUDE_sandwich_availability_l3541_354114

/-- Given an initial number of sandwich kinds and a number of sold-out sandwich kinds,
    prove that the current number of available sandwich kinds is their difference. -/
theorem sandwich_availability (initial : ℕ) (sold_out : ℕ) (h : sold_out ≤ initial) :
  initial - sold_out = initial - sold_out :=
by sorry

end NUMINAMATH_CALUDE_sandwich_availability_l3541_354114


namespace NUMINAMATH_CALUDE_ellipse_hyperbola_tangent_l3541_354171

/-- The value of n for which the ellipse 4x^2 + y^2 = 4 and the hyperbola x^2 - n(y-1)^2 = 1 are tangent -/
theorem ellipse_hyperbola_tangent : 
  ∃ (n : ℝ), 
    (∀ (x y : ℝ), 4 * x^2 + y^2 = 4 ∧ x^2 - n * (y - 1)^2 = 1) →
    (∃! (x₀ y₀ : ℝ), 4 * x₀^2 + y₀^2 = 4 ∧ x₀^2 - n * (y₀ - 1)^2 = 1) →
    n = 3 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_hyperbola_tangent_l3541_354171


namespace NUMINAMATH_CALUDE_test_probabilities_l3541_354100

/-- Probability of A passing the test -/
def prob_A : ℝ := 0.8

/-- Probability of B passing the test -/
def prob_B : ℝ := 0.6

/-- Probability of C passing the test -/
def prob_C : ℝ := 0.5

/-- Probability that all three pass the test -/
def prob_all_pass : ℝ := prob_A * prob_B * prob_C

/-- Probability that at least one passes the test -/
def prob_at_least_one_pass : ℝ := 1 - (1 - prob_A) * (1 - prob_B) * (1 - prob_C)

theorem test_probabilities :
  prob_all_pass = 0.24 ∧ prob_at_least_one_pass = 0.96 := by
  sorry

end NUMINAMATH_CALUDE_test_probabilities_l3541_354100


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3541_354149

/-- An arithmetic sequence with sum of first n terms S_n -/
structure ArithmeticSequence where
  S : ℕ → ℝ  -- Sum function
  is_arithmetic : ∀ n : ℕ, S (n + 2) - S (n + 1) = S (n + 1) - S n

/-- Theorem: If S_10 = 12 and S_20 = 17, then S_30 = 22 for an arithmetic sequence -/
theorem arithmetic_sequence_sum (seq : ArithmeticSequence) 
  (h1 : seq.S 10 = 12) (h2 : seq.S 20 = 17) : seq.S 30 = 22 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3541_354149


namespace NUMINAMATH_CALUDE_largest_inclination_angle_l3541_354186

-- Define the inclination angle function
noncomputable def inclinationAngle (m : ℝ) : ℝ := Real.arctan m

-- Define the lines
def line1 (x : ℝ) : ℝ := -x + 1
def line2 (x : ℝ) : ℝ := x + 1
def line3 (x : ℝ) : ℝ := 2*x + 1
def line4 : ℝ → Prop := λ x => x = 1

-- Theorem statement
theorem largest_inclination_angle :
  ∀ (θ1 θ2 θ3 θ4 : ℝ),
    θ1 = inclinationAngle (-1) →
    θ2 = inclinationAngle 1 →
    θ3 = inclinationAngle 2 →
    θ4 = Real.pi / 2 →
    θ1 > θ2 ∧ θ1 > θ3 ∧ θ1 > θ4 :=
sorry

end NUMINAMATH_CALUDE_largest_inclination_angle_l3541_354186


namespace NUMINAMATH_CALUDE_min_value_ratio_l3541_354111

theorem min_value_ratio (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  (∃ (x y : ℝ), x > y ∧ y > 0 ∧ 
    2*x + y + 1/(x-y) + 4/(x+2*y) < 2*a + b + 1/(a-b) + 4/(a+2*b)) ∨
  a/b = 4 := by
sorry

end NUMINAMATH_CALUDE_min_value_ratio_l3541_354111


namespace NUMINAMATH_CALUDE_not_divisible_by_5_and_7_count_count_less_than_1000_l3541_354107

theorem not_divisible_by_5_and_7_count : Nat → Nat
  | n => (n + 1) - (n / 5 + n / 7 - n / 35)

theorem count_less_than_1000 :
  not_divisible_by_5_and_7_count 999 = 686 := by
  sorry

end NUMINAMATH_CALUDE_not_divisible_by_5_and_7_count_count_less_than_1000_l3541_354107


namespace NUMINAMATH_CALUDE_hexagon_largest_angle_l3541_354144

/-- A convex hexagon with interior angles as consecutive integers has its largest angle equal to 122° -/
theorem hexagon_largest_angle : ∀ (a b c d e f : ℕ),
  -- The angles are natural numbers
  -- The angles are consecutive integers
  b = a + 1 ∧ c = a + 2 ∧ d = a + 3 ∧ e = a + 4 ∧ f = a + 5 →
  -- The sum of interior angles of a hexagon is 720°
  a + b + c + d + e + f = 720 →
  -- The largest angle is 122°
  f = 122 := by
sorry

end NUMINAMATH_CALUDE_hexagon_largest_angle_l3541_354144


namespace NUMINAMATH_CALUDE_cookies_in_class_l3541_354192

/-- The number of cookies brought by Mona, Jasmine, Rachel, and Carlos -/
def totalCookies (mona jasmine rachel carlos : ℕ) : ℕ :=
  mona + jasmine + rachel + carlos

/-- Theorem stating the total number of cookies brought to class -/
theorem cookies_in_class :
  ∀ (mona jasmine rachel carlos : ℕ),
  mona = 20 →
  jasmine = mona - 5 →
  rachel = jasmine + 10 →
  carlos = rachel * 2 →
  totalCookies mona jasmine rachel carlos = 110 := by
  sorry

end NUMINAMATH_CALUDE_cookies_in_class_l3541_354192


namespace NUMINAMATH_CALUDE_reverse_digits_problem_l3541_354104

/-- Given two two-digit numbers where the second is the reverse of the first,
    if their quotient is 1.75 and the product of the first with its tens digit
    is 3.5 times the second, then the numbers are 21 and 12. -/
theorem reverse_digits_problem (x y : ℕ) : 
  10 ≤ x ∧ x < 100 ∧  -- x is a two-digit number
  10 ≤ y ∧ y < 100 ∧  -- y is a two-digit number
  y = (x % 10) * 10 + (x / 10) ∧  -- y is the reverse of x
  (x : ℚ) / y = 1.75 ∧  -- their quotient is 1.75
  x * (x / 10) = (7 * y) / 2  -- product of x and its tens digit is 3.5 times y
  → x = 21 ∧ y = 12 := by
sorry

end NUMINAMATH_CALUDE_reverse_digits_problem_l3541_354104


namespace NUMINAMATH_CALUDE_first_term_of_sequence_l3541_354113

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  -- The sum of the first 30 terms
  sum_30 : ℚ
  -- The sum of terms from 31st to 80th
  sum_31_to_80 : ℚ
  -- Property: The sum of the first 30 terms is 300
  sum_30_eq : sum_30 = 300
  -- Property: The sum of terms from 31st to 80th is 3750
  sum_31_to_80_eq : sum_31_to_80 = 3750

/-- The first term of the arithmetic sequence is -217/16 -/
theorem first_term_of_sequence (seq : ArithmeticSequence) : 
  ∃ (a d : ℚ), a = -217/16 ∧ 
  (∀ n : ℕ, n > 0 → n ≤ 30 → seq.sum_30 = (n/2) * (2*a + (n-1)*d)) ∧
  (seq.sum_31_to_80 = 25 * (2*a + 109*d)) :=
sorry

end NUMINAMATH_CALUDE_first_term_of_sequence_l3541_354113


namespace NUMINAMATH_CALUDE_changhyeon_money_problem_l3541_354169

theorem changhyeon_money_problem (initial_money : ℕ) : 
  (initial_money / 2 - 300) / 2 - 400 = 0 → initial_money = 2200 := by
  sorry

end NUMINAMATH_CALUDE_changhyeon_money_problem_l3541_354169


namespace NUMINAMATH_CALUDE_quadratic_equation_solutions_l3541_354175

theorem quadratic_equation_solutions
  (a b c : ℝ)
  (h1 : a + b + c = 0)
  (h2 : a - b + c = 0)
  (h3 : a ≠ 0) :
  ∀ x : ℝ, a * x^2 + b * x + c = 0 ↔ x = 1 ∨ x = -1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solutions_l3541_354175


namespace NUMINAMATH_CALUDE_gcf_72_108_l3541_354143

theorem gcf_72_108 : Nat.gcd 72 108 = 36 := by
  sorry

end NUMINAMATH_CALUDE_gcf_72_108_l3541_354143


namespace NUMINAMATH_CALUDE_sqrt_90000_equals_300_l3541_354182

theorem sqrt_90000_equals_300 : Real.sqrt 90000 = 300 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_90000_equals_300_l3541_354182


namespace NUMINAMATH_CALUDE_inequality_solution_l3541_354106

theorem inequality_solution (x : ℝ) : 
  (x * (x + 2)) / ((x - 5)^2) ≥ 15 ↔ 
  (x ≥ 5/2 ∧ x < 5) ∨ (x > 5 ∧ x ≤ 75/7) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l3541_354106


namespace NUMINAMATH_CALUDE_inequality_solution_l3541_354121

theorem inequality_solution (x : ℝ) : 
  (1 / (x * (x - 1)) - 1 / ((x - 1) * (x - 2)) > 1 / 5) ↔ 
  (0 < x ∧ x < 1) ∨ (1 < x ∧ x < 2) := by sorry

end NUMINAMATH_CALUDE_inequality_solution_l3541_354121


namespace NUMINAMATH_CALUDE_log_intersects_x_axis_l3541_354145

theorem log_intersects_x_axis : ∃ x : ℝ, x > 0 ∧ Real.log x = 0 := by
  sorry

end NUMINAMATH_CALUDE_log_intersects_x_axis_l3541_354145


namespace NUMINAMATH_CALUDE_max_min_product_l3541_354115

theorem max_min_product (a b c : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_sum : a + b + c = 12) (h_prod_sum : a * b + b * c + c * a = 30) :
  ∃ (m : ℝ), m = min (a * b) (min (b * c) (c * a)) ∧ m ≤ 2 ∧ 
  ∀ (m' : ℝ), m' = min (a * b) (min (b * c) (c * a)) → m' ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_max_min_product_l3541_354115


namespace NUMINAMATH_CALUDE_triangle_theorem_l3541_354140

-- Define the triangle ABC
structure Triangle (A B C : ℝ × ℝ) : Prop where
  right_angle_at_A : (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0

-- Define the circle
def Circle (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Define the theorem
theorem triangle_theorem (A B C D E : ℝ × ℝ) :
  Triangle A B C →
  D ∈ Circle A (Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)) →
  E ∈ Circle A (Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)) →
  D.1 = C.1 ∧ D.2 = C.2 →
  E.1 = C.1 ∧ E.2 = C.2 →
  Real.sqrt ((D.1 - B.1)^2 + (D.2 - B.2)^2) = 20 →
  Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2) = 16 →
  (C.1 - A.1)^2 + (C.2 - A.2)^2 = 936 := by
sorry


end NUMINAMATH_CALUDE_triangle_theorem_l3541_354140


namespace NUMINAMATH_CALUDE_height_of_equilateral_triangle_l3541_354191

/-- An equilateral triangle with base 2 and an inscribed circle --/
structure EquilateralTriangleWithInscribedCircle where
  /-- The base of the triangle --/
  base : ℝ
  /-- The height of the triangle --/
  height : ℝ
  /-- The radius of the inscribed circle --/
  radius : ℝ
  /-- The base is 2 --/
  base_eq_two : base = 2
  /-- The radius is half the height --/
  radius_half_height : radius = height / 2

/-- The height of an equilateral triangle with base 2 and an inscribed circle is √3 --/
theorem height_of_equilateral_triangle
  (triangle : EquilateralTriangleWithInscribedCircle) :
  triangle.height = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_height_of_equilateral_triangle_l3541_354191


namespace NUMINAMATH_CALUDE_number_difference_l3541_354147

theorem number_difference (x y : ℝ) (sum_eq : x + y = 42) (prod_eq : x * y = 437) :
  |x - y| = 4 := by
sorry

end NUMINAMATH_CALUDE_number_difference_l3541_354147


namespace NUMINAMATH_CALUDE_investment_growth_l3541_354167

-- Define the initial investment
def initial_investment : ℝ := 359

-- Define the interest rate
def interest_rate : ℝ := 0.12

-- Define the number of years
def years : ℕ := 3

-- Define the final amount
def final_amount : ℝ := 504.32

-- Theorem statement
theorem investment_growth :
  initial_investment * (1 + interest_rate) ^ years = final_amount := by
  sorry

end NUMINAMATH_CALUDE_investment_growth_l3541_354167


namespace NUMINAMATH_CALUDE_max_sum_of_squares_l3541_354126

theorem max_sum_of_squares (m n : ℕ) : 
  m ∈ Finset.range 1982 ∧ 
  n ∈ Finset.range 1982 ∧ 
  (n^2 - m*n - m^2)^2 = 1 →
  m^2 + n^2 ≤ 3524578 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_squares_l3541_354126


namespace NUMINAMATH_CALUDE_three_numbers_sum_l3541_354154

theorem three_numbers_sum (a b c : ℝ) : 
  a ≤ b ∧ b ≤ c ∧  -- Ordering of numbers
  b = 10 ∧  -- Median is 10
  (a + b + c) / 3 = a + 20 ∧  -- Mean is 20 more than least
  (a + b + c) / 3 = c - 25  -- Mean is 25 less than greatest
  → a + b + c = 45 := by
  sorry

end NUMINAMATH_CALUDE_three_numbers_sum_l3541_354154
