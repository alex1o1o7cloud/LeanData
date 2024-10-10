import Mathlib

namespace cube_of_0_09_times_0_0007_l22_2249

theorem cube_of_0_09_times_0_0007 : (0.09 : ℝ)^3 * 0.0007 = 0.0000005103 := by
  sorry

end cube_of_0_09_times_0_0007_l22_2249


namespace j_travel_time_l22_2205

/-- Given two travelers J and L, where:
  * J takes 45 minutes less time than L to travel 45 miles
  * J travels 1/2 mile per hour faster than L
  * y is J's rate of speed in miles per hour
Prove that J's time to travel 45 miles is equal to 45/y -/
theorem j_travel_time (y : ℝ) (h1 : y > 0) : ∃ (t_j t_l : ℝ),
  t_j = 45 / y ∧
  t_l = 45 / (y - 1/2) ∧
  t_l - t_j = 3/4 :=
sorry

end j_travel_time_l22_2205


namespace july_birth_percentage_l22_2230

/-- The percentage of scientists born in July, given the total number of scientists and the number born in July. -/
theorem july_birth_percentage 
  (total_scientists : ℕ) 
  (july_births : ℕ) 
  (h1 : total_scientists = 200) 
  (h2 : july_births = 17) : 
  (july_births : ℚ) / total_scientists * 100 = 8.5 := by
sorry

end july_birth_percentage_l22_2230


namespace parallel_perpendicular_implication_l22_2246

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel and perpendicular relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)

-- State the theorem
theorem parallel_perpendicular_implication 
  (m n : Line) (a : Plane) : 
  parallel m n → perpendicular m a → perpendicular n a :=
sorry

end parallel_perpendicular_implication_l22_2246


namespace inverse_sum_equals_negative_six_l22_2228

-- Define the function f(x) = x|x|
def f (x : ℝ) : ℝ := x * |x|

-- Define the inverse function f^(-1)
noncomputable def f_inv (y : ℝ) : ℝ :=
  if y ≥ 0 then Real.sqrt y else -Real.sqrt (-y)

-- Theorem statement
theorem inverse_sum_equals_negative_six :
  f_inv 9 + f_inv (-81) = -6 := by
  sorry

end inverse_sum_equals_negative_six_l22_2228


namespace largest_integer_inequality_l22_2210

theorem largest_integer_inequality (x : ℤ) : x ≤ 4 ↔ x / 3 + 3 / 4 < 7 / 3 := by
  sorry

end largest_integer_inequality_l22_2210


namespace todays_production_l22_2294

theorem todays_production (n : ℕ) (past_average : ℝ) (new_average : ℝ) 
  (h1 : n = 9)
  (h2 : past_average = 50)
  (h3 : new_average = 54) :
  (n + 1) * new_average - n * past_average = 90 := by
  sorry

end todays_production_l22_2294


namespace log_problem_l22_2233

theorem log_problem (y : ℝ) (h : y = (Real.log 16 / Real.log 4) ^ (Real.log 4 / Real.log 16)) :
  Real.log y / Real.log 12 = 1 / (4 + 2 * Real.log 3 / Real.log 2) := by
  sorry

end log_problem_l22_2233


namespace complex_sum_problem_l22_2202

theorem complex_sum_problem (a b c d e f : ℝ) : 
  b = 4 →
  e = 2 * (-a - c) →
  Complex.mk (a + c + e) (b + d + f) = Complex.I * 6 →
  d + f = 2 := by
sorry

end complex_sum_problem_l22_2202


namespace fraction_pure_fuji_l22_2206

-- Define the total number of trees
def total_trees : ℕ := 180

-- Define the number of pure Fuji trees
def pure_fuji : ℕ := 135

-- Define the number of pure Gala trees
def pure_gala : ℕ := 27

-- Define the number of cross-pollinated trees
def cross_pollinated : ℕ := 18

-- Define the cross-pollination rate
def cross_pollination_rate : ℚ := 1/10

-- Theorem stating the fraction of pure Fuji trees
theorem fraction_pure_fuji :
  (pure_fuji : ℚ) / total_trees = 3/4 :=
by
  sorry

-- Conditions from the problem
axiom condition1 : pure_fuji + cross_pollinated = 153
axiom condition2 : (cross_pollinated : ℚ) / total_trees = cross_pollination_rate
axiom condition3 : total_trees = pure_fuji + pure_gala + cross_pollinated

end fraction_pure_fuji_l22_2206


namespace book_price_increase_l22_2218

theorem book_price_increase (original_price : ℝ) : 
  original_price * (1 + 0.6) = 480 → original_price = 300 := by
  sorry

end book_price_increase_l22_2218


namespace function_properties_l22_2201

def isEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)
def isOdd (f : ℝ → ℝ) : Prop := ∀ x, f x = -f (-x)

theorem function_properties (f : ℝ → ℝ) 
  (h1 : isEven (fun x ↦ f (x - 3)))
  (h2 : isOdd (fun x ↦ f (2 * x - 1))) :
  (f (-1) = 0) ∧ 
  (∀ x, f x = f (-x - 6)) ∧ 
  (f 7 = 0) := by
sorry

end function_properties_l22_2201


namespace even_periodic_function_derivative_zero_l22_2235

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem even_periodic_function_derivative_zero
  (f : ℝ → ℝ)
  (h_even : is_even f)
  (h_diff : Differentiable ℝ f)
  (h_period : has_period f 5) :
  deriv f 5 = 0 := by
  sorry

end even_periodic_function_derivative_zero_l22_2235


namespace geometric_series_constant_l22_2288

/-- A geometric series with sum of first n terms given by S_n = 3^(n+1) + a -/
def GeometricSeries (a : ℝ) : ℕ → ℝ := fun n ↦ 3^(n+1) + a

/-- The sum of the first n terms of the geometric series -/
def SeriesSum (a : ℝ) : ℕ → ℝ := fun n ↦ GeometricSeries a n

theorem geometric_series_constant (a : ℝ) : a = -3 :=
  sorry

end geometric_series_constant_l22_2288


namespace quadratic_real_roots_condition_specific_root_implies_m_and_other_root_l22_2279

/-- Given a quadratic equation x^2 + 2x - (m-2) = 0 with real roots -/
def quadratic_equation (x m : ℝ) : Prop := x^2 + 2*x - (m-2) = 0

/-- The discriminant of the quadratic equation is non-negative -/
def has_real_roots (m : ℝ) : Prop := 4*m - 4 ≥ 0

theorem quadratic_real_roots_condition (m : ℝ) :
  has_real_roots m ↔ m ≥ 1 := by sorry

theorem specific_root_implies_m_and_other_root :
  ∀ m : ℝ, quadratic_equation 1 m → m = 3 ∧ quadratic_equation (-3) m := by sorry

end quadratic_real_roots_condition_specific_root_implies_m_and_other_root_l22_2279


namespace jungkook_balls_left_l22_2252

/-- The number of balls left in a box after removing some balls -/
def balls_left (initial : ℕ) (removed : ℕ) : ℕ :=
  initial - removed

/-- Theorem: When Jungkook removes 3 balls from a box containing 10 balls, 7 balls are left -/
theorem jungkook_balls_left : balls_left 10 3 = 7 := by
  sorry

end jungkook_balls_left_l22_2252


namespace largest_n_base_7_double_l22_2275

def to_base_7 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
    if m = 0 then acc else aux (m / 7) ((m % 7) :: acc)
  aux n []

def from_base_7 (digits : List ℕ) : ℕ :=
  digits.foldl (fun acc d => acc * 7 + d) 0

theorem largest_n_base_7_double : ∀ n : ℕ, n > 156 → 2 * n ≠ from_base_7 (to_base_7 n) :=
sorry

end largest_n_base_7_double_l22_2275


namespace johns_share_l22_2248

theorem johns_share (total_amount : ℕ) (john_ratio jose_ratio binoy_ratio : ℕ) 
  (h1 : total_amount = 6000)
  (h2 : john_ratio = 2)
  (h3 : jose_ratio = 4)
  (h4 : binoy_ratio = 6) :
  (john_ratio : ℚ) / (john_ratio + jose_ratio + binoy_ratio : ℚ) * total_amount = 1000 :=
by sorry

end johns_share_l22_2248


namespace berts_money_l22_2213

-- Define Bert's initial amount of money
variable (n : ℚ)

-- Define the remaining money after each step
def remaining_after_hardware (n : ℚ) : ℚ := (3/4) * n
def remaining_after_dry_cleaners (n : ℚ) : ℚ := remaining_after_hardware n - 9
def remaining_after_grocery (n : ℚ) : ℚ := (1/2) * remaining_after_dry_cleaners n
def remaining_after_books (n : ℚ) : ℚ := (2/3) * remaining_after_grocery n
def final_remaining (n : ℚ) : ℚ := (4/5) * remaining_after_books n

-- Theorem stating the relationship between n and the final amount
theorem berts_money (n : ℚ) : final_remaining n = 27 ↔ n = 72 := by sorry

end berts_money_l22_2213


namespace two_talent_students_l22_2262

theorem two_talent_students (total : ℕ) (all_three : ℕ) (cant_sing : ℕ) (cant_dance : ℕ) (cant_act : ℕ) : 
  total = 150 →
  all_three = 10 →
  cant_sing = 70 →
  cant_dance = 90 →
  cant_act = 50 →
  ∃ (two_talents : ℕ), two_talents = 80 ∧ 
    (total - cant_sing) + (total - cant_dance) + (total - cant_act) - two_talents - 2 * all_three = total :=
by sorry

end two_talent_students_l22_2262


namespace sector_circumference_l22_2241

/-- Given a circular sector with area 2 and central angle 4 radians, 
    its circumference is 6. -/
theorem sector_circumference (area : ℝ) (angle : ℝ) (circumference : ℝ) : 
  area = 2 → angle = 4 → circumference = 6 := by
  sorry

end sector_circumference_l22_2241


namespace jake_weight_proof_l22_2295

/-- Jake's present weight in pounds -/
def jake_weight : ℝ := 196

/-- Jake's sister's weight in pounds -/
def sister_weight : ℝ := 290 - jake_weight

theorem jake_weight_proof :
  (jake_weight - 8 = 2 * sister_weight) ∧
  (jake_weight + sister_weight = 290) :=
by sorry

end jake_weight_proof_l22_2295


namespace expected_faces_six_die_six_rolls_l22_2273

/-- The number of sides on the die -/
def n : ℕ := 6

/-- The number of rolls -/
def k : ℕ := 6

/-- The probability of a specific face not appearing in a single roll -/
def p : ℚ := (n - 1) / n

/-- The expected number of different faces appearing when rolling an n-sided die k times -/
def expected_faces : ℚ := n * (1 - p^k)

/-- Theorem: The expected number of different faces appearing when a fair six-sided die 
    is rolled six times is equal to (6^6 - 5^6) / 6^5 -/
theorem expected_faces_six_die_six_rolls : 
  expected_faces = (n^k - (n-1)^k) / n^(k-1) := by
  sorry

#eval expected_faces

end expected_faces_six_die_six_rolls_l22_2273


namespace three_queens_or_at_least_one_jack_probability_l22_2240

/-- The number of cards in a standard deck -/
def deck_size : ℕ := 52

/-- The number of Jacks in a standard deck -/
def num_jacks : ℕ := 4

/-- The number of Queens in a standard deck -/
def num_queens : ℕ := 4

/-- The number of cards drawn -/
def cards_drawn : ℕ := 3

/-- The probability of drawing either three queens or at least one jack -/
def probability : ℚ := 142 / 1105

theorem three_queens_or_at_least_one_jack_probability :
  let total_combinations := (deck_size.choose cards_drawn : ℚ)
  let three_queens_prob := (num_queens.choose cards_drawn : ℚ) / total_combinations
  let at_least_one_jack_prob := 1 - ((deck_size - num_jacks).choose cards_drawn : ℚ) / total_combinations
  three_queens_prob + at_least_one_jack_prob - (three_queens_prob * at_least_one_jack_prob) = probability :=
by sorry

end three_queens_or_at_least_one_jack_probability_l22_2240


namespace focus_of_parabola_l22_2211

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- State the theorem
theorem focus_of_parabola :
  ∃ (f : ℝ × ℝ), f = (1, 0) ∧ 
  (∀ (x y : ℝ), parabola x y → 
    (x - f.1)^2 + y^2 = (x - f.1 + f.1)^2) :=
sorry

end focus_of_parabola_l22_2211


namespace min_value_of_log_expression_four_is_minimum_l22_2278

theorem min_value_of_log_expression (x y : ℝ) (hx : x > 1) (hy : y > 1) :
  (Real.log 2011 / Real.log x + Real.log 2011 / Real.log y) / (Real.log 2011 / (Real.log x + Real.log y)) ≥ 4 :=
by sorry

theorem four_is_minimum (ε : ℝ) (hε : ε > 0) :
  ∃ x y : ℝ, x > 1 ∧ y > 1 ∧
  (Real.log 2011 / Real.log x + Real.log 2011 / Real.log y) / (Real.log 2011 / (Real.log x + Real.log y)) < 4 + ε :=
by sorry

end min_value_of_log_expression_four_is_minimum_l22_2278


namespace rubber_boat_lost_at_4pm_l22_2264

/-- Represents the time when the rubber boat was lost (in hours before 5 PM) -/
def time_lost : ℝ := 1

/-- Represents the speed of the ship in still water -/
def ship_speed : ℝ := 1

/-- Represents the speed of the river flow -/
def river_speed : ℝ := 1

/-- Theorem stating that the rubber boat was lost at 4 PM -/
theorem rubber_boat_lost_at_4pm :
  (5 - time_lost) * (ship_speed - river_speed) + (6 - time_lost) * river_speed = ship_speed + river_speed :=
by sorry

end rubber_boat_lost_at_4pm_l22_2264


namespace opposite_sides_line_range_l22_2203

theorem opposite_sides_line_range (a : ℝ) : 
  (3 * 3 - 2 * 1 + a) * (3 * (-4) - 2 * 6 + a) < 0 ↔ -7 < a ∧ a < 24 :=
sorry

end opposite_sides_line_range_l22_2203


namespace silverware_reduction_l22_2224

theorem silverware_reduction (initial_per_type : ℕ) (num_types : ℕ) (total_purchased : ℕ) :
  initial_per_type = 15 →
  num_types = 4 →
  total_purchased = 44 →
  (initial_per_type * num_types - total_purchased) / num_types = 4 :=
by
  sorry

end silverware_reduction_l22_2224


namespace reflection_line_sum_l22_2229

/-- Given a line y = mx + b, where the point (2,3) is reflected to (8,7) across this line,
    prove that m + b = 9.5 -/
theorem reflection_line_sum (m b : ℝ) : 
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    x₁ = 2 ∧ y₁ = 3 ∧ x₂ = 8 ∧ y₂ = 7 ∧
    (y₂ - y₁) / (x₂ - x₁) = -1 / m ∧
    ((x₁ + x₂) / 2, (y₁ + y₂) / 2) ∈ {(x, y) | y = m * x + b}) →
  m + b = 9.5 := by
sorry


end reflection_line_sum_l22_2229


namespace factorization_x4_minus_81_l22_2234

theorem factorization_x4_minus_81 : 
  ∀ x : ℝ, x^4 - 81 = (x - 3) * (x + 3) * (x^2 + 9) :=
by
  sorry

end factorization_x4_minus_81_l22_2234


namespace salt_mixing_theorem_l22_2293

def salt_mixing_problem (x : ℚ) : Prop :=
  let known_salt_weight : ℚ := 40
  let known_salt_price : ℚ := 25 / 100
  let unknown_salt_weight : ℚ := 60
  let total_weight : ℚ := known_salt_weight + unknown_salt_weight
  let selling_price : ℚ := 48 / 100
  let profit_percentage : ℚ := 20 / 100
  let total_cost : ℚ := known_salt_weight * known_salt_price + unknown_salt_weight * x
  let selling_revenue : ℚ := total_weight * selling_price
  selling_revenue = total_cost * (1 + profit_percentage) ∧ x = 50 / 100

theorem salt_mixing_theorem : ∃ x : ℚ, salt_mixing_problem x :=
  sorry

end salt_mixing_theorem_l22_2293


namespace gcd_30_and_70_to_80_l22_2239

theorem gcd_30_and_70_to_80 : ∃! n : ℕ, 70 ≤ n ∧ n ≤ 80 ∧ Nat.gcd 30 n = 10 := by sorry

end gcd_30_and_70_to_80_l22_2239


namespace sequence_existence_l22_2266

theorem sequence_existence (a b : ℤ) (ha : a > 2) (hb : b > 2) :
  ∃ (k : ℕ) (n : ℕ → ℤ), 
    n 1 = a ∧ 
    n k = b ∧ 
    (∀ i : ℕ, 1 ≤ i ∧ i < k → (n i + n (i + 1)) ∣ (n i * n (i + 1))) :=
sorry

end sequence_existence_l22_2266


namespace total_soap_cost_two_years_l22_2253

/-- Represents the types of soap --/
inductive SoapType
  | Lavender
  | Lemon
  | Sandalwood

/-- Returns the price of a given soap type --/
def soapPrice (s : SoapType) : ℚ :=
  match s with
  | SoapType.Lavender => 4
  | SoapType.Lemon => 5
  | SoapType.Sandalwood => 6

/-- Applies the bulk discount to a given quantity and price --/
def applyDiscount (quantity : ℕ) (price : ℚ) : ℚ :=
  let totalPrice := price * quantity
  if quantity ≥ 10 then totalPrice * (1 - 0.15)
  else if quantity ≥ 7 then totalPrice * (1 - 0.10)
  else if quantity ≥ 4 then totalPrice * (1 - 0.05)
  else totalPrice

/-- Calculates the cost of soap for a given type over 2 years --/
def soapCostTwoYears (s : SoapType) : ℚ :=
  let price := soapPrice s
  applyDiscount 7 price + price

/-- Theorem: The total amount Elias spends on soap in 2 years is $109.50 --/
theorem total_soap_cost_two_years :
  soapCostTwoYears SoapType.Lavender +
  soapCostTwoYears SoapType.Lemon +
  soapCostTwoYears SoapType.Sandalwood = 109.5 := by
  sorry

end total_soap_cost_two_years_l22_2253


namespace average_weight_increase_l22_2287

theorem average_weight_increase (initial_count : ℕ) (old_weight new_weight : ℝ) :
  initial_count = 10 →
  old_weight = 65 →
  new_weight = 90 →
  (new_weight - old_weight) / initial_count = 2.5 := by
  sorry

end average_weight_increase_l22_2287


namespace exponent_multiplication_l22_2296

theorem exponent_multiplication (a : ℝ) : a^2 * a^3 = a^5 := by
  sorry

end exponent_multiplication_l22_2296


namespace article_price_decrease_l22_2227

theorem article_price_decrease (P : ℝ) : 
  (P * (1 - 0.24) * (1 - 0.10) = 760) → 
  ∃ ε > 0, |P - 111| < ε :=
sorry

end article_price_decrease_l22_2227


namespace trees_difference_l22_2267

theorem trees_difference (initial_trees : ℕ) (dead_trees : ℕ) 
  (h1 : initial_trees = 14) (h2 : dead_trees = 9) : 
  dead_trees - (initial_trees - dead_trees) = 4 := by
  sorry

end trees_difference_l22_2267


namespace smallest_square_area_l22_2245

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Calculates the minimum side length of a square that can contain two rectangles -/
def minSquareSide (r1 r2 : Rectangle) : ℕ :=
  min (max r1.width r2.width + min r1.height r2.height)
      (max r1.height r2.height + min r1.width r2.width)

/-- The theorem stating the smallest possible area of the square -/
theorem smallest_square_area (r1 r2 : Rectangle)
  (h1 : r1 = ⟨3, 5⟩)
  (h2 : r2 = ⟨4, 6⟩) :
  (minSquareSide r1 r2) ^ 2 = 81 := by
  sorry

#eval (minSquareSide ⟨3, 5⟩ ⟨4, 6⟩) ^ 2

end smallest_square_area_l22_2245


namespace not_square_of_integer_l22_2281

theorem not_square_of_integer (n : ℕ+) : ¬ ∃ m : ℤ, m^2 = 2*(n.val^2 + 1) - n.val := by
  sorry

end not_square_of_integer_l22_2281


namespace cookies_received_l22_2276

theorem cookies_received (brother sister cousin self : ℕ) 
  (h1 : brother = 12)
  (h2 : sister = 9)
  (h3 : cousin = 7)
  (h4 : self = 17) :
  brother + sister + cousin + self = 45 := by
  sorry

end cookies_received_l22_2276


namespace square_perimeter_l22_2247

theorem square_perimeter (area : ℝ) (side : ℝ) (h1 : area = 36) (h2 : side^2 = area) :
  4 * side = 24 := by
  sorry

end square_perimeter_l22_2247


namespace trapezoid_has_two_heights_l22_2280

/-- A trapezoid is a quadrilateral with at least one pair of parallel sides. -/
structure Trapezoid where
  vertices : Fin 4 → ℝ × ℝ
  parallel_sides : ∃ (i j : Fin 4), i ≠ j ∧ (vertices i).1 = (vertices j).1

/-- The number of heights in a trapezoid -/
def num_heights (t : Trapezoid) : ℕ := 2

/-- Theorem: A trapezoid has exactly 2 heights -/
theorem trapezoid_has_two_heights (t : Trapezoid) : num_heights t = 2 := by
  sorry

end trapezoid_has_two_heights_l22_2280


namespace seventh_root_of_unity_product_l22_2244

theorem seventh_root_of_unity_product (r : ℂ) (h1 : r^7 = 1) (h2 : r ≠ 1) :
  (r - 1) * (r^2 - 1) * (r^3 - 1) * (r^4 - 1) * (r^5 - 1) * (r^6 - 1) = 8 := by
  sorry

end seventh_root_of_unity_product_l22_2244


namespace fifth_term_of_specific_sequence_l22_2291

/-- Represents an arithmetic sequence -/
structure ArithmeticSequence where
  first : ℝ
  second : ℝ

/-- Get the nth term of an arithmetic sequence -/
def nthTerm (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  seq.first + (n - 1) * (seq.second - seq.first)

/-- The theorem to prove -/
theorem fifth_term_of_specific_sequence :
  let seq := ArithmeticSequence.mk 3 8
  nthTerm seq 5 = 23 := by sorry

end fifth_term_of_specific_sequence_l22_2291


namespace min_value_reciprocal_sum_l22_2297

theorem min_value_reciprocal_sum (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h_sum : a + 3*b = 1) :
  1/a + 3/b ≥ 16 ∧ ∃ (a₀ b₀ : ℝ), 0 < a₀ ∧ 0 < b₀ ∧ a₀ + 3*b₀ = 1 ∧ 1/a₀ + 3/b₀ = 16 :=
sorry

end min_value_reciprocal_sum_l22_2297


namespace evaluate_power_l22_2256

theorem evaluate_power : (64 : ℝ) ^ (3/4) = 16 * Real.sqrt 2 := by
  sorry

end evaluate_power_l22_2256


namespace circle_symmetry_l22_2232

-- Define the original circle
def original_circle (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 1 = 0

-- Define the line of symmetry
def symmetry_line (x y : ℝ) : Prop :=
  2*x - y + 3 = 0

-- Define the symmetric circle
def symmetric_circle (x y : ℝ) : Prop :=
  (x+3)^2 + (y-2)^2 = 2

-- Theorem statement
theorem circle_symmetry :
  ∀ (x₁ y₁ x₂ y₂ : ℝ),
  original_circle x₁ y₁ →
  symmetric_circle x₂ y₂ →
  ∃ (x_m y_m : ℝ),
    symmetry_line x_m y_m ∧
    x_m = (x₁ + x₂) / 2 ∧
    y_m = (y₁ + y₂) / 2 :=
sorry

end circle_symmetry_l22_2232


namespace expected_total_rainfall_l22_2269

/-- Represents the weather forecast for a day --/
structure WeatherForecast where
  sun_chance : ℝ
  rain_chance1 : ℝ
  rain_amount1 : ℝ
  rain_chance2 : ℝ
  rain_amount2 : ℝ

/-- Calculates the expected rainfall for a given weather forecast --/
def expected_rainfall (forecast : WeatherForecast) : ℝ :=
  forecast.sun_chance * 0 + forecast.rain_chance1 * forecast.rain_amount1 + 
  forecast.rain_chance2 * forecast.rain_amount2

/-- The weather forecast for weekdays --/
def weekday_forecast : WeatherForecast := {
  sun_chance := 0.3,
  rain_chance1 := 0.2,
  rain_amount1 := 5,
  rain_chance2 := 0.5,
  rain_amount2 := 8
}

/-- The weather forecast for weekend days --/
def weekend_forecast : WeatherForecast := {
  sun_chance := 0.5,
  rain_chance1 := 0.25,
  rain_amount1 := 2,
  rain_chance2 := 0.25,
  rain_amount2 := 6
}

/-- The number of weekdays --/
def num_weekdays : ℕ := 5

/-- The number of weekend days --/
def num_weekend_days : ℕ := 2

theorem expected_total_rainfall : 
  (num_weekdays : ℝ) * expected_rainfall weekday_forecast + 
  (num_weekend_days : ℝ) * expected_rainfall weekend_forecast = 29 := by
  sorry

end expected_total_rainfall_l22_2269


namespace greatest_lower_bound_sum_squares_roots_l22_2251

/-- A monic polynomial of degree n with real coefficients -/
def MonicPoly (n : ℕ) := Polynomial ℝ

/-- The coefficient of x^(n-1) in a monic polynomial -/
def a_n_minus_1 (p : MonicPoly n) : ℝ := p.coeff (n - 1)

/-- The coefficient of x^(n-2) in a monic polynomial -/
def a_n_minus_2 (p : MonicPoly n) : ℝ := p.coeff (n - 2)

/-- The sum of squares of roots of a polynomial -/
noncomputable def sum_of_squares_of_roots (p : MonicPoly n) : ℝ := 
  (p.roots.map (λ r => r^2)).sum

/-- Theorem: The greatest lower bound on the sum of squares of roots -/
theorem greatest_lower_bound_sum_squares_roots (n : ℕ) (p : MonicPoly n) 
  (h : a_n_minus_1 p = 2 * a_n_minus_2 p) :
  ∃ (lb : ℝ), lb = (1/4 : ℝ) ∧ 
    ∀ (q : MonicPoly n), a_n_minus_1 q = 2 * a_n_minus_2 q → 
      lb ≤ sum_of_squares_of_roots q :=
by sorry

end greatest_lower_bound_sum_squares_roots_l22_2251


namespace triangle_area_l22_2221

/-- The area of a triangle with sides 5, 4, and 4 units is (5√39)/4 square units. -/
theorem triangle_area (a b c : ℝ) (h1 : a = 5) (h2 : b = 4) (h3 : c = 4) :
  (1/2 : ℝ) * a * (((b^2 - (a/2)^2).sqrt : ℝ)) = (5 * Real.sqrt 39) / 4 := by
  sorry

end triangle_area_l22_2221


namespace green_ball_probability_l22_2238

structure Container where
  red : ℕ
  green : ℕ

def Set1 : List Container := [
  ⟨2, 8⟩,  -- Container A
  ⟨8, 2⟩,  -- Container B
  ⟨8, 2⟩   -- Container C
]

def Set2 : List Container := [
  ⟨8, 2⟩,  -- Container A
  ⟨2, 8⟩,  -- Container B
  ⟨2, 8⟩   -- Container C
]

def probability_green (set : List Container) : ℚ :=
  let total_balls (c : Container) := c.red + c.green
  let green_prob (c : Container) := c.green / (total_balls c)
  (set.map green_prob).sum / set.length

theorem green_ball_probability :
  (1 / 2 : ℚ) * probability_green Set1 + (1 / 2 : ℚ) * probability_green Set2 = 1 / 2 := by
  sorry

end green_ball_probability_l22_2238


namespace commission_change_point_l22_2285

/-- The sales amount where the commission rate changes -/
def X : ℝ := 1822.98

/-- The total sales amount -/
def total_sales : ℝ := 15885.42

/-- The amount remitted to the parent company -/
def remitted_amount : ℝ := 15000

/-- The commission rate for sales up to X -/
def low_rate : ℝ := 0.10

/-- The commission rate for sales exceeding X -/
def high_rate : ℝ := 0.05

theorem commission_change_point : 
  X * low_rate + (total_sales - X) * high_rate = total_sales - remitted_amount :=
sorry

end commission_change_point_l22_2285


namespace rectangle_roots_l22_2268

def polynomial (b : ℝ) (z : ℂ) : ℂ :=
  z^4 - 10*z^3 + 16*b*z^2 - 2*(3*b^2 - 5*b + 4)*z + 6

def forms_rectangle (b : ℝ) : Prop :=
  ∃ (z₁ z₂ z₃ z₄ : ℂ),
    polynomial b z₁ = 0 ∧
    polynomial b z₂ = 0 ∧
    polynomial b z₃ = 0 ∧
    polynomial b z₄ = 0 ∧
    (z₁.re = z₂.re ∧ z₁.im = -z₂.im) ∧
    (z₃.re = z₄.re ∧ z₃.im = -z₄.im) ∧
    (z₁.re - z₃.re = z₂.im - z₄.im) ∧
    (z₁.im - z₃.im = z₄.re - z₂.re)

theorem rectangle_roots :
  ∀ b : ℝ, forms_rectangle b ↔ (b = 5/3 ∨ b = 2) :=
sorry

end rectangle_roots_l22_2268


namespace parallelogram_smallest_angle_l22_2289

theorem parallelogram_smallest_angle (a b c d : ℝ) : 
  -- Conditions
  a + b + c + d = 360 →  -- Sum of angles in a quadrilateral is 360°
  a = c →                -- Opposite angles are equal
  b = d →                -- Opposite angles are equal
  max a b - min a b = 100 →  -- Largest angle is 100° greater than smallest
  -- Conclusion
  min a b = 40 :=
by sorry

end parallelogram_smallest_angle_l22_2289


namespace handmade_ornaments_excess_l22_2286

/-- Proves that the number of handmade ornaments exceeds 1/6 of the total ornaments by 20 -/
theorem handmade_ornaments_excess (total : ℕ) (handmade : ℕ) (antique : ℕ) : 
  total = 60 →
  3 * antique = total →
  2 * antique = handmade →
  antique = 20 →
  handmade - (total / 6) = 20 := by
  sorry

end handmade_ornaments_excess_l22_2286


namespace product_75_180_trailing_zeros_l22_2292

/-- The number of trailing zeros in the product of two positive integers -/
def trailingZeros (a b : ℕ+) : ℕ :=
  sorry

/-- Theorem: The number of trailing zeros in the product of 75 and 180 is 2 -/
theorem product_75_180_trailing_zeros :
  trailingZeros 75 180 = 2 := by
  sorry

end product_75_180_trailing_zeros_l22_2292


namespace probability_sum_greater_than_7_l22_2204

/-- A bag containing cards numbered from 0 to 5 -/
def Bag : Finset ℕ := {0, 1, 2, 3, 4, 5}

/-- The sample space of drawing one card from each bag -/
def SampleSpace : Finset (ℕ × ℕ) := Bag.product Bag

/-- The event where the sum of two drawn cards is greater than 7 -/
def EventSumGreaterThan7 : Finset (ℕ × ℕ) :=
  SampleSpace.filter (fun p => p.1 + p.2 > 7)

/-- The probability of the event -/
def ProbabilityEventSumGreaterThan7 : ℚ :=
  EventSumGreaterThan7.card / SampleSpace.card

theorem probability_sum_greater_than_7 :
  ProbabilityEventSumGreaterThan7 = 1 / 6 := by
  sorry

end probability_sum_greater_than_7_l22_2204


namespace diamond_spade_ratio_l22_2255

structure Deck :=
  (clubs : ℕ)
  (diamonds : ℕ)
  (hearts : ℕ)
  (spades : ℕ)

def is_valid_deck (d : Deck) : Prop :=
  d.clubs + d.diamonds + d.hearts + d.spades = 13 ∧
  d.clubs + d.spades = 7 ∧
  d.diamonds + d.hearts = 6 ∧
  d.hearts = 2 * d.diamonds ∧
  d.clubs = 6

theorem diamond_spade_ratio (d : Deck) (h : is_valid_deck d) :
  d.diamonds = 2 ∧ d.spades = 1 :=
sorry

end diamond_spade_ratio_l22_2255


namespace stone_123_is_12_l22_2217

/-- Represents the counting pattern on a circle of stones -/
def stone_count (n : ℕ) : ℕ := 
  let cycle := 28
  n % cycle

/-- The original position of a stone given its count number -/
def original_position (count : ℕ) : ℕ :=
  if count ≤ 15 then count
  else 16 - (count - 15)

theorem stone_123_is_12 : original_position (stone_count 123) = 12 := by sorry

end stone_123_is_12_l22_2217


namespace smallest_equal_hotdogs_and_buns_l22_2237

theorem smallest_equal_hotdogs_and_buns :
  (∃ n : ℕ+, ∀ k : ℕ+, (∃ m : ℕ+, 6 * k = 8 * m) → n ≤ k) ∧
  (∃ m : ℕ+, 6 * 4 = 8 * m) :=
by sorry

end smallest_equal_hotdogs_and_buns_l22_2237


namespace xy_value_l22_2236

theorem xy_value (x y : ℝ) (h : |x - 1| + (x + y)^2 = 0) : x * y = -1 := by
  sorry

end xy_value_l22_2236


namespace road_cost_calculation_l22_2272

theorem road_cost_calculation (lawn_length lawn_width road_length_width road_width_width : ℕ)
  (cost_length cost_width : ℚ) : 
  lawn_length = 80 ∧ 
  lawn_width = 60 ∧ 
  road_length_width = 12 ∧ 
  road_width_width = 15 ∧ 
  cost_length = 3 ∧ 
  cost_width = (5/2) →
  (lawn_length * road_length_width * cost_length + 
   lawn_width * road_width_width * cost_width : ℚ) = 5130 :=
by sorry

end road_cost_calculation_l22_2272


namespace A_B_symmetric_about_x_axis_l22_2212

/-- Two points are symmetric about the x-axis if their x-coordinates are equal
    and their y-coordinates are opposite. -/
def symmetric_about_x_axis (A B : ℝ × ℝ) : Prop :=
  A.1 = B.1 ∧ A.2 = -B.2

/-- Point A in the coordinate plane -/
def A : ℝ × ℝ := (-1, 3)

/-- Point B in the coordinate plane -/
def B : ℝ × ℝ := (-1, -3)

/-- Theorem stating that points A and B are symmetric about the x-axis -/
theorem A_B_symmetric_about_x_axis :
  symmetric_about_x_axis A B := by sorry

end A_B_symmetric_about_x_axis_l22_2212


namespace fib_105_mod_7_l22_2219

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fib n + fib (n + 1)

/-- The period of Fibonacci sequence modulo 7 -/
def fib_mod7_period : ℕ := 16

theorem fib_105_mod_7 : fib 104 % 7 = 2 := by
  sorry

#eval fib 104 % 7

end fib_105_mod_7_l22_2219


namespace coin_stack_arrangements_l22_2284

/-- Represents a coin with a color and a face side -/
inductive Coin
  | Gold : Bool → Coin
  | Silver : Bool → Coin

/-- A stack of coins -/
def CoinStack := List Coin

/-- Checks if two adjacent coins are not face to face -/
def validAdjacent : Coin → Coin → Bool
  | Coin.Gold true, Coin.Gold true => false
  | Coin.Gold true, Coin.Silver true => false
  | Coin.Silver true, Coin.Gold true => false
  | Coin.Silver true, Coin.Silver true => false
  | _, _ => true

/-- Checks if a stack of coins is valid (no adjacent face to face) -/
def validStack : CoinStack → Bool
  | [] => true
  | [_] => true
  | (c1 :: c2 :: rest) => validAdjacent c1 c2 && validStack (c2 :: rest)

/-- Counts the number of gold coins in a stack -/
def countGold : CoinStack → Nat
  | [] => 0
  | (Coin.Gold _) :: rest => 1 + countGold rest
  | _ :: rest => countGold rest

/-- Counts the number of silver coins in a stack -/
def countSilver : CoinStack → Nat
  | [] => 0
  | (Coin.Silver _) :: rest => 1 + countSilver rest
  | _ :: rest => countSilver rest

/-- The main theorem to prove -/
theorem coin_stack_arrangements :
  (∃ (validStacks : List CoinStack),
    (∀ s ∈ validStacks, validStack s = true) ∧
    (∀ s ∈ validStacks, countGold s = 5) ∧
    (∀ s ∈ validStacks, countSilver s = 5) ∧
    validStacks.length = 2772) := by
  sorry

end coin_stack_arrangements_l22_2284


namespace intersection_of_complement_and_Q_l22_2270

-- Define the sets P and Q
def P : Set ℝ := {x | x - 1 ≤ 0}
def Q : Set ℝ := {x | 0 < x ∧ x ≤ 2}

-- Define the complement of P in ℝ
def C_R_P : Set ℝ := {x | ¬(x ∈ P)}

-- State the theorem
theorem intersection_of_complement_and_Q : 
  (C_R_P ∩ Q) = {x : ℝ | 1 < x ∧ x ≤ 2} := by
  sorry

end intersection_of_complement_and_Q_l22_2270


namespace initial_position_of_moving_point_l22_2290

theorem initial_position_of_moving_point (M : ℝ) : 
  (M - 7) + 4 = 0 → M = 3 := by
  sorry

end initial_position_of_moving_point_l22_2290


namespace carl_weekly_earnings_l22_2220

/-- Represents Carl's earnings and candy bar purchases over 4 weeks -/
structure CarlEarnings where
  weeks : ℕ
  candyBars : ℕ
  candyBarPrice : ℚ
  weeklyEarnings : ℚ

/-- Theorem stating that Carl's weekly earnings are $0.75 given the conditions -/
theorem carl_weekly_earnings (e : CarlEarnings) 
  (h_weeks : e.weeks = 4)
  (h_candyBars : e.candyBars = 6)
  (h_candyBarPrice : e.candyBarPrice = 1/2) :
  e.weeklyEarnings = 3/4 := by
sorry

end carl_weekly_earnings_l22_2220


namespace orange_marbles_count_l22_2258

theorem orange_marbles_count (total : ℕ) (red : ℕ) (blue : ℕ) (orange : ℕ) : 
  total = 24 →
  blue = total / 2 →
  red = 6 →
  orange = total - blue - red →
  orange = 6 := by
sorry

end orange_marbles_count_l22_2258


namespace sum_of_rational_roots_l22_2254

/-- The polynomial p(x) = x^3 - 8x^2 + 17x - 10 -/
def p (x : ℚ) : ℚ := x^3 - 8*x^2 + 17*x - 10

/-- A number is a root of p if p(x) = 0 -/
def is_root (x : ℚ) : Prop := p x = 0

/-- The sum of the rational roots of p(x) is 8 -/
theorem sum_of_rational_roots :
  ∃ (S : Finset ℚ), (∀ x ∈ S, is_root x) ∧ (∀ x : ℚ, is_root x → x ∈ S) ∧ (S.sum id = 8) :=
sorry

end sum_of_rational_roots_l22_2254


namespace rectangle_area_is_100_l22_2282

-- Define the rectangle
def Rectangle (width : ℝ) (length : ℝ) : Type :=
  { w : ℝ // w = width } × { l : ℝ // l = length }

-- Define the properties of the rectangle
def rectangle_properties (r : Rectangle 5 (4 * 5)) : Prop :=
  r.2.1 = 4 * r.1.1

-- Define the area of a rectangle
def area (r : Rectangle 5 (4 * 5)) : ℝ :=
  r.1.1 * r.2.1

-- Theorem statement
theorem rectangle_area_is_100 (r : Rectangle 5 (4 * 5)) 
  (h : rectangle_properties r) : area r = 100 :=
by
  sorry

end rectangle_area_is_100_l22_2282


namespace second_concert_attendance_l22_2209

theorem second_concert_attendance 
  (first_concert : ℕ) 
  (attendance_increase : ℕ) 
  (h1 : first_concert = 65899)
  (h2 : attendance_increase = 119) :
  first_concert + attendance_increase = 66018 :=
by sorry

end second_concert_attendance_l22_2209


namespace initial_inventory_l22_2250

def bookshop_inventory (initial_books : ℕ) : Prop :=
  let saturday_instore := 37
  let saturday_online := 128
  let sunday_instore := 2 * saturday_instore
  let sunday_online := saturday_online + 34
  let shipment := 160
  let current_books := 502
  initial_books = current_books + saturday_instore + saturday_online + sunday_instore + sunday_online - shipment

theorem initial_inventory : ∃ (x : ℕ), bookshop_inventory x ∧ x = 743 := by
  sorry

end initial_inventory_l22_2250


namespace rhombus_perimeter_l22_2274

/-- The perimeter of a rhombus with diagonals of lengths 72 and 30 is 156 -/
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 72) (h2 : d2 = 30) : 
  4 * Real.sqrt ((d1/2)^2 + (d2/2)^2) = 156 := by
  sorry

end rhombus_perimeter_l22_2274


namespace some_number_equals_37_l22_2225

theorem some_number_equals_37 : ∃ x : ℤ, 45 - (28 - (x - (15 - 20))) = 59 ∧ x = 37 := by
  sorry

end some_number_equals_37_l22_2225


namespace phone_call_duration_l22_2283

/-- Calculates the duration of a phone call given the initial card value, cost per minute, and remaining credit. -/
def call_duration (initial_value : ℚ) (cost_per_minute : ℚ) (remaining_credit : ℚ) : ℚ :=
  (initial_value - remaining_credit) / cost_per_minute

/-- Theorem stating that given the specific values from the problem, the call duration is 22 minutes. -/
theorem phone_call_duration :
  let initial_value : ℚ := 30
  let cost_per_minute : ℚ := 16/100
  let remaining_credit : ℚ := 2648/100
  call_duration initial_value cost_per_minute remaining_credit = 22 := by
sorry

end phone_call_duration_l22_2283


namespace connect_to_inaccessible_intersection_l22_2223

-- Define the basic types
variable (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V] [FiniteDimensional ℝ V]

-- Define a line as a point and a direction vector
structure Line (V : Type*) [NormedAddCommGroup V] where
  point : V
  direction : V

-- Define the problem setup
variable (l₁ l₂ : Line V) (M : V)

-- State the theorem
theorem connect_to_inaccessible_intersection :
  ∃ (L : Line V), L.point = M ∧ 
    ∃ (t : ℝ), M + t • L.direction ∈ {x | ∃ (s₁ s₂ : ℝ), 
      x = l₁.point + s₁ • l₁.direction ∧ 
      x = l₂.point + s₂ • l₂.direction} :=
sorry

end connect_to_inaccessible_intersection_l22_2223


namespace kolya_parallelepiped_edge_length_l22_2222

/-- A rectangular parallelepiped constructed from unit cubes -/
structure Parallelepiped where
  length : ℕ
  width : ℕ
  height : ℕ
  volume : ℕ
  edge_min : ℕ

/-- The total length of all edges of a rectangular parallelepiped -/
def total_edge_length (p : Parallelepiped) : ℕ :=
  4 * (p.length + p.width + p.height)

/-- Theorem stating the total edge length of the specific parallelepiped -/
theorem kolya_parallelepiped_edge_length :
  ∃ (p : Parallelepiped),
    p.volume = 440 ∧
    p.edge_min = 5 ∧
    p.length ≥ p.edge_min ∧
    p.width ≥ p.edge_min ∧
    p.height ≥ p.edge_min ∧
    p.volume = p.length * p.width * p.height ∧
    total_edge_length p = 96 := by
  sorry

end kolya_parallelepiped_edge_length_l22_2222


namespace sqrt_product_simplification_l22_2259

theorem sqrt_product_simplification (p : ℝ) (hp : p > 0) :
  Real.sqrt (15 * p^3) * Real.sqrt (8 * p) * Real.sqrt (12 * p^5) = 60 * p^4 * Real.sqrt (2 * p) :=
by sorry

end sqrt_product_simplification_l22_2259


namespace mark_paid_54_l22_2208

/-- The total amount Mark paid for hiring a singer -/
def total_paid (hours : ℕ) (rate : ℚ) (tip_percentage : ℚ) : ℚ :=
  let base_cost := hours * rate
  let tip := base_cost * tip_percentage
  base_cost + tip

/-- Theorem stating that Mark paid $54 for hiring the singer -/
theorem mark_paid_54 :
  total_paid 3 15 (20 / 100) = 54 := by
  sorry

end mark_paid_54_l22_2208


namespace employee_savings_l22_2242

/-- Calculate the combined savings of three employees over a period of time. -/
def combined_savings (hourly_wage : ℚ) (hours_per_day : ℕ) (days_per_week : ℕ) 
  (robby_save_ratio : ℚ) (jaylen_save_ratio : ℚ) (miranda_save_ratio : ℚ) 
  (num_weeks : ℕ) : ℚ :=
  let weekly_salary := hourly_wage * hours_per_day * days_per_week
  let robby_savings := robby_save_ratio * weekly_salary
  let jaylen_savings := jaylen_save_ratio * weekly_salary
  let miranda_savings := miranda_save_ratio * weekly_salary
  (robby_savings + jaylen_savings + miranda_savings) * num_weeks

/-- The combined savings of three employees after four weeks is $3000. -/
theorem employee_savings : 
  combined_savings 10 10 5 (2/5) (3/5) (1/2) 4 = 3000 := by
  sorry

end employee_savings_l22_2242


namespace sasha_plucked_leaves_l22_2226

/-- The number of leaves Sasha plucked -/
def leaves_plucked : ℕ := 22

/-- The number of apple trees -/
def apple_trees : ℕ := 17

/-- The number of poplar trees -/
def poplar_trees : ℕ := 18

/-- The position of the apple tree after which Masha's phone memory was full -/
def masha_last_photo : ℕ := 10

/-- The number of trees that remained unphotographed by Masha -/
def unphotographed_trees : ℕ := 13

/-- The position of the apple tree from which Sasha started plucking leaves -/
def sasha_start : ℕ := 8

theorem sasha_plucked_leaves : 
  apple_trees = 17 ∧ 
  poplar_trees = 18 ∧ 
  masha_last_photo = 10 ∧ 
  unphotographed_trees = 13 ∧ 
  sasha_start = 8 → 
  leaves_plucked = 22 := by
  sorry

end sasha_plucked_leaves_l22_2226


namespace sin_eleven_pi_thirds_l22_2298

theorem sin_eleven_pi_thirds : Real.sin (11 * π / 3) = -Real.sqrt 3 / 2 := by
  sorry

end sin_eleven_pi_thirds_l22_2298


namespace weight_on_switch_l22_2231

theorem weight_on_switch (total_weight : ℕ) (additional_weight : ℕ) 
  (h1 : total_weight = 712)
  (h2 : additional_weight = 478) :
  total_weight - additional_weight = 234 := by
  sorry

end weight_on_switch_l22_2231


namespace multiply_23_by_4_l22_2263

theorem multiply_23_by_4 : 23 * 4 = 20 * 4 + 3 * 4 := by
  sorry

end multiply_23_by_4_l22_2263


namespace largest_five_digit_divisible_by_8_l22_2200

theorem largest_five_digit_divisible_by_8 : 
  ∀ n : ℕ, n ≤ 99999 ∧ n ≥ 10000 ∧ n % 8 = 0 → n ≤ 99992 :=
by
  sorry

end largest_five_digit_divisible_by_8_l22_2200


namespace sufficient_not_necessary_l22_2216

theorem sufficient_not_necessary (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ x : ℝ, HasDerivAt (fun x => a^x) (a^x * Real.log a) x ∧ a^x * Real.log a < 0) →
  (∀ x : ℝ, HasDerivAt (fun x => (2-a)*x^3) (3*(2-a)*x^2) x ∧ 3*(2-a)*x^2 > 0) ∧
  ¬((∀ x : ℝ, HasDerivAt (fun x => (2-a)*x^3) (3*(2-a)*x^2) x ∧ 3*(2-a)*x^2 > 0) →
    (∀ x : ℝ, HasDerivAt (fun x => a^x) (a^x * Real.log a) x ∧ a^x * Real.log a < 0)) :=
by sorry

end sufficient_not_necessary_l22_2216


namespace smallest_prime_perfect_square_plus_20_l22_2271

-- Define a function to check if a number is prime
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

-- Define a function to check if a number is a perfect square plus 20
def isPerfectSquarePlus20 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k^2 + 20

-- Theorem statement
theorem smallest_prime_perfect_square_plus_20 :
  isPrime 29 ∧ isPerfectSquarePlus20 29 ∧
  ∀ m : ℕ, m < 29 → ¬(isPrime m ∧ isPerfectSquarePlus20 m) :=
sorry

end smallest_prime_perfect_square_plus_20_l22_2271


namespace natalia_documentaries_l22_2265

/-- The number of documentaries in Natalia's library --/
def documentaries (novels comics albums crates_used crate_capacity : ℕ) : ℕ :=
  crates_used * crate_capacity - (novels + comics + albums)

/-- Theorem stating the number of documentaries in Natalia's library --/
theorem natalia_documentaries :
  documentaries 145 271 209 116 9 = 419 := by
  sorry

end natalia_documentaries_l22_2265


namespace pascal_ninth_row_interior_sum_l22_2277

/-- Sum of elements in row n of Pascal's Triangle -/
def pascal_row_sum (n : ℕ) : ℕ := 2^(n-1)

/-- Sum of interior elements in row n of Pascal's Triangle -/
def pascal_interior_sum (n : ℕ) : ℕ := pascal_row_sum n - 2

theorem pascal_ninth_row_interior_sum :
  pascal_interior_sum 9 = 254 := by sorry

end pascal_ninth_row_interior_sum_l22_2277


namespace tessa_initial_apples_l22_2257

/-- The initial number of apples Tessa had -/
def initial_apples : ℝ := sorry

/-- The number of apples Anita gives to Tessa -/
def apples_from_anita : ℝ := 5.0

/-- The number of apples needed to make a pie -/
def apples_for_pie : ℝ := 4.0

/-- The number of apples left after making the pie -/
def apples_left : ℝ := 11

/-- Theorem stating that Tessa initially had 10 apples -/
theorem tessa_initial_apples : 
  initial_apples + apples_from_anita - apples_for_pie = apples_left ∧ 
  initial_apples = 10 := by sorry

end tessa_initial_apples_l22_2257


namespace secret_reaches_2186_l22_2243

def secret_spread (day : ℕ) : ℕ :=
  if day = 0 then 1
  else secret_spread (day - 1) + 3^day

theorem secret_reaches_2186 :
  ∃ d : ℕ, d ≤ 7 ∧ secret_spread d ≥ 2186 :=
by sorry

end secret_reaches_2186_l22_2243


namespace factorization_sum_l22_2261

theorem factorization_sum (a b c : ℤ) : 
  (∀ x : ℝ, x^2 + 9*x + 14 = (x + a)*(x + b)) →
  (∀ x : ℝ, x^2 + 4*x - 21 = (x + b)*(x - c)) →
  a + b + c = 12 := by
sorry

end factorization_sum_l22_2261


namespace rectangle_area_l22_2299

/-- Rectangle PQRS with given coordinates and properties -/
structure Rectangle where
  P : ℝ × ℝ
  Q : ℝ × ℝ
  S : ℝ × ℝ
  is_rectangle : Bool

/-- The area of the rectangle PQRS is 200000 -/
theorem rectangle_area (rect : Rectangle) : 
  rect.P = (-15, 30) →
  rect.Q = (985, 230) →
  rect.S.1 = -13 →
  rect.is_rectangle = true →
  (rect.Q.1 - rect.P.1) * (rect.S.2 - rect.P.2) = 200000 := by
  sorry


end rectangle_area_l22_2299


namespace absolute_value_equation_unique_solution_l22_2207

theorem absolute_value_equation_unique_solution :
  ∃! x : ℝ, |x - 10| = |x + 5| + 2 :=
by
  -- The proof goes here
  sorry

end absolute_value_equation_unique_solution_l22_2207


namespace homework_time_decrease_l22_2260

theorem homework_time_decrease (x : ℝ) : 
  (∀ t : ℝ, t > 0 → (t * (1 - x))^2 = t * (1 - x)^2) →
  100 * (1 - x)^2 = 70 :=
by sorry

end homework_time_decrease_l22_2260


namespace triangle_existence_l22_2214

/-- Given two angles and a perimeter, prove the existence of a triangle with these properties -/
theorem triangle_existence (A B P : ℝ) (h_angle_sum : 0 < A + B ∧ A + B < 180) (h_perimeter : P > 0) :
  ∃ (a b c : ℝ), 
    a > 0 ∧ b > 0 ∧ c > 0 ∧  -- Positive side lengths
    a + b > c ∧ b + c > a ∧ a + c > b ∧  -- Triangle inequality
    a + b + c = P ∧  -- Perimeter condition
    ∃ (C : ℝ), C = 180 - (A + B) ∧  -- Third angle
    Real.cos A = (b^2 + c^2 - a^2) / (2*b*c) ∧  -- Cosine law for angle A
    Real.cos B = (a^2 + c^2 - b^2) / (2*a*c) ∧  -- Cosine law for angle B
    Real.cos C = (a^2 + b^2 - c^2) / (2*a*b) :=  -- Cosine law for angle C
by sorry


end triangle_existence_l22_2214


namespace pizza_meat_distribution_l22_2215

/-- Pizza meat distribution problem -/
theorem pizza_meat_distribution 
  (pepperoni : ℕ) 
  (ham : ℕ) 
  (sausage : ℕ) 
  (slices : ℕ) 
  (h1 : pepperoni = 30)
  (h2 : ham = 2 * pepperoni)
  (h3 : sausage = pepperoni + 12)
  (h4 : slices = 6)
  : (pepperoni + ham + sausage) / slices = 22 := by
  sorry

end pizza_meat_distribution_l22_2215
