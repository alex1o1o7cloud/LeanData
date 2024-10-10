import Mathlib

namespace boys_age_l2162_216283

theorem boys_age (current_age : ℕ) : 
  (current_age = 2 * (current_age - 5)) → current_age = 10 := by
  sorry

end boys_age_l2162_216283


namespace sqrt_equation_solution_l2162_216263

theorem sqrt_equation_solution (x : ℝ) (h : x > 0) : 18 / Real.sqrt x = 2 → x = 81 := by
  sorry

end sqrt_equation_solution_l2162_216263


namespace complete_square_sum_l2162_216213

theorem complete_square_sum (x : ℝ) : 
  (∃ d e : ℤ, (x + d:ℝ)^2 = e ∧ x^2 - 10*x + 15 = 0) → 
  (∃ d e : ℤ, (x + d:ℝ)^2 = e ∧ x^2 - 10*x + 15 = 0 ∧ d + e = 5) :=
by sorry

end complete_square_sum_l2162_216213


namespace quadratic_inequality_solution_set_l2162_216234

theorem quadratic_inequality_solution_set :
  ∀ x : ℝ, x^2 - 2*x - 3 > 0 ↔ x > 3 ∨ x < -1 := by
sorry

end quadratic_inequality_solution_set_l2162_216234


namespace shaded_area_square_with_quarter_circles_l2162_216277

/-- The area of the shaded region in a square with quarter circles at each corner -/
theorem shaded_area_square_with_quarter_circles 
  (square_side : ℝ) 
  (circle_radius : ℝ) 
  (h1 : square_side = 15) 
  (h2 : circle_radius = 5) : 
  square_side ^ 2 - 4 * (π / 4 * circle_radius ^ 2) = 225 - 25 * π :=
by sorry

end shaded_area_square_with_quarter_circles_l2162_216277


namespace sum_of_cubes_square_not_prime_product_l2162_216266

theorem sum_of_cubes_square_not_prime_product (a b : ℕ+) (n : ℕ) :
  a^3 + b^3 = n^2 →
  ¬∃ (p q : ℕ), Nat.Prime p ∧ Nat.Prime q ∧ p ≠ q ∧ a + b = p * q :=
sorry

end sum_of_cubes_square_not_prime_product_l2162_216266


namespace cubic_factorization_l2162_216204

theorem cubic_factorization (x : ℝ) : x^3 - 4*x = x*(x+2)*(x-2) := by
  sorry

end cubic_factorization_l2162_216204


namespace trapezoid_shorter_lateral_l2162_216258

/-- Represents a trapezoid with specific properties -/
structure Trapezoid where
  longer_lateral : ℝ
  base_difference : ℝ
  right_angle_intersection : Bool

/-- 
  Theorem: In a trapezoid where the lines containing the lateral sides intersect at a right angle,
  if the longer lateral side is 8 and the difference between the bases is 10,
  then the shorter lateral side is 6.
-/
theorem trapezoid_shorter_lateral 
  (t : Trapezoid) 
  (h1 : t.longer_lateral = 8) 
  (h2 : t.base_difference = 10) 
  (h3 : t.right_angle_intersection = true) : 
  ∃ (shorter_lateral : ℝ), shorter_lateral = 6 := by
  sorry

#check trapezoid_shorter_lateral

end trapezoid_shorter_lateral_l2162_216258


namespace difference_divisible_by_nine_l2162_216241

/-- Represents the digit reversal of a natural number -/
def digitReversal (n : ℕ) : ℕ := sorry

/-- Theorem stating that the difference between a natural number and its digit reversal is divisible by 9 -/
theorem difference_divisible_by_nine (n : ℕ) : 
  ∃ k : ℤ, (n : ℤ) - (digitReversal n : ℤ) = 9 * k := by sorry

end difference_divisible_by_nine_l2162_216241


namespace puppies_given_away_l2162_216280

def initial_puppies : ℕ := 7
def remaining_puppies : ℕ := 2

theorem puppies_given_away : initial_puppies - remaining_puppies = 5 := by
  sorry

end puppies_given_away_l2162_216280


namespace regular_polygon_sides_l2162_216261

theorem regular_polygon_sides (n : ℕ) (interior_angle exterior_angle : ℝ) : 
  n > 2 →
  interior_angle = exterior_angle + 60 →
  interior_angle + exterior_angle = 180 →
  n * exterior_angle = 360 →
  n = 6 := by
  sorry

end regular_polygon_sides_l2162_216261


namespace money_loses_exchange_value_valid_money_properties_l2162_216222

/-- Represents an individual on an island -/
structure Individual where
  name : String

/-- Represents money found on the island -/
structure Money where
  amount : ℕ

/-- Represents the state of being on a deserted island -/
structure DesertedIsland where
  inhabitants : List Individual

/-- Function to determine if money has value as a medium of exchange -/
def hasExchangeValue (island : DesertedIsland) (money : Money) : Prop :=
  island.inhabitants.length > 1

/-- Theorem stating that money loses its exchange value on a deserted island with only one inhabitant -/
theorem money_loses_exchange_value 
  (crusoe : Individual) 
  (island : DesertedIsland) 
  (money : Money) 
  (h1 : island.inhabitants = [crusoe]) : 
  ¬(hasExchangeValue island money) := by
  sorry

/-- Properties required for an item to be considered money -/
structure MoneyProperties where
  durability : Prop
  portability : Prop
  divisibility : Prop
  acceptability : Prop
  uniformity : Prop
  limitedSupply : Prop

/-- Function to determine if an item can be considered money -/
def isValidMoney (item : MoneyProperties) : Prop :=
  item.durability ∧ 
  item.portability ∧ 
  item.divisibility ∧ 
  item.acceptability ∧ 
  item.uniformity ∧ 
  item.limitedSupply

/-- Theorem stating that an item must possess all required properties to be considered valid money -/
theorem valid_money_properties (item : MoneyProperties) :
  isValidMoney item ↔ 
    (item.durability ∧ 
     item.portability ∧ 
     item.divisibility ∧ 
     item.acceptability ∧ 
     item.uniformity ∧ 
     item.limitedSupply) := by
  sorry

end money_loses_exchange_value_valid_money_properties_l2162_216222


namespace distance_between_trees_problem_l2162_216267

/-- The distance between consecutive trees in a yard -/
def distance_between_trees (yard_length : ℕ) (num_trees : ℕ) : ℚ :=
  (yard_length : ℚ) / (num_trees - 1 : ℚ)

/-- Theorem: The distance between consecutive trees in a 400-meter yard with 26 trees is 16 meters -/
theorem distance_between_trees_problem :
  distance_between_trees 400 26 = 16 := by
  sorry

end distance_between_trees_problem_l2162_216267


namespace no_fixed_points_composition_l2162_216215

-- Define the quadratic function f(x)
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + x

-- Theorem statement
theorem no_fixed_points_composition
  (a b : ℝ)
  (h : ∀ x : ℝ, f a b x ≠ x) :
  ∀ x : ℝ, f a b (f a b x) ≠ x :=
by sorry

end no_fixed_points_composition_l2162_216215


namespace evans_earnings_l2162_216212

/-- Proves that Evan earned $21 given the conditions of the problem -/
theorem evans_earnings (markese_earnings : ℕ) (total_earnings : ℕ) (earnings_difference : ℕ)
  (h1 : markese_earnings = 16)
  (h2 : total_earnings = 37)
  (h3 : markese_earnings + earnings_difference = total_earnings)
  (h4 : earnings_difference = 5) : 
  total_earnings - markese_earnings = 21 := by
  sorry

end evans_earnings_l2162_216212


namespace intersection_N_complement_M_l2162_216200

def M : Set ℝ := {x | x^2 > 4}
def N : Set ℝ := {x | 1 < x ∧ x < 3}

theorem intersection_N_complement_M :
  N ∩ (Set.univ \ M) = {x | 1 < x ∧ x ≤ 2} := by sorry

end intersection_N_complement_M_l2162_216200


namespace tadpoles_kept_calculation_l2162_216217

/-- The number of tadpoles Trent kept, given the initial number and percentage released -/
def tadpoles_kept (x : ℝ) : ℝ :=
  x * (1 - 0.825)

/-- Theorem stating that the number of tadpoles kept is 0.175 * x -/
theorem tadpoles_kept_calculation (x : ℝ) :
  tadpoles_kept x = 0.175 * x := by
  sorry

end tadpoles_kept_calculation_l2162_216217


namespace lada_elevator_speed_ratio_l2162_216240

/-- The ratio of Lada's original speed to the elevator's speed -/
def speed_ratio : ℚ := 11/4

/-- The number of floors in the first scenario -/
def floors_first : ℕ := 3

/-- The number of floors in the second scenario -/
def floors_second : ℕ := 7

/-- The factor by which Lada increases her speed in the second scenario -/
def speed_increase : ℚ := 2

/-- The factor by which Lada's waiting time increases in the second scenario -/
def wait_time_increase : ℚ := 3

theorem lada_elevator_speed_ratio :
  ∀ (V U : ℚ) (S : ℝ),
  V > 0 → U > 0 → S > 0 →
  (floors_second : ℚ) / (speed_increase * U) - floors_second / V = 
    wait_time_increase * (floors_first / U - floors_first / V) →
  U / V = speed_ratio := by sorry

end lada_elevator_speed_ratio_l2162_216240


namespace lines_coplanar_iff_k_eq_neg_29_div_9_l2162_216242

/-- Two lines in 3D space -/
structure Line3D where
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

/-- Determinant of a 3x3 matrix -/
def det3 (a b c : ℝ × ℝ × ℝ) : ℝ :=
  let (a1, a2, a3) := a
  let (b1, b2, b3) := b
  let (c1, c2, c3) := c
  a1 * (b2 * c3 - b3 * c2) - a2 * (b1 * c3 - b3 * c1) + a3 * (b1 * c2 - b2 * c1)

/-- Two lines are coplanar if the determinant of their direction vectors and the vector between their points is zero -/
def areCoplanar (l1 l2 : Line3D) : Prop :=
  let (x1, y1, z1) := l1.point
  let (x2, y2, z2) := l2.point
  let v := (x2 - x1, y2 - y1, z2 - z1)
  det3 l1.direction l2.direction v = 0

/-- The main theorem -/
theorem lines_coplanar_iff_k_eq_neg_29_div_9 :
  let l1 : Line3D := ⟨(3, 2, 4), (2, -1, 3)⟩
  let l2 : Line3D := ⟨(0, 4, 1), (3*k, 1, 2)⟩
  areCoplanar l1 l2 ↔ k = -29/9 := by
  sorry

end lines_coplanar_iff_k_eq_neg_29_div_9_l2162_216242


namespace consecutive_integers_prime_divisor_ratio_l2162_216294

theorem consecutive_integers_prime_divisor_ratio :
  ∃ a : ℕ, ∀ i ∈ Finset.range 2009,
    let n := a + i + 1
    ∃ p q : ℕ, Prime p ∧ Prime q ∧ p ∣ n ∧ q ∣ n ∧
      (∀ r : ℕ, Prime r → r ∣ n → p ≤ r) ∧
      (∀ r : ℕ, Prime r → r ∣ n → r ≤ q) ∧
      q > 20 * p :=
by
  sorry

end consecutive_integers_prime_divisor_ratio_l2162_216294


namespace total_donuts_three_days_l2162_216216

def monday_donuts : ℕ := 14

def tuesday_donuts : ℕ := monday_donuts / 2

def wednesday_donuts : ℕ := 4 * monday_donuts

theorem total_donuts_three_days : 
  monday_donuts + tuesday_donuts + wednesday_donuts = 77 := by
  sorry

end total_donuts_three_days_l2162_216216


namespace three_digit_cube_divisible_by_16_l2162_216247

theorem three_digit_cube_divisible_by_16 :
  ∃! n : ℕ, 100 ≤ 64 * n^3 ∧ 64 * n^3 ≤ 999 := by
  sorry

end three_digit_cube_divisible_by_16_l2162_216247


namespace sum_divisors_2_3_power_l2162_216271

/-- Sum of divisors function -/
def sum_of_divisors (n : ℕ) : ℕ := sorry

/-- Theorem: If the sum of divisors of 2^i * 3^j is 360, then i + j = 6 -/
theorem sum_divisors_2_3_power (i j : ℕ) :
  sum_of_divisors (2^i * 3^j) = 360 → i + j = 6 := by
  sorry

end sum_divisors_2_3_power_l2162_216271


namespace two_percent_as_decimal_l2162_216233

/-- Expresses a percentage as a decimal fraction -/
def percent_to_decimal (p : ℚ) : ℚ := p / 100

/-- Proves that 2% expressed as a decimal fraction is equal to 0.02 -/
theorem two_percent_as_decimal : percent_to_decimal 2 = 0.02 := by sorry

end two_percent_as_decimal_l2162_216233


namespace selection_methods_count_l2162_216287

def n : ℕ := 10  -- Total number of college student village officials
def k : ℕ := 3   -- Number of individuals to be selected

def total_without_b : ℕ := Nat.choose (n - 1) k
def without_a_and_c : ℕ := Nat.choose (n - 3) k

theorem selection_methods_count : 
  total_without_b - without_a_and_c = 49 := by sorry

end selection_methods_count_l2162_216287


namespace common_tangents_M_N_l2162_216211

/-- Circle M defined by the equation x^2 + y^2 - 4y = 0 -/
def circle_M (x y : ℝ) : Prop := x^2 + y^2 - 4*y = 0

/-- Circle N defined by the equation (x-1)^2 + (y-1)^2 = 1 -/
def circle_N (x y : ℝ) : Prop := (x-1)^2 + (y-1)^2 = 1

/-- The number of common tangent lines between two circles -/
def num_common_tangents (M N : (ℝ → ℝ → Prop)) : ℕ := sorry

/-- Theorem stating that the number of common tangent lines between circles M and N is 2 -/
theorem common_tangents_M_N : num_common_tangents circle_M circle_N = 2 := by sorry

end common_tangents_M_N_l2162_216211


namespace edward_book_purchase_l2162_216237

theorem edward_book_purchase (total_spent : ℝ) (num_books : ℕ) (cost_per_book : ℝ) : 
  total_spent = 6 ∧ num_books = 2 ∧ total_spent = num_books * cost_per_book → cost_per_book = 3 := by
  sorry

end edward_book_purchase_l2162_216237


namespace de_morgans_laws_l2162_216223

universe u

theorem de_morgans_laws {U : Type u} (A B : Set U) :
  (Set.compl (A ∪ B) = Set.compl A ∩ Set.compl B) ∧
  (Set.compl (A ∩ B) = Set.compl A ∪ Set.compl B) := by
  sorry

end de_morgans_laws_l2162_216223


namespace min_value_expression_l2162_216268

theorem min_value_expression (a b m n : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : m > 0) (h4 : n > 0) 
  (h5 : a + b = 1) (h6 : m * n = 2) :
  (a * m + b * n) * (b * m + a * n) ≥ 2 := by
  sorry

end min_value_expression_l2162_216268


namespace no_solution_condition_l2162_216275

theorem no_solution_condition (m : ℚ) : 
  (∀ x : ℚ, x ≠ 5 ∧ x ≠ -5 → 1 / (x - 5) + m / (x + 5) ≠ (m + 5) / (x^2 - 25)) ↔ 
  m = -1 ∨ m = 5 ∨ m = -5/11 := by
  sorry

end no_solution_condition_l2162_216275


namespace gym_membership_ratio_l2162_216264

theorem gym_membership_ratio (f m : ℕ) (hf : f > 0) (hm : m > 0) : 
  (35 * f + 30 * m) / (f + m) = 32 → f / m = 2 / 3 := by
sorry

end gym_membership_ratio_l2162_216264


namespace division_problem_l2162_216230

theorem division_problem (number : ℕ) : 
  (number / 20 = 6) ∧ (number % 20 = 2) → number = 122 := by
  sorry

end division_problem_l2162_216230


namespace sasha_remainder_l2162_216245

theorem sasha_remainder (n : ℕ) (a b c d : ℕ) : 
  n = 102 * a + b ∧ 
  n = 103 * c + d ∧ 
  0 ≤ b ∧ b < 102 ∧
  0 ≤ d ∧ d < 103 ∧
  a + d = 20 →
  b = 20 :=
by sorry

end sasha_remainder_l2162_216245


namespace unique_base_conversion_l2162_216296

/-- Converts a number from base 6 to base 10 -/
def base6ToBase10 (n : Nat) : Nat :=
  (n / 100) * 36 + ((n / 10) % 10) * 6 + (n % 10)

/-- Converts a number from base b to base 10 -/
def baseBToBase10 (n : Nat) (b : Nat) : Nat :=
  (n / 100) * b^2 + ((n / 10) % 10) * b + (n % 10)

theorem unique_base_conversion :
  ∃! (b : Nat), b > 0 ∧ base6ToBase10 125 = baseBToBase10 221 b :=
by sorry

end unique_base_conversion_l2162_216296


namespace alicia_miles_run_l2162_216285

/-- Represents the step counter's maximum value before reset --/
def max_steps : Nat := 99999

/-- Represents the number of times the counter reset --/
def reset_count : Nat := 50

/-- Represents the steps shown on the counter on the last day --/
def final_steps : Nat := 25000

/-- Represents the number of steps Alicia takes per mile --/
def steps_per_mile : Nat := 1500

/-- Calculates the total number of steps Alicia took over the year --/
def total_steps : Nat := (max_steps + 1) * reset_count + final_steps

/-- Calculates the approximate number of miles Alicia ran --/
def miles_run : Nat := total_steps / steps_per_mile

/-- Theorem stating that Alicia ran approximately 3350 miles --/
theorem alicia_miles_run : miles_run = 3350 := by
  sorry

end alicia_miles_run_l2162_216285


namespace calculate_markup_l2162_216201

/-- Calculate the markup for an article given its purchase price, overhead percentage, and required net profit. -/
theorem calculate_markup (purchase_price overhead_percentage net_profit : ℝ) : 
  purchase_price = 48 → 
  overhead_percentage = 0.20 → 
  net_profit = 12 → 
  let overhead_cost := purchase_price * overhead_percentage
  let total_cost := purchase_price + overhead_cost
  let selling_price := total_cost + net_profit
  let markup := selling_price - purchase_price
  markup = 21.60 := by
sorry


end calculate_markup_l2162_216201


namespace pen_problem_l2162_216269

theorem pen_problem (marked_price : ℝ) (num_pens : ℕ) : 
  marked_price > 0 →
  num_pens * marked_price = 46 * marked_price →
  (num_pens * marked_price * 0.99 - 46 * marked_price) / (46 * marked_price) * 100 = 29.130434782608695 →
  num_pens = 60 := by
sorry

end pen_problem_l2162_216269


namespace inscribed_circle_rectangle_area_l2162_216220

theorem inscribed_circle_rectangle_area :
  ∀ (r : ℝ) (length width : ℝ),
    r = 3 →
    length / width = 3 →
    2 * r = width →
    length * width = 108 :=
by
  sorry

end inscribed_circle_rectangle_area_l2162_216220


namespace third_discount_percentage_l2162_216250

/-- Given a car with an initial price and three successive discounts, 
    calculate the third discount percentage. -/
theorem third_discount_percentage 
  (initial_price : ℝ) 
  (first_discount second_discount : ℝ)
  (final_price : ℝ) :
  initial_price = 12000 →
  first_discount = 0.20 →
  second_discount = 0.15 →
  final_price = 7752 →
  ∃ (third_discount : ℝ),
    final_price = initial_price * 
      (1 - first_discount) * 
      (1 - second_discount) * 
      (1 - third_discount) ∧
    third_discount = 0.05 := by
  sorry


end third_discount_percentage_l2162_216250


namespace sqrt_3_squared_4_fourth_l2162_216249

theorem sqrt_3_squared_4_fourth : Real.sqrt (3^2 * 4^4) = 48 := by
  sorry

end sqrt_3_squared_4_fourth_l2162_216249


namespace complex_in_first_quadrant_l2162_216225

theorem complex_in_first_quadrant (z : ℂ) : z = Complex.mk (Real.sqrt 3) 1 → z.re > 0 ∧ z.im > 0 := by
  sorry

end complex_in_first_quadrant_l2162_216225


namespace circle_area_ratio_l2162_216299

theorem circle_area_ratio (R S : Real) (hR : R > 0) (hS : S > 0) 
  (h_diameter : R = 0.4 * S) : 
  (π * R^2) / (π * S^2) = 0.16 := by
  sorry

end circle_area_ratio_l2162_216299


namespace specific_field_planted_fraction_l2162_216244

/-- Represents a right-angled triangular field with an unplanted square at the right angle. -/
structure TriangularField where
  leg1 : ℝ
  leg2 : ℝ
  square_distance : ℝ

/-- Calculates the fraction of the field that is planted. -/
def planted_fraction (field : TriangularField) : ℚ :=
  sorry

/-- Theorem stating that for a specific field configuration, the planted fraction is 7/10. -/
theorem specific_field_planted_fraction :
  let field : TriangularField := { leg1 := 5, leg2 := 12, square_distance := 3 }
  planted_fraction field = 7 / 10 := by
  sorry

end specific_field_planted_fraction_l2162_216244


namespace min_value_of_function_l2162_216221

theorem min_value_of_function :
  ∀ x : ℝ, x^2 + 1 / (x^2 + 1) + 3 ≥ 4 :=
by sorry

end min_value_of_function_l2162_216221


namespace sports_conference_games_l2162_216252

/-- Calculates the number of games in a sports conference season -/
def conference_games (total_teams : ℕ) (teams_per_division : ℕ) 
  (intra_division_games : ℕ) (inter_division_games : ℕ) : ℕ :=
  let games_per_team := (teams_per_division - 1) * intra_division_games + teams_per_division * inter_division_games
  (total_teams * games_per_team) / 2

/-- Theorem: The number of games in the described sports conference is 232 -/
theorem sports_conference_games : 
  conference_games 16 8 3 1 = 232 := by sorry

end sports_conference_games_l2162_216252


namespace problem_statement_l2162_216284

def B : Set ℝ := {m | m < 2}

def A (a : ℝ) : Set ℝ := {x | a < x ∧ x < a + 1}

theorem problem_statement :
  (∀ m : ℝ, m ∈ B ↔ ∀ x : ℝ, x ≥ 2 → x^2 - x - m > 0) ∧
  (∀ a : ℝ, (∀ x : ℝ, x ∈ A a → x ∈ B) → A a ⊂ B → a ≤ 1) :=
by sorry

end problem_statement_l2162_216284


namespace x_equals_one_sufficient_not_necessary_l2162_216291

theorem x_equals_one_sufficient_not_necessary :
  (∃ x : ℝ, x ^ 2 - 3 * x + 2 = 0 ∧ x ≠ 1) ∧
  (∀ x : ℝ, x = 1 → x ^ 2 - 3 * x + 2 = 0) := by
  sorry

end x_equals_one_sufficient_not_necessary_l2162_216291


namespace xyz_value_l2162_216274

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 18)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 6) : 
  x * y * z = 4 := by
  sorry

end xyz_value_l2162_216274


namespace isosceles_trapezoid_side_length_l2162_216276

/-- An isosceles trapezoid with given base lengths and area -/
structure IsoscelesTrapezoid where
  base1 : ℝ
  base2 : ℝ
  area : ℝ

/-- The length of the sides of an isosceles trapezoid -/
def side_length (t : IsoscelesTrapezoid) : ℝ :=
  sorry

/-- Theorem: The side length of the given isosceles trapezoid is 5 -/
theorem isosceles_trapezoid_side_length :
  let t : IsoscelesTrapezoid := { base1 := 10, base2 := 16, area := 52 }
  side_length t = 5 := by sorry

end isosceles_trapezoid_side_length_l2162_216276


namespace pike_eel_fat_difference_l2162_216238

theorem pike_eel_fat_difference (herring_fat eel_fat : ℕ) (pike_fat : ℕ) 
  (fish_count : ℕ) (total_fat : ℕ) : 
  herring_fat = 40 →
  eel_fat = 20 →
  pike_fat > eel_fat →
  fish_count = 40 →
  fish_count * herring_fat + fish_count * eel_fat + fish_count * pike_fat = total_fat →
  total_fat = 3600 →
  pike_fat - eel_fat = 10 := by
sorry

end pike_eel_fat_difference_l2162_216238


namespace unique_perfect_square_sum_l2162_216290

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def distinct_perfect_square_sum (a b c : ℕ) : Prop :=
  is_perfect_square a ∧ is_perfect_square b ∧ is_perfect_square c ∧
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  a + b + c = 100

theorem unique_perfect_square_sum : 
  ∃! (abc : ℕ × ℕ × ℕ), distinct_perfect_square_sum abc.1 abc.2.1 abc.2.2 :=
sorry

end unique_perfect_square_sum_l2162_216290


namespace complement_intersection_equals_set_l2162_216278

def I : Finset Nat := {1,2,3,4,5,6,7,8}
def M : Finset Nat := {3,4,5}
def N : Finset Nat := {1,3,6}

theorem complement_intersection_equals_set : 
  (I \ M) ∩ (I \ N) = {2,7,8} := by sorry

end complement_intersection_equals_set_l2162_216278


namespace line_parallel_perpendicular_l2162_216292

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel and perpendicular relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)

-- State the theorem
theorem line_parallel_perpendicular 
  (m n : Line) (β : Plane) 
  (h1 : m ≠ n) 
  (h2 : parallel m n) 
  (h3 : perpendicular m β) : 
  perpendicular n β := by sorry

end line_parallel_perpendicular_l2162_216292


namespace sum_of_reciprocals_of_roots_l2162_216259

theorem sum_of_reciprocals_of_roots (p₁ p₂ : ℝ) : 
  p₁^2 - 17*p₁ + 8 = 0 → 
  p₂^2 - 17*p₂ + 8 = 0 → 
  p₁ ≠ p₂ →
  1/p₁ + 1/p₂ = 17/8 := by sorry

end sum_of_reciprocals_of_roots_l2162_216259


namespace rectangle_area_l2162_216226

theorem rectangle_area (square_area : ℝ) (rectangle_width : ℝ) (rectangle_length : ℝ) : 
  square_area = 16 →
  rectangle_width^2 = square_area →
  rectangle_length = 3 * rectangle_width →
  rectangle_width * rectangle_length = 48 := by
sorry

end rectangle_area_l2162_216226


namespace circle_ratio_theorem_l2162_216202

/-- Given two circles ω₁ and ω₂ with centers O₁ and O₂ and radii r₁ and r₂ respectively,
    where O₂ lies on ω₁, A is an intersection point of ω₁ and ω₂, B is an intersection of line O₁O₂ with ω₂,
    and AB = O₁A, prove that r₁/r₂ can only be (√5 - 1)/2 or (√5 + 1)/2 -/
theorem circle_ratio_theorem (r₁ r₂ : ℝ) (h₁ : r₁ > 0) (h₂ : r₂ > 0) :
  (∃ (O₁ O₂ A B : ℝ × ℝ),
    (‖O₂ - O₁‖ = r₁) ∧
    (‖A - O₁‖ = r₁) ∧
    (‖A - O₂‖ = r₂) ∧
    (‖B - O₂‖ = r₂) ∧
    (∃ t : ℝ, B = O₁ + t • (O₂ - O₁)) ∧
    (‖A - B‖ = ‖A - O₁‖)) →
  (r₁ / r₂ = (Real.sqrt 5 - 1) / 2 ∨ r₁ / r₂ = (Real.sqrt 5 + 1) / 2) :=
by sorry


end circle_ratio_theorem_l2162_216202


namespace item_list_price_l2162_216243

/-- The list price of an item -/
def list_price : ℝ := 33

/-- Alice's selling price -/
def alice_price (x : ℝ) : ℝ := x - 15

/-- Charles's selling price -/
def charles_price (x : ℝ) : ℝ := x - 18

/-- Alice's commission rate -/
def alice_rate : ℝ := 0.15

/-- Charles's commission rate -/
def charles_rate : ℝ := 0.18

theorem item_list_price :
  alice_rate * alice_price list_price = charles_rate * charles_price list_price :=
by sorry

end item_list_price_l2162_216243


namespace polynomial_roots_inequality_l2162_216214

theorem polynomial_roots_inequality (a b c : ℝ) : 
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
    12 * x^3 + a * x^2 + b * x + c = 0 ∧ 
    12 * y^3 + a * y^2 + b * y + c = 0 ∧ 
    12 * z^3 + a * z^2 + b * z + c = 0) →
  (∀ x : ℝ, (x^2 + x + 2001)^3 + a * (x^2 + x + 2001)^2 + b * (x^2 + x + 2001) + c ≠ 0) →
  2001^3 + a * 2001^2 + b * 2001 + c > 1/64 := by
sorry

end polynomial_roots_inequality_l2162_216214


namespace train_length_l2162_216246

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 126 → time = 9 → speed * time * (1000 / 3600) = 315 := by
  sorry

end train_length_l2162_216246


namespace goods_train_passing_time_goods_train_passing_time_approx_9_seconds_l2162_216239

/-- The time taken for a goods train to pass a man in another train -/
theorem goods_train_passing_time (passenger_train_speed goods_train_speed : ℝ) 
  (goods_train_length : ℝ) : ℝ :=
  let relative_speed := passenger_train_speed + goods_train_speed
  let relative_speed_mps := relative_speed * 1000 / 3600
  goods_train_length / relative_speed_mps

/-- Proof that the time taken is approximately 9 seconds -/
theorem goods_train_passing_time_approx_9_seconds : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |goods_train_passing_time 60 52 280 - 9| < ε :=
sorry

end goods_train_passing_time_goods_train_passing_time_approx_9_seconds_l2162_216239


namespace select_shoes_four_pairs_l2162_216206

/-- The number of ways to select 4 shoes from 4 pairs such that no two form a pair -/
def selectShoes (n : ℕ) : ℕ :=
  if n = 4 then 2^4 else 0

theorem select_shoes_four_pairs :
  selectShoes 4 = 16 :=
by sorry

end select_shoes_four_pairs_l2162_216206


namespace max_value_cube_sum_ratio_l2162_216295

theorem max_value_cube_sum_ratio (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x + y + z)^3 / (x^3 + y^3 + z^3) ≤ 9 := by
  sorry

end max_value_cube_sum_ratio_l2162_216295


namespace train_length_is_300_l2162_216289

/-- The length of the train in meters -/
def train_length : ℝ := 300

/-- The time (in seconds) it takes for the train to cross the platform -/
def platform_crossing_time : ℝ := 39

/-- The time (in seconds) it takes for the train to cross a signal pole -/
def pole_crossing_time : ℝ := 12

/-- The length of the platform in meters -/
def platform_length : ℝ := 675

/-- Theorem stating that the train length is 300 meters given the conditions -/
theorem train_length_is_300 :
  train_length = 300 ∧
  train_length + platform_length = (train_length / pole_crossing_time) * platform_crossing_time :=
by sorry

end train_length_is_300_l2162_216289


namespace rectangle_dimensions_l2162_216203

theorem rectangle_dimensions (x : ℝ) : 
  (x - 3 > 0) →
  (x - 3) * (3 * x + 7) = 11 * x - 4 →
  x = (13 + Real.sqrt 373) / 6 := by
sorry

end rectangle_dimensions_l2162_216203


namespace complex_sum_equals_negative_two_l2162_216231

theorem complex_sum_equals_negative_two (w : ℂ) : 
  w = Complex.cos (3 * Real.pi / 8) + Complex.I * Complex.sin (3 * Real.pi / 8) →
  2 * (w / (1 + w^3) + w^2 / (1 + w^6) + w^3 / (1 + w^9)) = -2 := by
  sorry

end complex_sum_equals_negative_two_l2162_216231


namespace monotonicity_undetermined_l2162_216279

-- Define the real numbers a, b, and c
variable (a b c : ℝ)

-- Assume a < b < c
variable (h1 : a < b) (h2 : b < c)

-- Define a function f from reals to reals
variable (f : ℝ → ℝ)

-- Define what it means for f to be increasing on an open interval
def IncreasingOn (f : ℝ → ℝ) (l r : ℝ) : Prop :=
  ∀ x y, l < x ∧ x < y ∧ y < r → f x < f y

-- State the theorem
theorem monotonicity_undetermined
  (h_ab : IncreasingOn f a b)
  (h_bc : IncreasingOn f b c) :
  ¬ (IncreasingOn f a c ∨ (∀ x y, a < x ∧ x < y ∧ y < c → f x > f y)) :=
sorry

end monotonicity_undetermined_l2162_216279


namespace tangent_line_intersection_three_distinct_solutions_l2162_216253

/-- The function f(x) = x³ - 9x -/
def f (x : ℝ) : ℝ := x^3 - 9*x

/-- The function g(x) = 3x² + a -/
def g (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + a

/-- The derivative of f -/
def f' (x : ℝ) : ℝ := 3*x^2 - 9

/-- The derivative of g -/
def g' (x : ℝ) : ℝ := 6*x

theorem tangent_line_intersection (a : ℝ) :
  (∃ m : ℝ, f' 0 = g' m ∧ f 0 + f' 0 * m = g a m) → a = 27/4 :=
sorry

theorem three_distinct_solutions (a : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f x = g a x ∧ f y = g a y ∧ f z = g a z) ↔ 
  -27 < a ∧ a < 5 :=
sorry

end tangent_line_intersection_three_distinct_solutions_l2162_216253


namespace nagy_birth_and_death_l2162_216219

def birth_year : ℕ := 1849
def death_year : ℕ := 1934
def grandchild_birth_year : ℕ := 1932
def num_grandchildren : ℕ := 24

theorem nagy_birth_and_death :
  (∃ (n : ℕ), birth_year = n^2) ∧
  (birth_year ≥ 1834 ∧ birth_year ≤ 1887) ∧
  (death_year - birth_year = 84) ∧
  (grandchild_birth_year - birth_year = 83) ∧
  (num_grandchildren = 24) :=
by sorry

end nagy_birth_and_death_l2162_216219


namespace not_all_diagonal_cells_good_l2162_216224

/-- Represents a cell in the table -/
structure Cell where
  row : Fin 13
  col : Fin 13

/-- Represents the table -/
def Table := Fin 13 → Fin 13 → Fin 25

/-- Checks if a cell is "good" -/
def is_good (t : Table) (c : Cell) : Prop :=
  ∀ n : Fin 25, (∃! i : Fin 13, t i c.col = n) ∧ (∃! j : Fin 13, t c.row j = n)

/-- Represents the main diagonal -/
def main_diagonal : List Cell :=
  List.map (λ i => ⟨i, i⟩) (List.range 13)

/-- The theorem to be proved -/
theorem not_all_diagonal_cells_good (t : Table) : 
  ¬(∀ c ∈ main_diagonal, is_good t c) := by
  sorry


end not_all_diagonal_cells_good_l2162_216224


namespace min_value_theorem_l2162_216209

/-- Two lines are perpendicular if the sum of products of their coefficients is zero -/
def perpendicular (a₁ b₁ a₂ b₂ : ℝ) : Prop := a₁ * a₂ + b₁ * b₂ = 0

/-- Definition of the first line l₁: (a-1)x + y - 1 = 0 -/
def line1 (a x y : ℝ) : Prop := (a - 1) * x + y - 1 = 0

/-- Definition of the second line l₂: x + 2by + 1 = 0 -/
def line2 (b x y : ℝ) : Prop := x + 2 * b * y + 1 = 0

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_perp : perpendicular (a - 1) 1 1 (2 * b)) :
  (∀ a' b', a' > 0 → b' > 0 → perpendicular (a' - 1) 1 1 (2 * b') → 2 / a + 1 / b ≤ 2 / a' + 1 / b') ∧ 
  2 / a + 1 / b = 8 :=
sorry

end min_value_theorem_l2162_216209


namespace person_B_age_l2162_216232

theorem person_B_age 
  (avg_ABC : (age_A + age_B + age_C) / 3 = 22)
  (avg_AB : (age_A + age_B) / 2 = 18)
  (avg_BC : (age_B + age_C) / 2 = 25)
  : age_B = 20 := by
  sorry

end person_B_age_l2162_216232


namespace midpoint_chain_l2162_216218

theorem midpoint_chain (A B C D E F G : ℝ) : 
  C = (A + B) / 2 →
  D = (A + C) / 2 →
  E = (A + D) / 2 →
  F = (A + E) / 2 →
  G = (A + F) / 2 →
  G - A = 5 →
  B - A = 160 := by
sorry

end midpoint_chain_l2162_216218


namespace marked_percentage_above_cost_price_l2162_216262

/-- Proves that for an article with given cost price, selling price, and discount percentage,
    the marked percentage above the cost price is correct. -/
theorem marked_percentage_above_cost_price
  (cost_price : ℝ)
  (selling_price : ℝ)
  (discount_percentage : ℝ)
  (h1 : cost_price = 540)
  (h2 : selling_price = 496.80)
  (h3 : discount_percentage = 19.999999999999996)
  : (((selling_price / (1 - discount_percentage / 100) - cost_price) / cost_price) * 100 = 15) := by
  sorry

end marked_percentage_above_cost_price_l2162_216262


namespace number_of_candidates_l2162_216270

/-- The number of ways to select a president and vice president -/
def selection_ways : ℕ := 30

/-- Theorem: Given 30 ways to select a president and vice president, 
    where the same person cannot be both, there are 6 candidates. -/
theorem number_of_candidates : 
  ∃ (n : ℕ), n > 0 ∧ n * (n - 1) = selection_ways := by
  sorry

end number_of_candidates_l2162_216270


namespace ellipse_minimum_area_l2162_216256

/-- An ellipse containing two specific circles has a minimum area -/
theorem ellipse_minimum_area (a b : ℝ) (h_positive_a : a > 0) (h_positive_b : b > 0) :
  (∀ x y : ℝ, x^2/a^2 + y^2/b^2 = 1 → ((x - 2)^2 + y^2 = 4 ∨ (x + 2)^2 + y^2 = 4)) →
  a * b ≥ 8 * Real.sqrt 2 := by
sorry

end ellipse_minimum_area_l2162_216256


namespace stone_slab_length_l2162_216207

theorem stone_slab_length (num_slabs : ℕ) (total_area : ℝ) (slab_length : ℝ) :
  num_slabs = 30 →
  total_area = 67.5 →
  num_slabs * (slab_length ^ 2) = total_area →
  slab_length = 1.5 := by
  sorry

end stone_slab_length_l2162_216207


namespace expression_simplification_l2162_216298

theorem expression_simplification (α : Real) (h : π < α ∧ α < (3*π)/2) :
  Real.sqrt ((1 - Real.cos α) / (1 + Real.cos α)) + Real.sqrt ((1 + Real.cos α) / (1 - Real.cos α)) = -2 / Real.sin α :=
by sorry

end expression_simplification_l2162_216298


namespace corner_cut_pentagon_area_l2162_216273

/-- Pentagon formed by cutting a triangular corner from a rectangle -/
structure CornerCutPentagon where
  sides : Finset ℕ
  is_valid : sides = {14, 21, 22, 28, 35}

/-- The area of the pentagon -/
def pentagon_area (p : CornerCutPentagon) : ℕ :=
  1421

/-- Theorem stating that the area of the specified pentagon is 1421 -/
theorem corner_cut_pentagon_area (p : CornerCutPentagon) : 
  pentagon_area p = 1421 := by sorry

end corner_cut_pentagon_area_l2162_216273


namespace nice_set_property_l2162_216297

def nice (P : Set (ℤ × ℤ)) : Prop :=
  (∀ a b, (a, b) ∈ P → (b, a) ∈ P) ∧
  (∀ a b c d, (a, b) ∈ P → (c, d) ∈ P → (a + c, b - d) ∈ P)

theorem nice_set_property (p q : ℤ) (h1 : Nat.gcd p.natAbs q.natAbs = 1) 
  (h2 : p % 2 ≠ q % 2) :
  ∀ (P : Set (ℤ × ℤ)), nice P → (p, q) ∈ P → P = Set.univ := by
  sorry

end nice_set_property_l2162_216297


namespace distinct_prime_factors_count_l2162_216281

def n : ℕ := 81 * 83 * 85 * 87 + 89

theorem distinct_prime_factors_count : Nat.card (Nat.factors n).toFinset = 6 := by
  sorry

end distinct_prime_factors_count_l2162_216281


namespace clara_stickers_l2162_216208

theorem clara_stickers (initial : ℕ) : 
  initial ≥ 10 →
  (initial - 10) % 2 = 0 →
  (initial - 10) / 2 - 45 = 45 →
  initial = 100 := by
sorry

end clara_stickers_l2162_216208


namespace sum_of_triangles_l2162_216228

/-- The triangle operation that sums three numbers -/
def triangle (a b c : ℕ) : ℕ := a + b + c

/-- The first given triangle -/
def triangle1 : ℕ × ℕ × ℕ := (2, 3, 5)

/-- The second given triangle -/
def triangle2 : ℕ × ℕ × ℕ := (3, 4, 6)

/-- Theorem stating that the sum of triangle operations for both given triangles equals 23 -/
theorem sum_of_triangles :
  triangle triangle1.1 triangle1.2.1 triangle1.2.2 +
  triangle triangle2.1 triangle2.2.1 triangle2.2.2 = 23 := by
  sorry

end sum_of_triangles_l2162_216228


namespace haley_carrots_count_l2162_216265

/-- The number of carrots Haley picked -/
def haley_carrots : ℕ := 39

/-- The number of carrots Haley's mom picked -/
def mom_carrots : ℕ := 38

/-- The number of good carrots -/
def good_carrots : ℕ := 64

/-- The number of bad carrots -/
def bad_carrots : ℕ := 13

theorem haley_carrots_count : haley_carrots = 39 := by
  have total_carrots : ℕ := good_carrots + bad_carrots
  have total_carrots_alt : ℕ := haley_carrots + mom_carrots
  have h1 : total_carrots = total_carrots_alt := by sorry
  sorry

end haley_carrots_count_l2162_216265


namespace percentage_problem_l2162_216210

theorem percentage_problem (x : ℝ) : 
  (x / 100) * 130 = 65 → x = 50 := by
  sorry

end percentage_problem_l2162_216210


namespace remaining_space_is_7200_mb_l2162_216293

/-- Conversion factor from GB to MB -/
def gb_to_mb : ℕ := 1024

/-- Total hard drive capacity in GB -/
def total_capacity_gb : ℕ := 300

/-- Used storage space in MB -/
def used_space_mb : ℕ := 300000

/-- Theorem: The remaining empty space on the hard drive is 7200 MB -/
theorem remaining_space_is_7200_mb :
  total_capacity_gb * gb_to_mb - used_space_mb = 7200 := by
  sorry

end remaining_space_is_7200_mb_l2162_216293


namespace boys_from_pine_l2162_216254

theorem boys_from_pine (total_students : ℕ) (total_boys : ℕ) (total_girls : ℕ)
  (maple_students : ℕ) (pine_students : ℕ) (maple_girls : ℕ)
  (h1 : total_students = 150)
  (h2 : total_boys = 90)
  (h3 : total_girls = 60)
  (h4 : maple_students = 50)
  (h5 : pine_students = 100)
  (h6 : maple_girls = 30)
  (h7 : total_students = total_boys + total_girls)
  (h8 : total_students = maple_students + pine_students)
  (h9 : total_girls = maple_girls + (total_girls - maple_girls)) :
  pine_students - (total_girls - maple_girls) = 70 := by
  sorry

end boys_from_pine_l2162_216254


namespace tangent_line_parallel_point_l2162_216288

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x^2 + x - 2

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 4 * x + 1

theorem tangent_line_parallel_point (P₀ : ℝ × ℝ) : 
  P₀.1 = 1 ∧ P₀.2 = f P₀.1 ∧ f' P₀.1 = 5 :=
sorry

end tangent_line_parallel_point_l2162_216288


namespace problem_statement_l2162_216255

theorem problem_statement : (2 * Real.sqrt 2 - 1)^2 + (1 + Real.sqrt 5) * (1 - Real.sqrt 5) = 5 - 4 * Real.sqrt 2 := by
  sorry

end problem_statement_l2162_216255


namespace lcm_of_15_25_35_l2162_216251

theorem lcm_of_15_25_35 : Nat.lcm 15 (Nat.lcm 25 35) = 525 := by sorry

end lcm_of_15_25_35_l2162_216251


namespace angle_BCA_measure_l2162_216205

-- Define the points
variable (A B C D M O : EuclideanPlane)

-- Define the quadrilateral ABCD
def is_convex_quadrilateral (A B C D : EuclideanPlane) : Prop := sorry

-- Define M as the midpoint of AD
def is_midpoint (M A D : EuclideanPlane) : Prop := sorry

-- Define the intersection of BM and AC at O
def intersect_at (B M A C O : EuclideanPlane) : Prop := sorry

-- Define the angle measure function
def angle_measure (P Q R : EuclideanPlane) : ℝ := sorry

-- State the theorem
theorem angle_BCA_measure 
  (h_quad : is_convex_quadrilateral A B C D)
  (h_midpoint : is_midpoint M A D)
  (h_intersect : intersect_at B M A C O)
  (h_ABM : angle_measure A B M = 55)
  (h_AMB : angle_measure A M B = 70)
  (h_BOC : angle_measure B O C = 80)
  (h_ADC : angle_measure A D C = 60) :
  angle_measure B C A = 35 := by sorry

end angle_BCA_measure_l2162_216205


namespace four_card_selection_ways_l2162_216236

/-- The number of cards in a standard deck -/
def deck_size : ℕ := 52

/-- The number of suits in a standard deck -/
def num_suits : ℕ := 4

/-- The number of cards in each suit -/
def cards_per_suit : ℕ := deck_size / num_suits

/-- The number of cards to choose -/
def cards_to_choose : ℕ := 4

/-- Theorem stating the number of ways to choose 4 cards from a standard deck
    with exactly two of the same suit and the other two of different suits -/
theorem four_card_selection_ways :
  (num_suits.choose 1) *
  ((num_suits - 1).choose 2) *
  (cards_per_suit.choose 2) *
  (cards_per_suit ^ 2) = 158004 := by
  sorry

end four_card_selection_ways_l2162_216236


namespace probability_multiple_5_or_7_l2162_216235

def is_multiple_of_5_or_7 (n : ℕ) : Bool :=
  n % 5 = 0 || n % 7 = 0

def count_multiples (n : ℕ) : ℕ :=
  (List.range n).filter is_multiple_of_5_or_7 |>.length

theorem probability_multiple_5_or_7 :
  count_multiples 50 / 50 = 8 / 25 := by
  sorry

end probability_multiple_5_or_7_l2162_216235


namespace total_students_is_184_l2162_216286

/-- Represents the number of students that can be transported in one car for a school --/
structure CarCapacity where
  capacity : ℕ

/-- Represents a school participating in the competition --/
structure School where
  students : ℕ
  carCapacity : CarCapacity

/-- Represents the state of both schools at a given point --/
structure CompetitionState where
  school1 : School
  school2 : School

/-- Checks if the given state satisfies the initial conditions --/
def initialConditionsSatisfied (state : CompetitionState) : Prop :=
  state.school1.students = state.school2.students ∧
  state.school1.carCapacity.capacity = 15 ∧
  state.school2.carCapacity.capacity = 13 ∧
  (state.school2.students + state.school2.carCapacity.capacity - 1) / state.school2.carCapacity.capacity =
    (state.school1.students / state.school1.carCapacity.capacity) + 1

/-- Checks if the given state satisfies the conditions after adding one student to each school --/
def middleConditionsSatisfied (state : CompetitionState) : Prop :=
  (state.school1.students + 1) / state.school1.carCapacity.capacity =
  (state.school2.students + 1) / state.school2.carCapacity.capacity

/-- Checks if the given state satisfies the final conditions --/
def finalConditionsSatisfied (state : CompetitionState) : Prop :=
  ((state.school1.students + 2) / state.school1.carCapacity.capacity) + 1 =
  (state.school2.students + 2) / state.school2.carCapacity.capacity

/-- The main theorem stating that under the given conditions, the total number of students is 184 --/
theorem total_students_is_184 (state : CompetitionState) :
  initialConditionsSatisfied state →
  middleConditionsSatisfied state →
  finalConditionsSatisfied state →
  state.school1.students + state.school2.students + 4 = 184 :=
by
  sorry

end total_students_is_184_l2162_216286


namespace bag_draw_comparison_l2162_216257

/-- A bag containing red and black balls -/
structure Bag where
  red : ℕ
  black : ℕ

/-- Random variable for drawing with replacement -/
def xi₁ (b : Bag) : ℕ → ℝ := sorry

/-- Random variable for drawing without replacement -/
def xi₂ (b : Bag) : ℕ → ℝ := sorry

/-- Expected value of a random variable -/
def expectation (X : ℕ → ℝ) : ℝ := sorry

/-- Variance of a random variable -/
def variance (X : ℕ → ℝ) : ℝ := sorry

/-- Theorem about expected values and variances of xi₁ and xi₂ -/
theorem bag_draw_comparison (b : Bag) (h : b.red = 1 ∧ b.black = 2) : 
  expectation (xi₁ b) = expectation (xi₂ b) ∧ 
  variance (xi₁ b) > variance (xi₂ b) := by sorry

end bag_draw_comparison_l2162_216257


namespace least_possible_value_z_minus_x_l2162_216229

theorem least_possible_value_z_minus_x
  (x y z : ℤ)
  (h1 : x < y ∧ y < z)
  (h2 : y - x > 9)
  (h3 : Even x)
  (h4 : Odd y ∧ Odd z) :
  ∀ w : ℤ, w ≥ 13 ∧ (∃ (a b c : ℤ), a < b ∧ b < c ∧ b - a > 9 ∧ Even a ∧ Odd b ∧ Odd c ∧ c - a = w) →
  z - x ≥ w :=
by sorry

end least_possible_value_z_minus_x_l2162_216229


namespace pages_per_comic_l2162_216227

theorem pages_per_comic (total_pages : ℕ) (initial_comics : ℕ) (final_comics : ℕ)
  (h1 : total_pages = 150)
  (h2 : initial_comics = 5)
  (h3 : final_comics = 11) :
  total_pages / (final_comics - initial_comics) = 25 := by
  sorry

end pages_per_comic_l2162_216227


namespace shooting_probabilities_l2162_216248

/-- Represents a shooter with a given probability of hitting the target -/
structure Shooter where
  hit_prob : ℝ
  hit_prob_nonneg : 0 ≤ hit_prob
  hit_prob_le_one : hit_prob ≤ 1

/-- The probability of both shooters hitting the target -/
def both_hit (a b : Shooter) : ℝ := a.hit_prob * b.hit_prob

/-- The probability of at least one shooter hitting the target -/
def at_least_one_hit (a b : Shooter) : ℝ := 1 - (1 - a.hit_prob) * (1 - b.hit_prob)

theorem shooting_probabilities (a b : Shooter) 
  (ha : a.hit_prob = 0.9) (hb : b.hit_prob = 0.8) : 
  both_hit a b = 0.72 ∧ at_least_one_hit a b = 0.98 := by
  sorry

end shooting_probabilities_l2162_216248


namespace two_solutions_l2162_216282

/-- The quadratic equation with absolute value term -/
def quadratic_abs_equation (x : ℝ) : Prop :=
  x^2 - |x| - 6 = 0

/-- The number of distinct real solutions to the equation -/
def num_solutions : ℕ := 2

/-- Theorem stating that the equation has exactly two distinct real solutions -/
theorem two_solutions :
  ∃ (a b : ℝ), a ≠ b ∧ 
  quadratic_abs_equation a ∧ 
  quadratic_abs_equation b ∧
  (∀ x : ℝ, quadratic_abs_equation x → x = a ∨ x = b) :=
sorry

end two_solutions_l2162_216282


namespace fraction_positivity_l2162_216260

theorem fraction_positivity (x : ℝ) : (x + 2) / ((x - 3)^3) > 0 ↔ x < -2 ∨ x > 3 := by
  sorry

end fraction_positivity_l2162_216260


namespace smallest_with_eight_odd_sixteen_even_divisors_l2162_216272

/-- Count of positive odd integer divisors of a number -/
def countOddDivisors (n : ℕ) : ℕ := sorry

/-- Count of positive even integer divisors of a number -/
def countEvenDivisors (n : ℕ) : ℕ := sorry

/-- Proposition: 3000 is the smallest positive integer with 8 odd and 16 even divisors -/
theorem smallest_with_eight_odd_sixteen_even_divisors :
  (∀ m : ℕ, m > 0 ∧ m < 3000 → 
    countOddDivisors m ≠ 8 ∨ countEvenDivisors m ≠ 16) ∧
  countOddDivisors 3000 = 8 ∧ 
  countEvenDivisors 3000 = 16 := by
  sorry

end smallest_with_eight_odd_sixteen_even_divisors_l2162_216272
