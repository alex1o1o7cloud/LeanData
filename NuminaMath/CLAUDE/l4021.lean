import Mathlib

namespace distance_between_points_l4021_402123

theorem distance_between_points (b : ℝ) :
  (∃ b₁ b₂ : ℝ, b = b₁ ∨ b = b₂) →
  (Real.sqrt ((3 * b - 1)^2 + (b + 1 - 4)^2) = 2 * Real.sqrt 13) →
  b * (5.47 - b) = 4.2 := by
sorry

end distance_between_points_l4021_402123


namespace favorite_numbers_sum_l4021_402100

/-- Given that Glory's favorite number is 450 and Misty's favorite number is 3 times smaller than Glory's,
    prove that the sum of their favorite numbers is 600. -/
theorem favorite_numbers_sum (glory_number : ℕ) (misty_number : ℕ)
    (h1 : glory_number = 450)
    (h2 : misty_number * 3 = glory_number) :
    misty_number + glory_number = 600 := by
  sorry

end favorite_numbers_sum_l4021_402100


namespace negative_fraction_comparison_l4021_402186

theorem negative_fraction_comparison : -3/2 < -2/3 := by
  sorry

end negative_fraction_comparison_l4021_402186


namespace moon_earth_distance_in_scientific_notation_l4021_402125

/-- The average distance from the moon to the earth in meters -/
def moon_earth_distance : ℝ := 384000000

/-- The scientific notation representation of the moon-earth distance -/
def moon_earth_distance_scientific : ℝ := 3.84 * (10 ^ 8)

theorem moon_earth_distance_in_scientific_notation :
  moon_earth_distance = moon_earth_distance_scientific := by
  sorry

end moon_earth_distance_in_scientific_notation_l4021_402125


namespace zeroes_at_end_of_600_times_50_l4021_402129

theorem zeroes_at_end_of_600_times_50 : ∃ n : ℕ, 600 * 50 = n * 10000 ∧ n % 10 ≠ 0 :=
by sorry

end zeroes_at_end_of_600_times_50_l4021_402129


namespace parabola_properties_l4021_402107

/-- A parabola is defined by the equation y = -x^2 + 1 --/
def parabola (x : ℝ) : ℝ := -x^2 + 1

theorem parabola_properties :
  (∀ x : ℝ, parabola x ≤ parabola 0) ∧ 
  parabola 0 = 1 :=
by sorry

end parabola_properties_l4021_402107


namespace parenthesized_subtraction_equality_l4021_402156

theorem parenthesized_subtraction_equality :
  1 - 2 - 3 - 4 - (5 - 6 - 7) = 0 := by
  sorry

end parenthesized_subtraction_equality_l4021_402156


namespace chord_intersection_lengths_l4021_402154

-- Define the circle and its properties
def circle_radius : ℝ := 6

-- Define the chord EJ and its properties
def chord_length : ℝ := 10

-- Define the point M where EJ intersects GH
def point_M (x : ℝ) : Prop := 0 < x ∧ x < 2 * circle_radius

-- Define the lengths of GM and MH
def length_GM (x : ℝ) : ℝ := x
def length_MH (x : ℝ) : ℝ := 2 * circle_radius - x

-- Theorem statement
theorem chord_intersection_lengths :
  ∃ x : ℝ, point_M x ∧ 
    length_GM x = 6 + Real.sqrt 11 ∧
    length_MH x = 6 - Real.sqrt 11 :=
sorry

end chord_intersection_lengths_l4021_402154


namespace intersection_of_sets_l4021_402143

theorem intersection_of_sets : 
  let A : Set ℕ := {1, 2, 3}
  let B : Set ℕ := {3, 4}
  A ∩ B = {3} := by sorry

end intersection_of_sets_l4021_402143


namespace simplify_expression_l4021_402170

theorem simplify_expression : (4 + 3) + (8 - 3 - 1) = 11 := by
  sorry

end simplify_expression_l4021_402170


namespace bacteria_after_three_hours_l4021_402124

/-- Represents the number of bacteria after a given time -/
def bacteria_count (initial_count : ℕ) (split_interval : ℕ) (total_time : ℕ) : ℕ :=
  initial_count * 2 ^ (total_time / split_interval)

/-- Theorem stating that the number of bacteria after 3 hours is 64 -/
theorem bacteria_after_three_hours :
  bacteria_count 1 30 180 = 64 := by
  sorry

#check bacteria_after_three_hours

end bacteria_after_three_hours_l4021_402124


namespace speed_limit_calculation_l4021_402187

/-- Proves that given a distance of 150 miles traveled in 2 hours,
    and driving 15 mph above the speed limit, the speed limit is 60 mph. -/
theorem speed_limit_calculation (distance : ℝ) (time : ℝ) (speed_above_limit : ℝ) 
    (h1 : distance = 150)
    (h2 : time = 2)
    (h3 : speed_above_limit = 15) :
    distance / time - speed_above_limit = 60 := by
  sorry

end speed_limit_calculation_l4021_402187


namespace quiche_cost_is_15_l4021_402190

/-- Represents the cost of a single quiche -/
def quiche_cost : ℝ := sorry

/-- Represents the number of quiches ordered -/
def num_quiches : ℕ := 2

/-- Represents the cost of a single croissant -/
def croissant_cost : ℝ := 3

/-- Represents the number of croissants ordered -/
def num_croissants : ℕ := 6

/-- Represents the cost of a single biscuit -/
def biscuit_cost : ℝ := 2

/-- Represents the number of biscuits ordered -/
def num_biscuits : ℕ := 6

/-- Represents the discount rate -/
def discount_rate : ℝ := 0.1

/-- Represents the discounted total cost -/
def discounted_total : ℝ := 54

/-- Theorem stating that the cost of each quiche is $15 -/
theorem quiche_cost_is_15 : quiche_cost = 15 := by
  sorry

end quiche_cost_is_15_l4021_402190


namespace layla_babysitting_earnings_l4021_402105

theorem layla_babysitting_earnings :
  let donaldson_rate : ℕ := 15
  let merck_rate : ℕ := 18
  let hille_rate : ℕ := 20
  let johnson_rate : ℕ := 22
  let ramos_rate : ℕ := 25
  let donaldson_hours : ℕ := 7
  let merck_hours : ℕ := 6
  let hille_hours : ℕ := 3
  let johnson_hours : ℕ := 4
  let ramos_hours : ℕ := 2
  donaldson_rate * donaldson_hours +
  merck_rate * merck_hours +
  hille_rate * hille_hours +
  johnson_rate * johnson_hours +
  ramos_rate * ramos_hours = 411 :=
by sorry

end layla_babysitting_earnings_l4021_402105


namespace error_percentage_division_vs_multiplication_error_percentage_proof_l4021_402111

theorem error_percentage_division_vs_multiplication : ℝ → Prop :=
  fun x => x ≠ 0 →
    let correct_result := 5 * x
    let incorrect_result := x / 10
    let error := correct_result - incorrect_result
    let percentage_error := (error / correct_result) * 100
    percentage_error = 98

-- The proof is omitted
theorem error_percentage_proof : ∀ x : ℝ, error_percentage_division_vs_multiplication x :=
sorry

end error_percentage_division_vs_multiplication_error_percentage_proof_l4021_402111


namespace sum_and_product_identities_l4021_402104

theorem sum_and_product_identities (a b : ℝ) (h1 : a + b = 2) (h2 : a * b = -1) : 
  (a^2 + b^2 = 6) ∧ ((a - b)^2 = 8) := by sorry

end sum_and_product_identities_l4021_402104


namespace pizza_slices_per_pizza_l4021_402193

theorem pizza_slices_per_pizza 
  (num_people : ℕ) 
  (slices_per_person : ℕ) 
  (num_pizzas : ℕ) 
  (h1 : num_people = 10) 
  (h2 : slices_per_person = 2) 
  (h3 : num_pizzas = 5) : 
  (num_people * slices_per_person) / num_pizzas = 4 :=
by sorry

end pizza_slices_per_pizza_l4021_402193


namespace parallel_lines_a_value_l4021_402194

/-- Given two parallel lines y = (a - a^2)x - 2 and y = (3a + 1)x + 1, prove that a = -1 -/
theorem parallel_lines_a_value (a : ℝ) : 
  (∀ x y : ℝ, y = (a - a^2) * x - 2 ↔ y = (3*a + 1) * x + 1) → a = -1 := by
  sorry

end parallel_lines_a_value_l4021_402194


namespace stating_broken_flagpole_height_l4021_402191

/-- Represents a broken flagpole scenario -/
structure BrokenFlagpole where
  initial_height : ℝ
  distance_from_base : ℝ
  break_height : ℝ

/-- 
Theorem stating that for a flagpole of height 8 meters, if it breaks at a point x meters 
above the ground and the upper part touches the ground 3 meters away from the base, 
then x = √73 / 2.
-/
theorem broken_flagpole_height (f : BrokenFlagpole) 
    (h1 : f.initial_height = 8)
    (h2 : f.distance_from_base = 3) :
  f.break_height = Real.sqrt 73 / 2 := by
  sorry


end stating_broken_flagpole_height_l4021_402191


namespace grandma_salad_ratio_l4021_402138

/-- Proves that the ratio of cherry tomatoes to mushrooms is 2:1 given the conditions of Grandma's salad --/
theorem grandma_salad_ratio : ∀ (cherry_tomatoes pickles bacon_bits : ℕ),
  pickles = 4 * cherry_tomatoes →
  bacon_bits = 4 * pickles →
  bacon_bits / 3 = 32 →
  cherry_tomatoes / 3 = 2 :=
by
  sorry

#check grandma_salad_ratio

end grandma_salad_ratio_l4021_402138


namespace company_average_salary_associates_avg_salary_l4021_402144

theorem company_average_salary 
  (num_managers : ℕ) 
  (num_associates : ℕ) 
  (avg_salary_managers : ℚ) 
  (avg_salary_company : ℚ) : ℚ :=
  let total_employees := num_managers + num_associates
  let total_salary := avg_salary_company * total_employees
  let managers_salary := avg_salary_managers * num_managers
  let associates_salary := total_salary - managers_salary
  associates_salary / num_associates

theorem associates_avg_salary 
  (h1 : company_average_salary 15 75 90000 40000 = 30000) : 
  company_average_salary 15 75 90000 40000 = 30000 := by
  sorry

end company_average_salary_associates_avg_salary_l4021_402144


namespace least_integer_square_36_more_than_thrice_neg_six_satisfies_equation_least_integer_square_36_more_than_thrice_is_neg_six_l4021_402198

theorem least_integer_square_36_more_than_thrice (x : ℤ) : x^2 = 3*x + 36 → x ≥ -6 := by
  sorry

theorem neg_six_satisfies_equation : (-6 : ℤ)^2 = 3*(-6) + 36 := by
  sorry

theorem least_integer_square_36_more_than_thrice_is_neg_six :
  ∃ (x : ℤ), x^2 = 3*x + 36 ∧ ∀ (y : ℤ), y^2 = 3*y + 36 → y ≥ x := by
  sorry

end least_integer_square_36_more_than_thrice_neg_six_satisfies_equation_least_integer_square_36_more_than_thrice_is_neg_six_l4021_402198


namespace playground_teachers_l4021_402122

theorem playground_teachers (boys girls : ℕ) (h1 : boys = 57) (h2 : girls = 82)
  (h3 : girls = boys + teachers + 13) : teachers = 12 := by
  sorry

end playground_teachers_l4021_402122


namespace equation_equivalence_l4021_402195

theorem equation_equivalence (x : ℝ) (Q : ℝ) (h : 5 * (3 * x - 4 * Real.pi) = Q) :
  10 * (6 * x - 8 * Real.pi) = 4 * Q := by
  sorry

end equation_equivalence_l4021_402195


namespace fraction_equality_l4021_402126

theorem fraction_equality (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : (4 * x - 3 * y) / (x + 4 * y) = 3) : 
  (x - 4 * y) / (4 * x + 3 * y) = 11 / 63 := by
  sorry

end fraction_equality_l4021_402126


namespace quadratic_inequality_l4021_402121

/-- A quadratic function with axis of symmetry at x = 2 -/
def f (b c : ℝ) (x : ℝ) : ℝ := x^2 + b*x + c

/-- The axis of symmetry is at x = 2 -/
def axis_of_symmetry (b : ℝ) : ℝ := 2

theorem quadratic_inequality (b c : ℝ) : 
  f b c (axis_of_symmetry b) < f b c 1 ∧ f b c 1 < f b c 4 := by sorry

end quadratic_inequality_l4021_402121


namespace cubic_equation_root_l4021_402139

theorem cubic_equation_root (a b : ℚ) : 
  (3 + Real.sqrt 5) ^ 3 + a * (3 + Real.sqrt 5) ^ 2 + b * (3 + Real.sqrt 5) + 20 = 0 → 
  b = -26 := by
sorry

end cubic_equation_root_l4021_402139


namespace water_bill_calculation_l4021_402165

def weekly_income : ℝ := 500
def tax_rate : ℝ := 0.10
def tithe_rate : ℝ := 0.10
def remaining_amount : ℝ := 345

theorem water_bill_calculation :
  let after_tax := weekly_income * (1 - tax_rate)
  let after_tithe := after_tax - (weekly_income * tithe_rate)
  let water_bill := after_tithe - remaining_amount
  water_bill = 55 := by sorry

end water_bill_calculation_l4021_402165


namespace negation_of_existence_negation_of_quadratic_equation_l4021_402134

theorem negation_of_existence (p : ℝ → Prop) : (¬ ∃ x, p x) ↔ (∀ x, ¬ p x) := by sorry

theorem negation_of_quadratic_equation :
  (¬ ∃ x : ℝ, x^2 + x + 1 = 0) ↔ (∀ x : ℝ, x^2 + x + 1 ≠ 0) := by sorry

end negation_of_existence_negation_of_quadratic_equation_l4021_402134


namespace kath_friends_count_l4021_402168

/-- The number of friends Kath took to the movie --/
def num_friends : ℕ :=
  -- Define this value
  sorry

/-- The number of Kath's siblings --/
def num_siblings : ℕ := 2

/-- The regular admission cost in dollars --/
def regular_cost : ℕ := 8

/-- The discount amount in dollars --/
def discount : ℕ := 3

/-- The total amount Kath paid in dollars --/
def total_paid : ℕ := 30

/-- The actual cost per person after discount --/
def discounted_cost : ℕ := regular_cost - discount

/-- The total number of people in Kath's group --/
def total_people : ℕ := total_paid / discounted_cost

theorem kath_friends_count :
  num_friends = total_people - (num_siblings + 1) ∧
  num_friends = 3 :=
sorry

end kath_friends_count_l4021_402168


namespace ordered_pairs_theorem_l4021_402141

def S : Set (ℕ × ℕ) := {(8, 4), (9, 3), (2, 1)}

def satisfies_conditions (pair : ℕ × ℕ) : Prop :=
  let (x, y) := pair
  x > y ∧ (x - y = 2 * x / y ∨ x - y = 2 * y / x)

theorem ordered_pairs_theorem :
  ∀ (pair : ℕ × ℕ), pair ∈ S ↔ satisfies_conditions pair ∧ pair.1 > 0 ∧ pair.2 > 0 :=
sorry

end ordered_pairs_theorem_l4021_402141


namespace product_xyz_is_zero_l4021_402127

theorem product_xyz_is_zero 
  (x y z : ℝ) 
  (h1 : x + 2/y = 2) 
  (h2 : y + 2/z = 2) : 
  x * y * z = 0 := by
sorry

end product_xyz_is_zero_l4021_402127


namespace rhombus_area_l4021_402128

/-- The area of a rhombus with vertices at (0, 3.5), (10, 0), (0, -3.5), and (-10, 0) is 70 square units. -/
theorem rhombus_area : 
  let vertices : List (ℝ × ℝ) := [(0, 3.5), (10, 0), (0, -3.5), (-10, 0)]
  let diag1 : ℝ := |3.5 - (-3.5)|
  let diag2 : ℝ := |10 - (-10)|
  (diag1 * diag2) / 2 = 70 := by
  sorry

#check rhombus_area

end rhombus_area_l4021_402128


namespace max_sum_after_swap_l4021_402131

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  ones : Nat
  is_valid : hundreds ≥ 1 ∧ hundreds ≤ 9 ∧ tens ≥ 0 ∧ tens ≤ 9 ∧ ones ≥ 0 ∧ ones ≤ 9

/-- Converts a ThreeDigitNumber to its numerical value -/
def ThreeDigitNumber.toNat (n : ThreeDigitNumber) : Nat :=
  100 * n.hundreds + 10 * n.tens + n.ones

/-- Swaps the first and last digits of a ThreeDigitNumber -/
def ThreeDigitNumber.swap (n : ThreeDigitNumber) : ThreeDigitNumber where
  hundreds := n.ones
  tens := n.tens
  ones := n.hundreds
  is_valid := by sorry

/-- The main theorem to prove -/
theorem max_sum_after_swap
  (a b c : ThreeDigitNumber)
  (h : a.toNat + b.toNat + c.toNat = 2019) :
  (a.swap.toNat + b.swap.toNat + c.swap.toNat) ≤ 2118 :=
by sorry

end max_sum_after_swap_l4021_402131


namespace workers_wage_before_promotion_l4021_402189

theorem workers_wage_before_promotion (wage_increase_percentage : ℝ) (new_wage : ℝ) : 
  wage_increase_percentage = 0.60 →
  new_wage = 45 →
  (1 + wage_increase_percentage) * (new_wage / (1 + wage_increase_percentage)) = 28.125 := by
sorry

end workers_wage_before_promotion_l4021_402189


namespace willy_finishes_series_in_30_days_l4021_402175

/-- Calculates the number of days needed to finish a TV series -/
def days_to_finish_series (seasons : ℕ) (episodes_per_season : ℕ) (episodes_per_day : ℕ) : ℕ :=
  (seasons * episodes_per_season) / episodes_per_day

/-- Proves that it takes 30 days to finish the given TV series -/
theorem willy_finishes_series_in_30_days :
  days_to_finish_series 3 20 2 = 30 := by
  sorry

end willy_finishes_series_in_30_days_l4021_402175


namespace rectangle_ratio_l4021_402171

/-- Given a rectangle divided into three congruent smaller rectangles,
    where each smaller rectangle is similar to the large rectangle,
    the ratio of the longer side to the shorter side is √3 : 1 for all rectangles. -/
theorem rectangle_ratio (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h_similar : x / y = (3 * y) / x) : x / y = Real.sqrt 3 := by
  sorry

end rectangle_ratio_l4021_402171


namespace city_mpg_is_24_l4021_402176

/-- Represents the fuel efficiency of a car in different driving conditions. -/
structure CarFuelEfficiency where
  highway_miles_per_tankful : ℝ
  city_miles_per_tankful : ℝ
  highway_city_mpg_difference : ℝ

/-- Calculates the city miles per gallon given the car's fuel efficiency data. -/
def city_mpg (car : CarFuelEfficiency) : ℝ :=
  -- The actual calculation is not provided here
  sorry

/-- Theorem stating that for the given car fuel efficiency data, 
    the city miles per gallon is 24. -/
theorem city_mpg_is_24 (car : CarFuelEfficiency) 
  (h1 : car.highway_miles_per_tankful = 462)
  (h2 : car.city_miles_per_tankful = 336)
  (h3 : car.highway_city_mpg_difference = 9) :
  city_mpg car = 24 := by
  sorry

end city_mpg_is_24_l4021_402176


namespace P_intersect_M_l4021_402181

def P : Set ℤ := {x : ℤ | 0 ≤ x ∧ x < 3}
def M : Set ℤ := {x : ℤ | x^2 ≤ 9}

theorem P_intersect_M : P ∩ M = {0, 1, 2} := by
  sorry

end P_intersect_M_l4021_402181


namespace pirate_theorem_l4021_402135

/-- Represents a pirate in the group -/
structure Pirate where
  id : Nat
  targets : Finset Nat

/-- Represents the group of pirates -/
def PirateGroup := Finset Pirate

/-- Counts the number of pirates killed in a given order -/
def countKilled (group : PirateGroup) (order : List Pirate) : Nat :=
  sorry

/-- Main theorem: If there exists an order where 28 pirates are killed,
    then in any other order, at least 10 pirates must be killed -/
theorem pirate_theorem (group : PirateGroup) :
  (∃ order : List Pirate, countKilled group order = 28) →
  (∀ order : List Pirate, countKilled group order ≥ 10) :=
by
  sorry


end pirate_theorem_l4021_402135


namespace area_of_rectangle_in_18_gon_l4021_402163

/-- Given a regular 18-sided polygon with area 2016 square centimeters,
    the area of a rectangle formed by connecting the midpoints of four adjacent sides
    is 448 square centimeters. -/
theorem area_of_rectangle_in_18_gon (A : ℝ) (h : A = 2016) :
  let rectangle_area := A / 18 * 4
  rectangle_area = 448 :=
by sorry

end area_of_rectangle_in_18_gon_l4021_402163


namespace divisor_problem_l4021_402158

theorem divisor_problem : ∃ (d : ℕ), d > 0 ∧ (1019 + 6) % d = 0 ∧ d = 5 := by
  sorry

end divisor_problem_l4021_402158


namespace caviar_cost_calculation_l4021_402184

/-- The cost of caviar per person for Alex's New Year's Eve appetizer -/
def caviar_cost (chips_cost creme_fraiche_cost total_cost : ℚ) : ℚ :=
  total_cost - (chips_cost + creme_fraiche_cost)

/-- Theorem stating the cost of caviar per person -/
theorem caviar_cost_calculation :
  caviar_cost 3 5 27 = 19 := by
  sorry

end caviar_cost_calculation_l4021_402184


namespace pelican_fish_count_l4021_402137

theorem pelican_fish_count (P : ℕ) : 
  (P + 7 = P + 7) →  -- Kingfisher caught 7 more fish than the pelican
  (3 * (P + (P + 7)) = P + 86) →  -- Fisherman caught 3 times the total and 86 more than the pelican
  P = 13 := by
sorry

end pelican_fish_count_l4021_402137


namespace smallest_12_digit_with_all_digits_div_36_proof_l4021_402116

def is_12_digit (n : ℕ) : Prop := 10^11 ≤ n ∧ n < 10^12

def contains_all_digits (n : ℕ) : Prop :=
  ∀ d : ℕ, d < 10 → ∃ k : ℕ, n / 10^k % 10 = d

def smallest_12_digit_with_all_digits_div_36 : ℕ := 100023457896

theorem smallest_12_digit_with_all_digits_div_36_proof :
  (is_12_digit smallest_12_digit_with_all_digits_div_36) ∧
  (contains_all_digits smallest_12_digit_with_all_digits_div_36) ∧
  (smallest_12_digit_with_all_digits_div_36 % 36 = 0) ∧
  (∀ m : ℕ, m < smallest_12_digit_with_all_digits_div_36 →
    ¬(is_12_digit m ∧ contains_all_digits m ∧ m % 36 = 0)) :=
by sorry

end smallest_12_digit_with_all_digits_div_36_proof_l4021_402116


namespace complex_expression_simplification_l4021_402149

/-- Given two real numbers x and y, where x ≠ 0, y ≠ 0, and x ≠ ±y, 
    prove that the given complex expression simplifies to (x-y)^(1/3) / (x+y) -/
theorem complex_expression_simplification (x y : ℝ) 
  (hx : x ≠ 0) (hy : y ≠ 0) (hxy : x ≠ y ∧ x ≠ -y) :
  let numerator := (x^9 - x^6*y^3)^(1/3) - y^2 * ((8*x^6/y^3 - 8*x^3))^(1/3) + 
                   x*y^3 * (y^3 - y^6/x^3)^(1/2)
  let denominator := x^(8/3)*(x^2 - 2*y^2) + (x^2*y^12)^(1/3)
  numerator / denominator = (x-y)^(1/3) / (x+y) := by
  sorry

end complex_expression_simplification_l4021_402149


namespace spinner_probability_l4021_402109

theorem spinner_probability (pA pB pC pD : ℚ) : 
  pA = 1/4 → pB = 1/3 → pD = 1/6 → pA + pB + pC + pD = 1 → pC = 1/4 := by
  sorry

end spinner_probability_l4021_402109


namespace paths_A_to_D_l4021_402117

/-- Represents a point in the graph --/
inductive Point : Type
| A : Point
| B : Point
| C : Point
| D : Point

/-- Represents a direct path between two points --/
inductive DirectPath : Point → Point → Type
| AB : DirectPath Point.A Point.B
| BC : DirectPath Point.B Point.C
| CD : DirectPath Point.C Point.D
| AC : DirectPath Point.A Point.C
| BD : DirectPath Point.B Point.D

/-- Counts the number of paths between two points --/
def countPaths (start finish : Point) : ℕ :=
  sorry

/-- The main theorem stating that there are 12 paths from A to D --/
theorem paths_A_to_D :
  countPaths Point.A Point.D = 12 :=
by
  sorry

end paths_A_to_D_l4021_402117


namespace emily_lost_lives_l4021_402112

theorem emily_lost_lives (initial_lives : ℕ) (lives_gained : ℕ) (final_lives : ℕ) : 
  initial_lives = 42 → lives_gained = 24 → final_lives = 41 → 
  ∃ (lives_lost : ℕ), initial_lives - lives_lost + lives_gained = final_lives ∧ lives_lost = 25 := by
sorry

end emily_lost_lives_l4021_402112


namespace find_x_value_l4021_402130

def A (x : ℕ) : Set ℕ := {1, 4, x}
def B (x : ℕ) : Set ℕ := {1, x^2}

theorem find_x_value (x : ℕ) (h : A x ∪ B x = A x) : x = 0 := by
  sorry

end find_x_value_l4021_402130


namespace different_city_probability_l4021_402185

theorem different_city_probability (pA_cityA pB_cityA : ℝ) 
  (h1 : 0 ≤ pA_cityA ∧ pA_cityA ≤ 1)
  (h2 : 0 ≤ pB_cityA ∧ pB_cityA ≤ 1)
  (h3 : pA_cityA = 0.6)
  (h4 : pB_cityA = 0.2) :
  (pA_cityA * (1 - pB_cityA)) + ((1 - pA_cityA) * pB_cityA) = 0.56 := by
sorry

end different_city_probability_l4021_402185


namespace survey_respondents_l4021_402179

theorem survey_respondents (preferred_x : ℕ) (ratio_x : ℕ) (ratio_y : ℕ) : 
  preferred_x = 360 → ratio_x = 9 → ratio_y = 1 → 
  ∃ (total : ℕ), total = preferred_x + (preferred_x * ratio_y) / ratio_x ∧ total = 400 :=
by
  sorry

end survey_respondents_l4021_402179


namespace quadratic_function_condition_l4021_402115

theorem quadratic_function_condition (m : ℝ) : (|m - 2| = 2 ∧ m - 4 ≠ 0) ↔ m = 0 := by
  sorry

end quadratic_function_condition_l4021_402115


namespace multiplication_equation_solution_l4021_402180

theorem multiplication_equation_solution : ∃ x : ℕ, 80641 * x = 806006795 ∧ x = 9995 := by
  sorry

end multiplication_equation_solution_l4021_402180


namespace students_playing_neither_l4021_402167

/-- Theorem: In a class of 39 students, where 26 play football, 20 play tennis, and 17 play both,
    the number of students who play neither football nor tennis is 10. -/
theorem students_playing_neither (N F T B : ℕ) 
  (h_total : N = 39)
  (h_football : F = 26)
  (h_tennis : T = 20)
  (h_both : B = 17) :
  N - (F + T - B) = 10 := by
  sorry

end students_playing_neither_l4021_402167


namespace divisibility_problem_l4021_402169

theorem divisibility_problem (a : ℕ) :
  (∃! n : Fin 4, ¬ (
    (n = 0 → a % 2 = 0) ∧
    (n = 1 → a % 4 = 0) ∧
    (n = 2 → a % 12 = 0) ∧
    (n = 3 → a % 24 = 0)
  )) →
  ¬(a % 24 = 0) :=
by sorry

end divisibility_problem_l4021_402169


namespace hyperbola_focal_length_l4021_402160

/-- The focal length of a hyperbola with equation x²/9 - y²/4 = 1 is 2√13 -/
theorem hyperbola_focal_length : 
  ∀ (x y : ℝ), x^2/9 - y^2/4 = 1 → 
  ∃ (f : ℝ), f = 2 * Real.sqrt 13 ∧ f = 2 * Real.sqrt ((9 : ℝ) + (4 : ℝ)) := by
  sorry

end hyperbola_focal_length_l4021_402160


namespace circle_center_correct_l4021_402140

/-- The equation of a circle in the xy-plane -/
def CircleEquation (x y : ℝ) : Prop :=
  x^2 + 4*x + y^2 - 6*y - 12 = 0

/-- The center of a circle -/
def CircleCenter : ℝ × ℝ := (-2, 3)

/-- Theorem: The center of the circle given by the equation x^2 + 4x + y^2 - 6y - 12 = 0 is (-2, 3) -/
theorem circle_center_correct :
  ∀ (x y : ℝ), CircleEquation x y ↔ (x + 2)^2 + (y - 3)^2 = 25 :=
by sorry

end circle_center_correct_l4021_402140


namespace M_remainder_mod_45_l4021_402106

def M : ℕ := sorry

theorem M_remainder_mod_45 : M % 45 = 15 := by sorry

end M_remainder_mod_45_l4021_402106


namespace polar_to_cartesian_circle_l4021_402183

/-- Given a curve C with polar equation ρ = 2cos(θ), 
    its Cartesian coordinate equation is x² + y² - 2x = 0 -/
theorem polar_to_cartesian_circle (x y : ℝ) :
  (∃ θ : ℝ, x = 2 * Real.cos θ * Real.cos θ ∧ y = 2 * Real.cos θ * Real.sin θ) ↔ 
  x^2 + y^2 - 2*x = 0 := by sorry

end polar_to_cartesian_circle_l4021_402183


namespace empty_solution_set_implies_a_range_l4021_402188

/-- The function f(x) = x^2 + (1-a)x - a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + (1-a)*x - a

/-- Theorem stating that if the solution set of f(f(x)) < 0 is empty,
    then -3 ≤ a ≤ 2√2 - 3 -/
theorem empty_solution_set_implies_a_range (a : ℝ) :
  (∀ x : ℝ, f a (f a x) ≥ 0) → -3 ≤ a ∧ a ≤ 2 * Real.sqrt 2 - 3 :=
by sorry


end empty_solution_set_implies_a_range_l4021_402188


namespace trig_expression_equals_one_l4021_402119

theorem trig_expression_equals_one :
  let expr := (Real.sin (24 * π / 180) * Real.cos (16 * π / 180) + 
               Real.cos (156 * π / 180) * Real.cos (96 * π / 180)) /
              (Real.sin (26 * π / 180) * Real.cos (14 * π / 180) + 
               Real.cos (154 * π / 180) * Real.cos (94 * π / 180))
  expr = 1 := by
sorry

end trig_expression_equals_one_l4021_402119


namespace printer_cost_l4021_402102

/-- The cost of a printer given the conditions of the merchant's purchase. -/
theorem printer_cost (total_cost : ℕ) (keyboard_cost : ℕ) (num_keyboards : ℕ) (num_printers : ℕ)
  (h1 : total_cost = 2050)
  (h2 : keyboard_cost = 20)
  (h3 : num_keyboards = 15)
  (h4 : num_printers = 25) :
  (total_cost - num_keyboards * keyboard_cost) / num_printers = 70 := by
  sorry

end printer_cost_l4021_402102


namespace milk_container_problem_l4021_402164

-- Define the capacity of container A
def A : ℝ := 1184

-- Define the quantity of milk in container B after initial pouring
def B : ℝ := 0.375 * A

-- Define the quantity of milk in container C after initial pouring
def C : ℝ := 0.625 * A

-- Define the amount transferred from C to B
def transfer : ℝ := 148

-- Theorem statement
theorem milk_container_problem :
  -- After transfer, B and C have equal quantities
  B + transfer = C - transfer ∧
  -- The sum of B and C equals A
  B + C = A ∧
  -- A is 1184 liters
  A = 1184 := by
  sorry

end milk_container_problem_l4021_402164


namespace smallest_perimeter_l4021_402155

/-- A triangle with consecutive integer side lengths, where the smallest side is greater than 2 -/
structure ConsecutiveIntegerTriangle where
  n : ℕ
  gt_two : n > 2

/-- The perimeter of a ConsecutiveIntegerTriangle -/
def perimeter (t : ConsecutiveIntegerTriangle) : ℕ := t.n + (t.n + 1) + (t.n + 2)

/-- Predicate to check if a ConsecutiveIntegerTriangle is valid (satisfies triangle inequality) -/
def is_valid_triangle (t : ConsecutiveIntegerTriangle) : Prop :=
  t.n + (t.n + 1) > t.n + 2 ∧
  t.n + (t.n + 2) > t.n + 1 ∧
  (t.n + 1) + (t.n + 2) > t.n

theorem smallest_perimeter :
  ∃ (t : ConsecutiveIntegerTriangle), 
    is_valid_triangle t ∧ 
    perimeter t = 12 ∧ 
    (∀ (t' : ConsecutiveIntegerTriangle), is_valid_triangle t' → perimeter t' ≥ 12) :=
sorry

end smallest_perimeter_l4021_402155


namespace shaded_area_ratio_l4021_402108

/-- The area of the region in a square of side length 5 bounded by lines from (0,0) to (2.5,5) and from (5,5) to (0,2.5) is half the area of the square. -/
theorem shaded_area_ratio (square_side : ℝ) (h : square_side = 5) : 
  let shaded_area := (1/2 * 2.5 * 2.5) + (2.5 * 2.5) + (1/2 * 2.5 * 2.5)
  shaded_area / (square_side ^ 2) = 1/2 := by
sorry

end shaded_area_ratio_l4021_402108


namespace converse_even_sum_l4021_402120

theorem converse_even_sum (a b : ℤ) : 
  (∀ a b : ℤ, (Even a ∧ Even b) → Even (a + b)) →
  (∀ a b : ℤ, Even (a + b) → (Even a ∧ Even b)) :=
sorry

end converse_even_sum_l4021_402120


namespace two_satisfying_functions_l4021_402178

/-- A function satisfying the given functional equation -/
def SatisfyingFunction (f : ℝ → ℝ) : Prop :=
  ∃ c : ℝ, ∀ x y : ℝ, f (x + y) * f (x - y) = (f x + f y)^2 - 4 * x^2 * f y + c

/-- The set of all functions satisfying the functional equation -/
def SatisfyingFunctions : Set (ℝ → ℝ) :=
  {f | SatisfyingFunction f}

/-- The constant zero function -/
def ZeroFunction : ℝ → ℝ := λ _ => 0

/-- The square function -/
def SquareFunction : ℝ → ℝ := λ x => x^2

theorem two_satisfying_functions :
  SatisfyingFunctions = {ZeroFunction, SquareFunction} := by sorry

end two_satisfying_functions_l4021_402178


namespace town_population_l4021_402142

def present_population : ℝ → Prop :=
  λ p => (1 + 0.04) * p = 1289.6

theorem town_population : ∃ p : ℝ, present_population p ∧ p = 1240 :=
  sorry

end town_population_l4021_402142


namespace sum_of_digits_M_l4021_402152

/-- M is a positive integer such that M^2 = 36^50 * 50^36 -/
def M : ℕ+ := sorry

/-- Function to calculate the sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem stating that the sum of digits of M is 21 -/
theorem sum_of_digits_M : sum_of_digits M.val = 21 := by sorry

end sum_of_digits_M_l4021_402152


namespace geometric_sequence_common_ratio_l4021_402166

def geometric_sequence (a : ℕ → ℚ) : Prop :=
  ∃ r : ℚ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_common_ratio
  (a : ℕ → ℚ)
  (h_geo : geometric_sequence a)
  (h_1 : a 0 = 32)
  (h_2 : a 1 = -48)
  (h_3 : a 2 = 72)
  (h_4 : a 3 = -108)
  (h_5 : a 4 = 162) :
  ∃ r : ℚ, r = -3/2 ∧ ∀ n : ℕ, a (n + 1) = r * a n :=
by sorry

end geometric_sequence_common_ratio_l4021_402166


namespace pill_cost_calculation_l4021_402151

/-- The cost of one pill in dollars -/
def pill_cost : ℝ := 1.50

/-- The number of pills John takes per day -/
def pills_per_day : ℕ := 2

/-- The number of days in a month -/
def days_in_month : ℕ := 30

/-- The percentage of the cost that John pays (insurance covers the rest) -/
def john_payment_percentage : ℝ := 0.60

/-- The amount John pays for pills in a month in dollars -/
def john_monthly_payment : ℝ := 54

theorem pill_cost_calculation :
  pill_cost = john_monthly_payment / (pills_per_day * days_in_month * john_payment_percentage) :=
sorry

end pill_cost_calculation_l4021_402151


namespace geometric_sequence_problem_l4021_402182

theorem geometric_sequence_problem (a₁ q aₙ : ℚ) (n : ℕ) :
  a₁ = 9/8 ∧ q = 2/3 ∧ aₙ = 1/3 ∧ aₙ = a₁ * q^(n-1) → n = 4 :=
by sorry

end geometric_sequence_problem_l4021_402182


namespace cost_price_percentage_l4021_402192

theorem cost_price_percentage (selling_price cost_price : ℝ) :
  selling_price > 0 →
  cost_price > 0 →
  (selling_price - cost_price) / cost_price = 11.11111111111111 / 100 →
  cost_price / selling_price = 9 / 10 := by
sorry

end cost_price_percentage_l4021_402192


namespace problem_l4021_402150

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x + a^(-x)

theorem problem (a : ℝ) (h1 : a > 1) (h2 : f a 1 = 3) :
  (f a 2 = 7) ∧
  (∀ x₁ x₂, 0 ≤ x₂ ∧ x₂ < x₁ → f a x₁ > f a x₂) ∧
  (∀ m x, 0 ≤ x ∧ x ≤ 1 → 
    f a (2*x) - m * f a x ≥ min (2 - 2*m) (min (-m^2/4 - 2) (7 - 3*m))) := by
  sorry

end problem_l4021_402150


namespace non_monotonic_interval_l4021_402153

-- Define the function
def f (x : ℝ) : ℝ := |2*x - 1|

-- Define the property of being non-monotonic in an interval
def is_non_monotonic (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ x y z, a < x ∧ x < y ∧ y < z ∧ z < b ∧ 
  ((f x < f y ∧ f y > f z) ∨ (f x > f y ∧ f y < f z))

-- State the theorem
theorem non_monotonic_interval (k : ℝ) :
  is_non_monotonic f (k - 1) (k + 1) ↔ -1 < k ∧ k < 1 :=
sorry

end non_monotonic_interval_l4021_402153


namespace pedoes_inequality_pedoes_inequality_equality_condition_l4021_402147

/-- Pedoe's inequality for triangles -/
theorem pedoes_inequality (a b c a₁ b₁ c₁ Δ Δ₁ : ℝ) 
  (h_abc : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_a₁b₁c₁ : 0 < a₁ ∧ 0 < b₁ ∧ 0 < c₁)
  (h_Δ : 0 < Δ)
  (h_Δ₁ : 0 < Δ₁)
  (h_abc_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_a₁b₁c₁_triangle : a₁ + b₁ > c₁ ∧ b₁ + c₁ > a₁ ∧ c₁ + a₁ > b₁)
  (h_Δ_def : Δ = Real.sqrt ((a + b + c) * (b + c - a) * (c + a - b) * (a + b - c)) / 4)
  (h_Δ₁_def : Δ₁ = Real.sqrt ((a₁ + b₁ + c₁) * (b₁ + c₁ - a₁) * (c₁ + a₁ - b₁) * (a₁ + b₁ - c₁)) / 4) :
  a^2 * (b₁^2 + c₁^2 - a₁^2) + b^2 * (c₁^2 + a₁^2 - b₁^2) + c^2 * (a₁^2 + b₁^2 - c₁^2) ≥ 16 * Δ * Δ₁ :=
by sorry

/-- Condition for equality in Pedoe's inequality -/
theorem pedoes_inequality_equality_condition (a b c a₁ b₁ c₁ Δ Δ₁ : ℝ) 
  (h_abc : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_a₁b₁c₁ : 0 < a₁ ∧ 0 < b₁ ∧ 0 < c₁)
  (h_Δ : 0 < Δ)
  (h_Δ₁ : 0 < Δ₁)
  (h_abc_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_a₁b₁c₁_triangle : a₁ + b₁ > c₁ ∧ b₁ + c₁ > a₁ ∧ c₁ + a₁ > b₁)
  (h_Δ_def : Δ = Real.sqrt ((a + b + c) * (b + c - a) * (c + a - b) * (a + b - c)) / 4)
  (h_Δ₁_def : Δ₁ = Real.sqrt ((a₁ + b₁ + c₁) * (b₁ + c₁ - a₁) * (c₁ + a₁ - b₁) * (a₁ + b₁ - c₁)) / 4) :
  (a^2 * (b₁^2 + c₁^2 - a₁^2) + b^2 * (c₁^2 + a₁^2 - b₁^2) + c^2 * (a₁^2 + b₁^2 - c₁^2) = 16 * Δ * Δ₁) ↔
  (∃ (k : ℝ), k > 0 ∧ a = k * a₁ ∧ b = k * b₁ ∧ c = k * c₁) :=
by sorry

end pedoes_inequality_pedoes_inequality_equality_condition_l4021_402147


namespace complement_of_A_l4021_402199

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4, 5, 6}

-- Define set A
def A : Set Nat := {3, 4, 5}

-- Theorem statement
theorem complement_of_A : 
  (U \ A) = {1, 2, 6} := by sorry

end complement_of_A_l4021_402199


namespace find_A_l4021_402132

theorem find_A : ∃ A : ℕ, ∃ B : ℕ, 
  (100 ≤ 600 + 10 * A + B) ∧ 
  (600 + 10 * A + B < 1000) ∧
  (600 + 10 * A + B - 41 = 591) ∧
  A = 3 := by
  sorry

end find_A_l4021_402132


namespace ellipse_triangle_perimeter_l4021_402157

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define the foci of the ellipse
def focus1 : ℝ × ℝ := sorry
def focus2 : ℝ × ℝ := sorry

-- Define points A and B on the ellipse
def A : ℝ × ℝ := sorry
def B : ℝ × ℝ := sorry

-- Define the line passing through F1, A, and B
def line_passes_through (p q r : ℝ × ℝ) : Prop := sorry

-- Theorem statement
theorem ellipse_triangle_perimeter :
  ellipse A.1 A.2 →
  ellipse B.1 B.2 →
  line_passes_through focus1 A B →
  (A.1 - focus1.1)^2 + (A.2 - focus1.2)^2 + 
  (A.1 - focus2.1)^2 + (A.2 - focus2.2)^2 = 16 →
  (B.1 - focus1.1)^2 + (B.2 - focus1.2)^2 + 
  (B.1 - focus2.1)^2 + (B.2 - focus2.2)^2 = 16 →
  let perimeter := 
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) +
    Real.sqrt ((A.1 - focus2.1)^2 + (A.2 - focus2.2)^2) +
    Real.sqrt ((B.1 - focus2.1)^2 + (B.2 - focus2.2)^2)
  perimeter = 8 := by sorry


end ellipse_triangle_perimeter_l4021_402157


namespace rectangular_box_volume_l4021_402159

theorem rectangular_box_volume : ∃ (x : ℕ), 
  x > 0 ∧ 20 * x^3 = 160 := by
  sorry

end rectangular_box_volume_l4021_402159


namespace present_age_of_b_l4021_402172

theorem present_age_of_b (a b : ℕ) : 
  (a + 10 = 2 * (b - 10)) →  -- In 10 years, A will be twice as old as B was 10 years ago
  (a = b + 8) →              -- A is currently 8 years older than B
  b = 38                     -- B's present age is 38
  := by sorry

end present_age_of_b_l4021_402172


namespace tenth_student_age_l4021_402118

theorem tenth_student_age (total_students : ℕ) (students_without_tenth : ℕ) 
  (avg_age_without_tenth : ℕ) (avg_age_increase : ℕ) :
  total_students = 10 →
  students_without_tenth = 9 →
  avg_age_without_tenth = 8 →
  avg_age_increase = 2 →
  (students_without_tenth * avg_age_without_tenth + 
    (total_students * (avg_age_without_tenth + avg_age_increase) - 
     students_without_tenth * avg_age_without_tenth)) = 28 := by
  sorry

end tenth_student_age_l4021_402118


namespace quadratic_root_implies_k_l4021_402162

theorem quadratic_root_implies_k (k : ℝ) : 
  (3 * ((-15 - Real.sqrt 229) / 4)^2 + 15 * ((-15 - Real.sqrt 229) / 4) + k = 0) → 
  k = -1/3 := by
sorry

end quadratic_root_implies_k_l4021_402162


namespace total_players_count_l4021_402133

/-- Represents the number of people playing kabaddi -/
def kabaddi_players : ℕ := 10

/-- Represents the number of people playing kho kho only -/
def kho_kho_only_players : ℕ := 35

/-- Represents the number of people playing both games -/
def both_games_players : ℕ := 5

/-- Calculates the total number of players -/
def total_players : ℕ := kabaddi_players - both_games_players + kho_kho_only_players + both_games_players

theorem total_players_count : total_players = 45 := by
  sorry

end total_players_count_l4021_402133


namespace max_elevation_l4021_402177

/-- The elevation function of a particle thrown vertically upwards -/
def s (t : ℝ) : ℝ := 240 * t - 24 * t^2

/-- The maximum elevation reached by the particle -/
theorem max_elevation : ∃ (t : ℝ), ∀ (t' : ℝ), s t ≥ s t' ∧ s t = 600 := by
  sorry

end max_elevation_l4021_402177


namespace sally_walked_2540_miles_l4021_402101

/-- Calculates the total miles walked given pedometer resets, final reading, steps per mile, and additional steps --/
def total_miles_walked (resets : ℕ) (final_reading : ℕ) (steps_per_mile : ℕ) (additional_steps : ℕ) : ℕ :=
  let total_steps := resets * 100000 + final_reading + additional_steps
  (total_steps + steps_per_mile - 1) / steps_per_mile

/-- Theorem stating that Sally walked 2540 miles during the year --/
theorem sally_walked_2540_miles :
  total_miles_walked 50 30000 2000 50000 = 2540 := by
  sorry

end sally_walked_2540_miles_l4021_402101


namespace monogram_combinations_l4021_402174

theorem monogram_combinations : ∀ n k : ℕ, 
  n = 14 ∧ k = 2 → (n.choose k) = 91 :=
by
  sorry

#check monogram_combinations

end monogram_combinations_l4021_402174


namespace circle_ratio_l4021_402148

theorem circle_ratio (r₁ r₂ : ℝ) (h : r₁ > 0 ∧ r₂ > 0) 
  (h_area : π * r₂^2 - π * r₁^2 = 4 * π * r₁^2) : 
  r₁ / r₂ = 1 / Real.sqrt 5 := by
sorry

end circle_ratio_l4021_402148


namespace count_solutions_x_plus_y_plus_z_15_l4021_402173

/-- The number of solutions to x + y + z = 15 where x, y, and z are positive integers -/
def num_solutions : ℕ := 91

/-- Theorem stating that the number of solutions to x + y + z = 15 where x, y, and z are positive integers is 91 -/
theorem count_solutions_x_plus_y_plus_z_15 :
  (Finset.filter (fun t : ℕ × ℕ × ℕ => t.1 + t.2.1 + t.2.2 = 15 ∧ t.1 > 0 ∧ t.2.1 > 0 ∧ t.2.2 > 0)
    (Finset.product (Finset.range 16) (Finset.product (Finset.range 16) (Finset.range 16)))).card = num_solutions := by
  sorry

end count_solutions_x_plus_y_plus_z_15_l4021_402173


namespace smallest_gcd_multiple_l4021_402110

theorem smallest_gcd_multiple (m n : ℕ) (h : m > 0 ∧ n > 0) (h_gcd : Nat.gcd m n = 18) :
  Nat.gcd (8 * m) (12 * n) ≥ 72 ∧ ∃ (m₀ n₀ : ℕ), m₀ > 0 ∧ n₀ > 0 ∧ Nat.gcd m₀ n₀ = 18 ∧ Nat.gcd (8 * m₀) (12 * n₀) = 72 :=
by sorry

end smallest_gcd_multiple_l4021_402110


namespace pet_center_final_count_l4021_402145

/-- 
Given:
- initial_dogs: The initial number of dogs in the pet center
- initial_cats: The initial number of cats in the pet center
- adopted_dogs: The number of dogs adopted
- new_cats: The number of new cats collected

Prove that the final number of pets in the pet center is 57.
-/
theorem pet_center_final_count 
  (initial_dogs : ℕ) 
  (initial_cats : ℕ) 
  (adopted_dogs : ℕ) 
  (new_cats : ℕ) 
  (h1 : initial_dogs = 36)
  (h2 : initial_cats = 29)
  (h3 : adopted_dogs = 20)
  (h4 : new_cats = 12) :
  initial_dogs - adopted_dogs + initial_cats + new_cats = 57 :=
by
  sorry


end pet_center_final_count_l4021_402145


namespace solve_linear_systems_l4021_402196

theorem solve_linear_systems :
  (∃ (x1 y1 : ℝ), x1 + y1 = 3 ∧ 2*x1 + 3*y1 = 8 ∧ x1 = 1 ∧ y1 = 2) ∧
  (∃ (x2 y2 : ℝ), 5*x2 - 2*y2 = 4 ∧ 2*x2 - 3*y2 = -5 ∧ x2 = 2 ∧ y2 = 3) :=
by sorry

end solve_linear_systems_l4021_402196


namespace product_of_three_numbers_l4021_402114

theorem product_of_three_numbers (a b c : ℚ) 
  (sum_eq : a + b + c = 30)
  (first_eq : a = 2 * (b + c))
  (second_eq : b = 6 * c) : 
  a * b * c = 12000 / 49 := by
sorry

end product_of_three_numbers_l4021_402114


namespace cylinder_cross_section_angle_l4021_402161

/-- Given a cylinder cut by a plane where the cross-section has an eccentricity of 2√2/3,
    the acute dihedral angle between this cross-section and the cylinder's base is arccos(1/3). -/
theorem cylinder_cross_section_angle (e : ℝ) (θ : ℝ) : 
  e = 2 * Real.sqrt 2 / 3 →
  θ = Real.arccos (1/3) →
  θ = Real.arccos (Real.sqrt (1 - e^2)) :=
by sorry

end cylinder_cross_section_angle_l4021_402161


namespace previous_year_profit_percentage_l4021_402103

/-- Given a company's financial data over two years, calculate the profit percentage in the previous year. -/
theorem previous_year_profit_percentage
  (R : ℝ)  -- Revenues in the previous year
  (P : ℝ)  -- Profits in the previous year
  (h1 : 0.8 * R = R - 0.2 * R)  -- Revenues fell by 20% in 2009
  (h2 : 0.09 * (0.8 * R) = 0.072 * R)  -- Profits were 9% of revenues in 2009
  (h3 : 0.072 * R = 0.72 * P)  -- Profits in 2009 were 72% of previous year's profits
  : P / R = 0.1 := by
  sorry

end previous_year_profit_percentage_l4021_402103


namespace julie_landscaping_rate_l4021_402113

/-- Julie's landscaping business problem -/
theorem julie_landscaping_rate :
  ∀ (mowing_rate : ℝ),
  let weeding_rate : ℝ := 8
  let mowing_hours : ℝ := 25
  let weeding_hours : ℝ := 3
  let total_earnings : ℝ := 248
  (2 * (mowing_rate * mowing_hours + weeding_rate * weeding_hours) = total_earnings) →
  mowing_rate = 4 := by
sorry

end julie_landscaping_rate_l4021_402113


namespace original_proposition_true_converse_false_l4021_402136

theorem original_proposition_true_converse_false :
  (∀ a b : ℝ, a + b ≥ 2 → (a ≥ 1 ∨ b ≥ 1)) ∧
  (∃ a b : ℝ, (a ≥ 1 ∨ b ≥ 1) ∧ a + b < 2) :=
by sorry

end original_proposition_true_converse_false_l4021_402136


namespace metallic_sheet_width_l4021_402146

/-- Represents the dimensions and properties of a metallic sheet and the box formed from it. -/
structure MetallicSheet where
  length : ℝ
  width : ℝ
  cutSquareSize : ℝ
  boxVolume : ℝ

/-- Calculates the volume of the box formed from the metallic sheet. -/
def boxVolumeCalc (sheet : MetallicSheet) : ℝ :=
  (sheet.length - 2 * sheet.cutSquareSize) * 
  (sheet.width - 2 * sheet.cutSquareSize) * 
  sheet.cutSquareSize

/-- Theorem stating the width of the metallic sheet given the conditions. -/
theorem metallic_sheet_width 
  (sheet : MetallicSheet)
  (h1 : sheet.length = 48)
  (h2 : sheet.cutSquareSize = 8)
  (h3 : sheet.boxVolume = 5120)
  (h4 : boxVolumeCalc sheet = sheet.boxVolume) :
  sheet.width = 36 :=
sorry

end metallic_sheet_width_l4021_402146


namespace circle_circumference_irrational_l4021_402197

theorem circle_circumference_irrational (d : ℚ) :
  Irrational (Real.pi * (d : ℝ)) := by
  sorry

end circle_circumference_irrational_l4021_402197
