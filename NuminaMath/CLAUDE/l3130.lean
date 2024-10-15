import Mathlib

namespace NUMINAMATH_CALUDE_box_length_proof_l3130_313067

/-- Proves that a rectangular box with given dimensions has a length of 55.5 meters -/
theorem box_length_proof (width : ℝ) (road_width : ℝ) (lawn_area : ℝ) :
  width = 40 →
  road_width = 3 →
  lawn_area = 2109 →
  ∃ (length : ℝ),
    length * width - 2 * (length / 3) * road_width = lawn_area ∧
    length = 55.5 := by
  sorry

end NUMINAMATH_CALUDE_box_length_proof_l3130_313067


namespace NUMINAMATH_CALUDE_sand_box_fill_time_l3130_313005

/-- The time required to fill a rectangular box with sand -/
theorem sand_box_fill_time
  (length width height : ℝ)
  (fill_rate : ℝ)
  (h_length : length = 7)
  (h_width : width = 6)
  (h_height : height = 2)
  (h_fill_rate : fill_rate = 4)
  : (length * width * height) / fill_rate = 21 := by
  sorry

end NUMINAMATH_CALUDE_sand_box_fill_time_l3130_313005


namespace NUMINAMATH_CALUDE_valid_arrangements_count_l3130_313069

/-- Represents a seating arrangement in the minibus -/
def SeatingArrangement := Fin 6 → Fin 6

/-- Checks if a seating arrangement is valid (no sibling sits directly in front) -/
def is_valid_arrangement (arr : SeatingArrangement) : Prop :=
  ∀ i j : Fin 3, arr i ≠ arr (i + 3)

/-- The total number of valid seating arrangements -/
def total_valid_arrangements : ℕ := sorry

/-- Theorem stating that the number of valid seating arrangements is 12 -/
theorem valid_arrangements_count : total_valid_arrangements = 12 := by sorry

end NUMINAMATH_CALUDE_valid_arrangements_count_l3130_313069


namespace NUMINAMATH_CALUDE_smallest_number_l3130_313059

theorem smallest_number (a b c d : ℝ) (ha : a = 1/2) (hb : b = Real.sqrt 3) (hc : c = 0) (hd : d = -2) :
  d ≤ a ∧ d ≤ b ∧ d ≤ c :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_l3130_313059


namespace NUMINAMATH_CALUDE_car_selling_price_l3130_313052

/-- Calculates the selling price of a car given its purchase price, repair costs, and profit percentage. -/
def selling_price (purchase_price repair_costs profit_percent : ℚ) : ℚ :=
  let total_cost := purchase_price + repair_costs
  let profit := (profit_percent / 100) * total_cost
  total_cost + profit

/-- Theorem stating that for the given conditions, the selling price is 64900. -/
theorem car_selling_price :
  selling_price 42000 8000 29.8 = 64900 := by
  sorry

end NUMINAMATH_CALUDE_car_selling_price_l3130_313052


namespace NUMINAMATH_CALUDE_percentage_calculation_l3130_313031

theorem percentage_calculation : (168 / 100 * 1265) / 6 = 354.2 := by
  sorry

end NUMINAMATH_CALUDE_percentage_calculation_l3130_313031


namespace NUMINAMATH_CALUDE_janice_spent_1940_l3130_313075

/-- Calculates the total amount spent by Janice given the prices and quantities of items purchased --/
def total_spent (juice_price : ℚ) (sandwich_price : ℚ) (pastry_price : ℚ) (salad_price : ℚ) : ℚ :=
  let discounted_salad_price := salad_price * (1 - 0.2)
  sandwich_price + juice_price + 2 * pastry_price + discounted_salad_price

/-- Theorem stating that Janice spent $19.40 given the conditions in the problem --/
theorem janice_spent_1940 :
  let juice_price : ℚ := 10 / 5
  let sandwich_price : ℚ := 6 / 2
  let pastry_price : ℚ := 4
  let salad_price : ℚ := 8
  total_spent juice_price sandwich_price pastry_price salad_price = 1940 / 100 := by
  sorry

#eval total_spent (10/5) (6/2) 4 8

end NUMINAMATH_CALUDE_janice_spent_1940_l3130_313075


namespace NUMINAMATH_CALUDE_equation_one_solutions_equation_two_solution_l3130_313019

-- Equation 1
theorem equation_one_solutions (x : ℝ) :
  4 * (x - 1)^2 = 25 ↔ x = 7/2 ∨ x = -3/2 :=
sorry

-- Equation 2
theorem equation_two_solution (x : ℝ) :
  1/3 * (x + 2)^3 - 9 = 0 ↔ x = 1 :=
sorry

end NUMINAMATH_CALUDE_equation_one_solutions_equation_two_solution_l3130_313019


namespace NUMINAMATH_CALUDE_cube_root_equation_solution_l3130_313094

theorem cube_root_equation_solution (Q P : ℝ) 
  (h1 : (13 * Q + 6 * P + 1) ^ (1/3) - (13 * Q - 6 * P - 1) ^ (1/3) = 2 ^ (1/3))
  (h2 : Q > 0) : 
  Q = 7 := by
sorry

end NUMINAMATH_CALUDE_cube_root_equation_solution_l3130_313094


namespace NUMINAMATH_CALUDE_quadratic_root_proof_l3130_313030

theorem quadratic_root_proof :
  let x : ℝ := (-5 + Real.sqrt (5^2 + 4*3*1)) / (2*3)
  3 * x^2 + 5 * x - 1 = 0 ∨
  let x : ℝ := (-5 - Real.sqrt (5^2 + 4*3*1)) / (2*3)
  3 * x^2 + 5 * x - 1 = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_proof_l3130_313030


namespace NUMINAMATH_CALUDE_sin_240_degrees_l3130_313032

theorem sin_240_degrees : Real.sin (240 * Real.pi / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_240_degrees_l3130_313032


namespace NUMINAMATH_CALUDE_geometry_class_eligibility_l3130_313027

def minimum_score (s1 s2 s3 s4 : ℝ) : ℝ :=
  let required_average := 85
  let total_required := 5 * required_average
  let current_sum := s1 + s2 + s3 + s4
  total_required - current_sum

theorem geometry_class_eligibility 
  (s1 s2 s3 s4 : ℝ) 
  (h1 : s1 = 86) 
  (h2 : s2 = 82) 
  (h3 : s3 = 80) 
  (h4 : s4 = 84) : 
  minimum_score s1 s2 s3 s4 = 93 := by
  sorry

#eval minimum_score 86 82 80 84

end NUMINAMATH_CALUDE_geometry_class_eligibility_l3130_313027


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l3130_313036

theorem sqrt_equation_solution (x : ℝ) :
  Real.sqrt (3 + Real.sqrt x) = 4 → x = 169 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l3130_313036


namespace NUMINAMATH_CALUDE_elements_not_in_either_set_l3130_313089

/-- Given sets A and B that are subsets of a finite universal set U, 
    this theorem calculates the number of elements in U that are not in either A or B. -/
theorem elements_not_in_either_set 
  (U A B : Finset ℕ) 
  (h_subset_A : A ⊆ U) 
  (h_subset_B : B ⊆ U) 
  (h_card_U : U.card = 193)
  (h_card_A : A.card = 116)
  (h_card_B : B.card = 41)
  (h_card_inter : (A ∩ B).card = 23) :
  (U \ (A ∪ B)).card = 59 := by
  sorry

#check elements_not_in_either_set

end NUMINAMATH_CALUDE_elements_not_in_either_set_l3130_313089


namespace NUMINAMATH_CALUDE_front_parking_spaces_l3130_313022

theorem front_parking_spaces (back_spaces : ℕ) (total_parked : ℕ) (available_spaces : ℕ)
  (h1 : back_spaces = 38)
  (h2 : total_parked = 39)
  (h3 : available_spaces = 32)
  (h4 : back_spaces / 2 + available_spaces + total_parked = back_spaces + front_spaces) :
  front_spaces = 33 := by
  sorry

end NUMINAMATH_CALUDE_front_parking_spaces_l3130_313022


namespace NUMINAMATH_CALUDE_cube_condition_l3130_313049

theorem cube_condition (n : ℤ) : 
  (∃ k : ℤ, 6 * n + 2 = k ^ 3) ↔ 
  (∃ m : ℤ, n = 36 * m ^ 3 + 36 * m ^ 2 + 12 * m + 1) := by
sorry

end NUMINAMATH_CALUDE_cube_condition_l3130_313049


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3130_313080

theorem arithmetic_sequence_sum : 
  ∀ (a d n : ℕ) (last : ℕ),
    a = 3 → d = 2 → last = 25 →
    last = a + (n - 1) * d →
    (n : ℝ) / 2 * (a + last) = 168 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3130_313080


namespace NUMINAMATH_CALUDE_symmetric_line_equation_l3130_313041

/-- The parabola that touches the x-axis at one point -/
def parabola (x : ℝ) : ℝ := 2 * x^2 + 4 * x + 2

/-- The x-coordinate of the axis of symmetry -/
def axis_of_symmetry : ℝ := -1

/-- The x-intercept of the parabola -/
def x_intercept : ℝ := -1

/-- The y-intercept of the parabola -/
def y_intercept : ℝ := 2

/-- The slope of the line symmetric to the line joining x-intercept and y-intercept -/
def symmetric_line_slope : ℝ := -2

/-- The y-intercept of the line symmetric to the line joining x-intercept and y-intercept -/
def symmetric_line_y_intercept : ℝ := -2

/-- Theorem stating that the symmetric line has the equation y = -2x - 2 -/
theorem symmetric_line_equation :
  ∀ x y : ℝ, y = symmetric_line_slope * x + symmetric_line_y_intercept ↔ 
  y = -2 * x - 2 :=
sorry

end NUMINAMATH_CALUDE_symmetric_line_equation_l3130_313041


namespace NUMINAMATH_CALUDE_julia_basketball_success_rate_increase_l3130_313063

theorem julia_basketball_success_rate_increase :
  let initial_success : ℕ := 3
  let initial_attempts : ℕ := 8
  let subsequent_success : ℕ := 12
  let subsequent_attempts : ℕ := 16
  let total_success := initial_success + subsequent_success
  let total_attempts := initial_attempts + subsequent_attempts
  let initial_rate := initial_success / initial_attempts
  let final_rate := total_success / total_attempts
  final_rate - initial_rate = 1/4 := by sorry

end NUMINAMATH_CALUDE_julia_basketball_success_rate_increase_l3130_313063


namespace NUMINAMATH_CALUDE_root_existence_condition_l3130_313081

theorem root_existence_condition (a : ℝ) :
  (∃ x ∈ Set.Icc (-1 : ℝ) 2, a * x + 3 = 0) ↔ (a ≤ -3/2 ∨ a ≥ 3) := by
  sorry

end NUMINAMATH_CALUDE_root_existence_condition_l3130_313081


namespace NUMINAMATH_CALUDE_girls_left_auditorium_l3130_313068

theorem girls_left_auditorium (initial_boys : ℕ) (initial_girls : ℕ) (remaining_students : ℕ) : 
  initial_boys = 24 →
  initial_girls = 14 →
  remaining_students = 30 →
  ∃ (left_girls : ℕ), left_girls = 4 ∧ 
    ∃ (left_boys : ℕ), left_boys = left_girls ∧
    initial_boys + initial_girls - (left_boys + left_girls) = remaining_students :=
by sorry

end NUMINAMATH_CALUDE_girls_left_auditorium_l3130_313068


namespace NUMINAMATH_CALUDE_negative_five_greater_than_negative_seven_l3130_313072

theorem negative_five_greater_than_negative_seven : -5 > -7 := by
  sorry

end NUMINAMATH_CALUDE_negative_five_greater_than_negative_seven_l3130_313072


namespace NUMINAMATH_CALUDE_triangle_ratio_l3130_313073

theorem triangle_ratio (a b c : ℝ) (A B C : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c →
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π →
  A + B + C = π →
  A = π / 3 →
  b = 1 →
  (1/2) * b * c * Real.sin A = Real.sqrt 3 →
  (a + 2*b - 3*c) / (Real.sin A + 2*Real.sin B - 3*Real.sin C) = 2 * Real.sqrt 39 / 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_ratio_l3130_313073


namespace NUMINAMATH_CALUDE_quadratic_shift_sum_l3130_313058

/-- Given a quadratic function f(x) = 3x^2 + 5x + 9, when shifted 6 units to the left,
    results in a new quadratic function g(x) = ax^2 + bx + c.
    This theorem proves that a + b + c = 191. -/
theorem quadratic_shift_sum (f g : ℝ → ℝ) (a b c : ℝ) :
  (∀ x, f x = 3*x^2 + 5*x + 9) →
  (∀ x, g x = f (x + 6)) →
  (∀ x, g x = a*x^2 + b*x + c) →
  a + b + c = 191 := by
sorry

end NUMINAMATH_CALUDE_quadratic_shift_sum_l3130_313058


namespace NUMINAMATH_CALUDE_product_in_N_not_in_M_l3130_313050

def M : Set ℤ := {x | ∃ m : ℤ, x = 3*m + 1}
def N : Set ℤ := {y | ∃ n : ℤ, y = 3*n + 2}

theorem product_in_N_not_in_M (x y : ℤ) (hx : x ∈ M) (hy : y ∈ N) :
  (x * y) ∈ N ∧ (x * y) ∉ M := by
  sorry

end NUMINAMATH_CALUDE_product_in_N_not_in_M_l3130_313050


namespace NUMINAMATH_CALUDE_three_digit_numbers_from_five_l3130_313043

/-- The number of ways to create a three-digit number using five different single-digit numbers -/
def three_digit_combinations (n : ℕ) (r : ℕ) : ℕ :=
  (n.factorial) / ((r.factorial) * ((n - r).factorial))

/-- The number of permutations of r items -/
def permutations (r : ℕ) : ℕ := r.factorial

theorem three_digit_numbers_from_five : 
  three_digit_combinations 5 3 * permutations 3 = 60 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_numbers_from_five_l3130_313043


namespace NUMINAMATH_CALUDE_total_baseball_fans_l3130_313061

theorem total_baseball_fans (yankees mets redsox : ℕ) : 
  yankees * 2 = mets * 3 →
  mets * 5 = redsox * 4 →
  mets = 88 →
  yankees + mets + redsox = 330 :=
by sorry

end NUMINAMATH_CALUDE_total_baseball_fans_l3130_313061


namespace NUMINAMATH_CALUDE_kingsley_pants_per_day_l3130_313056

/-- Represents the number of shirts Jenson makes per day -/
def jenson_shirts_per_day : ℕ := 3

/-- Represents the amount of fabric used for one shirt in yards -/
def fabric_per_shirt : ℕ := 2

/-- Represents the amount of fabric used for one pair of pants in yards -/
def fabric_per_pants : ℕ := 5

/-- Represents the total amount of fabric needed every 3 days in yards -/
def total_fabric_3days : ℕ := 93

/-- Theorem stating that Kingsley makes 5 pairs of pants per day given the conditions -/
theorem kingsley_pants_per_day :
  ∃ (p : ℕ), 
    p * fabric_per_pants + jenson_shirts_per_day * fabric_per_shirt = total_fabric_3days / 3 ∧
    p = 5 := by
  sorry

end NUMINAMATH_CALUDE_kingsley_pants_per_day_l3130_313056


namespace NUMINAMATH_CALUDE_original_number_proof_l3130_313035

theorem original_number_proof : ∃ x : ℝ, x / 12.75 = 16 ∧ x = 204 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l3130_313035


namespace NUMINAMATH_CALUDE_binomial_expansion_sum_l3130_313026

theorem binomial_expansion_sum (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (3 - 2*x)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₀ + a₁ + 2*a₂ + 3*a₃ + 4*a₄ + 5*a₅ = 233 := by
sorry

end NUMINAMATH_CALUDE_binomial_expansion_sum_l3130_313026


namespace NUMINAMATH_CALUDE_island_population_even_l3130_313029

/-- Represents the type of inhabitants on the island -/
inductive Inhabitant
| Knight
| Liar

/-- Represents a claim about the number of inhabitants -/
inductive Claim
| EvenKnights
| OddLiars

/-- Function that determines if a given inhabitant tells the truth about a claim -/
def tellsTruth (i : Inhabitant) (c : Claim) : Prop :=
  match i, c with
  | Inhabitant.Knight, _ => true
  | Inhabitant.Liar, _ => false

/-- The island population -/
structure Island where
  inhabitants : List Inhabitant
  claims : List (Inhabitant × Claim)
  all_claimed : ∀ i ∈ inhabitants, ∃ c, (i, c) ∈ claims

theorem island_population_even (isle : Island) : Even (List.length isle.inhabitants) := by
  sorry


end NUMINAMATH_CALUDE_island_population_even_l3130_313029


namespace NUMINAMATH_CALUDE_line_intersects_x_axis_l3130_313064

/-- The line equation 5y - 3x = 15 intersects the x-axis at the point (-5, 0). -/
theorem line_intersects_x_axis :
  ∃ (x y : ℝ), 5 * y - 3 * x = 15 ∧ y = 0 ∧ x = -5 := by
  sorry

end NUMINAMATH_CALUDE_line_intersects_x_axis_l3130_313064


namespace NUMINAMATH_CALUDE_symmetry_implies_sum_l3130_313013

/-- Two points are symmetric about the y-axis if their x-coordinates are opposite and y-coordinates are equal -/
def symmetric_about_y_axis (x1 y1 x2 y2 : ℝ) : Prop :=
  x1 = -x2 ∧ y1 = y2

theorem symmetry_implies_sum (a b : ℝ) :
  symmetric_about_y_axis (a + 1) 3 (-2) (b + 2) →
  a + b = 2 := by
  sorry

end NUMINAMATH_CALUDE_symmetry_implies_sum_l3130_313013


namespace NUMINAMATH_CALUDE_sum_of_reciprocal_relations_l3130_313090

theorem sum_of_reciprocal_relations (x y : ℝ) 
  (h1 : 1/x + 1/y = 4) 
  (h2 : 1/x - 1/y = -6) : 
  x + y = -4/5 := by
sorry

end NUMINAMATH_CALUDE_sum_of_reciprocal_relations_l3130_313090


namespace NUMINAMATH_CALUDE_gcd_digit_sum_theorem_l3130_313038

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

theorem gcd_digit_sum_theorem : 
  let a := 4665 - 1305
  let b := 6905 - 4665
  let c := 6905 - 1305
  let gcd_result := Nat.gcd (Nat.gcd a b) c
  sum_of_digits gcd_result = 4 := by sorry

end NUMINAMATH_CALUDE_gcd_digit_sum_theorem_l3130_313038


namespace NUMINAMATH_CALUDE_largest_three_digit_and_smallest_four_digit_l3130_313066

theorem largest_three_digit_and_smallest_four_digit : 
  (∃ n : ℕ, n = 999 ∧ ∀ m : ℕ, m < 1000 → m ≤ n) ∧
  (∃ k : ℕ, k = 1000 ∧ ∀ l : ℕ, l ≥ 1000 → l ≥ k) ∧
  (1000 - 999 = 1) :=
by sorry

end NUMINAMATH_CALUDE_largest_three_digit_and_smallest_four_digit_l3130_313066


namespace NUMINAMATH_CALUDE_second_team_soup_amount_l3130_313078

/-- Given the total required amount of soup and the amounts made by the first and third teams,
    calculate the amount the second team should prepare. -/
theorem second_team_soup_amount (total_required : ℕ) (first_team : ℕ) (third_team : ℕ) :
  total_required = 280 →
  first_team = 90 →
  third_team = 70 →
  total_required - (first_team + third_team) = 120 := by
sorry

end NUMINAMATH_CALUDE_second_team_soup_amount_l3130_313078


namespace NUMINAMATH_CALUDE_fourth_selected_is_48_l3130_313076

/-- Represents a systematic sampling scenario -/
structure SystematicSampling where
  total : ℕ
  sample_size : ℕ
  first_three : Fin 3 → ℕ

/-- Calculates the interval for systematic sampling -/
def sampling_interval (s : SystematicSampling) : ℕ :=
  s.total / s.sample_size

/-- Theorem: In the given systematic sampling scenario, the fourth selected number is 48 -/
theorem fourth_selected_is_48 (s : SystematicSampling) 
  (h_total : s.total = 60)
  (h_sample_size : s.sample_size = 4)
  (h_first_three : s.first_three = ![3, 18, 33]) :
  s.first_three 2 + sampling_interval s = 48 := by
  sorry

end NUMINAMATH_CALUDE_fourth_selected_is_48_l3130_313076


namespace NUMINAMATH_CALUDE_brendas_blisters_l3130_313082

theorem brendas_blisters (blisters_per_arm : ℕ) : 
  (2 * blisters_per_arm + 80 = 200) → blisters_per_arm = 60 := by
  sorry

end NUMINAMATH_CALUDE_brendas_blisters_l3130_313082


namespace NUMINAMATH_CALUDE_permutations_of_four_distinct_elements_l3130_313033

theorem permutations_of_four_distinct_elements : 
  Nat.factorial 4 = 24 := by
  sorry

end NUMINAMATH_CALUDE_permutations_of_four_distinct_elements_l3130_313033


namespace NUMINAMATH_CALUDE_max_boxes_fit_l3130_313093

def large_box_length : ℕ := 8
def large_box_width : ℕ := 7
def large_box_height : ℕ := 6

def small_box_length : ℕ := 4
def small_box_width : ℕ := 7
def small_box_height : ℕ := 6

def cm_per_meter : ℕ := 100

theorem max_boxes_fit (large_box_volume small_box_volume max_boxes : ℕ) : 
  large_box_volume = (large_box_length * cm_per_meter) * (large_box_width * cm_per_meter) * (large_box_height * cm_per_meter) →
  small_box_volume = small_box_length * small_box_width * small_box_height →
  max_boxes = large_box_volume / small_box_volume →
  max_boxes = 2000000 := by
  sorry

#check max_boxes_fit

end NUMINAMATH_CALUDE_max_boxes_fit_l3130_313093


namespace NUMINAMATH_CALUDE_multiply_586645_by_9999_l3130_313018

theorem multiply_586645_by_9999 : 586645 * 9999 = 5865864355 := by
  sorry

end NUMINAMATH_CALUDE_multiply_586645_by_9999_l3130_313018


namespace NUMINAMATH_CALUDE_train_crossing_time_l3130_313000

/-- Proves that a train 175 meters long, traveling at 180 km/hr, will take 3.5 seconds to cross an electric pole. -/
theorem train_crossing_time (train_length : Real) (train_speed_kmh : Real) 
    (h1 : train_length = 175)
    (h2 : train_speed_kmh = 180) : 
  let train_speed_ms : Real := train_speed_kmh * (1000 / 3600)
  let crossing_time : Real := train_length / train_speed_ms
  crossing_time = 3.5 := by
    sorry

#check train_crossing_time

end NUMINAMATH_CALUDE_train_crossing_time_l3130_313000


namespace NUMINAMATH_CALUDE_division_remainder_problem_l3130_313020

theorem division_remainder_problem :
  let dividend : ℕ := 12401
  let divisor : ℕ := 163
  let quotient : ℕ := 76
  dividend = quotient * divisor + 13 :=
by sorry

end NUMINAMATH_CALUDE_division_remainder_problem_l3130_313020


namespace NUMINAMATH_CALUDE_negation_of_proposition_l3130_313062

theorem negation_of_proposition (f : ℝ → ℝ) :
  (¬ (∀ x₁ x₂ : ℝ, (f x₂ - f x₁) * (x₂ - x₁) ≥ 0)) ↔
  (∃ x₁ x₂ : ℝ, (f x₂ - f x₁) * (x₂ - x₁) < 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l3130_313062


namespace NUMINAMATH_CALUDE_michelle_final_crayons_l3130_313007

/-- 
Given:
- Michelle initially has x crayons
- Janet initially has y crayons
- Both Michelle and Janet receive z more crayons each
- Janet gives all of her crayons to Michelle

Prove that Michelle will have x + y + 2z crayons in total.
-/
theorem michelle_final_crayons (x y z : ℕ) : x + z + (y + z) = x + y + 2*z :=
by sorry

end NUMINAMATH_CALUDE_michelle_final_crayons_l3130_313007


namespace NUMINAMATH_CALUDE_evaluate_expression_l3130_313074

theorem evaluate_expression : (3^4 + 3^4 + 3^4) / (3^(-4) + 3^(-4)) = 9841.5 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3130_313074


namespace NUMINAMATH_CALUDE_camp_cedar_counselors_l3130_313037

/-- The number of counselors needed at Camp Cedar --/
def counselors_needed (num_boys : ℕ) (num_girls : ℕ) : ℕ :=
  (num_boys / 6) + (num_girls / 10)

/-- Theorem: Camp Cedar needs 26 counselors --/
theorem camp_cedar_counselors :
  let num_boys : ℕ := 48
  let num_girls : ℕ := 4 * num_boys - 12
  counselors_needed num_boys num_girls = 26 := by
  sorry


end NUMINAMATH_CALUDE_camp_cedar_counselors_l3130_313037


namespace NUMINAMATH_CALUDE_number_of_divisors_of_2002_l3130_313086

theorem number_of_divisors_of_2002 : ∃ (d : ℕ → ℕ), d 2002 = 16 := by
  sorry

end NUMINAMATH_CALUDE_number_of_divisors_of_2002_l3130_313086


namespace NUMINAMATH_CALUDE_removed_number_theorem_l3130_313054

theorem removed_number_theorem (n : ℕ) (m : ℕ) :
  m ≤ n →
  (n * (n + 1) / 2 - m) / (n - 1) = 163/4 →
  m = 61 := by
sorry

end NUMINAMATH_CALUDE_removed_number_theorem_l3130_313054


namespace NUMINAMATH_CALUDE_inscribed_sphere_pyramid_volume_l3130_313051

/-- A regular quadrilateral pyramid with an inscribed sphere -/
structure InscribedSpherePyramid where
  /-- Side length of the base of the pyramid -/
  a : ℝ
  /-- The sphere touches the base and all lateral faces -/
  sphere_touches_all_faces : True
  /-- The sphere divides the height in a 4:5 ratio from the apex -/
  height_ratio : True

/-- Volume of the pyramid -/
noncomputable def pyramid_volume (p : InscribedSpherePyramid) : ℝ :=
  2 * p.a^3 / 5

/-- Theorem stating the volume of the pyramid -/
theorem inscribed_sphere_pyramid_volume (p : InscribedSpherePyramid) :
  pyramid_volume p = 2 * p.a^3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_sphere_pyramid_volume_l3130_313051


namespace NUMINAMATH_CALUDE_first_inequality_solution_system_of_inequalities_solution_integer_solutions_correct_l3130_313098

-- Define the set of integer solutions
def IntegerSolutions : Set ℤ := {0, 1, 2}

-- Theorem for the first inequality
theorem first_inequality_solution (x : ℝ) :
  3 * (2 * x + 2) > 4 * x - 1 + 7 ↔ x > -3/2 := by sorry

-- Theorem for the system of inequalities
theorem system_of_inequalities_solution (x : ℝ) :
  (x + 1 > 0 ∧ x ≤ (x - 2) / 3 + 2) ↔ (-1 < x ∧ x ≤ 2) := by sorry

-- Theorem for integer solutions
theorem integer_solutions_correct :
  ∀ (n : ℤ), n ∈ IntegerSolutions ↔ (n + 1 > 0 ∧ n ≤ (n - 2) / 3 + 2) := by sorry

end NUMINAMATH_CALUDE_first_inequality_solution_system_of_inequalities_solution_integer_solutions_correct_l3130_313098


namespace NUMINAMATH_CALUDE_circle_line_intersection_max_k_l3130_313097

theorem circle_line_intersection_max_k : 
  ∃ (k_max : ℝ),
    k_max = 4/3 ∧
    ∀ (k : ℝ),
      (∃ (x₀ y₀ : ℝ),
        y₀ = k * x₀ - 2 ∧
        ∃ (x y : ℝ),
          (x - 4)^2 + y^2 = 1 ∧
          (x - x₀)^2 + (y - y₀)^2 ≤ 1) →
      k ≤ k_max :=
by sorry

end NUMINAMATH_CALUDE_circle_line_intersection_max_k_l3130_313097


namespace NUMINAMATH_CALUDE_chocolates_distribution_l3130_313077

theorem chocolates_distribution (total_chocolates : ℕ) (total_children : ℕ) (boys : ℕ) (girls : ℕ) 
  (chocolates_per_girl : ℕ) (h1 : total_chocolates = 3000) (h2 : total_children = 120) 
  (h3 : boys = 60) (h4 : girls = 60) (h5 : chocolates_per_girl = 3) 
  (h6 : total_children = boys + girls) : 
  (total_chocolates - girls * chocolates_per_girl) / boys = 47 := by
  sorry

end NUMINAMATH_CALUDE_chocolates_distribution_l3130_313077


namespace NUMINAMATH_CALUDE_angle_division_l3130_313001

theorem angle_division (α : ℝ) (n : ℕ) (h1 : α = 19) (h2 : n = 19) :
  α / n = 1 := by
  sorry

end NUMINAMATH_CALUDE_angle_division_l3130_313001


namespace NUMINAMATH_CALUDE_thomas_weight_vest_cost_l3130_313012

/-- Calculates the total cost for Thomas to increase his weight vest weight --/
def calculate_total_cost (initial_weight : ℕ) (increase_percentage : ℕ) (ingot_weight : ℕ) (ingot_cost : ℕ) : ℚ :=
  let additional_weight := initial_weight * increase_percentage / 100
  let num_ingots := (additional_weight + ingot_weight - 1) / ingot_weight
  let base_cost := num_ingots * ingot_cost
  let discounted_cost := 
    if num_ingots ≤ 10 then base_cost
    else if num_ingots ≤ 20 then base_cost * 80 / 100
    else if num_ingots ≤ 30 then base_cost * 75 / 100
    else base_cost * 70 / 100
  let taxed_cost :=
    if num_ingots ≤ 20 then discounted_cost * 105 / 100
    else if num_ingots ≤ 30 then discounted_cost * 103 / 100
    else discounted_cost * 101 / 100
  let shipping_fee :=
    if num_ingots * ingot_weight ≤ 20 then 10
    else if num_ingots * ingot_weight ≤ 40 then 15
    else 20
  taxed_cost + shipping_fee

/-- Theorem stating that the total cost for Thomas is $90.60 --/
theorem thomas_weight_vest_cost :
  calculate_total_cost 60 60 2 5 = 9060 / 100 :=
sorry

end NUMINAMATH_CALUDE_thomas_weight_vest_cost_l3130_313012


namespace NUMINAMATH_CALUDE_subtract_product_equality_l3130_313014

theorem subtract_product_equality : 7899665 - 12 * 3 * 2 = 7899593 := by
  sorry

end NUMINAMATH_CALUDE_subtract_product_equality_l3130_313014


namespace NUMINAMATH_CALUDE_exponential_and_logarithm_inequalities_l3130_313087

-- Define the exponential function
noncomputable def exp (base : ℝ) (exponent : ℝ) : ℝ := Real.exp (exponent * Real.log base)

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

theorem exponential_and_logarithm_inequalities :
  (exp 0.8 (-0.1) < exp 0.8 (-0.2)) ∧ (log 7 6 > log 8 6) := by
  sorry

end NUMINAMATH_CALUDE_exponential_and_logarithm_inequalities_l3130_313087


namespace NUMINAMATH_CALUDE_triangle_side_length_l3130_313021

theorem triangle_side_length (a b x : ℝ) : 
  a = 2 → 
  b = 6 → 
  x^2 - 10*x + 21 = 0 → 
  x > 0 → 
  a + x > b → 
  b + x > a → 
  a + b > x → 
  x = 7 := by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3130_313021


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3130_313039

theorem complex_equation_solution (z : ℂ) : (z + 1) * Complex.I = 1 - Complex.I → z = -2 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3130_313039


namespace NUMINAMATH_CALUDE_irrational_approximation_l3130_313028

theorem irrational_approximation 
  (r₁ r₂ : ℝ) 
  (h_irrational : Irrational (r₁ / r₂)) :
  ∀ (x p : ℝ), p > 0 → ∃ (k₁ k₂ : ℤ), |x - (↑k₁ * r₁ + ↑k₂ * r₂)| < p := by
  sorry

end NUMINAMATH_CALUDE_irrational_approximation_l3130_313028


namespace NUMINAMATH_CALUDE_billy_decoration_rate_l3130_313009

/-- The number of eggs Mia can decorate per hour -/
def mia_rate : ℕ := 24

/-- The total number of eggs to be decorated -/
def total_eggs : ℕ := 170

/-- The time taken by Mia and Billy together to decorate all eggs (in hours) -/
def total_time : ℕ := 5

/-- Billy's decoration rate (in eggs per hour) -/
def billy_rate : ℕ := total_eggs / total_time - mia_rate

theorem billy_decoration_rate :
  billy_rate = 10 := by sorry

end NUMINAMATH_CALUDE_billy_decoration_rate_l3130_313009


namespace NUMINAMATH_CALUDE_village_population_equation_l3130_313071

/-- The initial population of a village in Sri Lanka -/
def initial_population : ℕ := 4500

/-- The fraction of people who survived the bombardment -/
def survival_rate : ℚ := 9/10

/-- The fraction of people who remained in the village after some left due to fear -/
def remaining_rate : ℚ := 4/5

/-- The final population of the village -/
def final_population : ℕ := 3240

/-- Theorem stating that the initial population satisfies the given conditions -/
theorem village_population_equation :
  ↑initial_population * (survival_rate * remaining_rate) = final_population := by
  sorry

end NUMINAMATH_CALUDE_village_population_equation_l3130_313071


namespace NUMINAMATH_CALUDE_cos_squared_plus_half_sin_double_l3130_313006

theorem cos_squared_plus_half_sin_double (θ : ℝ) :
  3 * Real.cos (π / 2 - θ) + Real.cos (π + θ) = 0 →
  Real.cos θ ^ 2 + (1 / 2) * Real.sin (2 * θ) = 6 / 5 := by
  sorry

end NUMINAMATH_CALUDE_cos_squared_plus_half_sin_double_l3130_313006


namespace NUMINAMATH_CALUDE_min_pizzas_to_break_even_l3130_313024

def car_cost : ℕ := 6000
def bag_cost : ℕ := 200
def earning_per_pizza : ℕ := 12
def gas_cost_per_delivery : ℕ := 4

theorem min_pizzas_to_break_even :
  let total_cost := car_cost + bag_cost
  let net_earning_per_pizza := earning_per_pizza - gas_cost_per_delivery
  (∀ n : ℕ, n * net_earning_per_pizza < total_cost → n < 775) ∧
  775 * net_earning_per_pizza ≥ total_cost :=
sorry

end NUMINAMATH_CALUDE_min_pizzas_to_break_even_l3130_313024


namespace NUMINAMATH_CALUDE_inequality_proof_l3130_313040

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a^2 * b^2 + b^2 * c^2 + a^2 * c^2 ≥ a * b * c * (a + b + c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3130_313040


namespace NUMINAMATH_CALUDE_complement_of_M_in_U_l3130_313083

-- Define the universal set U
def U : Set Nat := {1, 2, 3}

-- Define the set M
def M : Set Nat := {1}

-- State the theorem
theorem complement_of_M_in_U : 
  (U \ M) = {2, 3} := by sorry

end NUMINAMATH_CALUDE_complement_of_M_in_U_l3130_313083


namespace NUMINAMATH_CALUDE_trigonometric_problem_l3130_313002

open Real

theorem trigonometric_problem (α β : ℝ) 
  (h1 : sin α = (4 * Real.sqrt 3) / 7)
  (h2 : cos (β - α) = 13 / 14)
  (h3 : 0 < β) (h4 : β < α) (h5 : α < π / 2) : 
  tan (2 * α) = -(8 * Real.sqrt 3) / 47 ∧ cos β = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_problem_l3130_313002


namespace NUMINAMATH_CALUDE_parabola_coefficient_l3130_313070

/-- A parabola with equation x = ay² + by + c, vertex at (3, -1), and passing through (7, 3) has a = 1/4 -/
theorem parabola_coefficient (a b c : ℝ) : 
  (∀ y : ℝ, 3 = a * (-1)^2 + b * (-1) + c) →  -- vertex condition
  (∀ y : ℝ, 7 = a * 3^2 + b * 3 + c) →        -- point condition
  a = (1 : ℝ) / 4 := by sorry

end NUMINAMATH_CALUDE_parabola_coefficient_l3130_313070


namespace NUMINAMATH_CALUDE_latin_square_symmetric_diagonal_l3130_313016

/-- A Latin square of order 7 -/
def LatinSquare7 (A : Fin 7 → Fin 7 → Fin 7) : Prop :=
  ∀ i j : Fin 7, ∀ k : Fin 7, (∃! x : Fin 7, A i x = k) ∧ (∃! y : Fin 7, A y j = k)

/-- Symmetry with respect to the main diagonal -/
def SymmetricMatrix (A : Fin 7 → Fin 7 → Fin 7) : Prop :=
  ∀ i j : Fin 7, A i j = A j i

/-- All numbers from 1 to 7 appear on the main diagonal -/
def AllNumbersOnDiagonal (A : Fin 7 → Fin 7 → Fin 7) : Prop :=
  ∀ k : Fin 7, ∃ i : Fin 7, A i i = k

theorem latin_square_symmetric_diagonal 
  (A : Fin 7 → Fin 7 → Fin 7) 
  (h1 : LatinSquare7 A) 
  (h2 : SymmetricMatrix A) : 
  AllNumbersOnDiagonal A :=
sorry

end NUMINAMATH_CALUDE_latin_square_symmetric_diagonal_l3130_313016


namespace NUMINAMATH_CALUDE_perpendicular_implies_parallel_l3130_313088

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations between lines and planes
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)

-- Define the given conditions
variable (l m n : Line)
variable (α β γ : Plane)

variable (different_lines : l ≠ m ∧ m ≠ n ∧ l ≠ n)
variable (non_coincident_planes : α ≠ β ∧ β ≠ γ ∧ α ≠ γ)

-- State the theorem
theorem perpendicular_implies_parallel 
  (h1 : perpendicular m α) 
  (h2 : perpendicular m β) : 
  parallel α β :=
sorry

end NUMINAMATH_CALUDE_perpendicular_implies_parallel_l3130_313088


namespace NUMINAMATH_CALUDE_flea_treatment_result_l3130_313023

/-- The number of fleas on a dog after a series of treatments -/
def fleas_after_treatments (initial_fleas : ℕ) (num_treatments : ℕ) : ℕ :=
  initial_fleas / (2^num_treatments)

/-- Theorem: If a dog undergoes four flea treatments, where each treatment halves the number of fleas,
    and the initial number of fleas is 210 more than the final number, then the final number of fleas is 14. -/
theorem flea_treatment_result :
  ∀ F : ℕ,
  (F + 210 = fleas_after_treatments (F + 210) 4) →
  F = 14 :=
by sorry

end NUMINAMATH_CALUDE_flea_treatment_result_l3130_313023


namespace NUMINAMATH_CALUDE_increasing_f_range_l3130_313010

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (2 - a) * x + 1 else a^x

theorem increasing_f_range (a : ℝ) 
  (h1 : a > 0) (h2 : a ≠ 1) 
  (h3 : ∀ x y : ℝ, x < y → f a x < f a y) : 
  a ∈ Set.Icc (3/2) 2 ∧ a < 2 :=
sorry

end NUMINAMATH_CALUDE_increasing_f_range_l3130_313010


namespace NUMINAMATH_CALUDE_average_sleep_is_eight_l3130_313048

def monday_sleep : ℕ := 8
def tuesday_sleep : ℕ := 7
def wednesday_sleep : ℕ := 8
def thursday_sleep : ℕ := 10
def friday_sleep : ℕ := 7

def total_days : ℕ := 5

def total_sleep : ℕ := monday_sleep + tuesday_sleep + wednesday_sleep + thursday_sleep + friday_sleep

theorem average_sleep_is_eight :
  (total_sleep : ℚ) / total_days = 8 := by sorry

end NUMINAMATH_CALUDE_average_sleep_is_eight_l3130_313048


namespace NUMINAMATH_CALUDE_adam_picked_apples_for_30_days_l3130_313079

/-- The number of days Adam picked apples -/
def days_picked : ℕ := 30

/-- The number of apples Adam picked each day -/
def apples_per_day : ℕ := 4

/-- The number of remaining apples Adam collected after a month -/
def remaining_apples : ℕ := 230

/-- The total number of apples Adam collected -/
def total_apples : ℕ := 350

/-- Theorem stating that the number of days Adam picked apples is 30 -/
theorem adam_picked_apples_for_30_days :
  days_picked * apples_per_day + remaining_apples = total_apples :=
by sorry

end NUMINAMATH_CALUDE_adam_picked_apples_for_30_days_l3130_313079


namespace NUMINAMATH_CALUDE_toy_store_inventory_l3130_313045

/-- Calculates the final number of games in a toy store's inventory --/
theorem toy_store_inventory (initial : ℕ) (sold : ℕ) (received : ℕ) :
  initial = 95 →
  sold = 68 →
  received = 47 →
  initial - sold + received = 74 := by
  sorry

end NUMINAMATH_CALUDE_toy_store_inventory_l3130_313045


namespace NUMINAMATH_CALUDE_part_one_part_two_l3130_313099

-- Define propositions p and q
def p (x a : ℝ) : Prop := (x - a) * (x - 3 * a) < 0

def q (x : ℝ) : Prop := (x - 3) / (x - 2) ≤ 0

-- Part 1
theorem part_one (x : ℝ) (h1 : p x 1) (h2 : q x) : 2 < x ∧ x < 3 := by
  sorry

-- Part 2
theorem part_two (a : ℝ) (h : a > 0)
  (h_suff : ∀ x, ¬(p x a) → ¬(q x))
  (h_not_nec : ∃ x, q x ∧ p x a) : 
  1 < a ∧ a ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_part_one_part_two_l3130_313099


namespace NUMINAMATH_CALUDE_salary_unspent_fraction_l3130_313060

theorem salary_unspent_fraction (salary : ℝ) (salary_positive : salary > 0) :
  let first_week_spent := (1 / 4 : ℝ) * salary
  let each_other_week_spent := (1 / 5 : ℝ) * salary
  let total_spent := first_week_spent + 3 * each_other_week_spent
  (salary - total_spent) / salary = 3 / 20 := by
  sorry

end NUMINAMATH_CALUDE_salary_unspent_fraction_l3130_313060


namespace NUMINAMATH_CALUDE_rectangle_tiling_l3130_313042

theorem rectangle_tiling (a b m n : ℕ) (h : a > 0 ∧ b > 0 ∧ m > 0 ∧ n > 0) :
  (∃ (v h : ℕ → ℕ → ℕ), ∀ (i j : ℕ), i < m ∧ j < n →
    (v i j = b ∧ h i j = 0) ∨ (v i j = 0 ∧ h i j = a)) →
  b ∣ m ∨ a ∣ n := by
  sorry


end NUMINAMATH_CALUDE_rectangle_tiling_l3130_313042


namespace NUMINAMATH_CALUDE_man_speed_calculation_man_speed_specific_case_l3130_313091

/-- Calculates the speed of a man walking in the same direction as a train, given the train's length, speed, and time to cross the man. -/
theorem man_speed_calculation (train_length : Real) (train_speed_kmh : Real) (crossing_time : Real) : Real :=
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  let relative_speed := train_length / crossing_time
  train_speed_ms - relative_speed

/-- The speed of a man walking in the same direction as a train, given specific conditions. -/
theorem man_speed_specific_case : 
  ∃ (ε : Real), ε > 0 ∧ 
  |man_speed_calculation 100 63 5.999520038396929 - 0.831946| < ε :=
sorry

end NUMINAMATH_CALUDE_man_speed_calculation_man_speed_specific_case_l3130_313091


namespace NUMINAMATH_CALUDE_triangle_inequality_l3130_313003

theorem triangle_inequality (a b c A B C : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  3 * a^2 + 3 * b^2 = c^2 + 4 * a * b →
  Real.tan (Real.sin A) ≤ Real.tan (Real.cos B) :=
sorry

end NUMINAMATH_CALUDE_triangle_inequality_l3130_313003


namespace NUMINAMATH_CALUDE_stock_price_increase_l3130_313046

theorem stock_price_increase (opening_price closing_price : ℝ) 
  (h1 : opening_price = 8) 
  (h2 : closing_price = 9) : 
  (closing_price - opening_price) / opening_price * 100 = 12.5 := by
  sorry

end NUMINAMATH_CALUDE_stock_price_increase_l3130_313046


namespace NUMINAMATH_CALUDE_map_scale_l3130_313015

/-- Given a map scale where 15 cm represents 90 km, prove that 20 cm represents 120 km -/
theorem map_scale (map_cm : ℝ) (real_km : ℝ) (h : map_cm = 15 ∧ real_km = 90) :
  (20 : ℝ) * (real_km / map_cm) = 120 := by
  sorry

end NUMINAMATH_CALUDE_map_scale_l3130_313015


namespace NUMINAMATH_CALUDE_sequence_terms_l3130_313085

def S (n : ℕ) : ℕ := n^2 + 1

def a (n : ℕ) : ℕ := S n - S (n-1)

theorem sequence_terms : a 3 = 5 ∧ a 5 = 9 := by sorry

end NUMINAMATH_CALUDE_sequence_terms_l3130_313085


namespace NUMINAMATH_CALUDE_largest_n_divisible_by_seven_l3130_313044

theorem largest_n_divisible_by_seven (n : ℕ) : n < 100000 ∧ 
  (∃ k : ℤ, 6 * (n - 3)^5 - n^2 + 16*n - 36 = 7 * k) →
  n ≤ 99996 :=
by sorry

end NUMINAMATH_CALUDE_largest_n_divisible_by_seven_l3130_313044


namespace NUMINAMATH_CALUDE_john_vacation_money_l3130_313034

/-- Converts a number from base 8 to base 10 -/
def base8ToBase10 (n : ℕ) : ℕ :=
  sorry

/-- Calculates the remaining money after buying a ticket -/
def remainingMoney (savings : ℕ) (ticketCost : ℕ) : ℕ :=
  savings - ticketCost

theorem john_vacation_money :
  let savings := base8ToBase10 5555
  let ticketCost := 1200
  remainingMoney savings ticketCost = 1725 := by
  sorry

end NUMINAMATH_CALUDE_john_vacation_money_l3130_313034


namespace NUMINAMATH_CALUDE_trig_expression_equals_eight_thirds_l3130_313008

theorem trig_expression_equals_eight_thirds :
  let sin30 : ℝ := 1/2
  let cos30 : ℝ := Real.sqrt 3 / 2
  (cos30^2 - sin30^2) / (cos30^2 * sin30^2) = 8/3 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_equals_eight_thirds_l3130_313008


namespace NUMINAMATH_CALUDE_alcohol_concentration_is_40_percent_l3130_313084

-- Define the ratios of water to alcohol in solutions A and B
def waterToAlcoholRatioA : Rat := 4 / 1
def waterToAlcoholRatioB : Rat := 2 / 3

-- Define the amount of each solution mixed (assuming 1 unit each)
def amountA : Rat := 1
def amountB : Rat := 1

-- Define the function to calculate the alcohol concentration in the mixed solution
def alcoholConcentration (ratioA ratioB amountA amountB : Rat) : Rat :=
  let waterA := amountA * (ratioA / (ratioA + 1))
  let alcoholA := amountA * (1 / (ratioA + 1))
  let waterB := amountB * (ratioB / (ratioB + 1))
  let alcoholB := amountB * (1 / (ratioB + 1))
  let totalAlcohol := alcoholA + alcoholB
  let totalMixture := waterA + alcoholA + waterB + alcoholB
  totalAlcohol / totalMixture

-- Theorem statement
theorem alcohol_concentration_is_40_percent :
  alcoholConcentration waterToAlcoholRatioA waterToAlcoholRatioB amountA amountB = 2/5 := by
  sorry


end NUMINAMATH_CALUDE_alcohol_concentration_is_40_percent_l3130_313084


namespace NUMINAMATH_CALUDE_sam_win_probability_proof_l3130_313055

/-- The probability of hitting the target with one shot -/
def hit_probability : ℚ := 2/5

/-- The probability of missing the target with one shot -/
def miss_probability : ℚ := 3/5

/-- Sam wins when the total number of shots (including the last successful one) is odd -/
axiom sam_wins_on_odd : True

/-- The probability that Sam wins the game -/
def sam_win_probability : ℚ := 5/8

theorem sam_win_probability_proof : 
  sam_win_probability = hit_probability + miss_probability * miss_probability * sam_win_probability :=
sorry

end NUMINAMATH_CALUDE_sam_win_probability_proof_l3130_313055


namespace NUMINAMATH_CALUDE_radio_selling_price_l3130_313057

/-- Calculates the selling price of an item given its cost price and loss percentage. -/
def sellingPrice (costPrice : ℚ) (lossPercentage : ℚ) : ℚ :=
  costPrice * (1 - lossPercentage / 100)

/-- Theorem stating that a radio purchased for Rs 490 with a 5% loss has a selling price of Rs 465.5. -/
theorem radio_selling_price :
  sellingPrice 490 5 = 465.5 := by
  sorry

#eval sellingPrice 490 5

end NUMINAMATH_CALUDE_radio_selling_price_l3130_313057


namespace NUMINAMATH_CALUDE_remaining_tickets_proof_l3130_313011

def tickets_to_be_sold (total_tickets jude_tickets : ℕ) : ℕ :=
  let andrea_tickets := 6 * jude_tickets
  let sandra_tickets := 3 * jude_tickets + 10
  total_tickets - (jude_tickets + andrea_tickets + sandra_tickets)

theorem remaining_tickets_proof (total_tickets jude_tickets : ℕ) 
  (h1 : total_tickets = 300) 
  (h2 : jude_tickets = 24) : 
  tickets_to_be_sold total_tickets jude_tickets = 50 := by
  sorry

end NUMINAMATH_CALUDE_remaining_tickets_proof_l3130_313011


namespace NUMINAMATH_CALUDE_fourth_mile_relation_l3130_313092

/-- Represents the relationship between distance and time for a mile -/
structure MileData where
  n : ℕ      -- The mile number
  time : ℝ    -- Time taken to cover the mile (in hours)
  distance : ℝ -- Distance covered (in miles)

/-- The constant k in the inverse relationship -/
def k : ℝ := 2

/-- The inverse relationship between distance and time for a given mile -/
def inverse_relation (md : MileData) : Prop :=
  md.distance = k / md.time

/-- Theorem stating the relationship for the 2nd and 4th miles -/
theorem fourth_mile_relation 
  (mile2 : MileData) 
  (mile4 : MileData) 
  (h1 : mile2.n = 2) 
  (h2 : mile2.time = 2) 
  (h3 : mile2.distance = 1) 
  (h4 : mile4.n = 4) 
  (h5 : inverse_relation mile2) 
  (h6 : inverse_relation mile4) : 
  mile4.time = 4 ∧ mile4.distance = 0.5 := by
  sorry


end NUMINAMATH_CALUDE_fourth_mile_relation_l3130_313092


namespace NUMINAMATH_CALUDE_coffee_table_price_is_330_l3130_313065

/-- The price of the coffee table in a living room set purchase --/
def coffee_table_price (sofa_price armchair_price total_invoice : ℕ) (num_armchairs : ℕ) : ℕ :=
  total_invoice - (sofa_price + num_armchairs * armchair_price)

/-- Theorem stating the price of the coffee table in the given scenario --/
theorem coffee_table_price_is_330 :
  coffee_table_price 1250 425 2430 2 = 330 := by
  sorry

end NUMINAMATH_CALUDE_coffee_table_price_is_330_l3130_313065


namespace NUMINAMATH_CALUDE_water_evaporation_per_day_l3130_313047

/-- Proves that given a bowl with 10 ounces of water, where 0.04% of the original amount
    evaporates over 50 days, the amount of water evaporated each day is 0.0008 ounces. -/
theorem water_evaporation_per_day
  (initial_water : Real)
  (evaporation_period : Nat)
  (evaporation_percentage : Real)
  (h1 : initial_water = 10)
  (h2 : evaporation_period = 50)
  (h3 : evaporation_percentage = 0.04)
  : (initial_water * evaporation_percentage / 100) / evaporation_period = 0.0008 := by
  sorry

#check water_evaporation_per_day

end NUMINAMATH_CALUDE_water_evaporation_per_day_l3130_313047


namespace NUMINAMATH_CALUDE_intersection_of_P_and_M_l3130_313025

-- Define the sets P and M
def P : Set ℝ := {x | 0 ≤ x ∧ x < 3}
def M : Set ℝ := {x | x^2 ≤ 9}

-- State the theorem
theorem intersection_of_P_and_M : P ∩ M = {x | 0 ≤ x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_P_and_M_l3130_313025


namespace NUMINAMATH_CALUDE_negative_fraction_comparison_l3130_313095

theorem negative_fraction_comparison : -1/2 < -1/3 := by
  sorry

end NUMINAMATH_CALUDE_negative_fraction_comparison_l3130_313095


namespace NUMINAMATH_CALUDE_point_on_y_axis_l3130_313004

/-- A point P with coordinates (2m, m+8) lies on the y-axis if and only if its coordinates are (0, 8) -/
theorem point_on_y_axis (m : ℝ) : 
  (∃ (P : ℝ × ℝ), P = (2*m, m+8) ∧ P.1 = 0) ↔ (∃ (P : ℝ × ℝ), P = (0, 8)) :=
by sorry

end NUMINAMATH_CALUDE_point_on_y_axis_l3130_313004


namespace NUMINAMATH_CALUDE_stratified_sampling_athletes_l3130_313017

theorem stratified_sampling_athletes (total_male : ℕ) (total_female : ℕ) 
  (selected_male : ℕ) (selected_female : ℕ) 
  (h1 : total_male = 56) (h2 : total_female = 42) (h3 : selected_male = 8) :
  (selected_male : ℚ) / total_male = selected_female / total_female → selected_female = 6 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_athletes_l3130_313017


namespace NUMINAMATH_CALUDE_quadratic_coefficient_l3130_313096

/-- A quadratic function with integer coefficients -/
structure QuadraticFunction where
  a : ℤ
  b : ℤ
  c : ℤ

/-- The y-coordinate for a given x in a quadratic function -/
def QuadraticFunction.eval (f : QuadraticFunction) (x : ℚ) : ℚ :=
  f.a * x^2 + f.b * x + f.c

theorem quadratic_coefficient (f : QuadraticFunction) 
  (vertex_x : f.eval 2 = 3)
  (point : f.eval 0 = 7) : 
  f.a = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_l3130_313096


namespace NUMINAMATH_CALUDE_wall_height_proof_l3130_313053

/-- Proves that the height of a wall is 600 cm given specific conditions --/
theorem wall_height_proof (brick_length brick_width brick_height : ℝ)
                          (wall_length wall_width : ℝ)
                          (num_bricks : ℕ) :
  brick_length = 25 →
  brick_width = 11.25 →
  brick_height = 6 →
  wall_length = 850 →
  wall_width = 22.5 →
  num_bricks = 6800 →
  ∃ (wall_height : ℝ),
    wall_height = 600 ∧
    num_bricks * (brick_length * brick_width * brick_height) =
    wall_length * wall_width * wall_height :=
by
  sorry

#check wall_height_proof

end NUMINAMATH_CALUDE_wall_height_proof_l3130_313053
