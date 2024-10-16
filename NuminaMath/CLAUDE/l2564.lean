import Mathlib

namespace NUMINAMATH_CALUDE_house_payment_theorem_l2564_256494

/-- Calculates the amount still owed on a house given the initial price,
    down payment percentage, and percentage paid by parents. -/
def amount_owed (price : ℝ) (down_payment_percent : ℝ) (parents_payment_percent : ℝ) : ℝ :=
  let remaining_after_down := price * (1 - down_payment_percent)
  remaining_after_down * (1 - parents_payment_percent)

/-- Theorem stating that for a $100,000 house with 20% down payment
    and 30% of the remaining balance paid by parents, $56,000 is still owed. -/
theorem house_payment_theorem :
  amount_owed 100000 0.2 0.3 = 56000 := by
  sorry

end NUMINAMATH_CALUDE_house_payment_theorem_l2564_256494


namespace NUMINAMATH_CALUDE_distance_to_bus_stand_l2564_256495

/-- The distance to the bus stand in kilometers -/
def distance : ℝ := 13.5

/-- The time at which the bus arrives in hours -/
def bus_arrival_time : ℝ := 2.5

/-- Theorem stating that the distance to the bus stand is 13.5 km -/
theorem distance_to_bus_stand :
  (distance = 5 * (bus_arrival_time + 0.2)) ∧
  (distance = 6 * (bus_arrival_time - 0.25)) :=
by sorry

end NUMINAMATH_CALUDE_distance_to_bus_stand_l2564_256495


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2564_256447

/-- The eccentricity of a hyperbola with specific conditions -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let C := {(x, y) : ℝ × ℝ | x^2 / a^2 - y^2 / b^2 = 1}
  let l := {(x, y) : ℝ × ℝ | x - 2*y + 1 = 0}
  let asymptote_slope := b / a
  let line_slope := 1 / 2
  (asymptote_slope = 2 * line_slope / (1 - line_slope^2)) →
  Real.sqrt (1 + (b/a)^2) = 5/3 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2564_256447


namespace NUMINAMATH_CALUDE_new_person_weight_l2564_256489

/-- Given a group of 8 people, if replacing a person weighing 45 kg with a new person
    increases the average weight by 2.5 kg, then the new person weighs 65 kg. -/
theorem new_person_weight (initial_count : Nat) (weight_replaced : ℝ) (avg_increase : ℝ) :
  initial_count = 8 →
  weight_replaced = 45 →
  avg_increase = 2.5 →
  (initial_count : ℝ) * avg_increase + weight_replaced = 65 := by
  sorry

#check new_person_weight

end NUMINAMATH_CALUDE_new_person_weight_l2564_256489


namespace NUMINAMATH_CALUDE_count_special_numbers_eq_3776_l2564_256430

/-- A function that counts the number of five-digit numbers beginning with 2 
    that have exactly two identical digits -/
def count_special_numbers : ℕ :=
  let case1 := 4 * 8 * 8 * 8  -- Case where the two identical digits are 2s
  let case2 := 3 * 9 * 8 * 8  -- Case where the two identical digits are not 2s
  case1 + case2

/-- Theorem stating that there are exactly 3776 five-digit numbers 
    beginning with 2 that have exactly two identical digits -/
theorem count_special_numbers_eq_3776 :
  count_special_numbers = 3776 := by
  sorry

#eval count_special_numbers  -- Should output 3776

end NUMINAMATH_CALUDE_count_special_numbers_eq_3776_l2564_256430


namespace NUMINAMATH_CALUDE_partnership_profit_l2564_256418

/-- Given the investments of three partners and one partner's share of the profit,
    calculate the total profit of the partnership. -/
theorem partnership_profit
  (investment_A investment_B investment_C : ℕ)
  (profit_share_A : ℕ)
  (h1 : investment_A = 6300)
  (h2 : investment_B = 4200)
  (h3 : investment_C = 10500)
  (h4 : profit_share_A = 3660) :
  (investment_A + investment_B + investment_C) * profit_share_A / investment_A = 12200 :=
by sorry

end NUMINAMATH_CALUDE_partnership_profit_l2564_256418


namespace NUMINAMATH_CALUDE_number_divided_by_six_l2564_256445

theorem number_divided_by_six : ∃ n : ℝ, n / 6 = 26 ∧ n = 156 := by
  sorry

end NUMINAMATH_CALUDE_number_divided_by_six_l2564_256445


namespace NUMINAMATH_CALUDE_last_three_digits_of_3_to_12000_l2564_256464

theorem last_three_digits_of_3_to_12000 (h : 3^400 ≡ 1 [ZMOD 1000]) :
  3^12000 ≡ 1 [ZMOD 1000] := by
  sorry

end NUMINAMATH_CALUDE_last_three_digits_of_3_to_12000_l2564_256464


namespace NUMINAMATH_CALUDE_bus_car_ratio_l2564_256423

theorem bus_car_ratio (num_cars : ℕ) (num_buses : ℕ) : 
  num_cars = 65 →
  num_buses = num_cars - 60 →
  (num_buses : ℚ) / (num_cars : ℚ) = 1 / 13 := by
  sorry

end NUMINAMATH_CALUDE_bus_car_ratio_l2564_256423


namespace NUMINAMATH_CALUDE_work_group_size_work_group_size_is_9_l2564_256428

theorem work_group_size (days1 : ℕ) (days2 : ℕ) (men2 : ℕ) : ℕ :=
  let work_constant := men2 * days2
  let men1 := work_constant / days1
  men1

theorem work_group_size_is_9 :
  work_group_size 80 36 20 = 9 := by
  sorry

end NUMINAMATH_CALUDE_work_group_size_work_group_size_is_9_l2564_256428


namespace NUMINAMATH_CALUDE_max_temperature_range_l2564_256400

/-- Given weather conditions and temperatures, calculate the maximum temperature range --/
theorem max_temperature_range 
  (avg_temp : ℝ) 
  (lowest_temp : ℝ) 
  (temp_fluctuation : ℝ) 
  (h1 : avg_temp = 50)
  (h2 : lowest_temp = 45)
  (h3 : temp_fluctuation = 5) :
  (avg_temp + temp_fluctuation) - lowest_temp = 10 := by
sorry

end NUMINAMATH_CALUDE_max_temperature_range_l2564_256400


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2564_256403

theorem complex_equation_solution (a b : ℝ) (h : (3 + 4*I) * (1 + a*I) = b*I) : a = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2564_256403


namespace NUMINAMATH_CALUDE_gcd_n_minus_three_n_plus_three_eq_one_l2564_256477

theorem gcd_n_minus_three_n_plus_three_eq_one (n : ℕ+) 
  (h : (Nat.divisors (n.val^2 - 9)).card = 6) : 
  Nat.gcd (n.val - 3) (n.val + 3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_n_minus_three_n_plus_three_eq_one_l2564_256477


namespace NUMINAMATH_CALUDE_line_slope_product_l2564_256453

/-- Given two lines L₁ and L₂ with equations y = 3mx and y = nx respectively,
    where L₁ makes three times the angle with the horizontal as L₂,
    L₁ has 3 times the slope of L₂, and L₁ is not vertical,
    prove that the product mn equals 9/4. -/
theorem line_slope_product (m n : ℝ) : 
  (∃ θ₁ θ₂ : ℝ, θ₁ = 3 * θ₂ ∧ 
               3 * m = Real.tan θ₁ ∧ 
               n = Real.tan θ₂ ∧ 
               m = 3 * n ∧ 
               m ≠ 0) →
  m * n = 9 / 4 := by
sorry

end NUMINAMATH_CALUDE_line_slope_product_l2564_256453


namespace NUMINAMATH_CALUDE_circle_area_sum_l2564_256415

/-- The sum of the areas of an infinite sequence of circles, where the radius of the first circle
    is 1 and each subsequent circle's radius is 2/3 of the previous one, is equal to 9π/5. -/
theorem circle_area_sum : 
  let radius : ℕ → ℝ := λ n => (2/3)^(n-1)
  let area : ℕ → ℝ := λ n => π * (radius n)^2
  (∑' n, area n) = 9*π/5 := by sorry

end NUMINAMATH_CALUDE_circle_area_sum_l2564_256415


namespace NUMINAMATH_CALUDE_no_one_common_tangent_l2564_256440

/-- Represents a circle in a plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Predicate to check if two circles have different radii --/
def hasDifferentRadii (c1 c2 : Circle) : Prop :=
  c1.radius ≠ c2.radius

/-- Counts the number of common tangents between two circles --/
def commonTangentsCount (c1 c2 : Circle) : ℕ := sorry

/-- Theorem stating that two circles with different radii cannot have exactly one common tangent --/
theorem no_one_common_tangent (c1 c2 : Circle) 
  (h : hasDifferentRadii c1 c2) : 
  commonTangentsCount c1 c2 ≠ 1 := by sorry

end NUMINAMATH_CALUDE_no_one_common_tangent_l2564_256440


namespace NUMINAMATH_CALUDE_matching_probability_is_four_fifteenths_l2564_256414

/-- Represents the number of jelly beans of each color for a person -/
structure JellyBeans where
  green : ℕ
  red : ℕ
  blue : ℕ
  yellow : ℕ

/-- Calculates the total number of jelly beans -/
def JellyBeans.total (jb : JellyBeans) : ℕ :=
  jb.green + jb.red + jb.blue + jb.yellow

/-- Alice's jelly bean distribution -/
def alice : JellyBeans := { green := 2, red := 2, blue := 1, yellow := 0 }

/-- Carl's jelly bean distribution -/
def carl : JellyBeans := { green := 3, red := 1, blue := 0, yellow := 2 }

/-- The probability of selecting matching colors -/
def matchingProbability (a c : JellyBeans) : ℚ :=
  (a.green * c.green + a.red * c.red) / (a.total * c.total)

theorem matching_probability_is_four_fifteenths :
  matchingProbability alice carl = 4 / 15 := by
  sorry

end NUMINAMATH_CALUDE_matching_probability_is_four_fifteenths_l2564_256414


namespace NUMINAMATH_CALUDE_pythagorean_diagonal_l2564_256448

theorem pythagorean_diagonal (m : ℕ) (h : m ≥ 3) :
  let width := 2 * m
  let diagonal := m^2 - 1
  let height := diagonal - 2
  width^2 + height^2 = diagonal^2 :=
by sorry

end NUMINAMATH_CALUDE_pythagorean_diagonal_l2564_256448


namespace NUMINAMATH_CALUDE_middle_term_value_l2564_256439

/-- An arithmetic sequence with 9 terms -/
def ArithmeticSequence (a : ℕ → ℚ) : Prop :=
  ∃ (a₁ d : ℚ), ∀ n, a n = a₁ + (n - 1) * d

theorem middle_term_value
  (a : ℕ → ℚ)
  (h_arithmetic : ArithmeticSequence a)
  (h_sum_first_4 : (a 1) + (a 2) + (a 3) + (a 4) = 3)
  (h_sum_last_3 : (a 7) + (a 8) + (a 9) = 4) :
  a 5 = 19 / 148 := by
  sorry

end NUMINAMATH_CALUDE_middle_term_value_l2564_256439


namespace NUMINAMATH_CALUDE_rent_increase_problem_l2564_256417

theorem rent_increase_problem (initial_average : ℝ) (new_average : ℝ) (num_friends : ℕ) 
  (increase_percentage : ℝ) (h1 : initial_average = 800) (h2 : new_average = 850) 
  (h3 : num_friends = 4) (h4 : increase_percentage = 0.25) : 
  ∃ (original_rent : ℝ), 
    (num_friends * new_average - num_friends * initial_average) / increase_percentage = original_rent ∧ 
    original_rent = 800 := by
  sorry

end NUMINAMATH_CALUDE_rent_increase_problem_l2564_256417


namespace NUMINAMATH_CALUDE_inequality_proof_l2564_256419

theorem inequality_proof (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) 
  (h : a + b < c + d) : 
  a * c + b * d > a * b := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l2564_256419


namespace NUMINAMATH_CALUDE_triangle_theorem_l2564_256487

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the theorem
theorem triangle_theorem (abc : Triangle) (h1 : abc.b / abc.a = Real.sin abc.B / Real.sin (2 * abc.A))
  (h2 : abc.b = 2 * Real.sqrt 3) (h3 : 1/2 * abc.b * abc.c * Real.sin abc.A = 3 * Real.sqrt 3 / 2) :
  abc.A = π/3 ∧ abc.a = 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_theorem_l2564_256487


namespace NUMINAMATH_CALUDE_fruit_condition_percentage_l2564_256498

theorem fruit_condition_percentage 
  (total_oranges : ℕ) 
  (total_bananas : ℕ) 
  (rotten_oranges_percent : ℚ) 
  (rotten_bananas_percent : ℚ) 
  (h1 : total_oranges = 600) 
  (h2 : total_bananas = 400) 
  (h3 : rotten_oranges_percent = 15 / 100) 
  (h4 : rotten_bananas_percent = 8 / 100) : 
  (1 - (rotten_oranges_percent * total_oranges + rotten_bananas_percent * total_bananas) / 
   (total_oranges + total_bananas)) * 100 = 878 / 10 :=
by sorry

end NUMINAMATH_CALUDE_fruit_condition_percentage_l2564_256498


namespace NUMINAMATH_CALUDE_unique_function_theorem_l2564_256480

/-- A function from rational numbers to rational numbers -/
def RationalFunction := ℚ → ℚ

/-- The property that a function satisfies the given conditions -/
def SatisfiesConditions (f : RationalFunction) : Prop :=
  f 1 = 2 ∧ ∀ x y : ℚ, f (x * y) = f x * f y - f (x + y) + 1

/-- The theorem statement -/
theorem unique_function_theorem :
  ∀ f : RationalFunction, SatisfiesConditions f → ∀ x : ℚ, f x = x + 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_function_theorem_l2564_256480


namespace NUMINAMATH_CALUDE_car_speed_problem_l2564_256411

theorem car_speed_problem (distance : ℝ) (original_time : ℝ) (new_time : ℝ) (new_speed : ℝ) :
  original_time = 12 →
  new_time = 4 →
  new_speed = 30 →
  distance = new_speed * new_time →
  distance = (distance / original_time) * original_time →
  distance / original_time = 10 := by
sorry

end NUMINAMATH_CALUDE_car_speed_problem_l2564_256411


namespace NUMINAMATH_CALUDE_factorial_square_gt_power_1000_l2564_256432

theorem factorial_square_gt_power_1000 :
  (Nat.factorial 1000)^2 > 1000^1000 := by
  sorry

end NUMINAMATH_CALUDE_factorial_square_gt_power_1000_l2564_256432


namespace NUMINAMATH_CALUDE_tank_filling_time_l2564_256401

theorem tank_filling_time (fast_rate slow_rate : ℝ) (combined_time : ℝ) : 
  fast_rate = 4 * slow_rate →
  1 / combined_time = fast_rate + slow_rate →
  combined_time = 40 →
  1 / slow_rate = 200 := by
sorry

end NUMINAMATH_CALUDE_tank_filling_time_l2564_256401


namespace NUMINAMATH_CALUDE_problem1_l2564_256425

theorem problem1 (a b : ℝ) (h1 : a ≥ b) (h2 : b > 0) :
  2 * a^3 - b^3 ≥ 2 * a * b^2 - a^2 * b :=
by sorry

end NUMINAMATH_CALUDE_problem1_l2564_256425


namespace NUMINAMATH_CALUDE_product_cube_l2564_256499

theorem product_cube (a b c : ℕ+) (h : a * b * c = 180) : (a * b) ^ 3 = 216 := by
  sorry

end NUMINAMATH_CALUDE_product_cube_l2564_256499


namespace NUMINAMATH_CALUDE_book_input_time_l2564_256455

theorem book_input_time : ∃ (n : ℕ) (T : ℚ),
  T > 0 ∧
  n > 3 ∧
  (n : ℚ) * T = (n + 3) * (3/4 * T) ∧
  (n - 3) * (T + 5/6) = (n : ℚ) * T ∧
  T = 5/3 := by
sorry

end NUMINAMATH_CALUDE_book_input_time_l2564_256455


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l2564_256472

theorem sum_of_three_numbers (a b c : ℤ) 
  (sum_ab : a + b = 35)
  (sum_bc : b + c = 52)
  (sum_ca : c + a = 61) :
  a + b + c = 74 := by
sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l2564_256472


namespace NUMINAMATH_CALUDE_problem_solution_l2564_256486

theorem problem_solution (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x * y * z = 1)
  (h2 : x + 1 / z = 8)
  (h3 : y + 1 / x = 29) :
  z + 1 / y = 13 / 77 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2564_256486


namespace NUMINAMATH_CALUDE_starting_lineup_combinations_l2564_256437

/-- The number of players on the team -/
def total_players : ℕ := 15

/-- The number of players in the starting lineup -/
def lineup_size : ℕ := 6

/-- The number of pre-selected players (All-Stars) -/
def preselected_players : ℕ := 3

/-- The number of different starting lineups possible -/
def num_lineups : ℕ := 220

theorem starting_lineup_combinations :
  Nat.choose (total_players - preselected_players) (lineup_size - preselected_players) = num_lineups :=
sorry

end NUMINAMATH_CALUDE_starting_lineup_combinations_l2564_256437


namespace NUMINAMATH_CALUDE_polar_to_cartesian_l2564_256412

theorem polar_to_cartesian (M : ℝ × ℝ) :
  M.1 = 3 ∧ M.2 = π / 6 →
  (M.1 * Real.cos M.2 = 3 * Real.sqrt 3 / 2) ∧
  (M.1 * Real.sin M.2 = 3 / 2) := by
sorry

end NUMINAMATH_CALUDE_polar_to_cartesian_l2564_256412


namespace NUMINAMATH_CALUDE_square_area_from_diagonal_l2564_256431

theorem square_area_from_diagonal (d : ℝ) (h : d = 16) :
  let s := d / Real.sqrt 2
  s * s = 128 := by
  sorry

end NUMINAMATH_CALUDE_square_area_from_diagonal_l2564_256431


namespace NUMINAMATH_CALUDE_imaginary_number_real_part_l2564_256493

theorem imaginary_number_real_part (a : ℝ) : 
  let z : ℂ := a + (Complex.I / (1 - Complex.I))
  (∃ (b : ℝ), z = Complex.I * b) → a = (1/2 : ℝ) := by
sorry

end NUMINAMATH_CALUDE_imaginary_number_real_part_l2564_256493


namespace NUMINAMATH_CALUDE_prob_no_shaded_correct_l2564_256452

/-- Represents a rectangle in the 2 by 2005 grid -/
structure Rectangle where
  left : Fin 2006
  right : Fin 2006
  top : Fin 3
  bottom : Fin 3
  h_valid : left < right

/-- The total number of rectangles in the grid -/
def total_rectangles : ℕ := 3 * (1003 * 2005)

/-- The number of rectangles containing a shaded square -/
def shaded_rectangles : ℕ := 3 * (1003 * 1003)

/-- Predicate for whether a rectangle contains a shaded square -/
def contains_shaded (r : Rectangle) : Prop :=
  (r.left ≤ 1003 ∧ r.right > 1003) ∨ (r.top = 0 ∧ r.bottom = 1) ∨ (r.top = 1 ∧ r.bottom = 2)

/-- The probability of choosing a rectangle that does not contain a shaded square -/
def prob_no_shaded : ℚ := 1002 / 2005

theorem prob_no_shaded_correct :
  (total_rectangles - shaded_rectangles : ℚ) / total_rectangles = prob_no_shaded := by
  sorry

end NUMINAMATH_CALUDE_prob_no_shaded_correct_l2564_256452


namespace NUMINAMATH_CALUDE_lcm_problem_l2564_256413

theorem lcm_problem (n : ℕ+) : ∃ n, 
  n > 0 ∧ 
  216 % n = 0 ∧ 
  n % 3 = 0 ∧ 
  n ≤ 9 ∧
  Nat.lcm (Nat.lcm (Nat.lcm 8 24) 36) n = 216 := by
sorry

end NUMINAMATH_CALUDE_lcm_problem_l2564_256413


namespace NUMINAMATH_CALUDE_arrangements_six_people_one_restricted_l2564_256420

def number_of_arrangements (n : ℕ) : ℕ :=
  (n - 1) * (Nat.factorial (n - 1))

theorem arrangements_six_people_one_restricted :
  number_of_arrangements 6 = 600 := by
  sorry

end NUMINAMATH_CALUDE_arrangements_six_people_one_restricted_l2564_256420


namespace NUMINAMATH_CALUDE_correct_average_l2564_256451

theorem correct_average (n : ℕ) (incorrect_avg : ℚ) (incorrect_num correct_num : ℚ) :
  n = 10 ∧ 
  incorrect_avg = 46 ∧ 
  incorrect_num = 25 ∧ 
  correct_num = 65 →
  (n : ℚ) * incorrect_avg + (correct_num - incorrect_num) = n * 50 := by
sorry

end NUMINAMATH_CALUDE_correct_average_l2564_256451


namespace NUMINAMATH_CALUDE_evaluate_expression_l2564_256479

theorem evaluate_expression : 6 - 5 * (10 - (2 + 3)^2) * 2 = 306 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2564_256479


namespace NUMINAMATH_CALUDE_article_price_l2564_256409

theorem article_price (selling_price : ℝ) (gain_percent : ℝ) 
  (h1 : selling_price = 110)
  (h2 : gain_percent = 10) : 
  ∃ (original_price : ℝ), 
    selling_price = original_price * (1 + gain_percent / 100) ∧ 
    original_price = 100 := by
  sorry

end NUMINAMATH_CALUDE_article_price_l2564_256409


namespace NUMINAMATH_CALUDE_angle_cosine_equivalence_l2564_256488

-- Define a structure for a triangle
structure Triangle where
  A : Real
  B : Real
  C : Real
  angle_sum : A + B + C = π
  positive_angles : 0 < A ∧ 0 < B ∧ 0 < C

-- Define the theorem
theorem angle_cosine_equivalence (t : Triangle) :
  (t.A > t.B ↔ Real.cos t.A < Real.cos t.B) :=
by sorry

end NUMINAMATH_CALUDE_angle_cosine_equivalence_l2564_256488


namespace NUMINAMATH_CALUDE_cats_eating_mice_l2564_256407

/-- If n cats eat n mice in n hours, then p cats eat (p^2 / n) mice in p hours -/
theorem cats_eating_mice (n p : ℕ) (h : n ≠ 0) : 
  (n : ℚ) * (n : ℚ) / (n : ℚ) = n → (p : ℚ) * (p : ℚ) / (n : ℚ) = p^2 / n := by
  sorry

end NUMINAMATH_CALUDE_cats_eating_mice_l2564_256407


namespace NUMINAMATH_CALUDE_chris_birthday_money_l2564_256468

theorem chris_birthday_money (x : ℕ) : 
  x + 25 + 20 + 75 = 279 → x = 159 := by sorry

end NUMINAMATH_CALUDE_chris_birthday_money_l2564_256468


namespace NUMINAMATH_CALUDE_isosceles_triangle_side_lengths_l2564_256462

/-- Represents the side lengths of an isosceles triangle -/
structure IsoscelesTriangle where
  a : ℝ  -- Length of one side
  b : ℝ  -- Length of the base or the equal side
  h : a > 0 ∧ b > 0  -- Lengths are positive

/-- Checks if the given side lengths form a valid triangle -/
def is_valid_triangle (t : IsoscelesTriangle) : Prop :=
  t.a + t.b > t.a ∧ t.a + t.a > t.b

/-- Calculates the perimeter of an isosceles triangle -/
def perimeter (t : IsoscelesTriangle) : ℝ := 2 * t.a + t.b

theorem isosceles_triangle_side_lengths :
  ∀ t : IsoscelesTriangle,
    is_valid_triangle t →
    perimeter t = 17 →
    (t.a = 4 ∨ t.b = 4) →
    ((t.a = 6 ∧ t.b = 5) ∨ (t.a = 5 ∧ t.b = 7)) :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_side_lengths_l2564_256462


namespace NUMINAMATH_CALUDE_different_author_book_pairs_l2564_256441

/-- Given two groups of books, this theorem proves that the number of different pairs
    that can be formed by selecting one book from each group is equal to the product
    of the number of books in each group. -/
theorem different_author_book_pairs (group1 group2 : ℕ) (h1 : group1 = 6) (h2 : group2 = 9) :
  group1 * group2 = 54 := by
  sorry

end NUMINAMATH_CALUDE_different_author_book_pairs_l2564_256441


namespace NUMINAMATH_CALUDE_total_length_climbed_50_30_6_25_l2564_256459

/-- The total length of ladders climbed by two workers in centimeters -/
def total_length_climbed (keaton_ladder_height : ℕ) (keaton_climbs : ℕ) 
  (reece_ladder_diff : ℕ) (reece_climbs : ℕ) : ℕ :=
  let reece_ladder_height := keaton_ladder_height - reece_ladder_diff
  let keaton_total := keaton_ladder_height * keaton_climbs
  let reece_total := reece_ladder_height * reece_climbs
  (keaton_total + reece_total) * 100

/-- Theorem stating the total length climbed by both workers -/
theorem total_length_climbed_50_30_6_25 :
  total_length_climbed 50 30 6 25 = 260000 := by
  sorry

end NUMINAMATH_CALUDE_total_length_climbed_50_30_6_25_l2564_256459


namespace NUMINAMATH_CALUDE_find_m_l2564_256456

theorem find_m (f : ℝ → ℝ) (m : ℝ) 
  (h1 : ∀ x, f (1/2 * x - 1) = 2*x + 3)
  (h2 : f m = 6) :
  m = -1/4 := by
  sorry

end NUMINAMATH_CALUDE_find_m_l2564_256456


namespace NUMINAMATH_CALUDE_trig_fraction_value_l2564_256446

theorem trig_fraction_value (α : Real) (h : Real.tan α = 2) :
  1 / (2 * Real.sin α * Real.cos α + Real.cos α ^ 2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_trig_fraction_value_l2564_256446


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l2564_256497

-- Define the quadratic function
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define g(x) = f(x) - x^2
def g (a b c : ℝ) (x : ℝ) : ℝ := f a b c x - x^2

-- Theorem statement
theorem quadratic_function_properties
  (a b c : ℝ)
  (origin : f a b c 0 = 0)
  (symmetry : ∀ x, f a b c (1 - x) = f a b c (1 + x))
  (odd_g : ∀ x, g a b c x = -g a b c (-x))
  : f a b c = fun x ↦ x^2 - 2*x :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l2564_256497


namespace NUMINAMATH_CALUDE_library_books_l2564_256461

theorem library_books (shelves : ℝ) (books_per_shelf : ℕ) : 
  shelves = 14240.0 → books_per_shelf = 8 → shelves * (books_per_shelf : ℝ) = 113920 := by
  sorry

end NUMINAMATH_CALUDE_library_books_l2564_256461


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l2564_256424

/-- Given vectors a and b in R², if a is perpendicular to (t*a + b), then t = -5 -/
theorem perpendicular_vectors (a b : ℝ × ℝ) (t : ℝ) 
  (h1 : a = (1, -1))
  (h2 : b = (6, -4))
  (h3 : a.1 * (t * a.1 + b.1) + a.2 * (t * a.2 + b.2) = 0) :
  t = -5 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l2564_256424


namespace NUMINAMATH_CALUDE_bella_roses_l2564_256454

/-- The number of roses in a dozen -/
def dozen : ℕ := 12

/-- The number of dozens of roses Bella received from her parents -/
def roses_from_parents : ℕ := 2

/-- The number of Bella's dancer friends -/
def number_of_friends : ℕ := 10

/-- The number of roses each friend gave to Bella -/
def roses_per_friend : ℕ := 2

/-- The total number of roses Bella received -/
def total_roses : ℕ := roses_from_parents * dozen + number_of_friends * roses_per_friend

theorem bella_roses : total_roses = 44 := by
  sorry

end NUMINAMATH_CALUDE_bella_roses_l2564_256454


namespace NUMINAMATH_CALUDE_exam_score_problem_l2564_256457

theorem exam_score_problem (total_questions : ℕ) (correct_score : ℤ) (wrong_score : ℤ) (total_score : ℤ) :
  total_questions = 60 →
  correct_score = 4 →
  wrong_score = -1 →
  total_score = 130 →
  ∃ (correct_answers : ℕ),
    correct_answers ≤ total_questions ∧
    correct_score * correct_answers + wrong_score * (total_questions - correct_answers) = total_score ∧
    correct_answers = 38 := by
  sorry

end NUMINAMATH_CALUDE_exam_score_problem_l2564_256457


namespace NUMINAMATH_CALUDE_smallest_a_for_composite_f_l2564_256408

/-- A function that represents x^4 + a^2 --/
def f (x a : ℤ) : ℤ := x^4 + a^2

/-- Definition of a composite number --/
def is_composite (n : ℤ) : Prop := ∃ (a b : ℤ), a ≠ 1 ∧ a ≠ -1 ∧ a ≠ n ∧ a ≠ -n ∧ n = a * b

/-- The main theorem --/
theorem smallest_a_for_composite_f :
  ∀ x : ℤ, is_composite (f x 8) ∧
  ∀ a : ℕ, a > 0 ∧ a < 8 → ∃ x : ℤ, ¬is_composite (f x a) :=
sorry

end NUMINAMATH_CALUDE_smallest_a_for_composite_f_l2564_256408


namespace NUMINAMATH_CALUDE_difference_of_squares_503_497_l2564_256410

theorem difference_of_squares_503_497 : 503^2 - 497^2 = 6000 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_503_497_l2564_256410


namespace NUMINAMATH_CALUDE_blackboard_numbers_l2564_256404

def can_be_written (n : ℕ) : Prop :=
  ∃ x y : ℕ, n + 1 = 2^x * 3^y

theorem blackboard_numbers (n : ℕ) :
  can_be_written n ↔ 
  (n = 13121 ∨ (∃ a b : ℕ, can_be_written a ∧ can_be_written b ∧ n = a * b + a + b)) :=
sorry

end NUMINAMATH_CALUDE_blackboard_numbers_l2564_256404


namespace NUMINAMATH_CALUDE_multiplicative_magic_square_exists_l2564_256436

/-- Represents a 3x3 matrix --/
def Matrix3x3 := Fin 3 → Fin 3 → ℕ

/-- Check if two numbers have the same digits --/
def same_digits (a b : ℕ) : Prop := sorry

/-- The original magic square --/
def original_square : Matrix3x3 := 
  fun i j => match i, j with
  | 0, 0 => 27 | 0, 1 => 20 | 0, 2 => 25
  | 1, 0 => 22 | 1, 1 => 24 | 1, 2 => 26
  | 2, 0 => 23 | 2, 1 => 28 | 2, 2 => 21

/-- Product of elements in a row --/
def row_product (m : Matrix3x3) (i : Fin 3) : ℕ :=
  (m i 0) * (m i 1) * (m i 2)

/-- Product of elements in a column --/
def col_product (m : Matrix3x3) (j : Fin 3) : ℕ :=
  (m 0 j) * (m 1 j) * (m 2 j)

/-- Product of elements in the main diagonal --/
def diag_product (m : Matrix3x3) : ℕ :=
  (m 0 0) * (m 1 1) * (m 2 2)

/-- Product of elements in the anti-diagonal --/
def antidiag_product (m : Matrix3x3) : ℕ :=
  (m 0 2) * (m 1 1) * (m 2 0)

/-- The theorem to be proved --/
theorem multiplicative_magic_square_exists : ∃ (m : Matrix3x3), 
  (∀ i j, same_digits (m i j) (original_square i j)) ∧ 
  (∀ i : Fin 3, row_product m i = 7488) ∧
  (∀ j : Fin 3, col_product m j = 7488) ∧
  (diag_product m = 7488) ∧
  (antidiag_product m = 7488) := by
  sorry

end NUMINAMATH_CALUDE_multiplicative_magic_square_exists_l2564_256436


namespace NUMINAMATH_CALUDE_min_people_like_both_tea_and_coffee_l2564_256491

theorem min_people_like_both_tea_and_coffee
  (total : ℕ)
  (tea_lovers : ℕ)
  (coffee_lovers : ℕ)
  (h1 : total = 150)
  (h2 : tea_lovers = 120)
  (h3 : coffee_lovers = 100) :
  (tea_lovers + coffee_lovers - total : ℤ) ≥ 70 :=
sorry

end NUMINAMATH_CALUDE_min_people_like_both_tea_and_coffee_l2564_256491


namespace NUMINAMATH_CALUDE_parallelogram_height_l2564_256492

/-- The height of a parallelogram given its area and base -/
theorem parallelogram_height (area : ℝ) (base : ℝ) (height : ℝ) 
  (h_area : area = 216) 
  (h_base : base = 12) 
  (h_formula : area = base * height) : 
  height = 18 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_height_l2564_256492


namespace NUMINAMATH_CALUDE_melted_ice_cream_depth_l2564_256466

/-- Given a spherical scoop of ice cream with radius 3 inches that melts into a conical shape with radius 9 inches, prove that the height of the resulting cone is 4/3 inches, assuming constant density. -/
theorem melted_ice_cream_depth (sphere_radius : ℝ) (cone_radius : ℝ) (cone_height : ℝ) : 
  sphere_radius = 3 →
  cone_radius = 9 →
  (4 / 3) * Real.pi * sphere_radius^3 = (1 / 3) * Real.pi * cone_radius^2 * cone_height →
  cone_height = 4 / 3 := by
  sorry

#check melted_ice_cream_depth

end NUMINAMATH_CALUDE_melted_ice_cream_depth_l2564_256466


namespace NUMINAMATH_CALUDE_f_inequality_l2564_256405

noncomputable def f (x : ℝ) : ℝ := 
  if x ≤ 0 then x + 1 else Real.exp (x * Real.log 2)

theorem f_inequality (x : ℝ) : 
  f x + f (x - 1/2) > 1 ↔ x > -1/4 := by sorry

end NUMINAMATH_CALUDE_f_inequality_l2564_256405


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2564_256421

theorem inequality_solution_set (x : ℝ) :
  x ≠ 1 →
  ((x^2 + x - 6) / (x - 1) ≤ 0 ↔ x ∈ Set.Iic (-3) ∪ Set.Ioo 1 2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2564_256421


namespace NUMINAMATH_CALUDE_flag_designs_count_l2564_256490

/-- The number of colors available for each stripe -/
def num_colors : ℕ := 3

/-- The number of stripes in the flag -/
def num_stripes : ℕ := 3

/-- The total number of possible flag designs -/
def total_flag_designs : ℕ := num_colors ^ num_stripes

/-- Theorem stating that the total number of possible flag designs is 27 -/
theorem flag_designs_count : total_flag_designs = 27 := by
  sorry

end NUMINAMATH_CALUDE_flag_designs_count_l2564_256490


namespace NUMINAMATH_CALUDE_abs_sum_inequality_solution_set_square_sum_geq_sqrt_product_sum_l2564_256402

-- Part Ⅰ
theorem abs_sum_inequality_solution_set (x : ℝ) :
  (|2 + x| + |2 - x| ≤ 4) ↔ (-2 ≤ x ∧ x ≤ 2) := by sorry

-- Part Ⅱ
theorem square_sum_geq_sqrt_product_sum {a b : ℝ} (ha : 0 < a) (hb : 0 < b) :
  a^2 + b^2 ≥ Real.sqrt (a * b) * (a + b) := by sorry

end NUMINAMATH_CALUDE_abs_sum_inequality_solution_set_square_sum_geq_sqrt_product_sum_l2564_256402


namespace NUMINAMATH_CALUDE_correct_statements_l2564_256469

-- Define a differentiable function
variable (f : ℝ → ℝ) (hf : Differentiable ℝ f)

-- Define extremum
def HasExtremumAt (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  ∀ x, f x ≤ f x₀ ∨ f x ≥ f x₀

-- Define inductive and deductive reasoning
def InductiveReasoning : Prop :=
  ∃ (specific general : Prop), specific → general

def DeductiveReasoning : Prop :=
  ∃ (general specific : Prop), general → specific

-- Define synthetic and analytic methods
def SyntheticMethod : Prop :=
  ∃ (cause effect : Prop), cause → effect

def AnalyticMethod : Prop :=
  ∃ (effect cause : Prop), effect → cause

-- Theorem statement
theorem correct_statements
  (x₀ : ℝ)
  (h_extremum : HasExtremumAt f x₀) :
  (deriv f x₀ = 0) ∧
  InductiveReasoning ∧
  DeductiveReasoning ∧
  SyntheticMethod ∧
  AnalyticMethod :=
sorry

end NUMINAMATH_CALUDE_correct_statements_l2564_256469


namespace NUMINAMATH_CALUDE_hockey_league_games_l2564_256427

/-- The number of games played in a hockey league season --/
def hockey_games (n : ℕ) (m : ℕ) : ℕ :=
  n * (n - 1) / 2 * m

/-- Theorem: In a hockey league with 25 teams, where each team faces all other teams 15 times, 
    the total number of games played in the season is 4500. --/
theorem hockey_league_games : hockey_games 25 15 = 4500 := by
  sorry

end NUMINAMATH_CALUDE_hockey_league_games_l2564_256427


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_real_roots_l2564_256483

theorem quadratic_two_distinct_real_roots :
  let a : ℝ := 2
  let b : ℝ := 3
  let c : ℝ := -4
  let discriminant := b^2 - 4*a*c
  discriminant > 0 ∧ ∃ x y : ℝ, x ≠ y ∧ a*x^2 + b*x + c = 0 ∧ a*y^2 + b*y + c = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_real_roots_l2564_256483


namespace NUMINAMATH_CALUDE_binomial_coefficient_divisibility_l2564_256460

theorem binomial_coefficient_divisibility (p : ℕ) (hp : Nat.Prime p) :
  (Nat.choose (2 * p) p - 2) % (p^2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_divisibility_l2564_256460


namespace NUMINAMATH_CALUDE_problem_solution_l2564_256463

theorem problem_solution (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : y = x / (2 * x + 1)) :
  (3 * x - 3 * y + x * y) / (4 * x * y) = 7 / 4 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2564_256463


namespace NUMINAMATH_CALUDE_class_average_l2564_256465

/-- Proves that the average score of a class is 45.6 given the specified conditions -/
theorem class_average (total_students : ℕ) (top_scorers : ℕ) (zero_scorers : ℕ) 
  (top_score : ℕ) (rest_average : ℕ) :
  total_students = 25 →
  top_scorers = 3 →
  zero_scorers = 3 →
  top_score = 95 →
  rest_average = 45 →
  let rest_students := total_students - top_scorers - zero_scorers
  let total_score := top_scorers * top_score + zero_scorers * 0 + rest_students * rest_average
  (total_score : ℚ) / total_students = 45.6 := by
  sorry

end NUMINAMATH_CALUDE_class_average_l2564_256465


namespace NUMINAMATH_CALUDE_range_of_a_l2564_256434

/-- The range of a given the conditions in the problem -/
theorem range_of_a (a : ℝ) : 
  (a < 0) → 
  (∀ x : ℝ, (x^2 - 4*a*x + 3*a^2 < 0) → (x^2 + 2*x - 8 > 0)) →
  (∃ x : ℝ, (x^2 - 4*a*x + 3*a^2 < 0) ∧ (x^2 + 2*x - 8 ≤ 0)) →
  a ≤ -4 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l2564_256434


namespace NUMINAMATH_CALUDE_impossible_table_filling_l2564_256496

theorem impossible_table_filling (n : ℕ) (h : n ≥ 3) :
  ¬ ∃ (table : Fin n → Fin (n + 3) → ℕ),
    (∀ i j, table i j ∈ Finset.range (n * (n + 3) + 1)) ∧
    (∀ i j₁ j₂, j₁ ≠ j₂ → table i j₁ ≠ table i j₂) ∧
    (∀ i, ∃ j₁ j₂ j₃, j₁ ≠ j₂ ∧ j₁ ≠ j₃ ∧ j₂ ≠ j₃ ∧
      table i j₁ * table i j₂ = table i j₃) :=
by sorry

end NUMINAMATH_CALUDE_impossible_table_filling_l2564_256496


namespace NUMINAMATH_CALUDE_tangent_line_and_min_value_l2564_256416

/-- The function f(x) = -x^3 + 3x^2 + 9x + a -/
def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + 3*x^2 + 9*x + a

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := -3*x^2 + 6*x + 9

theorem tangent_line_and_min_value (a : ℝ) :
  (∀ x ∈ Set.Icc (-2 : ℝ) 2, f a x ≤ 22) →
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, f a x = 22) →
  (∀ y : ℝ, (9 * 2 - y + 2 = 0) ↔ (y - f (-2) 2 = f' 2 * (y - 2))) ∧
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, ∀ y ∈ Set.Icc (-2 : ℝ) 2, f 0 x ≤ f 0 y ∧ f 0 x = -7) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_and_min_value_l2564_256416


namespace NUMINAMATH_CALUDE_system_solution_l2564_256422

theorem system_solution (x y : ℝ) :
  (1 + x) * (1 + x^2) * (1 + x^4) = 1 + y^7 ∧
  (1 + y) * (1 + y^2) * (1 + y^4) = 1 + x^7 →
  (x = 0 ∧ y = 0) ∨ (x = -1 ∧ y = -1) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l2564_256422


namespace NUMINAMATH_CALUDE_number_equality_l2564_256474

theorem number_equality : ∃ x : ℝ, x / 0.144 = 14.4 / 0.0144 ∧ x = 144 := by
  sorry

end NUMINAMATH_CALUDE_number_equality_l2564_256474


namespace NUMINAMATH_CALUDE_age_of_b_l2564_256484

/-- Given three people A, B, and C, with their ages satisfying certain conditions,
    prove that B's age is 17 years. -/
theorem age_of_b (a b c : ℕ) : 
  (a + b + c) / 3 = 25 →  -- The average age of A, B, and C is 25
  (a + c) / 2 = 29 →      -- The average age of A and C is 29
  b = 17 := by            -- The age of B is 17
sorry

end NUMINAMATH_CALUDE_age_of_b_l2564_256484


namespace NUMINAMATH_CALUDE_inequality_proof_l2564_256444

theorem inequality_proof (a b : ℝ) (h1 : a < b) (h2 : b < 0) :
  (abs a > abs b) ∧ (b / a < a / b) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2564_256444


namespace NUMINAMATH_CALUDE_ab_nonnegative_l2564_256476

theorem ab_nonnegative (a b : ℚ) (ha : |a| = -a) (hb : |b| ≠ b) : a * b ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_ab_nonnegative_l2564_256476


namespace NUMINAMATH_CALUDE_germination_probability_l2564_256429

/-- The germination rate of seeds -/
def germination_rate : ℝ := 0.8

/-- The number of seeds sown -/
def num_seeds : ℕ := 5

/-- The probability of exactly k successes in n trials with probability p -/
def bernoulli_prob (n k : ℕ) (p : ℝ) : ℝ :=
  (Nat.choose n k : ℝ) * p^k * (1 - p)^(n - k)

/-- The probability of at least 4 out of 5 seeds germinating -/
def prob_at_least_4 : ℝ :=
  bernoulli_prob num_seeds 4 germination_rate + bernoulli_prob num_seeds 5 germination_rate

theorem germination_probability :
  prob_at_least_4 = 0.73728 := by sorry

end NUMINAMATH_CALUDE_germination_probability_l2564_256429


namespace NUMINAMATH_CALUDE_polynomial_sum_coefficients_l2564_256406

theorem polynomial_sum_coefficients (d : ℝ) (a b c e : ℤ) : 
  d ≠ 0 → 
  (16 * d + 15 + 18 * d^2 + 5 * d^3) + (4 * d + 3 + 3 * d^3) = a * d + b + c * d^2 + e * d^3 →
  a + b + c + e = 64 := by
sorry

end NUMINAMATH_CALUDE_polynomial_sum_coefficients_l2564_256406


namespace NUMINAMATH_CALUDE_income_expenditure_ratio_l2564_256443

def income : ℕ := 19000
def savings : ℕ := 3800
def expenditure : ℕ := income - savings

theorem income_expenditure_ratio :
  (income : ℚ) / (expenditure : ℚ) = 5 / 4 := by sorry

end NUMINAMATH_CALUDE_income_expenditure_ratio_l2564_256443


namespace NUMINAMATH_CALUDE_smallest_x_value_l2564_256435

theorem smallest_x_value (y : ℕ+) (x : ℕ+) (h : (3 : ℚ) / 4 = y / (178 + x)) : 
  ∃ (x_min : ℕ+), x_min ≤ x ∧ ∃ (y_min : ℕ+), (3 : ℚ) / 4 = y_min / (178 + x_min) ∧ x_min = 2 :=
sorry

end NUMINAMATH_CALUDE_smallest_x_value_l2564_256435


namespace NUMINAMATH_CALUDE_geometric_series_second_term_l2564_256467

theorem geometric_series_second_term 
  (r : ℝ) 
  (S : ℝ) 
  (h1 : r = 1/4) 
  (h2 : S = 16) 
  (h3 : S = a / (1 - r)) 
  (h4 : second_term = a * r) : second_term = 3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_second_term_l2564_256467


namespace NUMINAMATH_CALUDE_shaded_area_is_32_l2564_256470

/-- Represents a rectangle in the grid --/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Represents the grid configuration --/
structure Grid where
  totalWidth : ℕ
  totalHeight : ℕ
  rectangles : List Rectangle

def triangleArea (base height : ℕ) : ℕ :=
  base * height / 2

def rectangleArea (r : Rectangle) : ℕ :=
  r.width * r.height

def totalGridArea (g : Grid) : ℕ :=
  g.rectangles.foldl (fun acc r => acc + rectangleArea r) 0

theorem shaded_area_is_32 (g : Grid) 
    (h1 : g.totalWidth = 16)
    (h2 : g.totalHeight = 8)
    (h3 : g.rectangles = [⟨5, 4⟩, ⟨6, 6⟩, ⟨5, 8⟩])
    (h4 : triangleArea g.totalWidth g.totalHeight = 64) :
    totalGridArea g - triangleArea g.totalWidth g.totalHeight = 32 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_is_32_l2564_256470


namespace NUMINAMATH_CALUDE_right_pyramid_base_side_length_l2564_256473

/-- Given a right pyramid with a square base, if the area of one lateral face is 120 square meters
    and the slant height is 40 meters, then the length of the side of its base is 6 meters. -/
theorem right_pyramid_base_side_length
  (area_lateral_face : ℝ)
  (slant_height : ℝ)
  (h1 : area_lateral_face = 120)
  (h2 : slant_height = 40) :
  (2 * area_lateral_face) / slant_height = 6 :=
by sorry

end NUMINAMATH_CALUDE_right_pyramid_base_side_length_l2564_256473


namespace NUMINAMATH_CALUDE_exists_cycle_not_div_by_three_l2564_256442

/-- A graph is a structure with vertices and edges. -/
structure Graph (V : Type) :=
  (edges : V → V → Prop)

/-- The degree of a vertex in a graph is the number of edges incident to it. -/
def degree {V : Type} (G : Graph V) (v : V) : ℕ := sorry

/-- A path in a graph is a sequence of vertices where each consecutive pair is connected by an edge. -/
def is_path {V : Type} (G : Graph V) (path : List V) : Prop := sorry

/-- A cycle in a graph is a path that starts and ends at the same vertex. -/
def is_cycle {V : Type} (G : Graph V) (cycle : List V) : Prop := sorry

/-- The length of a path or cycle is the number of edges it contains. -/
def length {V : Type} (path : List V) : ℕ := path.length - 1

/-- Main theorem: In a graph where each vertex has degree at least 3, 
    there exists a cycle whose length is not divisible by 3. -/
theorem exists_cycle_not_div_by_three {V : Type} (G : Graph V) :
  (∀ v : V, degree G v ≥ 3) → 
  ∃ cycle : List V, is_cycle G cycle ∧ ¬(length cycle % 3 = 0) := by sorry

end NUMINAMATH_CALUDE_exists_cycle_not_div_by_three_l2564_256442


namespace NUMINAMATH_CALUDE_unique_increasing_function_l2564_256475

theorem unique_increasing_function :
  ∃! f : ℕ → ℕ,
    (∀ n m : ℕ, (2^m + 1) * f n * f (2^m * n) = 2^m * (f n)^2 + (f (2^m * n))^2 + (2^m - 1)^2 * n) ∧
    (∀ a b : ℕ, a < b → f a < f b) ∧
    (∀ n : ℕ, f n = n + 1) :=
by sorry

end NUMINAMATH_CALUDE_unique_increasing_function_l2564_256475


namespace NUMINAMATH_CALUDE_min_q_geq_half_l2564_256433

def q (a : ℕ) : ℚ := ((48 - a) * (47 - a) + (a - 1) * (a - 2)) / (2 * 1653)

theorem min_q_geq_half (n : ℕ) (h : n ≥ 1 ∧ n ≤ 60) :
  (∀ a : ℕ, a ≥ 1 ∧ a ≤ 60 → q a ≥ 1/2 → a ≥ n) →
  q n ≥ 1/2 →
  n = 10 :=
sorry

end NUMINAMATH_CALUDE_min_q_geq_half_l2564_256433


namespace NUMINAMATH_CALUDE_child_running_speed_l2564_256481

/-- Verify the child's running speed on a still sidewalk -/
theorem child_running_speed 
  (distance_with : ℝ)
  (distance_against : ℝ)
  (time_against : ℝ)
  (speed_still : ℝ)
  (h1 : distance_with = 372)
  (h2 : distance_against = 165)
  (h3 : time_against = 3)
  (h4 : speed_still = 74)
  (h5 : ∃ t, t > 0 ∧ (speed_still + (distance_against / time_against - speed_still)) * t = distance_with)
  (h6 : (speed_still - (distance_against / time_against - speed_still)) * time_against = distance_against) :
  speed_still = 74 := by
sorry

end NUMINAMATH_CALUDE_child_running_speed_l2564_256481


namespace NUMINAMATH_CALUDE_exactly_three_heads_probability_l2564_256450

/-- The probability of getting exactly k successes in n trials of a binomial experiment -/
def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  ↑(Nat.choose n k) * p^k * (1 - p)^(n - k)

/-- The number of coin flips -/
def num_flips : ℕ := 8

/-- The number of heads we're interested in -/
def num_heads : ℕ := 3

/-- The probability of getting heads on a single flip -/
def prob_heads : ℚ := 1/3

theorem exactly_three_heads_probability :
  binomial_probability num_flips num_heads prob_heads = 1792/6561 := by
  sorry

end NUMINAMATH_CALUDE_exactly_three_heads_probability_l2564_256450


namespace NUMINAMATH_CALUDE_solve_equation_l2564_256471

theorem solve_equation : ∃ x : ℝ, 0.3 * x + 0.1 * 0.5 = 0.29 ∧ x = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2564_256471


namespace NUMINAMATH_CALUDE_power_of_five_zeros_l2564_256485

theorem power_of_five_zeros (n : ℕ) : n = 1968 → ∃ k : ℕ, 5^n = 10^1968 * k ∧ k % 10 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_power_of_five_zeros_l2564_256485


namespace NUMINAMATH_CALUDE_min_value_of_complex_expression_l2564_256458

open Complex

theorem min_value_of_complex_expression (z : ℂ) (h : abs (z - (3 - 3*I)) = 3) :
  ∃ (min : ℝ), min = 100 ∧ ∀ (w : ℂ), abs (w - (3 - 3*I)) = 3 → 
    abs (w - (2 + 2*I))^2 + abs (w - (6 - 6*I))^2 ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_complex_expression_l2564_256458


namespace NUMINAMATH_CALUDE_cube_face_perimeter_l2564_256482

/-- The sum of the lengths of sides of one face of a cube with side length 9 cm is 36 cm -/
theorem cube_face_perimeter (cube_side_length : ℝ) (h : cube_side_length = 9) : 
  4 * cube_side_length = 36 := by
  sorry

end NUMINAMATH_CALUDE_cube_face_perimeter_l2564_256482


namespace NUMINAMATH_CALUDE_arithmetic_sequence_product_l2564_256478

-- Define the arithmetic sequence
def arithmetic_sequence (b : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n, b (n + 1) = b n + d

-- Define the increasing property
def increasing_sequence (b : ℕ → ℤ) : Prop :=
  ∀ n m, n < m → b n < b m

theorem arithmetic_sequence_product (b : ℕ → ℤ) (d : ℤ) :
  arithmetic_sequence b d →
  increasing_sequence b →
  b 3 * b 4 = 21 →
  b 2 * b 5 = -11 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_product_l2564_256478


namespace NUMINAMATH_CALUDE_distance_traveled_by_slower_person_l2564_256426

/-- The distance traveled by the slower person when two people walk towards each other -/
theorem distance_traveled_by_slower_person
  (total_distance : ℝ)
  (speed_1 : ℝ)
  (speed_2 : ℝ)
  (h1 : total_distance = 50)
  (h2 : speed_1 = 4)
  (h3 : speed_2 = 6)
  (h4 : speed_1 < speed_2) :
  speed_1 * (total_distance / (speed_1 + speed_2)) = 20 :=
by sorry

end NUMINAMATH_CALUDE_distance_traveled_by_slower_person_l2564_256426


namespace NUMINAMATH_CALUDE_symmetry_theorem_l2564_256438

-- Define the points and lines
def P : ℝ × ℝ := (1, 2)
def Q : ℝ × ℝ := (3, 0)
def l1 (x y : ℝ) : Prop := x - y - 1 = 0
def l2 (x y : ℝ) : Prop := x + 3*y - 1 = 0
def l3 (x y : ℝ) : Prop := x - y + 1 = 0

-- Define symmetry for points with respect to a line
def symmetric_point (A B : ℝ × ℝ) (l : ℝ → ℝ → Prop) : Prop :=
  let M := ((A.1 + B.1)/2, (A.2 + B.2)/2)
  l M.1 M.2 ∧ (B.1 - A.1) * (B.1 - A.1) + (B.2 - A.2) * (B.2 - A.2) = 
  4 * ((M.1 - A.1) * (M.1 - A.1) + (M.2 - A.2) * (M.2 - A.2))

-- Define symmetry for lines with respect to another line
def symmetric_line (l1 l2 l3 : ℝ → ℝ → Prop) : Prop :=
  ∀ x y : ℝ, l1 x y → ∃ x' y' : ℝ, l2 x' y' ∧ 
  symmetric_point (x, y) (x', y') l3

-- State the theorem
theorem symmetry_theorem :
  symmetric_point P Q l1 ∧
  symmetric_line l2 (fun x y => 3*x + y + 1 = 0) l3 :=
sorry

end NUMINAMATH_CALUDE_symmetry_theorem_l2564_256438


namespace NUMINAMATH_CALUDE_cube_painting_probability_l2564_256449

/-- The number of faces on a cube -/
def num_faces : ℕ := 6

/-- The number of color choices for each face -/
def num_colors : ℕ := 2

/-- The total number of ways to paint a single cube -/
def total_paint_ways : ℕ := num_colors ^ num_faces

/-- The total number of ways to paint two cubes -/
def total_two_cube_ways : ℕ := total_paint_ways ^ 2

/-- The number of ways two cubes can be painted to look identical after rotation -/
def identical_after_rotation : ℕ := 363

/-- The probability that two independently painted cubes are identical after rotation -/
def prob_identical_after_rotation : ℚ := identical_after_rotation / total_two_cube_ways

theorem cube_painting_probability :
  prob_identical_after_rotation = 363 / 4096 := by sorry

end NUMINAMATH_CALUDE_cube_painting_probability_l2564_256449
