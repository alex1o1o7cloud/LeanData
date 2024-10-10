import Mathlib

namespace sum_of_x_and_y_equals_two_l3706_370620

theorem sum_of_x_and_y_equals_two (x y : ℝ) (h : x^2 + y^2 = 8*x - 4*y - 20) : x + y = 2 := by
  sorry

end sum_of_x_and_y_equals_two_l3706_370620


namespace sequence_limit_property_l3706_370627

theorem sequence_limit_property (a : ℕ → ℝ) :
  (∀ ε > 0, ∃ N, ∀ n ≥ N, |a (n + 2) - a n| < ε) →
  (∀ ε > 0, ∃ N, ∀ n ≥ N, |((a (n + 1) - a n) : ℝ) / n| < ε) :=
by sorry

end sequence_limit_property_l3706_370627


namespace square_odd_implies_odd_l3706_370632

theorem square_odd_implies_odd (n : ℤ) : Odd (n^2) → Odd n := by
  sorry

end square_odd_implies_odd_l3706_370632


namespace smallest_n_for_equations_l3706_370633

theorem smallest_n_for_equations :
  (∃ (n : ℕ), 
    (∀ (m : ℕ), m < n → ¬∃ (x y : ℕ), x * (x + m) = y^2) ∧
    (∃ (x y : ℕ), x * (x + n) = y^2) ∧
    n = 3) ∧
  (∃ (n : ℕ),
    (∀ (m : ℕ), m < n → ¬∃ (x y : ℕ), x * (x + m) = y^3) ∧
    (∃ (x y : ℕ), x * (x + n) = y^3) ∧
    n = 2) :=
by sorry

end smallest_n_for_equations_l3706_370633


namespace expression_evaluation_l3706_370610

theorem expression_evaluation (a : ℝ) 
  (h1 : a ≥ 0) 
  (h2 : a ≠ 1) 
  (h3 : a ≠ 1 + Real.sqrt 2) 
  (h4 : a ≠ 1 - Real.sqrt 2) :
  (1 + 2 * a^(1/4) - a^(1/2)) / (1 - a + 4 * a^(3/4) - 4 * a^(1/2)) +
  (a^(1/4) - 2) / ((a^(1/4) - 1)^2) = 1 / (a^(1/4) - 1) := by
  sorry

end expression_evaluation_l3706_370610


namespace min_additional_squares_for_symmetry_l3706_370699

/-- Represents a square grid with shaded squares -/
structure SquareGrid :=
  (size : ℕ)
  (shaded : Set (ℕ × ℕ))

/-- Checks if a SquareGrid has at least one line of symmetry -/
def has_line_symmetry (grid : SquareGrid) : Prop :=
  sorry

/-- Checks if a SquareGrid has rotational symmetry of order 2 -/
def has_rotational_symmetry_order_2 (grid : SquareGrid) : Prop :=
  sorry

/-- Counts the number of additional squares shaded -/
def count_additional_shaded (initial grid : SquareGrid) : ℕ :=
  sorry

/-- The main theorem stating the minimum number of additional squares to be shaded -/
theorem min_additional_squares_for_symmetry (initial : SquareGrid) :
  ∃ (final : SquareGrid),
    (has_line_symmetry final ∧ has_rotational_symmetry_order_2 final) ∧
    (count_additional_shaded initial final = 3) ∧
    (∀ (other : SquareGrid),
      (has_line_symmetry other ∧ has_rotational_symmetry_order_2 other) →
      count_additional_shaded initial other ≥ 3) :=
  sorry

end min_additional_squares_for_symmetry_l3706_370699


namespace sharmila_hourly_wage_l3706_370636

/-- Sharmila's work schedule and earnings -/
structure WorkSchedule where
  monday_hours : ℕ
  tuesday_hours : ℕ
  wednesday_hours : ℕ
  thursday_hours : ℕ
  friday_hours : ℕ
  weekly_earnings : ℕ

/-- Calculate the total hours worked in a week -/
def total_hours (schedule : WorkSchedule) : ℕ :=
  schedule.monday_hours + schedule.tuesday_hours + schedule.wednesday_hours +
  schedule.thursday_hours + schedule.friday_hours

/-- Calculate the hourly wage -/
def hourly_wage (schedule : WorkSchedule) : ℚ :=
  schedule.weekly_earnings / (total_hours schedule)

/-- Sharmila's actual work schedule -/
def sharmila_schedule : WorkSchedule :=
  { monday_hours := 10
  , tuesday_hours := 8
  , wednesday_hours := 10
  , thursday_hours := 8
  , friday_hours := 10
  , weekly_earnings := 460 }

theorem sharmila_hourly_wage :
  hourly_wage sharmila_schedule = 10 := by
  sorry

end sharmila_hourly_wage_l3706_370636


namespace smallest_five_digit_congruent_to_3_mod_17_l3706_370611

theorem smallest_five_digit_congruent_to_3_mod_17 : ∃ (n : ℕ), 
  (n ≥ 10000 ∧ n < 100000) ∧  -- five-digit positive integer
  (n % 17 = 3) ∧              -- congruent to 3 modulo 17
  (∀ m : ℕ, (m ≥ 10000 ∧ m < 100000) ∧ (m % 17 = 3) → m ≥ n) ∧ 
  n = 10012 :=
by sorry

end smallest_five_digit_congruent_to_3_mod_17_l3706_370611


namespace isosceles_triangle_base_length_l3706_370692

/-- An isosceles triangle with congruent sides of 7 cm and perimeter of 21 cm has a base of 7 cm. -/
theorem isosceles_triangle_base_length : 
  ∀ (base congruent_side : ℝ),
  congruent_side = 7 →
  base + 2 * congruent_side = 21 →
  base = 7 := by
sorry

end isosceles_triangle_base_length_l3706_370692


namespace infinite_solutions_exist_l3706_370674

theorem infinite_solutions_exist : 
  ∀ n : ℕ, n > 0 → 
  ∃ x y : ℕ, x > 0 ∧ y > 0 ∧ 
  x = (1 + 1 / n) * y ∧
  (3 * x^3 + x * y^2) * (x^2 * y + 3 * y^3) = (x - y)^7 :=
by sorry

end infinite_solutions_exist_l3706_370674


namespace equation_solution_exists_l3706_370656

theorem equation_solution_exists : ∃ c : ℝ, 
  Real.sqrt (4 + Real.sqrt (12 + 6 * c)) + Real.sqrt (6 + Real.sqrt (3 + c)) = 4 + 2 * Real.sqrt 3 := by
  sorry

end equation_solution_exists_l3706_370656


namespace distances_product_bound_l3706_370651

/-- Given an equilateral triangle with side length 1 and a point P inside it,
    the distances from P to the three sides satisfy 0 < ab + bc + ca ≤ 1/4 -/
theorem distances_product_bound (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 → 
  a + b + c = Real.sqrt 3 / 2 → 
  0 < a * b + b * c + c * a ∧ a * b + b * c + c * a ≤ 1 / 4 := by
  sorry


end distances_product_bound_l3706_370651


namespace half_angle_quadrant_l3706_370675

-- Define what it means for an angle to be in the third quadrant
def in_third_quadrant (α : Real) : Prop :=
  ∃ k : Int, k * 2 * Real.pi + Real.pi < α ∧ α < k * 2 * Real.pi + 3 * Real.pi / 2

-- Define what it means for an angle to be in the second or fourth quadrant
def in_second_or_fourth_quadrant (α : Real) : Prop :=
  ∃ k : Int, (k * Real.pi + Real.pi / 2 < α ∧ α < k * Real.pi + Real.pi) ∨
             (k * Real.pi + 3 * Real.pi / 2 < α ∧ α < (k + 1) * Real.pi)

-- State the theorem
theorem half_angle_quadrant (α : Real) :
  in_third_quadrant α → in_second_or_fourth_quadrant (α / 2) := by
  sorry

end half_angle_quadrant_l3706_370675


namespace reciprocal_greater_than_one_l3706_370609

theorem reciprocal_greater_than_one (x : ℝ) : 
  (x ≠ 0 ∧ (1 / x) > 1) ↔ (0 < x ∧ x < 1) :=
by sorry

end reciprocal_greater_than_one_l3706_370609


namespace discounted_price_theorem_l3706_370642

/-- The original price of an article before discounts -/
def original_price : ℝ := 150

/-- The first discount rate -/
def discount1 : ℝ := 0.1

/-- The second discount rate -/
def discount2 : ℝ := 0.2

/-- The final sale price after discounts -/
def final_price : ℝ := 108

/-- Theorem stating that the original price results in the final price after discounts -/
theorem discounted_price_theorem :
  final_price = original_price * (1 - discount1) * (1 - discount2) := by
  sorry

#check discounted_price_theorem

end discounted_price_theorem_l3706_370642


namespace rice_distribution_l3706_370634

theorem rice_distribution (total_weight : ℚ) (num_containers : ℕ) (ounces_per_pound : ℕ) : 
  total_weight = 29 / 4 →
  num_containers = 4 →
  ounces_per_pound = 16 →
  (total_weight * ounces_per_pound / num_containers : ℚ) = 29 := by
  sorry

end rice_distribution_l3706_370634


namespace pentagon_area_sum_l3706_370685

/-- A pentagon formed by 11 segments of length 2 -/
structure Pentagon where
  /-- The number of segments forming the pentagon -/
  num_segments : ℕ
  /-- The length of each segment -/
  segment_length : ℝ
  /-- The area of the pentagon -/
  area : ℝ
  /-- The first positive integer in the area expression -/
  m : ℕ
  /-- The second positive integer in the area expression -/
  n : ℕ
  /-- Condition: The number of segments is 11 -/
  h_num_segments : num_segments = 11
  /-- Condition: The length of each segment is 2 -/
  h_segment_length : segment_length = 2
  /-- Condition: The area is expressed as √m + √n -/
  h_area : area = Real.sqrt m + Real.sqrt n
  /-- Condition: m is positive -/
  h_m_pos : m > 0
  /-- Condition: n is positive -/
  h_n_pos : n > 0

/-- Theorem: For the given pentagon, m + n = 23 -/
theorem pentagon_area_sum (p : Pentagon) : p.m + p.n = 23 := by
  sorry

end pentagon_area_sum_l3706_370685


namespace sets_and_range_theorem_l3706_370696

-- Define the sets A and B
def A : Set ℝ := {x | -x^2 + 3*x + 10 ≥ 0}
def B (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2*m - 1}

-- State the theorem
theorem sets_and_range_theorem (m : ℝ) (h : B m ⊆ A) : 
  A = {x | -2 ≤ x ∧ x ≤ 5} ∧ m ≤ 3 := by
  sorry

end sets_and_range_theorem_l3706_370696


namespace fraction_sum_to_ratio_proof_l3706_370691

theorem fraction_sum_to_ratio_proof (x y : ℝ) 
  (h1 : y > x) (h2 : x > 0) (h3 : x / y + y / x = 4) : 
  (x + y) / (x - y) = Real.sqrt 3 := by
  sorry

end fraction_sum_to_ratio_proof_l3706_370691


namespace photocopy_discount_is_25_percent_l3706_370659

/-- The discount percentage for bulk photocopy orders -/
def discount_percentage (cost_per_copy : ℚ) (copies_for_discount : ℕ) 
  (steve_copies : ℕ) (dinley_copies : ℕ) (individual_savings : ℚ) : ℚ :=
  let total_copies := steve_copies + dinley_copies
  let total_cost_without_discount := cost_per_copy * total_copies
  let total_savings := individual_savings * 2
  let total_cost_with_discount := total_cost_without_discount - total_savings
  (total_cost_without_discount - total_cost_with_discount) / total_cost_without_discount * 100

theorem photocopy_discount_is_25_percent :
  discount_percentage 0.02 100 80 80 0.40 = 25 := by
  sorry

end photocopy_discount_is_25_percent_l3706_370659


namespace a_1000_equals_divisors_of_1000_l3706_370607

/-- A sequence of real numbers satisfying the given power series equality -/
def PowerSeriesSequence (a : ℕ → ℝ) : Prop :=
  ∀ x : ℝ, abs x < 1 →
    (∑' n : ℕ, x^n / (1 - x^n)) = ∑' i : ℕ, a i * x^i

/-- The number of divisors of a natural number -/
def numberOfDivisors (n : ℕ) : ℕ :=
  (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

theorem a_1000_equals_divisors_of_1000 (a : ℕ → ℝ) (h : PowerSeriesSequence a) :
    a 1000 = numberOfDivisors 1000 := by
  sorry

end a_1000_equals_divisors_of_1000_l3706_370607


namespace union_of_M_and_N_l3706_370644

def M : Set ℕ := {0, 1, 3}
def N : Set ℕ := {x | ∃ a ∈ M, x = 3 * a}

theorem union_of_M_and_N : M ∪ N = {0, 1, 3, 9} := by sorry

end union_of_M_and_N_l3706_370644


namespace inequality_holds_l3706_370616

theorem inequality_holds (φ : Real) (h : φ > 0 ∧ φ < Real.pi / 2) :
  Real.sin (Real.cos φ) < Real.cos φ ∧ Real.cos φ < Real.cos (Real.sin φ) := by
  sorry

end inequality_holds_l3706_370616


namespace one_cow_drinking_time_l3706_370615

/-- Represents the drinking rate of cows and the spring inflow rate -/
structure PondSystem where
  /-- Amount of water one cow drinks per day -/
  cow_drink_rate : ℝ
  /-- Amount of water springs add to the pond per day -/
  spring_rate : ℝ
  /-- Total volume of the pond -/
  pond_volume : ℝ

/-- Given the conditions, proves that one cow will take 75 days to drink the pond -/
theorem one_cow_drinking_time (sys : PondSystem)
  (h1 : sys.pond_volume + 3 * sys.spring_rate = 3 * 17 * sys.cow_drink_rate)
  (h2 : sys.pond_volume + 30 * sys.spring_rate = 30 * 2 * sys.cow_drink_rate) :
  sys.pond_volume + 75 * sys.spring_rate = 75 * sys.cow_drink_rate :=
by sorry


end one_cow_drinking_time_l3706_370615


namespace amy_tickets_l3706_370612

/-- The number of tickets Amy started with -/
def initial_tickets : ℕ := 33

/-- The number of tickets Amy bought -/
def bought_tickets : ℕ := 21

/-- The total number of tickets Amy had -/
def total_tickets : ℕ := 54

theorem amy_tickets : initial_tickets + bought_tickets = total_tickets := by
  sorry

end amy_tickets_l3706_370612


namespace rower_downstream_speed_l3706_370676

/-- Calculates the downstream speed of a rower given their upstream and still water speeds. -/
def downstream_speed (upstream_speed still_water_speed : ℝ) : ℝ :=
  2 * still_water_speed - upstream_speed

/-- Theorem stating that given a man's upstream speed of 25 kmph and still water speed of 30 kmph,
    his downstream speed is 35 kmph. -/
theorem rower_downstream_speed :
  downstream_speed 25 30 = 35 := by
  sorry

end rower_downstream_speed_l3706_370676


namespace equal_cost_layover_l3706_370650

/-- Represents a city in the country -/
structure City where
  id : Nat

/-- Represents an airline operating in the country -/
structure Airline where
  id : Nat

/-- Represents the transportation network of the country -/
structure CountryNetwork where
  cities : Finset City
  airlines : Finset Airline
  connections : City → City → Finset Airline
  cost : City → City → ℚ

/-- The conditions of the problem -/
def ProblemConditions (network : CountryNetwork) : Prop :=
  (network.cities.card = 100) ∧
  (network.airlines.card = 146) ∧
  (∀ c1 c2 : City, c1 ∈ network.cities → c2 ∈ network.cities → c1 ≠ c2 → 
    network.connections c1 c2 ≠ ∅) ∧
  (∀ c1 c2 : City, c1 ∈ network.cities → c2 ∈ network.cities → c1 ≠ c2 → 
    network.cost c1 c2 = 1 / (network.connections c1 c2).card) ∧
  (∀ c1 c2 c3 : City, c1 ∈ network.cities → c2 ∈ network.cities → c3 ∈ network.cities → 
    c1 ≠ c2 → c2 ≠ c3 → c1 ≠ c3 → 
    network.cost c1 c2 + network.cost c2 c3 ≥ network.cost c1 c3)

/-- The theorem to be proved -/
theorem equal_cost_layover (network : CountryNetwork) 
  (h : ProblemConditions network) : 
  ∃ c1 c2 c3 : City, c1 ∈ network.cities ∧ c2 ∈ network.cities ∧ c3 ∈ network.cities ∧
  c1 ≠ c2 ∧ c2 ≠ c3 ∧ c1 ≠ c3 ∧
  network.cost c1 c2 = network.cost c2 c3 :=
sorry

end equal_cost_layover_l3706_370650


namespace range_of_m_l3706_370662

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, |x + 1| + |x - 3| ≥ |m - 1|) → m ∈ Set.Icc (-3) 5 := by
  sorry

end range_of_m_l3706_370662


namespace selection_with_at_least_one_girl_l3706_370652

def total_students : ℕ := 6
def boys : ℕ := 4
def girls : ℕ := 2
def students_to_select : ℕ := 4

theorem selection_with_at_least_one_girl :
  (Nat.choose total_students students_to_select) - (Nat.choose boys students_to_select) = 14 :=
by sorry

end selection_with_at_least_one_girl_l3706_370652


namespace thomas_final_amount_l3706_370678

-- Define the initial amounts
def michael_initial : ℚ := 42
def thomas_initial : ℚ := 17

-- Define the percentages
def michael_give_percent : ℚ := 35 / 100
def thomas_book_percent : ℚ := 25 / 100

-- Define the candy expense
def candy_expense : ℚ := 5

-- Theorem statement
theorem thomas_final_amount :
  let michael_give := michael_initial * michael_give_percent
  let thomas_after_michael := thomas_initial + michael_give
  let thomas_after_candy := thomas_after_michael - candy_expense
  let book_expense := thomas_after_candy * thomas_book_percent
  let thomas_final := thomas_after_candy - book_expense
  thomas_final = 20.02 := by sorry

end thomas_final_amount_l3706_370678


namespace customer_equation_l3706_370671

theorem customer_equation (X Y Z : ℕ) 
  (h1 : X - Y = 10)
  (h2 : (X - Y) - Z = 4) : 
  X - (X - 10) - 6 = 4 := by
  sorry

end customer_equation_l3706_370671


namespace xyz_inequality_l3706_370626

theorem xyz_inequality (x y z : ℝ) (n : ℕ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0) (h_sum : x + y + z = 1) (h_pos_n : n > 0) :
  x^n + y^n + z^n ≥ 1 / 3^(n-1) := by
  sorry

end xyz_inequality_l3706_370626


namespace complex_equation_product_l3706_370698

/-- Given (1+3i)(a+bi) = 10i, where i is the imaginary unit and a, b ∈ ℝ, prove that ab = 3 -/
theorem complex_equation_product (a b : ℝ) : 
  (Complex.I : ℂ) ^ 2 = -1 →
  (1 + 3 * Complex.I) * (a + b * Complex.I) = 10 * Complex.I →
  a * b = 3 := by sorry

end complex_equation_product_l3706_370698


namespace book_distribution_l3706_370660

theorem book_distribution (n : ℕ) (k : ℕ) : 
  n = 5 → k = 4 → (k^n - k * (k-1)^n + (k.choose 2) * (k-2)^n) = 292 := by
  sorry

end book_distribution_l3706_370660


namespace mortgage_loan_amount_l3706_370638

/-- The mortgage loan problem -/
theorem mortgage_loan_amount 
  (initial_payment : ℝ) 
  (loan_percentage : ℝ) 
  (h1 : initial_payment = 2000000)
  (h2 : loan_percentage = 0.75) : 
  ∃ (total_cost : ℝ), 
    total_cost = initial_payment + loan_percentage * total_cost ∧ 
    loan_percentage * total_cost = 6000000 :=
by sorry

end mortgage_loan_amount_l3706_370638


namespace log_equality_implies_relation_l3706_370625

theorem log_equality_implies_relation (p q r : ℝ) (hp : p > 0) (hq : q > 0) (hr : r > 0) :
  Real.log p + Real.log q + Real.log r = Real.log (p * q * r + p + q) → p = -q := by
  sorry

end log_equality_implies_relation_l3706_370625


namespace oranges_taken_l3706_370683

theorem oranges_taken (initial : ℕ) (remaining : ℕ) (taken : ℕ) : 
  initial = 70 → remaining = 51 → taken = initial - remaining → taken = 19 := by
sorry

end oranges_taken_l3706_370683


namespace quadratic_roots_imply_a_range_l3706_370619

theorem quadratic_roots_imply_a_range (a : ℝ) :
  (∃ x y : ℝ, x > 0 ∧ y < 0 ∧ 
   x^2 + 2*(a-1)*x + 2*a + 6 = 0 ∧
   y^2 + 2*(a-1)*y + 2*a + 6 = 0) →
  a < -3 := by
sorry

end quadratic_roots_imply_a_range_l3706_370619


namespace binomial_expansion_coefficient_l3706_370614

theorem binomial_expansion_coefficient (x : ℝ) : 
  let expansion := (x - 2 / Real.sqrt x) ^ 5
  ∃ c : ℝ, c = 40 ∧ 
    ∃ other_terms : ℝ → ℝ, 
      expansion = c * x^2 + other_terms x :=
by sorry

end binomial_expansion_coefficient_l3706_370614


namespace same_color_probability_l3706_370630

def total_balls : ℕ := 8 + 5 + 3

def prob_blue : ℚ := 8 / total_balls
def prob_green : ℚ := 5 / total_balls
def prob_red : ℚ := 3 / total_balls

theorem same_color_probability : 
  prob_blue * prob_blue + prob_green * prob_green + prob_red * prob_red = 49 / 128 := by
  sorry

end same_color_probability_l3706_370630


namespace group_frequency_problem_l3706_370672

/-- A problem about frequency and relative frequency in a grouped sample -/
theorem group_frequency_problem (total_sample : ℕ) (num_groups : ℕ) 
  (group_frequencies : Fin 8 → ℕ) :
  total_sample = 100 →
  num_groups = 8 →
  group_frequencies 0 = 10 →
  group_frequencies 1 = 13 →
  group_frequencies 3 = 14 →
  group_frequencies 4 = 15 →
  group_frequencies 5 = 13 →
  group_frequencies 6 = 12 →
  group_frequencies 7 = 9 →
  group_frequencies 2 = 14 ∧ 
  (group_frequencies 2 : ℚ) / total_sample = 14 / 100 :=
by sorry

end group_frequency_problem_l3706_370672


namespace completing_square_quadratic_l3706_370684

theorem completing_square_quadratic (x : ℝ) :
  x^2 + 4*x - 1 = 0 ↔ (x + 2)^2 = 5 :=
by sorry

end completing_square_quadratic_l3706_370684


namespace min_value_of_f_l3706_370682

/-- The function f(x) = 3x^2 - 12x + 7 + 749 -/
def f (x : ℝ) : ℝ := 3 * x^2 - 12 * x + 7 + 749

theorem min_value_of_f :
  ∃ (m : ℝ), m = 744 ∧ ∀ (x : ℝ), f x ≥ m :=
by sorry

end min_value_of_f_l3706_370682


namespace consumption_increase_l3706_370689

theorem consumption_increase (original_tax original_consumption : ℝ) 
  (h_tax_positive : original_tax > 0) 
  (h_consumption_positive : original_consumption > 0) : 
  ∃ (increase_percentage : ℝ),
    (original_tax * 0.8 * (original_consumption * (1 + increase_percentage / 100)) = 
     original_tax * original_consumption * 0.96) ∧
    increase_percentage = 20 := by
  sorry

end consumption_increase_l3706_370689


namespace raritet_encounters_l3706_370670

/-- Represents the number of days it takes for a ferry to travel between Dzerzhinsk and Lvov --/
def travel_time : ℕ := 8

/-- Represents the number of ferries departing from Dzerzhinsk during Raritet's journey --/
def ferries_during_journey : ℕ := travel_time

/-- Represents the number of ferries already en route when Raritet departs --/
def ferries_en_route : ℕ := travel_time

/-- Represents the ferry arriving in Lvov when Raritet departs --/
def arriving_ferry : ℕ := 1

/-- Theorem stating the total number of ferries Raritet meets --/
theorem raritet_encounters :
  ferries_during_journey + ferries_en_route + arriving_ferry = 17 :=
sorry

end raritet_encounters_l3706_370670


namespace complement_intersection_A_B_l3706_370641

def I : Set Nat := {1, 2, 3, 4, 5, 6}
def A : Set Nat := {1, 2, 3, 4}
def B : Set Nat := {3, 4, 5, 6}

theorem complement_intersection_A_B :
  (I \ (A ∩ B)) = {1, 2, 5, 6} := by
  sorry

end complement_intersection_A_B_l3706_370641


namespace peanuts_remaining_l3706_370695

theorem peanuts_remaining (initial_peanuts : ℕ) (brock_fraction : ℚ) (bonita_peanuts : ℕ) : 
  initial_peanuts = 148 →
  brock_fraction = 1/4 →
  bonita_peanuts = 29 →
  initial_peanuts - (initial_peanuts * brock_fraction).floor - bonita_peanuts = 82 :=
by
  sorry

end peanuts_remaining_l3706_370695


namespace ellipse_condition_range_l3706_370687

theorem ellipse_condition_range (m a : ℝ) : 
  (a > 0) →
  (m^2 + 12*a^2 < 7*a*m) →
  (∀ x y : ℝ, x^2 / (m - 1) + y^2 / (2 - m) = 1 → 
    ∃ c : ℝ, c > 0 ∧ ∀ p : ℝ × ℝ, p.1 = 0 → 
      (p.2 - c)^2 + p.1^2 = (m - 1)^2 ∨ (p.2 + c)^2 + p.1^2 = (m - 1)^2) →
  (∀ m : ℝ, (m^2 + 12*a^2 < 7*a*m) → 
    (∃ x y : ℝ, x^2 / (m - 1) + y^2 / (2 - m) = 1 ∧
      ∃ c : ℝ, c > 0 ∧ ∀ p : ℝ × ℝ, p.1 = 0 → 
        (p.2 - c)^2 + p.1^2 = (m - 1)^2 ∨ (p.2 + c)^2 + p.1^2 = (m - 1)^2)) →
  (∃ m : ℝ, (m^2 + 12*a^2 < 7*a*m) ∧ 
    ¬(∃ x y : ℝ, x^2 / (m - 1) + y^2 / (2 - m) = 1 ∧
      ∃ c : ℝ, c > 0 ∧ ∀ p : ℝ × ℝ, p.1 = 0 → 
        (p.2 - c)^2 + p.1^2 = (m - 1)^2 ∨ (p.2 + c)^2 + p.1^2 = (m - 1)^2)) →
  a ∈ Set.Icc (1/3 : ℝ) (3/8 : ℝ) := by
sorry

end ellipse_condition_range_l3706_370687


namespace rectangle_area_l3706_370618

/-- A rectangle ABCD is divided into four identical squares and has a perimeter of 160 cm. -/
structure Rectangle :=
  (side : ℝ)
  (perimeter_eq : 10 * side = 160)

/-- The area of the rectangle ABCD is 1024 square centimeters. -/
theorem rectangle_area (rect : Rectangle) : 4 * rect.side^2 = 1024 := by
  sorry

end rectangle_area_l3706_370618


namespace tiffany_cans_l3706_370635

theorem tiffany_cans (monday_bags : ℕ) (next_day_bags : ℕ) 
  (h1 : monday_bags = 8) 
  (h2 : monday_bags = next_day_bags + 1) : 
  next_day_bags = 7 := by
  sorry

end tiffany_cans_l3706_370635


namespace first_class_students_l3706_370601

/-- The number of students in the first class -/
def x : ℕ := 24

/-- The number of students in the second class -/
def second_class_students : ℕ := 50

/-- The average marks of the first class -/
def first_class_avg : ℚ := 40

/-- The average marks of the second class -/
def second_class_avg : ℚ := 60

/-- The average marks of all students combined -/
def total_avg : ℚ := 53513513513513516 / 1000000000000000

theorem first_class_students :
  (x * first_class_avg + second_class_students * second_class_avg) / (x + second_class_students) = total_avg := by
  sorry

end first_class_students_l3706_370601


namespace smallest_period_of_one_minus_cos_2x_l3706_370697

/-- The smallest positive period of y = 1 - cos(2x) is π -/
theorem smallest_period_of_one_minus_cos_2x (x : ℝ) : 
  let f : ℝ → ℝ := λ x => 1 - Real.cos (2 * x)
  ∃ T : ℝ, T > 0 ∧ T = π ∧ ∀ t : ℝ, f (t + T) = f t ∧ 
    ∀ S : ℝ, S > 0 ∧ (∀ t : ℝ, f (t + S) = f t) → T ≤ S :=
by sorry

end smallest_period_of_one_minus_cos_2x_l3706_370697


namespace triangle_properties_l3706_370623

noncomputable section

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
def triangle (a b c : ℝ) := true

theorem triangle_properties (a b c : ℝ) (h : triangle a b c) 
  (h1 : a^2 + 11*b^2 = 2 * Real.sqrt 3 * a * b)
  (h2 : Real.sin c = 2 * Real.sqrt 3 * Real.sin b)
  (h3 : Real.cos b * a * c = Real.tan b) :
  Real.cos b = 1/2 ∧ (1/2 * a * c * Real.sin b = 3/2) := by
  sorry

end triangle_properties_l3706_370623


namespace asterisk_replacement_l3706_370680

theorem asterisk_replacement : ∃! (x : ℝ), x > 0 ∧ (x / 21) * (x / 189) = 1 := by
  sorry

end asterisk_replacement_l3706_370680


namespace train_length_proof_l3706_370688

/-- Given two trains running in opposite directions with the same speed,
    prove that their length is 120 meters. -/
theorem train_length_proof (speed : ℝ) (crossing_time : ℝ) :
  speed = 36 → crossing_time = 12 → 
  ∃ (train_length : ℝ), train_length = 120 :=
by
  sorry

end train_length_proof_l3706_370688


namespace congruence_properties_l3706_370602

theorem congruence_properties (a b c d : ℤ) (h : b ≡ c [ZMOD d]) :
  (a + b ≡ a + c [ZMOD d]) ∧ 
  (a - b ≡ a - c [ZMOD d]) ∧ 
  (a * b ≡ a * c [ZMOD d]) := by
  sorry

end congruence_properties_l3706_370602


namespace polynomial_simplification_l3706_370640

-- Define the left-hand side of the equation
def lhs (p : ℝ) : ℝ := (7*p^5 - 4*p^3 + 8*p^2 - 5*p + 3) + (-p^5 + 3*p^3 - 7*p^2 + 6*p + 2)

-- Define the right-hand side of the equation
def rhs (p : ℝ) : ℝ := 6*p^5 - p^3 + p^2 + p + 5

-- Theorem statement
theorem polynomial_simplification (p : ℝ) : lhs p = rhs p := by
  sorry

end polynomial_simplification_l3706_370640


namespace village_population_original_inhabitants_l3706_370668

theorem village_population (final_population : ℕ) : ℕ :=
  let initial_reduction := 0.9
  let secondary_reduction := 0.75
  let total_reduction := initial_reduction * secondary_reduction
  (final_population : ℝ) / total_reduction
    |> round
    |> Int.toNat

/-- The original number of inhabitants in a village, given the final population after two reductions -/
theorem original_inhabitants : village_population 5265 = 7800 := by
  sorry

end village_population_original_inhabitants_l3706_370668


namespace trench_digging_time_l3706_370617

theorem trench_digging_time (a b c d : ℝ) : 
  (a + b + c + d = 1/6) →
  (2*a + (1/2)*b + c + d = 1/6) →
  ((1/2)*a + 2*b + c + d = 1/4) →
  (a + b + c = 1/6) :=
by sorry

end trench_digging_time_l3706_370617


namespace smallest_perfect_square_divisible_by_2_and_5_l3706_370667

theorem smallest_perfect_square_divisible_by_2_and_5 :
  ∀ n : ℕ, n > 0 → (∃ m : ℕ, n = m^2) → n % 2 = 0 → n % 5 = 0 → n ≥ 100 :=
by
  sorry

end smallest_perfect_square_divisible_by_2_and_5_l3706_370667


namespace glove_selection_theorem_l3706_370677

theorem glove_selection_theorem :
  let n : ℕ := 6  -- Total number of glove pairs
  let k : ℕ := 5  -- Number of gloves to select
  let same_pair : ℕ := 2  -- Number of gloves from the same pair

  -- Function to calculate the number of ways to select gloves
  let select_gloves : ℕ :=
    (n.choose 1) *  -- Choose 1 pair for the matching gloves
    ((n - 1).choose (k - same_pair)) *  -- Choose remaining pairs
    (2 ^ (k - same_pair))  -- Select one glove from each remaining pair

  select_gloves = 480 := by sorry

end glove_selection_theorem_l3706_370677


namespace unique_prime_with_prime_neighbors_l3706_370681

theorem unique_prime_with_prime_neighbors : 
  ∃! p : ℕ, Nat.Prime p ∧ Nat.Prime (p^2 - 6) ∧ Nat.Prime (p^2 + 6) :=
by sorry

end unique_prime_with_prime_neighbors_l3706_370681


namespace banana_arrangements_l3706_370663

def word_length : ℕ := 6
def a_count : ℕ := 3
def n_count : ℕ := 2
def b_count : ℕ := 1

theorem banana_arrangements : 
  (word_length.factorial) / (a_count.factorial * n_count.factorial * b_count.factorial) = 60 := by
  sorry

end banana_arrangements_l3706_370663


namespace donuts_left_l3706_370621

def initial_donuts : ℕ := 50
def bill_eats : ℕ := 2
def secretary_takes : ℕ := 4

def remaining_donuts : ℕ := 
  let after_bill := initial_donuts - bill_eats
  let after_secretary := after_bill - secretary_takes
  after_secretary / 2

theorem donuts_left : remaining_donuts = 22 := by sorry

end donuts_left_l3706_370621


namespace perseverance_arrangement_count_l3706_370653

/-- The number of letters in the word "PERSEVERANCE" -/
def total_letters : ℕ := 12

/-- The number of times the letter 'E' appears in "PERSEVERANCE" -/
def e_count : ℕ := 3

/-- The number of times the letter 'R' appears in "PERSEVERANCE" -/
def r_count : ℕ := 2

/-- The number of unique arrangements of the letters in "PERSEVERANCE" -/
def perseverance_arrangements : ℕ := Nat.factorial total_letters / (Nat.factorial e_count * Nat.factorial r_count)

theorem perseverance_arrangement_count : perseverance_arrangements = 39916800 := by
  sorry

end perseverance_arrangement_count_l3706_370653


namespace spadesuit_example_l3706_370613

def spadesuit (a b : ℝ) : ℝ := |a - b|

theorem spadesuit_example : spadesuit 3 (spadesuit 5 8) = 0 := by
  sorry

end spadesuit_example_l3706_370613


namespace boy_running_speed_l3706_370606

/-- The speed of a boy running around a square field -/
theorem boy_running_speed (side_length : Real) (time : Real) : 
  side_length = 60 → time = 72 → (4 * side_length) / time * (3600 / 1000) = 12 := by
  sorry

end boy_running_speed_l3706_370606


namespace trigonometric_identity_l3706_370631

theorem trigonometric_identity (α β : Real) :
  (Real.cos α)^2 + (Real.cos β)^2 - 2 * (Real.cos α) * (Real.cos β) * Real.cos (α + β) =
  (Real.sin α)^2 + (Real.sin β)^2 + 2 * (Real.sin α) * (Real.sin β) * Real.sin (α + β) := by
  sorry

end trigonometric_identity_l3706_370631


namespace no_solution_iff_n_eq_neg_one_l3706_370655

/-- A system of linear equations parameterized by n -/
def LinearSystem (n : ℝ) :=
  ∃ (x y z : ℝ), (n * x + y = 1) ∧ ((1/2) * n * y + z = 1) ∧ (x + (1/2) * n * z = 2)

/-- The theorem stating that the system has no solution if and only if n = -1 -/
theorem no_solution_iff_n_eq_neg_one :
  ∀ n : ℝ, ¬(LinearSystem n) ↔ n = -1 := by sorry

end no_solution_iff_n_eq_neg_one_l3706_370655


namespace derivative_even_implies_a_zero_l3706_370664

/-- Given a real number a and a function f(x) = x³ + ax² + (a-2)x,
    if f'(x) is an even function, then a = 0 -/
theorem derivative_even_implies_a_zero (a : ℝ) :
  let f : ℝ → ℝ := λ x ↦ x^3 + a*x^2 + (a-2)*x
  (∀ x, (deriv f) x = (deriv f) (-x)) →
  a = 0 := by
sorry

end derivative_even_implies_a_zero_l3706_370664


namespace inscribed_circle_area_l3706_370643

/-- The area of a circle inscribed in an equilateral triangle with side length 24 cm is 48π cm². -/
theorem inscribed_circle_area (s : ℝ) (h : s = 24) : 
  let r := s * Real.sqrt 3 / 6
  π * r^2 = 48 * π := by
  sorry

end inscribed_circle_area_l3706_370643


namespace odd_product_units_digit_l3706_370658

def is_odd (n : ℕ) : Prop := n % 2 = 1

def is_between (n a b : ℕ) : Prop := a < n ∧ n < b

def units_digit (n : ℕ) : ℕ := n % 10

theorem odd_product_units_digit :
  ∃ (prod : ℕ),
    (∀ n : ℕ, is_odd n ∧ is_between n 20 130 → n ∣ prod) ∧
    units_digit prod = 5 :=
by sorry

end odd_product_units_digit_l3706_370658


namespace smallest_prime_divisor_of_sum_l3706_370690

theorem smallest_prime_divisor_of_sum : 
  ∃ (n : ℕ), n = 6^15 + 9^11 ∧ (∀ p : ℕ, Prime p → p ∣ n → p ≥ 3) ∧ 3 ∣ n :=
by sorry

end smallest_prime_divisor_of_sum_l3706_370690


namespace max_advancing_teams_l3706_370608

/-- The number of teams in the tournament -/
def num_teams : ℕ := 8

/-- The minimum number of points required to advance -/
def min_points_to_advance : ℕ := 15

/-- The number of points awarded for a win -/
def win_points : ℕ := 3

/-- The number of points awarded for a draw -/
def draw_points : ℕ := 1

/-- The number of points awarded for a loss -/
def loss_points : ℕ := 0

/-- The total number of games played in the tournament -/
def total_games : ℕ := (num_teams * (num_teams - 1)) / 2

/-- The maximum total points possible in the tournament -/
def max_total_points : ℕ := total_games * win_points

/-- The maximum number of teams that can advance to the next round -/
theorem max_advancing_teams :
  ∃ (n : ℕ), n ≤ max_total_points / min_points_to_advance ∧
             n = 5 ∧
             (∀ m : ℕ, m > n → m * min_points_to_advance > max_total_points) :=
by sorry

end max_advancing_teams_l3706_370608


namespace arcsin_arccos_inequality_l3706_370645

theorem arcsin_arccos_inequality (x : ℝ) :
  x ∈ Set.Icc (-1 : ℝ) 1 →
  (Real.arcsin ((5 / (2 * Real.pi)) * Real.arccos x) > Real.arccos ((10 / (3 * Real.pi)) * Real.arcsin x)) ↔
  (x ∈ Set.Icc (Real.cos (2 * Real.pi / 5)) (Real.cos (8 * Real.pi / 25)) ∪
   Set.Ioo (Real.cos (8 * Real.pi / 25)) (Real.cos (Real.pi / 5))) :=
by sorry

end arcsin_arccos_inequality_l3706_370645


namespace sally_pens_ratio_l3706_370665

def sally_pens_problem (initial_pens : ℕ) (num_students : ℕ) (pens_per_student : ℕ) (pens_taken_home : ℕ) : Prop :=
  let pens_distributed := num_students * pens_per_student
  let pens_remaining := initial_pens - pens_distributed
  let pens_in_locker := pens_remaining - pens_taken_home
  pens_in_locker = pens_taken_home

theorem sally_pens_ratio : sally_pens_problem 342 44 7 17 := by
  sorry

end sally_pens_ratio_l3706_370665


namespace no_100_digit_page_numbering_l3706_370646

theorem no_100_digit_page_numbering :
  ¬ ∃ (n : ℕ), n > 0 ∧ (
    let single_digit_sum := min n 9
    let double_digit_sum := if n > 9 then 2 * (n - 9) else 0
    single_digit_sum + double_digit_sum = 100
  ) := by
  sorry

end no_100_digit_page_numbering_l3706_370646


namespace glass_piece_coloring_l3706_370657

/-- Represents the count of glass pieces for each color -/
structure GlassPieces where
  red : ℕ
  yellow : ℕ
  blue : ℕ
  sum_is_2005 : red + yellow + blue = 2005

/-- Represents a single operation on the glass pieces -/
inductive Operation
  | RedYellowToBlue
  | RedBlueToYellow
  | YellowBlueToRed

/-- Applies an operation to the glass pieces -/
def apply_operation (gp : GlassPieces) (op : Operation) : GlassPieces :=
  match op with
  | Operation.RedYellowToBlue => 
      { red := gp.red - 1, yellow := gp.yellow - 1, blue := gp.blue + 2, 
        sum_is_2005 := by sorry }
  | Operation.RedBlueToYellow => 
      { red := gp.red - 1, yellow := gp.yellow + 2, blue := gp.blue - 1, 
        sum_is_2005 := by sorry }
  | Operation.YellowBlueToRed => 
      { red := gp.red + 2, yellow := gp.yellow - 1, blue := gp.blue - 1, 
        sum_is_2005 := by sorry }

/-- A sequence of operations -/
def OperationSequence := List Operation

/-- Applies a sequence of operations to the glass pieces -/
def apply_sequence (gp : GlassPieces) (seq : OperationSequence) : GlassPieces :=
  match seq with
  | [] => gp
  | op :: rest => apply_sequence (apply_operation gp op) rest

/-- Predicate to check if all pieces are the same color -/
def all_same_color (gp : GlassPieces) : Prop :=
  (gp.red = 2005 ∧ gp.yellow = 0 ∧ gp.blue = 0) ∨
  (gp.red = 0 ∧ gp.yellow = 2005 ∧ gp.blue = 0) ∨
  (gp.red = 0 ∧ gp.yellow = 0 ∧ gp.blue = 2005)

theorem glass_piece_coloring
  (gp : GlassPieces) :
  (∃ (seq : OperationSequence), all_same_color (apply_sequence gp seq)) ∧
  (∀ (seq1 seq2 : OperationSequence),
    all_same_color (apply_sequence gp seq1) →
    all_same_color (apply_sequence gp seq2) →
    apply_sequence gp seq1 = apply_sequence gp seq2) := by
  sorry

end glass_piece_coloring_l3706_370657


namespace eric_marbles_l3706_370661

theorem eric_marbles (total : ℕ) (white : ℕ) (blue : ℕ) (green : ℕ) 
  (h1 : total = 20)
  (h2 : white = 12)
  (h3 : blue = 6)
  (h4 : green = total - (white + blue)) :
  green = 2 := by
sorry

end eric_marbles_l3706_370661


namespace ellipse_and_line_intersection_l3706_370694

-- Define the ellipse C
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Define the line L
def line (m : ℝ) (x y : ℝ) : Prop :=
  y = x + m

-- Define the theorem
theorem ellipse_and_line_intersection
  (a b : ℝ)
  (h_positive : a > b ∧ b > 0)
  (h_axis : b = a / 2)
  (h_max_distance : a + (a^2 - b^2).sqrt = 2 + Real.sqrt 3)
  (m : ℝ)
  (h_area : ∃ x₁ y₁ x₂ y₂ : ℝ,
    ellipse a b x₁ y₁ ∧
    ellipse a b x₂ y₂ ∧
    line m x₁ y₁ ∧
    line m x₂ y₂ ∧
    x₁ ≠ x₂ ∧
    abs ((x₂ - x₁) * (y₂ + y₁) / 2) = 1) :
  (a^2 = 4 ∧ b^2 = 1) ∧ m^2 = 5/2 :=
sorry

end ellipse_and_line_intersection_l3706_370694


namespace gcd_special_numbers_l3706_370673

theorem gcd_special_numbers : Nat.gcd 3333333 666666666 = 3 := by
  sorry

end gcd_special_numbers_l3706_370673


namespace student_selection_methods_l3706_370666

theorem student_selection_methods (n : ℕ) (h : n = 5) : 
  (n.choose 2) * ((n - 2).choose 1) * ((n - 3).choose 1) = 60 := by
  sorry

end student_selection_methods_l3706_370666


namespace rectangle_shading_convergence_l3706_370647

theorem rectangle_shading_convergence :
  let initial_shaded : ℚ := 1/2
  let subsequent_shading_ratio : ℚ := 1/16
  let shaded_series : ℕ → ℚ := λ n => initial_shaded * subsequent_shading_ratio^n
  let total_shaded : ℚ := ∑' n, shaded_series n
  total_shaded = 17/30 := by
sorry

end rectangle_shading_convergence_l3706_370647


namespace train_length_l3706_370637

/-- The length of a train that crosses two platforms of different lengths in given times. -/
theorem train_length 
  (platform1_length : ℝ) 
  (platform1_time : ℝ) 
  (platform2_length : ℝ) 
  (platform2_time : ℝ) 
  (h1 : platform1_length = 170)
  (h2 : platform1_time = 15)
  (h3 : platform2_length = 250)
  (h4 : platform2_time = 20) :
  ∃ (train_length : ℝ), 
    (train_length + platform1_length) / platform1_time = 
    (train_length + platform2_length) / platform2_time ∧ 
    train_length = 70 := by
sorry


end train_length_l3706_370637


namespace total_cleanings_is_777_l3706_370679

/-- Calculates the total number of times Michael, Angela, and Lucy clean themselves in 52 weeks --/
def total_cleanings : ℕ :=
  let weeks_in_year : ℕ := 52
  let days_in_week : ℕ := 7
  let month_in_weeks : ℕ := 4

  -- Michael's cleaning schedule
  let michael_baths_per_week : ℕ := 2
  let michael_showers_per_week : ℕ := 1
  let michael_vacation_weeks : ℕ := 3

  -- Angela's cleaning schedule
  let angela_showers_per_day : ℕ := 1
  let angela_vacation_weeks : ℕ := 2

  -- Lucy's regular cleaning schedule
  let lucy_baths_per_week : ℕ := 3
  let lucy_showers_per_week : ℕ := 2

  -- Lucy's modified schedule for one month
  let lucy_modified_baths_per_week : ℕ := 1
  let lucy_modified_showers_per_day : ℕ := 1

  -- Calculate total cleanings
  let michael_total := (michael_baths_per_week + michael_showers_per_week) * weeks_in_year - 
                       (michael_baths_per_week + michael_showers_per_week) * michael_vacation_weeks

  let angela_total := angela_showers_per_day * days_in_week * weeks_in_year - 
                      angela_showers_per_day * days_in_week * angela_vacation_weeks

  let lucy_regular_weeks := weeks_in_year - month_in_weeks
  let lucy_total := (lucy_baths_per_week + lucy_showers_per_week) * lucy_regular_weeks +
                    (lucy_modified_baths_per_week + lucy_modified_showers_per_day * days_in_week) * month_in_weeks

  michael_total + angela_total + lucy_total

theorem total_cleanings_is_777 : total_cleanings = 777 := by
  sorry

end total_cleanings_is_777_l3706_370679


namespace function_equality_l3706_370654

theorem function_equality (f : ℝ → ℝ) 
  (h1 : ∀ x, f x ≤ x) 
  (h2 : ∀ x y, f (x + y) ≤ f x + f y) : 
  ∀ x, f x = x := by
  sorry

end function_equality_l3706_370654


namespace min_perimeter_isosceles_triangles_l3706_370622

/-- Represents an isosceles triangle with integer side lengths -/
structure IsoscelesTriangle where
  leg : ℕ  -- length of equal sides
  base : ℕ  -- length of the base
  is_isosceles : leg > base / 2

/-- Calculates the perimeter of an isosceles triangle -/
def perimeter (t : IsoscelesTriangle) : ℕ := 2 * t.leg + t.base

/-- Calculates the area of an isosceles triangle -/
noncomputable def area (t : IsoscelesTriangle) : ℝ :=
  (t.base / 2 : ℝ) * Real.sqrt ((t.leg : ℝ)^2 - (t.base / 2 : ℝ)^2)

/-- Theorem: The minimum possible common perimeter of two noncongruent
    integer-sided isosceles triangles with the same area and a base ratio of 5:4 is 840 -/
theorem min_perimeter_isosceles_triangles :
  ∃ (t1 t2 : IsoscelesTriangle),
    t1 ≠ t2 ∧
    area t1 = area t2 ∧
    5 * t1.base = 4 * t2.base ∧
    perimeter t1 = perimeter t2 ∧
    perimeter t1 = 840 ∧
    (∀ (s1 s2 : IsoscelesTriangle),
      s1 ≠ s2 →
      area s1 = area s2 →
      5 * s1.base = 4 * s2.base →
      perimeter s1 = perimeter s2 →
      perimeter s1 ≥ 840) := by
  sorry

end min_perimeter_isosceles_triangles_l3706_370622


namespace min_sum_given_product_l3706_370669

theorem min_sum_given_product (a b : ℝ) : 
  a > 0 → b > 0 → a * b = a + b + 1 → a + b ≥ 2 + 2 * Real.sqrt 2 := by
  sorry

end min_sum_given_product_l3706_370669


namespace rectangular_parallelepiped_volume_l3706_370648

theorem rectangular_parallelepiped_volume 
  (x y z : ℝ) 
  (h1 : (x^2 + y^2) * z^2 = 13) 
  (h2 : (y^2 + z^2) * x^2 = 40) 
  (h3 : (x^2 + z^2) * y^2 = 45) : 
  x * y * z = 6 := by
sorry

end rectangular_parallelepiped_volume_l3706_370648


namespace factor_proof_l3706_370629

theorem factor_proof :
  (∃ n : ℤ, 24 = 4 * n) ∧ (∃ m : ℤ, 180 = 9 * m) := by
  sorry

end factor_proof_l3706_370629


namespace geometric_series_sum_specific_geometric_series_sum_l3706_370605

def geometric_series (a : ℝ) (r : ℝ) : ℕ → ℝ := fun n => a * r^n

theorem geometric_series_sum (a : ℝ) (r : ℝ) (h : |r| < 1) :
  ∑' n, geometric_series a r n = a / (1 - r) :=
sorry

theorem specific_geometric_series_sum :
  ∑' n, geometric_series 1 (1/3) n = 3/2 :=
sorry

end geometric_series_sum_specific_geometric_series_sum_l3706_370605


namespace advance_agency_fees_calculation_l3706_370639

/-- Proof of advance agency fees calculation -/
theorem advance_agency_fees_calculation 
  (C : ℕ) -- Commission
  (I : ℕ) -- Incentive
  (G : ℕ) -- Amount given to John
  (h1 : C = 25000)
  (h2 : I = 1780)
  (h3 : G = 18500)
  : C + I - G = 8280 := by
  sorry

end advance_agency_fees_calculation_l3706_370639


namespace labourer_monthly_income_labourer_monthly_income_proof_l3706_370686

/-- Proves that the monthly income of a labourer is 78 given specific expenditure patterns --/
theorem labourer_monthly_income : ℝ → Prop :=
  fun monthly_income =>
    let first_period_months : ℕ := 6
    let second_period_months : ℕ := 4
    let first_period_expenditure : ℝ := 85
    let second_period_expenditure : ℝ := 60
    let savings : ℝ := 30
    
    -- First period: fell into debt
    (monthly_income * first_period_months < first_period_expenditure * first_period_months) ∧
    
    -- Second period: cleared debt and saved
    (monthly_income * second_period_months = 
      second_period_expenditure * second_period_months + 
      (first_period_expenditure * first_period_months - monthly_income * first_period_months) + 
      savings) →
    
    monthly_income = 78

theorem labourer_monthly_income_proof : labourer_monthly_income 78 := by
  sorry

end labourer_monthly_income_labourer_monthly_income_proof_l3706_370686


namespace lucy_paid_correct_l3706_370600

/-- Calculate the total amount Lucy paid for fruits with discounts applied -/
def total_paid (grapes_kg : ℝ) (grapes_price : ℝ) (mangoes_kg : ℝ) (mangoes_price : ℝ)
                (apples_kg : ℝ) (apples_price : ℝ) (oranges_kg : ℝ) (oranges_price : ℝ)
                (grapes_apples_discount : ℝ) (mangoes_oranges_discount : ℝ) : ℝ :=
  let grapes_cost := grapes_kg * grapes_price
  let mangoes_cost := mangoes_kg * mangoes_price
  let apples_cost := apples_kg * apples_price
  let oranges_cost := oranges_kg * oranges_price
  let grapes_apples_total := grapes_cost + apples_cost
  let mangoes_oranges_total := mangoes_cost + oranges_cost
  let grapes_apples_discounted := grapes_apples_total * (1 - grapes_apples_discount)
  let mangoes_oranges_discounted := mangoes_oranges_total * (1 - mangoes_oranges_discount)
  grapes_apples_discounted + mangoes_oranges_discounted

theorem lucy_paid_correct :
  total_paid 6 74 9 59 4 45 12 32 0.07 0.05 = 1449.57 := by
  sorry

end lucy_paid_correct_l3706_370600


namespace probability_of_specific_arrangement_l3706_370603

def total_tiles : ℕ := 8
def x_tiles : ℕ := 5
def o_tiles : ℕ := 3

def specific_arrangement : List Char := ['X', 'X', 'X', 'O', 'O', 'X', 'O', 'X']

theorem probability_of_specific_arrangement :
  (1 : ℚ) / (Nat.choose total_tiles x_tiles) = 1 / 56 := by
  sorry

end probability_of_specific_arrangement_l3706_370603


namespace log_x_16_eq_0_8_implies_x_eq_32_l3706_370624

-- Define the logarithm function for our specific base
noncomputable def log_base (b : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log b

-- State the theorem
theorem log_x_16_eq_0_8_implies_x_eq_32 :
  ∀ x : ℝ, x > 0 → log_base x 16 = 0.8 → x = 32 := by
  sorry

end log_x_16_eq_0_8_implies_x_eq_32_l3706_370624


namespace square_plot_area_l3706_370628

/-- Given a square plot with a fence, prove that the area is 289 square feet
    when the price per foot is 59 and the total cost is 4012. -/
theorem square_plot_area (side_length : ℝ) (perimeter : ℝ) (price_per_foot : ℝ) (total_cost : ℝ) :
  price_per_foot = 59 →
  total_cost = 4012 →
  perimeter = 4 * side_length →
  total_cost = perimeter * price_per_foot →
  side_length ^ 2 = 289 := by
  sorry

end square_plot_area_l3706_370628


namespace special_triangle_common_area_l3706_370604

/-- A triangle with side lengths 18, 24, and 30 -/
structure SpecialTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : a = 18
  hb : b = 24
  hc : c = 30

/-- The common region of two overlapping triangles -/
def CommonRegion (t1 t2 : SpecialTriangle) : Set (ℝ × ℝ) := sorry

/-- The area of a set in ℝ² -/
noncomputable def area (s : Set (ℝ × ℝ)) : ℝ := sorry

/-- Two triangles share the same circumcircle -/
def ShareCircumcircle (t1 t2 : SpecialTriangle) : Prop := sorry

/-- Two triangles share the same inscribed circle -/
def ShareInscribedCircle (t1 t2 : SpecialTriangle) : Prop := sorry

/-- Two triangles do not completely overlap -/
def NotCompletelyOverlap (t1 t2 : SpecialTriangle) : Prop := sorry

theorem special_triangle_common_area 
  (t1 t2 : SpecialTriangle) 
  (h_circ : ShareCircumcircle t1 t2) 
  (h_insc : ShareInscribedCircle t1 t2) 
  (h_overlap : NotCompletelyOverlap t1 t2) : 
  area (CommonRegion t1 t2) = 132 := by sorry

end special_triangle_common_area_l3706_370604


namespace quadratic_equation_theorem_l3706_370649

theorem quadratic_equation_theorem (m : ℝ) (p : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 - 2*x + m - 1 = 0 ∧ y^2 - 2*y + m - 1 = 0) →  -- two real roots condition
  (p^2 - 2*p + m - 1 = 0) →  -- p is a root
  ((p^2 - 2*p + 3)*(m + 4) = 7) →  -- given equation
  (m = -3 ∧ m ≤ 2) := by
sorry

end quadratic_equation_theorem_l3706_370649


namespace equilateral_triangle_area_l3706_370693

/-- The area of an equilateral triangle with vertices at (1, 2), (1, 8), and (7, 2) is 9√3 square units. -/
theorem equilateral_triangle_area : 
  let E : ℝ × ℝ := (1, 2)
  let F : ℝ × ℝ := (1, 8)
  let G : ℝ × ℝ := (7, 2)
  let is_equilateral (A B C : ℝ × ℝ) : Prop := 
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = (B.1 - C.1)^2 + (B.2 - C.2)^2 ∧
    (B.1 - C.1)^2 + (B.2 - C.2)^2 = (C.1 - A.1)^2 + (C.2 - A.2)^2
  let triangle_area (A B C : ℝ × ℝ) : ℝ := 
    Real.sqrt 3 / 4 * ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  is_equilateral E F G → triangle_area E F G = 9 * Real.sqrt 3 :=
by
  sorry


end equilateral_triangle_area_l3706_370693
