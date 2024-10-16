import Mathlib

namespace NUMINAMATH_CALUDE_simple_interest_problem_l1270_127002

/-- Proves that given the conditions of the problem, the principal amount is 2800 --/
theorem simple_interest_problem (P : ℝ) : 
  (P * 4 * 5) / 100 = P - 2240 → P = 2800 := by sorry

end NUMINAMATH_CALUDE_simple_interest_problem_l1270_127002


namespace NUMINAMATH_CALUDE_school_c_sample_size_l1270_127077

/-- Represents the number of teachers in each school -/
structure SchoolPopulation where
  schoolA : ℕ
  schoolB : ℕ
  schoolC : ℕ

/-- Represents the sampling parameters -/
structure SamplingParams where
  totalSample : ℕ
  population : SchoolPopulation

/-- Calculates the stratified sample size for a given school -/
def stratifiedSampleSize (params : SamplingParams) (schoolSize : ℕ) : ℕ :=
  (schoolSize * params.totalSample) / (params.population.schoolA + params.population.schoolB + params.population.schoolC)

/-- Theorem stating that the stratified sample size for School C is 10 -/
theorem school_c_sample_size :
  let params : SamplingParams := {
    totalSample := 60,
    population := {
      schoolA := 180,
      schoolB := 270,
      schoolC := 90
    }
  }
  stratifiedSampleSize params params.population.schoolC = 10 := by
  sorry


end NUMINAMATH_CALUDE_school_c_sample_size_l1270_127077


namespace NUMINAMATH_CALUDE_negation_of_absolute_value_geq_one_l1270_127057

theorem negation_of_absolute_value_geq_one :
  (¬ ∀ x : ℝ, |x| ≥ 1) ↔ (∃ x : ℝ, |x| < 1) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_absolute_value_geq_one_l1270_127057


namespace NUMINAMATH_CALUDE_supplementary_angles_difference_l1270_127036

theorem supplementary_angles_difference (a b : ℝ) : 
  a + b = 180 →  -- supplementary angles
  a / b = 5 / 3 →  -- ratio of 5:3
  abs (a - b) = 45 :=  -- positive difference
by sorry

end NUMINAMATH_CALUDE_supplementary_angles_difference_l1270_127036


namespace NUMINAMATH_CALUDE_absolute_value_of_z_l1270_127069

theorem absolute_value_of_z (z : ℂ) (h : z^2 = 16 - 30*I) : Complex.abs z = Real.sqrt 34 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_of_z_l1270_127069


namespace NUMINAMATH_CALUDE_race_length_is_1000_l1270_127015

/-- Represents the length of the race in meters -/
def race_length : ℝ := 1000

/-- Represents the time difference between runners A and B in seconds -/
def time_difference : ℝ := 20

/-- Represents the distance difference between runners A and B in meters -/
def distance_difference : ℝ := 50

/-- Represents the time taken by runner A to complete the race in seconds -/
def time_A : ℝ := 380

/-- Theorem stating that the race length is 1000 meters given the conditions -/
theorem race_length_is_1000 :
  (distance_difference / time_difference) * (time_A + time_difference) = race_length := by
  sorry


end NUMINAMATH_CALUDE_race_length_is_1000_l1270_127015


namespace NUMINAMATH_CALUDE_bob_bake_time_proof_l1270_127071

/-- The time it takes Alice to bake a pie, in minutes -/
def alice_bake_time : ℝ := 5

/-- The time it takes Bob to bake a pie, in minutes -/
def bob_bake_time : ℝ := 6

/-- The total time available for baking, in minutes -/
def total_time : ℝ := 60

/-- The number of additional pies Alice can bake compared to Bob in the total time -/
def additional_pies : ℕ := 2

theorem bob_bake_time_proof :
  alice_bake_time = 5 ∧
  (total_time / alice_bake_time - total_time / bob_bake_time = additional_pies) →
  bob_bake_time = 6 := by
sorry

end NUMINAMATH_CALUDE_bob_bake_time_proof_l1270_127071


namespace NUMINAMATH_CALUDE_b_95_mod_49_l1270_127093

def b (n : ℕ) : ℕ := 5^n + 7^n + 3

theorem b_95_mod_49 : b 95 % 49 = 5 := by
  sorry

end NUMINAMATH_CALUDE_b_95_mod_49_l1270_127093


namespace NUMINAMATH_CALUDE_root_sum_theorem_l1270_127032

theorem root_sum_theorem (a b : ℝ) :
  (Complex.I : ℂ) ^ 2 = -1 →
  (2 + Complex.I : ℂ) ^ 3 + a * (2 + Complex.I : ℂ) + b = 0 →
  a + b = 9 := by
sorry

end NUMINAMATH_CALUDE_root_sum_theorem_l1270_127032


namespace NUMINAMATH_CALUDE_binomial_product_l1270_127049

theorem binomial_product (x : ℝ) : (4 * x + 3) * (2 * x - 7) = 8 * x^2 - 22 * x - 21 := by
  sorry

end NUMINAMATH_CALUDE_binomial_product_l1270_127049


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l1270_127078

theorem polynomial_division_remainder : ∃ q : Polynomial ℚ, 
  3 * X^2 - 22 * X + 70 = (X - 7) * q + 63 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l1270_127078


namespace NUMINAMATH_CALUDE_ryan_got_seven_books_l1270_127001

/-- Calculates the number of books Ryan got from the library given the conditions -/
def ryans_books (ryan_total_pages : ℕ) (brother_daily_pages : ℕ) (days : ℕ) (ryan_extra_daily_pages : ℕ) : ℕ :=
  ryan_total_pages / (brother_daily_pages + ryan_extra_daily_pages)

/-- Theorem stating that Ryan got 7 books from the library -/
theorem ryan_got_seven_books :
  ryans_books 2100 200 7 100 = 7 := by
  sorry

end NUMINAMATH_CALUDE_ryan_got_seven_books_l1270_127001


namespace NUMINAMATH_CALUDE_sample_customers_l1270_127055

theorem sample_customers (samples_per_box : ℕ) (boxes_opened : ℕ) (samples_leftover : ℕ) :
  samples_per_box = 20 →
  boxes_opened = 12 →
  samples_leftover = 5 →
  (samples_per_box * boxes_opened - samples_leftover : ℕ) = 235 :=
by sorry

end NUMINAMATH_CALUDE_sample_customers_l1270_127055


namespace NUMINAMATH_CALUDE_workshop_novelists_l1270_127098

theorem workshop_novelists (total : ℕ) (ratio_novelists : ℕ) (ratio_poets : ℕ) 
  (h1 : total = 24)
  (h2 : ratio_novelists = 5)
  (h3 : ratio_poets = 3) :
  (total * ratio_novelists) / (ratio_novelists + ratio_poets) = 15 := by
  sorry

end NUMINAMATH_CALUDE_workshop_novelists_l1270_127098


namespace NUMINAMATH_CALUDE_sum_of_w_and_y_is_eight_l1270_127062

theorem sum_of_w_and_y_is_eight (W X Y Z : ℤ) : 
  W ∈ ({1, 2, 3, 5} : Set ℤ) →
  X ∈ ({1, 2, 3, 5} : Set ℤ) →
  Y ∈ ({1, 2, 3, 5} : Set ℤ) →
  Z ∈ ({1, 2, 3, 5} : Set ℤ) →
  W ≠ X → W ≠ Y → W ≠ Z → X ≠ Y → X ≠ Z → Y ≠ Z →
  (W : ℚ) / (X : ℚ) - (Y : ℚ) / (Z : ℚ) = 1 →
  W + Y = 8 := by
sorry

end NUMINAMATH_CALUDE_sum_of_w_and_y_is_eight_l1270_127062


namespace NUMINAMATH_CALUDE_bart_firewood_consumption_l1270_127006

/-- The number of logs Bart burns per day -/
def logs_per_day (pieces_per_tree : ℕ) (trees_cut : ℕ) (days : ℕ) : ℚ :=
  (pieces_per_tree * trees_cut : ℚ) / days

theorem bart_firewood_consumption 
  (pieces_per_tree : ℕ) 
  (trees_cut : ℕ) 
  (days : ℕ) 
  (h1 : pieces_per_tree = 75)
  (h2 : trees_cut = 8)
  (h3 : days = 120) :
  logs_per_day pieces_per_tree trees_cut days = 5 := by
sorry

end NUMINAMATH_CALUDE_bart_firewood_consumption_l1270_127006


namespace NUMINAMATH_CALUDE_quadratic_real_solutions_l1270_127035

theorem quadratic_real_solutions (m : ℝ) :
  (∃ x : ℝ, m * x^2 - 4 * x + 1 = 0) → m ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_solutions_l1270_127035


namespace NUMINAMATH_CALUDE_hcf_problem_l1270_127040

theorem hcf_problem (a b : ℕ) (H : ℕ) : 
  (∃ k : ℕ, a = H * k) →  -- H divides a
  (∃ m : ℕ, b = H * m) →  -- H divides b
  (∃ n : ℕ, a * b = H * 13 * 14 * n) →  -- LCM condition
  (a = 322 ∨ b = 322) →  -- One of the numbers is 322
  (322 % 14 = 0) →  -- 322 is divisible by 14
  (322 % 13 ≠ 0) →  -- 322 is not divisible by 13
  H = 23 := by
sorry

end NUMINAMATH_CALUDE_hcf_problem_l1270_127040


namespace NUMINAMATH_CALUDE_smallest_part_of_proportional_division_l1270_127029

theorem smallest_part_of_proportional_division (total : ℝ) (p1 p2 p3 : ℝ) :
  total = 105 →
  p1 + p2 + p3 = total →
  p1 / 2 = p2 / (1/2) →
  p1 / 2 = p3 / (1/4) →
  min p1 (min p2 p3) = 10.5 :=
by sorry

end NUMINAMATH_CALUDE_smallest_part_of_proportional_division_l1270_127029


namespace NUMINAMATH_CALUDE_problem_solution_l1270_127045

theorem problem_solution (p q : ℝ) 
  (h1 : 1 < p) (h2 : p < q) 
  (h3 : 1 / p + 1 / q = 1) 
  (h4 : p * q = 12) : 
  q = 6 + 2 * Real.sqrt 6 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l1270_127045


namespace NUMINAMATH_CALUDE_new_person_weight_l1270_127041

theorem new_person_weight (initial_count : ℕ) (weight_increase : ℚ) (replaced_weight : ℚ) : 
  initial_count = 8 →
  weight_increase = 7/2 →
  replaced_weight = 62 →
  (initial_count : ℚ) * weight_increase + replaced_weight = 90 :=
by sorry

end NUMINAMATH_CALUDE_new_person_weight_l1270_127041


namespace NUMINAMATH_CALUDE_decision_has_two_exits_l1270_127018

-- Define the flowchart symbol types
inductive FlowchartSymbol
  | Terminal
  | InputOutput
  | Process
  | Decision

-- Define a function that returns the number of exit paths for each symbol
def exitPaths (symbol : FlowchartSymbol) : Nat :=
  match symbol with
  | FlowchartSymbol.Terminal => 1
  | FlowchartSymbol.InputOutput => 1
  | FlowchartSymbol.Process => 1
  | FlowchartSymbol.Decision => 2

-- Theorem statement
theorem decision_has_two_exits :
  ∀ (symbol : FlowchartSymbol),
    exitPaths symbol = 2 ↔ symbol = FlowchartSymbol.Decision :=
by sorry

end NUMINAMATH_CALUDE_decision_has_two_exits_l1270_127018


namespace NUMINAMATH_CALUDE_team_a_faster_by_three_hours_l1270_127063

/-- Proves that Team A finishes 3 hours faster than Team W in a 300-mile race -/
theorem team_a_faster_by_three_hours 
  (course_length : ℝ) 
  (speed_w : ℝ) 
  (speed_difference : ℝ) : 
  course_length = 300 → 
  speed_w = 20 → 
  speed_difference = 5 → 
  (course_length / speed_w) - (course_length / (speed_w + speed_difference)) = 3 := by
  sorry

#check team_a_faster_by_three_hours

end NUMINAMATH_CALUDE_team_a_faster_by_three_hours_l1270_127063


namespace NUMINAMATH_CALUDE_jerry_average_study_time_difference_l1270_127030

def daily_differences : List Int := [15, -5, 25, 0, -15, 10]

def extra_study_time : Int := 20

def adjust_difference (diff : Int) : Int :=
  if diff > 0 then diff + extra_study_time else diff

theorem jerry_average_study_time_difference :
  let adjusted_differences := daily_differences.map adjust_difference
  let total_difference := adjusted_differences.sum
  let num_days := daily_differences.length
  total_difference / num_days = -15 := by sorry

end NUMINAMATH_CALUDE_jerry_average_study_time_difference_l1270_127030


namespace NUMINAMATH_CALUDE_chess_team_selection_ways_l1270_127033

def total_members : ℕ := 18
def num_siblings : ℕ := 4
def team_size : ℕ := 8
def max_siblings_in_team : ℕ := 2

theorem chess_team_selection_ways :
  (Nat.choose total_members team_size) -
  (Nat.choose num_siblings (num_siblings) * Nat.choose (total_members - num_siblings) (team_size - num_siblings)) = 42757 := by
  sorry

end NUMINAMATH_CALUDE_chess_team_selection_ways_l1270_127033


namespace NUMINAMATH_CALUDE_union_complement_equal_to_set_l1270_127086

def U : Finset Nat := {0, 1, 2, 4, 6, 8}
def M : Finset Nat := {0, 4, 6}
def N : Finset Nat := {0, 1, 6}

theorem union_complement_equal_to_set :
  M ∪ (U \ N) = {0, 2, 4, 6, 8} := by sorry

end NUMINAMATH_CALUDE_union_complement_equal_to_set_l1270_127086


namespace NUMINAMATH_CALUDE_maggot_feeding_problem_l1270_127095

/-- The number of maggots attempted to be fed in the first feeding -/
def first_feeding : ℕ := 15

/-- The total number of maggots served -/
def total_maggots : ℕ := 20

/-- The number of maggots eaten in the first feeding -/
def eaten_first : ℕ := 1

/-- The number of maggots eaten in the second feeding -/
def eaten_second : ℕ := 3

theorem maggot_feeding_problem :
  first_feeding + eaten_first + eaten_second = total_maggots :=
by sorry

end NUMINAMATH_CALUDE_maggot_feeding_problem_l1270_127095


namespace NUMINAMATH_CALUDE_baker_cakes_l1270_127009

theorem baker_cakes (initial_cakes : ℕ) 
  (bought_cakes : ℕ := 103)
  (sold_cakes : ℕ := 86)
  (final_cakes : ℕ := 190)
  (h : initial_cakes + bought_cakes - sold_cakes = final_cakes) :
  initial_cakes = 173 := by
  sorry

end NUMINAMATH_CALUDE_baker_cakes_l1270_127009


namespace NUMINAMATH_CALUDE_det_A_squared_minus_3A_l1270_127028

def A : Matrix (Fin 2) (Fin 2) ℝ := !![2, 3; 2, 2]

theorem det_A_squared_minus_3A : Matrix.det ((A ^ 2) - 3 • A) = 10 := by
  sorry

end NUMINAMATH_CALUDE_det_A_squared_minus_3A_l1270_127028


namespace NUMINAMATH_CALUDE_A_subset_of_neg_one_one_l1270_127061

def A : Set ℝ := {x | x^2 - 1 = 0}

theorem A_subset_of_neg_one_one : A ⊆ {-1, 1} := by
  sorry

end NUMINAMATH_CALUDE_A_subset_of_neg_one_one_l1270_127061


namespace NUMINAMATH_CALUDE_clock_selling_theorem_l1270_127092

/-- Represents the clock selling scenario with given conditions -/
def ClockSelling (original_cost : ℚ) : Prop :=
  let initial_sale := original_cost * 1.2
  let buyback_price := initial_sale * 0.5
  let maintenance_cost := buyback_price * 0.1
  let total_spent := buyback_price + maintenance_cost
  let final_sale := total_spent * 1.8
  (original_cost - buyback_price = 100) ∧ (final_sale = 297)

/-- Theorem stating the existence of an original cost satisfying the ClockSelling conditions -/
theorem clock_selling_theorem : ∃ (original_cost : ℚ), ClockSelling original_cost :=
  sorry

end NUMINAMATH_CALUDE_clock_selling_theorem_l1270_127092


namespace NUMINAMATH_CALUDE_gift_distribution_sequences_l1270_127012

/-- The number of students in the class -/
def num_students : ℕ := 15

/-- The number of class meetings per week -/
def meetings_per_week : ℕ := 3

/-- The number of ways to distribute gifts in one class session -/
def ways_per_session : ℕ := num_students * num_students

/-- The total number of different gift distribution sequences in a week -/
def total_sequences : ℕ := ways_per_session ^ meetings_per_week

/-- Theorem stating the total number of different gift distribution sequences -/
theorem gift_distribution_sequences :
  total_sequences = 11390625 := by
  sorry

end NUMINAMATH_CALUDE_gift_distribution_sequences_l1270_127012


namespace NUMINAMATH_CALUDE_sum_of_numbers_greater_than_04_l1270_127094

def numbers : List ℚ := [0.8, 1/2, 0.3, 1/3]

theorem sum_of_numbers_greater_than_04 : 
  (numbers.filter (λ x => x > 0.4)).sum = 1.3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_numbers_greater_than_04_l1270_127094


namespace NUMINAMATH_CALUDE_xiaoming_age_proof_l1270_127052

/-- Xiaoming's current age -/
def xiaoming_age : ℕ := 6

/-- The current age of each of Xiaoming's younger brothers -/
def brother_age : ℕ := 2

/-- The number of Xiaoming's younger brothers -/
def num_brothers : ℕ := 3

/-- Years into the future for the second condition -/
def future_years : ℕ := 6

theorem xiaoming_age_proof :
  (xiaoming_age = num_brothers * brother_age) ∧
  (num_brothers * (brother_age + future_years) = 2 * (xiaoming_age + future_years)) →
  xiaoming_age = 6 := by
  sorry

end NUMINAMATH_CALUDE_xiaoming_age_proof_l1270_127052


namespace NUMINAMATH_CALUDE_stock_price_calculation_l1270_127090

theorem stock_price_calculation (initial_price : ℝ) (first_year_increase : ℝ) (second_year_decrease : ℝ) :
  initial_price = 120 →
  first_year_increase = 0.8 →
  second_year_decrease = 0.3 →
  let price_after_first_year := initial_price * (1 + first_year_increase)
  let final_price := price_after_first_year * (1 - second_year_decrease)
  final_price = 151.2 :=
by sorry

end NUMINAMATH_CALUDE_stock_price_calculation_l1270_127090


namespace NUMINAMATH_CALUDE_water_tank_capacity_l1270_127037

theorem water_tank_capacity : ∀ x : ℚ,
  (5/6 : ℚ) * x - 30 = (4/5 : ℚ) * x → x = 900 := by
  sorry

end NUMINAMATH_CALUDE_water_tank_capacity_l1270_127037


namespace NUMINAMATH_CALUDE_remainder_of_2456789_div_7_l1270_127005

theorem remainder_of_2456789_div_7 : 2456789 % 7 = 6 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_2456789_div_7_l1270_127005


namespace NUMINAMATH_CALUDE_circle_passes_through_points_and_center_on_line_l1270_127060

-- Define the points M and N
def M : ℝ × ℝ := (5, 2)
def N : ℝ × ℝ := (3, 2)

-- Define the line equation y = 2x - 3
def line_equation (x y : ℝ) : Prop := y = 2 * x - 3

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := (x - 4)^2 + (y - 5)^2 = 10

-- Theorem statement
theorem circle_passes_through_points_and_center_on_line :
  ∃ (center : ℝ × ℝ),
    line_equation center.1 center.2 ∧
    circle_equation M.1 M.2 ∧
    circle_equation N.1 N.2 :=
sorry

end NUMINAMATH_CALUDE_circle_passes_through_points_and_center_on_line_l1270_127060


namespace NUMINAMATH_CALUDE_expression_equals_three_l1270_127046

theorem expression_equals_three :
  |Real.sqrt 3 - 1| + (2023 - Real.pi)^0 - (-1/3)⁻¹ - 3 * Real.tan (30 * π / 180) = 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_three_l1270_127046


namespace NUMINAMATH_CALUDE_sqrt_meaningful_range_l1270_127016

theorem sqrt_meaningful_range (x : ℝ) :
  (∃ y : ℝ, y ^ 2 = x - 1) → x ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_range_l1270_127016


namespace NUMINAMATH_CALUDE_no_digit_satisfies_property_l1270_127097

/-- Represents a single digit (0-9) -/
def Digit := Fin 10

/-- Checks if the decimal representation of a natural number ends with at least k repetitions of a digit -/
def EndsWithRepeatedDigit (num : ℕ) (d : Digit) (k : ℕ) : Prop :=
  ∃ m : ℕ, num % (10^k) = d.val * ((10^k - 1) / 9)

/-- The main theorem stating that no digit satisfies the given property -/
theorem no_digit_satisfies_property : 
  ¬ ∃ z : Digit, ∀ k : ℕ, k ≥ 1 → ∃ n : ℕ, n ≥ 1 ∧ EndsWithRepeatedDigit (n^9) z k :=
by sorry


end NUMINAMATH_CALUDE_no_digit_satisfies_property_l1270_127097


namespace NUMINAMATH_CALUDE_chicken_eggs_l1270_127022

/-- The number of eggs laid by a chicken over two days -/
def total_eggs (today : ℕ) (yesterday : ℕ) : ℕ := today + yesterday

/-- Theorem: The chicken laid 49 eggs in total over two days -/
theorem chicken_eggs : total_eggs 30 19 = 49 := by
  sorry

end NUMINAMATH_CALUDE_chicken_eggs_l1270_127022


namespace NUMINAMATH_CALUDE_berts_spending_l1270_127099

/-- Bert's spending problem -/
theorem berts_spending (n : ℝ) : 
  (1/2) * ((2/3) * n - 7) = 10.5 → n = 42 := by
  sorry

end NUMINAMATH_CALUDE_berts_spending_l1270_127099


namespace NUMINAMATH_CALUDE_pen_count_l1270_127079

theorem pen_count (num_pencils : ℕ) (max_students : ℕ) (num_pens : ℕ) : 
  num_pencils = 520 →
  max_students = 40 →
  num_pencils % max_students = 0 →
  num_pens % max_students = 0 →
  (num_pencils / max_students = num_pens / max_students) →
  num_pens = 520 := by
sorry

end NUMINAMATH_CALUDE_pen_count_l1270_127079


namespace NUMINAMATH_CALUDE_product_trailing_zeros_l1270_127007

def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125) + (n / 625)

theorem product_trailing_zeros :
  trailingZeros 2014 = 501 := by
  sorry

end NUMINAMATH_CALUDE_product_trailing_zeros_l1270_127007


namespace NUMINAMATH_CALUDE_sum_of_coordinates_is_60_l1270_127070

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Define the conditions for the points
def satisfies_conditions (p : Point) : Prop :=
  (abs (p.y - 10) = 4) ∧
  ((p.x - 5)^2 + (p.y - 10)^2 = 12^2)

-- Theorem statement
theorem sum_of_coordinates_is_60 :
  ∀ (p1 p2 p3 p4 : Point),
    satisfies_conditions p1 →
    satisfies_conditions p2 →
    satisfies_conditions p3 →
    satisfies_conditions p4 →
    p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p2 ≠ p3 ∧ p2 ≠ p4 ∧ p3 ≠ p4 →
    p1.x + p1.y + p2.x + p2.y + p3.x + p3.y + p4.x + p4.y = 60 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_coordinates_is_60_l1270_127070


namespace NUMINAMATH_CALUDE_no_integer_satisfies_conditions_l1270_127096

theorem no_integer_satisfies_conditions : 
  ¬∃ (n : ℤ), (30 - 6 * n > 18) ∧ (2 * n + 5 = 11) := by
  sorry

end NUMINAMATH_CALUDE_no_integer_satisfies_conditions_l1270_127096


namespace NUMINAMATH_CALUDE_email_problem_l1270_127081

theorem email_problem (x : ℚ) : 
  x + x/2 + x/4 + x/8 = 30 → x = 16 := by
  sorry

end NUMINAMATH_CALUDE_email_problem_l1270_127081


namespace NUMINAMATH_CALUDE_triangle_distance_set_l1270_127075

theorem triangle_distance_set (a b k : ℝ) (ha : 0 < a) (hb : 0 < b) (hk : k^2 > 2*a^2/3 + 2*b^2/3) :
  let S := {P : ℝ × ℝ | P.1^2 + P.2^2 + (P.1 - a)^2 + P.2^2 + P.1^2 + (P.2 - b)^2 < k^2}
  let C := {P : ℝ × ℝ | (P.1 - a/3)^2 + (P.2 - b/3)^2 < (k^2 - 2*a^2/3 - 2*b^2/3) / 3}
  S = C := by sorry

end NUMINAMATH_CALUDE_triangle_distance_set_l1270_127075


namespace NUMINAMATH_CALUDE_parabola_directrix_l1270_127082

/-- Given a parabola with equation y = ax² and directrix y = 2, prove that a = -1/8 -/
theorem parabola_directrix (a : ℝ) : 
  (∀ x y : ℝ, y = a * x^2) →  -- Parabola equation
  (2 : ℝ) = -1 / (4 * a) →    -- Directrix equation (in standard form)
  a = -1/8 := by
sorry

end NUMINAMATH_CALUDE_parabola_directrix_l1270_127082


namespace NUMINAMATH_CALUDE_units_digit_of_5_to_12_l1270_127011

theorem units_digit_of_5_to_12 : ∃ n : ℕ, 5^12 ≡ 5 [ZMOD 10] :=
  sorry

end NUMINAMATH_CALUDE_units_digit_of_5_to_12_l1270_127011


namespace NUMINAMATH_CALUDE_cricket_team_ratio_l1270_127038

theorem cricket_team_ratio (total_players : ℕ) (throwers : ℕ) (right_handed : ℕ) :
  total_players = 70 →
  throwers = 37 →
  right_handed = 59 →
  (total_players - throwers : ℚ) / (total_players - throwers) = 3 / 4 :=
by sorry

end NUMINAMATH_CALUDE_cricket_team_ratio_l1270_127038


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l1270_127064

theorem min_value_expression (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  8 * a^4 + 12 * b^4 + 40 * c^4 + 2 * d^2 + 1 / (5 * a * b * c * d) ≥ 4 * Real.sqrt 10 / 5 :=
by sorry

theorem min_value_achievable :
  ∃ a b c d : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
  8 * a^4 + 12 * b^4 + 40 * c^4 + 2 * d^2 + 1 / (5 * a * b * c * d) = 4 * Real.sqrt 10 / 5 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l1270_127064


namespace NUMINAMATH_CALUDE_cone_surface_area_l1270_127072

/-- The surface area of a cone with base radius 1 and height √3 is 3π. -/
theorem cone_surface_area :
  let r : ℝ := 1
  let h : ℝ := Real.sqrt 3
  let l : ℝ := Real.sqrt (r^2 + h^2)
  let surface_area : ℝ := π * r^2 + π * r * l
  surface_area = 3 * π :=
by sorry

end NUMINAMATH_CALUDE_cone_surface_area_l1270_127072


namespace NUMINAMATH_CALUDE_skating_average_theorem_l1270_127023

/-- Represents Gage's skating schedule --/
structure SkatingSchedule :=
  (days_1 : Nat) (time_1 : Nat)
  (days_2 : Nat) (time_2 : Nat)
  (days_3 : Nat) (time_3 : Nat)
  (days_4 : Nat) (time_4 : Nat)

/-- Calculates the total skating time for 9 days --/
def total_time_9_days (s : SkatingSchedule) : Nat :=
  s.days_1 * s.time_1 + s.days_2 * s.time_2 + s.days_3 * s.time_3 + s.days_4 * s.time_4

/-- Theorem: Skating 85 minutes on the 10th day results in a 90-minute average --/
theorem skating_average_theorem (s : SkatingSchedule) 
  (h1 : s.days_1 = 5 ∧ s.time_1 = 75)
  (h2 : s.days_2 = 3 ∧ s.time_2 = 90)
  (h3 : s.days_3 = 1 ∧ s.time_3 = 120)
  (h4 : s.days_4 = 1 ∧ s.time_4 = 50) :
  (total_time_9_days s + 85) / 10 = 90 := by
  sorry

#check skating_average_theorem

end NUMINAMATH_CALUDE_skating_average_theorem_l1270_127023


namespace NUMINAMATH_CALUDE_motel_rent_theorem_l1270_127087

/-- Represents the total rent charged by a motel --/
def TotalRent (x y : ℕ) : ℕ := 40 * x + 60 * y

/-- The problem statement --/
theorem motel_rent_theorem (x y : ℕ) :
  (TotalRent (x + 10) (y - 10) = (TotalRent x y) / 2) →
  TotalRent x y = 800 :=
by sorry

end NUMINAMATH_CALUDE_motel_rent_theorem_l1270_127087


namespace NUMINAMATH_CALUDE_isosceles_triangle_lengths_l1270_127024

/-- An isosceles triangle with a median dividing the perimeter -/
structure IsoscelesTriangleWithMedian where
  leg : ℝ
  base : ℝ
  is_positive : 0 < leg ∧ 0 < base
  is_isosceles : leg > 0
  median_division : 2 * leg + base = 21
  perimeter_division : |2 * leg - base| = 9

/-- The legs of the triangle have length 10 and the base has length 1 -/
theorem isosceles_triangle_lengths (t : IsoscelesTriangleWithMedian) : 
  t.leg = 10 ∧ t.base = 1 := by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_lengths_l1270_127024


namespace NUMINAMATH_CALUDE_line_through_two_points_l1270_127066

/-- Given a line x = 8y + 5 passing through points (m, n) and (m + 2, n + p), prove that p = 1/4 -/
theorem line_through_two_points (m n : ℝ) :
  let line := fun y : ℝ => 8 * y + 5
  line n = m → line (n + p) = m + 2 → p = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_line_through_two_points_l1270_127066


namespace NUMINAMATH_CALUDE_A_intersect_B_is_singleton_one_l1270_127010

def A : Set ℝ := {x | x ≤ 1}
def B : Set ℝ := {y | ∃ x, y = x^2 + 2*x + 2}

theorem A_intersect_B_is_singleton_one : A ∩ B = {1} := by sorry

end NUMINAMATH_CALUDE_A_intersect_B_is_singleton_one_l1270_127010


namespace NUMINAMATH_CALUDE_sphere_surface_area_tangent_to_cube_l1270_127013

/-- The surface area of a sphere tangent to all faces of a cube -/
theorem sphere_surface_area_tangent_to_cube (cube_edge_length : ℝ) (sphere_radius : ℝ) :
  cube_edge_length = 2 →
  sphere_radius = cube_edge_length / 2 →
  4 * Real.pi * sphere_radius^2 = 4 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sphere_surface_area_tangent_to_cube_l1270_127013


namespace NUMINAMATH_CALUDE_log_sum_equals_two_l1270_127084

-- Define the base-10 logarithm
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the base-2 logarithm
noncomputable def log2 (x : ℝ) : ℝ := Real.log x / Real.log 2

-- Theorem statement
theorem log_sum_equals_two : lg 0.01 + log2 16 = 2 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_equals_two_l1270_127084


namespace NUMINAMATH_CALUDE_arithmetic_progression_squares_l1270_127048

theorem arithmetic_progression_squares (a d : ℝ) : 
  (a - d)^2 + a^2 = 100 ∧ a^2 + (a + d)^2 = 164 →
  ((a - d, a, a + d) = (6, 8, 10) ∨
   (a - d, a, a + d) = (-10, -8, -6) ∨
   (a - d, a, a + d) = (-7 * Real.sqrt 2, Real.sqrt 2, 9 * Real.sqrt 2) ∨
   (a - d, a, a + d) = (10 * Real.sqrt 2, 8 * Real.sqrt 2, Real.sqrt 2)) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_progression_squares_l1270_127048


namespace NUMINAMATH_CALUDE_article_pricing_l1270_127039

theorem article_pricing (CP : ℝ) (SP1 SP3 : ℝ) 
  (h1 : SP1 - CP = CP - 448)
  (h2 : SP3 = 1020)
  (h3 : SP3 = CP + 0.5 * CP) :
  SP3 = 1020 ∧ SP3 = 1.5 * CP := by
sorry

end NUMINAMATH_CALUDE_article_pricing_l1270_127039


namespace NUMINAMATH_CALUDE_f_has_root_in_interval_l1270_127056

-- Define the function f(x) = 2x + x - 2
def f (x : ℝ) : ℝ := 2 * x + x - 2

-- Theorem statement
theorem f_has_root_in_interval :
  ∃ c ∈ Set.Ioo 0 1, f c = 0 :=
sorry

end NUMINAMATH_CALUDE_f_has_root_in_interval_l1270_127056


namespace NUMINAMATH_CALUDE_waiter_tables_l1270_127089

/-- Calculates the number of tables given the initial number of customers,
    the number of customers who left, and the number of people at each remaining table. -/
def calculate_tables (initial_customers : ℕ) (customers_left : ℕ) (people_per_table : ℕ) : ℕ :=
  (initial_customers - customers_left) / people_per_table

/-- Theorem stating that for the given problem, the number of tables is 5. -/
theorem waiter_tables : calculate_tables 62 17 9 = 5 := by
  sorry


end NUMINAMATH_CALUDE_waiter_tables_l1270_127089


namespace NUMINAMATH_CALUDE_range_of_m_l1270_127088

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + x^2 - 2*a*x + 1

theorem range_of_m (m : ℝ) :
  (∃ x₀ : ℝ, x₀ ∈ Set.Ioc 0 1 ∧
    ∀ a : ℝ, a ∈ Set.Icc (-2) 0 →
      2*m*(Real.exp a) + f a x₀ > a^2 + 2*a + 4) →
  m ∈ Set.Ioo 1 (Real.exp 2) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l1270_127088


namespace NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l1270_127044

-- Define the repeating decimal
def repeating_decimal : ℚ := 37 / 100 + 264 / 99900

-- Define the fraction
def fraction : ℚ := 37189162 / 99900

-- Theorem statement
theorem repeating_decimal_equals_fraction : repeating_decimal = fraction := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l1270_127044


namespace NUMINAMATH_CALUDE_product_zero_iff_one_zero_l1270_127058

theorem product_zero_iff_one_zero (a b c : ℝ) : a * b * c = 0 ↔ a = 0 ∨ b = 0 ∨ c = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_zero_iff_one_zero_l1270_127058


namespace NUMINAMATH_CALUDE_modulus_of_complex_power_l1270_127073

theorem modulus_of_complex_power : 
  Complex.abs ((2 - 3 * Complex.I * Real.sqrt 3) ^ 4) = 961 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_complex_power_l1270_127073


namespace NUMINAMATH_CALUDE_volleyball_handshakes_l1270_127074

theorem volleyball_handshakes (total_handshakes : ℕ) (h : total_handshakes = 496) :
  ∃ (n : ℕ), 
    n * (n - 1) / 2 = total_handshakes ∧
    ∀ (coach_handshakes : ℕ), 
      n * (n - 1) / 2 + coach_handshakes = total_handshakes → 
      coach_handshakes ≥ 0 ∧
      (coach_handshakes = 0 → 
        ∀ (other_coach_handshakes : ℕ), 
          n * (n - 1) / 2 + other_coach_handshakes = total_handshakes → 
          other_coach_handshakes ≥ coach_handshakes) :=
by sorry

end NUMINAMATH_CALUDE_volleyball_handshakes_l1270_127074


namespace NUMINAMATH_CALUDE_pr_cr_relation_l1270_127083

theorem pr_cr_relation (p c : ℝ) :
  (6 * p * 4 = 360) → (p = 15 ∧ 6 * c * 4 = 24 * c) := by
  sorry

end NUMINAMATH_CALUDE_pr_cr_relation_l1270_127083


namespace NUMINAMATH_CALUDE_percentage_of_invalid_votes_l1270_127080

theorem percentage_of_invalid_votes
  (total_votes : ℕ)
  (candidate_a_percentage : ℚ)
  (candidate_a_votes : ℕ)
  (h1 : total_votes = 560000)
  (h2 : candidate_a_percentage = 80 / 100)
  (h3 : candidate_a_votes = 380800) :
  (total_votes - (candidate_a_votes / candidate_a_percentage)) / total_votes = 15 / 100 :=
sorry

end NUMINAMATH_CALUDE_percentage_of_invalid_votes_l1270_127080


namespace NUMINAMATH_CALUDE_faculty_reduction_proof_l1270_127004

/-- The original number of faculty members before reduction -/
def original_faculty : ℕ := 253

/-- The percentage of faculty remaining after reduction -/
def remaining_percentage : ℚ := 77 / 100

/-- The number of faculty members after reduction -/
def reduced_faculty : ℕ := 195

/-- Theorem stating that the original faculty count, when reduced by 23%, 
    results in approximately 195 professors -/
theorem faculty_reduction_proof : 
  ⌊(original_faculty : ℚ) * remaining_percentage⌋ = reduced_faculty :=
sorry

end NUMINAMATH_CALUDE_faculty_reduction_proof_l1270_127004


namespace NUMINAMATH_CALUDE_school_commute_time_l1270_127051

theorem school_commute_time (usual_rate : ℝ) (usual_time : ℝ) : 
  usual_time > 0 →
  usual_rate > 0 →
  (6 / 7 * usual_rate) * (usual_time - 4) = usual_rate * usual_time →
  usual_time = 28 := by
sorry

end NUMINAMATH_CALUDE_school_commute_time_l1270_127051


namespace NUMINAMATH_CALUDE_triangle_pieces_count_l1270_127027

/-- The number of rods in the nth row of the triangle -/
def rods_in_row (n : ℕ) : ℕ := 4 * n

/-- The total number of rods in a triangle with n rows -/
def total_rods (n : ℕ) : ℕ := (n * (n + 1) / 2) * 4

/-- The number of connectors in a triangle with n rows of rods -/
def total_connectors (n : ℕ) : ℕ := ((n + 1) * (n + 2)) / 2

/-- The total number of pieces (rods and connectors) in a triangle with n rows -/
def total_pieces (n : ℕ) : ℕ := total_rods n + total_connectors n

theorem triangle_pieces_count :
  total_pieces 10 = 286 := by sorry

end NUMINAMATH_CALUDE_triangle_pieces_count_l1270_127027


namespace NUMINAMATH_CALUDE_unique_solution_equation_l1270_127043

/-- There exists exactly one ordered pair of real numbers (x, y) satisfying the given equation -/
theorem unique_solution_equation :
  ∃! (x y : ℝ), 9^(x^2 + y) + 9^(x + y^2) = Real.sqrt 2 ∧ x = -1/2 ∧ y = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_equation_l1270_127043


namespace NUMINAMATH_CALUDE_range_of_m_l1270_127076

def A : Set ℝ := {x | 3 < x ∧ x < 10}
def B : Set ℝ := {x | x^2 - 9*x + 14 < 0}
def C (m : ℝ) : Set ℝ := {x | 5 - m < x ∧ x < 2*m}

theorem range_of_m :
  ∀ m : ℝ, (∀ x : ℝ, x ∈ C m → x ∈ A ∩ B) ∧
           (∃ y : ℝ, y ∈ A ∩ B ∧ y ∉ C m) ↔
  m ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l1270_127076


namespace NUMINAMATH_CALUDE_cake_difference_l1270_127054

/-- Given the initial number of cakes, the number of cakes sold, and the number of cakes bought,
    prove that the difference between cakes bought and sold is 63. -/
theorem cake_difference (initial : ℕ) (sold : ℕ) (bought : ℕ)
    (h1 : initial = 13)
    (h2 : sold = 91)
    (h3 : bought = 154) :
    bought - sold = 63 := by
  sorry

end NUMINAMATH_CALUDE_cake_difference_l1270_127054


namespace NUMINAMATH_CALUDE_rhombus_perimeter_l1270_127065

/-- The perimeter of a rhombus given its diagonals -/
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 24) (h2 : d2 = 16) :
  4 * Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2) = 16 * Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_perimeter_l1270_127065


namespace NUMINAMATH_CALUDE_max_leap_years_in_period_l1270_127042

/-- The number of years in the period -/
def period : ℕ := 125

/-- The interval between leap years -/
def leap_year_interval : ℕ := 5

/-- The maximum number of leap years in the given period -/
def max_leap_years : ℕ := period / leap_year_interval

theorem max_leap_years_in_period :
  max_leap_years = 25 := by sorry

end NUMINAMATH_CALUDE_max_leap_years_in_period_l1270_127042


namespace NUMINAMATH_CALUDE_large_tent_fabric_is_8_l1270_127091

/-- The amount of fabric needed for a small tent -/
def small_tent_fabric : ℝ := 4

/-- The amount of fabric needed for a large tent -/
def large_tent_fabric : ℝ := 2 * small_tent_fabric

/-- Theorem: The fabric needed for a large tent is 8 square meters -/
theorem large_tent_fabric_is_8 : large_tent_fabric = 8 := by
  sorry

end NUMINAMATH_CALUDE_large_tent_fabric_is_8_l1270_127091


namespace NUMINAMATH_CALUDE_program_output_for_351_l1270_127031

def program_output (x : ℕ) : ℕ :=
  if 100 < x ∧ x < 1000 then
    let a := x / 100
    let b := (x - a * 100) / 10
    let c := x % 10
    100 * c + 10 * b + a
  else
    x

theorem program_output_for_351 :
  program_output 351 = 153 :=
by
  sorry

end NUMINAMATH_CALUDE_program_output_for_351_l1270_127031


namespace NUMINAMATH_CALUDE_inequality_solution_range_l1270_127053

theorem inequality_solution_range (k : ℝ) : 
  (∃ (x y : ℕ), x ≠ y ∧ 
    (∀ (z : ℕ), z > 0 → (k * (z : ℝ)^2 ≤ Real.log z + 1) ↔ (z = x ∨ z = y))) →
  ((Real.log 3 + 1) / 9 < k ∧ k ≤ (Real.log 2 + 1) / 4) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l1270_127053


namespace NUMINAMATH_CALUDE_dot_product_equals_ten_l1270_127059

/-- Given two vectors a and b in ℝ², prove that their dot product is 10 -/
theorem dot_product_equals_ten (a b : ℝ × ℝ) :
  a = (-2, -6) →
  ‖b‖ = Real.sqrt 10 →
  Real.cos (60 * π / 180) = (a.1 * b.1 + a.2 * b.2) / (‖a‖ * ‖b‖) →
  a.1 * b.1 + a.2 * b.2 = 10 := by
  sorry

#check dot_product_equals_ten

end NUMINAMATH_CALUDE_dot_product_equals_ten_l1270_127059


namespace NUMINAMATH_CALUDE_all_options_satisfy_statement_l1270_127003

def sum_of_digits (n : ℕ) : ℕ := sorry

theorem all_options_satisfy_statement : 
  ∀ n ∈ ({54, 81, 99, 108} : Set ℕ), 
    (sum_of_digits n) % 9 = 0 → n % 9 = 0 ∧ n % 3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_all_options_satisfy_statement_l1270_127003


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l1270_127047

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℚ) : Prop :=
  ∃ q : ℚ, ∀ n : ℕ, a (n + 1) = a n * q

-- Define the theorem
theorem geometric_sequence_ratio
  (a : ℕ → ℚ)
  (h_geometric : geometric_sequence a)
  (h_a2 : a 2 = 2)
  (h_a6 : a 6 = 1/8) :
  ∃ q : ℚ, (q = 1/2 ∨ q = -1/2) ∧ ∀ n : ℕ, a (n + 1) = a n * q :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l1270_127047


namespace NUMINAMATH_CALUDE_player_B_hit_rate_player_A_hit_at_least_once_in_two_l1270_127017

-- Define the hit rates and probabilities
def player_A_hit_rate : ℚ := 1/2
def player_B_miss_twice_prob : ℚ := 1/16

-- Theorem for player B's hit rate
theorem player_B_hit_rate : 
  ∃ p : ℚ, (1 - p)^2 = player_B_miss_twice_prob ∧ p = 3/4 :=
sorry

-- Theorem for player A's probability of hitting at least once in two shots
theorem player_A_hit_at_least_once_in_two :
  1 - (1 - player_A_hit_rate)^2 = 3/4 :=
sorry

end NUMINAMATH_CALUDE_player_B_hit_rate_player_A_hit_at_least_once_in_two_l1270_127017


namespace NUMINAMATH_CALUDE_arithmetic_sequence_proof_l1270_127067

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_proof (a : ℕ → ℝ) (d : ℝ) :
  d ≠ 0 →
  a 3 = 5 →
  (a 2) ^ 2 = a 1 * a 5 →
  arithmetic_sequence a d →
  ∀ n, a n = 2 * n - 1 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_proof_l1270_127067


namespace NUMINAMATH_CALUDE_sum_of_reciprocal_relations_l1270_127034

theorem sum_of_reciprocal_relations (x y : ℝ) 
  (h1 : 1/x + 1/y = 4) 
  (h2 : 1/x - 1/y = -6) : 
  x + y = -4/5 := by
sorry

end NUMINAMATH_CALUDE_sum_of_reciprocal_relations_l1270_127034


namespace NUMINAMATH_CALUDE_paper_pieces_sum_l1270_127025

/-- The number of pieces of paper picked up by Olivia and Edward -/
theorem paper_pieces_sum (olivia_pieces edward_pieces : ℕ) 
  (h_olivia : olivia_pieces = 16) 
  (h_edward : edward_pieces = 3) : 
  olivia_pieces + edward_pieces = 19 := by
  sorry

end NUMINAMATH_CALUDE_paper_pieces_sum_l1270_127025


namespace NUMINAMATH_CALUDE_placard_distribution_l1270_127021

theorem placard_distribution (total_placards : ℕ) (total_people : ℕ) 
  (h1 : total_placards = 4634) (h2 : total_people = 2317) :
  total_placards / total_people = 2 := by
  sorry

end NUMINAMATH_CALUDE_placard_distribution_l1270_127021


namespace NUMINAMATH_CALUDE_triple_solution_l1270_127026

theorem triple_solution (a b c : ℝ) : 
  a * b * c = 8 ∧ 
  a^2 * b + b^2 * c + c^2 * a = 73 ∧ 
  a * (b - c)^2 + b * (c - a)^2 + c * (a - b)^2 = 98 →
  ((a = 4 ∧ b = 4 ∧ c = 1/2) ∨
   (a = 4 ∧ b = 1/2 ∧ c = 4) ∨
   (a = 1/2 ∧ b = 4 ∧ c = 4) ∨
   (a = 1 ∧ b = 1 ∧ c = 8) ∨
   (a = 1 ∧ b = 8 ∧ c = 1) ∨
   (a = 8 ∧ b = 1 ∧ c = 1)) := by
  sorry

end NUMINAMATH_CALUDE_triple_solution_l1270_127026


namespace NUMINAMATH_CALUDE_degree_plus_one_divides_l1270_127085

/-- A polynomial with coefficients in {1, 2022} -/
def SpecialPoly (R : Type*) [CommRing R] := Polynomial R

/-- Predicate to check if a polynomial has coefficients only in {1, 2022} -/
def HasSpecialCoeffs (p : Polynomial ℤ) : Prop :=
  ∀ (i : ℕ), p.coeff i = 1 ∨ p.coeff i = 2022 ∨ p.coeff i = 0

theorem degree_plus_one_divides
  (f g : Polynomial ℤ)
  (hf : HasSpecialCoeffs f)
  (hg : HasSpecialCoeffs g)
  (h_div : f ∣ g) :
  (Polynomial.degree f + 1) ∣ (Polynomial.degree g + 1) :=
sorry

end NUMINAMATH_CALUDE_degree_plus_one_divides_l1270_127085


namespace NUMINAMATH_CALUDE_fibonacci_property_l1270_127050

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | n + 2 => fibonacci (n + 1) + fibonacci n

theorem fibonacci_property :
  let a := fibonacci
  (a 0 * a 2 + a 1 * a 3 + a 2 * a 4 + a 3 * a 5 + a 4 * a 6 + a 5 * a 7) -
  (a 1^2 + a 2^2 + a 3^2 + a 4^2 + a 5^2 + a 6^2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_fibonacci_property_l1270_127050


namespace NUMINAMATH_CALUDE_expand_expression_l1270_127000

theorem expand_expression (x : ℝ) : 5 * (x + 3) * (x + 6) = 5 * x^2 + 45 * x + 90 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l1270_127000


namespace NUMINAMATH_CALUDE_cube_root_of_product_l1270_127068

theorem cube_root_of_product (a : ℕ) : a^3 = 21 * 25 * 45 * 49 → a = 105 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_product_l1270_127068


namespace NUMINAMATH_CALUDE_smallest_prime_divisor_of_quadratic_l1270_127014

theorem smallest_prime_divisor_of_quadratic : 
  ∃ (p : ℕ), Prime p ∧ 
  (∃ (n : ℤ), (n^2 + n + 11).natAbs % p = 0) ∧
  (∀ (q : ℕ), Prime q → q < p → 
    ∀ (m : ℤ), (m^2 + m + 11).natAbs % q ≠ 0) ∧
  p = 11 := by
sorry

end NUMINAMATH_CALUDE_smallest_prime_divisor_of_quadratic_l1270_127014


namespace NUMINAMATH_CALUDE_log_power_base_l1270_127020

theorem log_power_base (a k P : ℝ) (ha : a > 0) (ha1 : a ≠ 1) :
  Real.log P / Real.log (a^k) = Real.log P / Real.log a / k := by
  sorry

end NUMINAMATH_CALUDE_log_power_base_l1270_127020


namespace NUMINAMATH_CALUDE_square_area_from_adjacent_points_l1270_127019

/-- Given two adjacent points (1,1) and (1,5) on a square in a Cartesian coordinate plane,
    the area of the square is 16. -/
theorem square_area_from_adjacent_points :
  let p1 : ℝ × ℝ := (1, 1)
  let p2 : ℝ × ℝ := (1, 5)
  let side_length := Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)
  let area := side_length^2
  area = 16 := by
sorry


end NUMINAMATH_CALUDE_square_area_from_adjacent_points_l1270_127019


namespace NUMINAMATH_CALUDE_investment_ratio_l1270_127008

/-- 
Given two investors p and q who divide their profit in the ratio 4:5,
prove that if p invested 52000, then q invested 65000.
-/
theorem investment_ratio (p q : ℕ) (h1 : p = 52000) : 
  (p : ℚ) / q = 4 / 5 → q = 65000 := by
sorry

end NUMINAMATH_CALUDE_investment_ratio_l1270_127008
