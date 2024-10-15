import Mathlib

namespace NUMINAMATH_CALUDE_max_consecutive_integers_sum_l2437_243709

theorem max_consecutive_integers_sum (k : ℕ) : k ≤ 81 ↔ ∃ n : ℕ, 2 * 3^8 = (k * (2 * n + k + 1)) / 2 := by sorry

end NUMINAMATH_CALUDE_max_consecutive_integers_sum_l2437_243709


namespace NUMINAMATH_CALUDE_floor_abs_negative_l2437_243767

theorem floor_abs_negative : ⌊|(-47.6:ℝ)|⌋ = 47 := by sorry

end NUMINAMATH_CALUDE_floor_abs_negative_l2437_243767


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_remainder_l2437_243761

theorem arithmetic_sequence_sum_remainder (n : ℕ) (a d : ℤ) (h : n = 2013) (h1 : a = 105) (h2 : d = 35) :
  (n * (2 * a + (n - 1) * d) / 2) % 12 = 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_remainder_l2437_243761


namespace NUMINAMATH_CALUDE_remaining_budget_calculation_l2437_243739

def total_budget : ℝ := 80000000
def infrastructure_percentage : ℝ := 0.30
def public_transportation : ℝ := 10000000
def healthcare_percentage : ℝ := 0.15

theorem remaining_budget_calculation :
  total_budget - (infrastructure_percentage * total_budget + public_transportation + healthcare_percentage * total_budget) = 34000000 := by
  sorry

end NUMINAMATH_CALUDE_remaining_budget_calculation_l2437_243739


namespace NUMINAMATH_CALUDE_orthocenters_collinear_l2437_243794

-- Define a type for lines in a plane
def Line : Type := ℝ → ℝ → Prop

-- Define a type for points in a plane
def Point : Type := ℝ × ℝ

-- Define a function to check if three points are collinear
def collinear (p q r : Point) : Prop :=
  ∃ (t : ℝ), q.1 - p.1 = t * (r.1 - p.1) ∧ q.2 - p.2 = t * (r.2 - p.2)

-- Define a function to get the intersection point of two lines
noncomputable def intersect (l1 l2 : Line) : Point :=
  sorry

-- Define a function to get the orthocenter of a triangle
noncomputable def orthocenter (a b c : Point) : Point :=
  sorry

-- Main theorem
theorem orthocenters_collinear
  (l1 l2 l3 l4 : Line)
  (p1 p2 p3 p4 p5 p6 : Point)
  (h1 : p1 = intersect l1 l2)
  (h2 : p2 = intersect l1 l3)
  (h3 : p3 = intersect l1 l4)
  (h4 : p4 = intersect l2 l3)
  (h5 : p5 = intersect l2 l4)
  (h6 : p6 = intersect l3 l4)
  : collinear
      (orthocenter p1 p2 p4)
      (orthocenter p1 p3 p5)
      (orthocenter p2 p3 p6) :=
by
  sorry

end NUMINAMATH_CALUDE_orthocenters_collinear_l2437_243794


namespace NUMINAMATH_CALUDE_quadratic_polynomial_special_value_l2437_243729

/-- A quadratic polynomial -/
def QuadraticPolynomial (α : Type*) [Field α] := α → α

/-- Property: [q(x)]^2 - x^2 is divisible by (x - 2)(x + 2)(x - 5) -/
def HasSpecialDivisibility (q : QuadraticPolynomial ℝ) : Prop :=
  ∃ r : ℝ → ℝ, ∀ x : ℝ, (q x)^2 - x^2 = (x - 2) * (x + 2) * (x - 5) * (r x)

theorem quadratic_polynomial_special_value 
  (q : QuadraticPolynomial ℝ) 
  (h : HasSpecialDivisibility q) : 
  q 10 = 110 / 7 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_special_value_l2437_243729


namespace NUMINAMATH_CALUDE_checkerboard_exists_l2437_243724

/-- Represents a cell on the board -/
inductive Cell
| Black
| White

/-- Represents the board -/
def Board := Fin 100 → Fin 100 → Cell

/-- Checks if a cell is adjacent to the border -/
def isBorderAdjacent (i j : Fin 100) : Prop :=
  i = 0 ∨ i = 99 ∨ j = 0 ∨ j = 99

/-- Checks if a 2x2 square is monochromatic -/
def isMonochromatic (board : Board) (i j : Fin 100) : Prop :=
  ∃ c : Cell, 
    board i j = c ∧ board (i+1) j = c ∧ 
    board i (j+1) = c ∧ board (i+1) (j+1) = c

/-- Checks if a 2x2 square has a checkerboard pattern -/
def isCheckerboard (board : Board) (i j : Fin 100) : Prop :=
  (board i j = board (i+1) (j+1) ∧ board (i+1) j = board i (j+1)) ∧
  (board i j ≠ board (i+1) j)

/-- The main theorem -/
theorem checkerboard_exists (board : Board) 
  (border_black : ∀ i j : Fin 100, isBorderAdjacent i j → board i j = Cell.Black)
  (no_monochromatic : ∀ i j : Fin 100, ¬isMonochromatic board i j) :
  ∃ i j : Fin 100, isCheckerboard board i j :=
sorry

end NUMINAMATH_CALUDE_checkerboard_exists_l2437_243724


namespace NUMINAMATH_CALUDE_machine_no_repair_l2437_243712

/-- Represents the state of a portion measuring machine -/
structure PortionMachine where
  max_deviation : ℝ
  nominal_mass : ℝ
  unreadable_deviation_bound : ℝ
  standard_deviation : ℝ

/-- Determines if a portion measuring machine requires repair -/
def requires_repair (m : PortionMachine) : Prop :=
  m.max_deviation > 0.1 * m.nominal_mass ∨
  m.unreadable_deviation_bound ≥ m.max_deviation ∨
  m.standard_deviation > m.max_deviation

/-- Theorem stating that the given machine does not require repair -/
theorem machine_no_repair (m : PortionMachine)
  (h1 : m.max_deviation = 37)
  (h2 : m.max_deviation ≤ 0.1 * m.nominal_mass)
  (h3 : m.unreadable_deviation_bound < m.max_deviation)
  (h4 : m.standard_deviation ≤ m.max_deviation) :
  ¬(requires_repair m) :=
sorry

end NUMINAMATH_CALUDE_machine_no_repair_l2437_243712


namespace NUMINAMATH_CALUDE_number_of_friends_l2437_243700

theorem number_of_friends (total_cards : ℕ) (cards_per_friend : ℕ) (h1 : total_cards = 455) (h2 : cards_per_friend = 91) :
  total_cards / cards_per_friend = 5 := by
  sorry

end NUMINAMATH_CALUDE_number_of_friends_l2437_243700


namespace NUMINAMATH_CALUDE_circle_radius_increase_l2437_243733

theorem circle_radius_increase (r n : ℝ) : 
  r > 0 → r > n → π * (r + n)^2 = 3 * π * r^2 → r = n * (Real.sqrt 3 - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_increase_l2437_243733


namespace NUMINAMATH_CALUDE_student_calculation_l2437_243720

theorem student_calculation (chosen_number : ℕ) : 
  chosen_number = 40 → chosen_number * 7 - 150 = 130 := by
  sorry

end NUMINAMATH_CALUDE_student_calculation_l2437_243720


namespace NUMINAMATH_CALUDE_log_ratio_evaluation_l2437_243784

theorem log_ratio_evaluation : (Real.log 4 / Real.log 3) / (Real.log 8 / Real.log 9) = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_log_ratio_evaluation_l2437_243784


namespace NUMINAMATH_CALUDE_stratified_sample_class_size_l2437_243715

/-- Given two classes with a total of 100 students, if a stratified random sample
    of 10 students contains 4 from one class, then the other class has 60 students. -/
theorem stratified_sample_class_size (total : ℕ) (sample_size : ℕ) (class_a_sample : ℕ) :
  total = 100 →
  sample_size = 10 →
  class_a_sample = 4 →
  (total - (class_a_sample * total / sample_size) : ℕ) = 60 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sample_class_size_l2437_243715


namespace NUMINAMATH_CALUDE_fish_count_ratio_l2437_243798

/-- The ratio of fish counted on day 2 to fish counted on day 1 -/
theorem fish_count_ratio : 
  ∀ (fish_day1 fish_day2 sharks_total : ℕ) 
    (shark_percentage : ℚ),
  fish_day1 = 15 →
  sharks_total = 15 →
  shark_percentage = 1/4 →
  (↑fish_day1 * shark_percentage).floor + 
    (↑fish_day2 * shark_percentage).floor = sharks_total →
  fish_day2 / fish_day1 = 16/5 := by
sorry

end NUMINAMATH_CALUDE_fish_count_ratio_l2437_243798


namespace NUMINAMATH_CALUDE_arithmetic_sequence_nth_term_l2437_243734

theorem arithmetic_sequence_nth_term (a₁ a₂ aₙ n : ℤ) : 
  a₁ = 11 → a₂ = 8 → aₙ = -49 → 
  (∀ k : ℕ, k > 0 → a₁ + (k - 1) * (a₂ - a₁) = aₙ ↔ k = n) →
  n = 21 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_nth_term_l2437_243734


namespace NUMINAMATH_CALUDE_a_eq_one_sufficient_not_necessary_l2437_243771

-- Define the sets P and M
def P : Set ℝ := {x | x > 2}
def M (a : ℝ) : Set ℝ := {x | x > a}

-- Define what it means for a condition to be sufficient
def is_sufficient (a : ℝ) : Prop := P ⊆ M a

-- Define what it means for a condition to be necessary
def is_necessary (a : ℝ) : Prop := ∀ b : ℝ, P ⊆ M b → a ≤ b

-- State the theorem
theorem a_eq_one_sufficient_not_necessary :
  (is_sufficient 1) ∧ ¬(is_necessary 1) := by sorry

end NUMINAMATH_CALUDE_a_eq_one_sufficient_not_necessary_l2437_243771


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2437_243773

def arithmetic_sum (a₁ aₙ : Int) (d : Int) : Int :=
  let n := (aₙ - a₁) / d + 1
  n * (a₁ + aₙ) / 2

theorem arithmetic_sequence_sum :
  arithmetic_sum (-41) 1 2 = -440 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2437_243773


namespace NUMINAMATH_CALUDE_number_division_problem_l2437_243793

theorem number_division_problem : ∃ (n : ℕ), 
  n = 220025 ∧ 
  (n / (555 + 445) = 2 * (555 - 445)) ∧ 
  (n % (555 + 445) = 25) := by
  sorry

end NUMINAMATH_CALUDE_number_division_problem_l2437_243793


namespace NUMINAMATH_CALUDE_work_completion_proof_l2437_243780

/-- The number of days taken by the first group to complete the work -/
def days_first_group : ℕ := 18

/-- The number of men in the second group -/
def men_second_group : ℕ := 9

/-- The number of days taken by the second group to complete the work -/
def days_second_group : ℕ := 72

/-- The number of men in the first group -/
def men_first_group : ℕ := 36

theorem work_completion_proof :
  (men_first_group : ℚ) * days_first_group = men_second_group * days_second_group :=
sorry

end NUMINAMATH_CALUDE_work_completion_proof_l2437_243780


namespace NUMINAMATH_CALUDE_marble_probabilities_and_total_l2437_243727

/-- The probability of drawing a white marble -/
def prob_white : ℚ := 1/4

/-- The probability of drawing a green marble -/
def prob_green : ℚ := 2/7

/-- The probability of drawing either a red or blue marble -/
def prob_red_or_blue : ℚ := 13/28

/-- The total number of marbles in the box -/
def total_marbles : ℕ := 28

/-- Theorem stating that the given probabilities sum to 1 and the total number of marbles is 28 -/
theorem marble_probabilities_and_total : 
  prob_white + prob_green + prob_red_or_blue = 1 ∧ total_marbles = 28 := by sorry

end NUMINAMATH_CALUDE_marble_probabilities_and_total_l2437_243727


namespace NUMINAMATH_CALUDE_average_of_middle_two_l2437_243799

theorem average_of_middle_two (n₁ n₂ n₃ n₄ n₅ n₆ : ℝ) : 
  (n₁ + n₂ + n₃ + n₄ + n₅ + n₆) / 6 = 3.95 →
  (n₁ + n₂) / 2 = 3.6 →
  (n₅ + n₆) / 2 = 4.400000000000001 →
  (n₃ + n₄) / 2 = 3.85 :=
by sorry

end NUMINAMATH_CALUDE_average_of_middle_two_l2437_243799


namespace NUMINAMATH_CALUDE_expression_values_l2437_243759

theorem expression_values (a b : ℝ) (h : (2 * a) / (a + b) + b / (a - b) = 2) :
  (3 * a - b) / (a + 5 * b) = 3 ∨ (3 * a - b) / (a + 5 * b) = 1 :=
by sorry

end NUMINAMATH_CALUDE_expression_values_l2437_243759


namespace NUMINAMATH_CALUDE_sum_a_d_g_equals_six_l2437_243751

-- Define the variables
variable (a b c d e f g : ℤ)

-- State the theorem
theorem sum_a_d_g_equals_six 
  (eq1 : a + b + e = 7)
  (eq2 : b + c + f = 10)
  (eq3 : c + d + g = 6)
  (eq4 : e + f + g = 9) :
  a + d + g = 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_a_d_g_equals_six_l2437_243751


namespace NUMINAMATH_CALUDE_intersection_complement_theorem_l2437_243740

-- Define the sets
def A : Set ℝ := {y | ∃ x, y = x^2}
def B : Set ℝ := {x | x > 3}

-- State the theorem
theorem intersection_complement_theorem :
  A ∩ (Set.univ \ B) = Set.Icc 0 3 := by sorry

end NUMINAMATH_CALUDE_intersection_complement_theorem_l2437_243740


namespace NUMINAMATH_CALUDE_inverse_proportion_problem_l2437_243789

/-- Given two real numbers are inversely proportional, if one is 40 when the other is 5,
    then the first is 25 when the second is 8. -/
theorem inverse_proportion_problem (r s : ℝ) (h : ∃ k : ℝ, r * s = k) 
    (h1 : ∃ r0 : ℝ, r0 * 5 = 40 ∧ r0 * s = r * s) : 
    r * 8 = 25 * s := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_problem_l2437_243789


namespace NUMINAMATH_CALUDE_max_sum_of_squares_l2437_243783

theorem max_sum_of_squares (a b c d : ℝ) : 
  a + b = 17 →
  a * b + c + d = 85 →
  a * d + b * c = 196 →
  c * d = 120 →
  a^2 + b^2 + c^2 + d^2 ≤ 918 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_squares_l2437_243783


namespace NUMINAMATH_CALUDE_goods_lost_percentage_l2437_243736

-- Define the profit percentage
def profit_percentage : ℝ := 0.10

-- Define the loss percentage on selling price
def loss_percentage_on_selling_price : ℝ := 0.12

-- Theorem to prove
theorem goods_lost_percentage (original_value : ℝ) (original_value_positive : original_value > 0) :
  let selling_price := original_value * (1 + profit_percentage)
  let loss_value := selling_price * loss_percentage_on_selling_price
  let goods_lost_percentage := (loss_value / original_value) * 100
  goods_lost_percentage = 13.2 := by
  sorry

end NUMINAMATH_CALUDE_goods_lost_percentage_l2437_243736


namespace NUMINAMATH_CALUDE_expand_expression_l2437_243725

theorem expand_expression (x : ℝ) : (5*x - 3) * (x^3 + 4*x) = 5*x^4 - 3*x^3 + 20*x^2 - 12*x := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l2437_243725


namespace NUMINAMATH_CALUDE_square_root_equation_solution_l2437_243757

/-- Given a positive real number x equal to 3.3333333333333335, prove that the equation
    x * 10 / y = x^2 is satisfied when y = 3. -/
theorem square_root_equation_solution (x : ℝ) (hx : x = 3.3333333333333335) :
  ∃ y : ℝ, y = 3 ∧ x * 10 / y = x^2 := by
  sorry

end NUMINAMATH_CALUDE_square_root_equation_solution_l2437_243757


namespace NUMINAMATH_CALUDE_eve_distance_difference_l2437_243710

def running_intervals : List ℝ := [0.75, 0.85, 0.95]
def walking_intervals : List ℝ := [0.50, 0.65, 0.75, 0.80]

theorem eve_distance_difference :
  (running_intervals.sum - walking_intervals.sum) = -0.15 := by
  sorry

end NUMINAMATH_CALUDE_eve_distance_difference_l2437_243710


namespace NUMINAMATH_CALUDE_translate_line_example_l2437_243762

/-- Represents a line in the form y = mx + b -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Translates a line vertically by a given amount -/
def translate_line (l : Line) (y_shift : ℝ) : Line :=
  { slope := l.slope, intercept := l.intercept + y_shift }

/-- The theorem stating that translating y = 3x - 3 upwards by 5 units results in y = 3x + 2 -/
theorem translate_line_example :
  let original_line : Line := { slope := 3, intercept := -3 }
  let translated_line := translate_line original_line 5
  translated_line = { slope := 3, intercept := 2 } := by
  sorry

end NUMINAMATH_CALUDE_translate_line_example_l2437_243762


namespace NUMINAMATH_CALUDE_pipe_fill_time_l2437_243760

theorem pipe_fill_time (fill_rate_B fill_rate_both : ℝ) 
  (hB : fill_rate_B = 1 / 15)
  (hBoth : fill_rate_both = 1 / 6)
  (hSum : fill_rate_B + (1 / fill_time_A) = fill_rate_both) :
  fill_time_A = 10 := by
  sorry

end NUMINAMATH_CALUDE_pipe_fill_time_l2437_243760


namespace NUMINAMATH_CALUDE_optimal_swap_theorem_l2437_243775

/-- The distance at which car tires should be swapped to wear out equally -/
def optimal_swap_distance : ℝ := 9375

/-- The total distance a front tire can travel before wearing out -/
def front_tire_lifespan : ℝ := 25000

/-- The total distance a rear tire can travel before wearing out -/
def rear_tire_lifespan : ℝ := 15000

/-- Theorem stating that swapping tires at the optimal distance results in equal wear -/
theorem optimal_swap_theorem :
  let remaining_front := (3/5) * (front_tire_lifespan - optimal_swap_distance)
  let remaining_rear := (5/3) * (rear_tire_lifespan - optimal_swap_distance)
  remaining_front = remaining_rear := by sorry

end NUMINAMATH_CALUDE_optimal_swap_theorem_l2437_243775


namespace NUMINAMATH_CALUDE_expression_value_l2437_243708

theorem expression_value (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h_sum : x + y + z = 0) (h_prod : x*y + x*z + y*z ≠ 0) :
  (x^7 + y^7 + z^7) / (x*y*z*(x*y + x*z + y*z)) = -7 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2437_243708


namespace NUMINAMATH_CALUDE_sin_cos_difference_65_35_l2437_243750

theorem sin_cos_difference_65_35 :
  Real.sin (65 * π / 180) * Real.cos (35 * π / 180) -
  Real.cos (65 * π / 180) * Real.sin (35 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_difference_65_35_l2437_243750


namespace NUMINAMATH_CALUDE_remaining_payment_l2437_243764

theorem remaining_payment (deposit : ℝ) (deposit_percentage : ℝ) (h1 : deposit = 80) (h2 : deposit_percentage = 0.1) :
  let total_cost := deposit / deposit_percentage
  total_cost - deposit = 720 := by
sorry

end NUMINAMATH_CALUDE_remaining_payment_l2437_243764


namespace NUMINAMATH_CALUDE_smallest_base_for_perfect_square_l2437_243706

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

theorem smallest_base_for_perfect_square :
  ∀ b : ℕ, b > 6 →
    (∀ k : ℕ, k > 6 ∧ k < b → ¬ is_perfect_square (4 * k + 5)) →
    is_perfect_square (4 * b + 5) →
    b = 11 := by sorry

end NUMINAMATH_CALUDE_smallest_base_for_perfect_square_l2437_243706


namespace NUMINAMATH_CALUDE_truck_rental_theorem_l2437_243738

/-- Represents the number of trucks on a rental lot -/
structure TruckLot where
  monday : ℕ
  rented : ℕ
  returned : ℕ
  saturday : ℕ

/-- Conditions for the truck rental problem -/
def truck_rental_conditions (lot : TruckLot) : Prop :=
  lot.monday = 20 ∧
  lot.rented ≤ 20 ∧
  lot.returned = lot.rented / 2 ∧
  lot.saturday = lot.monday - lot.rented + lot.returned

theorem truck_rental_theorem (lot : TruckLot) :
  truck_rental_conditions lot → lot.saturday = 10 :=
by
  sorry

#check truck_rental_theorem

end NUMINAMATH_CALUDE_truck_rental_theorem_l2437_243738


namespace NUMINAMATH_CALUDE_caleb_picked_less_than_kayla_l2437_243711

/-- The number of apples picked by Kayla -/
def kayla_apples : ℕ := 20

/-- The number of apples picked by Suraya -/
def suraya_apples : ℕ := kayla_apples + 7

/-- The number of apples picked by Caleb -/
def caleb_apples : ℕ := suraya_apples - 12

theorem caleb_picked_less_than_kayla : kayla_apples - caleb_apples = 5 := by
  sorry

end NUMINAMATH_CALUDE_caleb_picked_less_than_kayla_l2437_243711


namespace NUMINAMATH_CALUDE_orange_bags_weight_l2437_243787

/-- If 12 bags of oranges weigh 24 pounds, then 8 bags of oranges weigh 16 pounds. -/
theorem orange_bags_weight (weight_12_bags : ℝ) (h : weight_12_bags = 24) : 
  (8 / 12) * weight_12_bags = 16 := by
  sorry

end NUMINAMATH_CALUDE_orange_bags_weight_l2437_243787


namespace NUMINAMATH_CALUDE_vertical_equality_puzzle_l2437_243753

theorem vertical_equality_puzzle :
  ∃ (a b c d e f g h i j : ℕ),
    a = 1 ∧ b = 9 ∧ c = 8 ∧ d = 5 ∧ e = 4 ∧ f = 0 ∧ g = 6 ∧ h = 7 ∧ i = 2 ∧ j = 3 ∧
    (100 * a + 10 * b + c) - (10 * d + c) = (100 * a + 10 * e + f) ∧
    g * h = 10 * e + i ∧
    (10 * j + j) + (10 * g + d) = 10 * b + c ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧ a ≠ i ∧ a ≠ j ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧ b ≠ i ∧ b ≠ j ∧
    c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧ c ≠ i ∧ c ≠ j ∧
    d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧ d ≠ i ∧ d ≠ j ∧
    e ≠ f ∧ e ≠ g ∧ e ≠ h ∧ e ≠ i ∧ e ≠ j ∧
    f ≠ g ∧ f ≠ h ∧ f ≠ i ∧ f ≠ j ∧
    g ≠ h ∧ g ≠ i ∧ g ≠ j ∧
    h ≠ i ∧ h ≠ j ∧
    i ≠ j :=
by
  sorry

end NUMINAMATH_CALUDE_vertical_equality_puzzle_l2437_243753


namespace NUMINAMATH_CALUDE_f_negative_a_is_zero_l2437_243717

-- Define the function f
def f (x : ℝ) : ℝ := x^3 + x + 1

-- State the theorem
theorem f_negative_a_is_zero (a : ℝ) (h : f a = 2) : f (-a) = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_negative_a_is_zero_l2437_243717


namespace NUMINAMATH_CALUDE_unique_solution_l2437_243716

/-- The set of solutions for the system of equations x + y = 2 and x - y = 0 -/
def solution_set : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 + p.2 = 2 ∧ p.1 - p.2 = 0}

/-- Theorem stating that the solution set contains only the point (1,1) -/
theorem unique_solution :
  solution_set = {(1, 1)} := by sorry

end NUMINAMATH_CALUDE_unique_solution_l2437_243716


namespace NUMINAMATH_CALUDE_no_real_roots_equation_3_l2437_243765

theorem no_real_roots_equation_3 
  (a b c : ℝ) 
  (pos_a : a > 0) (pos_b : b > 0) (pos_c : c > 0)
  (geom_seq : b^2 = a*c)
  (real_roots_1 : a^2 ≥ 4)
  (no_real_roots_2 : b^2 < 8) :
  c^2 < 16 :=
sorry

end NUMINAMATH_CALUDE_no_real_roots_equation_3_l2437_243765


namespace NUMINAMATH_CALUDE_not_necessary_not_sufficient_condition_l2437_243713

theorem not_necessary_not_sufficient_condition (a b : ℝ) : 
  ¬(((a ≠ 5 ∧ b ≠ -5) → (a + b ≠ 0)) ∧ ((a + b ≠ 0) → (a ≠ 5 ∧ b ≠ -5))) := by
  sorry

end NUMINAMATH_CALUDE_not_necessary_not_sufficient_condition_l2437_243713


namespace NUMINAMATH_CALUDE_polynomial_factorization_l2437_243752

theorem polynomial_factorization (x : ℝ) : 
  x^4 - 4*x^3 + 6*x^2 - 4*x + 1 = (x - 1)^4 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l2437_243752


namespace NUMINAMATH_CALUDE_factor_expression_l2437_243758

theorem factor_expression (x y : ℝ) : x * y^2 - 4 * x = x * (y + 2) * (y - 2) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l2437_243758


namespace NUMINAMATH_CALUDE_kay_weight_training_time_l2437_243732

/-- Represents the weekly exercise schedule -/
structure ExerciseSchedule where
  total_time : ℕ
  aerobics_ratio : ℕ
  weight_training_ratio : ℕ

/-- Calculates the time spent on weight training given an exercise schedule -/
def weight_training_time (schedule : ExerciseSchedule) : ℕ :=
  (schedule.total_time * schedule.weight_training_ratio) / (schedule.aerobics_ratio + schedule.weight_training_ratio)

/-- Theorem: Given Kay's exercise schedule, she spends 100 minutes on weight training -/
theorem kay_weight_training_time :
  let kay_schedule : ExerciseSchedule := {
    total_time := 250,
    aerobics_ratio := 3,
    weight_training_ratio := 2
  }
  weight_training_time kay_schedule = 100 := by
  sorry

end NUMINAMATH_CALUDE_kay_weight_training_time_l2437_243732


namespace NUMINAMATH_CALUDE_union_of_A_and_B_complement_of_intersection_A_and_B_l2437_243702

-- Define the sets A and B
def A : Set ℝ := {x | 3 ≤ x ∧ x < 10}
def B : Set ℝ := {x | 2 * x - 8 ≥ 0}

-- Theorem for A ∪ B
theorem union_of_A_and_B : A ∪ B = {x | x ≥ 3} := by sorry

-- Theorem for (A ∩ B)ᶜ
theorem complement_of_intersection_A_and_B : (A ∩ B)ᶜ = {x | x < 4 ∨ x ≥ 10} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_complement_of_intersection_A_and_B_l2437_243702


namespace NUMINAMATH_CALUDE_x_y_existence_l2437_243785

theorem x_y_existence : ∃ (x y : ℝ), x / 7 = 5 / 14 ∧ x / 7 + y = 10 := by
  sorry

end NUMINAMATH_CALUDE_x_y_existence_l2437_243785


namespace NUMINAMATH_CALUDE_decagon_diagonal_intersections_l2437_243743

/-- A regular decagon is a 10-sided polygon -/
def regular_decagon : ℕ := 10

/-- Number of ways to choose k elements from n elements -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of distinct interior intersection points of diagonals in a regular decagon -/
def interior_intersection_points (n : ℕ) : ℕ := choose n 4

theorem decagon_diagonal_intersections :
  interior_intersection_points regular_decagon = 210 :=
sorry

end NUMINAMATH_CALUDE_decagon_diagonal_intersections_l2437_243743


namespace NUMINAMATH_CALUDE_stream_speed_l2437_243703

theorem stream_speed (boat_speed : ℝ) (stream_speed : ℝ) : 
  boat_speed = 39 →
  (1 / (boat_speed - stream_speed)) = (2 * (1 / (boat_speed + stream_speed))) →
  stream_speed = 13 := by
  sorry

end NUMINAMATH_CALUDE_stream_speed_l2437_243703


namespace NUMINAMATH_CALUDE_football_pack_cost_proof_l2437_243728

/-- The cost of a pack of football cards -/
def football_pack_cost : ℝ := 2.73

/-- The number of football card packs bought -/
def football_packs : ℕ := 2

/-- The cost of a pack of Pokemon cards -/
def pokemon_pack_cost : ℝ := 4.01

/-- The cost of a deck of baseball cards -/
def baseball_deck_cost : ℝ := 8.95

/-- The total amount spent on cards -/
def total_spent : ℝ := 18.42

theorem football_pack_cost_proof :
  (football_pack_cost * football_packs) + pokemon_pack_cost + baseball_deck_cost = total_spent := by
  sorry

end NUMINAMATH_CALUDE_football_pack_cost_proof_l2437_243728


namespace NUMINAMATH_CALUDE_expression_evaluation_l2437_243770

theorem expression_evaluation (b : ℚ) (h : b = -3) :
  (3 * b⁻¹ + b⁻¹ / 3) / b = 10 / 27 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2437_243770


namespace NUMINAMATH_CALUDE_b_k_divisible_by_9_count_l2437_243714

/-- The sequence b_n is defined as the number obtained by concatenating
    integers from 1 to n and subtracting n -/
def b (n : ℕ) : ℕ := sorry

/-- g(n) represents the sum of digits of n -/
def g (n : ℕ) : ℕ := sorry

/-- The number of b_k divisible by 9 for 1 ≤ k ≤ 100 -/
def count_divisible_by_9 : ℕ := sorry

theorem b_k_divisible_by_9_count :
  count_divisible_by_9 = 22 := by sorry

end NUMINAMATH_CALUDE_b_k_divisible_by_9_count_l2437_243714


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_sets_l2437_243737

-- Define the quadratic function
def f (a b : ℝ) (x : ℝ) := a * x^2 - b * x - 1

-- Define the solution set of the first inequality
def solution_set (a b : ℝ) := {x : ℝ | f a b x ≥ 0}

-- State the theorem
theorem quadratic_inequality_solution_sets 
  (a b : ℝ) 
  (h1 : solution_set a b = Set.Icc (-1/2) (-1/3)) :
  {x : ℝ | x^2 - b*x - a < 0} = Set.Ioo 2 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_sets_l2437_243737


namespace NUMINAMATH_CALUDE_rug_inner_length_is_four_l2437_243754

/-- Represents a rectangular rug with three nested regions -/
structure Rug where
  inner_width : ℝ
  inner_length : ℝ
  middle_width : ℝ
  middle_length : ℝ
  outer_width : ℝ
  outer_length : ℝ

/-- Calculates the area of a rectangle -/
def area (width : ℝ) (length : ℝ) : ℝ := width * length

/-- Checks if three numbers form an arithmetic progression -/
def isArithmeticProgression (a b c : ℝ) : Prop := b - a = c - b

theorem rug_inner_length_is_four (r : Rug) : 
  r.inner_width = 2 ∧ 
  r.middle_width = r.inner_width + 4 ∧ 
  r.outer_width = r.middle_width + 4 ∧
  r.middle_length = r.inner_length + 4 ∧
  r.outer_length = r.middle_length + 4 ∧
  isArithmeticProgression 
    (area r.inner_width r.inner_length)
    (area r.middle_width r.middle_length - area r.inner_width r.inner_length)
    (area r.outer_width r.outer_length - area r.middle_width r.middle_length) →
  r.inner_length = 4 := by
sorry

end NUMINAMATH_CALUDE_rug_inner_length_is_four_l2437_243754


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l2437_243772

theorem fractional_equation_solution : 
  ∃ x : ℝ, (3 / (x + 1) - 2 / (x - 1) = 0) ∧ (x = 5) := by
  sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l2437_243772


namespace NUMINAMATH_CALUDE_odd_function_sum_l2437_243744

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem odd_function_sum (f : ℝ → ℝ) 
  (h_odd : is_odd f)
  (h_period : ∀ x, f (x + 2) = -f x)
  (h_f1 : f 1 = 8) :
  f 2008 + f 2009 + f 2010 = 8 :=
sorry

end NUMINAMATH_CALUDE_odd_function_sum_l2437_243744


namespace NUMINAMATH_CALUDE_minus_one_power_difference_l2437_243749

theorem minus_one_power_difference : (-1)^2024 - (-1)^2023 = 2 := by
  sorry

end NUMINAMATH_CALUDE_minus_one_power_difference_l2437_243749


namespace NUMINAMATH_CALUDE_larger_sphere_radius_l2437_243726

theorem larger_sphere_radius (r : ℝ) (n : ℕ) (h : r = 2 ∧ n = 6) :
  (n * (4 / 3 * π * r^3) = 4 / 3 * π * (2 * Real.rpow 3 (1/3))^3) :=
by sorry

end NUMINAMATH_CALUDE_larger_sphere_radius_l2437_243726


namespace NUMINAMATH_CALUDE_point_on_y_axis_l2437_243786

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define the y-axis
def on_y_axis (p : Point2D) : Prop := p.x = 0

-- Define our point P in terms of parameter a
def P (a : ℝ) : Point2D := ⟨2*a - 6, a + 1⟩

-- Theorem statement
theorem point_on_y_axis :
  ∃ a : ℝ, on_y_axis (P a) → P a = ⟨0, 4⟩ := by
  sorry

end NUMINAMATH_CALUDE_point_on_y_axis_l2437_243786


namespace NUMINAMATH_CALUDE_polynomial_simplification_l2437_243796

theorem polynomial_simplification (x : ℝ) : 
  (2 * x^6 + 3 * x^5 + 2 * x^4 + x + 15) - (x^6 + 4 * x^5 + 5 * x^4 - 2 * x^3 + 20) = 
  x^6 - x^5 - 3 * x^4 + 2 * x^3 + x - 5 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l2437_243796


namespace NUMINAMATH_CALUDE_symmetric_point_on_x_axis_l2437_243779

/-- Given a point P(m-1, m+1) that lies on the x-axis, 
    prove that its symmetric point with respect to the x-axis has coordinates (-2, 0) -/
theorem symmetric_point_on_x_axis (m : ℝ) :
  (m + 1 = 0) →  -- P lies on the x-axis
  ((-2 : ℝ), (0 : ℝ)) = (m - 1, -(m + 1)) :=
by sorry

end NUMINAMATH_CALUDE_symmetric_point_on_x_axis_l2437_243779


namespace NUMINAMATH_CALUDE_min_triangle_area_l2437_243721

/-- A point in the 2D Cartesian plane with integer coordinates -/
structure IntPoint where
  x : ℤ
  y : ℤ

/-- Definition of a rectangle OABC with O at origin and B at (9, 8) -/
def rectangle : Set IntPoint :=
  {p : IntPoint | 0 ≤ p.x ∧ p.x ≤ 9 ∧ 0 ≤ p.y ∧ p.y ≤ 8}

/-- Area of triangle OBX given point X -/
def triangleArea (X : IntPoint) : ℚ :=
  (1 / 2 : ℚ) * |9 * X.y - 8 * X.x|

/-- Theorem stating the minimum area of triangle OBX -/
theorem min_triangle_area :
  ∃ (min_area : ℚ), min_area = 1/2 ∧
  ∀ (X : IntPoint), X ∈ rectangle → triangleArea X ≥ min_area :=
sorry

end NUMINAMATH_CALUDE_min_triangle_area_l2437_243721


namespace NUMINAMATH_CALUDE_min_value_of_expression_l2437_243746

theorem min_value_of_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h : x * y * z * (x + y + z) = 1) : 
  (x + y) * (y + z) ≥ 2 ∧ ∃ (x' y' z' : ℝ), x' > 0 ∧ y' > 0 ∧ z' > 0 ∧ 
    x' * y' * z' * (x' + y' + z') = 1 ∧ (x' + y') * (y' + z') = 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l2437_243746


namespace NUMINAMATH_CALUDE_cauchy_schwarz_iag_equivalence_l2437_243745

theorem cauchy_schwarz_iag_equivalence :
  (∀ (a b c d : ℝ), (a * c + b * d)^2 ≤ (a^2 + b^2) * (c^2 + d^2)) ↔
  (∀ (x y : ℝ), x ≥ 0 → y ≥ 0 → Real.sqrt (x * y) ≤ (x + y) / 2) :=
by sorry

end NUMINAMATH_CALUDE_cauchy_schwarz_iag_equivalence_l2437_243745


namespace NUMINAMATH_CALUDE_student_correct_answers_l2437_243766

/-- Represents a multiple-choice test with scoring rules -/
structure MCTest where
  total_questions : ℕ
  correct_points : ℕ
  incorrect_points : ℕ

/-- Represents a student's test result -/
structure TestResult where
  test : MCTest
  total_score : ℤ

/-- Calculates the number of correctly answered questions -/
def correct_answers (result : TestResult) : ℕ :=
  sorry

/-- Theorem stating the problem and its solution -/
theorem student_correct_answers
  (test : MCTest)
  (result : TestResult)
  (h1 : test.total_questions = 25)
  (h2 : test.correct_points = 4)
  (h3 : test.incorrect_points = 1)
  (h4 : result.test = test)
  (h5 : result.total_score = 85) :
  correct_answers result = 22 := by
  sorry

end NUMINAMATH_CALUDE_student_correct_answers_l2437_243766


namespace NUMINAMATH_CALUDE_divisible_by_three_l2437_243731

theorem divisible_by_three (n : ℕ) : 
  (3 ∣ n * 2^n + 1) ↔ (n % 6 = 1 ∨ n % 6 = 2) :=
by sorry

end NUMINAMATH_CALUDE_divisible_by_three_l2437_243731


namespace NUMINAMATH_CALUDE_dodgeball_assistant_count_l2437_243748

theorem dodgeball_assistant_count :
  ∀ (total_students : ℕ) (boys girls : ℕ),
    total_students = 27 →
    boys + girls < total_students →
    boys % 4 = 0 →
    girls % 6 = 0 →
    boys / 2 + girls / 3 = girls / 2 + boys / 4 →
    (total_students - (boys + girls) = 7) ∨
    (total_students - (boys + girls) = 17) :=
by sorry

end NUMINAMATH_CALUDE_dodgeball_assistant_count_l2437_243748


namespace NUMINAMATH_CALUDE_bart_firewood_consumption_l2437_243723

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

end NUMINAMATH_CALUDE_bart_firewood_consumption_l2437_243723


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_for_m_3_find_m_for_given_intersection_l2437_243774

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 4*x - 5 ≤ 0}
def B (m : ℝ) : Set ℝ := {x | x^2 - 2*x - m < 0}

-- Part 1
theorem intersection_A_complement_B_for_m_3 : 
  A ∩ (Set.univ \ B 3) = {x | x = -1 ∨ (3 ≤ x ∧ x ≤ 5)} := by sorry

-- Part 2
theorem find_m_for_given_intersection :
  ∃ m : ℝ, A ∩ B m = {x | -1 ≤ x ∧ x < 4} ∧ m = 8 := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_for_m_3_find_m_for_given_intersection_l2437_243774


namespace NUMINAMATH_CALUDE_stone_162_is_12_l2437_243701

/-- The number of stones in the circular arrangement -/
def n : ℕ := 15

/-- The count we're interested in -/
def target_count : ℕ := 162

/-- The function that maps a count to its corresponding stone number -/
def stone_number (count : ℕ) : ℕ := 
  if count % n = 0 then n else count % n

theorem stone_162_is_12 : stone_number target_count = 12 := by
  sorry

end NUMINAMATH_CALUDE_stone_162_is_12_l2437_243701


namespace NUMINAMATH_CALUDE_q_zero_value_l2437_243705

-- Define polynomials p, q, and r
variable (p q r : ℝ → ℝ)

-- Define the relationship between p, q, and r
axiom relation : ∀ x, r x = p x * q x + 2

-- Define the constant terms of p and r
axiom p_constant : p 0 = 6
axiom r_constant : r 0 = 5

-- Theorem to prove
theorem q_zero_value : q 0 = 1/2 := by sorry

end NUMINAMATH_CALUDE_q_zero_value_l2437_243705


namespace NUMINAMATH_CALUDE_f_negative_a_l2437_243707

noncomputable def f (x : ℝ) : ℝ := x^2 + Real.log (x + Real.sqrt (x^2 + 1)) / Real.log 10

theorem f_negative_a (a M : ℝ) (h : f a = M) : f (-a) = 2 * a^2 - M := by
  sorry

end NUMINAMATH_CALUDE_f_negative_a_l2437_243707


namespace NUMINAMATH_CALUDE_unique_positive_solution_exists_distinct_real_solution_l2437_243755

-- Define the system of equations
def equation_system (x y z : ℝ) : Prop :=
  x * y + y * z + z * x = 12 ∧ x * y * z - x - y - z = 2

-- Theorem for unique positive solution
theorem unique_positive_solution :
  ∃! (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧ equation_system x y z ∧ x = 2 ∧ y = 2 ∧ z = 2 :=
sorry

-- Theorem for existence of distinct real solution
theorem exists_distinct_real_solution :
  ∃ (x y z : ℝ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ equation_system x y z :=
sorry

end NUMINAMATH_CALUDE_unique_positive_solution_exists_distinct_real_solution_l2437_243755


namespace NUMINAMATH_CALUDE_min_students_for_given_data_l2437_243718

/-- Represents the number of students receiving A's on each day of the week -/
structure GradeData where
  monday : Nat
  tuesday : Nat
  wednesday : Nat
  thursday : Nat
  friday : Nat

/-- The minimum number of students in the class given the grade data -/
def minStudents (data : GradeData) : Nat :=
  max (data.monday + data.tuesday)
    (max (data.tuesday + data.wednesday)
      (max (data.wednesday + data.thursday)
        (data.thursday + data.friday)))

/-- Theorem stating the minimum number of students given the specific grade data -/
theorem min_students_for_given_data :
  let data : GradeData := {
    monday := 5,
    tuesday := 8,
    wednesday := 6,
    thursday := 4,
    friday := 9
  }
  minStudents data = 14 := by sorry

end NUMINAMATH_CALUDE_min_students_for_given_data_l2437_243718


namespace NUMINAMATH_CALUDE_min_bottles_to_fill_jumbo_l2437_243788

def jumbo_capacity : ℕ := 1200
def regular_capacity : ℕ := 75
def mini_capacity : ℕ := 50

theorem min_bottles_to_fill_jumbo :
  (jumbo_capacity / regular_capacity = 16 ∧ jumbo_capacity % regular_capacity = 0) ∧
  (jumbo_capacity / mini_capacity = 24 ∧ jumbo_capacity % mini_capacity = 0) :=
by sorry

end NUMINAMATH_CALUDE_min_bottles_to_fill_jumbo_l2437_243788


namespace NUMINAMATH_CALUDE_max_value_n_is_3210_l2437_243742

/-- S(a) represents the sum of the digits of a natural number a -/
def S (a : ℕ) : ℕ := sorry

/-- allDigitsDifferent n is true if all digits of n are different -/
def allDigitsDifferent (n : ℕ) : Prop := sorry

/-- maxValueN is the maximum value of n satisfying the given conditions -/
def maxValueN : ℕ := 3210

theorem max_value_n_is_3210 :
  ∀ n : ℕ, allDigitsDifferent n → S (3 * n) = 3 * S n → n ≤ maxValueN := by
  sorry

end NUMINAMATH_CALUDE_max_value_n_is_3210_l2437_243742


namespace NUMINAMATH_CALUDE_fruit_mix_problem_l2437_243747

theorem fruit_mix_problem (total : ℕ) (apples oranges bananas plums : ℕ) : 
  total = 240 →
  oranges = 3 * apples →
  bananas = 2 * oranges →
  plums = 5 * bananas →
  total = apples + oranges + bananas + plums →
  apples = 6 := by
sorry

end NUMINAMATH_CALUDE_fruit_mix_problem_l2437_243747


namespace NUMINAMATH_CALUDE_optimal_carriages_and_passengers_l2437_243735

/-- The daily round trips as a function of the number of carriages -/
def daily_trips (x : ℕ) : ℝ :=
  -3 * x + 28

/-- The daily operating number of passengers as a function of the number of carriages -/
def daily_passengers (x : ℕ) : ℝ :=
  110 * x * daily_trips x

/-- The set of valid carriage numbers -/
def valid_carriages : Set ℕ :=
  {x | 1 ≤ x ∧ x ≤ 9}

theorem optimal_carriages_and_passengers :
  ∀ x ∈ valid_carriages,
    daily_passengers 5 ≥ daily_passengers x ∧
    daily_passengers 5 = 14300 :=
by sorry

end NUMINAMATH_CALUDE_optimal_carriages_and_passengers_l2437_243735


namespace NUMINAMATH_CALUDE_fliers_calculation_l2437_243730

theorem fliers_calculation (initial_fliers : ℕ) : 
  (initial_fliers : ℚ) * (9/10) * (3/4) = 1350 →
  initial_fliers = 2000 :=
by sorry

end NUMINAMATH_CALUDE_fliers_calculation_l2437_243730


namespace NUMINAMATH_CALUDE_manny_marbles_l2437_243741

theorem manny_marbles (total_marbles : ℕ) (marbles_per_pack : ℕ) (kept_packs : ℕ) (neil_fraction : ℚ) :
  total_marbles = 400 →
  marbles_per_pack = 10 →
  kept_packs = 25 →
  neil_fraction = 1/8 →
  let total_packs := total_marbles / marbles_per_pack
  let given_packs := total_packs - kept_packs
  let neil_packs := neil_fraction * total_packs
  let manny_packs := given_packs - neil_packs
  manny_packs / total_packs = 1/4 := by sorry

end NUMINAMATH_CALUDE_manny_marbles_l2437_243741


namespace NUMINAMATH_CALUDE_probability_intersection_l2437_243722

theorem probability_intersection (A B : ℝ) (union : ℝ) (h1 : 0 ≤ A ∧ A ≤ 1) (h2 : 0 ≤ B ∧ B ≤ 1) (h3 : 0 ≤ union ∧ union ≤ 1) :
  ∃ intersection : ℝ, 0 ≤ intersection ∧ intersection ≤ 1 ∧ union = A + B - intersection :=
by sorry

end NUMINAMATH_CALUDE_probability_intersection_l2437_243722


namespace NUMINAMATH_CALUDE_no_squares_end_in_seven_l2437_243776

theorem no_squares_end_in_seven : 
  ∀ n : ℕ, ¬(∃ m : ℕ, m * m = 10 * n + 7) :=
by
  sorry

end NUMINAMATH_CALUDE_no_squares_end_in_seven_l2437_243776


namespace NUMINAMATH_CALUDE_defective_products_m1_l2437_243763

theorem defective_products_m1 (m1_production m2_production m3_production : ℝ)
  (m2_defective m3_defective : ℝ) (non_defective_total : ℝ) :
  m1_production = 25 ∧ 
  m2_production = 35 ∧ 
  m3_production = 40 ∧ 
  m2_defective = 4 ∧ 
  m3_defective = 5 ∧ 
  non_defective_total = 96.1 →
  (100 - non_defective_total - (m2_production * m2_defective / 100 + m3_production * m3_defective / 100)) / m1_production * 100 = 2 := by
  sorry

end NUMINAMATH_CALUDE_defective_products_m1_l2437_243763


namespace NUMINAMATH_CALUDE_possible_x_values_l2437_243719

def A (x y : ℕ+) : ℕ := x^2 + y^2 + 2*x - 2*y + 2

def B (x : ℕ+) : ℤ := x^2 - 5*x + 5

theorem possible_x_values :
  ∀ x y : ℕ+, (B x)^(A x y) = 1 → x ∈ ({1, 2, 3, 4} : Set ℕ+) :=
sorry

end NUMINAMATH_CALUDE_possible_x_values_l2437_243719


namespace NUMINAMATH_CALUDE_deans_height_l2437_243795

theorem deans_height (depth water_depth : ℝ) (h1 : water_depth = 10 * depth) (h2 : water_depth = depth + 81) : depth = 9 := by
  sorry

end NUMINAMATH_CALUDE_deans_height_l2437_243795


namespace NUMINAMATH_CALUDE_a_plus_b_value_l2437_243756

theorem a_plus_b_value (a b : ℝ) 
  (h1 : |(-a)| = |(-1)|) 
  (h2 : b^2 = 9)
  (h3 : |a - b| = b - a) : 
  a + b = 2 ∨ a + b = 4 := by
sorry

end NUMINAMATH_CALUDE_a_plus_b_value_l2437_243756


namespace NUMINAMATH_CALUDE_calorie_difference_per_dollar_l2437_243778

/-- Calculates the difference in calories per dollar between burgers and burritos -/
theorem calorie_difference_per_dollar : 
  let burrito_count : ℕ := 10
  let burrito_price : ℚ := 6
  let burrito_calories : ℕ := 120
  let burger_count : ℕ := 5
  let burger_price : ℚ := 8
  let burger_calories : ℕ := 400
  let burrito_calories_per_dollar := (burrito_count * burrito_calories : ℚ) / burrito_price
  let burger_calories_per_dollar := (burger_count * burger_calories : ℚ) / burger_price
  burger_calories_per_dollar - burrito_calories_per_dollar = 50
:= by sorry


end NUMINAMATH_CALUDE_calorie_difference_per_dollar_l2437_243778


namespace NUMINAMATH_CALUDE_monthly_bill_increase_l2437_243792

theorem monthly_bill_increase (original_bill : ℝ) (increase_percentage : ℝ) : 
  original_bill = 60 →
  increase_percentage = 0.30 →
  original_bill + (increase_percentage * original_bill) = 78 := by
  sorry

end NUMINAMATH_CALUDE_monthly_bill_increase_l2437_243792


namespace NUMINAMATH_CALUDE_car_acceleration_at_one_second_l2437_243791

-- Define the velocity function
def v (t : ℝ) : ℝ := -t^2 + 10*t

-- Define the acceleration function as the derivative of velocity
def a (t : ℝ) : ℝ := -2*t + 10

-- Theorem statement
theorem car_acceleration_at_one_second :
  a 1 = 8 := by
  sorry

end NUMINAMATH_CALUDE_car_acceleration_at_one_second_l2437_243791


namespace NUMINAMATH_CALUDE_mathlon_solution_l2437_243797

/-- A Mathlon competition with M events and three participants -/
structure Mathlon where
  M : ℕ
  p₁ : ℕ
  p₂ : ℕ
  p₃ : ℕ
  scoreA : ℕ
  scoreB : ℕ
  scoreC : ℕ
  B_won_100m : Bool

/-- The conditions of the Mathlon problem -/
def mathlon_conditions (m : Mathlon) : Prop :=
  m.M > 0 ∧
  m.p₁ > m.p₂ ∧ m.p₂ > m.p₃ ∧ m.p₃ > 0 ∧
  m.scoreA = 22 ∧ m.scoreB = 9 ∧ m.scoreC = 9 ∧
  m.B_won_100m = true

/-- The theorem to prove -/
theorem mathlon_solution (m : Mathlon) (h : mathlon_conditions m) : 
  m.M = 5 ∧ ∃ (events : Fin m.M → Fin 3), 
    (∃ i, events i = 1) ∧  -- B wins one event (100m)
    (∃ i, events i = 2)    -- C is second in one event (high jump)
    := by sorry

end NUMINAMATH_CALUDE_mathlon_solution_l2437_243797


namespace NUMINAMATH_CALUDE_persimmon_basket_weight_l2437_243768

theorem persimmon_basket_weight (total_weight half_weight : ℝ)
  (h1 : total_weight = 62)
  (h2 : half_weight = 34)
  (h3 : ∃ (basket_weight persimmon_weight : ℝ),
    basket_weight + persimmon_weight = total_weight ∧
    basket_weight + persimmon_weight / 2 = half_weight) :
  ∃ (basket_weight : ℝ), basket_weight = 6 := by
sorry

end NUMINAMATH_CALUDE_persimmon_basket_weight_l2437_243768


namespace NUMINAMATH_CALUDE_fraction_evaluation_l2437_243781

theorem fraction_evaluation : (20 + 15) / (30 - 25) = 7 := by
  sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l2437_243781


namespace NUMINAMATH_CALUDE_base_conversion_sum_l2437_243769

-- Define a function to convert a number from base 8 to base 10
def base8To10 (n : Nat) : Nat :=
  let hundreds := n / 100
  let tens := (n % 100) / 10
  let ones := n % 10
  hundreds * 8^2 + tens * 8^1 + ones * 8^0

-- Define a function to convert a number from base 13 to base 10
def base13To10 (n : Nat) (c : Nat) : Nat :=
  let hundreds := n / 100
  let tens := (n % 100) / 10
  let ones := n % 10
  hundreds * 13^2 + c * 13^1 + ones * 13^0

theorem base_conversion_sum :
  base8To10 537 + base13To10 405 12 = 1188 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_sum_l2437_243769


namespace NUMINAMATH_CALUDE_tan_theta_minus_pi_fourth_l2437_243704

/-- Given that θ is in the fourth quadrant and sin(θ + π/4) = 5/13, 
    prove that tan(θ - π/4) = -12/5 -/
theorem tan_theta_minus_pi_fourth (θ : Real) 
  (h1 : π < θ ∧ θ < 2*π) -- θ is in the fourth quadrant
  (h2 : Real.sin (θ + π/4) = 5/13) : 
  Real.tan (θ - π/4) = -12/5 := by
sorry

end NUMINAMATH_CALUDE_tan_theta_minus_pi_fourth_l2437_243704


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l2437_243790

theorem simplify_and_evaluate (a b : ℚ) (h1 : a = -1/3) (h2 : b = -2) :
  ((3*a + b)^2 - (3*a + b)*(3*a - b)) / (2*b) = -3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l2437_243790


namespace NUMINAMATH_CALUDE_f_symmetry_l2437_243777

-- Define the function f
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^5 - b * x^3 + c * x - 3

-- State the theorem
theorem f_symmetry (a b c : ℝ) : f a b c (-3) = 7 → f a b c 3 = -13 := by
  sorry

end NUMINAMATH_CALUDE_f_symmetry_l2437_243777


namespace NUMINAMATH_CALUDE_product_726_4_base9_l2437_243782

/-- Convert a base-9 number represented as a list of digits to a natural number. -/
def base9ToNat (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => 9 * acc + d) 0

/-- Convert a natural number to its base-9 representation as a list of digits. -/
def natToBase9 (n : Nat) : List Nat :=
  if n < 9 then [n]
  else (n % 9) :: natToBase9 (n / 9)

/-- Theorem stating that the product of 726₉ and 4₉ is equal to 3216₉ in base 9. -/
theorem product_726_4_base9 :
  base9ToNat [6, 2, 7] * base9ToNat [4] = base9ToNat [6, 1, 2, 3] := by
  sorry

#eval base9ToNat [6, 2, 7] * base9ToNat [4] == base9ToNat [6, 1, 2, 3]

end NUMINAMATH_CALUDE_product_726_4_base9_l2437_243782
