import Mathlib

namespace NUMINAMATH_CALUDE_simultaneous_completion_time_specific_completion_time_l2298_229803

/-- The time taken for two machines to complete an order when working simultaneously, 
    given their individual completion times. -/
theorem simultaneous_completion_time (t1 t2 : ℝ) (h1 : t1 > 0) (h2 : t2 > 0) : 
  (t1 * t2) / (t1 + t2) = (t1 * t2) / ((t1 * t2) * (1 / t1 + 1 / t2)) := by
  sorry

/-- Proof that two machines with completion times of 9 hours and 8 hours respectively
    will take 72/17 hours to complete the order when working simultaneously. -/
theorem specific_completion_time : 
  (9 : ℝ) * 8 / (9 + 8) = 72 / 17 := by
  sorry

end NUMINAMATH_CALUDE_simultaneous_completion_time_specific_completion_time_l2298_229803


namespace NUMINAMATH_CALUDE_pyramid_volume_l2298_229851

/-- The volume of a triangular pyramid with an equilateral base of side length 6√3 and height 9 is 81√3 -/
theorem pyramid_volume : 
  let s : ℝ := 6 * Real.sqrt 3
  let base_area : ℝ := (Real.sqrt 3 / 4) * s^2
  let height : ℝ := 9
  let volume : ℝ := (1/3) * base_area * height
  volume = 81 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_pyramid_volume_l2298_229851


namespace NUMINAMATH_CALUDE_unique_number_with_seven_coprimes_l2298_229868

def connection (a b : ℕ) : ℚ :=
  (Nat.lcm a b : ℚ) / (a * b : ℚ)

def isCoprimeWithExactlyN (x n : ℕ) : Prop :=
  (∃ (S : Finset ℕ), S.card = n ∧ 
    (∀ y ∈ S, y < 20 ∧ connection x y = 1) ∧
    (∀ y < 20, y ∉ S → connection x y ≠ 1))

theorem unique_number_with_seven_coprimes :
  ∃! x, isCoprimeWithExactlyN x 7 :=
sorry

end NUMINAMATH_CALUDE_unique_number_with_seven_coprimes_l2298_229868


namespace NUMINAMATH_CALUDE_john_climbs_nine_flights_l2298_229843

/-- The number of flights climbed given step height, flight height, and number of steps -/
def flights_climbed (step_height_inches : ℚ) (flight_height_feet : ℚ) (num_steps : ℕ) : ℚ :=
  (step_height_inches / 12 * num_steps) / flight_height_feet

/-- Theorem: John climbs 9 flights of stairs -/
theorem john_climbs_nine_flights :
  flights_climbed 18 10 60 = 9 := by
  sorry

end NUMINAMATH_CALUDE_john_climbs_nine_flights_l2298_229843


namespace NUMINAMATH_CALUDE_three_students_same_group_l2298_229844

/-- The number of students in the school -/
def total_students : ℕ := 900

/-- The number of lunch groups -/
def num_groups : ℕ := 4

/-- The size of each lunch group -/
def group_size : ℕ := total_students / num_groups

/-- The probability of three specific students being in the same lunch group -/
def prob_same_group : ℚ := 1 / 16

theorem three_students_same_group :
  let n := total_students
  let k := num_groups
  let g := group_size
  prob_same_group = (g / n) * ((g - 1) / (n - 1)) * ((g - 2) / (n - 2)) :=
sorry

end NUMINAMATH_CALUDE_three_students_same_group_l2298_229844


namespace NUMINAMATH_CALUDE_grape_problem_l2298_229800

theorem grape_problem (x : ℕ) : x > 100 ∧ 
                                x % 3 = 1 ∧ 
                                x % 5 = 2 ∧ 
                                x % 7 = 4 → 
                                x ≤ 172 :=
by sorry

end NUMINAMATH_CALUDE_grape_problem_l2298_229800


namespace NUMINAMATH_CALUDE_earnings_difference_is_400_l2298_229865

/-- Represents the amount of jade Nancy has in grams -/
def total_jade : ℕ := 1920

/-- Represents the amount of jade needed for a giraffe statue in grams -/
def giraffe_jade : ℕ := 120

/-- Represents the price of a giraffe statue in dollars -/
def giraffe_price : ℕ := 150

/-- Represents the amount of jade needed for an elephant statue in grams -/
def elephant_jade : ℕ := 2 * giraffe_jade

/-- Represents the price of an elephant statue in dollars -/
def elephant_price : ℕ := 350

/-- Calculates the earnings difference between making all elephant statues
    and all giraffe statues from the total jade -/
def earnings_difference : ℕ :=
  (total_jade / elephant_jade) * elephant_price - (total_jade / giraffe_jade) * giraffe_price

theorem earnings_difference_is_400 : earnings_difference = 400 := by
  sorry

end NUMINAMATH_CALUDE_earnings_difference_is_400_l2298_229865


namespace NUMINAMATH_CALUDE_expression_evaluation_l2298_229899

theorem expression_evaluation :
  let x : ℚ := -3
  let numerator := 7 + x * (4 + x) - 4^2
  let denominator := x - 4 + x^2
  numerator / denominator = -6 :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2298_229899


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l2298_229896

theorem arithmetic_mean_of_fractions :
  (1 / 2 : ℚ) * ((2 / 5 : ℚ) + (4 / 7 : ℚ)) = (17 / 35 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l2298_229896


namespace NUMINAMATH_CALUDE_triangle_area_l2298_229855

theorem triangle_area (A B C : Real) (a b c : Real) : 
  -- Triangle ABC exists with sides a, b, c opposite to angles A, B, C
  (0 < A ∧ A < π) →
  (0 < B ∧ B < π) →
  (0 < C ∧ C < π) →
  (A + B + C = π) →
  (a > 0 ∧ b > 0 ∧ c > 0) →
  (a + b > c ∧ b + c > a ∧ c + a > b) →
  -- Given conditions
  (Real.cos B = 1/4) →
  (b = 3) →
  (Real.sin C = 2 * Real.sin A) →
  -- Conclusion
  (1/2 * a * c * Real.sin B = Real.sqrt 15 / 4) := by
sorry

end NUMINAMATH_CALUDE_triangle_area_l2298_229855


namespace NUMINAMATH_CALUDE_expected_lotus_seed_is_three_l2298_229871

/-- The total number of zongzi -/
def total_zongzi : ℕ := 180

/-- The number of lotus seed zongzi -/
def lotus_seed_zongzi : ℕ := 54

/-- The size of the random sample -/
def sample_size : ℕ := 10

/-- The expected number of lotus seed zongzi in the sample -/
def expected_lotus_seed : ℚ := (sample_size : ℚ) * (lotus_seed_zongzi : ℚ) / (total_zongzi : ℚ)

theorem expected_lotus_seed_is_three :
  expected_lotus_seed = 3 := by sorry

end NUMINAMATH_CALUDE_expected_lotus_seed_is_three_l2298_229871


namespace NUMINAMATH_CALUDE_no_negative_sum_of_squares_l2298_229815

theorem no_negative_sum_of_squares : ¬∃ (x y : ℝ), x^2 + y^2 < 0 := by
  sorry

end NUMINAMATH_CALUDE_no_negative_sum_of_squares_l2298_229815


namespace NUMINAMATH_CALUDE_friend_receives_fifty_boxes_l2298_229832

/-- Calculates the number of boxes each friend receives after distribution -/
def boxes_per_friend (initial_order : ℕ) (extra_percentage : ℚ) 
  (given_to_brother : ℕ) (given_to_sister : ℕ) (given_to_cousin : ℕ) 
  (additional_request : ℕ) (num_friends : ℕ) : ℚ :=
  let total_boxes := initial_order + (extra_percentage * initial_order)
  let given_away := given_to_brother + given_to_sister + given_to_cousin + additional_request
  let remaining_boxes := total_boxes - given_away
  remaining_boxes / num_friends

/-- Theorem stating that each friend receives 50 boxes -/
theorem friend_receives_fifty_boxes :
  boxes_per_friend 180 (40/100) 35 28 22 17 3 = 50 := by
  sorry

end NUMINAMATH_CALUDE_friend_receives_fifty_boxes_l2298_229832


namespace NUMINAMATH_CALUDE_coin_division_l2298_229881

theorem coin_division (n : ℕ) : 
  (n > 0) →
  (n % 6 = 4) → 
  (n % 5 = 3) → 
  (∀ m : ℕ, m > 0 ∧ m < n → (m % 6 ≠ 4 ∨ m % 5 ≠ 3)) →
  (n % 7 = 0) := by
sorry

end NUMINAMATH_CALUDE_coin_division_l2298_229881


namespace NUMINAMATH_CALUDE_derivative_of_y_l2298_229898

-- Define the function y
def y (x : ℝ) : ℝ := 2 * x^2 - 2 * x + 1

-- State the theorem
theorem derivative_of_y (x : ℝ) : 
  deriv y x = 4 * x - 2 := by sorry

end NUMINAMATH_CALUDE_derivative_of_y_l2298_229898


namespace NUMINAMATH_CALUDE_simplify_expressions_l2298_229802

variable (a b t : ℝ)

theorem simplify_expressions :
  (6 * a^2 - 2 * a * b - 2 * (3 * a^2 - (1/2) * a * b) = -a * b) ∧
  (-(t^2 - t - 1) + (2 * t^2 - 3 * t + 1) = t^2 - 2 * t + 2) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expressions_l2298_229802


namespace NUMINAMATH_CALUDE_triangle_side_length_l2298_229817

theorem triangle_side_length 
  (a : ℝ) 
  (C : ℝ) 
  (S : ℝ) :
  a = 3 * Real.sqrt 2 →
  Real.cos C = 1 / 3 →
  S = 4 * Real.sqrt 3 →
  ∃ (b : ℝ), b = 2 * Real.sqrt 3 ∧ 
    S = 1 / 2 * a * b * Real.sin C :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2298_229817


namespace NUMINAMATH_CALUDE_min_even_integers_l2298_229889

theorem min_even_integers (x y z a b c m n o : ℤ) : 
  x + y + z = 30 →
  x + y + z + a + b + c = 55 →
  x + y + z + a + b + c + m + n + o = 88 →
  ∃ (count : ℕ), count = (if Even x then 1 else 0) + 
                         (if Even y then 1 else 0) + 
                         (if Even z then 1 else 0) + 
                         (if Even a then 1 else 0) + 
                         (if Even b then 1 else 0) + 
                         (if Even c then 1 else 0) + 
                         (if Even m then 1 else 0) + 
                         (if Even n then 1 else 0) + 
                         (if Even o then 1 else 0) ∧
  count ≥ 1 ∧
  ∀ (other_count : ℕ), other_count ≥ count →
    ∃ (x' y' z' a' b' c' m' n' o' : ℤ),
      x' + y' + z' = 30 ∧
      x' + y' + z' + a' + b' + c' = 55 ∧
      x' + y' + z' + a' + b' + c' + m' + n' + o' = 88 ∧
      other_count = (if Even x' then 1 else 0) + 
                    (if Even y' then 1 else 0) + 
                    (if Even z' then 1 else 0) + 
                    (if Even a' then 1 else 0) + 
                    (if Even b' then 1 else 0) + 
                    (if Even c' then 1 else 0) + 
                    (if Even m' then 1 else 0) + 
                    (if Even n' then 1 else 0) + 
                    (if Even o' then 1 else 0) :=
by
  sorry

end NUMINAMATH_CALUDE_min_even_integers_l2298_229889


namespace NUMINAMATH_CALUDE_parallelogram_area_is_fifteen_l2298_229821

/-- Represents a parallelogram EFGH with base FG and height FH -/
structure Parallelogram where
  base : ℝ
  height : ℝ

/-- Calculates the area of a parallelogram -/
def area (p : Parallelogram) : ℝ := p.base * p.height

/-- The theorem stating that the area of the given parallelogram EFGH is 15 -/
theorem parallelogram_area_is_fifteen : ∃ (p : Parallelogram), p.base = 5 ∧ p.height = 3 ∧ area p = 15 := by
  sorry


end NUMINAMATH_CALUDE_parallelogram_area_is_fifteen_l2298_229821


namespace NUMINAMATH_CALUDE_max_value_of_f_l2298_229839

open Real

noncomputable def f (x : ℝ) : ℝ := 
  Real.sqrt (x^4 - 3*x^2 - 6*x + 13) - Real.sqrt (x^4 - x^2 + 1)

theorem max_value_of_f : 
  ∃ (M : ℝ), (∀ x, f x ≤ M) ∧ (∃ x, f x = M) ∧ M = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_f_l2298_229839


namespace NUMINAMATH_CALUDE_polynomial_expansion_l2298_229858

theorem polynomial_expansion :
  ∀ x : ℝ, (3 * x - 2) * (2 * x^2 + 4 * x - 6) = 6 * x^3 + 8 * x^2 - 26 * x + 12 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l2298_229858


namespace NUMINAMATH_CALUDE_cuboid_diagonal_l2298_229864

theorem cuboid_diagonal (a b : ℤ) :
  (∃ c : ℕ+, ∃ d : ℕ+, d * d = a * a + b * b + c * c) ↔ 2 ∣ (a * b) := by
  sorry

end NUMINAMATH_CALUDE_cuboid_diagonal_l2298_229864


namespace NUMINAMATH_CALUDE_computer_B_most_popular_l2298_229807

/-- Represents the sales data for a computer over three years -/
structure ComputerSales where
  year2018 : Nat
  year2019 : Nat
  year2020 : Nat

/-- Checks if the sales are consistently increasing -/
def isConsistentlyIncreasing (sales : ComputerSales) : Prop :=
  sales.year2018 < sales.year2019 ∧ sales.year2019 < sales.year2020

/-- Defines the sales data for computers A, B, and C -/
def computerA : ComputerSales := { year2018 := 600, year2019 := 610, year2020 := 590 }
def computerB : ComputerSales := { year2018 := 590, year2019 := 650, year2020 := 700 }
def computerC : ComputerSales := { year2018 := 650, year2019 := 670, year2020 := 660 }

/-- Theorem: Computer B is the most popular choice -/
theorem computer_B_most_popular :
  isConsistentlyIncreasing computerB ∧
  ¬isConsistentlyIncreasing computerA ∧
  ¬isConsistentlyIncreasing computerC :=
sorry

end NUMINAMATH_CALUDE_computer_B_most_popular_l2298_229807


namespace NUMINAMATH_CALUDE_journey_ratio_theorem_l2298_229808

/-- Represents the distance between two towns -/
structure Distance where
  miles : ℝ
  nonneg : miles ≥ 0

/-- Represents the speed of travel -/
structure Speed where
  mph : ℝ
  positive : mph > 0

/-- Represents a journey between two towns -/
structure Journey where
  distance : Distance
  speed : Speed

theorem journey_ratio_theorem 
  (speed_AB : Speed) 
  (speed_BC : Speed) 
  (avg_speed : Speed) 
  (h1 : speed_AB.mph = 60)
  (h2 : speed_BC.mph = 20)
  (h3 : avg_speed.mph = 36) :
  ∃ (dist_AB dist_BC : Distance),
    let journey_AB : Journey := ⟨dist_AB, speed_AB⟩
    let journey_BC : Journey := ⟨dist_BC, speed_BC⟩
    let total_distance : Distance := ⟨dist_AB.miles + dist_BC.miles, by sorry⟩
    let total_time : ℝ := dist_AB.miles / speed_AB.mph + dist_BC.miles / speed_BC.mph
    avg_speed.mph = total_distance.miles / total_time →
    dist_AB.miles / dist_BC.miles = 2 :=
by sorry

end NUMINAMATH_CALUDE_journey_ratio_theorem_l2298_229808


namespace NUMINAMATH_CALUDE_order_divides_exponent_l2298_229801

theorem order_divides_exponent (x m d p : ℕ) (hp : Prime p) : 
  (∀ k : ℕ, k > 0 ∧ k < d → x^k % p ≠ 1) →  -- d is the order of x modulo p
  x^d % p = 1 →                             -- definition of order
  x^m % p = 1 →                             -- given condition
  d ∣ m :=                                  -- conclusion: d divides m
sorry

end NUMINAMATH_CALUDE_order_divides_exponent_l2298_229801


namespace NUMINAMATH_CALUDE_teal_color_survey_l2298_229833

theorem teal_color_survey (total : ℕ) (more_green : ℕ) (both : ℕ) (neither : ℕ) (undecided : ℕ) : 
  total = 150 → 
  more_green = 90 → 
  both = 40 → 
  neither = 20 → 
  undecided = 10 → 
  ∃ (more_blue : ℕ), more_blue = 70 ∧ 
    total = more_green + more_blue - both + neither + undecided :=
by sorry

end NUMINAMATH_CALUDE_teal_color_survey_l2298_229833


namespace NUMINAMATH_CALUDE_duck_flying_days_l2298_229887

/-- The number of days it takes a duck to fly south during winter -/
def days_south : ℕ := 40

/-- The number of days it takes a duck to fly north during summer -/
def days_north : ℕ := 2 * days_south

/-- The number of days it takes a duck to fly east during spring -/
def days_east : ℕ := 60

/-- The total number of days a duck flies during winter, summer, and spring -/
def total_flying_days : ℕ := days_south + days_north + days_east

/-- Theorem stating that the total number of days a duck flies during winter, summer, and spring is 180 -/
theorem duck_flying_days : total_flying_days = 180 := by
  sorry

end NUMINAMATH_CALUDE_duck_flying_days_l2298_229887


namespace NUMINAMATH_CALUDE_adjacent_arrangement_count_not_head_tail_arrangement_count_two_rows_arrangement_count_l2298_229862

/-- The number of arrangements for six people in a row with persons A and B next to each other -/
def arrangements_adjacent (n : ℕ) : ℕ :=
  if n = 6 then 240 else 0

/-- The number of arrangements for six people in a row with person A not at the head and person B not at the tail -/
def arrangements_not_head_tail (n : ℕ) : ℕ :=
  if n = 6 then 504 else 0

/-- The number of arrangements for six people in two rows with three people per row, and the front row shorter than the back row -/
def arrangements_two_rows (n : ℕ) : ℕ :=
  if n = 6 then 36 else 0

theorem adjacent_arrangement_count :
  arrangements_adjacent 6 = 240 := by sorry

theorem not_head_tail_arrangement_count :
  arrangements_not_head_tail 6 = 504 := by sorry

theorem two_rows_arrangement_count :
  arrangements_two_rows 6 = 36 := by sorry

end NUMINAMATH_CALUDE_adjacent_arrangement_count_not_head_tail_arrangement_count_two_rows_arrangement_count_l2298_229862


namespace NUMINAMATH_CALUDE_earnings_difference_l2298_229849

def saheed_earnings : ℕ := 216
def vika_earnings : ℕ := 84

def kayla_earnings : ℕ := saheed_earnings / 4

theorem earnings_difference : vika_earnings - kayla_earnings = 30 :=
by sorry

end NUMINAMATH_CALUDE_earnings_difference_l2298_229849


namespace NUMINAMATH_CALUDE_linear_system_fraction_sum_l2298_229806

theorem linear_system_fraction_sum (a b c x y z : ℝ) 
  (eq1 : 17 * x + b * y + c * z = 0)
  (eq2 : a * x + 29 * y + c * z = 0)
  (eq3 : a * x + b * y + 53 * z = 0)
  (ha : a ≠ 17)
  (hx : x ≠ 0) :
  a / (a - 17) + b / (b - 29) + c / (c - 53) = 1 := by
sorry

end NUMINAMATH_CALUDE_linear_system_fraction_sum_l2298_229806


namespace NUMINAMATH_CALUDE_no_exact_change_for_57_can_make_change_for_15_l2298_229885

/-- Represents the available Tyro bill denominations -/
def tyro_bills : List ℕ := [35, 80]

/-- Checks if a given amount can be represented as a sum of available Tyro bills -/
def can_make_exact_change (amount : ℕ) : Prop :=
  ∃ (a b : ℕ), a * 35 + b * 80 = amount

/-- Checks if a given amount can be represented as a difference of sums of available Tyro bills -/
def can_make_change_with_subtraction (amount : ℕ) : Prop :=
  ∃ (a b c d : ℕ), a * 35 + b * 80 - (c * 35 + d * 80) = amount

/-- Theorem stating that exact change cannot be made for 57 Tyros -/
theorem no_exact_change_for_57 : ¬ can_make_exact_change 57 := by sorry

/-- Theorem stating that change can be made for 15 Tyros using subtraction -/
theorem can_make_change_for_15 : can_make_change_with_subtraction 15 := by sorry

end NUMINAMATH_CALUDE_no_exact_change_for_57_can_make_change_for_15_l2298_229885


namespace NUMINAMATH_CALUDE_prove_a_equals_3x_l2298_229830

theorem prove_a_equals_3x (a b x : ℝ) 
  (h1 : a ≠ b) 
  (h2 : a^3 - b^3 = 27*x^3) 
  (h3 : a - b = 3*x) : 
  a = 3*x := by sorry

end NUMINAMATH_CALUDE_prove_a_equals_3x_l2298_229830


namespace NUMINAMATH_CALUDE_last_l_replaced_by_p_l2298_229883

-- Define the alphabet size
def alphabet_size : ℕ := 26

-- Define the position of 'l' in the alphabet (1-indexed)
def l_position : ℕ := 12

-- Define the occurrence of the last 'l' in the message
def l_occurrence : ℕ := 2

-- Define the shift function
def shift (n : ℕ) : ℕ := 2^n

-- Define the function to calculate the new position
def new_position (start : ℕ) (shift : ℕ) : ℕ :=
  (start + shift - 1) % alphabet_size + 1

-- Define the position of 'p' in the alphabet (1-indexed)
def p_position : ℕ := 16

-- The theorem to prove
theorem last_l_replaced_by_p :
  new_position l_position (shift l_occurrence) = p_position := by
  sorry

end NUMINAMATH_CALUDE_last_l_replaced_by_p_l2298_229883


namespace NUMINAMATH_CALUDE_item_cost_before_tax_reduction_cost_is_1000_l2298_229836

theorem item_cost_before_tax_reduction (tax_difference : ℝ) (cost_difference : ℝ) : ℝ :=
  let original_tax_rate := 0.05
  let new_tax_rate := 0.04
  let item_cost := cost_difference / (original_tax_rate - new_tax_rate)
  item_cost

theorem cost_is_1000 :
  item_cost_before_tax_reduction 0.01 10 = 1000 := by
  sorry

end NUMINAMATH_CALUDE_item_cost_before_tax_reduction_cost_is_1000_l2298_229836


namespace NUMINAMATH_CALUDE_restaurant_service_charge_l2298_229895

theorem restaurant_service_charge (total_paid : ℝ) (service_charge_rate : ℝ) 
  (h1 : service_charge_rate = 0.04)
  (h2 : total_paid = 468) :
  ∃ (original_amount : ℝ), 
    original_amount * (1 + service_charge_rate) = total_paid ∧ 
    original_amount = 450 := by
  sorry

end NUMINAMATH_CALUDE_restaurant_service_charge_l2298_229895


namespace NUMINAMATH_CALUDE_roots_equation_value_l2298_229884

theorem roots_equation_value (x₁ x₂ : ℝ) : 
  x₁^2 - 3*x₁ - 4 = 0 → 
  x₂^2 - 3*x₂ - 4 = 0 → 
  x₁^2 - 4*x₁ - x₂ + 2*x₁*x₂ = -7 := by
  sorry

end NUMINAMATH_CALUDE_roots_equation_value_l2298_229884


namespace NUMINAMATH_CALUDE_bench_press_changes_l2298_229819

/-- Calculates the final bench press weight after a series of changes -/
def final_bench_press (initial_weight : ℝ) : ℝ :=
  let after_injury := initial_weight * (1 - 0.8)
  let after_recovery := after_injury * (1 + 0.6)
  let after_setback := after_recovery * (1 - 0.2)
  let final_weight := after_setback * 3
  final_weight

/-- Theorem stating that the final bench press weight is 384 pounds -/
theorem bench_press_changes (initial_weight : ℝ) 
  (h : initial_weight = 500) : 
  final_bench_press initial_weight = 384 := by
  sorry

#eval final_bench_press 500

end NUMINAMATH_CALUDE_bench_press_changes_l2298_229819


namespace NUMINAMATH_CALUDE_wheel_rotation_coincidence_l2298_229850

/-- The distance the larger wheel must roll for two initially coincident points to coincide again -/
theorem wheel_rotation_coincidence (big_circ small_circ : ℕ) 
  (h_big : big_circ = 12) 
  (h_small : small_circ = 8) : 
  Nat.lcm big_circ small_circ = 24 := by
  sorry

end NUMINAMATH_CALUDE_wheel_rotation_coincidence_l2298_229850


namespace NUMINAMATH_CALUDE_product_of_binary_and_ternary_l2298_229818

/-- Converts a list of digits in a given base to a natural number. -/
def to_nat (digits : List Nat) (base : Nat) : Nat :=
  digits.foldr (fun d acc => d + base * acc) 0

/-- The binary number 1011. -/
def binary_num : List Nat := [1, 1, 0, 1]

/-- The ternary number 1021. -/
def ternary_num : List Nat := [1, 2, 0, 1]

theorem product_of_binary_and_ternary :
  (to_nat binary_num 2) * (to_nat ternary_num 3) = 374 := by
  sorry

end NUMINAMATH_CALUDE_product_of_binary_and_ternary_l2298_229818


namespace NUMINAMATH_CALUDE_josh_pencils_calculation_l2298_229873

/-- The number of pencils Josh had initially -/
def initial_pencils : ℕ := 142

/-- The number of pencils Josh gave away -/
def pencils_given_away : ℕ := 31

/-- The number of pencils Josh is left with -/
def remaining_pencils : ℕ := initial_pencils - pencils_given_away

theorem josh_pencils_calculation : remaining_pencils = 111 := by
  sorry

end NUMINAMATH_CALUDE_josh_pencils_calculation_l2298_229873


namespace NUMINAMATH_CALUDE_bus_driver_regular_rate_l2298_229810

/-- Represents the compensation structure and work hours of a bus driver --/
structure BusDriverCompensation where
  regularRate : ℝ
  overtimeRate : ℝ
  regularHours : ℝ
  overtimeHours : ℝ
  totalCompensation : ℝ

/-- Theorem stating that given the conditions, the regular rate is $18 per hour --/
theorem bus_driver_regular_rate 
  (comp : BusDriverCompensation)
  (h1 : comp.regularHours = 40)
  (h2 : comp.overtimeHours = 48.12698412698413 - 40)
  (h3 : comp.overtimeRate = comp.regularRate * 1.75)
  (h4 : comp.totalCompensation = 976)
  (h5 : comp.totalCompensation = comp.regularRate * comp.regularHours + 
                                 comp.overtimeRate * comp.overtimeHours) :
  comp.regularRate = 18 := by
  sorry


end NUMINAMATH_CALUDE_bus_driver_regular_rate_l2298_229810


namespace NUMINAMATH_CALUDE_baker_revenue_l2298_229856

/-- The intended revenue for a baker selling birthday cakes -/
theorem baker_revenue (n : ℝ) : 
  (∀ (reduced_price : ℝ), reduced_price = 0.8 * n → 10 * reduced_price = 8 * n) →
  8 * n = 8 * n := by sorry

end NUMINAMATH_CALUDE_baker_revenue_l2298_229856


namespace NUMINAMATH_CALUDE_complement_of_union_l2298_229825

def U : Set Int := {-2, -1, 0, 1, 2, 3}
def A : Set Int := {-1, 0, 1}
def B : Set Int := {1, 2}

theorem complement_of_union :
  (U \ (A ∪ B)) = {-2, 3} := by sorry

end NUMINAMATH_CALUDE_complement_of_union_l2298_229825


namespace NUMINAMATH_CALUDE_equal_volumes_of_modified_cylinders_l2298_229875

/-- Theorem: Equal volumes of modified cylinders -/
theorem equal_volumes_of_modified_cylinders :
  let r : ℝ := 5  -- Initial radius in inches
  let h : ℝ := 4  -- Initial height in inches
  let dr : ℝ := 2  -- Increase in radius for the first cylinder
  ∀ y : ℝ,  -- Increase in height for the second cylinder
  y ≠ 0 →  -- y is non-zero
  π * (r + dr)^2 * h = π * r^2 * (h + y) →  -- Volumes are equal
  y = 96 / 25 := by
sorry

end NUMINAMATH_CALUDE_equal_volumes_of_modified_cylinders_l2298_229875


namespace NUMINAMATH_CALUDE_halves_in_two_sevenths_l2298_229861

theorem halves_in_two_sevenths : (2 : ℚ) / 7 / (1 : ℚ) / 2 = 4 / 7 := by
  sorry

end NUMINAMATH_CALUDE_halves_in_two_sevenths_l2298_229861


namespace NUMINAMATH_CALUDE_complex_multiplication_result_l2298_229878

theorem complex_multiplication_result : ∃ (a b : ℝ), (Complex.I + 1) * (2 - Complex.I) = Complex.mk a b ∧ a = 3 ∧ b = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_result_l2298_229878


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l2298_229863

theorem regular_polygon_sides (perimeter : ℝ) (side_length : ℝ) (h1 : perimeter = 108) (h2 : side_length = 12) :
  (perimeter / side_length : ℝ) = 9 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l2298_229863


namespace NUMINAMATH_CALUDE_square_difference_pattern_l2298_229853

theorem square_difference_pattern (n : ℕ+) : (2*n + 1)^2 - (2*n - 1)^2 = 8*n := by
  sorry

end NUMINAMATH_CALUDE_square_difference_pattern_l2298_229853


namespace NUMINAMATH_CALUDE_matrix_multiplication_result_l2298_229893

theorem matrix_multiplication_result : 
  let A : Matrix (Fin 2) (Fin 2) ℤ := !![3, -1; 6, -4]
  let B : Matrix (Fin 2) (Fin 2) ℤ := !![9, -3; 2, 2]
  A * B = !![25, -11; 46, -26] := by
  sorry

end NUMINAMATH_CALUDE_matrix_multiplication_result_l2298_229893


namespace NUMINAMATH_CALUDE_new_person_age_l2298_229814

theorem new_person_age (n : ℕ) (initial_avg : ℝ) (new_avg : ℝ) (new_person_age : ℝ) : 
  n = 9 → 
  initial_avg = 14 → 
  new_avg = 16 → 
  (n * initial_avg + new_person_age) / (n + 1) = new_avg → 
  new_person_age = 34 := by
sorry

end NUMINAMATH_CALUDE_new_person_age_l2298_229814


namespace NUMINAMATH_CALUDE_smallest_non_odd_units_digit_l2298_229877

def is_odd (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k + 1

def units_digit (n : ℕ) : ℕ := n % 10

def is_digit (d : ℕ) : Prop := d < 10

theorem smallest_non_odd_units_digit :
  ∀ d : ℕ, is_digit d →
    (∀ n : ℕ, is_odd n → units_digit n ≠ d) →
    (∀ e : ℕ, is_digit e → (∀ m : ℕ, is_odd m → units_digit m ≠ e) → d ≤ e) →
    d = 0 :=
sorry

end NUMINAMATH_CALUDE_smallest_non_odd_units_digit_l2298_229877


namespace NUMINAMATH_CALUDE_tile_perimeter_change_l2298_229892

/-- Represents a shape made of square tiles -/
structure TileShape where
  tiles : ℕ
  perimeter : ℕ

/-- Adds tiles to a shape and returns the new perimeter -/
def add_tiles (shape : TileShape) (new_tiles : ℕ) : Set ℕ :=
  sorry

theorem tile_perimeter_change (initial_shape : TileShape) :
  initial_shape.tiles = 10 →
  initial_shape.perimeter = 16 →
  ∃ (new_perimeter : Set ℕ),
    new_perimeter = add_tiles initial_shape 2 ∧
    new_perimeter = {23, 25} :=
by sorry

end NUMINAMATH_CALUDE_tile_perimeter_change_l2298_229892


namespace NUMINAMATH_CALUDE_r_amount_l2298_229860

theorem r_amount (total : ℝ) (p_q_amount : ℝ) (r_amount : ℝ) : 
  total = 6000 →
  r_amount = (2/3) * p_q_amount →
  total = p_q_amount + r_amount →
  r_amount = 2400 := by
sorry

end NUMINAMATH_CALUDE_r_amount_l2298_229860


namespace NUMINAMATH_CALUDE_total_games_is_32_l2298_229866

/-- The number of games won by Jerry -/
def jerry_games : ℕ := 7

/-- The number of games won by Dave -/
def dave_games : ℕ := jerry_games + 3

/-- The number of games won by Ken -/
def ken_games : ℕ := dave_games + 5

/-- The total number of games played -/
def total_games : ℕ := ken_games + dave_games + jerry_games

theorem total_games_is_32 : total_games = 32 := by
  sorry

end NUMINAMATH_CALUDE_total_games_is_32_l2298_229866


namespace NUMINAMATH_CALUDE_max_a_value_l2298_229809

theorem max_a_value (a : ℝ) : 
  (∀ x : ℝ, x^2 - 2*x - 3 > 0 → x < a) ∧ 
  (∃ x : ℝ, x < a ∧ x^2 - 2*x - 3 ≤ 0) →
  a ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_max_a_value_l2298_229809


namespace NUMINAMATH_CALUDE_four_collinear_points_l2298_229820

open Real

-- Define the curve
def curve (α : ℝ) (x : ℝ) : ℝ := x^4 + 9*x^3 + α*x^2 + 9*x + 4

-- Define the second derivative of the curve
def second_derivative (α : ℝ) (x : ℝ) : ℝ := 12*x^2 + 54*x + 2*α

-- Theorem statement
theorem four_collinear_points (α : ℝ) :
  (∃ x₁ x₂ x₃ x₄ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄ ∧
    (∃ a b : ℝ, curve α x₁ = a*x₁ + b ∧ 
                curve α x₂ = a*x₂ + b ∧ 
                curve α x₃ = a*x₃ + b ∧ 
                curve α x₄ = a*x₄ + b)) ↔
  α < 30.375 :=
by sorry

end NUMINAMATH_CALUDE_four_collinear_points_l2298_229820


namespace NUMINAMATH_CALUDE_apricot_trees_count_apricot_trees_proof_l2298_229834

theorem apricot_trees_count : ℕ → ℕ → Prop :=
  fun apricot_count peach_count =>
    peach_count = 3 * apricot_count →
    apricot_count + peach_count = 232 →
    apricot_count = 58

-- The proof is omitted as per instructions
theorem apricot_trees_proof : apricot_trees_count 58 174 := by
  sorry

end NUMINAMATH_CALUDE_apricot_trees_count_apricot_trees_proof_l2298_229834


namespace NUMINAMATH_CALUDE_quadratic_trinomial_equality_l2298_229876

/-- A quadratic trinomial function -/
def quadratic_trinomial (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_trinomial_equality (a b c : ℝ) (h : a ≠ 0) :
  (∀ x, quadratic_trinomial a b c (3.8 * x - 1) = quadratic_trinomial a b c (-3.8 * x)) →
  b = a := by sorry

end NUMINAMATH_CALUDE_quadratic_trinomial_equality_l2298_229876


namespace NUMINAMATH_CALUDE_volunteer_distribution_count_l2298_229852

/-- The number of volunteers --/
def num_volunteers : ℕ := 7

/-- The number of positions --/
def num_positions : ℕ := 4

/-- The number of ways to choose 2 people from 5 --/
def choose_two_from_five : ℕ := (5 * 4) / (2 * 1)

/-- The number of ways to permute 4 items --/
def permute_four : ℕ := 4 * 3 * 2 * 1

/-- The total number of ways to distribute volunteers when A and B can be in the same group --/
def total_ways : ℕ := choose_two_from_five * permute_four

/-- The number of ways where A and B are in the same position --/
def same_position_ways : ℕ := permute_four

/-- The number of ways for A and B not to serve at the same position --/
def different_position_ways : ℕ := total_ways - same_position_ways

theorem volunteer_distribution_count :
  different_position_ways = 216 :=
sorry

end NUMINAMATH_CALUDE_volunteer_distribution_count_l2298_229852


namespace NUMINAMATH_CALUDE_large_shoes_count_l2298_229894

/-- The number of pairs of large-size shoes initially stocked by the shop -/
def L : ℕ := sorry

/-- The number of pairs of medium-size shoes initially stocked by the shop -/
def medium_shoes : ℕ := 50

/-- The number of pairs of small-size shoes initially stocked by the shop -/
def small_shoes : ℕ := 24

/-- The number of pairs of shoes sold by the shop -/
def sold_shoes : ℕ := 83

/-- The number of pairs of shoes left after selling -/
def left_shoes : ℕ := 13

theorem large_shoes_count : L = 22 := by
  sorry

end NUMINAMATH_CALUDE_large_shoes_count_l2298_229894


namespace NUMINAMATH_CALUDE_percent_within_one_sd_is_68_l2298_229840

/-- A symmetric distribution with a given percentage below one standard deviation above the mean -/
structure SymmetricDistribution where
  /-- The percentage of the distribution below one standard deviation above the mean -/
  percent_below_one_sd : ℝ
  /-- Assumption that the percentage is 84% -/
  percent_is_84 : percent_below_one_sd = 84

/-- The percentage of a symmetric distribution that lies within one standard deviation of the mean -/
def percent_within_one_sd (d : SymmetricDistribution) : ℝ :=
  2 * d.percent_below_one_sd - 100

theorem percent_within_one_sd_is_68 (d : SymmetricDistribution) :
  percent_within_one_sd d = 68 := by
  sorry

end NUMINAMATH_CALUDE_percent_within_one_sd_is_68_l2298_229840


namespace NUMINAMATH_CALUDE_no_three_digit_divisible_by_15_ending_in_7_l2298_229847

theorem no_three_digit_divisible_by_15_ending_in_7 : 
  ∀ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 10 = 7 → ¬(n % 15 = 0) :=
by sorry

end NUMINAMATH_CALUDE_no_three_digit_divisible_by_15_ending_in_7_l2298_229847


namespace NUMINAMATH_CALUDE_english_only_students_l2298_229822

theorem english_only_students (total : ℕ) (both : ℕ) (french : ℕ) (english : ℕ) : 
  total = 30 ∧ 
  both = 2 ∧ 
  english = 3 * french ∧ 
  total = french + english - both → 
  english - both = 20 := by
sorry

end NUMINAMATH_CALUDE_english_only_students_l2298_229822


namespace NUMINAMATH_CALUDE_zeros_between_seven_and_three_l2298_229831

theorem zeros_between_seven_and_three : ∀ n : ℕ, 
  (7 * 10^(n + 1) + 3 = 70003) ↔ (n = 4) :=
by sorry

end NUMINAMATH_CALUDE_zeros_between_seven_and_three_l2298_229831


namespace NUMINAMATH_CALUDE_b_complete_time_l2298_229882

/-- The time it takes for A to complete the work alone -/
def a_time : ℚ := 14 / 3

/-- The time A and B work together -/
def together_time : ℚ := 1

/-- The time B works alone after A leaves -/
def b_remaining_time : ℚ := 41 / 14

/-- The time it takes for B to complete the work alone -/
def b_time : ℚ := 5

theorem b_complete_time : 
  (1 / a_time + 1 / b_time) * together_time + 
  (1 / b_time) * b_remaining_time = 1 := by sorry

end NUMINAMATH_CALUDE_b_complete_time_l2298_229882


namespace NUMINAMATH_CALUDE_frustum_height_l2298_229816

def pyramid_frustum (h_small : ℝ) (area_ratio : ℝ) (h_frustum : ℝ) : Prop :=
  let h_large := h_small * (area_ratio.sqrt)
  h_frustum = h_large - h_small

theorem frustum_height : 
  pyramid_frustum 3 4 3 := by sorry

end NUMINAMATH_CALUDE_frustum_height_l2298_229816


namespace NUMINAMATH_CALUDE_pentagonal_pyramid_base_areas_l2298_229813

theorem pentagonal_pyramid_base_areas (total_surface_area lateral_surface_area : ℝ) :
  total_surface_area = 30 →
  lateral_surface_area = 25 →
  total_surface_area - lateral_surface_area = 5 := by
  sorry

end NUMINAMATH_CALUDE_pentagonal_pyramid_base_areas_l2298_229813


namespace NUMINAMATH_CALUDE_stock_sale_loss_l2298_229841

/-- Calculates the overall loss amount for a stock sale scenario -/
theorem stock_sale_loss (stock_worth : ℝ) (profit_percent : ℝ) (loss_percent : ℝ) 
  (profit_stock_percent : ℝ) (loss_stock_percent : ℝ) :
  stock_worth = 10000 →
  profit_percent = 10 →
  loss_percent = 5 →
  profit_stock_percent = 20 →
  loss_stock_percent = 80 →
  let profit_amount := (profit_stock_percent / 100) * stock_worth * (1 + profit_percent / 100)
  let loss_amount := (loss_stock_percent / 100) * stock_worth * (1 - loss_percent / 100)
  let total_sale := profit_amount + loss_amount
  stock_worth - total_sale = 200 := by sorry

end NUMINAMATH_CALUDE_stock_sale_loss_l2298_229841


namespace NUMINAMATH_CALUDE_tan_alpha_value_l2298_229890

theorem tan_alpha_value (α : Real) 
  (h : (Real.sin α - 2 * Real.cos α) / (2 * Real.sin α + 5 * Real.cos α) = -5) :
  Real.tan α = -23/11 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l2298_229890


namespace NUMINAMATH_CALUDE_particular_number_problem_l2298_229886

theorem particular_number_problem : ∃! x : ℚ, 2 * (67 - (x / 23)) = 102 := by sorry

end NUMINAMATH_CALUDE_particular_number_problem_l2298_229886


namespace NUMINAMATH_CALUDE_final_result_l2298_229845

def loop_calculation (i : ℕ) (S : ℕ) : ℕ :=
  if i < 9 then S else loop_calculation (i - 1) (S * i)

theorem final_result :
  loop_calculation 11 1 = 990 := by
  sorry

end NUMINAMATH_CALUDE_final_result_l2298_229845


namespace NUMINAMATH_CALUDE_lingonberries_to_pick_thursday_l2298_229828

/-- The amount of money Steve wants to make in total -/
def total_money : ℕ := 100

/-- The number of days Steve has to make the money -/
def total_days : ℕ := 4

/-- The amount of money Steve earns per pound of lingonberries -/
def money_per_pound : ℕ := 2

/-- The amount of lingonberries Steve picked on Monday -/
def monday_picked : ℕ := 8

/-- The amount of lingonberries Steve picked on Tuesday relative to Monday -/
def tuesday_multiplier : ℕ := 3

/-- The amount of lingonberries Steve picked on Wednesday -/
def wednesday_picked : ℕ := 0

theorem lingonberries_to_pick_thursday : 
  (total_money / money_per_pound) - 
  (monday_picked + tuesday_multiplier * monday_picked + wednesday_picked) = 18 := by
  sorry

end NUMINAMATH_CALUDE_lingonberries_to_pick_thursday_l2298_229828


namespace NUMINAMATH_CALUDE_f_derivative_zero_l2298_229880

-- Define the function f
variable (f : ℝ → ℝ)

-- State the conditions
axiom f_differentiable : Differentiable ℝ f
axiom f_additive : ∀ x y : ℝ, f (x + y) = f x + f y

-- State the theorem
theorem f_derivative_zero : deriv f 0 = 0 := by sorry

end NUMINAMATH_CALUDE_f_derivative_zero_l2298_229880


namespace NUMINAMATH_CALUDE_jonathan_tax_calculation_l2298_229804

/-- Calculates the local tax amount in cents given an hourly wage in dollars and a tax rate as a percentage. -/
def localTaxInCents (hourlyWage : ℚ) (taxRate : ℚ) : ℚ :=
  hourlyWage * 100 * (taxRate / 100)

/-- Theorem stating that for an hourly wage of $25 and a tax rate of 2.4%, the local tax amount is 60 cents. -/
theorem jonathan_tax_calculation :
  localTaxInCents 25 2.4 = 60 := by
  sorry

#eval localTaxInCents 25 2.4

end NUMINAMATH_CALUDE_jonathan_tax_calculation_l2298_229804


namespace NUMINAMATH_CALUDE_geometric_series_sum_l2298_229859

theorem geometric_series_sum (a : ℝ) (h : |a| < 1) :
  (∑' n, a^n) = 1 / (1 - a) :=
sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l2298_229859


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l2298_229854

theorem quadratic_equation_roots (m : ℝ) : 
  (∃ x : ℝ, x^2 + 2*x + m - 2 = 0 ∧ x = -3) →
  (m = -1 ∧ ∃ y : ℝ, y^2 + 2*y + m - 2 = 0 ∧ y = 1) := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l2298_229854


namespace NUMINAMATH_CALUDE_remainder_3_100_plus_4_mod_5_l2298_229811

theorem remainder_3_100_plus_4_mod_5 : (3^100 + 4) % 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_remainder_3_100_plus_4_mod_5_l2298_229811


namespace NUMINAMATH_CALUDE_range_of_b_minus_a_l2298_229827

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 2*x

-- State the theorem
theorem range_of_b_minus_a (a b : ℝ) :
  (∀ x ∈ Set.Icc a b, f x ∈ Set.Icc (-1) 3) ∧
  (∀ y ∈ Set.Icc (-1) 3, ∃ x ∈ Set.Icc a b, f x = y) →
  b - a ∈ Set.Icc 2 4 :=
by sorry

end NUMINAMATH_CALUDE_range_of_b_minus_a_l2298_229827


namespace NUMINAMATH_CALUDE_greatest_common_remainder_l2298_229846

theorem greatest_common_remainder (a b c d : ℕ) (h1 : a % 2 = 0 ∧ a % 3 = 0 ∧ a % 5 = 0 ∧ a % 7 = 0 ∧ a % 11 = 0)
                                               (h2 : b % 2 = 0 ∧ b % 3 = 0 ∧ b % 5 = 0 ∧ b % 7 = 0 ∧ b % 11 = 0)
                                               (h3 : c % 2 = 0 ∧ c % 3 = 0 ∧ c % 5 = 0 ∧ c % 7 = 0 ∧ c % 11 = 0)
                                               (h4 : d % 2 = 0 ∧ d % 3 = 0 ∧ d % 5 = 0 ∧ d % 7 = 0 ∧ d % 11 = 0)
                                               (ha : a = 1260) (hb : b = 2310) (hc : c = 30030) (hd : d = 72930) :
  ∃! k : ℕ, k > 0 ∧ k ≤ 30 ∧ 
  ∃ r : ℕ, a % k = r ∧ b % k = r ∧ c % k = r ∧ d % k = r ∧
  ∀ m : ℕ, m > k → ¬(∃ s : ℕ, a % m = s ∧ b % m = s ∧ c % m = s ∧ d % m = s) :=
by sorry

end NUMINAMATH_CALUDE_greatest_common_remainder_l2298_229846


namespace NUMINAMATH_CALUDE_twenty_men_handshakes_l2298_229872

/-- The maximum number of handshakes without cyclic handshakes for n people -/
def max_handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: For 20 men, the maximum number of handshakes without cyclic handshakes is 190 -/
theorem twenty_men_handshakes :
  max_handshakes 20 = 190 := by
  sorry

#eval max_handshakes 20  -- This will evaluate to 190

end NUMINAMATH_CALUDE_twenty_men_handshakes_l2298_229872


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_times_i_l2298_229891

theorem imaginary_part_of_z_times_i (z : ℂ) : z = -1 + 2*I → Complex.im (z * I) = -1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_times_i_l2298_229891


namespace NUMINAMATH_CALUDE_joan_cake_flour_l2298_229897

theorem joan_cake_flour (recipe_total : ℕ) (already_added : ℕ) (h1 : recipe_total = 7) (h2 : already_added = 3) :
  recipe_total - already_added = 4 := by
  sorry

end NUMINAMATH_CALUDE_joan_cake_flour_l2298_229897


namespace NUMINAMATH_CALUDE_A_power_50_l2298_229823

def A : Matrix (Fin 2) (Fin 2) ℤ := !![5, 1; -12, -3]

theorem A_power_50 : A^50 = !![301, 50; -900, -301] := by sorry

end NUMINAMATH_CALUDE_A_power_50_l2298_229823


namespace NUMINAMATH_CALUDE_factorization_equality_l2298_229869

/-- For all real numbers a and b, ab² - 2ab + a = a(b-1)² --/
theorem factorization_equality (a b : ℝ) : a * b^2 - 2 * a * b + a = a * (b - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l2298_229869


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2298_229835

theorem sufficient_not_necessary_condition (m : ℝ) :
  (∀ m, m < -2 → ∃ x : ℝ, x^2 + m*x + 1 = 0) ∧
  ¬(∀ x : ℝ, x^2 + m*x + 1 = 0 → m < -2) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2298_229835


namespace NUMINAMATH_CALUDE_constant_function_value_l2298_229867

theorem constant_function_value (g : ℝ → ℝ) (h : ∀ x : ℝ, g x = -3) :
  ∀ x : ℝ, g (3 * x - 5) = -3 := by
  sorry

end NUMINAMATH_CALUDE_constant_function_value_l2298_229867


namespace NUMINAMATH_CALUDE_max_value_of_g_l2298_229874

-- Define the function g(x)
def g (x : ℝ) : ℝ := 4 * x - x^4

-- State the theorem
theorem max_value_of_g :
  ∃ (x : ℝ), 0 ≤ x ∧ x ≤ 2 ∧ g x = 3 ∧ ∀ (y : ℝ), 0 ≤ y ∧ y ≤ 2 → g y ≤ g x :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_g_l2298_229874


namespace NUMINAMATH_CALUDE_distance_2_neg5_distance_neg2_neg5_distance_1_neg3_solutions_abs_x_plus_1_eq_2_min_value_range_l2298_229888

-- Define distance between two points on a number line
def distance (a b : ℝ) : ℝ := |a - b|

-- Theorem 1: Distance between 2 and -5
theorem distance_2_neg5 : distance 2 (-5) = 7 := by sorry

-- Theorem 2: Distance between -2 and -5
theorem distance_neg2_neg5 : distance (-2) (-5) = 3 := by sorry

-- Theorem 3: Distance between 1 and -3
theorem distance_1_neg3 : distance 1 (-3) = 4 := by sorry

-- Theorem 4: Solutions for |x + 1| = 2
theorem solutions_abs_x_plus_1_eq_2 : 
  ∀ x : ℝ, |x + 1| = 2 ↔ x = 1 ∨ x = -3 := by sorry

-- Theorem 5: Range of x for minimum value of |x+1| + |x-2|
theorem min_value_range : 
  ∀ x : ℝ, (∀ y : ℝ, |x+1| + |x-2| ≤ |y+1| + |y-2|) → -1 ≤ x ∧ x ≤ 2 := by sorry

end NUMINAMATH_CALUDE_distance_2_neg5_distance_neg2_neg5_distance_1_neg3_solutions_abs_x_plus_1_eq_2_min_value_range_l2298_229888


namespace NUMINAMATH_CALUDE_odd_function_tangent_line_sum_l2298_229870

def f (a b x : ℝ) : ℝ := a * x^3 + x + b

theorem odd_function_tangent_line_sum (a b : ℝ) :
  (∀ x, f a b x = -f a b (-x)) →  -- f is odd
  (∃ m c : ℝ, ∀ x, m * x + c = f a b 1 + (3 * a * 1^2 + 1) * (x - 1) ∧ 
              m * 2 + c = 6) →  -- tangent line passes through (2, 6)
  a + b = 1 := by
sorry

end NUMINAMATH_CALUDE_odd_function_tangent_line_sum_l2298_229870


namespace NUMINAMATH_CALUDE_f_neg_one_equals_one_l2298_229805

-- Define the function f
variable (f : ℝ → ℝ)

-- State the theorem
theorem f_neg_one_equals_one (h : ∀ x, f (x - 1) = x^2 + 1) : f (-1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_f_neg_one_equals_one_l2298_229805


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l2298_229838

theorem quadratic_equation_roots (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    x₁^2 - 4*x₁ - 2*m + 5 = 0 ∧ 
    x₂^2 - 4*x₂ - 2*m + 5 = 0 ∧
    x₁*x₂ + x₁ + x₂ = m^2 + 6) →
  m = 1 ∧ m ≥ (1/2 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l2298_229838


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l2298_229848

theorem min_value_reciprocal_sum (x : ℝ) (h : x > 0) : x + 1/x ≥ 2 ∧ (x + 1/x = 2 ↔ x = 1) := by sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l2298_229848


namespace NUMINAMATH_CALUDE_binomial_unique_solution_l2298_229826

/-- Represents a binomial distribution with parameters n and p -/
structure BinomialDistribution where
  n : ℕ
  p : ℝ
  h1 : 0 < p
  h2 : p < 1

/-- The expectation of a binomial distribution -/
def expectation (ξ : BinomialDistribution) : ℝ := ξ.n * ξ.p

/-- The variance of a binomial distribution -/
def variance (ξ : BinomialDistribution) : ℝ := ξ.n * ξ.p * (1 - ξ.p)

/-- Theorem stating the unique solution for n and p given E(ξ) and D(ξ) -/
theorem binomial_unique_solution :
  ∀ ξ : BinomialDistribution,
    expectation ξ = 12 →
    variance ξ = 4 →
    ξ.n = 18 ∧ ξ.p = 2/3 :=
by sorry

end NUMINAMATH_CALUDE_binomial_unique_solution_l2298_229826


namespace NUMINAMATH_CALUDE_tan_equality_with_range_l2298_229812

theorem tan_equality_with_range (n : ℤ) : 
  -90 < n ∧ n < 90 ∧ Real.tan (n * π / 180) = Real.tan (850 * π / 180) → n = -50 := by
  sorry

end NUMINAMATH_CALUDE_tan_equality_with_range_l2298_229812


namespace NUMINAMATH_CALUDE_y_value_theorem_l2298_229837

theorem y_value_theorem (y₁ y₂ y₃ y₄ y₅ y₆ y₇ y₈ : ℝ) 
  (eq1 : y₁ + 4*y₂ + 9*y₃ + 16*y₄ + 25*y₅ + 36*y₆ + 49*y₇ + 64*y₈ = 3)
  (eq2 : 4*y₁ + 9*y₂ + 16*y₃ + 25*y₄ + 36*y₅ + 49*y₆ + 64*y₇ + 81*y₈ = 15)
  (eq3 : 9*y₁ + 16*y₂ + 25*y₃ + 36*y₄ + 49*y₅ + 64*y₆ + 81*y₇ + 100*y₈ = 140) :
  16*y₁ + 25*y₂ + 36*y₃ + 49*y₄ + 64*y₅ + 81*y₆ + 100*y₇ + 121*y₈ = 472 :=
by sorry

end NUMINAMATH_CALUDE_y_value_theorem_l2298_229837


namespace NUMINAMATH_CALUDE_min_distance_line_parabola_l2298_229879

/-- The minimum distance between a point on the line y = (12/5)x - 9 and a point on the parabola y = x^2 is 189/65 -/
theorem min_distance_line_parabola :
  let line := fun x : ℝ => (12/5) * x - 9
  let parabola := fun x : ℝ => x^2
  ∃ (a b : ℝ),
    (∀ x y : ℝ, 
      (y = line x ∨ y = parabola x) → 
      (y - line a)^2 + (x - a)^2 ≥ ((12/5) * a - 9 - b^2)^2 + (a - b)^2) ∧
    ((12/5) * a - 9 - b^2)^2 + (a - b)^2 = (189/65)^2 :=
by sorry

end NUMINAMATH_CALUDE_min_distance_line_parabola_l2298_229879


namespace NUMINAMATH_CALUDE_circle_tangent_intersection_l2298_229829

theorem circle_tangent_intersection (R : ℝ) (φ ψ : ℝ) : ∃ (BX : ℝ), 
  BX = (2 * R * Real.sin φ * Real.sin ψ) / Real.sin (|φ - ψ|) ∨ 
  BX = (2 * R * Real.sin φ * Real.sin ψ) / Real.sin (φ + ψ) := by
  sorry

end NUMINAMATH_CALUDE_circle_tangent_intersection_l2298_229829


namespace NUMINAMATH_CALUDE_beef_weight_loss_percentage_l2298_229842

/-- Given a side of beef weighing 800 pounds before processing and 640 pounds after processing,
    the percentage of weight lost during processing is 20%. -/
theorem beef_weight_loss_percentage (weight_before : ℝ) (weight_after : ℝ) :
  weight_before = 800 ∧ weight_after = 640 →
  (weight_before - weight_after) / weight_before * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_beef_weight_loss_percentage_l2298_229842


namespace NUMINAMATH_CALUDE_water_remaining_calculation_l2298_229857

/-- Calculates the remaining water in a bucket after some has leaked out. -/
def remaining_water (initial : ℚ) (leaked : ℚ) : ℚ :=
  initial - leaked

/-- Theorem stating that given the initial amount and leaked amount, 
    the remaining water is 0.50 gallon. -/
theorem water_remaining_calculation (initial leaked : ℚ) 
  (h1 : initial = 3/4) 
  (h2 : leaked = 1/4) : 
  remaining_water initial leaked = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_water_remaining_calculation_l2298_229857


namespace NUMINAMATH_CALUDE_geometric_sequence_propositions_l2298_229824

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- An increasing sequence -/
def IncreasingSequence (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, n < m → a n < a m

theorem geometric_sequence_propositions (a : ℕ → ℝ) (h : GeometricSequence a) :
  (((a 1 < a 2) ∧ (a 2 < a 3)) → IncreasingSequence a) ∧
  (IncreasingSequence a → ((a 1 < a 2) ∧ (a 2 < a 3))) ∧
  (¬((a 1 < a 2) ∧ (a 2 < a 3)) → ¬IncreasingSequence a) ∧
  (¬IncreasingSequence a → ¬((a 1 < a 2) ∧ (a 2 < a 3))) :=
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_propositions_l2298_229824
