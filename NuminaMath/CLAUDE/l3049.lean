import Mathlib

namespace largest_fraction_l3049_304980

theorem largest_fraction : 
  (151 : ℚ) / 301 > 3 / 7 ∧
  (151 : ℚ) / 301 > 4 / 9 ∧
  (151 : ℚ) / 301 > 17 / 35 ∧
  (151 : ℚ) / 301 > 100 / 201 := by
  sorry

end largest_fraction_l3049_304980


namespace new_quadratic_from_roots_sum_product_l3049_304952

theorem new_quadratic_from_roots_sum_product (a b c : ℝ) (ha : a ≠ 0) :
  let original_eq := fun x => a * x^2 + b * x + c
  let new_eq := fun x => a^2 * x^2 + (a*b - a*c) * x - b*c
  let sum_of_roots := -b / a
  let product_of_roots := c / a
  (∀ x, original_eq x = 0 ↔ x = sum_of_roots ∨ x = product_of_roots) →
  (∀ x, new_eq x = 0 ↔ x = sum_of_roots ∨ x = product_of_roots) :=
by sorry

end new_quadratic_from_roots_sum_product_l3049_304952


namespace plot_length_is_75_l3049_304967

/-- The length of a rectangular plot in meters -/
def length : ℝ := 75

/-- The breadth of a rectangular plot in meters -/
def breadth : ℝ := length - 50

/-- The cost of fencing per meter in rupees -/
def cost_per_meter : ℝ := 26.50

/-- The total cost of fencing in rupees -/
def total_cost : ℝ := 5300

theorem plot_length_is_75 :
  (2 * length + 2 * breadth) * cost_per_meter = total_cost ∧
  length = breadth + 50 ∧
  length = 75 := by sorry

end plot_length_is_75_l3049_304967


namespace remaining_milk_average_price_l3049_304992

/-- Calculates the average price of remaining milk packets after returning some packets. -/
theorem remaining_milk_average_price
  (total_packets : ℕ)
  (initial_avg_price : ℚ)
  (returned_packets : ℕ)
  (returned_avg_price : ℚ)
  (h1 : total_packets = 5)
  (h2 : initial_avg_price = 20/100)
  (h3 : returned_packets = 2)
  (h4 : returned_avg_price = 32/100)
  : (total_packets * initial_avg_price - returned_packets * returned_avg_price) / (total_packets - returned_packets) = 12/100 := by
  sorry

end remaining_milk_average_price_l3049_304992


namespace solution_sets_correct_l3049_304979

-- Define the solution sets for each inequality
def solution_set1 : Set ℝ := {x | x < -1 ∨ x > 3/2}
def solution_set2 : Set ℝ := {x | x < 2 ∨ x ≥ 5}

-- Define the inequalities
def inequality1 (x : ℝ) : Prop := -2 * x^2 + x < -3
def inequality2 (x : ℝ) : Prop := (x + 1) / (x - 2) ≤ 2

-- Theorem stating that the solution sets are correct
theorem solution_sets_correct :
  (∀ x : ℝ, x ∈ solution_set1 ↔ inequality1 x) ∧
  (∀ x : ℝ, x ∈ solution_set2 ↔ inequality2 x) :=
by sorry

end solution_sets_correct_l3049_304979


namespace sheep_buying_problem_l3049_304906

theorem sheep_buying_problem (x : ℝ) : 
  (∃ n : ℕ, n * 5 + 45 = x ∧ n * 7 + 3 = x) → (x - 45) / 5 = (x - 3) / 7 := by
  sorry

end sheep_buying_problem_l3049_304906


namespace no_intersection_implies_k_equals_one_l3049_304993

theorem no_intersection_implies_k_equals_one (k : ℕ+) :
  (∀ x y : ℝ, x^2 + y^2 = k^2 → x * y ≠ k) → k = 1 := by
  sorry

end no_intersection_implies_k_equals_one_l3049_304993


namespace inequality_equivalence_l3049_304913

theorem inequality_equivalence (x y : ℝ) (h : x > 0) :
  (Real.sqrt (y - x) / x ≤ 1) ↔ (x ≤ y ∧ y ≤ x^2 + x) :=
by sorry

end inequality_equivalence_l3049_304913


namespace bell_pepper_cost_l3049_304953

/-- The cost of a single bell pepper given the total cost of ingredients for tacos -/
theorem bell_pepper_cost (taco_shells_cost meat_price_per_pound meat_pounds total_spent bell_pepper_count : ℚ) :
  taco_shells_cost = 5 →
  bell_pepper_count = 4 →
  meat_pounds = 2 →
  meat_price_per_pound = 3 →
  total_spent = 17 →
  (total_spent - (taco_shells_cost + meat_price_per_pound * meat_pounds)) / bell_pepper_count = 3/2 := by
sorry

end bell_pepper_cost_l3049_304953


namespace probability_point_closer_to_center_l3049_304920

theorem probability_point_closer_to_center (R : Real) (r : Real) : 
  R = 3 → r = 1.5 → (π * r^2) / (π * R^2) = 1/4 := by sorry

end probability_point_closer_to_center_l3049_304920


namespace emma_age_when_sister_is_56_l3049_304959

/-- Emma's current age -/
def emma_age : ℕ := 7

/-- Age difference between Emma and her sister -/
def age_difference : ℕ := 9

/-- Age of Emma's sister when the problem is solved -/
def sister_future_age : ℕ := 56

/-- Emma's age when her sister reaches the future age -/
def emma_future_age : ℕ := emma_age + (sister_future_age - (emma_age + age_difference))

theorem emma_age_when_sister_is_56 : emma_future_age = 47 := by
  sorry

end emma_age_when_sister_is_56_l3049_304959


namespace tripled_division_l3049_304900

theorem tripled_division (a b q r : ℤ) 
  (h1 : a = b * q + r) 
  (h2 : 0 ≤ r ∧ r < b) : 
  ∃ (r' : ℤ), 3 * a = (3 * b) * q + r' ∧ r' = 3 * r := by
sorry

end tripled_division_l3049_304900


namespace absolute_value_equality_implies_inequality_l3049_304987

theorem absolute_value_equality_implies_inequality (m : ℝ) : 
  |m - 9| = 9 - m → m ≤ 9 := by
sorry

end absolute_value_equality_implies_inequality_l3049_304987


namespace pigeon_increase_l3049_304930

theorem pigeon_increase (total : ℕ) (initial : ℕ) (h1 : total = 21) (h2 : initial = 15) :
  total - initial = 6 := by
  sorry

end pigeon_increase_l3049_304930


namespace simplest_quadratic_radical_l3049_304942

-- Define the concept of a quadratic radical
def QuadraticRadical (x : ℝ) : Prop := ∃ (y : ℝ), x = y^2

-- Define the concept of simplest quadratic radical
def SimplestQuadraticRadical (x : ℝ) : Prop :=
  QuadraticRadical x ∧ 
  ∀ y : ℝ, (QuadraticRadical y ∧ y ≠ x) → (∃ z : ℝ, z ≠ 1 ∧ y = z * x)

-- Theorem statement
theorem simplest_quadratic_radical :
  SimplestQuadraticRadical (Real.sqrt 6) ∧
  ¬SimplestQuadraticRadical (Real.sqrt 12) ∧
  ¬SimplestQuadraticRadical (Real.sqrt (1/3)) ∧
  ¬SimplestQuadraticRadical (Real.sqrt 0.3) :=
sorry

end simplest_quadratic_radical_l3049_304942


namespace cos_18_cos_42_minus_cos_72_sin_42_l3049_304948

theorem cos_18_cos_42_minus_cos_72_sin_42 :
  Real.cos (18 * π / 180) * Real.cos (42 * π / 180) - 
  Real.cos (72 * π / 180) * Real.sin (42 * π / 180) = 1 / 2 := by
  sorry

end cos_18_cos_42_minus_cos_72_sin_42_l3049_304948


namespace teresa_pencil_distribution_l3049_304962

/-- Given Teresa's pencil collection and distribution rules, prove each sibling gets 13 pencils -/
theorem teresa_pencil_distribution :
  let colored_pencils : ℕ := 14
  let black_pencils : ℕ := 35
  let total_pencils : ℕ := colored_pencils + black_pencils
  let pencils_to_keep : ℕ := 10
  let number_of_siblings : ℕ := 3
  let pencils_to_distribute : ℕ := total_pencils - pencils_to_keep
  pencils_to_distribute / number_of_siblings = 13 :=
by
  sorry

#eval (14 + 35 - 10) / 3  -- This should output 13

end teresa_pencil_distribution_l3049_304962


namespace pet_snake_cost_l3049_304946

def initial_amount : ℕ := 73
def amount_left : ℕ := 18

theorem pet_snake_cost : initial_amount - amount_left = 55 := by sorry

end pet_snake_cost_l3049_304946


namespace smallest_prime_factor_in_C_l3049_304908

def C : Set Nat := {66, 68, 71, 73, 75}

theorem smallest_prime_factor_in_C : 
  ∃ (n : Nat), n ∈ C ∧ (∀ m ∈ C, ∀ p q : Nat, Prime p → Prime q → p ∣ n → q ∣ m → p ≤ q) ∧ n = 66 := by
  sorry

end smallest_prime_factor_in_C_l3049_304908


namespace smallest_positive_difference_l3049_304947

theorem smallest_positive_difference (a b : ℤ) (h : 17 * a + 6 * b = 13) :
  ∃ (k : ℤ), k > 0 ∧ a - b = k ∧ ∀ (m : ℤ), m > 0 ∧ (∃ (x y : ℤ), 17 * x + 6 * y = 13 ∧ x - y = m) → k ≤ m :=
by sorry

end smallest_positive_difference_l3049_304947


namespace three_heads_in_ten_flips_l3049_304909

/-- The probability of flipping exactly k heads in n flips of an unfair coin -/
def unfair_coin_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k : ℚ) * p^k * (1 - p)^(n - k)

/-- The main theorem: probability of 3 heads in 10 flips of a coin with 1/3 probability of heads -/
theorem three_heads_in_ten_flips :
  unfair_coin_probability 10 3 (1/3) = 15360 / 59049 := by
  sorry

end three_heads_in_ten_flips_l3049_304909


namespace exam_results_l3049_304926

/-- Represents the score distribution of students in an examination. -/
structure ScoreDistribution where
  scores : List (Nat × Nat)
  total_students : Nat
  sum_scores : Nat

/-- The given score distribution for the examination. -/
def exam_distribution : ScoreDistribution := {
  scores := [(95, 10), (85, 30), (75, 40), (65, 45), (55, 20), (45, 15)],
  total_students := 160,
  sum_scores := 11200
}

/-- Calculate the average score from a ScoreDistribution. -/
def average_score (d : ScoreDistribution) : Rat :=
  d.sum_scores / d.total_students

/-- Calculate the percentage of students scoring at least 60%. -/
def percentage_passing (d : ScoreDistribution) : Rat :=
  let passing_students := (d.scores.filter (fun p => p.fst ≥ 60)).map (fun p => p.snd) |>.sum
  (passing_students * 100) / d.total_students

theorem exam_results :
  average_score exam_distribution = 70 ∧
  percentage_passing exam_distribution = 78125 / 1000 := by
  sorry

#eval average_score exam_distribution
#eval percentage_passing exam_distribution

end exam_results_l3049_304926


namespace cyclist_distance_l3049_304912

/-- Cyclist's travel problem -/
theorem cyclist_distance :
  ∀ (v t : ℝ),
  v > 0 →
  t > 0 →
  (v + 1) * (3/4 * t) = v * t →
  (v - 1) * (t + 3) = v * t →
  v * t = 18 :=
by sorry

end cyclist_distance_l3049_304912


namespace hotel_room_charge_difference_l3049_304976

theorem hotel_room_charge_difference (G : ℝ) (h1 : G > 0) : 
  let R := 1.5000000000000002 * G
  let P := 0.6 * R
  (G - P) / G * 100 = 10 := by sorry

end hotel_room_charge_difference_l3049_304976


namespace prime_square_mod_504_l3049_304919

theorem prime_square_mod_504 (p : Nat) (h_prime : Nat.Prime p) (h_gt_7 : p > 7) :
  ∃! (s : Finset Nat), 
    (∀ r ∈ s, r < 504 ∧ ∃ q : Nat, p^2 = 504 * q + r) ∧ 
    s.card = 3 :=
sorry

end prime_square_mod_504_l3049_304919


namespace correct_street_loss_percentage_l3049_304982

/-- The percentage of marbles lost into the street -/
def street_loss_percentage : ℝ := 60

/-- The initial number of marbles -/
def initial_marbles : ℕ := 100

/-- The final number of marbles after losses -/
def final_marbles : ℕ := 20

/-- Theorem stating the correct percentage of marbles lost into the street -/
theorem correct_street_loss_percentage :
  street_loss_percentage = 60 ∧
  final_marbles = (initial_marbles - initial_marbles * street_loss_percentage / 100) / 2 :=
by sorry

end correct_street_loss_percentage_l3049_304982


namespace exchange_ways_eq_six_l3049_304941

/-- The number of ways to exchange 100 yuan into 20 yuan and 10 yuan bills -/
def exchange_ways : ℕ :=
  (Finset.filter (fun p : ℕ × ℕ => 20 * p.1 + 10 * p.2 = 100) (Finset.product (Finset.range 6) (Finset.range 11))).card

/-- Theorem stating that there are exactly 6 ways to exchange 100 yuan into 20 yuan and 10 yuan bills -/
theorem exchange_ways_eq_six : exchange_ways = 6 := by
  sorry

end exchange_ways_eq_six_l3049_304941


namespace job_completion_time_l3049_304916

theorem job_completion_time (efficiency_ratio : ℝ) (joint_completion_time : ℝ) :
  efficiency_ratio = (1 : ℝ) / 2 →
  joint_completion_time = 15 →
  ∃ (solo_completion_time : ℝ),
    solo_completion_time = (3 / 2) * joint_completion_time ∧
    solo_completion_time = 45 / 2 :=
by sorry

end job_completion_time_l3049_304916


namespace fifty_billion_scientific_notation_l3049_304937

theorem fifty_billion_scientific_notation :
  (50000000000 : ℝ) = 5.0 * (10 : ℝ) ^ 9 := by sorry

end fifty_billion_scientific_notation_l3049_304937


namespace wheel_distance_l3049_304985

/-- Given two wheels with different perimeters, prove that the distance traveled
    is 315 feet when the front wheel makes 10 more revolutions than the back wheel. -/
theorem wheel_distance (back_perimeter front_perimeter : ℝ) 
  (h1 : back_perimeter = 9)
  (h2 : front_perimeter = 7)
  (h3 : ∃ (back_revs front_revs : ℝ), 
    front_revs = back_revs + 10 ∧ 
    back_revs * back_perimeter = front_revs * front_perimeter) :
  ∃ (distance : ℝ), distance = 315 := by
  sorry

end wheel_distance_l3049_304985


namespace concert_tickets_l3049_304954

def choose (n k : ℕ) : ℕ := (n.factorial) / ((k.factorial) * ((n - k).factorial))

theorem concert_tickets : choose 7 4 = 35 := by
  sorry

end concert_tickets_l3049_304954


namespace fault_line_movement_l3049_304932

/-- Represents the movement of a fault line over two years -/
structure FaultLineMovement where
  past_year : ℝ
  year_before : ℝ

/-- Calculates the total movement of a fault line over two years -/
def total_movement (f : FaultLineMovement) : ℝ :=
  f.past_year + f.year_before

/-- Theorem: The total movement of the fault line is 6.50 inches -/
theorem fault_line_movement :
  let f : FaultLineMovement := { past_year := 1.25, year_before := 5.25 }
  total_movement f = 6.50 := by
  sorry

end fault_line_movement_l3049_304932


namespace unique_B_for_divisible_by_7_l3049_304943

def is_divisible_by_7 (n : ℕ) : Prop := ∃ k : ℕ, n = 7 * k

def four_digit_number (B : ℕ) : ℕ := 4000 + 100 * B + 10 * B + 3

theorem unique_B_for_divisible_by_7 :
  ∀ B : ℕ, B < 10 →
    is_divisible_by_7 (four_digit_number B) →
    B = 0 := by sorry

end unique_B_for_divisible_by_7_l3049_304943


namespace ellipse_and_line_intersection_ellipse_equation_l3049_304910

/-- Definition of the ellipse based on the sum of distances from two foci -/
def is_on_ellipse (x y : ℝ) : Prop :=
  Real.sqrt ((x - 0)^2 + (y + Real.sqrt 3)^2) +
  Real.sqrt ((x - 0)^2 + (y - Real.sqrt 3)^2) = 4

/-- Definition of the line y = kx + √3 -/
def is_on_line (k x y : ℝ) : Prop := y = k * x + Real.sqrt 3

/-- Definition of a point being on the circle with diameter AB passing through origin -/
def is_on_circle (xA yA xB yB x y : ℝ) : Prop :=
  x * (xA + xB) + y * (yA + yB) = xA * xB + yA * yB

theorem ellipse_and_line_intersection :
  ∃ (k : ℝ),
    (∃ (xA yA xB yB : ℝ),
      is_on_ellipse xA yA ∧ is_on_ellipse xB yB ∧
      is_on_line k xA yA ∧ is_on_line k xB yB ∧
      is_on_circle xA yA xB yB 0 0) ∧
    k = Real.sqrt 11 / 2 ∨ k = -Real.sqrt 11 / 2 := by sorry

theorem ellipse_equation :
  ∀ (x y : ℝ), is_on_ellipse x y ↔ x^2 + y^2 / 4 = 1 := by sorry

end ellipse_and_line_intersection_ellipse_equation_l3049_304910


namespace overall_profit_l3049_304935

def refrigerator_cost : ℝ := 15000
def mobile_cost : ℝ := 8000
def refrigerator_loss_percent : ℝ := 0.03
def mobile_profit_percent : ℝ := 0.10

def refrigerator_selling_price : ℝ := refrigerator_cost * (1 - refrigerator_loss_percent)
def mobile_selling_price : ℝ := mobile_cost * (1 + mobile_profit_percent)

def total_cost : ℝ := refrigerator_cost + mobile_cost
def total_selling_price : ℝ := refrigerator_selling_price + mobile_selling_price

theorem overall_profit : total_selling_price - total_cost = 350 := by
  sorry

end overall_profit_l3049_304935


namespace factorization_of_polynomial_l3049_304905

theorem factorization_of_polynomial (x : ℝ) :
  29 * 40 * x^4 + 64 = 29 * 40 * ((x^2 - 4*x + 8) * (x^2 + 4*x + 8)) :=
by sorry

end factorization_of_polynomial_l3049_304905


namespace quadratic_roots_properties_l3049_304970

theorem quadratic_roots_properties : ∃ (a b : ℝ), 
  (a^2 + a - 2023 = 0) ∧ 
  (b^2 + b - 2023 = 0) ∧ 
  (a * b = -2023) ∧ 
  (a^2 - b = 2024) := by
  sorry

end quadratic_roots_properties_l3049_304970


namespace isabel_homework_completion_l3049_304971

/-- Given that Isabel had 72.0 homework problems in total, each problem has 5 sub tasks,
    and she has to solve 200 sub tasks, prove that she finished 40 homework problems. -/
theorem isabel_homework_completion (total : ℝ) (subtasks_per_problem : ℕ) (subtasks_solved : ℕ) 
    (h1 : total = 72.0)
    (h2 : subtasks_per_problem = 5)
    (h3 : subtasks_solved = 200) :
    (subtasks_solved : ℝ) / subtasks_per_problem = 40 := by
  sorry

#check isabel_homework_completion

end isabel_homework_completion_l3049_304971


namespace distribute_six_to_four_l3049_304922

/-- The number of ways to distribute n distinct objects into k distinct groups,
    where each group must contain at least one object. -/
def distribute (n k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 6 distinct objects into 4 distinct groups,
    where each group must contain at least one object, is 1560. -/
theorem distribute_six_to_four : distribute 6 4 = 1560 := by sorry

end distribute_six_to_four_l3049_304922


namespace quadratic_from_means_l3049_304990

theorem quadratic_from_means (a b : ℝ) 
  (h_arithmetic : (a + b) / 2 = 8) 
  (h_geometric : Real.sqrt (a * b) = 12) : 
  ∀ x, x^2 - 16*x + 144 = 0 ↔ (x = a ∨ x = b) := by
sorry

end quadratic_from_means_l3049_304990


namespace units_digit_47_power_47_l3049_304977

theorem units_digit_47_power_47 : 47^47 % 10 = 3 := by
  sorry

end units_digit_47_power_47_l3049_304977


namespace max_value_condition_l3049_304966

/-- The function f(x) = kx^2 + kx + 1 -/
def f (k : ℝ) (x : ℝ) : ℝ := k * x^2 + k * x + 1

/-- The maximum value of f(x) on the interval [-2, 2] is 4 -/
def has_max_4 (k : ℝ) : Prop :=
  ∀ x ∈ Set.Icc (-2) 2, f k x ≤ 4 ∧ ∃ y ∈ Set.Icc (-2) 2, f k y = 4

/-- The theorem stating that k = 1/2 or k = -12 if and only if
    the maximum value of f(x) on [-2, 2] is 4 -/
theorem max_value_condition (k : ℝ) :
  has_max_4 k ↔ k = 1/2 ∨ k = -12 := by sorry

end max_value_condition_l3049_304966


namespace average_marks_first_class_l3049_304950

theorem average_marks_first_class 
  (students_first_class : ℕ) 
  (students_second_class : ℕ)
  (average_second_class : ℝ)
  (average_all : ℝ) :
  students_first_class = 35 →
  students_second_class = 55 →
  average_second_class = 65 →
  average_all = 57.22222222222222 →
  (students_first_class * (average_all * (students_first_class + students_second_class) - 
   students_second_class * average_second_class)) / 
   (students_first_class * students_first_class) = 45 := by
sorry

end average_marks_first_class_l3049_304950


namespace max_value_of_function_l3049_304934

theorem max_value_of_function (x : ℝ) (h1 : 0 < x) (h2 : x < 1/2) :
  x * (1 - 2*x) ≤ 1/8 ∧ ∃ x₀, 0 < x₀ ∧ x₀ < 1/2 ∧ x₀ * (1 - 2*x₀) = 1/8 :=
by sorry

end max_value_of_function_l3049_304934


namespace buckingham_visitors_theorem_l3049_304961

/-- Represents the number of visitors to Buckingham Palace -/
structure BuckinghamVisitors where
  total_85_days : ℕ
  previous_day : ℕ

/-- Calculates the number of visitors on a specific day -/
def visitors_on_day (bv : BuckinghamVisitors) : ℕ :=
  bv.total_85_days - bv.previous_day

/-- Theorem statement for the Buckingham Palace visitor calculation -/
theorem buckingham_visitors_theorem (bv : BuckinghamVisitors) 
  (h1 : bv.total_85_days = 829)
  (h2 : bv.previous_day = 45) :
  visitors_on_day bv = 784 := by
  sorry

#eval visitors_on_day { total_85_days := 829, previous_day := 45 }

end buckingham_visitors_theorem_l3049_304961


namespace expression_values_l3049_304938

theorem expression_values (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h1 : (x + y) / z = (y + z) / x) (h2 : (y + z) / x = (z + x) / y) :
  ((x + y) * (y + z) * (z + x)) / (x * y * z) = 8 ∨
  ((x + y) * (y + z) * (z + x)) / (x * y * z) = -1 := by
sorry

end expression_values_l3049_304938


namespace intersection_slope_inequality_l3049_304999

open Real

noncomputable def f (x : ℝ) : ℝ := log x

noncomputable def F (a : ℝ) (x : ℝ) : ℝ := 3/2 * x^2 - (6+a)*x + 2*a * f x

noncomputable def g (x : ℝ) : ℝ := f x / (deriv f x)

theorem intersection_slope_inequality (a k x₁ x₂ : ℝ) (h₁ : a > 0) (h₂ : x₁ < x₂) 
  (h₃ : ∃ y₁ y₂, (k * x₁ + y₁ = deriv g x₁) ∧ (k * x₂ + y₂ = deriv g x₂)) :
  x₁ < 1/k ∧ 1/k < x₂ := by sorry

end intersection_slope_inequality_l3049_304999


namespace solution_equivalence_l3049_304984

/-- Given constants m and n where mx + n > 0 is equivalent to x < 1/2, 
    prove that nx - m < 0 is equivalent to x < -2 -/
theorem solution_equivalence (m n : ℝ) 
    (h : ∀ x, mx + n > 0 ↔ x < (1/2)) : 
    ∀ x, nx - m < 0 ↔ x < -2 := by
  sorry

end solution_equivalence_l3049_304984


namespace female_worker_ants_l3049_304957

theorem female_worker_ants (total_ants : ℕ) (worker_ratio : ℚ) (male_ratio : ℚ) : 
  total_ants = 110 →
  worker_ratio = 1/2 →
  male_ratio = 1/5 →
  ⌊(total_ants : ℚ) * worker_ratio * (1 - male_ratio)⌋ = 44 := by
sorry

end female_worker_ants_l3049_304957


namespace new_group_average_age_l3049_304996

theorem new_group_average_age 
  (initial_count : ℕ) 
  (initial_avg : ℚ) 
  (new_count : ℕ) 
  (final_avg : ℚ) :
  initial_count = 20 →
  initial_avg = 16 →
  new_count = 20 →
  final_avg = 15.5 →
  (initial_count * initial_avg + new_count * (initial_count * final_avg - initial_count * initial_avg) / new_count) / (initial_count + new_count) = 15 :=
by sorry

end new_group_average_age_l3049_304996


namespace no_real_roots_l3049_304907

theorem no_real_roots : ¬∃ x : ℝ, Real.sqrt (x + 4) - Real.sqrt (x - 3) + 1 = 0 := by
  sorry

end no_real_roots_l3049_304907


namespace main_diagonal_equals_anti_diagonal_l3049_304969

/-- Represents a square board with side length 2^n -/
structure Board (n : ℕ) where
  size : ℕ := 2^n
  elements : Fin (size * size) → ℕ

/-- Defines the initial arrangement of numbers on the board -/
def initial_board (n : ℕ) : Board n where
  elements := λ i => i.val + 1

/-- Defines the anti-diagonal of a board -/
def anti_diagonal (b : Board n) : List ℕ :=
  List.range b.size |>.map (λ i => b.elements ⟨i + (b.size - 1 - i) * b.size, sorry⟩)

/-- Represents a transformation on the board -/
def transform (b : Board n) : Board n :=
  sorry

/-- Theorem: After transformations, the main diagonal equals the original anti-diagonal -/
theorem main_diagonal_equals_anti_diagonal (n : ℕ) :
  let final_board := (transform^[n] (initial_board n))
  List.range (2^n) |>.map (λ i => final_board.elements ⟨i + i * (2^n), sorry⟩) =
  anti_diagonal (initial_board n) := by
  sorry

end main_diagonal_equals_anti_diagonal_l3049_304969


namespace min_sum_of_product_l3049_304933

theorem min_sum_of_product (a b : ℤ) (h : a * b = 196) : 
  ∀ x y : ℤ, x * y = 196 → a + b ≤ x + y ∧ ∃ a b : ℤ, a * b = 196 ∧ a + b = -197 :=
by sorry

end min_sum_of_product_l3049_304933


namespace sherman_weekly_driving_time_l3049_304923

-- Define the daily commute time in minutes
def daily_commute : ℕ := 30 + 30

-- Define the number of workdays in a week
def workdays : ℕ := 5

-- Define the weekend driving time in hours
def weekend_driving : ℕ := 2 * 2

-- Theorem statement
theorem sherman_weekly_driving_time :
  (workdays * daily_commute) / 60 + weekend_driving = 9 := by
  sorry

end sherman_weekly_driving_time_l3049_304923


namespace regular_rate_is_three_l3049_304924

/-- Represents a worker's pay structure and hours worked -/
structure PayStructure where
  regularRate : ℝ
  overtimeHours : ℝ
  totalPay : ℝ

/-- Calculates the total pay for a given pay structure -/
def calculateTotalPay (p : PayStructure) : ℝ :=
  40 * p.regularRate + p.overtimeHours * (2 * p.regularRate)

/-- Theorem stating that given the conditions, the regular rate is $3 per hour -/
theorem regular_rate_is_three (p : PayStructure) 
  (h1 : p.overtimeHours = 8)
  (h2 : p.totalPay = 168)
  (h3 : calculateTotalPay p = p.totalPay) : 
  p.regularRate = 3 := by
  sorry


end regular_rate_is_three_l3049_304924


namespace orange_harvest_theorem_l3049_304949

/-- The number of oranges harvested per day (not discarded) -/
def oranges_harvested (sacks_per_day : ℕ) (sacks_discarded : ℕ) (oranges_per_sack : ℕ) : ℕ :=
  (sacks_per_day - sacks_discarded) * oranges_per_sack

theorem orange_harvest_theorem :
  oranges_harvested 76 64 50 = 600 := by
  sorry

end orange_harvest_theorem_l3049_304949


namespace missy_patient_count_l3049_304944

/-- Represents the total number of patients Missy is attending to -/
def total_patients : ℕ := 12

/-- Represents the time (in minutes) it takes to serve all patients -/
def total_serving_time : ℕ := 64

/-- Represents the time (in minutes) to serve a standard care patient -/
def standard_serving_time : ℕ := 5

/-- Represents the fraction of patients with special dietary requirements -/
def special_diet_fraction : ℚ := 1 / 3

/-- Represents the increase in serving time for special dietary patients -/
def special_diet_time_increase : ℚ := 1 / 5

theorem missy_patient_count :
  total_patients = 12 ∧
  (special_diet_fraction * total_patients : ℚ) * 
    (standard_serving_time : ℚ) * (1 + special_diet_time_increase) +
  ((1 - special_diet_fraction) * total_patients : ℚ) * 
    (standard_serving_time : ℚ) = total_serving_time := by
  sorry

end missy_patient_count_l3049_304944


namespace intersection_of_A_and_B_l3049_304936

-- Define set A
def A : Set ℝ := Set.univ

-- Define set B
def B : Set ℝ := {y : ℝ | ∃ x : ℝ, y = -x^2 - 2*x + 3}

-- Theorem statement
theorem intersection_of_A_and_B :
  A ∩ B = Set.Iic 4 := by
  sorry

end intersection_of_A_and_B_l3049_304936


namespace value_of_A_l3049_304915

def round_down_tens (n : ℕ) : ℕ := n / 10 * 10

theorem value_of_A (A : ℕ) : 
  A < 10 → 
  round_down_tens (900 + 10 * A + 7) = 930 → 
  A = 3 := by
sorry

end value_of_A_l3049_304915


namespace custom_mul_identity_l3049_304958

/-- Custom multiplication operation -/
def custom_mul (a b : ℝ) : ℝ := 4 * a * b

/-- Theorem stating that if a * x = x for all x, then a = 1/4 -/
theorem custom_mul_identity (a : ℝ) : 
  (∀ x, custom_mul a x = x) → a = (1/4 : ℝ) := by
  sorry

end custom_mul_identity_l3049_304958


namespace fruit_arrangement_problem_l3049_304904

def number_of_arrangements (n : ℕ) (a b c d : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial a * Nat.factorial b * Nat.factorial c * Nat.factorial d)

theorem fruit_arrangement_problem : number_of_arrangements 10 4 3 2 1 = 12600 := by
  sorry

end fruit_arrangement_problem_l3049_304904


namespace percentage_free_lunch_l3049_304955

/-- Proves that 40% of students receive a free lunch given the specified conditions --/
theorem percentage_free_lunch (total_students : ℕ) (total_cost : ℚ) (paying_price : ℚ) :
  total_students = 50 →
  total_cost = 210 →
  paying_price = 7 →
  (∃ (paying_students : ℕ), paying_students * paying_price = total_cost) →
  (total_students - (total_cost / paying_price : ℚ)) / total_students = 2/5 := by
  sorry

#check percentage_free_lunch

end percentage_free_lunch_l3049_304955


namespace flour_needed_for_butter_l3049_304951

/-- Given a recipe with a ratio of butter to flour, calculate the amount of flour needed for a given amount of butter -/
theorem flour_needed_for_butter 
  (original_butter : ℚ) 
  (original_flour : ℚ) 
  (used_butter : ℚ) 
  (h1 : original_butter > 0) 
  (h2 : original_flour > 0) 
  (h3 : used_butter > 0) : 
  (used_butter / original_butter) * original_flour = 30 := by
  sorry

#check flour_needed_for_butter 2 5 12

end flour_needed_for_butter_l3049_304951


namespace f_properties_l3049_304956

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

-- State the theorem
theorem f_properties (a : ℝ) (h : a < 0) :
  (∀ x : ℝ, x ≠ 0 → f a x + f a (-1/x) ≥ 2) ∧
  (∃ x : ℝ, f a x + f a (2*x) < 1/2 ↔ -1 < a ∧ a < 0) := by
  sorry

end f_properties_l3049_304956


namespace water_purifier_theorem_l3049_304972

/-- Represents a water purifier type -/
inductive PurifierType
| A
| B

/-- Represents the costs and prices of water purifiers -/
structure PurifierInfo where
  cost_A : ℝ
  cost_B : ℝ
  price_A : ℝ
  price_B : ℝ
  filter_cost_A : ℝ
  filter_cost_B : ℝ

/-- Represents a purchasing plan -/
structure PurchasePlan where
  num_A : ℕ
  num_B : ℕ

/-- The main theorem about water purifier costs and purchasing plans -/
theorem water_purifier_theorem 
  (info : PurifierInfo)
  (h1 : info.cost_B = info.cost_A + 600)
  (h2 : 36000 / info.cost_A = 2 * (27000 / info.cost_B))
  (h3 : info.price_A = 1350)
  (h4 : info.price_B = 2100)
  (h5 : info.filter_cost_A = 400)
  (h6 : info.filter_cost_B = 500) :
  info.cost_A = 1200 ∧ 
  info.cost_B = 1800 ∧
  (∃ (plans : List PurchasePlan), 
    (∀ p ∈ plans, 
      p.num_A * info.cost_A + p.num_B * info.cost_B ≤ 60000 ∧ 
      p.num_B ≤ 8) ∧
    plans.length = 4) ∧
  (∃ (num_filters_A num_filters_B : ℕ),
    num_filters_A + num_filters_B = 6 ∧
    ∃ (p : PurchasePlan), 
      p.num_A * (info.price_A - info.cost_A) + 
      p.num_B * (info.price_B - info.cost_B) - 
      (num_filters_A * info.filter_cost_A + num_filters_B * info.filter_cost_B) = 5250) :=
by sorry

end water_purifier_theorem_l3049_304972


namespace molly_total_swim_distance_l3049_304983

def saturday_distance : ℕ := 45
def sunday_distance : ℕ := 28

theorem molly_total_swim_distance :
  saturday_distance + sunday_distance = 73 :=
by sorry

end molly_total_swim_distance_l3049_304983


namespace triangle_classification_l3049_304988

theorem triangle_classification (a b : ℝ) (A B : ℝ) (h_positive : 0 < A ∧ A < π) 
  (h_eq : a * Real.cos A = b * Real.cos B) :
  A = B ∨ A + B = π / 2 := by
  sorry

end triangle_classification_l3049_304988


namespace anita_blueberry_cartons_l3049_304940

/-- Represents the number of cartons of berries in Anita's berry cobbler problem -/
structure BerryCobbler where
  total : ℕ
  strawberries : ℕ
  to_buy : ℕ

/-- Calculates the number of blueberry cartons Anita has -/
def blueberry_cartons (bc : BerryCobbler) : ℕ :=
  bc.total - bc.strawberries - bc.to_buy

/-- Theorem stating that Anita has 9 cartons of blueberries -/
theorem anita_blueberry_cartons :
  ∀ (bc : BerryCobbler),
    bc.total = 26 → bc.strawberries = 10 → bc.to_buy = 7 →
    blueberry_cartons bc = 9 := by
  sorry

end anita_blueberry_cartons_l3049_304940


namespace stockholm_uppsala_distance_l3049_304986

/-- The distance between Stockholm and Uppsala on a map in centimeters -/
def map_distance : ℝ := 45

/-- The scale of the map, representing how many kilometers in reality one centimeter on the map represents -/
def map_scale : ℝ := 10

/-- The actual distance between Stockholm and Uppsala in kilometers -/
def actual_distance : ℝ := map_distance * map_scale

theorem stockholm_uppsala_distance :
  actual_distance = 450 :=
by sorry

end stockholm_uppsala_distance_l3049_304986


namespace cos_300_degrees_l3049_304928

theorem cos_300_degrees : Real.cos (300 * π / 180) = 1 / 2 := by
  sorry

end cos_300_degrees_l3049_304928


namespace smaller_number_proof_l3049_304994

theorem smaller_number_proof (x y : ℝ) (h1 : x + y = 12) (h2 : x - y = 20) : y = -4 := by
  sorry

end smaller_number_proof_l3049_304994


namespace parallel_lines_x_value_l3049_304997

/-- Two points in ℝ² -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in ℝ² defined by two points -/
structure Line where
  p1 : Point
  p2 : Point

/-- Check if a line is vertical -/
def isVertical (l : Line) : Prop :=
  l.p1.x = l.p2.x

/-- Two lines are parallel if they are both vertical or have the same slope -/
def areParallel (l1 l2 : Line) : Prop :=
  (isVertical l1 ∧ isVertical l2) ∨
  (¬isVertical l1 ∧ ¬isVertical l2 ∧
    (l1.p2.y - l1.p1.y) / (l1.p2.x - l1.p1.x) = (l2.p2.y - l2.p1.y) / (l2.p2.x - l2.p1.x))

theorem parallel_lines_x_value (x : ℝ) :
  let l1 : Line := { p1 := { x := -1, y := -2 }, p2 := { x := -1, y := 4 } }
  let l2 : Line := { p1 := { x := 2, y := 1 }, p2 := { x := x, y := 6 } }
  areParallel l1 l2 → x = 2 := by
  sorry

end parallel_lines_x_value_l3049_304997


namespace polynomial_factorization_l3049_304921

theorem polynomial_factorization (m : ℤ) : 
  (∀ x : ℤ, x^2 + m*x - 35 = (x - 7)*(x + 5)) → m = -2 := by
  sorry

end polynomial_factorization_l3049_304921


namespace bus_assignment_count_l3049_304914

def num_buses : ℕ := 6
def num_destinations : ℕ := 4
def num_restricted_buses : ℕ := 2

def choose (n k : ℕ) : ℕ := Nat.choose n k

def arrange (n k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)

theorem bus_assignment_count : 
  choose num_destinations 1 * arrange (num_buses - num_restricted_buses) (num_destinations - 1) = 240 := by
  sorry

end bus_assignment_count_l3049_304914


namespace complex_division_1_complex_division_2_l3049_304911

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- Theorem for the first calculation
theorem complex_division_1 : (1 - i) * (1 + 2*i) / (1 + i) = 2 - i := by sorry

-- Theorem for the second calculation
theorem complex_division_2 : ((1 + 2*i)^2 + 3*(1 - i)) / (2 + i) = 3 - 6/5 * i := by sorry

end complex_division_1_complex_division_2_l3049_304911


namespace sin_negative_ninety_degrees_l3049_304945

theorem sin_negative_ninety_degrees :
  Real.sin (- π / 2) = -1 := by
  sorry

end sin_negative_ninety_degrees_l3049_304945


namespace ab_sum_problem_l3049_304929

theorem ab_sum_problem (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (ha_upper : a < 15) (hb_upper : b < 15) 
  (h_eq : a + b + a * b = 119) : a + b = 18 ∨ a + b = 19 := by
  sorry

end ab_sum_problem_l3049_304929


namespace unique_solution_l3049_304917

-- Define the combination function
def combination (n k : ℕ) : ℕ := Nat.choose n k

-- Define the equation
def equation (x : ℕ) : Prop :=
  combination x 3 + combination x 2 = 12 * (x - 1)

-- State the theorem
theorem unique_solution :
  ∃! x : ℕ, x ≥ 3 ∧ equation x :=
sorry

end unique_solution_l3049_304917


namespace double_line_chart_capabilities_l3049_304931

/-- Represents a data set -/
structure DataSet where
  values : List ℝ

/-- Represents a double line chart -/
structure DoubleLineChart where
  dataset1 : DataSet
  dataset2 : DataSet

/-- Function to calculate changes in a dataset -/
def calculateChanges (ds : DataSet) : List ℝ := sorry

/-- Function to analyze differences between two datasets -/
def analyzeDifferences (ds1 ds2 : DataSet) : List ℝ := sorry

theorem double_line_chart_capabilities (dlc : DoubleLineChart) :
  (∃ (changes1 changes2 : List ℝ), 
     changes1 = calculateChanges dlc.dataset1 ∧ 
     changes2 = calculateChanges dlc.dataset2) ∧
  (∃ (differences : List ℝ), 
     differences = analyzeDifferences dlc.dataset1 dlc.dataset2) := by
  sorry

end double_line_chart_capabilities_l3049_304931


namespace milk_addition_rate_l3049_304991

/-- Calculates the rate of milk addition given initial conditions --/
theorem milk_addition_rate
  (initial_milk : ℝ)
  (pump_rate : ℝ)
  (pump_time : ℝ)
  (addition_time : ℝ)
  (final_milk : ℝ)
  (h1 : initial_milk = 30000)
  (h2 : pump_rate = 2880)
  (h3 : pump_time = 4)
  (h4 : addition_time = 7)
  (h5 : final_milk = 28980) :
  let milk_pumped := pump_rate * pump_time
  let milk_before_addition := initial_milk - milk_pumped
  let milk_added := final_milk - milk_before_addition
  milk_added / addition_time = 1500 := by
  sorry

end milk_addition_rate_l3049_304991


namespace parallelogram_angles_l3049_304998

theorem parallelogram_angles (A B C D : Real) : 
  -- ABCD is a parallelogram
  (A + C = 180) →
  (B + D = 180) →
  -- ∠B - ∠A = 30°
  (B - A = 30) →
  -- Prove that ∠A = 75°, ∠B = 105°, ∠C = 75°, and ∠D = 105°
  (A = 75 ∧ B = 105 ∧ C = 75 ∧ D = 105) := by
sorry

end parallelogram_angles_l3049_304998


namespace course_selection_schemes_l3049_304927

/-- The number of elective courses in each category (physical education and art) -/
def n : ℕ := 4

/-- The total number of different course selection schemes -/
def total_schemes : ℕ := (n * n) + (n * (n - 1) * n) / 2

/-- Theorem stating that the total number of course selection schemes is 64 -/
theorem course_selection_schemes :
  total_schemes = 64 := by sorry

end course_selection_schemes_l3049_304927


namespace barkley_buried_bones_l3049_304918

/-- Calculates the number of bones Barkley has buried given the conditions -/
def bones_buried (bones_per_month : ℕ) (months_passed : ℕ) (available_bones : ℕ) : ℕ :=
  bones_per_month * months_passed - available_bones

/-- Theorem stating that Barkley has buried 42 bones under the given conditions -/
theorem barkley_buried_bones : 
  bones_buried 10 5 8 = 42 := by
  sorry

end barkley_buried_bones_l3049_304918


namespace complex_square_problem_l3049_304973

theorem complex_square_problem (a b : ℝ) (i : ℂ) :
  i^2 = -1 →
  a + i = 2 - b*i →
  (a + b*i)^2 = 3 - 4*i := by
sorry

end complex_square_problem_l3049_304973


namespace number_of_children_l3049_304902

/-- Given a person with some children and money to distribute, prove the number of children. -/
theorem number_of_children (total_money : ℕ) (share_d_and_e : ℕ) (children : List String) : 
  total_money = 12000 → share_d_and_e = 4800 → 
  children = ["a", "b", "c", "d", "e"] → 
  children.length = 5 := by
  sorry

end number_of_children_l3049_304902


namespace inscribed_circle_tangent_triangle_area_l3049_304963

/-- Given a right triangle with hypotenuse c, area T, and an inscribed circle of radius ρ,
    the area of the triangle formed by the points where the inscribed circle touches the sides
    of the right triangle is equal to (ρ/c) * T. -/
theorem inscribed_circle_tangent_triangle_area
  (c T ρ : ℝ)
  (h_positive : c > 0 ∧ T > 0 ∧ ρ > 0)
  (h_right_triangle : ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a^2 + b^2 = c^2)
  (h_area : T = (a + b + c) * ρ / 2)
  (h_inscribed : ρ = T / (a + b + c)) :
  (ρ / c) * T = (area_of_tangent_triangle : ℝ) :=
sorry

end inscribed_circle_tangent_triangle_area_l3049_304963


namespace pudding_cups_problem_l3049_304975

theorem pudding_cups_problem (students : ℕ) (additional_cups : ℕ) 
  (h1 : students = 218) 
  (h2 : additional_cups = 121) : 
  ∃ initial_cups : ℕ, 
    initial_cups + additional_cups = students ∧ 
    initial_cups = 97 := by
  sorry

end pudding_cups_problem_l3049_304975


namespace comic_book_stacks_theorem_l3049_304981

/-- The number of ways to stack comic books -/
def comic_book_stacks (spiderman : ℕ) (archie : ℕ) (garfield : ℕ) : ℕ :=
  (spiderman.factorial * archie.factorial * garfield.factorial * 2)

/-- Theorem: The number of ways to stack 7 Spiderman, 5 Archie, and 4 Garfield comic books,
    with Archie books on top and each series stacked together, is 29,030,400 -/
theorem comic_book_stacks_theorem :
  comic_book_stacks 7 5 4 = 29030400 := by
  sorry

end comic_book_stacks_theorem_l3049_304981


namespace p_necessary_not_sufficient_for_q_l3049_304995

theorem p_necessary_not_sufficient_for_q :
  (∀ x : ℝ, x ≥ 2 → x ≠ 1) ∧
  (∃ x : ℝ, x ≠ 1 ∧ x < 2) :=
by sorry

end p_necessary_not_sufficient_for_q_l3049_304995


namespace vector_problem_l3049_304974

def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (-3, 2)

theorem vector_problem :
  (∃ k : ℝ, k * a.1 + 2 * b.1 = 14 * (k * a.2 + 2 * b.2) / (-4) ∧ k = -1) ∧
  (∃ c : ℝ × ℝ, (c.1^2 + c.2^2 = 1) ∧
    ((c.1 + 3)^2 + (c.2 - 2)^2 = 20) ∧
    ((c = (5/13, -12/13)) ∨ (c = (1, 0)))) :=
by sorry

end vector_problem_l3049_304974


namespace new_student_weight_l3049_304901

theorem new_student_weight
  (n : ℕ)
  (initial_avg : ℝ)
  (new_avg : ℝ)
  (h1 : n = 29)
  (h2 : initial_avg = 28)
  (h3 : new_avg = 27.3) :
  (n + 1) * new_avg - n * initial_avg = 7 :=
by sorry

end new_student_weight_l3049_304901


namespace remaining_bag_weight_l3049_304960

def bag_weights : List ℕ := [15, 16, 18, 19, 20, 31]

def is_valid_partition (partition : List ℕ × List ℕ) : Prop :=
  let (group1, group2) := partition
  group1.length + group2.length = 5 ∧
  group1.sum = 2 * group2.sum ∧
  (∀ w ∈ group1, w ∈ bag_weights) ∧
  (∀ w ∈ group2, w ∈ bag_weights) ∧
  (∀ w ∈ group1, w ∉ group2) ∧
  (∀ w ∈ group2, w ∉ group1)

theorem remaining_bag_weight :
  ∃ (partition : List ℕ × List ℕ), is_valid_partition partition →
  bag_weights.sum - (partition.1.sum + partition.2.sum) = 20 :=
sorry

end remaining_bag_weight_l3049_304960


namespace marble_problem_l3049_304903

theorem marble_problem (initial_marbles : ℕ) : 
  (initial_marbles * 40 / 100 / 2 = 20) → initial_marbles = 100 := by
  sorry

end marble_problem_l3049_304903


namespace sum_areas_tangent_circles_l3049_304925

/-- Three mutually externally tangent circles whose centers form a 5-12-13 right triangle -/
structure TangentCircles where
  /-- Radius of the circle centered at the vertex opposite the side of length 5 -/
  a : ℝ
  /-- Radius of the circle centered at the vertex opposite the side of length 12 -/
  b : ℝ
  /-- Radius of the circle centered at the vertex opposite the side of length 13 -/
  c : ℝ
  /-- The circles are mutually externally tangent -/
  tangent_5 : a + b = 5
  tangent_12 : a + c = 12
  tangent_13 : b + c = 13

/-- The sum of the areas of three mutually externally tangent circles 
    whose centers form a 5-12-13 right triangle is 113π -/
theorem sum_areas_tangent_circles (circles : TangentCircles) :
  π * (circles.a^2 + circles.b^2 + circles.c^2) = 113 * π := by
  sorry

end sum_areas_tangent_circles_l3049_304925


namespace tan_alpha_plus_pi_fourth_l3049_304978

/-- If (3sin(α) + 2cos(α)) / (2sin(α) - cos(α)) = 8/3, then tan(α + π/4) = -3 -/
theorem tan_alpha_plus_pi_fourth (α : ℝ) 
  (h : (3 * Real.sin α + 2 * Real.cos α) / (2 * Real.sin α - Real.cos α) = 8/3) : 
  Real.tan (α + π/4) = -3 := by
  sorry

end tan_alpha_plus_pi_fourth_l3049_304978


namespace negation_of_forall_positive_negation_of_positive_square_plus_x_l3049_304965

theorem negation_of_forall_positive (P : ℝ → Prop) :
  (¬ ∀ x > 0, P x) ↔ (∃ x > 0, ¬ P x) := by sorry

theorem negation_of_positive_square_plus_x :
  (¬ ∀ x > 0, x^2 + x > 0) ↔ (∃ x > 0, x^2 + x ≤ 0) := by sorry

end negation_of_forall_positive_negation_of_positive_square_plus_x_l3049_304965


namespace flowers_picked_l3049_304939

/-- Proves that if a person can make 7 bouquets with 8 flowers each after 10 flowers have wilted,
    then they initially picked 66 flowers. -/
theorem flowers_picked (bouquets : ℕ) (flowers_per_bouquet : ℕ) (wilted_flowers : ℕ) :
  bouquets = 7 →
  flowers_per_bouquet = 8 →
  wilted_flowers = 10 →
  bouquets * flowers_per_bouquet + wilted_flowers = 66 :=
by sorry

end flowers_picked_l3049_304939


namespace team_a_more_uniform_l3049_304968

/-- Represents a dance team -/
structure DanceTeam where
  name : String
  variance : ℝ

/-- Compares the uniformity of heights between two dance teams -/
def more_uniform_heights (team1 team2 : DanceTeam) : Prop :=
  team1.variance < team2.variance

/-- The problem statement -/
theorem team_a_more_uniform : 
  let team_a : DanceTeam := ⟨"A", 1.5⟩
  let team_b : DanceTeam := ⟨"B", 2.4⟩
  more_uniform_heights team_a team_b := by
  sorry

end team_a_more_uniform_l3049_304968


namespace factorization_difference_l3049_304964

theorem factorization_difference (y : ℝ) (a b : ℤ) : 
  3 * y^2 - y - 24 = (3*y + a) * (y + b) → a - b = 11 := by
sorry

end factorization_difference_l3049_304964


namespace land_area_calculation_l3049_304989

theorem land_area_calculation (average_yield total_area first_area first_yield second_yield : ℝ) : 
  average_yield = 675 →
  first_area = 5 →
  first_yield = 705 →
  second_yield = 650 →
  total_area * average_yield = first_area * first_yield + (total_area - first_area) * second_yield →
  total_area = 11 :=
by sorry

end land_area_calculation_l3049_304989
