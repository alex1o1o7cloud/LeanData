import Mathlib

namespace NUMINAMATH_CALUDE_multiply_divide_equality_l3488_348889

theorem multiply_divide_equality : (3.241 * 14) / 100 = 0.45374 := by
  sorry

end NUMINAMATH_CALUDE_multiply_divide_equality_l3488_348889


namespace NUMINAMATH_CALUDE_polynomial_derivative_sum_l3488_348818

theorem polynomial_derivative_sum (a a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (2*x - 3)^5 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₁ + 2*a₂ + 3*a₃ + 4*a₄ + 5*a₅ = 10 := by
sorry

end NUMINAMATH_CALUDE_polynomial_derivative_sum_l3488_348818


namespace NUMINAMATH_CALUDE_petya_friends_count_l3488_348887

/-- The number of Petya's friends -/
def num_friends : ℕ := 19

/-- The number of stickers Petya has -/
def total_stickers : ℕ := num_friends * 5 + 8

/-- Theorem: Petya has 19 friends -/
theorem petya_friends_count : 
  (total_stickers = num_friends * 5 + 8) ∧ 
  (total_stickers = num_friends * 6 - 11) → 
  num_friends = 19 :=
by
  sorry


end NUMINAMATH_CALUDE_petya_friends_count_l3488_348887


namespace NUMINAMATH_CALUDE_ellipse_right_triangle_l3488_348883

-- Define the ellipse
def Γ (a b : ℝ) (x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the vertices
def A (a b : ℝ) : ℝ × ℝ := (-a, 0)
def B (a b : ℝ) : ℝ × ℝ := (a, 0)
def C (a b : ℝ) : ℝ × ℝ := (0, b)
def D (a b : ℝ) : ℝ × ℝ := (0, -b)

-- Define the theorem
theorem ellipse_right_triangle (a b : ℝ) (P Q R : ℝ × ℝ) :
  a > b ∧ b > 0 ∧
  Γ a b P.1 P.2 ∧
  Γ a b Q.1 Q.2 ∧
  P.1 ≥ 0 ∧ P.2 ≥ 0 ∧
  Q.1 ≥ 0 ∧ Q.2 ≥ 0 ∧
  (∃ k : ℝ, Q = k • (A a b - P)) ∧
  (∃ t : ℝ, R = t • ((P.1 / 2, P.2 / 2) : ℝ × ℝ)) ∧
  Γ a b R.1 R.2 →
  ‖Q‖^2 + ‖R‖^2 = ‖B a b - C a b‖^2 :=
sorry

end NUMINAMATH_CALUDE_ellipse_right_triangle_l3488_348883


namespace NUMINAMATH_CALUDE_marbles_remainder_l3488_348870

theorem marbles_remainder (a b c : ℤ) 
  (ha : a % 8 = 5)
  (hb : b % 8 = 7)
  (hc : c % 8 = 2) : 
  (a + b + c) % 8 = 6 := by
sorry

end NUMINAMATH_CALUDE_marbles_remainder_l3488_348870


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l3488_348839

-- Define the ★ operation
def star (a b : ℤ) : ℤ := a * b - 1

-- Theorem 1
theorem problem_1 : star (-1) 3 = -4 := by sorry

-- Theorem 2
theorem problem_2 : star (-2) (star (-3) (-4)) = -21 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l3488_348839


namespace NUMINAMATH_CALUDE_vocabulary_test_score_l3488_348806

theorem vocabulary_test_score (total_words : ℕ) (target_score : ℚ) 
  (h1 : total_words = 600) 
  (h2 : target_score = 90 / 100) : 
  ∃ (words_to_learn : ℕ), 
    (words_to_learn : ℚ) / total_words = target_score ∧ 
    words_to_learn = 540 := by
  sorry

end NUMINAMATH_CALUDE_vocabulary_test_score_l3488_348806


namespace NUMINAMATH_CALUDE_remainder_seven_n_mod_three_l3488_348867

theorem remainder_seven_n_mod_three (n : ℤ) (h : n % 3 = 2) : (7 * n) % 3 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_seven_n_mod_three_l3488_348867


namespace NUMINAMATH_CALUDE_carol_and_alex_peanuts_l3488_348833

def peanut_distribution (initial_peanuts : ℕ) (multiplier : ℕ) (num_people : ℕ) : ℕ :=
  (initial_peanuts + initial_peanuts * multiplier) / num_people

theorem carol_and_alex_peanuts :
  peanut_distribution 2 5 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_carol_and_alex_peanuts_l3488_348833


namespace NUMINAMATH_CALUDE_influenza_virus_diameter_l3488_348800

theorem influenza_virus_diameter (n : ℤ) : 0.000000203 = 2.03 * (10 : ℝ) ^ n → n = -7 := by
  sorry

end NUMINAMATH_CALUDE_influenza_virus_diameter_l3488_348800


namespace NUMINAMATH_CALUDE_gcd_192_144_320_l3488_348872

theorem gcd_192_144_320 : Nat.gcd 192 (Nat.gcd 144 320) = 16 := by sorry

end NUMINAMATH_CALUDE_gcd_192_144_320_l3488_348872


namespace NUMINAMATH_CALUDE_roberts_soccer_kicks_l3488_348854

theorem roberts_soccer_kicks (kicks_before_break kicks_after_break kicks_remaining : ℕ) :
  kicks_before_break = 43 →
  kicks_after_break = 36 →
  kicks_remaining = 19 →
  kicks_before_break + kicks_after_break + kicks_remaining = 98 := by
  sorry

end NUMINAMATH_CALUDE_roberts_soccer_kicks_l3488_348854


namespace NUMINAMATH_CALUDE_corporation_full_time_employees_l3488_348881

/-- Given a corporation with part-time and full-time employees, 
    we calculate the number of full-time employees. -/
theorem corporation_full_time_employees 
  (total_employees : ℕ) 
  (part_time_employees : ℕ) 
  (h1 : total_employees = 65134) 
  (h2 : part_time_employees = 2041) : 
  total_employees - part_time_employees = 63093 := by
  sorry

end NUMINAMATH_CALUDE_corporation_full_time_employees_l3488_348881


namespace NUMINAMATH_CALUDE_complex_square_equation_l3488_348890

theorem complex_square_equation : 
  ∀ z : ℂ, z^2 = -57 - 48*I ↔ z = 3 - 8*I ∨ z = -3 + 8*I :=
by sorry

end NUMINAMATH_CALUDE_complex_square_equation_l3488_348890


namespace NUMINAMATH_CALUDE_candy_distribution_l3488_348803

theorem candy_distribution (n : ℕ) (h : n = 30) :
  (min (n % 4) ((4 - n % 4) % 4)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_candy_distribution_l3488_348803


namespace NUMINAMATH_CALUDE_largest_m_satisfying_inequality_l3488_348842

theorem largest_m_satisfying_inequality :
  ∀ m : ℕ, (1 : ℚ) / 4 + (m : ℚ) / 6 < 3 / 2 ↔ m ≤ 7 :=
sorry

end NUMINAMATH_CALUDE_largest_m_satisfying_inequality_l3488_348842


namespace NUMINAMATH_CALUDE_find_fifth_month_sale_l3488_348893

def sales_problem (sales : Fin 6 → ℕ) (average : ℕ) : Prop :=
  sales 0 = 800 ∧
  sales 1 = 900 ∧
  sales 2 = 1000 ∧
  sales 3 = 700 ∧
  sales 5 = 900 ∧
  (sales 0 + sales 1 + sales 2 + sales 3 + sales 4 + sales 5) / 6 = average

theorem find_fifth_month_sale (sales : Fin 6 → ℕ) (average : ℕ) 
  (h : sales_problem sales average) : sales 4 = 800 := by
  sorry

end NUMINAMATH_CALUDE_find_fifth_month_sale_l3488_348893


namespace NUMINAMATH_CALUDE_smallest_k_value_l3488_348869

theorem smallest_k_value (a b c d x y z t : ℝ) :
  ∃ k : ℝ, k = 1 ∧ 
  (∀ k' : ℝ, (a * d - b * c + y * z - x * t + (a + c) * (y + t) - (b + d) * (x + z) ≤ 
    k' * (Real.sqrt (a^2 + b^2) + Real.sqrt (c^2 + d^2) + Real.sqrt (x^2 + y^2) + Real.sqrt (z^2 + t^2))^2) → 
  k ≤ k') ∧
  (a * d - b * c + y * z - x * t + (a + c) * (y + t) - (b + d) * (x + z) ≤ 
    k * (Real.sqrt (a^2 + b^2) + Real.sqrt (c^2 + d^2) + Real.sqrt (x^2 + y^2) + Real.sqrt (z^2 + t^2))^2) := by
  sorry

end NUMINAMATH_CALUDE_smallest_k_value_l3488_348869


namespace NUMINAMATH_CALUDE_painting_cost_is_147_l3488_348847

/-- Represents a side of the street with houses --/
structure StreetSide where
  start : ℕ
  diff : ℕ
  count : ℕ

/-- Calculates the cost of painting house numbers for a given street side --/
def calculate_side_cost (side : StreetSide) : ℕ := sorry

/-- Calculates the additional cost for numbers that are multiples of 10 --/
def calculate_multiples_of_10_cost (south : StreetSide) (north : StreetSide) : ℕ := sorry

/-- Main theorem: The total cost of painting all house numbers is $147 --/
theorem painting_cost_is_147 
  (south : StreetSide)
  (north : StreetSide)
  (h_south : south = { start := 5, diff := 7, count := 25 })
  (h_north : north = { start := 6, diff := 8, count := 25 }) :
  calculate_side_cost south + calculate_side_cost north + 
  calculate_multiples_of_10_cost south north = 147 := by sorry

end NUMINAMATH_CALUDE_painting_cost_is_147_l3488_348847


namespace NUMINAMATH_CALUDE_coefficient_x_cubed_sum_binomial_l3488_348814

theorem coefficient_x_cubed_sum_binomial (n : ℕ) (hn : n ≥ 3) :
  (Finset.range (n - 2)).sum (fun k => Nat.choose (k + 3) 3) = Nat.choose (n + 1) 4 := by
  sorry

#check coefficient_x_cubed_sum_binomial 2005

end NUMINAMATH_CALUDE_coefficient_x_cubed_sum_binomial_l3488_348814


namespace NUMINAMATH_CALUDE_mean_squares_sum_l3488_348879

theorem mean_squares_sum (x y z : ℝ) : 
  (x + y + z) / 3 = 10 →
  (x * y * z) ^ (1/3 : ℝ) = 6 →
  3 / (1/x + 1/y + 1/z) = 4 →
  x^2 + y^2 + z^2 = 576 := by
sorry

end NUMINAMATH_CALUDE_mean_squares_sum_l3488_348879


namespace NUMINAMATH_CALUDE_solution_set_part1_range_of_a_part2_l3488_348864

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + x

-- Part 1
theorem solution_set_part1 : 
  {x : ℝ | f 1 x ≤ 7} = Set.Iic 4 := by sorry

-- Part 2
theorem range_of_a_part2 : 
  {a : ℝ | ∀ x, f a x ≥ 2*a + 1} = Set.Iic (-1) := by sorry

end NUMINAMATH_CALUDE_solution_set_part1_range_of_a_part2_l3488_348864


namespace NUMINAMATH_CALUDE_valid_plans_count_l3488_348873

/-- Represents the three universities --/
inductive University : Type
| Peking : University
| Tsinghua : University
| Renmin : University

/-- Represents the five students --/
inductive Student : Type
| A : Student
| B : Student
| C : Student
| D : Student
| E : Student

/-- A recommendation plan is a function from Student to University --/
def RecommendationPlan := Student → University

/-- Checks if a recommendation plan is valid --/
def isValidPlan (plan : RecommendationPlan) : Prop :=
  (∃ s, plan s = University.Peking) ∧
  (∃ s, plan s = University.Tsinghua) ∧
  (∃ s, plan s = University.Renmin) ∧
  (plan Student.A ≠ University.Peking)

/-- The number of valid recommendation plans --/
def numberOfValidPlans : ℕ := sorry

theorem valid_plans_count : numberOfValidPlans = 100 := by sorry

end NUMINAMATH_CALUDE_valid_plans_count_l3488_348873


namespace NUMINAMATH_CALUDE_Q_characterization_l3488_348848

def Ω : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 ≤ 2008}

def superior (p q : ℝ × ℝ) : Prop := p.1 ≤ q.1 ∧ p.2 ≥ q.2

def Q : Set (ℝ × ℝ) := {q ∈ Ω | ∀ p ∈ Ω, superior p q → p = q}

theorem Q_characterization : Q = {p ∈ Ω | p.1^2 + p.2^2 = 2008 ∧ p.1 ≤ 0 ∧ p.2 ≥ 0} := by sorry

end NUMINAMATH_CALUDE_Q_characterization_l3488_348848


namespace NUMINAMATH_CALUDE_johns_drive_distance_l3488_348823

/-- Represents the total distance of John's drive in miles -/
def total_distance : ℝ := 360

/-- Represents the initial distance driven on battery alone in miles -/
def battery_distance : ℝ := 60

/-- Represents the gasoline consumption rate in gallons per mile -/
def gasoline_rate : ℝ := 0.03

/-- Represents the average fuel efficiency in miles per gallon -/
def avg_fuel_efficiency : ℝ := 40

/-- Theorem stating that given the conditions, the total distance of John's drive is 360 miles -/
theorem johns_drive_distance :
  total_distance = battery_distance + 
  (total_distance - battery_distance) * gasoline_rate * avg_fuel_efficiency :=
by sorry

end NUMINAMATH_CALUDE_johns_drive_distance_l3488_348823


namespace NUMINAMATH_CALUDE_age_ratio_proof_l3488_348857

/-- Prove that given the conditions, the ratio of B's age to C's age is 2:1 -/
theorem age_ratio_proof (A B C : ℕ) : 
  A = B + 2 →  -- A is two years older than B
  A + B + C = 37 →  -- The total of the ages of A, B, and C is 37
  B = 14 →  -- B is 14 years old
  B / C = 2  -- The ratio of B's age to C's age is 2:1
  :=
by sorry

end NUMINAMATH_CALUDE_age_ratio_proof_l3488_348857


namespace NUMINAMATH_CALUDE_car_value_reduction_l3488_348821

-- Define the original price of the car
def original_price : ℝ := 4000

-- Define the reduction rate
def reduction_rate : ℝ := 0.30

-- Define the current value of the car
def current_value : ℝ := original_price * (1 - reduction_rate)

-- Theorem to prove
theorem car_value_reduction : current_value = 2800 := by
  sorry

end NUMINAMATH_CALUDE_car_value_reduction_l3488_348821


namespace NUMINAMATH_CALUDE_cross_product_result_l3488_348882

def a : ℝ × ℝ × ℝ := (4, 3, -7)
def b : ℝ × ℝ × ℝ := (2, -1, 4)

def cross_product (v w : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (v.2.1 * w.2.2 - v.2.2 * w.2.1,
   v.2.2 * w.1 - v.1 * w.2.2,
   v.1 * w.2.1 - v.2.1 * w.1)

theorem cross_product_result : cross_product a b = (5, -30, -10) := by
  sorry

end NUMINAMATH_CALUDE_cross_product_result_l3488_348882


namespace NUMINAMATH_CALUDE_paco_cookies_bought_l3488_348826

-- Define the initial number of cookies
def initial_cookies : ℕ := 13

-- Define the number of cookies eaten
def cookies_eaten : ℕ := 2

-- Define the additional cookies compared to eaten ones
def additional_cookies : ℕ := 34

-- Define the function to calculate the number of cookies bought
def cookies_bought (initial : ℕ) (eaten : ℕ) (additional : ℕ) : ℕ :=
  additional + eaten

-- Theorem statement
theorem paco_cookies_bought :
  cookies_bought initial_cookies cookies_eaten additional_cookies = 36 := by
  sorry

end NUMINAMATH_CALUDE_paco_cookies_bought_l3488_348826


namespace NUMINAMATH_CALUDE_sum_of_a_and_b_l3488_348871

theorem sum_of_a_and_b (a b : ℝ) (h1 : |a| = 1) (h2 : |b| = 4) (h3 : a * b < 0) :
  a + b = 3 ∨ a + b = -3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_a_and_b_l3488_348871


namespace NUMINAMATH_CALUDE_simplify_expression_l3488_348897

theorem simplify_expression (s : ℝ) : 120 * s - 32 * s = 88 * s := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3488_348897


namespace NUMINAMATH_CALUDE_bear_food_per_day_l3488_348880

/-- The weight of Victor in pounds -/
def victor_weight : ℝ := 126

/-- The number of "Victors" worth of food a bear eats in 3 weeks -/
def victors_eaten : ℝ := 15

/-- The number of weeks -/
def weeks : ℝ := 3

/-- The number of days in a week -/
def days_per_week : ℝ := 7

/-- Theorem: A bear eats 90 pounds of food per day -/
theorem bear_food_per_day :
  (victor_weight * victors_eaten) / (weeks * days_per_week) = 90 := by
  sorry

end NUMINAMATH_CALUDE_bear_food_per_day_l3488_348880


namespace NUMINAMATH_CALUDE_shaded_rectangle_perimeter_l3488_348868

theorem shaded_rectangle_perimeter
  (total_perimeter : ℝ)
  (square_area : ℝ)
  (h_total_perimeter : total_perimeter = 30)
  (h_square_area : square_area = 9) :
  let square_side := Real.sqrt square_area
  let remaining_sum := (total_perimeter / 2) - 2 * square_side
  2 * remaining_sum = 18 :=
by sorry

end NUMINAMATH_CALUDE_shaded_rectangle_perimeter_l3488_348868


namespace NUMINAMATH_CALUDE_prob_not_face_card_is_ten_thirteenths_l3488_348829

-- Define the total number of cards in a standard deck
def total_cards : ℕ := 52

-- Define the number of face cards in a standard deck
def face_cards : ℕ := 12

-- Define the probability of not getting a face card
def prob_not_face_card : ℚ := (total_cards - face_cards) / total_cards

-- Theorem statement
theorem prob_not_face_card_is_ten_thirteenths :
  prob_not_face_card = 10 / 13 := by sorry

end NUMINAMATH_CALUDE_prob_not_face_card_is_ten_thirteenths_l3488_348829


namespace NUMINAMATH_CALUDE_quadratic_completing_square_l3488_348813

theorem quadratic_completing_square :
  ∀ x : ℝ, x^2 - 6*x - 5 = 0 ↔ (x - 3)^2 = 14 := by
sorry

end NUMINAMATH_CALUDE_quadratic_completing_square_l3488_348813


namespace NUMINAMATH_CALUDE_girls_combined_score_is_87_l3488_348843

-- Define the schools
structure School where
  boys_score : ℝ
  girls_score : ℝ
  combined_score : ℝ

-- Define the problem parameters
def cedar : School := { boys_score := 68, girls_score := 80, combined_score := 73 }
def drake : School := { boys_score := 75, girls_score := 88, combined_score := 83 }
def combined_boys_score : ℝ := 74

-- Theorem statement
theorem girls_combined_score_is_87 :
  ∃ (cedar_boys cedar_girls drake_boys drake_girls : ℕ),
    (cedar_boys : ℝ) * cedar.boys_score + (cedar_girls : ℝ) * cedar.girls_score = 
      (cedar_boys + cedar_girls : ℝ) * cedar.combined_score ∧
    (drake_boys : ℝ) * drake.boys_score + (drake_girls : ℝ) * drake.girls_score = 
      (drake_boys + drake_girls : ℝ) * drake.combined_score ∧
    ((cedar_boys : ℝ) * cedar.boys_score + (drake_boys : ℝ) * drake.boys_score) / 
      (cedar_boys + drake_boys : ℝ) = combined_boys_score ∧
    ((cedar_girls : ℝ) * cedar.girls_score + (drake_girls : ℝ) * drake.girls_score) / 
      (cedar_girls + drake_girls : ℝ) = 87 := by
  sorry

end NUMINAMATH_CALUDE_girls_combined_score_is_87_l3488_348843


namespace NUMINAMATH_CALUDE_triangle_angle_calculation_l3488_348891

theorem triangle_angle_calculation (A B C : Real) (a b c : Real) :
  -- Triangle ABC
  -- b = √2
  b = Real.sqrt 2 →
  -- c = 1
  c = 1 →
  -- B = 45°
  B = 45 * π / 180 →
  -- Then C = 30°
  C = 30 * π / 180 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_calculation_l3488_348891


namespace NUMINAMATH_CALUDE_usual_work_week_l3488_348895

/-- Proves that given the conditions, the employee's usual work week is 40 hours -/
theorem usual_work_week (hourly_rate : ℝ) (weekly_salary : ℝ) (worked_fraction : ℝ) :
  hourly_rate = 15 →
  weekly_salary = 480 →
  worked_fraction = 4 / 5 →
  worked_fraction * (weekly_salary / hourly_rate) = 40 := by
sorry

end NUMINAMATH_CALUDE_usual_work_week_l3488_348895


namespace NUMINAMATH_CALUDE_percent_teachers_without_conditions_l3488_348801

theorem percent_teachers_without_conditions (total : ℕ) (high_bp : ℕ) (heart_trouble : ℕ) (both : ℕ)
  (h1 : total = 150)
  (h2 : high_bp = 90)
  (h3 : heart_trouble = 50)
  (h4 : both = 30) :
  (((total - (high_bp + heart_trouble - both)) : ℚ) / total) * 100 = 2667 / 100 :=
sorry

end NUMINAMATH_CALUDE_percent_teachers_without_conditions_l3488_348801


namespace NUMINAMATH_CALUDE_pizza_toppings_combinations_l3488_348860

/-- The number of combinations of k items chosen from a set of n items. -/
def combinations (n k : ℕ) : ℕ :=
  if k ≤ n then
    Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))
  else
    0

/-- Theorem: The number of combinations of 3 toppings chosen from 7 available toppings is 35. -/
theorem pizza_toppings_combinations :
  combinations 7 3 = 35 := by
  sorry

end NUMINAMATH_CALUDE_pizza_toppings_combinations_l3488_348860


namespace NUMINAMATH_CALUDE_root_difference_of_cubic_l3488_348809

theorem root_difference_of_cubic (x : ℝ → ℝ) :
  (∀ t, 81 * (x t)^3 - 162 * (x t)^2 + 81 * (x t) - 8 = 0) →
  (∃ a d, ∀ t, x t = a + d * t) →
  (∃ t₁ t₂, ∀ t, x t₁ ≤ x t ∧ x t ≤ x t₂) →
  x t₂ - x t₁ = 4 * Real.sqrt 6 / 9 := by
sorry

end NUMINAMATH_CALUDE_root_difference_of_cubic_l3488_348809


namespace NUMINAMATH_CALUDE_four_boxes_volume_l3488_348830

/-- The volume of a cube with edge length a -/
def cube_volume (a : ℝ) : ℝ := a^3

/-- The total volume of n identical cubes with edge length a -/
def total_volume (n : ℕ) (a : ℝ) : ℝ := n * (cube_volume a)

/-- Theorem: The total volume of four cubic boxes, each with an edge length of 5 feet, is 500 cubic feet -/
theorem four_boxes_volume : total_volume 4 5 = 500 := by
  sorry

end NUMINAMATH_CALUDE_four_boxes_volume_l3488_348830


namespace NUMINAMATH_CALUDE_insufficient_information_for_unique_solution_l3488_348819

theorem insufficient_information_for_unique_solution :
  ∀ (x y z w : ℕ),
  x + y + z + w = 750 →
  10 * x + 20 * y + 50 * z + 100 * w = 27500 →
  ∃ (y' : ℕ), y ≠ y' ∧
  ∃ (x' z' w' : ℕ),
  x' + y' + z' + w' = 750 ∧
  10 * x' + 20 * y' + 50 * z' + 100 * w' = 27500 :=
by sorry

end NUMINAMATH_CALUDE_insufficient_information_for_unique_solution_l3488_348819


namespace NUMINAMATH_CALUDE_hyperbola_incenter_theorem_l3488_348816

/-- Hyperbola C: x²/4 - y²/5 = 1 -/
def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 / 5 = 1

/-- Point P is on the hyperbola in the first quadrant -/
structure PointOnHyperbola where
  x : ℝ
  y : ℝ
  on_hyperbola : hyperbola x y
  first_quadrant : x > 0 ∧ y > 0

/-- F₁ and F₂ are the left and right foci of the hyperbola -/
structure Foci where
  f₁ : ℝ × ℝ
  f₂ : ℝ × ℝ

/-- I is the incenter of triangle PF₁F₂ -/
def incenter (p : PointOnHyperbola) (f : Foci) : ℝ × ℝ := sorry

/-- |PF₁| = 2|PF₂| -/
def focal_distance_condition (p : PointOnHyperbola) (f : Foci) : Prop :=
  let (x₁, y₁) := f.f₁
  let (x₂, y₂) := f.f₂
  ((p.x - x₁)^2 + (p.y - y₁)^2) = 4 * ((p.x - x₂)^2 + (p.y - y₂)^2)

/-- Vector PI = x * Vector PF₁ + y * Vector PF₂ -/
def vector_condition (p : PointOnHyperbola) (f : Foci) (x y : ℝ) : Prop :=
  let i := incenter p f
  let (x₁, y₁) := f.f₁
  let (x₂, y₂) := f.f₂
  (i.1 - p.x, i.2 - p.y) = (x * (x₁ - p.x) + y * (x₂ - p.x), x * (y₁ - p.y) + y * (y₂ - p.y))

/-- Main theorem -/
theorem hyperbola_incenter_theorem (p : PointOnHyperbola) (f : Foci) (x y : ℝ) :
  focal_distance_condition p f →
  vector_condition p f x y →
  y - x = 2/9 := by sorry

end NUMINAMATH_CALUDE_hyperbola_incenter_theorem_l3488_348816


namespace NUMINAMATH_CALUDE_sphere_wedge_properties_l3488_348835

/-- Represents a sphere cut into eight congruent wedges -/
structure SphereWedge where
  circumference : ℝ
  num_wedges : ℕ

/-- Calculates the volume of one wedge of the sphere -/
def wedge_volume (s : SphereWedge) : ℝ := sorry

/-- Calculates the surface area of one wedge of the sphere -/
def wedge_surface_area (s : SphereWedge) : ℝ := sorry

theorem sphere_wedge_properties (s : SphereWedge) 
  (h1 : s.circumference = 16 * Real.pi)
  (h2 : s.num_wedges = 8) : 
  wedge_volume s = (256 / 3) * Real.pi ∧ 
  wedge_surface_area s = 32 * Real.pi := by sorry

end NUMINAMATH_CALUDE_sphere_wedge_properties_l3488_348835


namespace NUMINAMATH_CALUDE_reynald_soccer_balls_l3488_348866

/-- The number of soccer balls Reynald bought -/
def soccer_balls : ℕ := 20

/-- The total number of balls Reynald bought -/
def total_balls : ℕ := 145

/-- The number of volleyballs Reynald bought -/
def volleyballs : ℕ := 30

theorem reynald_soccer_balls :
  soccer_balls = 20 ∧
  soccer_balls + (soccer_balls + 5) + (2 * soccer_balls) + (soccer_balls + 10) + volleyballs = total_balls :=
by sorry

end NUMINAMATH_CALUDE_reynald_soccer_balls_l3488_348866


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l3488_348863

theorem min_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (∀ x y : ℝ, x > 0 → y > 0 → x + y = 1 → 1/x + 1/y ≥ 1/a + 1/b) →
  1/a + 1/b = 4 :=
by sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l3488_348863


namespace NUMINAMATH_CALUDE_min_value_of_expression_l3488_348892

theorem min_value_of_expression (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a + b) / (c + d) + (a + c) / (b + d) + (a + d) / (b + c) +
  (b + c) / (a + d) + (b + d) / (a + c) + (c + d) / (a + b) ≥ 6 ∧
  ((a + b) / (c + d) + (a + c) / (b + d) + (a + d) / (b + c) +
   (b + c) / (a + d) + (b + d) / (a + c) + (c + d) / (a + b) = 6 ↔ a = b ∧ b = c ∧ c = d) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l3488_348892


namespace NUMINAMATH_CALUDE_square_product_inequality_l3488_348858

theorem square_product_inequality (a b : ℝ) (h1 : a < b) (h2 : b < 0) : a^2 > a*b ∧ a*b > b^2 := by
  sorry

end NUMINAMATH_CALUDE_square_product_inequality_l3488_348858


namespace NUMINAMATH_CALUDE_sum_equals_seven_x_l3488_348810

theorem sum_equals_seven_x (x y z : ℝ) (h1 : y = 2 * x) (h2 : z = 2 * y) : 
  x + y + z = 7 * x := by
  sorry

end NUMINAMATH_CALUDE_sum_equals_seven_x_l3488_348810


namespace NUMINAMATH_CALUDE_new_tv_cost_l3488_348885

/-- The cost of a new TV given the dimensions and price of an old TV, and the price difference per square inch. -/
theorem new_tv_cost (old_width old_height old_cost new_width new_height price_diff : ℝ) :
  old_width = 24 →
  old_height = 16 →
  old_cost = 672 →
  new_width = 48 →
  new_height = 32 →
  price_diff = 1 →
  (old_cost / (old_width * old_height) - price_diff) * (new_width * new_height) = 1152 := by
  sorry

end NUMINAMATH_CALUDE_new_tv_cost_l3488_348885


namespace NUMINAMATH_CALUDE_count_squarish_numbers_l3488_348808

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k^2

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def first_two_digits (n : ℕ) : ℕ := n / 10000

def middle_two_digits (n : ℕ) : ℕ := (n / 100) % 100

def last_two_digits (n : ℕ) : ℕ := n % 100

def has_no_zero_digit (n : ℕ) : Prop :=
  ∀ d : ℕ, d < 6 → (n / 10^d) % 10 ≠ 0

def is_squarish (n : ℕ) : Prop :=
  100000 ≤ n ∧ n ≤ 999999 ∧
  is_perfect_square n ∧
  has_no_zero_digit n ∧
  is_perfect_square (first_two_digits n) ∧
  is_perfect_square (middle_two_digits n) ∧
  is_perfect_square (last_two_digits n)

theorem count_squarish_numbers :
  ∃! (s : Finset ℕ), (∀ n ∈ s, is_squarish n) ∧ s.card = 2 :=
sorry

end NUMINAMATH_CALUDE_count_squarish_numbers_l3488_348808


namespace NUMINAMATH_CALUDE_kneading_time_is_ten_l3488_348832

/-- Represents the time in minutes for bread-making process --/
structure BreadTime where
  total : ℕ
  rising : ℕ
  baking : ℕ

/-- Calculates the kneading time given the bread-making times --/
def kneadingTime (bt : BreadTime) : ℕ :=
  bt.total - (2 * bt.rising + bt.baking)

/-- Theorem stating that the kneading time is 10 minutes for the given conditions --/
theorem kneading_time_is_ten :
  let bt : BreadTime := { total := 280, rising := 120, baking := 30 }
  kneadingTime bt = 10 := by sorry

end NUMINAMATH_CALUDE_kneading_time_is_ten_l3488_348832


namespace NUMINAMATH_CALUDE_rotational_inertia_scaling_l3488_348827

/-- Represents a sphere with a given radius and rotational inertia about its center axis -/
structure Sphere where
  radius : ℝ
  rotationalInertia : ℝ

/-- Given two spheres with the same density, where the second sphere has twice the radius of the first,
    prove that the rotational inertia of the second sphere is 32 times that of the first sphere -/
theorem rotational_inertia_scaling (s1 s2 : Sphere) (h1 : s2.radius = 2 * s1.radius) :
  s2.rotationalInertia = 32 * s1.rotationalInertia := by
  sorry


end NUMINAMATH_CALUDE_rotational_inertia_scaling_l3488_348827


namespace NUMINAMATH_CALUDE_minjeong_marbles_l3488_348884

/-- Given that the total number of marbles is 43 and Yunjae has 5 more marbles than Minjeong,
    prove that Minjeong has 19 marbles. -/
theorem minjeong_marbles : 
  ∀ (y m : ℕ), y + m = 43 → y = m + 5 → m = 19 := by
  sorry

end NUMINAMATH_CALUDE_minjeong_marbles_l3488_348884


namespace NUMINAMATH_CALUDE_building_entrances_l3488_348807

/-- Represents a building with multiple entrances -/
structure Building where
  floors : ℕ
  apartments_per_floor : ℕ
  total_apartments : ℕ

/-- Calculates the number of entrances in a building -/
def number_of_entrances (b : Building) : ℕ :=
  b.total_apartments / (b.floors * b.apartments_per_floor)

/-- Theorem stating the number of entrances in the specific building -/
theorem building_entrances :
  let b : Building := {
    floors := 9,
    apartments_per_floor := 4,
    total_apartments := 180
  }
  number_of_entrances b = 5 := by
  sorry

end NUMINAMATH_CALUDE_building_entrances_l3488_348807


namespace NUMINAMATH_CALUDE_sum_34_27_base5_l3488_348894

def base10_to_base5 (n : ℕ) : List ℕ :=
  sorry

theorem sum_34_27_base5 :
  base10_to_base5 (34 + 27) = [2, 2, 1] :=
sorry

end NUMINAMATH_CALUDE_sum_34_27_base5_l3488_348894


namespace NUMINAMATH_CALUDE_second_hole_depth_calculation_l3488_348837

/-- Calculates the depth of a second hole given the conditions of two digging projects -/
def second_hole_depth (workers1 hours1 depth1 workers2 hours2 : ℕ) : ℚ :=
  let man_hours1 := workers1 * hours1
  let man_hours2 := workers2 * hours2
  (man_hours2 * depth1 : ℚ) / man_hours1

theorem second_hole_depth_calculation (workers1 hours1 depth1 extra_workers hours2 : ℕ) :
  second_hole_depth workers1 hours1 depth1 (workers1 + extra_workers) hours2 = 40 :=
by
  -- The proof goes here
  sorry

#eval second_hole_depth 45 8 30 80 6

end NUMINAMATH_CALUDE_second_hole_depth_calculation_l3488_348837


namespace NUMINAMATH_CALUDE_quadratic_factorization_l3488_348841

theorem quadratic_factorization (A B : ℤ) :
  (∀ y : ℝ, 10 * y^2 - 51 * y + 21 = (A * y - 7) * (B * y - 3)) →
  A * B + B = 12 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l3488_348841


namespace NUMINAMATH_CALUDE_two_distinct_real_roots_l3488_348811

-- Define the quadratic equation
def quadratic_equation (x m : ℝ) : Prop :=
  x^2 + x - 2 = m

-- Define the discriminant of the quadratic equation
def discriminant (m : ℝ) : ℝ :=
  4 * m + 9

-- Theorem statement
theorem two_distinct_real_roots (m : ℝ) (h : m > 0) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic_equation x₁ m ∧ quadratic_equation x₂ m :=
sorry

end NUMINAMATH_CALUDE_two_distinct_real_roots_l3488_348811


namespace NUMINAMATH_CALUDE_square_equation_solution_l3488_348859

theorem square_equation_solution : ∃ x : ℝ, (72 - x)^2 = x^2 ∧ x = 36 := by
  sorry

end NUMINAMATH_CALUDE_square_equation_solution_l3488_348859


namespace NUMINAMATH_CALUDE_largest_of_three_l3488_348824

theorem largest_of_three (x y z : ℝ) 
  (sum_eq : x + y + z = 3)
  (sum_prod_eq : x*y + y*z + z*x = -8)
  (prod_eq : x*y*z = -18) :
  max x (max y z) = Real.sqrt 5 :=
sorry

end NUMINAMATH_CALUDE_largest_of_three_l3488_348824


namespace NUMINAMATH_CALUDE_count_two_repeating_digits_l3488_348804

/-- A four-digit number is a natural number between 1000 and 9999 inclusive. -/
def FourDigitNumber (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

/-- A function that counts the occurrences of each digit in a four-digit number. -/
def DigitCount (n : ℕ) : ℕ → ℕ := sorry

/-- A four-digit number has exactly two repeating digits if exactly one digit appears twice
    and the other two digits appear once each. -/
def HasExactlyTwoRepeatingDigits (n : ℕ) : Prop :=
  FourDigitNumber n ∧ ∃! d : ℕ, DigitCount n d = 2

/-- The count of four-digit numbers with exactly two repeating digits. -/
def CountTwoRepeatingDigits : ℕ := sorry

/-- The main theorem stating that the count of four-digit numbers with exactly two repeating digits is 2736. -/
theorem count_two_repeating_digits :
  CountTwoRepeatingDigits = 2736 := by sorry

end NUMINAMATH_CALUDE_count_two_repeating_digits_l3488_348804


namespace NUMINAMATH_CALUDE_friends_picnic_only_l3488_348861

/-- Given information about friends meeting for different activities, 
    prove that the number of friends meeting for picnic only is 20. -/
theorem friends_picnic_only (total : ℕ) (movie : ℕ) (games : ℕ) 
  (movie_picnic : ℕ) (movie_games : ℕ) (picnic_games : ℕ) (all_three : ℕ) :
  total = 31 ∧ 
  movie = 10 ∧ 
  games = 5 ∧ 
  movie_picnic = 4 ∧ 
  movie_games = 2 ∧ 
  picnic_games = 0 ∧ 
  all_three = 2 → 
  ∃ (movie_only picnic_only games_only : ℕ),
    total = movie_only + picnic_only + games_only + movie_picnic + movie_games + picnic_games + all_three ∧
    movie = movie_only + movie_picnic + movie_games + all_three ∧
    games = games_only + movie_games + all_three ∧
    picnic_only = 20 := by
  sorry

end NUMINAMATH_CALUDE_friends_picnic_only_l3488_348861


namespace NUMINAMATH_CALUDE_fraction_proof_l3488_348822

theorem fraction_proof (x : ℝ) (f : ℝ) : 
  x = 300 → 0.70 * x = f * x + 110 → f = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_proof_l3488_348822


namespace NUMINAMATH_CALUDE_complex_equation_sum_l3488_348878

theorem complex_equation_sum (a b : ℝ) : 
  (a / (1 - Complex.I)) + (b / (1 - 2 * Complex.I)) = (1 + 3 * Complex.I) / 4 → 
  a + b = 2 := by
sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l3488_348878


namespace NUMINAMATH_CALUDE_original_fraction_proof_l3488_348896

theorem original_fraction_proof (x y : ℚ) : 
  (1.15 * x) / (0.92 * y) = 15 / 16 → x / y = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_original_fraction_proof_l3488_348896


namespace NUMINAMATH_CALUDE_sqrt3_expressions_l3488_348877

theorem sqrt3_expressions (x y : ℝ) 
  (hx : x = Real.sqrt 3 + 1) 
  (hy : y = Real.sqrt 3 - 1) : 
  (x^2 + 2*x*y + y^2 = 12) ∧ (x^2 - y^2 = 4 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_sqrt3_expressions_l3488_348877


namespace NUMINAMATH_CALUDE_min_max_bound_l3488_348876

theorem min_max_bound (x₁ x₂ x₃ : ℝ) (h_nonneg : x₁ ≥ 0 ∧ x₂ ≥ 0 ∧ x₃ ≥ 0) 
  (h_sum : x₁ + x₂ + x₃ = 1) : 
  1 ≤ (x₁ + 3*x₂ + 5*x₃)*(x₁ + x₂/3 + x₃/5) ∧ 
  (x₁ + 3*x₂ + 5*x₃)*(x₁ + x₂/3 + x₃/5) ≤ 9/5 := by
  sorry

#check min_max_bound

end NUMINAMATH_CALUDE_min_max_bound_l3488_348876


namespace NUMINAMATH_CALUDE_unique_number_property_l3488_348844

theorem unique_number_property : ∃! x : ℝ, x / 3 = x - 5 := by sorry

end NUMINAMATH_CALUDE_unique_number_property_l3488_348844


namespace NUMINAMATH_CALUDE_find_a_l3488_348898

theorem find_a : ∃ a : ℕ, 
  (∀ K : ℤ, K ≠ 27 → ∃ m : ℤ, a - K^3 = m * (27 - K)) → 
  a = 3^9 := by
sorry

end NUMINAMATH_CALUDE_find_a_l3488_348898


namespace NUMINAMATH_CALUDE_equation_solution_l3488_348836

theorem equation_solution :
  let f (x : ℝ) := 4 * (3 * x)^2 + (3 * x) + 5 - (3 * (9 * x^2 + 3 * x + 3))
  ∀ x : ℝ, f x = 0 ↔ x = (1 + Real.sqrt 5) / 3 ∨ x = (1 - Real.sqrt 5) / 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3488_348836


namespace NUMINAMATH_CALUDE_tenth_number_with_digit_sum_12_l3488_348865

/-- A function that returns the sum of digits of a positive integer -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- A function that returns the nth positive integer whose digits add up to 12 -/
def nth_number_with_digit_sum_12 (n : ℕ) : ℕ := sorry

/-- The theorem stating that the 10th number with digit sum 12 is 147 -/
theorem tenth_number_with_digit_sum_12 : nth_number_with_digit_sum_12 10 = 147 := by
  sorry

end NUMINAMATH_CALUDE_tenth_number_with_digit_sum_12_l3488_348865


namespace NUMINAMATH_CALUDE_johnsons_share_l3488_348874

/-- 
Given a profit-sharing ratio and Mike's total share, calculate Johnson's share.
-/
theorem johnsons_share 
  (mike_ratio : ℕ) 
  (johnson_ratio : ℕ) 
  (mike_total_share : ℕ) : 
  mike_ratio = 2 → 
  johnson_ratio = 5 → 
  mike_total_share = 1000 → 
  (mike_total_share * johnson_ratio) / mike_ratio = 2500 := by
  sorry

#check johnsons_share

end NUMINAMATH_CALUDE_johnsons_share_l3488_348874


namespace NUMINAMATH_CALUDE_student_turtle_difference_is_85_l3488_348845

/-- The number of fourth-grade classrooms -/
def num_classrooms : ℕ := 5

/-- The number of students in each fourth-grade classroom -/
def students_per_classroom : ℕ := 20

/-- The number of pet turtles in each fourth-grade classroom -/
def turtles_per_classroom : ℕ := 3

/-- The difference between the total number of students and the total number of turtles -/
def student_turtle_difference : ℕ :=
  num_classrooms * students_per_classroom - num_classrooms * turtles_per_classroom

theorem student_turtle_difference_is_85 : student_turtle_difference = 85 := by
  sorry

end NUMINAMATH_CALUDE_student_turtle_difference_is_85_l3488_348845


namespace NUMINAMATH_CALUDE_equilateral_triangle_on_parallel_lines_l3488_348886

/-- A line in a plane --/
structure Line :=
  (slope : ℝ)
  (intercept : ℝ)

/-- A point in a plane --/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Three parallel lines --/
def parallel_lines : Vector Line 3 :=
  sorry

/-- Definition of an equilateral triangle --/
def is_equilateral_triangle (a b c : Point) : Prop :=
  sorry

/-- Theorem: There exists an equilateral triangle with vertices on three parallel lines --/
theorem equilateral_triangle_on_parallel_lines :
  ∃ (a b c : Point),
    (∀ i : Fin 3, ∃ j : Fin 3, a.y = parallel_lines[i].slope * a.x + parallel_lines[i].intercept ∨
                               b.y = parallel_lines[i].slope * b.x + parallel_lines[i].intercept ∨
                               c.y = parallel_lines[i].slope * c.x + parallel_lines[i].intercept) ∧
    is_equilateral_triangle a b c :=
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_on_parallel_lines_l3488_348886


namespace NUMINAMATH_CALUDE_mount_everest_temperature_difference_l3488_348862

/-- Temperature difference between two points -/
def temperature_difference (t1 : ℝ) (t2 : ℝ) : ℝ := t1 - t2

/-- Temperature at the foot of Mount Everest in °C -/
def foot_temperature : ℝ := 24

/-- Temperature at the summit of Mount Everest in °C -/
def summit_temperature : ℝ := -50

/-- Theorem stating the temperature difference between the foot and summit of Mount Everest -/
theorem mount_everest_temperature_difference :
  temperature_difference foot_temperature summit_temperature = 74 := by
  sorry

end NUMINAMATH_CALUDE_mount_everest_temperature_difference_l3488_348862


namespace NUMINAMATH_CALUDE_find_a_l3488_348851

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x^2 - 2*x - 3 < 0}
def B : Set ℝ := Set.Ioo (-3) 2

-- Define the property that A ∩ B is the solution set of x^2 + ax + b < 0
def is_solution_set (a b : ℝ) : Prop :=
  ∀ x, x ∈ A ∩ B ↔ x^2 + a*x + b < 0

-- State the theorem
theorem find_a :
  ∃ b, is_solution_set (-1) b :=
sorry

end NUMINAMATH_CALUDE_find_a_l3488_348851


namespace NUMINAMATH_CALUDE_bowling_ball_weight_l3488_348820

theorem bowling_ball_weight (b k : ℝ) 
  (h1 : 5 * b = 3 * k) 
  (h2 : 4 * k = 120) : 
  b = 18 := by
sorry

end NUMINAMATH_CALUDE_bowling_ball_weight_l3488_348820


namespace NUMINAMATH_CALUDE_trajectory_of_P_l3488_348825

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 + 2*x = 15

-- Define point N
def point_N : ℝ × ℝ := (1, 0)

-- Define the property of point M being on circle C
def point_M_on_C (M : ℝ × ℝ) : Prop := circle_C M.1 M.2

-- Define point P as the intersection of perpendicular bisector of MN and CM
def point_P (M : ℝ × ℝ) (P : ℝ × ℝ) : Prop :=
  point_M_on_C M ∧ 
  -- Additional conditions for P would be defined here, but we omit the detailed geometric conditions
  True

-- State the theorem
theorem trajectory_of_P :
  ∀ P : ℝ × ℝ, (∃ M : ℝ × ℝ, point_P M P) →
  (P.1^2 / 4 + P.2^2 / 3 = 1) :=
sorry

end NUMINAMATH_CALUDE_trajectory_of_P_l3488_348825


namespace NUMINAMATH_CALUDE_race_distance_l3488_348802

/-- The race distance in meters -/
def d : ℝ := 75

/-- The speed of runner X -/
def x : ℝ := sorry

/-- The speed of runner Y -/
def y : ℝ := sorry

/-- The speed of runner Z -/
def z : ℝ := sorry

/-- Theorem stating that d is the correct race distance -/
theorem race_distance : 
  (d / x = (d - 25) / y) ∧ 
  (d / y = (d - 15) / z) ∧ 
  (d / x = (d - 35) / z) → 
  d = 75 := by sorry

end NUMINAMATH_CALUDE_race_distance_l3488_348802


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3488_348815

def A : Set ℝ := {x | x^2 - 4*x > 0}
def B : Set ℝ := {x | |x - 1| ≤ 2}

theorem intersection_of_A_and_B : A ∩ B = {x | -1 ≤ x ∧ x < 0} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3488_348815


namespace NUMINAMATH_CALUDE_frank_five_dollar_bills_l3488_348831

def peanut_cost_per_pound : ℕ := 3
def days_in_week : ℕ := 7
def pounds_per_day : ℕ := 3
def one_dollar_bills : ℕ := 7
def ten_dollar_bills : ℕ := 2
def twenty_dollar_bills : ℕ := 1
def change_amount : ℕ := 4

def total_without_fives : ℕ := one_dollar_bills + 10 * ten_dollar_bills + 20 * twenty_dollar_bills

def total_pounds_needed : ℕ := days_in_week * pounds_per_day

theorem frank_five_dollar_bills :
  ∃ (five_dollar_bills : ℕ),
    total_without_fives + 5 * five_dollar_bills - change_amount = peanut_cost_per_pound * total_pounds_needed ∧
    five_dollar_bills = 4 := by
  sorry

end NUMINAMATH_CALUDE_frank_five_dollar_bills_l3488_348831


namespace NUMINAMATH_CALUDE_grandparents_uncle_difference_l3488_348838

/-- Represents the money Gwen received from each family member -/
structure MoneyReceived where
  dad : ℕ
  mom : ℕ
  uncle : ℕ
  aunt : ℕ
  cousin : ℕ
  grandparents : ℕ

/-- The amount of money Gwen received for her birthday -/
def gwens_birthday_money : MoneyReceived :=
  { dad := 5
  , mom := 10
  , uncle := 8
  , aunt := 3
  , cousin := 6
  , grandparents := 15
  }

/-- Theorem stating the difference between money received from grandparents and uncle -/
theorem grandparents_uncle_difference :
  gwens_birthday_money.grandparents - gwens_birthday_money.uncle = 7 := by
  sorry

end NUMINAMATH_CALUDE_grandparents_uncle_difference_l3488_348838


namespace NUMINAMATH_CALUDE_power_of_power_l3488_348856

theorem power_of_power (a : ℝ) : (a^5)^2 = a^10 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l3488_348856


namespace NUMINAMATH_CALUDE_equal_strawberry_division_l3488_348875

def strawberry_division (brother_baskets : ℕ) (strawberries_per_basket : ℕ) : ℕ :=
  let brother_strawberries := brother_baskets * strawberries_per_basket
  let kimberly_strawberries := 8 * brother_strawberries
  let parents_strawberries := kimberly_strawberries - 93
  let total_strawberries := kimberly_strawberries + brother_strawberries + parents_strawberries
  total_strawberries / 4

theorem equal_strawberry_division :
  strawberry_division 3 15 = 168 := by
  sorry

end NUMINAMATH_CALUDE_equal_strawberry_division_l3488_348875


namespace NUMINAMATH_CALUDE_contractor_daily_wage_l3488_348855

/-- Calculates the daily wage for a contractor given the contract terms and outcomes. -/
def calculate_daily_wage (total_days : ℕ) (fine_per_day : ℚ) (total_received : ℚ) (absent_days : ℕ) : ℚ :=
  let worked_days := total_days - absent_days
  let total_fine := fine_per_day * absent_days
  (total_received + total_fine) / worked_days

/-- Proves that the daily wage is 25 given the specific contract terms. -/
theorem contractor_daily_wage :
  calculate_daily_wage 30 (7.5 : ℚ) 360 12 = 25 := by
  sorry

end NUMINAMATH_CALUDE_contractor_daily_wage_l3488_348855


namespace NUMINAMATH_CALUDE_probability_second_quality_l3488_348846

theorem probability_second_quality (p : ℝ) : 
  (1 - p^2 = 0.91) → p = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_probability_second_quality_l3488_348846


namespace NUMINAMATH_CALUDE_weeks_in_month_is_four_l3488_348899

/-- The number of weeks in a month -/
def weeks_in_month : ℕ := sorry

/-- The standard work hours per week -/
def standard_hours_per_week : ℕ := 20

/-- The number of months worked -/
def months_worked : ℕ := 2

/-- The additional hours worked due to covering a shift -/
def additional_hours : ℕ := 20

/-- The total hours worked over the period -/
def total_hours_worked : ℕ := 180

theorem weeks_in_month_is_four :
  weeks_in_month = 4 :=
by sorry

end NUMINAMATH_CALUDE_weeks_in_month_is_four_l3488_348899


namespace NUMINAMATH_CALUDE_probability_of_specific_pairing_l3488_348828

theorem probability_of_specific_pairing (n : ℕ) (h : n = 25) :
  let total_students := n
  let available_partners := n - 1
  (1 : ℚ) / available_partners = 1 / 24 :=
by
  sorry

end NUMINAMATH_CALUDE_probability_of_specific_pairing_l3488_348828


namespace NUMINAMATH_CALUDE_cell_phone_customers_l3488_348849

theorem cell_phone_customers (us_customers other_customers : ℕ) 
  (h1 : us_customers = 723)
  (h2 : other_customers = 6699) :
  us_customers + other_customers = 7422 := by
sorry

end NUMINAMATH_CALUDE_cell_phone_customers_l3488_348849


namespace NUMINAMATH_CALUDE_system_of_equations_solution_l3488_348834

theorem system_of_equations_solution :
  ∃! (a b : ℝ), 3*a + 2*b = -26 ∧ 2*a - b = -22 :=
by
  sorry

end NUMINAMATH_CALUDE_system_of_equations_solution_l3488_348834


namespace NUMINAMATH_CALUDE_parallelogram_circle_theorem_l3488_348888

/-- Represents a parallelogram KLMN with a circle tangent to NK and NM, passing through L, 
    and intersecting KL at C and ML at D. -/
structure ParallelogramWithCircle where
  -- The length of side KL
  kl : ℝ
  -- The ratio KC : LC
  kc_lc_ratio : ℝ × ℝ
  -- The ratio LD : MD
  ld_md_ratio : ℝ × ℝ

/-- Theorem stating that under the given conditions, KN = 10 -/
theorem parallelogram_circle_theorem (p : ParallelogramWithCircle) 
  (h1 : p.kl = 8)
  (h2 : p.kc_lc_ratio = (4, 5))
  (h3 : p.ld_md_ratio = (8, 1)) :
  ∃ (kn : ℝ), kn = 10 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_circle_theorem_l3488_348888


namespace NUMINAMATH_CALUDE_angle_theta_value_l3488_348805

theorem angle_theta_value (θ : Real) (A B : Set Real) : 
  A = {1, Real.cos θ} →
  B = {0, 1/2, 1} →
  A ⊆ B →
  0 < θ →
  θ < π/2 →
  θ = π/3 := by
sorry

end NUMINAMATH_CALUDE_angle_theta_value_l3488_348805


namespace NUMINAMATH_CALUDE_regular_polygon_exterior_angle_l3488_348840

/-- A regular polygon with interior angle sum of 540° has an exterior angle of 72° --/
theorem regular_polygon_exterior_angle (n : ℕ) : 
  (n - 2) * 180 = 540 → 360 / n = 72 := by sorry

end NUMINAMATH_CALUDE_regular_polygon_exterior_angle_l3488_348840


namespace NUMINAMATH_CALUDE_duck_buying_problem_l3488_348852

theorem duck_buying_problem (adelaide ephraim kolton : ℕ) : 
  adelaide = 30 →
  adelaide = 2 * ephraim →
  kolton = ephraim + 45 →
  (adelaide + ephraim + kolton) % 9 = 0 →
  ephraim ≥ 1 →
  kolton ≥ 1 →
  (adelaide + ephraim + kolton) / 3 = 36 :=
by sorry

end NUMINAMATH_CALUDE_duck_buying_problem_l3488_348852


namespace NUMINAMATH_CALUDE_equidistant_points_of_f_l3488_348817

/-- A point (x, y) is equidistant if |x| = |y| -/
def is_equidistant (x y : ℝ) : Prop := abs x = abs y

/-- The function f(x) = x^2 + 2x -/
def f (x : ℝ) : ℝ := x^2 + 2*x

/-- Theorem: The points (0,0), (-1,-1), and (-3,3) are equidistant points of f -/
theorem equidistant_points_of_f :
  is_equidistant 0 (f 0) ∧
  is_equidistant (-1) (f (-1)) ∧
  is_equidistant (-3) (f (-3)) :=
sorry

end NUMINAMATH_CALUDE_equidistant_points_of_f_l3488_348817


namespace NUMINAMATH_CALUDE_car_service_month_l3488_348850

/-- Represents the months of the year -/
inductive Month : Type
| jan | feb | mar | apr | may | jun | jul | aug | sep | oct | nov | dec

/-- Convert a number to a month -/
def num_to_month (n : Nat) : Month :=
  match n % 12 with
  | 1 => Month.jan
  | 2 => Month.feb
  | 3 => Month.mar
  | 4 => Month.apr
  | 5 => Month.may
  | 6 => Month.jun
  | 7 => Month.jul
  | 8 => Month.aug
  | 9 => Month.sep
  | 10 => Month.oct
  | 11 => Month.nov
  | _ => Month.dec

theorem car_service_month (service_interval : Nat) (first_service : Month) (n : Nat) :
  service_interval = 7 →
  first_service = Month.jan →
  n = 30 →
  num_to_month ((n - 1) * service_interval % 12 + 1) = Month.dec :=
by
  sorry

end NUMINAMATH_CALUDE_car_service_month_l3488_348850


namespace NUMINAMATH_CALUDE_irrational_x_with_rational_expressions_l3488_348853

theorem irrational_x_with_rational_expressions (x : ℝ) :
  Irrational x →
  ∃ q₁ q₂ : ℚ, (x^3 - 6*x : ℝ) = (q₁ : ℝ) ∧ (x^4 - 8*x^2 : ℝ) = (q₂ : ℝ) →
  x = Real.sqrt 6 ∨ x = -Real.sqrt 6 ∨
  x = 1 + Real.sqrt 3 ∨ x = -(1 + Real.sqrt 3) ∨
  x = 1 - Real.sqrt 3 ∨ x = -(1 - Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_irrational_x_with_rational_expressions_l3488_348853


namespace NUMINAMATH_CALUDE_time_addition_theorem_l3488_348812

/-- Represents a date and time --/
structure DateTime where
  year : Nat
  month : Nat
  day : Nat
  hour : Nat
  minute : Nat

/-- Adds minutes to a DateTime --/
def addMinutes (dt : DateTime) (minutes : Nat) : DateTime :=
  sorry

/-- Checks if two DateTimes are equal --/
def dateTimeEqual (dt1 dt2 : DateTime) : Prop :=
  dt1.year = dt2.year ∧
  dt1.month = dt2.month ∧
  dt1.day = dt2.day ∧
  dt1.hour = dt2.hour ∧
  dt1.minute = dt2.minute

theorem time_addition_theorem :
  let start := DateTime.mk 2023 7 4 12 0
  let end_time := DateTime.mk 2023 7 6 21 36
  dateTimeEqual (addMinutes start 3456) end_time :=
by sorry

end NUMINAMATH_CALUDE_time_addition_theorem_l3488_348812
