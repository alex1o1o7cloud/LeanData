import Mathlib

namespace NUMINAMATH_CALUDE_tan_theta_two_implies_expression_equals_negative_two_l1305_130564

theorem tan_theta_two_implies_expression_equals_negative_two (θ : Real) 
  (h : Real.tan θ = 2) : 
  (Real.sin (π / 2 + θ) - Real.cos (π - θ)) / 
  (Real.sin (π / 2 - θ) - Real.sin (π - θ)) = -2 := by
  sorry

end NUMINAMATH_CALUDE_tan_theta_two_implies_expression_equals_negative_two_l1305_130564


namespace NUMINAMATH_CALUDE_unique_intersection_l1305_130573

/-- 
Given two functions f(x) = ax² + 2x + 3 and g(x) = -2x - 3, 
this theorem states that these functions intersect at exactly one point 
if and only if a = 2/3.
-/
theorem unique_intersection (a : ℝ) : 
  (∃! x : ℝ, a * x^2 + 2 * x + 3 = -2 * x - 3) ↔ a = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_unique_intersection_l1305_130573


namespace NUMINAMATH_CALUDE_decimal_to_base5_250_l1305_130550

/-- Converts a base-10 number to its base-5 representation -/
def toBase5 (n : ℕ) : List ℕ := sorry

theorem decimal_to_base5_250 :
  toBase5 250 = [2, 0, 0, 0] := by sorry

end NUMINAMATH_CALUDE_decimal_to_base5_250_l1305_130550


namespace NUMINAMATH_CALUDE_monomial_2015_coeff_l1305_130583

/-- The coefficient of the nth monomial in the sequence -/
def monomial_coeff (n : ℕ) : ℤ := (-1)^n * (2*n - 1)

/-- The theorem stating that the 2015th monomial coefficient is -4029 -/
theorem monomial_2015_coeff : monomial_coeff 2015 = -4029 := by
  sorry

end NUMINAMATH_CALUDE_monomial_2015_coeff_l1305_130583


namespace NUMINAMATH_CALUDE_laurent_series_expansion_l1305_130501

open Complex

/-- The Laurent series expansion of f(z) = (z+2)/(z^2+4z+3) in the ring 2 < |z+1| < +∞ --/
theorem laurent_series_expansion (z : ℂ) (h : 2 < abs (z + 1)) :
  (z + 2) / (z^2 + 4*z + 3) = ∑' k, ((-2)^k + 1) / (z + 1)^(k + 1) := by sorry

end NUMINAMATH_CALUDE_laurent_series_expansion_l1305_130501


namespace NUMINAMATH_CALUDE_annual_growth_rate_l1305_130569

/-- Given a monthly average growth rate, calculate the annual average growth rate -/
theorem annual_growth_rate (P : ℝ) :
  let monthly_rate := P
  let annual_rate := (1 + P)^12 - 1
  annual_rate = ((1 + monthly_rate)^12 - 1) :=
by sorry

end NUMINAMATH_CALUDE_annual_growth_rate_l1305_130569


namespace NUMINAMATH_CALUDE_rikshaw_charge_theorem_l1305_130571

/-- Represents the rikshaw charging system in Mumbai -/
structure RikshawCharge where
  base_charge : ℝ  -- Charge for the first 1 km
  rate_1_5 : ℝ     -- Rate per km for 1-5 km
  rate_5_10 : ℝ    -- Rate per 1/3 km for 5-10 km
  rate_10_plus : ℝ -- Rate per 1/3 km beyond 10 km
  wait_rate : ℝ    -- Waiting charge per hour after first 10 minutes

/-- Calculates the total charge for a rikshaw ride -/
def calculate_charge (c : RikshawCharge) (distance : ℝ) (wait_time : ℝ) : ℝ :=
  sorry

/-- The theorem stating the total charge for the given ride -/
theorem rikshaw_charge_theorem (c : RikshawCharge) 
  (h1 : c.base_charge = 18.5)
  (h2 : c.rate_1_5 = 3)
  (h3 : c.rate_5_10 = 2.5)
  (h4 : c.rate_10_plus = 4)
  (h5 : c.wait_rate = 20) :
  calculate_charge c 16 1.5 = 170 :=
sorry

end NUMINAMATH_CALUDE_rikshaw_charge_theorem_l1305_130571


namespace NUMINAMATH_CALUDE_sophies_perceived_height_l1305_130510

/-- Calculates the perceived height in centimeters when doubled in a mirror reflection. -/
def perceivedHeightCm (actualHeightInches : ℝ) (conversionRate : ℝ) : ℝ :=
  2 * actualHeightInches * conversionRate

/-- Theorem stating that Sophie's perceived height in the mirror is 250.0 cm. -/
theorem sophies_perceived_height :
  let actualHeight : ℝ := 50
  let conversionRate : ℝ := 2.50
  perceivedHeightCm actualHeight conversionRate = 250.0 := by
  sorry

end NUMINAMATH_CALUDE_sophies_perceived_height_l1305_130510


namespace NUMINAMATH_CALUDE_total_pieces_is_4000_l1305_130508

/-- The number of pieces in the first puzzle -/
def first_puzzle_pieces : ℕ := 1000

/-- The number of pieces in the second and third puzzles -/
def other_puzzle_pieces : ℕ := first_puzzle_pieces + first_puzzle_pieces / 2

/-- The total number of pieces in all three puzzles -/
def total_pieces : ℕ := first_puzzle_pieces + 2 * other_puzzle_pieces

/-- Theorem stating that the total number of pieces in all three puzzles is 4000 -/
theorem total_pieces_is_4000 : total_pieces = 4000 := by
  sorry

end NUMINAMATH_CALUDE_total_pieces_is_4000_l1305_130508


namespace NUMINAMATH_CALUDE_multiply_powers_with_coefficient_l1305_130581

theorem multiply_powers_with_coefficient (a : ℝ) : 2 * (a^2 * a^4) = 2 * a^6 := by
  sorry

end NUMINAMATH_CALUDE_multiply_powers_with_coefficient_l1305_130581


namespace NUMINAMATH_CALUDE_ten_person_tournament_matches_l1305_130506

/-- Calculate the number of matches in a round-robin tournament. -/
def roundRobinMatches (n : ℕ) : ℕ :=
  n * (n - 1) / 2

/-- Theorem: A 10-person round-robin chess tournament has 45 matches. -/
theorem ten_person_tournament_matches :
  roundRobinMatches 10 = 45 := by
  sorry

#eval roundRobinMatches 10  -- Should output 45

end NUMINAMATH_CALUDE_ten_person_tournament_matches_l1305_130506


namespace NUMINAMATH_CALUDE_parallel_vectors_y_value_l1305_130544

/-- Given two parallel vectors a and b, prove that y = 7 -/
theorem parallel_vectors_y_value (a b : ℝ × ℝ) (y : ℝ) 
  (h1 : a = (2, 3)) 
  (h2 : b = (4, -1 + y)) 
  (h3 : ∃ (k : ℝ), k ≠ 0 ∧ a = k • b) : 
  y = 7 := by sorry

end NUMINAMATH_CALUDE_parallel_vectors_y_value_l1305_130544


namespace NUMINAMATH_CALUDE_smallest_number_of_eggs_l1305_130588

theorem smallest_number_of_eggs : ∀ n : ℕ,
  (n > 150) →
  (∃ k : ℕ, n = 15 * k - 6) →
  (∀ m : ℕ, (m > 150 ∧ ∃ j : ℕ, m = 15 * j - 6) → m ≥ n) →
  n = 159 := by
sorry

end NUMINAMATH_CALUDE_smallest_number_of_eggs_l1305_130588


namespace NUMINAMATH_CALUDE_product_increase_thirteen_times_l1305_130507

theorem product_increase_thirteen_times :
  ∃ (a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℕ),
    ((a₁ - 3) * (a₂ - 3) * (a₃ - 3) * (a₄ - 3) * (a₅ - 3) * (a₆ - 3) * (a₇ - 3)) / 
    (a₁ * a₂ * a₃ * a₄ * a₅ * a₆ * a₇ : ℚ) = 13 :=
by sorry

end NUMINAMATH_CALUDE_product_increase_thirteen_times_l1305_130507


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l1305_130509

theorem triangle_angle_measure (a b c : ℝ) (A B C : ℝ) :
  (a * Real.sin A + b * Real.sin B - c * Real.sin C) / (a * Real.sin B) = 2 * Real.sqrt 3 * Real.sin C →
  C = π / 6 :=
sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l1305_130509


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_find_m_for_intersection_l1305_130563

-- Define the sets A and B
def A : Set ℝ := {x | (x - 5) / (x + 1) ≤ 0}
def B (m : ℝ) : Set ℝ := {x | x^2 - 2*x - m < 0}

-- Theorem for part (1)
theorem intersection_A_complement_B : 
  A ∩ (Set.univ \ B 3) = Set.Icc 3 5 := by sorry

-- Theorem for part (2)
theorem find_m_for_intersection : 
  ∃ m : ℝ, A ∩ B m = Set.Ioo (-1) 4 → m = 8 := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_find_m_for_intersection_l1305_130563


namespace NUMINAMATH_CALUDE_digit_sequence_sum_value_l1305_130556

def is_increasing (n : ℕ) : Prop := sorry

def is_decreasing (n : ℕ) : Prop := sorry

def digit_sequence_sum : ℕ := sorry

theorem digit_sequence_sum_value : 
  digit_sequence_sum = (80 * 11^10 - 35 * 2^10) / 81 - 45 := by sorry

end NUMINAMATH_CALUDE_digit_sequence_sum_value_l1305_130556


namespace NUMINAMATH_CALUDE_problem_solution_l1305_130529

theorem problem_solution (a b : ℝ) : 
  (|a| = 6 ∧ |b| = 2) →
  (((a * b > 0) → |a + b| = 8) ∧
   ((|a + b| = a + b) → (a - b = 4 ∨ a - b = 8))) := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l1305_130529


namespace NUMINAMATH_CALUDE_two_digit_square_last_two_digits_l1305_130532

theorem two_digit_square_last_two_digits (x : ℕ) : 
  10 ≤ x ∧ x < 100 ∧ x^2 % 100 = x % 100 ↔ x = 25 ∨ x = 76 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_square_last_two_digits_l1305_130532


namespace NUMINAMATH_CALUDE_max_rooms_less_than_55_l1305_130575

/-- Represents the number of rooms with different combinations of bouquets -/
structure RoomCounts where
  chrysOnly : ℕ
  carnOnly : ℕ
  roseOnly : ℕ
  chrysCarn : ℕ
  chrysRose : ℕ
  carnRose : ℕ
  allThree : ℕ

/-- The conditions of the mansion and its bouquets -/
def MansionConditions (r : RoomCounts) : Prop :=
  r.chrysCarn = 2 ∧
  r.chrysRose = 3 ∧
  r.carnRose = 4 ∧
  r.chrysOnly + r.chrysCarn + r.chrysRose + r.allThree = 10 ∧
  r.carnOnly + r.chrysCarn + r.carnRose + r.allThree = 20 ∧
  r.roseOnly + r.chrysRose + r.carnRose + r.allThree = 30

/-- The total number of rooms in the mansion -/
def totalRooms (r : RoomCounts) : ℕ :=
  r.chrysOnly + r.carnOnly + r.roseOnly + r.chrysCarn + r.chrysRose + r.carnRose + r.allThree

/-- Theorem stating that the maximum number of rooms is less than 55 -/
theorem max_rooms_less_than_55 (r : RoomCounts) (h : MansionConditions r) : 
  totalRooms r < 55 := by
  sorry


end NUMINAMATH_CALUDE_max_rooms_less_than_55_l1305_130575


namespace NUMINAMATH_CALUDE_digital_root_of_2_pow_100_l1305_130579

/-- The digital root of a natural number is the single digit obtained by repeatedly summing its digits. -/
def digital_root (n : ℕ) : ℕ := sorry

/-- Theorem: The digital root of 2^100 is 7. -/
theorem digital_root_of_2_pow_100 : digital_root (2^100) = 7 := by sorry

end NUMINAMATH_CALUDE_digital_root_of_2_pow_100_l1305_130579


namespace NUMINAMATH_CALUDE_manager_salary_calculation_l1305_130511

/-- Calculates the manager's salary given the number of employees, their average salary,
    and the increase in average salary when the manager is included. -/
def manager_salary (num_employees : ℕ) (avg_salary : ℚ) (avg_increase : ℚ) : ℚ :=
  (avg_salary + avg_increase) * (num_employees + 1) - avg_salary * num_employees

/-- Theorem stating that given 25 employees with an average salary of 2500,
    if adding a manager's salary increases the average by 400,
    then the manager's salary is 12900. -/
theorem manager_salary_calculation :
  manager_salary 25 2500 400 = 12900 := by
  sorry

end NUMINAMATH_CALUDE_manager_salary_calculation_l1305_130511


namespace NUMINAMATH_CALUDE_square_root_calculation_l1305_130549

theorem square_root_calculation : 2 * (Real.sqrt 50625)^2 = 101250 := by
  sorry

end NUMINAMATH_CALUDE_square_root_calculation_l1305_130549


namespace NUMINAMATH_CALUDE_teacher_friends_count_l1305_130559

theorem teacher_friends_count (total_students : ℕ) 
  (both_friends : ℕ) (neither_friends : ℕ) (friend_difference : ℕ) :
  total_students = 50 →
  both_friends = 30 →
  neither_friends = 1 →
  friend_difference = 7 →
  ∃ (zhang_friends : ℕ),
    zhang_friends = 43 ∧
    zhang_friends + (zhang_friends - friend_difference) - both_friends + neither_friends = total_students :=
by sorry

end NUMINAMATH_CALUDE_teacher_friends_count_l1305_130559


namespace NUMINAMATH_CALUDE_sequence_problem_l1305_130531

theorem sequence_problem (x : ℕ → ℝ) 
  (h_distinct : ∀ n m, n ≥ 2 → m ≥ 2 → n ≠ m → x n ≠ x m)
  (h_relation : ∀ n, n ≥ 2 → x n = (x (n-1) + 398 * x n + x (n+1)) / 400) :
  Real.sqrt ((x 2023 - x 2) / 2021 * (2022 / (x 2023 - x 1))) + 2021 = 2022 := by
  sorry

end NUMINAMATH_CALUDE_sequence_problem_l1305_130531


namespace NUMINAMATH_CALUDE_complement_union_theorem_l1305_130540

def U : Set ℕ := {x | 0 ≤ x ∧ x < 6}
def A : Set ℕ := {1, 3, 5}
def B : Set ℕ := {x ∈ U | x^2 + 4 = 5*x}

theorem complement_union_theorem : 
  (U \ A) ∪ (U \ B) = {0, 2, 3, 4, 5} := by sorry

end NUMINAMATH_CALUDE_complement_union_theorem_l1305_130540


namespace NUMINAMATH_CALUDE_factorization_equality_l1305_130558

theorem factorization_equality (x : ℝ) : 9*x^3 - 18*x^2 + 9*x = 9*x*(x-1)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l1305_130558


namespace NUMINAMATH_CALUDE_student_professor_ratio_l1305_130578

def total_people : ℕ := 40000
def num_students : ℕ := 37500

theorem student_professor_ratio :
  let num_professors := total_people - num_students
  num_students / num_professors = 15 := by
  sorry

end NUMINAMATH_CALUDE_student_professor_ratio_l1305_130578


namespace NUMINAMATH_CALUDE_pencil_cost_l1305_130500

/-- The cost of a pencil given total money and number of pencils that can be bought --/
theorem pencil_cost (total_money : ℚ) (num_pencils : ℕ) (h : total_money = 50 ∧ num_pencils = 10) : 
  total_money / num_pencils = 5 := by
  sorry

#check pencil_cost

end NUMINAMATH_CALUDE_pencil_cost_l1305_130500


namespace NUMINAMATH_CALUDE_quadratic_properties_l1305_130520

def quadratic_function (x : ℝ) : ℝ := x^2 - 2*x - 3

theorem quadratic_properties :
  (quadratic_function (-1) = 0) ∧
  (∀ x : ℝ, quadratic_function (1 + x) = quadratic_function (1 - x)) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_properties_l1305_130520


namespace NUMINAMATH_CALUDE_subtraction_multiplication_equality_l1305_130589

theorem subtraction_multiplication_equality : (3.456 - 1.234) * 0.5 = 1.111 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_multiplication_equality_l1305_130589


namespace NUMINAMATH_CALUDE_baking_time_proof_l1305_130522

-- Define the total baking time for 4 pans
def total_time : ℕ := 28

-- Define the number of pans
def num_pans : ℕ := 4

-- Define the time for one pan
def time_per_pan : ℕ := total_time / num_pans

-- Theorem to prove
theorem baking_time_proof : time_per_pan = 7 := by
  sorry

end NUMINAMATH_CALUDE_baking_time_proof_l1305_130522


namespace NUMINAMATH_CALUDE_boat_savings_l1305_130546

/-- The cost of traveling by plane in dollars -/
def plane_cost : ℚ := 600

/-- The cost of traveling by boat in dollars -/
def boat_cost : ℚ := 254

/-- The amount saved by taking a boat instead of a plane -/
def money_saved : ℚ := plane_cost - boat_cost

theorem boat_savings : money_saved = 346 := by
  sorry

end NUMINAMATH_CALUDE_boat_savings_l1305_130546


namespace NUMINAMATH_CALUDE_probability_two_red_one_black_l1305_130580

def num_red_balls : ℕ := 4
def num_black_balls : ℕ := 3
def total_balls : ℕ := num_red_balls + num_black_balls
def num_draws : ℕ := 3

def prob_red : ℚ := num_red_balls / total_balls
def prob_black : ℚ := num_black_balls / total_balls

def prob_two_red_one_black : ℚ := 3 * (prob_red * prob_red * prob_black)

theorem probability_two_red_one_black : 
  prob_two_red_one_black = 144 / 343 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_red_one_black_l1305_130580


namespace NUMINAMATH_CALUDE_sons_age_l1305_130586

/-- Given a man and his son, where the man is 32 years older than his son,
    and in two years the man's age will be twice the age of his son,
    prove that the present age of the son is 30 years. -/
theorem sons_age (son_age man_age : ℕ) : 
  man_age = son_age + 32 →
  man_age + 2 = 2 * (son_age + 2) →
  son_age = 30 := by
sorry

end NUMINAMATH_CALUDE_sons_age_l1305_130586


namespace NUMINAMATH_CALUDE_amanda_kimberly_distance_l1305_130514

/-- The distance between Amanda's house and Kimberly's house -/
def distance : ℝ := 6

/-- The time Amanda spent walking -/
def walking_time : ℝ := 3

/-- Amanda's walking speed -/
def walking_speed : ℝ := 2

/-- Theorem: The distance between Amanda's house and Kimberly's house is 6 miles -/
theorem amanda_kimberly_distance : distance = walking_time * walking_speed := by
  sorry

end NUMINAMATH_CALUDE_amanda_kimberly_distance_l1305_130514


namespace NUMINAMATH_CALUDE_hide_and_seek_players_l1305_130582

-- Define the players
variable (Andrew Boris Vasya Gena Denis : Prop)

-- Define the conditions
axiom condition1 : Andrew → (Boris ∧ ¬Vasya)
axiom condition2 : Boris → (Gena ∨ Denis)
axiom condition3 : ¬Vasya → (¬Boris ∧ ¬Denis)
axiom condition4 : ¬Andrew → (Boris ∧ ¬Gena)

-- Theorem to prove
theorem hide_and_seek_players :
  (Boris ∧ Vasya ∧ Denis ∧ ¬Andrew ∧ ¬Gena) ∧
  ∀ (A B V G D : Bool),
    (A → (B ∧ ¬V)) →
    (B → (G ∨ D)) →
    (¬V → (¬B ∧ ¬D)) →
    (¬A → (B ∧ ¬G)) →
    (A, B, V, G, D) = (false, true, true, false, true) :=
by sorry

end NUMINAMATH_CALUDE_hide_and_seek_players_l1305_130582


namespace NUMINAMATH_CALUDE_min_value_of_f_l1305_130525

/-- The quadratic expression in x and y with parameter k -/
def f (k x y : ℝ) : ℝ := 9*x^2 - 6*k*x*y + (3*k^2 + 1)*y^2 - 6*x - 6*y + 7

/-- The theorem stating that k = 3 is the unique value for which f has a minimum of 1 -/
theorem min_value_of_f :
  ∃! k : ℝ, (∀ x y : ℝ, f k x y ≥ 1) ∧ (∃ x y : ℝ, f k x y = 1) ∧ k = 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_f_l1305_130525


namespace NUMINAMATH_CALUDE_no_solution_implies_n_greater_than_one_l1305_130545

theorem no_solution_implies_n_greater_than_one (n : ℝ) :
  (∀ x : ℝ, ¬(x ≤ 1 ∧ x ≥ n)) → n > 1 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_implies_n_greater_than_one_l1305_130545


namespace NUMINAMATH_CALUDE_one_solution_r_product_l1305_130527

theorem one_solution_r_product (r : ℝ) : 
  (∃! x : ℝ, (1 / (2 * x) = (r - x) / 9)) → 
  (∃ r₁ r₂ : ℝ, r = r₁ ∨ r = r₂) ∧ (r₁ * r₂ = -18) :=
sorry

end NUMINAMATH_CALUDE_one_solution_r_product_l1305_130527


namespace NUMINAMATH_CALUDE_hillarys_craft_price_l1305_130541

/-- Proves that the price of each craft is $12 given the conditions of Hillary's sales and deposits -/
theorem hillarys_craft_price :
  ∀ (price : ℕ),
  (3 * price + 7 = 18 + 25) →
  price = 12 := by
sorry

end NUMINAMATH_CALUDE_hillarys_craft_price_l1305_130541


namespace NUMINAMATH_CALUDE_existence_of_homomorphism_l1305_130596

variable {G : Type*} [Group G]

def special_function (φ : G → G) : Prop :=
  ∀ a b c d e f : G, a * b * c = 1 ∧ d * e * f = 1 → φ a * φ b * φ c = φ d * φ e * φ f

theorem existence_of_homomorphism (φ : G → G) (h : special_function φ) :
  ∃ k : G, ∀ x y : G, k * φ (x * y) = (k * φ x) * (k * φ y) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_homomorphism_l1305_130596


namespace NUMINAMATH_CALUDE_peanut_butter_weight_calculation_l1305_130503

-- Define the ratio of oil to peanuts
def oil_to_peanuts_ratio : ℚ := 2 / 8

-- Define the amount of oil used
def oil_used : ℚ := 4

-- Define the function to calculate the total weight of peanut butter
def peanut_butter_weight (oil_amount : ℚ) : ℚ :=
  oil_amount + (oil_amount / oil_to_peanuts_ratio) * 8

-- Theorem statement
theorem peanut_butter_weight_calculation :
  peanut_butter_weight oil_used = 20 := by
  sorry

end NUMINAMATH_CALUDE_peanut_butter_weight_calculation_l1305_130503


namespace NUMINAMATH_CALUDE_app_security_theorem_all_measures_secure_l1305_130585

/-- Represents a security measure for protecting credit card data -/
inductive SecurityMeasure
  | avoidStoringCardData
  | encryptStoredData
  | encryptDataInTransit
  | codeObfuscation
  | restrictRootedDevices
  | antivirusProtection

/-- Represents an online store app with credit card payment and home delivery -/
structure OnlineStoreApp :=
  (implementedMeasures : List SecurityMeasure)

/-- Defines what it means for an app to be secure -/
def isSecure (app : OnlineStoreApp) : Prop :=
  app.implementedMeasures.length ≥ 3

/-- Theorem stating that implementing at least three security measures 
    ensures the app is secure -/
theorem app_security_theorem (app : OnlineStoreApp) :
  app.implementedMeasures.length ≥ 3 → isSecure app :=
by
  sorry

/-- Corollary: An app with all six security measures is secure -/
theorem all_measures_secure (app : OnlineStoreApp) :
  app.implementedMeasures.length = 6 → isSecure app :=
by
  sorry

end NUMINAMATH_CALUDE_app_security_theorem_all_measures_secure_l1305_130585


namespace NUMINAMATH_CALUDE_hotel_rate_problem_l1305_130561

theorem hotel_rate_problem (f n : ℝ) 
  (h1 : f + 3 * n = 210)  -- 4-night stay cost
  (h2 : f + 6 * n = 350)  -- 7-night stay cost
  : f = 70 := by
  sorry

end NUMINAMATH_CALUDE_hotel_rate_problem_l1305_130561


namespace NUMINAMATH_CALUDE_cryptarithm_multiplication_l1305_130595

theorem cryptarithm_multiplication :
  ∃! n : ℕ, ∃ m : ℕ,
    100 ≤ n ∧ n < 1000 ∧
    10000 ≤ m ∧ m < 100000 ∧
    n * n = m ∧
    ∃ k : ℕ, 100 ≤ k ∧ k < 1000 ∧ m = k * 1000 + k :=
by sorry

end NUMINAMATH_CALUDE_cryptarithm_multiplication_l1305_130595


namespace NUMINAMATH_CALUDE_saltwater_volume_proof_l1305_130577

theorem saltwater_volume_proof (x : ℝ) : 
  (0.20 * x + 12) / (0.75 * x + 18) = 1/3 → x = 120 := by
  sorry

end NUMINAMATH_CALUDE_saltwater_volume_proof_l1305_130577


namespace NUMINAMATH_CALUDE_square_root_five_expansion_l1305_130519

theorem square_root_five_expansion 
  (a b m n : ℤ) 
  (h : a + b * Real.sqrt 5 = (m + n * Real.sqrt 5) ^ 2) : 
  a = m^2 + 5*n^2 ∧ b = 2*m*n := by
sorry

end NUMINAMATH_CALUDE_square_root_five_expansion_l1305_130519


namespace NUMINAMATH_CALUDE_catia_speed_theorem_l1305_130505

/-- The speed at which Cátia should travel to reach home at 5:00 PM -/
def required_speed : ℝ := 12

/-- The time Cátia leaves school every day -/
def departure_time : ℝ := 3.75 -- 3:45 PM in decimal hours

/-- The distance from school to Cátia's home -/
def distance : ℝ := 15

/-- Arrival time when traveling at 20 km/h -/
def arrival_time_fast : ℝ := 4.5 -- 4:30 PM in decimal hours

/-- Arrival time when traveling at 10 km/h -/
def arrival_time_slow : ℝ := 5.25 -- 5:15 PM in decimal hours

/-- The desired arrival time -/
def desired_arrival_time : ℝ := 5 -- 5:00 PM in decimal hours

theorem catia_speed_theorem :
  (distance / (arrival_time_fast - departure_time) = 20) →
  (distance / (arrival_time_slow - departure_time) = 10) →
  (distance / (desired_arrival_time - departure_time) = required_speed) :=
by sorry

end NUMINAMATH_CALUDE_catia_speed_theorem_l1305_130505


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l1305_130524

theorem arithmetic_calculation : 5 * 7 + 6 * 9 + 13 * 2 + 4 * 6 = 139 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l1305_130524


namespace NUMINAMATH_CALUDE_cross_country_winning_scores_l1305_130517

/-- Represents a cross-country meet between two teams -/
structure CrossCountryMeet where
  /-- Total number of runners -/
  total_runners : Nat
  /-- Number of runners per team -/
  runners_per_team : Nat
  /-- Minimum possible team score -/
  min_score : Nat
  /-- Maximum possible team score -/
  max_score : Nat

/-- Calculates the number of different winning scores possible in a cross-country meet -/
def count_winning_scores (meet : CrossCountryMeet) : Nat :=
  sorry

/-- Theorem stating the number of different winning scores in the given cross-country meet -/
theorem cross_country_winning_scores :
  ∃ (meet : CrossCountryMeet),
    meet.total_runners = 10 ∧
    meet.runners_per_team = 5 ∧
    meet.min_score = 15 ∧
    meet.max_score = 40 ∧
    count_winning_scores meet = 13 :=
  sorry

end NUMINAMATH_CALUDE_cross_country_winning_scores_l1305_130517


namespace NUMINAMATH_CALUDE_sqrt_sum_square_condition_l1305_130593

theorem sqrt_sum_square_condition (a b : ℝ) :
  Real.sqrt (a^2 + b^2 + 2*a*b) = a + b ↔ a + b ≥ 0 := by sorry

end NUMINAMATH_CALUDE_sqrt_sum_square_condition_l1305_130593


namespace NUMINAMATH_CALUDE_probability_is_one_fourteenth_l1305_130523

/-- Represents a cube with side length 4 and two adjacent painted faces -/
structure PaintedCube :=
  (side_length : ℕ)
  (total_cubes : ℕ)
  (two_face_cubes : ℕ)
  (no_face_cubes : ℕ)

/-- The probability of selecting one cube with two painted faces and one with no painted faces -/
def select_probability (c : PaintedCube) : ℚ :=
  (c.two_face_cubes * c.no_face_cubes) / (c.total_cubes.choose 2)

/-- The theorem stating the probability is 1/14 -/
theorem probability_is_one_fourteenth (c : PaintedCube) 
  (h1 : c.side_length = 4)
  (h2 : c.total_cubes = 64)
  (h3 : c.two_face_cubes = 4)
  (h4 : c.no_face_cubes = 36) :
  select_probability c = 1 / 14 := by
  sorry

#eval select_probability { side_length := 4, total_cubes := 64, two_face_cubes := 4, no_face_cubes := 36 }

end NUMINAMATH_CALUDE_probability_is_one_fourteenth_l1305_130523


namespace NUMINAMATH_CALUDE_triangle_midpoint_dot_product_l1305_130539

-- Define the triangle ABC
def Triangle (A B C : ℝ × ℝ) : Prop :=
  let AB := Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)
  let AC := Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2)
  let BC := Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2)
  AB = 10 ∧ AC = 6 ∧ BC = 8

-- Define the midpoint
def Midpoint (M A B : ℝ × ℝ) : Prop :=
  M.1 = (A.1 + B.1) / 2 ∧ M.2 = (A.2 + B.2) / 2

-- Define the dot product
def DotProduct (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

-- Theorem statement
theorem triangle_midpoint_dot_product 
  (A B C M : ℝ × ℝ) 
  (h1 : Triangle A B C) 
  (h2 : Midpoint M A B) : 
  DotProduct (M.1 - C.1, M.2 - C.2) (A.1 - C.1, A.2 - C.2) + 
  DotProduct (M.1 - C.1, M.2 - C.2) (B.1 - C.1, B.2 - C.2) = 50 := by
  sorry

end NUMINAMATH_CALUDE_triangle_midpoint_dot_product_l1305_130539


namespace NUMINAMATH_CALUDE_equation_solution_l1305_130551

theorem equation_solution (x : ℝ) : 
  (∀ y : ℝ, 12 * x * y - 18 * y + 3 * x - 9 / 2 = 0) ↔ x = 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l1305_130551


namespace NUMINAMATH_CALUDE_problem_statement_l1305_130555

theorem problem_statement (p q r s : ℝ) 
  (h : (p - q) * (r - s) / ((q - r) * (s - p)) = 3 / 7) :
  (p - r) * (q - s) / ((p - q) * (r - s)) = -3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1305_130555


namespace NUMINAMATH_CALUDE_weight_at_170cm_l1305_130518

/-- Represents the weight of a student in kg -/
def weight : ℝ → ℝ := λ x => 0.75 * x - 68.2

/-- Theorem stating that for a height of 170 cm, the weight is 59.3 kg -/
theorem weight_at_170cm : weight 170 = 59.3 := by
  sorry

end NUMINAMATH_CALUDE_weight_at_170cm_l1305_130518


namespace NUMINAMATH_CALUDE_all_statements_false_l1305_130597

theorem all_statements_false : 
  (¬ ∀ (x : ℚ), -x ≠ |x|) ∧ 
  (¬ ∀ (x y : ℚ), x = -y → x = y⁻¹) ∧ 
  (¬ ∀ (x y : ℚ), |x| = |y| → x = y) := by
  sorry

end NUMINAMATH_CALUDE_all_statements_false_l1305_130597


namespace NUMINAMATH_CALUDE_sqrt_two_irrational_and_greater_than_one_l1305_130562

theorem sqrt_two_irrational_and_greater_than_one :
  ∃ x : ℝ, Irrational x ∧ x > 1 :=
by
  use Real.sqrt 2
  sorry

end NUMINAMATH_CALUDE_sqrt_two_irrational_and_greater_than_one_l1305_130562


namespace NUMINAMATH_CALUDE_girls_in_college_l1305_130587

theorem girls_in_college (total_students : ℕ) (boy_ratio girl_ratio : ℕ) : 
  total_students = 1040 →
  boy_ratio = 8 →
  girl_ratio = 5 →
  (boy_ratio + girl_ratio) * (total_students / (boy_ratio + girl_ratio)) = total_students →
  girl_ratio * (total_students / (boy_ratio + girl_ratio)) = 400 :=
by
  sorry


end NUMINAMATH_CALUDE_girls_in_college_l1305_130587


namespace NUMINAMATH_CALUDE_difference_of_cubes_factorization_l1305_130591

theorem difference_of_cubes_factorization (a b c d e : ℚ) :
  (∀ x, 512 * x^3 - 27 = (a*x + b) * (c*x^2 + d*x + e)) →
  a + b + c + d + e = 102 := by
sorry

end NUMINAMATH_CALUDE_difference_of_cubes_factorization_l1305_130591


namespace NUMINAMATH_CALUDE_expansion_properties_l1305_130565

/-- The binomial coefficient -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The expansion of (x+2)^n -/
def expansion (n : ℕ) (x : ℝ) : ℝ := sorry

/-- The coefficient of x^k in the expansion of (x+2)^n -/
def coeff (n k : ℕ) : ℕ := sorry

theorem expansion_properties :
  let n : ℕ := 8
  let a₀ : ℕ := coeff n 0
  let a₁ : ℕ := coeff n 1
  let a₂ : ℕ := coeff n 2
  -- a₀, a₁, a₂ form an arithmetic sequence
  (a₁ - a₀ = a₂ - a₁) →
  -- The middle (5th) term is 1120x⁴
  (coeff n 4 = 1120) ∧
  -- The sum of coefficients of odd powers is 3280
  (coeff n 1 + coeff n 3 + coeff n 5 + coeff n 7 = 3280) :=
by sorry

end NUMINAMATH_CALUDE_expansion_properties_l1305_130565


namespace NUMINAMATH_CALUDE_kitchen_clock_correct_time_bedroom_clock_correct_time_clocks_same_time_l1305_130552

-- Constants
def minutes_per_hour : ℚ := 60
def hours_per_day : ℚ := 24
def clock_cycle_minutes : ℚ := 720

-- Clock rates
def kitchen_clock_advance_rate : ℚ := 1.5
def bedroom_clock_slow_rate : ℚ := 0.5

-- Theorem for kitchen clock
theorem kitchen_clock_correct_time (t : ℚ) :
  t * kitchen_clock_advance_rate = clock_cycle_minutes →
  t / (hours_per_day * minutes_per_hour) = 20 := by sorry

-- Theorem for bedroom clock
theorem bedroom_clock_correct_time (t : ℚ) :
  t * bedroom_clock_slow_rate = clock_cycle_minutes →
  t / (hours_per_day * minutes_per_hour) = 60 := by sorry

-- Theorem for both clocks showing the same time
theorem clocks_same_time (t : ℚ) :
  t * (kitchen_clock_advance_rate + bedroom_clock_slow_rate) = clock_cycle_minutes →
  t / (hours_per_day * minutes_per_hour) = 15 := by sorry

end NUMINAMATH_CALUDE_kitchen_clock_correct_time_bedroom_clock_correct_time_clocks_same_time_l1305_130552


namespace NUMINAMATH_CALUDE_tournament_games_theorem_l1305_130570

/-- A single-elimination tournament with no ties. -/
structure Tournament :=
  (num_teams : ℕ)
  (no_ties : Bool)

/-- The number of games needed to declare a winner in a single-elimination tournament. -/
def games_to_winner (t : Tournament) : ℕ :=
  t.num_teams - 1

/-- Theorem: In a single-elimination tournament with 23 teams and no ties, 
    the number of games played to declare a winner is 22. -/
theorem tournament_games_theorem (t : Tournament) 
  (h1 : t.num_teams = 23) 
  (h2 : t.no_ties = true) : 
  games_to_winner t = 22 := by
  sorry


end NUMINAMATH_CALUDE_tournament_games_theorem_l1305_130570


namespace NUMINAMATH_CALUDE_geese_duck_difference_l1305_130566

/-- The number of more geese than ducks remaining at the duck park after a series of events --/
theorem geese_duck_difference : ℕ := by
  -- Define initial numbers
  let initial_ducks : ℕ := 25
  let initial_geese : ℕ := 2 * initial_ducks - 10
  let initial_swans : ℕ := 3 * initial_ducks + 8

  -- Define changes in population
  let arriving_ducks : ℕ := 4
  let arriving_geese : ℕ := 7
  let leaving_swans : ℕ := 9
  let leaving_geese : ℕ := 5
  let returning_geese : ℕ := 15
  let returning_swans : ℕ := 11

  -- Calculate intermediate populations
  let ducks_after_arrival : ℕ := initial_ducks + arriving_ducks
  let geese_after_arrival : ℕ := initial_geese + arriving_geese
  let swans_after_leaving : ℕ := initial_swans - leaving_swans
  let geese_after_leaving : ℕ := geese_after_arrival - leaving_geese
  let final_geese : ℕ := geese_after_leaving + returning_geese
  let final_swans : ℕ := swans_after_leaving + returning_swans

  -- Calculate birds leaving
  let leaving_ducks : ℕ := 2 * ducks_after_arrival
  let leaving_swans : ℕ := final_swans / 2

  -- Calculate final populations
  let remaining_ducks : ℕ := ducks_after_arrival - leaving_ducks
  let remaining_geese : ℕ := final_geese

  -- Prove the difference
  have h : remaining_geese - remaining_ducks = 57 := by sorry

  exact 57

end NUMINAMATH_CALUDE_geese_duck_difference_l1305_130566


namespace NUMINAMATH_CALUDE_functional_equation_solutions_l1305_130557

/-- A function from positive reals to positive reals -/
def PositiveRealFunction := {f : ℝ → ℝ // ∀ x, 0 < x → 0 < f x}

/-- The functional equation property -/
def SatisfiesFunctionalEquation (f : PositiveRealFunction) : Prop :=
  ∀ x y, 0 < x → 0 < y → f.val (x^y) = (f.val x)^(f.val y)

/-- The main theorem -/
theorem functional_equation_solutions (f : PositiveRealFunction) 
  (h : SatisfiesFunctionalEquation f) :
  (∀ x, 0 < x → f.val x = 1) ∨ (∀ x, 0 < x → f.val x = x) :=
sorry

end NUMINAMATH_CALUDE_functional_equation_solutions_l1305_130557


namespace NUMINAMATH_CALUDE_paul_school_supplies_l1305_130504

/-- Given Paul's initial crayons and erasers, and the number of crayons left,
    prove the difference between erasers and crayons left is 70. -/
theorem paul_school_supplies (initial_crayons : ℕ) (initial_erasers : ℕ) (crayons_left : ℕ)
    (h1 : initial_crayons = 601)
    (h2 : initial_erasers = 406)
    (h3 : crayons_left = 336) :
    initial_erasers - crayons_left = 70 := by
  sorry

end NUMINAMATH_CALUDE_paul_school_supplies_l1305_130504


namespace NUMINAMATH_CALUDE_joe_monthly_income_correct_l1305_130599

/-- Joe's monthly income in dollars -/
def monthly_income : ℝ := 2120

/-- The fraction of Joe's income that goes to taxes -/
def tax_rate : ℝ := 0.4

/-- The amount Joe pays in taxes each month in dollars -/
def tax_paid : ℝ := 848

/-- Theorem stating that Joe's monthly income is correct given the tax rate and tax paid -/
theorem joe_monthly_income_correct : 
  tax_rate * monthly_income = tax_paid :=
by sorry

end NUMINAMATH_CALUDE_joe_monthly_income_correct_l1305_130599


namespace NUMINAMATH_CALUDE_fraction_equality_l1305_130536

theorem fraction_equality : (18 - 6) / ((3 + 3) * 2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1305_130536


namespace NUMINAMATH_CALUDE_sin_2alpha_minus_pi_3_l1305_130542

theorem sin_2alpha_minus_pi_3 (α : ℝ) (h : Real.cos (α + π / 12) = -3 / 4) :
  Real.sin (2 * α - π / 3) = -1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_sin_2alpha_minus_pi_3_l1305_130542


namespace NUMINAMATH_CALUDE_star_polygon_n_is_24_l1305_130567

/-- A n-pointed regular star polygon -/
structure StarPolygon (n : ℕ) where
  edges : Fin (2 * n) → ℝ
  angleA : Fin n → ℝ
  angleB : Fin n → ℝ
  edges_congruent : ∀ i j, edges i = edges j
  angleA_congruent : ∀ i j, angleA i = angleA j
  angleB_congruent : ∀ i j, angleB i = angleB j
  angle_difference : ∀ i, angleA i = angleB i - 15

/-- The theorem stating that n = 24 for the given star polygon -/
theorem star_polygon_n_is_24 (n : ℕ) (star : StarPolygon n) : n = 24 := by
  sorry

end NUMINAMATH_CALUDE_star_polygon_n_is_24_l1305_130567


namespace NUMINAMATH_CALUDE_circle_radius_l1305_130584

/-- A circle with equation x^2 + y^2 - 2x + my - 4 = 0 that is symmetric about the line 2x + y = 0 has a radius of 3 -/
theorem circle_radius (m : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 - 2*x + m*y - 4 = 0 → (∃ x' y' : ℝ, x'^2 + y'^2 - 2*x' + m*y' - 4 = 0 ∧ 
    2*x + y = 0 ∧ 2*x' + y' = 0 ∧ x + x' = 2*x ∧ y + y' = 2*y)) → 
  (∃ c_x c_y : ℝ, ∀ x y : ℝ, (x - c_x)^2 + (y - c_y)^2 = 3^2 ↔ x^2 + y^2 - 2*x + m*y - 4 = 0) :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_l1305_130584


namespace NUMINAMATH_CALUDE_quilt_shaded_fraction_l1305_130590

theorem quilt_shaded_fraction (total_squares : ℕ) (divided_squares : ℕ) 
  (h1 : total_squares = 16) 
  (h2 : divided_squares = 8) : 
  (divided_squares : ℚ) / (2 : ℚ) / total_squares = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_quilt_shaded_fraction_l1305_130590


namespace NUMINAMATH_CALUDE_probability_red_ball_in_bag_l1305_130538

/-- The probability of drawing a red ball from a bag -/
def probability_red_ball (total_balls : ℕ) (red_balls : ℕ) : ℚ :=
  red_balls / total_balls

/-- Theorem: The probability of drawing a red ball from an opaque bag with 5 balls, 2 of which are red, is 2/5 -/
theorem probability_red_ball_in_bag : probability_red_ball 5 2 = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_red_ball_in_bag_l1305_130538


namespace NUMINAMATH_CALUDE_complement_A_inter_B_a_range_l1305_130594

open Set

-- Define the sets A and B
def A : Set ℝ := {x | 2 ≤ x ∧ x ≤ 6}
def B : Set ℝ := {x | x ≥ 3}

-- Define the complement of the intersection of A and B
def complement_intersection : Set ℝ := {x | x < 3 ∨ x > 6}

-- Define set C
def C (a : ℝ) : Set ℝ := {x | x ≤ a}

-- Theorem 1: The complement of A ∩ B is equal to the defined complement_intersection
theorem complement_A_inter_B : (A ∩ B)ᶜ = complement_intersection := by sorry

-- Theorem 2: If A is a subset of C, then a is greater than or equal to 6
theorem a_range (a : ℝ) (h : A ⊆ C a) : a ≥ 6 := by sorry

end NUMINAMATH_CALUDE_complement_A_inter_B_a_range_l1305_130594


namespace NUMINAMATH_CALUDE_hyperbola_ellipse_intersection_l1305_130568

theorem hyperbola_ellipse_intersection (m : ℝ) : 
  (∃ e : ℝ, e > Real.sqrt 2 ∧ e^2 = (3 + m) / 3) ∧ 
  (m / 2 > m - 2 ∧ m - 2 > 0) → 
  m ∈ Set.Ioo 3 4 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_ellipse_intersection_l1305_130568


namespace NUMINAMATH_CALUDE_problem_statement_l1305_130598

def P : Set ℝ := {-1, 1}
def Q (a : ℝ) : Set ℝ := {x : ℝ | a * x = 1}

theorem problem_statement (a : ℝ) : P ∪ Q a = P → a ∈ ({-1, 0, 1} : Set ℝ) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1305_130598


namespace NUMINAMATH_CALUDE_sunnydale_walk_home_fraction_l1305_130512

/-- The fraction of students who walk home at Sunnydale Middle School -/
theorem sunnydale_walk_home_fraction :
  let bus_fraction : ℚ := 1/3
  let auto_fraction : ℚ := 1/5
  let bike_fraction : ℚ := 1/8
  let walk_fraction : ℚ := 1 - (bus_fraction + auto_fraction + bike_fraction)
  walk_fraction = 41/120 := by
  sorry

end NUMINAMATH_CALUDE_sunnydale_walk_home_fraction_l1305_130512


namespace NUMINAMATH_CALUDE_exponential_function_passes_through_one_l1305_130537

theorem exponential_function_passes_through_one (a : ℝ) (ha : a > 0) (hna : a ≠ 1) :
  let f : ℝ → ℝ := fun x ↦ a^x
  f 0 = 1 := by sorry

end NUMINAMATH_CALUDE_exponential_function_passes_through_one_l1305_130537


namespace NUMINAMATH_CALUDE_special_permutation_exists_l1305_130528

/-- A permutation of numbers from 1 to 2^n satisfying the special property -/
def SpecialPermutation (n : ℕ) : List ℕ :=
  sorry

/-- Predicate to check if a list satisfies the special property -/
def SatisfiesProperty (lst : List ℕ) : Prop :=
  ∀ i j, i < j → i < lst.length → j < lst.length →
    ∀ k, i < k ∧ k < j →
      (lst.get ⟨i, sorry⟩ + lst.get ⟨j, sorry⟩) / 2 ≠ lst.get ⟨k, sorry⟩

/-- Theorem stating that for any n, there exists a permutation of numbers
    from 1 to 2^n satisfying the special property -/
theorem special_permutation_exists (n : ℕ) :
  ∃ (perm : List ℕ), perm.length = 2^n ∧
    (∀ i, i ∈ perm ↔ 1 ≤ i ∧ i ≤ 2^n) ∧
    SatisfiesProperty perm :=
  sorry

end NUMINAMATH_CALUDE_special_permutation_exists_l1305_130528


namespace NUMINAMATH_CALUDE_johns_journey_length_l1305_130592

theorem johns_journey_length :
  ∀ (total_length : ℝ),
  (total_length / 4 : ℝ) + 30 + (1/3 : ℝ) * (total_length - total_length / 4 - 30) = total_length →
  total_length = 160 := by
sorry

end NUMINAMATH_CALUDE_johns_journey_length_l1305_130592


namespace NUMINAMATH_CALUDE_cubic_polynomial_integer_root_l1305_130548

theorem cubic_polynomial_integer_root 
  (d e : ℚ) 
  (h1 : ∃ x : ℝ, x^3 + d*x + e = 0 ∧ x = 2 - Real.sqrt 5)
  (h2 : ∃ n : ℤ, n^3 + d*n + e = 0) :
  ∃ n : ℤ, n^3 + d*n + e = 0 ∧ n = -4 :=
sorry

end NUMINAMATH_CALUDE_cubic_polynomial_integer_root_l1305_130548


namespace NUMINAMATH_CALUDE_tile_count_equivalence_l1305_130547

theorem tile_count_equivalence (area : ℝ) : 
  area = (0.3 : ℝ)^2 * 720 → area = (0.4 : ℝ)^2 * 405 := by
  sorry

end NUMINAMATH_CALUDE_tile_count_equivalence_l1305_130547


namespace NUMINAMATH_CALUDE_square_perimeter_l1305_130533

theorem square_perimeter (area_A : ℝ) (prob : ℝ) (perimeter_B : ℝ) : 
  area_A = 121 →
  prob = 0.8677685950413223 →
  prob = (area_A - (perimeter_B / 4)^2) / area_A →
  perimeter_B = 16 := by
sorry

end NUMINAMATH_CALUDE_square_perimeter_l1305_130533


namespace NUMINAMATH_CALUDE_cubic_three_zeros_l1305_130543

/-- A cubic function with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := 2 * x^3 - 3 * a * x^2 + a

/-- The derivative of f with respect to x -/
def f' (a : ℝ) (x : ℝ) : ℝ := 6 * x^2 - 6 * a * x

theorem cubic_three_zeros (a : ℝ) :
  (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧ 
    f a x₁ = 0 ∧ f a x₂ = 0 ∧ f a x₃ = 0) →
  a > 1 ∨ a < -1 := by
  sorry

end NUMINAMATH_CALUDE_cubic_three_zeros_l1305_130543


namespace NUMINAMATH_CALUDE_binary_110011_equals_51_l1305_130535

def binary_to_decimal (b : List Bool) : Nat :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_110011_equals_51 :
  binary_to_decimal [true, true, false, false, true, true] = 51 := by
  sorry

end NUMINAMATH_CALUDE_binary_110011_equals_51_l1305_130535


namespace NUMINAMATH_CALUDE_car_journey_downhill_distance_l1305_130534

/-- Proves that a car traveling 100 km uphill at 30 km/hr and an unknown distance downhill
    at 60 km/hr, with an average speed of 36 km/hr for the entire journey,
    travels 50 km downhill. -/
theorem car_journey_downhill_distance
  (uphill_speed : ℝ) (downhill_speed : ℝ) (uphill_distance : ℝ) (average_speed : ℝ)
  (h1 : uphill_speed = 30)
  (h2 : downhill_speed = 60)
  (h3 : uphill_distance = 100)
  (h4 : average_speed = 36)
  : ∃ (downhill_distance : ℝ),
    (uphill_distance + downhill_distance) / ((uphill_distance / uphill_speed) + (downhill_distance / downhill_speed)) = average_speed
    ∧ downhill_distance = 50 :=
by sorry

end NUMINAMATH_CALUDE_car_journey_downhill_distance_l1305_130534


namespace NUMINAMATH_CALUDE_polygon_angles_l1305_130576

theorem polygon_angles (n : ℕ) (h : n > 2) : 
  (n - 2) * 180 = 3 * 360 → n = 8 := by
  sorry

end NUMINAMATH_CALUDE_polygon_angles_l1305_130576


namespace NUMINAMATH_CALUDE_polynomial_factorization_l1305_130516

theorem polynomial_factorization (k : ℤ) :
  ∃ (p q : Polynomial ℤ),
    Polynomial.degree p = 4 ∧
    Polynomial.degree q = 4 ∧
    (X : Polynomial ℤ)^8 + (4 * k^4 - 8 * k^2 + 2) * X^4 + 1 = p * q :=
sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l1305_130516


namespace NUMINAMATH_CALUDE_select_two_from_five_assign_prizes_l1305_130554

/-- The number of ways to select 2 people from n employees and assign them distinct prizes -/
def select_and_assign (n : ℕ) : ℕ := n * (n - 1)

/-- Theorem: For 5 employees, there are 20 ways to select 2 and assign distinct prizes -/
theorem select_two_from_five_assign_prizes :
  select_and_assign 5 = 20 := by
  sorry

end NUMINAMATH_CALUDE_select_two_from_five_assign_prizes_l1305_130554


namespace NUMINAMATH_CALUDE_fewer_toys_by_machine_a_l1305_130572

/-- The number of toys machine A makes per minute -/
def machine_a_rate : ℕ := 8

/-- The number of toys machine B makes per minute -/
def machine_b_rate : ℕ := 10

/-- The number of toys machine B made -/
def machine_b_toys : ℕ := 100

/-- The time both machines operated, in minutes -/
def operation_time : ℕ := machine_b_toys / machine_b_rate

/-- The number of toys machine A made -/
def machine_a_toys : ℕ := machine_a_rate * operation_time

theorem fewer_toys_by_machine_a : machine_b_toys - machine_a_toys = 20 := by
  sorry

end NUMINAMATH_CALUDE_fewer_toys_by_machine_a_l1305_130572


namespace NUMINAMATH_CALUDE_incenter_characterization_l1305_130553

/-- Triangle ABC with point P inside -/
structure Triangle :=
  (A B C P : ℝ × ℝ)

/-- The perimeter of a triangle -/
def perimeter (t : Triangle) : ℝ := sorry

/-- The area of a triangle -/
def area (t : Triangle) : ℝ := sorry

/-- Perpendicular distance from a point to a line segment -/
def perpDistance (point : ℝ × ℝ) (segment : (ℝ × ℝ) × (ℝ × ℝ)) : ℝ := sorry

/-- Length of a line segment -/
def segmentLength (segment : (ℝ × ℝ) × (ℝ × ℝ)) : ℝ := sorry

/-- The incenter of a triangle -/
def incenter (t : Triangle) : ℝ × ℝ := sorry

theorem incenter_characterization (t : Triangle) :
  let l := perimeter t
  let s := area t
  let PD := perpDistance t.P (t.A, t.B)
  let PE := perpDistance t.P (t.B, t.C)
  let PF := perpDistance t.P (t.C, t.A)
  let AB := segmentLength (t.A, t.B)
  let BC := segmentLength (t.B, t.C)
  let CA := segmentLength (t.C, t.A)
  AB / PD + BC / PE + CA / PF ≤ l^2 / (2 * s) →
  t.P = incenter t :=
by sorry

end NUMINAMATH_CALUDE_incenter_characterization_l1305_130553


namespace NUMINAMATH_CALUDE_sum_of_fifth_powers_zero_l1305_130513

theorem sum_of_fifth_powers_zero (a b c : ℚ) 
  (sum_zero : a + b + c = 0) 
  (sum_cubes_nonzero : a^3 + b^3 + c^3 ≠ 0) : 
  a^5 + b^5 + c^5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fifth_powers_zero_l1305_130513


namespace NUMINAMATH_CALUDE_expression_value_l1305_130530

theorem expression_value :
  let a : ℝ := 10
  let b : ℝ := 4
  let c : ℝ := 3
  (a - (b - c^2)) - ((a - b) - c^2) = 18 := by sorry

end NUMINAMATH_CALUDE_expression_value_l1305_130530


namespace NUMINAMATH_CALUDE_mark_height_in_feet_l1305_130560

/-- Represents a height in feet and inches -/
structure Height where
  feet : ℕ
  inches : ℕ
  h : inches < 12

/-- Converts a Height to total inches -/
def Height.toInches (h : Height) : ℕ := h.feet * 12 + h.inches

/-- The height difference between Mike and Mark in inches -/
def heightDifference : ℕ := 10

/-- Mike's height -/
def mikeHeight : Height := ⟨6, 1, by sorry⟩

/-- Mark's height in inches -/
def markHeightInches : ℕ := mikeHeight.toInches - heightDifference

theorem mark_height_in_feet :
  ∃ (h : Height), h.toInches = markHeightInches ∧ h.feet = 5 ∧ h.inches = 3 := by
  sorry

end NUMINAMATH_CALUDE_mark_height_in_feet_l1305_130560


namespace NUMINAMATH_CALUDE_alpha_beta_relation_l1305_130515

open Real

theorem alpha_beta_relation (α β : ℝ) :
  π / 2 < α ∧ α < π ∧
  π / 2 < β ∧ β < π ∧
  (1 - cos (2 * α)) * (1 + sin β) = sin (2 * α) * cos β →
  2 * α + β = 5 * π / 2 := by
sorry

end NUMINAMATH_CALUDE_alpha_beta_relation_l1305_130515


namespace NUMINAMATH_CALUDE_largest_two_digit_prime_factor_of_binomial_l1305_130521

theorem largest_two_digit_prime_factor_of_binomial :
  ∃ (p : ℕ), 
    p.Prime ∧ 
    10 ≤ p ∧ p < 100 ∧
    p ∣ Nat.choose 150 75 ∧
    (∀ q : ℕ, q.Prime → 10 ≤ q → q < 100 → q ∣ Nat.choose 150 75 → q ≤ p) ∧
    (∀ q : ℕ, q > p → ¬(q.Prime ∧ 10 ≤ q ∧ q < 100 ∧ q ∣ Nat.choose 150 75)) :=
by
  sorry

#check largest_two_digit_prime_factor_of_binomial

end NUMINAMATH_CALUDE_largest_two_digit_prime_factor_of_binomial_l1305_130521


namespace NUMINAMATH_CALUDE_shoe_shopping_cost_l1305_130526

theorem shoe_shopping_cost 
  (price1 price2 price3 : ℝ) 
  (half_off_discount : ℝ → ℝ)
  (third_pair_discount : ℝ → ℝ)
  (extra_discount : ℝ → ℝ)
  (sales_tax : ℝ → ℝ)
  (h1 : price1 = 40)
  (h2 : price2 = 60)
  (h3 : price3 = 80)
  (h4 : half_off_discount x = x / 2)
  (h5 : third_pair_discount x = x * 0.7)
  (h6 : extra_discount x = x * 0.75)
  (h7 : sales_tax x = x * 1.08)
  : sales_tax (extra_discount (price1 + (price2 - half_off_discount price1) + third_pair_discount price3)) = 110.16 := by
  sorry

end NUMINAMATH_CALUDE_shoe_shopping_cost_l1305_130526


namespace NUMINAMATH_CALUDE_sin_to_cos_shift_l1305_130574

theorem sin_to_cos_shift (x : ℝ) :
  let f : ℝ → ℝ := λ t ↦ Real.sin (t - π/3)
  let g : ℝ → ℝ := λ t ↦ Real.cos t
  f (x + 5*π/6) = g x := by
sorry

end NUMINAMATH_CALUDE_sin_to_cos_shift_l1305_130574


namespace NUMINAMATH_CALUDE_student_allowance_l1305_130502

theorem student_allowance (allowance : ℝ) : 
  (allowance * 2/5 * 2/3 * 3/4 * 9/10 = 1.20) → 
  allowance = 60 := by
sorry

end NUMINAMATH_CALUDE_student_allowance_l1305_130502
