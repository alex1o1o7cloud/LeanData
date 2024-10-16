import Mathlib

namespace NUMINAMATH_CALUDE_complement_A_equals_negative_reals_l2086_208684

-- Define the universe U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define set A as the set of non-negative real numbers
def A : Set ℝ := { x : ℝ | x ≥ 0 }

-- Define the complement of A in U
def complement_A : Set ℝ := U \ A

-- Theorem statement
theorem complement_A_equals_negative_reals :
  complement_A = { x : ℝ | x < 0 } :=
sorry

end NUMINAMATH_CALUDE_complement_A_equals_negative_reals_l2086_208684


namespace NUMINAMATH_CALUDE_hexagon_area_theorem_l2086_208632

/-- A regular hexagon inscribed in a circle of unit area -/
structure RegularHexagonInCircle where
  /-- The circle has unit area -/
  circle_area : ℝ := 1
  /-- The hexagon is inscribed in the circle -/
  hexagon_inscribed : Bool

/-- A point Q inside the circle -/
structure PointInCircle where
  /-- The point is inside the circle -/
  inside : Bool

/-- The area of a region bounded by two sides of the hexagon and a minor arc -/
def area_region (h : RegularHexagonInCircle) (q : PointInCircle) (i j : Fin 6) : ℝ := sorry

theorem hexagon_area_theorem (h : RegularHexagonInCircle) (q : PointInCircle) :
  area_region h q 0 1 = 1 / 12 →
  area_region h q 2 3 = 1 / 15 →
  ∃ m : ℕ+, area_region h q 4 5 = 1 / 18 - Real.sqrt 3 / m →
  m = 20 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_hexagon_area_theorem_l2086_208632


namespace NUMINAMATH_CALUDE_complex_expression_equality_l2086_208641

-- Define the complex number z
def z : ℂ := 1 + Complex.I

-- State the theorem
theorem complex_expression_equality : (2 / z) + z^2 = 1 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_equality_l2086_208641


namespace NUMINAMATH_CALUDE_product_mod_30_l2086_208636

theorem product_mod_30 : ∃ m : ℕ, 0 ≤ m ∧ m < 30 ∧ (33 * 77 * 99) % 30 = m ∧ m = 9 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_30_l2086_208636


namespace NUMINAMATH_CALUDE_sin_square_sum_range_l2086_208673

theorem sin_square_sum_range (α β : ℝ) (h : 3 * (Real.sin α)^2 - 2 * Real.sin α + 2 * (Real.sin β)^2 = 0) :
  ∃ (x : ℝ), x = (Real.sin α)^2 + (Real.sin β)^2 ∧ 0 ≤ x ∧ x ≤ 4/9 :=
sorry

end NUMINAMATH_CALUDE_sin_square_sum_range_l2086_208673


namespace NUMINAMATH_CALUDE_largest_angle_in_pentagon_l2086_208626

theorem largest_angle_in_pentagon (A B C D E : ℝ) : 
  A = 70 → 
  B = 120 → 
  C = D → 
  E = 3 * C - 30 → 
  A + B + C + D + E = 540 → 
  max A (max B (max C (max D E))) = 198 :=
by sorry

end NUMINAMATH_CALUDE_largest_angle_in_pentagon_l2086_208626


namespace NUMINAMATH_CALUDE_sequence_properties_l2086_208602

-- Define the sequence a_n and its partial sum S_n
def S (n : ℕ) (a : ℕ → ℝ) : ℝ := 2 * a n - 2^n

theorem sequence_properties (a : ℕ → ℝ) :
  (∀ n, S n a = 2 * a n - 2^n) →
  (∃ r : ℝ, ∀ n, a (n + 1) - 2 * a n = r * (a n - 2 * a (n - 1))) ∧
  (∀ n, a n = (n + 1) * 2^(n - 1)) :=
by sorry

end NUMINAMATH_CALUDE_sequence_properties_l2086_208602


namespace NUMINAMATH_CALUDE_real_roots_iff_a_leq_two_l2086_208628

theorem real_roots_iff_a_leq_two (a : ℝ) : 
  (∃ x : ℝ, (a - 1) * x^2 - 2 * x + 1 = 0) ↔ a ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_real_roots_iff_a_leq_two_l2086_208628


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l2086_208618

theorem solution_set_of_inequality (x : ℝ) :
  Set.Icc (-1/2 : ℝ) 3 \ {3} = {x | (2*x + 1) / (3 - x) ≥ 0 ∧ x ≠ 3} :=
sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l2086_208618


namespace NUMINAMATH_CALUDE_solution_set_max_value_min_value_l2086_208698

-- Define the function f
def f (x : ℝ) : ℝ := |x + 1| - |2*x - 2|

-- Theorem 1: Solution set of f(x) ≥ x-1
theorem solution_set (x : ℝ) : f x ≥ x - 1 ↔ 0 ≤ x ∧ x ≤ 2 :=
sorry

-- Theorem 2: Maximum value of f
theorem max_value : ∃ (x : ℝ), ∀ (y : ℝ), f y ≤ f x ∧ f x = 2 :=
sorry

-- Theorem 3: Minimum value of expression
theorem min_value (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (sum : a + b + c = 2) :
  (b^2 / a) + (c^2 / b) + (a^2 / c) ≥ 2 :=
sorry

end NUMINAMATH_CALUDE_solution_set_max_value_min_value_l2086_208698


namespace NUMINAMATH_CALUDE_power_of_product_cube_l2086_208656

theorem power_of_product_cube (x : ℝ) : (2 * x^3)^2 = 4 * x^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_cube_l2086_208656


namespace NUMINAMATH_CALUDE_problem_statement_l2086_208601

noncomputable def f (x : ℝ) : ℝ := x * Real.log x - x
noncomputable def g (x a : ℝ) : ℝ := Real.log x - a * x + 1

theorem problem_statement :
  (∀ x : ℝ, x > 0 → deriv f x = Real.log x) ∧
  (∀ a : ℝ, (∀ x : ℝ, x > 0 → g x a ≤ 0) → a ≥ 1) ∧
  (∀ m x n : ℝ, 0 < m → m < x → x < n →
    (f x - f m) / (x - m) < (f x - f n) / (x - n)) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2086_208601


namespace NUMINAMATH_CALUDE_work_time_for_less_efficient_worker_l2086_208649

/-- Represents the time it takes for a worker to complete a job alone -/
def WorkTime := ℝ

/-- Represents the efficiency of a worker (fraction of job completed per day) -/
def Efficiency := ℝ

theorem work_time_for_less_efficient_worker 
  (total_time : ℝ) 
  (efficiency_ratio : ℝ) :
  total_time > 0 →
  efficiency_ratio > 1 →
  let joint_efficiency := 1 / total_time
  let less_efficient_worker_efficiency := joint_efficiency / (1 + efficiency_ratio)
  let work_time_less_efficient := 1 / less_efficient_worker_efficiency
  (total_time = 36 ∧ efficiency_ratio = 2) → work_time_less_efficient = 108 := by
  sorry

end NUMINAMATH_CALUDE_work_time_for_less_efficient_worker_l2086_208649


namespace NUMINAMATH_CALUDE_exactly_seven_numbers_l2086_208667

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def swap_digits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

def is_perfect_cube (n : ℤ) : Prop :=
  ∃ m : ℤ, n = m^3

def satisfies_condition (n : ℕ) : Prop :=
  is_two_digit n ∧ is_perfect_cube (n - swap_digits n)

theorem exactly_seven_numbers :
  ∃! (s : Finset ℕ), s.card = 7 ∧ ∀ n, n ∈ s ↔ satisfies_condition n :=
sorry

end NUMINAMATH_CALUDE_exactly_seven_numbers_l2086_208667


namespace NUMINAMATH_CALUDE_circle_through_points_l2086_208612

-- Define the points
def O : ℝ × ℝ := (0, 0)
def M1 : ℝ × ℝ := (1, 1)
def M2 : ℝ × ℝ := (4, 2)

-- Define the circle
def Circle (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Theorem statement
theorem circle_through_points : 
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    center = (4, -3) ∧ 
    radius = 5 ∧
    O ∈ Circle center radius ∧
    M1 ∈ Circle center radius ∧
    M2 ∈ Circle center radius ∧
    Circle center radius = {p : ℝ × ℝ | (p.1 - 4)^2 + (p.2 + 3)^2 = 25} := by
  sorry

end NUMINAMATH_CALUDE_circle_through_points_l2086_208612


namespace NUMINAMATH_CALUDE_min_value_m_plus_n_l2086_208646

theorem min_value_m_plus_n (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : Real.sqrt (a * b) = 2) 
  (m n : ℝ) (h4 : m = b + 1/a) (h5 : n = a + 1/b) : 
  ∀ x y : ℝ, x > 0 ∧ y > 0 ∧ Real.sqrt (x * y) = 2 → m + n ≤ x + y + 1/x + 1/y ∧ m + n ≥ 5 := by
  sorry

end NUMINAMATH_CALUDE_min_value_m_plus_n_l2086_208646


namespace NUMINAMATH_CALUDE_solve_for_a_l2086_208642

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

-- State the theorem
theorem solve_for_a (a : ℝ) (h1 : a > 1) :
  (∀ x, 1 ≤ x ∧ x ≤ 2 ↔ |f a (2*x + a) - 2*f a x| ≤ 2) →
  a = 3 := by
sorry

end NUMINAMATH_CALUDE_solve_for_a_l2086_208642


namespace NUMINAMATH_CALUDE_quadratic_transformation_l2086_208607

/-- The transformation from y = -2x^2 + 4x + 1 to y = -2x^2 -/
theorem quadratic_transformation (f g : ℝ → ℝ) (h_f : f = λ x => -2*x^2 + 4*x + 1) (h_g : g = λ x => -2*x^2) : 
  (∃ (a b : ℝ), ∀ x, f x = g (x + a) + b) ∧ 
  (∃ (vertex_f vertex_g : ℝ × ℝ), 
    vertex_f = (1, 3) ∧ 
    vertex_g = (0, 0) ∧ 
    vertex_f.1 - vertex_g.1 = 1 ∧ 
    vertex_f.2 - vertex_g.2 = 3) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_transformation_l2086_208607


namespace NUMINAMATH_CALUDE_brian_read_75_chapters_l2086_208631

/-- The number of chapters Brian read -/
def brian_total_chapters : ℕ :=
  let book1_chapters := 20
  let book2_chapters := 15
  let book3_chapters := 15
  let first_three_books := book1_chapters + book2_chapters + book3_chapters
  let book4_chapters := first_three_books / 2
  book1_chapters + book2_chapters + book3_chapters + book4_chapters

theorem brian_read_75_chapters : brian_total_chapters = 75 := by
  sorry

end NUMINAMATH_CALUDE_brian_read_75_chapters_l2086_208631


namespace NUMINAMATH_CALUDE_fuel_price_increase_l2086_208608

/-- Calculate the percentage increase in fuel prices given the original cost for one tank,
    the doubling of fuel capacity, and the new cost for both tanks. -/
theorem fuel_price_increase (original_cost : ℝ) (new_cost : ℝ) : 
  original_cost = 200 →
  new_cost = 480 →
  (new_cost / (2 * original_cost) - 1) * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_fuel_price_increase_l2086_208608


namespace NUMINAMATH_CALUDE_fifth_inequality_l2086_208691

theorem fifth_inequality (h1 : 1 / Real.sqrt 2 < 1)
  (h2 : 1 / Real.sqrt 2 + 1 / Real.sqrt 6 < Real.sqrt 2)
  (h3 : 1 / Real.sqrt 2 + 1 / Real.sqrt 6 + 1 / Real.sqrt 12 < Real.sqrt 3) :
  1 / Real.sqrt 2 + 1 / Real.sqrt 6 + 1 / Real.sqrt 12 + 1 / Real.sqrt 20 + 1 / Real.sqrt 30 < Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_fifth_inequality_l2086_208691


namespace NUMINAMATH_CALUDE_greatest_integer_for_domain_all_reals_l2086_208693

theorem greatest_integer_for_domain_all_reals : 
  ∃ (b : ℤ), b = 11 ∧ 
  (∀ (c : ℤ), c > b → 
    ∃ (x : ℝ), 2 * x^2 + c * x + 18 = 0) ∧
  (∀ (x : ℝ), 2 * x^2 + b * x + 18 ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_greatest_integer_for_domain_all_reals_l2086_208693


namespace NUMINAMATH_CALUDE_firstDigitOfPowerOfTwoNotPeriodic_l2086_208617

-- Define the sequence of first digits of powers of 2
def firstDigitOfPowerOfTwo (n : ℕ) : ℕ :=
  (2^n : ℕ).repr.front.toNat

-- Theorem statement
theorem firstDigitOfPowerOfTwoNotPeriodic :
  ¬ ∃ (d : ℕ), d > 0 ∧ ∀ (n : ℕ), firstDigitOfPowerOfTwo (n + d) = firstDigitOfPowerOfTwo n :=
sorry

end NUMINAMATH_CALUDE_firstDigitOfPowerOfTwoNotPeriodic_l2086_208617


namespace NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l2086_208623

theorem greatest_divisor_with_remainders (a b r1 r2 : ℕ) (h1 : a > r1) (h2 : b > r2) : 
  Nat.gcd (a - r1) (b - r2) = 
    Nat.gcd (a % (Nat.gcd (a - r1) (b - r2))) r1 ∧ 
    Nat.gcd (a - r1) (b - r2) = 
    Nat.gcd (b % (Nat.gcd (a - r1) (b - r2))) r2 → 
  Nat.gcd (a - r1) (b - r2) = 
    Nat.gcd (1642 - 6) (1856 - 4) := by
  sorry

#eval Nat.gcd (1642 - 6) (1856 - 4)

end NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l2086_208623


namespace NUMINAMATH_CALUDE_b_work_time_l2086_208670

/-- Represents the time in days it takes for a person to complete a task alone. -/
structure WorkTime where
  days : ℚ
  days_pos : days > 0

/-- Represents the rate at which a person completes a task, as a fraction of the task per day. -/
def workRate (wt : WorkTime) : ℚ := 1 / wt.days

/-- The combined work rate of multiple people working together. -/
def combinedWorkRate (rates : List ℚ) : ℚ := rates.sum

theorem b_work_time (a_time : WorkTime) (c_time : WorkTime) (abc_time : WorkTime) 
  (ha : a_time.days = 8)
  (hc : c_time.days = 24)
  (habc : abc_time.days = 4) :
  ∃ (b_time : WorkTime), b_time.days = 12 := by
  sorry

end NUMINAMATH_CALUDE_b_work_time_l2086_208670


namespace NUMINAMATH_CALUDE_little_john_money_distribution_l2086_208633

/-- Calculates the amount Little John gave to each of his two friends -/
def money_given_to_each_friend (initial_amount : ℚ) (spent_on_sweets : ℚ) (amount_left : ℚ) : ℚ :=
  (initial_amount - spent_on_sweets - amount_left) / 2

/-- Proves that Little John gave $2.20 to each of his two friends -/
theorem little_john_money_distribution :
  money_given_to_each_friend 10.50 2.25 3.85 = 2.20 := by
  sorry

#eval money_given_to_each_friend 10.50 2.25 3.85

end NUMINAMATH_CALUDE_little_john_money_distribution_l2086_208633


namespace NUMINAMATH_CALUDE_square_sum_from_product_and_sum_l2086_208605

theorem square_sum_from_product_and_sum (x y : ℝ) 
  (h1 : x * y = 12) 
  (h2 : x + y = 10) : 
  x^2 + y^2 = 76 := by
sorry

end NUMINAMATH_CALUDE_square_sum_from_product_and_sum_l2086_208605


namespace NUMINAMATH_CALUDE_pet_insurance_cost_l2086_208640

/-- Calculates the monthly cost of pet insurance given the surgery cost, insurance duration,
    coverage percentage, and total savings. -/
def monthly_insurance_cost (surgery_cost : ℚ) (insurance_duration : ℕ) 
    (coverage_percent : ℚ) (total_savings : ℚ) : ℚ :=
  let insurance_payment := surgery_cost * coverage_percent
  let total_insurance_cost := insurance_payment - total_savings
  total_insurance_cost / insurance_duration

/-- Theorem stating that the monthly insurance cost is $20 given the specified conditions. -/
theorem pet_insurance_cost :
  monthly_insurance_cost 5000 24 (4/5) 3520 = 20 := by
  sorry

end NUMINAMATH_CALUDE_pet_insurance_cost_l2086_208640


namespace NUMINAMATH_CALUDE_triangle_inequalities_l2086_208669

theorem triangle_inequalities (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a*(b-c)^2 + b*(c-a)^2 + c*(a-b)^2 + 4*a*b*c > a^3 + b^3 + c^3) ∧
  (2*a^2*b^2 + 2*b^2*c^2 + 2*c^2*a^2 > a^4 + b^4 + c^4) ∧
  (2*a*b + 2*b*c + 2*c*a > a^2 + b^2 + c^2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequalities_l2086_208669


namespace NUMINAMATH_CALUDE_third_layer_sugar_l2086_208696

def sugar_for_cake (smallest_layer sugar_second_layer sugar_third_layer : ℕ) : Prop :=
  (sugar_second_layer = 2 * smallest_layer) ∧ 
  (sugar_third_layer = 3 * sugar_second_layer)

theorem third_layer_sugar : ∀ (smallest_layer sugar_second_layer sugar_third_layer : ℕ),
  smallest_layer = 2 →
  sugar_for_cake smallest_layer sugar_second_layer sugar_third_layer →
  sugar_third_layer = 12 := by
  sorry

end NUMINAMATH_CALUDE_third_layer_sugar_l2086_208696


namespace NUMINAMATH_CALUDE_mary_remaining_sheep_l2086_208625

/-- Calculates the number of sheep Mary has left after distributing to her relatives --/
def remaining_sheep (initial : ℕ) : ℕ :=
  let after_sister := initial - (initial / 4)
  let after_brother := after_sister - (after_sister / 3)
  after_brother - (after_brother / 6)

/-- Theorem stating that Mary will have 500 sheep remaining --/
theorem mary_remaining_sheep :
  remaining_sheep 1200 = 500 := by
  sorry

end NUMINAMATH_CALUDE_mary_remaining_sheep_l2086_208625


namespace NUMINAMATH_CALUDE_quadratic_is_perfect_square_l2086_208678

theorem quadratic_is_perfect_square (c : ℝ) : 
  (∃ a b : ℝ, ∀ x : ℝ, 9*x^2 - 21*x + c = (a*x + b)^2) → c = 12.25 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_is_perfect_square_l2086_208678


namespace NUMINAMATH_CALUDE_num_possible_strings_l2086_208603

/-- Represents the allowed moves in the string transformation game -/
inductive Move
| HM_to_MH
| MT_to_TM
| TH_to_HT

/-- The initial string in the game -/
def initial_string : String := "HHMMMMTT"

/-- The number of H's in the initial string -/
def num_H : Nat := 2

/-- The number of M's in the initial string -/
def num_M : Nat := 4

/-- The number of T's in the initial string -/
def num_T : Nat := 2

/-- The total length of the string -/
def total_length : Nat := num_H + num_M + num_T

/-- Theorem stating that the number of possible strings after zero or more moves
    is equal to the number of ways to choose num_M positions out of total_length positions -/
theorem num_possible_strings :
  (Nat.choose total_length num_M) = 70 := by sorry

end NUMINAMATH_CALUDE_num_possible_strings_l2086_208603


namespace NUMINAMATH_CALUDE_solve_for_a_l2086_208680

theorem solve_for_a : ∃ a : ℝ, (3 + 2 * a = -1) ∧ (a = -2) := by sorry

end NUMINAMATH_CALUDE_solve_for_a_l2086_208680


namespace NUMINAMATH_CALUDE_double_price_increase_l2086_208679

theorem double_price_increase (original_price : ℝ) (h : original_price > 0) :
  (original_price * (1 + 0.06) * (1 + 0.06)) = (original_price * (1 + 0.1236)) := by
  sorry

end NUMINAMATH_CALUDE_double_price_increase_l2086_208679


namespace NUMINAMATH_CALUDE_pole_reconfiguration_l2086_208675

/-- Represents the configuration of electric poles on a road --/
structure RoadConfig where
  length : ℕ
  original_spacing : ℕ
  new_spacing : ℕ

/-- Calculates the number of holes needed for a given spacing --/
def holes_needed (config : RoadConfig) (spacing : ℕ) : ℕ :=
  config.length / spacing + 1

/-- Calculates the number of common holes between two spacings --/
def common_holes (config : RoadConfig) : ℕ :=
  config.length / (Nat.lcm config.original_spacing config.new_spacing) + 1

/-- The main theorem about the number of new holes and abandoned holes --/
theorem pole_reconfiguration (config : RoadConfig) 
  (h_length : config.length = 3000)
  (h_original : config.original_spacing = 50)
  (h_new : config.new_spacing = 60) :
  (holes_needed config config.new_spacing - common_holes config = 40) ∧
  (holes_needed config config.original_spacing - common_holes config = 50) := by
  sorry


end NUMINAMATH_CALUDE_pole_reconfiguration_l2086_208675


namespace NUMINAMATH_CALUDE_identify_coefficients_l2086_208699

-- Define the coefficients of a quadratic equation ax^2 + bx + c = 0
structure QuadraticCoefficients where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define our specific quadratic equation 2x^2 - x - 5 = 0
def our_quadratic : QuadraticCoefficients := ⟨2, -1, -5⟩

-- Theorem to prove
theorem identify_coefficients :
  our_quadratic.a = 2 ∧ our_quadratic.b = -1 := by
  sorry

end NUMINAMATH_CALUDE_identify_coefficients_l2086_208699


namespace NUMINAMATH_CALUDE_egg_weight_probability_l2086_208619

theorem egg_weight_probability (p_less_than_30 p_30_to_40 : ℝ) 
  (h1 : p_less_than_30 = 0.30)
  (h2 : p_30_to_40 = 0.50) :
  1 - p_less_than_30 = 0.70 :=
by sorry

end NUMINAMATH_CALUDE_egg_weight_probability_l2086_208619


namespace NUMINAMATH_CALUDE_power_sum_inequality_l2086_208604

theorem power_sum_inequality (a b c : ℝ) (n : ℕ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) (hn : 0 < n) :
  a^n + b^n + c^n ≥ a * b^(n-1) + b * c^(n-1) + c * a^(n-1) := by
  sorry

end NUMINAMATH_CALUDE_power_sum_inequality_l2086_208604


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l2086_208683

theorem cubic_equation_solution (p : ℝ) (a b c : ℝ) :
  (∀ x : ℝ, x^3 + p*x^2 + 3*x - 10 = 0 ↔ (x = a ∨ x = b ∨ x = c)) →
  c - b = b - a →
  b - a > 0 →
  a = -1 ∧ b = -1 ∧ c = -1 ∧ p = 3 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l2086_208683


namespace NUMINAMATH_CALUDE_complex_equation_sum_l2086_208668

/-- Given that (1+i)(x-yi) = 2, where x and y are real numbers and i is the imaginary unit, prove that x + y = 2 -/
theorem complex_equation_sum (x y : ℝ) : (Complex.I + 1) * (x - y * Complex.I) = 2 → x + y = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l2086_208668


namespace NUMINAMATH_CALUDE_modulus_of_Z_l2086_208610

/-- The modulus of the complex number Z = 1/(1+i) + i^3 is equal to √10/2 -/
theorem modulus_of_Z : Complex.abs (1 / (1 + Complex.I) + Complex.I^3) = Real.sqrt 10 / 2 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_Z_l2086_208610


namespace NUMINAMATH_CALUDE_trees_in_row_l2086_208648

/-- Given a plot of trees with the following properties:
  1. Trees are planted in rows of 4
  2. Each tree gives 5 apples
  3. Each apple is sold for $0.5
  4. Total revenue is $30
  Prove that the number of trees in one row is 4. -/
theorem trees_in_row (trees_per_row : ℕ) (apples_per_tree : ℕ) (price_per_apple : ℚ) (total_revenue : ℚ)
  (h1 : trees_per_row = 4)
  (h2 : apples_per_tree = 5)
  (h3 : price_per_apple = 1/2)
  (h4 : total_revenue = 30) :
  trees_per_row = 4 := by sorry

end NUMINAMATH_CALUDE_trees_in_row_l2086_208648


namespace NUMINAMATH_CALUDE_rain_probability_l2086_208692

theorem rain_probability (p_friday p_saturday p_sunday : ℝ) 
  (h_friday : p_friday = 0.4)
  (h_saturday : p_saturday = 0.5)
  (h_sunday : p_sunday = 0.3)
  (h_independent : True) -- Assumption of independence
  : p_friday * p_saturday * p_sunday = 0.06 := by
  sorry

end NUMINAMATH_CALUDE_rain_probability_l2086_208692


namespace NUMINAMATH_CALUDE_non_intersecting_probability_is_two_thirds_l2086_208681

/-- Two persons start from opposite corners of a rectangular grid and can only move up or right one step at a time. -/
structure GridWalk where
  m : ℕ  -- number of rows
  n : ℕ  -- number of columns

/-- The probability that the routes of two persons do not intersect -/
def non_intersecting_probability (g : GridWalk) : ℚ :=
  2/3

/-- Theorem stating that the probability of non-intersecting routes is 2/3 -/
theorem non_intersecting_probability_is_two_thirds (g : GridWalk) :
  non_intersecting_probability g = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_non_intersecting_probability_is_two_thirds_l2086_208681


namespace NUMINAMATH_CALUDE_valid_arrangements_count_l2086_208664

/-- Represents a soccer team with 11 players -/
def SoccerTeam := Fin 11

/-- The number of ways to arrange players from two soccer teams in a line
    such that no two adjacent players are from the same team -/
def valid_arrangements : ℕ :=
  2 * (Nat.factorial 11) ^ 2

/-- Theorem stating that the number of valid arrangements is correct -/
theorem valid_arrangements_count :
  valid_arrangements = 2 * (Nat.factorial 11) ^ 2 := by sorry

end NUMINAMATH_CALUDE_valid_arrangements_count_l2086_208664


namespace NUMINAMATH_CALUDE_second_number_value_l2086_208687

theorem second_number_value (x y z : ℚ) 
  (sum_eq : x + y + z = 120)
  (ratio_xy : x / y = 3 / 4)
  (ratio_yz : y / z = 7 / 9) :
  y = 672 / 17 := by
sorry

end NUMINAMATH_CALUDE_second_number_value_l2086_208687


namespace NUMINAMATH_CALUDE_novel_reading_distribution_l2086_208613

/-- Represents the reading assignment for three friends -/
structure ReadingAssignment where
  total_pages : ℕ
  alice_speed : ℕ
  bob_speed : ℕ
  chandra_speed : ℕ
  alice_pages : ℕ
  bob_pages : ℕ
  chandra_pages : ℕ

/-- Theorem stating the correct distribution of pages for the given conditions -/
theorem novel_reading_distribution (assignment : ReadingAssignment) :
  assignment.total_pages = 912 ∧
  assignment.alice_speed = 40 ∧
  assignment.bob_speed = 60 ∧
  assignment.chandra_speed = 48 ∧
  assignment.chandra_pages = 420 →
  assignment.alice_pages = 295 ∧
  assignment.bob_pages = 197 ∧
  assignment.alice_pages + assignment.bob_pages + assignment.chandra_pages = assignment.total_pages :=
by sorry

end NUMINAMATH_CALUDE_novel_reading_distribution_l2086_208613


namespace NUMINAMATH_CALUDE_smallest_b_for_factorization_l2086_208677

/-- 
Given a quadratic polynomial x^2 + bx + 3024, this theorem states that
111 is the smallest positive integer b for which the polynomial factors
into a product of two binomials with integer coefficients.
-/
theorem smallest_b_for_factorization : 
  ∃ (b : ℕ), b > 0 ∧ 
  (∀ (x : ℤ), ∃ (r s : ℤ), x^2 + b*x + 3024 = (x + r) * (x + s)) ∧
  (∀ (b' : ℕ), 0 < b' ∧ b' < b → 
    ¬(∀ (x : ℤ), ∃ (r s : ℤ), x^2 + b'*x + 3024 = (x + r) * (x + s))) ∧
  b = 111 :=
by sorry

end NUMINAMATH_CALUDE_smallest_b_for_factorization_l2086_208677


namespace NUMINAMATH_CALUDE_johnson_volunteers_count_l2086_208666

def total_volunteers (math_classes : ℕ) (students_per_class : ℕ) (teacher_volunteers : ℕ) (additional_needed : ℕ) : ℕ :=
  math_classes * students_per_class + teacher_volunteers + additional_needed

theorem johnson_volunteers_count :
  total_volunteers 6 5 13 7 = 50 := by
  sorry

end NUMINAMATH_CALUDE_johnson_volunteers_count_l2086_208666


namespace NUMINAMATH_CALUDE_stating_constant_sum_of_products_l2086_208697

/-- 
Represents the sum of all products of pile sizes during the division process
for n balls.
-/
def f (n : ℕ) : ℕ := sorry

/-- 
Theorem stating that the sum of all products of pile sizes during the division
process is constant for any division strategy.
-/
theorem constant_sum_of_products (n : ℕ) (h : n > 0) :
  ∀ (strategy1 strategy2 : ℕ → ℕ × ℕ),
  (∀ k, k ≤ n → (strategy1 k).1 + (strategy1 k).2 = k) →
  (∀ k, k ≤ n → (strategy2 k).1 + (strategy2 k).2 = k) →
  f n = f n :=
by sorry

/--
Lemma showing that f(n) equals n(n-1)/2 for all positive integers n.
This represents the insight from the solution, but is not directly
assumed from the problem statement.
-/
lemma f_equals_combinations (n : ℕ) (h : n > 0) :
  f n = n * (n - 1) / 2 :=
by sorry

end NUMINAMATH_CALUDE_stating_constant_sum_of_products_l2086_208697


namespace NUMINAMATH_CALUDE_at_most_two_greater_than_one_l2086_208650

theorem at_most_two_greater_than_one (a b c : ℝ) (h : a * b * c = 1) :
  ¬(2 * a - 1 / b > 1 ∧ 2 * b - 1 / c > 1 ∧ 2 * c - 1 / a > 1) := by
  sorry

end NUMINAMATH_CALUDE_at_most_two_greater_than_one_l2086_208650


namespace NUMINAMATH_CALUDE_inverse_theorem_not_exists_l2086_208652

-- Define a triangle
structure Triangle :=
  (a b c : ℝ)  -- side lengths
  (α β γ : ℝ)  -- angles

-- Define congruence for triangles
def isCongruent (t1 t2 : Triangle) : Prop :=
  t1.a = t2.a ∧ t1.b = t2.b ∧ t1.c = t2.c

-- Define equality of corresponding angles
def hasEqualAngles (t1 t2 : Triangle) : Prop :=
  t1.α = t2.α ∧ t1.β = t2.β ∧ t1.γ = t2.γ

-- Theorem statement
theorem inverse_theorem_not_exists :
  ¬(∀ t1 t2 : Triangle, hasEqualAngles t1 t2 → isCongruent t1 t2) :=
sorry

end NUMINAMATH_CALUDE_inverse_theorem_not_exists_l2086_208652


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l2086_208674

theorem cubic_equation_solution :
  ∀ x : ℝ, x^3 - 4*x = 0 ↔ x = 0 ∨ x = -2 ∨ x = 2 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l2086_208674


namespace NUMINAMATH_CALUDE_sandy_age_l2086_208600

theorem sandy_age (sandy_age molly_age : ℕ) : 
  molly_age = sandy_age + 12 →
  sandy_age * 9 = molly_age * 7 →
  sandy_age = 42 := by
sorry

end NUMINAMATH_CALUDE_sandy_age_l2086_208600


namespace NUMINAMATH_CALUDE_lindseys_remaining_money_l2086_208685

/-- Calculates Lindsey's remaining money after saving and spending --/
theorem lindseys_remaining_money (september : ℝ) (october : ℝ) (november : ℝ) 
  (h_sep : september = 50)
  (h_oct : october = 37)
  (h_nov : november = 11)
  (h_dec : december = november * 1.1)
  (h_mom_bonus : total_savings > 75 → mom_bonus = total_savings * 0.2)
  (h_spending : spending = (total_savings + mom_bonus) * 0.75)
  : remaining_money = 33.03 :=
by
  sorry

where
  december : ℝ := november * 1.1
  total_savings : ℝ := september + october + november + december
  mom_bonus : ℝ := if total_savings > 75 then total_savings * 0.2 else 0
  spending : ℝ := (total_savings + mom_bonus) * 0.75
  remaining_money : ℝ := total_savings + mom_bonus - spending

end NUMINAMATH_CALUDE_lindseys_remaining_money_l2086_208685


namespace NUMINAMATH_CALUDE_fifteen_factorial_digit_sum_l2086_208644

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem fifteen_factorial_digit_sum :
  ∃ (H M T : ℕ),
    H < 10 ∧ M < 10 ∧ T < 10 ∧
    factorial 15 = 1307674 * 10^6 + H * 10^5 + M * 10^3 + 776 * 10^2 + T * 10 + 80 ∧
    H + M + T = 17 := by
  sorry

end NUMINAMATH_CALUDE_fifteen_factorial_digit_sum_l2086_208644


namespace NUMINAMATH_CALUDE_negation_of_all_teachers_generous_l2086_208689

-- Define the universe of discourse
variable (U : Type)

-- Define predicates for being a teacher and being generous
variable (teacher : U → Prop)
variable (generous : U → Prop)

-- State the theorem
theorem negation_of_all_teachers_generous :
  (¬ ∀ x, teacher x → generous x) ↔ (∃ x, teacher x ∧ ¬ generous x) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_all_teachers_generous_l2086_208689


namespace NUMINAMATH_CALUDE_angle_relation_in_triangle_l2086_208647

/-- Given a triangle XYZ with an interior point E, where a, b, c, p are the measures of angles
    around E in degrees, and t is the exterior angle at vertex Y, prove that p = 180° - a - b + t. -/
theorem angle_relation_in_triangle (a b c p t : ℝ) : 
  (a + b + c + p = 360) →  -- Sum of angles around interior point E
  (t = 180 - c) →          -- Exterior angle relation
  (p = 180 - a - b + t) :=
by sorry

end NUMINAMATH_CALUDE_angle_relation_in_triangle_l2086_208647


namespace NUMINAMATH_CALUDE_quadratic_root_coefficients_l2086_208627

theorem quadratic_root_coefficients (b c : ℝ) : 
  (Complex.I : ℂ) ^ 2 = -1 →
  (1 - Complex.I * Real.sqrt 2) ^ 2 + b * (1 - Complex.I * Real.sqrt 2) + c = 0 →
  b = -2 ∧ c = 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_coefficients_l2086_208627


namespace NUMINAMATH_CALUDE_sum_of_numbers_with_lcm_and_ratio_l2086_208676

theorem sum_of_numbers_with_lcm_and_ratio (a b : ℕ+) : 
  Nat.lcm a b = 42 → 
  a * 3 = b * 2 → 
  (a:ℝ) + (b:ℝ) = 70 := by
sorry

end NUMINAMATH_CALUDE_sum_of_numbers_with_lcm_and_ratio_l2086_208676


namespace NUMINAMATH_CALUDE_regular_time_limit_proof_l2086_208635

/-- Represents the regular time limit in hours -/
def regular_time_limit : ℕ := 40

/-- Regular pay rate in dollars per hour -/
def regular_pay_rate : ℕ := 3

/-- Overtime pay rate in dollars per hour -/
def overtime_pay_rate : ℕ := 2 * regular_pay_rate

/-- Total pay received in dollars -/
def total_pay : ℕ := 192

/-- Overtime hours worked -/
def overtime_hours : ℕ := 12

theorem regular_time_limit_proof :
  regular_time_limit * regular_pay_rate + overtime_hours * overtime_pay_rate = total_pay :=
by sorry

end NUMINAMATH_CALUDE_regular_time_limit_proof_l2086_208635


namespace NUMINAMATH_CALUDE_maria_carrots_l2086_208639

def total_carrots (initial : ℕ) (thrown_out : ℕ) (new_picked : ℕ) : ℕ :=
  (initial - thrown_out) + new_picked

theorem maria_carrots (initial thrown_out new_picked : ℕ) 
  (h1 : initial ≥ thrown_out) : 
  total_carrots initial thrown_out new_picked = initial - thrown_out + new_picked :=
by
  sorry

end NUMINAMATH_CALUDE_maria_carrots_l2086_208639


namespace NUMINAMATH_CALUDE_f_at_2_l2086_208615

-- Define the polynomial f(x)
def f (x : ℝ) : ℝ := 5*x^5 + 4*x^4 + 3*x^3 + 2*x^2 + x + 1

-- Theorem statement
theorem f_at_2 : f 2 = 259 := by
  sorry

end NUMINAMATH_CALUDE_f_at_2_l2086_208615


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l2086_208695

/-- Given that the solution set of ax^2 + bx + 1 > 0 is (-1/2, 1/3), prove that a - b = -5 -/
theorem quadratic_inequality_solution (a b : ℝ) : 
  (∀ x, ax^2 + b*x + 1 > 0 ↔ -1/2 < x ∧ x < 1/3) → a - b = -5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l2086_208695


namespace NUMINAMATH_CALUDE_function_range_theorem_l2086_208606

open Real

theorem function_range_theorem (a : ℝ) (m n p : ℝ) : 
  let f := fun (x : ℝ) => -x^3 + 3*x + a
  (m ≠ n ∧ n ≠ p ∧ m ≠ p) →
  (f m = 2022 ∧ f n = 2022 ∧ f p = 2022) →
  (2020 < a ∧ a < 2024) := by
sorry

end NUMINAMATH_CALUDE_function_range_theorem_l2086_208606


namespace NUMINAMATH_CALUDE_arithmetic_sequence_constant_ratio_l2086_208657

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ (a₁ d : ℝ), ∀ n, a n = a₁ + (n - 1) * d

/-- The ratio of a_n to a_(2n) is constant -/
def ConstantRatio (a : ℕ → ℝ) : Prop :=
  ∃ c, ∀ n, a n ≠ 0 → a (2*n) ≠ 0 → a n / a (2*n) = c

theorem arithmetic_sequence_constant_ratio (a : ℕ → ℝ) 
    (h1 : ArithmeticSequence a) (h2 : ConstantRatio a) :
    ∃ c, (c = 1 ∨ c = 1/2) ∧ ∀ n, a n ≠ 0 → a (2*n) ≠ 0 → a n / a (2*n) = c :=
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_constant_ratio_l2086_208657


namespace NUMINAMATH_CALUDE_triangle_problem_l2086_208690

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  -- Triangle ABC with sides a, b, c opposite to angles A, B, C
  a > 0 ∧ b > 0 ∧ c > 0 →
  -- b(cos A - 2cos C) = (2c - a)cos B
  b * (Real.cos A - 2 * Real.cos C) = (2 * c - a) * Real.cos B →
  -- Part I: Prove c/a = 2
  c / a = 2 ∧
  -- Part II: If cos B = 1/4 and perimeter = 5, prove b = 2
  (Real.cos B = 1/4 ∧ a + b + c = 5 → b = 2) := by
sorry

end NUMINAMATH_CALUDE_triangle_problem_l2086_208690


namespace NUMINAMATH_CALUDE_sochi_puzzle_solution_l2086_208638

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Represents a four-digit number -/
structure FourDigitNumber where
  thousands : Digit
  hundreds : Digit
  tens : Digit
  ones : Digit

/-- Convert a FourDigitNumber to a natural number -/
def FourDigitNumber.toNat (n : FourDigitNumber) : Nat :=
  1000 * n.thousands.val + 100 * n.hundreds.val + 10 * n.tens.val + n.ones.val

/-- Check if all digits in a FourDigitNumber are unique -/
def FourDigitNumber.uniqueDigits (n : FourDigitNumber) : Prop :=
  n.thousands ≠ n.hundreds ∧ n.thousands ≠ n.tens ∧ n.thousands ≠ n.ones ∧
  n.hundreds ≠ n.tens ∧ n.hundreds ≠ n.ones ∧
  n.tens ≠ n.ones

theorem sochi_puzzle_solution :
  ∃ (year sochi : FourDigitNumber),
    year.uniqueDigits ∧
    sochi.uniqueDigits ∧
    2014 + year.toNat = sochi.toNat :=
  sorry

end NUMINAMATH_CALUDE_sochi_puzzle_solution_l2086_208638


namespace NUMINAMATH_CALUDE_ink_cost_per_ml_l2086_208659

/-- Proves that the cost of ink per milliliter is 50 cents given the specified conditions -/
theorem ink_cost_per_ml (num_classes : ℕ) (boards_per_class : ℕ) (ink_per_board : ℕ) (total_cost : ℕ) : 
  num_classes = 5 → 
  boards_per_class = 2 → 
  ink_per_board = 20 → 
  total_cost = 100 → 
  (total_cost * 100) / (num_classes * boards_per_class * ink_per_board) = 50 := by
  sorry

#check ink_cost_per_ml

end NUMINAMATH_CALUDE_ink_cost_per_ml_l2086_208659


namespace NUMINAMATH_CALUDE_find_divisor_l2086_208645

theorem find_divisor (dividend quotient remainder : ℕ) 
  (h1 : dividend = 15698)
  (h2 : quotient = 89)
  (h3 : remainder = 14)
  (h4 : dividend = quotient * 176 + remainder) :
  176 = dividend / quotient :=
by sorry

end NUMINAMATH_CALUDE_find_divisor_l2086_208645


namespace NUMINAMATH_CALUDE_cube_volume_ratio_l2086_208662

theorem cube_volume_ratio : 
  let cube_volume (edge : ℚ) : ℚ := edge ^ 3
  let cube1_edge : ℚ := 4
  let cube2_edge : ℚ := 10
  (cube_volume cube1_edge) / (cube_volume cube2_edge) = 8 / 125 := by
sorry

end NUMINAMATH_CALUDE_cube_volume_ratio_l2086_208662


namespace NUMINAMATH_CALUDE_dividend_calculation_l2086_208630

theorem dividend_calculation (divisor quotient remainder : ℕ) 
  (h1 : divisor = 17)
  (h2 : quotient = 9)
  (h3 : remainder = 8) :
  divisor * quotient + remainder = 161 := by
sorry

end NUMINAMATH_CALUDE_dividend_calculation_l2086_208630


namespace NUMINAMATH_CALUDE_problem_solution_l2086_208616

def f (x : ℝ) : ℝ := |x| - |2*x - 1|

def M : Set ℝ := {x | f x > -1}

theorem problem_solution :
  (M = {x : ℝ | 0 < x ∧ x < 2}) ∧
  (∀ a ∈ M,
    (0 < a ∧ a < 1 → a^2 - a + 1 < 1/a) ∧
    (a = 1 → a^2 - a + 1 = 1/a) ∧
    (1 < a ∧ a < 2 → a^2 - a + 1 > 1/a)) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l2086_208616


namespace NUMINAMATH_CALUDE_not_divisible_by_121_l2086_208653

theorem not_divisible_by_121 (n : ℤ) : ¬(121 ∣ (n^2 + 3*n + 5)) := by
  sorry

end NUMINAMATH_CALUDE_not_divisible_by_121_l2086_208653


namespace NUMINAMATH_CALUDE_cross_country_winning_scores_l2086_208694

/-- The number of teams in the cross-country meet -/
def num_teams : ℕ := 2

/-- The number of runners per team -/
def runners_per_team : ℕ := 6

/-- The total number of runners -/
def total_runners : ℕ := num_teams * runners_per_team

/-- The sum of positions from 1 to n -/
def sum_positions (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The total sum of all positions -/
def total_sum : ℕ := sum_positions total_runners

/-- A winning score is less than half of the total sum -/
def is_winning_score (score : ℕ) : Prop := score < total_sum / 2

/-- The minimum possible score for a team -/
def min_score : ℕ := sum_positions runners_per_team

/-- The maximum possible winning score -/
def max_winning_score : ℕ := total_sum / 2 - 1

/-- The number of different possible winning scores -/
def num_winning_scores : ℕ := max_winning_score - min_score + 1

theorem cross_country_winning_scores :
  num_winning_scores = 18 := by sorry

end NUMINAMATH_CALUDE_cross_country_winning_scores_l2086_208694


namespace NUMINAMATH_CALUDE_big_bonsai_cost_l2086_208643

/-- Represents the cost of a small bonsai in dollars -/
def small_bonsai_cost : ℕ := 30

/-- Represents the number of small bonsai sold -/
def small_bonsai_sold : ℕ := 3

/-- Represents the number of big bonsai sold -/
def big_bonsai_sold : ℕ := 5

/-- Represents the total earnings in dollars -/
def total_earnings : ℕ := 190

/-- Proves that the cost of a big bonsai is $20 -/
theorem big_bonsai_cost : 
  ∃ (big_bonsai_cost : ℕ), 
    small_bonsai_cost * small_bonsai_sold + big_bonsai_cost * big_bonsai_sold = total_earnings ∧ 
    big_bonsai_cost = 20 := by
  sorry

end NUMINAMATH_CALUDE_big_bonsai_cost_l2086_208643


namespace NUMINAMATH_CALUDE_sugar_per_cookie_l2086_208654

theorem sugar_per_cookie (initial_cookies : ℕ) (initial_sugar_per_cookie : ℚ) 
  (new_cookies : ℕ) (total_sugar : ℚ) :
  initial_cookies = 50 →
  initial_sugar_per_cookie = 1 / 10 →
  new_cookies = 25 →
  total_sugar = initial_cookies * initial_sugar_per_cookie →
  total_sugar / new_cookies = 1 / 5 := by
sorry

end NUMINAMATH_CALUDE_sugar_per_cookie_l2086_208654


namespace NUMINAMATH_CALUDE_first_consecutive_shot_probability_value_l2086_208629

/-- The probability of making a shot -/
def shot_probability : ℚ := 2/3

/-- The number of attempts before the first consecutive shot -/
def attempts : ℕ := 6

/-- The probability of making the first consecutive shot on the 7th attempt -/
def first_consecutive_shot_probability : ℚ :=
  (1 - shot_probability)^attempts * shot_probability^2

theorem first_consecutive_shot_probability_value :
  first_consecutive_shot_probability = 8/729 := by
  sorry

end NUMINAMATH_CALUDE_first_consecutive_shot_probability_value_l2086_208629


namespace NUMINAMATH_CALUDE_small_cube_edge_length_l2086_208611

/-- Given a cube with volume 1000 cm³, if 8 small cubes of equal size are cut off from its corners
    such that the remaining volume is 488 cm³, then the edge length of each small cube is 4 cm. -/
theorem small_cube_edge_length (x : ℝ) : 
  (1000 : ℝ) - 8 * x^3 = 488 → x = 4 := by sorry

end NUMINAMATH_CALUDE_small_cube_edge_length_l2086_208611


namespace NUMINAMATH_CALUDE_number_equation_solution_l2086_208621

theorem number_equation_solution : 
  ∃ x : ℝ, (10 * x = 2 * x - 36) ∧ (x = -4.5) := by
  sorry

end NUMINAMATH_CALUDE_number_equation_solution_l2086_208621


namespace NUMINAMATH_CALUDE_smallest_constant_for_inequality_l2086_208665

/-- The smallest possible real constant C such that 
    |x^3 + y^3 + z^3 + 1| ≤ C|x^5 + y^5 + z^5 + 1| 
    holds for all real x, y, z satisfying x + y + z = -1 -/
theorem smallest_constant_for_inequality :
  ∃ (C : ℝ), C = 1539 / 1449 ∧
  (∀ (x y z : ℝ), x + y + z = -1 →
    |x^3 + y^3 + z^3 + 1| ≤ C * |x^5 + y^5 + z^5 + 1|) ∧
  (∀ (C' : ℝ), C' < C →
    ∃ (x y z : ℝ), x + y + z = -1 ∧
      |x^3 + y^3 + z^3 + 1| > C' * |x^5 + y^5 + z^5 + 1|) :=
by sorry

end NUMINAMATH_CALUDE_smallest_constant_for_inequality_l2086_208665


namespace NUMINAMATH_CALUDE_shop_monthly_rent_l2086_208663

/-- The monthly rent of a rectangular shop given its dimensions and annual rent per square foot -/
def monthly_rent (length width annual_rent_per_sqft : ℕ) : ℕ :=
  length * width * annual_rent_per_sqft / 12

/-- Proof that the monthly rent of a shop with given dimensions is 3600 -/
theorem shop_monthly_rent :
  monthly_rent 18 20 120 = 3600 := by
  sorry

end NUMINAMATH_CALUDE_shop_monthly_rent_l2086_208663


namespace NUMINAMATH_CALUDE_amy_tips_calculation_l2086_208620

/-- Calculates the amount of tips earned by Amy given her hourly wage, hours worked, and total earnings. -/
theorem amy_tips_calculation (hourly_wage : ℝ) (hours_worked : ℝ) (total_earnings : ℝ) : 
  hourly_wage = 2 → hours_worked = 7 → total_earnings = 23 → 
  total_earnings - (hourly_wage * hours_worked) = 9 := by
  sorry

end NUMINAMATH_CALUDE_amy_tips_calculation_l2086_208620


namespace NUMINAMATH_CALUDE_largest_m_bound_l2086_208614

theorem largest_m_bound (x y z t : ℕ+) (h1 : x + y = z + t) (h2 : 2 * x * y = z * t) (h3 : x ≥ y) :
  ∃ (m : ℝ), m = 3 + 2 * Real.sqrt 2 ∧ 
  (∀ (m' : ℝ), (∀ (x' y' z' t' : ℕ+), 
    x' + y' = z' + t' → 2 * x' * y' = z' * t' → x' ≥ y' → 
    (x' : ℝ) / (y' : ℝ) ≥ m') → m' ≤ m) := by
  sorry

end NUMINAMATH_CALUDE_largest_m_bound_l2086_208614


namespace NUMINAMATH_CALUDE_valid_field_area_is_189_l2086_208686

/-- Represents a rectangular sports field with posts -/
structure SportsField where
  total_posts : ℕ
  post_distance : ℕ
  long_side_posts : ℕ
  short_side_posts : ℕ

/-- Checks if the field configuration is valid according to the problem conditions -/
def is_valid_field (field : SportsField) : Prop :=
  field.total_posts = 24 ∧
  field.post_distance = 3 ∧
  field.long_side_posts = 2 * field.short_side_posts ∧
  2 * (field.long_side_posts + field.short_side_posts - 2) = field.total_posts

/-- Calculates the area of the field given its configuration -/
def field_area (field : SportsField) : ℕ :=
  (field.short_side_posts - 1) * field.post_distance * 
  (field.long_side_posts - 1) * field.post_distance

/-- Theorem stating that a valid field configuration results in an area of 189 square yards -/
theorem valid_field_area_is_189 (field : SportsField) :
  is_valid_field field → field_area field = 189 := by
  sorry

#check valid_field_area_is_189

end NUMINAMATH_CALUDE_valid_field_area_is_189_l2086_208686


namespace NUMINAMATH_CALUDE_simplify_sqrt_product_l2086_208651

theorem simplify_sqrt_product : 
  Real.sqrt (3 * 5) * Real.sqrt (5^4 * 3^5) = 675 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_product_l2086_208651


namespace NUMINAMATH_CALUDE_great_dane_weight_l2086_208658

theorem great_dane_weight (chihuahua pitbull great_dane : ℕ) : 
  chihuahua + pitbull + great_dane = 439 →
  pitbull = 3 * chihuahua →
  great_dane = 10 + 3 * pitbull →
  great_dane = 307 := by
  sorry

end NUMINAMATH_CALUDE_great_dane_weight_l2086_208658


namespace NUMINAMATH_CALUDE_mary_baking_cake_l2086_208671

/-- Given a recipe that requires a certain amount of flour and an amount already added,
    calculate the remaining amount to be added. -/
def remaining_flour (required : ℕ) (added : ℕ) : ℕ :=
  required - added

/-- Prove that for a recipe requiring 7 cups of flour, with 2 cups already added,
    the remaining amount to be added is 5 cups. -/
theorem mary_baking_cake :
  remaining_flour 7 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_mary_baking_cake_l2086_208671


namespace NUMINAMATH_CALUDE_functional_equation_solution_l2086_208660

-- Define the function type
def RealFunction := ℝ → ℝ

-- State the theorem
theorem functional_equation_solution (f : RealFunction) :
  (∀ x y : ℝ, f (x - f y) = 1 - x - y) → 
  (∀ x : ℝ, f x = 1/2 - x) :=
by sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l2086_208660


namespace NUMINAMATH_CALUDE_ellipse_standard_equation_l2086_208624

/-- An ellipse with center at the origin, foci on the x-axis, and distances 3 and 1 
    from one intersection point on the x-axis to the foci has the standard equation 
    x^2/4 + y^2/3 = 1 -/
theorem ellipse_standard_equation (x y : ℝ) :
  let a : ℝ := 2  -- half the distance between the vertices
  let c : ℝ := 1  -- half the distance between the foci
  let b : ℝ := Real.sqrt 3  -- length of the semi-minor axis
  (a + c = 3) ∧ (a - c = 1) →
  (x^2 / 4 + y^2 / 3 = 1) ↔ 
  (x^2 / a^2 + y^2 / b^2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_standard_equation_l2086_208624


namespace NUMINAMATH_CALUDE_bus_journey_distance_l2086_208682

/-- Given a bus journey with two speeds, prove the distance covered at the slower speed. -/
theorem bus_journey_distance (total_distance : ℝ) (speed1 speed2 : ℝ) (total_time : ℝ) 
  (h1 : total_distance = 250)
  (h2 : speed1 = 40)
  (h3 : speed2 = 60)
  (h4 : total_time = 5.4)
  (h5 : total_distance > 0)
  (h6 : speed1 > 0)
  (h7 : speed2 > 0)
  (h8 : total_time > 0)
  (h9 : speed1 < speed2) :
  ∃ (distance1 : ℝ), 
    distance1 / speed1 + (total_distance - distance1) / speed2 = total_time ∧ 
    distance1 = 148 := by
  sorry

end NUMINAMATH_CALUDE_bus_journey_distance_l2086_208682


namespace NUMINAMATH_CALUDE_path_length_is_894_l2086_208672

/-- The length of the path with fencing and a bridge. -/
def path_length (pole_spacing : ℕ) (bridge_length : ℕ) (total_poles : ℕ) : ℕ :=
  let poles_one_side := total_poles / 2
  let intervals := poles_one_side - 1
  intervals * pole_spacing + bridge_length

/-- Theorem stating the length of the path given the conditions. -/
theorem path_length_is_894 :
  path_length 6 42 286 = 894 := by
  sorry

end NUMINAMATH_CALUDE_path_length_is_894_l2086_208672


namespace NUMINAMATH_CALUDE_bales_in_barn_l2086_208637

/-- The number of bales originally in the barn -/
def original_bales : ℕ := sorry

/-- The number of bales Keith stacked today -/
def keith_bales : ℕ := 67

/-- The total number of bales in the barn now -/
def total_bales : ℕ := 89

theorem bales_in_barn :
  original_bales + keith_bales = total_bales ∧ original_bales = 22 :=
by sorry

end NUMINAMATH_CALUDE_bales_in_barn_l2086_208637


namespace NUMINAMATH_CALUDE_searchlight_configuration_exists_l2086_208688

/-- Represents a searchlight with its position and direction --/
structure Searchlight where
  position : ℝ × ℝ
  direction : ℝ

/-- Checks if a point is within the illuminated region of a searchlight --/
def isIlluminated (s : Searchlight) (p : ℝ × ℝ) : Prop :=
  sorry

/-- Calculates the shadow length of a searchlight given a configuration --/
def shadowLength (s : Searchlight) (config : List Searchlight) : ℝ :=
  sorry

/-- Theorem: There exists a configuration of 7 searchlights where each casts a 7km shadow --/
theorem searchlight_configuration_exists : 
  ∃ (config : List Searchlight), 
    config.length = 7 ∧ 
    ∀ s ∈ config, shadowLength s config = 7 :=
  sorry

end NUMINAMATH_CALUDE_searchlight_configuration_exists_l2086_208688


namespace NUMINAMATH_CALUDE_max_value_of_f_l2086_208609

def f (x : ℝ) : ℝ := x^2 + 4*x + 1

theorem max_value_of_f :
  ∃ (m : ℝ), m = 4 ∧ ∀ x ∈ Set.Icc (-1 : ℝ) 1, f x ≤ m :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l2086_208609


namespace NUMINAMATH_CALUDE_inequality_range_l2086_208655

theorem inequality_range (x : ℝ) :
  (∀ a : ℝ, a ≥ 1 → a * x^2 + (a - 3) * x + (a - 4) > 0) →
  x < -1 ∨ x > 3 := by
sorry

end NUMINAMATH_CALUDE_inequality_range_l2086_208655


namespace NUMINAMATH_CALUDE_cloud_counting_l2086_208622

theorem cloud_counting (carson_clouds : ℕ) (brother_multiplier : ℕ) : 
  carson_clouds = 6 → 
  brother_multiplier = 3 → 
  carson_clouds + carson_clouds * brother_multiplier = 24 :=
by sorry

end NUMINAMATH_CALUDE_cloud_counting_l2086_208622


namespace NUMINAMATH_CALUDE_anna_cannot_afford_tour_l2086_208661

/-- Calculates the future value of an amount with compound interest -/
def futureValue (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

/-- Represents Anna's initial savings -/
def initialSavings : ℝ := 40000

/-- Represents the initial cost of the tour package -/
def initialCost : ℝ := 45000

/-- Represents the annual interest rate -/
def interestRate : ℝ := 0.05

/-- Represents the annual inflation rate -/
def inflationRate : ℝ := 0.05

/-- Represents the time period in years -/
def timePeriod : ℕ := 3

/-- Theorem stating that Anna cannot afford the tour package after 3 years -/
theorem anna_cannot_afford_tour : 
  futureValue initialSavings interestRate timePeriod < 
  futureValue initialCost inflationRate timePeriod := by
  sorry

end NUMINAMATH_CALUDE_anna_cannot_afford_tour_l2086_208661


namespace NUMINAMATH_CALUDE_sallys_score_l2086_208634

/-- Calculates the score for a math contest given the number of correct, incorrect, and unanswered questions. -/
def calculate_score (correct : ℕ) (incorrect : ℕ) (unanswered : ℕ) : ℚ :=
  (correct : ℚ) - 0.25 * (incorrect : ℚ)

/-- Proves that Sally's score in the math contest is 12.5 -/
theorem sallys_score :
  let correct := 15
  let incorrect := 10
  let unanswered := 5
  calculate_score correct incorrect unanswered = 12.5 := by
  sorry

#eval calculate_score 15 10 5

end NUMINAMATH_CALUDE_sallys_score_l2086_208634
