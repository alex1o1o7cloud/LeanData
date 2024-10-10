import Mathlib

namespace ab_length_is_two_l1915_191531

/-- Represents a point on a line --/
structure Point where
  position : ℝ

/-- Represents the distance between two points --/
def distance (p q : Point) : ℝ := abs (p.position - q.position)

/-- Theorem: Given points A, B, C, D on a line in order, if AC = 5, BD = 6, and CD = 3, then AB = 2 --/
theorem ab_length_is_two 
  (A B C D : Point) 
  (order : A.position < B.position ∧ B.position < C.position ∧ C.position < D.position)
  (ac_length : distance A C = 5)
  (bd_length : distance B D = 6)
  (cd_length : distance C D = 3) :
  distance A B = 2 := by
  sorry

end ab_length_is_two_l1915_191531


namespace condition_sufficient_not_necessary_l1915_191587

theorem condition_sufficient_not_necessary :
  (∀ x : ℝ, x > 1 → (x + 2) / (x - 1) > 0) ∧
  (∃ x : ℝ, x ≤ 1 ∧ (x + 2) / (x - 1) > 0) := by
  sorry

end condition_sufficient_not_necessary_l1915_191587


namespace sufficient_not_necessary_l1915_191569

theorem sufficient_not_necessary (x y : ℝ) :
  (∀ x y, x + y ≤ 1 → x ≤ 1/2 ∨ y ≤ 1/2) ∧
  (∃ x y, (x ≤ 1/2 ∨ y ≤ 1/2) ∧ x + y > 1) :=
by sorry

end sufficient_not_necessary_l1915_191569


namespace smallest_multiple_36_45_not_11_l1915_191546

theorem smallest_multiple_36_45_not_11 : ∃ (n : ℕ), 
  n > 0 ∧ 
  36 ∣ n ∧ 
  45 ∣ n ∧ 
  ¬(11 ∣ n) ∧ 
  ∀ (m : ℕ), m > 0 ∧ 36 ∣ m ∧ 45 ∣ m ∧ ¬(11 ∣ m) → n ≤ m :=
by
  -- The proof would go here
  sorry

end smallest_multiple_36_45_not_11_l1915_191546


namespace sqrt_square_abs_l1915_191585

theorem sqrt_square_abs (x : ℝ) : Real.sqrt (x^2) = |x| := by sorry

end sqrt_square_abs_l1915_191585


namespace percentage_difference_l1915_191545

theorem percentage_difference (x y : ℝ) (h : x = 5 * y) : 
  (x - y) / x * 100 = 80 := by
sorry

end percentage_difference_l1915_191545


namespace valid_solutions_l1915_191582

-- Define digits as natural numbers from 0 to 9
def Digit : Type := { n : ℕ // n ≤ 9 }

-- Define the conditions for each case
def case1_conditions (x y z : Digit) : Prop :=
  (10 * x.val + y.val = 3 * (10 * y.val + z.val)) ∧
  (x.val + y.val = y.val + z.val + 3)

def case2_conditions (x y z : Digit) : Prop :=
  (10 * x.val + y.val = 3 * (10 * z.val + x.val)) ∧
  (x.val + y.val = x.val + z.val + 3 ∨ x.val + y.val = x.val + z.val - 3)

def case3_conditions (x y z : Digit) : Prop :=
  (10 * x.val + y.val = 3 * (10 * x.val + z.val)) ∧
  (x.val + y.val = x.val + z.val + 3 ∨ x.val + y.val = x.val + z.val - 3)

def case4_conditions (x y z : Digit) : Prop :=
  (10 * x.val + y.val = 3 * (10 * z.val + y.val)) ∧
  (x.val + y.val = z.val + y.val + 3 ∨ x.val + y.val = z.val + y.val - 3)

-- Main theorem
theorem valid_solutions :
  ∀ (a b : ℕ) (x y z : Digit),
    a > b →
    (case1_conditions x y z ∨ case2_conditions x y z ∨ case3_conditions x y z ∨ case4_conditions x y z) →
    ((a = 72 ∧ b = 24) ∨ (a = 45 ∧ b = 15)) := by
  sorry

end valid_solutions_l1915_191582


namespace fraction_equality_l1915_191555

theorem fraction_equality : (2015 : ℤ) / (2015^2 - 2016 * 2014) = 2015 := by
  sorry

end fraction_equality_l1915_191555


namespace x_value_when_y_is_half_l1915_191504

theorem x_value_when_y_is_half :
  ∀ x y : ℝ, y = 1 / (4 * x + 2) → y = 1 / 2 → x = 0 := by
sorry

end x_value_when_y_is_half_l1915_191504


namespace password_probability_l1915_191538

/-- Represents the probability of using password A in week k -/
def P (k : ℕ) : ℚ :=
  3/4 * (-1/3)^(k-1) + 1/4

/-- The problem statement -/
theorem password_probability : P 7 = 61/243 := by
  sorry

end password_probability_l1915_191538


namespace squared_sum_equals_20_75_l1915_191521

theorem squared_sum_equals_20_75 
  (a b c : ℝ) 
  (eq1 : a^2 + 3*b = 9) 
  (eq2 : b^2 + 5*c = -8) 
  (eq3 : c^2 + 7*a = -18) : 
  a^2 + b^2 + c^2 = 20.75 := by
  sorry

end squared_sum_equals_20_75_l1915_191521


namespace sin_beta_value_l1915_191517

theorem sin_beta_value (α β : Real) 
  (h : Real.sin (α - β) * Real.cos α - Real.cos (α - β) * Real.sin α = 3/5) : 
  Real.sin β = -3/5 := by
sorry

end sin_beta_value_l1915_191517


namespace angle_D_is_100_l1915_191554

-- Define the triangle DEF
structure Triangle :=
  (D E F : ℝ)

-- Define the properties of the triangle
def is_right_triangle (t : Triangle) : Prop :=
  t.D + t.E + t.F = 180

def angle_E_is_30 (t : Triangle) : Prop :=
  t.E = 30

def angle_D_twice_F (t : Triangle) : Prop :=
  t.D = 2 * t.F

-- Theorem statement
theorem angle_D_is_100 (t : Triangle) 
  (h1 : is_right_triangle t) 
  (h2 : angle_E_is_30 t) 
  (h3 : angle_D_twice_F t) : 
  t.D = 100 :=
sorry

end angle_D_is_100_l1915_191554


namespace vehicle_value_fraction_l1915_191597

theorem vehicle_value_fraction (value_this_year value_last_year : ℚ) 
  (h1 : value_this_year = 16000)
  (h2 : value_last_year = 20000) :
  value_this_year / value_last_year = 4 / 5 := by
sorry

end vehicle_value_fraction_l1915_191597


namespace carpet_square_size_l1915_191598

theorem carpet_square_size (floor_length : ℝ) (floor_width : ℝ) 
  (total_cost : ℝ) (square_cost : ℝ) :
  floor_length = 24 →
  floor_width = 64 →
  total_cost = 576 →
  square_cost = 24 →
  ∃ (square_side : ℝ),
    square_side = 8 ∧
    (floor_length * floor_width) / (square_side * square_side) * square_cost = total_cost :=
by sorry

end carpet_square_size_l1915_191598


namespace problem_statement_l1915_191583

theorem problem_statement :
  (∀ x : ℝ, x^2 - 4*x + 5 > 0) ∧
  (∃ x : ℤ, 3*x^2 - 2*x - 1 = 0) ∧
  (¬ ∃ x : ℚ, x^2 = 5) ∧
  (¬ ∀ x : ℝ, x + 1/x > 2) :=
by sorry

end problem_statement_l1915_191583


namespace student_marks_l1915_191599

theorem student_marks (M P C : ℤ) 
  (h1 : C = P + 20) 
  (h2 : (M + C) / 2 = 45) : 
  M + P = 70 := by
  sorry

end student_marks_l1915_191599


namespace roselyn_initial_books_l1915_191564

/-- The number of books Roselyn initially had -/
def initial_books : ℕ := 220

/-- The number of books Rebecca received -/
def rebecca_books : ℕ := 40

/-- The number of books Mara received -/
def mara_books : ℕ := 3 * rebecca_books

/-- The number of books Roselyn remained with -/
def remaining_books : ℕ := 60

/-- Theorem stating that the initial number of books Roselyn had is 220 -/
theorem roselyn_initial_books :
  initial_books = mara_books + rebecca_books + remaining_books :=
by sorry

end roselyn_initial_books_l1915_191564


namespace expand_expression_l1915_191512

theorem expand_expression (x y z : ℝ) : 
  (2 * x + 5) * (3 * y + 4 * z + 15) = 6 * x * y + 8 * x * z + 30 * x + 15 * y + 20 * z + 75 := by
  sorry

end expand_expression_l1915_191512


namespace constant_term_is_180_l1915_191578

/-- The binomial expansion of (√x + 2/x²)^10 has its largest coefficient in the sixth term -/
axiom largest_coeff_sixth_term : ∃ k, k = 6 ∧ ∀ j, j ≠ k → 
  Nat.choose 10 (k-1) * 2^(k-1) ≥ Nat.choose 10 (j-1) * 2^(j-1)

/-- The constant term in the expansion of (√x + 2/x²)^10 -/
def constant_term : ℕ := Nat.choose 10 2 * 2^2

theorem constant_term_is_180 : constant_term = 180 := by
  sorry

end constant_term_is_180_l1915_191578


namespace sqrt_x_minus_8_real_l1915_191586

theorem sqrt_x_minus_8_real (x : ℝ) : (∃ y : ℝ, y ^ 2 = x - 8) ↔ x ≥ 8 := by sorry

end sqrt_x_minus_8_real_l1915_191586


namespace student_congress_sample_size_l1915_191523

/-- Represents a school with classes and a student congress. -/
structure School where
  num_classes : ℕ
  students_per_class : ℕ
  students_sent_per_class : ℕ

/-- Calculates the sample size for the Student Congress. -/
def sample_size (s : School) : ℕ :=
  s.num_classes * s.students_sent_per_class

/-- Theorem: The sample size for the given school is 120. -/
theorem student_congress_sample_size :
  let s : School := {
    num_classes := 40,
    students_per_class := 50,
    students_sent_per_class := 3
  }
  sample_size s = 120 := by
  sorry


end student_congress_sample_size_l1915_191523


namespace min_value_theorem_l1915_191539

theorem min_value_theorem (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 1/a + 1/b = 1) :
  ∀ x y : ℝ, 0 < x ∧ 0 < y ∧ 1/x + 1/y = 1 → 1/(a-1) + 9/(b-1) ≤ 1/(x-1) + 9/(y-1) :=
by sorry

end min_value_theorem_l1915_191539


namespace polynomial_division_quotient_l1915_191540

theorem polynomial_division_quotient :
  let dividend : Polynomial ℚ := 8 * X^3 + 16 * X^2 - 7 * X + 4
  let divisor : Polynomial ℚ := 2 * X + 5
  let quotient : Polynomial ℚ := 4 * X^2 - 2 * X + (3/2)
  dividend = divisor * quotient + (-7/2) := by sorry

end polynomial_division_quotient_l1915_191540


namespace inequality_solution_set_l1915_191508

-- Define the inequality
def inequality (x m : ℝ) : Prop := x^2 - (2*m-1)*x + m^2 - m > 0

-- Define the solution set
def solution_set (m : ℝ) : Set ℝ := {x | x < m-1 ∨ x > m}

-- Theorem statement
theorem inequality_solution_set (m : ℝ) : 
  {x : ℝ | inequality x m} = solution_set m := by sorry

end inequality_solution_set_l1915_191508


namespace larger_number_proof_l1915_191532

theorem larger_number_proof (a b : ℕ+) : 
  (Nat.gcd a b = 25) →
  (Nat.lcm a b = 4550) →
  (13 ∣ Nat.lcm a b) →
  (14 ∣ Nat.lcm a b) →
  max a b = 350 := by
sorry

end larger_number_proof_l1915_191532


namespace combined_mixture_ratio_l1915_191547

/-- Represents a mixture of milk and water -/
structure Mixture where
  milk : ℚ
  water : ℚ

/-- Combines two mixtures -/
def combineMixtures (m1 m2 : Mixture) : Mixture :=
  { milk := m1.milk + m2.milk, water := m1.water + m2.water }

theorem combined_mixture_ratio :
  let m1 : Mixture := { milk := 4, water := 2 }
  let m2 : Mixture := { milk := 5, water := 1 }
  let combined := combineMixtures m1 m2
  combined.milk / combined.water = 3 := by
  sorry

end combined_mixture_ratio_l1915_191547


namespace value_of_a_l1915_191593

theorem value_of_a (A B : Set ℝ) (a : ℝ) : 
  A = {0, 2, a} → 
  B = {1, a^2} → 
  A ∪ B = {0, 1, 2, 4, 16} → 
  a = 4 := by
sorry

end value_of_a_l1915_191593


namespace b_age_is_ten_l1915_191553

/-- Given the ages of three people a, b, and c, prove that b is 10 years old. -/
theorem b_age_is_ten (a b c : ℕ) : 
  a = b + 2 → 
  b = 2 * c → 
  a + b + c = 27 → 
  b = 10 := by
sorry

end b_age_is_ten_l1915_191553


namespace dice_prime_probability_l1915_191530

def probability_prime : ℚ := 5 / 12

def number_of_dice : ℕ := 5

def target_prime_count : ℕ := 3

theorem dice_prime_probability :
  let p := probability_prime
  let n := number_of_dice
  let k := target_prime_count
  (n.choose k) * p^k * (1 - p)^(n - k) = 6125 / 24883 := by sorry

end dice_prime_probability_l1915_191530


namespace ping_pong_game_ratio_l1915_191556

/-- Given that Frankie and Carla played 30 games of ping pong,
    and Carla won 20 games, prove that the ratio of games
    Frankie won to games Carla won is 1:2. -/
theorem ping_pong_game_ratio :
  let total_games : ℕ := 30
  let carla_wins : ℕ := 20
  let frankie_wins : ℕ := total_games - carla_wins
  (frankie_wins : ℚ) / (carla_wins : ℚ) = 1 / 2 := by
  sorry

end ping_pong_game_ratio_l1915_191556


namespace simplify_square_roots_l1915_191552

theorem simplify_square_roots : 81^(1/2) - 144^(1/2) = -63 := by sorry

end simplify_square_roots_l1915_191552


namespace function_value_proof_l1915_191535

/-- Given a function f(x) = x^5 - ax^3 + bx - 6 where f(-2) = 10, prove that f(2) = -22 -/
theorem function_value_proof (a b : ℝ) : 
  let f := λ x : ℝ => x^5 - a*x^3 + b*x - 6
  f (-2) = 10 → f 2 = -22 := by
sorry

end function_value_proof_l1915_191535


namespace sequence_non_positive_l1915_191573

theorem sequence_non_positive (n : ℕ) (a : ℕ → ℝ) 
  (h_n : n ≥ 3)
  (h_start : a 1 = 0)
  (h_end : a n = 0)
  (h_ineq : ∀ k : ℕ, 2 ≤ k ∧ k ≤ n - 1 → a (k - 1) + a (k + 1) ≥ 2 * a k) :
  ∀ i : ℕ, 1 ≤ i ∧ i ≤ n → a i ≤ 0 := by
sorry

end sequence_non_positive_l1915_191573


namespace fraction_comparison_l1915_191533

theorem fraction_comparison : (5 / 8 : ℚ) - (1 / 16 : ℚ) > (5 / 9 : ℚ) := by
  sorry

end fraction_comparison_l1915_191533


namespace fifty_seventh_digit_of_1_13_l1915_191500

def decimal_rep_1_13 : List Nat := [0, 7, 6, 9, 2, 3]

theorem fifty_seventh_digit_of_1_13 : 
  (decimal_rep_1_13[(57 - 1) % decimal_rep_1_13.length] = 6) := by
  sorry

end fifty_seventh_digit_of_1_13_l1915_191500


namespace hyperbola_asymptote_angle_l1915_191574

/-- Given a hyperbola with equation x²/p² - y²/q² = 1 where p > q,
    if the angle between its asymptotes is 45°, then p/q = √2 - 1 -/
theorem hyperbola_asymptote_angle (p q : ℝ) (h1 : p > q) (h2 : q > 0) :
  (∃ (x y : ℝ → ℝ), ∀ t, (x t)^2 / p^2 - (y t)^2 / q^2 = 1) →
  (∃ (m : ℝ), m = q / p ∧ 
    Real.tan (45 * π / 180) = |((m - (-m)) / (1 + m * (-m)))|) →
  p / q = Real.sqrt 2 - 1 := by
  sorry

end hyperbola_asymptote_angle_l1915_191574


namespace function_equality_l1915_191581

/-- Given a function f such that f(2x) = 2 / (2 + x) for all x > 0,
    prove that 2f(x) = 8 / (4 + x) -/
theorem function_equality (f : ℝ → ℝ) 
    (h : ∀ x > 0, f (2 * x) = 2 / (2 + x)) :
  ∀ x > 0, 2 * f x = 8 / (4 + x) := by
  sorry

end function_equality_l1915_191581


namespace boyfriend_texts_l1915_191563

theorem boyfriend_texts (total : ℕ) (grocery : ℕ) : 
  total = 33 → 
  grocery + 5 * grocery + (grocery + 5 * grocery) / 10 = total → 
  grocery = 5 := by
  sorry

end boyfriend_texts_l1915_191563


namespace intersection_of_A_and_B_l1915_191567

def A : Set (ℝ × ℝ) := {p | p.2 = p.1 + 1}
def B : Set (ℝ × ℝ) := {p | p.2 = -2 * p.1 + 4}

theorem intersection_of_A_and_B : A ∩ B = {(1, 2)} := by sorry

end intersection_of_A_and_B_l1915_191567


namespace mango_profit_percentage_l1915_191566

theorem mango_profit_percentage 
  (total_crates : ℕ) 
  (total_cost : ℝ) 
  (lost_crates : ℕ) 
  (selling_price : ℝ) : 
  total_crates = 10 → 
  total_cost = 160 → 
  lost_crates = 2 → 
  selling_price = 25 → 
  ((total_crates - lost_crates) * selling_price - total_cost) / total_cost * 100 = 25 := by
  sorry

end mango_profit_percentage_l1915_191566


namespace triangle_determinant_zero_l1915_191589

theorem triangle_determinant_zero (A B C : Real) 
  (h_triangle : A + B + C = Real.pi) : 
  let M : Matrix (Fin 3) (Fin 3) Real := 
    ![![Real.cos A ^ 2, Real.tan A, 1],
      ![Real.cos B ^ 2, Real.tan B, 1],
      ![Real.cos C ^ 2, Real.tan C, 1]]
  Matrix.det M = 0 := by
sorry

end triangle_determinant_zero_l1915_191589


namespace fruit_purchase_theorem_l1915_191594

/-- Calculates the total cost of a fruit purchase with a quantity-based discount --/
def fruitPurchaseCost (lemonPrice papayaPrice mangoPrice : ℕ) 
                      (lemonQty papayaQty mangoQty : ℕ) 
                      (discountPerFruits : ℕ) 
                      (discountAmount : ℕ) : ℕ :=
  let totalFruits := lemonQty + papayaQty + mangoQty
  let totalCost := lemonPrice * lemonQty + papayaPrice * papayaQty + mangoPrice * mangoQty
  let discountCount := totalFruits / discountPerFruits
  let totalDiscount := discountCount * discountAmount
  totalCost - totalDiscount

theorem fruit_purchase_theorem : 
  fruitPurchaseCost 2 1 4 6 4 2 4 1 = 21 := by
  sorry

end fruit_purchase_theorem_l1915_191594


namespace total_harvest_l1915_191595

def tomato_harvest (day1 : ℕ) (extra_day2 : ℕ) : ℕ := 
  let day2 := day1 + extra_day2
  let day3 := 2 * day2
  day1 + day2 + day3

theorem total_harvest : tomato_harvest 120 50 = 630 := by
  sorry

end total_harvest_l1915_191595


namespace upstream_rate_calculation_l1915_191551

/-- Represents the rowing rates and current speed in kilometers per hour. -/
structure RowingScenario where
  downstream_rate : ℝ
  still_water_rate : ℝ
  current_rate : ℝ

/-- Calculates the upstream rate given a RowingScenario. -/
def upstream_rate (scenario : RowingScenario) : ℝ :=
  scenario.still_water_rate - scenario.current_rate

/-- Theorem stating that for the given scenario, the upstream rate is 10 kmph. -/
theorem upstream_rate_calculation (scenario : RowingScenario) 
  (h1 : scenario.downstream_rate = 30)
  (h2 : scenario.still_water_rate = 20)
  (h3 : scenario.current_rate = 10) :
  upstream_rate scenario = 10 := by
  sorry

#check upstream_rate_calculation

end upstream_rate_calculation_l1915_191551


namespace laborer_income_l1915_191544

/-- Represents the monthly income of a laborer -/
def monthly_income : ℝ := 69

/-- Represents the average expenditure for the first 6 months -/
def first_6_months_expenditure : ℝ := 70

/-- Represents the reduced monthly expenditure for the next 4 months -/
def next_4_months_expenditure : ℝ := 60

/-- Represents the amount saved after 10 months -/
def amount_saved : ℝ := 30

/-- Theorem stating that the monthly income is 69 given the problem conditions -/
theorem laborer_income : 
  (6 * first_6_months_expenditure > 6 * monthly_income) ∧ 
  (4 * monthly_income = 4 * next_4_months_expenditure + (6 * first_6_months_expenditure - 6 * monthly_income) + amount_saved) →
  monthly_income = 69 := by
sorry

end laborer_income_l1915_191544


namespace davids_math_marks_l1915_191503

theorem davids_math_marks (english physics chemistry biology average : ℕ) 
  (h1 : english = 91)
  (h2 : physics = 82)
  (h3 : chemistry = 67)
  (h4 : biology = 85)
  (h5 : average = 78)
  (h6 : (english + physics + chemistry + biology + mathematics) / 5 = average) :
  mathematics = 65 := by
  sorry

#check davids_math_marks

end davids_math_marks_l1915_191503


namespace age_ratio_in_two_years_l1915_191514

theorem age_ratio_in_two_years (son_age : ℕ) (man_age : ℕ) : 
  son_age = 14 →
  man_age = son_age + 16 →
  ∃ k : ℕ, (man_age + 2) = k * (son_age + 2) →
  (man_age + 2) / (son_age + 2) = 2 := by
sorry

end age_ratio_in_two_years_l1915_191514


namespace orchids_cut_l1915_191513

theorem orchids_cut (initial_roses initial_orchids final_roses final_orchids : ℕ) 
  (h1 : initial_roses = 16)
  (h2 : initial_orchids = 3)
  (h3 : final_roses = 13)
  (h4 : final_orchids = 7) :
  final_orchids - initial_orchids = 4 := by
  sorry

end orchids_cut_l1915_191513


namespace min_value_sum_squares_and_reciprocals_l1915_191580

theorem min_value_sum_squares_and_reciprocals (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  a^2 + b^2 + 1/a^2 + 1/b^2 ≥ 4 ∧
  (a^2 + b^2 + 1/a^2 + 1/b^2 = 4 ↔ a = 1 ∧ b = 1) :=
by sorry

end min_value_sum_squares_and_reciprocals_l1915_191580


namespace symmetry_line_l1915_191515

-- Define the function g
variable (g : ℝ → ℝ)

-- Define the symmetry condition
def is_symmetric (g : ℝ → ℝ) : Prop :=
  ∀ x, g x = g (4 - x)

-- Define the line of symmetry
def line_of_symmetry (g : ℝ → ℝ) (k : ℝ) : Prop :=
  ∀ x, g (k + (x - k)) = g (k - (x - k))

-- Theorem statement
theorem symmetry_line (g : ℝ → ℝ) (h : is_symmetric g) :
  line_of_symmetry g 2 :=
sorry

end symmetry_line_l1915_191515


namespace committee_selection_count_l1915_191541

theorem committee_selection_count : Nat.choose 12 7 = 792 := by
  sorry

end committee_selection_count_l1915_191541


namespace exam_pass_percentage_l1915_191525

theorem exam_pass_percentage
  (failed_hindi : ℚ)
  (failed_english : ℚ)
  (failed_both : ℚ)
  (h1 : failed_hindi = 25 / 100)
  (h2 : failed_english = 48 / 100)
  (h3 : failed_both = 27 / 100) :
  1 - (failed_hindi + failed_english - failed_both) = 54 / 100 :=
by sorry

end exam_pass_percentage_l1915_191525


namespace q_satisfies_conditions_l1915_191576

/-- A quadratic polynomial that satisfies specific conditions -/
def q (x : ℚ) : ℚ := (10/9) * x^2 + (4/9) * x + 4/9

/-- Theorem stating that q satisfies the given conditions -/
theorem q_satisfies_conditions : 
  q (-2) = 4 ∧ q 1 = 2 ∧ q 3 = 10 := by
  sorry

#eval q (-2)
#eval q 1
#eval q 3

end q_satisfies_conditions_l1915_191576


namespace clock_equivalent_square_l1915_191559

theorem clock_equivalent_square : 
  ∃ (h : ℕ), h > 10 ∧ h ≤ 12 ∧ (h - h^2) % 12 = 0 ∧ 
  ∀ (k : ℕ), k > 10 ∧ k < h → (k - k^2) % 12 ≠ 0 :=
by sorry

end clock_equivalent_square_l1915_191559


namespace root_property_l1915_191524

theorem root_property (a : ℝ) (h : 2 * a^2 - 3 * a - 5 = 0) : -4 * a^2 + 6 * a = -10 := by
  sorry

end root_property_l1915_191524


namespace chess_group_players_l1915_191561

/-- The number of players in a chess group -/
def num_players : ℕ := 8

/-- The total number of games played -/
def total_games : ℕ := 28

/-- Calculates the number of games played given the number of players -/
def games_played (n : ℕ) : ℕ := n * (n - 1) / 2

theorem chess_group_players :
  (games_played num_players = total_games) ∧ 
  (∀ m : ℕ, m ≠ num_players → games_played m ≠ total_games) :=
sorry

end chess_group_players_l1915_191561


namespace project_time_calculation_l1915_191542

/-- Calculates the remaining time for writing a report given the total time available and time spent on research and proposal. -/
def remaining_time (total_time research_time proposal_time : ℕ) : ℕ :=
  total_time - (research_time + proposal_time)

/-- Proves that given 20 hours total, 10 hours for research, and 2 hours for proposal, 
    the remaining time for writing the report is 8 hours. -/
theorem project_time_calculation :
  remaining_time 20 10 2 = 8 := by
  sorry

end project_time_calculation_l1915_191542


namespace b_reaches_a_in_120_minutes_l1915_191527

/-- Represents the walking scenario of two people A and B -/
structure WalkingScenario where
  speed_B : ℝ  -- B's speed in meters per minute
  initial_distance : ℝ  -- Initial distance between A and B in meters
  meeting_time : ℝ  -- Time when A and B meet in minutes

/-- Calculates the time for B to reach point A after A has reached point B -/
def time_for_B_to_reach_A (scenario : WalkingScenario) : ℝ :=
  -- We'll implement the calculation here
  sorry

/-- Theorem stating that given the conditions, B will take 120 minutes to reach A after A reaches B -/
theorem b_reaches_a_in_120_minutes (scenario : WalkingScenario) 
    (h1 : scenario.meeting_time = 60)
    (h2 : scenario.initial_distance = 4 * scenario.speed_B * scenario.meeting_time) : 
    time_for_B_to_reach_A scenario = 120 :=
  sorry

end b_reaches_a_in_120_minutes_l1915_191527


namespace darias_initial_savings_l1915_191550

def couch_price : ℕ := 750
def table_price : ℕ := 100
def lamp_price : ℕ := 50
def remaining_debt : ℕ := 400

def total_furniture_cost : ℕ := couch_price + table_price + lamp_price

theorem darias_initial_savings : total_furniture_cost - remaining_debt = 500 := by
  sorry

end darias_initial_savings_l1915_191550


namespace ascending_concept_chain_l1915_191571

-- Define the concept hierarchy
def IsNatural (n : ℕ) : Prop := True
def IsInteger (n : ℤ) : Prop := True
def IsRational (q : ℚ) : Prop := True
def IsReal (r : ℝ) : Prop := True
def IsNumber (x : ℝ) : Prop := True

-- Define the chain of ascending concepts
def ConceptChain : Prop :=
  ∃ (n : ℕ) (z : ℤ) (q : ℚ) (r : ℝ),
    n = 3 ∧
    IsNatural n ∧
    (↑n : ℤ) = z ∧
    IsInteger z ∧
    (↑z : ℚ) = q ∧
    IsRational q ∧
    (↑q : ℝ) = r ∧
    IsReal r ∧
    IsNumber r

-- Theorem statement
theorem ascending_concept_chain : ConceptChain :=
  sorry

end ascending_concept_chain_l1915_191571


namespace total_players_count_l1915_191502

/-- The number of players who play kabadi (including those who play both) -/
def kabadi_players : ℕ := 10

/-- The number of players who play kho kho only -/
def kho_kho_only_players : ℕ := 15

/-- The number of players who play both games -/
def both_games_players : ℕ := 5

/-- The total number of players -/
def total_players : ℕ := kabadi_players + kho_kho_only_players - both_games_players

theorem total_players_count : total_players = 20 := by sorry

end total_players_count_l1915_191502


namespace function_non_negative_iff_a_leq_four_l1915_191534

theorem function_non_negative_iff_a_leq_four (a : ℝ) :
  (∀ x : ℝ, 2^(2*x) - a * 2^x + 4 ≥ 0) ↔ a ≤ 4 := by sorry

end function_non_negative_iff_a_leq_four_l1915_191534


namespace unique_number_between_zero_and_two_l1915_191558

theorem unique_number_between_zero_and_two : 
  ∃! (n : ℕ), n ≤ 9 ∧ n > 0 ∧ n < 2 := by sorry

end unique_number_between_zero_and_two_l1915_191558


namespace horner_method_for_f_at_2_l1915_191549

def f (x : ℝ) : ℝ := 2 * x^5 + 3 * x^4 + 2 * x^3 - 4 * x + 5

theorem horner_method_for_f_at_2 : f 2 = 125 := by
  sorry

end horner_method_for_f_at_2_l1915_191549


namespace sin_2alpha_minus_cos_squared_alpha_l1915_191590

theorem sin_2alpha_minus_cos_squared_alpha (α : Real) (h : Real.tan α = 2) :
  Real.sin (2 * α) - Real.cos α ^ 2 = 3 / 5 := by
  sorry

end sin_2alpha_minus_cos_squared_alpha_l1915_191590


namespace range_of_a_l1915_191510

def p (a : ℝ) : Prop := ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 - a ≥ 0

def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2*a*x + 4 = 0

theorem range_of_a (a : ℝ) : 
  (p a ∧ ¬q a) ∨ (¬p a ∧ q a) → a ∈ Set.Ioo (-2 : ℝ) 1 ∪ Set.Ici 2 :=
sorry

end range_of_a_l1915_191510


namespace cinema_rows_l1915_191565

def base8_to_decimal (n : ℕ) : ℕ :=
  (n / 100) * 64 + ((n / 10) % 10) * 8 + (n % 10)

theorem cinema_rows :
  let total_seats : ℕ := base8_to_decimal 351
  let seats_per_row : ℕ := 3
  (total_seats / seats_per_row : ℕ) = 77 := by
sorry

end cinema_rows_l1915_191565


namespace prime_sum_theorem_l1915_191519

theorem prime_sum_theorem : ∃ (A B : ℕ), 
  0 < A ∧ 0 < B ∧
  Nat.Prime A ∧ 
  Nat.Prime B ∧ 
  Nat.Prime (A - B) ∧ 
  Nat.Prime (A - 2*B) ∧
  A + B + (A - B) + (A - 2*B) = 17 := by
sorry

end prime_sum_theorem_l1915_191519


namespace magic_square_sum_l1915_191537

/-- Represents a 3x3 magic square with given values and variables -/
structure MagicSquare where
  a : ℕ
  b : ℕ
  c : ℕ
  d : ℕ
  e : ℕ
  sum : ℕ
  row1_eq : sum = 20 + e + 18
  row2_eq : sum = 15 + c + d
  row3_eq : sum = a + 25 + b
  col1_eq : sum = 20 + 15 + a
  col2_eq : sum = e + c + 25
  col3_eq : sum = 18 + d + b
  diag1_eq : sum = 20 + c + b
  diag2_eq : sum = a + c + 18

/-- Theorem: In the given magic square, d + e = 42 -/
theorem magic_square_sum (ms : MagicSquare) : ms.d + ms.e = 42 := by
  sorry

end magic_square_sum_l1915_191537


namespace imaginary_part_of_complex_fraction_l1915_191577

/-- The imaginary part of (2+i)/i is -2 -/
theorem imaginary_part_of_complex_fraction : Complex.im ((2 : Complex) + Complex.I) / Complex.I = -2 := by
  sorry

end imaginary_part_of_complex_fraction_l1915_191577


namespace files_remaining_l1915_191506

theorem files_remaining (initial_music : ℕ) (initial_video : ℕ) (deleted : ℕ) : 
  initial_music = 4 → initial_video = 21 → deleted = 23 → 
  initial_music + initial_video - deleted = 2 := by
  sorry

end files_remaining_l1915_191506


namespace inequality_squared_l1915_191509

theorem inequality_squared (a b c : ℝ) (h : a > b) : a * c^2 ≥ b * c^2 := by
  sorry

end inequality_squared_l1915_191509


namespace geometric_sequence_ratio_l1915_191516

theorem geometric_sequence_ratio (a : ℝ) (r : ℝ) :
  (∀ n : ℕ, a * r^n = 3 * (a * r^(n+1) + a * r^(n+2))) →
  (∀ n : ℕ, a * r^n > 0) →
  r = (-1 + Real.sqrt (7/3)) / 2 := by
  sorry

end geometric_sequence_ratio_l1915_191516


namespace shirt_problem_l1915_191575

/-- Represents the problem of determining the number of shirts and minimum selling price --/
theorem shirt_problem (first_batch_cost second_batch_cost : ℕ) 
  (h1 : first_batch_cost = 13200)
  (h2 : second_batch_cost = 28800)
  (h3 : ∃ x : ℕ, x > 0 ∧ second_batch_cost / (2 * x) = first_batch_cost / x + 10)
  (h4 : ∃ y : ℕ, y > 0 ∧ 350 * y ≥ (first_batch_cost + second_batch_cost) * 125 / 100) :
  (∃ x : ℕ, x = 120 ∧ second_batch_cost / (2 * x) = first_batch_cost / x + 10) ∧
  (∃ y : ℕ, y = 150 ∧ 350 * y ≥ (first_batch_cost + second_batch_cost) * 125 / 100) :=
by sorry


end shirt_problem_l1915_191575


namespace copper_alloy_impossibility_l1915_191584

/-- Proves the impossibility of creating a specific copper alloy mixture --/
theorem copper_alloy_impossibility : ∀ (x : ℝ),
  0 ≤ x ∧ x ≤ 100 →
  32 * 0.25 + 8 * (x / 100) ≠ 40 * 0.45 :=
by
  sorry

#check copper_alloy_impossibility

end copper_alloy_impossibility_l1915_191584


namespace three_zeros_sin_minus_one_l1915_191536

/-- The function f(x) = sin(ωx) - 1 has exactly 3 zeros in [0, 2π] iff ω ∈ [9/4, 13/4) -/
theorem three_zeros_sin_minus_one (ω : ℝ) : ω > 0 →
  (∃! (s : Finset ℝ), s.card = 3 ∧ (∀ x ∈ s, x ∈ Set.Icc 0 (2 * Real.pi) ∧ Real.sin (ω * x) = 1)) ↔
  ω ∈ Set.Icc (9 / 4) (13 / 4) := by
  sorry

#check three_zeros_sin_minus_one

end three_zeros_sin_minus_one_l1915_191536


namespace unique_triple_solution_l1915_191557

theorem unique_triple_solution : 
  ∃! (n p q : ℕ), n ≥ 2 ∧ n^p + n^q = n^2010 ∧ p = 2009 ∧ q = 2009 ∧ n = 2 :=
by sorry

end unique_triple_solution_l1915_191557


namespace equal_area_dividing_line_slope_l1915_191505

theorem equal_area_dividing_line_slope (r : ℝ) (c1 c2 p : ℝ × ℝ) (m : ℝ) : 
  r = 4 ∧ 
  c1 = (0, 20) ∧ 
  c2 = (6, 12) ∧ 
  p = (4, 0) ∧
  (∀ (x y : ℝ), y = m * (x - p.1) + p.2) ∧
  (∀ (x y : ℝ), (x - c1.1)^2 + (y - c1.2)^2 = r^2 → 
    (m * x - y + (p.2 - m * p.1))^2 / (m^2 + 1) = 
    (m * c2.1 - c2.2 + (p.2 - m * p.1))^2 / (m^2 + 1)) →
  |m| = 4/3 := by
sorry

end equal_area_dividing_line_slope_l1915_191505


namespace shoes_total_price_l1915_191560

/-- Given the conditions of Jeff's purchase, prove the total price of shoes. -/
theorem shoes_total_price (total_cost : ℕ) (shoe_pairs : ℕ) (jerseys : ℕ) 
  (h1 : total_cost = 560)
  (h2 : shoe_pairs = 6)
  (h3 : jerseys = 4)
  (h4 : ∃ (shoe_price : ℚ), total_cost = shoe_pairs * shoe_price + jerseys * (shoe_price / 4)) :
  shoe_pairs * (total_cost / (shoe_pairs + jerseys / 4 : ℚ)) = 480 := by
  sorry

#check shoes_total_price

end shoes_total_price_l1915_191560


namespace gcd_problem_l1915_191596

theorem gcd_problem :
  ∃! n : ℕ, 80 ≤ n ∧ n ≤ 100 ∧ Nat.gcd n 27 = 9 :=
by
  -- Proof goes here
  sorry

end gcd_problem_l1915_191596


namespace polygon_interior_exterior_angle_sum_l1915_191507

theorem polygon_interior_exterior_angle_sum (n : ℕ) : 
  (n ≥ 3) → (((n - 2) * 180 = 2 * 360) ↔ n = 6) := by
  sorry

end polygon_interior_exterior_angle_sum_l1915_191507


namespace horner_method_v1_l1915_191518

def f (x : ℝ) : ℝ := 3 * x^4 + 2 * x^2 + x + 4

def horner_v1 (a : ℝ) : ℝ := 3 * a + 0

theorem horner_method_v1 :
  let x : ℝ := 10
  horner_v1 x = 30 := by sorry

end horner_method_v1_l1915_191518


namespace probability_A1_or_B1_not_both_l1915_191528

def excellent_math : ℕ := 3
def excellent_physics : ℕ := 2
def excellent_chemistry : ℕ := 2

def total_combinations : ℕ := excellent_math * excellent_physics * excellent_chemistry

def favorable_outcomes : ℕ := 
  excellent_physics * excellent_chemistry + 
  (excellent_math - 1) * excellent_chemistry

theorem probability_A1_or_B1_not_both : 
  (favorable_outcomes : ℚ) / total_combinations = 1 / 2 := by sorry

end probability_A1_or_B1_not_both_l1915_191528


namespace min_sum_of_probabilities_l1915_191562

theorem min_sum_of_probabilities (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  let p_a := 4 / x
  let p_b := 1 / y
  (p_a + p_b = 1) → (∀ x' y' : ℝ, x' > 0 → y' > 0 → 4 / x' + 1 / y' = 1 → x' + y' ≥ x + y) →
  x + y = 9 :=
by sorry

end min_sum_of_probabilities_l1915_191562


namespace triangle_formation_l1915_191501

/-- Triangle inequality theorem: the sum of the lengths of any two sides 
    of a triangle must be greater than the length of the remaining side -/
def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Check if three line segments can form a triangle -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ triangle_inequality a b c

theorem triangle_formation :
  can_form_triangle 5 6 10 ∧
  ¬can_form_triangle 2 3 5 ∧
  ¬can_form_triangle 5 6 11 ∧
  ¬can_form_triangle 3 4 8 :=
sorry

end triangle_formation_l1915_191501


namespace students_in_all_workshops_l1915_191588

theorem students_in_all_workshops (total : ℕ) (robotics dance music : ℕ) (at_least_two : ℕ) 
  (h_total : total = 25)
  (h_robotics : robotics = 15)
  (h_dance : dance = 12)
  (h_music : music = 10)
  (h_at_least_two : at_least_two = 11)
  (h_sum : robotics + dance + music - 2 * at_least_two ≤ total) :
  ∃ (only_one only_two all_three : ℕ),
    only_one + only_two + all_three = total ∧
    only_two + 3 * all_three = at_least_two ∧
    all_three = 1 :=
by sorry

end students_in_all_workshops_l1915_191588


namespace expression_value_l1915_191592

theorem expression_value : 100 * (100 - 3) - (100 * 100 - 3) = -297 := by sorry

end expression_value_l1915_191592


namespace largest_510_triple_l1915_191570

/-- Converts a base-10 number to its base-5 representation as a list of digits -/
def toBase5 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) : List ℕ :=
    if m = 0 then [] else (m % 5) :: aux (m / 5)
  aux n

/-- Interprets a list of digits as a base-10 number -/
def fromDigits (digits : List ℕ) : ℕ :=
  digits.foldl (fun acc d => 10 * acc + d) 0

/-- Checks if a number is a 5-10 triple -/
def is510Triple (n : ℕ) : Prop :=
  fromDigits (toBase5 n) = 3 * n

theorem largest_510_triple :
  (∀ m : ℕ, m > 115 → ¬ is510Triple m) ∧ is510Triple 115 :=
sorry

end largest_510_triple_l1915_191570


namespace blaine_fish_count_l1915_191572

theorem blaine_fish_count :
  ∀ (blaine_fish keith_fish : ℕ),
    blaine_fish > 0 →
    keith_fish = 2 * blaine_fish →
    blaine_fish + keith_fish = 15 →
    blaine_fish = 5 := by
  sorry

end blaine_fish_count_l1915_191572


namespace astronaut_education_time_l1915_191548

theorem astronaut_education_time (total_time science_time : ℕ) : 
  total_time = 14 ∧ 
  total_time = science_time + 2 * science_time + 2 →
  science_time = 4 := by
sorry

end astronaut_education_time_l1915_191548


namespace factors_of_210_l1915_191579

def number_of_factors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

theorem factors_of_210 : number_of_factors 210 = 16 := by
  sorry

end factors_of_210_l1915_191579


namespace homework_difference_l1915_191526

theorem homework_difference (math reading history science : ℕ) 
  (h_math : math = 5)
  (h_reading : reading = 7)
  (h_history : history = 3)
  (h_science : science = 6) :
  (reading - math) + (science - history) = 5 :=
by sorry

end homework_difference_l1915_191526


namespace sine_of_vertex_angle_is_four_fifths_l1915_191520

/-- An isosceles triangle with a special property regarding inscribed rectangles. -/
structure SpecialIsoscelesTriangle where
  -- The vertex angle of the isosceles triangle
  vertex_angle : ℝ
  -- A function that takes two real numbers (representing the sides of a rectangle)
  -- and returns whether that rectangle can be inscribed in the triangle
  is_inscribable : ℝ → ℝ → Prop
  -- The constant perimeter of inscribable rectangles
  constant_perimeter : ℝ
  -- Property: The triangle is isosceles
  is_isosceles : Prop
  -- Property: Any rectangle that can be inscribed has the constant perimeter
  perimeter_is_constant : ∀ x y, is_inscribable x y → x + y = constant_perimeter

/-- 
The main theorem: In a special isosceles triangle where all inscribable rectangles 
have a constant perimeter, the sine of the vertex angle is 4/5.
-/
theorem sine_of_vertex_angle_is_four_fifths (t : SpecialIsoscelesTriangle) : 
  Real.sin t.vertex_angle = 4/5 := by
  sorry


end sine_of_vertex_angle_is_four_fifths_l1915_191520


namespace mitchell_has_30_pencils_l1915_191591

/-- The number of pencils Antonio has -/
def antonio_pencils : ℕ := sorry

/-- The number of pencils Mitchell has -/
def mitchell_pencils : ℕ := antonio_pencils + 6

/-- The total number of pencils Mitchell and Antonio have together -/
def total_pencils : ℕ := 54

theorem mitchell_has_30_pencils :
  mitchell_pencils = 30 :=
by
  sorry

#check mitchell_has_30_pencils

end mitchell_has_30_pencils_l1915_191591


namespace individual_contribution_proof_l1915_191529

def total_contribution : ℝ := 90
def class_funds : ℝ := 30
def num_students : ℝ := 25

theorem individual_contribution_proof :
  (total_contribution - class_funds) / num_students = 2.40 :=
by sorry

end individual_contribution_proof_l1915_191529


namespace right_triangle_hypotenuse_l1915_191543

theorem right_triangle_hypotenuse (x y h : ℝ) : 
  x > 0 → 
  y = 2 * x + 2 → 
  (1 / 2) * x * y = 72 → 
  x^2 + y^2 = h^2 → 
  h = Real.sqrt 388 := by
sorry

end right_triangle_hypotenuse_l1915_191543


namespace fifth_number_12th_row_l1915_191511

/-- Given a lattice with the following properties:
  - Has 12 rows
  - Each row contains 7 consecutive numbers
  - The first number in Row 1 is 1
  - The first number in each row increases by 8 as the row number increases
  This function calculates the nth number in the mth row -/
def latticeNumber (m n : ℕ) : ℕ :=
  (1 + 8 * (m - 1)) + (n - 1)

/-- The theorem states that the fifth number in the 12th row of the described lattice is 93 -/
theorem fifth_number_12th_row : latticeNumber 12 5 = 93 := by
  sorry

end fifth_number_12th_row_l1915_191511


namespace min_distance_sum_l1915_191522

noncomputable def parabola (x y : ℝ) : Prop := x^2 = 4*y

def focus : ℝ × ℝ := (0, 1)

def point_A : ℝ × ℝ := (2, 3)

theorem min_distance_sum (P : ℝ × ℝ) :
  parabola P.1 P.2 →
  Real.sqrt ((P.1 - point_A.1)^2 + (P.2 - point_A.2)^2) +
  Real.sqrt ((P.1 - focus.1)^2 + (P.2 - focus.2)^2) ≥ 4 := by
  sorry

end min_distance_sum_l1915_191522


namespace trajectory_of_B_l1915_191568

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space of the form ax + by = c -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a point lies on a given line -/
def Point.isOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y = l.c

/-- Defines a parallelogram ABCD -/
structure Parallelogram where
  A : Point
  B : Point
  C : Point
  D : Point
  is_parallelogram : (B.x - A.x = D.x - C.x) ∧ (B.y - A.y = D.y - C.y)

/-- Theorem: Trajectory of point B in a parallelogram ABCD -/
theorem trajectory_of_B (ABCD : Parallelogram)
  (hA : ABCD.A = Point.mk (-1) 3)
  (hC : ABCD.C = Point.mk (-3) 2)
  (hD : ABCD.D.isOnLine (Line.mk 1 (-3) 1)) :
  ABCD.B.isOnLine (Line.mk 1 (-3) 20) :=
sorry

end trajectory_of_B_l1915_191568
