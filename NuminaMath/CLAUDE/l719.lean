import Mathlib

namespace intersection_empty_implies_m_range_l719_71919

theorem intersection_empty_implies_m_range (m : ℝ) : 
  let A : Set ℝ := {x | x^2 - x - 6 > 0}
  let B : Set ℝ := {x | (x - m) * (x - 2*m) ≤ 0}
  A ∩ B = ∅ → m ∈ Set.Icc (-1 : ℝ) (3/2 : ℝ) := by
  sorry

end intersection_empty_implies_m_range_l719_71919


namespace absolute_value_simplification_l719_71985

theorem absolute_value_simplification : |-4^2 + 7| = 9 := by
  sorry

end absolute_value_simplification_l719_71985


namespace linear_function_property_l719_71927

/-- A function f satisfying f(x₁ + x₂) = f(x₁) + f(x₂) for all real x₁ and x₂ -/
def LinearFunction (f : ℝ → ℝ) : Prop :=
  ∀ (x₁ x₂ : ℝ), f (x₁ + x₂) = f x₁ + f x₂

/-- Theorem: A function of the form f(x) = kx, where k is a non-zero constant,
    satisfies the property f(x₁ + x₂) = f(x₁) + f(x₂) for all real x₁ and x₂ -/
theorem linear_function_property (k : ℝ) (hk : k ≠ 0) :
  LinearFunction (fun x ↦ k * x) := by
  sorry

end linear_function_property_l719_71927


namespace polynomial_factorization_l719_71928

theorem polynomial_factorization (x : ℤ) :
  3 * (x + 3) * (x + 4) * (x + 7) * (x + 8) - 2 * x^2 =
  (3 * x^2 + 35 * x + 72) * (x + 3) * (x + 6) := by
  sorry

end polynomial_factorization_l719_71928


namespace max_digit_sum_l719_71907

def DigitalClock := Fin 24 × Fin 60

def digit_sum (time : DigitalClock) : Nat :=
  let (h, m) := time
  let h1 := h.val / 10
  let h2 := h.val % 10
  let m1 := m.val / 10
  let m2 := m.val % 10
  h1 + h2 + m1 + m2

theorem max_digit_sum :
  ∃ (max_time : DigitalClock), ∀ (time : DigitalClock), digit_sum time ≤ digit_sum max_time ∧ digit_sum max_time = 19 := by
  sorry

end max_digit_sum_l719_71907


namespace excluded_students_average_mark_l719_71941

theorem excluded_students_average_mark
  (total_students : ℕ)
  (all_average : ℝ)
  (excluded_count : ℕ)
  (remaining_average : ℝ)
  (h1 : total_students = 30)
  (h2 : all_average = 80)
  (h3 : excluded_count = 5)
  (h4 : remaining_average = 92)
  : (total_students * all_average - (total_students - excluded_count) * remaining_average) / excluded_count = 20 :=
by sorry

end excluded_students_average_mark_l719_71941


namespace marble_jar_problem_l719_71981

theorem marble_jar_problem (total_marbles : ℕ) : 
  (∃ (marbles_per_person : ℕ), 
    total_marbles = 18 * marbles_per_person ∧ 
    total_marbles = 20 * (marbles_per_person - 1)) → 
  total_marbles = 180 := by
sorry

end marble_jar_problem_l719_71981


namespace count_eight_digit_numbers_seven_different_is_correct_l719_71902

/-- The number of 8-digit numbers where exactly 7 digits are all different -/
def count_eight_digit_numbers_seven_different : ℕ := 5080320

/-- Theorem stating that the count of 8-digit numbers where exactly 7 digits are all different is 5080320 -/
theorem count_eight_digit_numbers_seven_different_is_correct :
  count_eight_digit_numbers_seven_different = 5080320 := by sorry

end count_eight_digit_numbers_seven_different_is_correct_l719_71902


namespace steve_commute_speed_l719_71915

theorem steve_commute_speed (distance : ℝ) (total_time : ℝ) : 
  distance > 0 → 
  total_time > 0 → 
  ∃ (outbound_speed : ℝ), 
    outbound_speed > 0 ∧ 
    (distance / outbound_speed + distance / (2 * outbound_speed) = total_time) → 
    2 * outbound_speed = 14 := by
  sorry

end steve_commute_speed_l719_71915


namespace workshop_workers_l719_71984

/-- The number of technicians in the workshop -/
def num_technicians : ℕ := 7

/-- The average salary of all workers in the workshop -/
def avg_salary_all : ℕ := 8000

/-- The average salary of technicians in the workshop -/
def avg_salary_tech : ℕ := 12000

/-- The average salary of non-technicians in the workshop -/
def avg_salary_nontech : ℕ := 6000

/-- The total number of workers in the workshop -/
def total_workers : ℕ := 21

theorem workshop_workers :
  (total_workers * avg_salary_all = 
   num_technicians * avg_salary_tech + 
   (total_workers - num_technicians) * avg_salary_nontech) ∧
  (total_workers = 21) := by
  sorry

end workshop_workers_l719_71984


namespace inverse_of_B_cubed_l719_71975

def B_inv : Matrix (Fin 2) (Fin 2) ℝ := !![3, -1; 1, 1]

theorem inverse_of_B_cubed :
  let B_inv := !![3, -1; 1, 1]
  (B_inv^3)⁻¹ = !![20, -12; 12, -4] := by sorry

end inverse_of_B_cubed_l719_71975


namespace max_value_expression_l719_71926

theorem max_value_expression (a b c d : ℝ) 
  (ha : -8.5 ≤ a ∧ a ≤ 8.5)
  (hb : -8.5 ≤ b ∧ b ≤ 8.5)
  (hc : -8.5 ≤ c ∧ c ≤ 8.5)
  (hd : -8.5 ≤ d ∧ d ≤ 8.5) :
  (∀ x y z w, -8.5 ≤ x ∧ x ≤ 8.5 ∧ 
              -8.5 ≤ y ∧ y ≤ 8.5 ∧ 
              -8.5 ≤ z ∧ z ≤ 8.5 ∧ 
              -8.5 ≤ w ∧ w ≤ 8.5 → 
              x + 2*y + z + 2*w - x*y - y*z - z*w - w*x ≤ 306) ∧
  (∃ x y z w, -8.5 ≤ x ∧ x ≤ 8.5 ∧ 
              -8.5 ≤ y ∧ y ≤ 8.5 ∧ 
              -8.5 ≤ z ∧ z ≤ 8.5 ∧ 
              -8.5 ≤ w ∧ w ≤ 8.5 ∧ 
              x + 2*y + z + 2*w - x*y - y*z - z*w - w*x = 306) :=
by sorry

end max_value_expression_l719_71926


namespace parallel_vectors_l719_71986

def a : ℝ × ℝ := (-1, 1)
def b (m : ℝ) : ℝ × ℝ := (3, m)

theorem parallel_vectors (m : ℝ) : 
  (∃ (k : ℝ), a = k • (a + b m)) → m = -7 := by
  sorry

end parallel_vectors_l719_71986


namespace largest_n_satisfying_inequality_l719_71988

theorem largest_n_satisfying_inequality : 
  ∃ (n : ℕ), n^300 < 3^500 ∧ ∀ (m : ℕ), m^300 < 3^500 → m ≤ n :=
by
  use 6
  sorry

end largest_n_satisfying_inequality_l719_71988


namespace power_function_not_through_origin_l719_71952

theorem power_function_not_through_origin (m : ℝ) : 
  (m = 1 ∨ m = 2) → 
  ∀ x : ℝ, x ≠ 0 → (m^2 - 3*m + 3) * x^((m^2 - m - 2)/2) ≠ 0 :=
by sorry

end power_function_not_through_origin_l719_71952


namespace ferry_time_difference_l719_71998

/-- A ferry route between an island and the mainland -/
structure FerryRoute where
  speed : ℝ  -- Speed in km/h
  time : ℝ   -- Time in hours
  distance : ℝ -- Distance in km

/-- The problem setup for two ferry routes -/
def ferry_problem (p q : FerryRoute) : Prop :=
  p.speed = 8 ∧
  p.time = 3 ∧
  p.distance = p.speed * p.time ∧
  q.distance = 3 * p.distance ∧
  q.speed = p.speed + 1

theorem ferry_time_difference (p q : FerryRoute) 
  (h : ferry_problem p q) : q.time - p.time = 5 := by
  sorry

end ferry_time_difference_l719_71998


namespace infinitely_many_pairs_l719_71929

theorem infinitely_many_pairs (c : ℝ) : 
  (c > 0) → 
  (∀ k : ℕ, ∃ n m : ℕ, 
    n > 0 ∧ m > 0 ∧
    (n : ℝ) ≥ (m : ℝ) + c * Real.sqrt ((m : ℝ) - 1) + 1 ∧
    ∀ i ∈ Finset.range (2 * n - m - n + 1), ¬ ∃ j : ℕ, (n + i : ℝ) = (j : ℝ) ^ 2) ↔ 
  c ≤ 2 := by
sorry

end infinitely_many_pairs_l719_71929


namespace orchid_count_l719_71909

/-- The number of orchid bushes initially in the park -/
def initial_orchids : ℕ := 22

/-- The number of orchid bushes to be planted -/
def planted_orchids : ℕ := 13

/-- The final number of orchid bushes after planting -/
def final_orchids : ℕ := 35

/-- Theorem stating that the initial number of orchid bushes plus the planted ones equals the final number -/
theorem orchid_count : initial_orchids + planted_orchids = final_orchids := by
  sorry

end orchid_count_l719_71909


namespace largest_value_l719_71994

theorem largest_value (x y : ℝ) (h1 : 0 < x) (h2 : x < y) (h3 : x + y = 1) :
  (1/2 < x^2 + y^2) ∧ (2*x*y < x^2 + y^2) ∧ (x < x^2 + y^2) := by
  sorry

end largest_value_l719_71994


namespace company_employee_increase_l719_71996

theorem company_employee_increase (january_employees : ℝ) (increase_percentage : ℝ) :
  january_employees = 434.7826086956522 →
  increase_percentage = 15 →
  january_employees * (1 + increase_percentage / 100) = 500 := by
sorry

end company_employee_increase_l719_71996


namespace problem_statement_l719_71995

/-- Given xw + yz = 8 and (2x + y)(2z + w) = 20, prove that xz + yw = 1 -/
theorem problem_statement (x y z w : ℝ) 
  (h1 : x * w + y * z = 8)
  (h2 : (2 * x + y) * (2 * z + w) = 20) :
  x * z + y * w = 1 := by
  sorry

end problem_statement_l719_71995


namespace arithmetic_sequence_m_value_l719_71943

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def sum_of_b (b : ℕ → ℝ) : ℕ → ℝ
  | 0 => 0
  | n + 1 => sum_of_b b n + b (n + 1)

theorem arithmetic_sequence_m_value
  (a : ℕ → ℝ)
  (b : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_a3 : a 3 = 5)
  (h_a9 : a 9 = 17)
  (h_sum_b : ∀ n : ℕ, sum_of_b b n = 3^n - 1)
  (h_relation : ∃ m : ℕ, m > 0 ∧ 1 + a m = b 4) :
  ∃ m : ℕ, m > 0 ∧ 1 + a m = b 4 ∧ m = 27 :=
sorry

end arithmetic_sequence_m_value_l719_71943


namespace unique_solution_for_B_l719_71973

theorem unique_solution_for_B : ∃! B : ℕ, 
  B < 10 ∧ (∃ A : ℕ, A < 10 ∧ 500 + 10 * A + 8 - (100 * B + 14) = 364) :=
sorry

end unique_solution_for_B_l719_71973


namespace continuous_at_5_l719_71912

def f (x : ℝ) : ℝ := 3 * x^2 - 2

theorem continuous_at_5 :
  ∀ ε > 0, ∃ δ > 0, ∀ x, |x - 5| < δ → |f x - f 5| < ε :=
by sorry

end continuous_at_5_l719_71912


namespace positive_cubes_inequality_l719_71989

theorem positive_cubes_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) :
  a^3 + b^3 > a^2*b + a*b^2 := by
  sorry

end positive_cubes_inequality_l719_71989


namespace mixed_games_count_l719_71980

/-- Represents a chess competition with men and women players -/
structure ChessCompetition where
  womenCount : ℕ
  menCount : ℕ
  womenGames : ℕ
  menGames : ℕ

/-- Calculates the number of games between a man and a woman -/
def mixedGames (c : ChessCompetition) : ℕ :=
  c.womenCount * c.menCount

/-- Theorem stating the relationship between the number of games -/
theorem mixed_games_count (c : ChessCompetition) 
  (h1 : c.womenGames = 45)
  (h2 : c.menGames = 190)
  (h3 : c.womenGames = c.womenCount * (c.womenCount - 1) / 2)
  (h4 : c.menGames = c.menCount * (c.menCount - 1) / 2) :
  mixedGames c = 200 := by
  sorry


end mixed_games_count_l719_71980


namespace sum_of_x_solutions_is_zero_l719_71957

theorem sum_of_x_solutions_is_zero (x y : ℝ) : 
  y = 9 → x^2 + y^2 = 169 → ∃ x₁ x₂ : ℝ, x₁ + x₂ = 0 ∧ x₁^2 + y^2 = 169 ∧ x₂^2 + y^2 = 169 :=
by sorry

end sum_of_x_solutions_is_zero_l719_71957


namespace soccer_shoe_price_l719_71961

theorem soccer_shoe_price (total_pairs : Nat) (total_price : Nat) :
  total_pairs = 99 →
  total_price % 100 = 76 →
  total_price < 20000 →
  ∃ (price_per_pair : Nat), 
    price_per_pair * total_pairs = total_price ∧
    price_per_pair = 124 :=
by sorry

end soccer_shoe_price_l719_71961


namespace lunch_breakfast_difference_l719_71932

def muffin_cost : ℚ := 2
def coffee_cost : ℚ := 4
def soup_cost : ℚ := 3
def salad_cost : ℚ := 5.25
def lemonade_cost : ℚ := 0.75

def breakfast_cost : ℚ := muffin_cost + coffee_cost
def lunch_cost : ℚ := soup_cost + salad_cost + lemonade_cost

theorem lunch_breakfast_difference : lunch_cost - breakfast_cost = 3 := by
  sorry

end lunch_breakfast_difference_l719_71932


namespace N_mod_five_l719_71976

def base_nine_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (9^i)) 0

def N : Nat :=
  base_nine_to_decimal [2, 5, 0, 0, 0, 0, 0, 6, 0, 0, 7, 2]

theorem N_mod_five : N % 5 = 3 := by
  sorry

end N_mod_five_l719_71976


namespace apple_group_addition_l719_71944

/-- Given a basket of apples divided among a group, prove the number of people who joined --/
theorem apple_group_addition (total_apples : ℕ) (original_per_person : ℕ) (new_per_person : ℕ) :
  total_apples = 1430 →
  original_per_person = 22 →
  new_per_person = 13 →
  ∃ (original_group : ℕ) (joined_group : ℕ),
    original_group * original_per_person = total_apples ∧
    (original_group + joined_group) * new_per_person = total_apples ∧
    joined_group = 45 := by
  sorry


end apple_group_addition_l719_71944


namespace vertex_of_f_l719_71916

-- Define the quadratic function
def f (x : ℝ) : ℝ := -2 * (x + 1)^2 + 3

-- Theorem stating that the vertex of f is at (-1, 3)
theorem vertex_of_f :
  (∃ (a : ℝ), f a = 3 ∧ ∀ x, f x ≤ 3) ∧ f (-1) = 3 :=
sorry

end vertex_of_f_l719_71916


namespace largest_choir_size_l719_71969

theorem largest_choir_size : 
  ∃ (n s : ℕ), 
    n * s < 150 ∧ 
    n * s + 3 = (n + 2) * (s - 3) ∧ 
    ∀ (m n' s' : ℕ), 
      m < 150 → 
      m + 3 = n' * s' → 
      m = (n' + 2) * (s' - 3) → 
      m ≤ n * s :=
by sorry

end largest_choir_size_l719_71969


namespace shirt_price_proof_l719_71946

-- Define the original price
def original_price : ℝ := 32

-- Define the discount rate
def discount_rate : ℝ := 0.25

-- Define the final price
def final_price : ℝ := 18

-- Theorem statement
theorem shirt_price_proof :
  (1 - discount_rate) * (1 - discount_rate) * original_price = final_price :=
by sorry

end shirt_price_proof_l719_71946


namespace ipod_ratio_l719_71935

-- Define the initial number of iPods Emmy has
def emmy_initial : ℕ := 14

-- Define the number of iPods Emmy loses
def emmy_lost : ℕ := 6

-- Define the total number of iPods Emmy and Rosa have together
def total_ipods : ℕ := 12

-- Define Emmy's remaining iPods
def emmy_remaining : ℕ := emmy_initial - emmy_lost

-- Define Rosa's iPods
def rosa_ipods : ℕ := total_ipods - emmy_remaining

-- Theorem statement
theorem ipod_ratio : 
  emmy_remaining * 1 = rosa_ipods * 2 := by
  sorry

end ipod_ratio_l719_71935


namespace add_2057_minutes_to_3_15pm_l719_71954

/-- Represents time in 24-hour format -/
structure Time where
  hours : Nat
  minutes : Nat
  h_valid : hours < 24
  m_valid : minutes < 60

/-- Adds minutes to a given time -/
def addMinutes (t : Time) (m : Nat) : Time :=
  sorry

theorem add_2057_minutes_to_3_15pm (start : Time) (result : Time) :
  start.hours = 15 ∧ start.minutes = 15 →
  result = addMinutes start 2057 →
  result.hours = 1 ∧ result.minutes = 32 := by
  sorry

end add_2057_minutes_to_3_15pm_l719_71954


namespace cos_shift_equals_sin_l719_71965

theorem cos_shift_equals_sin (x : ℝ) : 
  Real.cos (x + π/3) = Real.sin (x + 5*π/6) := by
  sorry

end cos_shift_equals_sin_l719_71965


namespace final_bus_count_l719_71936

def bus_problem (initial : ℕ) (first_stop : ℕ) (second_stop : ℕ) (third_stop : ℕ) : ℕ :=
  initial + first_stop - second_stop + third_stop

theorem final_bus_count :
  bus_problem 128 67 34 54 = 215 := by
  sorry

end final_bus_count_l719_71936


namespace geometric_sequence_sum_l719_71918

/-- A geometric sequence with common ratio q < 0 -/
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  q < 0 ∧ ∀ n, a (n + 1) = a n * q

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) :
  geometric_sequence a q →
  (a 2 = 1 - a 1) →
  (a 4 = 4 - a 3) →
  a 5 + a 6 = 16 := by
sorry

end geometric_sequence_sum_l719_71918


namespace sqrt_sum_rationalization_l719_71991

theorem sqrt_sum_rationalization : ∃ (a b c : ℕ+), 
  (Real.sqrt 8 + (1 / Real.sqrt 8) + Real.sqrt 9 + (1 / Real.sqrt 9) = (a * Real.sqrt 8 + b * Real.sqrt 9) / c) ∧
  (∀ (a' b' c' : ℕ+), 
    (Real.sqrt 8 + (1 / Real.sqrt 8) + Real.sqrt 9 + (1 / Real.sqrt 9) = (a' * Real.sqrt 8 + b' * Real.sqrt 9) / c') →
    c ≤ c') ∧
  (a + b + c = 31) :=
by sorry

end sqrt_sum_rationalization_l719_71991


namespace units_digit_of_first_four_composites_product_l719_71968

def first_four_composites : List Nat := [4, 6, 8, 9]

def product_of_list (l : List Nat) : Nat :=
  l.foldl (·*·) 1

def units_digit (n : Nat) : Nat :=
  n % 10

theorem units_digit_of_first_four_composites_product :
  units_digit (product_of_list first_four_composites) = 8 := by
  sorry

end units_digit_of_first_four_composites_product_l719_71968


namespace pure_imaginary_ratio_l719_71917

theorem pure_imaginary_ratio (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) 
  (h3 : ∃ (y : ℝ), (3 - 8 * Complex.I) * (a + b * Complex.I) = y * Complex.I) : 
  a / b = -8 / 3 := by
sorry

end pure_imaginary_ratio_l719_71917


namespace row_sum_equals_2013_squared_l719_71999

theorem row_sum_equals_2013_squared :
  let n : ℕ := 1007
  let row_sum (k : ℕ) : ℕ := k * (2 * k - 1)
  row_sum n = 2013^2 := by
  sorry

end row_sum_equals_2013_squared_l719_71999


namespace largest_special_number_l719_71903

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

theorem largest_special_number : 
  ∀ n : ℕ, n < 200 → is_perfect_square n → n % 3 = 0 → n ≤ 144 :=
by sorry

end largest_special_number_l719_71903


namespace wheel_speed_l719_71979

/-- The speed (in miles per hour) of a wheel with a 10-foot circumference -/
def r : ℝ := sorry

/-- The time (in hours) for one complete rotation of the wheel -/
def t : ℝ := sorry

/-- Relation between speed, time, and distance for one rotation -/
axiom speed_time_relation : r * t = (10 / 5280)

/-- Relation between original and new speed and time -/
axiom speed_time_change : (r + 5) * (t - 1 / (3 * 3600)) = (10 / 5280)

theorem wheel_speed : r = 10 := by sorry

end wheel_speed_l719_71979


namespace line_intersects_parabola_at_one_point_l719_71937

/-- The parabola function -/
def parabola (y : ℝ) : ℝ := -3 * y^2 - 4 * y + 7

/-- The condition for the line to intersect the parabola at one point -/
def intersects_at_one_point (k : ℝ) : Prop :=
  ∃! y : ℝ, parabola y = k

/-- The theorem stating the value of k for which the line intersects the parabola at one point -/
theorem line_intersects_parabola_at_one_point :
  ∃! k : ℝ, intersects_at_one_point k ∧ k = 25/3 :=
sorry

end line_intersects_parabola_at_one_point_l719_71937


namespace decimal_point_problem_l719_71964

theorem decimal_point_problem : ∃ x : ℝ, x > 0 ∧ 1000 * x = 9 * (1 / x) := by
  sorry

end decimal_point_problem_l719_71964


namespace complex_point_not_in_third_quadrant_l719_71900

theorem complex_point_not_in_third_quadrant (m : ℝ) :
  ¬(m^2 + m - 2 < 0 ∧ 6 - m - m^2 < 0) := by
  sorry

end complex_point_not_in_third_quadrant_l719_71900


namespace cos_two_alpha_plus_two_beta_l719_71949

theorem cos_two_alpha_plus_two_beta (α β : ℝ) 
  (h1 : Real.sin (α - β) = 1/3)
  (h2 : Real.cos α * Real.sin β = 1/6) : 
  Real.cos (2*α + 2*β) = 1/9 := by
  sorry

end cos_two_alpha_plus_two_beta_l719_71949


namespace function_composition_equality_l719_71921

theorem function_composition_equality (a b c d : ℝ) 
  (f : ℝ → ℝ) (g : ℝ → ℝ)
  (hf : ∀ x, f x = a * x + b)
  (hg : ∀ x, g x = c * x + d)
  (ha : a = 2 * c) :
  (∀ x, f (g x) = g (f x)) ↔ (b = d ∨ c = 1/2) := by sorry

end function_composition_equality_l719_71921


namespace sheep_count_l719_71951

theorem sheep_count : ∀ (num_sheep : ℕ), 
  (∀ (sheep : ℕ), sheep ≤ num_sheep → (sheep * 1 = sheep)) →  -- One sheep eats one bag in 40 days
  (num_sheep * 1 = 40) →  -- Total bags eaten by all sheep is 40
  num_sheep = 40 := by
sorry

end sheep_count_l719_71951


namespace find_k_value_l719_71990

theorem find_k_value : ∃ k : ℝ, ∀ x : ℝ, -x^2 - (k + 11)*x - 8 = -(x - 2)*(x - 4) → k = -17 := by
  sorry

end find_k_value_l719_71990


namespace tangent_line_to_ln_curve_l719_71977

/-- Given a line y = kx tangent to y = ln x and passing through the origin, k = 1/e -/
theorem tangent_line_to_ln_curve (k : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ k * x = Real.log x ∧ k = 1 / x) → 
  k = 1 / Real.exp 1 := by
  sorry

end tangent_line_to_ln_curve_l719_71977


namespace time_sum_after_duration_l719_71983

/-- Represents time in hours, minutes, and seconds -/
structure Time where
  hours : Nat
  minutes : Nat
  seconds : Nat

/-- Adds a duration to a given time and returns the result on a 12-hour clock -/
def addDuration (startTime : Time) (durationHours durationMinutes durationSeconds : Nat) : Time :=
  sorry

theorem time_sum_after_duration (startTime : Time) (durationHours durationMinutes durationSeconds : Nat) :
  let endTime := addDuration startTime durationHours durationMinutes durationSeconds
  startTime.hours = 3 ∧ startTime.minutes = 0 ∧ startTime.seconds = 0 ∧
  durationHours = 315 ∧ durationMinutes = 58 ∧ durationSeconds = 16 →
  endTime.hours + endTime.minutes + endTime.seconds = 77 := by
  sorry

end time_sum_after_duration_l719_71983


namespace city_outgoing_roads_l719_71905

/-- Represents a city with squares and roads -/
structure City where
  /-- Number of squares in the city -/
  squares : ℕ
  /-- Number of streets going out of the city -/
  outgoing_streets : ℕ
  /-- Number of avenues going out of the city -/
  outgoing_avenues : ℕ
  /-- Number of crescents going out of the city -/
  outgoing_crescents : ℕ
  /-- Total number of outgoing roads is 3 -/
  outgoing_total : outgoing_streets + outgoing_avenues + outgoing_crescents = 3

/-- Theorem: In a city where exactly three roads meet at every square (one street, one avenue, and one crescent),
    and three roads go outside of the city, there must be exactly one street, one avenue, and one crescent going out of the city -/
theorem city_outgoing_roads (c : City) : 
  c.outgoing_streets = 1 ∧ c.outgoing_avenues = 1 ∧ c.outgoing_crescents = 1 := by
  sorry

end city_outgoing_roads_l719_71905


namespace david_average_marks_l719_71947

def david_marks : List ℕ := [86, 85, 82, 87, 85]

theorem david_average_marks :
  (List.sum david_marks) / (List.length david_marks) = 85 := by
  sorry

end david_average_marks_l719_71947


namespace bucket_capacity_problem_l719_71908

theorem bucket_capacity_problem (tank_capacity : ℝ) (first_case_buckets : ℕ) (second_case_buckets : ℕ) (second_case_capacity : ℝ) :
  first_case_buckets = 13 →
  second_case_buckets = 39 →
  second_case_capacity = 17 →
  tank_capacity = first_case_buckets * (tank_capacity / first_case_buckets) →
  tank_capacity = second_case_buckets * second_case_capacity →
  tank_capacity / first_case_buckets = 51 :=
by
  sorry

end bucket_capacity_problem_l719_71908


namespace starters_with_triplet_l719_71997

def total_players : ℕ := 12
def triplets : ℕ := 3
def starters : ℕ := 5

theorem starters_with_triplet (total_players : ℕ) (triplets : ℕ) (starters : ℕ) :
  total_players = 12 →
  triplets = 3 →
  starters = 5 →
  (Nat.choose total_players starters) - (Nat.choose (total_players - triplets) starters) = 666 :=
by sorry

end starters_with_triplet_l719_71997


namespace reciprocal_plus_two_product_l719_71948

theorem reciprocal_plus_two_product (x y : ℝ) : 
  x ≠ y → x = 1/x + 2 → y = 1/y + 2 → x * y = -1 := by
  sorry

end reciprocal_plus_two_product_l719_71948


namespace problem_solution_l719_71930

theorem problem_solution (a : ℚ) : a + a / 3 + a / 4 = 4 → a = 48 / 19 := by
  sorry

end problem_solution_l719_71930


namespace power_of_three_even_tens_digit_l719_71992

def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

theorem power_of_three_even_tens_digit (n : ℕ) (h : n ≥ 3) :
  Even (tens_digit (3^n)) := by
  sorry

end power_of_three_even_tens_digit_l719_71992


namespace movies_needed_for_even_distribution_movie_store_problem_l719_71910

theorem movies_needed_for_even_distribution (total_movies : Nat) (num_shelves : Nat) : Nat :=
  let movies_per_shelf := total_movies / num_shelves
  let movies_needed := (movies_per_shelf + 1) * num_shelves - total_movies
  movies_needed

theorem movie_store_problem : movies_needed_for_even_distribution 2763 17 = 155 := by
  sorry

end movies_needed_for_even_distribution_movie_store_problem_l719_71910


namespace binary_1101110_equals_3131_base4_l719_71923

/-- Converts a binary number to its decimal representation -/
def binary_to_decimal (binary : List Bool) : ℕ :=
  binary.foldr (λ b acc => 2 * acc + if b then 1 else 0) 0

/-- Converts a decimal number to its base 4 representation -/
def decimal_to_base4 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
    if m = 0 then acc
    else aux (m / 4) ((m % 4) :: acc)
  aux n []

/-- The binary representation of 1101110 -/
def binary_1101110 : List Bool := [true, true, false, true, true, true, false]

theorem binary_1101110_equals_3131_base4 :
  decimal_to_base4 (binary_to_decimal binary_1101110) = [3, 1, 3, 1] := by
  sorry

end binary_1101110_equals_3131_base4_l719_71923


namespace function_inequality_and_ratio_l719_71974

-- Define the function f
def f (x : ℝ) : ℝ := |x + 1| + |x - 2|

-- Define the maximum value T
def T : ℝ := 3

-- Theorem statement
theorem function_inequality_and_ratio (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a^2 + b^2 = T) :
  (∀ x : ℝ, f x ≥ T) ∧ (2 / (1/a + 1/b) ≤ Real.sqrt 6 / 2) := by
  sorry

end function_inequality_and_ratio_l719_71974


namespace not_prime_5n_plus_3_l719_71938

theorem not_prime_5n_plus_3 (n : ℕ) (h1 : ∃ a : ℕ, 2 * n + 1 = a ^ 2) (h2 : ∃ b : ℕ, 3 * n + 1 = b ^ 2) : 
  ¬ Nat.Prime (5 * n + 3) := by
sorry

end not_prime_5n_plus_3_l719_71938


namespace vorontsova_dashkova_lifespan_l719_71959

/-- Represents a person's lifespan across two centuries -/
structure Lifespan where
  total : ℕ
  diff_18th_19th : ℕ
  years_19th : ℕ
  birth_year : ℕ
  death_year : ℕ

/-- Theorem about E.P. Vorontsova-Dashkova's lifespan -/
theorem vorontsova_dashkova_lifespan :
  ∃ (l : Lifespan),
    l.total = 66 ∧
    l.diff_18th_19th = 46 ∧
    l.years_19th = 10 ∧
    l.birth_year = 1744 ∧
    l.death_year = 1810 ∧
    l.total = l.years_19th + (l.years_19th + l.diff_18th_19th) ∧
    l.birth_year + l.total = l.death_year ∧
    l.birth_year + (l.total - l.years_19th) = 1800 :=
by
  sorry


end vorontsova_dashkova_lifespan_l719_71959


namespace solution_set_when_a_is_one_range_of_a_given_subset_l719_71967

-- Define the function f
def f (a x : ℝ) : ℝ := |x + a| + |2*x + 1|

-- Part 1
theorem solution_set_when_a_is_one :
  {x : ℝ | f 1 x ≤ 1} = {x : ℝ | -1 ≤ x ∧ x ≤ -1/3} :=
sorry

-- Part 2
def P (a : ℝ) : Set ℝ := {x : ℝ | f a x ≤ -2*x + 1}

theorem range_of_a_given_subset :
  (∀ a : ℝ, Set.Icc (-1 : ℝ) (-1/4) ⊆ P a) →
  {a : ℝ | ∀ x ∈ Set.Icc (-1 : ℝ) (-1/4), f a x ≤ -2*x + 1} = Set.Icc (-3/4 : ℝ) (5/4) :=
sorry

end solution_set_when_a_is_one_range_of_a_given_subset_l719_71967


namespace problem_curve_is_line_segment_l719_71933

/-- A parametric curve in 2D space -/
structure ParametricCurve where
  x : ℝ → ℝ
  y : ℝ → ℝ
  t_min : ℝ
  t_max : ℝ

/-- Definition of a line segment -/
def IsLineSegment (curve : ParametricCurve) : Prop :=
  ∃ (a b : ℝ × ℝ),
    (∀ t, curve.t_min ≤ t ∧ t ≤ curve.t_max →
      (curve.x t, curve.y t) = ((1 - t) • a.1 + t • b.1, (1 - t) • a.2 + t • b.2))

/-- The specific parametric curve from the problem -/
def ProblemCurve : ParametricCurve where
  x := λ t => 2 * t
  y := λ _ => 2
  t_min := -1
  t_max := 1

/-- Theorem stating that the problem curve is a line segment -/
theorem problem_curve_is_line_segment : IsLineSegment ProblemCurve := by
  sorry


end problem_curve_is_line_segment_l719_71933


namespace power_difference_mod_six_l719_71931

theorem power_difference_mod_six :
  (47^2045 - 18^2045) % 6 = 5 := by
  sorry

end power_difference_mod_six_l719_71931


namespace function_lower_bound_l719_71911

theorem function_lower_bound (a : ℝ) (h : a > 0) :
  ∀ x : ℝ, |x - 1/a| + |x + a| ≥ 2 := by sorry

end function_lower_bound_l719_71911


namespace unique_solution_implies_coefficients_l719_71920

theorem unique_solution_implies_coefficients
  (a b : ℚ)
  (h1 : ∀ x y : ℚ, a * x + y = 2 ∧ x + b * y = 2 ↔ x = 2 ∧ y = 1) :
  a = 1/2 ∧ b = 0 := by
sorry

end unique_solution_implies_coefficients_l719_71920


namespace functional_equation_solution_l719_71960

/-- A function satisfying the given functional equation -/
def SatisfiesFunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x * f x - y * f y = (x - y) * f (x + y)

/-- The main theorem stating that any function satisfying the functional equation
    is of the form f(x) = ax + b for some real a and b -/
theorem functional_equation_solution (f : ℝ → ℝ) 
  (h : SatisfiesFunctionalEquation f) : 
  ∃ a b : ℝ, ∀ x : ℝ, f x = a * x + b := by
  sorry

end functional_equation_solution_l719_71960


namespace railing_distance_proof_l719_71934

/-- The distance between two railings with bicycles placed between them -/
def railing_distance (interval_distance : ℕ) (num_bicycles : ℕ) : ℕ :=
  interval_distance * (num_bicycles - 1)

/-- Theorem: The distance between two railings is 95 meters -/
theorem railing_distance_proof :
  railing_distance 5 19 = 95 := by
  sorry

end railing_distance_proof_l719_71934


namespace solution_value_l719_71904

theorem solution_value (a b : ℝ) (h : a - 2*b = 7) : -a + 2*b + 1 = -6 := by
  sorry

end solution_value_l719_71904


namespace amy_school_year_hours_l719_71942

/-- Calculates the required weekly hours for Amy's school year work --/
def school_year_weekly_hours (summer_weeks : ℕ) (summer_hours_per_week : ℕ) (summer_earnings : ℕ) 
  (school_year_weeks : ℕ) (school_year_target : ℕ) : ℕ :=
  let hourly_wage := summer_earnings / (summer_weeks * summer_hours_per_week)
  let total_hours_needed := school_year_target / hourly_wage
  total_hours_needed / school_year_weeks

/-- Theorem stating that Amy needs to work 15 hours per week during the school year --/
theorem amy_school_year_hours : 
  school_year_weekly_hours 8 40 3200 32 4800 = 15 := by
  sorry

end amy_school_year_hours_l719_71942


namespace sons_age_l719_71955

theorem sons_age (father_age son_age : ℕ) : 
  father_age = son_age + 16 →
  father_age + 2 = 2 * (son_age + 2) →
  son_age = 14 := by
sorry

end sons_age_l719_71955


namespace gold_cube_comparison_l719_71993

/-- Represents the properties of a cube of gold -/
structure GoldCube where
  side_length : ℝ
  weight : ℝ
  value : ℝ

/-- Theorem stating the relationship between two gold cubes of different sizes -/
theorem gold_cube_comparison (small_cube large_cube : GoldCube) :
  small_cube.side_length = 4 →
  small_cube.weight = 5 →
  small_cube.value = 1200 →
  large_cube.side_length = 6 →
  (large_cube.weight = 16.875 ∧ large_cube.value = 4050) :=
by
  sorry

#check gold_cube_comparison

end gold_cube_comparison_l719_71993


namespace bankers_discount_example_l719_71982

/-- Calculates the banker's discount given the face value and true discount of a bill -/
def bankers_discount (face_value : ℚ) (true_discount : ℚ) : ℚ :=
  let present_value := face_value - true_discount
  (true_discount * face_value) / present_value

/-- Theorem: Given a bill with face value 2460 and true discount 360, the banker's discount is 422 -/
theorem bankers_discount_example : bankers_discount 2460 360 = 422 := by
  sorry

#eval bankers_discount 2460 360

end bankers_discount_example_l719_71982


namespace least_subtrahend_for_divisibility_problem_solution_l719_71987

theorem least_subtrahend_for_divisibility (n : ℕ) (d : ℕ) (h : d > 0) :
  ∃ (k : ℕ), k < d ∧ (n - k) % d = 0 ∧ ∀ (m : ℕ), m < k → (n - m) % d ≠ 0 :=
by sorry

theorem problem_solution :
  let n := 13603
  let d := 87
  ∃ (k : ℕ), k < d ∧ (n - k) % d = 0 ∧ ∀ (m : ℕ), m < k → (n - m) % d ≠ 0 ∧ k = 31 :=
by sorry

end least_subtrahend_for_divisibility_problem_solution_l719_71987


namespace find_k_l719_71970

theorem find_k : ∃ k : ℝ, (5 * 2 - k * 3 - 7 = 0) ∧ k = 1 := by
  sorry

end find_k_l719_71970


namespace hotdogs_day1_proof_l719_71972

/-- Represents the price of a hamburger in dollars -/
def hamburger_price : ℚ := 2

/-- Represents the price of a hot dog in dollars -/
def hotdog_price : ℚ := 1

/-- Represents the number of hamburgers bought on day 1 -/
def hamburgers_day1 : ℕ := 3

/-- Represents the number of hamburgers bought on day 2 -/
def hamburgers_day2 : ℕ := 2

/-- Represents the number of hot dogs bought on day 2 -/
def hotdogs_day2 : ℕ := 3

/-- Represents the total cost of purchases on day 1 in dollars -/
def total_cost_day1 : ℚ := 10

/-- Represents the total cost of purchases on day 2 in dollars -/
def total_cost_day2 : ℚ := 7

/-- Calculates the number of hot dogs bought on day 1 -/
def hotdogs_day1 : ℕ := 4

theorem hotdogs_day1_proof : 
  hamburgers_day1 * hamburger_price + hotdogs_day1 * hotdog_price = total_cost_day1 ∧
  hamburgers_day2 * hamburger_price + hotdogs_day2 * hotdog_price = total_cost_day2 :=
by sorry

end hotdogs_day1_proof_l719_71972


namespace max_digit_sum_is_37_l719_71940

/-- Represents a two-digit display --/
structure TwoDigitDisplay where
  tens : Nat
  ones : Nat
  valid : tens ≤ 9 ∧ ones ≤ 9

/-- Represents a time display in 12-hour format --/
structure TimeDisplay where
  hours : TwoDigitDisplay
  minutes : TwoDigitDisplay
  seconds : TwoDigitDisplay
  valid_hours : hours.tens * 10 + hours.ones ≥ 1 ∧ hours.tens * 10 + hours.ones ≤ 12
  valid_minutes : minutes.tens * 10 + minutes.ones ≤ 59
  valid_seconds : seconds.tens * 10 + seconds.ones ≤ 59

/-- Calculates the sum of digits in a TwoDigitDisplay --/
def digitSum (d : TwoDigitDisplay) : Nat :=
  d.tens + d.ones

/-- Calculates the total sum of digits in a TimeDisplay --/
def totalDigitSum (t : TimeDisplay) : Nat :=
  digitSum t.hours + digitSum t.minutes + digitSum t.seconds

/-- The maximum possible sum of digits in a 12-hour format digital watch display --/
def maxDigitSum : Nat := 37

/-- Theorem: The maximum sum of digits in a 12-hour format digital watch display is 37 --/
theorem max_digit_sum_is_37 :
  ∀ t : TimeDisplay, totalDigitSum t ≤ maxDigitSum :=
by
  sorry  -- The proof would go here

#check max_digit_sum_is_37

end max_digit_sum_is_37_l719_71940


namespace concatenated_integers_divisible_by_55_l719_71945

def concatenate_integers (n : ℕ) : ℕ :=
  -- Definition of concatenating integers from 1 to n
  sorry

theorem concatenated_integers_divisible_by_55 :
  ∃ k : ℕ, concatenate_integers 55 = 55 * k := by
  sorry

end concatenated_integers_divisible_by_55_l719_71945


namespace ellipse_equation_l719_71953

/-- The standard equation of an ellipse with foci on the coordinate axes and passing through points A(√3, -2) and B(-2√3, 1) is x²/15 + y²/5 = 1 -/
theorem ellipse_equation (x y : ℝ) : 
  (∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a ≠ b ∧
    (x^2 / a^2 + y^2 / b^2 = 1 ↔ x^2 / 15 + y^2 / 5 = 1)) ∧
  (x^2 / 15 + y^2 / 5 = 1 → x = Real.sqrt 3 ∧ y = -2 ∨ x = -2 * Real.sqrt 3 ∧ y = 1) :=
by sorry

end ellipse_equation_l719_71953


namespace target_hit_probability_l719_71901

/-- The binomial probability function -/
def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (n.choose k) * p^k * (1 - p)^(n - k)

/-- The problem statement -/
theorem target_hit_probability :
  let n : ℕ := 6
  let k : ℕ := 5
  let p : ℝ := 0.8
  abs (binomial_probability n k p - 0.3932) < 0.00005 := by
  sorry

end target_hit_probability_l719_71901


namespace quadratic_roots_sum_product_l719_71939

theorem quadratic_roots_sum_product (α β : ℝ) : 
  α^2 + α - 1 = 0 → β^2 + β - 1 = 0 → α ≠ β → α*β + α + β = -2 := by
  sorry

end quadratic_roots_sum_product_l719_71939


namespace customs_waiting_time_l719_71966

/-- The time Jack waited to get through customs, given total waiting time and quarantine days. -/
theorem customs_waiting_time (total_hours quarantine_days : ℕ) : 
  total_hours = 356 ∧ quarantine_days = 14 → 
  total_hours - (quarantine_days * 24) = 20 := by
  sorry

end customs_waiting_time_l719_71966


namespace return_trip_time_l719_71913

/-- Represents the flight scenario between two towns -/
structure FlightScenario where
  d : ℝ  -- distance between towns
  p : ℝ  -- speed of plane in still air
  w : ℝ  -- speed of wind
  against_wind_time : ℝ  -- time of flight against wind
  still_air_time : ℝ  -- time of flight in still air

/-- The flight conditions as given in the problem -/
def flight_conditions (s : FlightScenario) : Prop :=
  s.against_wind_time = 90 ∧
  s.d = s.against_wind_time * (s.p - s.w) ∧
  s.d / (s.p + s.w) = s.still_air_time - 15

/-- The theorem stating that the return trip takes either 30 or 45 minutes -/
theorem return_trip_time (s : FlightScenario) :
  flight_conditions s →
  (s.d / (s.p + s.w) = 30 ∨ s.d / (s.p + s.w) = 45) :=
by sorry

end return_trip_time_l719_71913


namespace eugene_purchase_cost_l719_71922

def tshirt_price : ℚ := 20
def pants_price : ℚ := 80
def shoes_price : ℚ := 150
def hat_price : ℚ := 25
def jacket_price : ℚ := 120

def tshirt_discount : ℚ := 0.1
def pants_discount : ℚ := 0.1
def shoes_discount : ℚ := 0.15
def hat_discount : ℚ := 0.05
def jacket_discount : ℚ := 0.2

def sales_tax : ℚ := 0.06

def tshirt_quantity : ℕ := 4
def pants_quantity : ℕ := 3
def shoes_quantity : ℕ := 2
def hat_quantity : ℕ := 3
def jacket_quantity : ℕ := 1

theorem eugene_purchase_cost :
  let discounted_tshirt := tshirt_price * (1 - tshirt_discount)
  let discounted_pants := pants_price * (1 - pants_discount)
  let discounted_shoes := shoes_price * (1 - shoes_discount)
  let discounted_hat := hat_price * (1 - hat_discount)
  let discounted_jacket := jacket_price * (1 - jacket_discount)
  
  let total_before_tax := 
    discounted_tshirt * tshirt_quantity +
    discounted_pants * pants_quantity +
    discounted_shoes * shoes_quantity +
    discounted_hat * hat_quantity +
    discounted_jacket * jacket_quantity
  
  let total_with_tax := total_before_tax * (1 + sales_tax)
  
  total_with_tax = 752.87 := by sorry

end eugene_purchase_cost_l719_71922


namespace polygon_count_l719_71950

theorem polygon_count (n : ℕ) (h : n = 15) : 
  2^n - (n.choose 0 + n.choose 1 + n.choose 2 + n.choose 3) = 32192 :=
by sorry

end polygon_count_l719_71950


namespace wheel_configuration_theorem_l719_71958

/-- Represents a wheel with spokes -/
structure Wheel where
  spokes : ℕ

/-- Represents a configuration of wheels -/
structure WheelConfiguration where
  wheels : List Wheel
  total_spokes : ℕ
  max_visible_spokes : ℕ

/-- Checks if a configuration is valid based on the problem conditions -/
def is_valid_configuration (config : WheelConfiguration) : Prop :=
  config.total_spokes ≥ 7 ∧
  config.max_visible_spokes ≤ 3 ∧
  (∀ w ∈ config.wheels, w.spokes ≤ config.max_visible_spokes)

theorem wheel_configuration_theorem :
  ∃ (config_three : WheelConfiguration),
    config_three.wheels.length = 3 ∧
    is_valid_configuration config_three ∧
  ¬∃ (config_two : WheelConfiguration),
    config_two.wheels.length = 2 ∧
    is_valid_configuration config_two :=
sorry

end wheel_configuration_theorem_l719_71958


namespace right_triangle_hypotenuse_l719_71956

theorem right_triangle_hypotenuse (a b : ℝ) (ha : a = 24) (hb : b = 32) :
  Real.sqrt (a^2 + b^2) = 40 := by sorry

end right_triangle_hypotenuse_l719_71956


namespace local_minimum_implies_a_eq_neg_two_monotone_increasing_implies_a_nonneg_l719_71963

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x + a) * Real.exp x

theorem local_minimum_implies_a_eq_neg_two (a : ℝ) :
  (∃ δ > 0, ∀ x, |x - 1| < δ → f a x ≥ f a 1) → a = -2 := by sorry

theorem monotone_increasing_implies_a_nonneg (a : ℝ) :
  (∀ x y, -1 < x ∧ x < y ∧ y < 1 → f a x < f a y) → a ≥ 0 := by sorry

end local_minimum_implies_a_eq_neg_two_monotone_increasing_implies_a_nonneg_l719_71963


namespace log_50_between_consecutive_integers_l719_71925

theorem log_50_between_consecutive_integers :
  ∃ (a b : ℤ), (a + 1 = b) ∧ (a < Real.log 50 / Real.log 10) ∧ (Real.log 50 / Real.log 10 < b) ∧ (a + b = 3) := by
  sorry

end log_50_between_consecutive_integers_l719_71925


namespace one_million_divided_by_one_fourth_l719_71914

theorem one_million_divided_by_one_fourth : 
  (1000000 : ℝ) / (1/4 : ℝ) = 4000000 := by sorry

end one_million_divided_by_one_fourth_l719_71914


namespace divisor_problem_l719_71906

theorem divisor_problem (n d : ℕ) (h1 : n % d = 3) (h2 : (n^2) % d = 4) : d = 5 := by
  sorry

end divisor_problem_l719_71906


namespace no_upper_limit_for_q_q_determines_side_ratio_l719_71978

/-- Represents a rectangle with side lengths a and b -/
structure Rectangle where
  a : ℝ
  b : ℝ
  h : 0 < a ∧ 0 < b

/-- The area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ := r.a * r.b

/-- The combined area of two rotated congruent rectangles -/
noncomputable def combined_area (r : Rectangle) : ℝ :=
  if r.b / r.a ≥ Real.sqrt 2 - 1 then
    (1 - 1 / Real.sqrt 2) * (r.a + r.b)^2
  else
    2 * r.a * r.b - Real.sqrt 2 * r.b^2

/-- The ratio of combined area to single rectangle area -/
noncomputable def area_ratio (r : Rectangle) : ℝ :=
  combined_area r / r.area

theorem no_upper_limit_for_q :
  ∀ M : ℝ, ∃ r : Rectangle, area_ratio r > M :=
sorry

theorem q_determines_side_ratio {r : Rectangle} (h : Real.sqrt 2 ≤ area_ratio r ∧ area_ratio r < 2) :
  r.b / r.a = (2 - area_ratio r) / Real.sqrt 2 :=
sorry

end no_upper_limit_for_q_q_determines_side_ratio_l719_71978


namespace hexagon_probability_l719_71962

/-- Represents a hexagonal checkerboard -/
structure HexBoard :=
  (total_hexagons : ℕ)
  (side_length : ℕ)

/-- Calculates the number of hexagons on the perimeter of the board -/
def perimeter_hexagons (board : HexBoard) : ℕ :=
  6 * board.side_length - 6

/-- Calculates the number of hexagons not on the perimeter of the board -/
def inner_hexagons (board : HexBoard) : ℕ :=
  board.total_hexagons - perimeter_hexagons board

/-- Theorem: The probability of a randomly chosen hexagon not touching the outer edge -/
theorem hexagon_probability (board : HexBoard) 
  (h1 : board.total_hexagons = 91)
  (h2 : board.side_length = 5) :
  (inner_hexagons board : ℚ) / board.total_hexagons = 67 / 91 := by
  sorry

end hexagon_probability_l719_71962


namespace record_collection_problem_l719_71971

theorem record_collection_problem (shared_records : ℕ) (emily_total : ℕ) (mark_unique : ℕ) : 
  shared_records = 15 → emily_total = 25 → mark_unique = 10 →
  emily_total - shared_records + mark_unique = 20 := by
sorry

end record_collection_problem_l719_71971


namespace original_car_cost_l719_71924

/-- Proves that the original cost of a car is 39200 given the specified conditions -/
theorem original_car_cost (C : ℝ) : 
  C > 0 →  -- Ensure the cost is positive
  (68400 - (C + 8000)) / C * 100 = 54.054054054054056 →
  C = 39200 := by
  sorry

end original_car_cost_l719_71924
