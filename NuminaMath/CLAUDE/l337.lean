import Mathlib

namespace bus_rental_optimization_l337_33706

theorem bus_rental_optimization (total_people : ℕ) (small_bus_seats small_bus_cost : ℕ)
  (large_bus_seats large_bus_cost : ℕ) (total_buses : ℕ) :
  total_people = 600 →
  small_bus_seats = 32 →
  large_bus_seats = 45 →
  small_bus_cost + 2 * large_bus_cost = 2800 →
  large_bus_cost = (125 * small_bus_cost) / 100 →
  total_buses = 14 →
  ∃ (small_buses large_buses : ℕ),
    small_buses + large_buses = total_buses ∧
    small_buses * small_bus_seats + large_buses * large_bus_seats ≥ total_people ∧
    small_buses * small_bus_cost + large_buses * large_bus_cost = 13600 ∧
    ∀ (other_small other_large : ℕ),
      other_small + other_large = total_buses →
      other_small * small_bus_seats + other_large * large_bus_seats ≥ total_people →
      other_small * small_bus_cost + other_large * large_bus_cost ≥ 13600 :=
by sorry

end bus_rental_optimization_l337_33706


namespace expected_fib_value_l337_33779

-- Define Fibonacci sequence
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

-- Define the probability of getting tails on both coins
def p_both_tails : ℚ := 1 / 4

-- Define the probability of not getting tails on both coins
def p_not_both_tails : ℚ := 3 / 4

-- Define the expected value of Fₖ
def expected_fib : ℚ := 19 / 11

-- Theorem statement
theorem expected_fib_value :
  ∃ (S : ℕ → ℚ), 
    (∀ n, S n = p_both_tails * fib n + p_not_both_tails * S (n + 1)) ∧
    (S 0 = expected_fib) := by
  sorry

end expected_fib_value_l337_33779


namespace cos_72_sin_78_plus_sin_72_sin_12_equals_half_l337_33792

theorem cos_72_sin_78_plus_sin_72_sin_12_equals_half :
  Real.cos (72 * π / 180) * Real.sin (78 * π / 180) +
  Real.sin (72 * π / 180) * Real.sin (12 * π / 180) = 1 / 2 := by
  sorry

end cos_72_sin_78_plus_sin_72_sin_12_equals_half_l337_33792


namespace total_water_intake_l337_33787

def morning_water : Real := 0.26
def afternoon_water : Real := 0.37

theorem total_water_intake : morning_water + afternoon_water = 0.63 := by
  sorry

end total_water_intake_l337_33787


namespace pond_draining_time_l337_33710

theorem pond_draining_time (total_volume : ℝ) (pump_rate : ℝ) (h1 : pump_rate > 0) : 
  total_volume = 15 * 24 * pump_rate →
  ∃ (remaining_time : ℝ),
    remaining_time ≥ 144 ∧
    3 * 24 * pump_rate + remaining_time * (2 * pump_rate) = total_volume :=
by sorry

end pond_draining_time_l337_33710


namespace no_chord_length_8_in_circle_radius_3_l337_33760

theorem no_chord_length_8_in_circle_radius_3 (r : ℝ) (chord_length : ℝ) :
  r = 3 → chord_length ≤ 2 * r → chord_length ≠ 8 := by
  sorry

end no_chord_length_8_in_circle_radius_3_l337_33760


namespace intersection_of_powers_of_two_and_three_l337_33742

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

def middle_digit (n : ℕ) : ℕ := (n / 10) % 10

theorem intersection_of_powers_of_two_and_three :
  ∃! d : ℕ, 
    (∃ m : ℕ, is_three_digit (2^m) ∧ middle_digit (2^m) = d) ∧
    (∃ n : ℕ, is_three_digit (3^n) ∧ middle_digit (3^n) = d) :=
sorry

end intersection_of_powers_of_two_and_three_l337_33742


namespace triangle_sum_theorem_l337_33785

theorem triangle_sum_theorem (a b c : ℝ) (h_positive : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) (h_angle : c^2 = a^2 + b^2 - a*b) :
  a / (b + c) + b / (c + a) = 1 := by
sorry

end triangle_sum_theorem_l337_33785


namespace soccer_goals_product_l337_33794

def first_ten_games : List Nat := [5, 2, 4, 3, 6, 2, 7, 4, 1, 3]

def goals_sum (games : List Nat) : Nat :=
  games.sum

def is_integer (n : ℚ) : Prop :=
  ∃ m : ℤ, n = m

theorem soccer_goals_product :
  ∀ (g11 g12 : Nat),
    g11 < 10 →
    g12 < 10 →
    is_integer ((goals_sum first_ten_games + g11) / 11) →
    is_integer ((goals_sum first_ten_games + g11 + g12) / 12) →
    g11 * g12 = 28 :=
by sorry

end soccer_goals_product_l337_33794


namespace continuous_functional_equation_solution_l337_33754

/-- A function that satisfies the given functional equation and condition -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  (∀ x y, 4 * f (x + y) = f x * f y) ∧ f 1 = 12

theorem continuous_functional_equation_solution 
  (f : ℝ → ℝ) 
  (hf : Continuous f) 
  (heq : FunctionalEquation f) : 
  ∀ x, f x = 4 * (3 : ℝ) ^ x := by
  sorry

end continuous_functional_equation_solution_l337_33754


namespace sequence_sum_l337_33798

theorem sequence_sum (x : Fin 10 → ℝ) 
  (h : ∀ i : Fin 9, x i + 2 * x (i.succ) = 1) :
  x 0 + 512 * x 9 = 171 := by
  sorry

end sequence_sum_l337_33798


namespace cubic_root_sum_product_l337_33796

theorem cubic_root_sum_product (p q r : ℂ) : 
  (3 * p^3 - 5 * p^2 + 12 * p - 10 = 0) ∧
  (3 * q^3 - 5 * q^2 + 12 * q - 10 = 0) ∧
  (3 * r^3 - 5 * r^2 + 12 * r - 10 = 0) →
  p * q + q * r + r * p = 4 := by
sorry

end cubic_root_sum_product_l337_33796


namespace quadratic_form_nonnegative_l337_33782

theorem quadratic_form_nonnegative (x y : ℝ) : x^2 + x*y + y^2 ≥ 0 := by
  sorry

end quadratic_form_nonnegative_l337_33782


namespace sum_proper_divisors_30_l337_33717

/-- The sum of the proper divisors of 30 is 42. -/
theorem sum_proper_divisors_30 : (Finset.filter (λ x => x < 30 ∧ 30 % x = 0) (Finset.range 30)).sum id = 42 := by
  sorry

end sum_proper_divisors_30_l337_33717


namespace range_of_a_l337_33793

-- Define the condition from the problem
def condition (a : ℝ) : Prop :=
  ∀ x : ℝ, 0 < x ∧ x ≤ 1/2 → (2 : ℝ)^(-2*x) - Real.log x / Real.log a < 0

-- State the theorem
theorem range_of_a (a : ℝ) : condition a → 1/4 < a ∧ a < 1 := by
  sorry

end range_of_a_l337_33793


namespace function_properties_l337_33739

noncomputable def f (x : ℝ) : ℝ := 2 * (Real.cos (x / 2))^2 - Real.sqrt 3 * Real.sin x

theorem function_properties :
  ∃ (a : ℝ),
    (π / 2 < a ∧ a < π) ∧
    f (a - π / 3) = 1 / 3 →
    (∀ (x : ℝ), f (x + 2 * π) = f x) ∧
    (∀ (y : ℝ), -1 ≤ y ∧ y ≤ 3 ↔ ∃ (x : ℝ), f x = y) ∧
    (Real.cos (2 * a)) / (1 + Real.cos (2 * a) - Real.sin (2 * a)) = (1 - 2 * Real.sqrt 2) / 2 :=
by sorry

end function_properties_l337_33739


namespace binomial_square_condition_l337_33789

theorem binomial_square_condition (b : ℚ) : 
  (∃ (p q : ℚ), ∀ x, b * x^2 + 20 * x + 9 = (p * x + q)^2) → b = 100 / 9 := by
  sorry

end binomial_square_condition_l337_33789


namespace complement_A_union_B_eq_R_A_union_B_eq_A_l337_33790

-- Define sets A and B
def A : Set ℝ := {x : ℝ | 0 ≤ x ∧ x ≤ 2}
def B (a : ℝ) : Set ℝ := {x : ℝ | a ≤ x ∧ x ≤ 3 - 2*a}

-- Part 1
theorem complement_A_union_B_eq_R (a : ℝ) :
  (Set.univ \ A) ∪ B a = Set.univ ↔ a ≤ 0 := by sorry

-- Part 2
theorem A_union_B_eq_A (a : ℝ) :
  A ∪ B a = A ↔ a ≥ 1/2 := by sorry

end complement_A_union_B_eq_R_A_union_B_eq_A_l337_33790


namespace aria_apple_weeks_l337_33746

theorem aria_apple_weeks (total_apples : ℕ) (apples_per_day : ℕ) (days_per_week : ℕ) : 
  total_apples = 14 →
  apples_per_day = 1 →
  days_per_week = 7 →
  (total_apples / (apples_per_day * days_per_week) : ℚ) = 2 := by
  sorry

end aria_apple_weeks_l337_33746


namespace max_visible_cubes_9x9x9_l337_33707

/-- Represents a cube made of unit cubes -/
structure UnitCube where
  size : ℕ

/-- Calculates the maximum number of visible unit cubes from a single point -/
def max_visible_cubes (cube : UnitCube) : ℕ :=
  let face_size := cube.size^2
  let edge_size := cube.size - 1
  3 * face_size - 3 * edge_size + 1

/-- Theorem: For a 9×9×9 cube, the maximum number of visible unit cubes is 220 -/
theorem max_visible_cubes_9x9x9 :
  max_visible_cubes ⟨9⟩ = 220 := by sorry

end max_visible_cubes_9x9x9_l337_33707


namespace sphere_surface_area_from_cuboid_l337_33703

theorem sphere_surface_area_from_cuboid (a : ℝ) (h : a > 0) :
  let cuboid_dimensions := (2*a, a, a)
  let sphere_radius := Real.sqrt (3/2 * a^2)
  let sphere_surface_area := 4 * Real.pi * sphere_radius^2
  sphere_surface_area = 6 * Real.pi * a^2 := by sorry

end sphere_surface_area_from_cuboid_l337_33703


namespace common_tangent_range_l337_33756

/-- Given two curves y = x^2 - 1 and y = a ln x - 1, where a is a positive real number,
    if there exists a common tangent line to both curves, then 0 < a ≤ 2e. -/
theorem common_tangent_range (a : ℝ) (h_pos : a > 0) :
  (∃ (x₁ x₂ : ℝ), x₁ > 0 ∧ x₂ > 0 ∧
    (2 * x₁ : ℝ) = a / x₂ ∧
    x₁^2 + 1 = a + 1 - a * Real.log x₂) →
  0 < a ∧ a ≤ 2 * Real.exp 1 :=
by sorry


end common_tangent_range_l337_33756


namespace perimeter_of_specific_shape_l337_33727

/-- A shape with three sides of equal length -/
structure ThreeSidedShape where
  side_length : ℝ
  num_sides : ℕ
  h_num_sides : num_sides = 3

/-- The perimeter of a three-sided shape -/
def perimeter (shape : ThreeSidedShape) : ℝ :=
  shape.side_length * shape.num_sides

/-- Theorem: The perimeter of a shape with 3 sides, each of length 7 cm, is 21 cm -/
theorem perimeter_of_specific_shape :
  ∃ (shape : ThreeSidedShape), shape.side_length = 7 ∧ perimeter shape = 21 := by
  sorry

end perimeter_of_specific_shape_l337_33727


namespace incorrect_number_calculation_l337_33737

theorem incorrect_number_calculation (n : ℕ) (initial_avg correct_avg incorrect_num : ℚ) :
  n = 10 ∧ 
  initial_avg = 16 ∧ 
  correct_avg = 18 ∧ 
  incorrect_num = 25 →
  ∃ actual_num : ℚ,
    n * initial_avg + actual_num = n * correct_avg ∧
    actual_num = incorrect_num - (correct_avg - initial_avg) * n ∧
    actual_num = 5 := by sorry

end incorrect_number_calculation_l337_33737


namespace m_range_when_p_or_q_false_l337_33757

theorem m_range_when_p_or_q_false (m : ℝ) :
  (¬ ((∃ x : ℝ, m * x^2 + 1 ≤ 0) ∨ (∀ x : ℝ, x^2 + m * x + 1 > 0))) →
  m ≥ 2 :=
by sorry

end m_range_when_p_or_q_false_l337_33757


namespace find_a_minus_b_l337_33753

-- Define the functions
def f (a b : ℝ) (x : ℝ) : ℝ := a * x + b
def g (x : ℝ) : ℝ := -3 * x + 5
def h (a b : ℝ) (x : ℝ) : ℝ := f a b (g x)

-- State the theorem
theorem find_a_minus_b (a b : ℝ) : 
  (∀ x, h a b x = x - 7) → a - b = 5 := by
  sorry

end find_a_minus_b_l337_33753


namespace jills_salary_l337_33772

theorem jills_salary (net_salary : ℝ) 
  (h1 : net_salary > 0)
  (discretionary_income : ℝ)
  (h2 : discretionary_income = net_salary / 5)
  (vacation_fund : ℝ)
  (h3 : vacation_fund = 0.3 * discretionary_income)
  (savings : ℝ)
  (h4 : savings = 0.2 * discretionary_income)
  (socializing : ℝ)
  (h5 : socializing = 0.35 * discretionary_income)
  (remaining : ℝ)
  (h6 : remaining = discretionary_income - vacation_fund - savings - socializing)
  (h7 : remaining = 105) :
  net_salary = 3500 := by
sorry

end jills_salary_l337_33772


namespace cos_four_arccos_one_fourth_l337_33768

theorem cos_four_arccos_one_fourth : 
  Real.cos (4 * Real.arccos (1/4)) = 17/32 := by sorry

end cos_four_arccos_one_fourth_l337_33768


namespace hexagonal_dish_volume_l337_33713

/-- Represents a pyramidal frustum formed by bending four regular hexagons attached to a square --/
structure HexagonalDish where
  side_length : ℝ
  volume : ℝ

/-- The volume of the hexagonal dish is √(49/12) cubic meters when the side length is 1 meter --/
theorem hexagonal_dish_volume (dish : HexagonalDish) (h1 : dish.side_length = 1) :
  dish.volume = Real.sqrt (49 / 12) := by
  sorry

#check hexagonal_dish_volume

end hexagonal_dish_volume_l337_33713


namespace third_grade_students_l337_33715

theorem third_grade_students (num_buses : ℕ) (seats_per_bus : ℕ) (empty_seats : ℕ) : 
  num_buses = 18 → seats_per_bus = 15 → empty_seats = 3 →
  (num_buses * (seats_per_bus - empty_seats) = 216) := by
sorry

end third_grade_students_l337_33715


namespace roots_equation_m_value_l337_33708

theorem roots_equation_m_value (α : ℝ) (m : ℝ) : 
  (∀ x, x^2 + 3*x + m = 0 ↔ x = 1/Real.cos α ∨ x = Real.tan α) →
  m = 20/9 := by
sorry

end roots_equation_m_value_l337_33708


namespace f_of_tan_squared_plus_one_l337_33761

noncomputable def f (x : ℝ) : ℝ :=
  1 / ((x / (x - 1)) - 1)

theorem f_of_tan_squared_plus_one (t : ℝ) (h : 0 ≤ t ∧ t ≤ π/2) :
  f (Real.tan t ^ 2 + 1) = (Real.sin (2 * t)) ^ 2 / 4 :=
by sorry

end f_of_tan_squared_plus_one_l337_33761


namespace rhombus_diagonal_length_l337_33704

/-- Represents a rhombus with given diagonals and area -/
structure Rhombus where
  d1 : ℝ
  d2 : ℝ
  area : ℝ
  area_eq : area = (d1 * d2) / 2

/-- Theorem: In a rhombus with area 110 cm² and one diagonal 11 cm, the other diagonal is 20 cm -/
theorem rhombus_diagonal_length (r : Rhombus) 
    (h1 : r.d1 = 11) 
    (h2 : r.area = 110) : 
    r.d2 = 20 := by
  sorry

end rhombus_diagonal_length_l337_33704


namespace min_value_equality_l337_33736

/-- Given a function f(x) = x^2 + ax, prove that the minimum value of f(f(x)) 
    is equal to the minimum value of f(x) if and only if a ≤ 0 or a ≥ 2. -/
theorem min_value_equality (a : ℝ) : 
  let f : ℝ → ℝ := λ x => x^2 + a*x
  (∃ m : ℝ, ∀ x : ℝ, f x ≥ m ∧ ∃ y : ℝ, f y = m) →
  (∃ n : ℝ, ∀ x : ℝ, f (f x) ≥ n ∧ ∃ y : ℝ, f (f y) = n) →
  (∃ k : ℝ, (∀ x : ℝ, f x ≥ k ∧ ∃ y : ℝ, f y = k) ∧
            (∀ x : ℝ, f (f x) ≥ k ∧ ∃ y : ℝ, f (f y) = k)) ↔
  (a ≤ 0 ∨ a ≥ 2) :=
by sorry

end min_value_equality_l337_33736


namespace micah_envelope_stamps_l337_33791

/-- Represents the stamp distribution problem for Micah's envelopes --/
theorem micah_envelope_stamps :
  ∀ (total_stamps : ℕ) 
    (total_envelopes : ℕ) 
    (light_envelopes : ℕ) 
    (stamps_per_light : ℕ) 
    (stamps_per_heavy : ℕ),
  total_stamps = 52 →
  total_envelopes = 14 →
  light_envelopes = 6 →
  stamps_per_light = 2 →
  stamps_per_heavy = 5 →
  total_stamps = light_envelopes * stamps_per_light + 
                 (total_envelopes - light_envelopes) * stamps_per_heavy :=
by
  sorry


end micah_envelope_stamps_l337_33791


namespace cookies_given_proof_l337_33776

/-- The number of cookies Paco gave to his friend -/
def cookies_given_to_friend : ℕ := sorry

/-- The initial number of cookies Paco had -/
def initial_cookies : ℕ := 41

/-- The number of cookies Paco ate -/
def cookies_eaten : ℕ := 18

theorem cookies_given_proof :
  cookies_given_to_friend = 9 ∧
  initial_cookies = cookies_given_to_friend + cookies_eaten + cookies_given_to_friend ∧
  cookies_eaten = cookies_given_to_friend + 9 :=
sorry

end cookies_given_proof_l337_33776


namespace twelve_digit_159_div37_not_sum76_l337_33702

-- Define a function to check if a number consists only of digits 1, 5, and 9
def only_159_digits (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d = 1 ∨ d = 5 ∨ d = 9

-- Define a function to calculate the sum of digits
def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

-- Theorem statement
theorem twelve_digit_159_div37_not_sum76 (n : ℕ) :
  n ≥ 10^11 ∧ n < 10^12 ∧ only_159_digits n ∧ n % 37 = 0 →
  sum_of_digits n ≠ 76 := by
  sorry

end twelve_digit_159_div37_not_sum76_l337_33702


namespace smallest_solution_equation_smallest_solution_l337_33740

theorem smallest_solution_equation (x : ℝ) :
  (1 / (x - 3) + 1 / (x - 5) = 5 / (2 * (x - 4))) ↔ x = 4 - Real.sqrt 5 ∨ x = 4 + Real.sqrt 5 :=
sorry

theorem smallest_solution (x : ℝ) :
  (1 / (x - 3) + 1 / (x - 5) = 5 / (2 * (x - 4))) ∧ 
  (∀ y : ℝ, (1 / (y - 3) + 1 / (y - 5) = 5 / (2 * (y - 4))) → y ≥ x) →
  x = 4 - Real.sqrt 5 :=
sorry

end smallest_solution_equation_smallest_solution_l337_33740


namespace sequence_property_l337_33732

theorem sequence_property (a : ℕ → ℕ) 
    (h_nondecreasing : ∀ n m : ℕ, n ≤ m → a n ≤ a m)
    (h_nonconstant : ∃ n m : ℕ, a n ≠ a m)
    (h_divides : ∀ n : ℕ, a n ∣ n^2) :
  (∃ n₁ : ℕ, ∀ n ≥ n₁, a n = n) ∨ 
  (∃ n₂ : ℕ, ∀ n ≥ n₂, a n = n^2) := by
sorry

end sequence_property_l337_33732


namespace arithmetic_sequence_second_term_l337_33774

def is_arithmetic_sequence (a b c : ℝ) : Prop := b - a = c - b

theorem arithmetic_sequence_second_term 
  (y : ℝ) 
  (h1 : y > 0) 
  (h2 : is_arithmetic_sequence (5^2) (y^2) (13^2)) : 
  y = Real.sqrt 97 := by
sorry

end arithmetic_sequence_second_term_l337_33774


namespace boys_passed_exam_l337_33724

theorem boys_passed_exam (total_boys : ℕ) (avg_all : ℚ) (avg_passed : ℚ) (avg_failed : ℚ) 
  (h_total : total_boys = 120)
  (h_avg_all : avg_all = 37)
  (h_avg_passed : avg_passed = 39)
  (h_avg_failed : avg_failed = 15) :
  ∃ (passed_boys : ℕ), 
    passed_boys = 110 ∧ 
    (passed_boys : ℚ) * avg_passed + (total_boys - passed_boys : ℚ) * avg_failed = 
      (total_boys : ℚ) * avg_all :=
by
  sorry

end boys_passed_exam_l337_33724


namespace min_lcm_ac_l337_33714

theorem min_lcm_ac (a b c : ℕ+) (hab : Nat.lcm a b = 12) (hbc : Nat.lcm b c = 15) :
  ∃ (a' b' c' : ℕ+), Nat.lcm a' b' = 12 ∧ Nat.lcm b' c' = 15 ∧ 
  Nat.lcm a' c' = 20 ∧ ∀ (x y : ℕ+), Nat.lcm x y ≥ 20 :=
by sorry

end min_lcm_ac_l337_33714


namespace bedroom_area_l337_33705

/-- Proves that the area of each bedroom is 121 square feet given the specified house layout --/
theorem bedroom_area (total_area : ℝ) (num_bedrooms : ℕ) (num_bathrooms : ℕ)
  (bathroom_length bathroom_width : ℝ) (kitchen_area : ℝ) :
  total_area = 1110 →
  num_bedrooms = 4 →
  num_bathrooms = 2 →
  bathroom_length = 6 →
  bathroom_width = 8 →
  kitchen_area = 265 →
  ∃ (bedroom_area : ℝ),
    bedroom_area = 121 ∧
    total_area = num_bedrooms * bedroom_area + 
                 num_bathrooms * bathroom_length * bathroom_width +
                 2 * kitchen_area :=
by
  sorry

end bedroom_area_l337_33705


namespace jorge_goals_last_season_l337_33741

/-- The number of goals Jorge scored last season -/
def goals_last_season : ℕ := sorry

/-- The number of goals Jorge scored this season -/
def goals_this_season : ℕ := 187

/-- The total number of goals Jorge scored -/
def total_goals : ℕ := 343

/-- Theorem stating that Jorge scored 156 goals last season -/
theorem jorge_goals_last_season :
  goals_last_season = total_goals - goals_this_season ∧
  goals_last_season = 156 := by sorry

end jorge_goals_last_season_l337_33741


namespace no_simultaneous_perfect_squares_l337_33781

theorem no_simultaneous_perfect_squares (a b : ℕ+) : 
  ¬(∃ (x y : ℕ), (a.val^2 + 4*b.val = x^2) ∧ (b.val^2 + 4*a.val = y^2)) := by
  sorry

end no_simultaneous_perfect_squares_l337_33781


namespace linear_function_problem_l337_33763

-- Define a linear function
def linear_function (k b : ℝ) : ℝ → ℝ := λ x => k * x + b

-- Define the problem
theorem linear_function_problem :
  ∃ (k b : ℝ),
    -- The function passes through (3, 2) and (1, -2)
    (linear_function k b 3 = 2) ∧
    (linear_function k b 1 = -2) ∧
    -- The function is f(x) = 2x - 4
    (k = 2 ∧ b = -4) ∧
    -- The points (5, 6) and (-5, -14) lie on the line
    (linear_function k b 5 = 6) ∧
    (linear_function k b (-5) = -14) ∧
    -- These points are 5 units away from the y-axis
    (5 = 5 ∨ 5 = -5) :=
by
  sorry

end linear_function_problem_l337_33763


namespace quadratic_real_roots_m_range_l337_33786

theorem quadratic_real_roots_m_range (m : ℝ) : 
  (∃ x : ℝ, x^2 + 2*x + m = 0) → m ≤ 1 := by
  sorry

end quadratic_real_roots_m_range_l337_33786


namespace gcd_factorial_eight_ten_l337_33799

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem gcd_factorial_eight_ten : Nat.gcd (factorial 8) (factorial 10) = factorial 8 := by
  sorry

end gcd_factorial_eight_ten_l337_33799


namespace french_fries_price_l337_33758

/-- The cost of french fries at the burger hut -/
def french_fries_cost : ℝ := 1.50

/-- The cost of a burger -/
def burger_cost : ℝ := 5

/-- The cost of a soft drink -/
def soft_drink_cost : ℝ := 3

/-- The cost of a special burger meal -/
def special_meal_cost : ℝ := 9.50

theorem french_fries_price : 
  french_fries_cost = special_meal_cost - (burger_cost + soft_drink_cost) := by
  sorry

end french_fries_price_l337_33758


namespace rahims_average_book_price_l337_33755

/-- Calculates the average price per book given two purchases -/
def average_price_per_book (books1 books2 : ℕ) (price1 price2 : ℚ) : ℚ :=
  (price1 + price2) / (books1 + books2)

/-- Theorem stating the average price per book for Rahim's purchases -/
theorem rahims_average_book_price :
  let books1 : ℕ := 65
  let books2 : ℕ := 50
  let price1 : ℚ := 1160
  let price2 : ℚ := 920
  let avg_price := average_price_per_book books1 books2 price1 price2
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1/100 ∧ |avg_price - 18.09| < ε :=
sorry

end rahims_average_book_price_l337_33755


namespace S_intersect_T_eq_T_l337_33783

def S : Set Int := {s | ∃ n : Int, s = 2 * n + 1}
def T : Set Int := {t | ∃ n : Int, t = 4 * n + 1}

theorem S_intersect_T_eq_T : S ∩ T = T := by sorry

end S_intersect_T_eq_T_l337_33783


namespace gcd_180_126_l337_33721

theorem gcd_180_126 : Nat.gcd 180 126 = 18 := by
  sorry

end gcd_180_126_l337_33721


namespace sum_x_y_equals_negative_three_l337_33788

theorem sum_x_y_equals_negative_three (x y : ℝ) :
  (3 * x - y + 5)^2 + |2 * x - y + 3| = 0 → x + y = -3 := by
  sorry

end sum_x_y_equals_negative_three_l337_33788


namespace pizza_combinations_l337_33762

theorem pizza_combinations (n : ℕ) (h : n = 8) : 
  n + Nat.choose n 2 + Nat.choose n 3 = 92 := by
  sorry

end pizza_combinations_l337_33762


namespace unique_root_implies_a_equals_three_a_equals_three_implies_unique_root_unique_root_iff_a_equals_three_l337_33701

/-- The function f(x) defined by the equation --/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a * abs x + a^2 - 9

/-- Theorem stating that a = 3 is the only value for which f has a unique root --/
theorem unique_root_implies_a_equals_three :
  ∀ a : ℝ, (∃! x : ℝ, f a x = 0) → a = 3 :=
by sorry

/-- Theorem stating that when a = 3, f has a unique root --/
theorem a_equals_three_implies_unique_root :
  ∃! x : ℝ, f 3 x = 0 :=
by sorry

/-- The main theorem combining the above results --/
theorem unique_root_iff_a_equals_three :
  ∀ a : ℝ, (∃! x : ℝ, f a x = 0) ↔ a = 3 :=
by sorry

end unique_root_implies_a_equals_three_a_equals_three_implies_unique_root_unique_root_iff_a_equals_three_l337_33701


namespace player_catches_ball_l337_33748

/-- Represents the motion of an object with uniform acceleration --/
structure UniformMotion where
  initial_velocity : ℝ
  acceleration : ℝ

/-- Calculates the distance traveled by an object with uniform motion --/
def distance (m : UniformMotion) (t : ℝ) : ℝ :=
  m.initial_velocity * t + 0.5 * m.acceleration * t^2

theorem player_catches_ball (ball_motion player_motion : UniformMotion)
  (initial_distance sideline_distance : ℝ) : 
  ball_motion.initial_velocity = 4.375 ∧ 
  ball_motion.acceleration = -0.75 ∧
  player_motion.initial_velocity = 3.25 ∧
  player_motion.acceleration = 0.5 ∧
  initial_distance = 10 ∧
  sideline_distance = 23 →
  ∃ (t : ℝ), t = 5 ∧ 
  distance ball_motion t = distance player_motion t + initial_distance ∧
  distance ball_motion t < sideline_distance :=
by sorry

#check player_catches_ball

end player_catches_ball_l337_33748


namespace correlation_coefficient_is_negative_one_l337_33750

/-- A structure representing a set of sample data points -/
structure SampleData where
  n : ℕ
  x : Fin n → ℝ
  y : Fin n → ℝ
  h_n : n ≥ 2
  h_not_all_equal : ∃ i j, i ≠ j ∧ x i ≠ x j
  h_on_line : ∀ i, y i = -1/2 * x i + 1

/-- The correlation coefficient of a set of sample data -/
def correlationCoefficient (data : SampleData) : ℝ := sorry

/-- Theorem stating that the correlation coefficient is -1 for the given conditions -/
theorem correlation_coefficient_is_negative_one (data : SampleData) :
  correlationCoefficient data = -1 := by sorry

end correlation_coefficient_is_negative_one_l337_33750


namespace playoff_average_points_l337_33767

/-- Represents a hockey team's record --/
structure TeamRecord where
  wins : ℕ
  ties : ℕ

/-- Calculates the points for a team given their record --/
def calculatePoints (record : TeamRecord) : ℕ :=
  2 * record.wins + record.ties

/-- Theorem: The average points of the playoff teams is 27 --/
theorem playoff_average_points :
  let team1 : TeamRecord := ⟨12, 4⟩
  let team2 : TeamRecord := ⟨13, 1⟩
  let team3 : TeamRecord := ⟨8, 10⟩
  let totalPoints := calculatePoints team1 + calculatePoints team2 + calculatePoints team3
  totalPoints / 3 = 27 := by
  sorry


end playoff_average_points_l337_33767


namespace sum_of_special_primes_is_prime_l337_33797

theorem sum_of_special_primes_is_prime (A B : ℕ+) : 
  Nat.Prime A.val → 
  Nat.Prime B.val → 
  Nat.Prime (A.val - B.val) → 
  Nat.Prime (A.val - B.val - B.val) → 
  Nat.Prime (A.val + B.val + (A.val - B.val) + (A.val - B.val - B.val)) := by
  sorry

end sum_of_special_primes_is_prime_l337_33797


namespace property_characterization_l337_33745

/-- The sum of digits of a natural number -/
def sum_of_digits (m : ℕ) : ℕ := sorry

/-- Predicate for numbers with the desired property -/
def has_property (n : ℕ) : Prop :=
  n > 1 ∧ ∀ k, 0 ≤ k ∧ k < n → ∃ m : ℕ, n ∣ m ∧ sum_of_digits m % n = k

theorem property_characterization (n : ℕ) :
  has_property n ↔ (n > 1 ∧ ¬(3 ∣ n)) :=
sorry

end property_characterization_l337_33745


namespace intersection_min_a_l337_33735

noncomputable section

-- Define the functions
def f (x : ℝ) : ℝ := (1/2)^x
def g (a x : ℝ) : ℝ := Real.log x / Real.log a

-- State the theorem
theorem intersection_min_a (a : ℝ) (x₀ y₀ : ℝ) 
  (h1 : a > 0) 
  (h2 : a ≠ 1) 
  (h3 : x₀ ≥ 2) 
  (h4 : f x₀ = g a x₀) 
  (h5 : y₀ = f x₀) :
  a ≥ 16 :=
sorry

end intersection_min_a_l337_33735


namespace missing_digit_divisible_by_nine_l337_33720

theorem missing_digit_divisible_by_nine :
  ∀ x : ℕ,
  x < 10 →
  (13507 + 100 * x) % 9 = 0 →
  x = 2 := by
sorry

end missing_digit_divisible_by_nine_l337_33720


namespace equation_root_sum_l337_33734

theorem equation_root_sum (m n : ℝ) : 
  n > 0 → 
  (1 + n * Complex.I) ^ 2 + m * (1 + n * Complex.I) + 2 = 0 → 
  m + n = -1 := by sorry

end equation_root_sum_l337_33734


namespace school_boys_count_l337_33722

/-- Given a school with more girls than boys, calculate the number of boys. -/
theorem school_boys_count (girls : ℕ) (difference : ℕ) (h1 : girls = 739) (h2 : difference = 402) :
  girls - difference = 337 := by
  sorry

end school_boys_count_l337_33722


namespace min_sum_for_product_1386_l337_33769

theorem min_sum_for_product_1386 (a b c : ℕ+) : 
  a * b * c = 1386 → 
  ∀ x y z : ℕ+, x * y * z = 1386 → a + b + c ≤ x + y + z ∧ ∃ a b c : ℕ+, a * b * c = 1386 ∧ a + b + c = 34 :=
by sorry

end min_sum_for_product_1386_l337_33769


namespace odd_product_probability_zero_l337_33770

-- Define the type for our grid
def Grid := Fin 3 → Fin 3 → Fin 9

-- Define a function to check if a number is odd
def isOdd (n : Fin 9) : Prop := n.val % 2 = 1

-- Define a function to check if a row has all odd numbers
def rowAllOdd (g : Grid) (row : Fin 3) : Prop :=
  ∀ col : Fin 3, isOdd (g row col)

-- Define our main theorem
theorem odd_product_probability_zero :
  ∀ g : Grid, (∀ row : Fin 3, rowAllOdd g row) → False :=
sorry

end odd_product_probability_zero_l337_33770


namespace nurse_distribution_count_l337_33747

/-- The number of hospitals --/
def num_hospitals : ℕ := 6

/-- The number of nurses --/
def num_nurses : ℕ := 3

/-- The maximum number of nurses allowed per hospital --/
def max_nurses_per_hospital : ℕ := 2

/-- The total number of possible nurse distributions --/
def total_distributions : ℕ := num_hospitals ^ num_nurses

/-- The number of invalid distributions (all nurses in one hospital) --/
def invalid_distributions : ℕ := num_hospitals

/-- The number of valid nurse distribution plans --/
def valid_distribution_plans : ℕ := total_distributions - invalid_distributions

theorem nurse_distribution_count :
  valid_distribution_plans = 210 :=
sorry

end nurse_distribution_count_l337_33747


namespace simplify_fraction_l337_33728

theorem simplify_fraction (a : ℝ) (ha : a ≠ 0) :
  1 - (1 / (1 + (a - 1) / (a + 1))) = (a - 1) / (2 * a) := by
  sorry

end simplify_fraction_l337_33728


namespace min_chords_for_complete_circuit_l337_33773

/-- Given two concentric circles with chords of the larger circle tangent to the smaller circle,
    if the angle between two adjacent chords is 60°, then the minimum number of chords needed
    to return to the starting point is 3. -/
theorem min_chords_for_complete_circuit (angle_between_chords : ℝ) : 
  angle_between_chords = 60 → (∃ n : ℕ, n * (180 - angle_between_chords) = 360 ∧ ∀ m : ℕ, m * (180 - angle_between_chords) = 360 → n ≤ m) → 
  (∃ n : ℕ, n * (180 - angle_between_chords) = 360 ∧ ∀ m : ℕ, m * (180 - angle_between_chords) = 360 → n ≤ m) ∧ 
  (∀ n : ℕ, n * (180 - angle_between_chords) = 360 → n ≥ 3) :=
by sorry

end min_chords_for_complete_circuit_l337_33773


namespace milk_container_problem_l337_33749

/-- Proves that the initial quantity of milk in container B was 37.5% of container A's capacity -/
theorem milk_container_problem (A B C : ℝ) : 
  A = 1200 →  -- Container A's capacity is 1200 liters
  B + C = A →  -- All milk from A was poured into B and C
  (B + 150 = C - 150) →  -- After transferring 150 liters from C to B, both containers have equal quantities
  B / A = 0.375 :=  -- The initial quantity in B was 37.5% of A's capacity
by sorry

end milk_container_problem_l337_33749


namespace units_digit_sum_cubes_l337_33751

theorem units_digit_sum_cubes : (24^3 + 17^3) % 10 = 7 := by
  sorry

end units_digit_sum_cubes_l337_33751


namespace power_multiplication_problem_solution_l337_33712

theorem power_multiplication (a : ℕ) (m n : ℕ) :
  a * (a ^ n) = a ^ (n + 1) :=
by sorry

theorem problem_solution : 
  2000 * (2000 ^ 2000) = 2000 ^ 2001 :=
by sorry

end power_multiplication_problem_solution_l337_33712


namespace max_value_range_l337_33726

noncomputable def f (a x : ℝ) : ℝ := ((1 - a) * x^2 - a * x + a) / Real.exp x

theorem max_value_range (a : ℝ) :
  (∀ x : ℝ, x ≥ 0 → f a x ≤ a) ↔ a ∈ Set.Ici (4 / (Real.exp 2 + 5)) :=
sorry

end max_value_range_l337_33726


namespace solve_system_of_equations_no_solution_for_inequalities_l337_33795

-- Part 1: System of equations
theorem solve_system_of_equations :
  ∃! (x y : ℚ), x + y = 5 ∧ 3 * x + 10 * y = 30 :=
by sorry

-- Part 2: System of inequalities
theorem no_solution_for_inequalities :
  ¬∃ (x : ℚ), (x + 7) / 2 < 4 ∧ (3 * x - 1) / 2 ≤ 2 * x - 3 :=
by sorry

end solve_system_of_equations_no_solution_for_inequalities_l337_33795


namespace average_age_of_ten_students_l337_33700

theorem average_age_of_ten_students
  (total_students : Nat)
  (total_average_age : ℝ)
  (nine_students_average : ℝ)
  (twentieth_student_age : ℝ)
  (h1 : total_students = 20)
  (h2 : total_average_age = 20)
  (h3 : nine_students_average = 11)
  (h4 : twentieth_student_age = 61) :
  let remaining_students := total_students - 10
  let total_age := total_students * total_average_age
  let nine_students_total_age := 9 * nine_students_average
  let ten_students_total_age := total_age - nine_students_total_age - twentieth_student_age
  ten_students_total_age / remaining_students = 24 := by
  sorry

end average_age_of_ten_students_l337_33700


namespace power_equality_l337_33738

theorem power_equality (m n : ℕ) (h1 : 3^m = 5) (h2 : 9^n = 10) : 3^(m+2*n) = 50 := by
  sorry

end power_equality_l337_33738


namespace max_quarters_is_thirteen_l337_33723

/-- The value of a quarter in dollars -/
def quarter_value : ℚ := 25 / 100

/-- The value of a dime in dollars -/
def dime_value : ℚ := 10 / 100

/-- The value of a nickel in dollars -/
def nickel_value : ℚ := 5 / 100

/-- The total amount Carlos has in dollars -/
def total_amount : ℚ := 545 / 100

/-- 
Given $5.45 in U.S. coins, with an equal number of quarters, dimes, and nickels,
prove that the maximum number of quarters (and thus dimes and nickels) is 13.
-/
theorem max_quarters_is_thirteen :
  ∃ (n : ℕ), n * (quarter_value + dime_value + nickel_value) ≤ total_amount ∧
             ∀ (m : ℕ), m * (quarter_value + dime_value + nickel_value) ≤ total_amount → m ≤ n ∧
             n = 13 :=
by sorry

end max_quarters_is_thirteen_l337_33723


namespace groom_age_l337_33716

theorem groom_age (bride_age groom_age : ℕ) : 
  bride_age = groom_age + 19 → 
  bride_age + groom_age = 185 → 
  groom_age = 83 := by sorry

end groom_age_l337_33716


namespace solutions_equality_l337_33730

theorem solutions_equality (b c : ℝ) : 
  (∀ x : ℝ, (|x - 3| = 4) ↔ (x^2 + b*x + c = 0)) → 
  (b = -6 ∧ c = -7) := by
sorry

end solutions_equality_l337_33730


namespace prime_4k_plus_1_sum_of_squares_l337_33744

theorem prime_4k_plus_1_sum_of_squares (p : ℕ) (k : ℕ) :
  Prime p → p = 4 * k + 1 → ∃ x r : ℤ, p = x^2 + r^2 := by sorry

end prime_4k_plus_1_sum_of_squares_l337_33744


namespace paint_usage_l337_33771

theorem paint_usage (total_paint : ℝ) (first_week_fraction : ℝ) (second_week_fraction : ℝ) 
  (h1 : total_paint = 360)
  (h2 : first_week_fraction = 1/3)
  (h3 : second_week_fraction = 1/5) :
  let first_week_usage := first_week_fraction * total_paint
  let remaining_paint := total_paint - first_week_usage
  let second_week_usage := second_week_fraction * remaining_paint
  first_week_usage + second_week_usage = 168 := by
sorry


end paint_usage_l337_33771


namespace quadratic_completion_l337_33784

theorem quadratic_completion (y : ℝ) : ∃ a : ℝ, y^2 + 14*y + 60 = (y + a)^2 + 11 := by
  sorry

end quadratic_completion_l337_33784


namespace gcd_7920_13230_l337_33764

theorem gcd_7920_13230 : Nat.gcd 7920 13230 = 30 := by
  sorry

end gcd_7920_13230_l337_33764


namespace total_sales_over_three_days_l337_33718

def friday_sales : ℕ := 30

def saturday_sales : ℕ := 2 * friday_sales

def sunday_sales : ℕ := saturday_sales - 15

theorem total_sales_over_three_days : 
  friday_sales + saturday_sales + sunday_sales = 135 := by
  sorry

end total_sales_over_three_days_l337_33718


namespace range_of_a_l337_33777

-- Define the sets A and B
def A : Set ℝ := {x | (4*x - 3)^2 ≤ 1}
def B (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ a + 1}

-- Define the propositions p and q
def p : Prop := ∀ x, x ∈ A
def q (a : ℝ) : Prop := ∀ x, x ∈ B a

-- State the theorem
theorem range_of_a : 
  (∀ a : ℝ, (¬p → ¬(q a)) ∧ ¬(¬(q a) → ¬p)) → 
  (∀ a : ℝ, 0 ≤ a ∧ a ≤ 1/2 ↔ (∀ x, x ∈ A → x ∈ B a) ∧ A ≠ B a) :=
sorry

end range_of_a_l337_33777


namespace parabola_reflection_theorem_l337_33709

/-- Parabola structure -/
structure Parabola where
  p : ℝ
  h : p > 0

/-- Point on a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Line passing through two points -/
def Line (P Q : Point) : Set Point :=
  {R : Point | ∃ t : ℝ, R.x = P.x + t * (Q.x - P.x) ∧ R.y = P.y + t * (Q.y - P.y)}

/-- Check if a point is on the parabola -/
def onParabola (par : Parabola) (P : Point) : Prop :=
  P.y^2 = 2 * par.p * P.x

/-- Check if a point is on the axis of symmetry -/
def onAxisOfSymmetry (P : Point) : Prop :=
  P.y = 0

/-- Reflection of a point about y-axis -/
def reflectAboutYAxis (P : Point) : Point :=
  ⟨-P.x, P.y⟩

/-- Angle between three points -/
noncomputable def angle (A B C : Point) : ℝ := sorry

/-- Main theorem -/
theorem parabola_reflection_theorem (par : Parabola) (A : Point)
  (h_A_on_axis : onAxisOfSymmetry A)
  (h_A_inside : onParabola par A → False) :
  let B := reflectAboutYAxis A
  (∀ P Q : Point, onParabola par P → onParabola par Q →
    P.x * Q.x > 0 → P ∈ Line A Q → Q ∈ Line A P →
    angle P B A = angle Q B A) ∧
  (∀ P Q : Point, onParabola par P → onParabola par Q →
    P.x * Q.x > 0 → P ∈ Line B Q → Q ∈ Line B P →
    angle P A B + angle Q A B = 180) :=
by sorry

end parabola_reflection_theorem_l337_33709


namespace balance_is_132_l337_33780

/-- Calculates the balance of a bank account after two years with given parameters. -/
def balance_after_two_years (initial_deposit : ℝ) (interest_rate : ℝ) (annual_deposit : ℝ) : ℝ :=
  let balance_year_one := initial_deposit * (1 + interest_rate) + annual_deposit
  balance_year_one * (1 + interest_rate) + annual_deposit

/-- Theorem stating that given specific parameters, the balance after two years is $132. -/
theorem balance_is_132 :
  balance_after_two_years 100 0.1 10 = 132 := by
  sorry

#eval balance_after_two_years 100 0.1 10

end balance_is_132_l337_33780


namespace inequality_division_l337_33759

theorem inequality_division (a b : ℝ) (h : a > b) : a / (-3) < b / (-3) := by
  sorry

end inequality_division_l337_33759


namespace pentagonal_tiles_count_l337_33778

theorem pentagonal_tiles_count (t s p : ℕ) : 
  t + s + p = 30 →
  3 * t + 4 * s + 5 * p = 128 →
  p = 10 :=
by sorry

end pentagonal_tiles_count_l337_33778


namespace lily_account_balance_l337_33752

def calculate_remaining_balance (initial_balance shirt_cost book_cost num_books gift_percentage : ℚ) : ℚ :=
  let remaining_after_shirt := initial_balance - shirt_cost
  let shoe_cost := 3 * shirt_cost
  let remaining_after_shoes := remaining_after_shirt - shoe_cost
  let total_book_cost := book_cost * num_books
  let remaining_after_books := remaining_after_shoes - total_book_cost
  let gift_cost := gift_percentage * remaining_after_books
  remaining_after_books - gift_cost

theorem lily_account_balance :
  calculate_remaining_balance 55 7 4 5 0.2 = 5.6 := by
  sorry

end lily_account_balance_l337_33752


namespace hyperbola_equation_l337_33711

/-- Given a hyperbola with the equation (x²/a² - y²/b² = 1) where a > 0 and b > 0,
    if one of its asymptotes is y = (√3/2)x and one of its foci is on the directrix
    of the parabola y² = 4√7x, then a² = 4 and b² = 3. -/
theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∃ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1) →
  (∃ (x y : ℝ), y = (Real.sqrt 3 / 2) * x) →
  (∃ (x : ℝ), x = -Real.sqrt 7) →
  a^2 = 4 ∧ b^2 = 3 := by
  sorry

end hyperbola_equation_l337_33711


namespace xyz_equals_six_l337_33766

theorem xyz_equals_six (a b c x y z : ℂ) 
  (nonzero_a : a ≠ 0) (nonzero_b : b ≠ 0) (nonzero_c : c ≠ 0) 
  (nonzero_x : x ≠ 0) (nonzero_y : y ≠ 0) (nonzero_z : z ≠ 0)
  (eq_a : a = (b + c) / (x - 3))
  (eq_b : b = (a + c) / (y - 3))
  (eq_c : c = (a + b) / (z - 3))
  (sum_prod : x * y + x * z + y * z = 10)
  (sum : x + y + z = 6) : 
  x * y * z = 6 := by
  sorry

end xyz_equals_six_l337_33766


namespace parallel_vectors_trig_identity_l337_33765

/-- Given two vectors a and b in ℝ², where a = (6, 3) and b = (sinθ, cosθ),
    if a is parallel to b, then sin2θ - 2cos²θ = 2/5 -/
theorem parallel_vectors_trig_identity (θ : ℝ) :
  let a : Fin 2 → ℝ := ![6, 3]
  let b : Fin 2 → ℝ := ![Real.sin θ, Real.cos θ]
  (∃ (k : ℝ), ∀ i, a i = k * b i) →
  Real.sin (2 * θ) - 2 * (Real.cos θ)^2 = 2/5 :=
by sorry

end parallel_vectors_trig_identity_l337_33765


namespace find_n_l337_33731

theorem find_n (x y : ℝ) (h : 2 * x - y = 4) : 
  ∃ n : ℝ, 6 * x - n * y = 12 ∧ n = 3 := by
sorry

end find_n_l337_33731


namespace ticket_price_increase_l337_33743

/-- Represents the percentage increase in ticket price for each round -/
def x : ℝ := sorry

/-- The initial ticket price in yuan -/
def initial_price : ℝ := 108

/-- The final ticket price in yuan -/
def final_price : ℝ := 168

/-- Theorem stating that the equation 108(1+x)^2 = 168 correctly represents 
    the ticket price increase over two rounds -/
theorem ticket_price_increase : initial_price * (1 + x)^2 = final_price := by
  sorry

end ticket_price_increase_l337_33743


namespace geometric_sequence_product_l337_33725

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_product (a : ℕ → ℝ) :
  is_geometric_sequence a →
  (∀ n : ℕ, a n > 0) →
  Real.log (a 4) + Real.log (a 7) + Real.log (a 10) = 3 →
  a 1 * a 13 = 100 := by
  sorry

end geometric_sequence_product_l337_33725


namespace min_value_expression_l337_33733

theorem min_value_expression (x y : ℝ) (h1 : x > y) (h2 : y > 0) (h3 : x + y = 1) :
  (4 / (x + 3*y)) + (1 / (x - y)) ≥ 9/2 := by
  sorry

end min_value_expression_l337_33733


namespace cubic_sum_theorem_l337_33775

theorem cubic_sum_theorem (x y : ℝ) (h : x^3 + 21*x*y + y^3 = 343) :
  x + y = 7 ∨ x + y = -14 := by
sorry

end cubic_sum_theorem_l337_33775


namespace perfect_square_trinomial_l337_33729

theorem perfect_square_trinomial : 15^2 + 2*(15*3) + 3^2 = 324 := by
  sorry

end perfect_square_trinomial_l337_33729


namespace parking_space_area_l337_33719

/-- A rectangular parking space with three painted sides -/
structure ParkingSpace where
  length : ℝ
  width : ℝ
  painted_sides_sum : ℝ
  unpainted_side_length : ℝ

/-- The area of a parking space -/
def area (p : ParkingSpace) : ℝ := p.length * p.width

/-- Theorem: The area of a specific parking space is 126 square feet -/
theorem parking_space_area (p : ParkingSpace) 
  (h1 : p.unpainted_side_length = 9)
  (h2 : p.painted_sides_sum = 37)
  (h3 : p.length = p.unpainted_side_length)
  (h4 : p.painted_sides_sum = p.length + 2 * p.width) :
  area p = 126 := by
  sorry

end parking_space_area_l337_33719
