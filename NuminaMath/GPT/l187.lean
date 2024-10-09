import Mathlib

namespace range_of_x_l187_18795

theorem range_of_x (a b c x : ℝ) (h1 : a^2 + 2 * b^2 + 3 * c^2 = 6) (h2 : a + 2 * b + 3 * c > |x + 1|) : -7 < x ∧ x < 5 :=
by
  sorry

end range_of_x_l187_18795


namespace possible_sets_C_l187_18728

def M : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}

def is_partition (A B C : Set ℕ) : Prop :=
  A ∩ B = ∅ ∧ A ∩ C = ∅ ∧ B ∩ C = ∅ ∧ A ∪ B ∪ C = M

def conditions (A B C : Set ℕ) : Prop :=
  is_partition A B C ∧ (∃ (a1 a2 a3 a4 b1 b2 b3 b4 c1 c2 c3 c4 : ℕ), 
    A = {a1, a2, a3, a4} ∧
    B = {b1, b2, b3, b4} ∧
    C = {c1, c2, c3, c4} ∧
    c1 < c2 ∧ c2 < c3 ∧ c3 < c4 ∧
    a1 + b1 = c1 ∧ a2 + b2 = c2 ∧ a3 + b3 = c3 ∧ a4 + b4 = c4)

theorem possible_sets_C (A B C : Set ℕ) (h : conditions A B C) :
  C = {8, 9, 10, 12} ∨ C = {7, 9, 11, 12} ∨ C = {6, 10, 11, 12} :=
sorry

end possible_sets_C_l187_18728


namespace fraction_divisible_by_1963_l187_18758

theorem fraction_divisible_by_1963 (n : ℕ) (hn : 0 < n) :
  ∃ k : ℤ,
    13 * 733^n + 1950 * 582^n = 1963 * k ∧
    ∃ m : ℤ,
      333^n - 733^n - 1068^n + 431^n = 1963 * m :=
by
  sorry

end fraction_divisible_by_1963_l187_18758


namespace difference_of_squares_l187_18771

variable (a b : ℝ)

theorem difference_of_squares (h1 : a + b = 2) (h2 : a - b = 3) : a^2 - b^2 = 6 := 
by
  sorry

end difference_of_squares_l187_18771


namespace quadractic_roots_value_l187_18784

theorem quadractic_roots_value (c d : ℝ) (h₁ : 3*c^2 + 9*c - 21 = 0) (h₂ : 3*d^2 + 9*d - 21 = 0) :
  (3*c - 4) * (6*d - 8) = -22 := by
  sorry

end quadractic_roots_value_l187_18784


namespace find_n_from_equation_l187_18794

theorem find_n_from_equation :
  ∃ n : ℕ, (3 * 3 * 5 * 5 * 7 * 9 = 3 * 3 * 7 * n * n) → n = 15 :=
by
  sorry

end find_n_from_equation_l187_18794


namespace probability_at_least_5_heads_l187_18718

def fair_coin_probability_at_least_5_heads : ℚ :=
  (Nat.choose 7 5 + Nat.choose 7 6 + Nat.choose 7 7) / 2^7

theorem probability_at_least_5_heads :
  fair_coin_probability_at_least_5_heads = 29 / 128 := 
  by
    sorry

end probability_at_least_5_heads_l187_18718


namespace B_work_days_l187_18752

theorem B_work_days
  (A_work_rate : ℝ) (B_work_rate : ℝ) (A_days_worked : ℝ) (B_days_worked : ℝ)
  (total_work : ℝ) (remaining_work : ℝ) :
  A_work_rate = 1 / 15 →
  B_work_rate = total_work / 18 →
  A_days_worked = 5 →
  remaining_work = total_work - A_work_rate * A_days_worked →
  B_days_worked = 12 →
  remaining_work = B_work_rate * B_days_worked →
  total_work = 1 →
  B_days_worked = 12 →
  B_work_rate = total_work / 18 →
  B_days_alone = total_work / B_work_rate →
  B_days_alone = 18 := 
by
  intro hA_work_rate hB_work_rate hA_days_worked hremaining_work hB_days_worked hremaining_work_eq htotal_work hB_days_worked_again hsry_mul_inv hB_days_we_alone_eq
  sorry

end B_work_days_l187_18752


namespace fifteen_percent_eq_135_l187_18766

theorem fifteen_percent_eq_135 (x : ℝ) (h : (15 / 100) * x = 135) : x = 900 :=
sorry

end fifteen_percent_eq_135_l187_18766


namespace problem_1_problem_2_l187_18751

def f (x : ℝ) : ℝ := |x - 3| - 2
def g (x : ℝ) : ℝ := -|x + 1| + 4

theorem problem_1:
  { x : ℝ // 0 ≤ x ∧ x ≤ 6 } = { x : ℝ // f x ≤ 1 } :=
sorry

theorem problem_2:
  { m : ℝ // m ≤ -3 } = { m : ℝ // ∀ x : ℝ, f x - g x ≥ m + 1 } :=
sorry

end problem_1_problem_2_l187_18751


namespace solution_l187_18717

theorem solution (x y : ℝ) (h1 : x ≠ y) (h2 : x^2 - 2000 * x = y^2 - 2000 * y) : 
  x + y = 2000 := 
by 
  sorry

end solution_l187_18717


namespace square_roots_equal_implication_l187_18777

theorem square_roots_equal_implication (b : ℝ) (h : 5 * b = 3 + 2 * b) : -b = -1 := 
by sorry

end square_roots_equal_implication_l187_18777


namespace largest_multiple_of_15_less_than_500_l187_18782

theorem largest_multiple_of_15_less_than_500 : ∃ n : ℕ, (n > 0 ∧ 15 * n < 500) ∧ (∀ m : ℕ, m > n → 15 * m >= 500) ∧ 15 * n = 495 :=
by
  sorry

end largest_multiple_of_15_less_than_500_l187_18782


namespace matrix_B_power_103_l187_18719

def B : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![0, 1, 0], ![0, 0, 1], ![1, 0, 0]]

theorem matrix_B_power_103 :
  B ^ 103 = B :=
by
  sorry

end matrix_B_power_103_l187_18719


namespace total_students_in_class_l187_18763

def students_play_football : Nat := 26
def students_play_tennis : Nat := 20
def students_play_both : Nat := 17
def students_play_neither : Nat := 7

theorem total_students_in_class :
  (students_play_football + students_play_tennis - students_play_both + students_play_neither) = 36 :=
by
  sorry

end total_students_in_class_l187_18763


namespace original_price_of_trouser_l187_18756

theorem original_price_of_trouser (sale_price : ℝ) (percent_decrease : ℝ) (original_price : ℝ) 
  (h1 : sale_price = 75) 
  (h2 : percent_decrease = 0.25) 
  (h3 : original_price - percent_decrease * original_price = sale_price) : 
  original_price = 100 :=
by
  sorry

end original_price_of_trouser_l187_18756


namespace matrix_system_solution_range_l187_18723

theorem matrix_system_solution_range (m : ℝ) :
  (∃ x y: ℝ, 
    (m * x + y = m + 1) ∧ 
    (x + m * y = 2 * m)) ↔ m ≠ -1 :=
by
  sorry

end matrix_system_solution_range_l187_18723


namespace exists_x_for_every_n_l187_18740

theorem exists_x_for_every_n (n : ℕ) (hn : 0 < n) : ∃ x : ℤ, 2^n ∣ (x^2 - 17) :=
sorry

end exists_x_for_every_n_l187_18740


namespace sequence_value_a1_l187_18785

theorem sequence_value_a1 (a : ℕ → ℝ) 
  (h₁ : ∀ n, a (n + 1) = (1 / 2) * a n) 
  (h₂ : a 4 = 8) : a 1 = 64 :=
sorry

end sequence_value_a1_l187_18785


namespace div_4800_by_125_expr_13_mul_74_add_27_mul_13_sub_13_l187_18739

theorem div_4800_by_125 : 4800 / 125 = 38.4 :=
by
  sorry

theorem expr_13_mul_74_add_27_mul_13_sub_13 : 13 * 74 + 27 * 13 - 13 = 1300 :=
by
  sorry

end div_4800_by_125_expr_13_mul_74_add_27_mul_13_sub_13_l187_18739


namespace no_solution_for_x_l187_18702

theorem no_solution_for_x (m : ℝ) :
  (∀ x : ℝ, (1 / (x - 4)) + (m / (x + 4)) ≠ ((m + 3) / (x^2 - 16))) ↔ (m = -1 ∨ m = 5 ∨ m = -1 / 3) :=
sorry

end no_solution_for_x_l187_18702


namespace no_max_value_if_odd_and_symmetric_l187_18788

variable (f : ℝ → ℝ)

-- Definitions:
def domain_is_R (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = f x
def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x
def is_symmetric_about_1_1 (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (2 - x) = 2 - f x

-- The theorem stating that under the given conditions there is no maximum value.
theorem no_max_value_if_odd_and_symmetric :
  domain_is_R f → is_odd_function f → is_symmetric_about_1_1 f → ¬∃ M : ℝ, ∀ x : ℝ, f x ≤ M := by
  sorry

end no_max_value_if_odd_and_symmetric_l187_18788


namespace complex_power_difference_l187_18732

theorem complex_power_difference (i : ℂ) (h : i^2 = -1) : (1 + i) ^ 18 - (1 - i) ^ 18 = 1024 * i :=
by
  sorry

end complex_power_difference_l187_18732


namespace cylinder_unoccupied_volume_l187_18769

theorem cylinder_unoccupied_volume (r h_cylinder h_cone : ℝ) 
  (h : r = 10 ∧ h_cylinder = 30 ∧ h_cone = 15) :
  (π * r^2 * h_cylinder - 2 * (1/3 * π * r^2 * h_cone) = 2000 * π) :=
by
  rcases h with ⟨rfl, rfl, rfl⟩
  simp
  sorry

end cylinder_unoccupied_volume_l187_18769


namespace megatech_basic_astrophysics_degrees_l187_18791

def budget_allocation (microphotonics home_electronics food_additives gm_microorganisms industrial_lubricants: ℕ) :=
  100 - (microphotonics + home_electronics + food_additives + gm_microorganisms + industrial_lubricants)

noncomputable def degrees_for_astrophysics (percentage: ℕ) :=
  (percentage * 360) / 100

theorem megatech_basic_astrophysics_degrees (microphotonics home_electronics food_additives gm_microorganisms industrial_lubricants: ℕ) :
  microphotonics = 14 →
  home_electronics = 24 →
  food_additives = 10 →
  gm_microorganisms = 29 →
  industrial_lubricants = 8 →
  degrees_for_astrophysics (budget_allocation microphotonics home_electronics food_additives gm_microorganisms industrial_lubricants) = 54 :=
by
  sorry

end megatech_basic_astrophysics_degrees_l187_18791


namespace min_value_z_l187_18722

theorem min_value_z (x y : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_sum : x + y = 1) :
  ∃ z_min, z_min = (x + 1 / x) * (y + 1 / y) ∧ z_min = 33 / 4 :=
sorry

end min_value_z_l187_18722


namespace sum_of_x_y_l187_18714

theorem sum_of_x_y (x y : ℕ) (x_square_condition : ∃ x, ∃ n : ℕ, 450 * x = n^2)
                   (y_cube_condition : ∃ y, ∃ m : ℕ, 450 * y = m^3) :
                   x = 2 ∧ y = 4 → x + y = 6 := 
sorry

end sum_of_x_y_l187_18714


namespace find_sandwich_cost_l187_18733

theorem find_sandwich_cost (S : ℝ) :
  3 * S + 2 * 4 = 26 → S = 6 :=
by
  intro h
  sorry

end find_sandwich_cost_l187_18733


namespace symmetric_points_y_axis_l187_18715

theorem symmetric_points_y_axis :
  ∀ (m n : ℝ), (m + 4 = 0) → (n = 3) → (m + n) ^ 2023 = -1 :=
by
  intros m n Hm Hn
  sorry

end symmetric_points_y_axis_l187_18715


namespace length_of_AB_l187_18711

-- Define the problem variables
variables (AB CD : ℝ)
variables (h : ℝ)

-- Define the conditions
def ratio_condition (AB CD : ℝ) : Prop :=
  AB / CD = 7 / 3

def length_condition (AB CD : ℝ) : Prop :=
  AB + CD = 210

-- Lean statement combining the conditions and the final result
theorem length_of_AB (h : ℝ) (AB CD : ℝ) (h_ratio : ratio_condition AB CD) (h_length : length_condition AB CD) : 
  AB = 147 :=
by
  -- Definitions and proof would go here
  sorry

end length_of_AB_l187_18711


namespace g_x_plus_three_l187_18736

variable (x : ℝ)

def g (x : ℝ) : ℝ := x^2 - x

theorem g_x_plus_three : g (x + 3) = x^2 + 5 * x + 6 := by
  sorry

end g_x_plus_three_l187_18736


namespace arithmetic_sequence_sum_l187_18759

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (h : a 8 + a 10 = 2) : 
  (17 * (a 1 + a 17) / 2) = 17 := by
sorry

end arithmetic_sequence_sum_l187_18759


namespace intersection_l187_18747

def setA : Set ℝ := { x : ℝ | x^2 - 2*x - 3 < 0 }
def setB : Set ℝ := { x : ℝ | x > 1 }

theorem intersection (x : ℝ) : x ∈ setA ∧ x ∈ setB ↔ 1 < x ∧ x < 3 := by
  sorry

end intersection_l187_18747


namespace fraction_multiplication_l187_18712

theorem fraction_multiplication : (1 / 2) * (1 / 3) * (1 / 6) * 108 = 3 := by
  sorry

end fraction_multiplication_l187_18712


namespace find_x_l187_18768

-- Given condition: 144 / x = 14.4 / 0.0144
theorem find_x (x : ℝ) (h : 144 / x = 14.4 / 0.0144) : x = 0.144 := by
  sorry

end find_x_l187_18768


namespace Eunji_higher_than_Yoojung_l187_18706

-- Define floors for Yoojung and Eunji
def Yoojung_floor: ℕ := 17
def Eunji_floor: ℕ := 25

-- Assert that Eunji lives on a higher floor than Yoojung
theorem Eunji_higher_than_Yoojung : Eunji_floor > Yoojung_floor :=
  by
    sorry

end Eunji_higher_than_Yoojung_l187_18706


namespace part1_part2_l187_18748

-- Part 1
noncomputable def f (x a : ℝ) := x * Real.log x - a * x^2 + a

theorem part1 (a : ℝ) : (∀ x : ℝ, 0 < x → f x a ≤ a) → a ≥ 1 / Real.exp 1 :=
by
  sorry

-- Part 2
theorem part2 (a : ℝ) (x₀ : ℝ) : 
  (∀ x : ℝ, f x₀ a < f x a → x = x₀) → a < 1 / 2 → 2 * a - 1 < f x₀ a ∧ f x₀ a < 0 :=
by
  sorry

end part1_part2_l187_18748


namespace contrapositive_iff_l187_18762

theorem contrapositive_iff (a b : ℝ) :
  (a^2 - b^2 = 0 → a = b) ↔ (a ≠ b → a^2 - b^2 ≠ 0) :=
by
  sorry

end contrapositive_iff_l187_18762


namespace expression_bounds_l187_18707

theorem expression_bounds (x y z w : ℝ) (hx : 0 ≤ x) (hx1 : x ≤ 1) (hy : 0 ≤ y) (hy1 : y ≤ 1) (hz : 0 ≤ z) (hz1 : z ≤ 1) (hw : 0 ≤ w) (hw1 : w ≤ 1) :
  2 * Real.sqrt 2 ≤ Real.sqrt (x^2 + (1 - y)^2) + Real.sqrt (y^2 + (1 - z)^2) + Real.sqrt (z^2 + (1 - w)^2) + Real.sqrt (w^2 + (1 - x)^2) ∧
  Real.sqrt (x^2 + (1 - y)^2) + Real.sqrt (y^2 + (1 - z)^2) + Real.sqrt (z^2 + (1 - w)^2) + Real.sqrt (w^2 + (1 - x)^2) ≤ 4 :=
by
  sorry

end expression_bounds_l187_18707


namespace even_function_properties_l187_18703

theorem even_function_properties 
  (f : ℝ → ℝ)
  (h_even : ∀ x : ℝ, f x = f (-x))
  (h_increasing : ∀ x y : ℝ, 5 ≤ x ∧ x ≤ y ∧ y ≤ 7 → f x ≤ f y)
  (h_min_value : ∀ x : ℝ, 5 ≤ x ∧ x ≤ 7 → 6 ≤ f x) :
  (∀ x y : ℝ, -7 ≤ x ∧ x ≤ y ∧ y ≤ -5 → f y ≤ f x) ∧ (∀ x : ℝ, -7 ≤ x ∧ x ≤ -5 → 6 ≤ f x) :=
by
  sorry

end even_function_properties_l187_18703


namespace distance_between_A_and_B_l187_18730

variable (d : ℝ) -- Total distance between A and B

def car_speeds (vA vB t : ℝ) : Prop :=
vA = 80 ∧ vB = 100 ∧ t = 2

def total_covered_distance (vA vB t : ℝ) : ℝ :=
(vA + vB) * t

def percentage_distance (total_distance covered_distance : ℝ) : Prop :=
0.6 * total_distance = covered_distance

theorem distance_between_A_and_B (vA vB t : ℝ) (H1 : car_speeds vA vB t) 
  (H2 : percentage_distance d (total_covered_distance vA vB t)) : d = 600 := by
  sorry

end distance_between_A_and_B_l187_18730


namespace cost_percentage_l187_18778

variable (t b : ℝ)

def C := t * b ^ 4
def R := t * (2 * b) ^ 4

theorem cost_percentage : R = 16 * C := by
  sorry

end cost_percentage_l187_18778


namespace line_through_points_l187_18774

theorem line_through_points (a b : ℝ) (h1 : 3 = a * 2 + b) (h2 : 19 = a * 6 + b) :
  a - b = 9 :=
sorry

end line_through_points_l187_18774


namespace geometric_arithmetic_sequence_common_ratio_l187_18701

theorem geometric_arithmetic_sequence_common_ratio (a_1 a_2 a_3 q : ℝ) 
  (h1 : a_2 = a_1 * q) 
  (h2 : a_3 = a_1 * q^2)
  (h3 : 2 * a_3 = a_1 + a_2) : (q = 1) ∨ (q = -1) :=
by
  sorry

end geometric_arithmetic_sequence_common_ratio_l187_18701


namespace closest_perfect_square_to_1042_is_1024_l187_18721

theorem closest_perfect_square_to_1042_is_1024 :
  ∀ n : ℕ, (n = 32 ∨ n = 33) → ((1042 - n^2 = 18) ↔ n = 32):=
by
  intros n hn
  cases hn
  case inl h32 => sorry
  case inr h33 => sorry

end closest_perfect_square_to_1042_is_1024_l187_18721


namespace age_of_b_l187_18738

theorem age_of_b (a b : ℕ) 
(h1 : a + 10 = 2 * (b - 10)) 
(h2 : a = b + 4) : 
b = 34 := 
sorry

end age_of_b_l187_18738


namespace purple_valley_skirts_l187_18799

def AzureValley : ℕ := 60

def SeafoamValley (A : ℕ) : ℕ := (2 * A) / 3

def PurpleValley (S : ℕ) : ℕ := S / 4

theorem purple_valley_skirts :
  PurpleValley (SeafoamValley AzureValley) = 10 :=
by
  sorry

end purple_valley_skirts_l187_18799


namespace number_of_valid_trapezoids_l187_18790

noncomputable def calculate_number_of_trapezoids : ℕ :=
  let rows_1 := 7
  let rows_2 := 9
  let unit_spacing := 1
  let height := 2
  -- Here, we should encode the actual combinatorial calculation as per the problem solution
  -- but for the Lean 4 statement, we will provide the correct answer directly.
  361

theorem number_of_valid_trapezoids :
  calculate_number_of_trapezoids = 361 :=
sorry

end number_of_valid_trapezoids_l187_18790


namespace sin_B_of_arithmetic_sequence_angles_l187_18760

theorem sin_B_of_arithmetic_sequence_angles (A B C : ℝ) (h1 : A + C = 2 * B) (h2 : A + B + C = Real.pi) :
  Real.sin B = Real.sqrt 3 / 2 :=
sorry

end sin_B_of_arithmetic_sequence_angles_l187_18760


namespace sum_of_squares_of_real_solutions_l187_18743

theorem sum_of_squares_of_real_solutions (x : ℝ) (h : x ^ 64 = 16 ^ 16) : 
  (x = 2 ∨ x = -2) → (x ^ 2 + (-x) ^ 2) = 8 :=
by
  sorry

end sum_of_squares_of_real_solutions_l187_18743


namespace min_expr_value_l187_18793

theorem min_expr_value (α β : ℝ) :
  ∃ (c : ℝ), c = 36 ∧ ((3 * Real.cos α + 4 * Real.sin β - 5)^2 + (3 * Real.sin α + 4 * Real.cos β - 12)^2 = c) :=
sorry

end min_expr_value_l187_18793


namespace cars_on_river_road_l187_18725

-- Define the number of buses and cars
variables (B C : ℕ)

-- Given conditions
def ratio_condition : Prop := (B : ℚ) / C = 1 / 17
def fewer_buses_condition : Prop := B = C - 80

-- Problem statement
theorem cars_on_river_road (h_ratio : ratio_condition B C) (h_fewer : fewer_buses_condition B C) : C = 85 :=
by
  sorry

end cars_on_river_road_l187_18725


namespace quadratic_function_value_l187_18709

theorem quadratic_function_value
  (p q r : ℝ)
  (h1 : p + q + r = 3)
  (h2 : 4 * p + 2 * q + r = 12) :
  p + q + 3 * r = -5 :=
by
  sorry

end quadratic_function_value_l187_18709


namespace exists_larger_integer_l187_18796

theorem exists_larger_integer (a b : Nat) (h1 : b > a) (h2 : b - a = 5) (h3 : a * b = 88) :
  b = 11 :=
sorry

end exists_larger_integer_l187_18796


namespace harkamal_payment_l187_18797

noncomputable def calculate_total_cost : ℝ :=
  let price_grapes := 8 * 70
  let price_mangoes := 9 * 45
  let price_apples := 5 * 30
  let price_strawberries := 3 * 100
  let price_oranges := 10 * 40
  let price_kiwis := 6 * 60
  let total_grapes_and_apples := price_grapes + price_apples
  let discount_grapes_and_apples := 0.10 * total_grapes_and_apples
  let total_oranges_and_kiwis := price_oranges + price_kiwis
  let discount_oranges_and_kiwis := 0.05 * total_oranges_and_kiwis
  let total_mangoes_and_strawberries := price_mangoes + price_strawberries
  let tax_mangoes_and_strawberries := 0.12 * total_mangoes_and_strawberries
  let total_amount := price_grapes + price_mangoes + price_apples + price_strawberries + price_oranges + price_kiwis
  total_amount - discount_grapes_and_apples - discount_oranges_and_kiwis + tax_mangoes_and_strawberries

theorem harkamal_payment : calculate_total_cost = 2150.6 :=
by
  sorry

end harkamal_payment_l187_18797


namespace angle_E_in_quadrilateral_l187_18792

theorem angle_E_in_quadrilateral (E F G H : ℝ) 
  (h1 : E = 5 * H)
  (h2 : E = 4 * G)
  (h3 : E = (5/3) * F)
  (h_sum : E + F + G + H = 360) : 
  E = 131 := by 
  sorry

end angle_E_in_quadrilateral_l187_18792


namespace max_f_of_polynomial_l187_18776

theorem max_f_of_polynomial (f : ℝ → ℝ)
    (hf_nonneg : ∀ x, 0 ≤ f x)
    (h_poly : ∃ p : Polynomial ℝ, ∀ x, f x = Polynomial.eval x p)
    (h1 : f 4 = 16)
    (h2 : f 16 = 512) :
    f 8 ≤ 64 :=
by
  sorry

end max_f_of_polynomial_l187_18776


namespace phoebe_dog_peanut_butter_l187_18780

-- Definitions based on the conditions
def servings_per_jar : ℕ := 15
def jars_needed : ℕ := 4
def days : ℕ := 30

-- Problem statement
theorem phoebe_dog_peanut_butter :
  (jars_needed * servings_per_jar) / days / 2 = 1 :=
by sorry

end phoebe_dog_peanut_butter_l187_18780


namespace jerry_money_left_after_shopping_l187_18754

theorem jerry_money_left_after_shopping :
  let initial_money := 50
  let cost_mustard_oil := 2 * 13
  let cost_penne_pasta := 3 * 4
  let cost_pasta_sauce := 1 * 5
  let total_cost := cost_mustard_oil + cost_penne_pasta + cost_pasta_sauce
  let money_left := initial_money - total_cost
  money_left = 7 := 
sorry

end jerry_money_left_after_shopping_l187_18754


namespace minimum_perimeter_l187_18720

noncomputable def minimum_perimeter_triangle (l m n : ℕ) : ℕ :=
  l + m + n

theorem minimum_perimeter :
  ∀ (l m n : ℕ),
    (l > m) → (m > n) → 
    ((∃ k : ℕ, 10^4 ∣ 3^l - 3^m + k * 10^4) ∧ (∃ k : ℕ, 10^4 ∣ 3^m - 3^n + k * 10^4) ∧ (∃ k : ℕ, 10^4 ∣ 3^l - 3^n + k * 10^4)) →
    minimum_perimeter_triangle l m n = 3003 :=
by
  intros l m n hlm hmn hmod
  sorry

end minimum_perimeter_l187_18720


namespace child_b_share_l187_18741

def total_money : ℕ := 4320

def ratio_parts : List ℕ := [2, 3, 4, 5, 6]

def parts_sum (parts : List ℕ) : ℕ :=
  parts.foldl (· + ·) 0

def value_of_one_part (total : ℕ) (parts : ℕ) : ℕ :=
  total / parts

def b_share (value_per_part : ℕ) (b_parts : ℕ) : ℕ :=
  value_per_part * b_parts

theorem child_b_share :
  let total_money := 4320
  let ratio_parts := [2, 3, 4, 5, 6]
  let total_parts := parts_sum ratio_parts
  let one_part_value := value_of_one_part total_money total_parts
  let b_parts := 3
  b_share one_part_value b_parts = 648 := by
  let total_money := 4320
  let ratio_parts := [2, 3, 4, 5, 6]
  let total_parts := parts_sum ratio_parts
  let one_part_value := value_of_one_part total_money total_parts
  let b_parts := 3
  show b_share one_part_value b_parts = 648
  sorry

end child_b_share_l187_18741


namespace total_trip_time_l187_18755

theorem total_trip_time (driving_time : ℕ) (stuck_time : ℕ) (total_time : ℕ) :
  (stuck_time = 2 * driving_time) → (driving_time = 5) → (total_time = driving_time + stuck_time) → total_time = 15 :=
by
  intros h1 h2 h3
  sorry

end total_trip_time_l187_18755


namespace chess_tournament_participants_and_days_l187_18783

theorem chess_tournament_participants_and_days:
  ∃ n d : ℕ, 
    (n % 2 = 1) ∧
    (n * (n - 1) / 2 = 630) ∧
    (d = 34 / 2) ∧
    (n = 35) ∧
    (d = 17) :=
sorry

end chess_tournament_participants_and_days_l187_18783


namespace johns_starting_elevation_l187_18734

variable (horizontal_distance : ℝ) (final_elevation : ℝ) (initial_elevation : ℝ)
variable (vertical_ascent : ℝ)

-- Given conditions
axiom h1 : (vertical_ascent / horizontal_distance) = (1 / 2)
axiom h2 : final_elevation = 1450
axiom h3 : horizontal_distance = 2700

-- Prove that John's starting elevation is 100 feet
theorem johns_starting_elevation : initial_elevation = 100 := by
  sorry

end johns_starting_elevation_l187_18734


namespace cheetah_catches_deer_in_10_minutes_l187_18772

noncomputable def deer_speed : ℝ := 50 -- miles per hour
noncomputable def cheetah_speed : ℝ := 60 -- miles per hour
noncomputable def time_difference : ℝ := 2 / 60 -- 2 minutes converted to hours
noncomputable def distance_deer : ℝ := deer_speed * time_difference
noncomputable def speed_difference : ℝ := cheetah_speed - deer_speed
noncomputable def catch_up_time : ℝ := distance_deer / speed_difference

theorem cheetah_catches_deer_in_10_minutes :
  catch_up_time * 60 = 10 :=
by
  sorry

end cheetah_catches_deer_in_10_minutes_l187_18772


namespace james_total_points_l187_18749

def f : ℕ := 13
def s : ℕ := 20
def p_f : ℕ := 3
def p_s : ℕ := 2

def total_points : ℕ := (f * p_f) + (s * p_s)

theorem james_total_points : total_points = 79 := 
by
  -- Proof would go here.
  sorry

end james_total_points_l187_18749


namespace distinct_remainders_mod_3n_l187_18742

open Nat

theorem distinct_remainders_mod_3n 
  (n : ℕ) 
  (hn_odd : Odd n)
  (ai : ℕ → ℕ)
  (bi : ℕ → ℕ)
  (ai_def : ∀ i, 1 ≤ i ∧ i ≤ n → ai i = 3*i - 2)
  (bi_def : ∀ i, 1 ≤ i ∧ i ≤ n → bi i = 3*i - 3)
  (k : ℕ) 
  (hk : 0 < k ∧ k < n)
  : ∀ i, 1 ≤ i ∧ i ≤ n → (∀ j, 1 ≤ j ∧ j ≤ n → i ≠ j →
     ∀ ⦃ r s t u v : ℕ ⦄, 
       (r = (ai i + ai (i % n + 1)) % (3*n) ∧ 
        s = (ai i + bi i) % (3*n) ∧ 
        t = (bi i + bi ((i + k) % n + 1)) % (3*n)) →
       r ≠ s ∧ s ≠ t ∧ t ≠ r) := 
sorry

end distinct_remainders_mod_3n_l187_18742


namespace bamboo_consumption_correct_l187_18713

-- Define the daily bamboo consumption for adult and baby pandas
def adult_daily_bamboo : ℕ := 138
def baby_daily_bamboo : ℕ := 50

-- Define the number of days in a week
def days_in_week : ℕ := 7

-- Define the total bamboo consumed by an adult panda in a week
def adult_weekly_bamboo := adult_daily_bamboo * days_in_week

-- Define the total bamboo consumed by a baby panda in a week
def baby_weekly_bamboo := baby_daily_bamboo * days_in_week

-- Define the total bamboo consumed by both pandas in a week
def total_bamboo_consumed := adult_weekly_bamboo + baby_weekly_bamboo

-- The theorem states that the total bamboo consumption in a week is 1316 pounds
theorem bamboo_consumption_correct : total_bamboo_consumed = 1316 := by
  sorry

end bamboo_consumption_correct_l187_18713


namespace lindsey_squat_weight_l187_18710

-- Define the conditions
def num_bands : ℕ := 2
def resistance_per_band : ℤ := 5
def dumbbell_weight : ℤ := 10

-- Define the weight Lindsay will squat
def total_weight : ℤ := num_bands * resistance_per_band + dumbbell_weight

-- State the theorem
theorem lindsey_squat_weight : total_weight = 20 :=
by
  sorry

end lindsey_squat_weight_l187_18710


namespace xiaojuan_savings_l187_18757

-- Define the conditions
def spent_on_novel (savings : ℝ) : ℝ := 0.5 * savings
def mother_gave : ℝ := 5
def spent_on_dictionary (amount_given : ℝ) : ℝ := 0.5 * amount_given + 0.4
def remaining_amount : ℝ := 7.2

-- Define the theorem stating the equivalence
theorem xiaojuan_savings : ∃ (savings: ℝ), spent_on_novel savings + mother_gave - spent_on_dictionary mother_gave - remaining_amount = savings / 2 ∧ savings = 20.4 :=
by {
  sorry
}

end xiaojuan_savings_l187_18757


namespace transformed_data_properties_l187_18781

-- Definitions of the initial mean and variance
def initial_mean : ℝ := 2.8
def initial_variance : ℝ := 3.6

-- Definitions of transformation constants
def multiplier : ℝ := 2
def increment : ℝ := 60

-- New mean after transformation
def new_mean : ℝ := multiplier * initial_mean + increment

-- New variance after transformation
def new_variance : ℝ := (multiplier ^ 2) * initial_variance

-- Theorem statement
theorem transformed_data_properties :
  new_mean = 65.6 ∧ new_variance = 14.4 :=
by
  sorry

end transformed_data_properties_l187_18781


namespace circumference_of_tire_l187_18729

theorem circumference_of_tire (rotations_per_minute : ℕ) (speed_kmh : ℕ) 
  (h1 : rotations_per_minute = 400) (h2 : speed_kmh = 72) :
  let speed_mpm := speed_kmh * 1000 / 60
  let circumference := speed_mpm / rotations_per_minute
  circumference = 3 :=
by
  sorry

end circumference_of_tire_l187_18729


namespace cos_5theta_l187_18779

theorem cos_5theta (θ : ℝ) (h : Real.cos θ = 1/3) : Real.cos (5 * θ) = 241/243 :=
by
  sorry

end cos_5theta_l187_18779


namespace nonneg_real_inequality_l187_18750

theorem nonneg_real_inequality (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) :
  a^4 + b^4 + c^4 - 2 * (a^2 * b^2 + a^2 * c^2 + b^2 * c^2) + a^2 * b * c + b^2 * a * c + c^2 * a * b ≥ 0 := 
by
  sorry

end nonneg_real_inequality_l187_18750


namespace number_of_pencil_cartons_l187_18761

theorem number_of_pencil_cartons
  (P E : ℕ) 
  (h1 : P + E = 100)
  (h2 : 6 * P + 3 * E = 360) : 
  P = 20 := 
by
  sorry

end number_of_pencil_cartons_l187_18761


namespace parabola_vertex_point_l187_18753

theorem parabola_vertex_point (a b c : ℝ) : 
  (∀ x y : ℝ, y = a * x^2 + b * x + c → 
  ∃ k : ℝ, ∃ h : ℝ, y = a * (x - h)^2 + k ∧ h = 2 ∧ k = -1 ∧ 
  (∃ y₀ : ℝ, 7 = a * (0 - h)^2 + k) ∧ y₀ = 7) 
  → (a = 2 ∧ b = -8 ∧ c = 7) := by
  sorry

end parabola_vertex_point_l187_18753


namespace gcf_of_24_and_16_l187_18716

theorem gcf_of_24_and_16 :
  let n := 24
  let lcm := 48
  gcd n 16 = 8 :=
by
  sorry

end gcf_of_24_and_16_l187_18716


namespace solve_problem1_solve_problem2_l187_18745

noncomputable def problem1 (m n : ℝ) : Prop :=
  (m + n) ^ 2 - 10 * (m + n) + 25 = (m + n - 5) ^ 2

noncomputable def problem2 (x : ℝ) : Prop :=
  ((x ^ 2 - 6 * x + 8) * (x ^ 2 - 6 * x + 10) + 1) = (x - 3) ^ 4

-- Placeholder for proofs
theorem solve_problem1 (m n : ℝ) : problem1 m n :=
by
  sorry

theorem solve_problem2 (x : ℝ) : problem2 x :=
by
  sorry

end solve_problem1_solve_problem2_l187_18745


namespace problem_l187_18731

variables {a b c : ℝ}

-- Given positive numbers a, b, c
axiom a_pos : 0 < a
axiom b_pos : 0 < b
axiom c_pos : 0 < c

-- Given conditions
axiom h1 : a * b + a + b = 3
axiom h2 : b * c + b + c = 3
axiom h3 : a * c + a + c = 3

-- Goal statement
theorem problem : (a + 1) * (b + 1) * (c + 1) = 8 := 
by 
  sorry

end problem_l187_18731


namespace toms_animal_robots_l187_18700

theorem toms_animal_robots (h : ∀ (m t : ℕ), t = 2 * m) (hmichael : 8 = m) : ∃ t, t = 16 := 
by
  sorry

end toms_animal_robots_l187_18700


namespace circus_juggling_l187_18737

theorem circus_juggling (jugglers : ℕ) (balls_per_juggler : ℕ) (total_balls : ℕ)
  (h1 : jugglers = 5000)
  (h2 : balls_per_juggler = 12)
  (h3 : total_balls = jugglers * balls_per_juggler) :
  total_balls = 60000 :=
by
  rw [h1, h2] at h3
  exact h3

end circus_juggling_l187_18737


namespace cat_food_sufficiency_l187_18786

variable (L S : ℝ)

theorem cat_food_sufficiency (h : L + 4 * S = 14) : L + 3 * S >= 11 :=
sorry

end cat_food_sufficiency_l187_18786


namespace tan_pi_div_four_l187_18765

theorem tan_pi_div_four : Real.tan (π / 4) = 1 := by
  sorry

end tan_pi_div_four_l187_18765


namespace outer_perimeter_l187_18787

theorem outer_perimeter (F G H I J K L M N : ℕ) 
  (h_outer : F + G + H + I + J = 42) 
  (h_inner : K + L + M = 20) 
  (h_adjustment : N = 4) : 
  F + G + H + I + J - K - L - M + N = 26 := 
by 
  sorry

end outer_perimeter_l187_18787


namespace boxes_with_nothing_l187_18726

theorem boxes_with_nothing (h_total : 15 = total_boxes)
    (h_pencils : 9 = pencil_boxes)
    (h_pens : 5 = pen_boxes)
    (h_both_pens_and_pencils : 3 = both_pen_and_pencil_boxes)
    (h_markers : 4 = marker_boxes)
    (h_both_markers_and_pencils : 2 = both_marker_and_pencil_boxes)
    (h_no_markers_and_pens : no_marker_and_pen_boxes = 0)
    (h_no_all_three_items : no_all_three_items = 0) :
    ∃ (neither_boxes : ℕ), neither_boxes = 2 :=
by
  sorry

end boxes_with_nothing_l187_18726


namespace neither_sufficient_nor_necessary_l187_18724

theorem neither_sufficient_nor_necessary (a b : ℝ) (h : a^2 > b^2) : 
  ¬(a > b) ∨ ¬(b > a) := sorry

end neither_sufficient_nor_necessary_l187_18724


namespace range_of_m_l187_18705

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, (mx-1)*(x-2) > 0 ↔ (1/m < x ∧ x < 2)) → m < 0 :=
by
  sorry

end range_of_m_l187_18705


namespace age_difference_proof_l187_18798

theorem age_difference_proof (A B C : ℕ) (h1 : B = 12) (h2 : B = 2 * C) (h3 : A + B + C = 32) :
  A - B = 2 :=
by
  sorry

end age_difference_proof_l187_18798


namespace chloes_test_scores_l187_18704

theorem chloes_test_scores :
  ∃ (scores : List ℕ),
  scores = [93, 92, 86, 82, 79, 78] ∧
  (List.take 4 scores).sum = 339 ∧
  scores.sum / 6 = 85 ∧
  List.Nodup scores ∧
  ∀ score ∈ scores, score < 95 :=
by
  sorry

end chloes_test_scores_l187_18704


namespace right_triangle_incenter_distance_l187_18746

noncomputable def triangle_right_incenter_distance : ℝ :=
  let AB := 4 * Real.sqrt 2
  let BC := 6
  let AC := Real.sqrt (AB^2 + BC^2)
  let area := (1 / 2) * AB * BC
  let s := (AB + BC + AC) / 2
  let r := area / s
  r

theorem right_triangle_incenter_distance :
  let AB := 4 * Real.sqrt 2
  let BC := 6
  let AC := 2 * Real.sqrt 17
  let area := 12 * Real.sqrt 2
  let s := 2 * Real.sqrt 2 + 3 + Real.sqrt 17
  let BI := area / s
  BI = triangle_right_incenter_distance := sorry

end right_triangle_incenter_distance_l187_18746


namespace numberOfSolutions_l187_18735

noncomputable def numberOfRealPositiveSolutions(x : ℝ) : Prop := 
  (x^6 + 1) * (x^4 + x^2 + 1) = 6 * x^5

theorem numberOfSolutions : ∃! x : ℝ, numberOfRealPositiveSolutions x := 
by
  sorry

end numberOfSolutions_l187_18735


namespace initial_amount_spent_l187_18789

theorem initial_amount_spent
    (X : ℕ) -- initial amount of money to spend
    (sets_purchased : ℕ := 250) -- total sets purchased
    (sets_cost_20 : ℕ := 178) -- sets that cost $20 each
    (price_per_set : ℕ := 20) -- price of each set that cost $20
    (remaining_sets : ℕ := sets_purchased - sets_cost_20) -- remaining sets
    (spent_all : (X = sets_cost_20 * price_per_set + remaining_sets * 0)) -- spent all money, remaining sets assumed free to simplify as the exact price is not given or necessary
    : X = 3560 :=
    by
    sorry

end initial_amount_spent_l187_18789


namespace prime_b_plus_1_l187_18744

def is_a_good (a b : ℕ) : Prop :=
  ∀ n : ℕ, n > 0 → a * n ≥ b → (Nat.choose (a * n) b - 1) % (a * n + 1) = 0

theorem prime_b_plus_1 (a b : ℕ) (h1 : is_a_good a b) (h2 : ¬ is_a_good a (b + 2)) : Nat.Prime (b + 1) :=
by
  sorry

end prime_b_plus_1_l187_18744


namespace win_sector_area_l187_18770

theorem win_sector_area (r : ℝ) (p : ℝ) (h_r : r = 12) (h_p : p = 1 / 3) :
  ∃ A : ℝ, A = 48 * π :=
by {
  sorry
}

end win_sector_area_l187_18770


namespace baskets_count_l187_18764

theorem baskets_count (total_apples apples_per_basket : ℕ) (h1 : total_apples = 629) (h2 : apples_per_basket = 17) : (total_apples / apples_per_basket) = 37 :=
by
  sorry

end baskets_count_l187_18764


namespace triangle_angle_B_l187_18727

theorem triangle_angle_B (a b c A B C : ℝ) 
  (h1 : a * Real.cos B - b * Real.cos A = c)
  (h2 : C = Real.pi / 5) : 
  B = 3 * Real.pi / 10 := 
sorry

end triangle_angle_B_l187_18727


namespace lateral_surface_area_of_cone_l187_18773

theorem lateral_surface_area_of_cone (diameter height : ℝ) (h_d : diameter = 2) (h_h : height = 2) :
  let radius := diameter / 2
  let slant_height := Real.sqrt (radius ^ 2 + height ^ 2)
  π * radius * slant_height = Real.sqrt 5 * π := 
  by
    sorry

end lateral_surface_area_of_cone_l187_18773


namespace sqrt_of_expression_l187_18767

theorem sqrt_of_expression (x : ℝ) (h : x = 2) : Real.sqrt (2 * x - 3) = 1 :=
by
  rw [h]
  simp
  sorry

end sqrt_of_expression_l187_18767


namespace num_mystery_shelves_l187_18708

def num_books_per_shelf : ℕ := 9
def num_picture_shelves : ℕ := 2
def total_books : ℕ := 72
def num_books_from_picture_shelves : ℕ := num_picture_shelves * num_books_per_shelf
def num_books_from_mystery_shelves : ℕ := total_books - num_books_from_picture_shelves

theorem num_mystery_shelves :
  num_books_from_mystery_shelves / num_books_per_shelf = 6 := by
sorry

end num_mystery_shelves_l187_18708


namespace investment_total_correct_l187_18775

-- Define the initial investment, interest rate, and duration
def initial_investment : ℝ := 300
def monthly_interest_rate : ℝ := 0.10
def duration_in_months : ℝ := 2

-- Define the total amount after 2 months
noncomputable def total_after_two_months : ℝ := initial_investment * (1 + monthly_interest_rate) * (1 + monthly_interest_rate)

-- Define the correct answer
def correct_answer : ℝ := 363

-- The proof problem
theorem investment_total_correct :
  total_after_two_months = correct_answer :=
sorry

end investment_total_correct_l187_18775
