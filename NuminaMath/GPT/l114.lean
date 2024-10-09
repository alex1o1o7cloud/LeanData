import Mathlib

namespace remainder_when_product_divided_by_5_l114_11428

def n1 := 1483
def n2 := 1773
def n3 := 1827
def n4 := 2001
def mod5 (n : Nat) : Nat := n % 5

theorem remainder_when_product_divided_by_5 :
  mod5 (n1 * n2 * n3 * n4) = 3 :=
sorry

end remainder_when_product_divided_by_5_l114_11428


namespace find_larger_integer_l114_11407

theorem find_larger_integer (x : ℕ) (hx₁ : 4 * x > 0) (hx₂ : (x + 6) * 3 = 4 * x) : 4 * x = 72 :=
by
  sorry

end find_larger_integer_l114_11407


namespace correct_average_l114_11456

theorem correct_average (n : ℕ) (average incorrect correct : ℕ) (h1 : n = 10) (h2 : average = 15) 
(h3 : incorrect = 26) (h4 : correct = 36) :
  (n * average - incorrect + correct) / n = 16 :=
  sorry

end correct_average_l114_11456


namespace sin_cos_double_angle_identity_l114_11455

theorem sin_cos_double_angle_identity (α : ℝ) 
  (h1 : Real.sin α = 1/3) 
  (h2 : α ∈ Set.Ioc (π/2) π) : 
  Real.sin (2*α) + Real.cos (2*α) = (7 - 4 * Real.sqrt 2) / 9 := 
by
  sorry

end sin_cos_double_angle_identity_l114_11455


namespace divisible_iff_l114_11408

-- Definitions from the conditions
def a : ℕ → ℕ
  | 0     => 0
  | 1     => 1
  | (n+2) => 2 * a (n + 1) + a n

-- Main theorem statement.
theorem divisible_iff (n k : ℕ) : 2^k ∣ a n ↔ 2^k ∣ n := by
  sorry

end divisible_iff_l114_11408


namespace max_value_m_l114_11415

theorem max_value_m (x : ℝ) (h1 : 0 < x) (h2 : x < 1) :
  (∀ (m : ℝ), (4 / (1 - x) ≥ m - 1 / x)) ↔ (∃ (m : ℝ), m ≤ 9) :=
by
  sorry

end max_value_m_l114_11415


namespace find_c_l114_11476

theorem find_c {A B C : ℝ} (a b c : ℝ) (h1 : a = 3) (h2 : b = 2) 
(h3 : a * Real.sin A + b * Real.sin B - c * Real.sin C = (6 * Real.sqrt 7 / 7) * a * Real.sin B * Real.sin C) :
  c = 2 :=
sorry

end find_c_l114_11476


namespace proof_problem_l114_11443

variable (f : ℝ → ℝ)
variable (h_odd : ∀ x : ℝ, f (-x) = -f x)

-- Definition for statement 1
def statement1 := f 0 = 0

-- Definition for statement 2
def statement2 := (∃ x > 0, ∀ y > 0, f x ≥ f y) → (∃ x < 0, ∀ y < 0, f x ≤ f y)

-- Definition for statement 3
def statement3 := (∀ x ≥ 1, ∀ y ≥ 1, x < y → f x < f y) → (∀ x ≤ -1, ∀ y ≤ -1, x < y → f y < f x)

-- Definition for statement 4
def statement4 := (∀ x > 0, f x = x^2 - 2 * x) → (∀ x < 0, f x = -x^2 - 2 * x)

-- Combined proof problem
theorem proof_problem :
  (statement1 f) ∧ (statement2 f) ∧ (statement4 f) ∧ ¬ (statement3 f) :=
by sorry

end proof_problem_l114_11443


namespace count_integer_values_l114_11493

theorem count_integer_values (x : ℤ) (h1 : 4 < Real.sqrt (3 * x + 1)) (h2 : Real.sqrt (3 * x + 1) < 5) : 
  (5 < x ∧ x < 8 ∧ ∃ (N : ℕ), N = 2) :=
by sorry

end count_integer_values_l114_11493


namespace rectangle_color_invariance_l114_11484

/-- A theorem stating that in any 3x7 rectangle with some cells colored black at random, there necessarily exist four cells of the same color, whose centers are the vertices of a rectangle with sides parallel to the sides of the original rectangle. -/
theorem rectangle_color_invariance :
  ∀ (color : Fin 3 × Fin 7 → Bool), 
  ∃ i1 i2 j1 j2 : Fin 3, i1 < i2 ∧ j1 < j2 ∧ 
  color ⟨i1, j1⟩ = color ⟨i1, j2⟩ ∧ 
  color ⟨i1, j1⟩ = color ⟨i2, j1⟩ ∧ 
  color ⟨i1, j1⟩ = color ⟨i2, j2⟩ :=
by
  -- The proof is omitted
  sorry

end rectangle_color_invariance_l114_11484


namespace A_inter_B_eq_A_l114_11480

def A := {x : ℝ | 0 < x ∧ x ≤ 2}
def B := {x : ℝ | x ≤ 3}

theorem A_inter_B_eq_A : A ∩ B = A := 
by 
  sorry 

end A_inter_B_eq_A_l114_11480


namespace maximum_ratio_squared_l114_11445

theorem maximum_ratio_squared (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_ge : a ≥ b)
  (x y : ℝ) (h_x : 0 ≤ x) (h_xa : x < a) (h_y : 0 ≤ y) (h_yb : y < b)
  (h_eq1 : a^2 + y^2 = b^2 + x^2)
  (h_eq2 : b^2 + x^2 = (a - x)^2 + (b - y)^2) :
  (a / b)^2 ≤ 4 / 3 :=
sorry

end maximum_ratio_squared_l114_11445


namespace find_x_l114_11468

theorem find_x (x : ℝ) (h : 3550 - (x / 20.04) = 3500) : x = 1002 :=
by
  sorry

end find_x_l114_11468


namespace exterior_angle_of_regular_octagon_l114_11459

def sum_of_interior_angles (n : ℕ) : ℕ := 180 * (n - 2)
def interior_angle (s : ℕ) (n : ℕ) : ℕ := sum_of_interior_angles n / s
def exterior_angle (ia : ℕ) : ℕ := 180 - ia

theorem exterior_angle_of_regular_octagon : 
    exterior_angle (interior_angle 8 8) = 45 := 
by 
  sorry

end exterior_angle_of_regular_octagon_l114_11459


namespace smallest_c_such_that_one_in_range_l114_11423

theorem smallest_c_such_that_one_in_range :
  ∃ c : ℝ, (∀ x : ℝ, ∃ y : ℝ, y =  x^2 - 2 * x + c ∧ y = 1) ∧ c = 2 :=
by
  sorry

end smallest_c_such_that_one_in_range_l114_11423


namespace number_of_connections_l114_11492

theorem number_of_connections (n k : ℕ) (h1 : n = 30) (h2 : k = 4) :
  (n * k) / 2 = 60 :=
by
  sorry

end number_of_connections_l114_11492


namespace unique_two_digit_perfect_square_divisible_by_5_l114_11489

-- Define the conditions
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m * m

def two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def divisible_by_5 (n : ℕ) : Prop :=
  n % 5 = 0

-- The statement to prove: there is exactly 1 two-digit perfect square that is divisible by 5
theorem unique_two_digit_perfect_square_divisible_by_5 :
  ∃! n : ℕ, is_perfect_square n ∧ two_digit n ∧ divisible_by_5 n :=
sorry

end unique_two_digit_perfect_square_divisible_by_5_l114_11489


namespace min_increase_air_quality_days_l114_11485

theorem min_increase_air_quality_days {days_in_year : ℕ} (last_year_ratio next_year_ratio : ℝ) (good_air_days : ℕ) :
  days_in_year = 365 → last_year_ratio = 0.6 → next_year_ratio > 0.7 →
  (good_air_days / days_in_year < last_year_ratio → ∀ n: ℕ, good_air_days + n ≥ 37) :=
by
  intros hdays_in_year hlast_year_ratio hnext_year_ratio h_good_air_days
  sorry

end min_increase_air_quality_days_l114_11485


namespace simplify_fraction_expr_l114_11403

theorem simplify_fraction_expr (a : ℝ) (h : a ≠ 1) : (a / (a - 1) + 1 / (1 - a)) = 1 := by
  sorry

end simplify_fraction_expr_l114_11403


namespace meaningful_fraction_l114_11409

theorem meaningful_fraction (x : ℝ) : (x + 1 ≠ 0) ↔ (x ≠ -1) :=
by {
  sorry -- Proof goes here
}

end meaningful_fraction_l114_11409


namespace correct_operation_l114_11448

variables {a b : ℝ}

theorem correct_operation : (5 * a * b - 6 * a * b = -1 * a * b) := by
  sorry

end correct_operation_l114_11448


namespace prime_bound_l114_11420

-- The definition for the n-th prime number
def nth_prime (n : ℕ) : ℕ := sorry  -- placeholder for the primorial definition

-- The main theorem to prove
theorem prime_bound (n : ℕ) : nth_prime n ≤ 2 ^ 2 ^ (n - 1) := sorry

end prime_bound_l114_11420


namespace unicorn_tether_l114_11402

theorem unicorn_tether (a b c : ℕ) (h_c_prime : Prime c) :
  (∃ (a b c : ℕ), c = 1 ∧ (25 - 15 = 10 ∧ 10^2 + 10^2 = 15^2 ∧ 
  a = 10 ∧ b = 125) ∧ a + b + c = 136) :=
  sorry

end unicorn_tether_l114_11402


namespace possible_ages_l114_11426

-- Define the set of digits
def digits : Multiset ℕ := {1, 1, 2, 2, 2, 3}

-- Condition: The age must start with "211"
def starting_sequence : List ℕ := [2, 1, 1]

-- Calculate the count of possible ages
def count_ages : ℕ :=
  let remaining_digits := [2, 2, 1, 3]
  let total_permutations := Nat.factorial 4
  let repetitions := Nat.factorial 2
  total_permutations / repetitions

theorem possible_ages : count_ages = 12 := by
  -- Proof should go here but it's omitted according to instructions.
  sorry

end possible_ages_l114_11426


namespace root_of_quadratic_l114_11429

theorem root_of_quadratic (b : ℝ) : 
  (-9)^2 + b * (-9) - 45 = 0 -> b = 4 :=
by
  sorry

end root_of_quadratic_l114_11429


namespace benito_juarez_birth_year_l114_11400

theorem benito_juarez_birth_year (x : ℕ) (h1 : 1801 ≤ x ∧ x ≤ 1850) (h2 : x*x = 1849) : x = 1806 :=
by sorry

end benito_juarez_birth_year_l114_11400


namespace slowest_pipe_time_l114_11432

noncomputable def fill_tank_rate (R : ℝ) : Prop :=
  let rate1 := 6 * R
  let rate3 := 2 * R
  let combined_rate := 9 * R
  combined_rate = 1 / 30

theorem slowest_pipe_time (R : ℝ) (h : fill_tank_rate R) : 1 / R = 270 :=
by
  have h1 := h
  sorry

end slowest_pipe_time_l114_11432


namespace length_of_train_l114_11496

theorem length_of_train (speed_kmh : ℝ) (time_min : ℝ) (tunnel_length_m : ℝ) (train_length_m : ℝ) :
  speed_kmh = 78 → time_min = 1 → tunnel_length_m = 500 → train_length_m = 800.2 :=
by
  sorry

end length_of_train_l114_11496


namespace dan_total_purchase_cost_l114_11487

noncomputable def snake_toy_cost : ℝ := 11.76
noncomputable def cage_cost : ℝ := 14.54
noncomputable def heat_lamp_cost : ℝ := 6.25
noncomputable def cage_discount_rate : ℝ := 0.10
noncomputable def sales_tax_rate : ℝ := 0.08
noncomputable def found_dollar : ℝ := 1.00

noncomputable def total_cost : ℝ :=
  let cage_discount := cage_discount_rate * cage_cost
  let discounted_cage := cage_cost - cage_discount
  let subtotal_before_tax := snake_toy_cost + discounted_cage + heat_lamp_cost
  let sales_tax := sales_tax_rate * subtotal_before_tax
  let total_after_tax := subtotal_before_tax + sales_tax
  total_after_tax - found_dollar

theorem dan_total_purchase_cost : total_cost = 32.58 :=
  by 
    -- Placeholder for the proof
    sorry

end dan_total_purchase_cost_l114_11487


namespace find_p_l114_11469

theorem find_p (p : ℝ) :
  (∀ x : ℝ, x^2 + p * x + p - 1 = 0) →
  ((exists x1 x2 : ℝ, x^2 + p * x + p - 1 = 0 ∧ x1^2 + x1^3 = - (x2^2 + x2^3) ) → (p = 1 ∨ p = 2)) :=
by
  intro h
  sorry

end find_p_l114_11469


namespace find_n_from_sequence_l114_11439

theorem find_n_from_sequence (a : ℕ → ℝ) (h₁ : ∀ n : ℕ, a n = (1 / (Real.sqrt n + Real.sqrt (n + 1))))
  (h₂ : ∃ n : ℕ, a n + a (n + 1) = Real.sqrt 11 - 3) : 9 ∈ {n | a n + a (n + 1) = Real.sqrt 11 - 3} :=
by
  sorry

end find_n_from_sequence_l114_11439


namespace track_width_track_area_l114_11414

theorem track_width (r1 r2 : ℝ) (h1 : 2 * π * r1 - 2 * π * r2 = 24 * π) : r1 - r2 = 12 :=
by sorry

theorem track_area (r1 r2 : ℝ) (h1 : r1 = r2 + 12) : π * (r1^2 - r2^2) = π * (24 * r2 + 144) :=
by sorry

end track_width_track_area_l114_11414


namespace number_of_n_factorizable_l114_11446

theorem number_of_n_factorizable :
  ∃! n_values : Finset ℕ, (∀ n ∈ n_values, n ≤ 100 ∧ ∃ a b : ℤ, a + b = -2 ∧ a * b = -n) ∧ n_values.card = 9 := by
  sorry

end number_of_n_factorizable_l114_11446


namespace simplify_and_evaluate_expression_l114_11417

theorem simplify_and_evaluate_expression (a b : ℤ) (h₁ : a = -2) (h₂ : b = 1) :
  ((a - 2 * b) ^ 2 - (a + 3 * b) * (a - 2 * b)) / b = 20 :=
by
  sorry

end simplify_and_evaluate_expression_l114_11417


namespace largest_divisor_of_n_l114_11450

theorem largest_divisor_of_n (n : ℕ) (h_pos : 0 < n) (h_div : 7200 ∣ n^2) : 60 ∣ n := 
sorry

end largest_divisor_of_n_l114_11450


namespace students_on_bus_l114_11462

theorem students_on_bus
    (initial_students : ℝ) (first_get_on : ℝ) (first_get_off : ℝ)
    (second_get_on : ℝ) (second_get_off : ℝ)
    (third_get_on : ℝ) (third_get_off : ℝ) :
  initial_students = 21 →
  first_get_on = 7.5 → first_get_off = 2 → 
  second_get_on = 1.2 → second_get_off = 5.3 →
  third_get_on = 11 → third_get_off = 4.8 →
  (initial_students + (first_get_on - first_get_off) +
   (second_get_on - second_get_off) +
   (third_get_on - third_get_off)) = 28.6 := by
  intros
  sorry

end students_on_bus_l114_11462


namespace find_a_l114_11498

noncomputable def calculation (a : ℝ) (x : ℝ) (y : ℝ) (b : ℝ) (c : ℝ) : Prop :=
  (x * y) / (a * b * c) = 840

theorem find_a : calculation 50 0.0048 3.5 0.1 0.004 :=
by
  sorry

end find_a_l114_11498


namespace conclusion1_conclusion2_l114_11467

theorem conclusion1 (x y a b : ℝ) (h1 : 4^x = a) (h2 : 8^y = b) : 2^(2*x - 3*y) = a / b :=
sorry

theorem conclusion2 (x a : ℝ) (h1 : (x-1)*(x^2 + a*x + 1) - x^2 = x^3 - (a-1)*x^2 - (1-a)*x - 1) : a = 1 :=
sorry

end conclusion1_conclusion2_l114_11467


namespace joshua_bottle_caps_l114_11421

theorem joshua_bottle_caps (initial_caps : ℕ) (additional_caps : ℕ) (total_caps : ℕ) 
  (h1 : initial_caps = 40) 
  (h2 : additional_caps = 7) 
  (h3 : total_caps = initial_caps + additional_caps) : 
  total_caps = 47 := 
by 
  sorry

end joshua_bottle_caps_l114_11421


namespace trailing_zeros_sum_15_factorial_l114_11406

theorem trailing_zeros_sum_15_factorial : 
  let k := 5
  let h := 3
  k + h = 8 := by
  sorry

end trailing_zeros_sum_15_factorial_l114_11406


namespace simplify_fractional_exponents_l114_11495

theorem simplify_fractional_exponents :
  (5 ^ (1/6) * 5 ^ (1/2)) / 5 ^ (1/3) = 5 ^ (1/6) :=
by
  sorry

end simplify_fractional_exponents_l114_11495


namespace fg_of_3_eq_97_l114_11405

def f (x : ℕ) : ℕ := 4 * x - 3
def g (x : ℕ) : ℕ := (x + 2) ^ 2

theorem fg_of_3_eq_97 : f (g 3) = 97 := by
  sorry

end fg_of_3_eq_97_l114_11405


namespace math_problem_l114_11472

-- Definitions of the conditions
variable (x y : ℝ)
axiom h1 : x + y = 5
axiom h2 : x * y = 3

-- Prove the desired equality
theorem math_problem : x + (x^4 / y^3) + (y^4 / x^3) + y = 1021 := 
by 
sorry

end math_problem_l114_11472


namespace constant_in_quadratic_eq_l114_11488

theorem constant_in_quadratic_eq (C : ℝ) (x₁ x₂ : ℝ) 
  (h1 : 2 * x₁ * x₁ + 5 * x₁ - C = 0) 
  (h2 : 2 * x₂ * x₂ + 5 * x₂ - C = 0) 
  (h3 : x₁ - x₂ = 5.5) : C = 12 := 
sorry

end constant_in_quadratic_eq_l114_11488


namespace total_growing_space_l114_11453

noncomputable def garden_area : ℕ :=
  let area_3x3 := 3 * 3
  let total_area_3x3 := 2 * area_3x3
  let area_4x3 := 4 * 3
  let total_area_4x3 := 2 * area_4x3
  total_area_3x3 + total_area_4x3

theorem total_growing_space : garden_area = 42 :=
by
  sorry

end total_growing_space_l114_11453


namespace solve_quadratics_l114_11478

theorem solve_quadratics :
  ∃ x y : ℝ, (9 * x^2 - 36 * x - 81 = 0) ∧ (y^2 + 6 * y + 9 = 0) ∧ (x + y = -1 + Real.sqrt 13 ∨ x + y = -1 - Real.sqrt 13) := 
by 
  sorry

end solve_quadratics_l114_11478


namespace find_divisor_l114_11440

theorem find_divisor (n d : ℤ) (k : ℤ)
  (h1 : n % d = 3)
  (h2 : n^2 % d = 4) : d = 5 :=
sorry

end find_divisor_l114_11440


namespace addition_neg3_plus_2_multiplication_neg3_times_2_l114_11454

theorem addition_neg3_plus_2 : -3 + 2 = -1 :=
  by
    sorry

theorem multiplication_neg3_times_2 : (-3) * 2 = -6 :=
  by
    sorry

end addition_neg3_plus_2_multiplication_neg3_times_2_l114_11454


namespace gcd_not_perfect_square_l114_11433

theorem gcd_not_perfect_square
  (m n : ℕ)
  (h1 : (m % 3 = 0 ∨ n % 3 = 0) ∧ ¬(m % 3 = 0 ∧ n % 3 = 0))
  : ¬ ∃ k : ℕ, k * k = Nat.gcd (m^2 + n^2 + 2) (m^2 * n^2 + 3) :=
by
  sorry

end gcd_not_perfect_square_l114_11433


namespace parabola_focus_on_line_l114_11418

theorem parabola_focus_on_line (p : ℝ) (h₁ : 0 < p) (h₂ : (2 * (p / 2) + 0 - 2 = 0)) : p = 2 :=
sorry

end parabola_focus_on_line_l114_11418


namespace dave_has_20_more_than_derek_l114_11444

-- Define the amounts of money Derek and Dave start with
def initial_amount_derek : ℕ := 40
def initial_amount_dave : ℕ := 50

-- Define the amounts Derek spends
def spend_derek_lunch_self1 : ℕ := 14
def spend_derek_lunch_dad : ℕ := 11
def spend_derek_lunch_self2 : ℕ := 5
def spend_derek_dessert_sister : ℕ := 8

-- Define the amounts Dave spends
def spend_dave_lunch_mom : ℕ := 7
def spend_dave_lunch_cousin : ℕ := 12
def spend_dave_snacks_friends : ℕ := 9

-- Define calculations for total spending
def total_spend_derek : ℕ :=
  spend_derek_lunch_self1 + spend_derek_lunch_dad + spend_derek_lunch_self2 + spend_derek_dessert_sister

def total_spend_dave : ℕ :=
  spend_dave_lunch_mom + spend_dave_lunch_cousin + spend_dave_snacks_friends

-- Define remaining amount of money
def remaining_derek : ℕ :=
  initial_amount_derek - total_spend_derek

def remaining_dave : ℕ :=
  initial_amount_dave - total_spend_dave

-- Define the property to be proved
theorem dave_has_20_more_than_derek : remaining_dave - remaining_derek = 20 := by
  sorry

end dave_has_20_more_than_derek_l114_11444


namespace solution_set_inequality_l114_11431

theorem solution_set_inequality (x : ℝ) (h : 0 < x ∧ x ≤ 1) : 
  ∀ (x : ℝ), (0 < x ∧ x ≤ 1 ↔ ∀ a > 0, ∀ b ≤ 1, (2/x + (1-x) ^ (1/2) ≥ 1 + (1-x)^(1/2))) := sorry

end solution_set_inequality_l114_11431


namespace equation1_solution_equation2_no_solution_l114_11458

theorem equation1_solution (x: ℝ) (h: x ≠ -1/2 ∧ x ≠ 1):
  (1 / (x - 1) = 5 / (2 * x + 1)) ↔ (x = 2) :=
sorry

theorem equation2_no_solution (x: ℝ) (h: x ≠ 1 ∧ x ≠ -1):
  ¬ ( (x + 1) / (x - 1) - 4 / (x^2 - 1) = 1 ) :=
sorry

end equation1_solution_equation2_no_solution_l114_11458


namespace find_unknown_number_l114_11412

-- Define the problem conditions and required proof
theorem find_unknown_number (a b : ℕ) (h1 : 2 * a = 3 + b) (h2 : (a - 6)^2 = 3 * b) : b = 3 ∨ b = 27 :=
sorry

end find_unknown_number_l114_11412


namespace closest_point_l114_11465

theorem closest_point 
  (x y z : ℝ) 
  (h_plane : 3 * x - 4 * y + 5 * z = 30)
  (A : ℝ × ℝ × ℝ := (1, 2, 3)) 
  (P : ℝ × ℝ × ℝ := (x, y, z)) :
  P = (11 / 5, 2 / 5, 5) := 
sorry

end closest_point_l114_11465


namespace geom_prog_common_ratio_l114_11404

-- Definition of a geometric progression
def geom_prog (u : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n ≥ 1, u (n + 1) = u n + u (n - 1)

-- Statement of the problem
theorem geom_prog_common_ratio (u : ℕ → ℝ) (q : ℝ) (hq : ∀ n ≥ 1, u (n + 1) = u n + u (n - 1)) :
  (q = (1 + Real.sqrt 5) / 2) ∨ (q = (1 - Real.sqrt 5) / 2) :=
sorry

end geom_prog_common_ratio_l114_11404


namespace total_money_l114_11435

theorem total_money (p q r : ℕ)
  (h1 : r = 2000)
  (h2 : r = (2 / 3) * (p + q)) : 
  p + q + r = 5000 :=
by
  sorry

end total_money_l114_11435


namespace car_distance_problem_l114_11449

theorem car_distance_problem
  (d y z r : ℝ)
  (initial_distance : d = 113)
  (right_turn_distance : y = 15)
  (second_car_distance : z = 35)
  (remaining_distance : r = 28)
  (x : ℝ) :
  2 * x + z + y + r = d → 
  x = 17.5 :=
by
  intros h
  sorry  

end car_distance_problem_l114_11449


namespace solve_linear_system_l114_11490

theorem solve_linear_system (x y : ℝ) (h1 : 2 * x + 3 * y = 5) (h2 : 3 * x + 2 * y = 10) : x + y = 3 := 
by
  sorry

end solve_linear_system_l114_11490


namespace polygon_edges_l114_11461

theorem polygon_edges (n : ℕ) (h1 : (n - 2) * 180 = 4 * 360 + 180) : n = 11 :=
by {
  sorry
}

end polygon_edges_l114_11461


namespace cheaper_to_buy_more_books_l114_11470

def C (n : ℕ) : ℕ :=
  if n < 1 then 0
  else if n ≤ 20 then 15 * n
  else if n ≤ 40 then 14 * n - 5
  else 13 * n

noncomputable def apply_discount (n : ℕ) (cost : ℕ) : ℕ :=
  cost - 10 * (n / 10)

theorem cheaper_to_buy_more_books : 
  ∃ (n_vals : Finset ℕ), n_vals.card = 5 ∧ ∀ n ∈ n_vals, apply_discount (n + 1) (C (n + 1)) < apply_discount n (C n) :=
sorry

end cheaper_to_buy_more_books_l114_11470


namespace playground_area_l114_11419

theorem playground_area (w l : ℕ) (h1 : 2 * l + 2 * w = 72) (h2 : l = 3 * w) : l * w = 243 := by
  sorry

end playground_area_l114_11419


namespace problem1_problem2_problem3_problem4_l114_11451

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^3 - a * x - 1
noncomputable def f' (x : ℝ) (a : ℝ) : ℝ := 3 * x^2 - a

-- Prove that if f is increasing on ℝ, then a ∈ (-∞, 0]
theorem problem1 (a : ℝ) : (∀ x y : ℝ, x ≤ y → f x a ≤ f y a) → a ≤ 0 :=
sorry

-- Prove that if f is decreasing on (-1, 1), then a ∈ [3, ∞)
theorem problem2 (a : ℝ) : (∀ x y : ℝ, -1 < x → x < 1 → -1 < y → y < 1 → x ≤ y → f x a ≥ f y a) → 3 ≤ a :=
sorry

-- Prove that if the decreasing interval of f is (-1, 1), then a = 3
theorem problem3 (a : ℝ) : (∀ x : ℝ, (abs x < 1) ↔ f' x a < 0) → a = 3 :=
sorry

-- Prove that if f is not monotonic on (-1, 1), then a ∈ (0, 3)
theorem problem4 (a : ℝ) : (¬(∀ x : ℝ, -1 < x → x < 1 → (f' x a = 0) ∨ (f' x a ≠ 0))) → (0 < a ∧ a < 3) :=
sorry

end problem1_problem2_problem3_problem4_l114_11451


namespace initial_cherry_sweets_30_l114_11481

/-!
# Problem Statement
A packet of candy sweets has some cherry-flavored sweets (C), 40 strawberry-flavored sweets, 
and 50 pineapple-flavored sweets. Aaron eats half of each type of sweet and then gives away 
5 cherry-flavored sweets to his friend. There are still 55 sweets in the packet of candy.
Prove that the initial number of cherry-flavored sweets was 30.
-/

noncomputable def initial_cherry_sweets (C : ℕ) : Prop :=
  let remaining_cherry_sweets := C / 2 - 5
  let remaining_strawberry_sweets := 40 / 2
  let remaining_pineapple_sweets := 50 / 2
  remaining_cherry_sweets + remaining_strawberry_sweets + remaining_pineapple_sweets = 55

theorem initial_cherry_sweets_30 : initial_cherry_sweets 30 :=
  sorry

end initial_cherry_sweets_30_l114_11481


namespace probability_one_left_one_right_l114_11475

/-- Define the conditions: 12 left-handed gloves, 10 right-handed gloves. -/
def num_left_handed_gloves : ℕ := 12

def num_right_handed_gloves : ℕ := 10

/-- Total number of gloves is 22. -/
def total_gloves : ℕ := num_left_handed_gloves + num_right_handed_gloves

/-- Total number of ways to pick any two gloves from 22 gloves. -/
def total_pick_two_ways : ℕ := (total_gloves * (total_gloves - 1)) / 2

/-- Number of favorable outcomes picking one left-handed and one right-handed glove. -/
def favorable_outcomes : ℕ := num_left_handed_gloves * num_right_handed_gloves

/-- Define the probability as favorable outcomes divided by total outcomes. 
 It should yield 40/77. -/
theorem probability_one_left_one_right : 
  (favorable_outcomes : ℚ) / total_pick_two_ways = 40 / 77 :=
by
  -- Skip the proof.
  sorry

end probability_one_left_one_right_l114_11475


namespace child_stops_incur_yearly_cost_at_age_18_l114_11427

def john_contribution (years: ℕ) (cost_per_year: ℕ) : ℕ :=
  years * cost_per_year / 2

def university_contribution (university_cost: ℕ) : ℕ :=
  university_cost / 2

def total_contribution (years_after_8: ℕ) : ℕ :=
  john_contribution 8 10000 +
  john_contribution years_after_8 20000 +
  university_contribution 250000

theorem child_stops_incur_yearly_cost_at_age_18 :
  (total_contribution n = 265000) → (n + 8 = 18) :=
by
  sorry

end child_stops_incur_yearly_cost_at_age_18_l114_11427


namespace synthetic_analytic_incorrect_statement_l114_11447

theorem synthetic_analytic_incorrect_statement
  (basic_methods : ∀ (P Q : Prop), (P → Q) ∨ (Q → P))
  (synthetic_forward : ∀ (P Q : Prop), (P → Q))
  (analytic_backward : ∀ (P Q : Prop), (Q → P)) :
  ¬ (∀ (P Q : Prop), (P → Q) ∧ (Q → P)) :=
by
  sorry

end synthetic_analytic_incorrect_statement_l114_11447


namespace b_gets_more_than_c_l114_11438

-- Define A, B, and C as real numbers
variables (A B C : ℝ)

theorem b_gets_more_than_c 
  (h1 : A = 3 * B)
  (h2 : B = C + 25)
  (h3 : A + B + C = 645)
  (h4 : B = 134) : 
  B - C = 25 :=
by
  -- Using the conditions from the problem
  sorry

end b_gets_more_than_c_l114_11438


namespace total_area_calculations_l114_11422

noncomputable def total_area_in_hectares : ℝ :=
  let sections := 5
  let area_per_section := 60
  let conversion_factor_acre_to_hectare := 0.404686
  sections * area_per_section * conversion_factor_acre_to_hectare

noncomputable def total_area_in_square_meters : ℝ :=
  let conversion_factor_hectare_to_square_meter := 10000
  total_area_in_hectares * conversion_factor_hectare_to_square_meter

theorem total_area_calculations :
  total_area_in_hectares = 121.4058 ∧ total_area_in_square_meters = 1214058 := by
  sorry

end total_area_calculations_l114_11422


namespace boxes_needed_l114_11460

theorem boxes_needed (total_muffins available_boxes muffins_per_box : ℕ) (h1 : total_muffins = 95) 
  (h2 : available_boxes = 10) (h3 : muffins_per_box = 5) : 
  ((total_muffins - (available_boxes * muffins_per_box)) / muffins_per_box) = 9 := 
by
  -- the proof will be constructed here
  sorry

end boxes_needed_l114_11460


namespace triangle_third_side_length_l114_11401

theorem triangle_third_side_length (a b : ℕ) (h1 : a = 2) (h2 : b = 3) 
(h3 : ∃ x, x^2 - 10 * x + 21 = 0 ∧ (a + b > x) ∧ (a + x > b) ∧ (b + x > a)) :
  ∃ x, x = 3 := 
by 
  sorry

end triangle_third_side_length_l114_11401


namespace problem1_l114_11471

theorem problem1 (α : ℝ) (h : Real.tan (π/4 + α) = 2) : Real.sin (2 * α) + Real.cos α ^ 2 = 3 / 2 := 
sorry

end problem1_l114_11471


namespace average_speed_trip_l114_11452

theorem average_speed_trip :
  let distance_1 := 65
  let distance_2 := 45
  let distance_3 := 55
  let distance_4 := 70
  let distance_5 := 60
  let total_time := 5
  let total_distance := distance_1 + distance_2 + distance_3 + distance_4 + distance_5
  let average_speed := total_distance / total_time
  average_speed = 59 :=
by
  sorry

end average_speed_trip_l114_11452


namespace determine_house_height_l114_11474

-- Definitions for the conditions
def house_shadow : ℚ := 75
def tree_height : ℚ := 15
def tree_shadow : ℚ := 20

-- Desired Height of Lily's house
def house_height : ℚ := 56

-- Theorem stating the height of the house
theorem determine_house_height :
  (house_shadow / tree_shadow = house_height / tree_height) -> house_height = 56 :=
  by
  unfold house_shadow tree_height tree_shadow house_height
  sorry

end determine_house_height_l114_11474


namespace average_speed_of_train_l114_11491

theorem average_speed_of_train (distance time : ℝ) (h1 : distance = 80) (h2 : time = 8) :
  distance / time = 10 :=
by
  sorry

end average_speed_of_train_l114_11491


namespace lcm_18_28_45_65_eq_16380_l114_11411

theorem lcm_18_28_45_65_eq_16380 : Nat.lcm 18 (Nat.lcm 28 (Nat.lcm 45 65)) = 16380 :=
sorry

end lcm_18_28_45_65_eq_16380_l114_11411


namespace steve_speed_back_l114_11436

open Real

noncomputable def steves_speed_on_way_back : ℝ := 15

theorem steve_speed_back
  (distance_to_work : ℝ)
  (traffic_time_to_work : ℝ)
  (traffic_time_back : ℝ)
  (total_time : ℝ)
  (speed_ratio : ℝ) :
  distance_to_work = 30 →
  traffic_time_to_work = 30 →
  traffic_time_back = 15 →
  total_time = 405 →
  speed_ratio = 2 →
  steves_speed_on_way_back = 15 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end steve_speed_back_l114_11436


namespace children_difference_l114_11437

theorem children_difference (initial_count : ℕ) (remaining_count : ℕ) (difference : ℕ) 
  (h1 : initial_count = 41) (h2 : remaining_count = 18) :
  difference = initial_count - remaining_count := 
by
  sorry

end children_difference_l114_11437


namespace polygon_interior_sum_sum_of_exterior_angles_l114_11497

theorem polygon_interior_sum (n : ℕ) (h : (n - 2) * 180 = 1080) : n = 8 :=
by
  sorry

theorem sum_of_exterior_angles (n : ℕ) : 360 = 360 :=
by
  sorry

end polygon_interior_sum_sum_of_exterior_angles_l114_11497


namespace problem_statement_l114_11457

-- Definitions of A and B based on the given conditions
def A : ℤ := -5 * -3
def B : ℤ := 2 - 2

-- The theorem stating that A + B = 15
theorem problem_statement : A + B = 15 := 
by 
  sorry

end problem_statement_l114_11457


namespace cylinder_volume_increase_l114_11410

variable (r h : ℝ)

theorem cylinder_volume_increase :
  (π * (4 * r) ^ 2 * (2 * h)) = 32 * (π * r ^ 2 * h) :=
by
  sorry

end cylinder_volume_increase_l114_11410


namespace balloon_count_correct_l114_11434

def gold_balloons : ℕ := 141
def black_balloons : ℕ := 150
def silver_balloons : ℕ := 2 * gold_balloons
def total_balloons : ℕ := gold_balloons + silver_balloons + black_balloons

theorem balloon_count_correct : total_balloons = 573 := by
  sorry

end balloon_count_correct_l114_11434


namespace Vikki_take_home_pay_is_correct_l114_11430

noncomputable def Vikki_take_home_pay : ℝ :=
  let hours_worked : ℝ := 42
  let hourly_pay_rate : ℝ := 12
  let gross_earnings : ℝ := hours_worked * hourly_pay_rate

  let fed_tax_first_300 : ℝ := 300 * 0.15
  let amount_over_300 : ℝ := gross_earnings - 300
  let fed_tax_excess : ℝ := amount_over_300 * 0.22
  let total_federal_tax : ℝ := fed_tax_first_300 + fed_tax_excess

  let state_tax : ℝ := gross_earnings * 0.07
  let retirement_contribution : ℝ := gross_earnings * 0.06
  let insurance_cover : ℝ := gross_earnings * 0.03
  let union_dues : ℝ := 5

  let total_deductions : ℝ := total_federal_tax + state_tax + retirement_contribution + insurance_cover + union_dues
  let take_home_pay : ℝ := gross_earnings - total_deductions
  take_home_pay

theorem Vikki_take_home_pay_is_correct : Vikki_take_home_pay = 328.48 :=
by
  sorry

end Vikki_take_home_pay_is_correct_l114_11430


namespace monotonicity_and_range_l114_11483

noncomputable def f (a x : ℝ) : ℝ := a^2 * x^2 + a * x - 3 * Real.log x + 1

theorem monotonicity_and_range (a : ℝ) (h : a > 0) :
  (∀ x : ℝ, 0 < x ∧ x < 1/a → f a x < f a (1/a)) ∧ 
  (∀ x : ℝ, x > 1/a → f a x > f a (1/a)) ∧ 
  (∀ x : ℝ, f a x ≠ 0 → a > 1/Real.exp 1) :=
by
  sorry

end monotonicity_and_range_l114_11483


namespace smallest_a_value_l114_11425

theorem smallest_a_value (α β γ : ℕ) (hαβγ : α * β * γ = 2010) (hα : α > 0) (hβ : β > 0) (hγ : γ > 0) :
  α + β + γ = 78 :=
by
-- Proof would go here
sorry

end smallest_a_value_l114_11425


namespace arithmetic_sequence_sum_l114_11442

theorem arithmetic_sequence_sum 
  (a : ℕ → ℝ) 
  (h_arith : ∃ (d : ℝ), ∀ (n : ℕ), a (n + 1) = a n + d) 
  (h1 : a 1 + a 5 = 2) 
  (h2 : a 2 + a 14 = 12) : 
  (10 / 2) * (2 * a 1 + 9 * d) = 35 :=
by
  sorry

end arithmetic_sequence_sum_l114_11442


namespace restore_example_l114_11416

theorem restore_example (x : ℕ) (y : ℕ) :
  (10 ≤ x * 8 ∧ x * 8 < 100) ∧ (100 ≤ x * 9 ∧ x * 9 < 1000) ∧ y = 98 → x = 12 ∧ x * y = 1176 :=
by
  sorry

end restore_example_l114_11416


namespace weight_of_6m_rod_l114_11413

theorem weight_of_6m_rod (r ρ : ℝ) (h₁ : 11.25 > 0) (h₂ : 6 > 0) (h₃ : 0 < r) (h₄ : 42.75 = π * r^2 * 11.25 * ρ) : 
  (π * r^2 * 6 * (42.75 / (π * r^2 * 11.25))) = 22.8 :=
by
  sorry

end weight_of_6m_rod_l114_11413


namespace resistance_of_second_resistor_l114_11424

theorem resistance_of_second_resistor 
  (R1 R_total R2 : ℝ) 
  (hR1: R1 = 9) 
  (hR_total: R_total = 4.235294117647059) 
  (hFormula: 1/R_total = 1/R1 + 1/R2) : 
  R2 = 8 :=
by
  sorry

end resistance_of_second_resistor_l114_11424


namespace angle_A_area_triangle_l114_11466

-- The first problem: Proving angle A
theorem angle_A (a b c : ℝ) (A C : ℝ) 
  (h1 : (2 * b - c) * Real.cos A = a * Real.cos C) : 
  A = Real.pi / 3 :=
by sorry

-- The second problem: Finding the area of triangle ABC
theorem area_triangle (a b c : ℝ) (A : ℝ)
  (h1 : a = 3)
  (h2 : b = 2 * c)
  (h3 : A = Real.pi / 3) :
  0.5 * b * c * Real.sin A = 3 * Real.sqrt 3 / 2 :=
by sorry

end angle_A_area_triangle_l114_11466


namespace find_a2_l114_11482

open Classical

variable {a_n : ℕ → ℝ} {q : ℝ}

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n m : ℕ, a (n + m) = a n * q ^ m

theorem find_a2 (h1 : geometric_sequence a_n q)
                (h2 : a_n 7 = 1 / 4)
                (h3 : a_n 3 * a_n 5 = 4 * (a_n 4 - 1)) :
  a_n 2 = 8 :=
sorry

end find_a2_l114_11482


namespace star_k_l114_11479

def star (x y : ℤ) : ℤ := x^2 - 2 * y + 1

theorem star_k (k : ℤ) : star k (star k k) = -k^2 + 4 * k - 1 :=
by 
  sorry

end star_k_l114_11479


namespace price_of_20_percent_stock_l114_11477

theorem price_of_20_percent_stock (annual_income : ℝ) (investment : ℝ) (dividend_rate : ℝ) (price_of_stock : ℝ) :
  annual_income = 1000 →
  investment = 6800 →
  dividend_rate = 20 →
  price_of_stock = 136 :=
by
  intros h_income h_investment h_dividend_rate
  sorry

end price_of_20_percent_stock_l114_11477


namespace one_fifth_of_ten_x_plus_three_l114_11486

theorem one_fifth_of_ten_x_plus_three (x : ℝ) : 
  (1 / 5) * (10 * x + 3) = 2 * x + 3 / 5 := 
  sorry

end one_fifth_of_ten_x_plus_three_l114_11486


namespace problem1_problem2_problem3_l114_11464

-- Proof Problem 1: $A$ and $B$ are not standing together
theorem problem1 : 
  ∃ (n : ℕ), n = 480 ∧ 
  ∀ (students : Fin 6 → String),
    students 0 ≠ "A" ∨ students 1 ≠ "B" :=
sorry

-- Proof Problem 2: $C$ and $D$ must stand together
theorem problem2 : 
  ∃ (n : ℕ), n = 240 ∧ 
  ∀ (students : Fin 6 → String),
    (students 0 = "C" ∧ students 1 = "D") ∨ 
    (students 1 = "C" ∧ students 2 = "D") :=
sorry

-- Proof Problem 3: $E$ is not at the beginning and $F$ is not at the end
theorem problem3 : 
  ∃ (n : ℕ), n = 504 ∧ 
  ∀ (students : Fin 6 → String),
    students 0 ≠ "E" ∧ students 5 ≠ "F" :=
sorry

end problem1_problem2_problem3_l114_11464


namespace percentage_of_students_wearing_red_shirts_l114_11463

/-- In a school of 700 students:
    - 45% of students wear blue shirts.
    - 15% of students wear green shirts.
    - 119 students wear shirts of other colors.
    We are proving that the percentage of students wearing red shirts is 23%. --/
theorem percentage_of_students_wearing_red_shirts:
  let total_students := 700
  let blue_shirt_percentage := 45 / 100
  let green_shirt_percentage := 15 / 100
  let other_colors_students := 119
  let students_with_blue_shirts := blue_shirt_percentage * total_students
  let students_with_green_shirts := green_shirt_percentage * total_students
  let students_with_other_colors := other_colors_students
  let students_with_blue_green_or_red_shirts := total_students - students_with_other_colors
  let students_with_red_shirts := students_with_blue_green_or_red_shirts - students_with_blue_shirts - students_with_green_shirts
  (students_with_red_shirts / total_students) * 100 = 23 := by
  sorry

end percentage_of_students_wearing_red_shirts_l114_11463


namespace find_values_l114_11473

theorem find_values (a b c : ℕ) 
    (h1 : a + b + c = 1024) 
    (h2 : c = b - 88) 
    (h3 : a = b + c) : 
    a = 712 ∧ b = 400 ∧ c = 312 :=
by {
    sorry
}

end find_values_l114_11473


namespace product_of_cubes_l114_11494

theorem product_of_cubes :
  ( (2^3 - 1) / (2^3 + 1) * (3^3 - 1) / (3^3 + 1) * (4^3 - 1) / (4^3 + 1) * 
    (5^3 - 1) / (5^3 + 1) * (6^3 - 1) / (6^3 + 1) * (7^3 - 1) / (7^3 + 1) 
  ) = 57 / 72 := 
by
  sorry

end product_of_cubes_l114_11494


namespace quadrant_conditions_l114_11499

-- Formalizing function and conditions in Lean specifics
variable {a b : ℝ}

theorem quadrant_conditions 
  (h1 : a > 0) 
  (h2 : a ≠ 1) 
  (h3 : 0 < a ∧ a < 1)
  (h4 : ∀ x < 0, a^x + b - 1 > 0)
  (h5 : ∀ x > 0, a^x + b - 1 > 0) :
  0 < b ∧ b < 1 := 
sorry

end quadrant_conditions_l114_11499


namespace arithmetic_sequence_index_l114_11441

theorem arithmetic_sequence_index (a : ℕ → ℕ) (n : ℕ) (first_term comm_diff : ℕ):
  (∀ k, a k = first_term + comm_diff * (k - 1)) → a n = 2016 → n = 404 :=
by 
  sorry

end arithmetic_sequence_index_l114_11441
