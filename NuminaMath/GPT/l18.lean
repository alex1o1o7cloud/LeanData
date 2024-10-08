import Mathlib

namespace papi_calot_additional_plants_l18_18945

def initial_plants := 7 * 18

def total_plants := 141

def additional_plants := total_plants - initial_plants

theorem papi_calot_additional_plants : additional_plants = 15 :=
by
  sorry

end papi_calot_additional_plants_l18_18945


namespace gallons_added_in_fourth_hour_l18_18856

-- Defining the conditions
def initial_volume : ℕ := 40
def loss_rate_per_hour : ℕ := 2
def add_in_third_hour : ℕ := 1
def remaining_after_fourth_hour : ℕ := 36

-- Prove the problem statement
theorem gallons_added_in_fourth_hour :
  ∃ (x : ℕ), initial_volume - 2 * 4 + 1 - loss_rate_per_hour + x = remaining_after_fourth_hour :=
sorry

end gallons_added_in_fourth_hour_l18_18856


namespace percentage_increase_l18_18023

variable {x y : ℝ}
variable {P : ℝ} -- percentage

theorem percentage_increase (h1 : y = x * (1 + P / 100)) (h2 : x = y * 0.5882352941176471) : P = 70 := 
by
  sorry

end percentage_increase_l18_18023


namespace baker_new_cakes_l18_18696

theorem baker_new_cakes :
  ∀ (initial_bought new_bought sold final : ℕ),
  initial_bought = 173 →
  sold = 86 →
  final = 190 →
  final = initial_bought + new_bought - sold →
  new_bought = 103 :=
by
  intros initial_bought new_bought sold final H_initial H_sold H_final H_eq
  sorry

end baker_new_cakes_l18_18696


namespace find_constants_and_formula_l18_18211

namespace ArithmeticSequence

variable {a : ℕ → ℤ} -- Sequence a : ℕ → ℤ

-- Given conditions
axiom a_5 : a 5 = 11
axiom a_12 : a 12 = 31

-- Definitions to be proved
def a_1 := -2
def d := 3
def a_formula (n : ℕ) := a_1 + (n - 1) * d

theorem find_constants_and_formula :
  (a 1 = a_1) ∧
  (a 2 - a 1 = d) ∧
  (a 20 = 55) ∧
  (∀ n, a n = a_formula n) := by
  sorry

end ArithmeticSequence

end find_constants_and_formula_l18_18211


namespace positive_integer_satisfies_condition_l18_18079

def num_satisfying_pos_integers : ℕ :=
  1

theorem positive_integer_satisfies_condition :
  ∃ (n : ℕ), 16 - 4 * n > 10 ∧ n = num_satisfying_pos_integers := by
  sorry

end positive_integer_satisfies_condition_l18_18079


namespace complex_power_identity_l18_18544

theorem complex_power_identity (i : ℂ) (hi : i^2 = -1) :
  ( (1 + i) / (1 - i) ) ^ 2013 = i :=
by sorry

end complex_power_identity_l18_18544


namespace bottles_needed_l18_18634

-- Define specific values provided in conditions
def servings_per_guest : ℕ := 2
def number_of_guests : ℕ := 120
def servings_per_bottle : ℕ := 6

-- Define total servings needed
def total_servings : ℕ := servings_per_guest * number_of_guests

-- Define the number of bottles needed (as a proof statement)
theorem bottles_needed : total_servings / servings_per_bottle = 40 := by
  /-
    The proof will go here. For now we place a sorry to mark the place where
    a proof would be required. The statement should check the equivalence of 
    number of bottles needed being 40 given the total servings divided by 
    servings per bottle.
  -/
  sorry

end bottles_needed_l18_18634


namespace set_intersection_eq_l18_18507

def A : Set ℝ := {x | |x - 1| ≤ 2}
def B : Set ℝ := {x | x^2 - 4 * x > 0}

theorem set_intersection_eq :
  A ∩ (Set.univ \ B) = {x | 0 ≤ x ∧ x ≤ 3} := by
  sorry

end set_intersection_eq_l18_18507


namespace remainder_sum_of_first_six_primes_div_seventh_prime_l18_18422

theorem remainder_sum_of_first_six_primes_div_seventh_prime :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  let p6 := 13
  let p7 := 17
  ((p1 + p2 + p3 + p4 + p5 + p6) % p7) = 7 :=
by
  sorry

end remainder_sum_of_first_six_primes_div_seventh_prime_l18_18422


namespace tan_sum_eq_tan_product_l18_18467

theorem tan_sum_eq_tan_product {α β γ : ℝ} 
  (h_sum : α + β + γ = π) : 
    Real.tan α + Real.tan β + Real.tan γ = Real.tan α * Real.tan β * Real.tan γ :=
by
  sorry

end tan_sum_eq_tan_product_l18_18467


namespace calculate_expression_l18_18670

theorem calculate_expression :
  (Int.floor ((15:ℚ)/8 * ((-34:ℚ)/4)) - Int.ceil ((15:ℚ)/8 * Int.floor ((-34:ℚ)/4))) = 0 := 
  by sorry

end calculate_expression_l18_18670


namespace geometric_sequence_from_second_term_l18_18320

theorem geometric_sequence_from_second_term (S : ℕ → ℕ) (a : ℕ → ℕ) :
  S 1 = 1 ∧ S 2 = 2 ∧ (∀ n, n ≥ 2 → S (n + 1) - 3 * S n + 2 * S (n - 1) = 0) →
  (∀ n, n ≥ 2 → a (n + 1) = 2 * a n) :=
by
  sorry

end geometric_sequence_from_second_term_l18_18320


namespace find_seven_m_squared_minus_one_l18_18791

theorem find_seven_m_squared_minus_one (m : ℝ)
  (h1 : ∃ x₁, 5 * m + 3 * x₁ = 1 + x₁)
  (h2 : ∃ x₂, 2 * x₂ + m = 3 * m)
  (h3 : ∀ x₁ x₂, (5 * m + 3 * x₁ = 1 + x₁) → (2 * x₂ + m = 3 * m) → x₁ = x₂ + 2) :
  7 * m^2 - 1 = 2 / 7 :=
by
  let m := -3/7
  sorry

end find_seven_m_squared_minus_one_l18_18791


namespace An_is_integer_for_all_n_l18_18036

noncomputable def sin_theta (a b : ℕ) : ℝ :=
  if h : a^2 + b^2 ≠ 0 then (2 * a * b) / (a^2 + b^2) else 0

theorem An_is_integer_for_all_n (a b : ℕ) (n : ℕ) (h₁ : a > b) (h₂ : 0 < sin_theta a b) (h₃ : sin_theta a b < 1) :
  ∃ k : ℤ, ∀ n : ℕ, ((a^2 + b^2)^n * sin_theta a b) = k :=
sorry

end An_is_integer_for_all_n_l18_18036


namespace function_range_l18_18032

def function_defined (x : ℝ) : Prop := x ≠ 5

theorem function_range (x : ℝ) : x ≠ 5 → function_defined x :=
by
  intro h
  exact h

end function_range_l18_18032


namespace xz_less_than_half_l18_18858

theorem xz_less_than_half (x y z : ℝ) (h1 : x ≥ y) (h2 : y ≥ z) (h3 : xy + yz + zx = 1) : x * z < 1 / 2 :=
  sorry

end xz_less_than_half_l18_18858


namespace sequence_is_increasing_l18_18508

variable (a_n : ℕ → ℝ)

def sequence_positive_numbers (a_n : ℕ → ℝ) : Prop :=
∀ n, 0 < a_n n

def sequence_condition (a_n : ℕ → ℝ) : Prop :=
∀ n, a_n (n + 1) = 2 * a_n n

theorem sequence_is_increasing 
  (h1 : sequence_positive_numbers a_n) 
  (h2 : sequence_condition a_n) : 
  ∀ n, a_n (n + 1) > a_n n :=
by
  sorry

end sequence_is_increasing_l18_18508


namespace point_not_on_graph_l18_18658

theorem point_not_on_graph :
  ¬ (1 * 5 = 6) :=
by 
  sorry

end point_not_on_graph_l18_18658


namespace parabola_focus_coords_l18_18734

-- Define the parabola equation
def parabola (x y : ℝ) : Prop := y^2 = 4 * x

-- Define the focus coordinates
def focus (x y : ℝ) : Prop := (x, y) = (1, 0)

-- The math proof problem statement
theorem parabola_focus_coords :
  ∀ x y, parabola x y → focus x y :=
by
  intros x y hp
  sorry

end parabola_focus_coords_l18_18734


namespace right_triangle_inequality_l18_18090

variable (a b c : ℝ)

theorem right_triangle_inequality
  (h1 : b < a) -- shorter leg is less than longer leg
  (h2 : c = Real.sqrt (a^2 + b^2)) -- hypotenuse from Pythagorean theorem
  : a + b / 2 > c ∧ c > (8 / 9) * (a + b / 2) := 
sorry

end right_triangle_inequality_l18_18090


namespace place_value_ratio_l18_18753

theorem place_value_ratio :
  let d8_place := 0.1
  let d7_place := 10
  d8_place / d7_place = 0.01 :=
by
  -- proof skipped
  sorry

end place_value_ratio_l18_18753


namespace find_x_l18_18433

theorem find_x (x : ℝ) (h : 0.25 * x = 0.12 * 1500 - 15) : x = 660 :=
by
  -- Proof goes here
  sorry

end find_x_l18_18433


namespace xy_condition_l18_18697

theorem xy_condition (x y z : ℝ) (hxz : x ≠ z) (hxy : x ≠ y) (hyz : y ≠ z) (posx : 0 < x) (posy : 0 < y) (posz : 0 < z) 
  (h : y / (x - z) = (x + y) / z ∧ (x + y) / z = x / y) : x / y = 2 :=
by
  sorry

end xy_condition_l18_18697


namespace profit_percentage_is_10_percent_l18_18762

theorem profit_percentage_is_10_percent
  (market_price_per_pen : ℕ)
  (retailer_buys_40_pens_for_36_price : 40 * market_price_per_pen = 36 * market_price_per_pen)
  (discount_percentage : ℕ)
  (selling_price_with_discount : ℕ) :
  discount_percentage = 1 →
  selling_price_with_discount = market_price_per_pen - (market_price_per_pen / 100) →
  (selling_price_with_discount * 40 - 36 * market_price_per_pen) / (36 * market_price_per_pen) * 100 = 10 :=
by
  sorry

end profit_percentage_is_10_percent_l18_18762


namespace greatest_perfect_square_power_of_3_under_200_l18_18691

theorem greatest_perfect_square_power_of_3_under_200 :
  ∃ n : ℕ, n < 200 ∧ (∃ k : ℕ, k % 2 = 0 ∧ n = 3 ^ k) ∧ ∀ m : ℕ, (m < 200 ∧ (∃ k : ℕ, k % 2 = 0 ∧ m = 3 ^ k)) → m ≤ n :=
  sorry

end greatest_perfect_square_power_of_3_under_200_l18_18691


namespace total_points_l18_18502

noncomputable def Darius_points : ℕ := 10
noncomputable def Marius_points : ℕ := Darius_points + 3
noncomputable def Matt_points : ℕ := Darius_points + 5
noncomputable def Sofia_points : ℕ := 2 * Matt_points

theorem total_points : Darius_points + Marius_points + Matt_points + Sofia_points = 68 :=
by
  -- Definitions are directly from the problem statement, proof skipped 
  sorry

end total_points_l18_18502


namespace RandomEvent_Proof_l18_18432

-- Define the events and conditions
def EventA : Prop := ∀ (θ₁ θ₂ θ₃ : ℝ), θ₁ + θ₂ + θ₃ = 360 → False
def EventB : Prop := ∀ (n : ℕ), (1 ≤ n ∧ n ≤ 6) → n < 7
def EventC : Prop := ∃ (factors : ℕ → ℝ), (∃ (uncertainty : ℕ → ℝ), True)
def EventD : Prop := ∀ (balls : ℕ), (balls = 0 ∨ balls ≠ 0) → False

-- The theorem represents the proof problem
theorem RandomEvent_Proof : EventC :=
by
  sorry

end RandomEvent_Proof_l18_18432


namespace min_value_of_expression_l18_18445

theorem min_value_of_expression
  (a b : ℝ)
  (ha : a > 0)
  (hb : b > 0)
  (hlines : (∀ x y : ℝ, x + (a-4) * y + 1 = 0) ∧ (∀ x y : ℝ, 2 * b * x + y - 2 = 0) ∧ (∀ x y : ℝ, (x + (a-4) * y + 1 = 0) ∧ (2 * b * x + y - 2 = 0) → -1 * 1 / (a-4) * -2 * b = 1)) :
  ∃ (min_val : ℝ), min_val = (9/5) ∧ min_val = (a + 2)/(a + 1) + 1/(2 * b) :=
by
  sorry

end min_value_of_expression_l18_18445


namespace minimum_value_768_l18_18721

noncomputable def min_value_expression (a b c : ℝ) := a^2 + 8 * a * b + 16 * b^2 + 2 * c^5

theorem minimum_value_768 (a b c : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_condition : a * b^2 * c^3 = 256) : 
  min_value_expression a b c = 768 :=
sorry

end minimum_value_768_l18_18721


namespace train_crosses_man_in_6_seconds_l18_18678

/-- A train of length 240 meters, traveling at a speed of 144 km/h, will take 6 seconds to cross a man standing on the platform. -/
theorem train_crosses_man_in_6_seconds
  (length_of_train : ℕ)
  (speed_of_train : ℕ)
  (conversion_factor : ℕ)
  (speed_in_m_per_s : ℕ)
  (time_to_cross : ℕ)
  (h1 : length_of_train = 240)
  (h2 : speed_of_train = 144)
  (h3 : conversion_factor = 1000 / 3600)
  (h4 : speed_in_m_per_s = speed_of_train * conversion_factor)
  (h5 : speed_in_m_per_s = 40)
  (h6 : time_to_cross = length_of_train / speed_in_m_per_s) :
  time_to_cross = 6 := by
  sorry

end train_crosses_man_in_6_seconds_l18_18678


namespace problem_solution_l18_18622

-- Definitions for given conditions
variables {a_n b_n : ℕ → ℝ} -- Sequences {a_n} and {b_n}
variables {S T : ℕ → ℝ} -- Sums of the first n terms of {a_n} and {b_n}
variables (h1 : ∀ n, S n = (n * (a_n 1 + a_n n)) / 2)
variables (h2 : ∀ n, T n = (n * (b_n 1 + b_n n)) / 2)
variables (h3 : ∀ n, n > 0 → S n / T n = (2 * n + 1) / (n + 2))

-- The goal
theorem problem_solution :
  (a_n 7) / (b_n 7) = 9 / 5 :=
sorry

end problem_solution_l18_18622


namespace find_z_l18_18909

variable {x y z : ℝ}

theorem find_z (h : (1/x + 1/y = 1/z)) : z = (x * y) / (x + y) :=
  sorry

end find_z_l18_18909


namespace train_half_speed_time_l18_18533

-- Definitions for Lean
variables (S T D : ℝ)

-- Conditions
axiom cond1 : D = S * T
axiom cond2 : D = (1 / 2) * S * (T + 4)

-- Theorem Statement
theorem train_half_speed_time : 
  (T = 4) → (4 + 4 = 8) := 
by 
  intros hT
  simp [hT]

end train_half_speed_time_l18_18533


namespace problem_l18_18307

variable {a b c d : ℝ}

theorem problem (h1 : a > b) (h2 : b > c) (h3 : c > d) :
  (1 / (a - b)) + (1 / (b - c)) + (1 / (c - d)) ≥ (9 / (a - d)) :=
sorry

end problem_l18_18307


namespace no_solutions_for_sin_cos_eq_sqrt3_l18_18915

theorem no_solutions_for_sin_cos_eq_sqrt3 (x : ℝ) (hx : 0 ≤ x ∧ x < 2 * Real.pi) :
  ¬ (Real.sin x + Real.cos x = Real.sqrt 3) :=
by
  sorry

end no_solutions_for_sin_cos_eq_sqrt3_l18_18915


namespace find_the_number_l18_18883

theorem find_the_number 
  (x y n : ℤ)
  (h : 19 * (x + y) + 17 = 19 * (-x + y) - n)
  (hx : x = 1) :
  n = -55 :=
by
  sorry

end find_the_number_l18_18883


namespace circle_radius_l18_18372

-- Parameters of the problem
variables (k : ℝ) (r : ℝ)
-- Conditions
axiom cond_k_positive : k > 8
axiom tangency_y_8 : r = k - 8
axiom tangency_y_x : r = k / (Real.sqrt 2)

-- Statement to prove
theorem circle_radius (k : ℝ) (hk : k > 8) (r : ℝ) (hr1 : r = k - 8) (hr2 : r = k / (Real.sqrt 2)) : r = 8 * Real.sqrt 2 + 8 :=
sorry

end circle_radius_l18_18372


namespace satisfying_lines_l18_18842

theorem satisfying_lines (x y : ℝ) : (y^2 - 2*y = x^2 + 2*x) ↔ (y = x + 2 ∨ y = -x) :=
by
  sorry

end satisfying_lines_l18_18842


namespace find_number_l18_18645

-- Define the conditions as stated in the problem
def fifteen_percent_of_x_is_ninety (x : ℝ) : Prop :=
  (15 / 100) * x = 90

-- Define the theorem to prove that given the condition, x must be 600
theorem find_number (x : ℝ) (h : fifteen_percent_of_x_is_ninety x) : x = 600 :=
sorry

end find_number_l18_18645


namespace team_selection_l18_18027

theorem team_selection :
  let teachers := 5
  let students := 10
  (teachers * students = 50) :=
by
  sorry

end team_selection_l18_18027


namespace find_n_l18_18675

theorem find_n (a b : ℤ) (h₁ : a ≡ 25 [ZMOD 42]) (h₂ : b ≡ 63 [ZMOD 42]) :
  ∃ n, 200 ≤ n ∧ n ≤ 241 ∧ (a - b ≡ n [ZMOD 42]) ∧ n = 214 :=
by
  sorry

end find_n_l18_18675


namespace petyas_number_l18_18256

theorem petyas_number :
  ∃ (N : ℕ), 
  (N % 2 = 1 ∧ ∃ (M : ℕ), N = 149 * M ∧ (M = Nat.mod (N : ℕ) (100))) →
  (N = 745 ∨ N = 3725) :=
by
  sorry

end petyas_number_l18_18256


namespace number_of_ordered_pairs_l18_18677

theorem number_of_ordered_pairs : 
  ∃ (S : Finset (ℕ × ℕ)), (∀ x ∈ S, (x.1 * x.2 = 64) ∧ (x.1 > 0) ∧ (x.2 > 0)) ∧ S.card = 7 := 
sorry

end number_of_ordered_pairs_l18_18677


namespace concert_ticket_cost_l18_18415

theorem concert_ticket_cost :
  ∀ (x : ℝ), 
    (12 * x - 2 * 0.05 * x = 476) → 
    x = 40 :=
by
  intros x h
  sorry

end concert_ticket_cost_l18_18415


namespace find_alpha_plus_beta_l18_18704

theorem find_alpha_plus_beta (α β : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : 0 < β ∧ β < π / 2)
  (h3 : Real.cos α = (Real.sqrt 5) / 5) (h4 : Real.sin β = (3 * Real.sqrt 10) / 10) : 
  α + β = 3 * π / 4 :=
sorry

end find_alpha_plus_beta_l18_18704


namespace classroom_gpa_l18_18412

theorem classroom_gpa (n : ℕ) (h1 : 1 ≤ n) : 
  (1/3 : ℝ) * 30 + (2/3 : ℝ) * 33 = 32 :=
by sorry

end classroom_gpa_l18_18412


namespace value_of_x_l18_18711

theorem value_of_x (x : ℝ) (h : (x / 5 / 3) = (5 / (x / 3))) : x = 15 ∨ x = -15 := 
by sorry

end value_of_x_l18_18711


namespace number_of_integer_solutions_l18_18314

theorem number_of_integer_solutions
    (a : ℤ)
    (x : ℤ)
    (h1 : ∃ x : ℤ, (1 - a) / (x - 2) + 2 = 1 / (2 - x))
    (h2 : ∀ x : ℤ, 4 * x ≥ 3 * (x - 1) ∧ x + (2 * x - 1) / 2 < (a - 1) / 2) :
    (a = 4) :=
sorry

end number_of_integer_solutions_l18_18314


namespace hcf_of_three_numbers_l18_18476

theorem hcf_of_three_numbers (a b c : ℕ) (h1 : a + b + c = 60)
  (h2 : Nat.lcm (Nat.lcm a b) c = 180)
  (h3 : (1:ℚ)/a + 1/b + 1/c = 11/120)
  (h4 : a * b * c = 900) :
  Nat.gcd (Nat.gcd a b) c = 5 :=
by
  sorry

end hcf_of_three_numbers_l18_18476


namespace maximum_black_squares_l18_18326

theorem maximum_black_squares (n : ℕ) (h : n ≥ 2) : 
  (n % 2 = 0 → ∃ b : ℕ, b = (n^2 - 4) / 2) ∧ 
  (n % 2 = 1 → ∃ b : ℕ, b = (n^2 - 1) / 2) := 
by sorry

end maximum_black_squares_l18_18326


namespace geometric_mean_condition_l18_18179

variable {a : ℕ → ℝ}
variable {b : ℕ → ℝ}

theorem geometric_mean_condition
  (h_arith : (a 1 + a 2 + a 3 + a 4 + a 5 + a 6) / 6 = (a 3 + a 4) / 2)
  (h_geom_pos : ∀ n, 0 < b n) :
  Real.sqrt (b 1 * b 2 * b 3 * b 4 * b 5 * b 6) = Real.sqrt (b 3 * b 4) :=
sorry

end geometric_mean_condition_l18_18179


namespace isosceles_right_triangle_inscribed_circle_l18_18109

theorem isosceles_right_triangle_inscribed_circle
  (h r x : ℝ)
  (h_def : h = 2 * r)
  (r_def : r = Real.sqrt 2 / 4)
  (x_def : x = h - r) :
  x = Real.sqrt 2 / 4 :=
by
  sorry

end isosceles_right_triangle_inscribed_circle_l18_18109


namespace min_value_of_f_l18_18123

noncomputable def f (x : ℝ) : ℝ := 7 * x^2 - 28 * x + 1425

theorem min_value_of_f : ∃ (x : ℝ), f x = 1397 :=
by
  sorry

end min_value_of_f_l18_18123


namespace average_salary_l18_18465

theorem average_salary
  (num_technicians : ℕ) (avg_salary_technicians : ℝ)
  (num_other_workers : ℕ) (avg_salary_other_workers : ℝ)
  (total_num_workers : ℕ) (avg_salary_all_workers : ℝ) :
  num_technicians = 7 →
  avg_salary_technicians = 14000 →
  num_other_workers = total_num_workers - num_technicians →
  avg_salary_other_workers = 6000 →
  total_num_workers = 28 →
  avg_salary_all_workers = (num_technicians * avg_salary_technicians + num_other_workers * avg_salary_other_workers) / total_num_workers →
  avg_salary_all_workers = 8000 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end average_salary_l18_18465


namespace cos_alpha_eq_l18_18830

open Real

-- Define the angles and their conditions
variables (α β : ℝ)

-- Hypothesis and initial conditions
axiom ha1 : 0 < α ∧ α < π
axiom ha2 : 0 < β ∧ β < π
axiom h_cos_beta : cos β = -5 / 13
axiom h_sin_alpha_plus_beta : sin (α + β) = 3 / 5

-- The main theorem to prove
theorem cos_alpha_eq : cos α = 56 / 65 := sorry

end cos_alpha_eq_l18_18830


namespace sqrt_meaningful_implies_x_ge_2_l18_18632

theorem sqrt_meaningful_implies_x_ge_2 (x : ℝ) (h : 0 ≤ x - 2) : x ≥ 2 := 
sorry

end sqrt_meaningful_implies_x_ge_2_l18_18632


namespace lindsey_exercise_bands_l18_18315

theorem lindsey_exercise_bands (x : ℕ) 
  (h1 : ∀ n, n = 5 * x) 
  (h2 : ∀ m, m = 10 * x) 
  (h3 : ∀ d, d = m + 10) 
  (h4 : d = 30) : 
  x = 2 := 
by 
  sorry

end lindsey_exercise_bands_l18_18315


namespace soccer_ball_seams_l18_18867

theorem soccer_ball_seams 
  (num_pentagons : ℕ) 
  (num_hexagons : ℕ) 
  (sides_per_pentagon : ℕ) 
  (sides_per_hexagon : ℕ) 
  (total_pieces : ℕ) 
  (equal_sides : sides_per_pentagon = sides_per_hexagon)
  (total_pieces_eq : total_pieces = 32)
  (num_pentagons_eq : num_pentagons = 12)
  (num_hexagons_eq : num_hexagons = 20)
  (sides_per_pentagon_eq : sides_per_pentagon = 5)
  (sides_per_hexagon_eq : sides_per_hexagon = 6) :
  90 = (num_pentagons * sides_per_pentagon + num_hexagons * sides_per_hexagon) / 2 :=
by 
  sorry

end soccer_ball_seams_l18_18867


namespace correct_value_of_A_sub_B_l18_18232

variable {x y : ℝ}

-- Given two polynomials A and B where B = 3x - 2y, and a mistaken equation A + B = x - y,
-- we want to prove the correct value of A - B.
theorem correct_value_of_A_sub_B (A B : ℝ) (h1 : B = 3 * x - 2 * y) (h2 : A + B = x - y) :
  A - B = -5 * x + 3 * y :=
by
  sorry

end correct_value_of_A_sub_B_l18_18232


namespace area_at_stage_7_l18_18062

-- Define the size of one square added at each stage
def square_size : ℕ := 4

-- Define the area of one square
def area_of_one_square : ℕ := square_size * square_size

-- Define the number of stages
def number_of_stages : ℕ := 7

-- Define the total area at a given stage
def total_area (n : ℕ) : ℕ := n * area_of_one_square

-- The theorem which proves the area of the rectangle at Stage 7
theorem area_at_stage_7 : total_area number_of_stages = 112 :=
by
  -- proof goes here
  sorry

end area_at_stage_7_l18_18062


namespace total_payment_is_correct_l18_18613

-- Define the number of friends
def number_of_friends : ℕ := 7

-- Define the amount each friend paid
def amount_per_friend : ℝ := 70.0

-- Define the total amount paid
def total_amount_paid : ℝ := number_of_friends * amount_per_friend

-- Prove that the total amount paid is 490.0
theorem total_payment_is_correct : total_amount_paid = 490.0 := by 
  -- Here, the proof would be filled in
  sorry

end total_payment_is_correct_l18_18613


namespace max_value_expr_l18_18556

theorem max_value_expr (x y : ℝ) : (2 * x + 3 * y + 4) / (Real.sqrt (x^4 + y^2 + 1)) ≤ Real.sqrt 29 := sorry

end max_value_expr_l18_18556


namespace solve_for_z_l18_18385

theorem solve_for_z (i z : ℂ) (h0 : i^2 = -1) (h1 : i / z = 1 + i) : z = (1 + i) / 2 :=
by
  sorry

end solve_for_z_l18_18385


namespace carol_initial_cupcakes_l18_18751

/--
For the school bake sale, Carol made some cupcakes. She sold 9 of them and then made 28 more.
Carol had 49 cupcakes. We need to show that Carol made 30 cupcakes initially.
-/
theorem carol_initial_cupcakes (x : ℕ) 
  (h1 : x - 9 + 28 = 49) : 
  x = 30 :=
by 
  -- The proof is not required as per instruction.
  sorry

end carol_initial_cupcakes_l18_18751


namespace solve_for_a_l18_18157

noncomputable def area_of_triangle (b c : ℝ) : ℝ :=
  1 / 2 * b * c * Real.sin (Real.pi / 3)

theorem solve_for_a (a b c : ℝ) (hA : 60 = 60) 
  (h_area : area_of_triangle b c = 3 * Real.sqrt 3 / 2)
  (h_sum_bc : b + c = 3 * Real.sqrt 3) :
  a = 3 :=
sorry

end solve_for_a_l18_18157


namespace find_c_l18_18551

theorem find_c (a : ℕ) (c : ℕ) 
  (h1 : a = 105) 
  (h2 : a ^ 5 = 3 ^ 3 * 5 ^ 2 * 7 ^ 2 * 11 ^ 2 * 13 * c) : 
  c = 385875 := by 
  sorry

end find_c_l18_18551


namespace interest_rate_per_annum_l18_18368

theorem interest_rate_per_annum :
  ∃ (r : ℝ), 338 = 312.50 * (1 + r) ^ 2 :=
by
  sorry

end interest_rate_per_annum_l18_18368


namespace coin_loading_impossible_l18_18566

theorem coin_loading_impossible (p q : ℝ) (h1 : p ≠ 1 - p) (h2 : q ≠ 1 - q) 
    (h3 : p * q = 1/4) (h4 : p * (1 - q) = 1/4) (h5 : (1 - p) * q = 1/4) (h6 : (1 - p) * (1 - q) = 1/4) : 
    false := 
by 
  sorry

end coin_loading_impossible_l18_18566


namespace intersection_points_l18_18206

theorem intersection_points (k : ℝ) : ∃ (P : ℝ × ℝ), P = (1, 0) ∧ ∀ x y : ℝ, (kx - y - k = 0) → (x^2 + y^2 = 2) → ∃ y1 y2 : ℝ, (y = y1 ∨ y = y2) :=
by
  sorry

end intersection_points_l18_18206


namespace vertical_distance_l18_18818

variable (storiesPerTrip tripsPerDay daysPerWeek feetPerStory : ℕ)

def totalVerticalDistance
  (storiesPerTrip tripsPerDay daysPerWeek feetPerStory : ℕ) : ℕ :=
  2 * storiesPerTrip * feetPerStory * tripsPerDay * daysPerWeek

theorem vertical_distance (h1 : storiesPerTrip = 5)
                          (h2 : tripsPerDay = 3)
                          (h3 : daysPerWeek = 7)
                          (h4 : feetPerStory = 10) :
  totalVerticalDistance storiesPerTrip tripsPerDay daysPerWeek feetPerStory = 2100 := by
  sorry

end vertical_distance_l18_18818


namespace calculate_product_l18_18540

noncomputable def complex_number_r (r : ℂ) : Prop :=
r^6 = 1 ∧ r ≠ 1

theorem calculate_product (r : ℂ) (h₁ : r^6 = 1) (h₂ : r ≠ 1) :
  (r - 1) * (r^2 - 1) * (r^3 - 1) * (r^4 - 1) * (r^5 - 1) = 2 := 
sorry

end calculate_product_l18_18540


namespace sequence_existence_l18_18799

theorem sequence_existence (n : ℕ) : 
  (∃ (x : ℕ → ℤ), 
    (∀ i j, 1 ≤ i ∧ i ≤ j ∧ j ≤ n ∧ i + j ≤ n ∧ ((x i - x j) % 3 = 0) → (x (i + j) + x i + x j + 1) % 3 = 0)) ↔ (n = 8) := 
by 
  sorry

end sequence_existence_l18_18799


namespace geometric_progression_solution_l18_18969

theorem geometric_progression_solution 
  (b1 q : ℝ)
  (condition1 : (b1^2 / (1 + q + q^2) = 48 / 7))
  (condition2 : (b1^2 / (1 + q^2) = 144 / 17)) 
  : (b1 = 3 ∨ b1 = -3) ∧ q = 1 / 4 :=
by
  sorry

end geometric_progression_solution_l18_18969


namespace cookie_difference_l18_18270

def AlyssaCookies : ℕ := 129
def AiyannaCookies : ℕ := 140
def Difference : ℕ := 11

theorem cookie_difference : AiyannaCookies - AlyssaCookies = Difference := by
  sorry

end cookie_difference_l18_18270


namespace minimum_area_of_triangle_l18_18948

def parabola_focus : Prop :=
  ∃ F : ℝ × ℝ, F = (1, 0)

def on_parabola (A B : ℝ × ℝ) : Prop :=
  (A.2 ^ 2 = 4 * A.1 ∧ B.2 ^ 2 = 4 * B.1) ∧ (A.2 * B.2 < 0)

def dot_product_condition (A B : ℝ × ℝ) : Prop :=
  A.1 * B.1 + A.2 * B.2 = -4

noncomputable def area (A B : ℝ × ℝ) : ℝ :=
  1 / 2 * abs (A.1 * B.2 - B.1 * A.2)

theorem minimum_area_of_triangle
  (A B : ℝ × ℝ)
  (h_focus : parabola_focus)
  (h_on_parabola : on_parabola A B)
  (h_dot : dot_product_condition A B) :
  ∃ C : ℝ, C = 4 * Real.sqrt 2 ∧ area A B = C :=
by
  sorry

end minimum_area_of_triangle_l18_18948


namespace regression_lines_intersect_at_average_l18_18313

theorem regression_lines_intersect_at_average
  {x_vals1 x_vals2 : List ℝ} {y_vals1 y_vals2 : List ℝ}
  (n1 : x_vals1.length = 100) (n2 : x_vals2.length = 150)
  (mean_x1 : (List.sum x_vals1 / 100) = s) (mean_x2 : (List.sum x_vals2 / 150) = s)
  (mean_y1 : (List.sum y_vals1 / 100) = t) (mean_y2 : (List.sum y_vals2 / 150) = t)
  (regression_line1 : ℝ → ℝ)
  (regression_line2 : ℝ → ℝ)
  (on_line1 : ∀ x, regression_line1 x = (a1 * x + b1))
  (on_line2 : ∀ x, regression_line2 x = (a2 * x + b2))
  (sample_center1 : regression_line1 s = t)
  (sample_center2 : regression_line2 s = t) :
  regression_line1 s = regression_line2 s := sorry

end regression_lines_intersect_at_average_l18_18313


namespace max_four_by_one_in_six_by_six_grid_l18_18582

-- Define the grid and rectangle dimensions
def grid_width : ℕ := 6
def grid_height : ℕ := 6
def rect_width : ℕ := 4
def rect_height : ℕ := 1

-- Define the maximum number of rectangles that can be placed
def max_rectangles (grid_w grid_h rect_w rect_h : ℕ) (non_overlapping : Bool) (within_boundaries : Bool) : ℕ :=
  if grid_w = 6 ∧ grid_h = 6 ∧ rect_w = 4 ∧ rect_h = 1 ∧ non_overlapping ∧ within_boundaries then
    8
  else
    0

-- The theorem stating the maximum number of 4x1 rectangles in a 6x6 grid
theorem max_four_by_one_in_six_by_six_grid
  : max_rectangles grid_width grid_height rect_width rect_height true true = 8 := 
sorry

end max_four_by_one_in_six_by_six_grid_l18_18582


namespace quadratic_has_two_distinct_real_roots_l18_18238

theorem quadratic_has_two_distinct_real_roots (a : ℝ) (h : a ≠ 0): 
  (a < 4 / 3) ↔ (∃ x y : ℝ, x ≠ y ∧  a * x^2 - 4 * x + 3 = 0 ∧ a * y^2 - 4 * y + 3 = 0) := 
sorry

end quadratic_has_two_distinct_real_roots_l18_18238


namespace wall_with_5_peaks_has_14_cubes_wall_with_2014_peaks_has_6041_cubes_painted_area_wall_with_2014_peaks_l18_18749

noncomputable def number_of_cubes (n : ℕ) : ℕ :=
  n + (n - 1) + n

noncomputable def painted_area (n : ℕ) : ℕ :=
  (5 * n) + (3 * (n + 1)) + (2 * (n - 2))

theorem wall_with_5_peaks_has_14_cubes : number_of_cubes 5 = 14 :=
  by sorry

theorem wall_with_2014_peaks_has_6041_cubes : number_of_cubes 2014 = 6041 :=
  by sorry

theorem painted_area_wall_with_2014_peaks : painted_area 2014 = 20139 :=
  by sorry

end wall_with_5_peaks_has_14_cubes_wall_with_2014_peaks_has_6041_cubes_painted_area_wall_with_2014_peaks_l18_18749


namespace equal_partition_of_weights_l18_18405

theorem equal_partition_of_weights 
  (weights : Fin 2009 → ℕ) 
  (h1 : ∀ i : Fin 2008, (weights i + 1 = weights (i + 1)) ∨ (weights i = weights (i + 1) + 1))
  (h2 : ∀ i : Fin 2009, weights i ≤ 1000)
  (h3 : (Finset.univ.sum weights) % 2 = 0) :
  ∃ (A B : Finset (Fin 2009)), (A ∪ B = Finset.univ ∧ A ∩ B = ∅ ∧ A.sum weights = B.sum weights) :=
sorry

end equal_partition_of_weights_l18_18405


namespace solve_quad_1_solve_quad_2_l18_18563

theorem solve_quad_1 :
  ∀ (x : ℝ), x^2 - 5 * x - 6 = 0 ↔ x = 6 ∨ x = -1 := by
  sorry

theorem solve_quad_2 :
  ∀ (x : ℝ), (x + 1) * (x - 1) + x * (x + 2) = 7 + 6 * x ↔ x = 1 + Real.sqrt 5 ∨ x = 1 - Real.sqrt 5 := by
  sorry

end solve_quad_1_solve_quad_2_l18_18563


namespace mary_max_earnings_l18_18091

def regular_rate : ℝ := 8
def max_hours : ℝ := 60
def regular_hours : ℝ := 20
def overtime_rate : ℝ := regular_rate + 0.25 * regular_rate
def overtime_hours : ℝ := max_hours - regular_hours
def earnings_regular : ℝ := regular_hours * regular_rate
def earnings_overtime : ℝ := overtime_hours * overtime_rate
def total_earnings : ℝ := earnings_regular + earnings_overtime

theorem mary_max_earnings : total_earnings = 560 := by
  sorry

end mary_max_earnings_l18_18091


namespace inequality_and_equality_l18_18844

variables {x y z : ℝ}

theorem inequality_and_equality (x y z : ℝ) :
  (x^2 + y^4 + z^6 >= x * y^2 + y^2 * z^3 + x * z^3) ∧ (x^2 + y^4 + z^6 = x * y^2 + y^2 * z^3 + x * z^3 ↔ x = y^2 ∧ y^2 = z^3) :=
by sorry

end inequality_and_equality_l18_18844


namespace surface_area_of_interior_box_l18_18979

def original_sheet_width : ℕ := 40
def original_sheet_length : ℕ := 50
def corner_cut_side : ℕ := 8
def corners_count : ℕ := 4

def area_of_original_sheet : ℕ := original_sheet_width * original_sheet_length
def area_of_one_corner_cut : ℕ := corner_cut_side * corner_cut_side
def total_area_removed : ℕ := corners_count * area_of_one_corner_cut
def area_of_remaining_sheet : ℕ := area_of_original_sheet - total_area_removed

theorem surface_area_of_interior_box : area_of_remaining_sheet = 1744 :=
by
  sorry

end surface_area_of_interior_box_l18_18979


namespace min_value_frac_sq_l18_18942

theorem min_value_frac_sq (x : ℝ) (h : x > 12) : (x^2 / (x - 12)) >= 48 :=
by
  sorry

end min_value_frac_sq_l18_18942


namespace average_study_diff_l18_18067

theorem average_study_diff (diff : List ℤ) (h_diff : diff = [15, -5, 25, -10, 5, 20, -15]) :
  (List.sum diff) / (List.length diff) = 5 := by
  sorry

end average_study_diff_l18_18067


namespace functional_equation_solution_l18_18042

theorem functional_equation_solution :
  ∀ (f : ℝ → ℝ), (∀ x y : ℝ, (f x + y) * (f (x - y) + 1) = f (f (x * f (x + 1)) - y * f (y - 1))) → (∀ x : ℝ, f x = x) :=
by
  intros f h x
  -- Proof would go here
  sorry

end functional_equation_solution_l18_18042


namespace smallest_positive_multiple_l18_18759

theorem smallest_positive_multiple (a : ℕ) (h₁ : a % 6 = 0) (h₂ : a % 15 = 0) : a = 30 :=
sorry

end smallest_positive_multiple_l18_18759


namespace check_blank_value_l18_18483

/-- Define required constants and terms. -/
def six_point_five : ℚ := 6 + 1/2
def two_thirds : ℚ := 2/3
def three_point_five : ℚ := 3 + 1/2
def one_and_eight_fifteenths : ℚ := 1 + 8/15
def blank : ℚ := 3 + 1/20
def seventy_one_point_ninety_five : ℚ := 71 + 95/100

/-- The translated assumption and statement to be proved: -/
theorem check_blank_value :
  (six_point_five - two_thirds) / three_point_five - one_and_eight_fifteenths * (blank + seventy_one_point_ninety_five) = 1 :=
sorry

end check_blank_value_l18_18483


namespace find_triangle_base_l18_18289

theorem find_triangle_base (left_side : ℝ) (right_side : ℝ) (base : ℝ) 
  (h_left : left_side = 12) 
  (h_right : right_side = left_side + 2)
  (h_sum : left_side + right_side + base = 50) :
  base = 24 := 
sorry

end find_triangle_base_l18_18289


namespace geometric_series_sum_l18_18107

def sum_geometric_series (a r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_series_sum :
  sum_geometric_series (1/4) (1/4) 7 = 4/3 :=
by
  -- Proof is omitted
  sorry

end geometric_series_sum_l18_18107


namespace line_equation_l18_18663

theorem line_equation (x y : ℝ) (hx : ∃ t : ℝ, t ≠ 0 ∧ x = t * -3) (hy : ∃ t : ℝ, t ≠ 0 ∧ y = t * 4) :
  4 * x - 3 * y + 12 = 0 := 
sorry

end line_equation_l18_18663


namespace value_of_fraction_zero_l18_18505

theorem value_of_fraction_zero (x : ℝ) (h1 : x^2 - 1 = 0) (h2 : 1 - x ≠ 0) : x = -1 :=
by
  sorry

end value_of_fraction_zero_l18_18505


namespace adam_cat_food_packages_l18_18559

theorem adam_cat_food_packages (c : ℕ) 
  (dog_food_packages : ℕ := 7) 
  (cans_per_cat_package : ℕ := 10) 
  (cans_per_dog_package : ℕ := 5) 
  (extra_cat_food_cans : ℕ := 55) 
  (total_dog_cans : ℕ := dog_food_packages * cans_per_dog_package) 
  (total_cat_cans : ℕ := c * cans_per_cat_package)
  (h : total_cat_cans = total_dog_cans + extra_cat_food_cans) : 
  c = 9 :=
by
  sorry

end adam_cat_food_packages_l18_18559


namespace positive_integer_solution_l18_18917

theorem positive_integer_solution (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (h : x^4 = y^2 + 71) :
  x = 6 ∧ y = 35 :=
by
  sorry

end positive_integer_solution_l18_18917


namespace original_average_l18_18231

theorem original_average (n : ℕ) (A : ℝ) (new_avg : ℝ) 
  (h1 : n = 25) 
  (h2 : new_avg = 140) 
  (h3 : 2 * A = new_avg) : A = 70 :=
sorry

end original_average_l18_18231


namespace smallest_x_multiple_of_53_l18_18681

theorem smallest_x_multiple_of_53 :
  ∃ x : ℕ, (3 * x + 41) % 53 = 0 ∧ x > 0 ∧ x = 4 :=
by 
  sorry

end smallest_x_multiple_of_53_l18_18681


namespace son_completion_time_l18_18756

theorem son_completion_time (M S F : ℝ) 
  (h1 : M = 1 / 10) 
  (h2 : M + S = 1 / 5) 
  (h3 : S + F = 1 / 4) : 
  1 / S = 10 := 
  sorry

end son_completion_time_l18_18756


namespace sum_of_roots_of_P_is_8029_l18_18358

-- Define the polynomial
noncomputable def P : Polynomial ℚ :=
  (Polynomial.X - 1)^2008 + 
  3 * (Polynomial.X - 2)^2007 + 
  5 * (Polynomial.X - 3)^2006 + 
  -- Continue defining all terms up to:
  2009 * (Polynomial.X - 2008)^2 + 
  2011 * (Polynomial.X - 2009)

-- The proof problem statement
theorem sum_of_roots_of_P_is_8029 :
  (P.roots.sum = 8029) :=
sorry

end sum_of_roots_of_P_is_8029_l18_18358


namespace sum_of_intercepts_l18_18569

theorem sum_of_intercepts (x y : ℝ) (hx : y + 3 = 5 * (x - 6)) : 
  let x_intercept := 6 + 3/5;
  let y_intercept := -33;
  x_intercept + y_intercept = -26.4 := by
  sorry

end sum_of_intercepts_l18_18569


namespace stationery_shop_costs_l18_18114

theorem stationery_shop_costs (p n : ℝ) 
  (h1 : 9 * p + 6 * n = 3.21)
  (h2 : 8 * p + 5 * n = 2.84) :
  12 * p + 9 * n = 4.32 :=
sorry

end stationery_shop_costs_l18_18114


namespace distance_between_stripes_l18_18438

/-- Given a crosswalk parallelogram with curbs 60 feet apart, a base of 20 feet, 
and each stripe of length 50 feet, show that the distance between the stripes is 24 feet. -/
theorem distance_between_stripes (h : Real) (b : Real) (s : Real) : h = 60 ∧ b = 20 ∧ s = 50 → (b * h) / s = 24 :=
by
  sorry

end distance_between_stripes_l18_18438


namespace sum_eq_prod_nat_numbers_l18_18637

theorem sum_eq_prod_nat_numbers (A B C D E F : ℕ) :
  A + B + C + D + E + F = A * B * C * D * E * F →
  (A = 0 ∧ B = 0 ∧ C = 0 ∧ D = 0 ∧ E = 0 ∧ F = 0) ∨
  (A = 1 ∧ B = 1 ∧ C = 1 ∧ D = 1 ∧ E = 2 ∧ F = 6) :=
by
  sorry

end sum_eq_prod_nat_numbers_l18_18637


namespace correct_word_to_complete_sentence_l18_18618

theorem correct_word_to_complete_sentence
  (parents_spoke_language : Bool)
  (learning_difficulty : String) :
  learning_difficulty = "It was hard for him to learn English in a family, in which neither of the parents spoke the language." :=
by
  sorry

end correct_word_to_complete_sentence_l18_18618


namespace percent_enclosed_by_hexagons_l18_18723

variable (b : ℝ) -- side length of smaller squares

def area_of_small_square : ℝ := b^2
def area_of_large_square : ℝ := 16 * area_of_small_square b
def area_of_hexagon : ℝ := 3 * area_of_small_square b
def total_area_of_hexagons : ℝ := 2 * area_of_hexagon b

theorem percent_enclosed_by_hexagons :
  (total_area_of_hexagons b / area_of_large_square b) * 100 = 37.5 :=
by
  -- Proof omitted
  sorry

end percent_enclosed_by_hexagons_l18_18723


namespace calculation_a_squared_plus_b_squared_calculation_a_minus_b_squared_l18_18847

theorem calculation_a_squared_plus_b_squared
  (a b : ℝ)
  (h1 : a + b = 2)
  (h2 : a * b = -1) :
  a^2 + b^2 = 6 := by
  sorry

theorem calculation_a_minus_b_squared
  (a b : ℝ)
  (h1 : a + b = 2)
  (h2 : a * b = -1) :
  (a - b)^2 = 8 := by
  sorry

end calculation_a_squared_plus_b_squared_calculation_a_minus_b_squared_l18_18847


namespace original_integer_is_26_l18_18995

theorem original_integer_is_26 (x y z w : ℕ) (h₁ : 0 < x) (h₂ : 0 < y) (h₃ : 0 < z) (h₄ : 0 < w)
(h₅ : x ≠ y) (h₆ : x ≠ z) (h₇ : x ≠ w) (h₈ : y ≠ z) (h₉ : y ≠ w) (h₁₀ : z ≠ w)
(h₁₁ : (x + y + z) / 3 + w = 34)
(h₁₂ : (x + y + w) / 3 + z = 22)
(h₁₃ : (x + z + w) / 3 + y = 26)
(h₁₄ : (y + z + w) / 3 + x = 18) :
    w = 26 := 
sorry

end original_integer_is_26_l18_18995


namespace sam_runs_more_than_sarah_sue_runs_less_than_sarah_l18_18962

-- Definitions based on the problem conditions
def street_width : ℝ := 25
def block_side_length : ℝ := 500
def sarah_perimeter : ℝ := 4 * block_side_length
def sam_perimeter : ℝ := 4 * (block_side_length + 2 * street_width)
def sue_perimeter : ℝ := 4 * (block_side_length - 2 * street_width)

-- The proof problem statements
theorem sam_runs_more_than_sarah : sam_perimeter - sarah_perimeter = 200 := by
  sorry

theorem sue_runs_less_than_sarah : sarah_perimeter - sue_perimeter = 200 := by
  sorry

end sam_runs_more_than_sarah_sue_runs_less_than_sarah_l18_18962


namespace probability_excluded_probability_selected_l18_18953

-- Define the population size and the sample size
def population_size : ℕ := 1005
def sample_size : ℕ := 50
def excluded_count : ℕ := 5

-- Use these values within the theorems
theorem probability_excluded : (excluded_count : ℚ) / (population_size : ℚ) = 5 / 1005 :=
by sorry

theorem probability_selected : (sample_size : ℚ) / (population_size : ℚ) = 50 / 1005 :=
by sorry

end probability_excluded_probability_selected_l18_18953


namespace minimum_value_f_l18_18626

noncomputable def f (x : ℝ) : ℝ := max (3 - x) (x^2 - 4 * x + 3)

theorem minimum_value_f : ∃ m : ℝ, ∀ x : ℝ, f x ≥ m ∧ (∀ ε > 0, ∃ x : ℝ, x ≥ 0 ∧ f x < m + ε) ∧ m = 0 := 
sorry

end minimum_value_f_l18_18626


namespace complex_identity_l18_18914

variable (i : ℂ)
axiom i_squared : i^2 = -1

theorem complex_identity : 1 + i + i^2 = i :=
by sorry

end complex_identity_l18_18914


namespace ratio_buses_to_cars_l18_18471

theorem ratio_buses_to_cars (B C : ℕ) (h1 : B = C - 60) (h2 : C = 65) : B / C = 1 / 13 :=
by 
  sorry

end ratio_buses_to_cars_l18_18471


namespace probability_both_selected_l18_18615

def P_X : ℚ := 1 / 3
def P_Y : ℚ := 2 / 7

theorem probability_both_selected : P_X * P_Y = 2 / 21 :=
by
  sorry

end probability_both_selected_l18_18615


namespace value_of_a_l18_18785

def P : Set ℝ := { x | x^2 ≤ 4 }
def M (a : ℝ) : Set ℝ := { a }

theorem value_of_a (a : ℝ) (h : P ∪ {a} = P) : a ∈ { x : ℝ | -2 ≤ x ∧ x ≤ 2 } := by
  sorry

end value_of_a_l18_18785


namespace maximum_omega_l18_18880

noncomputable def f (omega varphi : ℝ) (x : ℝ) : ℝ :=
  Real.cos (omega * x + varphi)

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = -f (-x)

def is_monotonically_decreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y : ℝ, a < x → x < y → y < b → f y ≤ f x

theorem maximum_omega (omega varphi : ℝ)
    (h0 : omega > 0)
    (h1 : 0 < varphi ∧ varphi < π)
    (h2 : is_odd_function (f omega varphi))
    (h3 : is_monotonically_decreasing (f omega varphi) (-π/3) (π/6)) :
  omega ≤ 3/2 :=
sorry

end maximum_omega_l18_18880


namespace ab_cd_zero_l18_18881

theorem ab_cd_zero {a b c d : ℝ} (h1 : a^2 + b^2 = 1) (h2 : c^2 + d^2 = 1) (h3 : ac + bd = 0) : ab + cd = 0 :=
sorry

end ab_cd_zero_l18_18881


namespace least_number_to_make_divisible_by_3_l18_18676

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem least_number_to_make_divisible_by_3 : ∃ k : ℕ, (∃ n : ℕ, 
  sum_of_digits 625573 ≡ 28 [MOD 3] ∧ 
  (625573 + k) % 3 = 0 ∧ 
  k = 2) :=
by
  sorry

end least_number_to_make_divisible_by_3_l18_18676


namespace inequality_solution_l18_18886

theorem inequality_solution (x : ℝ) : 
  (2 * x) / (3 * x - 1) > 1 ↔ (1 / 3 < x ∧ x < 1) :=
sorry

end inequality_solution_l18_18886


namespace coefficients_balance_l18_18780

noncomputable def num_positive_coeffs (n : ℕ) : ℕ :=
  n + 1

noncomputable def num_negative_coeffs (n : ℕ) : ℕ :=
  n + 1

theorem coefficients_balance (n : ℕ) (h_odd: Odd n) (x : ℝ) :
  num_positive_coeffs n = num_negative_coeffs n :=
by
  sorry

end coefficients_balance_l18_18780


namespace problem_statement_l18_18640

theorem problem_statement (a x m : ℝ) (h₀ : |a| ≤ 1) (h₁ : |x| ≤ 1) :
  (∀ x a, |x^2 - a * x - a^2| ≤ m) ↔ m ≥ 5/4 :=
sorry

end problem_statement_l18_18640


namespace pow_neg_cubed_squared_l18_18101

variable (a : ℝ)

theorem pow_neg_cubed_squared : 
  (-a^3)^2 = a^6 := 
by 
  sorry

end pow_neg_cubed_squared_l18_18101


namespace roots_quadratic_l18_18451

theorem roots_quadratic (a b c d : ℝ) :
  (a + b = 3 * c / 2 ∧ a * b = 4 * d ∧ c + d = 3 * a / 2 ∧ c * d = 4 * b)
  ↔ ( (a = 4 ∧ b = 8 ∧ c = 4 ∧ d = 8) ∨
      (a = -2 ∧ b = -22 ∧ c = -8 ∧ d = 11) ∨
      (a = -8 ∧ b = 2 ∧ c = -2 ∧ d = -4) ) :=
by
  sorry

end roots_quadratic_l18_18451


namespace sum_of_variables_l18_18547

variables (a b c d : ℝ)

theorem sum_of_variables :
  (a - 2)^2 + (b - 5)^2 + (c - 6)^2 + (d - 3)^2 = 0 → a + b + c + d = 16 :=
by
  intro h
  -- your proof goes here
  sorry

end sum_of_variables_l18_18547


namespace lucky_ticket_N123456_l18_18918

def digits : List ℕ := [1, 2, 3, 4, 5, 6]

def is_lucky (digits : List ℕ) : Prop :=
  ∃ f : ℕ → ℕ → ℕ, (f 1 (f (f 2 3) 4) * f 5 6) = 100

theorem lucky_ticket_N123456 : is_lucky digits :=
  sorry

end lucky_ticket_N123456_l18_18918


namespace correct_assignment_statement_l18_18876

noncomputable def is_assignment_statement (stmt : String) : Bool :=
  -- Assume a simplified function that interprets whether the statement is an assignment
  match stmt with
  | "6 = M" => false
  | "M = -M" => true
  | "B = A = 8" => false
  | "x - y = 0" => false
  | _ => false

theorem correct_assignment_statement :
  is_assignment_statement "M = -M" = true :=
by
  rw [is_assignment_statement]
  exact rfl

end correct_assignment_statement_l18_18876


namespace work_days_B_l18_18464

theorem work_days_B (A_days B_days : ℕ) (hA : A_days = 12) (hTogether : (1/12 + 1/A_days) = (1/8)) : B_days = 24 := 
by
  revert hTogether -- reversing to tackle proof
  sorry

end work_days_B_l18_18464


namespace sum_of_powers_of_two_l18_18535

theorem sum_of_powers_of_two : 2^4 + 2^4 + 2^4 + 2^4 = 2^6 :=
by
  sorry

end sum_of_powers_of_two_l18_18535


namespace red_balls_in_total_color_of_158th_ball_l18_18161

def totalBalls : Nat := 200
def redBallsPerCycle : Nat := 5
def whiteBallsPerCycle : Nat := 4
def blackBallsPerCycle : Nat := 3
def cycleLength : Nat := redBallsPerCycle + whiteBallsPerCycle + blackBallsPerCycle

theorem red_balls_in_total :
  (totalBalls / cycleLength) * redBallsPerCycle + min redBallsPerCycle (totalBalls % cycleLength) = 85 :=
by sorry

theorem color_of_158th_ball :
  let positionInCycle := (158 - 1) % cycleLength + 1
  positionInCycle ≤ redBallsPerCycle := by sorry

end red_balls_in_total_color_of_158th_ball_l18_18161


namespace intersection_sets_l18_18434

noncomputable def set1 (x : ℝ) : Prop := (x - 2) / (x + 1) ≤ 0
noncomputable def set2 (x : ℝ) : Prop := x^2 - 2 * x - 3 < 0

theorem intersection_sets :
  { x : ℝ | set1 x } ∩ { x : ℝ | set2 x } = { x | (-1 : ℝ) < x ∧ x ≤ 2 } :=
by
  sorry

end intersection_sets_l18_18434


namespace XiaoMaHu_correct_calculation_l18_18971

theorem XiaoMaHu_correct_calculation :
  (∃ A B C D : Prop, (A = ((a b : ℝ) → (a - b)^2 = a^2 - b^2)) ∧ 
                   (B = ((a : ℝ) → (-2 * a^3)^2 = 4 * a^6)) ∧ 
                   (C = ((a : ℝ) → a^3 + a^2 = 2 * a^5)) ∧ 
                   (D = ((a : ℝ) → -(a - 1) = -a - 1)) ∧ 
                   (¬A ∧ B ∧ ¬C ∧ ¬D)) :=
sorry

end XiaoMaHu_correct_calculation_l18_18971


namespace hcf_of_two_numbers_l18_18276

noncomputable def number1 : ℕ := 414

noncomputable def lcm_factors : Set ℕ := {13, 18}

noncomputable def hcf (a b : ℕ) : ℕ := Nat.gcd a b

-- Statement to prove
theorem hcf_of_two_numbers (Y : ℕ) 
  (H : ℕ) 
  (lcm : ℕ) 
  (H_lcm_factors : lcm = H * 13 * 18)
  (H_lcm_prop : lcm = (number1 * Y) / H)
  (H_Y : Y = (H^2 * 13 * 18) / 414)
  : H = 23 := 
sorry

end hcf_of_two_numbers_l18_18276


namespace chocolate_milk_container_size_l18_18297

/-- Holly's chocolate milk consumption conditions and container size -/
theorem chocolate_milk_container_size
  (morning_initial: ℝ)  -- Initial amount in the morning
  (morning_drink: ℝ)    -- Amount drank in the morning with breakfast
  (lunch_drink: ℝ)      -- Amount drank at lunch
  (dinner_drink: ℝ)     -- Amount drank with dinner
  (end_of_day: ℝ)       -- Amount she ends the day with
  (lunch_container_size: ℝ) -- Size of the container bought at lunch
  (C: ℝ)                -- Container size she bought at lunch
  (h_initial: morning_initial = 16)
  (h_morning_drink: morning_drink = 8)
  (h_lunch_drink: lunch_drink = 8)
  (h_dinner_drink: dinner_drink = 8)
  (h_end_of_day: end_of_day = 56) :
  (morning_initial - morning_drink) + C - lunch_drink - dinner_drink = end_of_day → 
  lunch_container_size = 64 :=
by
  sorry

end chocolate_milk_container_size_l18_18297


namespace parallel_edges_octahedron_l18_18407

-- Definition of a regular octahedron's properties
structure regular_octahedron : Type :=
  (edges : ℕ) -- Number of edges in the octahedron

-- Constant to represent the regular octahedron with 12 edges.
def octahedron : regular_octahedron := { edges := 12 }

-- Definition to count unique pairs of parallel edges
def count_parallel_edge_pairs (o : regular_octahedron) : ℕ :=
  if o.edges = 12 then 12 else 0

-- Theorem to assert the number of pairs of parallel edges in a regular octahedron is 12
theorem parallel_edges_octahedron : count_parallel_edge_pairs octahedron = 12 :=
by
  -- Proof will be inserted here
  sorry

end parallel_edges_octahedron_l18_18407


namespace find_y_l18_18621

theorem find_y (y : ℕ) : (8000 * 6000 = 480 * 10 ^ y) → y = 5 :=
by
  intro h
  sorry

end find_y_l18_18621


namespace find_other_endpoint_l18_18539

theorem find_other_endpoint :
  ∀ (A B M : ℝ × ℝ),
  M = (2, 3) →
  A = (7, -4) →
  M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) →
  B = (-3, 10) :=
by
  intros A B M hM1 hA hM2
  sorry

end find_other_endpoint_l18_18539


namespace subset_proof_l18_18024

-- Define set M
def M : Set ℝ := {y : ℝ | ∃ x : ℝ, y = x^2}

-- Define set N
def N : Set ℝ := {y : ℝ | ∃ x : ℝ, y = Real.sqrt (x^2 + 2*x + 1)}

-- The problem statement
theorem subset_proof : M ⊆ N ∧ ∃ y ∈ N, y ∉ M :=
by
  sorry

end subset_proof_l18_18024


namespace remaining_yards_is_720_l18_18466

-- Definitions based on conditions:
def marathon_miles : Nat := 25
def marathon_yards : Nat := 500
def yards_in_mile : Nat := 1760
def num_of_marathons : Nat := 12

-- Total distance for one marathon in yards
def one_marathon_total_yards : Nat :=
  marathon_miles * yards_in_mile + marathon_yards

-- Total distance for twelve marathons in yards
def total_distance_yards : Nat :=
  num_of_marathons * one_marathon_total_yards

-- Remaining yards after converting the total distance into miles and yards
def y : Nat :=
  total_distance_yards % yards_in_mile

-- Condition ensuring y is the remaining yards and is within the bounds 0 ≤ y < 1760
theorem remaining_yards_is_720 : 
  y = 720 := sorry

end remaining_yards_is_720_l18_18466


namespace inequality_problem_l18_18515

open Real

theorem inequality_problem 
  (x y : ℝ) 
  (hx : 0 < x) 
  (hy : 0 < y) 
  (h1 : x + y^2016 ≥ 1) : 
  x^2016 + y > 1 - 1/100 :=
by
  sorry

end inequality_problem_l18_18515


namespace contains_zero_l18_18557

theorem contains_zero (a b c d e f: ℕ) (h1: 1 ≤ a ∧ a ≤ 9) (h2: 1 ≤ b ∧ b ≤ 9) (h3: 0 ≤ c ∧ c ≤ 9) 
  (h4: 1 ≤ d ∧ d ≤ 9) (h5: 0 ≤ e ∧ e ≤ 9) (h6: 0 ≤ f ∧ f ≤ 9) 
  (h7: c ≠ f) (h8: 10^4*a + 10^3*b + 10^2*c + 10^1*d + e + 10^4*a + 10^3*b + 10^2*f + 10^1*d + e = 111111) :
  c = 0 ∨ f = 0 := 
sorry

end contains_zero_l18_18557


namespace cos_alpha_value_l18_18863

theorem cos_alpha_value
  (a : ℝ) (h1 : π < a ∧ a < 3 * π / 2)
  (h2 : Real.tan a = 2) :
  Real.cos a = - (Real.sqrt 5) / 5 :=
sorry

end cos_alpha_value_l18_18863


namespace smallest_number_of_blocks_needed_l18_18200

/--
Given:
  A wall with the following properties:
  1. The wall is 100 feet long and 7 feet high.
  2. Blocks used are 1 foot high and either 1 foot or 2 feet long.
  3. Blocks cannot be cut.
  4. Vertical joins in the blocks must be staggered.
  5. The wall must be even on the ends.
Prove:
  The smallest number of blocks needed to build this wall is 353.
-/
theorem smallest_number_of_blocks_needed :
  let length := 100
  let height := 7
  let block_height := 1
  (∀ b : ℕ, b = 1 ∨ b = 2) →
  ∃ (blocks_needed : ℕ), blocks_needed = 353 :=
by sorry

end smallest_number_of_blocks_needed_l18_18200


namespace arithmetic_sequence_suff_nec_straight_line_l18_18919

variable (n : ℕ) (P_n : ℕ → ℝ)

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ m : ℕ, a (m + 1) = a m + d

def lies_on_straight_line (P : ℕ → ℝ) : Prop :=
  ∃ m b, ∀ n, P n = m * n + b

theorem arithmetic_sequence_suff_nec_straight_line
  (h_n : 0 < n)
  (h_arith : arithmetic_sequence P_n) :
  lies_on_straight_line P_n ↔ arithmetic_sequence P_n :=
sorry

end arithmetic_sequence_suff_nec_straight_line_l18_18919


namespace complement_union_eq_l18_18115

universe u

-- Definitions based on conditions in a)
def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {1, 4}
def N : Set ℕ := {2, 3}

-- The goal to prove based on c)
theorem complement_union_eq :
  (U \ (M ∪ N)) = {5, 6} := 
by sorry

end complement_union_eq_l18_18115


namespace digits_divisibility_property_l18_18684

-- Definition: Example function to sum the digits of a number
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.foldl (· + ·) 0

-- Theorem: Prove the correctness of the given mathematical problem
theorem digits_divisibility_property:
  ∀ n : ℕ, (n = 18 ∨ n = 27 ∨ n = 45 ∨ n = 63) →
  (sum_of_digits n % 9 = 0) → (n % 9 = 0) := by
  sorry

end digits_divisibility_property_l18_18684


namespace points_on_parabola_l18_18910

theorem points_on_parabola (t : ℝ) : 
  ∃ a b c : ℝ, ∀ (x y: ℝ), (x, y) = (Real.cos t ^ 2, Real.sin (2 * t)) → y^2 = 4 * x - 4 * x^2 := 
by
  sorry

end points_on_parabola_l18_18910


namespace maximum_value_inequality_l18_18147

theorem maximum_value_inequality (x y : ℝ) : 
  (3 * x + 4 * y + 5) / Real.sqrt (x^2 + y^2 + 4) ≤ Real.sqrt 50 :=
sorry

end maximum_value_inequality_l18_18147


namespace huangs_tax_is_65_yuan_l18_18239

noncomputable def monthly_salary : ℝ := 2900
noncomputable def tax_free_portion : ℝ := 2000
noncomputable def tax_rate_5_percent : ℝ := 0.05
noncomputable def tax_rate_10_percent : ℝ := 0.10

noncomputable def taxable_income_amount (income : ℝ) (exemption : ℝ) : ℝ := income - exemption

noncomputable def personal_income_tax (income : ℝ) : ℝ :=
  let taxable_income := taxable_income_amount income tax_free_portion
  if taxable_income ≤ 500 then
    taxable_income * tax_rate_5_percent
  else
    (500 * tax_rate_5_percent) + ((taxable_income - 500) * tax_rate_10_percent)

theorem huangs_tax_is_65_yuan : personal_income_tax monthly_salary = 65 :=
by
  sorry

end huangs_tax_is_65_yuan_l18_18239


namespace quadratic_inequality_solution_set_quadratic_inequality_solution_set2_l18_18612

-- Proof Problem 1 Statement
theorem quadratic_inequality_solution_set (a b : ℝ) (h : ∀ x : ℝ, b < x ∧ x < 1 → ax^2 + 3 * x + 2 > 0) : 
  a = -5 ∧ b = -2/5 := sorry

-- Proof Problem 2 Statement
theorem quadratic_inequality_solution_set2 (a : ℝ) (h_pos : a > 0) : 
  ((0 < a ∧ a < 3) → (∀ x : ℝ, x < -3 / a ∨ x > -1 → ax^2 + 3 * x + 2 > -ax - 1)) ∧
  (a = 3 → (∀ x : ℝ, x ≠ -1 → ax^2 + 3 * x + 2 > -ax - 1)) ∧
  (a > 3 → (∀ x : ℝ, x < -1 ∨ x > -3 / a → ax^2 + 3 * x + 2 > -ax - 1)) := sorry

end quadratic_inequality_solution_set_quadratic_inequality_solution_set2_l18_18612


namespace find_coefficients_l18_18715

theorem find_coefficients (A B : ℚ) :
  (∀ x : ℚ, 2 * x + 7 = A * (x + 7) + B * (x - 9)) →
  A = 25 / 16 ∧ B = 7 / 16 :=
by
  intro h
  sorry

end find_coefficients_l18_18715


namespace solve_boys_left_l18_18729

--given conditions
variable (boys_initial girls_initial boys_left girls_entered children_end: ℕ)
variable (h_boys_initial : boys_initial = 5)
variable (h_girls_initial : girls_initial = 4)
variable (h_girls_entered : girls_entered = 2)
variable (h_children_end : children_end = 8)

-- Problem definition
def boys_left_proof : Prop :=
  ∃ (B : ℕ), boys_left = B ∧ boys_initial - B + girls_initial + girls_entered = children_end ∧ B = 3

-- The statement to be proven
theorem solve_boys_left : boys_left_proof boys_initial girls_initial boys_left girls_entered children_end := by
  -- Proof will be provided here
  sorry

end solve_boys_left_l18_18729


namespace cannot_determine_right_triangle_l18_18745

/-- Proof that the condition \(a^2 = 5\), \(b^2 = 12\), \(c^2 = 13\) cannot determine that \(\triangle ABC\) is a right triangle. -/
theorem cannot_determine_right_triangle (a b c : ℝ) (ha : a^2 = 5) (hb : b^2 = 12) (hc : c^2 = 13) : 
  ¬(a^2 + b^2 = c^2 ∨ b^2 + c^2 = a^2 ∨ c^2 + a^2 = b^2) := 
by
  sorry

end cannot_determine_right_triangle_l18_18745


namespace total_toys_l18_18059

theorem total_toys (m a t : ℕ) (h1 : a = m + 3 * m) (h2 : t = a + 2) (h3 : m = 6) : m + a + t = 56 := by
  sorry

end total_toys_l18_18059


namespace find_radius_and_diameter_l18_18169

theorem find_radius_and_diameter (M N r d : ℝ) (h1 : M = π * r^2) (h2 : N = 2 * π * r) (h3 : M / N = 15) : 
  (r = 30) ∧ (d = 60) := by
  sorry

end find_radius_and_diameter_l18_18169


namespace canoes_rented_more_than_kayaks_l18_18703

-- Defining the constants
def canoe_cost : ℕ := 11
def kayak_cost : ℕ := 16
def total_revenue : ℕ := 460
def canoe_ratio : ℕ := 4
def kayak_ratio : ℕ := 3

-- Main statement to prove
theorem canoes_rented_more_than_kayaks :
  ∃ (C K : ℕ), canoe_cost * C + kayak_cost * K = total_revenue ∧ (canoe_ratio * K = kayak_ratio * C) ∧ (C - K = 5) :=
by
  have h1 : canoe_cost = 11 := rfl
  have h2 : kayak_cost = 16 := rfl
  have h3 : total_revenue = 460 := rfl
  have h4 : canoe_ratio = 4 := rfl
  have h5 : kayak_ratio = 3 := rfl
  sorry

end canoes_rented_more_than_kayaks_l18_18703


namespace krishan_nandan_investment_l18_18258

def investment_ratio (k r₁ r₂ : ℕ) (N T Gn : ℕ) : Prop :=
  k = r₁ ∧ r₂ = 1 ∧ Gn = N * T ∧ k * N * 3 * T + Gn = 26000 ∧ Gn = 2000

/-- Given the conditions, the ratio of Krishan's investment to Nandan's investment is 4:1. -/
theorem krishan_nandan_investment :
  ∃ k N T Gn Gn_total : ℕ, 
    investment_ratio k 4 1 N T Gn  ∧ k * N * 3 * T = 24000 :=
by
  sorry

end krishan_nandan_investment_l18_18258


namespace find_integer_pairs_l18_18338

theorem find_integer_pairs (x y: ℤ) :
  x^2 - y^4 = 2009 → (x = 45 ∧ (y = 2 ∨ y = -2)) ∨ (x = -45 ∧ (y = 2 ∨ y = -2)) :=
by
  sorry

end find_integer_pairs_l18_18338


namespace find_n_l18_18989

theorem find_n 
  (a : ℝ := 9 / 15)
  (S1 : ℝ := 15 / (1 - a))
  (b : ℝ := (9 + n) / 15)
  (S2 : ℝ := 3 * S1)
  (hS1 : S1 = 37.5)
  (hS2 : S2 = 112.5)
  (hb : b = 13 / 15)
  (hn : 13 = 9 + n) : 
  n = 4 :=
by
  sorry

end find_n_l18_18989


namespace rectangle_ratio_l18_18826

theorem rectangle_ratio (a b c d : ℝ) (h₀ : a = 4)
  (h₁ : b = (4 / 3)) (h₂ : c = (8 / 3)) (h₃ : d = 4) :
  (∃ XY YZ, XY * YZ = a * a ∧ XY / YZ = 0.9) :=
by
  -- Proof to be filled
  sorry

end rectangle_ratio_l18_18826


namespace find_f_prime_one_l18_18108

noncomputable def f (f'_1 : ℝ) (x : ℝ) := f'_1 * x^3 - 2 * x^2 + 3

theorem find_f_prime_one (f'_1 : ℝ) 
  (h_derivative : ∀ x : ℝ, deriv (f f'_1) x = 3 * f'_1 * x^2 - 4 * x)
  (h_value_at_1 : deriv (f f'_1) 1 = f'_1) :
  f'_1 = 2 :=
by 
  sorry

end find_f_prime_one_l18_18108


namespace train_length_eq_l18_18648

-- Definitions
def train_speed_kmh : Float := 45
def crossing_time_s : Float := 30
def total_length_m : Float := 245

-- Theorem statement
theorem train_length_eq :
  ∃ (train_length bridge_length: Float),
  bridge_length = total_length_m - train_length ∧
  train_speed_kmh * 1000 / 3600 * crossing_time_s = train_length + bridge_length ∧
  train_length = 130 :=
by
  sorry

end train_length_eq_l18_18648


namespace relationship_y1_y2_l18_18983

theorem relationship_y1_y2
    (b : ℝ) 
    (y1 y2 : ℝ)
    (h1 : y1 = - (1 / 2) * (-2) + b) 
    (h2 : y2 = - (1 / 2) * 3 + b) : 
    y1 > y2 :=
sorry

end relationship_y1_y2_l18_18983


namespace sum_of_arithmetic_progression_l18_18194

theorem sum_of_arithmetic_progression :
  let a := 30
  let d := -3
  let n := 20
  let S_n := n / 2 * (2 * a + (n - 1) * d)
  S_n = 30 :=
by
  sorry

end sum_of_arithmetic_progression_l18_18194


namespace part1_part2_part3_l18_18093

variable (a b c : ℝ) (f : ℝ → ℝ)
-- Defining the polynomial function f
def polynomial (x : ℝ) : ℝ := a * x^5 + b * x^3 + 4 * x + c

theorem part1 (h0 : polynomial a b 6 0 = 6) : c = 6 :=
by sorry

theorem part2 (h1 : polynomial a b (-2) 0 = -2) (h2 : polynomial a b (-2) 1 = 5) : polynomial a b (-2) (-1) = -9 :=
by sorry

theorem part3 (h3 : polynomial a b 3 5 + polynomial a b 3 (-5) = 6) (h4 : polynomial a b 3 2 = 8) : polynomial a b 3 (-2) = -2 :=
by sorry

end part1_part2_part3_l18_18093


namespace not_converge_to_a_l18_18379

theorem not_converge_to_a (x : ℕ → ℝ) (a : ℝ) :
  (∀ ε > 0, ∀ k : ℕ, ∃ n : ℕ, n > k ∧ |x n - a| ≥ ε) →
  ¬ (∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |x n - a| < ε) :=
by sorry

end not_converge_to_a_l18_18379


namespace total_oranges_correct_l18_18575

-- Define the conditions
def oranges_per_child : Nat := 3
def number_of_children : Nat := 4

-- Define the total number of oranges and the statement to be proven
def total_oranges : Nat := oranges_per_child * number_of_children

theorem total_oranges_correct : total_oranges = 12 := by
  sorry

end total_oranges_correct_l18_18575


namespace union_cardinality_inequality_l18_18737

open Set

/-- Given three finite sets A, B, and C such that A ∩ B ∩ C = ∅,
prove that |A ∪ B ∪ C| ≥ 1/2 (|A| + |B| + |C|) -/
theorem union_cardinality_inequality (A B C : Finset ℕ)
  (h : (A ∩ B ∩ C) = ∅) : (A ∪ B ∪ C).card ≥ (A.card + B.card + C.card) / 2 := sorry

end union_cardinality_inequality_l18_18737


namespace total_points_sum_l18_18897

def g (n : ℕ) : ℕ :=
  if n % 3 = 0 then 8
  else if n % 2 = 0 then 3
  else 0

def allie_rolls := [6, 2, 5, 3, 4]
def carlos_rolls := [3, 2, 2, 6, 1]

def score (rolls : List ℕ) : ℕ :=
  rolls.map g |>.sum

theorem total_points_sum :
  score allie_rolls + score carlos_rolls = 44 :=
by
  sorry

end total_points_sum_l18_18897


namespace xy_equals_252_l18_18070

-- Definitions and conditions
variables (x y : ℕ) -- positive integers
variable (h1 : x + y = 36)
variable (h2 : 4 * x * y + 12 * x = 5 * y + 390)

-- Statement of the problem
theorem xy_equals_252 (h1 : x + y = 36) (h2 : 4 * x * y + 12 * x = 5 * y + 390) : x * y = 252 := by 
  sorry

end xy_equals_252_l18_18070


namespace equation_solution_l18_18075

theorem equation_solution (x : ℝ) (h : (x + 6) / (x - 3) = 4) : x = 6 :=
by sorry

end equation_solution_l18_18075


namespace total_interest_is_350_l18_18822

-- Define the principal amounts, rates, and time
def principal1 : ℝ := 1000
def rate1 : ℝ := 0.03
def principal2 : ℝ := 1200
def rate2 : ℝ := 0.05
def time : ℝ := 3.888888888888889

-- Calculate the interest for one year for each loan
def interest_per_year1 : ℝ := principal1 * rate1
def interest_per_year2 : ℝ := principal2 * rate2

-- Calculate the total interest for the time period for each loan
def total_interest1 : ℝ := interest_per_year1 * time
def total_interest2 : ℝ := interest_per_year2 * time

-- Finally, calculate the total interest amount
def total_interest_amount : ℝ := total_interest1 + total_interest2

-- The proof problem: Prove that total_interest_amount == 350 Rs
theorem total_interest_is_350 : total_interest_amount = 350 := by
  sorry

end total_interest_is_350_l18_18822


namespace perfect_squares_in_interval_l18_18037

theorem perfect_squares_in_interval (s : Set Int) (h1 : ∃ a : Nat, ∀ x ∈ s, a^4 ≤ x ∧ x ≤ (a+9)^4)
                                     (h2 : ∃ b : Nat, ∀ x ∈ s, b^3 ≤ x ∧ x ≤ (b+99)^3) :
  ∃ c : Nat, c ≥ 2000 ∧ ∀ x ∈ s, x = c^2 :=
sorry

end perfect_squares_in_interval_l18_18037


namespace incorrect_axis_symmetry_l18_18862

noncomputable def quadratic_function (x : ℝ) : ℝ := - (x + 2)^2 - 3

theorem incorrect_axis_symmetry :
  (∀ x : ℝ, quadratic_function x < 0) ∧
  (∀ x : ℝ, x > -1 → (quadratic_function x < quadratic_function (-2))) ∧
  (¬∃ x : ℝ, quadratic_function x = 0) ∧
  (¬ ∀ x : ℝ, x = 2) →
  false :=
by
  sorry

end incorrect_axis_symmetry_l18_18862


namespace middle_digit_base_7_of_reversed_base_9_l18_18446

noncomputable def middle_digit_of_number_base_7 (N : ℕ) : ℕ :=
  let x := (N / 81) % 9  -- Extract the first digit in base-9
  let y := (N / 9) % 9   -- Extract the middle digit in base-9
  let z := N % 9         -- Extract the last digit in base-9
  -- Given condition: 81x + 9y + z = 49z + 7y + x
  let eq1 := 81 * x + 9 * y + z
  let eq2 := 49 * z + 7 * y + x
  let condition := eq1 = eq2 ∧ 0 ≤ y ∧ y < 7 -- y is a digit in base-7
  if condition then y else sorry

theorem middle_digit_base_7_of_reversed_base_9 (N : ℕ) :
  (∃ (x y z : ℕ), x < 9 ∧ y < 9 ∧ z < 9 ∧
  N = 81 * x + 9 * y + z ∧ N = 49 * z + 7 * y + x) → middle_digit_of_number_base_7 N = 0 :=
  by sorry

end middle_digit_base_7_of_reversed_base_9_l18_18446


namespace employed_females_percentage_l18_18907

def P_total : ℝ := 0.64
def P_males : ℝ := 0.46

theorem employed_females_percentage : 
  ((P_total - P_males) / P_total) * 100 = 28.125 :=
by
  sorry

end employed_females_percentage_l18_18907


namespace sphere_surface_area_l18_18644

theorem sphere_surface_area (R r : ℝ) (h1 : 2 * OM = R) (h2 : ∀ r, π * r^2 = 3 * π) : 4 * π * R^2 = 16 * π :=
by
  sorry

end sphere_surface_area_l18_18644


namespace minimum_k_for_mutual_criticism_l18_18245

theorem minimum_k_for_mutual_criticism (k : ℕ) (h1 : 15 * k > 105) : k ≥ 8 := by
  sorry

end minimum_k_for_mutual_criticism_l18_18245


namespace bread_last_days_is_3_l18_18899

-- Define conditions
def num_members : ℕ := 4
def slices_breakfast : ℕ := 3
def slices_snacks : ℕ := 2
def slices_loaf : ℕ := 12
def num_loaves : ℕ := 5

-- Define the problem statement
def bread_last_days : ℕ :=
  (num_loaves * slices_loaf) / (num_members * (slices_breakfast + slices_snacks))

-- State the theorem to be proved
theorem bread_last_days_is_3 : bread_last_days = 3 :=
  sorry

end bread_last_days_is_3_l18_18899


namespace wristwatch_cost_proof_l18_18838

-- Definition of the problem conditions
def allowance_per_week : ℕ := 5
def initial_weeks : ℕ := 10
def initial_savings : ℕ := 20
def additional_weeks : ℕ := 16

-- The total cost of the wristwatch
def wristwatch_cost : ℕ := 100

-- Let's state the proof problem
theorem wristwatch_cost_proof :
  (initial_savings + additional_weeks * allowance_per_week) = wristwatch_cost :=
by
  sorry

end wristwatch_cost_proof_l18_18838


namespace age_is_nine_l18_18746

-- Define the conditions
def current_age (X : ℕ) :=
  X = 3 * (X - 6)

-- The theorem: Prove that the age X is equal to 9 under the conditions given
theorem age_is_nine (X : ℕ) (h : current_age X) : X = 9 :=
by
  -- The proof is omitted
  sorry

end age_is_nine_l18_18746


namespace area_of_region_l18_18225

theorem area_of_region : 
  (∃ A : ℝ, 
    (∀ x y : ℝ, 
      (|4 * x - 20| + |3 * y + 9| ≤ 4) → 
      A = (32 / 3))) :=
by 
  sorry

end area_of_region_l18_18225


namespace probability_odd_and_multiple_of_5_l18_18651

/-- Given three distinct integers selected at random between 1 and 2000, inclusive, the probability that the product of the three integers is odd and a multiple of 5 is between 0.01 and 0.05. -/
theorem probability_odd_and_multiple_of_5 :
  ∃ p : ℚ, (0.01 < p ∧ p < 0.05) :=
sorry

end probability_odd_and_multiple_of_5_l18_18651


namespace radius_of_shorter_tank_l18_18246

theorem radius_of_shorter_tank (h : ℝ) (r : ℝ) 
  (volume_eq : ∀ (π : ℝ), π * (10^2) * (2 * h) = π * (r^2) * h) : 
  r = 10 * Real.sqrt 2 := 
by 
  sorry

end radius_of_shorter_tank_l18_18246


namespace hoseok_result_l18_18472

theorem hoseok_result :
  ∃ X : ℤ, (X - 46 = 15) ∧ (X - 29 = 32) :=
by
  sorry

end hoseok_result_l18_18472


namespace sum_partition_36_l18_18383

theorem sum_partition_36 : 
  ∃ (S : Finset ℕ), S.card = 36 ∧ S.sum id = ((Finset.range 72).sum id) / 2 :=
by
  sorry

end sum_partition_36_l18_18383


namespace rectangular_prism_volume_dependency_l18_18348

theorem rectangular_prism_volume_dependency (a : ℝ) (V : ℝ) (h : a > 2) :
  V = a * 2 * 1 → (∀ a₀ > 2, a ≠ a₀ → V ≠ a₀ * 2 * 1) :=
by
  sorry

end rectangular_prism_volume_dependency_l18_18348


namespace pentagon_largest_angle_l18_18459

theorem pentagon_largest_angle (x : ℝ) (h : 2 * x + 3 * x + 4 * x + 5 * x + 6 * x = 540) : 6 * x = 162 :=
sorry

end pentagon_largest_angle_l18_18459


namespace smallest_k_for_divisibility_l18_18956

theorem smallest_k_for_divisibility (z : ℂ) (hz : z^7 = 1) : ∃ k : ℕ, (∀ m : ℕ, z ^ (m * k) = 1) ∧ k = 84 :=
sorry

end smallest_k_for_divisibility_l18_18956


namespace sues_answer_l18_18851

theorem sues_answer (x : ℕ) (hx : x = 6) : 
  let b := 2 * (x + 1)
  let s := 2 * (b - 1)
  s = 26 :=
by
  sorry

end sues_answer_l18_18851


namespace trajectory_of_point_l18_18384

theorem trajectory_of_point 
  (P : ℝ × ℝ) 
  (h1 : abs (P.1 - 4) + P.2^2 - 1 = abs (P.1 + 5)) : 
  P.2^2 = 16 * P.1 := 
sorry

end trajectory_of_point_l18_18384


namespace solution_set_of_inequality_l18_18291

theorem solution_set_of_inequality (a t : ℝ) (h1 : ∀ x : ℝ, x^2 - 2 * a * x + a > 0) : 
  a > 0 ∧ a < 1 → (a^(2*t + 1) < a^(t^2 + 2*t - 3) ↔ -2 < t ∧ t < 2) :=
by
  intro ha
  have h : (0 < a ∧ a < 1) := sorry
  exact sorry

end solution_set_of_inequality_l18_18291


namespace container_emptying_l18_18292

theorem container_emptying (a b c : ℕ) : ∃ m n k : ℕ,
  (m = 0 ∨ n = 0 ∨ k = 0) ∧
  (∀ a' b' c', 
    (a' = a ∧ b' = b ∧ c' = c) ∨ 
    (a' + 2 * b' = a' ∧ b' = b ∧ c' + 2 * b' = c') ∨ 
    (a' + 2 * c' = a' ∧ b' + 2 * c' = b' ∧ c' = c') ∨ 
    (a + 2 * b' + c' = a' + 2 * m * (a + b') ∧ b' = n * (a + b') ∧ c' = k * (a + b')) 
  -> (a' = 0 ∨ b' = 0 ∨ c' = 0)) :=
sorry

end container_emptying_l18_18292


namespace treaty_signed_on_tuesday_l18_18395

-- Define a constant for the start date and the number of days
def start_day_of_week : ℕ := 1 -- Monday is represented by 1
def days_until_treaty : ℕ := 1301

-- Function to calculate the resulting day of the week
def day_of_week_after_days (start_day : ℕ) (days : ℕ) : ℕ :=
  (start_day + days) % 7

-- Theorem statement: Prove that 1301 days after Monday is Tuesday
theorem treaty_signed_on_tuesday :
  day_of_week_after_days start_day_of_week days_until_treaty = 2 :=
by
  -- placeholder for the proof
  sorry

end treaty_signed_on_tuesday_l18_18395


namespace area_of_triangle_example_l18_18757

noncomputable def area_of_triangle (A B C : (ℝ × ℝ)) : ℝ :=
  0.5 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

theorem area_of_triangle_example : 
  area_of_triangle (3, 3) (3, 10) (12, 19) = 31.5 :=
by
  sorry

end area_of_triangle_example_l18_18757


namespace no_intersection_l18_18146

def f₁ (x : ℝ) : ℝ := abs (3 * x + 6)
def f₂ (x : ℝ) : ℝ := -abs (4 * x - 1)

theorem no_intersection : ∀ x, f₁ x ≠ f₂ x :=
by
  sorry

end no_intersection_l18_18146


namespace pens_sold_to_recover_investment_l18_18160

-- Given the conditions
variables (P C : ℝ) (N : ℝ)
-- P is the total cost of 30 pens
-- C is the cost price of each pen
-- N is the number of pens sold to recover the initial investment

-- Stating the conditions
axiom h1 : P = 30 * C
axiom h2 : N * 1.5 * C = P

-- Proving that N = 20
theorem pens_sold_to_recover_investment (P C N : ℝ) (h1 : P = 30 * C) (h2 : N * 1.5 * C = P) : N = 20 :=
by
  sorry

end pens_sold_to_recover_investment_l18_18160


namespace track_circumference_l18_18972

variable (A B : Nat → ℝ)
variable (speedA speedB : ℝ)
variable (x : ℝ) -- half the circumference of the track
variable (y : ℝ) -- the circumference of the track

theorem track_circumference
  (x_pos : 0 < x)
  (y_def : y = 2 * x)
  (start_opposite : A 0 = 0 ∧ B 0 = x)
  (B_first_meet_150 : ∃ t₁, B t₁ = 150 ∧ A t₁ = x - 150)
  (A_second_meet_90 : ∃ t₂, A t₂ = 2 * x - 90 ∧ B t₂ = x + 90) :
  y = 720 := 
by 
  sorry

end track_circumference_l18_18972


namespace cos_sin_sum_l18_18752

open Real

theorem cos_sin_sum (α : ℝ) (h : (cos (2 * α)) / (sin (α - π / 4)) = -sqrt 2 / 2) : cos α + sin α = 1 / 2 := by
  sorry

end cos_sin_sum_l18_18752


namespace fireworks_display_l18_18141

def num_digits_year : ℕ := 4
def fireworks_per_digit : ℕ := 6
def regular_letters_phrase : ℕ := 12
def fireworks_per_regular_letter : ℕ := 5

def fireworks_H : ℕ := 8
def fireworks_E : ℕ := 7
def fireworks_L : ℕ := 6
def fireworks_O : ℕ := 9

def num_boxes : ℕ := 100
def fireworks_per_box : ℕ := 10

def total_fireworks : ℕ :=
  (num_digits_year * fireworks_per_digit) +
  (regular_letters_phrase * fireworks_per_regular_letter) +
  (fireworks_H + fireworks_E + 2 * fireworks_L + fireworks_O) + 
  (num_boxes * fireworks_per_box)

theorem fireworks_display : total_fireworks = 1120 := by
  sorry

end fireworks_display_l18_18141


namespace min_expression_value_l18_18282

theorem min_expression_value (a b c : ℝ) (ha : 1 ≤ a) (hbc : b ≥ a) (hcb : c ≥ b) (hc5 : c ≤ 5) :
  (a - 1)^2 + (b / a - 1)^2 + (c / b - 1)^2 + (5 / c - 1)^2 ≥ 2 * Real.sqrt 5 - 4 * Real.sqrt (5^(1/4)) + 4 :=
sorry

end min_expression_value_l18_18282


namespace ratio_doubled_to_original_l18_18904

theorem ratio_doubled_to_original (x y : ℕ) (h1 : y = 2 * x + 9) (h2 : 3 * y = 57) : 2 * x = 2 * (x / 1) := 
by sorry

end ratio_doubled_to_original_l18_18904


namespace units_digit_of_a_l18_18049

theorem units_digit_of_a :
  (2003^2004 - 2004^2003) % 10 = 7 :=
by
  sorry

end units_digit_of_a_l18_18049


namespace selling_price_correct_l18_18097

-- Define the parameters
def stamp_duty_rate : ℝ := 0.002
def commission_rate : ℝ := 0.0035
def bought_shares : ℝ := 3000
def buying_price_per_share : ℝ := 12
def profit : ℝ := 5967

-- Define the selling price per share
noncomputable def selling_price_per_share (x : ℝ) : ℝ :=
  bought_shares * x - bought_shares * buying_price_per_share -
  bought_shares * x * (stamp_duty_rate + commission_rate) - 
  bought_shares * buying_price_per_share * (stamp_duty_rate + commission_rate)

-- The target selling price per share
def target_selling_price_per_share : ℝ := 14.14

-- Statement of the problem
theorem selling_price_correct (x : ℝ) : selling_price_per_share x = profit → x = target_selling_price_per_share := by
  sorry

end selling_price_correct_l18_18097


namespace tabby_swimming_speed_l18_18131

theorem tabby_swimming_speed :
  ∃ (S : ℝ), S = 4.125 ∧ (∀ (D : ℝ), 6 = (2 * D) / ((D / S) + (D / 11))) :=
by {
 sorry
}

end tabby_swimming_speed_l18_18131


namespace negation_proof_l18_18280

theorem negation_proof :
  (¬ ∃ x : ℝ, x ≤ 1 ∨ x^2 > 4) ↔ (∀ x : ℝ, x > 1 ∧ x^2 ≤ 4) :=
by
  sorry

end negation_proof_l18_18280


namespace compound_interest_semiannual_l18_18111

theorem compound_interest_semiannual
  (P : ℝ) (r : ℝ) (n : ℕ) (t : ℝ)
  (initial_amount : P = 900)
  (interest_rate : r = 0.10)
  (compounding_periods : n = 2)
  (time_period : t = 1) :
  P * (1 + r / n) ^ (n * t) = 992.25 :=
by
  sorry

end compound_interest_semiannual_l18_18111


namespace decreasing_interval_implies_range_of_a_l18_18138

noncomputable def f (a x : ℝ) : ℝ := x^2 + 2 * (a - 1) * x + 2

theorem decreasing_interval_implies_range_of_a (a : ℝ)
  (h : ∀ x y : ℝ, x ≤ y → y ≤ 4 → f a x ≥ f a y) : a ≤ -3 :=
by
  sorry

end decreasing_interval_implies_range_of_a_l18_18138


namespace find_x_for_parallel_vectors_l18_18730

noncomputable def vector_m : (ℝ × ℝ) := (1, 2)
noncomputable def vector_n (x : ℝ) : (ℝ × ℝ) := (x, 2 - 2 * x)

theorem find_x_for_parallel_vectors :
  ∀ x : ℝ, (1, 2).fst * (2 - 2 * x) - (1, 2).snd * x = 0 → x = 1 / 2 :=
by
  intros
  exact sorry

end find_x_for_parallel_vectors_l18_18730


namespace negation_seated_l18_18047

variable (Person : Type) (in_room : Person → Prop) (seated : Person → Prop)

theorem negation_seated :
  ¬ (∀ x, in_room x → seated x) ↔ ∃ x, in_room x ∧ ¬ seated x :=
by sorry

end negation_seated_l18_18047


namespace range_of_x_l18_18159

noncomputable def f (x : ℝ) : ℝ :=
  Real.log (Real.exp x + Real.exp (-x)) + x^2

theorem range_of_x (x : ℝ) : 
  (∃ y z : ℝ, y = 2 * x - 1 ∧ f x > f y ∧ x > 1 / 3 ∧ x < 1) :=
sorry

end range_of_x_l18_18159


namespace factorize_expression_l18_18441

variable {a b : ℝ} -- define a and b as real numbers

theorem factorize_expression : a^2 * b - 9 * b = b * (a + 3) * (a - 3) :=
by
  sorry

end factorize_expression_l18_18441


namespace slope_of_asymptotes_l18_18365

noncomputable def hyperbola_asymptote_slope (x y : ℝ) : Prop :=
  (x^2 / 144 - y^2 / 81 = 1)

theorem slope_of_asymptotes (x y : ℝ) (h : hyperbola_asymptote_slope x y) :
  ∃ m : ℝ, m = 3 / 4 ∨ m = -3 / 4 :=
sorry

end slope_of_asymptotes_l18_18365


namespace largest_number_is_A_l18_18671

def numA : ℝ := 0.989
def numB : ℝ := 0.9879
def numC : ℝ := 0.98809
def numD : ℝ := 0.9807
def numE : ℝ := 0.9819

theorem largest_number_is_A :
  (numA > numB) ∧ (numA > numC) ∧ (numA > numD) ∧ (numA > numE) :=
by sorry

end largest_number_is_A_l18_18671


namespace number_of_zeros_of_f_l18_18120

noncomputable def f (x : ℝ) : ℝ := Real.cos x - Real.sin (2 * x)

theorem number_of_zeros_of_f : (∃ l : List ℝ, (∀ x ∈ l, 0 ≤ x ∧ x ≤ 2 * Real.pi ∧ f x = 0) ∧ l.length = 4) := 
by
  sorry

end number_of_zeros_of_f_l18_18120


namespace people_got_on_at_third_stop_l18_18045

theorem people_got_on_at_third_stop :
  let people_1st_stop := 10
  let people_off_2nd_stop := 3
  let twice_people_1st_stop := 2 * people_1st_stop
  let people_off_3rd_stop := 18
  let people_after_3rd_stop := 12

  let people_after_1st_stop := people_1st_stop
  let people_after_2nd_stop := (people_after_1st_stop - people_off_2nd_stop) + twice_people_1st_stop
  let people_after_3rd_stop_but_before_new_ones := people_after_2nd_stop - people_off_3rd_stop
  let people_on_at_3rd_stop := people_after_3rd_stop - people_after_3rd_stop_but_before_new_ones

  people_on_at_3rd_stop = 3 := 
by
  sorry

end people_got_on_at_third_stop_l18_18045


namespace freeze_time_l18_18470

theorem freeze_time :
  ∀ (minutes_per_smoothie total_minutes num_smoothies freeze_time: ℕ),
    minutes_per_smoothie = 3 →
    total_minutes = 55 →
    num_smoothies = 5 →
    freeze_time = total_minutes - (num_smoothies * minutes_per_smoothie) →
    freeze_time = 40 :=
by
  intros minutes_per_smoothie total_minutes num_smoothies freeze_time
  intros H1 H2 H3 H4
  subst H1
  subst H2
  subst H3
  subst H4
  sorry

end freeze_time_l18_18470


namespace reflected_line_equation_l18_18233

-- Definitions based on given conditions
def incident_line (x : ℝ) : ℝ := 2 * x + 1
def reflection_line (x : ℝ) : ℝ := x

-- Statement of the mathematical problem
theorem reflected_line_equation :
  ∀ x y : ℝ, (incident_line x = y) → (reflection_line x = x) → y = (1/2) * x - (1/2) :=
sorry

end reflected_line_equation_l18_18233


namespace minimum_bounces_to_reach_height_l18_18764

noncomputable def height_after_bounces (initial_height : ℝ) (bounce_factor : ℝ) (k : ℕ) : ℝ :=
  initial_height * (bounce_factor ^ k)

theorem minimum_bounces_to_reach_height
  (initial_height : ℝ) (bounce_factor : ℝ) (min_height : ℝ) :
  initial_height = 800 → bounce_factor = 0.5 → min_height = 2 →
  (∀ k : ℕ, height_after_bounces initial_height bounce_factor k < min_height ↔ k ≥ 9) := 
by
  intros h₀ b₀ m₀
  rw [h₀, b₀, m₀]
  sorry

end minimum_bounces_to_reach_height_l18_18764


namespace movies_left_to_watch_l18_18317

theorem movies_left_to_watch (total_movies : ℕ) (movies_watched : ℕ) : total_movies = 17 ∧ movies_watched = 7 → (total_movies - movies_watched) = 10 :=
by
  sorry

end movies_left_to_watch_l18_18317


namespace sum_of_possible_values_of_k_l18_18275

open Complex

theorem sum_of_possible_values_of_k (x y z k : ℂ) (hxyz : x ≠ y ∧ y ≠ z ∧ z ≠ x)
    (h : x / (1 - y + z) = k ∧ y / (1 - z + x) = k ∧ z / (1 - x + y) = k) : k = 1 :=
by
  sorry

end sum_of_possible_values_of_k_l18_18275


namespace triangle_is_right_triangle_l18_18846

theorem triangle_is_right_triangle
  (a b c : ℝ)
  (A B C : ℝ)
  (h₁ : a ≠ b)
  (h₂ : (a^2 + b^2) * Real.sin (A - B) = (a^2 - b^2) * Real.sin (A + B))
  (A_ne_B : A ≠ B)
  (hABC : A + B + C = Real.pi) :
  C = Real.pi / 2 :=
by
  sorry

end triangle_is_right_triangle_l18_18846


namespace N_prime_iff_k_eq_2_l18_18486

/-- Define the number N for a given k -/
def N (k : ℕ) : ℕ := (10 ^ (2 * k) - 1) / 99

/-- Statement: Prove that N is prime if and only if k = 2 -/
theorem N_prime_iff_k_eq_2 (k : ℕ) : Prime (N k) ↔ k = 2 := by
  sorry

end N_prime_iff_k_eq_2_l18_18486


namespace machine_initial_value_l18_18033

-- Conditions
def initial_value (P : ℝ) : Prop := P * (0.75 ^ 2) = 4000

noncomputable def initial_market_value : ℝ := 4000 / (0.75 ^ 2)

-- Proof problem statement
theorem machine_initial_value (P : ℝ) (h : initial_value P) : P = 4000 / (0.75 ^ 2) :=
by
  sorry

end machine_initial_value_l18_18033


namespace triangle_angle_sum_l18_18308

theorem triangle_angle_sum (y : ℝ) (h : 40 + 3 * y + (y + 10) = 180) : y = 32.5 :=
by
  sorry

end triangle_angle_sum_l18_18308


namespace height_of_pole_l18_18553

/-- A telephone pole is supported by a steel cable extending from the top of the pole to a point on the ground 3 meters from its base.
When Leah, who is 1.5 meters tall, stands 2.5 meters from the base of the pole towards the point where the cable is attached to the ground,
her head just touches the cable. Prove that the height of the pole is 9 meters. -/
theorem height_of_pole 
  (cable_length_from_base : ℝ)
  (leah_distance_from_base : ℝ)
  (leah_height : ℝ)
  : cable_length_from_base = 3 → leah_distance_from_base = 2.5 → leah_height = 1.5 → 
    (∃ height_of_pole : ℝ, height_of_pole = 9) := 
by
  intros h1 h2 h3
  sorry

end height_of_pole_l18_18553


namespace no_prime_divisible_by_57_l18_18166

/-- A prime number is a natural number greater than 1 that has no positive divisors other than 1 and itself. --/
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

/-- Given that 57 is equal to 3 times 19.--/
theorem no_prime_divisible_by_57 : ∀ p : ℕ, is_prime p → ¬ (57 ∣ p) :=
by
  sorry

end no_prime_divisible_by_57_l18_18166


namespace not_integer_20_diff_l18_18193

theorem not_integer_20_diff (a b : ℝ) (hne : a ≠ b) 
  (no_roots1 : ∀ x, x^2 + 20 * a * x + 10 * b ≠ 0) 
  (no_roots2 : ∀ x, x^2 + 20 * b * x + 10 * a ≠ 0) : 
  ¬ (∃ k : ℤ, 20 * (b - a) = k) :=
by
  sorry

end not_integer_20_diff_l18_18193


namespace eight_b_value_l18_18094

theorem eight_b_value (a b : ℝ) (h1 : 6 * a + 3 * b = 0) (h2 : a = b - 3) : 8 * b = 16 :=
by
  sorry

end eight_b_value_l18_18094


namespace total_marks_more_than_physics_l18_18650

variable (P C M : ℕ)

theorem total_marks_more_than_physics :
  (P + C + M > P) ∧ ((C + M) / 2 = 75) → (P + C + M) - P = 150 := by
  intros h
  sorry

end total_marks_more_than_physics_l18_18650


namespace minimum_distance_l18_18928

noncomputable def point_on_curve (x : ℝ) : ℝ := -x^2 + 3 * Real.log x

noncomputable def point_on_line (x : ℝ) : ℝ := x + 2

theorem minimum_distance 
  (a b c d : ℝ) 
  (hP : b = point_on_curve a) 
  (hQ : d = point_on_line c) 
  : (a - c)^2 + (b - d)^2 = 8 :=
by
  sorry

end minimum_distance_l18_18928


namespace students_with_B_l18_18073

theorem students_with_B (students_jacob : ℕ) (students_B_jacob : ℕ) (students_smith : ℕ) (ratio_same : (students_B_jacob / students_jacob : ℚ) = 2 / 5) : 
  ∃ y : ℕ, (y / students_smith : ℚ) = 2 / 5 ∧ y = 12 :=
by 
  use 12
  sorry

end students_with_B_l18_18073


namespace jamie_dimes_l18_18748

theorem jamie_dimes (p n d : ℕ) (h1 : p + n + d = 50) (h2 : p + 5 * n + 10 * d = 240) : d = 10 :=
sorry

end jamie_dimes_l18_18748


namespace point_on_circle_l18_18584

noncomputable def distance_from_origin (x : ℝ) (y : ℝ) : ℝ :=
  Real.sqrt (x^2 + y^2)

theorem point_on_circle : distance_from_origin (-3) 4 = 5 := by
  sorry

end point_on_circle_l18_18584


namespace number_of_ordered_pairs_xy_2007_l18_18965

theorem number_of_ordered_pairs_xy_2007 : 
  ∃ n, n = 6 ∧ (∀ x y : ℕ, x * y = 2007 → x > 0 ∧ y > 0) :=
sorry

end number_of_ordered_pairs_xy_2007_l18_18965


namespace sin_P_equals_one_l18_18708

theorem sin_P_equals_one
  (x y : ℝ) (h1 : (1 / 2) * x * y * Real.sin 1 = 50) (h2 : x * y = 100) :
  Real.sin 1 = 1 :=
by sorry

end sin_P_equals_one_l18_18708


namespace units_digit_of_5_to_4_l18_18431

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_of_5_to_4 : units_digit (5^4) = 5 := by
  -- The definition ensures that 5^4 = 625 and the units digit is 5
  sorry

end units_digit_of_5_to_4_l18_18431


namespace test_average_score_l18_18143

theorem test_average_score (A : ℝ) (h : 0.90 * A + 5 = 86) : A = 90 := 
by
  sorry

end test_average_score_l18_18143


namespace value_of_t_l18_18511

theorem value_of_t (k m r s t : ℕ) 
  (hk : 1 ≤ k) (hm : 2 ≤ m) (hr : r = 13) (hs : s = 14)
  (h : k < m) (h' : m < r) (h'' : r < s) (h''' : s < t)
  (average_condition : (k + m + r + s + t) / 5 = 10) :
  t = 20 := 
sorry

end value_of_t_l18_18511


namespace circle_diameter_line_eq_l18_18440

theorem circle_diameter_line_eq (x y : ℝ) :
  x^2 + y^2 - 2*x + 6*y + 8 = 0 → (2 * 1 + (-3) + 1 = 0) :=
by
  sorry

end circle_diameter_line_eq_l18_18440


namespace winning_strategy_for_B_l18_18662

theorem winning_strategy_for_B (N : ℕ) (h : N < 15) : N = 7 ↔ (∃ strategy : (Fin 6 → ℕ) → ℕ, ∀ f : Fin 6 → ℕ, (strategy f) % 1001 = 0) :=
by
  sorry

end winning_strategy_for_B_l18_18662


namespace find_2a_minus_b_l18_18163

-- Define conditions
def f (x : ℝ) (a b : ℝ) := a * x + b
def g (x : ℝ) := -5 * x + 7
def h (x : ℝ) (a b : ℝ) := f (g x) a b
def h_inv (x : ℝ) := x - 9

-- Statement to prove
theorem find_2a_minus_b (a b : ℝ) 
(h_eq : ∀ x, h x a b = a * (-5 * x + 7) + b)
(h_inv_eq : ∀ x, h_inv x = x - 9)
(h_hinv_eq : ∀ x, h (h_inv x) a b = x) :
  2 * a - b = -54 / 5 := sorry

end find_2a_minus_b_l18_18163


namespace det_B_squared_sub_3B_eq_10_l18_18452

noncomputable def B : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![2, 3], ![2, 2]]

theorem det_B_squared_sub_3B_eq_10 : 
  Matrix.det (B * B - 3 • B) = 10 := by
  sorry

end det_B_squared_sub_3B_eq_10_l18_18452


namespace circle_equation1_circle_equation2_l18_18806

-- Definitions for the first question
def center1 : (ℝ × ℝ) := (2, -2)
def pointP : (ℝ × ℝ) := (6, 3)

-- Definitions for the second question
def pointA : (ℝ × ℝ) := (-4, -5)
def pointB : (ℝ × ℝ) := (6, -1)

-- Theorems we need to prove
theorem circle_equation1 : (x - 2)^2 + (y + 2)^2 = 41 :=
sorry

theorem circle_equation2 : (x - 1)^2 + (y + 3)^2 = 29 :=
sorry

end circle_equation1_circle_equation2_l18_18806


namespace jake_hours_of_work_l18_18501

def initialDebt : ℕ := 100
def amountPaid : ℕ := 40
def workRate : ℕ := 15
def remainingDebt : ℕ := initialDebt - amountPaid

theorem jake_hours_of_work : remainingDebt / workRate = 4 := by
  sorry

end jake_hours_of_work_l18_18501


namespace area_triangle_parabola_l18_18424

noncomputable def area_of_triangle_ABC (d : ℝ) (x : ℝ) : ℝ :=
  let A := (x, x^2)
  let B := (x + d, (x + d)^2)
  let C := (x + 2 * d, (x + 2 * d)^2)
  1 / 2 * abs (x * ((x + 2 * d)^2 - (x + d)^2) + (x + d) * ((x + 2 * d)^2 - x^2) + (x + 2 * d) * (x^2 - (x + d)^2))

theorem area_triangle_parabola (d : ℝ) (h_d : 0 < d) (x : ℝ) : 
  area_of_triangle_ABC d x = d^2 := sorry

end area_triangle_parabola_l18_18424


namespace increasing_intervals_g_l18_18122

noncomputable def f (x : ℝ) : ℝ := x^2 - 2 * x + 3

noncomputable def g (x : ℝ) : ℝ := f (2 - x^2)

theorem increasing_intervals_g : 
  (∀ x ∈ Set.Icc (-1 : ℝ) (0 : ℝ), ∀ y ∈ Set.Icc (-1 : ℝ) (0 : ℝ), x ≤ y → g x ≤ g y) ∧
  (∀ x ∈ Set.Ici (1 : ℝ), ∀ y ∈ Set.Ici (1 : ℝ), x ≤ y → g x ≤ g y) := 
sorry

end increasing_intervals_g_l18_18122


namespace twelve_months_game_probability_l18_18414

/-- The card game "Twelve Months" involves turning over cards according to a set of rules.
Given the rules, we are asked to find the probability that all 12 columns of cards can be fully turned over. -/
def twelve_months_probability : ℚ :=
  1 / 12

theorem twelve_months_game_probability :
  twelve_months_probability = 1 / 12 :=
by
  -- The conditions and their representations are predefined.
  sorry

end twelve_months_game_probability_l18_18414


namespace multiple_of_pumpkins_l18_18081

theorem multiple_of_pumpkins (M S : ℕ) (hM : M = 14) (hS : S = 54) (h : S = x * M + 12) : x = 3 := sorry

end multiple_of_pumpkins_l18_18081


namespace total_production_first_four_days_max_min_production_difference_total_wage_for_week_l18_18649

open Int

/-- Problem Statement -/
def planned_production : Int := 220

def production_change : List Int :=
  [5, -2, -4, 13, -10, 16, -9]

/-- Proof problem for total production in the first four days -/
theorem total_production_first_four_days :
  let first_four_days := production_change.take 4
  let total_change := first_four_days.sum
  let planned_first_four_days := planned_production * 4
  planned_first_four_days + total_change = 892 := 
by
  sorry

/-- Proof problem for difference in production between highest and lowest days -/
theorem max_min_production_difference :
  let max_change := production_change.maximum.getD 0
  let min_change := production_change.minimum.getD 0
  max_change - min_change = 26 := 
by
  sorry

/-- Proof problem for total wage calculation for the week -/
theorem total_wage_for_week :
  let total_change := production_change.sum
  let planned_week_total := planned_production * 7
  let actual_total := planned_week_total + total_change
  let base_wage := actual_total * 100
  let additional_wage := total_change * 20
  base_wage + additional_wage = 155080 := 
by
  sorry

end total_production_first_four_days_max_min_production_difference_total_wage_for_week_l18_18649


namespace swimming_distance_l18_18344

theorem swimming_distance
  (t : ℝ) (d_up : ℝ) (d_down : ℝ) (v_man : ℝ) (v_stream : ℝ)
  (h1 : v_man = 5) (h2 : t = 5) (h3 : d_up = 20) 
  (h4 : d_up = (v_man - v_stream) * t) :
  d_down = (v_man + v_stream) * t :=
by
  sorry

end swimming_distance_l18_18344


namespace basketball_teams_l18_18273

theorem basketball_teams (boys girls : ℕ) (total_players : ℕ) (team_size : ℕ) (ways : ℕ) :
  boys = 7 → girls = 3 → total_players = 10 → team_size = 5 → ways = 105 → 
  ∃ (girls_in_team1 girls_in_team2 : ℕ), 
    girls_in_team1 + girls_in_team2 = 3 ∧ 
    1 ≤ girls_in_team1 ∧ 
    1 ≤ girls_in_team2 ∧ 
    girls_in_team1 ≠ 0 ∧ 
    girls_in_team2 ≠ 0 ∧ 
    ways = 105 :=
by 
  sorry

end basketball_teams_l18_18273


namespace sum_of_transformed_numbers_l18_18092

theorem sum_of_transformed_numbers (a b S : ℝ) (h : a + b = S) : 3 * (a - 5) + 3 * (b - 5) = 3 * S - 30 :=
by
  sorry

end sum_of_transformed_numbers_l18_18092


namespace ones_digit_of_largest_power_of_three_dividing_27_factorial_l18_18891

theorem ones_digit_of_largest_power_of_three_dividing_27_factorial :
  let k := (27 / 3) + (27 / 9) + (27 / 27)
  let x := 3^k
  (x % 10) = 3 := by
  sorry

end ones_digit_of_largest_power_of_three_dividing_27_factorial_l18_18891


namespace range_of_a_for_negative_root_l18_18038

theorem range_of_a_for_negative_root (a : ℝ) :
  (∃ x : ℝ, x < 0 ∧ 7^(x + 1) - 7^x * a - a - 5 = 0) ↔ -5 < a ∧ a < 1 :=
by
  sorry

end range_of_a_for_negative_root_l18_18038


namespace arithmetic_mean_difference_l18_18210

theorem arithmetic_mean_difference (p q r : ℝ)
  (h1 : (p + q) / 2 = 10)
  (h2 : (q + r) / 2 = 22) :
  r - p = 24 :=
by
  sorry

end arithmetic_mean_difference_l18_18210


namespace fraction_of_Bhupathi_is_point4_l18_18963

def abhinav_and_bhupathi_amounts (A B : ℝ) : Prop :=
  A + B = 1210 ∧ B = 484

theorem fraction_of_Bhupathi_is_point4 (A B : ℝ) (x : ℝ) (h : abhinav_and_bhupathi_amounts A B) :
  (4 / 15) * A = x * B → x = 0.4 :=
by
  sorry

end fraction_of_Bhupathi_is_point4_l18_18963


namespace product_of_consecutive_integers_l18_18922

theorem product_of_consecutive_integers (l : List ℤ) (h1 : l.length = 2019) (h2 : l.sum = 2019) : l.prod = 0 := 
sorry

end product_of_consecutive_integers_l18_18922


namespace percentage_of_page_used_l18_18590

theorem percentage_of_page_used (length width side_margin top_margin : ℝ) (h_length : length = 30) (h_width : width = 20) (h_side_margin : side_margin = 2) (h_top_margin : top_margin = 3) :
  ( ((length - 2 * top_margin) * (width - 2 * side_margin)) / (length * width) ) * 100 = 64 := 
by
  sorry

end percentage_of_page_used_l18_18590


namespace tables_needed_for_luncheon_l18_18191

theorem tables_needed_for_luncheon (invited attending remaining tables_needed : ℕ) (H1 : invited = 24) (H2 : remaining = 10) (H3 : attending = invited - remaining) (H4 : tables_needed = attending / 7) : tables_needed = 2 :=
by
  sorry

end tables_needed_for_luncheon_l18_18191


namespace average_comparison_l18_18255

theorem average_comparison (x : ℝ) : 
    (14 + 32 + 53) / 3 = 3 + (21 + 47 + x) / 3 → 
    x = 22 :=
by 
  sorry

end average_comparison_l18_18255


namespace cos_theta_value_projection_value_l18_18230

noncomputable def vec_a : (ℝ × ℝ) := (3, 1)
noncomputable def vec_b : (ℝ × ℝ) := (-2, 4)

theorem cos_theta_value :
  let a := vec_a
  let b := vec_b
  let dot_product := a.1 * b.1 + a.2 * b.2
  let magnitude_a := Real.sqrt (a.1^2 + a.2^2)
  let magnitude_b := Real.sqrt (b.1^2 + b.2^2)
  dot_product / (magnitude_a * magnitude_b) = - Real.sqrt 2 / 10 :=
by 
  sorry

theorem projection_value :
  let a := vec_a
  let b := vec_b
  let dot_product := a.1 * b.1 + a.2 * b.2
  let magnitude_a := Real.sqrt (a.1^2 + a.2^2)
  let magnitude_b := Real.sqrt (b.1^2 + b.2^2)
  let cos_theta := dot_product / (magnitude_a * magnitude_b)
  cos_theta = - Real.sqrt 2 / 10 →
  magnitude_a * cos_theta = - Real.sqrt 5 / 5 :=
by 
  sorry

end cos_theta_value_projection_value_l18_18230


namespace time_to_upload_file_l18_18303

-- Define the conditions
def file_size : ℕ := 160
def upload_speed : ℕ := 8

-- Define the question as a proof goal
theorem time_to_upload_file :
  file_size / upload_speed = 20 := 
sorry

end time_to_upload_file_l18_18303


namespace remainder_of_exp_l18_18705

theorem remainder_of_exp (x : ℝ) :
  (x + 1) ^ 2100 % (x^4 - x^2 + 1) = x^2 := 
sorry

end remainder_of_exp_l18_18705


namespace arithmetic_mean_bc_diff_l18_18595

variables (a b c μ : ℝ)

theorem arithmetic_mean_bc_diff 
  (h1 : (a + b) / 2 = μ + 5)
  (h2 : (a + c) / 2 = μ - 8)
  (h3 : μ = (a + b + c) / 3) :
  (b + c) / 2 = μ + 3 :=
sorry

end arithmetic_mean_bc_diff_l18_18595


namespace antonov_packs_remaining_l18_18894

theorem antonov_packs_remaining (total_candies : ℕ) (pack_size : ℕ) (packs_given : ℕ) (candies_remaining : ℕ) (packs_remaining : ℕ) :
  total_candies = 60 →
  pack_size = 20 →
  packs_given = 1 →
  candies_remaining = total_candies - pack_size * packs_given →
  packs_remaining = candies_remaining / pack_size →
  packs_remaining = 2 := by
  sorry

end antonov_packs_remaining_l18_18894


namespace fraction_division_l18_18552

theorem fraction_division (a b : ℚ) (ha : a = 3) (hb : b = 4) :
  (1 / b) / (1 / a) = 3 / 4 :=
by 
  -- Solve the proof
  sorry

end fraction_division_l18_18552


namespace intersection_A_B_l18_18974

def A : Set ℤ := {-2, 0, 2}
def B : Set ℤ := {x | x^2 - x - 2 = 0}

theorem intersection_A_B : A ∩ B = {2} := by
  -- Proof to be filled
  sorry

end intersection_A_B_l18_18974


namespace partA_l18_18084

theorem partA (a b : ℝ) : (a - b) ^ 2 ≥ 0 → (a^2 + b^2) / 2 ≥ a * b := 
by
  intro h
  sorry

end partA_l18_18084


namespace binomial_square_formula_l18_18840

theorem binomial_square_formula (a b : ℝ) :
  let e1 := (4 * a + b) * (4 * a - 2 * b)
  let e2 := (a - 2 * b) * (2 * b - a)
  let e3 := (2 * a - b) * (-2 * a + b)
  let e4 := (a - b) * (a + b)
  (e4 = a^2 - b^2) :=
by
  sorry

end binomial_square_formula_l18_18840


namespace isosceles_right_triangle_area_l18_18010

theorem isosceles_right_triangle_area (h : ℝ) (h_eq : h = 6 * Real.sqrt 2) : 
  ∃ A : ℝ, A = 18 := by 
  sorry

end isosceles_right_triangle_area_l18_18010


namespace combined_weight_of_candles_l18_18248

theorem combined_weight_of_candles 
  (beeswax_weight_per_candle : ℕ)
  (coconut_oil_weight_per_candle : ℕ)
  (total_candles : ℕ)
  (candles_made : ℕ) 
  (total_weight: ℕ) 
  : 
  beeswax_weight_per_candle = 8 → 
  coconut_oil_weight_per_candle = 1 → 
  total_candles = 10 → 
  candles_made = total_candles - 3 →
  total_weight = candles_made * (beeswax_weight_per_candle + coconut_oil_weight_per_candle) →
  total_weight = 63 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end combined_weight_of_candles_l18_18248


namespace boat_speed_in_still_water_l18_18725

theorem boat_speed_in_still_water (b : ℝ) (h : (36 / (b - 2)) - (36 / (b + 2)) = 1.5) : b = 10 :=
by
  sorry

end boat_speed_in_still_water_l18_18725


namespace perpendicular_vectors_l18_18878

noncomputable def a (k : ℝ) : ℝ × ℝ := (2 * k - 4, 3)
noncomputable def b (k : ℝ) : ℝ × ℝ := (-3, k)

theorem perpendicular_vectors (k : ℝ) (h : (2 * k - 4) * (-3) + 3 * k = 0) : k = 4 :=
sorry

end perpendicular_vectors_l18_18878


namespace length_of_train_l18_18116

-- Define the conditions
def bridge_length : ℕ := 200
def train_crossing_time : ℕ := 60
def train_speed : ℕ := 5

-- Define the total distance traveled by the train while crossing the bridge
def total_distance : ℕ := train_speed * train_crossing_time

-- The problem is to show the length of the train
theorem length_of_train :
  total_distance - bridge_length = 100 :=
by sorry

end length_of_train_l18_18116


namespace sugar_water_inequality_triangle_inequality_l18_18086

-- Condition for question (1)
variable (x y m : ℝ)
variable (hx : x > 0) (hy : y > 0) (hxy : x > y) (hm : m > 0)

-- Proof problem for question (1)
theorem sugar_water_inequality : y / x < (y + m) / (x + m) :=
sorry

-- Condition for question (2)
variable (a b c : ℝ)
variable (ha : a > 0) (hb : b > 0) (hc : c > 0)
variable (hab : b + c > a) (hac : a + c > b) (hbc : a + b > c)

-- Proof problem for question (2)
theorem triangle_inequality : 
  a / (b + c) + b / (a + c) + c / (a + b) < 2 :=
sorry

end sugar_water_inequality_triangle_inequality_l18_18086


namespace angle_B_possible_values_l18_18800

theorem angle_B_possible_values
  (a b : ℝ) (A B : ℝ)
  (h_a : a = 2)
  (h_b : b = 2 * Real.sqrt 3)
  (h_A : A = Real.pi / 6) 
  (h_A_range : (0 : ℝ) < A ∧ A < Real.pi) :
  B = Real.pi / 3 ∨ B = 2 * Real.pi / 3 :=
  sorry

end angle_B_possible_values_l18_18800


namespace matrix_exp_1000_l18_18461

-- Define the matrix as a constant
noncomputable def A : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![1, 0], ![2, 1]]

-- The property of matrix exponentiation
theorem matrix_exp_1000 :
  A^1000 = ![![1, 0], ![2000, 1]] :=
by
  sorry

end matrix_exp_1000_l18_18461


namespace log_difference_l18_18593

theorem log_difference {x y a : ℝ} (h : Real.log x - Real.log y = a) :
  Real.log ((x / 2)^3) - Real.log ((y / 2)^3) = 3 * a :=
by 
  sorry

end log_difference_l18_18593


namespace equal_probabilities_partitioned_nonpartitioned_conditions_for_equal_probabilities_l18_18624

variable (v1 v2 f1 f2 : ℝ)

theorem equal_probabilities_partitioned_nonpartitioned :
  (v1 * (v2 + f2) + v2 * (v1 + f1)) / (2 * (v1 + f1) * (v2 + f2)) =
  (v1 + v2) / ((v1 + f1) + (v2 + f2)) :=
by sorry

theorem conditions_for_equal_probabilities :
  (v1 * f2 = v2 * f1) ∨ (v1 + f1 = v2 + f2) :=
by sorry

end equal_probabilities_partitioned_nonpartitioned_conditions_for_equal_probabilities_l18_18624


namespace adult_tickets_sold_l18_18035

open Nat

theorem adult_tickets_sold (A C : ℕ) (h₁ : A + C = 522) (h₂ : 15 * A + 8 * C = 5086) :
  A = 130 :=
by
  sorry

end adult_tickets_sold_l18_18035


namespace problem_solution_l18_18119

theorem problem_solution
  (N1 N2 : ℤ)
  (h : ∀ x : ℝ, 50 * x - 42 ≠ 0 → x ≠ 2 → x ≠ 3 → 
    (50 * x - 42) / (x ^ 2 - 5 * x + 6) = N1 / (x - 2) + N2 / (x - 3)) : 
  N1 * N2 = -6264 :=
sorry

end problem_solution_l18_18119


namespace arithmetic_sequence_a10_gt_0_l18_18252

variable {α : Type*} [LinearOrderedField α]

-- Definitions of the conditions
def arithmetic_sequence (a : ℕ → α) := ∀ n1 n2, a n1 - a n2 = (n1 - n2) * (a 1 - a 0)
def a9_lt_0 (a : ℕ → α) := a 9 < 0
def a1_add_a18_gt_0 (a : ℕ → α) := a 1 + a 18 > 0

-- The proof statement
theorem arithmetic_sequence_a10_gt_0 
  (a : ℕ → α) 
  (h_arith : arithmetic_sequence a) 
  (h_a9 : a9_lt_0 a) 
  (h_a1_a18 : a1_add_a18_gt_0 a) : 
  a 10 > 0 := 
sorry

end arithmetic_sequence_a10_gt_0_l18_18252


namespace inequality_for_pos_reals_l18_18251

-- Definitions for positive real numbers
variables {x y : ℝ}
def is_pos_real (x : ℝ) : Prop := x > 0

-- Theorem statement
theorem inequality_for_pos_reals (hx : is_pos_real x) (hy : is_pos_real y) : 
  2 * (x^2 + y^2) ≥ (x + y)^2 :=
by
  sorry

end inequality_for_pos_reals_l18_18251


namespace largest_multiple_of_11_lt_neg150_l18_18952

theorem largest_multiple_of_11_lt_neg150 : ∃ (x : ℤ), (x % 11 = 0) ∧ (x < -150) ∧ (∀ y : ℤ, y % 11 = 0 → y < -150 → y ≤ x) ∧ x = -154 :=
by
  sorry

end largest_multiple_of_11_lt_neg150_l18_18952


namespace problem_l18_18583

theorem problem (x y : ℝ) (h1 : x + y = 5) (h2 : x * y = 3) :
  x + (x^2 / y) + (y^2 / x) + y = 95 / 3 := by
  sorry

end problem_l18_18583


namespace combination_sum_l18_18435

-- Definition of combination, also known as binomial coefficient
def combination (n k : ℕ) : ℕ :=
  Nat.choose n k

-- Theorem statement
theorem combination_sum :
  (combination 8 2) + (combination 8 3) = 84 :=
by
  sorry

end combination_sum_l18_18435


namespace proof_problem_l18_18099

-- Given conditions: 
variables (a b c d : ℝ)
axiom condition : (2 * a + b) / (b + 2 * c) = (c + 3 * d) / (4 * d + a)

-- Proof problem statement:
theorem proof_problem : (a = c ∨ 3 * a + 4 * b + 5 * c + 6 * d = 0 ∨ (a = c ∧ 3 * a + 4 * b + 5 * c + 6 * d = 0)) :=
by
  sorry

end proof_problem_l18_18099


namespace base9_to_decimal_unique_solution_l18_18106

theorem base9_to_decimal_unique_solution :
  ∃ m : ℕ, 1 * 9^4 + 6 * 9^3 + m * 9^2 + 2 * 9^1 + 7 = 11203 ∧ m = 3 :=
by
  sorry

end base9_to_decimal_unique_solution_l18_18106


namespace sqrt_fraction_sum_l18_18854

theorem sqrt_fraction_sum : 
    Real.sqrt ((1 / 25) + (1 / 36)) = (Real.sqrt 61) / 30 := 
by
  sorry

end sqrt_fraction_sum_l18_18854


namespace complement_union_eq_l18_18008

variable (U : Set ℝ := Set.univ)
variable (A : Set ℝ := {x | x < -1 ∨ (2 ≤ x ∧ x < 3)})
variable (B : Set ℝ := {x | -2 ≤ x ∧ x < 4})

theorem complement_union_eq : (U \ A) ∪ B = {x | x ≥ -2} := by
  sorry

end complement_union_eq_l18_18008


namespace train_passes_platform_in_39_2_seconds_l18_18082

def length_of_train : ℝ := 360
def speed_in_kmh : ℝ := 45
def length_of_platform : ℝ := 130

noncomputable def speed_in_mps : ℝ := speed_in_kmh * 1000 / 3600
noncomputable def total_distance : ℝ := length_of_train + length_of_platform
noncomputable def time_to_pass_platform : ℝ := total_distance / speed_in_mps

theorem train_passes_platform_in_39_2_seconds :
  time_to_pass_platform = 39.2 := by
  sorry

end train_passes_platform_in_39_2_seconds_l18_18082


namespace evaluate_expression_l18_18416

theorem evaluate_expression (x : ℝ) (h : x = Real.sqrt 3) : 
  ( (x^2 - 2*x + 1) / (x^2 - x) / (x - 1) ) = Real.sqrt 3 / 3 :=
by
  sorry

end evaluate_expression_l18_18416


namespace fraction_denominator_l18_18350

theorem fraction_denominator (x y Z : ℚ) (h : x / y = 7 / 3) (h2 : (x + y) / Z = 2.5) :
    Z = (4 * y) / 3 :=
by sorry

end fraction_denominator_l18_18350


namespace mary_mortgage_payment_l18_18498

theorem mary_mortgage_payment :
  let a1 := 400
  let r := 2
  let n := 11
  let sum_geom_series (a1 r : ℕ) (n : ℕ) : ℕ := (a1 * (1 - r^n)) / (1 - r)
  sum_geom_series a1 r n = 819400 :=
by
  let a1 := 400
  let r := 2
  let n := 11
  let sum_geom_series (a1 r : ℕ) (n : ℕ) : ℕ := (a1 * (1 - r^n)) / (1 - r)
  have h : sum_geom_series a1 r n = 819400 := sorry
  exact h

end mary_mortgage_payment_l18_18498


namespace least_possible_area_l18_18518

def perimeter (x y : ℕ) : ℕ := 2 * (x + y)

def area (x y : ℕ) : ℕ := x * y

theorem least_possible_area :
  ∃ (x y : ℕ), 
    perimeter x y = 120 ∧ 
    (∀ x y, perimeter x y = 120 → area x y ≥ 59) ∧ 
    area x y = 59 := 
sorry

end least_possible_area_l18_18518


namespace distribution_of_K_l18_18078

theorem distribution_of_K (x y z : ℕ) 
  (h_total : x + y + z = 370)
  (h_diff : y + z - x = 50)
  (h_prop : x * z = y^2) :
  x = 160 ∧ y = 120 ∧ z = 90 := by
  sorry

end distribution_of_K_l18_18078


namespace original_price_color_tv_l18_18382

theorem original_price_color_tv (x : ℝ) : 
  1.4 * x * 0.8 - x = 270 → x = 2250 :=
by
  intro h
  simp at h
  sorry

end original_price_color_tv_l18_18382


namespace cos_C_value_triangle_perimeter_l18_18870

variables (A B C a b c : ℝ)
variables (cos_B : ℝ) (A_eq_2B : A = 2 * B) (cos_B_val : cos_B = 2 / 3)
variables (dot_product_88 : a * b * (Real.cos C) = 88)

theorem cos_C_value (A B : ℝ) (a b : ℝ) (cos_B : ℝ) (cos_C : ℝ) (dot_product_88 : a * b * cos_C = 88) :
  A = 2 * B →
  cos_B = 2 / 3 →
  cos_C = 22 / 27 :=
sorry

theorem triangle_perimeter (A B C a b c : ℝ) (cos_B : ℝ)
  (A_eq_2B : A = 2 * B) (cos_B_val : cos_B = 2 / 3) (dot_product_88 : a * b * (Real.cos C) = 88)
  (a_val : a = 12) (b_val : b = 9) (c_val : c = 7) :
  a + b + c = 28 :=
sorry

end cos_C_value_triangle_perimeter_l18_18870


namespace fresh_grapes_weight_l18_18195

theorem fresh_grapes_weight (F D : ℝ) (h1 : D = 0.625) (h2 : 0.10 * F = 0.80 * D) : F = 5 := by
  -- Using premises h1 and h2, we aim to prove that F = 5
  sorry

end fresh_grapes_weight_l18_18195


namespace tan_x_tan_y_relation_l18_18957

/-- If 
  (sin x / cos y) + (sin y / cos x) = 2 
  and 
  (cos x / sin y) + (cos y / sin x) = 3, 
  then 
  (tan x / tan y) + (tan y / tan x) = 16 / 3.
 -/
theorem tan_x_tan_y_relation (x y : ℝ)
  (h1 : (Real.sin x / Real.cos y) + (Real.sin y / Real.cos x) = 2)
  (h2 : (Real.cos x / Real.sin y) + (Real.cos y / Real.sin x) = 3) :
  (Real.tan x / Real.tan y) + (Real.tan y / Real.tan x) = 16 / 3 :=
sorry

end tan_x_tan_y_relation_l18_18957


namespace arrow_in_48th_position_l18_18668

def arrow_sequence : List (String) := ["→", "↑", "↓", "←", "↘"]

theorem arrow_in_48th_position :
  arrow_sequence.get? ((48 % 5) - 1) = some "↓" :=
by
  norm_num
  sorry

end arrow_in_48th_position_l18_18668


namespace value_does_not_appear_l18_18065

theorem value_does_not_appear : 
  let f : ℕ → ℕ := fun x => 5*x^5 + 4*x^4 + 3*x^3 + 2*x^2 + x + 1
  let x := 2
  let values := [14, 31, 64, 129, 259]
  127 ∉ values :=
by
  sorry

end value_does_not_appear_l18_18065


namespace range_of_expression_l18_18747

theorem range_of_expression (x : ℝ) (h1 : 1 - 3 * x ≥ 0) (h2 : 2 * x ≠ 0) : x ≤ 1 / 3 ∧ x ≠ 0 := by
  sorry

end range_of_expression_l18_18747


namespace seating_profession_solution_l18_18516

inductive Profession
| architect
| barista
| veterinarian
| guitarist

open Profession

inductive Friend
| Andrey
| Boris
| Vyacheslav
| Gennady

open Friend

structure SeatArrangement :=
(seat1: Friend)
(seat2: Friend)
(seat3: Friend)
(seat4: Friend)

structure ProfessionAssignment :=
(Andrey_profession: Profession)
(Boris_profession: Profession)
(Vyacheslav_profession: Profession)
(Gennady_profession: Profession)

noncomputable def correct_assignment (seats: SeatArrangement) : ProfessionAssignment :=
{ Andrey_profession := veterinarian,
  Boris_profession := architect,
  Vyacheslav_profession := guitarist,
  Gennady_profession := barista }

theorem seating_profession_solution
  (seats: SeatArrangement)
  (cond1 : ∀ f, f ∈ [seats.seat2, seats.seat3] ↔ ∃ p, p ∈ [architect, guitarist])
  (cond2 : seats.seat2 = Boris)
  (cond3 : (seats.seat3 = Vyacheslav ∧ seats.seat1 = Andrey) ∨ (seats.seat4 = Vyacheslav ∧ seats.seat2 = Andrey))
  (cond4 : (seats.seat1 = Andrey ∨ seats.seat2 = Andrey) ∧ Andrey ∉ [seats.seat1, seats.seat4])
  (cond5 : seats.seat1 ≠ Vyacheslav ∧ (seats.seat2 ≠ Gennady ∨ seats.seat3 ≠ Gennady)) :
  correct_assignment seats = 
  { Andrey_profession := veterinarian,
    Boris_profession := architect,
    Vyacheslav_profession := guitarist,
    Gennady_profession := barista } :=
sorry

end seating_profession_solution_l18_18516


namespace total_pens_bought_l18_18755

-- Define the problem conditions
def pens_given_to_friends : ℕ := 22
def pens_kept_for_herself : ℕ := 34

-- Theorem statement
theorem total_pens_bought : pens_given_to_friends + pens_kept_for_herself = 56 := by
  sorry

end total_pens_bought_l18_18755


namespace merchant_profit_l18_18481

theorem merchant_profit (C S : ℝ) (h: 20 * C = 15 * S) : 
  (S - C) / C * 100 = 33.33 := by
sorry

end merchant_profit_l18_18481


namespace terry_age_proof_l18_18102

theorem terry_age_proof
  (nora_age : ℕ)
  (h1 : nora_age = 10)
  (terry_age_in_10_years : ℕ)
  (h2 : terry_age_in_10_years = 4 * nora_age)
  (nora_age_in_5_years : ℕ)
  (h3 : nora_age_in_5_years = nora_age + 5)
  (sam_age_in_5_years : ℕ)
  (h4 : sam_age_in_5_years = 2 * nora_age_in_5_years)
  (sam_current_age : ℕ)
  (h5 : sam_current_age = sam_age_in_5_years - 5)
  (terry_current_age : ℕ)
  (h6 : sam_current_age = terry_current_age + 6) :
  terry_current_age = 19 :=
by
  sorry

end terry_age_proof_l18_18102


namespace prob_same_color_l18_18572

-- Define the given conditions
def total_pieces : ℕ := 15
def black_pieces : ℕ := 6
def white_pieces : ℕ := 9
def prob_two_black : ℚ := 1/7
def prob_two_white : ℚ := 12/35

-- Define the statement to be proved
theorem prob_same_color : prob_two_black + prob_two_white = 17 / 35 := by
  sorry

end prob_same_color_l18_18572


namespace measure_of_B_l18_18423

theorem measure_of_B (a b : ℝ) (A B : ℝ) (angleA_nonneg : 0 < A ∧ A < 180) (angleB_nonneg : 0 < B ∧ B < 180)
    (a_eq : a = 1) (b_eq : b = Real.sqrt 3) (A_eq : A = 30) :
    B = 60 :=
by
  sorry

end measure_of_B_l18_18423


namespace find_p_series_l18_18016

theorem find_p_series (p : ℝ) (h : 5 + (5 + p) / 5 + (5 + 2 * p) / 5^2 + (5 + 3 * p) / 5^3 + ∑' (n : ℕ), (5 + (n + 1) * p) / 5^(n + 1) = 10) : p = 16 :=
sorry

end find_p_series_l18_18016


namespace units_digit_of_2_pow_20_minus_1_l18_18089

theorem units_digit_of_2_pow_20_minus_1 : (2^20 - 1) % 10 = 5 := 
  sorry

end units_digit_of_2_pow_20_minus_1_l18_18089


namespace continuity_at_x_0_l18_18760

def f (x : ℝ) := -2 * x^2 + 9
def x_0 : ℝ := 4

theorem continuity_at_x_0 :
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, |x - x_0| < δ → |f x - f x_0| < ε :=
by
  sorry

end continuity_at_x_0_l18_18760


namespace three_digit_numbers_divisible_by_5_l18_18798

theorem three_digit_numbers_divisible_by_5 : ∃ n : ℕ, n = 181 ∧ ∀ x : ℕ, (100 ≤ x ∧ x ≤ 999 ∧ x % 5 = 0) → ∃ k : ℕ, x = 100 + k * 5 ∧ k < n := sorry

end three_digit_numbers_divisible_by_5_l18_18798


namespace find_c_l18_18397

-- Define the necessary conditions for the circle equation and the radius
variable (c : ℝ)

-- The given conditions
def circle_eq := ∀ (x y : ℝ), x^2 + 8*x + y^2 - 6*y + c = 0
def radius_five := (∀ (h k r : ℝ), r = 5 → ∃ (x y : ℝ), (x - h)^2 + (y - k)^2 = r^2)

theorem find_c (h k r : ℝ) (r_eq : r = 5) : c = 0 :=
by {
  sorry
}

end find_c_l18_18397


namespace oranges_count_l18_18688

def oranges_per_box : ℝ := 10
def boxes_per_day : ℝ := 2650
def total_oranges (x y : ℝ) : ℝ := x * y

theorem oranges_count :
  total_oranges oranges_per_box boxes_per_day = 26500 := 
  by sorry

end oranges_count_l18_18688


namespace reciprocal_lcm_of_24_and_208_l18_18727

theorem reciprocal_lcm_of_24_and_208 :
  (1 / (Nat.lcm 24 208)) = (1 / 312) :=
by
  sorry

end reciprocal_lcm_of_24_and_208_l18_18727


namespace selling_price_l18_18061

theorem selling_price (profit_percent : ℝ) (cost_price : ℝ) (h_profit : profit_percent = 5) (h_cp : cost_price = 2400) :
  let profit := (profit_percent / 100) * cost_price 
  let selling_price := cost_price + profit
  selling_price = 2520 :=
by
  sorry

end selling_price_l18_18061


namespace find_CD_squared_l18_18493

noncomputable def first_circle (x y : ℝ) : Prop := (x - 5)^2 + y^2 = 25
noncomputable def second_circle (x y : ℝ) : Prop := x^2 + (y - 5)^2 = 25

theorem find_CD_squared : ∃ C D : ℝ × ℝ, 
  (first_circle C.1 C.2 ∧ second_circle C.1 C.2) ∧ 
  (first_circle D.1 D.2 ∧ second_circle D.1 D.2) ∧ 
  (C ≠ D) ∧ 
  ((D.1 - C.1)^2 + (D.2 - C.2)^2 = 50) :=
by
  sorry

end find_CD_squared_l18_18493


namespace a_beats_b_by_4_rounds_l18_18199

variable (T_a T_b : ℝ)
variable (race_duration : ℝ) -- duration of the 4-round race in minutes
variable (time_difference : ℝ) -- Time that a beats b by in the 4-round race

open Real

-- Given conditions
def conditions :=
  (T_a = 7.5) ∧                             -- a's time to complete one round
  (race_duration = T_a * 4 + 10) ∧          -- a beats b by 10 minutes in a 4-round race
  (time_difference = T_b - T_a)             -- The time difference per round is T_b - T_a

-- Mathematical proof statement
theorem a_beats_b_by_4_rounds
  (h : conditions T_a T_b race_duration time_difference) :
  10 / time_difference = 4 := by
  sorry

end a_beats_b_by_4_rounds_l18_18199


namespace part1_l18_18620

noncomputable def f (x : ℝ) : ℝ := x^2 - 1
noncomputable def g (x a : ℝ) : ℝ := a * |x - 1|

theorem part1 (a : ℝ) : (∀ x : ℝ, f x ≥ g x a) ↔ a ≤ -2 := by
  sorry

end part1_l18_18620


namespace hours_buses_leave_each_day_l18_18772

theorem hours_buses_leave_each_day
  (num_buses : ℕ)
  (num_days : ℕ)
  (buses_per_half_hour : ℕ)
  (h1 : num_buses = 120)
  (h2 : num_days = 5)
  (h3 : buses_per_half_hour = 2) :
  (num_buses / num_days) / buses_per_half_hour = 12 :=
by
  sorry

end hours_buses_leave_each_day_l18_18772


namespace incorrect_statement_g2_l18_18469

def g (x : ℚ) : ℚ := (2 * x + 3) / (x - 2)

theorem incorrect_statement_g2 : g 2 ≠ 0 := by
  sorry

end incorrect_statement_g2_l18_18469


namespace remaining_amount_division_l18_18970

-- Definitions
def total_amount : ℕ := 2100
def number_of_participants : ℕ := 8
def amount_already_raised : ℕ := 150

-- Proof problem statement
theorem remaining_amount_division :
  (total_amount - amount_already_raised) / (number_of_participants - 1) = 279 :=
by
  sorry

end remaining_amount_division_l18_18970


namespace quotient_change_l18_18002

variables {a b : ℝ} (h : a / b = 0.78)

theorem quotient_change (a b : ℝ) (h : a / b = 0.78) : (10 * a) / (b / 10) = 78 :=
by
  sorry

end quotient_change_l18_18002


namespace sugar_amount_l18_18510

-- Definitions based on conditions
variables (S F B C : ℝ) -- S = amount of sugar, F = amount of flour, B = amount of baking soda, C = amount of chocolate chips

-- Conditions
def ratio_sugar_flour (S F : ℝ) : Prop := S / F = 5 / 4
def ratio_flour_baking_soda (F B : ℝ) : Prop := F / B = 10 / 1
def ratio_baking_soda_chocolate_chips (B C : ℝ) : Prop := B / C = 3 / 2
def new_ratio_flour_baking_soda_chocolate_chips (F B C : ℝ) : Prop :=
  F / (B + 120) = 16 / 3 ∧ F / (C + 50) = 16 / 2

-- Prove that the current amount of sugar is 1714 pounds
theorem sugar_amount (S F B C : ℝ) (h1 : ratio_sugar_flour S F)
  (h2 : ratio_flour_baking_soda F B) (h3 : ratio_baking_soda_chocolate_chips B C)
  (h4 : new_ratio_flour_baking_soda_chocolate_chips F B C) : 
  S = 1714 :=
sorry

end sugar_amount_l18_18510


namespace total_squares_after_erasing_lines_l18_18268

theorem total_squares_after_erasing_lines :
  ∀ (a : ℕ) (b : ℕ) (c : ℕ) (d : ℕ), a = 16 → b = 4 → c = 9 → d = 2 → 
  a - b + c - d + (a / 16) = 22 := 
by
  intro a b c d h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end total_squares_after_erasing_lines_l18_18268


namespace problem1_l18_18137

theorem problem1 : 2 * (-5) + 2^3 - 3 + (1/2 : ℚ) = -15 / 2 := 
by
  sorry

end problem1_l18_18137


namespace total_feathers_needed_l18_18448

theorem total_feathers_needed
  (animals_first_group : ℕ := 934)
  (feathers_first_group : ℕ := 7)
  (animals_second_group : ℕ := 425)
  (colored_feathers_second_group : ℕ := 7)
  (golden_feathers_second_group : ℕ := 5)
  (animals_third_group : ℕ := 289)
  (colored_feathers_third_group : ℕ := 4)
  (golden_feathers_third_group : ℕ := 10) :
  (animals_first_group * feathers_first_group) +
  (animals_second_group * (colored_feathers_second_group + golden_feathers_second_group)) +
  (animals_third_group * (colored_feathers_third_group + golden_feathers_third_group)) = 15684 := by
  sorry

end total_feathers_needed_l18_18448


namespace coordinates_of_B_l18_18868

-- Definitions of the points and vectors are given as conditions.
def A : ℝ × ℝ := (-1, -1)
def a : ℝ × ℝ := (2, 3)

-- Statement of the problem translated to Lean
theorem coordinates_of_B (B : ℝ × ℝ) (h : B = (5, 8)) :
  (B.1 + 1, B.2 + 1) = (3 * a.1, 3 * a.2) :=
sorry

end coordinates_of_B_l18_18868


namespace average_salary_rest_l18_18961

theorem average_salary_rest (number_of_workers : ℕ) 
                            (avg_salary_all : ℝ) 
                            (number_of_technicians : ℕ) 
                            (avg_salary_technicians : ℝ) 
                            (rest_workers : ℕ) 
                            (total_salary_all : ℝ) 
                            (total_salary_technicians : ℝ) 
                            (total_salary_rest : ℝ) 
                            (avg_salary_rest : ℝ) 
                            (h1 : number_of_workers = 28)
                            (h2 : avg_salary_all = 8000)
                            (h3 : number_of_technicians = 7)
                            (h4 : avg_salary_technicians = 14000)
                            (h5 : rest_workers = number_of_workers - number_of_technicians)
                            (h6 : total_salary_all = number_of_workers * avg_salary_all)
                            (h7 : total_salary_technicians = number_of_technicians * avg_salary_technicians)
                            (h8 : total_salary_rest = total_salary_all - total_salary_technicians)
                            (h9 : avg_salary_rest = total_salary_rest / rest_workers) :
  avg_salary_rest = 6000 :=
by {
  -- the proof would go here
  sorry
}

end average_salary_rest_l18_18961


namespace find_value_of_expression_l18_18357

theorem find_value_of_expression (x : ℝ) (h : 5 * x^2 + 4 = 3 * x + 9) : (10 * x - 3)^2 = 109 := 
sorry

end find_value_of_expression_l18_18357


namespace wheel_moves_in_one_hour_l18_18920

theorem wheel_moves_in_one_hour
  (rotations_per_minute : ℕ)
  (distance_per_rotation_cm : ℕ)
  (minutes_in_hour : ℕ) :
  rotations_per_minute = 20 →
  distance_per_rotation_cm = 35 →
  minutes_in_hour = 60 →
  let distance_per_rotation_m : ℚ := distance_per_rotation_cm / 100
  let total_rotations_per_hour : ℕ := rotations_per_minute * minutes_in_hour
  let total_distance_in_hour : ℚ := distance_per_rotation_m * total_rotations_per_hour
  total_distance_in_hour = 420 := by
  intros
  sorry

end wheel_moves_in_one_hour_l18_18920


namespace no_such_abc_l18_18990

theorem no_such_abc :
  ¬ ∃ (a b c : ℕ+),
    (∃ k1 : ℕ, a ^ 2 * b * c + 2 = k1 ^ 2) ∧
    (∃ k2 : ℕ, b ^ 2 * c * a + 2 = k2 ^ 2) ∧
    (∃ k3 : ℕ, c ^ 2 * a * b + 2 = k3 ^ 2) := 
sorry

end no_such_abc_l18_18990


namespace sheep_ratio_l18_18130

theorem sheep_ratio (S : ℕ) (h1 : 400 - S = 2 * 150) :
  S / 400 = 1 / 4 :=
by
  sorry

end sheep_ratio_l18_18130


namespace neg_p_is_correct_l18_18041

def is_positive_integer (x : ℕ) : Prop := x > 0

def proposition_p (x : ℕ) : Prop := (1 / 2 : ℝ) ^ x ≤ 1 / 2

def negation_of_p : Prop := ∃ x : ℕ, is_positive_integer x ∧ ¬ proposition_p x

theorem neg_p_is_correct : negation_of_p :=
sorry

end neg_p_is_correct_l18_18041


namespace inequality_f_lt_g_range_of_a_l18_18679

def f (x : ℝ) : ℝ := |x - 4|
def g (x : ℝ) : ℝ := |2 * x + 1|

theorem inequality_f_lt_g :
  ∀ x : ℝ, f x = |x - 4| ∧ g x = |2 * x + 1| →
  (f x < g x ↔ (x < -5 ∨ x > 1)) :=
by
   sorry

theorem range_of_a :
  ∀ x a : ℝ, f x = |x - 4| ∧ g x = |2 * x + 1| →
  (2 * f x + g x > a * x) →
  (-4 ≤ a ∧ a < 9/4) :=
by
   sorry

end inequality_f_lt_g_range_of_a_l18_18679


namespace evaluate_expression_l18_18186

-- Given conditions
def a : ℕ := 3
def b : ℕ := 2

-- Proof problem statement
theorem evaluate_expression : (1 / 3 : ℝ) ^ (b - a) = 3 := sorry

end evaluate_expression_l18_18186


namespace min_value_ineq_inequality_proof_l18_18528

variable (a b x1 x2 : ℝ)
variable (ha_pos : 0 < a) (hb_pos : 0 < b) (hx1_pos : 0 < x1) (hx2_pos : 0 < x2) (hab_sum : a + b = 1)

-- First problem: Prove that the minimum value of the given expression is 6.
theorem min_value_ineq : (x1 / a) + (x2 / b) + (2 / (x1 * x2)) ≥ 6 := by
  sorry

-- Second problem: Prove the given inequality.
theorem inequality_proof : (a * x1 + b * x2) * (a * x2 + b * x1) ≥ x1 * x2 := by
  sorry

end min_value_ineq_inequality_proof_l18_18528


namespace value_of_a_l18_18722

theorem value_of_a (a b c : ℤ) (h1 : a < b) (h2 : b < c) (h3 : (a + b + c) / 3 = 4 * b) (h4 : c / b = 11) : a = 0 :=
by
  sorry

end value_of_a_l18_18722


namespace power_sums_l18_18360

-- Definitions as per the given conditions
variables (m n a b : ℕ)
variables (hm : 0 < m) (hn : 0 < n)
variables (ha : 2^m = a) (hb : 2^n = b)

-- The theorem statement
theorem power_sums (hmn : 0 < m + n) : 2^(m + n) = a * b :=
by
  sorry

end power_sums_l18_18360


namespace david_more_pushups_than_zachary_l18_18449

def zacharyPushUps : ℕ := 59
def davidPushUps : ℕ := 78

theorem david_more_pushups_than_zachary :
  davidPushUps - zacharyPushUps = 19 :=
by
  sorry

end david_more_pushups_than_zachary_l18_18449


namespace initial_integers_is_three_l18_18361

def num_initial_integers (n m : Int) : Prop :=
  3 * n + m = 17 ∧ 2 * m + n = 23

theorem initial_integers_is_three {n m : Int} (h : num_initial_integers n m) : n = 3 :=
by
  sorry

end initial_integers_is_three_l18_18361


namespace kangaroo_fraction_sum_l18_18204

theorem kangaroo_fraction_sum (G P : ℕ) (hG : 1 ≤ G) (hP : 1 ≤ P) (hTotal : G + P = 2016) : 
  (G * (P / G) + P * (G / P) = 2016) :=
by
  sorry

end kangaroo_fraction_sum_l18_18204


namespace product_of_areas_eq_square_of_volume_l18_18514

theorem product_of_areas_eq_square_of_volume 
(x y z d : ℝ) 
(h1 : d^2 = x^2 + y^2 + z^2) :
  (x * y) * (y * z) * (z * x) = (x * y * z) ^ 2 :=
by sorry

end product_of_areas_eq_square_of_volume_l18_18514


namespace base6_divisible_by_13_l18_18208

theorem base6_divisible_by_13 (d : ℕ) (h : d < 6) : 13 ∣ (435 + 42 * d) ↔ d = 5 := 
by
  -- Proof implementation will go here, but is currently omitted
  sorry

end base6_divisible_by_13_l18_18208


namespace number_of_students_l18_18605

theorem number_of_students (avg_age_students : ℕ) (teacher_age : ℕ) (new_avg_age : ℕ) (n : ℕ) (T : ℕ) 
    (h1 : avg_age_students = 10) (h2 : teacher_age = 26) (h3 : new_avg_age = 11)
    (h4 : T = n * avg_age_students) 
    (h5 : (T + teacher_age) / (n + 1) = new_avg_age) : n = 15 :=
by
  -- Proof should go here
  sorry

end number_of_students_l18_18605


namespace brenda_cakes_l18_18153

-- Definitions based on the given conditions
def cakes_per_day : ℕ := 20
def days : ℕ := 9
def total_cakes_baked : ℕ := cakes_per_day * days
def cakes_sold : ℕ := total_cakes_baked / 2
def cakes_left : ℕ := total_cakes_baked - cakes_sold

-- Formulate the theorem
theorem brenda_cakes : cakes_left = 90 :=
by {
  -- To skip the proof steps
  sorry
}

end brenda_cakes_l18_18153


namespace find_m_in_arith_seq_l18_18457

noncomputable def arith_seq (a : ℕ → ℝ) (d : ℝ) : Prop :=
∀ n : ℕ, a (n + 1) = a n + d

theorem find_m_in_arith_seq (a : ℕ → ℝ) (d : ℝ) (h_d : d ≠ 0) 
  (h_seq : arith_seq a d) 
  (h_sum : a 3 + a 6 + a 10 + a 13 = 32) 
  (h_am : ∃ m, a m = 8) : 
  ∃ m, m = 8 := 
sorry

end find_m_in_arith_seq_l18_18457


namespace problem_statement_l18_18843

def f (x : ℝ) : ℝ := x^2
def g (x : ℝ) : ℝ := 3 * x^2 + 4

theorem problem_statement : f (g (-3)) = 961 := by
  sorry

end problem_statement_l18_18843


namespace system_solution_l18_18148

theorem system_solution (x b y : ℝ) (h1 : 4 * x + 2 * y = b) (h2 : 3 * x + 4 * y = 3 * b) (h3 : x = 3) :
  b = -1 :=
by
  -- proof to be filled in
  sorry

end system_solution_l18_18148


namespace mnpq_product_l18_18009

noncomputable def prove_mnpq_product (a b x y : ℝ) : Prop :=
  ∃ (m n p q : ℤ), (a^m * x - a^n) * (a^p * y - a^q) = a^3 * b^4 ∧
                    m * n * p * q = 4

theorem mnpq_product (a b x y : ℝ) (h : a^7 * x * y - a^6 * y - a^5 * x = a^3 * (b^4 - 1)) :
  prove_mnpq_product a b x y :=
sorry

end mnpq_product_l18_18009


namespace fraction_identity_l18_18156

variables {a b : ℝ}

theorem fraction_identity (h : a^2 + b^2 ≠ 0) :
  (a^4 + b^4) / (a^2 + b^2) = a^2 + b^2 - (2 * a^2 * b^2) / (a^2 + b^2) :=
by sorry

end fraction_identity_l18_18156


namespace ratio_of_inverse_l18_18054

theorem ratio_of_inverse (a b c d : ℝ) (h : ∀ x, (3 * (a * x + b) / (c * x + d) - 2) / ((a * x + b) / (c * x + d) + 4) = x) : 
  a / c = -4 :=
sorry

end ratio_of_inverse_l18_18054


namespace stickers_per_student_l18_18478

theorem stickers_per_student 
  (gold_stickers : ℕ) 
  (silver_stickers : ℕ) 
  (bronze_stickers : ℕ) 
  (students : ℕ)
  (h1 : gold_stickers = 50)
  (h2 : silver_stickers = 2 * gold_stickers)
  (h3 : bronze_stickers = silver_stickers - 20)
  (h4 : students = 5) : 
  (gold_stickers + silver_stickers + bronze_stickers) / students = 46 :=
by
  sorry

end stickers_per_student_l18_18478


namespace find_g_value_l18_18565

noncomputable def g (x : ℝ) (a b c : ℝ) : ℝ := a * x^6 + b * x^4 + c * x^2 + 7

theorem find_g_value (a b c : ℝ) (h1 : g (-4) a b c = 13) : g 4 a b c = 13 := by
  sorry

end find_g_value_l18_18565


namespace graph_properties_l18_18031

theorem graph_properties (k b : ℝ) (hk : k ≠ 0) (hb : b ≠ 0) (positive_kb : k * b > 0) :
  (∃ (f g : ℝ → ℝ),
    (∀ x, f x = k * x + b) ∧
    (∀ x (hx : x ≠ 0), g x = k * b / x) ∧
    -- Under the given conditions, the graphs must match option (B)
    (True)) := sorry

end graph_properties_l18_18031


namespace sum_base8_l18_18439

theorem sum_base8 (a b c : ℕ) (h₁ : a = 7*8^2 + 7*8 + 7)
                           (h₂ : b = 7*8 + 7)
                           (h₃ : c = 7) :
  a + b + c = 1*8^3 + 1*8^2 + 0*8 + 5 :=
by
  sorry

end sum_base8_l18_18439


namespace sector_perimeter_ratio_l18_18848

theorem sector_perimeter_ratio (α : ℝ) (r R : ℝ) 
  (h1 : α > 0) 
  (h2 : r > 0) 
  (h3 : R > 0) 
  (h4 : (1/2) * α * r^2 / ((1/2) * α * R^2) = 1/4) :
  (2 * r + α * r) / (2 * R + α * R) = 1 / 2 := 
sorry

end sector_perimeter_ratio_l18_18848


namespace range_of_k_l18_18967

noncomputable def f (x : ℝ) : ℝ := x - Real.sin x

theorem range_of_k (k : ℝ) :
  (∀ x : ℝ, f (-x^2 + 3 * x) + f (x - 2 * k) ≤ 0) ↔ k ≥ 2 :=
by
  sorry

end range_of_k_l18_18967


namespace range_of_m_l18_18096

open Set

theorem range_of_m (m : ℝ) :
  (∃ f : ℤ → Prop, (∀ x, f x ↔ x + 5 > 0 ∧ x - m ≤ 1) ∧ (∃ a b c : ℤ, f a ∧ f b ∧ f c))
  → (-3 ≤ m ∧ m < -2) := 
sorry

end range_of_m_l18_18096


namespace book_chapters_not_determinable_l18_18387

variable (pages_initially pages_later pages_total total_pages book_chapters : ℕ)

def problem_statement : Prop :=
  pages_initially = 37 ∧ pages_later = 25 ∧ pages_total = 62 ∧ total_pages = 95 ∧ book_chapters = 0

theorem book_chapters_not_determinable (h: problem_statement pages_initially pages_later pages_total total_pages book_chapters) :
  book_chapters = 0 :=
by
  sorry

end book_chapters_not_determinable_l18_18387


namespace find_a20_l18_18374

variable (a : ℕ → ℝ)
variable (d : ℝ)

axiom a_arithmetic : ∀ n, a (n + 1) = a 1 + n * d
axiom a1_a3_a5_eq_105 : a 1 + a 3 + a 5 = 105
axiom a2_a4_a6_eq_99 : a 2 + a 4 + a 6 = 99

theorem find_a20 : a 20 = 1 :=
by sorry

end find_a20_l18_18374


namespace arithmetic_sequence_sum_l18_18389

-- Definitions for the conditions
def a := 70
def d := 3
def n := 10
def l := 97

-- Sum of the arithmetic series
def S := (n / 2) * (a + l)

-- Final calculation
theorem arithmetic_sequence_sum :
  3 * (70 + 73 + 76 + 79 + 82 + 85 + 88 + 91 + 94 + 97) = 2505 :=
by
  -- Lean will calculate these interactively when proving.
  sorry

end arithmetic_sequence_sum_l18_18389


namespace simple_interest_correct_l18_18664

theorem simple_interest_correct (P R T : ℝ) (hP : P = 400) (hR : R = 12.5) (hT : T = 2) : 
  (P * R * T) / 100 = 50 :=
by
  sorry -- Proof to be provided

end simple_interest_correct_l18_18664


namespace sequence_formula_l18_18377

theorem sequence_formula (a : ℕ → ℚ) (h₁ : a 1 = 0)
  (h : ∀ n, a (n + 1) = 1 / (2 - a n)) :
  ∀ n, a n = (n - 1) / n :=
sorry

end sequence_formula_l18_18377


namespace shortest_path_octahedron_l18_18964

theorem shortest_path_octahedron 
  (edge_length : ℝ) (h : edge_length = 2) 
  (d : ℝ) : d = 2 :=
by
  sorry

end shortest_path_octahedron_l18_18964


namespace part1_solution_part2_solution_l18_18599

variable {x a : ℝ}

def f (x a : ℝ) : ℝ := abs (x - a)

theorem part1_solution (h1 : 0 ≤ x) (h2 : x ≤ 4) (h3 : f x a ≤ 2) : a = 2 :=
  sorry

theorem part2_solution (ha : 0 ≤ a) (hb : a ≤ 3) : (f (x + a) a + f (x - a) a ≥ f (a * x) a - a * f x a) :=
  sorry

end part1_solution_part2_solution_l18_18599


namespace complement_of_intersection_l18_18968

open Set

-- Define the universal set U
def U := @univ ℝ
-- Define the sets M and N
def M : Set ℝ := {x | x >= 2}
def N : Set ℝ := {x | 0 <= x ∧ x < 5}

-- Define M ∩ N
def M_inter_N := M ∩ N

-- Define the complement of M ∩ N with respect to U
def C_U (A : Set ℝ) := Aᶜ

theorem complement_of_intersection :
  C_U M_inter_N = {x : ℝ | x < 2 ∨ x ≥ 5} := 
by 
  sorry

end complement_of_intersection_l18_18968


namespace find_A_coords_find_AC_equation_l18_18473

theorem find_A_coords
  (B : ℝ × ℝ) (hB : B = (1, -2))
  (median_CM : ∀ x y, 2 * x - y + 1 = 0)
  (angle_bisector_BAC : ∀ x y, x + 7 * y - 12 = 0) :
  ∃ A : ℝ × ℝ, A = (-2, 2) :=
by
  sorry

theorem find_AC_equation
  (A B : ℝ × ℝ) (hA : A = (-2, 2)) (hB : B = (1, -2))
  (median_CM : ∀ x y, 2 * x - y + 1 = 0)
  (angle_bisector_BAC : ∀ x y, x + 7 * y - 12 = 0) :
  ∃ k b : ℝ, ∀ x y, y = k * x + b ↔ 3 * x - 4 * y + 14 = 0 :=
by
  sorry

end find_A_coords_find_AC_equation_l18_18473


namespace cost_per_unit_range_of_type_A_purchases_maximum_profit_l18_18641

-- Definitions of the problem conditions
def cost_type_A : ℕ := 15
def cost_type_B : ℕ := 20

def profit_type_A : ℕ := 3
def profit_type_B : ℕ := 4

def budget_min : ℕ := 2750
def budget_max : ℕ := 2850

def total_units : ℕ := 150
def profit_min : ℕ := 565

-- Main proof statements as Lean theorems
theorem cost_per_unit : 
  ∃ (x y : ℕ), 
    2 * x + 3 * y = 90 ∧ 
    3 * x + y = 65 ∧ 
    x = cost_type_A ∧ 
    y = cost_type_B := 
sorry

theorem range_of_type_A_purchases : 
  ∃ (a : ℕ), 
    30 ≤ a ∧ 
    a ≤ 50 ∧ 
    budget_min ≤ cost_type_A * a + cost_type_B * (total_units - a) ∧ 
    cost_type_A * a + cost_type_B * (total_units - a) ≤ budget_max := 
sorry

theorem maximum_profit : 
  ∃ (a : ℕ), 
    30 ≤ a ∧ 
    a ≤ 35 ∧ 
    profit_min ≤ profit_type_A * a + profit_type_B * (total_units - a) ∧ 
    ¬∃ (b : ℕ), 
      30 ≤ b ∧ 
      b ≤ 35 ∧ 
      b ≠ a ∧ 
      profit_type_A * b + profit_type_B * (total_units - b) > profit_type_A * a + profit_type_B * (total_units - a) :=
sorry

end cost_per_unit_range_of_type_A_purchases_maximum_profit_l18_18641


namespace right_triangle_satisfies_pythagorean_l18_18001

-- Definition of the sides of the triangle
def a : ℕ := 3
def b : ℕ := 4
def c : ℕ := 5

-- The theorem to prove
theorem right_triangle_satisfies_pythagorean :
  a^2 + b^2 = c^2 :=
by
  sorry

end right_triangle_satisfies_pythagorean_l18_18001


namespace min_value_of_function_l18_18375

theorem min_value_of_function (x : ℝ) (h : x > 0) : (∃ y : ℝ, y = x^2 + 3 * x + 1 ∧ ∀ z, z = x^2 + 3 * x + 1 → y ≤ z) → y = 5 :=
by
  sorry

end min_value_of_function_l18_18375


namespace incorrect_statement_C_l18_18564

noncomputable def f (x a b c : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

theorem incorrect_statement_C (a b c : ℝ) (x0 : ℝ) (h_local_min : ∀ y, f x0 a b c ≤ f y a b c) :
  ∃ z, z < x0 ∧ ¬ (f z a b c ≤ f (z + ε) a b c) := sorry

end incorrect_statement_C_l18_18564


namespace expectation_of_two_fair_dice_l18_18609

noncomputable def E_X : ℝ :=
  (2 * (1/36) + 3 * (2/36) + 4 * (3/36) + 5 * (4/36) + 6 * (5/36) + 7 * (6/36) + 
   8 * (5/36) + 9 * (4/36) + 10 * (3/36) + 11 * (2/36) + 12 * (1/36))

theorem expectation_of_two_fair_dice : E_X = 7 := by
  sorry

end expectation_of_two_fair_dice_l18_18609


namespace correct_truth_values_l18_18819

open Real

def proposition_p : Prop := ∀ (a : ℝ), 0 < a → a^2 ≠ 0

def converse_p : Prop := ∀ (a : ℝ), a^2 ≠ 0 → 0 < a

def inverse_p : Prop := ∀ (a : ℝ), ¬(0 < a) → a^2 = 0

def contrapositive_p : Prop := ∀ (a : ℝ), a^2 = 0 → ¬(0 < a)

def negation_p : Prop := ∃ (a : ℝ), 0 < a ∧ a^2 = 0

theorem correct_truth_values : 
  (converse_p = False) ∧ 
  (inverse_p = False) ∧ 
  (contrapositive_p = True) ∧ 
  (negation_p = False) := by
  sorry

end correct_truth_values_l18_18819


namespace apple_cost_calculation_l18_18978

theorem apple_cost_calculation
    (original_price : ℝ)
    (price_raise : ℝ)
    (amount_per_person : ℝ)
    (num_people : ℝ) :
  original_price = 1.6 →
  price_raise = 0.25 →
  amount_per_person = 2 →
  num_people = 4 →
  (num_people * amount_per_person * (original_price * (1 + price_raise))) = 16 :=
by
  -- insert the mathematical proof steps/cardinality here
  sorry

end apple_cost_calculation_l18_18978


namespace sum_of_digits_divisible_by_9_l18_18290

theorem sum_of_digits_divisible_by_9 (D E : ℕ) (hD : D < 10) (hE : E < 10) : 
  (D + E + 37) % 9 = 0 → ((D + E = 8) ∨ (D + E = 17)) →
  (8 + 17 = 25) := 
by
  intro h1 h2
  sorry

end sum_of_digits_divisible_by_9_l18_18290


namespace krishna_fraction_wins_l18_18766

theorem krishna_fraction_wins (matches_total : ℕ) (callum_points : ℕ) (points_per_win : ℕ) (callum_wins : ℕ) :
  matches_total = 8 → callum_points = 20 → points_per_win = 10 → callum_wins = callum_points / points_per_win →
  (matches_total - callum_wins) / matches_total = 3 / 4 :=
by
  intros h1 h2 h3 h4
  sorry

end krishna_fraction_wins_l18_18766


namespace initial_members_in_family_c_l18_18188

theorem initial_members_in_family_c 
  (a b d e f : ℕ)
  (ha : a = 7)
  (hb : b = 8)
  (hd : d = 13)
  (he : e = 6)
  (hf : f = 10)
  (average_after_moving : (a - 1) + (b - 1) + (d - 1) + (e - 1) + (f - 1) + (x : ℕ) - 1 = 48) :
  x = 10 := by
  sorry

end initial_members_in_family_c_l18_18188


namespace sum_of_numbers_in_ratio_l18_18834

theorem sum_of_numbers_in_ratio (x : ℝ) (h1 : 8 * x - 3 * x = 20) : 3 * x + 8 * x = 44 :=
by
  sorry

end sum_of_numbers_in_ratio_l18_18834


namespace function_identity_l18_18019

theorem function_identity (f : ℝ → ℝ)
  (h : ∀ x y : ℝ, f (x^2 + f y) = y + (f x)^2) :
  ∀ x : ℝ, f x = x :=
by
  sorry

end function_identity_l18_18019


namespace Beth_peas_count_l18_18462

-- Definitions based on conditions
def number_of_corn : ℕ := 10
def number_of_peas (number_of_corn : ℕ) : ℕ := 2 * number_of_corn + 15

-- Theorem that represents the proof problem
theorem Beth_peas_count : number_of_peas 10 = 35 :=
by
  sorry

end Beth_peas_count_l18_18462


namespace factor_expression_l18_18491

noncomputable def expression (x : ℝ) : ℝ := (15 * x^3 + 80 * x - 5) - (-4 * x^3 + 4 * x - 5)

theorem factor_expression (x : ℝ) : expression x = 19 * x * (x^2 + 4) := 
by 
  sorry

end factor_expression_l18_18491


namespace find_c_l18_18887

-- Definitions for the conditions
def line1 (x y : ℝ) : Prop := 4 * y + 2 * x + 6 = 0
def line2 (x y : ℝ) (c : ℝ) : Prop := 5 * y + c * x + 4 = 0
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

-- Main theorem
theorem find_c (c : ℝ) : 
  (∀ x y : ℝ, line1 x y → y = -1/2 * x - 3/2) ∧ 
  (∀ x y : ℝ, line2 x y c → y = -c/5 * x - 4/5) ∧ 
  perpendicular (-1/2) (-c/5) → 
  c = -10 := by
  sorry

end find_c_l18_18887


namespace melanie_total_amount_l18_18739

theorem melanie_total_amount :
  let g1 := 12
  let g2 := 15
  let g3 := 8
  let g4 := 10
  let g5 := 20
  g1 + g2 + g3 + g4 + g5 = 65 :=
by
  sorry

end melanie_total_amount_l18_18739


namespace range_of_a_l18_18216

theorem range_of_a (a : ℝ) : (∀ x : ℝ, |x| = ax + 1 → x < 0) → a > 1 :=
by
  sorry

end range_of_a_l18_18216


namespace smallest_percent_increase_l18_18893

-- Define the values of each question.
def value (n : ℕ) : ℕ :=
  match n with
  | 1  => 150
  | 2  => 300
  | 3  => 450
  | 4  => 600
  | 5  => 800
  | 6  => 1500
  | 7  => 3000
  | 8  => 6000
  | 9  => 12000
  | 10 => 24000
  | 11 => 48000
  | 12 => 96000
  | 13 => 192000
  | 14 => 384000
  | 15 => 768000
  | _ => 0

-- Define the percent increase between two values.
def percent_increase (v1 v2 : ℕ) : ℚ :=
  ((v2 - v1 : ℕ) : ℚ) / v1 * 100 

-- Prove that the smallest percent increase is between question 4 and 5.
theorem smallest_percent_increase :
  percent_increase (value 4) (value 5) = 33.33 := 
by
  sorry

end smallest_percent_increase_l18_18893


namespace number_a_eq_223_l18_18069

theorem number_a_eq_223 (A B : ℤ) (h1 : A - B = 144) (h2 : A = 3 * B - 14) : A = 223 :=
by
  sorry

end number_a_eq_223_l18_18069


namespace audrey_ratio_in_3_years_l18_18158

-- Define the ages and the conditions
def Heracles_age : ℕ := 10
def Audrey_age := Heracles_age + 7
def Audrey_age_in_3_years := Audrey_age + 3

-- Statement: Prove that the ratio of Audrey's age in 3 years to Heracles' current age is 2:1
theorem audrey_ratio_in_3_years : (Audrey_age_in_3_years / Heracles_age) = 2 := sorry

end audrey_ratio_in_3_years_l18_18158


namespace number_of_solutions_l18_18356

theorem number_of_solutions : ∃ n : ℕ, 1 < n ∧ 
  (∃ a b : ℕ, gcd a b = 1 ∧
  (∃ x y : ℕ, x^(a*n) + y^(b*n) = 2^2010)) ∧
  (∃ count : ℕ, count = 54) :=
sorry

end number_of_solutions_l18_18356


namespace parabola_point_l18_18311

theorem parabola_point (a b c : ℝ) (hA : 0.64 * a - 0.8 * b + c = 4.132)
  (hB : 1.44 * a + 1.2 * b + c = -1.948) (hC : 7.84 * a + 2.8 * b + c = -3.932) :
  0.5 * (1.8)^2 - 3.24 * 1.8 + 1.22 = -2.992 :=
by
  -- Proof is intentionally omitted
  sorry

end parabola_point_l18_18311


namespace expand_product_l18_18223

theorem expand_product (x : ℝ): (x + 4) * (x - 5 + 2) = x^2 + x - 12 :=
by 
  sorry

end expand_product_l18_18223


namespace maximum_value_l18_18816

theorem maximum_value (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) (h4 : x + y + z = 3) : 
  (x^2 - x * y + y^2) * (x^2 - x * z + z^2) * (y^2 - y * z + z^2) * (z^2 - z * x + x^2)  ≤ 1 :=
sorry

end maximum_value_l18_18816


namespace utility_bills_l18_18152

-- Definitions for the conditions
def four_hundred := 4 * 100
def five_fifty := 5 * 50
def seven_twenty := 7 * 20
def eight_ten := 8 * 10
def total := four_hundred + five_fifty + seven_twenty + eight_ten

-- Lean statement for the proof problem
theorem utility_bills : total = 870 :=
by
  -- inserting skip proof placeholder
  sorry

end utility_bills_l18_18152


namespace trig_identity_l18_18003

noncomputable def sin_deg (x : ℝ) := Real.sin (x * Real.pi / 180)
noncomputable def cos_deg (x : ℝ) := Real.cos (x * Real.pi / 180)
noncomputable def tan_deg (x : ℝ) := Real.tan (x * Real.pi / 180)

theorem trig_identity :
  (2 * sin_deg 50 + sin_deg 10 * (1 + Real.sqrt 3 * tan_deg 10) * Real.sqrt 2 * (sin_deg 80)^2) = Real.sqrt 6 :=
by
  sorry

end trig_identity_l18_18003


namespace apples_hand_out_l18_18661

theorem apples_hand_out (t p a h : ℕ) (h_t : t = 62) (h_p : p = 6) (h_a : a = 9) : h = t - (p * a) → h = 8 :=
by
  intros
  sorry

end apples_hand_out_l18_18661


namespace gcd_of_198_and_286_l18_18579

theorem gcd_of_198_and_286:
  let a := 198 
  let b := 286 
  let pf1 : a = 2 * 3^2 * 11 := by rfl
  let pf2 : b = 2 * 11 * 13 := by rfl
  gcd a b = 22 := by sorry

end gcd_of_198_and_286_l18_18579


namespace cube_painted_four_faces_l18_18594

theorem cube_painted_four_faces (n : ℕ) (hn : n ≠ 0) (h : (4 * n^2) / (6 * n^3) = 1 / 3) : n = 2 :=
by
  have : 4 * n^2 = 4 * n^2 := by rfl
  sorry

end cube_painted_four_faces_l18_18594


namespace xu_jun_age_l18_18927

variable (x y : ℕ)

def condition1 : Prop := y - 2 = 3 * (x - 2)
def condition2 : Prop := y + 8 = 2 * (x + 8)

theorem xu_jun_age (h1 : condition1 x y) (h2 : condition2 x y) : x = 12 :=
by 
sorry

end xu_jun_age_l18_18927


namespace cost_condition_shirt_costs_purchasing_plans_maximize_profit_l18_18155

/-- Define the costs and prices of shirts A and B -/
def cost_A (m : ℝ) : ℝ := m
def cost_B (m : ℝ) : ℝ := m - 10
def price_A : ℝ := 260
def price_B : ℝ := 180

/-- Condition: total cost of 3 A shirts and 2 B shirts is 480 -/
theorem cost_condition (m : ℝ) : 3 * (cost_A m) + 2 * (cost_B m) = 480 := by
  sorry

/-- The cost of each A shirt is 100 and each B shirt is 90 -/
theorem shirt_costs : ∃ m, cost_A m = 100 ∧ cost_B m = 90 := by
  sorry

/-- Number of purchasing plans for at least $34,000 profit with 300 shirts and at most 110 A shirts -/
theorem purchasing_plans : ∃ x, 100 ≤ x ∧ x ≤ 110 ∧ 
  (260 * x + 180 * (300 - x) - 100 * x - 90 * (300 - x) ≥ 34000) := by
  sorry

/- Maximize profit given 60 < a < 80:
   - 60 < a < 70: 110 A shirts, 190 B shirts.
   - a = 70: any combination satisfying conditions.
   - 70 < a < 80: 100 A shirts, 200 B shirts. -/

theorem maximize_profit (a : ℝ) (ha : 60 < a ∧ a < 80) : 
  ∃ x, ((60 < a ∧ a < 70 ∧ x = 110 ∧ (300 - x) = 190) ∨ 
        (a = 70) ∨ 
        (70 < a ∧ a < 80 ∧ x = 100 ∧ (300 - x) = 200)) := by
  sorry

end cost_condition_shirt_costs_purchasing_plans_maximize_profit_l18_18155


namespace parrots_per_cage_l18_18912

theorem parrots_per_cage (total_birds : ℕ) (num_cages : ℕ) (parakeets_per_cage : ℕ) (total_parrots : ℕ) :
  total_birds = 48 → num_cages = 6 → parakeets_per_cage = 2 → total_parrots = 36 →
  ∀ P : ℕ, (total_parrots = P * num_cages) → P = 6 :=
by
  intros h1 h2 h3 h4 P h5
  subst h1 h2 h3 h4
  sorry

end parrots_per_cage_l18_18912


namespace square_side_length_l18_18710

theorem square_side_length (A : ℝ) (h : A = 169) : ∃ s : ℝ, s^2 = A ∧ s = 13 := by
  sorry

end square_side_length_l18_18710


namespace no_two_points_same_color_distance_one_l18_18619

/-- Prove that if a plane is colored using seven colors, it is not necessary that there will be two points of the same color exactly 1 unit apart. -/
theorem no_two_points_same_color_distance_one (coloring : ℝ × ℝ → Fin 7) :
  ¬ ∀ (x y : ℝ × ℝ), (dist x y = 1) → (coloring x = coloring y) :=
by
  sorry

end no_two_points_same_color_distance_one_l18_18619


namespace map_distance_to_real_distance_l18_18419

theorem map_distance_to_real_distance (d_map : ℝ) (scale : ℝ) (d_real : ℝ) 
    (h1 : d_map = 7.5) (h2 : scale = 8) : d_real = 60 :=
by
  sorry

end map_distance_to_real_distance_l18_18419


namespace part1_part2_l18_18337

open Real

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := log x - a * x - 1
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := exp (a * x) * f x a + x

theorem part1 (a : ℝ) : 
  (a ≤ 0 → ∀ x, ∀ y, f x a ≤ y) ∧ (a > 0 → ∃ x, ∀ y, f x a ≤ y ∧ y = log (1 / a) - 2) :=
sorry

theorem part2 (a m : ℝ) (h_a : a > 0) (x1 x2 : ℝ) (h_x1 : 0 < x1) (h_x2 : x1 < x2) 
  (h_g1 : g x1 a = 0) (h_g2 : g x2 a = 0) : x1 * (x2 ^ 2) > exp m → m ≤ 3 :=
sorry

end part1_part2_l18_18337


namespace ratio_platform_to_pole_l18_18278

variables (l t T v : ℝ)
-- Conditions
axiom constant_velocity : ∀ t l, l = v * t
axiom pass_pole : l = v * t
axiom pass_platform : 6 * l = v * T 

theorem ratio_platform_to_pole (h1 : l = v * t) (h2 : 6 * l = v * T) : T / t = 6 := 
  by sorry

end ratio_platform_to_pole_l18_18278


namespace range_x_minus_2y_l18_18139

variable (x y : ℝ)

def cond1 : Prop := -1 ≤ x ∧ x < 2
def cond2 : Prop := 0 < y ∧ y ≤ 1

theorem range_x_minus_2y 
  (h1 : cond1 x) 
  (h2 : cond2 y) : 
  -3 ≤ x - 2 * y ∧ x - 2 * y < 2 := 
by
  sorry

end range_x_minus_2y_l18_18139


namespace bonnets_per_orphanage_correct_l18_18947

-- Definitions for each day's bonnet count
def monday_bonnets := 10
def tuesday_and_wednesday_bonnets := 2 * monday_bonnets
def thursday_bonnets := monday_bonnets + 5
def friday_bonnets := thursday_bonnets - 5
def saturday_bonnets := friday_bonnets - 8
def sunday_bonnets := 3 * saturday_bonnets

-- Total bonnets made in the week
def total_bonnets := 
  monday_bonnets +
  tuesday_and_wednesday_bonnets +
  thursday_bonnets +
  friday_bonnets +
  saturday_bonnets +
  sunday_bonnets

-- The number of orphanages
def orphanages := 10

-- Bonnets sent to each orphanage
def bonnets_per_orphanage := total_bonnets / orphanages

theorem bonnets_per_orphanage_correct :
  bonnets_per_orphanage = 6 :=
by
  sorry

end bonnets_per_orphanage_correct_l18_18947


namespace sampled_individual_l18_18354

theorem sampled_individual {population_size sample_size : ℕ} (population_size_cond : population_size = 1000)
  (sample_size_cond : sample_size = 20) (sampled_number : ℕ) (sampled_number_cond : sampled_number = 15) :
  (∃ n : ℕ, sampled_number + n * (population_size / sample_size) = 65) :=
by 
  sorry

end sampled_individual_l18_18354


namespace percent_of_absent_students_l18_18244

noncomputable def absent_percentage : ℚ :=
  let total_students := 120
  let boys := 70
  let girls := 50
  let absent_boys := boys * (1/5 : ℚ)
  let absent_girls := girls * (1/4 : ℚ)
  let total_absent := absent_boys + absent_girls
  (total_absent / total_students) * 100

theorem percent_of_absent_students : absent_percentage = 22.5 := sorry

end percent_of_absent_students_l18_18244


namespace base_log_eq_l18_18591

theorem base_log_eq (x : ℝ) : (5 : ℝ)^(x + 7) = (6 : ℝ)^x → x = Real.logb (6 / 5 : ℝ) (5^7 : ℝ) := by
  sorry

end base_log_eq_l18_18591


namespace savings_for_mother_l18_18176

theorem savings_for_mother (Liam_oranges Claire_oranges: ℕ) (Liam_price_per_pair Claire_price_per_orange: ℝ) :
  Liam_oranges = 40 ∧ Liam_price_per_pair = 2.50 ∧
  Claire_oranges = 30 ∧ Claire_price_per_orange = 1.20 →
  ((Liam_oranges / 2) * Liam_price_per_pair + Claire_oranges * Claire_price_per_orange) = 86 :=
by
  intros
  sorry

end savings_for_mother_l18_18176


namespace contrapositive_proof_l18_18570

-- Defining the necessary variables and the hypothesis
variables (a b : ℝ)

theorem contrapositive_proof (h : a^2 - b^2 + 2 * a - 4 * b - 3 ≠ 0) : a - b ≠ 1 :=
sorry

end contrapositive_proof_l18_18570


namespace intersection_of_sets_l18_18124

def A : Set ℕ := {1, 2, 5}
def B : Set ℕ := {1, 3, 5}

theorem intersection_of_sets : A ∩ B = {1, 5} :=
by
  sorry

end intersection_of_sets_l18_18124


namespace problem_statement_l18_18053

theorem problem_statement (p q r : ℝ) (h1 : p + q + r = 5) (h2 : 1 / (p + q) + 1 / (q + r) + 1 / (p + r) = 9) :
  r / (p + q) + p / (q + r) + q / (p + r) = 42 := by
  sorry

end problem_statement_l18_18053


namespace arithmetic_sequence_a5_l18_18454

-- Definitions of the conditions
def is_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
∀ n, a (n + 1) = a n + 2

-- Statement of the theorem with conditions and conclusion
theorem arithmetic_sequence_a5 :
  ∃ a : ℕ → ℕ, is_arithmetic_sequence a ∧ a 1 = 1 ∧ a 5 = 9 :=
by
  sorry

end arithmetic_sequence_a5_l18_18454


namespace worst_player_is_son_or_sister_l18_18597

axiom Family : Type
axiom Woman : Family
axiom Brother : Family
axiom Son : Family
axiom Daughter : Family
axiom Sister : Family

axiom are_chess_players : ∀ f : Family, Prop
axiom is_twin : Family → Family → Prop
axiom is_best_player : Family → Prop
axiom is_worst_player : Family → Prop
axiom same_age : Family → Family → Prop
axiom opposite_sex : Family → Family → Prop
axiom is_sibling : Family → Family → Prop

-- Conditions
axiom all_are_chess_players : ∀ f, are_chess_players f
axiom worst_best_opposite_sex : ∀ w b, is_worst_player w → is_best_player b → opposite_sex w b
axiom worst_best_same_age : ∀ w b, is_worst_player w → is_best_player b → same_age w b
axiom twins_relationship : ∀ t1 t2, is_twin t1 t2 → (is_sibling t1 t2 ∨ (t1 = Woman ∧ t2 = Sister))

-- Goal
theorem worst_player_is_son_or_sister :
  ∃ w, (is_worst_player w ∧ (w = Son ∨ w = Sister)) :=
sorry

end worst_player_is_son_or_sister_l18_18597


namespace radius_of_O2016_l18_18110

-- Define the centers and radii of circles
variable (a : ℝ) (n : ℕ) (r : ℕ → ℝ)

-- Conditions
-- Radius of the first circle
def initial_radius := r 1 = 1 / (2 * a)
-- Sequence of the radius difference based on solution step
def radius_recursive := ∀ n > 1, r (n + 1) - r n = 1 / a

-- The final statement to be proven
theorem radius_of_O2016 (h1 : initial_radius a r) (h2 : radius_recursive a r) :
  r 2016 = 4031 / (2 * a) := 
by sorry

end radius_of_O2016_l18_18110


namespace rocco_total_usd_l18_18607

noncomputable def total_usd_quarters : ℝ := 40 * 0.25
noncomputable def total_usd_nickels : ℝ := 90 * 0.05

noncomputable def cad_to_usd : ℝ := 0.8
noncomputable def eur_to_usd : ℝ := 1.18
noncomputable def gbp_to_usd : ℝ := 1.4

noncomputable def total_cad_dimes : ℝ := 60 * 0.10 * 0.8
noncomputable def total_eur_cents : ℝ := 50 * 0.01 * 1.18
noncomputable def total_gbp_pence : ℝ := 30 * 0.01 * 1.4

noncomputable def total_usd : ℝ :=
  total_usd_quarters + total_usd_nickels + total_cad_dimes +
  total_eur_cents + total_gbp_pence

theorem rocco_total_usd : total_usd = 20.31 := sorry

end rocco_total_usd_l18_18607


namespace total_amount_paid_l18_18931

def p1 := 20
def p2 := p1 + 2
def p3 := p2 + 3
def p4 := p3 + 4

theorem total_amount_paid : p1 + p2 + p3 + p4 = 96 :=
by
  sorry

end total_amount_paid_l18_18931


namespace peanuts_in_box_l18_18235

   theorem peanuts_in_box (initial_peanuts : ℕ) (added_peanuts : ℕ) (total_peanuts : ℕ) 
     (h1 : initial_peanuts = 4) (h2 : added_peanuts = 6) : total_peanuts = initial_peanuts + added_peanuts :=
   by
     sorry

   example : peanuts_in_box 4 6 10 rfl rfl = rfl :=
   by
     sorry
   
end peanuts_in_box_l18_18235


namespace choose_bar_length_l18_18026

theorem choose_bar_length (x : ℝ) (h1 : 1 < x) (h2 : x < 4) : x = 3 :=
by
  sorry

end choose_bar_length_l18_18026


namespace smallest_prime_factor_of_setC_l18_18325

def setC : Set ℕ := {51, 53, 54, 56, 57}

def prime_factors (n : ℕ) : Set ℕ :=
  { p | p.Prime ∧ p ∣ n }

theorem smallest_prime_factor_of_setC :
  (∃ n ∈ setC, ∀ m ∈ setC, ∀ p ∈ prime_factors n, ∀ q ∈ prime_factors m, p ≤ q) ∧
  (∃ m ∈ setC, ∀ p ∈ prime_factors 54, ∀ q ∈ prime_factors m, p = q) := 
sorry

end smallest_prime_factor_of_setC_l18_18325


namespace prob_allergic_prescribed_l18_18833

def P (a : Prop) : ℝ := sorry

axiom P_conditional (A B : Prop) : P B > 0 → P (A ∧ B) = P A * P (B ∧ A) / P B

def A : Prop := sorry -- represent the event that a patient is prescribed Undetenin
def B : Prop := sorry -- represent the event that a patient is allergic to Undetenin

axiom P_A : P A = 0.10
axiom P_B_given_A : P (B ∧ A) / P A = 0.02
axiom P_B : P B = 0.04

theorem prob_allergic_prescribed : P (A ∧ B) / P B = 0.05 :=
by
  have h1 : P (A ∧ B) / P A = 0.10 * 0.02 := sorry -- using definition of P_A and P_B_given_A
  have h2 : P (A ∧ B) = 0.002 := sorry -- calculating the numerator P(B and A)
  exact sorry -- use the axiom P_B to complete the theorem

end prob_allergic_prescribed_l18_18833


namespace min_abs_sum_l18_18187

theorem min_abs_sum (a b c d : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0)
  (h1 : a^2 + b * c = 9) (h2 : b * c + d^2 = 9) (h3 : a * b + b * d = 0) (h4 : a * c + c * d = 0) :
  |a| + |b| + |c| + |d| = 8 :=
sorry

end min_abs_sum_l18_18187


namespace art_performance_selection_l18_18803

-- Definitions from the conditions
def total_students := 6
def singers := 3
def dancers := 2
def both := 1

-- Mathematical expression in Lean
noncomputable def ways_to_select (n k : ℕ) : ℕ := Nat.choose n k

theorem art_performance_selection 
    (total_students singers dancers both: ℕ) 
    (h1 : total_students = 6)
    (h2 : singers = 3)
    (h3 : dancers = 2)
    (h4 : both = 1) :
  (ways_to_select 4 2 * 3 - 1) = (Nat.choose 4 2 * 3 - 1) := 
sorry

end art_performance_selection_l18_18803


namespace find_c_l18_18617

-- Definitions from the problem conditions
variables (a c : ℕ)
axiom cond1 : 2 ^ a = 8
axiom cond2 : a = 3 * c

-- The goal is to prove c = 1
theorem find_c : c = 1 :=
by
  sorry

end find_c_l18_18617


namespace school_girls_more_than_boys_l18_18353

def num_initial_girls := 632
def num_initial_boys := 410
def num_new_girls := 465
def num_total_girls := num_initial_girls + num_new_girls
def num_difference_girls_boys := num_total_girls - num_initial_boys

theorem school_girls_more_than_boys :
  num_difference_girls_boys = 687 :=
by
  sorry

end school_girls_more_than_boys_l18_18353


namespace salt_solution_proof_l18_18513

theorem salt_solution_proof (x : ℝ) (P : ℝ) (hx : x = 28.571428571428573) :
  ((P / 100) * 100 + x) = 0.30 * (100 + x) → P = 10 :=
by
  sorry

end salt_solution_proof_l18_18513


namespace initial_amount_l18_18692

-- Define the conditions
def cost_small_glass : ℕ := 3
def cost_large_glass : ℕ := 5
def num_small_glasses : ℕ := 8
def num_large_glasses : ℕ := 5
def change_left : ℕ := 1

-- Define the pieces based on conditions
def total_cost_small_glasses : ℕ := num_small_glasses * cost_small_glass
def total_cost_large_glasses : ℕ := num_large_glasses * cost_large_glass
def total_cost_glasses : ℕ := total_cost_small_glasses + total_cost_large_glasses

-- The theorem we need to prove
theorem initial_amount (h1 : total_cost_small_glasses = 24)
                       (h2 : total_cost_large_glasses = 25)
                       (h3 : total_cost_glasses = 49) : total_cost_glasses + change_left = 50 :=
by sorry

end initial_amount_l18_18692


namespace largest_consecutive_sum_to_35_l18_18129

theorem largest_consecutive_sum_to_35 (n : ℕ) (h : ∃ a : ℕ, (n * (2 * a + n - 1)) / 2 = 35) : n ≤ 7 :=
by
  sorry

end largest_consecutive_sum_to_35_l18_18129


namespace range_of_a_l18_18787

open Set

variable (a : ℝ)

noncomputable def I := univ ℝ
noncomputable def A := {x : ℝ | x ≤ a + 1}
noncomputable def B := {x : ℝ | x ≥ 1}
noncomputable def complement_B := {x : ℝ | x < 1}

theorem range_of_a (h : A a ⊆ complement_B) : a < 0 := sorry

end range_of_a_l18_18787


namespace broken_line_AEC_correct_l18_18669

noncomputable def length_of_broken_line_AEC 
  (side_length : ℝ)
  (height_of_pyramid : ℝ)
  (radius_of_equiv_circle : ℝ) 
  (length_AE : ℝ)
  (length_AEC : ℝ) : Prop :=
  side_length = 230.0 ∧
  height_of_pyramid = 146.423 ∧
  radius_of_equiv_circle = height_of_pyramid ∧
  length_AE = ((230.0 * 186.184) / 218.837) ∧
  length_AEC = 2 * length_AE ∧
  round (length_AEC * 100) = 39136

theorem broken_line_AEC_correct :
  length_of_broken_line_AEC 230 146.423 (146.423) 195.681 391.362 :=
by
  sorry

end broken_line_AEC_correct_l18_18669


namespace garage_has_18_wheels_l18_18409

namespace Garage

def bike_wheels_per_bike : ℕ := 2
def bikes_assembled : ℕ := 9

theorem garage_has_18_wheels
  (b : ℕ := bikes_assembled) 
  (w : ℕ := bike_wheels_per_bike) :
  b * w = 18 :=
by
  sorry

end Garage

end garage_has_18_wheels_l18_18409


namespace inequality_solution_l18_18167

theorem inequality_solution 
  (x : ℝ) : 
  (x^2 / (x+2)^2 ≥ 0) ↔ x ≠ -2 := 
by
  sorry

end inequality_solution_l18_18167


namespace average_income_A_B_l18_18349

def monthly_incomes (A B C : ℝ) : Prop :=
  (A = 4000) ∧
  ((B + C) / 2 = 6250) ∧
  ((A + C) / 2 = 5200)

theorem average_income_A_B (A B C X : ℝ) (h : monthly_incomes A B C) : X = 5050 :=
by
  have hA : A = 4000 := h.1
  have hBC : (B + C) / 2 = 6250 := h.2.1
  have hAC : (A + C) / 2 = 5200 := h.2.2
  sorry

end average_income_A_B_l18_18349


namespace negation_of_p_l18_18126

variable (f : ℝ → ℝ)

theorem negation_of_p :
  (¬ (∀ x1 x2 : ℝ, (f x2 - f x1) * (x2 - x1) ≥ 0)) ↔ (∃ x1 x2 : ℝ, (f x2 - f x1) * (x2 - x1) < 0) :=
by
  sorry

end negation_of_p_l18_18126


namespace sum_infinite_geometric_series_l18_18827

theorem sum_infinite_geometric_series :
  ∑' (n : ℕ), (3 : ℝ) * ((1 / 3) ^ n) = (9 / 2 : ℝ) :=
sorry

end sum_infinite_geometric_series_l18_18827


namespace three_pow_2040_mod_5_l18_18712

theorem three_pow_2040_mod_5 : (3^2040) % 5 = 1 := by
  sorry

end three_pow_2040_mod_5_l18_18712


namespace smallest_positive_four_digit_equivalent_to_5_mod_8_l18_18642

theorem smallest_positive_four_digit_equivalent_to_5_mod_8 : 
  ∃ (n : ℕ), n ≥ 1000 ∧ n % 8 = 5 ∧ n = 1005 :=
by
  sorry

end smallest_positive_four_digit_equivalent_to_5_mod_8_l18_18642


namespace asymptotes_of_hyperbola_l18_18198

theorem asymptotes_of_hyperbola :
  ∀ (x y : ℝ), (x^2 / 4 - y^2 / 9 = 1) → (y = 3/2 * x ∨ y = -3/2 * x) :=
by
  intro x y h
  -- Proof would go here
  sorry

end asymptotes_of_hyperbola_l18_18198


namespace master_codes_count_l18_18993

def num_colors : ℕ := 7
def num_slots : ℕ := 5

theorem master_codes_count : num_colors ^ num_slots = 16807 := by
  sorry

end master_codes_count_l18_18993


namespace logic_problem_l18_18296

variable (p q : Prop)

theorem logic_problem (h₁ : ¬ p) (h₂ : p ∨ q) : p = False ∧ q = True :=
by
  sorry

end logic_problem_l18_18296


namespace circles_intersect_twice_l18_18789

noncomputable def circle1 (x y : ℝ) : Prop :=
  (x - 3)^2 + y^2 = 9

noncomputable def circle2 (x y : ℝ) : Prop :=
  x^2 + (y - 1.5)^2 = 9 / 4

theorem circles_intersect_twice : 
  (∃ (p : ℝ × ℝ), circle1 p.1 p.2 ∧ circle2 p.1 p.2) ∧ 
  (∀ (p q : ℝ × ℝ), circle1 p.1 p.2 ∧ circle2 p.1 p.2 ∧ circle1 q.1 q.2 ∧ circle2 q.1 q.2 → (p = q ∨ p ≠ q)) →
  ∃ (p1 p2 : ℝ × ℝ), 
    p1 ≠ p2 ∧
    circle1 p1.1 p1.2 ∧ circle2 p1.1 p1.2 ∧
    circle1 p2.1 p2.2 ∧ circle2 p2.1 p2.2 := 
by {
  sorry
}

end circles_intersect_twice_l18_18789


namespace ship_speeds_l18_18004

theorem ship_speeds (x : ℝ) 
  (h1 : (2 * x) ^ 2 + (2 * (x + 3)) ^ 2 = 174 ^ 2) :
  x = 60 ∧ x + 3 = 63 :=
by
  sorry

end ship_speeds_l18_18004


namespace initial_percentage_of_water_l18_18145

/-
Initial conditions:
- Let W be the initial percentage of water in 10 liters of milk.
- The mixture should become 2% water after adding 15 liters of pure milk to it.
-/

theorem initial_percentage_of_water (W : ℚ) 
  (H1 : 0 < W) (H2 : W < 100) 
  (H3 : (10 * (100 - W) / 100 + 15) / 25 = 0.98) : 
  W = 5 := 
sorry

end initial_percentage_of_water_l18_18145


namespace fish_price_relation_l18_18698

variables (b_c m_c b_v m_v : ℝ)

axiom cond1 : 3 * b_c + m_c = 5 * b_v
axiom cond2 : 2 * b_c + m_c = 3 * b_v + m_v

theorem fish_price_relation : 5 * m_v = b_c + 2 * m_c :=
by
  sorry

end fish_price_relation_l18_18698


namespace expand_expression_l18_18390

theorem expand_expression (x : ℝ) : 12 * (3 * x - 4) = 36 * x - 48 := by
  sorry

end expand_expression_l18_18390


namespace find_principal_and_rate_l18_18792

variables (P R : ℝ)

theorem find_principal_and_rate
  (h1 : 20 = P * R * 2 / 100)
  (h2 : 22 = P * ((1 + R / 100) ^ 2 - 1)) :
  P = 50 ∧ R = 20 :=
by
  sorry

end find_principal_and_rate_l18_18792


namespace systematic_sampling_interval_l18_18726

-- Definitions based on the conditions in part a)
def total_students : ℕ := 1500
def sample_size : ℕ := 30

-- The goal is to prove that the interval k in systematic sampling equals 50
theorem systematic_sampling_interval :
  (total_students / sample_size = 50) :=
by
  sorry

end systematic_sampling_interval_l18_18726


namespace original_number_is_7_l18_18212

theorem original_number_is_7 (N : ℕ) (h : ∃ (k : ℤ), N = 12 * k + 7) : N = 7 :=
sorry

end original_number_is_7_l18_18212


namespace abc_sum_eq_sixteen_l18_18835

theorem abc_sum_eq_sixteen (a b c : ℤ) (h1 : a ≠ b ∨ a ≠ c ∨ b ≠ c) (h2 : a ≥ 4 ∧ b ≥ 4 ∧ c ≥ 4) (h3 : 4 * a * b * c = (a + 3) * (b + 3) * (c + 3)) : a + b + c = 16 :=
by 
  sorry

end abc_sum_eq_sixteen_l18_18835


namespace correct_expression_l18_18638

theorem correct_expression :
  (2 + Real.sqrt 3 ≠ 2 * Real.sqrt 3) ∧ 
  (Real.sqrt 8 - Real.sqrt 3 ≠ Real.sqrt 5) ∧ 
  (Real.sqrt 3 * Real.sqrt 2 = Real.sqrt 6) ∧ 
  (Real.sqrt 27 / Real.sqrt 3 ≠ 9) := 
by
  sorry

end correct_expression_l18_18638


namespace unique_solution_linear_system_l18_18776

theorem unique_solution_linear_system
  (a11 a22 a33 : ℝ) (a12 a13 a21 a23 a31 a32 : ℝ) 
  (x1 x2 x3 : ℝ) 
  (h1 : 0 < a11) (h2 : 0 < a22) (h3 : 0 < a33)
  (h4 : a12 < 0) (h5 : a13 < 0) (h6 : a21 < 0) (h7 : a23 < 0) (h8 : a31 < 0) (h9 : a32 < 0)
  (h10 : 0 < a11 + a12 + a13) (h11 : 0 < a21 + a22 + a23) (h12 : 0 < a31 + a32 + a33) :
  (a11 * x1 + a12 * x2 + a13 * x3 = 0) →
  (a21 * x1 + a22 * x2 + a23 * x3 = 0) →
  (a31 * x1 + a32 * x2 + a33 * x3 = 0) →
  x1 = 0 ∧ x2 = 0 ∧ x3 = 0 := by
  sorry

end unique_solution_linear_system_l18_18776


namespace geom_series_first_term_l18_18901

theorem geom_series_first_term (r : ℚ) (S : ℚ) (a : ℚ) 
  (h_r : r = 1 / 4) 
  (h_S : S = 80)
  : a = 60 :=
by
  sorry

end geom_series_first_term_l18_18901


namespace question1_question2_l18_18984

def f (x : ℝ) : ℝ := |x + 7| + |x - 1|

theorem question1 (x : ℝ) : ∀ m : ℝ, (∀ x : ℝ, f x ≥ m) → m ≤ 8 :=
by sorry

theorem question2 (x : ℝ) : (∀ x : ℝ, |x - 3| - 2 * x ≤ 2 * 8 - 12) ↔ (x ≥ -1/3) :=
by sorry

end question1_question2_l18_18984


namespace mock_exam_girls_count_l18_18056

theorem mock_exam_girls_count
  (B G Bc Gc : ℕ)
  (h1: B + G = 400)
  (h2: Bc = 60 * B / 100)
  (h3: Gc = 80 * G / 100)
  (h4: Bc + Gc = 65 * 400 / 100)
  : G = 100 :=
sorry

end mock_exam_girls_count_l18_18056


namespace ratio_night_to_day_l18_18716

-- Definitions based on conditions
def birds_day : ℕ := 8
def birds_total : ℕ := 24
def birds_night : ℕ := birds_total - birds_day

-- Theorem statement
theorem ratio_night_to_day : birds_night / birds_day = 2 := by
  sorry

end ratio_night_to_day_l18_18716


namespace tangent_line_equation_l18_18981

noncomputable def curve := fun x : ℝ => Real.sin (x + Real.pi / 3)

def tangent_line (x y : ℝ) : Prop :=
  x - 2 * y + Real.sqrt 3 = 0

theorem tangent_line_equation :
  tangent_line 0 (curve 0) := by
  unfold curve tangent_line
  sorry

end tangent_line_equation_l18_18981


namespace find_m_l18_18388

def U : Set Nat := {1, 2, 3}
def A (m : Nat) : Set Nat := {1, m}
def complement (s t : Set Nat) : Set Nat := {x | x ∈ s ∧ x ∉ t}

theorem find_m (m : Nat) (h1 : complement U (A m) = {2}) : m = 3 :=
by
  sorry

end find_m_l18_18388


namespace Alex_sandwich_count_l18_18083

theorem Alex_sandwich_count :
  let meats := 10
  let cheeses := 9
  let sandwiches := meats * (cheeses.choose 2)
  sandwiches = 360 :=
by
  -- Here start your proof
  sorry

end Alex_sandwich_count_l18_18083


namespace marks_lost_per_wrong_answer_l18_18028

theorem marks_lost_per_wrong_answer 
  (marks_per_correct : ℕ)
  (total_questions : ℕ)
  (total_marks : ℕ)
  (correct_answers : ℕ)
  (wrong_answers : ℕ)
  (score_from_correct : ℕ := correct_answers * marks_per_correct)
  (remaining_marks : ℕ := score_from_correct - total_marks)
  (marks_lost_per_wrong : ℕ) :
  total_questions = correct_answers + wrong_answers →
  total_marks = 130 →
  correct_answers = 38 →
  total_questions = 60 →
  marks_per_correct = 4 →
  marks_lost_per_wrong * wrong_answers = remaining_marks →
  marks_lost_per_wrong = 1 := 
sorry

end marks_lost_per_wrong_answer_l18_18028


namespace evaluate_expression_l18_18340

theorem evaluate_expression :
  -5^2 + 2 * (-3)^2 - (-8) / (-1 + 1/3) = -13 :=
by 
  sorry

end evaluate_expression_l18_18340


namespace marcus_leah_together_l18_18740

def num_games_with_combination (n k : ℕ) : ℕ :=
  Nat.choose n k

def num_games_together (total_players players_per_game : ℕ) (games_with_each_combination: ℕ) : ℕ :=
  total_players / players_per_game * games_with_each_combination

/-- Prove that Marcus and Leah play 210 games together. -/
theorem marcus_leah_together :
  let total_players := 12
  let players_per_game := 6
  let total_games := num_games_with_combination total_players players_per_game
  let marc_per_game := total_games / 2
  let together_pcnt := 5 / 11
  together_pcnt * marc_per_game = 210 :=
by
  sorry

end marcus_leah_together_l18_18740


namespace mass_percentage_O_in_N2O_is_approximately_36_35_l18_18545

noncomputable def atomic_mass_N : ℝ := 14.01
noncomputable def atomic_mass_O : ℝ := 16.00

noncomputable def number_of_N : ℕ := 2
noncomputable def number_of_O : ℕ := 1

noncomputable def molar_mass_N2O : ℝ := (number_of_N * atomic_mass_N) + (number_of_O * atomic_mass_O)

noncomputable def mass_percentage_O : ℝ := (atomic_mass_O / molar_mass_N2O) * 100

theorem mass_percentage_O_in_N2O_is_approximately_36_35 :
  abs (mass_percentage_O - 36.35) < 0.01 := sorry

end mass_percentage_O_in_N2O_is_approximately_36_35_l18_18545


namespace minimum_value_fraction_l18_18733

theorem minimum_value_fraction (a : ℝ) (h : a > 1) : (a^2 - a + 1) / (a - 1) ≥ 3 :=
by
  sorry

end minimum_value_fraction_l18_18733


namespace sqrt_6_approx_l18_18639

noncomputable def newton_iteration (x : ℝ) : ℝ :=
  (1 / 2) * x + (3 / x)

theorem sqrt_6_approx :
  let x0 : ℝ := 2
  let x1 : ℝ := newton_iteration x0
  let x2 : ℝ := newton_iteration x1
  let x3 : ℝ := newton_iteration x2
  abs (x3 - 2.4495) < 0.0001 :=
by
  sorry

end sqrt_6_approx_l18_18639


namespace time_for_B_and_C_l18_18482

variables (a b c : ℝ)

-- Conditions
axiom cond1 : a = (1 / 2) * b
axiom cond2 : b = 2 * c
axiom cond3 : a + b + c = 1 / 26
axiom cond4 : a + b = 1 / 13
axiom cond5 : a + c = 1 / 39

-- Statement to prove
theorem time_for_B_and_C (a b c : ℝ) (cond1 : a = (1 / 2) * b)
                                      (cond2 : b = 2 * c)
                                      (cond3 : a + b + c = 1 / 26)
                                      (cond4 : a + b = 1 / 13)
                                      (cond5 : a + c = 1 / 39) :
  (1 / (b + c)) = 104 / 3 :=
sorry

end time_for_B_and_C_l18_18482


namespace taco_price_theorem_l18_18172

noncomputable def price_hard_shell_taco_proof
  (H : ℤ)
  (price_soft : ℤ := 2)
  (num_hard_tacos_family : ℤ := 4)
  (num_soft_tacos_family : ℤ := 3)
  (num_additional_customers : ℤ := 10)
  (total_earnings : ℤ := 66)
  : Prop :=
  4 * H + 3 * price_soft + 10 * 2 * price_soft = total_earnings → H = 5

theorem taco_price_theorem : price_hard_shell_taco_proof 5 := 
by
  sorry

end taco_price_theorem_l18_18172


namespace trigonometry_expression_zero_l18_18177

variable {r : ℝ} {A B C : ℝ}
variable (a b c : ℝ) (sinA sinB sinC : ℝ)

-- The conditions from the problem
axiom Law_of_Sines_a : a = 2 * r * sinA
axiom Law_of_Sines_b : b = 2 * r * sinB
axiom Law_of_Sines_c : c = 2 * r * sinC

-- The theorem statement
theorem trigonometry_expression_zero :
  a * (sinC - sinB) + b * (sinA - sinC) + c * (sinB - sinA) = 0 :=
by
  -- Skipping the proof
  sorry

end trigonometry_expression_zero_l18_18177


namespace value_of_n_l18_18429

def is_3_digit_integer (n : ℕ) : Prop := (100 ≤ n) ∧ (n < 1000)

def not_divisible_by (n k : ℕ) : Prop := ¬ (k ∣ n)

def least_common_multiple (a b c : ℕ) : Prop := Nat.lcm a b = c

theorem value_of_n (d n : ℕ) (h1 : least_common_multiple d n 690) 
  (h2 : not_divisible_by n 3) (h3 : not_divisible_by d 2) (h4 : is_3_digit_integer n) : n = 230 :=
by
  sorry

end value_of_n_l18_18429


namespace fibonacci_problem_l18_18490

theorem fibonacci_problem 
  (F : ℕ → ℕ)
  (h1 : F 1 = 1)
  (h2 : F 2 = 1)
  (h3 : ∀ n ≥ 3, F n = F (n - 1) + F (n - 2))
  (a b c : ℕ)
  (h4 : F c = 2 * F b - F a)
  (h5 : F c - F a = F a)
  (h6 : a + c = 1700) :
  a = 849 := 
sorry

end fibonacci_problem_l18_18490


namespace operation_correct_l18_18718

def operation (a b : ℝ) : ℝ := 2 * a + 5 * b

theorem operation_correct :
  operation 4 2 = 18 :=
by
  show 2 * 4 + 5 * 2 = 18
  sorry

end operation_correct_l18_18718


namespace roots_of_quadratic_l18_18873

theorem roots_of_quadratic (x : ℝ) (h : x^2 = x) : x = 0 ∨ x = 1 :=
by
  sorry

end roots_of_quadratic_l18_18873


namespace identify_INPUT_statement_l18_18051

/-- Definition of the PRINT statement --/
def is_PRINT_statement (s : String) : Prop := s = "PRINT"

/-- Definition of the INPUT statement --/
def is_INPUT_statement (s : String) : Prop := s = "INPUT"

/-- Definition of the IF statement --/
def is_IF_statement (s : String) : Prop := s = "IF"

/-- Definition of the WHILE statement --/
def is_WHILE_statement (s : String) : Prop := s = "WHILE"

/-- Proof statement that the INPUT statement is the one for input --/
theorem identify_INPUT_statement (s : String) (h1 : is_PRINT_statement "PRINT") (h2: is_INPUT_statement "INPUT") (h3 : is_IF_statement "IF") (h4 : is_WHILE_statement "WHILE") : s = "INPUT" :=
sorry

end identify_INPUT_statement_l18_18051


namespace cos_330_eq_sqrt3_over_2_l18_18635

theorem cos_330_eq_sqrt3_over_2 : Real.cos (330 * Real.pi / 180) = Real.sqrt 3 / 2 :=
by
  sorry

end cos_330_eq_sqrt3_over_2_l18_18635


namespace units_digit_product_of_four_consecutive_integers_l18_18347

theorem units_digit_product_of_four_consecutive_integers (n : ℕ) (h : n % 2 = 1) : (n * (n + 1) * (n + 2) * (n + 3)) % 10 = 0 := 
by 
  sorry

end units_digit_product_of_four_consecutive_integers_l18_18347


namespace martha_found_blocks_l18_18121

variable (initial_blocks final_blocks found_blocks : ℕ)

theorem martha_found_blocks 
    (h_initial : initial_blocks = 4) 
    (h_final : final_blocks = 84) 
    (h_found : found_blocks = final_blocks - initial_blocks) : 
    found_blocks = 80 := by
  sorry

end martha_found_blocks_l18_18121


namespace algorithm_contains_sequential_structure_l18_18173

theorem algorithm_contains_sequential_structure :
  (∀ algorithm : Type, ∃ seq_struct : Prop, seq_struct) ∧
  (∀ algorithm : Type, ∃ sel_struct : Prop, sel_struct ∨ ¬ sel_struct) ∧
  (∀ algorithm : Type, ∃ loop_struct : Prop, loop_struct) →
  (∀ algorithm : Type, ∃ seq_struct : Prop, seq_struct) := by
  sorry

end algorithm_contains_sequential_structure_l18_18173


namespace total_strawberries_l18_18293

-- Define the number of original strawberries and the number of picked strawberries
def original_strawberries : ℕ := 42
def picked_strawberries : ℕ := 78

-- Prove the total number of strawberries
theorem total_strawberries : original_strawberries + picked_strawberries = 120 := by
  -- Proof goes here
  sorry

end total_strawberries_l18_18293


namespace david_marks_in_biology_l18_18269

theorem david_marks_in_biology (marks_english marks_math marks_physics marks_chemistry : ℕ)
  (average_marks num_subjects total_marks_known : ℕ)
  (h1 : marks_english = 76)
  (h2 : marks_math = 65)
  (h3 : marks_physics = 82)
  (h4 : marks_chemistry = 67)
  (h5 : average_marks = 75)
  (h6 : num_subjects = 5)
  (h7 : total_marks_known = marks_english + marks_math + marks_physics + marks_chemistry)
  (h8 : total_marks_known = 290)
  : ∃ biology_marks : ℕ, biology_marks = 85 ∧ biology_marks = (average_marks * num_subjects) - total_marks_known :=
by
  -- placeholder for proof
  sorry

end david_marks_in_biology_l18_18269


namespace range_of_a1_l18_18217

theorem range_of_a1 (a1 : ℝ) :
  (∃ (a2 a3 : ℝ), 
    ((a2 = 2 * a1 - 12) ∨ (a2 = a1 / 2 + 12)) ∧
    ((a3 = 2 * a2 - 12) ∨ (a3 = a2 / 2 + 12)) ) →
  ((a3 > a1) ↔ ((a1 ≤ 12) ∨ (24 ≤ a1))) :=
by
  sorry

end range_of_a1_l18_18217


namespace bobby_books_count_l18_18113

variable (KristiBooks BobbyBooks : ℕ)

theorem bobby_books_count (h1 : KristiBooks = 78) (h2 : BobbyBooks = KristiBooks + 64) : BobbyBooks = 142 :=
by
  sorry

end bobby_books_count_l18_18113


namespace solve_for_n_l18_18266

theorem solve_for_n (n : ℕ) : (9^n * 9^n * 9^n * 9^n = 729^4) -> n = 3 := 
by
  sorry

end solve_for_n_l18_18266


namespace solveInequalityRegion_l18_18237

noncomputable def greatestIntegerLessThan (x : ℝ) : ℤ :=
  Int.floor x

theorem solveInequalityRegion :
  ∀ (x y : ℝ), abs x < 1 → abs y < 1 → x * y ≠ 0 → (greatestIntegerLessThan (x + y) ≤ 
  greatestIntegerLessThan x + greatestIntegerLessThan y) :=
by
  intros x y h1 h2 h3
  sorry

end solveInequalityRegion_l18_18237


namespace inequality_pos_reals_l18_18370

theorem inequality_pos_reals (x y z : ℝ) (h₁ : 0 < x) (h₂ : 0 < y) (h₃ : 0 < z) : 
  (x^2 + 2) * (y^2 + 2) * (z^2 + 2) ≥ 9 * (x * y + y * z + z * x) :=
by
  sorry

end inequality_pos_reals_l18_18370


namespace spherical_ball_radius_l18_18261

noncomputable def largest_spherical_ball_radius (inner_radius outer_radius : ℝ) (center : ℝ × ℝ × ℝ) (table_z : ℝ) : ℝ :=
  let r := 4
  r

theorem spherical_ball_radius
  (inner_radius outer_radius : ℝ)
  (center : ℝ × ℝ × ℝ)
  (table_z : ℝ)
  (h1 : inner_radius = 3)
  (h2 : outer_radius = 5)
  (h3 : center = (4,0,1))
  (h4 : table_z = 0) :
  largest_spherical_ball_radius inner_radius outer_radius center table_z = 4 :=
by sorry

end spherical_ball_radius_l18_18261


namespace haley_lives_gained_l18_18779

-- Define the given conditions
def initial_lives : ℕ := 14
def lives_lost : ℕ := 4
def total_lives_after_gain : ℕ := 46

-- Define the goal: How many lives did Haley gain in the next level?
theorem haley_lives_gained : (total_lives_after_gain = initial_lives - lives_lost + lives_gained) → lives_gained = 36 :=
by
  intro h
  sorry

end haley_lives_gained_l18_18779


namespace maximize_revenue_l18_18064

theorem maximize_revenue (p : ℝ) (hp : p ≤ 30) :
  (p = 12 ∨ p = 13) → (∀ p : ℤ, p ≤ 30 → 200 * p - 8 * p * p ≤ 1248) :=
by
  intros h1 h2
  sorry

end maximize_revenue_l18_18064


namespace exists_integers_a_b_c_d_l18_18190

-- Define the problem statement in Lean 4

theorem exists_integers_a_b_c_d (x y z : ℕ) (h : x * y = z^2 + 1) :
  ∃ (a b c d : ℤ), x = a^2 + b^2 ∧ y = c^2 + d^2 ∧ z = a * c + b * d :=
by
  sorry

end exists_integers_a_b_c_d_l18_18190


namespace valid_rearrangements_count_l18_18850

noncomputable def count_valid_rearrangements : ℕ := sorry

theorem valid_rearrangements_count :
  count_valid_rearrangements = 7 :=
sorry

end valid_rearrangements_count_l18_18850


namespace unique_number_not_in_range_l18_18168

noncomputable def g (p q r s : ℝ) (x : ℝ) : ℝ :=
  (p * x + q) / (r * x + s)

theorem unique_number_not_in_range (p q r s : ℝ) (h₀ : p ≠ 0) (h₁ : q ≠ 0) (h₂ : r ≠ 0) (h₃ : s ≠ 0) 
  (h₄ : g p q r s 23 = 23) (h₅ : g p q r s 101 = 101) (h₆ : ∀ x, x ≠ -s/r → g p q r s (g p q r s x) = x) :
  p / r = 62 :=
sorry

end unique_number_not_in_range_l18_18168


namespace number_of_maple_trees_planted_l18_18943

def before := 53
def after := 64
def planted := after - before

theorem number_of_maple_trees_planted : planted = 11 := by
  sorry

end number_of_maple_trees_planted_l18_18943


namespace num_solutions_system_eqns_l18_18994

theorem num_solutions_system_eqns :
  ∃ (c : ℕ), 
    (∀ (a1 a2 a3 a4 a5 a6 : ℕ), 
       a1 + 2 * a2 + 3 * a3 + 4 * a4 + 5 * a5 + 6 * a6 = 26 ∧ 
       a1 + a2 + a3 + a4 + a5 + a6 = 5 → 
       (a1, a2, a3, a4, a5, a6) ∈ (solutions : Finset (ℕ × ℕ × ℕ × ℕ × ℕ × ℕ))) ∧
    solutions.card = 5 := sorry

end num_solutions_system_eqns_l18_18994


namespace problem_l18_18606

noncomputable def f (x : ℝ) : ℝ := Real.exp x + x - 2
noncomputable def g (x : ℝ) : ℝ := Real.log x + x^2 - 3

theorem problem (a b : ℝ) (H1 : f a = 0) (H2 : g b = 0) : g a < 0 ∧ 0 < f b :=
by
  sorry

end problem_l18_18606


namespace age_difference_l18_18495

variable (y m e : ℕ)

theorem age_difference (h1 : m = y + 3) (h2 : e = 3 * y) (h3 : e = 15) : 
  ∃ x, e = y + m + x ∧ x = 2 := by
  sorry

end age_difference_l18_18495


namespace a_squared_gt_b_squared_l18_18512

theorem a_squared_gt_b_squared {a b : ℝ} (h : a ≠ 0) (hb : b ≠ 0) (hb_domain : b > -1 ∧ b < 1) (h_eq : a = Real.log (1 + b) - Real.log (1 - b)) :
  a^2 > b^2 := 
sorry

end a_squared_gt_b_squared_l18_18512


namespace sum_of_real_values_l18_18707

theorem sum_of_real_values (x : ℝ) (h : |3 * x - 15| + |x - 5| = 92) : (x = 28 ∨ x = -18) → x + 10 = 0 := by
  sorry

end sum_of_real_values_l18_18707


namespace f_positive_when_a_1_f_negative_solution_sets_l18_18534

section

variable (f : ℝ → ℝ) (a x : ℝ)

def f_def := f x = (x - a) * (x - 2)

-- (Ⅰ) Problem statement
theorem f_positive_when_a_1 : (∀ x, f_def f 1 x → f x > 0 ↔ (x < 1) ∨ (x > 2)) :=
by sorry

-- (Ⅱ) Problem statement
theorem f_negative_solution_sets (a : ℝ) : 
  (∀ x, f_def f a x ∧ a = 2 → False) ∧ 
  (∀ x, f_def f a x ∧ a > 2 → 2 < x ∧ x < a) ∧ 
  (∀ x, f_def f a x ∧ a < 2 → a < x ∧ x < 2) :=
by sorry

end

end f_positive_when_a_1_f_negative_solution_sets_l18_18534


namespace quadratic_behavior_l18_18262

theorem quadratic_behavior (x : ℝ) : x < 3 → ∃ y : ℝ, y = 5 * (x - 3) ^ 2 + 2 ∧ ∀ x1 x2 : ℝ, x1 < x2 ∧ x1 < 3 ∧ x2 < 3 → (5 * (x1 - 3) ^ 2 + 2) > (5 * (x2 - 3) ^ 2 + 2) := 
by
  sorry

end quadratic_behavior_l18_18262


namespace minimum_value_of_fraction_l18_18795

variable {a b : ℝ}

theorem minimum_value_of_fraction (h1 : a > b) (h2 : a * b = 1) : 
  ∃ (c : ℝ), c = 2 * Real.sqrt 2 ∧ ∀ x > b, a * x = 1 -> 
  (x - b + 2 / (x - b) ≥ c) :=
by
  sorry

end minimum_value_of_fraction_l18_18795


namespace smallest_N_triangle_ineq_l18_18355

theorem smallest_N_triangle_ineq (a b c : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c)
  (h₄ : c < a + b) : (a^2 + b^2 + a * b) / c^2 < 1 := 
sorry

end smallest_N_triangle_ineq_l18_18355


namespace decimal_equivalent_of_one_quarter_l18_18813

theorem decimal_equivalent_of_one_quarter:
  ( (1:ℚ) / (4:ℚ) )^1 = 0.25 := 
sorry

end decimal_equivalent_of_one_quarter_l18_18813


namespace mean_equality_and_find_y_l18_18017

theorem mean_equality_and_find_y : 
  (8 + 9 + 18) / 3 = (15 + (25 / 3)) / 2 :=
by
  sorry

end mean_equality_and_find_y_l18_18017


namespace domain_of_log_function_l18_18577

theorem domain_of_log_function (x : ℝ) : 1 - x > 0 ↔ x < 1 := by
  sorry

end domain_of_log_function_l18_18577


namespace rectangle_circle_ratio_l18_18997

theorem rectangle_circle_ratio (r s : ℝ) (h : ∀ r s : ℝ, 2 * r * s - π * r^2 = π * r^2) : s / (2 * r) = π / 2 :=
by
  sorry

end rectangle_circle_ratio_l18_18997


namespace y_pow_x_eq_nine_l18_18660

theorem y_pow_x_eq_nine (x y : ℝ) (h : x^2 + y^2 - 4 * x + 6 * y + 13 = 0) : y^x = 9 := by
  sorry

end y_pow_x_eq_nine_l18_18660


namespace polynomial_value_l18_18924

theorem polynomial_value (a b : ℝ) : 
  (|a - 2| + (b + 1/2)^2 = 0) → (2 * a * b^2 + a^2 * b) - (3 * a * b^2 + a^2 * b - 1) = 1/2 :=
by
  sorry

end polynomial_value_l18_18924


namespace find_f_neg3_l18_18150

theorem find_f_neg3 : 
  (∃ f : ℝ → ℝ, (∀ x : ℝ, x ≠ 0 → 5 * f (1 / x) + 3 * f x / x = 2 * x^2) ∧ f (-3) = 14029 / 72) :=
sorry

end find_f_neg3_l18_18150


namespace original_triangle_area_l18_18259

-- Define the scaling factor and given areas
def scaling_factor : ℕ := 2
def new_triangle_area : ℕ := 32

-- State that if the dimensions of the original triangle are doubled, the area becomes 32 square feet
theorem original_triangle_area (original_area : ℕ) : (scaling_factor * scaling_factor) * original_area = new_triangle_area → original_area = 8 := 
by
  intros h
  sorry

end original_triangle_area_l18_18259


namespace no_such_function_exists_l18_18322

theorem no_such_function_exists :
  ¬ ∃ f : ℝ → ℝ, (f 0 > 0) ∧ (∀ x y : ℝ, f (x + y) ≥ f x + y * f (f x)) :=
by
  -- proof to be completed
  sorry

end no_such_function_exists_l18_18322


namespace product_base_8_units_digit_l18_18542

theorem product_base_8_units_digit :
  let sum := 324 + 73
  let product := sum * 27
  product % 8 = 7 :=
by
  let sum := 324 + 73
  let product := sum * 27
  have h : product % 8 = 7 := by
    sorry
  exact h

end product_base_8_units_digit_l18_18542


namespace pagoda_lanterns_l18_18068

-- Definitions
def top_layer_lanterns (a₁ : ℕ) : ℕ := a₁
def bottom_layer_lanterns (a₁ : ℕ) : ℕ := a₁ * 2^6
def sum_of_lanterns (a₁ : ℕ) : ℕ := (a₁ * (1 - 2^7)) / (1 - 2)
def total_lanterns : ℕ := 381
def layers : ℕ := 7
def common_ratio : ℕ := 2

-- Problem Statement
theorem pagoda_lanterns (a₁ : ℕ) (h : sum_of_lanterns a₁ = total_lanterns) : 
  top_layer_lanterns a₁ + bottom_layer_lanterns a₁ = 195 := sorry

end pagoda_lanterns_l18_18068


namespace line_passes_through_fixed_point_l18_18241

theorem line_passes_through_fixed_point (m : ℝ) : 
  (2 + m) * (-1) + (1 - 2 * m) * (-2) + 4 - 3 * m = 0 :=
by
  sorry

end line_passes_through_fixed_point_l18_18241


namespace angle_D_measure_l18_18797

theorem angle_D_measure (B C E F D : ℝ) 
  (h₁ : B = 120)
  (h₂ : B + C = 180)
  (h₃ : E = 45)
  (h₄ : F = C) 
  (h₅ : D + E + F = 180) :
  D = 75 := sorry

end angle_D_measure_l18_18797


namespace initial_blue_marbles_l18_18951

theorem initial_blue_marbles (B R : ℕ) 
    (h1 : 3 * B = 5 * R) 
    (h2 : 4 * (B - 10) = R + 25) : 
    B = 19 := 
sorry

end initial_blue_marbles_l18_18951


namespace set_equivalence_l18_18568

theorem set_equivalence :
  {p : ℝ × ℝ | p.1 + p.2 = 1 ∧ 2 * p.1 - p.2 = 2} = {(1, 0)} :=
by
  sorry

end set_equivalence_l18_18568


namespace orthocenter_of_ABC_is_correct_l18_18926

structure Point3D where
  x : ℚ
  y : ℚ
  z : ℚ

def A : Point3D := {x := 2, y := 3, z := -1}
def B : Point3D := {x := 6, y := -1, z := 2}
def C : Point3D := {x := 4, y := 5, z := 4}

def orthocenter (A B C : Point3D) : Point3D := {
  x := 101 / 33,
  y := 95 / 33,
  z := 47 / 33
}

theorem orthocenter_of_ABC_is_correct : orthocenter A B C = {x := 101 / 33, y := 95 / 33, z := 47 / 33} :=
  sorry

end orthocenter_of_ABC_is_correct_l18_18926


namespace total_time_to_make_cookies_l18_18531

def time_to_make_batter := 10
def baking_time := 15
def cooling_time := 15
def white_icing_time := 30
def chocolate_icing_time := 30

theorem total_time_to_make_cookies : 
  time_to_make_batter + baking_time + cooling_time + white_icing_time + chocolate_icing_time = 100 := 
by
  sorry

end total_time_to_make_cookies_l18_18531


namespace cricketer_total_matches_l18_18103

theorem cricketer_total_matches (n : ℕ)
  (avg_total : ℝ) (avg_first_6 : ℝ) (avg_last_4 : ℝ)
  (total_runs_eq : 6 * avg_first_6 + 4 * avg_last_4 = n * avg_total) :
  avg_total = 38.9 ∧ avg_first_6 = 42 ∧ avg_last_4 = 34.25 → n = 10 :=
by
  sorry

end cricketer_total_matches_l18_18103


namespace optimal_direction_l18_18257

-- Define the conditions as hypotheses
variables (a : ℝ) (V_first V_second : ℝ) (d : ℝ)
variable (speed_rel : V_first = 2 * V_second)
variable (dist : d = a)

-- Create a theorem statement for the problem
theorem optimal_direction (H : d = a) (vel_rel : V_first = 2 * V_second) : true := 
  sorry

end optimal_direction_l18_18257


namespace number_from_first_group_is_6_l18_18811

-- Defining conditions
def num_students : Nat := 160
def sample_size : Nat := 20
def groups := List.range' 0 num_students (num_students / sample_size)

def num_from_group_16 (x : Nat) : Nat := 8 * 15 + x
def drawn_number_from_16 : Nat := 126

-- Main theorem
theorem number_from_first_group_is_6 : ∃ x : Nat, num_from_group_16 x = drawn_number_from_16 ∧ x = 6 := 
by
  sorry

end number_from_first_group_is_6_l18_18811


namespace detergent_for_9_pounds_l18_18849

-- Define the given condition.
def detergent_per_pound : ℕ := 2

-- Define the total weight of clothes
def weight_of_clothes : ℕ := 9

-- Define the result of the detergent used.
def detergent_used (d : ℕ) (w : ℕ) : ℕ := d * w

-- Prove that the detergent used to wash 9 pounds of clothes is 18 ounces
theorem detergent_for_9_pounds :
  detergent_used detergent_per_pound weight_of_clothes = 18 := 
sorry

end detergent_for_9_pounds_l18_18849


namespace find_abs_product_l18_18021

noncomputable def distinct_nonzero_real (a b c : ℝ) : Prop :=
a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0

theorem find_abs_product (a b c : ℝ) (h1 : distinct_nonzero_real a b c) 
(h2 : a + 1/(b^2) = b + 1/(c^2))
(h3 : b + 1/(c^2) = c + 1/(a^2)) :
  |a * b * c| = 1 :=
sorry

end find_abs_product_l18_18021


namespace lcm_gcf_ratio_240_630_l18_18655

theorem lcm_gcf_ratio_240_630 :
  let a := 240
  let b := 630
  Nat.lcm a b / Nat.gcd a b = 168 := by
  sorry

end lcm_gcf_ratio_240_630_l18_18655


namespace simplify_expression_l18_18616

variable (a : ℝ)

theorem simplify_expression : 5 * a + 2 * a + 3 * a - 2 * a = 8 * a :=
by
  sorry

end simplify_expression_l18_18616


namespace prec_property_l18_18794

noncomputable def prec (a b : ℕ) : Prop :=
  sorry -- The construction of the relation from the problem

axiom prec_total : ∀ a b : ℕ, (prec a b ∨ prec b a ∨ a = b)
axiom prec_trans : ∀ a b c : ℕ, (prec a b ∧ prec b c) → prec a c

theorem prec_property : ∀ a b c : ℕ, (prec a b ∧ prec b c) → 2 * b ≠ a + c :=
by
  sorry

end prec_property_l18_18794


namespace rectangular_container_volume_l18_18896

theorem rectangular_container_volume (a b c : ℝ) 
  (h1 : a * b = 30) 
  (h2 : b * c = 20) 
  (h3 : c * a = 12) : 
  a * b * c = 60 :=
by
  sorry

end rectangular_container_volume_l18_18896


namespace miles_to_mall_l18_18249

noncomputable def miles_to_grocery_store : ℕ := 10
noncomputable def miles_to_pet_store : ℕ := 5
noncomputable def miles_back_home : ℕ := 9
noncomputable def miles_per_gallon : ℕ := 15
noncomputable def cost_per_gallon : ℝ := 3.50
noncomputable def total_cost_of_gas : ℝ := 7.00
noncomputable def total_miles_driven := 2 * miles_per_gallon

theorem miles_to_mall : total_miles_driven -
  (miles_to_grocery_store + miles_to_pet_store + miles_back_home) = 6 :=
by
  -- proof omitted 
  sorry

end miles_to_mall_l18_18249


namespace largest_positive_integer_n_l18_18034

def binary_operation (n : ℤ) : ℤ := n - (n * 5)

theorem largest_positive_integer_n (x : ℤ) (h : x = -15) : 
  ∃ (n : ℤ), n > 0 ∧ binary_operation n < x ∧ ∀ m > 0, binary_operation m < x → m ≤ n :=
by
  sorry

end largest_positive_integer_n_l18_18034


namespace min_value_of_expression_l18_18243

theorem min_value_of_expression (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x * y * z = 1) : 
  (3 * x + y) * (x + 3 * z) * (y + z + 1) ≥ 48 :=
by
  sorry

end min_value_of_expression_l18_18243


namespace average_speed_eq_l18_18699

variables (v₁ v₂ : ℝ) (t₁ t₂ : ℝ)

theorem average_speed_eq (h₁ : t₁ > 0) (h₂ : t₂ > 0) : 
  ((v₁ * t₁) + (v₂ * t₂)) / (t₁ + t₂) = (v₁ + v₂) / 2 := 
sorry

end average_speed_eq_l18_18699


namespace oranges_for_juice_l18_18672

theorem oranges_for_juice 
  (bags : ℕ) (oranges_per_bag : ℕ) (rotten_oranges : ℕ) (oranges_sold : ℕ)
  (h_bags : bags = 10)
  (h_oranges_per_bag : oranges_per_bag = 30)
  (h_rotten_oranges : rotten_oranges = 50)
  (h_oranges_sold : oranges_sold = 220):
  (bags * oranges_per_bag - rotten_oranges - oranges_sold = 30) :=
by 
  sorry

end oranges_for_juice_l18_18672


namespace regular_price_correct_l18_18562

noncomputable def regular_price_of_one_tire (x : ℝ) : Prop :=
  3 * x + 5 - 10 = 302

theorem regular_price_correct (x : ℝ) : regular_price_of_one_tire x → x = 307 / 3 := by
  intro h
  sorry

end regular_price_correct_l18_18562


namespace volume_of_rectangular_solid_l18_18864

variable {x y z : ℝ}
variable (hx : x * y = 3) (hy : x * z = 5) (hz : y * z = 15)

theorem volume_of_rectangular_solid : x * y * z = 15 :=
by sorry

end volume_of_rectangular_solid_l18_18864


namespace twelfth_term_arithmetic_sequence_l18_18571

-- Given conditions
def first_term : ℚ := 1 / 4
def common_difference : ℚ := 1 / 2

-- Statement to prove
theorem twelfth_term_arithmetic_sequence :
  (first_term + 11 * common_difference) = 23 / 4 :=
by
  sorry

end twelfth_term_arithmetic_sequence_l18_18571


namespace var_X_is_86_over_225_l18_18831

/-- The probability of Person A hitting the target is 2/3. -/
def prob_A : ℚ := 2 / 3

/-- The probability of Person B hitting the target is 4/5. -/
def prob_B : ℚ := 4 / 5

/-- The events of A and B hitting or missing the target are independent. -/
def independent_events : Prop := true -- In Lean, independence would involve more complex definitions.

def prob_X (x : ℕ) : ℚ :=
  if x = 0 then (1 - prob_A) * (1 - prob_B)
  else if x = 1 then (1 - prob_A) * prob_B + (1 - prob_B) * prob_A
  else if x = 2 then prob_A * prob_B
  else 0

/-- Expected value of X -/
noncomputable def expect_X : ℚ :=
  0 * prob_X 0 + 1 * prob_X 1 + 2 * prob_X 2

/-- Variance of X -/
noncomputable def var_X : ℚ :=
  (0 - expect_X) ^ 2 * prob_X 0 +
  (1 - expect_X) ^ 2 * prob_X 1 +
  (2 - expect_X) ^ 2 * prob_X 2

theorem var_X_is_86_over_225 : var_X = 86 / 225 :=
by {
  sorry
}

end var_X_is_86_over_225_l18_18831


namespace third_stick_length_l18_18824

theorem third_stick_length (x : ℝ) (h1 : 2 > 0) (h2 : 5 > 0) (h3 : 3 < x) (h4 : x < 7) : x = 4 :=
by
  sorry

end third_stick_length_l18_18824


namespace fraction_of_area_below_line_l18_18588

noncomputable def rectangle_area_fraction (x1 y1 x2 y2 : ℝ) (x3 y3 x4 y4 : ℝ) : ℝ :=
  let m := (y2 - y1) / (x2 - x1)
  let b := y1 - m * x1
  let y_intercept := b
  let base := x4 - x1
  let height := y4 - y3
  let triangle_area := 0.5 * base * height
  triangle_area / (base * height)

theorem fraction_of_area_below_line : 
  rectangle_area_fraction 1 3 5 1 1 0 5 4 = 1 / 8 := 
by
  sorry

end fraction_of_area_below_line_l18_18588


namespace parabola_standard_equation_l18_18629

theorem parabola_standard_equation :
  ∃ p1 p2 : ℝ, p1 > 0 ∧ p2 > 0 ∧ (y^2 = 2 * p1 * x ∨ x^2 = 2 * p2 * y) ∧ ((6, 4) ∈ {(x, y) | y^2 = 2 * p1 * x} ∨ (6, 4) ∈ {(x, y) | x^2 = 2 * p2 * y}) := 
  sorry

end parabola_standard_equation_l18_18629


namespace rival_awards_l18_18529

theorem rival_awards (scott_awards jessie_awards rival_awards : ℕ)
  (h1 : scott_awards = 4)
  (h2 : jessie_awards = 3 * scott_awards)
  (h3 : rival_awards = 2 * jessie_awards) :
  rival_awards = 24 :=
by sorry

end rival_awards_l18_18529


namespace min_value_mn_squared_l18_18977

theorem min_value_mn_squared (a b c m n : ℝ) 
  (h_triangle: a^2 + b^2 = c^2)
  (h_line: a * m + b * n + 2 * c = 0):
  m^2 + n^2 = 4 :=
by
  sorry

end min_value_mn_squared_l18_18977


namespace betty_eggs_per_teaspoon_vanilla_l18_18905

theorem betty_eggs_per_teaspoon_vanilla
  (sugar_cream_cheese_ratio : ℚ)
  (vanilla_cream_cheese_ratio : ℚ)
  (sugar_in_cups : ℚ)
  (eggs_used : ℕ)
  (expected_ratio : ℚ) :
  sugar_cream_cheese_ratio = 1/4 →
  vanilla_cream_cheese_ratio = 1/2 →
  sugar_in_cups = 2 →
  eggs_used = 8 →
  expected_ratio = 2 →
  (eggs_used / (sugar_in_cups * 4 * vanilla_cream_cheese_ratio)) = expected_ratio :=
by
  intros h1 h2 h3 h4 h5
  sorry

end betty_eggs_per_teaspoon_vanilla_l18_18905


namespace train_length_l18_18342

theorem train_length (L : ℝ) (v1 v2 : ℝ) 
  (h1 : v1 = (L + 140) / 15)
  (h2 : v2 = (L + 250) / 20) 
  (h3 : v1 = v2) :
  L = 190 :=
by sorry

end train_length_l18_18342


namespace molly_christmas_shipping_cost_l18_18537

def cost_per_package : ℕ := 5
def num_parents : ℕ := 2
def num_brothers : ℕ := 3
def num_sisters_in_law_per_brother : ℕ := 1
def num_children_per_brother : ℕ := 2

def total_relatives : ℕ :=
  num_parents + num_brothers + (num_brothers * num_sisters_in_law_per_brother) + (num_brothers * num_children_per_brother)

theorem molly_christmas_shipping_cost : total_relatives * cost_per_package = 70 :=
by
  sorry

end molly_christmas_shipping_cost_l18_18537


namespace abs_m_minus_1_greater_eq_abs_m_minus_1_l18_18674

theorem abs_m_minus_1_greater_eq_abs_m_minus_1 (m : ℝ) : |m - 1| ≥ |m| - 1 := 
sorry

end abs_m_minus_1_greater_eq_abs_m_minus_1_l18_18674


namespace factorize_expression_l18_18456

theorem factorize_expression (a : ℝ) : a^3 + 2*a^2 + a = a*(a+1)^2 :=
  sorry

end factorize_expression_l18_18456


namespace find_m_l18_18600

def g (n : ℤ) : ℤ :=
if n % 2 = 1 then n + 5 else n / 2

theorem find_m (m : ℤ) (h1 : m % 2 = 1) (h2 : g (g (g m)) = 35) : m = 135 :=
sorry

end find_m_l18_18600


namespace number_of_people_in_room_l18_18866

theorem number_of_people_in_room (P : ℕ) 
  (h1 : 1/4 * P = P / 4) 
  (h2 : 3/4 * P = 3 * P / 4) 
  (h3 : P / 4 = 20) : 
  P = 80 :=
sorry

end number_of_people_in_room_l18_18866


namespace men_entered_l18_18610

theorem men_entered (M W x : ℕ) (h1 : 4 * W = 5 * M)
                    (h2 : M + x = 14)
                    (h3 : 2 * (W - 3) = 24) :
                    x = 2 :=
by
  sorry

end men_entered_l18_18610


namespace least_positive_integer_solution_l18_18336

theorem least_positive_integer_solution :
  ∃ b : ℕ, b ≡ 2 [MOD 3] ∧ b ≡ 3 [MOD 4] ∧ b ≡ 4 [MOD 5] ∧ b ≡ 8 [MOD 9] ∧ b = 179 :=
by
  sorry

end least_positive_integer_solution_l18_18336


namespace find_r_values_l18_18814

theorem find_r_values (r : ℝ) (h1 : r ≥ 8) (h2 : r ≤ 20) :
  16 ≤ (r - 4) ^ (3/2) ∧ (r - 4) ^ (3/2) ≤ 128 :=
by {
  sorry
}

end find_r_values_l18_18814


namespace no_intersection_abs_eq_l18_18527

theorem no_intersection_abs_eq (x : ℝ) : ∀ y : ℝ, y = |3 * x + 6| → y = -|2 * x - 4| → false := 
by
  sorry

end no_intersection_abs_eq_l18_18527


namespace find_p_l18_18142

theorem find_p (m n p : ℝ) :
  m = (n / 7) - (2 / 5) →
  m + p = ((n + 21) / 7) - (2 / 5) →
  p = 3 := by
  sorry

end find_p_l18_18142


namespace transformed_curve_l18_18908

theorem transformed_curve :
  (∀ x y : ℝ, 3*x = x' ∧ 4*y = y' → x^2 + y^2 = 1) ↔ (x'^2 / 9 + y'^2 / 16 = 1) :=
by
  sorry

end transformed_curve_l18_18908


namespace inequality_comparison_l18_18652

theorem inequality_comparison 
  (a : ℝ) (b : ℝ) (c : ℝ) 
  (h₁ : a = (1 / Real.log 3 / Real.log 2))
  (h₂ : b = Real.exp 0.5)
  (h₃ : c = Real.log 2) :
  b > c ∧ c > a := 
by
  sorry

end inequality_comparison_l18_18652


namespace constant_term_proof_l18_18793

noncomputable def constant_term_in_binomial_expansion (c : ℚ) (x : ℚ) : ℚ :=
  if h : (c = (2 : ℚ) - (1 / (8 * x^3))∧ x ≠ 0) then 
    28
  else 
    0

theorem constant_term_proof : 
  constant_term_in_binomial_expansion ((2 : ℚ) - (1 / (8 * (1 : ℚ)^3))) 1 = 28 := 
by
  sorry

end constant_term_proof_l18_18793


namespace dartboard_points_proof_l18_18627

variable (points_one points_two points_three points_four : ℕ)

theorem dartboard_points_proof
  (h1 : points_one = 30)
  (h2 : points_two = 38)
  (h3 : points_three = 41)
  (h4 : 2 * points_four = points_one + points_two) :
  points_four = 34 :=
by {
  sorry
}

end dartboard_points_proof_l18_18627


namespace first_group_checked_correctly_l18_18736

-- Define the given conditions
def total_factories : ℕ := 169
def checked_by_second_group : ℕ := 52
def remaining_unchecked : ℕ := 48

-- Define the number of factories checked by the first group
def checked_by_first_group : ℕ := total_factories - checked_by_second_group - remaining_unchecked

-- State the theorem to be proved
theorem first_group_checked_correctly : checked_by_first_group = 69 :=
by
  -- The proof is not provided, use sorry to skip the proof steps
  sorry

end first_group_checked_correctly_l18_18736


namespace find_b_l18_18411

theorem find_b
  (a b c d : ℝ)
  (h₁ : -a + b - c + d = 0)
  (h₂ : a + b + c + d = 0)
  (h₃ : d = 2) :
  b = -2 := 
by 
  sorry

end find_b_l18_18411


namespace chalk_breaking_probability_l18_18832

/-- Given you start with a single piece of chalk of length 1,
    and every second you choose a piece of chalk uniformly at random and break it in half,
    until you have 8 pieces of chalk,
    prove that the probability of all pieces having length 1/8 is 1/63. -/
theorem chalk_breaking_probability :
  let initial_pieces := 1
  let final_pieces := 8
  let total_breaks := final_pieces - initial_pieces
  let favorable_sequences := 20 * 4
  let total_sequences := Nat.factorial total_breaks
  (initial_pieces = 1) →
  (final_pieces = 8) →
  (total_breaks = 7) →
  (favorable_sequences = 80) →
  (total_sequences = 5040) →
  (favorable_sequences / total_sequences = 1 / 63) :=
by
  intros
  sorry

end chalk_breaking_probability_l18_18832


namespace painted_faces_cube_eq_54_l18_18228

def painted_faces (n : ℕ) : ℕ :=
  if n = 5 then (3 * 3) * 6 else 0

theorem painted_faces_cube_eq_54 : painted_faces 5 = 54 := by {
  sorry
}

end painted_faces_cube_eq_54_l18_18228


namespace total_fish_in_lake_l18_18298

-- Given conditions:
def initiallyTaggedFish : ℕ := 100
def capturedFish : ℕ := 100
def taggedFishInAugust : ℕ := 5
def taggedFishMortalityRate : ℝ := 0.3
def newcomerFishRate : ℝ := 0.2

-- Proof to show that the total number of fish at the beginning of April is 1120
theorem total_fish_in_lake (initiallyTaggedFish capturedFish taggedFishInAugust : ℕ) 
  (taggedFishMortalityRate newcomerFishRate : ℝ) : 
  (taggedFishInAugust : ℝ) / (capturedFish * (1 - newcomerFishRate)) = 
  ((initiallyTaggedFish * (1 - taggedFishMortalityRate)) : ℝ) / (1120 : ℝ) :=
by 
  sorry

end total_fish_in_lake_l18_18298


namespace alchemerion_age_problem_l18_18801

theorem alchemerion_age_problem 
  (A S F : ℕ)
  (h1 : A = 3 * S)
  (h2 : F = 2 * A + 40)
  (h3 : A = 360) :
  A + S + F = 1240 :=
by 
  sorry

end alchemerion_age_problem_l18_18801


namespace ratio_c_a_l18_18555

theorem ratio_c_a (a b c : ℚ) (h1 : a * b = 3) (h2 : b * c = 8 / 5) : c / a = 8 / 15 := 
by 
  sorry

end ratio_c_a_l18_18555


namespace no_such_function_exists_l18_18713

theorem no_such_function_exists :
  ¬(∃ (f : ℝ → ℝ), ∀ x y : ℝ, |f (x + y) + Real.sin x + Real.sin y| < 2) :=
sorry

end no_such_function_exists_l18_18713


namespace problem_l18_18941

def op (x y : ℝ) : ℝ := x^2 + y^3

theorem problem (k : ℝ) : op k (op k k) = k^2 + k^6 + 6*k^7 + k^9 :=
by
  sorry

end problem_l18_18941


namespace train_speed_kmph_l18_18316

noncomputable def train_speed_mps : ℝ := 60.0048

def conversion_factor : ℝ := 3.6

theorem train_speed_kmph : train_speed_mps * conversion_factor = 216.01728 := by
  sorry

end train_speed_kmph_l18_18316


namespace smallest_a_for_quadratic_poly_l18_18046

theorem smallest_a_for_quadratic_poly (a : ℕ) (a_pos : 0 < a) :
  (∃ b c : ℤ, ∀ x : ℝ, 0 < x ∧ x < 1 → a*x^2 + b*x + c = 0 → (2 : ℝ)^2 - (4 : ℝ)*(a * c) < 0 ∧ b^2 - 4*a*c ≥ 1) → a ≥ 5 := 
sorry

end smallest_a_for_quadratic_poly_l18_18046


namespace scientific_notation_of_0_0000007_l18_18810

theorem scientific_notation_of_0_0000007 :
  0.0000007 = 7 * 10 ^ (-7) :=
  by
  sorry

end scientific_notation_of_0_0000007_l18_18810


namespace shopkeeper_total_cards_l18_18633

-- Conditions
def num_standard_decks := 3
def cards_per_standard_deck := 52
def num_tarot_decks := 2
def cards_per_tarot_deck := 72
def num_trading_sets := 5
def cards_per_trading_set := 100
def additional_random_cards := 27

-- Calculate total cards
def total_standard_cards := num_standard_decks * cards_per_standard_deck
def total_tarot_cards := num_tarot_decks * cards_per_tarot_deck
def total_trading_cards := num_trading_sets * cards_per_trading_set
def total_cards := total_standard_cards + total_tarot_cards + total_trading_cards + additional_random_cards

-- Proof statement
theorem shopkeeper_total_cards : total_cards = 827 := by
    sorry

end shopkeeper_total_cards_l18_18633


namespace ratio_of_larger_to_smaller_l18_18973

theorem ratio_of_larger_to_smaller (a b : ℝ) (h : a > 0) (h' : b > 0) (h_sum_diff : a + b = 7 * (a - b)) :
  a / b = 4 / 3 :=
by
  sorry

end ratio_of_larger_to_smaller_l18_18973


namespace find_f_neg_two_l18_18234

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_f_neg_two (h : ∀ x : ℝ, x ≠ 0 → f (1 / x) + (1 / x) * f (-x) = 2 * x) :
  f (-2) = 7 / 2 :=
by
  sorry

end find_f_neg_two_l18_18234


namespace find_x_value_l18_18784

-- Define the conditions and the proof problem as Lean 4 statement
theorem find_x_value 
  (k : ℚ)
  (h1 : ∀ (x y : ℚ), (2 * x - 3) / (2 * y + 10) = k)
  (h2 : (2 * 4 - 3) / (2 * 5 + 10) = k)
  : (∃ x : ℚ, (2 * x - 3) / (2 * 10 + 10) = k) ↔ x = 5.25 :=
by
  sorry

end find_x_value_l18_18784


namespace series_value_l18_18778

noncomputable def sum_series (a b c : ℝ) (h_positivity : 0 < c ∧ 0 < b ∧ 0 < a) (h_order : a > b ∧ b > c) : ℝ :=
∑' n : ℕ, (if h : n > 0 then
             1 / (((n - 1) * c - (n - 2) * b) * (n * c - (n - 1) * a))
           else 
             0)

theorem series_value (a b c : ℝ) (h_positivity : 0 < c ∧ 0 < b ∧ 0 < a) (h_order : a > b ∧ b > c) :
  sum_series a b c h_positivity h_order = 1 / ((c - a) * b) :=
by
  sorry

end series_value_l18_18778


namespace fraction_spent_on_food_l18_18653

variable (salary : ℝ) (food_fraction rent_fraction clothes_fraction remaining_amount : ℝ)
variable (salary_condition : salary = 180000)
variable (rent_fraction_condition : rent_fraction = 1/10)
variable (clothes_fraction_condition : clothes_fraction = 3/5)
variable (remaining_amount_condition : remaining_amount = 18000)

theorem fraction_spent_on_food :
  rent_fraction * salary + clothes_fraction * salary + food_fraction * salary + remaining_amount = salary →
  food_fraction = 1/5 :=
by
  intros
  sorry

end fraction_spent_on_food_l18_18653


namespace reverse_geometric_diff_l18_18596

-- A digit must be between 0 and 9
def digit (d : ℕ) : Prop := d ≥ 0 ∧ d ≤ 9

-- Distinct digits
def distinct_digits (a b c : ℕ) : Prop := a ≠ b ∧ b ≠ c ∧ a ≠ c

-- Reverse geometric sequence 
def reverse_geometric (a b c : ℕ) : Prop := ∃ r : ℚ, b = c * r ∧ a = b * r

-- Check if abc forms a valid 3-digit reverse geometric sequence
def valid_reverse_geometric_number (a b c : ℕ) : Prop :=
  digit a ∧ digit b ∧ digit c ∧ distinct_digits a b c ∧ reverse_geometric a b c

theorem reverse_geometric_diff (a b c d e f : ℕ) 
  (h1: valid_reverse_geometric_number a b c) 
  (h2: valid_reverse_geometric_number d e f) :
  (a * 100 + b * 10 + c) - (d * 100 + e * 10 + f) = 789 :=
sorry

end reverse_geometric_diff_l18_18596


namespace tracy_customers_l18_18254

theorem tracy_customers
  (total_customers : ℕ)
  (customers_bought_two_each : ℕ)
  (customers_bought_one_each : ℕ)
  (customers_bought_four_each : ℕ)
  (total_paintings_sold : ℕ)
  (h1 : total_customers = 20)
  (h2 : customers_bought_one_each = 12)
  (h3 : customers_bought_four_each = 4)
  (h4 : total_paintings_sold = 36)
  (h5 : 2 * customers_bought_two_each + customers_bought_one_each + 4 * customers_bought_four_each = total_paintings_sold) :
  customers_bought_two_each = 4 :=
by
  sorry

end tracy_customers_l18_18254


namespace total_packets_needed_l18_18213

theorem total_packets_needed :
  let oak_seedlings := 420
  let oak_per_packet := 7
  let maple_seedlings := 825
  let maple_per_packet := 5
  let pine_seedlings := 2040
  let pine_per_packet := 12
  let oak_packets := oak_seedlings / oak_per_packet
  let maple_packets := maple_seedlings / maple_per_packet
  let pine_packets := pine_seedlings / pine_per_packet
  let total_packets := oak_packets + maple_packets + pine_packets
  total_packets = 395 := 
by {
  sorry
}

end total_packets_needed_l18_18213


namespace find_divisible_by_3_l18_18966

theorem find_divisible_by_3 (n : ℕ) : 
  (∀ k : ℕ, k ≤ 12 → (3 * k + 12) ≤ n) ∧ 
  (∀ m : ℕ, m ≥ 13 → (3 * m + 12) > n) →
  n = 48 :=
by
  sorry

end find_divisible_by_3_l18_18966


namespace part1_part2_l18_18447

-- Part 1: Definition of "consecutive roots quadratic equation"
def consecutive_roots (a b : ℤ) : Prop := a = b + 1 ∨ b = a + 1

-- Statement that for some k and constant term, the roots of the quadratic form consecutive roots
theorem part1 (k : ℤ) : consecutive_roots 7 8 → k = -15 → (∀ x : ℤ, x^2 + k * x + 56 = 0 → x = 7 ∨ x = 8) :=
by
  sorry

-- Part 2: Generalizing to the nth equation
theorem part2 (n : ℕ) : 
  (∀ x : ℤ, x^2 - (2 * n - 1) * x + n * (n - 1) = 0 → x = n ∨ x = n - 1) :=
by
  sorry

end part1_part2_l18_18447


namespace gumballs_per_package_l18_18560

theorem gumballs_per_package (total_gumballs : ℕ) (packages : ℝ) (h1 : total_gumballs = 100) (h2 : packages = 20.0) :
  total_gumballs / packages = 5 :=
by sorry

end gumballs_per_package_l18_18560


namespace coinsSold_l18_18578

-- Given conditions
def initialCoins : Nat := 250
def additionalCoins : Nat := 75
def coinsToKeep : Nat := 135

-- Theorem to prove
theorem coinsSold : (initialCoins + additionalCoins - coinsToKeep) = 190 := 
by
  -- Proof omitted 
  sorry

end coinsSold_l18_18578


namespace number_of_sunflowers_l18_18623

noncomputable def cost_per_red_rose : ℝ := 1.5
noncomputable def cost_per_sunflower : ℝ := 3
noncomputable def total_cost : ℝ := 45
noncomputable def cost_of_red_roses : ℝ := 24 * cost_per_red_rose
noncomputable def money_left_for_sunflowers : ℝ := total_cost - cost_of_red_roses

theorem number_of_sunflowers :
  (money_left_for_sunflowers / cost_per_sunflower) = 3 :=
by
  sorry

end number_of_sunflowers_l18_18623


namespace find_a_l18_18367

noncomputable def star (a b : ℝ) := a * (a + b) + b

theorem find_a (a : ℝ) (h : star a 2.5 = 28.5) : a = 4 ∨ a = -13/2 := 
sorry

end find_a_l18_18367


namespace largest_stores_visited_l18_18523

theorem largest_stores_visited 
  (stores : ℕ) (total_visits : ℕ) (shoppers : ℕ) 
  (two_store_visitors : ℕ) (min_visits_per_person : ℕ)
  (h1 : stores = 8)
  (h2 : total_visits = 22)
  (h3 : shoppers = 12)
  (h4 : two_store_visitors = 8)
  (h5 : min_visits_per_person = 1)
  : ∃ (max_stores : ℕ), max_stores = 3 := 
by 
  -- Define the exact details given in the conditions
  have h_total_two_store_visits : two_store_visitors * 2 = 16 := by sorry
  have h_remaining_visits : total_visits - 16 = 6 := by sorry
  have h_remaining_shoppers : shoppers - two_store_visitors = 4 := by sorry
  have h_each_remaining_one_visit : 4 * 1 = 4 := by sorry
  -- Prove the largest number of stores visited by any one person is 3
  have h_max_stores : 1 + 2 = 3 := by sorry
  exact ⟨3, h_max_stores⟩

end largest_stores_visited_l18_18523


namespace Jackie_apples_count_l18_18828

variable (Adam_apples Jackie_apples : ℕ)

-- Conditions
axiom Adam_has_14_apples : Adam_apples = 14
axiom Adam_has_5_more_than_Jackie : Adam_apples = Jackie_apples + 5

-- Theorem to prove
theorem Jackie_apples_count : Jackie_apples = 9 := by
  -- Use the conditions to derive the answer
  sorry

end Jackie_apples_count_l18_18828


namespace sum_of_areas_squares_l18_18959

theorem sum_of_areas_squares (a : ℝ) : 
  (∑' n : ℕ, (a^2 / 4^n)) = (4 * a^2 / 3) :=
by
  sorry

end sum_of_areas_squares_l18_18959


namespace relationship_between_x_y_z_l18_18201

noncomputable def x := Real.sqrt 0.82
noncomputable def y := Real.sin 1
noncomputable def z := Real.log 7 / Real.log 3

theorem relationship_between_x_y_z : y < z ∧ z < x := 
by sorry

end relationship_between_x_y_z_l18_18201


namespace total_number_of_plugs_l18_18302

variables (pairs_mittens pairs_plugs : ℕ)

-- Conditions
def initial_pairs_mittens : ℕ := 150
def initial_pairs_plugs : ℕ := initial_pairs_mittens + 20
def added_pairs_plugs : ℕ := 30
def total_pairs_plugs : ℕ := initial_pairs_plugs + added_pairs_plugs

-- The proposition we're going to prove:
theorem total_number_of_plugs : initial_pairs_mittens = 150 ∧ initial_pairs_plugs = initial_pairs_mittens + 20 ∧ added_pairs_plugs = 30 → 
  total_pairs_plugs * 2 = 400 := sorry

end total_number_of_plugs_l18_18302


namespace num_lines_in_grid_l18_18426

theorem num_lines_in_grid (columns rows : ℕ) (H1 : columns = 4) (H2 : rows = 3) 
    (total_points : ℕ) (H3 : total_points = columns * rows) :
    ∃ lines, lines = 40 :=
by
  sorry

end num_lines_in_grid_l18_18426


namespace sum_of_y_neg_l18_18499

-- Define the conditions from the problem
def condition1 (x y : ℝ) : Prop := x + y = 7
def condition2 (x z : ℝ) : Prop := x * z = -180
def condition3 (x y z : ℝ) : Prop := (x + y + z)^2 = 4

-- Define the main theorem to prove
theorem sum_of_y_neg (x y z : ℝ) (S : ℝ) :
  (condition1 x y) ∧ (condition2 x z) ∧ (condition3 x y z) →
  (S = (-29) + (-13)) →
  -S = 42 :=
by
  sorry

end sum_of_y_neg_l18_18499


namespace gcd_gx_x_is_450_l18_18611

def g (x : ℕ) : ℕ := (3 * x + 2) * (8 * x + 3) * (14 * x + 5) * (x + 15)

noncomputable def gcd_gx_x (x : ℕ) (h : 49356 ∣ x) : ℕ :=
  Nat.gcd (g x) x

theorem gcd_gx_x_is_450 (x : ℕ) (h : 49356 ∣ x) : gcd_gx_x x h = 450 := by
  sorry

end gcd_gx_x_is_450_l18_18611


namespace line_parallel_to_x_axis_l18_18057

variable (k : ℝ)

theorem line_parallel_to_x_axis :
  let point1 := (3, 2 * k + 1)
  let point2 := (8, 4 * k - 5)
  (point1.2 = point2.2) ↔ (k = 3) :=
by
  sorry

end line_parallel_to_x_axis_l18_18057


namespace cos_14_pi_over_3_l18_18380

theorem cos_14_pi_over_3 : Real.cos (14 * Real.pi / 3) = -1 / 2 :=
by 
  -- Proof is omitted according to the instructions
  sorry

end cos_14_pi_over_3_l18_18380


namespace rodney_lift_l18_18719

theorem rodney_lift :
  ∃ (Ry : ℕ), 
  (∃ (Re R Ro : ℕ), 
  Re + Ry + R + Ro = 450 ∧
  Ry = 2 * R ∧
  R = Ro + 5 ∧
  Re = 3 * Ro - 20 ∧
  20 ≤ Ry ∧ Ry ≤ 200 ∧
  20 ≤ R ∧ R ≤ 200 ∧
  20 ≤ Ro ∧ Ro ≤ 200 ∧
  20 ≤ Re ∧ Re ≤ 200) ∧
  Ry = 140 :=
by
  sorry

end rodney_lift_l18_18719


namespace base9_39457_to_base10_is_26620_l18_18741

-- Define the components of the base 9 number 39457_9
def base9_39457 : ℕ := 39457
def base9_digits : List ℕ := [3, 9, 4, 5, 7]

-- Define the base
def base : ℕ := 9

-- Convert each position to its base 10 equivalent
def base9_to_base10 : ℕ :=
  3 * base ^ 4 + 9 * base ^ 3 + 4 * base ^ 2 + 5 * base ^ 1 + 7 * base ^ 0

-- State the theorem
theorem base9_39457_to_base10_is_26620 : base9_to_base10 = 26620 := by
  sorry

end base9_39457_to_base10_is_26620_l18_18741


namespace average_price_of_5_baskets_l18_18702

/-- Saleem bought 4 baskets with an average cost of $4 each. --/
def average_cost_first_4_baskets : ℝ := 4

/-- Saleem buys the fifth basket with the price of $8. --/
def price_fifth_basket : ℝ := 8

/-- Prove that the average price of the 5 baskets is $4.80. --/
theorem average_price_of_5_baskets :
  (4 * average_cost_first_4_baskets + price_fifth_basket) / 5 = 4.80 := 
by
  sorry

end average_price_of_5_baskets_l18_18702


namespace find_integers_satisfying_condition_l18_18171

-- Define the inequality condition
def condition (x : ℤ) : Prop := x * x < 3 * x

-- Prove that the set of integers satisfying the condition is {1, 2}
theorem find_integers_satisfying_condition :
  { x : ℤ | condition x } = {1, 2} := 
by {
  sorry
}

end find_integers_satisfying_condition_l18_18171


namespace sum_xyz_l18_18958

theorem sum_xyz (x y z : ℝ) (h : (x - 2)^2 + (y - 3)^2 + (z - 6)^2 = 0) : x + y + z = 11 := 
by
  sorry

end sum_xyz_l18_18958


namespace max_buses_in_city_l18_18525

theorem max_buses_in_city (n : ℕ) (stops : ℕ) (shared : ℕ) (condition1 : n = 9) (condition2 : stops = 3) (condition3 : shared ≤ 1) : n = 12 :=
sorry

end max_buses_in_city_l18_18525


namespace inequality_proof_l18_18327

variable {a b c : ℝ}

theorem inequality_proof (h1: a > 0) (h2: b > 0) (h3: c > 0) (h4: a^2 = b^2 + c^2) :
  a^3 + b^3 + c^3 ≥ (2*Real.sqrt 2 + 1) / 7 * (a^2 * (b + c) + b^2 * (c + a) + c^2 * (a + b)) := 
sorry

end inequality_proof_l18_18327


namespace total_ice_cream_volume_l18_18643

def cone_height : ℝ := 10
def cone_radius : ℝ := 1.5
def cylinder_height : ℝ := 2
def cylinder_radius : ℝ := 1.5
def hemisphere_radius : ℝ := 1.5

theorem total_ice_cream_volume : 
  (1 / 3 * π * cone_radius ^ 2 * cone_height) +
  (π * cylinder_radius ^ 2 * cylinder_height) +
  (2 / 3 * π * hemisphere_radius ^ 3) = 14.25 * π :=
by sorry

end total_ice_cream_volume_l18_18643


namespace distance_traveled_eq_2400_l18_18364

-- Definitions of the conditions
def circumference_front : ℕ := 30
def circumference_back : ℕ := 32
def revolutions_difference : ℕ := 5

-- Define the number of revolutions made by the back wheel
def revs_back (R : ℕ) := R

-- Define the number of revolutions made by the front wheel
def revs_front (R : ℕ) := R + revolutions_difference

-- Define the distance traveled by the back and front wheels
def distance_back (R : ℕ) : ℕ := revs_back R * circumference_back
def distance_front (R : ℕ) : ℕ := revs_front R * circumference_front

-- State the theorem without a proof (using sorry)
theorem distance_traveled_eq_2400 :
  ∃ R : ℕ, distance_back R = 2400 ∧ distance_back R = distance_front R :=
by {
  sorry
}

end distance_traveled_eq_2400_l18_18364


namespace probability_of_different_colors_l18_18162

noncomputable def total_chips := 6 + 5 + 4

noncomputable def prob_diff_color : ℚ :=
  let pr_blue := 6 / total_chips
  let pr_red := 5 / total_chips
  let pr_yellow := 4 / total_chips

  let pr_not_blue := (5 + 4) / total_chips
  let pr_not_red := (6 + 4) / total_chips
  let pr_not_yellow := (6 + 5) / total_chips

  pr_blue * pr_not_blue + pr_red * pr_not_red + pr_yellow * pr_not_yellow

theorem probability_of_different_colors :
  prob_diff_color = 148 / 225 :=
sorry

end probability_of_different_colors_l18_18162


namespace trigonometric_identity_l18_18183

theorem trigonometric_identity :
  Real.cos (70 * Real.pi / 180) * Real.sin (80 * Real.pi / 180) +
  Real.cos (20 * Real.pi / 180) * Real.sin (10 * Real.pi / 180) = 1 / 2 :=
by
  sorry

end trigonometric_identity_l18_18183


namespace initial_depth_dug_l18_18144

theorem initial_depth_dug :
  (∀ days : ℕ, 75 * 8 * days / D = 140 * 6 * days / 70) → D = 50 :=
by
  sorry

end initial_depth_dug_l18_18144


namespace fedora_cleaning_time_l18_18294

-- Definitions based on given conditions
def cleaning_time_per_section (total_time sections_cleaned : ℕ) : ℕ :=
  total_time / sections_cleaned

def remaining_sections (total_sections cleaned_sections : ℕ) : ℕ :=
  total_sections - cleaned_sections

def total_cleaning_time (remaining_sections time_per_section : ℕ) : ℕ :=
  remaining_sections * time_per_section

-- Theorem statement
theorem fedora_cleaning_time 
  (total_time : ℕ) 
  (sections_cleaned : ℕ)
  (additional_time : ℕ)
  (additional_sections : ℕ)
  (cleaned_sections : ℕ)
  (total_sections : ℕ)
  (h1 : total_time = 33)
  (h2 : sections_cleaned = 3)
  (h3 : additional_time = 165)
  (h4 : additional_sections = 15)
  (h5 : cleaned_sections = 3)
  (h6 : total_sections = 18)
  (h7 : cleaning_time_per_section total_time sections_cleaned = 11)
  (h8 : remaining_sections total_sections cleaned_sections = additional_sections)
  : total_cleaning_time additional_sections (cleaning_time_per_section total_time sections_cleaned) = additional_time := sorry

end fedora_cleaning_time_l18_18294


namespace a5_value_l18_18352

variable (a : ℕ → ℤ) (S : ℕ → ℤ)

-- Assume the sequence is arithmetic
def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

-- Given conditions
axiom sum_S6 : S 6 = 12
axiom term_a2 : a 2 = 5
axiom sum_formula (n : ℕ) : S n = n * (a 1 + a n) / 2

-- Prove a5 is -1
theorem a5_value (h_arith : arithmetic_sequence a)
  (h_S6 : S 6 = 12) (h_a2 : a 2 = 5) (h_sum_formula : ∀ n, S n = n * (a 1 + a n) / 2) :
  a 5 = -1 :=
sorry

end a5_value_l18_18352


namespace num_signs_in_sign_language_l18_18869

theorem num_signs_in_sign_language (n : ℕ) (h : n^2 - (n - 2)^2 = 888) : n = 223 := 
sorry

end num_signs_in_sign_language_l18_18869


namespace largest_possible_n_l18_18946

open Nat

-- Define arithmetic sequences a_n and b_n with given initial conditions
def arithmetic_seq (a_n : ℕ → ℕ) (b_n : ℕ → ℕ) :=
  a_n 1 = 1 ∧ b_n 1 = 1 ∧ 
  a_n 2 ≤ b_n 2 ∧
  (∃n : ℕ, a_n n * b_n n = 1764)

-- Given the arithmetic sequences defined above, prove that the largest possible value of n is 44
theorem largest_possible_n : 
  ∀ (a_n b_n : ℕ → ℕ), arithmetic_seq a_n b_n →
  ∀ (n : ℕ), (a_n n * b_n n = 1764) → n ≤ 44 :=
sorry

end largest_possible_n_l18_18946


namespace least_positive_integer_a_l18_18391

theorem least_positive_integer_a (a : ℕ) (n : ℕ) 
  (h1 : 2001 = 3 * 23 * 29)
  (h2 : 55 % 3 = 1)
  (h3 : 32 % 3 = -1)
  (h4 : 55 % 23 = 32 % 23)
  (h5 : 55 % 29 = -32 % 29)
  (h6 : n % 2 = 1)
  : a = 436 := 
sorry

end least_positive_integer_a_l18_18391


namespace dog_catches_fox_at_120m_l18_18381

theorem dog_catches_fox_at_120m :
  let initial_distance := 30
  let dog_leap := 2
  let fox_leap := 1
  let dog_leap_frequency := 2
  let fox_leap_frequency := 3
  let dog_distance_per_time_unit := dog_leap * dog_leap_frequency
  let fox_distance_per_time_unit := fox_leap * fox_leap_frequency
  let relative_closure_rate := dog_distance_per_time_unit - fox_distance_per_time_unit
  let time_units_to_catch := initial_distance / relative_closure_rate
  let total_dog_distance := time_units_to_catch * dog_distance_per_time_unit
  total_dog_distance = 120 := sorry

end dog_catches_fox_at_120m_l18_18381


namespace sum_geom_seq_nine_l18_18339

noncomputable def geom_seq (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n : ℕ, a (n + 1) = a n * q

theorem sum_geom_seq_nine {a : ℕ → ℝ} {q : ℝ} (h_geom : geom_seq a q)
  (h1 : a 1 * (1 + q + q^2) = 30) 
  (h2 : a 4 * (1 + q + q^2) = 120) :
  a 7 + a 8 + a 9 = 480 :=
  sorry

end sum_geom_seq_nine_l18_18339


namespace find_x1_l18_18631

theorem find_x1 (x1 x2 x3 x4 : ℝ) 
  (h1 : 0 ≤ x4) 
  (h2 : x4 ≤ x3) 
  (h3 : x3 ≤ x2) 
  (h4 : x2 ≤ x1) 
  (h5 : x1 ≤ 1) 
  (condition : (1 - x1)^2 + (x1 - x2)^2 + (x2 - x3)^2 + (x3 - x4)^2 + x4^2 = 1 / 5) : 
  x1 = 4 / 5 := 
sorry

end find_x1_l18_18631


namespace set_intersection_and_polynomial_solution_l18_18180

theorem set_intersection_and_polynomial_solution {a b : ℝ} :
  let A := {x : ℝ | x + 2 < 0}
  let B := {x : ℝ | (x + 3) * (x - 1) > 0}
  (A ∩ B = {x | x < -3}) ∧ ((A ∪ B = {x | x < -2 ∨ x > 1}) →
    (a = 2 ∧ b = -4)) :=
by
  let A := {x : ℝ | x + 2 < 0}
  let B := {x : ℝ | (x + 3) * (x - 1) > 0}
  sorry

end set_intersection_and_polynomial_solution_l18_18180


namespace solve_quadratic_eq_l18_18885

theorem solve_quadratic_eq (x : ℝ) :
  x^2 + 4 * x + 2 = 0 ↔ (x = -2 + Real.sqrt 2 ∨ x = -2 - Real.sqrt 2) :=
by
  -- This is a statement only. No proof is required.
  sorry

end solve_quadratic_eq_l18_18885


namespace min_points_in_set_M_l18_18485
-- Import the necessary library

-- Define the problem conditions and the result to prove
theorem min_points_in_set_M :
  ∃ (M : Finset ℝ) (C₁ C₂ C₃ C₄ C₅ C₆ C₇ : Finset ℝ),
  C₇.card = 7 ∧
  C₆.card = 6 ∧
  C₅.card = 5 ∧
  C₄.card = 4 ∧
  C₃.card = 3 ∧
  C₂.card = 2 ∧
  C₁.card = 1 ∧
  C₇ ⊆ M ∧
  C₆ ⊆ M ∧
  C₅ ⊆ M ∧
  C₄ ⊆ M ∧
  C₃ ⊆ M ∧
  C₂ ⊆ M ∧
  C₁ ⊆ M ∧
  M.card = 12 :=
sorry

end min_points_in_set_M_l18_18485


namespace axis_of_symmetry_imp_cond_l18_18039

-- Necessary definitions
variables {p q r s x y : ℝ}

-- Given conditions
def curve_eq (x y p q r s : ℝ) : Prop := y = (2 * p * x + q) / (r * x + 2 * s)
def axis_of_symmetry (x y : ℝ) : Prop := y = x

-- Main statement
theorem axis_of_symmetry_imp_cond (h1 : curve_eq x y p q r s) (h2 : axis_of_symmetry x y) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (hs : s ≠ 0) : p = -2 * s :=
sorry

end axis_of_symmetry_imp_cond_l18_18039


namespace find_inverse_of_25_l18_18224

-- Define the inverses and the modulo
def inverse_mod (a m i : ℤ) : Prop :=
  (a * i) % m = 1

-- The given condition in the problem
def condition (m : ℤ) : Prop :=
  inverse_mod 5 m 39

-- The theorem we want to prove
theorem find_inverse_of_25 (m : ℤ) (h : condition m) : inverse_mod 25 m 8 :=
by
  sorry

end find_inverse_of_25_l18_18224


namespace distance_between_parallel_lines_l18_18442

-- Definitions of lines l_1 and l_2
def line_l1 (x y : ℝ) : Prop := 3*x + 4*y - 2 = 0
def line_l2 (x y : ℝ) : Prop := 6*x + 8*y - 5 = 0

-- Proof statement that the distance between the two lines is 1/10
theorem distance_between_parallel_lines (x y : ℝ) :
  ∃ d : ℝ, d = 1/10 ∧ ∀ p : ℝ × ℝ,
  (line_l1 p.1 p.2 ∧ line_l2 p.1 p.2 → p = (x, y)) :=
sorry

end distance_between_parallel_lines_l18_18442


namespace picture_distance_l18_18630

theorem picture_distance (w t s p d : ℕ) (h1 : w = 25) (h2 : t = 2) (h3 : s = 1) (h4 : 2 * p + s = t + s + t) 
  (h5 : w = 2 * d + p) : d = 10 :=
by
  sorry

end picture_distance_l18_18630


namespace sum_of_ages_equal_to_grandpa_l18_18673

-- Conditions
def grandpa_age : Nat := 75
def grandchild_age_1 : Nat := 13
def grandchild_age_2 : Nat := 15
def grandchild_age_3 : Nat := 17

-- Main Statement
theorem sum_of_ages_equal_to_grandpa (t : Nat) :
  (grandchild_age_1 + t) + (grandchild_age_2 + t) + (grandchild_age_3 + t) = grandpa_age + t 
  ↔ t = 15 := 
by {
  sorry
}

end sum_of_ages_equal_to_grandpa_l18_18673


namespace add_and_simplify_fractions_l18_18987

theorem add_and_simplify_fractions :
  (1 / 462) + (23 / 42) = 127 / 231 :=
by
  sorry

end add_and_simplify_fractions_l18_18987


namespace stella_dolls_count_l18_18306

variables (D : ℕ) (clocks glasses P_doll P_clock P_glass cost profit : ℕ)

theorem stella_dolls_count (h_clocks : clocks = 2)
                     (h_glasses : glasses = 5)
                     (h_P_doll : P_doll = 5)
                     (h_P_clock : P_clock = 15)
                     (h_P_glass : P_glass = 4)
                     (h_cost : cost = 40)
                     (h_profit : profit = 25) :
  D = 3 :=
by sorry

end stella_dolls_count_l18_18306


namespace central_angle_radian_measure_l18_18875

-- Definitions for the conditions
def circumference (r l : ℝ) : Prop := 2 * r + l = 8
def area (r l : ℝ) : Prop := (1/2) * l * r = 4
def radian_measure (l r θ : ℝ) : Prop := θ = l / r

-- Prove the radian measure of the central angle of the sector is 2
theorem central_angle_radian_measure (r l θ : ℝ) : 
  circumference r l → 
  area r l → 
  radian_measure l r θ → 
  θ = 2 :=
by
  sorry

end central_angle_radian_measure_l18_18875


namespace geometric_series_sum_test_l18_18731

-- Let's define all necessary variables
variable (a : ℤ) (r : ℤ) (n : ℕ)

-- Define the geometric series sum formula
noncomputable def geometric_series_sum (a r : ℤ) (n : ℕ) : ℤ :=
  a * ((r ^ n - 1) / (r - 1))

-- Define the specific test case as per our conditions
theorem geometric_series_sum_test :
  geometric_series_sum (-2) 3 7 = -2186 :=
by
  sorry

end geometric_series_sum_test_l18_18731


namespace tan_alpha_plus_pi_div_four_l18_18825

theorem tan_alpha_plus_pi_div_four (α : ℝ) (h : (3 * Real.sin α + 2 * Real.cos α) / (2 * Real.sin α - Real.cos α) = 8 / 3) : 
  Real.tan (α + Real.pi / 4) = -3 := 
by 
  sorry

end tan_alpha_plus_pi_div_four_l18_18825


namespace tony_comics_average_l18_18343

theorem tony_comics_average :
  let a1 := 10
  let d := 6
  let n := 8
  let a_n (n : ℕ) := a1 + (n - 1) * d
  let S_n (n : ℕ) := n / 2 * (a1 + a_n n)
  (S_n n) / n = 31 := by
  sorry

end tony_comics_average_l18_18343


namespace sum_first_5_terms_arithmetic_l18_18417

variable {a : ℕ → ℝ} -- Defining a sequence

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

-- Given conditions
axiom a2_eq_1 : a 2 = 1
axiom a4_eq_5 : a 4 = 5

-- Theorem statement
theorem sum_first_5_terms_arithmetic (h_arith : is_arithmetic_sequence a) : 
  a 1 + a 2 + a 3 + a 4 + a 5 = 15 := by
  sorry

end sum_first_5_terms_arithmetic_l18_18417


namespace intersection_A_B_l18_18285

def A : Set ℤ := { -2, -1, 0, 1, 2 }
def B : Set ℤ := { x : ℤ | x < 1 }

theorem intersection_A_B : A ∩ B = { -2, -1, 0 } :=
by sorry

end intersection_A_B_l18_18285


namespace abs_diff_51st_terms_correct_l18_18775

-- Definition of initial conditions for sequences A and C
def seqA_first_term : ℤ := 40
def seqA_common_difference : ℤ := 8

def seqC_first_term : ℤ := 40
def seqC_common_difference : ℤ := -5

-- Definition of the nth term function for an arithmetic sequence
def nth_term (a₁ d n : ℤ) : ℤ := a₁ + d * (n - 1)

-- 51st term of sequence A
def a_51 : ℤ := nth_term seqA_first_term seqA_common_difference 51

-- 51st term of sequence C
def c_51 : ℤ := nth_term seqC_first_term seqC_common_difference 51

-- Absolute value of the difference
def abs_diff_51st_terms : ℤ := Int.natAbs (a_51 - c_51)

-- The theorem to be proved
theorem abs_diff_51st_terms_correct : abs_diff_51st_terms = 650 := by
  sorry

end abs_diff_51st_terms_correct_l18_18775


namespace friedEdgeProb_l18_18025

-- Define a data structure for positions on the grid
inductive Pos
| A1 | A2 | A3 | A4
| B1 | B2 | B3 | B4
| C1 | C2 | C3 | C4
| D1 | D2 | D3 | D4
deriving DecidableEq, Repr

-- Define whether a position is an edge square (excluding corners)
def isEdge : Pos → Prop
| Pos.A2 | Pos.A3 | Pos.B1 | Pos.B4 | Pos.C1 | Pos.C4 | Pos.D2 | Pos.D3 => True
| _ => False

-- Define the initial state and max hops
def initialState := Pos.B2
def maxHops := 5

-- Define the recursive probability function (details omitted for brevity)
noncomputable def probabilityEdge (p : Pos) (hops : Nat) : ℚ := sorry

-- The proof problem statement
theorem friedEdgeProb :
  probabilityEdge initialState maxHops = 94 / 256 := sorry

end friedEdgeProb_l18_18025


namespace sector_area_l18_18366

-- Define radius and central angle as conditions
def radius : ℝ := 1
def central_angle : ℝ := 2

-- Define the theorem to prove that the area of the sector is 1 cm² given the conditions
theorem sector_area : (1 / 2) * radius * central_angle = 1 := 
by 
  -- sorry is used to skip the actual proof
  sorry

end sector_area_l18_18366


namespace no_solution_prob1_l18_18665

theorem no_solution_prob1 : ¬ ∃ x : ℝ, x ≠ 2 ∧ (1 / (x - 2) + 3 = (1 - x) / (2 - x)) :=
by
  sorry

end no_solution_prob1_l18_18665


namespace subtraction_result_l18_18768

noncomputable def division_value : ℝ := 1002 / 20.04

theorem subtraction_result : 2500 - division_value = 2450.0499 :=
by
  have division_eq : division_value = 49.9501 := by sorry
  rw [division_eq]
  norm_num

end subtraction_result_l18_18768


namespace find_solution_set_l18_18319

noncomputable def is_solution (x : ℝ) : Prop :=
(1 / (x * (x + 1))) - (1 / ((x + 1) * (x + 2))) < 1 / 4

theorem find_solution_set :
  { x : ℝ | is_solution x } = { x : ℝ | x < -2 } ∪ { x : ℝ | -1 < x ∧ x < 0 } ∪ { x : ℝ | x > 1 } :=
by
  sorry

end find_solution_set_l18_18319


namespace max_hawthorns_satisfying_conditions_l18_18763

theorem max_hawthorns_satisfying_conditions :
  ∃ x : ℕ, 
    x > 100 ∧ 
    x % 3 = 1 ∧ 
    x % 4 = 2 ∧ 
    x % 5 = 3 ∧ 
    x % 6 = 4 ∧ 
    (∀ y : ℕ, 
      y > 100 ∧ 
      y % 3 = 1 ∧ 
      y % 4 = 2 ∧ 
      y % 5 = 3 ∧ 
      y % 6 = 4 → y ≤ 178) :=
sorry

end max_hawthorns_satisfying_conditions_l18_18763


namespace c_n_monotonically_decreasing_l18_18524

variable (a : ℕ → ℝ) (b : ℕ → ℝ) (c : ℕ → ℝ)

theorem c_n_monotonically_decreasing 
    (h_a0 : a 0 = 0)
    (h_b : ∀ n ≥ 1, b n = a n - a (n - 1))
    (h_c : ∀ n ≥ 1, c n = a n / n)
    (h_bn_decrease : ∀ n ≥ 1, b n ≥ b (n + 1)) : 
    ∀ n ≥ 2, c n ≤ c (n - 1) := 
by
  sorry

end c_n_monotonically_decreasing_l18_18524


namespace can_spend_all_money_l18_18628

theorem can_spend_all_money (n : Nat) (h : n > 7) : 
  ∃ (x y : Nat), 3 * x + 5 * y = n :=
by
  sorry

end can_spend_all_money_l18_18628


namespace jon_and_mary_frosting_l18_18900

-- Jon frosts a cupcake every 40 seconds
def jon_frost_rate : ℚ := 1 / 40

-- Mary frosts a cupcake every 24 seconds
def mary_frost_rate : ℚ := 1 / 24

-- Combined frosting rate of Jon and Mary
def combined_frost_rate : ℚ := jon_frost_rate + mary_frost_rate

-- Total time in seconds for 12 minutes
def total_time_seconds : ℕ := 12 * 60

-- Calculate the total number of cupcakes frosted in 12 minutes
def total_cupcakes_frosted (time_seconds : ℕ) (rate : ℚ) : ℚ :=
  time_seconds * rate

theorem jon_and_mary_frosting : total_cupcakes_frosted total_time_seconds combined_frost_rate = 48 := by
  sorry

end jon_and_mary_frosting_l18_18900


namespace ratio_girls_to_boys_l18_18929

-- Definitions of the conditions
def numGirls : ℕ := 10
def numBoys : ℕ := 20

-- Statement of the proof problem
theorem ratio_girls_to_boys : (numGirls / Nat.gcd numGirls numBoys) = 1 ∧ (numBoys / Nat.gcd numGirls numBoys) = 2 :=
by
  sorry

end ratio_girls_to_boys_l18_18929


namespace range_of_a_for_real_roots_l18_18783

theorem range_of_a_for_real_roots (a : ℝ) (h : a ≠ 0) :
  (∃ (x : ℝ), a*x^2 + 2*x + 1 = 0) ↔ a ≤ 1 :=
by
  sorry

end range_of_a_for_real_roots_l18_18783


namespace factorize_expr_l18_18913

theorem factorize_expr (y : ℝ) : 3 * y ^ 2 - 6 * y + 3 = 3 * (y - 1) ^ 2 :=
by
  sorry

end factorize_expr_l18_18913


namespace prove_zero_l18_18288

variable {a b c : ℝ}

theorem prove_zero (h : a / (b - c) + b / (c - a) + c / (a - b) = 0) :
  a / (b - c) ^ 2 + b / (c - a) ^ 2 + c / (a - b) ^ 2 = 0 :=
by
  sorry

end prove_zero_l18_18288


namespace range_of_real_number_l18_18134

theorem range_of_real_number (a : ℝ) : (a > 0) ∧ (a - 1 > 0) → a > 1 :=
by
  sorry

end range_of_real_number_l18_18134


namespace find_value_of_fraction_l18_18506

noncomputable def a : ℝ := 5 * (Real.sqrt 2) + 7

theorem find_value_of_fraction (h : (20 * a) / (a^2 + 1) = Real.sqrt 2) (h1 : 1 < a) : 
  (14 * a) / (a^2 - 1) = 1 := 
by 
  have h_sqrt : 20 * a = Real.sqrt 2 * a^2 + Real.sqrt 2 := by sorry
  have h_rearrange : Real.sqrt 2 * a^2 - 20 * a + Real.sqrt 2 = 0 := by sorry
  have h_solution : a = 5 * (Real.sqrt 2) + 7 := by sorry
  have h_asquare : a^2 = 99 + 70 * (Real.sqrt 2) := by sorry
  exact sorry

end find_value_of_fraction_l18_18506


namespace highest_average_speed_interval_l18_18939

theorem highest_average_speed_interval
  (d : ℕ → ℕ)
  (h0 : d 0 = 45)        -- Distance from 0 to 30 minutes
  (h1 : d 1 = 135)       -- Distance from 30 to 60 minutes
  (h2 : d 2 = 255)       -- Distance from 60 to 90 minutes
  (h3 : d 3 = 325) :     -- Distance from 90 to 120 minutes
  (1 / 2) * ((d 2 - d 1 : ℕ) : ℝ) > 
  max ((1 / 2) * ((d 1 - d 0 : ℕ) : ℝ)) 
      (max ((1 / 2) * ((d 3 - d 2 : ℕ) : ℝ))
          ((1 / 2) * ((d 3 - d 1 : ℕ) : ℝ))) :=
by
  sorry

end highest_average_speed_interval_l18_18939


namespace parallel_lines_condition_l18_18937

theorem parallel_lines_condition (a : ℝ) : 
  (∀ x y : ℝ, ax + y + 1 = 0 ↔ x + ay - 1 = 0) ↔ (a = 1) :=
sorry

end parallel_lines_condition_l18_18937


namespace sum_arithmetic_sequence_l18_18332

theorem sum_arithmetic_sequence :
  let a : ℤ := -25
  let d : ℤ := 4
  let a_n : ℤ := 19
  let n : ℤ := (a_n - a) / d + 1
  let S : ℤ := n * (a + a_n) / 2
  S = -36 :=
by 
  let a := -25
  let d := 4
  let a_n := 19
  let n := (a_n - a) / d + 1
  let S := n * (a + a_n) / 2
  show S = -36
  sorry

end sum_arithmetic_sequence_l18_18332


namespace min_value_expression_l18_18272

theorem min_value_expression (x1 x2 x3 x4 : ℝ) (h1 : 0 < x1) (h2 : 0 < x2) (h3 : 0 < x3) (h4 : 0 < x4) (h_sum : x1 + x2 + x3 + x4 = Real.pi) :
  (2 * (Real.sin x1) ^ 2 + 1 / (Real.sin x1) ^ 2) *
  (2 * (Real.sin x2) ^ 2 + 1 / (Real.sin x2) ^ 2) *
  (2 * (Real.sin x3) ^ 2 + 1 / (Real.sin x3) ^ 2) *
  (2 * (Real.sin x4) ^ 2 + 1 / (Real.sin x4) ^ 2) = 81 := 
sorry

end min_value_expression_l18_18272


namespace curve_in_second_quadrant_l18_18181

theorem curve_in_second_quadrant (a : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 + 2*a*x - 4*a*y + 5*a^2 - 4 = 0 → x < 0 ∧ y > 0) ↔ (a > 2) :=
sorry

end curve_in_second_quadrant_l18_18181


namespace jane_number_of_muffins_l18_18341

theorem jane_number_of_muffins 
    (m b c : ℕ) 
    (h1 : m + b + c = 6) 
    (h2 : b = 2) 
    (h3 : (50 * m + 75 * b + 65 * c) % 100 = 0) : 
    m = 4 := 
sorry

end jane_number_of_muffins_l18_18341


namespace area_of_field_l18_18492

theorem area_of_field : ∀ (L W : ℕ), L = 20 → L + 2 * W = 88 → L * W = 680 :=
by
  intros L W hL hEq
  rw [hL] at hEq
  sorry

end area_of_field_l18_18492


namespace smallest_positive_angle_l18_18503

theorem smallest_positive_angle (theta : ℝ) (h_theta : theta = -2002) :
  ∃ α : ℝ, 0 < α ∧ α < 360 ∧ ∃ k : ℤ, theta = α + k * 360 ∧ α = 158 := 
by
  sorry

end smallest_positive_angle_l18_18503


namespace given_equation_roots_sum_cubes_l18_18189

theorem given_equation_roots_sum_cubes (r s t : ℝ) 
    (h1 : 6 * r ^ 3 + 1506 * r + 3009 = 0)
    (h2 : 6 * s ^ 3 + 1506 * s + 3009 = 0)
    (h3 : 6 * t ^ 3 + 1506 * t + 3009 = 0)
    (sum_roots : r + s + t = 0) :
    (r + s) ^ 3 + (s + t) ^ 3 + (t + r) ^ 3 = 1504.5 := 
by 
  -- proof omitted
  sorry

end given_equation_roots_sum_cubes_l18_18189


namespace right_triangle_third_side_square_l18_18300

theorem right_triangle_third_side_square (a b : ℕ) (c : ℕ) (h₁ : a = 6) (h₂ : b = 8) (h₃ : c^2 = a^2 + b^2 ∨ a^2 = b^2 + c^2 ∨ b^2 = a^2 + c^2) :
  c^2 = 28 ∨ c^2 = 100 :=
by { sorry }

end right_triangle_third_side_square_l18_18300


namespace number_of_cans_per_set_l18_18573

noncomputable def ice_cream_original_price : ℝ := 12
noncomputable def ice_cream_discount : ℝ := 2
noncomputable def ice_cream_sale_price : ℝ := ice_cream_original_price - ice_cream_discount
noncomputable def number_of_tubs : ℝ := 2
noncomputable def total_money_spent : ℝ := 24
noncomputable def cost_of_juice_set : ℝ := 2
noncomputable def number_of_cans_in_juice_set : ℕ := 10

theorem number_of_cans_per_set (n : ℕ) (h : cost_of_juice_set * n = number_of_cans_in_juice_set) : (n / 2) = 5 :=
by sorry

end number_of_cans_per_set_l18_18573


namespace find_digits_l18_18884

theorem find_digits :
  ∃ (A B C D : ℕ), A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
  0 ≤ A ∧ A ≤ 9 ∧ 0 ≤ B ∧ B ≤ 9 ∧ 0 ≤ C ∧ C ≤ 9 ∧ 0 ≤ D ∧ D ≤ 9 ∧
  (A * 1000 + B * 100 + C * 10 + D = 1098) :=
by {
  sorry
}

end find_digits_l18_18884


namespace product_diff_squares_l18_18015

theorem product_diff_squares (a b c d x1 y1 x2 y2 x3 y3 x4 y4 : ℕ) 
  (ha : a = x1^2 - y1^2) 
  (hb : b = x2^2 - y2^2) 
  (hc : c = x3^2 - y3^2) 
  (hd : d = x4^2 - y4^2)
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_pos_d : 0 < d) :
  ∃ X Y : ℕ, a * b * c * d = X^2 - Y^2 :=
by
  sorry

end product_diff_squares_l18_18015


namespace solution_set_inequality_x0_1_solution_set_inequality_x0_half_l18_18742

noncomputable def f (x : ℝ) : ℝ := abs (Real.log x)

theorem solution_set_inequality_x0_1 : 
  ∀ (c : ℝ), (∀ x, 0 < x → f x - f 1 ≥ c * (x - 1)) ↔ c ∈ Set.Icc (-1) 1 := 
by
  sorry

theorem solution_set_inequality_x0_half : 
  ∀ (c : ℝ), (∀ x, 0 < x → f x - f (1 / 2) ≥ c * (x - 1 / 2)) ↔ c = -2 :=
by
  sorry

end solution_set_inequality_x0_1_solution_set_inequality_x0_half_l18_18742


namespace distance_from_apex_l18_18421

theorem distance_from_apex (a₁ a₂ : ℝ) (d : ℝ)
  (ha₁ : a₁ = 150 * Real.sqrt 3)
  (ha₂ : a₂ = 300 * Real.sqrt 3)
  (hd : d = 10) :
  ∃ h : ℝ, h = 10 * Real.sqrt 2 :=
by
  sorry

end distance_from_apex_l18_18421


namespace zero_point_six_six_six_is_fraction_l18_18659

def is_fraction (x : ℝ) : Prop := ∃ (n d : ℤ), d ≠ 0 ∧ x = (n : ℝ) / (d : ℝ)

theorem zero_point_six_six_six_is_fraction:
  let sqrt_2_div_3 := (Real.sqrt 2) / 3
  let neg_sqrt_4 := - Real.sqrt 4
  let zero_point_six_six_six := 0.666
  let one_seventh := 1 / 7
  is_fraction zero_point_six_six_six :=
by sorry

end zero_point_six_six_six_is_fraction_l18_18659


namespace max_xy_l18_18351

theorem max_xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y = 18) : xy ≤ 81 :=
by sorry

end max_xy_l18_18351


namespace find_a4_l18_18807

variable (a : ℕ → ℝ)
variable (q : ℝ)
variable (T_7 : ℝ)

-- Conditions
axiom geom_seq (n : ℕ) : a (n + 1) = q * a n
axiom common_ratio_ne_one : q ≠ 1
axiom product_first_seven_terms : (a 1) * (a 2) * (a 3) * (a 4) * (a 5) * (a 6) * (a 7) = 128

-- Goal
theorem find_a4 : a 4 = 2 :=
sorry

end find_a4_l18_18807


namespace find_d_value_l18_18657

theorem find_d_value (d : ℝ) :
  (∀ x, (8 * x^3 + 27 * x^2 + d * x + 55 = 0) → (2 * x + 5 = 0)) → d = 39.5 :=
by
  sorry

end find_d_value_l18_18657


namespace axes_are_not_vectors_l18_18589

def is_vector (v : Type) : Prop :=
  ∃ (magnitude : ℝ) (direction : ℝ), magnitude > 0

def x_axis : Type := ℝ
def y_axis : Type := ℝ

-- The Cartesian x-axis and y-axis are not vectors
theorem axes_are_not_vectors : ¬ (is_vector x_axis) ∧ ¬ (is_vector y_axis) :=
by
  sorry

end axes_are_not_vectors_l18_18589


namespace initial_position_of_M_l18_18455

theorem initial_position_of_M :
  ∃ x : ℤ, (x + 7) - 4 = 0 ∧ x = -3 :=
by sorry

end initial_position_of_M_l18_18455


namespace guppies_to_angelfish_ratio_l18_18286

noncomputable def goldfish : ℕ := 8
noncomputable def angelfish : ℕ := goldfish + 4
noncomputable def total_fish : ℕ := 44
noncomputable def guppies : ℕ := total_fish - (goldfish + angelfish)

theorem guppies_to_angelfish_ratio :
    guppies / angelfish = 2 := by
    sorry

end guppies_to_angelfish_ratio_l18_18286


namespace range_of_m_l18_18558

theorem range_of_m (m : ℝ) : (∀ x : ℝ, m^2 * x^2 + 2 * m * x - 4 < 2 * x^2 + 4 * x) ↔ -2 < m ∧ m ≤ 2 :=
by
  sorry

end range_of_m_l18_18558


namespace circle_center_coordinates_l18_18221

theorem circle_center_coordinates :
  let p1 := (2, -3)
  let p2 := (8, 9)
  let midpoint (x₁ y₁ x₂ y₂ : ℝ) : ℝ × ℝ := ((x₁ + x₂) / 2, (y₁ + y₂) / 2)
  midpoint (2 : ℝ) (-3) 8 9 = (5, 3) :=
by
  sorry

end circle_center_coordinates_l18_18221


namespace trapezium_area_l18_18976

theorem trapezium_area (a b h : ℝ) (ha : a = 20) (hb : b = 18) (hh : h = 15) : 
  1/2 * (a + b) * h = 285 :=
by {
  sorry
}

end trapezium_area_l18_18976


namespace two_digit_solution_l18_18882

def two_digit_number (x y : ℕ) : ℕ := 10 * x + y

theorem two_digit_solution :
  ∃ (x y : ℕ), 
    two_digit_number x y = 24 ∧ 
    two_digit_number x y = x^3 + y^2 ∧ 
    0 ≤ x ∧ x ≤ 9 ∧ 
    0 ≤ y ∧ y ≤ 9 :=
by
  sorry

end two_digit_solution_l18_18882


namespace infinite_series_sum_eq_l18_18724

noncomputable def infinite_series_sum : Rat :=
  ∑' n : ℕ, (2 * n + 1) * (2000⁻¹) ^ n

theorem infinite_series_sum_eq : infinite_series_sum = (2003000 / 3996001) := by
  sorry

end infinite_series_sum_eq_l18_18724


namespace wire_length_l18_18402

theorem wire_length (S L : ℝ) (h1 : S = 10) (h2 : S = (2 / 5) * L) : S + L = 35 :=
by
  sorry

end wire_length_l18_18402


namespace line_from_complex_condition_l18_18592

theorem line_from_complex_condition (z : ℂ) (h : ∃ x y : ℝ, z = x + y * I ∧ (3 * y + 4 * x = 0)) : 
  ∃ (a b : ℝ), (∀ (x y : ℝ), z = x + y * I → 3 * y + 4 * x = 0 → z = a + b * I ∧ 4 * x + 3 * y = 0) := 
sorry

end line_from_complex_condition_l18_18592


namespace factoring_expression_l18_18773

theorem factoring_expression (a b c x y : ℝ) :
  -a * (x - y) - b * (y - x) + c * (x - y) = -(x - y) * (a + b - c) :=
by
  sorry

end factoring_expression_l18_18773


namespace complete_remaining_parts_l18_18430

-- Define the main conditions and the proof goal in Lean 4
theorem complete_remaining_parts :
  ∀ (total_parts processed_parts workers days_off remaining_parts_per_day),
  total_parts = 735 →
  processed_parts = 135 →
  workers = 5 →
  days_off = 1 →
  remaining_parts_per_day = total_parts - processed_parts →
  (workers * 2 - days_off) * 15 = processed_parts →
  remaining_parts_per_day / (workers * 15) = 8 :=
by
  -- Starting the proof
  intros total_parts processed_parts workers days_off remaining_parts_per_day
  intros h_total_parts h_processed_parts h_workers h_days_off h_remaining_parts_per_day h_productivity
  -- Replace given variables with their values
  sorry

end complete_remaining_parts_l18_18430


namespace jori_remaining_water_l18_18369

-- Having the necessary libraries for arithmetic and fractions.

-- Definitions directly from the conditions in a).
def initial_water_quantity : ℚ := 4
def used_water_quantity : ℚ := 9 / 4 -- Converted 2 1/4 to an improper fraction

-- The statement proving the remaining quantity of water is 1 3/4 gallons.
theorem jori_remaining_water : initial_water_quantity - used_water_quantity = 7 / 4 := by
  sorry

end jori_remaining_water_l18_18369


namespace distance_between_first_and_last_bushes_l18_18598

theorem distance_between_first_and_last_bushes 
  (bushes : Nat)
  (spaces_per_bush : ℕ) 
  (distance_first_to_fifth : ℕ) 
  (total_bushes : bushes = 10)
  (fifth_bush_distance : distance_first_to_fifth = 100)
  : ∃ (d : ℕ), d = 225 :=
by
  sorry

end distance_between_first_and_last_bushes_l18_18598


namespace total_goals_scored_l18_18401

-- Definitions based on the problem conditions
def kickers_first_period_goals : ℕ := 2
def kickers_second_period_goals : ℕ := 2 * kickers_first_period_goals
def spiders_first_period_goals : ℕ := kickers_first_period_goals / 2
def spiders_second_period_goals : ℕ := 2 * kickers_second_period_goals

-- The theorem we need to prove
theorem total_goals_scored : 
  kickers_first_period_goals + kickers_second_period_goals +
  spiders_first_period_goals + spiders_second_period_goals = 15 := 
by
  -- proof steps will go here
  sorry

end total_goals_scored_l18_18401


namespace isosceles_triangle_perimeter_l18_18682

def is_isosceles_triangle (a b c : ℝ) : Prop :=
(a = b ∨ b = c ∨ c = a) ∧ a + b > c ∧ a + c > b ∧ b + c > a

def perimeter (a b c : ℝ) : ℝ := a + b + c

theorem isosceles_triangle_perimeter (a b : ℝ) (h1 : a = 5) (h2 : b = 10) :
∃ c : ℝ, is_isosceles_triangle a b c ∧ perimeter a b c = 25 :=
by {
  sorry
}

end isosceles_triangle_perimeter_l18_18682


namespace largest_s_value_l18_18453

theorem largest_s_value (r s : ℕ) (h_r : r ≥ 3) (h_s : s ≥ 3) 
  (h_angle : (r - 2) * 180 / r = (5 * (s - 2) * 180) / (4 * s)) : s ≤ 130 :=
by {
  sorry
}

end largest_s_value_l18_18453


namespace nabla_2_3_2_eq_4099_l18_18071

def nabla (a b : ℕ) : ℕ := 3 + b ^ a

theorem nabla_2_3_2_eq_4099 : nabla (nabla 2 3) 2 = 4099 :=
by
  sorry

end nabla_2_3_2_eq_4099_l18_18071


namespace direction_vector_b_l18_18133

theorem direction_vector_b (b : ℝ) 
  (P Q : ℝ × ℝ) (hP : P = (-3, 1)) (hQ : Q = (1, 5))
  (hdir : 3 - (-3) = 3 ∧ 5 - 1 = b) : b = 3 := by
  sorry

end direction_vector_b_l18_18133


namespace find_quadruples_l18_18872

theorem find_quadruples (x y z n : ℕ) : 
  x^2 + y^2 + z^2 + 1 = 2^n → 
  (x = 0 ∧ y = 0 ∧ z = 0 ∧ n = 0) ∨ 
  (x = 1 ∧ y = 0 ∧ z = 0 ∧ n = 1) ∨ 
  (x = 0 ∧ y = 1 ∧ z = 0 ∧ n = 1) ∨ 
  (x = 0 ∧ y = 0 ∧ z = 1 ∧ n = 1) ∨ 
  (x = 1 ∧ y = 1 ∧ z = 1 ∧ n = 2) :=
by
  sorry

end find_quadruples_l18_18872


namespace min_value_l18_18548

theorem min_value (x y : ℝ) (h1 : xy > 0) (h2 : x + 4 * y = 3) : 
  ∃ (m : ℝ), m = 3 ∧ ∀ x y, xy > 0 → x + 4 * y = 3 → (1 / x + 1 / y) ≥ 3 := sorry

end min_value_l18_18548


namespace decimal_fraction_eq_l18_18295

theorem decimal_fraction_eq {b : ℕ} (hb : 0 < b) :
  (4 * b + 19 : ℚ) / (6 * b + 11) = 0.76 → b = 19 :=
by
  -- Proof goes here
  sorry

end decimal_fraction_eq_l18_18295


namespace sin_cos_of_angle_l18_18264

theorem sin_cos_of_angle (a : ℝ) (h₀ : a ≠ 0) :
  ∃ (s c : ℝ), (∃ (k : ℝ), s = k * (8 / 17) ∧ c = -k * (15 / 17) ∧ k = if a > 0 then 1 else -1) :=
by
  sorry

end sin_cos_of_angle_l18_18264


namespace forest_leaves_count_correct_l18_18330

def number_of_trees : ℕ := 20
def number_of_main_branches_per_tree : ℕ := 15
def number_of_sub_branches_per_main_branch : ℕ := 25
def number_of_tertiary_branches_per_sub_branch : ℕ := 30
def number_of_leaves_per_sub_branch : ℕ := 75
def number_of_leaves_per_tertiary_branch : ℕ := 45

def total_leaves_on_sub_branches_per_tree :=
  number_of_main_branches_per_tree * number_of_sub_branches_per_main_branch * number_of_leaves_per_sub_branch

def total_sub_branches_per_tree :=
  number_of_main_branches_per_tree * number_of_sub_branches_per_main_branch

def total_leaves_on_tertiary_branches_per_tree :=
  total_sub_branches_per_tree * number_of_tertiary_branches_per_sub_branch * number_of_leaves_per_tertiary_branch

def total_leaves_per_tree :=
  total_leaves_on_sub_branches_per_tree + total_leaves_on_tertiary_branches_per_tree

def total_leaves_in_forest :=
  total_leaves_per_tree * number_of_trees

theorem forest_leaves_count_correct :
  total_leaves_in_forest = 10687500 := 
by sorry

end forest_leaves_count_correct_l18_18330


namespace quadratic_coeff_b_is_4_sqrt_15_l18_18408

theorem quadratic_coeff_b_is_4_sqrt_15 :
  ∃ m b : ℝ, (x^2 + bx + 72 = (x + m)^2 + 12) → (m = 2 * Real.sqrt 15) → (b = 4 * Real.sqrt 15) ∧ b > 0 :=
by
  -- Note: Proof not included as per the instruction.
  sorry

end quadratic_coeff_b_is_4_sqrt_15_l18_18408


namespace waiter_tables_l18_18815

theorem waiter_tables (init_customers : ℕ) (left_customers : ℕ) (people_per_table : ℕ) (remaining_customers : ℕ) (num_tables : ℕ) :
  init_customers = 44 →
  left_customers = 12 →
  people_per_table = 8 →
  remaining_customers = init_customers - left_customers →
  num_tables = remaining_customers / people_per_table →
  num_tables = 4 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end waiter_tables_l18_18815


namespace solution_concentration_l18_18058

theorem solution_concentration (C : ℝ) :
  (0.16 + 0.01 * C * 2 = 0.36) ↔ (C = 10) :=
by
  sorry

end solution_concentration_l18_18058


namespace number_of_selection_plans_l18_18732

-- Definitions based on conditions
def male_students : Nat := 5
def female_students : Nat := 4
def total_volunteers : Nat := 3

def choose (n k : Nat) : Nat :=
  Nat.choose n k

def arrangement_count : Nat :=
  Nat.factorial total_volunteers

-- Theorem that states the total number of selection plans
theorem number_of_selection_plans :
  (choose male_students 2 * choose female_students 1 + choose male_students 1 * choose female_students 2) * arrangement_count = 420 :=
by
  sorry

end number_of_selection_plans_l18_18732


namespace common_difference_of_arithmetic_sequence_l18_18063

noncomputable def smallest_angle : ℝ := 25
noncomputable def largest_angle : ℝ := 105
noncomputable def num_angles : ℕ := 5

theorem common_difference_of_arithmetic_sequence :
  ∃ d : ℝ, (smallest_angle + (num_angles - 1) * d = largest_angle) ∧ d = 20 :=
by
  sorry

end common_difference_of_arithmetic_sequence_l18_18063


namespace inches_of_rain_received_so_far_l18_18841

def total_days_in_year : ℕ := 365
def days_left_in_year : ℕ := 100
def rain_per_day_initial_avg : ℝ := 2
def rain_per_day_required_avg : ℝ := 3

def total_annually_expected_rain : ℝ := rain_per_day_initial_avg * total_days_in_year
def days_passed_in_year : ℕ := total_days_in_year - days_left_in_year
def total_rain_needed_remaining : ℝ := rain_per_day_required_avg * days_left_in_year

variable (S : ℝ) -- inches of rain received so far

theorem inches_of_rain_received_so_far (S : ℝ) :
  S + total_rain_needed_remaining = total_annually_expected_rain → S = 430 :=
  by
  sorry

end inches_of_rain_received_so_far_l18_18841


namespace expand_expression_l18_18127

variable {R : Type*} [CommRing R]
variable (x y : R)

theorem expand_expression : 
  ((10 * x - 6 * y + 9) * 3 * y) = (30 * x * y - 18 * y * y + 27 * y) :=
by
  sorry

end expand_expression_l18_18127


namespace max_possible_x_l18_18428

noncomputable section

def tan_deg (x : ℕ) : ℝ := Real.tan (x * Real.pi / 180)

theorem max_possible_x (x y : ℕ) (h₁ : tan_deg x - tan_deg y = 1 + tan_deg x * tan_deg y)
  (h₂ : tan_deg x * tan_deg y = 1) (h₃ : x = 98721) : x = 98721 := sorry

end max_possible_x_l18_18428


namespace MarcoScoresAreCorrect_l18_18574

noncomputable def MarcoTestScores : List ℕ := [94, 82, 76, 75, 64]

theorem MarcoScoresAreCorrect : 
  ∀ (scores : List ℕ),
    scores = [82, 76, 75] ∧ 
    (∃ t4 t5, t4 < 95 ∧ t5 < 95 ∧ 82 ≠ t4 ∧ 82 ≠ t5 ∧ 76 ≠ t4 ∧ 76 ≠ t5 ∧ 75 ≠ t4 ∧ 75 ≠ t5 ∧ 
       t4 ≠ t5 ∧
       (82 + 76 + 75 + t4 + t5 = 5 * 85) ∧ 
       (82 + 76 = t4 + t5)) → 
    (scores = [94, 82, 76, 75, 64]) := 
by 
  sorry

end MarcoScoresAreCorrect_l18_18574


namespace subtracted_result_correct_l18_18538

theorem subtracted_result_correct (n : ℕ) (h1 : 96 / n = 6) : 34 - n = 18 :=
by
  sorry

end subtracted_result_correct_l18_18538


namespace _l18_18475

@[simp] theorem upper_base_length (ABCD is_trapezoid: Boolean)
  (point_M: Boolean)
  (perpendicular_DM_AB: Boolean)
  (MC_eq_CD: Boolean)
  (AD_eq_d: ℝ)
  : BC = d / 2 := sorry

end _l18_18475


namespace average_incorrect_l18_18250

theorem average_incorrect : ¬( (1 + 1 + 0 + 2 + 4) / 5 = 2) :=
by {
  sorry
}

end average_incorrect_l18_18250


namespace min_value_of_f_l18_18550

open Real

noncomputable def f (x : ℝ) : ℝ := (x^2 + 9) / sqrt (x^2 + 5)

theorem min_value_of_f : ∀ x : ℝ, f x ≥ 4 :=
by
  sorry

end min_value_of_f_l18_18550


namespace multiplication_problem_division_problem_l18_18135

theorem multiplication_problem :
  125 * 76 * 4 * 8 * 25 = 7600000 :=
sorry

theorem division_problem :
  (6742 + 6743 + 6738 + 6739 + 6741 + 6743) / 6 = 6741 :=
sorry

end multiplication_problem_division_problem_l18_18135


namespace pages_written_in_a_year_l18_18443

-- Definitions based on conditions
def pages_per_letter : ℕ := 3
def letters_per_week : ℕ := 2
def friends : ℕ := 2
def weeks_per_year : ℕ := 52

-- Definition to calculate total pages written in a week
def weekly_pages (pages_per_letter : ℕ) (letters_per_week : ℕ) (friends : ℕ) : ℕ :=
  pages_per_letter * letters_per_week * friends

-- Definition to calculate total pages written in a year
def yearly_pages (weekly_pages : ℕ) (weeks_per_year : ℕ) : ℕ :=
  weekly_pages * weeks_per_year

-- Theorem to prove the total pages written in a year
theorem pages_written_in_a_year : yearly_pages (weekly_pages pages_per_letter letters_per_week friends) weeks_per_year = 624 :=
by 
  sorry

end pages_written_in_a_year_l18_18443


namespace waiter_income_fraction_l18_18949

theorem waiter_income_fraction (S T : ℝ) (hT : T = 5/4 * S) :
  T / (S + T) = 5 / 9 :=
by
  sorry

end waiter_income_fraction_l18_18949


namespace seventeen_power_sixty_three_mod_seven_l18_18242

theorem seventeen_power_sixty_three_mod_seven : (17^63) % 7 = 6 := by
  -- Here you would write the actual proof demonstrating the equivalence:
  -- 1. 17 ≡ 3 (mod 7)
  -- 2. Calculate 3^63 (mod 7)
  sorry

end seventeen_power_sixty_three_mod_seven_l18_18242


namespace fractional_part_tiled_l18_18950

def room_length : ℕ := 12
def room_width : ℕ := 20
def number_of_tiles : ℕ := 40
def tile_area : ℕ := 1

theorem fractional_part_tiled :
  (number_of_tiles * tile_area : ℚ) / (room_length * room_width) = 1 / 6 :=
by
  sorry

end fractional_part_tiled_l18_18950


namespace value_of_frac_sum_l18_18855

theorem value_of_frac_sum (x y : ℚ) (h1 : 2 * x + y = 6) (h2 : x + 2 * y = 5) : (x + y) / 3 = 11 / 9 :=
by
  sorry

end value_of_frac_sum_l18_18855


namespace neg_prop_p_l18_18178

def prop_p (x : ℝ) : Prop := x ≥ 0 → Real.log (x^2 + 1) ≥ 0

theorem neg_prop_p : (¬ (∀ x ≥ 0, Real.log (x^2 + 1) ≥ 0)) ↔ (∃ x ≥ 0, Real.log (x^2 + 1) < 0) := by
  sorry

end neg_prop_p_l18_18178


namespace inv_38_mod_53_l18_18044

theorem inv_38_mod_53 (h : 15 * 31 % 53 = 1) : ∃ x : ℤ, 38 * x % 53 = 1 ∧ (x % 53 = 22) :=
by
  sorry

end inv_38_mod_53_l18_18044


namespace probability_two_red_cards_l18_18182

theorem probability_two_red_cards : 
  let total_cards := 100;
  let red_cards := 50;
  let black_cards := 50;
  (red_cards / total_cards : ℝ) * ((red_cards - 1) / (total_cards - 1) : ℝ) = 49 / 198 := 
by
  sorry

end probability_two_red_cards_l18_18182


namespace decimal_equivalent_of_one_half_squared_l18_18207

theorem decimal_equivalent_of_one_half_squared : (1 / 2 : ℝ) ^ 2 = 0.25 := 
sorry

end decimal_equivalent_of_one_half_squared_l18_18207


namespace cube_net_count_l18_18151

/-- A net of a cube is a two-dimensional arrangement of six squares.
    A regular tetrahedron has exactly 2 unique nets.
    For a cube, consider all possible ways in which the six faces can be arranged such that they 
    form a cube when properly folded. -/
theorem cube_net_count : cube_nets_count = 11 :=
sorry

end cube_net_count_l18_18151


namespace three_digit_odd_nums_using_1_2_3_4_5_without_repetition_l18_18936

def three_digit_odd_nums (digits : Finset ℕ) : ℕ :=
  let odd_digits := digits.filter (λ n => n % 2 = 1)
  let num_choices_for_units_place := odd_digits.card
  let remaining_digits := digits \ odd_digits
  let num_choices_for_hundreds_tens_places := remaining_digits.card * (remaining_digits.card - 1)
  num_choices_for_units_place * num_choices_for_hundreds_tens_places

theorem three_digit_odd_nums_using_1_2_3_4_5_without_repetition :
  three_digit_odd_nums {1, 2, 3, 4, 5} = 36 :=
by
  -- Proof is skipped
  sorry

end three_digit_odd_nums_using_1_2_3_4_5_without_repetition_l18_18936


namespace evaluate_expression_l18_18196

theorem evaluate_expression (x y : ℝ) (h1 : x > y) (h2 : y > 0) :
  (x^(2*y) * y^(2*x)) / (y^(2*y) * x^(2*x)) = (x / y)^(2 * (y - x)) :=
by
  sorry

end evaluate_expression_l18_18196


namespace triangle_side_lengths_m_range_l18_18520

theorem triangle_side_lengths_m_range (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (m : ℝ) :
  (2 - Real.sqrt 3) < m ∧ m < (2 + Real.sqrt 3) ↔
  (x + y) + Real.sqrt (x^2 + x * y + y^2) > m * Real.sqrt (x * y) ∧
  (x + y) + m * Real.sqrt (x * y) > Real.sqrt (x^2 + x * y + y^2) ∧
  Real.sqrt (x^2 + x * y + y^2) + m * Real.sqrt (x * y) > (x + y) :=
by sorry

end triangle_side_lengths_m_range_l18_18520


namespace total_votes_l18_18205

theorem total_votes (V : ℝ) (win_percentage : ℝ) (majority : ℝ) (lose_percentage : ℝ)
  (h1 : win_percentage = 0.75) (h2 : lose_percentage = 0.25) (h3 : majority = 420) :
  V = 840 :=
by
  sorry

end total_votes_l18_18205


namespace fraction_left_handed_l18_18603

-- Definitions based on given conditions
def red_ratio : ℝ := 10
def blue_ratio : ℝ := 5
def green_ratio : ℝ := 3
def yellow_ratio : ℝ := 2

def red_left_handed_percent : ℝ := 0.37
def blue_left_handed_percent : ℝ := 0.61
def green_left_handed_percent : ℝ := 0.26
def yellow_left_handed_percent : ℝ := 0.48

-- Statement we want to prove
theorem fraction_left_handed : 
  (red_left_handed_percent * red_ratio + blue_left_handed_percent * blue_ratio +
  green_left_handed_percent * green_ratio + yellow_left_handed_percent * yellow_ratio) /
  (red_ratio + blue_ratio + green_ratio + yellow_ratio) = 8.49 / 20 :=
  sorry

end fraction_left_handed_l18_18603


namespace lipstick_cost_correct_l18_18279

noncomputable def cost_of_lipsticks (total_cost: ℕ) (cost_slippers: ℚ) (cost_hair_color: ℚ) (paid: ℚ) (number_lipsticks: ℕ) : ℚ :=
  (paid - (6 * cost_slippers + 8 * cost_hair_color)) / number_lipsticks

theorem lipstick_cost_correct :
  cost_of_lipsticks 6 (2.5:ℚ) (3:ℚ) (44:ℚ) 4 = 1.25 := by
  sorry

end lipstick_cost_correct_l18_18279


namespace min_value_inequality_l18_18378

theorem min_value_inequality (θ φ : ℝ) : 
  (3 * Real.cos θ + 4 * Real.sin φ - 10)^2 + (3 * Real.sin θ + 4 * Real.cos φ - 20)^2 ≥ 549 - 140 * Real.sqrt 5 := 
by
  sorry

end min_value_inequality_l18_18378


namespace ellipse_eccentricity_l18_18689

theorem ellipse_eccentricity (x y : ℝ) (h : x^2 / 25 + y^2 / 9 = 1) : 
  let a := 5
  let b := 3
  let c := 4
  let e := c / a
  e = 4 / 5 :=
by
  sorry

end ellipse_eccentricity_l18_18689


namespace slopes_angle_l18_18399

theorem slopes_angle (k_1 k_2 : ℝ) (θ : ℝ) 
  (h1 : 6 * k_1^2 + k_1 - 1 = 0)
  (h2 : 6 * k_2^2 + k_2 - 1 = 0) :
  θ = π / 4 ∨ θ = 3 * π / 4 := 
by sorry

end slopes_angle_l18_18399


namespace alex_silver_tokens_l18_18804

theorem alex_silver_tokens :
  let R : Int -> Int -> Int := fun x y => 100 - 3 * x + 2 * y
  let B : Int -> Int -> Int := fun x y => 50 + 2 * x - 4 * y
  let x := 61
  let y := 42
  100 - 3 * x + 2 * y < 3 → 50 + 2 * x - 4 * y < 4 → x + y = 103 :=
by
  intro hR hB
  sorry

end alex_silver_tokens_l18_18804


namespace percentage_died_by_bombardment_l18_18754

noncomputable def initial_population : ℕ := 8515
noncomputable def final_population : ℕ := 6514

theorem percentage_died_by_bombardment :
  ∃ (x : ℝ), (0 ≤ x ∧ x ≤ 100) ∧
  8515 - ((x / 100) * 8515) - (15 / 100) * (8515 - ((x / 100) * 8515)) = 6514 ∧
  x = 10 :=
by
  sorry

end percentage_died_by_bombardment_l18_18754


namespace Lisa_flight_time_l18_18076

theorem Lisa_flight_time :
  let distance := 500
  let speed := 45
  (distance : ℝ) / (speed : ℝ) = 500 / 45 := by
  sorry

end Lisa_flight_time_l18_18076


namespace parcel_total_weight_l18_18519

theorem parcel_total_weight (x y z : ℝ) 
  (h1 : x + y = 132) 
  (h2 : y + z = 146) 
  (h3 : z + x = 140) : 
  x + y + z = 209 :=
by
  sorry

end parcel_total_weight_l18_18519


namespace percent_of_value_l18_18020

theorem percent_of_value (decimal_form : Real) (value : Nat) (expected_result : Real) : 
  decimal_form = 0.25 ∧ value = 300 ∧ expected_result = 75 → 
  decimal_form * value = expected_result := by
  sorry

end percent_of_value_l18_18020


namespace balance_of_diamondsuits_and_bullets_l18_18750

variable (a b c : ℕ)

theorem balance_of_diamondsuits_and_bullets 
  (h1 : 4 * a + 2 * b = 12 * c)
  (h2 : a = b + 3 * c) :
  3 * b = 6 * c := 
sorry

end balance_of_diamondsuits_and_bullets_l18_18750


namespace base7_digit_sum_l18_18088

theorem base7_digit_sum (A B C : ℕ) (hA : 1 ≤ A ∧ A < 7) (hB : 1 ≤ B ∧ B < 7) 
  (hC : 1 ≤ C ∧ C < 7) (h_distinct : A ≠ B ∧ B ≠ C ∧ A ≠ C) 
  (h_eq : 7^2 * A + 7 * B + C + 7^2 * B + 7 * C + A + 7^2 * C + 7 * A + B = 7^3 * A + 7^2 * A + 7 * A + 1) : 
  B + C = 6 := 
sorry

end base7_digit_sum_l18_18088


namespace problem_statement_l18_18625

noncomputable def a : ℝ := 31 / 32
noncomputable def b : ℝ := Real.cos (1 / 4)
noncomputable def c : ℝ := 4 * Real.sin (1 / 4)

theorem problem_statement : c > b ∧ b > a := 
by {
  sorry
}

end problem_statement_l18_18625


namespace herder_bulls_l18_18413

theorem herder_bulls (total_bulls : ℕ) (herder_fraction : ℚ) (claims : total_bulls = 70) (fraction_claim : herder_fraction = (2/3) * (1/3)) : herder_fraction * (total_bulls : ℚ) = 315 :=
by sorry

end herder_bulls_l18_18413


namespace inequality_1_inequality_2_inequality_3_inequality_4_l18_18986

noncomputable def triangle_angles (a b c : ℝ) : Prop :=
  a + b + c = Real.pi

theorem inequality_1 (a b c : ℝ) (h : triangle_angles a b c) :
  Real.sin a + Real.sin b + Real.sin c ≤ (3 * Real.sqrt 3 / 2) :=
sorry

theorem inequality_2 (a b c : ℝ) (h : triangle_angles a b c) :
  Real.cos (a / 2) + Real.cos (b / 2) + Real.cos (c / 2) ≤ (3 * Real.sqrt 3 / 2) :=
sorry

theorem inequality_3 (a b c : ℝ) (h : triangle_angles a b c) :
  Real.cos a * Real.cos b * Real.cos c ≤ (1 / 8) :=
sorry

theorem inequality_4 (a b c : ℝ) (h : triangle_angles a b c) :
  Real.sin (2 * a) + Real.sin (2 * b) + Real.sin (2 * c) ≤ Real.sin a + Real.sin b + Real.sin c :=
sorry

end inequality_1_inequality_2_inequality_3_inequality_4_l18_18986


namespace tangent_normal_lines_l18_18164

theorem tangent_normal_lines :
  ∃ m_t b_t m_n b_n,
    (∀ x y, y = 1 / (1 + x^2) → y = m_t * x + b_t → 4 * x + 25 * y - 13 = 0) ∧
    (∀ x y, y = 1 / (1 + x^2) → y = m_n * x + b_n → 125 * x - 20 * y - 246 = 0) :=
by
  sorry

end tangent_normal_lines_l18_18164


namespace problem_statement_l18_18209

-- Definitions based on conditions
def f (x : ℝ) : ℝ := x^2 - 1
def g (x : ℝ) : ℝ := 3 * x + 2

-- Theorem statement
theorem problem_statement : f (g 3) = 120 ∧ f 3 = 8 :=
by sorry

end problem_statement_l18_18209


namespace exists_additive_function_close_to_f_l18_18509

variable (f : ℝ → ℝ)

theorem exists_additive_function_close_to_f (h : ∀ x y : ℝ, |f (x + y) - f x - f y| ≤ 1) :
  ∃ g : ℝ → ℝ, (∀ x : ℝ, |f x - g x| ≤ 1) ∧ (∀ x y : ℝ, g (x + y) = g x + g y) := by
  sorry

end exists_additive_function_close_to_f_l18_18509


namespace undefined_expression_value_l18_18720

theorem undefined_expression_value {a : ℝ} : (a^3 - 8 = 0) ↔ (a = 2) :=
by sorry

end undefined_expression_value_l18_18720


namespace min_sum_of_squares_l18_18494

theorem min_sum_of_squares (a b c d : ℤ) (h1 : a^2 ≠ b^2 ∧ a^2 ≠ c^2 ∧ a^2 ≠ d^2 ∧ b^2 ≠ c^2 ∧ b^2 ≠ d^2 ∧ c^2 ≠ d^2)
                           (h2 : (a * b + c * d)^2 + (a * d - b * c)^2 = 2004) :
  a^2 + b^2 + c^2 + d^2 = 2 * Int.sqrt 2004 :=
sorry

end min_sum_of_squares_l18_18494


namespace solution_set_inequality_system_l18_18487

theorem solution_set_inequality_system (
  x : ℝ
) : (x + 1 ≥ 0 ∧ (x - 1) / 2 < 1) ↔ (-1 ≤ x ∧ x < 3) := by
  sorry

end solution_set_inequality_system_l18_18487


namespace number_of_votes_for_winner_l18_18786

-- Define the conditions
def total_votes : ℝ := 1000
def winner_percentage : ℝ := 0.55
def margin_of_victory : ℝ := 100

-- The statement to prove
theorem number_of_votes_for_winner :
  0.55 * total_votes = 550 :=
by
  -- We are supposed to provide the proof but it's skipped here
  sorry

end number_of_votes_for_winner_l18_18786


namespace T_simplified_l18_18050

-- Define the polynomial expression T
def T (x : ℝ) : ℝ := (x-2)^4 - 4*(x-2)^3 + 6*(x-2)^2 - 4*(x-2) + 1

-- Prove that T simplifies to (x-3)^4
theorem T_simplified (x : ℝ) : T x = (x - 3)^4 := by
  sorry

end T_simplified_l18_18050


namespace arithmetic_mean_odd_primes_lt_30_l18_18112

theorem arithmetic_mean_odd_primes_lt_30 : 
  (3 + 5 + 7 + 11 + 13 + 17 + 19 + 23 + 29) / 9 = 14 :=
by
  sorry

end arithmetic_mean_odd_primes_lt_30_l18_18112


namespace stream_speed_zero_l18_18125

theorem stream_speed_zero (v_c v_s : ℝ)
  (h1 : v_c - v_s - 2 = 9)
  (h2 : v_c + v_s + 1 = 12) :
  v_s = 0 := 
sorry

end stream_speed_zero_l18_18125


namespace c_alone_finishes_in_60_days_l18_18074

-- Definitions for rates of work
variables (A B C : ℝ)

-- The conditions given in the problem
-- A and B together can finish the job in 15 days
def condition1 : Prop := A + B = 1 / 15
-- A, B, and C together can finish the job in 12 days
def condition2 : Prop := A + B + C = 1 / 12

-- The statement to prove: C alone can finish the job in 60 days
theorem c_alone_finishes_in_60_days 
  (h1 : condition1 A B) 
  (h2 : condition2 A B C) : 
  (1 / C) = 60 :=
by
  sorry

end c_alone_finishes_in_60_days_l18_18074


namespace system_of_linear_eq_l18_18861

theorem system_of_linear_eq :
  ∃ (x y : ℝ), x + y = 5 ∧ y = 2 :=
sorry

end system_of_linear_eq_l18_18861


namespace percentage_male_red_ants_proof_l18_18604

noncomputable def percentage_red_ants : ℝ := 0.85
noncomputable def percentage_female_red_ants : ℝ := 0.45
noncomputable def percentage_male_red_ants : ℝ := percentage_red_ants * (1 - percentage_female_red_ants)

theorem percentage_male_red_ants_proof : percentage_male_red_ants = 0.4675 :=
by
  -- Proof will go here
  sorry

end percentage_male_red_ants_proof_l18_18604


namespace multiplication_to_squares_l18_18479

theorem multiplication_to_squares :
  85 * 135 = 85^2 + 50^2 + 35^2 + 15^2 + 15^2 + 5^2 + 5^2 + 5^2 :=
by
  sorry

end multiplication_to_squares_l18_18479


namespace sum_of_A_and_B_l18_18694

theorem sum_of_A_and_B (A B : ℕ) (h1 : 7 - B = 3) (h2 : A - 5 = 4) (h_diff : A ≠ B) : A + B = 13 :=
sorry

end sum_of_A_and_B_l18_18694


namespace ethan_presents_l18_18321

variable (A E : ℝ)

theorem ethan_presents (h1 : A = 9) (h2 : A = E - 22.0) : E = 31 := 
by
  sorry

end ethan_presents_l18_18321


namespace total_notes_l18_18104

theorem total_notes (total_money : ℕ) (fifty_notes : ℕ) (fifty_value : ℕ) (fivehundred_value : ℕ) (fivehundred_notes : ℕ) :
  total_money = 10350 →
  fifty_notes = 57 →
  fifty_value = 50 →
  fivehundred_value = 500 →
  57 * 50 + fivehundred_notes * 500 = 10350 →
  fifty_notes + fivehundred_notes = 72 :=
by
  intros h_total_money h_fifty_notes h_fifty_value h_fivehundred_value h_equation
  sorry

end total_notes_l18_18104


namespace baron_munchausen_failed_l18_18998

theorem baron_munchausen_failed : 
  ∀ (n : ℕ), 10 ≤ n ∧ n ≤ 99 → ¬∃ (d1 d2 : ℕ), ∃ (k : ℕ), n * 100 + (d1 * 10 + d2) = k^2 := 
by
  intros n hn
  obtain ⟨h10, h99⟩ := hn
  sorry

end baron_munchausen_failed_l18_18998


namespace point_on_angle_bisector_l18_18802

theorem point_on_angle_bisector (a b : ℝ) (h : ∃ p : ℝ, (a, b) = (p, -p)) : a = -b :=
sorry

end point_on_angle_bisector_l18_18802


namespace determine_a_l18_18437

def A (a : ℝ) : Set ℝ := {1, a}
def B (a : ℝ) : Set ℝ := {a^2}

theorem determine_a (a : ℝ) (A_union_B_eq_A : A a ∪ B a = A a) : a = -1 ∨ a = 0 := by
  sorry

end determine_a_l18_18437


namespace magic_trick_constant_l18_18329

theorem magic_trick_constant (a : ℚ) : ((2 * a + 8) / 4 - a / 2) = 2 :=
by
  sorry

end magic_trick_constant_l18_18329


namespace no_solution_nat_x_satisfies_eq_l18_18185

def sum_digits (x : ℕ) : ℕ := x.digits 10 |>.sum

theorem no_solution_nat_x_satisfies_eq (x : ℕ) :
  ¬ (x + sum_digits x + sum_digits (sum_digits x) = 2014) :=
by
  sorry

end no_solution_nat_x_satisfies_eq_l18_18185


namespace list_price_is_40_l18_18080

theorem list_price_is_40 (x : ℝ) :
  (0.15 * (x - 15) = 0.25 * (x - 25)) → x = 40 :=
by
  intro h
  -- The proof steps would go here, but we'll use sorry to indicate we're skipping the proof.
  sorry

end list_price_is_40_l18_18080


namespace find_x0_l18_18585

def f (x : ℝ) := x * abs x

theorem find_x0 (x0 : ℝ) (h : f x0 = 4) : x0 = 2 :=
by
  sorry

end find_x0_l18_18585


namespace binkie_gemstones_l18_18022

variables (F B S : ℕ)

theorem binkie_gemstones :
  (B = 4 * F) →
  (S = (1 / 2 : ℝ) * F - 2) →
  (S = 1) →
  B = 24 :=
by
  sorry

end binkie_gemstones_l18_18022


namespace mushroom_collectors_l18_18410

theorem mushroom_collectors :
  ∃ (n m : ℕ), 13 * n - 10 * m = 2 ∧ 9 ≤ n ∧ n ≤ 15 ∧ 11 ≤ m ∧ m ≤ 20 ∧ n = 14 ∧ m = 18 := by sorry

end mushroom_collectors_l18_18410


namespace time_to_cross_platform_l18_18318

-- Definitions based on the given conditions
def train_length : ℝ := 300
def time_to_cross_pole : ℝ := 18
def platform_length : ℝ := 350

-- The question reformulated as a theorem in Lean 4
theorem time_to_cross_platform 
  (l_train : ℝ := train_length)
  (t_pole_cross : ℝ := time_to_cross_pole)
  (l_platform : ℝ := platform_length) :
  (l_train / t_pole_cross * (l_train + l_platform) = 39) :=
sorry

end time_to_cross_platform_l18_18318


namespace side_c_possibilities_l18_18890

theorem side_c_possibilities (A : ℝ) (a b c : ℝ) (hA : A = 30) (ha : a = 4) (hb : b = 4 * Real.sqrt 3) :
  c = 4 ∨ c = 8 :=
sorry

end side_c_possibilities_l18_18890


namespace calculate_final_price_l18_18253

def original_price : ℝ := 120
def fixture_discount : ℝ := 0.20
def decor_discount : ℝ := 0.15

def discounted_price_after_first_discount (p : ℝ) (d : ℝ) : ℝ :=
  p * (1 - d)

def final_price (p : ℝ) (d1 : ℝ) (d2 : ℝ) : ℝ :=
  let price_after_first_discount := discounted_price_after_first_discount p d1
  price_after_first_discount * (1 - d2)

theorem calculate_final_price :
  final_price original_price fixture_discount decor_discount = 81.60 :=
by sorry

end calculate_final_price_l18_18253


namespace spencer_session_duration_l18_18923

-- Definitions of the conditions
def jumps_per_minute : ℕ := 4
def sessions_per_day : ℕ := 2
def total_jumps : ℕ := 400
def total_days : ℕ := 5

-- Calculation target: find the duration of each session
def jumps_per_day : ℕ := total_jumps / total_days
def jumps_per_session : ℕ := jumps_per_day / sessions_per_day
def session_duration := jumps_per_session / jumps_per_minute

theorem spencer_session_duration :
  session_duration = 10 := 
sorry

end spencer_session_duration_l18_18923


namespace contrapositive_of_zero_squared_l18_18581

theorem contrapositive_of_zero_squared {x y : ℝ} :
  (x^2 + y^2 = 0 → x = 0 ∧ y = 0) →
  (x ≠ 0 ∨ y ≠ 0 → x^2 + y^2 ≠ 0) :=
by
  intro h1
  intro h2
  sorry

end contrapositive_of_zero_squared_l18_18581


namespace problem_part1_problem_part2_l18_18839

-- Definitions of the vectors
def a (x : ℝ) : ℝ × ℝ := (1, 2 * x)
def b (x : ℝ) : ℝ × ℝ := (x, 3)
def c : ℝ × ℝ := (-2, 0)

-- Definitions for vector operations
def add_vec (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 + v.1, u.2 + v.2)
def scalar_mul (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (k * v.1, k * v.2)
def parallel (u v : ℝ × ℝ) : Prop := u.1 * v.2 = u.2 * v.1
def perpendicular (u v : ℝ × ℝ) : Prop := u.1 * v.1 + u.2 * v.2 = 0

noncomputable def part1 (x : ℝ) : Prop := parallel (add_vec (a x) (scalar_mul 2 (b x))) (add_vec (scalar_mul 2 (a x)) (scalar_mul (-1) c))

noncomputable def part2 (x : ℝ) : Prop := perpendicular (add_vec (a x) (scalar_mul 2 (b x))) (add_vec (scalar_mul 2 (a x)) (scalar_mul (-1) c))

theorem problem_part1 : part1 2 ∧ part1 (-3 / 2) := sorry

theorem problem_part2 : part2 ((-4 + Real.sqrt 14) / 2) ∧ part2 ((-4 - Real.sqrt 14) / 2) := sorry

end problem_part1_problem_part2_l18_18839


namespace problem_solution_l18_18680

theorem problem_solution :
  let m := 9
  let n := 20
  let lhs := (2 + 4 + 6 + 8) / (1 + 3 + 5 + 7) - (1 + 3 + 5 + 7) / (2 + 4 + 6 + 8)
  let rhs := 9 / 20
  lhs = rhs → 10 * m + n = 110 :=
by sorry

end problem_solution_l18_18680


namespace jack_jill_total_difference_l18_18647

theorem jack_jill_total_difference :
  let original_price := 90.00
  let discount_rate := 0.20
  let tax_rate := 0.06

  -- Jack's calculation
  let jack_total :=
    let price_with_tax := original_price * (1 + tax_rate)
    price_with_tax * (1 - discount_rate)
  
  -- Jill's calculation
  let jill_total :=
    let discounted_price := original_price * (1 - discount_rate)
    discounted_price * (1 + tax_rate)

  -- Equality check
  jack_total = jill_total := 
by
  -- Place the proof here
  sorry

end jack_jill_total_difference_l18_18647


namespace determine_m_l18_18743

-- Definition of complex numbers z1 and z2
def z1 (m : ℝ) : ℂ := m + 2 * Complex.I
def z2 : ℂ := 2 + Complex.I

-- Condition that the product z1 * z2 is a pure imaginary number
def pure_imaginary (c : ℂ) : Prop := c.re = 0 

-- The proof statement
theorem determine_m (m : ℝ) : pure_imaginary (z1 m * z2) → m = 1 := 
sorry

end determine_m_l18_18743


namespace find_prime_pairs_l18_18100

def is_solution_pair (p q : ℕ) : Prop :=
  Nat.Prime p ∧ Nat.Prime q ∧ (p ∣ 5^q + 1) ∧ (q ∣ 5^p + 1)

theorem find_prime_pairs :
  {pq : ℕ × ℕ | is_solution_pair pq.1 pq.2} =
  { (2, 13), (13, 2), (3, 7), (7, 3) } :=
by
  sorry

end find_prime_pairs_l18_18100


namespace order_of_magnitude_l18_18488

theorem order_of_magnitude (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let m := a / Real.sqrt b + b / Real.sqrt a
  let n := Real.sqrt a + Real.sqrt b
  let p := Real.sqrt (a + b)
  m ≥ n ∧ n > p := 
sorry

end order_of_magnitude_l18_18488


namespace horse_catches_up_l18_18247

-- Definitions based on given conditions
def dog_speed := 20 -- derived from 5 steps * 4 meters
def horse_speed := 21 -- derived from 3 steps * 7 meters
def initial_distance := 30 -- dog has already run 30 meters

-- Statement to be proved
theorem horse_catches_up (d h : ℕ) (time : ℕ) :
  d = dog_speed → h = horse_speed →
  initial_distance = 30 →
  h * time = initial_distance + dog_speed * time →
  time = 600 / (h - d) ∧ h * time - initial_distance = 600 :=
by
  intros
  -- Proof placeholders
  sorry  -- Omit the actual proof steps

end horse_catches_up_l18_18247


namespace find_inverse_sum_l18_18781

theorem find_inverse_sum (a b : ℝ) (h1 : a ≠ b) (h2 : a / b + a = b / a + b) : 1 / a + 1 / b = -1 :=
sorry

end find_inverse_sum_l18_18781


namespace chickens_after_9_years_l18_18029

-- Definitions from the conditions
def annual_increase : ℕ := 150
def current_chickens : ℕ := 550
def years : ℕ := 9

-- Lean statement for the proof
theorem chickens_after_9_years : current_chickens + annual_increase * years = 1900 :=
by
  sorry

end chickens_after_9_years_l18_18029


namespace find_value_l18_18400

variable {x y : ℝ}

theorem find_value (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y + x * y = 0) : y / x + x / y = -2 := 
sorry

end find_value_l18_18400


namespace dogs_in_kennel_l18_18128

theorem dogs_in_kennel (C D : ℕ) (h1 : C = D - 8) (h2 : C * 4 = 3 * D) : D = 32 :=
sorry

end dogs_in_kennel_l18_18128


namespace log_identity_l18_18954

theorem log_identity (a b : ℝ) (h1 : a = Real.log 144 / Real.log 4) (h2 : b = Real.log 12 / Real.log 2) : a = b := 
by
  sorry

end log_identity_l18_18954


namespace employees_after_reduction_l18_18767

def reduction (original : Float) (percent : Float) : Float :=
  original - (percent * original)

theorem employees_after_reduction :
  reduction 243.75 0.20 = 195 := by
  sorry

end employees_after_reduction_l18_18767


namespace smallest_positive_int_l18_18576

open Nat

theorem smallest_positive_int (x : ℕ) :
  (x % 6 = 3) ∧ (x % 8 = 5) ∧ (x % 9 = 2) → x = 237 := by
  sorry

end smallest_positive_int_l18_18576


namespace least_positive_t_l18_18782

theorem least_positive_t (t : ℕ) (α : ℝ) (h1 : 0 < α ∧ α < π / 2)
  (h2 : π / 10 < α ∧ α ≤ π / 6) 
  (h3 : (3 * α)^2 = α * (π - 5 * α)) :
  t = 27 :=
by
  have hα : α = π / 14 := 
    by
      sorry
  sorry

end least_positive_t_l18_18782


namespace find_t_l18_18055

-- Definitions from the given conditions
def earning (hours : ℕ) (rate : ℕ) : ℕ := hours * rate

-- The main theorem based on the translated problem
theorem find_t
  (t : ℕ)
  (h1 : earning (t - 4) (3 * t - 7) = earning (3 * t - 12) (t - 3)) :
  t = 4 := 
sorry

end find_t_l18_18055


namespace find_digits_l18_18903

theorem find_digits (A B D E C : ℕ) 
  (hC : C = 9) 
  (hA : 2 < A ∧ A < 4)
  (hB : B = 5)
  (hE : E = 6)
  (hD : D = 0)
  (h_distinct : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ C ≠ D ∧ C ≠ E ∧ D ≠ E) :
  (A, B, D, E) = (3, 5, 0, 6) := by
  sorry

end find_digits_l18_18903


namespace calculate_expression_l18_18805

theorem calculate_expression :
  ((2000000000000 - 1234567890123) * 3 = 2296296329631) :=
by 
  sorry

end calculate_expression_l18_18805


namespace cricket_team_average_age_l18_18331

open Real

-- Definitions based on the conditions given
def team_size := 11
def captain_age := 27
def wicket_keeper_age := 30
def remaining_players_size := team_size - 2

-- The mathematically equivalent proof problem in Lean statement
theorem cricket_team_average_age :
  ∃ A : ℝ,
    (A - 1) * remaining_players_size = (A * team_size) - (captain_age + wicket_keeper_age) ∧
    A = 24 :=
by
  sorry

end cricket_team_average_age_l18_18331


namespace sandy_age_l18_18222

theorem sandy_age (S M : ℕ) 
  (h1 : M = S + 16) 
  (h2 : (↑S : ℚ) / ↑M = 7 / 9) : 
  S = 56 :=
by sorry

end sandy_age_l18_18222


namespace remainder_9_5_4_6_5_7_mod_7_l18_18305

theorem remainder_9_5_4_6_5_7_mod_7 :
  ((9^5 + 4^6 + 5^7) % 7) = 2 :=
by sorry

end remainder_9_5_4_6_5_7_mod_7_l18_18305


namespace probability_of_bayonet_base_on_third_try_is_7_over_120_l18_18821

noncomputable def probability_picking_bayonet_base_bulb_on_third_try : ℚ :=
  (3 / 10) * (2 / 9) * (7 / 8)

/-- Given a box containing 3 screw base bulbs and 7 bayonet base bulbs, all with the
same shape and power and placed with their bases down. An electrician takes one bulb
at a time without returning it. The probability that he gets a bayonet base bulb on his
third try is 7/120. -/
theorem probability_of_bayonet_base_on_third_try_is_7_over_120 :
  probability_picking_bayonet_base_bulb_on_third_try = 7 / 120 :=
by 
  sorry

end probability_of_bayonet_base_on_third_try_is_7_over_120_l18_18821


namespace n_value_l18_18267

theorem n_value (n : ℤ) (h1 : (18888 - n) % 11 = 0) : n = 7 :=
sorry

end n_value_l18_18267


namespace semicircle_arc_length_l18_18895

theorem semicircle_arc_length (a b : ℝ) (hypotenuse_sum : a + b = 70) (a_eq_30 : a = 30) (b_eq_40 : b = 40) :
  ∃ (R : ℝ), (R = 24) ∧ (π * R = 12 * π) :=
by
  sorry

end semicircle_arc_length_l18_18895


namespace negation_of_inequality_l18_18304

theorem negation_of_inequality :
  ¬ (∀ x : ℝ, x^2 - x + 2 ≥ 0) ↔ ∃ x : ℝ, x^2 - x + 2 < 0 := 
sorry

end negation_of_inequality_l18_18304


namespace simplify_expression_l18_18925

theorem simplify_expression (x : ℝ) : (3 * x + 30) + (150 * x - 45) = 153 * x - 15 :=
by
  sorry

end simplify_expression_l18_18925


namespace gold_copper_ratio_l18_18214

theorem gold_copper_ratio (G C : ℕ) 
  (h1 : 19 * G + 9 * C = 18 * (G + C)) : 
  G = 9 * C :=
by
  sorry

end gold_copper_ratio_l18_18214


namespace value_of_expression_l18_18277

theorem value_of_expression (a b c d x y : ℤ) 
  (h1 : a = -b) 
  (h2 : c * d = 1)
  (h3 : abs x = 3)
  (h4 : y = -1) : 
  2 * x - c * d + 6 * (a + b) - abs y = 4 ∨ 2 * x - c * d + 6 * (a + b) - abs y = -8 := 
by 
  sorry

end value_of_expression_l18_18277


namespace exists_monochromatic_rectangle_l18_18014

theorem exists_monochromatic_rectangle 
  (coloring : ℤ × ℤ → Prop)
  (h : ∀ p : ℤ × ℤ, coloring p = red ∨ coloring p = blue)
  : ∃ (a b c d : ℤ × ℤ), (a.1 = b.1) ∧ (c.1 = d.1) ∧ (a.2 = c.2) ∧ (b.2 = d.2) ∧ (coloring a = coloring b) ∧ (coloring b = coloring c) ∧ (coloring c = coloring d) :=
sorry

end exists_monochromatic_rectangle_l18_18014


namespace fraction_of_earth_habitable_l18_18933

theorem fraction_of_earth_habitable :
  ∀ (earth_surface land_area inhabitable_land_area : ℝ),
    land_area = 1 / 3 → 
    inhabitable_land_area = 1 / 4 → 
    (earth_surface * land_area * inhabitable_land_area) = 1 / 12 :=
  by
    intros earth_surface land_area inhabitable_land_area h_land h_inhabitable
    sorry

end fraction_of_earth_habitable_l18_18933


namespace M_inter_N_eq_singleton_l18_18687

def M (x y : ℝ) : Prop := x + y = 2
def N (x y : ℝ) : Prop := x - y = 4

theorem M_inter_N_eq_singleton :
  {p : ℝ × ℝ | M p.1 p.2} ∩ {p : ℝ × ℝ | N p.1 p.2} = { (3, -1) } :=
by
  sorry

end M_inter_N_eq_singleton_l18_18687


namespace solve_fraction_equation_l18_18960

theorem solve_fraction_equation (x : ℝ) (h₁ : x ≠ 0) (h₂ : x ≠ -3) :
  (2 / x + x / (x + 3) = 1) ↔ x = 6 := 
by
  sorry

end solve_fraction_equation_l18_18960


namespace min_value_f_l18_18012

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * x ^ 3 + b * Real.arcsin x + 3

theorem min_value_f (a b : ℝ) (hab : a ≠ 0 ∧ b ≠ 0) (hmax : ∃ x, f a b x = 10) : ∃ y, f a b y = -4 := by
  sorry

end min_value_f_l18_18012


namespace bruce_bhishma_meet_again_l18_18040

theorem bruce_bhishma_meet_again (L S_B S_H : ℕ) (hL : L = 600) (hSB : S_B = 30) (hSH : S_H = 20) : 
  ∃ t : ℕ, t = 60 ∧ (t * S_B - t * S_H) % L = 0 :=
by
  sorry

end bruce_bhishma_meet_again_l18_18040


namespace hemisphere_surface_area_l18_18554

theorem hemisphere_surface_area (r : ℝ) (π : ℝ) (h1: 0 < π) (h2: A = 3) (h3: S = 4 * π * r^2):
  ∃ t, t = 9 :=
by
  sorry

end hemisphere_surface_area_l18_18554


namespace find_c_quadratic_solution_l18_18543

theorem find_c_quadratic_solution (c : ℝ) :
  (Polynomial.eval (-5) (Polynomial.C (-45) + Polynomial.X * Polynomial.C c + Polynomial.X^2) = 0) →
  c = -4 :=
by 
  intros h
  sorry

end find_c_quadratic_solution_l18_18543


namespace find_missing_number_l18_18706

theorem find_missing_number
  (x y : ℕ)
  (h1 : 30 = 6 * 5)
  (h2 : 600 = 30 * x)
  (h3 : x = 5 * y) :
  y = 4 :=
by
  sorry

end find_missing_number_l18_18706


namespace find_m_l18_18274

-- Mathematical definitions from the given conditions
def condition1 (m : ℝ) : Prop := m^2 - 2 * m - 2 = 1
def condition2 (m : ℝ) : Prop := m + 1/2 * m^2 > 0

-- The proof problem summary
theorem find_m (m : ℝ) (h1 : condition1 m) (h2 : condition2 m) : m = 3 :=
by
  sorry

end find_m_l18_18274


namespace sum_of_interior_angles_10th_polygon_l18_18656

theorem sum_of_interior_angles_10th_polygon (n : ℕ) (h1 : n = 10) : 
  180 * (n - 2) = 1440 :=
by
  sorry

end sum_of_interior_angles_10th_polygon_l18_18656


namespace xyz_zero_unique_solution_l18_18496

theorem xyz_zero_unique_solution {x y z : ℝ} (h1 : x^2 * y + y^2 * z + z^2 = 0)
                                 (h2 : z^3 + z^2 * y + z * y^3 + x^2 * y = 1 / 4 * (x^4 + y^4)) :
  x = 0 ∧ y = 0 ∧ z = 0 :=
sorry

end xyz_zero_unique_solution_l18_18496


namespace fraction_sum_of_lcm_and_gcd_l18_18376

theorem fraction_sum_of_lcm_and_gcd 
  (m n : ℕ) 
  (h_gcd : Nat.gcd m n = 6) 
  (h_lcm : Nat.lcm m n = 210) 
  (h_sum : m + n = 72) :
  1 / (m : ℚ) + 1 / (n : ℚ) = 12 / 210 := 
by
sorry

end fraction_sum_of_lcm_and_gcd_l18_18376


namespace product_is_zero_l18_18955

def product_series (a : ℤ) : ℤ :=
  (a - 12) * (a - 11) * (a - 10) * (a - 9) * (a - 8) * (a - 7) * (a - 6) * (a - 5) * 
  (a - 4) * (a - 3) * (a - 2) * (a - 1) * a

theorem product_is_zero : product_series 3 = 0 :=
by
  sorry

end product_is_zero_l18_18955


namespace ratio_AB_to_AD_l18_18771

/-
In rectangle ABCD, 30% of its area overlaps with square EFGH. Square EFGH shares 40% of its area with rectangle ABCD. If AD equals one-tenth of the side length of square EFGH, what is AB/AD?
-/

theorem ratio_AB_to_AD (s x y : ℝ)
  (h1 : 0.3 * (x * y) = 0.4 * s^2)
  (h2 : y = s / 10):
  (x / y) = 400 / 3 :=
by
  sorry

end ratio_AB_to_AD_l18_18771


namespace sum_of_squares_pattern_l18_18043

theorem sum_of_squares_pattern (a b : ℝ)
  (h1 : a + b = 1)
  (h2 : a^2 + b^2 = 3)
  (h3 : a^3 + b^3 = 4)
  (h4 : a^4 + b^4 = 7)
  (h5 : a^5 + b^5 = 11) :
  a^6 + b^6 = 18 :=
sorry

end sum_of_squares_pattern_l18_18043


namespace inequalities_hold_l18_18549

theorem inequalities_hold (b : ℝ) :
  (b ∈ Set.Ioo (-(1 : ℝ) - Real.sqrt 2 / 4) (0 : ℝ) ∨ b < -(1 : ℝ) - Real.sqrt 2 / 4) →
  (∀ x y : ℝ, 2 * b * Real.cos (2 * (x - y)) + 8 * b^2 * Real.cos (x - y) + 8 * b^2 * (b + 1) + 5 * b < 0) ∧ 
  (∀ x y : ℝ, x^2 + y^2 + 1 > 2 * b * x + 2 * y + b - b^2) :=
by 
  intro h
  sorry

end inequalities_hold_l18_18549


namespace pre_bought_tickets_l18_18203

theorem pre_bought_tickets (P : ℕ) 
  (h1 : ∃ P, 155 * P + 2900 = 6000) : P = 20 :=
by {
  -- Insert formalization of steps leading to P = 20
  sorry
}

end pre_bought_tickets_l18_18203


namespace distance_city_A_B_l18_18427

theorem distance_city_A_B (D : ℝ) : 
  (3 : ℝ) + (2.5 : ℝ) = 5.5 → 
  ∃ T_saved, T_saved = 1 →
  80 = (2 * D) / (5.5 - T_saved) →
  D = 180 :=
by
  intros
  sorry

end distance_city_A_B_l18_18427


namespace sample_size_divided_into_six_groups_l18_18580

theorem sample_size_divided_into_six_groups
  (n : ℕ)
  (c1 c2 c3 : ℕ)
  (k : ℚ)
  (h1 : c1 + c2 + c3 = 36)
  (h2 : 20 * k = 1)
  (h3 : 2 * k * n = c1)
  (h4 : 3 * k * n = c2)
  (h5 : 4 * k * n = c3) :
  n = 80 :=
by
  sorry

end sample_size_divided_into_six_groups_l18_18580


namespace Sally_quarters_l18_18324

theorem Sally_quarters : 760 + 418 - 152 = 1026 := 
by norm_num

end Sally_quarters_l18_18324


namespace sam_initial_dimes_l18_18335

theorem sam_initial_dimes (given_away : ℕ) (left : ℕ) (initial : ℕ) 
  (h1 : given_away = 7) (h2 : left = 2) (h3 : initial = given_away + left) : 
  initial = 9 := by
  rw [h1, h2] at h3
  exact h3

end sam_initial_dimes_l18_18335


namespace max_minus_min_all_three_languages_l18_18072

def student_population := 1500

def english_students (e : ℕ) : Prop := 1050 ≤ e ∧ e ≤ 1125
def spanish_students (s : ℕ) : Prop := 750 ≤ s ∧ s ≤ 900
def german_students (g : ℕ) : Prop := 300 ≤ g ∧ g ≤ 450

theorem max_minus_min_all_three_languages (e s g e_s e_g s_g e_s_g : ℕ) 
    (he : english_students e)
    (hs : spanish_students s)
    (hg : german_students g)
    (pie : e + s + g - e_s - e_g - s_g + e_s_g = student_population) 
    : (M - m = 450) :=
sorry

end max_minus_min_all_three_languages_l18_18072


namespace infinite_solutions_implies_d_eq_five_l18_18184

theorem infinite_solutions_implies_d_eq_five (d : ℝ) :
  (∀ y : ℝ, 3 * (5 + d * y) = 15 * y + 15) ↔ (d = 5) := by
sorry

end infinite_solutions_implies_d_eq_five_l18_18184


namespace function_value_proof_l18_18889

theorem function_value_proof (f : ℝ → ℝ) (a b : ℝ) 
    (h1 : ∀ x, f (x + 1) = -f (-x + 1))
    (h2 : ∀ x, f (x + 2) = f (-x + 2))
    (h3 : ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → f x = a * x^2 + b)
    (h4 : ∀ x y : ℝ, x - y - 3 = 0)
    : f (9/2) = 5/4 := by
  sorry

end function_value_proof_l18_18889


namespace second_term_geometric_sequence_l18_18654

-- Given conditions
def a3 : ℕ := 12
def a4 : ℕ := 18
def q := a4 / a3 -- common ratio

-- Geometric progression definition
noncomputable def a2 := a3 / q

-- Theorem to prove
theorem second_term_geometric_sequence : a2 = 8 :=
by
  -- proof not required
  sorry

end second_term_geometric_sequence_l18_18654


namespace solve_for_x_in_equation_l18_18898

theorem solve_for_x_in_equation (x : ℝ)
  (h : (2 / 7) * (1 / 4) * x = 12) : x = 168 :=
sorry

end solve_for_x_in_equation_l18_18898


namespace hyperbola_a_value_l18_18373

theorem hyperbola_a_value (a : ℝ) :
  (∀ x y : ℝ, (x^2 / (a + 3) - y^2 / 3 = 1)) ∧ 
  (∀ e : ℝ, e = 2) → 
  a = -2 :=
by sorry

end hyperbola_a_value_l18_18373


namespace find_d_l18_18418

theorem find_d :
  ∃ d : ℝ, (∀ x y : ℝ, x^2 + 3 * y^2 + 6 * x - 18 * y + d = 0 → x = -3 ∧ y = 3) ↔ d = -27 :=
by {
  sorry
}

end find_d_l18_18418


namespace chord_length_l18_18546

theorem chord_length (ρ θ : ℝ) (p : ℝ) : 
  (∀ θ, ρ = 6 * Real.cos θ) ∧ (θ = Real.pi / 4) → 
  ∃ l : ℝ, l = 3 * Real.sqrt 2 :=
by
  sorry

end chord_length_l18_18546


namespace express_1997_using_elevent_fours_l18_18271

def number_expression_uses_eleven_fours : Prop :=
  (4 * 444 + 44 * 4 + 44 + 4 / 4 = 1997)
  
theorem express_1997_using_elevent_fours : number_expression_uses_eleven_fours :=
by
  sorry

end express_1997_using_elevent_fours_l18_18271


namespace find_quotient_l18_18646

def dividend : ℝ := 13787
def remainder : ℝ := 14
def divisor : ℝ := 154.75280898876406
def quotient : ℝ := 89

theorem find_quotient :
  (dividend - remainder) / divisor = quotient :=
sorry

end find_quotient_l18_18646


namespace option_B_not_well_defined_l18_18359

-- Definitions based on given conditions 
def is_well_defined_set (description : String) : Prop :=
  match description with
  | "All positive numbers" => True
  | "All elderly people" => False
  | "All real numbers that are not equal to 0" => True
  | "The four great inventions of ancient China" => True
  | _ => False

-- Theorem stating option B "All elderly people" is not a well-defined set
theorem option_B_not_well_defined : ¬ is_well_defined_set "All elderly people" :=
  by sorry

end option_B_not_well_defined_l18_18359


namespace remaining_amount_is_correct_l18_18284

-- Define the original price based on the deposit paid
def original_price : ℝ := 1500

-- Define the discount percentage
def discount_percentage : ℝ := 0.05

-- Define the sales tax percentage
def tax_percentage : ℝ := 0.075

-- Define the deposit already paid
def deposit_paid : ℝ := 150

-- Define the discounted price
def discounted_price : ℝ := original_price * (1 - discount_percentage)

-- Define the sales tax amount
def sales_tax : ℝ := discounted_price * tax_percentage

-- Define the final cost after adding sales tax
def final_cost : ℝ := discounted_price + sales_tax

-- Define the remaining amount to be paid
def remaining_amount : ℝ := final_cost - deposit_paid

-- The statement we need to prove
theorem remaining_amount_is_correct : remaining_amount = 1381.875 :=
by
  -- We'd normally write the proof here, but that's not required for this task.
  sorry

end remaining_amount_is_correct_l18_18284


namespace agnes_weekly_hours_l18_18829

-- Given conditions
def mila_hourly_rate : ℝ := 10
def agnes_hourly_rate : ℝ := 15
def mila_hours_per_month : ℝ := 48

-- Derived condition that Mila's earnings in a month equal Agnes's in a month
def mila_monthly_earnings : ℝ := mila_hourly_rate * mila_hours_per_month

-- Prove that Agnes must work 8 hours each week to match Mila's monthly earnings
theorem agnes_weekly_hours (A : ℝ) : 
  agnes_hourly_rate * 4 * A = mila_monthly_earnings → A = 8 := 
by
  intro h
  -- sorry here is a placeholder for the proof
  sorry

end agnes_weekly_hours_l18_18829


namespace sum_of_dice_not_in_set_l18_18436

theorem sum_of_dice_not_in_set (a b c : ℕ) (h₁ : 1 ≤ a ∧ a ≤ 6) (h₂ : 1 ≤ b ∧ b ≤ 6) (h₃ : 1 ≤ c ∧ c ≤ 6) 
  (h₄ : a * b * c = 72) (h₅ : a = 4 ∨ b = 4 ∨ c = 4) :
  a + b + c ≠ 12 ∧ a + b + c ≠ 14 ∧ a + b + c ≠ 15 ∧ a + b + c ≠ 16 :=
by
  sorry

end sum_of_dice_not_in_set_l18_18436


namespace gcd_45736_123456_l18_18744

theorem gcd_45736_123456 : Nat.gcd 45736 123456 = 352 :=
by sorry

end gcd_45736_123456_l18_18744


namespace cakes_served_yesterday_l18_18561

theorem cakes_served_yesterday:
  ∃ y : ℕ, (5 + 6 + y = 14) ∧ y = 3 := 
by
  sorry

end cakes_served_yesterday_l18_18561


namespace condition_a_gt_1_iff_a_gt_0_l18_18817

theorem condition_a_gt_1_iff_a_gt_0 : ∀ (a : ℝ), (a > 1) ↔ (a > 0) :=
by 
  sorry

end condition_a_gt_1_iff_a_gt_0_l18_18817


namespace inequality_solution_exists_l18_18988

theorem inequality_solution_exists (a : ℝ) : 
  ∃ x : ℝ, x > 2 ∧ x > -1 ∧ x > a := 
by
  sorry

end inequality_solution_exists_l18_18988


namespace length_inequality_l18_18406

noncomputable def l_a (A B C : ℝ) : ℝ := 
  sorry -- Definition according to the mathematical problem

noncomputable def l_b (A B C : ℝ) : ℝ := 
  sorry -- Definition according to the mathematical problem

noncomputable def l_c (A B C : ℝ) : ℝ := 
  sorry -- Definition according to the mathematical problem

noncomputable def perimeter (A B C : ℝ) : ℝ :=
  A + B + C

theorem length_inequality (A B C : ℝ) (hA : A > 0) (hB : B > 0) (hC : C > 0) :
  (l_a A B C * l_b A B C * l_c A B C) / (perimeter A B C)^3 ≤ 1 / 64 :=
by
  sorry

end length_inequality_l18_18406


namespace solution_set_of_inequality_l18_18517

theorem solution_set_of_inequality :
  {x : ℝ | abs (x^2 - 5 * x + 6) < x^2 - 4} = { x : ℝ | x > 2 } :=
sorry

end solution_set_of_inequality_l18_18517


namespace mrs_oaklyn_rugs_l18_18444

theorem mrs_oaklyn_rugs (buying_price selling_price total_profit : ℕ) (h1 : buying_price = 40) (h2 : selling_price = 60) (h3 : total_profit = 400) : 
  ∃ (num_rugs : ℕ), num_rugs = 20 :=
by
  sorry

end mrs_oaklyn_rugs_l18_18444


namespace find_x_l18_18345

variable (x : ℝ)
def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (x, 1)
def u : ℝ × ℝ := (1 + 2 * x, 4)
def v : ℝ × ℝ := (2 - 2 * x, 2)

theorem find_x (h : 2 * (1 + 2 * x) = 4 * (2 - 2 * x)) : x = 1 / 2 := by
  sorry

end find_x_l18_18345


namespace find_number_subtracted_l18_18030

-- Given a number x, where the ratio of the two natural numbers is 6:5,
-- and another number y is subtracted to both numbers such that the new ratio becomes 5:4,
-- and the larger number exceeds the smaller number by 5,
-- prove that y = 5.
theorem find_number_subtracted (x y : ℕ) (h1 : 6 * x - 5 * x = 5) (h2 : (30 - y) * 4 = (25 - y) * 5) : y = 5 :=
by sorry

end find_number_subtracted_l18_18030


namespace packets_of_chips_l18_18312

theorem packets_of_chips (x : ℕ) 
  (h1 : ∀ x, 2 * (x : ℝ) + 1.5 * (10 : ℝ) = 45) : 
  x = 15 := 
by 
  sorry

end packets_of_chips_l18_18312


namespace find_number_l18_18301

theorem find_number (x : ℤ) (h : x = 1) : x + 1 = 2 :=
  by
  sorry

end find_number_l18_18301


namespace total_weight_of_bars_l18_18853

-- Definitions for weights of each gold bar
variables (C1 C2 C3 C4 C5 C6 C7 C8 C9 C10 C11 C12 C13 : ℝ)
variables (W1 W2 W3 W4 W5 W6 W7 W8 : ℝ)

-- Definitions for the weighings
axiom weight_C1_C2 : W1 = C1 + C2
axiom weight_C1_C3 : W2 = C1 + C3
axiom weight_C2_C3 : W3 = C2 + C3
axiom weight_C4_C5 : W4 = C4 + C5
axiom weight_C6_C7 : W5 = C6 + C7
axiom weight_C8_C9 : W6 = C8 + C9
axiom weight_C10_C11 : W7 = C10 + C11
axiom weight_C12_C13 : W8 = C12 + C13

-- Prove the total weight of all gold bars
theorem total_weight_of_bars :
  (C1 + C2 + C3 + C4 + C5 + C6 + C7 + C8 + C9 + C10 + C11 + C12 + C13)
  = (W1 + W2 + W3) / 2 + W4 + W5 + W6 + W7 + W8 :=
by sorry

end total_weight_of_bars_l18_18853


namespace Pythagorean_triple_l18_18386

theorem Pythagorean_triple (n : ℕ) (hn : n % 2 = 1) (hn_geq : n ≥ 3) :
  n^2 + ((n^2 - 1) / 2)^2 = ((n^2 + 1) / 2)^2 := by
  sorry

end Pythagorean_triple_l18_18386


namespace solution_exists_l18_18013

theorem solution_exists (a b c : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (gcd_ca : Nat.gcd c a = 1) (gcd_cb : Nat.gcd c b = 1) : 
  ∃ x y z : ℕ, x > 0 ∧ y > 0 ∧ z > 0 ∧ x^a + y^b = z^c :=
sorry

end solution_exists_l18_18013


namespace barry_should_pay_l18_18105

def original_price : ℝ := 80
def discount_rate : ℝ := 0.15

theorem barry_should_pay:
  original_price * (1 - discount_rate) = 68 := 
by 
  -- Original price: 80
  -- Discount rate: 0.15
  -- Question: Final price after discount
  sorry

end barry_should_pay_l18_18105


namespace product_consecutive_two_digits_l18_18098

theorem product_consecutive_two_digits (a b c : ℕ) : 
  ¬(∃ n : ℕ, (ab % 100 = n ∧ bc % 100 = n + 1 ∧ ac % 100 = n + 2)) :=
by
  sorry

end product_consecutive_two_digits_l18_18098


namespace exists_eight_integers_sum_and_product_eight_l18_18982

theorem exists_eight_integers_sum_and_product_eight :
  ∃ (a1 a2 a3 a4 a5 a6 a7 a8 : ℤ), 
  a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 = 8 ∧ 
  a1 * a2 * a3 * a4 * a5 * a6 * a7 * a8 = 8 :=
by
  -- The existence proof can be constructed here
  sorry

end exists_eight_integers_sum_and_product_eight_l18_18982


namespace find_x_l18_18229

/-- Given vectors a and b, and a is parallel to b -/
def vectors (x : ℝ) : Prop :=
  let a := (x, 2)
  let b := (2, 1)
  a.1 * b.2 = a.2 * b.1

theorem find_x: ∀ x : ℝ, vectors x → x = 4 :=
by
  intros x h
  sorry

end find_x_l18_18229


namespace shyam_weight_increase_l18_18192

theorem shyam_weight_increase (x : ℝ) 
    (h1 : x > 0)
    (ratio : ∀ Ram Shyam : ℝ, (Ram / Shyam) = 7 / 5)
    (ram_increase : ∀ Ram : ℝ, Ram' = Ram + 0.1 * Ram)
    (total_weight_after : Ram' + Shyam' = 82.8)
    (total_weight_increase : 82.8 = 1.15 * total_weight) :
    (Shyam' - Shyam) / Shyam * 100 = 22 :=
by
  sorry

end shyam_weight_increase_l18_18192


namespace alice_speed_exceed_l18_18522

theorem alice_speed_exceed (d : ℝ) (t₁ t₂ : ℝ) (t₃ : ℝ) :
  d = 220 →
  t₁ = 220 / 40 →
  t₂ = t₁ - 0.5 →
  t₃ = 220 / t₂ →
  t₃ = 44 :=
by
  intros h1 h2 h3 h4
  sorry

end alice_speed_exceed_l18_18522


namespace value_of_m_making_365m_divisible_by_12_l18_18000

theorem value_of_m_making_365m_divisible_by_12
  (m : ℕ)
  (h1 : (3650 + m) % 3 = 0)
  (h2 : (50 + m) % 4 = 0) :
  m = 0 :=
sorry

end value_of_m_making_365m_divisible_by_12_l18_18000


namespace greatest_unexpressible_sum_l18_18006

def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ d, d > 1 ∧ d < n ∧ n % d = 0

theorem greatest_unexpressible_sum : 
  ∀ (n : ℕ), (∀ a b : ℕ, is_composite a → is_composite b → a + b ≠ n) → n ≤ 11 :=
by
  sorry

end greatest_unexpressible_sum_l18_18006


namespace ralph_socks_problem_l18_18836

theorem ralph_socks_problem :
  ∃ x y z : ℕ, x + y + z = 10 ∧ x + 2 * y + 4 * z = 30 ∧ 1 ≤ x ∧ 1 ≤ y ∧ 1 ≤ z ∧ x = 2 :=
by
  sorry

end ralph_socks_problem_l18_18836


namespace initial_stock_before_shipment_l18_18686

-- Define the conditions for the problem
def initial_stock (total_shelves new_shipment_bears bears_per_shelf: ℕ) : ℕ :=
  let total_bears_on_shelves := total_shelves * bears_per_shelf
  total_bears_on_shelves - new_shipment_bears

-- State the theorem with the conditions
theorem initial_stock_before_shipment : initial_stock 2 10 7 = 4 := by
  -- Mathematically, the calculation details will be handled here
  sorry

end initial_stock_before_shipment_l18_18686


namespace calc_result_l18_18693

theorem calc_result :
  12 / 4 - 3 - 16 + 4 * 6 = 8 := by
  sorry

end calc_result_l18_18693


namespace neg_p_l18_18310

variable {α : Type}
variable (x : α)

def p (x : Real) : Prop := ∀ x : Real, x > 1 → x^2 - 1 > 0

theorem neg_p : ¬( ∀ x : Real, x > 1 → x^2 - 1 > 0) ↔ ∃ x : Real, x > 1 ∧ x^2 - 1 ≤ 0 := 
by 
  sorry

end neg_p_l18_18310


namespace solution_set_of_inequality_l18_18852

theorem solution_set_of_inequality :
  {x : ℝ | 3 * x ^ 2 - 7 * x - 10 ≥ 0} = {x : ℝ | x ≥ (10 / 3) ∨ x ≤ -1} :=
sorry

end solution_set_of_inequality_l18_18852


namespace arithmetic_sequence_seventh_term_l18_18714

noncomputable def a3 := (2 : ℚ) / 11
noncomputable def a11 := (5 : ℚ) / 6

noncomputable def a7 := (a3 + a11) / 2

theorem arithmetic_sequence_seventh_term :
  a7 = 67 / 132 := by
  sorry

end arithmetic_sequence_seventh_term_l18_18714


namespace calculate_tax_l18_18132

noncomputable def cadastral_value : ℝ := 3000000 -- 3 million rubles
noncomputable def tax_rate : ℝ := 0.001        -- 0.1% converted to decimal
noncomputable def tax : ℝ := cadastral_value * tax_rate -- Tax formula

theorem calculate_tax : tax = 3000 := by
  sorry

end calculate_tax_l18_18132


namespace apples_per_pie_l18_18991

theorem apples_per_pie (total_apples handed_out_apples pies made_pies remaining_apples : ℕ) 
  (h_initial : total_apples = 86)
  (h_handout : handed_out_apples = 30)
  (h_made_pies : made_pies = 7)
  (h_remaining : remaining_apples = total_apples - handed_out_apples) :
  remaining_apples / made_pies = 8 :=
by
  sorry

end apples_per_pie_l18_18991


namespace circle_probability_l18_18812

noncomputable def problem_statement : Prop :=
  let outer_radius := 3
  let inner_radius := 1
  let pivotal_radius := 2
  let outer_area := Real.pi * outer_radius ^ 2
  let inner_area := Real.pi * pivotal_radius ^ 2
  let probability := inner_area / outer_area
  probability = 4 / 9

theorem circle_probability : problem_statement := sorry

end circle_probability_l18_18812


namespace bronze_status_families_count_l18_18796

theorem bronze_status_families_count :
  ∃ B : ℕ, (B * 25) = (700 - (7 * 50 + 1 * 100)) ∧ B = 10 := 
sorry

end bronze_status_families_count_l18_18796


namespace rides_on_roller_coaster_l18_18860

-- Definitions based on the conditions given.
def roller_coaster_cost : ℕ := 17
def total_tickets : ℕ := 255
def tickets_spent_on_other_activities : ℕ := 78

-- The proof statement.
theorem rides_on_roller_coaster : (total_tickets - tickets_spent_on_other_activities) / roller_coaster_cost = 10 :=
by 
  sorry

end rides_on_roller_coaster_l18_18860


namespace values_of_x_plus_y_l18_18532

theorem values_of_x_plus_y (x y : ℝ) (h1 : |x| = 3) (h2 : |y| = 6) (h3 : x > y) : x + y = -3 ∨ x + y = -9 :=
sorry

end values_of_x_plus_y_l18_18532


namespace volume_of_pool_l18_18420

theorem volume_of_pool :
  let diameter := 60
  let radius := diameter / 2
  let height_shallow := 3
  let height_deep := 15
  let height_total := height_shallow + height_deep
  let volume_cylinder := π * radius^2 * height_total
  volume_cylinder / 2 = 8100 * π :=
by
  sorry

end volume_of_pool_l18_18420


namespace part_one_solution_part_two_solution_l18_18877

-- Define the function f(x)
def f (x a : ℝ) : ℝ := |x - a| + |x - 2|

-- Part (1): "When a = 1, find the solution set of the inequality f(x) ≥ 3"
theorem part_one_solution (x : ℝ) : f x 1 ≥ 3 ↔ x ≤ 0 ∨ x ≥ 3 :=
by sorry

-- Part (2): "If f(x) ≥ 2a - 1, find the range of values for a"
theorem part_two_solution (a : ℝ) : (∀ x : ℝ, f x a ≥ 2 * a - 1) ↔ a ≤ 1 :=
by sorry

end part_one_solution_part_two_solution_l18_18877


namespace sum_of_numbers_is_37_l18_18460

theorem sum_of_numbers_is_37 :
  ∃ (A B : ℕ), 
    1 ≤ A ∧ A ≤ 50 ∧ 1 ≤ B ∧ B ≤ 50 ∧ A ≠ B ∧
    (50 * B + A = k^2) ∧ Prime B ∧ B > 10 ∧
    A + B = 37 
  := by
    sorry

end sum_of_numbers_is_37_l18_18460


namespace find_f2_l18_18683

-- Define the function f(x) = ax + b
def f (a b x : ℝ) : ℝ := a * x + b

-- Condition: f'(x) = a
def f_derivative (a b x : ℝ) : ℝ := a

-- Given conditions
variables (a b : ℝ)
axiom h1 : f a b 1 = 2
axiom h2 : f_derivative a b 1 = 2

theorem find_f2 : f a b 2 = 4 :=
by
  sorry

end find_f2_l18_18683


namespace base_conversion_problem_l18_18980

theorem base_conversion_problem (b : ℕ) (h : b^2 + 2 * b - 25 = 0) : b = 3 :=
sorry

end base_conversion_problem_l18_18980


namespace problem_statement_l18_18837

noncomputable def given_expression (x y z : ℝ) : ℝ :=
  (45 + (23 / 89) * Real.sin x) * (4 * y^2 - 7 * z^3)

theorem problem_statement : given_expression (Real.pi / 6) 3 (-2) = 4186 := by
  sorry

end problem_statement_l18_18837


namespace value_at_neg_9_over_2_l18_18690

def f : ℝ → ℝ := sorry 

axiom odd_function (x : ℝ) : f (-x) + f x = 0

axiom symmetric_y_axis (x : ℝ) : f (1 + x) = f (1 - x)

axiom functional_eq (x k : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) (hk : 0 ≤ k ∧ k ≤ 1) : f (k * x) + 1 = (f x + 1) ^ k

axiom f_at_1 : f 1 = - (1 / 2)

theorem value_at_neg_9_over_2 : f (- (9 / 2)) = 1 - (Real.sqrt 2) / 2 := 
sorry

end value_at_neg_9_over_2_l18_18690


namespace ishaan_age_eq_6_l18_18497

-- Variables for ages
variable (I : ℕ) -- Ishaan's current age

-- Constants for ages
def daniel_current_age := 69
def years := 15
def daniel_future_age := daniel_current_age + years

-- Lean theorem statement
theorem ishaan_age_eq_6 
    (h1 : daniel_current_age = 69)
    (h2 : daniel_future_age = 4 * (I + years)) : 
    I = 6 := by
  sorry

end ishaan_age_eq_6_l18_18497


namespace similar_triangles_height_l18_18215

theorem similar_triangles_height (h_small: ℝ) (area_ratio: ℝ) (h_large: ℝ) :
  h_small = 5 ∧ area_ratio = 1/9 ∧ h_large = 3 * h_small → h_large = 15 :=
by
  intro h 
  sorry

end similar_triangles_height_l18_18215


namespace trigonometric_expression_simplification_l18_18136

theorem trigonometric_expression_simplification (θ : ℝ) (h : Real.tan θ = 3) :
  (Real.sin (3 * Real.pi / 2 + θ) + 2 * Real.cos (Real.pi - θ)) /
  (Real.sin (Real.pi / 2 - θ) - Real.sin (Real.pi - θ)) = 3 / 2 := 
sorry

end trigonometric_expression_simplification_l18_18136


namespace none_of_these_l18_18666

noncomputable def x (t : ℝ) : ℝ := t ^ (3 / (t - 1))
noncomputable def y (t : ℝ) : ℝ := t ^ ((t + 1) / (t - 1))

theorem none_of_these (t : ℝ) (ht_pos : t > 0) (ht_ne_one : t ≠ 1) :
  ¬ (y t ^ x t = x t ^ y t) ∧ ¬ (x t ^ x t = y t ^ y t) ∧
  ¬ (x t ^ (y t ^ x t) = y t ^ (x t ^ y t)) ∧ ¬ (x t ^ y t = y t ^ x t) :=
sorry

end none_of_these_l18_18666


namespace expression_evaluation_valid_l18_18892

theorem expression_evaluation_valid (a : ℝ) (h1 : a = 4) :
  (1 + (4 / (a ^ 2 - 4))) * ((a + 2) / a) = 2 := by
  sorry

end expression_evaluation_valid_l18_18892


namespace min_value_of_f_l18_18944

noncomputable def f (x : ℝ) : ℝ := 3 * x + 12 / x ^ 2

theorem min_value_of_f : ∀ x > 0, f x ≥ 9 ∧ (f x = 9 ↔ x = 2) :=
by
  sorry

end min_value_of_f_l18_18944


namespace least_positive_integer_mod_conditions_l18_18396

theorem least_positive_integer_mod_conditions :
  ∃ N : ℕ, (N % 4 = 3) ∧ (N % 5 = 4) ∧ (N % 6 = 5) ∧ (N % 7 = 6) ∧ (N % 11 = 10) ∧ N = 4619 :=
by
  sorry

end least_positive_integer_mod_conditions_l18_18396


namespace rectangle_area_l18_18197

theorem rectangle_area (length : ℝ) (width : ℝ) (increased_width : ℝ) (area : ℝ)
  (h1 : length = 12)
  (h2 : increased_width = width * 1.2)
  (h3 : increased_width = 12)
  (h4 : area = length * width) : 
  area = 120 := 
by
  sorry

end rectangle_area_l18_18197


namespace triangle_inequality_l18_18935

-- Let α, β, γ be the angles of a triangle opposite to its sides with lengths a, b, and c, respectively.
variables (α β γ a b c : ℝ)

-- Assume that α, β, γ are positive.
axiom positive_angles : α > 0 ∧ β > 0 ∧ γ > 0
-- Assume that a, b, c are the sides opposite to angles α, β, γ respectively.
axiom positive_sides : a > 0 ∧ b > 0 ∧ c > 0

theorem triangle_inequality :
  a * (1 / β + 1 / γ) + b * (1 / γ + 1 / α) + c * (1 / α + 1 / β) ≥ 
  2 * (a / α + b / β + c / γ) :=
sorry

end triangle_inequality_l18_18935


namespace find_initial_amount_l18_18362

-- Definitions for conditions
def final_amount : ℝ := 5565
def rate_year1 : ℝ := 0.05
def rate_year2 : ℝ := 0.06

-- Theorem statement to prove the initial amount
theorem find_initial_amount (P : ℝ) 
  (H : final_amount = (P * (1 + rate_year1)) * (1 + rate_year2)) :
  P = 5000 := 
sorry

end find_initial_amount_l18_18362


namespace relationship_among_abc_l18_18007

noncomputable def a : ℝ := 36^(1/5)
noncomputable def b : ℝ := 3^(4/3)
noncomputable def c : ℝ := 9^(2/5)

theorem relationship_among_abc (a_def : a = 36^(1/5)) 
                              (b_def : b = 3^(4/3)) 
                              (c_def : c = 9^(2/5)) : a < c ∧ c < b :=
by
  rw [a_def, b_def, c_def]
  sorry

end relationship_among_abc_l18_18007


namespace mixed_gender_groups_l18_18530

theorem mixed_gender_groups (boys girls : ℕ) (h_boys : boys = 28) (h_girls : girls = 4) :
  ∃ groups : ℕ, (groups ≤ girls) ∧ (groups * 2 ≤ boys) ∧ groups = 4 :=
by
   sorry

end mixed_gender_groups_l18_18530


namespace log_base_change_l18_18480

theorem log_base_change (a b : ℝ) (h₁ : Real.log 2 / Real.log 10 = a) (h₂ : Real.log 3 / Real.log 10 = b) :
    Real.log 18 / Real.log 5 = (a + 2 * b) / (1 - a) := by
  sorry

end log_base_change_l18_18480


namespace distance_between_vertices_hyperbola_l18_18066

-- Defining the hyperbola equation and necessary constants
def hyperbola_eq (x y : ℝ) : Prop :=
  (x^2) / 64 - (y^2) / 81 = 1

-- Proving the distance between the vertices is 16
theorem distance_between_vertices_hyperbola : ∀ x y : ℝ, hyperbola_eq x y → 16 = 16 :=
by
  intros x y h
  sorry

end distance_between_vertices_hyperbola_l18_18066


namespace floor_factorial_even_l18_18902

theorem floor_factorial_even (n : ℕ) (hn : n > 0) : 
  Nat.floor ((Nat.factorial (n - 1) : ℝ) / (n * (n + 1))) % 2 = 0 := 
sorry

end floor_factorial_even_l18_18902


namespace rita_remaining_money_l18_18403

theorem rita_remaining_money :
  let dresses_cost := 5 * 20
  let pants_cost := 3 * 12
  let jackets_cost := 4 * 30
  let transport_cost := 5
  let total_expenses := dresses_cost + pants_cost + jackets_cost + transport_cost
  let initial_money := 400
  let remaining_money := initial_money - total_expenses
  remaining_money = 139 := 
by
  sorry

end rita_remaining_money_l18_18403


namespace monomial_properties_l18_18586

def coefficient (m : String) : ℤ := 
  if m = "-2xy^3" then -2 
  else sorry

def degree (m : String) : ℕ := 
  if m = "-2xy^3" then 4 
  else sorry

theorem monomial_properties : coefficient "-2xy^3" = -2 ∧ degree "-2xy^3" = 4 := 
by 
  exact ⟨rfl, rfl⟩

end monomial_properties_l18_18586


namespace largest_n_for_factorable_polynomial_l18_18149

theorem largest_n_for_factorable_polynomial : ∃ n, 
  (∀ A B : ℤ, (6 * B + A = n) → (A * B = 144)) ∧ 
  (∀ n', (∀ A B : ℤ, (6 * B + A = n') → (A * B = 144)) → n' ≤ n) ∧ 
  (n = 865) :=
by
  sorry

end largest_n_for_factorable_polynomial_l18_18149


namespace area_of_rectangle_perimeter_of_rectangle_l18_18992

-- Define the input conditions
variables (AB AC BC : ℕ)
def is_right_triangle (a b c : ℕ) : Prop := a * a + b * b = c * c
def area_rect (l w : ℕ) : ℕ := l * w
def perimeter_rect (l w : ℕ) : ℕ := 2 * (l + w)

-- Given the conditions for the problem
axiom AB_eq_15 : AB = 15
axiom AC_eq_17 : AC = 17
axiom right_triangle : is_right_triangle AB BC AC

-- Prove the area and perimeter of the rectangle
theorem area_of_rectangle : area_rect AB BC = 120 := by sorry

theorem perimeter_of_rectangle : perimeter_rect AB BC = 46 := by sorry

end area_of_rectangle_perimeter_of_rectangle_l18_18992


namespace xy_value_l18_18174

theorem xy_value (x y : ℝ) (h : |x - 5| + |y + 3| = 0) : x * y = -15 := by
  sorry

end xy_value_l18_18174


namespace feed_days_l18_18879

theorem feed_days (morning_food evening_food total_food : ℕ) (h1 : morning_food = 1) (h2 : evening_food = 1) (h3 : total_food = 32)
: (total_food / (morning_food + evening_food)) = 16 := by
  sorry

end feed_days_l18_18879


namespace child_ticket_cost_l18_18788

def cost_of_adult_ticket : ℕ := 22
def number_of_adults : ℕ := 2
def number_of_children : ℕ := 2
def total_family_cost : ℕ := 58
def cost_of_child_ticket : ℕ := 7

theorem child_ticket_cost :
  2 * cost_of_adult_ticket + number_of_children * cost_of_child_ticket = total_family_cost :=
by
  sorry

end child_ticket_cost_l18_18788


namespace p_sufficient_but_not_necessary_for_q_l18_18985

def p (x : ℝ) : Prop := -1 ≤ x ∧ x ≤ 5
def q (x : ℝ) : Prop := (x - 5) * (x + 1) < 0

theorem p_sufficient_but_not_necessary_for_q :
  (∀ x : ℝ, p x → q x) ∧ ∃ x : ℝ, q x ∧ ¬ p x :=
by
  sorry

end p_sufficient_but_not_necessary_for_q_l18_18985


namespace train_length_l18_18536

/-- Given problem conditions -/
def speed_kmh := 72
def length_platform_m := 270
def time_sec := 26

/-- Convert speed to meters per second -/
def speed_mps := speed_kmh * 1000 / 3600

/-- Calculate the total distance covered -/
def distance_covered := speed_mps * time_sec

theorem train_length :
  (distance_covered - length_platform_m) = 250 :=
by
  sorry

end train_length_l18_18536


namespace number_of_honey_bees_l18_18283

theorem number_of_honey_bees (total_honey : ℕ) (honey_one_bee : ℕ) (days : ℕ) (h1 : total_honey = 30) (h2 : honey_one_bee = 1) (h3 : days = 30) : 
  (total_honey / honey_one_bee) = 30 :=
by
  -- Given total_honey = 30 grams in 30 days
  -- Given honey_one_bee = 1 gram in 30 days
  -- We need to prove (total_honey / honey_one_bee) = 30
  sorry

end number_of_honey_bees_l18_18283


namespace positive_integer_pairs_l18_18865

theorem positive_integer_pairs (m n : ℕ) (p : ℕ) (hp_prime : Prime p) (h_diff : m - n = p) (h_square : ∃ k : ℕ, m * n = k^2) :
  ∃ p' : ℕ, (Prime p') ∧ m = (p' + 1) / 2 ^ 2 ∧ n = (p' - 1) / 2 ^ 2 :=
sorry

end positive_integer_pairs_l18_18865


namespace elena_total_pens_l18_18018

theorem elena_total_pens 
  (cost_X : ℝ) (cost_Y : ℝ) (total_spent : ℝ) (num_brand_X : ℕ) (num_brand_Y : ℕ) (total_pens : ℕ)
  (h1 : cost_X = 4.0) 
  (h2 : cost_Y = 2.8) 
  (h3 : total_spent = 40.0) 
  (h4 : num_brand_X = 8) 
  (h5 : total_pens = num_brand_X + num_brand_Y) 
  (h6 : total_spent = num_brand_X * cost_X + num_brand_Y * cost_Y) :
  total_pens = 10 :=
sorry

end elena_total_pens_l18_18018


namespace number_of_triangles_from_8_points_on_circle_l18_18500

-- Definitions based on the problem conditions
def points_on_circle : ℕ := 8

-- Problem statement without the proof
theorem number_of_triangles_from_8_points_on_circle :
  ∃ n : ℕ, n = (points_on_circle.choose 3) ∧ n = 56 := 
by
  sorry

end number_of_triangles_from_8_points_on_circle_l18_18500


namespace find_window_width_on_second_wall_l18_18859

noncomputable def total_wall_area (width length height: ℝ) : ℝ :=
  4 * width * height

noncomputable def doorway_area (width height : ℝ) : ℝ :=
  width * height

noncomputable def window_area (width height : ℝ) : ℝ :=
  width * height

theorem find_window_width_on_second_wall :
  let room_width := 20
  let room_length := 20
  let room_height := 8
  let first_doorway_width := 3
  let first_doorway_height := 7
  let second_doorway_width := 5
  let second_doorway_height := 7
  let window_height := 4
  let area_to_paint := 560
  let total_area := total_wall_area room_width room_length room_height
  let first_doorway := doorway_area first_doorway_width first_doorway_height
  let second_doorway := doorway_area second_doorway_width second_doorway_height
  total_area - first_doorway - second_doorway - window_area w window_height = area_to_paint
  → w = 6 :=
by
  let room_width := 20
  let room_length := 20
  let room_height := 8
  let first_doorway_width := 3
  let first_doorway_height := 7
  let second_doorway_width := 5
  let second_doorway_height := 7
  let window_height := 4
  let area_to_paint := 560
  let total_area := total_wall_area room_width room_length room_height
  let first_doorway := doorway_area first_doorway_width first_doorway_height
  let second_doorway := doorway_area second_doorway_width second_doorway_height
  sorry

end find_window_width_on_second_wall_l18_18859


namespace sue_cost_l18_18667

def cost_of_car : ℝ := 2100
def total_days_in_week : ℝ := 7
def sue_days : ℝ := 3

theorem sue_cost : (cost_of_car * (sue_days / total_days_in_week)) = 899.99 :=
by
  sorry

end sue_cost_l18_18667


namespace find_a_l18_18048

theorem find_a (a x y : ℝ) 
  (h1 : (|y + 9| + |x + 2| - 2) * (x^2 + y^2 - 3) = 0) 
  (h2 : (x + 2)^2 + (y + 4)^2 = a) 
  (h3 : ∃! x y, (|y + 9| + |x + 2| - 2) * (x^2 + y^2 - 3) = 0 ∧ (x + 2)^2 + (y + 4)^2 = a) :
  a = 9 ∨ a = 23 + 4 * Real.sqrt 15 :=
sorry

end find_a_l18_18048


namespace biker_distance_and_speed_l18_18758

variable (D V : ℝ)

theorem biker_distance_and_speed (h1 : D / 2 = V * 2.5)
                                  (h2 : D / 2 = (V + 2) * (7 / 3)) :
  D = 140 ∧ V = 28 :=
by
  sorry

end biker_distance_and_speed_l18_18758


namespace Foster_Farms_donated_45_chickens_l18_18874

def number_of_dressed_chickens_donated_by_Foster_Farms (C AS H BB D : ℕ) : Prop :=
  C + AS + H + BB + D = 375 ∧
  AS = 2 * C ∧
  H = 3 * C ∧
  BB = C ∧
  D = 2 * C - 30

theorem Foster_Farms_donated_45_chickens:
  ∃ C, number_of_dressed_chickens_donated_by_Foster_Farms C (2*C) (3*C) C (2*C - 30) ∧ C = 45 :=
by 
  sorry

end Foster_Farms_donated_45_chickens_l18_18874


namespace sum_is_correct_l18_18916

-- Define the variables and conditions
variables (a b c d : ℝ)
variable (x : ℝ)

-- Define the condition
def condition : Prop :=
  a + 1 = x ∧
  b + 2 = x ∧
  c + 3 = x ∧
  d + 4 = x ∧
  a + b + c + d + 5 = x

-- The theorem we need to prove
theorem sum_is_correct (h : condition a b c d x) : a + b + c + d = -10 / 3 :=
  sorry

end sum_is_correct_l18_18916


namespace max_value_expr_l18_18077

theorem max_value_expr (a b c d : ℝ) (ha : -12.5 ≤ a ∧ a ≤ 12.5) (hb : -12.5 ≤ b ∧ b ≤ 12.5) (hc : -12.5 ≤ c ∧ c ≤ 12.5) (hd : -12.5 ≤ d ∧ d ≤ 12.5) :
  a + 2 * b + c + 2 * d - a * b - b * c - c * d - d * a ≤ 650 :=
sorry

end max_value_expr_l18_18077


namespace bisection_method_root_interval_l18_18392

noncomputable def f (x : ℝ) : ℝ := 2^x + 3 * x - 7

theorem bisection_method_root_interval :
  f 1 < 0 → f 2 > 0 → f 3 > 0 → ∃ (c : ℝ), 1 < c ∧ c < 2 ∧ f c = 0 :=
by
  sorry

end bisection_method_root_interval_l18_18392


namespace simplify_and_evaluate_expression_l18_18371

theorem simplify_and_evaluate_expression :
  ∀ (x y : ℝ), 
  x = -1 / 3 → y = -2 → 
  (3 * x + 2 * y) * (3 * x - 2 * y) - 5 * x * (x - y) - (2 * x - y)^2 = -14 :=
by
  intros x y hx hy
  sorry

end simplify_and_evaluate_expression_l18_18371


namespace inequality_unequal_positive_numbers_l18_18323

theorem inequality_unequal_positive_numbers (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a ≠ b) : 
  (a + b) / 2 > (2 * a * b) / (a + b) :=
by
sorry

end inequality_unequal_positive_numbers_l18_18323


namespace youseff_distance_l18_18709

theorem youseff_distance (x : ℕ) 
  (walk_time_per_block : ℕ := 1)
  (bike_time_per_block_secs : ℕ := 20)
  (time_difference : ℕ := 12) :
  (x : ℕ) = 18 :=
by
  -- walking time
  let walk_time := x * walk_time_per_block
  
  -- convert bike time per block to minutes
  let bike_time_per_block := (bike_time_per_block_secs : ℚ) / 60

  -- biking time
  let bike_time := x * bike_time_per_block

  -- set up the equation for time difference
  have time_eq := walk_time - bike_time = time_difference
  
  -- from here, the actual proof steps would follow, 
  -- but we include "sorry" as a placeholder since the focus is on the statement.
  sorry

end youseff_distance_l18_18709


namespace find_length_QS_l18_18521

theorem find_length_QS 
  (cosR : ℝ) (RS : ℝ) (QR : ℝ) (QS : ℝ)
  (h1 : cosR = 3 / 5)
  (h2 : RS = 10)
  (h3 : cosR = QR / RS) :
  QS = 8 :=
by
  sorry

end find_length_QS_l18_18521


namespace two_wheeler_wheels_l18_18636

-- Define the total number of wheels and the number of four-wheelers
def total_wheels : Nat := 46
def num_four_wheelers : Nat := 11

-- Define the number of wheels per vehicle type
def wheels_per_four_wheeler : Nat := 4
def wheels_per_two_wheeler : Nat := 2

-- Define the number of two-wheelers
def num_two_wheelers : Nat := (total_wheels - num_four_wheelers * wheels_per_four_wheeler) / wheels_per_two_wheeler

-- Proposition stating the number of wheels of the two-wheeler
theorem two_wheeler_wheels : wheels_per_two_wheeler * num_two_wheelers = 2 := by
  sorry

end two_wheeler_wheels_l18_18636


namespace prism_cut_out_l18_18218

theorem prism_cut_out (x y : ℕ)
  (H1 : 15 * 5 * 4 - y * 5 * x = 120)
  (H2 : x < 4) :
  x = 3 ∧ y = 12 :=
sorry

end prism_cut_out_l18_18218


namespace roots_sum_condition_l18_18458

theorem roots_sum_condition (a b : ℝ) 
  (h1 : ∃ (x y z : ℝ), (x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ x > 0 ∧ y > 0 ∧ z > 0 ∧ x + y + z = 9) 
    ∧ (x * y + y * z + x * z = a) ∧ (x * y * z = b)) :
  a + b = 38 := 
sorry

end roots_sum_condition_l18_18458


namespace olympic_iberic_sets_containing_33_l18_18940

/-- A set of positive integers is iberic if it is a subset of {2, 3, ..., 2018},
    and whenever m, n are both in the set, gcd(m, n) is also in the set. -/
def is_iberic_set (X : Set ℕ) : Prop :=
  X ⊆ {n | 2 ≤ n ∧ n ≤ 2018} ∧ ∀ m n, m ∈ X → n ∈ X → Nat.gcd m n ∈ X

/-- An iberic set is olympic if it is not properly contained in any other iberic set. -/
def is_olympic_set (X : Set ℕ) : Prop :=
  is_iberic_set X ∧ ∀ Y, is_iberic_set Y → X ⊂ Y → False

/-- The olympic iberic sets containing 33 are exactly {3, 6, 9, ..., 2016} and {11, 22, 33, ..., 2013}. -/
theorem olympic_iberic_sets_containing_33 :
  ∀ X, is_iberic_set X ∧ 33 ∈ X → X = {n | 3 ∣ n ∧ 2 ≤ n ∧ n ≤ 2016} ∨ X = {n | 11 ∣ n ∧ 11 ≤ n ∧ n ≤ 2013} :=
by
  sorry

end olympic_iberic_sets_containing_33_l18_18940


namespace lucas_can_afford_book_l18_18240

-- Definitions from the conditions
def book_cost : ℝ := 28.50
def two_ten_dollar_bills : ℝ := 2 * 10
def five_one_dollar_bills : ℝ := 5 * 1
def six_quarters : ℝ := 6 * 0.25
def nickel_value : ℝ := 0.05

-- Given the conditions, we need to prove that if Lucas has at least 40 nickels, he can afford the book.
theorem lucas_can_afford_book (m : ℝ) (h : m >= 40) : 
  (two_ten_dollar_bills + five_one_dollar_bills + six_quarters + m * nickel_value) >= book_cost :=
by {
  sorry
}

end lucas_can_afford_book_l18_18240


namespace sum_of_roots_l18_18394

theorem sum_of_roots (a b c : ℝ) (h : 6 * a^3 + 7 * a^2 - 12 * a = 0) : 
  - (7 / 6 : ℝ) = -1.17 := 
sorry

end sum_of_roots_l18_18394


namespace evaluate_combinations_l18_18608

theorem evaluate_combinations (n : ℕ) (h1 : 0 ≤ 5 - n) (h2 : 5 - n ≤ n) (h3 : 0 ≤ 10 - n) (h4 : 10 - n ≤ n + 1) (h5 : n > 0) :
  Nat.choose n (5 - n) + Nat.choose (n + 1) (10 - n) = 7 :=
sorry

end evaluate_combinations_l18_18608


namespace percentage_decrease_is_24_l18_18921

-- Define the given constants Rs. 820 and Rs. 1078.95
def current_price : ℝ := 820
def original_price : ℝ := 1078.95

-- Define the percentage decrease P
def percentage_decrease (P : ℝ) : Prop :=
  original_price - (P / 100) * original_price = current_price

-- Prove that percentage decrease P is approximately 24
theorem percentage_decrease_is_24 : percentage_decrease 24 :=
by
  unfold percentage_decrease
  sorry

end percentage_decrease_is_24_l18_18921


namespace find_number_l18_18346

theorem find_number (x : ℝ) : 0.75 * x + 2 = 8 → x = 8 :=
by
  sorry

end find_number_l18_18346


namespace carp_and_population_l18_18526

-- Define the characteristics of an individual and a population
structure Individual where
  birth : Prop
  death : Prop
  gender : Prop
  age : Prop

structure Population where
  birth_rate : Prop
  death_rate : Prop
  gender_ratio : Prop
  age_composition : Prop

-- Define the conditions as hypotheses
axiom a : Individual
axiom b : Population

-- State the theorem: If "a" has characteristics of an individual and "b" has characteristics
-- of a population, then "a" is a carp and "b" is a carp population
theorem carp_and_population : 
  (a.birth ∧ a.death ∧ a.gender ∧ a.age) ∧
  (b.birth_rate ∧ b.death_rate ∧ b.gender_ratio ∧ b.age_composition) →
  (a = ⟨True, True, True, True⟩ ∧ b = ⟨True, True, True, True⟩) := 
by 
  sorry

end carp_and_population_l18_18526


namespace second_integer_is_64_l18_18934

theorem second_integer_is_64
  (n : ℤ)
  (h1 : (n - 2) + (n + 2) = 128) :
  n = 64 := 
  sorry

end second_integer_is_64_l18_18934


namespace find_k_l18_18777

def triangle_sides (a b c : ℕ) : Prop :=
a < b + c ∧ b < a + c ∧ c < a + b

def is_right_triangle (a b c : ℕ) : Prop :=
a * a + b * b = c * c

def angle_bisector_length (a b c l : ℕ) : Prop :=
∃ k : ℚ, l = k * Real.sqrt 2 ∧ k = 5 / 2

theorem find_k :
  ∀ (AB BC AC BD : ℕ),
  triangle_sides AB BC AC ∧ is_right_triangle AB BC AC ∧
  AB = 5 ∧ BC = 12 ∧ AC = 13 ∧ angle_bisector_length 5 12 13 BD →
  ∃ k : ℚ, BD = k * Real.sqrt 2 ∧ k = 5 / 2 := by
  sorry

end find_k_l18_18777


namespace yellow_area_is_1_5625_percent_l18_18823

def square_flag_area (s : ℝ) : ℝ := s ^ 2

def cross_yellow_occupies_25_percent (s : ℝ) (w : ℝ) : Prop :=
  4 * w * s - 4 * w ^ 2 = 0.25 * s ^ 2

def yellow_area (s w : ℝ) : ℝ := 4 * w ^ 2

def percent_of_flag_area_is_yellow (s w : ℝ) : Prop :=
  yellow_area s w = 0.015625 * s ^ 2

theorem yellow_area_is_1_5625_percent (s w : ℝ) (h1: cross_yellow_occupies_25_percent s w) : 
  percent_of_flag_area_is_yellow s w :=
by sorry

end yellow_area_is_1_5625_percent_l18_18823


namespace habitable_fraction_of_earth_l18_18468

theorem habitable_fraction_of_earth :
  (1 / 2) * (1 / 4) = 1 / 8 := by
  sorry

end habitable_fraction_of_earth_l18_18468


namespace six_x_mod_nine_l18_18975

theorem six_x_mod_nine (x : ℕ) (k : ℕ) (hx : x = 9 * k + 5) : (6 * x) % 9 = 3 :=
by
  sorry

end six_x_mod_nine_l18_18975


namespace yard_length_l18_18685

-- Define the given conditions
def num_trees : ℕ := 26
def dist_between_trees : ℕ := 13

-- Calculate the length of the yard
def num_gaps : ℕ := num_trees - 1
def length_of_yard : ℕ := num_gaps * dist_between_trees

-- Theorem statement: the length of the yard is 325 meters
theorem yard_length : length_of_yard = 325 := by
  sorry

end yard_length_l18_18685


namespace kids_prefer_peas_l18_18334

variable (total_kids children_prefer_carrots children_prefer_corn : ℕ)

theorem kids_prefer_peas (H1 : children_prefer_carrots = 9)
(H2 : children_prefer_corn = 5)
(H3 : children_prefer_corn * 4 = total_kids) :
total_kids - (children_prefer_carrots + children_prefer_corn) = 6 := by
sorry

end kids_prefer_peas_l18_18334


namespace avery_egg_cartons_l18_18871

theorem avery_egg_cartons 
  (num_chickens : ℕ) (eggs_per_chicken : ℕ) (carton_capacity : ℕ)
  (h1 : num_chickens = 20) (h2 : eggs_per_chicken = 6) (h3 : carton_capacity = 12) :
  (num_chickens * eggs_per_chicken) / carton_capacity = 10 :=
by sorry

end avery_egg_cartons_l18_18871


namespace find_k_l18_18911

theorem find_k (x y k : ℝ)
  (h1 : 3 * x + 2 * y = k + 1)
  (h2 : 2 * x + 3 * y = k)
  (h3 : x + y = 3) : k = 7 := sorry

end find_k_l18_18911


namespace ratio_jl_jm_l18_18996

-- Define the side length of the square NOPQ as s
variable (s : ℝ)

-- Define the length (l) and width (m) of the rectangle JKLM
variable (l m : ℝ)

-- Conditions given in the problem
variable (area_overlap : ℝ)
variable (area_condition1 : area_overlap = 0.25 * s * s)
variable (area_condition2 : area_overlap = 0.40 * l * m)

theorem ratio_jl_jm (h1 : area_overlap = 0.25 * s * s) (h2 : area_overlap = 0.40 * l * m) : l / m = 2 / 5 :=
by
  sorry

end ratio_jl_jm_l18_18996


namespace total_stickers_l18_18450

def stickers_in_first_box : ℕ := 23
def stickers_in_second_box : ℕ := stickers_in_first_box + 12

theorem total_stickers :
  stickers_in_first_box + stickers_in_second_box = 58 := 
by
  sorry

end total_stickers_l18_18450


namespace sugar_needed_for_third_layer_l18_18140

-- Let cups be the amount of sugar, and define the layers
def first_layer_sugar : ℕ := 2
def second_layer_sugar : ℕ := 2 * first_layer_sugar
def third_layer_sugar : ℕ := 3 * second_layer_sugar

-- The theorem we want to prove
theorem sugar_needed_for_third_layer : third_layer_sugar = 12 := by
  sorry

end sugar_needed_for_third_layer_l18_18140


namespace order_of_three_numbers_l18_18226

theorem order_of_three_numbers :
  let a := (7 : ℝ) ^ (0.3 : ℝ)
  let b := (0.3 : ℝ) ^ (7 : ℝ)
  let c := Real.log (0.3 : ℝ)
  a > b ∧ b > c ∧ a > c :=
by
  sorry

end order_of_three_numbers_l18_18226


namespace solution_set_of_inequality_group_l18_18601

theorem solution_set_of_inequality_group (x : ℝ) : (x > -3 ∧ x < 5) ↔ (-3 < x ∧ x < 5) :=
by
  sorry

end solution_set_of_inequality_group_l18_18601


namespace max_abs_c_l18_18602

theorem max_abs_c (a b c d e : ℝ) (h : ∀ (x : ℝ), -1 ≤ x ∧ x ≤ 1 → -1 ≤ a * x^4 + b * x^3 + c * x^2 + d * x + e ∧ a * x^4 + b * x^3 + c * x^2 + d * x + e ≤ 1) : abs c ≤ 8 :=
by {
  sorry
}

end max_abs_c_l18_18602


namespace finish_time_is_1_10_PM_l18_18165

-- Definitions of the problem conditions
def start_time := 9 * 60 -- 9:00 AM in minutes past midnight
def third_task_finish_time := 11 * 60 + 30 -- 11:30 AM in minutes past midnight
def num_tasks := 5
def tasks1_to_3_duration := third_task_finish_time - start_time
def one_task_duration := tasks1_to_3_duration / 3
def total_duration := one_task_duration * num_tasks

-- Statement to prove the final time when John finishes the fifth task
theorem finish_time_is_1_10_PM : 
  start_time + total_duration = 13 * 60 + 10 := 
by 
  sorry

end finish_time_is_1_10_PM_l18_18165


namespace remove_parentheses_l18_18769

variable (a b c : ℝ)

theorem remove_parentheses :
  -3 * a - (2 * b - c) = -3 * a - 2 * b + c :=
by
  sorry

end remove_parentheses_l18_18769


namespace determine_base_l18_18087

theorem determine_base (r : ℕ) (a b x : ℕ) (h₁ : r ≤ 100) 
  (h₂ : x = a * r + a) (h₃ : a < r) (h₄ : a > 0) 
  (h₅ : x^2 = b * r^3 + b) : r = 2 ∨ r = 23 :=
by
  sorry

end determine_base_l18_18087


namespace find_divisor_l18_18857

theorem find_divisor
  (n : ℕ) (h1 : n > 0)
  (h2 : (n + 1) % 6 = 4)
  (h3 : ∃ d : ℕ, n % d = 1) :
  ∃ d : ℕ, (n % d = 1) ∧ d = 2 :=
by
  sorry

end find_divisor_l18_18857


namespace lazy_worker_days_worked_l18_18938

theorem lazy_worker_days_worked :
  ∃ x : ℕ, 24 * x - 6 * (30 - x) = 0 ∧ x = 6 :=
by
  existsi 6
  sorry

end lazy_worker_days_worked_l18_18938


namespace least_common_multiple_of_wang_numbers_l18_18701

noncomputable def wang_numbers (n : ℕ) : List ℕ :=
  -- A function that returns the wang numbers in the set from 1 to n
  sorry

noncomputable def LCM (list : List ℕ) : ℕ :=
  -- A function that computes the least common multiple of a list of natural numbers
  sorry

theorem least_common_multiple_of_wang_numbers :
  LCM (wang_numbers 100) = 10080 :=
sorry

end least_common_multiple_of_wang_numbers_l18_18701


namespace range_of_composite_function_l18_18060

noncomputable def range_of_function : Set ℝ :=
  {y | ∃ x : ℝ, y = (1/2) ^ (|x + 1|)}

theorem range_of_composite_function : range_of_function = Set.Ioc 0 1 :=
by
  sorry

end range_of_composite_function_l18_18060


namespace motherGaveMoney_l18_18477

-- Define the given constants and fact
def initialMoney : Real := 0.85
def foundMoney : Real := 0.50
def toyCost : Real := 1.60
def remainingMoney : Real := 0.15

-- Define the unknown amount given by his mother
def motherMoney (M : Real) := initialMoney + M + foundMoney - toyCost = remainingMoney

-- Statement to prove
theorem motherGaveMoney : ∃ M : Real, motherMoney M ∧ M = 0.40 :=
by
  sorry

end motherGaveMoney_l18_18477


namespace symmetric_point_l18_18281

theorem symmetric_point (P : ℝ × ℝ) (a b : ℝ) (h1 : P = (2, 7)) (h2 : 1 * (a - 2) + (b - 7) * (-1) = 0) (h3 : (a + 2) / 2 + (b + 7) / 2 + 1 = 0) :
  (a, b) = (-8, -3) :=
sorry

end symmetric_point_l18_18281


namespace marble_selection_l18_18425

-- Definitions based on the conditions
def total_marbles : ℕ := 15
def special_marbles : ℕ := 4
def other_marbles : ℕ := total_marbles - special_marbles

-- Define combination function for ease of use in the theorem
def combination (n k : ℕ) : ℕ :=
  Nat.choose n k

-- Statement of the theorem based on the question and the correct answer
theorem marble_selection : combination other_marbles 4 * special_marbles = 1320 := by
  -- Define specific values based on the problem
  have other_marbles_val : other_marbles = 11 := rfl
  have comb_11_4 : combination 11 4 = 330 := by
    rw [combination]
    rfl
  rw [other_marbles_val, comb_11_4]
  norm_num
  sorry

end marble_selection_l18_18425


namespace find_a_9_l18_18761

variable {a : ℕ → ℤ}
variable {S : ℕ → ℤ}
variable (d : ℤ)

-- Assumptions and definitions from the problem
def arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop := ∀ n : ℕ, a (n + 1) = a n + d
def sum_of_arithmetic_sequence (S : ℕ → ℤ) (a : ℕ → ℤ) : Prop := ∀ n : ℕ, S n = n * (a 1 + a n) / 2
def condition_one (a : ℕ → ℤ) : Prop := (a 1) + (a 2)^2 = -3
def condition_two (S : ℕ → ℤ) : Prop := S 5 = 10

-- Main theorem statement
theorem find_a_9 (h_arithmetic : arithmetic_sequence a d)
                 (h_sum : sum_of_arithmetic_sequence S a)
                 (h_cond1 : condition_one a)
                 (h_cond2 : condition_two S) : a 9 = 20 := 
sorry

end find_a_9_l18_18761


namespace result_when_decreased_by_5_and_divided_by_7_l18_18820

theorem result_when_decreased_by_5_and_divided_by_7 (x y : ℤ)
  (h1 : (x - 5) / 7 = y)
  (h2 : (x - 6) / 8 = 6) :
  y = 7 :=
by
  sorry

end result_when_decreased_by_5_and_divided_by_7_l18_18820


namespace bryden_collection_value_l18_18700

-- Define the conditions
def face_value_half_dollar : ℝ := 0.5
def face_value_quarter : ℝ := 0.25
def num_half_dollars : ℕ := 5
def num_quarters : ℕ := 3
def multiplier : ℝ := 30

-- Define the problem statement as a theorem
theorem bryden_collection_value : 
  (multiplier * (num_half_dollars * face_value_half_dollar + num_quarters * face_value_quarter)) = 97.5 :=
by
  -- Proof is skipped since it's not required
  sorry

end bryden_collection_value_l18_18700


namespace A_neg10_3_eq_neg1320_l18_18809

noncomputable def A (x : ℝ) (m : ℕ) : ℝ :=
  if m = 0 then 1 else x * A (x - 1) (m - 1)

theorem A_neg10_3_eq_neg1320 : A (-10) 3 = -1320 := 
by
  sorry

end A_neg10_3_eq_neg1320_l18_18809


namespace gcd_g102_g103_eq_one_l18_18052

def g (x : ℤ) : ℤ := x^2 - 2*x + 2023

theorem gcd_g102_g103_eq_one : Nat.gcd (g 102).natAbs (g 103).natAbs = 1 := by
  sorry

end gcd_g102_g103_eq_one_l18_18052


namespace remainder_when_divided_by_13_l18_18906

theorem remainder_when_divided_by_13 (N : ℤ) (k : ℤ) (h : N = 39 * k + 17) : 
  N % 13 = 4 :=
by
  sorry

end remainder_when_divided_by_13_l18_18906


namespace sandra_oranges_l18_18170

theorem sandra_oranges (S E B: ℕ) (h1: E = 7 * S) (h2: E = 252) (h3: B = 12) : S / B = 3 := by
  sorry

end sandra_oranges_l18_18170


namespace larger_number_is_l18_18695

-- Given definitions and conditions
def HCF (a b: ℕ) : ℕ := 23
def other_factor_1 : ℕ := 11
def other_factor_2 : ℕ := 12
def LCM (a b: ℕ) : ℕ := HCF a b * other_factor_1 * other_factor_2

-- Statement to be proven
theorem larger_number_is (a b: ℕ) (h: HCF a b = 23) (hA: a = 23 * 12) (hB: b ∣ a) : a = 276 :=
by { sorry }

end larger_number_is_l18_18695


namespace middle_number_in_8th_row_l18_18328

-- Define a function that describes the number on the far right of the nth row.
def far_right_number (n : ℕ) : ℕ := n^2

-- Define a function that calculates the number of elements in the nth row.
def row_length (n : ℕ) : ℕ := 2 * n - 1

-- Define the middle number in the nth row.
def middle_number (n : ℕ) : ℕ := 
  let mid_index := (row_length n + 1) / 2
  far_right_number (n - 1) + mid_index

-- Statement to prove the middle number in the 8th row is 57
theorem middle_number_in_8th_row : middle_number 8 = 57 :=
by
  -- Placeholder for proof
  sorry

end middle_number_in_8th_row_l18_18328


namespace train_time_to_B_l18_18808

theorem train_time_to_B (T : ℝ) (M : ℝ) :
  (∃ (D : ℝ), (T + 5) * (D + M) / T = 6 * M ∧ 2 * D = 5 * M) → T = 7 :=
by
  sorry

end train_time_to_B_l18_18808


namespace measure_angle_R_l18_18219

-- Given conditions
variables {P Q R : Type}
variable {x : ℝ} -- x represents the measure of angles P and Q

-- Setting up the given conditions
def isosceles_triangle (P Q R : Type) (x : ℝ) : Prop :=
  x + x + (x + 40) = 180

-- Statement we need to prove
theorem measure_angle_R (P Q R : Type) (x : ℝ) (h : isosceles_triangle P Q R x) : ∃ r : ℝ, r = 86.67 :=
by {
  sorry
}

end measure_angle_R_l18_18219


namespace max_possible_acute_angled_triangles_l18_18220
-- Define the sets of points on lines a and b
def maxAcuteAngledTriangles (n : Nat) : Nat :=
  let sum1 := (n * (n - 1) / 2)  -- Sum of first (n-1) natural numbers
  let sum2 := (sum1 * 50) - (n * (n - 1) * (2 * n - 1) / 6) -- Applying the given formula
  (2 * sum2)  -- Multiply by 2 for both colors of alternating points

-- Define the main theorem
theorem max_possible_acute_angled_triangles : maxAcuteAngledTriangles 50 = 41650 := by
  sorry

end max_possible_acute_angled_triangles_l18_18220


namespace simplify_fraction_l18_18738

variable (x y : ℝ)
variable (h1 : x ≠ 0)
variable (h2 : y ≠ 0)
variable (h3 : x - y^2 ≠ 0)

theorem simplify_fraction :
  (y^2 - 1/x) / (x - y^2) = (x * y^2 - 1) / (x^2 - x * y^2) :=
by
  sorry

end simplify_fraction_l18_18738


namespace restore_original_price_l18_18930

def price_after_increases (p : ℝ) : ℝ :=
  let p1 := p * 1.10
  let p2 := p1 * 1.10
  let p3 := p2 * 1.05
  p3

theorem restore_original_price (p : ℝ) (h : p = 1) : 
  ∃ x : ℝ, x = 22 ∧ (price_after_increases p) * (1 - x / 100) = 1 := 
by 
  sorry

end restore_original_price_l18_18930


namespace cows_on_farm_l18_18932

theorem cows_on_farm (weekly_production_per_6_cows : ℕ) 
                     (production_over_5_weeks : ℕ) 
                     (number_of_weeks : ℕ) 
                     (cows : ℕ) :
  weekly_production_per_6_cows = 108 →
  production_over_5_weeks = 2160 →
  number_of_weeks = 5 →
  (cows * (weekly_production_per_6_cows / 6) * number_of_weeks = production_over_5_weeks) →
  cows = 24 :=
by
  intros h1 h2 h3 h4
  sorry

end cows_on_farm_l18_18932


namespace correct_probability_l18_18265

noncomputable def T : ℕ := 44
noncomputable def num_books : ℕ := T - 35
noncomputable def n : ℕ := 9
noncomputable def favorable_outcomes : ℕ := (Nat.choose n 6) * 2
noncomputable def total_arrangements : ℕ := (Nat.factorial n)
noncomputable def probability : Rat := (favorable_outcomes : ℚ) / (total_arrangements : ℚ)
noncomputable def m : ℕ := 1
noncomputable def p : Nat := Nat.gcd 168 362880
noncomputable def final_prob_form : Rat := 1 / 2160
noncomputable def answer : ℕ := m + 2160

theorem correct_probability : 
  probability = final_prob_form ∧ answer = 2161 := 
by
  sorry

end correct_probability_l18_18265


namespace find_y_in_terms_of_x_l18_18614

theorem find_y_in_terms_of_x (x y : ℝ) (h : x - 2 = 4 * (y - 1) + 3) : 
  y = (1 / 4) * x - (1 / 4) := 
by
  sorry

end find_y_in_terms_of_x_l18_18614


namespace percentage_increase_l18_18175

theorem percentage_increase (employees_dec : ℝ) (employees_jan : ℝ) (inc : ℝ) (percentage : ℝ) :
  employees_dec = 470 →
  employees_jan = 408.7 →
  inc = employees_dec - employees_jan →
  percentage = (inc / employees_jan) * 100 →
  percentage = 15 := 
sorry

end percentage_increase_l18_18175


namespace committee_probability_l18_18770

theorem committee_probability :
  let total_committees := Nat.choose 30 6
  let boys_choose := Nat.choose 18 3
  let girls_choose := Nat.choose 12 3
  let specific_committees := boys_choose * girls_choose
  specific_committees / total_committees = 64 / 211 := 
by
  let total_committees := Nat.choose 30 6
  let boys_choose := Nat.choose 18 3
  let girls_choose := Nat.choose 12 3
  let specific_committees := boys_choose * girls_choose
  have h_total_committees : total_committees = 593775 := by sorry
  have h_boys_choose : boys_choose = 816 := by sorry
  have h_girls_choose : girls_choose = 220 := by sorry
  have h_specific_committees : specific_committees = 179520 := by sorry
  have h_probability : specific_committees / total_committees = 64 / 211 := by sorry
  exact h_probability

end committee_probability_l18_18770


namespace initial_distance_l18_18717

-- Definitions based on conditions
def speed_thief : ℝ := 8 -- in km/hr
def speed_policeman : ℝ := 10 -- in km/hr
def distance_thief_runs : ℝ := 0.7 -- in km

-- Theorem statement
theorem initial_distance
  (relative_speed := speed_policeman - speed_thief) -- Relative speed (in km/hr)
  (time_to_overtake := distance_thief_runs / relative_speed) -- Time for the policeman to overtake the thief (in hours)
  (initial_distance := speed_policeman * time_to_overtake) -- Initial distance (in km)
  : initial_distance = 3.5 :=
by
  sorry

end initial_distance_l18_18717


namespace no_rearrangement_to_positive_and_negative_roots_l18_18484

theorem no_rearrangement_to_positive_and_negative_roots (a b c : ℝ) :
  (∃ x1 x2 : ℝ, x1 < 0 ∧ x2 < 0 ∧ a ≠ 0 ∧ b = -a * (x1 + x2) ∧ c = a * x1 * x2) →
  (∃ y1 y2 : ℝ, y1 > 0 ∧ y2 > 0 ∧ a ≠ 0 ∧ b != 0 ∧ c != 0 ∧ 
    (∃ b' c' : ℝ, b' ≠ b ∧ c' ≠ c ∧ 
      b' = -a * (y1 + y2) ∧ c' = a * y1 * y2)) →
  False := by
  sorry

end no_rearrangement_to_positive_and_negative_roots_l18_18484


namespace total_cleaning_validation_l18_18504

-- Define the cleaning frequencies and their vacations
def Michael_bath_week := 2
def Michael_shower_week := 1
def Michael_vacation_weeks := 3

def Angela_shower_day := 1
def Angela_vacation_weeks := 2

def Lucy_bath_week := 3
def Lucy_shower_week := 2
def Lucy_alter_weeks := 4
def Lucy_alter_shower_day := 1
def Lucy_alter_bath_week := 1

def weeks_year := 52
def days_week := 7

-- Calculate Michael's total cleaning times in a year
def Michael_total := (Michael_bath_week * weeks_year) + (Michael_shower_week * weeks_year)
def Michael_vacation_reduction := Michael_vacation_weeks * (Michael_bath_week + Michael_shower_week)
def Michael_cleaning_times := Michael_total - Michael_vacation_reduction

-- Calculate Angela's total cleaning times in a year
def Angela_total := (Angela_shower_day * days_week * weeks_year)
def Angela_vacation_reduction := Angela_vacation_weeks * (Angela_shower_day * days_week)
def Angela_cleaning_times := Angela_total - Angela_vacation_reduction

-- Calculate Lucy's total cleaning times in a year
def Lucy_baths_total := Lucy_bath_week * weeks_year
def Lucy_showers_total := Lucy_shower_week * weeks_year
def Lucy_alter_showers := Lucy_alter_shower_day * days_week * Lucy_alter_weeks
def Lucy_alter_baths_reduction := (Lucy_bath_week - Lucy_alter_bath_week) * Lucy_alter_weeks
def Lucy_cleaning_times := Lucy_baths_total + Lucy_showers_total + Lucy_alter_showers - Lucy_alter_baths_reduction

-- Calculate the total times they clean themselves in 52 weeks
def total_cleaning_times := Michael_cleaning_times + Angela_cleaning_times + Lucy_cleaning_times

-- The proof statement
theorem total_cleaning_validation : total_cleaning_times = 777 :=
by simp [Michael_cleaning_times, Angela_cleaning_times, Lucy_cleaning_times, total_cleaning_times]; sorry

end total_cleaning_validation_l18_18504


namespace find_T_l18_18404

variable (a b c T : ℕ)

theorem find_T (h1 : a + b + c = 84) (h2 : a - 5 = T) (h3 : b + 9 = T) (h4 : 5 * c = T) : T = 40 :=
sorry

end find_T_l18_18404


namespace solve_logarithmic_inequality_l18_18999

theorem solve_logarithmic_inequality :
  {x : ℝ | 2 * (Real.log x / Real.log 0.5)^2 + 9 * (Real.log x / Real.log 0.5) + 9 ≤ 0} = 
  {x : ℝ | 2 * Real.sqrt 2 ≤ x ∧ x ≤ 8} :=
sorry

end solve_logarithmic_inequality_l18_18999


namespace total_selection_ways_l18_18541

-- Defining the conditions
def groupA_male_students : ℕ := 5
def groupA_female_students : ℕ := 3
def groupB_male_students : ℕ := 6
def groupB_female_students : ℕ := 2

-- Define combinations (choose function)
noncomputable def C (n k : ℕ) : ℕ := Nat.choose n k

-- The required theorem statement
theorem total_selection_ways :
  C groupA_female_students 1 * C groupA_male_students 1 * C groupB_male_students 2 +
  C groupB_female_students 1 * C groupB_male_students 1 * C groupA_male_students 2 = 345 :=
by
  sorry

end total_selection_ways_l18_18541


namespace cherry_tomatoes_weight_l18_18790

def kilogram_to_grams (kg : ℕ) : ℕ := kg * 1000

theorem cherry_tomatoes_weight (kg_tomatoes : ℕ) (extra_tomatoes_g : ℕ) : kg_tomatoes = 2 → extra_tomatoes_g = 560 → kilogram_to_grams kg_tomatoes + extra_tomatoes_g = 2560 :=
by
  intros h1 h2
  rw [h1, h2]
  sorry

end cherry_tomatoes_weight_l18_18790


namespace largest_of_a_b_c_l18_18393

noncomputable def a : ℝ := 1 / 2
noncomputable def b : ℝ := Real.log 3 / Real.log 4
noncomputable def c : ℝ := Real.sin (Real.pi / 8)

theorem largest_of_a_b_c : b = max (max a b) c :=
by
  have ha : a = 1 / 2 := rfl
  have hb : b = Real.log 3 / Real.log 4 := rfl
  have hc : c = Real.sin (Real.pi / 8) := rfl
  sorry

end largest_of_a_b_c_l18_18393


namespace second_fisherman_more_fish_l18_18363

-- Define the given conditions
def days_in_season : ℕ := 213
def rate_first_fisherman : ℕ := 3
def rate_second_fisherman_phase_1 : ℕ := 1
def rate_second_fisherman_phase_2 : ℕ := 2
def rate_second_fisherman_phase_3 : ℕ := 4
def days_phase_1 : ℕ := 30
def days_phase_2 : ℕ := 60
def days_phase_3 : ℕ := days_in_season - (days_phase_1 + days_phase_2)

-- Define the total number of fish caught by each fisherman
def total_fish_first_fisherman : ℕ := rate_first_fisherman * days_in_season
def total_fish_second_fisherman : ℕ := 
  (rate_second_fisherman_phase_1 * days_phase_1) + 
  (rate_second_fisherman_phase_2 * days_phase_2) + 
  (rate_second_fisherman_phase_3 * days_phase_3)

-- Define the theorem statement
theorem second_fisherman_more_fish : 
  total_fish_second_fisherman = total_fish_first_fisherman + 3 := by sorry

end second_fisherman_more_fish_l18_18363


namespace inequality_solution_eq_l18_18236

theorem inequality_solution_eq :
  ∀ y : ℝ, 2 ≤ |y - 5| ∧ |y - 5| ≤ 8 ↔ (-3 ≤ y ∧ y ≤ 3) ∨ (7 ≤ y ∧ y ≤ 13) :=
by
  sorry

end inequality_solution_eq_l18_18236


namespace inequality_transformation_l18_18765

theorem inequality_transformation (m n : ℝ) (h : -m / 2 < -n / 6) : 3 * m > n := by
  sorry

end inequality_transformation_l18_18765


namespace average_speed_of_train_l18_18095

theorem average_speed_of_train (d1 d2 : ℝ) (t1 t2 : ℝ) (h1 : d1 = 125) (h2 : d2 = 270) (h3 : t1 = 2.5) (h4 : t2 = 3) :
  (d1 + d2) / (t1 + t2) = 71.82 :=
by
  sorry

end average_speed_of_train_l18_18095


namespace sum_x_y_eq_two_l18_18728

theorem sum_x_y_eq_two (x y : ℝ) (h : x^2 + y^2 = 8*x - 4*y - 28) : x + y = 2 :=
sorry

end sum_x_y_eq_two_l18_18728


namespace largest_of_four_numbers_l18_18774

theorem largest_of_four_numbers 
  (a b c d : ℝ) 
  (h1 : a + 5 = b^2 - 1) 
  (h2 : a + 5 = c^2 + 3) 
  (h3 : a + 5 = d - 4) 
  : d > max (max a b) c :=
sorry

end largest_of_four_numbers_l18_18774


namespace arithmetic_sequence_a3a6_l18_18011

theorem arithmetic_sequence_a3a6 (a : ℕ → ℤ)
  (h1 : ∀ n, a n = a 0 + n * (a 1 - a 0))
  (h_inc : ∀ n, a n < a (n + 1))
  (h_eq : a 3 * a 4 = 45): 
  a 2 * a 5 = 13 := 
sorry

end arithmetic_sequence_a3a6_l18_18011


namespace arithmetic_sequence_a15_l18_18202

theorem arithmetic_sequence_a15 (a_n S_n : ℕ → ℝ) (a_9 : a_n 9 = 4) (S_15 : S_n 15 = 30) :
  let a_1 := (-12 : ℝ)
  let d := (2 : ℝ)
  a_n 15 = 16 :=
by
  sorry

end arithmetic_sequence_a15_l18_18202


namespace sum_of_repeating_decimals_correct_l18_18085

/-- Convert repeating decimals to fractions -/
def rep_dec_1 : ℚ := 1 / 9
def rep_dec_2 : ℚ := 2 / 9
def rep_dec_3 : ℚ := 1 / 3
def rep_dec_4 : ℚ := 4 / 9
def rep_dec_5 : ℚ := 5 / 9
def rep_dec_6 : ℚ := 2 / 3
def rep_dec_7 : ℚ := 7 / 9
def rep_dec_8 : ℚ := 8 / 9

/-- Define the terms in the sum -/
def term_1 : ℚ := 8 + rep_dec_1
def term_2 : ℚ := 7 + 1 + rep_dec_2
def term_3 : ℚ := 6 + 2 + rep_dec_3
def term_4 : ℚ := 5 + 3 + rep_dec_4
def term_5 : ℚ := 4 + 4 + rep_dec_5
def term_6 : ℚ := 3 + 5 + rep_dec_6
def term_7 : ℚ := 2 + 6 + rep_dec_7
def term_8 : ℚ := 1 + 7 + rep_dec_8

/-- Define the sum of the terms -/
def total_sum : ℚ := term_1 + term_2 + term_3 + term_4 + term_5 + term_6 + term_7 + term_8

/-- Proof problem statement -/
theorem sum_of_repeating_decimals_correct : total_sum = 39.2 := 
sorry

end sum_of_repeating_decimals_correct_l18_18085


namespace sum_of_inverses_gt_one_l18_18309

variable (a1 a2 a3 S : ℝ)

theorem sum_of_inverses_gt_one
  (h1 : a1 > 1)
  (h2 : a2 > 1)
  (h3 : a3 > 1)
  (h_sum : a1 + a2 + a3 = S)
  (ineq1 : a1^2 / (a1 - 1) > S)
  (ineq2 : a2^2 / (a2 - 1) > S)
  (ineq3 : a3^2 / (a3 - 1) > S) :
  1 / (a1 + a2) + 1 / (a2 + a3) + 1 / (a3 + a1) > 1 := by
  sorry

end sum_of_inverses_gt_one_l18_18309


namespace no_valid_pairs_l18_18260

theorem no_valid_pairs (a b : ℕ) (h1 : 0 ≤ a ∧ a ≤ 9) (h2 : 0 ≤ b ∧ b ≤ 9) :
  ¬(1000 * a + 100 * b + 32) % 99 = 0 :=
by
  sorry

end no_valid_pairs_l18_18260


namespace find_a_from_complex_condition_l18_18474

theorem find_a_from_complex_condition (a : ℝ) (x y : ℝ) 
  (h : x = -1 ∧ y = -2 * a)
  (h_line : x - y = 0) : a = 1 / 2 :=
by
  sorry

end find_a_from_complex_condition_l18_18474


namespace pizza_slices_leftover_l18_18227

def slices_per_small_pizza := 4
def slices_per_large_pizza := 8
def small_pizzas_purchased := 3
def large_pizzas_purchased := 2

def george_slices := 3
def bob_slices := george_slices + 1
def susie_slices := bob_slices / 2
def bill_slices := 3
def fred_slices := 3
def mark_slices := 3

def total_slices := small_pizzas_purchased * slices_per_small_pizza + large_pizzas_purchased * slices_per_large_pizza
def total_eaten_slices := george_slices + bob_slices + susie_slices + bill_slices + fred_slices + mark_slices

def slices_leftover := total_slices - total_eaten_slices

theorem pizza_slices_leftover : slices_leftover = 10 := by
  sorry

end pizza_slices_leftover_l18_18227


namespace component_probability_l18_18299

theorem component_probability (p : ℝ) 
  (h : (1 - p)^3 = 0.001) : 
  p = 0.9 :=
sorry

end component_probability_l18_18299


namespace train_length_is_correct_l18_18118

noncomputable def length_of_train (train_speed : ℝ) (time_to_cross : ℝ) (bridge_length : ℝ) : ℝ :=
  let speed_m_s := train_speed * (1000 / 3600)
  let total_distance := speed_m_s * time_to_cross
  total_distance - bridge_length

theorem train_length_is_correct :
  length_of_train 36 24.198064154867613 132 = 109.98064154867613 :=
by
  sorry

end train_length_is_correct_l18_18118


namespace victor_earnings_l18_18333

def hourly_wage := 6 -- dollars per hour
def hours_monday := 5 -- hours
def hours_tuesday := 5 -- hours

theorem victor_earnings : (hourly_wage * (hours_monday + hours_tuesday)) = 60 :=
by
  sorry

end victor_earnings_l18_18333


namespace toms_weekly_income_l18_18567

variable (num_buckets : ℕ) (crabs_per_bucket : ℕ) (price_per_crab : ℕ) (days_per_week : ℕ)

def daily_crabs := num_buckets * crabs_per_bucket
def daily_income := daily_crabs * price_per_crab
def weekly_income := daily_income * days_per_week

theorem toms_weekly_income 
  (h1 : num_buckets = 8)
  (h2 : crabs_per_bucket = 12)
  (h3 : price_per_crab = 5)
  (h4 : days_per_week = 7) :
  weekly_income num_buckets crabs_per_bucket price_per_crab days_per_week = 3360 :=
by
  sorry

end toms_weekly_income_l18_18567


namespace calculate_total_houses_built_l18_18489

theorem calculate_total_houses_built :
  let initial_houses := 1426
  let final_houses := 2000
  let rate_a := 25
  let time_a := 6
  let rate_b := 15
  let time_b := 9
  let rate_c := 30
  let time_c := 4
  let total_houses_built := (rate_a * time_a) + (rate_b * time_b) + (rate_c * time_c)
  total_houses_built = 405 :=
by
  sorry

end calculate_total_houses_built_l18_18489


namespace remainder_div_x_plus_1_l18_18735

noncomputable def polynomial1 : Polynomial ℝ := Polynomial.X ^ 11 - 1

theorem remainder_div_x_plus_1 :
  Polynomial.eval (-1) polynomial1 = -2 := by
  sorry

end remainder_div_x_plus_1_l18_18735


namespace ethanol_relationship_l18_18154

variables (a b c x : ℝ)
def total_capacity := a + b + c = 300
def ethanol_content := x = 0.10 * a + 0.15 * b + 0.20 * c
def ethanol_bounds := 30 ≤ x ∧ x ≤ 60

theorem ethanol_relationship : total_capacity a b c → ethanol_bounds x → ethanol_content a b c x :=
by
  intros h_total h_bounds
  unfold total_capacity at h_total
  unfold ethanol_bounds at h_bounds
  unfold ethanol_content
  sorry

end ethanol_relationship_l18_18154


namespace number_of_terms_in_arithmetic_sequence_l18_18463

theorem number_of_terms_in_arithmetic_sequence :
  ∃ n : ℕ, (∀ k : ℕ, (1 ≤ k ∧ k ≤ n → 6 + (k - 1) * 2 = 202)) ∧ n = 99 :=
by
  sorry

end number_of_terms_in_arithmetic_sequence_l18_18463


namespace rectangle_percentage_excess_l18_18287

variable (L W : ℝ) -- The lengths of the sides of the rectangle
variable (x : ℝ) -- The percentage excess for the first side (what we want to prove)

theorem rectangle_percentage_excess 
    (h1 : W' = W * 0.95)                    -- Condition: second side is taken with 5% deficit
    (h2 : L' = L * (1 + x/100))             -- Condition: first side is taken with x% excess
    (h3 : A = L * W)                        -- Actual area of the rectangle
    (h4 : 1.064 = (L' * W') / A) :           -- Condition: error percentage in the area is 6.4%
    x = 12 :=                                -- Proof that x equals 12
sorry

end rectangle_percentage_excess_l18_18287


namespace stamps_total_l18_18845

def Lizette_stamps : ℕ := 813
def Minerva_stamps : ℕ := Lizette_stamps - 125
def Jermaine_stamps : ℕ := Lizette_stamps + 217

def total_stamps : ℕ := Minerva_stamps + Lizette_stamps + Jermaine_stamps

theorem stamps_total :
  total_stamps = 2531 := by
  sorry

end stamps_total_l18_18845


namespace polygon_sides_l18_18888

theorem polygon_sides (s : ℕ) (h : 180 * (s - 2) = 720) : s = 6 :=
by
  sorry

end polygon_sides_l18_18888


namespace prob_three_cards_in_sequence_l18_18398

theorem prob_three_cards_in_sequence : 
  let total_cards := 52
  let spades_count := 13
  let hearts_count := 13
  let sequence_prob := (spades_count / total_cards) * (hearts_count / (total_cards - 1)) * ((spades_count - 1) / (total_cards - 2))
  sequence_prob = (78 / 5100) :=
by
  sorry

end prob_three_cards_in_sequence_l18_18398


namespace fare_from_midpoint_C_to_B_l18_18263

noncomputable def taxi_fare (d : ℝ) : ℝ :=
  if d <= 5 then 10.8 else 10.8 + 1.2 * (d - 5)

theorem fare_from_midpoint_C_to_B (x : ℝ) (h1 : taxi_fare x = 24)
    (h2 : taxi_fare (x - 0.46) = 24) :
    taxi_fare (x / 2) = 14.4 :=
by
  sorry

end fare_from_midpoint_C_to_B_l18_18263


namespace psychologist_diagnosis_l18_18587

theorem psychologist_diagnosis :
  let initial_patients := 26
  let doubling_factor := 2
  let probability := 1 / 4
  let total_patients := initial_patients * doubling_factor
  let expected_patients_with_ZYX := total_patients * probability
  expected_patients_with_ZYX = 13 := by
  sorry

end psychologist_diagnosis_l18_18587


namespace unique_function_l18_18117

noncomputable def find_function (f : ℕ → ℕ) : Prop :=
  ∀ a b : ℕ, a > 0 → b > 0 → a + b > 2019 → a + f b ∣ a^2 + b * f a

theorem unique_function (r : ℕ) (f : ℕ → ℕ) :
  find_function f → (∀ x : ℕ, f x = r * x) :=
sorry

end unique_function_l18_18117


namespace find_other_parallel_side_l18_18005

variable (a b h : ℝ) (Area : ℝ)

-- Conditions
axiom h_pos : h = 13
axiom a_val : a = 18
axiom area_val : Area = 247
axiom area_formula : Area = (1 / 2) * (a + b) * h

-- Theorem (to be proved by someone else)
theorem find_other_parallel_side (a b h : ℝ) 
  (h_pos : h = 13) 
  (a_val : a = 18) 
  (area_val : Area = 247) 
  (area_formula : Area = (1 / 2) * (a + b) * h) : 
  b = 20 :=
by
  sorry

end find_other_parallel_side_l18_18005
