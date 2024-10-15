import Mathlib

namespace NUMINAMATH_GPT_amount_invested_l1894_189495

variables (P y : ℝ)

-- Conditions
def condition1 : Prop := 800 = P * (2 * y) / 100
def condition2 : Prop := 820 = P * ((1 + y / 100) ^ 2 - 1)

-- The proof we seek
theorem amount_invested (h1 : condition1 P y) (h2 : condition2 P y) : P = 8000 :=
by
  -- Place the proof here
  sorry

end NUMINAMATH_GPT_amount_invested_l1894_189495


namespace NUMINAMATH_GPT_number_of_people_per_taxi_l1894_189418

def num_people_in_each_taxi (x : ℕ) (cars taxis vans total : ℕ) : Prop :=
  (cars = 3 * 4) ∧ (vans = 2 * 5) ∧ (total = 58) ∧ (taxis = 6 * x) ∧ (cars + vans + taxis = total)

theorem number_of_people_per_taxi
  (x cars taxis vans total : ℕ)
  (h1 : cars = 3 * 4)
  (h2 : vans = 2 * 5)
  (h3 : total = 58)
  (h4 : taxis = 6 * x)
  (h5 : cars + vans + taxis = total) :
  x = 6 :=
by
  sorry

end NUMINAMATH_GPT_number_of_people_per_taxi_l1894_189418


namespace NUMINAMATH_GPT_solution1_solution2_l1894_189444

noncomputable def problem1 : Prop :=
  ∃ (a b : ℤ), 
  (∃ (n : ℤ), 3*a - 14 = n ∧ a - 2 = n) ∧ 
  (b - 15 = -27) ∧ 
  a = 4 ∧ 
  b = -12 ∧ 
  (4*a + b = 4)

noncomputable def problem2 : Prop :=
  ∀ (a b : ℤ), 
  (a = 4) ∧ 
  (b = -12) → 
  (4*a + b = 4) → 
  (∃ n, n^2 = 4 ∧ (n = 2 ∨ n = -2))

theorem solution1 : problem1 := by { sorry }
theorem solution2 : problem2 := by { sorry }

end NUMINAMATH_GPT_solution1_solution2_l1894_189444


namespace NUMINAMATH_GPT_max_cities_l1894_189493

def city (X : Type) := X

variable (A B C D E : Prop)

-- Conditions as given in the problem
axiom condition1 : A → B
axiom condition2 : D ∨ E
axiom condition3 : B ↔ ¬C
axiom condition4 : C ↔ D
axiom condition5 : E → (A ∧ D)

-- Proof problem: Given the conditions, prove that the maximum set of cities that can be visited is {C, D}
theorem max_cities (h1 : A → B) (h2 : D ∨ E) (h3 : B ↔ ¬C) (h4 : C ↔ D) (h5 : E → (A ∧ D)) : (C ∧ D) ∧ ¬A ∧ ¬B ∧ ¬E :=
by
  -- The core proof would use the constraints to show C and D, and exclude A, B, E
  sorry

end NUMINAMATH_GPT_max_cities_l1894_189493


namespace NUMINAMATH_GPT_suitable_k_first_third_quadrants_l1894_189496

theorem suitable_k_first_third_quadrants (k : ℝ) : 
  (∀ x : ℝ, (x > 0 → k * x > 0) ∧ (x < 0 → k * x < 0)) → k > 0 :=
by
  sorry

end NUMINAMATH_GPT_suitable_k_first_third_quadrants_l1894_189496


namespace NUMINAMATH_GPT_ratio_amy_jeremy_l1894_189484

variable (Amy Chris Jeremy : ℕ)

theorem ratio_amy_jeremy (h1 : Amy + Jeremy + Chris = 132) (h2 : Jeremy = 66) (h3 : Chris = 2 * Amy) : 
  Amy / Jeremy = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_ratio_amy_jeremy_l1894_189484


namespace NUMINAMATH_GPT_lemon_heads_each_person_l1894_189416

-- Define the constants used in the problem
def totalLemonHeads : Nat := 72
def numberOfFriends : Nat := 6

-- The theorem stating the problem and the correct answer
theorem lemon_heads_each_person :
  totalLemonHeads / numberOfFriends = 12 := 
by
  sorry

end NUMINAMATH_GPT_lemon_heads_each_person_l1894_189416


namespace NUMINAMATH_GPT_problem_solution_l1894_189478

noncomputable def omega : ℂ := sorry -- Choose a suitable representative for ω

variables (a b c d : ℝ) (h₀ : a ≠ -1) (h₁ : b ≠ -1) (h₂ : c ≠ -1) (h₃ : d ≠ -1)
          (hω : ω^3 = 1 ∧ ω ≠ 1)
          (h : (1 / (a + ω) + 1 / (b + ω) + 1 / (c + ω) + 1 / (d + ω)) = 3 / ω)

theorem problem_solution (a b c d : ℝ) (h₀ : a ≠ -1) (h₁ : b ≠ -1) (h₂ : c ≠ -1) (h₃ : d ≠ -1)
  (hω : ω^3 = 1 ∧ ω ≠ 1)
  (h : (1 / (a + ω) + 1 / (b + ω) + 1 / (c + ω) + 1 / (d + ω)) = 3 / ω) :
  (1 / (a + 1) + 1 / (b + 1) + 1 / (c + 1) + 1 / (d + 1)) = 3 :=
sorry

end NUMINAMATH_GPT_problem_solution_l1894_189478


namespace NUMINAMATH_GPT_false_log_exists_x_l1894_189406

theorem false_log_exists_x {x : ℝ} : ¬ ∃ x : ℝ, Real.log x = 0 :=
by sorry

end NUMINAMATH_GPT_false_log_exists_x_l1894_189406


namespace NUMINAMATH_GPT_min_likes_both_l1894_189441

-- Definitions corresponding to the conditions
def total_people : ℕ := 200
def likes_beethoven : ℕ := 160
def likes_chopin : ℕ := 150

-- Problem statement to prove
theorem min_likes_both : ∃ x : ℕ, x = 110 ∧ x = likes_beethoven - (total_people - likes_chopin) := by
  sorry

end NUMINAMATH_GPT_min_likes_both_l1894_189441


namespace NUMINAMATH_GPT_stock_price_return_to_initial_l1894_189486

variable (P₀ : ℝ) -- Initial price
variable (y : ℝ) -- Percentage increase during the fourth week

/-- The main theorem stating the required percentage increase in the fourth week -/
theorem stock_price_return_to_initial
  (h1 : P₀ * 1.30 * 0.75 * 1.20 = 117) -- Condition after three weeks
  (h2 : P₃ = P₀) : -- Price returns to initial
  y = -15 := 
by
  sorry

end NUMINAMATH_GPT_stock_price_return_to_initial_l1894_189486


namespace NUMINAMATH_GPT_total_distance_to_run_l1894_189410

theorem total_distance_to_run
  (track_length : ℕ)
  (initial_laps : ℕ)
  (additional_laps : ℕ)
  (total_laps := initial_laps + additional_laps) :
  track_length = 150 →
  initial_laps = 6 →
  additional_laps = 4 →
  total_laps * track_length = 1500 := by
  sorry

end NUMINAMATH_GPT_total_distance_to_run_l1894_189410


namespace NUMINAMATH_GPT_face_opposite_A_l1894_189445
noncomputable def cube_faces : List String := ["A", "B", "C", "D", "E", "F"]

theorem face_opposite_A (cube_faces : List String) 
  (h1 : cube_faces.length = 6)
  (h2 : "A" ∈ cube_faces) 
  (h3 : "B" ∈ cube_faces)
  (h4 : "C" ∈ cube_faces) 
  (h5 : "D" ∈ cube_faces)
  (h6 : "E" ∈ cube_faces) 
  (h7 : "F" ∈ cube_faces)
  : ("D" ≠ "A") := 
by
  sorry

end NUMINAMATH_GPT_face_opposite_A_l1894_189445


namespace NUMINAMATH_GPT_questionnaire_visitors_l1894_189419

noncomputable def total_visitors :=
  let V := 600
  let E := (3 / 4) * V
  V

theorem questionnaire_visitors:
  ∃ (V : ℕ), V = 600 ∧
  (∀ (E : ℕ), E = (3 / 4) * V ∧ E + 150 = V) :=
by
    use 600
    sorry

end NUMINAMATH_GPT_questionnaire_visitors_l1894_189419


namespace NUMINAMATH_GPT_sqrt_difference_of_cubes_is_integer_l1894_189467

theorem sqrt_difference_of_cubes_is_integer (a b : ℕ) (h1 : a = 105) (h2 : b = 104) :
  (Int.sqrt (a^3 - b^3) = 181) :=
by
  sorry

end NUMINAMATH_GPT_sqrt_difference_of_cubes_is_integer_l1894_189467


namespace NUMINAMATH_GPT_volume_of_tetrahedron_l1894_189413

-- Define the setup of tetrahedron D-ABC
def tetrahedron_volume (V : ℝ) : Prop :=
  ∃ (DA : ℝ) (A B C D : ℝ × ℝ × ℝ), 
  A = (0, 0, 0) ∧ 
  B = (2, 0, 0) ∧ 
  C = (1, Real.sqrt 3, 0) ∧
  D = (1, Real.sqrt 3/3, DA) ∧
  DA = 2 * Real.sqrt 3 ∧
  ∃ tan_dihedral : ℝ, tan_dihedral = 2 ∧
  V = 2

-- The statement to prove the volume is indeed 2 given the conditions.
theorem volume_of_tetrahedron : ∃ V, tetrahedron_volume V :=
by 
  sorry

end NUMINAMATH_GPT_volume_of_tetrahedron_l1894_189413


namespace NUMINAMATH_GPT_scientific_notation_l1894_189425

theorem scientific_notation (h : 0.0000046 = 4.6 * 10^(-6)) : True :=
by 
  sorry

end NUMINAMATH_GPT_scientific_notation_l1894_189425


namespace NUMINAMATH_GPT_quadratic_ineq_solution_set_l1894_189435

theorem quadratic_ineq_solution_set {m : ℝ} :
  (∀ x : ℝ, m * x^2 - (m + 3) * x - 1 < 0) ↔ -9 < m ∧ m < -1 :=
sorry

end NUMINAMATH_GPT_quadratic_ineq_solution_set_l1894_189435


namespace NUMINAMATH_GPT_bus_stop_time_l1894_189401

theorem bus_stop_time (v_no_stop v_with_stop : ℝ) (t_per_hour_minutes : ℝ) (h1 : v_no_stop = 48) (h2 : v_with_stop = 24) : t_per_hour_minutes = 30 := 
sorry

end NUMINAMATH_GPT_bus_stop_time_l1894_189401


namespace NUMINAMATH_GPT_complex_repair_cost_l1894_189471

theorem complex_repair_cost
  (charge_tire : ℕ)
  (cost_part_tire : ℕ)
  (num_tires : ℕ)
  (charge_complex : ℕ)
  (num_complex : ℕ)
  (profit_retail : ℕ)
  (fixed_expenses : ℕ)
  (total_profit : ℕ)
  (profit_tire : ℕ := charge_tire - cost_part_tire)
  (total_profit_tire : ℕ := num_tires * profit_tire)
  (total_revenue_complex : ℕ := num_complex * charge_complex)
  (initial_profit : ℕ :=
    total_profit_tire + profit_retail - fixed_expenses)
  (needed_profit_complex : ℕ := total_profit - initial_profit) :
  needed_profit_complex = 100 / num_complex :=
by
  sorry

end NUMINAMATH_GPT_complex_repair_cost_l1894_189471


namespace NUMINAMATH_GPT_correct_system_equations_l1894_189492

theorem correct_system_equations (x y : ℤ) : 
  (8 * x - y = 3) ∧ (y - 7 * x = 4) ↔ 
    (8 * x - y = 3) ∧ (y - 7 * x = 4) := by
  sorry

end NUMINAMATH_GPT_correct_system_equations_l1894_189492


namespace NUMINAMATH_GPT_max_volume_is_16_l1894_189423

noncomputable def max_volume (width : ℝ) (material : ℝ) : ℝ :=
  let l := (material - 2 * width) / (2 + 2 * width)
  let h := (material - 2 * l) / (2 * width + 2 * l)
  l * width * h

theorem max_volume_is_16 :
  max_volume 2 32 = 16 :=
by
  sorry

end NUMINAMATH_GPT_max_volume_is_16_l1894_189423


namespace NUMINAMATH_GPT_workers_complete_job_together_in_time_l1894_189450

theorem workers_complete_job_together_in_time :
  let work_rate_A := 1 / 10 
  let work_rate_B := 1 / 15
  let work_rate_C := 1 / 20
  let combined_work_rate := work_rate_A + work_rate_B + work_rate_C
  let time := 1 / combined_work_rate
  time = 60 / 13 :=
by
  let work_rate_A := 1 / 10
  let work_rate_B := 1 / 15
  let work_rate_C := 1 / 20
  let combined_work_rate := work_rate_A + work_rate_B + work_rate_C
  let time := 1 / combined_work_rate
  sorry

end NUMINAMATH_GPT_workers_complete_job_together_in_time_l1894_189450


namespace NUMINAMATH_GPT_min_value_of_quartic_function_l1894_189456

theorem min_value_of_quartic_function : 
  ∃ x : ℝ, (0 ≤ x ∧ x ≤ 1) ∧ (∀ y : ℝ, (0 ≤ y ∧ y ≤ 1) → x^4 + (1 - x)^4 ≤ y^4 + (1 - y)^4) ∧ (x^4 + (1 - x)^4 = 1 / 8) :=
by
  sorry

end NUMINAMATH_GPT_min_value_of_quartic_function_l1894_189456


namespace NUMINAMATH_GPT_option_C_not_like_terms_l1894_189479

theorem option_C_not_like_terms :
  ¬ (2 * (m : ℝ) == 2 * (n : ℝ)) :=
by
  sorry

end NUMINAMATH_GPT_option_C_not_like_terms_l1894_189479


namespace NUMINAMATH_GPT_fish_worth_apples_l1894_189468

-- Defining the variables
variables (f l r a : ℝ)

-- Conditions based on the problem
def condition1 : Prop := 5 * f = 3 * l
def condition2 : Prop := l = 6 * r
def condition3 : Prop := 3 * r = 2 * a

-- The statement of the problem
theorem fish_worth_apples (h1 : condition1 f l) (h2 : condition2 l r) (h3 : condition3 r a) : f = 12 / 5 * a :=
by
  sorry

end NUMINAMATH_GPT_fish_worth_apples_l1894_189468


namespace NUMINAMATH_GPT_increasing_condition_l1894_189464

noncomputable def f (x a : ℝ) : ℝ := (Real.exp x) + a * (Real.exp (-x))

theorem increasing_condition (a : ℝ) : (∀ x : ℝ, 0 ≤ (Real.exp (2 * x) - a) / (Real.exp x)) ↔ a ≤ 0 :=
by
  sorry

end NUMINAMATH_GPT_increasing_condition_l1894_189464


namespace NUMINAMATH_GPT_actual_distance_between_towns_l1894_189473

theorem actual_distance_between_towns
  (d_map : ℕ) (scale1 : ℕ) (scale2 : ℕ) (distance1 : ℕ) (distance2 : ℕ) (remaining_distance : ℕ) :
  d_map = 9 →
  scale1 = 10 →
  distance1 = 5 →
  scale2 = 8 →
  remaining_distance = d_map - distance1 →
  d_map = distance1 + remaining_distance →
  (distance1 * scale1 + remaining_distance * scale2 = 82) := by
  intros h1 h2 h3 h4 h5 h6
  sorry

end NUMINAMATH_GPT_actual_distance_between_towns_l1894_189473


namespace NUMINAMATH_GPT_pentagon_square_ratio_l1894_189485

theorem pentagon_square_ratio (p s : ℝ) (h₁ : 5 * p = 20) (h₂ : 4 * s = 20) : p / s = 4 / 5 := 
by 
  sorry

end NUMINAMATH_GPT_pentagon_square_ratio_l1894_189485


namespace NUMINAMATH_GPT_find_m_n_l1894_189405

theorem find_m_n (m n : ℕ) (h_pos_m : m > 0) (h_pos_n : n > 0) (h_sol : (m + Real.sqrt n)^2 - 10 * (m + Real.sqrt n) + 1 = Real.sqrt (m + Real.sqrt n) * (m + Real.sqrt n + 1)) : m + n = 55 :=
sorry

end NUMINAMATH_GPT_find_m_n_l1894_189405


namespace NUMINAMATH_GPT_length_of_woods_l1894_189475

theorem length_of_woods (area width : ℝ) (h_area : area = 24) (h_width : width = 8) : (area / width) = 3 :=
by
  sorry

end NUMINAMATH_GPT_length_of_woods_l1894_189475


namespace NUMINAMATH_GPT_solve_for_x_l1894_189448

def F (x y z : ℝ) : ℝ := x * y^3 + z^2

theorem solve_for_x :
  F x 3 2 = F x 2 5 → x = 21/19 :=
  by
  sorry

end NUMINAMATH_GPT_solve_for_x_l1894_189448


namespace NUMINAMATH_GPT_transformed_line_equation_l1894_189433

theorem transformed_line_equation {A B C x₀ y₀ : ℝ} 
    (h₀ : ¬(A = 0 ∧ B = 0)) 
    (h₁ : A * x₀ + B * y₀ + C = 0) : 
    ∀ {x y : ℝ}, A * x + B * y + C = 0 ↔ A * (x - x₀) + B * (y - y₀) = 0 :=
by
    sorry

end NUMINAMATH_GPT_transformed_line_equation_l1894_189433


namespace NUMINAMATH_GPT_six_times_eightx_plus_tenpi_eq_fourP_l1894_189440

variable {x : ℝ} {π P : ℝ}

theorem six_times_eightx_plus_tenpi_eq_fourP (h : 3 * (4 * x + 5 * π) = P) : 
    6 * (8 * x + 10 * π) = 4 * P :=
sorry

end NUMINAMATH_GPT_six_times_eightx_plus_tenpi_eq_fourP_l1894_189440


namespace NUMINAMATH_GPT_expressions_equal_iff_l1894_189469

variable (a b c : ℝ)

theorem expressions_equal_iff :
  a^2 + b*c = (a - b)*(a - c) ↔ a = 0 ∨ b + c = 0 :=
by
  sorry

end NUMINAMATH_GPT_expressions_equal_iff_l1894_189469


namespace NUMINAMATH_GPT_remainder_of_98_mult_102_div_12_l1894_189462

theorem remainder_of_98_mult_102_div_12 : (98 * 102) % 12 = 0 := by
    sorry

end NUMINAMATH_GPT_remainder_of_98_mult_102_div_12_l1894_189462


namespace NUMINAMATH_GPT_train_crossing_time_l1894_189466

theorem train_crossing_time
  (length_of_train : ℝ)
  (speed_in_kmh : ℝ)
  (speed_in_mps : ℝ)
  (conversion_factor : ℝ)
  (time : ℝ)
  (h1 : length_of_train = 160)
  (h2 : speed_in_kmh = 36)
  (h3 : conversion_factor = 1 / 3.6)
  (h4 : speed_in_mps = speed_in_kmh * conversion_factor)
  (h5 : time = length_of_train / speed_in_mps) : time = 16 :=
by
  sorry

end NUMINAMATH_GPT_train_crossing_time_l1894_189466


namespace NUMINAMATH_GPT_temperature_at_night_is_minus_two_l1894_189461

theorem temperature_at_night_is_minus_two (temperature_noon temperature_afternoon temperature_drop_by_night temperature_night : ℤ) : 
  temperature_noon = 5 → temperature_afternoon = 7 → temperature_drop_by_night = 9 → 
  temperature_night = temperature_afternoon - temperature_drop_by_night → 
  temperature_night = -2 := 
by
  intros h1 h2 h3 h4
  rw [h2, h3] at h4
  exact h4


end NUMINAMATH_GPT_temperature_at_night_is_minus_two_l1894_189461


namespace NUMINAMATH_GPT_family_travel_time_l1894_189430

theorem family_travel_time (D : ℕ) (v1 v2 : ℕ) (d1 d2 : ℕ) (t1 t2 : ℕ) :
  D = 560 → 
  v1 = 35 → 
  v2 = 40 → 
  d1 = D / 2 →
  d2 = D / 2 →
  t1 = d1 / v1 →
  t2 = d2 / v2 → 
  t1 + t2 = 15 :=
by
  sorry

end NUMINAMATH_GPT_family_travel_time_l1894_189430


namespace NUMINAMATH_GPT_cakes_difference_l1894_189457

theorem cakes_difference (cakes_bought cakes_sold : ℕ) (h1 : cakes_bought = 139) (h2 : cakes_sold = 145) : cakes_sold - cakes_bought = 6 :=
by
  sorry

end NUMINAMATH_GPT_cakes_difference_l1894_189457


namespace NUMINAMATH_GPT_total_flowers_tuesday_l1894_189408

def ginger_flower_shop (lilacs_monday roses_monday gardenias_monday tulips_monday orchids_monday: ℕ) := 
  let lilacs_tuesday := lilacs_monday + lilacs_monday * 5 / 100
  let roses_tuesday := roses_monday - roses_monday * 4 / 100
  let tulips_tuesday := tulips_monday - tulips_monday * 7 / 100
  let gardenias_tuesday := gardenias_monday
  let orchids_tuesday := orchids_monday
  lilacs_tuesday + roses_tuesday + tulips_tuesday + gardenias_tuesday + orchids_tuesday

theorem total_flowers_tuesday (lilacs_monday roses_monday gardenias_monday tulips_monday orchids_monday: ℕ) 
  (h1: lilacs_monday = 15)
  (h2: roses_monday = 3 * lilacs_monday)
  (h3: gardenias_monday = lilacs_monday / 2)
  (h4: tulips_monday = 2 * (roses_monday + gardenias_monday))
  (h5: orchids_monday = (roses_monday + gardenias_monday + tulips_monday) / 3):
  ginger_flower_shop lilacs_monday roses_monday gardenias_monday tulips_monday orchids_monday = 214 :=
by
  sorry

end NUMINAMATH_GPT_total_flowers_tuesday_l1894_189408


namespace NUMINAMATH_GPT_length_of_train_l1894_189453

theorem length_of_train (speed : ℝ) (time : ℝ) (h1: speed = 48 * (1000 / 3600) * (1 / 1)) (h2: time = 9) : 
  (speed * time) = 119.97 :=
by
  sorry

end NUMINAMATH_GPT_length_of_train_l1894_189453


namespace NUMINAMATH_GPT_inequality_conditions_l1894_189417

theorem inequality_conditions (x y z : ℝ) (h1 : y - x < 1.5 * abs x) (h2 : z = 2 * (y + x)) : 
  (x ≥ 0 → z < 7 * x) ∧ (x < 0 → z < 0) :=
by
  sorry

end NUMINAMATH_GPT_inequality_conditions_l1894_189417


namespace NUMINAMATH_GPT_tens_digit_of_8_pow_2048_l1894_189494

theorem tens_digit_of_8_pow_2048 : (8^2048 % 100) / 10 = 8 := 
by
  sorry

end NUMINAMATH_GPT_tens_digit_of_8_pow_2048_l1894_189494


namespace NUMINAMATH_GPT_hyperbola_ratio_l1894_189443

theorem hyperbola_ratio (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h_hyperbola : ∀ x y, (x^2 / a^2) - (y^2 / b^2) = 1)
  (h_foci_distance : c^2 = a^2 + b^2)
  (h_midpoint_on_hyperbola : ∀ x y, 
    (x, y) = (-(c / 2), c / 2) → ∃ (k l : ℝ), (k^2 / a^2) - (l^2 / b^2) = 1) :
  c / a = (Real.sqrt 10 + Real.sqrt 2) / 2 := 
sorry

end NUMINAMATH_GPT_hyperbola_ratio_l1894_189443


namespace NUMINAMATH_GPT_exists_third_degree_poly_with_positive_and_negative_roots_l1894_189434

theorem exists_third_degree_poly_with_positive_and_negative_roots :
  ∃ (P : ℝ → ℝ), (∃ x : ℝ, P x = 0 ∧ x > 0) ∧ (∃ y : ℝ, (deriv P) y = 0 ∧ y < 0) :=
sorry

end NUMINAMATH_GPT_exists_third_degree_poly_with_positive_and_negative_roots_l1894_189434


namespace NUMINAMATH_GPT_x_intercept_l1894_189490

theorem x_intercept (x y : ℝ) (h : 4 * x - 3 * y = 24) (hy : y = 0) : (x, y) = (6, 0) :=
by {
  sorry
}

end NUMINAMATH_GPT_x_intercept_l1894_189490


namespace NUMINAMATH_GPT_gift_exchange_equation_l1894_189407

theorem gift_exchange_equation
  (x : ℕ)
  (total_gifts : ℕ)
  (H : total_gifts = 56)
  (H1 : 2 * total_gifts = x * (x - 1)) :
  x * (x - 1) = 56 :=
by
  sorry

end NUMINAMATH_GPT_gift_exchange_equation_l1894_189407


namespace NUMINAMATH_GPT_statement_c_correct_l1894_189426

theorem statement_c_correct (a b c : ℝ) (h : a * c^2 > b * c^2) : a > b :=
by sorry

end NUMINAMATH_GPT_statement_c_correct_l1894_189426


namespace NUMINAMATH_GPT_train_length_l1894_189463

theorem train_length 
  (speed_jogger_kmph : ℕ)
  (initial_distance_m : ℕ)
  (speed_train_kmph : ℕ)
  (pass_time_s : ℕ)
  (h_speed_jogger : speed_jogger_kmph = 9)
  (h_initial_distance : initial_distance_m = 230)
  (h_speed_train : speed_train_kmph = 45)
  (h_pass_time : pass_time_s = 35) : 
  ∃ length_train_m : ℕ, length_train_m = 580 := sorry

end NUMINAMATH_GPT_train_length_l1894_189463


namespace NUMINAMATH_GPT_Meadowood_problem_l1894_189459

theorem Meadowood_problem (s h : ℕ) : ¬(26 * s + 3 * h = 58) :=
sorry

end NUMINAMATH_GPT_Meadowood_problem_l1894_189459


namespace NUMINAMATH_GPT_red_stripe_area_l1894_189458

theorem red_stripe_area (diameter height stripe_width : ℝ) (num_revolutions : ℕ) 
  (diam_pos : 0 < diameter) (height_pos : 0 < height) (width_pos : 0 < stripe_width) (height_eq_80 : height = 80)
  (width_eq_3 : stripe_width = 3) (revolutions_eq_2 : num_revolutions = 2) :
  240 = stripe_width * height := 
by
  sorry

end NUMINAMATH_GPT_red_stripe_area_l1894_189458


namespace NUMINAMATH_GPT_yuna_grandfather_age_l1894_189442

def age_yuna : ℕ := 8
def age_father : ℕ := age_yuna + 20
def age_grandfather : ℕ := age_father + 25

theorem yuna_grandfather_age : age_grandfather = 53 := by
  sorry

end NUMINAMATH_GPT_yuna_grandfather_age_l1894_189442


namespace NUMINAMATH_GPT_mr_william_farm_tax_l1894_189400

noncomputable def total_tax_collected : ℝ := 3840
noncomputable def mr_william_percentage : ℝ := 16.666666666666668 / 100  -- Convert percentage to decimal

theorem mr_william_farm_tax : (total_tax_collected * mr_william_percentage) = 640 := by
  sorry

end NUMINAMATH_GPT_mr_william_farm_tax_l1894_189400


namespace NUMINAMATH_GPT_base16_to_base2_bits_l1894_189476

theorem base16_to_base2_bits :
  ∀ (n : ℕ), n = 16^4 * 7 + 16^3 * 7 + 16^2 * 7 + 16 * 7 + 7 → (2^18 ≤ n ∧ n < 2^19) → 
  ∃ b : ℕ, b = 19 := 
by
  intros n hn hpow
  sorry

end NUMINAMATH_GPT_base16_to_base2_bits_l1894_189476


namespace NUMINAMATH_GPT_white_pieces_total_l1894_189498

theorem white_pieces_total (B W : ℕ) 
  (h_total_pieces : B + W = 300) 
  (h_total_piles : 100 * 3 = B + W) 
  (h_piles_1_white : {n : ℕ | n = 27}) 
  (h_piles_2_3_black : {m : ℕ | m = 42}) 
  (h_piles_3_black_3_white : 15 = 15) :
  W = 158 :=
by
  sorry

end NUMINAMATH_GPT_white_pieces_total_l1894_189498


namespace NUMINAMATH_GPT_monotonic_f_deriv_nonneg_l1894_189449

theorem monotonic_f_deriv_nonneg (k : ℝ) :
  (∀ x : ℝ, (1 / 2) < x → k - 1 / x ≥ 0) ↔ k ≥ 2 :=
by sorry

end NUMINAMATH_GPT_monotonic_f_deriv_nonneg_l1894_189449


namespace NUMINAMATH_GPT_exists_maximum_value_of_f_l1894_189491

-- Define the function f(x, y)
noncomputable def f (x y : ℝ) : ℝ := (3 * x * y + 1) * Real.exp (-(x^2 + y^2))

-- Maximum value proof statement
theorem exists_maximum_value_of_f :
  ∃ (x y : ℝ), f x y = (3 / 2) * Real.exp (-1 / 3) :=
sorry

end NUMINAMATH_GPT_exists_maximum_value_of_f_l1894_189491


namespace NUMINAMATH_GPT_find_line_through_intersection_and_perpendicular_l1894_189499

-- Definitions for the given conditions
def line1 (x y : ℝ) : Prop := 3 * x - 2 * y + 1 = 0
def line2 (x y : ℝ) : Prop := x + 3 * y + 4 = 0
def perpendicular (x y m : ℝ) : Prop := x + 3 * y + 4 = 0 ∧ 3 * x - y + m = 0

theorem find_line_through_intersection_and_perpendicular :
  ∃ m : ℝ, ∃ x y : ℝ, line1 x y ∧ line2 x y ∧ perpendicular x y m → 3 * x - y + 2 = 0 :=
by
  sorry

end NUMINAMATH_GPT_find_line_through_intersection_and_perpendicular_l1894_189499


namespace NUMINAMATH_GPT_initial_toys_count_l1894_189451

theorem initial_toys_count (T : ℕ) (h : 10 * T + 300 = 580) : T = 28 :=
by
  sorry

end NUMINAMATH_GPT_initial_toys_count_l1894_189451


namespace NUMINAMATH_GPT_find_f_13_l1894_189427

noncomputable def f : ℕ → ℕ :=
  sorry

axiom condition1 (x : ℕ) : f (x + f x) = 3 * f x
axiom condition2 : f 1 = 3

theorem find_f_13 : f 13 = 27 :=
  sorry

end NUMINAMATH_GPT_find_f_13_l1894_189427


namespace NUMINAMATH_GPT_marie_tasks_finish_time_l1894_189402

noncomputable def total_time (times : List ℕ) : ℕ :=
  times.foldr (· + ·) 0

theorem marie_tasks_finish_time :
  let task_times := [30, 40, 50, 60]
  let start_time := 8 * 60 -- Start time in minutes (8:00 AM)
  let end_time := start_time + total_time task_times
  end_time = 11 * 60 := -- 11:00 AM in minutes
by
  -- Add a placeholder for the proof
  sorry

end NUMINAMATH_GPT_marie_tasks_finish_time_l1894_189402


namespace NUMINAMATH_GPT_determine_phi_l1894_189480

variable (ω : ℝ) (varphi : ℝ)

noncomputable def f (ω varphi x: ℝ) : ℝ := Real.sin (ω * x + varphi)

theorem determine_phi
  (hω : ω > 0)
  (hvarphi : 0 < varphi ∧ varphi < π)
  (hx1 : f ω varphi (π/4) = Real.sin (ω * (π / 4) + varphi))
  (hx2 : f ω varphi (5 * π / 4) = Real.sin (ω * (5 * π / 4) + varphi))
  (hsym : ∀ x, f ω varphi x = f ω varphi (π - x))
  : varphi = π / 4 :=
sorry

end NUMINAMATH_GPT_determine_phi_l1894_189480


namespace NUMINAMATH_GPT_arithmetic_sum_l1894_189472

def arithmetic_sequence (a d : ℕ) (n : ℕ) : ℕ := a + (n * d)

def sum_first_n_terms (a d : ℕ) (n : ℕ) : ℕ := n * (2 * a + (n - 1) * d) / 2

theorem arithmetic_sum :
  ∀ (a d : ℕ),
  arithmetic_sequence a d 2 + arithmetic_sequence a d 3 + arithmetic_sequence a d 4 = 12 →
  sum_first_n_terms a d 7 = 28 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sum_l1894_189472


namespace NUMINAMATH_GPT_net_profit_is_correct_l1894_189497

-- Define the known quantities
def purchase_price : ℝ := 48
def overhead_percentage : ℝ := 0.20
def markup : ℝ := 45

-- Define the derived quantities based on the conditions
def overhead : ℝ := overhead_percentage * purchase_price
def total_cost : ℝ := purchase_price + overhead
def selling_price : ℝ := total_cost + markup
def net_profit : ℝ := selling_price - total_cost

-- The statement to prove
theorem net_profit_is_correct : net_profit = 45 := by
  sorry

end NUMINAMATH_GPT_net_profit_is_correct_l1894_189497


namespace NUMINAMATH_GPT_non_negative_real_inequality_l1894_189422

theorem non_negative_real_inequality
  {a b c : ℝ} (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) :
  a^4 + b^4 + c^4 - 2 * (a^2 * b^2 + a^2 * c^2 + b^2 * c^2) + a^2 * b * c + b^2 * a * c + c^2 * a * b ≥ 0 :=
by
  sorry

end NUMINAMATH_GPT_non_negative_real_inequality_l1894_189422


namespace NUMINAMATH_GPT_parabola_solution_unique_l1894_189465

theorem parabola_solution_unique (a b c : ℝ) (h1 : a + b + c = 1) (h2 : 4 * a + 2 * b + c = -1) (h3 : 4 * a + b = 1) :
  a = 3 ∧ b = -11 ∧ c = 9 := 
  by sorry

end NUMINAMATH_GPT_parabola_solution_unique_l1894_189465


namespace NUMINAMATH_GPT_circle_area_increase_l1894_189424

theorem circle_area_increase (r : ℝ) :
  let A_initial := Real.pi * r^2
  let A_new := Real.pi * (2*r)^2
  let delta_A := A_new - A_initial
  let percentage_increase := (delta_A / A_initial) * 100
  percentage_increase = 300 := by
  sorry

end NUMINAMATH_GPT_circle_area_increase_l1894_189424


namespace NUMINAMATH_GPT_silverware_probability_l1894_189409

-- Define the contents of the drawer
def forks := 6
def spoons := 6
def knives := 6

-- Total number of pieces of silverware
def total_silverware := forks + spoons + knives

-- Combinations formula for choosing r items out of n
def choose (n r : ℕ) : ℕ := Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r))

-- Total number of ways to choose 3 pieces out of 18
def total_ways := choose total_silverware 3

-- Number of ways to choose 1 fork, 1 spoon, and 1 knife
def specific_ways := forks * spoons * knives

-- Calculated probability
def probability := specific_ways / total_ways

theorem silverware_probability : probability = 9 / 34 := 
  sorry
 
end NUMINAMATH_GPT_silverware_probability_l1894_189409


namespace NUMINAMATH_GPT_remainder_of_x_mod_11_l1894_189428

theorem remainder_of_x_mod_11 {x : ℤ} (h : x % 66 = 14) : x % 11 = 3 :=
sorry

end NUMINAMATH_GPT_remainder_of_x_mod_11_l1894_189428


namespace NUMINAMATH_GPT_ohara_triple_example_l1894_189438

noncomputable def is_ohara_triple (a b x : ℕ) : Prop := 
  (Real.sqrt a + Real.sqrt b = x)

theorem ohara_triple_example : 
  is_ohara_triple 49 16 11 ∧ 11 ≠ 100 / 5 := 
by
  sorry

end NUMINAMATH_GPT_ohara_triple_example_l1894_189438


namespace NUMINAMATH_GPT_total_marbles_l1894_189411

/--
Some marbles in a bag are red and the rest are blue.
If one red marble is removed, then one-seventh of the remaining marbles are red.
If two blue marbles are removed instead of one red, then one-fifth of the remaining marbles are red.
Prove that the total number of marbles in the bag originally is 22.
-/
theorem total_marbles (r b : ℕ) (h1 : (r - 1) / (r + b - 1) = 1 / 7) (h2 : r / (r + b - 2) = 1 / 5) :
  r + b = 22 := by
  sorry

end NUMINAMATH_GPT_total_marbles_l1894_189411


namespace NUMINAMATH_GPT_infinitely_many_n_divide_2n_plus_1_l1894_189403

theorem infinitely_many_n_divide_2n_plus_1 :
    ∃ (S : Set ℕ), (∀ n ∈ S, n > 0 ∧ n ∣ (2 * n + 1)) ∧ Set.Infinite S :=
by
  sorry

end NUMINAMATH_GPT_infinitely_many_n_divide_2n_plus_1_l1894_189403


namespace NUMINAMATH_GPT_probability_red_ball_l1894_189460

-- Let P_red be the probability of drawing a red ball.
-- Let P_white be the probability of drawing a white ball.
-- Let P_black be the probability of drawing a black ball.
-- Let P_red_or_white be the probability of drawing a red or white ball.
-- Let P_red_or_black be the probability of drawing a red or black ball.

variable (P_red P_white P_black : ℝ)
variable (P_red_or_white P_red_or_black : ℝ)

-- Given conditions
axiom P_red_or_white_condition : P_red_or_white = 0.58
axiom P_red_or_black_condition : P_red_or_black = 0.62

-- The total probability must sum to 1.
axiom total_probability_condition : P_red + P_white + P_black = 1

-- Prove that the probability of drawing a red ball is 0.2.
theorem probability_red_ball : P_red = 0.2 :=
by
  -- To be proven
  sorry

end NUMINAMATH_GPT_probability_red_ball_l1894_189460


namespace NUMINAMATH_GPT_inequality_system_solution_l1894_189421

theorem inequality_system_solution (x : ℝ) :
  (x > -6 - 2 * x) ∧ (x ≤ (3 + x) / 4) ↔ (-2 < x ∧ x ≤ 1) := 
by
  sorry

end NUMINAMATH_GPT_inequality_system_solution_l1894_189421


namespace NUMINAMATH_GPT_problem1_l1894_189447

theorem problem1 (f : ℝ → ℝ) (x : ℝ) : 
  (f (x + 1/x) = x^2 + 1/x^2) -> f x = x^2 - 2 := 
sorry

end NUMINAMATH_GPT_problem1_l1894_189447


namespace NUMINAMATH_GPT_Problem1_l1894_189483

theorem Problem1 (x y : ℝ) (h : x^2 + y^2 = 1) : x^6 + 3*x^2*y^2 + y^6 = 1 := 
by
  sorry

end NUMINAMATH_GPT_Problem1_l1894_189483


namespace NUMINAMATH_GPT_inscribed_square_product_l1894_189420

theorem inscribed_square_product (a b : ℝ)
  (h1 : a + b = 2 * Real.sqrt 5)
  (h2 : Real.sqrt (a^2 + b^2) = 4 * Real.sqrt 2) :
  a * b = -6 := 
by
  sorry

end NUMINAMATH_GPT_inscribed_square_product_l1894_189420


namespace NUMINAMATH_GPT_prairie_total_area_l1894_189452

theorem prairie_total_area :
  let dust_covered := 64535
  let untouched := 522
  (dust_covered + untouched) = 65057 :=
by {
  let dust_covered := 64535
  let untouched := 522
  trivial
}

end NUMINAMATH_GPT_prairie_total_area_l1894_189452


namespace NUMINAMATH_GPT_quadratic_equation_real_roots_k_value_l1894_189431

theorem quadratic_equation_real_roots_k_value :
  (∀ k : ℕ, (∃ x : ℝ, k * x^2 - 3 * x + 2 = 0) <-> k = 1) :=
by
  sorry
  
end NUMINAMATH_GPT_quadratic_equation_real_roots_k_value_l1894_189431


namespace NUMINAMATH_GPT_proof_problem_l1894_189412

-- Define the propositions and conditions
def p : Prop := ∀ x > 0, 3^x > 1
def neg_p : Prop := ∃ x > 0, 3^x ≤ 1
def q (a : ℝ) : Prop := a < -2
def f (a : ℝ) (x : ℝ) : ℝ := a * x + 3

-- The condition that q is a sufficient condition for f(x) to have a zero in [-1,2]
def has_zero_in_interval (a : ℝ) : Prop := 
  (-a + 3) * (2 * a + 3) ≤ 0

-- The proof problem statement
theorem proof_problem (a : ℝ) (P : p) (Q : has_zero_in_interval a) : ¬ p ∧ q a :=
by
  sorry

end NUMINAMATH_GPT_proof_problem_l1894_189412


namespace NUMINAMATH_GPT_average_snowfall_per_hour_l1894_189414

theorem average_snowfall_per_hour (total_snowfall : ℕ) (hours_per_week : ℕ) (total_snowfall_eq : total_snowfall = 210) (hours_per_week_eq : hours_per_week = 7 * 24) : 
  total_snowfall / hours_per_week = 5 / 4 :=
by
  -- skip the proof
  sorry

end NUMINAMATH_GPT_average_snowfall_per_hour_l1894_189414


namespace NUMINAMATH_GPT_problem_1_problem_2_l1894_189439

-- Define the sets A, B, C
def SetA (a : ℝ) : Set ℝ := { x | x^2 - a * x + a^2 - 19 = 0 }
def SetB : Set ℝ := { x | x^2 - 5 * x + 6 = 0 }
def SetC : Set ℝ := { x | x^2 + 2 * x - 8 = 0 }

-- Problem 1
theorem problem_1 (a : ℝ) : SetA a = SetB → a = 5 := by
  sorry

-- Problem 2
theorem problem_2 (a : ℝ) : (SetA a ∩ SetB).Nonempty ∧ (SetA a ∩ SetC = ∅) → a = -2 := by
  sorry

end NUMINAMATH_GPT_problem_1_problem_2_l1894_189439


namespace NUMINAMATH_GPT_rectangle_area_l1894_189415

variables (y : ℝ) (length : ℝ) (width : ℝ)

-- Definitions based on conditions
def is_diagonal_y (length width y : ℝ) : Prop :=
  y^2 = length^2 + width^2

def is_length_three_times_width (length width : ℝ) : Prop :=
  length = 3 * width

-- Statement to prove
theorem rectangle_area (y : ℝ) (length width : ℝ)
  (h1 : is_diagonal_y length width y)
  (h2 : is_length_three_times_width length width) :
  length * width = 3 * (y^2 / 10) :=
sorry

end NUMINAMATH_GPT_rectangle_area_l1894_189415


namespace NUMINAMATH_GPT_train_length_is_correct_l1894_189487

noncomputable def train_length (speed_kmph : ℝ) (time_sec : ℝ) (bridge_length : ℝ) : ℝ :=
  let speed_mps := speed_kmph * 1000 / 3600
  let total_distance := speed_mps * time_sec
  total_distance - bridge_length

theorem train_length_is_correct :
  train_length 60 20.99832013438925 240 = 110 :=
by
  sorry

end NUMINAMATH_GPT_train_length_is_correct_l1894_189487


namespace NUMINAMATH_GPT_probability_of_first_hearts_and_second_clubs_l1894_189474

noncomputable def probability_first_hearts_second_clubs : ℚ :=
  let total_cards := 52
  let hearts_count := 13
  let clubs_count := 13
  let probability_first_hearts := hearts_count / total_cards
  let probability_second_clubs_given_first_hearts := clubs_count / (total_cards - 1)
  probability_first_hearts * probability_second_clubs_given_first_hearts

theorem probability_of_first_hearts_and_second_clubs :
  probability_first_hearts_second_clubs = 13 / 204 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_first_hearts_and_second_clubs_l1894_189474


namespace NUMINAMATH_GPT_breadth_of_garden_l1894_189477

theorem breadth_of_garden (P L B : ℝ) (hP : P = 1800) (hL : L = 500) : B = 400 :=
by
  sorry

end NUMINAMATH_GPT_breadth_of_garden_l1894_189477


namespace NUMINAMATH_GPT_reciprocals_of_roots_l1894_189437

variable (a b c k : ℝ)

theorem reciprocals_of_roots (kr ks : ℝ) (h_eq : a * kr^2 + k * c * kr + b = 0) (h_eq2 : a * ks^2 + k * c * ks + b = 0) :
  (1 / (kr^2)) + (1 / (ks^2)) = (k^2 * c^2 - 2 * a * b) / (b^2) :=
by
  sorry

end NUMINAMATH_GPT_reciprocals_of_roots_l1894_189437


namespace NUMINAMATH_GPT_find_original_strength_l1894_189429

variable (original_strength : ℕ)
variable (total_students : ℕ := original_strength + 12)
variable (original_avg_age : ℕ := 40)
variable (new_students : ℕ := 12)
variable (new_students_avg_age : ℕ := 32)
variable (new_avg_age_reduction : ℕ := 4)
variable (new_avg_age : ℕ := original_avg_age - new_avg_age_reduction)

theorem find_original_strength (h : (original_avg_age * original_strength + new_students * new_students_avg_age) / total_students = new_avg_age) :
  original_strength = 12 := 
sorry

end NUMINAMATH_GPT_find_original_strength_l1894_189429


namespace NUMINAMATH_GPT_least_value_of_d_l1894_189446

theorem least_value_of_d (c d : ℕ) (hc_pos : 0 < c) (hd_pos : 0 < d)
  (hc_factors : (∃ a b : ℕ, a > 1 ∧ b > 1 ∧ a ≠ b ∧ c = a * b) ∨ (∃ p : ℕ, p > 1 ∧ c = p^3))
  (hd_factors : ∃ factors : ℕ, factors = c ∧ ∃ divisors : Finset ℕ, divisors.card = factors ∧ ∀ k ∈ divisors, d % k = 0)
  (div_cd : d % c = 0) : d = 18 :=
sorry

end NUMINAMATH_GPT_least_value_of_d_l1894_189446


namespace NUMINAMATH_GPT_complementary_events_A_B_l1894_189489

def is_odd (n : ℕ) : Prop := n % 2 = 1
def is_even (n : ℕ) : Prop := n % 2 = 0
def is_multiple_of_3 (n : ℕ) : Prop := n % 3 = 0

def A (n : ℕ) : Prop := is_odd n
def B (n : ℕ) : Prop := is_even n
def C (n : ℕ) : Prop := is_multiple_of_3 n

theorem complementary_events_A_B :
  (∀ n, A n → ¬ B n) ∧ (∀ n, B n → ¬ A n) ∧ (∀ n, A n ∨ B n) :=
  sorry

end NUMINAMATH_GPT_complementary_events_A_B_l1894_189489


namespace NUMINAMATH_GPT_largest_multiple_of_7_less_than_neg_100_l1894_189455

theorem largest_multiple_of_7_less_than_neg_100 : 
  ∃ (x : ℤ), (∃ n : ℤ, x = 7 * n) ∧ x < -100 ∧ ∀ y : ℤ, (∃ m : ℤ, y = 7 * m) ∧ y < -100 → y ≤ x :=
by
  sorry

end NUMINAMATH_GPT_largest_multiple_of_7_less_than_neg_100_l1894_189455


namespace NUMINAMATH_GPT_optionA_is_square_difference_l1894_189454

theorem optionA_is_square_difference (x y : ℝ) : 
  (-x + y) * (x + y) = -(x + y) * (x - y) :=
by sorry

end NUMINAMATH_GPT_optionA_is_square_difference_l1894_189454


namespace NUMINAMATH_GPT_wheel_rpm_is_approximately_5000_23_l1894_189436

noncomputable def bus_wheel_rpm (radius : ℝ) (speed : ℝ) : ℝ :=
  let circumference := 2 * Real.pi * radius
  let speed_cm_per_min := (speed * 1000 * 100) / 60
  speed_cm_per_min / circumference

-- Conditions
def radius := 35
def speed := 66

-- Question (to be proved)
theorem wheel_rpm_is_approximately_5000_23 : 
  abs (bus_wheel_rpm radius speed - 5000.23) < 0.01 :=
by
  sorry

end NUMINAMATH_GPT_wheel_rpm_is_approximately_5000_23_l1894_189436


namespace NUMINAMATH_GPT_area_evaluation_l1894_189482

noncomputable def radius : ℝ := 6
noncomputable def central_angle : ℝ := 90
noncomputable def p := 18
noncomputable def q := 3
noncomputable def r : ℝ := -27 / 2

theorem area_evaluation :
  p + q + r = 7.5 :=
by
  sorry

end NUMINAMATH_GPT_area_evaluation_l1894_189482


namespace NUMINAMATH_GPT_same_terminal_side_l1894_189481

theorem same_terminal_side (k : ℤ) : ∃ k : ℤ, (2 * k * Real.pi - Real.pi / 6) = 11 * Real.pi / 6 := by
  sorry

end NUMINAMATH_GPT_same_terminal_side_l1894_189481


namespace NUMINAMATH_GPT_oxen_count_b_l1894_189432

theorem oxen_count_b 
  (a_oxen : ℕ) (a_months : ℕ)
  (b_months : ℕ) (x : ℕ)
  (c_oxen : ℕ) (c_months : ℕ)
  (total_rent : ℝ) (c_rent : ℝ)
  (h1 : a_oxen * a_months = 70)
  (h2 : c_oxen * c_months = 45)
  (h3 : c_rent / total_rent = 27 / 105)
  (h4 : total_rent = 105) :
  x = 12 :=
by 
  sorry

end NUMINAMATH_GPT_oxen_count_b_l1894_189432


namespace NUMINAMATH_GPT_problem_solution_l1894_189404

theorem problem_solution (a b : ℝ) (h1 : b > a) (h2 : a > 0) :
  a^2 < b^2 ∧ ab < b^2 :=
sorry

end NUMINAMATH_GPT_problem_solution_l1894_189404


namespace NUMINAMATH_GPT_even_function_has_zero_coefficient_l1894_189470

theorem even_function_has_zero_coefficient (a : ℝ) :
  (∀ x : ℝ, (x^2 + a*x) = (x^2 + a*(-x))) → a = 0 :=
by
  intro h
  -- the proof part is omitted as requested
  sorry

end NUMINAMATH_GPT_even_function_has_zero_coefficient_l1894_189470


namespace NUMINAMATH_GPT_inequality_abc_l1894_189488

theorem inequality_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 + 2) * (b^2 + 2) * (c^2 + 2) >= 9 * (a * b + b * c + c * a) :=
by
  sorry

end NUMINAMATH_GPT_inequality_abc_l1894_189488
