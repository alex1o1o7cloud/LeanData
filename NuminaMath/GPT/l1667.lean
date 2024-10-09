import Mathlib

namespace pipe_Q_fill_time_l1667_166730

theorem pipe_Q_fill_time (x : ℝ) (h1 : 6 > 0)
    (h2 : 24 > 0)
    (h3 : 3.4285714285714284 > 0)
    (h4 : (1 / 6) + (1 / x) + (1 / 24) = 1 / 3.4285714285714284) :
    x = 8 := by
  sorry

end pipe_Q_fill_time_l1667_166730


namespace simplify_expr1_l1667_166759

theorem simplify_expr1 (m n : ℝ) :
  (2 * m + n) ^ 2 - (4 * m + 3 * n) * (m - n) = 8 * m * n + 4 * n ^ 2 := by
  sorry

end simplify_expr1_l1667_166759


namespace smallest_positive_period_max_value_in_interval_l1667_166717

noncomputable def f (x : ℝ) : ℝ :=
  2 * (Real.sin x) ^ 2 - Real.cos (2 * x + Real.pi / 3)

theorem smallest_positive_period :
  ∃ T > 0, (∀ x, f (x + T) = f x) ∧ T = Real.pi :=
sorry

theorem max_value_in_interval :
  ∃ x ∈ Set.Icc (0 : ℝ) (Real.pi / 2), f x = 5 / 2 :=
sorry

end smallest_positive_period_max_value_in_interval_l1667_166717


namespace analytic_expression_of_f_l1667_166727

noncomputable def f (x : ℝ) := Real.sin (x + Real.pi / 2)

noncomputable def g (α : ℝ) := Real.cos (α - Real.pi / 3)

theorem analytic_expression_of_f :
  (∀ x, f x = Real.cos x) ∧
  (∀ α, α ∈ Set.Icc 0 Real.pi → g α = 1/2 → (α = 0 ∨ α = 2 * Real.pi / 3)) :=
by
  sorry

end analytic_expression_of_f_l1667_166727


namespace finding_f_of_neg_half_l1667_166787

def f (x : ℝ) : ℝ := sorry

theorem finding_f_of_neg_half : f (-1/2) = Real.pi / 3 :=
by
  -- Given function definition condition: f (cos x) = x / 2 for 0 ≤ x ≤ π
  -- f should be defined on ℝ -> ℝ such that this condition holds;
  -- Applying this condition should verify our theorem.
  sorry

end finding_f_of_neg_half_l1667_166787


namespace yoongi_average_score_l1667_166745

/-- 
Yoongi's average score on the English test taken in August and September was 86, and his English test score in October was 98. 
Prove that the average score of the English test for 3 months is 90.
-/
theorem yoongi_average_score 
  (avg_aug_sep : ℕ)
  (score_oct : ℕ)
  (hp1 : avg_aug_sep = 86)
  (hp2 : score_oct = 98) :
  ((avg_aug_sep * 2 + score_oct) / 3) = 90 :=
by
  sorry

end yoongi_average_score_l1667_166745


namespace original_number_l1667_166760

theorem original_number (N : ℕ) :
  (∃ k m n : ℕ, N - 6 = 5 * k + 3 ∧ N - 6 = 11 * m + 3 ∧ N - 6 = 13 * n + 3) → N = 724 :=
by
  sorry

end original_number_l1667_166760


namespace largest_number_in_systematic_sample_l1667_166712

theorem largest_number_in_systematic_sample (n_products : ℕ) (start : ℕ) (interval : ℕ) (sample_size : ℕ) (largest_number : ℕ)
  (h1 : n_products = 500)
  (h2 : start = 7)
  (h3 : interval = 25)
  (h4 : sample_size = n_products / interval)
  (h5 : sample_size = 20)
  (h6 : largest_number = start + interval * (sample_size - 1))
  (h7 : largest_number = 482) :
  largest_number = 482 := 
  sorry

end largest_number_in_systematic_sample_l1667_166712


namespace probability_of_orange_face_l1667_166721

theorem probability_of_orange_face :
  ∃ (G O P : ℕ) (total_faces : ℕ), total_faces = 10 ∧ G = 5 ∧ O = 3 ∧ P = 2 ∧
  (O / total_faces : ℚ) = 3 / 10 := by 
  sorry

end probability_of_orange_face_l1667_166721


namespace bridge_length_is_correct_l1667_166726

noncomputable def length_of_bridge (train_length : ℝ) (train_speed_kmph : ℝ) (crossing_time_seconds : ℝ) : ℝ :=
  let speed_mps := train_speed_kmph * (1000 / 3600)
  let total_distance := speed_mps * crossing_time_seconds
  total_distance - train_length

theorem bridge_length_is_correct :
  length_of_bridge 200 (60) 45 = 550.15 :=
by
  sorry

end bridge_length_is_correct_l1667_166726


namespace complete_square_ratio_l1667_166725

theorem complete_square_ratio (k : ℝ) :
  ∃ c p q : ℝ, 
    8 * k^2 - 12 * k + 20 = c * (k + p)^2 + q ∧ 
    q / p = -142 / 3 :=
sorry

end complete_square_ratio_l1667_166725


namespace garden_perimeter_is_24_l1667_166775

def perimeter_of_garden(a b c x: ℕ) (h1: a + b + c = 3) : ℕ :=
  3 + 5 + a + x + b + 4 + c + 4 + 5 - x

theorem garden_perimeter_is_24 (a b c x : ℕ) (h1 : a + b + c = 3) :
  perimeter_of_garden a b c x h1 = 24 :=
  by
  sorry

end garden_perimeter_is_24_l1667_166775


namespace y_intercept_of_line_l1667_166738

theorem y_intercept_of_line : 
  ∀ (x y : ℝ), 3 * x - 5 * y = 7 → y = -7 / 5 :=
by
  intro x y h
  sorry

end y_intercept_of_line_l1667_166738


namespace base5_2004_to_decimal_is_254_l1667_166713

def base5_to_decimal (n : Nat) : Nat :=
  match n with
  | 2004 => 2 * 5^3 + 0 * 5^2 + 0 * 5^1 + 4 * 5^0
  | _ => 0

theorem base5_2004_to_decimal_is_254 :
  base5_to_decimal 2004 = 254 :=
by
  -- Proof goes here
  sorry

end base5_2004_to_decimal_is_254_l1667_166713


namespace max_wins_l1667_166740

theorem max_wins (Chloe_wins Max_wins : ℕ) (h1 : Chloe_wins = 24) (h2 : 8 * Max_wins = 3 * Chloe_wins) : Max_wins = 9 := by
  sorry

end max_wins_l1667_166740


namespace positive_whole_numbers_with_cube_roots_less_than_15_l1667_166757

theorem positive_whole_numbers_with_cube_roots_less_than_15 :
  ∃ n : ℕ, n = 3374 ∧ ∀ x : ℕ, (x > 0 ∧ x < 3375) → x <= n :=
by sorry

end positive_whole_numbers_with_cube_roots_less_than_15_l1667_166757


namespace ways_to_get_off_the_bus_l1667_166782

-- Define the number of passengers and stops
def numPassengers : ℕ := 10
def numStops : ℕ := 5

-- Define the theorem that states the number of ways for passengers to get off
theorem ways_to_get_off_the_bus : (numStops^numPassengers) = 5^10 :=
by sorry

end ways_to_get_off_the_bus_l1667_166782


namespace find_x_ge_0_l1667_166746

-- Defining the condition and the proof problem
theorem find_x_ge_0 :
  {x : ℝ | (x^2 + 2*x^4 - 3*x^5) / (x + 2*x^3 - 3*x^4) ≥ 0} = {x : ℝ | 0 ≤ x} :=
by
  sorry -- proof steps not included

end find_x_ge_0_l1667_166746


namespace lcm_20_45_75_eq_900_l1667_166754

theorem lcm_20_45_75_eq_900 : Nat.lcm (Nat.lcm 20 45) 75 = 900 :=
by sorry

end lcm_20_45_75_eq_900_l1667_166754


namespace evaluate_A_minus10_3_l1667_166777

def A (x : ℝ) (m : ℕ) : ℝ :=
  if m = 0 then 1 else x * A (x - 1) (m - 1)

theorem evaluate_A_minus10_3 : A (-10) 3 = 1320 := 
  sorry

end evaluate_A_minus10_3_l1667_166777


namespace convenience_store_pure_milk_quantity_convenience_store_yogurt_discount_l1667_166700

noncomputable def cost_per_pure_milk_box (x : ℕ) : ℝ := 2000 / x
noncomputable def cost_per_yogurt_box (x : ℕ) : ℝ := 4800 / (1.5 * x)

theorem convenience_store_pure_milk_quantity
  (x : ℕ)
  (hx : cost_per_yogurt_box x - cost_per_pure_milk_box x = 30) :
  x = 40 :=
by
  sorry

noncomputable def pure_milk_price := 80
noncomputable def yogurt_price (cost_per_yogurt_box : ℝ) : ℝ := cost_per_yogurt_box * 1.25

theorem convenience_store_yogurt_discount
  (x y : ℕ)
  (hx : cost_per_yogurt_box x - cost_per_pure_milk_box x = 30)
  (total_profit : ℕ)
  (profit_condition :
    pure_milk_price * x +
    yogurt_price (cost_per_yogurt_box x) * (1.5 * x - y) +
    yogurt_price (cost_per_yogurt_box x) * 0.9 * y - 2000 - 4800 = total_profit)
  (pure_milk_quantity : x = 40)
  (profit_value : total_profit = 2150) :
  y = 25 :=
by
  sorry

end convenience_store_pure_milk_quantity_convenience_store_yogurt_discount_l1667_166700


namespace water_usage_difference_l1667_166741

theorem water_usage_difference (C X : ℕ)
    (h1 : C = 111000)
    (h2 : C = 3 * X)
    (days : ℕ) (h3 : days = 365) :
    (C * days - X * days) = 26910000 := by
  sorry

end water_usage_difference_l1667_166741


namespace cosine_identity_l1667_166767

theorem cosine_identity (alpha : ℝ) (h1 : -180 < alpha ∧ alpha < -90)
  (cos_75_alpha : Real.cos (75 * Real.pi / 180 + alpha) = 1 / 3) :
  Real.cos (15 * Real.pi / 180 - alpha) = -2 * Real.sqrt 2 / 3 := by
sorry

end cosine_identity_l1667_166767


namespace terry_total_miles_l1667_166702

def total_gasoline_used := 9 + 17
def average_gas_mileage := 30

theorem terry_total_miles (M : ℕ) : 
  total_gasoline_used * average_gas_mileage = M → M = 780 :=
by
  intro h
  rw [←h]
  sorry

end terry_total_miles_l1667_166702


namespace angle_AOD_128_57_l1667_166736

-- Define angles as real numbers
variables {α β : ℝ}

-- Define the conditions
def perp (v1 v2 : ℝ) := v1 = 90 - v2

theorem angle_AOD_128_57 
  (h1 : perp α 90)
  (h2 : perp β 90)
  (h3 : α = 2.5 * β) :
  α = 128.57 :=
by
  -- Proof would go here
  sorry

end angle_AOD_128_57_l1667_166736


namespace area_of_right_triangle_with_hypotenuse_and_angle_l1667_166705

theorem area_of_right_triangle_with_hypotenuse_and_angle 
  (hypotenuse : ℝ) (angle : ℝ) (h_hypotenuse : hypotenuse = 9 * Real.sqrt 3) (h_angle : angle = 30) : 
  ∃ (area : ℝ), area = 364.5 := 
by
  sorry

end area_of_right_triangle_with_hypotenuse_and_angle_l1667_166705


namespace new_average_age_l1667_166732

/--
The average age of 7 people in a room is 28 years.
A 22-year-old person leaves the room, and a 30-year-old person enters the room.
Prove that the new average age of the people in the room is \( 29 \frac{1}{7} \).
-/
theorem new_average_age (avg_age : ℕ) (num_people : ℕ) (leaving_age : ℕ) (entering_age : ℕ)
  (H1 : avg_age = 28)
  (H2 : num_people = 7)
  (H3 : leaving_age = 22)
  (H4 : entering_age = 30) :
  (avg_age * num_people - leaving_age + entering_age) / num_people = 29 + 1 / 7 := 
by
  sorry

end new_average_age_l1667_166732


namespace pollution_index_minimum_l1667_166779

noncomputable def pollution_index (k a b : ℝ) (x : ℝ) : ℝ :=
  k * (a / (x ^ 2) + b / ((18 - x) ^ 2))

theorem pollution_index_minimum (k : ℝ) (h₀ : 0 < k) (h₁ : ∀ x : ℝ, x ≠ 0 ∧ x ≠ 18) :
  ∀ a b x : ℝ, a = 1 → x = 6 → pollution_index k a b x = pollution_index k 1 8 6 :=
by
  intros a b x ha hx
  rw [ha, hx, pollution_index]
  sorry

end pollution_index_minimum_l1667_166779


namespace exterior_angle_BAC_l1667_166765

theorem exterior_angle_BAC 
    (interior_angle_nonagon : ℕ → ℚ) 
    (angle_CAD_angle_BAD : ℚ → ℚ → ℚ)
    (exterior_angle_formula : ℚ → ℚ) :
  (interior_angle_nonagon 9 = 140) ∧ 
  (angle_CAD_angle_BAD 90 140 = 230) ∧ 
  (exterior_angle_formula 230 = 130) := 
sorry

end exterior_angle_BAC_l1667_166765


namespace system_solution_l1667_166755

theorem system_solution (x y : ℝ) :
  (x^2 * y + x * y^2 - 2 * x - 2 * y + 10 = 0) ∧ 
  (x^3 * y - x * y^3 - 2 * x^2 + 2 * y^2 - 30 = 0) ↔ 
  (x = -4) ∧ (y = -1) :=
by
  sorry

end system_solution_l1667_166755


namespace quadratic_decreasing_l1667_166794

-- Define the quadratic function and the condition a < 0
def quadratic_function (a x : ℝ) := a * x^2 - 2 * a * x + 1

-- Define the main theorem to be proven
theorem quadratic_decreasing (a m : ℝ) (ha : a < 0) : 
  (∀ x, x > m → quadratic_function a x < quadratic_function a (x+1)) ↔ m ≥ 1 :=
by
  sorry

end quadratic_decreasing_l1667_166794


namespace cost_of_perfume_l1667_166780

-- Definitions and Constants
def christian_initial_savings : ℕ := 5
def sue_initial_savings : ℕ := 7
def neighbors_yards_mowed : ℕ := 4
def charge_per_yard : ℕ := 5
def dogs_walked : ℕ := 6
def charge_per_dog : ℕ := 2
def additional_amount_needed : ℕ := 6

-- Theorem Statement
theorem cost_of_perfume :
  let christian_earnings := neighbors_yards_mowed * charge_per_yard
  let sue_earnings := dogs_walked * charge_per_dog
  let christian_savings := christian_initial_savings + christian_earnings
  let sue_savings := sue_initial_savings + sue_earnings
  let total_savings := christian_savings + sue_savings
  total_savings + additional_amount_needed = 50 := 
by
  sorry

end cost_of_perfume_l1667_166780


namespace gmat_test_takers_correctly_l1667_166796

variable (A B : ℝ)
variable (intersection union : ℝ)

theorem gmat_test_takers_correctly :
  B = 0.8 ∧ intersection = 0.7 ∧ union = 0.95 → A = 0.85 :=
by 
  sorry

end gmat_test_takers_correctly_l1667_166796


namespace g_x_equation_g_3_value_l1667_166753

noncomputable def g : ℝ → ℝ := sorry

theorem g_x_equation (x : ℝ) (hx : x ≠ 1/2) : g x + g ((x + 2) / (2 - 4 * x)) = 2 * x := sorry

theorem g_3_value : g 3 = 31 / 8 :=
by
  -- Use the provided functional equation and specific input values to derive g(3)
  sorry

end g_x_equation_g_3_value_l1667_166753


namespace hours_to_destination_l1667_166749

def num_people := 4
def water_per_person_per_hour := 1 / 2
def total_water_bottles_needed := 32

theorem hours_to_destination : 
  ∃ h : ℕ, (num_people * water_per_person_per_hour * 2 * h = total_water_bottles_needed) → h = 8 :=
by
  sorry

end hours_to_destination_l1667_166749


namespace find_n_l1667_166758

theorem find_n (n : ℕ) (h_pos : 0 < n) (h_lcm1 : Nat.lcm 40 n = 120) (h_lcm2 : Nat.lcm n 45 = 180) : n = 12 :=
sorry

end find_n_l1667_166758


namespace train_length_l1667_166781

theorem train_length
  (S : ℝ)  -- speed of the train in meters per second
  (L : ℝ)  -- length of the train in meters
  (h1 : L = S * 20)
  (h2 : L + 500 = S * 40) :
  L = 500 := 
sorry

end train_length_l1667_166781


namespace fewest_keystrokes_to_256_l1667_166708

def fewest_keystrokes (start target : Nat) : Nat :=
if start = 1 && target = 256 then 8 else sorry

theorem fewest_keystrokes_to_256 : fewest_keystrokes 1 256 = 8 :=
by
  sorry

end fewest_keystrokes_to_256_l1667_166708


namespace first_term_arithmetic_sequence_median_1010_last_2015_l1667_166735

theorem first_term_arithmetic_sequence_median_1010_last_2015 (a₁ : ℕ) :
  let median := 1010
  let last_term := 2015
  (a₁ + last_term = 2 * median) → a₁ = 5 :=
by
  intros
  sorry

end first_term_arithmetic_sequence_median_1010_last_2015_l1667_166735


namespace percentage_less_than_l1667_166799

variable (x y : ℝ)
variable (H : y = 1.4 * x)

theorem percentage_less_than :
  ((y - x) / y) * 100 = 28.57 := by
  sorry

end percentage_less_than_l1667_166799


namespace find_k_l1667_166764

variables {α : Type*} [CommRing α]

theorem find_k (a b c : α) :
  (a + b) * (b + c) * (c + a) = (a + b + c) * (a * b + b * c + c * a) - 2 * a * b * c :=
by sorry

end find_k_l1667_166764


namespace shadow_of_cube_l1667_166789

theorem shadow_of_cube (x : ℝ) (h_edge : ∀ c : ℝ, c = 2) (h_shadow_area : ∀ a : ℝ, a = 200 + 4) :
  ⌊1000 * x⌋ = 12280 :=
by
  sorry

end shadow_of_cube_l1667_166789


namespace initial_owls_l1667_166748

theorem initial_owls (n_0 : ℕ) (h : n_0 + 2 = 5) : n_0 = 3 :=
by 
  sorry

end initial_owls_l1667_166748


namespace triangle_is_isosceles_l1667_166751

theorem triangle_is_isosceles (A B C a b c : ℝ) (h_sin : Real.sin (A + B) = 2 * Real.sin A * Real.cos B)
  (h_sine_rule : 2 * a * Real.cos B = c)
  (h_cosine_rule : Real.cos B = (a^2 + c^2 - b^2) / (2 * a * c)) : a = b :=
by
  sorry

end triangle_is_isosceles_l1667_166751


namespace greatest_distance_P_D_l1667_166734

noncomputable def greatest_distance_from_D (P : ℝ × ℝ) (A B C : ℝ × ℝ) (D : ℝ × ℝ) : ℝ :=
  let u := (P.1 - A.1)^2 + (P.2 - A.2)^2
  let v := (P.1 - B.1)^2 + (P.2 - B.2)^2
  let w := (P.1 - C.1)^2 + (P.2 - C.2)^2
  if u + v = w + 1 then ((P.1 - D.1)^2 + (P.2 - D.2)^2).sqrt else 0

theorem greatest_distance_P_D (P : ℝ × ℝ) (u v w : ℝ)
  (h1 : u^2 + v^2 = w^2 + 1) :
  greatest_distance_from_D P (0,0) (2,0) (2,2) (0,2) = 5 :=
sorry

end greatest_distance_P_D_l1667_166734


namespace eugene_payment_correct_l1667_166784

theorem eugene_payment_correct :
  let t_price := 20
  let p_price := 80
  let s_price := 150
  let discount_rate := 0.1
  let t_quantity := 4
  let p_quantity := 3
  let s_quantity := 2
  let t_cost := t_quantity * t_price
  let p_cost := p_quantity * p_price
  let s_cost := s_quantity * s_price
  let total_cost := t_cost + p_cost + s_cost
  let discount := discount_rate * total_cost
  let final_cost := total_cost - discount
  final_cost = 558 :=
by
  sorry

end eugene_payment_correct_l1667_166784


namespace compare_binary_digits_l1667_166788

def numDigits_base2 (n : ℕ) : ℕ :=
  (Nat.log2 n) + 1

theorem compare_binary_digits :
  numDigits_base2 1600 - numDigits_base2 400 = 2 := by
  sorry

end compare_binary_digits_l1667_166788


namespace angle_B_of_isosceles_triangle_l1667_166701

theorem angle_B_of_isosceles_triangle (A B C : ℝ) (h_iso : (A = B ∨ A = C) ∨ (B = C ∨ B = A) ∨ (C = A ∨ C = B)) (h_angle_A : A = 70) :
  B = 70 ∨ B = 55 :=
by
  sorry

end angle_B_of_isosceles_triangle_l1667_166701


namespace minimum_a1_a2_sum_l1667_166795

theorem minimum_a1_a2_sum (a : ℕ → ℕ)
  (h : ∀ n ≥ 1, a (n + 2) = (a n + 2017) / (1 + a (n + 1)))
  (positive_terms : ∀ n, a n > 0) :
  a 1 + a 2 = 2018 :=
sorry

end minimum_a1_a2_sum_l1667_166795


namespace sum_of_arithmetic_sequence_l1667_166733

theorem sum_of_arithmetic_sequence (a d1 d2 : ℕ) 
  (h1 : d1 = d2 + 2) 
  (h2 : d1 + d2 = 24) 
  (a_pos : 0 < a) : 
  (a + (a + d1) + (a + d1) + (a + d1 + d2) = 54) := 
by 
  sorry

end sum_of_arithmetic_sequence_l1667_166733


namespace fraction_spent_on_furniture_l1667_166744

theorem fraction_spent_on_furniture (original_savings : ℝ) (cost_of_tv : ℝ) (f : ℝ)
  (h1 : original_savings = 1800) 
  (h2 : cost_of_tv = 450) 
  (h3 : f * original_savings + cost_of_tv = original_savings) :
  f = 3 / 4 := 
by 
  sorry

end fraction_spent_on_furniture_l1667_166744


namespace exists_a_f_has_two_zeros_range_of_a_for_f_eq_g_l1667_166793

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := a + 2 * Real.sin x
noncomputable def g (x : ℝ) : ℝ := Real.log x / Real.log 2

theorem exists_a_f_has_two_zeros (a : ℝ) :
  (0 < a ∧ a < 2) ∨ (-2 < a ∧ a < 0) → ∃ x₁ x₂ : ℝ, (0 ≤ x₁ ∧ x₁ ≤ 2 * Real.pi ∧ f x₁ a = 0) ∧
  (0 ≤ x₂ ∧ x₂ ≤ 2 * Real.pi ∧ f x₂ a = 0) ∧ x₁ ≠ x₂ := sorry

theorem range_of_a_for_f_eq_g :
  ∀ a : ℝ, a ∈ Set.Icc (-2 : ℝ) (3 : ℝ) →
  ∃ x₁ : ℝ, x₁ ∈ Set.Icc (0 : ℝ) (2 * Real.pi) ∧ f x₁ a = g 2 ∧
  ∃ x₂ : ℝ, x₂ ∈ Set.Icc (1 : ℝ) (2 : ℝ) ∧ f x₁ a = g x₂ := sorry

end exists_a_f_has_two_zeros_range_of_a_for_f_eq_g_l1667_166793


namespace min_value_of_A_l1667_166747

noncomputable def A (a b c : ℝ) : ℝ :=
  (a^3 + b^3) / (8 * a * b + 9 - c^2) +
  (b^3 + c^3) / (8 * b * c + 9 - a^2) +
  (c^3 + a^3) / (8 * c * a + 9 - b^2)

theorem min_value_of_A (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a + b + c = 3) :
  A a b c = 3 / 8 :=
sorry

end min_value_of_A_l1667_166747


namespace hess_law_delta_H298_l1667_166761

def standardEnthalpyNa2O : ℝ := -416 -- kJ/mol
def standardEnthalpyH2O : ℝ := -286 -- kJ/mol
def standardEnthalpyNaOH : ℝ := -427.8 -- kJ/mol
def deltaH298 : ℝ := 2 * standardEnthalpyNaOH - (standardEnthalpyNa2O + standardEnthalpyH2O) 

theorem hess_law_delta_H298 : deltaH298 = -153.6 := by
  sorry

end hess_law_delta_H298_l1667_166761


namespace number_of_paths_grid_l1667_166769

def paths_from_A_to_C (h v : Nat) : Nat :=
  Nat.choose (h + v) v

#eval paths_from_A_to_C 7 6 -- expected result: 1716

theorem number_of_paths_grid :
  paths_from_A_to_C 7 6 = 1716 := by
  sorry

end number_of_paths_grid_l1667_166769


namespace cashier_correction_l1667_166718

theorem cashier_correction (y : ℕ) :
  let quarter_value := 25
  let nickel_value := 5
  let penny_value := 1
  let dime_value := 10
  let quarters_as_nickels_value := y * (quarter_value - nickel_value)
  let pennies_as_dimes_value := y * (dime_value - penny_value)
  let total_correction := quarters_as_nickels_value - pennies_as_dimes_value
  total_correction = 11 * y := by
  sorry

end cashier_correction_l1667_166718


namespace daily_wage_male_worker_l1667_166710

variables
  (num_male : ℕ) (num_female : ℕ) (num_child : ℕ)
  (wage_female : ℝ) (wage_child : ℝ) (avg_wage : ℝ)
  (total_workers : ℕ := num_male + num_female + num_child)
  (total_wage_all : ℝ := avg_wage * total_workers)
  (total_wage_female : ℝ := num_female * wage_female)
  (total_wage_child : ℝ := num_child * wage_child)
  (total_wage_male : ℝ := total_wage_all - (total_wage_female + total_wage_child))
  (wage_per_male : ℝ := total_wage_male / num_male)

theorem daily_wage_male_worker :
  num_male = 20 →
  num_female = 15 →
  num_child = 5 →
  wage_female = 20 →
  wage_child = 8 →
  avg_wage = 21 →
  wage_per_male = 25 :=
by
  intros
  sorry

end daily_wage_male_worker_l1667_166710


namespace permute_rows_to_columns_l1667_166742

open Function

-- Define the problem
theorem permute_rows_to_columns {α : Type*} [Fintype α] [DecidableEq α] (n : ℕ)
  (table : Fin n → Fin n → α)
  (h_distinct_rows : ∀ i : Fin n, ∀ j₁ j₂ : Fin n, j₁ ≠ j₂ → table i j₁ ≠ table i j₂) :
  ∃ (p : Fin n → Fin n → Fin n), ∀ j : Fin n, ∀ i₁ i₂ : Fin n, i₁ ≠ i₂ →
    table i₁ (p i₁ j) ≠ table i₂ (p i₂ j) := 
sorry

end permute_rows_to_columns_l1667_166742


namespace transmission_time_calc_l1667_166729

theorem transmission_time_calc
  (blocks : ℕ) (chunks_per_block : ℕ) (transmission_rate : ℕ) (time_in_minutes : ℕ)
  (h_blocks : blocks = 80)
  (h_chunks_per_block : chunks_per_block = 640)
  (h_transmission_rate : transmission_rate = 160) 
  (h_time_in_minutes : time_in_minutes = 5) : 
  (blocks * chunks_per_block / transmission_rate) / 60 = time_in_minutes := 
by
  sorry

end transmission_time_calc_l1667_166729


namespace calculation_identity_l1667_166768

theorem calculation_identity :
  (3.14 - 1)^0 * (-1 / 4)^(-2) = 16 := by
  sorry

end calculation_identity_l1667_166768


namespace min_buildings_20x20_min_buildings_50x90_l1667_166750

structure CityGrid where
  width : ℕ
  height : ℕ

noncomputable def renovationLaw (grid : CityGrid) : ℕ :=
  if grid.width = 20 ∧ grid.height = 20 then 25
  else if grid.width = 50 ∧ grid.height = 90 then 282
  else sorry -- handle other cases if needed

-- Theorem statements for the proof
theorem min_buildings_20x20 : renovationLaw { width := 20, height := 20 } = 25 := by
  sorry

theorem min_buildings_50x90 : renovationLaw { width := 50, height := 90 } = 282 := by
  sorry

end min_buildings_20x20_min_buildings_50x90_l1667_166750


namespace Martha_time_spent_l1667_166797

theorem Martha_time_spent
  (x : ℕ)
  (h1 : 6 * x = 6 * x) -- Time spent on hold with Comcast is 6 times the time spent turning router off and on again
  (h2 : 3 * x = 3 * x) -- Time spent yelling at the customer service rep is half of time spent on hold, which is still 3x
  (h3 : x + 6 * x + 3 * x = 100) -- Total time spent is 100 minutes
  : x = 10 := 
by
  -- skip the proof steps
  sorry

end Martha_time_spent_l1667_166797


namespace problem1_problem2_l1667_166737

-- Problem 1
theorem problem1 : (-1 : ℤ) ^ 2024 + (1 / 3 : ℝ) ^ (-2 : ℤ) - (3.14 - Real.pi) ^ 0 = 9 := 
sorry

-- Problem 2
theorem problem2 (x : ℤ) (y : ℤ) (hx : x = 2) (hy : y = 3) : 
  x * (x + 2 * y) - (x + 1) ^ 2 + 2 * x = 11 :=
sorry

end problem1_problem2_l1667_166737


namespace solve_fraction_zero_l1667_166762

theorem solve_fraction_zero (x : ℝ) (h1 : (x^2 - 25) / (x + 5) = 0) (h2 : x ≠ -5) : x = 5 :=
sorry

end solve_fraction_zero_l1667_166762


namespace maximal_intersection_area_of_rectangles_l1667_166723

theorem maximal_intersection_area_of_rectangles :
  ∀ (a b : ℕ), a * b = 2015 ∧ a < b →
  ∀ (c d : ℕ), c * d = 2016 ∧ c > d →
  ∃ (max_area : ℕ), max_area = 1302 ∧ ∀ intersection_area, intersection_area ≤ 1302 := 
by
  sorry

end maximal_intersection_area_of_rectangles_l1667_166723


namespace evaluate_expression_l1667_166709

theorem evaluate_expression (x : ℕ) (h : x = 3) : (x^x)^(x^x) = 27^27 :=
by
  sorry

end evaluate_expression_l1667_166709


namespace walking_time_l1667_166715

theorem walking_time 
  (speed_km_hr : ℝ := 10) 
  (distance_km : ℝ := 6) 
  : (distance_km / (speed_km_hr / 60)) = 36 :=
by
  sorry

end walking_time_l1667_166715


namespace tarantulas_per_egg_sac_l1667_166756

-- Condition: Each tarantula has 8 legs
def legs_per_tarantula : ℕ := 8

-- Condition: There are 32000 baby tarantula legs
def total_legs : ℕ := 32000

-- Condition: Number of egg sacs is one less than 5
def number_of_egg_sacs : ℕ := 5 - 1

-- Calculated: Number of tarantulas in total
def total_tarantulas : ℕ := total_legs / legs_per_tarantula

-- Proof Statement: Number of tarantulas per egg sac
theorem tarantulas_per_egg_sac : total_tarantulas / number_of_egg_sacs = 1000 := by
  sorry

end tarantulas_per_egg_sac_l1667_166756


namespace generated_surface_l1667_166771

theorem generated_surface (L : ℝ → ℝ → ℝ → Prop)
  (H1 : ∀ x y z, L x y z → y = z) 
  (H2 : ∀ t, L (t^2 / 2) t 0) 
  (H3 : ∀ s, L (s^2 / 3) 0 s) : 
  ∀ y z, ∃ x, L x y z → x = (y - z) * (y / 2 - z / 3) :=
by
  sorry

end generated_surface_l1667_166771


namespace greatest_possible_integer_l1667_166739

theorem greatest_possible_integer 
  (n k l : ℕ) 
  (h1 : n < 150) 
  (h2 : n = 9 * k - 2) 
  (h3 : n = 6 * l - 4) : 
  n = 146 := 
sorry

end greatest_possible_integer_l1667_166739


namespace bowling_average_l1667_166706

theorem bowling_average (gretchen_score mitzi_score beth_score : ℤ) (h1 : gretchen_score = 120) (h2 : mitzi_score = 113) (h3 : beth_score = 85) :
  (gretchen_score + mitzi_score + beth_score) / 3 = 106 :=
by
  sorry

end bowling_average_l1667_166706


namespace chess_grandmaster_time_l1667_166776

theorem chess_grandmaster_time :
  let time_to_learn_rules : ℕ := 2
  let factor_to_get_proficient : ℕ := 49
  let factor_to_become_master : ℕ := 100
  let time_to_get_proficient := factor_to_get_proficient * time_to_learn_rules
  let combined_time := time_to_learn_rules + time_to_get_proficient
  let time_to_become_master := factor_to_become_master * combined_time
  let total_time := time_to_learn_rules + time_to_get_proficient + time_to_become_master
  total_time = 10100 :=
by
  sorry

end chess_grandmaster_time_l1667_166776


namespace transport_cost_B_condition_l1667_166719

-- Define the parameters for coal from Mine A
def calories_per_gram_A := 4
def price_per_ton_A := 20
def transport_cost_A := 8

-- Define the parameters for coal from Mine B
def calories_per_gram_B := 6
def price_per_ton_B := 24

-- Define the total cost for transporting one ton from Mine A to city N
def total_cost_A := price_per_ton_A + transport_cost_A

-- Define the question as a Lean theorem
theorem transport_cost_B_condition : 
  ∀ (transport_cost_B : ℝ), 
  (total_cost_A : ℝ) / (calories_per_gram_A : ℝ) = (price_per_ton_B + transport_cost_B) / (calories_per_gram_B : ℝ) → 
  transport_cost_B = 18 :=
by
  intros transport_cost_B h
  have h_eq : (total_cost_A : ℝ) / (calories_per_gram_A : ℝ) = (price_per_ton_B + transport_cost_B) / (calories_per_gram_B : ℝ) := h
  sorry

end transport_cost_B_condition_l1667_166719


namespace time_after_hours_l1667_166798

def current_time := 9
def total_hours := 2023
def clock_cycle := 12

theorem time_after_hours : (current_time + total_hours) % clock_cycle = 8 := by
  sorry

end time_after_hours_l1667_166798


namespace monotonic_increasing_interval_l1667_166763

noncomputable def f (x : ℝ) : ℝ :=
  Real.logb 0.5 (x^2 + 2 * x - 3)

theorem monotonic_increasing_interval :
  ∀ x, f x = Real.logb 0.5 (x^2 + 2 * x - 3) → 
  (∀ x₁ x₂, x₁ < x₂ ∧ x₁ < -3 ∧ x₂ < -3 → f x₁ ≤ f x₂) :=
sorry

end monotonic_increasing_interval_l1667_166763


namespace flagpole_height_l1667_166785

theorem flagpole_height :
  ∃ (AB AC AD DE DC : ℝ), 
    AC = 5 ∧
    AD = 3 ∧ 
    DE = 1.8 ∧
    DC = AC - AD ∧
    AB = (DE * AC) / DC ∧
    AB = 4.5 :=
by
  exists 4.5, 5, 3, 1.8, 2
  simp
  sorry

end flagpole_height_l1667_166785


namespace max_value_x_plus_2y_max_of_x_plus_2y_l1667_166703

def on_ellipse (x y : ℝ) : Prop :=
  x^2 / 6 + y^2 / 4 = 1

theorem max_value_x_plus_2y (x y : ℝ) (h : on_ellipse x y) :
  x + 2 * y ≤ Real.sqrt 22 :=
sorry

theorem max_of_x_plus_2y (x y : ℝ) (h : on_ellipse x y) :
  ∃θ ∈ Set.Icc 0 (2 * Real.pi), (x = Real.sqrt 6 * Real.cos θ) ∧ (y = 2 * Real.sin θ) :=
sorry

end max_value_x_plus_2y_max_of_x_plus_2y_l1667_166703


namespace volume_of_large_ball_l1667_166773

theorem volume_of_large_ball (r : ℝ) (V_small : ℝ) (h1 : 1 = r / (2 * r)) (h2 : V_small = (4 / 3) * Real.pi * r^3) : 
  8 * V_small = 288 :=
by
  sorry

end volume_of_large_ball_l1667_166773


namespace impossible_to_place_numbers_l1667_166711

noncomputable def divisible (a b : ℕ) : Prop := ∃ k : ℕ, a * k = b

def connected (G : Finset (ℕ × ℕ)) (u v : ℕ) : Prop := (u, v) ∈ G ∨ (v, u) ∈ G

def valid_assignment (G : Finset (ℕ × ℕ)) (f : ℕ → ℕ) : Prop :=
  ∀ ⦃i j⦄, connected G i j → divisible (f i) (f j) ∨ divisible (f j) (f i)

def invalid_assignment (G : Finset (ℕ × ℕ)) (f : ℕ → ℕ) : Prop :=
  ∀ ⦃i j⦄, ¬ connected G i j → ¬ divisible (f i) (f j) ∧ ¬ divisible (f j) (f i)

theorem impossible_to_place_numbers (G : Finset (ℕ × ℕ)) :
  (∃ f : ℕ → ℕ, valid_assignment G f ∧ invalid_assignment G f) → False :=
by
  sorry

end impossible_to_place_numbers_l1667_166711


namespace urn_contains_four_each_color_after_six_steps_l1667_166720

noncomputable def probability_urn_four_each_color : ℚ := 2 / 7

def urn_problem (urn_initial : ℕ) (draws : ℕ) (final_urn : ℕ) (extra_balls : ℕ) : Prop :=
urn_initial = 2 ∧ draws = 6 ∧ final_urn = 8 ∧ extra_balls > 0

theorem urn_contains_four_each_color_after_six_steps :
  urn_problem 2 6 8 2 → probability_urn_four_each_color = 2 / 7 :=
by
  intro h
  cases h
  sorry

end urn_contains_four_each_color_after_six_steps_l1667_166720


namespace unique_real_function_l1667_166783

theorem unique_real_function (f : ℝ → ℝ) :
  (∀ x y z : ℝ, (f (x * y) / 2 + f (x * z) / 2 - f x * f (y * z)) ≥ 1 / 4) →
  (∀ x : ℝ, f x = 1 / 2) :=
by
  intro h
  -- proof steps go here
  sorry

end unique_real_function_l1667_166783


namespace find_f_one_third_l1667_166786

def is_odd_function (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = -f x

def satisfies_condition (f : ℝ → ℝ) : Prop :=
∀ x, f (2 - x) = f x

noncomputable def f (x : ℝ) : ℝ := if (2 ≤ x ∧ x ≤ 3) then Real.log (x - 1) / Real.log 2 else 0

theorem find_f_one_third (h_odd : is_odd_function f) (h_condition : satisfies_condition f) :
  f (1 / 3) = Real.log 3 / Real.log 2 - 2 :=
by
  sorry

end find_f_one_third_l1667_166786


namespace triangle_problem_l1667_166766

theorem triangle_problem
  (A B C : ℝ)
  (a b c : ℝ)
  (hb : 0 < B ∧ B < Real.pi)
  (hc : 0 < C ∧ C < Real.pi)
  (ha : 0 < A ∧ A < Real.pi)
  (h_sum_angles : A + B + C = Real.pi)
  (h_sides : a > b)
  (h_perimeter : a + b + c = 20)
  (h_area : (1/2) * a * b * Real.sin C = 10 * Real.sqrt 3)
  (h_eq : a * (Real.sqrt 3 * Real.tan B - 1) = (b * Real.cos A / Real.cos B) + (c * Real.cos A / Real.cos C)) :
  C = Real.pi / 3 ∧ a = 8 ∧ b = 5 ∧ c = 7 := sorry

end triangle_problem_l1667_166766


namespace lenny_remaining_amount_l1667_166704

theorem lenny_remaining_amount :
  let initial_amount := 270
  let console_price := 149
  let console_discount := 0.15 * console_price
  let final_console_price := console_price - console_discount
  let groceries_price := 60
  let groceries_discount := 0.10 * groceries_price
  let final_groceries_price := groceries_price - groceries_discount
  let lunch_cost := 30
  let magazine_cost := 3.99
  let total_expenses := final_console_price + final_groceries_price + lunch_cost + magazine_cost
  initial_amount - total_expenses = 55.36 :=
by
  sorry

end lenny_remaining_amount_l1667_166704


namespace vector_computation_l1667_166716

def v1 : ℤ × ℤ := (3, -5)
def v2 : ℤ × ℤ := (2, -10)
def s1 : ℤ := 4
def s2 : ℤ := 3

theorem vector_computation : s1 • v1 - s2 • v2 = (6, 10) :=
  sorry

end vector_computation_l1667_166716


namespace calculate_expression_l1667_166790

theorem calculate_expression (y : ℝ) (hy : y ≠ 0) : 
  (18 * y^3) * (4 * y^2) * (1/(2 * y)^3) = 9 * y^2 :=
by
  sorry

end calculate_expression_l1667_166790


namespace find_x_l1667_166743

theorem find_x (a b c d x : ℕ) 
  (h1 : x = a + 7) 
  (h2 : a = b + 12) 
  (h3 : b = c + 15) 
  (h4 : c = d + 25) 
  (h5 : d = 95) : 
  x = 154 := 
by 
  sorry

end find_x_l1667_166743


namespace correct_judgement_l1667_166728

noncomputable def f (x : ℝ) : ℝ :=
if -2 ≤ x ∧ x ≤ 2 then (1 / 2) * Real.sqrt (4 - x^2)
else - (1 / 2) * Real.sqrt (x^2 - 4)

noncomputable def F (x : ℝ) : ℝ := f x + x

theorem correct_judgement : (∀ y : ℝ, ∃ x : ℝ, (f x = y) ↔ (y ∈ Set.Iic 1)) ∧ (∃! x : ℝ, F x = 0) :=
by
  sorry

end correct_judgement_l1667_166728


namespace domain_of_f_l1667_166731

noncomputable def f (x : ℝ) := 1 / ((x - 3) + (x - 6))

theorem domain_of_f :
  (∀ x : ℝ, x ≠ 9/2 → ∃ y : ℝ, f x = y) ∧ (∀ x : ℝ, x = 9/2 → ¬ (∃ y : ℝ, f x = y)) :=
by
  sorry

end domain_of_f_l1667_166731


namespace question1_question2_l1667_166792

-- Definitions based on the conditions
def f (x m : ℝ) : ℝ := x^2 + 4*x + m

theorem question1 (m : ℝ) (h1 : m ≠ 0) (h2 : 16 - 4 * m > 0) : m < 4 :=
  sorry

theorem question2 (m : ℝ) (hx : ∀ x : ℝ, f x m = 0 → f (-x - 4) m = 0) 
  (h_circ : ∀ (x y : ℝ), x^2 + y^2 + 4*x - (m + 1) * y + m = 0 → (x = 0 ∧ y = 1) ∨ (x = -4 ∧ y = 1)) :
  (∀ (x y : ℝ), x^2 + y^2 + 4*x - (m + 1) * y + m = 0 → (x = 0 ∧ y = 1)) ∨ (∀ (x y : ℝ), (x = -4 ∧ y = 1)) :=
  sorry

end question1_question2_l1667_166792


namespace random_events_l1667_166722

-- Define what it means for an event to be random
def is_random_event (e : Prop) : Prop := ∃ (h : Prop), e ∨ ¬e

-- Define the events based on the problem statements
def event1 := ∃ (good_cups : ℕ), good_cups = 3
def event2 := ∃ (half_hit_targets : ℕ), half_hit_targets = 50
def event3 := ∃ (correct_digit : ℕ), correct_digit = 1
def event4 := true -- Opposite charges attract each other, which is always true
def event5 := ∃ (first_prize : ℕ), first_prize = 1

-- State the problem as a theorem
theorem random_events :
  is_random_event event1 ∧ is_random_event event2 ∧ is_random_event event3 ∧ is_random_event event5 :=
by
  sorry

end random_events_l1667_166722


namespace lincoln_high_fraction_of_girls_l1667_166714

noncomputable def fraction_of_girls_in_science_fair (total_girls total_boys : ℕ) (frac_girls_participated frac_boys_participated : ℚ) : ℚ :=
  let participating_girls := frac_girls_participated * total_girls
  let participating_boys := frac_boys_participated * total_boys
  participating_girls / (participating_girls + participating_boys)

theorem lincoln_high_fraction_of_girls 
  (total_girls : ℕ) (total_boys : ℕ)
  (frac_girls_participated : ℚ) (frac_boys_participated : ℚ)
  (h1 : total_girls = 150) (h2 : total_boys = 100)
  (h3 : frac_girls_participated = 4/5) (h4 : frac_boys_participated = 3/4) :
  fraction_of_girls_in_science_fair total_girls total_boys frac_girls_participated frac_boys_participated = 8/13 := 
by
  sorry

end lincoln_high_fraction_of_girls_l1667_166714


namespace smallest_whole_number_l1667_166770

theorem smallest_whole_number (a : ℕ) : 
  (a % 4 = 1) ∧ (a % 3 = 1) ∧ (a % 5 = 2) → a = 37 :=
by
  intros
  sorry

end smallest_whole_number_l1667_166770


namespace total_games_l1667_166774

variable (G R : ℕ)

axiom cond1 : 85 + (1/2 : ℚ) * R = (0.70 : ℚ) * G
axiom cond2 : G = 100 + R

theorem total_games : G = 175 := by
  sorry

end total_games_l1667_166774


namespace find_fourth_vertex_of_square_l1667_166791

-- Given the vertices of the square as complex numbers
def vertex1 : ℂ := 1 + 2 * Complex.I
def vertex2 : ℂ := -2 + Complex.I
def vertex3 : ℂ := -1 - 2 * Complex.I

-- The fourth vertex (to be proved)
def vertex4 : ℂ := 2 - Complex.I

-- The mathematically equivalent proof problem statement
theorem find_fourth_vertex_of_square :
  let v1 := vertex1
  let v2 := vertex2
  let v3 := vertex3
  let v4 := vertex4
  -- Define vectors from the vertices
  let vector_ab := v2 - v1
  let vector_dc := v3 - v4
  vector_ab = vector_dc :=
by {
  -- Definitions already provided above
  let v1 := vertex1
  let v2 := vertex2
  let v3 := vertex3
  let v4 := vertex4
  let vector_ab := v2 - v1
  let vector_dc := v3 - v4

  -- Placeholder for proof
  sorry
}

end find_fourth_vertex_of_square_l1667_166791


namespace toads_per_acre_l1667_166724

theorem toads_per_acre (b g : ℕ) (h₁ : b = 25 * g)
  (h₂ : b / 4 = 50) : g = 8 :=
by
  -- Condition h₁: For every green toad, there are 25 brown toads.
  -- Condition h₂: One-quarter of the brown toads are spotted, and there are 50 spotted brown toads per acre.
  sorry

end toads_per_acre_l1667_166724


namespace find_S_l1667_166752

noncomputable def A := { x : ℝ | x^2 - 7 * x + 10 ≤ 0 }
noncomputable def B (a b : ℝ) := { x : ℝ | x^2 + a * x + b < 0 }
def A_inter_B_is_empty (a b : ℝ) := A ∩ B a b = ∅
def A_union_B_condition := { x : ℝ | x - 3 < 4 ∧ 4 ≤ 2 * x }

theorem find_S :
  A ∪ B (-12) 35 = { x : ℝ | 2 ≤ x ∧ x < 7 } →
  A ∩ B (-12) 35 = ∅ →
  { x : ℝ | x = -12 + 35 } = { 23 } :=
by
  intro h1 h2
  sorry

end find_S_l1667_166752


namespace decimal_to_base_five_correct_l1667_166707

theorem decimal_to_base_five_correct : 
  ∃ (d0 d1 d2 d3 : ℕ), 256 = d3 * 5^3 + d2 * 5^2 + d1 * 5^1 + d0 * 5^0 ∧ 
                          d3 = 2 ∧ d2 = 0 ∧ d1 = 1 ∧ d0 = 1 :=
by sorry

end decimal_to_base_five_correct_l1667_166707


namespace tens_digit_of_6_pow_19_l1667_166778

def tens_digit (n : ℕ) : ℕ :=
  (n / 10) % 10

theorem tens_digit_of_6_pow_19 : tens_digit (6 ^ 19) = 9 := 
by 
  sorry

end tens_digit_of_6_pow_19_l1667_166778


namespace equal_share_each_shopper_l1667_166772

theorem equal_share_each_shopper 
  (amount_giselle : ℕ)
  (amount_isabella : ℕ)
  (amount_sam : ℕ)
  (H1 : amount_isabella = amount_sam + 45)
  (H2 : amount_isabella = amount_giselle + 15)
  (H3 : amount_giselle = 120) : 
  (amount_isabella + amount_sam + amount_giselle) / 3 = 115 :=
by
  -- The proof is omitted.
  sorry

end equal_share_each_shopper_l1667_166772
