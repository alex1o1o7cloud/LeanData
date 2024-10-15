import Mathlib

namespace NUMINAMATH_GPT_Ben_Cards_Left_l1247_124786

theorem Ben_Cards_Left :
  (4 * 10 + 5 * 8 - 58) = 22 :=
by
  sorry

end NUMINAMATH_GPT_Ben_Cards_Left_l1247_124786


namespace NUMINAMATH_GPT_cherries_left_l1247_124708

def initial_cherries : ℕ := 77
def cherries_used : ℕ := 60

theorem cherries_left : initial_cherries - cherries_used = 17 := by
  sorry

end NUMINAMATH_GPT_cherries_left_l1247_124708


namespace NUMINAMATH_GPT_find_theta_plus_3phi_l1247_124743

variables (θ φ : ℝ)

-- The conditions
variables (h1 : 0 < θ ∧ θ < π / 2) (h2 : 0 < φ ∧ φ < π / 2)
variables (h3 : Real.tan θ = 1 / 3) (h4 : Real.sin φ = 3 / 5)

theorem find_theta_plus_3phi :
  θ + 3 * φ = π - Real.arctan (199 / 93) :=
sorry

end NUMINAMATH_GPT_find_theta_plus_3phi_l1247_124743


namespace NUMINAMATH_GPT_find_f_six_l1247_124706

noncomputable def f : ℝ → ℝ := sorry

theorem find_f_six (f : ℝ → ℝ)
  (h1 : ∀ x y : ℝ, x * f y = y * f x)
  (h2 : f 18 = 24) :
  f 6 = 8 :=
sorry

end NUMINAMATH_GPT_find_f_six_l1247_124706


namespace NUMINAMATH_GPT_find_k_b_find_x_when_y_neg_8_l1247_124731

theorem find_k_b (k b : ℤ) (h1 : -20 = 4 * k + b) (h2 : 16 = -2 * k + b) : k = -6 ∧ b = 4 := 
sorry

theorem find_x_when_y_neg_8 (x : ℤ) (k b : ℤ) (h_k : k = -6) (h_b : b = 4) (h_target : -8 = k * x + b) : x = 2 := 
sorry

end NUMINAMATH_GPT_find_k_b_find_x_when_y_neg_8_l1247_124731


namespace NUMINAMATH_GPT_only_p_eq_3_l1247_124722

theorem only_p_eq_3 (p : ℕ) (h1 : Prime p) (h2 : Prime (8 * p ^ 2 + 1)) : p = 3 := 
by
  sorry

end NUMINAMATH_GPT_only_p_eq_3_l1247_124722


namespace NUMINAMATH_GPT_polygon_area_is_400_l1247_124790

def Point : Type := (ℤ × ℤ)

def area_of_polygon (vertices : List Point) : ℤ := 
  -- Formula to calculate polygon area would go here
  -- As a placeholder, for now we return 400 since proof details aren't required
  400

theorem polygon_area_is_400 :
  area_of_polygon [(0,0), (20,0), (30,10), (20,20), (0,20), (10,10), (0,0)] = 400 := by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_polygon_area_is_400_l1247_124790


namespace NUMINAMATH_GPT_non_negative_sequence_l1247_124717

theorem non_negative_sequence
  (a : Fin 100 → ℝ)
  (h₁ : a 0 = a 99)
  (h₂ : ∀ i : Fin 97, a i - 2 * a (i+1) + a (i+2) ≤ 0)
  (h₃ : a 0 ≥ 0) :
  ∀ i : Fin 100, a i ≥ 0 :=
by
  sorry

end NUMINAMATH_GPT_non_negative_sequence_l1247_124717


namespace NUMINAMATH_GPT_linear_equation_solution_l1247_124766

theorem linear_equation_solution (x y : ℝ) (h : 3 * x - y = 5) : y = 3 * x - 5 :=
sorry

end NUMINAMATH_GPT_linear_equation_solution_l1247_124766


namespace NUMINAMATH_GPT_find_square_number_divisible_by_five_l1247_124700

noncomputable def is_square (n : ℕ) : Prop :=
∃ k : ℕ, k * k = n

theorem find_square_number_divisible_by_five :
  ∃ x : ℕ, x ≥ 50 ∧ x ≤ 120 ∧ is_square x ∧ x % 5 = 0 ↔ x = 100 := by
sorry

end NUMINAMATH_GPT_find_square_number_divisible_by_five_l1247_124700


namespace NUMINAMATH_GPT_infinitely_many_coprime_binomials_l1247_124751

theorem infinitely_many_coprime_binomials (k l : ℕ) (hk : 0 < k) (hl : 0 < l) :
  ∃ᶠ n in at_top, n > k ∧ Nat.gcd (Nat.choose n k) l = 1 := by
  sorry

end NUMINAMATH_GPT_infinitely_many_coprime_binomials_l1247_124751


namespace NUMINAMATH_GPT_jogger_usual_speed_l1247_124761

theorem jogger_usual_speed (V T : ℝ) 
    (h_actual: 30 = V * T) 
    (h_condition: 40 = 16 * T) 
    (h_distance: T = 30 / V) :
  V = 12 := 
by
  sorry

end NUMINAMATH_GPT_jogger_usual_speed_l1247_124761


namespace NUMINAMATH_GPT_find_value_l1247_124744

theorem find_value (x : ℝ) (hx : x + 1/x = 4) : x^3 + 1/x^3 = 52 := 
by 
  sorry

end NUMINAMATH_GPT_find_value_l1247_124744


namespace NUMINAMATH_GPT_right_triangle_can_form_isosceles_l1247_124701

-- Definitions for the problem
structure RightTriangle :=
  (a b : ℝ) -- The legs of the right triangle
  (c : ℝ)  -- The hypotenuse of the right triangle
  (h1 : c = Real.sqrt (a ^ 2 + b ^ 2)) -- Pythagoras theorem

-- The triangle attachment requirement definition
def IsoscelesTriangleAttachment (rightTriangle : RightTriangle) : Prop :=
  ∃ (b1 b2 : ℝ), -- Two base sides of the new triangle sharing one side with the right triangle
    (b1 ≠ b2) ∧ -- They should be different to not overlap
    (b1 = rightTriangle.a ∨ b1 = rightTriangle.b) ∧ -- Share one side with the right triangle
    (b2 ≠ rightTriangle.a ∧ b2 ≠ rightTriangle.b) ∧ -- Ensure non-overlapping
    (b1^2 + b2^2 = rightTriangle.c^2)

-- The statement to prove
theorem right_triangle_can_form_isosceles (T : RightTriangle) : IsoscelesTriangleAttachment T :=
sorry

end NUMINAMATH_GPT_right_triangle_can_form_isosceles_l1247_124701


namespace NUMINAMATH_GPT_solve_equation_l1247_124788

theorem solve_equation : ∀ x : ℝ, ((1 - x) / (x - 4)) + (1 / (4 - x)) = 1 → x = 2 :=
by
  intros x h
  sorry

end NUMINAMATH_GPT_solve_equation_l1247_124788


namespace NUMINAMATH_GPT_find_value_of_a_l1247_124707

theorem find_value_of_a (a b : ℝ) (h1 : ∀ x, (2 < x ∧ x < 4) ↔ (a - b < x ∧ x < a + b)) : a = 3 := by
  sorry

end NUMINAMATH_GPT_find_value_of_a_l1247_124707


namespace NUMINAMATH_GPT_inscribed_circle_radius_of_triangle_l1247_124774

theorem inscribed_circle_radius_of_triangle (a b c : ℕ)
  (h₁ : a = 50) (h₂ : b = 120) (h₃ : c = 130) :
  ∃ r : ℕ, r = 20 :=
by sorry

end NUMINAMATH_GPT_inscribed_circle_radius_of_triangle_l1247_124774


namespace NUMINAMATH_GPT_visible_product_divisible_by_48_l1247_124762

-- We represent the eight-sided die as the set {1, 2, 3, 4, 5, 6, 7, 8}.
-- Q is the product of any seven numbers from this set.

theorem visible_product_divisible_by_48 
   (Q : ℕ)
   (H : ∃ (numbers : Finset ℕ), numbers ⊆ ({1, 2, 3, 4, 5, 6, 7, 8} : Finset ℕ) ∧ numbers.card = 7 ∧ Q = numbers.prod id) :
   48 ∣ Q :=
by
  sorry

end NUMINAMATH_GPT_visible_product_divisible_by_48_l1247_124762


namespace NUMINAMATH_GPT_train_a_constant_rate_l1247_124778

theorem train_a_constant_rate
  (d : ℕ)
  (v_b : ℕ)
  (d_a : ℕ)
  (v : ℕ)
  (h1 : d = 350)
  (h2 : v_b = 30)
  (h3 : d_a = 200)
  (h4 : v * (d_a / v) + v_b * (d_a / v) = d) :
  v = 40 := by
  sorry

end NUMINAMATH_GPT_train_a_constant_rate_l1247_124778


namespace NUMINAMATH_GPT_final_composite_score_is_correct_l1247_124704

-- Defining scores
def written_exam_score : ℝ := 94
def interview_score : ℝ := 80
def practical_operation_score : ℝ := 90

-- Defining weights
def written_exam_weight : ℝ := 5
def interview_weight : ℝ := 2
def practical_operation_weight : ℝ := 3
def total_weight : ℝ := written_exam_weight + interview_weight + practical_operation_weight

-- Final composite score
noncomputable def composite_score : ℝ :=
  (written_exam_score * written_exam_weight + interview_score * interview_weight + practical_operation_score * practical_operation_weight)
  / total_weight

-- The theorem to be proved
theorem final_composite_score_is_correct : composite_score = 90 := by
  sorry

end NUMINAMATH_GPT_final_composite_score_is_correct_l1247_124704


namespace NUMINAMATH_GPT_choose_three_positive_or_two_negative_l1247_124729

theorem choose_three_positive_or_two_negative (n : ℕ) (hn : n ≥ 3) (a : Fin n → ℝ) :
  ∃ (i j k : Fin n), i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ (0 < a i + a j + a k) ∨ ∃ (i j : Fin n), i ≠ j ∧ (a i + a j < 0) := sorry

end NUMINAMATH_GPT_choose_three_positive_or_two_negative_l1247_124729


namespace NUMINAMATH_GPT_average_weight_increase_l1247_124747

theorem average_weight_increase (old_weight : ℕ) (new_weight : ℕ) (n : ℕ) (increase : ℕ) :
  old_weight = 45 → new_weight = 93 → n = 8 → increase = (new_weight - old_weight) / n → increase = 6 :=
by
  intros h_old h_new h_n h_increase
  rw [h_old, h_new, h_n] at h_increase
  simp at h_increase
  exact h_increase

end NUMINAMATH_GPT_average_weight_increase_l1247_124747


namespace NUMINAMATH_GPT_square_area_l1247_124775

theorem square_area
  (E_on_AD : ∃ E : ℝ × ℝ, ∃ s : ℝ, s > 0 ∧ E = (0, s))
  (F_on_extension_BC : ∃ F : ℝ × ℝ, ∃ s : ℝ, s > 0 ∧ F = (s, 0))
  (BE_20 : ∃ B E : ℝ × ℝ, ∃ s : ℝ, B = (s, 0) ∧ E = (0, s) ∧ dist B E = 20)
  (EF_25 : ∃ E F : ℝ × ℝ, ∃ s : ℝ, E = (0, s) ∧ F = (s, 0) ∧ dist E F = 25)
  (FD_20 : ∃ F D : ℝ × ℝ, ∃ s : ℝ, F = (s, 0) ∧ D = (s, s) ∧ dist F D = 20) :
  ∃ s : ℝ, s > 0 ∧ s^2 = 400 :=
by
  -- Hypotheses are laid out in conditions as defined above
  sorry

end NUMINAMATH_GPT_square_area_l1247_124775


namespace NUMINAMATH_GPT_three_digit_number_uniq_l1247_124709

theorem three_digit_number_uniq (n : ℕ) (h : 100 ≤ n ∧ n < 1000)
  (hundreds_digit : n / 100 = 5) (units_digit : n % 10 = 3)
  (div_by_9 : n % 9 = 0) : n = 513 :=
sorry

end NUMINAMATH_GPT_three_digit_number_uniq_l1247_124709


namespace NUMINAMATH_GPT_ratio_of_a_over_5_to_b_over_4_l1247_124746

theorem ratio_of_a_over_5_to_b_over_4 (a b : ℝ) (h1 : 4 * a = 5 * b) (h2 : a ≠ 0 ∧ b ≠ 0) :
  (a / 5) / (b / 4) = 1 := by
  sorry

end NUMINAMATH_GPT_ratio_of_a_over_5_to_b_over_4_l1247_124746


namespace NUMINAMATH_GPT_intersecting_absolute_value_functions_l1247_124720

theorem intersecting_absolute_value_functions (a b c d : ℝ) (h1 : -|2 - a| + b = 5) (h2 : -|8 - a| + b = 3) (h3 : |2 - c| + d = 5) (h4 : |8 - c| + d = 3) (ha : 2 < a) (h8a : a < 8) (hc : 2 < c) (h8c : c < 8) : a + c = 10 :=
sorry

end NUMINAMATH_GPT_intersecting_absolute_value_functions_l1247_124720


namespace NUMINAMATH_GPT_total_marbles_l1247_124784

theorem total_marbles (x : ℕ) (h1 : 5 * x - 2 = 18) : 4 * x + 5 * x = 36 :=
by
  sorry

end NUMINAMATH_GPT_total_marbles_l1247_124784


namespace NUMINAMATH_GPT_triangle_inequality_l1247_124703

theorem triangle_inequality (a b c R r : ℝ) 
  (habc : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_area1 : a * b * c = 4 * R * S)
  (h_area2 : S = r * (a + b + c) / 2) :
  (b^2 + c^2) / (2 * b * c) ≤ R / (2 * r) := 
sorry

end NUMINAMATH_GPT_triangle_inequality_l1247_124703


namespace NUMINAMATH_GPT_find_k_l1247_124710

theorem find_k (k x y : ℝ) (h_ne_zero : k ≠ 0) (h_x : x = 4) (h_y : y = -1/2) (h_eq : y = k / x) : k = -2 :=
by
  -- This is where the proof would go
  sorry

end NUMINAMATH_GPT_find_k_l1247_124710


namespace NUMINAMATH_GPT_least_number_to_subtract_l1247_124768

theorem least_number_to_subtract (n : ℕ) (h : n = 9876543210) : 
  ∃ m, m = 6 ∧ (n - m) % 29 = 0 := 
sorry

end NUMINAMATH_GPT_least_number_to_subtract_l1247_124768


namespace NUMINAMATH_GPT_ms_cole_students_l1247_124787

theorem ms_cole_students (S6 S4 S7 : ℕ)
  (h1: S6 = 40)
  (h2: S4 = 4 * S6)
  (h3: S7 = 2 * S4) :
  S6 + S4 + S7 = 520 :=
by
  sorry

end NUMINAMATH_GPT_ms_cole_students_l1247_124787


namespace NUMINAMATH_GPT_circle_area_with_diameter_CD_l1247_124712

noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem circle_area_with_diameter_CD (C D E : ℝ × ℝ)
  (hC : C = (-1, 2)) (hD : D = (5, -6)) (hE : E = (2, -2))
  (hE_midpoint : E = ((C.1 + D.1) / 2, (C.2 + D.2) / 2)) :
  ∃ (A : ℝ), A = 25 * Real.pi :=
by
  -- Define the coordinates of points C and D
  let Cx := -1
  let Cy := 2
  let Dx := 5
  let Dy := -6

  -- Calculate the distance (diameter) between C and D
  let diameter := distance Cx Cy Dx Dy

  -- Calculate the radius of the circle
  let radius := diameter / 2

  -- Calculate the area of the circle
  let area := Real.pi * radius^2

  -- Prove the area is 25π
  use area
  sorry

end NUMINAMATH_GPT_circle_area_with_diameter_CD_l1247_124712


namespace NUMINAMATH_GPT_non_neg_int_solutions_l1247_124799

def operation (a b : ℝ) : ℝ := a * (a - b) + 1

theorem non_neg_int_solutions (x : ℕ) :
  2 * (2 - x) + 1 ≥ 3 ↔ x = 0 ∨ x = 1 := by
  sorry

end NUMINAMATH_GPT_non_neg_int_solutions_l1247_124799


namespace NUMINAMATH_GPT_find_dallas_age_l1247_124724

variable (Dallas_last_year Darcy_last_year Dexter_age Darcy_this_year Derek this_year_age : ℕ)

-- Conditions
axiom cond1 : Dallas_last_year = 3 * Darcy_last_year
axiom cond2 : Darcy_this_year = 2 * Dexter_age
axiom cond3 : Dexter_age = 8
axiom cond4 : Derek = this_year_age + 4

-- Theorem: Proving Dallas's current age
theorem find_dallas_age (Dallas_last_year : ℕ)
  (H1 : Dallas_last_year = 3 * (Darcy_this_year - 1))
  (H2 : Darcy_this_year = 2 * Dexter_age)
  (H3 : Dexter_age = 8)
  (H4 : Derek = (Dallas_last_year + 1) + 4) :
  Dallas_last_year + 1 = 46 :=
by
  sorry

end NUMINAMATH_GPT_find_dallas_age_l1247_124724


namespace NUMINAMATH_GPT_first_nonzero_digit_one_over_137_l1247_124763

noncomputable def first_nonzero_digit_right_of_decimal (n : ℚ) : ℕ := sorry

theorem first_nonzero_digit_one_over_137 : first_nonzero_digit_right_of_decimal (1 / 137) = 7 := sorry

end NUMINAMATH_GPT_first_nonzero_digit_one_over_137_l1247_124763


namespace NUMINAMATH_GPT_root_value_l1247_124739

theorem root_value (a : ℝ) (h: 3 * a^2 - 4 * a + 1 = 0) : 6 * a^2 - 8 * a + 5 = 3 := 
by 
  sorry

end NUMINAMATH_GPT_root_value_l1247_124739


namespace NUMINAMATH_GPT_first_day_of_month_l1247_124759

theorem first_day_of_month 
  (d_24: ℕ) (mod_7: d_24 % 7 = 6) : 
  (d_24 - 23) % 7 = 4 :=
by 
  -- denotes the 24th day is a Saturday (Saturday is the 6th day in a 0-6 index)
  -- hence mod_7: d_24 % 7 = 6 means d_24 falls on a Saturday
  sorry

end NUMINAMATH_GPT_first_day_of_month_l1247_124759


namespace NUMINAMATH_GPT_cost_of_corn_per_acre_l1247_124723

def TotalLand : ℕ := 4500
def CostWheat : ℕ := 35
def Capital : ℕ := 165200
def LandWheat : ℕ := 3400
def LandCorn := TotalLand - LandWheat

theorem cost_of_corn_per_acre :
  ∃ C : ℕ, (Capital = (C * LandCorn) + (CostWheat * LandWheat)) ∧ C = 42 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_corn_per_acre_l1247_124723


namespace NUMINAMATH_GPT_intervals_of_monotonicity_max_min_on_interval_l1247_124771

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x

theorem intervals_of_monotonicity :
  (∀ x y : ℝ, x ∈ (Set.Iio (-1) ∪ Set.Ioi (1)) → y ∈ (Set.Iio (-1) ∪ Set.Ioi (1)) → x < y → f x < f y) ∧
  (∀ x y : ℝ, x ∈ (Set.Ioo (-1) 1) → y ∈ (Set.Ioo (-1) 1) → x < y → f x > f y) :=
by
  sorry

theorem max_min_on_interval :
  (∀ x : ℝ, x ∈ Set.Icc (-3) 2 → f x ≤ 2) ∧
  (∀ x : ℝ, x ∈ Set.Icc (-3) 2 → -18 ≤ f x) ∧
  ((∃ x₁ : ℝ, x₁ ∈ Set.Icc (-3) 2 ∧ f x₁ = 2) ∧ (∃ x₂ : ℝ, x₂ ∈ Set.Icc (-3) 2 ∧ f x₂ = -18)) :=
by
  sorry

end NUMINAMATH_GPT_intervals_of_monotonicity_max_min_on_interval_l1247_124771


namespace NUMINAMATH_GPT_max_single_player_salary_l1247_124732

theorem max_single_player_salary
    (num_players : ℕ) (min_salary : ℕ) (total_salary_cap : ℕ)
    (num_player_min_salary : ℕ) (max_salary : ℕ)
    (h1 : num_players = 18)
    (h2 : min_salary = 20000)
    (h3 : total_salary_cap = 600000)
    (h4 : num_player_min_salary = 17)
    (h5 : num_players = num_player_min_salary + 1)
    (h6 : total_salary_cap = num_player_min_salary * min_salary + max_salary) :
    max_salary = 260000 :=
by
  sorry

end NUMINAMATH_GPT_max_single_player_salary_l1247_124732


namespace NUMINAMATH_GPT_adam_coin_collection_value_l1247_124792

-- Definitions related to the problem conditions
def value_per_first_type_coin := 15 / 5
def value_per_second_type_coin := 18 / 6

def total_value_first_type (num_first_type_coins : ℕ) := num_first_type_coins * value_per_first_type_coin
def total_value_second_type (num_second_type_coins : ℕ) := num_second_type_coins * value_per_second_type_coin

-- The main theorem, stating that the total collection value is 90 dollars given the conditions
theorem adam_coin_collection_value :
  total_value_first_type 18 + total_value_second_type 12 = 90 := 
sorry

end NUMINAMATH_GPT_adam_coin_collection_value_l1247_124792


namespace NUMINAMATH_GPT_quadratic_function_a_equals_one_l1247_124705

noncomputable def quadratic_function (a b c : ℝ) (x : ℝ) : ℝ :=
  a * x^2 + b * x + c

theorem quadratic_function_a_equals_one
  (a b c : ℝ)
  (h1 : 1 < x)
  (h2 : x < c)
  (h_neg : ∀ x, 1 < x → x < c → quadratic_function a b c x < 0):
  a = 1 := by
  sorry

end NUMINAMATH_GPT_quadratic_function_a_equals_one_l1247_124705


namespace NUMINAMATH_GPT_solve_for_x_l1247_124781

theorem solve_for_x : ∃ (x : ℝ), (x - 5) ^ 2 = (1 / 16)⁻¹ ∧ (x = 9 ∨ x = 1) :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l1247_124781


namespace NUMINAMATH_GPT_max_range_of_temperatures_l1247_124769

theorem max_range_of_temperatures (avg_temp : ℝ) (low_temp : ℝ) (days : ℕ) (total_temp: ℝ) (high_temp : ℝ) 
  (h1 : avg_temp = 60) (h2 : low_temp = 50) (h3 : days = 5) (h4 : total_temp = avg_temp * days) 
  (h5 : total_temp = 300) (h6 : 4 * low_temp + high_temp = total_temp) : 
  high_temp - low_temp = 50 := 
by
  sorry

end NUMINAMATH_GPT_max_range_of_temperatures_l1247_124769


namespace NUMINAMATH_GPT_hyperbola_foci_distance_l1247_124714

-- Definitions based on the problem conditions
def hyperbola (x y : ℝ) : Prop := x^2 - (y^2 / 9) = 1

def foci_distance (PF1 : ℝ) : Prop := PF1 = 5

-- Main theorem stating the problem and expected outcome
theorem hyperbola_foci_distance (x y PF2 : ℝ) 
  (P_on_hyperbola : hyperbola x y) 
  (PF1_dist : foci_distance (dist (x, y) (some_focal_point_x1, 0))) :
  dist (x, y) (some_focal_point_x2, 0) = 7 ∨ dist (x, y) (some_focal_point_x2, 0) = 3 :=
sorry

end NUMINAMATH_GPT_hyperbola_foci_distance_l1247_124714


namespace NUMINAMATH_GPT_a4_binomial_coefficient_l1247_124725

theorem a4_binomial_coefficient :
  ∀ (a_n a_1 a_2 a_3 a_4 a_5 : ℝ) (x : ℝ),
  (x^5 = a_n + a_1 * (x - 1) + a_2 * (x - 1)^2 + a_3 * (x - 1)^3 + a_4 * (x - 1)^4 + a_5 * (x - 1)^5) →
  (x^5 = (1 + (x - 1))^5) →
  a_4 = 5 :=
by
  intros a_n a_1 a_2 a_3 a_4 a_5 x hx1 hx2
  sorry

end NUMINAMATH_GPT_a4_binomial_coefficient_l1247_124725


namespace NUMINAMATH_GPT_nth_row_equation_l1247_124727

theorem nth_row_equation (n : ℕ) : 2 * n + 1 = (n + 1) ^ 2 - n ^ 2 := 
sorry

end NUMINAMATH_GPT_nth_row_equation_l1247_124727


namespace NUMINAMATH_GPT_least_possible_BC_l1247_124718

-- Define given lengths
def AB := 7 -- cm
def AC := 18 -- cm
def DC := 10 -- cm
def BD := 25 -- cm

-- Define the proof statement
theorem least_possible_BC : 
  ∃ (BC : ℕ), (BC > AC - AB) ∧ (BC > BD - DC) ∧ BC = 16 := by
  sorry

end NUMINAMATH_GPT_least_possible_BC_l1247_124718


namespace NUMINAMATH_GPT_smallest_positive_integer_for_terminating_decimal_l1247_124721

theorem smallest_positive_integer_for_terminating_decimal: ∃ n: ℕ, (n > 0) ∧ (∀ p : ℕ, (p ∣ (n + 150)) → (p=1 ∨ p=2 ∨ p=4 ∨ p=5 ∨ p=8 ∨ p=10 ∨ p=16 ∨ p=20 ∨ p=25 ∨ p=32 ∨ p=40 ∨ p=50 ∨ p=64 ∨ p=80 ∨ p=100 ∨ p=125 ∨ p=128 ∨ p=160)) ∧ n = 10 :=
by
  sorry

end NUMINAMATH_GPT_smallest_positive_integer_for_terminating_decimal_l1247_124721


namespace NUMINAMATH_GPT_xyz_mod_3_l1247_124794

theorem xyz_mod_3 {x y z : ℕ} (hx : x = 3) (hy : y = 3) (hz : z = 2) : 
  (x^2 + y^2 + z^2) % 3 = 1 := by
  sorry

end NUMINAMATH_GPT_xyz_mod_3_l1247_124794


namespace NUMINAMATH_GPT_suffering_correctness_l1247_124780

noncomputable def expected_total_suffering (n m : ℕ) : ℕ :=
  if n = 8 ∧ m = 256 then (2^135 - 2^128 + 1) / (2^119 * 129) else 0

theorem suffering_correctness :
  expected_total_suffering 8 256 = (2^135 - 2^128 + 1) / (2^119 * 129) :=
sorry

end NUMINAMATH_GPT_suffering_correctness_l1247_124780


namespace NUMINAMATH_GPT_f_at_8_5_l1247_124793

def f : ℝ → ℝ := sorry

axiom even_function : ∀ x : ℝ, f x = f (-x)
axiom odd_function_shifted : ∀ x : ℝ, f (x - 1) = -f (1 - x)
axiom f_half : f 0.5 = 9

theorem f_at_8_5 : f 8.5 = 9 := by
  sorry

end NUMINAMATH_GPT_f_at_8_5_l1247_124793


namespace NUMINAMATH_GPT_find_c_l1247_124702

theorem find_c (x c : ℝ) (h₁ : 3 * x + 6 = 0) (h₂ : c * x - 15 = -3) : c = -6 := 
by
  -- sorry is used here as we are not required to provide the proof steps
  sorry

end NUMINAMATH_GPT_find_c_l1247_124702


namespace NUMINAMATH_GPT_square_of_1017_l1247_124798

theorem square_of_1017 : 1017^2 = 1034289 :=
by
  sorry

end NUMINAMATH_GPT_square_of_1017_l1247_124798


namespace NUMINAMATH_GPT_nathalie_total_coins_l1247_124735

theorem nathalie_total_coins
  (quarters dimes nickels : ℕ)
  (ratio_condition : quarters = 9 * nickels ∧ dimes = 3 * nickels)
  (value_condition : 25 * quarters + 10 * dimes + 5 * nickels = 1820) :
  quarters + dimes + nickels = 91 :=
by
  sorry

end NUMINAMATH_GPT_nathalie_total_coins_l1247_124735


namespace NUMINAMATH_GPT_problem1_problem2_l1247_124733

theorem problem1 : -7 + 13 - 6 + 20 = 20 := 
by
  sorry

theorem problem2 : -2^3 + (2 - 3) - 2 * (-1)^2023 = -7 := 
by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l1247_124733


namespace NUMINAMATH_GPT_value_of_m_l1247_124736

theorem value_of_m
  (x y m : ℝ)
  (h1 : 2 * x + 3 * y = 4)
  (h2 : 3 * x + 2 * y = 2 * m - 3)
  (h3 : x + y = -3/5) :
  m = -2 :=
sorry

end NUMINAMATH_GPT_value_of_m_l1247_124736


namespace NUMINAMATH_GPT_jaydee_typing_speed_l1247_124770

theorem jaydee_typing_speed (hours : ℕ) (total_words : ℕ) (minutes_per_hour : ℕ := 60) 
  (h1 : hours = 2) (h2 : total_words = 4560) : (total_words / (hours * minutes_per_hour) = 38) :=
by
  sorry

end NUMINAMATH_GPT_jaydee_typing_speed_l1247_124770


namespace NUMINAMATH_GPT_remainder_abc_mod_5_l1247_124749

theorem remainder_abc_mod_5
  (a b c : ℕ)
  (h₀ : a < 5)
  (h₁ : b < 5)
  (h₂ : c < 5)
  (h₃ : (a + 2 * b + 3 * c) % 5 = 0)
  (h₄ : (2 * a + 3 * b + c) % 5 = 2)
  (h₅ : (3 * a + b + 2 * c) % 5 = 3) :
  (a * b * c) % 5 = 3 :=
by
  sorry

end NUMINAMATH_GPT_remainder_abc_mod_5_l1247_124749


namespace NUMINAMATH_GPT_sweets_remainder_l1247_124748

theorem sweets_remainder (m : ℕ) (h : m % 7 = 6) : (4 * m) % 7 = 3 :=
by
  sorry

end NUMINAMATH_GPT_sweets_remainder_l1247_124748


namespace NUMINAMATH_GPT_difference_between_two_greatest_values_l1247_124783

-- Definition of the variables and conditions
def is_digit (n : ℕ) : Prop := n ≥ 0 ∧ n < 10

variables (a b c x : ℕ)

def conditions (a b c : ℕ) := is_digit a ∧ is_digit b ∧ is_digit c ∧ 2 * a = b ∧ b = 4 * c ∧ a > 0

-- Definition of x as a 3-digit number given a, b, and c
def three_digit_number (a b c : ℕ) : ℕ := 100 * a + 10 * b + c

def smallest_x : ℕ := three_digit_number 2 4 1
def largest_x : ℕ := three_digit_number 4 8 2

def difference_two_greatest_values (a b c : ℕ) : ℕ := largest_x - smallest_x

-- The proof statement
theorem difference_between_two_greatest_values (a b c : ℕ) (h : conditions a b c) : 
  ∀ x1 x2 : ℕ, 
    three_digit_number 2 4 1 = x1 →
    three_digit_number 4 8 2 = x2 →
    difference_two_greatest_values a b c = 241 :=
by
  sorry

end NUMINAMATH_GPT_difference_between_two_greatest_values_l1247_124783


namespace NUMINAMATH_GPT_hyperbola_center_l1247_124730

theorem hyperbola_center :
  ∃ (center : ℝ × ℝ), center = (2.5, 4) ∧
    (∀ x y : ℝ, 9 * x^2 - 45 * x - 16 * y^2 + 128 * y + 207 = 0 ↔ 
      (1/1503) * (36 * (x - 2.5)^2 - 64 * (y - 4)^2) = 1) :=
sorry

end NUMINAMATH_GPT_hyperbola_center_l1247_124730


namespace NUMINAMATH_GPT_no_real_solutions_l1247_124776

noncomputable def original_eq (x : ℝ) : Prop := (x^2 + x + 1) / (x + 1) = x^2 + 5 * x + 6

theorem no_real_solutions (x : ℝ) : ¬ original_eq x :=
by
  sorry

end NUMINAMATH_GPT_no_real_solutions_l1247_124776


namespace NUMINAMATH_GPT_number_of_people_l1247_124757

-- Define the given constants
def total_cookies := 35
def cookies_per_person := 7

-- Goal: Prove that the number of people equal to 5
theorem number_of_people : total_cookies / cookies_per_person = 5 :=
by
  sorry

end NUMINAMATH_GPT_number_of_people_l1247_124757


namespace NUMINAMATH_GPT_find_integer_solutions_xy_l1247_124737

theorem find_integer_solutions_xy :
  ∀ (x y : ℕ), (x * y = x + y + 3) → (x, y) = (2, 5) ∨ (x, y) = (5, 2) ∨ (x, y) = (3, 3) := by
  intros x y h
  sorry

end NUMINAMATH_GPT_find_integer_solutions_xy_l1247_124737


namespace NUMINAMATH_GPT_product_of_fractions_l1247_124756

theorem product_of_fractions : (2 / 5) * (3 / 4) = 3 / 10 := 
  sorry

end NUMINAMATH_GPT_product_of_fractions_l1247_124756


namespace NUMINAMATH_GPT_arithmetic_seq_a7_l1247_124753

theorem arithmetic_seq_a7 (a : ℕ → ℤ) (d : ℤ) 
  (h_arith : ∀ n, a (n + 1) = a n + d)
  (h_a3 : a 3 = 50) 
  (h_a5 : a 5 = 30) : 
  a 7 = 10 := 
by
  sorry

end NUMINAMATH_GPT_arithmetic_seq_a7_l1247_124753


namespace NUMINAMATH_GPT_students_count_l1247_124719

theorem students_count (x : ℕ) (h1 : x / 2 + x / 4 + x / 7 + 3 = x) : x = 28 :=
  sorry

end NUMINAMATH_GPT_students_count_l1247_124719


namespace NUMINAMATH_GPT_total_students_l1247_124795

-- Definitions from the conditions
def ratio_boys_to_girls (B G : ℕ) : Prop := B / G = 1 / 2
def girls_count := 60

-- The main statement to prove
theorem total_students (B G : ℕ) (h1 : ratio_boys_to_girls B G) (h2 : G = girls_count) : B + G = 90 := sorry

end NUMINAMATH_GPT_total_students_l1247_124795


namespace NUMINAMATH_GPT_john_total_amount_to_pay_l1247_124711

-- Define constants for the problem
def total_cost : ℝ := 6650
def rebate_percentage : ℝ := 0.06
def sales_tax_percentage : ℝ := 0.10

-- The main theorem to prove the final amount John needs to pay
theorem john_total_amount_to_pay : total_cost * (1 - rebate_percentage) * (1 + sales_tax_percentage) = 6876.10 := by
  sorry    -- Proof skipped

end NUMINAMATH_GPT_john_total_amount_to_pay_l1247_124711


namespace NUMINAMATH_GPT_sunny_bakes_initial_cakes_l1247_124754

theorem sunny_bakes_initial_cakes (cakes_after_giving_away : ℕ) (total_candles : ℕ) (candles_per_cake : ℕ) (given_away_cakes : ℕ) (initial_cakes : ℕ) :
  cakes_after_giving_away = total_candles / candles_per_cake →
  given_away_cakes = 2 →
  total_candles = 36 →
  candles_per_cake = 6 →
  initial_cakes = cakes_after_giving_away + given_away_cakes →
  initial_cakes = 8 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_sunny_bakes_initial_cakes_l1247_124754


namespace NUMINAMATH_GPT_A_cubed_inv_l1247_124750

variable (A : Matrix (Fin 2) (Fin 2) ℝ)

-- Given condition
def A_inv : Matrix (Fin 2) (Fin 2) ℝ := ![![3, 7], ![-2, -4]]

-- Goal to prove
theorem A_cubed_inv :
  (A^3)⁻¹ = ![![11, 17], ![2, 6]] :=
  sorry

end NUMINAMATH_GPT_A_cubed_inv_l1247_124750


namespace NUMINAMATH_GPT_g_range_l1247_124765

noncomputable def g (x y z : ℝ) : ℝ := 
  (x^2 / (x^2 + y^2)) + (y^2 / (y^2 + z^2)) + (z^2 / (z^2 + x^2))

theorem g_range (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  3 / 2 ≤ g x y z ∧ g x y z ≤ 2 :=
sorry

end NUMINAMATH_GPT_g_range_l1247_124765


namespace NUMINAMATH_GPT_papaya_tree_height_after_5_years_l1247_124779

def first_year_growth := 2
def second_year_growth := first_year_growth + (first_year_growth / 2)
def third_year_growth := second_year_growth + (second_year_growth / 2)
def fourth_year_growth := third_year_growth * 2
def fifth_year_growth := fourth_year_growth / 2

theorem papaya_tree_height_after_5_years : 
  first_year_growth + second_year_growth + third_year_growth + fourth_year_growth + fifth_year_growth = 23 :=
by
  sorry

end NUMINAMATH_GPT_papaya_tree_height_after_5_years_l1247_124779


namespace NUMINAMATH_GPT_find_a_plus_d_l1247_124715

noncomputable def f (a b c d x : ℚ) : ℚ := (a * x + b) / (c * x + d)

theorem find_a_plus_d (a b c d : ℚ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) (h₂ : c ≠ 0) (h₃ : d ≠ 0)
  (h₄ : ∀ x : ℚ, f a b c d (f a b c d x) = x) :
  a + d = 0 := by
  sorry

end NUMINAMATH_GPT_find_a_plus_d_l1247_124715


namespace NUMINAMATH_GPT_find_vasya_floor_l1247_124738

theorem find_vasya_floor (steps_petya: ℕ) (steps_vasya: ℕ) (petya_floors: ℕ) (steps_per_floor: ℝ):
  steps_petya = 36 → petya_floors = 2 → steps_vasya = 72 → 
  steps_per_floor = steps_petya / petya_floors → 
  (1 + (steps_vasya / steps_per_floor)) = 5 := by 
  intros h1 h2 h3 h4 
  sorry

end NUMINAMATH_GPT_find_vasya_floor_l1247_124738


namespace NUMINAMATH_GPT_unique_positive_integer_solution_l1247_124777

-- Definitions of the given points
def P1 : ℚ × ℚ := (4, 11)
def P2 : ℚ × ℚ := (16, 1)

-- Definition for the line equation in standard form
def line_equation (x y : ℤ) : Prop := 5 * x + 6 * y = 43

-- Proof for the existence of only one solution with positive integer coordinates
theorem unique_positive_integer_solution :
  ∃ P : ℤ × ℤ, P.1 > 0 ∧ P.2 > 0 ∧ line_equation P.1 P.2 ∧ (∀ Q : ℤ × ℤ, line_equation Q.1 Q.2 → Q.1 > 0 ∧ Q.2 > 0 → Q = (5, 3)) :=
by 
  sorry

end NUMINAMATH_GPT_unique_positive_integer_solution_l1247_124777


namespace NUMINAMATH_GPT_solution_set_inequality_l1247_124767

theorem solution_set_inequality (x : ℝ) : (x + 3) / (x - 1) > 0 ↔ x < -3 ∨ x > 1 :=
sorry

end NUMINAMATH_GPT_solution_set_inequality_l1247_124767


namespace NUMINAMATH_GPT_gloria_coins_l1247_124742

theorem gloria_coins (qd qda qdc : ℕ) (h1 : qdc = 350) (h2 : qda = qdc / 5) (h3 : qd = qda - (2 * qda / 5)) :
  qd + qdc = 392 :=
by sorry

end NUMINAMATH_GPT_gloria_coins_l1247_124742


namespace NUMINAMATH_GPT_sum_local_values_2345_l1247_124797

theorem sum_local_values_2345 : 
  let n := 2345
  let digit_2_value := 2000
  let digit_3_value := 300
  let digit_4_value := 40
  let digit_5_value := 5
  digit_2_value + digit_3_value + digit_4_value + digit_5_value = n := 
by
  sorry

end NUMINAMATH_GPT_sum_local_values_2345_l1247_124797


namespace NUMINAMATH_GPT_numbers_starting_with_6_div_by_25_no_numbers_divisible_by_35_after_first_digit_removed_l1247_124752

-- Definitions based on conditions
def starts_with_six (x : ℕ) : Prop :=
  ∃ n y, x = 6 * 10^n + y

def is_divisible_by_25 (y : ℕ) : Prop :=
  y % 25 = 0

def is_divisible_by_35 (y : ℕ) : Prop :=
  y % 35 = 0

-- Main theorem statements
theorem numbers_starting_with_6_div_by_25:
  ∀ x, starts_with_six x → ∃ k, x = 625 * 10^k :=
by
  sorry

theorem no_numbers_divisible_by_35_after_first_digit_removed:
  ∀ a x, a ≠ 0 → 
  ∃ n, x = a * 10^n + y →
  ¬(is_divisible_by_35 y) :=
by
  sorry

end NUMINAMATH_GPT_numbers_starting_with_6_div_by_25_no_numbers_divisible_by_35_after_first_digit_removed_l1247_124752


namespace NUMINAMATH_GPT_no_perfect_squares_in_sequence_l1247_124740

def tau (a : ℕ) : ℕ := sorry -- Define tau function here

def a_seq (k : ℕ) (n : ℕ) : ℕ :=
  if n = 0 then k else tau (a_seq k (n-1))

theorem no_perfect_squares_in_sequence (k : ℕ) (hk : Prime k) :
  ∀ n : ℕ, ∃ m : ℕ, a_seq k n = m * m → False :=
sorry

end NUMINAMATH_GPT_no_perfect_squares_in_sequence_l1247_124740


namespace NUMINAMATH_GPT_general_formula_a_sum_b_condition_l1247_124773

noncomputable def sequence_a (n : ℕ) : ℕ := sorry
noncomputable def sum_a (n : ℕ) : ℕ := sorry

-- Conditions
def a_2_condition : Prop := sequence_a 2 = 4
def sum_condition (n : ℕ) : Prop := 2 * sum_a n = n * sequence_a n + n

-- General formula for the n-th term of the sequence a_n
theorem general_formula_a : 
  (∀ n, sequence_a n = 3 * n - 2) ↔
  (a_2_condition ∧ ∀ n, sum_condition n) :=
sorry

noncomputable def sequence_c (n : ℕ) : ℕ := sorry
noncomputable def sequence_b (n : ℕ) : ℕ := sorry
noncomputable def sum_b (n : ℕ) : ℝ := sorry

-- Geometric sequence condition
def geometric_sequence_condition : Prop :=
  ∀ n, sequence_c n = 4^n

-- Condition for a_n = b_n * c_n
def a_b_c_relation (n : ℕ) : Prop := 
  sequence_a n = sequence_b n * sequence_c n

-- Sum condition T_n < 2/3
theorem sum_b_condition :
  (∀ n, a_b_c_relation n) ∧ geometric_sequence_condition →
  (∀ n, sum_b n < 2 / 3) :=
sorry

end NUMINAMATH_GPT_general_formula_a_sum_b_condition_l1247_124773


namespace NUMINAMATH_GPT_least_number_to_add_l1247_124791

theorem least_number_to_add (n : ℕ) (m : ℕ) : (1156 + 19) % 25 = 0 :=
by
  sorry

end NUMINAMATH_GPT_least_number_to_add_l1247_124791


namespace NUMINAMATH_GPT_length_of_body_diagonal_l1247_124728

theorem length_of_body_diagonal (a b c : ℝ) 
  (h1 : 2 * (a * b + b * c + a * c) = 11)
  (h2 : 4 * (a + b + c) = 24) :
  (a^2 + b^2 + c^2).sqrt = 5 :=
by {
  -- proof to be filled
  sorry
}

end NUMINAMATH_GPT_length_of_body_diagonal_l1247_124728


namespace NUMINAMATH_GPT_problem_statement_l1247_124745

theorem problem_statement (a x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12 x13 : ℤ) 
  (h1 : a = (1 + x1) * (1 + x2) * (1 + x3) * (1 + x4) * (1 + x5) * (1 + x6) * (1 + x7) * (1 + x8) * (1 + x9) * (1 + x10) * (1 + x11) * (1 + x12) * (1 + x13))
  (h2 : a = (1 - x1) * (1 - x2) * (1 - x3) * (1 - x4) * (1 - x5) * (1 - x6) * (1 - x7) * (1 - x8) * (1 - x9) * (1 - x10) * (1 - x11) * (1 - x12) * (1 - x13)) : 
  a * x1 * x2 * x3 * x4 * x5 * x6 * x7 * x8 * x9 * x10 * x11 * x12 * x13 = 0 := 
sorry

end NUMINAMATH_GPT_problem_statement_l1247_124745


namespace NUMINAMATH_GPT_sequence_value_a10_l1247_124782

theorem sequence_value_a10 (a : ℕ → ℕ) (h₁ : a 1 = 1) (h₂ : ∀ n, a (n + 1) = a n + 2^n) : a 10 = 1023 := by
  sorry

end NUMINAMATH_GPT_sequence_value_a10_l1247_124782


namespace NUMINAMATH_GPT_final_price_correct_l1247_124764

noncomputable def price_cucumbers : ℝ := 5
noncomputable def price_tomatoes : ℝ := price_cucumbers - 0.20 * price_cucumbers
noncomputable def total_cost_before_discount : ℝ := 2 * price_tomatoes + 3 * price_cucumbers
noncomputable def discount : ℝ := 0.10 * total_cost_before_discount
noncomputable def final_price : ℝ := total_cost_before_discount - discount

theorem final_price_correct : final_price = 20.70 := by
  sorry

end NUMINAMATH_GPT_final_price_correct_l1247_124764


namespace NUMINAMATH_GPT_frequency_of_3rd_group_l1247_124785

theorem frequency_of_3rd_group (m : ℕ) (h_m : m ≥ 3) (x : ℝ) (h_area_relation : ∀ k, k ≠ 3 → 4 * x = k):
  100 * x = 20 :=
by
  sorry

end NUMINAMATH_GPT_frequency_of_3rd_group_l1247_124785


namespace NUMINAMATH_GPT_boy_needs_to_sell_75_oranges_to_make_150c_profit_l1247_124758

-- Definitions based on the conditions
def cost_per_orange : ℕ := 12 / 4
def sell_price_per_orange : ℕ := 30 / 6
def profit_per_orange : ℕ := sell_price_per_orange - cost_per_orange

-- Problem declaration
theorem boy_needs_to_sell_75_oranges_to_make_150c_profit : 
  (150 / profit_per_orange) = 75 :=
by
  -- Proof will be added here
  sorry

end NUMINAMATH_GPT_boy_needs_to_sell_75_oranges_to_make_150c_profit_l1247_124758


namespace NUMINAMATH_GPT_praveen_initial_investment_l1247_124760

theorem praveen_initial_investment
  (H : ℝ) (P : ℝ)
  (h_H : H = 9000.000000000002)
  (h_profit_ratio : (P * 12) / (H * 7) = 2 / 3) :
  P = 3500 := by
  sorry

end NUMINAMATH_GPT_praveen_initial_investment_l1247_124760


namespace NUMINAMATH_GPT_snow_at_least_once_three_days_l1247_124713

-- Define the probability of snow on a given day
def prob_snow : ℚ := 2 / 3

-- Define the event that it snows at least once in three days
def prob_snow_at_least_once_in_three_days : ℚ :=
  1 - (1 - prob_snow)^3

-- State the theorem
theorem snow_at_least_once_three_days : prob_snow_at_least_once_in_three_days = 26 / 27 :=
by
  sorry

end NUMINAMATH_GPT_snow_at_least_once_three_days_l1247_124713


namespace NUMINAMATH_GPT_problem1_problem2_l1247_124789

theorem problem1 (a b : ℝ) (h1 : a = 3) :
  (∀ x : ℝ, 2 * x ^ 2 + (2 - a) * x - a > 0 ↔ x < -1 ∨ x > 3 / 2) :=
by
  sorry

theorem problem2 (a b : ℝ) (h1 : a = 3) :
  (∀ x : ℝ, 3 * x ^ 2 + b * x + 3 ≥ 0) ↔ (-6 ≤ b ∧ b ≤ 6) :=
by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l1247_124789


namespace NUMINAMATH_GPT_complete_square_form_l1247_124734

theorem complete_square_form (x : ℝ) (a : ℝ) 
  (h : x^2 - 2 * x - 4 = 0) : (x - 1)^2 = a ↔ a = 5 :=
by
  sorry

end NUMINAMATH_GPT_complete_square_form_l1247_124734


namespace NUMINAMATH_GPT_platform_length_l1247_124741

theorem platform_length 
  (train_length : ℝ) (train_speed_kmph : ℝ) (time_s : ℝ) (platform_length : ℝ)
  (H1 : train_length = 360) 
  (H2 : train_speed_kmph = 45) 
  (H3 : time_s = 40)
  (H4 : platform_length = (train_speed_kmph * 1000 / 3600 * time_s) - train_length ) :
  platform_length = 140 :=
by {
 sorry
}

end NUMINAMATH_GPT_platform_length_l1247_124741


namespace NUMINAMATH_GPT_quadratic_residue_iff_l1247_124726

open Nat

theorem quadratic_residue_iff (p : ℕ) [Fact (Nat.Prime p)] (hp : p % 2 = 1) (n : ℤ) (hn : n % p ≠ 0) :
  (∃ a : ℤ, (a^2) % p = n % p) ↔ (n ^ ((p - 1) / 2)) % p = 1 :=
sorry

end NUMINAMATH_GPT_quadratic_residue_iff_l1247_124726


namespace NUMINAMATH_GPT_blue_pill_cost_l1247_124796

theorem blue_pill_cost :
  ∀ (cost_yellow cost_blue : ℝ) (days : ℕ) (total_cost : ℝ),
    (days = 21) →
    (total_cost = 882) →
    (cost_blue = cost_yellow + 3) →
    (total_cost = days * (cost_blue + cost_yellow)) →
    cost_blue = 22.50 :=
by sorry

end NUMINAMATH_GPT_blue_pill_cost_l1247_124796


namespace NUMINAMATH_GPT_female_officers_count_l1247_124755

theorem female_officers_count (total_officers_on_duty : ℕ) 
  (percent_female_on_duty : ℝ) 
  (female_officers_on_duty : ℕ) 
  (half_of_total_on_duty_is_female : total_officers_on_duty / 2 = female_officers_on_duty) 
  (percent_condition : percent_female_on_duty * (total_officers_on_duty / 2) = female_officers_on_duty) :
  total_officers_on_duty = 250 :=
by
  sorry

end NUMINAMATH_GPT_female_officers_count_l1247_124755


namespace NUMINAMATH_GPT_age_of_boy_not_included_l1247_124772

theorem age_of_boy_not_included (average_age_11_boys : ℕ) (average_age_first_6 : ℕ) (average_age_last_6 : ℕ) 
(first_6_sum : ℕ) (last_6_sum : ℕ) (total_sum : ℕ) (X : ℕ):
  average_age_11_boys = 50 ∧ average_age_first_6 = 49 ∧ average_age_last_6 = 52 ∧ 
  first_6_sum = 6 * average_age_first_6 ∧ last_6_sum = 6 * average_age_last_6 ∧ 
  total_sum = 11 * average_age_11_boys ∧ first_6_sum + last_6_sum - X = total_sum →
  X = 56 :=
by
  sorry

end NUMINAMATH_GPT_age_of_boy_not_included_l1247_124772


namespace NUMINAMATH_GPT_find_f_x_l1247_124716

theorem find_f_x (f : ℝ → ℝ) (h : ∀ x : ℝ, f x + 1 = 3 * x + 2) : ∀ x : ℝ, f x = 3 * x - 1 :=
by
  sorry

end NUMINAMATH_GPT_find_f_x_l1247_124716
