import Mathlib

namespace NUMINAMATH_GPT_inequality_solution_l867_86774

theorem inequality_solution (x : ℝ) :
  (7 : ℝ) / 30 + abs (x - 7 / 60) < 11 / 20 ↔ -1 / 5 < x ∧ x < 13 / 30 :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_l867_86774


namespace NUMINAMATH_GPT_reporters_percentage_l867_86746

theorem reporters_percentage (total_reporters : ℕ) (local_politics_percentage : ℝ) (non_politics_percentage : ℝ) :
  local_politics_percentage = 28 → non_politics_percentage = 60 → 
  let political_reporters := total_reporters * ((100 - non_politics_percentage) / 100)
  let local_political_reporters := total_reporters * (local_politics_percentage / 100)
  let non_local_political_reporters := political_reporters - local_political_reporters
  100 * (non_local_political_reporters / political_reporters) = 30 :=
by
  intros
  let political_reporters := total_reporters * ((100 - non_politics_percentage) / 100)
  let local_political_reporters := total_reporters * (local_politics_percentage / 100)
  let non_local_political_reporters := political_reporters - local_political_reporters
  sorry

end NUMINAMATH_GPT_reporters_percentage_l867_86746


namespace NUMINAMATH_GPT_Eddy_travel_time_l867_86787

theorem Eddy_travel_time (T V_e V_f : ℝ) 
  (dist_AB dist_AC : ℝ) 
  (time_Freddy : ℝ) 
  (speed_ratio : ℝ) 
  (h1 : dist_AB = 600) 
  (h2 : dist_AC = 300) 
  (h3 : time_Freddy = 3) 
  (h4 : speed_ratio = 2)
  (h5 : V_f = dist_AC / time_Freddy)
  (h6 : V_e = speed_ratio * V_f)
  (h7 : T = dist_AB / V_e) :
  T = 3 :=
by
  sorry

end NUMINAMATH_GPT_Eddy_travel_time_l867_86787


namespace NUMINAMATH_GPT_otherWorkStations_accommodate_students_l867_86727

def numTotalStudents := 38
def numStations := 16
def numWorkStationsForTwo := 10
def capacityWorkStationsForTwo := 2

theorem otherWorkStations_accommodate_students : 
  (numTotalStudents - numWorkStationsForTwo * capacityWorkStationsForTwo) = 18 := 
by
  sorry

end NUMINAMATH_GPT_otherWorkStations_accommodate_students_l867_86727


namespace NUMINAMATH_GPT_f_range_and_boundedness_g_odd_and_bounded_g_upper_bound_l867_86728

noncomputable def f (x : ℝ) : ℝ := (1 / 4) ^ x + (1 / 2) ^ x - 1
noncomputable def g (x m : ℝ) : ℝ := (1 - m * 2 ^ x) / (1 + m * 2 ^ x)

theorem f_range_and_boundedness :
  ∀ x : ℝ, x < 0 → 1 < f x ∧ ¬(∃ M : ℝ, ∀ x : ℝ, x < 0 → |f x| ≤ M) :=
by sorry

theorem g_odd_and_bounded (x : ℝ) :
  g x 1 = -g (-x) 1 ∧ |g x 1| < 1 :=
by sorry

theorem g_upper_bound (m : ℝ) (hm : 0 < m ∧ m < 1 / 2) :
  ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → g x m ≤ (1 - m) / (1 + m) :=
by sorry

end NUMINAMATH_GPT_f_range_and_boundedness_g_odd_and_bounded_g_upper_bound_l867_86728


namespace NUMINAMATH_GPT_factorize_expression_l867_86737

theorem factorize_expression (a b : ℝ) : a^2 + a * b = a * (a + b) := 
by
  sorry

end NUMINAMATH_GPT_factorize_expression_l867_86737


namespace NUMINAMATH_GPT_volume_of_rectangular_prism_l867_86718

theorem volume_of_rectangular_prism
  (x y z : ℝ)
  (h1 : x * y = 20)
  (h2 : y * z = 15)
  (h3 : z * x = 12) :
  x * y * z = 60 :=
sorry

end NUMINAMATH_GPT_volume_of_rectangular_prism_l867_86718


namespace NUMINAMATH_GPT_circle_standard_equation_l867_86766

theorem circle_standard_equation (x y : ℝ) (h : (x + 1)^2 + (y - 2)^2 = 4) : 
  (x + 1)^2 + (y - 2)^2 = 4 :=
sorry

end NUMINAMATH_GPT_circle_standard_equation_l867_86766


namespace NUMINAMATH_GPT_range_of_m_l867_86732

theorem range_of_m (m : ℝ) (p : |m + 1| ≤ 2) (q : ¬(m^2 - 4 ≥ 0)) : -2 < m ∧ m ≤ 1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l867_86732


namespace NUMINAMATH_GPT_parents_give_per_year_l867_86789

def Mikail_age (x : ℕ) : Prop :=
  x = 3 * (x - 3)

noncomputable def money_per_year (total_money : ℕ) (age : ℕ) : ℕ :=
  total_money / age

theorem parents_give_per_year 
  (x : ℕ) (hx : Mikail_age x) : 
  money_per_year 45 x = 5 :=
sorry

end NUMINAMATH_GPT_parents_give_per_year_l867_86789


namespace NUMINAMATH_GPT_polynomial_divisibility_l867_86770

theorem polynomial_divisibility (t : ℤ) : 
  (∀ x : ℤ, (5 * x^3 - 15 * x^2 + t * x - 20) ∣ (x - 2)) → (t = 20) → 
  ∀ x : ℤ, (5 * x^3 - 15 * x^2 + 20 * x - 20) ∣ (5 * x^2 + 5 * x + 5) :=
by
  intro h₁ h₂
  sorry

end NUMINAMATH_GPT_polynomial_divisibility_l867_86770


namespace NUMINAMATH_GPT_part1_part2_l867_86716

variables (q x : ℝ)
def f (x : ℝ) (q : ℝ) : ℝ := x^2 - 16*x + q + 3
def g (x : ℝ) (q : ℝ) : ℝ := f x q + 51

theorem part1 (h1 : ∃ x ∈ Set.Icc (-1 : ℝ) 1, f x q = 0):
  (-20 : ℝ) ≤ q ∧ q ≤ 12 := 
  sorry

theorem part2 (h2 : ∀ x ∈ Set.Icc (q : ℝ) 10, g x q ≥ 0) : 
  9 ≤ q ∧ q < 10 := 
  sorry

end NUMINAMATH_GPT_part1_part2_l867_86716


namespace NUMINAMATH_GPT_add_congruence_l867_86763

variable (a b c d m : ℤ)

theorem add_congruence (h₁ : a ≡ b [ZMOD m]) (h₂ : c ≡ d [ZMOD m]) : (a + c) ≡ (b + d) [ZMOD m] :=
sorry

end NUMINAMATH_GPT_add_congruence_l867_86763


namespace NUMINAMATH_GPT_simplify_fraction_eq_l867_86760

theorem simplify_fraction_eq : (180 / 270 : ℚ) = 2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_simplify_fraction_eq_l867_86760


namespace NUMINAMATH_GPT_smallest_number_in_set_l867_86764

open Real

theorem smallest_number_in_set :
  ∀ (a b c d : ℝ), a = -3 → b = 3⁻¹ → c = -abs (-1 / 3) → d = 0 →
    a < b ∧ a < c ∧ a < d :=
by
  intros a b c d ha hb hc hd
  rw [ha, hb, hc, hd]
  sorry

end NUMINAMATH_GPT_smallest_number_in_set_l867_86764


namespace NUMINAMATH_GPT_smallest_number_l867_86757

theorem smallest_number (x y z : ℕ) (h1 : y = 4 * x) (h2 : z = 2 * y) 
(h3 : (x + y + z) / 3 = 78) : x = 18 := 
by 
    sorry

end NUMINAMATH_GPT_smallest_number_l867_86757


namespace NUMINAMATH_GPT_count_of_numbers_with_digit_3_eq_71_l867_86755

-- Define the problem space
def count_numbers_without_digit_3 : ℕ := 729
def total_numbers : ℕ := 800
def count_numbers_with_digit_3 : ℕ := total_numbers - count_numbers_without_digit_3

-- Prove that the count of numbers from 1 to 800 containing at least one digit 3 is 71
theorem count_of_numbers_with_digit_3_eq_71 :
  count_numbers_with_digit_3 = 71 :=
by
  sorry

end NUMINAMATH_GPT_count_of_numbers_with_digit_3_eq_71_l867_86755


namespace NUMINAMATH_GPT_breadth_of_hall_l867_86701

theorem breadth_of_hall (length_hall : ℝ) (stone_length_dm : ℝ) (stone_breadth_dm : ℝ)
    (num_stones : ℕ) (area_stone_m2 : ℝ) (total_area_m2 : ℝ) (breadth_hall : ℝ):
    length_hall = 36 → 
    stone_length_dm = 8 → 
    stone_breadth_dm = 5 → 
    num_stones = 1350 → 
    area_stone_m2 = (stone_length_dm * stone_breadth_dm) / 100 → 
    total_area_m2 = num_stones * area_stone_m2 → 
    breadth_hall = total_area_m2 / length_hall → 
    breadth_hall = 15 :=
by 
  intros h1 h2 h3 h4 h5 h6 h7
  rw [h1, h2, h3, h4] at *
  simp [h5, h6, h7]
  sorry

end NUMINAMATH_GPT_breadth_of_hall_l867_86701


namespace NUMINAMATH_GPT_parabola_standard_eq_line_m_tangent_l867_86743

open Real

variables (p k : ℝ) (x y : ℝ)

-- Definitions based on conditions
def parabola_equation (p : ℝ) : Prop := ∀ x y : ℝ, x^2 = 2 * p * y
def line_m (k : ℝ) : Prop := ∀ x y : ℝ, y = k * x + 6

-- Problem statement
theorem parabola_standard_eq (p : ℝ) (hp : p = 2) :
  parabola_equation p ↔ (∀ x y : ℝ, x^2 = 4 * y) :=
sorry

theorem line_m_tangent (k : ℝ) (x1 x2 : ℝ)
  (hpq : x1 + x2 = 4 * k ∧ x1 * x2 = -24)
  (hk : k = 1/2 ∨ k = -1/2) :
  line_m k ↔ ((k = 1/2 ∧ ∀ x y : ℝ, y = 1/2 * x + 6) ∨ (k = -1/2 ∧ ∀ x y : ℝ, y = -1/2 * x + 6)) :=
sorry

end NUMINAMATH_GPT_parabola_standard_eq_line_m_tangent_l867_86743


namespace NUMINAMATH_GPT_problem_statement_l867_86776

noncomputable def find_pq_sum (XZ YZ : ℕ) (XY_perimeter_ratio : ℕ × ℕ) : ℕ :=
  let XY := Real.sqrt (XZ^2 + YZ^2)
  let ZD := Real.sqrt (XZ * YZ)
  let O_radius := 0.5 * ZD
  let tangent_length := Real.sqrt ((XY / 2)^2 - O_radius^2)
  let perimeter := XY + 2 * tangent_length
  let (p, q) := XY_perimeter_ratio
  p + q

theorem problem_statement :
  find_pq_sum 8 15 (30, 17) = 47 :=
by sorry

end NUMINAMATH_GPT_problem_statement_l867_86776


namespace NUMINAMATH_GPT_prism_pyramid_fusion_l867_86721

theorem prism_pyramid_fusion :
  ∃ (result_faces result_edges result_vertices : ℕ),
    result_faces + result_edges + result_vertices = 28 ∧
    ((result_faces = 8 ∧ result_edges = 13 ∧ result_vertices = 7) ∨
    (result_faces = 7 ∧ result_edges = 12 ∧ result_vertices = 7)) :=
by
  sorry

end NUMINAMATH_GPT_prism_pyramid_fusion_l867_86721


namespace NUMINAMATH_GPT_problem_1_problem_2_l867_86712

noncomputable def f (a b x : ℝ) : ℝ := (a * x + b) / (x - 2)

theorem problem_1 (a b : ℝ) (h1 : f a b 3 - 3 + 12 = 0) (h2 : f a b 4 - 4 + 12 = 0) :
  f a b x = (2 - x) / (x - 2) := sorry

theorem problem_2 (k : ℝ) (h : k > 1) :
  ∀ x, f (-1) 2 x < k ↔ (if 1 < k ∧ k < 2 then (1 < x ∧ x < k) ∨ (2 < x) 
                         else if k = 2 then 1 < x ∧ x ≠ 2 
                         else (1 < x ∧ x < 2) ∨ (k < x)) := sorry

-- Function definition for clarity
noncomputable def f_spec (x : ℝ) : ℝ := (2 - x) / (x - 2)

end NUMINAMATH_GPT_problem_1_problem_2_l867_86712


namespace NUMINAMATH_GPT_least_possible_value_of_smallest_integer_l867_86733

theorem least_possible_value_of_smallest_integer :
  ∀ (A B C D : ℕ), 
    A ≠ B → A ≠ C → A ≠ D → B ≠ C → B ≠ D → C ≠ D →
    (A + B + C + D) / 4 = 68 →
    D = 90 →
    A = 5 :=
by
  intros A B C D h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end NUMINAMATH_GPT_least_possible_value_of_smallest_integer_l867_86733


namespace NUMINAMATH_GPT_weeks_to_buy_bicycle_l867_86793

-- Definitions based on problem conditions
def hourly_wage : Int := 5
def hours_monday : Int := 2
def hours_wednesday : Int := 1
def hours_friday : Int := 3
def weekly_hours : Int := hours_monday + hours_wednesday + hours_friday
def weekly_earnings : Int := weekly_hours * hourly_wage
def bicycle_cost : Int := 180

-- Statement of the theorem to prove
theorem weeks_to_buy_bicycle : ∃ w : Nat, w * weekly_earnings = bicycle_cost :=
by
  -- Since this is a statement only, the proof is omitted
  sorry

end NUMINAMATH_GPT_weeks_to_buy_bicycle_l867_86793


namespace NUMINAMATH_GPT_symmetrical_line_equation_l867_86782

-- Definitions for the conditions
def line_symmetrical (eq1 eq2 : String) : Prop :=
  eq1 = "x - 2y + 3 = 0" ∧ eq2 = "x + 2y + 3 = 0"

-- Prove the statement
theorem symmetrical_line_equation : line_symmetrical "x - 2y + 3 = 0" "x + 2y + 3 = 0" :=
  by
  -- This is just the proof skeleton; the actual proof is not required
  sorry

end NUMINAMATH_GPT_symmetrical_line_equation_l867_86782


namespace NUMINAMATH_GPT_december_fraction_of_yearly_sales_l867_86768

theorem december_fraction_of_yearly_sales (A : ℝ) (h_sales : ∀ (x : ℝ), x = 6 * A) :
    let yearly_sales := 11 * A + 6 * A
    let december_sales := 6 * A
    december_sales / yearly_sales = 6 / 17 := by
  sorry

end NUMINAMATH_GPT_december_fraction_of_yearly_sales_l867_86768


namespace NUMINAMATH_GPT_initial_percentage_of_water_l867_86702

theorem initial_percentage_of_water (C V final_volume : ℝ) (P : ℝ) 
  (hC : C = 80)
  (hV : V = 36)
  (h_final_volume : final_volume = (3/4) * C)
  (h_initial_equation: (P / 100) * C + V = final_volume) : 
  P = 30 :=
by
  sorry

end NUMINAMATH_GPT_initial_percentage_of_water_l867_86702


namespace NUMINAMATH_GPT_domain_of_c_eq_real_l867_86717

theorem domain_of_c_eq_real (m : ℝ) : (∀ x : ℝ, m * x^2 - 3 * x + 2 * m ≠ 0) ↔ (m < -3 * Real.sqrt 2 / 4 ∨ m > 3 * Real.sqrt 2 / 4) :=
by
  sorry

end NUMINAMATH_GPT_domain_of_c_eq_real_l867_86717


namespace NUMINAMATH_GPT_square_traffic_sign_perimeter_l867_86709

-- Define the side length of the square
def side_length : ℕ := 4

-- Define the number of sides of the square
def number_of_sides : ℕ := 4

-- Define the perimeter of the square
def perimeter (l : ℕ) (n : ℕ) : ℕ := l * n

-- The theorem to be proved
theorem square_traffic_sign_perimeter : perimeter side_length number_of_sides = 16 :=
by
  sorry

end NUMINAMATH_GPT_square_traffic_sign_perimeter_l867_86709


namespace NUMINAMATH_GPT_least_number_to_add_l867_86775

theorem least_number_to_add (x : ℕ) : (1053 + x) % 23 = 0 ↔ x = 5 := by
  sorry

end NUMINAMATH_GPT_least_number_to_add_l867_86775


namespace NUMINAMATH_GPT_actual_price_of_food_before_tax_and_tip_l867_86715

theorem actual_price_of_food_before_tax_and_tip 
  (total_paid : ℝ)
  (tip_percentage : ℝ)
  (tax_percentage : ℝ)
  (pre_tax_food_price : ℝ)
  (h1 : total_paid = 132)
  (h2 : tip_percentage = 0.20)
  (h3 : tax_percentage = 0.10)
  (h4 : total_paid = (1 + tip_percentage) * (1 + tax_percentage) * pre_tax_food_price) :
  pre_tax_food_price = 100 :=
by sorry

end NUMINAMATH_GPT_actual_price_of_food_before_tax_and_tip_l867_86715


namespace NUMINAMATH_GPT_prime_between_30_and_40_with_remainder_1_l867_86726

theorem prime_between_30_and_40_with_remainder_1 (n : ℕ) : 
  n.Prime → 
  30 < n → n < 40 → 
  n % 6 = 1 → 
  n = 37 := 
sorry

end NUMINAMATH_GPT_prime_between_30_and_40_with_remainder_1_l867_86726


namespace NUMINAMATH_GPT_circle_center_coordinates_l867_86798

theorem circle_center_coordinates (x y : ℝ) :
  (x^2 + y^2 - 2*x + 4*y + 3 = 0) → (x = 1 ∧ y = -2) :=
by
  sorry

end NUMINAMATH_GPT_circle_center_coordinates_l867_86798


namespace NUMINAMATH_GPT_minimize_expr_l867_86769

-- Define the function we need to minimize
noncomputable def expr (α β : ℝ) : ℝ := 
  (2 * Real.cos α + 5 * Real.sin β - 8)^2 + (2 * Real.sin α + 5 * Real.cos β - 15)^2

-- State the theorem to prove the minimum value of this expression
theorem minimize_expr (α β : ℝ) : ∃ (α β : ℝ), expr α β = 100 := 
sorry

end NUMINAMATH_GPT_minimize_expr_l867_86769


namespace NUMINAMATH_GPT_find_number_l867_86724

-- Definitions based on the given conditions
def area (s : ℝ) := s^2
def perimeter (s : ℝ) := 4 * s
def given_perimeter : ℝ := 36
def equation (s : ℝ) (n : ℝ) := 5 * area s = 10 * perimeter s + n

-- Statement of the problem
theorem find_number :
  ∃ n : ℝ, equation (given_perimeter / 4) n ∧ n = 45 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l867_86724


namespace NUMINAMATH_GPT_perimeter_of_triangle_l867_86723

-- Define the average length of the sides of the triangle
def average_length (a b c : ℕ) : ℕ := (a + b + c) / 3

-- Define the perimeter of the triangle
def perimeter (a b c : ℕ) : ℕ := a + b + c

-- The theorem we want to prove
theorem perimeter_of_triangle {a b c : ℕ} (h_avg : average_length a b c = 12) : perimeter a b c = 36 :=
sorry

end NUMINAMATH_GPT_perimeter_of_triangle_l867_86723


namespace NUMINAMATH_GPT_days_matt_and_son_eat_only_l867_86799

theorem days_matt_and_son_eat_only (x y : ℕ) 
  (h1 : x + y = 7)
  (h2 : 2 * x + 8 * y = 38) : 
  x = 3 :=
by
  sorry

end NUMINAMATH_GPT_days_matt_and_son_eat_only_l867_86799


namespace NUMINAMATH_GPT_find_initial_sum_l867_86706

-- Define the conditions as constants
def A1 : ℝ := 590
def A2 : ℝ := 815
def t1 : ℝ := 2
def t2 : ℝ := 7

-- Define the variables
variable (P r : ℝ)

-- First condition after 2 years
def condition1 : Prop := A1 = P + P * r * t1

-- Second condition after 7 years
def condition2 : Prop := A2 = P + P * r * t2

-- The statement we need to prove: the initial sum of money P is 500
theorem find_initial_sum (h1 : condition1 P r) (h2 : condition2 P r) : P = 500 :=
sorry

end NUMINAMATH_GPT_find_initial_sum_l867_86706


namespace NUMINAMATH_GPT_find_d_l867_86794

theorem find_d (A B C D : ℕ) (h1 : (A + B + C) / 3 = 130) (h2 : (A + B + C + D) / 4 = 126) : D = 114 :=
by
  sorry

end NUMINAMATH_GPT_find_d_l867_86794


namespace NUMINAMATH_GPT_g_inequality_solution_range_of_m_l867_86751

noncomputable def f (x : ℝ) : ℝ := x^2 - 2*x - 8
noncomputable def g (x : ℝ) : ℝ := 2*x^2 - 4*x - 16
noncomputable def h (x m : ℝ) : ℝ := x^2 - (4 + m)*x + (m + 7)

theorem g_inequality_solution:
  {x : ℝ | g x < 0} = {x : ℝ | -2 < x ∧ x < 4} :=
by
  sorry

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, x > 1 → f x ≥ (m + 2) * x - m - 15) ↔ m ≤ 4 :=
by
  sorry

end NUMINAMATH_GPT_g_inequality_solution_range_of_m_l867_86751


namespace NUMINAMATH_GPT_solve_p_l867_86711

theorem solve_p (p q : ℚ) (h1 : 5 * p + 3 * q = 7) (h2 : 2 * p + 5 * q = 8) : 
  p = 11 / 19 :=
by
  sorry

end NUMINAMATH_GPT_solve_p_l867_86711


namespace NUMINAMATH_GPT_lowest_possible_sale_price_is_30_percent_l867_86700

noncomputable def list_price : ℝ := 80
noncomputable def discount_factor : ℝ := 0.5
noncomputable def additional_discount : ℝ := 0.2
noncomputable def lowest_price := (list_price - discount_factor * list_price) - additional_discount * list_price
noncomputable def percent_of_list_price := (lowest_price / list_price) * 100

theorem lowest_possible_sale_price_is_30_percent :
  percent_of_list_price = 30 := by
  sorry

end NUMINAMATH_GPT_lowest_possible_sale_price_is_30_percent_l867_86700


namespace NUMINAMATH_GPT_average_marks_first_class_l867_86762

theorem average_marks_first_class
  (n1 n2 : ℕ)
  (avg2 : ℝ)
  (combined_avg : ℝ)
  (h_n1 : n1 = 35)
  (h_n2 : n2 = 55)
  (h_avg2 : avg2 = 65)
  (h_combined_avg : combined_avg = 57.22222222222222) :
  (∃ avg1 : ℝ, avg1 = 45) :=
by
  sorry

end NUMINAMATH_GPT_average_marks_first_class_l867_86762


namespace NUMINAMATH_GPT_volume_of_max_area_rect_prism_l867_86730

noncomputable def side_length_of_square_base (P: ℕ) : ℕ := P / 4

noncomputable def area_of_square_base (side: ℕ) : ℕ := side * side

noncomputable def volume_of_rectangular_prism (base_area: ℕ) (height: ℕ) : ℕ := base_area * height

theorem volume_of_max_area_rect_prism
  (P : ℕ) (hP : P = 32) 
  (H : ℕ) (hH : H = 9) 
  : volume_of_rectangular_prism (area_of_square_base (side_length_of_square_base P)) H = 576 := 
by
  sorry

end NUMINAMATH_GPT_volume_of_max_area_rect_prism_l867_86730


namespace NUMINAMATH_GPT_sqrt_inequality_l867_86741

theorem sqrt_inequality (a b c : ℝ) (θ : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h : a * (Real.cos θ)^2 + b * (Real.sin θ)^2 < c) :
  Real.sqrt a * (Real.cos θ)^2 + Real.sqrt b * (Real.sin θ)^2 < Real.sqrt c :=
sorry

end NUMINAMATH_GPT_sqrt_inequality_l867_86741


namespace NUMINAMATH_GPT_relationship_between_a_b_c_l867_86778

variable (a b c : ℝ)
variable (h_a : a = 0.4 ^ 0.2)
variable (h_b : b = 0.4 ^ 0.6)
variable (h_c : c = 2.1 ^ 0.2)

-- Prove the relationship c > a > b
theorem relationship_between_a_b_c : c > a ∧ a > b := by
  sorry

end NUMINAMATH_GPT_relationship_between_a_b_c_l867_86778


namespace NUMINAMATH_GPT_range_of_x_l867_86735

theorem range_of_x (x : ℝ) : 
  (∀ (m : ℝ), |m| ≤ 1 → x^2 - 2 > m * x) ↔ (x < -2 ∨ x > 2) :=
by 
  sorry

end NUMINAMATH_GPT_range_of_x_l867_86735


namespace NUMINAMATH_GPT_option_C_correct_l867_86729

theorem option_C_correct (a b : ℝ) : (-2 * a * b^3)^2 = 4 * a^2 * b^6 :=
by
  sorry

end NUMINAMATH_GPT_option_C_correct_l867_86729


namespace NUMINAMATH_GPT_cube_as_difference_of_squares_l867_86761

theorem cube_as_difference_of_squares (a : ℕ) : 
  a^3 = (a * (a + 1) / 2)^2 - (a * (a - 1) / 2)^2 := 
by 
  -- The proof portion would go here, but since we only need the statement:
  sorry

end NUMINAMATH_GPT_cube_as_difference_of_squares_l867_86761


namespace NUMINAMATH_GPT_problem_l867_86739

-- Define proposition p: for all x in ℝ, x^2 + 1 ≥ 1
def p : Prop := ∀ x : ℝ, x^2 + 1 ≥ 1

-- Define proposition q: for angles A and B in a triangle, A > B ↔ sin A > sin B
def q : Prop := ∀ {A B : ℝ}, A > B ↔ Real.sin A > Real.sin B

-- The problem definition: prove that p ∨ q is true
theorem problem (hp : p) (hq : q) : p ∨ q := sorry

end NUMINAMATH_GPT_problem_l867_86739


namespace NUMINAMATH_GPT_present_age_of_son_l867_86790

-- Define variables for the current ages of the son and the man (father).
variables (S M : ℕ)

-- Define the conditions:
-- The man is 35 years older than his son.
def condition1 : Prop := M = S + 35

-- In two years, the man's age will be twice the age of his son.
def condition2 : Prop := M + 2 = 2 * (S + 2)

-- The theorem that we need to prove:
theorem present_age_of_son : condition1 S M ∧ condition2 S M → S = 33 :=
by
  -- Add sorry to skip the proof.
  sorry

end NUMINAMATH_GPT_present_age_of_son_l867_86790


namespace NUMINAMATH_GPT_not_prime_5n_plus_3_l867_86780

def isSquare (x : ℕ) : Prop := ∃ (k : ℕ), k * k = x

theorem not_prime_5n_plus_3 (n k m : ℕ) (h₁ : 2 * n + 1 = k * k) (h₂ : 3 * n + 1 = m * m) (n_pos : 0 < n) (k_pos : 0 < k) (m_pos : 0 < m) :
  ¬ Nat.Prime (5 * n + 3) :=
sorry -- Proof to be completed

end NUMINAMATH_GPT_not_prime_5n_plus_3_l867_86780


namespace NUMINAMATH_GPT_find_union_A_B_r_find_range_m_l867_86796

def A (x : ℝ) : Prop := x^2 - 2 * x - 3 < 0
def B (x m : ℝ) : Prop := (x - m) * (x - m - 1) ≥ 0

theorem find_union_A_B_r (x : ℝ) : A x ∨ B x 1 := by
  sorry

theorem find_range_m (m : ℝ) (x : ℝ) : (∀ x, A x ↔ B x m) ↔ (m ≥ 3 ∨ m ≤ -2) := by
  sorry

end NUMINAMATH_GPT_find_union_A_B_r_find_range_m_l867_86796


namespace NUMINAMATH_GPT_find_certain_number_l867_86740

theorem find_certain_number (x : ℕ) (h: x - 82 = 17) : x = 99 :=
by
  sorry

end NUMINAMATH_GPT_find_certain_number_l867_86740


namespace NUMINAMATH_GPT_geometric_sequence_sum_a_l867_86791

theorem geometric_sequence_sum_a (a : ℝ) (S : ℕ → ℝ) (h : ∀ n, S n = 4^n + a) :
  a = -1 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_sum_a_l867_86791


namespace NUMINAMATH_GPT_num_possible_n_l867_86797

theorem num_possible_n (n : ℕ) : (∃ a b c : ℕ, 9 * a + 99 * b + 999 * c = 5000 ∧ n = a + 2 * b + 3 * c) ↔ n ∈ {x | x = a + 2 * b + 3 * c ∧ 0 ≤ 9 * (b + 12 * c) ∧ 9 * (b + 12 * c) ≤ 555} :=
sorry

end NUMINAMATH_GPT_num_possible_n_l867_86797


namespace NUMINAMATH_GPT_binomial_product_l867_86783

theorem binomial_product (x : ℝ) : 
  (2 - x^4) * (3 + x^5) = -x^9 - 3 * x^4 + 2 * x^5 + 6 :=
by 
  sorry

end NUMINAMATH_GPT_binomial_product_l867_86783


namespace NUMINAMATH_GPT_acute_triangle_orthocenter_l867_86703

variables (A B C H : Point) (a b c h_a h_b h_c : Real)

def acute_triangle (α β γ : Point) : Prop := 
-- Definition that ensures triangle αβγ is acute
sorry

def orthocenter (α β γ ω : Point) : Prop := 
-- Definition that ω is the orthocenter of triangle αβγ 
sorry

def sides_of_triangle (α β γ : Point) : (Real × Real × Real) := 
-- Function that returns the side lengths of triangle αβγ as (a, b, c)
sorry

def altitudes_of_triangle (α β γ θ : Point) : (Real × Real × Real) := 
-- Function that returns the altitudes of triangle αβγ with orthocenter θ as (h_a, h_b, h_c)
sorry

theorem acute_triangle_orthocenter 
  (A B C H : Point)
  (a b c h_a h_b h_c : Real)
  (ht : acute_triangle A B C)
  (orth : orthocenter A B C H)
  (sides : sides_of_triangle A B C = (a, b, c))
  (alts : altitudes_of_triangle A B C H = (h_a, h_b, h_c)) :
  AH * h_a + BH * h_b + CH * h_c = (a^2 + b^2 + c^2) / 2 :=
by sorry


end NUMINAMATH_GPT_acute_triangle_orthocenter_l867_86703


namespace NUMINAMATH_GPT_simplify_expression_l867_86719

theorem simplify_expression (x : ℝ) (hx : x ≠ 0) : (x^2)⁻¹ - 2 = (1 - 2 * x^2) / (x^2) :=
by
  -- proof here
  sorry

end NUMINAMATH_GPT_simplify_expression_l867_86719


namespace NUMINAMATH_GPT_total_population_correct_l867_86772

/-- Define the populations of each city -/
def Population.Seattle : ℕ := sorry
def Population.LakeView : ℕ := 24000
def Population.Boise : ℕ := (3 * Population.Seattle) / 5

/-- Population of Lake View is 4000 more than the population of Seattle -/
axiom lake_view_population : Population.LakeView = Population.Seattle + 4000

/-- Define the total population -/
def total_population : ℕ :=
  Population.Seattle + Population.LakeView + Population.Boise

/-- Prove that total population of the three cities is 56000 -/
theorem total_population_correct :
  total_population = 56000 :=
sorry

end NUMINAMATH_GPT_total_population_correct_l867_86772


namespace NUMINAMATH_GPT_customers_left_proof_l867_86781

def initial_customers : ℕ := 21
def tables : ℕ := 3
def people_per_table : ℕ := 3
def remaining_customers : ℕ := tables * people_per_table
def customers_left (initial remaining : ℕ) : ℕ := initial - remaining

theorem customers_left_proof : customers_left initial_customers remaining_customers = 12 := sorry

end NUMINAMATH_GPT_customers_left_proof_l867_86781


namespace NUMINAMATH_GPT_vasya_improved_example1_vasya_improved_example2_l867_86720

theorem vasya_improved_example1 : (333 / 3) - (33 / 3) = 100 := by
  sorry

theorem vasya_improved_example2 : (33 * 3) + (3 / 3) = 100 := by
  sorry

end NUMINAMATH_GPT_vasya_improved_example1_vasya_improved_example2_l867_86720


namespace NUMINAMATH_GPT_mrs_hilt_total_spent_l867_86784

def kids_ticket_usual_cost : ℕ := 1 -- $1 for 4 tickets
def adults_ticket_usual_cost : ℕ := 2 -- $2 for 3 tickets

def kids_ticket_deal_cost : ℕ := 4 -- $4 for 20 tickets
def adults_ticket_deal_cost : ℕ := 8 -- $8 for 15 tickets

def kids_tickets_purchased : ℕ := 24
def adults_tickets_purchased : ℕ := 18

def total_kids_ticket_cost : ℕ :=
  let kids_deal_tickets := kids_ticket_deal_cost
  let remaining_kids_tickets := kids_ticket_usual_cost
  kids_deal_tickets + remaining_kids_tickets

def total_adults_ticket_cost : ℕ :=
  let adults_deal_tickets := adults_ticket_deal_cost
  let remaining_adults_tickets := adults_ticket_usual_cost
  adults_deal_tickets + remaining_adults_tickets

def total_cost (kids_cost adults_cost : ℕ) : ℕ :=
  kids_cost + adults_cost

theorem mrs_hilt_total_spent : total_cost total_kids_ticket_cost total_adults_ticket_cost = 15 := by
  sorry

end NUMINAMATH_GPT_mrs_hilt_total_spent_l867_86784


namespace NUMINAMATH_GPT_kennedy_softball_park_miles_l867_86753

theorem kennedy_softball_park_miles :
  let miles_per_gallon := 19
  let gallons_of_gas := 2
  let total_drivable_miles := miles_per_gallon * gallons_of_gas
  let miles_to_school := 15
  let miles_to_burger_restaurant := 2
  let miles_to_friends_house := 4
  let miles_home := 11
  total_drivable_miles - (miles_to_school + miles_to_burger_restaurant + miles_to_friends_house + miles_home) = 6 :=
by
  sorry

end NUMINAMATH_GPT_kennedy_softball_park_miles_l867_86753


namespace NUMINAMATH_GPT_range_of_m_l867_86745

def y1 (m x : ℝ) : ℝ :=
  m * (x - 2 * m) * (x + m + 2)

def y2 (x : ℝ) : ℝ :=
  x - 1

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, y1 m x < 0 ∨ y2 x < 0) ∧ (∃ x : ℝ, x < -3 ∧ y1 m x * y2 x < 0) ↔ (-4 < m ∧ m < -3/2) := 
by
  sorry

end NUMINAMATH_GPT_range_of_m_l867_86745


namespace NUMINAMATH_GPT_product_of_distinct_roots_l867_86747

theorem product_of_distinct_roots (x1 x2 : ℝ) (hx1 : x1 ^ 2 - 2 * x1 = 1) (hx2 : x2 ^ 2 - 2 * x2 = 1) (h_distinct : x1 ≠ x2) : 
  x1 * x2 = -1 := 
  sorry

end NUMINAMATH_GPT_product_of_distinct_roots_l867_86747


namespace NUMINAMATH_GPT_linear_function_l867_86725

theorem linear_function (f : ℝ → ℝ)
  (h : ∀ x, f (f x) = 4 * x + 6) :
  (∀ x, f x = 2 * x + 2) ∨ (∀ x, f x = -2 * x - 6) :=
sorry

end NUMINAMATH_GPT_linear_function_l867_86725


namespace NUMINAMATH_GPT_pizza_eaten_after_six_trips_l867_86750

theorem pizza_eaten_after_six_trips :
  let a := (1 : ℝ) / 3
  let r := (1 : ℝ) / 3
  let n := 6
  let S_n := a * (1 - r^n) / (1 - r)
  S_n = 364 / 729 :=
by
  let a := (1 : ℝ) / 3
  let r := (1 : ℝ) / 3
  let n := 6
  let S_n := a * (1 - r^n) / (1 - r)
  have : S_n = (1 / 3) * (1 - (1 / 3)^6) / (1 - 1 / 3) := by sorry
  have : S_n = 364 / 729 := by sorry
  exact this

end NUMINAMATH_GPT_pizza_eaten_after_six_trips_l867_86750


namespace NUMINAMATH_GPT_gcd_polynomial_is_25_l867_86708

theorem gcd_polynomial_is_25 (b : ℕ) (h : ∃ k : ℕ, b = 2700 * k) :
  Nat.gcd (b^2 + 27 * b + 75) (b + 25) = 25 :=
by 
    sorry

end NUMINAMATH_GPT_gcd_polynomial_is_25_l867_86708


namespace NUMINAMATH_GPT_candies_left_is_correct_l867_86714

-- Define the number of candies bought on different days
def candiesBoughtTuesday : ℕ := 3
def candiesBoughtThursday : ℕ := 5
def candiesBoughtFriday : ℕ := 2

-- Define the number of candies eaten
def candiesEaten : ℕ := 6

-- Define the total candies left
def candiesLeft : ℕ := (candiesBoughtTuesday + candiesBoughtThursday + candiesBoughtFriday) - candiesEaten

theorem candies_left_is_correct : candiesLeft = 4 := by
  -- Placeholder proof: replace 'sorry' with the actual proof when necessary
  sorry

end NUMINAMATH_GPT_candies_left_is_correct_l867_86714


namespace NUMINAMATH_GPT_find_all_triples_l867_86779

def satisfying_triples (a b c : ℝ) : Prop :=
  0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 
  (a^2 + a*b = c) ∧ 
  (b^2 + b*c = a) ∧ 
  (c^2 + c*a = b)

theorem find_all_triples (a b c : ℝ) : satisfying_triples a b c ↔ (a = 0 ∧ b = 0 ∧ c = 0) ∨ (a = 1/2 ∧ b = 1/2 ∧ c = 1/2) :=
by sorry

end NUMINAMATH_GPT_find_all_triples_l867_86779


namespace NUMINAMATH_GPT_gambler_difference_eq_two_l867_86786

theorem gambler_difference_eq_two (x y : ℕ) (x_lost y_lost : ℕ) :
  20 * x + 100 * y = 3000 ∧
  x + y = 14 ∧
  20 * (14 - y_lost) + 100 * y_lost = 760 →
  (x_lost - y_lost = 2) := sorry

end NUMINAMATH_GPT_gambler_difference_eq_two_l867_86786


namespace NUMINAMATH_GPT_num_new_students_l867_86792

-- Definitions based on the provided conditions
def original_class_strength : ℕ := 10
def original_average_age : ℕ := 40
def new_students_avg_age : ℕ := 32
def decrease_in_average_age : ℕ := 4
def new_average_age : ℕ := original_average_age - decrease_in_average_age
def new_class_strength (n : ℕ) : ℕ := original_class_strength + n

-- The proof statement
theorem num_new_students (n : ℕ) :
  (original_class_strength * original_average_age + n * new_students_avg_age) 
  = new_class_strength n * new_average_age → n = 10 :=
by
  sorry

end NUMINAMATH_GPT_num_new_students_l867_86792


namespace NUMINAMATH_GPT_min_value_a_plus_b_l867_86738

theorem min_value_a_plus_b (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) 
  (h3 : a^2 ≥ 8 * b) (h4 : b^2 ≥ a) : a + b ≥ 6 := by
  sorry

end NUMINAMATH_GPT_min_value_a_plus_b_l867_86738


namespace NUMINAMATH_GPT_effective_discount_l867_86771

theorem effective_discount (original_price sale_price price_after_coupon : ℝ) :
  sale_price = 0.4 * original_price →
  price_after_coupon = 0.7 * sale_price →
  (original_price - price_after_coupon) / original_price * 100 = 72 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_effective_discount_l867_86771


namespace NUMINAMATH_GPT_inequality_property_l867_86785

variable {a b : ℝ} (h : a > b) (c : ℝ)

theorem inequality_property : a * |c| ≥ b * |c| :=
sorry

end NUMINAMATH_GPT_inequality_property_l867_86785


namespace NUMINAMATH_GPT_ship_cargo_weight_l867_86744

theorem ship_cargo_weight (initial_cargo_tons additional_cargo_tons : ℝ) (unloaded_cargo_pounds : ℝ)
    (ton_to_kg pound_to_kg : ℝ) :
    initial_cargo_tons = 5973.42 →
    additional_cargo_tons = 8723.18 →
    unloaded_cargo_pounds = 2256719.55 →
    ton_to_kg = 907.18474 →
    pound_to_kg = 0.45359237 →
    (initial_cargo_tons * ton_to_kg + additional_cargo_tons * ton_to_kg - unloaded_cargo_pounds * pound_to_kg = 12302024.7688159) :=
by
  intros
  sorry

end NUMINAMATH_GPT_ship_cargo_weight_l867_86744


namespace NUMINAMATH_GPT_find_k_range_l867_86722

noncomputable def f (k x : ℝ) : ℝ := (k * x + 1 / 3) * Real.exp x - x

theorem find_k_range : 
  (∃ (k : ℝ), ∀ (x : ℕ), x > 0 → (f k (x : ℝ) < 0 ↔ x = 1)) ↔
  (k ≥ 1 / (Real.exp 2) - 1 / 6 ∧ k < 1 / Real.exp 1 - 1 / 3) :=
sorry

end NUMINAMATH_GPT_find_k_range_l867_86722


namespace NUMINAMATH_GPT_perfect_squares_from_equation_l867_86734

theorem perfect_squares_from_equation (x y : ℕ) (h : 2 * x^2 + x = 3 * y^2 + y) :
  ∃ a b c : ℕ, x - y = a^2 ∧ 2 * x + 2 * y + 1 = b^2 ∧ 3 * x + 3 * y + 1 = c^2 :=
by
  sorry

end NUMINAMATH_GPT_perfect_squares_from_equation_l867_86734


namespace NUMINAMATH_GPT_sarah_score_l867_86705

-- Given conditions
variable (s g : ℕ) -- Sarah's score and Greg's score
variable (h1 : s = g + 60) -- Sarah's score is 60 points more than Greg's
variable (h2 : (s + g) / 2 = 130) -- The average of their two scores is 130

-- Proof statement
theorem sarah_score : s = 160 :=
by
  sorry

end NUMINAMATH_GPT_sarah_score_l867_86705


namespace NUMINAMATH_GPT_pump_B_time_l867_86767

theorem pump_B_time (T_B : ℝ) (h1 : ∀ (h1 : T_B > 0),
  (1 / 4 + 1 / T_B = 3 / 4)) :
  T_B = 2 := 
by
  sorry

end NUMINAMATH_GPT_pump_B_time_l867_86767


namespace NUMINAMATH_GPT_range_of_a_l867_86773

noncomputable def f (a x : ℝ) : ℝ := x + (a^2) / (4 * x)
noncomputable def g (x : ℝ) : ℝ := x - Real.log x

theorem range_of_a (a : ℝ) :
  (∀ x1 x2, 1 ≤ x1 ∧ x1 ≤ Real.exp 1 ∧ 1 ≤ x2 ∧ x2 ≤ Real.exp 1 → f a x1 ≥ g x2) → 
  2 * Real.sqrt (Real.exp 1 - 2) ≤ a := sorry

end NUMINAMATH_GPT_range_of_a_l867_86773


namespace NUMINAMATH_GPT_smallest_integer_modulus_l867_86777

theorem smallest_integer_modulus :
  ∃ n : ℕ, 0 < n ∧ (7 ^ n ≡ n ^ 4 [MOD 3]) ∧
  ∀ m : ℕ, 0 < m ∧ (7 ^ m ≡ m ^ 4 [MOD 3]) → n ≤ m :=
by
  sorry

end NUMINAMATH_GPT_smallest_integer_modulus_l867_86777


namespace NUMINAMATH_GPT_domain_of_f_l867_86749

def domain_f (x : ℝ) : Prop := x ≤ 4 ∧ x ≠ 1

theorem domain_of_f :
  {x : ℝ | ∃(h1 : 4 - x ≥ 0) (h2 : x - 1 ≠ 0), true} = {x : ℝ | domain_f x} :=
by
  sorry

end NUMINAMATH_GPT_domain_of_f_l867_86749


namespace NUMINAMATH_GPT_part1_part2_l867_86752

open Set

def A : Set ℤ := { x | ∃ (m n : ℤ), x = m^2 - n^2 }

theorem part1 : 3 ∈ A := 
by sorry

theorem part2 (k : ℤ) : 4 * k - 2 ∉ A := 
by sorry

end NUMINAMATH_GPT_part1_part2_l867_86752


namespace NUMINAMATH_GPT_first_quadrant_sin_cos_inequality_l867_86748

def is_first_quadrant_angle (α : ℝ) : Prop :=
  0 < Real.sin α ∧ 0 < Real.cos α

theorem first_quadrant_sin_cos_inequality (α : ℝ) :
  (is_first_quadrant_angle α ↔ Real.sin α + Real.cos α > 1) :=
by
  sorry

end NUMINAMATH_GPT_first_quadrant_sin_cos_inequality_l867_86748


namespace NUMINAMATH_GPT_trip_time_l867_86758

theorem trip_time (x : ℝ) (T : ℝ) :
  (70 * 4 + 60 * 5 + 50 * x) / (4 + 5 + x) = 58 → 
  T = 4 + 5 + x → 
  T = 16.25 :=
by
  intro h1 h2
  sorry

end NUMINAMATH_GPT_trip_time_l867_86758


namespace NUMINAMATH_GPT_z_is_1_2_decades_younger_than_x_l867_86754

variable (X Y Z : ℝ)

theorem z_is_1_2_decades_younger_than_x (h : X + Y = Y + Z + 12) : (X - Z) / 10 = 1.2 :=
by
  sorry

end NUMINAMATH_GPT_z_is_1_2_decades_younger_than_x_l867_86754


namespace NUMINAMATH_GPT_pipe_fills_tank_without_leak_l867_86707

theorem pipe_fills_tank_without_leak (T : ℝ) (h1 : 1 / 6 = 1 / T - 1 / 12) : T = 4 :=
by
  sorry

end NUMINAMATH_GPT_pipe_fills_tank_without_leak_l867_86707


namespace NUMINAMATH_GPT_curve_crosses_itself_l867_86710

-- Definitions of the parametric equations
def x (t : ℝ) : ℝ := t^2 - 4
def y (t : ℝ) : ℝ := t^3 - 6*t + 3

-- The theorem statement
theorem curve_crosses_itself :
  ∃ t₁ t₂ : ℝ, t₁ ≠ t₂ ∧ x t₁ = x t₂ ∧ y t₁ = y t₂ ∧ (x t₁, y t₁) = (2, 3) :=
by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_curve_crosses_itself_l867_86710


namespace NUMINAMATH_GPT_hyperbola_eccentricity_l867_86759

theorem hyperbola_eccentricity (a c b : ℝ) (h₀ : b = 3)
  (h₁ : ∃ p, (p = 5) ∧ (a^2 + b^2 = (p : ℝ)^2))
  (h₂ : ∃ f, f = (p : ℝ)) :
  ∃ e, e = c / a ∧ e = 5 / 4 :=
by
  obtain ⟨p, hp, hap⟩ := h₁
  obtain ⟨f, hf⟩ := h₂
  sorry

end NUMINAMATH_GPT_hyperbola_eccentricity_l867_86759


namespace NUMINAMATH_GPT_find_m_l867_86736

theorem find_m (x y m : ℝ)
  (h1 : 6 * x + 3 = 0)
  (h2 : 3 * y + m = 15)
  (h3 : x * y = 1) : m = 21 := 
sorry

end NUMINAMATH_GPT_find_m_l867_86736


namespace NUMINAMATH_GPT_new_area_is_497_l867_86756

noncomputable def rect_area_proof : Prop :=
  ∃ (l w l' w' : ℝ),
    -- initial area condition
    l * w = 540 ∧ 
    -- conditions for new dimensions
    l' = 0.8 * l ∧
    w' = 1.15 * w ∧
    -- final area calculation
    l' * w' = 497

theorem new_area_is_497 : rect_area_proof := by
  sorry

end NUMINAMATH_GPT_new_area_is_497_l867_86756


namespace NUMINAMATH_GPT_differences_l867_86704

def seq (n : ℕ) : ℕ := n^2 + 1

def first_diff (n : ℕ) : ℕ := (seq (n + 1)) - (seq n)

def second_diff (n : ℕ) : ℕ := (first_diff (n + 1)) - (first_diff n)

def third_diff (n : ℕ) : ℕ := (second_diff (n + 1)) - (second_diff n)

theorem differences (n : ℕ) : first_diff n = 2 * n + 1 ∧ 
                             second_diff n = 2 ∧ 
                             third_diff n = 0 := by 
  sorry

end NUMINAMATH_GPT_differences_l867_86704


namespace NUMINAMATH_GPT_find_divisor_l867_86765

theorem find_divisor
  (Dividend : ℕ)
  (Quotient : ℕ)
  (Remainder : ℕ)
  (h1 : Dividend = 686)
  (h2 : Quotient = 19)
  (h3 : Remainder = 2) :
  ∃ (Divisor : ℕ), (Dividend = (Divisor * Quotient) + Remainder) ∧ Divisor = 36 :=
by
  sorry

end NUMINAMATH_GPT_find_divisor_l867_86765


namespace NUMINAMATH_GPT_problem_II_l867_86788

noncomputable def a_n (n : ℕ) : ℝ :=
  if n = 0 then 0 else (1 / 3)^n

noncomputable def S_n (n : ℕ) : ℝ :=
  if n = 0 then 0 else (1 / 2) * (1 - (1 / 3)^n)

lemma problem_I_1 (n : ℕ) (hn : n > 0) : a_n n = (1 / 3)^n := by
  sorry

lemma problem_I_2 (n : ℕ) (hn : n > 0) : S_n n = (1 / 2) * (1 - (1 / 3)^n) := by
  sorry

theorem problem_II (t : ℝ) : S_n 1 = 1 / 3 ∧ S_n 2 = 4 / 9 ∧ S_n 3 = 13 / 27 ∧
  (S_n 1 + 3 * (S_n 2 + S_n 3) = 2 * (S_n 1 + S_n 2) * t) ↔ t = 2 := by
  sorry

end NUMINAMATH_GPT_problem_II_l867_86788


namespace NUMINAMATH_GPT_find_b_l867_86795

def p (x : ℝ) : ℝ := 2 * x - 3
def q (x : ℝ) (b : ℝ) : ℝ := 5 * x - b

theorem find_b (b : ℝ) (h : p (q 3 b) = 13) : b = 7 :=
by sorry

end NUMINAMATH_GPT_find_b_l867_86795


namespace NUMINAMATH_GPT_AB_complete_work_together_in_10_days_l867_86713

-- Definitions for the work rates
def rate_A (work : ℕ) : ℚ := work / 14 -- A's rate of work (work per day)
def rate_AB (work : ℕ) : ℚ := work / 10 -- A and B together's rate of work (work per day)

-- Definition for B's rate of work derived from the combined rate and A's rate
def rate_B (work : ℕ) : ℚ := rate_AB work - rate_A work

-- Definition of the fact that the combined rate should equal their individual rates summed
def combined_rate_equals_sum (work : ℕ) : Prop := rate_AB work = (rate_A work + rate_B work)

-- Statement we need to prove:
theorem AB_complete_work_together_in_10_days (work : ℕ) (h : combined_rate_equals_sum work) : rate_AB work = work / 10 :=
by {
  -- Given conditions are implicitly used without a formal proof here.
  -- To prove that A and B together can indeed complete the work in 10 days.
  sorry
}


end NUMINAMATH_GPT_AB_complete_work_together_in_10_days_l867_86713


namespace NUMINAMATH_GPT_slope_intercept_of_line_l867_86742

theorem slope_intercept_of_line :
  ∃ (l : ℝ → ℝ), (∀ x, l x = (4 * x - 9) / 3) ∧ l 3 = 1 ∧ ∃ k, k / (1 + k^2) = 1 / 2 ∧ l x = (k^2 - 1) / (1 + k^2) := sorry

end NUMINAMATH_GPT_slope_intercept_of_line_l867_86742


namespace NUMINAMATH_GPT_proof_problem_l867_86731

variable (y θ Q : ℝ)

-- Given condition
def condition : Prop := 5 * (3 * y + 7 * Real.sin θ) = Q

-- Goal to be proved
def goal : Prop := 15 * (9 * y + 21 * Real.sin θ) = 9 * Q

theorem proof_problem (h : condition y θ Q) : goal y θ Q :=
by
  sorry

end NUMINAMATH_GPT_proof_problem_l867_86731
