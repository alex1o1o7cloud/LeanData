import Mathlib

namespace fruit_basket_count_l1134_113485

theorem fruit_basket_count :
  let pears := 8
  let bananas := 12
  let total_baskets := (pears + 1) * (bananas + 1) - 1
  total_baskets = 116 :=
by
  sorry

end fruit_basket_count_l1134_113485


namespace sally_balloons_l1134_113446

theorem sally_balloons (F S : ℕ) (h1 : F = 3 * S) (h2 : F = 18) : S = 6 :=
by sorry

end sally_balloons_l1134_113446


namespace original_price_l1134_113436

theorem original_price (x : ℝ) (h : x * (1 / 8) = 8) : x = 64 := by
  -- To be proved
  sorry

end original_price_l1134_113436


namespace gcd_between_35_and_7_l1134_113460

theorem gcd_between_35_and_7 {n : ℕ} (h1 : 65 < n) (h2 : n < 75) (h3 : gcd 35 n = 7) : n = 70 := 
sorry

end gcd_between_35_and_7_l1134_113460


namespace combined_weight_difference_l1134_113407

def john_weight : ℕ := 81
def roy_weight : ℕ := 79
def derek_weight : ℕ := 91
def samantha_weight : ℕ := 72

theorem combined_weight_difference :
  derek_weight - samantha_weight = 19 :=
by
  sorry

end combined_weight_difference_l1134_113407


namespace sum_ages_l1134_113481

theorem sum_ages (A_years B_years C_years : ℕ) (h1 : B_years = 30)
  (h2 : 10 * (B_years - 10) = (A_years - 10) * 2)
  (h3 : 10 * (B_years - 10) = (C_years - 10) * 3) :
  A_years + B_years + C_years = 90 :=
sorry

end sum_ages_l1134_113481


namespace xiao_hua_correct_questions_l1134_113414

-- Definitions of the problem conditions
def n : Nat := 20
def p_correct : Int := 5
def p_wrong : Int := -2
def score : Int := 65

-- Theorem statement to prove the number of correct questions
theorem xiao_hua_correct_questions : 
  ∃ k : Nat, k = ((n : Int) - ((n * p_correct - score) / (p_correct - p_wrong))) ∧ 
               k = 15 :=
by
  sorry

end xiao_hua_correct_questions_l1134_113414


namespace bananas_per_friend_l1134_113431

theorem bananas_per_friend (total_bananas : ℤ) (total_friends : ℤ) (H1 : total_bananas = 21) (H2 : total_friends = 3) : 
  total_bananas / total_friends = 7 :=
by
  sorry

end bananas_per_friend_l1134_113431


namespace line_y_axis_intersect_l1134_113429

theorem line_y_axis_intersect (x1 y1 x2 y2 : ℝ) (h1 : x1 = 3 ∧ y1 = 27) (h2 : x2 = -7 ∧ y2 = -1) :
  ∃ y : ℝ, (∀ x : ℝ, y = (y2 - y1) / (x2 - x1) * (x - x1) + y1) ∧ y = 18.6 :=
by
  sorry

end line_y_axis_intersect_l1134_113429


namespace reduction_percentage_price_increase_l1134_113477

-- Proof Problem 1: Reduction Percentage
theorem reduction_percentage (a : ℝ) (h₁ : (50 * (1 - a)^2 = 32)) : a = 0.2 := by
  sorry

-- Proof Problem 2: Price Increase for Daily Profit
theorem price_increase 
  (x : ℝ)
  (profit_per_kg : ℝ := 10)
  (initial_sales : ℕ := 500)
  (sales_decrease_per_unit : ℝ := 20)
  (required_profit : ℝ := 6000)
  (h₁ : (10 + x) * (initial_sales - sales_decrease_per_unit * x) = required_profit) : 
  x = 5 := by
  sorry

end reduction_percentage_price_increase_l1134_113477


namespace stockings_total_cost_l1134_113458

-- Define a function to compute the total cost given the conditions
def total_cost (n_grandchildren n_children : Nat) 
               (stocking_price discount monogram_cost : Nat) : Nat :=
  let total_stockings := n_grandchildren + n_children
  let discounted_price := stocking_price - (stocking_price * discount / 100)
  let total_stockings_cost := discounted_price * total_stockings
  let total_monogram_cost := monogram_cost * total_stockings
  total_stockings_cost + total_monogram_cost

-- Prove that the total cost calculation is correct given the conditions
theorem stockings_total_cost :
  total_cost 5 4 20 10 5 = 207 :=
by
  -- A placeholder for the proof
  sorry

end stockings_total_cost_l1134_113458


namespace max_number_of_9_letter_palindromes_l1134_113413

theorem max_number_of_9_letter_palindromes : 26^5 = 11881376 :=
by sorry

end max_number_of_9_letter_palindromes_l1134_113413


namespace triangle_inequality_sqrt_equality_condition_l1134_113483

theorem triangle_inequality_sqrt 
  {a b c : ℝ} 
  (h1 : a + b > c) 
  (h2 : b + c > a) 
  (h3 : c + a > b) :
  (Real.sqrt (a + b - c) + Real.sqrt (c + a - b) + Real.sqrt (b + c - a) 
  ≤ Real.sqrt a + Real.sqrt b + Real.sqrt c) := 
sorry

theorem equality_condition 
  {a b c : ℝ} 
  (h1 : a + b > c) 
  (h2 : b + c > a) 
  (h3 : c + a > b) :
  (Real.sqrt (a + b - c) + Real.sqrt (c + a - b) + Real.sqrt (b + c - a) 
  = Real.sqrt a + Real.sqrt b + Real.sqrt c) → 
  (a = b ∧ b = c) := 
sorry

end triangle_inequality_sqrt_equality_condition_l1134_113483


namespace john_paintball_times_l1134_113437

theorem john_paintball_times (x : ℕ) (cost_per_box : ℕ) (boxes_per_play : ℕ) (monthly_spending : ℕ) :
  (cost_per_box = 25) → (boxes_per_play = 3) → (monthly_spending = 225) → (boxes_per_play * cost_per_box * x = monthly_spending) → x = 3 :=
by
  intros h1 h2 h3 h4
  -- proof would go here
  sorry

end john_paintball_times_l1134_113437


namespace no_such_abc_exists_l1134_113405

theorem no_such_abc_exists :
  ¬ ∃ (a b c : ℝ), 
      ((a > 0 ∧ b > 0 ∧ c < 0) ∨ (a > 0 ∧ c > 0 ∧ b < 0) ∨ (b > 0 ∧ c > 0 ∧ a < 0) ∨
       (a < 0 ∧ b < 0 ∧ c > 0) ∨ (a < 0 ∧ c < 0 ∧ b > 0) ∨ (b < 0 ∧ c < 0 ∧ a > 0)) ∧
      ((a < 0 ∧ b < 0 ∧ c > 0) ∨ (a < 0 ∧ c < 0 ∧ b > 0) ∨ (b < 0 ∨ c < 0 ∧ a > 0) ∨
       (a > 0 ∧ b > 0 ∧ c < 0) ∨ (a > 0 ∧ c > 0 ∧ b < 0) ∨ (b > 0 ∧ c > 0 ∧ a < 0)) :=
by {
  sorry
}

end no_such_abc_exists_l1134_113405


namespace second_derivative_parametric_l1134_113450

noncomputable def x (t : ℝ) := Real.sqrt (t - 1)
noncomputable def y (t : ℝ) := 1 / Real.sqrt t

noncomputable def y_xx (t : ℝ) := (2 * t - 3) * Real.sqrt t / t^3

theorem second_derivative_parametric :
  ∀ t, y_xx t = (2 * t - 3) * Real.sqrt t / t^3 := sorry

end second_derivative_parametric_l1134_113450


namespace fence_calculation_l1134_113423

def width : ℕ := 4
def length : ℕ := 2 * width - 1
def fence_needed : ℕ := 2 * length + 2 * width

theorem fence_calculation : fence_needed = 22 := by
  sorry

end fence_calculation_l1134_113423


namespace area_of_triangle_PQR_l1134_113440

def Point := (ℝ × ℝ)
def area_of_triangle (P Q R : Point) : ℝ :=
  0.5 * abs (P.1 * (Q.2 - R.2) + Q.1 * (R.2 - P.2) + R.1 * (P.2 - Q.2))

def P : Point := (1, 1)
def Q : Point := (4, 5)
def R : Point := (7, 2)

theorem area_of_triangle_PQR :
  area_of_triangle P Q R = 10.5 := by
  sorry

end area_of_triangle_PQR_l1134_113440


namespace minimum_value_f_l1134_113457

noncomputable def f (x y : ℝ) : ℝ :=
  (x^2 / (y - 2)^2) + (y^2 / (x - 2)^2)

theorem minimum_value_f (x y : ℝ) (hx : x > 2) (hy : y > 2) :
  ∃ (a : ℝ), (∀ (b : ℝ), f x y >= b) ∧ a = 10 := sorry

end minimum_value_f_l1134_113457


namespace find_y_l1134_113419

-- Definitions of the angles
def angle_ABC : ℝ := 80
def angle_BAC : ℝ := 70
def angle_BCA : ℝ := 180 - angle_ABC - angle_BAC -- calculation of third angle in triangle ABC

-- Right angle in triangle CDE
def angle_ECD : ℝ := 90

-- Defining the proof problem
theorem find_y (y : ℝ) : 
  angle_BCA = 30 →
  angle_CDE = angle_BCA →
  angle_CDE + y + angle_ECD = 180 → 
  y = 60 := by
  intro h1 h2 h3
  sorry

end find_y_l1134_113419


namespace set_intersection_l1134_113438

-- Definitions of sets M and N
def M : Set ℤ := {-1, 1, 2}
def N : Set ℤ := {1, 2, 3}

-- The statement to prove that M ∩ N = {1, 2}
theorem set_intersection :
  M ∩ N = {1, 2} := by
  sorry

end set_intersection_l1134_113438


namespace trail_length_proof_l1134_113465

theorem trail_length_proof (x1 x2 x3 x4 x5 : ℝ)
  (h1 : x1 + x2 = 28)
  (h2 : x2 + x3 = 30)
  (h3 : x3 + x4 + x5 = 42)
  (h4 : x1 + x4 = 30) :
  x1 + x2 + x3 + x4 + x5 = 70 := by
  sorry

end trail_length_proof_l1134_113465


namespace kangaroo_can_jump_exact_200_in_30_jumps_l1134_113417

/-!
  A kangaroo can jump:
  - 3 meters using its left leg
  - 5 meters using its right leg
  - 7 meters using both legs
  - -3 meters backward
  We need to prove that the kangaroo can jump exactly 200 meters in 30 jumps.
 -/

theorem kangaroo_can_jump_exact_200_in_30_jumps :
  ∃ (n3 n5 n7 nm3 : ℕ),
    (n3 + n5 + n7 + nm3 = 30) ∧
    (3 * n3 + 5 * n5 + 7 * n7 - 3 * nm3 = 200) :=
sorry

end kangaroo_can_jump_exact_200_in_30_jumps_l1134_113417


namespace solve_for_x_l1134_113418

theorem solve_for_x (x : ℝ) : (1 + 2*x + 3*x^2) / (3 + 2*x + x^2) = 3 → x = -2 :=
by
  intro h
  sorry

end solve_for_x_l1134_113418


namespace range_of_a_l1134_113495

theorem range_of_a (a : ℝ) :
  (¬ ∃ x : ℝ, |x - a| + |x - 1| ≤ 2) → (a > 3 ∨ a < -1) :=
by
  sorry

end range_of_a_l1134_113495


namespace find_a_b_find_c_range_l1134_113424

noncomputable def f (a b c x : ℝ) : ℝ := 2 * x^3 + 3 * a * x^2 + 3 * b * x + 8 * c

theorem find_a_b (a b c : ℝ) (extreme_x1 extreme_x2 : ℝ) (h1 : extreme_x1 = 1) (h2 : extreme_x2 = 2) 
  (h3 : (deriv (f a b c) 1) = 0) (h4 : (deriv (f a b c) 2) = 0) : 
  a = -3 ∧ b = 4 :=
by sorry

theorem find_c_range (c : ℝ) (h : ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 3 → f (-3) 4 c x < c^2) : 
  c ∈ Set.Iio (-1) ∪ Set.Ioi 9 :=
by sorry

end find_a_b_find_c_range_l1134_113424


namespace digit_product_equality_l1134_113445

theorem digit_product_equality :
  ∃ (a b c d e f g h i j : ℕ),
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧ a ≠ i ∧ a ≠ j ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧ b ≠ i ∧ b ≠ j ∧
    c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧ c ≠ i ∧ c ≠ j ∧
    d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧ d ≠ i ∧ d ≠ j ∧
    e ≠ f ∧ e ≠ g ∧ e ≠ h ∧ e ≠ i ∧ e ≠ j ∧
    f ≠ g ∧ f ≠ h ∧ f ≠ i ∧ f ≠ j ∧
    g ≠ h ∧ g ≠ i ∧ g ≠ j ∧
    h ≠ i ∧ h ≠ j ∧
    i ≠ j ∧
    a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ e < 10 ∧ f < 10 ∧ g < 10 ∧ h < 10 ∧ i < 10 ∧ j < 10 ∧
    a * (10 * b + c) * (100 * d + 10 * e + f) = (1000 * g + 100 * h + 10 * i + j) :=
sorry

end digit_product_equality_l1134_113445


namespace range_of_m_l1134_113452

noncomputable def setA := {x : ℝ | -2 ≤ x ∧ x ≤ 7}
noncomputable def setB (m : ℝ) := {x : ℝ | m + 1 ≤ x ∧ x ≤ 2 * m - 1}

theorem range_of_m (m : ℝ) : (setA ∪ setB m = setA) → m ≤ 4 :=
by
  intro h
  sorry

end range_of_m_l1134_113452


namespace yolkino_to_palkino_distance_l1134_113499

theorem yolkino_to_palkino_distance 
  (n : ℕ) 
  (digit_sum : ℕ → ℕ) 
  (h1 : ∀ k : ℕ, k ≤ n → digit_sum k + digit_sum (n - k) = 13) : 
  n = 49 := 
by 
  sorry

end yolkino_to_palkino_distance_l1134_113499


namespace sum_of_three_consecutive_divisible_by_three_l1134_113463

theorem sum_of_three_consecutive_divisible_by_three (n : ℕ) : ∃ k : ℕ, (n + (n + 1) + (n + 2)) = 3 * k := by
  sorry

end sum_of_three_consecutive_divisible_by_three_l1134_113463


namespace sqrt_sq_eq_abs_l1134_113421

theorem sqrt_sq_eq_abs (x : ℝ) : Real.sqrt (x^2) = |x| :=
sorry

end sqrt_sq_eq_abs_l1134_113421


namespace day_of_week_dec_26_l1134_113498

theorem day_of_week_dec_26 (nov_26_is_thu : true) : true :=
sorry

end day_of_week_dec_26_l1134_113498


namespace remainder_2345678901_div_101_l1134_113486

theorem remainder_2345678901_div_101 : 2345678901 % 101 = 12 :=
sorry

end remainder_2345678901_div_101_l1134_113486


namespace hex_351_is_849_l1134_113484

noncomputable def hex_to_decimal : ℕ := 1 * 16^0 + 5 * 16^1 + 3 * 16^2

-- The following statement is the core of the proof problem
theorem hex_351_is_849 : hex_to_decimal = 849 := by
  -- Here the proof steps would normally go
  sorry

end hex_351_is_849_l1134_113484


namespace sum_of_numbers_eq_8140_l1134_113479

def numbers : List ℤ := [1200, 1300, 1400, 1510, 1530, 1200]

theorem sum_of_numbers_eq_8140 : (numbers.sum = 8140) :=
by
  sorry

end sum_of_numbers_eq_8140_l1134_113479


namespace log_value_l1134_113425

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem log_value (x : ℝ) (h : log_base 3 (5 * x) = 3) : log_base x 125 = 3 / 2 :=
  by
  sorry

end log_value_l1134_113425


namespace shaded_area_correct_l1134_113480

-- Define points as vectors in the 2D plane.
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Define points K, L, M, J based on the given coordinates.
def K : Point := {x := 0, y := 0}
def L : Point := {x := 5, y := 0}
def M : Point := {x := 5, y := 6}
def J : Point := {x := 0, y := 6}

-- Define intersection point N based on the equations of lines.
def N : Point := {x := 2.5, y := 3}

-- Define the function to calculate area of a trapezoid.
def trapezoid_area (b1 b2 h : ℝ) : ℝ :=
  0.5 * (b1 + b2) * h

-- Define the function to calculate area of a triangle.
def triangle_area (b h : ℝ) : ℝ :=
  0.5 * b * h

-- Compute total shaded area according to the problem statement.
def shaded_area (K L M J N : Point) : ℝ :=
  trapezoid_area 5 2.5 3 + triangle_area 2.5 1

theorem shaded_area_correct : shaded_area K L M J N = 12.5 := by
  sorry

end shaded_area_correct_l1134_113480


namespace train_crossing_time_l1134_113478

-- Condition definitions
def length_train : ℝ := 100
def length_bridge : ℝ := 150
def speed_kmph : ℝ := 54
def speed_mps : ℝ := 15

-- Given the conditions, prove the time to cross the bridge is 16.67 seconds
theorem train_crossing_time :
  (100 + 150) / (54 * 1000 / 3600) = 16.67 := by sorry

end train_crossing_time_l1134_113478


namespace find_f_neg_five_half_l1134_113443

noncomputable def f (x : ℝ) : ℝ :=
if 0 ≤ x ∧ x ≤ 1 then 2 * x * (1 - x)
else if 0 ≤ x + 2 ∧ x + 2 ≤ 1 then 2 * (x + 2) * (1 - (x + 2))
     else -2 * abs x * (1 - abs x)

theorem find_f_neg_five_half (x : ℝ) (h_odd : ∀ x, f (-x) = -f x) (h_period : ∀ x, f (x + 2) = f x)
  (h_def : ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = 2 * x * (1 - x)) : 
  f (-5 / 2) = -1 / 2 :=
  by sorry

end find_f_neg_five_half_l1134_113443


namespace weight_of_b_l1134_113472

theorem weight_of_b (A B C : ℝ) 
  (h1 : A + B + C = 129)
  (h2 : A + B = 96)
  (h3 : B + C = 84) : 
  B = 51 :=
sorry

end weight_of_b_l1134_113472


namespace number_div_0_04_eq_200_9_l1134_113455

theorem number_div_0_04_eq_200_9 (n : ℝ) (h : n / 0.04 = 200.9) : n = 8.036 :=
sorry

end number_div_0_04_eq_200_9_l1134_113455


namespace slope_angle_of_tangent_line_expx_at_0_l1134_113493

theorem slope_angle_of_tangent_line_expx_at_0 :
  let f := fun x : ℝ => Real.exp x 
  let f' := fun x : ℝ => Real.exp x
  ∀ x : ℝ, f' x = Real.exp x → 
  (∃ α : ℝ, Real.tan α = 1) →
  α = Real.pi / 4 :=
by
  intros f f' h_deriv h_slope
  sorry

end slope_angle_of_tangent_line_expx_at_0_l1134_113493


namespace min_value_of_f_l1134_113476

open Real

noncomputable def f (x : ℝ) := x + 1 / (x - 2)

theorem min_value_of_f : ∃ x : ℝ, x > 2 ∧ ∀ y : ℝ, y > 2 → f y ≥ f 3 := by
  sorry

end min_value_of_f_l1134_113476


namespace maximize_wind_power_l1134_113456

variable {C S ρ v_0 : ℝ}

theorem maximize_wind_power : 
  ∃ v : ℝ, (∀ (v' : ℝ),
           let F := (C * S * ρ * (v_0 - v)^2) / 2;
           let N := F * v;
           let N' := (C * S * ρ / 2) * (v_0^2 - 4 * v_0 * v + 3 * v^2);
           N' = 0
         → N ≤ (C * S * ρ / 2) * (v_0^2 * (v_0/3) - 2 * v_0 * (v_0/3)^2 + (v_0/3)^3)) ∧ v = v_0 / 3 :=
by sorry

end maximize_wind_power_l1134_113456


namespace regular_polygon_sides_l1134_113490

theorem regular_polygon_sides (h : ∀ n : ℕ, n ≥ 3 → (total_internal_angle_sum / n) = 150) :
    n = 12 := by
  sorry

end regular_polygon_sides_l1134_113490


namespace skittles_problem_l1134_113426

def initial_skittles : ℕ := 76
def shared_skittles : ℕ := 72
def final_skittles (initial shared : ℕ) : ℕ := initial - shared

theorem skittles_problem : final_skittles initial_skittles shared_skittles = 4 := by
  sorry

end skittles_problem_l1134_113426


namespace range_of_k_l1134_113415

noncomputable def f (k : ℝ) (x : ℝ) := (Real.exp x) / (x^2) + 2 * k * Real.log x - k * x

theorem range_of_k (k : ℝ) (h₁ : ∀ x > 0, (deriv (f k) x = 0) → x = 2) : k < Real.exp 2 / 4 :=
by
  sorry

end range_of_k_l1134_113415


namespace factor_polynomial_l1134_113487

theorem factor_polynomial :
  (x : ℝ) → (x^2 - 6*x + 9 - 64*x^4) = (-8*x^2 + x - 3) * (8*x^2 + x - 3) :=
by
  intro x
  sorry

end factor_polynomial_l1134_113487


namespace C_investment_l1134_113435

theorem C_investment (A B total_profit A_share : ℝ) (x : ℝ) :
  A = 6300 → B = 4200 → total_profit = 12600 → A_share = 3780 →
  (A / (A + B + x) = A_share / total_profit) → x = 10500 :=
by
  intros hA hB h_total_profit h_A_share h_ratio
  sorry

end C_investment_l1134_113435


namespace four_machines_save_11_hours_l1134_113467

-- Define the conditions
def three_machines_complete_order_in_44_hours := 3 * (1 / (3 * 44)) * 44 = 1

def additional_machine_reduces_time (T : ℝ) := 4 * (1 / (3 * 44)) * T = 1

-- Define the theorem to prove the number of hours saved
theorem four_machines_save_11_hours : 
  (∃ T : ℝ, additional_machine_reduces_time T ∧ three_machines_complete_order_in_44_hours) → 
  44 - 33 = 11 :=
by
  sorry

end four_machines_save_11_hours_l1134_113467


namespace employees_count_l1134_113449

theorem employees_count (n : ℕ) (avg_salary : ℝ) (manager_salary : ℝ)
  (new_avg_salary : ℝ) (total_employees_with_manager : ℝ) : 
  avg_salary = 1500 → 
  manager_salary = 3600 → 
  new_avg_salary = avg_salary + 100 → 
  total_employees_with_manager = (n + 1) * 1600 → 
  (n * avg_salary + manager_salary) / total_employees_with_manager = new_avg_salary →
  n = 20 := by
  intros
  sorry

end employees_count_l1134_113449


namespace distance_to_destination_l1134_113401

-- Conditions
def Speed : ℝ := 65 -- speed in km/hr
def Time : ℝ := 3   -- time in hours

-- Question to prove
theorem distance_to_destination : Speed * Time = 195 := by
  sorry

end distance_to_destination_l1134_113401


namespace machine_a_produces_6_sprockets_per_hour_l1134_113492

theorem machine_a_produces_6_sprockets_per_hour : 
  ∀ (A G T : ℝ), 
  (660 = A * (T + 10)) → 
  (660 = G * T) → 
  (G = 1.10 * A) → 
  A = 6 := 
by
  intros A G T h1 h2 h3
  sorry

end machine_a_produces_6_sprockets_per_hour_l1134_113492


namespace interval_of_increase_l1134_113408

noncomputable def f (x : ℝ) : ℝ :=
  -abs x

theorem interval_of_increase :
  ∀ x, f x ≤ f (x + 1) ↔ x ≤ 0 := by
  sorry

end interval_of_increase_l1134_113408


namespace negation_of_p_l1134_113470

-- Define the proposition p
def p : Prop := ∀ x : ℝ, Real.exp x > Real.log x

-- Define the negation of p
def neg_p : Prop := ∃ x : ℝ, Real.exp x ≤ Real.log x

-- The statement we want to prove
theorem negation_of_p : ¬p ↔ neg_p :=
by sorry

end negation_of_p_l1134_113470


namespace range_of_x_l1134_113433

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := |x - 4| + |x - a|

theorem range_of_x (a : ℝ) (h1 : a > 1) (h2 : ∀ x : ℝ, f x a ≥ |a - 4|) (h3 : |a - 4| = 3) :
  { x : ℝ | f x a ≤ 5 } = { x : ℝ | 3 ≤ x ∧ x ≤ 8 } := 
sorry

end range_of_x_l1134_113433


namespace susan_homework_start_time_l1134_113488

def start_time_homework (finish_time : ℕ) (homework_duration : ℕ) (interval_duration : ℕ) : ℕ :=
  finish_time - homework_duration - interval_duration

theorem susan_homework_start_time :
  let finish_time : ℕ := 16 * 60 -- 4:00 p.m. in minutes
  let homework_duration : ℕ := 96 -- Homework duration in minutes
  let interval_duration : ℕ := 25 -- Interval between homework finish and practice in minutes
  start_time_homework finish_time homework_duration interval_duration = 13 * 60 + 59 := -- 13:59 in minutes
by
  sorry

end susan_homework_start_time_l1134_113488


namespace long_furred_brown_dogs_l1134_113448

-- Definitions based on given conditions
def T : ℕ := 45
def L : ℕ := 36
def B : ℕ := 27
def N : ℕ := 8

-- The number of long-furred brown dogs (LB) that needs to be proved
def LB : ℕ := 26

-- Lean 4 statement to prove LB
theorem long_furred_brown_dogs :
  L + B - LB = T - N :=
by 
  unfold T L B N LB -- we unfold definitions to simplify the theorem
  sorry

end long_furred_brown_dogs_l1134_113448


namespace abs_ineq_one_abs_ineq_two_l1134_113464

-- First proof problem: |x-1| + |x+3| < 6 implies -4 < x < 2
theorem abs_ineq_one (x : ℝ) : |x - 1| + |x + 3| < 6 → -4 < x ∧ x < 2 :=
by
  sorry

-- Second proof problem: 1 < |3x-2| < 4 implies -2/3 ≤ x < 1/3 or 1 < x ≤ 2
theorem abs_ineq_two (x : ℝ) : 1 < |3 * x - 2| ∧ |3 * x - 2| < 4 → (-2/3) ≤ x ∧ x < (1/3) ∨ 1 < x ∧ x ≤ 2 :=
by
  sorry

end abs_ineq_one_abs_ineq_two_l1134_113464


namespace decimal_to_base_five_l1134_113471

theorem decimal_to_base_five : 
  (2 * 5^3 + 1 * 5^1 + 0 * 5^2 + 0 * 5^0 = 255) := 
by
  sorry

end decimal_to_base_five_l1134_113471


namespace cristine_final_lemons_l1134_113468

def cristine_lemons_initial : ℕ := 12
def cristine_lemons_given_to_neighbor : ℕ := 1 / 4 * cristine_lemons_initial
def cristine_lemons_left_after_giving : ℕ := cristine_lemons_initial - cristine_lemons_given_to_neighbor
def cristine_lemons_exchanged_for_oranges : ℕ := 1 / 3 * cristine_lemons_left_after_giving
def cristine_lemons_left_after_exchange : ℕ := cristine_lemons_left_after_giving - cristine_lemons_exchanged_for_oranges

theorem cristine_final_lemons : cristine_lemons_left_after_exchange = 6 :=
by
  sorry

end cristine_final_lemons_l1134_113468


namespace find_f_of_2_l1134_113494

def f (x : ℝ) (a b : ℝ) : ℝ := x^3 + a * x^2 + b * x + a^2

theorem find_f_of_2 (a b : ℝ)
  (h1 : 3 + 2 * a + b = 0)
  (h2 : 1 + a + b + a^2 = 10)
  (ha : a = 4)
  (hb : b = -11) :
  f 2 a b = 18 := by {
  -- We assume the values of a and b provided by the user as the correct pair.
  sorry
}

end find_f_of_2_l1134_113494


namespace wrapping_paper_area_l1134_113447

variable (a b h w : ℝ) (a_gt_b : a > b)

theorem wrapping_paper_area : 
  ∃ total_area, total_area = 4 * (a * b + a * w + b * w + w ^ 2) :=
by
  sorry

end wrapping_paper_area_l1134_113447


namespace solve_for_y_l1134_113451

theorem solve_for_y (y : ℝ) : 4 * y + 6 * y = 450 - 10 * (y - 5) → y = 25 :=
by
  sorry

end solve_for_y_l1134_113451


namespace david_marks_in_physics_l1134_113434

theorem david_marks_in_physics
  (marks_english : ℤ)
  (marks_math : ℤ)
  (marks_chemistry : ℤ)
  (marks_biology : ℤ)
  (average_marks : ℚ)
  (number_of_subjects : ℤ)
  (h_english : marks_english = 96)
  (h_math : marks_math = 98)
  (h_chemistry : marks_chemistry = 100)
  (h_biology : marks_biology = 98)
  (h_average : average_marks = 98.2)
  (h_subjects : number_of_subjects = 5) : 
  ∃ (marks_physics : ℤ), marks_physics = 99 := 
by {
  sorry
}

end david_marks_in_physics_l1134_113434


namespace remaining_money_after_shopping_l1134_113475

theorem remaining_money_after_shopping (initial_money : ℝ) (percentage_spent : ℝ) (final_amount : ℝ) :
  initial_money = 1200 → percentage_spent = 0.30 → final_amount = initial_money - (percentage_spent * initial_money) → final_amount = 840 :=
by
  intros h_initial h_percentage h_final
  sorry

end remaining_money_after_shopping_l1134_113475


namespace jack_marathon_time_l1134_113444

theorem jack_marathon_time :
  ∀ {marathon_distance : ℝ} {jill_time : ℝ} {speed_ratio : ℝ},
    marathon_distance = 40 → 
    jill_time = 4 → 
    speed_ratio = 0.888888888888889 → 
    (marathon_distance / (speed_ratio * (marathon_distance / jill_time))) = 4.5 :=
by
  intros marathon_distance jill_time speed_ratio h1 h2 h3
  rw [h1, h2, h3]
  sorry

end jack_marathon_time_l1134_113444


namespace cheryl_found_more_eggs_l1134_113459

theorem cheryl_found_more_eggs :
  let kevin_eggs := 5
  let bonnie_eggs := 13
  let george_eggs := 9
  let cheryl_eggs := 56
  cheryl_eggs - (kevin_eggs + bonnie_eggs + george_eggs) = 29 := by
  sorry

end cheryl_found_more_eggs_l1134_113459


namespace angle_measure_l1134_113473

theorem angle_measure (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 := by
  sorry

end angle_measure_l1134_113473


namespace calculate_area_correct_l1134_113400

-- Define the side length of the square
def side_length : ℝ := 5

-- Define the rotation angles in degrees
def rotation_angles : List ℝ := [0, 30, 45, 60]

-- Define the area calculation function (to be implemented)
def calculate_overlap_area (s : ℝ) (angles : List ℝ) : ℝ := sorry

-- Define the proof that the calculated area is equal to 123.475
theorem calculate_area_correct : calculate_overlap_area side_length rotation_angles = 123.475 :=
by
  sorry

end calculate_area_correct_l1134_113400


namespace ellipse_foci_distance_l1134_113496

noncomputable def distance_between_foci (a b : ℝ) : ℝ := 
  Real.sqrt (a^2 - b^2)

theorem ellipse_foci_distance :
  ∀ (a b : ℝ), a = 6 → b = 3 → distance_between_foci a b = 3 * Real.sqrt 3 :=
by
  intros a b h_a h_b
  rw [h_a, h_b]
  simp [distance_between_foci]
  sorry

end ellipse_foci_distance_l1134_113496


namespace least_whole_number_clock_equiv_l1134_113427

theorem least_whole_number_clock_equiv (h : ℕ) (h_gt_10 : h > 10) : 
  ∃ k, k = 12 ∧ (h^2 - h) % 12 = 0 ∧ h = 12 :=
by 
  sorry

end least_whole_number_clock_equiv_l1134_113427


namespace tom_spending_is_correct_l1134_113422

-- Conditions
def cost_per_square_foot : ℕ := 5
def square_feet_per_seat : ℕ := 12
def number_of_seats : ℕ := 500
def construction_multiplier : ℕ := 2
def partner_contribution_ratio : ℚ := 0.40

-- Calculate and verify Tom's spending
def total_square_footage := number_of_seats * square_feet_per_seat
def land_cost := total_square_footage * cost_per_square_foot
def construction_cost := construction_multiplier * land_cost
def total_cost := land_cost + construction_cost
def partner_contribution := partner_contribution_ratio * total_cost
def tom_spending := (1 - partner_contribution_ratio) * total_cost

theorem tom_spending_is_correct : tom_spending = 54000 := 
by 
    -- The theorems calculate specific values 
    sorry

end tom_spending_is_correct_l1134_113422


namespace neighbors_receive_28_mangoes_l1134_113441

/-- 
  Mr. Wong harvested 560 mangoes. He sold half, gave 50 to his family,
  and divided the remaining mangoes equally among 8 neighbors.
  Each neighbor should receive 28 mangoes.
-/
theorem neighbors_receive_28_mangoes : 
  ∀ (initial : ℕ) (sold : ℕ) (given : ℕ) (neighbors : ℕ), 
  initial = 560 → 
  sold = initial / 2 → 
  given = 50 → 
  neighbors = 8 → 
  (initial - sold - given) / neighbors = 28 := 
by 
  intros initial sold given neighbors
  sorry

end neighbors_receive_28_mangoes_l1134_113441


namespace selection_methods_count_l1134_113469

/-- Consider a school with 16 teachers, divided into four departments (First grade, Second grade, Third grade, and Administrative department), with 4 teachers each. 
We need to select 3 leaders such that not all leaders are from the same department and at least one leader is from the Administrative department. 
Prove that the number of different selection methods that satisfy these conditions is 336. -/
theorem selection_methods_count :
  let num_teachers := 16
  let teachers_per_department := 4
  ∃ (choose : ℕ → ℕ → ℕ), 
  choose num_teachers 3 = 336 :=
  sorry

end selection_methods_count_l1134_113469


namespace sin_double_angle_plus_pi_over_six_l1134_113430

variable (θ : ℝ)
variable (h : 7 * Real.sqrt 3 * Real.sin θ = 1 + 7 * Real.cos θ)

theorem sin_double_angle_plus_pi_over_six :
  Real.sin (2 * θ + π / 6) = 97 / 98 :=
by
  sorry

end sin_double_angle_plus_pi_over_six_l1134_113430


namespace building_height_l1134_113428

noncomputable def height_of_building (H_f L_f L_b : ℝ) : ℝ :=
  (H_f * L_b) / L_f

theorem building_height (H_f L_f L_b H_b : ℝ)
  (H_f_val : H_f = 17.5)
  (L_f_val : L_f = 40.25)
  (L_b_val : L_b = 28.75)
  (H_b_val : H_b = 12.4375) :
  height_of_building H_f L_f L_b = H_b := by
  rw [H_f_val, L_f_val, L_b_val, H_b_val]
  -- sorry to skip the proof
  sorry

end building_height_l1134_113428


namespace max_value_of_a_l1134_113420

theorem max_value_of_a (a b c d : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_pos_d : 0 < d)
  (h1 : a < 3 * b) (h2 : b < 2 * c) (h3 : c < 5 * d) (h4 : d < 150) : a ≤ 4460 :=
by
  sorry

end max_value_of_a_l1134_113420


namespace decimal_to_fraction_l1134_113432

theorem decimal_to_fraction (h : 0.36 = 36 / 100): (36 / 100 = 9 / 25) := by
    sorry

end decimal_to_fraction_l1134_113432


namespace circle_area_with_radius_three_is_9pi_l1134_113474

theorem circle_area_with_radius_three_is_9pi (r : ℝ) (h : r = 3) : Real.pi * r^2 = 9 * Real.pi := by
  sorry

end circle_area_with_radius_three_is_9pi_l1134_113474


namespace basic_computer_price_l1134_113482

theorem basic_computer_price (C P : ℝ) 
  (h1 : C + P = 2500)
  (h2 : P = 1 / 8 * ((C + 500) + P)) :
  C = 2125 :=
by
  sorry

end basic_computer_price_l1134_113482


namespace rahuls_share_l1134_113497

theorem rahuls_share (total_payment : ℝ) (rahul_days : ℝ) (rajesh_days : ℝ) (rahul_share : ℝ)
  (rahul_work_one_day : rahul_days > 0) (rajesh_work_one_day : rajesh_days > 0)
  (total_payment_eq : total_payment = 105) 
  (rahul_days_eq : rahul_days = 3) 
  (rajesh_days_eq : rajesh_days = 2) :
  rahul_share = 42 := 
by
  sorry

end rahuls_share_l1134_113497


namespace g_g_3_eq_3606651_l1134_113489

def g (x: ℤ) : ℤ := 3 * x^3 + 3 * x^2 - x + 1

theorem g_g_3_eq_3606651 : g (g 3) = 3606651 := 
by {
  sorry
}

end g_g_3_eq_3606651_l1134_113489


namespace solve_cubic_equation_l1134_113412

theorem solve_cubic_equation : 
  ∃ x y : ℕ, 0 < x ∧ 0 < y ∧ x^3 - y^3 = 999 ∧ (x, y) = (12, 9) ∨ (x, y) = (10, 1) := 
  by
  sorry

end solve_cubic_equation_l1134_113412


namespace rate_calculation_l1134_113461

noncomputable def rate_per_sq_meter
  (lawn_length : ℝ) (lawn_breadth : ℝ)
  (road_width : ℝ) (total_cost : ℝ) : ℝ :=
  let area_road_1 := road_width * lawn_breadth
  let area_road_2 := road_width * lawn_length
  let area_intersection := road_width * road_width
  let total_area_roads := (area_road_1 + area_road_2) - area_intersection
  total_cost / total_area_roads

theorem rate_calculation :
  rate_per_sq_meter 100 60 10 4500 = 3 := by
  sorry

end rate_calculation_l1134_113461


namespace incorrect_statement_D_l1134_113403

theorem incorrect_statement_D :
  (∃ x : ℝ, x ^ 3 = -64 ∧ x = -4) ∧
  (∃ y : ℝ, y ^ 2 = 49 ∧ y = 7) ∧
  (∃ z : ℝ, z ^ 3 = 1 / 27 ∧ z = 1 / 3) ∧
  (∀ w : ℝ, w ^ 2 = 1 / 16 → w = 1 / 4 ∨ w = -1 / 4)
  → ¬ (∀ w : ℝ, w ^ 2 = 1 / 16 → w = 1 / 4) :=
by
  sorry

end incorrect_statement_D_l1134_113403


namespace exponent_multiplication_l1134_113406

theorem exponent_multiplication (a : ℝ) : (a^3) * (a^2) = a^5 := 
by
  -- Using the property of exponents: a^m * a^n = a^(m + n)
  sorry

end exponent_multiplication_l1134_113406


namespace bathroom_cleaning_time_ratio_l1134_113454

noncomputable def hourlyRate : ℝ := 5
noncomputable def vacuumingHours : ℝ := 2 -- per session
noncomputable def vacuumingSessions : ℕ := 2
noncomputable def washingDishesTime : ℝ := 0.5
noncomputable def totalEarnings : ℝ := 30

theorem bathroom_cleaning_time_ratio :
  let vacuumingEarnings := vacuumingHours * vacuumingSessions * hourlyRate
  let washingDishesEarnings := washingDishesTime * hourlyRate
  let knownEarnings := vacuumingEarnings + washingDishesEarnings
  let bathroomEarnings := totalEarnings - knownEarnings
  let bathroomCleaningTime := bathroomEarnings / hourlyRate
  bathroomCleaningTime / washingDishesTime = 3 := 
by
  sorry

end bathroom_cleaning_time_ratio_l1134_113454


namespace average_height_students_l1134_113402

/-- Given the average heights of female and male students, and the ratio of men to women, the average height -/
theorem average_height_students
  (avg_female_height : ℕ)
  (avg_male_height : ℕ)
  (ratio_men_women : ℕ)
  (h1 : avg_female_height = 170)
  (h2 : avg_male_height = 182)
  (h3 : ratio_men_women = 5) :
  (avg_female_height + 5 * avg_male_height) / (1 + 5) = 180 :=
by
  sorry

end average_height_students_l1134_113402


namespace thomas_annual_insurance_cost_l1134_113466

theorem thomas_annual_insurance_cost (total_cost : ℕ) (number_of_years : ℕ) 
  (h1 : total_cost = 40000) (h2 : number_of_years = 10) : 
  total_cost / number_of_years = 4000 := 
by 
  sorry

end thomas_annual_insurance_cost_l1134_113466


namespace sales_tax_difference_l1134_113416

theorem sales_tax_difference (price : ℝ) (rate1 rate2 : ℝ) : 
  rate1 = 0.085 → rate2 = 0.07 → price = 50 → 
  (price * rate1 - price * rate2) = 0.75 := 
by 
  intros h_rate1 h_rate2 h_price
  rw [h_rate1, h_rate2, h_price] 
  simp
  sorry

end sales_tax_difference_l1134_113416


namespace fraction_A_BC_l1134_113411

-- Definitions for amounts A, B, C and the total T
variable (T : ℝ) (A : ℝ) (B : ℝ) (C : ℝ)

-- Given conditions
def conditions : Prop :=
  T = 300 ∧
  A = 120.00000000000001 ∧
  B = (6 / 9) * (A + C) ∧
  A + B + C = T

-- The fraction of the amount A gets compared to B and C together
def fraction (x : ℝ) : Prop :=
  A = x * (B + C)

-- The proof goal
theorem fraction_A_BC : conditions T A B C → fraction A B C (2 / 3) :=
by
  sorry

end fraction_A_BC_l1134_113411


namespace shortest_distance_between_semicircles_l1134_113404

theorem shortest_distance_between_semicircles
  (ABCD : Type)
  (AD : ℝ)
  (shaded_area : ℝ)
  (is_rectangle : true)
  (AD_eq_10 : AD = 10)
  (shaded_area_eq_100 : shaded_area = 100) :
  ∃ d : ℝ, d = 2.5 * Real.pi :=
by
  sorry

end shortest_distance_between_semicircles_l1134_113404


namespace a0_a1_consecutive_l1134_113462

variable (a : ℕ → ℤ)
variable (cond : ∀ i ≥ 2, a i = 2 * a (i - 1) - a (i - 2) ∨ a i = 2 * a (i - 2) - a (i - 1))
variable (consec : |a 2024 - a 2023| = 1)

theorem a0_a1_consecutive :
  |a 1 - a 0| = 1 :=
by
  -- Proof skipped
  sorry

end a0_a1_consecutive_l1134_113462


namespace inequality_B_l1134_113442

variable {x y : ℝ}

theorem inequality_B (hx : 0 < x) (hy : 0 < y) (hxy : x > y) : x + 1 / (2 * y) > y + 1 / x :=
sorry

end inequality_B_l1134_113442


namespace original_number_is_842_l1134_113410

theorem original_number_is_842 (x y z : ℕ) (h1 : x * z = y^2)
  (h2 : 100 * z + x = 100 * x + z - 594)
  (h3 : 10 * z + y = 10 * y + z - 18)
  (hx : x = 8) (hy : y = 4) (hz : z = 2) :
  100 * x + 10 * y + z = 842 :=
by
  sorry

end original_number_is_842_l1134_113410


namespace parabola_2_second_intersection_x_l1134_113409

-- Definitions of the conditions in the problem
def parabola_1_intersects : Prop := 
  (∀ x : ℝ, (x = 10 ∨ x = 13) → (∃ y : ℝ, (x, y) ∈ ({p | p = (10, 0)} ∪ {p | p = (13, 0)})))

def parabola_2_intersects : Prop := 
  (∃ x : ℝ, x = 13)

def vertex_bisects_segment : Prop := 
  (∃ a : ℝ, 2 * 11.5 = a)

-- The theorem we want to prove
theorem parabola_2_second_intersection_x : 
  parabola_1_intersects ∧ parabola_2_intersects ∧ vertex_bisects_segment → 
  (∃ t : ℝ, t = 33) := 
  by
  sorry

end parabola_2_second_intersection_x_l1134_113409


namespace sampling_methods_correct_l1134_113491

-- Define the conditions given in the problem.
def total_students := 200
def method_1_is_simple_random := true
def method_2_is_systematic := true

-- The proof problem statement, no proof is required.
theorem sampling_methods_correct :
  (method_1_is_simple_random = true) ∧
  (method_2_is_systematic = true) :=
by
  -- using conditions defined above, we state the theorem we need to prove
  sorry

end sampling_methods_correct_l1134_113491


namespace max_value_10x_plus_3y_plus_12z_l1134_113439

theorem max_value_10x_plus_3y_plus_12z (x y z : ℝ) 
  (h1 : 9 * x^2 + 4 * y^2 + 25 * z^2 = 1) 
  (h2 : z = 2 * y) : 
  10 * x + 3 * y + 12 * z ≤ Real.sqrt 253 :=
sorry

end max_value_10x_plus_3y_plus_12z_l1134_113439


namespace find_radius_squared_l1134_113453

theorem find_radius_squared (r : ℝ) (AB_len CD_len BP : ℝ) (angle_APD : ℝ) (h1 : AB_len = 12)
    (h2 : CD_len = 9) (h3 : BP = 10) (h4 : angle_APD = 60) : r^2 = 111 := by
  have AB_len := h1
  have CD_len := h2
  have BP := h3
  have angle_APD := h4
  sorry

end find_radius_squared_l1134_113453
