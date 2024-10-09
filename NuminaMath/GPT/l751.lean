import Mathlib

namespace centers_collinear_l751_75161

theorem centers_collinear (k : ℝ) (hk : k ≠ -1) :
    ∀ p : ℝ × ℝ, p = (-k, -2*k-5) → (2*p.1 - p.2 - 5 = 0) :=
by
  sorry

end centers_collinear_l751_75161


namespace billy_restaurant_total_payment_l751_75187

noncomputable def cost_of_meal
  (adult_count child_count : ℕ)
  (adult_cost child_cost : ℕ) : ℕ :=
  adult_count * adult_cost + child_count * child_cost

noncomputable def cost_of_dessert
  (total_people : ℕ)
  (dessert_cost : ℕ) : ℕ :=
  total_people * dessert_cost

noncomputable def total_cost_before_discount
  (adult_count child_count : ℕ)
  (adult_cost child_cost dessert_cost : ℕ) : ℕ :=
  (cost_of_meal adult_count child_count adult_cost child_cost) +
  (cost_of_dessert (adult_count + child_count) dessert_cost)

noncomputable def discount_amount
  (total : ℕ)
  (discount_rate : ℝ) : ℝ :=
  total * discount_rate

noncomputable def total_amount_to_pay
  (total : ℕ)
  (discount : ℝ) : ℝ :=
  total - discount

theorem billy_restaurant_total_payment :
  total_amount_to_pay
  (total_cost_before_discount 2 5 7 3 2)
  (discount_amount (total_cost_before_discount 2 5 7 3 2) 0.15) = 36.55 := by
  sorry

end billy_restaurant_total_payment_l751_75187


namespace range_of_fraction_l751_75194

-- Definition of the quadratic equation with roots within specified intervals
variables (a b : ℝ)
variables (x1 x2 : ℝ)
variables (h_distinct_roots : x1 ≠ x2)
variables (h_interval_x1 : 0 < x1 ∧ x1 < 1)
variables (h_interval_x2 : 1 < x2 ∧ x2 < 2)
variables (h_quadratic : ∀ x : ℝ, x^2 + a * x + 2 * b - 2 = 0)

-- Prove range of expression
theorem range_of_fraction (a b : ℝ)
  (x1 x2 h_distinct_roots : ℝ) (h_interval_x1 : 0 < x1 ∧ x1 < 1)
  (h_interval_x2 : 1 < x2 ∧ x2 < 2)
  (h_quadratic : ∀ x, x^2 + a * x + 2 * b - 2 = 0) :
  (1/2 < (b - 4) / (a - 1)) ∧ ((b - 4) / (a - 1) < 3/2) :=
by
  -- proof placeholder
  sorry

end range_of_fraction_l751_75194


namespace find_xy_l751_75100

theorem find_xy (x y : ℝ) (h1 : x + y = 5) (h2 : x^3 + y^3 = 125) : x * y = 0 :=
by
  sorry

end find_xy_l751_75100


namespace division_remainder_l751_75138

theorem division_remainder :
  ∃ (r : ℝ), ∀ (z : ℝ), (4 * z^3 - 5 * z^2 - 17 * z + 4) = (4 * z + 6) * (z^2 - 4 * z + 1/2) + r ∧ r = 1 :=
sorry

end division_remainder_l751_75138


namespace paint_required_for_frame_l751_75124

theorem paint_required_for_frame :
  ∀ (width height thickness : ℕ) 
    (coverage : ℚ),
  width = 6 →
  height = 9 →
  thickness = 1 →
  coverage = 5 →
  (width * height - (width - 2 * thickness) * (height - 2 * thickness) + 2 * width * thickness + 2 * height * thickness) / coverage = 11.2 :=
by
  intros
  sorry

end paint_required_for_frame_l751_75124


namespace tournament_participants_l751_75157

theorem tournament_participants (n : ℕ) (h : (n * (n - 1)) / 2 = 171) : n = 19 :=
by
  sorry

end tournament_participants_l751_75157


namespace smallest_value_div_by_13_l751_75131

theorem smallest_value_div_by_13 : 
  ∃ (A B : ℕ), 
    (0 ≤ A ∧ A ≤ 9) ∧ (0 ≤ B ∧ B ≤ 9) ∧ 
    A ≠ B ∧ 
    1001 * A + 110 * B = 1771 ∧ 
    (1001 * A + 110 * B) % 13 = 0 :=
by
  sorry

end smallest_value_div_by_13_l751_75131


namespace find_values_l751_75140

def isInInterval (x : ℝ) : Prop := -2 ≤ x ∧ x ≤ 2

theorem find_values 
  (a b c d e : ℝ)
  (ha : isInInterval a) 
  (hb : isInInterval b) 
  (hc : isInInterval c) 
  (hd : isInInterval d)
  (he : isInInterval e)
  (h1 : a + b + c + d + e = 0)
  (h2 : a^3 + b^3 + c^3 + d^3 + e^3 = 0)
  (h3 : a^5 + b^5 + c^5 + d^5 + e^5 = 10) : 
  (a = 2 ∧ b = (Real.sqrt 5 - 1) / 2 ∧ c = (Real.sqrt 5 - 1) / 2 ∧ d = - (1 + Real.sqrt 5) / 2 ∧ e = - (1 + Real.sqrt 5) / 2) ∨
  (a = (Real.sqrt 5 - 1) / 2 ∧ b = 2 ∧ c = (Real.sqrt 5 - 1) / 2 ∧ d = - (1 + Real.sqrt 5) / 2 ∧ e = - (1 + Real.sqrt 5) / 2) ∨
  (a = (Real.sqrt 5 - 1) / 2 ∧ b = (Real.sqrt 5 - 1) / 2 ∧ c = 2 ∧ d = - (1 + Real.sqrt 5) / 2 ∧ e = - (1 + Real.sqrt 5) / 2) ∨
  (a = (Real.sqrt 5 - 1) / 2 ∧ b = (Real.sqrt 5 - 1) / 2 ∧ c = - (1 + Real.sqrt 5) / 2 ∧ d = 2 ∧ e = - (1 + Real.sqrt 5) / 2) ∨
  (a = (Real.sqrt 5 - 1) / 2 ∧ b = (Real.sqrt 5 - 1) / 2 ∧ c = - (1 + Real.sqrt 5) / 2 ∧ d = - (1 + Real.sqrt 5) / 2 ∧ e = 2) :=
sorry

end find_values_l751_75140


namespace kevin_total_distance_l751_75123

noncomputable def kevin_hop_total_distance_after_seven_leaps : ℚ :=
  let a := (1 / 4 : ℚ)
  let r := (3 / 4 : ℚ)
  let n := 7
  a * (1 - r^n) / (1 - r)

theorem kevin_total_distance (total_distance : ℚ) :
  total_distance = kevin_hop_total_distance_after_seven_leaps → 
  total_distance = 14197 / 16384 := by
  intro h
  sorry

end kevin_total_distance_l751_75123


namespace xiao_wang_parts_processed_l751_75144

-- Definitions for the processing rates and conditions
def xiao_wang_rate := 15 -- parts per hour
def xiao_wang_max_continuous_hours := 2
def xiao_wang_break_hours := 1

def xiao_li_rate := 12 -- parts per hour

-- Constants for the problem setup
def xiao_wang_process_time := 4 -- hours including breaks after first cycle
def xiao_li_process_time := 5 -- hours including no breaks

-- Total parts processed by both when they finish simultaneously
def parts_processed_when_finished_simultaneously := 60

theorem xiao_wang_parts_processed :
  (xiao_wang_rate * xiao_wang_max_continuous_hours) * (xiao_wang_process_time / 
  (xiao_wang_max_continuous_hours + xiao_wang_break_hours)) =
  parts_processed_when_finished_simultaneously :=
sorry

end xiao_wang_parts_processed_l751_75144


namespace portrait_is_in_Silver_l751_75143

def Gold_inscription (located_in : String → Prop) : Prop := located_in "Gold"
def Silver_inscription (located_in : String → Prop) : Prop := ¬located_in "Silver"
def Lead_inscription (located_in : String → Prop) : Prop := ¬located_in "Gold"

def is_true (inscription : Prop) : Prop := inscription
def is_false (inscription : Prop) : Prop := ¬inscription

noncomputable def portrait_in_Silver_Given_Statements : Prop :=
  ∃ located_in : String → Prop,
    (is_true (Gold_inscription located_in) ∨ is_true (Silver_inscription located_in) ∨ is_true (Lead_inscription located_in)) ∧
    (is_false (Gold_inscription located_in) ∨ is_false (Silver_inscription located_in) ∨ is_false (Lead_inscription located_in)) ∧
    located_in "Silver"

theorem portrait_is_in_Silver : portrait_in_Silver_Given_Statements :=
by {
    sorry
}

end portrait_is_in_Silver_l751_75143


namespace smallest_AAB_value_l751_75195

theorem smallest_AAB_value : ∃ (A B : ℕ), 1 ≤ A ∧ A ≤ 9 ∧ 1 ≤ B ∧ B ≤ 9 ∧ 110 * A + B = 8 * (10 * A + B) ∧ ¬ (A = B) ∧ 110 * A + B = 773 :=
by sorry

end smallest_AAB_value_l751_75195


namespace range_is_80_l751_75173

def dataSet : List ℕ := [60, 100, 80, 40, 20]

def minValue (l : List ℕ) : ℕ :=
  match l with
  | [] => 0
  | (x :: xs) => List.foldl min x xs

def maxValue (l : List ℕ) : ℕ :=
  match l with
  | [] => 0
  | (x :: xs) => List.foldl max x xs

def range (l : List ℕ) : ℕ :=
  maxValue l - minValue l

theorem range_is_80 : range dataSet = 80 :=
by
  sorry

end range_is_80_l751_75173


namespace polygon_sides_l751_75151

theorem polygon_sides (n : ℕ) 
    (h1 : (n-2) * 180 = 3 * 360 - 180) 
    (h2 : ∀ k, k > 2 → (k-2) * 180 = 180 * (k - 2)) 
    (h3 : 360 = 360) : n = 5 := 
by
  sorry

end polygon_sides_l751_75151


namespace sum_of_altitudes_of_triangle_l751_75112

-- Define the line equation as a condition
def line_eq (x y : ℝ) : Prop := 15 * x + 8 * y = 120

-- Define the triangle formed by the line with the coordinate axes
def forms_triangle_with_axes (x y : ℝ) : Prop := 
  line_eq x 0 ∧ line_eq 0 y

-- Prove the sum of the lengths of the altitudes is 511/17
theorem sum_of_altitudes_of_triangle : 
  ∃ x y : ℝ, forms_triangle_with_axes x y → 
  15 + 8 + (120 / 17) = 511 / 17 :=
by
  sorry

end sum_of_altitudes_of_triangle_l751_75112


namespace valid_combinations_l751_75115

theorem valid_combinations :
  ∀ (x y z : ℕ), 
  10 ≤ x ∧ x ≤ 20 → 
  10 ≤ y ∧ y ≤ 20 →
  10 ≤ z ∧ z ≤ 20 →
  3 * x^2 - y^2 - 7 * z = 99 →
  (x, y, z) = (15, 10, 12) ∨ (x, y, z) = (16, 12, 11) ∨ (x, y, z) = (18, 15, 13) := 
by
  intros x y z hx hy hz h
  sorry

end valid_combinations_l751_75115


namespace sum_of_ages_l751_75177

theorem sum_of_ages (J M R : ℕ) (hJ : J = 10) (hM : M = J - 3) (hR : R = J + 2) : M + R = 19 :=
by
  sorry

end sum_of_ages_l751_75177


namespace length_of_each_stone_l751_75162

theorem length_of_each_stone {L : ℝ} (hall_length hall_breadth : ℝ) (stone_breadth : ℝ) (num_stones : ℕ) (area_hall : ℝ) (area_stone : ℝ) :
  hall_length = 36 * 10 ∧ hall_breadth = 15 * 10 ∧ stone_breadth = 5 ∧ num_stones = 3600 ∧
  area_hall = hall_length * hall_breadth ∧ area_stone = L * stone_breadth ∧
  area_stone * num_stones = area_hall →
  L = 3 :=
by
  sorry

end length_of_each_stone_l751_75162


namespace salt_weight_l751_75106

theorem salt_weight {S : ℝ} (h1 : 16 + S = 46) : S = 30 :=
by
  sorry

end salt_weight_l751_75106


namespace find_two_sets_l751_75121

theorem find_two_sets :
  ∃ (a1 a2 a3 a4 a5 b1 b2 b3 b4 b5 : ℕ),
    a1 + a2 + a3 + a4 + a5 = a1 * a2 * a3 * a4 * a5 ∧
    b1 + b2 + b3 + b4 + b5 = b1 * b2 * b3 * b4 * b5 ∧
    (a1, a2, a3, a4, a5) ≠ (b1, b2, b3, b4, b5) := by
  sorry

end find_two_sets_l751_75121


namespace oil_bill_january_l751_75109

-- Declare the constants for January and February oil bills
variables (J F : ℝ)

-- State the conditions
def condition_1 : Prop := F / J = 3 / 2
def condition_2 : Prop := (F + 20) / J = 5 / 3

-- State the theorem based on the conditions and the target statement
theorem oil_bill_january (h1 : condition_1 F J) (h2 : condition_2 F J) : J = 120 :=
by
  sorry

end oil_bill_january_l751_75109


namespace initial_profit_price_reduction_for_target_profit_l751_75135

-- Define given conditions
def purchase_price : ℝ := 280
def initial_selling_price : ℝ := 360
def items_sold_per_month : ℕ := 60
def target_profit : ℝ := 7200
def increment_per_reduced_yuan : ℕ := 5

-- Problem 1: Prove the initial profit per month before the price reduction
theorem initial_profit : 
  items_sold_per_month * (initial_selling_price - purchase_price) = 4800 := by
sorry

-- Problem 2: Prove that reducing the price by 60 yuan achieves the target profit
theorem price_reduction_for_target_profit : 
  ∃ x : ℝ, 
    ((initial_selling_price - x) - purchase_price) * (items_sold_per_month + (increment_per_reduced_yuan * x)) = target_profit ∧
    x = 60 := by
sorry

end initial_profit_price_reduction_for_target_profit_l751_75135


namespace value_2_stddev_less_than_mean_l751_75185

theorem value_2_stddev_less_than_mean :
  let mean := 17.5
  let stddev := 2.5
  mean - 2 * stddev = 12.5 :=
by
  sorry

end value_2_stddev_less_than_mean_l751_75185


namespace classroom_student_count_l751_75118

theorem classroom_student_count (n : ℕ) (students_avg : ℕ) (teacher_age : ℕ) (combined_avg : ℕ) 
  (h1 : students_avg = 8) (h2 : teacher_age = 32) (h3 : combined_avg = 11) 
  (h4 : (8 * n + 32) / (n + 1) = 11) : n + 1 = 8 :=
by
  sorry

end classroom_student_count_l751_75118


namespace number_of_valid_N_count_valid_N_is_seven_l751_75155

theorem number_of_valid_N (N : ℕ) :  ∃ (m : ℕ), (m > 3) ∧ (m ∣ 48) ∧ (N = m - 3) := sorry

theorem count_valid_N_is_seven : 
  (∃ f : Fin 7 → ℕ, ∀ k, ∃ m, (m > 3) ∧ (m ∣ 48) ∧ (f k = m - 3)) ∧
  (∀ g : Fin 8 → ℕ, (∀ k, ∃ m, (m > 3) ∧ (m ∣ 48) ∧ (g k = m - 3)) → False) := sorry

end number_of_valid_N_count_valid_N_is_seven_l751_75155


namespace arithmetic_sequence_c_d_sum_l751_75116

theorem arithmetic_sequence_c_d_sum (c d : ℕ) 
  (h1 : 10 - 3 = 7) 
  (h2 : 17 - 10 = 7) 
  (h3 : c = 17 + 7) 
  (h4 : d = c + 7) 
  (h5 : d + 7 = 38) :
  c + d = 55 := 
by
  sorry

end arithmetic_sequence_c_d_sum_l751_75116


namespace eccentricity_of_ellipse_equation_of_ellipse_l751_75186

variable {a b : ℝ}
variable {x y : ℝ}

/-- Problem 1: Eccentricity of the given ellipse --/
theorem eccentricity_of_ellipse (ha : a = 2 * b) (hb0 : 0 < b) :
  ∃ e : ℝ, e = Real.sqrt (1 - (b / a) ^ 2) ∧ e = Real.sqrt 3 / 2 := by
  sorry

/-- Problem 2: Equation of the ellipse with respect to maximizing the area of triangle OMN --/
theorem equation_of_ellipse (ha : a = 2 * b) (hb0 : 0 < b) :
  ∃ l : ℝ → ℝ, (∃ k : ℝ, ∀ x, l x = k * x + 2) →
  ∀ x y : ℝ, (x^2 / (a^2) + y^2 / (b^2) = 1) →
  (∀ x' y' : ℝ, (x'^2 + 4 * y'^2 = 4 * b^2) ∧ y' = k * x' + 2) →
  (∃ a b : ℝ, a = 8 ∧ b = 2 ∧ x^2 / a + y^2 / b = 1) := by
  sorry

end eccentricity_of_ellipse_equation_of_ellipse_l751_75186


namespace LindaCandiesLeft_l751_75170

variable (initialCandies : ℝ)
variable (candiesGiven : ℝ)

theorem LindaCandiesLeft (h1 : initialCandies = 34.0) (h2 : candiesGiven = 28.0) : initialCandies - candiesGiven = 6.0 := by
  sorry

end LindaCandiesLeft_l751_75170


namespace ratio_of_sphere_surface_areas_l751_75148

noncomputable def inscribed_sphere_radius (a : ℝ) : ℝ := a / 2
noncomputable def circumscribed_sphere_radius (a : ℝ) : ℝ := a * (Real.sqrt 3) / 2
noncomputable def sphere_surface_area (r : ℝ) : ℝ := 4 * Real.pi * r^2

theorem ratio_of_sphere_surface_areas (a : ℝ) (h : 0 < a) : 
  (sphere_surface_area (circumscribed_sphere_radius a)) / (sphere_surface_area (inscribed_sphere_radius a)) = 3 :=
by
  sorry

end ratio_of_sphere_surface_areas_l751_75148


namespace wickets_before_last_match_l751_75153

theorem wickets_before_last_match (R W : ℝ) (h1 : R = 12.4 * W) (h2 : R + 26 = 12 * (W + 7)) :
  W = 145 := 
by 
  sorry

end wickets_before_last_match_l751_75153


namespace fixed_point_of_function_l751_75176

-- Definition: The function passes through a fixed point (a, b) for all real numbers k.
def passes_through_fixed_point (f : ℝ → ℝ) (a b : ℝ) := ∀ k : ℝ, f a = b

-- Given the function y = 9x^2 + 3kx - 6k, we aim to prove the fixed point is (2, 36).
theorem fixed_point_of_function : passes_through_fixed_point (fun x => 9 * x^2 + 3 * k * x - 6 * k) 2 36 := by
  sorry

end fixed_point_of_function_l751_75176


namespace next_tutoring_day_lcm_l751_75113

theorem next_tutoring_day_lcm :
  Nat.lcm (Nat.lcm (Nat.lcm 4 5) 6) 8 = 120 := by
  sorry

end next_tutoring_day_lcm_l751_75113


namespace find_x_y_l751_75104

theorem find_x_y 
  (x y : ℚ)
  (h1 : (x / 6) * 12 = 10)
  (h2 : (y / 4) * 8 = x) :
  x = 5 ∧ y = 2.5 :=
by
  sorry

end find_x_y_l751_75104


namespace arithmetic_sequence_number_of_terms_l751_75199

def arithmetic_sequence_terms_count (a d l : ℕ) : ℕ :=
  sorry

theorem arithmetic_sequence_number_of_terms :
  arithmetic_sequence_terms_count 13 3 73 = 21 :=
sorry

end arithmetic_sequence_number_of_terms_l751_75199


namespace min_value_frac_inv_l751_75122

theorem min_value_frac_inv (a b : ℝ) (h1: a > 0) (h2: b > 0) (h3: a + 3 * b = 2) : 
  (2 + Real.sqrt 3) ≤ (1 / a + 1 / b) :=
sorry

end min_value_frac_inv_l751_75122


namespace solution_to_equation_l751_75190

theorem solution_to_equation (x : ℝ) : x * (x - 2) = 2 * x ↔ (x = 0 ∨ x = 4) := by
  sorry

end solution_to_equation_l751_75190


namespace maximum_sphere_radius_squared_l751_75152

def cone_base_radius : ℝ := 4
def cone_height : ℝ := 10
def axes_intersection_distance_from_base : ℝ := 4

theorem maximum_sphere_radius_squared :
  let m : ℕ := 144
  let n : ℕ := 29
  m + n = 173 :=
by
  sorry

end maximum_sphere_radius_squared_l751_75152


namespace bob_overtime_pay_rate_l751_75103

theorem bob_overtime_pay_rate :
  let regular_pay_rate := 5
  let total_hours := (44, 48)
  let total_pay := 472
  let overtime_hours (hours : Nat) := max 0 (hours - 40)
  let regular_hours (hours : Nat) := min 40 hours
  let total_regular_hours := regular_hours 44 + regular_hours 48
  let total_regular_pay := total_regular_hours * regular_pay_rate
  let total_overtime_hours := overtime_hours 44 + overtime_hours 48
  let total_overtime_pay := total_pay - total_regular_pay
  let overtime_pay_rate := total_overtime_pay / total_overtime_hours
  overtime_pay_rate = 6 := by sorry

end bob_overtime_pay_rate_l751_75103


namespace value_of_x_plus_y_squared_l751_75107

theorem value_of_x_plus_y_squared (x y : ℝ) 
  (h₁ : x^2 + y^2 = 20) 
  (h₂ : x * y = 6) : 
  (x + y)^2 = 32 :=
by
  sorry

end value_of_x_plus_y_squared_l751_75107


namespace power_comparison_l751_75174

theorem power_comparison : (9^20 : ℝ) < (9999^10 : ℝ) :=
sorry

end power_comparison_l751_75174


namespace train_journey_time_l751_75145

theorem train_journey_time {X : ℝ} (h1 : 0 < X) (h2 : X < 60) (h3 : ∀ T_A M_A T_B M_B : ℝ, M_A - T_A = X ∧ M_B - T_B = X) :
    X = 360 / 7 :=
by
  sorry

end train_journey_time_l751_75145


namespace house_number_count_l751_75120

noncomputable def count_valid_house_numbers : Nat :=
  let two_digit_primes := [11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
  let valid_combinations := two_digit_primes.product two_digit_primes |>.filter (λ (WX, YZ) => WX ≠ YZ)
  valid_combinations.length

theorem house_number_count : count_valid_house_numbers = 110 :=
  by
    sorry

end house_number_count_l751_75120


namespace appropriate_weight_design_l751_75134

def weight_design (w_l w_s w_r w_w : ℕ) : Prop :=
  w_l > w_s ∧ w_l > w_w ∧ w_w > w_r ∧ w_s = w_w

theorem appropriate_weight_design :
  weight_design 5 2 1 2 :=
by {
  sorry -- skipped proof
}

end appropriate_weight_design_l751_75134


namespace symmetric_point_yOz_l751_75168

-- Given point A in 3D Cartesian system
def A : ℝ × ℝ × ℝ := (1, -3, 5)

-- Plane yOz where x = 0
def symmetric_yOz (point : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := point
  (-x, y, z)

-- Proof statement (without the actual proof)
theorem symmetric_point_yOz : symmetric_yOz A = (-1, -3, 5) :=
by sorry

end symmetric_point_yOz_l751_75168


namespace owen_profit_l751_75111

/-- Given the initial purchases and sales, calculate Owen's overall profit. -/
theorem owen_profit :
  let boxes_9_dollars := 8
  let boxes_12_dollars := 4
  let cost_9_dollars := 9
  let cost_12_dollars := 12
  let masks_per_box := 50
  let packets_25_pieces := 100
  let price_25_pieces := 5
  let packets_100_pieces := 28
  let price_100_pieces := 12
  let remaining_masks1 := 150
  let price_remaining1 := 3
  let remaining_masks2 := 150
  let price_remaining2 := 4
  let total_cost := (boxes_9_dollars * cost_9_dollars) + (boxes_12_dollars * cost_12_dollars)
  let total_repacked_masks := (packets_25_pieces * price_25_pieces) + (packets_100_pieces * price_100_pieces)
  let total_remaining_masks := (remaining_masks1 * price_remaining1) + (remaining_masks2 * price_remaining2)
  let total_revenue := total_repacked_masks + total_remaining_masks
  let overall_profit := total_revenue - total_cost
  overall_profit = 1766 := by
  sorry

end owen_profit_l751_75111


namespace factorial_product_square_root_square_l751_75102

theorem factorial_product_square_root_square :
  (Real.sqrt (Nat.factorial 5 * Nat.factorial 4 * Nat.factorial 3))^2 = 17280 := 
by
  sorry

end factorial_product_square_root_square_l751_75102


namespace simplify_and_evaluate_expression_l751_75128

theorem simplify_and_evaluate_expression 
  (a b : ℚ) 
  (ha : a = 2) 
  (hb : b = 1 / 3) : 
  (a / (a - b)) * ((1 / b) - (1 / a)) + ((a - 1) / b) = 6 := 
by
  -- Place the steps verifying this here. For now:
  sorry

end simplify_and_evaluate_expression_l751_75128


namespace product_fraction_l751_75198

theorem product_fraction :
  (1 + 1/2) * (1 + 1/4) * (1 + 1/6) * (1 + 1/8) * (1 + 1/10) = 693 / 256 := by
  sorry

end product_fraction_l751_75198


namespace incorrect_mark_l751_75165

theorem incorrect_mark (n : ℕ) (correct_mark incorrect_entry : ℕ) (average_increase : ℕ) :
  n = 40 → correct_mark = 63 → average_increase = 1/2 →
  incorrect_entry - correct_mark = average_increase * n →
  incorrect_entry = 83 :=
by
  intro h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  sorry

end incorrect_mark_l751_75165


namespace stratified_sampling_third_year_l751_75179

theorem stratified_sampling_third_year :
  ∀ (total students_first_year students_second_year sample_size students_third_year sampled_students : ℕ),
  (total = 900) →
  (students_first_year = 240) →
  (students_second_year = 260) →
  (sample_size = 45) →
  (students_third_year = total - students_first_year - students_second_year) →
  (sampled_students = sample_size * students_third_year / total) →
  sampled_students = 20 :=
by
  intros
  sorry

end stratified_sampling_third_year_l751_75179


namespace find_f_4_l751_75180

noncomputable def f : ℝ → ℝ := sorry

axiom functional_equation : ∀ x : ℝ, f x + 2 * f (1 - x) = 3 * x^2

theorem find_f_4 : f 4 = 2 := 
by {
    -- The proof is omitted as per the task.
    sorry
}

end find_f_4_l751_75180


namespace new_quadratic_eq_l751_75184

def quadratic_roots_eq (a b c : ℝ) (x1 x2 : ℝ) : Prop :=
  a * x1^2 + b * x1 + c = 0 ∧ a * x2^2 + b * x2 + c = 0

theorem new_quadratic_eq
  (a b c : ℝ) (x1 x2 : ℝ)
  (h1 : quadratic_roots_eq a b c x1 x2)
  (h_sum : x1 + x2 = -b / a)
  (h_prod : x1 * x2 = c / a) :
  a^3 * x^2 - a * b^2 * x + 2 * c * (b^2 - 2 * a * c) = 0 :=
sorry

end new_quadratic_eq_l751_75184


namespace circumscribed_circle_diameter_l751_75114

theorem circumscribed_circle_diameter (a : ℝ) (A : ℝ) (h_a : a = 16) (h_A : A = 30) :
    let D := a / Real.sin (A * Real.pi / 180)
    D = 32 := by
  sorry

end circumscribed_circle_diameter_l751_75114


namespace cistern_length_l751_75129

-- Definitions of the given conditions
def width : ℝ := 4
def depth : ℝ := 1.25
def total_wet_surface_area : ℝ := 49

-- Mathematical problem: prove the length of the cistern
theorem cistern_length : ∃ (L : ℝ), (L * width + 2 * L * depth + 2 * width * depth = total_wet_surface_area) ∧ L = 6 :=
by
sorry

end cistern_length_l751_75129


namespace smallest_integer_M_exists_l751_75125

theorem smallest_integer_M_exists :
  ∃ (M : ℕ), 
    (M > 0) ∧ 
    (∃ (x y z : ℕ), 
      (x = M ∨ x = M + 1 ∨ x = M + 2) ∧ 
      (y = M ∨ y = M + 1 ∨ y = M + 2) ∧ 
      (z = M ∨ z = M + 1 ∨ z = M + 2) ∧ 
      ((x = M ∨ x = M + 1 ∨ x = M + 2) ∧ x % 8 = 0) ∧ 
      ((y = M ∨ y = M + 1 ∨ y = M + 2) ∧ y % 9 = 0) ∧ 
      ((z = M ∨ z = M + 1 ∨ z = M + 2) ∧ z % 25 = 0) ) ∧ 
    M = 200 := 
by
  sorry

end smallest_integer_M_exists_l751_75125


namespace series_sum_equals_1_over_400_l751_75178

noncomputable def series_term (n : ℕ) : ℝ :=
  (4 * n + 3) / ((4 * n + 1)^2 * (4 * n + 5)^2)

theorem series_sum_equals_1_over_400 :
  ∑' n, series_term (n + 1) = 1 / 400 := by
  sorry

end series_sum_equals_1_over_400_l751_75178


namespace number_of_divisors_of_720_l751_75139

theorem number_of_divisors_of_720 : 
  let n := 720
  let prime_factorization := [(2, 4), (3, 2), (5, 1)] 
  let num_divisors := (4 + 1) * (2 + 1) * (1 + 1)
  n = 2^4 * 3^2 * 5^1 →
  num_divisors = 30 := 
by
  -- Placeholder for the proof
  sorry

end number_of_divisors_of_720_l751_75139


namespace cost_per_chair_l751_75127

theorem cost_per_chair (total_spent : ℕ) (chairs_bought : ℕ) (cost : ℕ) 
  (h1 : total_spent = 180) 
  (h2 : chairs_bought = 12) 
  (h3 : cost = total_spent / chairs_bought) : 
  cost = 15 :=
by
  -- Proof steps go here (skipped with sorry)
  sorry

end cost_per_chair_l751_75127


namespace total_marks_l751_75141

theorem total_marks (k l d : ℝ) (hk : k = 3.5) (hl : l = 3.2 * k) (hd : d = l + 5.7) : k + l + d = 31.6 :=
by
  rw [hk] at hl
  rw [hl] at hd
  rw [hk, hl, hd]
  sorry

end total_marks_l751_75141


namespace oak_trees_remaining_is_7_l751_75182

-- Define the number of oak trees initially in the park
def initial_oak_trees : ℕ := 9

-- Define the number of oak trees cut down by workers
def oak_trees_cut_down : ℕ := 2

-- Define the remaining oak trees calculation
def remaining_oak_trees : ℕ := initial_oak_trees - oak_trees_cut_down

-- Prove that the remaining oak trees is equal to 7
theorem oak_trees_remaining_is_7 : remaining_oak_trees = 7 := by
  sorry

end oak_trees_remaining_is_7_l751_75182


namespace eighth_box_contains_65_books_l751_75167

theorem eighth_box_contains_65_books (total_books boxes first_seven_books per_box eighth_box : ℕ) :
  total_books = 800 →
  boxes = 8 →
  first_seven_books = 7 →
  per_box = 105 →
  eighth_box = total_books - (first_seven_books * per_box) →
  eighth_box = 65 := by
  sorry

end eighth_box_contains_65_books_l751_75167


namespace gcd_8p_18q_l751_75146

theorem gcd_8p_18q (p q : ℕ) (hp : p > 0) (hq : q > 0) (hg : Nat.gcd p q = 9) : Nat.gcd (8 * p) (18 * q) = 18 := 
sorry

end gcd_8p_18q_l751_75146


namespace anya_hairs_wanted_more_l751_75189

def anya_initial_number_of_hairs : ℕ := 0 -- for simplicity, assume she starts with 0 hairs
def hairs_lost_washing : ℕ := 32
def hairs_lost_brushing : ℕ := hairs_lost_washing / 2
def total_hairs_lost : ℕ := hairs_lost_washing + hairs_lost_brushing
def hairs_to_grow_back : ℕ := 49

theorem anya_hairs_wanted_more : total_hairs_lost + hairs_to_grow_back = 97 :=
by
  sorry

end anya_hairs_wanted_more_l751_75189


namespace tan_2x_eq_sin_x_has_three_solutions_l751_75197

theorem tan_2x_eq_sin_x_has_three_solutions :
  ∃ (S : Finset ℝ), (∀ x ∈ S, 0 ≤ x ∧ x ≤ 2 * Real.pi ∧ Real.tan (2 * x) = Real.sin x) ∧ S.card = 3 :=
by
  sorry

end tan_2x_eq_sin_x_has_three_solutions_l751_75197


namespace greatest_number_of_unit_segments_l751_75160

-- Define the conditions
def is_equilateral (n : ℕ) : Prop := n > 0

-- Define the theorem
theorem greatest_number_of_unit_segments (n : ℕ) (h : is_equilateral n) : 
  -- Prove the greatest number of unit segments such that no three of them form a single triangle
  ∃(m : ℕ), m = n * (n + 1) := 
sorry

end greatest_number_of_unit_segments_l751_75160


namespace spilled_bag_candies_l751_75171

theorem spilled_bag_candies (c1 c2 c3 c4 c5 c6 c7 : ℕ) (avg_candies_per_bag : ℕ) (x : ℕ) 
  (h_counts : c1 = 12 ∧ c2 = 14 ∧ c3 = 18 ∧ c4 = 22 ∧ c5 = 24 ∧ c6 = 26 ∧ c7 = 29)
  (h_avg : avg_candies_per_bag = 22)
  (h_total : c1 + c2 + c3 + c4 + c5 + c6 + c7 + x = 8 * avg_candies_per_bag) : x = 31 := 
by
  sorry

end spilled_bag_candies_l751_75171


namespace number_of_positive_solutions_l751_75166

theorem number_of_positive_solutions (x y z : ℕ) (h_cond : x > 0 ∧ y > 0 ∧ z > 0 ∧ x + y + z = 12) :
    ∃ (n : ℕ), n = 55 :=
by 
  sorry

end number_of_positive_solutions_l751_75166


namespace find_quadruples_l751_75191

def valid_quadruple (x1 x2 x3 x4 : ℝ) : Prop :=
  x1 + x2 * x3 * x4 = 2 ∧ 
  x2 + x3 * x4 * x1 = 2 ∧ 
  x3 + x4 * x1 * x2 = 2 ∧ 
  x4 + x1 * x2 * x3 = 2

theorem find_quadruples (x1 x2 x3 x4 : ℝ) :
  valid_quadruple x1 x2 x3 x4 ↔ (x1, x2, x3, x4) = (1, 1, 1, 1) ∨ 
                                   (x1, x2, x3, x4) = (3, -1, -1, -1) ∨ 
                                   (x1, x2, x3, x4) = (-1, 3, -1, -1) ∨ 
                                   (x1, x2, x3, x4) = (-1, -1, 3, -1) ∨ 
                                   (x1, x2, x3, x4) = (-1, -1, -1, 3) := by
  sorry

end find_quadruples_l751_75191


namespace bake_sale_money_raised_correct_l751_75133

def bake_sale_money_raised : Prop :=
  let chocolate_chip_cookies_baked := 4 * 12
  let oatmeal_raisin_cookies_baked := 6 * 12
  let regular_brownies_baked := 2 * 12
  let sugar_cookies_baked := 6 * 12
  let blondies_baked := 3 * 12
  let cream_cheese_swirled_brownies_baked := 5 * 12
  let chocolate_chip_cookies_price := 1.50
  let oatmeal_raisin_cookies_price := 1.00
  let regular_brownies_price := 2.50
  let sugar_cookies_price := 1.25
  let blondies_price := 2.75
  let cream_cheese_swirled_brownies_price := 3.00
  let chocolate_chip_cookies_sold := 0.75 * chocolate_chip_cookies_baked
  let oatmeal_raisin_cookies_sold := 0.85 * oatmeal_raisin_cookies_baked
  let regular_brownies_sold := 0.60 * regular_brownies_baked
  let sugar_cookies_sold := 0.90 * sugar_cookies_baked
  let blondies_sold := 0.80 * blondies_baked
  let cream_cheese_swirled_brownies_sold := 0.50 * cream_cheese_swirled_brownies_baked
  let total_money_raised := 
    chocolate_chip_cookies_sold * chocolate_chip_cookies_price + 
    oatmeal_raisin_cookies_sold * oatmeal_raisin_cookies_price + 
    regular_brownies_sold * regular_brownies_price + 
    sugar_cookies_sold * sugar_cookies_price + 
    blondies_sold * blondies_price + 
    cream_cheese_swirled_brownies_sold * cream_cheese_swirled_brownies_price
  total_money_raised = 397.00

theorem bake_sale_money_raised_correct : bake_sale_money_raised := by
  sorry

end bake_sale_money_raised_correct_l751_75133


namespace RupertCandles_l751_75172

-- Definitions corresponding to the conditions
def PeterAge : ℕ := 10
def RupertRelativeAge : ℝ := 3.5

-- Define Rupert's age based on Peter's age and the given relative age factor
def RupertAge : ℝ := RupertRelativeAge * PeterAge

-- Statement of the theorem
theorem RupertCandles : RupertAge = 35 := by
  -- Proof is omitted
  sorry

end RupertCandles_l751_75172


namespace values_of_fractions_l751_75163

theorem values_of_fractions (A B : ℝ) :
  (∀ x : ℝ, 3 * x ^ 2 + 2 * x - 8 ≠ 0) →
  (∀ x : ℝ, (6 * x - 7) / (3 * x ^ 2 + 2 * x - 8) = A / (x - 2) + B / (3 * x + 4)) →
  A = 1 / 2 ∧ B = 4.5 :=
by
  intros h1 h2
  sorry

end values_of_fractions_l751_75163


namespace smallest_a_exists_l751_75137

theorem smallest_a_exists : ∃ a b c : ℤ, a > 0 ∧ b^2 > 4*a*c ∧ 
  (∀ x : ℝ, x > 0 ∧ x < 1 → (a * x^2 - b * x + c) = 0 → false) 
  ∧ a = 5 :=
by sorry

end smallest_a_exists_l751_75137


namespace solve_diophantine_equations_l751_75142

theorem solve_diophantine_equations :
  { (a, b, c, d) : ℤ × ℤ × ℤ × ℤ |
    a * b - 2 * c * d = 3 ∧
    a * c + b * d = 1 } =
  { (1, 3, 1, 0), (-1, -3, -1, 0), (3, 1, 0, 1), (-3, -1, 0, -1) } :=
by
  sorry

end solve_diophantine_equations_l751_75142


namespace distance_between_lines_l751_75147

/-- Define the lines by their equations -/
def line1 (x y : ℝ) : Prop := 3 * x + 4 * y - 12 = 0
def line2 (x y : ℝ) : Prop := 6 * x + 8 * y + 6 = 0

/-- Define the simplified form of the second line -/
def simplified_line2 (x y : ℝ) : Prop := 3 * x + 4 * y + 3 = 0

/-- Prove the distance between the two lines is 3 -/
theorem distance_between_lines : 
  let A : ℝ := 3
  let B : ℝ := 4
  let C1 : ℝ := -12
  let C2 : ℝ := 3
  (|C2 - C1| / Real.sqrt (A^2 + B^2) = 3) :=
by
  sorry

end distance_between_lines_l751_75147


namespace m_range_l751_75117

noncomputable def f (x : ℝ) : ℝ :=
  Real.sqrt 3 * Real.sin (2018 * Real.pi - x) * Real.sin (3 * Real.pi / 2 + x) 
  - Real.cos x ^ 2 + 1

def valid_m (m : ℝ) : Prop := 
  ∀ x ∈ Set.Icc (-Real.pi / 12) (Real.pi / 2), abs (f x - m) ≤ 1

theorem m_range : 
  ∀ m : ℝ, valid_m m ↔ (m ∈ Set.Icc (1 / 2) ((3 - Real.sqrt 3) / 2)) :=
by sorry

end m_range_l751_75117


namespace rectangular_coordinates_of_polar_2_pi_over_3_l751_75136

noncomputable def polar_to_rectangular (ρ θ : ℝ) : ℝ × ℝ :=
  (ρ * Real.cos θ, ρ * Real.sin θ)

theorem rectangular_coordinates_of_polar_2_pi_over_3 :
  polar_to_rectangular 2 (Real.pi / 3) = (1, Real.sqrt 3) :=
by
  sorry

end rectangular_coordinates_of_polar_2_pi_over_3_l751_75136


namespace find_x_l751_75105

theorem find_x (x : ℝ) (h : 3 * x = 36 - x + 16) : x = 13 :=
by
  sorry

end find_x_l751_75105


namespace jack_black_balloons_l751_75193

def nancy_balloons := 7
def mary_balloons := 4 * nancy_balloons
def total_mary_nancy_balloons := nancy_balloons + mary_balloons
def jack_balloons := total_mary_nancy_balloons + 3

theorem jack_black_balloons : jack_balloons = 38 := by
  -- proof goes here
  sorry

end jack_black_balloons_l751_75193


namespace lauri_ate_days_l751_75164

theorem lauri_ate_days
    (simone_rate : ℚ)
    (simone_days : ℕ)
    (lauri_rate : ℚ)
    (total_apples : ℚ)
    (simone_apples : ℚ)
    (lauri_apples : ℚ)
    (lauri_days : ℚ) :
  simone_rate = 1/2 → 
  simone_days = 16 →
  lauri_rate = 1/3 →
  total_apples = 13 →
  simone_apples = simone_rate * simone_days →
  lauri_apples = total_apples - simone_apples →
  lauri_days = lauri_apples / lauri_rate →
  lauri_days = 15 :=
by
  intros
  sorry

end lauri_ate_days_l751_75164


namespace liu_xiang_hurdles_l751_75158

theorem liu_xiang_hurdles :
  let total_distance := 110
  let first_hurdle_distance := 13.72
  let last_hurdle_distance := 14.02
  let best_time_first_segment := 2.5
  let best_time_last_segment := 1.4
  let hurdle_cycle_time := 0.96
  let num_hurdles := 10
  (total_distance - first_hurdle_distance - last_hurdle_distance) / num_hurdles = 8.28 ∧
  best_time_first_segment + num_hurdles * hurdle_cycle_time + best_time_last_segment  = 12.1 :=
by
  sorry

end liu_xiang_hurdles_l751_75158


namespace find_four_digit_number_l751_75181

/-- 
  If there exists a positive integer M and M² both end in the same sequence of 
  five digits abcde in base 10 where a ≠ 0, 
  then the four-digit number abcd derived from M = 96876 is 9687.
-/
theorem find_four_digit_number
  (M : ℕ)
  (h_end_digits : (M % 100000) = (M * M % 100000))
  (h_first_digit_nonzero : 10000 ≤ M % 100000  ∧ M % 100000 < 100000)
  : (M = 96876 → (M / 10 % 10000 = 9687)) :=
by { sorry }

end find_four_digit_number_l751_75181


namespace total_expenditure_is_3500_l751_75196

def expenditure_mon : ℕ := 450
def expenditure_tue : ℕ := 600
def expenditure_wed : ℕ := 400
def expenditure_thurs : ℕ := 500
def expenditure_sat : ℕ := 550
def expenditure_sun : ℕ := 300
def cost_earphone : ℕ := 620
def cost_pen : ℕ := 30
def cost_notebook : ℕ := 50

def expenditure_fri : ℕ := cost_earphone + cost_pen + cost_notebook
def total_expenditure : ℕ := expenditure_mon + expenditure_tue + expenditure_wed + expenditure_thurs + expenditure_fri + expenditure_sat + expenditure_sun

theorem total_expenditure_is_3500 : total_expenditure = 3500 := by
  sorry

end total_expenditure_is_3500_l751_75196


namespace range_of_a_l751_75175

theorem range_of_a (p q : Prop)
  (hp : ∀ a : ℝ, (1 < a ↔ p))
  (hq : ∀ a : ℝ, (2 ≤ a ∨ a ≤ -2 ↔ q))
  (hpq : ∀ a : ℝ, ∀ (p : Prop), ∀ (q : Prop), (p ∧ q) → p ∧ q) :
    ∀ a : ℝ, p ∧ q → 2 ≤ a :=
sorry

end range_of_a_l751_75175


namespace multiple_of_fair_tickets_l751_75169

theorem multiple_of_fair_tickets (fair_tickets_sold : ℕ) (game_tickets_sold : ℕ) (h : fair_tickets_sold = game_tickets_sold * x + 6) :
  25 = 56 * x + 6 → x = 19 / 56 := by
  sorry

end multiple_of_fair_tickets_l751_75169


namespace sufficient_but_not_necessary_condition_for_monotonicity_l751_75110

theorem sufficient_but_not_necessary_condition_for_monotonicity
  (a : ℕ → ℝ)
  (h_recurrence : ∀ n : ℕ, n > 0 → a (n + 1) = a n ^ 2)
  (h_initial : a 1 = 2) :
  (∀ n : ℕ, n > 0 → a n > a 0) :=
by
  sorry

end sufficient_but_not_necessary_condition_for_monotonicity_l751_75110


namespace boat_speed_ratio_l751_75156

variable (B S : ℝ)

theorem boat_speed_ratio (h : 1 / (B - S) = 2 * (1 / (B + S))) : B / S = 3 := 
by
  sorry

end boat_speed_ratio_l751_75156


namespace long_show_episode_duration_is_one_hour_l751_75132

-- Definitions for the given conditions
def total_shows : ℕ := 2
def short_show_length : ℕ := 24
def short_show_episode_duration : ℝ := 0.5
def long_show_episodes : ℕ := 12
def total_viewing_time : ℝ := 24

-- Definition of the length of each episode of the longer show
def long_show_episode_length (L : ℝ) : Prop :=
  (short_show_length * short_show_episode_duration) + (long_show_episodes * L) = total_viewing_time

-- Main statement to prove
theorem long_show_episode_duration_is_one_hour : long_show_episode_length 1 :=
by
  -- Proof placeholder
  sorry

end long_show_episode_duration_is_one_hour_l751_75132


namespace laptop_full_price_l751_75154

theorem laptop_full_price (p : ℝ) (deposit : ℝ) (h1 : deposit = 0.25 * p) (h2 : deposit = 400) : p = 1600 :=
by
  sorry

end laptop_full_price_l751_75154


namespace sum_of_a3_a4_a5_l751_75149

def geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a n = 3 * q ^ n

theorem sum_of_a3_a4_a5 
  (a : ℕ → ℝ) (q : ℝ) 
  (h_geometric : geometric_sequence_sum a q)
  (h_pos : ∀ n, a n > 0)
  (h_first_term : a 0 = 3)
  (h_sum_first_three : a 0 + a 1 + a 2 = 21) :
  a 2 + a 3 + a 4 = 84 :=
sorry

end sum_of_a3_a4_a5_l751_75149


namespace trains_meet_at_9am_l751_75159

-- Definitions of conditions
def distance_AB : ℝ := 65
def speed_train_A : ℝ := 20
def speed_train_B : ℝ := 25
def start_time_train_A : ℝ := 7
def start_time_train_B : ℝ := 8

-- This function calculates the meeting time of the two trains
noncomputable def meeting_time (distance_AB : ℝ) (speed_train_A : ℝ) (speed_train_B : ℝ) 
    (start_time_train_A : ℝ) (start_time_train_B : ℝ) : ℝ :=
  let distance_train_A := speed_train_A * (start_time_train_B - start_time_train_A)
  let remaining_distance := distance_AB - distance_train_A
  let relative_speed := speed_train_A + speed_train_B
  start_time_train_B + remaining_distance / relative_speed

-- Theorem stating the time when the two trains meet
theorem trains_meet_at_9am :
    meeting_time distance_AB speed_train_A speed_train_B start_time_train_A start_time_train_B = 9 := sorry

end trains_meet_at_9am_l751_75159


namespace find_C_l751_75108

theorem find_C (A B C : ℕ) 
  (h1 : A + B + C = 500) 
  (h2 : A + C = 200) 
  (h3 : B + C = 320) : 
  C = 20 := 
by 
  sorry

end find_C_l751_75108


namespace kiril_konstantinovich_age_is_full_years_l751_75130

theorem kiril_konstantinovich_age_is_full_years
  (years : ℕ)
  (months : ℕ)
  (weeks : ℕ)
  (days : ℕ)
  (hours : ℕ) :
  (years = 48) →
  (months = 48) →
  (weeks = 48) →
  (days = 48) →
  (hours = 48) →
  Int.floor (
    years + 
    (months / 12 : ℝ) + 
    (weeks * 7 / 365 : ℝ) + 
    (days / 365 : ℝ) + 
    (hours / (24 * 365) : ℝ)
  ) = 53 :=
by
  intro hyears hmonths hweeks hdays hhours
  rw [hyears, hmonths, hweeks, hdays, hhours]
  sorry

end kiril_konstantinovich_age_is_full_years_l751_75130


namespace exam_max_marks_l751_75126

theorem exam_max_marks (M : ℝ) (h1: 0.30 * M = 66) : M = 220 :=
by
  sorry

end exam_max_marks_l751_75126


namespace geometric_sequence_sum_is_five_eighths_l751_75150

noncomputable def geometric_sequence_sum (a₁ : ℝ) (q : ℝ) : ℝ :=
  if q = 1 then 4 * a₁ else a₁ * (1 - q^4) / (1 - q)

theorem geometric_sequence_sum_is_five_eighths
  (a₁ q : ℝ)
  (h₀ : q ≠ 1)
  (h₁ : a₁ * (a₁ * q) * (a₁ * q^2) = -1 / 8)
  (h₂ : 2 * (a₁ * q^2) = a₁ * q + a₁ * q^2) :
  geometric_sequence_sum a₁ q = 5 / 8 := by
sorry

end geometric_sequence_sum_is_five_eighths_l751_75150


namespace player_1_winning_strategy_l751_75119

-- Define the properties and rules of the game
def valid_pair (a b : ℕ) : Prop := a > 0 ∧ b > 0 ∧ a + b = 2005

def move (current t a b : ℕ) : Prop := 
  current = t - a ∨ current = t - b

def first_player_wins (t a b : ℕ) : Prop :=
  ∀ k : ℕ, t > k * 2005 → ∃ m : ℕ, move (t - m) t a b

-- Main theorem statement
theorem player_1_winning_strategy : ∃ (t : ℕ) (a b : ℕ), valid_pair a b ∧ first_player_wins t a b :=
sorry

end player_1_winning_strategy_l751_75119


namespace translation_correct_l751_75183

def vector_a : ℝ × ℝ := (1, 1)

def translate_right (v : ℝ × ℝ) (d : ℝ) : ℝ × ℝ := (v.1 + d, v.2)
def translate_down (v : ℝ × ℝ) (d : ℝ) : ℝ × ℝ := (v.1, v.2 - d)

def vector_b := translate_down (translate_right vector_a 2) 1

theorem translation_correct :
  vector_b = (3, 0) :=
by
  -- proof steps will go here
  sorry

end translation_correct_l751_75183


namespace find_x_l751_75101

theorem find_x (x y : ℝ) (h1 : 0.65 * x = 0.20 * y)
  (h2 : y = 617.5 ^ 2 - 42) : 
  x = 117374.3846153846 :=
by
  sorry

end find_x_l751_75101


namespace area_difference_is_correct_l751_75188

noncomputable def area_rectangle (length width : ℝ) : ℝ := length * width

noncomputable def area_equilateral_triangle (side : ℝ) : ℝ := (Real.sqrt 3 / 4) * side ^ 2

noncomputable def area_circle (diameter : ℝ) : ℝ := (Real.pi * (diameter / 2) ^ 2)

noncomputable def combined_area_difference : ℝ :=
  (area_rectangle 11 11 + area_rectangle 5.5 11) - 
  (area_equilateral_triangle 6 + area_circle 4)
 
theorem area_difference_is_correct :
  |combined_area_difference - 153.35| < 0.001 :=
by
  sorry

end area_difference_is_correct_l751_75188


namespace derivative_at_one_l751_75192

open Real

noncomputable def f (x : ℝ) : ℝ := exp x / x

theorem derivative_at_one : deriv f 1 = 0 :=
by
  sorry

end derivative_at_one_l751_75192
