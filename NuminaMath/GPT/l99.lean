import Mathlib

namespace smallest_n_terminating_decimal_l99_99169

theorem smallest_n_terminating_decimal :
  ∃ n : ℕ, (∀ m : ℕ, (∃ k : ℕ, m + 150 = 2^k * (5^0))
  → (m ≥ n)) ∧ (n > 0) ∧ (∃ k : ℕ, n + 150 = 2^k * (5^0)) := 
sorry

end smallest_n_terminating_decimal_l99_99169


namespace marble_problem_l99_99346

theorem marble_problem
  (x : ℕ) (h1 : 144 / x = 144 / (x + 2) + 1) :
  x = 16 :=
sorry

end marble_problem_l99_99346


namespace sum_of_coefficients_l99_99385

-- Define the polynomial P(x)
def P (x : ℝ) : ℝ := 5 * (2 * x^8 - 3 * x^5 + 9 * x^3 - 6) + 4 * (7 * x^6 - 2 * x^3 + 8)

-- Statement to prove that the sum of the coefficients of P(x) is 62
theorem sum_of_coefficients : P 1 = 62 := sorry

end sum_of_coefficients_l99_99385


namespace multiply_101_self_l99_99861

theorem multiply_101_self : 101 * 101 = 10201 := 
by
  -- Proof omitted
  sorry

end multiply_101_self_l99_99861


namespace hexagon_perimeter_arithmetic_sequence_l99_99317

theorem hexagon_perimeter_arithmetic_sequence :
  let a₁ := 10
  let a₂ := 12
  let a₃ := 14
  let a₄ := 16
  let a₅ := 18
  let a₆ := 20
  let lengths := [a₁, a₂, a₃, a₄, a₅, a₆]
  let perimeter := lengths.sum
  perimeter = 90 :=
by
  sorry

end hexagon_perimeter_arithmetic_sequence_l99_99317


namespace children_got_on_bus_l99_99511

-- Definitions based on conditions
def initial_children : ℕ := 22
def children_got_off : ℕ := 60
def children_after_stop : ℕ := 2

-- Define the problem
theorem children_got_on_bus : ∃ x : ℕ, initial_children - children_got_off + x = children_after_stop ∧ x = 40 :=
by
  sorry

end children_got_on_bus_l99_99511


namespace peanut_price_is_correct_l99_99656

noncomputable def price_per_pound_of_peanuts : ℝ := 
  let total_weight := 100
  let mixed_price_per_pound := 2.5
  let cashew_weight := 60
  let cashew_price_per_pound := 4
  let peanut_weight := total_weight - cashew_weight
  let total_revenue := total_weight * mixed_price_per_pound
  let cashew_cost := cashew_weight * cashew_price_per_pound
  let peanut_cost := total_revenue - cashew_cost
  peanut_cost / peanut_weight

theorem peanut_price_is_correct :
  price_per_pound_of_peanuts = 0.25 := 
by sorry

end peanut_price_is_correct_l99_99656


namespace problem_I_problem_II_l99_99408

-- Problem (I)
theorem problem_I (x : ℝ) (f : ℝ → ℝ) (hf : ∀ x, f x = |x + 1|) : 
  (f (x + 8) ≥ 10 - f x) ↔ (x ≤ -10 ∨ x ≥ 0) :=
sorry

-- Problem (II)
theorem problem_II (x y : ℝ) (f : ℝ → ℝ) (hf : ∀ x, f x = |x + 1|) 
(h_abs_x : |x| > 1) (h_abs_y : |y| < 1) :
  f y < |x| * f (y / x^2) :=
sorry

end problem_I_problem_II_l99_99408


namespace balance_difference_l99_99275

def compounded_balance (P : ℝ) (r : ℝ) (n : ℕ) : ℝ := P * (1 + r)^n

def simple_interest_balance (P : ℝ) (r : ℝ) (t : ℕ) : ℝ := P * (1 + r * t)

/-- Cedric deposits $15,000 into an account that pays 6% interest compounded annually,
    Daniel deposits $15,000 into an account that pays 8% simple annual interest.
    After 10 years, the positive difference between their balances is $137. -/
theorem balance_difference :
  let P : ℝ := 15000
  let r_cedric : ℝ := 0.06
  let r_daniel : ℝ := 0.08
  let t : ℕ := 10
  compounded_balance P r_cedric t - simple_interest_balance P r_daniel t = 137 := 
sorry

end balance_difference_l99_99275


namespace cake_pieces_per_sister_l99_99550

theorem cake_pieces_per_sister (total_pieces : ℕ) (percentage_eaten : ℕ) (sisters : ℕ)
  (h1 : total_pieces = 240) (h2 : percentage_eaten = 60) (h3 : sisters = 3) :
  (total_pieces * (1 - percentage_eaten / 100)) / sisters = 32 :=
by
  sorry

end cake_pieces_per_sister_l99_99550


namespace eval_frac_l99_99054

theorem eval_frac 
  (a b : ℚ)
  (h₀ : a = 7) 
  (h₁ : b = 2) :
  3 / (a + b) = 1 / 3 :=
by
  sorry

end eval_frac_l99_99054


namespace employee_percentage_six_years_or_more_l99_99976

theorem employee_percentage_six_years_or_more
  (x : ℕ)
  (total_employees : ℕ := 36 * x)
  (employees_6_or_more : ℕ := 8 * x) :
  (employees_6_or_more : ℚ) / (total_employees : ℚ) * 100 = 22.22 := 
sorry

end employee_percentage_six_years_or_more_l99_99976


namespace must_be_nonzero_l99_99471

noncomputable def Q (a b c d : ℝ) : ℝ → ℝ :=
  λ x => x^5 + a * x^4 + b * x^3 + c * x^2 + d * x

theorem must_be_nonzero (a b c d : ℝ)
  (h_roots : ∃ p q r s : ℝ, (∀ y : ℝ, Q a b c d y = 0 → y = 0 ∨ y = -1 ∨ y = p ∨ y = q ∨ y = r ∨ y = s) ∧ p ≠ 0 ∧ p ≠ -1 ∧ q ≠ 0 ∧ q ≠ -1 ∧ r ≠ 0 ∧ r ≠ -1 ∧ s ≠ 0 ∧ s ≠ -1)
  (h_distinct : (∀ x₁ x₂ : ℝ, Q a b c d x₁ = 0 ∧ Q a b c d x₂ = 0 → x₁ ≠ x₂ ∨ x₁ = x₂) → False)
  (h_f_zero : Q a b c d 0 = 0) :
  d ≠ 0 := by
  sorry

end must_be_nonzero_l99_99471


namespace product_of_solutions_l99_99396

theorem product_of_solutions :
  (∀ y : ℝ, (|y| = 2 * (|y| - 1)) → y = 2 ∨ y = -2) →
  (∀ y1 y2 : ℝ, (y1 = 2 ∧ y2 = -2) → y1 * y2 = -4) :=
by
  intro h
  have h1 := h 2
  have h2 := h (-2)
  sorry

end product_of_solutions_l99_99396


namespace lines_intersect_not_perpendicular_l99_99874

noncomputable def slopes_are_roots (m k1 k2 : ℝ) : Prop :=
  k1^2 + m*k1 - 2 = 0 ∧ k2^2 + m*k2 - 2 = 0

theorem lines_intersect_not_perpendicular (m k1 k2 : ℝ) (h : slopes_are_roots m k1 k2) : (k1 * k2 = -2 ∧ k1 ≠ k2) → ∃ l1 l2 : ℝ, l1 ≠ l2 ∧ l1 = k1 ∧ l2 = k2 :=
by
  sorry

end lines_intersect_not_perpendicular_l99_99874


namespace min_value_ratio_l99_99871

variable {α : Type*} [LinearOrderedField α]

theorem min_value_ratio (a : ℕ → α) (h1 : a 7 = a 6 + 2 * a 5) (h2 : ∃ m n : ℕ, a m * a n = 8 * a 1^2) :
  ∃ m n : ℕ, (1 / m + 4 / n = 11 / 6) :=
by
  sorry

end min_value_ratio_l99_99871


namespace tripling_base_exponent_l99_99911

variables (a b x : ℝ)

theorem tripling_base_exponent (b_ne_zero : b ≠ 0) (r_def : (3 * a)^(3 * b) = a^b * x^b) : x = 27 * a^2 :=
by
  -- Proof omitted as requested
  sorry

end tripling_base_exponent_l99_99911


namespace simplify_expression_l99_99464

theorem simplify_expression :
  (0.7264 * 0.4329 * 0.5478) + (0.1235 * 0.3412 * 0.6214) - ((0.1289 * 0.5634 * 0.3921) / (0.3785 * 0.4979 * 0.2884)) - (0.2956 * 0.3412 * 0.6573) = -0.3902 :=
by
  sorry

end simplify_expression_l99_99464


namespace total_profit_l99_99361

variable (InvestmentA InvestmentB InvestmentTimeA InvestmentTimeB ShareA : ℝ)
variable (hA : InvestmentA = 150)
variable (hB : InvestmentB = 200)
variable (hTimeA : InvestmentTimeA = 12)
variable (hTimeB : InvestmentTimeB = 6)
variable (hShareA : ShareA = 60)

theorem total_profit (TotalProfit : ℝ) :
  (ShareA / 3) * 5 = TotalProfit := 
by
  sorry

end total_profit_l99_99361


namespace new_box_volume_eq_5_76_m3_l99_99945

-- Given conditions:
def original_width_cm := 80
def original_length_cm := 75
def original_height_cm := 120
def conversion_factor_cm3_to_m3 := 1000000

-- New dimensions after doubling
def new_width_cm := 2 * original_width_cm
def new_length_cm := 2 * original_length_cm
def new_height_cm := 2 * original_height_cm

-- Statement of the problem
theorem new_box_volume_eq_5_76_m3 :
  (new_width_cm * new_length_cm * new_height_cm : ℝ) / conversion_factor_cm3_to_m3 = 5.76 := 
  sorry

end new_box_volume_eq_5_76_m3_l99_99945


namespace total_floor_area_covered_l99_99946

theorem total_floor_area_covered (A B C : ℝ) 
  (h1 : A + B + C = 200) 
  (h2 : B = 24) 
  (h3 : C = 19) : 
  A - (B - C) - 2 * C = 138 := 
by sorry

end total_floor_area_covered_l99_99946


namespace contrapositive_proposition_l99_99117

theorem contrapositive_proposition (x a b : ℝ) : (x < 2 * a * b) → (x < a^2 + b^2) :=
sorry

end contrapositive_proposition_l99_99117


namespace florist_first_picking_l99_99978

theorem florist_first_picking (x : ℝ) (h1 : 37.0 + x + 19.0 = 72.0) : x = 16.0 :=
by
  sorry

end florist_first_picking_l99_99978


namespace probability_same_plane_l99_99722

-- Define the number of vertices in a cube
def num_vertices : ℕ := 8

-- Define the number of vertices to be selected
def selection : ℕ := 4

-- Define the total number of ways to select 4 vertices out of 8
def total_ways : ℕ := Nat.choose num_vertices selection

-- Define the number of favorable ways to have 4 vertices lie in the same plane
def favorable_ways : ℕ := 12

-- Define the probability that the 4 selected vertices lie in the same plane
def probability : ℚ := favorable_ways / total_ways

-- The statement we need to prove
theorem probability_same_plane : probability = 6 / 35 := by
  sorry

end probability_same_plane_l99_99722


namespace circle_radius_l99_99865

theorem circle_radius (x y : ℝ) : x^2 - 8 * x + y^2 - 4 * y + 16 = 0 → sqrt ((4:ℝ)^2) = 2 :=
by
  sorry

end circle_radius_l99_99865


namespace smallest_positive_integer_for_terminating_decimal_l99_99181

theorem smallest_positive_integer_for_terminating_decimal: ∃ n: ℕ, (n > 0) ∧ (∀ p : ℕ, (p ∣ (n + 150)) → (p=1 ∨ p=2 ∨ p=4 ∨ p=5 ∨ p=8 ∨ p=10 ∨ p=16 ∨ p=20 ∨ p=25 ∨ p=32 ∨ p=40 ∨ p=50 ∨ p=64 ∨ p=80 ∨ p=100 ∨ p=125 ∨ p=128 ∨ p=160)) ∧ n = 10 :=
by
  sorry

end smallest_positive_integer_for_terminating_decimal_l99_99181


namespace fraction_covered_by_triangle_l99_99016

structure Point where
  x : ℤ
  y : ℤ

def area_of_triangle (A B C : Point) : ℚ :=
  (1/2 : ℚ) * abs (A.x * (B.y - C.y) + B.x * (C.y - A.y) + C.x * (A.y - B.y))

def area_of_grid (length width : ℤ) : ℚ :=
  (length * width : ℚ)

def fraction_of_grid_covered (A B C : Point) (length width : ℤ) : ℚ :=
  (area_of_triangle A B C) / (area_of_grid length width)

theorem fraction_covered_by_triangle :
  fraction_of_grid_covered ⟨2, 4⟩ ⟨7, 2⟩ ⟨6, 5⟩ 8 6 = 13 / 96 :=
by
  sorry

end fraction_covered_by_triangle_l99_99016


namespace race_time_A_l99_99434

noncomputable def time_for_A_to_cover_distance (distance : ℝ) (time_of_B : ℝ) (remaining_distance_for_B : ℝ) : ℝ :=
  let speed_of_B := distance / time_of_B
  let time_for_B_to_cover_remaining := remaining_distance_for_B / speed_of_B
  time_for_B_to_cover_remaining

theorem race_time_A (distance : ℝ) (time_of_B : ℝ) (remaining_distance_for_B : ℝ) :
  distance = 100 ∧ time_of_B = 25 ∧ remaining_distance_for_B = distance - 20 →
  time_for_A_to_cover_distance distance time_of_B remaining_distance_for_B = 20 :=
by
  intros h
  rcases h with ⟨h_distance, h_time_of_B, h_remaining_distance_for_B⟩
  rw [h_distance, h_time_of_B, h_remaining_distance_for_B]
  sorry

end race_time_A_l99_99434


namespace sum_of_eight_numbers_l99_99745

variable (avg : ℝ) (n : ℕ)

-- Given condition
def average_eq_of_eight_numbers : avg = 5.5 := rfl
def number_of_items_eq_eight : n = 8 := rfl

-- Theorem to prove
theorem sum_of_eight_numbers (h1 : average_eq_of_eight_numbers avg)
                             (h2 : number_of_items_eq_eight n) :
  avg * n = 44 :=
by
  -- Proof will be inserted here
  sorry

end sum_of_eight_numbers_l99_99745


namespace largest_angle_in_consecutive_integer_hexagon_l99_99485

theorem largest_angle_in_consecutive_integer_hexagon : 
  ∀ (x : ℤ), 
  (x - 3) + (x - 2) + (x - 1) + x + (x + 1) + (x + 2) = 720 → 
  (x + 2 = 122) :=
by intros x h
   sorry

end largest_angle_in_consecutive_integer_hexagon_l99_99485


namespace gcd_polynomial_l99_99405

theorem gcd_polynomial (b : ℤ) (hb : ∃ k : ℤ, b = 570 * k) :
  Int.gcd (5 * b^3 + b^2 + 6 * b + 95) b = 95 :=
by
  sorry

end gcd_polynomial_l99_99405


namespace smallest_n_for_terminating_decimal_l99_99147

theorem smallest_n_for_terminating_decimal :
  ∃ n : ℕ, n > 0 ∧ (∀ m : ℕ, m > 0 ∧ (∀ p: ℕ, (p ∣ (m + 150) → (p = 2 ∨ p = 5)) → m ≥ n)) ∧
  (∃ two_five_factors : ℕ, (two_five_factors = 5 ^ 3 * 2 ^ 3) ∧ (n + 150 = two_five_factors)) := 
by 
   exact ⟨50, by sorry⟩

end smallest_n_for_terminating_decimal_l99_99147


namespace radius_of_circumscribed_circle_l99_99368

-- Definitions based on conditions
def sector (radius : ℝ) (central_angle : ℝ) : Prop :=
  central_angle = 120 ∧ radius = 10

-- Statement of the theorem we want to prove
theorem radius_of_circumscribed_circle (r R : ℝ) (h : sector r 120) : R = 20 := 
by
  sorry

end radius_of_circumscribed_circle_l99_99368


namespace projectile_height_time_l99_99469

theorem projectile_height_time :
  ∃ t, t ≥ 0 ∧ -16 * t^2 + 80 * t = 72 ↔ t = 1 := 
by sorry

end projectile_height_time_l99_99469


namespace expression_equivalence_l99_99504

theorem expression_equivalence:
  let a := 10006 - 8008
  let b := 10000 - 8002
  a = b :=
by {
  sorry
}

end expression_equivalence_l99_99504


namespace girls_boys_difference_l99_99255

variables (B G : ℕ) (x : ℕ)

-- Condition that relates boys and girls with a ratio
def ratio_condition : Prop := 3 * x = B ∧ 4 * x = G

-- Condition that the total number of students is 42
def total_students_condition : Prop := B + G = 42

-- We want to prove that the difference between the number of girls and boys is 6
theorem girls_boys_difference (h_ratio : ratio_condition B G x) (h_total : total_students_condition B G) : 
  G - B = 6 :=
sorry

end girls_boys_difference_l99_99255


namespace shopping_money_l99_99352

theorem shopping_money (X : ℝ) (h : 0.70 * X = 840) : X = 1200 :=
sorry

end shopping_money_l99_99352


namespace tangent_line_to_circle_range_mn_l99_99784

theorem tangent_line_to_circle_range_mn (m n : ℝ) 
  (h1 : (m + 1) * (m + 1) + (n + 1) * (n + 1) = 4) :
  (m + n ≤ 2 - 2 * Real.sqrt 2) ∨ (m + n ≥ 2 + 2 * Real.sqrt 2) :=
sorry

end tangent_line_to_circle_range_mn_l99_99784


namespace acute_angle_comparison_l99_99852

noncomputable def f : ℝ → ℝ := sorry

def even_function (f : ℝ → ℝ) :=
  ∀ x, f x = f (-x)

def periodic_function (f : ℝ → ℝ) (p : ℝ) :=
  ∀ x, f (x + p) = f x

def increasing_on_interval (f : ℝ → ℝ) (a b : ℝ) :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

theorem acute_angle_comparison (A B : ℝ)
  (hA : 0 < A ∧ A < π / 2)
  (hB : 0 < B ∧ B < π / 2)
  (f_even : even_function f)
  (f_periodic : ∀ x, f (x + 1) + f x = 0)
  (f_increasing : increasing_on_interval f 3 4) :
  f (Real.sin A) < f (Real.cos B) :=
sorry

end acute_angle_comparison_l99_99852


namespace solution_set_of_inequality_l99_99943

theorem solution_set_of_inequality :
  { x : ℝ | (x - 5) / (x + 1) ≤ 0 } = { x : ℝ | -1 < x ∧ x ≤ 5 } :=
sorry

end solution_set_of_inequality_l99_99943


namespace hyperbola_asymptotes_l99_99502

theorem hyperbola_asymptotes :
  ∀ (x y : ℝ), (x^2 - 4 * y^2 = 1) → (x = 2 * y ∨ x = -2 * y) :=
by
  intros x y h
  sorry

end hyperbola_asymptotes_l99_99502


namespace smallest_angle_between_radii_l99_99818

theorem smallest_angle_between_radii (n : ℕ) (k : ℕ) (angle_step : ℕ) (angle_smallest : ℕ) 
(h_n : n = 40) 
(h_k : k = 23) 
(h_angle_step : angle_step = k) 
(h_angle_smallest : angle_smallest = 23) : 
angle_smallest = 23 :=
sorry

end smallest_angle_between_radii_l99_99818


namespace mod_inverse_5_221_l99_99390

theorem mod_inverse_5_221 : ∃ x : ℤ, 0 ≤ x ∧ x < 221 ∧ (5 * x) % 221 = 1 % 221 :=
by
  use 177
  sorry

end mod_inverse_5_221_l99_99390


namespace vessel_base_length_l99_99197

variables (L : ℝ) (edge : ℝ) (W : ℝ) (h : ℝ)
def volume_cube := edge^3
def volume_rise := L * W * h

theorem vessel_base_length :
  (volume_cube 16 = volume_rise L 15 13.653333333333334) →
  L = 20 :=
by sorry

end vessel_base_length_l99_99197


namespace repeating_decimal_sum_l99_99341

theorem repeating_decimal_sum :
  let x : ℚ := 45 / 99 in
  let simp_fraction := (5, 11) in
  simp_fraction.fst + simp_fraction.snd = 16 :=
by
  let x : ℚ := 45 / 99
  let simp_fraction := (5, 11)
  have h_gcd : Int.gcd 45 99 = 9 := by norm_num
  have h_simplify : x = simp_fraction.fst / simp_fraction.snd := by
    rw [num_denom, h_gcd]
    norm_cast
    simp
  show simp_fraction.fst + simp_fraction.snd = 16 from
    by norm_num
  simp_fraction.rfl

end repeating_decimal_sum_l99_99341


namespace ratio_p_q_l99_99388

-- Definitions of probabilities p and q based on combinatorial choices and probabilities described.
noncomputable def p : ℚ :=
  (Nat.choose 6 1) * (Nat.choose 5 2) * (Nat.choose 24 2) * (Nat.choose 22 4) * (Nat.choose 18 4) * (Nat.choose 14 5) * (Nat.choose 9 5) * (Nat.choose 4 5) / (6 ^ 24)

noncomputable def q : ℚ :=
  (Nat.choose 6 2) * (Nat.choose 24 3) * (Nat.choose 21 3) * (Nat.choose 18 4) * (Nat.choose 14 4) * (Nat.choose 10 4) * (Nat.choose 6 4) / (6 ^ 24)

-- Lean statement to prove p / q = 6
theorem ratio_p_q : p / q = 6 := by
  sorry

end ratio_p_q_l99_99388


namespace andrew_apples_l99_99530

theorem andrew_apples : ∃ (A n : ℕ), (6 * n = A) ∧ (5 * (n + 2) = A) ∧ (A = 60) :=
by 
  sorry

end andrew_apples_l99_99530


namespace segment_length_cd_l99_99793

theorem segment_length_cd
  (AB : ℝ)
  (M : ℝ)
  (N : ℝ)
  (P : ℝ)
  (C : ℝ)
  (D : ℝ)
  (h₁ : AB = 60)
  (h₂ : N = M / 2)
  (h₃ : P = (AB - M) / 2)
  (h₄ : C = N / 2)
  (h₅ : D = P / 2) :
  |C - D| = 15 :=
by
  sorry

end segment_length_cd_l99_99793


namespace loss_percentage_is_30_l99_99118

theorem loss_percentage_is_30
  (cost_price : ℝ)
  (selling_price : ℝ)
  (h1 : cost_price = 1900)
  (h2 : selling_price = 1330) :
  (cost_price - selling_price) / cost_price * 100 = 30 :=
by
  -- This is a placeholder for the actual proof
  sorry

end loss_percentage_is_30_l99_99118


namespace probability_of_different_topics_l99_99519

theorem probability_of_different_topics (n : ℕ) (m : ℕ) (prob : ℚ)
  (h1 : n = 36)
  (h2 : m = 30)
  (h3 : prob = 5/6) :
  (m : ℚ) / (n : ℚ) = prob :=
sorry

end probability_of_different_topics_l99_99519


namespace average_difference_l99_99116

theorem average_difference (t : ℚ) (ht : t = 4) :
  let m := (13 + 16 + 10 + 15 + 11) / 5
  let n := (16 + t + 3 + 13) / 4
  m - n = 4 :=
by
  sorry

end average_difference_l99_99116


namespace cyclic_quad_angles_l99_99578

theorem cyclic_quad_angles (A B C D : ℝ) (x : ℝ)
  (h_ratio : A = 5 * x ∧ B = 6 * x ∧ C = 4 * x)
  (h_cyclic : A + D = 180 ∧ B + C = 180):
  (B = 108) ∧ (C = 72) :=
by
  sorry

end cyclic_quad_angles_l99_99578


namespace digit_divisibility_by_7_l99_99856

theorem digit_divisibility_by_7 (d : ℕ) (h : d < 10) : (10000 + 100 * d + 10) % 7 = 0 ↔ d = 5 :=
by
  sorry

end digit_divisibility_by_7_l99_99856


namespace equation_has_three_solutions_l99_99738

theorem equation_has_three_solutions :
  ∃ (s : Finset ℝ), s.card = 3 ∧ ∀ x, x ∈ s ↔ x^2 * (x - 1) * (x - 2) = 0 := 
by
  sorry

end equation_has_three_solutions_l99_99738


namespace find_a_g_range_l99_99878

noncomputable def f (x a : ℝ) : ℝ := x^2 + 4 * a * x + 2 * a + 6
noncomputable def g (a : ℝ) : ℝ := 2 - a * |a - 1|

theorem find_a (x a : ℝ) :
  (∀ x, f x a ≥ 0) ∧ (∀ x, f x a = 0 → x^2 + 4 * a * x + 2 * a + 6 = 0) ↔ (a = -1 ∨ a = 3 / 2) :=
  sorry

theorem g_range :
  (∀ x, f x a ≥ 0) ∧ (-1 ≤ a ∧ a ≤ 3/2) → (∀ a, (5 / 4 ≤ g a ∧ g a ≤ 4)) :=
  sorry

end find_a_g_range_l99_99878


namespace problem_statement_l99_99401

variables {Line Plane : Type}

-- Defining the perpendicular relationship between a line and a plane
def perp (a : Line) (α : Plane) : Prop := sorry

-- Defining the parallel relationship between two planes
def para (α β : Plane) : Prop := sorry

-- The main statement to prove
theorem problem_statement (a : Line) (α β : Plane) (h1 : perp a α) (h2 : perp a β) : para α β := 
sorry

end problem_statement_l99_99401


namespace four_m_plus_one_2013_eq_neg_one_l99_99887

theorem four_m_plus_one_2013_eq_neg_one (m : ℝ) (h : |m| = m + 1) : (4 * m + 1) ^ 2013 = -1 := 
sorry

end four_m_plus_one_2013_eq_neg_one_l99_99887


namespace b_share_1500_l99_99956

theorem b_share_1500 (total_amount : ℕ) (parts_A parts_B parts_C : ℕ)
  (h_total_amount : total_amount = 4500)
  (h_ratio : (parts_A, parts_B, parts_C) = (2, 3, 4)) :
  parts_B * (total_amount / (parts_A + parts_B + parts_C)) = 1500 :=
by
  sorry

end b_share_1500_l99_99956


namespace determine_a_l99_99877

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (2 / (3 ^ x + 1)) - a

theorem determine_a (a : ℝ) :
  (∀ x : ℝ, f a (-x) = -f a x) ↔ a = 1 :=
by
  sorry

end determine_a_l99_99877


namespace find_x_l99_99364

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

theorem find_x (x : ℝ) (hx : x > 0) :
  distance (1, 3) (x, -4) = 15 → x = 1 + Real.sqrt 176 :=
by
  sorry

end find_x_l99_99364


namespace caitlin_bracelets_l99_99534

/-- 
Caitlin makes bracelets to sell at the farmer’s market every weekend. 
Each bracelet takes twice as many small beads as it does large beads. 
If each bracelet uses 12 large beads, and Caitlin has 528 beads with equal amounts of large and small beads, 
prove that Caitlin can make 11 bracelets for this weekend.
-/
theorem caitlin_bracelets (total_beads large_beads_per_bracelet small_beads_per_bracelet total_large_beads total_small_beads bracelets : ℕ)
  (h1 : total_beads = 528)
  (h2 : total_beads = total_large_beads + total_small_beads)
  (h3 : total_large_beads = total_small_beads)
  (h4 : large_beads_per_bracelet = 12)
  (h5 : small_beads_per_bracelet = 2 * large_beads_per_bracelet)
  (h6 : bracelets = total_small_beads / small_beads_per_bracelet) : 
  bracelets = 11 := 
by {
  sorry
}

end caitlin_bracelets_l99_99534


namespace sin_double_angle_l99_99077

theorem sin_double_angle (θ : Real) (h : Real.sin θ = 3/5) : Real.sin (2*θ) = 24/25 :=
by
  sorry

end sin_double_angle_l99_99077


namespace a_1000_value_l99_99429

theorem a_1000_value :
  ∃ (a : ℕ → ℤ), 
    (a 1 = 2010) ∧
    (a 2 = 2011) ∧
    (∀ n : ℕ, n ≥ 1 → a n + a (n + 1) + a (n + 2) = 2 * n + 3) ∧
    (a 1000 = 2676) :=
by {
  -- sorry is used to skip the proof
  sorry 
}

end a_1000_value_l99_99429


namespace problem_l99_99066

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.sin x + a * Real.cos x

theorem problem (a : ℝ) (h₀ : a < 0) (h₁ : ∀ x : ℝ, f x a ≤ 2) : f (π / 6) a = -1 :=
by {
  sorry
}

end problem_l99_99066


namespace product_gcd_lcm_4000_l99_99665

-- Definitions of gcd and lcm for the given numbers
def gcd_40_100 := Nat.gcd 40 100
def lcm_40_100 := Nat.lcm 40 100

-- Problem: Prove that the product of the gcd and lcm of 40 and 100 equals 4000
theorem product_gcd_lcm_4000 : gcd_40_100 * lcm_40_100 = 4000 := by
  sorry

end product_gcd_lcm_4000_l99_99665


namespace height_of_pyramid_l99_99649

theorem height_of_pyramid :
  let edge_cube := 6
  let edge_base_square_pyramid := 10
  let cube_volume := edge_cube ^ 3
  let sphere_volume := cube_volume
  let pyramid_volume := 2 * sphere_volume
  let base_area_square_pyramid := edge_base_square_pyramid ^ 2
  let height_pyramid := 12.96
  pyramid_volume = (1 / 3) * base_area_square_pyramid * height_pyramid :=
by
  sorry

end height_of_pyramid_l99_99649


namespace solve_a_perpendicular_l99_99749

theorem solve_a_perpendicular (a : ℝ) : 
  ((2 * a + 5) * (2 - a) + (a - 2) * (a + 3) = 0) ↔ (a = 2 ∨ a = -2) :=
by
  sorry

end solve_a_perpendicular_l99_99749


namespace matrix_linear_combination_l99_99590

noncomputable section

open Matrix

variables {α : Type*} [AddCommGroup α] [Module ℝ α]
variables (M : Matrix (Fin 2) (Fin 2) α) (v w : α)

def mv_eq_v : M.mul_vec v = ![2, -3] := sorry
def mw_eq_w : M.mul_vec w = ![4, 1] := sorry

theorem matrix_linear_combination :
  M.mul_vec (3 • v - 2 • w) = ![-2, -11] :=
begin
  sorry
end

end matrix_linear_combination_l99_99590


namespace solve_system_of_equations_l99_99927

theorem solve_system_of_equations (x1 x2 x3 x4 x5 y : ℝ) :
  x5 + x2 = y * x1 ∧
  x1 + x3 = y * x2 ∧
  x2 + x4 = y * x3 ∧
  x3 + x5 = y * x4 ∧
  x4 + x1 = y * x5 →
  (y = 2 ∧ x1 = x2 ∧ x2 = x3 ∧ x3 = x4 ∧ x4 = x5) ∨
  (y ≠ 2 ∧ (y^2 + y - 1 ≠ 0 ∧ x1 = 0 ∧ x2 = 0 ∧ x3 = 0 ∧ x4 = 0 ∧ x5 = 0) ∨
  (y^2 + y - 1 = 0 ∧ y = (1 / 2) * (-1 + Real.sqrt 5) ∨ y = (1 / 2) * (-1 - Real.sqrt 5) ∧
    ∃ a b : ℝ, x1 = a ∧ x2 = b ∧ x3 = y * b - a ∧ x4 = - y * (a + b) ∧ x5 = y * a - b))
:=
sorry

end solve_system_of_equations_l99_99927


namespace cocktail_cost_per_litre_is_accurate_l99_99625

noncomputable def mixed_fruit_juice_cost_per_litre : ℝ := 262.85
noncomputable def acai_berry_juice_cost_per_litre : ℝ := 3104.35
noncomputable def mixed_fruit_juice_litres : ℝ := 35
noncomputable def acai_berry_juice_litres : ℝ := 23.333333333333336

noncomputable def cocktail_total_cost : ℝ := 
  (mixed_fruit_juice_cost_per_litre * mixed_fruit_juice_litres) +
  (acai_berry_juice_cost_per_litre * acai_berry_juice_litres)

noncomputable def cocktail_total_volume : ℝ := 
  mixed_fruit_juice_litres + acai_berry_juice_litres

noncomputable def cocktail_cost_per_litre : ℝ := 
  cocktail_total_cost / cocktail_total_volume

theorem cocktail_cost_per_litre_is_accurate : 
  abs (cocktail_cost_per_litre - 1399.99) < 0.01 := by
  sorry

end cocktail_cost_per_litre_is_accurate_l99_99625


namespace ratio_u_v_l99_99324

theorem ratio_u_v (b : ℝ) (hb : b ≠ 0) (u v : ℝ) 
  (h1 : 0 = 8 * u + b) 
  (h2 : 0 = 4 * v + b) 
  : u / v = 1 / 2 := 
by sorry

end ratio_u_v_l99_99324


namespace cube_sum_l99_99782

theorem cube_sum (a b : ℝ) (h1 : a + b = 13) (h2 : a * b = 41) : a^3 + b^3 = 598 :=
by
  sorry

end cube_sum_l99_99782


namespace daniel_paid_more_l99_99010

noncomputable def num_slices : ℕ := 10
noncomputable def plain_cost : ℕ := 10
noncomputable def truffle_extra_cost : ℕ := 5
noncomputable def total_cost : ℕ := plain_cost + truffle_extra_cost
noncomputable def cost_per_slice : ℝ := total_cost / num_slices

noncomputable def truffle_slices_cost : ℝ := 5 * cost_per_slice
noncomputable def plain_slices_cost : ℝ := 5 * cost_per_slice

noncomputable def daniel_cost : ℝ := 5 * cost_per_slice + 2 * cost_per_slice
noncomputable def carl_cost : ℝ := 3 * cost_per_slice

noncomputable def payment_difference : ℝ := daniel_cost - carl_cost

theorem daniel_paid_more : payment_difference = 6 :=
by 
  sorry

end daniel_paid_more_l99_99010


namespace count_real_solutions_l99_99883

theorem count_real_solutions :
  ∃ x1 x2 : ℝ, (|x1-1| = |x1-2| + |x1-3| + |x1-4| ∧ |x2-1| = |x2-2| + |x2-3| + |x2-4|)
  ∧ (x1 ≠ x2) :=
sorry

end count_real_solutions_l99_99883


namespace turnips_bag_l99_99996

theorem turnips_bag (weights : List ℕ) (h : weights = [13, 15, 16, 17, 21, 24])
  (turnips: ℕ)
  (is_turnip : turnips ∈ weights)
  (o c : ℕ)
  (h1 : o + c = 106 - turnips)
  (h2 : c = 2 * o) :
  turnips = 13 ∨ turnips = 16 := by
  sorry

end turnips_bag_l99_99996


namespace jean_to_shirt_ratio_l99_99195

theorem jean_to_shirt_ratio (shirts_sold jeans_sold shirt_cost total_revenue: ℕ) (h1: shirts_sold = 20) (h2: jeans_sold = 10) (h3: shirt_cost = 10) (h4: total_revenue = 400) : 
(shirt_cost * shirts_sold + jeans_sold * ((total_revenue - (shirt_cost * shirts_sold)) / jeans_sold)) / (total_revenue - (shirt_cost * shirts_sold)) / jeans_sold = 2 := 
sorry

end jean_to_shirt_ratio_l99_99195


namespace ArcherInGoldenArmorProof_l99_99759

-- Definitions of the problem
variables (soldiers : Nat) (archers soldiersInGolden : Nat) 
variables (soldiersInBlack archersInGolden archersInBlack swordsmenInGolden swordsmenInBlack : Nat)
variables (truthfulSwordsmenInBlack lyingArchersInBlack lyingSwordsmenInGold truthfulArchersInGold : Nat)
variables (yesToGold yesToArcher yesToMonday : Nat)

-- Given conditions
def ProblemStatement : Prop :=
  soldiers = 55 ∧
  yesToGold = 44 ∧
  yesToArcher = 33 ∧
  yesToMonday = 22 ∧
  soldiers = archers + (soldiers - archers) ∧
  soldiers = soldiersInGolden + soldiersInBlack ∧
  archers = archersInGolden + archersInBlack ∧
  soldiersInGolden = archersInGolden + swordsmenInGolden ∧
  soldiersInBlack = archersInBlack + swordsmenInBlack ∧
  truthfulSwordsmenInBlack = swordsmenInBlack ∧
  lyingArchersInBlack = archersInBlack ∧
  lyingSwordsmenInGold = swordsmenInGolden ∧
  truthfulArchersInGold = archersInGolden ∧
  yesToGold = truthfulArchersInGold + (swordsmenInGolden + archersInBlack) ∧
  yesToArcher = truthfulArchersInGold + lyingSwordsmenInGold

-- Conclusion
def Conclusion : Prop :=
  archersInGolden = 22

-- Proof statement
theorem ArcherInGoldenArmorProof : ProblemStatement → Conclusion :=
by
  sorry

end ArcherInGoldenArmorProof_l99_99759


namespace general_term_formula_no_pos_int_for_S_n_gt_40n_plus_600_exists_pos_int_for_S_n_gt_40n_plus_600_l99_99559

noncomputable def arith_seq (n : ℕ) (d : ℝ) :=
  2 + (n - 1) * d

theorem general_term_formula :
  ∃ d, ∀ n, arith_seq n d = 2 ∨ arith_seq n d = 4 * n - 2 :=
by sorry

theorem no_pos_int_for_S_n_gt_40n_plus_600 :
  ∀ n, (arith_seq n 0) * n ≤ 40 * n + 600 :=
by sorry

theorem exists_pos_int_for_S_n_gt_40n_plus_600 :
  ∃ n, (arith_seq n 4) * n > 40 * n + 600 ∧ n = 31 :=
by sorry

end general_term_formula_no_pos_int_for_S_n_gt_40n_plus_600_exists_pos_int_for_S_n_gt_40n_plus_600_l99_99559


namespace fraction_of_satisfactory_grades_l99_99085

theorem fraction_of_satisfactory_grades (A B C D E F G : ℕ) (ha : A = 6) (hb : B = 3) (hc : C = 4) 
    (hd : D = 2) (he : E = 1) (hf : F = 7) (hg : G = 2) :
    (A + B + C + D + E) / (A + B + C + D + E + F + G) = 16 / 25 :=
by
  sorry

end fraction_of_satisfactory_grades_l99_99085


namespace golden_apples_first_six_months_l99_99659

-- Use appropriate namespaces
namespace ApolloProblem

-- Define the given conditions
def total_cost : ℕ := 54
def months_in_half_year : ℕ := 6

-- Prove that the number of golden apples charged for the first six months is 18
theorem golden_apples_first_six_months (X : ℕ) 
  (h1 : 6 * X + 6 * (2 * X) = total_cost) : 
  6 * X = 18 := 
sorry

end ApolloProblem

end golden_apples_first_six_months_l99_99659


namespace quadrilateral_diagonal_areas_relation_l99_99006

-- Defining the areas of the four triangles and the quadrilateral
variables (A B C D Q : ℝ)

-- Stating the property to be proven
theorem quadrilateral_diagonal_areas_relation 
  (H1 : Q = A + B + C + D) :
  A * B * C * D = ((A + B) * (B + C) * (C + D) * (D + A))^2 / Q^4 :=
by sorry

end quadrilateral_diagonal_areas_relation_l99_99006


namespace find_percentage_l99_99648

theorem find_percentage (P : ℕ) (h: (P / 100) * 180 - (1 / 3) * (P / 100) * 180 = 18) : P = 15 :=
sorry

end find_percentage_l99_99648


namespace ratio_of_x_intercepts_l99_99330

theorem ratio_of_x_intercepts (b : ℝ) (hb : b ≠ 0) (u v: ℝ)
  (h1: 0 = 8 * u + b) (h2: 0 = 4 * v + b) : u / v = 1 / 2 :=
by sorry

end ratio_of_x_intercepts_l99_99330


namespace convex_hexagon_largest_angle_l99_99479

theorem convex_hexagon_largest_angle 
  (x : ℝ)                                 -- Denote the measure of the third smallest angle as x.
  (angles : Fin 6 → ℝ)                     -- Define the angles as a function from Fin 6 to ℝ.
  (h1 : ∀ i : Fin 6, angles i = x + (i : ℝ) - 3)  -- The six angles in increasing order.
  (h2 : 0 < x - 3 ∧ x - 3 < 180)           -- Convex condition: each angle is between 0 and 180.
  (h3 : angles ⟨0⟩ + angles ⟨1⟩ + angles ⟨2⟩ + angles ⟨3⟩ + angles ⟨4⟩ + angles ⟨5⟩ = 720) -- Sum of interior angles of a hexagon.
  : (∃ a, a = angles ⟨5⟩ ∧ a = 122.5) :=   -- Prove the largest angle in this arrangement is 122.5.
sorry

end convex_hexagon_largest_angle_l99_99479


namespace inverse_variation_l99_99924

theorem inverse_variation (a b : ℝ) (k : ℝ) (h1 : a * b = k) (h2 : 800 * 0.5 = k) (h3 : a = 1600) : b = 0.25 :=
by 
  sorry

end inverse_variation_l99_99924


namespace hawkeye_charged_4_times_l99_99416

variables (C B L S : ℝ) (N : ℕ)
def hawkeye_charging_problem : Prop :=
  C = 3.5 ∧ B = 20 ∧ L = 6 ∧ S = B - L ∧ N = (S / C) → N = 4 

theorem hawkeye_charged_4_times : hawkeye_charging_problem C B L S N :=
by {
  repeat { sorry }
}

end hawkeye_charged_4_times_l99_99416


namespace smallest_x_mod_7_one_sq_l99_99966

theorem smallest_x_mod_7_one_sq (x : ℕ) (h : 1 < x) (hx : (x * x) % 7 = 1) : x = 6 :=
  sorry

end smallest_x_mod_7_one_sq_l99_99966


namespace GuntherFreeTime_l99_99414

def GuntherCleaning : Nat := 45 + 60 + 30 + 15

def TotalFreeTime : Nat := 180

theorem GuntherFreeTime : TotalFreeTime - GuntherCleaning = 30 := by
  sorry

end GuntherFreeTime_l99_99414


namespace chord_equation_l99_99222

-- Definitions and conditions
def parabola (x y : ℝ) := y^2 = 8 * x
def point_Q := (4, 1)

-- Statement to prove
theorem chord_equation :
  ∃ (m : ℝ) (c : ℝ), m = 4 ∧ c = -15 ∧
    ∀ (x y : ℝ), (parabola x y ∧ x + y = 8 ∧ y + y = 2) →
      4 * x - y = 15 :=
by
  sorry -- Proof elided

end chord_equation_l99_99222


namespace niko_total_profit_l99_99604

noncomputable def calculate_total_profit : ℝ :=
  let pairs := 9
  let price_per_pair := 2
  let discount_rate := 0.10
  let shipping_cost := 5
  let profit_4_pairs := 0.25
  let profit_5_pairs := 0.20
  let tax_rate := 0.05
  let cost_socks := pairs * price_per_pair
  let discount := discount_rate * cost_socks
  let cost_after_discount := cost_socks - discount
  let total_cost := cost_after_discount + shipping_cost
  let resell_price_4_pairs := (price_per_pair * (1 + profit_4_pairs)) * 4
  let resell_price_5_pairs := (price_per_pair * (1 + profit_5_pairs)) * 5
  let total_resell_price := resell_price_4_pairs + resell_price_5_pairs
  let sales_tax := tax_rate * total_resell_price
  let total_resell_price_after_tax := total_resell_price + sales_tax
  let total_profit := total_resell_price_after_tax - total_cost
  total_profit

theorem niko_total_profit : calculate_total_profit = 0.85 :=
by
  sorry

end niko_total_profit_l99_99604


namespace probability_same_plane_l99_99720

-- Define the number of vertices in a cube
def num_vertices : ℕ := 8

-- Define the number of vertices to be selected
def selection : ℕ := 4

-- Define the total number of ways to select 4 vertices out of 8
def total_ways : ℕ := Nat.choose num_vertices selection

-- Define the number of favorable ways to have 4 vertices lie in the same plane
def favorable_ways : ℕ := 12

-- Define the probability that the 4 selected vertices lie in the same plane
def probability : ℚ := favorable_ways / total_ways

-- The statement we need to prove
theorem probability_same_plane : probability = 6 / 35 := by
  sorry

end probability_same_plane_l99_99720


namespace price_of_other_stock_l99_99276

theorem price_of_other_stock (total_shares : ℕ) (total_spent : ℝ) (share_1_quantity : ℕ) (share_1_price : ℝ) :
  total_shares = 450 ∧ total_spent = 1950 ∧ share_1_quantity = 400 ∧ share_1_price = 3 →
  (750 / 50 = 15) :=
by sorry

end price_of_other_stock_l99_99276


namespace smallest_positive_integer_for_terminating_decimal_l99_99164

theorem smallest_positive_integer_for_terminating_decimal (n : ℕ) (h1 : n > 0) (h2 : (∀ m : ℕ, (m > n + 150) -> (m % (n + 150)) ∉ {2, 5})) :
  n = 50 :=
sorry

end smallest_positive_integer_for_terminating_decimal_l99_99164


namespace average_difference_l99_99804

theorem average_difference :
  let a1 := 20
  let a2 := 40
  let a3 := 60
  let b1 := 10
  let b2 := 70
  let b3 := 13
  (a1 + a2 + a3) / 3 - (b1 + b2 + b3) / 3 = 9 := by
sorry

end average_difference_l99_99804


namespace smallest_n_term_dec_l99_99154

theorem smallest_n_term_dec (n : ℕ) (h_pos : 0 < n) (h : ∀ d, 0 < d → d = n + 150 → ∀ p, p ∣ d → (p = 2 ∨ p = 5)) :
  n = 10 :=
by {
  sorry
}

end smallest_n_term_dec_l99_99154


namespace sufficient_not_necessary_condition_l99_99062

variable (x y : ℝ)

theorem sufficient_not_necessary_condition :
  (x > 1 ∧ y > 1) → (x + y > 2 ∧ x * y > 1) ∧
  ¬((x + y > 2 ∧ x * y > 1) → (x > 1 ∧ y > 1)) :=
by
  sorry

end sufficient_not_necessary_condition_l99_99062


namespace tangent_line_parabola_l99_99309

theorem tangent_line_parabola (d : ℝ) : 
  (∀ x y : ℝ, y = 3 * x + d → ∃! x y, y^2 = 12 * x) → d = 3 := 
by
  intro h
  -- Here, "h" would be our hypothesis where we assume the line is tangent to the parabola
  sorry

end tangent_line_parabola_l99_99309


namespace line_tangent_to_parabola_proof_l99_99296

noncomputable def line_tangent_to_parabola (d : ℝ) := (∀ x y : ℝ, y = 3 * x + d → y^2 = 12 * x) → d = 1

theorem line_tangent_to_parabola_proof (d : ℝ) (h : ∀ x y : ℝ, y = 3 * x + d → y^2 = 12 * x) : d = 1 :=
sorry

end line_tangent_to_parabola_proof_l99_99296


namespace inequality_sum_squares_products_l99_99193

theorem inequality_sum_squares_products {a b c d : ℝ} (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d) (h5 : a * b * c * d = 1) :
  a^2 + b^2 + c^2 + d^2 + a * b + a * c + a * d + b * c + b * d + c * d ≥ 10 :=
by
  sorry

end inequality_sum_squares_products_l99_99193


namespace second_smallest_relative_prime_210_l99_99398

theorem second_smallest_relative_prime_210 (x : ℕ) (h1 : x > 1) (h2 : Nat.gcd x 210 = 1) : x = 13 :=
sorry

end second_smallest_relative_prime_210_l99_99398


namespace mean_age_euler_family_l99_99113

theorem mean_age_euler_family :
  let ages := [6, 6, 9, 11, 13, 16]
  let total_children := 6
  let total_sum := 61
  (total_sum / total_children : ℝ) = (61 / 6 : ℝ) :=
by
  sorry

end mean_age_euler_family_l99_99113


namespace kylie_daisies_l99_99778

theorem kylie_daisies :
  ∀ (initial_daisies sister_daisies final_daisies daisies_given_to_mother total_daisies : ℕ),
    initial_daisies = 5 →
    sister_daisies = 9 →
    final_daisies = 7 →
    total_daisies = initial_daisies + sister_daisies →
    daisies_given_to_mother = total_daisies - final_daisies →
    daisies_given_to_mother * 2 = total_daisies :=
by
  intros initial_daisies sister_daisies final_daisies daisies_given_to_mother total_daisies h1 h2 h3 h4 h5
  sorry

end kylie_daisies_l99_99778


namespace find_x_when_y_is_72_l99_99595

theorem find_x_when_y_is_72 
  (x y : ℝ) (k : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) 
  (h_const : ∀ x y, 0 < x → 0 < y → x^2 * y = k)
  (h_initial : 9 * 8 = k)
  (h_y_72 : y = 72)
  (h_x2_factor : x^2 = 4 * 9) :
  x = 1 :=
sorry

end find_x_when_y_is_72_l99_99595


namespace linear_increase_y_l99_99102

-- Progressively increase x and track y

theorem linear_increase_y (Δx Δy : ℝ) (x_increase : Δx = 4) (y_increase : Δy = 10) :
  12 * (Δy / Δx) = 30 := by
  sorry

end linear_increase_y_l99_99102


namespace smallest_n_term_dec_l99_99158

theorem smallest_n_term_dec (n : ℕ) (h_pos : 0 < n) (h : ∀ d, 0 < d → d = n + 150 → ∀ p, p ∣ d → (p = 2 ∨ p = 5)) :
  n = 10 :=
by {
  sorry
}

end smallest_n_term_dec_l99_99158


namespace czechoslovak_inequality_l99_99019

-- Define the triangle and the points
structure Triangle (α : Type) [LinearOrderedRing α] :=
(A B C : α × α)

variables {α : Type} [LinearOrderedRing α]

-- Define the condition that O is on the segment AB but is not a vertex
def on_segment (O A B : α × α) : Prop :=
  ∃ x : α, 0 < x ∧ x < 1 ∧ O = (A.1 + x * (B.1 - A.1), A.2 + x * (B.2 - A.2))

-- Define the dot product for vectors
def dot (u v: α × α) : α := u.1 * v.1 + u.2 * v.2

-- Main statement
theorem czechoslovak_inequality (T : Triangle α) (O : α × α) (hO : on_segment O T.A T.B) :
  dot O T.C * dot T.A T.B < dot T.A O * dot T.B T.C + dot T.B O * dot T.A T.C :=
sorry

end czechoslovak_inequality_l99_99019


namespace ab_value_l99_99337

theorem ab_value (a b : ℝ) (h1 : a + b = 8) (h2 : a^3 + b^3 = 152) : a * b = 15 := by
  sorry

end ab_value_l99_99337


namespace sum_of_solutions_l99_99092

-- Define the system of equations as lean functions
def equation1 (x y : ℝ) : Prop := |x - 4| = |y - 10|
def equation2 (x y : ℝ) : Prop := |x - 10| = 3 * |y - 4|

-- Statement of the theorem
theorem sum_of_solutions : 
  ∃ (solutions : List (ℝ × ℝ)), 
    (∀ (sol : ℝ × ℝ), sol ∈ solutions → equation1 sol.1 sol.2 ∧ equation2 sol.1 sol.2) ∧ 
    (List.sum (solutions.map (fun sol => sol.1 + sol.2)) = 24) :=
  sorry

end sum_of_solutions_l99_99092


namespace max_marks_is_667_l99_99369

-- Definitions based on the problem's conditions
def pass_threshold (M : ℝ) : ℝ := 0.45 * M
def student_score : ℝ := 225
def failed_by : ℝ := 75
def passing_marks := student_score + failed_by

-- The actual theorem stating that if the conditions are met, then the maximum marks M is 667
theorem max_marks_is_667 : ∃ M : ℝ, pass_threshold M = passing_marks ∧ M = 667 :=
by
  sorry -- Proof is omitted as per the instructions

end max_marks_is_667_l99_99369


namespace range_of_a_l99_99960

def is_monotonically_increasing (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x y, (0 < x) → (x < y) → (f x ≤ f y)

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.log x - a * (x - 1) / (x + 1)

theorem range_of_a (a : ℝ) : 
  is_monotonically_increasing (f a) a → a ≤ 2 :=
sorry

end range_of_a_l99_99960


namespace remainder_1125_1127_1129_div_12_l99_99644

theorem remainder_1125_1127_1129_div_12 :
  (1125 * 1127 * 1129) % 12 = 3 :=
by
  -- Proof can be written here
  sorry

end remainder_1125_1127_1129_div_12_l99_99644


namespace turnips_bag_l99_99994

theorem turnips_bag (weights : List ℕ) (h : weights = [13, 15, 16, 17, 21, 24])
  (turnips: ℕ)
  (is_turnip : turnips ∈ weights)
  (o c : ℕ)
  (h1 : o + c = 106 - turnips)
  (h2 : c = 2 * o) :
  turnips = 13 ∨ turnips = 16 := by
  sorry

end turnips_bag_l99_99994


namespace num_archers_golden_armor_proof_l99_99760
noncomputable section

structure Soldier :=
  (is_archer : Bool)
  (is_golden : Bool)
  (tells_truth : Bool)

def count_soldiers (soldiers : List Soldier) : Nat :=
  soldiers.length

def count_truthful_responses (soldiers : List Soldier) (q : Soldier → Bool) : Nat :=
  (soldiers.filter (λ s => s.tells_truth)).count q

def count_lying_responses (soldiers : List Soldier) (q : Soldier → Bool) : Nat :=
  (soldiers.filter (λ s => ¬s.tells_truth)).count q

def num_archers_golden_armor (soldiers : List Soldier) : Nat :=
  (soldiers.filter (λ s => s.is_archer ∧ s.is_golden)).length

theorem num_archers_golden_armor_proof (soldiers : List Soldier)
  (h1 : count_soldiers soldiers = 55)
  (h2 : count_truthful_responses soldiers (λ s => s.is_golden) + 
        count_lying_responses soldiers (λ s => s.is_golden) = 44)
  (h3 : count_truthful_responses soldiers (λ s => s.is_archer) + 
        count_lying_responses soldiers (λ s => s.is_archer) = 33)
  (h4 : count_truthful_responses soldiers (λ s => true) + 
        count_lying_responses soldiers (λ s => false) = 22) :
  num_archers_golden_armor soldiers = 22 := by
  sorry

end num_archers_golden_armor_proof_l99_99760


namespace exists_periodic_sequence_of_period_ge_two_l99_99942

noncomputable def periodic_sequence (x : ℕ → ℝ) (p : ℕ) : Prop :=
  ∀ n, x (n + p) = x n

theorem exists_periodic_sequence_of_period_ge_two :
  ∀ (p : ℕ), p ≥ 2 →
  ∃ (x : ℕ → ℝ), periodic_sequence x p ∧ 
  ∀ n, x (n + 1) = x n - (1 / x n) :=
by {
  sorry
}

end exists_periodic_sequence_of_period_ge_two_l99_99942


namespace find_x_range_l99_99873

variable (f : ℝ → ℝ)

def even_function (f : ℝ → ℝ) := ∀ x : ℝ, f x = f (-x)

def decreasing_on_nonnegative (f : ℝ → ℝ) :=
  ∀ x1 x2 : ℝ, x1 ≥ 0 → x2 ≥ 0 → x1 ≠ x2 → (f x2 - f x1) / (x2 - x1) < 0

theorem find_x_range (f : ℝ → ℝ)
  (h1 : even_function f)
  (h2 : decreasing_on_nonnegative f)
  (h3 : f (1/3) = 3/4)
  (h4 : ∀ x : ℝ, 4 * f (Real.logb (1/8) x) > 3) :
  ∀ x : ℝ, (1/2 < x ∧ x < 2) ↔ True := sorry

end find_x_range_l99_99873


namespace M_inter_N_eq_interval_l99_99915

def M : Set ℝ := {x | x^2 - x ≤ 0}
def N : Set ℝ := {x | -1 < x ∧ x < 1}

theorem M_inter_N_eq_interval : (M ∩ N) = {x | 0 ≤ x ∧ x < 1} := 
  sorry

end M_inter_N_eq_interval_l99_99915


namespace one_third_percent_of_200_l99_99922

theorem one_third_percent_of_200 : ((1206 / 3) / 200) * 100 = 201 := by
  sorry

end one_third_percent_of_200_l99_99922


namespace graph_description_l99_99186

theorem graph_description : ∀ x y : ℝ, (x + y)^2 = 2 * (x^2 + y^2) → x = 0 ∧ y = 0 :=
by 
  sorry

end graph_description_l99_99186


namespace large_rectangle_perimeter_l99_99694

-- Definitions for conditions
def rectangle_area (l b : ℝ) := l * b
def is_large_rectangle_perimeter (l b perimeter : ℝ) := perimeter = 2 * (l + b)

-- Statement of the theorem
theorem large_rectangle_perimeter :
  ∃ (l b : ℝ), rectangle_area l b = 8 ∧ 
               (∀ l_rect b_rect: ℝ, is_large_rectangle_perimeter l_rect b_rect 32) :=
by
  sorry

end large_rectangle_perimeter_l99_99694


namespace evaluate_S_l99_99616

variables {x : ℕ → ℝ} {S : ℝ}

-- setting up the conditions as hypothesis
def condition (x : ℕ → ℝ) := 
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 1004 → x n + (n + 2) = (∑ k in finset.range 1004, x k) + 1005

theorem evaluate_S (h : condition x) :
  ∥∑ n in finset.range 1004, x n∥.toNat = 498 := by
  sorry

end evaluate_S_l99_99616


namespace find_x_l99_99879

def vector := (ℝ × ℝ)

def a (x : ℝ) : vector := (x, 2)
def b : vector := (1, -1)

-- Dot product of two vectors
def dot_product (v1 v2 : vector) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

-- Orthogonality condition rewritten in terms of dot product
def orthogonal (v1 v2 : vector) : Prop := dot_product v1 v2 = 0

-- Main theorem to prove
theorem find_x (x : ℝ) (h : orthogonal ((a x).1 - b.1, (a x).2 - b.2) b) : x = 4 :=
by sorry

end find_x_l99_99879


namespace smallest_n_for_terminating_decimal_l99_99149

theorem smallest_n_for_terminating_decimal :
  ∃ n : ℕ, n > 0 ∧ (∀ m : ℕ, m > 0 ∧ (∀ p: ℕ, (p ∣ (m + 150) → (p = 2 ∨ p = 5)) → m ≥ n)) ∧
  (∃ two_five_factors : ℕ, (two_five_factors = 5 ^ 3 * 2 ^ 3) ∧ (n + 150 = two_five_factors)) := 
by 
   exact ⟨50, by sorry⟩

end smallest_n_for_terminating_decimal_l99_99149


namespace Bernardo_has_higher_probability_l99_99663

-- Defining the sets from which Bernardo and Silvia pick their numbers
def BernardoSet := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
def SilviaSet := {1, 2, 3, 4, 5, 6, 7, 8, 9}

-- Defining the problem statement, which is the probability comparison as per the given conditions
noncomputable def probability_Bernardo_larger : ℚ := 13 / 20

-- The Lean 4 theorem statement for the problem
theorem Bernardo_has_higher_probability :
  ∃ (prob : ℚ), prob = probability_Bernardo_larger :=
begin
  use probability_Bernardo_larger,
  sorry
end

end Bernardo_has_higher_probability_l99_99663


namespace polar_equation_of_circle_c_range_of_op_oq_l99_99435

noncomputable def circle_param_eq (φ : ℝ) : ℝ × ℝ :=
  (1 + Real.cos φ, Real.sin φ)

noncomputable def line_kl_eq (θ : ℝ) : ℝ :=
  3 * Real.sqrt 3 / (Real.sin θ + Real.sqrt 3 * Real.cos θ)

theorem polar_equation_of_circle_c :
  ∀ θ : ℝ, ∃ ρ : ℝ, ρ = 2 * Real.cos θ :=
by sorry

theorem range_of_op_oq (θ₁ : ℝ) (hθ : 0 < θ₁ ∧ θ₁ < Real.pi / 2) :
  0 < (2 * Real.cos θ₁) * (3 * Real.sqrt 3 / (Real.sin θ₁ + Real.sqrt 3 * Real.cos θ₁)) ∧
  (2 * Real.cos θ₁) * (3 * Real.sqrt 3 / (Real.sin θ₁ + Real.sqrt 3 * Real.cos θ₁)) < 6 :=
by sorry

end polar_equation_of_circle_c_range_of_op_oq_l99_99435


namespace remainder_5_pow_5_pow_5_pow_5_mod_500_l99_99045

-- Conditions
def λ (n : ℕ) : ℕ := n.gcd20p1.factorial5div
def M : ℕ := 5 ^ (5 ^ 5)

-- Theorem: Prove the remainder
theorem remainder_5_pow_5_pow_5_pow_5_mod_500 :
  M % 500 = 125 :=
by sorry

end remainder_5_pow_5_pow_5_pow_5_mod_500_l99_99045


namespace units_digit_of_m_squared_plus_two_to_m_is_3_l99_99596

def m := 2017^2 + 2^2017

theorem units_digit_of_m_squared_plus_two_to_m_is_3 : (m^2 + 2^m) % 10 = 3 := 
by 
  sorry

end units_digit_of_m_squared_plus_two_to_m_is_3_l99_99596


namespace fraction_orange_juice_in_large_container_l99_99949

-- Definitions according to the conditions
def pitcher1_capacity : ℕ := 800
def pitcher2_capacity : ℕ := 500
def pitcher1_fraction_orange_juice : ℚ := 1 / 4
def pitcher2_fraction_orange_juice : ℚ := 3 / 5

-- Prove the fraction of orange juice
theorem fraction_orange_juice_in_large_container :
  ( (pitcher1_capacity * pitcher1_fraction_orange_juice + pitcher2_capacity * pitcher2_fraction_orange_juice) / 
    (pitcher1_capacity + pitcher2_capacity) ) = 5 / 13 :=
by
  sorry

end fraction_orange_juice_in_large_container_l99_99949


namespace height_of_oil_truck_tank_l99_99359

/-- 
Given that a stationary oil tank is a right circular cylinder 
with a radius of 100 feet and its oil level dropped by 0.025 feet,
proving that if this oil is transferred to a right circular 
cylindrical oil truck's tank with a radius of 5 feet, then the 
height of the oil in the truck's tank will be 10 feet. 
-/
theorem height_of_oil_truck_tank
    (radius_stationary : ℝ) (height_drop_stationary : ℝ) (radius_truck : ℝ) 
    (height_truck : ℝ) (π : ℝ)
    (h1 : radius_stationary = 100)
    (h2 : height_drop_stationary = 0.025)
    (h3 : radius_truck = 5)
    (pi_approx : π = 3.14159265) :
    height_truck = 10 :=
by
    sorry

end height_of_oil_truck_tank_l99_99359


namespace geometric_sequence_problem_l99_99554

noncomputable def geometric_sum (a1 q : ℝ) (n : ℕ) : ℝ :=
if q = 1 then n * a1 else a1 * (1 - q ^ n) / (1 - q)

theorem geometric_sequence_problem
  (a1 q : ℝ) (a2 : ℝ := a1 * q) (a5 : ℝ := a1 * q^4)
  (S2 : ℝ := geometric_sum a1 q 2) (S4 : ℝ := geometric_sum a1 q 4)
  (h1 : 8 * a2 + a5 = 0) :
  S4 / S2 = 5 :=
by
  sorry

end geometric_sequence_problem_l99_99554


namespace correct_statements_identification_l99_99017

-- Definitions based on given conditions
def syntheticMethodCauseToEffect := True
def syntheticMethodForward := True
def analyticMethodEffectToCause := True
def analyticMethodIndirect := False
def analyticMethodBackward := True

-- The main statement to be proved
theorem correct_statements_identification :
  (syntheticMethodCauseToEffect = True) ∧ 
  (syntheticMethodForward = True) ∧ 
  (analyticMethodEffectToCause = True) ∧ 
  (analyticMethodBackward = True) ∧ 
  (analyticMethodIndirect = False) :=
by
  sorry

end correct_statements_identification_l99_99017


namespace tv_station_ads_l99_99011

theorem tv_station_ads (n m : ℕ) :
  n > 1 → 
  ∃ (an : ℕ → ℕ), 
  (an 0 = m) ∧ 
  (∀ k, 1 ≤ k ∧ k < n → an k = an (k - 1) - (k + (1 / 8) * (an (k - 1) - k))) ∧
  an n = 0 →
  (n = 7 ∧ m = 49) :=
by
  intro h
  exists sorry
  sorry

-- The proof steps are omitted

end tv_station_ads_l99_99011


namespace tangent_line_parabola_l99_99310

theorem tangent_line_parabola (d : ℝ) : 
  (∀ x y : ℝ, y = 3 * x + d → ∃! x y, y^2 = 12 * x) → d = 3 := 
by
  intro h
  -- Here, "h" would be our hypothesis where we assume the line is tangent to the parabola
  sorry

end tangent_line_parabola_l99_99310


namespace line_tangent_parabola_unique_d_l99_99301

theorem line_tangent_parabola_unique_d :
  (∃ d : ℝ, ∀ x : ℝ, y = 3 * x + d -> y^2 = 12 * x) -> d = 1 := by
sorry

end line_tangent_parabola_unique_d_l99_99301


namespace bug_returns_starting_vertex_eighth_move_l99_99828

/-- Initial Probability Definition -/
def P : ℕ → ℚ
| 0 => 1
| n + 1 => 1 / 3 * (1 - P n)

-- Define the theorem to prove
theorem bug_returns_starting_vertex_eighth_move :
  let m := 547
  let n := 2187
  P 8 = m / n ∧ m + n = 2734 :=
by
  -- sorry to defer the actual proof
  sorry

end bug_returns_starting_vertex_eighth_move_l99_99828


namespace smallest_n_for_terminating_fraction_l99_99176

-- Define what it means for a number to be a product of only prime factors of 2 and 5
def isTerminatingDenominator (d : ℕ) : Prop := ∃ (a b : ℕ), d = 2^a * 5^b

-- The main statement to prove
theorem smallest_n_for_terminating_fraction : 
  ∃ (n : ℕ), 0 < n ∧ isTerminatingDenominator (n + 150) ∧ 
  ∀ (m : ℕ), (0 < m → isTerminatingDenominator (m + 150) → n ≤ m)
:= sorry

end smallest_n_for_terminating_fraction_l99_99176


namespace angle_A_measure_l99_99894

theorem angle_A_measure 
  (B : ℝ) 
  (angle_in_smaller_triangle : ℝ) 
  (sum_of_triangle_angles_eq_180 : ∀ (x y z : ℝ), x + y + z = 180)
  (C : ℝ) 
  (angle_pair_linear : ∀ (x y : ℝ), x + y = 180) 
  (A : ℝ) 
  (C_eq_180_minus_B : C = 180 - B) 
  (A_eq_180_minus_angle_in_smaller_triangle_minus_C : 
    A = 180 - angle_in_smaller_triangle - C) :
  A = 70 :=
by
  sorry

end angle_A_measure_l99_99894


namespace min_value_expression_ge_072_l99_99912

theorem min_value_expression_ge_072 (x y z : ℝ) 
  (hx : -0.5 ≤ x ∧ x ≤ 0.5) 
  (hy : |y| ≤ 0.5) 
  (hz : 0 ≤ z ∧ z < 1) :
  ((1 / ((1 - x) * (1 - y) * (1 - z))) - (1 / ((2 + x) * (2 + y) * (2 + z)))) ≥ 0.72 := sorry

end min_value_expression_ge_072_l99_99912


namespace smallest_n_for_terminating_fraction_l99_99172

-- Define what it means for a number to be a product of only prime factors of 2 and 5
def isTerminatingDenominator (d : ℕ) : Prop := ∃ (a b : ℕ), d = 2^a * 5^b

-- The main statement to prove
theorem smallest_n_for_terminating_fraction : 
  ∃ (n : ℕ), 0 < n ∧ isTerminatingDenominator (n + 150) ∧ 
  ∀ (m : ℕ), (0 < m → isTerminatingDenominator (m + 150) → n ≤ m)
:= sorry

end smallest_n_for_terminating_fraction_l99_99172


namespace polynomial_expansion_l99_99073

theorem polynomial_expansion (a_0 a_1 a_2 a_3 a_4 : ℤ)
  (h1 : a_0 + a_1 + a_2 + a_3 + a_4 = 5^4)
  (h2 : a_0 - a_1 + a_2 - a_3 + a_4 = 1) :
  (a_0 + a_2 + a_4)^2 - (a_1 + a_3)^2 = 625 :=
by
  sorry

end polynomial_expansion_l99_99073


namespace compute_ab_l99_99932

theorem compute_ab (a b : ℝ) 
  (h1 : b^2 - a^2 = 25) 
  (h2 : a^2 + b^2 = 49) : 
  |a * b| = Real.sqrt 444 := 
by 
  sorry

end compute_ab_l99_99932


namespace prove_area_and_sum_l99_99839

-- Define the coordinates of the vertices of the quadrilateral.
variables (a b : ℤ)

-- Define the non-computable requirements related to the problem.
noncomputable def problem_statement : Prop :=
  ∃ (a b : ℤ), a > 0 ∧ b > 0 ∧ a > b ∧ (4 * a * b = 32) ∧ (a + b = 5)

theorem prove_area_and_sum : problem_statement := 
sorry

end prove_area_and_sum_l99_99839


namespace smallest_n_for_terminating_decimal_l99_99146

def is_terminating_decimal (n d : ℕ) : Prop :=
  ∀ (m : ℕ), d = 2^m ∨ d = 5^m ∨ d = (2^m) * (5 : ℕ) ∨ d = (5^m) * (2 : ℕ)
  
theorem smallest_n_for_terminating_decimal :
  ∃ n : ℕ, 0 < n ∧ is_terminating_decimal n (n + 150) ∧ (∀ m: ℕ, (is_terminating_decimal m (m + 150) ∧ 0 < m) → n ≤ m) :=
sorry

end smallest_n_for_terminating_decimal_l99_99146


namespace max_students_exam_l99_99373

/--
An exam contains 4 multiple-choice questions, each with three options (A, B, C). Several students take the exam.
For any group of 3 students, there is at least one question where their answers are all different.
Each student answers all questions. Prove that the maximum number of students who can take the exam is 9.
-/
theorem max_students_exam (n : ℕ) (A B C : ℕ → ℕ → ℕ) (q : ℕ) :
  (∀ (s1 s2 s3 : ℕ), ∃ (q : ℕ), (1 ≤ q ∧ q ≤ 4) ∧ (A s1 q ≠ A s2 q ∧ A s1 q ≠ A s3 q ∧ A s2 q ≠ A s3 q)) →
  q = 4 ∧ (∀ s, 1 ≤ s → s ≤ n) → n ≤ 9 :=
by
  sorry

end max_students_exam_l99_99373


namespace total_dots_not_visible_l99_99695

theorem total_dots_not_visible :
  let total_dots := 4 * 21
  let visible_sum := 1 + 2 + 3 + 3 + 4 + 5 + 5 + 6
  total_dots - visible_sum = 55 :=
by
  sorry

end total_dots_not_visible_l99_99695


namespace travel_time_and_speed_l99_99734

theorem travel_time_and_speed :
  (total_time : ℝ) = 5.5 →
  (bus_whole_journey : ℝ) = 1 →
  (bus_half_journey : ℝ) = bus_whole_journey / 2 →
  (walk_half_journey : ℝ) = total_time - bus_half_journey →
  (walk_whole_journey : ℝ) = 2 * walk_half_journey →
  (bus_speed_factor : ℝ) = walk_whole_journey / bus_whole_journey →
  walk_whole_journey = 10 ∧ bus_speed_factor = 10 := by
  intros h1 h2 h3 h4 h5 h6
  sorry

end travel_time_and_speed_l99_99734


namespace henry_seashells_l99_99248

theorem henry_seashells (H L : ℕ) (h1 : H + 24 + L = 59) (h2 : H + 24 + (3 * L) / 4 = 53) : H = 11 := by
  sorry

end henry_seashells_l99_99248


namespace total_area_l99_99847

variable (A : ℝ)

-- Defining the conditions
def first_carpet : Prop := 0.55 * A = 36
def second_carpet : Prop := 0.25 * A = A * 0.25
def third_carpet : Prop := 0.15 * A = 18 + 6
def remaining_floor : Prop := 0.05 * A + 0.55 * A + 0.25 * A + 0.15 * A = A

-- Main theorem to prove the total area
theorem total_area : first_carpet A → second_carpet A → third_carpet A → remaining_floor A → A = 65.45 :=
by
  sorry

end total_area_l99_99847


namespace Travis_spends_312_dollars_on_cereal_l99_99131

/-- Given that Travis eats 2 boxes of cereal a week, each box costs $3.00, 
and there are 52 weeks in a year, he spends $312.00 on cereal in a year. -/
theorem Travis_spends_312_dollars_on_cereal
  (boxes_per_week : ℕ)
  (cost_per_box : ℝ)
  (weeks_in_year : ℕ)
  (consumption : boxes_per_week = 2)
  (cost : cost_per_box = 3)
  (weeks : weeks_in_year = 52) :
  boxes_per_week * cost_per_box * weeks_in_year = 312 :=
by
  simp [consumption, cost, weeks]
  norm_num
  sorry

end Travis_spends_312_dollars_on_cereal_l99_99131


namespace find_f_expression_l99_99602

noncomputable def f (x : ℝ) := -2 * Real.cos (2 * x)

noncomputable def g (x : ℝ) := 2 * Real.sin (2 * x + Real.pi / 6)

theorem find_f_expression :
    (∀ x : ℝ, f (x) = g (x - Real.pi / 3)) → (f = -2 * Real.cos ∘ (λ x, 2 * x)) :=
by
  intro h
  funext x
  specialize h x
  simp only [f, g] at h
  rw [← Real.sin_add_pi_div_two] at h
  exact h
  sorry

end find_f_expression_l99_99602


namespace marie_keeps_lollipops_l99_99458

def total_lollipops (raspberry mint blueberry coconut : ℕ) : ℕ :=
  raspberry + mint + blueberry + coconut

def lollipops_per_friend (total friends : ℕ) : ℕ :=
  total / friends

def lollipops_kept (total friends : ℕ) : ℕ :=
  total % friends

theorem marie_keeps_lollipops :
  lollipops_kept (total_lollipops 75 132 9 315) 13 = 11 :=
by
  sorry

end marie_keeps_lollipops_l99_99458


namespace pq_even_impossible_l99_99574

theorem pq_even_impossible {p q : ℤ} (h : (p^2 + q^2 + p*q) % 2 = 1) : ¬(p % 2 = 0 ∧ q % 2 = 0) :=
by
  sorry

end pq_even_impossible_l99_99574


namespace sum_of_projections_l99_99770

-- Define the given lengths of the triangle sides
def a : ℝ := 5
def b : ℝ := 7
def c : ℝ := 6

-- Define the centroid projections to the sides
noncomputable def GP : ℝ := 2 * real.sqrt 6 / 3
noncomputable def GQ : ℝ := 4 * real.sqrt 6 / 7
noncomputable def GR : ℝ := 4 * real.sqrt 6 / 5

-- Prove the sum of GP, GQ, and GR
theorem sum_of_projections : GP + GQ + GR = (122 * real.sqrt 6) / 105 :=
by
  sorry

end sum_of_projections_l99_99770


namespace problem_l99_99800

theorem problem (a b c d e : ℤ) 
  (h1 : a - b + c - e = 7)
  (h2 : b - c + d + e = 8)
  (h3 : c - d + a - e = 4)
  (h4 : d - a + b + e = 3) :
  a + b + c + d + e = 22 := by
  sorry

end problem_l99_99800


namespace montana_more_than_ohio_l99_99455

-- Define the total number of combinations for Ohio and Montana
def ohio_combinations : ℕ := 26^4 * 10^3
def montana_combinations : ℕ := 26^5 * 10^2

-- The total number of combinations from both states
def ohio_total : ℕ := ohio_combinations
def montana_total : ℕ := montana_combinations

-- Prove the difference
theorem montana_more_than_ohio : montana_total - ohio_total = 731161600 := by
  sorry

end montana_more_than_ohio_l99_99455


namespace dima_always_wins_l99_99541

theorem dima_always_wins (n : ℕ) (P : Prop) : 
  (∀ (gosha dima : ℕ → Prop), 
    (∀ k : ℕ, k < n → (gosha k ∨ dima k))
    ∧ (∀ i : ℕ, i < 14 → (gosha i ∨ dima i))
    ∧ (∃ j : ℕ, j ≤ n ∧ (∃ k ≤ j + 7, dima k))
    ∧ (∃ l : ℕ, l ≤ 14 ∧ (∃ m ≤ l + 7, dima m))
    → P) → P := sorry

end dima_always_wins_l99_99541


namespace smallest_n_for_terminating_decimal_l99_99142

def is_terminating_decimal (n d : ℕ) : Prop :=
  ∀ (m : ℕ), d = 2^m ∨ d = 5^m ∨ d = (2^m) * (5 : ℕ) ∨ d = (5^m) * (2 : ℕ)
  
theorem smallest_n_for_terminating_decimal :
  ∃ n : ℕ, 0 < n ∧ is_terminating_decimal n (n + 150) ∧ (∀ m: ℕ, (is_terminating_decimal m (m + 150) ∧ 0 < m) → n ≤ m) :=
sorry

end smallest_n_for_terminating_decimal_l99_99142


namespace ratio_of_buttons_to_magnets_per_earring_l99_99283

-- Definitions related to the problem statement
def gemstones_per_button : ℕ := 3
def magnets_per_earring : ℕ := 2
def sets_of_earrings : ℕ := 4
def required_gemstones : ℕ := 24

-- Problem statement translation into Lean 4
theorem ratio_of_buttons_to_magnets_per_earring :
  (required_gemstones / gemstones_per_button / (sets_of_earrings * 2)) = 1 / 2 := by
  sorry

end ratio_of_buttons_to_magnets_per_earring_l99_99283


namespace total_cases_after_third_day_l99_99529

-- Definitions for the conditions
def day1_cases : Nat := 2000
def day2_new_cases : Nat := 500
def day2_recoveries : Nat := 50
def day3_new_cases : Nat := 1500
def day3_recoveries : Nat := 200

-- Theorem stating the total number of cases after the third day
theorem total_cases_after_third_day : day1_cases + (day2_new_cases - day2_recoveries) + (day3_new_cases - day3_recoveries) = 3750 :=
by
  sorry

end total_cases_after_third_day_l99_99529


namespace number_of_restaurants_l99_99972

def units_in_building : ℕ := 300
def residential_units := units_in_building / 2
def remaining_units := units_in_building - residential_units
def restaurants := remaining_units / 2

theorem number_of_restaurants : restaurants = 75 :=
by
  sorry

end number_of_restaurants_l99_99972


namespace turnip_bag_weights_l99_99992

theorem turnip_bag_weights :
  ∃ (T : ℕ), (T = 13 ∨ T = 16) ∧ 
  (∀ T', T' ≠ T → (
    (2 * total_weight_of_has_weight_1.not_turnip T' O + O = 106 - T') → 
    let turnip_condition := 106 - T in
    turnip_condition % 3 = 0)) ∧
  ∀ (bag_weights : List ℕ) (other : bag_weights = [13, 15, 16, 17, 21, 24] ∧ 
                                          List.length bag_weights = 6),
  True := by sorry

end turnip_bag_weights_l99_99992


namespace strawberry_harvest_l99_99192

theorem strawberry_harvest
  (length : ℕ) (width : ℕ)
  (plants_per_sqft : ℕ) (yield_per_plant : ℕ)
  (garden_area : ℕ := length * width) 
  (total_plants : ℕ := plants_per_sqft * garden_area) 
  (expected_strawberries : ℕ := yield_per_plant * total_plants) :
  length = 10 ∧ width = 12 ∧ plants_per_sqft = 5 ∧ yield_per_plant = 8 → 
  expected_strawberries = 4800 := by
  sorry

end strawberry_harvest_l99_99192


namespace simple_interest_rate_l99_99189

theorem simple_interest_rate 
  (P A T : ℝ) 
  (hP : P = 900) 
  (hA : A = 950) 
  (hT : T = 5) 
  : (A - P) * 100 / (P * T) = 1.11 :=
by
  sorry

end simple_interest_rate_l99_99189


namespace problem_statement_l99_99063

def f (x : ℝ) : ℝ := x^3 + x^2 + 2

def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem problem_statement : odd_function f → f (-2) = -14 := by
  intro h
  sorry

end problem_statement_l99_99063


namespace speed_of_man_l99_99843

-- Define all given conditions and constants

def trainLength : ℝ := 110 -- in meters
def trainSpeed : ℝ := 40 -- in km/hr
def timeToPass : ℝ := 8.799296056315494 -- in seconds

-- We want to prove that the speed of the man is approximately 4.9968 km/hr
theorem speed_of_man :
  let trainSpeedMS := trainSpeed * (1000 / 3600)
  let relativeSpeed := trainLength / timeToPass
  let manSpeedMS := relativeSpeed - trainSpeedMS
  let manSpeedKMH := manSpeedMS * (3600 / 1000)
  abs (manSpeedKMH - 4.9968) < 0.01 := sorry

end speed_of_man_l99_99843


namespace smallest_positive_integer_for_terminating_decimal_l99_99161

theorem smallest_positive_integer_for_terminating_decimal (n : ℕ) (h1 : n > 0) (h2 : (∀ m : ℕ, (m > n + 150) -> (m % (n + 150)) ∉ {2, 5})) :
  n = 50 :=
sorry

end smallest_positive_integer_for_terminating_decimal_l99_99161


namespace largest_r_satisfying_condition_l99_99797

theorem largest_r_satisfying_condition :
  ∃ M : ℕ, ∀ (a : ℕ → ℕ) (r : ℝ) (h : ∀ n : ℕ, a n ≤ a (n + 2) ∧ a (n + 2) ≤ Real.sqrt (a n ^ 2 + r * a (n + 1))),
  (∀ n : ℕ, n ≥ M → a (n + 2) = a n) → r = 2 := 
by
  sorry

end largest_r_satisfying_condition_l99_99797


namespace problem1_problem2_l99_99914

open Nat

def seq (a : ℕ → ℕ) :=
  ∀ n : ℕ, n > 0 → a n < a (n + 1) ∧ a n > 0

def b_seq (a : ℕ → ℕ) (n : ℕ) :=
  a (a n)

def c_seq (a : ℕ → ℕ) (n : ℕ) :=
  a (a n + 1)

theorem problem1 (a : ℕ → ℕ) (h_seq : seq a) (h_bseq : ∀ n, n > 0 → b_seq a n = 3 * n) : a 1 = 2 ∧ c_seq a 1 = 6 :=
  sorry

theorem problem2 (a : ℕ → ℕ) (h_seq : seq a) (h_cseq : ∀ n, n > 0 → c_seq a (n + 1) - c_seq a n = 1) : 
  ∀ n, n > 0 → a (n + 1) - a n = 1 :=
  sorry

end problem1_problem2_l99_99914


namespace pat_stickers_at_end_of_week_l99_99921

def initial_stickers : ℕ := 39
def monday_transaction : ℕ := 15
def tuesday_transaction : ℕ := 22
def wednesday_transaction : ℕ := 10
def thursday_trade_net_loss : ℕ := 4
def friday_find : ℕ := 5

def final_stickers (initial : ℕ) (mon : ℕ) (tue : ℕ) (wed : ℕ) (thu : ℕ) (fri : ℕ) : ℕ :=
  initial + mon - tue + wed - thu + fri

theorem pat_stickers_at_end_of_week :
  final_stickers initial_stickers 
                 monday_transaction 
                 tuesday_transaction 
                 wednesday_transaction 
                 thursday_trade_net_loss 
                 friday_find = 43 :=
by
  sorry

end pat_stickers_at_end_of_week_l99_99921


namespace mr_hernandez_tax_l99_99460

theorem mr_hernandez_tax :
  let taxable_income := 42500
  let resident_months := 9
  let standard_deduction := if resident_months > 6 then 5000 else 0
  let adjusted_income := taxable_income - standard_deduction
  let tax_bracket_1 := min adjusted_income 10000 * 0.01
  let tax_bracket_2 := min (max (adjusted_income - 10000) 0) 20000 * 0.03
  let tax_bracket_3 := min (max (adjusted_income - 30000) 0) 30000 * 0.05
  let total_tax_before_credit := tax_bracket_1 + tax_bracket_2 + tax_bracket_3
  let tax_credit := if resident_months < 10 then 500 else 0
  total_tax_before_credit - tax_credit = 575 := 
by
  sorry
  
end mr_hernandez_tax_l99_99460


namespace ratio_of_x_intercepts_l99_99333

theorem ratio_of_x_intercepts (b : ℝ) (hb: b ≠ 0) (u v: ℝ) (h₁: 8 * u + b = 0) (h₂: 4 * v + b = 0) : 
  u / v = 1 / 2 :=
by
  sorry

end ratio_of_x_intercepts_l99_99333


namespace standard_deviation_bound_l99_99489

theorem standard_deviation_bound (mu sigma : ℝ) (h_mu : mu = 51) (h_ineq : mu - 3 * sigma > 44) : sigma < 7 / 3 :=
by
  sorry

end standard_deviation_bound_l99_99489


namespace find_p_minus_q_l99_99068

theorem find_p_minus_q (x y p q : ℤ) (hx : x ≠ 0) (hy : y ≠ 0) (hp : p ≠ 0) (hq : q ≠ 0)
  (h1 : 3 / (x * p) = 8) (h2 : 5 / (y * q) = 18)
  (hminX : ∀ x', x' ≠ 0 → 3 / (x' * 3) ≠ 1 / 8)
  (hminY : ∀ y', y' ≠ 0 → 5 / (y' * 5) ≠ 1 / 18) :
  p - q = 0 :=
sorry

end find_p_minus_q_l99_99068


namespace largest_angle_of_consecutive_integers_hexagon_l99_99474

theorem largest_angle_of_consecutive_integers_hexagon (a b c d e f : ℝ) 
  (h1 : a < b) (h2 : b < c) (h3 : c < d) (h4 : d < e) (h5 : e < f) 
  (h6 : a + b + c + d + e + f = 720) : 
  ∃ x, f = x + 2 ∧ (x + 2 = 122.5) :=
  sorry

end largest_angle_of_consecutive_integers_hexagon_l99_99474


namespace total_triangles_in_grid_l99_99376

-- Conditions
def bottom_row_triangles : Nat := 3
def next_row_triangles : Nat := 2
def top_row_triangles : Nat := 1
def additional_triangle : Nat := 1

def small_triangles := bottom_row_triangles + next_row_triangles + top_row_triangles + additional_triangle

-- Combining the triangles into larger triangles
def larger_triangles := 1 -- Formed by combining 4 small triangles
def largest_triangle := 1 -- Formed by combining all 7 small triangles

-- Math proof problem
theorem total_triangles_in_grid : small_triangles + larger_triangles + largest_triangle = 9 :=
by
  sorry

end total_triangles_in_grid_l99_99376


namespace proof_x_eq_y_l99_99061

variable (x y z : ℝ)

theorem proof_x_eq_y (h1 : x = 6 - y) (h2 : z^2 = x * y - 9) : x = y := 
  sorry

end proof_x_eq_y_l99_99061


namespace sin_alpha_terminal_point_l99_99254

theorem sin_alpha_terminal_point :
  let alpha := (2 * Real.cos (120 * (π / 180)), Real.sqrt 2 * Real.sin (225 * (π / 180)))
  α = -π / 4 →
  α.sin = - Real.sqrt 2 / 2
:=
by
  intro α_definition
  sorry

end sin_alpha_terminal_point_l99_99254


namespace butterfly_cocoon_time_l99_99443

theorem butterfly_cocoon_time :
  ∀ (L C : ℕ), L = 3 * C ∧ L + C = 120 → C = 30 := by
  intros L C h
  cases' h with h1 h2
  have h3 : 3 * C + C = 120 := by rw [h1, add_comm] at h2
  have h4 : 4 * C = 120 := by rw add_mul 3 1 C at h3
  rw mul_comm at h4
  exact Nat.eq_of_mul_eq_mul_left (ne_of_gt (show 0 < 4 from by norm_num)) h4

end butterfly_cocoon_time_l99_99443


namespace inequality_satisfied_equality_condition_l99_99825

theorem inequality_satisfied (x y : ℝ) : x^2 + y^2 + 1 ≥ 2 * (x * y - x + y) :=
sorry

theorem equality_condition (x y : ℝ) : (x^2 + y^2 + 1 = 2 * (x * y - x + y)) ↔ (x = y - 1) :=
sorry

end inequality_satisfied_equality_condition_l99_99825


namespace ratio_blue_to_total_l99_99515

theorem ratio_blue_to_total (total_marbles red_marbles green_marbles yellow_marbles blue_marbles : ℕ)
    (h_total : total_marbles = 164)
    (h_red : red_marbles = total_marbles / 4)
    (h_green : green_marbles = 27)
    (h_yellow : yellow_marbles = 14)
    (h_blue : blue_marbles = total_marbles - (red_marbles + green_marbles + yellow_marbles)) :
  blue_marbles / total_marbles = 1 / 2 :=
by
  sorry

end ratio_blue_to_total_l99_99515


namespace vec_c_is_linear_comb_of_a_b_l99_99732

structure Vec2 :=
  (x : ℝ)
  (y : ℝ)

def a := Vec2.mk 1 2
def b := Vec2.mk (-2) 3
def c := Vec2.mk 4 1

theorem vec_c_is_linear_comb_of_a_b : c = Vec2.mk (2 * a.x - b.x) (2 * a.y - b.y) :=
  by
    sorry

end vec_c_is_linear_comb_of_a_b_l99_99732


namespace stock_price_after_two_years_l99_99855

def initial_price : ℝ := 120

def first_year_increase (p : ℝ) : ℝ := p * 2

def second_year_decrease (p : ℝ) : ℝ := p * 0.30

def final_price (initial : ℝ) : ℝ :=
  let after_first_year := first_year_increase initial
  after_first_year - second_year_decrease after_first_year

theorem stock_price_after_two_years : final_price initial_price = 168 :=
by
  sorry

end stock_price_after_two_years_l99_99855


namespace remainder_5_pow_5_pow_5_pow_5_mod_500_l99_99046

-- Conditions
def λ (n : ℕ) : ℕ := n.gcd20p1.factorial5div
def M : ℕ := 5 ^ (5 ^ 5)

-- Theorem: Prove the remainder
theorem remainder_5_pow_5_pow_5_pow_5_mod_500 :
  M % 500 = 125 :=
by sorry

end remainder_5_pow_5_pow_5_pow_5_mod_500_l99_99046


namespace ordered_systems_of_pos_rationals_l99_99673

def is_integer (q : ℚ) : Prop := ∃ n : ℤ, q = n

theorem ordered_systems_of_pos_rationals (x y z : ℚ)
  (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (hxy : is_integer (x + 1 / y))
  (hyz : is_integer (y + 1 / z))
  (hzx : is_integer (z + 1 / x)) :
  (x = 1 ∧ y = 1 ∧ z = 1) ∨
  (x = 1 ∧ y = 1/2 ∧ z = 2) ∨
  (x = 1/2 ∧ y = 2 ∧ z = 1) ∨
  (x = 2 ∧ y = 1 ∧ z = 1/2) ∨
  (x = 1/2 ∧ y = 2/3 ∧ z = 3) ∨
  (x = 3 ∧ y = 1/2 ∧ z = 2/3) ∨
  (x = 2/3 ∧ y = 3 ∧ z = 1/2) ∨
  (x = 1/3 ∧ y = 3/2 ∧ z = 2) ∨
  (x = 2 ∧ y = 1/3 ∧ z = 3/2) ∨
  (x = 3/2 ∧ y = 2 ∧ z = 1/3) :=
sorry

end ordered_systems_of_pos_rationals_l99_99673


namespace distance_covered_by_train_l99_99208

-- Define the average speed and the total duration of the journey
def speed : ℝ := 10
def time : ℝ := 8

-- Use these definitions to state and prove the distance covered by the train
theorem distance_covered_by_train : speed * time = 80 := by
  sorry

end distance_covered_by_train_l99_99208


namespace geometric_sequence_common_ratio_l99_99470

theorem geometric_sequence_common_ratio (a : ℕ → ℝ) (a_mono : ∀ n, a n < a (n+1))
    (a2a5_eq_6 : a 2 * a 5 = 6)
    (a3a4_eq_5 : a 3 + a 4 = 5) 
    (q : ℝ) (hq : ∀ n, a n = a 1 * q ^ (n - 1)) :
    q = 3 / 2 :=
by
    sorry

end geometric_sequence_common_ratio_l99_99470


namespace subcommittee_count_l99_99488

theorem subcommittee_count :
  let total_members := 12
  let teachers := 5
  let total_subcommittees := (Nat.choose total_members 4)
  let subcommittees_with_zero_teachers := (Nat.choose 7 4)
  let subcommittees_with_one_teacher := (Nat.choose teachers 1) * (Nat.choose 7 3)
  let subcommittees_with_fewer_than_two_teachers := subcommittees_with_zero_teachers + subcommittees_with_one_teacher
  let subcommittees_with_at_least_two_teachers := total_subcommittees - subcommittees_with_fewer_than_two_teachers
  subcommittees_with_at_least_two_teachers = 285 := by
  sorry

end subcommittee_count_l99_99488


namespace fair_contest_perfect_square_l99_99022

theorem fair_contest_perfect_square (n : ℕ) (h: 2 * n > 0) :
  ∃ k : ℕ, 
    let f : ℕ → ℕ := λ n, ((Nat.doubleFactorial (2 * n - 1)) ^ 2)
    in f n = k * k :=
sorry

end fair_contest_perfect_square_l99_99022


namespace exists_triangle_sides_l99_99563

theorem exists_triangle_sides (a b c : ℝ) (hpos : 0 < a ∧ 0 < b ∧ 0 < c)
  (h1 : a * b * c ≤ 1 / 4)
  (h2 : 1 / (a^2) + 1 / (b^2) + 1 / (c^2) < 9) : 
  a + b > c ∧ b + c > a ∧ c + a > b := 
by
  sorry

end exists_triangle_sides_l99_99563


namespace parallel_vectors_eq_l99_99412

theorem parallel_vectors_eq (x : ℝ) :
  let a := (x, 1)
  let b := (2, 4)
  (a.1 / b.1 = a.2 / b.2) → x = 1 / 2 :=
by
  intros h
  sorry

end parallel_vectors_eq_l99_99412


namespace no_obtuse_triangle_l99_99713

-- Conditions
def points_on_circle_uniformly_at_random (n : ℕ) : Prop :=
  ∀ i < n, ∀ j < n, i ≠ j → ∃ θ_ij : ℝ, 0 ≤ θ_ij ∧ θ_ij ≤ π

-- Theorem statement
theorem no_obtuse_triangle (hn : points_on_circle_uniformly_at_random 4) :
  let p := \left(\frac{1}{2}\right)^6 in
  p = \frac{1}{64} :=
sorry

end no_obtuse_triangle_l99_99713


namespace giftWrapperPerDay_l99_99389

variable (giftWrapperPerBox : ℕ)
variable (boxesPer3Days : ℕ)

def giftWrapperUsedIn3Days := giftWrapperPerBox * boxesPer3Days

theorem giftWrapperPerDay (h_giftWrapperPerBox : giftWrapperPerBox = 18)
  (h_boxesPer3Days : boxesPer3Days = 15) : giftWrapperUsedIn3Days giftWrapperPerBox boxesPer3Days / 3 = 90 :=
by
  sorry

end giftWrapperPerDay_l99_99389


namespace min_value_expression_l99_99795

theorem min_value_expression (a b c : ℝ) 
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_sum : a + b + c = 3) : 
  let A := (a^3 + b^3) / (8 * a * b + 9 - c^2) + 
           (b^3 + c^3) / (8 * b * c + 9 - a^2) + 
           (c^3 + a^3) / (8 * c * a + 9 - b^2) 
  in A = 3 / 8 :=
sorry

end min_value_expression_l99_99795


namespace tangent_points_l99_99289

theorem tangent_points (x y : ℝ) (h : y = x^3 - 3 * x) (slope_zero : 3 * x^2 - 3 = 0) :
  (x = -1 ∧ y = 2) ∨ (x = 1 ∧ y = -2) :=
sorry

end tangent_points_l99_99289


namespace c_work_rate_l99_99188

theorem c_work_rate (x : ℝ) : 
  (1 / 7 + 1 / 14 + 1 / x = 1 / 4) → x = 28 :=
by
  sorry

end c_work_rate_l99_99188


namespace moles_of_nacl_formed_l99_99679

noncomputable def reaction (nh4cl: ℕ) (naoh: ℕ) : ℕ :=
  if nh4cl = naoh then nh4cl else min nh4cl naoh

theorem moles_of_nacl_formed (nh4cl: ℕ) (naoh: ℕ) (h_nh4cl: nh4cl = 2) (h_naoh: naoh = 2) :
  reaction nh4cl naoh = 2 :=
by
  rw [h_nh4cl, h_naoh]
  sorry

end moles_of_nacl_formed_l99_99679


namespace number_of_boys_is_10_l99_99849

-- Definitions based on given conditions
def num_children := 20
def has_blue_neighbor_clockwise (i : ℕ) : Prop := true -- Dummy predicate representing condition
def has_red_neighbor_counterclockwise (i : ℕ) : Prop := true -- Dummy predicate representing condition

axiom boys_and_girls_exist : ∃ b g : ℤ, b + g = num_children ∧ b > 0 ∧ g > 0

-- Theorem based on the problem statement
theorem number_of_boys_is_10 (b g : ℤ) 
  (total_children: b + g = num_children)
  (boys_exist: b > 0)
  (girls_exist: g > 0)
  (each_boy_has_blue_neighbor: ∀ i, has_blue_neighbor_clockwise i → true)
  (each_girl_has_red_neighbor: ∀ i, has_red_neighbor_counterclockwise i → true): 
  b = 10 :=
by
  sorry

end number_of_boys_is_10_l99_99849


namespace no_7_edges_edges_greater_than_5_l99_99957

-- Define the concept of a convex polyhedron in terms of its edges and faces.
structure ConvexPolyhedron where
  V : ℕ    -- Number of vertices
  E : ℕ    -- Number of edges
  F : ℕ    -- Number of faces
  Euler : V - E + F = 2   -- Euler's characteristic

-- Define properties of convex polyhedron

-- Part (a) statement: A convex polyhedron cannot have exactly 7 edges.
theorem no_7_edges (P : ConvexPolyhedron) : P.E ≠ 7 :=
sorry

-- Part (b) statement: A convex polyhedron can have any number of edges greater than 5 and different from 7.
theorem edges_greater_than_5 (n : ℕ) (h : n > 5) (h2 : n ≠ 7) : ∃ P : ConvexPolyhedron, P.E = n :=
sorry

end no_7_edges_edges_greater_than_5_l99_99957


namespace probability_both_good_probability_both_defective_probability_exact_one_good_l99_99099

noncomputable def machine_a_quality := 0.90
noncomputable def machine_b_quality := 0.80

axiom independence (A B : Prop) : P(A && B) = P(A) * P(B)
axiom complement (p : ℝ) : P(¬A) = 1 - P(A)

def good_quality_A := P (select good part from machine A) = machine_a_quality
def good_quality_B := P (select good part from machine B) = machine_b_quality

theorem probability_both_good :
  P(selecting a good part from machine A && selecting a good part from machine B) = 0.72 :=
sorry

theorem probability_both_defective :
  P(~selecting a good part from machine A && ~selecting a good part from machine B) = 0.02 :=
sorry

theorem probability_exact_one_good :
  P(selecting a good part from machine A && ~selecting a good part from machine B || ~selecting a good part from machine A && selecting a good part from machine B) = 0.26 :=
sorry

end probability_both_good_probability_both_defective_probability_exact_one_good_l99_99099


namespace turnip_bag_weights_l99_99991

theorem turnip_bag_weights :
  ∃ (T : ℕ), (T = 13 ∨ T = 16) ∧ 
  (∀ T', T' ≠ T → (
    (2 * total_weight_of_has_weight_1.not_turnip T' O + O = 106 - T') → 
    let turnip_condition := 106 - T in
    turnip_condition % 3 = 0)) ∧
  ∀ (bag_weights : List ℕ) (other : bag_weights = [13, 15, 16, 17, 21, 24] ∧ 
                                          List.length bag_weights = 6),
  True := by sorry

end turnip_bag_weights_l99_99991


namespace divisible_by_11_and_smallest_n_implies_77_l99_99834

theorem divisible_by_11_and_smallest_n_implies_77 (n : ℕ) (h₁ : n = 7) : ∃ m : ℕ, m = 11 * n := 
sorry

end divisible_by_11_and_smallest_n_implies_77_l99_99834


namespace necessary_but_not_sufficient_l99_99431

def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop := 
  ∀ n : ℕ, a n = a 0 * q^n

def is_increasing_sequence (s : ℕ → ℝ) : Prop := 
  ∀ n : ℕ, s n < s (n + 1)

def sum_first_n_terms (a : ℕ → ℝ) (q : ℝ) : ℕ → ℝ 
| 0 => a 0
| (n+1) => (sum_first_n_terms a q n) + (a 0 * q ^ n)

theorem necessary_but_not_sufficient (a : ℕ → ℝ) (q : ℝ) (h_geometric: is_geometric_sequence a q) :
  (q > 0) ∧ is_increasing_sequence (sum_first_n_terms a q) ↔ (q > 0)
:= sorry

end necessary_but_not_sufficient_l99_99431


namespace turnip_bag_weights_l99_99990

theorem turnip_bag_weights :
  ∃ (T : ℕ), (T = 13 ∨ T = 16) ∧ 
  (∀ T', T' ≠ T → (
    (2 * total_weight_of_has_weight_1.not_turnip T' O + O = 106 - T') → 
    let turnip_condition := 106 - T in
    turnip_condition % 3 = 0)) ∧
  ∀ (bag_weights : List ℕ) (other : bag_weights = [13, 15, 16, 17, 21, 24] ∧ 
                                          List.length bag_weights = 6),
  True := by sorry

end turnip_bag_weights_l99_99990


namespace percentage_of_men_l99_99257

theorem percentage_of_men (M W : ℝ) (h1 : M + W = 1) (h2 : 0.60 * M + 0.2364 * W = 0.40) : M = 0.45 :=
by
  sorry

end percentage_of_men_l99_99257


namespace trajectory_equation_l99_99125

theorem trajectory_equation (x y : ℝ) : x^2 + y^2 = 2 * |x| + 2 * |y| → x^2 + y^2 = 2 * |x| + 2 * |y| :=
by
  sorry

end trajectory_equation_l99_99125


namespace max_profit_achieved_at_180_l99_99127

-- Definitions:
def cost (x : ℝ) : ℝ := 0.1 * x^2 - 11 * x + 3000  -- Condition 1
def selling_price_per_unit : ℝ := 25  -- Condition 2

-- Statement to prove that the maximum profit is achieved at x = 180
theorem max_profit_achieved_at_180 :
  ∃ (S : ℝ), ∀ (x : ℝ),
    S = -0.1 * (x - 180)^2 + 240 → S = 25 * 180 - cost 180 :=
by
  sorry

end max_profit_achieved_at_180_l99_99127


namespace ratio_of_x_intercepts_l99_99331

theorem ratio_of_x_intercepts (b : ℝ) (hb : b ≠ 0) (u v: ℝ)
  (h1: 0 = 8 * u + b) (h2: 0 = 4 * v + b) : u / v = 1 / 2 :=
by sorry

end ratio_of_x_intercepts_l99_99331


namespace factor_value_l99_99965

theorem factor_value 
  (m : ℝ) 
  (h : ∀ x : ℝ, x + 5 = 0 → (x^2 - m * x - 40) = 0) : 
  m = 3 := 
sorry

end factor_value_l99_99965


namespace caitlin_bracelets_l99_99535

/-- 
Caitlin makes bracelets to sell at the farmer’s market every weekend. 
Each bracelet takes twice as many small beads as it does large beads. 
If each bracelet uses 12 large beads, and Caitlin has 528 beads with equal amounts of large and small beads, 
prove that Caitlin can make 11 bracelets for this weekend.
-/
theorem caitlin_bracelets (total_beads large_beads_per_bracelet small_beads_per_bracelet total_large_beads total_small_beads bracelets : ℕ)
  (h1 : total_beads = 528)
  (h2 : total_beads = total_large_beads + total_small_beads)
  (h3 : total_large_beads = total_small_beads)
  (h4 : large_beads_per_bracelet = 12)
  (h5 : small_beads_per_bracelet = 2 * large_beads_per_bracelet)
  (h6 : bracelets = total_small_beads / small_beads_per_bracelet) : 
  bracelets = 11 := 
by {
  sorry
}

end caitlin_bracelets_l99_99535


namespace remainder_of_exponentiation_is_correct_l99_99047

-- Define the given conditions
def modulus := 500
def exponent := 5 ^ (5 ^ 5)
def carmichael_500 := 100
def carmichael_100 := 20

-- Prove the main theorem
theorem remainder_of_exponentiation_is_correct :
  (5 ^ exponent) % modulus = 125 := 
by
  -- Skipping the proof
  sorry

end remainder_of_exponentiation_is_correct_l99_99047


namespace arithmetic_progression_contains_sixth_power_l99_99889

theorem arithmetic_progression_contains_sixth_power
  (a h : ℕ) (a_pos : 0 < a) (h_pos : 0 < h)
  (sq : ∃ n : ℕ, a + n * h = k^2)
  (cube : ∃ m : ℕ, a + m * h = l^3) :
  ∃ p : ℕ, ∃ q : ℕ, a + q * h = p^6 := sorry

end arithmetic_progression_contains_sixth_power_l99_99889


namespace line_tangent_parabola_unique_d_l99_99300

theorem line_tangent_parabola_unique_d :
  (∃ d : ℝ, ∀ x : ℝ, y = 3 * x + d -> y^2 = 12 * x) -> d = 1 := by
sorry

end line_tangent_parabola_unique_d_l99_99300


namespace option_C_incorrect_l99_99382

variable (a b : ℝ)

theorem option_C_incorrect : ((-a^3)^2 * (-b^2)^3) ≠ (a^6 * b^6) :=
by {
  sorry
}

end option_C_incorrect_l99_99382


namespace Larry_spends_108_minutes_l99_99449

-- Define conditions
def half_hour_twice_daily := 30 * 2
def fifth_of_an_hour_daily := 60 / 5
def quarter_hour_twice_daily := 15 * 2
def tenth_of_an_hour_daily := 60 / 10

-- Define total times spent on each pet
def total_time_dog := half_hour_twice_daily + fifth_of_an_hour_daily
def total_time_cat := quarter_hour_twice_daily + tenth_of_an_hour_daily

-- Define the total time spent on pets
def total_time_pets := total_time_dog + total_time_cat

-- Lean theorem statement
theorem Larry_spends_108_minutes : total_time_pets = 108 := 
  by 
    sorry

end Larry_spends_108_minutes_l99_99449


namespace min_correct_answers_l99_99259

/-- 
Given:
1. There are 25 questions in the preliminary round.
2. Scoring rules: 
   - 4 points for each correct answer,
   - -1 point for each incorrect or unanswered question.
3. A score of at least 60 points is required to advance to the next round.

Prove that the minimum number of correct answers needed to advance is 17.
-/
theorem min_correct_answers (x : ℕ) (h1 : 0 ≤ x) (h2 : x ≤ 25) (h3 : 4 * x - (25 - x) ≥ 60) : x ≥ 17 :=
sorry

end min_correct_answers_l99_99259


namespace num_real_solutions_abs_eq_l99_99880

theorem num_real_solutions_abs_eq :
  (∃ x y : ℝ, x ≠ y ∧ |x-1| = |x-2| + |x-3| + |x-4| 
    ∧ |y-1| = |y-2| + |y-3| + |y-4| 
    ∧ ∀ z : ℝ, |z-1| = |z-2| + |z-3| + |z-4| → (z = x ∨ z = y)) := sorry

end num_real_solutions_abs_eq_l99_99880


namespace ratio_out_of_state_to_in_state_l99_99025

/-
Given:
- total job applications Carly sent is 600
- job applications sent to companies in her state is 200

Prove:
- The ratio of job applications sent to companies in other states to the number sent to companies in her state is 2:1.
-/

def total_applications : ℕ := 600
def in_state_applications : ℕ := 200
def out_of_state_applications : ℕ := total_applications - in_state_applications

theorem ratio_out_of_state_to_in_state :
  (out_of_state_applications / in_state_applications) = 2 :=
by
  sorry

end ratio_out_of_state_to_in_state_l99_99025


namespace jellybeans_in_new_bag_l99_99351

theorem jellybeans_in_new_bag (average_per_bag : ℕ) (num_bags : ℕ) (additional_avg_increase : ℕ) (total_jellybeans_old : ℕ) (total_jellybeans_new : ℕ) (num_bags_new : ℕ) (new_bag_jellybeans : ℕ) : 
  average_per_bag = 117 → 
  num_bags = 34 → 
  additional_avg_increase = 7 → 
  total_jellybeans_old = num_bags * average_per_bag → 
  total_jellybeans_new = (num_bags + 1) * (average_per_bag + additional_avg_increase) → 
  new_bag_jellybeans = total_jellybeans_new - total_jellybeans_old → 
  new_bag_jellybeans = 362 := 
by 
  intros 
  sorry

end jellybeans_in_new_bag_l99_99351


namespace rectangle_perimeter_l99_99518

variable (a b : ℝ)
variable (h1 : a * b = 24)
variable (h2 : a^2 + b^2 = 121)

theorem rectangle_perimeter : 2 * (a + b) = 26 := 
by
  sorry

end rectangle_perimeter_l99_99518


namespace cube_partition_exists_l99_99104

theorem cube_partition_exists : ∃ (n_0 : ℕ), (0 < n_0) ∧ (∀ (n : ℕ), n ≥ n_0 → ∃ k : ℕ, n = k) := sorry

end cube_partition_exists_l99_99104


namespace students_catching_up_on_homework_l99_99751

-- Definitions for the given conditions
def total_students := 120
def silent_reading_students := (2/5 : ℚ) * total_students
def board_games_students := (3/10 : ℚ) * total_students
def group_discussions_students := (1/8 : ℚ) * total_students
def other_activities_students := silent_reading_students + board_games_students + group_discussions_students
def catching_up_homework_students := total_students - other_activities_students

-- Statement of the proof problem
theorem students_catching_up_on_homework : catching_up_homework_students = 21 := by
  sorry

end students_catching_up_on_homework_l99_99751


namespace speed_with_stream_l99_99517

noncomputable def man_speed_still_water : ℝ := 5
noncomputable def speed_against_stream : ℝ := 4

theorem speed_with_stream :
  ∃ V_s, man_speed_still_water + V_s = 6 :=
by
  use man_speed_still_water - speed_against_stream
  sorry

end speed_with_stream_l99_99517


namespace line_passes_through_fixed_point_l99_99122

-- Define the condition that represents the family of lines
def family_of_lines (k : ℝ) (x y : ℝ) : Prop := k * x + y + 2 * k + 1 = 0

-- Formulate the theorem stating that (-2, -1) always lies on the line
theorem line_passes_through_fixed_point (k : ℝ) : family_of_lines k (-2) (-1) :=
by
  -- Proof skipped with sorry.
  sorry

end line_passes_through_fixed_point_l99_99122


namespace smallest_positive_four_digit_multiple_of_18_l99_99692

-- Define the predicates for conditions
def four_digit_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

def multiple_of_18 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 18 * k

-- Define the main theorem
theorem smallest_positive_four_digit_multiple_of_18 : 
  ∃ n : ℕ, four_digit_number n ∧ multiple_of_18 n ∧ ∀ m : ℕ, four_digit_number m ∧ multiple_of_18 m → n ≤ m :=
begin
  use 1008,
  split,
  { -- proof that 1008 is a four-digit number
    split,
    { linarith, },
    { linarith, }
  },

  split,
  { -- proof that 1008 is a multiple of 18
    use 56,
    norm_num,
  },

  { -- proof that 1008 is the smallest such number
    intros m h1 h2,
    have h3 := Nat.le_of_lt,
    sorry, -- Detailed proof would go here
  }
end

end smallest_positive_four_digit_multiple_of_18_l99_99692


namespace length_of_train_is_correct_l99_99348

noncomputable def train_length (speed_km_hr : ℝ) (time_sec : ℝ) : ℝ :=
  let speed_m_s := (speed_km_hr * 1000) / 3600
  speed_m_s * time_sec

theorem length_of_train_is_correct (speed_km_hr : ℝ) (time_sec : ℝ) (expected_length : ℝ) :
  speed_km_hr = 60 → time_sec = 21 → expected_length = 350.07 →
  train_length speed_km_hr time_sec = expected_length :=
by
  intros h1 h2 h3
  simp [h1, h2, train_length]
  sorry

end length_of_train_is_correct_l99_99348


namespace function_increasing_interval_l99_99672

noncomputable def f (x : ℝ) : ℝ := Real.exp x - x

theorem function_increasing_interval :
  ∀ x : ℝ, x > 0 → deriv f x > 0 := 
sorry

end function_increasing_interval_l99_99672


namespace solve_for_x_l99_99076

theorem solve_for_x (x : ℝ) (h : 8 * (2 + 1 / x) = 18) : x = 4 := by
  sorry

end solve_for_x_l99_99076


namespace large_marshmallows_are_eight_l99_99103

-- Definition for the total number of marshmallows
def total_marshmallows : ℕ := 18

-- Definition for the number of mini marshmallows
def mini_marshmallows : ℕ := 10

-- Definition for the number of large marshmallows
def large_marshmallows : ℕ := total_marshmallows - mini_marshmallows

-- Theorem stating that the number of large marshmallows is 8
theorem large_marshmallows_are_eight : large_marshmallows = 8 := by
  sorry

end large_marshmallows_are_eight_l99_99103


namespace sequence_periodic_l99_99557

def sequence_condition (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 2) = |a (n + 1)| - a n

theorem sequence_periodic (a : ℕ → ℝ) (m_0 : ℕ) (h : sequence_condition a) :
  ∀ m ≥ m_0, a (m + 9) = a m := 
sorry

end sequence_periodic_l99_99557


namespace probability_diff_topic_l99_99520

theorem probability_diff_topic (n : ℕ) (m : ℕ) : 
  n = 6 → m = 5 → 
  (n * m : ℚ) / (6 * 6 : ℚ) = 5 / 6 := 
by
  intros h1 h2
  rw [h1, h2]
  norm_num
  sorry

end probability_diff_topic_l99_99520


namespace system_equations_sum_14_l99_99926

theorem system_equations_sum_14 (a b c d : ℝ) 
  (h1 : a + c = 4) 
  (h2 : a * d + b * c = 5) 
  (h3 : a * c + b + d = 8) 
  (h4 : b * d = 1) :
  a + b + c + d = 7 ∨ a + b + c + d = 7 → (a + b + c + d) * 2 = 14 := 
by {
  sorry
}

end system_equations_sum_14_l99_99926


namespace probability_in_interval_l99_99564

noncomputable def normal := distribution.normal 3 sigma_sq

theorem probability_in_interval (X : ℝ → Prop) 
    (h1 : X follows normal 3 σ^2)
    (h2 : ∫ (x in IIQ X 5), pdf normal x = 0.8) : 
    ∫ (x in IIO 1 3), pdf normal x = 0.3 :=
sorry

end probability_in_interval_l99_99564


namespace percentage_difference_l99_99835

theorem percentage_difference (n z x y y_decreased : ℝ)
  (h1 : x = 8 * y)
  (h2 : y = 2 * |z - n|)
  (h3 : z = 1.1 * n)
  (h4 : y_decreased = 0.75 * y) :
  (x - y_decreased) / x * 100 = 90.625 := by
sorry

end percentage_difference_l99_99835


namespace range_of_x_l99_99089

theorem range_of_x (x : ℝ) : (∃ y : ℝ, y = 1 / (Real.sqrt (x - 2))) ↔ x > 2 :=
by
  sorry

end range_of_x_l99_99089


namespace exists_integer_cube_ends_with_2007_ones_l99_99387

theorem exists_integer_cube_ends_with_2007_ones :
  ∃ x : ℕ, x^3 % 10^2007 = 10^2007 - 1 :=
sorry

end exists_integer_cube_ends_with_2007_ones_l99_99387


namespace total_legs_among_tables_l99_99833

noncomputable def total_legs (total_tables four_legged_tables: ℕ) : ℕ :=
  let three_legged_tables := total_tables - four_legged_tables
  4 * four_legged_tables + 3 * three_legged_tables

theorem total_legs_among_tables : total_legs 36 16 = 124 := by
  sorry

end total_legs_among_tables_l99_99833


namespace total_matches_in_group_l99_99584

theorem total_matches_in_group (n : ℕ) (hn : n = 6) : 2 * (n * (n - 1) / 2) = 30 :=
by
  sorry

end total_matches_in_group_l99_99584


namespace triangle_geometric_sequence_sine_rule_l99_99427

noncomputable def sin60 : Real := Real.sqrt 3 / 2

theorem triangle_geometric_sequence_sine_rule 
  {a b c : Real} 
  {A B C : Real} 
  (h1 : a / b = b / c) 
  (h2 : A = 60 * Real.pi / 180) :
  b * Real.sin B / c = Real.sqrt 3 / 2 :=
by
  sorry

end triangle_geometric_sequence_sine_rule_l99_99427


namespace problem_inequality_1_problem_inequality_2_l99_99907

theorem problem_inequality_1 (x : ℝ) (α : ℝ) (hx : x > -1) (hα : 0 < α ∧ α < 1) : 
  (1 + x) ^ α ≤ 1 + α * x :=
sorry

theorem problem_inequality_2 (x : ℝ) (α : ℝ) (hx : x > -1) (hα : α < 0 ∨ α > 1) : 
  (1 + x) ^ α ≥ 1 + α * x :=
sorry

end problem_inequality_1_problem_inequality_2_l99_99907


namespace geo_seq_property_l99_99407

theorem geo_seq_property (a : ℕ → ℤ) (r : ℤ) (h_geom : ∀ n, a (n+1) = r * a n)
  (h4_8 : a 4 + a 8 = -3) : a 6 * (a 2 + 2 * a 6 + a 10) = 9 := 
sorry

end geo_seq_property_l99_99407


namespace maria_trip_distance_l99_99542

variable (D : ℕ) -- Defining the total distance D as a natural number

-- Defining the conditions given in the problem
def first_stop_distance := D / 2
def second_stop_distance := first_stop_distance - (1 / 3 * first_stop_distance)
def third_stop_distance := second_stop_distance - (2 / 5 * second_stop_distance)
def remaining_distance := 180

-- The statement to prove
theorem maria_trip_distance : third_stop_distance = remaining_distance → D = 900 :=
by
  sorry

end maria_trip_distance_l99_99542


namespace total_candles_in_small_boxes_l99_99230

-- Definitions of the conditions
def num_small_boxes_per_big_box := 4
def num_big_boxes := 50
def candles_per_small_box := 40

-- The total number of small boxes
def total_small_boxes : Nat := num_small_boxes_per_big_box * num_big_boxes

-- The statement to prove the total number of candles in all small boxes is 8000
theorem total_candles_in_small_boxes : candles_per_small_box * total_small_boxes = 8000 :=
by 
  sorry

end total_candles_in_small_boxes_l99_99230


namespace smallest_positive_integer_for_terminating_decimal_l99_99159

theorem smallest_positive_integer_for_terminating_decimal (n : ℕ) (h1 : n > 0) (h2 : (∀ m : ℕ, (m > n + 150) -> (m % (n + 150)) ∉ {2, 5})) :
  n = 50 :=
sorry

end smallest_positive_integer_for_terminating_decimal_l99_99159


namespace cost_of_each_fish_is_four_l99_99267

-- Definitions according to the conditions
def number_of_fish_given_to_dog := 40
def number_of_fish_given_to_cat := number_of_fish_given_to_dog / 2
def total_fish := number_of_fish_given_to_dog + number_of_fish_given_to_cat
def total_cost := 240
def cost_per_fish := total_cost / total_fish

-- The main statement / theorem that needs to be proved
theorem cost_of_each_fish_is_four :
  cost_per_fish = 4 :=
by
  sorry

end cost_of_each_fish_is_four_l99_99267


namespace sum_of_altitudes_of_triangle_l99_99120

theorem sum_of_altitudes_of_triangle (a b c : ℝ) (h_line : ∀ x y, 8 * x + 10 * y = 80 → x = 10 ∨ y = 8) :
  (8 + 10 + 40/Real.sqrt 41) = 18 + 40/Real.sqrt 41 :=
by
  sorry

end sum_of_altitudes_of_triangle_l99_99120


namespace amplitude_of_resultant_wave_l99_99128

noncomputable def y1 (t : ℝ) := 3 * Real.sqrt 2 * Real.sin (100 * Real.pi * t)
noncomputable def y2 (t : ℝ) := 3 * Real.sin (100 * Real.pi * t - Real.pi / 4)
noncomputable def y (t : ℝ) := y1 t + y2 t

theorem amplitude_of_resultant_wave :
  ∃ R : ℝ, R = 3 * Real.sqrt 5 ∧ ∀ t : ℝ, y t = R * Real.sin (100 * Real.pi * t - θ) :=
by
  let y_combined := y
  use 3 * Real.sqrt 5
  sorry

end amplitude_of_resultant_wave_l99_99128


namespace longest_boat_length_l99_99788

variable (saved money : ℕ) (license_fee docking_multiplier boat_cost : ℕ)

theorem longest_boat_length (h1 : saved = 20000) 
                           (h2 : license_fee = 500) 
                           (h3 : docking_multiplier = 3)
                           (h4 : boat_cost = 1500) : 
                           (saved - license_fee - docking_multiplier * license_fee) / boat_cost = 12 := 
by 
  sorry

end longest_boat_length_l99_99788


namespace lollipops_remainder_l99_99114

theorem lollipops_remainder :
  let total_lollipops := 8362
  let lollipops_per_package := 12
  total_lollipops % lollipops_per_package = 10 :=
by
  let total_lollipops := 8362
  let lollipops_per_package := 12
  sorry

end lollipops_remainder_l99_99114


namespace find_coordinates_of_P_l99_99413

-- Define the problem conditions
def P (m : ℤ) := (2 * m + 4, m - 1)
def A := (2, -4)
def line_l (y : ℤ) := y = -4
def P_on_line_l (m : ℤ) := line_l (m - 1)

theorem find_coordinates_of_P (m : ℤ) (h : P_on_line_l m) : P m = (-2, -4) := 
  by sorry

end find_coordinates_of_P_l99_99413


namespace sum_of_squares_and_product_pos_ints_l99_99490

variable (x y : ℕ)

theorem sum_of_squares_and_product_pos_ints :
  x^2 + y^2 = 289 ∧ x * y = 120 → x + y = 23 :=
by
  intro h
  sorry

end sum_of_squares_and_product_pos_ints_l99_99490


namespace club_planning_committee_l99_99514

theorem club_planning_committee : Nat.choose 20 3 = 1140 := 
by sorry

end club_planning_committee_l99_99514


namespace pairs_of_socks_calculation_l99_99446

variable (num_pairs_socks : ℤ)
variable (cost_per_pair : ℤ := 950) -- in cents
variable (cost_shoes : ℤ := 9200) -- in cents
variable (money_jack_has : ℤ := 4000) -- in cents
variable (money_needed : ℤ := 7100) -- in cents
variable (total_money_needed : ℤ := money_jack_has + money_needed)

theorem pairs_of_socks_calculation (x : ℤ) (h : cost_per_pair * x + cost_shoes = total_money_needed) : x = 2 :=
by
  sorry

end pairs_of_socks_calculation_l99_99446


namespace problem_part1_problem_part2_l99_99247

noncomputable def f (x : ℝ) (m : ℝ) := Real.sqrt (|x + 2| + |x - 4| - m)

theorem problem_part1 (m : ℝ) : 
  (∀ x : ℝ, |x + 2| + |x - 4| - m ≥ 0) ↔ m ≤ 6 := 
by
  sorry

theorem problem_part2 (a b : ℕ) (n : ℝ) (h1 : (0 < a) ∧ (0 < b)) (h2 : n = 6) 
  (h3 : (4 / (a + 5 * b)) + (1 / (3 * a + 2 * b)) = n) : 
  ∃ (value : ℝ), 4 * a + 7 * b = 3 / 2 := 
by
  sorry

end problem_part1_problem_part2_l99_99247


namespace probability_cube_vertices_in_plane_l99_99717

open Finset

noncomputable def choose (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem probability_cube_vertices_in_plane : 
  let total_ways := choose 8 4 in
  let favorable_ways := 12 in
  0 < total_ways →  -- Ensure total_ways is non-zero to avoid division by zero
  let P := (favorable_ways : ℝ) / (total_ways : ℝ) in
  P = 6 / 35 :=
by 
  sorry

end probability_cube_vertices_in_plane_l99_99717


namespace factorize_expression_l99_99544

theorem factorize_expression (x y : ℝ) : 
  6 * x^3 * y^2 + 3 * x^2 * y^2 = 3 * x^2 * y^2 * (2 * x + 1) := 
by 
  sorry

end factorize_expression_l99_99544


namespace evaluate_fraction_sum_l99_99872

variable (a b c : ℝ)

theorem evaluate_fraction_sum
  (h : (a / (30 - a)) + (b / (70 - b)) + (c / (80 - c)) = 9) :
  (6 / (30 - a)) + (14 / (70 - b)) + (16 / (80 - c)) = 2.4 :=
by
  sorry

end evaluate_fraction_sum_l99_99872


namespace negated_proposition_l99_99277

theorem negated_proposition : ∀ x : ℝ, x^2 + 2 * x + 2 ≥ 0 := by
  sorry

end negated_proposition_l99_99277


namespace even_product_implies_sum_of_squares_odd_product_implies_no_sum_of_squares_l99_99908

theorem even_product_implies_sum_of_squares (a b : ℕ) (h : ∃ (a b : ℕ), a * b % 2 = 0 → ∃ (c d : ℕ), a^2 + b^2 + c^2 = d^2) : 
  ∃ (c d : ℕ), a^2 + b^2 + c^2 = d^2 :=
sorry

theorem odd_product_implies_no_sum_of_squares (a b : ℕ) (h : ∃ (a b : ℕ), a * b % 2 ≠ 0 → ¬∃ (c d : ℕ), a^2 + b^2 + c^2 = d^2) : 
  ¬∃ (c d : ℕ), a^2 + b^2 + c^2 = d^2 :=
sorry

end even_product_implies_sum_of_squares_odd_product_implies_no_sum_of_squares_l99_99908


namespace archers_in_golden_armor_l99_99753

theorem archers_in_golden_armor (total_soldiers archers_swordsmen total_affirmations armor_affirmations archer_affirmations monday_affirmations : ℕ)
  (h1: total_soldiers = 55)
  (h2: armor_affirmations = 44)
  (h3: archer_affirmations = 33)
  (h4: monday_affirmations = 22)
  (h5: ∑ x in ({armor_affirmations, archer_affirmations, monday_affirmations} : finset ℕ), x = 99) 
  : ∃ (archers_in_golden : ℕ), archers_in_golden = 22 := by
  -- Definitions and theorems will go here
  sorry

end archers_in_golden_armor_l99_99753


namespace min_value_of_a_squared_plus_b_squared_l99_99728

-- Problem definition and condition
def is_on_circle (a b : ℝ) : Prop :=
  (a^2 + b^2 - 2*a + 4*b - 20) = 0

-- Theorem statement
theorem min_value_of_a_squared_plus_b_squared (a b : ℝ) (h : is_on_circle a b) :
  a^2 + b^2 = 30 - 10 * Real.sqrt 5 :=
sorry

end min_value_of_a_squared_plus_b_squared_l99_99728


namespace right_triangle_sides_l99_99126

theorem right_triangle_sides (x y z : ℕ) (h_sum : x + y + z = 156) (h_area : x * y = 2028) (h_pythagorean : z^2 = x^2 + y^2) :
  (x = 39 ∧ y = 52 ∧ z = 65) ∨ (x = 52 ∧ y = 39 ∧ z = 65) :=
by
  admit -- proof goes here

-- Additional details for importing required libraries and setting up the environment
-- are intentionally simplified as per instruction to cover a broader import.

end right_triangle_sides_l99_99126


namespace average_people_per_hour_l99_99768

-- Define the conditions
def people_moving : ℕ := 3000
def days : ℕ := 5
def hours_per_day : ℕ := 24
def total_hours : ℕ := days * hours_per_day

-- State the problem
theorem average_people_per_hour :
  people_moving / total_hours = 25 :=
by
  -- Proof goes here
  sorry

end average_people_per_hour_l99_99768


namespace fractions_sum_equals_one_l99_99450

variable {a b c x y z : ℝ}

variables (h1 : 17 * x + b * y + c * z = 0)
variables (h2 : a * x + 29 * y + c * z = 0)
variables (h3 : a * x + b * y + 53 * z = 0)
variables (ha : a ≠ 17)
variables (hx : x ≠ 0)

theorem fractions_sum_equals_one (a b c x y z : ℝ) 
  (h1 : 17 * x + b * y + c * z = 0)
  (h2 : a * x + 29 * y + c * z = 0)
  (h3 : a * x + b * y + 53 * z = 0)
  (ha : a ≠ 17)
  (hx : x ≠ 0) :
  (a / (a - 17)) + (b / (b - 29)) + (c / (c - 53)) = 1 := by 
  sorry

end fractions_sum_equals_one_l99_99450


namespace find_capacity_l99_99003

noncomputable def pool_capacity (V1 V2 q : ℝ) : Prop :=
  V1 = q / 120 ∧ V2 = V1 + 50 ∧ V1 + V2 = q / 48

theorem find_capacity (q : ℝ) : ∃ V1 V2, pool_capacity V1 V2 q → q = 12000 :=
by 
  sorry

end find_capacity_l99_99003


namespace interval_comparison_l99_99869

theorem interval_comparison (x : ℝ) :
  ((x - 1) * (x + 3) < 0) → ¬((x + 1) * (x - 3) < 0) ∧ ¬((x + 1) * (x - 3) < 0 → (x - 1) * (x + 3) < 0) :=
by
  sorry

end interval_comparison_l99_99869


namespace sufficiency_not_necessity_condition_l99_99964

theorem sufficiency_not_necessity_condition (a : ℝ) (h : a > 1) : (a^2 > 1) ∧ ¬(∀ x : ℝ, x^2 > 1 → x > 1) :=
by
  sorry

end sufficiency_not_necessity_condition_l99_99964


namespace line_tangent_to_parabola_proof_l99_99298

noncomputable def line_tangent_to_parabola (d : ℝ) := (∀ x y : ℝ, y = 3 * x + d → y^2 = 12 * x) → d = 1

theorem line_tangent_to_parabola_proof (d : ℝ) (h : ∀ x y : ℝ, y = 3 * x + d → y^2 = 12 * x) : d = 1 :=
sorry

end line_tangent_to_parabola_proof_l99_99298


namespace solve_system_l99_99419

variable {x y z : ℝ}

theorem solve_system :
  (y + z = 16 - 4 * x) →
  (x + z = -18 - 4 * y) →
  (x + y = 13 - 4 * z) →
  2 * x + 2 * y + 2 * z = 11 / 3 :=
by
  intros h1 h2 h3
  -- proof skips, to be completed
  sorry

end solve_system_l99_99419


namespace flowers_per_set_l99_99445

variable (totalFlowers : ℕ) (numSets : ℕ)

theorem flowers_per_set (h1 : totalFlowers = 270) (h2 : numSets = 3) : totalFlowers / numSets = 90 :=
by
  sorry

end flowers_per_set_l99_99445


namespace restaurant_dinners_sold_on_Monday_l99_99461

theorem restaurant_dinners_sold_on_Monday (M : ℕ) 
  (h1 : ∀ tues_dinners, tues_dinners = M + 40) 
  (h2 : ∀ wed_dinners, wed_dinners = (M + 40) / 2)
  (h3 : ∀ thurs_dinners, thurs_dinners = ((M + 40) / 2) + 3)
  (h4 : M + (M + 40) + ((M + 40) / 2) + (((M + 40) / 2) + 3) = 203) : 
  M = 40 := 
sorry

end restaurant_dinners_sold_on_Monday_l99_99461


namespace radius_of_base_circle_of_cone_l99_99977

theorem radius_of_base_circle_of_cone (θ : ℝ) (r_sector : ℝ) (L : ℝ) (C : ℝ) (r_base : ℝ) :
  θ = 120 ∧ r_sector = 6 ∧ L = (θ / 360) * 2 * Real.pi * r_sector ∧ C = L ∧ C = 2 * Real.pi * r_base → r_base = 2 := by
  sorry

end radius_of_base_circle_of_cone_l99_99977


namespace rectangle_extraction_l99_99060

theorem rectangle_extraction (m : ℤ) (h1 : m > 12) : 
  ∃ (x y : ℤ), x ≤ y ∧ x * y > m ∧ x * (y - 1) < m :=
by
  sorry

end rectangle_extraction_l99_99060


namespace find_eighth_number_l99_99620

theorem find_eighth_number (x : ℕ) (h1 : (1 + 2 + 4 + 5 + 6 + 9 + 9 + x + 12) / 9 = 7) : x = 27 :=
sorry

end find_eighth_number_l99_99620


namespace common_points_line_circle_l99_99568

theorem common_points_line_circle (a : ℝ) : 
  (∀ x y: ℝ, (x - 2*y + a = 0) → ((x - 2)^2 + y^2 = 1)) ↔ (-2 - Real.sqrt 5 ≤ a ∧ a ≤ -2 + Real.sqrt 5) :=
by sorry

end common_points_line_circle_l99_99568


namespace num_positive_integers_n_l99_99646

theorem num_positive_integers_n (n : ℕ) : 
  (∃ n, ( ∃ k : ℕ, n = 2015 * k^2 ∧ ∃ m, m^2 = 2015 * n) ∧ 
          (∃ k : ℕ, n = 2015 * k^2 ∧  ∃ l : ℕ, 2 * 2015 * k^2 = l * (1 + k^2)))
  →
  n = 5 := sorry

end num_positive_integers_n_l99_99646


namespace trigonometric_product_eq_l99_99506

open Real

theorem trigonometric_product_eq :
  3.420 * (sin (10 * pi / 180)) * (sin (20 * pi / 180)) * (sin (30 * pi / 180)) *
  (sin (40 * pi / 180)) * (sin (50 * pi / 180)) * (sin (60 * pi / 180)) *
  (sin (70 * pi / 180)) * (sin (80 * pi / 180)) = 3 / 256 := 
sorry

end trigonometric_product_eq_l99_99506


namespace simplify_expression_l99_99868

theorem simplify_expression :
  ((3 + 4 + 5 + 6) ^ 2 / 4) + ((3 * 6 + 9) ^ 2 / 3) = 324 := 
  sorry

end simplify_expression_l99_99868


namespace evaluate_101_times_101_l99_99858

theorem evaluate_101_times_101 : (101 * 101 = 10201) :=
by {
  sorry
}

end evaluate_101_times_101_l99_99858


namespace quadrilaterals_property_A_false_l99_99123

theorem quadrilaterals_property_A_false (Q A : Type → Prop) 
  (h : ¬ ∃ x, Q x ∧ A x) : ¬ ∀ x, Q x → A x :=
by
  sorry

end quadrilaterals_property_A_false_l99_99123


namespace wendy_total_sales_correct_l99_99338

noncomputable def wendy_total_sales : ℝ :=
  let morning_apples := 40 * 1.50
  let morning_oranges := 30 * 1
  let morning_bananas := 10 * 0.75
  let afternoon_apples := 50 * 1.35
  let afternoon_oranges := 40 * 0.90
  let afternoon_bananas := 20 * 0.675
  let unsold_bananas := 20 * 0.375
  let unsold_oranges := 10 * 0.50
  let total_morning := morning_apples + morning_oranges + morning_bananas
  let total_afternoon := afternoon_apples + afternoon_oranges + afternoon_bananas
  let total_day_sales := total_morning + total_afternoon
  let total_unsold_sales := unsold_bananas + unsold_oranges
  total_day_sales + total_unsold_sales

theorem wendy_total_sales_correct :
  wendy_total_sales = 227 := by
  unfold wendy_total_sales
  sorry

end wendy_total_sales_correct_l99_99338


namespace intersection_product_is_15_l99_99638

-- Define the first circle equation as a predicate
def first_circle (x y : ℝ) : Prop :=
  x^2 - 4 * x + y^2 - 6 * y + 12 = 0

-- Define the second circle equation as a predicate
def second_circle (x y : ℝ) : Prop :=
  x^2 - 10 * x + y^2 - 6 * y + 34 = 0

-- The Lean statement for the proof problem
theorem intersection_product_is_15 :
  ∃ x y : ℝ, first_circle x y ∧ second_circle x y ∧ (x * y = 15) :=
by
  sorry

end intersection_product_is_15_l99_99638


namespace var_X_is_86_over_225_l99_99606

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

end var_X_is_86_over_225_l99_99606


namespace proof_equivalent_triples_l99_99391

noncomputable def valid_triples := 
  { (a, b, c) : ℝ × ℝ × ℝ |
    a * b + b * c + c * a = 1 ∧
    a^2 * b + c = b^2 * c + a ∧
    a^2 * b + c = c^2 * a + b }

noncomputable def desired_solutions := 
  { (a, b, c) |
    (a = 0 ∧ b = 1 ∧ c = 1) ∨
    (a = 0 ∧ b = 1 ∧ c = -1) ∨
    (a = 0 ∧ b = -1 ∧ c = 1) ∨
    (a = 0 ∧ b = -1 ∧ c = -1) ∨

    (a = 1 ∧ b = 1 ∧ c = 0) ∨
    (a = 1 ∧ b = -1 ∧ c = 0) ∨
    (a = -1 ∧ b = 1 ∧ c = 0) ∨
    (a = -1 ∧ b = -1 ∧ c = 0) ∨

    (a = 1 ∧ b = 0 ∧ c = 1) ∨
    (a = 1 ∧ b = 0 ∧ c = -1) ∨
    (a = -1 ∧ b = 0 ∧ c = 1) ∨
    (a = -1 ∧ b = 0 ∧ c = -1) ∨

    ((a = (Real.sqrt 3) / 3 ∧ b = (Real.sqrt 3) / 3 ∧ 
      c = (Real.sqrt 3) / 3) ∨
     (a = -(Real.sqrt 3) / 3 ∧ b = -(Real.sqrt 3) / 3 ∧ 
      c = -(Real.sqrt 3) / 3)) }

theorem proof_equivalent_triples :
  valid_triples = desired_solutions :=
sorry

end proof_equivalent_triples_l99_99391


namespace mod_equiv_solution_l99_99418

theorem mod_equiv_solution (a b : ℤ) (n : ℤ) 
  (h₁ : a ≡ 22 [ZMOD 50])
  (h₂ : b ≡ 78 [ZMOD 50])
  (h₃ : 150 ≤ n ∧ n ≤ 201)
  (h₄ : n = 194) :
  a - b ≡ n [ZMOD 50] :=
by
  sorry

end mod_equiv_solution_l99_99418


namespace tina_assignment_time_l99_99130

theorem tina_assignment_time (total_time clean_time_per_key remaining_keys assignment_time : ℕ) 
  (h1 : total_time = 52) 
  (h2 : clean_time_per_key = 3) 
  (h3 : remaining_keys = 14) 
  (h4 : assignment_time = total_time - remaining_keys * clean_time_per_key) :
  assignment_time = 10 :=
by
  rw [h1, h2, h3] at h4
  assumption

end tina_assignment_time_l99_99130


namespace tickets_total_l99_99014

theorem tickets_total (x y : ℕ) 
  (h1 : 12 * x + 8 * y = 3320)
  (h2 : y = x + 190) : 
  x + y = 370 :=
by
  sorry

end tickets_total_l99_99014


namespace turnip_bag_weighs_l99_99985

theorem turnip_bag_weighs (bags : List ℕ) (T : ℕ)
  (h_weights : bags = [13, 15, 16, 17, 21, 24])
  (h_turnip : T ∈ bags)
  (h_carrot_onion_relation : ∃ O C: ℕ, C = 2 * O ∧ C + O = 106 - T) :
  T = 13 ∨ T = 16 := by
  sorry

end turnip_bag_weighs_l99_99985


namespace Ali_is_8_l99_99371

open Nat

-- Definitions of the variables based on the conditions
def YusafAge (UmarAge : ℕ) : ℕ := UmarAge / 2
def AliAge (YusafAge : ℕ) : ℕ := YusafAge + 3

-- The specific given conditions
def UmarAge : ℕ := 10
def Yusaf : ℕ := YusafAge UmarAge
def Ali : ℕ := AliAge Yusaf

-- The theorem to be proved
theorem Ali_is_8 : Ali = 8 :=
by
  sorry

end Ali_is_8_l99_99371


namespace tangent_line_at_1_2_l99_99291

noncomputable def f (x : ℝ) : ℝ := x^3 - x^2 + x + 1

def f' (x : ℝ) : ℝ := 3 * x^2 - 2 * x + 1

def tangent_eq (x y : ℝ) : Prop := y = 2 * x

theorem tangent_line_at_1_2 : tangent_eq 1 2 :=
by
  have f_1 := 1
  have f'_1 := 2
  sorry

end tangent_line_at_1_2_l99_99291


namespace total_legs_and_hands_on_ground_is_118_l99_99583

-- Definitions based on the conditions given
def total_dogs := 20
def dogs_on_two_legs := total_dogs / 2
def dogs_on_four_legs := total_dogs / 2

def total_cats := 10
def cats_on_two_legs := total_cats / 3
def cats_on_four_legs := total_cats - cats_on_two_legs

def total_horses := 5
def horses_on_two_legs := 2
def horses_on_four_legs := total_horses - horses_on_two_legs

def total_acrobats := 6
def acrobats_on_one_hand := 4
def acrobats_on_two_hands := 2

-- Functions to calculate the number of legs/paws/hands on the ground
def dogs_legs_on_ground := (dogs_on_two_legs * 2) + (dogs_on_four_legs * 4)
def cats_legs_on_ground := (cats_on_two_legs * 2) + (cats_on_four_legs * 4)
def horses_legs_on_ground := (horses_on_two_legs * 2) + (horses_on_four_legs * 4)
def acrobats_hands_on_ground := (acrobats_on_one_hand * 1) + (acrobats_on_two_hands * 2)

-- Total legs/paws/hands on the ground
def total_legs_on_ground := dogs_legs_on_ground + cats_legs_on_ground + horses_legs_on_ground + acrobats_hands_on_ground

-- The theorem to prove
theorem total_legs_and_hands_on_ground_is_118 : total_legs_on_ground = 118 :=
by sorry

end total_legs_and_hands_on_ground_is_118_l99_99583


namespace initial_geese_count_l99_99210

theorem initial_geese_count (G : ℕ) (h1 : G / 2 + 4 = 12) : G = 16 := by
  sorry

end initial_geese_count_l99_99210


namespace caitlin_bracelets_l99_99532

-- Define the conditions
def twice_as_many_small_beads (x y : Nat) : Prop :=
  y = 2 * x

def total_large_small_beads (total large small : Nat) : Prop :=
  total = large + small ∧ large = small

def bracelet_beads (large_beads_per_bracelet small_beads_per_bracelet large_per_bracelet : Nat) : Prop :=
  small_beads_per_bracelet = 2 * large_per_bracelet

def total_bracelets (total_large_beads large_per_bracelet bracelets : Nat) : Prop :=
  bracelets = total_large_beads / large_per_bracelet

-- The theorem to be proved
theorem caitlin_bracelets (total_beads large_per_bracelet small_per_bracelet : Nat) (bracelets : Nat) :
    total_beads = 528 ∧
    large_per_bracelet = 12 ∧
    twice_as_many_small_beads large_per_bracelet small_per_bracelet ∧
    total_large_small_beads total_beads 264 264 ∧
    bracelet_beads large_per_bracelet small_per_bracelet 12 ∧
    total_bracelets 264 12 bracelets
  → bracelets = 22 := by
  sorry

end caitlin_bracelets_l99_99532


namespace sufficient_condition_a_gt_1_l99_99962

variable (a : ℝ)

theorem sufficient_condition_a_gt_1 (h : a > 1) : a^2 > 1 :=
by sorry

end sufficient_condition_a_gt_1_l99_99962


namespace minimum_sum_am_gm_l99_99383

theorem minimum_sum_am_gm (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a / (3 * b)) + (b / (6 * c)) + (c / (9 * a)) ≥ (1 / 2) :=
sorry

end minimum_sum_am_gm_l99_99383


namespace math_problem_l99_99639

theorem math_problem : (4 + 6 + 7) * 2 - 2 + (3 / 3) = 33 := 
by
  sorry

end math_problem_l99_99639


namespace friends_belong_special_team_l99_99634

-- Define a type for students
universe u
variable {Student : Type u}

-- Assume a friendship relation among students
variable (friend : Student → Student → Prop)

-- Assume the conditions as given in the problem
variable (S : Student → Set (Set Student))
variable (students : Set Student)
variable (S_non_empty : ∀ v : Student, S v ≠ ∅)
variable (friendship_condition : 
  ∀ u v : Student, friend u v → 
    (∃ w : Student, S u ∩ S v ⊇ S w))
variable (special_team : ∀ (T : Set Student),
  (∃ v ∈ T, ∀ w : Student, w ∈ T → friend v w) ↔
  (∃ v ∈ T, ∀ w : Student, friend v w → w ∈ T))

-- Prove that any two friends belong to some special team
theorem friends_belong_special_team :
  ∀ u v : Student, friend u v → 
    (∃ T : Set Student, T ∈ S u ∩ S v ∧ 
      (∃ w ∈ T, ∀ x : Student, friend w x → x ∈ T)) :=
by
  sorry  -- Proof omitted


end friends_belong_special_team_l99_99634


namespace expression_evaluates_at_1_l99_99088

variable (x : ℚ)

def original_expr (x : ℚ) : ℚ := (x + 2) / (x - 3)

def substituted_expr (x : ℚ) : ℚ :=
  (original_expr (original_expr x) + 2) / (original_expr (original_expr x) - 3)

theorem expression_evaluates_at_1 :
  substituted_expr 1 = -1 / 9 :=
by
  sorry

end expression_evaluates_at_1_l99_99088


namespace find_r_l99_99420

-- Define the basic conditions based on the given problem.
def pr (r : ℕ) := 360 / 6
def p := pr 4 / 4
def cr (c r : ℕ) := 6 * c * r

-- Prove that r = 4 given the conditions.
theorem find_r (r : ℕ) : r = 4 :=
by
  sorry

end find_r_l99_99420


namespace reduced_price_per_dozen_bananas_l99_99822

noncomputable def original_price (P : ℝ) := P
noncomputable def reduced_price_one_banana (P : ℝ) := 0.60 * P
noncomputable def number_bananas_original (P : ℝ) := 40 / P
noncomputable def number_bananas_reduced (P : ℝ) := 40 / (0.60 * P)
noncomputable def difference_bananas (P : ℝ) := (number_bananas_reduced P) - (number_bananas_original P)

theorem reduced_price_per_dozen_bananas 
  (P : ℝ) 
  (h1 : difference_bananas P = 67) 
  (h2 : P = 16 / 40.2) :
  12 * reduced_price_one_banana P = 2.856 :=
sorry

end reduced_price_per_dozen_bananas_l99_99822


namespace total_spending_is_140_l99_99112

-- Define definitions for each day's spending based on the conditions.
def monday_spending : ℕ := 6
def tuesday_spending : ℕ := 2 * monday_spending
def wednesday_spending : ℕ := 2 * (monday_spending + tuesday_spending)
def thursday_spending : ℕ := (monday_spending + tuesday_spending + wednesday_spending) / 3
def friday_spending : ℕ := thursday_spending - 4
def saturday_spending : ℕ := friday_spending + (friday_spending / 2)
def sunday_spending : ℕ := tuesday_spending + saturday_spending

-- The total spending for the week.
def total_spending : ℕ := 
  monday_spending + 
  tuesday_spending + 
  wednesday_spending + 
  thursday_spending + 
  friday_spending + 
  saturday_spending + 
  sunday_spending

-- The theorem to prove that the total spending is $140.
theorem total_spending_is_140 : total_spending = 140 := 
  by {
    -- Due to the problem's requirement, we skip the proof steps.
    sorry
  }

end total_spending_is_140_l99_99112


namespace astroid_arc_length_l99_99035

theorem astroid_arc_length (a : ℝ) (h_a : a > 0) :
  ∃ l : ℝ, (l = 6 * a) ∧ 
  ((a = 1 → l = 6) ∧ (a = 2/3 → l = 4)) := 
by
  sorry

end astroid_arc_length_l99_99035


namespace sufficiency_not_necessity_condition_l99_99963

theorem sufficiency_not_necessity_condition (a : ℝ) (h : a > 1) : (a^2 > 1) ∧ ¬(∀ x : ℝ, x^2 > 1 → x > 1) :=
by
  sorry

end sufficiency_not_necessity_condition_l99_99963


namespace power_modulo_calculation_l99_99050

open Nat

theorem power_modulo_calculation :
  let λ500 := 100
  let λ100 := 20
  (5^5 : ℕ) ≡ 25 [MOD 100]
  (125^5 : ℕ) ≡ 125 [MOD 500]
  (5^{5^{5^5}} : ℕ) % 500 = 125 :=
by
  let λ500 := 100
  let λ100 := 20
  have h1 : (5^5 : ℕ) ≡ 25 [MOD 100] := by sorry
  have h2 : (125^5 : ℕ) ≡ 125 [MOD 500] := by sorry
  sorry

end power_modulo_calculation_l99_99050


namespace opposite_of_neg_nine_l99_99938

theorem opposite_of_neg_nine : -(-9) = 9 :=
by
  sorry

end opposite_of_neg_nine_l99_99938


namespace probability_cube_vertices_in_plane_l99_99719

open Finset

noncomputable def choose (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem probability_cube_vertices_in_plane : 
  let total_ways := choose 8 4 in
  let favorable_ways := 12 in
  0 < total_ways →  -- Ensure total_ways is non-zero to avoid division by zero
  let P := (favorable_ways : ℝ) / (total_ways : ℝ) in
  P = 6 / 35 :=
by 
  sorry

end probability_cube_vertices_in_plane_l99_99719


namespace ratio_PM_MQ_eq_1_l99_99264

theorem ratio_PM_MQ_eq_1
  (A B C D E M P Q : ℝ × ℝ)
  (square_side : ℝ)
  (h_square_side : square_side = 15)
  (hA : A = (0, square_side))
  (hB : B = (square_side, square_side))
  (hC : C = (square_side, 0))
  (hD : D = (0, 0))
  (hE : E = (8, 0))
  (hM : M = ((A.1 + E.1) / 2, (A.2 + E.2) / 2))
  (h_slope_AE : E.2 - A.2 = (E.1 - A.1) * -15 / 8)
  (h_P_on_AD : P.2 = 15)
  (h_Q_on_BC : Q.2 = 0)
  (h_PM_len : dist M P = dist M Q) :
  dist P M = dist M Q :=
by sorry

end ratio_PM_MQ_eq_1_l99_99264


namespace cube_vertices_probability_l99_99715

theorem cube_vertices_probability (totalVertices : ℕ) (selectedVertices : ℕ) 
   (totalCombinations : ℕ) (favorableOutcomes : ℕ) : 
   totalVertices = 8 ∧ selectedVertices = 4 ∧ totalCombinations = 70 ∧ favorableOutcomes = 12 → 
   (favorableOutcomes : ℚ) / totalCombinations = 6 / 35 := by
   sorry

end cube_vertices_probability_l99_99715


namespace gunther_free_time_remaining_l99_99415

-- Define the conditions
def vacuum_time : ℕ := 45
def dust_time : ℕ := 60
def mop_time : ℕ := 30
def brushing_time_per_cat : ℕ := 5
def number_of_cats : ℕ := 3
def free_time_hours : ℕ := 3

-- Convert the conditions into a proof problem
theorem gunther_free_time_remaining : 
  let total_cleaning_time := vacuum_time + dust_time + mop_time + brushing_time_per_cat * number_of_cats
  let free_time_minutes := free_time_hours * 60
  in free_time_minutes - total_cleaning_time = 30 :=
by
  sorry

end gunther_free_time_remaining_l99_99415


namespace pythagorean_ratio_l99_99930

variables (a b : ℝ)

theorem pythagorean_ratio (h1 : a > 0) (h2 : b > a) (h3 : b^2 = 13 * (b - a)^2) :
  a / b = 2 / 3 :=
sorry

end pythagorean_ratio_l99_99930


namespace reverse_difference_198_l99_99951

theorem reverse_difference_198 (a : ℤ) : 
  let N := 100 * (a - 1) + 10 * a + (a + 1)
  let M := 100 * (a + 1) + 10 * a + (a - 1)
  M - N = 198 := 
by
  sorry

end reverse_difference_198_l99_99951


namespace smallest_n_term_dec_l99_99155

theorem smallest_n_term_dec (n : ℕ) (h_pos : 0 < n) (h : ∀ d, 0 < d → d = n + 150 → ∀ p, p ∣ d → (p = 2 ∨ p = 5)) :
  n = 10 :=
by {
  sorry
}

end smallest_n_term_dec_l99_99155


namespace perimeter_non_shaded_region_l99_99803

def shaded_area : ℤ := 78
def large_rect_area : ℤ := 80
def small_rect_area : ℤ := 8
def total_area : ℤ := large_rect_area + small_rect_area
def non_shaded_area : ℤ := total_area - shaded_area
def non_shaded_width : ℤ := 2
def non_shaded_length : ℤ := non_shaded_area / non_shaded_width
def non_shaded_perimeter : ℤ := 2 * (non_shaded_length + non_shaded_width)

theorem perimeter_non_shaded_region : non_shaded_perimeter = 14 := 
by
  exact rfl

end perimeter_non_shaded_region_l99_99803


namespace distance_between_centers_l99_99592

noncomputable def distance_centers_inc_exc (PQ PR QR: ℝ) (hPQ: PQ = 17) (hPR: PR = 15) (hQR: QR = 8) : ℝ :=
  let s := (PQ + PR + QR) / 2
  let area := Real.sqrt (s * (s - PQ) * (s - PR) * (s - QR))
  let r := area / s
  let r' := area / (s - QR)
  let PU := s - PQ
  let PV := s
  let PI := Real.sqrt ((PU)^2 + (r)^2)
  let PE := Real.sqrt ((PV)^2 + (r')^2)
  PE - PI

theorem distance_between_centers (PQ PR QR : ℝ) (hPQ: PQ = 17) (hPR: PR = 15) (hQR: QR = 8) :
  distance_centers_inc_exc PQ PR QR hPQ hPR hQR = 5 * Real.sqrt 17 - 3 * Real.sqrt 2 :=
by sorry

end distance_between_centers_l99_99592


namespace find_H_over_G_l99_99293

variable (G H : ℤ)
variable (x : ℝ)

-- Conditions
def condition (G H : ℤ) (x : ℝ) : Prop :=
  x ≠ -7 ∧ x ≠ 0 ∧ x ≠ 6 ∧
  (↑G / (x + 7) + ↑H / (x * (x - 6)) = (x^2 - 3 * x + 15) / (x^3 + x^2 - 42 * x))

-- Theorem Statement
theorem find_H_over_G (G H : ℤ) (x : ℝ) (h : condition G H x) : (H : ℝ) / G = 15 / 7 :=
sorry

end find_H_over_G_l99_99293


namespace smallest_five_digit_multiple_of_18_l99_99547

theorem smallest_five_digit_multiple_of_18 : ∃ n : ℕ, 10000 ≤ n ∧ n ≤ 99999 ∧ n % 18 = 0 ∧ ∀ m : ℕ, 10000 ≤ m ∧ m ≤ 99999 ∧ m % 18 = 0 → n ≤ m :=
  sorry

end smallest_five_digit_multiple_of_18_l99_99547


namespace arithmetic_sequence_l99_99404

theorem arithmetic_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) (n : ℕ) (h_n : n > 0) 
  (h_Sn : S (2 * n) - S (2 * n - 1) + a 2 = 424) : 
  a (n + 1) = 212 :=
sorry

end arithmetic_sequence_l99_99404


namespace marked_percentage_above_cost_l99_99601

theorem marked_percentage_above_cost (CP SP : ℝ) (discount_percentage MP : ℝ) 
  (h1 : CP = 540) 
  (h2 : SP = 457) 
  (h3 : discount_percentage = 26.40901771336554) 
  (h4 : SP = MP * (1 - discount_percentage / 100)) : 
  ((MP - CP) / CP) * 100 = 15 :=
by
  sorry

end marked_percentage_above_cost_l99_99601


namespace children_got_off_bus_l99_99194

-- Conditions
def original_number_of_children : ℕ := 43
def children_left_on_bus : ℕ := 21

-- Definition of the number of children who got off the bus
def children_got_off : ℕ := original_number_of_children - children_left_on_bus

-- Theorem stating the number of children who got off the bus
theorem children_got_off_bus : children_got_off = 22 :=
by
  -- This is to indicate where the proof would go
  sorry

end children_got_off_bus_l99_99194


namespace problem_proof_l99_99904

open Real

noncomputable def angle_B (A C : ℝ) : ℝ := π / 3

noncomputable def area_triangle (a b c : ℝ) : ℝ := 
  (1/2) * a * c * (sqrt 3 / 2)

theorem problem_proof (A B C a b c : ℝ)
  (h1 : 2 * cos A * cos C * (tan A * tan C - 1) = 1)
  (h2 : a + c = sqrt 15)
  (h3 : b = sqrt 3)
  (h4 : B = π / 3) :
  (B = angle_B A C) ∧ 
  (area_triangle a b c = sqrt 3) :=
by
  sorry

end problem_proof_l99_99904


namespace fixed_point_linear_l99_99555

-- Define the linear function y = kx + k + 2
def linear_function (k x : ℝ) : ℝ := k * x + k + 2

-- Prove that the point (-1, 2) lies on the graph of the function for any k
theorem fixed_point_linear (k : ℝ) : linear_function k (-1) = 2 := by
  sorry

end fixed_point_linear_l99_99555


namespace Goat_guilty_l99_99136

-- Condition definitions
def Goat_lied : Prop := sorry
def Beetle_testimony_true : Prop := sorry
def Mosquito_testimony_true : Prop := sorry
def Goat_accused_Beetle_or_Mosquito : Prop := sorry
def Beetle_accused_Goat_or_Mosquito : Prop := sorry
def Mosquito_accused_Beetle_or_Goat : Prop := sorry

-- Theorem: The Goat is guilty
theorem Goat_guilty (G_lied : Goat_lied) 
    (B_true : Beetle_testimony_true) 
    (M_true : Mosquito_testimony_true)
    (G_accuse : Goat_accused_Beetle_or_Mosquito)
    (B_accuse : Beetle_accused_Goat_or_Mosquito)
    (M_accuse : Mosquito_accused_Beetle_or_Goat) : 
  Prop :=
  sorry

end Goat_guilty_l99_99136


namespace quadrilateral_with_three_right_angles_is_rectangle_l99_99505

-- Define a quadrilateral with angles
structure Quadrilateral :=
  (a1 a2 a3 a4 : ℝ)
  (sum_angles : a1 + a2 + a3 + a4 = 360)

-- Define a right angle
def is_right_angle (angle : ℝ) : Prop :=
  angle = 90

-- Define a rectangle
def is_rectangle (q : Quadrilateral) : Prop :=
  is_right_angle q.a1 ∧ is_right_angle q.a2 ∧ is_right_angle q.a3 ∧ is_right_angle q.a4

-- The main theorem: if a quadrilateral has three right angles, it is a rectangle
theorem quadrilateral_with_three_right_angles_is_rectangle 
  (q : Quadrilateral) 
  (h1 : is_right_angle q.a1) 
  (h2 : is_right_angle q.a2) 
  (h3 : is_right_angle q.a3) 
  : is_rectangle q :=
sorry

end quadrilateral_with_three_right_angles_is_rectangle_l99_99505


namespace four_points_no_obtuse_triangle_l99_99708

noncomputable def probability_no_obtuse_triangle : ℝ :=
1 / 64

theorem four_points_no_obtuse_triangle (A B C D : circle) :
  (∀ (P Q : circle) (PQ_angle : ℝ), PQ_angle < π/2) → 
  probability_no_obtuse_triangle = 1 / 64 :=
sorry

end four_points_no_obtuse_triangle_l99_99708


namespace ratio_of_trout_l99_99612

-- Definition of the conditions
def trout_caught_by_Sara : Nat := 5
def trout_caught_by_Melanie : Nat := 10

-- Theorem stating the main claim to be proved
theorem ratio_of_trout : trout_caught_by_Melanie / trout_caught_by_Sara = 2 := by
  sorry

end ratio_of_trout_l99_99612


namespace ball_distribution_ways_l99_99448

theorem ball_distribution_ways :
  let R := 5
  let W := 3
  let G := 2
  let total_balls := 10
  let balls_in_first_box := 4
  ∃ (distributions : ℕ), distributions = (Nat.choose total_balls balls_in_first_box) ∧ distributions = 210 :=
by
  sorry

end ball_distribution_ways_l99_99448


namespace marble_ratio_correct_l99_99662

-- Necessary given conditions
variables (x : ℕ) (Ben_initial John_initial : ℕ) (John_post Ben_post : ℕ)
variables (h1 : Ben_initial = 18)
variables (h2 : John_initial = 17)
variables (h3 : Ben_post = Ben_initial - x)
variables (h4 : John_post = John_initial + x)
variables (h5 : John_post = Ben_post + 17)

-- Define the ratio of the number of marbles Ben gave to John to the number of marbles Ben had initially
def marble_ratio := (x : ℕ) / Ben_initial

-- The theorem we want to prove
theorem marble_ratio_correct (h1 : Ben_initial = 18) (h2 : John_initial = 17) (h3 : Ben_post = Ben_initial - x)
(h4 : John_post = John_initial + x) (h5 : John_post = Ben_post + 17) : marble_ratio x Ben_initial = 1/2 := by 
  sorry

end marble_ratio_correct_l99_99662


namespace stratified_sampling_third_grade_l99_99905

theorem stratified_sampling_third_grade:
  ∀ (students_first students_second students_third total_sample : ℕ),
  students_first = 400 →
  students_second = 400 →
  students_third = 500 →
  total_sample = 65 →
  let sampling_ratio := total_sample / (students_first + students_second + students_third : ℕ)
  in students_third * sampling_ratio = 25 :=
by
  intros students_first students_second students_third total_sample
  intros hf hs ht htots
  rw [hf, hs, ht, htots]
  let sampling_ratio := total_sample / (students_first + students_second + students_third : ℕ)
  rw [sampling_ratio]
  rw [total_sample, students_first, students_second, students_third]
  norm_num
  sorry

end stratified_sampling_third_grade_l99_99905


namespace dog_paws_ground_l99_99033

theorem dog_paws_ground (total_dogs : ℕ) (two_thirds_back_legs : ℕ) (remaining_dogs_four_legs : ℕ) (two_paws_per_back_leg_dog : ℕ) (four_paws_per_four_leg_dog : ℕ) :
  total_dogs = 24 →
  two_thirds_back_legs = 2 * total_dogs / 3 →
  remaining_dogs_four_legs = total_dogs - two_thirds_back_legs →
  two_paws_per_back_leg_dog = 2 →
  four_paws_per_four_leg_dog = 4 →
  (two_thirds_back_legs * two_paws_per_back_leg_dog + remaining_dogs_four_legs * four_paws_per_four_leg_dog) = 64 := 
by 
  sorry

end dog_paws_ground_l99_99033


namespace remainder_5_to_5_to_5_to_5_mod_1000_l99_99038

theorem remainder_5_to_5_to_5_to_5_mod_1000 : (5^(5^(5^5))) % 1000 = 125 :=
by {
  sorry
}

end remainder_5_to_5_to_5_to_5_mod_1000_l99_99038


namespace chord_bisection_l99_99084

theorem chord_bisection {r : ℝ} (PQ RS : Set (ℝ × ℝ)) (O T P Q R S M : ℝ × ℝ)
  (radius_OP : dist O P = 6) (radius_OQ : dist O Q = 6)
  (radius_OR : dist O R = 6) (radius_OS : dist O S = 6) (radius_OT : dist O T = 6)
  (radius_OM : dist O M = 2 * Real.sqrt 13) 
  (PT_eq_8 : dist P T = 8) (TQ_eq_8 : dist T Q = 8)
  (sin_theta_eq_4_5 : Real.sin (Real.arcsin (8 / 10)) = 4 / 5) :
  4 * 5 = 20 :=
by
  sorry

end chord_bisection_l99_99084


namespace pyramid_volume_is_correct_l99_99094

noncomputable def volume_pyramid (A B C G : ℝ × ℝ × ℝ) : ℝ :=
  let base_area := 1 / 2 * (2 * 2) in
  let height := 2 in
  1 / 3 * base_area * height

theorem pyramid_volume_is_correct
  (A B C G : ℝ × ℝ × ℝ)
  (hA : A = (0, 0, 0))
  (hB : B = (2, 0, 0))
  (hC : C = (0, 2, 0))
  (hG : G = (0, 0, 2))
  (side_length : ℝ)
  (side_length_eq : side_length = 2) :
  volume_pyramid A B C G = 4 / 3 :=
by
  rw [hA, hB, hC, hG, side_length_eq]
  sorry

end pyramid_volume_is_correct_l99_99094


namespace projected_increase_is_25_l99_99786

variable (R P : ℝ) -- variables for last year's revenue and projected increase in percentage

-- Conditions
axiom h1 : ∀ (R : ℝ), R > 0
axiom h2 : ∀ (P : ℝ), P/100 ≥ 0
axiom h3 : ∀ (R : ℝ), 0.75 * R = 0.60 * (R + (P/100) * R)

-- Goal
theorem projected_increase_is_25 (R : ℝ) : P = 25 :=
by {
    -- import the required axioms and provide the necessary proof
    apply sorry
}

end projected_increase_is_25_l99_99786


namespace power_mod_remainder_l99_99053

theorem power_mod_remainder (a b c : ℕ) (h1 : 7^40 % 500 = 1) (h2 : 7^4 % 40 = 1) : (7^(7^25) % 500 = 43) :=
sorry

end power_mod_remainder_l99_99053


namespace ratio_of_x_intercepts_l99_99336

theorem ratio_of_x_intercepts (b : ℝ) (hb: b ≠ 0) (u v: ℝ) (h₁: 8 * u + b = 0) (h₂: 4 * v + b = 0) : 
  u / v = 1 / 2 :=
by
  sorry

end ratio_of_x_intercepts_l99_99336


namespace power_modulo_calculation_l99_99052

open Nat

theorem power_modulo_calculation :
  let λ500 := 100
  let λ100 := 20
  (5^5 : ℕ) ≡ 25 [MOD 100]
  (125^5 : ℕ) ≡ 125 [MOD 500]
  (5^{5^{5^5}} : ℕ) % 500 = 125 :=
by
  let λ500 := 100
  let λ100 := 20
  have h1 : (5^5 : ℕ) ≡ 25 [MOD 100] := by sorry
  have h2 : (125^5 : ℕ) ≡ 125 [MOD 500] := by sorry
  sorry

end power_modulo_calculation_l99_99052


namespace dice_sum_lt_10_probability_l99_99498

open Probability

-- Define the event space for rolling two six-sided dice
noncomputable def event_space := {a : ℕ × ℕ | 1 ≤ a.1 ∧ a.1 ≤ 6 ∧ 1 ≤ a.2 ∧ a.2 ≤ 6}

-- Define the event where the sum of numbers showing on two dice is less than 10
def sum_lt_10 (a : ℕ × ℕ) : Prop := (a.1 + a.2 < 10)

-- Calculate the probability that the sum of two fair six-sided dice is less than 10
theorem dice_sum_lt_10_probability : ∃ p : ℚ, p = 5/6 ∧
  (Pr {a ∈ event_space | sum_lt_10 a} = p) :=
begin
  sorry
end

end dice_sum_lt_10_probability_l99_99498


namespace pen_ratio_l99_99032

theorem pen_ratio 
  (Dorothy_pens Julia_pens Robert_pens : ℕ)
  (pen_cost total_cost : ℚ)
  (h1 : Dorothy_pens = Julia_pens / 2)
  (h2 : Robert_pens = 4)
  (h3 : pen_cost = 1.5)
  (h4 : total_cost = 33)
  (h5 : total_cost / pen_cost = Dorothy_pens + Julia_pens + Robert_pens) :
  (Julia_pens / Robert_pens : ℚ) = 3 :=
  sorry

end pen_ratio_l99_99032


namespace find_number_that_satisfies_condition_l99_99658

theorem find_number_that_satisfies_condition : ∃ x : ℝ, x / 3 + 12 = 20 ∧ x = 24 :=
by
  sorry

end find_number_that_satisfies_condition_l99_99658


namespace carnival_masks_costumes_min_l99_99580

theorem carnival_masks_costumes_min : 
  ∀ n x : ℕ, (n = 42) → (3 * n / 7 = 18) → (5 * n / 6 = 35) → 
  (n = 3 * n / 7 + 5 * n / 6 - x) → x = 11 :=
by
  intros n x h₁ h₂ h₃ h₄
  have hn := h₁
  have h_masks := h₂
  have h_costumes := h₃
  have h_eq := h₄
  sorry

end carnival_masks_costumes_min_l99_99580


namespace car_speed_problem_l99_99629

theorem car_speed_problem (S1 S2 : ℝ) (T : ℝ) (avg_speed : ℝ) (H1 : S1 = 70) (H2 : T = 2) (H3 : avg_speed = 80) :
  S2 = 90 :=
by
  have avg_speed_eq : avg_speed = (S1 + S2) / T := sorry
  have h : S2 = 90 := sorry
  exact h

end car_speed_problem_l99_99629


namespace scenic_spots_arrangement_l99_99072

def arrangements : ℕ :=
  let C (n k : ℕ) := Nat.choose n k
  (C 5 3 * C (5 - 3) 1 * C (5 - 3 - 1) 1 + C 5 2 * C (5 - 2) 2 * C (5 - 2 - 2) 1) * 6

theorem scenic_spots_arrangement :
  arrangements = 150 :=
by
  sorry

end scenic_spots_arrangement_l99_99072


namespace line_tangent_to_parabola_proof_l99_99297

noncomputable def line_tangent_to_parabola (d : ℝ) := (∀ x y : ℝ, y = 3 * x + d → y^2 = 12 * x) → d = 1

theorem line_tangent_to_parabola_proof (d : ℝ) (h : ∀ x y : ℝ, y = 3 * x + d → y^2 = 12 * x) : d = 1 :=
sorry

end line_tangent_to_parabola_proof_l99_99297


namespace curve_equation_l99_99546

theorem curve_equation :
  (∃ (x y : ℝ), 2 * x + y - 8 = 0 ∧ x - 2 * y + 1 = 0 ∧ x = 3 ∧ y = 2) ∧
  (∃ (C : ℝ), 
    8 * 3 + 6 * 2 + C = 0 ∧
    8 * x + 6 * y + C = 0 ∧
    4 * x + 3 * y - 18 = 0 ∧
    ∀ x y, 6 * x - 8 * y + 3 = 0 → 
    4 * x + 3 * y - 18 = 0) ∧
  (∃ (a : ℝ), ∀ x y, (x + 1)^2 + 1 = (x - 1)^2 + 9 →
    ((x - 2)^2 + y^2 = 10 ∧ a = 2)) :=
sorry

end curve_equation_l99_99546


namespace residue_of_neg_1235_mod_29_l99_99030

theorem residue_of_neg_1235_mod_29 : 
  ∃ r, 0 ≤ r ∧ r < 29 ∧ (-1235) % 29 = r ∧ r = 12 :=
by
  sorry

end residue_of_neg_1235_mod_29_l99_99030


namespace jonah_total_lemonade_l99_99676

theorem jonah_total_lemonade : 
  0.25 + 0.4166666666666667 + 0.25 + 0.5833333333333334 = 1.5 :=
by
  sorry

end jonah_total_lemonade_l99_99676


namespace geometric_sequence_ratio_l99_99733

theorem geometric_sequence_ratio (a b c q : ℝ) :
  (∃ x : ℝ, x ≠ 0 ∧ b + c - a = x * q ∧ c + a - b = x * q^2 ∧ a + b - c = x * q^3 ∧ a + b + c = x) →
  q^3 + q^2 + q = 1 :=
by
  sorry

end geometric_sequence_ratio_l99_99733


namespace geo_seq_sum_monotone_l99_99433

theorem geo_seq_sum_monotone (q a1 : ℝ) (n : ℕ) (S : ℕ → ℝ) :
  (∀ n, S (n + 1) > S n) ↔ (a1 > 0 ∧ q > 0) :=
sorry -- Proof of the theorem (omitted)

end geo_seq_sum_monotone_l99_99433


namespace line_intersects_parabola_at_one_point_l99_99540

theorem line_intersects_parabola_at_one_point (k : ℝ) : (∃ y : ℝ, -y^2 - 4 * y + 2 = k) ↔ k = 6 :=
by 
  sorry

end line_intersects_parabola_at_one_point_l99_99540


namespace archers_in_golden_l99_99756

variables (soldiers archers swordsmen wearing_golden wearing_black : ℕ)
variables (truth_swordsmen_black lie_swordsmen_golden lie_archers_black truth_archers_golden : ℕ)

-- Given conditions
variables (cond1 : archers + swordsmen = 55)
variables (cond2 : wearing_golden + wearing_black = 55)
variables (cond3 : truth_swordsmen_black + lie_swordsmen_golden + lie_archers_black + truth_archers_golden = 55)
variables (cond4 : wearing_golden = 44)
variables (cond5 : archers = 33)
variables (cond6 : truth_swordsmen_black + lie_archers_black = 22)

-- Define the mathematic equivalent proof problem
theorem archers_in_golden : archers = 22 :=
by
  sorry

end archers_in_golden_l99_99756


namespace imaginary_part_zero_iff_a_eq_neg1_l99_99057

theorem imaginary_part_zero_iff_a_eq_neg1 (a : ℝ) (h : (Complex.I * (a + Complex.I) + a - 1).im = 0) : 
  a = -1 :=
sorry

end imaginary_part_zero_iff_a_eq_neg1_l99_99057


namespace painted_cells_solutions_l99_99588

def painted_cells (k l : ℕ) : ℕ := (2 * k + 1) * (2 * l + 1) - 74

theorem painted_cells_solutions : ∃ k l : ℕ, k * l = 74 ∧ (painted_cells k l = 373 ∨ painted_cells k l = 301) :=
by
  sorry

end painted_cells_solutions_l99_99588


namespace trigonometric_relationship_l99_99560

theorem trigonometric_relationship :
  let a := [10, 9, 8, 7, 6, 4, 3, 2, 1]
  let sum_of_a := a.sum
  let x := Real.sin sum_of_a
  let y := Real.cos sum_of_a
  let z := Real.tan sum_of_a
  sum_of_a = 50 →
  z < x ∧ x < y :=
by
  sorry

end trigonometric_relationship_l99_99560


namespace no_obtuse_triangle_l99_99712

-- Conditions
def points_on_circle_uniformly_at_random (n : ℕ) : Prop :=
  ∀ i < n, ∀ j < n, i ≠ j → ∃ θ_ij : ℝ, 0 ≤ θ_ij ∧ θ_ij ≤ π

-- Theorem statement
theorem no_obtuse_triangle (hn : points_on_circle_uniformly_at_random 4) :
  let p := \left(\frac{1}{2}\right)^6 in
  p = \frac{1}{64} :=
sorry

end no_obtuse_triangle_l99_99712


namespace trig_expression_value_l99_99723

theorem trig_expression_value (α : ℝ) (h : Real.tan (Real.pi + α) = 2) : 
  (Real.sin (α - Real.pi) + Real.cos (Real.pi - α)) / (Real.sin (Real.pi + α) - Real.cos (Real.pi - α)) = 3 := 
by
  sorry

end trig_expression_value_l99_99723


namespace hoseok_wire_length_l99_99249

theorem hoseok_wire_length (side_length : ℕ) (equilateral : Prop) (leftover_wire : ℕ) (total_wire : ℕ)  
  (eq_side : side_length = 19) (eq_leftover : leftover_wire = 15) 
  (eq_equilateral : equilateral) : total_wire = 72 :=
sorry

end hoseok_wire_length_l99_99249


namespace complement_U_A_l99_99599

open Set

-- Definitions of the universal set U and the set A
def U : Set ℕ := {1, 2, 3}
def A : Set ℕ := {1, 2}

-- Proof statement: the complement of A with respect to U is {3}
theorem complement_U_A : U \ A = {3} :=
by
  sorry

end complement_U_A_l99_99599


namespace find_years_simple_interest_l99_99207

variable (R T : ℝ)
variable (P : ℝ := 6000)
variable (additional_interest : ℝ := 360)
variable (rate_diff : ℝ := 2)
variable (H : P * ((R + rate_diff) / 100) * T = P * (R / 100) * T + additional_interest)

theorem find_years_simple_interest (h : P = 6000) (h₁ : P * ((R + 2) / 100) * T = P * (R / 100) * T + 360) : 
T = 3 :=
sorry

end find_years_simple_interest_l99_99207


namespace fewer_hours_worked_l99_99379

noncomputable def total_earnings_summer := 6000
noncomputable def total_weeks_summer := 10
noncomputable def hours_per_week_summer := 50
noncomputable def total_earnings_school_year := 8000
noncomputable def total_weeks_school_year := 40

noncomputable def hourly_wage := total_earnings_summer / (hours_per_week_summer * total_weeks_summer)
noncomputable def total_hours_school_year := total_earnings_school_year / hourly_wage
noncomputable def hours_per_week_school_year := total_hours_school_year / total_weeks_school_year
noncomputable def fewer_hours_per_week := hours_per_week_summer - hours_per_week_school_year

theorem fewer_hours_worked :
  fewer_hours_per_week = hours_per_week_summer - (total_earnings_school_year / hourly_wage / total_weeks_school_year) := by
  sorry

end fewer_hours_worked_l99_99379


namespace smallest_four_digit_multiple_of_18_l99_99690

theorem smallest_four_digit_multiple_of_18 : ∃ n : ℕ, n = 1008 ∧ (1000 ≤ n) ∧ (n < 10000) ∧ (n % 18 = 0) ∧ 
                                ∀ m : ℕ, ((1000 ≤ m) ∧ (m < 10000) ∧ (m % 18 = 0)) → 1008 ≤ m :=
by
  sorry

end smallest_four_digit_multiple_of_18_l99_99690


namespace sum_of_reciprocals_l99_99944

theorem sum_of_reciprocals (x y : ℝ) (h1 : x + y = 14) (h2 : x * y = 45) : 
  1/x + 1/y = 14/45 := 
sorry

end sum_of_reciprocals_l99_99944


namespace turnip_bag_weight_l99_99988

/-- Given six bags with weights 13, 15, 16, 17, 21, and 24 kg,
one bag contains turnips, and the others contain either onions or carrots.
The total weight of the carrots equals twice the total weight of the onions.
Prove that the bag containing turnips can weigh either 13 kg or 16 kg. -/
theorem turnip_bag_weight (ws : list ℕ) (T : ℕ) (O C : ℕ) (h_ws : ws = [13, 15, 16, 17, 21, 24])
  (h_sum : ws.sum = 106) (h_co : C = 2 * O) (h_weight : C + O = 106 - T) :
  T = 13 ∨ T = 16 :=
sorry

end turnip_bag_weight_l99_99988


namespace tangent_line_parabola_l99_99305

theorem tangent_line_parabola (d : ℝ) : 
    (∀ y : ℝ, (-4)^2 - 4 * (y^2 - 4 * y + 4 * d) = 0) ↔ d = 1 :=
by
    sorry

end tangent_line_parabola_l99_99305


namespace no_obtuse_triangle_probability_l99_99699

noncomputable def probability_no_obtuse_triangle : ℝ :=
 let θ := 1/2 in
 let prob_A2_A3_given_A0A1 := (3/8) * (3/8) in
 θ * prob_A2_A3_given_A0A1

theorem no_obtuse_triangle_probability :
  probability_no_obtuse_triangle = 9/128 :=
by
  sorry

end no_obtuse_triangle_probability_l99_99699


namespace triangle_angle_inequality_l99_99572

theorem triangle_angle_inequality (A B C : ℝ) (h : A + B + C = Real.pi) (hA : 0 < A) (hB : 0 < B) (hC : 0 < C) :
  4 / A + 1 / (B + C) ≥ 9 / Real.pi := by
  sorry

end triangle_angle_inequality_l99_99572


namespace probability_no_obtuse_triangle_l99_99696

namespace CirclePoints

noncomputable def no_obtuse_triangle_probability : ℝ := 
  let p := 1/64 in
    p

theorem probability_no_obtuse_triangle (X : ℕ → ℝ) (hcirc : ∀ n, 0 ≤ X n ∧ X n < 2 * π) (hpoints : (∀ n m, n ≠ m → X n ≠ X m)) :
  no_obtuse_triangle_probability = 1/64 :=
sorry

end CirclePoints

end probability_no_obtuse_triangle_l99_99696


namespace smallest_n_terminating_decimal_l99_99168

theorem smallest_n_terminating_decimal :
  ∃ n : ℕ, (∀ m : ℕ, (∃ k : ℕ, m + 150 = 2^k * (5^0))
  → (m ≥ n)) ∧ (n > 0) ∧ (∃ k : ℕ, n + 150 = 2^k * (5^0)) := 
sorry

end smallest_n_terminating_decimal_l99_99168


namespace exponent_rule_example_l99_99664

theorem exponent_rule_example : 3000 * (3000 ^ 3000) = 3000 ^ 3001 := by
  sorry

end exponent_rule_example_l99_99664


namespace expected_participants_2003_l99_99438

theorem expected_participants_2003 :
  let participants : ℕ → ℝ := λ n, 1000 * 1.6 ^ n
  participants 3 = 4096 := by
  sorry

end expected_participants_2003_l99_99438


namespace range_of_a_l99_99285

theorem range_of_a (m : ℝ) (a : ℝ) : 
  m ∈ Set.Icc (-1 : ℝ) (1 : ℝ) →
  (∀ x₁ x₂ : ℝ, x₁^2 - m * x₁ - 2 = 0 ∧ x₂^2 - m * x₂ - 2 = 0 → a^2 - 5 * a - 3 ≥ |x₁ - x₂|) ↔ (a ≥ 6 ∨ a ≤ -1) :=
by
  sorry

end range_of_a_l99_99285


namespace not_possible_coloring_l99_99772

def color : Nat → Option ℕ := sorry

def all_colors_used (f : Nat → Option ℕ) : Prop := 
  (∃ n, f n = some 0) ∧ (∃ n, f n = some 1) ∧ (∃ n, f n = some 2)

def valid_coloring (f : Nat → Option ℕ) : Prop :=
  ∀ (a b : Nat), 1 < a → 1 < b → f a ≠ f b → f (a * b) ≠ f a ∧ f (a * b) ≠ f b

theorem not_possible_coloring : ¬ (∃ f : Nat → Option ℕ, all_colors_used f ∧ valid_coloring f) := 
sorry

end not_possible_coloring_l99_99772


namespace remainder_mod_500_l99_99041

theorem remainder_mod_500 :
  ( 5^(5^(5^5)) ) % 500 = 125 :=
by
  -- proof goes here
  sorry

end remainder_mod_500_l99_99041


namespace ratio_u_v_l99_99321

theorem ratio_u_v (b : ℝ) (hb : b ≠ 0) (u v : ℝ) 
  (h1 : 0 = 8 * u + b) 
  (h2 : 0 = 4 * v + b) 
  : u / v = 1 / 2 := 
by sorry

end ratio_u_v_l99_99321


namespace tank_length_l99_99655

theorem tank_length (W D : ℝ) (cost_per_sq_m total_cost : ℝ) (L : ℝ):
  W = 12 →
  D = 6 →
  cost_per_sq_m = 0.70 →
  total_cost = 520.8 →
  total_cost = cost_per_sq_m * ((2 * (W * D)) + (2 * (L * D)) + (L * W)) →
  L = 25 :=
by
  intros hW hD hCostPerSqM hTotalCost hEquation
  sorry

end tank_length_l99_99655


namespace gcd_12a_18b_l99_99573

theorem gcd_12a_18b (a b : ℕ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : a.gcd b = 15) : (12 * a).gcd (18 * b) = 90 :=
by sorry

end gcd_12a_18b_l99_99573


namespace ratio_of_x_intercepts_l99_99332

theorem ratio_of_x_intercepts (b : ℝ) (hb : b ≠ 0) (u v: ℝ)
  (h1: 0 = 8 * u + b) (h2: 0 = 4 * v + b) : u / v = 1 / 2 :=
by sorry

end ratio_of_x_intercepts_l99_99332


namespace chocolates_for_sister_l99_99906
-- Importing necessary library

-- Lean 4 statement of the problem
theorem chocolates_for_sister (S : ℕ) 
  (herself_chocolates_per_saturday : ℕ := 2)
  (birthday_gift_chocolates : ℕ := 10)
  (saturdays_in_month : ℕ := 4)
  (total_chocolates : ℕ := 22) 
  (monthly_chocolates_herself := saturdays_in_month * herself_chocolates_per_saturday) 
  (equation : saturdays_in_month * S + monthly_chocolates_herself + birthday_gift_chocolates = total_chocolates) : 
  S = 1 :=
  sorry

end chocolates_for_sister_l99_99906


namespace x_intercept_of_line_is_six_l99_99339

theorem x_intercept_of_line_is_six : ∃ x : ℝ, (∃ y : ℝ, y = 0) ∧ (2*x - 4*y = 12) ∧ x = 6 :=
by {
  sorry
}

end x_intercept_of_line_is_six_l99_99339


namespace investment_calculation_l99_99950

theorem investment_calculation
    (R Trishul Vishal Alok Harshit : ℝ)
    (hTrishul : Trishul = 0.9 * R)
    (hVishal : Vishal = 0.99 * R)
    (hAlok : Alok = 1.035 * Trishul)
    (hHarshit : Harshit = 0.95 * Vishal)
    (hTotal : R + Trishul + Vishal + Alok + Harshit = 22000) :
  R = 22000 / 3.8655 ∧
  Trishul = 0.9 * R ∧
  Vishal = 0.99 * R ∧
  Alok = 1.035 * Trishul ∧
  Harshit = 0.95 * Vishal ∧
  R + Trishul + Vishal + Alok + Harshit = 22000 :=
sorry

end investment_calculation_l99_99950


namespace average_age_new_students_l99_99115

theorem average_age_new_students (A : ℚ)
    (avg_original_age : ℚ := 48)
    (num_new_students : ℚ := 120)
    (new_avg_age : ℚ := 44)
    (total_students : ℚ := 160) :
    let num_original_students := total_students - num_new_students
    let total_age_original := num_original_students * avg_original_age
    let total_age_all := total_students * new_avg_age
    total_age_original + (num_new_students * A) = total_age_all → A = 42.67 := 
by
  intros
  sorry

end average_age_new_students_l99_99115


namespace no_obtuse_triangle_probability_l99_99701

noncomputable def probability_no_obtuse_triangle : ℝ :=
 let θ := 1/2 in
 let prob_A2_A3_given_A0A1 := (3/8) * (3/8) in
 θ * prob_A2_A3_given_A0A1

theorem no_obtuse_triangle_probability :
  probability_no_obtuse_triangle = 9/128 :=
by
  sorry

end no_obtuse_triangle_probability_l99_99701


namespace work_efficiency_ratio_l99_99955

theorem work_efficiency_ratio (a b k : ℝ) (ha : a = k * b) (hb : b = 1/15)
  (hab : a + b = 1/5) : k = 2 :=
by sorry

end work_efficiency_ratio_l99_99955


namespace sum_of_roots_l99_99885

theorem sum_of_roots (x1 x2 : ℝ) (h : x1 * x2 = -3) (hx1 : x1 + x2 = 2) :
  x1 + x2 = 2 :=
by {
  sorry
}

end sum_of_roots_l99_99885


namespace total_birds_count_l99_99526

def blackbirds_per_tree : ℕ := 3
def tree_count : ℕ := 7
def magpies : ℕ := 13

theorem total_birds_count : (blackbirds_per_tree * tree_count) + magpies = 34 := by
  sorry

end total_birds_count_l99_99526


namespace rhombus_area_and_perimeter_l99_99622

theorem rhombus_area_and_perimeter (d1 d2 : ℝ) (h1 : d1 = 18) (h2 : d2 = 26) :
  let area := (d1 * d2) / 2
  let s := Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2)
  let perimeter := 4 * s
  area = 234 ∧ perimeter = 20 * Real.sqrt 10 := by
  sorry

end rhombus_area_and_perimeter_l99_99622


namespace ratio_of_intercepts_l99_99325

variable {c : ℝ} (non_zero_c : c ≠ 0) (u v : ℝ)
-- Condition: The first line, slope 8, y-intercept c, x-intercept (u, 0)
variable (h_u : u = -c / 8)
-- Condition: The second line, slope 4, y-intercept c, x-intercept (v, 0)
variable (h_v : v = -c / 4)

theorem ratio_of_intercepts (non_zero_c : c ≠ 0)
    (h_u : u = -c / 8) (h_v : v = -c / 4) : u / v = 1 / 2 :=
by
  sorry

end ratio_of_intercepts_l99_99325


namespace monotonic_increasing_iff_l99_99245

noncomputable def f (x b : ℝ) : ℝ := (x - b) * Real.log x + x^2

theorem monotonic_increasing_iff (b : ℝ) :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ Real.exp 1 → 0 ≤ (Real.log x - b/x + 1 + 2*x)) ↔ b ∈ Set.Iic (3 : ℝ) :=
by
  sorry

end monotonic_increasing_iff_l99_99245


namespace value_of_a_l99_99640

theorem value_of_a (x a : ℝ) (h1 : 0 < x) (h2 : x < 1 / a) (h3 : ∀ x, x * (1 - a * x) ≤ 1 / 12) : a = 3 :=
sorry

end value_of_a_l99_99640


namespace product_of_y_coordinates_l99_99281

noncomputable def distance (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((A.1 - B.1) ^ 2 + (A.2 - B.2) ^ 2)

theorem product_of_y_coordinates : 
  let P1 := (1, 2 + 4 * Real.sqrt 2)
  let P2 := (1, 2 - 4 * Real.sqrt 2)
  distance (5, 2) P1 = 12 ∧ distance (5, 2) P2 = 12 →
  (P1.2 * P2.2 = -28) :=
by
  intros
  sorry

end product_of_y_coordinates_l99_99281


namespace fraction_to_decimal_l99_99824

/-- The decimal equivalent of 1/4 is 0.25. -/
theorem fraction_to_decimal : (1 : ℝ) / 4 = 0.25 :=
by
  sorry

end fraction_to_decimal_l99_99824


namespace second_order_derivative_parametric_l99_99508

noncomputable def x (t : ℝ) : ℝ := t ^ (1/2)
noncomputable def y (t : ℝ) : ℝ := (t - 1) ^ (1/3)

-- Define the first derivative of x with respect to t
noncomputable def dx_dt (t : ℝ) : ℝ :=
  deriv x t

-- Define the first derivative of y with respect to t
noncomputable def dy_dt (t : ℝ) : ℝ :=
  deriv y t

-- Define dy/dx
noncomputable def dy_dx (t : ℝ) : ℝ :=
  dy_dt t / dx_dt t

-- Define the second derivative y''_xx
noncomputable def d2y_xx (t : ℝ) : ℝ :=
  deriv (λ t, dy_dx t) t / dx_dt t

-- The theorem that matches the question to the answer
theorem second_order_derivative_parametric :
  ∀ t : ℝ, d2y_xx t = - (4 * (t + 3)) / (9 * ((t - 1) ^ (5/3))) :=
by
  sorry

end second_order_derivative_parametric_l99_99508


namespace find_k_unique_solution_l99_99036

theorem find_k_unique_solution :
  ∀ k : ℝ, (∀ x : ℝ, x ≠ 0 → (1/(3*x) = (k - x)/8) → (3*x^2 + (8 - 3*k)*x = 0)) →
    k = 8 / 3 :=
by
  intros k h
  -- Using sorry here to skip the proof
  sorry

end find_k_unique_solution_l99_99036


namespace totalPoundsOfFoodConsumed_l99_99848

def maxConsumptionPerGuest : ℝ := 2.5
def minNumberOfGuests : ℕ := 165

theorem totalPoundsOfFoodConsumed : 
    maxConsumptionPerGuest * (minNumberOfGuests : ℝ) = 412.5 := by
  sorry

end totalPoundsOfFoodConsumed_l99_99848


namespace coefficient_m5n5_in_mn_pow10_l99_99636

-- Definition of the binomial coefficient
def binomial_coeff (n k : ℕ) : ℕ := Nat.choose n k

-- The main theorem statement
theorem coefficient_m5n5_in_mn_pow10 : 
  (∃ c, (m + n) ^ 10 = c * m^5 * n^5 + ∑ (k ≠ 5), (binomial_coeff 10 k) * m^(10 - k) * n^k) → 
  c = 252 := 
by 
  sorry

end coefficient_m5n5_in_mn_pow10_l99_99636


namespace neg_sqrt_two_sq_l99_99024

theorem neg_sqrt_two_sq : (- Real.sqrt 2) ^ 2 = 2 := 
by
  sorry

end neg_sqrt_two_sq_l99_99024


namespace find_a_tangent_line_l99_99287

theorem find_a_tangent_line (a : ℝ) : 
  (∃ (x0 y0 : ℝ), y0 = a * x0^2 + (15/4 : ℝ) * x0 - 9 ∧ 
                  (y0 = 0 ∨ (x0 = 3/2 ∧ y0 = 27/4)) ∧ 
                  ∃ (m : ℝ), (0 - y0) = m * (1 - x0) ∧ (m = 2 * a * x0 + 15/4)) → 
  (a = -1 ∨ a = -25/64) := 
sorry

end find_a_tangent_line_l99_99287


namespace symmetric_point_reflection_l99_99608

theorem symmetric_point_reflection (x y : ℝ) : (2, -(-5)) = (2, 5) := by
  sorry

end symmetric_point_reflection_l99_99608


namespace sum_digits_in_possibilities_l99_99437

noncomputable def sum_of_digits (a b c d : ℕ) : ℕ :=
  a + b + c + d

theorem sum_digits_in_possibilities :
  ∃ (a b c d : ℕ), 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
  0 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 
  0 ≤ c ∧ c ≤ 9 ∧ 0 ≤ d ∧ d ≤ 9 ∧ 
  (sum_of_digits a b c d = 10 ∨ sum_of_digits a b c d = 18 ∨ sum_of_digits a b c d = 19) := sorry

end sum_digits_in_possibilities_l99_99437


namespace no_obtuse_triangle_l99_99711

-- Conditions
def points_on_circle_uniformly_at_random (n : ℕ) : Prop :=
  ∀ i < n, ∀ j < n, i ≠ j → ∃ θ_ij : ℝ, 0 ≤ θ_ij ∧ θ_ij ≤ π

-- Theorem statement
theorem no_obtuse_triangle (hn : points_on_circle_uniformly_at_random 4) :
  let p := \left(\frac{1}{2}\right)^6 in
  p = \frac{1}{64} :=
sorry

end no_obtuse_triangle_l99_99711


namespace diver_descend_rate_l99_99204

theorem diver_descend_rate (depth : ℕ) (time : ℕ) (rate : ℕ) 
  (h1 : depth = 6400) (h2 : time = 200) : rate = 32 :=
by
  sorry

end diver_descend_rate_l99_99204


namespace monotonicity_f_range_of_a_l99_99566

noncomputable theory

-- Definitions of the functions
def f (a : ℝ) (x : ℝ) := (2*x - 1) * Real.exp x - a * (x^2 + x)
def g (a : ℝ) (x : ℝ) := -a * (x^2 + 1)

-- Monotonicity proof statement
theorem monotonicity_f (a : ℝ) :
  (a ≤ 0 →
    (∀ x, x < -0.5 → deriv (f a) x < 0) ∧
    (∀ x, x > -0.5 → deriv (f a) x > 0)) ∧
  (0 < a ∧ a < Real.exp (-0.5) →
    (∀ x, x < Real.log a → deriv (f a) x > 0) ∧
    (∀ x, Real.log a < x ∧ x < -0.5 → deriv (f a) x < 0) ∧
    (∀ x, x > -0.5 → deriv (f a) x > 0)) ∧
  (a = Real.exp (-0.5) → ∀ x, deriv (f a) x ≥ 0) ∧
  (a > Real.exp (-0.5) →
    (∀ x, x < -0.5 → deriv (f a) x > 0) ∧
    (∀ x, -0.5 < x ∧ x < Real.log a → deriv (f a) x < 0) ∧
    (∀ x, x > Real.log a → deriv (f a) x > 0)) :=
sorry

-- Range of 'a' proof statement
theorem range_of_a (a : ℝ) :
  (∀ x, f a x ≥ g a x) → (1 ≤ a ∧ a ≤ 4 * Real.exp 1.5) :=
sorry

end monotonicity_f_range_of_a_l99_99566


namespace average_student_headcount_l99_99137

theorem average_student_headcount (h1 : ℕ := 10900) (h2 : ℕ := 10500) (h3 : ℕ := 10700) (h4 : ℕ := 11300) : 
  (h1 + h2 + h3 + h4) / 4 = 10850 := 
by 
  sorry

end average_student_headcount_l99_99137


namespace butterfly_cocoon_l99_99444

theorem butterfly_cocoon (c l : ℕ) (h1 : l + c = 120) (h2 : l = 3 * c) : c = 30 :=
by
  sorry

end butterfly_cocoon_l99_99444


namespace value_calculation_l99_99510

-- Definition of constants used in the problem
def a : ℝ := 1.3333
def b : ℝ := 3.615
def expected_value : ℝ := 4.81998845

-- The proposition to be proven
theorem value_calculation : a * b = expected_value :=
by sorry

end value_calculation_l99_99510


namespace arithmetic_sequence_problem_l99_99087

theorem arithmetic_sequence_problem 
  (a : ℕ → ℕ) 
  (a1 : a 1 = 3) 
  (d : ℕ := 2) 
  (h : ∀ n, a n = a 1 + (n - 1) * d) 
  (h_25 : a n = 25) : 
  n = 12 := 
by
  sorry

end arithmetic_sequence_problem_l99_99087


namespace sin_beta_l99_99558

open Real

theorem sin_beta {α β : ℝ} (hα : 0 < α ∧ α < π / 2)
  (hβ : 0 < β ∧ β < π / 2)
  (h_cosα : cos α = 2 * sqrt 5 / 5)
  (h_sinαβ : sin (α - β) = -3 / 5) :
  sin β = 2 * sqrt 5 / 5 := 
sorry

end sin_beta_l99_99558


namespace probability_of_two_red_two_blue_l99_99969

-- Definitions for the conditions
def total_red_marbles : ℕ := 12
def total_blue_marbles : ℕ := 8
def total_marbles : ℕ := total_red_marbles + total_blue_marbles
def num_selected_marbles : ℕ := 4
def num_red_selected : ℕ := 2
def num_blue_selected : ℕ := 2

-- Definition for binomial coefficient (combinations)
def C (n k : ℕ) : ℕ := n.choose k

-- Probability calculation
def probability_two_red_two_blue :=
  (C total_red_marbles num_red_selected * C total_blue_marbles num_blue_selected : ℚ) / C total_marbles num_selected_marbles

-- The theorem statement
theorem probability_of_two_red_two_blue :
  probability_two_red_two_blue = 1848 / 4845 :=
by
  sorry

end probability_of_two_red_two_blue_l99_99969


namespace train_passes_man_in_approximately_24_seconds_l99_99015

noncomputable def train_length : ℝ := 880 -- length of the train in meters
noncomputable def train_speed_kmph : ℝ := 120 -- speed of the train in km/h
noncomputable def man_speed_kmph : ℝ := 12 -- speed of the man in km/h

noncomputable def kmph_to_mps (speed: ℝ) : ℝ := speed * (1000 / 3600)

noncomputable def train_speed_mps : ℝ := kmph_to_mps train_speed_kmph
noncomputable def man_speed_mps : ℝ := kmph_to_mps man_speed_kmph
noncomputable def relative_speed : ℝ := train_speed_mps + man_speed_mps

noncomputable def time_to_pass : ℝ := train_length / relative_speed

theorem train_passes_man_in_approximately_24_seconds :
  abs (time_to_pass - 24) < 1 :=
sorry

end train_passes_man_in_approximately_24_seconds_l99_99015


namespace domain_shift_l99_99078

theorem domain_shift (f : ℝ → ℝ) :
  {x : ℝ | 1 ≤ x ∧ x ≤ 2} = {x | -2 ≤ x ∧ x ≤ -1} →
  {x : ℝ | ∃ y : ℝ, x = y - 1 ∧ 1 ≤ y ∧ y ≤ 2} =
  {x : ℝ | ∃ y : ℝ, x = y + 2 ∧ -2 ≤ y ∧ y ≤ -1} :=
by
  sorry

end domain_shift_l99_99078


namespace find_a_l99_99440

theorem find_a (a : ℝ) (A B : ℝ × ℝ × ℝ) (hA : A = (-1, 1, -a)) (hB : B = (-a, 3, -1)) (hAB : dist A B = 2) : a = -1 := by
  sorry

end find_a_l99_99440


namespace inverse_function_log3_l99_99294

noncomputable def f (x : ℝ) : ℝ := Real.logb 3 x

theorem inverse_function_log3 :
  ∀ x : ℝ, x > 0 →
  ∃ y : ℝ, f (3 ^ y) = y := 
sorry

end inverse_function_log3_l99_99294


namespace at_least_one_good_part_l99_99814

theorem at_least_one_good_part 
  (total_parts : ℕ) 
  (good_parts : ℕ) 
  (defective_parts : ℕ) 
  (choose : ℕ → ℕ → ℕ) 
  (total_ways : ℕ) 
  (defective_ways : ℕ) 
  (result : ℕ) : 
  total_parts = 20 → 
  good_parts = 16 → 
  defective_parts = 4 → 
  choose 20 3 = total_ways → 
  choose 4 3 = defective_ways → 
  total_ways - defective_ways = result → 
  result = 1136 :=
by 
  intros;
  sorry

end at_least_one_good_part_l99_99814


namespace roots_squared_sum_l99_99666

theorem roots_squared_sum (p q r : ℂ) (h : ∀ x : ℂ, 3 * x ^ 3 - 3 * x ^ 2 + 6 * x - 9 = 0 → x = p ∨ x = q ∨ x = r) :
  p^2 + q^2 + r^2 = -3 :=
by
  sorry

end roots_squared_sum_l99_99666


namespace num_ordered_triples_l99_99202

theorem num_ordered_triples 
  (a b c : ℕ)
  (h_cond1 : 1 ≤ a ∧ a ≤ b ∧ b ≤ c)
  (h_cond2 : a * b * c = 4 * (a * b + b * c + c * a)) : 
  ∃ (n : ℕ), n = 5 :=
sorry

end num_ordered_triples_l99_99202


namespace ratio_u_v_l99_99323

theorem ratio_u_v (b : ℝ) (hb : b ≠ 0) (u v : ℝ) 
  (h1 : 0 = 8 * u + b) 
  (h2 : 0 = 4 * v + b) 
  : u / v = 1 / 2 := 
by sorry

end ratio_u_v_l99_99323


namespace fruit_salad_weight_l99_99796

theorem fruit_salad_weight (melon berries : ℝ) (h_melon : melon = 0.25) (h_berries : berries = 0.38) : melon + berries = 0.63 :=
by
  sorry

end fruit_salad_weight_l99_99796


namespace total_cost_l99_99660

theorem total_cost (cost_sandwich cost_soda cost_cookie : ℕ)
    (num_sandwich num_soda num_cookie : ℕ) 
    (h1 : cost_sandwich = 4) 
    (h2 : cost_soda = 3) 
    (h3 : cost_cookie = 2) 
    (h4 : num_sandwich = 4) 
    (h5 : num_soda = 6) 
    (h6 : num_cookie = 7):
    cost_sandwich * num_sandwich + cost_soda * num_soda + cost_cookie * num_cookie = 48 :=
by
  sorry

end total_cost_l99_99660


namespace remaining_amount_correct_l99_99191

-- Definitions for the given conditions
def deposit_percentage : ℝ := 0.05
def deposit_amount : ℝ := 50

-- The correct answer we need to prove
def remaining_amount_to_be_paid : ℝ := 950

-- Stating the theorem (proof not required)
theorem remaining_amount_correct (total_price : ℝ) 
    (H1 : deposit_amount = total_price * deposit_percentage) : 
    total_price - deposit_amount = remaining_amount_to_be_paid :=
by
  sorry

end remaining_amount_correct_l99_99191


namespace smallest_positive_integer_for_terminating_decimal_l99_99178

theorem smallest_positive_integer_for_terminating_decimal: ∃ n: ℕ, (n > 0) ∧ (∀ p : ℕ, (p ∣ (n + 150)) → (p=1 ∨ p=2 ∨ p=4 ∨ p=5 ∨ p=8 ∨ p=10 ∨ p=16 ∨ p=20 ∨ p=25 ∨ p=32 ∨ p=40 ∨ p=50 ∨ p=64 ∨ p=80 ∨ p=100 ∨ p=125 ∨ p=128 ∨ p=160)) ∧ n = 10 :=
by
  sorry

end smallest_positive_integer_for_terminating_decimal_l99_99178


namespace seating_arrangements_total_l99_99263

def num_round_tables := 3
def num_rect_tables := 4
def num_square_tables := 2
def num_couches := 2
def num_benches := 3
def num_extra_chairs := 5

def seats_per_round_table := 6
def seats_per_rect_table := 7
def seats_per_square_table := 4
def seats_per_couch := 3
def seats_per_bench := 5

def total_seats : Nat :=
  num_round_tables * seats_per_round_table +
  num_rect_tables * seats_per_rect_table +
  num_square_tables * seats_per_square_table +
  num_couches * seats_per_couch +
  num_benches * seats_per_bench +
  num_extra_chairs

theorem seating_arrangements_total :
  total_seats = 80 :=
by
  simp [total_seats, num_round_tables, seats_per_round_table,
        num_rect_tables, seats_per_rect_table, num_square_tables,
        seats_per_square_table, num_couches, seats_per_couch,
        num_benches, seats_per_bench, num_extra_chairs]
  done

end seating_arrangements_total_l99_99263


namespace smallest_four_digit_multiple_of_18_l99_99686

-- Define the concept of a four-digit number
def four_digit (N : ℕ) : Prop := 1000 ≤ N ∧ N < 10000

-- Define the concept of a multiple of 18
def multiple_of_18 (N : ℕ) : Prop := ∃ k : ℕ, N = 18 * k

-- Define the combined condition for N being a four-digit multiple of 18
def four_digit_multiple_of_18 (N : ℕ) : Prop := four_digit N ∧ multiple_of_18 N

-- State that 1008 is the smallest such number
theorem smallest_four_digit_multiple_of_18 : ∀ N : ℕ, four_digit_multiple_of_18 N → 1008 ≤ N := 
by
  intros N H
  sorry

end smallest_four_digit_multiple_of_18_l99_99686


namespace sandy_comic_books_l99_99110

-- Problem definition
def initial_comic_books := 14
def sold_comic_books := initial_comic_books / 2
def remaining_comic_books := initial_comic_books - sold_comic_books
def bought_comic_books := 6
def final_comic_books := remaining_comic_books + bought_comic_books

-- Proof statement
theorem sandy_comic_books : final_comic_books = 13 := by
  sorry

end sandy_comic_books_l99_99110


namespace min_range_of_three_test_takers_l99_99959

-- Proposition: The minimum possible range in scores of the 3 test-takers
-- where the ranges of their scores in the 5 practice tests are 18, 26, and 32, is 76.
theorem min_range_of_three_test_takers (r1 r2 r3: ℕ) 
  (h1 : r1 = 18) (h2 : r2 = 26) (h3 : r3 = 32) : 
  (r1 + r2 + r3) = 76 := by
  sorry

end min_range_of_three_test_takers_l99_99959


namespace base_angle_isosceles_l99_99260

-- Define an isosceles triangle with one angle being 100 degrees
def isosceles_triangle (A B C : Type) (angle_A : ℝ) (angle_B : ℝ) (angle_C : ℝ) : Prop :=
  (A = B ∨ B = C ∨ C = A) ∧ (angle_A + angle_B + angle_C = 180) ∧ (angle_A = 100)

-- The main theorem statement
theorem base_angle_isosceles {A B C : Type} (angle_A : ℝ) (angle_B : ℝ) (angle_C : ℝ) :
  isosceles_triangle A B C angle_A angle_B angle_C → (angle_B = 40 ∨ angle_C = 40) :=
  sorry

end base_angle_isosceles_l99_99260


namespace find_n_l99_99790

theorem find_n :
  ∃ n : ℕ, 120 ^ 5 + 105 ^ 5 + 78 ^ 5 + 33 ^ 5 = n ^ 5 ∧ 
  (∀ m : ℕ, 120 ^ 5 + 105 ^ 5 + 78 ^ 5 + 33 ^ 5 = m ^ 5 → m = 144) :=
by
  sorry

end find_n_l99_99790


namespace isosceles_triangle_side_length_l99_99653

/-- A regular hexagon with a side length of 2 forms three isosceles triangles, 
    each with a vertex at the center and base as one of the hexagon's sides.
    Given that the sum of the areas of these triangles equals half the hexagon's area,
    the length of one of the two congruent sides of each isosceles triangle is 2. -/
theorem isosceles_triangle_side_length (s : ℝ) (a_hexagon : ℝ)
    (h_side_length : s = 2)
    (h_area_condition : 3 * (1 / 2) * s * (s / 2) = (1 / 2) * a_hexagon): 
    s = 2 := 
sorry

end isosceles_triangle_side_length_l99_99653


namespace triangle_with_angle_ratio_obtuse_l99_99900

theorem triangle_with_angle_ratio_obtuse 
  (a b c : ℝ) 
  (h_sum : a + b + c = 180) 
  (h_ratio : a = 2 * d ∧ b = 2 * d ∧ c = 5 * d) : 
  90 < c :=
by
  sorry

end triangle_with_angle_ratio_obtuse_l99_99900


namespace sufficient_but_not_necessary_condition_l99_99594

theorem sufficient_but_not_necessary_condition (a b : ℝ) : 
  (a ≥ 1 ∧ b ≥ 1) → (a + b ≥ 2) ∧ ¬((a + b ≥ 2) → (a ≥ 1 ∧ b ≥ 1)) :=
by {
  sorry
}

end sufficient_but_not_necessary_condition_l99_99594


namespace simple_interest_correct_l99_99523

def principal : ℝ := 10040.625
def rate : ℝ := 8
def time : ℕ := 5

theorem simple_interest_correct :
  (principal * rate * time / 100) = 40162.5 :=
by 
  sorry

end simple_interest_correct_l99_99523


namespace original_expenditure_beginning_month_l99_99661

theorem original_expenditure_beginning_month (A E : ℝ)
  (h1 : E = 35 * A)
  (h2 : E + 84 = 42 * (A - 1))
  (h3 : E + 124 = 37 * (A + 1))
  (h4 : E + 154 = 40 * (A + 1)) :
  E = 630 := 
sorry

end original_expenditure_beginning_month_l99_99661


namespace smallest_n_for_terminating_fraction_l99_99173

-- Define what it means for a number to be a product of only prime factors of 2 and 5
def isTerminatingDenominator (d : ℕ) : Prop := ∃ (a b : ℕ), d = 2^a * 5^b

-- The main statement to prove
theorem smallest_n_for_terminating_fraction : 
  ∃ (n : ℕ), 0 < n ∧ isTerminatingDenominator (n + 150) ∧ 
  ∀ (m : ℕ), (0 < m → isTerminatingDenominator (m + 150) → n ≤ m)
:= sorry

end smallest_n_for_terminating_fraction_l99_99173


namespace train_pass_time_l99_99349

theorem train_pass_time
  (v : ℝ) (l_tunnel l_train : ℝ) (h_v : v = 75) (h_l_tunnel : l_tunnel = 3.5) (h_l_train : l_train = 0.25) :
  (l_tunnel + l_train) / v * 60 = 3 :=
by 
  -- Placeholder for the proof
  sorry

end train_pass_time_l99_99349


namespace sum_of_731_and_one_fifth_l99_99340

theorem sum_of_731_and_one_fifth :
  (7.31 + (1 / 5) = 7.51) :=
sorry

end sum_of_731_and_one_fifth_l99_99340


namespace jasmine_paperclips_l99_99586

theorem jasmine_paperclips :
  ∃ k : ℕ, (4 * 3^k > 500) ∧ (∀ n < k, 4 * 3^n ≤ 500) ∧ k = 5 ∧ (n = 6) :=
by {
  sorry
}

end jasmine_paperclips_l99_99586


namespace binary_to_base4_conversion_l99_99537

theorem binary_to_base4_conversion :
  let b := 110110100
  let b_2 := Nat.ofDigits 2 [1, 1, 0, 1, 1, 0, 1, 0, 0]
  let b_4 := Nat.ofDigits 4 [3, 1, 2, 2, 0]
  b_2 = b → b_4 = 31220 :=
by
  intros b b_2 b_4 h
  sorry

end binary_to_base4_conversion_l99_99537


namespace one_fifty_percent_of_eighty_l99_99503

theorem one_fifty_percent_of_eighty : (150 / 100) * 80 = 120 :=
  by sorry

end one_fifty_percent_of_eighty_l99_99503


namespace rowing_time_l99_99200

def man_speed_still := 10.0
def river_speed := 1.2
def total_distance := 9.856

def upstream_speed := man_speed_still - river_speed
def downstream_speed := man_speed_still + river_speed

def one_way_distance := total_distance / 2
def time_upstream := one_way_distance / upstream_speed
def time_downstream := one_way_distance / downstream_speed

theorem rowing_time :
  time_upstream + time_downstream = 1 :=
by
  sorry

end rowing_time_l99_99200


namespace remainder_of_power_mod_l99_99039

noncomputable def carmichael (n : ℕ) : ℕ := sorry  -- Define Carmichael function (as a placeholder)

theorem remainder_of_power_mod :
  ∀ (n : ℕ), carmichael 1000 = 100 → carmichael 100 = 20 → 
    (5 ^ 5 ^ 5 ^ 5) % 1000 = 625 :=
by
  intros n h₁ h₂
  sorry

end remainder_of_power_mod_l99_99039


namespace base8_addition_l99_99209

theorem base8_addition : (234 : ℕ) + (157 : ℕ) = (4 * 8^2 + 1 * 8^1 + 3 * 8^0 : ℕ) :=
by sorry

end base8_addition_l99_99209


namespace tangent_line_parabola_l99_99308

theorem tangent_line_parabola (d : ℝ) : 
  (∀ x y : ℝ, y = 3 * x + d → ∃! x y, y^2 = 12 * x) → d = 3 := 
by
  intro h
  -- Here, "h" would be our hypothesis where we assume the line is tangent to the parabola
  sorry

end tangent_line_parabola_l99_99308


namespace line_tangent_parabola_unique_d_l99_99303

theorem line_tangent_parabola_unique_d :
  (∃ d : ℝ, ∀ x : ℝ, y = 3 * x + d -> y^2 = 12 * x) -> d = 1 := by
sorry

end line_tangent_parabola_unique_d_l99_99303


namespace largest_binomial_coeff_and_rational_terms_l99_99064

theorem largest_binomial_coeff_and_rational_terms 
  (n : ℕ) 
  (h_sum_coeffs : 4^n - 2^n = 992) 
  (T : ℕ → ℝ → ℝ)
  (x : ℝ) :
  (∃ (r1 r2 : ℕ), T r1 x = 270 * x^(22/3) ∧ T r2 x = 90 * x^6)
  ∧
  (∃ (r3 r4 : ℕ), T r3 x = 243 * x^10 ∧ T r4 x = 90 * x^6)
:= 
  
sorry

end largest_binomial_coeff_and_rational_terms_l99_99064


namespace solve_polynomial_relation_l99_99615

--Given Conditions
def polynomial_relation (x y : ℤ) : Prop := y^3 = x^3 + 8 * x^2 - 6 * x + 8 

--Proof Problem
theorem solve_polynomial_relation : ∃ (x y : ℤ), (polynomial_relation x y) ∧ 
  ((y = 11 ∧ x = 9) ∨ (y = 2 ∧ x = 0)) :=
by 
  sorry

end solve_polynomial_relation_l99_99615


namespace remainder_of_exponentiation_is_correct_l99_99049

-- Define the given conditions
def modulus := 500
def exponent := 5 ^ (5 ^ 5)
def carmichael_500 := 100
def carmichael_100 := 20

-- Prove the main theorem
theorem remainder_of_exponentiation_is_correct :
  (5 ^ exponent) % modulus = 125 := 
by
  -- Skipping the proof
  sorry

end remainder_of_exponentiation_is_correct_l99_99049


namespace reeya_fourth_subject_score_l99_99925

theorem reeya_fourth_subject_score (s1 s2 s3 s4 : ℕ) (avg : ℕ) (n : ℕ)
  (h_avg : avg = 75) (h_n : n = 4) (h_s1 : s1 = 65) (h_s2 : s2 = 67) (h_s3 : s3 = 76)
  (h_total_sum : avg * n = s1 + s2 + s3 + s4) : s4 = 92 := by
  sorry

end reeya_fourth_subject_score_l99_99925


namespace number_of_true_statements_l99_99082

noncomputable def f (x : ℝ) : ℝ := x^2
noncomputable def g (x : ℝ) : ℝ := 1/x
noncomputable def h (x : ℝ) : ℝ := 2 * Real.exp (Real.ln x)

lemma is_monotonically_increasing_F_on_interval : 
  ∀ x ∈ Ioo (- (1 / (2 : ℝ)^(1 / 3))) 0, deriv (λ x, f x - g x) x > 0 := 
sorry

lemma separation_line_f_g_exists :
  ∃ (k b : ℝ), (∀ x, f x ≥ k * x + b) ∧ (∀ x < 0, g x ≤ k * x + b) ∧ b = -4 := 
sorry

lemma separation_line_f_g_k_range :
  ∃ (k b : ℝ), (-4 < k ∧ k ≤ 0) ∧ (∀ x, f x ≥ k * x + b) ∧ (∀ x < 0, g x ≤ k * x + b) := 
sorry

lemma unique_separation_line_f_h :
  ∃! (k b : ℝ), (∀ x, f x ≥ k * x + b) ∧ (∀ x > 0, h x ≤ k * x + b) ∧ k = 2 * Real.sqrt 2 ∧ b = - Real.exp 1 := 
sorry

theorem number_of_true_statements : 
  (is_monotonically_increasing_F_on_interval ∧ separation_line_f_g_exists ∧ ¬ separation_line_f_g_k_range ∧ unique_separation_line_f_h) = 3 := 
sorry

end number_of_true_statements_l99_99082


namespace find_x_l99_99941

-- Define the given conditions
def constant_ratio (k : ℚ) : Prop :=
  ∀ (x y : ℚ), (3 * x - 4) / (y + 15) = k

def initial_condition (k : ℚ) : Prop :=
  (3 * 5 - 4) / (4 + 15) = k

def new_condition (k : ℚ) (x : ℚ) : Prop :=
  (3 * x - 4) / 30 = k

-- Prove that x = 406/57 given the conditions
theorem find_x (k : ℚ) (x : ℚ) :
  constant_ratio k →
  initial_condition k →
  new_condition k x →
  x = 406 / 57 :=
  sorry

end find_x_l99_99941


namespace complement_union_l99_99096

def A : Set ℝ := { x | -1 < x ∧ x < 1 }
def B : Set ℝ := { x | x ≥ 1 }
def C (s : Set ℝ) : Set ℝ := { x | ¬ s x }

theorem complement_union :
  C (A ∪ B) = { x | x ≤ -1 } :=
by {
  sorry
}

end complement_union_l99_99096


namespace dante_coconuts_l99_99792

theorem dante_coconuts (P : ℕ) (D : ℕ) (S : ℕ) (hP : P = 14) (hD : D = 3 * P) (hS : S = 10) :
  (D - S) = 32 :=
by
  sorry

end dante_coconuts_l99_99792


namespace decreasing_interval_f_l99_99937

noncomputable def f (x : ℝ) : ℝ := (1 / 2) * x^2 - Real.log x

theorem decreasing_interval_f :
  ∀ x : ℝ, x > 0 → (f (x) = (1 / 2) * x^2 - Real.log x) →
  (∃ a b : ℝ, 0 < a ∧ a ≤ b ∧ b = 1 ∧ ∀ y, a < y ∧ y ≤ b → f (y) ≤ f (y+1)) := sorry

end decreasing_interval_f_l99_99937


namespace cube_vertices_probability_l99_99714

theorem cube_vertices_probability (totalVertices : ℕ) (selectedVertices : ℕ) 
   (totalCombinations : ℕ) (favorableOutcomes : ℕ) : 
   totalVertices = 8 ∧ selectedVertices = 4 ∧ totalCombinations = 70 ∧ favorableOutcomes = 12 → 
   (favorableOutcomes : ℚ) / totalCombinations = 6 / 35 := by
   sorry

end cube_vertices_probability_l99_99714


namespace remainder_5_pow_5_pow_5_pow_5_mod_500_l99_99044

-- Conditions
def λ (n : ℕ) : ℕ := n.gcd20p1.factorial5div
def M : ℕ := 5 ^ (5 ^ 5)

-- Theorem: Prove the remainder
theorem remainder_5_pow_5_pow_5_pow_5_mod_500 :
  M % 500 = 125 :=
by sorry

end remainder_5_pow_5_pow_5_pow_5_mod_500_l99_99044


namespace solution_set_of_quadratic_inequality_l99_99875

theorem solution_set_of_quadratic_inequality 
  (f : ℝ → ℝ) 
  (h₁ : ∀ x, f x < 0 ↔ x < -1 ∨ x > 1 / 3)
  (h₂ : ∀ x, f (Real.exp x) > 0 ↔ x < -Real.log 3) : 
  ∀ x, f (Real.exp x) > 0 ↔ x < -Real.log 3 := 
by
  intro x
  exact h₂ x

end solution_set_of_quadratic_inequality_l99_99875


namespace max_A_value_l99_99403

-- Define the conditions
variable (n : ℕ) (x : ℕ → Bool)
variable (h_odd : n % 2 = 1)

-- Define the counting function A
def count_triplets : ℕ :=
  ∑ i in Finset.range n, ∑ j in Finset.range i, ∑ k in Finset.range j,
  if x k ≠ x j ∧ (x i, x j, x k) = (false, true, false) ∨ (x i, x j, x k) = (true, false, true) then 1 else 0

-- Maximum value of A for odd n
theorem max_A_value : count_triplets n x = n * (n^2 - 1) / 24 :=
sorry

end max_A_value_l99_99403


namespace binary_sequences_no_three_consecutive_zeros_l99_99219

-- Define the primary theorem to be proven
theorem binary_sequences_no_three_consecutive_zeros :
  let total_sequences := 
      (∑ m in Finset.range 6, 
         (Nat.choose 11 m) * (Nat.choose (11 - m) (10 - 2 * m))) in
  total_sequences = 24068 :=
by {
  -- Using the specific transformation without the need to provide proof steps here
  sorry
}

end binary_sequences_no_three_consecutive_zeros_l99_99219


namespace find_f_a_l99_99593

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then 4 * Real.logb 2 (-x) else abs (x^2 + a * x)

theorem find_f_a (a : ℝ) (h : a ≠ 0) (h1 : f a (f a (-Real.sqrt 2)) = 4) : f a a = 8 :=
sorry

end find_f_a_l99_99593


namespace algebra_or_drafting_not_both_l99_99467

theorem algebra_or_drafting_not_both {A D : Finset ℕ} (h1 : (A ∩ D).card = 10) (h2 : A.card = 24) (h3 : D.card - (A ∩ D).card = 11) : (A ∪ D).card - (A ∩ D).card = 25 := by
  sorry

end algebra_or_drafting_not_both_l99_99467


namespace perimeter_of_figure_l99_99651

-- Given conditions
def side_length : Nat := 2
def num_horizontal_segments : Nat := 16
def num_vertical_segments : Nat := 10

-- Define a function to calculate the perimeter based on the given conditions
def calculate_perimeter (side_length : Nat) (num_horizontal_segments : Nat) (num_vertical_segments : Nat) : Nat :=
  (num_horizontal_segments * side_length) + (num_vertical_segments * side_length)

-- Statement to be proved
theorem perimeter_of_figure : calculate_perimeter side_length num_horizontal_segments num_vertical_segments = 52 :=
by
  -- The proof would go here
  sorry

end perimeter_of_figure_l99_99651


namespace coeff_m5n5_in_m_plus_n_pow_10_l99_99637

theorem coeff_m5n5_in_m_plus_n_pow_10 :
  binomial (10, 5) = 252 := by
sorry

end coeff_m5n5_in_m_plus_n_pow_10_l99_99637


namespace range_of_f_1_over_f_2_l99_99360

theorem range_of_f_1_over_f_2 {f : ℝ → ℝ} (h1 : ∀ x > 0, f x > 0)
  (h2 : ∀ x > 0, 2 * f x < x * (deriv f x) ∧ x * (deriv f x) < 3 * f x) :
  1 / 8 < f 1 / f 2 ∧ f 1 / f 2 < 1 / 4 :=
by sorry

end range_of_f_1_over_f_2_l99_99360


namespace ticket_sales_total_l99_99345

variable (price_adult : ℕ) (price_child : ℕ) (total_tickets : ℕ) (child_tickets : ℕ)

def total_money_collected (price_adult : ℕ) (price_child : ℕ) (total_tickets : ℕ) (child_tickets : ℕ) : ℕ :=
  let adult_tickets := total_tickets - child_tickets
  let total_child := child_tickets * price_child
  let total_adult := adult_tickets * price_adult
  total_child + total_adult

theorem ticket_sales_total :
  price_adult = 6 →
  price_child = 4 →
  total_tickets = 21 →
  child_tickets = 11 →
  total_money_collected price_adult price_child total_tickets child_tickets = 104 :=
by
  intros
  unfold total_money_collected
  simp
  sorry

end ticket_sales_total_l99_99345


namespace smallest_positive_period_intervals_of_monotonic_decrease_range_m_value_l99_99272

noncomputable def f (x m : ℝ) : ℝ := 2 * (Real.cos x)^2 + 2 * Real.sqrt 3 * (Real.sin x) * (Real.cos x) + m

theorem smallest_positive_period (m : ℝ) : ∃ T > 0, ∀ x, f x m = f (x + T) m ∧ T = Real.pi := sorry

theorem intervals_of_monotonic_decrease (m : ℝ) : ∀ k : ℤ, is_monotonic_decreasing (f x m) ($$left_bound = ℝ.., interval : [ℝ]) := sorry

theorem range_m_value (x : ℝ) : x ∈ [0, Real.pi / 2] → ∃ m : ℝ, m = 1 / 2 ∧ (∀ y ∈ (range (f x m)), y ∈ [1 / 2, 7 / 2]) := sorry

end smallest_positive_period_intervals_of_monotonic_decrease_range_m_value_l99_99272


namespace find_natural_numbers_l99_99948

theorem find_natural_numbers (x y : ℕ) (h1 : x > y) (h2 : x + y + (x - y) + x * y + x / y = 3^5) : 
  (x = 6 ∧ y = 3) := 
sorry

end find_natural_numbers_l99_99948


namespace triangle_angle_B_l99_99897

theorem triangle_angle_B {A B C : ℝ} (h1 : A = 60) (h2 : B = 2 * C) (h3 : A + B + C = 180) : B = 80 :=
sorry

end triangle_angle_B_l99_99897


namespace number_of_restaurants_l99_99974

theorem number_of_restaurants
  (total_units : ℕ)
  (residential_units : ℕ)
  (non_residential_units : ℕ)
  (restaurants : ℕ)
  (h1 : total_units = 300)
  (h2 : residential_units = total_units / 2)
  (h3 : non_residential_units = total_units - residential_units)
  (h4 : restaurants = non_residential_units / 2)
  : restaurants = 75 := 
by
  sorry

end number_of_restaurants_l99_99974


namespace circle_radius_l99_99866

theorem circle_radius (x y : ℝ) :
  (∃ r, r > 0 ∧ (∀ x y, x^2 - 8*x + y^2 - 4*y + 16 = 0 → r = 2)) :=
sorry

end circle_radius_l99_99866


namespace num_real_solutions_abs_eq_l99_99881

theorem num_real_solutions_abs_eq :
  (∃ x y : ℝ, x ≠ y ∧ |x-1| = |x-2| + |x-3| + |x-4| 
    ∧ |y-1| = |y-2| + |y-3| + |y-4| 
    ∧ ∀ z : ℝ, |z-1| = |z-2| + |z-3| + |z-4| → (z = x ∨ z = y)) := sorry

end num_real_solutions_abs_eq_l99_99881


namespace shorter_side_length_l99_99958

theorem shorter_side_length (L W : ℝ) (h₁ : L * W = 120) (h₂ : 2 * L + 2 * W = 46) : L = 8 ∨ W = 8 := 
by 
  sorry

end shorter_side_length_l99_99958


namespace greatest_k_dividing_n_l99_99838

noncomputable def num_divisors (n : ℕ) : ℕ :=
  n.divisors.card

theorem greatest_k_dividing_n (n : ℕ) (h_pos : n > 0)
  (h_n_divisors : num_divisors n = 120)
  (h_5n_divisors : num_divisors (5 * n) = 144) :
  ∃ k : ℕ, 5^k ∣ n ∧ (∀ m : ℕ, 5^m ∣ n → m ≤ k) ∧ k = 4 :=
by sorry

end greatest_k_dividing_n_l99_99838


namespace profit_A_after_upgrade_profit_B_constrained_l99_99831

-- Part Ⅰ
theorem profit_A_after_upgrade (x : ℝ) (h : x^2 - 300 * x ≤ 0) : 0 < x ∧ x ≤ 300 := sorry

-- Part Ⅱ
theorem profit_B_constrained (a x : ℝ) (h1 : a ≤ (x/125 + 500/x + 3/2)) (h2 : x = 250) : 0 < a ∧ a ≤ 5.5 := sorry

end profit_A_after_upgrade_profit_B_constrained_l99_99831


namespace inequality_Cauchy_Schwarz_l99_99552

theorem inequality_Cauchy_Schwarz (a b : ℝ) : 
  (a^4 + b^4) * (a^2 + b^2) ≥ (a^3 + b^3)^2 :=
by
  sorry

end inequality_Cauchy_Schwarz_l99_99552


namespace matrix_addition_correct_l99_99681

def matrixA : Matrix (Fin 2) (Fin 2) ℤ := fun i j =>
  if i = 0 then
    if j = 0 then 4 else -2
  else
    if j = 0 then -3 else 5

def matrixB : Matrix (Fin 2) (Fin 2) ℤ := fun i j =>
  if i = 0 then
    if j = 0 then -6 else 0
  else
    if j = 0 then 7 else -8

def resultMatrix : Matrix (Fin 2) (Fin 2) ℤ := fun i j =>
  if i = 0 then
    if j = 0 then -2 else -2
  else
    if j = 0 then 4 else -3

theorem matrix_addition_correct :
  matrixA + matrixB = resultMatrix :=
by
  sorry

end matrix_addition_correct_l99_99681


namespace sequence_fifth_number_l99_99055

theorem sequence_fifth_number : (5^2 - 1) = 24 :=
by {
  sorry
}

end sequence_fifth_number_l99_99055


namespace minimum_n_value_l99_99400

def satisfies_terms_condition (n : ℕ) : Prop :=
  (n + 1) * (n + 1) ≥ 2021

theorem minimum_n_value :
  ∃ n : ℕ, n > 0 ∧ satisfies_terms_condition n ∧ ∀ m : ℕ, m > 0 ∧ satisfies_terms_condition m → n ≤ m := by
  sorry

end minimum_n_value_l99_99400


namespace tan_half_sum_sq_l99_99251

theorem tan_half_sum_sq (a b : ℝ) : 
  3 * (Real.cos a + Real.cos b) + 5 * (Real.cos a * Real.cos b + 1) = 0 → 
  ∃ (x : ℝ), (x = (Real.tan (a / 2) + Real.tan (b / 2))^2) ∧ (x = 6 ∨ x = 26) := 
by
  intro h
  sorry

end tan_half_sum_sq_l99_99251


namespace future_ratio_l99_99631

variable (j e : ℕ)

-- Conditions
axiom condition1 : j - 3 = 4 * (e - 3)
axiom condition2 : j - 5 = 5 * (e - 5)

-- Theorem to be proved
theorem future_ratio : ∃ x : ℕ, x = 1 ∧ ((j + x) / (e + x) = 3) := by
  sorry

end future_ratio_l99_99631


namespace smallest_n_term_dec_l99_99156

theorem smallest_n_term_dec (n : ℕ) (h_pos : 0 < n) (h : ∀ d, 0 < d → d = n + 150 → ∀ p, p ∣ d → (p = 2 ∨ p = 5)) :
  n = 10 :=
by {
  sorry
}

end smallest_n_term_dec_l99_99156


namespace ab_geq_3_plus_cd_l99_99095

theorem ab_geq_3_plus_cd (a b c d : ℝ) 
  (h1 : a ≥ b) (h2 : b ≥ c) (h3 : c ≥ d)
  (h4 : a + b + c + d = 13) (h5 : a^2 + b^2 + c^2 + d^2 = 43) :
  a * b ≥ 3 + c * d := 
sorry

end ab_geq_3_plus_cd_l99_99095


namespace min_value_x2_y2_z2_l99_99093

theorem min_value_x2_y2_z2 (x y z : ℝ) (h : x^3 + y^3 + z^3 - 3 * x * y * z = 8) : x^2 + y^2 + z^2 ≥ 3 :=
sorry

end min_value_x2_y2_z2_l99_99093


namespace vertex_h_is_3_l99_99627

open Real

theorem vertex_h_is_3 (a b c : ℝ) (h : ℝ)
    (h_cond : 3 * (a * 3^2 + b * 3 + c) + 6 = 3) : 
    4 * (a * x^2 + b * x + c) = 12 * (x - 3)^2 + 24 → 
    h = 3 := 
by 
sorry

end vertex_h_is_3_l99_99627


namespace find_m_value_l99_99623

theorem find_m_value (m : ℤ) (h1 : m - 2 ≠ 0) (h2 : |m| = 2) : m = -2 :=
by {
  sorry
}

end find_m_value_l99_99623


namespace smallest_four_digit_multiple_of_18_l99_99687

-- Define the concept of a four-digit number
def four_digit (N : ℕ) : Prop := 1000 ≤ N ∧ N < 10000

-- Define the concept of a multiple of 18
def multiple_of_18 (N : ℕ) : Prop := ∃ k : ℕ, N = 18 * k

-- Define the combined condition for N being a four-digit multiple of 18
def four_digit_multiple_of_18 (N : ℕ) : Prop := four_digit N ∧ multiple_of_18 N

-- State that 1008 is the smallest such number
theorem smallest_four_digit_multiple_of_18 : ∀ N : ℕ, four_digit_multiple_of_18 N → 1008 ≤ N := 
by
  intros N H
  sorry

end smallest_four_digit_multiple_of_18_l99_99687


namespace number_of_ordered_triples_l99_99225

/-- 
Prove the number of ordered triples (x, y, z) of positive integers that satisfy 
  lcm(x, y) = 180, lcm(x, z) = 210, and lcm(y, z) = 420 is 2.
-/
theorem number_of_ordered_triples (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h₁ : Nat.lcm x y = 180) (h₂ : Nat.lcm x z = 210) (h₃ : Nat.lcm y z = 420) : 
  ∃ (n : ℕ), n = 2 := 
sorry

end number_of_ordered_triples_l99_99225


namespace joanna_marbles_l99_99265

theorem joanna_marbles (m n : ℕ) (h1 : m * n = 720) (h2 : m > 1) (h3 : n > 1) :
  ∃ (count : ℕ), count = 28 :=
by
  -- Use the properties of divisors and conditions to show that there are 28 valid pairs (m, n).
  sorry

end joanna_marbles_l99_99265


namespace jenni_age_l99_99124

theorem jenni_age (B J : ℕ) (h1 : B + J = 70) (h2 : B - J = 32) : J = 19 :=
by
  sorry

end jenni_age_l99_99124


namespace thabo_hardcover_books_l99_99000

theorem thabo_hardcover_books:
  ∃ (H P F : ℕ), H + P + F = 280 ∧ P = H + 20 ∧ F = 2 * P ∧ H = 55 := by
  sorry

end thabo_hardcover_books_l99_99000


namespace max_value_of_k_l99_99224

theorem max_value_of_k (n : ℕ) (k : ℕ) (h : 3^11 = k * (2 * n + k + 1) / 2) : k = 486 :=
sorry

end max_value_of_k_l99_99224


namespace evaluate_101_times_101_l99_99859

theorem evaluate_101_times_101 : (101 * 101 = 10201) :=
by {
  sorry
}

end evaluate_101_times_101_l99_99859


namespace num_initial_pairs_of_shoes_l99_99100

theorem num_initial_pairs_of_shoes (lost_shoes remaining_pairs : ℕ)
  (h1 : lost_shoes = 9)
  (h2 : remaining_pairs = 20) :
  (initial_pairs : ℕ) = 25 :=
sorry

end num_initial_pairs_of_shoes_l99_99100


namespace initial_dogs_l99_99837

theorem initial_dogs (D : ℕ) (h : D + 5 + 3 = 10) : D = 2 :=
by sorry

end initial_dogs_l99_99837


namespace jogger_ahead_engine_l99_99362

-- Define the given constants for speed and length
def jogger_speed : ℝ := 2.5 -- in m/s
def train_speed : ℝ := 12.5 -- in m/s
def train_length : ℝ := 120 -- in meters
def passing_time : ℝ := 40 -- in seconds

-- Define the target distance
def jogger_ahead : ℝ := 280 -- in meters

-- Lean 4 statement to prove the jogger is 280 meters ahead of the train's engine
theorem jogger_ahead_engine :
  passing_time * (train_speed - jogger_speed) - train_length = jogger_ahead :=
by
  sorry

end jogger_ahead_engine_l99_99362


namespace four_points_no_obtuse_triangle_l99_99709

noncomputable def probability_no_obtuse_triangle : ℝ :=
1 / 64

theorem four_points_no_obtuse_triangle (A B C D : circle) :
  (∀ (P Q : circle) (PQ_angle : ℝ), PQ_angle < π/2) → 
  probability_no_obtuse_triangle = 1 / 64 :=
sorry

end four_points_no_obtuse_triangle_l99_99709


namespace range_of_a_minimize_S_l99_99028

open Real

-- Problem 1: Prove the range of a 
theorem range_of_a (a : ℝ) : (∃ x ≠ 0, x^3 - 3*x^2 + (2 - a)*x = 0) ↔ a > -1 / 4 := sorry

-- Problem 2: Prove the minimizing value of a for the area function S(a)
noncomputable def S (a : ℝ) : ℝ := 
  let α := sorry -- α is the root depending on a (to be determined from the context)
  let β := sorry -- β is the root depending on a (to be determined from the context)
  (1/4 * α^4 - α^3 + (1/2) * (2-a) * α^2) + (1/4 * β^4 - β^3 + (1/2) * (2-a) * β^2)

theorem minimize_S (a : ℝ) : a = 38 - 27 * sqrt 2 → S a = S (38 - 27 * sqrt 2) := sorry

end range_of_a_minimize_S_l99_99028


namespace daphne_two_visits_in_365_days_l99_99668

def visits_in_days (d1 d2 : ℕ) (days : ℕ) : ℕ :=
  days / Nat.lcm d1 d2

theorem daphne_two_visits_in_365_days :
  let days := 365
  let lcm_all := Nat.lcm 4 (Nat.lcm 6 (Nat.lcm 8 10))
  (visits_in_days 4 6 lcm_all + 
   visits_in_days 4 8 lcm_all + 
   visits_in_days 4 10 lcm_all + 
   visits_in_days 6 8 lcm_all + 
   visits_in_days 6 10 lcm_all + 
   visits_in_days 8 10 lcm_all) * 
   (days / lcm_all) = 129 :=
by
  sorry

end daphne_two_visits_in_365_days_l99_99668


namespace total_price_correct_l99_99979

-- Definitions of given conditions
def original_price : Float := 120
def discount_rate : Float := 0.30
def tax_rate : Float := 0.08

-- Definition of the final selling price
def sale_price : Float := original_price * (1 - discount_rate)
def total_selling_price : Float := sale_price * (1 + tax_rate)

-- Lean 4 statement to prove the total selling price is 90.72
theorem total_price_correct : total_selling_price = 90.72 := by
  sorry

end total_price_correct_l99_99979


namespace remainder_of_power_mod_l99_99040

noncomputable def carmichael (n : ℕ) : ℕ := sorry  -- Define Carmichael function (as a placeholder)

theorem remainder_of_power_mod :
  ∀ (n : ℕ), carmichael 1000 = 100 → carmichael 100 = 20 → 
    (5 ^ 5 ^ 5 ^ 5) % 1000 = 625 :=
by
  intros n h₁ h₂
  sorry

end remainder_of_power_mod_l99_99040


namespace analogical_reasoning_l99_99343

theorem analogical_reasoning {a b c : ℝ} (h1 : c ≠ 0) : 
  (a + b) * c = a * c + b * c → (a + b) / c = a / c + b / c := 
by 
  sorry

end analogical_reasoning_l99_99343


namespace compute_abs_ab_eq_2_sqrt_111_l99_99934

theorem compute_abs_ab_eq_2_sqrt_111 (a b : ℝ) 
  (h1 : b^2 - a^2 = 25)
  (h2 : a^2 + b^2 = 49) : 
  |a * b| = 2 * Real.sqrt 111 := 
sorry

end compute_abs_ab_eq_2_sqrt_111_l99_99934


namespace find_n_from_binomial_condition_l99_99741

theorem find_n_from_binomial_condition (n : ℕ) (h : Nat.choose n 3 = 7 * Nat.choose n 1) : n = 43 :=
by
  -- The proof steps would be filled in here
  sorry

end find_n_from_binomial_condition_l99_99741


namespace circumscribed_circle_radius_l99_99742

theorem circumscribed_circle_radius (b c : ℝ) (A : ℝ) (R : ℝ)
  (h1 : b = 6) (h2 : c = 2) (h3 : A = π / 3) :
  R = (2 * Real.sqrt 21) / 3 :=
by
  sorry

end circumscribed_circle_radius_l99_99742


namespace polynomial_inequality_l99_99315

-- Define the polynomial P and its condition
def P (a b c : ℝ) (x : ℝ) : ℝ := 12 * x^3 + a * x^2 + b * x + c
-- Define the polynomial Q and its condition
def Q (a b c : ℝ) (x : ℝ) : ℝ := (x^2 + x + 2001)^3 + a * (x^2 + x + 2001)^2 + b * (x^2 + x + 2001) + c

-- Assumptions
axiom P_has_distinct_roots (a b c : ℝ) : ∃ p q r : ℝ, p ≠ q ∧ q ≠ r ∧ p ≠ r ∧ P a b c p = 0 ∧ P a b c q = 0 ∧ P a b c r = 0
axiom Q_has_no_real_roots (a b c : ℝ) : ¬ ∃ x : ℝ, Q a b c x = 0

-- The goal to prove
theorem polynomial_inequality (a b c : ℝ) (h1 : ∃ p q r : ℝ, p ≠ q ∧ q ≠ r ∧ p ≠ r ∧ P a b c p = 0 ∧ P a b c q = 0 ∧ P a b c r = 0)
  (h2 : ¬ ∃ x : ℝ, Q a b c x = 0) : 2001^3 + a * 2001^2 + b * 2001 + c > 1 / 64 :=
by {
  -- sorry is added to skip the proof part
  sorry
}

end polynomial_inequality_l99_99315


namespace residue_of_neg_1235_mod_29_l99_99029

theorem residue_of_neg_1235_mod_29 : 
  ∃ r, 0 ≤ r ∧ r < 29 ∧ (-1235) % 29 = r ∧ r = 12 :=
by
  sorry

end residue_of_neg_1235_mod_29_l99_99029


namespace min_rectangles_to_cover_minimum_number_of_rectangles_required_l99_99501

-- Definitions based on the conditions
def corners_type1 : Nat := 12
def corners_type2 : Nat := 12

theorem min_rectangles_to_cover (type1_corners type2_corners : Nat) (h1 : type1_corners = corners_type1) (h2 : type2_corners = corners_type2) : Nat :=
12

theorem minimum_number_of_rectangles_required (type1_corners type2_corners : Nat) (h1 : type1_corners = corners_type1) (h2 : type2_corners = corners_type2) :
  min_rectangles_to_cover type1_corners type2_corners h1 h2 = 12 := by
  sorry

end min_rectangles_to_cover_minimum_number_of_rectangles_required_l99_99501


namespace tangent_line_parabola_l99_99306

theorem tangent_line_parabola (d : ℝ) : 
    (∀ y : ℝ, (-4)^2 - 4 * (y^2 - 4 * y + 4 * d) = 0) ↔ d = 1 :=
by
    sorry

end tangent_line_parabola_l99_99306


namespace simplify_expression_l99_99614

variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variable (a b : V)

theorem simplify_expression :
  (1/2 : ℝ) • (2 • a + 8 • b) - (4 • a - 2 • b) = 6 • b - 3 • a :=
by sorry

end simplify_expression_l99_99614


namespace loaves_count_l99_99654

theorem loaves_count 
  (init_loaves : ℕ)
  (sold_percent : ℕ) 
  (bulk_purchase : ℕ)
  (bulk_discount_percent : ℕ)
  (evening_purchase : ℕ)
  (evening_discount_percent : ℕ)
  (final_loaves : ℕ)
  (h1 : init_loaves = 2355)
  (h2 : sold_percent = 30)
  (h3 : bulk_purchase = 750)
  (h4 : bulk_discount_percent = 20)
  (h5 : evening_purchase = 489)
  (h6 : evening_discount_percent = 15)
  (h7 : final_loaves = 2888) :
  let mid_morning_sold := init_loaves * sold_percent / 100
  let loaves_after_sale := init_loaves - mid_morning_sold
  let bulk_discount_loaves := bulk_purchase * bulk_discount_percent / 100
  let loaves_after_bulk_purchase := loaves_after_sale + bulk_purchase
  let evening_discount_loaves := evening_purchase * evening_discount_percent / 100
  let loaves_after_evening_purchase := loaves_after_bulk_purchase + evening_purchase
  loaves_after_evening_purchase = final_loaves :=
by
  sorry

end loaves_count_l99_99654


namespace find_f_2017_l99_99727

noncomputable def f : ℝ → ℝ :=
sorry

axiom cond1 : ∀ x : ℝ, f (1 + x) + f (1 - x) = 0
axiom cond2 : ∀ x : ℝ, f (-x) = f x
axiom cond3 : ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → f x = 2^x - 1

theorem find_f_2017 : f 2017 = 1 :=
by
  sorry

end find_f_2017_l99_99727


namespace find_a_l99_99780

-- define the necessary mathematical objects and properties

noncomputable def f (x : ℝ) : ℝ := 3 + (Real.log x / Real.log 3)

def is_monotonic (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, 0 < x → 0 < y → (x ≤ y → f x ≤ f y)

-- define the function g
def g (x : ℝ) : ℝ :=
  f x - 2* (1 / (x * Real.log 3)) - 3

theorem find_a (a : ℕ) (x0 : ℝ) (h_monotonic : is_monotonic f)
  (h_cond : ∀ x : ℝ, 0 < x → f (f x - Real.log x / Real.log 3) = 4)
  (h_sol : g x0 = 0) (h_interval : x0 ∈ Set.Ioo a (a + 1)) :
  a = 2 :=
sorry

end find_a_l99_99780


namespace find_z_plus_1_over_y_l99_99801

theorem find_z_plus_1_over_y (x y z : ℝ) (h1 : x * y * z = 1) (h2 : x + 1 / z = 7) (h3 : y + 1 / x = 20) : 
  z + 1 / y = 29 / 139 := 
by 
  sorry

end find_z_plus_1_over_y_l99_99801


namespace largest_divisor_of_odd_sequence_for_even_n_l99_99138

theorem largest_divisor_of_odd_sequence_for_even_n (n : ℕ) (h : n % 2 = 0) : 
  ∃ d : ℕ, d = 105 ∧ ∀ k : ℕ, k = (n+1)*(n+3)*(n+5)*(n+7)*(n+9)*(n+11)*(n+13) → 105 ∣ k :=
sorry

end largest_divisor_of_odd_sequence_for_even_n_l99_99138


namespace cows_with_no_spots_l99_99917

-- Definitions of conditions
def total_cows : Nat := 140
def cows_with_red_spot : Nat := (40 * total_cows) / 100
def cows_without_red_spot : Nat := total_cows - cows_with_red_spot
def cows_with_blue_spot : Nat := (25 * cows_without_red_spot) / 100

-- Theorem statement asserting the number of cows with no spots
theorem cows_with_no_spots : (total_cows - cows_with_red_spot - cows_with_blue_spot) = 63 := by
  -- Proof would go here
  sorry

end cows_with_no_spots_l99_99917


namespace bombardment_deaths_l99_99901

variable (initial_population final_population : ℕ)
variable (fear_factor death_percentage : ℝ)

theorem bombardment_deaths (h1 : initial_population = 4200)
                           (h2 : final_population = 3213)
                           (h3 : fear_factor = 0.15)
                           (h4 : ∃ x, death_percentage = x / 100 ∧ 
                                       4200 - (x / 100) * 4200 - fear_factor * (4200 - (x / 100) * 4200) = 3213) :
                           death_percentage = 0.1 :=
by
  sorry

end bombardment_deaths_l99_99901


namespace g_sum_eq_neg_one_l99_99079

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

-- Main theorem to prove g(1) + g(-1) = -1 given the conditions
theorem g_sum_eq_neg_one
  (h1 : ∀ x y : ℝ, f (x - y) = f x * g y - g x * f y)
  (h2 : f (-2) = f 1)
  (h3 : f 1 ≠ 0) :
  g 1 + g (-1) = -1 :=
sorry

end g_sum_eq_neg_one_l99_99079


namespace length_of_bridge_l99_99844

-- Define the conditions
def length_of_train : ℝ := 750
def speed_of_train_kmh : ℝ := 120
def crossing_time : ℝ := 45
def wind_resistance_factor : ℝ := 0.10

-- Define the conversion from km/hr to m/s
def kmh_to_ms (v : ℝ) : ℝ := v * 0.27778

-- Define the actual speed considering wind resistance
def actual_speed_ms (v : ℝ) (resistance : ℝ) : ℝ := (kmh_to_ms v) * (1 - resistance)

-- Define the total distance covered
def total_distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

-- Theorem: Length of the bridge
theorem length_of_bridge : total_distance (actual_speed_ms speed_of_train_kmh wind_resistance_factor) crossing_time - length_of_train = 600 := by
  sorry

end length_of_bridge_l99_99844


namespace part1_part2_l99_99246

noncomputable def f (x : ℝ) : ℝ := Real.exp x - x^2 - 1

theorem part1 (x : ℝ) : f x ≥ -x^2 + x := sorry

theorem part2 (k : ℝ) : (∀ x > 0, f x ≥ k * x) → k ≤ Real.exp 1 - 2 := sorry

end part1_part2_l99_99246


namespace no_lattice_points_on_hyperbola_l99_99314

theorem no_lattice_points_on_hyperbola : ∀ x y : ℤ, x^2 - y^2 ≠ 2022 :=
by
  intro x y
  -- proof omitted
  sorry

end no_lattice_points_on_hyperbola_l99_99314


namespace lizette_has_813_stamps_l99_99098

def minervas_stamps : ℕ := 688
def additional_stamps : ℕ := 125
def lizettes_stamps : ℕ := minervas_stamps + additional_stamps

theorem lizette_has_813_stamps : lizettes_stamps = 813 := by
  sorry

end lizette_has_813_stamps_l99_99098


namespace problem_statement_l99_99240

theorem problem_statement (x : ℝ) (h : x - 1/x = 5) : x^4 - (1 / x)^4 = 527 :=
sorry

end problem_statement_l99_99240


namespace sandwich_cost_90_cents_l99_99777

def sandwich_cost (bread_cost ham_cost cheese_cost : ℕ) : ℕ :=
  2 * bread_cost + ham_cost + cheese_cost

theorem sandwich_cost_90_cents :
  sandwich_cost 15 25 35 = 90 :=
by
  -- Proof goes here
  sorry

end sandwich_cost_90_cents_l99_99777


namespace anne_bob_total_difference_l99_99813

-- Define specific values as constants
def original_price : ℝ := 120.00
def discount_rate : ℝ := 0.25
def sales_tax_rate : ℝ := 0.08

-- Define the calculations according to Anne's method
def anne_total : ℝ := (original_price * (1 + sales_tax_rate)) * (1 - discount_rate)

-- Define the calculations according to Bob's method
def bob_total : ℝ := (original_price * (1 - discount_rate)) * (1 + sales_tax_rate)

-- State the theorem that the difference between Anne's and Bob's totals is zero
theorem anne_bob_total_difference : anne_total - bob_total = 0 :=
by sorry  -- Proof not required

end anne_bob_total_difference_l99_99813


namespace infinite_integer_solutions_l99_99931

variable (x : ℤ)

theorem infinite_integer_solutions (x : ℤ) : 
  ∃ (k : ℤ), ∀ n : ℤ, n > 2 → k = n :=
by {
  sorry
}

end infinite_integer_solutions_l99_99931


namespace fine_per_day_of_absence_l99_99366

theorem fine_per_day_of_absence :
  ∃ x: ℝ, ∀ (total_days work_wage total_received_days absent_days: ℝ),
  total_days = 30 →
  work_wage = 10 →
  total_received_days = 216 →
  absent_days = 7 →
  (total_days - absent_days) * work_wage - (absent_days * x) = total_received_days :=
sorry

end fine_per_day_of_absence_l99_99366


namespace sufficient_condition_for_perpendicular_l99_99268

variables {Plane : Type} {Line : Type} 
variables (α β γ : Plane) (m n : Line)

-- Definitions based on conditions
variables (perpendicular : Plane → Plane → Prop)
variables (perpendicular_line : Line → Plane → Prop)
variables (intersection : Plane → Plane → Line)

-- Conditions from option D
variable (h1 : perpendicular_line n α)
variable (h2 : perpendicular_line n β)
variable (h3 : perpendicular_line m α)

-- Statement to prove
theorem sufficient_condition_for_perpendicular (h1 : perpendicular_line n α)
  (h2 : perpendicular_line n β) (h3 : perpendicular_line m α) : 
  perpendicular_line m β := 
sorry

end sufficient_condition_for_perpendicular_l99_99268


namespace area_of_triangle_PF1F2_l99_99779

noncomputable def ellipse := {P : ℝ × ℝ // (4 * P.1^2) / 49 + (P.2^2) / 6 = 1}

noncomputable def area_triangle (P F1 F2 : ℝ × ℝ) :=
  1 / 2 * abs ((F1.1 - P.1) * (F2.2 - P.2) - (F1.2 - P.2) * (F2.1 - P.1))

theorem area_of_triangle_PF1F2 :
  ∀ (F1 F2 : ℝ × ℝ) (P : ellipse), 
    (dist P.1 F1 = 4) →
    (dist P.1 F2 = 3) →
    (dist F1 F2 = 5) →
    area_triangle P.1 F1 F2 = 6 :=
by sorry

end area_of_triangle_PF1F2_l99_99779


namespace min_groups_required_l99_99647

-- Define the conditions
def total_children : ℕ := 30
def max_children_per_group : ℕ := 12
def largest_divisor (n : ℕ) (d : ℕ) : Prop := d ∣ n ∧ d ≤ max_children_per_group

-- Define the property that we are interested in: the minimum number of groups required
def min_num_groups (total : ℕ) (group_size : ℕ) : ℕ := total / group_size

-- Prove the minimum number of groups is 3 given the conditions
theorem min_groups_required : ∃ d, largest_divisor total_children d ∧ min_num_groups total_children d = 3 :=
sorry

end min_groups_required_l99_99647


namespace ticket_distribution_count_l99_99675

theorem ticket_distribution_count :
  ∃ (count : ℕ), count = 96 ∧
    (∃ (distributions : Finset (list (Fin 4))),
      ∀ (d : list (Fin 4)), d ∈ distributions →
        (length d = 5 ∧
         (∀ (p : Fin 4), (countp (λ x, x = p) d) ≥ 1) ∧
         (∀ (p : Fin 4), (countp (λ x, x = p) d) ≤ 2) ∧
         (∀ (p : Fin 4), (countp (λ x, x = p) d = 2 → ∃ (i j : ℕ), (abs (i - j) = 1))))) :=
begin
  sorry -- Proof goes here
end

end ticket_distribution_count_l99_99675


namespace shipCargoCalculation_l99_99205

def initialCargo : Int := 5973
def cargoLoadedInBahamas : Int := 8723
def totalCargo (initial : Int) (loaded : Int) : Int := initial + loaded

theorem shipCargoCalculation : totalCargo initialCargo cargoLoadedInBahamas = 14696 := by
  sorry

end shipCargoCalculation_l99_99205


namespace turnip_bag_weight_l99_99987

/-- Given six bags with weights 13, 15, 16, 17, 21, and 24 kg,
one bag contains turnips, and the others contain either onions or carrots.
The total weight of the carrots equals twice the total weight of the onions.
Prove that the bag containing turnips can weigh either 13 kg or 16 kg. -/
theorem turnip_bag_weight (ws : list ℕ) (T : ℕ) (O C : ℕ) (h_ws : ws = [13, 15, 16, 17, 21, 24])
  (h_sum : ws.sum = 106) (h_co : C = 2 * O) (h_weight : C + O = 106 - T) :
  T = 13 ∨ T = 16 :=
sorry

end turnip_bag_weight_l99_99987


namespace tangent_line_parabola_l99_99307

theorem tangent_line_parabola (d : ℝ) : 
    (∀ y : ℝ, (-4)^2 - 4 * (y^2 - 4 * y + 4 * d) = 0) ↔ d = 1 :=
by
    sorry

end tangent_line_parabola_l99_99307


namespace sum_of_eight_numbers_l99_99744

theorem sum_of_eight_numbers (nums : List ℝ) (h_len : nums.length = 8) (h_avg : (nums.sum / 8) = 5.5) : nums.sum = 44 :=
by
  sorry

end sum_of_eight_numbers_l99_99744


namespace area_of_wrapping_paper_l99_99008

theorem area_of_wrapping_paper (l w h: ℝ) (l_pos: 0 < l) (w_pos: 0 < w) (h_pos: 0 < h) :
  ∃ s: ℝ, s = l + w ∧ s^2 = (l + w)^2 :=
by 
  sorry

end area_of_wrapping_paper_l99_99008


namespace smallest_n_for_terminating_decimal_l99_99150

theorem smallest_n_for_terminating_decimal :
  ∃ n : ℕ, n > 0 ∧ (∀ m : ℕ, m > 0 ∧ (∀ p: ℕ, (p ∣ (m + 150) → (p = 2 ∨ p = 5)) → m ≥ n)) ∧
  (∃ two_five_factors : ℕ, (two_five_factors = 5 ^ 3 * 2 ^ 3) ∧ (n + 150 = two_five_factors)) := 
by 
   exact ⟨50, by sorry⟩

end smallest_n_for_terminating_decimal_l99_99150


namespace estimate_students_less_than_2_hours_probability_one_male_one_female_l99_99815

-- Definitions from the conditions
def total_students_surveyed : ℕ := 40
def total_grade_ninth_students : ℕ := 400
def freq_0_1 : ℕ := 8
def freq_1_2 : ℕ := 20
def freq_2_3 : ℕ := 7
def freq_3_4 : ℕ := 5
def male_students_at_least_3_hours : ℕ := 2
def female_students_at_least_3_hours : ℕ := 3

-- Question 1 proof statement
theorem estimate_students_less_than_2_hours :
  total_grade_ninth_students * (freq_0_1 + freq_1_2) / total_students_surveyed = 280 :=
by sorry

-- Question 2 proof statement
theorem probability_one_male_one_female :
  (male_students_at_least_3_hours * female_students_at_least_3_hours) / (Nat.choose 5 2) = (3 / 5) :=
by sorry

end estimate_students_less_than_2_hours_probability_one_male_one_female_l99_99815


namespace bells_ring_together_l99_99363

open Nat

theorem bells_ring_together :
  let library_interval := 18
  let fire_station_interval := 24
  let hospital_interval := 30
  let start_time := 0
  let next_ring_time := Nat.lcm (Nat.lcm library_interval fire_station_interval) hospital_interval
  let total_minutes_in_an_hour := 60
  next_ring_time / total_minutes_in_an_hour = 6 :=
by
  let library_interval := 18
  let fire_station_interval := 24
  let hospital_interval := 30
  let start_time := 0
  let next_ring_time := Nat.lcm (Nat.lcm library_interval fire_station_interval) hospital_interval
  let total_minutes_in_an_hour := 60
  have h_next_ring_time : next_ring_time = 360 := by
    sorry
  have h_hours : next_ring_time / total_minutes_in_an_hour = 6 := by
    sorry
  exact h_hours

end bells_ring_together_l99_99363


namespace smallest_n_for_terminating_decimal_l99_99143

def is_terminating_decimal (n d : ℕ) : Prop :=
  ∀ (m : ℕ), d = 2^m ∨ d = 5^m ∨ d = (2^m) * (5 : ℕ) ∨ d = (5^m) * (2 : ℕ)
  
theorem smallest_n_for_terminating_decimal :
  ∃ n : ℕ, 0 < n ∧ is_terminating_decimal n (n + 150) ∧ (∀ m: ℕ, (is_terminating_decimal m (m + 150) ∧ 0 < m) → n ≤ m) :=
sorry

end smallest_n_for_terminating_decimal_l99_99143


namespace part1_part2_l99_99271

variables (a x : ℝ)

def p (a x : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0
def q (x : ℝ) : Prop := (|x - 1| ≤ 2) ∧ ((x + 3) / (x - 2) ≥ 0)

-- Part 1
theorem part1 (h_a : a = 1) (h_p : p a x) (h_q : q x) : 2 < x ∧ x < 3 := sorry

-- Part 2
theorem part2 (h_suff : ∀ x, q x → p a x) : 1 < a ∧ a ≤ 2 := sorry

end part1_part2_l99_99271


namespace medium_stores_to_select_l99_99083

-- Definitions based on conditions in a)
def total_stores := 1500
def ratio_large := 1
def ratio_medium := 5
def ratio_small := 9
def sample_size := 30
def medium_proportion := ratio_medium / (ratio_large + ratio_medium + ratio_small)

-- Main theorem to prove
theorem medium_stores_to_select : (sample_size * medium_proportion) = 10 :=
by sorry

end medium_stores_to_select_l99_99083


namespace rectangular_prism_inequality_l99_99273

variable {a b c l : ℝ}

theorem rectangular_prism_inequality (h_diag : l^2 = a^2 + b^2 + c^2) :
  (l^4 - a^4) * (l^4 - b^4) * (l^4 - c^4) ≥ 512 * a^4 * b^4 * c^4 := 
sorry

end rectangular_prism_inequality_l99_99273


namespace resulting_polygon_sides_l99_99218

theorem resulting_polygon_sides 
    (triangle_sides : ℕ := 3) 
    (square_sides : ℕ := 4) 
    (pentagon_sides : ℕ := 5) 
    (heptagon_sides : ℕ := 7) 
    (hexagon_sides : ℕ := 6) 
    (octagon_sides : ℕ := 8) 
    (shared_sides : ℕ := 1) :
    (2 * shared_sides + 4 * (shared_sides + 1)) = 16 := by 
  sorry

end resulting_polygon_sides_l99_99218


namespace triangle_angle_B_l99_99896

theorem triangle_angle_B {A B C : ℝ} (h1 : A = 60) (h2 : B = 2 * C) (h3 : A + B + C = 180) : B = 80 :=
sorry

end triangle_angle_B_l99_99896


namespace wheat_distribution_l99_99789

def mill1_rate := 19 / 3 -- quintals per hour
def mill2_rate := 32 / 5 -- quintals per hour
def mill3_rate := 5     -- quintals per hour

def total_wheat := 1330 -- total wheat in quintals

theorem wheat_distribution :
    ∃ (x1 x2 x3 : ℚ), 
    x1 = 475 ∧ x2 = 480 ∧ x3 = 375 ∧ 
    x1 / mill1_rate = x2 / mill2_rate ∧ x2 / mill2_rate = x3 / mill3_rate ∧ 
    x1 + x2 + x3 = total_wheat :=
by {
  sorry
}

end wheat_distribution_l99_99789


namespace comprehensive_score_correct_l99_99007

def comprehensive_score
  (study_score hygiene_score discipline_score participation_score : ℕ)
  (study_weight hygiene_weight discipline_weight participation_weight : ℚ) : ℚ :=
  study_score * study_weight +
  hygiene_score * hygiene_weight +
  discipline_score * discipline_weight +
  participation_score * participation_weight

theorem comprehensive_score_correct :
  let study_score := 80
  let hygiene_score := 90
  let discipline_score := 84
  let participation_score := 70
  let study_weight := 0.4
  let hygiene_weight := 0.25
  let discipline_weight := 0.25
  let participation_weight := 0.1
  comprehensive_score study_score hygiene_score discipline_score participation_score
                      study_weight hygiene_weight discipline_weight participation_weight
  = 82.5 :=
by 
  sorry

#eval comprehensive_score 80 90 84 70 0.4 0.25 0.25 0.1  -- output should be 82.5

end comprehensive_score_correct_l99_99007


namespace power_modulo_calculation_l99_99051

open Nat

theorem power_modulo_calculation :
  let λ500 := 100
  let λ100 := 20
  (5^5 : ℕ) ≡ 25 [MOD 100]
  (125^5 : ℕ) ≡ 125 [MOD 500]
  (5^{5^{5^5}} : ℕ) % 500 = 125 :=
by
  let λ500 := 100
  let λ100 := 20
  have h1 : (5^5 : ℕ) ≡ 25 [MOD 100] := by sorry
  have h2 : (125^5 : ℕ) ≡ 125 [MOD 500] := by sorry
  sorry

end power_modulo_calculation_l99_99051


namespace triangle_cosine_identity_l99_99284

open Real

variables {A B C a b c : ℝ}

theorem triangle_cosine_identity (h : b = (a + c) / 2) : cos (A - C) + 4 * cos B = 3 :=
sorry

end triangle_cosine_identity_l99_99284


namespace largest_angle_of_consecutive_integers_in_hexagon_l99_99482

theorem largest_angle_of_consecutive_integers_in_hexagon : 
  ∀ (a : ℕ), 
    (a - 2) + (a - 1) + a + (a + 1) + (a + 2) + (a + 3) = 720 → 
    a + 3 = 122.5 :=
by sorry

end largest_angle_of_consecutive_integers_in_hexagon_l99_99482


namespace coin_probability_l99_99465

theorem coin_probability :
  let PA := 3/4
  let PB := 1/2
  let PC := 1/4
  (PA * PB * (1 - PC)) = 9/32 :=
by
  sorry

end coin_probability_l99_99465


namespace arithmetic_sequence_ratio_l99_99781

variable {α : Type}
variable [LinearOrderedField α]

def a1 (a_1 : α) : Prop := a_1 ≠ 0 
def a2_eq_3a1 (a_1 a_2 : α) : Prop := a_2 = 3 * a_1 

noncomputable def common_difference (a_1 a_2 : α) : α :=
  a_2 - a_1

noncomputable def S (n : ℕ) (a_1 d : α) : α :=
  n * (2 * a_1 + (n - 1) * d) / 2

theorem arithmetic_sequence_ratio
  (a_1 a_2 : α)
  (h₀ : a1 a_1)
  (h₁ : a2_eq_3a1 a_1 a_2) :
  (S 10 a_1 (common_difference a_1 a_2)) / (S 5 a_1 (common_difference a_1 a_2)) = 4 := 
by
  sorry

end arithmetic_sequence_ratio_l99_99781


namespace smallest_positive_four_digit_multiple_of_18_l99_99691

-- Define the predicates for conditions
def four_digit_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

def multiple_of_18 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 18 * k

-- Define the main theorem
theorem smallest_positive_four_digit_multiple_of_18 : 
  ∃ n : ℕ, four_digit_number n ∧ multiple_of_18 n ∧ ∀ m : ℕ, four_digit_number m ∧ multiple_of_18 m → n ≤ m :=
begin
  use 1008,
  split,
  { -- proof that 1008 is a four-digit number
    split,
    { linarith, },
    { linarith, }
  },

  split,
  { -- proof that 1008 is a multiple of 18
    use 56,
    norm_num,
  },

  { -- proof that 1008 is the smallest such number
    intros m h1 h2,
    have h3 := Nat.le_of_lt,
    sorry, -- Detailed proof would go here
  }
end

end smallest_positive_four_digit_multiple_of_18_l99_99691


namespace angle_QPR_l99_99134

theorem angle_QPR (PQ QR PR RS : Real) (angle_PQR angle_PRS : Real) 
  (h1 : PQ = QR) (h2 : PR = RS) (h3 : angle_PQR = 50) (h4 : angle_PRS = 100) : 
  ∃ angle_QPR : Real, angle_QPR = 25 :=
by
  -- We are proving that angle_QPR is 25 given the conditions.
  sorry

end angle_QPR_l99_99134


namespace triangle_is_3_l99_99221

def base6_addition_valid (delta : ℕ) : Prop :=
  delta < 6 ∧ 
  2 + delta + delta + 4 < 6 ∧ -- No carry effect in the middle digits
  ((delta + 3) % 6 = 4) ∧
  ((5 + delta + (2 + delta + delta + 4) / 6) % 6 = 3) ∧
  ((4 + (5 + delta + (2 + delta + delta + 4) / 6) / 6) % 6 = 5)

theorem triangle_is_3 : ∃ (δ : ℕ), base6_addition_valid δ ∧ δ = 3 :=
by
  use 3
  sorry

end triangle_is_3_l99_99221


namespace largest_n_digit_number_divisible_by_89_l99_99139

theorem largest_n_digit_number_divisible_by_89 (n : ℕ) (h1 : n % 2 = 1) (h2 : 3 ≤ n ∧ n ≤ 7) :
  ∃ x, x = 9999951 ∧ (x % 89 = 0 ∧ (10 ^ (n-1) ≤ x ∧ x < 10 ^ n)) :=
by
  sorry

end largest_n_digit_number_divisible_by_89_l99_99139


namespace problem1_problem2_l99_99553

-- Define the function f(x) = |x + 2| + |x - 1|
def f (x : ℝ) : ℝ := |x + 2| + |x - 1|

-- 1. Prove the solution set of f(x) > 5 is {x | x < -3 or x > 2}
theorem problem1 : {x : ℝ | f x > 5} = {x : ℝ | x < -3 ∨ x > 2} :=
by
  sorry

-- 2. Prove that if f(x) ≥ a^2 - 2a always holds, then -1 ≤ a ≤ 3
theorem problem2 (a : ℝ) (h : ∀ x : ℝ, f x ≥ a^2 - 2 * a) : -1 ≤ a ∧ a ≤ 3 :=
by
  sorry

end problem1_problem2_l99_99553


namespace angle_QPR_l99_99135

theorem angle_QPR (PQ QR PR RS : ℝ)
    (hPQQR : PQ = QR) (hPRRS : PR = RS)
    (h_angle_PQR : ∠PQR = 50)
    (h_angle_PRS : ∠PRS = 100) :
    ∠QPR = 25 :=
begin
  sorry
end

end angle_QPR_l99_99135


namespace line_tangent_to_parabola_proof_l99_99299

noncomputable def line_tangent_to_parabola (d : ℝ) := (∀ x y : ℝ, y = 3 * x + d → y^2 = 12 * x) → d = 1

theorem line_tangent_to_parabola_proof (d : ℝ) (h : ∀ x y : ℝ, y = 3 * x + d → y^2 = 12 * x) : d = 1 :=
sorry

end line_tangent_to_parabola_proof_l99_99299


namespace three_digit_integers_with_product_36_l99_99736

/--
There are 21 distinct 3-digit integers such that the product of their digits equals 36, and each digit is between 1 and 9.
-/
theorem three_digit_integers_with_product_36 : 
  ∃ n : ℕ, digit_product_count 36 3 n ∧ n = 21 :=
sorry

end three_digit_integers_with_product_36_l99_99736


namespace scientists_nobel_greater_than_not_nobel_by_three_l99_99967

-- Definitions of the given conditions
def total_scientists := 50
def wolf_prize_laureates := 31
def nobel_prize_laureates := 25
def wolf_and_nobel_laureates := 14

-- Derived quantities
def no_wolf_prize := total_scientists - wolf_prize_laureates
def only_wolf_prize := wolf_prize_laureates - wolf_and_nobel_laureates
def only_nobel_prize := nobel_prize_laureates - wolf_and_nobel_laureates
def nobel_no_wolf := only_nobel_prize
def no_wolf_no_nobel := no_wolf_prize - nobel_no_wolf
def difference := nobel_no_wolf - no_wolf_no_nobel

-- The theorem to be proved
theorem scientists_nobel_greater_than_not_nobel_by_three :
  difference = 3 := 
sorry

end scientists_nobel_greater_than_not_nobel_by_three_l99_99967


namespace brett_blue_marbles_more_l99_99023

theorem brett_blue_marbles_more (r b : ℕ) (hr : r = 6) (hb : b = 5 * r) : b - r = 24 := by
  rw [hr, hb]
  norm_num
  sorry

end brett_blue_marbles_more_l99_99023


namespace three_digit_integers_with_product_36_l99_99735

-- Definition of the problem conditions
def is_three_digit_integer (n : Nat) : Prop :=
  100 ≤ n ∧ n < 1000

def digit_product_is_36 (n : Nat) : Prop :=
  let d1 := n / 100
  let d2 := (n / 10) % 10
  let d3 := n % 10
  1 ≤ d1 ∧ d1 ≤ 9 ∧ 1 ≤ d2 ∧ d2 ≤ 9 ∧ 1 ≤ d3 ∧ d3 ≤ 9 ∧ (d1 * d2 * d3 = 36)

-- The statement of the proof
theorem three_digit_integers_with_product_36 :
  {n : Nat | is_three_digit_integer n ∧ digit_product_is_36 n}.toFinset.card = 21 := 
by
  sorry

end three_digit_integers_with_product_36_l99_99735


namespace function_periodic_l99_99909

open Real

def periodic (f : ℝ → ℝ) := ∃ T > 0, ∀ x, f (x + T) = f x

theorem function_periodic (a : ℚ) (b d c : ℝ) (f : ℝ → ℝ) 
    (hf : ∀ x : ℝ, f (x + ↑a + b) - f (x + b) = c * (x + 2 * ↑a + ⌊x⌋ - 2 * ⌊x + ↑a⌋ - ⌊b⌋) + d) : 
    periodic f :=
sorry

end function_periodic_l99_99909


namespace convert_quadratic_l99_99851

theorem convert_quadratic :
  ∀ x : ℝ, (x^2 + 2*x + 4) = ((x + 1)^2 + 3) :=
by
  sorry

end convert_quadratic_l99_99851


namespace pieces_per_sister_l99_99549

-- Defining the initial conditions
def initial_cake_pieces : ℕ := 240
def percentage_eaten : ℕ := 60
def number_of_sisters : ℕ := 3

-- Defining the statements to be proved
theorem pieces_per_sister (initial_cake_pieces : ℕ) (percentage_eaten : ℕ) (number_of_sisters : ℕ) :
  let pieces_eaten := (percentage_eaten * initial_cake_pieces) / 100
  let remaining_pieces := initial_cake_pieces - pieces_eaten
  let pieces_per_sister := remaining_pieces / number_of_sisters
  pieces_per_sister = 32 :=
by 
  sorry

end pieces_per_sister_l99_99549


namespace cows_with_no_spot_l99_99919

theorem cows_with_no_spot (total_cows : ℕ) (percent_red_spot : ℚ) (percent_blue_spot : ℚ) :
  total_cows = 140 ∧ percent_red_spot = 0.40 ∧ percent_blue_spot = 0.25 → 
  ∃ (no_spot_cows : ℕ), no_spot_cows = 63 :=
by 
  sorry

end cows_with_no_spot_l99_99919


namespace number_of_archers_in_golden_armor_l99_99755

-- Define the problem context
structure Soldier where
  is_archer : Bool
  wears_golden_armor : Bool

def truth_teller (s : Soldier) (is_black_armor : Bool) : Bool :=
  if s.is_archer then s.wears_golden_armor = is_black_armor
  else s.wears_golden_armor ≠ is_black_armor

def response (s : Soldier) (question : String) (is_black_armor : Bool) : Bool :=
  match question with
  | "Are you wearing golden armor?" => if truth_teller s is_black_armor then s.wears_golden_armor else ¬s.wears_golden_armor
  | "Are you an archer?" => if truth_teller s is_black_armor then s.is_archer else ¬s.is_archer
  | "Is today Monday?" => if truth_teller s is_black_armor then True else False -- An assumption that today not being Monday means False
  | _ => False

-- Problem condition setup
def total_soldiers : Nat := 55
def golden_armor_yes : Nat := 44
def archer_yes : Nat := 33
def monday_yes : Nat := 22

-- Define the main theorem
theorem number_of_archers_in_golden_armor :
  ∃ l : List Soldier,
    l.length = total_soldiers ∧
    l.countp (λ s => response s "Are you wearing golden armor?" True) = golden_armor_yes ∧
    l.countp (λ s => response s "Are you an archer?" True) = archer_yes ∧
    l.countp (λ s => response s "Is today Monday?" True) = monday_yes ∧
    l.countp (λ s => s.is_archer ∧ s.wears_golden_armor) = 22 :=
sorry

end number_of_archers_in_golden_armor_l99_99755


namespace min_selling_price_is_400_l99_99375

-- Definitions for the problem conditions
def total_products := 20
def average_price := 1200
def less_than_1000_count := 10
def price_of_most_expensive := 11000
def total_retail_price := total_products * average_price

-- The theorem to state the problem condition and the expected result
theorem min_selling_price_is_400 (x : ℕ) :
  -- Condition 1: Total retail price
  total_retail_price =
  -- 10 products sell for x dollars
  (10 * x) +
  -- 9 products sell for 1000 dollars
  (9 * 1000) +
  -- 1 product sells for the maximum price 11000
  price_of_most_expensive → 
  -- Conclusion: The minimum price x is 400
  x = 400 :=
by
  sorry

end min_selling_price_is_400_l99_99375


namespace pen_and_notebook_cost_l99_99836

theorem pen_and_notebook_cost (pen_cost : ℝ) (notebook_cost : ℝ) 
  (h1 : pen_cost = 4.5) 
  (h2 : pen_cost = notebook_cost + 1.8) : 
  pen_cost + notebook_cost = 7.2 := 
  by
    sorry

end pen_and_notebook_cost_l99_99836


namespace smallest_n_terminating_decimal_l99_99166

theorem smallest_n_terminating_decimal :
  ∃ n : ℕ, (∀ m : ℕ, (∃ k : ℕ, m + 150 = 2^k * (5^0))
  → (m ≥ n)) ∧ (n > 0) ∧ (∃ k : ℕ, n + 150 = 2^k * (5^0)) := 
sorry

end smallest_n_terminating_decimal_l99_99166


namespace find_number_l99_99888

theorem find_number (X : ℝ) (h : 30 = 0.50 * X + 10) : X = 40 :=
by
  sorry

end find_number_l99_99888


namespace odd_multiple_of_9_implies_multiple_of_3_l99_99004

theorem odd_multiple_of_9_implies_multiple_of_3 :
  ∀ (S : ℤ), (∀ (n : ℤ), 9 * n = S → ∃ (m : ℤ), 3 * m = S) ∧ (S % 2 ≠ 0) → (∃ (m : ℤ), 3 * m = S) :=
by
  sorry

end odd_multiple_of_9_implies_multiple_of_3_l99_99004


namespace farmer_ducks_sold_l99_99198

theorem farmer_ducks_sold (D : ℕ) (earnings : ℕ) :
  (earnings = (10 * D) + (5 * 8)) →
  ((earnings / 2) * 2 = 60) →
  D = 2 := by
  sorry

end farmer_ducks_sold_l99_99198


namespace sunday_saturday_ratio_is_two_to_one_l99_99857

-- Define the conditions as given in the problem
def total_pages : ℕ := 360
def saturday_morning_read : ℕ := 40
def saturday_night_read : ℕ := 10
def remaining_pages : ℕ := 210

-- Define Ethan's total pages read so far
def total_read : ℕ := total_pages - remaining_pages

-- Define pages read on Saturday
def saturday_total_read : ℕ := saturday_morning_read + saturday_night_read

-- Define pages read on Sunday
def sunday_total_read : ℕ := total_read - saturday_total_read

-- Define the ratio of pages read on Sunday to pages read on Saturday
def sunday_to_saturday_ratio : ℕ := sunday_total_read / saturday_total_read

-- Theorem statement: ratio of pages read on Sunday to pages read on Saturday is 2:1
theorem sunday_saturday_ratio_is_two_to_one : sunday_to_saturday_ratio = 2 :=
by
  -- This part should contain the detailed proof
  sorry

end sunday_saturday_ratio_is_two_to_one_l99_99857


namespace logan_money_left_l99_99785

-- Defining the given conditions
def income : ℕ := 65000
def rent_expense : ℕ := 20000
def groceries_expense : ℕ := 5000
def gas_expense : ℕ := 8000
def additional_income_needed : ℕ := 10000

-- Calculating total expenses
def total_expense : ℕ := rent_expense + groceries_expense + gas_expense

-- Desired income
def desired_income : ℕ := income + additional_income_needed

-- The theorem to prove
theorem logan_money_left : (desired_income - total_expense) = 42000 :=
by
  -- A placeholder for the proof
  sorry

end logan_money_left_l99_99785


namespace sqrt_x_minus_1_meaningful_l99_99423

theorem sqrt_x_minus_1_meaningful (x : ℝ) : (∃ y : ℝ, y = real.sqrt (x - 1)) → x ≥ 1 := by
  intros h
  cases h with y hy
  rw hy
  have := real.sqrt_nonneg (x - 1)
  sorry

end sqrt_x_minus_1_meaningful_l99_99423


namespace four_points_no_obtuse_triangle_l99_99710

noncomputable def probability_no_obtuse_triangle : ℝ :=
1 / 64

theorem four_points_no_obtuse_triangle (A B C D : circle) :
  (∀ (P Q : circle) (PQ_angle : ℝ), PQ_angle < π/2) → 
  probability_no_obtuse_triangle = 1 / 64 :=
sorry

end four_points_no_obtuse_triangle_l99_99710


namespace young_li_age_l99_99319

theorem young_li_age (x : ℝ) (old_li_age : ℝ) 
  (h1 : old_li_age = 2.5 * x)  
  (h2 : old_li_age + 10 = 2 * (x + 10)) : 
  x = 20 := 
by
  sorry

end young_li_age_l99_99319


namespace milk_added_is_10_l99_99256

variable (x y : ℚ) -- x and y are rational numbers representing initial amounts and added milk respectively.

-- Conditions
def initial_ratio_milk_water (x : ℚ) := (4 * x, 3 * x)
def new_ratio_milk_water (x y : ℚ) := (4 * x + y, 3 * x)
def capacity_of_can : ℚ := 30
def new_ratio : ℚ := 5 / 2

-- Proof Statement
def proof_milk_added (x y : ℚ) : Prop :=
  (initial_ratio_milk_water x).1 + (initial_ratio_milk_water x).2 = capacity_of_can ∧
  new_ratio_milk_water x y = new_ratio ∧
  x = 20 / 7 ∧
  y = 10

theorem milk_added_is_10 : ∃ (x y : ℚ), proof_milk_added x y := by
  sorry

end milk_added_is_10_l99_99256


namespace solution_of_fractional_equation_l99_99747

theorem solution_of_fractional_equation :
  (∃ x, x ≠ 3 ∧ (x / (x - 3) - 2 = (m - 1) / (x - 3))) → m = 4 := by
  sorry

end solution_of_fractional_equation_l99_99747


namespace paths_A_to_D_through_B_and_C_l99_99571

-- Define points and paths in a grid
structure Point where
  x : ℕ
  y : ℕ

def A : Point := ⟨0, 0⟩
def B : Point := ⟨2, 3⟩
def C : Point := ⟨6, 4⟩
def D : Point := ⟨9, 6⟩

-- Calculate binomial coefficient
def choose (n k : ℕ) : ℕ :=
  if k > n then 0
  else Nat.div (Nat.factorial n) ((Nat.factorial k) * (Nat.factorial (n - k)))

-- Number of paths from one point to another in a grid
def numPaths (p1 p2 : Point) : ℕ :=
  let stepsRight := p2.x - p1.x
  let stepsDown := p2.y - p1.y
  choose (stepsRight + stepsDown) stepsRight

theorem paths_A_to_D_through_B_and_C : numPaths A B * numPaths B C * numPaths C D = 500 := by
  -- Using the conditions provided:
  -- numPaths A B = choose 5 2 = 10
  -- numPaths B C = choose 5 1 = 5
  -- numPaths C D = choose 5 2 = 10
  -- Therefore, numPaths A B * numPaths B C * numPaths C D = 10 * 5 * 10 = 500
  sorry

end paths_A_to_D_through_B_and_C_l99_99571


namespace trigonometric_identity_l99_99237

theorem trigonometric_identity (α : Real) (h : Real.tan α = 2 * Real.tan (Real.pi / 5)) :
  (Real.cos (α - 3 * Real.pi / 10) / Real.sin (α - Real.pi / 5) = 3) :=
sorry

end trigonometric_identity_l99_99237


namespace has_only_one_minimum_point_and_no_maximum_point_l99_99598

noncomputable def f (x : ℝ) : ℝ := 3 * x^4 - 4 * x^3

theorem has_only_one_minimum_point_and_no_maximum_point :
  ∃! c : ℝ, (deriv f c = 0 ∧ ∀ x < c, deriv f x < 0 ∧ ∀ x > c, deriv f x > 0) ∧
  ∀ x, f x ≥ f c ∧ (∀ x, deriv f x > 0 ∨ deriv f x < 0) := sorry

end has_only_one_minimum_point_and_no_maximum_point_l99_99598


namespace all_roots_of_P_are_roots_of_R_R_has_no_multiple_roots_l99_99645

open Polynomial

variables {R : Type*} [CommRing R]

-- Define the polynomial P(x)
variable (P : R[X])

-- Define the gcd of P(x) and its derivative P'(x)
noncomputable def Q := gcd P P.derivative

-- Define the quotient polynomial R(x)
noncomputable def R := P / Q

-- Prove that all roots of P are roots of R
theorem all_roots_of_P_are_roots_of_R :
  ∀ (r : R), is_root P r → is_root (R P) r :=
sorry

-- Prove that R does not have multiple roots
theorem R_has_no_multiple_roots :
  ∀ (r : R) (k : ℕ), multiplicity r (R P) = k → k ≤ 1 :=
sorry

end all_roots_of_P_are_roots_of_R_R_has_no_multiple_roots_l99_99645


namespace number_of_teams_l99_99829

theorem number_of_teams (x : ℕ) (h : (x * (x - 1)) / 2 = 28) : x = 8 :=
sorry

end number_of_teams_l99_99829


namespace julia_more_kids_on_Monday_l99_99266

def kids_played_on_Tuesday : Nat := 14
def kids_played_on_Monday : Nat := 22

theorem julia_more_kids_on_Monday : kids_played_on_Monday - kids_played_on_Tuesday = 8 :=
by {
  sorry
}

end julia_more_kids_on_Monday_l99_99266


namespace color_of_241st_marble_l99_99516

def sequence_color (n : ℕ) : String :=
  if n % 14 < 6 then "blue"
  else if n % 14 < 11 then "red"
  else "green"

theorem color_of_241st_marble : sequence_color 240 = "blue" :=
  by
  sorry

end color_of_241st_marble_l99_99516


namespace ratio_of_intercepts_l99_99327

variable {c : ℝ} (non_zero_c : c ≠ 0) (u v : ℝ)
-- Condition: The first line, slope 8, y-intercept c, x-intercept (u, 0)
variable (h_u : u = -c / 8)
-- Condition: The second line, slope 4, y-intercept c, x-intercept (v, 0)
variable (h_v : v = -c / 4)

theorem ratio_of_intercepts (non_zero_c : c ≠ 0)
    (h_u : u = -c / 8) (h_v : v = -c / 4) : u / v = 1 / 2 :=
by
  sorry

end ratio_of_intercepts_l99_99327


namespace avg_salary_increase_l99_99468

theorem avg_salary_increase (A1 : ℝ) (M : ℝ) (n : ℕ) (N : ℕ) 
  (h1 : n = 20) (h2 : A1 = 1500) (h3 : M = 4650) (h4 : N = n + 1) :
  (20 * A1 + M) / N - A1 = 150 :=
by
  -- proof goes here
  sorry

end avg_salary_increase_l99_99468


namespace cows_with_no_spots_l99_99918

-- Definitions of conditions
def total_cows : Nat := 140
def cows_with_red_spot : Nat := (40 * total_cows) / 100
def cows_without_red_spot : Nat := total_cows - cows_with_red_spot
def cows_with_blue_spot : Nat := (25 * cows_without_red_spot) / 100

-- Theorem statement asserting the number of cows with no spots
theorem cows_with_no_spots : (total_cows - cows_with_red_spot - cows_with_blue_spot) = 63 := by
  -- Proof would go here
  sorry

end cows_with_no_spots_l99_99918


namespace inequality_solution_set_empty_range_l99_99080

theorem inequality_solution_set_empty_range (m : ℝ) :
  (∀ x : ℝ, mx^2 - mx - 1 < 0) ↔ -4 < m ∧ m ≤ 0 :=
by
  sorry

end inequality_solution_set_empty_range_l99_99080


namespace find_p_l99_99119

noncomputable def p (x : ℝ) : ℝ := (9/5) * (x^2 - 4)

theorem find_p :
  ∃ (a : ℝ), (∀ x, p(x) = a * (x + 2) * (x - 2)) ∧ p(-3) = 9 :=
by
  use 9/5
  split
  { intro x
    rw p
    ring }
  { rw p
    norm_num }
  sorry

end find_p_l99_99119


namespace problem_statement_l99_99657

theorem problem_statement (x : ℤ) (h : Even (3 * x + 1)) : Odd (7 * x + 4) :=
  sorry

end problem_statement_l99_99657


namespace minimum_value_l99_99384

noncomputable def minSum (a b c : ℝ) : ℝ :=
  a / (3 * b) + b / (6 * c) + c / (9 * a)

theorem minimum_value (a b c : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c) :
  ∃ x : ℝ, (∀ a b c : ℝ, 0 < a → 0 < b → 0 < c → minSum a b c ≥ x) ∧ x = 3 / Real.cbrt 162 :=
sorry

end minimum_value_l99_99384


namespace smallest_positive_integer_for_terminating_decimal_l99_99177

theorem smallest_positive_integer_for_terminating_decimal: ∃ n: ℕ, (n > 0) ∧ (∀ p : ℕ, (p ∣ (n + 150)) → (p=1 ∨ p=2 ∨ p=4 ∨ p=5 ∨ p=8 ∨ p=10 ∨ p=16 ∨ p=20 ∨ p=25 ∨ p=32 ∨ p=40 ∨ p=50 ∨ p=64 ∨ p=80 ∨ p=100 ∨ p=125 ∨ p=128 ∨ p=160)) ∧ n = 10 :=
by
  sorry

end smallest_positive_integer_for_terminating_decimal_l99_99177


namespace three_digit_repeated_digits_percentage_l99_99740

noncomputable def percentage_repeated_digits : ℝ :=
  let total_numbers := 900
  let non_repeated := 9 * 9 * 8
  let repeated := total_numbers - non_repeated
  (repeated / total_numbers) * 100

theorem three_digit_repeated_digits_percentage :
  percentage_repeated_digits = 28.0 := by
  sorry

end three_digit_repeated_digits_percentage_l99_99740


namespace turnips_bag_l99_99995

theorem turnips_bag (weights : List ℕ) (h : weights = [13, 15, 16, 17, 21, 24])
  (turnips: ℕ)
  (is_turnip : turnips ∈ weights)
  (o c : ℕ)
  (h1 : o + c = 106 - turnips)
  (h2 : c = 2 * o) :
  turnips = 13 ∨ turnips = 16 := by
  sorry

end turnips_bag_l99_99995


namespace simple_interest_rate_l99_99001

theorem simple_interest_rate (P : ℝ) (T : ℝ) (R : ℝ) (SI : ℝ) (h1 : T = 15) (h2 : SI = 3 * P) (h3 : SI = P * R * T / 100) : R = 20 :=
by 
  sorry

end simple_interest_rate_l99_99001


namespace segment_length_reflection_l99_99816

theorem segment_length_reflection (F : ℝ × ℝ) (F' : ℝ × ℝ)
  (hF : F = (-4, -2)) (hF' : F' = (4, -2)) :
  dist F F' = 8 :=
by
  sorry

end segment_length_reflection_l99_99816


namespace smallest_n_for_terminating_decimal_l99_99148

theorem smallest_n_for_terminating_decimal :
  ∃ n : ℕ, n > 0 ∧ (∀ m : ℕ, m > 0 ∧ (∀ p: ℕ, (p ∣ (m + 150) → (p = 2 ∨ p = 5)) → m ≥ n)) ∧
  (∃ two_five_factors : ℕ, (two_five_factors = 5 ^ 3 * 2 ^ 3) ∧ (n + 150 = two_five_factors)) := 
by 
   exact ⟨50, by sorry⟩

end smallest_n_for_terminating_decimal_l99_99148


namespace seeds_per_plant_l99_99776

theorem seeds_per_plant :
  let trees := 2
  let plants_per_tree := 20
  let total_plants := trees * plants_per_tree
  let planted_trees := 24
  let planting_fraction := 0.60
  exists S : ℝ, planting_fraction * (total_plants * S) = planted_trees ∧ S = 1 :=
by
  sorry

end seeds_per_plant_l99_99776


namespace decimal_equivalence_l99_99807

theorem decimal_equivalence : 4 + 3 / 10 + 9 / 1000 = 4.309 := 
by
  sorry

end decimal_equivalence_l99_99807


namespace smallest_positive_integer_for_terminating_decimal_l99_99162

theorem smallest_positive_integer_for_terminating_decimal (n : ℕ) (h1 : n > 0) (h2 : (∀ m : ℕ, (m > n + 150) -> (m % (n + 150)) ∉ {2, 5})) :
  n = 50 :=
sorry

end smallest_positive_integer_for_terminating_decimal_l99_99162


namespace determine_numbers_l99_99597

theorem determine_numbers (n : ℕ) (m : ℕ) (x y z u v : ℕ) (h₁ : 10000 <= n ∧ n < 100000)
(h₂ : n = 10000 * x + 1000 * y + 100 * z + 10 * u + v)
(h₃ : m = 1000 * x + 100 * y + 10 * u + v)
(h₄ : x ≠ 0)
(h₅ : n % m = 0) :
∃ a : ℕ, (10 <= a ∧ a <= 99 ∧ n = a * 1000) :=
sorry

end determine_numbers_l99_99597


namespace smallest_positive_integer_for_terminating_decimal_l99_99182

theorem smallest_positive_integer_for_terminating_decimal: ∃ n: ℕ, (n > 0) ∧ (∀ p : ℕ, (p ∣ (n + 150)) → (p=1 ∨ p=2 ∨ p=4 ∨ p=5 ∨ p=8 ∨ p=10 ∨ p=16 ∨ p=20 ∨ p=25 ∨ p=32 ∨ p=40 ∨ p=50 ∨ p=64 ∨ p=80 ∨ p=100 ∨ p=125 ∨ p=128 ∨ p=160)) ∧ n = 10 :=
by
  sorry

end smallest_positive_integer_for_terminating_decimal_l99_99182


namespace no_real_roots_of_quad_eq_l99_99867

theorem no_real_roots_of_quad_eq (k : ℝ) : ¬(k ≠ 0 ∧ ∃ x : ℝ, x^2 + k * x + 2 * k^2 = 0) :=
by
  sorry

end no_real_roots_of_quad_eq_l99_99867


namespace intersection_P_Q_l99_99913

def P : Set ℝ := {x | |x| > 1}
def Q : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}

theorem intersection_P_Q : P ∩ Q = {x | -2 ≤ x ∧ x < -1 ∨ 1 < x ∧ x ≤ 2} :=
by
  sorry

end intersection_P_Q_l99_99913


namespace option_a_correct_option_b_incorrect_option_c_incorrect_option_d_incorrect_l99_99819

theorem option_a_correct (a : ℝ) : 2 * a^2 - 3 * a^2 = - a^2 :=
by
  sorry

theorem option_b_incorrect : (-3)^2 ≠ 6 :=
by
  sorry

theorem option_c_incorrect (a : ℝ) : 6 * a^3 + 4 * a^4 ≠ 10 * a^7 :=
by
  sorry

theorem option_d_incorrect (a b : ℝ) : 3 * a^2 * b - 3 * b^2 * a ≠ 0 :=
by
  sorry

end option_a_correct_option_b_incorrect_option_c_incorrect_option_d_incorrect_l99_99819


namespace correct_calculation_l99_99187

theorem correct_calculation (x : ℕ) (h : x / 9 = 30) : x - 37 = 233 :=
by sorry

end correct_calculation_l99_99187


namespace percentage_loss_is_correct_l99_99980

-- Define the cost price and selling price
def cost_price : ℕ := 2000
def selling_price : ℕ := 1800

-- Define the calculation of loss and percentage loss
def loss (cp sp : ℕ) := cp - sp
def percentage_loss (loss cp : ℕ) := (loss * 100) / cp

-- The goal is to prove that the percentage loss is 10%
theorem percentage_loss_is_correct : percentage_loss (loss cost_price selling_price) cost_price = 10 := by
  sorry

end percentage_loss_is_correct_l99_99980


namespace floral_shop_bouquets_l99_99358

theorem floral_shop_bouquets (T : ℕ) 
  (h1 : 12 + T + T / 3 = 60) 
  (hT : T = 36) : T / 12 = 3 :=
by
  -- Proof steps go here
  sorry

end floral_shop_bouquets_l99_99358


namespace number_of_valid_subsets_l99_99509

theorem number_of_valid_subsets (n : ℕ) :
  let total      := 16^n
  let invalid1   := 3 * 12^n
  let invalid2   := 2 * 10^n
  let invalidAll := 8^n
  let valid      := total - invalid1 + invalid2 + 9^n - invalidAll
  valid = 16^n - 3 * 12^n + 2 * 10^n + 9^n - 8^n :=
by {
  -- Proof steps would go here
  sorry
}

end number_of_valid_subsets_l99_99509


namespace smallest_n_for_terminating_decimal_l99_99144

def is_terminating_decimal (n d : ℕ) : Prop :=
  ∀ (m : ℕ), d = 2^m ∨ d = 5^m ∨ d = (2^m) * (5 : ℕ) ∨ d = (5^m) * (2 : ℕ)
  
theorem smallest_n_for_terminating_decimal :
  ∃ n : ℕ, 0 < n ∧ is_terminating_decimal n (n + 150) ∧ (∀ m: ℕ, (is_terminating_decimal m (m + 150) ∧ 0 < m) → n ≤ m) :=
sorry

end smallest_n_for_terminating_decimal_l99_99144


namespace initial_big_bottles_l99_99521

theorem initial_big_bottles (B : ℝ)
  (initial_small : ℝ := 6000)
  (sold_small : ℝ := 0.11)
  (sold_big : ℝ := 0.12)
  (remaining_total : ℝ := 18540) :
  (initial_small * (1 - sold_small) + B * (1 - sold_big) = remaining_total) → B = 15000 :=
by
  intro h
  sorry

end initial_big_bottles_l99_99521


namespace remainder_mod_500_l99_99042

theorem remainder_mod_500 :
  ( 5^(5^(5^5)) ) % 500 = 125 :=
by
  -- proof goes here
  sorry

end remainder_mod_500_l99_99042


namespace probability_same_plane_l99_99721

-- Define the number of vertices in a cube
def num_vertices : ℕ := 8

-- Define the number of vertices to be selected
def selection : ℕ := 4

-- Define the total number of ways to select 4 vertices out of 8
def total_ways : ℕ := Nat.choose num_vertices selection

-- Define the number of favorable ways to have 4 vertices lie in the same plane
def favorable_ways : ℕ := 12

-- Define the probability that the 4 selected vertices lie in the same plane
def probability : ℚ := favorable_ways / total_ways

-- The statement we need to prove
theorem probability_same_plane : probability = 6 / 35 := by
  sorry

end probability_same_plane_l99_99721


namespace rate_downstream_l99_99652

-- Define the man's rate in still water
def rate_still_water : ℝ := 24.5

-- Define the rate of the current
def rate_current : ℝ := 7.5

-- Define the man's rate upstream (unused in the proof but given in the problem)
def rate_upstream : ℝ := 17.0

-- Prove that the man's rate when rowing downstream is as stated given the conditions
theorem rate_downstream : rate_still_water + rate_current = 32 := by
  simp [rate_still_water, rate_current]
  norm_num

end rate_downstream_l99_99652


namespace Ursula_hours_per_day_l99_99817

theorem Ursula_hours_per_day (hourly_wage : ℝ) (days_per_month : ℕ) (annual_salary : ℝ) (months_per_year : ℕ) :
  hourly_wage = 8.5 →
  days_per_month = 20 →
  annual_salary = 16320 →
  months_per_year = 12 →
  (annual_salary / months_per_year / days_per_month / hourly_wage) = 8 :=
by
  intros
  sorry

end Ursula_hours_per_day_l99_99817


namespace boundary_line_f_g_l99_99562

open Real

noncomputable def f (x : ℝ) : ℝ := x * log x

noncomputable def g (x : ℝ) : ℝ := 0.5 * (x - 1 / x)

theorem boundary_line_f_g :
  ∀ (x : ℝ), 1 ≤ x → (x - 1) ≤ f x ∧ (g x) ≤ (x - 1) :=
by
  intro x hx
  sorry

end boundary_line_f_g_l99_99562


namespace no_valid_coloring_l99_99775

open Nat

-- Define the color type
inductive Color
| blue
| red
| green

-- Define the coloring function
def color : ℕ → Color := sorry

-- Define the properties of the coloring function
def valid_coloring (color : ℕ → Color) : Prop :=
  ∀ (m n : ℕ), m > 1 → n > 1 → color m ≠ color n → 
    color (m * n) ≠ color m ∧ color (m * n) ≠ color n

-- Theorem: It is not possible to color all natural numbers greater than 1 as described
theorem no_valid_coloring : ¬ ∃ (color : ℕ → Color), valid_coloring color :=
by
  sorry

end no_valid_coloring_l99_99775


namespace necessary_condition_to_contain_circle_in_parabola_l99_99059

def M (x y : ℝ) : Prop := y ≥ x^2
def N (x y a : ℝ) : Prop := x^2 + (y - a)^2 ≤ 1

theorem necessary_condition_to_contain_circle_in_parabola (a : ℝ) : 
  (∀ x y, N x y a → M x y) ↔ a ≥ 5 / 4 := 
sorry

end necessary_condition_to_contain_circle_in_parabola_l99_99059


namespace distance_walked_l99_99002

theorem distance_walked (D : ℝ) (t1 t2 : ℝ): 
  (t1 = D / 4) → 
  (t2 = D / 3) → 
  (t2 - t1 = 1 / 2) → 
  D = 6 := 
by
  sorry

end distance_walked_l99_99002


namespace concurrency_of_lines_l99_99910

open EuclideanGeometry
open Real

theorem concurrency_of_lines
  (A B C D P Q : EuclideanGeometry.Point)
  (h_parallel : parallelogram A B C D)
  (hP_on_BC : collinear B C P)
  (hQ_on_CD : collinear C D Q)
  (h_sim : similar (triangle.mk A B P) (triangle.mk A D Q)) :
  concurrent (line_through B D) (line_through P Q) (tangent_to_circumcircle (triangle.mk A P Q)) :=
  sorry

end concurrency_of_lines_l99_99910


namespace find_t_l99_99576

theorem find_t (s t : ℝ) (h1 : 12 * s + 7 * t = 165) (h2 : s = t + 3) : t = 6.789 := 
by 
  sorry

end find_t_l99_99576


namespace average_people_per_hour_l99_99769

-- Define the conditions
def people_moving : ℕ := 3000
def days : ℕ := 5
def hours_per_day : ℕ := 24
def total_hours : ℕ := days * hours_per_day

-- State the problem
theorem average_people_per_hour :
  people_moving / total_hours = 25 :=
by
  -- Proof goes here
  sorry

end average_people_per_hour_l99_99769


namespace number_of_restaurants_l99_99975

theorem number_of_restaurants
  (total_units : ℕ)
  (residential_units : ℕ)
  (non_residential_units : ℕ)
  (restaurants : ℕ)
  (h1 : total_units = 300)
  (h2 : residential_units = total_units / 2)
  (h3 : non_residential_units = total_units - residential_units)
  (h4 : restaurants = non_residential_units / 2)
  : restaurants = 75 := 
by
  sorry

end number_of_restaurants_l99_99975


namespace sum_of_squares_and_product_pos_ints_l99_99491

variable (x y : ℕ)

theorem sum_of_squares_and_product_pos_ints :
  x^2 + y^2 = 289 ∧ x * y = 120 → x + y = 23 :=
by
  intro h
  sorry

end sum_of_squares_and_product_pos_ints_l99_99491


namespace ab_value_l99_99075

theorem ab_value (a b : ℕ) (ha : a > 0) (hb : b > 0) (h : a^2 + 3 * b = 33) : a * b = 24 := 
by 
  sorry

end ab_value_l99_99075


namespace room_volume_l99_99203

theorem room_volume (b l h : ℝ) (h1 : l = 3 * b) (h2 : h = 2 * b) (h3 : l * b = 12) :
  l * b * h = 48 :=
by sorry

end room_volume_l99_99203


namespace largest_angle_of_consecutive_integers_hexagon_l99_99475

theorem largest_angle_of_consecutive_integers_hexagon (a b c d e f : ℝ) 
  (h1 : a < b) (h2 : b < c) (h3 : c < d) (h4 : d < e) (h5 : e < f) 
  (h6 : a + b + c + d + e + f = 720) : 
  ∃ x, f = x + 2 ∧ (x + 2 = 122.5) :=
  sorry

end largest_angle_of_consecutive_integers_hexagon_l99_99475


namespace max_real_root_lt_100_l99_99282

theorem max_real_root_lt_100 (k a b c : ℕ) (r : ℝ)
  (ha : ∃ m : ℕ, a = k^m)
  (hb : ∃ n : ℕ, b = k^n)
  (hc : ∃ l : ℕ, c = k^l)
  (one_real_solution : b^2 = 4 * a * c)
  (r_is_root : ∃ r : ℝ, a * r^2 - b * r + c = 0)
  (r_lt_100 : r < 100) :
  r ≤ 64 := sorry

end max_real_root_lt_100_l99_99282


namespace option_a_option_b_option_d_l99_99729

open MeasureTheory

variables {Ω : Type*} [MeasurableSpace Ω]
variables {P : Measure Ω} [ProbabilityMeasure P]
variables {A B : Set Ω} [MeasurableSet A] [MeasurableSet B]

theorem option_a (hB_subset_A : B ⊆ A) (hPA : P A = 0.4) (hPB : P B = 0.2) :
  P (A ∪ B) = 0.4 ∧ P (A ∩ B) = 0.2 := sorry

theorem option_b (h_mutually_exclusive : P (A ∩ B) = 0) (hPA : P A = 0.4) (hPB : P B = 0.2) :
  P (A ∪ B) = 0.6 := sorry

theorem option_d (h_independent : indep_sets (measurable_set A) (measurable_set B) P)
  (hPA : P A = 0.4) (hPB : P B = 0.2) :
  P (A ∩ B) = 0.08 ∧ P (A ∪ B) = 0.52 := sorry

end option_a_option_b_option_d_l99_99729


namespace algebra_expression_value_l99_99561

theorem algebra_expression_value (a : ℤ) (h : (2023 - a) ^ 2 + (a - 2022) ^ 2 = 7) :
  (2023 - a) * (a - 2022) = -3 := 
sorry

end algebra_expression_value_l99_99561


namespace ArcherInGoldenArmorProof_l99_99758

-- Definitions of the problem
variables (soldiers : Nat) (archers soldiersInGolden : Nat) 
variables (soldiersInBlack archersInGolden archersInBlack swordsmenInGolden swordsmenInBlack : Nat)
variables (truthfulSwordsmenInBlack lyingArchersInBlack lyingSwordsmenInGold truthfulArchersInGold : Nat)
variables (yesToGold yesToArcher yesToMonday : Nat)

-- Given conditions
def ProblemStatement : Prop :=
  soldiers = 55 ∧
  yesToGold = 44 ∧
  yesToArcher = 33 ∧
  yesToMonday = 22 ∧
  soldiers = archers + (soldiers - archers) ∧
  soldiers = soldiersInGolden + soldiersInBlack ∧
  archers = archersInGolden + archersInBlack ∧
  soldiersInGolden = archersInGolden + swordsmenInGolden ∧
  soldiersInBlack = archersInBlack + swordsmenInBlack ∧
  truthfulSwordsmenInBlack = swordsmenInBlack ∧
  lyingArchersInBlack = archersInBlack ∧
  lyingSwordsmenInGold = swordsmenInGolden ∧
  truthfulArchersInGold = archersInGolden ∧
  yesToGold = truthfulArchersInGold + (swordsmenInGolden + archersInBlack) ∧
  yesToArcher = truthfulArchersInGold + lyingSwordsmenInGold

-- Conclusion
def Conclusion : Prop :=
  archersInGolden = 22

-- Proof statement
theorem ArcherInGoldenArmorProof : ProblemStatement → Conclusion :=
by
  sorry

end ArcherInGoldenArmorProof_l99_99758


namespace city_population_l99_99005

theorem city_population (P: ℝ) (h: 0.85 * P = 85000) : P = 100000 := 
by
  sorry

end city_population_l99_99005


namespace archers_in_golden_l99_99757

variables (soldiers archers swordsmen wearing_golden wearing_black : ℕ)
variables (truth_swordsmen_black lie_swordsmen_golden lie_archers_black truth_archers_golden : ℕ)

-- Given conditions
variables (cond1 : archers + swordsmen = 55)
variables (cond2 : wearing_golden + wearing_black = 55)
variables (cond3 : truth_swordsmen_black + lie_swordsmen_golden + lie_archers_black + truth_archers_golden = 55)
variables (cond4 : wearing_golden = 44)
variables (cond5 : archers = 33)
variables (cond6 : truth_swordsmen_black + lie_archers_black = 22)

-- Define the mathematic equivalent proof problem
theorem archers_in_golden : archers = 22 :=
by
  sorry

end archers_in_golden_l99_99757


namespace range_of_a_l99_99411

theorem range_of_a (a : ℝ) : 
  (∀ x y : ℝ, ((x - 2) ^ 2 + (y - 1) ^ 2 ≤ 1) → (2 * |x - 1| + |y - 1| ≤ a)) ↔ a ≥ 2 + Real.sqrt 5 :=
by 
  sorry

end range_of_a_l99_99411


namespace equivalence_mod_equivalence_divisible_l99_99425

theorem equivalence_mod (a b c : ℤ) :
  (∃ k : ℤ, a - b = k * c) ↔ (a % c = b % c) := by
  sorry

theorem equivalence_divisible (a b c : ℤ) :
  (a % c = b % c) ↔ (∃ k : ℤ, a - b = k * c) := by
  sorry

end equivalence_mod_equivalence_divisible_l99_99425


namespace area_enclosed_by_circle_l99_99853

theorem area_enclosed_by_circle :
  let center := (3, -10)
  let radius := 3
  let equation := ∀ (x y : ℝ), (x - center.1) ^ 2 + (y - center.2) ^ 2 = radius ^ 2
  ∃ enclosed_area : ℝ, enclosed_area = 9 * Real.pi :=
by
  sorry

end area_enclosed_by_circle_l99_99853


namespace well_diameter_l99_99393

theorem well_diameter 
  (h : ℝ) 
  (P : ℝ) 
  (C : ℝ) 
  (V : ℝ) 
  (r : ℝ) 
  (d : ℝ) 
  (π : ℝ) 
  (h_eq : h = 14)
  (P_eq : P = 15)
  (C_eq : C = 1484.40)
  (V_eq : V = C / P)
  (volume_eq : V = π * r^2 * h)
  (radius_eq : r^2 = V / (π * h))
  (diameter_eq : d = 2 * r) : 
  d = 3 :=
by
  sorry

end well_diameter_l99_99393


namespace necessary_but_not_sufficient_l99_99430

def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop := 
  ∀ n : ℕ, a n = a 0 * q^n

def is_increasing_sequence (s : ℕ → ℝ) : Prop := 
  ∀ n : ℕ, s n < s (n + 1)

def sum_first_n_terms (a : ℕ → ℝ) (q : ℝ) : ℕ → ℝ 
| 0 => a 0
| (n+1) => (sum_first_n_terms a q n) + (a 0 * q ^ n)

theorem necessary_but_not_sufficient (a : ℕ → ℝ) (q : ℝ) (h_geometric: is_geometric_sequence a q) :
  (q > 0) ∧ is_increasing_sequence (sum_first_n_terms a q) ↔ (q > 0)
:= sorry

end necessary_but_not_sufficient_l99_99430


namespace cereal_expense_in_a_year_l99_99133

def weekly_cereal_boxes := 2
def box_cost := 3.00
def weeks_in_year := 52

theorem cereal_expense_in_a_year : weekly_cereal_boxes * weeks_in_year * box_cost = 312.00 := 
by
  sorry

end cereal_expense_in_a_year_l99_99133


namespace modulus_of_z_equals_two_l99_99454

namespace ComplexProblem

open Complex

-- Definition and conditions of the problem
def satisfies_condition (z : ℂ) : Prop :=
  (z + I) * (1 + I) = 1 - I

-- Statement that needs to be proven
theorem modulus_of_z_equals_two (z : ℂ) (h : satisfies_condition z) : abs z = 2 :=
sorry

end ComplexProblem

end modulus_of_z_equals_two_l99_99454


namespace sum_center_radius_eq_neg2_l99_99269

theorem sum_center_radius_eq_neg2 (c d s : ℝ) (h_eq : ∀ x y : ℝ, x^2 + 14 * x + y^2 - 8 * y = -64 ↔ (x + c)^2 + (y + d)^2 = s^2) :
  c + d + s = -2 :=
sorry

end sum_center_radius_eq_neg2_l99_99269


namespace no_common_solution_l99_99386

theorem no_common_solution :
  ¬(∃ y : ℚ, (6 * y^2 + 11 * y - 1 = 0) ∧ (18 * y^2 + y - 1 = 0)) :=
by
  sorry

end no_common_solution_l99_99386


namespace smallest_n_term_dec_l99_99157

theorem smallest_n_term_dec (n : ℕ) (h_pos : 0 < n) (h : ∀ d, 0 < d → d = n + 150 → ∀ p, p ∣ d → (p = 2 ∨ p = 5)) :
  n = 10 :=
by {
  sorry
}

end smallest_n_term_dec_l99_99157


namespace lilith_additional_fund_l99_99600

theorem lilith_additional_fund
  (num_water_bottles : ℕ)
  (original_price : ℝ)
  (reduced_price : ℝ)
  (expected_difference : ℝ)
  (h1 : num_water_bottles = 5 * 12)
  (h2 : original_price = 2)
  (h3 : reduced_price = 1.85)
  (h4 : expected_difference = 9) :
  (num_water_bottles * original_price) - (num_water_bottles * reduced_price) = expected_difference :=
by
  sorry

end lilith_additional_fund_l99_99600


namespace number_of_restaurants_l99_99973

def units_in_building : ℕ := 300
def residential_units := units_in_building / 2
def remaining_units := units_in_building - residential_units
def restaurants := remaining_units / 2

theorem number_of_restaurants : restaurants = 75 :=
by
  sorry

end number_of_restaurants_l99_99973


namespace prism_sphere_surface_area_l99_99288

theorem prism_sphere_surface_area :
  ∀ (a b c : ℝ), (a * b = 6) → (b * c = 2) → (a * c = 3) → 
  4 * Real.pi * ((Real.sqrt ((a ^ 2) + (b ^ 2) + (c ^ 2))) / 2) ^ 2 = 14 * Real.pi :=
by
  intros a b c hab hbc hac
  sorry

end prism_sphere_surface_area_l99_99288


namespace sweeties_remainder_l99_99677

theorem sweeties_remainder (m k : ℤ) (h : m = 12 * k + 11) :
  (4 * m) % 12 = 8 :=
by
  -- The proof steps will go here
  sorry

end sweeties_remainder_l99_99677


namespace probability_no_obtuse_triangle_l99_99697

namespace CirclePoints

noncomputable def no_obtuse_triangle_probability : ℝ := 
  let p := 1/64 in
    p

theorem probability_no_obtuse_triangle (X : ℕ → ℝ) (hcirc : ∀ n, 0 ≤ X n ∧ X n < 2 * π) (hpoints : (∀ n m, n ≠ m → X n ≠ X m)) :
  no_obtuse_triangle_probability = 1/64 :=
sorry

end CirclePoints

end probability_no_obtuse_triangle_l99_99697


namespace total_money_divided_l99_99355

theorem total_money_divided (A B C : ℝ) (hA : A = 280) (h1 : A = (2 / 3) * (B + C)) (h2 : B = (2 / 3) * (A + C)) :
  A + B + C = 700 := by
  sorry

end total_money_divided_l99_99355


namespace turnip_bag_weighs_l99_99983

theorem turnip_bag_weighs (bags : List ℕ) (T : ℕ)
  (h_weights : bags = [13, 15, 16, 17, 21, 24])
  (h_turnip : T ∈ bags)
  (h_carrot_onion_relation : ∃ O C: ℕ, C = 2 * O ∧ C + O = 106 - T) :
  T = 13 ∨ T = 16 := by
  sorry

end turnip_bag_weighs_l99_99983


namespace smallest_n_for_terminating_fraction_l99_99175

-- Define what it means for a number to be a product of only prime factors of 2 and 5
def isTerminatingDenominator (d : ℕ) : Prop := ∃ (a b : ℕ), d = 2^a * 5^b

-- The main statement to prove
theorem smallest_n_for_terminating_fraction : 
  ∃ (n : ℕ), 0 < n ∧ isTerminatingDenominator (n + 150) ∧ 
  ∀ (m : ℕ), (0 < m → isTerminatingDenominator (m + 150) → n ≤ m)
:= sorry

end smallest_n_for_terminating_fraction_l99_99175


namespace closest_vector_l99_99227

theorem closest_vector 
  (s : ℝ)
  (u b d : ℝ × ℝ × ℝ)
  (h₁ : u = (3, -2, 4) + s • (6, 4, 2))
  (h₂ : b = (1, 7, 6))
  (hdir : d = (6, 4, 2))
  (h₃ : (u - b) = (2 + 6 * s, -9 + 4 * s, -2 + 2 * s)) :
  ((2 + 6 * s) * 6 + (-9 + 4 * s) * 4 + (-2 + 2 * s) * 2) = 0 →
  s = 1 / 2 :=
by
  -- Skipping the proof, adding sorry
  sorry

end closest_vector_l99_99227


namespace celine_buys_two_laptops_l99_99841

variable (number_of_laptops : ℕ)
variable (laptop_cost : ℕ := 600)
variable (smartphone_cost : ℕ := 400)
variable (number_of_smartphones : ℕ := 4)
variable (total_money_spent : ℕ := 3000)
variable (change_back : ℕ := 200)

def total_spent : ℕ := total_money_spent - change_back

def cost_of_laptops (n : ℕ) : ℕ := n * laptop_cost

def cost_of_smartphones (n : ℕ) : ℕ := n * smartphone_cost

theorem celine_buys_two_laptops :
  cost_of_laptops number_of_laptops + cost_of_smartphones number_of_smartphones = total_spent →
  number_of_laptops = 2 := by
  sorry

end celine_buys_two_laptops_l99_99841


namespace books_total_pages_l99_99279

theorem books_total_pages (x y z : ℕ) 
  (h1 : (2 / 3 : ℚ) * x - (1 / 3 : ℚ) * x = 20)
  (h2 : (3 / 5 : ℚ) * y - (2 / 5 : ℚ) * y = 15)
  (h3 : (3 / 4 : ℚ) * z - (1 / 4 : ℚ) * z = 30) : 
  x = 60 ∧ y = 75 ∧ z = 60 :=
by
  sorry

end books_total_pages_l99_99279


namespace total_time_late_l99_99215

theorem total_time_late
  (charlize_late : ℕ)
  (classmate_late : ℕ → ℕ)
  (h1 : charlize_late = 20)
  (h2 : ∀ n, n < 4 → classmate_late n = charlize_late + 10) :
  charlize_late + (∑ i in Finset.range 4, classmate_late i) = 140 := by
  sorry

end total_time_late_l99_99215


namespace factorial_zeros_in_base_16_l99_99810

theorem factorial_zeros_in_base_16 : 
  let div_count := (Nat.factorial 15).factorization 2
  4 * 2 ≤ div_count ∧ div_count < 4 * 3 :=
by
  sorry

end factorial_zeros_in_base_16_l99_99810


namespace tangent_line_passes_through_origin_l99_99893

noncomputable def curve (α : ℝ) (x : ℝ) : ℝ := x^α + 1

theorem tangent_line_passes_through_origin (α : ℝ)
  (h_tangent : ∀ (x : ℝ), curve α 1 + (α * (x - 1)) - 2 = curve α x) :
  α = 2 :=
sorry

end tangent_line_passes_through_origin_l99_99893


namespace count_real_solutions_l99_99882

theorem count_real_solutions :
  ∃ x1 x2 : ℝ, (|x1-1| = |x1-2| + |x1-3| + |x1-4| ∧ |x2-1| = |x2-2| + |x2-3| + |x2-4|)
  ∧ (x1 ≠ x2) :=
sorry

end count_real_solutions_l99_99882


namespace mow_lawn_time_l99_99097

noncomputable def time_to_mow (lawn_length lawn_width: ℝ) 
(swat_width overlap width_conversion: ℝ) (speed: ℝ) : ℝ :=
(lawn_length * lawn_width) / (((swat_width - overlap) / width_conversion) * lawn_length * speed)

theorem mow_lawn_time : 
  time_to_mow 120 180 30 6 12 6000 = 1.8 := 
by
  -- Given:
  -- Lawn dimensions: 120 feet by 180 feet
  -- Mower swath: 30 inches with 6 inches overlap
  -- Walking speed: 6000 feet per hour
  -- Conversion factor: 12 inches = 1 foot
  sorry

end mow_lawn_time_l99_99097


namespace right_triangle_property_l99_99426

-- Variables representing the lengths of the sides and the height of the right triangle
variables (a b c h : ℝ)

-- Hypotheses from the conditions
-- 1. a and b are the lengths of the legs of the right triangle
-- 2. c is the length of the hypotenuse
-- 3. h is the height to the hypotenuse
-- Given equation: 1/2 * a * b = 1/2 * c * h
def given_equation (a b c h : ℝ) : Prop := (1 / 2) * a * b = (1 / 2) * c * h

-- The theorem to prove
theorem right_triangle_property (a b c h : ℝ) (h_eq : given_equation a b c h) : (1 / a^2 + 1 / b^2) = 1 / h^2 :=
sorry

end right_triangle_property_l99_99426


namespace ratio_of_x_intercepts_l99_99334

theorem ratio_of_x_intercepts (b : ℝ) (hb: b ≠ 0) (u v: ℝ) (h₁: 8 * u + b = 0) (h₂: 4 * v + b = 0) : 
  u / v = 1 / 2 :=
by
  sorry

end ratio_of_x_intercepts_l99_99334


namespace distinct_real_roots_range_l99_99870

theorem distinct_real_roots_range (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x1^2 - 4*x1 - a = 0) ∧ (x2^2 - 4*x2 - a = 0)) ↔ a > -4 :=
by
  sorry

end distinct_real_roots_range_l99_99870


namespace molecular_weight_correct_l99_99850

-- Definition of atomic weights for the elements
def atomic_weight_C : ℝ := 12.01
def atomic_weight_H : ℝ := 1.008
def atomic_weight_O : ℝ := 16.00

-- Number of atoms in Ascorbic acid (C6H8O6)
def count_C : ℕ := 6
def count_H : ℕ := 8
def count_O : ℕ := 6

-- Calculation of molecular weight
def molecular_weight_ascorbic_acid : ℝ :=
  (count_C * atomic_weight_C) +
  (count_H * atomic_weight_H) +
  (count_O * atomic_weight_O)

theorem molecular_weight_correct :
  molecular_weight_ascorbic_acid = 176.124 :=
by sorry


end molecular_weight_correct_l99_99850


namespace mady_balls_2010th_step_l99_99457

theorem mady_balls_2010th_step :
  let base_5_digits (n : Nat) : List Nat := (Nat.digits 5 n)
  (base_5_digits 2010).sum = 6 := by
  sorry

end mady_balls_2010th_step_l99_99457


namespace min_value_of_A_l99_99794

noncomputable def A (a b c : ℝ) : ℝ :=
  (a^3 + b^3) / (8 * a * b + 9 - c^2) +
  (b^3 + c^3) / (8 * b * c + 9 - a^2) +
  (c^3 + a^3) / (8 * c * a + 9 - b^2)

theorem min_value_of_A (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a + b + c = 3) :
  A a b c = 3 / 8 :=
sorry

end min_value_of_A_l99_99794


namespace part_I_part_II_l99_99556

def sequence_def (x : ℕ → ℝ) (p : ℝ) : Prop :=
  x 1 = 1 ∧ ∀ n ∈ (Nat.succ <$> {n | n > 0}), x (n + 1) = 1 + x n / (p + x n)

theorem part_I (x : ℕ → ℝ) (p : ℝ) (h_seq : sequence_def x p) :
  p = 2 → ∀ n ∈ (Nat.succ <$> {n | n > 0}), x n < Real.sqrt 2 :=
sorry

theorem part_II (x : ℕ → ℝ) (p : ℝ) (h_seq : sequence_def x p) :
  (∀ n ∈ (Nat.succ <$> {n | n > 0}), x (n + 1) > x n) → ¬ ∃ M ∈ {n | n > 0}, ∀ n > 0, x M ≥ x n :=
sorry

end part_I_part_II_l99_99556


namespace turnips_bag_l99_99997

theorem turnips_bag (weights : List ℕ) (h : weights = [13, 15, 16, 17, 21, 24])
  (turnips: ℕ)
  (is_turnip : turnips ∈ weights)
  (o c : ℕ)
  (h1 : o + c = 106 - turnips)
  (h2 : c = 2 * o) :
  turnips = 13 ∨ turnips = 16 := by
  sorry

end turnips_bag_l99_99997


namespace abs_neg_three_l99_99536

theorem abs_neg_three : abs (-3) = 3 := 
by
  sorry

end abs_neg_three_l99_99536


namespace probability_of_number_less_than_three_l99_99947

theorem probability_of_number_less_than_three :
  let faces : Finset ℕ := {1, 2, 3, 4, 5, 6}
  let favorable_outcomes : Finset ℕ := {1, 2}
  (favorable_outcomes.card : ℚ) / (faces.card : ℚ) = 1 / 3 :=
by
  -- This is the placeholder for the actual proof.
  sorry

end probability_of_number_less_than_three_l99_99947


namespace fraction_of_visitors_who_both_enjoyed_and_understood_is_3_over_8_l99_99582

def visitors_enjoyed_understood_fraction (E U : ℕ) (total_visitors no_enjoy_no_understood : ℕ) : Prop :=
  E = U ∧
  no_enjoy_no_understood = 110 ∧
  total_visitors = 440 ∧
  E = (total_visitors - no_enjoy_no_understood) / 2 ∧
  E = 165 ∧
  (E / total_visitors) = 3 / 8

theorem fraction_of_visitors_who_both_enjoyed_and_understood_is_3_over_8 :
  ∃ (E U : ℕ), visitors_enjoyed_understood_fraction E U 440 110 :=
by
  sorry

end fraction_of_visitors_who_both_enjoyed_and_understood_is_3_over_8_l99_99582


namespace chromium_percentage_new_alloy_l99_99086

-- Define the weights and chromium percentages of the alloys
def weight_alloy1 : ℝ := 15
def weight_alloy2 : ℝ := 35
def chromium_percent_alloy1 : ℝ := 0.15
def chromium_percent_alloy2 : ℝ := 0.08

-- Define the theorem to calculate the chromium percentage of the new alloy
theorem chromium_percentage_new_alloy :
  ((weight_alloy1 * chromium_percent_alloy1 + weight_alloy2 * chromium_percent_alloy2)
  / (weight_alloy1 + weight_alloy2) * 100) = 10.1 :=
by
  sorry

end chromium_percentage_new_alloy_l99_99086


namespace probability_no_obtuse_triangle_is_9_over_64_l99_99707

noncomputable def probability_no_obtuse_triangle (A0 A1 A2 A3 : ℝ) : ℝ :=
  -- Define the probabilistic model according to the problem conditions
  -- Assuming A0, A1, A2, and A3 are positions of points on the circle parametrized by angles in radians
  let θ := real.angle.lower_half (A1 - A0) in
  let prob_A2 := (π - θ) / (2 * π) in
  let prob_A3 := (π - θ) / (2 * π) in
  (1 / 2) * (prob_A2 * prob_A3)

theorem probability_no_obtuse_triangle_is_9_over_64 :
  probability_no_obtuse_triangle A0 A1 A2 A3 = 9 / 64 :=
by sorry

end probability_no_obtuse_triangle_is_9_over_64_l99_99707


namespace turnip_bag_weight_l99_99986

/-- Given six bags with weights 13, 15, 16, 17, 21, and 24 kg,
one bag contains turnips, and the others contain either onions or carrots.
The total weight of the carrots equals twice the total weight of the onions.
Prove that the bag containing turnips can weigh either 13 kg or 16 kg. -/
theorem turnip_bag_weight (ws : list ℕ) (T : ℕ) (O C : ℕ) (h_ws : ws = [13, 15, 16, 17, 21, 24])
  (h_sum : ws.sum = 106) (h_co : C = 2 * O) (h_weight : C + O = 106 - T) :
  T = 13 ∨ T = 16 :=
sorry

end turnip_bag_weight_l99_99986


namespace solve_for_x_l99_99798

theorem solve_for_x:
  ∀ (x : ℝ), (x + 10) / (x - 4) = (x - 3) / (x + 6) → x = -(48 / 23) :=
by
  sorry

end solve_for_x_l99_99798


namespace remainder_mod_500_l99_99043

theorem remainder_mod_500 :
  ( 5^(5^(5^5)) ) % 500 = 125 :=
by
  -- proof goes here
  sorry

end remainder_mod_500_l99_99043


namespace spade_to_heart_l99_99378

-- Definition for spade and heart can be abstract geometric shapes
structure Spade := (arcs_top: ℕ) (stem_bottom: ℕ)
structure Heart := (arcs_top: ℕ) (pointed_bottom: ℕ)

-- Condition: the spade symbol must be cut into three parts
def cut_spade (s: Spade) : List (ℕ × ℕ) :=
  [(s.arcs_top, 0), (0, s.stem_bottom), (0, s.stem_bottom)]

-- Define a function to verify if the rearranged parts form a heart
def can_form_heart (pieces: List (ℕ × ℕ)) : Prop :=
  pieces = [(1, 0), (0, 1), (0, 1)]

-- The theorem that the spade parts can form a heart
theorem spade_to_heart (s: Spade) (h: Heart):
  (cut_spade s) = [(s.arcs_top, 0), (0, s.stem_bottom), (0, s.stem_bottom)] →
  can_form_heart [(s.arcs_top, 0), (s.stem_bottom, 0), (s.stem_bottom, 0)] := 
by
  sorry


end spade_to_heart_l99_99378


namespace angle_B_in_triangle_l99_99898

theorem angle_B_in_triangle (A B C : ℝ) (hA : A = 60) (hB : B = 2 * C) (hSum : A + B + C = 180) : B = 80 :=
by sorry

end angle_B_in_triangle_l99_99898


namespace total_candles_l99_99231

theorem total_candles (num_big_boxes : ℕ) (small_boxes_per_big_box : ℕ) (candles_per_small_box : ℕ) :
  num_big_boxes = 50 ∧ small_boxes_per_big_box = 4 ∧ candles_per_small_box = 40 → 
  (num_big_boxes * (small_boxes_per_big_box * candles_per_small_box) = 8000) :=
by
  intro h
  cases h with h1 h2
  cases h2 with h3 h4
  rw [h1, h3, h4]
  norm_num
  sorry

end total_candles_l99_99231


namespace cost_reduction_l99_99487

variable (a : ℝ) -- original cost
variable (p : ℝ) -- percentage reduction (in decimal form)
variable (m : ℕ) -- number of years

def cost_after_years (a p : ℝ) (m : ℕ) : ℝ :=
  a * (1 - p) ^ m

theorem cost_reduction (a p : ℝ) (m : ℕ) :
  m > 0 → cost_after_years a p m = a * (1 - p) ^ m :=
sorry

end cost_reduction_l99_99487


namespace quadratic_c_over_b_l99_99940

theorem quadratic_c_over_b :
  ∃ (b c : ℤ), (x^2 + 500 * x + 1000 = (x + b)^2 + c) ∧ (c / b = -246) :=
by sorry

end quadratic_c_over_b_l99_99940


namespace sqrt_meaningful_range_l99_99424

theorem sqrt_meaningful_range (x : ℝ) (h : 0 ≤ x - 1) : 1 ≤ x :=
by
sorry

end sqrt_meaningful_range_l99_99424


namespace smallest_n_for_terminating_decimal_l99_99141

def is_terminating_decimal (n d : ℕ) : Prop :=
  ∀ (m : ℕ), d = 2^m ∨ d = 5^m ∨ d = (2^m) * (5 : ℕ) ∨ d = (5^m) * (2 : ℕ)
  
theorem smallest_n_for_terminating_decimal :
  ∃ n : ℕ, 0 < n ∧ is_terminating_decimal n (n + 150) ∧ (∀ m: ℕ, (is_terminating_decimal m (m + 150) ∧ 0 < m) → n ≤ m) :=
sorry

end smallest_n_for_terminating_decimal_l99_99141


namespace count_three_digit_numbers_using_1_and_2_l99_99811

theorem count_three_digit_numbers_using_1_and_2 : 
  let n := 3
  let d := [1, 2]
  let total_combinations := (List.length d)^n
  let invalid_combinations := 2
  total_combinations - invalid_combinations = 6 :=
by
  let n := 3
  let d := [1, 2]
  let total_combinations := (List.length d)^n
  let invalid_combinations := 2
  show total_combinations - invalid_combinations = 6
  sorry

end count_three_digit_numbers_using_1_and_2_l99_99811


namespace largest_angle_of_consecutive_integers_in_hexagon_l99_99481

theorem largest_angle_of_consecutive_integers_in_hexagon : 
  ∀ (a : ℕ), 
    (a - 2) + (a - 1) + a + (a + 1) + (a + 2) + (a + 3) = 720 → 
    a + 3 = 122.5 :=
by sorry

end largest_angle_of_consecutive_integers_in_hexagon_l99_99481


namespace equal_intercepts_l99_99473

theorem equal_intercepts (a : ℝ) (h : ∃ (x y : ℝ), (x = (2 + a) / a ∧ y = 2 + a ∧ x = y)) :
  a = -2 ∨ a = 1 :=
by sorry

end equal_intercepts_l99_99473


namespace cards_given_l99_99105

-- Defining the conditions
def initial_cards : ℕ := 4
def final_cards : ℕ := 12

-- The theorem to be proved
theorem cards_given : final_cards - initial_cards = 8 := by
  -- Proof will go here
  sorry

end cards_given_l99_99105


namespace hyperbola_eccentricity_sqrt2_l99_99295

theorem hyperbola_eccentricity_sqrt2
  (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c = Real.sqrt (a^2 + b^2))
  (h : (c + a)^2 + (b^2 / a)^2 = 2 * c * (c + a)) :
  Real.sqrt (1 + (b / a)^2) = Real.sqrt 2 :=
by
  sorry

end hyperbola_eccentricity_sqrt2_l99_99295


namespace evaluate_expression_l99_99494

theorem evaluate_expression : (4 * 4 + 4) / (2 * 2 - 2) = 10 := by
  sorry

end evaluate_expression_l99_99494


namespace sum_abcd_eq_neg_46_div_3_l99_99451

theorem sum_abcd_eq_neg_46_div_3
  (a b c d : ℝ) 
  (h : a + 2 = b + 3 ∧ b + 3 = c + 4 ∧ c + 4 = d + 5 ∧ d + 5 = a + b + c + d + 15) :
  a + b + c + d = -46 / 3 := 
by sorry

end sum_abcd_eq_neg_46_div_3_l99_99451


namespace correlation_implies_slope_positive_l99_99565

-- Definition of the regression line
def regression_line (x y : ℝ) (b a : ℝ) : Prop :=
  y = b * x + a

-- Given conditions
variables (x y : ℝ)
variables (b a r : ℝ)

-- The statement of the proof problem
theorem correlation_implies_slope_positive (h1 : r > 0) (h2 : regression_line x y b a) : b > 0 :=
sorry

end correlation_implies_slope_positive_l99_99565


namespace Linda_sold_7_tees_l99_99456

variables (T : ℕ)
variables (jeans_price tees_price total_money_from_jeans total_money total_money_from_tees : ℕ)
variables (jeans_sold : ℕ)

def tees_sold :=
  jeans_price = 11 ∧ tees_price = 8 ∧ jeans_sold = 4 ∧
  total_money = 100 ∧ total_money_from_jeans = jeans_sold * jeans_price ∧
  total_money_from_tees = total_money - total_money_from_jeans ∧
  T = total_money_from_tees / tees_price
  
theorem Linda_sold_7_tees (h : tees_sold T jeans_price tees_price total_money_from_jeans total_money total_money_from_tees jeans_sold) : T = 7 :=
by
  sorry

end Linda_sold_7_tees_l99_99456


namespace drying_time_correct_l99_99090

theorem drying_time_correct :
  let short_haired_dog_drying_time := 10
  let full_haired_dog_drying_time := 2 * short_haired_dog_drying_time
  let num_short_haired_dogs := 6
  let num_full_haired_dogs := 9
  let total_short_haired_dogs_time := num_short_haired_dogs * short_haired_dog_drying_time
  let total_full_haired_dogs_time := num_full_haired_dogs * full_haired_dog_drying_time
  let total_drying_time_in_minutes := total_short_haired_dogs_time + total_full_haired_dogs_time
  let total_drying_time_in_hours := total_drying_time_in_minutes / 60
  total_drying_time_in_hours = 4 := 
by
  sorry

end drying_time_correct_l99_99090


namespace archers_in_golden_armor_l99_99752

theorem archers_in_golden_armor (total_soldiers archers_swordsmen total_affirmations armor_affirmations archer_affirmations monday_affirmations : ℕ)
  (h1: total_soldiers = 55)
  (h2: armor_affirmations = 44)
  (h3: archer_affirmations = 33)
  (h4: monday_affirmations = 22)
  (h5: ∑ x in ({armor_affirmations, archer_affirmations, monday_affirmations} : finset ℕ), x = 99) 
  : ∃ (archers_in_golden : ℕ), archers_in_golden = 22 := by
  -- Definitions and theorems will go here
  sorry

end archers_in_golden_armor_l99_99752


namespace angle_B_in_triangle_l99_99899

theorem angle_B_in_triangle (A B C : ℝ) (hA : A = 60) (hB : B = 2 * C) (hSum : A + B + C = 180) : B = 80 :=
by sorry

end angle_B_in_triangle_l99_99899


namespace starting_positions_P0_P1024_l99_99591

noncomputable def sequence_fn (x : ℝ) : ℝ := 4 * x / (x^2 + 1)

def find_starting_positions (n : ℕ) : ℕ := 2^n - 2

theorem starting_positions_P0_P1024 :
  ∃ P0 : ℝ, ∀ n : ℕ, P0 = sequence_fn^[n] P0 → P0 = sequence_fn^[1024] P0 ↔ find_starting_positions 1024 = 2^1024 - 2 :=
sorry

end starting_positions_P0_P1024_l99_99591


namespace division_correct_l99_99821

theorem division_correct (x : ℝ) (h : 10 / x = 2) : 20 / x = 4 :=
by
  sorry

end division_correct_l99_99821


namespace fraction_of_males_l99_99214

theorem fraction_of_males (M F : ℚ) (h1 : M + F = 1)
  (h2 : (3/4) * M + (5/6) * F = 7/9) :
  M = 2/3 :=
by sorry

end fraction_of_males_l99_99214


namespace selection_schemes_l99_99513

theorem selection_schemes (boys girls : ℕ) (hb : boys = 4) (hg : girls = 2) :
  (boys * girls = 8) :=
by
  -- Proof goes here
  intros
  sorry

end selection_schemes_l99_99513


namespace sequence_gcd_is_index_l99_99628

theorem sequence_gcd_is_index (a : ℕ → ℕ) 
  (h : ∀ i j : ℕ, i ≠ j → Nat.gcd (a i) (a j) = Nat.gcd i j) :
  ∀ i : ℕ, a i = i :=
by
  sorry

end sequence_gcd_is_index_l99_99628


namespace intersects_x_axis_vertex_coordinates_l99_99409

-- Definition of the quadratic function and conditions
def quadratic_function (a : ℝ) (x : ℝ) : ℝ :=
  x^2 - a * x - 2 * a^2

-- Condition: a ≠ 0
axiom a_nonzero (a : ℝ) : a ≠ 0

-- Statement for the first part of the problem
theorem intersects_x_axis (a : ℝ) (h : a ≠ 0) :
  ∃ x₁ x₂ : ℝ, quadratic_function a x₁ = 0 ∧ quadratic_function a x₂ = 0 ∧ x₁ * x₂ < 0 :=
by 
  sorry

-- Statement for the second part of the problem
theorem vertex_coordinates (a : ℝ) (h : a ≠ 0) (hy_intercept : quadratic_function a 0 = -2) :
  ∃ x_vertex : ℝ, quadratic_function a x_vertex = (if a = 1 then (1/2)^2 - 9/4 else (1/2)^2 - 9/4) :=
by 
  sorry


end intersects_x_axis_vertex_coordinates_l99_99409


namespace tangent_line_parabola_l99_99304

theorem tangent_line_parabola (d : ℝ) : 
    (∀ y : ℝ, (-4)^2 - 4 * (y^2 - 4 * y + 4 * d) = 0) ↔ d = 1 :=
by
    sorry

end tangent_line_parabola_l99_99304


namespace ratio_of_x_intercepts_l99_99335

theorem ratio_of_x_intercepts (b : ℝ) (hb: b ≠ 0) (u v: ℝ) (h₁: 8 * u + b = 0) (h₂: 4 * v + b = 0) : 
  u / v = 1 / 2 :=
by
  sorry

end ratio_of_x_intercepts_l99_99335


namespace find_vector_b_coordinates_l99_99876

theorem find_vector_b_coordinates 
  (a b : ℝ × ℝ) 
  (h₁ : a = (-3, 4)) 
  (h₂ : ∃ m : ℝ, m < 0 ∧ b = (-3 * m, 4 * m)) 
  (h₃ : ‖b‖ = 10) : 
  b = (6, -8) := 
by
  sorry

end find_vector_b_coordinates_l99_99876


namespace Mia_studied_fraction_l99_99787

-- Define the conditions
def total_minutes_per_day := 1440
def time_spent_watching_TV := total_minutes_per_day * 1 / 5
def time_spent_studying := 288
def remaining_time := total_minutes_per_day - time_spent_watching_TV
def fraction_studying := time_spent_studying / remaining_time

-- State the proof goal
theorem Mia_studied_fraction : fraction_studying = 1 / 4 := by
  sorry

end Mia_studied_fraction_l99_99787


namespace xiaoming_money_l99_99497

open Real

noncomputable def verify_money_left (M P_L : ℝ) : Prop := M = 12 * P_L

noncomputable def verify_money_right (M P_R : ℝ) : Prop := M = 14 * P_R

noncomputable def price_relationship (P_L P_R : ℝ) : Prop := P_R = P_L - 1

theorem xiaoming_money (M P_L P_R : ℝ) 
  (h1 : verify_money_left M P_L) 
  (h2 : verify_money_right M P_R) 
  (h3 : price_relationship P_L P_R) : 
  M = 84 := 
  by
  sorry

end xiaoming_money_l99_99497


namespace ratio_of_x_intercepts_l99_99329

theorem ratio_of_x_intercepts (b : ℝ) (hb : b ≠ 0) (u v: ℝ)
  (h1: 0 = 8 * u + b) (h2: 0 = 4 * v + b) : u / v = 1 / 2 :=
by sorry

end ratio_of_x_intercepts_l99_99329


namespace part_one_part_two_l99_99730

def f (x : ℝ) := |x + 2|

theorem part_one (x : ℝ) : 2 * f x < 4 - |x - 1| ↔ -7 / 3 < x ∧ x < -1 := sorry

theorem part_two (m n : ℝ) (x a : ℝ) (h : m > 0) (h : n > 0) (h : m + n = 1) :
  (|x - a| - f x ≤ 1/m + 1/n) ↔ (-6 ≤ a ∧ a ≤ 2) := sorry

end part_one_part_two_l99_99730


namespace total_amount_correct_l99_99009

noncomputable def total_amount : ℝ :=
  let nissin_noodles := 24 * 1.80 * 0.80
  let master_kong_tea := 6 * 1.70 * 0.80
  let shanlin_soup := 5 * 3.40
  let shuanghui_sausage := 3 * 11.20 * 0.90
  nissin_noodles + master_kong_tea + shanlin_soup + shuanghui_sausage

theorem total_amount_correct : total_amount = 89.96 := by
  sorry

end total_amount_correct_l99_99009


namespace probability_left_red_off_second_blue_on_right_blue_on_l99_99611

def num_red_lamps : ℕ := 4
def num_blue_lamps : ℕ := 4
def total_lamps : ℕ := num_red_lamps + num_blue_lamps
def num_on : ℕ := 4
def position := Fin total_lamps
def lamp_state := {state // state < (total_lamps.choose num_red_lamps) * (total_lamps.choose num_on)}

def valid_configuration (leftmost : position) (second_left : position) (rightmost : position) (s : lamp_state) : Prop :=
(leftmost.1 = 1 ∧ second_left.1 = 2 ∧ rightmost.1 = 8) ∧ (s.1 =  (((total_lamps - 3).choose 3) * ((total_lamps - 3).choose 2)))

theorem probability_left_red_off_second_blue_on_right_blue_on :
  ∀ (leftmost second_left rightmost : position) (s : lamp_state),
  valid_configuration leftmost second_left rightmost s ->
  ((total_lamps.choose num_red_lamps) * (total_lamps.choose num_on)) = 49 :=
sorry

end probability_left_red_off_second_blue_on_right_blue_on_l99_99611


namespace smallest_n_for_terminating_decimal_l99_99152

theorem smallest_n_for_terminating_decimal :
  ∃ n : ℕ, n > 0 ∧ (∀ m : ℕ, m > 0 ∧ (∀ p: ℕ, (p ∣ (m + 150) → (p = 2 ∨ p = 5)) → m ≥ n)) ∧
  (∃ two_five_factors : ℕ, (two_five_factors = 5 ^ 3 * 2 ^ 3) ∧ (n + 150 = two_five_factors)) := 
by 
   exact ⟨50, by sorry⟩

end smallest_n_for_terminating_decimal_l99_99152


namespace probability_correct_l99_99196

noncomputable def probability_two_faces_and_none_face : ℚ := 
  let total_unit_cubes := 64
  let cubes_with_two_painted_faces := 4
  let cubes_with_no_painted_faces := 36
  let total_ways_to_choose_two_cubes := Nat.choose total_unit_cubes 2
  let successful_outcomes := cubes_with_two_painted_faces * cubes_with_no_painted_faces
  successful_outcomes / total_ways_to_choose_two_cubes

theorem probability_correct : probability_two_faces_and_none_face = 1 / 14 := by
  sorry

end probability_correct_l99_99196


namespace reflection_across_x_axis_l99_99262

theorem reflection_across_x_axis :
  let initial_point := (-3, 5)
  let reflected_point := (-3, -5)
  reflected_point = (initial_point.1, -initial_point.2) :=
by
  sorry

end reflection_across_x_axis_l99_99262


namespace probability_of_event_B_l99_99650

def fair_dice := { n : ℕ // 1 ≤ n ∧ n ≤ 8 }

def event_B (x y : fair_dice) : Prop := x.val = y.val + 2

def total_outcomes : ℕ := 64

def favorable_outcomes : ℕ := 6

theorem probability_of_event_B : (favorable_outcomes : ℚ) / total_outcomes = 3/32 := by
  have h1 : (64 : ℚ) = 8 * 8 := by norm_num
  have h2 : (6 : ℚ) / 64 = 3 / 32 := by norm_num
  sorry

end probability_of_event_B_l99_99650


namespace pq_sub_l99_99190

-- Assuming the conditions
theorem pq_sub (p q : ℚ) 
  (h₁ : 3 / p = 4) 
  (h₂ : 3 / q = 18) : 
  p - q = 7 / 12 := 
  sorry

end pq_sub_l99_99190


namespace problem_statement_l99_99241

theorem problem_statement (x : ℝ) (h : x - 1/x = 5) : x^4 - (1 / x)^4 = 527 :=
sorry

end problem_statement_l99_99241


namespace value_of_a_l99_99809

theorem value_of_a {a : ℝ} 
  (h : ∀ x y : ℝ, ax - 2*y + 2 = 0 ↔ x + (a-3)*y + 1 = 0) : 
  a = 1 := 
by 
  sorry

end value_of_a_l99_99809


namespace sandy_comic_books_l99_99109

-- Problem definition
def initial_comic_books := 14
def sold_comic_books := initial_comic_books / 2
def remaining_comic_books := initial_comic_books - sold_comic_books
def bought_comic_books := 6
def final_comic_books := remaining_comic_books + bought_comic_books

-- Proof statement
theorem sandy_comic_books : final_comic_books = 13 := by
  sorry

end sandy_comic_books_l99_99109


namespace math_problem_l99_99421

theorem math_problem (n : ℤ) : 12 ∣ (n^2 * (n^2 - 1)) := 
by
  sorry

end math_problem_l99_99421


namespace lock_combination_l99_99613

-- Define the digits as distinct
def distinct_digits (V E N U S I A R : ℕ) : Prop :=
  V ≠ E ∧ V ≠ N ∧ V ≠ U ∧ V ≠ S ∧ V ≠ I ∧ V ≠ A ∧ V ≠ R ∧
  E ≠ N ∧ E ≠ U ∧ E ≠ S ∧ E ≠ I ∧ E ≠ A ∧ E ≠ R ∧
  N ≠ U ∧ N ≠ S ∧ N ≠ I ∧ N ≠ A ∧ N ≠ R ∧
  U ≠ S ∧ U ≠ I ∧ U ≠ A ∧ U ≠ R ∧
  S ≠ I ∧ S ≠ A ∧ S ≠ R ∧
  I ≠ A ∧ I ≠ R ∧
  A ≠ R

-- Define the base 12 addition for the equation
def base12_addition (V E N U S I A R : ℕ) : Prop :=
  let VENUS := V * 12^4 + E * 12^3 + N * 12^2 + U * 12^1 + S
  let IS := I * 12^1 + S
  let NEAR := N * 12^3 + E * 12^2 + A * 12^1 + R
  let SUN := S * 12^2 + U * 12^1 + N
  VENUS + IS + NEAR = SUN

-- The theorem statement
theorem lock_combination :
  ∃ (V E N U S I A R : ℕ),
    distinct_digits V E N U S I A R ∧
    base12_addition V E N U S I A R ∧
    (S * 12^2 + U * 12^1 + N) = 655 := 
sorry

end lock_combination_l99_99613


namespace solution_set_inequality_f_solution_range_a_l99_99067

-- Define the function f 
def f (x : ℝ) := |x + 1| + |x - 3|

-- Statement for question 1
theorem solution_set_inequality_f (x : ℝ) : f x < 6 ↔ -2 < x ∧ x < 4 :=
sorry

-- Statement for question 2
theorem solution_range_a (a : ℝ) (h : ∃ x : ℝ, f x = |a - 2|) : a ≥ 6 ∨ a ≤ -2 :=
sorry

end solution_set_inequality_f_solution_range_a_l99_99067


namespace price_of_shoes_on_Monday_l99_99278

noncomputable def priceOnThursday : ℝ := 50

noncomputable def increasedPriceOnFriday : ℝ := priceOnThursday * 1.2

noncomputable def discountedPriceOnMonday : ℝ := increasedPriceOnFriday * 0.85

noncomputable def finalPriceOnMonday : ℝ := discountedPriceOnMonday * 1.05

theorem price_of_shoes_on_Monday :
  finalPriceOnMonday = 53.55 :=
by
  sorry

end price_of_shoes_on_Monday_l99_99278


namespace expected_winnings_is_correct_l99_99367

noncomputable def peculiar_die_expected_winnings : ℝ :=
  (1/4) * 2 + (1/2) * 5 + (1/4) * (-10)

theorem expected_winnings_is_correct :
  peculiar_die_expected_winnings = 0.5 := by
  sorry

end expected_winnings_is_correct_l99_99367


namespace radius_of_sphere_with_surface_area_4pi_l99_99244

noncomputable def sphere_radius (surface_area: ℝ) : ℝ :=
  sorry

theorem radius_of_sphere_with_surface_area_4pi :
  sphere_radius (4 * Real.pi) = 1 :=
by
  sorry

end radius_of_sphere_with_surface_area_4pi_l99_99244


namespace remainder_5_to_5_to_5_to_5_mod_1000_l99_99037

theorem remainder_5_to_5_to_5_to_5_mod_1000 : (5^(5^(5^5))) % 1000 = 125 :=
by {
  sorry
}

end remainder_5_to_5_to_5_to_5_mod_1000_l99_99037


namespace sufficient_but_not_necessary_condition_l99_99402

variable (a b : ℝ)

theorem sufficient_but_not_necessary_condition (h1 : a < b) : 
  ((a - b) * a^2 < 0) ↔ (a < b) :=
sorry

end sufficient_but_not_necessary_condition_l99_99402


namespace ratio_books_to_pens_l99_99812

-- Define the given ratios and known constants.
def ratio_pencils : ℕ := 14
def ratio_pens : ℕ := 4
def ratio_books : ℕ := 3
def actual_pencils : ℕ := 140

-- Assume the actual number of pens can be calculated from ratio.
def actual_pens : ℕ := (actual_pencils / ratio_pencils) * ratio_pens

-- Prove that the ratio of exercise books to pens is as expected.
theorem ratio_books_to_pens (h1 : actual_pencils = 140) 
                            (h2 : actual_pens = 40) : 
  ((actual_pencils / ratio_pencils) * ratio_books) / actual_pens = 3 / 4 :=
by
  -- The following proof steps are omitted as per instruction
  sorry

end ratio_books_to_pens_l99_99812


namespace prob_A_winning_l99_99607

variable (P_draw P_B : ℚ)

def P_A_winning := 1 - P_draw - P_B

theorem prob_A_winning (h1 : P_draw = 1 / 2) (h2 : P_B = 1 / 3) :
  P_A_winning P_draw P_B = 1 / 6 :=
by
  rw [P_A_winning, h1, h2]
  norm_num
  done

end prob_A_winning_l99_99607


namespace no_flippy_numbers_divisible_by_11_and_6_l99_99545

def is_flippy (n : ℕ) : Prop :=
  let d1 := n / 10000
  let d2 := (n / 1000) % 10
  let d3 := (n / 100) % 10
  let d4 := (n / 10) % 10
  let d5 := n % 10
  (d1 = d3 ∧ d3 = d5 ∧ d2 = d4 ∧ d1 ≠ d2) ∨ 
  (d2 = d4 ∧ d4 = d5 ∧ d1 = d3 ∧ d1 ≠ d2)

def is_divisible_by_11 (n : ℕ) : Prop :=
  (n % 11) = 0

def sum_of_digits (n : ℕ) : ℕ :=
  (n / 10000) + (n / 1000) % 10 + (n / 100) % 10 + (n / 10) % 10 + n % 10

def sum_divisible_by_6 (n : ℕ) : Prop :=
  (sum_of_digits n) % 6 = 0

theorem no_flippy_numbers_divisible_by_11_and_6 :
  ∀ n, (10000 ≤ n ∧ n < 100000) → is_flippy n → is_divisible_by_11 n → sum_divisible_by_6 n → false :=
by
  intros n h_range h_flippy h_div11 h_sum6
  sorry

end no_flippy_numbers_divisible_by_11_and_6_l99_99545


namespace smallest_n_for_terminating_decimal_l99_99145

def is_terminating_decimal (n d : ℕ) : Prop :=
  ∀ (m : ℕ), d = 2^m ∨ d = 5^m ∨ d = (2^m) * (5 : ℕ) ∨ d = (5^m) * (2 : ℕ)
  
theorem smallest_n_for_terminating_decimal :
  ∃ n : ℕ, 0 < n ∧ is_terminating_decimal n (n + 150) ∧ (∀ m: ℕ, (is_terminating_decimal m (m + 150) ∧ 0 < m) → n ≤ m) :=
sorry

end smallest_n_for_terminating_decimal_l99_99145


namespace am_minus_hm_lt_bound_l99_99570

theorem am_minus_hm_lt_bound (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x < y) :
  (x - y)^2 / (2 * (x + y)) < (x - y)^2 / (8 * x) := 
by
  sorry

end am_minus_hm_lt_bound_l99_99570


namespace largest_angle_in_consecutive_integer_hexagon_l99_99483

theorem largest_angle_in_consecutive_integer_hexagon : 
  ∀ (x : ℤ), 
  (x - 3) + (x - 2) + (x - 1) + x + (x + 1) + (x + 2) = 720 → 
  (x + 2 = 122) :=
by intros x h
   sorry

end largest_angle_in_consecutive_integer_hexagon_l99_99483


namespace magician_earnings_at_least_l99_99365

def magician_starting_decks := 15
def magician_remaining_decks := 3
def decks_sold := magician_starting_decks - magician_remaining_decks
def standard_price_per_deck := 3
def discount := 1
def discounted_price_per_deck := standard_price_per_deck - discount
def min_earnings := decks_sold * discounted_price_per_deck

theorem magician_earnings_at_least :
  min_earnings ≥ 24 :=
by sorry

end magician_earnings_at_least_l99_99365


namespace min_value_inverse_sum_l99_99290

theorem min_value_inverse_sum {m n : ℝ} (h1 : -2 * m - 2 * n + 1 = 0) (h2 : m * n > 0) : 
  (1 / m + 1 / n) ≥ 8 :=
sorry

end min_value_inverse_sum_l99_99290


namespace smallest_n_for_terminating_fraction_l99_99171

-- Define what it means for a number to be a product of only prime factors of 2 and 5
def isTerminatingDenominator (d : ℕ) : Prop := ∃ (a b : ℕ), d = 2^a * 5^b

-- The main statement to prove
theorem smallest_n_for_terminating_fraction : 
  ∃ (n : ℕ), 0 < n ∧ isTerminatingDenominator (n + 150) ∧ 
  ∀ (m : ℕ), (0 < m → isTerminatingDenominator (m + 150) → n ≤ m)
:= sorry

end smallest_n_for_terminating_fraction_l99_99171


namespace f_identically_zero_l99_99406

open Real

-- Define the function f and its properties
noncomputable def f : ℝ → ℝ := sorry

-- Given conditions
axiom func_eqn (a b : ℝ) : f (a * b) = a * f b + b * f a 
axiom func_bounded (x : ℝ) : |f x| ≤ 1

-- Goal: Prove that f is identically zero
theorem f_identically_zero : ∀ x : ℝ, f x = 0 := 
by
  sorry

end f_identically_zero_l99_99406


namespace smallest_n_for_terminating_decimal_l99_99151

theorem smallest_n_for_terminating_decimal :
  ∃ n : ℕ, n > 0 ∧ (∀ m : ℕ, m > 0 ∧ (∀ p: ℕ, (p ∣ (m + 150) → (p = 2 ∨ p = 5)) → m ≥ n)) ∧
  (∃ two_five_factors : ℕ, (two_five_factors = 5 ^ 3 * 2 ^ 3) ∧ (n + 150 = two_five_factors)) := 
by 
   exact ⟨50, by sorry⟩

end smallest_n_for_terminating_decimal_l99_99151


namespace calculate_B_l99_99428
open Real

theorem calculate_B 
  (A B : ℝ) 
  (a b : ℝ) 
  (hA : A = π / 6) 
  (ha : a = 1) 
  (hb : b = sqrt 3) 
  (h_sin_relation : sin B = (b * sin A) / a) : 
  (B = π / 3 ∨ B = 2 * π / 3) :=
sorry

end calculate_B_l99_99428


namespace find_real_pairs_l99_99864

theorem find_real_pairs (x y : ℝ) (h : 2 * x / (1 + x^2) = (1 + y^2) / (2 * y)) : 
  (x = 1 ∧ y = 1) ∨ (x = -1 ∧ y = -1) :=
sorry

end find_real_pairs_l99_99864


namespace heights_equal_l99_99292

-- Define base areas and volumes
variables {V : ℝ} {S : ℝ}

-- Assume equal volumes and base areas for the prism and cylinder
variables (h_prism h_cylinder : ℝ) (volume_eq : V = S * h_prism) (base_area_eq : S = S)

-- Define a proof goal
theorem heights_equal 
  (equal_volumes : V = S * h_prism) 
  (equal_base_areas : S = S) : 
  h_prism = h_cylinder :=
sorry

end heights_equal_l99_99292


namespace total_birds_correct_l99_99528

-- Define the conditions
def number_of_trees : ℕ := 7
def blackbirds_per_tree : ℕ := 3
def magpies : ℕ := 13

-- Define the total number of blackbirds using the conditions
def total_blackbirds : ℕ := number_of_trees * blackbirds_per_tree

-- Define the total number of birds using the total number of blackbirds and the number of magpies
def total_birds : ℕ := total_blackbirds + magpies

-- The theorem statement that should be proven
theorem total_birds_correct : total_birds = 34 := 
sorry

end total_birds_correct_l99_99528


namespace volume_of_cube_in_pyramid_l99_99895

open Real

noncomputable def side_length_of_base := 2
noncomputable def height_of_equilateral_triangle := sqrt 6
noncomputable def cube_side_length := sqrt 6 / 3
noncomputable def volume_of_cube := cube_side_length ^ 3

theorem volume_of_cube_in_pyramid 
  (side_length_of_base : ℝ) (height_of_equilateral_triangle : ℝ) (cube_side_length : ℝ) :
  volume_of_cube = 2 * sqrt 6 / 9 := 
by
  sorry

end volume_of_cube_in_pyramid_l99_99895


namespace billy_can_play_l99_99531

-- Define the conditions
def total_songs : ℕ := 52
def songs_to_learn : ℕ := 28

-- Define the statement to be proved
theorem billy_can_play : total_songs - songs_to_learn = 24 := by
  -- Proof goes here
  sorry

end billy_can_play_l99_99531


namespace potato_yield_l99_99916

/-- Mr. Green's gardening problem -/
theorem potato_yield
  (steps_length : ℝ)
  (steps_width : ℝ)
  (step_size : ℝ)
  (yield_rate : ℝ)
  (feet_length := steps_length * step_size)
  (feet_width := steps_width * step_size)
  (area := feet_length * feet_width)
  (yield := area * yield_rate) :
  steps_length = 18 →
  steps_width = 25 →
  step_size = 2.5 →
  yield_rate = 0.75 →
  yield = 2109.375 :=
by
  sorry

end potato_yield_l99_99916


namespace archers_in_golden_armor_count_l99_99762

theorem archers_in_golden_armor_count : 
  ∃ (archers_golden : ℕ), 
    let total_soldiers := 55 in
    let golden_armor_affirmative := 44 in
    let archer_affirmative := 33 in
    let monday_affirmative := 22 in
    let number_archers_golden := archers_golden in
    (number_archers_golden = 22) ∧ 
    ∃ (archer : ℕ) (swordsman : ℕ) (golden : ℕ) (black : ℕ),
      archer + swordsman = total_soldiers ∧
      golden + black = total_soldiers ∧
      golden + black = total_soldiers ∧
      (swordsman * golden + archer * golden = golden_armor_affirmative) ∧
      (swordsman * archer + swordsman * golden = archer_affirmative) ∧
      ((!monday_affirmative ∧ swordsman * golden) + (!monday_affirmative ∧ archer * black) = monday_affirmative)
:= sorry

end archers_in_golden_armor_count_l99_99762


namespace leak_empty_tank_time_l99_99280

theorem leak_empty_tank_time (fill_time_A : ℝ) (fill_time_A_with_leak : ℝ) (leak_empty_time : ℝ) :
  fill_time_A = 6 → fill_time_A_with_leak = 9 → leak_empty_time = 18 :=
by
  intros hA hL
  -- Here follows the proof we skip
  sorry

end leak_empty_tank_time_l99_99280


namespace problem1_problem2_problem3_l99_99018

-- Definition of given quantities and conditions
variables (a b x : ℝ) (α β : ℝ)

-- Given Conditions
@[simp] def cond1 := true
@[simp] def cond2 := true
@[simp] def cond3 := true
@[simp] def cond4 := true

-- First Question
theorem problem1 (h1 : cond1) (h2 : cond2) (h3 : cond3) (h4 : cond4) : 
    a * Real.sin α = b * Real.sin β := sorry

-- Second Question
theorem problem2 (h1 : cond1) (h2 : cond2) (h3 : cond3) (h4 : cond4) : 
    Real.sin β ≤ a / b := sorry

-- Third Question
theorem problem3 (h1 : cond1) (h2 : cond2) (h3 : cond3) (h4 : cond4) : 
    x = a * (1 - Real.cos α) + b * (1 - Real.cos β) := sorry

end problem1_problem2_problem3_l99_99018


namespace find_angle_between_vectors_l99_99731

variables (a b : ℝ → ℝ) [nonzero a] [nonzero b] -- defining vector functions

def orthogonal (a b : ℝ → ℝ) : Prop := 
  (a · (2 * a + b)) = 0 -- dot product equality for orthogonality

def norm_squared (v : ℝ → ℝ) : ℝ := 
  (v · v) -- norm squared is the dot product with itself

def magnitude_relation (a b : ℝ → ℝ) (h : norm_squared b = 16 * norm_squared a) :=
  h -- magnitude condition inferred from norms

def angle_between (a b : ℝ → ℝ) : ℝ :=
  let cos_theta := (-1 / 2) in
  real.acos cos_theta -- angle θ such that cos θ = -1/2

theorem find_angle_between_vectors
  (a b : ℝ → ℝ)
  [nonzero a]
  [nonzero b]
  (h1 : orthogonal a b)
  (h2 : |b| = 4*|a|): 
  angle_between a b = 2 * real.pi / 3 :=
begin
  sorry -- proof
end

end find_angle_between_vectors_l99_99731


namespace price_first_variety_is_126_l99_99617

variable (x : ℝ) -- price of the first variety per kg (unknown we need to solve for)
variable (p2 : ℝ := 135) -- price of the second variety per kg
variable (p3 : ℝ := 175.5) -- price of the third variety per kg
variable (mix_ratio : ℝ := 4) -- total weight ratio of the mixture
variable (mix_price : ℝ := 153) -- price of the mixture per kg
variable (w1 w2 w3 : ℝ := 1) -- weights of the first two varieties
variable (w4 : ℝ := 2) -- weight of the third variety

theorem price_first_variety_is_126:
  (w1 * x + w2 * p2 + w4 * p3) / mix_ratio = mix_price → x = 126 := by
  sorry

end price_first_variety_is_126_l99_99617


namespace sandy_comic_books_l99_99108

-- Define Sandy's initial number of comic books
def initial_comic_books : ℕ := 14

-- Define the number of comic books Sandy sold
def sold_comic_books (n : ℕ) : ℕ := n / 2

-- Define the number of comic books Sandy bought
def bought_comic_books : ℕ := 6

-- Define the number of comic books Sandy has now
def final_comic_books (initial : ℕ) (sold : ℕ) (bought : ℕ) : ℕ :=
  initial - sold + bought

-- The theorem statement to prove the final number of comic books
theorem sandy_comic_books : final_comic_books initial_comic_books (sold_comic_books initial_comic_books) bought_comic_books = 13 := by
  sorry

end sandy_comic_books_l99_99108


namespace number_of_children_in_group_l99_99199

-- Definitions based on the conditions
def num_adults : ℕ := 55
def meal_for_adults : ℕ := 70
def meal_for_children : ℕ := 90
def remaining_children_after_adults : ℕ := 81
def num_adults_eaten : ℕ := 7
def ratio_adult_to_child : ℚ := (70 : ℚ) / 90

-- Statement of the problem to prove number of children in the group
theorem number_of_children_in_group : 
  ∃ C : ℕ, 
    (meal_for_adults - num_adults_eaten) * (ratio_adult_to_child) = (remaining_children_after_adults) ∧
    C = remaining_children_after_adults := 
sorry

end number_of_children_in_group_l99_99199


namespace number_of_students_playing_soccer_l99_99903

-- Definitions of the conditions
def total_students : ℕ := 500
def total_boys : ℕ := 350
def percent_boys_playing_soccer : ℚ := 0.86
def girls_not_playing_soccer : ℕ := 115

-- To be proved
theorem number_of_students_playing_soccer :
  ∃ (S : ℕ), S = 250 ∧ 0.14 * (S : ℚ) = 35 :=
sorry

end number_of_students_playing_soccer_l99_99903


namespace cars_no_air_conditioning_l99_99581

variables {A R AR : Nat}

/-- Given a total of 100 cars, of which at least 51 have racing stripes,
and the greatest number of cars that could have air conditioning but not racing stripes is 49,
prove that the number of cars that do not have air conditioning is 49. -/
theorem cars_no_air_conditioning :
  ∀ (A R AR : ℕ), 
  (A = AR + 49) → 
  (R ≥ 51) → 
  (AR ≤ R) → 
  (AR ≤ 51) → 
  (100 - A = 49) :=
by
  intros A R AR h1 h2 h3 h4
  exact sorry

end cars_no_air_conditioning_l99_99581


namespace red_candies_difference_l99_99320

def jar1_ratio_red : ℕ := 7
def jar1_ratio_yellow : ℕ := 3
def jar2_ratio_red : ℕ := 5
def jar2_ratio_yellow : ℕ := 4
def total_yellow : ℕ := 108

theorem red_candies_difference :
  ∀ (x y : ℚ), jar1_ratio_yellow * x + jar2_ratio_yellow * y = total_yellow ∧ jar1_ratio_red + jar1_ratio_yellow = jar2_ratio_red + jar2_ratio_yellow → 10 * x = 9 * y → 7 * x - 5 * y = 21 := 
by sorry

end red_candies_difference_l99_99320


namespace complement_union_l99_99671

def R := Set ℝ

def A : Set ℝ := {x | x ≥ 1}

def B : Set ℝ := {y | ∃ x, x ≥ 1 ∧ y = Real.exp x}

theorem complement_union (R : Set ℝ) (A : Set ℝ) (B : Set ℝ) :
  (A ∪ B)ᶜ = {x | x < 1} := by
  sorry

end complement_union_l99_99671


namespace not_sixth_power_of_integer_l99_99463

theorem not_sixth_power_of_integer (n : ℕ) : ¬ ∃ k : ℤ, 6 * n^3 + 3 = k^6 :=
by
  sorry

end not_sixth_power_of_integer_l99_99463


namespace smallest_n_terminating_decimal_l99_99167

theorem smallest_n_terminating_decimal :
  ∃ n : ℕ, (∀ m : ℕ, (∃ k : ℕ, m + 150 = 2^k * (5^0))
  → (m ≥ n)) ∧ (n > 0) ∧ (∃ k : ℕ, n + 150 = 2^k * (5^0)) := 
sorry

end smallest_n_terminating_decimal_l99_99167


namespace problem1_l99_99354

theorem problem1 :
  (2021 - Real.pi)^0 + (Real.sqrt 3 - 1) - 2 + (2 * Real.sqrt 3) = 3 * Real.sqrt 3 - 2 :=
by
  sorry

end problem1_l99_99354


namespace probability_no_obtuse_triangle_l99_99698

namespace CirclePoints

noncomputable def no_obtuse_triangle_probability : ℝ := 
  let p := 1/64 in
    p

theorem probability_no_obtuse_triangle (X : ℕ → ℝ) (hcirc : ∀ n, 0 ≤ X n ∧ X n < 2 * π) (hpoints : (∀ n m, n ≠ m → X n ≠ X m)) :
  no_obtuse_triangle_probability = 1/64 :=
sorry

end CirclePoints

end probability_no_obtuse_triangle_l99_99698


namespace terminal_side_angle_l99_99081

open Real

theorem terminal_side_angle (α : ℝ) (m n : ℝ) (h_line : n = 3 * m) (h_radius : m^2 + n^2 = 10) (h_sin : sin α < 0) (h_coincide : tan α = 3) : m - n = 2 :=
by
  sorry

end terminal_side_angle_l99_99081


namespace symmetric_points_product_l99_99585

theorem symmetric_points_product (a b : ℝ) 
    (h1 : a + 2 = -4) 
    (h2 : b = 2) : 
    a * b = -12 := 
sorry

end symmetric_points_product_l99_99585


namespace product_xyz_equals_1080_l99_99452

noncomputable def xyz_product (x y z : ℝ) : ℝ :=
  if (x > 0) ∧ (y > 0) ∧ (z > 0) ∧ (x * (y + z) = 198) ∧ (y * (z + x) = 216) ∧ (z * (x + y) = 234)
  then x * y * z
  else 0 

theorem product_xyz_equals_1080 {x y z : ℝ} :
  (x > 0) ∧ (y > 0) ∧ (z > 0) ∧ (x * (y + z) = 198) ∧ (y * (z + x) = 216) ∧ (z * (x + y) = 234) →
  xyz_product x y z = 1080 :=
by
  intros h
  -- Proof skipped
  sorry

end product_xyz_equals_1080_l99_99452


namespace factor_polynomial_l99_99056

theorem factor_polynomial (y : ℝ) : 3 * y ^ 2 - 75 = 3 * (y - 5) * (y + 5) :=
by
  sorry

end factor_polynomial_l99_99056


namespace common_difference_arithmetic_geometric_sequence_l99_99065

theorem common_difference_arithmetic_geometric_sequence (a : ℕ → ℝ) (d : ℝ) 
  (h_arith : ∀ n, a (n + 1) = a n + d)
  (h_geom : ∃ r, ∀ n, a (n+1) = a n * r)
  (h_a1 : a 1 = 1) :
  d = 0 :=
by
  sorry

end common_difference_arithmetic_geometric_sequence_l99_99065


namespace find_number_l99_99347

theorem find_number : ∃ x : ℝ, (x / 5 + 7 = x / 4 - 7) ∧ x = 280 :=
by
  -- Here, we state the existence of a real number x
  -- such that the given condition holds and x = 280.
  sorry

end find_number_l99_99347


namespace circle_equation_l99_99121

def parabola (x : ℝ) : ℝ := x^2 - 2*x - 3

theorem circle_equation : ∃ (x y : ℝ), (x - 1)^2 + (y + 1)^2 = 5 ∧ (y = parabola x) ∧ (x = -1 ∨ x = 3 ∨ (x = 0 ∧ y = -3)) :=
by { sorry }

end circle_equation_l99_99121


namespace no_obtuse_triangle_probability_eq_l99_99702

noncomputable def probability_no_obtuse_triangle : ℝ :=
  let num_points := 4
  -- Condition (1): Four points are chosen uniformly at random on a circle.
  -- Condition (2): An obtuse angle occurs if the minor arc exceeds π/2.
  9 / 64

theorem no_obtuse_triangle_probability_eq :
  let num_points := 4
  ∀ (points : Fin num_points → ℝ), 
  (∀ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k →
    ∠ (points i, points j, points k) < π / 2) →
  probability_no_obtuse_triangle = 9 / 64 :=
by
  sorry

end no_obtuse_triangle_probability_eq_l99_99702


namespace area_of_smaller_circle_l99_99020

theorem area_of_smaller_circle (r R : ℝ) (PA AB : ℝ) 
  (h1 : R = 2 * r) (h2 : PA = 4) (h3 : AB = 4) :
  π * r^2 = 2 * π :=
by
  sorry

end area_of_smaller_circle_l99_99020


namespace k_zero_only_solution_l99_99031

noncomputable def polynomial_factorable (k : ℤ) : Prop :=
  ∃ (A B C D E F : ℤ), (A * D = 1) ∧ (B * E = 4) ∧ (A * E + B * D = k) ∧ (A * F + C * D = 1) ∧ (C * F = -k)

theorem k_zero_only_solution : ∀ k : ℤ, polynomial_factorable k ↔ k = 0 :=
by 
  sorry

end k_zero_only_solution_l99_99031


namespace selection_methods_l99_99496

-- Define the number of students and lectures.
def numberOfStudents : Nat := 6
def numberOfLectures : Nat := 5

-- Define the problem as proving the number of selection methods equals 5^6.
theorem selection_methods : (numberOfLectures ^ numberOfStudents) = 15625 := by
  -- Include the proper mathematical equivalence statement
  sorry

end selection_methods_l99_99496


namespace mr_bodhi_adds_twenty_sheep_l99_99603

def cows : ℕ := 20
def foxes : ℕ := 15
def zebras : ℕ := 3 * foxes
def required_total : ℕ := 100

def sheep := required_total - (cows + foxes + zebras)

theorem mr_bodhi_adds_twenty_sheep : sheep = 20 :=
by
  -- Proof for the theorem is not required and is thus replaced with sorry.
  sorry

end mr_bodhi_adds_twenty_sheep_l99_99603


namespace line_tangent_parabola_unique_d_l99_99302

theorem line_tangent_parabola_unique_d :
  (∃ d : ℝ, ∀ x : ℝ, y = 3 * x + d -> y^2 = 12 * x) -> d = 1 := by
sorry

end line_tangent_parabola_unique_d_l99_99302


namespace ratio_a_over_3_to_b_over_2_l99_99575

theorem ratio_a_over_3_to_b_over_2 (a b c : ℝ) (h1 : 2 * a = 3 * b) (h2 : c ≠ 0) (h3 : 3 * a + 2 * b = c) :
  (a / 3) / (b / 2) = 1 :=
sorry

end ratio_a_over_3_to_b_over_2_l99_99575


namespace barge_arrives_at_B_at_2pm_l99_99021

noncomputable def barge_arrival_time
  (constant_barge_speed : ℝ)
  (river_current_speed : ℝ)
  (distance_AB : ℝ)
  (time_depart_A : ℕ)
  (wait_time_B : ℝ)
  (time_return_A : ℝ) :
  ℝ := by
  sorry

theorem barge_arrives_at_B_at_2pm :
  ∀ (constant_barge_speed : ℝ), 
    (river_current_speed = 3) →
    (distance_AB = 60) →
    (time_depart_A = 9) →
    (wait_time_B = 2) →
    (time_return_A = 19 + 20 / 60) →
    barge_arrival_time constant_barge_speed river_current_speed distance_AB time_depart_A wait_time_B time_return_A = 14 := by
  sorry

end barge_arrives_at_B_at_2pm_l99_99021


namespace count_false_propositions_l99_99313

theorem count_false_propositions 
  (P : Prop) 
  (inverse_P : Prop) 
  (negation_P : Prop) 
  (converse_P : Prop) 
  (h1 : ¬P) 
  (h2 : inverse_P) 
  (h3 : negation_P ↔ ¬P) 
  (h4 : converse_P ↔ P) : 
  ∃ n : ℕ, n = 2 ∧ 
  ¬P ∧ ¬converse_P ∧ 
  inverse_P ∧ negation_P := 
sorry

end count_false_propositions_l99_99313


namespace simplify_fraction_l99_99106

theorem simplify_fraction (a b : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) (h₃ : a ≠ b) :
  (a^3 - b^3) / (a * b) - (a * b^2 - b^3) / (a * b - a^3) = (a^2 + a * b + b^2) / b :=
by {
  -- Proof skipped
  sorry
}

end simplify_fraction_l99_99106


namespace college_students_freshmen_psych_majors_l99_99213

variable (T : ℕ)
variable (hT : T > 0)

def freshmen (T : ℕ) : ℕ := 40 * T / 100
def lib_arts (F : ℕ) : ℕ := 50 * F / 100
def psych_majors (L : ℕ) : ℕ := 50 * L / 100
def percent_freshmen_psych_majors (P : ℕ) (T : ℕ) : ℕ := 100 * P / T

theorem college_students_freshmen_psych_majors :
  percent_freshmen_psych_majors (psych_majors (lib_arts (freshmen T))) T = 10 := by
  sorry

end college_students_freshmen_psych_majors_l99_99213


namespace ratio_u_v_l99_99322

theorem ratio_u_v (b : ℝ) (hb : b ≠ 0) (u v : ℝ) 
  (h1 : 0 = 8 * u + b) 
  (h2 : 0 = 4 * v + b) 
  : u / v = 1 / 2 := 
by sorry

end ratio_u_v_l99_99322


namespace lorelai_jellybeans_l99_99610

variable (Gigi Rory Luke Lane Lorelai : ℕ)
variable (h1 : Gigi = 15)
variable (h2 : Rory = Gigi + 30)
variable (h3 : Luke = 2 * Rory)
variable (h4 : Lane = Gigi + 10)
variable (h5 : Lorelai = 3 * (Gigi + Luke + Lane))

theorem lorelai_jellybeans : Lorelai = 390 := by
  sorry

end lorelai_jellybeans_l99_99610


namespace championship_outcomes_l99_99232

theorem championship_outcomes :
  ∀ (students events : ℕ), students = 4 → events = 3 → students ^ events = 64 :=
by
  intros students events h_students h_events
  rw [h_students, h_events]
  exact rfl

end championship_outcomes_l99_99232


namespace geo_seq_sum_monotone_l99_99432

theorem geo_seq_sum_monotone (q a1 : ℝ) (n : ℕ) (S : ℕ → ℝ) :
  (∀ n, S (n + 1) > S n) ↔ (a1 > 0 ∧ q > 0) :=
sorry -- Proof of the theorem (omitted)

end geo_seq_sum_monotone_l99_99432


namespace rachel_pizza_eaten_l99_99543

theorem rachel_pizza_eaten (pizza_total : ℕ) (pizza_bella : ℕ) (pizza_rachel : ℕ) :
  pizza_total = pizza_bella + pizza_rachel → pizza_bella = 354 → pizza_total = 952 → pizza_rachel = 598 :=
by
  intros h1 h2 h3
  rw [h2, h3] at h1
  sorry

end rachel_pizza_eaten_l99_99543


namespace how_many_pints_did_Annie_pick_l99_99854

theorem how_many_pints_did_Annie_pick (x : ℕ) (h1 : Kathryn = x + 2)
                                      (h2 : Ben = Kathryn - 3)
                                      (h3 : x + Kathryn + Ben = 25) : x = 8 :=
  sorry

end how_many_pints_did_Annie_pick_l99_99854


namespace g_half_eq_neg_one_l99_99577

noncomputable def f (x : ℝ) : ℝ := 2^x
noncomputable def g (x : ℝ) : ℝ := Real.log x / Real.log 2

theorem g_half_eq_neg_one : g (1/2) = -1 := by 
  sorry

end g_half_eq_neg_one_l99_99577


namespace cake_pieces_l99_99587

theorem cake_pieces (pan_length : ℕ) (pan_width : ℕ) (piece_length : ℕ) (piece_width : ℕ) 
  (pan_dim : pan_length = 24 ∧ pan_width = 15) 
  (piece_dim : piece_length = 3 ∧ piece_width = 2) : 
  (pan_length * pan_width) / (piece_length * piece_width) = 60 :=
sorry

end cake_pieces_l99_99587


namespace smallest_w_for_factors_l99_99823

theorem smallest_w_for_factors (w : ℕ) (h_pos : 0 < w) :
  (2^5 ∣ 936 * w) ∧ (3^3 ∣ 936 * w) ∧ (13^2 ∣ 936 * w) ↔ w = 156 := 
sorry

end smallest_w_for_factors_l99_99823


namespace percentage_error_l99_99846

theorem percentage_error (e : ℝ) : (1 + e / 100)^2 = 1.1025 → e = 5.125 := 
by sorry

end percentage_error_l99_99846


namespace smallest_four_digit_multiple_of_18_l99_99684

theorem smallest_four_digit_multiple_of_18 : ∃ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ 18 ∣ n ∧ ∀ m : ℕ, 1000 ≤ m ∧ m ≤ 9999 ∧ 18 ∣ m → n ≤ m := by
  use 1008
  sorry

end smallest_four_digit_multiple_of_18_l99_99684


namespace book_length_l99_99746

variable (length width perimeter : ℕ)

theorem book_length
  (h1 : perimeter = 100)
  (h2 : width = 20)
  (h3 : perimeter = 2 * (length + width)) :
  length = 30 :=
by sorry

end book_length_l99_99746


namespace max_profit_is_4sqrt6_add_21_l99_99357

noncomputable def profit (x : ℝ) : ℝ :=
  let y1 : ℝ := -2 * (3 - x)^2 + 14 * (3 - x)
  let y2 : ℝ := - (1 / 3) * x^3 + 2 * x^2 + 5 * x
  let F : ℝ := y1 + y2 - 3
  F

theorem max_profit_is_4sqrt6_add_21 : 
  ∃ x : ℝ, 0 ≤ x ∧ x ≤ 3 ∧ profit x = 4 * Real.sqrt 6 + 21 :=
sorry

end max_profit_is_4sqrt6_add_21_l99_99357


namespace smallest_four_digit_multiple_of_18_l99_99685

-- Define the concept of a four-digit number
def four_digit (N : ℕ) : Prop := 1000 ≤ N ∧ N < 10000

-- Define the concept of a multiple of 18
def multiple_of_18 (N : ℕ) : Prop := ∃ k : ℕ, N = 18 * k

-- Define the combined condition for N being a four-digit multiple of 18
def four_digit_multiple_of_18 (N : ℕ) : Prop := four_digit N ∧ multiple_of_18 N

-- State that 1008 is the smallest such number
theorem smallest_four_digit_multiple_of_18 : ∀ N : ℕ, four_digit_multiple_of_18 N → 1008 ≤ N := 
by
  intros N H
  sorry

end smallest_four_digit_multiple_of_18_l99_99685


namespace proof_inequality_l99_99058

noncomputable def proof_problem (x : ℝ) (Hx : x ∈ Set.Ioo (Real.exp (-1)) (1)) : Prop :=
  let a := Real.log x
  let b := (1 / 2) ^ (Real.log x)
  let c := Real.exp (Real.log x)
  b > c ∧ c > a

theorem proof_inequality {x : ℝ} (Hx : x ∈ Set.Ioo (Real.exp (-1)) (1)) :
  proof_problem x Hx :=
sorry

end proof_inequality_l99_99058


namespace length_of_solution_set_l99_99624

variable {a b : ℝ}

theorem length_of_solution_set (h : ∀ x : ℝ, a ≤ 3 * x + 4 ∧ 3 * x + 4 ≤ b → 12 = (b - a) / 3) : b - a = 36 :=
sorry

end length_of_solution_set_l99_99624


namespace B_and_C_mutually_exclusive_l99_99233

-- Defining events in terms of products being defective or not
def all_not_defective (products : List Bool) : Prop := 
  ∀ x ∈ products, ¬x

def all_defective (products : List Bool) : Prop := 
  ∀ x ∈ products, x

def not_all_defective (products : List Bool) : Prop := 
  ∃ x ∈ products, ¬x

-- Given a batch of three products, define events A, B, and C
def A (products : List Bool) : Prop := all_not_defective products
def B (products : List Bool) : Prop := all_defective products
def C (products : List Bool) : Prop := not_all_defective products

-- The theorem to prove that B and C are mutually exclusive
theorem B_and_C_mutually_exclusive (products : List Bool) (h : products.length = 3) : 
  ¬ (B products ∧ C products) :=
by
  sorry

end B_and_C_mutually_exclusive_l99_99233


namespace cake_pieces_per_sister_l99_99551

theorem cake_pieces_per_sister (total_pieces : ℕ) (percentage_eaten : ℕ) (sisters : ℕ)
  (h1 : total_pieces = 240) (h2 : percentage_eaten = 60) (h3 : sisters = 3) :
  (total_pieces * (1 - percentage_eaten / 100)) / sisters = 32 :=
by
  sorry

end cake_pieces_per_sister_l99_99551


namespace coefficient_m5_n5_in_expansion_l99_99635

-- Definition of the binomial coefficient
def binomial (n k : ℕ) : ℕ := Nat.choose n k

-- Goal: prove the coefficient of m^5 n^5 in the expansion of (m+n)^{10} is 252
theorem coefficient_m5_n5_in_expansion : binomial 10 5 = 252 :=
by
  sorry

end coefficient_m5_n5_in_expansion_l99_99635


namespace convex_hexagon_largest_angle_l99_99478

theorem convex_hexagon_largest_angle 
  (x : ℝ)                                 -- Denote the measure of the third smallest angle as x.
  (angles : Fin 6 → ℝ)                     -- Define the angles as a function from Fin 6 to ℝ.
  (h1 : ∀ i : Fin 6, angles i = x + (i : ℝ) - 3)  -- The six angles in increasing order.
  (h2 : 0 < x - 3 ∧ x - 3 < 180)           -- Convex condition: each angle is between 0 and 180.
  (h3 : angles ⟨0⟩ + angles ⟨1⟩ + angles ⟨2⟩ + angles ⟨3⟩ + angles ⟨4⟩ + angles ⟨5⟩ = 720) -- Sum of interior angles of a hexagon.
  : (∃ a, a = angles ⟨5⟩ ∧ a = 122.5) :=   -- Prove the largest angle in this arrangement is 122.5.
sorry

end convex_hexagon_largest_angle_l99_99478


namespace sum_of_two_numbers_eq_l99_99939

theorem sum_of_two_numbers_eq (x y : ℝ) (h1 : x * y = 16) (h2 : 1 / x = 3 * (1 / y)) : x + y = (16 * Real.sqrt 3) / 3 :=
by sorry

end sum_of_two_numbers_eq_l99_99939


namespace cows_with_no_spot_l99_99920

theorem cows_with_no_spot (total_cows : ℕ) (percent_red_spot : ℚ) (percent_blue_spot : ℚ) :
  total_cows = 140 ∧ percent_red_spot = 0.40 ∧ percent_blue_spot = 0.25 → 
  ∃ (no_spot_cows : ℕ), no_spot_cows = 63 :=
by 
  sorry

end cows_with_no_spot_l99_99920


namespace angle_ratio_l99_99436

-- Definitions as per the conditions
def bisects (x y z : ℝ) : Prop := x = y / 2
def trisects (x y z : ℝ) : Prop := y = x / 3

theorem angle_ratio (ABC PBQ BM x : ℝ) (h1 : bisects PBQ ABC PQ)
                                    (h2 : trisects PBQ BM M) :
  PBQ = 2 * x →
  PBQ = ABC / 2 →
  MBQ = x →
  ABQ = 4 * x →
  MBQ / ABQ = 1 / 4 :=
by
  intros
  sorry

end angle_ratio_l99_99436


namespace dave_bought_26_tshirts_l99_99380

def total_tshirts :=
  let white_tshirts := 3 * 6
  let blue_tshirts := 2 * 4
  white_tshirts + blue_tshirts

theorem dave_bought_26_tshirts : total_tshirts = 26 :=
by
  unfold total_tshirts
  have white_tshirts : 3 * 6 = 18 := by norm_num
  have blue_tshirts : 2 * 4 = 8 := by norm_num
  rw [white_tshirts, blue_tshirts]
  norm_num

end dave_bought_26_tshirts_l99_99380


namespace production_difference_l99_99669

theorem production_difference (w t : ℕ) (h1 : w = 3 * t) :
  (w * t) - ((w + 6) * (t - 3)) = 3 * t + 18 :=
by
  sorry

end production_difference_l99_99669


namespace derivative_of_f_eval_deriv_at_pi_over_6_l99_99725

noncomputable def f (x : Real) : Real := (Real.sin x) ^ 4 + (Real.cos x) ^ 4

theorem derivative_of_f : ∀ x, deriv f x = -Real.sin (4 * x) :=
by
  intro x
  sorry

theorem eval_deriv_at_pi_over_6 : deriv f (Real.pi / 6) = -Real.sqrt 3 / 2 :=
by
  rw [derivative_of_f]
  sorry

end derivative_of_f_eval_deriv_at_pi_over_6_l99_99725


namespace Sarah_collected_40_today_l99_99820

noncomputable def Sarah_yesterday : ℕ := 50
noncomputable def Lara_yesterday : ℕ := Sarah_yesterday + 30
noncomputable def Lara_today : ℕ := 70
noncomputable def Total_yesterday : ℕ := Sarah_yesterday + Lara_yesterday
noncomputable def Total_today : ℕ := Total_yesterday - 20
noncomputable def Sarah_today : ℕ := Total_today - Lara_today

theorem Sarah_collected_40_today : Sarah_today = 40 := 
by
  sorry

end Sarah_collected_40_today_l99_99820


namespace shanghai_masters_total_matches_l99_99034

theorem shanghai_masters_total_matches : 
  let players := 8
  let groups := 2
  let players_per_group := 4
  let round_robin_matches_per_group := (players_per_group * (players_per_group - 1)) / 2
  let round_robin_total_matches := round_robin_matches_per_group * groups
  let elimination_matches := 2 * (groups - 1)  -- semi-final matches
  let final_matches := 2  -- one final and one third-place match
  round_robin_total_matches + elimination_matches + final_matches = 16 :=
by
  sorry

end shanghai_masters_total_matches_l99_99034


namespace archers_in_golden_armor_count_l99_99763

theorem archers_in_golden_armor_count : 
  ∃ (archers_golden : ℕ), 
    let total_soldiers := 55 in
    let golden_armor_affirmative := 44 in
    let archer_affirmative := 33 in
    let monday_affirmative := 22 in
    let number_archers_golden := archers_golden in
    (number_archers_golden = 22) ∧ 
    ∃ (archer : ℕ) (swordsman : ℕ) (golden : ℕ) (black : ℕ),
      archer + swordsman = total_soldiers ∧
      golden + black = total_soldiers ∧
      golden + black = total_soldiers ∧
      (swordsman * golden + archer * golden = golden_armor_affirmative) ∧
      (swordsman * archer + swordsman * golden = archer_affirmative) ∧
      ((!monday_affirmative ∧ swordsman * golden) + (!monday_affirmative ∧ archer * black) = monday_affirmative)
:= sorry

end archers_in_golden_armor_count_l99_99763


namespace ratio_of_intercepts_l99_99326

variable {c : ℝ} (non_zero_c : c ≠ 0) (u v : ℝ)
-- Condition: The first line, slope 8, y-intercept c, x-intercept (u, 0)
variable (h_u : u = -c / 8)
-- Condition: The second line, slope 4, y-intercept c, x-intercept (v, 0)
variable (h_v : v = -c / 4)

theorem ratio_of_intercepts (non_zero_c : c ≠ 0)
    (h_u : u = -c / 8) (h_v : v = -c / 4) : u / v = 1 / 2 :=
by
  sorry

end ratio_of_intercepts_l99_99326


namespace binom_15_13_eq_105_l99_99374

theorem binom_15_13_eq_105 : nat.choose 15 13 = 105 :=
by
sorry

end binom_15_13_eq_105_l99_99374


namespace probability_exact_n_points_l99_99632

open Classical

noncomputable def probability_of_n_points (n : ℕ) : ℚ :=
  1/3 * (2 + (-1/2)^n)

theorem probability_exact_n_points (n : ℕ) :
  ∀ n : ℕ, probability_of_n_points n = 1/3 * (2 + (-1/2)^n) :=
sorry

end probability_exact_n_points_l99_99632


namespace russian_pairing_probability_l99_99929

-- Definitions based on conditions
def total_players : ℕ := 10
def russian_players : ℕ := 4
def non_russian_players : ℕ := total_players - russian_players

-- Probability calculation as a hypothesis
noncomputable def pairing_probability (rs: ℕ) (ns: ℕ) : ℚ :=
  (rs * (rs - 1)) / (total_players * (total_players - 1))

theorem russian_pairing_probability :
  pairing_probability russian_players non_russian_players = 1 / 21 :=
sorry

end russian_pairing_probability_l99_99929


namespace jordan_probability_l99_99472

-- Definitions based on conditions.
def total_students := 28
def enrolled_in_french := 20
def enrolled_in_spanish := 23
def enrolled_in_both := 17

-- Calculate students enrolled only in one language.
def only_french := enrolled_in_french - enrolled_in_both
def only_spanish := enrolled_in_spanish - enrolled_in_both

-- Calculation of combinations.
def total_combinations := Nat.choose total_students 2
def only_french_combinations := Nat.choose only_french 2
def only_spanish_combinations := Nat.choose only_spanish 2

-- Probability calculations.
def prob_both_one_language := (only_french_combinations + only_spanish_combinations) / total_combinations

def prob_both_languages : ℚ := 1 - prob_both_one_language

theorem jordan_probability :
  prob_both_languages = (20 : ℚ) / 21 := by
  sorry

end jordan_probability_l99_99472


namespace problem_solution_l99_99312

noncomputable def f (x a : ℝ) : ℝ :=
  2 * (Real.cos x)^2 - 2 * a * Real.cos x - (2 * a + 1)

noncomputable def g (a : ℝ) : ℝ :=
  if a < -2 then 1
  else if a < 2 then -a^2 / 2 - 2 * a - 1
  else 1 - 4 * a

theorem problem_solution :
  g a = 1 ∨ g a = (-a^2 / 2 - 2 * a - 1) ∨ g a = 1 - 4 * a →
  (∀ a, g a = 1 / 2 → a = -1) ∧ (f x (-1) ≤ 5) :=
sorry

end problem_solution_l99_99312


namespace smallest_positive_integer_for_terminating_decimal_l99_99160

theorem smallest_positive_integer_for_terminating_decimal (n : ℕ) (h1 : n > 0) (h2 : (∀ m : ℕ, (m > n + 150) -> (m % (n + 150)) ∉ {2, 5})) :
  n = 50 :=
sorry

end smallest_positive_integer_for_terminating_decimal_l99_99160


namespace avg_people_moving_to_florida_per_hour_l99_99767

theorem avg_people_moving_to_florida_per_hour (people : ℕ) (days : ℕ) (hours_per_day : ℕ) 
  (h1 : people = 3000) (h2 : days = 5) (h3 : hours_per_day = 24) : 
  people / (days * hours_per_day) = 25 := by
  sorry

end avg_people_moving_to_florida_per_hour_l99_99767


namespace sin_double_angle_l99_99236

theorem sin_double_angle (α : ℝ) (h1 : α ∈ Set.Ioo (π / 2) π) (h2 : Real.sin α = 4 / 5) : Real.sin (2 * α) = -24 / 25 :=
by
  sorry

end sin_double_angle_l99_99236


namespace problem1_problem2_l99_99217

theorem problem1 : 
  -(3^3) * ((-1 : ℚ)/ 3)^2 - 24 * (3/4 - 1/6 + 3/8) = -26 := 
by 
  sorry

theorem problem2 : 
  -(1^100 : ℚ) - (3/4) / (((-2)^2) * ((-1 / 4) ^ 2) - 1 / 2) = 2 := 
by 
  sorry

end problem1_problem2_l99_99217


namespace problem_solution_l99_99399

theorem problem_solution (x y : ℝ) 
  (hx : x ≠ 0) 
  (hy : y ≠ 0) 
  (h : x - y = x / y) : 
  (1 / x - 1 / y = -1 / y^2) := 
by sorry

end problem_solution_l99_99399


namespace sum_of_m_and_n_l99_99417

theorem sum_of_m_and_n (m n : ℚ) (h : (m - 3) * (Real.sqrt 5) + 2 - n = 0) : m + n = 5 :=
sorry

end sum_of_m_and_n_l99_99417


namespace surface_area_of_cube_given_sphere_surface_area_l99_99892

noncomputable def edge_length_of_cube (sphere_surface_area : ℝ) : ℝ :=
  let a_square := 2
  Real.sqrt a_square

def surface_area_of_cube (a : ℝ) : ℝ :=
  6 * a^2

theorem surface_area_of_cube_given_sphere_surface_area (sphere_surface_area : ℝ) :
  sphere_surface_area = 6 * Real.pi → 
  surface_area_of_cube (edge_length_of_cube sphere_surface_area) = 12 :=
by
  sorry

end surface_area_of_cube_given_sphere_surface_area_l99_99892


namespace binomial_expansion_fifth_term_constant_l99_99891

open Classical -- Allows the use of classical logic

noncomputable def binomial_term (n r : ℕ) (x : ℝ) : ℝ :=
  (Nat.choose n r) * (x ^ (n - r) / (x ^ r * (2 ^ r / x ^ r)))

theorem binomial_expansion_fifth_term_constant (n : ℕ) :
  (binomial_term n 4 x = (x ^ (n - 3 * 4) * (-2) ^ 4)) → n = 12 := by
  intro h
  sorry

end binomial_expansion_fifth_term_constant_l99_99891


namespace net_population_change_l99_99495

theorem net_population_change (P : ℝ) : 
  let P1 := P * (6/5)
  let P2 := P1 * (7/10)
  let P3 := P2 * (6/5)
  let P4 := P3 * (7/10)
  (P4 / P - 1) * 100 = -29 := 
by
  sorry

end net_population_change_l99_99495


namespace total_selling_price_l99_99981

theorem total_selling_price (CP : ℕ) (num_toys : ℕ) (gain_toys : ℕ) (TSP : ℕ)
  (h1 : CP = 1300)
  (h2 : num_toys = 18)
  (h3 : gain_toys = 3) :
  TSP = 27300 := by
  sorry

end total_selling_price_l99_99981


namespace not_possible_coloring_l99_99773

def color : Nat → Option ℕ := sorry

def all_colors_used (f : Nat → Option ℕ) : Prop := 
  (∃ n, f n = some 0) ∧ (∃ n, f n = some 1) ∧ (∃ n, f n = some 2)

def valid_coloring (f : Nat → Option ℕ) : Prop :=
  ∀ (a b : Nat), 1 < a → 1 < b → f a ≠ f b → f (a * b) ≠ f a ∧ f (a * b) ≠ f b

theorem not_possible_coloring : ¬ (∃ f : Nat → Option ℕ, all_colors_used f ∧ valid_coloring f) := 
sorry

end not_possible_coloring_l99_99773


namespace rotate_90deg_l99_99344

def Shape := Type

structure Figure :=
(triangle : Shape)
(circle : Shape)
(square : Shape)
(pentagon : Shape)

def rotated_position (fig : Figure) : Figure :=
{ triangle := fig.circle,
  circle := fig.square,
  square := fig.pentagon,
  pentagon := fig.triangle }

theorem rotate_90deg (fig : Figure) :
  rotated_position fig = { triangle := fig.circle,
                           circle := fig.square,
                           square := fig.pentagon,
                           pentagon := fig.triangle } :=
by {
  sorry
}

end rotate_90deg_l99_99344


namespace jenna_eel_length_l99_99447

theorem jenna_eel_length (J B L : ℝ)
  (h1 : J = (2 / 5) * B)
  (h2 : J = (3 / 7) * L)
  (h3 : J + B + L = 124) : 
  J = 21 := 
sorry

end jenna_eel_length_l99_99447


namespace max_value_of_a_l99_99239

noncomputable def max_a : ℝ :=
  6 - 6 * Real.log 6

theorem max_value_of_a :
  ∀ a : ℝ, (∀ k : ℝ, -1 ≤ k ∧ k ≤ 1 → ∀ x : ℝ, 0 < x ∧ x ≤ 6 →
  6 * Real.log x + x^2 - 8 * x + a ≤ k * x) → a ≤ max_a :=
begin
  sorry
end

end max_value_of_a_l99_99239


namespace dan_initial_amount_l99_99026

theorem dan_initial_amount (left_amount : ℕ) (candy_cost : ℕ) : left_amount = 3 ∧ candy_cost = 2 → left_amount + candy_cost = 5 :=
by
  sorry

end dan_initial_amount_l99_99026


namespace problem_statement_l99_99372

noncomputable def count_propositions_and_true_statements 
  (statements : List String)
  (is_proposition : String → Bool)
  (is_true_proposition : String → Bool) 
  : Nat × Nat :=
  let props := statements.filter is_proposition
  let true_props := props.filter is_true_proposition
  (props.length, true_props.length)

theorem problem_statement : 
  (count_propositions_and_true_statements 
     ["Isn't an equilateral triangle an isosceles triangle?",
      "Are two lines perpendicular to the same line necessarily parallel?",
      "A number is either positive or negative",
      "What a beautiful coastal city Zhuhai is!",
      "If x + y is a rational number, then x and y are also rational numbers",
      "Construct △ABC ∼ △A₁B₁C₁"]
     (fun s => 
        s = "A number is either positive or negative" ∨ 
        s = "If x + y is a rational number, then x and y are also rational numbers")
     (fun s => false))
  = (2, 0) :=
by
  sorry

end problem_statement_l99_99372


namespace compute_abs_ab_eq_2_sqrt_111_l99_99935

theorem compute_abs_ab_eq_2_sqrt_111 (a b : ℝ) 
  (h1 : b^2 - a^2 = 25)
  (h2 : a^2 + b^2 = 49) : 
  |a * b| = 2 * Real.sqrt 111 := 
sorry

end compute_abs_ab_eq_2_sqrt_111_l99_99935


namespace smallest_four_digit_multiple_of_18_l99_99683

theorem smallest_four_digit_multiple_of_18 : ∃ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ 18 ∣ n ∧ ∀ m : ℕ, 1000 ≤ m ∧ m ≤ 9999 ∧ 18 ∣ m → n ≤ m := by
  use 1008
  sorry

end smallest_four_digit_multiple_of_18_l99_99683


namespace sum_of_star_tips_l99_99466

/-- Given ten points that are evenly spaced on a circle and connected to form a 10-pointed star,
prove that the sum of the angle measurements of the ten tips of the star is 720 degrees. -/
theorem sum_of_star_tips (n : ℕ) (h : n = 10) :
  (10 * 72 = 720) :=
by
  sorry

end sum_of_star_tips_l99_99466


namespace Travis_spends_on_cereal_l99_99132

theorem Travis_spends_on_cereal (boxes_per_week : ℕ) (cost_per_box : ℝ) (weeks_per_year : ℕ) 
  (h1 : boxes_per_week = 2) 
  (h2 : cost_per_box = 3.00) 
  (h3 : weeks_per_year = 52) 
: boxes_per_week * weeks_per_year * cost_per_box = 312.00 := 
by
  sorry

end Travis_spends_on_cereal_l99_99132


namespace find_angle_degree_l99_99805

theorem find_angle_degree (x : ℝ) (h : 90 - x = (1 / 3) * (180 - x) + 20) : x = 75 := by
    sorry

end find_angle_degree_l99_99805


namespace dante_coconuts_left_l99_99791

variable (Paolo : ℕ) (Dante : ℕ)

theorem dante_coconuts_left :
  Paolo = 14 →
  Dante = 3 * Paolo →
  Dante - 10 = 32 :=
by
  intros hPaolo hDante
  rw [hPaolo, hDante]
  sorry

end dante_coconuts_left_l99_99791


namespace total_birds_correct_l99_99527

-- Define the conditions
def number_of_trees : ℕ := 7
def blackbirds_per_tree : ℕ := 3
def magpies : ℕ := 13

-- Define the total number of blackbirds using the conditions
def total_blackbirds : ℕ := number_of_trees * blackbirds_per_tree

-- Define the total number of birds using the total number of blackbirds and the number of magpies
def total_birds : ℕ := total_blackbirds + magpies

-- The theorem statement that should be proven
theorem total_birds_correct : total_birds = 34 := 
sorry

end total_birds_correct_l99_99527


namespace solve_equation_l99_99111

theorem solve_equation (x : ℝ) : 
  (x + 1) / 6 = 4 / 3 - x ↔ x = 1 :=
sorry

end solve_equation_l99_99111


namespace probability_red_ball_is_correct_l99_99667

noncomputable def probability_red_ball : ℚ :=
  let prob_A := 1 / 3
  let prob_B := 1 / 3
  let prob_C := 1 / 3
  let prob_red_A := 3 / 10
  let prob_red_B := 7 / 10
  let prob_red_C := 5 / 11
  (prob_A * prob_red_A) + (prob_B * prob_red_B) + (prob_C * prob_red_C)

theorem probability_red_ball_is_correct : probability_red_ball = 16 / 33 := 
by
  sorry

end probability_red_ball_is_correct_l99_99667


namespace probability_of_green_ball_is_2_over_5_l99_99377

noncomputable def container_probabilities : ℚ :=
  let prob_A_selected : ℚ := 1/2
  let prob_B_selected : ℚ := 1/2
  let prob_green_in_A : ℚ := 5/10
  let prob_green_in_B : ℚ := 3/10

  prob_A_selected * prob_green_in_A + prob_B_selected * prob_green_in_B

theorem probability_of_green_ball_is_2_over_5 :
  container_probabilities = 2 / 5 := by
  sorry

end probability_of_green_ball_is_2_over_5_l99_99377


namespace sum_of_roots_l99_99886

theorem sum_of_roots (x1 x2 : ℝ) (h : x1 * x2 = -3) (hx1 : x1 + x2 = 2) :
  x1 + x2 = 2 :=
by {
  sorry
}

end sum_of_roots_l99_99886


namespace problem1_problem2_problem3_l99_99274

noncomputable def Sn (n : ℕ) : ℝ := sorry  -- Sum of the first n terms of the sequence {a_n}
noncomputable def an (n : ℕ) : ℝ := sorry  -- nth term of the sequence {a_n}
noncomputable def xn (n : ℕ) : ℝ := sorry  -- The other root of the equation x^2 - a_n x - a_n = 0
noncomputable def Tn (n : ℕ) : ℝ := sorry  -- Sum of the first n terms of the sequence \( \frac{1}{2^n x_n} \)

-- Condition: The equation x^2 - a_n x - a_n = 0 has a root S_n - 1
axiom eq1 (n : ℕ) (hn : 1 ≤ n) : (Sn n - 1) ^ 2 - an n * (Sn n - 1) - an n = 0

-- Problem 1: Prove that { \frac{1}{S_n - 1} } is an arithmetic sequence
theorem problem1 : ∃ a d : ℝ, ∀ n : ℕ, 1 ≤ n → 1 / (Sn n - 1) = a + d * (n - 1) := sorry

-- Problem 2: Find 2^2013 * (2 - T_2013)
theorem problem2 : 2 ^ 2013 * (2 - Tn 2013) = 2015 := sorry

-- Problem 3: Prove the existence of positive integers p and q such that S_1, S_p, and S_q form a geometric sequence
theorem problem3 : ∃ p q : ℕ, 1 ≤ p ∧ 1 ≤ q ∧ p ≠ q ∧ (Sn p)^2 = (Sn 1) * (Sn q) :=
  begin
    use [2, 8],
    sorry
  end

end problem1_problem2_problem3_l99_99274


namespace probability_no_obtuse_triangle_is_9_over_64_l99_99705

noncomputable def probability_no_obtuse_triangle (A0 A1 A2 A3 : ℝ) : ℝ :=
  -- Define the probabilistic model according to the problem conditions
  -- Assuming A0, A1, A2, and A3 are positions of points on the circle parametrized by angles in radians
  let θ := real.angle.lower_half (A1 - A0) in
  let prob_A2 := (π - θ) / (2 * π) in
  let prob_A3 := (π - θ) / (2 * π) in
  (1 / 2) * (prob_A2 * prob_A3)

theorem probability_no_obtuse_triangle_is_9_over_64 :
  probability_no_obtuse_triangle A0 A1 A2 A3 = 9 / 64 :=
by sorry

end probability_no_obtuse_triangle_is_9_over_64_l99_99705


namespace cos_squared_sum_sin_squared_sum_l99_99771

theorem cos_squared_sum (A B C : ℝ) (h : A + B + C = Real.pi) :
  Real.cos (A / 2) ^ 2 + Real.cos (B / 2) ^ 2 + Real.cos (C / 2) ^ 2 =
  2 * (1 + Real.sin (A / 2) * Real.sin (B / 2) * Real.sin (C / 2)) :=
sorry

theorem sin_squared_sum (A B C : ℝ) (h : A + B + C = Real.pi) :
  Real.sin (A / 2) ^ 2 + Real.sin (B / 2) ^ 2 + Real.sin (C / 2) ^ 2 =
  1 - 2 * Real.sin (A / 2) * Real.sin (B / 2) * Real.sin (C / 2) :=
sorry

end cos_squared_sum_sin_squared_sum_l99_99771


namespace marble_probability_l99_99968

theorem marble_probability :
  let total_marbles : ℕ := 20
  let red_marbles : ℕ := 12
  let blue_marbles : ℕ := 8
  let total_probability : ℚ := (6 * ((12 * 11 * 8 * 7) / (20 * 19 * 18 * 17))) in
  total_probability = (1232 / 4845) :=
by
  sorry

end marble_probability_l99_99968


namespace vector_dot_product_result_l99_99069

variable {α : Type*} [Field α]

structure Vector2 (α : Type*) :=
(x : α)
(y : α)

def vector_add (a b : Vector2 α) : Vector2 α :=
  ⟨a.x + b.x, a.y + b.y⟩

def vector_sub (a b : Vector2 α) : Vector2 α :=
  ⟨a.x - b.x, a.y - b.y⟩

def dot_product (a b : Vector2 α) : α :=
  a.x * b.x + a.y * b.y

variable (a b : Vector2 ℝ)

theorem vector_dot_product_result
  (h1 : vector_add a b = ⟨1, -3⟩)
  (h2 : vector_sub a b = ⟨3, 7⟩) :
  dot_product a b = -12 :=
by
  sorry

end vector_dot_product_result_l99_99069


namespace room_dimension_l99_99806

theorem room_dimension
  (x : ℕ)
  (cost_per_sqft : ℕ := 4)
  (dimension_1 : ℕ := 15)
  (dimension_2 : ℕ := 12)
  (door_width : ℕ := 6)
  (door_height : ℕ := 3)
  (num_windows : ℕ := 3)
  (window_width : ℕ := 4)
  (window_height : ℕ := 3)
  (total_cost : ℕ := 3624) :
  (2 * (x * dimension_1) + 2 * (x * dimension_2) - (door_width * door_height + num_windows * (window_width * window_height))) * cost_per_sqft = total_cost →
  x = 18 :=
by
  sorry

end room_dimension_l99_99806


namespace turnip_weight_possible_l99_99999

-- Define the weights of the 6 bags
def bag_weights : List ℕ := [13, 15, 16, 17, 21, 24]

-- Define the weight of the turnip bag
def is_turnip_bag (T : ℕ) : Prop :=
  T ∈ bag_weights ∧
  ∃ O : ℕ, 3 * O = (bag_weights.sum - T)

theorem turnip_weight_possible : ∀ T, is_turnip_bag T ↔ T = 13 ∨ T = 16 :=
by sorry

end turnip_weight_possible_l99_99999


namespace smallest_n_term_dec_l99_99153

theorem smallest_n_term_dec (n : ℕ) (h_pos : 0 < n) (h : ∀ d, 0 < d → d = n + 150 → ∀ p, p ∣ d → (p = 2 ∨ p = 5)) :
  n = 10 :=
by {
  sorry
}

end smallest_n_term_dec_l99_99153


namespace odd_function_at_zero_l99_99724

-- Define the property of being an odd function
def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f (x)

theorem odd_function_at_zero (f : ℝ → ℝ) (h : is_odd_function f) : f 0 = 0 :=
by
  -- assume the definitions but leave the proof steps and focus on the final conclusion
  sorry

end odd_function_at_zero_l99_99724


namespace tech_gadget_cost_inr_l99_99462

def conversion_ratio (a b : ℝ) : Prop := a = b

theorem tech_gadget_cost_inr :
  (forall a b c : ℝ, conversion_ratio (a / b) c) →
  (forall a b c d : ℝ, conversion_ratio (a / b) c → conversion_ratio (a / d) c) →
  ∀ (n_usd : ℝ) (n_inr : ℝ) (cost_n : ℝ), 
    n_usd = 8 →
    n_inr = 5 →
    cost_n = 160 →
    cost_n / n_usd * n_inr = 100 :=
by
  sorry

end tech_gadget_cost_inr_l99_99462


namespace log_expression_as_product_l99_99862

noncomputable def log (b : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log b

theorem log_expression_as_product (A m n p : ℝ) (hm : 0 < m) (hn : 0 < n) (hp : 0 < p) (hA : 0 < A) :
  log m A * log n A + log n A * log p A + log p A * log m A =
  log A (m * n * p) * log p A * log n A * log m A :=
by
  sorry

end log_expression_as_product_l99_99862


namespace line_through_parabola_no_intersection_l99_99270

-- Definitions of the conditions
def parabola (x : ℝ) : ℝ := x^2 
def point_Q := (10, 5)

-- The main theorem statement
theorem line_through_parabola_no_intersection :
  ∃ r s : ℝ, (∀ (m : ℝ), (r < m ∧ m < s) ↔ ¬ ∃ x : ℝ, parabola x = m * (x - 10) + 5) ∧ r + s = 40 :=
sorry

end line_through_parabola_no_intersection_l99_99270


namespace minimum_shift_value_l99_99748

noncomputable def f : ℝ → ℝ := λ x, Real.sin (1 / 2 * x)
noncomputable def g : ℝ → ℝ := λ x, Real.cos (1 / 2 * x)

theorem minimum_shift_value (ϕ : ℝ) (hϕ : ϕ > 0) (h : ∀ x, f (x + ϕ) = g x) : ϕ = π := by
  have h1 : ∀ x, Real.sin (1 / 2 * (x + ϕ)) = Real.cos (1 / 2 * x), from h
  sorry

end minimum_shift_value_l99_99748


namespace sum_of_numerator_and_denominator_of_repeating_decimal_l99_99342

theorem sum_of_numerator_and_denominator_of_repeating_decimal :
  let x := 0.45
  let a := 9 -- GCD of 45 and 99
  let numerator := 5
  let denominator := 11
  numerator + denominator = 16 :=
by { 
  sorry 
}

end sum_of_numerator_and_denominator_of_repeating_decimal_l99_99342


namespace turnip_bag_weighs_l99_99982

theorem turnip_bag_weighs (bags : List ℕ) (T : ℕ)
  (h_weights : bags = [13, 15, 16, 17, 21, 24])
  (h_turnip : T ∈ bags)
  (h_carrot_onion_relation : ∃ O C: ℕ, C = 2 * O ∧ C + O = 106 - T) :
  T = 13 ∨ T = 16 := by
  sorry

end turnip_bag_weighs_l99_99982


namespace multiple_of_four_l99_99453

open BigOperators

theorem multiple_of_four (n : ℕ) (x y z : Fin n → ℤ)
  (hx : ∀ i, x i = 1 ∨ x i = -1)
  (hy : ∀ i, y i = 1 ∨ y i = -1)
  (hz : ∀ i, z i = 1 ∨ z i = -1)
  (hxy : ∑ i, x i * y i = 0)
  (hxz : ∑ i, x i * z i = 0)
  (hyz : ∑ i, y i * z i = 0) :
  (n % 4 = 0) :=
sorry

end multiple_of_four_l99_99453


namespace compute_g_five_times_l99_99589

def g (x : ℤ) : ℤ :=
  if x ≥ 0 then -x^3 else x + 6

theorem compute_g_five_times :
  g (g (g (g (g 1)))) = -113 :=
  by sorry

end compute_g_five_times_l99_99589


namespace max_tan_B_triangle_l99_99441

open Real
open EuclideanGeometry

theorem max_tan_B_triangle (A B C : Point) (hAB : dist A B = 25) (hBC : dist B C = 20) (hRightAngle : angle C A B = π / 2) :  
  tan (angle A B C) = 3 / 4 :=
by
  sorry

end max_tan_B_triangle_l99_99441


namespace number_of_archers_in_golden_armor_l99_99754

-- Define the problem context
structure Soldier where
  is_archer : Bool
  wears_golden_armor : Bool

def truth_teller (s : Soldier) (is_black_armor : Bool) : Bool :=
  if s.is_archer then s.wears_golden_armor = is_black_armor
  else s.wears_golden_armor ≠ is_black_armor

def response (s : Soldier) (question : String) (is_black_armor : Bool) : Bool :=
  match question with
  | "Are you wearing golden armor?" => if truth_teller s is_black_armor then s.wears_golden_armor else ¬s.wears_golden_armor
  | "Are you an archer?" => if truth_teller s is_black_armor then s.is_archer else ¬s.is_archer
  | "Is today Monday?" => if truth_teller s is_black_armor then True else False -- An assumption that today not being Monday means False
  | _ => False

-- Problem condition setup
def total_soldiers : Nat := 55
def golden_armor_yes : Nat := 44
def archer_yes : Nat := 33
def monday_yes : Nat := 22

-- Define the main theorem
theorem number_of_archers_in_golden_armor :
  ∃ l : List Soldier,
    l.length = total_soldiers ∧
    l.countp (λ s => response s "Are you wearing golden armor?" True) = golden_armor_yes ∧
    l.countp (λ s => response s "Are you an archer?" True) = archer_yes ∧
    l.countp (λ s => response s "Is today Monday?" True) = monday_yes ∧
    l.countp (λ s => s.is_archer ∧ s.wears_golden_armor) = 22 :=
sorry

end number_of_archers_in_golden_armor_l99_99754


namespace find_x_to_be_2_l99_99569

variable (x : ℝ)

def a := (2, x)
def b := (3, x + 1)

theorem find_x_to_be_2 (h : a x = b x) : x = 2 := by
  sorry

end find_x_to_be_2_l99_99569


namespace smallest_n_terminating_decimal_l99_99165

theorem smallest_n_terminating_decimal :
  ∃ n : ℕ, (∀ m : ℕ, (∃ k : ℕ, m + 150 = 2^k * (5^0))
  → (m ≥ n)) ∧ (n > 0) ∧ (∃ k : ℕ, n + 150 = 2^k * (5^0)) := 
sorry

end smallest_n_terminating_decimal_l99_99165


namespace find_c_l99_99392

-- Define c and the floor function
def c : ℝ := 13.1

theorem find_c (h : c + ⌊c⌋ = 25.6) : c = 13.1 :=
sorry

end find_c_l99_99392


namespace range_of_a_l99_99567

noncomputable def f (a x : ℝ) : ℝ := x * Real.exp x + (1 / 2) * a * x^2 + a * x

theorem range_of_a (a : ℝ) : 
    (∀ x : ℝ, 2 * Real.exp (f a x) + Real.exp 1 + 2 ≥ 0) ↔ (0 ≤ a ∧ a ≤ 1) := 
sorry

end range_of_a_l99_99567


namespace find_prime_n_l99_99184

theorem find_prime_n (n k m : ℤ) (h1 : n - 6 = k ^ 2) (h2 : n + 10 = m ^ 2) (h3 : m ^ 2 - k ^ 2 = 16) (h4 : Nat.Prime (Int.natAbs n)) : n = 71 := by
  sorry

end find_prime_n_l99_99184


namespace flour_already_added_l99_99101

theorem flour_already_added (sugar flour salt additional_flour : ℕ) 
  (h1 : sugar = 9) 
  (h2 : flour = 14) 
  (h3 : salt = 40)
  (h4 : additional_flour = sugar + 1) : 
  flour - additional_flour = 4 :=
by
  sorry

end flour_already_added_l99_99101


namespace length_of_each_piece_l99_99071

-- Definitions based on conditions
def total_length : ℝ := 42.5
def number_of_pieces : ℝ := 50

-- The statement that we need to prove
theorem length_of_each_piece (h1 : total_length = 42.5) (h2 : number_of_pieces = 50) : 
  total_length / number_of_pieces = 0.85 := 
by
  sorry

end length_of_each_piece_l99_99071


namespace total_spider_legs_l99_99522

-- Define the number of legs per spider.
def legs_per_spider : ℕ := 8

-- Define half of the legs per spider.
def half_legs : ℕ := legs_per_spider / 2

-- Define the number of spiders in the group.
def num_spiders : ℕ := half_legs + 10

-- Prove the total number of spider legs in the group is 112.
theorem total_spider_legs : num_spiders * legs_per_spider = 112 := by
  -- Use 'sorry' to skip the detailed proof steps.
  sorry

end total_spider_legs_l99_99522


namespace sin_value_l99_99234

theorem sin_value (theta : ℝ) (h : Real.cos (3 * Real.pi / 14 - theta) = 1 / 3) : 
  Real.sin (2 * Real.pi / 7 + theta) = 1 / 3 :=
by
  -- Sorry replaces the actual proof which is not required for this task
  sorry

end sin_value_l99_99234


namespace smallest_positive_integer_for_terminating_decimal_l99_99179

theorem smallest_positive_integer_for_terminating_decimal: ∃ n: ℕ, (n > 0) ∧ (∀ p : ℕ, (p ∣ (n + 150)) → (p=1 ∨ p=2 ∨ p=4 ∨ p=5 ∨ p=8 ∨ p=10 ∨ p=16 ∨ p=20 ∨ p=25 ∨ p=32 ∨ p=40 ∨ p=50 ∨ p=64 ∨ p=80 ∨ p=100 ∨ p=125 ∨ p=128 ∨ p=160)) ∧ n = 10 :=
by
  sorry

end smallest_positive_integer_for_terminating_decimal_l99_99179


namespace combined_transform_is_correct_l99_99394

def dilation_matrix (k : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  !![k, 0; 0, k]

def reflection_x_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  !![1, 0; 0, -1]

def combined_transform (dilation_factor : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  dilation_matrix dilation_factor * reflection_x_matrix

theorem combined_transform_is_correct :
  combined_transform 5 = !![5, 0; 0, -5] :=
by
  sorry

end combined_transform_is_correct_l99_99394


namespace no_obtuse_triangle_probability_l99_99700

noncomputable def probability_no_obtuse_triangle : ℝ :=
 let θ := 1/2 in
 let prob_A2_A3_given_A0A1 := (3/8) * (3/8) in
 θ * prob_A2_A3_given_A0A1

theorem no_obtuse_triangle_probability :
  probability_no_obtuse_triangle = 9/128 :=
by
  sorry

end no_obtuse_triangle_probability_l99_99700


namespace segment_combination_l99_99261

theorem segment_combination (x y : ℕ) :
  7 * x + 12 * y = 100 ↔ (x, y) = (4, 6) :=
by
  sorry

end segment_combination_l99_99261


namespace second_quadrant_implies_value_of_m_l99_99902

theorem second_quadrant_implies_value_of_m (m : ℝ) : 4 - m < 0 → m = 5 := by
  intro h
  have ineq : m > 4 := by
    linarith
  sorry

end second_quadrant_implies_value_of_m_l99_99902


namespace solve_firm_problem_l99_99954

def firm_problem : Prop :=
  ∃ (P A : ℕ), 
    (P / A = 2 / 63) ∧ 
    (P / (A + 50) = 1 / 34) ∧ 
    (P = 20)

theorem solve_firm_problem : firm_problem :=
  sorry

end solve_firm_problem_l99_99954


namespace time_45_minutes_after_10_20_is_11_05_l99_99953

def time := Nat × Nat -- Represents time as (hours, minutes)

noncomputable def add_minutes (t : time) (m : Nat) : time :=
  let (hours, minutes) := t
  let total_minutes := minutes + m
  let new_hours := hours + total_minutes / 60
  let new_minutes := total_minutes % 60
  (new_hours, new_minutes)

theorem time_45_minutes_after_10_20_is_11_05 :
  add_minutes (10, 20) 45 = (11, 5) :=
  sorry

end time_45_minutes_after_10_20_is_11_05_l99_99953


namespace misha_is_older_l99_99802

-- Definitions for the conditions
def tanya_age_19_months_ago : ℕ := 16
def months_ago_for_tanya : ℕ := 19
def misha_age_in_16_months : ℕ := 19
def months_ahead_for_misha : ℕ := 16

-- Convert months to years and residual months
def months_to_years_months (m : ℕ) : ℕ × ℕ := (m / 12, m % 12)

-- Computation for Tanya's current age
def tanya_age_now : ℕ × ℕ :=
  let (years, months) := months_to_years_months months_ago_for_tanya
  (tanya_age_19_months_ago + years, months)

-- Computation for Misha's current age
def misha_age_now : ℕ × ℕ :=
  let (years, months) := months_to_years_months months_ahead_for_misha
  (misha_age_in_16_months - years, months)

-- Proof statement
theorem misha_is_older : misha_age_now > tanya_age_now := by
  sorry

end misha_is_older_l99_99802


namespace next_consecutive_time_l99_99832

theorem next_consecutive_time (current_hour : ℕ) (current_minute : ℕ) 
  (valid_minutes : 0 ≤ current_minute ∧ current_minute < 60) 
  (valid_hours : 0 ≤ current_hour ∧ current_hour < 24) : 
  current_hour = 4 ∧ current_minute = 56 →
  ∃ next_hour next_minute : ℕ, 
    (0 ≤ next_minute ∧ next_minute < 60) ∧ 
    (0 ≤ next_hour ∧ next_hour < 24) ∧
    (next_hour, next_minute) = (12, 34) ∧ 
    (next_hour * 60 + next_minute) - (current_hour * 60 + current_minute) = 458 := 
by sorry

end next_consecutive_time_l99_99832


namespace max_modulus_z_i_l99_99243

open Complex

theorem max_modulus_z_i (z : ℂ) (hz : abs z = 2) : ∃ z₂ : ℂ, abs z₂ = 2 ∧ abs (z₂ - I) = 3 :=
sorry

end max_modulus_z_i_l99_99243


namespace sum_of_squares_and_product_l99_99493

open Real

theorem sum_of_squares_and_product (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (h1 : x^2 + y^2 = 325) (h2 : x * y = 120) :
    x + y = Real.sqrt 565 := by
  sorry

end sum_of_squares_and_product_l99_99493


namespace x4_plus_inverse_x4_l99_99643

theorem x4_plus_inverse_x4 (x : ℝ) (hx : x ^ 2 + 1 / x ^ 2 = 2) : x ^ 4 + 1 / x ^ 4 = 2 := 
sorry

end x4_plus_inverse_x4_l99_99643


namespace axis_of_symmetry_l99_99621

variables (a : ℝ) (x : ℝ)

def parabola := a * (x + 1) * (x - 3)

theorem axis_of_symmetry (h : a ≠ 0) : x = 1 := 
sorry

end axis_of_symmetry_l99_99621


namespace three_primes_sum_odd_l99_99129

theorem three_primes_sum_odd (primes : Finset ℕ) (h_prime : ∀ p ∈ primes, Prime p) :
  primes = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29} →
  (Nat.choose 9 3 / Nat.choose 10 3 : ℚ) = 7 / 10 := by
  -- Let the set of first ten prime numbers.
  -- As per condition, primes = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}
  -- Then show that the probability calculation yields 7/10
  sorry

end three_primes_sum_odd_l99_99129


namespace pheromone_effect_on_population_l99_99500

-- Definitions of conditions
def disrupt_sex_ratio (uses_pheromones : Bool) : Bool :=
  uses_pheromones = true

def decrease_birth_rate (disrupt_sex_ratio : Bool) : Bool :=
  disrupt_sex_ratio = true

def decrease_population_density (decrease_birth_rate : Bool) : Bool :=
  decrease_birth_rate = true

-- Problem Statement for Lean 4
theorem pheromone_effect_on_population (uses_pheromones : Bool) :
  disrupt_sex_ratio uses_pheromones = true →
  decrease_birth_rate (disrupt_sex_ratio uses_pheromones) = true →
  decrease_population_density (decrease_birth_rate (disrupt_sex_ratio uses_pheromones)) = true :=
sorry

end pheromone_effect_on_population_l99_99500


namespace find_n_l99_99410

-- Define the sequence a_n
def a (n : ℕ) : ℤ := 3 * n + 4

-- Define the condition a_n = 13
def condition (n : ℕ) : Prop := a n = 13

-- Prove that under this condition, n = 3
theorem find_n (n : ℕ) (h : condition n) : n = 3 :=
by {
  sorry
}

end find_n_l99_99410


namespace number_of_bird_cages_l99_99201

-- Definitions for the problem conditions
def birds_per_cage : ℕ := 2 + 7
def total_birds : ℕ := 72

-- The theorem to prove the number of bird cages is 8
theorem number_of_bird_cages : total_birds / birds_per_cage = 8 := by
  sorry

end number_of_bird_cages_l99_99201


namespace sum_of_prime_factors_240345_l99_99183

theorem sum_of_prime_factors_240345 : ∀ {p1 p2 p3 : ℕ}, 
  Prime p1 → Prime p2 → Prime p3 →
  p1 * p2 * p3 = 240345 →
  p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 →
  p1 + p2 + p3 = 16011 :=
by
  intros p1 p2 p3 hp1 hp2 hp3 hprod hdiff
  sorry

end sum_of_prime_factors_240345_l99_99183


namespace inequality_problem_l99_99641

theorem inequality_problem (x : ℝ) : x^2 + 1 ≥ 2 * |x| :=
by
  sorry

end inequality_problem_l99_99641


namespace rectangle_perimeter_l99_99633

theorem rectangle_perimeter (a b c width : ℕ) (area : ℕ) 
  (h1 : a = 5) 
  (h2 : b = 12) 
  (h3 : c = 13) 
  (h4 : a^2 + b^2 = c^2) 
  (h5 : area = (a * b) / 2) 
  (h6 : width = 5) 
  (h7 : area = width * ((area * 2) / (a * b)))
  : 2 * (width + (area / width)) = 22 := 
by 
  sorry

end rectangle_perimeter_l99_99633


namespace total_birds_count_l99_99525

def blackbirds_per_tree : ℕ := 3
def tree_count : ℕ := 7
def magpies : ℕ := 13

theorem total_birds_count : (blackbirds_per_tree * tree_count) + magpies = 34 := by
  sorry

end total_birds_count_l99_99525


namespace distance_between_starting_points_l99_99499

theorem distance_between_starting_points :
  let speed1 := 70
  let speed2 := 80
  let start_time := 10 -- in hours (10 am)
  let meet_time := 14 -- in hours (2 pm)
  let travel_time := meet_time - start_time
  let distance1 := speed1 * travel_time
  let distance2 := speed2 * travel_time
  distance1 + distance2 = 600 :=
by
  sorry

end distance_between_starting_points_l99_99499


namespace geometric_progression_common_ratio_l99_99750

theorem geometric_progression_common_ratio (y r : ℝ) (h : (40 + y)^2 = (10 + y) * (90 + y)) :
  r = (40 + y) / (10 + y) → r = (90 + y) / (40 + y) → r = 5 / 3 :=
by
  sorry

end geometric_progression_common_ratio_l99_99750


namespace find_value_of_expression_l99_99726

theorem find_value_of_expression (m n : ℝ) 
  (h1 : m^2 + 2 * m * n = 3) 
  (h2 : m * n + n^2 = 4) : 
  m^2 + 3 * m * n + n^2 = 7 := 
by
  sorry

end find_value_of_expression_l99_99726


namespace find_other_number_l99_99316

-- Define the conditions
variable (B : ℕ)
variable (HCF : ℕ → ℕ → ℕ)
variable (LCM : ℕ → ℕ → ℕ)

axiom hcf_cond : HCF 24 B = 15
axiom lcm_cond : LCM 24 B = 312

-- The theorem statement
theorem find_other_number (B : ℕ) (HCF : ℕ → ℕ → ℕ) (LCM : ℕ → ℕ → ℕ) 
  (hcf_cond : HCF 24 B = 15) (lcm_cond : LCM 24 B = 312) : 
  B = 195 :=
sorry

end find_other_number_l99_99316


namespace pipeline_problem_l99_99356

theorem pipeline_problem 
  (length_pipeline : ℕ) 
  (extra_meters : ℕ) 
  (days_saved : ℕ) 
  (x : ℕ)
  (h1 : length_pipeline = 4000) 
  (h2 : extra_meters = 10) 
  (h3 : days_saved = 20) 
  (h4 : (4000:ℕ) / (x - extra_meters) - (4000:ℕ) / x = days_saved) :
  x = 4000 / ((4000 / (x - extra_meters) + 20)) + extra_meters :=
by
  -- The proof goes here
  sorry

end pipeline_problem_l99_99356


namespace count_sum_or_diff_squares_at_least_1500_l99_99250

theorem count_sum_or_diff_squares_at_least_1500 : 
  (∃ (n : ℕ), 1 ≤ n ∧ n ≤ 2000 ∧ (∃ (x y : ℕ), n = x^2 + y^2 ∨ n = x^2 - y^2)) → 
  1500 ≤ 2000 :=
by
  sorry

end count_sum_or_diff_squares_at_least_1500_l99_99250


namespace problem_1_problem_2_problem_3_l99_99228

def pair_otimes (a b c d : ℚ) : ℚ := b * c - a * d

-- Problem (1)
theorem problem_1 : pair_otimes 5 3 (-2) 1 = -11 := 
by 
  unfold pair_otimes 
  sorry

-- Problem (2)
theorem problem_2 (x : ℚ) (h : pair_otimes 2 (3 * x - 1) 6 (x + 2) = 22) : x = 2 := 
by 
  unfold pair_otimes at h
  sorry

-- Problem (3)
theorem problem_3 (x k : ℤ) (h : pair_otimes 4 (k - 2) x (2 * x - 1) = 6) : 
  k = 8 ∨ k = 9 ∨ k = 11 ∨ k = 12 := 
by 
  unfold pair_otimes at h
  sorry

end problem_1_problem_2_problem_3_l99_99228


namespace total_tshirts_bought_l99_99381

-- Given conditions
def white_packs : ℕ := 3
def white_tshirts_per_pack : ℕ := 6
def blue_packs : ℕ := 2
def blue_tshirts_per_pack : ℕ := 4

-- Theorem statement: Total number of T-shirts Dave bought
theorem total_tshirts_bought : white_packs * white_tshirts_per_pack + blue_packs * blue_tshirts_per_pack = 26 := by
  sorry

end total_tshirts_bought_l99_99381


namespace height_of_rectangular_block_l99_99630

variable (V A h : ℕ)

theorem height_of_rectangular_block :
  V = 120 ∧ A = 24 ∧ V = A * h → h = 5 :=
by
  sorry

end height_of_rectangular_block_l99_99630


namespace MarkBenchPressAmount_l99_99027

def DaveWeight : ℝ := 175
def DaveBenchPressMultiplier : ℝ := 3
def CraigBenchPressFraction : ℝ := 0.20
def MarkDeficitFromCraig : ℝ := 50

theorem MarkBenchPressAmount : 
  let DaveBenchPress := DaveWeight * DaveBenchPressMultiplier
  let CraigBenchPress := DaveBenchPress * CraigBenchPressFraction
  let MarkBenchPress := CraigBenchPress - MarkDeficitFromCraig
  MarkBenchPress = 55 := by
  let DaveBenchPress := DaveWeight * DaveBenchPressMultiplier
  let CraigBenchPress := DaveBenchPress * CraigBenchPressFraction
  let MarkBenchPress := CraigBenchPress - MarkDeficitFromCraig
  sorry

end MarkBenchPressAmount_l99_99027


namespace probability_no_obtuse_triangle_is_9_over_64_l99_99706

noncomputable def probability_no_obtuse_triangle (A0 A1 A2 A3 : ℝ) : ℝ :=
  -- Define the probabilistic model according to the problem conditions
  -- Assuming A0, A1, A2, and A3 are positions of points on the circle parametrized by angles in radians
  let θ := real.angle.lower_half (A1 - A0) in
  let prob_A2 := (π - θ) / (2 * π) in
  let prob_A3 := (π - θ) / (2 * π) in
  (1 / 2) * (prob_A2 * prob_A3)

theorem probability_no_obtuse_triangle_is_9_over_64 :
  probability_no_obtuse_triangle A0 A1 A2 A3 = 9 / 64 :=
by sorry

end probability_no_obtuse_triangle_is_9_over_64_l99_99706


namespace probability_six_distinct_numbers_l99_99140

theorem probability_six_distinct_numbers :
  let outcomes := 6^7
  let favorable_outcomes := 1 * 6 * (Nat.choose 7 2) * (Nat.factorial 5)
  let probability := favorable_outcomes / outcomes
  probability = (35 / 648) := 
by
  let outcomes := 6^7
  let favorable_outcomes := 1 * 6 * (Nat.choose 7 2) * (Nat.factorial 5)
  let probability := favorable_outcomes / outcomes
  have h : favorable_outcomes = 15120 := by sorry
  have h2 : outcomes = 279936 := by sorry
  have prob : probability = (15120 / 279936) := by sorry
  have gcd_calc : gcd 15120 279936 = 432 := by sorry
  have simplified_prob : (15120 / 279936) = (35 / 648) := by sorry
  exact simplified_prob

end probability_six_distinct_numbers_l99_99140


namespace Black_Queen_thought_Black_King_asleep_l99_99952

theorem Black_Queen_thought_Black_King_asleep (BK_awake : Prop) (BQ_awake : Prop) :
  (∃ t : ℕ, t = 10 * 60 + 55 → 
  ∀ (BK : Prop) (BQ : Prop),
    ((BK_awake ↔ ¬BK) ∧ (BQ_awake ↔ ¬BQ)) ∧
    (BK → BQ → BQ_awake) ∧
    (¬BK → ¬BQ → BK_awake)) →
  ((BQ ↔ BK) ∧ (BQ_awake ↔ ¬BQ)) →
  (∃ (BQ_thought : Prop), BQ_thought ↔ BK) := 
sorry

end Black_Queen_thought_Black_King_asleep_l99_99952


namespace three_digit_integers_product_36_l99_99737

theorem three_digit_integers_product_36 : 
  ∃ (num_digits : ℕ), num_digits = 21 ∧ 
    ∀ (a b c : ℕ), 
      (1 ≤ a ∧ a ≤ 9) ∧ 
      (1 ≤ b ∧ b ≤ 9) ∧ 
      (1 ≤ c ∧ c ≤ 9) ∧ 
      (a * b * c = 36) → 
      num_digits = 21 :=
sorry

end three_digit_integers_product_36_l99_99737


namespace smallest_four_digit_multiple_of_18_l99_99689

theorem smallest_four_digit_multiple_of_18 : ∃ n : ℕ, n = 1008 ∧ (1000 ≤ n) ∧ (n < 10000) ∧ (n % 18 = 0) ∧ 
                                ∀ m : ℕ, ((1000 ≤ m) ∧ (m < 10000) ∧ (m % 18 = 0)) → 1008 ≤ m :=
by
  sorry

end smallest_four_digit_multiple_of_18_l99_99689


namespace number_of_small_pipes_needed_l99_99626

theorem number_of_small_pipes_needed :
  let diameter_large := 8
  let diameter_small := 1
  let radius_large := diameter_large / 2
  let radius_small := diameter_small / 2
  let area_large := Real.pi * radius_large^2
  let area_small := Real.pi * radius_small^2
  let num_small_pipes := area_large / area_small
  num_small_pipes = 64 :=
by
  sorry

end number_of_small_pipes_needed_l99_99626


namespace problem_solution_l99_99538

def is_right_triangle (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

def multiplicative_inverse (a m : ℕ) (inv : ℕ) : Prop := 
  (a * inv) % m = 1

theorem problem_solution :
  is_right_triangle 60 144 156 ∧ multiplicative_inverse 300 3751 3618 :=
by
  sorry

end problem_solution_l99_99538


namespace trains_at_start_2016_l99_99936

def traversal_time_red := 7
def traversal_time_blue := 8
def traversal_time_green := 9

def return_period_red := 2 * traversal_time_red
def return_period_blue := 2 * traversal_time_blue
def return_period_green := 2 * traversal_time_green

def train_start_pos_time := 2016
noncomputable def lcm_period := Nat.lcm return_period_red (Nat.lcm return_period_blue return_period_green)

theorem trains_at_start_2016 :
  train_start_pos_time % lcm_period = 0 :=
by
  have return_period_red := 2 * traversal_time_red
  have return_period_blue := 2 * traversal_time_blue
  have return_period_green := 2 * traversal_time_green
  have lcm_period := Nat.lcm return_period_red (Nat.lcm return_period_blue return_period_green)
  have train_start_pos_time := 2016
  exact sorry

end trains_at_start_2016_l99_99936


namespace perfect_square_iff_l99_99674

theorem perfect_square_iff (x y z : ℕ) (hx : x ≥ y) (hy : y ≥ z) (hz : z > 0) :
  ∃ k : ℕ, 4^x + 4^y + 4^z = k^2 ↔ ∃ b : ℕ, b > 0 ∧ x = 2 * b - 1 + z ∧ y = b + z :=
by
  sorry

end perfect_square_iff_l99_99674


namespace correct_factorization_l99_99211

-- Definitions for the given conditions of different options
def condition_A (a : ℝ) : Prop := 2 * a^2 - 2 * a + 1 = 2 * a * (a - 1) + 1
def condition_B (x y : ℝ) : Prop := (x + y) * (x - y) = x^2 - y^2
def condition_C (x y : ℝ) : Prop := x^2 - 4 * x * y + 4 * y^2 = (x - 2 * y)^2
def condition_D (x : ℝ) : Prop := x^2 + 1 = x * (x + 1 / x)

-- The theorem to prove that option C is correct
theorem correct_factorization (x y : ℝ) : condition_C x y :=
by sorry

end correct_factorization_l99_99211


namespace min_val_proof_l99_99783

noncomputable def minimum_value (x y z: ℝ) := 9 / x + 4 / y + 1 / z

theorem min_val_proof (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x + 2 * y + 3 * z = 12) :
  minimum_value x y z ≥ 49 / 12 :=
by {
  sorry
}

end min_val_proof_l99_99783


namespace field_area_l99_99840

def length : ℝ := 80 -- Length of the uncovered side
def total_fencing : ℝ := 97 -- Total fencing required

theorem field_area : ∃ (W L : ℝ), L = length ∧ 2 * W + L = total_fencing ∧ L * W = 680 := by
  sorry

end field_area_l99_99840


namespace time_difference_l99_99579

-- Definitions of speeds and distance
def distance : Nat := 12
def alice_speed : Nat := 7
def bob_speed : Nat := 9

-- Calculations of total times based on speeds and distance
def alice_time : Nat := alice_speed * distance
def bob_time : Nat := bob_speed * distance

-- Statement of the problem
theorem time_difference : bob_time - alice_time = 24 := by
  sorry

end time_difference_l99_99579


namespace cross_product_example_l99_99220

open Matrix

def vector_cross_product (u v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (ux, uy, uz) := u
  let (vx, vy, vz) := v
  (uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx)

theorem cross_product_example :
  vector_cross_product (-3, 4, 2) (8, -5, 6) = (34, -34, -17) := by
  sorry

end cross_product_example_l99_99220


namespace smallest_four_digit_multiple_of_18_l99_99688

theorem smallest_four_digit_multiple_of_18 : ∃ n : ℕ, n = 1008 ∧ (1000 ≤ n) ∧ (n < 10000) ∧ (n % 18 = 0) ∧ 
                                ∀ m : ℕ, ((1000 ≤ m) ∧ (m < 10000) ∧ (m % 18 = 0)) → 1008 ≤ m :=
by
  sorry

end smallest_four_digit_multiple_of_18_l99_99688


namespace mod_45_remainder_of_14_to_100_l99_99397

theorem mod_45_remainder_of_14_to_100 (gcd_14_45 : Nat.gcd 14 45 = 1)
    (phi_45 : Nat.totient 45 = 24)
    (euler_theorem : (14 ^ 24) % 45 = 1) :
    (14 ^ 100) % 45 = 31 := 
by 
  sorry

end mod_45_remainder_of_14_to_100_l99_99397


namespace hyperbola_with_given_asymptotes_l99_99539

/-
Problem: Prove that the equation of the hyperbola with foci on the x-axis 
and asymptotes given by y = ± 2x is x^2 - (y^2 / 4) = 1.
-/

def hyperbola_equation (a b : ℝ) : Prop := (x² / (a^2)) - (y² / (b^2)) = 1

def asymptote_slope (a b : ℝ) : Prop := b / a = 2

theorem hyperbola_with_given_asymptotes 
  (a b : ℝ)
  (h₁ : hyperbola_equation a b)
  (h₂ : asymptote_slope a b) : 
  (x^2 - (y^2 / 4) = 1) :=
sorry

end hyperbola_with_given_asymptotes_l99_99539


namespace multiply_101_self_l99_99860

theorem multiply_101_self : 101 * 101 = 10201 := 
by
  -- Proof omitted
  sorry

end multiply_101_self_l99_99860


namespace num_three_digit_numbers_l99_99884

theorem num_three_digit_numbers (a b c : ℕ) :
  a ≠ 0 →
  b = (a + c) / 2 →
  c = a - b →
  ∃ n1 n2 n3 : ℕ, 
    (n1 = 100 * 3 + 10 * 2 + 1) ∧
    (n2 = 100 * 9 + 10 * 6 + 3) ∧
    (n3 = 100 * 6 + 10 * 4 + 2) ∧ 
    3 = 3 := 
sorry  

end num_three_digit_numbers_l99_99884


namespace sum_of_ages_equal_to_grandpa_l99_99070

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

end sum_of_ages_equal_to_grandpa_l99_99070


namespace peter_reads_one_book_18_hours_l99_99923

-- Definitions of conditions given in the problem
variables (P : ℕ)

-- Condition: Peter can read three times as fast as Kristin
def reads_three_times_as_fast (P : ℕ) : Prop :=
  ∀ (K : ℕ), K = 3 * P

-- Condition: Kristin reads half of her 20 books in 540 hours
def half_books_in_540_hours (K : ℕ) : Prop :=
  K = 54

-- Theorem stating the main proof problem: proving P equals 18 hours
theorem peter_reads_one_book_18_hours
  (H1 : reads_three_times_as_fast P)
  (H2 : half_books_in_540_hours (3 * P)) :
  P = 18 :=
sorry

end peter_reads_one_book_18_hours_l99_99923


namespace volume_of_pool_l99_99013

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

end volume_of_pool_l99_99013


namespace find_two_digit_number_l99_99422

theorem find_two_digit_number (x y a b : ℕ) :
  10 * x + y + 46 = 10 * a + b →
  a * b = 6 →
  a + b = 14 →
  (x = 7 ∧ y = 7) ∨ (x = 8 ∧ y = 6) :=
by {
  sorry
}

end find_two_digit_number_l99_99422


namespace square_field_side_length_l99_99827

theorem square_field_side_length (time_sec : ℕ) (speed_kmh : ℕ) (perimeter : ℕ) (side_length : ℕ)
  (h1 : time_sec = 96)
  (h2 : speed_kmh = 9)
  (h3 : perimeter = (9 * 1000 / 3600 : ℕ) * 96)
  (h4 : perimeter = 4 * side_length) :
  side_length = 60 :=
by
  sorry

end square_field_side_length_l99_99827


namespace smallest_positive_integer_for_terminating_decimal_l99_99163

theorem smallest_positive_integer_for_terminating_decimal (n : ℕ) (h1 : n > 0) (h2 : (∀ m : ℕ, (m > n + 150) -> (m % (n + 150)) ∉ {2, 5})) :
  n = 50 :=
sorry

end smallest_positive_integer_for_terminating_decimal_l99_99163


namespace remainder_for_second_number_l99_99223

theorem remainder_for_second_number (G R1 : ℕ) (first_number second_number : ℕ)
  (hG : G = 144) (hR1 : R1 = 23) (hFirst : first_number = 6215) (hSecond : second_number = 7373) :
  ∃ q2 R2, second_number = G * q2 + R2 ∧ R2 = 29 := 
by {
  -- Ensure definitions are in scope
  exact sorry
}

end remainder_for_second_number_l99_99223


namespace pieces_per_sister_l99_99548

-- Defining the initial conditions
def initial_cake_pieces : ℕ := 240
def percentage_eaten : ℕ := 60
def number_of_sisters : ℕ := 3

-- Defining the statements to be proved
theorem pieces_per_sister (initial_cake_pieces : ℕ) (percentage_eaten : ℕ) (number_of_sisters : ℕ) :
  let pieces_eaten := (percentage_eaten * initial_cake_pieces) / 100
  let remaining_pieces := initial_cake_pieces - pieces_eaten
  let pieces_per_sister := remaining_pieces / number_of_sisters
  pieces_per_sister = 32 :=
by 
  sorry

end pieces_per_sister_l99_99548


namespace smallest_x_l99_99074

theorem smallest_x (x y : ℕ) (h_pos: x > 0 ∧ y > 0) (h_eq: 8 / 10 = y / (186 + x)) : x = 4 :=
sorry

end smallest_x_l99_99074


namespace constant_term_in_expansion_is_neg_220_l99_99765

-- Define expansion and its properties
noncomputable def expansion := (∛x - 1/x)^12

theorem constant_term_in_expansion_is_neg_220 
  (x : ℝ) :
  (sqrt[3] x - 1/x)^12 = expansion → 
  -- Conditions from original problem
  (2^12 = 4096) ↔ 
  -- Conclusion
  (constant_term expansion = -220) :=
by
  sorry

end constant_term_in_expansion_is_neg_220_l99_99765


namespace geometric_sequence_fifth_term_l99_99318

theorem geometric_sequence_fifth_term (a r : ℝ) (h1 : a * r^2 = 16) (h2 : a * r^6 = 2) : a * r^4 = 2 :=
sorry

end geometric_sequence_fifth_term_l99_99318


namespace smallest_n_for_terminating_fraction_l99_99174

-- Define what it means for a number to be a product of only prime factors of 2 and 5
def isTerminatingDenominator (d : ℕ) : Prop := ∃ (a b : ℕ), d = 2^a * 5^b

-- The main statement to prove
theorem smallest_n_for_terminating_fraction : 
  ∃ (n : ℕ), 0 < n ∧ isTerminatingDenominator (n + 150) ∧ 
  ∀ (m : ℕ), (0 < m → isTerminatingDenominator (m + 150) → n ≤ m)
:= sorry

end smallest_n_for_terminating_fraction_l99_99174


namespace no_valid_coloring_l99_99774

open Nat

-- Define the color type
inductive Color
| blue
| red
| green

-- Define the coloring function
def color : ℕ → Color := sorry

-- Define the properties of the coloring function
def valid_coloring (color : ℕ → Color) : Prop :=
  ∀ (m n : ℕ), m > 1 → n > 1 → color m ≠ color n → 
    color (m * n) ≠ color m ∧ color (m * n) ≠ color n

-- Theorem: It is not possible to color all natural numbers greater than 1 as described
theorem no_valid_coloring : ¬ ∃ (color : ℕ → Color), valid_coloring color :=
by
  sorry

end no_valid_coloring_l99_99774


namespace average_age_group_l99_99507

theorem average_age_group (n : ℕ) (T : ℕ) (h1 : T = n * 14) (h2 : T + 32 = (n + 1) * 15) : n = 17 :=
by
  sorry

end average_age_group_l99_99507


namespace min_minutes_for_B_cheaper_l99_99216

-- Define the relevant constants and costs associated with each plan
def cost_A (x : ℕ) : ℕ := 12 * x
def cost_B (x : ℕ) : ℕ := 2500 + 6 * x
def cost_C (x : ℕ) : ℕ := 9 * x

-- Lean statement for the proof problem
theorem min_minutes_for_B_cheaper : ∃ (x : ℕ), x = 834 ∧ cost_B x < cost_A x ∧ cost_B x < cost_C x := 
sorry

end min_minutes_for_B_cheaper_l99_99216


namespace chord_intersection_probability_l99_99253

theorem chord_intersection_probability :
  ∀ (A B C D E F : ℕ), 1 ≤ A ∧ A ≤ 2004 → 
                        1 ≤ B ∧ B ≤ 2004 → 
                        1 ≤ C ∧ C ≤ 2004 → 
                        1 ≤ D ∧ D ≤ 2004 → 
                        1 ≤ E ∧ E ≤ 2004 → 
                        1 ≤ F ∧ F ≤ 2004 → 
                        A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ F ∧ 
                        B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ F ∧ 
                        C ≠ D ∧ C ≠ E ∧ C ≠ F ∧ 
                        D ≠ E ∧ D ≠ F ∧ 
                        E ≠ F → 
  (probability_of_chord_intersection A B C D E F) = 1 / 3 := 
sorry

end chord_intersection_probability_l99_99253


namespace range_of_m_l99_99826

theorem range_of_m (x m : ℝ) (h1: |x - m| < 1) (h2: x^2 - 8 * x + 12 < 0) (h3: ∀ x, (x^2 - 8 * x + 12 < 0) → ((m - 1) < x ∧ x < (m + 1))) : 
  3 ≤ m ∧ m ≤ 5 := 
sorry

end range_of_m_l99_99826


namespace min_distance_mn_l99_99286

noncomputable def f (x : ℝ) : ℝ := x^2
noncomputable def g (x : ℝ) : ℝ := Real.log x

theorem min_distance_mn : ∃ m > 0, ∀ x > 0, |f x - g x| = 1/2 + 1/2 * Real.log 2 :=
by
  sorry

end min_distance_mn_l99_99286


namespace tangent_line_parabola_l99_99311

theorem tangent_line_parabola (d : ℝ) : 
  (∀ x y : ℝ, y = 3 * x + d → ∃! x y, y^2 = 12 * x) → d = 3 := 
by
  intro h
  -- Here, "h" would be our hypothesis where we assume the line is tangent to the parabola
  sorry

end tangent_line_parabola_l99_99311


namespace Ramsey_number_bound_l99_99229

noncomputable def Ramsey_number (k : ℕ) : ℕ := sorry

theorem Ramsey_number_bound (k : ℕ) (h : k ≥ 3) : Ramsey_number k > 2^(k / 2) := sorry

end Ramsey_number_bound_l99_99229


namespace length_side_AB_is_4_l99_99353

-- Defining a triangle ABC with area 6
variables {A B C K L Q : Type*}
variables {side_AB : Float} {ratio_K : Float} {ratio_L : Float} {dist_Q : Float}
variables (area_ABC : ℝ := 6) (ratio_AK_BK : ℝ := 2 / 3) (ratio_AL_LC : ℝ := 5 / 3)
variables (dist_Q_to_AB : ℝ := 1.5)

theorem length_side_AB_is_4 : 
  side_AB = 4 → 
  (area_ABC = 6 ∧ ratio_AK_BK = 2 / 3 ∧ ratio_AL_LC = 5 / 3 ∧ dist_Q_to_AB = 1.5) :=
by
  sorry

end length_side_AB_is_4_l99_99353


namespace min_value_expression_l99_99395

theorem min_value_expression : 
  ∀ (x y : ℝ), (3 * x * x + 4 * x * y + 4 * y * y - 12 * x - 8 * y ≥ -28) ∧ 
  (3 * ((8:ℝ)/3) * ((8:ℝ)/3) + 4 * ((8:ℝ)/3) * -1 + 4 * -1 * -1 - 12 * ((8:ℝ)/3) - 8 * -1 = -28) := 
by sorry

end min_value_expression_l99_99395


namespace turnip_bag_weight_l99_99989

/-- Given six bags with weights 13, 15, 16, 17, 21, and 24 kg,
one bag contains turnips, and the others contain either onions or carrots.
The total weight of the carrots equals twice the total weight of the onions.
Prove that the bag containing turnips can weigh either 13 kg or 16 kg. -/
theorem turnip_bag_weight (ws : list ℕ) (T : ℕ) (O C : ℕ) (h_ws : ws = [13, 15, 16, 17, 21, 24])
  (h_sum : ws.sum = 106) (h_co : C = 2 * O) (h_weight : C + O = 106 - T) :
  T = 13 ∨ T = 16 :=
sorry

end turnip_bag_weight_l99_99989


namespace sufficient_condition_a_gt_1_l99_99961

variable (a : ℝ)

theorem sufficient_condition_a_gt_1 (h : a > 1) : a^2 > 1 :=
by sorry

end sufficient_condition_a_gt_1_l99_99961


namespace anita_total_cartons_l99_99212

-- Defining the conditions
def cartons_of_strawberries : ℕ := 10
def cartons_of_blueberries : ℕ := 9
def additional_cartons_needed : ℕ := 7

-- Adding the core theorem to be proved
theorem anita_total_cartons :
  cartons_of_strawberries + cartons_of_blueberries + additional_cartons_needed = 26 := 
by
  sorry

end anita_total_cartons_l99_99212


namespace find_x_l99_99370

theorem find_x (A V R S x : ℝ) 
  (h1 : A + x = V - x)
  (h2 : V + 2 * x = A - 2 * x + 30)
  (h3 : (A + R / 2) + (V + R / 2) = 120)
  (h4 : S - 0.25 * S + 10 = 2 * (R / 2)) :
  x = 5 :=
  sorry

end find_x_l99_99370


namespace cones_to_cylinder_volume_ratio_l99_99258

theorem cones_to_cylinder_volume_ratio :
  let π := Real.pi
  let r_cylinder := 4
  let h_cylinder := 18
  let r_cone := 4
  let h_cone1 := 6
  let h_cone2 := 9
  let V_cylinder := π * r_cylinder^2 * h_cylinder
  let V_cone1 := (1 / 3) * π * r_cone^2 * h_cone1
  let V_cone2 := (1 / 3) * π * r_cone^2 * h_cone2
  let V_totalCones := V_cone1 + V_cone2
  V_totalCones / V_cylinder = 5 / 18 :=
by
  sorry

end cones_to_cylinder_volume_ratio_l99_99258


namespace max_water_bottles_one_athlete_l99_99012

-- Define variables and key conditions
variable (total_bottles : Nat := 40)
variable (total_athletes : Nat := 25)
variable (at_least_one : ∀ i, i < total_athletes → Nat.succ i ≥ 1)

-- Define the problem as a theorem
theorem max_water_bottles_one_athlete (h_distribution : total_bottles = 40) :
  ∃ max_bottles, max_bottles = 16 :=
by
  sorry

end max_water_bottles_one_athlete_l99_99012


namespace num_archers_golden_armor_proof_l99_99761
noncomputable section

structure Soldier :=
  (is_archer : Bool)
  (is_golden : Bool)
  (tells_truth : Bool)

def count_soldiers (soldiers : List Soldier) : Nat :=
  soldiers.length

def count_truthful_responses (soldiers : List Soldier) (q : Soldier → Bool) : Nat :=
  (soldiers.filter (λ s => s.tells_truth)).count q

def count_lying_responses (soldiers : List Soldier) (q : Soldier → Bool) : Nat :=
  (soldiers.filter (λ s => ¬s.tells_truth)).count q

def num_archers_golden_armor (soldiers : List Soldier) : Nat :=
  (soldiers.filter (λ s => s.is_archer ∧ s.is_golden)).length

theorem num_archers_golden_armor_proof (soldiers : List Soldier)
  (h1 : count_soldiers soldiers = 55)
  (h2 : count_truthful_responses soldiers (λ s => s.is_golden) + 
        count_lying_responses soldiers (λ s => s.is_golden) = 44)
  (h3 : count_truthful_responses soldiers (λ s => s.is_archer) + 
        count_lying_responses soldiers (λ s => s.is_archer) = 33)
  (h4 : count_truthful_responses soldiers (λ s => true) + 
        count_lying_responses soldiers (λ s => false) = 22) :
  num_archers_golden_armor soldiers = 22 := by
  sorry

end num_archers_golden_armor_proof_l99_99761


namespace constant_term_value_l99_99739

variable (y : ℝ)

def constant_term_in_expansion (y : ℝ) (n : ℕ) : ℝ :=
  -- The function calculating the constant term of (y + 2/y)^n.
  sorry -- this would be the detailed expansion expression

theorem constant_term_value :
  let n := 3 * (∫ x in (-real.pi / 2)..(real.pi / 2), real.sin x + real.cos x) in
  constant_term_in_expansion y (nat.floor n) = 160 := by
  sorry

end constant_term_value_l99_99739


namespace path_traveled_by_A_l99_99609

-- Define the initial conditions
def RectangleABCD (A B C D : ℝ × ℝ) :=
  dist A B = 3 ∧ dist C D = 3 ∧ dist B C = 5 ∧ dist D A = 5

-- Define the transformations
def rotated90Clockwise (D : ℝ × ℝ) (A : ℝ × ℝ) (A' : ℝ × ℝ) : Prop :=
  -- 90-degree clockwise rotation moves point A to A'
  A' = (D.1 + D.2 - A.2, D.2 - D.1 + A.1)

def translated3AlongDC (D C A' : ℝ × ℝ) (A'' : ℝ × ℝ) : Prop :=
  -- Translation by 3 units along line DC moves point A' to A''
  A'' = (A'.1 - 3, A'.2)

-- Define the total path traveled
noncomputable def totalPathTraveled (rotatedPath translatedPath : ℝ) : ℝ :=
  rotatedPath + translatedPath

-- Prove the total path is 2.5*pi + 3
theorem path_traveled_by_A (A B C D A' A'' : ℝ × ℝ) (hRect : RectangleABCD A B C D) (hRotate : rotated90Clockwise D A A') (hTranslate : translated3AlongDC D C A' A'') :
  totalPathTraveled (2.5 * Real.pi) 3 = (2.5 * Real.pi + 3) := by
  sorry

end path_traveled_by_A_l99_99609


namespace turnip_weight_possible_l99_99998

-- Define the weights of the 6 bags
def bag_weights : List ℕ := [13, 15, 16, 17, 21, 24]

-- Define the weight of the turnip bag
def is_turnip_bag (T : ℕ) : Prop :=
  T ∈ bag_weights ∧
  ∃ O : ℕ, 3 * O = (bag_weights.sum - T)

theorem turnip_weight_possible : ∀ T, is_turnip_bag T ↔ T = 13 ∨ T = 16 :=
by sorry

end turnip_weight_possible_l99_99998


namespace subtracted_number_from_32_l99_99252

theorem subtracted_number_from_32 (x : ℕ) (h : 32 - x = 23) : x = 9 := 
by 
  sorry

end subtracted_number_from_32_l99_99252


namespace distance_C_to_D_l99_99439

noncomputable def side_length_smaller_square (perimeter : ℝ) : ℝ := perimeter / 4
noncomputable def side_length_larger_square (area : ℝ) : ℝ := Real.sqrt area

theorem distance_C_to_D 
  (perimeter_smaller : ℝ) (area_larger : ℝ) (h1 : perimeter_smaller = 8) (h2 : area_larger = 36) :
  let s_smaller := side_length_smaller_square perimeter_smaller
  let s_larger := side_length_larger_square area_larger 
  let leg1 := s_larger 
  let leg2 := s_larger - 2 * s_smaller 
  Real.sqrt (leg1 ^ 2 + leg2 ^ 2) = 2 * Real.sqrt 10 :=
by
  sorry

end distance_C_to_D_l99_99439


namespace probability_two_red_two_blue_l99_99970

theorem probability_two_red_two_blue :
  let total_marbles := 20
  let total_red := 12
  let total_blue := 8
  let choose_4 := Nat.choose 20 4
  let choose_2_red := Nat.choose 12 2
  let choose_2_blue := Nat.choose 8 2
  (choose_2_red * choose_2_blue).toRational / choose_4.toRational = 3696 / 9690 := by
  let total_marbles := 20
  let total_red := 12
  let total_blue := 8
  let choose_4 := Nat.choose total_marbles 4
  let choose_2_red := Nat.choose total_red 2
  let choose_2_blue := Nat.choose total_blue 2
  show (choose_2_red * choose_2_blue).toRational / choose_4.toRational = 3696 / 9690
  sorry

end probability_two_red_two_blue_l99_99970


namespace isosceles_triangle_problem_l99_99619

theorem isosceles_triangle_problem 
  (a h b : ℝ) 
  (area_relation : (1/2) * a * h = (1/3) * a ^ 2) 
  (leg_relation : b = a - 1)
  (height_relation : h = (2/3) * a) 
  (pythagorean_theorem : h ^ 2 + (a / 2) ^ 2 = b ^ 2) : 
  a = 6 ∧ b = 5 ∧ h = 4 :=
sorry

end isosceles_triangle_problem_l99_99619


namespace faye_books_l99_99678

theorem faye_books (initial_books given_away final_books books_bought: ℕ) 
  (h1 : initial_books = 34) 
  (h2 : given_away = 3) 
  (h3 : final_books = 79) 
  (h4 : final_books = initial_books - given_away + books_bought) : 
  books_bought = 48 := 
by 
  sorry

end faye_books_l99_99678


namespace largest_angle_of_consecutive_integers_hexagon_l99_99476

theorem largest_angle_of_consecutive_integers_hexagon (a b c d e f : ℝ) 
  (h1 : a < b) (h2 : b < c) (h3 : c < d) (h4 : d < e) (h5 : e < f) 
  (h6 : a + b + c + d + e + f = 720) : 
  ∃ x, f = x + 2 ∧ (x + 2 = 122.5) :=
  sorry

end largest_angle_of_consecutive_integers_hexagon_l99_99476


namespace compute_ab_l99_99933

theorem compute_ab (a b : ℝ) 
  (h1 : b^2 - a^2 = 25) 
  (h2 : a^2 + b^2 = 49) : 
  |a * b| = Real.sqrt 444 := 
by 
  sorry

end compute_ab_l99_99933


namespace maximum_triangle_area_l99_99524

-- Define the maximum area of a triangle given two sides.
theorem maximum_triangle_area (a b : ℝ) (h_a : a = 1984) (h_b : b = 2016) :
  ∃ (max_area : ℝ), max_area = 1998912 :=
by
  sorry

end maximum_triangle_area_l99_99524


namespace turnip_bag_weights_l99_99993

theorem turnip_bag_weights :
  ∃ (T : ℕ), (T = 13 ∨ T = 16) ∧ 
  (∀ T', T' ≠ T → (
    (2 * total_weight_of_has_weight_1.not_turnip T' O + O = 106 - T') → 
    let turnip_condition := 106 - T in
    turnip_condition % 3 = 0)) ∧
  ∀ (bag_weights : List ℕ) (other : bag_weights = [13, 15, 16, 17, 21, 24] ∧ 
                                          List.length bag_weights = 6),
  True := by sorry

end turnip_bag_weights_l99_99993


namespace caitlin_bracelets_l99_99533

-- Define the conditions
def twice_as_many_small_beads (x y : Nat) : Prop :=
  y = 2 * x

def total_large_small_beads (total large small : Nat) : Prop :=
  total = large + small ∧ large = small

def bracelet_beads (large_beads_per_bracelet small_beads_per_bracelet large_per_bracelet : Nat) : Prop :=
  small_beads_per_bracelet = 2 * large_per_bracelet

def total_bracelets (total_large_beads large_per_bracelet bracelets : Nat) : Prop :=
  bracelets = total_large_beads / large_per_bracelet

-- The theorem to be proved
theorem caitlin_bracelets (total_beads large_per_bracelet small_per_bracelet : Nat) (bracelets : Nat) :
    total_beads = 528 ∧
    large_per_bracelet = 12 ∧
    twice_as_many_small_beads large_per_bracelet small_per_bracelet ∧
    total_large_small_beads total_beads 264 264 ∧
    bracelet_beads large_per_bracelet small_per_bracelet 12 ∧
    total_bracelets 264 12 bracelets
  → bracelets = 22 := by
  sorry

end caitlin_bracelets_l99_99533


namespace min_value_x2_plus_y2_l99_99743

theorem min_value_x2_plus_y2 :
  ∀ x y : ℝ, (x + 5)^2 + (y - 12)^2 = 196 → x^2 + y^2 ≥ 1 :=
by
  intros x y h
  sorry

end min_value_x2_plus_y2_l99_99743


namespace midpoint_AB_is_correct_l99_99764

/--
In the Cartesian coordinate system, given points A (-1, 2) and B (3, 0), prove that the coordinates of the midpoint of segment AB are (1, 1).
-/
theorem midpoint_AB_is_correct :
  let A := (-1, 2)
  let B := (3, 0)
  (A.1 + B.1) / 2 = 1 ∧ (A.2 + B.2) / 2 = 1 := 
by {
  let A := (-1, 2)
  let B := (3, 0)
  sorry -- this part is omitted as no proof is needed
}

end midpoint_AB_is_correct_l99_99764


namespace convex_hexagon_largest_angle_l99_99477

theorem convex_hexagon_largest_angle 
  (x : ℝ)                                 -- Denote the measure of the third smallest angle as x.
  (angles : Fin 6 → ℝ)                     -- Define the angles as a function from Fin 6 to ℝ.
  (h1 : ∀ i : Fin 6, angles i = x + (i : ℝ) - 3)  -- The six angles in increasing order.
  (h2 : 0 < x - 3 ∧ x - 3 < 180)           -- Convex condition: each angle is between 0 and 180.
  (h3 : angles ⟨0⟩ + angles ⟨1⟩ + angles ⟨2⟩ + angles ⟨3⟩ + angles ⟨4⟩ + angles ⟨5⟩ = 720) -- Sum of interior angles of a hexagon.
  : (∃ a, a = angles ⟨5⟩ ∧ a = 122.5) :=   -- Prove the largest angle in this arrangement is 122.5.
sorry

end convex_hexagon_largest_angle_l99_99477


namespace johns_tour_program_days_l99_99350

/-- John has Rs 360 for his expenses. If he exceeds his days by 4 days, he must cut down daily expenses by Rs 3. Prove that the number of days of John's tour program is 20. -/
theorem johns_tour_program_days
    (d e : ℕ)
    (h1 : 360 = e * d)
    (h2 : 360 = (e - 3) * (d + 4)) : 
    d = 20 := 
  sorry

end johns_tour_program_days_l99_99350


namespace find_smallest_angle_b1_l99_99830

-- Definitions and conditions
def smallest_angle_in_sector (b1 e : ℕ) (k : ℕ := 5) : Prop :=
  2 * b1 + (k - 1) * k * e = 360 ∧ b1 + 2 * e = 36

theorem find_smallest_angle_b1 (b1 e : ℕ) : smallest_angle_in_sector b1 e → b1 = 30 :=
  sorry

end find_smallest_angle_b1_l99_99830


namespace ratio_of_intercepts_l99_99328

variable {c : ℝ} (non_zero_c : c ≠ 0) (u v : ℝ)
-- Condition: The first line, slope 8, y-intercept c, x-intercept (u, 0)
variable (h_u : u = -c / 8)
-- Condition: The second line, slope 4, y-intercept c, x-intercept (v, 0)
variable (h_v : v = -c / 4)

theorem ratio_of_intercepts (non_zero_c : c ≠ 0)
    (h_u : u = -c / 8) (h_v : v = -c / 4) : u / v = 1 / 2 :=
by
  sorry

end ratio_of_intercepts_l99_99328


namespace no_obtuse_triangle_probability_eq_l99_99704

noncomputable def probability_no_obtuse_triangle : ℝ :=
  let num_points := 4
  -- Condition (1): Four points are chosen uniformly at random on a circle.
  -- Condition (2): An obtuse angle occurs if the minor arc exceeds π/2.
  9 / 64

theorem no_obtuse_triangle_probability_eq :
  let num_points := 4
  ∀ (points : Fin num_points → ℝ), 
  (∀ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k →
    ∠ (points i, points j, points k) < π / 2) →
  probability_no_obtuse_triangle = 9 / 64 :=
by
  sorry

end no_obtuse_triangle_probability_eq_l99_99704


namespace total_marbles_l99_99512

theorem total_marbles (x : ℕ) (h1 : 5 * x - 2 = 18) : 4 * x + 5 * x = 36 :=
by
  sorry

end total_marbles_l99_99512


namespace factory_material_equation_correct_l99_99863

variable (a b x : ℝ)
variable (h_a : a = 180)
variable (h_b : b = 120)
variable (h_condition : (a - 2 * x) - (b + x) = 30)

theorem factory_material_equation_correct : (180 - 2 * x) - (120 + x) = 30 := by
  rw [←h_a, ←h_b]
  exact h_condition

end factory_material_equation_correct_l99_99863


namespace largest_angle_in_consecutive_integer_hexagon_l99_99484

theorem largest_angle_in_consecutive_integer_hexagon : 
  ∀ (x : ℤ), 
  (x - 3) + (x - 2) + (x - 1) + x + (x + 1) + (x + 2) = 720 → 
  (x + 2 = 122) :=
by intros x h
   sorry

end largest_angle_in_consecutive_integer_hexagon_l99_99484


namespace ratio_bc_cd_l99_99642

-- Definitions based on given conditions.
variable (a b c d e : ℝ)
variable (h_ab : b - a = 5)
variable (h_ac : c - a = 11)
variable (h_de : e - d = 8)
variable (h_ae : e - a = 22)

-- The theorem to prove bc : cd = 2 : 1.
theorem ratio_bc_cd (h_ab : b - a = 5) (h_ac : c - a = 11) (h_de : e - d = 8) (h_ae : e - a = 22) :
  (c - b) / (d - c) = 2 :=
by
  sorry

end ratio_bc_cd_l99_99642


namespace find_k_from_polynomial_l99_99808

theorem find_k_from_polynomial :
  ∃ (k : ℝ),
  (∃ (x₁ x₂ x₃ x₄ : ℝ), 
    x₁ * x₂ * x₃ * x₄ = -1984 ∧
    x₁ * x₂ + x₁ * x₃ + x₁ * x₄ + x₂ * x₃ + x₂ * x₄ + x₃ * x₄ = k ∧
    x₁ + x₂ + x₃ + x₄ = 18 ∧
    (x₁ * x₂ = -32 ∨ x₁ * x₃ = -32 ∨ x₁ * x₄ = -32 ∨ x₂ * x₃ = -32 ∨ x₂ * x₄ = -32 ∨ x₃ * x₄ = -32))
  → k = 86 :=
by
  sorry

end find_k_from_polynomial_l99_99808


namespace smallest_positive_four_digit_multiple_of_18_l99_99693

-- Define the predicates for conditions
def four_digit_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

def multiple_of_18 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 18 * k

-- Define the main theorem
theorem smallest_positive_four_digit_multiple_of_18 : 
  ∃ n : ℕ, four_digit_number n ∧ multiple_of_18 n ∧ ∀ m : ℕ, four_digit_number m ∧ multiple_of_18 m → n ≤ m :=
begin
  use 1008,
  split,
  { -- proof that 1008 is a four-digit number
    split,
    { linarith, },
    { linarith, }
  },

  split,
  { -- proof that 1008 is a multiple of 18
    use 56,
    norm_num,
  },

  { -- proof that 1008 is the smallest such number
    intros m h1 h2,
    have h3 := Nat.le_of_lt,
    sorry, -- Detailed proof would go here
  }
end

end smallest_positive_four_digit_multiple_of_18_l99_99693


namespace avg_people_moving_to_florida_per_hour_l99_99766

theorem avg_people_moving_to_florida_per_hour (people : ℕ) (days : ℕ) (hours_per_day : ℕ) 
  (h1 : people = 3000) (h2 : days = 5) (h3 : hours_per_day = 24) : 
  people / (days * hours_per_day) = 25 := by
  sorry

end avg_people_moving_to_florida_per_hour_l99_99766


namespace probability_cube_vertices_in_plane_l99_99718

open Finset

noncomputable def choose (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem probability_cube_vertices_in_plane : 
  let total_ways := choose 8 4 in
  let favorable_ways := 12 in
  0 < total_ways →  -- Ensure total_ways is non-zero to avoid division by zero
  let P := (favorable_ways : ℝ) / (total_ways : ℝ) in
  P = 6 / 35 :=
by 
  sorry

end probability_cube_vertices_in_plane_l99_99718


namespace remainder_of_exponentiation_is_correct_l99_99048

-- Define the given conditions
def modulus := 500
def exponent := 5 ^ (5 ^ 5)
def carmichael_500 := 100
def carmichael_100 := 20

-- Prove the main theorem
theorem remainder_of_exponentiation_is_correct :
  (5 ^ exponent) % modulus = 125 := 
by
  -- Skipping the proof
  sorry

end remainder_of_exponentiation_is_correct_l99_99048


namespace smallest_four_digit_multiple_of_18_l99_99682

theorem smallest_four_digit_multiple_of_18 : ∃ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ 18 ∣ n ∧ ∀ m : ℕ, 1000 ≤ m ∧ m ≤ 9999 ∧ 18 ∣ m → n ≤ m := by
  use 1008
  sorry

end smallest_four_digit_multiple_of_18_l99_99682


namespace third_chapter_pages_l99_99971

theorem third_chapter_pages (x : ℕ) (h : 18 = x + 15) : x = 3 :=
by
  sorry

end third_chapter_pages_l99_99971


namespace turnip_bag_weighs_l99_99984

theorem turnip_bag_weighs (bags : List ℕ) (T : ℕ)
  (h_weights : bags = [13, 15, 16, 17, 21, 24])
  (h_turnip : T ∈ bags)
  (h_carrot_onion_relation : ∃ O C: ℕ, C = 2 * O ∧ C + O = 106 - T) :
  T = 13 ∨ T = 16 := by
  sorry

end turnip_bag_weighs_l99_99984


namespace multiple_of_A_share_l99_99206

theorem multiple_of_A_share (a b c : ℤ) (hC : c = 84) (hSum : a + b + c = 427)
  (hEquality1 : ∃ x : ℤ, x * a = 4 * b) (hEquality2 : 7 * c = 4 * b) : ∃ x : ℤ, x = 3 :=
by {
  sorry
}

end multiple_of_A_share_l99_99206


namespace find_value_of_f_l99_99238

axiom f : ℝ → ℝ

theorem find_value_of_f :
  (∀ x : ℝ, f (Real.cos x) = Real.sin (3 * x)) →
  f (Real.sin (Real.pi / 9)) = -1 / 2 :=
sorry

end find_value_of_f_l99_99238


namespace min_additional_games_l99_99618

def num_initial_games : ℕ := 4
def num_lions_won : ℕ := 3
def num_eagles_won : ℕ := 1
def win_threshold : ℝ := 0.90

theorem min_additional_games (M : ℕ) : (num_eagles_won + M) / (num_initial_games + M) ≥ win_threshold ↔ M ≥ 26 :=
by
  sorry

end min_additional_games_l99_99618


namespace cube_vertices_probability_l99_99716

theorem cube_vertices_probability (totalVertices : ℕ) (selectedVertices : ℕ) 
   (totalCombinations : ℕ) (favorableOutcomes : ℕ) : 
   totalVertices = 8 ∧ selectedVertices = 4 ∧ totalCombinations = 70 ∧ favorableOutcomes = 12 → 
   (favorableOutcomes : ℚ) / totalCombinations = 6 / 35 := by
   sorry

end cube_vertices_probability_l99_99716


namespace area_of_triangle_union_reflection_l99_99845

def area_of_union_of_reflected_triangle : ℝ :=
  let A := (2:ℝ, 3:ℝ)
  let B := (4:ℝ, 1:ℝ)
  let C := (6:ℝ, 6:ℝ)
  let A' := (2:ℝ, 1:ℝ)
  let B' := (4:ℝ, 3:ℝ)
  let C' := (6:ℝ, -2:ℝ)
  let area (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) : ℝ :=
    (0.5) * float.abs (x₁ * (y₂ - y₃) + x₂ * (y₃ - y₁) + x₃ * (y₁ - y₂))

  let area_original := area 2 3 4 1 6 6
  let area_reflected := area 2 1 4 3 6 (-2)

  area_original + area_reflected

theorem area_of_triangle_union_reflection :
  area_of_union_of_reflected_triangle = 14 := sorry

end area_of_triangle_union_reflection_l99_99845


namespace smallest_n_terminating_decimal_l99_99170

theorem smallest_n_terminating_decimal :
  ∃ n : ℕ, (∀ m : ℕ, (∃ k : ℕ, m + 150 = 2^k * (5^0))
  → (m ≥ n)) ∧ (n > 0) ∧ (∃ k : ℕ, n + 150 = 2^k * (5^0)) := 
sorry

end smallest_n_terminating_decimal_l99_99170


namespace sum_of_squares_and_product_l99_99492

open Real

theorem sum_of_squares_and_product (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (h1 : x^2 + y^2 = 325) (h2 : x * y = 120) :
    x + y = Real.sqrt 565 := by
  sorry

end sum_of_squares_and_product_l99_99492


namespace field_area_restriction_l99_99680

theorem field_area_restriction (S : ℚ) (b : ℤ) (a : ℚ) (x y : ℚ) 
  (h1 : 10 * 300 * S ≤ 10000)
  (h2 : 2 * a = - b)
  (h3 : abs (6 * y) + 3 ≥ 3)
  (h4 : 2 * abs (2 * x) - abs b ≤ 9)
  (h5 : b ∈ [-4, -3, -2, -1, 0, 1, 2, 3, 4])
: S ≤ 10 / 3 := sorry

end field_area_restriction_l99_99680


namespace three_2x2_squares_exceed_100_l99_99486

open BigOperators

noncomputable def sum_of_1_to_64 : ℕ :=
  (64 * (64 + 1)) / 2

theorem three_2x2_squares_exceed_100 :
  ∀ (s : Fin 16 → ℕ),
    (∑ i, s i = sum_of_1_to_64) →
    (∀ i j, i ≠ j → s i = s j ∨ s i > s j ∨ s i < s j) →
    (∃ i₁ i₂ i₃, i₁ ≠ i₂ ∧ i₂ ≠ i₃ ∧ i₁ ≠ i₃ ∧ s i₁ > 100 ∧ s i₂ > 100 ∧ s i₃ > 100) := sorry

end three_2x2_squares_exceed_100_l99_99486


namespace parts_rate_relation_l99_99459

theorem parts_rate_relation
  (x : ℝ)
  (total_parts_per_hour : ℝ)
  (master_parts : ℝ)
  (apprentice_parts : ℝ)
  (h_total : total_parts_per_hour = 40)
  (h_master : master_parts = 300)
  (h_apprentice : apprentice_parts = 100)
  (h : total_parts_per_hour = x + (40 - x)) :
  (master_parts / x) = (apprentice_parts / (40 - x)) := 
by
  sorry

end parts_rate_relation_l99_99459


namespace points_eq_l99_99670

-- Definition of the operation 
def star (a b : ℝ) : ℝ := a^2 * b + a * b^2

-- The property we want to prove
theorem points_eq : {p : ℝ × ℝ | star p.1 p.2 = star p.2 p.1} =
    {p : ℝ × ℝ | p.1 = 0} ∪ {p : ℝ × ℝ | p.2 = 0} ∪ {p : ℝ × ℝ | p.1 + p.2 = 0} :=
by
  sorry

end points_eq_l99_99670


namespace triangle_inequality_problem_l99_99242

theorem triangle_inequality_problem
  (x : ℝ)
  (side1 side2 : ℝ)
  (h1 : side1 = 5)
  (h2 : side2 = 7)
  (h3 : x = 10) :
  (2 < x ∧ x < 12) := 
by
  sorry

end triangle_inequality_problem_l99_99242


namespace minimum_value_fraction_l99_99890

theorem minimum_value_fraction (a b : ℝ) (h1 : a > 1) (h2 : b > 2) (h3 : 2 * a + b - 6 = 0) :
  (1 / (a - 1) + 2 / (b - 2)) = 4 := 
  sorry

end minimum_value_fraction_l99_99890


namespace probability_top_four_cards_is_2_over_95_l99_99842

noncomputable def probability_top_four_hearts :
  ℚ :=
  ((13 * 12 * 11 * 10) : ℚ) / ((52 * 51 * 50 * 49) : ℚ)

theorem probability_top_four_cards_is_2_over_95 :
  probability_top_four_hearts = 2 / 95 :=
by
  -- This space is intentionally left without proof
  sorry

end probability_top_four_cards_is_2_over_95_l99_99842


namespace time_correct_l99_99442

theorem time_correct {t : ℝ} (h : 0 < t ∧ t < 60) :
  |6 * (t + 5) - (90 + 0.5 * (t - 4))| = 180 → t = 43 := by
  sorry

end time_correct_l99_99442


namespace closest_to_9_l99_99185

noncomputable def optionA : ℝ := 10.01
noncomputable def optionB : ℝ := 9.998
noncomputable def optionC : ℝ := 9.9
noncomputable def optionD : ℝ := 9.01
noncomputable def target : ℝ := 9

theorem closest_to_9 : 
  abs (optionD - target) < abs (optionA - target) ∧ 
  abs (optionD - target) < abs (optionB - target) ∧ 
  abs (optionD - target) < abs (optionC - target) := 
by
  sorry

end closest_to_9_l99_99185


namespace sin_double_angle_given_sum_identity_l99_99235

theorem sin_double_angle_given_sum_identity {α : ℝ} 
  (h : Real.sin (Real.pi / 4 + α) = Real.sqrt 5 / 5) : 
  Real.sin (2 * α) = -3 / 5 := 
by 
  sorry

end sin_double_angle_given_sum_identity_l99_99235


namespace no_obtuse_triangle_probability_eq_l99_99703

noncomputable def probability_no_obtuse_triangle : ℝ :=
  let num_points := 4
  -- Condition (1): Four points are chosen uniformly at random on a circle.
  -- Condition (2): An obtuse angle occurs if the minor arc exceeds π/2.
  9 / 64

theorem no_obtuse_triangle_probability_eq :
  let num_points := 4
  ∀ (points : Fin num_points → ℝ), 
  (∀ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k →
    ∠ (points i, points j, points k) < π / 2) →
  probability_no_obtuse_triangle = 9 / 64 :=
by
  sorry

end no_obtuse_triangle_probability_eq_l99_99703


namespace non_isosceles_count_l99_99928

def n : ℕ := 20

def total_triangles : ℕ := Nat.choose n 3

def isosceles_triangles_per_vertex : ℕ := 9

def total_isosceles_triangles : ℕ := n * isosceles_triangles_per_vertex

def non_isosceles_triangles : ℕ := total_triangles - total_isosceles_triangles

theorem non_isosceles_count :
  non_isosceles_triangles = 960 := 
  by 
    -- proof details would go here
    sorry

end non_isosceles_count_l99_99928


namespace skittles_left_l99_99605

theorem skittles_left (initial_skittles : ℕ) (skittles_given : ℕ) (final_skittles : ℕ) :
  initial_skittles = 50 → skittles_given = 7 → final_skittles = initial_skittles - skittles_given → final_skittles = 43 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end skittles_left_l99_99605


namespace quadratic_eq1_solution_quadratic_eq2_solution_l99_99799

-- Define the first problem and its conditions
theorem quadratic_eq1_solution :
  ∀ x : ℝ, 4 * x^2 + x - (1 / 2) = 0 ↔ (x = -1 / 2 ∨ x = 1 / 4) :=
by
  -- The proof is omitted
  sorry

-- Define the second problem and its conditions
theorem quadratic_eq2_solution :
  ∀ y : ℝ, (y - 2) * (y + 3) = 6 ↔ (y = -4 ∨ y = 3) :=
by
  -- The proof is omitted
  sorry

end quadratic_eq1_solution_quadratic_eq2_solution_l99_99799


namespace roots_of_polynomial_l99_99226

noncomputable def poly (x : ℝ) : ℝ := x^5 - 3*x^4 + 3*x^3 - x^2 - 4*x + 4

theorem roots_of_polynomial :
  ∀ x : ℝ, poly x = 0 ↔ (x = -1 ∨ x = 1 ∨ x = 2) :=
by
  sorry

end roots_of_polynomial_l99_99226


namespace sandy_comic_books_l99_99107

-- Define Sandy's initial number of comic books
def initial_comic_books : ℕ := 14

-- Define the number of comic books Sandy sold
def sold_comic_books (n : ℕ) : ℕ := n / 2

-- Define the number of comic books Sandy bought
def bought_comic_books : ℕ := 6

-- Define the number of comic books Sandy has now
def final_comic_books (initial : ℕ) (sold : ℕ) (bought : ℕ) : ℕ :=
  initial - sold + bought

-- The theorem statement to prove the final number of comic books
theorem sandy_comic_books : final_comic_books initial_comic_books (sold_comic_books initial_comic_books) bought_comic_books = 13 := by
  sorry

end sandy_comic_books_l99_99107


namespace smallest_positive_integer_for_terminating_decimal_l99_99180

theorem smallest_positive_integer_for_terminating_decimal: ∃ n: ℕ, (n > 0) ∧ (∀ p : ℕ, (p ∣ (n + 150)) → (p=1 ∨ p=2 ∨ p=4 ∨ p=5 ∨ p=8 ∨ p=10 ∨ p=16 ∨ p=20 ∨ p=25 ∨ p=32 ∨ p=40 ∨ p=50 ∨ p=64 ∨ p=80 ∨ p=100 ∨ p=125 ∨ p=128 ∨ p=160)) ∧ n = 10 :=
by
  sorry

end smallest_positive_integer_for_terminating_decimal_l99_99180


namespace kendra_and_tony_keep_two_each_l99_99091

-- Define the conditions
def kendra_packs : Nat := 4
def tony_packs : Nat := 2
def pens_per_pack : Nat := 3
def pens_given_to_friends : Nat := 14

-- Define the total pens each has
def kendra_pens : Nat := kendra_packs * pens_per_pack
def tony_pens : Nat := tony_packs * pens_per_pack

-- Define the total pens
def total_pens : Nat := kendra_pens + tony_pens

-- Define the pens left after distribution
def pens_left : Nat := total_pens - pens_given_to_friends

-- Define the number of pens each keeps
def pens_each_kept : Nat := pens_left / 2

-- Prove the final statement
theorem kendra_and_tony_keep_two_each :
  pens_each_kept = 2 :=
by
  sorry

end kendra_and_tony_keep_two_each_l99_99091


namespace largest_angle_of_consecutive_integers_in_hexagon_l99_99480

theorem largest_angle_of_consecutive_integers_in_hexagon : 
  ∀ (a : ℕ), 
    (a - 2) + (a - 1) + a + (a + 1) + (a + 2) + (a + 3) = 720 → 
    a + 3 = 122.5 :=
by sorry

end largest_angle_of_consecutive_integers_in_hexagon_l99_99480
