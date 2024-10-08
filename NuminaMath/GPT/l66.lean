import Mathlib

namespace problem_solution_l66_66038

theorem problem_solution
  (a b : ℝ)
  (h1 : a * b = 2)
  (h2 : a - b = 3) :
  a^3 * b - 2 * a^2 * b^2 + a * b^3 = 18 :=
by
  sorry

end problem_solution_l66_66038


namespace find_smallest_x_l66_66053

theorem find_smallest_x :
  ∃ x : ℕ, x > 0 ∧
  (45 * x + 9) % 25 = 3 ∧
  (2 * x) % 5 = 8 ∧
  x = 20 :=
by
  sorry

end find_smallest_x_l66_66053


namespace rectangular_plot_breadth_l66_66136

theorem rectangular_plot_breadth:
  ∀ (b l : ℝ), (l = b + 10) → (24 * b = l * b) → b = 14 :=
by
  intros b l hl hs
  sorry

end rectangular_plot_breadth_l66_66136


namespace smallest_mn_sum_l66_66686

theorem smallest_mn_sum {n m : ℕ} (h1 : n > m) (h2 : 1978 ^ n % 1000 = 1978 ^ m % 1000) (h3 : m ≥ 1) : m + n = 106 := 
sorry

end smallest_mn_sum_l66_66686


namespace tom_saves_money_l66_66864

-- Defining the cost of a normal doctor's visit
def normal_doctor_cost : ℕ := 200

-- Defining the discount percentage for the discount clinic
def discount_percentage : ℕ := 70

-- Defining the cost reduction based on the discount percentage
def discount_amount (cost percentage : ℕ) : ℕ := (percentage * cost) / 100

-- Defining the cost of a visit to the discount clinic
def discount_clinic_cost (normal_cost discount_amount : ℕ ) : ℕ := normal_cost - discount_amount

-- Defining the number of visits to the discount clinic
def discount_clinic_visits : ℕ := 2

-- Defining the total cost for the discount clinic visits
def total_discount_clinic_cost (visit_cost visits : ℕ) : ℕ := visits * visit_cost

-- The final cost savings calculation
def cost_savings (normal_cost total_discount_cost : ℕ) : ℕ := normal_cost - total_discount_cost

-- Proving the amount Tom saves by going to the discount clinic
theorem tom_saves_money : cost_savings normal_doctor_cost (total_discount_clinic_cost (discount_clinic_cost normal_doctor_cost (discount_amount normal_doctor_cost discount_percentage)) discount_clinic_visits) = 80 :=
by
  sorry

end tom_saves_money_l66_66864


namespace resistor_parallel_l66_66002

theorem resistor_parallel (x y r : ℝ)
  (h1 : x = 5)
  (h2 : r = 2.9166666666666665)
  (h3 : 1 / r = 1 / x + 1 / y) : y = 7 :=
by
  -- proof omitted
  sorry

end resistor_parallel_l66_66002


namespace find_m_positive_root_l66_66590

theorem find_m_positive_root :
  (∃ x > 0, (x - 4) / (x - 3) - m - 4 = m / (3 - x)) → m = 1 :=
by
  sorry

end find_m_positive_root_l66_66590


namespace time_to_pass_telegraph_post_l66_66095

def conversion_factor_km_per_hour_to_m_per_sec := 1000 / 3600

noncomputable def train_length := 70
noncomputable def train_speed_kmph := 36

noncomputable def train_speed_m_per_sec := train_speed_kmph * conversion_factor_km_per_hour_to_m_per_sec

theorem time_to_pass_telegraph_post : (train_length / train_speed_m_per_sec) = 7 := by
  sorry

end time_to_pass_telegraph_post_l66_66095


namespace tom_books_total_l66_66518

theorem tom_books_total :
  (2 + 6 + 10 + 14 + 18) = 50 :=
by {
  -- Proof steps would go here.
  sorry
}

end tom_books_total_l66_66518


namespace distance_AB_l66_66405

def C1_polar (ρ θ : Real) : Prop :=
  ρ = 2 * Real.cos θ

def C2_polar (ρ θ : Real) : Prop :=
  ρ^2 * (1 + (Real.sin θ)^2) = 2

def ray_polar (θ : Real) : Prop :=
  θ = Real.pi / 6

theorem distance_AB :
  let ρ1 := 2 * Real.cos (Real.pi / 6)
  let ρ2 := Real.sqrt 10 * 2 / 5
  |ρ1 - ρ2| = Real.sqrt 3 - (2 * Real.sqrt 10) / 5 :=
by
  sorry

end distance_AB_l66_66405


namespace perfect_square_factors_450_l66_66213

theorem perfect_square_factors_450 :
  ∃ (n : ℕ), n = 4 ∧ ∀ (d : ℕ), d ∣ 450 → ∃ (k : ℕ), k^2 = d ↔ d = 1 ∨ d = 9 ∨ d = 25 ∨ d = 225 :=
by
  sorry

end perfect_square_factors_450_l66_66213


namespace find_g_l66_66257

open Function

def linear_system (a b c d e f g : ℚ) :=
  a + b + c + d + e = 1 ∧
  b + c + d + e + f = 2 ∧
  c + d + e + f + g = 3 ∧
  d + e + f + g + a = 4 ∧
  e + f + g + a + b = 5 ∧
  f + g + a + b + c = 6 ∧
  g + a + b + c + d = 7

theorem find_g (a b c d e f g : ℚ) (h : linear_system a b c d e f g) : 
  g = 13 / 3 :=
sorry

end find_g_l66_66257


namespace intersection_point_is_correct_l66_66620

def line1 (x y : ℝ) := x - 2 * y + 7 = 0
def line2 (x y : ℝ) := 2 * x + y - 1 = 0

theorem intersection_point_is_correct : line1 (-1) 3 ∧ line2 (-1) 3 :=
by
  sorry

end intersection_point_is_correct_l66_66620


namespace tenth_term_is_98415_over_262144_l66_66978

def first_term : ℚ := 5
def common_ratio : ℚ := 3 / 4

def tenth_term_geom_seq (a r : ℚ) (n : ℕ) : ℚ := a * r^(n - 1)

theorem tenth_term_is_98415_over_262144 :
  tenth_term_geom_seq first_term common_ratio 10 = 98415 / 262144 :=
sorry

end tenth_term_is_98415_over_262144_l66_66978


namespace max_r1_minus_r2_l66_66680

noncomputable def ellipse (x y : ℝ) : Prop :=
  (x^2) / 2 + y^2 = 1

def F1 : ℝ × ℝ := (-1, 0)
def F2 : ℝ × ℝ := (1, 0)

def P (x y : ℝ) : Prop :=
  ellipse x y ∧ x > 0 ∧ y > 0

def r1 (x y : ℝ) (Q2 : ℝ × ℝ) : ℝ := 
  -- Assume a function that calculates the inradius of triangle ΔPF1Q2
  sorry

def r2 (x y : ℝ) (Q1 : ℝ × ℝ) : ℝ :=
  -- Assume a function that calculates the inradius of triangle ΔPF2Q1
  sorry

theorem max_r1_minus_r2 :
  ∃ (x y : ℝ) (Q1 Q2 : ℝ × ℝ), P x y →
    r1 x y Q2 - r2 x y Q1 = 1/3 := 
sorry

end max_r1_minus_r2_l66_66680


namespace product_increased_five_times_l66_66346

variables (A B : ℝ)

theorem product_increased_five_times (h : A * B = 1.6) : (5 * A) * (5 * B) = 40 :=
by
  sorry

end product_increased_five_times_l66_66346


namespace erased_number_l66_66468

theorem erased_number (n i : ℕ) (h : (n * (n + 1) / 2 - i) / (n - 1) = 602 / 17) : i = 7 :=
sorry

end erased_number_l66_66468


namespace total_cost_of_plates_and_cups_l66_66510

theorem total_cost_of_plates_and_cups 
  (P C : ℝ)
  (h : 100 * P + 200 * C = 7.50) :
  20 * P + 40 * C = 1.50 :=
by
  sorry

end total_cost_of_plates_and_cups_l66_66510


namespace total_number_of_members_l66_66793

-- Define the basic setup
def committees := Fin 5
def members := {m : Finset committees // m.card = 2}

-- State the theorem
theorem total_number_of_members :
  (∃ s : Finset members, s.card = 10) :=
sorry

end total_number_of_members_l66_66793


namespace problem_inequality_l66_66442

theorem problem_inequality (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a + b + c + 2 = a * b * c) :
  (a + 1) * (b + 1) * (c + 1) ≥ 27 :=
sorry

end problem_inequality_l66_66442


namespace average_of_scores_with_average_twice_l66_66884

variable (scores: List ℝ) (A: ℝ) (A': ℝ)
variable (h1: scores.length = 50)
variable (h2: A = (scores.sum) / 50)
variable (h3: A' = ((scores.sum + 2 * A) / 52))

theorem average_of_scores_with_average_twice (h1: scores.length = 50) (h2: A = (scores.sum) / 50) (h3: A' = ((scores.sum + 2 * A) / 52)) :
  A' = A :=
by
  sorry

end average_of_scores_with_average_twice_l66_66884


namespace sum_of_edges_l66_66549

theorem sum_of_edges (a b c : ℝ)
  (h1 : a * b * c = 8)
  (h2 : 2 * (a * b + b * c + c * a) = 32)
  (h3 : b ^ 2 = a * c) :
  4 * (a + b + c) = 32 := 
sorry

end sum_of_edges_l66_66549


namespace initial_donuts_30_l66_66900

variable (x y : ℝ)
variable (p : ℝ := 0.30)

theorem initial_donuts_30 (h1 : y = 9) (h2 : y = p * x) : x = 30 := by
  sorry

end initial_donuts_30_l66_66900


namespace unknown_angles_are_80_l66_66268

theorem unknown_angles_are_80 (y : ℝ) (h1 : y + y + 200 = 360) : y = 80 :=
by
  sorry

end unknown_angles_are_80_l66_66268


namespace cost_of_1000_gums_in_dollars_l66_66802

theorem cost_of_1000_gums_in_dollars :
  let cost_per_piece_in_cents := 1
  let pieces := 1000
  let cents_per_dollar := 100
  ∃ cost_in_dollars : ℝ, cost_in_dollars = (cost_per_piece_in_cents * pieces) / cents_per_dollar :=
sorry

end cost_of_1000_gums_in_dollars_l66_66802


namespace double_neg_cancel_l66_66688

theorem double_neg_cancel (a : ℤ) : - (-2) = 2 :=
sorry

end double_neg_cancel_l66_66688


namespace circle_equation_solution_l66_66362

theorem circle_equation_solution (m : ℝ) :
  (∃ x y : ℝ, x^2 + y^2 - 2 * m * x - 2 * m * y + 2 * m^2 + m - 1 = 0) ↔ m < 1 :=
sorry

end circle_equation_solution_l66_66362


namespace sequence_b_l66_66676

theorem sequence_b (b : ℕ → ℝ) (h₁ : b 1 = 1)
  (h₂ : ∀ n : ℕ, n ≥ 1 → (b (n + 1)) ^ 4 = 64 * (b n) ^ 4) :
  b 50 = 2 ^ 49 := by
  sorry

end sequence_b_l66_66676


namespace find_function_l66_66861

theorem find_function (α : ℝ) (hα : 0 < α) (f : ℕ+ → ℝ) 
  (h : ∀ k m : ℕ+, α * m ≤ k → k ≤ (α + 1) * m → f (k + m) = f k + f m) :
  ∃ D : ℝ, ∀ n : ℕ+, f n = n * D :=
sorry

end find_function_l66_66861


namespace expression_value_l66_66628

theorem expression_value (a b : ℝ) (h : a^2 * b^2 / (a^4 - 2 * b^4) = 1) : 
  (a^2 - b^2) / (a^2 + b^2) = 1 / 3 := 
by 
  sorry

end expression_value_l66_66628


namespace oranges_in_second_group_l66_66164

namespace oranges_problem

-- Definitions coming from conditions
def cost_of_apple : ℝ := 0.21
def total_cost_1 : ℝ := 1.77
def total_cost_2 : ℝ := 1.27
def num_apples_group1 : ℕ := 6
def num_oranges_group1 : ℕ := 3
def num_apples_group2 : ℕ := 2
def cost_of_orange : ℝ := 0.17
def num_oranges_group2 : ℕ := 5 -- derived from the solution involving $0.85/$0.17.

-- Price calculation functions and conditions
def price_group1 (cost_of_orange : ℝ) : ℝ :=
  num_apples_group1 * cost_of_apple + num_oranges_group1 * cost_of_orange

def price_group2 (num_oranges_group2 cost_of_orange : ℝ) : ℝ :=
  num_apples_group2 * cost_of_apple + num_oranges_group2 * cost_of_orange

theorem oranges_in_second_group :
  (price_group1 cost_of_orange = total_cost_1) →
  (price_group2 num_oranges_group2 cost_of_orange = total_cost_2) →
  num_oranges_group2 = 5 :=
by
  intros h1 h2
  sorry

end oranges_problem

end oranges_in_second_group_l66_66164


namespace solve_x_l66_66466

theorem solve_x (x: ℝ) (h: -4 * x - 15 = 12 * x + 5) : x = -5 / 4 :=
sorry

end solve_x_l66_66466


namespace Mary_books_check_out_l66_66289

theorem Mary_books_check_out
  (initial_books : ℕ)
  (returned_unhelpful_books : ℕ)
  (returned_later_books : ℕ)
  (checked_out_later_books : ℕ)
  (total_books_now : ℕ)
  (h1 : initial_books = 5)
  (h2 : returned_unhelpful_books = 3)
  (h3 : returned_later_books = 2)
  (h4 : checked_out_later_books = 7)
  (h5 : total_books_now = 12) :
  ∃ (x : ℕ), (initial_books - returned_unhelpful_books + x - returned_later_books + checked_out_later_books = total_books_now) ∧ x = 5 :=
by {
  sorry
}

end Mary_books_check_out_l66_66289


namespace total_amount_to_pay_l66_66681

theorem total_amount_to_pay (cost_earbuds cost_smartwatch : ℕ) (tax_rate_earbuds tax_rate_smartwatch : ℚ) 
  (h1 : cost_earbuds = 200) (h2 : cost_smartwatch = 300) 
  (h3 : tax_rate_earbuds = 0.15) (h4 : tax_rate_smartwatch = 0.12) : 
  (cost_earbuds + cost_earbuds * tax_rate_earbuds + cost_smartwatch + cost_smartwatch * tax_rate_smartwatch = 566) := 
by 
  sorry

end total_amount_to_pay_l66_66681


namespace integer_solutions_count_l66_66426

theorem integer_solutions_count :
  ∃ (count : ℤ), (∀ (a : ℤ), 
  (∃ x : ℤ, x^2 + a * x + 8 * a = 0) ↔ count = 8) :=
sorry

end integer_solutions_count_l66_66426


namespace polynomial_identity_and_sum_of_squares_l66_66015

theorem polynomial_identity_and_sum_of_squares :
  ∃ (p q r s t u : ℤ), (∀ (x : ℤ), 512 * x^3 + 64 = (p * x^2 + q * x + r) * (s * x^2 + t * x + u)) ∧
    p^2 + q^2 + r^2 + s^2 + t^2 + u^2 = 5472 :=
sorry

end polynomial_identity_and_sum_of_squares_l66_66015


namespace area_of_hexagon_correct_l66_66941

variable (α β γ : ℝ) (S : ℝ) (r R : ℝ)
variable (AB BC AC : ℝ)
variable (A' B' C' : ℝ)

noncomputable def area_of_hexagon (AB BC AC : ℝ) (R : ℝ) (S : ℝ) (r : ℝ) : ℝ :=
  2 * (S / (r * r))

theorem area_of_hexagon_correct
  (hAB : AB = 13) (hBC : BC = 14) (hAC : AC = 15)
  (hR : R = 65 / 8) (hS : S = 1344 / 65) :
  area_of_hexagon AB BC AC R S r = 2 * (S / (r * r)) :=
sorry

end area_of_hexagon_correct_l66_66941


namespace percentage_of_percentage_l66_66028

theorem percentage_of_percentage (a b : ℝ) (h_a : a = 0.03) (h_b : b = 0.05) : (a / b) * 100 = 60 :=
by
  sorry

end percentage_of_percentage_l66_66028


namespace odd_function_negative_value_l66_66366

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_negative_value {f : ℝ → ℝ} (h_odd : is_odd_function f) :
  (∀ x, 0 < x → f x = x^2 - x - 1) → (∀ x, x < 0 → f x = -x^2 - x + 1) :=
by
  sorry

end odd_function_negative_value_l66_66366


namespace gray_part_area_l66_66345

theorem gray_part_area (area_rect1 area_rect2 area_black area_white gray_part_area : ℕ)
  (h_rect1 : area_rect1 = 80)
  (h_rect2 : area_rect2 = 108)
  (h_black : area_black = 37)
  (h_white : area_white = area_rect1 - area_black)
  (h_white_correct : area_white = 43)
  : gray_part_area = area_rect2 - area_white :=
by
  sorry

end gray_part_area_l66_66345


namespace product_of_decimals_l66_66201

theorem product_of_decimals :
  0.5 * 0.8 = 0.40 :=
by
  -- Proof will go here; using sorry to skip for now
  sorry

end product_of_decimals_l66_66201


namespace average_time_for_relay_race_l66_66544

noncomputable def average_leg_time (y_time z_time w_time x_time : ℕ) : ℚ :=
  (y_time + z_time + w_time + x_time) / 4

theorem average_time_for_relay_race :
  let y_time := 58
  let z_time := 26
  let w_time := 2 * z_time
  let x_time := 35
  average_leg_time y_time z_time w_time x_time = 42.75 := by
    sorry

end average_time_for_relay_race_l66_66544


namespace number_of_pencils_l66_66191

-- Definitions based on the conditions
def ratio_pens_pencils (P L : ℕ) : Prop := P * 6 = 5 * L
def pencils_more_than_pens (P L : ℕ) : Prop := L = P + 4

-- Statement to prove the number of pencils
theorem number_of_pencils : ∃ L : ℕ, (∃ P : ℕ, ratio_pens_pencils P L ∧ pencils_more_than_pens P L) ∧ L = 24 :=
by
  sorry

end number_of_pencils_l66_66191


namespace cos_C_in_acute_triangle_l66_66750

theorem cos_C_in_acute_triangle 
  (a b c : ℝ) (A B C : ℝ) 
  (h_acute : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2)
  (h_sides_angles : a * Real.cos B = 4 * c * Real.sin C - b * Real.cos A) 
  : Real.cos C = Real.sqrt 15 / 4 := 
sorry

end cos_C_in_acute_triangle_l66_66750


namespace phone_number_C_value_l66_66907

/-- 
In a phone number formatted as ABC-DEF-GHIJ, each letter symbolizes a distinct digit.
Digits in each section ABC, DEF, and GHIJ are in ascending order i.e., A < B < C, D < E < F, and G < H < I < J.
Moreover, D, E, F are consecutive odd digits, and G, H, I, J are consecutive even digits.
Also, A + B + C = 15. Prove that the value of C is 9. 
-/
theorem phone_number_C_value :
  ∃ (A B C D E F G H I J : ℕ), 
  A < B ∧ B < C ∧ D < E ∧ E < F ∧ G < H ∧ H < I ∧ I < J ∧
  (D % 2 = 1) ∧ (E % 2 = 1) ∧ (F % 2 = 1) ∧
  (G % 2 = 0) ∧ (H % 2 = 0) ∧ (I % 2 = 0) ∧ (J % 2 = 0) ∧
  (E = D + 2) ∧ (F = D + 4) ∧ (H = G + 2) ∧ (I = G + 4) ∧ (J = G + 6) ∧
  A + B + C = 15 ∧
  C = 9 := by 
  sorry

end phone_number_C_value_l66_66907


namespace triangle_side_length_x_l66_66010

theorem triangle_side_length_x
  (y : ℝ) (z : ℝ) (cos_Y_minus_Z : ℝ)
  (hy : y = 7)
  (hz : z = 3)
  (hcos : cos_Y_minus_Z = 7 / 8) :
  ∃ x : ℝ, x = Real.sqrt 18.625 :=
by
  sorry

end triangle_side_length_x_l66_66010


namespace point_on_x_axis_l66_66618

theorem point_on_x_axis (A B C D : ℝ × ℝ) : B = (3,0) → B.2 = 0 :=
by
  intros h
  subst h
  exact rfl

end point_on_x_axis_l66_66618


namespace jennifer_fruits_left_l66_66122

-- Definitions based on the conditions
def pears : ℕ := 15
def oranges : ℕ := 30
def apples : ℕ := 2 * pears
def cherries : ℕ := oranges / 2
def grapes : ℕ := 3 * apples
def pineapples : ℕ := pears + oranges + apples + cherries + grapes

-- Definitions for the number of fruits given to the sister
def pears_given : ℕ := 3
def oranges_given : ℕ := 5
def apples_given : ℕ := 5
def cherries_given : ℕ := 7
def grapes_given : ℕ := 3

-- Calculations based on the conditions for what's left after giving fruits
def pears_left : ℕ := pears - pears_given
def oranges_left : ℕ := oranges - oranges_given
def apples_left : ℕ := apples - apples_given
def cherries_left : ℕ := cherries - cherries_given
def grapes_left : ℕ := grapes - grapes_given

def remaining_pineapples : ℕ := pineapples - (pineapples / 2)

-- Total number of fruits left
def total_fruits_left : ℕ := pears_left + oranges_left + apples_left + cherries_left + grapes_left + remaining_pineapples

-- Theorem statement
theorem jennifer_fruits_left : total_fruits_left = 247 :=
by
  -- The detailed proof would go here
  sorry

end jennifer_fruits_left_l66_66122


namespace find_pairs_l66_66665

theorem find_pairs (a b : ℕ) (ha : 0 < a) (hb : 0 < b) : 
  (3 * a + 1 ∣ 4 * b - 1) ∧ (2 * b + 1 ∣ 3 * a - 1) ↔ (a = 2 ∧ b = 2) := 
by 
  sorry

end find_pairs_l66_66665


namespace brian_tape_needed_l66_66107

-- Define lengths and number of each type of box
def long_side_15_30 := 32
def short_side_15_30 := 17
def num_15_30 := 5

def side_40_40 := 42
def num_40_40 := 2

def long_side_20_50 := 52
def short_side_20_50 := 22
def num_20_50 := 3

-- Calculate the total tape required
def total_tape : Nat :=
  (num_15_30 * (long_side_15_30 + 2 * short_side_15_30)) +
  (num_40_40 * (3 * side_40_40)) +
  (num_20_50 * (long_side_20_50 + 2 * short_side_20_50))

-- Proof statement
theorem brian_tape_needed : total_tape = 870 := by
  sorry

end brian_tape_needed_l66_66107


namespace additional_hours_on_days_without_practice_l66_66697

def total_weekday_homework_hours : ℕ := 2 + 3 + 4 + 3 + 1
def total_weekend_homework_hours : ℕ := 8
def total_homework_hours : ℕ := total_weekday_homework_hours + total_weekend_homework_hours
def total_chore_hours : ℕ := 1 + 1
def total_hours : ℕ := total_homework_hours + total_chore_hours

theorem additional_hours_on_days_without_practice : ∀ (practice_nights : ℕ), 
  (2 ≤ practice_nights ∧ practice_nights ≤ 3) →
  (∃ tuesday_wednesday_thursday_weekend_day_hours : ℕ,
    tuesday_wednesday_thursday_weekend_day_hours = 15) :=
by
  intros practice_nights practice_nights_bounds
  -- Define days without practice in the worst case scenario
  let tuesday_hours := 3
  let wednesday_homework_hours := 4
  let wednesday_chore_hours := 1
  let thursday_hours := 3
  let weekend_day_hours := 4
  let days_without_practice_hours := tuesday_hours + (wednesday_homework_hours + wednesday_chore_hours) + thursday_hours + weekend_day_hours
  use days_without_practice_hours
  -- In the worst case, the total additional hours on days without practice should be 15.
  sorry

end additional_hours_on_days_without_practice_l66_66697


namespace janice_class_girls_l66_66612

theorem janice_class_girls : ∃ (g b : ℕ), (3 * b = 4 * g) ∧ (g + b + 2 = 32) ∧ (g = 13) := by
  sorry

end janice_class_girls_l66_66612


namespace range_of_a_l66_66169

variable (a : ℝ)
def f (x : ℝ) : ℝ := a * x^2 - 2 * a * x - 4

theorem range_of_a :
  (∀ x : ℝ, f a x < 0) → (-4 < a ∧ a ≤ 0) :=
by
  sorry

end range_of_a_l66_66169


namespace radius_of_circle_l66_66894

def circle_eq_def (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 9

theorem radius_of_circle {x y r : ℝ} (h : circle_eq_def x y) : r = 3 := 
by
  -- Proof skipped
  sorry

end radius_of_circle_l66_66894


namespace divisible_by_five_l66_66127

theorem divisible_by_five (x y z : ℤ) (hxy : x ≠ y) (hyz : y ≠ z) (hzx : z ≠ x) :
  ∃ k : ℤ, (x-y)^5 + (y-z)^5 + (z-x)^5 = 5 * k * (y-z) * (z-x) * (x-y) :=
  sorry

end divisible_by_five_l66_66127


namespace number_of_tulips_l66_66093

theorem number_of_tulips (T : ℕ) (roses : ℕ) (used_flowers : ℕ) (extra_flowers : ℕ) 
  (total_flowers : ℕ) (h1 : roses = 37) (h2 : used_flowers = 70) 
  (h3 : extra_flowers = 3) (h4: total_flowers = 73) 
  (h5 : T + roses = total_flowers) : T = 36 := 
by
  sorry

end number_of_tulips_l66_66093


namespace janet_pills_monthly_l66_66034

def daily_intake_first_two_weeks := 2 + 3 -- 2 multivitamins + 3 calcium supplements
def daily_intake_last_two_weeks := 2 + 1 -- 2 multivitamins + 1 calcium supplement
def days_in_two_weeks := 2 * 7

theorem janet_pills_monthly :
  (daily_intake_first_two_weeks * days_in_two_weeks) + (daily_intake_last_two_weeks * days_in_two_weeks) = 112 :=
by
  sorry

end janet_pills_monthly_l66_66034


namespace AC_total_l66_66078

theorem AC_total (A B C : ℕ) (h1 : A + B + C = 600) (h2 : B + C = 450) (h3 : C = 100) : A + C = 250 := by
  sorry

end AC_total_l66_66078


namespace area_relationship_l66_66284

theorem area_relationship (a b c : ℝ) (h : a^2 + b^2 = c^2) : (a + b)^2 = a^2 + 2*a*b + b^2 := 
by sorry

end area_relationship_l66_66284


namespace wipes_per_pack_l66_66133

theorem wipes_per_pack (days : ℕ) (wipes_per_day : ℕ) (packs : ℕ) (total_wipes : ℕ) (n : ℕ)
    (h1 : days = 360)
    (h2 : wipes_per_day = 2)
    (h3 : packs = 6)
    (h4 : total_wipes = wipes_per_day * days)
    (h5 : total_wipes = n * packs) : 
    n = 120 := 
by 
  sorry

end wipes_per_pack_l66_66133


namespace handshaking_pairs_l66_66966

-- Definition of the problem: Given 8 people, pair them up uniquely and count the ways modulo 1000
theorem handshaking_pairs (N : ℕ) (H : N=105) : (N % 1000) = 105 :=
by {
  -- The proof is omitted.
  sorry
}

end handshaking_pairs_l66_66966


namespace john_age_multiple_of_james_age_l66_66429

-- Define variables for the problem conditions
def john_current_age : ℕ := 39
def john_age_3_years_ago : ℕ := john_current_age - 3

def james_brother_age : ℕ := 16
def james_brother_older : ℕ := 4

def james_current_age : ℕ := james_brother_age - james_brother_older
def james_age_in_6_years : ℕ := james_current_age + 6

-- The goal is to prove the multiple relationship
theorem john_age_multiple_of_james_age :
  john_age_3_years_ago = 2 * james_age_in_6_years :=
by {
  -- Skip the proof
  sorry
}

end john_age_multiple_of_james_age_l66_66429


namespace objective_function_range_l66_66334

theorem objective_function_range (x y : ℝ) 
  (h1 : x + 2 * y > 2) 
  (h2 : 2 * x + y ≤ 4) 
  (h3 : 4 * x - y ≥ 1) : 
  ∃ z_min z_max : ℝ, (∀ z : ℝ, z = 3 * x + y → z_min ≤ z ∧ z ≤ z_max) ∧ z_min = 1 ∧ z_max = 6 := 
sorry

end objective_function_range_l66_66334


namespace flower_team_participation_l66_66364

-- Definitions based on the conditions in the problem
def num_rows : ℕ := 60
def first_row_people : ℕ := 40
def people_increment : ℕ := 1

-- Statement to be proved in Lean
theorem flower_team_participation (x : ℕ) (hx : 1 ≤ x ∧ x ≤ num_rows) : 
  ∃ y : ℕ, y = first_row_people - people_increment + x :=
by
  -- Placeholder for the proof
  sorry

end flower_team_participation_l66_66364


namespace squirrel_travel_time_l66_66704

theorem squirrel_travel_time :
  ∀ (speed distance : ℝ), speed = 5 → distance = 3 →
  (distance / speed) * 60 = 36 := by
  intros speed distance h_speed h_distance
  rw [h_speed, h_distance]
  norm_num

end squirrel_travel_time_l66_66704


namespace sufficient_not_necessary_l66_66767

theorem sufficient_not_necessary (a : ℝ) : (a > 1 → a^2 > 1) ∧ ¬(a^2 > 1 → a > 1) :=
by {
  sorry
}

end sufficient_not_necessary_l66_66767


namespace pencil_cost_l66_66700

theorem pencil_cost (P : ℝ) : 
  (∀ pen_cost total : ℝ, pen_cost = 3.50 → total = 291 → 38 * P + 56 * pen_cost = total → P = 2.50) :=
by
  intros pen_cost total h1 h2 h3
  sorry

end pencil_cost_l66_66700


namespace combinatorial_calculation_l66_66717

-- Define the proof problem.
theorem combinatorial_calculation : (Nat.choose 20 6) = 2583 := sorry

end combinatorial_calculation_l66_66717


namespace carly_practice_time_l66_66828

-- conditions
def practice_time_butterfly_weekly (hours_per_day : ℕ) (days_per_week : ℕ) : ℕ :=
  hours_per_day * days_per_week

def practice_time_backstroke_weekly (hours_per_day : ℕ) (days_per_week : ℕ) : ℕ :=
  hours_per_day * days_per_week

def total_weekly_practice (butterfly_hours : ℕ) (backstroke_hours : ℕ) : ℕ :=
  butterfly_hours + backstroke_hours

def monthly_practice (weekly_hours : ℕ) (weeks_per_month : ℕ) : ℕ :=
  weekly_hours * weeks_per_month

-- Proof Problem Statement
theorem carly_practice_time :
  practice_time_butterfly_weekly 3 4 + practice_time_backstroke_weekly 2 6 * 4 = 96 :=
by
  sorry

end carly_practice_time_l66_66828


namespace age_problem_l66_66132

theorem age_problem (c b a : ℕ) (h1 : b = 2 * c) (h2 : a = b + 2) (h3 : a + b + c = 47) : b = 18 :=
by
  sorry

end age_problem_l66_66132


namespace mara_correct_answers_l66_66161

theorem mara_correct_answers :
  let math_total    := 30
  let science_total := 20
  let history_total := 50
  let math_percent  := 0.85
  let science_percent := 0.75
  let history_percent := 0.65
  let math_correct  := math_percent * math_total
  let science_correct := science_percent * science_total
  let history_correct := history_percent * history_total
  let total_correct := math_correct + science_correct + history_correct
  let total_problems := math_total + science_total + history_total
  let overall_percent := total_correct / total_problems
  overall_percent = 0.73 :=
by
  sorry

end mara_correct_answers_l66_66161


namespace neg_sqrt_comparison_l66_66885

theorem neg_sqrt_comparison : -Real.sqrt 7 > -Real.sqrt 11 := by
  sorry

end neg_sqrt_comparison_l66_66885


namespace candy_given_away_l66_66025

-- Define the conditions
def pieces_per_student := 2
def number_of_students := 9

-- Define the problem statement as a theorem
theorem candy_given_away : pieces_per_student * number_of_students = 18 := by
  -- This is where the proof would go, but we omit it with sorry.
  sorry

end candy_given_away_l66_66025


namespace polynomial_divisibility_l66_66131

def poly1 (x : ℝ) (k : ℝ) : ℝ := 3*x^3 - 9*x^2 + k*x - 12

theorem polynomial_divisibility (k : ℝ) :
  (∀ (x : ℝ), poly1 x k = (x - 3) * (3*x^2 + 4)) → (poly1 3 k = 0) := sorry

end polynomial_divisibility_l66_66131


namespace earning_80_yuan_represents_l66_66662

-- Defining the context of the problem
def spending (n : Int) : Int := -n
def earning (n : Int) : Int := n

-- The problem statement as a Lean theorem
theorem earning_80_yuan_represents (x : Int) (hx : earning x = 80) : x = 80 := 
by
  sorry

end earning_80_yuan_represents_l66_66662


namespace rationalize_denominator_sum_l66_66273

theorem rationalize_denominator_sum :
  let A := 3
  let B := -9
  let C := -9
  let D := 9
  let E := 165
  let F := 51
  A + B + C + D + E + F = 210 :=
by
  let A := 3
  let B := -9
  let C := -9
  let D := 9
  let E := 165
  let F := 51
  show 3 + -9 + -9 + 9 + 165 + 51 = 210
  sorry

end rationalize_denominator_sum_l66_66273


namespace min_value_of_x_plus_2y_l66_66716

theorem min_value_of_x_plus_2y (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : 1 / x + 1 / y = 2) : 
  x + 2 * y ≥ (3 + 2 * Real.sqrt 2) / 2 :=
sorry

end min_value_of_x_plus_2y_l66_66716


namespace find_matrix_A_l66_66569

-- Define the condition that A v = 3 v for all v in R^3
def satisfiesCondition (A : Matrix (Fin 3) (Fin 3) ℝ) : Prop :=
  ∀ (v : Fin 3 → ℝ), A.mulVec v = 3 • v

theorem find_matrix_A (A : Matrix (Fin 3) (Fin 3) ℝ) :
  satisfiesCondition A → A = 3 • 1 :=
by
  intro h
  sorry

end find_matrix_A_l66_66569


namespace solve_for_x_l66_66056

theorem solve_for_x (x : ℝ) (h1 : (3 - x) / (x + 2) + (3 * x - 9) / (3 - x) = 2) 
  (h2 : x ≠ -2) (h3 : x ≠ 3) : x = -1 / 6 :=
by
  sorry

end solve_for_x_l66_66056


namespace area_of_new_geometric_figure_correct_l66_66447

noncomputable def area_of_new_geometric_figure (a b : ℝ) : ℝ := 
  let d := Real.sqrt (a^2 + b^2)
  a * b + (b * d) / 4

theorem area_of_new_geometric_figure_correct (a b : ℝ) :
  area_of_new_geometric_figure a b = a * b + (b * Real.sqrt (a^2 + b^2)) / 4 :=
by 
  sorry

end area_of_new_geometric_figure_correct_l66_66447


namespace angle_D_is_20_degrees_l66_66347

theorem angle_D_is_20_degrees (A B C D : ℝ) 
  (h1 : A + B = 180)
  (h2 : C = D)
  (h3 : A = 40)
  (h4 : B + C = 160) : D = 20 :=
by
  sorry

end angle_D_is_20_degrees_l66_66347


namespace average_rate_of_change_nonzero_l66_66150

-- Define the conditions related to the average rate of change.
variables {x0 : ℝ} {Δx : ℝ}

-- Define the statement to prove that in the definition of the average rate of change, Δx ≠ 0.
theorem average_rate_of_change_nonzero (h : Δx ≠ 0) : True :=
sorry  -- The proof is omitted as per instruction.

end average_rate_of_change_nonzero_l66_66150


namespace minimize_cost_l66_66762

-- Define the prices at each salon
def GustranSalonHaircut : ℕ := 45
def GustranSalonFacial : ℕ := 22
def GustranSalonNails : ℕ := 30

def BarbarasShopHaircut : ℕ := 30
def BarbarasShopFacial : ℕ := 28
def BarbarasShopNails : ℕ := 40

def FancySalonHaircut : ℕ := 34
def FancySalonFacial : ℕ := 30
def FancySalonNails : ℕ := 20

-- Define the total cost at each salon
def GustranSalonTotal : ℕ := GustranSalonHaircut + GustranSalonFacial + GustranSalonNails
def BarbarasShopTotal : ℕ := BarbarasShopHaircut + BarbarasShopFacial + BarbarasShopNails
def FancySalonTotal : ℕ := FancySalonHaircut + FancySalonFacial + FancySalonNails

-- Prove that the minimum total cost is $84
theorem minimize_cost : min GustranSalonTotal (min BarbarasShopTotal FancySalonTotal) = 84 := by
  -- proof goes here
  sorry

end minimize_cost_l66_66762


namespace sibling_of_5_over_7_child_unique_parent_one_over_2008_descendent_of_one_l66_66280

-- Part (a):
def sibling_of_frac (x : ℚ) : Prop :=
  x = 5/7

theorem sibling_of_5_over_7 : ∃ (y : ℚ), sibling_of_frac (y / (y + 1)) ∧ y + 1 = 7/2 :=
  sorry

-- Part (b):
def child (x y : ℚ) : Prop :=
  y = x + 1 ∨ y = x / (x + 1)

theorem child_unique_parent (x y z : ℚ) (hx : 0 < x) (hz : 0 < z) (hyx : child x y) (hyz : child z y) : x = z :=
  sorry

-- Part (c):
def descendent (x y : ℚ) : Prop :=
  ∃ n : ℕ, y = 1 / (x + n)

theorem one_over_2008_descendent_of_one : descendent 1 (1 / 2008) :=
  sorry

end sibling_of_5_over_7_child_unique_parent_one_over_2008_descendent_of_one_l66_66280


namespace final_price_of_book_l66_66998

theorem final_price_of_book (original_price : ℝ) (d1_percentage : ℝ) (d2_percentage : ℝ) 
  (first_discount : ℝ) (second_discount : ℝ) (new_price1 : ℝ) (final_price : ℝ) :
  original_price = 15 ∧ d1_percentage = 0.20 ∧ d2_percentage = 0.25 ∧
  first_discount = d1_percentage * original_price ∧ new_price1 = original_price - first_discount ∧
  second_discount = d2_percentage * new_price1 ∧ 
  final_price = new_price1 - second_discount → final_price = 9 := 
by 
  sorry

end final_price_of_book_l66_66998


namespace lucia_hiphop_classes_l66_66669

def cost_hiphop_class : Int := 10
def cost_ballet_class : Int := 12
def cost_jazz_class : Int := 8
def num_ballet_classes : Int := 2
def num_jazz_classes : Int := 1
def total_cost : Int := 52

def num_hiphop_classes : Int := (total_cost - (num_ballet_classes * cost_ballet_class + num_jazz_classes * cost_jazz_class)) / cost_hiphop_class

theorem lucia_hiphop_classes : num_hiphop_classes = 2 := by
  sorry

end lucia_hiphop_classes_l66_66669


namespace remainder_determined_l66_66540

theorem remainder_determined (p a b : ℤ) (h₀: Nat.Prime (Int.natAbs p)) (h₁ : ¬ (p ∣ a)) (h₂ : ¬ (p ∣ b)) :
  ∃ (r : ℤ), (r ≡ a [ZMOD p]) ∧ (r ≡ b [ZMOD p]) ∧ (r ≡ (a * b) [ZMOD p]) →
  (a ≡ r [ZMOD p]) := sorry

end remainder_determined_l66_66540


namespace division_problem_l66_66208

variables (a b c : ℤ)

theorem division_problem 
  (h1 : a ∣ b * c - 1)
  (h2 : b ∣ c * a - 1)
  (h3 : c ∣ a * b - 1) : 
  abc ∣ ab + bc + ca - 1 := 
sorry

end division_problem_l66_66208


namespace zeke_estimate_smaller_l66_66317

variable (x y k : ℝ)
variable (hx_pos : 0 < x)
variable (hy_pos : 0 < y)
variable (h_inequality : x > 2 * y)
variable (hk_pos : 0 < k)

theorem zeke_estimate_smaller : (x + k) - 2 * (y + k) < x - 2 * y :=
by
  sorry

end zeke_estimate_smaller_l66_66317


namespace xiao_ming_percentile_l66_66693

theorem xiao_ming_percentile (total_students : ℕ) (rank : ℕ) 
  (h1 : total_students = 48) (h2 : rank = 5) :
  ∃ p : ℕ, (p = 90 ∨ p = 91) ∧ (43 < (p * total_students) / 100) ∧ ((p * total_students) / 100 ≤ 44) :=
by
  sorry

end xiao_ming_percentile_l66_66693


namespace afternoon_pear_sales_l66_66142

theorem afternoon_pear_sales (morning_sales afternoon_sales total_sales : ℕ)
  (h1 : afternoon_sales = 2 * morning_sales)
  (h2 : total_sales = morning_sales + afternoon_sales)
  (h3 : total_sales = 420) : 
  afternoon_sales = 280 :=
by {
  -- placeholders for the proof
  sorry 
}

end afternoon_pear_sales_l66_66142


namespace rhombus_diagonal_BD_equation_rhombus_diagonal_AD_equation_l66_66359

theorem rhombus_diagonal_BD_equation (A C : ℝ × ℝ) (AB_eq : ∀ x y : ℝ, 3 * x - y + 2 = 0) : 
  A = (0, 2) ∧ C = (4, 6) → ∃ k b : ℝ, k = 1 ∧ b = 6 ∧ ∀ x y : ℝ, x + y - 6 = 0 := by
  sorry

theorem rhombus_diagonal_AD_equation (A C : ℝ × ℝ) (AB_eq BD_eq : ∀ x y : ℝ, 3 * x - y + 2 = 0 ∧ x + y - 6 = 0) : 
  A = (0, 2) ∧ C = (4, 6) → ∃ k b : ℝ, k = 3 ∧ b = 14 ∧ ∀ x y : ℝ, x - 3 * y + 14 = 0 := by
  sorry

end rhombus_diagonal_BD_equation_rhombus_diagonal_AD_equation_l66_66359


namespace samuel_apples_left_l66_66923

def bonnieApples : ℕ := 8
def extraApples : ℕ := 20
def samuelTotalApples : ℕ := bonnieApples + extraApples
def samuelAte : ℕ := samuelTotalApples / 2
def samuelRemainingAfterEating : ℕ := samuelTotalApples - samuelAte
def samuelUsedForPie : ℕ := samuelRemainingAfterEating / 7
def samuelFinalRemaining : ℕ := samuelRemainingAfterEating - samuelUsedForPie

theorem samuel_apples_left :
  samuelFinalRemaining = 12 := by
  sorry

end samuel_apples_left_l66_66923


namespace sum_squares_seven_consecutive_not_perfect_square_l66_66880

theorem sum_squares_seven_consecutive_not_perfect_square : 
  ∀ (n : ℤ), ¬ ∃ k : ℤ, k * k = (n-3)^2 + (n-2)^2 + (n-1)^2 + n^2 + (n+1)^2 + (n+2)^2 + (n+3)^2 :=
by
  sorry

end sum_squares_seven_consecutive_not_perfect_square_l66_66880


namespace initial_volume_of_solution_l66_66064

theorem initial_volume_of_solution (V : ℝ) (h0 : 0.10 * V = 0.08 * (V + 20)) : V = 80 :=
by
  sorry

end initial_volume_of_solution_l66_66064


namespace induction_step_l66_66930

theorem induction_step (x y : ℕ) (k : ℕ) (odd_k : k % 2 = 1) 
  (hk : (x + y) ∣ (x^k + y^k)) : (x + y) ∣ (x^(k+2) + y^(k+2)) :=
sorry

end induction_step_l66_66930


namespace man_l66_66407

theorem man's_speed_downstream (v : ℝ) (speed_of_stream : ℝ) (speed_upstream : ℝ) : 
  speed_upstream = v - speed_of_stream ∧ speed_of_stream = 1.5 ∧ speed_upstream = 8 → v + speed_of_stream = 11 :=
by
  sorry

end man_l66_66407


namespace john_ate_12_ounces_of_steak_l66_66443

-- Conditions
def original_weight : ℝ := 30
def burned_fraction : ℝ := 0.5
def eaten_fraction : ℝ := 0.8

-- Theorem statement
theorem john_ate_12_ounces_of_steak :
  (original_weight * (1 - burned_fraction) * eaten_fraction) = 12 := by
  sorry

end john_ate_12_ounces_of_steak_l66_66443


namespace sector_central_angle_l66_66804

theorem sector_central_angle (r l θ : ℝ) (h_perimeter : 2 * r + l = 8) (h_area : (1 / 2) * l * r = 4) : θ = 2 :=
by
  sorry

end sector_central_angle_l66_66804


namespace workshop_output_comparison_l66_66228

theorem workshop_output_comparison (a x : ℝ)
  (h1 : ∀n:ℕ, n ≥ 0 → (1 + n * a) = (1 + x)^n) :
  (1 + 3 * a) > (1 + x)^3 := sorry

end workshop_output_comparison_l66_66228


namespace find_number_l66_66921

theorem find_number (x : ℝ) (h : 1345 - x / 20.04 = 1295) : x = 1002 :=
sorry

end find_number_l66_66921


namespace johns_average_speed_l66_66351

theorem johns_average_speed :
  let distance1 := 20
  let speed1 := 10
  let distance2 := 30
  let speed2 := 20
  let total_distance := distance1 + distance2
  let time1 := distance1 / speed1
  let time2 := distance2 / speed2
  let total_time := time1 + time2
  let average_speed := total_distance / total_time
  average_speed = 14.29 :=
by
  sorry

end johns_average_speed_l66_66351


namespace math_test_score_l66_66830

theorem math_test_score (K E M : ℕ) 
  (h₁ : (K + E) / 2 = 92) 
  (h₂ : (K + E + M) / 3 = 94) : 
  M = 98 := 
by 
  sorry

end math_test_score_l66_66830


namespace rosie_laps_l66_66534

theorem rosie_laps (lou_distance : ℝ) (track_length : ℝ) (lou_speed_factor : ℝ) (rosie_speed_multiplier : ℝ) 
    (number_of_laps_by_lou : ℝ) (number_of_laps_by_rosie : ℕ) :
  lou_distance = 3 ∧ 
  track_length = 1 / 4 ∧ 
  lou_speed_factor = 0.75 ∧ 
  rosie_speed_multiplier = 2 ∧ 
  number_of_laps_by_lou = lou_distance / track_length ∧ 
  number_of_laps_by_rosie = rosie_speed_multiplier * number_of_laps_by_lou → 
  number_of_laps_by_rosie = 18 := 
sorry

end rosie_laps_l66_66534


namespace largest_prime_value_of_quadratic_expression_l66_66523

theorem largest_prime_value_of_quadratic_expression : 
  ∃ n : ℕ, n > 0 ∧ Prime (n^2 - 12 * n + 27) ∧ ∀ m : ℕ, m > 0 → Prime (m^2 - 12 * m + 27) → (n^2 - 12 * n + 27) ≥ (m^2 - 12 * m + 27) := 
by
  sorry


end largest_prime_value_of_quadratic_expression_l66_66523


namespace total_popsicle_sticks_l66_66465

def Gino_popsicle_sticks : ℕ := 63
def My_popsicle_sticks : ℕ := 50
def Nick_popsicle_sticks : ℕ := 82

theorem total_popsicle_sticks : Gino_popsicle_sticks + My_popsicle_sticks + Nick_popsicle_sticks = 195 := by
  sorry

end total_popsicle_sticks_l66_66465


namespace athlete_speed_l66_66645

theorem athlete_speed (distance time : ℝ) (h1 : distance = 200) (h2 : time = 25) :
  (distance / time) = 8 := by
  sorry

end athlete_speed_l66_66645


namespace original_kittens_count_l66_66718

theorem original_kittens_count 
  (K : ℕ) 
  (h1 : K - 3 + 9 = 12) : 
  K = 6 := by
sorry

end original_kittens_count_l66_66718


namespace expected_participants_in_2005_l66_66593

open Nat

def initial_participants : ℕ := 500
def annual_increase_rate : ℚ := 1.2
def num_years : ℕ := 5
def expected_participants_2005 : ℚ := 1244

theorem expected_participants_in_2005 :
  (initial_participants : ℚ) * annual_increase_rate ^ num_years = expected_participants_2005 := by
  sorry

end expected_participants_in_2005_l66_66593


namespace log_sum_l66_66486

-- Define the common logarithm function using Lean's natural logarithm with a change of base
noncomputable def log_base_10 (x : ℝ) : ℝ := Real.log x / Real.log 10

theorem log_sum : log_base_10 5 + log_base_10 0.2 = 0 :=
by
  -- Placeholder for the proof to be completed
  sorry

end log_sum_l66_66486


namespace sqrt_fraction_sum_as_common_fraction_l66_66396

theorem sqrt_fraction_sum_as_common_fraction (a b c d : ℚ) (ha : a = 25) (hb : b = 36) (hc : c = 16) (hd : d = 9) :
  Real.sqrt ((a / b) + (c / d)) = Real.sqrt 89 / 6 := by
  sorry

end sqrt_fraction_sum_as_common_fraction_l66_66396


namespace johns_sixth_quiz_score_l66_66325

theorem johns_sixth_quiz_score (s1 s2 s3 s4 s5 : ℕ) (mean : ℕ) (n : ℕ) :
  s1 = 86 ∧ s2 = 91 ∧ s3 = 83 ∧ s4 = 88 ∧ s5 = 97 ∧ mean = 90 ∧ n = 6 →
  ∃ s6 : ℕ, (s1 + s2 + s3 + s4 + s5 + s6) / n = mean ∧ s6 = 95 :=
by
  intro h
  obtain ⟨hs1, hs2, hs3, hs4, hs5, hmean, hn⟩ := h
  have htotal : s1 + s2 + s3 + s4 + s5 + 95 = 540 := by sorry
  have hmean_eq : (s1 + s2 + s3 + s4 + s5 + 95) / n = mean := by sorry
  exact ⟨95, hmean_eq, rfl⟩

end johns_sixth_quiz_score_l66_66325


namespace exercise_b_c_values_l66_66994

open Set

universe u

theorem exercise_b_c_values : 
  ∀ (b c : ℝ), let U : Set ℝ := {2, 3, 5}
               let A : Set ℝ := {x | x^2 + b * x + c = 0}
               (U \ A = {2}) → (b = -8 ∧ c = 15) :=
by
  intros b c U A H
  let U : Set ℝ := {2, 3, 5}
  let A : Set ℝ := {x | x^2 + b * x + c = 0}
  have H1 : U \ A = {2} := H
  sorry

end exercise_b_c_values_l66_66994


namespace part1_part2_l66_66600

-- Definitions and conditions
variables {A B C a b c : ℝ}
variable (h1 : sin C * sin (A - B) = sin B * sin (C - A)) -- Given condition

-- Part (1): If A = 2B, then find C
theorem part1 (h2 : A = 2 * B) : C = (5 / 8) * π := by
  sorry

-- Part (2): Prove that 2a² = b² + c²
theorem part2 : 2 * a^2 = b^2 + c^2 := by
  sorry

end part1_part2_l66_66600


namespace inequality_solution_l66_66242

theorem inequality_solution (m : ℝ) : 
  (∀ x : ℝ, 2 * x + 7 > 3 * x + 2 ∧ 2 * x - 2 < 2 * m → x < 5) → m ≥ 4 :=
by
  sorry

end inequality_solution_l66_66242


namespace complex_equilateral_triangle_expression_l66_66337

noncomputable def omega : ℂ :=
  Complex.exp (Complex.I * 2 * Real.pi / 3)

def is_root_of_quadratic (z : ℂ) (a b : ℂ) : Prop :=
  z^2 + a * z + b = 0

theorem complex_equilateral_triangle_expression (z1 z2 a b : ℂ) (h1 : is_root_of_quadratic z1 a b) 
  (h2 : is_root_of_quadratic z2 a b) (h3 : z2 = omega * z1) : a^2 / b = 1 := by
  sorry

end complex_equilateral_triangle_expression_l66_66337


namespace bike_growth_equation_l66_66527

-- Declare the parameters
variables (b1 b3 : ℕ) (x : ℝ)
-- Define the conditions
def condition1 : b1 = 1000 := sorry
def condition2 : b3 = b1 + 440 := sorry

-- Define the proposition to be proved
theorem bike_growth_equation (cond1 : b1 = 1000) (cond2 : b3 = b1 + 440) :
  b1 * (1 + x)^2 = b3 :=
sorry

end bike_growth_equation_l66_66527


namespace number_of_distinct_linear_recurrences_l66_66951

open BigOperators

/-
  Let p be a prime positive integer.
  Define a mod-p recurrence of degree n to be a sequence {a_k}_{k >= 0} of numbers modulo p 
  satisfying a relation of the form:

  ai+n = c_n-1 ai+n-1 + ... + c_1 ai+1 + c_0 ai
  for all i >= 0, where c_0, c_1, ..., c_n-1 are integers and c_0 not equivalent to 0 mod p.
  Compute the number of distinct linear recurrences of degree at most n in terms of p and n.
-/
theorem number_of_distinct_linear_recurrences (p n : ℕ) (hp : Nat.Prime p) : 
  ∃ d : ℕ, 
    (∀ {a : ℕ → ℕ} {c : ℕ → ℕ} (h : ∀ i, a (i + n) = ∑ j in Finset.range n, c j * a (i + j))
     (hc0 : c 0 ≠ 0), 
      d = (1 - n * (p - 1) / (p + 1) + p^2 * (p^(2 * n) - 1) / (p + 1)^2 : ℚ)) :=
  sorry

end number_of_distinct_linear_recurrences_l66_66951


namespace peach_tree_average_production_l66_66293

-- Definitions derived from the conditions
def num_apple_trees : ℕ := 30
def kg_per_apple_tree : ℕ := 150
def num_peach_trees : ℕ := 45
def total_mass_fruit : ℕ := 7425

-- Main Statement to be proven
theorem peach_tree_average_production : 
  (total_mass_fruit - (num_apple_trees * kg_per_apple_tree)) = (num_peach_trees * 65) :=
by
  sorry

end peach_tree_average_production_l66_66293


namespace berries_per_bird_per_day_l66_66249

theorem berries_per_bird_per_day (birds : ℕ) (total_berries : ℕ) (days : ℕ) (berries_per_bird_per_day : ℕ) 
  (h_birds : birds = 5)
  (h_total_berries : total_berries = 140)
  (h_days : days = 4) :
  berries_per_bird_per_day = 7 :=
  sorry

end berries_per_bird_per_day_l66_66249


namespace conic_section_is_ellipse_l66_66236

theorem conic_section_is_ellipse :
  ∀ x y : ℝ, 4 * x^2 + y^2 - 12 * x - 2 * y + 4 = 0 →
  ∃ a b h k : ℝ, a > 0 ∧ b > 0 ∧ (a * (x - h)^2 + b * (y - k)^2 = 1) :=
by
  sorry

end conic_section_is_ellipse_l66_66236


namespace intersection_M_N_l66_66973

def M : Set ℤ := { x | x^2 > 1 }
def N : Set ℤ := { -2, -1, 0, 1, 2 }

theorem intersection_M_N : (M ∩ N) = { -2, 2 } :=
sorry

end intersection_M_N_l66_66973


namespace percentage_of_copper_buttons_l66_66469

-- Definitions for conditions
def total_items : ℕ := 100
def pin_percentage : ℕ := 30
def button_percentage : ℕ := 100 - pin_percentage
def brass_button_percentage : ℕ := 60
def copper_button_percentage : ℕ := 100 - brass_button_percentage

-- Theorem statement proving the question
theorem percentage_of_copper_buttons (h1 : pin_percentage = 30)
  (h2 : button_percentage = total_items - pin_percentage)
  (h3 : brass_button_percentage = 60)
  (h4 : copper_button_percentage = total_items - brass_button_percentage) :
  (button_percentage * copper_button_percentage) / total_items = 28 := 
sorry

end percentage_of_copper_buttons_l66_66469


namespace Jorge_Giuliana_cakes_l66_66565

theorem Jorge_Giuliana_cakes (C : ℕ) :
  (2 * 7 + 2 * C + 2 * 30 = 110) → (C = 18) :=
by
  sorry

end Jorge_Giuliana_cakes_l66_66565


namespace find_milk_ounces_l66_66771

def bathroom_limit : ℕ := 32
def grape_juice_ounces : ℕ := 16
def water_ounces : ℕ := 8
def total_liquid_limit : ℕ := bathroom_limit
def total_liquid_intake : ℕ := grape_juice_ounces + water_ounces
def milk_ounces := total_liquid_limit - total_liquid_intake

theorem find_milk_ounces : milk_ounces = 8 := by
  sorry

end find_milk_ounces_l66_66771


namespace simplify_power_of_power_l66_66058

theorem simplify_power_of_power (a : ℝ) : (a^2)^3 = a^6 :=
by 
  sorry

end simplify_power_of_power_l66_66058


namespace sum_of_cubes_l66_66124

-- Definitions based on the conditions
variables (a b : ℝ)
variables (h1 : a + b = 2) (h2 : a * b = -3)

-- The Lean statement to prove the sum of their cubes is 26
theorem sum_of_cubes (a b : ℝ) (h1 : a + b = 2) (h2 : a * b = -3) : a^3 + b^3 = 26 :=
by
  sorry

end sum_of_cubes_l66_66124


namespace total_cost_is_96_l66_66313

noncomputable def hair_updo_cost : ℕ := 50
noncomputable def manicure_cost : ℕ := 30
noncomputable def tip_rate : ℚ := 0.20

def total_cost_with_tip (hair_cost manicure_cost : ℕ) (tip_rate : ℚ) : ℚ :=
  let hair_tip := hair_cost * tip_rate
  let manicure_tip := manicure_cost * tip_rate
  let total_tips := hair_tip + manicure_tip
  let total_before_tips := (hair_cost : ℚ) + (manicure_cost : ℚ)
  total_before_tips + total_tips

theorem total_cost_is_96 :
  total_cost_with_tip hair_updo_cost manicure_cost tip_rate = 96 := by
  sorry

end total_cost_is_96_l66_66313


namespace false_propositions_l66_66932

open Classical

theorem false_propositions :
  ¬ (∀ x : ℝ, x^2 + 3 < 0) ∧ ¬ (∀ x : ℕ, x^2 > 1) ∧ (∃ x : ℤ, x^5 < 1) ∧ ¬ (∃ x : ℚ, x^2 = 3) :=
by
  sorry

end false_propositions_l66_66932


namespace probability_of_stopping_after_2nd_shot_l66_66248

-- Definitions based on the conditions
def shootingProbability : ℚ := 2 / 3

noncomputable def scoring (n : ℕ) : ℕ := 12 - n

def stopShootingProbabilityAfterNthShot (n : ℕ) (probOfShooting : ℚ) : ℚ :=
  if n = 2 then (1 / 3) * (2 / 3) * sorry -- Note: Here, filling in the remaining calculation steps according to problem logic.
  else sorry -- placeholder for other cases

theorem probability_of_stopping_after_2nd_shot :
  stopShootingProbabilityAfterNthShot 2 shootingProbability = 8 / 729 :=
by
  sorry

end probability_of_stopping_after_2nd_shot_l66_66248


namespace bus_full_people_could_not_take_l66_66570

-- Definitions of the given conditions
def bus_capacity : ℕ := 80
def first_pickup_people : ℕ := (3 / 5) * bus_capacity
def people_exit_at_second_pickup : ℕ := 25
def people_waiting_at_second_pickup : ℕ := 90

-- The Lean statement to prove the number of people who could not take the bus
theorem bus_full_people_could_not_take (h1 : bus_capacity = 80)
                                       (h2 : first_pickup_people = 48)
                                       (h3 : people_exit_at_second_pickup = 25)
                                       (h4 : people_waiting_at_second_pickup = 90) :
  90 - (80 - (48 - 25)) = 33 :=
by
  sorry

end bus_full_people_could_not_take_l66_66570


namespace quadratic_root_neg3_l66_66582

theorem quadratic_root_neg3 : ∃ x : ℝ, x^2 - 9 = 0 ∧ (x = -3) :=
by
  sorry

end quadratic_root_neg3_l66_66582


namespace johnson_and_carter_tie_in_september_l66_66683

def monthly_home_runs_johnson : List ℕ := [3, 14, 18, 13, 10, 16, 14, 5]
def monthly_home_runs_carter : List ℕ := [5, 9, 22, 11, 15, 17, 9, 9]

def cumulative_home_runs (runs : List ℕ) (up_to : ℕ) : ℕ :=
  (runs.take up_to).sum

theorem johnson_and_carter_tie_in_september :
  cumulative_home_runs monthly_home_runs_johnson 7 = cumulative_home_runs monthly_home_runs_carter 7 :=
by
  sorry

end johnson_and_carter_tie_in_september_l66_66683


namespace total_seven_flights_time_l66_66974

def time_for_nth_flight (n : ℕ) : ℕ :=
  25 + (n - 1) * 8

def total_time_for_flights (n : ℕ) : ℕ :=
  (Finset.range n).sum (λ k => time_for_nth_flight (k + 1))

theorem total_seven_flights_time :
  total_time_for_flights 7 = 343 :=
  by
    sorry

end total_seven_flights_time_l66_66974


namespace sector_area_l66_66452

noncomputable def radius_of_sector (l α : ℝ) : ℝ := l / α

noncomputable def area_of_sector (r l : ℝ) : ℝ := (1 / 2) * r * l

theorem sector_area {α l S : ℝ} (hα : α = 2) (hl : l = 3 * Real.pi) (hS : S = 9 * Real.pi ^ 2 / 4) :
  area_of_sector (radius_of_sector l α) l = S := 
by 
  rw [hα, hl, hS]
  rw [radius_of_sector, area_of_sector]
  sorry

end sector_area_l66_66452


namespace apples_b_lighter_than_a_l66_66504

-- Definitions based on conditions
def total_weight : ℕ := 72
def weight_basket_a : ℕ := 42
def weight_basket_b : ℕ := total_weight - weight_basket_a

-- Theorem to prove the question equals the answer given the conditions
theorem apples_b_lighter_than_a : (weight_basket_a - weight_basket_b) = 12 := by
  -- Placeholder for proof
  sorry

end apples_b_lighter_than_a_l66_66504


namespace mary_take_home_pay_l66_66342

def hourly_wage : ℝ := 8
def regular_hours : ℝ := 20
def first_overtime_hours : ℝ := 10
def second_overtime_hours : ℝ := 10
def third_overtime_hours : ℝ := 10
def remaining_overtime_hours : ℝ := 20
def social_security_tax_rate : ℝ := 0.08
def medicare_tax_rate : ℝ := 0.02
def insurance_premium : ℝ := 50

def regular_earnings := regular_hours * hourly_wage
def first_overtime_earnings := first_overtime_hours * (hourly_wage * 1.25)
def second_overtime_earnings := second_overtime_hours * (hourly_wage * 1.5)
def third_overtime_earnings := third_overtime_hours * (hourly_wage * 1.75)
def remaining_overtime_earnings := remaining_overtime_hours * (hourly_wage * 2)

def total_earnings := 
    regular_earnings + 
    first_overtime_earnings + 
    second_overtime_earnings + 
    third_overtime_earnings + 
    remaining_overtime_earnings

def social_security_tax := total_earnings * social_security_tax_rate
def medicare_tax := total_earnings * medicare_tax_rate
def total_taxes := social_security_tax + medicare_tax

def earnings_after_taxes := total_earnings - total_taxes
def earnings_take_home := earnings_after_taxes - insurance_premium

theorem mary_take_home_pay : earnings_take_home = 706 := by
  sorry

end mary_take_home_pay_l66_66342


namespace spoiled_apples_l66_66199

theorem spoiled_apples (S G : ℕ) (h1 : S + G = 8) (h2 : (G * (G - 1)) / 2 = 21) : S = 1 :=
by
  sorry

end spoiled_apples_l66_66199


namespace unique_zero_of_function_l66_66218

theorem unique_zero_of_function (a : ℝ) :
  (∃! x : ℝ, e^(abs x) + 2 * a - 1 = 0) ↔ a = 0 := 
by 
  sorry

end unique_zero_of_function_l66_66218


namespace value_of_y_at_x_eq_1_l66_66524

noncomputable def quadractic_function (x : ℝ) (m : ℝ) : ℝ := 4 * x^2 - m * x + 5

theorem value_of_y_at_x_eq_1 (m : ℝ) (h1 : ∀ x : ℝ, x ≤ -2 → quadractic_function x m < quadractic_function (x + 1) m)
    (h2 : ∀ x : ℝ, x ≥ -2 → quadractic_function x m < quadractic_function (x + 1) m) :
    quadractic_function 1 16 = 25 :=
sorry

end value_of_y_at_x_eq_1_l66_66524


namespace number_of_terms_in_arithmetic_sequence_l66_66598

theorem number_of_terms_in_arithmetic_sequence 
  (a d n l : ℤ) (h1 : a = 7) (h2 : d = 2) (h3 : l = 145) 
  (h4 : l = a + (n - 1) * d) : n = 70 := 
by sorry

end number_of_terms_in_arithmetic_sequence_l66_66598


namespace ninth_term_geometric_sequence_l66_66798

noncomputable def geometric_seq (a r : ℝ) (n : ℕ) : ℝ := a * r ^ (n - 1)

theorem ninth_term_geometric_sequence (a r : ℝ) (h_positive : ∀ n, 0 < geometric_seq a r n)
  (h_fifth_term : geometric_seq a r 5 = 32)
  (h_eleventh_term : geometric_seq a r 11 = 2) :
  geometric_seq a r 9 = 2 :=
by
{
  sorry
}

end ninth_term_geometric_sequence_l66_66798


namespace number_of_guests_l66_66281

-- Defining the given conditions
def appetizers_per_guest : ℕ := 6
def deviled_eggs_dozen : ℕ := 3
def pigs_in_blanket_dozen : ℕ := 2
def kebabs_dozen : ℕ := 2
def additional_appetizers_dozen : ℕ := 8

-- The main theorem to prove the number of guests Patsy is expecting
theorem number_of_guests : 
  (deviled_eggs_dozen + pigs_in_blanket_dozen + kebabs_dozen + additional_appetizers_dozen) * 12 / appetizers_per_guest = 30 :=
by
  sorry

end number_of_guests_l66_66281


namespace probability_both_selected_l66_66780

theorem probability_both_selected (p_ram : ℚ) (p_ravi : ℚ) (h_ram : p_ram = 5/7) (h_ravi : p_ravi = 1/5) : 
  (p_ram * p_ravi = 1/7) := 
by
  sorry

end probability_both_selected_l66_66780


namespace therese_older_than_aivo_l66_66866

-- Definitions based on given conditions
variables {Aivo Jolyn Leon Therese : ℝ}
variables (h1 : Jolyn = Therese + 2)
variables (h2 : Leon = Aivo + 2)
variables (h3 : Jolyn = Leon + 5)

-- Statement to prove
theorem therese_older_than_aivo :
  Therese = Aivo + 5 :=
by
  sorry

end therese_older_than_aivo_l66_66866


namespace complex_sum_to_zero_l66_66361

noncomputable def z : ℂ := sorry

theorem complex_sum_to_zero 
  (h₁ : z ^ 3 = 1) 
  (h₂ : z ≠ 1) : 
  z ^ 103 + z ^ 104 + z ^ 105 + z ^ 106 + z ^ 107 + z ^ 108 = 0 :=
sorry

end complex_sum_to_zero_l66_66361


namespace factorize_m_sq_minus_one_l66_66743

theorem factorize_m_sq_minus_one (m : ℝ) : m^2 - 1 = (m + 1) * (m - 1) := 
by
  sorry

end factorize_m_sq_minus_one_l66_66743


namespace geomSeriesSum_eq_683_l66_66042

/-- Define the first term, common ratio, and number of terms -/
def firstTerm : ℤ := -1
def commonRatio : ℤ := -2
def numTerms : ℕ := 11

/-- Function to calculate the sum of the geometric series -/
def geomSeriesSum (a r : ℤ) (n : ℕ) : ℤ :=
  a * ((r^n - 1) / (r - 1))

/-- The main theorem stating that the sum of the series equals 683 -/
theorem geomSeriesSum_eq_683 :
  geomSeriesSum firstTerm commonRatio numTerms = 683 :=
by sorry

end geomSeriesSum_eq_683_l66_66042


namespace total_votes_l66_66041

theorem total_votes (Ben_votes Matt_votes total_votes : ℕ)
  (h_ratio : 2 * Matt_votes = 3 * Ben_votes)
  (h_Ben_votes : Ben_votes = 24) :
  total_votes = Ben_votes + Matt_votes :=
sorry

end total_votes_l66_66041


namespace value_of_x_l66_66439

theorem value_of_x (x : ℝ) :
  (4 / x) * 12 = 8 ↔ x = 6 :=
by
  sorry

end value_of_x_l66_66439


namespace classroom_position_l66_66029

theorem classroom_position (a b c d : ℕ) (h : (1, 2) = (a, b)) : (3, 2) = (c, d) :=
by
  sorry

end classroom_position_l66_66029


namespace range_of_f_l66_66479

noncomputable def f (x : ℝ) : ℝ := 1 / x - 4 / Real.sqrt x + 3

theorem range_of_f : ∀ y, (∃ x, (1/16 : ℝ) ≤ x ∧ x ≤ 1 ∧ f x = y) ↔ -1 ≤ y ∧ y ≤ 3 := by
  sorry

end range_of_f_l66_66479


namespace ratio_of_amount_divided_to_total_savings_is_half_l66_66508

theorem ratio_of_amount_divided_to_total_savings_is_half :
  let husband_weekly_contribution := 335
  let wife_weekly_contribution := 225
  let weeks_in_six_months := 6 * 4
  let total_weekly_contribution := husband_weekly_contribution + wife_weekly_contribution
  let total_savings := total_weekly_contribution * weeks_in_six_months
  let amount_per_child := 1680
  let number_of_children := 4
  let total_amount_divided := amount_per_child * number_of_children
  (total_amount_divided : ℝ) / total_savings = 0.5 := 
by
  sorry

end ratio_of_amount_divided_to_total_savings_is_half_l66_66508


namespace purely_imaginary_a_eq_2_l66_66068

theorem purely_imaginary_a_eq_2 (a : ℝ) (h : (2 - a) / 2 = 0) : a = 2 :=
sorry

end purely_imaginary_a_eq_2_l66_66068


namespace side_length_of_square_l66_66575

variable (n : ℝ)

theorem side_length_of_square (h : n^2 = 9/16) : n = 3/4 :=
sorry

end side_length_of_square_l66_66575


namespace find_sachins_age_l66_66374

variable (S R : ℕ)

theorem find_sachins_age (h1 : R = S + 8) (h2 : S * 9 = R * 7) : S = 28 := by
  sorry

end find_sachins_age_l66_66374


namespace intersection_eq_l66_66671

def A : Set ℤ := {x | x ∈ Set.Icc (-2 : ℤ) 2}
def B : Set ℝ := {y | y ≤ 1}

theorem intersection_eq : A ∩ {y | y ∈ Set.Icc (-2 : ℤ) 1} = {-2, -1, 0, 1} := by
  sorry

end intersection_eq_l66_66671


namespace find_fourth_number_l66_66670

variable (a : ℕ → ℕ)

theorem find_fourth_number (h₁ : a 7 = 42) (h₂ : a 9 = 110)
    (h₃ : ∀ n, n ≥ 3 → a n = a (n-1) + a (n-2)) : a 4 = 10 :=
by
  sorry

end find_fourth_number_l66_66670


namespace carpet_covering_cost_l66_66653

noncomputable def carpet_cost (floor_length floor_width carpet_length carpet_width carpet_cost_per_square : ℕ) : ℕ :=
  let floor_area := floor_length * floor_width
  let carpet_area := carpet_length * carpet_width
  let num_of_squares := floor_area / carpet_area
  num_of_squares * carpet_cost_per_square

theorem carpet_covering_cost :
  carpet_cost 6 10 2 2 15 = 225 :=
by
  sorry

end carpet_covering_cost_l66_66653


namespace neg_number_among_set_l66_66808

theorem neg_number_among_set :
  ∃ n ∈ ({5, 1, -2, 0} : Set ℤ), n < 0 ∧ n = -2 :=
by
  sorry

end neg_number_among_set_l66_66808


namespace arithmetic_sequence_properties_l66_66573

noncomputable def arithmetic_sequence (n : ℕ) : ℕ :=
  4 * n - 3

noncomputable def sum_of_first_n_terms (n : ℕ) : ℕ :=
  2 * n^2 - n

noncomputable def sum_of_reciprocal_sequence (n : ℕ) : ℝ :=
  n / (4 * n + 1)

theorem arithmetic_sequence_properties :
  (arithmetic_sequence 3 = 9) →
  (arithmetic_sequence 8 = 29) →
  (∀ n, arithmetic_sequence n = 4 * n - 3) ∧
  (∀ n, sum_of_first_n_terms n = 2 * n^2 - n) ∧
  (∀ n, sum_of_reciprocal_sequence n = n / (4 * n + 1)) :=
by
  sorry

end arithmetic_sequence_properties_l66_66573


namespace not_possible_to_partition_into_groups_of_5_with_remainder_3_l66_66778

theorem not_possible_to_partition_into_groups_of_5_with_remainder_3 (m : ℤ) :
  ¬ (m^2 % 5 = 3) :=
by sorry

end not_possible_to_partition_into_groups_of_5_with_remainder_3_l66_66778


namespace r_expansion_l66_66307

theorem r_expansion (r : ℝ) (h : (r + 1/r)^2 = 5) : r^4 + 1/r^4 = 7 :=
by
  sorry

end r_expansion_l66_66307


namespace length_of_rectangle_l66_66496

-- Given conditions as per the problem statement
variables {s l : ℝ} -- side length of the square, length of the rectangle
def width_rectangle : ℝ := 10 -- width of the rectangle

-- Conditions
axiom sq_perimeter : 4 * s = 200
axiom area_relation : s^2 = 5 * (l * width_rectangle)

-- Goal to prove
theorem length_of_rectangle : l = 50 :=
by
  sorry

end length_of_rectangle_l66_66496


namespace wood_burned_in_afternoon_l66_66818

theorem wood_burned_in_afternoon 
  (burned_morning : ℕ) 
  (start_bundles : ℕ) 
  (end_bundles : ℕ) 
  (burned_afternoon : ℕ) 
  (h1 : burned_morning = 4) 
  (h2 : start_bundles = 10) 
  (h3 : end_bundles = 3) 
  (h4 : burned_morning + burned_afternoon = start_bundles - end_bundles) :
  burned_afternoon = 3 := 
sorry

end wood_burned_in_afternoon_l66_66818


namespace functional_equation_solution_l66_66372

noncomputable def function_nat_nat (f : ℕ → ℕ) : Prop :=
  ∀ x y : ℕ, f (x + y) = f x + f y

theorem functional_equation_solution :
  ∀ f : ℕ → ℕ, function_nat_nat f → ∃ a : ℕ, ∀ x : ℕ, f x = a * x :=
by
  sorry

end functional_equation_solution_l66_66372


namespace value_of_y_l66_66799

theorem value_of_y (x y : ℝ) (h1 : x - y = 16) (h2 : x + y = 4) : y = -6 :=
by
  sorry

end value_of_y_l66_66799


namespace quotient_of_integers_l66_66457

variable {x y : ℤ}

theorem quotient_of_integers (h : 1996 * x + y / 96 = x + y) : 
  (x / y = 1 / 2016) ∨ (y / x = 2016) := by
  sorry

end quotient_of_integers_l66_66457


namespace difference_of_two_numbers_l66_66156

theorem difference_of_two_numbers (a b : ℕ) 
(h1 : a + b = 17402) 
(h2 : ∃ k : ℕ, b = 10 * k) 
(h3 : ∃ k : ℕ, a + 9 * k = b) : 
10 * a - a = 14238 :=
by sorry

end difference_of_two_numbers_l66_66156


namespace percentage_volume_occupied_is_100_l66_66737

-- Define the dimensions of the box and cube
def box_length : ℕ := 8
def box_width : ℕ := 4
def box_height : ℕ := 12
def cube_side : ℕ := 2

-- Define the volumes
def box_volume : ℕ := box_length * box_width * box_height
def cube_volume : ℕ := cube_side * cube_side * cube_side

-- Define the number of cubes that fit in each dimension
def cubes_along_length : ℕ := box_length / cube_side
def cubes_along_width : ℕ := box_width / cube_side
def cubes_along_height : ℕ := box_height / cube_side

-- Define the total number of cubes and the volume they occupy
def total_cubes : ℕ := cubes_along_length * cubes_along_width * cubes_along_height
def volume_occupied_by_cubes : ℕ := total_cubes * cube_volume

-- Define the percentage of the box volume occupied by the cubes
def percentage_volume_occupied : ℕ := (volume_occupied_by_cubes * 100) / box_volume

-- Statement to prove
theorem percentage_volume_occupied_is_100 : percentage_volume_occupied = 100 := by
  sorry

end percentage_volume_occupied_is_100_l66_66737


namespace option_d_always_correct_l66_66282

variable {a b : ℝ}

theorem option_d_always_correct (h1 : a < b) (h2 : b < 0) (h3 : a < 0) :
  (a + 1 / b)^2 > (b + 1 / a)^2 :=
by
  -- Lean proof code would go here.
  sorry

end option_d_always_correct_l66_66282


namespace geo_seq_product_l66_66113

theorem geo_seq_product (a : ℕ → ℝ) (r : ℝ) (h_pos : ∀ n, 0 < a n) 
  (h_geom : ∀ n, a (n + 1) = a n * r) (h_a1a9 : a 1 * a 9 = 16) :
  a 2 * a 5 * a 8 = 64 :=
sorry

end geo_seq_product_l66_66113


namespace total_messages_l66_66162

theorem total_messages (x : ℕ) (h : x * (x - 1) = 420) : x * (x - 1) = 420 :=
by
  sorry

end total_messages_l66_66162


namespace percentage_of_sum_is_14_l66_66420

-- Define variables x, y as real numbers
variables (x y P : ℝ)

-- Define condition 1: y is 17.647058823529413% of x
def y_is_percentage_of_x : Prop := y = 0.17647058823529413 * x

-- Define condition 2: 20% of (x - y) is equal to P% of (x + y)
def percentage_equation : Prop := 0.20 * (x - y) = (P / 100) * (x + y)

-- Define the statement to be proved: P is 14
theorem percentage_of_sum_is_14 (h1 : y_is_percentage_of_x x y) (h2 : percentage_equation x y P) : 
  P = 14 :=
by
  sorry

end percentage_of_sum_is_14_l66_66420


namespace suresh_work_hours_l66_66936

variable (x : ℕ) -- Number of hours Suresh worked

theorem suresh_work_hours :
  (1/15 : ℝ) * x + (4 * (1/10 : ℝ)) = 1 -> x = 9 :=
by
  sorry

end suresh_work_hours_l66_66936


namespace boris_stopped_saving_in_may_2020_l66_66814

theorem boris_stopped_saving_in_may_2020 :
  ∀ (B V : ℕ) (start_date_B start_date_V stop_date : ℕ), 
    (∀ t, start_date_B + t ≤ stop_date → B = 200 * t) →
    (∀ t, start_date_V + t ≤ stop_date → V = 300 * t) → 
    V = 6 * B →
    stop_date = 17 → 
    B / 200 = 4 → 
    stop_date - B/200 = 2020 * 12 + 5 :=
by
  sorry

end boris_stopped_saving_in_may_2020_l66_66814


namespace find_number_l66_66559

theorem find_number (x : ℝ) (h : (((x + 1.4) / 3 - 0.7) * 9 = 5.4)) : x = 2.5 :=
by 
  sorry

end find_number_l66_66559


namespace probability_product_divisible_by_4_gt_half_l66_66522

theorem probability_product_divisible_by_4_gt_half :
  let n := 2023
  let even_count := n / 2
  let four_div_count := n / 4
  let select_five := 5
  (true) ∧ (even_count = 1012) ∧ (four_div_count = 505)
  → 0.5 < (1 - ((2023 - even_count) / 2023) * ((2022 - (even_count - 1)) / 2022) * ((2021 - (even_count - 2)) / 2021) * ((2020 - (even_count - 3)) / 2020) * ((2019 - (even_count - 4)) / 2019)) :=
by
  sorry

end probability_product_divisible_by_4_gt_half_l66_66522


namespace simplify_expression_l66_66513

-- Define the statement we want to prove
theorem simplify_expression (s : ℕ) : (105 * s - 63 * s) = 42 * s :=
  by
    -- Placeholder for the proof
    sorry

end simplify_expression_l66_66513


namespace floor_of_sum_eq_l66_66167

theorem floor_of_sum_eq (x y z w : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hw : w > 0)
  (hxy : x^2 + y^2 = 2500) (hzw : z^2 + w^2 = 2500) (hxz : x * z = 1200) (hyw : y * w = 1200) :
  ⌊x + y + z + w⌋ = 140 := by
  sorry

end floor_of_sum_eq_l66_66167


namespace train_travel_time_l66_66110

theorem train_travel_time 
  (speed : ℝ := 120) -- speed in kmph
  (distance : ℝ := 80) -- distance in km
  (minutes_in_hour : ℝ := 60) -- conversion factor
  : (distance / speed) * minutes_in_hour = 40 :=
by
  -- Sorry is used as a placeholder for the proof
  sorry

end train_travel_time_l66_66110


namespace champagne_bottles_needed_l66_66788

-- Define the initial conditions of the problem
def num_guests : ℕ := 120
def glasses_per_guest : ℕ := 2
def servings_per_bottle : ℕ := 6

-- The statement we need to prove
theorem champagne_bottles_needed : 
  (num_guests * glasses_per_guest) / servings_per_bottle = 40 := 
by
  sorry

end champagne_bottles_needed_l66_66788


namespace LCM_of_apple_and_cherry_pies_l66_66269

theorem LCM_of_apple_and_cherry_pies :
  let apple_pies := (13 : ℚ) / 2
  let cherry_pies := (21 : ℚ) / 4
  let lcm_numerators := Nat.lcm 26 21
  let common_denominator := 4
  (lcm_numerators : ℚ) / (common_denominator : ℚ) = 273 / 2 :=
by
  let apple_pies := (13 : ℚ) / 2
  let cherry_pies := (21 : ℚ) / 4
  let lcm_numerators := Nat.lcm 26 21
  let common_denominator := 4
  have h : (lcm_numerators : ℚ) / (common_denominator : ℚ) = 273 / 2 := sorry
  exact h

end LCM_of_apple_and_cherry_pies_l66_66269


namespace wooden_block_even_blue_faces_l66_66004

theorem wooden_block_even_blue_faces :
  let length := 6
  let width := 6
  let height := 2
  let total_cubes := length * width * height
  let corners := 8
  let edges_not_corners := 24
  let faces_not_edges := 24
  let interior := 16
  let even_blue_faces := edges_not_corners + interior
  total_cubes = 72 →
  even_blue_faces = 40 :=
by
  sorry

end wooden_block_even_blue_faces_l66_66004


namespace true_supporters_of_rostov_l66_66533

theorem true_supporters_of_rostov
  (knights_liars_fraction : ℕ → ℕ)
  (rostov_support_yes : ℕ)
  (zenit_support_yes : ℕ)
  (lokomotiv_support_yes : ℕ)
  (cska_support_yes : ℕ)
  (h1 : knights_liars_fraction 100 = 10)
  (h2 : rostov_support_yes = 40)
  (h3 : zenit_support_yes = 30)
  (h4 : lokomotiv_support_yes = 50)
  (h5 : cska_support_yes = 0):
  rostov_support_yes - knights_liars_fraction 100 = 30 := 
sorry

end true_supporters_of_rostov_l66_66533


namespace min_trips_to_fill_hole_l66_66306

def hole_filling_trips (initial_gallons : ℕ) (required_gallons : ℕ) (capacity_2gallon : ℕ)
  (capacity_5gallon : ℕ) (capacity_8gallon : ℕ) (time_limit : ℕ) (time_per_trip : ℕ) : ℕ :=
  if initial_gallons < required_gallons then
    let remaining_gallons := required_gallons - initial_gallons
    let num_8gallon := remaining_gallons / capacity_8gallon
    let remaining_after_8gallon := remaining_gallons % capacity_8gallon
    let num_2gallon := if remaining_after_8gallon = 3 then 1 else 0
    let num_5gallon := if remaining_after_8gallon = 3 then 1 else remaining_after_8gallon / capacity_5gallon
    let total_trips := num_8gallon + num_2gallon + num_5gallon
    if total_trips <= time_limit / time_per_trip then
      total_trips
    else
      sorry -- If calculations overflow time limit
  else
    0

theorem min_trips_to_fill_hole : 
  hole_filling_trips 676 823 2 5 8 45 1 = 20 :=
by rfl

end min_trips_to_fill_hole_l66_66306


namespace max_reciprocal_sum_eq_2_l66_66243

theorem max_reciprocal_sum_eq_2 (r1 r2 t q : ℝ) (h1 : r1 + r2 = t) (h2 : r1 * r2 = q)
  (h3 : ∀ n : ℕ, n > 0 → r1 + r2 = r1^n + r2^n) :
  1 / r1^2010 + 1 / r2^2010 = 2 :=
by
  sorry

end max_reciprocal_sum_eq_2_l66_66243


namespace avg_prime_factors_of_multiples_of_10_l66_66279

theorem avg_prime_factors_of_multiples_of_10 : 
  (2 + 5) / 2 = 3.5 :=
by
  -- The prime factors of 10 are 2 and 5.
  -- Therefore, the average of these prime factors is (2 + 5) / 2.
  sorry

end avg_prime_factors_of_multiples_of_10_l66_66279


namespace cos_identity_l66_66116

theorem cos_identity (α : ℝ) (h : Real.cos (π / 3 - α) = 3 / 5) : 
  Real.cos (2 * π / 3 + α) = -3 / 5 :=
by
  sorry

end cos_identity_l66_66116


namespace general_term_formula_sum_of_first_n_terms_l66_66195

noncomputable def a (n : ℕ) : ℕ :=
(n + 2^n)^2

theorem general_term_formula :
  ∀ n : ℕ, a n = n^2 + n * 2^(n+1) + 4^n :=
sorry

noncomputable def S (n : ℕ) : ℕ :=
(n-1) * 2^(n+2) + 4 + (4^(n+1) - 4) / 3

theorem sum_of_first_n_terms :
  ∀ n : ℕ, S n = (n-1) * 2^(n+2) + 4 + (4^(n+1) - 4) / 3 :=
sorry

end general_term_formula_sum_of_first_n_terms_l66_66195


namespace remainder_when_1_stmt_l66_66500

-- Define the polynomial g(s)
def g (s : ℚ) : ℚ := s^15 + 1

-- Define the remainder theorem statement in the context of this problem
theorem remainder_when_1_stmt (s : ℚ) : g 1 = 2 :=
  sorry

end remainder_when_1_stmt_l66_66500


namespace print_time_nearest_whole_l66_66870

theorem print_time_nearest_whole 
  (pages_per_minute : ℕ) (total_pages : ℕ) (expected_time : ℕ)
  (h1 : pages_per_minute = 25) (h2 : total_pages = 575) : 
  expected_time = 23 :=
by
  sorry

end print_time_nearest_whole_l66_66870


namespace power_sum_inequality_l66_66895

theorem power_sum_inequality (a b c d : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d) (h5 : a * b * c * d = 1) :
  a^5 + b^5 + c^5 + d^5 ≥ a + b + c + d :=
by sorry

end power_sum_inequality_l66_66895


namespace inradius_plus_circumradius_le_height_l66_66102

theorem inradius_plus_circumradius_le_height {α β γ : ℝ} 
    (h : ℝ) (r R : ℝ)
    (h_triangle : α ≥ β ∧ β ≥ γ ∧ γ ≥ 0 ∧ α + β + γ = π )
    (h_non_obtuse : π / 2 ≥ α ∧ π / 2 ≥ β ∧ π / 2 ≥ γ)
    (h_greatest_height : true) -- Assuming this condition holds as given
    :
    r + R ≤ h :=
sorry

end inradius_plus_circumradius_le_height_l66_66102


namespace min_value_of_f_in_interval_l66_66594

def f (x k : ℝ) : ℝ := x^2 - k * x - 1

theorem min_value_of_f_in_interval (k : ℝ) :
  (f 1 k = -k ∧ k ≤ 2) ∨ 
  (∃ k', k' = 2 ∧ f (k'/2) k = - (k'^2) / 4 - 1 ∧ 2 < k ∧ k < 8) ∨ 
  (f 4 k = 15 - 4 * k ∧ k ≥ 8) :=
by sorry

end min_value_of_f_in_interval_l66_66594


namespace triangle_height_l66_66874

theorem triangle_height (A : ℝ) (b : ℝ) (h : ℝ) 
  (hA : A = 615) 
  (hb : b = 123)
  (h_area : A = 0.5 * b * h) : 
  h = 10 :=
by 
  -- Placeholder for the proof
  sorry

end triangle_height_l66_66874


namespace largest_angle_in_right_isosceles_triangle_l66_66083

theorem largest_angle_in_right_isosceles_triangle (X Y Z : Type) 
  (angle_X : ℝ) (angle_Y : ℝ) (angle_Z : ℝ) 
  (h1 : angle_X = 45) 
  (h2 : angle_Y = 90)
  (h3 : angle_Y + angle_X + angle_Z = 180) 
  (h4 : angle_X = angle_Z) : angle_Y = 90 := by 
  sorry

end largest_angle_in_right_isosceles_triangle_l66_66083


namespace solve_fraction_zero_l66_66538

theorem solve_fraction_zero (x : ℝ) (h : (x + 5) / (x - 2) = 0) : x = -5 :=
by
  sorry

end solve_fraction_zero_l66_66538


namespace product_of_a_and_b_is_zero_l66_66421

theorem product_of_a_and_b_is_zero
  (a b : ℕ)
  (h1 : 10 ≤ a ∧ a < 100)
  (h2 : b < 10)
  (h3 : a * (b + 10) = 190) :
  a * b = 0 :=
sorry

end product_of_a_and_b_is_zero_l66_66421


namespace area_of_inscribed_rectangle_l66_66770

theorem area_of_inscribed_rectangle 
    (DA : ℝ) 
    (GD HD : ℝ) 
    (rectangle_inscribed : ∀ (A B C D G H : Type), true) 
    (radius : ℝ) 
    (GH : ℝ):
    DA = 20 ∧ GD = 5 ∧ HD = 5 ∧ GH = GD + DA + HD ∧ radius = GH / 2 → 
    200 * Real.sqrt 2 = DA * (Real.sqrt (radius^2 - (GD^2))) :=
by
  sorry

end area_of_inscribed_rectangle_l66_66770


namespace sufficient_but_not_necessary_l66_66241

theorem sufficient_but_not_necessary (a : ℝ) : 
  (a > 2 → a^2 > 2 * a) ∧ ¬(a^2 > 2 * a → a > 2) :=
by
  sorry

end sufficient_but_not_necessary_l66_66241


namespace max_volume_l66_66721

variable (x y z : ℝ) (V : ℝ)
variable (k : ℝ)

-- Define the constraint
def constraint := x + 2 * y + 3 * z = 180

-- Define the volume
def volume := x * y * z

-- The goal is to show that under the constraint, the maximum possible volume is 36000 cubic cm.
theorem max_volume :
  (∀ (x y z : ℝ) (h : constraint x y z), volume x y z ≤ 36000) :=
  sorry

end max_volume_l66_66721


namespace valid_range_and_difference_l66_66000

/- Assume side lengths as given expressions -/
def BC (x : ℝ) : ℝ := x + 11
def AC (x : ℝ) : ℝ := x + 6
def AB (x : ℝ) : ℝ := 3 * x + 2

/- Define the inequalities representing the triangle inequalities and largest angle condition -/
def triangle_inequality1 (x : ℝ) : Prop := AB x + AC x > BC x
def triangle_inequality2 (x : ℝ) : Prop := AB x + BC x > AC x
def triangle_inequality3 (x : ℝ) : Prop := AC x + BC x > AB x
def largest_angle_condition (x : ℝ) : Prop := BC x > AB x

/- Define the combined condition for x, ensuring all relevant conditions are met -/
def valid_x_range (x : ℝ) : Prop :=
  1 < x ∧ x < 4.5 ∧ triangle_inequality1 x ∧ triangle_inequality2 x ∧ triangle_inequality3 x ∧ largest_angle_condition x

/- Compute n - m for the interval (m, n) where x lies -/
def n_minus_m : ℝ :=
  4.5 - 1

/- Main theorem stating the final result -/
theorem valid_range_and_difference :
  (∃ x : ℝ, valid_x_range x) ∧ (n_minus_m = 7 / 2) :=
by
  sorry

end valid_range_and_difference_l66_66000


namespace waiter_slices_l66_66417

theorem waiter_slices (total_slices : ℕ) (buzz_ratio waiter_ratio : ℕ)
  (h_total_slices : total_slices = 78)
  (h_ratios : buzz_ratio = 5 ∧ waiter_ratio = 8) :
  20 < (waiter_ratio * (total_slices / (buzz_ratio + waiter_ratio))) →
  28 = waiter_ratio * (total_slices / (buzz_ratio + waiter_ratio)) - 20 :=
by
  sorry

end waiter_slices_l66_66417


namespace math_problem_l66_66376

theorem math_problem (x t : ℝ) (h1 : 6 * x + t = 4 * x - 9) (h2 : t = 7) : x + 4 = -4 := by
  sorry

end math_problem_l66_66376


namespace proof_a_squared_plus_1_l66_66172

theorem proof_a_squared_plus_1 (a : ℤ) (h1 : 3 < a) (h2 : a < 5) : a^2 + 1 = 17 :=
  by
  sorry

end proof_a_squared_plus_1_l66_66172


namespace color_change_probability_l66_66230

-- Definitions based directly on conditions in a)
def light_cycle_duration := 93
def change_intervals_duration := 15
def expected_probability := 5 / 31

-- The Lean 4 statement for the proof problem
theorem color_change_probability :
  (change_intervals_duration / light_cycle_duration) = expected_probability :=
by
  sorry

end color_change_probability_l66_66230


namespace sarah_photos_l66_66658

theorem sarah_photos (photos_Cristina photos_John photos_Clarissa total_slots : ℕ)
  (hCristina : photos_Cristina = 7)
  (hJohn : photos_John = 10)
  (hClarissa : photos_Clarissa = 14)
  (hTotal : total_slots = 40) :
  ∃ photos_Sarah, photos_Sarah = total_slots - (photos_Cristina + photos_John + photos_Clarissa) ∧ photos_Sarah = 9 :=
by
  sorry

end sarah_photos_l66_66658


namespace coffee_price_l66_66546

theorem coffee_price (C : ℝ) :
  (7 * C) + (8 * 4) = 67 → C = 5 :=
by
  intro h
  sorry

end coffee_price_l66_66546


namespace smaller_number_l66_66539

theorem smaller_number (x y : ℤ) (h1 : x + y = 24) (h2 : x - y = 16) : y = 4 := by
  sorry

end smaller_number_l66_66539


namespace prove_a_eq_1_l66_66493

variables {a b c d k m : ℕ}
variables (h_odd_a : a%2 = 1) 
          (h_odd_b : b%2 = 1) 
          (h_odd_c : c%2 = 1) 
          (h_odd_d : d%2 = 1)
          (h_a_pos : 0 < a) 
          (h_ineq1 : a < b) 
          (h_ineq2 : b < c) 
          (h_ineq3 : c < d)
          (h_eqn1 : a * d = b * c)
          (h_eqn2 : a + d = 2^k) 
          (h_eqn3 : b + c = 2^m)

theorem prove_a_eq_1 
  (h_odd_a : a%2 = 1) 
  (h_odd_b : b%2 = 1) 
  (h_odd_c : c%2 = 1) 
  (h_odd_d : d%2 = 1)
  (h_a_pos : 0 < a) 
  (h_ineq1 : a < b) 
  (h_ineq2 : b < c) 
  (h_ineq3 : c < d)
  (h_eqn1 : a * d = b * c)
  (h_eqn2 : a + d = 2^k) 
  (h_eqn3 : b + c = 2^m) :
  a = 1 := by
  sorry

end prove_a_eq_1_l66_66493


namespace password_count_correct_l66_66661

-- Defining variables
def n_letters := 26
def n_digits := 10

-- The number of permutations for selecting 2 different letters
def perm_letters := n_letters * (n_letters - 1)
-- The number of permutations for selecting 2 different numbers
def perm_digits := n_digits * (n_digits - 1)

-- The total number of possible passwords
def total_permutations := perm_letters * perm_digits

-- The theorem we need to prove
theorem password_count_correct :
  total_permutations = (n_letters * (n_letters - 1)) * (n_digits * (n_digits - 1)) :=
by
  -- The proof goes here
  sorry

end password_count_correct_l66_66661


namespace expr_min_value_expr_min_at_15_l66_66418

theorem expr_min_value (a x : ℝ) (h1 : 0 ≤ a) (h2 : a ≤ 15) (h3 : a ≤ x) (h4 : x ≤ 15) :
  (|x - a| + |x - 15| + |x - (a + 15)|) = 30 - x := 
sorry

theorem expr_min_at_15 (a : ℝ) (h : 0 ≤ a ∧ a ≤ 15) : 
  (|15 - a| + |15 - 15| + |15 - (a + 15)|) = 15 := 
sorry

end expr_min_value_expr_min_at_15_l66_66418


namespace find_number_l66_66455

theorem find_number (x : ℝ) (h : ((x / 3) * 24) - 7 = 41) : x = 6 :=
by
  sorry

end find_number_l66_66455


namespace square_area_example_l66_66040

noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

noncomputable def square_area (x1 y1 x2 y2 : ℝ) : ℝ :=
  (distance x1 y1 x2 y2)^2

theorem square_area_example : square_area 1 3 5 6 = 25 :=
by
  sorry

end square_area_example_l66_66040


namespace unique_8_tuple_real_l66_66178

theorem unique_8_tuple_real (x : Fin 8 → ℝ) :
  (1 - x 0)^2 + (x 0 - x 1)^2 + (x 1 - x 2)^2 + (x 2 - x 3)^2 + (x 3 - x 4)^2 + (x 4 - x 5)^2 + (x 5 - x 6)^2 + (x 6 - x 7)^2 + x 7^2 = 1 / 8 →
  ∃! (y : Fin 8 → ℝ), (1 - y 0)^2 + (y 0 - y 1)^2 + (y 1 - y 2)^2 + (y 2 - y 3)^2 + (y 3 - y 4)^2 + (y 4 - y 5)^2 + (y 5 - y 6)^2 + (y 6 - y 7)^2 + y 7^2 = 1 / 8 :=
by
  sorry

end unique_8_tuple_real_l66_66178


namespace market_value_of_13_percent_stock_yielding_8_percent_l66_66338

noncomputable def market_value_of_stock (yield rate dividend_per_share : ℝ) : ℝ :=
  (dividend_per_share / yield) * 100

theorem market_value_of_13_percent_stock_yielding_8_percent
  (yield_rate : ℝ) (dividend_per_share : ℝ) (market_value : ℝ)
  (h_yield_rate : yield_rate = 0.08)
  (h_dividend_per_share : dividend_per_share = 13) :
  market_value = 162.50 :=
by
  sorry

end market_value_of_13_percent_stock_yielding_8_percent_l66_66338


namespace count_four_digit_numbers_divisible_by_5_ending_in_45_l66_66505

theorem count_four_digit_numbers_divisible_by_5_ending_in_45 : 
  ∃ n : ℕ, (∀ x : ℕ, x ≥ 1000 ∧ x < 10000 ∧ x % 100 = 45 → (x % 5 = 0 ∧ x % 100 = 45)) 
  ∧ (n = 90) :=
by
  sorry

end count_four_digit_numbers_divisible_by_5_ending_in_45_l66_66505


namespace price_of_each_apple_l66_66148

-- Define the constants and conditions
def price_banana : ℝ := 0.60
def total_fruits : ℕ := 9
def total_cost : ℝ := 5.60

-- Declare the variables for number of apples and price of apples
variables (A : ℝ) (x y : ℕ)

-- Define the conditions in Lean
axiom h1 : x + y = total_fruits
axiom h2 : A * x + price_banana * y = total_cost

-- Prove that the price of each apple is $0.80
theorem price_of_each_apple : A = 0.80 :=
by sorry

end price_of_each_apple_l66_66148


namespace find_larger_number_l66_66166

theorem find_larger_number (L S : ℕ) (h1 : L - S = 2415) (h2 : L = 21 * S + 15) : L = 2535 := 
by
  sorry

end find_larger_number_l66_66166


namespace mike_max_marks_l66_66695

theorem mike_max_marks (m : ℕ) (h : 30 * m = 237 * 10) : m = 790 := by
  sorry

end mike_max_marks_l66_66695


namespace cos_double_angle_l66_66259

theorem cos_double_angle (x : ℝ) (h : Real.sin (x + Real.pi / 2) = 1 / 3) : Real.cos (2 * x) = -7 / 9 :=
sorry

end cos_double_angle_l66_66259


namespace marcus_brought_30_peanut_butter_cookies_l66_66736

/-- Jenny brought in 40 peanut butter cookies. -/
def jenny_peanut_butter_cookies := 40

/-- Jenny brought in 50 chocolate chip cookies. -/
def jenny_chocolate_chip_cookies := 50

/-- Marcus brought in 20 lemon cookies. -/
def marcus_lemon_cookies := 20

/-- The total number of non-peanut butter cookies is the sum of chocolate chip and lemon cookies. -/
def non_peanut_butter_cookies := jenny_chocolate_chip_cookies + marcus_lemon_cookies

/-- The total number of peanut butter cookies is Jenny's plus Marcus'. -/
def total_peanut_butter_cookies (marcus_peanut_butter_cookies : ℕ) := jenny_peanut_butter_cookies + marcus_peanut_butter_cookies

/-- If Renee has a 50% chance of picking a peanut butter cookie, the number of peanut butter cookies must equal the number of non-peanut butter cookies. -/
theorem marcus_brought_30_peanut_butter_cookies (x : ℕ) : total_peanut_butter_cookies x = non_peanut_butter_cookies → x = 30 :=
by
  sorry

end marcus_brought_30_peanut_butter_cookies_l66_66736


namespace complement_intersection_l66_66957

def setM : Set ℝ := { x | 2 / x < 1 }
def setN : Set ℝ := { y | ∃ x, y = Real.sqrt (x - 1) }

theorem complement_intersection 
  (R : Set ℝ) : ((R \ setM) ∩ setN = { y | 0 ≤ y ∧ y ≤ 2 }) :=
  sorry

end complement_intersection_l66_66957


namespace tangent_chord_equation_l66_66074

theorem tangent_chord_equation (x1 y1 x2 y2 : ℝ) :
  (x1^2 + y1^2 = 1) →
  (x2^2 + y2^2 = 1) →
  (2*x1 + 2*y1 + 1 = 0) →
  (2*x2 + 2*y2 + 1 = 0) →
  ∀ (x y : ℝ), 2*x + 2*y + 1 = 0 :=
by
  intros hx1 hy1 hx2 hy2 x y
  exact sorry

end tangent_chord_equation_l66_66074


namespace solve_arcsin_arccos_l66_66779

open Real

theorem solve_arcsin_arccos (x : ℝ) (h_condition : - (1 / 2 : ℝ) ≤ x ∧ x ≤ 1 / 2) :
  arcsin x + arcsin (2 * x) = arccos x ↔ x = 0 :=
sorry

end solve_arcsin_arccos_l66_66779


namespace marbles_total_l66_66596

theorem marbles_total (yellow blue red total : ℕ)
  (hy : yellow = 5)
  (h_ratio : blue / red = 3 / 4)
  (h_red : red = yellow + 3)
  (h_total : total = yellow + blue + red) : total = 19 :=
by
  sorry

end marbles_total_l66_66596


namespace tenth_term_arithmetic_sequence_l66_66073

theorem tenth_term_arithmetic_sequence :
  ∀ (a : ℕ → ℚ), a 1 = 5/6 ∧ a 16 = 7/8 →
  a 10 = 103/120 :=
by
  sorry

end tenth_term_arithmetic_sequence_l66_66073


namespace min_value_ge_8_min_value_8_at_20_l66_66567

noncomputable def min_value (x : ℝ) (h : x > 4) : ℝ := (x + 12) / Real.sqrt (x - 4)

theorem min_value_ge_8 (x : ℝ) (h : x > 4) : min_value x h ≥ 8 := sorry

theorem min_value_8_at_20 : min_value 20 (by norm_num) = 8 := sorry

end min_value_ge_8_min_value_8_at_20_l66_66567


namespace max_value_of_a_l66_66657

variable {R : Type*} [LinearOrderedField R]

def det (a b c d : R) : R := a * d - b * c

theorem max_value_of_a (a : R) :
  (∀ x : R, det (x - 1) (a - 2) (a + 1) x ≥ 1) → a ≤ (3 / 2 : R) :=
by
  sorry

end max_value_of_a_l66_66657


namespace selling_price_l66_66118

theorem selling_price (cost_price profit_percentage selling_price : ℝ) (h1 : cost_price = 86.95652173913044)
  (h2 : profit_percentage = 0.15) : 
  selling_price = 100 :=
by
  sorry

end selling_price_l66_66118


namespace first_candidate_percentage_l66_66120

noncomputable
def passing_marks_approx : ℝ := 240

noncomputable
def total_marks (P : ℝ) : ℝ := (P + 30) / 0.45

noncomputable
def percentage_marks (T P : ℝ) : ℝ := ((P - 60) / T) * 100

theorem first_candidate_percentage :
  let P := passing_marks_approx
  let T := total_marks P
  percentage_marks T P = 30 :=
by
  sorry

end first_candidate_percentage_l66_66120


namespace solve_inequality_l66_66377

-- Define the domain and inequality conditions
def inequality_condition (x : ℝ) : Prop := (1 / (x - 1)) > 1
def domain_condition (x : ℝ) : Prop := x ≠ 1

-- State the theorem to be proved.
theorem solve_inequality (x : ℝ) : domain_condition x → inequality_condition x → 1 < x ∧ x < 2 :=
by
  intros h_domain h_ineq
  sorry

end solve_inequality_l66_66377


namespace white_roses_needed_l66_66436

theorem white_roses_needed (bouquets table_decorations white_roses_per_table_decoration white_roses_per_bouquet : ℕ)
  (h_bouquets : bouquets = 5)
  (h_table_decorations : table_decorations = 7)
  (h_white_roses_per_table_decoration : white_roses_per_table_decoration = 12)
  (h_white_roses_per_bouquet : white_roses_per_bouquet = 5):
  bouquets * white_roses_per_bouquet + table_decorations * white_roses_per_table_decoration = 109 := by
  sorry

end white_roses_needed_l66_66436


namespace articles_production_l66_66299

theorem articles_production (x y : ℕ) (e : ℝ) :
  (x * x * x * e / x = x^2 * e) → (y * (y + 2) * y * (e / x) = (e * y * (y^2 + 2 * y)) / x) :=
by 
  sorry

end articles_production_l66_66299


namespace total_dots_not_visible_l66_66458

-- Define the total dot sum for each die
def sum_of_dots_per_die : Nat := 1 + 2 + 3 + 4 + 5 + 6

-- Define the total number of dice
def number_of_dice : Nat := 4

-- Calculate the total dot sum for all dice
def total_dots_all_dice : Nat := sum_of_dots_per_die * number_of_dice

-- Sum of visible dots
def sum_of_visible_dots : Nat := 1 + 1 + 2 + 2 + 3 + 3 + 4 + 5 + 6 + 6

-- Prove the total dots not visible
theorem total_dots_not_visible : total_dots_all_dice - sum_of_visible_dots = 51 := by
  sorry

end total_dots_not_visible_l66_66458


namespace geom_sequence_sum_correct_l66_66807

noncomputable def geom_sequence_sum (a₁ a₄ : ℕ) (S₅ : ℕ) :=
  ∃ q : ℕ, a₁ = 1 ∧ a₄ = a₁ * q ^ 3 ∧ S₅ = (a₁ * (1 - q ^ 5)) / (1 - q)

theorem geom_sequence_sum_correct : geom_sequence_sum 1 8 31 :=
by {
  sorry
}

end geom_sequence_sum_correct_l66_66807


namespace stratified_sampling_2nd_year_students_l66_66584

theorem stratified_sampling_2nd_year_students
  (students_1st_year : ℕ) (students_2nd_year : ℕ) (students_3rd_year : ℕ) (total_sample_size : ℕ) :
  students_1st_year = 1000 ∧ students_2nd_year = 800 ∧ students_3rd_year = 700 ∧ total_sample_size = 100 →
  (students_2nd_year * total_sample_size / (students_1st_year + students_2nd_year + students_3rd_year) = 32) :=
by
  intro h
  sorry

end stratified_sampling_2nd_year_students_l66_66584


namespace johns_average_speed_last_hour_l66_66272

theorem johns_average_speed_last_hour
  (total_distance : ℕ)
  (total_time : ℕ)
  (speed_first_hour : ℕ)
  (speed_second_hour : ℕ)
  (distance_last_hour : ℕ)
  (average_speed_last_hour : ℕ)
  (H1 : total_distance = 120)
  (H2 : total_time = 3)
  (H3 : speed_first_hour = 40)
  (H4 : speed_second_hour = 50)
  (H5 : distance_last_hour = total_distance - (speed_first_hour + speed_second_hour))
  (H6 : average_speed_last_hour = distance_last_hour / 1)
  : average_speed_last_hour = 30 := 
by
  -- Placeholder for the proof
  sorry

end johns_average_speed_last_hour_l66_66272


namespace pastries_eaten_l66_66937

theorem pastries_eaten (total_p: ℕ)
  (hare_fraction: ℚ)
  (dormouse_fraction: ℚ)
  (hare_eaten: ℕ)
  (remaining_after_hare: ℕ)
  (dormouse_eaten: ℕ)
  (final_remaining: ℕ) 
  (hatter_with_left: ℕ) :
  (final_remaining = hatter_with_left) -> hare_fraction = 5 / 16 -> dormouse_fraction = 7 / 11 -> hatter_with_left = 8 -> total_p = 32 -> 
  (total_p = hare_eaten + remaining_after_hare) -> (remaining_after_hare - dormouse_eaten = hatter_with_left) -> (hare_eaten = 10) ∧ (dormouse_eaten = 14) := 
by {
  sorry
}

end pastries_eaten_l66_66937


namespace chessboard_queen_placements_l66_66380

theorem chessboard_queen_placements :
  ∃ (n : ℕ), n = 864 ∧
  (∀ (qpos : Finset (Fin 8 × Fin 8)), 
    qpos.card = 3 ∧
    (∀ (q1 q2 q3 : Fin 8 × Fin 8), 
      q1 ∈ qpos ∧ q2 ∈ qpos ∧ q3 ∈ qpos ∧ q1 ≠ q2 ∧ q2 ≠ q3 ∧ q1 ≠ q3 → 
      (q1.1 = q2.1 ∨ q1.2 = q2.2 ∨ abs (q1.1 - q2.1) = abs (q1.2 - q2.2)) ∧ 
      (q1.1 = q3.1 ∨ q1.2 = q3.2 ∨ abs (q1.1 - q3.1) = abs (q1.2 - q3.2)) ∧ 
      (q2.1 = q3.1 ∨ q2.2 = q3.2 ∨ abs (q2.1 - q3.1) = abs (q2.2 - q3.2)))) ↔ n = 864
:=
by
  sorry

end chessboard_queen_placements_l66_66380


namespace negate_proposition_l66_66263

theorem negate_proposition :
  (¬(∀ x : ℝ, x^2 + x + 1 ≠ 0)) ↔ (∃ x : ℝ, x^2 + x + 1 = 0) :=
by
  sorry

end negate_proposition_l66_66263


namespace trivia_team_average_points_l66_66729

noncomputable def average_points_per_member (total_members didn't_show_up total_points : ℝ) : ℝ :=
  total_points / (total_members - didn't_show_up)

@[simp]
theorem trivia_team_average_points :
  let total_members := 8.0
  let didn't_show_up := 3.5
  let total_points := 12.5
  ∃ avg_points, avg_points = 2.78 ∧ avg_points = average_points_per_member total_members didn't_show_up total_points :=
by
  sorry

end trivia_team_average_points_l66_66729


namespace max_gcd_of_polynomials_l66_66963

def max_gcd (a b : ℤ) : ℤ :=
  let g := Nat.gcd a.natAbs b.natAbs
  Int.ofNat g

theorem max_gcd_of_polynomials :
  ∃ n : ℕ, (n > 0) → max_gcd (14 * ↑n + 5) (9 * ↑n + 2) = 4 :=
by
  sorry

end max_gcd_of_polynomials_l66_66963


namespace tangent_line_at_pi_l66_66949

noncomputable def tangent_equation (x : ℝ) : ℝ := x * Real.sin x

theorem tangent_line_at_pi :
  let f := tangent_equation
  let f' := fun x => Real.sin x + x * Real.cos x
  let x : ℝ := Real.pi
  let y : ℝ := f x
  let slope : ℝ := f' x
  y + slope * x - Real.pi^2 = 0 :=
by
  -- This is where the proof would go
  sorry

end tangent_line_at_pi_l66_66949


namespace min_value_frac_l66_66104

theorem min_value_frac (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 1) :
  ∃ (x : ℝ), x = 16 ∧ (forall y, y = 9 / a + 1 / b → x ≤ y) :=
sorry

end min_value_frac_l66_66104


namespace value_of_A_l66_66384

theorem value_of_A (A B C D : ℕ) (h1 : A * B = 60) (h2 : C * D = 60) (h3 : A - B = C + D) (h4 : A ≠ B) (h5 : A ≠ C) (h6 : A ≠ D) (h7 : B ≠ C) (h8 : B ≠ D) (h9 : C ≠ D) : A = 20 :=
by sorry

end value_of_A_l66_66384


namespace tiffany_max_points_l66_66129

section
  variables
  (initial_money : ℕ := 3)
  (cost_per_game : ℕ := 1)
  (rings_per_game : ℕ := 5)
  (points_red_bucket : ℕ := 2)
  (points_green_bucket : ℕ := 3)
  (points_miss : ℕ := 0)
  (games_played : ℕ := 2)
  (red_buckets : ℕ := 4)
  (green_buckets : ℕ := 5)
  (additional_games : ℕ := initial_money - games_played)
  (points_per_game_from_green_buckets : ℕ := rings_per_game * points_green_bucket)
  (total_points : ℕ := (red_buckets * points_red_bucket) + (green_buckets * points_green_bucket) + (additional_games * points_per_game_from_green_buckets))

  theorem tiffany_max_points : total_points = 38 := 
  sorry
end

end tiffany_max_points_l66_66129


namespace ratio_of_additional_hours_james_danced_l66_66092

-- Definitions based on given conditions
def john_first_dance_time : ℕ := 3
def john_break_time : ℕ := 1
def john_second_dance_time : ℕ := 5
def combined_dancing_time_excluding_break : ℕ := 20

-- Calculations to be proved
def john_total_resting_dancing_time : ℕ :=
  john_first_dance_time + john_break_time + john_second_dance_time

def john_total_dancing_time : ℕ :=
  john_first_dance_time + john_second_dance_time

def james_dancing_time : ℕ :=
  combined_dancing_time_excluding_break - john_total_dancing_time

def additional_hours_james_danced : ℕ :=
  james_dancing_time - john_total_dancing_time

def desired_ratio : ℕ × ℕ :=
  (additional_hours_james_danced, john_total_resting_dancing_time)

-- Theorem to be proved according to the problem statement
theorem ratio_of_additional_hours_james_danced :
  desired_ratio = (4, 9) :=
by
  -- Placeholder for the proof
  sorry

end ratio_of_additional_hours_james_danced_l66_66092


namespace smallest_possible_N_l66_66470

theorem smallest_possible_N (l m n : ℕ) (h_visible : (l - 1) * (m - 1) * (n - 1) = 252) : l * m * n = 392 :=
sorry

end smallest_possible_N_l66_66470


namespace driver_net_rate_of_pay_is_25_l66_66757

noncomputable def net_rate_of_pay_per_hour (hours_traveled : ℝ) (speed : ℝ) (fuel_efficiency : ℝ) (pay_per_mile : ℝ) (fuel_cost_per_gallon : ℝ) : ℝ :=
  let total_distance := speed * hours_traveled
  let total_fuel_used := total_distance / fuel_efficiency
  let total_earnings := pay_per_mile * total_distance
  let total_fuel_cost := fuel_cost_per_gallon * total_fuel_used
  let net_earnings := total_earnings - total_fuel_cost
  net_earnings / hours_traveled

theorem driver_net_rate_of_pay_is_25 :
  net_rate_of_pay_per_hour 3 50 25 0.6 2.5 = 25 := sorry

end driver_net_rate_of_pay_is_25_l66_66757


namespace sum_converges_to_one_l66_66925

noncomputable def series_sum (n: ℕ) : ℝ :=
  if n ≥ 2 then (6 * n^3 - 2 * n^2 - 2 * n + 1) / (n^6 - 2 * n^5 + 2 * n^4 - n^3 + n^2 - 2 * n)
  else 0

theorem sum_converges_to_one : 
  (∑' n, series_sum n) = 1 := by
  sorry

end sum_converges_to_one_l66_66925


namespace solution_y_eq_2_l66_66103

theorem solution_y_eq_2 (y : ℝ) (h_pos : y > 0) (h_eq : y^6 = 64) : y = 2 :=
sorry

end solution_y_eq_2_l66_66103


namespace geometric_series_common_ratio_l66_66193

theorem geometric_series_common_ratio (a S r : ℝ) (h1 : a = 512) (h2 : S = 3072) 
(h3 : S = a / (1 - r)) : r = 5/6 := 
sorry

end geometric_series_common_ratio_l66_66193


namespace remainder_div_eq_4_l66_66756

theorem remainder_div_eq_4 {x y : ℕ} (h1 : y = 25) (h2 : (x / y : ℝ) = 96.16) : x % y = 4 := 
sorry

end remainder_div_eq_4_l66_66756


namespace t_range_inequality_l66_66216

theorem t_range_inequality (t : ℝ) :
  (1/8) * (2 * t - t^2) ≤ -1/4 ∧ 3 - t^2 ≥ 2 ↔ -1 ≤ t ∧ t ≤ 1 - Real.sqrt 3 :=
by
  sorry

end t_range_inequality_l66_66216


namespace monomials_like_terms_l66_66332

theorem monomials_like_terms (a b : ℝ) (m n : ℤ) 
  (h1 : 2 * (a^4) * (b^(-2 * m + 7)) = 3 * (a^(2 * m)) * (b^(n + 2))) :
  m + n = 3 := 
by {
  -- Our proof will be placed here
  sorry
}

end monomials_like_terms_l66_66332


namespace prove_unattainable_y_l66_66706

noncomputable def unattainable_y : Prop :=
  ∀ (x y : ℝ), x ≠ -4 / 3 → y = (2 - x) / (3 * x + 4) → y ≠ -1 / 3

theorem prove_unattainable_y : unattainable_y :=
by
  intro x y h1 h2
  sorry

end prove_unattainable_y_l66_66706


namespace simplify_expression_l66_66862

variable (x : ℝ)

theorem simplify_expression : (x + 2)^2 - (x + 1) * (x + 3) = 1 := 
by 
  sorry

end simplify_expression_l66_66862


namespace find_n_divisible_by_6_l66_66730

theorem find_n_divisible_by_6 (n : Nat) : (71230 + n) % 6 = 0 ↔ n = 2 ∨ n = 8 := by
  sorry

end find_n_divisible_by_6_l66_66730


namespace power_ordering_l66_66939

theorem power_ordering (a b c : ℝ) : 
  (a = 2^30) → (b = 6^10) → (c = 3^20) → (a < b) ∧ (b < c) :=
by
  intros ha hb hc
  rw [ha, hb, hc]
  have h1 : 6^10 = (3 * 2)^10 := by sorry
  have h2 : 3^20 = (3^10)^2 := by sorry
  have h3 : 2^30 = (2^10)^3 := by sorry
  sorry

end power_ordering_l66_66939


namespace part1_part1_eq_part2_tangent_part3_center_range_l66_66896

-- Define the conditions
def A : ℝ × ℝ := (0, 3)
def line_l (x : ℝ) : ℝ := 2 * x - 4
def circle_center_condition (x : ℝ) : ℝ := -x + 5
def radius : ℝ := 1

-- Part (1)
theorem part1 (x y : ℝ) (hx : y = line_l x) (hy : y = circle_center_condition x) :
  (x = 3 ∧ y = 2) :=
sorry

theorem part1_eq :
  ∃ C : ℝ × ℝ, C = (3, 2) ∧ ∀ (x y : ℝ), (x - 3) ^ 2 + (y - 2) ^ 2 = 1 :=
sorry

-- Part (2)
theorem part2_tangent (x y : ℝ) (hx : y = 3) (hy : 3 * x + 4 * y - 12 = 0) :
  ∀ (a b : ℝ), a = 0 ∧ b = -3 / 4 :=
sorry

-- Part (3)
theorem part3_center_range (a : ℝ) (M : ℝ × ℝ) :
  (|2 * a - 4 - 3 / 2| ≤ 1) ->
  (9 / 4 ≤ a ∧ a ≤ 13 / 4) :=
sorry

end part1_part1_eq_part2_tangent_part3_center_range_l66_66896


namespace find_y_l66_66915

theorem find_y (x y : ℤ) (h1 : x^2 - 2*x + 5 = y + 3) (h2 : x = -3) : y = 17 := by
  sorry

end find_y_l66_66915


namespace compare_logs_l66_66441

noncomputable def e := Real.exp 1
noncomputable def log_base_10 (x : Real) := Real.log x / Real.log 10

theorem compare_logs (x : Real) (hx : e < x ∧ x < 10) :
  let a := Real.log (Real.log x)
  let b := log_base_10 (log_base_10 x)
  let c := Real.log (log_base_10 x)
  let d := log_base_10 (Real.log x)
  c < b ∧ b < d ∧ d < a := 
sorry

end compare_logs_l66_66441


namespace complement_set_A_is_04_l66_66699

theorem complement_set_A_is_04 :
  let U := {0, 1, 2, 4}
  let compA := {1, 2}
  ∃ (A : Set ℕ), A = {0, 4} ∧ U = {0, 1, 2, 4} ∧ (U \ A) = compA := 
by
  sorry

end complement_set_A_is_04_l66_66699


namespace arithmetic_geometric_inequality_l66_66709

theorem arithmetic_geometric_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a ≠ b) :
  let A := (a + b) / 2
  let B := Real.sqrt (a * b)
  B < (a - b)^2 / (8 * (A - B)) ∧ (a - b)^2 / (8 * (A - B)) < A :=
by
  let A := (a + b) / 2
  let B := Real.sqrt (a * b)
  sorry

end arithmetic_geometric_inequality_l66_66709


namespace cube_volume_l66_66060

theorem cube_volume (a : ℕ) (h : a^3 - ((a - 2) * a * (a + 2)) = 16) : a^3 = 64 := by
  sorry

end cube_volume_l66_66060


namespace circumradius_of_consecutive_triangle_l66_66795

theorem circumradius_of_consecutive_triangle
  (a b c : ℕ)
  (h : a = b - 1)
  (h1 : c = b + 1)
  (r : ℝ)
  (h2 : r = 4)
  (h3 : a + b > c)
  (h4 : a + c > b)
  (h5 : b + c > a)
  : ∃ R : ℝ, R = 65 / 8 :=
by {
  sorry
}

end circumradius_of_consecutive_triangle_l66_66795


namespace barbara_typing_time_l66_66826

theorem barbara_typing_time:
  let original_speed := 212
  let speed_decrease := 40
  let document_length := 3440
  let new_speed := original_speed - speed_decrease
  (new_speed > 0) → 
  (document_length / new_speed = 20) :=
by
  intros
  sorry

end barbara_typing_time_l66_66826


namespace radius_of_third_circle_l66_66553

open Real

theorem radius_of_third_circle (r : ℝ) :
  let r_large := 40
  let r_small := 25
  let area_large := π * r_large^2
  let area_small := π * r_small^2
  let region_area := area_large - area_small
  let half_region_area := region_area / 2
  let third_circle_area := π * r^2
  (third_circle_area = half_region_area) -> r = 15 * sqrt 13 :=
by
  sorry

end radius_of_third_circle_l66_66553


namespace term_10_of_sequence_l66_66274

theorem term_10_of_sequence (S : ℕ → ℕ) (a : ℕ → ℕ) :
  (∀ n, S n = n * (2 * n + 1)) →
  (∀ n, a n = S n - S (n - 1)) →
  a 10 = 39 :=
by
  intros hS ha
  sorry

end term_10_of_sequence_l66_66274


namespace quadratic_root_expression_value_l66_66964

theorem quadratic_root_expression_value (a : ℝ) 
  (h : a^2 - 2 * a - 3 = 0) : 2 * a^2 - 4 * a + 1 = 7 :=
by
  sorry

end quadratic_root_expression_value_l66_66964


namespace inverse_sum_l66_66117

noncomputable def g (x : ℝ) : ℝ :=
if x < 15 then 2 * x + 4 else 3 * x - 1

theorem inverse_sum :
  g⁻¹ (10) + g⁻¹ (50) = 20 :=
sorry

end inverse_sum_l66_66117


namespace starting_number_range_l66_66109

theorem starting_number_range (n : ℕ) (h₁: ∀ m : ℕ, (m > n) → (m ≤ 50) → (m = 55) → True) : n = 54 :=
sorry

end starting_number_range_l66_66109


namespace largest_possible_number_of_markers_l66_66958

theorem largest_possible_number_of_markers (n_m n_c : ℕ) 
  (h_m : n_m = 72) (h_c : n_c = 48) : Nat.gcd n_m n_c = 24 :=
by
  sorry

end largest_possible_number_of_markers_l66_66958


namespace intersection_points_on_circle_l66_66735

theorem intersection_points_on_circle (u : ℝ) :
  ∃ (r : ℝ), ∀ (x y : ℝ), (u * x - 3 * y - 2 * u = 0) ∧ (2 * x - 3 * u * y + u = 0) → (x^2 + y^2 = r^2) :=
sorry

end intersection_points_on_circle_l66_66735


namespace find_sum_A_B_C_l66_66080

theorem find_sum_A_B_C (A B C : ℤ)
  (h1 : ∀ x > 4, (x^2 : ℝ) / (A * x^2 + B * x + C) > 0.4)
  (h2 : A * (-2)^2 + B * (-2) + C = 0)
  (h3 : A * (3)^2 + B * (3) + C = 0)
  (h4 : 0.4 < 1 / (A : ℝ) ∧ 1 / (A : ℝ) < 1) :
  A + B + C = -12 :=
by
  sorry

end find_sum_A_B_C_l66_66080


namespace quadratic_increasing_l66_66061

noncomputable def quadratic (a b c x : ℝ) := a * x^2 + b * x + c

theorem quadratic_increasing (a b c : ℝ) 
  (h1 : quadratic a b c 0 = quadratic a b c 6)
  (h2 : quadratic a b c 0 < quadratic a b c 7) :
  ∀ x, x > 3 → ∀ y, y > 3 → x < y → quadratic a b c x < quadratic a b c y :=
sorry

end quadratic_increasing_l66_66061


namespace noemi_lost_on_roulette_l66_66415

theorem noemi_lost_on_roulette (initial_purse := 1700) (final_purse := 800) (loss_on_blackjack := 500) :
  (initial_purse - final_purse) - loss_on_blackjack = 400 := by
  sorry

end noemi_lost_on_roulette_l66_66415


namespace gabriel_pages_correct_l66_66205

-- Given conditions
def beatrix_pages : ℕ := 704

def cristobal_pages (b : ℕ) : ℕ := 3 * b + 15

def gabriel_pages (c b : ℕ) : ℕ := 3 * (c + b)

-- Problem statement
theorem gabriel_pages_correct : gabriel_pages (cristobal_pages beatrix_pages) beatrix_pages = 8493 :=
by 
  sorry

end gabriel_pages_correct_l66_66205


namespace min_value_of_quadratic_function_l66_66719

def quadratic_function (x : ℝ) : ℝ :=
  x^2 + 2 * x - 5

theorem min_value_of_quadratic_function :
  ∃ x : ℝ, quadratic_function x = -1 :=
by
  sorry

end min_value_of_quadratic_function_l66_66719


namespace a_eq_bn_l66_66581

theorem a_eq_bn (a b n : ℕ) :
  (∀ k : ℕ, k ≠ b → ∃ m : ℕ, a - k^n = m * (b - k)) → a = b^n :=
by
  sorry

end a_eq_bn_l66_66581


namespace fifty_third_card_is_A_l66_66081

noncomputable def card_seq : List String := 
  ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]

theorem fifty_third_card_is_A : card_seq[(53 % 13)] = "A" := 
by 
  simp [card_seq] 
  sorry

end fifty_third_card_is_A_l66_66081


namespace howard_rewards_l66_66261

theorem howard_rewards (initial_bowls : ℕ) (customers : ℕ) (customers_bought_20 : ℕ) 
                       (bowls_remaining : ℕ) (rewards_per_bowl : ℕ) :
  initial_bowls = 70 → 
  customers = 20 → 
  customers_bought_20 = 10 → 
  bowls_remaining = 30 → 
  rewards_per_bowl = 2 →
  ∀ (bowls_bought_per_customer : ℕ), bowls_bought_per_customer = 20 → 
  2 * (200 / 20) = 10 := 
by 
  intros h1 h2 h3 h4 h5 h6
  sorry

end howard_rewards_l66_66261


namespace age_of_b_l66_66264

theorem age_of_b (a b c : ℕ) (h1 : a = b + 2) (h2 : b = 2 * c) (h3 : a + b + c = 47) : b = 18 :=
by
  sorry

end age_of_b_l66_66264


namespace divisor_is_3_l66_66244

theorem divisor_is_3 (divisor quotient remainder : ℕ) (h_dividend : 22 = (divisor * quotient) + remainder) 
  (h_quotient : quotient = 7) (h_remainder : remainder = 1) : divisor = 3 :=
by
  sorry

end divisor_is_3_l66_66244


namespace dividend_is_10_l66_66993

theorem dividend_is_10
  (q d r : ℕ)
  (hq : q = 3)
  (hd : d = 3)
  (hr : d = 3 * r) :
  (q * d + r = 10) :=
by
  sorry

end dividend_is_10_l66_66993


namespace lines_through_three_distinct_points_l66_66151

theorem lines_through_three_distinct_points : 
  ∃ n : ℕ, n = 54 ∧ (∀ (i j k : ℕ), 1 ≤ i ∧ i ≤ 3 ∧ 1 ≤ j ∧ j ≤ 3 ∧ 1 ≤ k ∧ k ≤ 3 → 
  ∃ (a b c : ℤ), -- Direction vector (a, b, c)
  abs a ≤ 1 ∧ abs b ≤ 1 ∧ abs c ≤ 1 ∧
  ((i + a > 0 ∧ i + a ≤ 3) ∧ (j + b > 0 ∧ j + b ≤ 3) ∧ (k + c > 0 ∧ k + c ≤ 3) ∧
  (i + 2 * a > 0 ∧ i + 2 * a ≤ 3) ∧ (j + 2 * b > 0 ∧ j + 2 * b ≤ 3) ∧ (k + 2 * c > 0 ∧ k + 2 * c ≤ 3))) := 
sorry

end lines_through_three_distinct_points_l66_66151


namespace part1_part2_l66_66296

noncomputable def a_n (n : ℕ) : ℕ :=
  2^(n - 1)

noncomputable def b_n (n : ℕ) : ℕ :=
  2 * n

noncomputable def S_n (n : ℕ) : ℕ :=
  n^2 + n

theorem part1 (n : ℕ) : 
  S_n n = n^2 + n := 
sorry

noncomputable def C_n (n : ℕ) : ℚ :=
  (n^2 + n) / 2^(n - 1)

theorem part2 (n : ℕ) (k : ℕ) (k_gt_0 : 0 < k) : 
  (∀ n, C_n n ≤ C_n k) ↔ (k = 2 ∨ k = 3) :=
sorry

end part1_part2_l66_66296


namespace marbles_per_friend_l66_66422

theorem marbles_per_friend (total_marbles : ℕ) (num_friends : ℕ) (h_total : total_marbles = 30) (h_friends : num_friends = 5) :
  total_marbles / num_friends = 6 :=
by
  -- Proof skipped
  sorry

end marbles_per_friend_l66_66422


namespace lap_length_l66_66140

theorem lap_length (I P : ℝ) (K : ℝ) 
  (h1 : 2 * I - 2 * P = 3 * K) 
  (h2 : 3 * I + 10 - 3 * P = 7 * K) : 
  K = 4 :=
by 
  -- Proof goes here
  sorry

end lap_length_l66_66140


namespace pen_and_notebook_cost_l66_66478

theorem pen_and_notebook_cost :
  ∃ (p n : ℕ), 15 * p + 5 * n = 130 ∧ p > n ∧ p + n = 10 := by
  sorry

end pen_and_notebook_cost_l66_66478


namespace tangent_line_at_point_P_l66_66183

-- Definitions from Conditions
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 5
def point_on_circle : Prop := circle_eq 1 2

-- Statement to Prove
theorem tangent_line_at_point_P : 
  point_on_circle → ∃ (m : ℝ) (b : ℝ), (m = -1/2) ∧ (b = 5/2) ∧ (∀ x y : ℝ, y = m * x + b ↔ x + 2 * y - 5 = 0) :=
by
  sorry

end tangent_line_at_point_P_l66_66183


namespace value_of_y_when_x_is_zero_l66_66794

noncomputable def quadratic_y (h x : ℝ) : ℝ := -(x + h)^2

theorem value_of_y_when_x_is_zero :
  ∀ (h : ℝ), (∀ x, x < -3 → quadratic_y h x < quadratic_y h (-3)) →
            (∀ x, x > -3 → quadratic_y h x < quadratic_y h (-3)) →
            quadratic_y h 0 = -9 :=
by
  sorry

end value_of_y_when_x_is_zero_l66_66794


namespace minimum_value_f_l66_66554

noncomputable def f (x y : ℝ) : ℝ :=
  x^2 + y^2 + 16 / (x + y)^2

theorem minimum_value_f (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  ∃ t ≥ 0, ∀ (x y : ℝ), 0 < x → 0 < y → f x y ≥ t ∧ t = 4 * Real.sqrt 2 :=
sorry

end minimum_value_f_l66_66554


namespace area_enclosed_by_S_l66_66822

open Complex

def five_presentable (v : ℂ) : Prop := abs v = 5

def S : Set ℂ := {u | ∃ v : ℂ, five_presentable v ∧ u = v - (1 / v)}

theorem area_enclosed_by_S : 
  ∃ (area : ℝ), area = 624 / 25 * Real.pi :=
by
  sorry

end area_enclosed_by_S_l66_66822


namespace sequence_sum_l66_66059

theorem sequence_sum (a : ℕ → ℚ) (S : ℕ → ℚ) (h : ∀ n : ℕ, S n + a n = 2 * n + 1) :
  ∀ n : ℕ, a n = 2 - (1 / 2^n) :=
by
  sorry

end sequence_sum_l66_66059


namespace inequality_solution_l66_66085

variable (a x : ℝ)

noncomputable def inequality_solutions :=
  if a = 0 then
    {x | x > 1}
  else if a > 1 then
    {x | (1 / a) < x ∧ x < 1}
  else if a = 1 then
    ∅
  else if 0 < a ∧ a < 1 then
    {x | 1 < x ∧ x < (1 / a)}
  else if a < 0 then
    {x | x < (1 / a) ∨ x > 1}
  else
    ∅

theorem inequality_solution (h : a ≠ 0) :
  if a = 0 then
    ∀ x, (a * x - 1) * (x - 1) < 0 → x > 1
  else if a > 1 then
    ∀ x, (a * x - 1) * (x - 1) < 0 ↔ ((1 / a) < x ∧ x < 1)
  else if a = 1 then
    ∀ x, ¬((a * x - 1) * (x - 1) < 0)
  else if 0 < a ∧ a < 1 then
    ∀ x, (a * x - 1) * (x - 1) < 0 ↔ (1 < x ∧ x < (1 / a))
  else if a < 0 then
    ∀ x, (a * x - 1) * (x - 1) < 0 ↔ (x < (1 / a) ∨ x > 1)
  else
    True := sorry

end inequality_solution_l66_66085


namespace min_value_of_expression_l66_66449

theorem min_value_of_expression (a b c : ℝ) (h : 0 < a) (h1 : 0 < b) (h2 : 0 < c) (h3 : a * b * c = 27) :
  a^2 + 2*a*b + b^2 + 3*c^2 ≥ 324 :=
sorry

end min_value_of_expression_l66_66449


namespace deposit_amount_l66_66375

theorem deposit_amount (P : ℝ) (h₀ : 0.1 * P + 720 = P) : 0.1 * P = 80 :=
by
  sorry

end deposit_amount_l66_66375


namespace rate_per_kg_for_mangoes_l66_66514

theorem rate_per_kg_for_mangoes (quantity_grapes : ℕ)
    (rate_grapes : ℕ)
    (quantity_mangoes : ℕ)
    (total_payment : ℕ)
    (rate_mangoes : ℕ) :
    quantity_grapes = 8 →
    rate_grapes = 70 →
    quantity_mangoes = 9 →
    total_payment = 1055 →
    8 * 70 + 9 * rate_mangoes = 1055 →
    rate_mangoes = 55 := by
  intros h1 h2 h3 h4 h5
  have h6 : 8 * 70 = 560 := by norm_num
  have h7 : 560 + 9 * rate_mangoes = 1055 := by rw [h5]
  have h8 : 1055 - 560 = 495 := by norm_num
  have h9 : 9 * rate_mangoes = 495 := by linarith
  have h10 : rate_mangoes = 55 := by linarith
  exact h10

end rate_per_kg_for_mangoes_l66_66514


namespace part_1_part_2_part_3_l66_66557

def whiteHorseNumber (a b c : ℚ) : ℚ :=
  min (a - b) (min ((a - c) / 2) ((b - c) / 3))

theorem part_1 : 
  whiteHorseNumber (-2) (-4) 1 = -5/3 :=
by sorry

theorem part_2 : 
  max (whiteHorseNumber (-2) (-4) 1) (max (whiteHorseNumber (-2) 1 (-4)) 
  (max (whiteHorseNumber (-4) (-2) 1) (max (whiteHorseNumber (-4) 1 (-2)) 
  (max (whiteHorseNumber 1 (-4) (-2)) (whiteHorseNumber 1 (-2) (-4)) )))) = 2/3 :=
by sorry

theorem part_3 (x : ℚ) (h : ∃a b c : ℚ, a = -1 ∧ b = 6 ∧ c = x ∧ whiteHorseNumber a b c = 2) : 
  x = -7 ∨ x = 8 :=
by sorry

end part_1_part_2_part_3_l66_66557


namespace quotient_of_5_divided_by_y_is_5_point_3_l66_66530

theorem quotient_of_5_divided_by_y_is_5_point_3 (y : ℝ) (h : 5 / y = 5.3) : y = 26.5 :=
by
  sorry

end quotient_of_5_divided_by_y_is_5_point_3_l66_66530


namespace probability_first_hearts_second_ace_correct_l66_66082

noncomputable def probability_first_hearts_second_ace : ℚ :=
  let total_cards := 104
  let total_aces := 8 -- 4 aces per deck, 2 decks
  let hearts_count := 2 * 13 -- 13 hearts per deck, 2 decks
  let ace_of_hearts_count := 2

  -- Case 1: the first is an ace of hearts
  let prob_first_ace_of_hearts := (ace_of_hearts_count : ℚ) / total_cards
  let prob_second_ace_given_first_ace_of_hearts := (total_aces - 1 : ℚ) / (total_cards - 1)

  -- Case 2: the first is a hearts but not an ace
  let prob_first_hearts_not_ace := (hearts_count - ace_of_hearts_count : ℚ) / total_cards
  let prob_second_ace_given_first_hearts_not_ace := total_aces / (total_cards - 1)

  -- Combined probability
  (prob_first_ace_of_hearts * prob_second_ace_given_first_ace_of_hearts) +
  (prob_first_hearts_not_ace * prob_second_ace_given_first_hearts_not_ace)

theorem probability_first_hearts_second_ace_correct : 
  probability_first_hearts_second_ace = 7 / 453 := 
sorry

end probability_first_hearts_second_ace_correct_l66_66082


namespace range_of_a_l66_66392

-- Definitions from conditions 
def p (x : ℝ) : Prop := |x + 1| > 2
def q (x a : ℝ) : Prop := x > a

-- The Lean statement for the problem
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, -3 ≤ x ∧ x ≤ 1 → x ≤ a) → a ≥ 1 :=
by sorry

end range_of_a_l66_66392


namespace sqrt_of_16_is_4_l66_66027

def arithmetic_square_root (x : ℕ) : ℕ :=
  if x = 0 then 0 else Nat.sqrt x

theorem sqrt_of_16_is_4 : arithmetic_square_root 16 = 4 :=
by
  sorry

end sqrt_of_16_is_4_l66_66027


namespace sample_size_l66_66278

variable (x n : ℕ)

-- Conditions as definitions
def staff_ratio : Prop := 15 * x + 3 * x + 2 * x = 20 * x
def sales_staff : Prop := 30 / n = 15 / 20

-- Main statement to prove
theorem sample_size (h1: staff_ratio x) (h2: sales_staff n) : n = 40 := by
  sorry

end sample_size_l66_66278


namespace shaded_regions_area_sum_l66_66171

theorem shaded_regions_area_sum (side_len : ℚ) (radius : ℚ) (a b c : ℤ) :
  side_len = 16 → radius = side_len / 2 →
  a = (64 / 3) ∧ b = 32 ∧ c = 3 →
  (∃ x : ℤ, x = a + b + c ∧ x = 99) :=
by
  intros hside_len hradius h_constituents
  sorry

end shaded_regions_area_sum_l66_66171


namespace vector_decomposition_l66_66632

noncomputable def x : ℝ × ℝ × ℝ := (5, 15, 0)
noncomputable def p : ℝ × ℝ × ℝ := (1, 0, 5)
noncomputable def q : ℝ × ℝ × ℝ := (-1, 3, 2)
noncomputable def r : ℝ × ℝ × ℝ := (0, -1, 1)

theorem vector_decomposition : x = (4 : ℝ) • p + (-1 : ℝ) • q + (-18 : ℝ) • r :=
by
  sorry

end vector_decomposition_l66_66632


namespace count_neither_3_nor_4_l66_66980

def is_multiple_of_3_or_4 (n : Nat) : Bool := (n % 3 = 0) ∨ (n % 4 = 0)

def three_digit_numbers := List.range' 100 900 -- Generates a list from 100 to 999 (inclusive)

def count_multiples_of_3_or_4 : Nat := three_digit_numbers.filter is_multiple_of_3_or_4 |>.length

def count_total := 900 -- Since three-digit numbers range from 100 to 999

theorem count_neither_3_nor_4 : count_total - count_multiples_of_3_or_4 = 450 := by
  sorry

end count_neither_3_nor_4_l66_66980


namespace side_length_of_square_l66_66368

noncomputable def area_of_circle : ℝ := 3848.4510006474966
noncomputable def pi : ℝ := Real.pi

theorem side_length_of_square :
  ∃ s : ℝ, (∃ r : ℝ, area_of_circle = pi * r * r ∧ 2 * r = s) ∧ s = 70 := 
by
  sorry

end side_length_of_square_l66_66368


namespace jade_handled_84_transactions_l66_66114

def Mabel_transactions : ℕ := 90

def Anthony_transactions (mabel : ℕ) : ℕ := mabel + mabel / 10

def Cal_transactions (anthony : ℕ) : ℕ := (2 * anthony) / 3

def Jade_transactions (cal : ℕ) : ℕ := cal + 18

theorem jade_handled_84_transactions :
  Jade_transactions (Cal_transactions (Anthony_transactions Mabel_transactions)) = 84 := 
sorry

end jade_handled_84_transactions_l66_66114


namespace find_q_l66_66451

variable (p q : ℝ) (hp : p > 1) (hq : q > 1) (h_cond1 : 1 / p + 1 / q = 1) (h_cond2 : p * q = 9)

theorem find_q : q = (9 + 3 * Real.sqrt 5) / 2 :=
sorry

end find_q_l66_66451


namespace least_integer_value_x_l66_66503

theorem least_integer_value_x (x : ℤ) (h : |(2 : ℤ) * x + 3| ≤ 12) : x = -7 :=
by
  sorry

end least_integer_value_x_l66_66503


namespace trader_loses_l66_66475

theorem trader_loses 
  (l_1 l_2 q : ℝ) 
  (h1 : l_1 ≠ l_2) 
  (p_1 p_2 : ℝ) 
  (h2 : p_1 = q * (l_2 / l_1)) 
  (h3 : p_2 = q * (l_1 / l_2)) :
  p_1 + p_2 > 2 * q :=
by {
  sorry
}

end trader_loses_l66_66475


namespace max_g_equals_sqrt3_l66_66748

noncomputable def f (x : ℝ) : ℝ :=
  Real.sin (x + Real.pi / 9) + Real.sin (5 * Real.pi / 9 - x)

noncomputable def g (x : ℝ) : ℝ :=
  f (f x)

theorem max_g_equals_sqrt3 : ∀ x, g x ≤ Real.sqrt 3 :=
by
  sorry

end max_g_equals_sqrt3_l66_66748


namespace solve_for_ab_l66_66959

theorem solve_for_ab (a b : ℤ) 
  (h1 : a + 3 * b = 27) 
  (h2 : 5 * a + 4 * b = 47) : 
  a + b = 11 :=
sorry

end solve_for_ab_l66_66959


namespace simplify_fraction_l66_66373

theorem simplify_fraction : (2 / (1 - (2 / 3))) = 6 :=
by
  sorry

end simplify_fraction_l66_66373


namespace smallest_integer_C_l66_66710

-- Define the function f(n) = 6^n / n!
def f (n : ℕ) : ℚ := (6 ^ n) / (Nat.factorial n)

theorem smallest_integer_C (C : ℕ) (h : ∀ n : ℕ, n > 0 → f n ≤ C) : C = 65 :=
by
  sorry

end smallest_integer_C_l66_66710


namespace number_of_happy_configurations_is_odd_l66_66320

def S (m n : ℕ) := {p : ℕ × ℕ | 1 ≤ p.1 ∧ p.1 ≤ 2 * m ∧ 1 ≤ p.2 ∧ p.2 ≤ 2 * n}

def happy_configurations (m n : ℕ) : ℕ := 
  sorry -- definition of the number of happy configurations is abstracted for this statement.

theorem number_of_happy_configurations_is_odd (m n : ℕ) (hm : 0 < m) (hn : 0 < n) : 
  happy_configurations m n % 2 = 1 := 
sorry

end number_of_happy_configurations_is_odd_l66_66320


namespace slope_of_tangent_line_l66_66781

theorem slope_of_tangent_line 
  (center point : ℝ × ℝ) 
  (h_center : center = (5, 3)) 
  (h_point : point = (8, 8)) 
  : (∃ m : ℚ, m = -3/5) :=
sorry

end slope_of_tangent_line_l66_66781


namespace range_a_l66_66189

noncomputable def f (x : ℝ) : ℝ := Real.log ((1 + x) / (x - 2))

def domain_A : Set ℝ := { x | x < -1 ∨ x > 2 }

def solution_set_B (a : ℝ) : Set ℝ := { x | x < a ∨ x > a + 1 }

theorem range_a (a : ℝ)
  (h : (domain_A ∪ solution_set_B a) = solution_set_B a) :
  -1 ≤ a ∧ a ≤ 1 :=
by
  sorry

end range_a_l66_66189


namespace least_five_digit_perfect_square_and_cube_l66_66971

theorem least_five_digit_perfect_square_and_cube :
  ∃ n : ℕ, 10000 ≤ n ∧ n < 100000 ∧ ∃ k : ℕ, k^6 = n ∧ n = 15625 :=
by
  sorry

end least_five_digit_perfect_square_and_cube_l66_66971


namespace maximum_value_F_l66_66141

noncomputable def f (x : Real) : Real := Real.sin x + Real.cos x

noncomputable def f' (x : Real) : Real := Real.cos x - Real.sin x

noncomputable def F (x : Real) : Real := f x * f' x + (f x) ^ 2

theorem maximum_value_F : ∃ x : Real, F x = 1 + Real.sqrt 2 :=
by
  -- The proof steps are to be added here.
  sorry

end maximum_value_F_l66_66141


namespace sequence_a_n_sequence_b_n_range_k_l66_66018

-- Define the geometric sequence {a_n} with initial conditions
def a (n : ℕ) : ℕ :=
  3 * 2^(n-1)

-- Define the sequence {b_n} with the given recurrence relation
def b : ℕ → ℕ
| 0 => 1
| (n+1) => 2 * (b n) + 1

theorem sequence_a_n (n : ℕ) : 
  (a n = 3 * 2^(n-1)) := sorry

theorem sequence_b_n (n : ℕ) :
  (b n = 2^n - 1) := sorry

-- Define the condition for k and the inequality
def condition_k (k : ℝ) : Prop :=
  ∀ n : ℕ, n > 0 → (k * (↑(b n) + 5) / 2 - 3 * 2^(n-1) ≥ 8*n + 2*k - 24)

-- Prove the range for k
theorem range_k (k : ℝ) :
  (condition_k k ↔ k ≥ 4) := sorry

end sequence_a_n_sequence_b_n_range_k_l66_66018


namespace remainder_when_divided_by_8_l66_66339

theorem remainder_when_divided_by_8 (x k : ℤ) (h : x = 63 * k + 27) : x % 8 = 3 :=
sorry

end remainder_when_divided_by_8_l66_66339


namespace price_reduction_l66_66945

theorem price_reduction (x : ℝ) : 
  188 * (1 - x) ^ 2 = 108 :=
sorry

end price_reduction_l66_66945


namespace inequality_abc_l66_66840

theorem inequality_abc 
  (a b c : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hc : c > 0) : 
  a * b * c ≥ (b + c - a) * (a + c - b) * (a + b - c) := 
by 
  sorry

end inequality_abc_l66_66840


namespace triangle_area_l66_66367

noncomputable def a := 5
noncomputable def b := 4
noncomputable def s := (13 : ℝ) / 2 -- semi-perimeter
noncomputable def area := Real.sqrt (s * (s - a) * (s - b) * (s - b))

theorem triangle_area :
  a + 2 * b = 13 →
  (a > 0) → (b > 0) →
  (a < 2 * b) →
  (a + b > b) → 
  (a + b > b) →
  area = Real.sqrt 61.09375 :=
by
  intros h1 h2 h3 h4 h5 h6
  -- We assume validity of these conditions and skip the proof for brevity.
  sorry

end triangle_area_l66_66367


namespace lambs_goats_solution_l66_66005

theorem lambs_goats_solution : ∃ l g : ℕ, l > 0 ∧ g > 0 ∧ 30 * l + 32 * g = 1200 ∧ l = 24 ∧ g = 15 :=
by
  existsi 24
  existsi 15
  repeat { split }
  sorry

end lambs_goats_solution_l66_66005


namespace work_rate_D_time_A_B_D_time_D_l66_66933

def workRate (person : String) : ℚ :=
  if person = "A" then 1/12 else
  if person = "B" then 1/6 else
  if person = "A_D" then 1/4 else
  0

theorem work_rate_D : workRate "A_D" - workRate "A" = 1/6 := by
  sorry

theorem time_A_B_D : (1 / (workRate "A" + workRate "B" + (workRate "A_D" - workRate "A"))) = 2.4 := by
  sorry
  
theorem time_D : (1 / (workRate "A_D" - workRate "A")) = 6 := by
  sorry

end work_rate_D_time_A_B_D_time_D_l66_66933


namespace five_eight_sided_dice_not_all_same_l66_66149

noncomputable def probability_not_all_same : ℚ :=
  let total_outcomes := 8^5
  let same_number_outcomes := 8
  1 - (same_number_outcomes / total_outcomes)

theorem five_eight_sided_dice_not_all_same :
  probability_not_all_same = 4095 / 4096 :=
by
  sorry

end five_eight_sided_dice_not_all_same_l66_66149


namespace tagged_fish_proportion_l66_66639

def total_fish_in_pond : ℕ := 750
def tagged_fish_first_catch : ℕ := 30
def fish_second_catch : ℕ := 50
def tagged_fish_second_catch := 2

theorem tagged_fish_proportion :
  (tagged_fish_second_catch : ℤ) * (total_fish_in_pond : ℤ) = (tagged_fish_first_catch : ℤ) * (fish_second_catch : ℤ) :=
by
  -- The statement should reflect the given proportion:
  -- T * 750 = 30 * 50
  -- Given T = 2
  sorry

end tagged_fish_proportion_l66_66639


namespace profit_without_discount_l66_66914

theorem profit_without_discount (CP SP MP : ℝ) (discountRate profitRate : ℝ)
  (h1 : CP = 100)
  (h2 : discountRate = 0.05)
  (h3 : profitRate = 0.235)
  (h4 : SP = CP * (1 + profitRate))
  (h5 : MP = SP / (1 - discountRate)) :
  (((MP - CP) / CP) * 100) = 30 := 
sorry

end profit_without_discount_l66_66914


namespace greater_number_is_18_l66_66644

theorem greater_number_is_18 (x y : ℕ) (h₁ : x + y = 30) (h₂ : x - y = 6) : x = 18 :=
by
  sorry

end greater_number_is_18_l66_66644


namespace find_three_digit_number_in_decimal_l66_66207

theorem find_three_digit_number_in_decimal :
  ∃ (A B C : ℕ), ∀ (hA : A ≠ 0 ∧ A < 7) (hB : B ≠ 0 ∧ B < 7) (hC : C ≠ 0 ∧ C < 7) (h_distinct : A ≠ B ∧ B ≠ C ∧ A ≠ C)
    (h1 : (7 * A + B) + C = 7 * C)
    (h2 : (7 * A + B) + (7 * B + A) = 7 * B + 6), 
    A * 100 + B * 10 + C = 425 :=
by
  sorry

end find_three_digit_number_in_decimal_l66_66207


namespace birds_not_herons_are_geese_l66_66850

-- Define the given conditions
def percentage_geese : ℝ := 0.35
def percentage_swans : ℝ := 0.20
def percentage_herons : ℝ := 0.15
def percentage_ducks : ℝ := 0.30

-- Definition without herons
def percentage_non_herons : ℝ := 1 - percentage_herons

-- Theorem to prove
theorem birds_not_herons_are_geese :
  (percentage_geese / percentage_non_herons) * 100 = 41 :=
by
  sorry

end birds_not_herons_are_geese_l66_66850


namespace greatest_value_x_plus_y_l66_66035

theorem greatest_value_x_plus_y (x y : ℝ) (h1 : x^2 + y^2 = 100) (h2 : x * y = 40) : 
  x + y ≤ 6 * Real.sqrt 5 := 
by
  sorry

end greatest_value_x_plus_y_l66_66035


namespace circle_center_coordinates_l66_66917

theorem circle_center_coordinates (b c p q : ℝ) 
    (h_circle_eq : ∀ x y : ℝ, x^2 + y^2 - 2 * p * x - 2 * q * y + 2 * q - 1 = 0) 
    (h_quad_roots : ∀ x : ℝ, x^2 + b * x + c = 0) 
    (h_condition : b^2 - 4 * c ≥ 0) : 
    (p = -b / 2) ∧ (q = (1 + c) / 2) := 
sorry

end circle_center_coordinates_l66_66917


namespace mean_of_sets_l66_66856

theorem mean_of_sets (x : ℝ) (h : (28 + x + 42 + 78 + 104) / 5 = 62) : 
  (48 + 62 + 98 + 124 + x) / 5 = 78 :=
by
  sorry

end mean_of_sets_l66_66856


namespace geologists_probability_l66_66674

theorem geologists_probability
  (n roads : ℕ) (speed_per_hour : ℕ) 
  (angle_between_neighbors : ℕ)
  (distance_limit : ℝ) : 
  n = 6 ∧ speed_per_hour = 4 ∧ angle_between_neighbors = 60 ∧ distance_limit = 6 → 
  prob_distance_at_least_6_km = 0.5 :=
by
  sorry

noncomputable def prob_distance_at_least_6_km : ℝ := 0.5  -- Placeholder definition

end geologists_probability_l66_66674


namespace bacteria_reaches_final_in_24_hours_l66_66445

-- Define the initial number of bacteria
def initial_bacteria : ℕ := 200

-- Define the final number of bacteria
def final_bacteria : ℕ := 16200

-- Define the tripling period in hours
def tripling_period : ℕ := 6

-- Define the tripling factor
def tripling_factor : ℕ := 3

-- Define the number of hours needed to reach final number of bacteria
def hours_to_reach_final_bacteria : ℕ := 24

-- Define a function that models the number of bacteria after t hours
def bacteria_after (t : ℕ) : ℕ :=
  initial_bacteria * tripling_factor^((t / tripling_period))

-- Main statement of the problem: prove that the number of bacteria is 16200 after 24 hours
theorem bacteria_reaches_final_in_24_hours :
  bacteria_after hours_to_reach_final_bacteria = final_bacteria :=
sorry

end bacteria_reaches_final_in_24_hours_l66_66445


namespace product_of_two_numbers_l66_66902

theorem product_of_two_numbers (a b : ℕ) (h_gcd : Nat.gcd a b = 8) (h_lcm : Nat.lcm a b = 72) : a * b = 576 := 
by
  sorry

end product_of_two_numbers_l66_66902


namespace right_triangle_acute_angles_l66_66634

theorem right_triangle_acute_angles (α β : ℝ) 
  (h1 : α + β = 90)
  (h2 : ∀ (δ1 δ2 ε1 ε2 : ℝ), δ1 + ε1 = 135 ∧ δ1 / ε1 = 13 / 17 
                       ∧ ε2 = 180 - ε1 ∧ δ2 = 180 - δ1) :
  α = 63 ∧ β = 27 := 
  sorry

end right_triangle_acute_angles_l66_66634


namespace average_monthly_increase_l66_66979

theorem average_monthly_increase (x : ℝ) (turnover_january turnover_march : ℝ)
  (h_jan : turnover_january = 2)
  (h_mar : turnover_march = 2.88)
  (h_growth : turnover_march = turnover_january * (1 + x) * (1 + x)) :
  x = 0.2 :=
by
  sorry

end average_monthly_increase_l66_66979


namespace main_theorem_l66_66647

def d_digits (d : ℕ) : Prop :=
  ∃ (d_1 d_2 d_3 d_4 d_5 d_6 d_7 d_8 d_9 : ℕ),
    d = d_1 * 10^8 + d_2 * 10^7 + d_3 * 10^6 + d_4 * 10^5 + d_5 * 10^4 + d_6 * 10^3 + d_7 * 10^2 + d_8 * 10 + d_9

noncomputable def condition1 (d e : ℕ) (i : ℕ) : Prop :=
  (e - (d / 10^(8 - i) % 10)) * 10^(8 - i) + d ≡ 0 [MOD 7]

noncomputable def condition2 (e f : ℕ) (i : ℕ) : Prop :=
  (f - (e / 10^(8 - i) % 10)) * 10^(8 - i) + e ≡ 0 [MOD 7]

theorem main_theorem
  (d e f : ℕ)
  (h1 : d_digits d)
  (h2 : ∀ i, 1 ≤ i ∧ i ≤ 9 → condition1 d e i)
  (h3 : ∀ i, 1 ≤ i ∧ i ≤ 9 → condition2 e f i) :
  ∀ i, 1 ≤ i ∧ i ≤ 9 → (d / 10^(8 - i) % 10) ≡ (f / 10^(8 - i) % 10) [MOD 7] := sorry

end main_theorem_l66_66647


namespace infinitesimal_alpha_as_t_to_zero_l66_66444

open Real

noncomputable def alpha (t : ℝ) : ℝ × ℝ :=
  (t, sin t)

theorem infinitesimal_alpha_as_t_to_zero : 
  ∀ ε > 0, ∃ δ > 0, ∀ t : ℝ, abs t < δ → abs (alpha t).fst + abs (alpha t).snd < ε := by
  sorry

end infinitesimal_alpha_as_t_to_zero_l66_66444


namespace arithmetic_mean_25_41_50_l66_66763

theorem arithmetic_mean_25_41_50 :
  (25 + 41 + 50) / 3 = 116 / 3 := by
  sorry

end arithmetic_mean_25_41_50_l66_66763


namespace trader_profit_l66_66536

noncomputable def original_price (P : ℝ) : ℝ := P
noncomputable def purchase_price (P : ℝ) : ℝ := 0.8 * P
noncomputable def depreciation1 (P : ℝ) : ℝ := 0.04 * P
noncomputable def depreciation2 (P : ℝ) : ℝ := 0.038 * P
noncomputable def value_after_depreciation (P : ℝ) : ℝ := 0.722 * P
noncomputable def taxes (P : ℝ) : ℝ := 0.024 * P
noncomputable def insurance (P : ℝ) : ℝ := 0.032 * P
noncomputable def maintenance (P : ℝ) : ℝ := 0.01 * P
noncomputable def total_cost (P : ℝ) : ℝ := value_after_depreciation P + taxes P + insurance P + maintenance P
noncomputable def selling_price (P : ℝ) : ℝ := 1.70 * total_cost P
noncomputable def profit (P : ℝ) : ℝ := selling_price P - original_price P
noncomputable def profit_percent (P : ℝ) : ℝ := (profit P / original_price P) * 100

theorem trader_profit (P : ℝ) : profit_percent P = 33.96 :=
  by
    sorry

end trader_profit_l66_66536


namespace find_pairs_l66_66052

theorem find_pairs (x y : Nat) (h : 1 + x + x^2 + x^3 + x^4 = y^2) : (x, y) = (0, 1) ∨ (x, y) = (3, 11) := by
  sorry

end find_pairs_l66_66052


namespace range_of_a_l66_66220

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, |x + 3| + |x - 1| ≥ a^2 - 3 * a) ↔ -1 ≤ a ∧ a ≤ 4 :=
by
  sorry

end range_of_a_l66_66220


namespace find_circle_diameter_l66_66839

noncomputable def circle_diameter (AB CD : ℝ) (h_AB : AB = 16) (h_CD : CD = 4)
  (h_perp : ∃ M : ℝ → ℝ → Prop, M AB CD) : ℝ :=
  2 * 10

theorem find_circle_diameter (AB CD : ℝ)
  (h_AB : AB = 16)
  (h_CD : CD = 4)
  (h_perp : ∃ M : ℝ → ℝ → Prop, M AB CD) :
  circle_diameter AB CD h_AB h_CD h_perp = 20 := 
  by sorry

end find_circle_diameter_l66_66839


namespace min_value_of_f_l66_66168

def f (x y : ℝ) : ℝ := x^3 + y^3 + x^2 * y + x * y^2 - 3 * (x^2 + y^2 + x * y) + 3 * (x + y)

theorem min_value_of_f : ∀ x y : ℝ, x ≥ 1/2 → y ≥ 1/2 → f x y ≥ 1
    := by
      intros x y hx hy
      -- Rest of the proof would go here
      sorry

end min_value_of_f_l66_66168


namespace find_x2017_l66_66499

-- Define that f is an odd function
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = -f (-x)

-- Define that f is increasing
def is_increasing_function (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y
  
-- Define the arithmetic sequence
def arithmetic_sequence (a d : ℝ) (n : ℕ) : ℝ :=
  a + n * d

-- Main theorem
theorem find_x2017
  (f : ℝ → ℝ) (x : ℕ → ℝ)
  (Hodd : is_odd_function f)
  (Hinc : is_increasing_function f)
  (Hseq : ∀ n, x (n + 1) = x n + 2)
  (H7_8 : f (x 7) + f (x 8) = 0) :
  x 2017 = 4019 := 
sorry

end find_x2017_l66_66499


namespace sin_theta_value_l66_66484

open Real

noncomputable def sin_theta_sol (theta : ℝ) : ℝ :=
  (-5 + Real.sqrt 41) / 4

theorem sin_theta_value (theta : ℝ) (h1 : 5 * tan theta = 2 * cos theta) (h2 : 0 < theta) (h3 : theta < π) :
  sin theta = sin_theta_sol theta :=
by
  sorry

end sin_theta_value_l66_66484


namespace tan_beta_minus_2alpha_l66_66175

noncomputable def tan_alpha := 1 / 2
noncomputable def tan_beta_minus_alpha := 2 / 5
theorem tan_beta_minus_2alpha (α β : ℝ) (h1 : Real.tan α = tan_alpha) (h2 : Real.tan (β - α) = tan_beta_minus_alpha) :
  Real.tan (β - 2 * α) = -1 / 12 := 
by
  sorry

end tan_beta_minus_2alpha_l66_66175


namespace original_inhabitants_l66_66137

theorem original_inhabitants (X : ℝ) 
  (h1 : 10 ≤ X) 
  (h2 : 0.9 * X * 0.75 + 0.225 * X * 0.15 = 5265) : 
  X = 7425 := 
sorry

end original_inhabitants_l66_66137


namespace chocolate_more_expensive_l66_66702

variables (C P : ℝ)
theorem chocolate_more_expensive (h : 7 * C > 8 * P) : 8 * C > 9 * P :=
sorry

end chocolate_more_expensive_l66_66702


namespace max_min_z_diff_correct_l66_66890

noncomputable def max_min_z_diff (x y z : ℝ) (h1 : x + y + z = 3) (h2 : x^2 + y^2 + z^2 = 18) : ℝ :=
  6

theorem max_min_z_diff_correct (x y z : ℝ) (h1 : x + y + z = 3) (h2 : x^2 + y^2 + z^2 = 18) :
  max_min_z_diff x y z h1 h2 = 6 :=
sorry

end max_min_z_diff_correct_l66_66890


namespace correct_option_division_l66_66983

theorem correct_option_division (x : ℝ) : 
  (-6 * x^3) / (-2 * x^2) = 3 * x :=
by 
  sorry

end correct_option_division_l66_66983


namespace number_of_solutions_l66_66604

theorem number_of_solutions :
  (∃ (xs : List ℤ), (∀ x ∈ xs, |3 * x + 4| ≤ 10) ∧ xs.length = 7) := sorry

end number_of_solutions_l66_66604


namespace john_paid_more_than_jane_by_540_l66_66209

noncomputable def original_price : ℝ := 36.000000000000036
noncomputable def discount_percentage : ℝ := 0.10
noncomputable def tip_percentage : ℝ := 0.15

noncomputable def discounted_price : ℝ := original_price * (1 - discount_percentage)
noncomputable def john_tip : ℝ := original_price * tip_percentage
noncomputable def jane_tip : ℝ := discounted_price * tip_percentage

noncomputable def john_total_payment : ℝ := discounted_price + john_tip
noncomputable def jane_total_payment : ℝ := discounted_price + jane_tip

noncomputable def difference : ℝ := john_total_payment - jane_total_payment

theorem john_paid_more_than_jane_by_540 :
  difference = 0.5400000000000023 := sorry

end john_paid_more_than_jane_by_540_l66_66209


namespace not_exists_cube_in_sequence_l66_66262

-- Lean statement of the proof problem
theorem not_exists_cube_in_sequence : ∀ n : ℕ, ¬ ∃ k : ℤ, 2 ^ (2 ^ n) + 1 = k ^ 3 := 
by 
    intro n
    intro ⟨k, h⟩
    sorry

end not_exists_cube_in_sequence_l66_66262


namespace center_of_hyperbola_l66_66461

theorem center_of_hyperbola :
  ∃ (h k : ℝ), (h = 2 ∧ k = 4) ∧ (9 * (x - h)^2 - 16 * (y - k)^2 = 180) :=
  sorry

end center_of_hyperbola_l66_66461


namespace problem_proof_l66_66026

open Set

noncomputable def A : Set ℝ := {x | abs (4 * x - 1) < 9}
noncomputable def B : Set ℝ := {x | x / (x + 3) ≥ 0}
noncomputable def complement_A : Set ℝ := {x | x ≤ -2 ∨ x ≥ 5 / 2}
noncomputable def correct_answer : Set ℝ := Iio (-3) ∪ Ici (5 / 2)

theorem problem_proof : (compl A) ∩ B = correct_answer := 
  by
    sorry

end problem_proof_l66_66026


namespace ratio_of_frank_to_joystick_l66_66886

-- Define the costs involved
def cost_table : ℕ := 140
def cost_chair : ℕ := 100
def cost_joystick : ℕ := 20
def diff_spent : ℕ := 30

-- Define the payments
def F_j := 5
def E_j := 15

-- The ratio we need to prove
def ratio_frank_to_total_joystick (F_j : ℕ) (total_joystick : ℕ) : (ℕ × ℕ) :=
  (F_j / Nat.gcd F_j total_joystick, total_joystick / Nat.gcd F_j total_joystick)

theorem ratio_of_frank_to_joystick :
  let F_j := 5
  let total_joystick := 20
  ratio_frank_to_total_joystick F_j total_joystick = (1, 4) := by
  sorry

end ratio_of_frank_to_joystick_l66_66886


namespace bedroom_curtain_width_l66_66119

theorem bedroom_curtain_width
  (initial_fabric_area : ℕ)
  (living_room_curtain_area : ℕ)
  (fabric_left : ℕ)
  (bedroom_curtain_height : ℕ)
  (bedroom_curtain_area : ℕ)
  (bedroom_curtain_width : ℕ) :
  initial_fabric_area = 16 * 12 →
  living_room_curtain_area = 4 * 6 →
  fabric_left = 160 →
  bedroom_curtain_height = 4 →
  bedroom_curtain_area = 168 - 160 →
  bedroom_curtain_area = bedroom_curtain_width * bedroom_curtain_height →
  bedroom_curtain_width = 2 :=
by
  intros h1 h2 h3 h4 h5 h6
  -- Skipping the proof
  sorry

end bedroom_curtain_width_l66_66119


namespace op_two_four_l66_66911

def op (a b : ℝ) : ℝ := 5 * a + 2 * b

theorem op_two_four : op 2 4 = 18 := by
  sorry

end op_two_four_l66_66911


namespace sweater_markup_l66_66389

-- Conditions
variables (W R : ℝ)
axiom h1 : 0.40 * R = 1.20 * W

-- Theorem statement
theorem sweater_markup (W R : ℝ) (h1 : 0.40 * R = 1.20 * W) : (R - W) / W * 100 = 200 :=
sorry

end sweater_markup_l66_66389


namespace subset_singleton_zero_l66_66190

def X : Set ℤ := {x | -2 ≤ x ∧ x ≤ 2}

theorem subset_singleton_zero : {0} ⊆ X :=
by
  sorry

end subset_singleton_zero_l66_66190


namespace smallest_k_for_a_n_digital_l66_66406

theorem smallest_k_for_a_n_digital (a n : ℕ) (h : 10^2013 ≤ a^n ∧ a^n < 10^2014) : 
  ∀ k : ℕ, (∀ b : ℕ, 10^(k-1) ≤ b → b < 10^k → (¬(10^2013 ≤ b^n ∧ b^n < 10^2014))) ↔ k = 2014 :=
by 
  sorry

end smallest_k_for_a_n_digital_l66_66406


namespace pentagon_coloring_count_l66_66777

-- Define the three colors
inductive Color
| Red
| Yellow
| Green

open Color

-- Define the pentagon coloring problem
def adjacent_different (color1 color2 : Color) : Prop :=
color1 ≠ color2

-- Define a coloring for the pentagon
structure PentagonColoring :=
(A B C D E : Color)
(adjAB : adjacent_different A B)
(adjBC : adjacent_different B C)
(adjCD : adjacent_different C D)
(adjDE : adjacent_different D E)
(adjEA : adjacent_different E A)

-- The main statement to prove
theorem pentagon_coloring_count :
  ∃ (colorings : Finset PentagonColoring), colorings.card = 30 := sorry

end pentagon_coloring_count_l66_66777


namespace line_perpendicular_l66_66541

theorem line_perpendicular (m : ℝ) : 
  -- Conditions
  (∀ x y : ℝ, x - 2 * y + 5 = 0 → y = 1/2 * x + 5/2) →  -- Slope of the first line
  (∀ x y : ℝ, 2 * x + m * y - 6 = 0 → y = -2/m * x + 6/m) →  -- Slope of the second line
  -- Perpendicular condition
  ((1/2) * (-2/m) = -1) →
  -- Conclusion
  m = 1 := 
sorry

end line_perpendicular_l66_66541


namespace part1_part2_l66_66650

def f (x : ℝ) : ℝ := x^2 - 1

theorem part1 (m x : ℝ) (hx : 1 ≤ x ∧ x ≤ 2) (ineq : 4 * m^2 * |f x| + 4 * f m ≤ |f (x-1)|) : 
    -1/2 ≤ m ∧ m ≤ 1/2 := 
sorry

theorem part2 (x1 : ℝ) (hx1 : 1 ≤ x1 ∧ x1 ≤ 2) : 
    (∃ x2 : ℝ, 1 ≤ x2 ∧ x2 ≤ 2 ∧ f x1 = |2 * f x2 - a * x2|) →
    (0 ≤ a ∧ a ≤ 3/2 ∨ a = 3) := 
sorry

end part1_part2_l66_66650


namespace measure_of_AED_l66_66381

-- Importing the necessary modules for handling angles and geometry
variables {A B C D E : Type}
noncomputable def angle (p q r : Type) : ℝ := sorry -- Definition to represent angles in general

-- Given conditions
variables
  (hD_on_AC : D ∈ line_segment A C)
  (hE_on_BC : E ∈ line_segment B C)
  (h_angle_ABD : angle A B D = 30)
  (h_angle_BAE : angle B A E = 60)
  (h_angle_CAE : angle C A E = 20)
  (h_angle_CBD : angle C B D = 30)

-- The goal to prove
theorem measure_of_AED :
  angle A E D = 20 :=
by
  -- Proof details will go here
  sorry

end measure_of_AED_l66_66381


namespace zucchini_pounds_l66_66938

theorem zucchini_pounds :
  let eggplants_pounds := 5
  let eggplants_cost_per_pound := 2.00
  let tomatoes_pounds := 4
  let tomatoes_cost_per_pound := 3.50
  let onions_pounds := 3
  let onions_cost_per_pound := 1.00
  let basil_pounds := 1
  let basil_cost_per_half_pound := 2.50
  let quarts := 4
  let cost_per_quart := 10.00
  let total_cost := quarts * cost_per_quart
  let cost_of_eggplants := eggplants_pounds * eggplants_cost_per_pound
  let cost_of_tomatoes := tomatoes_pounds * tomatoes_cost_per_pound
  let cost_of_onions := onions_pounds * onions_cost_per_pound
  let cost_of_basil := basil_pounds * (basil_cost_per_half_pound * 2)
  let other_ingredients_cost := cost_of_eggplants + cost_of_tomatoes + cost_of_onions + cost_of_basil
  let cost_of_zucchini := total_cost - other_ingredients_cost
  let zucchini_cost_per_pound := 2.00
  let pounds_of_zucchini := cost_of_zucchini / zucchini_cost_per_pound
  pounds_of_zucchini = 4 :=
by
  sorry

end zucchini_pounds_l66_66938


namespace Glenn_total_expenditure_l66_66714

-- Define initial costs and discounts
def ticket_cost_Monday : ℕ := 5
def ticket_cost_Wednesday : ℕ := 2 * ticket_cost_Monday
def ticket_cost_Saturday : ℕ := 5 * ticket_cost_Monday
def discount_Wednesday (cost : ℕ) : ℕ := cost * 90 / 100
def additional_expense_Saturday : ℕ := 7

-- Define number of attendees
def attendees_Wednesday : ℕ := 4
def attendees_Saturday : ℕ := 2

-- Calculate total costs
def total_cost_Wednesday : ℕ :=
  attendees_Wednesday * discount_Wednesday ticket_cost_Wednesday
def total_cost_Saturday : ℕ :=
  attendees_Saturday * ticket_cost_Saturday + additional_expense_Saturday

-- Calculate the total money spent by Glenn
def total_spent : ℕ :=
  total_cost_Wednesday + total_cost_Saturday

-- Combine all conditions and conclusions into proof statement
theorem Glenn_total_expenditure : total_spent = 93 := by
  sorry

end Glenn_total_expenditure_l66_66714


namespace least_positive_integer_mod_cond_l66_66357

theorem least_positive_integer_mod_cond (N : ℕ) :
  (N % 6 = 5) ∧ 
  (N % 7 = 6) ∧ 
  (N % 8 = 7) ∧ 
  (N % 9 = 8) ∧ 
  (N % 10 = 9) ∧ 
  (N % 11 = 10) →
  N = 27719 :=
by
  sorry

end least_positive_integer_mod_cond_l66_66357


namespace total_dots_not_visible_l66_66333

noncomputable def total_dots_on_die : ℕ := 1 + 2 + 3 + 4 + 5 + 6
noncomputable def total_dice : ℕ := 3
noncomputable def total_visible_faces : ℕ := 5

def visible_faces : List ℕ := [1, 2, 3, 3, 4]

theorem total_dots_not_visible :
  (total_dots_on_die * total_dice) - (visible_faces.sum) = 50 := by
  sorry

end total_dots_not_visible_l66_66333


namespace range_of_a_l66_66340

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x < - (Real.sqrt 3) / 3 ∨ x > (Real.sqrt 3) / 3 →
    a * (3 * x^2 - 1) > 0) →
  a > 0 :=
by
  sorry

end range_of_a_l66_66340


namespace original_price_of_iWatch_l66_66482

theorem original_price_of_iWatch (P : ℝ) (h1 : 800 > 0) (h2 : P > 0)
    (h3 : 680 + 0.90 * P > 0) (h4 : 0.98 * (680 + 0.90 * P) = 931) :
    P = 300 := by
  sorry

end original_price_of_iWatch_l66_66482


namespace lcm_inequality_l66_66935

theorem lcm_inequality
  (a b c d e : ℤ)
  (h1 : 1 ≤ a)
  (h2 : a < b)
  (h3 : b < c)
  (h4 : c < d)
  (h5 : d < e) :
  (1 : ℚ) / Int.lcm a b + (1 : ℚ) / Int.lcm b c + 
  (1 : ℚ) / Int.lcm c d + (1 : ℚ) / Int.lcm d e ≤ (15 : ℚ) / 16 := by
  sorry

end lcm_inequality_l66_66935


namespace perpendicular_vectors_l66_66990

/-- If vectors a = (1, 2) and b = (x, 4) are perpendicular, then x = -8. -/
theorem perpendicular_vectors (x : ℝ) (a b : ℝ × ℝ) 
  (ha : a = (1, 2)) (hb : b = (x, 4)) (h_perp : a.1 * b.1 + a.2 * b.2 = 0) : x = -8 :=
by {
  sorry
}

end perpendicular_vectors_l66_66990


namespace star_7_3_eq_neg_5_l66_66021

def star_operation (a b : ℤ) : ℤ := 4 * a + 3 * b - 2 * a * b

theorem star_7_3_eq_neg_5 : star_operation 7 3 = -5 :=
by
  -- proof goes here
  sorry

end star_7_3_eq_neg_5_l66_66021


namespace john_has_22_quarters_l66_66212

-- Definitions based on conditions
def number_of_quarters (Q : ℕ) : ℕ := Q
def number_of_dimes (Q : ℕ) : ℕ := Q + 3
def number_of_nickels (Q : ℕ) : ℕ := Q - 6

-- Total number of coins condition
def total_number_of_coins (Q : ℕ) : Prop := 
  (number_of_quarters Q) + (number_of_dimes Q) + (number_of_nickels Q) = 63

-- Goal: Proving the number of quarters is 22
theorem john_has_22_quarters : ∃ Q : ℕ, total_number_of_coins Q ∧ Q = 22 :=
by
  -- Proof skipped 
  sorry

end john_has_22_quarters_l66_66212


namespace last_donation_on_saturday_l66_66502

def total_amount : ℕ := 2010
def daily_donation : ℕ := 10
def first_day_donation : ℕ := 0 -- where 0 represents Monday, 6 represents Sunday

def total_days : ℕ := total_amount / daily_donation

def last_donation_day_of_week : ℕ := (total_days % 7 + first_day_donation) % 7

theorem last_donation_on_saturday : last_donation_day_of_week = 5 := by
  -- Prove it by calculation
  sorry

end last_donation_on_saturday_l66_66502


namespace intersection_S_T_l66_66509

def S := {x : ℝ | (x - 2) * (x - 3) ≥ 0}
def T := {x : ℝ | x > 0}

theorem intersection_S_T :
  (S ∩ T) = (Set.Ioc 0 2 ∪ Set.Ici 3) :=
by
  sorry

end intersection_S_T_l66_66509


namespace cycle_cost_price_l66_66912

theorem cycle_cost_price (SP : ℝ) (loss_percentage : ℝ) (C : ℝ) 
  (h1 : SP = 1360) 
  (h2 : loss_percentage = 0.15) :
  SP = (1 - loss_percentage) * C → C = 1600 :=
by
  sorry

end cycle_cost_price_l66_66912


namespace problem_statement_l66_66847

theorem problem_statement :
  ∃ (a b c : ℕ), gcd a (gcd b c) = 1 ∧
  (∃ x y : ℝ, 2 * y = 8 * x - 7) ∧
  a ^ 2 + b ^ 2 + (c:ℤ) ^ 2 = 117 :=
sorry

end problem_statement_l66_66847


namespace calculate_fixed_payment_calculate_variable_payment_compare_plans_for_x_eq_30_l66_66625

noncomputable def cost_plan1_fixed (num_suits num_ties : ℕ) : ℕ :=
  if num_ties > num_suits then 200 * num_suits + 40 * (num_ties - num_suits)
  else 200 * num_suits

noncomputable def cost_plan2_fixed (num_suits num_ties : ℕ) : ℕ :=
  (200 * num_suits + 40 * num_ties) * 9 / 10

noncomputable def cost_plan1_variable (num_suits : ℕ) (x : ℕ) : ℕ :=
  200 * num_suits + 40 * (x - num_suits)

noncomputable def cost_plan2_variable (num_suits : ℕ) (x : ℕ) : ℕ :=
  (200 * num_suits + 40 * x) * 9 / 10

theorem calculate_fixed_payment :
  cost_plan1_fixed 20 22 = 4080 ∧ cost_plan2_fixed 20 22 = 4392 :=
by sorry

theorem calculate_variable_payment (x : ℕ) (hx : x > 20) :
  cost_plan1_variable 20 x = 40 * x + 3200 ∧ cost_plan2_variable 20 x = 36 * x + 3600 :=
by sorry

theorem compare_plans_for_x_eq_30 :
  cost_plan1_variable 20 30 < cost_plan2_variable 20 30 :=
by sorry


end calculate_fixed_payment_calculate_variable_payment_compare_plans_for_x_eq_30_l66_66625


namespace base_number_is_4_l66_66792

theorem base_number_is_4 (some_number : ℕ) (h : 16^8 = some_number^16) : some_number = 4 :=
sorry

end base_number_is_4_l66_66792


namespace calculate_s_at_2_l66_66400

-- Given definitions
def t (x : ℝ) : ℝ := 2 * x^2 - 5 * x + 1
def s (p : ℝ) : ℝ := p^3 - 4 * p^2 + p + 6

-- The target statement
theorem calculate_s_at_2 : s 2 = ((5 + Real.sqrt 33) / 4)^3 - 4 * ((5 + Real.sqrt 33) / 4)^2 + ((5 + Real.sqrt 33) / 4) + 6 := 
by 
  sorry

end calculate_s_at_2_l66_66400


namespace javier_time_outlining_l66_66410

variable (O : ℕ)
variable (W : ℕ := O + 28)
variable (P : ℕ := (O + 28) / 2)
variable (total_time : ℕ := O + W + P)

theorem javier_time_outlining
  (h1 : total_time = 117)
  (h2 : W = O + 28)
  (h3 : P = (O + 28) / 2)
  : O = 30 := by 
  sorry

end javier_time_outlining_l66_66410


namespace find_expression_for_x_l66_66852

variable (x : ℝ) (hx : x^3 + (1 / x^3) = -52)

theorem find_expression_for_x : x + (1 / x) = -4 :=
by sorry

end find_expression_for_x_l66_66852


namespace Gwen_walking_and_elevation_gain_l66_66185

theorem Gwen_walking_and_elevation_gain :
  ∀ (jogging_time walking_time total_time elevation_gain : ℕ)
    (jogging_feet total_feet : ℤ),
    jogging_time = 15 ∧ jogging_feet = 500 ∧ (jogging_time + walking_time = total_time) ∧
    (5 * walking_time = 3 * jogging_time) ∧ (total_time * jogging_feet = 15 * total_feet)
    → walking_time = 9 ∧ total_feet = 800 := by 
  sorry

end Gwen_walking_and_elevation_gain_l66_66185


namespace train_trip_length_l66_66867

theorem train_trip_length (v D : ℝ) :
  (3 + (3 * D - 6 * v) / (2 * v) = 4 + D / v) ∧ 
  (2.5 + 120 / v + (6 * D - 12 * v - 720) / (5 * v) = 3.5 + D / v) →
  (D = 420 ∨ D = 480 ∨ D = 540 ∨ D = 600 ∨ D = 660) :=
by
  sorry

end train_trip_length_l66_66867


namespace area_between_curves_l66_66467

-- Function definitions:
def quartic (a b c d e x : ℝ) : ℝ := a * x^4 + b * x^3 + c * x^2 + d * x + e
def line (p q x : ℝ) : ℝ := p * x + q

-- Conditions:
variables (a b c d e p q α β : ℝ)
variable (a_ne_zero : a ≠ 0)
variable (α_lt_β : α < β)
variable (touch_at_α : quartic a b c d e α = line p q α ∧ deriv (quartic a b c d e) α = p)
variable (touch_at_β : quartic a b c d e β = line p q β ∧ deriv (quartic a b c d e) β = p)

-- Theorem:
theorem area_between_curves :
  ∫ x in α..β, |quartic a b c d e x - line p q x| = (a * (β - α)^5) / 30 :=
by sorry

end area_between_curves_l66_66467


namespace quadratic_function_properties_l66_66057

theorem quadratic_function_properties
    (f : ℝ → ℝ)
    (h_vertex : ∀ x, f x = -(x - 2)^2 + 1)
    (h_point : f (-1) = -8) :
  (∀ x, f x = -(x - 2)^2 + 1) ∧
  (f 1 = 0) ∧ (f 3 = 0) ∧ (f 0 = 1) :=
  by
    sorry

end quadratic_function_properties_l66_66057


namespace unique_integer_solution_l66_66463

theorem unique_integer_solution (x y z : ℤ) (h : 2 * x^2 + 3 * y^2 = z^2) : x = 0 ∧ y = 0 ∧ z = 0 :=
by {
  sorry
}

end unique_integer_solution_l66_66463


namespace at_most_n_maximum_distance_pairs_l66_66555

theorem at_most_n_maximum_distance_pairs (n : ℕ) (h : n > 2) 
(points : Fin n → ℝ × ℝ) :
  ∃ (maxDistPairs : Finset (Fin n × Fin n)), (maxDistPairs.card ≤ n) ∧ 
  ∀ (p1 p2 : Fin n), (p1, p2) ∈ maxDistPairs → 
  (∀ (q1 q2 : Fin n), dist (points q1) (points q2) ≤ dist (points p1) (points p2)) :=
sorry

end at_most_n_maximum_distance_pairs_l66_66555


namespace four_digit_positive_integers_count_l66_66531

theorem four_digit_positive_integers_count :
  let p := 17
  let a := 4582 % p
  let b := 902 % p
  let c := 2345 % p
  ∃ (n : ℕ), 
    (1000 ≤ 14 + p * n ∧ 14 + p * n ≤ 9999) ∧ 
    (4582 * (14 + p * n) + 902 ≡ 2345 [MOD p]) ∧ 
    n = 530 := sorry

end four_digit_positive_integers_count_l66_66531


namespace amelia_distance_l66_66363

theorem amelia_distance (total_distance amelia_monday_distance amelia_tuesday_distance : ℕ) 
  (h1 : total_distance = 8205) 
  (h2 : amelia_monday_distance = 907) 
  (h3 : amelia_tuesday_distance = 582) : 
  total_distance - (amelia_monday_distance + amelia_tuesday_distance) = 6716 := 
by 
  sorry

end amelia_distance_l66_66363


namespace problem_solved_prob_l66_66955

theorem problem_solved_prob (pA pB : ℝ) (HA : pA = 1 / 3) (HB : pB = 4 / 5) :
  ((1 - (1 - pA) * (1 - pB)) = 13 / 15) :=
by
  sorry

end problem_solved_prob_l66_66955


namespace country_x_income_l66_66050

variable (income : ℝ)
variable (tax_paid : ℝ)
variable (income_first_40000_tax : ℝ := 40000 * 0.1)
variable (income_above_40000_tax_rate : ℝ := 0.2)
variable (total_tax_paid : ℝ := 8000)
variable (income_above_40000 : ℝ := (total_tax_paid - income_first_40000_tax) / income_above_40000_tax_rate)

theorem country_x_income : 
  income = 40000 + income_above_40000 → 
  total_tax_paid = tax_paid → 
  tax_paid = income_first_40000_tax + (income_above_40000 * income_above_40000_tax_rate) →
  income = 60000 :=
by sorry

end country_x_income_l66_66050


namespace mikes_age_is_18_l66_66013

-- Define variables for Mike's age (m) and his uncle's age (u)
variables (m u : ℕ)

-- Condition 1: Mike is 18 years younger than his uncle
def condition1 : Prop := m = u - 18

-- Condition 2: The sum of their ages is 54 years
def condition2 : Prop := m + u = 54

-- Statement: Prove that Mike's age is 18 given the conditions
theorem mikes_age_is_18 (h1 : condition1 m u) (h2 : condition2 m u) : m = 18 :=
by
  -- Proof skipped with sorry
  sorry

end mikes_age_is_18_l66_66013


namespace ski_boat_rental_cost_per_hour_l66_66516

-- Let the cost per hour to rent a ski boat be x dollars
variable (x : ℝ)

-- Conditions
def cost_sailboat : ℝ := 60
def duration : ℝ := 3 * 2 -- 3 hours a day for 2 days
def cost_ken : ℝ := cost_sailboat * 2 -- Ken's total cost
def additional_cost : ℝ := 120
def cost_aldrich : ℝ := cost_ken + additional_cost -- Aldrich's total cost

-- Statement to prove
theorem ski_boat_rental_cost_per_hour (h : (duration * x = cost_aldrich)) : x = 40 := by
  sorry

end ski_boat_rental_cost_per_hour_l66_66516


namespace sum_of_three_numbers_l66_66535

theorem sum_of_three_numbers (a b c : ℕ) (h1 : b = 10)
                            (h2 : (a + b + c) / 3 = a + 15)
                            (h3 : (a + b + c) / 3 = c - 25) :
                            a + b + c = 60 :=
sorry

end sum_of_three_numbers_l66_66535


namespace train_speed_is_72_l66_66731

def distance : ℕ := 24
def time_minutes : ℕ := 20
def time_hours : ℚ := time_minutes / 60
def speed := distance / time_hours

theorem train_speed_is_72 :
  speed = 72 := by
  sorry

end train_speed_is_72_l66_66731


namespace each_person_eats_3_Smores_l66_66476

-- Definitions based on the conditions in (a)
def people := 8
def cost_per_4_Smores := 3
def total_cost := 18

-- The statement we need to prove
theorem each_person_eats_3_Smores (h1 : total_cost = people * (cost_per_4_Smores * 4 / 3)) :
  (total_cost / cost_per_4_Smores) * 4 / people = 3 :=
by
  sorry

end each_person_eats_3_Smores_l66_66476


namespace product_of_squares_of_consecutive_even_integers_l66_66229

theorem product_of_squares_of_consecutive_even_integers :
  ∃ (a : ℤ), (a - 2) * a * (a + 2) = 36 * a ∧ (a > 0) ∧ (a % 2 = 0) ∧
  ((a - 2)^2 * a^2 * (a + 2)^2) = 36864 :=
by
  sorry

end product_of_squares_of_consecutive_even_integers_l66_66229


namespace solve_for_x_l66_66701

noncomputable def log_b (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem solve_for_x (b x : ℝ) (hb : b > 1) (hx : x > 0) :
  (4 * x) ^ log_b b 4 - (5 * x) ^ log_b b 5 + x = 0 ↔ x = 1 :=
by
  -- Proof placeholder
  sorry

end solve_for_x_l66_66701


namespace least_positive_integer_condition_l66_66365

theorem least_positive_integer_condition
  (a : ℤ) (ha1 : a % 4 = 1) (ha2 : a % 5 = 2) (ha3 : a % 6 = 3) :
  a > 0 → a = 57 :=
by
  intro ha_pos
  -- Proof omitted for brevity
  sorry

end least_positive_integer_condition_l66_66365


namespace slope_of_line_l66_66605

theorem slope_of_line : ∀ (x y : ℝ), (x / 4 - y / 3 = 1) → ((3 * x / 4) - 3) = 0 → (y = (3 / 4) * x - 3) :=
by 
  intros x y h_eq h_slope 
  sorry

end slope_of_line_l66_66605


namespace hexagonal_tiles_in_box_l66_66842

theorem hexagonal_tiles_in_box :
  ∃ a b c : ℕ, a + b + c = 35 ∧ 3 * a + 4 * b + 6 * c = 128 ∧ c = 6 :=
by
  sorry

end hexagonal_tiles_in_box_l66_66842


namespace min_value_of_f_l66_66556

noncomputable def f (x : ℝ) := x + 2 * Real.cos x

theorem min_value_of_f :
  ∀ (x : ℝ), -Real.pi / 2 ≤ x ∧ x ≤ 0 → f x ≥ f (-Real.pi / 2) :=
by
  intro x hx
  -- conditions are given, statement declared, but proof is not provided
  sorry

end min_value_of_f_l66_66556


namespace cosine_of_angle_in_third_quadrant_l66_66616

theorem cosine_of_angle_in_third_quadrant (B : ℝ) (hB : B ∈ Set.Ioo (π : ℝ) (3 * π / 2)) (hSinB : Real.sin B = -5 / 13) :
  Real.cos B = -12 / 13 :=
sorry

end cosine_of_angle_in_third_quadrant_l66_66616


namespace roots_of_abs_exp_eq_b_l66_66869

theorem roots_of_abs_exp_eq_b (b : ℝ) (h : 0 < b ∧ b < 1) : 
  ∃! (x1 x2 : ℝ), x1 ≠ x2 ∧ abs (2^x1 - 1) = b ∧ abs (2^x2 - 1) = b :=
sorry

end roots_of_abs_exp_eq_b_l66_66869


namespace problem1_problem2_l66_66424

-- Problem 1: Simplification and Evaluation
theorem problem1 (x : ℝ) : (x = -3) → 
  ((x^2 - 6*x + 9) / (x^2 - 1)) / ((x^2 - 3*x) / (x + 1))
  = -1 / 2 := sorry

-- Problem 2: Solving the Equation
theorem problem2 (x : ℝ) : 
  (∀ y, (y = x) → 
    (y / (y + 1) = 2*y / (3*y + 3) - 1)) → x = -3 / 4 := sorry

end problem1_problem2_l66_66424


namespace Darla_electricity_bill_l66_66428

theorem Darla_electricity_bill :
  let tier1_rate := 4
  let tier2_rate := 3.5
  let tier3_rate := 3
  let tier1_limit := 300
  let tier2_limit := 500
  let late_fee1 := 150
  let late_fee2 := 200
  let late_fee3 := 250
  let consumption := 1200
  let cost_tier1 := tier1_limit * tier1_rate
  let cost_ttier2 := tier2_limit * tier2_rate
  let cost_tier3 := (consumption - (tier1_limit + tier2_limit)) * tier3_rate
  let total_cost := cost_tier1 + cost_tier2 + cost_tier3
  let late_fee := late_fee3
  let final_cost := total_cost + late_fee
  final_cost = 4400 :=
by
  sorry

end Darla_electricity_bill_l66_66428


namespace fraction_of_smaller_jar_l66_66266

theorem fraction_of_smaller_jar (S L : ℝ) (W : ℝ) (F : ℝ) 
  (h1 : W = F * S) 
  (h2 : W = 1/2 * L) 
  (h3 : 2 * W = 2/3 * L) 
  (h4 : S = 2/3 * L) :
  F = 3 / 4 :=
by
  sorry

end fraction_of_smaller_jar_l66_66266


namespace highest_power_of_3_dividing_N_is_1_l66_66099

-- Define the integer N as described in the problem
def N : ℕ := 313233515253

-- State the problem
theorem highest_power_of_3_dividing_N_is_1 : ∃ k : ℕ, (3^k ∣ N) ∧ ∀ m > 1, ¬ (3^m ∣ N) ∧ k = 1 :=
by
  -- Specific solution details and steps are not required here
  sorry

end highest_power_of_3_dividing_N_is_1_l66_66099


namespace shifted_graph_sum_l66_66097

noncomputable def f (x : ℝ) : ℝ := 3 * x ^ 2 - 2 * x + 5

def shift_right (f : ℝ → ℝ) (h : ℝ) (x : ℝ) : ℝ := f (x - h)
def shift_up (f : ℝ → ℝ) (k : ℝ) (x : ℝ) : ℝ := f x + k

noncomputable def g (x : ℝ) : ℝ := shift_up (shift_right f 7) 3 x

theorem shifted_graph_sum : (∃ (a b c : ℝ), g x = a * x ^ 2 + b * x + c ∧ (a + b + c = 128)) :=
by
  sorry

end shifted_graph_sum_l66_66097


namespace data_division_into_groups_l66_66986

-- Conditions
def data_set_size : Nat := 90
def max_value : Nat := 141
def min_value : Nat := 40
def class_width : Nat := 10

-- Proof statement
theorem data_division_into_groups : (max_value - min_value) / class_width + 1 = 11 :=
by
  sorry

end data_division_into_groups_l66_66986


namespace percentage_material_B_new_mixture_l66_66270

theorem percentage_material_B_new_mixture :
  let mixtureA := 8 -- kg of Mixture A
  let addOil := 2 -- kg of additional oil
  let addMixA := 6 -- kg of additional Mixture A
  let oil_percent := 0.20 -- 20% oil in Mixture A
  let materialB_percent := 0.80 -- 80% material B in Mixture A

  -- Initial amounts in 8 kg of Mixture A
  let initial_oil := oil_percent * mixtureA
  let initial_materialB := materialB_percent * mixtureA

  -- New mixture after adding 2 kg oil
  let new_oil := initial_oil + addOil
  let new_materialB := initial_materialB

  -- Adding 6 kg of Mixture A
  let added_oil := oil_percent * addMixA
  let added_materialB := materialB_percent * addMixA

  -- Total amounts in the new mixture
  let total_oil := new_oil + added_oil
  let total_materialB := new_materialB + added_materialB
  let total_weight := mixtureA + addOil + addMixA

  -- Percent calculation
  let percent_materialB := (total_materialB / total_weight) * 100

  percent_materialB = 70 := sorry

end percentage_material_B_new_mixture_l66_66270


namespace sum_first_nine_terms_arithmetic_sequence_l66_66672

theorem sum_first_nine_terms_arithmetic_sequence
  (a : ℕ → ℝ)
  (h_arith : ∀ n, a (n + 1) - a n = (a 2 - a 1))
  (h1 : a 1 + a 4 + a 7 = 45)
  (h2 : a 3 + a 6 + a 9 = 27) :
  (a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9) = 108 := 
sorry

end sum_first_nine_terms_arithmetic_sequence_l66_66672


namespace kanul_total_amount_l66_66643

variable (T : ℝ)
variable (H1 : 3000 + 2000 + 0.10 * T = T)

theorem kanul_total_amount : T = 5555.56 := 
by 
  /- with the conditions given, 
     we can proceed to prove T = 5555.56 -/
  sorry

end kanul_total_amount_l66_66643


namespace smallest_sum_of_cubes_two_ways_l66_66892

theorem smallest_sum_of_cubes_two_ways :
  ∃ (n : ℕ) (a b c d e f : ℕ),
  n = a^3 + b^3 + c^3 ∧ n = d^3 + e^3 + f^3 ∧
  (a, b, c) ≠ (d, e, f) ∧
  (d, e, f) ≠ (a, b, c) ∧ n = 251 :=
by
  sorry

end smallest_sum_of_cubes_two_ways_l66_66892


namespace triangle_angle_A_l66_66572

theorem triangle_angle_A (C : ℝ) (c : ℝ) (a : ℝ) 
  (hC : C = 45) (hc : c = Real.sqrt 2) (ha : a = Real.sqrt 3) :
  (∃ A : ℝ, A = 60 ∨ A = 120) :=
by
  sorry

end triangle_angle_A_l66_66572


namespace boys_difference_twice_girls_l66_66956

theorem boys_difference_twice_girls :
  ∀ (total_students girls boys : ℕ),
  total_students = 68 →
  girls = 28 →
  boys = total_students - girls →
  2 * girls - boys = 16 :=
by
  intros total_students girls boys h1 h2 h3
  sorry

end boys_difference_twice_girls_l66_66956


namespace arithmetic_geometric_proof_l66_66432

noncomputable def arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
∀ n, a (n + 1) = a n + d

noncomputable def geometric_sequence (b : ℕ → ℤ) (r : ℤ) : Prop :=
∀ n, b (n + 1) = b n * r

theorem arithmetic_geometric_proof
  (a : ℕ → ℤ) (b : ℕ → ℤ) (d r : ℤ)
  (h_arith : arithmetic_sequence a d)
  (h_geom : geometric_sequence b r)
  (h_cond1 : 3 * a 1 - a 8 * a 8 + 3 * a 15 = 0)
  (h_cond2 : a 8 = b 10):
  b 3 * b 17 = 36 :=
sorry

end arithmetic_geometric_proof_l66_66432


namespace calculation_l66_66494

theorem calculation : 
  let a := 20 / 9 
  let b := -53 / 4 
  (⌈ a * ⌈ b ⌉ ⌉ - ⌊ a * ⌊ b ⌋ ⌋) = 4 :=
by
  sorry

end calculation_l66_66494


namespace red_balls_count_l66_66996

theorem red_balls_count (R W N_1 N_2 : ℕ) 
  (h1 : R - 2 * N_1 = 18) 
  (h2 : W = 3 * N_1) 
  (h3 : R - 5 * N_2 = 0) 
  (h4 : W - 3 * N_2 = 18)
  : R = 50 :=
sorry

end red_balls_count_l66_66996


namespace perp_a_beta_l66_66790

noncomputable def line : Type := sorry
noncomputable def plane : Type := sorry
noncomputable def Incident (l : line) (p : plane) : Prop := sorry
noncomputable def Perpendicular (l1 l2 : line) : Prop := sorry
noncomputable def Parallel (l1 l2 : line) : Prop := sorry

variables {α β : plane} {a AB : line}

-- Conditions extracted from the problem
axiom condition1 : Perpendicular α β
axiom condition2 : Incident AB β ∧ Incident AB α
axiom condition3 : Parallel a α
axiom condition4 : Perpendicular a AB

-- The statement that needs to be proved
theorem perp_a_beta : Perpendicular a β :=
  sorry

end perp_a_beta_l66_66790


namespace pie_eating_contest_difference_l66_66310

-- Definition of given conditions
def num_students := 8
def emma_pies := 8
def sam_pies := 1

-- Statement to prove
theorem pie_eating_contest_difference :
  emma_pies - sam_pies = 7 :=
by
  -- Omitting the proof, as requested.
  sorry

end pie_eating_contest_difference_l66_66310


namespace split_trout_equally_l66_66863

-- Definitions for conditions
def Total_trout : ℕ := 18
def People : ℕ := 2

-- Statement we need to prove
theorem split_trout_equally 
(H1 : Total_trout = 18)
(H2 : People = 2) : 
  (Total_trout / People = 9) :=
by
  sorry

end split_trout_equally_l66_66863


namespace gcd_result_is_two_l66_66739

theorem gcd_result_is_two
  (n m k j: ℕ) (hn : n > 0) (hm : m > 0) (hk : k > 0) (hj : j > 0) :
  Nat.gcd (Nat.gcd (16 * n) (20 * m)) (Nat.gcd (18 * k) (24 * j)) = 2 := 
by
  sorry

end gcd_result_is_two_l66_66739


namespace next_perfect_square_l66_66298

theorem next_perfect_square (x : ℤ) (h : ∃ k : ℤ, x = k^2) : ∃ z : ℤ, z = x + 2 * Int.sqrt x + 1 :=
by
  sorry

end next_perfect_square_l66_66298


namespace isosceles_triangle_circumradius_l66_66254

theorem isosceles_triangle_circumradius (b : ℝ) (s : ℝ) (R : ℝ) (hb : b = 6) (hs : s = 5) :
  R = 25 / 8 :=
by 
  sorry

end isosceles_triangle_circumradius_l66_66254


namespace coordinates_of_A_after_move_l66_66624

noncomputable def moved_coordinates (a : ℝ) : ℝ × ℝ :=
  let x := 2 * a - 9 + 5
  let y := 1 - 2 * a
  (x, y)

theorem coordinates_of_A_after_move (a : ℝ) (h : moved_coordinates a = (0, 1 - 2 * a)) :
  moved_coordinates 2 = (-5, -3) :=
by
  -- Proof omitted
  sorry

end coordinates_of_A_after_move_l66_66624


namespace fishing_tomorrow_l66_66235

-- Conditions
def every_day_fishers : Nat := 7
def every_other_day_fishers : Nat := 8
def every_three_days_fishers : Nat := 3
def yesterday_fishers : Nat := 12
def today_fishers : Nat := 10

-- Determine the number who will fish tomorrow
def fishers_tomorrow : Nat :=
  let every_day_tomorrow := every_day_fishers
  let every_three_day_tomorrow := every_three_days_fishers
  let every_other_day_yesterday := yesterday_fishers - every_day_fishers
  let every_other_day_tomorrow := every_other_day_fishers - every_other_day_yesterday
  every_day_tomorrow + every_three_day_tomorrow + every_other_day_tomorrow

theorem fishing_tomorrow : fishers_tomorrow = 15 :=
  by
    -- skipping the actual proof with sorry
    sorry

end fishing_tomorrow_l66_66235


namespace total_amount_sold_l66_66942

theorem total_amount_sold (metres_sold : ℕ) (loss_per_metre cost_price_per_metre : ℕ) 
  (h1 : metres_sold = 600) (h2 : loss_per_metre = 5) (h3 : cost_price_per_metre = 35) :
  (cost_price_per_metre - loss_per_metre) * metres_sold = 18000 :=
by
  sorry

end total_amount_sold_l66_66942


namespace number_of_snakes_l66_66305

-- Define the variables
variable (S : ℕ) -- Number of snakes

-- Define the cost constants
def cost_per_gecko := 15
def cost_per_iguana := 5
def cost_per_snake := 10

-- Define the number of each pet
def num_geckos := 3
def num_iguanas := 2

-- Define the yearly cost
def yearly_cost := 1140

-- Calculate the total monthly cost
def monthly_cost := num_geckos * cost_per_gecko + num_iguanas * cost_per_iguana + S * cost_per_snake

-- Calculate the total yearly cost
def total_yearly_cost := 12 * monthly_cost

-- Prove the number of snakes
theorem number_of_snakes : total_yearly_cost = yearly_cost → S = 4 := by
  sorry

end number_of_snakes_l66_66305


namespace sara_no_ingredients_pies_l66_66197

theorem sara_no_ingredients_pies:
  ∀ (total_pies : ℕ) (berries_pies : ℕ) (cream_pies : ℕ) (nuts_pies : ℕ) (coconut_pies : ℕ),
  total_pies = 60 →
  berries_pies = 1/3 * total_pies →
  cream_pies = 1/2 * total_pies →
  nuts_pies = 3/5 * total_pies →
  coconut_pies = 1/5 * total_pies →
  (total_pies - nuts_pies) = 24 :=
by
  intros total_pies berries_pies cream_pies nuts_pies coconut_pies ht hb hc hn hcoc
  sorry

end sara_no_ingredients_pies_l66_66197


namespace find_value_of_a_l66_66138

theorem find_value_of_a (a : ℝ) (h : (3 + a + 10) / 3 = 5) : a = 2 := 
by {
  sorry
}

end find_value_of_a_l66_66138


namespace completing_square_l66_66626

theorem completing_square (x : ℝ) : (x^2 - 2 * x = 2) → ((x - 1)^2 = 3) :=
by
  sorry

end completing_square_l66_66626


namespace sum_of_cubes_l66_66391

theorem sum_of_cubes (x y : ℝ) (hx : x + y = 10) (hxy : x * y = 12) : x^3 + y^3 = 640 := 
by
  sorry

end sum_of_cubes_l66_66391


namespace unused_square_is_teal_l66_66271

-- Define the set of colors
inductive Color
| Cyan
| Magenta
| Lime
| Purple
| Teal
| Silver
| Violet

open Color

-- Define the condition that Lime is opposite Purple in the cube
def opposite (a b : Color) : Prop :=
  (a = Lime ∧ b = Purple) ∨ (a = Purple ∧ b = Lime)

-- Define the problem: seven squares are colored and one color remains unused.
def seven_squares_set (hinge : List Color) : Prop :=
  hinge.length = 6 ∧ 
  opposite Lime Purple ∧
  Color.Cyan ∈ hinge ∧
  Color.Magenta ∈ hinge ∧ 
  Color.Lime ∈ hinge ∧ 
  Color.Purple ∈ hinge ∧ 
  Color.Teal ∈ hinge ∧ 
  Color.Silver ∈ hinge ∧ 
  Color.Violet ∈ hinge

theorem unused_square_is_teal :
  ∃ hinge : List Color, seven_squares_set hinge ∧ ¬ (Teal ∈ hinge) := 
by sorry

end unused_square_is_teal_l66_66271


namespace num_employees_excluding_manager_l66_66797

/-- 
If the average monthly salary of employees is Rs. 1500, 
and adding a manager with salary Rs. 14100 increases 
the average salary by Rs. 600, prove that the number 
of employees (excluding the manager) is 20.
-/
theorem num_employees_excluding_manager 
  (avg_salary : ℕ) 
  (manager_salary : ℕ) 
  (new_avg_increase : ℕ) : 
  (∃ n : ℕ, 
    avg_salary = 1500 ∧ 
    manager_salary = 14100 ∧ 
    new_avg_increase = 600 ∧ 
    n = 20) := 
sorry

end num_employees_excluding_manager_l66_66797


namespace problem1_problem2_l66_66525

-- Problem (1)
theorem problem1 (x : ℝ) : (2 * |x - 1| ≥ 1) ↔ (x ≤ 1/2 ∨ x ≥ 3/2) := sorry

-- Problem (2)
theorem problem2 (a : ℝ) (h : a > 0) : (∀ x : ℝ, |a * x - 1| + |a * x - a| ≥ 1) ↔ a ≥ 2 := sorry

end problem1_problem2_l66_66525


namespace nancy_initial_files_correct_l66_66811

-- Definitions based on the problem conditions
def initial_files (deleted_files : ℕ) (folder_count : ℕ) (files_per_folder : ℕ) : ℕ :=
  (folder_count * files_per_folder) + deleted_files

-- The proof statement
theorem nancy_initial_files_correct :
  initial_files 31 7 7 = 80 :=
by
  sorry

end nancy_initial_files_correct_l66_66811


namespace train_speed_l66_66312

theorem train_speed (length_train time_cross : ℝ)
  (h1 : length_train = 180)
  (h2 : time_cross = 9) : 
  (length_train / time_cross) * 3.6 = 72 :=
by
  -- This is just a placeholder proof. Replace with the actual proof.
  sorry

end train_speed_l66_66312


namespace polynomial_identity_l66_66322

theorem polynomial_identity (a_0 a_1 a_2 a_3 a_4 : ℝ) (x : ℝ) 
  (h : (2 * x + 1)^4 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4) : 
  a_0 - a_1 + a_2 - a_3 + a_4 = 1 :=
by
  sorry

end polynomial_identity_l66_66322


namespace jack_further_down_l66_66435

-- Define the conditions given in the problem
def flights_up := 3
def flights_down := 6
def steps_per_flight := 12
def height_per_step_in_inches := 8
def inches_per_foot := 12

-- Define the number of steps and height calculations
def steps_up := flights_up * steps_per_flight
def steps_down := flights_down * steps_per_flight
def net_steps_down := steps_down - steps_up
def net_height_down_in_inches := net_steps_down * height_per_step_in_inches
def net_height_down_in_feet := net_height_down_in_inches / inches_per_foot

-- The proof statement to be shown
theorem jack_further_down : net_height_down_in_feet = 24 := sorry

end jack_further_down_l66_66435


namespace min_value_expr_l66_66552

theorem min_value_expr (a : ℝ) (h₁ : 0 < a) (h₂ : a < 3) : 
  ∃ m : ℝ, (∀ x : ℝ, 0 < x → x < 3 → (1/x + 9/(3 - x)) ≥ m) ∧ m = 16 / 3 :=
sorry

end min_value_expr_l66_66552


namespace fido_reachable_area_l66_66776

theorem fido_reachable_area (r : ℝ) (a b : ℕ) (a_nonzero : a ≠ 0) (b_nonzero : b ≠ 0)
  (h_leash : ∃ (r : ℝ), r > 0) (h_fraction : (a : ℝ) / b * π = π) : a * b = 1 :=
by
  sorry

end fido_reachable_area_l66_66776


namespace min_groups_with_conditions_l66_66179

theorem min_groups_with_conditions (n a b m : ℕ) (h_n : n = 8) (h_a : a = 4) (h_b : b = 1) :
  m ≥ 2 :=
sorry

end min_groups_with_conditions_l66_66179


namespace find_exponent_l66_66419

theorem find_exponent (m x y a : ℝ) (h : y = m * x ^ a) (hx : x = 1 / 4) (hy : y = 1 / 2) : a = 1 / 2 :=
by
  sorry

end find_exponent_l66_66419


namespace tea_price_l66_66687

theorem tea_price 
  (x : ℝ)
  (total_cost_80kg_tea : ℝ := 80 * x)
  (total_cost_20kg_tea : ℝ := 20 * 20)
  (total_selling_price : ℝ := 1920)
  (profit_condition : 1.2 * (total_cost_80kg_tea + total_cost_20kg_tea) = total_selling_price) :
  x = 15 :=
by
  sorry

end tea_price_l66_66687


namespace dennis_total_cost_l66_66943

-- Define the cost of items and quantities
def cost_pants : ℝ := 110.0
def cost_socks : ℝ := 60.0
def quantity_pants : ℝ := 4
def quantity_socks : ℝ := 2
def discount_rate : ℝ := 0.30

-- Define the total costs before and after discount
def total_cost_pants_before_discount : ℝ := cost_pants * quantity_pants
def total_cost_socks_before_discount : ℝ := cost_socks * quantity_socks
def total_cost_before_discount : ℝ := total_cost_pants_before_discount + total_cost_socks_before_discount
def total_discount : ℝ := total_cost_before_discount * discount_rate
def total_cost_after_discount : ℝ := total_cost_before_discount - total_discount

-- Theorem asserting the total amount after discount
theorem dennis_total_cost : total_cost_after_discount = 392 := by 
  sorry

end dennis_total_cost_l66_66943


namespace recurring_fraction_division_l66_66652

-- Define the values
def x : ℚ := 8 / 11
def y : ℚ := 20 / 11

-- The theorem statement function to prove x / y = 2 / 5
theorem recurring_fraction_division :
  (x / y = (2 : ℚ) / 5) :=
by 
  -- Skip the proof
  sorry

end recurring_fraction_division_l66_66652


namespace ramu_profit_percent_l66_66177

noncomputable def profit_percent (purchase_price repair_cost selling_price : ℕ) : ℕ :=
  let total_cost := purchase_price + repair_cost
  let profit := selling_price - total_cost
  (profit * 100) / total_cost

theorem ramu_profit_percent :
  profit_percent 42000 13000 64900 = 18 := by
  sorry

end ramu_profit_percent_l66_66177


namespace probability_correct_l66_66871

noncomputable def probability_point_between_lines : ℝ :=
  let intersection_x_l := 4    -- x-intercept of line l
  let intersection_x_m := 3    -- x-intercept of line m
  let area_under_l := (1 / 2) * intersection_x_l * 8 -- area under line l
  let area_under_m := (1 / 2) * intersection_x_m * 9 -- area under line m
  let area_between := area_under_l - area_under_m    -- area between lines
  (area_between / area_under_l : ℝ)

theorem probability_correct : probability_point_between_lines = 0.16 :=
by
  simp only [probability_point_between_lines]
  sorry

end probability_correct_l66_66871


namespace units_digit_base_9_l66_66848

theorem units_digit_base_9 (a b : ℕ) (h1 : a = 3 * 9 + 5) (h2 : b = 4 * 9 + 7) : 
  ((a + b) % 9) = 3 := by
  sorry

end units_digit_base_9_l66_66848


namespace intersection_A_B_l66_66635

def A : Set ℝ := {x | abs x <= 1}

def B : Set ℝ := {y | ∃ x : ℝ, y = x^2}

theorem intersection_A_B :
  (A ∩ B) = {x | 0 ≤ x ∧ x ≤ 1} := sorry

end intersection_A_B_l66_66635


namespace solution_of_ab_l66_66922

theorem solution_of_ab (a b : ℝ) 
  (h1 : ∀ x : ℝ, (ax^2 + b > 0 ↔ x < -1/2 ∨ x > 1/3)) : 
  a * b = 24 := 
sorry

end solution_of_ab_l66_66922


namespace deposit_paid_l66_66948

variable (P : ℝ) (Deposit Remaining : ℝ)

-- Define the conditions
def deposit_condition : Prop := Deposit = 0.10 * P
def remaining_condition : Prop := Remaining = 0.90 * P
def remaining_amount_given : Prop := Remaining = 1170

-- The goal to prove: the deposit paid is $130
theorem deposit_paid (h₁ : deposit_condition P Deposit) (h₂ : remaining_condition P Remaining) (h₃ : remaining_amount_given Remaining) : 
  Deposit = 130 :=
  sorry

end deposit_paid_l66_66948


namespace product_of_differences_l66_66292

-- Define the context where x and y are real numbers
variables (x y : ℝ)

-- State the theorem to be proved
theorem product_of_differences (x y : ℝ) : 
  (-x + y) * (-x - y) = x^2 - y^2 :=
sorry

end product_of_differences_l66_66292


namespace like_terms_mn_l66_66152

theorem like_terms_mn (m n : ℤ) 
  (H1 : m - 2 = 3) 
  (H2 : n + 2 = 1) : 
  m * n = -5 := 
by
  sorry

end like_terms_mn_l66_66152


namespace area_excluding_hole_l66_66621

theorem area_excluding_hole (x : ℝ) : 
  (2 * x + 8) * (x + 6) - (2 * x - 2) * (x - 1) = 24 * x + 46 :=
by
  sorry

end area_excluding_hole_l66_66621


namespace interest_rate_l66_66267

-- Definitions based on given conditions
def SumLent : ℝ := 1500
def InterestTime : ℝ := 4
def InterestAmount : ℝ := SumLent - 1260

-- Main theorem to prove the interest rate r is 4%
theorem interest_rate (r : ℝ) : (InterestAmount = SumLent * r / 100 * InterestTime) → r = 4 :=
by
  sorry

end interest_rate_l66_66267


namespace temperature_at_tian_du_peak_height_of_mountain_peak_l66_66946

-- Problem 1: Temperature at the top of Tian Du Peak
theorem temperature_at_tian_du_peak
  (height : ℝ) (drop_rate : ℝ) (initial_temp : ℝ)
  (H : height = 1800) (D : drop_rate = 0.6) (I : initial_temp = 18) :
  (initial_temp - (height / 100 * drop_rate)) = 7.2 :=
by
  sorry

-- Problem 2: Height of the mountain peak
theorem height_of_mountain_peak
  (drop_rate : ℝ) (foot_temp top_temp : ℝ)
  (D : drop_rate = 0.6) (F : foot_temp = 10) (T : top_temp = -8) :
  (foot_temp - top_temp) / drop_rate * 100 = 3000 :=
by
  sorry

end temperature_at_tian_du_peak_height_of_mountain_peak_l66_66946


namespace range_f_x_le_neg_five_l66_66812

noncomputable def f (x : ℝ) : ℝ :=
if h : 0 < x then 2^x - 3 else
if h : x < 0 then 3 - 2^(-x) else 0

theorem range_f_x_le_neg_five :
  ∀ x : ℝ, f x ≤ -5 ↔ x ≤ -3 :=
by sorry

end range_f_x_le_neg_five_l66_66812


namespace jerry_total_cost_l66_66255

-- Definition of the costs and quantities
def cost_color : ℕ := 32
def cost_bw : ℕ := 27
def num_color : ℕ := 3
def num_bw : ℕ := 1

-- Definition of the total cost
def total_cost : ℕ := (cost_color * num_color) + (cost_bw * num_bw)

-- The theorem that needs to be proved
theorem jerry_total_cost : total_cost = 123 :=
by
  sorry

end jerry_total_cost_l66_66255


namespace smaller_side_of_rectangle_l66_66196

theorem smaller_side_of_rectangle (r : ℝ) (h1 : r = 42) 
                                   (h2 : ∀ L W : ℝ, L / W = 6 / 5 → 2 * (L + W) = 2 * π * r) : 
                                   ∃ W : ℝ, W = (210 * π) / 11 := 
by {
    sorry
}

end smaller_side_of_rectangle_l66_66196


namespace angle_b_is_acute_l66_66875

-- Definitions for angles being right, acute, and sum of angles in a triangle
def is_right_angle (θ : ℝ) : Prop := θ = 90
def is_acute_angle (θ : ℝ) : Prop := 0 < θ ∧ θ < 90
def angles_sum_to_180 (α β γ : ℝ) : Prop := α + β + γ = 180

-- Main theorem statement
theorem angle_b_is_acute {α β γ : ℝ} (hC : is_right_angle γ) (hSum : angles_sum_to_180 α β γ) : is_acute_angle β :=
by
  sorry

end angle_b_is_acute_l66_66875


namespace roots_cubic_reciprocal_sum_l66_66927

theorem roots_cubic_reciprocal_sum (a b c : ℝ) 
(h₁ : a + b + c = 12) (h₂ : a * b + b * c + c * a = 27) (h₃ : a * b * c = 18) :
  1 / a^3 + 1 / b^3 + 1 / c^3 = 13 / 24 :=
by
  sorry

end roots_cubic_reciprocal_sum_l66_66927


namespace probability_not_orange_not_white_l66_66918

theorem probability_not_orange_not_white (num_orange num_black num_white : ℕ)
    (h_orange : num_orange = 8) (h_black : num_black = 7) (h_white : num_white = 6) :
    (num_black : ℚ) / (num_orange + num_black + num_white : ℚ) = 1 / 3 :=
  by
    -- Solution will be here.
    sorry

end probability_not_orange_not_white_l66_66918


namespace max_profit_at_35_l66_66769

-- Define the conditions
def unit_purchase_price : ℝ := 20
def base_selling_price : ℝ := 30
def base_sales_volume : ℕ := 400
def price_increase_effect : ℝ := 1
def sales_volume_decrease_per_dollar : ℝ := 20

-- Define the profit function
def profit (x : ℝ) : ℝ := (x - unit_purchase_price) * (base_sales_volume - sales_volume_decrease_per_dollar * (x - base_selling_price))

-- Lean statement to prove that the selling price which maximizes the profit is 35
theorem max_profit_at_35 : ∃ x : ℝ, x = 35 ∧ ∀ y : ℝ, profit y ≤ profit 35 := 
  sorry

end max_profit_at_35_l66_66769


namespace thompson_class_average_l66_66473

theorem thompson_class_average
  (n : ℕ) (initial_avg : ℚ) (final_avg : ℚ) (bridget_index : ℕ) (first_n_score_sum : ℚ)
  (total_students : ℕ) (final_score_sum : ℚ)
  (h1 : n = 17) -- Number of students initially graded
  (h2 : initial_avg = 76) -- Average score of the first 17 students
  (h3 : final_avg = 78) -- Average score after adding Bridget's test
  (h4 : bridget_index = 18) -- Total number of students
  (h5 : total_students = 18) -- Total number of students
  (h6 : first_n_score_sum = n * initial_avg) -- Total score of the first 17 students
  (h7 : final_score_sum = total_students * final_avg) -- Total score of the 18 students):
  -- Bridget's score
  (bridgets_score : ℚ) :
  bridgets_score = final_score_sum - first_n_score_sum :=
sorry

end thompson_class_average_l66_66473


namespace value_of_M_l66_66409

theorem value_of_M (x y z M : ℚ) : 
  (x + y + z = 48) ∧ (x - 5 = M) ∧ (y + 9 = M) ∧ (z / 5 = M) → M = 52 / 7 :=
by
  sorry

end value_of_M_l66_66409


namespace polynomial_form_l66_66471

def is_even_poly (P : ℝ → ℝ) : Prop := 
  ∀ x, P x = P (-x)

theorem polynomial_form (P : ℝ → ℝ) (hP : ∀ a b c : ℝ, (a * b + b * c + c * a = 0) → 
  P (a - b) + P (b - c) + P (c - a) = 2 * P (a + b + c)) : 
  ∃ (a b : ℝ), ∀ x : ℝ, P x = a * x ^ 4 + b * x ^ 2 := 
  sorry

end polynomial_form_l66_66471


namespace find_four_digit_number_l66_66823

-- Definitions of the digit variables a, b, c, d, and their constraints.
def four_digit_expressions_meet_condition (abcd abc ab : ℕ) (a : ℕ) :=
  ∃ (b c d : ℕ), abcd = (1000 * a + 100 * b + 10 * c + d)
  ∧ abc = (100 * a + 10 * b + c)
  ∧ ab = (10 * a + b)
  ∧ abcd - abc - ab - a = 1787

-- Main statement to be proven.
theorem find_four_digit_number
: ∀ a b c d : ℕ, 
  four_digit_expressions_meet_condition (1000 * a + 100 * b + 10 * c + d) (100 * a + 10 * b + c) (10 * a + b) a
  → (a = 2 ∧ b = 0 ∧ ((c = 0 ∧ d = 9) ∨ (c = 1 ∧ d = 0))) :=
sorry

end find_four_digit_number_l66_66823


namespace min_value_of_xy_l66_66066

theorem min_value_of_xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_eq : 4 * x * y - x - 2 * y = 4) : 
  xy >= 2 :=
sorry

end min_value_of_xy_l66_66066


namespace greatest_prime_factor_f24_is_11_value_of_f12_l66_66843

def is_even (n : ℕ) : Prop := n % 2 = 0

def f (n : ℕ) : ℕ := (List.range' 2 ((n + 1) / 2)).map (λ x => 2 * x) |> List.prod

theorem greatest_prime_factor_f24_is_11 : 
  ¬ ∃ p, Prime p ∧ p ∣ f 24 ∧ p > 11 := 
  sorry

theorem value_of_f12 : f 12 = 46080 := 
  sorry

end greatest_prime_factor_f24_is_11_value_of_f12_l66_66843


namespace area_of_shaded_region_l66_66622

theorem area_of_shaded_region :
  let inner_square_side_length := 3
  let triangle_base := 2
  let triangle_height := 1
  let number_of_triangles := 8
  let area_inner_square := inner_square_side_length * inner_square_side_length
  let area_one_triangle := (1/2) * triangle_base * triangle_height
  let total_area_triangles := number_of_triangles * area_one_triangle
  let total_area_shaded := area_inner_square + total_area_triangles
  total_area_shaded = 17 :=
sorry

end area_of_shaded_region_l66_66622


namespace negate_exponential_inequality_l66_66260

theorem negate_exponential_inequality :
  ¬ (∀ x : ℝ, Real.exp x > x) ↔ ∃ x : ℝ, Real.exp x ≤ x :=
by
  sorry

end negate_exponential_inequality_l66_66260


namespace remainder_13_plus_x_l66_66344

theorem remainder_13_plus_x (x : ℕ) (h1 : 7 * x % 31 = 1) : (13 + x) % 31 = 22 := 
by
  sorry

end remainder_13_plus_x_l66_66344


namespace chess_tournament_participants_l66_66302

-- Define the number of grandmasters
variables (x : ℕ)

-- Define the number of masters as three times the number of grandmasters
def num_masters : ℕ := 3 * x

-- Condition on total points scored: Master's points is 1.2 times the Grandmaster's points
def points_condition (g m : ℕ) : Prop := m = 12 * g / 10

-- Proposition that the total number of participants is 12
theorem chess_tournament_participants (x_nonnegative: 0 < x) (g m : ℕ)
  (masters_points: points_condition g m) : 
  4 * x = 12 := 
sorry

end chess_tournament_participants_l66_66302


namespace intersecting_lines_sum_constant_l66_66385

theorem intersecting_lines_sum_constant
  (c d : ℝ)
  (h1 : 3 = (1 / 3) * 3 + c)
  (h2 : 3 = (1 / 3) * 3 + d) :
  c + d = 4 :=
by
  sorry

end intersecting_lines_sum_constant_l66_66385


namespace solve_for_x_and_y_l66_66411

theorem solve_for_x_and_y : 
  (∃ x y : ℝ, 0.65 * 900 = 0.40 * x ∧ 0.35 * 1200 = 0.25 * y) → 
  ∃ x y : ℝ, x + y = 3142.5 :=
by
  sorry

end solve_for_x_and_y_l66_66411


namespace range_of_a_in_circle_l66_66615

theorem range_of_a_in_circle (a : ℝ) : 
  ((1 - a)^2 + (1 + a)^2 < 4) ↔ (-1 < a ∧ a < 1) :=
by
  sorry

end range_of_a_in_circle_l66_66615


namespace necessary_but_not_sufficient_condition_l66_66904

-- Define the set A
def A := {x : ℝ | -1 < x ∧ x < 2}

-- Define the necessary but not sufficient condition
def necessary_condition (a : ℝ) : Prop := a ≥ 1

-- Define the proposition that needs to be proved
def proposition (a : ℝ) : Prop := ∀ x ∈ A, x^2 - a < 0

-- The proof statement
theorem necessary_but_not_sufficient_condition (a : ℝ) :
  necessary_condition a → ∃ x ∈ A, proposition a :=
sorry

end necessary_but_not_sufficient_condition_l66_66904


namespace sum_arithmetic_sequence_satisfies_conditions_l66_66865

theorem sum_arithmetic_sequence_satisfies_conditions :
  ∀ (a : ℕ → ℤ) (d : ℤ),
  (a 1 = 1) ∧ (d ≠ 0) ∧ ((a 3)^2 = (a 2) * (a 6)) →
  (6 * a 1 + (6 * 5 / 2) * d = -24) :=
by
  sorry

end sum_arithmetic_sequence_satisfies_conditions_l66_66865


namespace percent_only_cats_l66_66412

def total_students := 500
def total_cats := 120
def total_dogs := 200
def both_cats_and_dogs := 40
def only_cats := total_cats - both_cats_and_dogs

theorem percent_only_cats:
  (only_cats : ℕ) / (total_students : ℕ) * 100 = 16 := 
by 
  sorry

end percent_only_cats_l66_66412


namespace unique_positive_integers_exists_l66_66246

theorem unique_positive_integers_exists (p : ℕ) (hp_prime : Nat.Prime p) (hp_odd : p % 2 = 1) : 
  ∃! m n : ℕ, m^2 = n * (n + p) ∧ m = (p^2 - 1) / 2 ∧ n = (p - 1)^2 / 4 := by
  sorry

end unique_positive_integers_exists_l66_66246


namespace avg_licks_l66_66084

theorem avg_licks (Dan Michael Sam David Lance : ℕ) 
  (hDan : Dan = 58) 
  (hMichael : Michael = 63) 
  (hSam : Sam = 70) 
  (hDavid : David = 70) 
  (hLance : Lance = 39) : 
  (Dan + Michael + Sam + David + Lance) / 5 = 60 :=
by 
  sorry

end avg_licks_l66_66084


namespace largest_number_of_right_angles_in_convex_octagon_l66_66851

theorem largest_number_of_right_angles_in_convex_octagon : 
  ∀ (angles : Fin 8 → ℝ), 
  (∀ i, 0 < angles i ∧ angles i < 180) → 
  (angles 0 + angles 1 + angles 2 + angles 3 + angles 4 + angles 5 + angles 6 + angles 7 = 1080) → 
  ∃ k, k ≤ 6 ∧ (∀ i < 8, if angles i = 90 then k = 6 else true) := 
by 
  sorry

end largest_number_of_right_angles_in_convex_octagon_l66_66851


namespace carol_savings_l66_66413

theorem carol_savings (S : ℝ) (h1 : ∀ t : ℝ, t = S - (2/3) * S) (h2 : S + (S - (2/3) * S) = 1/4) : S = 3/16 :=
by {
  sorry
}

end carol_savings_l66_66413


namespace f_1986_eq_one_l66_66485

def f : ℕ → ℤ := sorry

axiom f_def (a b : ℕ) : f (a + b) = f a + f b - 2 * f (a * b) + 1
axiom f_one : f 1 = 1

theorem f_1986_eq_one : f 1986 = 1 :=
sorry

end f_1986_eq_one_l66_66485


namespace solve_phi_eq_l66_66578

noncomputable def φ := (1 + Real.sqrt 5) / 2
noncomputable def φ_hat := (1 - Real.sqrt 5) / 2
noncomputable def F : ℕ → ℤ
| n =>
  if n = 0 then 0
  else if n = 1 then 1
  else F (n - 1) + F (n - 2)

theorem solve_phi_eq (n : ℕ) :
  ∃ x y : ℤ, x * φ ^ (n + 1) + y * φ^n = 1 ∧ 
    x = (-1 : ℤ)^(n+1) * F n ∧ y = (-1 : ℤ)^n * F (n + 1) := by
  sorry

end solve_phi_eq_l66_66578


namespace number_composite_l66_66733

theorem number_composite (n : ℕ) : 
  n = 10^(2^1974 + 2^1000 - 1) + 1 →
  ∃ a b : ℕ, 1 < a ∧ a < n ∧ n = a * b :=
by sorry

end number_composite_l66_66733


namespace box_volume_of_pyramid_l66_66324

/-- A theorem to prove the volume of the smallest cube-shaped box that can house the given rectangular pyramid. -/
theorem box_volume_of_pyramid :
  (∀ (h l w : ℕ), h = 15 ∧ l = 8 ∧ w = 12 → (∀ (v : ℕ), v = (max h (max l w)) ^ 3 → v = 3375)) :=
by
  intros h l w h_condition v v_def
  sorry

end box_volume_of_pyramid_l66_66324


namespace part1_part2_part3_l66_66752

namespace Problem

-- Definitions and conditions for problem 1
def f (m x : ℝ) : ℝ := (m + 1) * x^2 - (m - 1) * x + (m - 1)

theorem part1 (m : ℝ) :
  (∀ x : ℝ, f m x < 0) ↔ m < -5/3 := sorry

-- Definitions and conditions for problem 2
theorem part2 (m : ℝ) (h : m < 0) :
  ((-1 < m ∧ m < 0) → ∀ x : ℝ, x ≤ 1 ∨ x ≥ 1 / (m + 1)) ∧
  (m = -1 → ∀ x : ℝ, x ≤ 1) ∧
  (m < -1 → ∀ x : ℝ, 1 / (m + 1) ≤ x ∧ x ≤ 1) := sorry

-- Definitions and conditions for problem 3
theorem part3 (m : ℝ) :
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 → f m x ≥ x^2 + 2 * x) ↔ m ≥ (2 * Real.sqrt 3) / 3 + 1 := sorry

end Problem

end part1_part2_part3_l66_66752


namespace inequality_solution_set_l66_66354

noncomputable def increasing_function (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x ≤ y → f x ≤ f y

theorem inequality_solution_set (f : ℝ → ℝ)
  (h_increasing : increasing_function f)
  (h_A : f 0 = -2)
  (h_B : f 3 = 2) :
  {x : ℝ | |f (x+1)| ≥ 2} = {x | x ≤ -1} ∪ {x | x ≥ 2} :=
sorry

end inequality_solution_set_l66_66354


namespace subset_of_intervals_l66_66521

def A (x : ℝ) := -2 ≤ x ∧ x ≤ 5
def B (m x : ℝ) := m + 1 ≤ x ∧ x ≤ 2 * m - 1
def is_subset_of (B A : ℝ → Prop) := ∀ x, B x → A x
def possible_values_m (m : ℝ) := m ≤ 3

theorem subset_of_intervals (m : ℝ) :
  is_subset_of (B m) A ↔ possible_values_m m := by
  sorry

end subset_of_intervals_l66_66521


namespace arithmetic_sequence_100th_term_l66_66459

-- Define the first term and the common difference
def first_term : ℕ := 3
def common_difference : ℕ := 7

-- Define the formula for the nth term of an arithmetic sequence
def nth_term (a1 d n : ℕ) : ℕ := a1 + (n - 1) * d

-- Theorem: The 100th term of the arithmetic sequence is 696.
theorem arithmetic_sequence_100th_term :
  nth_term first_term common_difference 100 = 696 :=
  sorry

end arithmetic_sequence_100th_term_l66_66459


namespace average_rainfall_per_hour_eq_l66_66926

-- Define the conditions
def february_days_non_leap_year : ℕ := 28
def hours_per_day : ℕ := 24
def total_rainfall_in_inches : ℕ := 280
def total_hours_in_february : ℕ := february_days_non_leap_year * hours_per_day

-- Define the goal
theorem average_rainfall_per_hour_eq :
  total_rainfall_in_inches / total_hours_in_february = 5 / 12 :=
sorry

end average_rainfall_per_hour_eq_l66_66926


namespace tan_identity_l66_66233

theorem tan_identity (α β γ : ℝ) (h : α + β + γ = 45 * π / 180) :
  (1 + Real.tan α) * (1 + Real.tan β) * (1 + Real.tan γ) / (1 + Real.tan α * Real.tan β * Real.tan γ) = 2 :=
by
  sorry

end tan_identity_l66_66233


namespace zinc_in_combined_mass_l66_66775

def mixture1_copper_zinc_ratio : ℕ × ℕ := (13, 7)
def mixture2_copper_zinc_ratio : ℕ × ℕ := (5, 3)
def mixture1_mass : ℝ := 100
def mixture2_mass : ℝ := 50

theorem zinc_in_combined_mass :
  let zinc1 := (mixture1_copper_zinc_ratio.2 : ℝ) / (mixture1_copper_zinc_ratio.1 + mixture1_copper_zinc_ratio.2) * mixture1_mass
  let zinc2 := (mixture2_copper_zinc_ratio.2 : ℝ) / (mixture2_copper_zinc_ratio.1 + mixture2_copper_zinc_ratio.2) * mixture2_mass
  zinc1 + zinc2 = 53.75 :=
by
  sorry

end zinc_in_combined_mass_l66_66775


namespace unique_a_values_l66_66327

theorem unique_a_values :
  ∃ a_values : Finset ℝ,
    (∀ a ∈ a_values, ∃ r s : ℤ, (r + s = -a) ∧ (r * s = 8 * a)) ∧ a_values.card = 4 :=
by
  sorry

end unique_a_values_l66_66327


namespace ashley_family_spent_30_l66_66483

def cost_of_child_ticket : ℝ := 4.25
def cost_of_adult_ticket : ℝ := cost_of_child_ticket + 3.25
def discount : ℝ := 2.00
def num_adult_tickets : ℕ := 2
def num_child_tickets : ℕ := 4

def total_cost : ℝ := num_adult_tickets * cost_of_adult_ticket + num_child_tickets * cost_of_child_ticket - discount

theorem ashley_family_spent_30 :
  total_cost = 30.00 :=
sorry

end ashley_family_spent_30_l66_66483


namespace no_integer_valued_function_l66_66698

theorem no_integer_valued_function (f : ℤ → ℤ) (h : ∀ (m n : ℤ), f (m + f n) = f m - n) : False :=
sorry

end no_integer_valued_function_l66_66698


namespace motorist_spent_on_petrol_l66_66563

def original_price_per_gallon : ℝ := 5.56
def reduction_percentage : ℝ := 0.10
def new_price_per_gallon := original_price_per_gallon - (0.10 * original_price_per_gallon)
def gallons_more_after_reduction : ℝ := 5

theorem motorist_spent_on_petrol (X : ℝ) 
  (h1 : new_price_per_gallon = original_price_per_gallon - (reduction_percentage * original_price_per_gallon))
  (h2 : (X / new_price_per_gallon) - (X / original_price_per_gallon) = gallons_more_after_reduction) :
  X = 250.22 :=
by
  sorry

end motorist_spent_on_petrol_l66_66563


namespace star_value_when_c_2_d_3_l66_66219

def star (c d : ℕ) : ℕ := c^3 + 3*c^2*d + 3*c*d^2 + d^3

theorem star_value_when_c_2_d_3 :
  star 2 3 = 125 :=
by
  sorry

end star_value_when_c_2_d_3_l66_66219


namespace jean_total_calories_l66_66036

-- Define the conditions
def pages_per_donut : ℕ := 2
def written_pages : ℕ := 12
def calories_per_donut : ℕ := 150

-- Define the question as a theorem
theorem jean_total_calories : (written_pages / pages_per_donut) * calories_per_donut = 900 := by
  sorry

end jean_total_calories_l66_66036


namespace fraction_equality_l66_66813

theorem fraction_equality (a b c : ℝ) (hc : c ≠ 0) (h : a / c = b / c) : a = b := 
by
  sorry

end fraction_equality_l66_66813


namespace cannot_bisect_abs_function_l66_66316

theorem cannot_bisect_abs_function 
  (f : ℝ → ℝ)
  (hf1 : ∀ x, f x = |x|) :
  ¬ (∃ a b, a < b ∧ f a * f b < 0) :=
by
  sorry

end cannot_bisect_abs_function_l66_66316


namespace largest_common_multiple_of_7_8_l66_66330

noncomputable def largest_common_multiple_of_7_8_sub_2 (n : ℕ) : ℕ :=
  if n <= 100 then n else 0

theorem largest_common_multiple_of_7_8 :
  ∃ x : ℕ, x <= 100 ∧ (x - 2) % Nat.lcm 7 8 = 0 ∧ x = 58 :=
by
  let x := 58
  use x
  have h1 : x <= 100 := by norm_num
  have h2 : (x - 2) % Nat.lcm 7 8 = 0 := by norm_num
  have h3 : x = 58 := by norm_num
  exact ⟨h1, h2, h3⟩

end largest_common_multiple_of_7_8_l66_66330


namespace which_calc_is_positive_l66_66046

theorem which_calc_is_positive :
  (-3 + 7 - 5 < 0) ∧
  ((1 - 2) * 3 < 0) ∧
  (-16 / (↑(-3)^2) < 0) ∧
  (-2^4 * (-6) > 0) :=
by
sorry

end which_calc_is_positive_l66_66046


namespace product_of_areas_eq_square_of_volume_l66_66481

theorem product_of_areas_eq_square_of_volume
    (a b c : ℝ)
    (bottom_area : ℝ) (side_area : ℝ) (front_area : ℝ)
    (volume : ℝ)
    (h1 : bottom_area = a * b)
    (h2 : side_area = b * c)
    (h3 : front_area = c * a)
    (h4 : volume = a * b * c) :
    bottom_area * side_area * front_area = volume ^ 2 := by
  -- proof omitted
  sorry

end product_of_areas_eq_square_of_volume_l66_66481


namespace fish_tagged_initially_l66_66497

theorem fish_tagged_initially (N T : ℕ) (hN : N = 1500) 
  (h_ratio : 2 / 50 = (T:ℕ) / N) : T = 60 :=
by
  -- The proof is omitted
  sorry

end fish_tagged_initially_l66_66497


namespace geometric_sequence_value_of_b_l66_66023

theorem geometric_sequence_value_of_b : 
  ∃ b : ℝ, 180 * (b / 180) = b ∧ (b / 180) * b = 64 / 25 ∧ b > 0 ∧ b = 21.6 :=
by sorry

end geometric_sequence_value_of_b_l66_66023


namespace evaluate_x_from_geometric_series_l66_66157

theorem evaluate_x_from_geometric_series (x : ℝ) (h : ∑' n : ℕ, x ^ n = 4) : x = 3 / 4 :=
sorry

end evaluate_x_from_geometric_series_l66_66157


namespace exponent_form_l66_66599

theorem exponent_form (x : ℕ) (k : ℕ) : (3^x) % 10 = 7 ↔ x = 4 * k + 3 :=
by
  sorry

end exponent_form_l66_66599


namespace find_G_8_l66_66265

noncomputable def G : Polynomial ℝ := sorry 

variable (x : ℝ)

theorem find_G_8 :
  G.eval 4 = 8 ∧ 
  (∀ x, (G.eval (2*x)) / (G.eval (x+2)) = 4 - (16 * x) / (x^2 + 2 * x + 2)) →
  G.eval 8 = 40 := 
sorry

end find_G_8_l66_66265


namespace student_good_probability_l66_66474

-- Defining the conditions as given in the problem
def P_A1 := 0.25          -- Probability of selecting a student from School A
def P_A2 := 0.4           -- Probability of selecting a student from School B
def P_A3 := 0.35          -- Probability of selecting a student from School C

def P_B_given_A1 := 0.3   -- Probability that a student's level is good given they are from School A
def P_B_given_A2 := 0.6   -- Probability that a student's level is good given they are from School B
def P_B_given_A3 := 0.5   -- Probability that a student's level is good given they are from School C

-- Main theorem statement
theorem student_good_probability : 
  P_A1 * P_B_given_A1 + P_A2 * P_B_given_A2 + P_A3 * P_B_given_A3 = 0.49 := 
by sorry

end student_good_probability_l66_66474


namespace probability_A_and_B_selected_l66_66606

theorem probability_A_and_B_selected :
  let n := 5
  let r := 3
  let total_ways := Nat.choose n r
  let favorable_ways := Nat.choose 3 1
  let probability := favorable_ways / total_ways
  probability = (3 / 10) := 
by
  sorry

end probability_A_and_B_selected_l66_66606


namespace max_m_value_l66_66430

theorem max_m_value (m : ℕ) (h1 : m > 0) (h2 : ∃ k : ℕ, m^4 + 16 * m + 8 = k * (k + 1)) : m ≤ 2 :=
sorry

end max_m_value_l66_66430


namespace strange_number_l66_66947

theorem strange_number (x : ℤ) (h : (x - 7) * 7 = (x - 11) * 11) : x = 18 :=
sorry

end strange_number_l66_66947


namespace average_speed_is_one_l66_66286

-- Definition of distance and time
def distance : ℕ := 1800
def time_in_minutes : ℕ := 30
def time_in_seconds : ℕ := time_in_minutes * 60

-- Definition of average speed as distance divided by time
def average_speed (distance : ℕ) (time : ℕ) : ℚ :=
  distance / time

-- Theorem: Given the distance and time, the average speed is 1 meter per second
theorem average_speed_is_one : average_speed distance time_in_seconds = 1 :=
  by
    sorry

end average_speed_is_one_l66_66286


namespace range_of_a_l66_66287

def p (a : ℝ) : Prop := (a + 2) > 1
def q (a : ℝ) : Prop := (4 - 4 * a) ≥ 0
def prop_and (a : ℝ) : Prop := p a ∧ q a
def prop_or (a : ℝ) : Prop := p a ∨ q a
def valid_a (a : ℝ) : Prop := (a ∈ Set.Iic (-1)) ∨ (a ∈ Set.Ioi 1)

theorem range_of_a (a : ℝ) (h_and : ¬ prop_and a) (h_or : prop_or a) : valid_a a := 
sorry

end range_of_a_l66_66287


namespace biology_marks_l66_66353

theorem biology_marks 
  (e : ℕ) (m : ℕ) (p : ℕ) (c : ℕ) (a : ℕ) (n : ℕ) (b : ℕ) 
  (h_e : e = 96) (h_m : m = 95) (h_p : p = 82) (h_c : c = 97) (h_a : a = 93) (h_n : n = 5)
  (h_total : e + m + p + c + b = a * n) :
  b = 95 :=
by 
  sorry

end biology_marks_l66_66353


namespace solution_set_abs_ineq_l66_66277

theorem solution_set_abs_ineq (x : ℝ) : abs (2 - x) ≤ 1 ↔ 1 ≤ x ∧ x ≤ 3 := by
  sorry

end solution_set_abs_ineq_l66_66277


namespace last_digit_of_sum_1_to_5_last_digit_of_sum_1_to_2012_l66_66734

theorem last_digit_of_sum_1_to_5 : 
  (1 ^ 2012 + 2 ^ 2012 + 3 ^ 2012 + 4 ^ 2012 + 5 ^ 2012) % 10 = 9 :=
  sorry

theorem last_digit_of_sum_1_to_2012 : 
  (List.sum (List.map (λ k => k ^ 2012) (List.range 2012).tail)) % 10 = 0 :=
  sorry

end last_digit_of_sum_1_to_5_last_digit_of_sum_1_to_2012_l66_66734


namespace distance_of_ladder_to_building_l66_66577

theorem distance_of_ladder_to_building :
  ∀ (c a b : ℕ), c = 25 ∧ a = 20 ∧ (a^2 + b^2 = c^2) → b = 15 :=
by
  intros c a b h
  rcases h with ⟨hc, ha, hpyth⟩
  have h1 : c = 25 := hc
  have h2 : a = 20 := ha
  have h3 : a^2 + b^2 = c^2 := hpyth
  sorry

end distance_of_ladder_to_building_l66_66577


namespace total_money_collected_l66_66295

theorem total_money_collected (attendees : ℕ) (reserved_price unreserved_price : ℝ) (reserved_sold unreserved_sold : ℕ)
  (h_attendees : attendees = 1096)
  (h_reserved_price : reserved_price = 25.00)
  (h_unreserved_price : unreserved_price = 20.00)
  (h_reserved_sold : reserved_sold = 246)
  (h_unreserved_sold : unreserved_sold = 246) :
  (reserved_price * reserved_sold + unreserved_price * unreserved_sold) = 11070.00 :=
by
  sorry

end total_money_collected_l66_66295


namespace initial_mean_of_observations_l66_66336

theorem initial_mean_of_observations (M : ℚ) (h : 50 * M + 11 = 50 * 36.5) : M = 36.28 := 
by
  sorry

end initial_mean_of_observations_l66_66336


namespace find_x_coordinate_l66_66810

-- Define the center and radius of the circle
structure Circle where
  center : (ℝ × ℝ)
  radius : ℝ

-- Define the points on the circle
def lies_on_circle (C : Circle) (P : ℝ × ℝ) : Prop :=
  let (x_c, y_c) := C.center
  let (x_p, y_p) := P
  (x_p - x_c)^2 + (y_p - y_c)^2 = C.radius^2

-- Lean 4 statement
theorem find_x_coordinate :
  ∀ (C : Circle), C.radius = 2 → lies_on_circle C (2, 0) ∧ lies_on_circle C (-2, 0) → 2 = 2 := by
  intro C h_radius ⟨h_lies_on_2_0, h_lies_on__2_0⟩
  sorry

end find_x_coordinate_l66_66810


namespace choir_members_minimum_l66_66186

theorem choir_members_minimum (n : ℕ) : (∃ n, n % 8 = 0 ∧ n % 9 = 0 ∧ n % 10 = 0 ∧ ∀ m, (m % 8 = 0 ∧ m % 9 = 0 ∧ m % 10 = 0) → n ≤ m) → n = 360 :=
by
  sorry

end choir_members_minimum_l66_66186


namespace cos_sum_of_angles_l66_66805

theorem cos_sum_of_angles (α β : Real) (h1 : Real.sin α = 4/5) (h2 : (π/2) < α ∧ α < π) 
(h3 : Real.cos β = -5/13) (h4 : 0 < β ∧ β < π/2) : 
  Real.cos (α + β) = -33/65 := 
by
  sorry

end cos_sum_of_angles_l66_66805


namespace painted_by_all_three_l66_66579

/-
Statement: Given that 75% of the floor is painted red, 70% painted green, and 65% painted blue,
prove that at least 10% of the floor is painted with all three colors.
-/

def painted_by_red (floor : ℝ) : ℝ := 0.75 * floor
def painted_by_green (floor : ℝ) : ℝ := 0.70 * floor
def painted_by_blue (floor : ℝ) : ℝ := 0.65 * floor

theorem painted_by_all_three (floor : ℝ) :
  ∃ (x : ℝ), x = 0.10 * floor ∧
  (painted_by_red floor) + (painted_by_green floor) + (painted_by_blue floor) ≥ 2 * floor :=
sorry

end painted_by_all_three_l66_66579


namespace inequality_proof_l66_66656

-- Define the conditions and the theorem statement
variables {a b c d : ℝ}

theorem inequality_proof (h1 : c < d) (h2 : a > b) (h3 : b > 0) : a - c > b - d :=
by
  sorry

end inequality_proof_l66_66656


namespace find_ratio_of_radii_l66_66200

noncomputable def ratio_of_radii (a b : ℝ) (h1 : π * b ^ 2 - π * a ^ 2 = 4 * π * a ^ 2) : Prop :=
  a / b = Real.sqrt 5 / 5

theorem find_ratio_of_radii (a b : ℝ) (h1 : π * b ^ 2 - π * a ^ 2 = 4 * π * a ^ 2) :
  ratio_of_radii a b h1 :=
sorry

end find_ratio_of_radii_l66_66200


namespace value_of_expression_l66_66785

theorem value_of_expression (x : ℝ) (h : 2 * x^2 + 3 * x + 7 = 8) : 9 - 4 * x^2 - 6 * x = 7 := by
  sorry

end value_of_expression_l66_66785


namespace quotient_zero_l66_66204

theorem quotient_zero (D d R Q : ℕ) (hD : D = 12) (hd : d = 17) (hR : R = 8) (h : D = d * Q + R) : Q = 0 :=
by
  sorry

end quotient_zero_l66_66204


namespace expected_number_of_digits_l66_66899

-- Define a noncomputable expected_digits function for an icosahedral die
noncomputable def expected_digits : ℝ :=
  let p1 := 9 / 20
  let p2 := 11 / 20
  (p1 * 1) + (p2 * 2)

theorem expected_number_of_digits :
  expected_digits = 1.55 :=
by
  -- The proof will be filled in here
  sorry

end expected_number_of_digits_l66_66899


namespace animal_costs_l66_66253

theorem animal_costs (S K L : ℕ) (h1 : K = 4 * S) (h2 : L = 4 * K) (h3 : S + 2 * K + L = 200) :
  S = 8 ∧ K = 32 ∧ L = 128 :=
by
  sorry

end animal_costs_l66_66253


namespace p_is_sufficient_but_not_necessary_l66_66003

-- Definitions based on conditions
def p (x y : Int) : Prop := x + y ≠ -2
def q (x y : Int) : Prop := ¬(x = -1 ∧ y = -1)

theorem p_is_sufficient_but_not_necessary (x y : Int) : 
  (p x y → q x y) ∧ ¬(q x y → p x y) :=
by
  sorry

end p_is_sufficient_but_not_necessary_l66_66003


namespace sum_of_roots_l66_66226

theorem sum_of_roots (r s t : ℝ) (h : 3 * r * s * t - 9 * (r * s + s * t + t * r) - 28 * (r + s + t) + 12 = 0) : r + s + t = 3 :=
by sorry

end sum_of_roots_l66_66226


namespace complex_numbers_count_l66_66568

theorem complex_numbers_count (z : ℂ) (h1 : z^24 = 1) (h2 : ∃ r : ℝ, z^6 = r) : ℕ :=
  sorry -- Proof goes here

end complex_numbers_count_l66_66568


namespace maximize_profit_l66_66378

noncomputable def profit (m : ℝ) : ℝ := 
  29 - (16 / (m + 1) + (m + 1))

theorem maximize_profit : 
  ∃ m : ℝ, m = 3 ∧ m ≥ 0 ∧ profit m = 21 :=
by
  use 3
  repeat { sorry }

end maximize_profit_l66_66378


namespace triangle_side_a_l66_66997

theorem triangle_side_a (a : ℝ) (h1 : 4 < a) (h2 : a < 10) : a = 8 :=
  by
  sorry

end triangle_side_a_l66_66997


namespace rashmi_speed_second_day_l66_66153

noncomputable def rashmi_speed (distance speed1 time_late time_early : ℝ) : ℝ :=
  let time1 := distance / speed1
  let on_time := time1 - time_late / 60
  let time2 := on_time - time_early / 60
  distance / time2

theorem rashmi_speed_second_day :
  rashmi_speed 9.999999999999993 5 10 10 = 6 := by
  sorry

end rashmi_speed_second_day_l66_66153


namespace factory_production_eq_l66_66887

theorem factory_production_eq (x : ℝ) (h1 : x > 50) : 450 / (x - 50) - 400 / x = 1 := 
by 
  sorry

end factory_production_eq_l66_66887


namespace find_ratio_l66_66753

theorem find_ratio (a b : ℝ) (h1 : a ≠ b) (h2 : a / b + (a + 10 * b) / (b + 10 * a) = 2) : a / b = 0.8 :=
  sorry

end find_ratio_l66_66753


namespace area_of_region_W_l66_66960

structure Rhombus (P Q R T : Type) :=
  (side_length : ℝ)
  (angle_Q : ℝ)

def Region_W
  (P Q R T : Type)
  (r : Rhombus P Q R T)
  (h_side : r.side_length = 5)
  (h_angle : r.angle_Q = 90) : ℝ :=
6.25

theorem area_of_region_W
  {P Q R T : Type}
  (r : Rhombus P Q R T)
  (h_side : r.side_length = 5)
  (h_angle : r.angle_Q = 90) :
  Region_W P Q R T r h_side h_angle = 6.25 :=
sorry

end area_of_region_W_l66_66960


namespace tan_neg440_eq_neg_sqrt_one_minus_m_sq_div_m_l66_66723

theorem tan_neg440_eq_neg_sqrt_one_minus_m_sq_div_m (m : ℝ) (h : Real.cos (80 * Real.pi / 180) = m) :
    Real.tan (-440 * Real.pi / 180) = - (Real.sqrt (1 - m^2) / m) :=
by
  -- proof goes here
  sorry

end tan_neg440_eq_neg_sqrt_one_minus_m_sq_div_m_l66_66723


namespace savings_after_four_weeks_l66_66751

noncomputable def hourly_wage (name : String) : ℝ :=
  match name with
  | "Robby" | "Jaylen" | "Miranda" => 10
  | "Alex" => 12
  | "Beth" => 15
  | "Chris" => 20
  | _ => 0

noncomputable def daily_hours (name : String) : ℝ :=
  match name with
  | "Robby" | "Miranda" => 10
  | "Jaylen" => 8
  | "Alex" => 6
  | "Beth" => 4
  | "Chris" => 3
  | _ => 0

noncomputable def saving_rate (name : String) : ℝ :=
  match name with
  | "Robby" => 2/5
  | "Jaylen" => 3/5
  | "Miranda" => 1/2
  | "Alex" => 1/3
  | "Beth" => 1/4
  | "Chris" => 3/4
  | _ => 0

noncomputable def weekly_earning (name : String) : ℝ :=
  hourly_wage name * daily_hours name * 5

noncomputable def weekly_saving (name : String) : ℝ :=
  weekly_earning name * saving_rate name

noncomputable def combined_savings : ℝ :=
  4 * (weekly_saving "Robby" + 
       weekly_saving "Jaylen" + 
       weekly_saving "Miranda" + 
       weekly_saving "Alex" + 
       weekly_saving "Beth" + 
       weekly_saving "Chris")

theorem savings_after_four_weeks :
  combined_savings = 4440 :=
by
  sorry

end savings_after_four_weeks_l66_66751


namespace remainder_when_divided_by_5_l66_66543

theorem remainder_when_divided_by_5 : (1234 * 1987 * 2013 * 2021) % 5 = 4 :=
by
  sorry

end remainder_when_divided_by_5_l66_66543


namespace ratio_of_couch_to_table_l66_66677

theorem ratio_of_couch_to_table
    (C T X : ℝ)
    (h1 : T = 3 * C)
    (h2 : X = 300)
    (h3 : C + T + X = 380) :
  X / T = 5 := 
by 
  sorry

end ratio_of_couch_to_table_l66_66677


namespace son_distance_from_father_is_correct_l66_66659

noncomputable def distance_between_son_and_father 
  (L F S d : ℝ) 
  (h_L : L = 6) 
  (h_F : F = 1.8) 
  (h_S : S = 0.9) 
  (h_d : d = 2.1) 
  (incident_point_condition : F / d = L / (d + x) ∧ S / x = F / (d + x)) : ℝ :=
  4.9

theorem son_distance_from_father_is_correct (L F S d : ℝ) 
  (h_L : L = 6) 
  (h_F : F = 1.8) 
  (h_S : S = 0.9) 
  (h_d : d = 2.1)
  (incident_point_condition : F / d = L / (d + 4.9) ∧ S / 4.9 = F / (d + 4.9)) : 
  distance_between_son_and_father L F S d h_L h_F h_S h_d incident_point_condition = 4.9 :=
sorry

end son_distance_from_father_is_correct_l66_66659


namespace find_n_l66_66086

theorem find_n (n : ℤ) (h : (n + 1999) / 2 = -1) : n = -2001 := 
sorry

end find_n_l66_66086


namespace division_value_of_712_5_by_12_5_is_57_l66_66824

theorem division_value_of_712_5_by_12_5_is_57 : 712.5 / 12.5 = 57 :=
  by
    sorry

end division_value_of_712_5_by_12_5_is_57_l66_66824


namespace new_ratio_of_boarders_to_day_scholars_l66_66033

theorem new_ratio_of_boarders_to_day_scholars
  (B_initial D_initial : ℕ)
  (B_initial_eq : B_initial = 560)
  (ratio_initial : B_initial / D_initial = 7 / 16)
  (new_boarders : ℕ)
  (new_boarders_eq : new_boarders = 80)
  (B_new : ℕ)
  (B_new_eq : B_new = B_initial + new_boarders)
  (D_new : ℕ)
  (D_new_eq : D_new = D_initial) :
  B_new / D_new = 1 / 2 :=
by
  sorry

end new_ratio_of_boarders_to_day_scholars_l66_66033


namespace average_cost_is_thirteen_l66_66597

noncomputable def averageCostPerPen (pensCost shippingCost : ℝ) (totalPens : ℕ) : ℕ :=
  Nat.ceil ((pensCost + shippingCost) * 100 / totalPens)

theorem average_cost_is_thirteen :
  averageCostPerPen 29.85 8.10 300 = 13 :=
by
  sorry

end average_cost_is_thirteen_l66_66597


namespace correct_exponent_operation_l66_66720

theorem correct_exponent_operation (x : ℝ) : x ^ 3 * x ^ 2 = x ^ 5 :=
by sorry

end correct_exponent_operation_l66_66720


namespace circle_ratio_new_diameter_circumference_l66_66741

theorem circle_ratio_new_diameter_circumference (r : ℝ) :
  let new_radius := r + 2
  let new_diameter := 2 * new_radius
  let new_circumference := 2 * Real.pi * new_radius
  new_circumference / new_diameter = Real.pi := 
by
  sorry

end circle_ratio_new_diameter_circumference_l66_66741


namespace Maggie_earnings_l66_66506

theorem Maggie_earnings :
  let family_commission := 7
  let neighbor_commission := 6
  let bonus_fixed := 10
  let bonus_threshold := 10
  let bonus_per_subscription := 1
  let monday_family := 4 + 1 
  let tuesday_neighbors := 2 + 2 * 2
  let wednesday_family := 3 + 1
  let total_family := monday_family + wednesday_family
  let total_neighbors := tuesday_neighbors
  let total_subscriptions := total_family + total_neighbors
  let bonus := if total_subscriptions > bonus_threshold then 
                 bonus_fixed + bonus_per_subscription * (total_subscriptions - bonus_threshold)
               else 0
  let total_earnings := total_family * family_commission + total_neighbors * neighbor_commission + bonus
  total_earnings = 114 := 
by {
  -- Placeholder for the proof. We assume this step will contain a verification of derived calculations.
  sorry
}

end Maggie_earnings_l66_66506


namespace find_a6_l66_66836

-- Define the arithmetic sequence properties
variables (a : ℕ → ℤ) (d : ℤ)

-- Define the initial conditions
axiom h1 : a 4 = 1
axiom h2 : a 7 = 16
axiom h_arith_seq : ∀ n, a (n + 1) - a n = d

-- Statement to prove
theorem find_a6 : a 6 = 11 :=
by
  sorry

end find_a6_l66_66836


namespace geometric_seq_problem_l66_66251

theorem geometric_seq_problem
  (a : Nat → ℝ)
  (r : ℝ)
  (h_geom : ∀ n, a (n + 1) = a n * r)
  (h_cond : a 1 * a 99 = 16) :
  a 20 * a 80 = 16 := 
sorry

end geometric_seq_problem_l66_66251


namespace less_than_half_l66_66928

theorem less_than_half (a b c : ℝ) (h₁ : a = 43.2) (h₂ : b = 0.5) (h₃ : c = 42.7) : a - b = c := by
  sorry

end less_than_half_l66_66928


namespace kerosene_cost_l66_66139

theorem kerosene_cost (R E K : ℕ) (h1 : E = R) (h2 : K = 6 * E) (h3 : R = 24) : 2 * K = 288 :=
by
  sorry

end kerosene_cost_l66_66139


namespace find_part_length_in_inches_find_part_length_in_feet_and_inches_l66_66155

def feetToInches (feet : ℕ) : ℕ := feet * 12

def totalLengthInInches (feet : ℕ) (inches : ℕ) : ℕ := feetToInches feet + inches

def partLengthInInches (totalLength : ℕ) (parts : ℕ) : ℕ := totalLength / parts

def inchesToFeetAndInches (inches : ℕ) : Nat × Nat := (inches / 12, inches % 12)

theorem find_part_length_in_inches (feet : ℕ) (inches : ℕ) (parts : ℕ)
    (h1 : feet = 10) (h2 : inches = 5) (h3 : parts = 5) :
    partLengthInInches (totalLengthInInches feet inches) parts = 25 := by
  sorry

theorem find_part_length_in_feet_and_inches (feet : ℕ) (inches : ℕ) (parts : ℕ)
    (h1 : feet = 10) (h2 : inches = 5) (h3 : parts = 5) :
    inchesToFeetAndInches (partLengthInInches (totalLengthInInches feet inches) parts) = (2, 1) := by
  sorry

end find_part_length_in_inches_find_part_length_in_feet_and_inches_l66_66155


namespace ratio_of_chicken_to_beef_l66_66488

theorem ratio_of_chicken_to_beef
  (beef_pounds : ℕ)
  (chicken_price_per_pound : ℕ)
  (total_cost : ℕ)
  (beef_price_per_pound : ℕ)
  (beef_cost : ℕ)
  (chicken_cost : ℕ)
  (chicken_pounds : ℕ) :
  beef_pounds = 1000 →
  beef_price_per_pound = 8 →
  total_cost = 14000 →
  beef_cost = beef_pounds * beef_price_per_pound →
  chicken_cost = total_cost - beef_cost →
  chicken_price_per_pound = 3 →
  chicken_pounds = chicken_cost / chicken_price_per_pound →
  chicken_pounds / beef_pounds = 2 :=
by
  intros
  sorry

end ratio_of_chicken_to_beef_l66_66488


namespace smallest_n_satisfying_conditions_l66_66438

theorem smallest_n_satisfying_conditions :
  ∃ n : ℕ, n ≥ 100 ∧ n < 1000 ∧ n ≡ 1 [MOD 7] ∧ n ≡ 1 [MOD 4] ∧ n = 113 :=
by
  sorry

end smallest_n_satisfying_conditions_l66_66438


namespace final_points_l66_66472

-- Definitions of the points in each round
def first_round_points : Int := 16
def second_round_points : Int := 33
def last_round_points : Int := -48

-- The theorem to prove Emily's final points
theorem final_points :
  first_round_points + second_round_points + last_round_points = 1 :=
by
  sorry

end final_points_l66_66472


namespace M_subset_N_l66_66399

-- Define M and N using the given conditions
def M : Set ℝ := {α | ∃ (k : ℤ), α = k * 90} ∪ {α | ∃ (k : ℤ), α = k * 180 + 45}
def N : Set ℝ := {α | ∃ (k : ℤ), α = k * 45}

-- Prove that M is a subset of N
theorem M_subset_N : M ⊆ N :=
by
  sorry

end M_subset_N_l66_66399


namespace solve_for_x_l66_66562

theorem solve_for_x 
  (y : ℚ) (x : ℚ)
  (h : x / (x - 1) = (y^3 + 2 * y^2 - 2) / (y^3 + 2 * y^2 - 3)) :
  x = (y^3 + 2 * y^2 - 2) / 2 :=
sorry

end solve_for_x_l66_66562


namespace find_number_l66_66480

theorem find_number (n x : ℤ)
  (h1 : (2 * x + 1) = (x - 7)) 
  (h2 : ∃ x : ℤ, n = (2 * x + 1) ^ 2) : 
  n = 25 := 
sorry

end find_number_l66_66480


namespace solve_inequality_l66_66214

noncomputable def f (x : ℝ) : ℝ := sorry

axiom f_add (m n : ℝ) : f (m + n) = f m * f n
axiom f_pos (x : ℝ) : 0 < x → 0 < f x ∧ f x < 1

theorem solve_inequality (x : ℝ) : f (x^2) * f (2 * x - 3) > 1 ↔ -3 < x ∧ x < 1 := sorry

end solve_inequality_l66_66214


namespace total_books_proof_l66_66453

noncomputable def economics_books (T : ℝ) := (1/4) * T + 10
noncomputable def rest_books (T : ℝ) := T - economics_books T
noncomputable def social_studies_books (T : ℝ) := (3/5) * rest_books T - 5
noncomputable def other_books := 13
noncomputable def science_books := 12
noncomputable def total_books_equation (T : ℝ) :=
  T = economics_books T + social_studies_books T + science_books + other_books

theorem total_books_proof : ∃ T : ℝ, total_books_equation T ∧ T = 80 := by
  sorry

end total_books_proof_l66_66453


namespace surface_area_circumscribed_sphere_l66_66821

theorem surface_area_circumscribed_sphere (a b c : ℝ) (ha : a = 3) (hb : b = 4) (hc : c = 5) :
    4 * Real.pi * ((Real.sqrt (a^2 + b^2 + c^2) / 2)^2) = 50 * Real.pi :=
by
  rw [ha, hb, hc]
  -- prove the equality step-by-step
  sorry

end surface_area_circumscribed_sphere_l66_66821


namespace value_of_x_l66_66728

theorem value_of_x (x : ℚ) (h : (3 * x + 4) / 7 = 15) : x = 101 / 3 :=
by
  sorry

end value_of_x_l66_66728


namespace arithmetic_sequence__geometric_sequence__l66_66039

-- Part 1: Arithmetic Sequence
theorem arithmetic_sequence_
  (d : ℤ) (n : ℤ) (a_n : ℤ) (a_1 : ℤ) (S_n : ℤ)
  (h_d : d = 2) (h_n : n = 15) (h_a_n : a_n = -10)
  (h_a_1 : a_1 = -38) (h_S_n : S_n = -360) :
  a_n = a_1 + (n - 1) * d ∧ S_n = n * (a_1 + a_n) / 2 :=
by
  sorry

-- Part 2: Geometric Sequence
theorem geometric_sequence_
  (a_1 : ℝ) (q : ℝ) (S_10 : ℝ)
  (a_2 : ℝ) (a_3 : ℝ) (a_4 : ℝ)
  (h_a_2_3 : a_2 + a_3 = 6) (h_a_3_4 : a_3 + a_4 = 12)
  (h_a_1 : a_1 = 1) (h_q : q = 2) (h_S_10 : S_10 = 1023) :
  a_2 = a_1 * q ∧ a_3 = a_1 * q^2 ∧ a_4 = a_1 * q^3 ∧ S_10 = a_1 * (1 - q^10) / (1 - q) :=
by
  sorry

end arithmetic_sequence__geometric_sequence__l66_66039


namespace solve_system_of_equations_l66_66147

theorem solve_system_of_equations : 
  ∀ x y : ℝ, 
    (2 * x^2 - 3 * x * y + y^2 = 3) ∧ 
    (x^2 + 2 * x * y - 2 * y^2 = 6) 
    ↔ (x = 2 ∧ y = 1) ∨ (x = -2 ∧ y = -1) :=
by
  sorry

end solve_system_of_equations_l66_66147


namespace problem_l66_66285

noncomputable def f (ω φ : ℝ) (x : ℝ) := 4 * Real.sin (ω * x + φ)

theorem problem (ω : ℝ) (φ : ℝ) (x1 x2 α : ℝ) (hω : 0 < ω) (hφ : |φ| < Real.pi / 2)
  (h0 : f ω φ 0 = 2 * Real.sqrt 3)
  (hx1 : f ω φ x1 = 0) (hx2 : f ω φ x2 = 0) (hx1x2 : |x1 - x2| = Real.pi / 2)
  (hα : α ∈ Set.Ioo (Real.pi / 12) (Real.pi / 2)) :
  f 2 (Real.pi / 3) α = 12 / 5 ∧ Real.sin (2 * α) = (3 + 4 * Real.sqrt 3) / 10 :=
sorry

end problem_l66_66285


namespace washing_machine_regular_wash_l66_66001

variable {R : ℕ}

/-- A washing machine uses 20 gallons of water for a heavy wash,
2 gallons of water for a light wash, and an additional light wash
is added when bleach is used. Given conditions:
- Two heavy washes are done.
- Three regular washes are done.
- One light wash is done.
- Two loads are bleached.
- Total water used is 76 gallons.
Prove the washing machine uses 10 gallons of water for a regular wash. -/
theorem washing_machine_regular_wash (h : 2 * 20 + 3 * R + 1 * 2 + 2 * 2 = 76) : R = 10 :=
by
  sorry

end washing_machine_regular_wash_l66_66001


namespace find_natural_numbers_l66_66456

theorem find_natural_numbers (x y : ℕ) (h1 : x > y) (h2 : x + y + (x - y) + x * y + x / y = 3^5) : 
  (x = 6 ∧ y = 3) := 
sorry

end find_natural_numbers_l66_66456


namespace sqrt_product_l66_66636

theorem sqrt_product (a b : ℝ) (ha : a = 20) (hb : b = 1/5) : Real.sqrt a * Real.sqrt b = 2 := 
by
  sorry

end sqrt_product_l66_66636


namespace initial_pencils_l66_66995

theorem initial_pencils (pencils_added initial_pencils total_pencils : ℕ) 
  (h1 : pencils_added = 3) 
  (h2 : total_pencils = 5) :
  initial_pencils = total_pencils - pencils_added := 
by 
  sorry

end initial_pencils_l66_66995


namespace find_x_l66_66726

theorem find_x (x y : ℤ) (some_number : ℤ) (h1 : y = 2) (h2 : some_number = 14) (h3 : 2 * x - y = some_number) : x = 8 :=
by 
  sorry

end find_x_l66_66726


namespace triangle_ABC_properties_l66_66402

noncomputable def is_arithmetic_sequence (α β γ : ℝ) : Prop :=
γ - β = β - α

theorem triangle_ABC_properties
  (A B C a c : ℝ)
  (b : ℝ := Real.sqrt 3)
  (h1 : a * Real.cos C + c * Real.cos A = 2 * b * Real.cos B) :
  is_arithmetic_sequence A B C ∧
  ∃ (max_area : ℝ), max_area = (3 * Real.sqrt 3) / 4 := by sorry

end triangle_ABC_properties_l66_66402


namespace unique_integer_solution_l66_66030

theorem unique_integer_solution :
  ∃! (z : ℤ), 5 * z ≤ 2 * z - 8 ∧ -3 * z ≥ 18 ∧ 7 * z ≤ -3 * z - 21 :=
by
  sorry

end unique_integer_solution_l66_66030


namespace find_integer_x_l66_66498

theorem find_integer_x : ∃ x : ℤ, x^5 - 3 * x^2 = 216 ∧ x = 3 :=
by {
  sorry
}

end find_integer_x_l66_66498


namespace ratio_of_areas_l66_66477

noncomputable def area (A B C D : ℝ) : ℝ := 0  -- Placeholder, exact area definition will require geometrical formalism.

variables (A B C D P Q R S : ℝ)

-- Define the conditions
variables (h1 : AB = BP) (h2 : BC = CQ) (h3 : CD = DR) (h4 : DA = AS)

-- Lean 4 statement for the proof problem
theorem ratio_of_areas : area A B C D / area P Q R S = 1/5 :=
sorry

end ratio_of_areas_l66_66477


namespace min_value_expression_l66_66069

theorem min_value_expression : 
  ∃ (x y : ℝ), x^2 + 2 * x * y + 2 * y^2 + 3 * x - 5 * y = -8.5 := by
  sorry

end min_value_expression_l66_66069


namespace interest_rate_per_annum_l66_66889

theorem interest_rate_per_annum
  (P : ℕ := 450) 
  (t : ℕ := 8) 
  (I : ℕ := P - 306) 
  (simple_interest : ℕ := P * r * t / 100) :
  r = 4 :=
by
  sorry

end interest_rate_per_annum_l66_66889


namespace find_a10_l66_66845

variable {q : ℝ}
variable {a : ℕ → ℝ}

-- Sequence conditions
axiom geo_seq (n : ℕ) : a (n + 1) = a n * q
axiom positive_ratio : 0 < q
axiom condition_1 : a 2 = 1
axiom condition_2 : a 4 * a 8 = 2 * (a 5) ^ 2

theorem find_a10 : a 10 = 16 := by
  sorry

end find_a10_l66_66845


namespace find_range_m_l66_66987

def p (m : ℝ) : Prop := m > 2 ∨ m < -2
def q (m : ℝ) : Prop := 1 < m ∧ m < 3

theorem find_range_m (h₁ : ¬ p m) (h₂ : q m) : (1 : ℝ) < m ∧ m ≤ 2 :=
by sorry

end find_range_m_l66_66987


namespace double_neg_five_eq_five_l66_66953

theorem double_neg_five_eq_five : -(-5) = 5 := 
sorry

end double_neg_five_eq_five_l66_66953


namespace bees_lost_each_day_l66_66542

theorem bees_lost_each_day
    (initial_bees : ℕ)
    (daily_hatch : ℕ)
    (days : ℕ)
    (total_bees_after_days : ℕ)
    (bees_lost_each_day : ℕ) :
    initial_bees = 12500 →
    daily_hatch = 3000 →
    days = 7 →
    total_bees_after_days = 27201 →
    (initial_bees + days * (daily_hatch - bees_lost_each_day) = total_bees_after_days) →
    bees_lost_each_day = 899 :=
by
  intros h_initial h_hatch h_days h_total h_eq
  sorry

end bees_lost_each_day_l66_66542


namespace bellas_score_l66_66094

-- Definitions from the problem conditions
def n : Nat := 17
def x : Nat := 75
def new_n : Nat := n + 1
def y : Nat := 76

-- Assertion that Bella's score is 93
theorem bellas_score : (new_n * y) - (n * x) = 93 :=
by
  -- This is where the proof would go
  sorry

end bellas_score_l66_66094


namespace pet_store_cages_l66_66760

-- Definitions and conditions
def initial_puppies : ℕ := 56
def sold_puppies : ℕ := 24
def puppies_per_cage : ℕ := 4
def remaining_puppies : ℕ := initial_puppies - sold_puppies
def cages_used : ℕ := remaining_puppies / puppies_per_cage

-- Theorem statement
theorem pet_store_cages : cages_used = 8 := by sorry

end pet_store_cages_l66_66760


namespace proof_problem_l66_66106

theorem proof_problem (x y z : ℝ) (h₁ : x ≠ y) 
  (h₂ : (x^2 - y*z) / (x * (1 - y*z)) = (y^2 - x*z) / (y * (1 - x*z))) :
  x + y + z = 1/x + 1/y + 1/z :=
sorry

end proof_problem_l66_66106


namespace sourdough_cost_eq_nine_l66_66054

noncomputable def cost_per_visit (white_bread_cost baguette_cost croissant_cost: ℕ) : ℕ :=
  2 * white_bread_cost + baguette_cost + croissant_cost

noncomputable def total_spent (weekly_cost num_weeks: ℕ) : ℕ :=
  weekly_cost * num_weeks

noncomputable def total_sourdough_spent (total_spent weekly_cost num_weeks: ℕ) : ℕ :=
  total_spent - weekly_cost * num_weeks

noncomputable def total_sourdough_per_week (total_sourdough_spent num_weeks: ℕ) : ℕ :=
  total_sourdough_spent / num_weeks

theorem sourdough_cost_eq_nine (white_bread_cost baguette_cost croissant_cost total_spent_over_4_weeks: ℕ)
  (h₁: white_bread_cost = 350) (h₂: baguette_cost = 150) (h₃: croissant_cost = 200) (h₄: total_spent_over_4_weeks = 7800) :
  total_sourdough_per_week (total_sourdough_spent total_spent_over_4_weeks (cost_per_visit white_bread_cost baguette_cost croissant_cost) 4) 4 = 900 :=
by 
  sorry

end sourdough_cost_eq_nine_l66_66054


namespace vector_dot_product_correct_l66_66764

-- Definitions of the vectors
def vector_a : ℝ × ℝ := (1, 2)
def vector_b : ℝ × ℝ :=
  let x := 4 - 2 * vector_a.1
  let y := 1 - 2 * vector_a.2
  (x, y)

-- Theorem to prove the dot product is correct
theorem vector_dot_product_correct :
  (vector_a.1 * vector_b.1 + vector_a.2 * vector_b.2) = 4 := by
  sorry

end vector_dot_product_correct_l66_66764


namespace john_total_water_usage_l66_66288

-- Define the basic conditions
def total_days_in_weeks (weeks : ℕ) : ℕ := weeks * 7
def showers_every_other_day (days : ℕ) : ℕ := days / 2
def total_minutes_shower (showers : ℕ) (minutes_per_shower : ℕ) : ℕ := showers * minutes_per_shower
def total_water_usage (total_minutes : ℕ) (water_per_minute : ℕ) : ℕ := total_minutes * water_per_minute

-- Main statement
theorem john_total_water_usage :
  total_water_usage (total_minutes_shower (showers_every_other_day (total_days_in_weeks 4)) 10) 2 = 280 :=
by
  sorry

end john_total_water_usage_l66_66288


namespace square_perimeter_ratio_l66_66397

theorem square_perimeter_ratio (a b : ℝ) (h : a^2 / b^2 = 16 / 25) :
  (4 * a) / (4 * b) = 4 / 5 :=
by
  sorry

end square_perimeter_ratio_l66_66397


namespace height_of_platform_l66_66712

variable (h l w : ℕ)

-- Define the conditions as hypotheses
def measured_length_first_configuration : Prop := l + h - w = 40
def measured_length_second_configuration : Prop := w + h - l = 34

-- The goal is to prove that the height is 37 inches
theorem height_of_platform
  (h l w : ℕ)
  (config1 : measured_length_first_configuration h l w)
  (config2 : measured_length_second_configuration h l w) : 
  h = 37 := 
sorry

end height_of_platform_l66_66712


namespace area_percentage_decrease_l66_66067

theorem area_percentage_decrease {a b : ℝ} 
  (h1 : 2 * b = 0.1 * 4 * a) :
  ((b^2) / (a^2) * 100 = 4) :=
by
  sorry

end area_percentage_decrease_l66_66067


namespace calculate_g_l66_66112

def g (a b c : ℚ) : ℚ := (2 * c + a) / (b - c)

theorem calculate_g : g 3 6 (-1) = 1 / 7 :=
by
    -- Proof is not included
    sorry

end calculate_g_l66_66112


namespace cos_alpha_value_l66_66188
open Real

theorem cos_alpha_value (α : ℝ) (h0 : 0 < α ∧ α < π / 2) 
  (h1 : sin (α - π / 6) = 1 / 3) : 
  cos α = (2 * sqrt 6 - 1) / 6 := 
by 
  sorry

end cos_alpha_value_l66_66188


namespace percentage_decrease_equivalent_l66_66371

theorem percentage_decrease_equivalent :
  ∀ (P D : ℝ), 
    (D = 10) →
    ((1.25 * P) - (D / 100) * (1.25 * P) = 1.125 * P) :=
by
  intros P D h
  rw [h]
  sorry

end percentage_decrease_equivalent_l66_66371


namespace amy_spent_32_l66_66574

theorem amy_spent_32 (x: ℝ) (h1: 0.15 * x + 1.6 * x + x = 55) : 1.6 * x = 32 :=
by
  sorry

end amy_spent_32_l66_66574


namespace cristina_nicky_head_start_l66_66234

theorem cristina_nicky_head_start (s_c s_n : ℕ) (t d : ℕ) 
  (h1 : s_c = 5) 
  (h2 : s_n = 3) 
  (h3 : t = 30)
  (h4 : d = s_n * t):
  d = 90 := 
by
  sorry

end cristina_nicky_head_start_l66_66234


namespace find_first_term_of_arithmetic_sequence_l66_66815

theorem find_first_term_of_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ)
  (h_arith_seq : ∀ n, a (n + 1) = a n + d)
  (h_a3 : a 3 = 2)
  (h_d : d = -1/2) : a 1 = 3 :=
sorry

end find_first_term_of_arithmetic_sequence_l66_66815


namespace triangle_medians_and_area_l66_66256

/-- Given a triangle with side lengths 13, 14, and 15,
    prove that the sum of the squares of the lengths of the medians is 385
    and the area of the triangle is 84. -/
theorem triangle_medians_and_area :
  let a := 13
  let b := 14
  let c := 15
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  let m_a := Real.sqrt (2 * b^2 + 2 * c^2 - a^2) / 2
  let m_b := Real.sqrt (2 * c^2 + 2 * a^2 - b^2) / 2
  let m_c := Real.sqrt (2 * a^2 + 2 * b^2 - c^2) / 2
  m_a^2 + m_b^2 + m_c^2 = 385 ∧ area = 84 := sorry

end triangle_medians_and_area_l66_66256


namespace reinforcement_size_l66_66075

theorem reinforcement_size (R : ℕ) : 
  2000 * 39 = (2000 + R) * 20 → R = 1900 :=
by
  intro h
  sorry

end reinforcement_size_l66_66075


namespace eq_root_condition_l66_66707

theorem eq_root_condition (k : ℝ) 
    (h_discriminant : -4 * k + 5 ≥ 0)
    (h_roots : ∃ x1 x2 : ℝ, 
        (x1 + x2 = 1 - 2 * k) ∧ 
        (x1 * x2 = k^2 - 1) ∧ 
        (x1^2 + x2^2 = 16 + x1 * x2)) :
    k = -2 :=
sorry

end eq_root_condition_l66_66707


namespace speed_on_local_roads_l66_66727

theorem speed_on_local_roads (v : ℝ) (h1 : 60 + 120 = 180) (h2 : (60 + 120) / (60 / v + 120 / 60) = 36) : v = 20 :=
by
  sorry

end speed_on_local_roads_l66_66727


namespace tangent_length_from_A_to_circle_l66_66586

noncomputable def point_A_polar : (ℝ × ℝ) := (6, Real.pi)
noncomputable def circle_eq_polar (θ : ℝ) : ℝ := -4 * Real.cos θ

theorem tangent_length_from_A_to_circle : 
  ∃ (length : ℝ), length = 2 * Real.sqrt 3 ∧ 
  (∃ (ρ θ : ℝ), point_A_polar = (6, Real.pi) ∧ ρ = circle_eq_polar θ) :=
sorry

end tangent_length_from_A_to_circle_l66_66586


namespace rhombus_side_length_l66_66924

variables (r α : ℝ) (hα : 0 < α ∧ α < π / 2) (hr : 0 < r)

theorem rhombus_side_length (r α : ℝ) (hα : 0 < α ∧ α < π / 2) (hr : 0 < r) :
  ∃ s : ℝ, s = 2 * r / Real.sin α :=
sorry

end rhombus_side_length_l66_66924


namespace largest_integer_among_four_l66_66860

theorem largest_integer_among_four 
  (p q r s : ℤ)
  (h1 : p + q + r = 210)
  (h2 : p + q + s = 230)
  (h3 : p + r + s = 250)
  (h4 : q + r + s = 270) :
  max (max p q) (max r s) = 110 :=
by
  sorry

end largest_integer_among_four_l66_66860


namespace triangle_inequality_l66_66145

theorem triangle_inequality (ABC: Triangle) (M : Point) (a b c : ℝ)
  (h1 : a = BC) (h2 : b = CA) (h3 : c = AB) :
  (1 / a^2) + (1 / b^2) + (1 / c^2) ≥ 3 / (MA^2 + MB^2 + MC^2) := 
sorry

end triangle_inequality_l66_66145


namespace sixteen_powers_five_equals_four_power_ten_l66_66462

theorem sixteen_powers_five_equals_four_power_ten : 
  (16 * 16 * 16 * 16 * 16 = 4 ^ 10) :=
by
  sorry

end sixteen_powers_five_equals_four_power_ten_l66_66462


namespace students_play_both_football_and_cricket_l66_66962

theorem students_play_both_football_and_cricket :
  ∀ (total F C N both : ℕ),
  total = 460 →
  F = 325 →
  C = 175 →
  N = 50 →
  total - N = F + C - both →
  both = 90 :=
by
  intros
  sorry

end students_play_both_football_and_cricket_l66_66962


namespace line_eq_l66_66532

theorem line_eq (a b : ℝ) (h_a_pos : 0 < a) (h_b_pos : 0 < b) (h_eq : 1 / a + 9 / b = 1) (h_min_interp : a + b = 16) : 
  ∃ l : ℝ × ℝ → ℝ, ∀ x y : ℝ, l (x, y) = 3 * x + y - 12 :=
by
  sorry

end line_eq_l66_66532


namespace solution_set_x2_minus_x_lt_0_l66_66685

theorem solution_set_x2_minus_x_lt_0 :
  ∀ x : ℝ, (0 < x ∧ x < 1) ↔ x^2 - x < 0 := 
by
  sorry

end solution_set_x2_minus_x_lt_0_l66_66685


namespace position_of_point_l66_66383

theorem position_of_point (a b : ℝ) (h_tangent: (a ≠ 0 ∨ b ≠ 0) ∧ (a^2 + b^2 = 1)) : a^2 + b^2 = 1 :=
by
  sorry

end position_of_point_l66_66383


namespace solve_quadratic_l66_66318

theorem solve_quadratic (x : ℝ) (h : x^2 - 6*x + 8 = 0) : x = 2 ∨ x = 4 :=
sorry

end solve_quadratic_l66_66318


namespace intersection_of_M_and_N_l66_66501

def M := {x : ℝ | abs x ≤ 2}
def N := {x : ℝ | x^2 - 3 * x = 0}

theorem intersection_of_M_and_N : M ∩ N = {0} :=
by
  sorry

end intersection_of_M_and_N_l66_66501


namespace at_least_one_not_less_than_two_l66_66985

theorem at_least_one_not_less_than_two (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + 1 / b) ≥ 2 ∨ (b + 1 / c) ≥ 2 ∨ (c + 1 / a) ≥ 2 :=
sorry

end at_least_one_not_less_than_two_l66_66985


namespace intersection_points_of_parabolas_l66_66211

/-- Let P1 be the equation of the first parabola: y = 3x^2 - 8x + 2 -/
def P1 (x : ℝ) : ℝ := 3 * x^2 - 8 * x + 2

/-- Let P2 be the equation of the second parabola: y = 6x^2 + 4x + 2 -/
def P2 (x : ℝ) : ℝ := 6 * x^2 + 4 * x + 2

/-- Prove that the intersection points of P1 and P2 are (-4, 82) and (0, 2) -/
theorem intersection_points_of_parabolas : 
  {p : ℝ × ℝ | ∃ x, p = (x, P1 x) ∧ P1 x = P2 x} = 
    {(-4, 82), (0, 2)} :=
sorry

end intersection_points_of_parabolas_l66_66211


namespace largest_common_value_less_than_1000_l66_66560

theorem largest_common_value_less_than_1000 :
  ∃ a : ℕ, a < 1000 ∧ (∃ n : ℤ, a = 4 + 5 * n) ∧ (∃ m : ℤ, a = 5 + 8 * m) ∧ 
            (∀ b : ℕ, (b < 1000 ∧ (∃ n : ℤ, b = 4 + 5 * n) ∧ (∃ m : ℤ, b = 5 + 8 * m)) → b ≤ a) :=
sorry

end largest_common_value_less_than_1000_l66_66560


namespace largest_of_six_consecutive_sum_2070_is_347_l66_66529

theorem largest_of_six_consecutive_sum_2070_is_347 (n : ℕ) :
  n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) = 2070 → n + 5 = 347 :=
by
  intro h
  sorry

end largest_of_six_consecutive_sum_2070_is_347_l66_66529


namespace negative_number_is_d_l66_66981

def a : Int := -(-2)
def b : Int := abs (-2)
def c : Int := (-2) ^ 2
def d : Int := (-2) ^ 3

theorem negative_number_is_d : d < 0 :=
  by
  sorry

end negative_number_is_d_l66_66981


namespace geometric_series_sum_l66_66537

theorem geometric_series_sum :
  let a := 2
  let r := -2
  let n := 10
  let Sn := (a : ℚ) * (r^n - 1) / (r - 1)
  Sn = 2050 / 3 :=
by
  sorry

end geometric_series_sum_l66_66537


namespace mike_travel_time_l66_66648

-- Definitions of conditions
def dave_steps_per_min : ℕ := 85
def dave_step_length_cm : ℕ := 70
def dave_time_min : ℕ := 20
def mike_steps_per_min : ℕ := 95
def mike_step_length_cm : ℕ := 65

-- Calculate Dave's speed in cm/min
def dave_speed_cm_per_min := dave_steps_per_min * dave_step_length_cm

-- Calculate the distance to school in cm
def school_distance_cm := dave_speed_cm_per_min * dave_time_min

-- Calculate Mike's speed in cm/min
def mike_speed_cm_per_min := mike_steps_per_min * mike_step_length_cm

-- Calculate the time for Mike to get to school in minutes as a rational number
def mike_time_min := (school_distance_cm : ℚ) / mike_speed_cm_per_min

-- The proof problem statement
theorem mike_travel_time :
  mike_time_min = 19 + 2 / 7 :=
sorry

end mike_travel_time_l66_66648


namespace ratio_A_B_correct_l66_66913

-- Define the shares of A, B, and C
def A_share := 372
def B_share := 93
def C_share := 62

-- Total amount distributed
def total_share := A_share + B_share + C_share

-- The ratio of A's share to B's share
def ratio_A_to_B := A_share / B_share

theorem ratio_A_B_correct : 
  total_share = 527 ∧ 
  ¬(B_share = (1 / 4) * C_share) ∧ 
  ratio_A_to_B = 4 := 
by
  sorry

end ratio_A_B_correct_l66_66913


namespace cosine_identity_l66_66591

variable (α : ℝ)

theorem cosine_identity (h : Real.sin (Real.pi / 6 - α) = 1 / 3) : 
  Real.cos (2 * Real.pi / 3 + 2 * α) = -7 / 9 :=
by
  sorry

end cosine_identity_l66_66591


namespace molecular_weight_BaSO4_l66_66819

-- Definitions for atomic weights of elements.
def atomic_weight_Ba : ℝ := 137.33
def atomic_weight_S : ℝ := 32.07
def atomic_weight_O : ℝ := 16.00

-- Defining the number of atoms in BaSO4
def num_Ba : ℕ := 1
def num_S : ℕ := 1
def num_O : ℕ := 4

-- Statement to be proved
theorem molecular_weight_BaSO4 :
  (num_Ba * atomic_weight_Ba + num_S * atomic_weight_S + num_O * atomic_weight_O) = 233.40 := 
by
  sorry

end molecular_weight_BaSO4_l66_66819


namespace alma_carrots_leftover_l66_66825

/-- Alma has 47 baby carrots and wishes to distribute them equally among 4 goats.
    We need to prove that the number of leftover carrots after such distribution is 3. -/
theorem alma_carrots_leftover (total_carrots : ℕ) (goats : ℕ) (leftover : ℕ) 
  (h1 : total_carrots = 47) (h2 : goats = 4) (h3 : leftover = total_carrots % goats) : 
  leftover = 3 :=
by
  sorry

end alma_carrots_leftover_l66_66825


namespace letter_puzzle_solutions_l66_66247

theorem letter_puzzle_solutions (A B : ℕ) : 
  (1 ≤ A ∧ A < 10) ∧ (1 ≤ B ∧ B < 10) ∧ (A ≠ B) ∧ (A^B = 10 * B + A) → 
  (A = 2 ∧ B = 5) ∨ (A = 6 ∧ B = 2) ∨ (A = 4 ∧ B = 3) :=
by
  sorry

end letter_puzzle_solutions_l66_66247


namespace clock_angle_7_15_l66_66820

noncomputable def hour_angle_at (hour : ℕ) (minutes : ℕ) : ℝ :=
  hour * 30 + (minutes * 0.5)

noncomputable def minute_angle_at (minutes : ℕ) : ℝ :=
  minutes * 6

noncomputable def small_angle (angle1 angle2 : ℝ) : ℝ :=
  let diff := abs (angle1 - angle2)
  if diff <= 180 then diff else 360 - diff

theorem clock_angle_7_15 : small_angle (hour_angle_at 7 15) (minute_angle_at 15) = 127.5 :=
by
  sorry

end clock_angle_7_15_l66_66820


namespace smallest_fraction_numerator_l66_66742

theorem smallest_fraction_numerator :
  ∃ (a b : ℕ), a ≥ 10 ∧ a ≤ 99 ∧ b ≥ 10 ∧ b ≤ 99 ∧ (4 * b < 9 * a) ∧ 
  (∀ (a' b' : ℕ), a' ≥ 10 ∧ a' ≤ 99 ∧ b' ≥ 10 ∧ b' ≤ 99 ∧ (4 * b' < 9 * a') → b * a' ≥ a * b') ∧ a = 41 :=
sorry

end smallest_fraction_numerator_l66_66742


namespace minimum_A2_minus_B2_l66_66690

noncomputable def A (x y z : ℝ) : ℝ := 
  Real.sqrt (x + 6) + Real.sqrt (y + 7) + Real.sqrt (z + 12)

noncomputable def B (x y z : ℝ) : ℝ :=
  Real.sqrt (x + 2) + Real.sqrt (y + 3) + Real.sqrt (z + 5)

theorem minimum_A2_minus_B2 (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) : 
  (A x y z)^2 - (B x y z)^2 = 49.25 := 
by 
  sorry 

end minimum_A2_minus_B2_l66_66690


namespace inequality_solution_l66_66491

noncomputable def solve_inequality : Set ℝ :=
  {x | (x - 5) / ((x - 3)^2) < 0}

theorem inequality_solution :
  solve_inequality = {x | x < 3} ∪ {x | 3 < x ∧ x < 5} :=
by
  sorry

end inequality_solution_l66_66491


namespace tom_searching_days_l66_66215

variable (d : ℕ) (total_cost : ℕ)

theorem tom_searching_days :
  (∀ n, n ≤ 5 → total_cost = n * 100 + (d - n) * 60) →
  (∀ n, n > 5 → total_cost = 5 * 100 + (d - 5) * 60) →
  total_cost = 800 →
  d = 10 :=
by
  intros h1 h2 h3
  sorry

end tom_searching_days_l66_66215


namespace line_equation_through_P_and_intercepts_l66_66882

-- Define the conditions
structure Point (α : Type*) := 
  (x : α) 
  (y : α)

-- Given point P
def P : Point ℝ := ⟨5, 6⟩

-- Equation of a line passing through (x₀, y₀) and 
-- having the intercepts condition: the x-intercept is twice the y-intercept

theorem line_equation_through_P_and_intercepts :
  (∃ (a b c : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ (a * 5 + b * 6 + c = 0) ∧ 
   ((-c / a = 2 * (-c / b)) ∧ (c ≠ 0)) ∧
   (a = 1 ∧ b = 2 ∧ c = -17) ∨
   (a = 6 ∧ b = -5 ∧ c = 0)) :=
sorry

end line_equation_through_P_and_intercepts_l66_66882


namespace single_elimination_games_l66_66844

theorem single_elimination_games (n : ℕ) (h : n = 512) : ∃ g : ℕ, g = n - 1 :=
by
  have h1 : n = 512 := h
  use 511
  sorry

end single_elimination_games_l66_66844


namespace goats_more_than_pigs_l66_66613

-- Defining the number of goats
def number_of_goats : ℕ := 66

-- Condition: there are twice as many chickens as goats
def number_of_chickens : ℕ := 2 * number_of_goats

-- Calculating the total number of goats and chickens
def total_goats_and_chickens : ℕ := number_of_goats + number_of_chickens

-- Condition: the number of ducks is half of the total number of goats and chickens
def number_of_ducks : ℕ := total_goats_and_chickens / 2

-- Condition: the number of pigs is a third of the number of ducks
def number_of_pigs : ℕ := number_of_ducks / 3

-- The statement we need to prove
theorem goats_more_than_pigs : number_of_goats - number_of_pigs = 33 := by
  -- The proof is omitted as instructed
  sorry

end goats_more_than_pigs_l66_66613


namespace slices_per_friend_l66_66965

theorem slices_per_friend (total_slices friends : ℕ) (h1 : total_slices = 16) (h2 : friends = 4) : (total_slices / friends) = 4 :=
by
  sorry

end slices_per_friend_l66_66965


namespace sin_y_eq_neg_one_l66_66440

noncomputable def α := Real.arccos (-1 / 5)

theorem sin_y_eq_neg_one (x y z : ℝ) (h1 : x = y - α) (h2 : z = y + α)
  (h3 : (2 + Real.sin x) * (2 + Real.sin z) = (2 + Real.sin y) ^ 2) : Real.sin y = -1 :=
sorry

end sin_y_eq_neg_one_l66_66440


namespace calculate_expression_l66_66160

theorem calculate_expression :
  -15 - 21 + 8 = -28 :=
by
  sorry

end calculate_expression_l66_66160


namespace range_of_k_l66_66817

theorem range_of_k
  (a b c : ℝ)
  (ha : 0 < a)
  (hb : 0 < b)
  (hc : 0 < c)
  (h1 : a^2 + c^2 = 16)
  (h2 : b^2 + c^2 = 25) : 
  9 < a^2 + b^2 ∧ a^2 + b^2 < 41 :=
by
  sorry

end range_of_k_l66_66817


namespace number_of_distinct_configurations_l66_66585

-- Definitions of the problem conditions
structure CubeConfig where
  white_cubes : Finset (Fin 8)
  blue_cubes : Finset (Fin 8)
  condition_1 : white_cubes.card = 5
  condition_2 : blue_cubes.card = 3
  condition_3 : ∀ x ∈ white_cubes, x ∉ blue_cubes

def distinctConfigCount (configs : Finset CubeConfig) : ℕ :=
  (configs.filter (λ config => 
    config.white_cubes.card = 5 ∧
    config.blue_cubes.card = 3 ∧
    (∀ x ∈ config.white_cubes, x ∉ config.blue_cubes)
  )).card

-- Theorem stating the correct number of distinct configurations
theorem number_of_distinct_configurations : distinctConfigCount ∅ = 5 := 
  sorry

end number_of_distinct_configurations_l66_66585


namespace ram_shyam_weight_ratio_l66_66326

theorem ram_shyam_weight_ratio
    (R S : ℝ)
    (h1 : 1.10 * R + 1.22 * S = 82.8)
    (h2 : R + S = 72) :
    R / S = 7 / 5 :=
by sorry

end ram_shyam_weight_ratio_l66_66326


namespace least_positive_integer_l66_66203

theorem least_positive_integer (n : ℕ) (h1 : n > 1) 
  (h2 : n % 2 = 1) (h3 : n % 3 = 1) (h4 : n % 5 = 1) 
  (h5 : n % 7 = 1) (h6 : n % 11 = 1): 
  n = 2311 := 
by
  sorry

end least_positive_integer_l66_66203


namespace geometric_sequence_common_ratio_l66_66898

theorem geometric_sequence_common_ratio
  (a : ℕ → ℝ) (q : ℝ)
  (h1 : ∀ n, a n > 0)
  (h2 : a 3 = 1/2)
  (h3 : a 1 * (1 + q) = 3) :
  q = 1/2 :=
sorry

end geometric_sequence_common_ratio_l66_66898


namespace arithmetic_sequence_third_term_l66_66838

theorem arithmetic_sequence_third_term (S : ℕ → ℕ) (a : ℕ → ℕ) (d : ℕ) :
  (S 5 = 10) ∧ (S n = n * (a 1 + a n) / 2) ∧ (a 5 = a 1 + 4 * d) ∧ 
  (∀ n, a n = a 1 + (n-1) * d) → (a 3 = 2) :=
by
  intro h
  sorry

end arithmetic_sequence_third_term_l66_66838


namespace cos_alpha_add_beta_div2_l66_66629

open Real 

theorem cos_alpha_add_beta_div2 (α β : ℝ) 
  (h_range : -π/2 < β ∧ β < 0 ∧ 0 < α ∧ α < π/2)
  (h_cos1 : cos (π/4 + α) = 1/3)
  (h_cos2 : cos (π/4 - β/2) = sqrt 3 / 3) :
  cos (α + β/2) = 5 * sqrt 3 / 9 :=
sorry

end cos_alpha_add_beta_div2_l66_66629


namespace painting_problem_equation_l66_66846

def dougPaintingRate := 1 / 3
def davePaintingRate := 1 / 4
def combinedPaintingRate := dougPaintingRate + davePaintingRate
def timeRequiredToComplete (t : ℝ) : Prop := 
  (t - 1) * combinedPaintingRate = 2 / 3

theorem painting_problem_equation : ∃ t : ℝ, timeRequiredToComplete t :=
sorry

end painting_problem_equation_l66_66846


namespace find_t_from_x_l66_66519

theorem find_t_from_x (x : ℝ) (h : x + 1/x = 5) : x^2 + 1/x^2 = 23 :=
by
  sorry

end find_t_from_x_l66_66519


namespace michael_lap_time_l66_66349

theorem michael_lap_time :
  ∃ T : ℝ, (∀ D : ℝ, D = 45 → (9 * T = 10 * D) → T = 50) :=
by
  sorry

end michael_lap_time_l66_66349


namespace find_AD_l66_66774

-- Given conditions as definitions
def AB := 5 -- given length in meters
def angle_ABC := 85 -- given angle in degrees
def angle_BCA := 45 -- given angle in degrees
def angle_DBC := 20 -- given angle in degrees

-- Lean theorem statement to prove the result
theorem find_AD : AD = AB := by
  -- The proof will be filled in afterwards; currently, we leave it as sorry.
  sorry

end find_AD_l66_66774


namespace number_of_ways_to_put_7_balls_in_2_boxes_l66_66602

theorem number_of_ways_to_put_7_balls_in_2_boxes :
  let distributions := [(7,0), (6,1), (5,2), (4,3)]
  let binom : (ℕ × ℕ) → ℕ := fun p => Nat.choose p.fst p.snd
  let counts := [1, binom (7,6), binom (7,5), binom (7,4)]
  counts.sum = 64 := by sorry

end number_of_ways_to_put_7_balls_in_2_boxes_l66_66602


namespace tetrahedron_faces_congruent_iff_face_angle_sum_straight_l66_66487

-- Defining the Tetrahedron and its properties
structure Tetrahedron (V : Type*) :=
(A B C D : V)
(face_angle_sum_at_vertex : V → Prop)
(congruent_faces : Prop)

-- Translating the problem into a Lean 4 theorem statement
theorem tetrahedron_faces_congruent_iff_face_angle_sum_straight (V : Type*) 
  (T : Tetrahedron V) :
  T.face_angle_sum_at_vertex T.A = T.face_angle_sum_at_vertex T.B ∧ 
  T.face_angle_sum_at_vertex T.B = T.face_angle_sum_at_vertex T.C ∧ 
  T.face_angle_sum_at_vertex T.C = T.face_angle_sum_at_vertex T.D ↔ T.congruent_faces :=
sorry


end tetrahedron_faces_congruent_iff_face_angle_sum_straight_l66_66487


namespace simplify_and_evaluate_l66_66610

theorem simplify_and_evaluate (x y : ℤ) (hx : x = -1) (hy : y = 2) : 
  x^2 - 2 * (3 * y^2 - x * y) + (y^2 - 2 * x * y) = -19 := 
by
  -- Proof will go here, but it's omitted as per instructions
  sorry

end simplify_and_evaluate_l66_66610


namespace ratio_current_to_past_l66_66969

-- Conditions
def current_posters : ℕ := 22
def posters_after_summer (p : ℕ) : ℕ := p + 6
def posters_two_years_ago : ℕ := 14

-- Proof problem statement
theorem ratio_current_to_past (h₁ : current_posters = 22) (h₂ : posters_two_years_ago = 14) : 
  (current_posters / Nat.gcd current_posters posters_two_years_ago) = 11 ∧ 
  (posters_two_years_ago / Nat.gcd current_posters posters_two_years_ago) = 7 :=
by
  sorry

end ratio_current_to_past_l66_66969


namespace find_f_one_seventh_l66_66165

-- Define the function f
variable (f : ℝ → ℝ)

-- Given conditions
variable (monotonic_f : MonotonicOn f (Set.Ioi 0))
variable (h : ∀ x ∈ Set.Ioi (0 : ℝ), f (f x - 1 / x) = 2)

-- Define the domain
variable (x : ℝ)
variable (hx : x ∈ Set.Ioi (0 : ℝ))

-- The theorem to prove
theorem find_f_one_seventh : f (1 / 7) = 8 := by
  -- proof starts here
  sorry

end find_f_one_seventh_l66_66165


namespace radius_is_independent_variable_l66_66982

theorem radius_is_independent_variable 
  (r C : ℝ)
  (h : C = 2 * Real.pi * r) : 
  ∃ r_independent, r_independent = r := 
by
  sorry

end radius_is_independent_variable_l66_66982


namespace negation_of_statement_l66_66047

theorem negation_of_statement (x : ℝ) :
  (¬ (x^2 = 1 → x = 1 ∨ x = -1)) ↔ (x^2 = 1 ∧ (x ≠ 1 ∧ x ≠ -1)) :=
sorry

end negation_of_statement_l66_66047


namespace motorboat_speed_l66_66758

theorem motorboat_speed 
  (c : ℝ) (h_c : c = 2.28571428571)
  (t_up : ℝ) (h_t_up : t_up = 20 / 60)
  (t_down : ℝ) (h_t_down : t_down = 15 / 60) :
  ∃ v : ℝ, v = 16 :=
by
  sorry

end motorboat_speed_l66_66758


namespace sum_of_last_three_digits_9_pow_15_plus_15_pow_15_l66_66868

theorem sum_of_last_three_digits_9_pow_15_plus_15_pow_15 :
  (9 ^ 15 + 15 ^ 15) % 1000 = 24 :=
by
  sorry

end sum_of_last_three_digits_9_pow_15_plus_15_pow_15_l66_66868


namespace remaining_students_average_l66_66692

theorem remaining_students_average
  (N : ℕ) (A : ℕ) (M : ℕ) (B : ℕ) (E : ℕ)
  (h1 : N = 20)
  (h2 : A = 80)
  (h3 : M = 5)
  (h4 : B = 50)
  (h5 : E = (N - M))
  : (N * A - M * B) / E = 90 :=
by
  -- Using sorries to skip the proof
  sorry

end remaining_students_average_l66_66692


namespace train_cross_pole_time_l66_66761

noncomputable def L_train : ℝ := 300 -- Length of the train in meters
noncomputable def L_platform : ℝ := 870 -- Length of the platform in meters
noncomputable def t_platform : ℝ := 39 -- Time to cross the platform in seconds

theorem train_cross_pole_time
  (L_train : ℝ)
  (L_platform : ℝ)
  (t_platform : ℝ)
  (D : ℝ := L_train + L_platform)
  (v : ℝ := D / t_platform)
  (t_pole : ℝ := L_train / v) :
  t_pole = 10 :=
by sorry

end train_cross_pole_time_l66_66761


namespace smallest_positive_integer_k_l66_66916

theorem smallest_positive_integer_k:
  ∀ T : ℕ, ∀ n : ℕ, (T = n * (n + 1) / 2) → ∃ m : ℕ, 81 * T + 10 = m * (m + 1) / 2 :=
by
  intro T n h
  sorry

end smallest_positive_integer_k_l66_66916


namespace range_of_f_find_a_l66_66416

-- Define the function f
def f (a x : ℝ) : ℝ := -a^2 * x - 2 * a * x + 1

-- Define the proposition for part (1)
theorem range_of_f (a : ℝ) (h : a > 1) : Set.range (f a) = Set.Iio 1 := sorry

-- Define the proposition for part (2)
theorem find_a (a : ℝ) (h : a > 1) (min_value : ∀ x, x ∈ Set.Icc (-2 : ℝ) 1 → f a x ≥ -7) : a = 2 :=
sorry

end range_of_f_find_a_l66_66416


namespace find_principal_amount_l66_66782

theorem find_principal_amount :
  ∃ P : ℝ, P * (1 + 0.05) ^ 4 = 9724.05 ∧ P = 8000 :=
by
  sorry

end find_principal_amount_l66_66782


namespace no_integers_satisfy_eq_l66_66879

theorem no_integers_satisfy_eq (m n : ℤ) : ¬ (m^2 + 1954 = n^2) := 
by
  sorry

end no_integers_satisfy_eq_l66_66879


namespace part_I_part_II_l66_66012

noncomputable def f (a x : ℝ) : ℝ := a - 1 / (2^x + 1)

theorem part_I (a : ℝ) : ∀ x : ℝ, (0 < (2^x * Real.log 2) / (2^x + 1)^2) :=
by
  sorry

theorem part_II (h : ∀ x : ℝ, f a x = -f a (-x)) : 
  a = (1:ℝ)/2 ∧ ∀ x : ℝ, -((1:ℝ)/2) < f (1/2) x ∧ f (1/2) x < (1:ℝ)/2 :=
by
  sorry

end part_I_part_II_l66_66012


namespace percentage_of_number_l66_66786

variable (N P : ℝ)

theorem percentage_of_number 
  (h₁ : (1 / 4) * (1 / 3) * (2 / 5) * N = 10) 
  (h₂ : (P / 100) * N = 120) : 
  P = 40 := 
by 
  sorry

end percentage_of_number_l66_66786


namespace parallelogram_proof_l66_66062

noncomputable def sin_angle_degrees (θ : ℝ) : ℝ := Real.sin (θ * Real.pi / 180)

theorem parallelogram_proof (x : ℝ) (A : ℝ) (r : ℝ) (side1 side2 : ℝ) (P : ℝ):
  (A = 972) → (r = 4 / 3) → (sin_angle_degrees 45 = Real.sqrt 2 / 2) →
  (side1 = 4 * x) → (side2 = 3 * x) →
  (A = side1 * (side2 * (Real.sqrt 2 / 2 / 3))) →
  x = 9 * 2^(3/4) →
  side1 = 36 * 2^(3/4) →
  side2 = 27 * 2^(3/4) →
  (P = 2 * (side1 + side2)) →
  (P = 126 * 2^(3/4)) :=
by
  intros
  sorry

end parallelogram_proof_l66_66062


namespace problem_statement_l66_66291

theorem problem_statement (a b : ℝ) (h1 : a - b > 0) (h2 : a + b < 0) : b < 0 ∧ |b| > |a| :=
by
  sorry

end problem_statement_l66_66291


namespace david_average_speed_l66_66893

theorem david_average_speed (d t : ℚ) (h1 : d = 49 / 3) (h2 : t = 7 / 3) :
  (d / t) = 7 :=
by
  rw [h1, h2]
  norm_num

end david_average_speed_l66_66893


namespace hamburger_varieties_l66_66091

-- Define the problem conditions as Lean definitions.
def condiments := 9  -- There are 9 condiments
def patty_choices := 3  -- Choices of 1, 2, or 3 patties

-- The goal is to prove that the number of different kinds of hamburgers is 1536.
theorem hamburger_varieties : (3 * 2^9) = 1536 := by
  sorry

end hamburger_varieties_l66_66091


namespace divisibility_by_7_l66_66088

theorem divisibility_by_7 (n : ℕ) : (3^(2 * n + 1) + 2^(n + 2)) % 7 = 0 :=
by
  sorry

end divisibility_by_7_l66_66088


namespace maximum_value_l66_66772

noncomputable def f (x : ℝ) : ℝ := Real.exp x + x

theorem maximum_value : ∀ x ∈ Set.Icc (-1 : ℝ) 1, f x ≤ f 1 :=
by
  intros x hx
  sorry

end maximum_value_l66_66772


namespace r_can_complete_work_in_R_days_l66_66016

theorem r_can_complete_work_in_R_days (W : ℝ) : 
  (∀ p q r P Q R : ℝ, 
    (P = W / 24) ∧
    (Q = W / 9) ∧
    (10.000000000000002 * (W / 24) + 3 * (W / 9 + W / R) = W) 
  -> R = 12) :=
by
  intros
  sorry

end r_can_complete_work_in_R_days_l66_66016


namespace calculate_square_difference_l66_66849

theorem calculate_square_difference : 2023^2 - 2022^2 = 4045 := by
  sorry

end calculate_square_difference_l66_66849


namespace inequality_holds_for_all_reals_l66_66564

theorem inequality_holds_for_all_reals (x : ℝ) : 
  7 / 20 + |3 * x - 2 / 5| ≥ 1 / 4 :=
sorry

end inequality_holds_for_all_reals_l66_66564


namespace betty_bracelets_l66_66126

theorem betty_bracelets : (140 / 14) = 10 := 
by
  norm_num

end betty_bracelets_l66_66126


namespace cleaner_flow_rate_after_second_unclogging_l66_66032

theorem cleaner_flow_rate_after_second_unclogging
  (rate1 rate2 : ℕ) (time1 time2 total_time total_cleaner : ℕ)
  (used_cleaner1 used_cleaner2 : ℕ)
  (final_rate : ℕ)
  (H1 : rate1 = 2)
  (H2 : rate2 = 3)
  (H3 : time1 = 15)
  (H4 : time2 = 10)
  (H5 : total_time = 30)
  (H6 : total_cleaner = 80)
  (H7 : used_cleaner1 = rate1 * time1)
  (H8 : used_cleaner2 = rate2 * time2)
  (H9 : used_cleaner1 + used_cleaner2 ≤ total_cleaner)
  (H10 : final_rate = (total_cleaner - (used_cleaner1 + used_cleaner2)) / (total_time - (time1 + time2))) :
  final_rate = 4 := by
  sorry

end cleaner_flow_rate_after_second_unclogging_l66_66032


namespace is_periodic_l66_66952

noncomputable def f : ℝ → ℝ := sorry

axiom domain (x : ℝ) : true
axiom not_eq_neg1_and_not_eq_0 (x : ℝ) : f x ≠ -1 ∧ f x ≠ 0
axiom functional_eq (x y : ℝ) : f (x - y) = - (f x / (1 + f y))

theorem is_periodic : ∃ p, p > 0 ∧ ∀ x, f (x + p) = f x :=
sorry

end is_periodic_l66_66952


namespace hyperbola_properties_l66_66641

-- Definitions from the conditions
def line_l (x y : ℝ) : Prop := 4 * x - 3 * y + 20 = 0
def asymptote_l (x y : ℝ) : Prop := 4 * x - 3 * y = 0
def foci_on_x_axis (x y : ℝ) : Prop := y = 0

-- Standard equation of the hyperbola
def hyperbola_equation (x y : ℝ) : Prop := (x^2 / 9) - (y^2 / 16) = 1

-- Define eccentricity
def eccentricity := 5 / 3

-- Proof statement
theorem hyperbola_properties :
  (∃ x y : ℝ, line_l x y ∧ foci_on_x_axis x y) →
  (∃ x y : ℝ, asymptote_l x y) →
  ∃ x y : ℝ, hyperbola_equation x y ∧ eccentricity = 5 / 3 :=
by
  sorry

end hyperbola_properties_l66_66641


namespace no_solution_fermat_like_l66_66858

theorem no_solution_fermat_like (x y z k : ℕ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hk : k > 0) 
  (hxk : x < k) (hyk : y < k) (hxk_eq : x ^ k + y ^ k = z ^ k) : false :=
sorry

end no_solution_fermat_like_l66_66858


namespace smallest_prime_p_l66_66722

theorem smallest_prime_p 
  (p q r : ℕ) 
  (h1 : Nat.Prime p) 
  (h2 : Nat.Prime q) 
  (h3 : r > 0) 
  (h4 : p + q = r) 
  (h5 : q < p) 
  (h6 : q = 2) 
  (h7 : Nat.Prime r)  
  : p = 3 := 
sorry

end smallest_prime_p_l66_66722


namespace inv_100_mod_101_l66_66489

theorem inv_100_mod_101 : (100 : ℤ) * (100 : ℤ) % 101 = 1 := by
  sorry

end inv_100_mod_101_l66_66489


namespace number_of_cities_experienced_protests_l66_66520

variables (days_of_protest : ℕ) (arrests_per_day : ℕ) (days_pre_trial : ℕ) 
          (days_post_trial_in_weeks : ℕ) (combined_weeks_jail : ℕ)

def total_days_in_jail_per_person := days_pre_trial + (days_post_trial_in_weeks * 7) / 2

theorem number_of_cities_experienced_protests 
  (h1 : days_of_protest = 30) 
  (h2 : arrests_per_day = 10) 
  (h3 : days_pre_trial = 4) 
  (h4 : days_post_trial_in_weeks = 2) 
  (h5 : combined_weeks_jail = 9900) : 
  (combined_weeks_jail * 7) / total_days_in_jail_per_person 
  = 21 :=
by
  sorry

end number_of_cities_experienced_protests_l66_66520


namespace ratio_of_wealth_l66_66897

theorem ratio_of_wealth (P W : ℝ) (hP : P > 0) (hW : W > 0) : 
  let wX := (0.40 * W) / (0.20 * P)
  let wY := (0.30 * W) / (0.10 * P)
  (wX / wY) = 2 / 3 := 
by
  sorry

end ratio_of_wealth_l66_66897


namespace set_S_infinite_l66_66637

-- Definition of a power
def is_power (n : ℕ) : Prop := 
  ∃ (a k : ℕ), a > 0 ∧ k ≥ 2 ∧ n = a^k

-- Definition of the set S, those integers which cannot be expressed as the sum of two powers
def in_S (n : ℕ) : Prop := 
  ¬ ∃ (a b k m : ℕ), a > 0 ∧ b > 0 ∧ k ≥ 2 ∧ m ≥ 2 ∧ n = a^k + b^m

-- The theorem statement asserting that S is infinite
theorem set_S_infinite : Infinite {n : ℕ | in_S n} :=
sorry

end set_S_infinite_l66_66637


namespace two_students_solve_all_problems_l66_66135

theorem two_students_solve_all_problems
    (students : Fin 15 → Fin 6 → Prop)
    (h : ∀ (p : Fin 6), (∃ (s1 s2 s3 s4 s5 s6 s7 s8 : Fin 15), 
          students s1 p ∧ students s2 p ∧ students s3 p ∧ students s4 p ∧ 
          students s5 p ∧ students s6 p ∧ students s7 p ∧ students s8 p)) :
    ∃ (s1 s2 : Fin 15), ∀ (p : Fin 6), students s1 p ∨ students s2 p := 
by
    sorry

end two_students_solve_all_problems_l66_66135


namespace first_term_arith_seq_l66_66703

noncomputable def is_increasing (a b c : ℕ) (d : ℕ) : Prop := b = a + d ∧ c = a + 2 * d ∧ 0 < d

theorem first_term_arith_seq (a₁ a₂ a₃ : ℕ) (d: ℕ) :
  is_increasing a₁ a₂ a₃ d ∧ a₁ + a₂ + a₃ = 12 ∧ a₁ * a₂ * a₃ = 48 → a₁ = 2 := sorry

end first_term_arith_seq_l66_66703


namespace candles_to_new_five_oz_l66_66077

theorem candles_to_new_five_oz 
  (h_wax_percent: ℝ)
  (h_candles_20oz_count: ℕ) 
  (h_candles_5oz_count: ℕ) 
  (h_candles_1oz_count: ℕ) 
  (h_candles_20oz_wax: ℝ) 
  (h_candles_5oz_wax: ℝ)
  (h_candles_1oz_wax: ℝ):
  h_wax_percent = 0.10 →
  h_candles_20oz_count = 5 →
  h_candles_5oz_count = 5 → 
  h_candles_1oz_count = 25 →
  h_candles_20oz_wax = 20 →
  h_candles_5oz_wax = 5 →
  h_candles_1oz_wax = 1 →
  (h_wax_percent * h_candles_20oz_wax * h_candles_20oz_count + 
   h_wax_percent * h_candles_5oz_wax * h_candles_5oz_count + 
   h_wax_percent * h_candles_1oz_wax * h_candles_1oz_count) / 5 = 3 :=
by
  sorry

end candles_to_new_five_oz_l66_66077


namespace value_of_a_l66_66970

theorem value_of_a (x a : ℤ) (h : x = 3 ∧ x^2 = a) : a = 9 :=
sorry

end value_of_a_l66_66970


namespace fundraising_exceeded_goal_l66_66835

theorem fundraising_exceeded_goal (ken mary scott : ℕ) (goal: ℕ) 
  (h_ken : ken = 600)
  (h_mary_ken : mary = 5 * ken)
  (h_mary_scott : mary = 3 * scott)
  (h_goal : goal = 4000) :
  (ken + mary + scott) - goal = 600 := 
  sorry

end fundraising_exceeded_goal_l66_66835


namespace people_receiving_roses_l66_66464

-- Defining the conditions.
def initial_roses : Nat := 40
def stolen_roses : Nat := 4
def roses_per_person : Nat := 4

-- Stating the theorem.
theorem people_receiving_roses : 
  (initial_roses - stolen_roses) / roses_per_person = 9 :=
by sorry

end people_receiving_roses_l66_66464


namespace find_monotonic_intervals_max_min_on_interval_l66_66754

noncomputable def f (x : ℝ) : ℝ := Real.exp x * Real.cos x - x

noncomputable def f' (x : ℝ) : ℝ := (Real.cos x - Real.sin x) * Real.exp x - 1

theorem find_monotonic_intervals (k : ℤ) : 
  ((2 * k * Real.pi - Real.pi < x ∧ x < 2 * k * Real.pi) → 0 < (f' x)) ∧
  ((2 * k * Real.pi < x ∧ x < 2 * k * Real.pi + Real.pi) → (f' x) < 0) :=
sorry

theorem max_min_on_interval : 
  (∀ x, 0 ≤ x ∧ x ≤ (2 * Real.pi / 3) → f 0 = 1 ∧ f (2 * Real.pi / 3) =  -((1/2) * Real.exp (2/3 * Real.pi)) - (2 * Real.pi / 3)) :=
sorry

end find_monotonic_intervals_max_min_on_interval_l66_66754


namespace factor_expression_l66_66043

theorem factor_expression (x : ℝ) : 
  3 * x^2 * (x - 5) + 4 * x * (x - 5) + 6 * (x - 5) = (3 * x^2 + 4 * x + 6) * (x - 5) :=
  sorry

end factor_expression_l66_66043


namespace calculate_expression_l66_66857

variable (x y : ℚ)

theorem calculate_expression (h₁ : x = 4 / 6) (h₂ : y = 5 / 8) : 
  (6 * x + 8 * y) / (48 * x * y) = 9 / 20 :=
by
  -- proof steps here
  sorry

end calculate_expression_l66_66857


namespace birds_not_hawks_warbler_kingfisher_l66_66395

variables (B : ℝ)
variables (hawks paddyfield_warblers kingfishers : ℝ)

-- Conditions
def condition1 := hawks = 0.30 * B
def condition2 := paddyfield_warblers = 0.40 * (B - hawks)
def condition3 := kingfishers = 0.25 * paddyfield_warblers

-- Question: Prove the percentage of birds that are not hawks, paddyfield-warblers, or kingfishers is 35%
theorem birds_not_hawks_warbler_kingfisher (B hawks paddyfield_warblers kingfishers : ℝ) 
 (h1 : hawks = 0.30 * B) 
 (h2 : paddyfield_warblers = 0.40 * (B - hawks)) 
 (h3 : kingfishers = 0.25 * paddyfield_warblers) : 
 (1 - (hawks + paddyfield_warblers + kingfishers) / B) * 100 = 35 :=
by
  sorry

end birds_not_hawks_warbler_kingfisher_l66_66395


namespace root_in_interval_l66_66766

noncomputable def f (x : ℝ) : ℝ := Real.log x - x + 2

theorem root_in_interval : ∃ x ∈ Set.Ioo (3 : ℝ) (4 : ℝ), f x = 0 := sorry

end root_in_interval_l66_66766


namespace solve_for_x_l66_66765

theorem solve_for_x (x : ℕ) : x * 12 = 173 * 240 → x = 3460 :=
by
  sorry

end solve_for_x_l66_66765


namespace large_number_exponent_l66_66174

theorem large_number_exponent (h : 10000 = 10 ^ 4) : 10000 ^ 50 * 10 ^ 5 = 10 ^ 205 := 
by
  sorry

end large_number_exponent_l66_66174


namespace find_A_plus_B_l66_66019

theorem find_A_plus_B {A B : ℚ} (h : ∀ x : ℚ, 
                     (Bx - 17) / (x^2 - 9 * x + 20) = A / (x - 4) + 5 / (x - 5)) : 
                     A + B = 9 / 5 := sorry

end find_A_plus_B_l66_66019


namespace triangle_perimeter_l66_66967

/-- In a triangle ABC, where sides a, b, c are opposite to angles A, B, C respectively.
Given the area of the triangle = 15 * sqrt 3 / 4, 
angle A = 60 degrees and 5 * sin B = 3 * sin C,
prove that the perimeter of triangle ABC is 8 + sqrt 19. -/
theorem triangle_perimeter
  (a b c : ℝ)
  (A B C : ℝ)
  (hA : A = 60)
  (h_area : (1 / 2) * b * c * (Real.sin (A / (180 / Real.pi))) = 15 * Real.sqrt 3 / 4)
  (h_sin : 5 * Real.sin B = 3 * Real.sin C) :
  a + b + c = 8 + Real.sqrt 19 :=
sorry

end triangle_perimeter_l66_66967


namespace original_length_of_wood_l66_66323

theorem original_length_of_wood (s cl ol : ℝ) (h1 : s = 2.3) (h2 : cl = 6.6) (h3 : ol = cl + s) : 
  ol = 8.9 := 
by 
  sorry

end original_length_of_wood_l66_66323


namespace inscribed_circle_radius_l66_66446

theorem inscribed_circle_radius :
  ∀ (r : ℝ), 
    (∀ (R : ℝ), R = 12 →
      (∀ (d : ℝ), d = 12 → r = 3)) :=
by sorry

end inscribed_circle_radius_l66_66446


namespace geometric_seq_increasing_condition_l66_66076

theorem geometric_seq_increasing_condition (q : ℝ) (a : ℕ → ℝ): 
  (∀ n : ℕ, a (n + 1) = q * a n) → (¬ (∀ a : ℕ → ℝ, (∀ n : ℕ, a (n + 1) = q * a n) → ∀ n m : ℕ, n < m → a n < a m) ∧ ¬ (¬ (∀ a : ℕ → ℝ, (∀ n : ℕ, a (n + 1) = q * a n) → ∀ n m : ℕ, n < m → a n < a m))) :=
sorry

end geometric_seq_increasing_condition_l66_66076


namespace factorial_fraction_simplification_l66_66096

-- Define necessary factorial function
def fact : ℕ → ℕ
| 0       => 1
| (n + 1) => (n + 1) * fact n

-- Define the problem
theorem factorial_fraction_simplification :
  (4 * fact 6 + 20 * fact 5) / fact 7 = 22 / 21 := by
  sorry

end factorial_fraction_simplification_l66_66096


namespace a5_eq_neg3_l66_66749

-- Define arithmetic sequence condition
def is_arithmetic_sequence (a : ℕ → ℤ) :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

-- Define the sequence with given conditions
def a (n : ℕ) : ℤ :=
  if n = 2 then -5
  else if n = 8 then 1
  else sorry  -- Placeholder for other values

axiom a3_eq_neg5 : a 2 = -5
axiom a9_eq_1 : a 8 = 1
axiom a_is_arithmetic : is_arithmetic_sequence a

-- Statement to prove
theorem a5_eq_neg3 : a 4 = -3 :=
by
  sorry

end a5_eq_neg3_l66_66749


namespace train_length_l66_66877

/-- 
  Given:
  - jogger_speed is the jogger's speed in km/hr (9 km/hr)
  - train_speed is the train's speed in km/hr (45 km/hr)
  - jogger_ahead is the jogger's initial lead in meters (240 m)
  - passing_time is the time in seconds for the train to pass the jogger (36 s)
  
  Prove that the length of the train is 120 meters.
-/
theorem train_length
  (jogger_speed : ℕ) -- in km/hr
  (train_speed : ℕ) -- in km/hr
  (jogger_ahead : ℕ) -- in meters
  (passing_time : ℕ) -- in seconds
  (h_jogger_speed : jogger_speed = 9)
  (h_train_speed : train_speed = 45)
  (h_jogger_ahead : jogger_ahead = 240)
  (h_passing_time : passing_time = 36)
  : ∃ length_of_train : ℕ, length_of_train = 120 :=
by
  sorry

end train_length_l66_66877


namespace left_handed_rock_music_lovers_l66_66827

theorem left_handed_rock_music_lovers (total_club_members left_handed_members rock_music_lovers right_handed_dislike_rock: ℕ)
  (h1 : total_club_members = 25)
  (h2 : left_handed_members = 10)
  (h3 : rock_music_lovers = 18)
  (h4 : right_handed_dislike_rock = 3)
  (h5 : total_club_members = left_handed_members + (total_club_members - left_handed_members))
  : (∃ x : ℕ, x = 6 ∧ x + (left_handed_members - x) + (rock_music_lovers - x) + right_handed_dislike_rock = total_club_members) :=
sorry

end left_handed_rock_music_lovers_l66_66827


namespace arithmetic_sequence_problem_l66_66024

theorem arithmetic_sequence_problem
  (a : ℕ → ℚ)
  (h : a 2 + a 4 + a 9 + a 11 = 32) :
  a 6 + a 7 = 16 :=
sorry

end arithmetic_sequence_problem_l66_66024


namespace not_sunny_prob_l66_66321

theorem not_sunny_prob (P_sunny : ℚ) (h : P_sunny = 5/7) : 1 - P_sunny = 2/7 :=
by sorry

end not_sunny_prob_l66_66321


namespace value_of_a_l66_66450

theorem value_of_a (a : ℤ) (h1 : 2 * a + 6 + (3 - a) = 0) : a = -9 :=
sorry

end value_of_a_l66_66450


namespace find_speed_of_current_l66_66022

variable {m c : ℝ}

theorem find_speed_of_current
  (h1 : m + c = 15)
  (h2 : m - c = 10) :
  c = 2.5 :=
sorry

end find_speed_of_current_l66_66022


namespace gcd_of_16_and_12_l66_66238

theorem gcd_of_16_and_12 : Nat.gcd 16 12 = 4 := by
  sorry

end gcd_of_16_and_12_l66_66238


namespace stewart_farm_horseFood_l66_66961

variable (sheep horses horseFoodPerHorse : ℕ)
variable (ratio_sh_to_hs : ℕ × ℕ)
variable (totalHorseFood : ℕ)

noncomputable def horse_food_per_day (sheep : ℕ) (ratio_sh_to_hs : ℕ × ℕ) (totalHorseFood : ℕ) : ℕ :=
  let horses := (sheep * ratio_sh_to_hs.2) / ratio_sh_to_hs.1
  totalHorseFood / horses

theorem stewart_farm_horseFood (h_ratio : ratio_sh_to_hs = (4, 7))
                                (h_sheep : sheep = 32)
                                (h_total : totalHorseFood = 12880) :
    horse_food_per_day sheep ratio_sh_to_hs totalHorseFood = 230 := by
  sorry

end stewart_farm_horseFood_l66_66961


namespace b_joined_after_a_l66_66158

def months_b_joined (a_investment : ℕ) (b_investment : ℕ) (profit_ratio : ℕ × ℕ) (total_months : ℕ) : ℕ :=
  let a_months := total_months
  let b_months := total_months - (b_investment / (3500 * profit_ratio.snd / profit_ratio.fst / b_investment))
  total_months - b_months

theorem b_joined_after_a (a_investment b_investment total_months : ℕ) (profit_ratio : ℕ × ℕ) (h_a_investment : a_investment = 3500)
   (h_b_investment : b_investment = 21000) (h_profit_ratio : profit_ratio = (2, 3)) : months_b_joined a_investment b_investment profit_ratio total_months = 9 := by
  sorry

end b_joined_after_a_l66_66158


namespace floor_sqrt_30_squared_eq_25_l66_66646

theorem floor_sqrt_30_squared_eq_25 (h1 : 5 < Real.sqrt 30) (h2 : Real.sqrt 30 < 6) : Int.floor (Real.sqrt 30) ^ 2 = 25 := 
by
  sorry

end floor_sqrt_30_squared_eq_25_l66_66646


namespace travel_time_l66_66950

-- Definitions: 
def speed := 20 -- speed in km/hr
def distance := 160 -- distance in km

-- Proof statement: 
theorem travel_time (s : ℕ) (d : ℕ) (h1 : s = speed) (h2 : d = distance) : 
  d / s = 8 :=
by {
  sorry
}

end travel_time_l66_66950


namespace find_ABC_plus_DE_l66_66968

theorem find_ABC_plus_DE (ABCDE : Nat) (h1 : ABCDE = 13579 * 6) : (ABCDE / 1000 + ABCDE % 1000 % 100) = 888 :=
by
  sorry

end find_ABC_plus_DE_l66_66968


namespace calculate_expression_l66_66548

variable {a : ℝ}

theorem calculate_expression (h₁ : a ≠ 0) (h₂ : a ≠ 1) :
  (a - 1 / a) / ((a - 1) / a) = a + 1 := 
sorry

end calculate_expression_l66_66548


namespace smallest_n_multiple_of_7_l66_66592

theorem smallest_n_multiple_of_7 (x y n : ℤ) (h1 : x + 2 ≡ 0 [ZMOD 7]) (h2 : y - 2 ≡ 0 [ZMOD 7]) :
  x^2 + x * y + y^2 + n ≡ 0 [ZMOD 7] → n = 3 :=
by
  sorry

end smallest_n_multiple_of_7_l66_66592


namespace target_runs_is_282_l66_66837

-- Define the conditions
def run_rate_first_10_overs : ℝ := 3.2
def overs_first_segment : ℝ := 10
def run_rate_remaining_20_overs : ℝ := 12.5
def overs_second_segment : ℝ := 20

-- Define the calculation of runs in the first 10 overs
def runs_first_segment : ℝ := run_rate_first_10_overs * overs_first_segment

-- Define the calculation of runs in the remaining 20 overs
def runs_second_segment : ℝ := run_rate_remaining_20_overs * overs_second_segment

-- Define the target runs
def target_runs : ℝ := runs_first_segment + runs_second_segment

-- State the theorem
theorem target_runs_is_282 : target_runs = 282 :=
by
  -- This is where the proof would go, but it is omitted.
  sorry

end target_runs_is_282_l66_66837


namespace parabola_chord_length_l66_66008

theorem parabola_chord_length (x₁ x₂ : ℝ) (y₁ y₂ : ℝ) 
(h1 : y₁^2 = 4 * x₁) 
(h2 : y₂^2 = 4 * x₂) 
(h3 : x₁ + x₂ = 6) : 
|y₁ - y₂| = 8 :=
sorry

end parabola_chord_length_l66_66008


namespace sum_of_reciprocals_of_factors_of_13_l66_66878

theorem sum_of_reciprocals_of_factors_of_13 : 
  (1 : ℚ) + (1 / 13) = 14 / 13 :=
by {
  sorry
}

end sum_of_reciprocals_of_factors_of_13_l66_66878


namespace multiplication_identity_l66_66660

theorem multiplication_identity (x y : ℝ) : 
  (2*x^3 - 5*y^2) * (4*x^6 + 10*x^3*y^2 + 25*y^4) = 8*x^9 - 125*y^6 := 
by
  sorry

end multiplication_identity_l66_66660


namespace words_per_minute_after_break_l66_66831

variable (w : ℕ)

theorem words_per_minute_after_break (h : 10 * 5 - (w * 5) = 10) : w = 8 := by
  sorry

end words_per_minute_after_break_l66_66831


namespace power_function_at_4_l66_66901

noncomputable def power_function (α : ℝ) (x : ℝ) : ℝ := x^α

theorem power_function_at_4 {α : ℝ} :
  power_function α 2 = (Real.sqrt 2) / 2 →
  α = -1/2 →
  power_function α 4 = 1 / 2 :=
by
  intros h1 h2
  rw [h2, power_function]
  sorry

end power_function_at_4_l66_66901


namespace area_of_sector_l66_66217

theorem area_of_sector {R θ: ℝ} (hR: R = 2) (hθ: θ = (2 * Real.pi) / 3) :
  (1 / 2) * R^2 * θ = (4 / 3) * Real.pi :=
by
  simp [hR, hθ]
  norm_num
  linarith

end area_of_sector_l66_66217


namespace least_number_with_remainder_l66_66224

theorem least_number_with_remainder (x : ℕ) :
  (x % 6 = 4) ∧ (x % 7 = 4) ∧ (x % 9 = 4) ∧ (x % 18 = 4) ↔ x = 130 :=
by
  sorry

end least_number_with_remainder_l66_66224


namespace card_game_fairness_l66_66089

theorem card_game_fairness :
  let deck_size := 52
  let aces := 2
  let total_pairings := Nat.choose deck_size aces  -- Number of ways to choose 2 positions from 52
  let tie_cases := deck_size - 1                  -- Number of ways for consecutive pairs
  let non_tie_outcomes := total_pairings - tie_cases
  non_tie_outcomes / 2 = non_tie_outcomes / 2
:= sorry

end card_game_fairness_l66_66089


namespace volume_of_ABDH_is_4_3_l66_66991

-- Define the vertices of the cube
def A : (ℝ × ℝ × ℝ) := (0, 0, 0)
def B : (ℝ × ℝ × ℝ) := (2, 0, 0)
def D : (ℝ × ℝ × ℝ) := (0, 2, 0)
def H : (ℝ × ℝ × ℝ) := (0, 0, 2)

-- Function to calculate the volume of the pyramid
noncomputable def volume_of_pyramid (A B D H : ℝ × ℝ × ℝ) : ℝ :=
  (1 / 3) * (1 / 2) * 2 * 2 * 2

-- Theorem stating the volume of the pyramid ABDH is 4/3 cubic units
theorem volume_of_ABDH_is_4_3 : volume_of_pyramid A B D H = 4 / 3 := by
  sorry

end volume_of_ABDH_is_4_3_l66_66991


namespace number_of_freshmen_to_sample_l66_66791

-- Define parameters
def total_students : ℕ := 900
def sample_size : ℕ := 45
def freshmen_count : ℕ := 400
def sophomores_count : ℕ := 300
def juniors_count : ℕ := 200

-- Define the stratified sampling calculation
def stratified_sampling_calculation (group_size : ℕ) (total_size : ℕ) (sample_size : ℕ) : ℕ :=
  (group_size * sample_size) / total_size

-- Theorem stating that the number of freshmen to be sampled is 20
theorem number_of_freshmen_to_sample : stratified_sampling_calculation freshmen_count total_students sample_size = 20 := by
  sorry

end number_of_freshmen_to_sample_l66_66791


namespace already_installed_windows_l66_66427

-- Definitions based on given conditions
def total_windows : ℕ := 9
def hours_per_window : ℕ := 6
def remaining_hours : ℕ := 18

-- Main statement to prove
theorem already_installed_windows : (total_windows - remaining_hours / hours_per_window) = 6 :=
by
  -- To prove: total_windows - (remaining_hours / hours_per_window) = 6
  -- This step is intentionally left incomplete (proof to be filled in by the user)
  sorry

end already_installed_windows_l66_66427


namespace isosceles_triangle_base_length_l66_66773

theorem isosceles_triangle_base_length (a b c : ℝ) (h₀ : a = 5) (h₁ : b = 5) (h₂ : a + b + c = 17) : c = 7 :=
by
  -- proof would go here
  sorry

end isosceles_triangle_base_length_l66_66773


namespace unsolved_problems_exist_l66_66355

noncomputable def main_theorem: Prop :=
  ∃ (P : Prop), ¬(P = true) ∧ ¬(P = false)

theorem unsolved_problems_exist : main_theorem :=
sorry

end unsolved_problems_exist_l66_66355


namespace primes_sum_product_composite_l66_66630

theorem primes_sum_product_composite {p q r : ℕ} (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r) (hdistinct_pq : p ≠ q) (hdistinct_pr : p ≠ r) (hdistinct_qr : q ≠ r) :
  ¬ Nat.Prime (p + q + r + p * q * r) :=
by
  sorry

end primes_sum_product_composite_l66_66630


namespace cuboid_surface_area_l66_66232

-- Definition of the problem with given conditions and the statement we need to prove.
theorem cuboid_surface_area (h l w: ℝ) (H1: 4 * (2 * h) + 4 * (2 * h) + 4 * h = 100)
                            (H2: l = 2 * h)
                            (H3: w = 2 * h) :
                            (2 * (l * w + l * h + w * h) = 400) :=
by
  sorry

end cuboid_surface_area_l66_66232


namespace evaluate_expression_l66_66617

theorem evaluate_expression:
  (-2)^2002 + (-1)^2003 + 2^2004 + (-1)^2005 = 3 * 2^2002 - 2 :=
by
  sorry

end evaluate_expression_l66_66617


namespace conic_section_eccentricities_cubic_l66_66547

theorem conic_section_eccentricities_cubic : 
  ∃ (e1 e2 e3 : ℝ), 
    (e1 = 1) ∧ 
    (0 < e2 ∧ e2 < 1) ∧ 
    (e3 > 1) ∧ 
    2 * e1^3 - 7 * e1^2 + 7 * e1 - 2 = 0 ∧
    2 * e2^3 - 7 * e2^2 + 7 * e2 - 2 = 0 ∧
    2 * e3^3 - 7 * e3^2 + 7 * e3 - 2 = 0 := 
by
  sorry

end conic_section_eccentricities_cubic_l66_66547


namespace lcm_23_46_827_l66_66859

theorem lcm_23_46_827 :
  (23 * 46 * 827) / gcd (23 * 2) 827 = 38042 := by
  sorry

end lcm_23_46_827_l66_66859


namespace angle_in_triangle_l66_66783

theorem angle_in_triangle (A B C x : ℝ) (hA : A = 40)
    (hB : B = 3 * x) (hC : C = x) (h_sum : A + B + C = 180) : x = 35 :=
by
  sorry

end angle_in_triangle_l66_66783


namespace john_ate_half_package_l66_66245

def fraction_of_package_john_ate (servings : ℕ) (calories_per_serving : ℕ) (calories_consumed : ℕ) : ℚ :=
  calories_consumed / (servings * calories_per_serving : ℚ)

theorem john_ate_half_package (servings : ℕ) (calories_per_serving : ℕ) (calories_consumed : ℕ) 
    (h_servings : servings = 3) (h_calories_per_serving : calories_per_serving = 120) (h_calories_consumed : calories_consumed = 180) :
    fraction_of_package_john_ate servings calories_per_serving calories_consumed = 1 / 2 :=
by
  -- Replace the actual proof with sorry to ensure the statement compiles.
  sorry

end john_ate_half_package_l66_66245


namespace total_employees_l66_66350

-- Defining the number of part-time and full-time employees
def p : ℕ := 2041
def f : ℕ := 63093

-- Statement that the total number of employees is the sum of part-time and full-time employees
theorem total_employees : p + f = 65134 :=
by
  -- Use Lean's built-in arithmetic to calculate the sum
  rfl

end total_employees_l66_66350


namespace maria_sandwich_count_l66_66551

open Nat

noncomputable def numberOfSandwiches (meat_choices cheese_choices topping_choices : Nat) :=
  (choose meat_choices 2) * (choose cheese_choices 2) * (choose topping_choices 2)

theorem maria_sandwich_count : numberOfSandwiches 12 11 8 = 101640 := by
  sorry

end maria_sandwich_count_l66_66551


namespace negation_of_exists_sin_gt_one_equiv_forall_sin_le_one_l66_66571

open Real

theorem negation_of_exists_sin_gt_one_equiv_forall_sin_le_one :
  (¬ (∃ x : ℝ, sin x > 1)) ↔ (∀ x : ℝ, sin x ≤ 1) :=
sorry

end negation_of_exists_sin_gt_one_equiv_forall_sin_le_one_l66_66571


namespace complex_number_solution_l66_66301

theorem complex_number_solution (z : ℂ) (i : ℂ) (hi : i * i = -1) (h : i * z = 1) : z = -i :=
by
  -- Mathematical proof will be here
  sorry

end complex_number_solution_l66_66301


namespace number_of_intersections_l66_66258

theorem number_of_intersections : 
  (∃ p : ℝ × ℝ, p.1^2 + 9 * p.2^2 = 9 ∧ 9 * p.1^2 + p.2^2 = 1) 
  ∧ (∃! p₁ p₂ : ℝ × ℝ, p₁ ≠ p₂ ∧ p₁.1^2 + 9 * p₁.2^2 = 9 ∧ 9 * p₁.1^2 + p₁.2^2 = 1 ∧
    p₂.1^2 + 9 * p₂.2^2 = 9 ∧ 9 * p₂.1^2 + p₂.2^2 = 1) :=
by
  -- The proof will be here
  sorry

end number_of_intersections_l66_66258


namespace total_value_of_coins_is_correct_l66_66999

-- Definitions for the problem conditions
def number_of_dimes : ℕ := 22
def number_of_quarters : ℕ := 10
def value_of_dime : ℝ := 0.10
def value_of_quarter : ℝ := 0.25
def total_value_of_dimes : ℝ := number_of_dimes * value_of_dime
def total_value_of_quarters : ℝ := number_of_quarters * value_of_quarter
def total_value : ℝ := total_value_of_dimes + total_value_of_quarters

-- Theorem statement
theorem total_value_of_coins_is_correct : total_value = 4.70 := sorry

end total_value_of_coins_is_correct_l66_66999


namespace geo_seq_sum_neg_six_l66_66649

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
∃ (a₁ q : ℝ), q ≠ 0 ∧ ∀ n, a n = a₁ * q^n

theorem geo_seq_sum_neg_six
  (a : ℕ → ℝ)
  (hgeom : geometric_sequence a)
  (ha_neg : a 1 < 0)
  (h_condition : a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 36) :
  a 3 + a 5 = -6 :=
  sorry

end geo_seq_sum_neg_six_l66_66649


namespace room_length_perimeter_ratio_l66_66335

theorem room_length_perimeter_ratio :
  ∀ (L W : ℕ), L = 19 → W = 11 → (L : ℚ) / (2 * (L + W)) = 19 / 60 := by
  intros L W hL hW
  sorry

end room_length_perimeter_ratio_l66_66335


namespace problem_solution_l66_66666

def equal_group_B : Prop :=
  (-2)^3 = -(2^3)

theorem problem_solution : equal_group_B := by
  sorry

end problem_solution_l66_66666


namespace find_cos_alpha_l66_66239

theorem find_cos_alpha (α : ℝ) (h0 : 0 ≤ α ∧ α ≤ π / 2) (h1 : Real.sin (α - π / 6) = 3 / 5) : 
  Real.cos α = (4 * Real.sqrt 3 - 3) / 10 :=
sorry

end find_cos_alpha_l66_66239


namespace digits_sum_is_15_l66_66223

theorem digits_sum_is_15 (f o g : ℕ) (h1 : f * 100 + o * 10 + g = 366) (h2 : 4 * (f * 100 + o * 10 + g) = 1464) (h3 : f < 10 ∧ o < 10 ∧ g < 10) :
  f + o + g = 15 :=
sorry

end digits_sum_is_15_l66_66223


namespace total_money_l66_66048

theorem total_money (A B C : ℕ) (h1 : A + C = 200) (h2 : B + C = 330) (h3 : C = 30) : 
  A + B + C = 500 :=
by
  sorry

end total_money_l66_66048


namespace inequality_proof_l66_66806

variable (a b c : ℝ)

theorem inequality_proof (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
    (a / Real.sqrt (a^2 + 8 * b * c)) +
    (b / Real.sqrt (b^2 + 8 * a * c)) +
    (c / Real.sqrt (c^2 + 8 * a * b)) ≥ 1 :=
sorry

end inequality_proof_l66_66806


namespace find_x_l66_66329

noncomputable def f (x : ℝ) : ℝ := 5 * x^3 - 3

theorem find_x (x : ℝ) : (f⁻¹ (-2) = x) → x = -43 := by
  sorry

end find_x_l66_66329


namespace rachel_speed_painting_video_time_l66_66071

theorem rachel_speed_painting_video_time :
  let num_videos := 4
  let setup_time := 1
  let cleanup_time := 1
  let painting_time_per_video := 1
  let editing_time_per_video := 1.5
  (setup_time + cleanup_time + painting_time_per_video * num_videos + editing_time_per_video * num_videos) / num_videos = 3 :=
by
  sorry

end rachel_speed_painting_video_time_l66_66071


namespace total_payroll_l66_66170

theorem total_payroll 
  (heavy_operator_pay : ℕ) 
  (laborer_pay : ℕ) 
  (total_people : ℕ) 
  (laborers : ℕ)
  (heavy_operators : ℕ)
  (total_payroll : ℕ)
  (h1: heavy_operator_pay = 140)
  (h2: laborer_pay = 90)
  (h3: total_people = 35)
  (h4: laborers = 19)
  (h5: heavy_operators = total_people - laborers)
  (h6: total_payroll = (heavy_operators * heavy_operator_pay) + (laborers * laborer_pay)) :
  total_payroll = 3950 :=
by sorry

end total_payroll_l66_66170


namespace widgets_unloaded_l66_66608
-- We import the necessary Lean library for general mathematical purposes.

-- We begin the lean statement for our problem.
theorem widgets_unloaded (n_doo n_geegaw n_widget n_yamyam : ℕ) :
  (2^n_doo) * (11^n_geegaw) * (5^n_widget) * (7^n_yamyam) = 104350400 →
  n_widget = 2 := by
  -- Placeholder for proof
  sorry

end widgets_unloaded_l66_66608


namespace original_number_is_two_over_three_l66_66515

theorem original_number_is_two_over_three (x : ℚ) (h : 1 + 1/x = 5/2) : x = 2/3 :=
sorry

end original_number_is_two_over_three_l66_66515


namespace least_number_subtracted_from_102932847_to_be_divisible_by_25_is_22_l66_66276

def least_subtrahend (n m : ℕ) (k : ℕ) : Prop :=
  (n - k) % m = 0 ∧ ∀ k' : ℕ, k' < k → (n - k') % m ≠ 0

theorem least_number_subtracted_from_102932847_to_be_divisible_by_25_is_22 :
  least_subtrahend 102932847 25 22 :=
sorry

end least_number_subtracted_from_102932847_to_be_divisible_by_25_is_22_l66_66276


namespace jackson_hermit_crabs_l66_66304

theorem jackson_hermit_crabs (H : ℕ) (total_souvenirs : ℕ) 
  (h1 : total_souvenirs = H + 3 * H + 6 * H) 
  (h2 : total_souvenirs = 450) : H = 45 :=
by {
  sorry
}

end jackson_hermit_crabs_l66_66304


namespace polynomial_divisibility_l66_66403

theorem polynomial_divisibility 
  (a b c : ℤ)
  (P : ℤ → ℤ)
  (root_condition : ∃ u v : ℤ, u * v * (u + v) = -c ∧ u * v = b) 
  (P_def : ∀ x, P x = x^3 + a * x^2 + b * x + c) :
  2 * P (-1) ∣ (P 1 + P (-1) - 2 * (1 + P 0)) :=
by
  sorry

end polynomial_divisibility_l66_66403


namespace sequence_count_l66_66388

theorem sequence_count :
  ∃ (a : ℕ → ℕ), 
    a 10 = 3 * a 1 ∧ 
    a 2 + a 8 = 2 * a 5 ∧ 
    (∀ i, 1 ≤ i ∧ i ≤ 9 → a (i + 1) = 1 + a i ∨ a (i + 1) = 2 + a i) ∧ 
    (∃ n, n = 80) :=
sorry

end sequence_count_l66_66388


namespace geometric_sequence_m_value_l66_66079

theorem geometric_sequence_m_value 
  (a : ℕ → ℝ) (q : ℝ) (m : ℕ) 
  (h1 : a 1 = 1) 
  (h2 : ∀ n, a n = a 1 * q^(n-1))
  (h3 : |q| ≠ 1) 
  (h4 : a m = a 1 * a 2 * a 3 * a 4 * a 5) : 
  m = 11 := by
  sorry

end geometric_sequence_m_value_l66_66079


namespace percentage_of_amount_l66_66623

theorem percentage_of_amount :
  (0.25 * 300) = 75 :=
by
  sorry

end percentage_of_amount_l66_66623


namespace find_x_l66_66009

theorem find_x (x y : ℤ) (hx : x > y) (hy : y > 0)
  (coins_megan : ℤ := 42)
  (coins_shana : ℤ := 35)
  (shana_win : ℕ := 2)
  (total_megan : shana_win * x + (total_races - shana_win) * y = coins_shana)
  (total_shana : (total_races - shana_win) * x + shana_win * y = coins_megan) :
  x = 4 := by
  sorry

end find_x_l66_66009


namespace certain_number_is_51_l66_66121

theorem certain_number_is_51 (G C : ℤ) 
  (h1 : G = 33) 
  (h2 : 3 * G = 2 * C - 3) : 
  C = 51 := 
by
  sorry

end certain_number_is_51_l66_66121


namespace range_of_m_l66_66130

variable {x m : ℝ}

def quadratic (x m : ℝ) : ℝ := x^2 + (m - 1) * x + (m^2 - 3 * m + 1)

def absolute_quadratic (x m : ℝ) : ℝ := abs (quadratic x m)

theorem range_of_m (h : ∀ x ∈ Set.Icc (-1 : ℝ) 0, absolute_quadratic x m ≥ absolute_quadratic (x - 1) m) :
  m = 1 ∨ m ≥ 3 :=
sorry

end range_of_m_l66_66130


namespace xy_difference_l66_66796

theorem xy_difference (x y : ℝ) (h1 : x + y = 8) (h2 : x^2 - y^2 = 24) : x - y = 3 := 
by {
    sorry
}

end xy_difference_l66_66796


namespace binomial_20_19_eq_20_l66_66275

theorem binomial_20_19_eq_20 : Nat.choose 20 19 = 20 :=
by
  sorry

end binomial_20_19_eq_20_l66_66275


namespace B_share_in_profit_l66_66919

theorem B_share_in_profit (A B C : ℝ) (total_profit : ℝ) 
    (h1 : A = 3 * B)
    (h2 : B = (2/3) * C)
    (h3 : total_profit = 6600) :
    (B / (A + B + C)) * total_profit = 1200 := 
by
  sorry

end B_share_in_profit_l66_66919


namespace max_sum_abc_l66_66768

theorem max_sum_abc (a b c : ℝ) (h1 : 1 ≤ a) (h2 : 1 ≤ b) (h3 : 1 ≤ c) 
  (h4 : a * b * c + 2 * a^2 + 2 * b^2 + 2 * c^2 + c * a - c * b - 4 * a + 4 * b - c = 28) :
  a + b + c ≤ 6 :=
sorry

end max_sum_abc_l66_66768


namespace log10_cubic_solution_l66_66159

noncomputable def log10 (x: ℝ) : ℝ := Real.log x / Real.log 10

open Real

theorem log10_cubic_solution 
  (x : ℝ) 
  (hx1 : x < 1) 
  (hx2 : (log10 x)^3 - log10 (x^4) = 640) : 
  (log10 x)^4 - log10 (x^4) = 645 := 
by 
  sorry

end log10_cubic_solution_l66_66159


namespace Q_lies_in_third_quadrant_l66_66789

theorem Q_lies_in_third_quadrant (b : ℝ) (P_in_fourth_quadrant : 2 > 0 ∧ b < 0) :
    b < 0 ∧ -2 < 0 ↔
    (b < 0 ∧ -2 < 0) :=
by
  sorry

end Q_lies_in_third_quadrant_l66_66789


namespace subtracted_number_divisible_by_5_l66_66526

theorem subtracted_number_divisible_by_5 : ∃ k : ℕ, 9671 - 1 = 5 * k :=
by
  sorry

end subtracted_number_divisible_by_5_l66_66526


namespace smallest_n_l66_66328

noncomputable def smallest_positive_integer (x y : ℤ) (h1 : (x + 1) % 7 = 0) (h2 : (y - 5) % 7 = 0) : ℕ :=
  if 3 % 7 = 0 then 7 else 7

theorem smallest_n (x y : ℤ) (h1 : (x + 1) % 7 = 0) (h2 : (y - 5) % 7 = 0) : smallest_positive_integer x y h1 h2 = 7 := 
  by
  admit

end smallest_n_l66_66328


namespace solve_digits_A_B_l66_66705

theorem solve_digits_A_B :
    ∃ (A B : ℕ), A ≠ B ∧ A < 10 ∧ B < 10 ∧ 
    (A * (10 * A + B) = 100 * B + 10 * A + A) ∧ A = 8 ∧ B = 6 :=
by
  sorry

end solve_digits_A_B_l66_66705


namespace find_a_squared_plus_b_squared_l66_66603

theorem find_a_squared_plus_b_squared 
  (a b : ℝ) 
  (h1 : a + b = 40) 
  (h2 : a * b = 104) : 
  a^2 + b^2 = 1392 := 
by 
  sorry

end find_a_squared_plus_b_squared_l66_66603


namespace area_of_R_sum_m_n_l66_66631

theorem area_of_R_sum_m_n  (s : ℕ) 
  (square_area : ℕ) 
  (rectangle1_area : ℕ)
  (rectangle2_area : ℕ) :
  square_area = 4 → rectangle1_area = 8 → rectangle2_area = 2 → s = 6 → 
  36 - (square_area + rectangle1_area + rectangle2_area) = 22 :=
by
  intros
  sorry

end area_of_R_sum_m_n_l66_66631


namespace no_right_triangle_l66_66975

theorem no_right_triangle (a b c : ℝ) (h₁ : a = Real.sqrt 3) (h₂ : b = 2) (h₃ : c = Real.sqrt 5) : 
  a^2 + b^2 ≠ c^2 :=
by
  sorry

end no_right_triangle_l66_66975


namespace find_all_a_l66_66638

def digit_sum_base_4038 (n : ℕ) : ℕ :=
  n.digits 4038 |>.sum

def is_good (n : ℕ) : Prop :=
  2019 ∣ digit_sum_base_4038 n

def is_bad (n : ℕ) : Prop :=
  ¬ is_good n

def satisfies_condition (seq : ℕ → ℕ) (a : ℝ) : Prop :=
  (∀ n, seq n ≤ a * n) ∧ ∀ n, seq n = seq (n + 1) + 1

theorem find_all_a (a : ℝ) (h1 : 1 ≤ a) :
  (∀ seq, (∀ n m, n ≠ m → seq n ≠ seq m) → satisfies_condition seq a →
    ∃ n_infinitely, is_bad (seq n_infinitely)) ↔ a < 2019 := sorry

end find_all_a_l66_66638


namespace ratio_red_to_black_l66_66143

theorem ratio_red_to_black (a b x : ℕ) (h1 : x + b = 3 * a) (h2 : x = 2 * b - 3 * a) :
  a / b = 1 / 2 := by
  sorry

end ratio_red_to_black_l66_66143


namespace smallest_AAB_l66_66954

theorem smallest_AAB (A B : ℕ) (hA : 1 ≤ A ∧ A ≤ 9) (hB : 1 ≤ B ∧ B ≤ 9) 
  (h : 10 * A + B = (110 * A + B) / 7) : 110 * A + B = 996 :=
by
  sorry

end smallest_AAB_l66_66954


namespace four_pow_four_mul_five_pow_four_l66_66369

theorem four_pow_four_mul_five_pow_four : (4 ^ 4) * (5 ^ 4) = 160000 := by
  sorry

end four_pow_four_mul_five_pow_four_l66_66369


namespace sum_of_square_areas_l66_66633

theorem sum_of_square_areas (a b : ℝ) (h1 : a + b = 3) (h2 : a * b = 1) : a^2 + b^2 = 7 :=
sorry

end sum_of_square_areas_l66_66633


namespace part1_part2_l66_66651
open Real

noncomputable def f (x : ℝ) (m : ℝ) := x^2 - m * log x
noncomputable def h (x : ℝ) (a : ℝ) := x^2 - x + a
noncomputable def k (x : ℝ) (a : ℝ) := x - 2 * log x - a

theorem part1 (x : ℝ) (m : ℝ) (h_pos_x : 1 < x) : 
  (f x m) - (h x 0) ≥ 0 → m ≤ exp 1 :=
sorry

theorem part2 (a : ℝ) :
  (∀ x, 1 ≤ x ∧ x < 2 → k x a < 0) ∧ 
  (k 2 a < 0) ∧ 
  (∀ x, 2 < x ∧ x ≤ 3 → k x a > 0) →
  2 - 2 * log 2 < a ∧ a ≤ 3 - 2 * log 3 :=
sorry

end part1_part2_l66_66651


namespace find_eighth_number_l66_66944

-- Define the given problem with the conditions
noncomputable def sum_of_sixteen_numbers := 16 * 55
noncomputable def sum_of_first_eight_numbers := 8 * 60
noncomputable def sum_of_last_eight_numbers := 8 * 45
noncomputable def sum_of_last_nine_numbers := 9 * 50
noncomputable def sum_of_first_ten_numbers := 10 * 62

-- Define what we want to prove
theorem find_eighth_number :
  (exists (x : ℕ), x = 90) →
  sum_of_first_eight_numbers = 480 →
  sum_of_last_eight_numbers = 360 →
  sum_of_last_nine_numbers = 450 →
  sum_of_first_ten_numbers = 620 →
  sum_of_sixteen_numbers = 880 →
  x = 90 :=
by sorry

end find_eighth_number_l66_66944


namespace right_triangle_area_l66_66382

/-- Given a right triangle with one leg of length 3 and the hypotenuse of length 5,
    the area of the triangle is 6. -/
theorem right_triangle_area (a b c : ℝ) (h₁ : a = 3) (h₂ : c = 5) (h₃ : c^2 = a^2 + b^2) :
  (1 / 2) * a * b = 6 := 
sorry

end right_triangle_area_l66_66382


namespace original_number_of_employees_l66_66725

theorem original_number_of_employees (E : ℝ) :
  (E - 0.125 * E) - 0.09 * (E - 0.125 * E) = 12385 → E = 15545 := 
by  -- Start the proof
  sorry  -- Placeholder for the proof, which is not required

end original_number_of_employees_l66_66725


namespace problem1_problem2_problem3_problem4_l66_66490

-- Definitions of conversion rates used in the conditions
def sq_m_to_sq_dm : Nat := 100
def hectare_to_sq_m : Nat := 10000
def sq_cm_to_sq_dm_div : Nat := 100
def sq_km_to_hectare : Nat := 100

-- The problem statement with the expected values
theorem problem1 : 3 * sq_m_to_sq_dm = 300 := by
  sorry

theorem problem2 : 2 * hectare_to_sq_m = 20000 := by
  sorry

theorem problem3 : 5000 / sq_cm_to_sq_dm_div = 50 := by
  sorry

theorem problem4 : 8 * sq_km_to_hectare = 800 := by
  sorry

end problem1_problem2_problem3_problem4_l66_66490


namespace add_pure_acid_to_obtain_final_concentration_l66_66044

   variable (x : ℝ)

   def initial_solution_volume : ℝ := 60
   def initial_acid_concentration : ℝ := 0.10
   def final_acid_concentration : ℝ := 0.15

   axiom calculate_pure_acid (x : ℝ) :
     initial_acid_concentration * initial_solution_volume + x = final_acid_concentration * (initial_solution_volume + x)

   noncomputable def pure_acid_solution : ℝ := 3/0.85

   theorem add_pure_acid_to_obtain_final_concentration :
     x = pure_acid_solution := by
     sorry
   
end add_pure_acid_to_obtain_final_concentration_l66_66044


namespace correct_bushes_needed_l66_66225

def yield_per_bush := 10
def containers_per_zucchini := 3
def zucchinis_needed := 36
def bushes_needed (yield_per_bush containers_per_zucchini zucchinis_needed : ℕ) : ℕ :=
  Nat.ceil ((zucchinis_needed * containers_per_zucchini : ℕ) / yield_per_bush)

theorem correct_bushes_needed : bushes_needed yield_per_bush containers_per_zucchini zucchinis_needed = 11 := 
by
  sorry

end correct_bushes_needed_l66_66225


namespace solution_range_l66_66398

-- Given conditions from the table
variable (a b c : ℝ)

def f (x : ℝ) : ℝ := a * x^2 + b * x + c

axiom h₁ : f a b c 1.1 = -0.59
axiom h₂ : f a b c 1.2 = 0.84
axiom h₃ : f a b c 1.3 = 2.29
axiom h₄ : f a b c 1.4 = 3.76

theorem solution_range (a b c : ℝ) : 
  ∃ x : ℝ, 1.3 < x ∧ x < 1.4 ∧ f a b c x = 3 :=
sorry

end solution_range_l66_66398


namespace transformation_composition_l66_66888

-- Define the transformations f and g
def f (m n : ℝ) : ℝ × ℝ := (m, -n)
def g (m n : ℝ) : ℝ × ℝ := (-m, -n)

-- The proof statement that we need to prove
theorem transformation_composition : g (f (-3) 2).1 (f (-3) 2).2 = (3, 2) :=
by sorry

end transformation_composition_l66_66888


namespace fraction_compare_l66_66311

theorem fraction_compare (a b c d e : ℚ) : 
  a = 0.3333333 → 
  b = 1 / (3 * 10^6) →
  ∃ x : ℚ, 
  x = 1 / 3 ∧ 
  (x > a + d ∧ 
   x = a + b ∧
   d = b ∧
   d = -1 / (3 * 10^6)) := 
  sorry

end fraction_compare_l66_66311


namespace pow_three_not_sum_of_two_squares_l66_66240

theorem pow_three_not_sum_of_two_squares (k : ℕ) (hk : 0 < k) : 
  ¬ ∃ x y : ℕ, 0 < x ∧ 0 < y ∧ x^2 + y^2 = 3^k :=
by
  sorry

end pow_three_not_sum_of_two_squares_l66_66240


namespace inverse_function_value_l66_66906

def f (x : ℝ) : ℝ := 2 * x ^ 3 - 3

theorem inverse_function_value :
  f 3 = 51 :=
by
  sorry

end inverse_function_value_l66_66906


namespace leopards_to_rabbits_ratio_l66_66691

theorem leopards_to_rabbits_ratio :
  let antelopes := 80
  let rabbits := antelopes + 34
  let hyenas := antelopes + rabbits - 42
  let wild_dogs := hyenas + 50
  let total_animals := 605
  let leopards := total_animals - antelopes - rabbits - hyenas - wild_dogs
  leopards / rabbits = 1 / 2 :=
by
  let antelopes := 80
  let rabbits := antelopes + 34
  let hyenas := antelopes + rabbits - 42
  let wild_dogs := hyenas + 50
  let total_animals := 605
  let leopards := total_animals - antelopes - rabbits - hyenas - wild_dogs
  sorry

end leopards_to_rabbits_ratio_l66_66691


namespace paintable_area_correct_l66_66020

-- Defining lengths
def bedroom_length : ℕ := 15
def bedroom_width : ℕ := 11
def bedroom_height : ℕ := 9

-- Defining the number of bedrooms
def num_bedrooms : ℕ := 4

-- Defining the total area not to be painted per bedroom
def area_not_painted_per_bedroom : ℕ := 80

-- The total wall area calculation
def total_wall_area_per_bedroom : ℕ :=
  2 * (bedroom_length * bedroom_height) + 2 * (bedroom_width * bedroom_height)

-- The paintable wall area per bedroom calculation
def paintable_area_per_bedroom : ℕ :=
  total_wall_area_per_bedroom - area_not_painted_per_bedroom

-- The total paintable area across all bedrooms calculation
def total_paintable_area : ℕ :=
  paintable_area_per_bedroom * num_bedrooms

-- The theorem statement
theorem paintable_area_correct : total_paintable_area = 1552 := by
  sorry -- Proof is omitted

end paintable_area_correct_l66_66020


namespace sum_constants_l66_66294

theorem sum_constants (a b x : ℝ) 
  (h1 : (x - a) / (x + b) = (x^2 - 50 * x + 621) / (x^2 + 75 * x - 3400))
  (h2 : x^2 - 50 * x + 621 = (x - 27) * (x - 23))
  (h3 : x^2 + 75 * x - 3400 = (x - 40) * (x + 85)) :
  a + b = 112 :=
sorry

end sum_constants_l66_66294


namespace geometric_sequence_seventh_term_l66_66063

theorem geometric_sequence_seventh_term (a r : ℝ) (ha : 0 < a) (hr : 0 < r) 
  (h4 : a * r^3 = 16) (h10 : a * r^9 = 2) : 
  a * r^6 = 2 :=
by
  sorry

end geometric_sequence_seventh_term_l66_66063


namespace proof_true_proposition_l66_66816

open Classical

def P : Prop := ∀ x : ℝ, x^2 ≥ 0
def Q : Prop := ∃ x : ℚ, x^2 ≠ 3
def true_proposition (p q : Prop) := p ∨ ¬q

theorem proof_true_proposition : P ∧ ¬Q → true_proposition P Q :=
by
  intro h
  sorry

end proof_true_proposition_l66_66816


namespace crayons_total_l66_66903

theorem crayons_total (blue red green : ℕ) 
  (h1 : red = 4 * blue) 
  (h2 : green = 2 * red) 
  (h3 : blue = 3) : 
  blue + red + green = 39 := 
by
  sorry

end crayons_total_l66_66903


namespace socks_ratio_l66_66745

theorem socks_ratio 
  (g : ℕ) -- number of pairs of green socks
  (y : ℝ) -- price per pair of green socks
  (h1 : y > 0) -- price per pair of green socks is positive
  (h2 : 3 * g * y + 3 * y = 1.2 * (9 * y + g * y)) -- swapping resulted in a 20% increase in the bill
  : 3 / g = 3 / 4 :=
by sorry

end socks_ratio_l66_66745


namespace absolute_value_inequality_l66_66492

theorem absolute_value_inequality (x : ℝ) : (|x + 1| > 3) ↔ (x > 2 ∨ x < -4) :=
by
  sorry

end absolute_value_inequality_l66_66492


namespace samantha_hike_distance_l66_66992

theorem samantha_hike_distance :
  let A : ℝ × ℝ := (0, 0)  -- Samantha's starting point
  let B := (0, 3)           -- Point after walking northward 3 miles
  let C := (5 / (2 : ℝ) * Real.sqrt 2, 3) -- Point after walking 5 miles at 45 degrees eastward
  (dist A C = Real.sqrt 86 / 2) :=
by
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (0, 3)
  let C : ℝ × ℝ := (5 / (2 : ℝ) * Real.sqrt 2, 3)
  show dist A C = Real.sqrt 86 / 2
  sorry

end samantha_hike_distance_l66_66992


namespace difference_of_squares_l66_66654

theorem difference_of_squares (a b : ℕ) (h₁ : a + b = 60) (h₂ : a - b = 14) : a^2 - b^2 = 840 := by
  sorry

end difference_of_squares_l66_66654


namespace minimize_material_l66_66007

theorem minimize_material (π V R h : ℝ) (hV : V > 0) (h_cond : π * R^2 * h = V) :
  R = h / 2 :=
sorry

end minimize_material_l66_66007


namespace min_dancers_l66_66187

theorem min_dancers (N : ℕ) (h1 : N % 4 = 0) (h2 : N % 9 = 0) (h3 : N % 10 = 0) (h4 : N > 50) : N = 180 :=
  sorry

end min_dancers_l66_66187


namespace common_ratio_is_2_l66_66587

noncomputable def arithmetic_sequence_common_ratio (a : ℕ → ℝ) (q : ℝ) : Prop :=
∀ n, 2 * (a (n + 2) - a n) = 3 * a (n + 1)

theorem common_ratio_is_2 
  (a : ℕ → ℝ) 
  (q : ℝ) 
  (h1 : ∀ n, a (n + 1) = a n * q)
  (h2 : a 1 > 0)
  (h3 : arithmetic_sequence_common_ratio a q) :
  q = 2 :=
sorry

end common_ratio_is_2_l66_66587


namespace inequality_for_all_real_l66_66037

theorem inequality_for_all_real (a b c : ℝ) : 
  a^6 + b^6 + c^6 - 3 * a^2 * b^2 * c^2 ≥ 1/2 * (a - b)^2 * (b - c)^2 * (c - a)^2 :=
by 
  sorry

end inequality_for_all_real_l66_66037


namespace interval_between_segments_l66_66989

def population_size : ℕ := 800
def sample_size : ℕ := 40

theorem interval_between_segments : population_size / sample_size = 20 :=
by
  -- Insert proof here
  sorry

end interval_between_segments_l66_66989


namespace supplementary_angle_measure_l66_66290

theorem supplementary_angle_measure (a b : ℝ) 
  (h1 : a + b = 180) 
  (h2 : a / 5 = b / 4) : b = 80 :=
by
  sorry

end supplementary_angle_measure_l66_66290


namespace simplify_expression_l66_66146

theorem simplify_expression : 4 * (14 / 5) * (20 / -42) = -4 / 15 := 
by sorry

end simplify_expression_l66_66146


namespace smallest_n_for_convex_100gon_l66_66601

def isConvexPolygon (P : List (Real × Real)) : Prop := sorry -- Assumption for polygon convexity
def canBeIntersectedByTriangles (P : List (Real × Real)) (n : ℕ) : Prop := sorry -- Assumption for intersection by n triangles

theorem smallest_n_for_convex_100gon :
  ∀ (P : List (Real × Real)),
  isConvexPolygon P →
  List.length P = 100 →
  (∀ n, canBeIntersectedByTriangles P n → n ≥ 50) ∧ canBeIntersectedByTriangles P 50 :=
sorry

end smallest_n_for_convex_100gon_l66_66601


namespace force_on_dam_l66_66314

noncomputable def calculate_force (ρ g a b h : ℝ) :=
  ρ * g * h^2 * (b / 2 - (b - a) / 3)

theorem force_on_dam :
  let ρ := 1000
  let g := 10
  let a := 6.0
  let b := 9.6
  let h := 4.0
  calculate_force ρ g a b h = 576000 :=
by sorry

end force_on_dam_l66_66314


namespace smallest_next_divisor_l66_66711

theorem smallest_next_divisor (n : ℕ) (h_even : n % 2 = 0) (h_4_digit : 1000 ≤ n ∧ n < 10000) (h_div_493 : 493 ∣ n) :
  ∃ d : ℕ, (d > 493 ∧ d ∣ n) ∧ ∀ e, (e > 493 ∧ e ∣ n) → d ≤ e ∧ d = 510 := by
  sorry

end smallest_next_divisor_l66_66711


namespace pilot_fish_speed_when_moved_away_l66_66065

/-- Conditions -/
def keanu_speed : ℕ := 20
def shark_new_speed (k : ℕ) : ℕ := 2 * k
def pilot_fish_increase_speed (k s_new : ℕ) : ℕ := k + (s_new - k) / 2

/-- The problem statement to prove -/
theorem pilot_fish_speed_when_moved_away (k : ℕ) (s_new : ℕ) (p_new : ℕ) 
  (h1 : k = 20) 
  (h2 : s_new = shark_new_speed k) 
  (h3 : p_new = pilot_fish_increase_speed k s_new) : 
  p_new = 30 :=
by
  rw [h1] at h2
  rw [h2, h1] at h3
  rw [h3]
  sorry

end pilot_fish_speed_when_moved_away_l66_66065


namespace gcd_lcm_1365_910_l66_66100

theorem gcd_lcm_1365_910 :
  gcd 1365 910 = 455 ∧ lcm 1365 910 = 2730 :=
by
  sorry

end gcd_lcm_1365_910_l66_66100


namespace frog_jumps_further_l66_66319

-- Definitions according to conditions
def grasshopper_jump : ℕ := 36
def frog_jump : ℕ := 53

-- Theorem: The frog jumped 17 inches farther than the grasshopper
theorem frog_jumps_further (g_jump f_jump : ℕ) (h1 : g_jump = grasshopper_jump) (h2 : f_jump = frog_jump) :
  f_jump - g_jump = 17 :=
by
  -- Proof is skipped in this statement
  sorry

end frog_jumps_further_l66_66319


namespace weight_of_b_l66_66607

theorem weight_of_b (A B C : ℝ)
  (h1 : A + B + C = 135)
  (h2 : A + B = 80)
  (h3 : B + C = 94) : 
  B = 39 := 
by 
  sorry

end weight_of_b_l66_66607


namespace pencils_given_away_l66_66507

-- Define the basic values and conditions
def initial_pencils : ℕ := 39
def bought_pencils : ℕ := 22
def final_pencils : ℕ := 43

-- Let x be the number of pencils Brian gave away
variable (x : ℕ)

-- State the theorem we need to prove
theorem pencils_given_away : (initial_pencils - x) + bought_pencils = final_pencils → x = 18 := by
  sorry

end pencils_given_away_l66_66507


namespace max_full_pikes_l66_66182

theorem max_full_pikes (initial_pikes : ℕ) (pike_full_condition : ℕ → Prop) (remaining_pikes : ℕ) 
  (h_initial : initial_pikes = 30)
  (h_condition : ∀ n, pike_full_condition n → n ≥ 3)
  (h_remaining : remaining_pikes ≥ 1) :
    ∃ max_full : ℕ, max_full ≤ 9 := 
sorry

end max_full_pikes_l66_66182


namespace division_of_pow_of_16_by_8_eq_2_pow_4041_l66_66070

theorem division_of_pow_of_16_by_8_eq_2_pow_4041 :
  (16^1011) / 8 = 2^4041 :=
by
  -- Assume m = 16^1011
  let m := 16^1011
  -- Then expressing m in base 2
  have h_m_base2 : m = 2^4044 := by sorry
  -- Dividing m by 8
  have h_division : m / 8 = 2^4041 := by sorry
  -- Conclusion
  exact h_division

end division_of_pow_of_16_by_8_eq_2_pow_4041_l66_66070


namespace natural_number_195_is_solution_l66_66909

-- Define the conditions
def is_odd_digit (n : ℕ) : Prop :=
  n > 0 ∧ n % 2 = 1

def all_digits_odd (n : ℕ) : Prop :=
  ∀ d : ℕ, n / 10 ^ d % 10 < 10 → is_odd_digit (n / 10 ^ d % 10)

-- Define the proof problem
theorem natural_number_195_is_solution :
  195 < 200 ∧ all_digits_odd 195 ∧ (∃ (a b : ℕ), 10 ≤ a ∧ a < 100 ∧ 10 ≤ b ∧ b < 100 ∧ a * b = 195) :=
by
  sorry

end natural_number_195_is_solution_l66_66909


namespace average_earnings_per_minute_l66_66854

theorem average_earnings_per_minute (race_duration : ℕ) (lap_distance : ℕ) (certificate_rate : ℝ) (laps_run : ℕ) :
  race_duration = 12 → 
  lap_distance = 100 → 
  certificate_rate = 3.5 → 
  laps_run = 24 → 
  ((laps_run * lap_distance / 100) * certificate_rate) / race_duration = 7 :=
by
  intros hrace_duration hlap_distance hcertificate_rate hlaps_run
  rw [hrace_duration, hlap_distance, hcertificate_rate, hlaps_run]
  sorry

end average_earnings_per_minute_l66_66854


namespace average_speed_l66_66227

def s (t : ℝ) : ℝ := 3 + t^2

theorem average_speed {t1 t2 : ℝ} (h1 : t1 = 2) (h2: t2 = 2.1) :
  (s t2 - s t1) / (t2 - t1) = 4.1 :=
by
  sorry

end average_speed_l66_66227


namespace b_investment_less_c_l66_66358

theorem b_investment_less_c (A B C : ℕ) (y : ℕ) (total_investment : ℕ) (profit : ℕ) (A_share : ℕ)
    (h1 : A + B + C = total_investment)
    (h2 : A = B + 6000)
    (h3 : C = B + y)
    (h4 : profit = 8640)
    (h5 : A_share = 3168) :
    y = 3000 :=
by
  sorry

end b_investment_less_c_l66_66358


namespace intersection_points_l66_66181

theorem intersection_points (a : ℝ) :
  (∀ x y : ℝ, (x^2 + y^2 = a^2) ↔ (y = x^2 - 2 * a)) ↔ (0 < a ∧ a < 1) :=
sorry

end intersection_points_l66_66181


namespace remainder_1234_5678_9012_div_5_l66_66740

theorem remainder_1234_5678_9012_div_5 : (1234 * 5678 * 9012) % 5 = 4 := by
  sorry

end remainder_1234_5678_9012_div_5_l66_66740


namespace anns_age_l66_66006

theorem anns_age (a b : ℕ) (h1 : a + b = 54) 
(h2 : b = a - (a - b) + (a - b)): a = 29 :=
sorry

end anns_age_l66_66006


namespace area_of_rectangle_l66_66387

-- Define the lengths in meters
def length : ℝ := 1.2
def width : ℝ := 0.5

-- Define the function to calculate the area of a rectangle
def area (l w : ℝ) : ℝ := l * w

-- Prove that the area of the rectangle with given length and width is 0.6 square meters
theorem area_of_rectangle :
  area length width = 0.6 := by
  -- This is just the statement. We omit the proof with sorry.
  sorry

end area_of_rectangle_l66_66387


namespace who_is_who_l66_66423

-- Defining the structure and terms
structure Brother :=
  (name : String)
  (has_purple_card : Bool)

-- Conditions
def first_brother := Brother.mk "Tralalya" true
def second_brother := Brother.mk "Trulalya" false

/-- Proof that the names and cards of the brothers are as stated. -/
theorem who_is_who :
  ((first_brother.name = "Tralalya" ∧ first_brother.has_purple_card = false) ∧
   (second_brother.name = "Trulalya" ∧ second_brother.has_purple_card = true)) :=
by sorry

end who_is_who_l66_66423


namespace equal_or_equal_exponents_l66_66550

theorem equal_or_equal_exponents
  (a b c p q r : ℕ)
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_pos_p : 0 < p) (h_pos_q : 0 < q) (h_pos_r : 0 < r)
  (h1 : a^p + b^q + c^r = a^q + b^r + c^p)
  (h2 : a^q + b^r + c^p = a^r + b^p + c^q) :
  a = b ∧ b = c ∧ c = a ∨ p = q ∧ q = r ∧ r = p :=
  sorry

end equal_or_equal_exponents_l66_66550


namespace obtuse_triangle_of_sin_cos_sum_l66_66595

theorem obtuse_triangle_of_sin_cos_sum
  (A : ℝ) (hA : 0 < A ∧ A < π) 
  (h_eq : Real.sin A + Real.cos A = 12 / 25) :
  π / 2 < A ∧ A < π :=
sorry

end obtuse_triangle_of_sin_cos_sum_l66_66595


namespace part_i_l66_66231

theorem part_i (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : (a^2 + b^2 + c^2)^2 > 2 * (a^4 + b^4 + c^4)) : 
  a + b > c ∧ a + c > b ∧ b + c > a := sorry

end part_i_l66_66231


namespace fresh_grapes_weight_eq_l66_66784

-- Definitions of the conditions from a)
def fresh_grapes_water_percent : ℝ := 0.80
def dried_grapes_water_percent : ℝ := 0.20
def dried_grapes_weight : ℝ := 10
def fresh_grapes_non_water_percent : ℝ := 1 - fresh_grapes_water_percent
def dried_grapes_non_water_percent : ℝ := 1 - dried_grapes_water_percent

-- Proving the weight of fresh grapes
theorem fresh_grapes_weight_eq :
  let F := (dried_grapes_non_water_percent * dried_grapes_weight) / fresh_grapes_non_water_percent
  F = 40 := by
  -- The proof has been omitted
  sorry

end fresh_grapes_weight_eq_l66_66784


namespace marble_203_is_green_l66_66198

-- Define the conditions
def total_marbles : ℕ := 240
def cycle_length : ℕ := 15
def red_count : ℕ := 6
def green_count : ℕ := 5
def blue_count : ℕ := 4
def marble_pattern (n : ℕ) : String :=
  if n % cycle_length < red_count then "red"
  else if n % cycle_length < red_count + green_count then "green"
  else "blue"

-- Define the color of the 203rd marble
def marble_203 : String := marble_pattern 202

-- State the theorem
theorem marble_203_is_green : marble_203 = "green" :=
by
  sorry

end marble_203_is_green_l66_66198


namespace number_of_students_in_class_l66_66588

theorem number_of_students_in_class
  (total_stickers : ℕ) (stickers_to_friends : ℕ) (stickers_left : ℝ) (students_each : ℕ → ℝ)
  (n_friends : ℕ) (remaining_stickers : ℝ) :
  total_stickers = 300 →
  stickers_to_friends = (n_friends * (n_friends + 1)) / 2 →
  stickers_left = 7.5 →
  ∀ n, n_friends = 10 →
  remaining_stickers = total_stickers - stickers_to_friends - (students_each n_friends) * (n - n_friends - 1) →
  (∃ n : ℕ, remaining_stickers = 7.5 ∧
              total_stickers - (stickers_to_friends + (students_each (n - n_friends - 1) * (n - n_friends - 1))) = 7.5) :=
by
  sorry

end number_of_students_in_class_l66_66588


namespace snowman_volume_l66_66437

theorem snowman_volume (r1 r2 r3 : ℝ) (V1 V2 V3 : ℝ) (π : ℝ) 
  (h1 : r1 = 4) (h2 : r2 = 6) (h3 : r3 = 8) 
  (hV1 : V1 = (4/3) * π * (r1^3)) 
  (hV2 : V2 = (4/3) * π * (r2^3)) 
  (hV3 : V3 = (4/3) * π * (r3^3)) :
  V1 + V2 + V3 = (3168/3) * π :=
by 
  sorry

end snowman_volume_l66_66437


namespace num_intersection_points_l66_66393

-- Define the equations of the lines as conditions
def line1 (x y : ℝ) : Prop := 2 * y - 3 * x = 4
def line2 (x y : ℝ) : Prop := x + 3 * y = 3
def line3 (x y : ℝ) : Prop := 6 * x - 4 * y = 8

-- The theorem to prove the number of intersection points
theorem num_intersection_points :
  ∃! p : ℝ × ℝ, (line1 p.1 p.2 ∧ line2 p.1 p.2) ∨ (line2 p.1 p.2 ∧ line3 p.1 p.2) :=
sorry

end num_intersection_points_l66_66393


namespace sampling_method_is_systematic_l66_66715

-- Define the conditions of the problem
def conveyor_belt_transport : Prop := true
def inspectors_sampling_every_ten_minutes : Prop := true

-- Define what needs to be proved
theorem sampling_method_is_systematic :
  conveyor_belt_transport ∧ inspectors_sampling_every_ten_minutes → is_systematic_sampling :=
by
  sorry

-- Example definition that could be used in the proof
def is_systematic_sampling : Prop := true

end sampling_method_is_systematic_l66_66715


namespace min_cars_needed_l66_66910

theorem min_cars_needed (h1 : ∀ d ∈ Finset.range 7, ∃ s : Finset ℕ, s.card = 2 ∧ (∃ n : ℕ, 7 * (n - 10) ≥ 2 * n)) : 
  ∃ n, n ≥ 14 :=
by
  sorry

end min_cars_needed_l66_66910


namespace value_of_a_minus_b_l66_66303

theorem value_of_a_minus_b (a b : ℝ) (h1 : 2 * a + b = 7) (h2 : 2 * a - b = 1) : a - b = -1 :=
by
  sorry

end value_of_a_minus_b_l66_66303


namespace triangle_side_length_l66_66696

theorem triangle_side_length (BC : ℝ) (A : ℝ) (B : ℝ) (AB : ℝ) :
  BC = 2 → A = π / 3 → B = π / 4 → AB = (3 * Real.sqrt 2 + Real.sqrt 6) / 3 :=
by
  sorry

end triangle_side_length_l66_66696


namespace positive_integers_sum_reciprocal_l66_66833

theorem positive_integers_sum_reciprocal (a b c : ℕ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : c > 0) 
  (h_sum : a + b + c = 2010) (h_recip : (1 : ℚ)/a + (1 : ℚ)/b + (1 : ℚ)/c = 1/58) :
  (a = 1740 ∧ b = 180 ∧ c = 90) ∨ 
  (a = 1740 ∧ b = 90 ∧ c = 180) ∨ 
  (a = 180 ∧ b = 90 ∧ c = 1740) ∨ 
  (a = 180 ∧ b = 1740 ∧ c = 90) ∨ 
  (a = 90 ∧ b = 1740 ∧ c = 180) ∨ 
  (a = 90 ∧ b = 180 ∧ c = 1740) := 
sorry

end positive_integers_sum_reciprocal_l66_66833


namespace books_sold_on_friday_l66_66512

theorem books_sold_on_friday
  (total_books : ℕ)
  (books_sold_mon : ℕ)
  (books_sold_tue : ℕ)
  (books_sold_wed : ℕ)
  (books_sold_thu : ℕ)
  (pct_unsold : ℚ)
  (initial_stock : total_books = 1400)
  (sold_mon : books_sold_mon = 62)
  (sold_tue : books_sold_tue = 62)
  (sold_wed : books_sold_wed = 60)
  (sold_thu : books_sold_thu = 48)
  (percentage_unsold : pct_unsold = 0.8057142857142857) :
  total_books - (books_sold_mon + books_sold_tue + books_sold_wed + books_sold_thu + 40) = total_books * pct_unsold :=
by
  sorry

end books_sold_on_friday_l66_66512


namespace max_annual_profit_at_x_9_l66_66454

noncomputable def annual_profit (x : ℝ) : ℝ :=
if h : (0 < x ∧ x ≤ 10) then
  8.1 * x - x^3 / 30 - 10
else
  98 - 1000 / (3 * x) - 2.7 * x

theorem max_annual_profit_at_x_9 (x : ℝ) (h1 : 0 < x) (h2 : x ≤ 10) :
  annual_profit x ≤ annual_profit 9 :=
sorry

end max_annual_profit_at_x_9_l66_66454


namespace patrick_savings_ratio_l66_66988

theorem patrick_savings_ratio (S : ℕ) (bike_cost : ℕ) (lent_amt : ℕ) (remaining_amt : ℕ)
  (h1 : bike_cost = 150)
  (h2 : lent_amt = 50)
  (h3 : remaining_amt = 25)
  (h4 : S = remaining_amt + lent_amt) :
  (S / bike_cost : ℚ) = 1 / 2 := 
sorry

end patrick_savings_ratio_l66_66988


namespace part1_inequality_l66_66787

theorem part1_inequality (a b x y : ℝ) (h_nonneg_a : 0 ≤ a) (h_nonneg_b : 0 ≤ b) 
    (h_nonneg_x : 0 ≤ x) (h_nonneg_y : 0 ≤ y) (h_a_ge_x : a ≥ x) : 
    (a - x) ^ 2 + (b - y) ^ 2 ≤ (a + b - x) ^ 2 + y ^ 2 := 
by 
  sorry

end part1_inequality_l66_66787


namespace percentage_multiplication_l66_66667

theorem percentage_multiplication :
  (0.15 * 0.20 * 0.25) * 100 = 0.75 := 
by
  sorry

end percentage_multiplication_l66_66667


namespace Doug_lost_marbles_l66_66832

theorem Doug_lost_marbles (D E L : ℕ) 
    (h1 : E = D + 22) 
    (h2 : E = D - L + 30) 
    : L = 8 := by
  sorry

end Doug_lost_marbles_l66_66832


namespace solve_for_q_l66_66642

theorem solve_for_q (n m q: ℚ)
  (h1 : 3 / 4 = n / 88)
  (h2 : 3 / 4 = (m + n) / 100)
  (h3 : 3 / 4 = (q - m) / 150) :
  q = 121.5 :=
sorry

end solve_for_q_l66_66642


namespace find_x_eq_728_l66_66576

theorem find_x_eq_728 (n : ℕ) (x : ℕ) (hx : x = 9 ^ n - 1)
  (hprime_factors : ∃ (p q r : ℕ), (p ≠ q ∧ p ≠ r ∧ q ≠ r) ∧ (Nat.Prime p) ∧ (Nat.Prime q) ∧ (Nat.Prime r) ∧ (p * q * r) ∣ x)
  (h7 : 7 ∣ x) : x = 728 :=
sorry

end find_x_eq_728_l66_66576


namespace second_smallest_sum_l66_66090

theorem second_smallest_sum (a b c d : ℕ) (h1 : a ≠ b) (h2 : a ≠ c) (h3 : a ≠ d)
                           (h4 : b ≠ c) (h5 : b ≠ d) (h6 : c ≠ d)
                           (h7 : a + b + c = 180) (h8 : a + c + d = 197)
                           (h9 : b + c + d = 208) (h10 : a + b + d = 222) :
  208 ≠ 180 ∧ 208 ≠ 197 ∧ 208 ≠ 222 := 
sorry

end second_smallest_sum_l66_66090


namespace percent_apple_juice_in_blend_l66_66566

noncomputable def juice_blend_apple_percentage : ℚ :=
  let apple_juice_per_apple := 9 / 2
  let plum_juice_per_plum := 12 / 3
  let total_apple_juice := 4 * apple_juice_per_apple
  let total_plum_juice := 6 * plum_juice_per_plum
  let total_juice := total_apple_juice + total_plum_juice
  (total_apple_juice / total_juice) * 100

theorem percent_apple_juice_in_blend :
  juice_blend_apple_percentage = 43 :=
by
  sorry

end percent_apple_juice_in_blend_l66_66566


namespace average_hidden_primes_l66_66192

theorem average_hidden_primes
  (visible_card1 visible_card2 visible_card3 : ℕ)
  (hidden_card1 hidden_card2 hidden_card3 : ℕ)
  (h1 : visible_card1 = 68)
  (h2 : visible_card2 = 39)
  (h3 : visible_card3 = 57)
  (prime1 : Nat.Prime hidden_card1)
  (prime2 : Nat.Prime hidden_card2)
  (prime3 : Nat.Prime hidden_card3)
  (common_sum : ℕ)
  (h4 : visible_card1 + hidden_card1 = common_sum)
  (h5 : visible_card2 + hidden_card2 = common_sum)
  (h6 : visible_card3 + hidden_card3 = common_sum) :
  (hidden_card1 + hidden_card2 + hidden_card3) / 3 = 15 + 1/3 :=
sorry

end average_hidden_primes_l66_66192


namespace restore_original_price_l66_66014

theorem restore_original_price (original_price promotional_price : ℝ) (h₀ : original_price = 1) (h₁ : promotional_price = original_price * 0.8) : (original_price - promotional_price) / promotional_price = 0.25 :=
by sorry

end restore_original_price_l66_66014


namespace remainder_modulus_l66_66101

theorem remainder_modulus :
  (9^4 + 8^5 + 7^6 + 5^3) % 3 = 2 :=
by
  sorry

end remainder_modulus_l66_66101


namespace problem_statement_l66_66309

theorem problem_statement (w x y z : ℕ) (h : 2^w * 3^x * 5^y * 7^z = 882) : 2 * w + 3 * x + 5 * y + 7 * z = 22 :=
sorry

end problem_statement_l66_66309


namespace part_I_n_3_not_relevant_part_I_n_3_is_relevant_part_II_part_III_min_value_of_relevant_number_l66_66045

-- Part I
def is_relevant_number (n m : ℕ) : Prop :=
  ∀ {P : Finset ℕ}, (P ⊆ (Finset.range (2*n + 1)) ∧ P.card = m) →
  ∃ (a b c d : ℕ), a ∈ P ∧ b ∈ P ∧ c ∈ P ∧ d ∈ P ∧ a + b + c + d = 4*n + 1

theorem part_I_n_3_not_relevant :
  ¬ is_relevant_number 3 5 := sorry

theorem part_I_n_3_is_relevant :
  is_relevant_number 3 6 := sorry

-- Part II
theorem part_II (n m : ℕ) (h : is_relevant_number n m) : m - n - 3 ≥ 0 := sorry

-- Part III
theorem part_III_min_value_of_relevant_number (n : ℕ) : 
  ∃ m : ℕ, is_relevant_number n m ∧ ∀ k, is_relevant_number n k → m ≤ k := sorry

end part_I_n_3_not_relevant_part_I_n_3_is_relevant_part_II_part_III_min_value_of_relevant_number_l66_66045


namespace find_m_l66_66115

-- Define the conditions
def parabola_eq (m : ℝ) (x y : ℝ) : Prop := x^2 = m * y
def vertex_to_directrix_dist (d : ℝ) : Prop := d = 1 / 2

-- State the theorem
theorem find_m (m : ℝ) (x y d : ℝ) 
  (h1 : parabola_eq m x y) 
  (h2 : vertex_to_directrix_dist d) :
  m = 2 :=
by
  sorry

end find_m_l66_66115


namespace balance_balls_l66_66379

-- Define the weights of the balls as variables
variables (B R O S : ℝ)

-- Given conditions
axiom h1 : R = 2 * B
axiom h2 : O = (7 / 3) * B
axiom h3 : S = (5 / 3) * B

-- Statement to prove
theorem balance_balls :
  (5 * R + 3 * O + 4 * S) = (71 / 3) * B :=
by {
  -- The proof is omitted
  sorry
}

end balance_balls_l66_66379


namespace anna_money_ratio_l66_66976

theorem anna_money_ratio (total_money spent_furniture left_money given_to_Anna : ℕ)
  (h_total : total_money = 2000)
  (h_spent : spent_furniture = 400)
  (h_left : left_money = 400)
  (h_after_furniture : total_money - spent_furniture = given_to_Anna + left_money) :
  (given_to_Anna / left_money) = 3 :=
by
  have h1 : total_money - spent_furniture = 1600 := by sorry
  have h2 : given_to_Anna = 1200 := by sorry
  have h3 : given_to_Anna / left_money = 3 := by sorry
  exact h3

end anna_money_ratio_l66_66976


namespace correct_algorithm_description_l66_66611

def conditions_about_algorithms (desc : String) : Prop :=
  (desc = "A" → false) ∧
  (desc = "B" → false) ∧
  (desc = "C" → true) ∧
  (desc = "D" → false)

theorem correct_algorithm_description : ∃ desc : String, 
  conditions_about_algorithms desc :=
by
  use "C"
  unfold conditions_about_algorithms
  simp
  sorry

end correct_algorithm_description_l66_66611


namespace max_value_of_g_on_interval_l66_66315

noncomputable def g (x : ℝ) : ℝ := 5 * x^2 - 2 * x^4

theorem max_value_of_g_on_interval : ∃ x : ℝ, (0 ≤ x ∧ x ≤ Real.sqrt 2) ∧ (∀ y : ℝ, (0 ≤ y ∧ y ≤ Real.sqrt 2) → g y ≤ g x) ∧ g x = 25 / 8 := by
  sorry

end max_value_of_g_on_interval_l66_66315


namespace triangle_area_squared_l66_66809

theorem triangle_area_squared
  (R : ℝ)
  (A : ℝ)
  (AC_minus_AB : ℝ)
  (area : ℝ)
  (hx : R = 4)
  (hy : A = 60)
  (hz : AC_minus_AB = 4)
  (area_eq : area = 8 * Real.sqrt 3) :
  area^2 = 192 :=
by
  -- We include the conditions 
  have hR := hx
  have hA := hy
  have hAC_AB := hz
  have harea := area_eq
  -- We will use these to construct the required proof 
  sorry

end triangle_area_squared_l66_66809


namespace A_more_likely_than_B_l66_66341

-- Define the conditions
variables (n : ℕ) (k : ℕ)
-- n is the total number of programs, k is the chosen number of programs
def total_programs : ℕ := 10
def selected_programs : ℕ := 3
-- Probability of person B correctly completing each program
def probability_B_correct : ℚ := 3/5
-- Person A can correctly complete 6 out of 10 programs
def person_A_correct : ℕ := 6

-- The probability of person B successfully completing the challenge
def probability_B_success : ℚ := (3 * (9/25) * (2/5)) + (27/125)

-- Define binomial coefficient function for easier combination calculations
def choose (n k : ℕ) : ℕ := Nat.choose n k

-- The probabilities for the number of correct programs for person A
def P_X_0 : ℚ := (choose 4 3 : ℕ) / (choose 10 3 : ℕ)
def P_X_1 : ℚ := (choose 6 1 * choose 4 2 : ℕ) / (choose 10 3 : ℕ)
def P_X_2 : ℚ := (choose 6 2 * choose 4 1 : ℕ) / (choose 10 3 : ℕ)
def P_X_3 : ℚ := (choose 6 3 : ℕ) / (choose 10 3 : ℕ)

-- The distribution and expectation of X for person A
def E_X : ℚ := 0 * P_X_0 + 1 * P_X_1 + 2 * P_X_2 + 3 * P_X_3

-- The probability of person A successfully completing the challenge
def P_A_success : ℚ := P_X_2 + P_X_3

-- Final comparisons to determine who is more likely to succeed
def compare_success : Prop := P_A_success > probability_B_success

-- Lean statement
theorem A_more_likely_than_B : compare_success := by
  sorry

end A_more_likely_than_B_l66_66341


namespace train_speed_km_per_hr_l66_66713

theorem train_speed_km_per_hr
  (train_length : ℝ) 
  (platform_length : ℝ)
  (time_seconds : ℝ) 
  (h_train_length : train_length = 470) 
  (h_platform_length : platform_length = 520) 
  (h_time_seconds : time_seconds = 64.79481641468682) :
  (train_length + platform_length) / time_seconds * 3.6 = 54.975 := 
sorry

end train_speed_km_per_hr_l66_66713


namespace total_protest_days_l66_66401

-- Definitions for the problem conditions
def first_protest_days : ℕ := 4
def second_protest_days : ℕ := first_protest_days + (first_protest_days / 4)

-- The proof statement
theorem total_protest_days : first_protest_days + second_protest_days = 9 := sorry

end total_protest_days_l66_66401


namespace function_quadrants_l66_66370

theorem function_quadrants (a b : ℝ) (h_a : a > 1) (h_b : b < -1) :
  (∀ x : ℝ, a^x + b > 0 → ∃ x1 : ℝ, a^x1 + b < 0 → ∃ x2 : ℝ, a^x2 + b < 0) :=
sorry

end function_quadrants_l66_66370


namespace boat_ratio_l66_66017

theorem boat_ratio (b c d1 d2 : ℝ) 
  (h1 : b = 20) 
  (h2 : c = 4) 
  (h3 : d1 = 4) 
  (h4 : d2 = 2) : 
  (d1 + d2) / ((d1 / (b + c)) + (d2 / (b - c))) / b = 36 / 35 :=
by 
  sorry

end boat_ratio_l66_66017


namespace walnut_trees_planted_today_l66_66194

-- Define the number of walnut trees before planting
def walnut_trees_before_planting : ℕ := 22

-- Define the number of walnut trees after planting
def walnut_trees_after_planting : ℕ := 55

-- Define a theorem to prove the number of walnut trees planted
theorem walnut_trees_planted_today : 
  walnut_trees_after_planting - walnut_trees_before_planting = 33 :=
by
  -- The proof will be inserted here.
  sorry

end walnut_trees_planted_today_l66_66194


namespace four_hash_two_equals_forty_l66_66682

def hash_op (a b : ℕ) : ℤ := (a^2 + b^2) * (a - b)

theorem four_hash_two_equals_forty : hash_op 4 2 = 40 := 
by
  sorry

end four_hash_two_equals_forty_l66_66682


namespace value_of_r_when_n_is_3_l66_66154

def r (s : ℕ) : ℕ := 4^s - 2 * s
def s (n : ℕ) : ℕ := 3^n + 2
def n : ℕ := 3

theorem value_of_r_when_n_is_3 : r (s n) = 4^29 - 58 :=
by
  sorry

end value_of_r_when_n_is_3_l66_66154


namespace total_canoes_boatsRUs_l66_66176

-- Definitions for the conditions
def initial_production := 10
def common_ratio := 3
def months := 6

-- The function to compute the total number of canoes built using the geometric sequence sum formula
noncomputable def total_canoes (a : ℕ) (r : ℕ) (n : ℕ) := a * (r^n - 1) / (r - 1)

-- Statement of the theorem
theorem total_canoes_boatsRUs : 
  total_canoes initial_production common_ratio months = 3640 :=
sorry

end total_canoes_boatsRUs_l66_66176


namespace min_value_expr_l66_66834

theorem min_value_expr (x y : ℝ) (hx : x > 1) (hy : y > 1) : (x^3 / (y - 1)) + (y^3 / (x - 1)) ≥ 24 :=
sorry

end min_value_expr_l66_66834


namespace quadratic_has_real_roots_b_3_c_1_l66_66511

theorem quadratic_has_real_roots_b_3_c_1 :
  ∃ x₁ x₂ : ℝ, (∀ x : ℝ, x * x + 3 * x + 1 = 0 ↔ x = x₁ ∨ x = x₂) ∧
  x₁ = (-3 + Real.sqrt 5) / 2 ∧
  x₂ = (-3 - Real.sqrt 5) / 2 :=
by
  sorry

end quadratic_has_real_roots_b_3_c_1_l66_66511


namespace x_interval_l66_66689

theorem x_interval (x : ℝ) (h1 : 1 / x < 3) (h2 : 1 / x > -4) (h3 : 2 * x - 1 > 0) : x > 1 / 2 := 
sorry

end x_interval_l66_66689


namespace initial_amount_l66_66144

theorem initial_amount (spent_sweets friends_each left initial : ℝ) 
  (h1 : spent_sweets = 3.25) (h2 : friends_each = 2.20) (h3 : left = 2.45) :
  initial = spent_sweets + (friends_each * 2) + left :=
by
  sorry

end initial_amount_l66_66144


namespace size_of_first_file_l66_66431

theorem size_of_first_file (internet_speed_mbps : ℝ) (time_hours : ℝ) (file2_mbps : ℝ) (file3_mbps : ℝ) (total_downloaded_mbps : ℝ) :
  internet_speed_mbps = 2 →
  time_hours = 2 →
  file2_mbps = 90 →
  file3_mbps = 70 →
  total_downloaded_mbps = internet_speed_mbps * 60 * time_hours →
  total_downloaded_mbps - (file2_mbps + file3_mbps) = 80 :=
by
  intros
  sorry

end size_of_first_file_l66_66431


namespace probability_diff_colors_l66_66708

theorem probability_diff_colors :
  let total_marbles := 24
  let prob_diff_colors := 
    (4 / 24) * (5 / 23) + 
    (4 / 24) * (12 / 23) + 
    (4 / 24) * (3 / 23) + 
    (5 / 24) * (12 / 23) + 
    (5 / 24) * (3 / 23) + 
    (12 / 24) * (3 / 23)
  prob_diff_colors = 191 / 552 :=
by sorry

end probability_diff_colors_l66_66708


namespace smallest_cubes_to_fill_box_l66_66829

theorem smallest_cubes_to_fill_box
  (L W D : ℕ)
  (hL : L = 30)
  (hW : W = 48)
  (hD : D = 12) :
  ∃ (n : ℕ), n = (L * W * D) / ((Nat.gcd (Nat.gcd L W) D) ^ 3) ∧ n = 80 := 
by
  sorry

end smallest_cubes_to_fill_box_l66_66829


namespace max_a_value_l66_66390

theorem max_a_value (a : ℝ) : 
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → -2022 ≤ (a - 1) * x^2 - (a - 1) * x + 2022 ∧ 
                                (a - 1) * x^2 - (a - 1) * x + 2022 ≤ 2022) →
  a = 16177 :=
sorry

end max_a_value_l66_66390


namespace range_of_a_l66_66031

theorem range_of_a (a x y : ℝ) (h1 : x - y = a + 3) (h2 : 2 * x + y = 5 * a) (h3 : x < y) : a < -3 :=
by
  sorry

end range_of_a_l66_66031


namespace expression_for_an_l66_66221

noncomputable def arithmetic_sequence (d : ℕ) (n : ℕ) : ℕ :=
  2 + (n - 1) * d

theorem expression_for_an (d : ℕ) (n : ℕ) 
  (h1 : d > 0)
  (h2 : (arithmetic_sequence d 1) = 2)
  (h3 : (arithmetic_sequence d 1) < (arithmetic_sequence d 2))
  (h4 : (arithmetic_sequence d 2)^2 = 2 * (arithmetic_sequence d 4)) :
  arithmetic_sequence d n = 2 * n := sorry

end expression_for_an_l66_66221


namespace find_x_l66_66448

structure Vector2D where
  x : ℝ
  y : ℝ

def vecAdd (v1 v2 : Vector2D) : Vector2D :=
  ⟨v1.x + v2.x, v1.y + v2.y⟩

def vecScale (c : ℝ) (v : Vector2D) : Vector2D :=
  ⟨c * v.x, c * v.y⟩

def areParallel (v1 v2 : Vector2D) : Prop :=
  v1.x * v2.y = v1.y * v2.x

theorem find_x (x : ℝ)
  (a : Vector2D := ⟨1, 2⟩)
  (b : Vector2D := ⟨x, 1⟩)
  (h : areParallel (vecAdd a (vecScale 2 b)) (vecAdd (vecScale 2 a) (vecScale (-2) b))) :
  x = 1 / 2 :=
by
  sorry

end find_x_l66_66448


namespace ratio_w_y_l66_66202

theorem ratio_w_y (w x y z : ℝ) 
  (h1 : w / x = 5 / 4) 
  (h2 : y / z = 3 / 2) 
  (h3 : z / x = 1 / 4) 
  (h4 : w + x + y + z = 60) : 
  w / y = 10 / 3 :=
sorry

end ratio_w_y_l66_66202


namespace diamond_evaluation_l66_66934

-- Define the diamond operation as a function using the given table
def diamond (a b : ℕ) : ℕ :=
  match (a, b) with
  | (1, 1) => 4 | (1, 2) => 1 | (1, 3) => 3 | (1, 4) => 2
  | (2, 1) => 1 | (2, 2) => 3 | (2, 3) => 2 | (2, 4) => 4
  | (3, 1) => 3 | (3, 2) => 2 | (3, 3) => 4 | (3, 4) => 1
  | (4, 1) => 2 | (4, 2) => 4 | (4, 3) => 1 | (4, 4) => 3
  | (_, _) => 0  -- default case (should not occur)

-- State the proof problem
theorem diamond_evaluation : diamond (diamond 3 1) (diamond 4 2) = 1 := by
  sorry

end diamond_evaluation_l66_66934


namespace geo_seq_fifth_term_l66_66180

theorem geo_seq_fifth_term (a : ℕ → ℝ) (q : ℝ) (h1 : q = 2) (h2 : a 3 = 3) :
  a 5 = 12 := 
sorry

end geo_seq_fifth_term_l66_66180


namespace speed_of_stream_l66_66237

-- Definitions based on the conditions
def upstream_speed (c v : ℝ) : Prop := c - v = 4
def downstream_speed (c v : ℝ) : Prop := c + v = 12

-- Main theorem to prove
theorem speed_of_stream (c v : ℝ) (h1 : upstream_speed c v) (h2 : downstream_speed c v) : v = 4 :=
by
  sorry

end speed_of_stream_l66_66237


namespace max_rectangle_area_l66_66300

theorem max_rectangle_area (l w : ℕ) (h1 : 2 * l + 2 * w = 40) : l * w ≤ 100 :=
by
  have h2 : l + w = 20 := by linarith
  -- Further steps would go here but we're just stating it
  sorry

end max_rectangle_area_l66_66300


namespace sum_of_numbers_l66_66801

def a : ℝ := 217
def b : ℝ := 2.017
def c : ℝ := 0.217
def d : ℝ := 2.0017

theorem sum_of_numbers :
  a + b + c + d = 221.2357 :=
by
  sorry

end sum_of_numbers_l66_66801


namespace fraction_identity_l66_66580

theorem fraction_identity (x y : ℚ) (h : x / y = 5 / 3) : y / (x - y) = 3 / 2 :=
by { sorry }

end fraction_identity_l66_66580


namespace man_work_m_alone_in_15_days_l66_66386

theorem man_work_m_alone_in_15_days (M : ℕ) (h1 : 1/M + 1/10 = 1/6) : M = 15 := sorry

end man_work_m_alone_in_15_days_l66_66386


namespace total_money_shared_l66_66433

theorem total_money_shared (k t : ℕ) (h1 : k = 1750) (h2 : t = 2 * k) : k + t = 5250 :=
by
  sorry

end total_money_shared_l66_66433


namespace snow_at_mrs_hilts_house_l66_66627

theorem snow_at_mrs_hilts_house
    (snow_at_school : ℕ)
    (extra_snow_at_house : ℕ) 
    (school_snow_amount : snow_at_school = 17) 
    (extra_snow_amount : extra_snow_at_house = 12) :
  snow_at_school + extra_snow_at_house = 29 := 
by
  sorry

end snow_at_mrs_hilts_house_l66_66627


namespace projectile_time_to_meet_l66_66206

theorem projectile_time_to_meet
  (d v1 v2 : ℝ)
  (hd : d = 1455)
  (hv1 : v1 = 470)
  (hv2 : v2 = 500) :
  (d / (v1 + v2)) * 60 = 90 := by
  sorry

end projectile_time_to_meet_l66_66206


namespace simple_interest_rate_l66_66694

theorem simple_interest_rate 
  (P A T : ℝ) 
  (hP : P = 900) 
  (hA : A = 950) 
  (hT : T = 5) 
  : (A - P) * 100 / (P * T) = 1.11 :=
by
  sorry

end simple_interest_rate_l66_66694


namespace minimum_volume_sum_l66_66881

section pyramid_volume

variables {R : Type*} [OrderedRing R]
variables {V : Type*} [AddCommGroup V] [Module R V]

-- Define the volumes of the pyramids
variables (V_SABR1 V_SR2P2R3Q2 V_SCDR4 : R)
variables (V_SR1P1R2Q1 V_SR3P3R4Q3 : R)

-- Given condition
axiom volume_condition : V_SR1P1R2Q1 + V_SR3P3R4Q3 = 78

-- The theorem to be proved
theorem minimum_volume_sum : 
  V_SABR1^2 + V_SR2P2R3Q2^2 + V_SCDR4^2 ≥ 2028 :=
sorry

end pyramid_volume

end minimum_volume_sum_l66_66881


namespace find_second_expression_l66_66853

theorem find_second_expression (a : ℕ) (x : ℕ) (h1 : (2 * a + 16 + x) / 2 = 69) (h2 : a = 26) : x = 70 := 
by
  sorry

end find_second_expression_l66_66853


namespace polynomial_root_exists_l66_66404

theorem polynomial_root_exists
  (P : ℝ → ℝ)
  (a1 a2 a3 b1 b2 b3 : ℝ)
  (h_nonzero : a1 ≠ 0 ∧ a2 ≠ 0 ∧ a3 ≠ 0)
  (h_eq : ∀ x : ℝ, P (a1 * x + b1) + P (a2 * x + b2) = P (a3 * x + b3)) :
  ∃ r : ℝ, P r = 0 :=
sorry

end polynomial_root_exists_l66_66404


namespace invertible_from_c_l66_66545

-- Define the function f
def f (x : ℝ) : ℝ := (x - 3)^2 - 4

-- Define the condition for c and the statement to prove
theorem invertible_from_c (c : ℝ) (h : ∀ x1 x2 : ℝ, c ≤ x1 → c ≤ x2 → f x1 = f x2 → x1 = x2) : c = 3 :=
sorry

end invertible_from_c_l66_66545


namespace total_pages_is_1200_l66_66984

theorem total_pages_is_1200 (A B : ℕ) (h1 : 24 * (A + B) = 60 * A) (h2 : B = A + 10) : (60 * A) = 1200 := by
  sorry

end total_pages_is_1200_l66_66984


namespace faster_train_speed_l66_66343

theorem faster_train_speed
  (length_per_train : ℝ)
  (speed_slower_train : ℝ)
  (passing_time_secs : ℝ)
  (speed_faster_train : ℝ) :
  length_per_train = 80 / 1000 →
  speed_slower_train = 36 →
  passing_time_secs = 36 →
  speed_faster_train = 52 :=
by
  intro h_length_per_train h_speed_slower_train h_passing_time_secs
  -- Skipped steps would go here
  sorry

end faster_train_speed_l66_66343


namespace william_napkins_l66_66678

-- Define the given conditions
variables (O A C G W : ℕ)
variables (ho: O = 10)
variables (ha: A = 2 * O)
variables (hc: C = A / 2)
variables (hg: G = 3 * C)
variables (hw: W = 15)

-- Prove the total number of napkins William has now
theorem william_napkins (O A C G W : ℕ) (ho: O = 10) (ha: A = 2 * O)
  (hc: C = A / 2) (hg: G = 3 * C) (hw: W = 15) : W + (O + A + C + G) = 85 :=
by {
  sorry
}

end william_napkins_l66_66678


namespace amount_spent_on_petrol_l66_66673

theorem amount_spent_on_petrol
    (rent milk groceries education miscellaneous savings salary petrol : ℝ)
    (h1 : rent = 5000)
    (h2 : milk = 1500)
    (h3 : groceries = 4500)
    (h4 : education = 2500)
    (h5 : miscellaneous = 2500)
    (h6 : savings = 0.10 * salary)
    (h7 : savings = 2000)
    (total_salary : salary = 20000) : petrol = 2000 := by
  sorry

end amount_spent_on_petrol_l66_66673


namespace find_numbers_l66_66098

theorem find_numbers :
  ∃ (a b c d : ℕ), 
  (a + 2 = 22) ∧ 
  (b - 2 = 22) ∧ 
  (c * 2 = 22) ∧ 
  (d / 2 = 22) ∧ 
  (a + b + c + d = 99) :=
sorry

end find_numbers_l66_66098


namespace triangle_area_eq_l66_66664

/--
Given:
1. The base of the triangle is 4 meters.
2. The height of the triangle is 5 meters.

Prove:
The area of the triangle is 10 square meters.
-/
theorem triangle_area_eq (base height : ℝ) (h_base : base = 4) (h_height : height = 5) : 
  (base * height / 2) = 10 := by
  sorry

end triangle_area_eq_l66_66664


namespace rate_percent_calculation_l66_66394

theorem rate_percent_calculation 
  (SI : ℝ) (P : ℝ) (T : ℝ) (R : ℝ) 
  (h1 : SI = 3125) 
  (h2 : P = 12500) 
  (h3 : T = 7) 
  (h4 : SI = P * R * T / 100) :
  R = 3.57 :=
by
  sorry

end rate_percent_calculation_l66_66394


namespace find_length_PB_l66_66855

-- Define the conditions of the problem
variables (AC AP PB : ℝ) (x : ℝ)

-- Condition: The length of chord AC is x
def length_AC := AC = x

-- Condition: The length of segment AP is x + 1
def length_AP := AP = x + 1

-- Statement of the theorem to prove the length of segment PB
theorem find_length_PB (h_AC : length_AC AC x) (h_AP : length_AP AP x) :
  PB = 2 * x + 1 :=
sorry

end find_length_PB_l66_66855


namespace storyteller_friends_house_number_l66_66759

theorem storyteller_friends_house_number
  (x y : ℕ)
  (htotal : 50 < x ∧ x < 500)
  (hsum : 2 * y = x * (x + 1)) :
  y = 204 :=
by
  sorry

end storyteller_friends_house_number_l66_66759


namespace repaired_shoes_lifespan_l66_66356

-- Definitions of given conditions
def cost_repair : Float := 11.50
def cost_new : Float := 28.00
def lifespan_new : Float := 2.0
def percentage_increase : Float := 21.73913043478261 / 100

-- Cost per year of new shoes
def cost_per_year_new : Float := cost_new / lifespan_new

-- Cost per year of repaired shoes
def cost_per_year_repair (T : Float) : Float := cost_repair / T

-- Theorem statement (goal)
theorem repaired_shoes_lifespan (T : Float) (h : cost_per_year_new = cost_per_year_repair T * (1 + percentage_increase)) : T = 0.6745 :=
by
  sorry

end repaired_shoes_lifespan_l66_66356


namespace factor_diff_of_squares_l66_66929

theorem factor_diff_of_squares (y : ℝ) : 25 - 16 * y^2 = (5 - 4 * y) * (5 + 4 * y) := 
sorry

end factor_diff_of_squares_l66_66929


namespace necessary_not_sufficient_l66_66331

theorem necessary_not_sufficient (x : ℝ) : (x^2 ≥ 1) ↔ (x ≥ 1 ∨ x ≤ -1) ≠ (x ≥ 1) :=
by
  sorry

end necessary_not_sufficient_l66_66331


namespace cost_of_soap_for_year_l66_66134

theorem cost_of_soap_for_year
  (months_per_bar cost_per_bar : ℕ)
  (months_in_year : ℕ)
  (h1 : months_per_bar = 2)
  (h2 : cost_per_bar = 8)
  (h3 : months_in_year = 12) :
  (months_in_year / months_per_bar) * cost_per_bar = 48 := by
  sorry

end cost_of_soap_for_year_l66_66134


namespace max_flags_l66_66908

theorem max_flags (n : ℕ) (h1 : ∀ k, n = 9 * k) (h2 : n ≤ 200)
  (h3 : ∃ m, n = 9 * m + k ∧ k ≤ 2 ∧ k + 1 ≠ 0 ∧ k - 2 ≠ 0) : n = 198 :=
by {
  sorry
}

end max_flags_l66_66908


namespace average_age_l66_66105

theorem average_age (Devin_age Eden_age mom_age : ℕ)
  (h1 : Devin_age = 12)
  (h2 : Eden_age = 2 * Devin_age)
  (h3 : mom_age = 2 * Eden_age) :
  (Devin_age + Eden_age + mom_age) / 3 = 28 := by
  sorry

end average_age_l66_66105


namespace joan_total_cents_l66_66675

-- Conditions
def quarters : ℕ := 12
def dimes : ℕ := 8
def nickels : ℕ := 15
def pennies : ℕ := 25

def value_of_quarter : ℕ := 25
def value_of_dime : ℕ := 10
def value_of_nickel : ℕ := 5
def value_of_penny : ℕ := 1

-- The problem statement
theorem joan_total_cents : 
  (quarters * value_of_quarter + dimes * value_of_dime + nickels * value_of_nickel + pennies * value_of_penny) = 480 := 
  sorry

end joan_total_cents_l66_66675


namespace cube_volume_from_surface_area_l66_66250

theorem cube_volume_from_surface_area (s : ℝ) (h : 6 * s^2 = 54) : s^3 = 27 :=
sorry

end cube_volume_from_surface_area_l66_66250


namespace exp_addition_property_l66_66655

theorem exp_addition_property (x y : ℝ) : (Real.exp (x + y)) = (Real.exp x) * (Real.exp y) := 
sorry

end exp_addition_property_l66_66655


namespace simplify_expr_l66_66738

-- Define the terms
def a : ℕ := 2 ^ 10
def b : ℕ := 5 ^ 6

-- Define the expression we need to simplify
def expr := (a * b : ℝ)^(1/3)

-- Define the simplified form
def c : ℕ := 200
def d : ℕ := 2
def simplified_expr := (c : ℝ) * (d : ℝ)^(1/3)

-- The statement we need to prove
theorem simplify_expr : expr = simplified_expr ∧ (c + d = 202) := by
  sorry

end simplify_expr_l66_66738


namespace stratified_sampling_grade10_students_l66_66308

-- Definitions based on the given problem
def total_students := 900
def grade10_students := 300
def sample_size := 45

-- Calculation of the number of Grade 10 students in the sample
theorem stratified_sampling_grade10_students : (grade10_students * sample_size) / total_students = 15 := by
  sorry

end stratified_sampling_grade10_students_l66_66308


namespace find_remaining_score_l66_66283

-- Define the problem conditions
def student_scores : List ℕ := [70, 80, 90]
def average_score : ℕ := 70

-- Define the remaining score to prove it equals 40
def remaining_score : ℕ := 40

-- The theorem statement
theorem find_remaining_score (scores : List ℕ) (avg : ℕ) (r : ℕ) 
    (h_scores : scores = [70, 80, 90]) 
    (h_avg : avg = 70) 
    (h_length : scores.length = 3) 
    (h_avg_eq : (scores.sum + r) / (scores.length + 1) = avg) 
    : r = 40 := 
by
  sorry

end find_remaining_score_l66_66283


namespace white_water_addition_l66_66561

theorem white_water_addition :
  ∃ (W H I T E A R : ℕ), 
  W ≠ H ∧ W ≠ I ∧ W ≠ T ∧ W ≠ E ∧ W ≠ A ∧ W ≠ R ∧
  H ≠ I ∧ H ≠ T ∧ H ≠ E ∧ H ≠ A ∧ H ≠ R ∧
  I ≠ T ∧ I ≠ E ∧ I ≠ A ∧ I ≠ R ∧
  T ≠ E ∧ T ≠ A ∧ T ≠ R ∧
  E ≠ A ∧ E ≠ R ∧
  A ≠ R ∧
  W = 8 ∧ I = 6 ∧ P = 1 ∧ C = 9 ∧ N = 0 ∧
  (10000 * W + 1000 * H + 100 * I + 10 * T + E) + 
  (10000 * W + 1000 * A + 100 * T + 10 * E + R) = 169069 :=
by 
  sorry

end white_water_addition_l66_66561


namespace relationship_between_T_and_S_l66_66460

variable (a b : ℝ)

def T : ℝ := a + 2 * b
def S : ℝ := a + b^2 + 1

theorem relationship_between_T_and_S : T a b ≤ S a b := by
  sorry

end relationship_between_T_and_S_l66_66460


namespace total_distance_l66_66724

theorem total_distance (x : ℝ) (h : (1/2) * (x - 1) = (1/3) * x + 1) : x = 9 := 
by 
  sorry

end total_distance_l66_66724


namespace math_problem_l66_66977

noncomputable def proof_problem (n : ℝ) (A B : ℝ) : Prop :=
  A = n^2 ∧ B = n^2 + 1 ∧ (1 * n^4 + 2 * n^2 + 3 + 2 * (n^2 + 1) + 1 = 5 * (2 * n^2 + 1)) → 
  A + B = 7 + 4 * Real.sqrt 2

theorem math_problem (n : ℝ) (A B : ℝ) :
  proof_problem n A B :=
sorry

end math_problem_l66_66977


namespace leadership_selection_ways_l66_66123

theorem leadership_selection_ways (M : ℕ) (chiefs : ℕ) (supporting_chiefs : ℕ) (officers_per_supporting_chief : ℕ) 
  (M_eq : M = 15) (chiefs_eq : chiefs = 1) (supporting_chiefs_eq : supporting_chiefs = 2) 
  (officers_eq : officers_per_supporting_chief = 3) : 
  (M * (M - 1) * (M - 2) * (Nat.choose (M - 3) officers_per_supporting_chief) * (Nat.choose (M - 6) officers_per_supporting_chief)) = 3243240 := by
  simp [M_eq, chiefs_eq, supporting_chiefs_eq, officers_eq]
  norm_num
  sorry

end leadership_selection_ways_l66_66123


namespace sum_of_ages_l66_66128

variables (Matthew Rebecca Freddy: ℕ)
variables (H1: Matthew = Rebecca + 2)
variables (H2: Matthew = Freddy - 4)
variables (H3: Freddy = 15)

theorem sum_of_ages
  (H1: Matthew = Rebecca + 2)
  (H2: Matthew = Freddy - 4)
  (H3: Freddy = 15):
  Matthew + Rebecca + Freddy = 35 :=
  sorry

end sum_of_ages_l66_66128


namespace research_development_success_l66_66668

theorem research_development_success 
  (P_A : ℝ)  -- probability of Team A successfully developing a product
  (P_B : ℝ)  -- probability of Team B successfully developing a product
  (independent : Bool)  -- independence condition (dummy for clarity)
  (h1 : P_A = 2/3)
  (h2 : P_B = 3/5) 
  (h3 : independent = true) :
  (1 - (1 - P_A) * (1 - P_B) = 13/15) :=
by
  sorry

end research_development_success_l66_66668


namespace smallest_common_multiple_5_6_l66_66679

theorem smallest_common_multiple_5_6 (n : ℕ) 
  (h_pos : 0 < n) 
  (h_5 : 5 ∣ n) 
  (h_6 : 6 ∣ n) :
  n = 30 :=
sorry

end smallest_common_multiple_5_6_l66_66679


namespace range_of_PF1_minus_PF2_l66_66360

noncomputable def ellipse_property (x0 : ℝ) (h1 : 0 < x0) (h2 : x0 < Real.sqrt 5) : Prop :=
  ∃ f : ℝ, f = (2 * Real.sqrt 5 / 5) * x0 ∧ f > 0 ∧ f < 2

theorem range_of_PF1_minus_PF2 (x0 : ℝ) (h1 : 0 < x0) (h2 : x0 < Real.sqrt 5) : 
  ellipse_property x0 h1 h2 := by
  sorry

end range_of_PF1_minus_PF2_l66_66360


namespace prime_sum_55_l66_66684

theorem prime_sum_55 (p q r s : ℕ) (hp : Prime p) (hq : Prime q) (hr : Prime r) (hs : Prime s)
  (hpqrs : p < q ∧ q < r ∧ r < s) 
  (h_eqn : 1 - (1 : ℚ)/p - (1 : ℚ)/q - (1 : ℚ)/r - (1 : ℚ)/s = 1 / (p * q * r * s)) :
  p + q + r + s = 55 := 
sorry

end prime_sum_55_l66_66684


namespace sliced_meat_cost_per_type_with_rush_shipping_l66_66619

theorem sliced_meat_cost_per_type_with_rush_shipping:
  let original_cost := 40.0
  let rush_delivery_percentage := 0.3
  let num_types := 4
  let rush_delivery_cost := rush_delivery_percentage * original_cost
  let total_cost := original_cost + rush_delivery_cost
  let cost_per_type := total_cost / num_types
  cost_per_type = 13.0 :=
by
  sorry

end sliced_meat_cost_per_type_with_rush_shipping_l66_66619


namespace donovan_lap_time_is_45_l66_66210

-- Definitions based on the conditions
def circular_track_length : ℕ := 600
def michael_lap_time : ℕ := 40
def michael_laps_to_pass_donovan : ℕ := 9

-- The theorem to prove
theorem donovan_lap_time_is_45 : ∃ D : ℕ, 8 * D = michael_laps_to_pass_donovan * michael_lap_time ∧ D = 45 := by
  sorry

end donovan_lap_time_is_45_l66_66210


namespace value_of_f_3x_minus_7_l66_66583

def f (x : ℝ) : ℝ := 3 * x + 5

theorem value_of_f_3x_minus_7 (x : ℝ) : f (3 * x - 7) = 9 * x - 16 :=
by
  -- Proof goes here
  sorry

end value_of_f_3x_minus_7_l66_66583


namespace max_value_of_sum_of_cubes_l66_66891

theorem max_value_of_sum_of_cubes (a b c d e : ℝ) 
  (h : a^2 + b^2 + c^2 + d^2 + e^2 = 5) :
  a^3 + b^3 + c^3 + d^3 + e^3 ≤ 5 * Real.sqrt 5 := by
  sorry

end max_value_of_sum_of_cubes_l66_66891


namespace find_x_minus_y_l66_66352

theorem find_x_minus_y (x y : ℝ) (hx : |x| = 4) (hy : |y| = 2) (hxy : x * y < 0) : x - y = 6 ∨ x - y = -6 :=
by sorry

end find_x_minus_y_l66_66352


namespace kindergarten_children_l66_66055

theorem kindergarten_children (x y z n : ℕ) 
  (h1 : 2 * x + 3 * y + 4 * z = n)
  (h2 : x + y + z = 26)
  : n = 24 := 
sorry

end kindergarten_children_l66_66055


namespace sample_size_correct_l66_66414

-- Define the conditions as lean variables
def total_employees := 120
def male_employees := 90
def female_sample := 9

-- Define the proof problem statement
theorem sample_size_correct : ∃ n : ℕ, (total_employees - male_employees) / total_employees = female_sample / n ∧ n = 36 := by 
  sorry

end sample_size_correct_l66_66414


namespace number_of_dimes_l66_66614

-- Definitions based on conditions
def total_coins : Nat := 28
def nickels : Nat := 4

-- Definition of the number of dimes.
def dimes : Nat := total_coins - nickels

-- Theorem statement with the expected answer
theorem number_of_dimes : dimes = 24 := by
  -- Proof is skipped with sorry
  sorry

end number_of_dimes_l66_66614


namespace find_integer_pairs_l66_66746

theorem find_integer_pairs (a b : ℤ) : 
  (∃ d : ℤ, d ≥ 2 ∧ ∀ n : ℕ, n > 0 → d ∣ (a^n + b^n + 1)) → 
  (∃ k₁ k₂ : ℤ, ((a = 2 * k₁) ∧ (b = 2 * k₂ + 1)) ∨ ((a = 3 * k₁ + 1) ∧ (b = 3 * k₂ + 1))) :=
by
  sorry

end find_integer_pairs_l66_66746


namespace no_integers_a_b_existence_no_positive_integers_a_b_c_existence_l66_66434

-- Part (a)
theorem no_integers_a_b_existence (a b : ℤ) :
  ¬(a^2 - 3 * (b^2) = 8) :=
sorry

-- Part (b)
theorem no_positive_integers_a_b_c_existence (a b c : ℕ) (ha: a > 0) (hb: b > 0) (hc: c > 0 ) :
  ¬(a^2 + b^2 = 3 * (c^2)) :=
sorry

end no_integers_a_b_existence_no_positive_integers_a_b_c_existence_l66_66434


namespace car_speed_first_hour_l66_66589

theorem car_speed_first_hour 
  (x : ℝ)  -- Speed of the car in the first hour.
  (s2 : ℝ)  -- Speed of the car in the second hour is fixed at 40 km/h.
  (avg_speed : ℝ)  -- Average speed over two hours is 65 km/h.
  (h1 : s2 = 40)  -- speed in the second hour is 40 km/h.
  (h2 : avg_speed = 65)  -- average speed is 65 km/h
  (h3 : avg_speed = (x + s2) / 2)  -- definition of average speed
  : x = 90 := 
  sorry

end car_speed_first_hour_l66_66589


namespace twice_a_plus_one_non_negative_l66_66931

theorem twice_a_plus_one_non_negative (a : ℝ) : 2 * a + 1 ≥ 0 :=
sorry

end twice_a_plus_one_non_negative_l66_66931


namespace buses_in_parking_lot_l66_66425

def initial_buses : ℕ := 7
def additional_buses : ℕ := 6
def total_buses : ℕ := initial_buses + additional_buses

theorem buses_in_parking_lot : total_buses = 13 := by
  sorry

end buses_in_parking_lot_l66_66425


namespace point_D_number_l66_66920

theorem point_D_number (x : ℝ) :
    (5 + 8 - 10 + x = -5 - 8 + 10 - x) ↔ x = -3 :=
by
  sorry

end point_D_number_l66_66920


namespace part_a_part_b_l66_66744

noncomputable def withdraw_rubles_after_one_year
  (initial_deposit : ℤ) (initial_rate : ℤ) (annual_yield : ℚ)
  (final_rate : ℤ) (conversion_commission : ℚ) (broker_commission : ℚ) : ℚ :=
  let deposit_in_dollars := initial_deposit / initial_rate
  let interest_earned := deposit_in_dollars * annual_yield
  let total_in_dollars := deposit_in_dollars + interest_earned
  let broker_fee := interest_earned * broker_commission
  let amount_after_fee := total_in_dollars - broker_fee
  let total_in_rubles := amount_after_fee * final_rate
  let conversion_fee := total_in_rubles * conversion_commission
  total_in_rubles - conversion_fee

theorem part_a
  (initial_deposit : ℤ) (initial_rate : ℤ) (annual_yield : ℚ)
  (final_rate : ℤ) (conversion_commission : ℚ) (broker_commission : ℚ) :
  withdraw_rubles_after_one_year initial_deposit initial_rate annual_yield final_rate conversion_commission broker_commission =
  16476.8 := sorry

def effective_yield (initial_rubles final_rubles : ℚ) : ℚ :=
  (final_rubles / initial_rubles - 1) * 100

theorem part_b
  (initial_deposit : ℤ) (final_rubles : ℚ) :
  effective_yield initial_deposit final_rubles = 64.77 := sorry

end part_a_part_b_l66_66744


namespace same_cost_number_of_guests_l66_66873

theorem same_cost_number_of_guests (x : ℕ) : 
  (800 + 30 * x = 500 + 35 * x) ↔ (x = 60) :=
by {
  sorry
}

end same_cost_number_of_guests_l66_66873


namespace perfect_square_condition_l66_66087

noncomputable def isPerfectSquareQuadratic (m : ℤ) (x y : ℤ) :=
  ∃ (k : ℤ), (4 * x^2 + m * x * y + 25 * y^2) = k^2

theorem perfect_square_condition (m : ℤ) :
  (∀ x y : ℤ, isPerfectSquareQuadratic m x y) → (m = 20 ∨ m = -20) :=
by
  sorry

end perfect_square_condition_l66_66087


namespace william_tickets_l66_66111

theorem william_tickets (initial_tickets final_tickets : ℕ) (h1 : initial_tickets = 15) (h2 : final_tickets = 18) : 
  final_tickets - initial_tickets = 3 := 
by
  sorry

end william_tickets_l66_66111


namespace jessica_and_sibling_age_l66_66252

theorem jessica_and_sibling_age
  (J M S : ℕ)
  (h1 : J = M / 2)
  (h2 : M + 10 = 70)
  (h3 : S = J + ((70 - M) / 2)) :
  J = 40 ∧ S = 45 :=
by
  sorry

end jessica_and_sibling_age_l66_66252


namespace arithmetic_sequence_general_term_l66_66125

theorem arithmetic_sequence_general_term (S : ℕ → ℤ) (a : ℕ → ℤ) :
  (∀ n, S n = 3 * n^2 + 2 * n) →
  a 1 = S 1 ∧ (∀ n ≥ 2, a n = S n - S (n - 1)) →
  ∀ n, a n = 6 * n - 1 :=
by
  sorry

end arithmetic_sequence_general_term_l66_66125


namespace find_special_integers_l66_66609

theorem find_special_integers (n : ℕ) (h : n > 1) :
  (∀ d, d ∣ n ∧ d > 1 → ∃ a r, a > 0 ∧ r > 1 ∧ d = a^r + 1) ↔ (n = 10 ∨ ∃ a, a > 0 ∧ n = a^2 + 1) :=
by
  sorry

end find_special_integers_l66_66609


namespace calories_left_for_dinner_l66_66163

def breakfast_calories : ℝ := 353
def lunch_calories : ℝ := 885
def snack_calories : ℝ := 130
def daily_limit : ℝ := 2200

def total_consumed : ℝ :=
  breakfast_calories + lunch_calories + snack_calories

theorem calories_left_for_dinner : daily_limit - total_consumed = 832 :=
by
  sorry

end calories_left_for_dinner_l66_66163


namespace no_rational_root_l66_66297

theorem no_rational_root (x : ℚ) : 3 * x^4 - 2 * x^3 - 8 * x^2 + x + 1 ≠ 0 := 
by
  sorry

end no_rational_root_l66_66297


namespace num_cages_l66_66222

-- Define the conditions as given
def parrots_per_cage : ℕ := 8
def parakeets_per_cage : ℕ := 2
def total_birds_in_store : ℕ := 40

-- Prove that the number of bird cages is 4
theorem num_cages (x : ℕ) (h : 10 * x = total_birds_in_store) : x = 4 :=
sorry

end num_cages_l66_66222


namespace number_of_teams_l66_66495

-- Given the conditions and the required proof problem
theorem number_of_teams (n : ℕ) (h : (n * (n - 1)) / 2 = 28) : n = 8 := by
  sorry

end number_of_teams_l66_66495


namespace new_person_weight_l66_66640

theorem new_person_weight
  (initial_avg_weight : ℝ := 57)
  (num_people : ℕ := 8)
  (weight_to_replace : ℝ := 55)
  (weight_increase_first : ℝ := 1.5)
  (weight_increase_second : ℝ := 2)
  (weight_increase_third : ℝ := 2.5)
  (weight_increase_fourth : ℝ := 3)
  (weight_increase_fifth : ℝ := 3.5)
  (weight_increase_sixth : ℝ := 4)
  (weight_increase_seventh : ℝ := 4.5) :
  ∃ x : ℝ, x = 67 :=
by
  sorry

end new_person_weight_l66_66640


namespace total_cost_pants_and_belt_l66_66876

theorem total_cost_pants_and_belt (P B : ℝ) 
  (hP : P = 34.0) 
  (hCondition : P = B - 2.93) : 
  P + B = 70.93 :=
by
  -- Placeholder for proof
  sorry

end total_cost_pants_and_belt_l66_66876


namespace mean_temperature_l66_66011

def temperatures : List Int := [-8, -3, -3, -6, 2, 4, 1]

theorem mean_temperature :
  (temperatures.sum / temperatures.length : Int) = -2 := by
  sorry

end mean_temperature_l66_66011


namespace contractor_pays_male_worker_rs_35_l66_66803

theorem contractor_pays_male_worker_rs_35
  (num_male_workers : ℕ)
  (num_female_workers : ℕ)
  (num_child_workers : ℕ)
  (female_worker_wage : ℕ)
  (child_worker_wage : ℕ)
  (average_wage_per_day : ℕ)
  (total_workers : ℕ := num_male_workers + num_female_workers + num_child_workers)
  (total_wage : ℕ := average_wage_per_day * total_workers)
  (total_female_wage : ℕ := num_female_workers * female_worker_wage)
  (total_child_wage : ℕ := num_child_workers * child_worker_wage)
  (total_male_wage : ℕ := total_wage - total_female_wage - total_child_wage) :
  num_male_workers = 20 →
  num_female_workers = 15 →
  num_child_workers = 5 →
  female_worker_wage = 20 →
  child_worker_wage = 8 →
  average_wage_per_day = 26 →
  total_male_wage / num_male_workers = 35 :=
by
  intros h20 h15 h5 h20w h8w h26
  sorry

end contractor_pays_male_worker_rs_35_l66_66803


namespace prank_people_combinations_l66_66517

theorem prank_people_combinations (Monday Tuesday Wednesday Thursday Friday : ℕ) 
  (hMonday : Monday = 2)
  (hTuesday : Tuesday = 3)
  (hWednesday : Wednesday = 6)
  (hThursday : Thursday = 4)
  (hFriday : Friday = 3) :
  Monday * Tuesday * Wednesday * Thursday * Friday = 432 :=
  by sorry

end prank_people_combinations_l66_66517


namespace compound_cost_correct_l66_66108

noncomputable def compound_cost_per_pound (limestone_cost shale_mix_cost : ℝ) (total_weight limestone_weight : ℝ) : ℝ :=
  let shale_mix_weight := total_weight - limestone_weight
  let total_cost := (limestone_weight * limestone_cost) + (shale_mix_weight * shale_mix_cost)
  total_cost / total_weight

theorem compound_cost_correct :
  compound_cost_per_pound 3 5 100 37.5 = 4.25 := by
  sorry

end compound_cost_correct_l66_66108


namespace probability_of_snow_at_least_once_l66_66348

/-- Probability of snow during the first week of January -/
theorem probability_of_snow_at_least_once :
  let p_no_snow_first_four_days := (3/4 : ℚ)
  let p_no_snow_next_three_days := (2/3 : ℚ)
  let p_no_snow_first_week := p_no_snow_first_four_days^4 * p_no_snow_next_three_days^3
  let p_snow_at_least_once := 1 - p_no_snow_first_week
  p_snow_at_least_once = 68359 / 100000 :=
by
  sorry

end probability_of_snow_at_least_once_l66_66348


namespace area_square_EFGH_l66_66072

theorem area_square_EFGH (AB BE : ℝ) (h : BE = 2) (h2 : AB = 10) :
  ∃ s : ℝ, (s = 8 * Real.sqrt 6 - 2) ∧ s^2 = (8 * Real.sqrt 6 - 2)^2 := by
  sorry

end area_square_EFGH_l66_66072


namespace value_corresponds_l66_66049

-- Define the problem
def certain_number (x : ℝ) : Prop :=
  0.30 * x = 120

-- State the theorem to be proved
theorem value_corresponds (x : ℝ) (h : certain_number x) : 0.40 * x = 160 :=
by
  sorry

end value_corresponds_l66_66049


namespace number_of_real_solutions_l66_66747

theorem number_of_real_solutions :
  (∃ (x : ℝ), (x ^ 2 + x - 12) ^ 2 = 81) →
  (∃! (x : ℝ), (x ^ 2 + x - 12) ^ 2 = 81) :=
sorry

end number_of_real_solutions_l66_66747


namespace problem1_problem2_l66_66883

-- Problem 1 Definition: Operation ※
def operation (m n : ℚ) : ℚ := 3 * m - n

-- Lean 4 statement: Prove 2※10 = -4
theorem problem1 : operation 2 10 = -4 := by
  sorry

-- Lean 4 statement: Prove that ※ does not satisfy the distributive law
theorem problem2 (a b c : ℚ) : 
  operation a (b + c) ≠ operation a b + operation a c := by
  sorry

end problem1_problem2_l66_66883


namespace sum_of_integers_eq_17_l66_66173

theorem sum_of_integers_eq_17 (a b : ℕ) (h1 : a * b + a + b = 87) 
  (h2 : Nat.gcd a b = 1) (h3 : a < 15) (h4 : b < 15) (h5 : Even a ∨ Even b) :
  a + b = 17 := 
sorry

end sum_of_integers_eq_17_l66_66173


namespace ratio_M_N_l66_66408

-- Definitions of M, Q and N based on the given conditions
variables (M Q P N : ℝ)
variable (h1 : M = 0.40 * Q)
variable (h2 : Q = 0.30 * P)
variable (h3 : N = 0.50 * P)

theorem ratio_M_N : M / N = 6 / 25 :=
by
  -- Proof steps would go here
  sorry

end ratio_M_N_l66_66408


namespace ellipse_general_equation_l66_66755

theorem ellipse_general_equation (x y : ℝ) (α : ℝ) (h1 : x = 5 * Real.cos α) (h2 : y = 3 * Real.sin α) :
  x^2 / 25 + y^2 / 9 = 1 :=
sorry

end ellipse_general_equation_l66_66755


namespace average_weight_of_section_A_l66_66528

theorem average_weight_of_section_A (nA nB : ℕ) (WB WC : ℝ) (WA : ℝ) :
  nA = 50 →
  nB = 40 →
  WB = 70 →
  WC = 58.89 →
  50 * WA + 40 * WB = 58.89 * 90 →
  WA = 50.002 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end average_weight_of_section_A_l66_66528


namespace smallest_nat_mul_47_last_four_digits_l66_66184

theorem smallest_nat_mul_47_last_four_digits (N : ℕ) :
  (47 * N) % 10000 = 1969 ↔ N = 8127 :=
sorry

end smallest_nat_mul_47_last_four_digits_l66_66184


namespace adam_more_apples_than_combined_l66_66558

def adam_apples : Nat := 10
def jackie_apples : Nat := 2
def michael_apples : Nat := 5

theorem adam_more_apples_than_combined : 
  adam_apples - (jackie_apples + michael_apples) = 3 :=
by
  sorry

end adam_more_apples_than_combined_l66_66558


namespace annual_income_correct_l66_66051

def investment (amount : ℕ) := 6800
def dividend_rate (rate : ℕ) := 20
def stock_price (price : ℕ) := 136
def face_value : ℕ := 100
def calculate_annual_income (amount rate price value : ℕ) : ℕ := 
  let shares := amount / price
  let annual_income_per_share := value * rate / 100
  shares * annual_income_per_share

theorem annual_income_correct : calculate_annual_income (investment 6800) (dividend_rate 20) (stock_price 136) face_value = 1000 :=
by
  sorry

end annual_income_correct_l66_66051


namespace beetle_total_distance_l66_66732

theorem beetle_total_distance (r : ℝ) (r_eq : r = 75) : (2 * r + r + r) = 300 := 
by
  sorry

end beetle_total_distance_l66_66732


namespace find_remainder_2500th_term_l66_66872

theorem find_remainder_2500th_term : 
    let seq_position (n : ℕ) := n * (n + 1) / 2 
    let n := ((1 + Int.ofNat 20000).natAbs.sqrt + 1) / 2
    let term_2500 := if seq_position n < 2500 then n + 1 else n
    (term_2500 % 7) = 1 := by 
    sorry

end find_remainder_2500th_term_l66_66872


namespace range_of_a_l66_66800

theorem range_of_a (a : ℝ) :
  (¬ ∃ x : ℝ, |x - a| + |x + 1| ≤ 2) ↔ (a < -3 ∨ a > 1) :=
by
  sorry

end range_of_a_l66_66800


namespace red_given_red_l66_66905

def p_i (i : ℕ) : ℚ := sorry
axiom lights_probs_eq : p_i 1 + p_i 2 = 2 / 3
axiom lights_probs_eq2 : p_i 1 + p_i 3 = 2 / 3
axiom green_given_green : p_i 1 / (p_i 1 + p_i 2) = 3 / 4
axiom total_prob : p_i 1 + p_i 2 + p_i 3 + p_i 4 = 1

theorem red_given_red : (p_i 4 / (p_i 3 + p_i 4)) = 1 / 2 := 
sorry

end red_given_red_l66_66905


namespace ternary_to_decimal_l66_66940

def to_decimal (ternary : Nat) : Nat :=
  match ternary with
  | 121 => 1 * 3^2 + 2 * 3^1 + 1 * 3^0
  | _ => 0

theorem ternary_to_decimal : to_decimal 121 = 16 := by
  sorry

end ternary_to_decimal_l66_66940


namespace cave_depth_l66_66972

theorem cave_depth (current_depth remaining_distance : ℕ) (h₁ : current_depth = 849) (h₂ : remaining_distance = 369) :
  current_depth + remaining_distance = 1218 :=
by
  sorry

end cave_depth_l66_66972


namespace arithmetic_sequence_eleven_term_l66_66663

theorem arithmetic_sequence_eleven_term (a1 d a11 : ℕ) (h_sum7 : 7 * (2 * a1 + 6 * d) = 154) (h_a1 : a1 = 5) :
  a11 = a1 + 10 * d → a11 = 25 :=
by
  sorry

end arithmetic_sequence_eleven_term_l66_66663


namespace solve_inequality_l66_66841

theorem solve_inequality (x : Real) : 
  (abs ((3 * x + 2) / (x - 2)) > 3) ↔ (x ∈ Set.Ioo (2 / 3) 2) := by
  sorry

end solve_inequality_l66_66841
