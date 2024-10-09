import Mathlib

namespace remainder_of_expression_l1079_107973

theorem remainder_of_expression (m : ℤ) (h : m % 9 = 3) : (3 * m + 2436) % 9 = 0 := 
by 
  sorry

end remainder_of_expression_l1079_107973


namespace divisor_is_four_l1079_107942

theorem divisor_is_four (d n : ℤ) (k j : ℤ) 
  (h1 : n % d = 3) 
  (h2 : 2 * n % d = 2): d = 4 :=
sorry

end divisor_is_four_l1079_107942


namespace cheaper_price_difference_is_75_cents_l1079_107989

noncomputable def list_price := 42.50
noncomputable def store_a_discount := 12.00
noncomputable def store_b_discount_percent := 0.30

noncomputable def store_a_price := list_price - store_a_discount
noncomputable def store_b_price := (1 - store_b_discount_percent) * list_price
noncomputable def price_difference_in_dollars := store_a_price - store_b_price
noncomputable def price_difference_in_cents := price_difference_in_dollars * 100

theorem cheaper_price_difference_is_75_cents :
  price_difference_in_cents = 75 := by
  sorry

end cheaper_price_difference_is_75_cents_l1079_107989


namespace functional_equation_solution_l1079_107902

noncomputable def f : ℕ → ℕ := sorry

theorem functional_equation_solution (f : ℕ → ℕ)
    (h : ∀ n : ℕ, f (f (f n)) + f (f n) + f n = 3 * n) :
    ∀ n : ℕ, f n = n := sorry

end functional_equation_solution_l1079_107902


namespace factorize_quadratic_l1079_107983

theorem factorize_quadratic (x : ℝ) : 2 * x^2 - 4 * x + 2 = 2 * (x - 1)^2 := 
by {
  sorry
}

end factorize_quadratic_l1079_107983


namespace area_original_is_504_l1079_107987

-- Define the sides of the three rectangles
variable (a1 b1 a2 b2 a3 b3 : ℕ)

-- Define the perimeters of the three rectangles
def P1 := 2 * (a1 + b1)
def P2 := 2 * (a2 + b2)
def P3 := 2 * (a3 + b3)

-- Define the conditions given in the problem
axiom P1_equal_P2_plus_20 : P1 = P2 + 20
axiom P2_equal_P3_plus_16 : P2 = P3 + 16

-- Define the calculation for the area of the original rectangle
def area_original := a1 * b1

-- Proof goal: the area of the original rectangle is 504
theorem area_original_is_504 : area_original = 504 := 
sorry

end area_original_is_504_l1079_107987


namespace jordan_more_novels_than_maxime_l1079_107927

theorem jordan_more_novels_than_maxime :
  let jordan_novels := 130
  let alexandre_novels := (1 / 10) * jordan_novels
  let camille_novels := 2 * alexandre_novels
  let total_novels := jordan_novels + alexandre_novels + camille_novels
  let maxime_novels := (1 / 2) * total_novels - 5
  jordan_novels - maxime_novels = 51 :=
by
  let jordan_novels := 130
  let alexandre_novels := (1 / 10) * jordan_novels
  let camille_novels := 2 * alexandre_novels
  let total_novels := jordan_novels + alexandre_novels + camille_novels
  let maxime_novels := (1 / 2) * total_novels - 5
  sorry

end jordan_more_novels_than_maxime_l1079_107927


namespace frosting_sugar_calc_l1079_107982

theorem frosting_sugar_calc (total_sugar cake_sugar : ℝ) (h1 : total_sugar = 0.8) (h2 : cake_sugar = 0.2) : 
  total_sugar - cake_sugar = 0.6 :=
by
  rw [h1, h2]
  sorry  -- Proof should go here

end frosting_sugar_calc_l1079_107982


namespace gcd_n4_plus_27_n_plus_3_l1079_107961

theorem gcd_n4_plus_27_n_plus_3 (n : ℕ) (h_pos : n > 9) : 
  gcd (n^4 + 27) (n + 3) = if n % 3 = 0 then 3 else 1 := 
by
  sorry

end gcd_n4_plus_27_n_plus_3_l1079_107961


namespace james_received_stickers_l1079_107952

theorem james_received_stickers (initial_stickers given_away final_stickers received_stickers : ℕ) 
  (h_initial : initial_stickers = 269)
  (h_given : given_away = 48)
  (h_final : final_stickers = 423)
  (h_total_before_giving_away : initial_stickers + received_stickers = given_away + final_stickers) :
  received_stickers = 202 :=
by
  sorry

end james_received_stickers_l1079_107952


namespace abs_g_eq_abs_gx_l1079_107999

noncomputable def g (x : ℝ) : ℝ :=
if -3 <= x ∧ x <= 0 then x^2 - 2 else -x + 2

noncomputable def abs_g (x : ℝ) : ℝ :=
if -3 <= x ∧ x <= -Real.sqrt 2 then x^2 - 2
else if -Real.sqrt 2 < x ∧ x <= Real.sqrt 2 then 2 - x^2
else if Real.sqrt 2 < x ∧ x <= 2 then 2 - x
else x - 2

theorem abs_g_eq_abs_gx (x : ℝ) (hx1 : -3 <= x ∧ x <= -Real.sqrt 2) 
  (hx2 : -Real.sqrt 2 < x ∧ x <= Real.sqrt 2)
  (hx3 : Real.sqrt 2 < x ∧ x <= 2)
  (hx4 : 2 < x ∧ x <= 3) :
  abs_g x = |g x| :=
by
  sorry

end abs_g_eq_abs_gx_l1079_107999


namespace B_initial_investment_l1079_107951

-- Definitions for investments and conditions
def A_init_invest : Real := 3000
def A_later_invest := 2 * A_init_invest

def A_yearly_investment := (A_init_invest * 6) + (A_later_invest * 6)

-- The amount B needs to invest for the yearly investment to be equal in the profit ratio 1:1
def B_investment (x : Real) := x * 12 

-- Definition of the proof problem
theorem B_initial_investment (x : Real) : A_yearly_investment = B_investment x → x = 4500 := 
by 
  sorry

end B_initial_investment_l1079_107951


namespace g_neg501_l1079_107931

noncomputable def g : ℝ → ℝ := sorry

axiom g_eq (x y : ℝ) : g (x * y) + 2 * x = x * g y + g x

axiom g_neg1 : g (-1) = 7

theorem g_neg501 : g (-501) = 507 :=
by
  sorry

end g_neg501_l1079_107931


namespace triangle_side_condition_angle_condition_l1079_107971

variable (a b c A B C : ℝ)

theorem triangle_side_condition (a_eq : a = 2) (b_eq : b = Real.sqrt 7) (h : a = b * Real.cos C + (Real.sqrt 3 / 3) * c * Real.sin B) :
  c = 3 :=
  sorry

theorem angle_condition (angle_eq : Real.sqrt 3 * Real.sin (2 * A - π / 6) - 2 * Real.sin (C - π / 12)^2 = 0) :
  A = π / 4 :=
  sorry

end triangle_side_condition_angle_condition_l1079_107971


namespace find_alpha_beta_sum_l1079_107906

theorem find_alpha_beta_sum
  (a : ℝ) (α β φ : ℝ)
  (h1 : 3 * Real.sin α + 4 * Real.cos α = a)
  (h2 : 3 * Real.sin β + 4 * Real.cos β = a)
  (h3 : α ≠ β)
  (h4 : 0 < α ∧ α < 2 * Real.pi)
  (h5 : 0 < β ∧ β < 2 * Real.pi)
  (hφ : φ = Real.arcsin (4/5)) :
  α + β = Real.pi - 2 * φ ∨ α + β = 3 * Real.pi - 2 * φ :=
by
  sorry

end find_alpha_beta_sum_l1079_107906


namespace determinant_example_l1079_107923

def determinant_2x2 (a b c d : ℤ) : ℤ :=
  a * d - b * c

theorem determinant_example : determinant_2x2 5 (-4) 2 3 = 23 := 
by 
  sorry

end determinant_example_l1079_107923


namespace c_is_11_years_younger_than_a_l1079_107958

variable (A B C : ℕ) (h : A + B = B + C + 11)

theorem c_is_11_years_younger_than_a (A B C : ℕ) (h : A + B = B + C + 11) : C = A - 11 := by
  sorry

end c_is_11_years_younger_than_a_l1079_107958


namespace jenna_less_than_bob_l1079_107967

def bob_amount : ℕ := 60
def phil_amount : ℕ := (1 / 3) * bob_amount
def jenna_amount : ℕ := 2 * phil_amount

theorem jenna_less_than_bob : bob_amount - jenna_amount = 20 := by
  sorry

end jenna_less_than_bob_l1079_107967


namespace trains_meeting_distance_l1079_107985

theorem trains_meeting_distance :
  ∃ D T : ℕ, (D = 20 * T) ∧ (D + 60 = 25 * T) ∧ (2 * D + 60 = 540) :=
by
  sorry

end trains_meeting_distance_l1079_107985


namespace sale_savings_l1079_107969

theorem sale_savings (price_fox : ℝ) (price_pony : ℝ) 
(discount_fox : ℝ) (discount_pony : ℝ) 
(total_discount : ℝ) (num_fox : ℕ) (num_pony : ℕ) 
(price_saved_during_sale : ℝ) :
price_fox = 15 → 
price_pony = 18 → 
num_fox = 3 → 
num_pony = 2 → 
total_discount = 22 → 
discount_pony = 15 → 
discount_fox = total_discount - discount_pony → 
price_saved_during_sale = num_fox * price_fox * (discount_fox / 100) + num_pony * price_pony * (discount_pony / 100) →
price_saved_during_sale = 8.55 := 
by sorry

end sale_savings_l1079_107969


namespace triangle_area_proof_l1079_107930

noncomputable def area_of_triangle (a b c : ℝ) (A B C : ℝ) : ℝ := 
  1 / 2 * a * c * Real.sin B

theorem triangle_area_proof (a b c : ℝ) (A B C : ℝ) (h1 : b = 3) 
  (h2 : Real.cos B = 1 / 4) 
  (h3 : Real.sin C = 2 * Real.sin A) 
  (h4 : c = 2 * a) 
  (h5 : 9 = 5 * a ^ 2 - 4 * a ^ 2 * Real.cos B): 
  area_of_triangle a b c A B C = 9 * Real.sqrt 15 / 16 :=
by 
  sorry

end triangle_area_proof_l1079_107930


namespace problem_solution_l1079_107950

-- Define the ellipse equation and foci positions.
def ellipse (x y : ℝ) : Prop := (x^2 / 3) + (y^2 / 2) = 1
def F1 : ℝ × ℝ := (-1, 0)
def F2 : ℝ × ℝ := (1, 0)

-- Define the line equation
def line (x y k : ℝ) : Prop := y = k * x + 1

-- Define the intersection points A and B
variable (A B : ℝ × ℝ)
variable (k : ℝ)

-- Define the points lie on the line and ellipse
def A_on_line := ∃ x y, A = (x, y) ∧ line x y k
def B_on_line := ∃ x y, B = (x, y) ∧ line x y k

-- Define the parallel and perpendicular conditions
def parallel (v1 v2 : ℝ × ℝ) : Prop := ∃ k, v1.1 = k * v2.1 ∧ v1.2 = k * v2.2
def perpendicular (v1 v2 : ℝ × ℝ) : Prop := v1.1 * v2.1 + v1.2 * v2.2 = 0

-- Lean theorem for the conclusions of the problem
theorem problem_solution (A_cond : A_on_line A k ∧ ellipse A.1 A.2) 
                          (B_cond : B_on_line B k ∧ ellipse B.1 B.2) :

  -- Prove these two statements
  ¬ parallel (A.1 + 1, A.2) (B.1 - 1, B.2) ∧
  ¬ perpendicular (A.1 + 1, A.2) (A.1 - 1, A.2) :=
sorry

end problem_solution_l1079_107950


namespace team_points_l1079_107911

theorem team_points (wins losses ties : ℕ) (points_per_win points_per_loss points_per_tie : ℕ) :
  wins = 9 → losses = 3 → ties = 4 → points_per_win = 2 → points_per_loss = 0 → points_per_tie = 1 →
  (points_per_win * wins + points_per_loss * losses + points_per_tie * ties = 22) :=
by
  intro h_wins h_losses h_ties h_points_per_win h_points_per_loss h_points_per_tie
  sorry

end team_points_l1079_107911


namespace number_142857_has_property_l1079_107939

noncomputable def has_desired_property (n : ℕ) : Prop :=
∀ m ∈ [1, 2, 3, 4, 5, 6], ∀ d ∈ (Nat.digits 10 (n * m)), d ∈ (Nat.digits 10 n)

theorem number_142857_has_property : has_desired_property 142857 :=
sorry

end number_142857_has_property_l1079_107939


namespace total_points_scored_l1079_107925

-- Define the variables
def games : ℕ := 10
def points_per_game : ℕ := 12

-- Formulate the proposition to prove
theorem total_points_scored : games * points_per_game = 120 :=
by
  sorry

end total_points_scored_l1079_107925


namespace coffee_equals_milk_l1079_107986

theorem coffee_equals_milk (S : ℝ) (h : 0 < S ∧ S < 1/2) :
  let initial_milk := 1 / 2
  let initial_coffee := 1 / 2
  let glass1_initial := initial_milk
  let glass2_initial := initial_coffee
  let glass2_after_first_transfer := glass2_initial + S
  let coffee_transferred_back := (S * initial_coffee) / (initial_coffee + S)
  let milk_transferred_back := (S^2) / (initial_coffee + S)
  let glass1_after_second_transfer := glass1_initial - S + milk_transferred_back
  let glass2_after_second_transfer := glass2_initial + S - coffee_transferred_back
  (glass1_initial - S + milk_transferred_back) = (glass2_initial + S - coffee_transferred_back) :=
sorry

end coffee_equals_milk_l1079_107986


namespace cos_alpha_in_second_quadrant_l1079_107959

variable (α : Real) -- Define the variable α as a Real number (angle in radians)
variable (h1 : α > π / 2 ∧ α < π) -- Condition that α is in the second quadrant
variable (h2 : Real.sin α = 2 / 3) -- Condition that sin(α) = 2/3

theorem cos_alpha_in_second_quadrant (α : Real) (h1 : α > π / 2 ∧ α < π)
  (h2 : Real.sin α = 2 / 3) : Real.cos α = - Real.sqrt (1 - (2 / 3) ^ 2) :=
by
  sorry

end cos_alpha_in_second_quadrant_l1079_107959


namespace oula_deliveries_count_l1079_107924

-- Define the conditions for the problem
def num_deliveries_Oula (O : ℕ) (T : ℕ) : Prop :=
  T = (3 / 4 : ℚ) * O ∧ (100 * O - 100 * T = 2400)

-- Define the theorem we want to prove
theorem oula_deliveries_count : ∃ (O : ℕ), ∃ (T : ℕ), num_deliveries_Oula O T ∧ O = 96 :=
sorry

end oula_deliveries_count_l1079_107924


namespace bus_A_speed_l1079_107915

-- Define the conditions
variables (v_A v_B : ℝ)
axiom equation1 : v_A - v_B = 15
axiom equation2 : v_A + v_B = 75

-- The main theorem we want to prove
theorem bus_A_speed : v_A = 45 :=
by {
  sorry
}

end bus_A_speed_l1079_107915


namespace total_revenue_from_selling_snakes_l1079_107905

-- Definitions based on conditions
def num_snakes := 3
def eggs_per_snake := 2
def standard_price := 250
def rare_multiplier := 4

-- Prove the total revenue Jake gets from selling all baby snakes is $2250
theorem total_revenue_from_selling_snakes : 
  (num_snakes * eggs_per_snake - 1) * standard_price + (standard_price * rare_multiplier) = 2250 := 
by
  sorry

end total_revenue_from_selling_snakes_l1079_107905


namespace max_x_value_l1079_107964

variables {x y : ℝ}
variables (data : list (ℝ × ℝ))
variables (linear_relation : ℝ → ℝ → Prop)

def max_y : ℝ := 10

-- Given conditions
axiom linear_data :
  (data = [(16, 11), (14, 9), (12, 8), (8, 5)]) ∧
  (∀ (p : ℝ × ℝ), p ∈ data → linear_relation p.1 p.2)

-- Prove the maximum value of x for which y ≤ max_y
theorem max_x_value (h : ∀ (x y : ℝ), linear_relation x y → y = 11 - (16 - x) / 3):
  ∀ (x : ℝ), (∃ y : ℝ, linear_relation x y) → y ≤ max_y → x ≤ 15 :=
sorry

end max_x_value_l1079_107964


namespace total_coins_last_month_l1079_107978

theorem total_coins_last_month (m s : ℝ) : 
  (100 = 1.25 * m) ∧ (100 = 0.80 * s) → m + s = 205 :=
by sorry

end total_coins_last_month_l1079_107978


namespace find_a_l1079_107996

theorem find_a (a b c : ℕ) (h1 : a + b = c) (h2 : b + c = 6) (h3 : c = 4) : a = 2 :=
by
  sorry

end find_a_l1079_107996


namespace not_black_cows_count_l1079_107988

theorem not_black_cows_count (total_cows : ℕ) (black_cows : ℕ) (h1 : total_cows = 18) (h2 : black_cows = 5 + total_cows / 2) :
  total_cows - black_cows = 4 :=
by 
  -- Insert the actual proof here
  sorry

end not_black_cows_count_l1079_107988


namespace probability_rachel_robert_in_picture_l1079_107901

theorem probability_rachel_robert_in_picture :
  let lap_rachel := 120 -- Rachel's lap time in seconds
  let lap_robert := 100 -- Robert's lap time in seconds
  let duration := 900 -- 15 minutes in seconds
  let picture_duration := 60 -- Picture duration in seconds
  let one_third_rachel := lap_rachel / 3 -- One third of Rachel's lap time
  let one_third_robert := lap_robert / 3 -- One third of Robert's lap time
  let rachel_in_window_start := 20 -- Rachel in the window from 20 to 100s
  let rachel_in_window_end := 100
  let robert_in_window_start := 0 -- Robert in the window from 0 to 66.66s
  let robert_in_window_end := 66.66
  let overlap_start := max rachel_in_window_start robert_in_window_start -- The start of overlap
  let overlap_end := min rachel_in_window_end robert_in_window_end -- The end of overlap
  let overlap_duration := overlap_end - overlap_start -- Duration of the overlap
  let probability := overlap_duration / picture_duration -- Probability of both in the picture
  probability = 46.66 / 60 := sorry

end probability_rachel_robert_in_picture_l1079_107901


namespace reciprocal_of_neg_2023_l1079_107976

theorem reciprocal_of_neg_2023 : 1 / (-2023) = -1 / 2023 :=
by sorry

end reciprocal_of_neg_2023_l1079_107976


namespace shekar_average_marks_l1079_107912

-- Define the scores for each subject
def mathematics := 76
def science := 65
def social_studies := 82
def english := 67
def biology := 55
def computer_science := 89
def history := 74
def geography := 63
def physics := 78
def chemistry := 71

-- Define the total number of subjects
def number_of_subjects := 10

-- State the theorem to prove the average marks
theorem shekar_average_marks :
  (mathematics + science + social_studies + english + biology +
   computer_science + history + geography + physics + chemistry) 
   / number_of_subjects = 72 := 
by
  -- Proof is omitted
  sorry

end shekar_average_marks_l1079_107912


namespace michael_remaining_money_l1079_107991

variables (m b n : ℝ) (h1 : (1 : ℝ) / 3 * m = 1 / 2 * n * b) (h2 : 5 = m / 15)

theorem michael_remaining_money : m - (2 / 3 * m + m / 15) = 4 / 15 * m :=
by
  have hb1 : 2 / 3 * m = (2 * m) / 3 := by ring
  have hb2 : m / 15 = (1 * m) / 15 := by ring
  rw [hb1, hb2]
  sorry

end michael_remaining_money_l1079_107991


namespace total_goals_in_five_matches_is_4_l1079_107984

theorem total_goals_in_five_matches_is_4
    (A : ℚ) -- defining the average number of goals before the fifth match as rational
    (h1 : A * 4 + 2 = (A + 0.3) * 5) : -- condition representing total goals equation
    4 = (4 * A + 2) := -- statement that the total number of goals in 5 matches is 4
by
  sorry

end total_goals_in_five_matches_is_4_l1079_107984


namespace triangle_BDC_is_isosceles_l1079_107980

-- Define the given conditions
variables {A B C D : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]
variables (AB AC BC AD DC : ℝ)
variables (a : ℝ)
variables (α : ℝ)

-- Given conditions
def is_isosceles_triangle (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] 
  (AB AC : ℝ) : Prop :=
AB = AC

def angle_BAC_120 (α : ℝ) : Prop :=
α = 120

def point_D_extension (AD AB : ℝ) : Prop :=
AD = 2 * AB

-- Let triangle ABC be isosceles with AB = AC and angle BAC = 120 degrees
axiom isosceles_triangle_ABC (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C]
  (AB AC : ℝ) : is_isosceles_triangle A B C AB AC

axiom angle_BAC (α : ℝ) : angle_BAC_120 α

axiom point_D (AD AB : ℝ) : point_D_extension AD AB

-- Prove that triangle BDC is isosceles
theorem triangle_BDC_is_isosceles 
  (A B C D : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]
  (AB AC BC AD DC : ℝ) 
  (α : ℝ) 
  (h1 : is_isosceles_triangle A B C AB AC)
  (h2 : angle_BAC_120 α)
  (h3 : point_D_extension AD AB) :
  BC = DC :=
sorry

end triangle_BDC_is_isosceles_l1079_107980


namespace ratio_second_shop_to_shirt_l1079_107977

-- Define the initial conditions in Lean
def initial_amount : ℕ := 55
def spent_on_shirt : ℕ := 7
def final_amount : ℕ := 27

-- Define the amount spent in the second shop calculation
def spent_in_second_shop (i_amt s_shirt f_amt : ℕ) : ℕ :=
  (i_amt - s_shirt) - f_amt

-- Define the ratio calculation
def ratio (a b : ℕ) : ℕ := a / b

-- Lean 4 statement proving the ratio of amounts
theorem ratio_second_shop_to_shirt : 
  ratio (spent_in_second_shop initial_amount spent_on_shirt final_amount) spent_on_shirt = 3 := 
by
  sorry

end ratio_second_shop_to_shirt_l1079_107977


namespace max_side_length_l1079_107934

theorem max_side_length (a b c : ℕ) (h1 : a ≥ b) (h2 : b ≥ c) (h3 : a + b + c = 24)
  (h4 : b + c > a) (h5 : a ≠ b) (h6 : b ≠ c) (h7 : a ≠ c) : a ≤ 11 :=
by
  sorry

end max_side_length_l1079_107934


namespace part_a_part_b_l1079_107900

section
-- Definitions based on the conditions
variable (n : ℕ)  -- Variable n representing the number of cities

-- Given a condition function T_n that returns an integer (number of ways to build roads)
def T_n (n : ℕ) : ℕ := sorry  -- Definition placeholder for T_n function

-- Part (a): For all odd n, T_n(n) is divisible by n
theorem part_a (hn : n % 2 = 1) : T_n n % n = 0 := sorry

-- Part (b): For all even n, T_n(n) is divisible by n / 2
theorem part_b (hn : n % 2 = 0) : T_n n % (n / 2) = 0 := sorry

end

end part_a_part_b_l1079_107900


namespace average_hamburgers_sold_per_day_l1079_107998

theorem average_hamburgers_sold_per_day 
  (total_hamburgers : ℕ) (days_in_week : ℕ)
  (h1 : total_hamburgers = 63) (h2 : days_in_week = 7) :
  total_hamburgers / days_in_week = 9 :=
by
  sorry

end average_hamburgers_sold_per_day_l1079_107998


namespace quadratic_points_relation_l1079_107946

theorem quadratic_points_relation
  (k y₁ y₂ y₃ : ℝ)
  (hA : y₁ = -((-1) - 1)^2 + k)
  (hB : y₂ = -(2 - 1)^2 + k)
  (hC : y₃ = -(4 - 1)^2 + k) : y₃ < y₁ ∧ y₁ < y₂ :=
by
  sorry

end quadratic_points_relation_l1079_107946


namespace correct_calculation_l1079_107908

theorem correct_calculation (x : ℕ) (h : 637 = x + 238) : x - 382 = 17 :=
by
  sorry

end correct_calculation_l1079_107908


namespace least_number_to_subtract_l1079_107990

theorem least_number_to_subtract (x : ℕ) :
  (2590 - x) % 9 = 6 ∧ 
  (2590 - x) % 11 = 6 ∧ 
  (2590 - x) % 13 = 6 ↔ 
  x = 16 := 
sorry

end least_number_to_subtract_l1079_107990


namespace problem_inequality_l1079_107995

theorem problem_inequality (a b c : ℝ) : a^2 + b^2 + c^2 + 4 ≥ ab + 3*b + 2*c := 
by 
  sorry

end problem_inequality_l1079_107995


namespace Billie_has_2_caps_l1079_107963

-- Conditions as definitions in Lean
def Sammy_caps : ℕ := 8
def Janine_caps : ℕ := Sammy_caps - 2
def Billie_caps : ℕ := Janine_caps / 3

-- Problem statement to prove
theorem Billie_has_2_caps : Billie_caps = 2 := by
  sorry

end Billie_has_2_caps_l1079_107963


namespace fraction_of_time_to_cover_distance_l1079_107993

-- Definitions for the given conditions
def distance : ℝ := 540
def initial_time : ℝ := 12
def new_speed : ℝ := 60

-- The statement we need to prove
theorem fraction_of_time_to_cover_distance :
  ∃ (x : ℝ), (x = 3 / 4) ∧ (distance / (initial_time * x) = new_speed) :=
by
  -- Proof steps would go here
  sorry

end fraction_of_time_to_cover_distance_l1079_107993


namespace vec_subtraction_l1079_107945

variables (a b : Prod ℝ ℝ)
def vec1 : Prod ℝ ℝ := (1, 2)
def vec2 : Prod ℝ ℝ := (3, 1)

theorem vec_subtraction : (2 * (vec1.fst, vec1.snd) - (vec2.fst, vec2.snd)) = (-1, 3) := by
  -- Proof here, skipped
  sorry

end vec_subtraction_l1079_107945


namespace gcd_lcm_of_300_105_l1079_107903

theorem gcd_lcm_of_300_105 :
  ∃ g l : ℕ, g = Int.gcd 300 105 ∧ l = Nat.lcm 300 105 ∧ g = 15 ∧ l = 2100 :=
by
  let g := Int.gcd 300 105
  let l := Nat.lcm 300 105
  have g_def : g = 15 := sorry
  have l_def : l = 2100 := sorry
  exact ⟨g, l, ⟨g_def, ⟨l_def, ⟨g_def, l_def⟩⟩⟩⟩

end gcd_lcm_of_300_105_l1079_107903


namespace avg_weight_section_B_l1079_107968

theorem avg_weight_section_B 
  (W_B : ℝ) 
  (num_students_A : ℕ := 36) 
  (avg_weight_A : ℝ := 30) 
  (num_students_B : ℕ := 24) 
  (total_students : ℕ := 60) 
  (avg_weight_class : ℝ := 30) 
  (h1 : num_students_A * avg_weight_A + num_students_B * W_B = total_students * avg_weight_class) :
  W_B = 30 :=
sorry

end avg_weight_section_B_l1079_107968


namespace game_completion_days_l1079_107975

theorem game_completion_days (initial_playtime hours_per_day : ℕ) (initial_days : ℕ) (completion_percentage : ℚ) (increased_playtime : ℕ) (remaining_days : ℕ) :
  initial_playtime = 4 →
  hours_per_day = 2 * 7 →
  completion_percentage = 0.4 →
  increased_playtime = 7 →
  ((initial_playtime * hours_per_day) / completion_percentage) - (initial_playtime * hours_per_day) = increased_playtime * remaining_days →
  remaining_days = 12 :=
by
  intros
  sorry

end game_completion_days_l1079_107975


namespace gretchen_work_hours_l1079_107921

noncomputable def walking_ratio (walking: ℤ) (sitting: ℤ) : Prop :=
  walking * 90 = sitting * 10

theorem gretchen_work_hours (walking_time: ℤ) (h: ℤ) (condition1: walking_ratio 40 (60 * h)) :
  h = 6 :=
by sorry

end gretchen_work_hours_l1079_107921


namespace lewis_weekly_earning_l1079_107916

def total_amount_earned : ℕ := 178
def number_of_weeks : ℕ := 89
def weekly_earning (total : ℕ) (weeks : ℕ) : ℕ := total / weeks

theorem lewis_weekly_earning : weekly_earning total_amount_earned number_of_weeks = 2 :=
by
  -- The proof will go here
  sorry

end lewis_weekly_earning_l1079_107916


namespace exist_integers_xy_divisible_by_p_l1079_107926

theorem exist_integers_xy_divisible_by_p (p : ℕ) [Fact (Nat.Prime p)] : ∃ x y : ℤ, (x^2 + y^2 + 2) % p = 0 := by
  sorry

end exist_integers_xy_divisible_by_p_l1079_107926


namespace hiker_speed_third_day_l1079_107943

-- Define the conditions
def first_day_distance : ℕ := 18
def first_day_speed : ℕ := 3
def second_day_distance : ℕ :=
  let first_day_hours := first_day_distance / first_day_speed
  let second_day_hours := first_day_hours - 1
  let second_day_speed := first_day_speed + 1
  second_day_hours * second_day_speed
def total_distance : ℕ := 53
def third_day_hours : ℕ := 3

-- Define the speed on the third day based on given conditions
def speed_on_third_day : ℕ :=
  let third_day_distance := total_distance - first_day_distance - second_day_distance
  third_day_distance / third_day_hours

-- The theorem we need to prove
theorem hiker_speed_third_day : speed_on_third_day = 5 := by
  sorry

end hiker_speed_third_day_l1079_107943


namespace combined_age_l1079_107928

-- Conditions as definitions
def AmyAge (j : ℕ) : ℕ :=
  j / 3

def ChrisAge (a : ℕ) : ℕ :=
  2 * a

-- Given condition
def JeremyAge : ℕ := 66

-- Question to prove
theorem combined_age : 
  let j := JeremyAge
  let a := AmyAge j
  let c := ChrisAge a
  a + j + c = 132 :=
by
  sorry

end combined_age_l1079_107928


namespace simplify_expression_l1079_107992

variable (b c d x y : ℝ)

theorem simplify_expression :
  (cx * (b^2 * x^3 + 3 * b^2 * y^3 + c^3 * y^3) + dy * (b^2 * x^3 + 3 * c^3 * x^3 + c^3 * y^3)) / (cx + dy) 
  = b^2 * x^3 + 3 * c^2 * xy^3 + c^3 * y^3 :=
by sorry

end simplify_expression_l1079_107992


namespace sum_first_2018_terms_of_given_sequence_l1079_107949

def arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ := a + n * d

def sum_of_first_n_terms (a d : ℤ) (n : ℕ) : ℤ :=
  n * (2 * a + (n - 1) * d) / 2

theorem sum_first_2018_terms_of_given_sequence :
  let a := 1
  let d := -1 / 2017
  S_2018 = 1009 :=
by
  sorry

end sum_first_2018_terms_of_given_sequence_l1079_107949


namespace initial_money_given_l1079_107966

def bracelet_cost : ℕ := 15
def necklace_cost : ℕ := 10
def mug_cost : ℕ := 20
def num_bracelets : ℕ := 3
def num_necklaces : ℕ := 2
def num_mugs : ℕ := 1
def change_received : ℕ := 15

theorem initial_money_given : num_bracelets * bracelet_cost + num_necklaces * necklace_cost + num_mugs * mug_cost + change_received = 100 := 
sorry

end initial_money_given_l1079_107966


namespace max_area_of_triangle_l1079_107936

theorem max_area_of_triangle (a c : ℝ)
    (h1 : a^2 + c^2 = 16 + a * c) : 
    ∃ s : ℝ, s = 4 * Real.sqrt 3 := by
  sorry

end max_area_of_triangle_l1079_107936


namespace cost_of_projector_and_whiteboard_l1079_107914

variable (x : ℝ)

def cost_of_projector : ℝ := x
def cost_of_whiteboard : ℝ := x + 4000
def total_cost_eq_44000 : Prop := 4 * (x + 4000) + 3 * x = 44000

theorem cost_of_projector_and_whiteboard 
  (h : total_cost_eq_44000 x) : 
  cost_of_projector x = 4000 ∧ cost_of_whiteboard x = 8000 :=
by
  sorry

end cost_of_projector_and_whiteboard_l1079_107914


namespace arithmetic_sequence_Sn_l1079_107918

noncomputable def S (n : ℕ) : ℕ := sorry -- S is the sequence function

theorem arithmetic_sequence_Sn {n : ℕ} (h1 : S n = 2) (h2 : S (3 * n) = 18) : S (4 * n) = 26 :=
  sorry

end arithmetic_sequence_Sn_l1079_107918


namespace units_digit_of_quotient_l1079_107948

theorem units_digit_of_quotient : 
  let n := 1993
  let term1 := 4 ^ n
  let term2 := 6 ^ n
  (term1 + term2) % 5 = 0 →
  let quotient := (term1 + term2) / 5
  (quotient % 10 = 0) := 
by 
  sorry

end units_digit_of_quotient_l1079_107948


namespace number_of_six_digit_numbers_formable_by_1_2_3_4_l1079_107941

theorem number_of_six_digit_numbers_formable_by_1_2_3_4
  (digits : Finset ℕ := {1, 2, 3, 4})
  (pairs_count : ℕ := 2)
  (non_adjacent_pair : ℕ := 1)
  (adjacent_pair : ℕ := 1)
  (six_digit_numbers : ℕ := 432) :
  ∃ (n : ℕ), n = 432 :=
by
  -- Proof will go here
  sorry

end number_of_six_digit_numbers_formable_by_1_2_3_4_l1079_107941


namespace students_present_l1079_107933

theorem students_present (total_students : ℕ) (absent_percent : ℝ) (total_absent : ℝ) (total_present : ℝ) :
  total_students = 50 → absent_percent = 0.12 → total_absent = total_students * absent_percent →
  total_present = total_students - total_absent →
  total_present = 44 :=
by
  intros _ _ _ _; sorry

end students_present_l1079_107933


namespace sum_and_product_of_three_numbers_l1079_107954

variables (a b c : ℝ)

-- Conditions
axiom h1 : a + b = 35
axiom h2 : b + c = 47
axiom h3 : c + a = 52

-- Prove the sum and product
theorem sum_and_product_of_three_numbers : a + b + c = 67 ∧ a * b * c = 9600 :=
by {
  sorry
}

end sum_and_product_of_three_numbers_l1079_107954


namespace negation_of_forall_exp_positive_l1079_107947

theorem negation_of_forall_exp_positive :
  ¬ (∀ x : ℝ, Real.exp x > 0) ↔ ∃ x : ℝ, Real.exp x ≤ 0 :=
by {
  sorry
}

end negation_of_forall_exp_positive_l1079_107947


namespace min_students_in_group_l1079_107907

theorem min_students_in_group 
  (g1 g2 : ℕ) 
  (n1 n2 e1 e2 f1 f2 : ℕ)
  (H_equal_groups : g1 = g2)
  (H_both_languages_g1 : n1 = 5)
  (H_both_languages_g2 : n2 = 5)
  (H_french_students : f1 * 3 = f2)
  (H_english_students : e1 = 4 * e2)
  (H_total_g1 : g1 = f1 + e1 - n1)
  (H_total_g2 : g2 = f2 + e2 - n2) 
: g1 = 28 :=
sorry

end min_students_in_group_l1079_107907


namespace remainder_of_expression_l1079_107940

theorem remainder_of_expression (n : ℤ) (h : n % 60 = 1) : (n^2 + 2 * n + 3) % 60 = 6 := 
by
  sorry

end remainder_of_expression_l1079_107940


namespace path_traveled_is_correct_l1079_107953

-- Define the original triangle and the circle.
def side_a : ℝ := 8
def side_b : ℝ := 10
def side_c : ℝ := 12.5
def radius : ℝ := 1.5

-- Define the condition that the circle is rolling inside the triangle.
def new_side (original_side : ℝ) (r : ℝ) : ℝ := original_side - 2 * r

-- Calculate the new sides of the smaller triangle path.
def new_side_a := new_side side_a radius
def new_side_b := new_side side_b radius
def new_side_c := new_side side_c radius

-- Calculate the perimeter of the path traced by the circle's center.
def path_perimeter := new_side_a + new_side_b + new_side_c

-- Prove that this perimeter equals 21.5 units under given conditions.
theorem path_traveled_is_correct : path_perimeter = 21.5 := by
  simp [new_side, new_side_a, new_side_b, new_side_c, path_perimeter]
  sorry

end path_traveled_is_correct_l1079_107953


namespace Mitch_needs_to_keep_500_for_license_and_registration_l1079_107944

-- Define the constants and variables
def total_savings : ℕ := 20000
def cost_per_foot : ℕ := 1500
def longest_boat_length : ℕ := 12
def docking_fee_factor : ℕ := 3

-- Define the price of the longest boat
def cost_longest_boat : ℕ := longest_boat_length * cost_per_foot

-- Define the amount for license and registration
def license_and_registration (L : ℕ) : Prop :=
  total_savings - cost_longest_boat = L * (docking_fee_factor + 1)

-- The statement to be proved
theorem Mitch_needs_to_keep_500_for_license_and_registration :
  ∃ L : ℕ, license_and_registration L ∧ L = 500 :=
by
  -- Conditions and setup have already been defined, we now state the proof goal.
  sorry

end Mitch_needs_to_keep_500_for_license_and_registration_l1079_107944


namespace probability_black_or_white_l1079_107981

-- Defining the probabilities of drawing red and white balls
def prob_red : ℝ := 0.45
def prob_white : ℝ := 0.25

-- Defining the total probability
def total_prob : ℝ := 1.0

-- Define the probability of drawing a black or white ball
def prob_black_or_white : ℝ := total_prob - prob_red

-- The theorem stating the required proof
theorem probability_black_or_white : 
  prob_black_or_white = 0.55 := by
    sorry

end probability_black_or_white_l1079_107981


namespace area_ratio_of_squares_l1079_107938

open Real

theorem area_ratio_of_squares (a b : ℝ) (h : 4 * a = 4 * 4 * b) : (a^2) / (b^2) = 16 := 
by
  sorry

end area_ratio_of_squares_l1079_107938


namespace alcohol_concentration_l1079_107997

theorem alcohol_concentration 
  (x : ℝ) -- concentration of alcohol in the first vessel (as a percentage)
  (h1 : 0 ≤ x ∧ x ≤ 100) -- percentage is between 0 and 100
  (h2 : (x / 100) * 2 + (55 / 100) * 6 = (37 / 100) * 10) -- given condition for concentration balance
  : x = 20 :=
sorry

end alcohol_concentration_l1079_107997


namespace complement_intersection_l1079_107935

open Set

variable (U A B : Set ℕ)

theorem complement_intersection (U : Set ℕ) (A B : Set ℕ) (hU : U = {1, 2, 3, 4, 5, 6}) (hA : A = {1, 3, 6}) (hB : B = {1, 2}) :
  ((U \ A) ∩ B) = {2} :=
by
  rw [hU, hA, hB]
  sorry

end complement_intersection_l1079_107935


namespace min_value_of_reciprocals_l1079_107979

theorem min_value_of_reciprocals (m n : ℝ) (h1 : m + n = 2) (h2 : m * n > 0) : 
  (1 / m) + (1 / n) = 2 :=
by
  -- the proof needs to be completed here.
  sorry

end min_value_of_reciprocals_l1079_107979


namespace factorize_x2_minus_9_l1079_107957

theorem factorize_x2_minus_9 (x : ℝ) : x^2 - 9 = (x + 3) * (x - 3) := 
sorry

end factorize_x2_minus_9_l1079_107957


namespace first_player_winning_strategy_l1079_107970

def game_strategy (S : ℕ) : Prop :=
  ∃ k, (1 ≤ k ∧ k ≤ 5 ∧ (S - k) % 6 = 1)

theorem first_player_winning_strategy : game_strategy 100 :=
sorry

end first_player_winning_strategy_l1079_107970


namespace person_B_catches_up_after_meeting_point_on_return_l1079_107955
noncomputable def distance_A := 46
noncomputable def speed_A := 15
noncomputable def speed_B := 40
noncomputable def initial_gap_time := 1

-- Prove that Person B catches up to Person A after 3/5 hours.
theorem person_B_catches_up_after : 
  ∃ x : ℚ, 40 * x = 15 * (x + 1) ∧ x = 3 / 5 := 
by
  sorry

-- Prove that they meet 10 kilometers away from point B on the return journey.
theorem meeting_point_on_return : 
  ∃ y : ℚ, (46 - y) / 15 - (46 + y) / 40 = 1 ∧ y = 10 := 
by 
  sorry

end person_B_catches_up_after_meeting_point_on_return_l1079_107955


namespace x_plus_inv_x_eq_8_then_power_4_l1079_107920

theorem x_plus_inv_x_eq_8_then_power_4 (x : ℝ) (h : x + 1/x = 8) : x^4 + 1/x^4 = 3842 :=
sorry

end x_plus_inv_x_eq_8_then_power_4_l1079_107920


namespace value_y1_y2_l1079_107994

variable {x1 x2 y1 y2 : ℝ}

-- Points on the inverse proportion function
def on_graph (x y : ℝ) : Prop := y = -3 / x

-- Given conditions
theorem value_y1_y2 (hx1 : on_graph x1 y1) (hx2 : on_graph x2 y2) (hxy : x1 * x2 = 2) : y1 * y2 = 9 / 2 :=
by
  sorry

end value_y1_y2_l1079_107994


namespace factorize_2070_l1079_107932

-- Define the conditions
def is_two_digit (n : ℕ) : Prop := n ≥ 10 ∧ n < 100
def is_unique_factorization (n a b : ℕ) : Prop := a * b = n ∧ is_two_digit a ∧ is_two_digit b

-- The final theorem statement
theorem factorize_2070 : 
  (∃ a b : ℕ, is_unique_factorization 2070 a b) ∧ 
  ∀ a b : ℕ, is_unique_factorization 2070 a b → (a = 30 ∧ b = 69) ∨ (a = 69 ∧ b = 30) :=
by 
  sorry

end factorize_2070_l1079_107932


namespace reversed_digit_multiple_of_sum_l1079_107937

variable (u v k : ℕ)

theorem reversed_digit_multiple_of_sum (h1 : 10 * u + v = k * (u + v)) :
  10 * v + u = (11 - k) * (u + v) :=
sorry

end reversed_digit_multiple_of_sum_l1079_107937


namespace find_k_solution_l1079_107974

noncomputable def vec1 : ℝ × ℝ := (3, -4)
noncomputable def vec2 : ℝ × ℝ := (5, 8)
noncomputable def target_norm : ℝ := 3 * Real.sqrt 10

theorem find_k_solution : ∃ k : ℝ, 0 ≤ k ∧ ‖(k * vec1.1 - vec2.1, k * vec1.2 - vec2.2)‖ = target_norm ∧ k = 0.0288 :=
by
  sorry

end find_k_solution_l1079_107974


namespace solution_set_of_inequality_l1079_107972

theorem solution_set_of_inequality :
  {x : ℝ | (x - 1) * (2 - x) ≥ 0} = {x : ℝ | 1 ≤ x ∧ x ≤ 2} :=
sorry

end solution_set_of_inequality_l1079_107972


namespace find_constant_c_l1079_107913

theorem find_constant_c (c : ℝ) (h : (x + 7) ∣ (c*x^3 + 19*x^2 - 3*c*x + 35)) : c = 3 := by
  sorry

end find_constant_c_l1079_107913


namespace initial_bowls_eq_70_l1079_107919

def customers : ℕ := 20
def bowls_per_customer : ℕ := 20
def reward_ratio := 10
def reward_bowls := 2
def remaining_bowls : ℕ := 30

theorem initial_bowls_eq_70 :
  let rewards_per_customer := (bowls_per_customer / reward_ratio) * reward_bowls
  let total_rewards := (customers / 2) * rewards_per_customer
  (remaining_bowls + total_rewards) = 70 :=
by
  sorry

end initial_bowls_eq_70_l1079_107919


namespace simplify_expression_l1079_107922

theorem simplify_expression :
  ( ∀ (a b c : ℕ), c > 0 ∧ (∀ p : ℕ, Prime p → ¬ p^2 ∣ c) →
  (a - b * Real.sqrt c = (28 - 16 * Real.sqrt 3) * 2 ^ (-2 - Real.sqrt 5))) :=
sorry

end simplify_expression_l1079_107922


namespace mn_value_l1079_107909

theorem mn_value (m n : ℤ) (h1 : 2 * m = 6) (h2 : m - n = 2) : m * n = 3 := by
  sorry

end mn_value_l1079_107909


namespace compute_expression_l1079_107917

theorem compute_expression : 12 * (1 / 17) * 34 = 24 := 
by {
  sorry
}

end compute_expression_l1079_107917


namespace two_numbers_are_opposites_l1079_107965

theorem two_numbers_are_opposites (x y z : ℝ) (h : (1 / x) + (1 / y) + (1 / z) = 1 / (x + y + z)) :
  (x + y = 0) ∨ (x + z = 0) ∨ (y + z = 0) :=
by
  sorry

end two_numbers_are_opposites_l1079_107965


namespace darnell_texts_l1079_107962

theorem darnell_texts (T : ℕ) (unlimited_plan_cost alternative_text_cost alternative_call_cost : ℕ) 
    (call_minutes : ℕ) (cost_difference : ℕ) :
    unlimited_plan_cost = 12 →
    alternative_text_cost = 1 →
    alternative_call_cost = 3 →
    call_minutes = 60 →
    cost_difference = 1 →
    (alternative_text_cost * T / 30 + alternative_call_cost * call_minutes / 20) = 
      unlimited_plan_cost - cost_difference →
    T = 60 := 
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end darnell_texts_l1079_107962


namespace symmetric_shading_additional_squares_l1079_107929

theorem symmetric_shading_additional_squares :
  let initial_shaded : List (ℕ × ℕ) := [(1, 1), (2, 4), (4, 3)]
  let required_horizontal_symmetry := [(4, 1), (1, 6), (4, 6)]
  let required_vertical_symmetry := [(2, 3), (1, 3)]
  let total_additional_squares := required_horizontal_symmetry ++ required_vertical_symmetry
  let final_shaded := initial_shaded ++ total_additional_squares
  ∀ s ∈ total_additional_squares, s ∉ initial_shaded →
    final_shaded.length - initial_shaded.length = 5 :=
by
  sorry

end symmetric_shading_additional_squares_l1079_107929


namespace find_x_l1079_107904

theorem find_x (x : ℕ) : (4 + x) / (7 + x) = 3 / 4 → x = 5 :=
by
  sorry

end find_x_l1079_107904


namespace expression_evaluation_l1079_107960

theorem expression_evaluation : 
  76 + (144 / 12) + (15 * 19)^2 - 350 - (270 / 6) = 80918 :=
by
  sorry

end expression_evaluation_l1079_107960


namespace solve_eq_l1079_107910

theorem solve_eq : ∀ x : ℂ, (x^3 + 3*x^2 + 4*x + 6) / (x + 5) = x^2 + 10 ↔
  x = (-3 + Complex.I * Real.sqrt 79) / 2 ∨ x = (-3 - Complex.I * Real.sqrt 79) / 2 :=
by 
  intro x
  sorry

end solve_eq_l1079_107910


namespace fourth_root_equiv_l1079_107956

theorem fourth_root_equiv (x : ℝ) (hx : 0 < x) : (x * (x ^ (3 / 4))) ^ (1 / 4) = x ^ (7 / 16) :=
sorry

end fourth_root_equiv_l1079_107956
