import Mathlib

namespace inequality_solutions_l580_58012

theorem inequality_solutions (a : ℝ) (h_pos : 0 < a) 
  (h_ineq_1 : ∃! x : ℕ, 10 < a ^ x ∧ a ^ x < 100) : ∃! x : ℕ, 100 < a ^ x ∧ a ^ x < 1000 :=
by
  sorry

end inequality_solutions_l580_58012


namespace min_expression_n_12_l580_58079

theorem min_expression_n_12 : ∃ n : ℕ, (n > 0) ∧ (∀ m : ℕ, m > 0 → (n = 12 → (n / 3 + 50 / n ≤ 
                        m / 3 + 50 / m))) :=
by
  sorry

end min_expression_n_12_l580_58079


namespace smallest_number_l580_58033

theorem smallest_number (a b c d : ℤ) (h_a : a = 0) (h_b : b = -1) (h_c : c = -4) (h_d : d = 5) : 
  c < b ∧ c < a ∧ c < d :=
by {
  sorry
}

end smallest_number_l580_58033


namespace find_number_l580_58091

theorem find_number (x : ℝ) (h : (((x + 45) / 2) / 2) + 45 = 85) : x = 115 :=
by
  sorry

end find_number_l580_58091


namespace dynaco_shares_sold_l580_58078

-- Define the conditions
def MicrotronPrice : ℝ := 36
def DynacoPrice : ℝ := 44
def TotalShares : ℕ := 300
def AvgPrice : ℝ := 40
def TotalValue : ℝ := TotalShares * AvgPrice

-- Define unknown variables
variables (M D : ℕ)

-- Express conditions in Lean
def total_shares_eq : Prop := M + D = TotalShares
def total_value_eq : Prop := MicrotronPrice * M + DynacoPrice * D = TotalValue

-- Define the problem statement
theorem dynaco_shares_sold : ∃ D : ℕ, 
  (∃ M : ℕ, total_shares_eq M D ∧ total_value_eq M D) ∧ D = 150 :=
by
  sorry

end dynaco_shares_sold_l580_58078


namespace speed_ratio_A_to_B_l580_58019

variables {u v : ℝ}

axiom perp_lines_intersect_at_o : true
axiom points_move_along_lines_at_constant_speed : true
axiom point_A_at_O_B_500_yards_away_at_t_0 : true
axiom after_2_minutes_A_and_B_equidistant : 2 * u = 500 - 2 * v
axiom after_10_minutes_A_and_B_equidistant : 10 * u = 10 * v - 500

theorem speed_ratio_A_to_B : u / v = 2 / 3 :=
by 
  sorry

end speed_ratio_A_to_B_l580_58019


namespace average_speed_interval_l580_58035

theorem average_speed_interval {s t : ℝ → ℝ} (h_eq : ∀ t, s t = t^2 + 1) : 
  (s 2 - s 1) / (2 - 1) = 3 :=
by
  sorry

end average_speed_interval_l580_58035


namespace foci_ellipsoid_hyperboloid_l580_58050

theorem foci_ellipsoid_hyperboloid (a b : ℝ) 
(h1 : ∀ x y, x^2 / a^2 + y^2 / b^2 = 1 → dist (0,y) (0, 5) = 5)
(h2 : ∀ x y, x^2 / a^2 - y^2 / b^2 = 1 → dist (x,0) (7, 0) = 7) :
  |a * b| = Real.sqrt 444 := sorry

end foci_ellipsoid_hyperboloid_l580_58050


namespace shirts_needed_for_vacation_l580_58066

def vacation_days := 7
def same_shirt_days := 2
def different_shirts_per_day := 2
def different_shirt_days := vacation_days - same_shirt_days

theorem shirts_needed_for_vacation : different_shirt_days * different_shirts_per_day + same_shirt_days = 11 := by
  sorry

end shirts_needed_for_vacation_l580_58066


namespace tan_45_eq_one_l580_58055

theorem tan_45_eq_one 
  (θ : ℝ)
  (hθ : θ = 45)
  (sin_45_eq : Real.sin (θ * Real.pi / 180) = Real.sqrt 2 / 2)
  (cos_45_eq : Real.cos (θ * Real.pi /180) = Real.sqrt 2 / 2)
  (tan_def : ∀ x : ℝ, 0 < Real.cos x → Real.tan x = Real.sin x / Real.cos x) : 
  Real.tan (θ * Real.pi / 180) = 1 := sorry

end tan_45_eq_one_l580_58055


namespace triangle_inequality_satisfied_for_n_six_l580_58015

theorem triangle_inequality_satisfied_for_n_six :
  ∀ (a b c : ℝ), 0 < a → 0 < b → 0 < c → 6 * (a * b + b * c + c * a) > 5 * (a^2 + b^2 + c^2) → 
  (a + b > c ∧ a + c > b ∧ b + c > a) := sorry

end triangle_inequality_satisfied_for_n_six_l580_58015


namespace log_equation_solution_l580_58030

theorem log_equation_solution (x : ℝ) (hpos : x > 0) (hneq : x ≠ 1) : (Real.log 8 / Real.log x) * (2 * Real.log x / Real.log 2) = 6 * Real.log 2 :=
by
  sorry

end log_equation_solution_l580_58030


namespace rolls_combinations_l580_58040

theorem rolls_combinations {n k : ℕ} (h_n : n = 4) (h_k : k = 5) :
  (Nat.choose (n + k - 1) k) = 56 :=
by
  rw [h_n, h_k]
  norm_num
  sorry

end rolls_combinations_l580_58040


namespace circle_radius_l580_58044

theorem circle_radius (k r : ℝ) (h : k > 8) 
  (h1 : r = |k - 8|)
  (h2 : r = k / Real.sqrt 5) : 
  r = 8 * Real.sqrt 5 + 8 := 
sorry

end circle_radius_l580_58044


namespace root_not_less_than_a_l580_58060

noncomputable def f (x : ℝ) : ℝ := (1/2)^x - x^3

theorem root_not_less_than_a (a b c x0 : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : b < c)
  (h4 : f a * f b * f c < 0) (hx : f x0 = 0) : ¬ (x0 < a) :=
sorry

end root_not_less_than_a_l580_58060


namespace select_p_elements_with_integer_mean_l580_58069

theorem select_p_elements_with_integer_mean {p : ℕ} (hp : Nat.Prime p) (p_odd : p % 2 = 1) :
  ∃ (M : Finset ℕ), (M.card = (p^2 + 1) / 2) ∧ ∃ (S : Finset ℕ), (S.card = p) ∧ ((S.sum id) % p = 0) :=
by
  -- sorry to skip the proof
  sorry

end select_p_elements_with_integer_mean_l580_58069


namespace smallest_total_hot_dogs_l580_58018

def packs_hot_dogs := 12
def packs_buns := 9
def packs_mustard := 18
def packs_ketchup := 24

theorem smallest_total_hot_dogs : Nat.lcm (Nat.lcm (Nat.lcm packs_hot_dogs packs_buns) packs_mustard) packs_ketchup = 72 := by
  sorry

end smallest_total_hot_dogs_l580_58018


namespace trapezoid_base_lengths_l580_58038

noncomputable def trapezoid_bases (d h : Real) : Real × Real :=
  let b := h - 2 * d
  let B := h + 2 * d
  (b, B)

theorem trapezoid_base_lengths :
  ∀ (d : Real), d = Real.sqrt 3 →
  ∀ (h : Real), h = Real.sqrt 48 →
  ∃ (b B : Real), trapezoid_bases d h = (b, B) ∧ b = Real.sqrt 48 - 2 * Real.sqrt 3 ∧ B = Real.sqrt 48 + 2 * Real.sqrt 3 := by 
  sorry

end trapezoid_base_lengths_l580_58038


namespace perimeter_ABCDEFG_l580_58098

variables {Point : Type}
variables {dist : Point → Point → ℝ}  -- Distance function

-- Definitions for midpoint and equilateral triangles
def is_midpoint (M A B : Point) : Prop := dist A M = dist M B ∧ dist A B = 2 * dist A M
def is_equilateral (A B C : Point) : Prop := dist A B = dist B C ∧ dist B C = dist C A

variables {A B C D E F G : Point}  -- Points in the plane
variables (h_eq_triangle_ABC : is_equilateral A B C)
variables (h_eq_triangle_ADE : is_equilateral A D E)
variables (h_eq_triangle_EFG : is_equilateral E F G)
variables (h_midpoint_D : is_midpoint D A C)
variables (h_midpoint_G : is_midpoint G A E)
variables (h_midpoint_F : is_midpoint F D E)
variables (h_AB_length : dist A B = 6)

theorem perimeter_ABCDEFG : 
  dist A B + dist B C + dist C D + dist D E + dist E F + dist F G + dist G A = 24 :=
sorry

end perimeter_ABCDEFG_l580_58098


namespace original_price_dish_l580_58032

-- Conditions
variables (P : ℝ) -- Original price of the dish
-- Discount and tips
def john_discounted_and_tip := 0.9 * P + 0.15 * P
def jane_discounted_and_tip := 0.9 * P + 0.135 * P

-- Condition of payment difference
def payment_difference := john_discounted_and_tip P = jane_discounted_and_tip P + 0.36

-- The theorem to prove
theorem original_price_dish : payment_difference P → P = 24 :=
by
  intro h
  sorry

end original_price_dish_l580_58032


namespace television_final_price_l580_58097

theorem television_final_price :
  let original_price := 1200
  let discount_percent := 0.30
  let tax_percent := 0.08
  let rebate := 50
  let discount := discount_percent * original_price
  let sale_price := original_price - discount
  let tax := tax_percent * sale_price
  let price_including_tax := sale_price + tax
  let final_amount := price_including_tax - rebate
  final_amount = 857.2 :=
by
{
  -- The proof would go here, but it's omitted as per instructions.
  sorry
}

end television_final_price_l580_58097


namespace g_minus_6_eq_neg_20_l580_58071

noncomputable def g : ℤ → ℤ := sorry

axiom condition1 : g 1 - 1 > 0
axiom condition2 : ∀ x y : ℤ, g x * g y + x + y + x * y = g (x + y) + x * g y + y * g x
axiom condition3 : ∀ x : ℤ, 3 * g (x + 1) = g x + 2 * x + 3

theorem g_minus_6_eq_neg_20 : g (-6) = -20 := sorry

end g_minus_6_eq_neg_20_l580_58071


namespace distance_between_stations_is_correct_l580_58085

noncomputable def distance_between_stations : ℕ := 200

theorem distance_between_stations_is_correct 
  (start_hour_p : ℕ := 7) 
  (speed_p : ℕ := 20) 
  (start_hour_q : ℕ := 8) 
  (speed_q : ℕ := 25) 
  (meeting_hour : ℕ := 12)
  (time_travel_p := meeting_hour - start_hour_p) -- Time traveled by train from P
  (time_travel_q := meeting_hour - start_hour_q) -- Time traveled by train from Q 
  (distance_travel_p := speed_p * time_travel_p) 
  (distance_travel_q := speed_q * time_travel_q) : 
  distance_travel_p + distance_travel_q = distance_between_stations :=
by 
  sorry

end distance_between_stations_is_correct_l580_58085


namespace point_on_y_axis_coordinates_l580_58054

theorem point_on_y_axis_coordinates (m : ℤ) (P : ℤ × ℤ) (hP : P = (m - 1, m + 3)) (hY : P.1 = 0) : P = (0, 4) :=
sorry

end point_on_y_axis_coordinates_l580_58054


namespace time_to_fill_pool_l580_58059

theorem time_to_fill_pool :
  ∀ (total_volume : ℝ) (filling_rate : ℝ) (leaking_rate : ℝ),
  total_volume = 60 →
  filling_rate = 1.6 →
  leaking_rate = 0.1 →
  (total_volume / (filling_rate - leaking_rate)) = 40 :=
by
  intros total_volume filling_rate leaking_rate hv hf hl
  rw [hv, hf, hl]
  sorry

end time_to_fill_pool_l580_58059


namespace initial_bottles_of_water_l580_58021

theorem initial_bottles_of_water {B : ℕ} (h1 : 100 - (6 * B + 5) = 71) : B = 4 :=
by
  sorry

end initial_bottles_of_water_l580_58021


namespace find_x_l580_58042

noncomputable def angle_sum_triangle (A B C: ℝ) : Prop :=
  A + B + C = 180

noncomputable def vertical_angles_equal (A B: ℝ) : Prop :=
  A = B

noncomputable def right_angle_sum (D E: ℝ) : Prop :=
  D + E = 90

theorem find_x 
  (angle_ABC angle_BAC angle_DCE : ℝ) 
  (h1 : angle_ABC = 70)
  (h2 : angle_BAC = 50)
  (h3 : angle_sum_triangle angle_ABC angle_BAC angle_DCE)
  (h4 : vertical_angles_equal angle_DCE angle_DCE)
  (h5 : right_angle_sum angle_DCE 30) :
  angle_DCE = 60 :=
by
  sorry

end find_x_l580_58042


namespace find_original_number_l580_58045

def is_valid_digit (d : ℕ) : Prop := d < 10

def original_number (a b c : ℕ) : Prop :=
  is_valid_digit a ∧ is_valid_digit b ∧ is_valid_digit c ∧
  222 * (a + b + c) - 5 * (100 * a + 10 * b + c) = 3194

theorem find_original_number (a b c : ℕ) (h_valid: is_valid_digit a ∧ is_valid_digit b ∧ is_valid_digit c)
  (h_sum : 222 * (a + b + c) - 5 * (100 * a + 10 * b + c) = 3194) : 
  100 * a + 10 * b + c = 358 := 
sorry

end find_original_number_l580_58045


namespace fruit_basket_cost_is_28_l580_58093

def basket_total_cost : ℕ := 4 * 1 + 3 * 2 + (24 / 12) * 4 + 2 * 3 + 2 * 2

theorem fruit_basket_cost_is_28 : basket_total_cost = 28 := by
  sorry

end fruit_basket_cost_is_28_l580_58093


namespace carnival_rent_l580_58025

-- Define the daily popcorn earnings
def daily_popcorn : ℝ := 50
-- Define the multiplier for cotton candy earnings
def multiplier : ℝ := 3
-- Define the number of operational days
def days : ℕ := 5
-- Define the cost of ingredients
def ingredients_cost : ℝ := 75
-- Define the net earnings after expenses
def net_earnings : ℝ := 895
-- Define the total earnings from selling popcorn for all days
def total_popcorn_earnings : ℝ := daily_popcorn * days
-- Define the total earnings from selling cotton candy for all days
def total_cottoncandy_earnings : ℝ := (daily_popcorn * multiplier) * days
-- Define the total earnings before expenses
def total_earnings : ℝ := total_popcorn_earnings + total_cottoncandy_earnings
-- Define the amount remaining after paying the rent (which includes net earnings and ingredient cost)
def remaining_after_rent : ℝ := net_earnings + ingredients_cost
-- Define the rent
def rent : ℝ := total_earnings - remaining_after_rent

theorem carnival_rent : rent = 30 := by
  sorry

end carnival_rent_l580_58025


namespace meaningful_fraction_l580_58095

theorem meaningful_fraction (x : ℝ) : (x + 1 ≠ 0) ↔ (x ≠ -1) :=
by
  sorry

end meaningful_fraction_l580_58095


namespace find_a_from_derivative_l580_58026

-- Define the function f(x) = ax^3 + 3x^2 - 6
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + 3 * x^2 - 6

-- Define the derivative of f
def f' (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 6 * x

-- State the theorem to prove that a = 10/3 given f'(-1) = 4
theorem find_a_from_derivative (a : ℝ) (h : f' a (-1) = 4) : a = 10 / 3 := 
  sorry

end find_a_from_derivative_l580_58026


namespace angle_same_terminal_side_l580_58070

theorem angle_same_terminal_side (α θ : ℝ) (hα : α = 1690) (hθ : 0 < θ) (hθ2 : θ < 360) (h_terminal_side : ∃ k : ℤ, α = k * 360 + θ) : θ = 250 :=
by
  sorry

end angle_same_terminal_side_l580_58070


namespace pet_store_has_70_birds_l580_58023

-- Define the given conditions
def num_cages : ℕ := 7
def parrots_per_cage : ℕ := 4
def parakeets_per_cage : ℕ := 3
def cockatiels_per_cage : ℕ := 2
def canaries_per_cage : ℕ := 1

-- Total number of birds in one cage
def birds_per_cage : ℕ := parrots_per_cage + parakeets_per_cage + cockatiels_per_cage + canaries_per_cage

-- Total number of birds in all cages
def total_birds := birds_per_cage * num_cages

-- Prove that the total number of birds is 70
theorem pet_store_has_70_birds : total_birds = 70 :=
sorry

end pet_store_has_70_birds_l580_58023


namespace anne_equals_bob_l580_58020

-- Define the conditions as constants and functions
def original_price : ℝ := 120.00
def tax_rate : ℝ := 0.06
def discount_rate : ℝ := 0.25

-- Calculation models for Anne and Bob
def anne_total (price : ℝ) (tax : ℝ) (discount : ℝ) : ℝ :=
  (price * (1 + tax)) * (1 - discount)

def bob_total (price : ℝ) (tax : ℝ) (discount : ℝ) : ℝ :=
  (price * (1 - discount)) * (1 + tax)

-- The theorem that states what we need to prove
theorem anne_equals_bob : anne_total original_price tax_rate discount_rate = bob_total original_price tax_rate discount_rate :=
by
  sorry

end anne_equals_bob_l580_58020


namespace sum_of_fractions_l580_58096

theorem sum_of_fractions (p q r : ℝ) 
  (h1 : p + q + r = 5) 
  (h2 : 1 / (p + q) + 1 / (q + r) + 1 / (p + r) = 9) : 
  r / (p + q) + p / (q + r) + q / (p + r) = 42 := 
by 
  sorry

end sum_of_fractions_l580_58096


namespace tap_b_fill_time_l580_58029

theorem tap_b_fill_time (t : ℝ) (h1 : t > 0) : 
  (∀ (A_fill B_fill together_fill : ℝ), 
    A_fill = 1/45 ∧ 
    B_fill = 1/t ∧ 
    together_fill = A_fill + B_fill ∧ 
    (9 * A_fill) + (23 * B_fill) = 1) → 
    t = 115 / 4 :=
by
  sorry

end tap_b_fill_time_l580_58029


namespace megan_pages_left_l580_58092

theorem megan_pages_left (total_problems completed_problems problems_per_page : ℕ)
    (h_total : total_problems = 40)
    (h_completed : completed_problems = 26)
    (h_problems_per_page : problems_per_page = 7) :
    (total_problems - completed_problems) / problems_per_page = 2 :=
by
  sorry

end megan_pages_left_l580_58092


namespace sum_of_seven_consecutive_integers_l580_58039

theorem sum_of_seven_consecutive_integers (n : ℤ) :
  n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) + (n + 6) = 7 * n + 21 :=
by
  sorry

end sum_of_seven_consecutive_integers_l580_58039


namespace necessary_condition_for_x_gt_5_l580_58009

theorem necessary_condition_for_x_gt_5 (x : ℝ) : x > 5 → x > 3 :=
by
  intros h
  exact lt_trans (show 3 < 5 from by linarith) h

end necessary_condition_for_x_gt_5_l580_58009


namespace solve_for_y_l580_58053

theorem solve_for_y (y : ℕ) : 9^y = 3^12 → y = 6 :=
by
  sorry

end solve_for_y_l580_58053


namespace angle_in_triangle_PQR_l580_58007

theorem angle_in_triangle_PQR
  (Q P R : ℝ)
  (h1 : P = 2 * Q)
  (h2 : R = 5 * Q)
  (h3 : Q + P + R = 180) : 
  P = 45 := 
by sorry

end angle_in_triangle_PQR_l580_58007


namespace function_identity_l580_58074

variable (f : ℕ+ → ℕ+)

theorem function_identity (h : ∀ n : ℕ+, f (n + 1) > f (f n)) : ∀ n : ℕ+, f n = n := sorry

end function_identity_l580_58074


namespace find_principal_l580_58086

theorem find_principal 
  (SI : ℝ) 
  (R : ℝ) 
  (T : ℝ) 
  (h_SI : SI = 4052.25) 
  (h_R : R = 9) 
  (h_T : T = 5) : 
  (SI * 100) / (R * T) = 9005 := 
by 
  rw [h_SI, h_R, h_T]
  sorry

end find_principal_l580_58086


namespace product_fraction_simplification_l580_58002

theorem product_fraction_simplification :
  (1 - (1 / 3)) * (1 - (1 / 4)) * (1 - (1 / 5)) = 2 / 5 :=
by
  sorry

end product_fraction_simplification_l580_58002


namespace relationship_between_y1_y2_l580_58081

theorem relationship_between_y1_y2 (b y1 y2 : ℝ) 
  (h1 : y1 = -(-2) + b) 
  (h2 : y2 = -(3) + b) : 
  y1 > y2 := 
by {
  sorry
}

end relationship_between_y1_y2_l580_58081


namespace unique_positive_integer_triples_l580_58051

theorem unique_positive_integer_triples (a b c : ℕ) (h1 : ab + 3 * b * c = 63) (h2 : ac + 3 * b * c = 39) : 
∃! (a b c : ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧ ab + 3 * b * c = 63 ∧ ac + 3 * b * c = 39 :=
by sorry

end unique_positive_integer_triples_l580_58051


namespace catherine_pencils_per_friend_l580_58065

theorem catherine_pencils_per_friend :
  ∀ (pencils pens given_pens : ℕ), 
  pencils = pens ∧ pens = 60 ∧ given_pens = 8 ∧ 
  (∃ remaining_items : ℕ, remaining_items = 22 ∧ 
    ∀ friends : ℕ, friends = 7 → 
    remaining_items = (pens - (given_pens * friends)) + (pencils - (given_pens * friends * (pencils / pens)))) →
  ((pencils - (given_pens * friends * (pencils / pens))) / friends) = 6 :=
by 
  sorry

end catherine_pencils_per_friend_l580_58065


namespace other_root_l580_58047

theorem other_root (m n : ℝ) (h : (3 : ℂ) + (1 : ℂ) * Complex.I ∈ {x : ℂ | x^2 + ↑m * x + ↑n = 0}) : 
    (3 : ℂ) - (1 : ℂ) * Complex.I ∈ {x : ℂ | x^2 + ↑m * x + ↑n = 0} :=
sorry

end other_root_l580_58047


namespace isosceles_triangle_base_vertex_trajectory_l580_58073

theorem isosceles_triangle_base_vertex_trajectory :
  ∀ (x y : ℝ), 
  (∀ (A : ℝ × ℝ) (B : ℝ × ℝ), 
    A = (2, 4) ∧ B = (2, 8) ∧ 
    ((x-2)^2 + (y-4)^2 = 16)) → 
  ((x ≠ 2) ∧ (y ≠ 8) → (x-2)^2 + (y-4)^2 = 16) :=
sorry

end isosceles_triangle_base_vertex_trajectory_l580_58073


namespace three_pow_12_mul_three_pow_8_equals_243_pow_4_l580_58049

theorem three_pow_12_mul_three_pow_8_equals_243_pow_4 : 3^12 * 3^8 = 243^4 := 
by sorry

end three_pow_12_mul_three_pow_8_equals_243_pow_4_l580_58049


namespace right_triangle_sides_l580_58013

theorem right_triangle_sides (a d : ℝ) (k : ℕ) (h_pos_a : 0 < a) (h_pos_d : 0 < d) (h_pos_k : 0 < k) :
  (a = 3) ∧ (d = 1) ∧ (k = 2) ↔ (a^2 + (a + d)^2 = (a + k * d)^2) :=
by 
  sorry

end right_triangle_sides_l580_58013


namespace sequence_x_sequence_y_sequence_z_sequence_t_l580_58090

theorem sequence_x (n : ℕ) : 
  n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4 → 
  (if n = 1 then (n^2 + n = 2) else 
   if n = 2 then (n^2 + n = 6) else 
   if n = 3 then (n^2 + n = 12) else 
   if n = 4 then (n^2 + n = 20) else true) := 
by sorry

theorem sequence_y (n : ℕ) : 
  n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4 → 
  (if n = 1 then (2 * n^2 = 2) else 
   if n = 2 then (2 * n^2 = 8) else 
   if n = 3 then (2 * n^2 = 18) else 
   if n = 4 then (2 * n^2 = 32) else true) := 
by sorry

theorem sequence_z (n : ℕ) : 
  n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4 → 
  (if n = 1 then (n^3 = 1) else 
   if n = 2 then (n^3 = 8) else 
   if n = 3 then (n^3 = 27) else 
   if n = 4 then (n^3 = 64) else true) := 
by sorry

theorem sequence_t (n : ℕ) : 
  n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4 → 
  (if n = 1 then (2^n = 2) else 
   if n = 2 then (2^n = 4) else 
   if n = 3 then (2^n = 8) else 
   if n = 4 then (2^n = 16) else true) := 
by sorry

end sequence_x_sequence_y_sequence_z_sequence_t_l580_58090


namespace area_of_right_triangle_l580_58000

-- Define a structure for the triangle with the given conditions
structure Triangle :=
(A B C : ℝ × ℝ)
(right_angle_at_C : (C.1 = 0 ∧ C.2 = 0))
(hypotenuse_length : (A.1 - B.1) ^ 2 + (A.2 - B.2) ^ 2 = 50 ^ 2)
(median_A : ∀ x: ℝ, A.2 = A.1 + 5)
(median_B : ∀ x: ℝ, B.2 = 2 * B.1 + 2)

-- Theorem statement
theorem area_of_right_triangle (t : Triangle) : 
  ∃ area : ℝ, area = 500 :=
sorry

end area_of_right_triangle_l580_58000


namespace license_plate_count_l580_58057

/-- Number of vowels available for the license plate -/
def num_vowels := 6

/-- Number of consonants available for the license plate -/
def num_consonants := 20

/-- Number of possible digits for the license plate -/
def num_digits := 10

/-- Number of special characters available for the license plate -/
def num_special_chars := 2

/-- Calculate the total number of possible license plates -/
def total_license_plates : Nat :=
  num_vowels * num_consonants * num_digits * num_consonants * num_special_chars

/- Prove that the total number of possible license plates is 48000 -/
theorem license_plate_count : total_license_plates = 48000 :=
  by
    unfold total_license_plates
    sorry

end license_plate_count_l580_58057


namespace function_domain_l580_58099

noncomputable def domain_function : Set ℝ :=
  {x : ℝ | x ≠ 8}

theorem function_domain :
  ∀ x, x ∈ domain_function ↔ x ∈ (Set.Iio 8 ∪ Set.Ioi 8) := by
  intro x
  sorry

end function_domain_l580_58099


namespace tan_alpha_sub_beta_l580_58068

theorem tan_alpha_sub_beta (α β : ℝ) (h₁ : Real.tan α = 9) (h₂ : Real.tan β = 6) : Real.tan (α - β) = 3 / 55 := 
sorry

end tan_alpha_sub_beta_l580_58068


namespace zero_of_f_l580_58075

noncomputable def f (x : ℝ) : ℝ := Real.logb 5 (x - 1)

theorem zero_of_f :
  ∃ x : ℝ, f x = 0 ∧ x = 2 :=
by
  use 2
  unfold f
  sorry -- Skip the proof steps, as instructed.

end zero_of_f_l580_58075


namespace pqr_problem_l580_58088

noncomputable def pqr_sums_to_44 (p q r : ℝ) : Prop :=
  (p < q) ∧ (∀ x, (x < -6 ∨ |x - 20| ≤ 2) ↔ ( (x - p) * (x - q) / (x - r) ≥ 0 ))

theorem pqr_problem (p q r : ℝ) (h : pqr_sums_to_44 p q r) : p + 2*q + 3*r = 44 :=
sorry

end pqr_problem_l580_58088


namespace tan_alpha_value_l580_58001

open Real

-- Define the angle alpha in the third quadrant
variable {α : ℝ}

-- Given conditions
def third_quadrant (α : ℝ) : Prop :=  π < α ∧ α < 3 * π / 2
def sin_alpha (α : ℝ) : Prop := sin α = -4 / 5

-- Statement to prove
theorem tan_alpha_value (h1 : third_quadrant α) (h2 : sin_alpha α) : tan α = 4 / 3 :=
sorry

end tan_alpha_value_l580_58001


namespace average_of_remaining_two_numbers_l580_58034

theorem average_of_remaining_two_numbers 
  (avg_6 : ℝ) (avg1_2 : ℝ) (avg2_2 : ℝ)
  (n1 n2 n3 : ℕ)
  (h_avg6 : n1 = 6 ∧ avg_6 = 4.60)
  (h_avg1_2 : n2 = 2 ∧ avg1_2 = 3.4)
  (h_avg2_2 : n3 = 2 ∧ avg2_2 = 3.8) :
  ∃ avg_rem2 : ℝ, avg_rem2 = 6.6 :=
by {
  sorry
}

end average_of_remaining_two_numbers_l580_58034


namespace part1_part2_l580_58017

def custom_op (a b : ℤ) : ℤ := a^2 - b + a * b

theorem part1  : custom_op (-3) (-2) = 17 := by
  sorry

theorem part2 : custom_op (-2) (custom_op (-3) (-2)) = -47 := by
  sorry

end part1_part2_l580_58017


namespace smallest_odd_number_with_five_different_prime_factors_l580_58014

noncomputable def smallest_odd_with_five_prime_factors : Nat :=
  3 * 5 * 7 * 11 * 13

theorem smallest_odd_number_with_five_different_prime_factors :
  smallest_odd_with_five_prime_factors = 15015 :=
by
  sorry -- Proof to be filled in later

end smallest_odd_number_with_five_different_prime_factors_l580_58014


namespace roots_of_equation_l580_58016

theorem roots_of_equation (x : ℝ) : x * (x - 1) = 0 ↔ x = 0 ∨ x = 1 := by
  sorry

end roots_of_equation_l580_58016


namespace max_value_l580_58094

theorem max_value (a b c : ℕ) (h1 : a = 2^35) (h2 : b = 26) (h3 : c = 1) : max a (max b c) = 2^35 :=
by
  -- This is where the proof would go
  sorry

end max_value_l580_58094


namespace monotonic_increasing_f_C_l580_58061

noncomputable def f_A (x : ℝ) : ℝ := -Real.log x
noncomputable def f_B (x : ℝ) : ℝ := 1 / (2^x)
noncomputable def f_C (x : ℝ) : ℝ := -(1 / x)
noncomputable def f_D (x : ℝ) : ℝ := 3^(abs (x - 1))

theorem monotonic_increasing_f_C : ∀ x y : ℝ, 0 < x → 0 < y → x < y → f_C x < f_C y :=
sorry

end monotonic_increasing_f_C_l580_58061


namespace line_equation_isosceles_triangle_l580_58024

theorem line_equation_isosceles_triangle 
  (x y : ℝ)
  (l : ℝ → ℝ → Prop)
  (h1 : l 3 2)
  (h2 : ∀ x y, l x y → (x = y ∨ x + y = 2 * intercept))
  (intercept : ℝ) :
  l x y ↔ (x - y = 1 ∨ x + y = 5) :=
by
  sorry

end line_equation_isosceles_triangle_l580_58024


namespace benzoic_acid_molecular_weight_l580_58004

-- Definitions for atomic weights
def atomic_weight_C : ℝ := 12.01
def atomic_weight_H : ℝ := 1.008
def atomic_weight_O : ℝ := 16.00

-- Molecular formula for Benzoic acid: C7H6O2
def benzoic_acid_formula : ℕ × ℕ × ℕ := (7, 6, 2)

-- Definition for the molecular weight calculation
def molecular_weight := λ (c h o : ℝ) (nC nH nO : ℕ) => 
  (nC * c) + (nH * h) + (nO * o)

-- Proof statement
theorem benzoic_acid_molecular_weight :
  molecular_weight atomic_weight_C atomic_weight_H atomic_weight_O 7 6 2 = 122.118 := by
  sorry

end benzoic_acid_molecular_weight_l580_58004


namespace jerry_books_vs_action_figures_l580_58082

-- Define the initial conditions as constants
def initial_books : ℕ := 7
def initial_action_figures : ℕ := 3
def added_action_figures : ℕ := 2

-- Define the total number of action figures after adding
def total_action_figures : ℕ := initial_action_figures + added_action_figures

-- The theorem we need to prove
theorem jerry_books_vs_action_figures : initial_books - total_action_figures = 2 :=
by
  -- Proof placeholder
  sorry

end jerry_books_vs_action_figures_l580_58082


namespace mark_cans_l580_58003

theorem mark_cans (R J M : ℕ) (h1 : J = 2 * R + 5) (h2 : M = 4 * J) (h3 : R + J + M = 135) : M = 100 :=
by
  sorry

end mark_cans_l580_58003


namespace m_minus_n_is_perfect_square_l580_58084

theorem m_minus_n_is_perfect_square (m n : ℕ) (h : 0 < m) (h1 : 0 < n) (h2 : 2001 * m^2 + m = 2002 * n^2 + n) : ∃ k : ℕ, m = n + k^2 :=
by
    sorry

end m_minus_n_is_perfect_square_l580_58084


namespace minimum_number_of_guests_l580_58046

def total_food : ℤ := 327
def max_food_per_guest : ℤ := 2

theorem minimum_number_of_guests :
  ∀ (n : ℤ), total_food ≤ n * max_food_per_guest → n = 164 :=
by
  sorry

end minimum_number_of_guests_l580_58046


namespace songs_in_first_two_albums_l580_58011

/-
Beyonce releases 5 different singles on iTunes.
She releases 2 albums that each has some songs.
She releases 1 album that has 20 songs.
Beyonce has released 55 songs in total.
Prove that the total number of songs in the first two albums is 30.
-/

theorem songs_in_first_two_albums {A B : ℕ} 
  (h1 : 5 + A + B + 20 = 55) : 
  A + B = 30 :=
by
  sorry

end songs_in_first_two_albums_l580_58011


namespace no_real_sqrt_neg_six_pow_three_l580_58028

theorem no_real_sqrt_neg_six_pow_three : 
  ∀ x : ℝ, 
    (¬ ∃ y : ℝ, y * y = -6 ^ 3) :=
by
  sorry

end no_real_sqrt_neg_six_pow_three_l580_58028


namespace side_length_of_square_l580_58067

-- Define the areas of the triangles AOR, BOP, and CRQ
def S1 := 1
def S2 := 3
def S3 := 1

-- Prove that the side length of the square OPQR is 2
theorem side_length_of_square (side_length : ℝ) : 
  S1 = 1 ∧ S2 = 3 ∧ S3 = 1 → side_length = 2 :=
by
  intros h
  sorry

end side_length_of_square_l580_58067


namespace amaya_movie_watching_time_l580_58043

theorem amaya_movie_watching_time :
  let uninterrupted_time_1 := 35
  let uninterrupted_time_2 := 45
  let uninterrupted_time_3 := 20
  let rewind_time_1 := 5
  let rewind_time_2 := 15
  let total_uninterrupted := uninterrupted_time_1 + uninterrupted_time_2 + uninterrupted_time_3
  let total_rewind := rewind_time_1 + rewind_time_2
  let total_time := total_uninterrupted + total_rewind
  total_time = 120 := by
  sorry

end amaya_movie_watching_time_l580_58043


namespace initial_outlay_l580_58008

-- Definition of given conditions
def manufacturing_cost (I : ℝ) (sets : ℕ) (cost_per_set : ℝ) : ℝ := I + sets * cost_per_set
def revenue (sets : ℕ) (price_per_set : ℝ) : ℝ := sets * price_per_set
def profit (revenue manufacturing_cost : ℝ) : ℝ := revenue - manufacturing_cost

-- Given data
def sets : ℕ := 500
def cost_per_set : ℝ := 20
def price_per_set : ℝ := 50
def given_profit : ℝ := 5000

-- The statement to prove
theorem initial_outlay (I : ℝ) : 
  profit (revenue sets price_per_set) (manufacturing_cost I sets cost_per_set) = given_profit → 
  I = 10000 := by
  sorry

end initial_outlay_l580_58008


namespace sum_n_k_eq_eight_l580_58080

theorem sum_n_k_eq_eight (n k : Nat) (h1 : 4 * k = n - 3) (h2 : 8 * k + 13 = 3 * n) : n + k = 8 :=
by
  sorry

end sum_n_k_eq_eight_l580_58080


namespace expression_expansion_l580_58006

noncomputable def expand_expression : Polynomial ℤ :=
 -2 * (5 * Polynomial.X^3 - 7 * Polynomial.X^2 + Polynomial.X - 4)

theorem expression_expansion :
  expand_expression = -10 * Polynomial.X^3 + 14 * Polynomial.X^2 - 2 * Polynomial.X + 8 :=
by
  sorry

end expression_expansion_l580_58006


namespace solve_special_sequence_l580_58089

noncomputable def special_sequence (a : ℕ → ℕ) : Prop :=
  a 1 = 1010 ∧ a 2 = 1015 ∧ ∀ n ≥ 1, a n + a (n + 1) + a (n + 2) = 2 * n + 1

theorem solve_special_sequence :
  ∃ a : ℕ → ℕ, special_sequence a ∧ a 1000 = 1676 :=
by
  sorry

end solve_special_sequence_l580_58089


namespace right_triangle_primes_l580_58083

-- Define what it means for a number to be a prime
def is_prime (n : ℕ) : Prop := ∃ m : ℕ, m > 1 ∧ m < n ∧ n % m = 0

-- State the problem
theorem right_triangle_primes
  (a b : ℕ)
  (ha : is_prime a)
  (hb : is_prime b)
  (a_gt_b : a > b)
  (a_plus_b : a + b = 90)
  (a_minus_b_prime : is_prime (a - b)) :
  b = 17 :=
sorry

end right_triangle_primes_l580_58083


namespace ellipse_equation_x_axis_ellipse_equation_y_axis_parabola_equation_x_axis_parabola_equation_y_axis_l580_58022

-- Define the conditions for the ellipse problem
def major_axis_length : ℝ := 10
def focal_length : ℝ := 4

-- Define the conditions for the parabola problem
def point_P : ℝ × ℝ := (-2, -4)

-- The equations to be proven
theorem ellipse_equation_x_axis :
  2 * (5 : ℝ) = major_axis_length ∧ 2 * (2 : ℝ) = focal_length →
  (5 : ℝ)^2 - (2 : ℝ)^2 = 21 →
  (∀ x y : ℝ, x^2 / 25 + y^2 / 21 = 1) := sorry

theorem ellipse_equation_y_axis :
  2 * (5 : ℝ) = major_axis_length ∧ 2 * (2 : ℝ) = focal_length →
  (5 : ℝ)^2 - (2 : ℝ)^2 = 21 →
  (∀ x y : ℝ, y^2 / 25 + x^2 / 21 = 1) := sorry

theorem parabola_equation_x_axis :
  point_P = (-2, -4) →
  (∀ x y : ℝ, y^2 = -8 * x) := sorry

theorem parabola_equation_y_axis :
  point_P = (-2, -4) →
  (∀ x y : ℝ, x^2 = -y) := sorry

end ellipse_equation_x_axis_ellipse_equation_y_axis_parabola_equation_x_axis_parabola_equation_y_axis_l580_58022


namespace cost_of_2000_pieces_of_gum_l580_58056

theorem cost_of_2000_pieces_of_gum
  (cost_per_piece_in_cents : Nat)
  (pieces_of_gum : Nat)
  (conversion_rate_cents_to_dollars : Nat)
  (h1 : cost_per_piece_in_cents = 5)
  (h2 : pieces_of_gum = 2000)
  (h3 : conversion_rate_cents_to_dollars = 100) :
  (cost_per_piece_in_cents * pieces_of_gum) / conversion_rate_cents_to_dollars = 100 := 
by
  sorry

end cost_of_2000_pieces_of_gum_l580_58056


namespace annual_interest_earned_l580_58027
noncomputable section

-- Define the total money
def total_money : ℝ := 3200

-- Define the first part of the investment
def P1 : ℝ := 800

-- Define the second part of the investment as total money minus the first part
def P2 : ℝ := total_money - P1

-- Define the interest rates for both parts
def rate1 : ℝ := 0.03
def rate2 : ℝ := 0.05

-- Define the time period (in years)
def time_period : ℝ := 1

-- Define the interest earned from each part
def interest1 : ℝ := P1 * rate1 * time_period
def interest2 : ℝ := P2 * rate2 * time_period

-- The total interest earned from both investments
def total_interest : ℝ := interest1 + interest2

-- The proof statement
theorem annual_interest_earned : total_interest = 144 := by
  sorry

end annual_interest_earned_l580_58027


namespace units_digit_of_30_factorial_is_0_l580_58063

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def units_digit (n : ℕ) : ℕ :=
  n % 10

theorem units_digit_of_30_factorial_is_0 : units_digit (factorial 30) = 0 := by
  sorry

end units_digit_of_30_factorial_is_0_l580_58063


namespace gcd_204_85_l580_58076

theorem gcd_204_85 : Nat.gcd 204 85 = 17 := by
  sorry

end gcd_204_85_l580_58076


namespace find_c_l580_58041

noncomputable def f (x c : ℝ) := x * (x - c) ^ 2
noncomputable def f' (x c : ℝ) := 3 * x ^ 2 - 4 * c * x + c ^ 2
noncomputable def f'' (x c : ℝ) := 6 * x - 4 * c

theorem find_c (c : ℝ) : f' 2 c = 0 ∧ f'' 2 c < 0 → c = 6 :=
by {
  sorry
}

end find_c_l580_58041


namespace exists_five_numbers_l580_58031

theorem exists_five_numbers :
  ∃ a1 a2 a3 a4 a5 : ℤ,
  a1 + a2 < 0 ∧
  a2 + a3 < 0 ∧
  a3 + a4 < 0 ∧
  a4 + a5 < 0 ∧
  a5 + a1 < 0 ∧
  a1 + a2 + a3 + a4 + a5 > 0 :=
by
  sorry

end exists_five_numbers_l580_58031


namespace product_of_divisors_of_30_l580_58005

open Nat

def divisors_of_30 : List ℕ := [1, 2, 3, 5, 6, 10, 15, 30]

theorem product_of_divisors_of_30 :
  (divisors_of_30.foldr (· * ·) 1) = 810000 := by
  sorry

end product_of_divisors_of_30_l580_58005


namespace ivanov_voted_against_kuznetsov_l580_58010

theorem ivanov_voted_against_kuznetsov
    (members : List String)
    (vote : String → String)
    (majority_dismissed : (String × Nat))
    (petrov_statement : String)
    (ivanov_concluded : Bool) :
  members = ["Ivanov", "Petrov", "Sidorov", "Kuznetsov"] →
  (∀ x ∈ members, vote x ∈ members ∧ vote x ≠ x) →
  majority_dismissed = ("Ivanov", 3) →
  petrov_statement = "Petrov voted against Kuznetsov" →
  ivanov_concluded = True →
  vote "Ivanov" = "Kuznetsov" :=
by
  intros members_cond vote_cond majority_cond petrov_cond ivanov_cond
  sorry

end ivanov_voted_against_kuznetsov_l580_58010


namespace seventh_root_of_unity_sum_l580_58087

theorem seventh_root_of_unity_sum (z : ℂ) (h1 : z^7 = 1) (h2 : z ≠ 1) :
  z + z^2 + z^4 = (-1 + Complex.I * Real.sqrt 11) / 2 ∨ z + z^2 + z^4 = (-1 - Complex.I * Real.sqrt 11) / 2 := 
by sorry

end seventh_root_of_unity_sum_l580_58087


namespace theorem_perimeter_shaded_region_theorem_area_shaded_region_l580_58036

noncomputable section

-- Definitions based on the conditions
def r : ℝ := Real.sqrt (1 / Real.pi)  -- radius of the unit circle

-- Define the perimeter and area functions for the shaded region
def perimeter_shaded_region (r : ℝ) : ℝ :=
  2 * Real.sqrt Real.pi

def area_shaded_region (r : ℝ) : ℝ :=
  1 / 5

-- Main theorem statements to prove
theorem theorem_perimeter_shaded_region
  (h : Real.pi * r^2 = 1) : perimeter_shaded_region r = 2 * Real.sqrt Real.pi :=
by
  sorry

theorem theorem_area_shaded_region
  (h : Real.pi * r^2 = 1) : area_shaded_region r = 1 / 5 :=
by
  sorry

end theorem_perimeter_shaded_region_theorem_area_shaded_region_l580_58036


namespace min_total_cost_l580_58062

-- Defining the variables involved
variables (x y z : ℝ)
variables (h : ℝ := 1) (V : ℝ := 4)
def base_cost (x y : ℝ) : ℝ := 200 * (x * y)
def side_cost (x y : ℝ) (h : ℝ) : ℝ := 100 * (2 * (x + y)) * h
def total_cost (x y h : ℝ) : ℝ := base_cost x y + side_cost x y h

-- The condition that volume is 4 m^3
theorem min_total_cost : 
  (∀ x y, x * y = V) → 
  ∃ x y, total_cost x y h = 1600 :=
by
  sorry

end min_total_cost_l580_58062


namespace compare_a_b_l580_58072

theorem compare_a_b (a b : ℝ) (h : 5 * (a - 1) = b + a ^ 2) : a > b :=
sorry

end compare_a_b_l580_58072


namespace expressing_population_in_scientific_notation_l580_58048

def population_in_scientific_notation (population : ℝ) : Prop :=
  population = 1.412 * 10^9

theorem expressing_population_in_scientific_notation : 
  population_in_scientific_notation (1.412 * 10^9) :=
by
  sorry

end expressing_population_in_scientific_notation_l580_58048


namespace necessarily_positive_l580_58064

theorem necessarily_positive (a b c : ℝ) (ha : 0 < a ∧ a < 2) (hb : -2 < b ∧ b < 0) (hc : 0 < c ∧ c < 3) :
  (b + c) > 0 :=
sorry

end necessarily_positive_l580_58064


namespace amount_exceeds_l580_58052

theorem amount_exceeds (N : ℕ) (A : ℕ) (h1 : N = 1925) (h2 : N / 7 - N / 11 = A) :
  A = 100 :=
sorry

end amount_exceeds_l580_58052


namespace sphere_surface_area_l580_58077

variable (x y z : ℝ)

theorem sphere_surface_area :
  (x^2 + y^2 + z^2 = 1) → (4 * Real.pi) = 4 * Real.pi :=
by
  intro h
  -- The proof will be inserted here
  sorry

end sphere_surface_area_l580_58077


namespace range_of_values_for_m_l580_58037

theorem range_of_values_for_m (m : ℝ) :
  (∃ x : ℝ, |x - 3| + |x - 4| < m) → m > 1 :=
by
  sorry

end range_of_values_for_m_l580_58037


namespace number_of_mismatching_socks_l580_58058

def SteveTotalSocks := 48
def StevePairsMatchingSocks := 11

theorem number_of_mismatching_socks :
  SteveTotalSocks - (StevePairsMatchingSocks * 2) = 26 := by
  sorry

end number_of_mismatching_socks_l580_58058
