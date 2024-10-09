import Mathlib

namespace sum_of_cosines_bounds_l2428_242803

theorem sum_of_cosines_bounds (x₁ x₂ x₃ x₄ x₅ : ℝ) 
  (h₁ : 0 ≤ x₁ ∧ x₁ ≤ π / 2)
  (h₂ : 0 ≤ x₂ ∧ x₂ ≤ π / 2)
  (h₃ : 0 ≤ x₃ ∧ x₃ ≤ π / 2)
  (h₄ : 0 ≤ x₄ ∧ x₄ ≤ π / 2)
  (h₅ : 0 ≤ x₅ ∧ x₅ ≤ π / 2)
  (sum_sines_eq : Real.sin x₁ + Real.sin x₂ + Real.sin x₃ + Real.sin x₄ + Real.sin x₅ = 3) : 
  2 ≤ Real.cos x₁ + Real.cos x₂ + Real.cos x₃ + Real.cos x₄ + Real.cos x₅ ∧ 
      Real.cos x₁ + Real.cos x₂ + Real.cos x₃ + Real.cos x₄ + Real.cos x₅ ≤ 4 :=
by
  sorry

end sum_of_cosines_bounds_l2428_242803


namespace linear_function_does_not_pass_first_quadrant_l2428_242808

theorem linear_function_does_not_pass_first_quadrant (k b : ℝ) (h : ∀ x : ℝ, y = k * x + b) :
  k = -1 → b = -2 → ¬∃ x y : ℝ, x > 0 ∧ y > 0 ∧ y = k * x + b :=
by
  sorry

end linear_function_does_not_pass_first_quadrant_l2428_242808


namespace mass_increase_l2428_242832

theorem mass_increase (ρ₁ ρ₂ m₁ m₂ a₁ a₂ : ℝ) (cond1 : ρ₂ = 2 * ρ₁) 
                      (cond2 : a₂ = 2 * a₁) (cond3 : m₁ = ρ₁ * (a₁^3)) 
                      (cond4 : m₂ = ρ₂ * (a₂^3)) : 
                      ((m₂ - m₁) / m₁) * 100 = 1500 := by
  sorry

end mass_increase_l2428_242832


namespace closest_point_on_line_l2428_242821

theorem closest_point_on_line (x y : ℝ) (h : y = (x - 3) / 3) : 
  (∃ p : ℝ × ℝ, p = (4, -2) ∧ ∀ q : ℝ × ℝ, (q.1, q.2) = (x, y) ∧ q ≠ p → dist p q ≥ dist p (33/10, 1/10)) :=
sorry

end closest_point_on_line_l2428_242821


namespace point_on_x_axis_l2428_242835

theorem point_on_x_axis (a : ℝ) (h : (1, a + 1).snd = 0) : a = -1 :=
by
  sorry

end point_on_x_axis_l2428_242835


namespace find_xyz_sum_cube_l2428_242806

variable (x y z c d : ℝ) 

theorem find_xyz_sum_cube (h1 : x * y * z = c) (h2 : 1 / x^3 + 1 / y^3 + 1 / z^3 = d) :
  (x + y + z)^3 = d * c^3 + 3 * c - 3 * c * d := 
by
  sorry

end find_xyz_sum_cube_l2428_242806


namespace discriminant_eq_complete_square_form_l2428_242812

theorem discriminant_eq_complete_square_form (a b c t : ℝ) (h : a ≠ 0) (ht : a * t^2 + b * t + c = 0) :
  (b^2 - 4 * a * c) = (2 * a * t + b)^2 := 
sorry

end discriminant_eq_complete_square_form_l2428_242812


namespace nim_maximum_product_l2428_242831

def nim_max_product (x y : ℕ) : ℕ :=
43 * 99 * x * y

theorem nim_maximum_product :
  ∃ x y : ℕ, (43 ≠ 0) ∧ (99 ≠ 0) ∧ (x ≠ 0) ∧ (y ≠ 0) ∧
  (43 + 99 + x + y = 0) ∧ (nim_max_product x y = 7704) :=
sorry

end nim_maximum_product_l2428_242831


namespace daily_profit_at_35_yuan_selling_price_for_600_profit_selling_price_impossible_for_900_profit_l2428_242811

-- Definitions based on given conditions
noncomputable def purchase_price : ℝ := 30
noncomputable def max_selling_price : ℝ := 55
noncomputable def daily_sales_volume (x : ℝ) : ℝ := -2 * x + 140

-- Definition of daily profit based on selling price x
noncomputable def daily_profit (x : ℝ) : ℝ := (x - purchase_price) * daily_sales_volume x

-- Lean 4 statements for the proofs
theorem daily_profit_at_35_yuan : daily_profit 35 = 350 := sorry

theorem selling_price_for_600_profit : ∃ x, 30 ≤ x ∧ x ≤ 55 ∧ daily_profit x = 600 ∧ x = 40 := sorry

theorem selling_price_impossible_for_900_profit :
  ∀ x, 30 ≤ x ∧ x ≤ 55 → daily_profit x ≠ 900 := sorry

end daily_profit_at_35_yuan_selling_price_for_600_profit_selling_price_impossible_for_900_profit_l2428_242811


namespace yoongi_hoseok_age_sum_l2428_242801

-- Definitions of given conditions
def age_aunt : ℕ := 38
def diff_aunt_yoongi : ℕ := 23
def diff_yoongi_hoseok : ℕ := 4

-- Definitions related to ages of Yoongi and Hoseok derived from given conditions
def age_yoongi : ℕ := age_aunt - diff_aunt_yoongi
def age_hoseok : ℕ := age_yoongi - diff_yoongi_hoseok

-- The theorem we need to prove
theorem yoongi_hoseok_age_sum : age_yoongi + age_hoseok = 26 := by
  sorry

end yoongi_hoseok_age_sum_l2428_242801


namespace average_speed_of_train_l2428_242814

theorem average_speed_of_train
  (d1 d2 : ℝ) (t1 t2 : ℝ)
  (h1 : d1 = 290) (h2 : d2 = 400) (h3 : t1 = 4.5) (h4 : t2 = 5.5) :
  ((d1 + d2) / (t1 + t2)) = 69 :=
by
  -- proof steps can be filled in later
  sorry

end average_speed_of_train_l2428_242814


namespace decrease_in_combined_area_l2428_242826

theorem decrease_in_combined_area (r1 r2 r3 : ℝ) :
    let π := Real.pi
    let A_original := π * (r1 ^ 2) + π * (r2 ^ 2) + π * (r3 ^ 2)
    let r1' := r1 * 0.5
    let r2' := r2 * 0.5
    let r3' := r3 * 0.5
    let A_new := π * (r1' ^ 2) + π * (r2' ^ 2) + π * (r3' ^ 2)
    let Decrease := A_original - A_new
    Decrease = 0.75 * π * (r1 ^ 2) + 0.75 * π * (r2 ^ 2) + 0.75 * π * (r3 ^ 2) :=
by
  sorry

end decrease_in_combined_area_l2428_242826


namespace total_books_borrowed_lunchtime_correct_l2428_242818

def shelf_A_borrowed (X : ℕ) : Prop :=
  110 - X = 60 ∧ X = 50

def shelf_B_borrowed (Y : ℕ) : Prop :=
  150 - 50 + 20 - Y = 80 ∧ Y = 80

def shelf_C_borrowed (Z : ℕ) : Prop :=
  210 - 45 = 165 ∧ 165 - 130 = Z ∧ Z = 35

theorem total_books_borrowed_lunchtime_correct :
  ∃ (X Y Z : ℕ),
    shelf_A_borrowed X ∧
    shelf_B_borrowed Y ∧
    shelf_C_borrowed Z ∧
    X + Y + Z = 165 :=
by
  sorry

end total_books_borrowed_lunchtime_correct_l2428_242818


namespace num_audio_cassettes_in_second_set_l2428_242834

-- Define the variables and constants
def costOfAudio (A : ℕ) : ℕ := A
def costOfVideo (V : ℕ) : ℕ := V
def totalCost (numOfAudio : ℕ) (numOfVideo : ℕ) (A : ℕ) (V : ℕ) : ℕ :=
  numOfAudio * (costOfAudio A) + numOfVideo * (costOfVideo V)

-- Given conditions
def condition1 (A V : ℕ) : Prop := ∃ X : ℕ, totalCost X 4 A V = 1350
def condition2 (A V : ℕ) : Prop := totalCost 7 3 A V = 1110
def condition3 : Prop := costOfVideo 300 = 300

-- Main theorem to prove: The number of audio cassettes in the second set is 7
theorem num_audio_cassettes_in_second_set :
  ∃ (A : ℕ), condition1 A 300 ∧ condition2 A 300 ∧ condition3 →
  7 = 7 :=
by
  sorry

end num_audio_cassettes_in_second_set_l2428_242834


namespace find_x_l2428_242817

variables (a b c k : ℝ) (h : k ≠ 0)

theorem find_x (x y z : ℝ)
  (h1 : (xy + k) / (x + y) = a)
  (h2 : (xz + k) / (x + z) = b)
  (h3 : (yz + k) / (y + z) = c) :
  x = 2 * a * b * c * d / (b * (a * c - k) + c * (a * b - k) - a * (b * c - k)) := sorry

end find_x_l2428_242817


namespace jason_car_cost_l2428_242802

theorem jason_car_cost
    (down_payment : ℕ := 8000)
    (monthly_payment : ℕ := 525)
    (months : ℕ := 48)
    (interest_rate : ℝ := 0.05) :
    (down_payment + monthly_payment * months + interest_rate * (monthly_payment * months)) = 34460 := 
by
  sorry

end jason_car_cost_l2428_242802


namespace cost_per_tshirt_l2428_242820
-- Import necessary libraries

-- Define the given conditions
def t_shirts : ℕ := 20
def total_cost : ℝ := 199

-- Define the target proof statement
theorem cost_per_tshirt : (total_cost / t_shirts) = 9.95 := 
sorry

end cost_per_tshirt_l2428_242820


namespace transistors_in_2010_l2428_242815

theorem transistors_in_2010 
  (initial_transistors : ℕ) 
  (initial_year : ℕ) 
  (final_year : ℕ) 
  (doubling_period : ℕ)
  (initial_transistors_eq: initial_transistors = 500000)
  (initial_year_eq: initial_year = 1985)
  (final_year_eq: final_year = 2010)
  (doubling_period_eq : doubling_period = 2) :
  initial_transistors * 2^((final_year - initial_year) / doubling_period) = 2048000000 := 
by 
  -- the proof goes here
  sorry

end transistors_in_2010_l2428_242815


namespace lilibeth_and_friends_strawberries_l2428_242824

-- Define the conditions
def baskets_filled_by_lilibeth : ℕ := 6
def strawberries_per_basket : ℕ := 50
def friends_count : ℕ := 3

-- Define the total number of strawberries picked by Lilibeth and her friends 
def total_strawberries_picked : ℕ :=
  (baskets_filled_by_lilibeth * strawberries_per_basket) * (1 + friends_count)

-- The theorem to prove
theorem lilibeth_and_friends_strawberries : total_strawberries_picked = 1200 := 
by
  sorry

end lilibeth_and_friends_strawberries_l2428_242824


namespace intersection_of_sets_l2428_242800

variable (A B : Set ℝ) (x : ℝ)

def setA : Set ℝ := { x | x > 0 }
def setB : Set ℝ := { x | -1 < x ∧ x ≤ 2 }

theorem intersection_of_sets : A ∩ B = { x | 0 < x ∧ x ≤ 2 } :=
by
  sorry

end intersection_of_sets_l2428_242800


namespace kaleb_toys_can_buy_l2428_242805

theorem kaleb_toys_can_buy (saved_money : ℕ) (allowance_received : ℕ) (allowance_increase_percent : ℕ) (toy_cost : ℕ) (half_total_spend : ℕ) :
  saved_money = 21 →
  allowance_received = 15 →
  allowance_increase_percent = 20 →
  toy_cost = 6 →
  half_total_spend = (saved_money + allowance_received) / 2 →
  (half_total_spend / toy_cost) = 3 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end kaleb_toys_can_buy_l2428_242805


namespace ratio_problem_l2428_242825

theorem ratio_problem
  (A B C : ℚ)
  (h : A / B = 2 / 1)
  (h1 : B / C = 1 / 4) :
  (3 * A + 2 * B) / (4 * C - A) = 4 / 7 := 
sorry

end ratio_problem_l2428_242825


namespace original_square_area_l2428_242819

theorem original_square_area :
  ∀ (a b : ℕ), 
  (a * a = 24 * 1 * 1 + b * b ∧ 
  ((∃ m n : ℕ, (a + b = m ∧ a - b = n ∧ m * n = 24) ∨ 
  (a + b = n ∧ a - b = m ∧ m * n = 24)))) →
  a * a = 25 :=
by
  sorry

end original_square_area_l2428_242819


namespace cos_eight_arccos_one_fourth_l2428_242830

theorem cos_eight_arccos_one_fourth :
  Real.cos (8 * Real.arccos (1 / 4)) = 172546 / 1048576 :=
sorry

end cos_eight_arccos_one_fourth_l2428_242830


namespace decreasing_power_function_l2428_242828

open Nat

/-- For the power function y = x^(m^2 + 2*m - 3) (where m : ℕ) 
    to be a decreasing function in the interval (0, +∞), prove that m = 0. -/
theorem decreasing_power_function (m : ℕ) (h : m^2 + 2 * m - 3 < 0) : m = 0 := 
by
  sorry

end decreasing_power_function_l2428_242828


namespace original_price_l2428_242833

theorem original_price (x: ℝ) (h1: x * 1.1 * 0.8 = 2) : x = 25 / 11 :=
by
  sorry

end original_price_l2428_242833


namespace James_age_is_47_5_l2428_242810

variables (James_Age Mara_Age : ℝ)

def condition1 : Prop := James_Age = 3 * Mara_Age - 20
def condition2 : Prop := James_Age + Mara_Age = 70

theorem James_age_is_47_5 (h1 : condition1 James_Age Mara_Age) (h2 : condition2 James_Age Mara_Age) : James_Age = 47.5 :=
by
  sorry

end James_age_is_47_5_l2428_242810


namespace ship_distances_l2428_242807

-- Define the conditions based on the initial problem statement
variables (f : ℕ → ℝ)
def distances_at_known_times : Prop :=
  f 0 = 49 ∧ f 2 = 25 ∧ f 3 = 121

-- Define the questions to prove the distances at unknown times
def distance_at_time_1 : Prop :=
  f 1 = 1

def distance_at_time_4 : Prop :=
  f 4 = 289

-- The proof problem
theorem ship_distances
  (f : ℕ → ℝ)
  (hf : ∀ t, ∃ a b c, f t = a*t^2 + b*t + c)
  (h_known : distances_at_known_times f) :
  distance_at_time_1 f ∧ distance_at_time_4 f :=
by
  sorry

end ship_distances_l2428_242807


namespace bird_costs_l2428_242809

-- Define the cost of a small bird and a large bird
def cost_small_bird (x : ℕ) := x
def cost_large_bird (x : ℕ) := 2 * x

-- Define total cost calculations for the first and second ladies
def cost_first_lady (x : ℕ) := 5 * cost_large_bird x + 3 * cost_small_bird x
def cost_second_lady (x : ℕ) := 5 * cost_small_bird x + 3 * cost_large_bird x

-- State the main theorem
theorem bird_costs (x : ℕ) (hx : cost_first_lady x = cost_second_lady x + 20) : 
(cost_small_bird x = 10) ∧ (cost_large_bird x = 20) := 
by {
  sorry
}

end bird_costs_l2428_242809


namespace min_a_l2428_242804

theorem min_a (a : ℝ) (h : ∀ x y : ℝ, 0 < x → 0 < y → (x + y) * (1/x + a/y) ≥ 25) : a ≥ 16 :=
sorry  -- Proof is omitted

end min_a_l2428_242804


namespace optimal_selling_price_l2428_242813

-- Define the constants given in the problem
def purchase_price : ℝ := 40
def initial_selling_price : ℝ := 50
def initial_sales_volume : ℝ := 50

-- Define the function that represents the profit based on the change in price x
def profit (x : ℝ) : ℝ := (initial_selling_price + x) * (initial_sales_volume - x) - (initial_sales_volume - x) * purchase_price

-- State the theorem
theorem optimal_selling_price : ∃ x : ℝ, profit x = -x^2 + 40*x + 500 ∧ (initial_selling_price + x = 70) :=
by
  sorry

end optimal_selling_price_l2428_242813


namespace height_difference_l2428_242823

theorem height_difference (B_height A_height : ℝ) (h : A_height = 0.6 * B_height) :
  (B_height - A_height) / A_height * 100 = 66.67 := 
sorry

end height_difference_l2428_242823


namespace y_value_when_x_neg_one_l2428_242827

theorem y_value_when_x_neg_one (t : ℝ) (x y : ℝ) 
  (h1 : x = 3 - 2 * t) 
  (h2 : y = t^2 + 3 * t + 6) 
  (h3 : x = -1) : 
  y = 16 := 
by sorry

end y_value_when_x_neg_one_l2428_242827


namespace eq_positive_root_a_value_l2428_242829

theorem eq_positive_root_a_value (x a : ℝ) (hx : x > 0) :
  ((x + a) / (x + 3) - 2 / (x + 3) = 0) → a = 5 :=
by
  sorry

end eq_positive_root_a_value_l2428_242829


namespace range_of_a_l2428_242816

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, |x - 3| + |x - a| < 4) → -1 < a ∧ a < 7 :=
  sorry

end range_of_a_l2428_242816


namespace cakes_bought_l2428_242822

theorem cakes_bought (initial_cakes remaining_cakes : ℕ) (h_initial : initial_cakes = 155) (h_remaining : remaining_cakes = 15) : initial_cakes - remaining_cakes = 140 :=
by {
  sorry
}

end cakes_bought_l2428_242822
