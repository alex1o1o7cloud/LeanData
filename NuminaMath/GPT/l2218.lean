import Mathlib

namespace probability_of_different_colors_l2218_221896

noncomputable def total_chips := 6 + 5 + 4

noncomputable def prob_diff_color : ℚ :=
  let pr_blue := 6 / total_chips
  let pr_red := 5 / total_chips
  let pr_yellow := 4 / total_chips

  let pr_not_blue := (5 + 4) / total_chips
  let pr_not_red := (6 + 4) / total_chips
  let pr_not_yellow := (6 + 5) / total_chips

  pr_blue * pr_not_blue + pr_red * pr_not_red + pr_yellow * pr_not_yellow

theorem probability_of_different_colors :
  prob_diff_color = 148 / 225 :=
sorry

end probability_of_different_colors_l2218_221896


namespace cubic_roots_identity_l2218_221821

theorem cubic_roots_identity 
  (x1 x2 x3 p q : ℝ) 
  (hq : ∀ x, x^3 + p * x + q = (x - x1) * (x - x2) * (x - x3))
  (h_sum : x1 + x2 + x3 = 0)
  (h_prod : x1 * x2 + x2 * x3 + x3 * x1 = p):
  x2^2 + x2 * x3 + x3^2 = -p ∧ x1^2 + x1 * x3 + x3^2 = -p ∧ x1^2 + x1 * x2 + x2^2 = -p :=
sorry

end cubic_roots_identity_l2218_221821


namespace three_digit_number_divisible_by_8_and_even_tens_digit_l2218_221876

theorem three_digit_number_divisible_by_8_and_even_tens_digit (d : ℕ) (hd : d % 2 = 0) (hdiv : (100 * 5 + 10 * d + 4) % 8 = 0) :
  100 * 5 + 10 * d + 4 = 544 :=
by
  sorry

end three_digit_number_divisible_by_8_and_even_tens_digit_l2218_221876


namespace balls_picked_at_random_eq_two_l2218_221814

-- Define the initial conditions: number of balls of each color
def num_red_balls : ℕ := 5
def num_blue_balls : ℕ := 4
def num_green_balls : ℕ := 3

-- Define the total number of balls
def total_balls : ℕ := num_red_balls + num_blue_balls + num_green_balls

-- Define the given probability
def given_probability : ℚ := 0.15151515151515152

-- Define the probability calculation for picking two red balls
def probability_two_reds : ℚ :=
  (num_red_balls / total_balls) * ((num_red_balls - 1) / (total_balls - 1))

-- The theorem to prove
theorem balls_picked_at_random_eq_two :
  probability_two_reds = given_probability → n = 2 :=
by
  sorry

end balls_picked_at_random_eq_two_l2218_221814


namespace angelina_journey_equation_l2218_221884

theorem angelina_journey_equation (t : ℝ) :
    4 = t + 15/60 + (4 - 15/60 - t) →
    60 * t + 90 * (15/4 - t) = 255 :=
    by
    sorry

end angelina_journey_equation_l2218_221884


namespace minimum_value_l2218_221867

theorem minimum_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y = 1) : 
  (1 / x + 9 / y) = 16 :=
sorry

end minimum_value_l2218_221867


namespace variable_cost_per_book_fixed_cost_l2218_221831

theorem variable_cost_per_book_fixed_cost (fixed_costs : ℝ) (selling_price_per_book : ℝ) 
(number_of_books : ℝ) (total_costs total_revenue : ℝ) (variable_cost_per_book : ℝ) 
(h1 : fixed_costs = 35630) (h2 : selling_price_per_book = 20.25) (h3 : number_of_books = 4072)
(h4 : total_costs = fixed_costs + variable_cost_per_book * number_of_books)
(h5 : total_revenue = selling_price_per_book * number_of_books)
(h6 : total_costs = total_revenue) : variable_cost_per_book = 11.50 := by
  sorry

end variable_cost_per_book_fixed_cost_l2218_221831


namespace area_of_triangle_formed_by_tangent_line_l2218_221873
-- Import necessary libraries from Mathlib

-- Set up the problem
theorem area_of_triangle_formed_by_tangent_line
  (f : ℝ → ℝ) (h_f : ∀ x, f x = x^2) :
  let slope := (deriv f 1)
  let tangent_line (x : ℝ) := slope * (x - 1) + f 1
  let x_intercept := (0 : ℝ)
  let y_intercept := tangent_line 0
  let area := 0.5 * abs x_intercept * abs y_intercept
  area = 1 / 4 :=
by
  sorry -- Proof to be completed

end area_of_triangle_formed_by_tangent_line_l2218_221873


namespace find_radius_and_diameter_l2218_221898

theorem find_radius_and_diameter (M N r d : ℝ) (h1 : M = π * r^2) (h2 : N = 2 * π * r) (h3 : M / N = 15) : 
  (r = 30) ∧ (d = 60) := by
  sorry

end find_radius_and_diameter_l2218_221898


namespace diamond_fifteen_two_l2218_221845

def diamond (a b : ℤ) : ℤ := a + (a / (b + 1))

theorem diamond_fifteen_two : diamond 15 2 = 20 := 
by 
    sorry

end diamond_fifteen_two_l2218_221845


namespace find_p_l2218_221899

theorem find_p (m n p : ℝ) :
  m = (n / 7) - (2 / 5) →
  m + p = ((n + 21) / 7) - (2 / 5) →
  p = 3 := by
  sorry

end find_p_l2218_221899


namespace function_intersection_le_one_l2218_221879

theorem function_intersection_le_one (f : ℝ → ℝ)
  (h : ∀ x t : ℝ, t ≠ 0 → t * (f (x + t) - f x) > 0) :
  ∀ a : ℝ, ∃! x : ℝ, f x = a :=
by 
sorry

end function_intersection_le_one_l2218_221879


namespace find_y_coordinate_of_first_point_l2218_221849

theorem find_y_coordinate_of_first_point :
  ∃ y1 : ℝ, ∀ k : ℝ, (k = 0.8) → (k = (0.8 - y1) / (5 - (-1))) → y1 = 4 :=
by
  sorry

end find_y_coordinate_of_first_point_l2218_221849


namespace line_equation_unique_l2218_221874

theorem line_equation_unique (m b k : ℝ) (h_intersect_dist : |(k^2 + 6*k + 5) - (m*k + b)| = 7)
  (h_passing_point : 8 = 2*m + b) (hb_nonzero : b ≠ 0) :
  y = 10*x - 12 :=
by
  sorry

end line_equation_unique_l2218_221874


namespace other_bill_denomination_l2218_221820

-- Define the conditions of the problem
def cost_shirt : ℕ := 80
def ten_dollar_bills : ℕ := 2
def other_bills (x : ℕ) : ℕ := ten_dollar_bills + 1

-- The amount paid with $10 bills
def amount_with_ten_dollar_bills : ℕ := ten_dollar_bills * 10

-- The total amount should match the cost of the shirt
def total_amount (x : ℕ) : ℕ := amount_with_ten_dollar_bills + (other_bills x) * x

-- Statement to prove
theorem other_bill_denomination : 
  ∃ (x : ℕ), total_amount x = cost_shirt ∧ x = 20 :=
by
  sorry

end other_bill_denomination_l2218_221820


namespace problem_a_range_l2218_221853

theorem problem_a_range (a : ℝ) :
  (∀ x : ℝ, (a - 1) * x^2 - 2 * (a - 1) * x - 2 < 0) ↔ (-1 < a ∧ a ≤ 1) :=
by
  sorry

end problem_a_range_l2218_221853


namespace cocos_August_bill_l2218_221819

noncomputable def total_cost (a_monthly_cost: List (Float × Float)) :=
a_monthly_cost.foldr (fun x acc => (x.1 * x.2 * 0.09) + acc) 0

theorem cocos_August_bill :
  let oven        := (2.4, 25)
  let air_cond    := (1.6, 150)
  let refrigerator := (0.15, 720)
  let washing_mach := (0.5, 20) 
  total_cost [oven, air_cond, refrigerator, washing_mach] = 37.62 :=
by
  sorry

end cocos_August_bill_l2218_221819


namespace find_y_l2218_221827

theorem find_y (Y : ℝ) (h : (200 + 200 / Y) * Y = 18200) : Y = 90 :=
by
  sorry

end find_y_l2218_221827


namespace books_remaining_after_second_day_l2218_221842

variable (x a b c d : ℕ)

theorem books_remaining_after_second_day :
  let books_borrowed_first_day := a * b
  let books_borrowed_second_day := c
  let books_returned_second_day := (d * books_borrowed_first_day) / 100
  x - books_borrowed_first_day - books_borrowed_second_day + books_returned_second_day =
  x - (a * b) - c + ((d * (a * b)) / 100) :=
sorry

end books_remaining_after_second_day_l2218_221842


namespace initial_percentage_of_water_l2218_221894

/-
Initial conditions:
- Let W be the initial percentage of water in 10 liters of milk.
- The mixture should become 2% water after adding 15 liters of pure milk to it.
-/

theorem initial_percentage_of_water (W : ℚ) 
  (H1 : 0 < W) (H2 : W < 100) 
  (H3 : (10 * (100 - W) / 100 + 15) / 25 = 0.98) : 
  W = 5 := 
sorry

end initial_percentage_of_water_l2218_221894


namespace gcd_factorial_eight_nine_eq_8_factorial_l2218_221847

theorem gcd_factorial_eight_nine_eq_8_factorial : Nat.gcd (Nat.factorial 8) (Nat.factorial 9) = Nat.factorial 8 := 
by 
  sorry

end gcd_factorial_eight_nine_eq_8_factorial_l2218_221847


namespace solve_inequality_correct_l2218_221864

noncomputable def solve_inequality (a x : ℝ) : Set ℝ :=
  if a > 1 ∨ a < 0 then {x | x ≤ a ∨ x ≥ a^2 }
  else if a = 1 ∨ a = 0 then {x | True}
  else {x | x ≤ a^2 ∨ x ≥ a}

theorem solve_inequality_correct (a x : ℝ) :
  (x^2 - (a^2 + a) * x + a^3 ≥ 0) ↔ 
    (if a > 1 ∨ a < 0 then x ≤ a ∨ x ≥ a^2
      else if a = 1 ∨ a = 0 then True
      else x ≤ a^2 ∨ x ≥ a) :=
by sorry

end solve_inequality_correct_l2218_221864


namespace calculate_weekly_charge_l2218_221883

-- Defining conditions as constraints
def daily_charge : ℕ := 30
def total_days : ℕ := 11
def total_cost : ℕ := 310

-- Defining the weekly charge
def weekly_charge : ℕ := 190

-- Prove that the weekly charge for the first week of rental is $190
theorem calculate_weekly_charge (daily_charge total_days total_cost weekly_charge: ℕ) (daily_charge_eq : daily_charge = 30) (total_days_eq : total_days = 11) (total_cost_eq : total_cost = 310) : 
  weekly_charge = 190 :=
by
  sorry

end calculate_weekly_charge_l2218_221883


namespace find_prices_and_max_basketballs_l2218_221871

def unit_price_condition (x : ℕ) (y : ℕ) : Prop :=
  y = 2*x - 30

def cost_ratio_condition (x : ℕ) (y : ℕ) : Prop :=
  3 * x = 2 * y - 60

def total_cost_condition (total_cost : ℕ) (num_basketballs : ℕ) (num_soccerballs : ℕ) : Prop :=
  total_cost ≤ 15500 ∧ num_basketballs + num_soccerballs = 200

theorem find_prices_and_max_basketballs
  (x y : ℕ) (total_cost : ℕ) (num_basketballs : ℕ) (num_soccerballs : ℕ)
  (h1 : unit_price_condition x y)
  (h2 : cost_ratio_condition x y)
  (h3 : total_cost_condition total_cost num_basketballs num_soccerballs)
  (h4 : total_cost = 90 * num_basketballs + 60 * num_soccerballs)
  : x = 60 ∧ y = 90 ∧ num_basketballs ≤ 116 :=
sorry

end find_prices_and_max_basketballs_l2218_221871


namespace minimize_transport_cost_l2218_221818

noncomputable def total_cost (v : ℝ) (a : ℝ) : ℝ :=
  if v > 0 ∧ v ≤ 80 then
    1000 * (v / 4 + a / v)
  else
    0

theorem minimize_transport_cost :
  ∀ v a : ℝ, a = 400 → (0 < v ∧ v ≤ 80) → total_cost v a = 20000 → v = 40 :=
by
  intros v a ha h_dom h_cost
  sorry

end minimize_transport_cost_l2218_221818


namespace base_of_parallelogram_l2218_221804

variable (Area Height Base : ℝ)

def parallelogram_area (base height : ℝ) : ℝ := base * height

theorem base_of_parallelogram
  (h_area : Area = 200)
  (h_height : Height = 20)
  (h_area_def : parallelogram_area Base Height = Area) :
  Base = 10 :=
by sorry

end base_of_parallelogram_l2218_221804


namespace parallel_vectors_x_value_l2218_221833

def vectors_are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_x_value :
  ∀ x : ℝ, vectors_are_parallel (-1, 4) (x, 2) → x = -1 / 2 := 
by 
  sorry

end parallel_vectors_x_value_l2218_221833


namespace geometric_progression_first_term_one_l2218_221838

theorem geometric_progression_first_term_one (a r : ℝ) (gp : ℕ → ℝ)
  (h_gp : ∀ n, gp n = a * r^(n - 1))
  (h_product_in_gp : ∀ i j, ∃ k, gp i * gp j = gp k) :
  a = 1 := 
sorry

end geometric_progression_first_term_one_l2218_221838


namespace area_union_square_circle_l2218_221888

noncomputable def side_length_square : ℝ := 12
noncomputable def radius_circle : ℝ := 15
noncomputable def area_union : ℝ := 144 + 168.75 * Real.pi

theorem area_union_square_circle : 
  let area_square := side_length_square ^ 2
  let area_circle := Real.pi * radius_circle ^ 2
  let area_quarter_circle := area_circle / 4
  area_union = area_square + area_circle - area_quarter_circle :=
by
  -- The actual proof is omitted
  sorry

end area_union_square_circle_l2218_221888


namespace _l2218_221822

noncomputable def waiter_fraction_from_tips (S T I : ℝ) : Prop :=
  T = (5 / 2) * S ∧
  I = S + T ∧
  T / I = 5 / 7

lemma waiter_tips_fraction_theorem (S T I : ℝ) : waiter_fraction_from_tips S T I → T / I = 5 / 7 :=
by
  intro h
  rw [waiter_fraction_from_tips] at h
  obtain ⟨h₁, h₂, h₃⟩ := h
  exact h₃

end _l2218_221822


namespace greatest_least_S_T_l2218_221856

theorem greatest_least_S_T (a b c : ℝ) (h : a ≤ b ∧ b ≤ c) (triangle : a + b > c ∧ a + c > b ∧ b + c > a) :
  4 ≤ (a + b + c)^2 / (b * c) ∧ (a + b + c)^2 / (b * c) ≤ 9 :=
by sorry

end greatest_least_S_T_l2218_221856


namespace roof_ratio_l2218_221878

theorem roof_ratio (L W : ℝ) (h1 : L * W = 576) (h2 : L - W = 36) : L / W = 4 := 
by
  sorry

end roof_ratio_l2218_221878


namespace tangent_line_equation_l2218_221840

theorem tangent_line_equation (P : ℝ × ℝ) (hP : P = (-4, -3)) :
  ∃ (a b c : ℝ), a * -4 + b * -3 + c = 0 ∧ a * a + b * b = (5:ℝ)^2 ∧ 
                 a = 4 ∧ b = 3 ∧ c = 25 := 
sorry

end tangent_line_equation_l2218_221840


namespace gcd_of_g_y_l2218_221810

def g (y : ℕ) : ℕ := (3 * y + 4) * (8 * y + 3) * (11 * y + 5) * (y + 11)

theorem gcd_of_g_y (y : ℕ) (hy : ∃ k, y = 30492 * k) : Nat.gcd (g y) y = 660 :=
by
  sorry

end gcd_of_g_y_l2218_221810


namespace audrey_ratio_in_3_years_l2218_221893

-- Define the ages and the conditions
def Heracles_age : ℕ := 10
def Audrey_age := Heracles_age + 7
def Audrey_age_in_3_years := Audrey_age + 3

-- Statement: Prove that the ratio of Audrey's age in 3 years to Heracles' current age is 2:1
theorem audrey_ratio_in_3_years : (Audrey_age_in_3_years / Heracles_age) = 2 := sorry

end audrey_ratio_in_3_years_l2218_221893


namespace no_intersection_of_sets_l2218_221854

noncomputable def A (a b x y : ℝ) :=
  a * (Real.sin x + Real.sin y) + (b - 1) * (Real.cos x + Real.cos y) = 0

noncomputable def B (a b x y : ℝ) :=
  (b + 1) * Real.sin (x + y) - a * Real.cos (x + y) = a

noncomputable def C (a b : ℝ) :=
  ∀ z : ℝ, z^2 - 2 * (a - b) * z + (a + b)^2 - 2 > 0

theorem no_intersection_of_sets (a b x y : ℝ) (h1 : 0 < x) (h2 : x < Real.pi / 2) (h3 : 0 < y) (h4 : y < Real.pi / 2) :
  (C a b) → ¬(∃ x y, A a b x y ∧ B a b x y) :=
by 
  sorry

end no_intersection_of_sets_l2218_221854


namespace prove_b_value_l2218_221851

theorem prove_b_value (b : ℚ) (h : b + b / 4 = 10 / 4) : b = 2 :=
sorry

end prove_b_value_l2218_221851


namespace round_trip_percentage_l2218_221862

-- Definitions based on the conditions
variable (P : ℝ) -- Total number of passengers
variable (R : ℝ) -- Number of round-trip ticket holders

-- First condition: 20% of passengers held round-trip tickets and took their cars aboard
def condition1 := 0.20 * P = 0.60 * R

-- Second condition: 40% of passengers with round-trip tickets did not take their cars aboard (implies 60% did)
theorem round_trip_percentage (h1 : condition1 P R) : (R / P) * 100 = 33.33 := by
  sorry

end round_trip_percentage_l2218_221862


namespace translation_result_l2218_221891

variables (P : ℝ × ℝ) (P' : ℝ × ℝ)

def translate_left (P : ℝ × ℝ) (units : ℝ) := (P.1 - units, P.2)
def translate_down (P : ℝ × ℝ) (units : ℝ) := (P.1, P.2 - units)

theorem translation_result :
    P = (-4, 3) -> P' = translate_down (translate_left P 2) 2 -> P' = (-6, 1) :=
by
  intros h1 h2
  sorry

end translation_result_l2218_221891


namespace decreased_amount_l2218_221812

theorem decreased_amount {N A : ℝ} (h₁ : 0.20 * N - A = 6) (h₂ : N = 50) : A = 4 := by
  sorry

end decreased_amount_l2218_221812


namespace isosceles_triangle_l2218_221885

theorem isosceles_triangle (a b c : ℝ) (A B C : ℝ) (hAcosB : a * Real.cos B = b * Real.cos A) 
  (h_triangle : a + b > c ∧ a + c > b ∧ b + c > a) :
  (a = b ∨ b = c ∨ a = c) :=
sorry

end isosceles_triangle_l2218_221885


namespace find_f_10_l2218_221806

-- Defining the function f as an odd, periodic function with period 2
def odd_func_periodic (f : ℝ → ℝ) : Prop :=
  (∀ x : ℝ, f (-x) = - f x) ∧ (∀ x : ℝ, f (x + 2) = f x)

-- Stating the theorem that f(10) is 0 given the conditions
theorem find_f_10 (f : ℝ → ℝ) (h1 : odd_func_periodic f) : f 10 = 0 :=
sorry

end find_f_10_l2218_221806


namespace area_of_field_l2218_221877

theorem area_of_field (w l A : ℝ) 
    (h1 : l = 2 * w + 35) 
    (h2 : 2 * (w + l) = 700) : 
    A = 25725 :=
by sorry

end area_of_field_l2218_221877


namespace lana_extra_flowers_l2218_221848

theorem lana_extra_flowers (tulips roses used total extra : ℕ) 
  (h1 : tulips = 36) 
  (h2 : roses = 37) 
  (h3 : used = 70) 
  (h4 : total = tulips + roses) 
  (h5 : extra = total - used) : 
  extra = 3 := 
sorry

end lana_extra_flowers_l2218_221848


namespace cost_of_pink_notebook_l2218_221823

theorem cost_of_pink_notebook
    (total_cost : ℕ) 
    (black_cost : ℕ) 
    (green_cost : ℕ) 
    (num_green : ℕ) 
    (num_black : ℕ) 
    (num_pink : ℕ)
    (total_notebooks : ℕ)
    (h_total_cost : total_cost = 45)
    (h_black_cost : black_cost = 15) 
    (h_green_cost : green_cost = 10) 
    (h_num_green : num_green = 2) 
    (h_num_black : num_black = 1) 
    (h_num_pink : num_pink = 1)
    (h_total_notebooks : total_notebooks = 4) 
    : (total_cost - (num_green * green_cost + black_cost) = 10) :=
by
  sorry

end cost_of_pink_notebook_l2218_221823


namespace total_monsters_l2218_221816

theorem total_monsters (a1 a2 a3 a4 a5 : ℕ) 
  (h1 : a1 = 2) 
  (h2 : a2 = 2 * a1) 
  (h3 : a3 = 2 * a2) 
  (h4 : a4 = 2 * a3) 
  (h5 : a5 = 2 * a4) : 
  a1 + a2 + a3 + a4 + a5 = 62 :=
by
  sorry

end total_monsters_l2218_221816


namespace program_output_l2218_221863

theorem program_output :
  let a := 1
  let b := 3
  let a := a + b
  let b := b * a
  a = 4 ∧ b = 12 :=
by
  sorry

end program_output_l2218_221863


namespace fraction_sum_eq_l2218_221829

-- Given conditions
variables (w x y : ℝ)
axiom hx : w / x = 1 / 6
axiom hy : w / y = 1 / 5

-- Proof goal
theorem fraction_sum_eq : (x + y) / y = 11 / 5 :=
by sorry

end fraction_sum_eq_l2218_221829


namespace largest_number_l2218_221850

theorem largest_number (a b c d : ℤ)
  (h1 : (a + b + c) / 3 + d = 17)
  (h2 : (a + b + d) / 3 + c = 21)
  (h3 : (a + c + d) / 3 + b = 23)
  (h4 : (b + c + d) / 3 + a = 29) :
  d = 21 := 
sorry

end largest_number_l2218_221850


namespace find_y_given_conditions_l2218_221837

theorem find_y_given_conditions (x y : ℝ) (hx : x = 102) 
                                (h : x^3 * y - 3 * x^2 * y + 3 * x * y = 106200) : 
  y = 10 / 97 :=
by
  sorry

end find_y_given_conditions_l2218_221837


namespace wall_building_time_l2218_221866

theorem wall_building_time
  (m1 m2 : ℕ) 
  (d1 d2 : ℝ)
  (h1 : m1 = 20)
  (h2 : d1 = 3.0)
  (h3 : m2 = 30)
  (h4 : ∃ k, m1 * d1 = k ∧ m2 * d2 = k) :
  d2 = 2.0 :=
by
  sorry

end wall_building_time_l2218_221866


namespace reciprocal_sum_l2218_221870

variable {x y z a b c : ℝ}

-- The function statement where we want to show the equivalence.
theorem reciprocal_sum (h1 : x ≠ y) (h2 : x ≠ z) (h3 : y ≠ z)
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (hxy : (x * y) / (x - y) = a)
  (hxz : (x * z) / (x - z) = b)
  (hyz : (y * z) / (y - z) = c) :
  (1/x + 1/y + 1/z) = ((1/a + 1/b + 1/c) / 2) :=
sorry

end reciprocal_sum_l2218_221870


namespace compressor_distances_distances_when_a_15_l2218_221886

theorem compressor_distances (a : ℝ) (x y z : ℝ) (h1 : x + y = 2 * z) (h2 : x + z = y + a) (h3 : x + z = 75) :
  0 < a ∧ a < 100 → 
  let x := (75 + a) / 3;
  let y := 75 - a;
  let z := 75 - x;
  x + y = 2 * z ∧ x + z = y + a ∧ x + z = 75 :=
sorry

theorem distances_when_a_15 (x y z : ℝ) (h : 15 = 15) :
  let x := (75 + 15) / 3;
  let y := 75 - 15;
  let z := 75 - x;
  x = 30 ∧ y = 60 ∧ z = 45 :=
sorry

end compressor_distances_distances_when_a_15_l2218_221886


namespace max_t_for_real_root_l2218_221844

theorem max_t_for_real_root (t : ℝ) (x : ℝ) 
  (h : 0 < x ∧ x < π ∧ (t+1) * Real.cos x - t * Real.sin x = t + 2) : t = -1 :=
sorry

end max_t_for_real_root_l2218_221844


namespace possible_values_of_a_l2218_221826

theorem possible_values_of_a (a : ℝ) :
  (∃ x, ∀ y, (y = x) ↔ (a * y^2 + 2 * y + a = 0))
  → (a = 0 ∨ a = 1 ∨ a = -1) :=
by
  sorry

end possible_values_of_a_l2218_221826


namespace evaluate_expression_l2218_221839

def a := 3 + 6 + 9
def b := 2 + 5 + 8
def c := 3 + 6 + 9
def d := 2 + 5 + 8

theorem evaluate_expression : (a / b) - (d / c) = 11 / 30 :=
by
  sorry

end evaluate_expression_l2218_221839


namespace root_expression_value_l2218_221881

noncomputable def value_of_expression (p q r : ℝ) (h1 : p + q + r = 8) (h2 : pq + pr + qr = 10) (h3 : pqr = 3) : ℝ :=
  sorry

theorem root_expression_value (p q r : ℝ) (h1 : p + q + r = 8) (h2 : pq + pr + qr = 10) (h3 : pqr = 3) :
  value_of_expression p q r h1 h2 h3 = 367 / 183 :=
sorry

end root_expression_value_l2218_221881


namespace sequence_property_l2218_221803

theorem sequence_property (a : ℕ → ℕ) (h1 : a 1 = 1)
  (h_rec : ∀ m n : ℕ, a (m + n) = a m + a n + m * n) :
  a 10 = 55 :=
sorry

end sequence_property_l2218_221803


namespace find_k_l2218_221830

noncomputable def k_val : ℝ := 19.2

theorem find_k (k : ℝ) :
  (4 + ∑' n : ℕ, (4 + n * k) / (5^(n + 1))) = 10 ↔ k = k_val :=
  sorry

end find_k_l2218_221830


namespace damage_in_usd_correct_l2218_221808

def exchange_rate := (125 : ℚ) / 100
def damage_CAD := 45000000
def damage_USD := damage_CAD / exchange_rate

theorem damage_in_usd_correct (CAD_to_USD : exchange_rate = (125 : ℚ) / 100) (damage_in_cad : damage_CAD = 45000000) : 
  damage_USD = 36000000 :=
by
  sorry

end damage_in_usd_correct_l2218_221808


namespace difference_between_picked_and_left_is_five_l2218_221880

theorem difference_between_picked_and_left_is_five :
  let dave_sticks := 14
  let amy_sticks := 9
  let ben_sticks := 12
  let total_initial_sticks := 65
  let total_picked_up := dave_sticks + amy_sticks + ben_sticks
  let sticks_left := total_initial_sticks - total_picked_up
  total_picked_up - sticks_left = 5 :=
by
  sorry

end difference_between_picked_and_left_is_five_l2218_221880


namespace no_intersection_l2218_221895

def f₁ (x : ℝ) : ℝ := abs (3 * x + 6)
def f₂ (x : ℝ) : ℝ := -abs (4 * x - 1)

theorem no_intersection : ∀ x, f₁ x ≠ f₂ x :=
by
  sorry

end no_intersection_l2218_221895


namespace solve_for_x_l2218_221801

theorem solve_for_x : ∃ x : ℝ, (6 * x) / 1.5 = 3.8 ∧ x = 0.95 := by
  use 0.95
  exact ⟨by norm_num, by norm_num⟩

end solve_for_x_l2218_221801


namespace am_gm_inequality_l2218_221872

noncomputable def arithmetic_mean (a c : ℝ) : ℝ := (a + c) / 2

noncomputable def geometric_mean (a c : ℝ) : ℝ := Real.sqrt (a * c)

theorem am_gm_inequality (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a < b) (h5 : b < c) :
  (arithmetic_mean a c - geometric_mean a c < (c - a)^2 / (8 * a)) :=
sorry

end am_gm_inequality_l2218_221872


namespace points_on_circle_l2218_221824

theorem points_on_circle (t : ℝ) :
  let x := (t^3 - 1) / (t^3 + 1);
  let y := (2 * t^3) / (t^3 + 1);
  x^2 + y^2 = 1 :=
by
  let x := (t^3 - 1) / (t^3 + 1)
  let y := (2 * t^3) / (t^3 + 1)
  have h1 : x^2 + y^2 = ((t^3 - 1) / (t^3 + 1))^2 + ((2 * t^3) / (t^3 + 1))^2 := by rfl
  have h2 : (x^2 + y^2) = ( (t^3 - 1)^2 + (2 * t^3)^2 ) / (t^3 + 1)^2 := by sorry
  have h3 : (x^2 + y^2) = ( t^6 - 2 * t^3 + 1 + 4 * t^6 ) / (t^3 + 1)^2 := by sorry
  have h4 : (x^2 + y^2) = 1 := by sorry
  exact h4

end points_on_circle_l2218_221824


namespace green_pill_cost_l2218_221809

-- Definitions for the problem conditions
def number_of_days : ℕ := 21
def total_cost : ℚ := 819
def daily_cost : ℚ := total_cost / number_of_days
def cost_green_pill (x : ℚ) : ℚ := x
def cost_pink_pill (x : ℚ) : ℚ := x - 1
def total_daily_pill_cost (x : ℚ) : ℚ := cost_green_pill x + 2 * cost_pink_pill x

-- Theorem to be proven
theorem green_pill_cost : ∃ x : ℚ, total_daily_pill_cost x = daily_cost ∧ x = 41 / 3 :=
sorry

end green_pill_cost_l2218_221809


namespace arithmetic_problem_l2218_221817

theorem arithmetic_problem : 90 + 5 * 12 / (180 / 3) = 91 := by
  sorry

end arithmetic_problem_l2218_221817


namespace largest_n_for_factorable_polynomial_l2218_221892

theorem largest_n_for_factorable_polynomial : ∃ n, 
  (∀ A B : ℤ, (6 * B + A = n) → (A * B = 144)) ∧ 
  (∀ n', (∀ A B : ℤ, (6 * B + A = n') → (A * B = 144)) → n' ≤ n) ∧ 
  (n = 865) :=
by
  sorry

end largest_n_for_factorable_polynomial_l2218_221892


namespace Carol_weight_equals_nine_l2218_221887

-- conditions in Lean definitions
def Mildred_weight : ℤ := 59
def weight_difference : ℤ := 50

-- problem statement to prove in Lean 4
theorem Carol_weight_equals_nine (Carol_weight : ℤ) :
  Mildred_weight = Carol_weight + weight_difference → Carol_weight = 9 :=
by
  sorry

end Carol_weight_equals_nine_l2218_221887


namespace min_fraction_sum_is_15_l2218_221835

theorem min_fraction_sum_is_15
  (A B C D : ℕ)
  (h_distinct : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D)
  (h_digits : A < 10 ∧ B < 10 ∧ C < 10 ∧ D < 10)
  (h_nonzero_int : ∃ k : ℤ, k ≠ 0 ∧ (A + B : ℤ) = k * (C + D))
  : C + D = 15 :=
sorry

end min_fraction_sum_is_15_l2218_221835


namespace polynomial_factor_pair_l2218_221858

theorem polynomial_factor_pair (a b : ℝ) :
  (∃ (c d : ℝ), 3 * x^4 + a * x^3 + 48 * x^2 + b * x + 12 = (2 * x^2 - 3 * x + 2) * (c * x^2 + d * x + 6)) →
  (a, b) = (-26.5, -40) :=
by
  sorry

end polynomial_factor_pair_l2218_221858


namespace largest_integer_y_l2218_221828

theorem largest_integer_y (y : ℤ) : 
  (∃ k : ℤ, (y^2 + 3*y + 10) = k * (y - 4)) → y ≤ 42 :=
sorry

end largest_integer_y_l2218_221828


namespace smallest_part_is_correct_l2218_221846

-- Conditions
def total_value : ℕ := 360
def proportion1 : ℕ := 5
def proportion2 : ℕ := 7
def proportion3 : ℕ := 4
def proportion4 : ℕ := 8
def total_parts := proportion1 + proportion2 + proportion3 + proportion4
def value_per_part := total_value / total_parts
def smallest_proportion : ℕ := proportion3

-- Theorem to prove
theorem smallest_part_is_correct : value_per_part * smallest_proportion = 60 := by
  dsimp [total_value, total_parts, value_per_part, smallest_proportion]
  norm_num
  sorry

end smallest_part_is_correct_l2218_221846


namespace negate_original_is_correct_l2218_221889

-- Define the original proposition
def original_proposition (a b : ℕ) : Prop := (a * b = 0) → (a = 0 ∨ b = 0)

-- Define the negated proposition
def negated_proposition (a b : ℕ) : Prop := (a * b ≠ 0) → (a ≠ 0 ∧ b ≠ 0)

-- The theorem stating that the negation of the original proposition is the given negated proposition
theorem negate_original_is_correct (a b : ℕ) : ¬ original_proposition a b ↔ negated_proposition a b := by
  sorry

end negate_original_is_correct_l2218_221889


namespace division_remainder_correct_l2218_221841

def polynomial_div_remainder (x : ℝ) : ℝ :=
  3 * x^4 + 14 * x^3 - 50 * x^2 - 72 * x + 55

def divisor (x : ℝ) : ℝ :=
  x^2 + 8 * x - 4

theorem division_remainder_correct :
  ∀ x : ℝ, polynomial_div_remainder x % divisor x = 224 * x - 113 :=
by
  sorry

end division_remainder_correct_l2218_221841


namespace isosceles_triangle_area_l2218_221807

theorem isosceles_triangle_area (a b c : ℝ) (h: a = 5 ∧ b = 5 ∧ c = 6)
  (altitude_splits_base : ∀ (h : 3^2 + x^2 = 25), x = 4) : 
  ∃ (area : ℝ), area = 12 := 
by
  sorry

end isosceles_triangle_area_l2218_221807


namespace distinct_numbers_in_union_set_l2218_221843

def first_seq_term (k : ℕ) : ℤ := 5 * ↑k - 3
def second_seq_term (m : ℕ) : ℤ := 9 * ↑m - 3

def first_seq_set : Finset ℤ := ((Finset.range 1003).image first_seq_term)
def second_seq_set : Finset ℤ := ((Finset.range 1003).image second_seq_term)

def union_set : Finset ℤ := first_seq_set ∪ second_seq_set

theorem distinct_numbers_in_union_set : union_set.card = 1895 := by
  sorry

end distinct_numbers_in_union_set_l2218_221843


namespace square_areas_l2218_221834

theorem square_areas (s1 s2 s3 : ℕ)
  (h1 : s3 = s2 + 1)
  (h2 : s3 = s1 + 2)
  (h3 : s2 = 18)
  (h4 : s1 = s2 - 1) :
  s3^2 = 361 ∧ s2^2 = 324 ∧ s1^2 = 289 :=
by {
sorry
}

end square_areas_l2218_221834


namespace order_of_products_l2218_221868

theorem order_of_products (x a b : ℝ) (h1 : x < a) (h2 : a < b) (h3 : b < 0) : b * x > a * x ∧ a * x > a ^ 2 :=
by
  sorry

end order_of_products_l2218_221868


namespace all_perfect_squares_l2218_221836

theorem all_perfect_squares (a b c : ℕ) (h_gcd : Nat.gcd (Nat.gcd a b) c = 1) 
  (h_eq : a ^ 2 + b ^ 2 + c ^ 2 = 2 * (a * b + b * c + c * a)) : 
  ∃ (k l m : ℕ), a = k ^ 2 ∧ b = l ^ 2 ∧ c = m ^ 2 :=
sorry

end all_perfect_squares_l2218_221836


namespace larger_number_is_22_l2218_221890

theorem larger_number_is_22 (x y : ℕ) (h1 : y = x + 10) (h2 : x + y = 34) : y = 22 :=
by
  sorry

end larger_number_is_22_l2218_221890


namespace angles_of_triangle_arith_seq_l2218_221875

theorem angles_of_triangle_arith_seq (A B C a b c : ℝ) (h1 : A + B + C = 180) (h2 : A = B - (B - C)) (h3 : (1 / a + 1 / c) / 2 = 1 / b) : 
  A = 60 ∧ B = 60 ∧ C = 60 :=
sorry

end angles_of_triangle_arith_seq_l2218_221875


namespace topping_cost_l2218_221811

noncomputable def cost_of_topping (ic_cost sundae_cost number_of_toppings: ℝ) : ℝ :=
(sundae_cost - ic_cost) / number_of_toppings

theorem topping_cost
  (ic_cost : ℝ)
  (sundae_cost : ℝ)
  (number_of_toppings : ℕ)
  (h_ic_cost : ic_cost = 2)
  (h_sundae_cost : sundae_cost = 7)
  (h_number_of_toppings : number_of_toppings = 10) :
  cost_of_topping ic_cost sundae_cost number_of_toppings = 0.5 :=
  by
  -- Proof will be here
  sorry

end topping_cost_l2218_221811


namespace value_of_f_neg1_l2218_221802

def f (x : ℤ) : ℤ := x^2 - 2 * x

theorem value_of_f_neg1 : f (-1) = 3 := by
  sorry

end value_of_f_neg1_l2218_221802


namespace chickens_count_l2218_221852

-- Define conditions
def cows : Nat := 4
def sheep : Nat := 3
def bushels_per_cow : Nat := 2
def bushels_per_sheep : Nat := 2
def bushels_per_chicken : Nat := 3
def total_bushels_needed : Nat := 35

-- The main theorem to be proven
theorem chickens_count : 
  (total_bushels_needed - ((cows * bushels_per_cow) + (sheep * bushels_per_sheep))) / bushels_per_chicken = 7 :=
by
  sorry

end chickens_count_l2218_221852


namespace find_b_l2218_221865

noncomputable def f (x : ℝ) : ℝ := (x+1)^3 + (x / (x + 1))

theorem find_b (b : ℝ) (h_sum : ∃ x1 x2 : ℝ, f x1 = -x1 + b ∧ f x2 = -x2 + b ∧ x1 + x2 = -2) : b = 0 :=
by
  sorry

end find_b_l2218_221865


namespace part_a_part_b_l2218_221859

theorem part_a (n : ℕ) (hn : n % 2 = 1) (h_pos : n > 0) :
  ∃ k : ℕ, 1 ≤ k ∧ k ≤ n-1 ∧ ∃ f : (ℕ → ℕ), f k ≥ (n - 1) / 2 :=
sorry

theorem part_b : ∃ᶠ n in at_top, ∃ f : (ℕ → ℕ), ∀ k : ℕ, 1 ≤ k ∧ k ≤ n-1 → f k ≤ (n - 1) / 2 :=
sorry

end part_a_part_b_l2218_221859


namespace pond_water_after_45_days_l2218_221805

theorem pond_water_after_45_days :
  let initial_amount := 300
  let daily_evaporation := 1
  let rain_every_third_day := 2
  let total_days := 45
  let non_third_days := total_days - (total_days / 3)
  let third_days := total_days / 3
  let total_net_change := (non_third_days * (-daily_evaporation)) + (third_days * (rain_every_third_day - daily_evaporation))
  let final_amount := initial_amount + total_net_change
  final_amount = 285 :=
by
  sorry

end pond_water_after_45_days_l2218_221805


namespace smallest_root_equation_l2218_221861

theorem smallest_root_equation :
  ∃ x : ℝ, (3 * x) / (x - 2) + (2 * x^2 - 28) / x = 11 ∧ ∀ y, (3 * y) / (y - 2) + (2 * y^2 - 28) / y = 11 → x ≤ y ∧ x = (-1 - Real.sqrt 17) / 2 :=
sorry

end smallest_root_equation_l2218_221861


namespace find_c_minus_a_l2218_221800

theorem find_c_minus_a (a b c : ℝ) (h1 : (a + b) / 2 = 45) (h2 : (b + c) / 2 = 50) : c - a = 10 :=
sorry

end find_c_minus_a_l2218_221800


namespace correct_average_marks_l2218_221813

def incorrect_average := 100
def number_of_students := 10
def incorrect_mark := 60
def correct_mark := 10
def difference := incorrect_mark - correct_mark
def incorrect_total := incorrect_average * number_of_students
def correct_total := incorrect_total - difference

theorem correct_average_marks : correct_total / number_of_students = 95 := by
  sorry

end correct_average_marks_l2218_221813


namespace point_C_coordinates_line_MN_equation_area_triangle_ABC_l2218_221825

-- Define the points A and B
def A : ℝ × ℝ := (5, -2)
def B : ℝ × ℝ := (7, 3)

-- Let C be an unknown point that we need to determine
variables (x y : ℝ)

-- Define the conditions given in the problem
axiom midpoint_M : (x + 5) / 2 = 0 ∧ (y + 3) / 2 = 0 -- Midpoint M lies on the y-axis
axiom midpoint_N : (x + 7) / 2 = 1 ∧ (y + 3) / 2 = 0 -- Midpoint N lies on the x-axis

-- The problem consists of proving three assertions
theorem point_C_coordinates :
  ∃ (x y : ℝ), (x, y) = (-5, -3) :=
by
  sorry

theorem line_MN_equation :
  ∃ (a b c : ℝ), a = 5 ∧ b = -2 ∧ c = -5 :=
by
  sorry

theorem area_triangle_ABC :
  ∃ (S : ℝ), S = 841 / 20 :=
by
  sorry

end point_C_coordinates_line_MN_equation_area_triangle_ABC_l2218_221825


namespace larger_cylinder_volume_l2218_221860

theorem larger_cylinder_volume (v: ℝ) (r: ℝ) (R: ℝ) (h: ℝ) (hR : R = 2 * r) (hv : v = 100) : 
  π * R^2 * h = 4 * v := 
by 
  sorry

end larger_cylinder_volume_l2218_221860


namespace total_legs_and_hands_on_ground_is_118_l2218_221882

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

end total_legs_and_hands_on_ground_is_118_l2218_221882


namespace points_lost_calculation_l2218_221855

variable (firstRound secondRound finalScore : ℕ)
variable (pointsLost : ℕ)

theorem points_lost_calculation 
  (h1 : firstRound = 40) 
  (h2 : secondRound = 50) 
  (h3 : finalScore = 86) 
  (h4 : pointsLost = firstRound + secondRound - finalScore) :
  pointsLost = 4 := 
sorry

end points_lost_calculation_l2218_221855


namespace well_diameter_l2218_221832

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

end well_diameter_l2218_221832


namespace range_of_x_l2218_221897

noncomputable def f (x : ℝ) : ℝ :=
  Real.log (Real.exp x + Real.exp (-x)) + x^2

theorem range_of_x (x : ℝ) : 
  (∃ y z : ℝ, y = 2 * x - 1 ∧ f x > f y ∧ x > 1 / 3 ∧ x < 1) :=
sorry

end range_of_x_l2218_221897


namespace evaluate_expression_l2218_221857

theorem evaluate_expression : 4 * 5 - 3 + 2^3 - 3 * 2 = 19 := by
  sorry

end evaluate_expression_l2218_221857


namespace find_m_ineq_soln_set_min_value_a2_b2_l2218_221869

-- Problem 1
theorem find_m_ineq_soln_set (m x : ℝ) (h1 : m - |x - 2| ≥ 1) (h2 : x ∈ Set.Icc 0 4) : m = 3 := by
  sorry

-- Problem 2
theorem min_value_a2_b2 (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 3) : a^2 + b^2 ≥ 9 / 2 := by
  sorry

end find_m_ineq_soln_set_min_value_a2_b2_l2218_221869


namespace initial_balance_l2218_221815

theorem initial_balance (X : ℝ) : 
  (X - 60 - 30 - 0.25 * (X - 60 - 30) - 10 = 100) ↔ (X = 236.67) := 
  by
    sorry

end initial_balance_l2218_221815
