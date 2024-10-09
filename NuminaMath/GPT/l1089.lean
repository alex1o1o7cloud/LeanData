import Mathlib

namespace heather_initial_oranges_l1089_108986

theorem heather_initial_oranges (given_oranges: ℝ) (total_oranges: ℝ) (initial_oranges: ℝ) 
    (h1: given_oranges = 35.0) 
    (h2: total_oranges = 95) : 
    initial_oranges = 60 :=
by
  sorry

end heather_initial_oranges_l1089_108986


namespace factorize_expression_l1089_108918

theorem factorize_expression (a : ℚ) : 2 * a^2 - 4 * a = 2 * a * (a - 2) := by
  sorry

end factorize_expression_l1089_108918


namespace function_behaviour_l1089_108915

theorem function_behaviour (a : ℝ) (h : a ≠ 0) :
  ¬ ((a * (-2)^2 + 2 * a * (-2) + 1 > a * (-1)^2 + 2 * a * (-1) + 1) ∧
     (a * (-1)^2 + 2 * a * (-1) + 1 > a * 0^2 + 2 * a * 0 + 1)) :=
by
  sorry

end function_behaviour_l1089_108915


namespace bacteria_reach_target_l1089_108929

def bacteria_growth (initial : ℕ) (target : ℕ) (doubling_time : ℕ) (delay : ℕ) : ℕ :=
  let doubling_count := Nat.log2 (target / initial)
  doubling_count * doubling_time + delay

theorem bacteria_reach_target : 
  bacteria_growth 800 25600 5 3 = 28 := by
  sorry

end bacteria_reach_target_l1089_108929


namespace negation_of_universal_prop_l1089_108998

theorem negation_of_universal_prop :
  (¬ (∀ x : ℝ, x^2 - 5 * x + 3 ≤ 0)) ↔ (∃ x : ℝ, x^2 - 5 * x + 3 > 0) :=
by sorry

end negation_of_universal_prop_l1089_108998


namespace polygon_sides_l1089_108967

/-- If the sum of the interior angles of a polygon is three times the sum of its exterior angles,
    then the number of sides of the polygon is 8. -/
theorem polygon_sides (n : ℕ) (h1 : 180 * (n - 2) = 3 * 360) : n = 8 :=
sorry

end polygon_sides_l1089_108967


namespace parallelogram_base_length_l1089_108992

theorem parallelogram_base_length (b h : ℝ) (area : ℝ) (angle : ℝ) (h_area : area = 200) 
(h_altitude : h = 2 * b) (h_angle : angle = 60) : b = 10 :=
by
  -- Placeholder for proof
  sorry

end parallelogram_base_length_l1089_108992


namespace find_c_l1089_108966

theorem find_c (a b c : ℕ) (hpos_a : 0 < a) (hpos_b : 0 < b) (hpos_c : 0 < c)
(h_asc : a < b) (h_asc2 : b < c)
(h_sum : a + b + c = 11)
(h_eq : 1 / a + 1 / b + 1 / c = 1) : c = 6 := 
sorry

end find_c_l1089_108966


namespace x_midpoint_of_MN_l1089_108976

-- Definition: Given the parabola y^2 = 4x
def parabola (y x : ℝ) : Prop := y^2 = 4 * x

-- Definition: Point F is the focus of the parabola y^2 = 4x
def focus (F : ℝ × ℝ) : Prop := F = (1, 0)

-- Definition: Points M and N are on the parabola
def on_parabola (M N : ℝ × ℝ) : Prop :=
  parabola M.2 M.1 ∧ parabola N.2 N.1

-- Definition: The sum of distances |MF| + |NF| = 6
def sum_of_distances (M N : ℝ × ℝ) (F : ℝ × ℝ) : Prop :=
  dist M F + dist N F = 6

-- Theorem: Prove that the x-coordinate of the midpoint of MN is 2
theorem x_midpoint_of_MN (M N : ℝ × ℝ) (F : ℝ × ℝ) 
  (hF : focus F) (hM_N : on_parabola M N) (hDist : sum_of_distances M N F) :
  (M.1 + N.1) / 2 = 2 :=
sorry

end x_midpoint_of_MN_l1089_108976


namespace jack_sugar_remaining_l1089_108972

-- Define the initial amount of sugar and all daily transactions
def jack_initial_sugar : ℝ := 65
def jack_use_day1 : ℝ := 18.5
def alex_borrow_day1 : ℝ := 5.3
def jack_buy_day2 : ℝ := 30.2
def jack_use_day2 : ℝ := 12.7
def emma_give_day2 : ℝ := 4.75
def jack_buy_day3 : ℝ := 20.5
def jack_use_day3 : ℝ := 8.25
def alex_return_day3 : ℝ := 2.8
def alex_borrow_day3 : ℝ := 1.2
def jack_use_day4 : ℝ := 9.5
def olivia_give_day4 : ℝ := 6.35
def jack_use_day5 : ℝ := 10.75
def emma_borrow_day5 : ℝ := 3.1
def alex_return_day5 : ℝ := 3

-- Calculate the remaining sugar each day
def jack_sugar_day1 : ℝ := jack_initial_sugar - jack_use_day1 - alex_borrow_day1
def jack_sugar_day2 : ℝ := jack_sugar_day1 + jack_buy_day2 - jack_use_day2 + emma_give_day2
def jack_sugar_day3 : ℝ := jack_sugar_day2 + jack_buy_day3 - jack_use_day3 + alex_return_day3 - alex_borrow_day3
def jack_sugar_day4 : ℝ := jack_sugar_day3 - jack_use_day4 + olivia_give_day4
def jack_sugar_day5 : ℝ := jack_sugar_day4 - jack_use_day5 - emma_borrow_day5 + alex_return_day5

-- Final proof statement: Jack ends up with 63.3 pounds of sugar
theorem jack_sugar_remaining : jack_sugar_day5 = 63.3 := 
by sorry

end jack_sugar_remaining_l1089_108972


namespace full_time_worked_year_l1089_108930

-- Define the conditions as constants
def total_employees : ℕ := 130
def full_time : ℕ := 80
def worked_year : ℕ := 100
def neither : ℕ := 20

-- Define the question as a theorem stating the correct answer
theorem full_time_worked_year : full_time + worked_year - total_employees + neither = 70 :=
by
  sorry

end full_time_worked_year_l1089_108930


namespace find_a_l1089_108944

noncomputable def f (a x : ℝ) : ℝ :=
if x < 1 then 2 * x + a else -x - 2 * a

theorem find_a (a : ℝ) (h : a ≠ 0) (h_eq : f a (1 - a) = f a (1 + a)) : a = -3 / 4 :=
sorry

end find_a_l1089_108944


namespace cube_volume_surface_area_value_l1089_108953

theorem cube_volume_surface_area_value (x : ℝ) : 
  (∃ s : ℝ, s = (6 * x)^(1 / 3) ∧ 6 * s^2 = 2 * x) → 
  x = 1 / 972 :=
by {
  sorry
}

end cube_volume_surface_area_value_l1089_108953


namespace outlet_pipe_rate_l1089_108909

theorem outlet_pipe_rate (V_ft : ℝ) (cf : ℝ) (V_in : ℝ) (r_in : ℝ) (r_out1 : ℝ) (t : ℝ) (r_out2 : ℝ) :
    V_ft = 30 ∧ cf = 1728 ∧
    V_in = V_ft * cf ∧
    r_in = 5 ∧ r_out1 = 9 ∧ t = 4320 ∧
    V_in = (r_out1 + r_out2 - r_in) * t →
    r_out2 = 8 := by
  intros h
  sorry

end outlet_pipe_rate_l1089_108909


namespace max_element_sum_l1089_108975

-- Definitions based on conditions
def S : Set ℚ :=
  {r | ∃ (p q : ℕ), r = p / q ∧ q ≤ 2009 ∧ p / q < 1257/2009}

-- Maximum element of S in reduced form
def max_element_S (r : ℚ) : Prop := r ∈ S ∧ ∀ s ∈ S, r ≥ s

-- Main statement to be proven
theorem max_element_sum : 
  ∃ p0 q0 : ℕ, max_element_S (p0 / q0) ∧ Nat.gcd p0 q0 = 1 ∧ p0 + q0 = 595 := 
sorry

end max_element_sum_l1089_108975


namespace increase_in_p_does_not_imply_increase_in_equal_points_probability_l1089_108974

noncomputable def probability_equal_points (p : ℝ) : ℝ :=
  (3 * p^2 - 2 * p + 1) / 4

theorem increase_in_p_does_not_imply_increase_in_equal_points_probability :
  ¬ ∀ p1 p2 : ℝ, p1 < p2 → p1 ≥ 0 → p2 ≤ 1 → probability_equal_points p1 < probability_equal_points p2 := 
sorry

end increase_in_p_does_not_imply_increase_in_equal_points_probability_l1089_108974


namespace train_cross_bridge_time_l1089_108985

noncomputable def time_to_cross_bridge (L_train : ℕ) (v_kmph : ℕ) (L_bridge : ℕ) : ℝ :=
  let v_mps := (v_kmph * 1000) / 3600
  let total_distance := L_train + L_bridge
  total_distance / v_mps

theorem train_cross_bridge_time :
  time_to_cross_bridge 145 54 660 = 53.67 := by
    sorry

end train_cross_bridge_time_l1089_108985


namespace original_prices_correct_l1089_108943

-- Define the problem conditions
def Shirt_A_discount1 := 0.10
def Shirt_A_discount2 := 0.20
def Shirt_A_final_price := 420

def Shirt_B_discount1 := 0.15
def Shirt_B_discount2 := 0.25
def Shirt_B_final_price := 405

def Shirt_C_discount1 := 0.05
def Shirt_C_discount2 := 0.15
def Shirt_C_final_price := 680

def sales_tax := 0.05

-- Define the original prices for each shirt.
def original_price_A := 420 / (0.9 * 0.8)
def original_price_B := 405 / (0.85 * 0.75)
def original_price_C := 680 / (0.95 * 0.85)

-- Prove the original prices of the shirts
theorem original_prices_correct:
  original_price_A = 583.33 ∧ 
  original_price_B = 635 ∧ 
  original_price_C = 842.24 := 
by
  sorry

end original_prices_correct_l1089_108943


namespace num_ways_to_make_change_l1089_108942

-- Define the standard U.S. coins
def penny : ℕ := 1
def nickel : ℕ := 5
def dime : ℕ := 10
def quarter : ℕ := 25

-- Define the total amount
def total_amount : ℕ := 50

-- Condition to exclude two quarters
def valid_combination (num_pennies num_nickels num_dimes num_quarters : ℕ) : Prop :=
  (num_quarters != 2) ∧ (num_pennies + 5 * num_nickels + 10 * num_dimes + 25 * num_quarters = total_amount)

-- Prove that there are 39 ways to make change for 50 cents
theorem num_ways_to_make_change : 
  ∃ count : ℕ, count = 39 ∧ (∀ 
    (num_pennies num_nickels num_dimes num_quarters : ℕ),
    valid_combination num_pennies num_nickels num_dimes num_quarters → 
    (num_pennies, num_nickels, num_dimes, num_quarters) = count) :=
sorry

end num_ways_to_make_change_l1089_108942


namespace x_squared_y_cubed_plus_y_squared_x_cubed_eq_zero_l1089_108993

theorem x_squared_y_cubed_plus_y_squared_x_cubed_eq_zero
  (x y : ℝ)
  (h1 : x + y = 2)
  (h2 : x * y = 4) : x^2 * y^3 + y^2 * x^3 = 0 := 
sorry

end x_squared_y_cubed_plus_y_squared_x_cubed_eq_zero_l1089_108993


namespace quadratic_roots_l1089_108928

theorem quadratic_roots {a : ℝ} :
  (4 < a ∧ a < 6) ∨ (a > 12) → 
  (∃ x1 x2 : ℝ, x1 = a + Real.sqrt (18 * (a - 4)) ∧ x2 = a - Real.sqrt (18 * (a - 4)) ∧ x1 > 0 ∧ x2 > 0) :=
by sorry

end quadratic_roots_l1089_108928


namespace triangle_side_AC_l1089_108958

theorem triangle_side_AC (B : Real) (BC AB : Real) (AC : Real) (hB : B = 30 * Real.pi / 180) (hBC : BC = 2) (hAB : AB = Real.sqrt 3) : AC = 1 :=
by
  sorry

end triangle_side_AC_l1089_108958


namespace triangle_area_202_2192_pi_squared_l1089_108962

noncomputable def triangle_area (a b c : ℝ) : ℝ := 
  let r := (a + b + c) / (2 * Real.pi)
  let theta := 20.0 * Real.pi / 180.0  -- converting 20 degrees to radians
  let angle1 := 5 * theta
  let angle2 := 6 * theta
  let angle3 := 7 * theta
  (1 / 2) * r * r * (Real.sin angle1 + Real.sin angle2 + Real.sin angle3)

theorem triangle_area_202_2192_pi_squared (a b c : ℝ) (h1 : a = 5) (h2 : b = 6) (h3 : c = 7) : 
  triangle_area a b c = 202.2192 / (Real.pi * Real.pi) := 
by {
  sorry
}

end triangle_area_202_2192_pi_squared_l1089_108962


namespace sum_of_distinct_products_of_6_23H_508_3G4_l1089_108919

theorem sum_of_distinct_products_of_6_23H_508_3G4 (G H : ℕ) : 
  (G < 10) → (H < 10) →
  (623 * 1000 + H * 100 + 508 * 10 + 3 * 10 + G * 1 + 4) % 72 = 0 →
  (if G = 0 then 0 + if G = 4 then 4 else 0 else 0) = 4 :=
by
  intros
  sorry

end sum_of_distinct_products_of_6_23H_508_3G4_l1089_108919


namespace refrigerator_cost_l1089_108984

theorem refrigerator_cost
  (R : ℝ)
  (mobile_phone_cost : ℝ := 8000)
  (loss_percent_refrigerator : ℝ := 0.04)
  (profit_percent_mobile_phone : ℝ := 0.09)
  (overall_profit : ℝ := 120)
  (selling_price_refrigerator : ℝ := 0.96 * R)
  (selling_price_mobile_phone : ℝ := 8720)
  (total_selling_price : ℝ := selling_price_refrigerator + selling_price_mobile_phone)
  (total_cost_price : ℝ := R + mobile_phone_cost)
  (balance_profit_eq : total_selling_price = total_cost_price + overall_profit):
  R = 15000 :=
by
  sorry

end refrigerator_cost_l1089_108984


namespace tennis_preference_combined_percentage_l1089_108983

theorem tennis_preference_combined_percentage :
  let total_north_students := 1500
  let total_south_students := 1800
  let north_tennis_percentage := 0.30
  let south_tennis_percentage := 0.35
  let north_tennis_students := total_north_students * north_tennis_percentage
  let south_tennis_students := total_south_students * south_tennis_percentage
  let total_tennis_students := north_tennis_students + south_tennis_students
  let total_students := total_north_students + total_south_students
  let combined_percentage := (total_tennis_students / total_students) * 100
  combined_percentage = 33 := 
by
  sorry

end tennis_preference_combined_percentage_l1089_108983


namespace problem_1_problem_2_l1089_108924

theorem problem_1 (a b c : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : c > 0) (h₃ : ∀ x, |x + a| + |x - b| + c ≥ 4) : 
  a + b + c = 4 :=
sorry

theorem problem_2 (a b c : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : c > 0) (h₃ : a + b + c = 4) : 
  (1/4) * a^2 + (1/9) * b^2 + c^2 = 8 / 7 :=
sorry

end problem_1_problem_2_l1089_108924


namespace mutually_exclusive_complementary_event_l1089_108913

-- Definitions of events
def hitting_target_at_least_once (shots: ℕ) : Prop := shots > 0
def not_hitting_target_at_all (shots: ℕ) : Prop := shots = 0

-- The statement to prove
theorem mutually_exclusive_complementary_event : 
  ∀ (shots: ℕ), (not_hitting_target_at_all shots ↔ ¬ hitting_target_at_least_once shots) :=
by 
  sorry

end mutually_exclusive_complementary_event_l1089_108913


namespace inverse_of_49_mod_89_l1089_108980

theorem inverse_of_49_mod_89 (h : (7 * 55 ≡ 1 [MOD 89])) : (49 * 1 ≡ 1 [MOD 89]) := 
by
  sorry

end inverse_of_49_mod_89_l1089_108980


namespace shorter_piece_length_correct_l1089_108900

noncomputable def shorter_piece_length (total_length : ℝ) (ratio : ℝ) : ℝ := 
  total_length * ratio / (ratio + 1)

theorem shorter_piece_length_correct :
  shorter_piece_length 57.134 (3.25678 / 7.81945) = 16.790 :=
by
  sorry

end shorter_piece_length_correct_l1089_108900


namespace books_bought_l1089_108952

theorem books_bought (initial_books bought_books total_books : ℕ) 
    (h_initial : initial_books = 35)
    (h_total : total_books = 56) :
    bought_books = total_books - initial_books → bought_books = 21 := 
by
  sorry

end books_bought_l1089_108952


namespace interest_rate_per_annum_l1089_108948

variable (P : ℝ := 1200) (T : ℝ := 1) (diff : ℝ := 2.999999999999936) (r : ℝ)
noncomputable def SI (P : ℝ) (r : ℝ) (T : ℝ) : ℝ := P * r * T
noncomputable def CI (P : ℝ) (r : ℝ) (T : ℝ) : ℝ := P * ((1 + r / 2) ^ (2 * T) - 1)

theorem interest_rate_per_annum :
  CI P r T - SI P r T = diff → r = 0.1 :=
by
  -- Proof to be provided
  sorry

end interest_rate_per_annum_l1089_108948


namespace max_non_overlapping_areas_l1089_108954

theorem max_non_overlapping_areas (n : ℕ) : 
  ∃ (max_areas : ℕ), max_areas = 3 * n := by
  sorry

end max_non_overlapping_areas_l1089_108954


namespace pattern_equation_l1089_108926

theorem pattern_equation (n : ℕ) : n^2 + n = n * (n + 1) := 
  sorry

end pattern_equation_l1089_108926


namespace trapezoid_area_l1089_108989

def isosceles_triangle (Δ : Type) (A B C : Δ) : Prop :=
  -- Define the property that triangle ABC is isosceles with AB = AC
  sorry

def similar_triangles (Δ₁ Δ₂ : Type) (A₁ B₁ C₁ : Δ₁) (A₂ B₂ C₂ : Δ₂) : Prop :=
  -- Define the property that triangles Δ₁ and Δ₂ are similar
  sorry

def area (Δ : Type) (A B C : Δ) : ℝ :=
  -- Define the area of a triangle Δ with vertices A, B, and C
  sorry

theorem trapezoid_area
  (Δ : Type)
  {A B C D E : Δ}
  (ABC_is_isosceles : isosceles_triangle Δ A B C)
  (all_similar : ∀ (Δ₁ Δ₂ : Type) (A₁ B₁ C₁ : Δ₁) (A₂ B₂ C₂ : Δ₂), 
    similar_triangles Δ₁ Δ₂ A₁ B₁ C₁ A₂ B₂ C₂ → (area Δ₁ A₁ B₁ C₁ = 1 → area Δ₂ A₂ B₂ C₂ = 1))
  (smallest_triangles_area : area Δ A B C = 50)
  (area_ADE : area Δ A D E = 5) :
  area Δ D B C + area Δ C E B = 45 := 
sorry

end trapezoid_area_l1089_108989


namespace bird_families_flew_away_l1089_108959

theorem bird_families_flew_away (original : ℕ) (left : ℕ) (flew_away : ℕ) (h1 : original = 67) (h2 : left = 35) (h3 : flew_away = original - left) : flew_away = 32 :=
by
  rw [h1, h2] at h3
  exact h3

end bird_families_flew_away_l1089_108959


namespace valid_values_for_D_l1089_108971

-- Definitions for the distinct digits and the non-zero condition
def is_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9
def distinct_nonzero_digits (A B C D : ℕ) : Prop :=
  is_digit A ∧ is_digit B ∧ is_digit C ∧ is_digit D ∧
  A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0 ∧ D ≠ 0 ∧
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D

-- Condition for the carry situation
def carry_in_addition (A B C D : ℕ) : Prop :=
  ∃ carry1 carry2 carry3 carry4 : ℕ,
  (A + B + carry1) % 10 = D ∧ (B + C + carry2) % 10 = A ∧
  (C + C + carry3) % 10 = B ∧ (A + B + carry4) % 10 = C ∧
  (carry1 = 1 ∨ carry2 = 1 ∨ carry3 = 1 ∨ carry4 = 1)

-- Main statement
theorem valid_values_for_D (A B C D : ℕ) :
  distinct_nonzero_digits A B C D →
  carry_in_addition A B C D →
  ∃ n, n = 5 :=
sorry

end valid_values_for_D_l1089_108971


namespace Bruce_initial_eggs_l1089_108957

variable (B : ℕ)

theorem Bruce_initial_eggs (h : B - 70 = 5) : B = 75 := by
  sorry

end Bruce_initial_eggs_l1089_108957


namespace emily_lemon_juice_fraction_l1089_108941

/-- 
Emily places 6 ounces of tea into a twelve-ounce cup and 6 ounces of honey into a second cup
of the same size. Then she adds 3 ounces of lemon juice to the second cup. Next, she pours half
the tea from the first cup into the second, mixes thoroughly, and then pours one-third of the
mixture in the second cup back into the first. 
Prove that the fraction of the mixture in the first cup that is lemon juice is 1/7.
--/
theorem emily_lemon_juice_fraction :
  let cup1_tea := 6
  let cup2_honey := 6
  let cup2_lemon_juice := 3
  let cup1_tea_transferred := cup1_tea / 2
  let cup1 := cup1_tea - cup1_tea_transferred
  let cup2 := cup2_honey + cup2_lemon_juice + cup1_tea_transferred
  let mix_ratio (x y : ℕ) := (x : ℚ) / (x + y)
  let cup1_after_transfer := cup1 + (cup2 / 3)
  let cup2_tea := cup1_tea_transferred
  let cup2_honey := cup2_honey
  let cup2_lemon_juice := cup2_lemon_juice
  let cup1_lemon_transferred := 1
  cup1_tea + (cup2 / 3) = 3 + (cup2_tea * (1 / 3)) + 1 + (cup2_honey * (1 / 3)) + cup2_lemon_juice / 3 →
  cup1 / (cup1 + cup2_honey) = 1/7 :=
sorry

end emily_lemon_juice_fraction_l1089_108941


namespace sequence_term_n_l1089_108950

theorem sequence_term_n (a : ℕ → ℕ) (a1 d : ℕ) (n : ℕ) (h1 : a 1 = a1) (h2 : d = 2)
  (h3 : a n = 19) (h_seq : ∀ n, a n = a1 + (n - 1) * d) : n = 10 :=
by
  sorry

end sequence_term_n_l1089_108950


namespace measles_cases_in_1990_l1089_108970

noncomputable def measles_cases_1970 := 480000
noncomputable def measles_cases_2000 := 600
noncomputable def years_between := 2000 - 1970
noncomputable def total_decrease := measles_cases_1970 - measles_cases_2000
noncomputable def decrease_per_year := total_decrease / years_between
noncomputable def years_from_1970_to_1990 := 1990 - 1970
noncomputable def decrease_to_1990 := years_from_1970_to_1990 * decrease_per_year
noncomputable def measles_cases_1990 := measles_cases_1970 - decrease_to_1990

theorem measles_cases_in_1990 : measles_cases_1990 = 160400 := by
  sorry

end measles_cases_in_1990_l1089_108970


namespace grazing_b_l1089_108937

theorem grazing_b (A_oxen_months B_oxen_months C_oxen_months total_months total_rent C_rent B_oxen : ℕ) 
  (hA : A_oxen_months = 10 * 7)
  (hB : B_oxen_months = B_oxen * 5)
  (hC : C_oxen_months = 15 * 3)
  (htotal : total_months = A_oxen_months + B_oxen_months + C_oxen_months)
  (hrent : total_rent = 175)
  (hC_rent : C_rent = 45)
  (hC_share : C_oxen_months / total_months = C_rent / total_rent) :
  B_oxen = 12 :=
by
  sorry

end grazing_b_l1089_108937


namespace average_people_moving_l1089_108932

theorem average_people_moving (days : ℕ) (total_people : ℕ) 
    (h_days : days = 5) (h_total_people : total_people = 3500) : 
    (total_people / days) = 700 :=
by
  sorry

end average_people_moving_l1089_108932


namespace cannot_determine_students_answered_both_correctly_l1089_108905

-- Definitions based on the given conditions
def students_enrolled : ℕ := 25
def students_answered_q1_correctly : ℕ := 22
def students_not_taken_test : ℕ := 3
def some_students_answered_q2_correctly : Prop := -- definition stating that there's an undefined number of students that answered question 2 correctly
  ∃ n : ℕ, (n ≤ students_enrolled) ∧ n > 0

-- Statement for the proof problem
theorem cannot_determine_students_answered_both_correctly :
  ∃ n, (n ≤ students_answered_q1_correctly) ∧ n > 0 → false :=
by sorry

end cannot_determine_students_answered_both_correctly_l1089_108905


namespace remainder_of_poly_division_l1089_108922

theorem remainder_of_poly_division :
  ∀ (x : ℝ), (x^2023 + x + 1) % (x^6 - x^4 + x^2 - 1) = x^7 + x + 1 :=
by
  sorry

end remainder_of_poly_division_l1089_108922


namespace find_y_l1089_108995

theorem find_y (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h1 : x = 2 + 1/y) (h2 : y = 2 + 1/x) : y = x :=
sorry

end find_y_l1089_108995


namespace total_tiles_l1089_108940

theorem total_tiles (n : ℕ) (h : 3 * n - 2 = 55) : n^2 = 361 :=
by
  sorry

end total_tiles_l1089_108940


namespace customers_in_each_car_l1089_108978

def total_customers (sports_store_sales music_store_sales : ℕ) : ℕ :=
  sports_store_sales + music_store_sales

def customers_per_car (total_customers cars : ℕ) : ℕ :=
  total_customers / cars

theorem customers_in_each_car :
  let cars := 10
  let sports_store_sales := 20
  let music_store_sales := 30
  let total_customers := total_customers sports_store_sales music_store_sales
  total_customers / cars = 5 := by
  let cars := 10
  let sports_store_sales := 20
  let music_store_sales := 30
  let total_customers := total_customers sports_store_sales music_store_sales
  show total_customers / cars = 5
  sorry

end customers_in_each_car_l1089_108978


namespace display_glasses_count_l1089_108921

noncomputable def tall_cupboards := 2
noncomputable def wide_cupboards := 2
noncomputable def narrow_cupboards := 2
noncomputable def shelves_per_narrow_cupboard := 3
noncomputable def glasses_tall_cupboard := 30
noncomputable def glasses_wide_cupboard := 2 * glasses_tall_cupboard
noncomputable def glasses_narrow_cupboard := 45
noncomputable def broken_shelf_glasses := glasses_narrow_cupboard / shelves_per_narrow_cupboard

theorem display_glasses_count :
  (tall_cupboards * glasses_tall_cupboard) +
  (wide_cupboards * glasses_wide_cupboard) +
  (1 * (broken_shelf_glasses * (shelves_per_narrow_cupboard - 1)) + glasses_narrow_cupboard) =
  255 :=
by sorry

end display_glasses_count_l1089_108921


namespace max_value_f_1_max_value_f_2_max_value_f_3_l1089_108933
noncomputable def f (x : ℝ) (m : ℝ) : ℝ := Real.log x - m * x

theorem max_value_f_1 (m : ℝ) (h : m ≤ 1 / Real.exp 1) :
  ∀ (x : ℝ), 1 ≤ x ∧ x ≤ Real.exp 1 → f x m ≤ 1 - m * Real.exp 1 :=
sorry

theorem max_value_f_2 (m : ℝ) (h1 : 1 / Real.exp 1 < m) (h2 : m < 1) :
  ∀ (x : ℝ), 1 ≤ x ∧ x ≤ Real.exp 1 → f x m ≤ -Real.log m - 1 :=
sorry

theorem max_value_f_3 (m : ℝ) (h : m ≥ 1) :
  ∀ (x : ℝ), 1 ≤ x ∧ x ≤ Real.exp 1 → f x m ≤ -m :=
sorry

end max_value_f_1_max_value_f_2_max_value_f_3_l1089_108933


namespace trig_expression_value_l1089_108927

theorem trig_expression_value (α : ℝ) (h : Real.tan α = 3) : 
  (1 + 2 * Real.sin α * Real.cos α) / (Real.sin α ^ 2 - Real.cos α ^ 2) = 2 :=
sorry

end trig_expression_value_l1089_108927


namespace convex_cyclic_quadrilaterals_perimeter_40_l1089_108938

theorem convex_cyclic_quadrilaterals_perimeter_40 :
  ∃ (n : ℕ), n = 750 ∧ ∀ (a b c d : ℕ), a + b + c + d = 40 → a ≥ b → b ≥ c → c ≥ d →
  (a < b + c + d) ∧ (b < a + c + d) ∧ (c < a + b + d) ∧ (d < a + b + c) :=
sorry

end convex_cyclic_quadrilaterals_perimeter_40_l1089_108938


namespace toothpicks_needed_for_8_step_staircase_l1089_108981

theorem toothpicks_needed_for_8_step_staircase:
  ∀ n toothpicks : ℕ, n = 4 → toothpicks = 30 → 
  (∃ additional_toothpicks : ℕ, additional_toothpicks = 88) :=
by
  sorry

end toothpicks_needed_for_8_step_staircase_l1089_108981


namespace general_formula_an_general_formula_bn_exists_arithmetic_sequence_bn_l1089_108955

variable (a_n : ℕ → ℝ)
variable (b_n : ℕ → ℝ)
variable (S_n : ℕ → ℝ)
variable (d : ℝ)

-- Define the initial conditions
axiom a2_a3_condition : a_n 2 * a_n 3 = 15
axiom S4_condition : S_n 4 = 16
axiom b_recursion : ∀ (n : ℕ), b_n (n + 1) - b_n n = 1 / (a_n n * a_n (n + 1))

-- Define the proofs
theorem general_formula_an : ∀ (n : ℕ), a_n n = 2 * n - 1 :=
sorry

theorem general_formula_bn : ∀ (n : ℕ), b_n n = (3 * n - 2) / (2 * n - 1) :=
sorry

theorem exists_arithmetic_sequence_bn : ∃ (m n : ℕ), m ≠ n ∧ b_n 2 + b_n n = 2 * b_n m ∧ b_n 2 = 4 / 3 ∧ (n = 8 ∧ m = 3) :=
sorry

end general_formula_an_general_formula_bn_exists_arithmetic_sequence_bn_l1089_108955


namespace polynomials_symmetric_l1089_108907

noncomputable def P : ℕ → (ℝ → ℝ → ℝ → ℝ)
  | 0       => λ x y z => 1
  | (m + 1) => λ x y z => (x + z) * (y + z) * (P m x y (z + 1)) - z^2 * (P m x y z)

theorem polynomials_symmetric (m : ℕ) (x y z : ℝ) : 
  P m x y z = P m y x z ∧ P m x y z = P m x z y := 
sorry

end polynomials_symmetric_l1089_108907


namespace total_distance_l1089_108906

def morning_distance : ℕ := 2
def evening_multiplier : ℕ := 5

theorem total_distance : morning_distance + (evening_multiplier * morning_distance) = 12 :=
by
  sorry

end total_distance_l1089_108906


namespace find_angle_y_l1089_108923

-- Definitions of the angles in the triangle
def angle_ACD : ℝ := 90
def angle_DEB : ℝ := 58

-- Theorem proving the value of angle DCE (denoted as y)
theorem find_angle_y (angle_sum_property : angle_ACD + y + angle_DEB = 180) : y = 32 :=
by sorry

end find_angle_y_l1089_108923


namespace ratio_black_bears_to_white_bears_l1089_108963

theorem ratio_black_bears_to_white_bears
  (B W Br : ℕ)
  (hB : B = 60)
  (hBr : Br = B + 40)
  (h_total : B + W + Br = 190) :
  B / W = 2 :=
by
  sorry

end ratio_black_bears_to_white_bears_l1089_108963


namespace exists_positive_integer_divisible_by_14_with_sqrt_between_25_and_25_3_l1089_108977

theorem exists_positive_integer_divisible_by_14_with_sqrt_between_25_and_25_3 :
  ∃ (x : ℕ), x % 14 = 0 ∧ 625 <= x ∧ x <= 640 ∧ x = 630 := 
by 
  sorry

end exists_positive_integer_divisible_by_14_with_sqrt_between_25_and_25_3_l1089_108977


namespace remainder_of_power_l1089_108999

theorem remainder_of_power :
  (4^215) % 9 = 7 := by
sorry

end remainder_of_power_l1089_108999


namespace sin_585_eq_neg_sqrt2_div_2_l1089_108910

theorem sin_585_eq_neg_sqrt2_div_2 : Real.sin (585 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by
  sorry

end sin_585_eq_neg_sqrt2_div_2_l1089_108910


namespace smallest_n_l1089_108935

def in_interval (x y z : ℝ) (n : ℕ) : Prop :=
  2 ≤ x ∧ x ≤ n ∧ 2 ≤ y ∧ y ≤ n ∧ 2 ≤ z ∧ z ≤ n

def no_two_within_one_unit (x y z : ℝ) : Prop :=
  abs (x - y) ≥ 1 ∧ abs (y - z) ≥ 1 ∧ abs (z - x) ≥ 1

def more_than_two_units_apart (x y z : ℝ) (n : ℕ) : Prop :=
  x > 2 ∧ x < n - 2 ∧ y > 2 ∧ y < n - 2 ∧ z > 2 ∧ z < n - 2

def probability_condition (n : ℕ) : Prop :=
  (n-4)^3 / (n-2)^3 > 1/3

theorem smallest_n (n : ℕ) : 11 = n → (∃ x y z : ℝ, in_interval x y z n ∧ no_two_within_one_unit x y z ∧ more_than_two_units_apart x y z n ∧ probability_condition n) :=
by
  sorry

end smallest_n_l1089_108935


namespace cookie_total_l1089_108960

-- Definitions of the conditions
def rows_large := 5
def rows_medium := 4
def rows_small := 6
def cookies_per_row_large := 6
def cookies_per_row_medium := 7
def cookies_per_row_small := 8
def number_of_trays := 4
def extra_row_large_first_tray := 1
def total_large_cookies := rows_large * cookies_per_row_large * number_of_trays + extra_row_large_first_tray * cookies_per_row_large
def total_medium_cookies := rows_medium * cookies_per_row_medium * number_of_trays
def total_small_cookies := rows_small * cookies_per_row_small * number_of_trays

-- Theorem to prove the total number of cookies is 430
theorem cookie_total : 
  total_large_cookies + total_medium_cookies + total_small_cookies = 430 :=
by
  -- Proof is omitted
  sorry

end cookie_total_l1089_108960


namespace cell_cycle_correct_statement_l1089_108908

theorem cell_cycle_correct_statement :
  ∃ (correct_statement : String), correct_statement = "In the cell cycle, chromatin DNA is easier to replicate than chromosome DNA" :=
by
  let A := "The separation of alleles occurs during the interphase of the cell cycle"
  let B := "In the cell cycle of plant cells, spindle fibers appear during the interphase"
  let C := "In the cell cycle, chromatin DNA is easier to replicate than chromosome DNA"
  let D := "In the cell cycle of liver cells, chromosomes exist for a longer time than chromatin"
  existsi C
  sorry

end cell_cycle_correct_statement_l1089_108908


namespace find_k_for_infinite_solutions_l1089_108988

noncomputable def has_infinitely_many_solutions (k : ℝ) : Prop :=
  ∀ x : ℝ, 5 * (3 * x - k) = 3 * (5 * x + 15)

theorem find_k_for_infinite_solutions :
  has_infinitely_many_solutions (-9) :=
by
  sorry

end find_k_for_infinite_solutions_l1089_108988


namespace real_solutions_count_l1089_108916

-- Define the system of equations
def sys_eqs (x y z w : ℝ) :=
  (x = z + w + z * w * x) ∧
  (z = x + y + x * y * z) ∧
  (y = w + x + w * x * y) ∧
  (w = y + z + y * z * w)

-- The statement of the proof problem
theorem real_solutions_count : ∃ S : Finset (ℝ × ℝ × ℝ × ℝ), (∀ t : ℝ × ℝ × ℝ × ℝ, t ∈ S ↔ sys_eqs t.1 t.2.1 t.2.2.1 t.2.2.2) ∧ S.card = 5 :=
by {
  sorry
}

end real_solutions_count_l1089_108916


namespace platform_length_l1089_108964

theorem platform_length (train_length : ℕ) (time_post : ℕ) (time_platform : ℕ) (speed : ℕ)
    (h1 : train_length = 150)
    (h2 : time_post = 15)
    (h3 : time_platform = 25)
    (h4 : speed = train_length / time_post)
    : (train_length + 100) / time_platform = speed :=
by
  sorry

end platform_length_l1089_108964


namespace g_two_gt_one_third_g_n_gt_one_third_l1089_108917

def seq_a (n : ℕ) : ℕ := 3 * n - 2
noncomputable def f (n : ℕ) : ℝ := (Finset.range n).sum (λ i => 1 / (seq_a (i + 1) : ℝ))
noncomputable def g (n : ℕ) : ℝ := f (n^2) - f (n - 1)

theorem g_two_gt_one_third : g 2 > 1 / 3 :=
sorry

theorem g_n_gt_one_third (n : ℕ) (h : n ≥ 3) : g n > 1 / 3 :=
sorry

end g_two_gt_one_third_g_n_gt_one_third_l1089_108917


namespace f_odd_and_increasing_l1089_108946

noncomputable def f (x : ℝ) : ℝ := 3^x - (1/3)^x

theorem f_odd_and_increasing : (∀ x : ℝ, f (-x) = -f x) ∧ (∀ x y : ℝ, x < y → f x < f y) :=
by
  sorry

end f_odd_and_increasing_l1089_108946


namespace find_f1_find_fx_find_largest_m_l1089_108951

noncomputable def f (a b c : ℝ) (x : ℝ) := a * x ^ 2 + b * x + c

axiom min_value_eq_zero (a b c : ℝ) : ∀ x : ℝ, f a b c x ≥ 0 ∨ f a b c x ≤ 0
axiom symmetry_condition (a b c : ℝ) : ∀ x : ℝ, f a b c (x - 1) = f a b c (-x - 1)
axiom inequality_condition (a b c : ℝ) : ∀ x : ℝ, 0 < x ∧ x < 5 → x ≤ f a b c x ∧ f a b c x ≤ 2 * |x - 1| + 1

theorem find_f1 (a b c : ℝ) : f a b c 1 = 1 := sorry

theorem find_fx (a b c : ℝ) : ∀ x : ℝ, f a b c x = (1 / 4) * (x + 1) ^ 2 := sorry

theorem find_largest_m (a b c : ℝ) : ∃ m : ℝ, m > 1 ∧ ∀ t x : ℝ, 1 ≤ x ∧ x ≤ m → f a b c (x + t) ≤ x := sorry

end find_f1_find_fx_find_largest_m_l1089_108951


namespace complement_of_A_in_U_l1089_108936

open Set

variable (U : Set ℕ) (A : Set ℕ)

theorem complement_of_A_in_U (hU : U = {1, 2, 3, 4, 5, 6}) (hA : A = {2, 4, 6}) :
  (U \ A) = {1, 3, 5} := by 
  sorry

end complement_of_A_in_U_l1089_108936


namespace five_people_six_chairs_l1089_108982

/-- Number of ways to sit 5 people in 6 chairs -/
def ways_to_sit_in_chairs : ℕ :=
  6 * 5 * 4 * 3 * 2

theorem five_people_six_chairs : ways_to_sit_in_chairs = 720 := by
  -- placeholder for the proof
  sorry

end five_people_six_chairs_l1089_108982


namespace Randy_trip_distance_l1089_108961

theorem Randy_trip_distance (x : ℝ) (h1 : x = 4 * (x / 4 + 30 + x / 6)) : x = 360 / 7 :=
by
  have h2 : x = ((3 * x + 36 * 30 + 2 * x) / 12) := sorry
  have h3 : x = (5 * x / 12 + 30) := sorry
  have h4 : 30 = x - (5 * x / 12) := sorry
  have h5 : 30 = 7 * x / 12 := sorry
  have h6 : x = (12 * 30) / 7 := sorry
  have h7 : x = 360 / 7 := sorry
  exact h7

end Randy_trip_distance_l1089_108961


namespace total_hours_charged_l1089_108911

theorem total_hours_charged (K P M : ℕ) 
  (h₁ : P = 2 * K)
  (h₂ : P = (1 / 3 : ℚ) * (K + 80))
  (h₃ : M = K + 80) : K + P + M = 144 :=
by {
    sorry
}

end total_hours_charged_l1089_108911


namespace negate_exists_l1089_108949

theorem negate_exists : 
  (¬ ∃ x : ℝ, x^2 + x - 1 > 0) ↔ (∀ x : ℝ, x^2 + x - 1 ≤ 0) :=
by sorry

end negate_exists_l1089_108949


namespace solution_set_of_inequality_l1089_108939

theorem solution_set_of_inequality 
  {f : ℝ → ℝ}
  (hf : ∀ x y : ℝ, x < y → f x > f y)
  (hA : f 0 = -2)
  (hB : f (-3) = 2) :
  {x : ℝ | |f (x - 2)| > 2 } = {x : ℝ | x < -1 ∨ x > 2} :=
by
  sorry

end solution_set_of_inequality_l1089_108939


namespace value_of_f_at_3_l1089_108947

noncomputable def f (x : ℝ) : ℝ := 8 * x^3 - 6 * x^2 - 4 * x + 5

theorem value_of_f_at_3 : f 3 = 155 := by
  sorry

end value_of_f_at_3_l1089_108947


namespace interest_rate_l1089_108994

theorem interest_rate (SI P T : ℕ) (h1 : SI = 2000) (h2 : P = 5000) (h3 : T = 10) :
  (SI = (P * R * T) / 100) -> R = 4 :=
by
  sorry

end interest_rate_l1089_108994


namespace cloth_cost_price_l1089_108965

theorem cloth_cost_price
  (meters_of_cloth : ℕ) (selling_price : ℕ) (profit_per_meter : ℕ)
  (total_profit : ℕ) (total_cost_price : ℕ) (cost_price_per_meter : ℕ) :
  meters_of_cloth = 45 →
  selling_price = 4500 →
  profit_per_meter = 14 →
  total_profit = profit_per_meter * meters_of_cloth →
  total_cost_price = selling_price - total_profit →
  cost_price_per_meter = total_cost_price / meters_of_cloth →
  cost_price_per_meter = 86 :=
by
  intros
  sorry

end cloth_cost_price_l1089_108965


namespace mary_initial_baseball_cards_l1089_108904

theorem mary_initial_baseball_cards (X : ℕ) :
  (X - 8 + 26 + 40 = 84) → (X = 26) :=
by
  sorry

end mary_initial_baseball_cards_l1089_108904


namespace train_cross_signal_pole_time_l1089_108956

theorem train_cross_signal_pole_time :
  ∀ (l_t l_p t_p : ℕ), l_t = 450 → l_p = 525 → t_p = 39 → 
  (l_t * t_p) / (l_t + l_p) = 18 := by
  sorry

end train_cross_signal_pole_time_l1089_108956


namespace total_bricks_required_l1089_108902

def courtyard_length : ℕ := 24 * 100  -- convert meters to cm
def courtyard_width : ℕ := 14 * 100  -- convert meters to cm
def brick_length : ℕ := 25
def brick_width : ℕ := 15

-- Calculate the area of the courtyard in square centimeters
def courtyard_area : ℕ := courtyard_length * courtyard_width

-- Calculate the area of one brick in square centimeters
def brick_area : ℕ := brick_length * brick_width

theorem total_bricks_required :  courtyard_area / brick_area = 8960 := by
  -- This part will have the proof, for now, we use sorry to skip it
  sorry

end total_bricks_required_l1089_108902


namespace radio_price_and_total_items_l1089_108931

theorem radio_price_and_total_items :
  ∃ (n : ℕ) (p : ℝ),
    (∀ (i : ℕ), (1 ≤ i ∧ i ≤ n) → (i = 1 ∨ ∃ (j : ℕ), i = j + 1 ∧ p = 1 + (j * 0.50))) ∧
    (n - 49 = 85) ∧
    (p = 43) ∧
    (n = 134) :=
by {
  sorry
}

end radio_price_and_total_items_l1089_108931


namespace fixer_used_30_percent_kitchen_l1089_108979

def fixer_percentage (x : ℝ) : Prop :=
  let initial_nails := 400
  let remaining_after_kitchen := initial_nails * ((100 - x) / 100)
  let remaining_after_fence := remaining_after_kitchen * 0.3
  remaining_after_fence = 84

theorem fixer_used_30_percent_kitchen : fixer_percentage 30 :=
by
  exact sorry

end fixer_used_30_percent_kitchen_l1089_108979


namespace cistern_filling_time_l1089_108997

open Real

theorem cistern_filling_time :
  let rate1 := 1 / 10
  let rate2 := 1 / 12
  let rate3 := -1 / 25
  let rate4 := 1 / 15
  let rate5 := -1 / 30
  let combined_rate := rate1 + rate2 + rate4 + rate3 + rate5
  (300 / combined_rate) = (300 / 53) := by
  let rate1 := 1 / 10
  let rate2 := 1 / 12
  let rate3 := -1 / 25
  let rate4 := 1 / 15
  let rate5 := -1 / 30
  let combined_rate := rate1 + rate2 + rate4 + rate3 + rate5
  sorry

end cistern_filling_time_l1089_108997


namespace tickets_used_l1089_108925

def total_rides (ferris_wheel_rides bumper_car_rides : ℕ) : ℕ :=
  ferris_wheel_rides + bumper_car_rides

def tickets_per_ride : ℕ := 3

def total_tickets (total_rides tickets_per_ride : ℕ) : ℕ :=
  total_rides * tickets_per_ride

theorem tickets_used :
  total_tickets (total_rides 7 3) tickets_per_ride = 30 := by
  sorry

end tickets_used_l1089_108925


namespace ratio_of_third_week_growth_l1089_108912

-- Define the given conditions
def week1_growth : ℕ := 2  -- growth in week 1
def week2_growth : ℕ := 2 * week1_growth  -- growth in week 2
def total_height : ℕ := 22  -- total height after three weeks

/- 
  Statement: Prove that the growth in the third week divided by 
  the growth in the second week is 4, i.e., the ratio 4:1.
-/
theorem ratio_of_third_week_growth :
  ∃ x : ℕ, 4 * x = (total_height - week1_growth - week2_growth) ∧ x = 4 :=
by
  use 4
  sorry

end ratio_of_third_week_growth_l1089_108912


namespace part_a_part_b_part_c_part_d_l1089_108901

-- Part a
theorem part_a (x : ℝ) : 
  (5 / x - x / 3 = 1 / 6) ↔ x = 6 := 
by
  sorry

-- Part b
theorem part_b (a : ℝ) : 
  ¬ ∃ a, (1 / 2 + a / 4 = a / 4) := 
by
  sorry

-- Part c
theorem part_c (y : ℝ) : 
  (9 / y - y / 21 = 17 / 21) ↔ y = 7 := 
by
  sorry

-- Part d
theorem part_d (z : ℝ) : 
  (z / 8 - 1 / z = 3 / 8) ↔ z = 4 := 
by
  sorry

end part_a_part_b_part_c_part_d_l1089_108901


namespace determine_sum_l1089_108987

theorem determine_sum (P R : ℝ) (h : 3 * P * (R + 1) / 100 - 3 * P * R / 100 = 78) : 
  P = 2600 :=
sorry

end determine_sum_l1089_108987


namespace integer_pairs_satisfying_equation_l1089_108996

theorem integer_pairs_satisfying_equation :
  {p : ℤ × ℤ | (p.1)^3 + (p.2)^3 - 3*(p.1)^2 + 6*(p.2)^2 + 3*(p.1) + 12*(p.2) + 6 = 0}
  = {(1, -1), (2, -2)} := 
sorry

end integer_pairs_satisfying_equation_l1089_108996


namespace gcd_7488_12467_eq_39_l1089_108914

noncomputable def gcd_7488_12467 : ℕ := Nat.gcd 7488 12467

theorem gcd_7488_12467_eq_39 : gcd_7488_12467 = 39 :=
sorry

end gcd_7488_12467_eq_39_l1089_108914


namespace payback_period_l1089_108990

def system_unit_cost : ℕ := 9499 -- cost in RUB
def graphics_card_cost : ℕ := 20990 -- cost per card in RUB
def num_graphics_cards : ℕ := 2
def system_unit_power : ℕ := 120 -- power in watts
def graphics_card_power : ℕ := 185 -- power per card in watts
def earnings_per_card_per_day_ethereum : ℚ := 0.00630
def ethereum_to_rub : ℚ := 27790.37 -- RUB per ETH
def electricity_cost_per_kwh : ℚ := 5.38 -- RUB per kWh
def total_investment : ℕ := system_unit_cost + num_graphics_cards * graphics_card_cost
def total_power_consumption_watts : ℕ := system_unit_power + num_graphics_cards * graphics_card_power
def total_power_consumption_kwh_per_day : ℚ := total_power_consumption_watts / 1000 * 24
def daily_earnings_rub : ℚ := earnings_per_card_per_day_ethereum * num_graphics_cards * ethereum_to_rub
def daily_energy_cost : ℚ := total_power_consumption_kwh_per_day * electricity_cost_per_kwh
def net_daily_profit : ℚ := daily_earnings_rub - daily_energy_cost

theorem payback_period : total_investment / net_daily_profit = 179 := by
  sorry

end payback_period_l1089_108990


namespace overhead_percentage_l1089_108968

def purchase_price : ℝ := 48
def markup : ℝ := 30
def net_profit : ℝ := 12

-- Define the theorem to be proved
theorem overhead_percentage : ((markup - net_profit) / purchase_price) * 100 = 37.5 := by
  sorry

end overhead_percentage_l1089_108968


namespace village_population_rate_l1089_108969

theorem village_population_rate (r : ℕ) :
  let PX := 72000
  let PY := 42000
  let decrease_rate_X := 1200
  let years := 15
  let population_X_after_years := PX - decrease_rate_X * years
  let population_Y_after_years := PY + r * years
  population_X_after_years = population_Y_after_years → r = 800 :=
by
  sorry

end village_population_rate_l1089_108969


namespace value_of_5_T_3_l1089_108991

def operation (a b : ℕ) : ℕ := 4 * a + 6 * b

theorem value_of_5_T_3 : operation 5 3 = 38 :=
by
  -- proof (which is not required)
  sorry

end value_of_5_T_3_l1089_108991


namespace exists_N_for_sqrt_expressions_l1089_108945

theorem exists_N_for_sqrt_expressions 
  (p q n : ℕ) (hp : 0 < p) (hq : 0 < q) (hn : 0 < n) (h_q_le_p2 : q ≤ p^2) :
  ∃ N : ℕ, 
    (N > 0) ∧ 
    ((p - Real.sqrt (p^2 - q))^n = N - Real.sqrt (N^2 - q^n)) ∧ 
    ((p + Real.sqrt (p^2 - q))^n = N + Real.sqrt (N^2 - q^n)) :=
sorry

end exists_N_for_sqrt_expressions_l1089_108945


namespace fixed_monthly_fee_l1089_108934

variable (x y : Real)

theorem fixed_monthly_fee :
  (x + y = 15.30) →
  (x + 1.5 * y = 20.55) →
  (x = 4.80) :=
by
  intros h1 h2
  sorry

end fixed_monthly_fee_l1089_108934


namespace greatest_possible_median_l1089_108973

theorem greatest_possible_median (k m r s t : ℕ) (h_avg : (k + m + r + s + t) / 5 = 10) (h_order : k < m ∧ m < r ∧ r < s ∧ s < t) (h_t : t = 20) : r = 8 :=
by
  sorry

end greatest_possible_median_l1089_108973


namespace E1_E2_complementary_l1089_108903

-- Define the universal set for a fair die with six faces
def universalSet : Set ℕ := {1, 2, 3, 4, 5, 6}

-- Define each event as a set based on the problem conditions
def E1 : Set ℕ := {1, 3, 5}
def E2 : Set ℕ := {2, 4, 6}
def E3 : Set ℕ := {4, 5, 6}
def E4 : Set ℕ := {1, 2}

-- Define complementary events
def areComplementary (A B : Set ℕ) : Prop :=
  (A ∪ B = universalSet) ∧ (A ∩ B = ∅)

-- State the theorem that events E1 and E2 are complementary
theorem E1_E2_complementary : areComplementary E1 E2 :=
sorry

end E1_E2_complementary_l1089_108903


namespace find_number_A_l1089_108920

theorem find_number_A (A B : ℝ) (h₁ : A + B = 14.85) (h₂ : B = 10 * A) : A = 1.35 :=
sorry

end find_number_A_l1089_108920
