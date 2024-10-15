import Mathlib

namespace NUMINAMATH_GPT_grandma_red_bacon_bits_l1476_147638

theorem grandma_red_bacon_bits:
  ∀ (mushrooms cherryTomatoes pickles baconBits redBaconBits : ℕ),
    mushrooms = 3 →
    cherryTomatoes = 2 * mushrooms →
    pickles = 4 * cherryTomatoes →
    baconBits = 4 * pickles →
    redBaconBits = 1 / 3 * baconBits →
    redBaconBits = 32 := 
by
  intros mushrooms cherryTomatoes pickles baconBits redBaconBits
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_grandma_red_bacon_bits_l1476_147638


namespace NUMINAMATH_GPT_water_consumption_150_litres_per_household_4_months_6000_litres_l1476_147688

def number_of_households (household_water_use_per_month : ℕ) (water_supply : ℕ) (duration_months : ℕ) : ℕ :=
  water_supply / (household_water_use_per_month * duration_months)

theorem water_consumption_150_litres_per_household_4_months_6000_litres : 
  number_of_households 150 6000 4 = 10 :=
by
  sorry

end NUMINAMATH_GPT_water_consumption_150_litres_per_household_4_months_6000_litres_l1476_147688


namespace NUMINAMATH_GPT_solve_equation1_solve_equation2_l1476_147689

-- Define the two equations
def equation1 (x : ℝ) := 3 * x - 4 = -2 * (x - 1)
def equation2 (x : ℝ) := 1 + (2 * x + 1) / 3 = (3 * x - 2) / 2

-- The statements to prove
theorem solve_equation1 : ∃ x : ℝ, equation1 x ∧ x = 1.2 :=
by
  sorry

theorem solve_equation2 : ∃ x : ℝ, equation2 x ∧ x = 2.8 :=
by
  sorry

end NUMINAMATH_GPT_solve_equation1_solve_equation2_l1476_147689


namespace NUMINAMATH_GPT_problem_I_problem_II_problem_III_l1476_147633

noncomputable def f (x a b : ℝ) : ℝ := x^3 + a * x^2 + b * x + 3

theorem problem_I (a b : ℝ) (h_a : a = 0) :
  (b ≥ 0 → ∀ x : ℝ, 3 * x^2 + b ≥ 0) ∧
  (b < 0 → 
    ∀ x : ℝ, (x < -Real.sqrt (-b / 3) ∨ x > Real.sqrt (-b / 3)) → 
      3 * x^2 + b > 0) := sorry

theorem problem_II (b : ℝ) :
  ∃ x0 : ℝ, f x0 0 b = x0 ∧ (3 * x0^2 + b = 0) ↔ b = -3 := sorry

theorem problem_III :
  ∀ a b : ℝ, ¬ (∃ x1 x2 : ℝ, x1 ≠ x2 ∧
    (3 * x1^2 + 2 * a * x1 + b = 0) ∧
    (3 * x2^2 + 2 * a * x2 + b = 0) ∧
    (f x1 a b = x1) ∧
    (f x2 a b = x2)) := sorry

end NUMINAMATH_GPT_problem_I_problem_II_problem_III_l1476_147633


namespace NUMINAMATH_GPT_parametric_to_line_segment_l1476_147611

theorem parametric_to_line_segment :
  ∀ t : ℝ, 0 ≤ t ∧ t ≤ 5 →
  ∃ x y : ℝ, x = 3 * t^2 + 2 ∧ y = t^2 - 1 ∧ (x - 3 * y = 5) ∧ (-1 ≤ y ∧ y ≤ 24) :=
by
  sorry

end NUMINAMATH_GPT_parametric_to_line_segment_l1476_147611


namespace NUMINAMATH_GPT_f_le_g_for_a_eq_neg1_l1476_147655

noncomputable def f (a : ℝ) (b : ℝ) (x : ℝ) : ℝ :=
  (a * x + b) * Real.exp x

noncomputable def g (t : ℝ) (x : ℝ) : ℝ :=
  (1 / 2) * x - Real.log x + t

theorem f_le_g_for_a_eq_neg1 (t : ℝ) :
  let b := 3
  ∃ x ∈ Set.Ioi 0, f (-1) b x ≤ g t x ↔ t ≤ Real.exp 2 - 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_f_le_g_for_a_eq_neg1_l1476_147655


namespace NUMINAMATH_GPT_tommy_needs_to_save_l1476_147601

theorem tommy_needs_to_save (books : ℕ) (cost_per_book : ℕ) (money_he_has : ℕ) 
  (total_cost : ℕ) (money_needed : ℕ) 
  (h1 : books = 8)
  (h2 : cost_per_book = 5)
  (h3 : money_he_has = 13)
  (h4 : total_cost = books * cost_per_book) :
  money_needed = total_cost - money_he_has ∧ money_needed = 27 :=
by 
  sorry

end NUMINAMATH_GPT_tommy_needs_to_save_l1476_147601


namespace NUMINAMATH_GPT_intersection_A_B_l1476_147690

open Set

variable (x : ℝ)

def setA : Set ℝ := {x | x^2 - 3 * x ≤ 0}
def setB : Set ℝ := {1, 2}

theorem intersection_A_B : setA ∩ setB = {1, 2} :=
by
  sorry

end NUMINAMATH_GPT_intersection_A_B_l1476_147690


namespace NUMINAMATH_GPT_average_price_over_3_months_l1476_147642

theorem average_price_over_3_months (dMay : ℕ) 
  (pApril pMay pJune : ℝ) 
  (h1 : pApril = 1.20) 
  (h2 : pMay = 1.20) 
  (h3 : pJune = 3.00) 
  (h4 : dApril = 2 / 3 * dMay) 
  (h5 : dJune = 2 * dApril) :
  ((dApril * pApril + dMay * pMay + dJune * pJune) / (dApril + dMay + dJune) = 2) := 
by sorry

end NUMINAMATH_GPT_average_price_over_3_months_l1476_147642


namespace NUMINAMATH_GPT_canoe_vs_kayak_l1476_147695

theorem canoe_vs_kayak (
  C K : ℕ 
) (h1 : 14 * C + 15 * K = 288) 
  (h2 : C = (3 * K) / 2) : 
  C - K = 4 := 
sorry

end NUMINAMATH_GPT_canoe_vs_kayak_l1476_147695


namespace NUMINAMATH_GPT_divide_fractions_l1476_147616

theorem divide_fractions : (3 / 8) / (1 / 4) = 3 / 2 :=
by sorry

end NUMINAMATH_GPT_divide_fractions_l1476_147616


namespace NUMINAMATH_GPT_blue_tshirts_in_pack_l1476_147672

theorem blue_tshirts_in_pack
  (packs_white : ℕ := 2) 
  (white_per_pack : ℕ := 5) 
  (packs_blue : ℕ := 4)
  (cost_per_tshirt : ℕ := 3)
  (total_cost : ℕ := 66)
  (B : ℕ := 3) :
  (packs_white * white_per_pack * cost_per_tshirt) + (packs_blue * B * cost_per_tshirt) = total_cost := 
by
  sorry

end NUMINAMATH_GPT_blue_tshirts_in_pack_l1476_147672


namespace NUMINAMATH_GPT_line_through_origin_and_intersection_of_lines_l1476_147603

theorem line_through_origin_and_intersection_of_lines 
  (x y : ℝ)
  (h1 : x - 3 * y + 4 = 0)
  (h2 : 2 * x + y + 5 = 0) :
  3 * x + 19 * y = 0 :=
sorry

end NUMINAMATH_GPT_line_through_origin_and_intersection_of_lines_l1476_147603


namespace NUMINAMATH_GPT_one_fourth_one_third_two_fifths_l1476_147615

theorem one_fourth_one_third_two_fifths (N : ℝ)
  (h₁ : 0.40 * N = 300) :
  (1/4) * (1/3) * (2/5) * N = 25 := 
sorry

end NUMINAMATH_GPT_one_fourth_one_third_two_fifths_l1476_147615


namespace NUMINAMATH_GPT_value_of_x_l1476_147618

theorem value_of_x (u w z y x : ℤ) (h1 : u = 95) (h2 : w = u + 10) (h3 : z = w + 25) (h4 : y = z + 15) (h5 : x = y + 12) : x = 157 := by
  sorry

end NUMINAMATH_GPT_value_of_x_l1476_147618


namespace NUMINAMATH_GPT_oldest_child_age_l1476_147680

theorem oldest_child_age 
  (avg_age : ℕ) (child1 : ℕ) (child2 : ℕ) (child3 : ℕ) (child4 : ℕ)
  (h_avg : avg_age = 8) 
  (h_child1 : child1 = 5) 
  (h_child2 : child2 = 7) 
  (h_child3 : child3 = 10)
  (h_avg_eq : (child1 + child2 + child3 + child4) / 4 = avg_age) :
  child4 = 10 := 
by 
  sorry

end NUMINAMATH_GPT_oldest_child_age_l1476_147680


namespace NUMINAMATH_GPT_washing_machine_heavy_washes_l1476_147606

theorem washing_machine_heavy_washes
  (H : ℕ)                                  -- The number of heavy washes
  (heavy_wash_gallons : ℕ := 20)            -- Gallons of water for a heavy wash
  (regular_wash_gallons : ℕ := 10)          -- Gallons of water for a regular wash
  (light_wash_gallons : ℕ := 2)             -- Gallons of water for a light wash
  (num_regular_washes : ℕ := 3)             -- Number of regular washes
  (num_light_washes : ℕ := 1)               -- Number of light washes
  (num_bleach_rinses : ℕ := 2)              -- Number of bleach rinses (extra light washes)
  (total_water_needed : ℕ := 76)            -- Total gallons of water needed
  (h_regular_wash_water : num_regular_washes * regular_wash_gallons = 30)
  (h_light_wash_water : num_light_washes * light_wash_gallons = 2)
  (h_bleach_rinse_water : num_bleach_rinses * light_wash_gallons = 4) :
  20 * H + 30 + 2 + 4 = 76 → H = 2 :=
by
  intros
  sorry

end NUMINAMATH_GPT_washing_machine_heavy_washes_l1476_147606


namespace NUMINAMATH_GPT_trigonometric_identity_l1476_147614

open Real

theorem trigonometric_identity (α φ : ℝ) :
  cos α ^ 2 + cos φ ^ 2 + cos (α + φ) ^ 2 - 2 * cos α * cos φ * cos (α + φ) = 1 :=
sorry

end NUMINAMATH_GPT_trigonometric_identity_l1476_147614


namespace NUMINAMATH_GPT_isosceles_triangle_l1476_147691

def triangle_is_isosceles (a b c : ℝ) (A B C : ℝ) : Prop :=
  a^2 + b^2 - (a * Real.cos B + b * Real.cos A)^2 = 2 * a * b * Real.cos B → (B = C)

theorem isosceles_triangle (a b c A B C : ℝ) (h : a^2 + b^2 - (a * Real.cos B + b * Real.cos A)^2 = 2 * a * b * Real.cos B) : B = C :=
  sorry

end NUMINAMATH_GPT_isosceles_triangle_l1476_147691


namespace NUMINAMATH_GPT_apples_in_blue_basket_l1476_147646

-- Define the number of bananas in the blue basket
def bananas := 12

-- Define the total number of fruits in the blue basket
def totalFruits := 20

-- Define the number of apples as total fruits minus bananas
def apples := totalFruits - bananas

-- Prove that the number of apples in the blue basket is 8
theorem apples_in_blue_basket : apples = 8 := by
  sorry

end NUMINAMATH_GPT_apples_in_blue_basket_l1476_147646


namespace NUMINAMATH_GPT_minimum_area_sum_l1476_147621

-- Define the coordinates and the conditions
variable {x1 y1 x2 y2 : ℝ}
variable (on_parabola_A : y1^2 = x1)
variable (on_parabola_B : y2^2 = x2)
variable (y1_pos : y1 > 0)
variable (y2_neg : y2 < 0)
variable (dot_product : x1 * x2 + y1 * y2 = 2)

-- Define the function to calculate areas
noncomputable def area_sum (y1 y2 x1 x2 : ℝ) : ℝ :=
  1/2 * 2 * (y1 - y2) + 1/2 * 1/4 * y1

theorem minimum_area_sum :
  ∃ y1 y2 x1 x2, y1^2 = x1 ∧ y2^2 = x2 ∧ y1 > 0 ∧ y2 < 0 ∧ x1 * x2 + y1 * y2 = 2 ∧
  (area_sum y1 y2 x1 x2 = 3) := sorry

end NUMINAMATH_GPT_minimum_area_sum_l1476_147621


namespace NUMINAMATH_GPT_flowers_sold_difference_l1476_147662

def number_of_daisies_sold_on_second_day (d2 : ℕ) (d3 : ℕ) (d_sum : ℕ) : Prop :=
  d3 = 2 * d2 - 10 ∧
  d_sum = 45 + d2 + d3 + 120

theorem flowers_sold_difference (d2 : ℕ) (d3 : ℕ) (d_sum : ℕ) 
  (h : number_of_daisies_sold_on_second_day d2 d3 d_sum) :
  45 + d2 + d3 + 120 = 350 → 
  d2 - 45 = 20 := 
by
  sorry

end NUMINAMATH_GPT_flowers_sold_difference_l1476_147662


namespace NUMINAMATH_GPT_total_points_earned_l1476_147627

def defeated_enemies := 15
def points_per_enemy := 12
def level_completion_points := 20
def special_challenges_completed := 5
def points_per_special_challenge := 10

theorem total_points_earned :
  defeated_enemies * points_per_enemy
  + level_completion_points
  + special_challenges_completed * points_per_special_challenge = 250 :=
by
  -- The proof would be developed here.
  sorry

end NUMINAMATH_GPT_total_points_earned_l1476_147627


namespace NUMINAMATH_GPT_initial_red_martians_l1476_147637

/-- Red Martians always tell the truth, while Blue Martians lie and then turn red.
    In a group of 2018 Martians, they answered in the sequence 1, 2, 3, ..., 2018 to the question
    of how many of them were red at that moment. Prove that the initial number of red Martians was 0 or 1. -/
theorem initial_red_martians (N : ℕ) (answers : Fin (N+1) → ℕ) :
  (∀ i : Fin (N+1), answers i = i.succ) → N = 2018 → (initial_red_martians_count = 0 ∨ initial_red_martians_count = 1)
:= sorry

end NUMINAMATH_GPT_initial_red_martians_l1476_147637


namespace NUMINAMATH_GPT_john_has_388_pennies_l1476_147653

theorem john_has_388_pennies (k : ℕ) (j : ℕ) (hk : k = 223) (hj : j = k + 165) : j = 388 := by
  sorry

end NUMINAMATH_GPT_john_has_388_pennies_l1476_147653


namespace NUMINAMATH_GPT_tan_alpha_value_l1476_147692

noncomputable def f (x : ℝ) := 3 * Real.sin x + 4 * Real.cos x

theorem tan_alpha_value (α : ℝ) (h : ∀ x : ℝ, f x ≥ f α) : Real.tan α = 3 / 4 := 
sorry

end NUMINAMATH_GPT_tan_alpha_value_l1476_147692


namespace NUMINAMATH_GPT_range_a_range_b_l1476_147675

def set_A : Set ℝ := {x | Real.log x / Real.log 2 > 2}
def set_B (a : ℝ) : Set ℝ := {x | x > a}
def set_C (b : ℝ) : Set ℝ := {x | b + 1 < x ∧ x < 2 * b + 1}

-- Part (1)
theorem range_a (a : ℝ) : (∀ x, x ∈ set_A → x ∈ set_B a) ↔ a ∈ Set.Iic 4 := sorry

-- Part (2)
theorem range_b (b : ℝ) : (set_A ∪ set_C b = set_A) ↔ b ∈ Set.Iic 0 ∪ Set.Ici 3 := sorry

end NUMINAMATH_GPT_range_a_range_b_l1476_147675


namespace NUMINAMATH_GPT_turtles_order_l1476_147666

-- Define variables for each turtle as real numbers representing their positions
variables (O P S E R : ℝ)

-- Define the conditions given in the problem
def condition1 := S = O - 10
def condition2 := S = R + 25
def condition3 := R = E - 5
def condition4 := E = P - 25

-- Define the order of arrival
def order_of_arrival (O P S E R : ℝ) := 
     O = 0 ∧ 
     P = -5 ∧
     S = -10 ∧
     E = -30 ∧
     R = -35

-- Theorem to show the given conditions imply the order of arrival
theorem turtles_order (h1 : condition1 S O)
                     (h2 : condition2 S R)
                     (h3 : condition3 R E)
                     (h4 : condition4 E P) :
  order_of_arrival O P S E R :=
by sorry

end NUMINAMATH_GPT_turtles_order_l1476_147666


namespace NUMINAMATH_GPT_ticket_cost_calculation_l1476_147623

theorem ticket_cost_calculation :
  let adult_price := 12
  let child_price := 10
  let num_adults := 3
  let num_children := 3
  let total_cost := (num_adults * adult_price) + (num_children * child_price)
  total_cost = 66 := 
by
  rfl -- or add sorry to skip proof

end NUMINAMATH_GPT_ticket_cost_calculation_l1476_147623


namespace NUMINAMATH_GPT_calc_6_4_3_199_plus_100_l1476_147667

theorem calc_6_4_3_199_plus_100 (a b : ℕ) (h_a : a = 199) (h_b : b = 100) :
  6 * a + 4 * a + 3 * a + a + b = 2886 :=
by
  sorry

end NUMINAMATH_GPT_calc_6_4_3_199_plus_100_l1476_147667


namespace NUMINAMATH_GPT_min_value_of_a_l1476_147605

theorem min_value_of_a (a : ℝ) (h : a > 0) (h₁ : ∀ x : ℝ, |x - a| + |1 - x| ≥ 1) : a ≥ 2 := 
sorry

end NUMINAMATH_GPT_min_value_of_a_l1476_147605


namespace NUMINAMATH_GPT_first_number_in_sum_l1476_147682

theorem first_number_in_sum (a b c : ℝ) (h : a + b + c = 3.622) : a = 3.15 :=
by
  -- Assume the given values of b and c
  have hb : b = 0.014 := sorry
  have hc : c = 0.458 := sorry
  -- From the assumption h and hb, hc, we deduce a = 3.15
  sorry

end NUMINAMATH_GPT_first_number_in_sum_l1476_147682


namespace NUMINAMATH_GPT_milk_removal_replacement_l1476_147661

theorem milk_removal_replacement (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 45) :
  (45 - x) * (45 - x) / 45 = 28.8 → x = 9 :=
by
  -- skipping the proof for now
  sorry

end NUMINAMATH_GPT_milk_removal_replacement_l1476_147661


namespace NUMINAMATH_GPT_find_a_b_l1476_147656

theorem find_a_b (a b : ℝ) :
  (∀ x : ℝ, (x < -2 ∨ x > 1) → (x^2 + a * x + b > 0)) →
  (a = 1 ∧ b = -2) :=
by
  sorry

end NUMINAMATH_GPT_find_a_b_l1476_147656


namespace NUMINAMATH_GPT_solve_inequality1_solve_inequality2_l1476_147657

-- Problem 1: Solve the inequality (1)
theorem solve_inequality1 (x : ℝ) (h : x ≠ -4) : 
  (2 - x) / (x + 4) ≤ 0 ↔ (x ≥ 2 ∨ x < -4) := sorry

-- Problem 2: Solve the inequality (2) for different cases of a
theorem solve_inequality2 (x a : ℝ) : 
  (x^2 - 3 * a * x + 2 * a^2 ≥ 0) ↔
  (if a > 0 then (x ≥ 2 * a ∨ x ≤ a) 
   else if a < 0 then (x ≥ a ∨ x ≤ 2 * a) 
   else true) := sorry

end NUMINAMATH_GPT_solve_inequality1_solve_inequality2_l1476_147657


namespace NUMINAMATH_GPT_best_marksman_score_l1476_147610

def team_size : ℕ := 6
def total_points : ℕ := 497
def hypothetical_best_score : ℕ := 92
def hypothetical_average : ℕ := 84

theorem best_marksman_score :
  let total_with_hypothetical_best := team_size * hypothetical_average
  let difference := total_with_hypothetical_best - total_points
  let actual_best_score := hypothetical_best_score - difference
  actual_best_score = 85 := 
by
  -- Definitions in Lean are correctly set up
  intro total_with_hypothetical_best difference actual_best_score
  sorry

end NUMINAMATH_GPT_best_marksman_score_l1476_147610


namespace NUMINAMATH_GPT_polynomial_roots_l1476_147613

theorem polynomial_roots :
  (∃ x : ℝ, x^4 - 16*x^3 + 91*x^2 - 216*x + 180 = 0) ↔ (x = 2 ∨ x = 3 ∨ x = 5 ∨ x = 6) := 
sorry

end NUMINAMATH_GPT_polynomial_roots_l1476_147613


namespace NUMINAMATH_GPT_smallest_perfect_cube_divisor_l1476_147693

theorem smallest_perfect_cube_divisor (p q r : ℕ) [hp : Fact (Nat.Prime p)] [hq : Fact (Nat.Prime q)] [hr : Fact (Nat.Prime r)] (hpq : p ≠ q) (hpr : p ≠ r) (hqr : q ≠ r) :
  ∃ k : ℕ, (k = (p * q * r^2)^3) ∧ (∃ n, n = p * q^3 * r^4 ∧ n ∣ k) := 
sorry

end NUMINAMATH_GPT_smallest_perfect_cube_divisor_l1476_147693


namespace NUMINAMATH_GPT_RiverJoe_popcorn_shrimp_price_l1476_147617

theorem RiverJoe_popcorn_shrimp_price
  (price_catfish : ℝ)
  (total_orders : ℕ)
  (total_revenue : ℝ)
  (orders_popcorn_shrimp : ℕ)
  (catfish_revenue : ℝ)
  (popcorn_shrimp_price : ℝ) :
  price_catfish = 6.00 →
  total_orders = 26 →
  total_revenue = 133.50 →
  orders_popcorn_shrimp = 9 →
  catfish_revenue = (total_orders - orders_popcorn_shrimp) * price_catfish →
  catfish_revenue + orders_popcorn_shrimp * popcorn_shrimp_price = total_revenue →
  popcorn_shrimp_price = 3.50 :=
by
  intros price_catfish_eq total_orders_eq total_revenue_eq orders_popcorn_shrimp_eq catfish_revenue_eq revenue_eq
  sorry

end NUMINAMATH_GPT_RiverJoe_popcorn_shrimp_price_l1476_147617


namespace NUMINAMATH_GPT_value_of_b_minus_d_squared_l1476_147644

theorem value_of_b_minus_d_squared (a b c d : ℤ) 
  (h1 : a - b - c + d = 18) 
  (h2 : a + b - c - d = 6) : 
  (b - d)^2 = 36 := 
by 
  sorry

end NUMINAMATH_GPT_value_of_b_minus_d_squared_l1476_147644


namespace NUMINAMATH_GPT_numeric_puzzle_AB_eq_B_pow_V_l1476_147671

theorem numeric_puzzle_AB_eq_B_pow_V 
  (A B V : ℕ)
  (h_A_different_digits : A ≠ B ∧ A ≠ V ∧ B ≠ V)
  (h_AB_two_digits : 10 ≤ 10 * A + B ∧ 10 * A + B < 100) :
  (10 * A + B = B^V) ↔ 
  (10 * A + B = 32 ∨ 10 * A + B = 36 ∨ 10 * A + B = 64) :=
sorry

end NUMINAMATH_GPT_numeric_puzzle_AB_eq_B_pow_V_l1476_147671


namespace NUMINAMATH_GPT_cricket_innings_l1476_147681

theorem cricket_innings (n : ℕ) (h1 : (36 * n) / n = 36) (h2 : (36 * n + 80) / (n + 1) = 40) : n = 10 := by
  -- The proof goes here
  sorry

end NUMINAMATH_GPT_cricket_innings_l1476_147681


namespace NUMINAMATH_GPT_triangle_side_length_l1476_147660

theorem triangle_side_length (a b p : ℝ) (H_perimeter : a + b + 10 = p) (H_a : a = 7) (H_b : b = 15) (H_p : p = 32) : 10 = 10 :=
by
  sorry

end NUMINAMATH_GPT_triangle_side_length_l1476_147660


namespace NUMINAMATH_GPT_num_remainders_prime_squares_mod_210_l1476_147674

theorem num_remainders_prime_squares_mod_210 :
  (∃ (p : ℕ) (hp : p > 7) (hprime : Prime p), 
    ∀ r : Finset ℕ, 
      (∀ q ∈ r, (∃ (k : ℕ), p = 210 * k + q)) 
      → r.card = 8) :=
sorry

end NUMINAMATH_GPT_num_remainders_prime_squares_mod_210_l1476_147674


namespace NUMINAMATH_GPT_solution_fractional_equation_l1476_147608

noncomputable def solve_fractional_equation : Prop :=
  ∀ x : ℝ, (4/(x-2) = 2/x) ↔ x = -2

theorem solution_fractional_equation :
  solve_fractional_equation :=
by
  sorry

end NUMINAMATH_GPT_solution_fractional_equation_l1476_147608


namespace NUMINAMATH_GPT_mean_equality_l1476_147624

theorem mean_equality (x : ℚ) : 
  (3 + 7 + 15) / 3 = (x + 10) / 2 → x = 20 / 3 := 
by 
  sorry

end NUMINAMATH_GPT_mean_equality_l1476_147624


namespace NUMINAMATH_GPT_waiter_date_trick_l1476_147687

theorem waiter_date_trick :
  ∃ d₂ : ℕ, ∃ x : ℝ, 
  (∀ d₁ : ℕ, ∀ x : ℝ, x + d₁ = 168) ∧
  3 * x + d₂ = 486 ∧
  3 * (x + d₂) = 516 ∧
  d₂ = 15 :=
by
  sorry

end NUMINAMATH_GPT_waiter_date_trick_l1476_147687


namespace NUMINAMATH_GPT_min_value_is_3_plus_2_sqrt_2_l1476_147645

noncomputable def minimum_value (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 2*b = a*b) : ℝ :=
a + b

theorem min_value_is_3_plus_2_sqrt_2 (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 2*b = a*b) :
  minimum_value a b h1 h2 h3 = 3 + 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_min_value_is_3_plus_2_sqrt_2_l1476_147645


namespace NUMINAMATH_GPT_trip_is_400_miles_l1476_147619

def fuel_per_mile_empty_plane := 20
def fuel_increase_per_person := 3
def fuel_increase_per_bag := 2
def number_of_passengers := 30
def number_of_crew := 5
def bags_per_person := 2
def total_fuel_needed := 106000

def fuel_consumption_per_mile :=
  fuel_per_mile_empty_plane +
  (number_of_passengers + number_of_crew) * fuel_increase_per_person +
  (number_of_passengers + number_of_crew) * bags_per_person * fuel_increase_per_bag

def trip_length := total_fuel_needed / fuel_consumption_per_mile

theorem trip_is_400_miles : trip_length = 400 := 
by sorry

end NUMINAMATH_GPT_trip_is_400_miles_l1476_147619


namespace NUMINAMATH_GPT_solve_for_pure_imaginary_l1476_147628

theorem solve_for_pure_imaginary (x : ℝ) 
  (h1 : x^2 - 1 = 0) 
  (h2 : x - 1 ≠ 0) 
  : x = -1 :=
sorry

end NUMINAMATH_GPT_solve_for_pure_imaginary_l1476_147628


namespace NUMINAMATH_GPT_triangle_lengths_relationship_l1476_147678

-- Given data
variables {a b c f_a f_b f_c t_a t_b t_c : ℝ}
-- Conditions/assumptions
variables (h1 : f_a * t_a = b * c)
variables (h2 : f_b * t_b = a * c)
variables (h3 : f_c * t_c = a * b)

-- Theorem to prove
theorem triangle_lengths_relationship :
  a^2 * b^2 * c^2 = f_a * f_b * f_c * t_a * t_b * t_c :=
by sorry

end NUMINAMATH_GPT_triangle_lengths_relationship_l1476_147678


namespace NUMINAMATH_GPT_minimum_value_of_quadratic_function_l1476_147698

noncomputable def quadratic_function (x : ℝ) : ℝ :=
  x^2 + 2

theorem minimum_value_of_quadratic_function :
  ∃ m : ℝ, (∀ x : ℝ, quadratic_function x ≥ m) ∧ (∀ ε > 0, ∃ x : ℝ, quadratic_function x < m + ε) ∧ m = 2 :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_of_quadratic_function_l1476_147698


namespace NUMINAMATH_GPT_temple_shop_total_cost_l1476_147663

theorem temple_shop_total_cost :
  let price_per_object := 11
  let num_people := 5
  let items_per_person := 4
  let extra_items := 4
  let total_objects := num_people * items_per_person + extra_items
  let total_cost := total_objects * price_per_object
  total_cost = 374 :=
by
  let price_per_object := 11
  let num_people := 5
  let items_per_person := 4
  let extra_items := 4
  let total_objects := num_people * items_per_person + extra_items
  let total_cost := total_objects * price_per_object
  show total_cost = 374
  sorry

end NUMINAMATH_GPT_temple_shop_total_cost_l1476_147663


namespace NUMINAMATH_GPT_Mr_Pendearly_optimal_speed_l1476_147651

noncomputable def optimal_speed (d t : ℝ) : ℝ := d / t

theorem Mr_Pendearly_optimal_speed :
  ∀ (d t : ℝ),
  (d = 45 * (t + 1/15)) →
  (d = 75 * (t - 1/15)) →
  optimal_speed d t = 56.25 :=
by
  intros d t h1 h2
  have h_d_eq_45 := h1
  have h_d_eq_75 := h2
  sorry

end NUMINAMATH_GPT_Mr_Pendearly_optimal_speed_l1476_147651


namespace NUMINAMATH_GPT_find_b_l1476_147612

theorem find_b (a b c : ℕ) (h1 : a * b + b * c - c * a = 0) (h2 : a - c = 101) (h3 : a > 0) (h4 : b > 0) (h5 : c > 0) : b = 2550 :=
sorry

end NUMINAMATH_GPT_find_b_l1476_147612


namespace NUMINAMATH_GPT_number_of_ways_to_make_78_rubles_l1476_147669

theorem number_of_ways_to_make_78_rubles : ∃ n, n = 5 ∧ ∃ x y : ℕ, 78 = 5 * x + 3 * y := sorry

end NUMINAMATH_GPT_number_of_ways_to_make_78_rubles_l1476_147669


namespace NUMINAMATH_GPT_min_boat_trips_l1476_147683
-- Import Mathlib to include necessary libraries

-- Define the problem using noncomputable theory if necessary
theorem min_boat_trips (students boat_capacity : ℕ) (h1 : students = 37) (h2 : boat_capacity = 5) : ∃ x : ℕ, x ≥ 9 :=
by
  -- Here we need to prove the assumption and goal, hence adding sorry
  sorry

end NUMINAMATH_GPT_min_boat_trips_l1476_147683


namespace NUMINAMATH_GPT_find_y_l1476_147654

theorem find_y (x y : ℕ) (h1 : x = 2407) (h2 : x^y + y^x = 2408) : y = 1 :=
sorry

end NUMINAMATH_GPT_find_y_l1476_147654


namespace NUMINAMATH_GPT_sequence_general_term_l1476_147626

noncomputable def a_n (n : ℕ) : ℝ :=
  if n = 0 then 4 else 4 * (-1 / 3)^(n - 1) 

theorem sequence_general_term (n : ℕ) (hn : n ≥ 1) 
  (hrec : ∀ n, 3 * a_n (n + 1) + a_n n = 0)
  (hinit : a_n 2 = -4 / 3) :
  a_n n = 4 * (-1 / 3)^(n - 1) := by
  sorry

end NUMINAMATH_GPT_sequence_general_term_l1476_147626


namespace NUMINAMATH_GPT_find_factors_of_224_l1476_147639

theorem find_factors_of_224 : ∃ (a b c : ℕ), a * b * c = 224 ∧ c = 2 * a ∧ a ≠ b ∧ b ≠ c :=
by
  -- Prove that the factors meeting the criteria exist
  sorry

end NUMINAMATH_GPT_find_factors_of_224_l1476_147639


namespace NUMINAMATH_GPT_total_income_per_minute_l1476_147622

theorem total_income_per_minute :
  let black_shirt_price := 30
  let black_shirt_quantity := 250
  let white_shirt_price := 25
  let white_shirt_quantity := 200
  let red_shirt_price := 28
  let red_shirt_quantity := 100
  let blue_shirt_price := 25
  let blue_shirt_quantity := 50

  let black_discount := 0.05
  let white_discount := 0.08
  let red_discount := 0.10

  let total_black_income_before_discount := black_shirt_quantity * black_shirt_price
  let total_white_income_before_discount := white_shirt_quantity * white_shirt_price
  let total_red_income_before_discount := red_shirt_quantity * red_shirt_price
  let total_blue_income_before_discount := blue_shirt_quantity * blue_shirt_price

  let total_income_before_discount :=
    total_black_income_before_discount + total_white_income_before_discount + total_red_income_before_discount + total_blue_income_before_discount

  let total_black_discount := black_discount * total_black_income_before_discount
  let total_white_discount := white_discount * total_white_income_before_discount
  let total_red_discount := red_discount * total_red_income_before_discount

  let total_discount :=
    total_black_discount + total_white_discount + total_red_discount

  let total_income_after_discount :=
    total_income_before_discount - total_discount

  let total_minutes := 40
  let total_income_per_minute := total_income_after_discount / total_minutes

  total_income_per_minute = 387.38 := by
  sorry

end NUMINAMATH_GPT_total_income_per_minute_l1476_147622


namespace NUMINAMATH_GPT_car_return_speed_l1476_147699

theorem car_return_speed (d : ℕ) (r : ℕ) (h₁ : d = 180) (h₂ : (2 * d) / ((d / 90) + (d / r)) = 60) : r = 45 :=
by
  rw [h₁] at h₂
  have h3 : 2 * 180 / ((180 / 90) + (180 / r)) = 60 := h₂
  -- The rest of the proof involves solving for r, but here we only need the statement
  sorry

end NUMINAMATH_GPT_car_return_speed_l1476_147699


namespace NUMINAMATH_GPT_black_king_eventually_in_check_l1476_147649

theorem black_king_eventually_in_check 
  (n : ℕ) (h1 : n = 1000) (r : ℕ) (h2 : r = 499)
  (rooks : Fin r → (ℕ × ℕ)) (king : ℕ × ℕ)
  (take_not_allowed : ∀ rk : Fin r, rooks rk ≠ king) :
  ∃ m : ℕ, m ≤ 1000 ∧ (∃ t : Fin r, rooks t = king) :=
by
  sorry

end NUMINAMATH_GPT_black_king_eventually_in_check_l1476_147649


namespace NUMINAMATH_GPT_inequality_4th_power_l1476_147673

theorem inequality_4th_power (a b : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0) (h_ineq : a ≥ b) :
  (a^4 + b^4) / 2 ≥ ((a + b) / 2)^4 :=
sorry

end NUMINAMATH_GPT_inequality_4th_power_l1476_147673


namespace NUMINAMATH_GPT_find_sum_of_min_area_ks_l1476_147636

def point := ℝ × ℝ

def A : point := (2, 9)
def B : point := (14, 18)

def is_int (k : ℝ) : Prop := ∃ (n : ℤ), k = n

def min_triangle_area (P Q R : point) : ℝ := sorry
-- Placeholder for the area formula of a triangle given three points

def valid_ks (k : ℝ) : Prop :=
  is_int k ∧ min_triangle_area A B (6, k) ≠ 0

theorem find_sum_of_min_area_ks :
  (∃ k1 k2 : ℤ, valid_ks k1 ∧ valid_ks k2 ∧ (k1 + k2) = 31) :=
sorry

end NUMINAMATH_GPT_find_sum_of_min_area_ks_l1476_147636


namespace NUMINAMATH_GPT_xy_squared_sum_l1476_147668

theorem xy_squared_sum {x y : ℝ} (h1 : (x + y)^2 = 49) (h2 : x * y = 10) : x^2 + y^2 = 29 :=
by
  sorry

end NUMINAMATH_GPT_xy_squared_sum_l1476_147668


namespace NUMINAMATH_GPT_ratio_of_areas_l1476_147620

-- Definitions of the side lengths of the triangles
noncomputable def sides_GHI : (ℕ × ℕ × ℕ) := (7, 24, 25)
noncomputable def sides_JKL : (ℕ × ℕ × ℕ) := (9, 40, 41)

-- Function to compute the area of a right triangle given its legs
def area_right_triangle (a b : ℕ) : ℚ :=
  (a * b) / 2

-- Areas of the triangles
noncomputable def area_GHI := area_right_triangle 7 24
noncomputable def area_JKL := area_right_triangle 9 40

-- Theorem: Ratio of the areas of the triangles GHI to JKL
theorem ratio_of_areas : (area_GHI / area_JKL) = 7 / 15 :=
by {
  sorry -- Proof is skipped as per instructions
}

end NUMINAMATH_GPT_ratio_of_areas_l1476_147620


namespace NUMINAMATH_GPT_problem_I_problem_II_l1476_147650

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x^2 - 2*x - 3 ≥ 0}
def B : Set ℝ := {x : ℝ | x ≥ 1}

-- Define the complement of A in the universal set U which is ℝ
def complement_U_A : Set ℝ := {x : ℝ | -1 < x ∧ x < 3}

-- Define the union of complement_U_A and B
def union_complement_U_A_B : Set ℝ := complement_U_A ∪ B

-- Proof Problem I: Prove that the set A is as specified
theorem problem_I : A = {x : ℝ | x ≤ -1 ∨ x ≥ 3} := sorry

-- Proof Problem II: Prove that the union of the complement of A and B is as specified
theorem problem_II : union_complement_U_A_B = {x : ℝ | x > -1} := sorry

end NUMINAMATH_GPT_problem_I_problem_II_l1476_147650


namespace NUMINAMATH_GPT_aira_fewer_bands_than_joe_l1476_147697

-- Define initial conditions
variables (samantha_bands aira_bands joe_bands : ℕ)
variables (shares_each : ℕ) (total_bands: ℕ)

-- Conditions from the problem
axiom h1 : shares_each = 6
axiom h2 : samantha_bands = aira_bands + 5
axiom h3 : total_bands = shares_each * 3
axiom h4 : aira_bands = 4
axiom h5 : samantha_bands + aira_bands + joe_bands = total_bands

-- The statement to be proven
theorem aira_fewer_bands_than_joe : joe_bands - aira_bands = 1 :=
sorry

end NUMINAMATH_GPT_aira_fewer_bands_than_joe_l1476_147697


namespace NUMINAMATH_GPT_second_valve_emits_more_l1476_147609

noncomputable def V1 : ℝ := 12000 / 120 -- Rate of first valve (100 cubic meters/minute)
noncomputable def V2 : ℝ := 12000 / 48 - V1 -- Rate of second valve

theorem second_valve_emits_more : V2 - V1 = 50 :=
by
  sorry

end NUMINAMATH_GPT_second_valve_emits_more_l1476_147609


namespace NUMINAMATH_GPT_car_overtakes_buses_l1476_147631

/-- 
  Buses leave the airport every 3 minutes. 
  A bus takes 60 minutes to travel from the airport to the city center. 
  A car takes 35 minutes to travel from the airport to the city center. 
  Prove that the car overtakes 8 buses on its way to the city center excluding the bus it left with.
--/
theorem car_overtakes_buses (arr_bus : ℕ) (arr_car : ℕ) (interval : ℕ) (diff : ℕ) : 
  interval = 3 → arr_bus = 60 → arr_car = 35 → diff = arr_bus - arr_car →
  ∃ n : ℕ, n = diff / interval ∧ n = 8 := by
  sorry

end NUMINAMATH_GPT_car_overtakes_buses_l1476_147631


namespace NUMINAMATH_GPT_number_of_customers_l1476_147634

theorem number_of_customers (offices_sandwiches : Nat)
                            (group_per_person_sandwiches : Nat)
                            (total_sandwiches : Nat)
                            (half_group : Nat) :
  (offices_sandwiches = 3 * 10) →
  (total_sandwiches = 54) →
  (half_group * group_per_person_sandwiches = total_sandwiches - offices_sandwiches) →
  (2 * half_group = 12) := 
by
  sorry

end NUMINAMATH_GPT_number_of_customers_l1476_147634


namespace NUMINAMATH_GPT_number_of_sarees_l1476_147659

-- Define variables representing the prices of one saree and one shirt
variables (X S T : ℕ)

-- Define the conditions 
def condition1 := X * S + 4 * T = 1600
def condition2 := S + 6 * T = 1600
def condition3 := 12 * T = 2400

-- The proof problem (statement only, without proof)
theorem number_of_sarees (X S T : ℕ) (h1 : condition1 X S T) (h2 : condition2 S T) (h3 : condition3 T) : X = 2 := by
  sorry

end NUMINAMATH_GPT_number_of_sarees_l1476_147659


namespace NUMINAMATH_GPT_calculation_expression_solve_system_of_equations_l1476_147630

-- Part 1: Prove the calculation
theorem calculation_expression :
  (6 - 2 * Real.sqrt 3) * Real.sqrt 3 - Real.sqrt ((2 - Real.sqrt 2) ^ 2) + 1 / Real.sqrt 2 = 
  6 * Real.sqrt 3 - 8 + 3 * Real.sqrt 2 / 2 :=
by
  -- proof will be here
  sorry

-- Part 2: Prove the solution of the system of equations
theorem solve_system_of_equations (x y : ℝ) :
  (5 * x - y = -9) ∧ (3 * x + y = 1) → (x = -1 ∧ y = 4) :=
by
  -- proof will be here
  sorry

end NUMINAMATH_GPT_calculation_expression_solve_system_of_equations_l1476_147630


namespace NUMINAMATH_GPT_lucy_total_packs_l1476_147684

-- Define the number of packs of cookies Lucy bought
def packs_of_cookies : ℕ := 12

-- Define the number of packs of noodles Lucy bought
def packs_of_noodles : ℕ := 16

-- Define the total number of packs of groceries Lucy bought
def total_packs_of_groceries : ℕ := packs_of_cookies + packs_of_noodles

-- Proof statement: The total number of packs of groceries Lucy bought is 28
theorem lucy_total_packs : total_packs_of_groceries = 28 := by
  sorry

end NUMINAMATH_GPT_lucy_total_packs_l1476_147684


namespace NUMINAMATH_GPT_sum_of_x_coords_Q3_l1476_147607

-- Definitions
def Q1_vertices_sum_x (S : ℝ) := S = 1050

def Q2_vertices_sum_x (S' : ℝ) (S : ℝ) := S' = S

def Q3_vertices_sum_x (S'' : ℝ) (S' : ℝ) := S'' = S'

-- Lean 4 statement
theorem sum_of_x_coords_Q3 (S : ℝ) (S' : ℝ) (S'' : ℝ) :
  Q1_vertices_sum_x S →
  Q2_vertices_sum_x S' S →
  Q3_vertices_sum_x S'' S' →
  S'' = 1050 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_x_coords_Q3_l1476_147607


namespace NUMINAMATH_GPT_area_of_rectangle_l1476_147640

theorem area_of_rectangle (y : ℕ) (h1 : 4 * (y^2) = 4 * 20^2) (h2 : 8 * y = 160) : 
    4 * (20^2) = 1600 := by 
  sorry -- Skip proof, only statement required

end NUMINAMATH_GPT_area_of_rectangle_l1476_147640


namespace NUMINAMATH_GPT_valid_triangle_side_l1476_147629

theorem valid_triangle_side (x : ℝ) (h1 : 2 + x > 6) (h2 : 2 + 6 > x) (h3 : x + 6 > 2) : x = 6 :=
by
  sorry

end NUMINAMATH_GPT_valid_triangle_side_l1476_147629


namespace NUMINAMATH_GPT_problem_solution_l1476_147632

variable (α β : ℝ)

-- Conditions
variable (h1 : 3 * Real.sin α - Real.cos α = 0)
variable (h2 : 7 * Real.sin β + Real.cos β = 0)
variable (h3 : 0 < α ∧ α < π / 2 ∧ π / 2 < β ∧ β < π)

theorem problem_solution : 2 * α - β = - (3 * π / 4) := by
  sorry

end NUMINAMATH_GPT_problem_solution_l1476_147632


namespace NUMINAMATH_GPT_a_4_is_4_l1476_147665

-- Define the general term formula of the sequence
def a (n : ℕ) : ℤ := (-1)^n * n

-- State the desired proof goal
theorem a_4_is_4 : a 4 = 4 :=
by
  -- Proof to be provided here,
  -- adding 'sorry' as we are only defining the statement, not solving it
  sorry

end NUMINAMATH_GPT_a_4_is_4_l1476_147665


namespace NUMINAMATH_GPT_julie_savings_fraction_l1476_147685

variables (S : ℝ) (x : ℝ)
theorem julie_savings_fraction (h : 12 * S * x = 4 * S * (1 - x)) : 1 - x = 3 / 4 :=
sorry

end NUMINAMATH_GPT_julie_savings_fraction_l1476_147685


namespace NUMINAMATH_GPT_negation_of_existential_prop_l1476_147694

open Real

theorem negation_of_existential_prop :
  ¬ (∃ x, x ≥ π / 2 ∧ sin x > 1) ↔ ∀ x, x < π / 2 → sin x ≤ 1 :=
by
  sorry

end NUMINAMATH_GPT_negation_of_existential_prop_l1476_147694


namespace NUMINAMATH_GPT_sachin_is_younger_by_8_years_l1476_147652

variable (S R : ℕ)

-- Conditions
axiom age_of_sachin : S = 28
axiom ratio_of_ages : S * 9 = R * 7

-- Goal
theorem sachin_is_younger_by_8_years (S R : ℕ) (h1 : S = 28) (h2 : S * 9 = R * 7) : R - S = 8 :=
by
  sorry

end NUMINAMATH_GPT_sachin_is_younger_by_8_years_l1476_147652


namespace NUMINAMATH_GPT_diane_owes_money_l1476_147677

theorem diane_owes_money (initial_amount winnings total_losses : ℤ) (h_initial : initial_amount = 100) (h_winnings : winnings = 65) (h_losses : total_losses = 215) : 
  initial_amount + winnings - total_losses = -50 := by
  sorry

end NUMINAMATH_GPT_diane_owes_money_l1476_147677


namespace NUMINAMATH_GPT_days_to_cover_half_lake_l1476_147696

-- Define the problem conditions in Lean
def doubles_every_day (size: ℕ → ℝ) : Prop :=
  ∀ n : ℕ, size (n + 1) = 2 * size n

def takes_25_days_to_cover_lake (size: ℕ → ℝ) (lake_size: ℝ) : Prop :=
  size 25 = lake_size

-- Define the main theorem
theorem days_to_cover_half_lake (size: ℕ → ℝ) (lake_size: ℝ) 
  (h1: doubles_every_day size) (h2: takes_25_days_to_cover_lake size lake_size) : 
  size 24 = lake_size / 2 :=
sorry

end NUMINAMATH_GPT_days_to_cover_half_lake_l1476_147696


namespace NUMINAMATH_GPT_exponentiation_problem_l1476_147647

theorem exponentiation_problem 
(a b : ℝ) 
(h : a ^ b = 1 / 8) : a ^ (-3 * b) = 512 := 
sorry

end NUMINAMATH_GPT_exponentiation_problem_l1476_147647


namespace NUMINAMATH_GPT_ones_digit_of_power_l1476_147643

theorem ones_digit_of_power (n : ℕ) : 
  (13 ^ (13 * (12 ^ 12)) % 10) = 9 :=
by
  sorry

end NUMINAMATH_GPT_ones_digit_of_power_l1476_147643


namespace NUMINAMATH_GPT_circumcircle_radius_l1476_147600

open Real

theorem circumcircle_radius (a b c A B C S R : ℝ) 
  (h1 : S = (1/2) * sin A * sin B * sin C)
  (h2 : S = (1/2) * a * b * sin C)
  (h3 : ∀ x y, x = y → x * cos 0 = y * cos 0):
  R = (1/2) :=
by
  sorry

end NUMINAMATH_GPT_circumcircle_radius_l1476_147600


namespace NUMINAMATH_GPT_find_angle_C_l1476_147686

theorem find_angle_C (a b c : ℝ) (h : a ^ 2 + b ^ 2 - c ^ 2 + a * b = 0) : 
  C = 2 * pi / 3 := 
sorry

end NUMINAMATH_GPT_find_angle_C_l1476_147686


namespace NUMINAMATH_GPT_grayson_vs_rudy_distance_l1476_147664

-- Definitions based on the conditions
def grayson_first_part_distance : Real := 25 * 1
def grayson_second_part_distance : Real := 20 * 0.5
def total_grayson_distance : Real := grayson_first_part_distance + grayson_second_part_distance
def rudy_distance : Real := 10 * 3

-- Theorem stating the problem to be proved
theorem grayson_vs_rudy_distance : total_grayson_distance - rudy_distance = 5 := by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_grayson_vs_rudy_distance_l1476_147664


namespace NUMINAMATH_GPT_eggs_in_each_basket_is_15_l1476_147670
open Nat

theorem eggs_in_each_basket_is_15 :
  ∃ n : ℕ, (n ∣ 30) ∧ (n ∣ 45) ∧ (n ≥ 5) ∧ (n = 15) :=
sorry

end NUMINAMATH_GPT_eggs_in_each_basket_is_15_l1476_147670


namespace NUMINAMATH_GPT_total_peaches_l1476_147658

variable (numberOfBaskets : ℕ)
variable (redPeachesPerBasket : ℕ)
variable (greenPeachesPerBasket : ℕ)

theorem total_peaches (h1 : numberOfBaskets = 1) 
                      (h2 : redPeachesPerBasket = 4)
                      (h3 : greenPeachesPerBasket = 3) :
  numberOfBaskets * (redPeachesPerBasket + greenPeachesPerBasket) = 7 := 
by
  sorry

end NUMINAMATH_GPT_total_peaches_l1476_147658


namespace NUMINAMATH_GPT_greatest_multiple_of_four_l1476_147625

theorem greatest_multiple_of_four (x : ℕ) (hx : x > 0) (h4 : x % 4 = 0) (hcube : x^3 < 800) : x ≤ 8 :=
by {
  sorry
}

end NUMINAMATH_GPT_greatest_multiple_of_four_l1476_147625


namespace NUMINAMATH_GPT_geometric_sequence_a5_l1476_147635

theorem geometric_sequence_a5 (a : ℕ → ℝ) (q : ℝ) (h1 : a 1 * a 5 = 16) (h2 : a 4 = 8) (h3 : ∀ n, a n > 0) : a 5 = 16 := 
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_a5_l1476_147635


namespace NUMINAMATH_GPT_Sophie_Spends_72_80_l1476_147679

noncomputable def SophieTotalCost : ℝ :=
  let cupcakesCost := 5 * 2
  let doughnutsCost := 6 * 1
  let applePieCost := 4 * 2
  let cookiesCost := 15 * 0.60
  let chocolateBarsCost := 8 * 1.50
  let sodaCost := 12 * 1.20
  let gumCost := 3 * 0.80
  let chipsCost := 10 * 1.10
  cupcakesCost + doughnutsCost + applePieCost + cookiesCost + chocolateBarsCost + sodaCost + gumCost + chipsCost

theorem Sophie_Spends_72_80 : SophieTotalCost = 72.80 :=
by
  sorry

end NUMINAMATH_GPT_Sophie_Spends_72_80_l1476_147679


namespace NUMINAMATH_GPT_evaluate_three_squared_raised_four_l1476_147676

theorem evaluate_three_squared_raised_four : (3^2)^4 = 6561 := by
  sorry

end NUMINAMATH_GPT_evaluate_three_squared_raised_four_l1476_147676


namespace NUMINAMATH_GPT_product_of_translated_roots_l1476_147641

noncomputable def roots (a b c : ℝ) (x : ℝ) : Prop :=
  a * x^2 + b * x + c = 0

theorem product_of_translated_roots
  {d e : ℝ}
  (h_d : roots 3 4 (-7) d)
  (h_e : roots 3 4 (-7) e)
  (sum_roots : d + e = -4 / 3)
  (product_roots : d * e = -7 / 3) :
  (d - 1) * (e - 1) = 1 :=
by
  sorry

end NUMINAMATH_GPT_product_of_translated_roots_l1476_147641


namespace NUMINAMATH_GPT_circle_range_of_a_l1476_147602

theorem circle_range_of_a (a : ℝ) : 
  (∃ (x y : ℝ), x^2 + y^2 + 2*x - 4*y + a = 0) → a < 5 := by
  sorry

end NUMINAMATH_GPT_circle_range_of_a_l1476_147602


namespace NUMINAMATH_GPT_minimize_S_l1476_147604

theorem minimize_S (n : ℕ) (a : ℕ → ℤ) (h : ∀ n, a n = 3 * n - 23) : n = 7 ↔ ∃ (m : ℕ), (∀ k ≤ m, a k <= 0) ∧ m = 7 :=
by
  sorry

end NUMINAMATH_GPT_minimize_S_l1476_147604


namespace NUMINAMATH_GPT_determine_k_l1476_147648

variable (x y z k : ℝ)

theorem determine_k
  (h1 : 9 / (x - y) = 16 / (z + y))
  (h2 : k / (x + z) = 16 / (z + y)) :
  k = 25 := by
  sorry

end NUMINAMATH_GPT_determine_k_l1476_147648
