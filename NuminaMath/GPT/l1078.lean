import Mathlib

namespace NUMINAMATH_GPT_candy_bars_per_friend_l1078_107826

-- Definitions based on conditions
def total_candy_bars : ℕ := 24
def spare_candy_bars : ℕ := 10
def number_of_friends : ℕ := 7

-- The problem statement as a Lean theorem
theorem candy_bars_per_friend :
  (total_candy_bars - spare_candy_bars) / number_of_friends = 2 := 
by
  sorry

end NUMINAMATH_GPT_candy_bars_per_friend_l1078_107826


namespace NUMINAMATH_GPT_part_i_l1078_107873

theorem part_i (n : ℤ) : (∃ k : ℤ, n = 225 * k + 99) ↔ (n % 9 = 0 ∧ (n + 1) % 25 = 0) :=
by 
  sorry

end NUMINAMATH_GPT_part_i_l1078_107873


namespace NUMINAMATH_GPT_committee_count_with_president_l1078_107830

-- Define the conditions
def total_people : ℕ := 12
def committee_size : ℕ := 5
def remaining_people : ℕ := 11
def president_inclusion : ℕ := 1

-- Define the calculation of binomial coefficient
noncomputable def binomial (n k : ℕ) : ℕ :=
  if h : k ≤ n then (Nat.choose n k) else 0

-- State the problem in Lean 4
theorem committee_count_with_president : 
  binomial remaining_people (committee_size - president_inclusion) = 330 :=
sorry

end NUMINAMATH_GPT_committee_count_with_president_l1078_107830


namespace NUMINAMATH_GPT_general_term_formula_l1078_107807

variable (a S : ℕ → ℚ)

-- Condition 1: The sum of the first n terms of the sequence {a_n} is S_n
def sum_first_n_terms (n : ℕ) : ℚ := S n

-- Condition 2: a_n = 3S_n - 2
def a_n (n : ℕ) : Prop := a n = 3 * S n - 2

theorem general_term_formula (n : ℕ) (h1 : a 1 = 1)
  (h2 : ∀ k, k ≥ 2 → a (k) = - (1/2) * a (k - 1) ) : 
  a n = (-1/2)^(n-1) :=
sorry

end NUMINAMATH_GPT_general_term_formula_l1078_107807


namespace NUMINAMATH_GPT_find_A_l1078_107825

theorem find_A (A B : ℕ) (h1 : 10 * A + 7 + (30 + B) = 73) : A = 3 := by
  sorry

end NUMINAMATH_GPT_find_A_l1078_107825


namespace NUMINAMATH_GPT_coat_price_reduction_l1078_107876

theorem coat_price_reduction (original_price : ℝ) (reduction_percent : ℝ)
  (price_is_500 : original_price = 500)
  (reduction_is_30 : reduction_percent = 0.30) :
  original_price * reduction_percent = 150 :=
by
  sorry

end NUMINAMATH_GPT_coat_price_reduction_l1078_107876


namespace NUMINAMATH_GPT_value_of_y_l1078_107802

variable (y : ℚ)

def first_boy_marbles : ℚ := 4 * y + 2
def second_boy_marbles : ℚ := 2 * y
def third_boy_marbles : ℚ := y + 3
def total_marbles : ℚ := 31

theorem value_of_y (h : first_boy_marbles y + second_boy_marbles y + third_boy_marbles y = total_marbles) :
  y = 26 / 7 :=
by
  sorry

end NUMINAMATH_GPT_value_of_y_l1078_107802


namespace NUMINAMATH_GPT_popsicle_sticks_left_l1078_107855

-- Defining the conditions
def total_money : ℕ := 10
def cost_of_molds : ℕ := 3
def cost_of_sticks : ℕ := 1
def cost_of_juice_bottle : ℕ := 2
def popsicles_per_bottle : ℕ := 20
def initial_sticks : ℕ := 100

-- Statement of the problem
theorem popsicle_sticks_left : 
  let remaining_money := total_money - cost_of_molds - cost_of_sticks
  let bottles_of_juice := remaining_money / cost_of_juice_bottle
  let total_popsicles := bottles_of_juice * popsicles_per_bottle
  let sticks_left := initial_sticks - total_popsicles
  sticks_left = 40 := by
  sorry

end NUMINAMATH_GPT_popsicle_sticks_left_l1078_107855


namespace NUMINAMATH_GPT_tank_capacity_l1078_107898

theorem tank_capacity (liters_cost : ℕ) (liters_amount : ℕ) (full_tank_cost : ℕ) (h₁ : liters_cost = 18) (h₂ : liters_amount = 36) (h₃ : full_tank_cost = 32) : 
  (full_tank_cost * liters_amount / liters_cost) = 64 :=
by 
  sorry

end NUMINAMATH_GPT_tank_capacity_l1078_107898


namespace NUMINAMATH_GPT_roots_of_polynomial_l1078_107870

   -- We need to define the polynomial and then state that the roots are exactly {0, 3, -5}
   def polynomial (x : ℝ) : ℝ := x * (x - 3)^2 * (5 + x)

   theorem roots_of_polynomial :
     {x : ℝ | polynomial x = 0} = {0, 3, -5} :=
   by
     sorry
   
end NUMINAMATH_GPT_roots_of_polynomial_l1078_107870


namespace NUMINAMATH_GPT_inequality_proof_l1078_107810

theorem inequality_proof (x y : ℝ) (hx : x > 1) (hy : y > 1) : 
  (x^2 / (y - 1) + y^2 / (x - 1) ≥ 8) :=
  sorry

end NUMINAMATH_GPT_inequality_proof_l1078_107810


namespace NUMINAMATH_GPT_find_angle_y_l1078_107869

theorem find_angle_y (ABC BAC BCA DCE CED y : ℝ)
  (h1 : ABC = 80) (h2 : BAC = 60)
  (h3 : ABC + BAC + BCA = 180)
  (h4 : CED = 90)
  (h5 : DCE = BCA)
  (h6 : DCE + CED + y = 180) :
  y = 50 :=
by
  sorry

end NUMINAMATH_GPT_find_angle_y_l1078_107869


namespace NUMINAMATH_GPT_bananas_to_oranges_cost_l1078_107824

noncomputable def cost_equivalence (bananas apples oranges : ℕ) : Prop :=
  (5 * bananas = 3 * apples) ∧
  (8 * apples = 5 * oranges)

theorem bananas_to_oranges_cost (bananas apples oranges : ℕ) 
  (h : cost_equivalence bananas apples oranges) :
  oranges = 9 :=
by sorry

end NUMINAMATH_GPT_bananas_to_oranges_cost_l1078_107824


namespace NUMINAMATH_GPT_boy_two_girls_work_completion_days_l1078_107863

-- Work rates definitions
def man_work_rate := 1 / 6
def woman_work_rate := 1 / 18
def girl_work_rate := 1 / 12
def team_work_rate := 1 / 3

-- Boy's work rate
def boy_work_rate := 1 / 36

-- Combined work rate of boy and two girls
def boy_two_girls_work_rate := boy_work_rate + 2 * girl_work_rate

-- Prove that the number of days it will take for a boy and two girls to complete the work is 36 / 7
theorem boy_two_girls_work_completion_days : (1 / boy_two_girls_work_rate) = 36 / 7 :=
by
  sorry

end NUMINAMATH_GPT_boy_two_girls_work_completion_days_l1078_107863


namespace NUMINAMATH_GPT_fraction_irreducible_l1078_107859
-- Import necessary libraries

-- Define the problem to prove
theorem fraction_irreducible (n: ℕ) (h: n > 0) : gcd (21 * n + 4) (14 * n + 3) = 1 := 
  sorry

end NUMINAMATH_GPT_fraction_irreducible_l1078_107859


namespace NUMINAMATH_GPT_loss_eq_cost_price_of_x_balls_l1078_107845

theorem loss_eq_cost_price_of_x_balls (cp ball_count sp : ℕ) (cp_ball : ℕ) 
  (hc1 : cp_ball = 60) (hc2 : cp = ball_count * cp_ball) (hs : sp = 720) 
  (hb : ball_count = 17) :
  ∃ x : ℕ, (cp - sp = x * cp_ball) ∧ x = 5 :=
by
  sorry

end NUMINAMATH_GPT_loss_eq_cost_price_of_x_balls_l1078_107845


namespace NUMINAMATH_GPT_max_gcd_of_sequence_l1078_107882

/-- Define the sequence as a function. -/
def a (n : ℕ) : ℕ := 100 + n^2

/-- Define the greatest common divisor of the sequence terms. -/
def d (n : ℕ) : ℕ := Nat.gcd (a n) (a (n + 1))

/-- State the theorem of the maximum value of d. -/
theorem max_gcd_of_sequence : ∃ n : ℕ, d n = 401 := sorry

end NUMINAMATH_GPT_max_gcd_of_sequence_l1078_107882


namespace NUMINAMATH_GPT_initial_customers_l1078_107860

theorem initial_customers (x : ℕ) (h : x - 3 + 39 = 50) : x = 14 :=
by
  sorry

end NUMINAMATH_GPT_initial_customers_l1078_107860


namespace NUMINAMATH_GPT_walking_time_12_hours_l1078_107816

theorem walking_time_12_hours :
  ∀ t : ℝ, 
  (∀ (v1 v2 : ℝ), 
  v1 = 7 ∧ v2 = 3 →
  120 = (v1 + v2) * t) →
  t = 12 := 
by
  intros t h
  specialize h 7 3 ⟨rfl, rfl⟩
  sorry

end NUMINAMATH_GPT_walking_time_12_hours_l1078_107816


namespace NUMINAMATH_GPT_person_A_work_days_l1078_107864

theorem person_A_work_days (A : ℕ) (h1 : ∀ (B : ℕ), B = 45) (h2 : 4 * (1/A + 1/45) = 2/9) : A = 30 := 
by
  sorry

end NUMINAMATH_GPT_person_A_work_days_l1078_107864


namespace NUMINAMATH_GPT_plums_for_20_oranges_l1078_107877

noncomputable def oranges_to_pears (oranges : ℕ) : ℕ :=
  (oranges / 5) * 3

noncomputable def pears_to_plums (pears : ℕ) : ℕ :=
  (pears / 4) * 6

theorem plums_for_20_oranges :
  oranges_to_pears 20 = 12 ∧ pears_to_plums 12 = 18 :=
by
  sorry

end NUMINAMATH_GPT_plums_for_20_oranges_l1078_107877


namespace NUMINAMATH_GPT_fish_weight_l1078_107871

theorem fish_weight (θ H T : ℝ) (h1 : θ = 4) (h2 : H = θ + 0.5 * T) (h3 : T = H + θ) : H + T + θ = 32 :=
by
  sorry

end NUMINAMATH_GPT_fish_weight_l1078_107871


namespace NUMINAMATH_GPT_intersecting_lines_fixed_point_l1078_107806

variable (p a b : ℝ)
variable (h1 : a ≠ 0)
variable (h2 : b ≠ 0)
variable (h3 : b^2 ≠ 2 * p * a)

def parabola (M : ℝ × ℝ) : Prop := M.2^2 = 2 * p * M.1

def fixed_points (A B : ℝ × ℝ) : Prop :=
  A = (a, b) ∧ B = (-a, 0)

def intersect_parabola (M1 M2 M : ℝ × ℝ) : Prop :=
  parabola p M ∧ parabola p M1 ∧ parabola p M2 ∧ M ≠ M1 ∧ M ≠ M2

theorem intersecting_lines_fixed_point (M M1 M2 : ℝ × ℝ)
  (hP : parabola p M) 
  (hA : (a, b) ≠ M) 
  (hB : (-a, 0) ≠ M) 
  (h_intersect : intersect_parabola p M1 M2 M) :
  ∃ C : ℝ × ℝ, C = (a, 2 * p * a / b) :=
sorry

end NUMINAMATH_GPT_intersecting_lines_fixed_point_l1078_107806


namespace NUMINAMATH_GPT_guzman_boxes_l1078_107879

noncomputable def total_doughnuts : Nat := 48
noncomputable def doughnuts_per_box : Nat := 12

theorem guzman_boxes :
  ∃ (N : Nat), N = total_doughnuts / doughnuts_per_box ∧ N = 4 :=
by
  use 4
  sorry

end NUMINAMATH_GPT_guzman_boxes_l1078_107879


namespace NUMINAMATH_GPT_difference_in_distances_l1078_107808

noncomputable def circumference (r : ℝ) : ℝ := 2 * Real.pi * r

noncomputable def distance_covered (r : ℝ) (revolutions : ℕ) : ℝ :=
  circumference r * revolutions

theorem difference_in_distances :
  let r1 := 22.4
  let r2 := 34.2
  let revolutions := 400
  let D1 := distance_covered r1 revolutions
  let D2 := distance_covered r2 revolutions
  D2 - D1 = 29628 :=
by
  sorry

end NUMINAMATH_GPT_difference_in_distances_l1078_107808


namespace NUMINAMATH_GPT_subsets_of_A_value_of_a_l1078_107890

def A := {x : ℝ | x^2 - 3*x + 2 = 0}
def B (a : ℝ) := {x : ℝ | x^2 - a*x + 2 = 0}

theorem subsets_of_A : 
  (A = {1, 2} ∧ (∀ S, S ⊆ A → S = ∅ ∨ S = {1} ∨ S = {2} ∨ S = {1, 2}))  :=
by
  sorry

theorem value_of_a (a : ℝ) (B_non_empty : B a ≠ ∅) (B_subset_A : ∀ x, x ∈ B a → x ∈ A): 
  a = 3 :=
by
  sorry

end NUMINAMATH_GPT_subsets_of_A_value_of_a_l1078_107890


namespace NUMINAMATH_GPT_trains_same_distance_at_meeting_l1078_107813

theorem trains_same_distance_at_meeting
  (d v : ℝ) (h_d : 0 < d) (h_v : 0 < v) :
  ∃ t : ℝ, v * t + v * (t - 1) = d ∧ 
  v * t = (d + v) / 2 ∧ 
  d - (v * (t - 1)) = (d + v) / 2 :=
by
  sorry

end NUMINAMATH_GPT_trains_same_distance_at_meeting_l1078_107813


namespace NUMINAMATH_GPT_value_of_expression_l1078_107895

theorem value_of_expression
  (m n : ℝ)
  (h1 : n = -2 * m + 3) :
  4 * m + 2 * n + 1 = 7 :=
sorry

end NUMINAMATH_GPT_value_of_expression_l1078_107895


namespace NUMINAMATH_GPT_twelfth_term_is_three_l1078_107868

-- Define the first term and the common difference of the arithmetic sequence
def first_term : ℚ := 1 / 4
def common_difference : ℚ := 1 / 4

-- Define the nth term of an arithmetic sequence
def nth_term (a₁ d : ℚ) (n : ℕ) : ℚ :=
  a₁ + (n - 1) * d

-- Prove that the twelfth term is equal to 3
theorem twelfth_term_is_three : nth_term first_term common_difference 12 = 3 := 
  by 
    sorry

end NUMINAMATH_GPT_twelfth_term_is_three_l1078_107868


namespace NUMINAMATH_GPT_katie_initial_candies_l1078_107866

theorem katie_initial_candies (K : ℕ) (h1 : K + 23 - 8 = 23) : K = 8 :=
sorry

end NUMINAMATH_GPT_katie_initial_candies_l1078_107866


namespace NUMINAMATH_GPT_ellipse_slope_ratio_l1078_107803

theorem ellipse_slope_ratio (a b x1 y1 x2 y2 c k1 k2 : ℝ) (h1 : a > b) (h2 : b > 0)
  (h3 : c = a / 2) (h4 : a = 2) (h5 : c = 1) (h6 : b = Real.sqrt 3) 
  (h7 : 3 * x1 ^ 2 + 4 * y1 ^ 2 = 12 * c ^ 2) 
  (h8 : 3 * x2 ^ 2 + 4 * y2 ^ 2 = 12 * c ^ 2) 
  (h9 : x1 = y1 - c) (h10 : x2 = y2 - c)
  (h11 : y1^2 = 9 / 4)
  (h12 : y1 = -3 / 2 ∨ y1 = 3 / 2) 
  (h13 : k1 = -3 / 2) 
  (h14 : k2 = -1 / 2) :
  k1 / k2 = 3 := 
  sorry

end NUMINAMATH_GPT_ellipse_slope_ratio_l1078_107803


namespace NUMINAMATH_GPT_seeds_total_l1078_107834

noncomputable def seeds_planted (x : ℕ) (y : ℕ) (z : ℕ) : ℕ :=
x + y + z

theorem seeds_total (x : ℕ) (H1 :  y = 5 * x) (H2 : x + y = 156) (z : ℕ) 
(H3 : z = 4) : seeds_planted x y z = 160 :=
by
  sorry

end NUMINAMATH_GPT_seeds_total_l1078_107834


namespace NUMINAMATH_GPT_maria_bought_9_hardcover_volumes_l1078_107861

def total_volumes (h p : ℕ) : Prop := h + p = 15
def total_cost (h p : ℕ) : Prop := 10 * p + 30 * h = 330

theorem maria_bought_9_hardcover_volumes (h p : ℕ) (h_vol : total_volumes h p) (h_cost : total_cost h p) : h = 9 :=
by
  sorry

end NUMINAMATH_GPT_maria_bought_9_hardcover_volumes_l1078_107861


namespace NUMINAMATH_GPT_playground_area_l1078_107812

theorem playground_area
  (w l : ℕ)
  (h₁ : l = 2 * w + 25)
  (h₂ : 2 * (l + w) = 650) :
  w * l = 22500 := 
sorry

end NUMINAMATH_GPT_playground_area_l1078_107812


namespace NUMINAMATH_GPT_monochromatic_triangle_probability_l1078_107872

noncomputable def probability_monochromatic_triangle : ℚ := sorry

theorem monochromatic_triangle_probability :
  -- Condition: Each of the 6 sides and the 9 diagonals of a regular hexagon are randomly and independently colored red, blue, or green with equal probability.
  -- Proof: The probability that at least one triangle whose vertices are among the vertices of the hexagon has all its sides of the same color is equal to 872/1000.
  probability_monochromatic_triangle = 872 / 1000 :=
sorry

end NUMINAMATH_GPT_monochromatic_triangle_probability_l1078_107872


namespace NUMINAMATH_GPT_max_g_at_8_l1078_107867

noncomputable def g : ℝ → ℝ :=
  sorry -- We define g here abstractly, with nonnegative coefficients

axiom g_nonneg_coeffs : ∀ x, 0 ≤ g x
axiom g_at_4 : g 4 = 16
axiom g_at_16 : g 16 = 256

theorem max_g_at_8 : g 8 ≤ 64 :=
by sorry

end NUMINAMATH_GPT_max_g_at_8_l1078_107867


namespace NUMINAMATH_GPT_ratio_smaller_triangle_to_trapezoid_area_l1078_107831

theorem ratio_smaller_triangle_to_trapezoid_area (a b : ℕ) (sqrt_three : ℝ) 
  (h_a : a = 10) (h_b : b = 2) (h_sqrt_three : sqrt_three = Real.sqrt 3) :
  ( ( (sqrt_three / 4 * (b ^ 2)) / 
      ( (sqrt_three / 4 * (a ^ 2)) - 
         (sqrt_three / 4 * (b ^ 2)))) = 1 / 24 ) := 
by
  -- conditions from the problem
  have h1: a = 10 := by exact h_a
  have h2: b = 2 := by exact h_b
  have h3: sqrt_three = Real.sqrt 3 := by exact h_sqrt_three
  sorry

end NUMINAMATH_GPT_ratio_smaller_triangle_to_trapezoid_area_l1078_107831


namespace NUMINAMATH_GPT_Tabitha_age_proof_l1078_107838

variable (Tabitha_age current_hair_colors: ℕ)
variable (Adds_new_color_per_year: ℕ)
variable (initial_hair_colors: ℕ)
variable (years_passed: ℕ)

theorem Tabitha_age_proof (h1: Adds_new_color_per_year = 1)
                          (h2: initial_hair_colors = 2)
                          (h3: ∀ years_passed, Tabitha_age  = 15 + years_passed)
                          (h4: Adds_new_color_per_year  = 1 )
                          (h5: current_hair_colors =  8 - 3)
                          (h6: current_hair_colors  =  initial_hair_colors + 3)
                          : Tabitha_age = 18 := 
by {
  sorry  -- Proof omitted
}

end NUMINAMATH_GPT_Tabitha_age_proof_l1078_107838


namespace NUMINAMATH_GPT_marbles_in_jar_l1078_107846

theorem marbles_in_jar (M : ℕ) (h1 : ∀ n : ℕ, n = 20 → ∀ m : ℕ, m = M / n → ∀ a b : ℕ, a = n + 2 → b = m - 1 → ∀ k : ℕ, k = M / a → k = b) : M = 220 :=
by 
  sorry

end NUMINAMATH_GPT_marbles_in_jar_l1078_107846


namespace NUMINAMATH_GPT_percentage_difference_is_20_l1078_107837

/-
Given:
Height of sunflowers from Packet A = 192 inches
Height of sunflowers from Packet B = 160 inches

Show:
Percentage difference in height between Packet A and Packet B is 20%.
-/

-- Definitions of heights
def height_packet_A : ℤ := 192
def height_packet_B : ℤ := 160

-- Definition of percentage difference formula
def percentage_difference (hA hB : ℤ) : ℤ := ((hA - hB) * 100) / hB

-- Theorem statement
theorem percentage_difference_is_20 :
  percentage_difference height_packet_A height_packet_B = 20 :=
sorry

end NUMINAMATH_GPT_percentage_difference_is_20_l1078_107837


namespace NUMINAMATH_GPT_simple_interest_fraction_l1078_107820

theorem simple_interest_fraction (P : ℝ) (R : ℝ) (T : ℝ) (SI : ℝ) (F : ℝ)
  (h1 : R = 5)
  (h2 : T = 4)
  (h3 : SI = (P * R * T) / 100)
  (h4 : SI = F * P) :
  F = 1/5 :=
by
  sorry

end NUMINAMATH_GPT_simple_interest_fraction_l1078_107820


namespace NUMINAMATH_GPT_gain_percentage_second_book_l1078_107856

theorem gain_percentage_second_book (CP1 CP2 SP1 SP2 : ℝ)
  (h1 : CP1 = 350) 
  (h2 : CP1 + CP2 = 600)
  (h3 : SP1 = CP1 - (0.15 * CP1))
  (h4 : SP1 = SP2) :
  SP2 = CP2 + (19 / 100 * CP2) :=
by
  sorry

end NUMINAMATH_GPT_gain_percentage_second_book_l1078_107856


namespace NUMINAMATH_GPT_vector_perpendicular_to_plane_l1078_107817

theorem vector_perpendicular_to_plane
  (a b c d : ℝ)
  (x1 y1 z1 x2 y2 z2 : ℝ)
  (h1 : a * x1 + b * y1 + c * z1 + d = 0)
  (h2 : a * x2 + b * y2 + c * z2 + d = 0) :
  a * (x1 - x2) + b * (y1 - y2) + c * (z1 - z2) = 0 :=
sorry

end NUMINAMATH_GPT_vector_perpendicular_to_plane_l1078_107817


namespace NUMINAMATH_GPT_rectangle_area_increase_l1078_107885

theorem rectangle_area_increase
  (l w : ℝ)
  (h₀ : l > 0) -- original length is positive
  (h₁ : w > 0) -- original width is positive
  (length_increase : l' = 1.3 * l) -- new length after increase
  (width_increase : w' = 1.15 * w) -- new width after increase
  (new_area : A' = l' * w') -- new area after increase
  (original_area : A = l * w) -- original area
  :
  ((A' / A) * 100 - 100) = 49.5 := by
  sorry

end NUMINAMATH_GPT_rectangle_area_increase_l1078_107885


namespace NUMINAMATH_GPT_eggs_left_over_l1078_107833

theorem eggs_left_over (David_eggs Ella_eggs Fiona_eggs : ℕ)
  (hD : David_eggs = 45)
  (hE : Ella_eggs = 58)
  (hF : Fiona_eggs = 29) :
  (David_eggs + Ella_eggs + Fiona_eggs) % 10 = 2 :=
by
  sorry

end NUMINAMATH_GPT_eggs_left_over_l1078_107833


namespace NUMINAMATH_GPT_sin_pi_over_4_plus_alpha_l1078_107800

open Real

theorem sin_pi_over_4_plus_alpha
  (α : ℝ)
  (hα : 0 < α ∧ α < π)
  (h_tan : tan (α - π / 4) = 1 / 3) :
  sin (π / 4 + α) = 3 * sqrt 10 / 10 :=
sorry

end NUMINAMATH_GPT_sin_pi_over_4_plus_alpha_l1078_107800


namespace NUMINAMATH_GPT_smallest_positive_period_l1078_107899

theorem smallest_positive_period :
  ∀ (x : ℝ), 5 * Real.sin ((π / 6) - (π / 3) * x) = 5 * Real.sin ((π / 6) - (π / 3) * (x + 6)) :=
by
  sorry

end NUMINAMATH_GPT_smallest_positive_period_l1078_107899


namespace NUMINAMATH_GPT_distinct_paper_count_l1078_107836

theorem distinct_paper_count (n : ℕ) :
  let sides := 4  -- 4 rotations and 4 reflections
  let identity_fixed := n^25 
  let rotation_90_fixed := n^7
  let rotation_270_fixed := n^7
  let rotation_180_fixed := n^13
  let reflection_fixed := n^15
  (1 / 8) * (identity_fixed + 4 * reflection_fixed + rotation_180_fixed + 2 * rotation_90_fixed) 
  = (1 / 8) * (n^25 + 4 * n^15 + n^13 + 2 * n^7) :=
  by 
    sorry

end NUMINAMATH_GPT_distinct_paper_count_l1078_107836


namespace NUMINAMATH_GPT_cos_value_given_sin_condition_l1078_107801

open Real

theorem cos_value_given_sin_condition (x : ℝ) (h : sin (x + π / 12) = -1/4) : 
  cos (5 * π / 6 - 2 * x) = -7 / 8 :=
sorry -- Proof steps are omitted.

end NUMINAMATH_GPT_cos_value_given_sin_condition_l1078_107801


namespace NUMINAMATH_GPT_circle_ratio_l1078_107886

theorem circle_ratio (R r : ℝ) (h₁ : R > 0) (h₂ : r > 0) 
                     (h₃ : π * R^2 - π * r^2 = 3 * π * r^2) : R = 2 * r :=
by
  sorry

end NUMINAMATH_GPT_circle_ratio_l1078_107886


namespace NUMINAMATH_GPT_count_arithmetic_progressions_22_1000_l1078_107823

def num_increasing_arithmetic_progressions (n k max_val : ℕ) : ℕ :=
  -- This is a stub for the arithmetic sequence counting function.
  sorry

theorem count_arithmetic_progressions_22_1000 :
  num_increasing_arithmetic_progressions 22 22 1000 = 23312 :=
sorry

end NUMINAMATH_GPT_count_arithmetic_progressions_22_1000_l1078_107823


namespace NUMINAMATH_GPT_rice_containers_l1078_107848

theorem rice_containers (total_weight_pounds : ℚ) (weight_per_container_ounces : ℚ) (pound_to_ounces : ℚ) : 
  total_weight_pounds = 29/4 → 
  weight_per_container_ounces = 29 → 
  pound_to_ounces = 16 → 
  (total_weight_pounds * pound_to_ounces) / weight_per_container_ounces = 4 := 
by
  intros h1 h2 h3 
  rw [h1, h2, h3]
  sorry

end NUMINAMATH_GPT_rice_containers_l1078_107848


namespace NUMINAMATH_GPT_problem_statement_l1078_107843

noncomputable def log_three_four : ℝ := Real.log 4 / Real.log 3
noncomputable def a : ℝ := Real.log (log_three_four) / Real.log (3/4)
noncomputable def b : ℝ := Real.rpow (3/4 : ℝ) 0.5
noncomputable def c : ℝ := Real.rpow (4/3 : ℝ) 0.5

theorem problem_statement : a < b ∧ b < c :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l1078_107843


namespace NUMINAMATH_GPT_pupils_like_burgers_total_l1078_107841

theorem pupils_like_burgers_total (total_pupils pizza_lovers both_lovers : ℕ) :
  total_pupils = 200 →
  pizza_lovers = 125 →
  both_lovers = 40 →
  (pizza_lovers - both_lovers) + (total_pupils - pizza_lovers - both_lovers) + both_lovers = 115 :=
by
  intros h_total h_pizza h_both
  rw [h_total, h_pizza, h_both]
  sorry

end NUMINAMATH_GPT_pupils_like_burgers_total_l1078_107841


namespace NUMINAMATH_GPT_bill_new_win_percentage_l1078_107887

theorem bill_new_win_percentage :
  ∀ (initial_games : ℕ) (initial_win_percentage : ℚ) (additional_games : ℕ) (losses_in_additional_games : ℕ),
  initial_games = 200 →
  initial_win_percentage = 0.63 →
  additional_games = 100 →
  losses_in_additional_games = 43 →
  ((initial_win_percentage * initial_games + (additional_games - losses_in_additional_games)) / (initial_games + additional_games)) * 100 = 61 := 
by
  intros initial_games initial_win_percentage additional_games losses_in_additional_games h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_bill_new_win_percentage_l1078_107887


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l1078_107809

theorem arithmetic_sequence_sum :
  let first_term := 1
  let common_diff := 2
  let last_term := 33
  let n := (last_term + 1) / common_diff
  (n * (first_term + last_term)) / 2 = 289 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l1078_107809


namespace NUMINAMATH_GPT_triangle_properties_l1078_107840

theorem triangle_properties (A B C : ℝ) (a b c : ℝ) 
  (angle_A : A = 30) (angle_B : B = 45) (side_a : a = Real.sqrt 2) :
  b = 2 ∧ (1 / 2) * a * b * Real.sin (105 * Real.pi / 180) = (Real.sqrt 3 + 1) / 2 := by
sorry

end NUMINAMATH_GPT_triangle_properties_l1078_107840


namespace NUMINAMATH_GPT_find_c_plus_d_l1078_107814

noncomputable def g (c d : ℝ) (x : ℝ) : ℝ :=
if x < 3 then c * x + d else 10 - 2 * x

theorem find_c_plus_d (c d : ℝ) (h : ∀ x, g c d (g c d x) = x) : c + d = 4.5 :=
sorry

end NUMINAMATH_GPT_find_c_plus_d_l1078_107814


namespace NUMINAMATH_GPT_rohan_monthly_salary_l1078_107842

theorem rohan_monthly_salary (s : ℝ) 
  (h_food : s * 0.40 = f)
  (h_rent : s * 0.20 = hr) 
  (h_entertainment : s * 0.10 = e)
  (h_conveyance : s * 0.10 = c)
  (h_savings : s * 0.20 = 1000) : 
  s = 5000 := 
sorry

end NUMINAMATH_GPT_rohan_monthly_salary_l1078_107842


namespace NUMINAMATH_GPT_consecutive_even_sum_l1078_107891

theorem consecutive_even_sum : 
  ∃ n : ℕ, 
  (∃ x : ℕ, (∀ i : ℕ, i < n → (2 * i + x = 14 → i = 2) → 
  2 * x + (n - 1) * n = 52) ∧ n = 4) :=
by
  sorry

end NUMINAMATH_GPT_consecutive_even_sum_l1078_107891


namespace NUMINAMATH_GPT_arithmetic_sequence_term_l1078_107847

theorem arithmetic_sequence_term :
  ∀ a : ℕ → ℕ, (a 1 = 1) → (∀ n : ℕ, a (n + 1) - a n = 2) → (a 6 = 11) :=
by
  intros a h1 hrec
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_term_l1078_107847


namespace NUMINAMATH_GPT_todd_money_left_l1078_107889

-- Define the initial amount of money Todd has
def initial_amount : ℕ := 20

-- Define the number of candy bars Todd buys
def number_of_candy_bars : ℕ := 4

-- Define the cost per candy bar
def cost_per_candy_bar : ℕ := 2

-- Define the total cost of the candy bars
def total_cost : ℕ := number_of_candy_bars * cost_per_candy_bar

-- Define the final amount of money Todd has left
def final_amount : ℕ := initial_amount - total_cost

-- The statement to be proven in Lean
theorem todd_money_left : final_amount = 12 := by
  -- The proof is omitted
  sorry

end NUMINAMATH_GPT_todd_money_left_l1078_107889


namespace NUMINAMATH_GPT_max_x_of_conditions_l1078_107894

theorem max_x_of_conditions (x y z : ℝ) (h1 : x + y + z = 6) (h2 : xy + xz + yz = 11) : x ≤ 2 :=
by
  -- Placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_max_x_of_conditions_l1078_107894


namespace NUMINAMATH_GPT_same_terminal_side_angle_l1078_107874

theorem same_terminal_side_angle (k : ℤ) : 
  (∃ k : ℤ, - (π / 6) = 2 * k * π + a) → a = 11 * π / 6 :=
sorry

end NUMINAMATH_GPT_same_terminal_side_angle_l1078_107874


namespace NUMINAMATH_GPT_sum_2004_impossible_sum_2005_possible_l1078_107897

-- Condition Definitions
def is_valid_square (s : ℕ × ℕ × ℕ × ℕ) : Prop :=
  s = (1, 2, 3, 4) ∨ s = (1, 2, 4, 3) ∨ s = (1, 3, 2, 4) ∨ s = (1, 3, 4, 2) ∨ 
  s = (1, 4, 2, 3) ∨ s = (1, 4, 3, 2) ∨ s = (2, 1, 3, 4) ∨ s = (2, 1, 4, 3) ∨ 
  s = (2, 3, 1, 4) ∨ s = (2, 3, 4, 1) ∨ s = (2, 4, 1, 3) ∨ s = (2, 4, 3, 1) ∨ 
  s = (3, 1, 2, 4) ∨ s = (3, 1, 4, 2) ∨ s = (3, 2, 1, 4) ∨ s = (3, 2, 4, 1) ∨ 
  s = (3, 4, 1, 2) ∨ s = (3, 4, 2, 1) ∨ s = (4, 1, 2, 3) ∨ s = (4, 1, 3, 2) ∨ 
  s = (4, 2, 1, 3) ∨ s = (4, 2, 3, 1) ∨ s = (4, 3, 1, 2) ∨ s = (4, 3, 2, 1)

-- Proof Problems
theorem sum_2004_impossible (n : ℕ) (corners : ℕ → ℕ × ℕ × ℕ × ℕ) (h : ∀ i, is_valid_square (corners i)) :
  4 * 2004 ≠ n * 10 := 
sorry

theorem sum_2005_possible (h : ∃ n, ∃ corners : ℕ → ℕ × ℕ × ℕ × ℕ, (∀ i, is_valid_square (corners i)) ∧ 4 * 2005 = n * 10 + 2005) :
  true := 
sorry

end NUMINAMATH_GPT_sum_2004_impossible_sum_2005_possible_l1078_107897


namespace NUMINAMATH_GPT_juanita_spends_more_l1078_107853

-- Define the expenditures
def grant_yearly_expenditure : ℝ := 200.00

def juanita_weekday_expenditure : ℝ := 0.50

def juanita_sunday_expenditure : ℝ := 2.00

def weeks_per_year : ℕ := 52

-- Given conditions translated to Lean
def juanita_weekly_expenditure : ℝ :=
  (juanita_weekday_expenditure * 6) + juanita_sunday_expenditure

def juanita_yearly_expenditure : ℝ :=
  juanita_weekly_expenditure * weeks_per_year

-- The statement we need to prove
theorem juanita_spends_more : (juanita_yearly_expenditure - grant_yearly_expenditure) = 60.00 :=
by
  sorry

end NUMINAMATH_GPT_juanita_spends_more_l1078_107853


namespace NUMINAMATH_GPT_correct_sum_l1078_107804

theorem correct_sum (x y : ℕ) (h1 : x > y) (h2 : x - y = 4) (h3 : x * y = 98) : x + y = 18 := 
by
  sorry

end NUMINAMATH_GPT_correct_sum_l1078_107804


namespace NUMINAMATH_GPT_minimize_y_l1078_107821

theorem minimize_y (a b : ℝ) : 
  ∃ x, x = (a + b) / 2 ∧ ∀ x', ((x' - a)^3 + (x' - b)^3) ≥ ((x - a)^3 + (x - b)^3) :=
sorry

end NUMINAMATH_GPT_minimize_y_l1078_107821


namespace NUMINAMATH_GPT_three_digit_number_parity_count_equal_l1078_107858

/-- Prove the number of three-digit numbers with all digits having the same parity is equal to the number of three-digit numbers where adjacent digits have different parity. -/
theorem three_digit_number_parity_count_equal :
  ∃ (same_parity_count alternating_parity_count : ℕ),
    same_parity_count = alternating_parity_count ∧
    -- Condition for digits of the same parity
    same_parity_count = (4 * 5 * 5) + (5 * 5 * 5) ∧
    -- Condition for alternating parity digits (patterns EOE and OEO)
    alternating_parity_count = (4 * 5 * 5) + (5 * 5 * 5) := by
  sorry

end NUMINAMATH_GPT_three_digit_number_parity_count_equal_l1078_107858


namespace NUMINAMATH_GPT_problem_1_l1078_107862

theorem problem_1 (a : ℝ) : (1 + a * x) * (1 + x) ^ 5 = 1 + 5 * x + 5 * i * x^2 → a = -1 := sorry

end NUMINAMATH_GPT_problem_1_l1078_107862


namespace NUMINAMATH_GPT_max_chocolates_eaten_by_Ben_l1078_107851

-- Define the situation with Ben and Carol sharing chocolates
variable (b c k : ℕ) -- b for Ben, c for Carol, k is the multiplier

-- Define the conditions
def chocolates_shared (b c : ℕ) : Prop := b + c = 30
def carol_eats_multiple (b c k : ℕ) : Prop := c = k * b ∧ k > 0

-- The theorem statement that we want to prove
theorem max_chocolates_eaten_by_Ben 
  (h1 : chocolates_shared b c) 
  (h2 : carol_eats_multiple b c k) : 
  b ≤ 15 := by
  sorry

end NUMINAMATH_GPT_max_chocolates_eaten_by_Ben_l1078_107851


namespace NUMINAMATH_GPT_initial_cookie_count_l1078_107865

variable (cookies_left_after_week : ℕ)
variable (cookies_taken_each_day : ℕ)
variable (total_cookies_taken_in_four_days : ℕ)
variable (initial_cookies : ℕ)
variable (days_per_week : ℕ)

theorem initial_cookie_count :
  cookies_left_after_week = 28 →
  total_cookies_taken_in_four_days = 24 →
  days_per_week = 7 →
  (∀ d (h : d ∈ Finset.range days_per_week), cookies_taken_each_day = 6) →
  initial_cookies = 52 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_initial_cookie_count_l1078_107865


namespace NUMINAMATH_GPT_problem1_problem2_problem3_problem4_l1078_107828

theorem problem1 : -16 - (-12) - 24 + 18 = -10 := 
by
  sorry

theorem problem2 : 0.125 + (1 / 4) + (-9 / 4) + (-0.25) = -2 := 
by
  sorry

theorem problem3 : (-1 / 12 - 1 / 36 + 1 / 6) * (-36) = -2 := 
by
  sorry

theorem problem4 : (-2 + 3) * 3 - (-2)^3 / 4 = 5 := 
by
  sorry

end NUMINAMATH_GPT_problem1_problem2_problem3_problem4_l1078_107828


namespace NUMINAMATH_GPT_probability_two_white_balls_is_4_over_15_l1078_107839

-- Define the conditions of the problem
def total_balls : ℕ := 15
def white_balls_initial : ℕ := 8
def black_balls : ℕ := 7
def balls_drawn : ℕ := 2 -- Note: Even though not explicitly required, it's part of the context

-- Calculate the probability of drawing two white balls without replacement
noncomputable def probability_two_white_balls : ℚ :=
  (white_balls_initial / total_balls) * ((white_balls_initial - 1) / (total_balls - 1))

-- The theorem to prove
theorem probability_two_white_balls_is_4_over_15 :
  probability_two_white_balls = 4 / 15 := by
  sorry

end NUMINAMATH_GPT_probability_two_white_balls_is_4_over_15_l1078_107839


namespace NUMINAMATH_GPT_student_weekly_allowance_l1078_107857

theorem student_weekly_allowance (A : ℝ) :
  (3 / 4) * (1 / 3) * ((2 / 5) * A + 4) - 2 = 0 ↔ A = 100/3 := sorry

end NUMINAMATH_GPT_student_weekly_allowance_l1078_107857


namespace NUMINAMATH_GPT_broth_for_third_l1078_107829

theorem broth_for_third (b : ℚ) (h : b = 6 + 3/4) : b / 3 = 2 + 1/4 := by
  sorry

end NUMINAMATH_GPT_broth_for_third_l1078_107829


namespace NUMINAMATH_GPT_general_term_correct_l1078_107896

-- Define the sequence a_n
def seq (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧ ∀ n, a (n + 1) = 2 * a n + 2^n

-- Define the general term formula for the sequence a_n
def general_term (a : ℕ → ℕ) : Prop :=
  ∀ n, a n = n * 2^(n - 1)

-- Theorem statement: the general term formula holds for the sequence a_n
theorem general_term_correct (a : ℕ → ℕ) (h_seq : seq a) : general_term a :=
by
  sorry

end NUMINAMATH_GPT_general_term_correct_l1078_107896


namespace NUMINAMATH_GPT_sum_of_squares_base_6_l1078_107852

def to_base (n b : ℕ) : ℕ := sorry

theorem sum_of_squares_base_6 :
  let squares := (List.range 12).map (λ x => x.succ ^ 2);
  let squares_base6 := squares.map (λ x => to_base x 6);
  (squares_base6.sum) = to_base 10515 6 :=
by sorry

end NUMINAMATH_GPT_sum_of_squares_base_6_l1078_107852


namespace NUMINAMATH_GPT_hyperbola_asymptotes_l1078_107819

def hyperbola (x y : ℝ) : Prop := (x^2 / 8) - (y^2 / 2) = 1

theorem hyperbola_asymptotes (x y : ℝ) :
  hyperbola x y → (y = (1/2) * x ∨ y = - (1/2) * x) :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_asymptotes_l1078_107819


namespace NUMINAMATH_GPT_matrix_det_is_zero_l1078_107827

noncomputable def matrixDetProblem (a b : ℝ) : ℝ :=
  Matrix.det ![
    ![1, Real.cos (a - b), Real.sin a],
    ![Real.cos (a - b), 1, Real.sin b],
    ![Real.sin a, Real.sin b, 1]
  ]

theorem matrix_det_is_zero (a b : ℝ) : matrixDetProblem a b = 0 :=
  sorry

end NUMINAMATH_GPT_matrix_det_is_zero_l1078_107827


namespace NUMINAMATH_GPT_min_sum_dimensions_l1078_107892

theorem min_sum_dimensions (a b c : ℕ) (h : a * b * c = 2310) : a + b + c ≥ 52 :=
sorry

end NUMINAMATH_GPT_min_sum_dimensions_l1078_107892


namespace NUMINAMATH_GPT_original_radius_l1078_107854

theorem original_radius (r : Real) (h : Real) (z : Real) 
  (V : Real) (Vh : Real) (Vr : Real) :
  h = 3 → 
  V = π * r^2 * h → 
  Vh = π * r^2 * (h + 3) → 
  Vr = π * (r + 3)^2 * h → 
  Vh - V = z → 
  Vr - V = z →
  r = 3 + 3 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_original_radius_l1078_107854


namespace NUMINAMATH_GPT_fraction_simplifies_to_two_l1078_107805

theorem fraction_simplifies_to_two :
  (2 + 4 + 6 + 8 + 10 + 12 + 14 + 16 + 18 + 20) / (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10) = 2 := by
  sorry

end NUMINAMATH_GPT_fraction_simplifies_to_two_l1078_107805


namespace NUMINAMATH_GPT_sqrt_square_identity_l1078_107884

-- Define the concept of square root
noncomputable def sqrt (x : ℝ) : ℝ := Real.sqrt x

-- Problem statement: prove (sqrt 12321)^2 = 12321
theorem sqrt_square_identity (x : ℝ) : (sqrt x) ^ 2 = x := by
  sorry

-- Specific instance for the given number
example : (sqrt 12321) ^ 2 = 12321 := sqrt_square_identity 12321

end NUMINAMATH_GPT_sqrt_square_identity_l1078_107884


namespace NUMINAMATH_GPT_race_distance_l1078_107883

theorem race_distance {d x y z : ℝ} :
  (d / x = (d - 25) / y) →
  (d / y = (d - 15) / z) →
  (d / x = (d - 37) / z) →
  d = 125 :=
by
  intros h1 h2 h3
  -- Insert proof here
  sorry

end NUMINAMATH_GPT_race_distance_l1078_107883


namespace NUMINAMATH_GPT_find_second_number_l1078_107881

theorem find_second_number (x y z : ℚ)
  (h1 : x + y + z = 120)
  (h2 : x / y = 3 / 4)
  (h3 : y / z = 4 / 7) :
  y = 240 / 7 :=
by sorry

end NUMINAMATH_GPT_find_second_number_l1078_107881


namespace NUMINAMATH_GPT_number_of_roof_tiles_l1078_107880

def land_cost : ℝ := 50
def bricks_cost_per_1000 : ℝ := 100
def roof_tile_cost : ℝ := 10
def land_required : ℝ := 2000
def bricks_required : ℝ := 10000
def total_construction_cost : ℝ := 106000

theorem number_of_roof_tiles :
  let land_total := land_cost * land_required
  let bricks_total := (bricks_required / 1000) * bricks_cost_per_1000
  let remaining_cost := total_construction_cost - (land_total + bricks_total)
  let roof_tiles := remaining_cost / roof_tile_cost
  roof_tiles = 500 := by
  sorry

end NUMINAMATH_GPT_number_of_roof_tiles_l1078_107880


namespace NUMINAMATH_GPT_permutation_sum_inequality_l1078_107815

noncomputable def permutations (n : ℕ) : List (List ℚ) :=
  List.permutations ((List.range (n+1)).map (fun i => if i = 0 then (1 : ℚ) else (1 : ℚ) / i))

theorem permutation_sum_inequality (n : ℕ) (a b : Fin n → ℚ)
  (ha : ∃ p : List ℚ, p ∈ permutations n ∧ ∀ i, a i = p.get? i) 
  (hb : ∃ q : List ℚ, q ∈ permutations n ∧ ∀ i, b i = q.get? i)
  (h_sum : ∀ i j : Fin n, i ≤ j → a i + b i ≥ a j + b j) 
  (m : Fin n) :
  a m + b m ≤ 4 / (m + 1) :=
sorry

end NUMINAMATH_GPT_permutation_sum_inequality_l1078_107815


namespace NUMINAMATH_GPT_min_value_expression_l1078_107875

theorem min_value_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
    4.5 ≤ (8 * z) / (3 * x + 2 * y) + (8 * x) / (2 * y + 3 * z) + y / (x + z) :=
by
  sorry

end NUMINAMATH_GPT_min_value_expression_l1078_107875


namespace NUMINAMATH_GPT_least_possible_value_of_a_plus_b_l1078_107818

theorem least_possible_value_of_a_plus_b : ∃ (a b : ℕ), a > 0 ∧ b > 0 ∧
  Nat.gcd (a + b) 330 = 1 ∧
  b ∣ a^a ∧ 
  ∀ k : ℕ, b^3 ∣ a^a → (k ∣ a → k = 1) ∧
  a + b = 392 :=
by
  sorry

end NUMINAMATH_GPT_least_possible_value_of_a_plus_b_l1078_107818


namespace NUMINAMATH_GPT_cody_spent_19_dollars_l1078_107893

-- Given conditions
def initial_money : ℕ := 45
def birthday_gift : ℕ := 9
def remaining_money : ℕ := 35

-- Problem: Prove that the amount of money spent on the game is $19.
theorem cody_spent_19_dollars :
  (initial_money + birthday_gift - remaining_money) = 19 :=
by sorry

end NUMINAMATH_GPT_cody_spent_19_dollars_l1078_107893


namespace NUMINAMATH_GPT_correct_option_D_l1078_107832

noncomputable def total_students := 40
noncomputable def male_students := 25
noncomputable def female_students := 15
noncomputable def class_president := 1
noncomputable def prob_class_president := class_president / total_students
noncomputable def prob_class_president_from_females := 0

theorem correct_option_D
  (h1 : total_students = 40)
  (h2 : male_students = 25)
  (h3 : female_students = 15)
  (h4 : class_president = 1) :
  prob_class_president = 1 / 40 ∧ prob_class_president_from_females = 0 := 
by
  sorry

end NUMINAMATH_GPT_correct_option_D_l1078_107832


namespace NUMINAMATH_GPT_remainder_x_101_div_x2_plus1_x_plus1_l1078_107822

theorem remainder_x_101_div_x2_plus1_x_plus1 : 
  (x^101) % ((x^2 + 1) * (x + 1)) = x :=
by
  sorry

end NUMINAMATH_GPT_remainder_x_101_div_x2_plus1_x_plus1_l1078_107822


namespace NUMINAMATH_GPT_coconut_grove_l1078_107849

theorem coconut_grove (x : ℕ) :
  (60 * (x + 1) + 120 * x + 180 * (x - 1)) = 300 * x → x = 2 :=
by
  intro h
  -- We can leave the proof part to prove this later.
  sorry

end NUMINAMATH_GPT_coconut_grove_l1078_107849


namespace NUMINAMATH_GPT_correct_statement_l1078_107811

theorem correct_statement (a b : ℚ) :
  (|a| = b → a = b) ∧ (|a| > |b| → a > b) ∧ (|a| > b → |a| > |b|) ∧ (|a| = b → a^2 = (-b)^2) ↔ 
  (true ∧ false ∧ false ∧ true) :=
by
  sorry

end NUMINAMATH_GPT_correct_statement_l1078_107811


namespace NUMINAMATH_GPT_sum_of_angles_l1078_107878

theorem sum_of_angles (θ₁ θ₂ θ₃ θ₄ : ℝ)
  (h₁ : θ₁ = 67.5) (h₂ : θ₂ = 157.5) (h₃ : θ₃ = 247.5) (h₄ : θ₄ = 337.5) :
  θ₁ + θ₂ + θ₃ + θ₄ = 810 :=
by
  -- These parameters are used only to align with provided conditions
  let r₁ := 1
  let r₂ := r₁
  let r₃ := r₁
  let r₄ := r₁
  have z₁ := r₁ * (Complex.cos θ₁ + Complex.sin θ₁ * Complex.I)
  have z₂ := r₂ * (Complex.cos θ₂ + Complex.sin θ₂ * Complex.I)
  have z₃ := r₃ * (Complex.cos θ₃ + Complex.sin θ₃ * Complex.I)
  have z₄ := r₄ * (Complex.cos θ₄ + Complex.sin θ₄ * Complex.I)
  sorry

end NUMINAMATH_GPT_sum_of_angles_l1078_107878


namespace NUMINAMATH_GPT_upstream_speed_is_8_l1078_107850

-- Definitions of given conditions
def downstream_speed : ℝ := 13
def stream_speed : ℝ := 2.5
def man's_upstream_speed : ℝ := downstream_speed - 2 * stream_speed

-- Theorem to prove
theorem upstream_speed_is_8 : man's_upstream_speed = 8 :=
by
  rw [man's_upstream_speed, downstream_speed, stream_speed]
  sorry

end NUMINAMATH_GPT_upstream_speed_is_8_l1078_107850


namespace NUMINAMATH_GPT_sum_of_angles_in_segments_outside_pentagon_l1078_107835

theorem sum_of_angles_in_segments_outside_pentagon 
  (α β γ δ ε : ℝ) 
  (hα : α = 0.5 * (360 - arc_BCDE))
  (hβ : β = 0.5 * (360 - arc_CDEA))
  (hγ : γ = 0.5 * (360 - arc_DEAB))
  (hδ : δ = 0.5 * (360 - arc_EABC))
  (hε : ε = 0.5 * (360 - arc_ABCD)) 
  (arc_BCDE arc_CDEA arc_DEAB arc_EABC arc_ABCD : ℝ) :
  α + β + γ + δ + ε = 720 := 
by 
  sorry

end NUMINAMATH_GPT_sum_of_angles_in_segments_outside_pentagon_l1078_107835


namespace NUMINAMATH_GPT_max_value_of_expression_l1078_107888

variable (a b c d : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c) (h₄ : 0 < d) (h₅ : a + b + c + d = 3)

theorem max_value_of_expression :
  3 * a^2 * b^3 * c * d^2 ≤ 177147 / 40353607 :=
sorry

end NUMINAMATH_GPT_max_value_of_expression_l1078_107888


namespace NUMINAMATH_GPT_wire_around_field_l1078_107844

theorem wire_around_field (A L : ℕ) (hA : A = 69696) (hL : L = 15840) : L / (4 * (Nat.sqrt A)) = 15 :=
by
  sorry

end NUMINAMATH_GPT_wire_around_field_l1078_107844
