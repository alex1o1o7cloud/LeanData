import Mathlib

namespace triangle_angle_C_l1069_106947

open Real

theorem triangle_angle_C (b c : ℝ) (B C : ℝ) (hb : b = sqrt 2) (hc : c = 1) (hB : B = 45) : C = 30 :=
sorry

end triangle_angle_C_l1069_106947


namespace triangle_altitude_length_l1069_106927

-- Define the problem
theorem triangle_altitude_length (l w h : ℝ) (hl : l = 2 * w) 
  (h_triangle_area : 0.5 * l * h = 0.5 * (l * w)) : h = w := 
by 
  -- Use the provided conditions and the equation setup to continue the proof
  sorry

end triangle_altitude_length_l1069_106927


namespace trig_identity_proof_l1069_106930

theorem trig_identity_proof :
  let sin := Real.sin
  let cos := Real.cos
  let deg_to_rad := fun θ : ℝ => θ * Real.pi / 180
  sin (deg_to_rad 30) * sin (deg_to_rad 75) - sin (deg_to_rad 60) * cos (deg_to_rad 105) = Real.sqrt 2 / 2 :=
by
  sorry

end trig_identity_proof_l1069_106930


namespace frequency_of_group_5_l1069_106900

/-- Let the total number of data points be 50, number of data points in groups 1, 2, 3, and 4 be
  2, 8, 15, and 5 respectively. Prove that the frequency of group 5 is 0.4. -/
theorem frequency_of_group_5 :
  let total_data_points := 50
  let group1_data_points := 2
  let group2_data_points := 8
  let group3_data_points := 15
  let group4_data_points := 5
  let group5_data_points := total_data_points - group1_data_points - group2_data_points - group3_data_points - group4_data_points
  let frequency_group5 := (group5_data_points : ℝ) / total_data_points
  frequency_group5 = 0.4 := 
by
  sorry

end frequency_of_group_5_l1069_106900


namespace probability_neither_square_nor_cube_l1069_106918

theorem probability_neither_square_nor_cube :
  let count_squares := 14
  let count_cubes := 5
  let overlap := 2
  let total_range := 200
  let neither_count := total_range - (count_squares + count_cubes - overlap)
  let probability := (neither_count : ℚ) / total_range
  probability = 183 / 200 :=
by {
  sorry
}

end probability_neither_square_nor_cube_l1069_106918


namespace circle_radius_l1069_106928

noncomputable def circle_problem (rD rE : ℝ) (m n : ℝ) :=
  rD = 2 * rE ∧
  rD = (Real.sqrt m) - n ∧
  m ≥ 0 ∧ n ≥ 0

theorem circle_radius (rE rD : ℝ) (m n : ℝ) (h : circle_problem rD rE m n) :
  m + n = 5.76 :=
by
  sorry

end circle_radius_l1069_106928


namespace ax5_by5_eq_neg1065_l1069_106906

theorem ax5_by5_eq_neg1065 (a b x y : ℝ) 
  (h1 : a*x + b*y = 5) 
  (h2 : a*x^2 + b*y^2 = 9) 
  (h3 : a*x^3 + b*y^3 = 20) 
  (h4 : a*x^4 + b*y^4 = 48) 
  (h5 : x + y = -15) 
  (h6 : x^2 + y^2 = 55) : 
  a * x^5 + b * y^5 = -1065 := 
sorry

end ax5_by5_eq_neg1065_l1069_106906


namespace remainder_when_A_divided_by_9_l1069_106912

theorem remainder_when_A_divided_by_9 (A B : ℕ) (h1 : A = B * 9 + 13) : A % 9 = 4 := 
by {
  sorry
}

end remainder_when_A_divided_by_9_l1069_106912


namespace triangle_area_right_angled_l1069_106988

theorem triangle_area_right_angled (a : ℝ) (h₁ : 0 < a) (h₂ : a < 24) :
  let b := 24
  let c := 48 - a
  (a^2 + b^2 = c^2) → (1/2) * a * b = 216 :=
by
  sorry

end triangle_area_right_angled_l1069_106988


namespace total_amount_collected_l1069_106958

theorem total_amount_collected (h1 : ∀ (P_I P_II : ℕ), P_I * 50 = P_II) 
                               (h2 : ∀ (F_I F_II : ℕ), F_I = 3 * F_II) 
                               (h3 : ∀ (P_II F_II : ℕ), P_II * F_II = 1250) : 
                               ∃ (Total : ℕ), Total = 1325 :=
by
  sorry

end total_amount_collected_l1069_106958


namespace expression_value_l1069_106926

theorem expression_value :
  (100 - (3000 - 300) + (3000 - (300 - 100)) = 200) := by
  sorry

end expression_value_l1069_106926


namespace poles_intersection_l1069_106980

-- Define the known heights and distances
def heightOfIntersection (d h1 h2 x : ℝ) : ℝ := sorry

theorem poles_intersection :
  heightOfIntersection 120 30 60 40 = 20 := by
  sorry

end poles_intersection_l1069_106980


namespace diagonal_of_rectangle_l1069_106982

noncomputable def L : ℝ := 40 * Real.sqrt 3
noncomputable def W : ℝ := 30 * Real.sqrt 3
noncomputable def d : ℝ := Real.sqrt (L^2 + W^2)

theorem diagonal_of_rectangle :
  d = 50 * Real.sqrt 3 :=
by sorry

end diagonal_of_rectangle_l1069_106982


namespace totalFourOfAKindCombinations_l1069_106964

noncomputable def numberOfFourOfAKindCombinations : Nat :=
  13 * 48

theorem totalFourOfAKindCombinations : numberOfFourOfAKindCombinations = 624 := by
  sorry

end totalFourOfAKindCombinations_l1069_106964


namespace calc_product_l1069_106904

def x : ℝ := 150.15
def y : ℝ := 12.01
def z : ℝ := 1500.15
def w : ℝ := 12

theorem calc_product :
  x * y * z * w = 32467532.8227 :=
by
  sorry

end calc_product_l1069_106904


namespace abs_ab_eq_2_sqrt_111_l1069_106993

theorem abs_ab_eq_2_sqrt_111 (a b : ℝ) (h1 : b^2 - a^2 = 25) (h2 : a^2 + b^2 = 49) : |a * b| = 2 * Real.sqrt 111 := sorry

end abs_ab_eq_2_sqrt_111_l1069_106993


namespace value_of_a_sq_sub_b_sq_l1069_106948

theorem value_of_a_sq_sub_b_sq (a b : ℝ) (h1 : a + b = 20) (h2 : a - b = 4) : a^2 - b^2 = 80 :=
by
  sorry

end value_of_a_sq_sub_b_sq_l1069_106948


namespace john_investment_years_l1069_106915

theorem john_investment_years (P FVt : ℝ) (r1 r2 : ℝ) (n1 t : ℝ) :
  P = 2000 →
  r1 = 0.08 →
  r2 = 0.12 →
  n1 = 2 →
  FVt = 6620 →
  P * (1 + r1)^n1 * (1 + r2)^(t - n1) = FVt →
  t = 11 :=
by
  sorry

end john_investment_years_l1069_106915


namespace scale_down_multiplication_l1069_106908

theorem scale_down_multiplication (h : 14.97 * 46 = 688.62) : 1.497 * 4.6 = 6.8862 :=
by
  -- here we assume the necessary steps to justify the statement.
  sorry

end scale_down_multiplication_l1069_106908


namespace negation_proposition_l1069_106969

theorem negation_proposition {x : ℝ} : ¬ (x^2 - x + 3 > 0) ↔ x^2 - x + 3 ≤ 0 := sorry

end negation_proposition_l1069_106969


namespace ratio_w_y_l1069_106921

theorem ratio_w_y (w x y z : ℚ) 
  (h1 : w / x = 5 / 2) 
  (h2 : y / z = 5 / 3) 
  (h3 : z / x = 1 / 6) : 
  w / y = 9 := 
by 
  sorry

end ratio_w_y_l1069_106921


namespace total_flour_required_l1069_106989

-- Definitions specified based on the given conditions
def flour_already_put_in : ℕ := 10
def flour_needed : ℕ := 2

-- Lean 4 statement to prove the total amount of flour required by the recipe
theorem total_flour_required : (flour_already_put_in + flour_needed) = 12 :=
by
  sorry

end total_flour_required_l1069_106989


namespace smallest_square_side_length_l1069_106951

theorem smallest_square_side_length (s : ℕ) :
  (∃ s, s > 3 ∧ s ≤ 4 ∧ (s - 1) * (s - 1) = 5) ↔ s = 4 := by
  sorry

end smallest_square_side_length_l1069_106951


namespace incircle_radius_of_right_triangle_l1069_106955

noncomputable def radius_of_incircle (a b c : ℝ) : ℝ := (a + b - c) / 2

theorem incircle_radius_of_right_triangle
  (a : ℝ) (b_proj_hypotenuse : ℝ) (r : ℝ) :
  a = 15 ∧ b_proj_hypotenuse = 16 ∧ r = 5 :=
by
  sorry

end incircle_radius_of_right_triangle_l1069_106955


namespace number_of_different_pairs_l1069_106950

theorem number_of_different_pairs :
  let mystery := 4
  let fantasy := 4
  let science_fiction := 4
  (mystery * fantasy) + (mystery * science_fiction) + (fantasy * science_fiction) = 48 :=
by
  let mystery := 4
  let fantasy := 4
  let science_fiction := 4
  show (mystery * fantasy) + (mystery * science_fiction) + (fantasy * science_fiction) = 48
  sorry

end number_of_different_pairs_l1069_106950


namespace total_hexagons_calculation_l1069_106924

-- Define the conditions
-- Regular hexagon side length
def hexagon_side_length : ℕ := 3

-- Number of smaller triangles
def small_triangle_count : ℕ := 54

-- Small triangle side length
def small_triangle_side_length : ℕ := 1

-- Define the total number of hexagons calculated
def total_hexagons : ℕ := 36

-- Theorem stating that given the conditions, the total number of hexagons is 36
theorem total_hexagons_calculation :
    (hexagon_side_length = 3) →
    (small_triangle_count = 54) →
    (small_triangle_side_length = 1) →
    total_hexagons = 36 :=
    by
    intros
    sorry

end total_hexagons_calculation_l1069_106924


namespace difference_between_wins_and_losses_l1069_106991

noncomputable def number_of_wins (n m : ℕ) : Prop :=
  0 ≤ n ∧ 0 ≤ m ∧ n + m ≤ 42 ∧ n + (42 - n - m) / 2 = 30 / 1

theorem difference_between_wins_and_losses (n m : ℕ) (h : number_of_wins n m) : n - m = 18 :=
sorry

end difference_between_wins_and_losses_l1069_106991


namespace base_7_to_10_of_23456_l1069_106987

theorem base_7_to_10_of_23456 : 
  (2 * 7 ^ 4 + 3 * 7 ^ 3 + 4 * 7 ^ 2 + 5 * 7 ^ 1 + 6 * 7 ^ 0) = 6068 :=
by sorry

end base_7_to_10_of_23456_l1069_106987


namespace initial_customers_count_l1069_106953

theorem initial_customers_count (left_count remaining_people_per_table tables remaining_customers : ℕ) 
  (h1 : left_count = 14) 
  (h2 : remaining_people_per_table = 4) 
  (h3 : tables = 2) 
  (h4 : remaining_customers = tables * remaining_people_per_table) 
  : n = 22 :=
  sorry

end initial_customers_count_l1069_106953


namespace scatter_plot_exists_l1069_106939

theorem scatter_plot_exists (sample_data : List (ℝ × ℝ)) :
  ∃ plot : List (ℝ × ℝ), plot = sample_data :=
by
  sorry

end scatter_plot_exists_l1069_106939


namespace printer_x_time_l1069_106995

-- Define the basic parameters given in the problem
def job_time_printer_y := 12
def job_time_printer_z := 8
def ratio := 10 / 3

-- Work rates of the printers
def work_rate_y := 1 / job_time_printer_y
def work_rate_z := 1 / job_time_printer_z

-- Combined work rate and total time for printers Y and Z
def combined_work_rate_y_z := work_rate_y + work_rate_z
def time_printers_y_z := 1 / combined_work_rate_y_z

-- Given ratio relation
def time_printer_x := ratio * time_printers_y_z

-- Mathematical statement to prove: time it takes for printer X to do the job alone
theorem printer_x_time : time_printer_x = 16 := by
  sorry

end printer_x_time_l1069_106995


namespace find_digit_A_l1069_106925

theorem find_digit_A (A M C : ℕ) (h1 : A < 10) (h2 : M < 10) (h3 : C < 10) (h4 : (100 * A + 10 * M + C) * (A + M + C) = 2008) : 
  A = 2 :=
sorry

end find_digit_A_l1069_106925


namespace side_length_of_octagon_l1069_106994

-- Define the conditions
def is_octagon (n : ℕ) := n = 8
def perimeter (p : ℕ) := p = 72

-- Define the problem statement
theorem side_length_of_octagon (n p l : ℕ) 
  (h1 : is_octagon n) 
  (h2 : perimeter p) 
  (h3 : p / n = l) :
  l = 9 := 
  sorry

end side_length_of_octagon_l1069_106994


namespace bret_total_spend_l1069_106974

/-- Bret and his team are working late along with another team of 4 co-workers.
He decides to order dinner for everyone. -/

def team_A : ℕ := 4 -- Bret’s team
def team_B : ℕ := 4 -- Other team

def main_meal_cost : ℕ := 12
def team_A_appetizers_cost : ℕ := 2 * 6  -- Two appetizers at $6 each
def team_B_appetizers_cost : ℕ := 3 * 8  -- Three appetizers at $8 each
def sharing_plates_cost : ℕ := 4 * 10    -- Four sharing plates at $10 each

def tip_percentage : ℝ := 0.20           -- Tip is 20%
def rush_order_fee : ℕ := 5              -- Rush order fee is $5
def sales_tax : ℝ := 0.07                -- Local sales tax is 7%

def total_cost_without_tip_and_tax : ℕ :=
  team_A * main_meal_cost + team_B * main_meal_cost + team_A_appetizers_cost +
  team_B_appetizers_cost + sharing_plates_cost

def total_cost_with_tip : ℝ :=
  total_cost_without_tip_and_tax + 
  (tip_percentage * total_cost_without_tip_and_tax)

def total_cost_before_tax : ℝ :=
  total_cost_with_tip + rush_order_fee

def final_total_cost : ℝ :=
  total_cost_before_tax + (sales_tax * total_cost_with_tip)


theorem bret_total_spend : final_total_cost = 225.85 := by
  sorry

end bret_total_spend_l1069_106974


namespace lally_internet_days_l1069_106999

-- Definitions based on the conditions
def cost_per_day : ℝ := 0.5
def debt_limit : ℝ := 5
def initial_payment : ℝ := 7
def initial_balance : ℝ := 0

-- Proof problem statement
theorem lally_internet_days : ∀ (d : ℕ), 
  (initial_balance + initial_payment - cost_per_day * d ≤ debt_limit) -> (d = 14) :=
sorry

end lally_internet_days_l1069_106999


namespace seating_arrangement_l1069_106986

theorem seating_arrangement (x y : ℕ) (h1 : 9 * x + 7 * y = 61) : x = 6 :=
by 
  sorry

end seating_arrangement_l1069_106986


namespace find_c_l1069_106976

theorem find_c (a c : ℤ) (h1 : 3 * a + 2 = 2) (h2 : c - a = 3) : c = 3 := by
  sorry

end find_c_l1069_106976


namespace part1_part2_l1069_106922

variable (a m : ℝ)

def f (x : ℝ) : ℝ := 2 * |x - 1| - a

theorem part1 (h : ∃ x, f a x - 2 * |x - 7| ≤ 0) : a ≥ -12 :=
sorry

theorem part2 (h : ∀ x, f 1 x + |x + 7| ≥ m) : m ≤ 7 :=
sorry

end part1_part2_l1069_106922


namespace fraction_meaningful_iff_l1069_106998

theorem fraction_meaningful_iff (x : ℝ) : (∃ y, y = 1 / (x - 1)) ↔ x ≠ 1 :=
by 
  sorry

end fraction_meaningful_iff_l1069_106998


namespace solve_quadratic_equation_l1069_106910

theorem solve_quadratic_equation (x : ℝ) :
  x^2 - 2 * x - 8 = 0 ↔ (x = 4 ∨ x = -2) :=
by sorry

end solve_quadratic_equation_l1069_106910


namespace factorization_identity_l1069_106916

theorem factorization_identity (m : ℝ) : m^3 - m = m * (m + 1) * (m - 1) :=
by
  sorry

end factorization_identity_l1069_106916


namespace geo_seq_sum_monotone_l1069_106956

theorem geo_seq_sum_monotone (q a1 : ℝ) (n : ℕ) (S : ℕ → ℝ) :
  (∀ n, S (n + 1) > S n) ↔ (a1 > 0 ∧ q > 0) :=
sorry -- Proof of the theorem (omitted)

end geo_seq_sum_monotone_l1069_106956


namespace more_customers_left_than_stayed_l1069_106961

-- Define the initial number of customers.
def initial_customers : ℕ := 11

-- Define the number of customers who stayed behind.
def customers_stayed : ℕ := 3

-- Define the number of customers who left.
def customers_left : ℕ := initial_customers - customers_stayed

-- Prove that the number of customers who left is 5 more than those who stayed behind.
theorem more_customers_left_than_stayed : customers_left - customers_stayed = 5 := by
  -- Sorry to skip the proof 
  sorry

end more_customers_left_than_stayed_l1069_106961


namespace find_bottle_caps_l1069_106971

variable (B : ℕ) -- Number of bottle caps Danny found at the park.

-- Conditions
variable (current_wrappers : ℕ := 67) -- Danny has 67 wrappers in his collection now.
variable (current_bottle_caps : ℕ := 35) -- Danny has 35 bottle caps in his collection now.
variable (found_wrappers : ℕ := 18) -- Danny found 18 wrappers at the park.
variable (more_wrappers_than_bottle_caps : ℕ := 32) -- Danny has 32 more wrappers than bottle caps.

-- Given the conditions, prove that Danny found 18 bottle caps at the park.
theorem find_bottle_caps (h1 : current_wrappers = current_bottle_caps + more_wrappers_than_bottle_caps)
                         (h2 : current_bottle_caps - B + found_wrappers = current_wrappers - more_wrappers_than_bottle_caps - B) :
  B = 18 :=
by
  sorry

end find_bottle_caps_l1069_106971


namespace intersection_of_A_and_B_l1069_106965

open Set

-- Definitions of sets A and B as per conditions in the problem
def A := {x : ℝ | -1 < x ∧ x < 2}
def B := {x : ℝ | -3 < x ∧ x ≤ 1}

-- The proof statement that A ∩ B = {x | -1 < x ∧ x ≤ 1}
theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | -1 < x ∧ x ≤ 1} := by
  sorry

end intersection_of_A_and_B_l1069_106965


namespace real_y_iff_x_interval_l1069_106970

theorem real_y_iff_x_interval (x : ℝ) :
  (∃ y : ℝ, 3*y^2 + 2*x*y + x + 5 = 0) ↔ (x ≤ -3 ∨ x ≥ 5) :=
by
  sorry

end real_y_iff_x_interval_l1069_106970


namespace description_of_T_l1069_106978

def T (x y : ℝ) : Prop :=
  (5 = x+3 ∧ y-6 ≤ 5) ∨
  (5 = y-6 ∧ x+3 ≤ 5) ∨
  ((x+3 = y-6) ∧ 5 ≤ x+3)

theorem description_of_T :
  ∀ (x y : ℝ), T x y ↔ (x = 2 ∧ y ≤ 11) ∨ (y = 11 ∧ x ≤ 2) ∨ (y = x + 9 ∧ x ≥ 2) :=
sorry

end description_of_T_l1069_106978


namespace problem_b_50_l1069_106977

def seq (b : ℕ → ℕ) : Prop :=
  b 1 = 3 ∧ ∀ n ≥ 1, b (n + 1) = b n + 3 * n

theorem problem_b_50 (b : ℕ → ℕ) (h : seq b) : b 50 = 3678 := 
sorry

end problem_b_50_l1069_106977


namespace carX_travel_distance_after_carY_started_l1069_106931

-- Define the conditions
def carX_speed : ℝ := 35
def carY_speed : ℝ := 40
def delay_time : ℝ := 1.2

-- Define the problem to prove the question is equal to the correct answer given the conditions
theorem carX_travel_distance_after_carY_started : 
  ∃ t : ℝ, carY_speed * t = carX_speed * t + carX_speed * delay_time ∧ 
           carX_speed * t = 294 :=
by
  sorry

end carX_travel_distance_after_carY_started_l1069_106931


namespace bridge_length_l1069_106972

theorem bridge_length 
  (train_length : ℝ) 
  (train_speed_kmph : ℝ) 
  (cross_time_seconds : ℝ)
  (train_length_eq : train_length = 150)
  (train_speed_kmph_eq : train_speed_kmph = 45)
  (cross_time_seconds_eq : cross_time_seconds = 30) : 
  ∃ (bridge_length : ℝ), bridge_length = 225 := 
  by
  sorry

end bridge_length_l1069_106972


namespace blue_to_red_ratio_l1069_106935

-- Define the conditions as given in the problem
def initial_red_balls : ℕ := 16
def lost_red_balls : ℕ := 6
def bought_yellow_balls : ℕ := 32
def total_balls_after_events : ℕ := 74

-- Based on the conditions, we define the remaining red balls and the total balls equation
def remaining_red_balls := initial_red_balls - lost_red_balls

-- Suppose B is the number of blue balls
def blue_balls (B : ℕ) : Prop :=
  remaining_red_balls + B + bought_yellow_balls = total_balls_after_events

-- Now, state the theorem to prove the ratio of blue balls to red balls is 16:5
theorem blue_to_red_ratio (B : ℕ) (h : blue_balls B) : B = 32 → B / remaining_red_balls = 16 / 5 :=
by
  intro B_eq
  subst B_eq
  have h1 : remaining_red_balls = 10 := rfl
  have h2 : 32 / 10  = 16 / 5 := by rfl
  exact h2

-- Note: The proof itself is skipped, so the statement is left with sorry.

end blue_to_red_ratio_l1069_106935


namespace original_price_per_tire_l1069_106962

-- Definitions derived from the problem
def number_of_tires : ℕ := 4
def sale_price_per_tire : ℝ := 75
def total_savings : ℝ := 36

-- Goal to prove the original price of each tire
theorem original_price_per_tire :
  (sale_price_per_tire + total_savings / number_of_tires) = 84 :=
by sorry

end original_price_per_tire_l1069_106962


namespace polygon_edges_of_set_S_l1069_106902

variable (a : ℝ)

def in_set_S(x y : ℝ) : Prop :=
  (a / 2 ≤ x ∧ x ≤ 2 * a) ∧
  (a / 2 ≤ y ∧ y ≤ 2 * a) ∧
  (x + y ≥ a) ∧
  (x + a ≥ y) ∧
  (y + a ≥ x)

theorem polygon_edges_of_set_S (a : ℝ) (h : 0 < a) :
  (∃ n, ∀ x y, in_set_S a x y → n = 6) :=
sorry

end polygon_edges_of_set_S_l1069_106902


namespace friends_total_l1069_106903

-- Define the conditions as constants
def can_go : Nat := 8
def can't_go : Nat := 7

-- Define the total number of friends and the correct answer
def total_friends : Nat := can_go + can't_go
def correct_answer : Nat := 15

-- Prove that the total number of friends is 15
theorem friends_total : total_friends = correct_answer := by
  -- We use the definitions and the conditions directly here
  sorry

end friends_total_l1069_106903


namespace exponentiation_rule_l1069_106923

theorem exponentiation_rule (m n : ℤ) : (-2 * m^3 * n^2)^2 = 4 * m^6 * n^4 :=
by
  sorry

end exponentiation_rule_l1069_106923


namespace three_monotonic_intervals_l1069_106975

open Real

noncomputable def f (b : ℝ) (x : ℝ) : ℝ := - (4 / 3) * x ^ 3 + (b - 1) * x

noncomputable def f' (b : ℝ) (x : ℝ) : ℝ := -4 * x ^ 2 + (b - 1)

theorem three_monotonic_intervals (b : ℝ) (h : (b - 1) > 0) : b > 1 := 
by
  have discriminant : 16 * (b - 1) > 0 := sorry
  sorry

end three_monotonic_intervals_l1069_106975


namespace total_students_in_class_l1069_106963

def students_chorus := 18
def students_band := 26
def students_both := 2
def students_neither := 8

theorem total_students_in_class : 
  (students_chorus + students_band - students_both) + students_neither = 50 := by
  sorry

end total_students_in_class_l1069_106963


namespace selling_price_of_cricket_bat_l1069_106983

variable (profit : ℝ) (profit_percentage : ℝ)
variable (selling_price : ℝ)

theorem selling_price_of_cricket_bat 
  (h1 : profit = 215)
  (h2 : profit_percentage = 33.85826771653544) : 
  selling_price = 849.70 :=
sorry

end selling_price_of_cricket_bat_l1069_106983


namespace events_related_with_99_confidence_l1069_106917

theorem events_related_with_99_confidence (K_squared : ℝ) (h : K_squared > 6.635) : 
  events_A_B_related_with_99_confidence :=
sorry

end events_related_with_99_confidence_l1069_106917


namespace positive_difference_of_solutions_is_zero_l1069_106909

theorem positive_difference_of_solutions_is_zero : ∀ (x : ℂ), (x ^ 2 + 3 * x + 4 = 0) → 
  ∀ (y : ℂ), (y ^ 2 + 3 * y + 4 = 0) → |y.re - x.re| = 0 :=
by
  intro x hx y hy
  sorry

end positive_difference_of_solutions_is_zero_l1069_106909


namespace factor_expression_l1069_106929

variables (a : ℝ)

theorem factor_expression : (45 * a^2 + 135 * a + 90 * a^3) = 45 * a * (90 * a^2 + a + 3) :=
by sorry

end factor_expression_l1069_106929


namespace paul_crayons_l1069_106938

def initial_crayons : ℝ := 479.0
def additional_crayons : ℝ := 134.0
def total_crayons : ℝ := initial_crayons + additional_crayons

theorem paul_crayons : total_crayons = 613.0 :=
by
  sorry

end paul_crayons_l1069_106938


namespace area_of_circle_with_endpoints_l1069_106911

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

noncomputable def radius (d : ℝ) : ℝ :=
  d / 2

noncomputable def area_of_circle (r : ℝ) : ℝ :=
  Real.pi * r^2

theorem area_of_circle_with_endpoints :
  area_of_circle (radius (distance (5, 9) (13, 17))) = 32 * Real.pi :=
by
  sorry

end area_of_circle_with_endpoints_l1069_106911


namespace total_weight_gain_l1069_106920

def orlando_gained : ℕ := 5

def jose_gained (orlando : ℕ) : ℕ :=
  2 * orlando + 2

def fernando_gained (jose : ℕ) : ℕ :=
  jose / 2 - 3

theorem total_weight_gain (O J F : ℕ) 
  (ho : O = orlando_gained) 
  (hj : J = jose_gained O) 
  (hf : F = fernando_gained J) :
  O + J + F = 20 :=
by
  sorry

end total_weight_gain_l1069_106920


namespace sum_of_reciprocals_of_squares_l1069_106919

theorem sum_of_reciprocals_of_squares (a b : ℕ) (h : a * b = 19) : 1 / (a * a : ℚ) + 1 / (b * b : ℚ) = 362 / 361 := 
by
  sorry

end sum_of_reciprocals_of_squares_l1069_106919


namespace quiz_answer_key_count_l1069_106913

theorem quiz_answer_key_count :
  let tf_combinations := 6 -- Combinations of true-false questions
  let mc_combinations := 4 ^ 3 -- Combinations of multiple-choice questions
  tf_combinations * mc_combinations = 384 := by
  -- The values and conditions are directly taken from the problem statement.
  let tf_combinations := 6
  let mc_combinations := 4 ^ 3
  sorry

end quiz_answer_key_count_l1069_106913


namespace cos_alpha_neg_3_5_l1069_106957

open Real

variables {α : ℝ} (h_alpha : sin α = 4 / 5) (h_quadrant : π / 2 < α ∧ α < π)

theorem cos_alpha_neg_3_5 : cos α = -3 / 5 :=
by
  -- Proof omitted
  sorry

end cos_alpha_neg_3_5_l1069_106957


namespace scientific_notation_of_125000_l1069_106914

theorem scientific_notation_of_125000 :
  125000 = 1.25 * 10^5 := sorry

end scientific_notation_of_125000_l1069_106914


namespace intersection_complement_l1069_106944

open Set

def U : Set ℤ := univ
def M : Set ℤ := {1, 2}
def P : Set ℤ := {-2, -1, 0, 1, 2}

theorem intersection_complement :
  P ∩ (U \ M) = {-2, -1, 0} :=
by
  sorry

end intersection_complement_l1069_106944


namespace problem_solution_l1069_106990

/-- Let ⌊x⌋ denote the greatest integer less than or equal to x. Prove
    that the number of real solutions to the equation x² - 2⌊x⌋ - 3 = 0 is 3. -/
theorem problem_solution : ∃ (s : Finset ℝ), s.card = 3 ∧ ∀ x ∈ s, x^2 - 2 * ⌊x⌋ - 3 = 0 := 
sorry

end problem_solution_l1069_106990


namespace quadratic_to_vertex_form_l1069_106905

-- Define the given quadratic function.
def quadratic_function (x : ℝ) : ℝ := x^2 - 2 * x + 3

-- Define the vertex form of the quadratic function.
def vertex_form (x : ℝ) : ℝ := (x - 1)^2 + 2

-- State the equivalence we want to prove.
theorem quadratic_to_vertex_form :
  ∀ x : ℝ, quadratic_function x = vertex_form x :=
by
  intro x
  show quadratic_function x = vertex_form x
  sorry

end quadratic_to_vertex_form_l1069_106905


namespace students_with_all_three_pets_l1069_106945

theorem students_with_all_three_pets :
  ∀ (total_students : ℕ)
    (dog_fraction cat_fraction : ℚ)
    (other_pets students_no_pets dogs_only cats_only other_pets_only x y z w : ℕ),
    total_students = 40 →
    dog_fraction = 5 / 8 →
    cat_fraction = 1 / 4 →
    other_pets = 8 →
    students_no_pets = 4 →
    dogs_only = 15 →
    cats_only = 3 →
    other_pets_only = 2 →
    dogs_only + x + z + w = total_students * dog_fraction →
    cats_only + x + y + w = total_students * cat_fraction →
    other_pets_only + y + z + w = other_pets →
    dogs_only + cats_only + other_pets_only + x + y + z + w = total_students - students_no_pets →
    w = 4  := 
by
  sorry

end students_with_all_three_pets_l1069_106945


namespace find_x_l1069_106942

def integers_x_y (x y : ℤ) : Prop :=
  x > y ∧ y > 0 ∧ x + y + x * y = 110

theorem find_x (x y : ℤ) (h : integers_x_y x y) : x = 36 := sorry

end find_x_l1069_106942


namespace tom_used_10_plates_l1069_106967

theorem tom_used_10_plates
  (weight_per_plate : ℕ := 30)
  (felt_weight : ℕ := 360)
  (heavier_factor : ℚ := 1.20) :
  (felt_weight / heavier_factor / weight_per_plate : ℚ) = 10 := by
  sorry

end tom_used_10_plates_l1069_106967


namespace fraction_of_constants_l1069_106932

theorem fraction_of_constants :
  ∃ a b c : ℤ, (4 : ℤ) * a * (k + b)^2 + c = 4 * k^2 - 8 * k + 16 ∧
             4 * -1 * (k + (-1))^2 + 12 = 4 * k^2 - 8 * k + 16 ∧
             a = 4 ∧ b = -1 ∧ c = 12 ∧ c / b = -12 :=
by
  sorry

end fraction_of_constants_l1069_106932


namespace range_of_m_l1069_106966

open Real

noncomputable def x (y : ℝ) : ℝ := 2 / (1 - 1 / y)

theorem range_of_m (y : ℝ) (m : ℝ) (h1 : y > 0) (h2 : 1 - 1 / y > 0) (h3 : -4 < m) (h4 : m < 2) : 
  x y + 2 * y > m^2 + 2 * m := 
by 
  have hx_pos : x y > 0 := sorry
  have hxy_eq : 2 / x y + 1 / y = 1 := sorry
  have hxy_ge : x y + 2 * y ≥ 8 := sorry
  have h_m_le : 8 > m^2 + 2 * m := sorry
  exact sorry

end range_of_m_l1069_106966


namespace angle_in_second_quadrant_l1069_106952

theorem angle_in_second_quadrant (n : ℤ) : (460 : ℝ) = 360 * n + 100 := by
  sorry

end angle_in_second_quadrant_l1069_106952


namespace number_of_students_joined_l1069_106940

theorem number_of_students_joined
  (A : ℝ)
  (x : ℕ)
  (h1 : A = 50)
  (h2 : (100 + x) * (A - 10) = 5400) 
  (h3 : 100 * A + 400 = 5400) :
  x = 35 := 
by 
  -- all conditions in a) are used as definitions in Lean 4 statement
  sorry

end number_of_students_joined_l1069_106940


namespace valid_relationship_l1069_106968

noncomputable def proof_statement (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_eq : a^2 + c^2 = 2 * b * c) : Prop :=
  b > a ∧ a > c

theorem valid_relationship (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_eq : a^2 + c^2 = 2 * b * c) : proof_statement a b c h_distinct h_pos h_eq :=
  sorry

end valid_relationship_l1069_106968


namespace tan_double_angle_l1069_106907

theorem tan_double_angle (α : ℝ) 
  (h1 : α ∈ Set.Ioo (π / 2) π) 
  (h2 : Real.sin (α - π / 2) = 3 / 5) : 
  Real.tan (2 * α) = 24 / 7 :=
by
  sorry

end tan_double_angle_l1069_106907


namespace arithmetic_geometric_sequence_l1069_106959

theorem arithmetic_geometric_sequence (a b c : ℝ) 
  (a_ne_b : a ≠ b) (b_ne_c : b ≠ c) (a_ne_c : a ≠ c)
  (h1 : 2 * b = a + c)
  (h2 : (a * b)^2 = a * b * c^2)
  (h3 : a + b + c = 15) : a = 20 := 
by 
  sorry

end arithmetic_geometric_sequence_l1069_106959


namespace gear_ratio_proportion_l1069_106936

variables {x y z w : ℕ} {ω_A ω_B ω_C ω_D : ℝ}

theorem gear_ratio_proportion 
  (h1: x * ω_A = y * ω_B) 
  (h2: y * ω_B = z * ω_C) 
  (h3: z * ω_C = w * ω_D):
  ω_A / ω_B = y * z * w / (x * z * w) ∧ 
  ω_B / ω_C = x * z * w / (y * x * w) ∧ 
  ω_C / ω_D = x * y * w / (z * y * w) ∧ 
  ω_D / ω_A = x * y * z / (w * z * y) :=
sorry  -- Proof is not included

end gear_ratio_proportion_l1069_106936


namespace problem_A_plus_B_l1069_106946

variable {A B : ℝ} (h1 : A ≠ B) (h2 : ∀ x : ℝ, (A * (B * x + A) + B) - (B * (A * x + B) + A) = 2 * (B - A))

theorem problem_A_plus_B : A + B = -2 :=
by
  sorry

end problem_A_plus_B_l1069_106946


namespace roots_interlaced_l1069_106979

variable {α : Type*} [LinearOrderedField α]
variables {f g : α → α}

theorem roots_interlaced
    (x1 x2 x3 x4 : α)
    (h1 : x1 < x2) (h2 : x3 < x4)
    (hfx1 : f x1 = 0) (hfx2 : f x2 = 0)
    (hfx_distinct : x1 ≠ x2)
    (hgx3 : g x3 = 0) (hgx4 : g x4 = 0)
    (hgx_distinct : x3 ≠ x4)
    (hgx1_ne_0 : g x1 ≠ 0) (hgx2_ne_0 : g x2 ≠ 0)
    (hgx1_gx2_lt_0 : g x1 * g x2 < 0) :
    (x1 < x3 ∧ x3 < x2 ∧ x2 < x4) ∨ (x3 < x1 ∧ x1 < x4 ∧ x4 < x2) :=
sorry

end roots_interlaced_l1069_106979


namespace other_root_and_m_l1069_106943

-- Definitions for the conditions
def quadratic_eq (m : ℝ) := ∀ x : ℝ, x^2 + 2 * x + m = 0
def root (x : ℝ) (m : ℝ) := x^2 + 2 * x + m = 0

-- Theorem statement
theorem other_root_and_m (m : ℝ) (h : root 2 m) : ∃ t : ℝ, (2 + t = -2) ∧ (2 * t = m) ∧ t = -4 ∧ m = -8 := 
by {
  -- Placeholder for the actual proof
  sorry
}

end other_root_and_m_l1069_106943


namespace probability_calculation_l1069_106949

noncomputable def probability_floor_sqrt_x_eq_17_given_floor_sqrt_2x_eq_25 : ℝ :=
  let total_interval_length := 100
  let intersection_interval_length := 324 - 312.5
  intersection_interval_length / total_interval_length

theorem probability_calculation : probability_floor_sqrt_x_eq_17_given_floor_sqrt_2x_eq_25 = 23 / 200 := by
  sorry

end probability_calculation_l1069_106949


namespace profit_share_difference_l1069_106985

theorem profit_share_difference (P : ℝ) (hP : P = 1000) 
  (rX rY : ℝ) (hRatio : rX / rY = (1/2) / (1/3)) : 
  let total_parts := (1/2) + (1/3)
  let value_per_part := P / total_parts
  let x_share := (1/2) * value_per_part
  let y_share := (1/3) * value_per_part
  x_share - y_share = 200 := by 
  sorry

end profit_share_difference_l1069_106985


namespace length_of_tunnel_l1069_106954

theorem length_of_tunnel (time : ℝ) (speed : ℝ) (train_length : ℝ) (total_distance : ℝ) (tunnel_length : ℝ) 
  (h1 : time = 30) (h2 : speed = 100 / 3) (h3 : train_length = 400) (h4 : total_distance = speed * time) 
  (h5 : tunnel_length = total_distance - train_length) : 
  tunnel_length = 600 :=
by
  sorry

end length_of_tunnel_l1069_106954


namespace tree_count_l1069_106901

theorem tree_count (m N : ℕ) 
  (h1 : 12 ≡ (33 - m) [MOD N])
  (h2 : (105 - m) ≡ 8 [MOD N]) :
  N = 76 := 
sorry

end tree_count_l1069_106901


namespace number_of_balls_condition_l1069_106941

theorem number_of_balls_condition (X : ℕ) (h1 : 25 - 20 = X - 25) : X = 30 :=
by
  sorry

end number_of_balls_condition_l1069_106941


namespace simplify_expression_l1069_106937

theorem simplify_expression (a : ℝ) : 2 * (a + 2) - 2 * a = 4 :=
by
  sorry

end simplify_expression_l1069_106937


namespace painted_rooms_l1069_106933

/-- Given that there are a total of 11 rooms to paint, each room takes 7 hours to paint,
and the painter has 63 hours of work left to paint the remaining rooms,
prove that the painter has already painted 2 rooms. -/
theorem painted_rooms (total_rooms : ℕ) (hours_per_room : ℕ) (hours_left : ℕ) 
  (h_total_rooms : total_rooms = 11) (h_hours_per_room : hours_per_room = 7) 
  (h_hours_left : hours_left = 63) : 
  (total_rooms - hours_left / hours_per_room) = 2 := 
by
  sorry

end painted_rooms_l1069_106933


namespace fraction_book_read_l1069_106960

theorem fraction_book_read (read_pages : ℚ) (h : read_pages = 3/7) :
  (1 - read_pages = 4/7) ∧ (read_pages / (1 - read_pages) = 3/4) :=
by
  sorry

end fraction_book_read_l1069_106960


namespace james_oranges_l1069_106984

-- Define the problem conditions
variables (o a : ℕ) -- o is number of oranges, a is number of apples

-- Condition: James bought apples and oranges over a seven-day week
def days_week := o + a = 7

-- Condition: The total cost must be a whole number of dollars (divisible by 100 cents)
def total_cost := 65 * o + 40 * a ≡ 0 [MOD 100]

-- We need to prove: James bought 4 oranges
theorem james_oranges (o a : ℕ) (h_days_week : days_week o a) (h_total_cost : total_cost o a) : o = 4 :=
sorry

end james_oranges_l1069_106984


namespace log35_28_l1069_106934

variable (a b : ℝ)
variable (log : ℝ → ℝ → ℝ)

-- Conditions
axiom log14_7_eq_a : log 14 7 = a
axiom log14_5_eq_b : log 14 5 = b

-- Theorem to prove
theorem log35_28 (h1 : log 14 7 = a) (h2 : log 14 5 = b) : log 35 28 = (2 - a) / (a + b) :=
sorry

end log35_28_l1069_106934


namespace figure_square_count_l1069_106996

theorem figure_square_count (f : ℕ → ℕ)
  (h0 : f 0 = 2)
  (h1 : f 1 = 8)
  (h2 : f 2 = 18)
  (h3 : f 3 = 32) :
  f 100 = 20402 :=
sorry

end figure_square_count_l1069_106996


namespace calculate_f_g_l1069_106981

noncomputable def f (x : ℕ) : ℕ := 4 * x + 3
noncomputable def g (x : ℕ) : ℕ := (x + 2) ^ 2

theorem calculate_f_g : f (g 3) = 103 :=
by 
  -- Proof omitted.
  sorry

end calculate_f_g_l1069_106981


namespace find_circle_center_l1069_106973

-- Definition of the circle equation
def circle_eq (x y : ℝ) : Prop := x^2 - 6*x + y^2 + 10*y - 7 = 0

-- The main statement to prove
theorem find_circle_center :
  (∃ center : ℝ × ℝ, center = (3, -5) ∧ ∀ x y : ℝ, circle_eq x y ↔ (x - 3)^2 + (y + 5)^2 = 41) :=
sorry

end find_circle_center_l1069_106973


namespace m_perp_n_α_perp_β_l1069_106997

variables {Plane Line : Type}
variables (α β : Plane) (m n : Line)

def perpendicular_to_plane (l : Line) (p : Plane) : Prop := sorry
def perpendicular_lines (l1 l2 : Line) : Prop := sorry
def perpendicular_planes (p1 p2 : Plane) : Prop := sorry

-- Problem 1:
axiom m_perp_α : perpendicular_to_plane m α
axiom n_perp_β : perpendicular_to_plane n β
axiom α_perp_β : perpendicular_planes α β

theorem m_perp_n : perpendicular_lines m n :=
sorry

-- Problem 2:
axiom m_perp_n' : perpendicular_lines m n
axiom m_perp_α' : perpendicular_to_plane m α
axiom n_perp_β' : perpendicular_to_plane n β

theorem α_perp_β' : perpendicular_planes α β :=
sorry

end m_perp_n_α_perp_β_l1069_106997


namespace largest_gcd_of_sum_1729_l1069_106992

theorem largest_gcd_of_sum_1729 (x y : ℕ) (h1 : x > 0) (h2 : y > 0) (h3 : x + y = 1729) :
  ∃ g, g = Nat.gcd x y ∧ g = 247 := sorry

end largest_gcd_of_sum_1729_l1069_106992
