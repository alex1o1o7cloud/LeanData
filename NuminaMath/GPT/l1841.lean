import Mathlib

namespace remainder_of_x_l1841_184139

theorem remainder_of_x (x : ℕ) 
(H1 : 4 + x ≡ 81 [MOD 16])
(H2 : 6 + x ≡ 16 [MOD 36])
(H3 : 8 + x ≡ 36 [MOD 64]) :
  x ≡ 37 [MOD 48] :=
sorry

end remainder_of_x_l1841_184139


namespace find_y_l1841_184100

theorem find_y (y : ℝ) (h : 3 * y / 7 = 21) : y = 49 := 
sorry

end find_y_l1841_184100


namespace parallel_line_through_point_l1841_184114

theorem parallel_line_through_point :
  ∀ {x y : ℝ}, (3 * x + 4 * y + 1 = 0) ∧ (∃ (a b : ℝ), a = 1 ∧ b = 2 ∧ (3 * a + 4 * b + x0 = 0) → (x = -11)) :=
sorry

end parallel_line_through_point_l1841_184114


namespace dragos_wins_l1841_184184

variable (S : Set ℕ) [Infinite S]
variable (x : ℕ → ℕ)
variable (M N : ℕ)
variable (p : ℕ)

theorem dragos_wins (h_prime_p : Nat.Prime p) (h_subset_S : p ∈ S) 
  (h_xn_distinct : ∀ i j, i ≠ j → x i ≠ x j) 
  (h_pM_div_xn : ∀ n, n ≥ N → p^M ∣ x n): 
  ∃ N, ∀ n, n ≥ N → p^M ∣ x n :=
sorry

end dragos_wins_l1841_184184


namespace asha_win_probability_l1841_184112

theorem asha_win_probability :
  let P_Lose := (3 : ℚ) / 8
  let P_Tie := (1 : ℚ) / 4
  P_Lose + P_Tie < 1 → 1 - P_Lose - P_Tie = (3 : ℚ) / 8 := 
by
  sorry

end asha_win_probability_l1841_184112


namespace range_of_a_l1841_184178

theorem range_of_a (a : ℝ) :
  (¬ ∃ x : ℝ, x^2 + (1 - a) * x + 1 < 0) → a ∈ Set.Icc (-1 : ℝ) 3 :=
by
  sorry

end range_of_a_l1841_184178


namespace fewer_free_throws_l1841_184115

noncomputable def Deshawn_free_throws : ℕ := 12
noncomputable def Kayla_free_throws : ℕ := Deshawn_free_throws + (Deshawn_free_throws / 2)
noncomputable def Annieka_free_throws : ℕ := 14

theorem fewer_free_throws :
  Annieka_free_throws = Kayla_free_throws - 4 :=
by
  sorry

end fewer_free_throws_l1841_184115


namespace find_m_n_l1841_184170

-- Define the vectors OA, OB, OC
def vector_oa (m : ℝ) : ℝ × ℝ := (-2, m)
def vector_ob (n : ℝ) : ℝ × ℝ := (n, 1)
def vector_oc : ℝ × ℝ := (5, -1)

-- Define the condition that OA is perpendicular to OB
def perpendicular (v1 v2 : ℝ × ℝ) : Prop := v1.1 * v2.1 + v1.2 * v2.2 = 0

-- Define the condition that points A, B, and C are collinear.
def collinear (A B C : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ (A.1 - B.1) * (C.2 - A.2) = k * ((C.1 - A.1) * (A.2 - B.2))

theorem find_m_n (m n : ℝ) :
  collinear (-2, m) (n, 1) (5, -1) ∧ perpendicular (-2, m) (n, 1) → m = 3 ∧ n = 3 / 2 := by
  intro h
  sorry

end find_m_n_l1841_184170


namespace equivalent_proof_problem_l1841_184120

-- Define the real numbers x, y, z and the operation ⊗
variables {x y z : ℝ}

def otimes (a b : ℝ) : ℝ := (a - b) ^ 2

theorem equivalent_proof_problem : otimes ((x + z) ^ 2) ((z + y) ^ 2) = (x ^ 2 + 2 * x * z - y ^ 2 - 2 * z * y) ^ 2 :=
by sorry

end equivalent_proof_problem_l1841_184120


namespace sam_walked_distance_when_meeting_l1841_184141

variable (D_s D_f : ℝ)
variable (t : ℝ)

theorem sam_walked_distance_when_meeting
  (h1 : 55 = D_f + D_s)
  (h2 : D_f = 6 * t)
  (h3 : D_s = 5 * t) :
  D_s = 25 :=
by 
  -- This is where the proof would go
  sorry

end sam_walked_distance_when_meeting_l1841_184141


namespace largest_value_b_l1841_184142

theorem largest_value_b (b : ℚ) : (3 * b + 7) * (b - 2) = 9 * b -> b = (4 + Real.sqrt 58) / 3 :=
by
  sorry

end largest_value_b_l1841_184142


namespace square_area_correct_l1841_184182

-- Define the length of the side of the square
def side_length : ℕ := 15

-- Define the area calculation for a square
def square_area (side : ℕ) : ℕ := side * side

-- Define the area calculation for a triangle using the square area division
def triangle_area (square_area : ℕ) : ℕ := square_area / 2

-- Theorem stating that the area of a square with given side length is 225 square units
theorem square_area_correct : square_area side_length = 225 := by
  sorry

end square_area_correct_l1841_184182


namespace product_value_l1841_184113

theorem product_value :
  (1 / 3) * 9 * (1 / 27) * 81 * (1 / 243) * 729 * (1 / 2187) * 6561 = 81 :=
  sorry

end product_value_l1841_184113


namespace max_radius_of_circle_touching_graph_l1841_184172

theorem max_radius_of_circle_touching_graph :
  ∃ r : ℝ, (∀ (x : ℝ), (x^2 + (x^4 - r)^2 = r^2) → r ≤ (3 * (2:ℝ)^(1/3)) / 4) ∧
           r = (3 * (2:ℝ)^(1/3)) / 4 :=
by
  sorry

end max_radius_of_circle_touching_graph_l1841_184172


namespace channel_depth_l1841_184173

theorem channel_depth
  (top_width bottom_width area : ℝ)
  (h : ℝ)
  (trapezium_area_formula : area = (1 / 2) * (top_width + bottom_width) * h)
  (top_width_val : top_width = 14)
  (bottom_width_val : bottom_width = 8)
  (area_val : area = 770) :
  h = 70 := 
by
  sorry

end channel_depth_l1841_184173


namespace tom_won_whack_a_mole_l1841_184192

variable (W : ℕ)  -- let W be the number of tickets Tom won playing 'whack a mole'
variable (won_skee_ball : ℕ := 25)  -- Tom won 25 tickets playing 'skee ball'
variable (spent_on_hat : ℕ := 7)  -- Tom spent 7 tickets on a hat
variable (tickets_left : ℕ := 50)  -- Tom has 50 tickets left

theorem tom_won_whack_a_mole :
  W + 25 + 50 = 57 →
  W = 7 :=
by
  sorry  -- proof goes here

end tom_won_whack_a_mole_l1841_184192


namespace dogs_sold_correct_l1841_184124

-- Definitions based on conditions
def ratio_cats_to_dogs (cats dogs : ℕ) := 2 * dogs = cats

-- Given conditions
def cats_sold := 16
def dogs_sold := 8

-- The theorem to prove
theorem dogs_sold_correct (h : ratio_cats_to_dogs cats_sold dogs_sold) : dogs_sold = 8 :=
by
  sorry

end dogs_sold_correct_l1841_184124


namespace batsman_average_increase_l1841_184150

def average_increase (avg_before : ℕ) (runs_12th_inning : ℕ) (avg_after : ℕ) : ℕ :=
  avg_after - avg_before

theorem batsman_average_increase :
  ∀ (avg_before runs_12th_inning avg_after : ℕ),
    (runs_12th_inning = 70) →
    (avg_after = 37) →
    (11 * avg_before + runs_12th_inning = 12 * avg_after) →
    average_increase avg_before runs_12th_inning avg_after = 3 :=
by
  intros avg_before runs_12th_inning avg_after h_runs h_avg_after h_total
  sorry

end batsman_average_increase_l1841_184150


namespace find_f7_l1841_184171

noncomputable def f : ℝ → ℝ := sorry

-- The conditions provided in the problem
axiom odd_function : ∀ x : ℝ, f (-x) = -f x
axiom periodic_function : ∀ x : ℝ, f (x + 4) = f x
axiom function_in_interval : ∀ x, 0 < x ∧ x < 2 → f x = 2 * x^2

-- The final proof goal
theorem find_f7 : f 7 = -2 :=
by sorry

end find_f7_l1841_184171


namespace intersection_point_zero_a_0_l₁_passes_through_0_2_l₁_perpendicular_l₂_l₁_parallel_l₂_l1841_184186

-- Definitions based on the conditions
def l₁ (a : ℝ) (x y : ℝ) : Prop := a * x - y + 2 = 0
def l₂ (a : ℝ) (x y : ℝ) : Prop := x - a * y + 2 = 0
def perpendicular (a : ℝ) : Prop := a = 0
def parallel (a : ℝ) : Prop := a = 1 ∨ a = -1

-- Theorem 1: Intersection point when a = 0 is (-2, 2)
theorem intersection_point_zero_a_0 :
  ∀ x y : ℝ, l₁ 0 x y → l₂ 0 x y → (x, y) = (-2, 2) := 
by
  sorry

-- Theorem 2: Line l₁ always passes through (0, 2)
theorem l₁_passes_through_0_2 :
  ∀ a : ℝ, l₁ a 0 2 := 
by
  sorry

-- Theorem 3: l₁ is perpendicular to l₂ implies a = 0
theorem l₁_perpendicular_l₂ :
  (∀ x y : ℝ, l₁ a x y → l₂ a x y → ∀ m n, (a * m + (n / a) = 0)) → (a = 0) :=
by
  sorry

-- Theorem 4: l₁ is parallel to l₂ implies a = 1 or a = -1
theorem l₁_parallel_l₂ :
  ∀ a : ℝ, parallel a → (a = 1 ∨ a = -1) :=
by
  sorry

end intersection_point_zero_a_0_l₁_passes_through_0_2_l₁_perpendicular_l₂_l₁_parallel_l₂_l1841_184186


namespace determine_x_l1841_184157

theorem determine_x (x : Nat) (h1 : x % 9 = 0) (h2 : x^2 > 225) (h3 : x < 30) : x = 18 ∨ x = 27 :=
sorry

end determine_x_l1841_184157


namespace yellow_chips_are_one_l1841_184136

-- Definitions based on conditions
def yellow_chip_points : ℕ := 2
def blue_chip_points : ℕ := 4
def green_chip_points : ℕ := 5

variables (Y B G : ℕ)

-- Given conditions
def point_product_condition : Prop := (yellow_chip_points^Y * blue_chip_points^B * green_chip_points^G = 16000)
def equal_blue_green : Prop := (B = G)

-- Theorem to prove the number of yellow chips
theorem yellow_chips_are_one (Y B G : ℕ) (hprod : point_product_condition Y B G) (heq : equal_blue_green B G) : Y = 1 :=
by {
    sorry -- Proof omitted
}

end yellow_chips_are_one_l1841_184136


namespace girls_friends_count_l1841_184189

variable (days_in_week : ℕ)
variable (total_friends : ℕ)
variable (boys : ℕ)

axiom H1 : days_in_week = 7
axiom H2 : total_friends = 2 * days_in_week
axiom H3 : boys = 11

theorem girls_friends_count : total_friends - boys = 3 :=
by sorry

end girls_friends_count_l1841_184189


namespace calculation_result_l1841_184155

theorem calculation_result : (18 * 23 - 24 * 17) / 3 + 5 = 7 :=
by
  sorry

end calculation_result_l1841_184155


namespace project_scientists_total_l1841_184151

def total_scientists (S : ℕ) : Prop :=
  S / 2 + S / 5 + 21 = S

theorem project_scientists_total : ∃ S, total_scientists S ∧ S = 70 :=
by
  existsi 70
  unfold total_scientists
  sorry

end project_scientists_total_l1841_184151


namespace max_diagonals_in_grid_l1841_184108

-- Define the dimensions of the grid
def grid_width := 8
def grid_height := 5

-- Define the number of 1x2 rectangles
def number_of_1x2_rectangles := grid_width / 2 * grid_height

-- State the theorem
theorem max_diagonals_in_grid : number_of_1x2_rectangles = 20 := 
by 
  -- Simplify the expression
  sorry

end max_diagonals_in_grid_l1841_184108


namespace original_decimal_number_l1841_184161

theorem original_decimal_number (x : ℝ) (h : x / 100 = x - 1.485) : x = 1.5 := 
by
  sorry

end original_decimal_number_l1841_184161


namespace accessories_cost_is_200_l1841_184174

variable (c_cost a_cost : ℕ)
variable (ps_value ps_sold : ℕ)
variable (john_paid : ℕ)

-- Given Conditions
def computer_cost := 700
def accessories_cost := a_cost
def playstation_value := 400
def playstation_sold := ps_value - (ps_value * 20 / 100)
def john_paid_amount := 580

-- Theorem to be proved
theorem accessories_cost_is_200 :
  ps_value = 400 →
  ps_sold = playstation_sold →
  c_cost = 700 →
  john_paid = 580 →
  john_paid + ps_sold - c_cost = a_cost →
  a_cost = 200 :=
by
  intros
  sorry

end accessories_cost_is_200_l1841_184174


namespace quotient_of_division_l1841_184103

theorem quotient_of_division
  (larger smaller : ℕ)
  (h1 : larger - smaller = 1370)
  (h2 : larger = 1626)
  (h3 : ∃ q r, larger = smaller * q + r ∧ r = 15) :
  ∃ q, larger = smaller * q + 15 ∧ q = 6 :=
by
  sorry

end quotient_of_division_l1841_184103


namespace line_through_points_l1841_184187

theorem line_through_points (m b : ℝ)
  (h_slope : m = (-1 - 3) / (-3 - 1))
  (h_point : 3 = m * 1 + b) :
  m + b = 3 :=
sorry

end line_through_points_l1841_184187


namespace math_problem_equivalent_l1841_184129

-- Given that the problem requires four distinct integers a, b, c, d which are less than 12 and invertible modulo 12.
def coprime_with_12 (x : ℕ) : Prop := Nat.gcd x 12 = 1

theorem math_problem_equivalent 
  (a b c d : ℕ) (ha : coprime_with_12 a) (hb : coprime_with_12 b) 
  (hc : coprime_with_12 c) (hd : coprime_with_12 d) 
  (hab : a ≠ b) (hac : a ≠ c) (had : a ≠ d) (hbc : b ≠ c)
  (hbd : b ≠ d) (hcd : c ≠ d) :
  ((a * b * c * d) + (a * b * c) + (a * b * d) + (a * c * d) + (b * c * d)) * Nat.gcd (a * b * c * d) 12 = 1 :=
sorry

end math_problem_equivalent_l1841_184129


namespace find_vector_c_l1841_184179

-- Definitions of the given vectors
def vector_a : ℝ × ℝ := (3, -1)
def vector_b : ℝ × ℝ := (-1, 2)
def vector_c : ℝ × ℝ := (2 * vector_a.1 + vector_b.1, 2 * vector_a.2 + vector_b.2)

-- The goal is to prove that vector_c = (5, 0)
theorem find_vector_c : vector_c = (5, 0) :=
by
  -- proof steps would go here
  sorry

end find_vector_c_l1841_184179


namespace lateral_surface_area_of_cylinder_l1841_184177

theorem lateral_surface_area_of_cylinder (V : ℝ) (hV : V = 27 * Real.pi) : 
  ∃ (S : ℝ), S = 18 * Real.pi :=
by
  sorry

end lateral_surface_area_of_cylinder_l1841_184177


namespace find_5a_plus_5b_l1841_184116

noncomputable def g (x : ℝ) : ℝ := 5 * x - 4
noncomputable def f (a b x : ℝ) : ℝ := a * x + b
noncomputable def f_inv (a b x : ℝ) : ℝ := g x + 3

theorem find_5a_plus_5b (a b : ℝ) (h_inverse : ∀ x, f_inv a b (f a b x) = x) : 5 * a + 5 * b = 2 :=
by
  sorry

end find_5a_plus_5b_l1841_184116


namespace percentage_increase_of_return_trip_l1841_184176

noncomputable def speed_increase_percentage (initial_speed avg_speed : ℝ) : ℝ :=
  ((2 * avg_speed * initial_speed) / avg_speed - initial_speed) * 100 / initial_speed

theorem percentage_increase_of_return_trip :
  let initial_speed := 30
  let avg_speed := 34.5
  speed_increase_percentage initial_speed avg_speed = 35.294 :=
  sorry

end percentage_increase_of_return_trip_l1841_184176


namespace necessarily_positive_b_plus_3c_l1841_184193

theorem necessarily_positive_b_plus_3c 
  (a b c : ℝ) 
  (ha : 0 < a ∧ a < 2) 
  (hb : -2 < b ∧ b < 0) 
  (hc : 0 < c ∧ c < 3) : 
  b + 3 * c > 0 := 
sorry

end necessarily_positive_b_plus_3c_l1841_184193


namespace domain_of_function_l1841_184183

/-- Prove the domain of the function f(x) = log10(2 * cos x - 1) + sqrt(49 - x^2) -/
theorem domain_of_function :
  { x : ℝ | -7 ≤ x ∧ x < - (5 * Real.pi) / 3 ∨ - Real.pi / 3 < x ∧ x < Real.pi / 3 ∨ (5 * Real.pi) / 3 < x ∧ x ≤ 7 }
  = { x : ℝ | 2 * Real.cos x - 1 > 0 ∧ 49 - x^2 ≥ 0 } :=
by {
  sorry
}

end domain_of_function_l1841_184183


namespace shooting_to_practice_ratio_l1841_184168

variable (practiceTime minutes weightliftingTime runningTime shootingTime : ℕ)
variable (runningWeightliftingRatio : ℕ)

axiom practiceTime_def : practiceTime = 2 * 60 -- converting 2 hours to minutes
axiom weightliftingTime_def : weightliftingTime = 20
axiom runningWeightliftingRatio_def : runningWeightliftingRatio = 2
axiom runningTime_def : runningTime = runningWeightliftingRatio * weightliftingTime
axiom shootingTime_def : shootingTime = practiceTime - (runningTime + weightliftingTime)

theorem shooting_to_practice_ratio (practiceTime minutes weightliftingTime runningTime shootingTime : ℕ) 
                                   (runningWeightliftingRatio : ℕ) :
  practiceTime = 120 →
  weightliftingTime = 20 →
  runningWeightliftingRatio = 2 →
  runningTime = runningWeightliftingRatio * weightliftingTime →
  shootingTime = practiceTime - (runningTime + weightliftingTime) →
  (shootingTime : ℚ) / practiceTime = 1 / 2 :=
by sorry

end shooting_to_practice_ratio_l1841_184168


namespace selection_schemes_l1841_184198

theorem selection_schemes (people : Finset ℕ) (A B C : ℕ) (h_people : people.card = 5) 
(h_A_B_individuals : A ∈ people ∧ B ∈ people) (h_A_B_C_exclusion : A ≠ C ∧ B ≠ C) :
  ∃ (number_of_schemes : ℕ), number_of_schemes = 36 :=
by
  sorry

end selection_schemes_l1841_184198


namespace least_number_to_add_l1841_184180

theorem least_number_to_add (a : ℕ) (b : ℕ) (n : ℕ) (h : a = 1056) (h1: b = 26) (h2 : n = 10) : 
  (a + n) % b = 0 := 
sorry

end least_number_to_add_l1841_184180


namespace reciprocal_inequality_l1841_184105

variable (a b : ℝ)

theorem reciprocal_inequality (ha : a < 0) (hb : b > 0) : (1 / a) < (1 / b) := sorry

end reciprocal_inequality_l1841_184105


namespace allens_mothers_age_l1841_184156

-- Define the conditions
variables (A M S : ℕ) -- Declare variables for ages of Allen, his mother, and his sister

-- Define Allen is 30 years younger than his mother
axiom h1 : A = M - 30

-- Define Allen's sister is 5 years older than him
axiom h2 : S = A + 5

-- Define in 7 years, the sum of their ages will be 110
axiom h3 : (A + 7) + (M + 7) + (S + 7) = 110

-- Define the age difference between Allen's mother and sister is 25 years
axiom h4 : M - S = 25

-- State the theorem: what is the present age of Allen's mother
theorem allens_mothers_age : M = 48 :=
by sorry

end allens_mothers_age_l1841_184156


namespace no_real_roots_of_quadratic_l1841_184169

theorem no_real_roots_of_quadratic (k : ℝ) (h : k ≠ 0) : ¬ ∃ x : ℝ, x^2 + k * x + 2 * k^2 = 0 :=
by sorry

end no_real_roots_of_quadratic_l1841_184169


namespace cost_prices_of_products_l1841_184175

-- Define the variables and conditions from the problem
variables (x y : ℝ)

-- Theorem statement
theorem cost_prices_of_products (h1 : 20 * x + 15 * y = 380) (h2 : 15 * x + 10 * y = 280) : 
  x = 16 ∧ y = 4 :=
by
  -- Proof omitted
  sorry

end cost_prices_of_products_l1841_184175


namespace biscuits_more_than_cookies_l1841_184128

theorem biscuits_more_than_cookies :
  let morning_butter_cookies := 20
  let morning_biscuits := 40
  let afternoon_butter_cookies := 10
  let afternoon_biscuits := 20
  let total_butter_cookies := morning_butter_cookies + afternoon_butter_cookies
  let total_biscuits := morning_biscuits + afternoon_biscuits
  total_biscuits - total_butter_cookies = 30 :=
by
  sorry

end biscuits_more_than_cookies_l1841_184128


namespace simplify_expression_l1841_184140

theorem simplify_expression (x y : ℝ) (h : x - 2 * y = -2) : 9 - 2 * x + 4 * y = 13 :=
by sorry

end simplify_expression_l1841_184140


namespace find_alpha_plus_beta_l1841_184123

theorem find_alpha_plus_beta (α β : ℝ)
  (h : ∀ x : ℝ, x ≠ 45 → (x - α) / (x + β) = (x^2 - 90 * x + 1981) / (x^2 + 63 * x - 3420)) :
  α + β = 113 :=
by
  sorry

end find_alpha_plus_beta_l1841_184123


namespace smallest_y_value_l1841_184110

theorem smallest_y_value : 
  ∀ y : ℝ, (3 * y^2 + 15 * y - 90 = y * (y + 20)) → y ≥ -6 :=
by
  sorry

end smallest_y_value_l1841_184110


namespace min_value_fraction_l1841_184162

theorem min_value_fraction (x : ℝ) (h : x > 4) : 
  ∃ y : ℝ, (y = 2 * Real.sqrt 19) ∧ (∀ z : ℝ, (z = (x + 15) / Real.sqrt (x - 4)) → z ≥ y) :=
by
  sorry

end min_value_fraction_l1841_184162


namespace chromium_percentage_alloy_l1841_184134

theorem chromium_percentage_alloy 
  (w1 w2 w3 w4 : ℝ)
  (p1 p2 p3 p4 : ℝ)
  (h_w1 : w1 = 15)
  (h_w2 : w2 = 30)
  (h_w3 : w3 = 10)
  (h_w4 : w4 = 5)
  (h_p1 : p1 = 0.12)
  (h_p2 : p2 = 0.08)
  (h_p3 : p3 = 0.15)
  (h_p4 : p4 = 0.20) :
  (w1 * p1 + w2 * p2 + w3 * p3 + w4 * p4) / (w1 + w2 + w3 + w4) * 100 = 11.17 := 
  sorry

end chromium_percentage_alloy_l1841_184134


namespace Joel_laps_count_l1841_184101

-- Definitions of conditions
def Yvonne_laps := 10
def sister_laps := Yvonne_laps / 2
def Joel_laps := sister_laps * 3

-- Statement to be proved
theorem Joel_laps_count : Joel_laps = 15 := by
  -- currently, proof is not required, so we defer it with 'sorry'
  sorry

end Joel_laps_count_l1841_184101


namespace slope_angle_correct_l1841_184109

def parametric_line (α : ℝ) : Prop :=
  α = 50 * (Real.pi / 180)

theorem slope_angle_correct : ∀ (t : ℝ),
  parametric_line 50 →
  ∀ α : ℝ, α = 140 * (Real.pi / 180) :=
by
  intro t
  intro h
  intro α
  sorry

end slope_angle_correct_l1841_184109


namespace average_candies_l1841_184102

theorem average_candies {a b c d e f : ℕ} (h₁ : a = 16) (h₂ : b = 22) (h₃ : c = 30) (h₄ : d = 26) (h₅ : e = 18) (h₆ : f = 20) :
  (a + b + c + d + e + f) / 6 = 22 := by
  sorry

end average_candies_l1841_184102


namespace correct_calculation_l1841_184138

theorem correct_calculation (a x y b : ℝ) :
  (-a - a = 0) = False ∧
  (- (x + y) = -x - y) = True ∧
  (3 * (b - 2 * a) = 3 * b - 2 * a) = False ∧
  (8 * a^4 - 6 * a^2 = 2 * a^2) = False :=
by
  sorry

end correct_calculation_l1841_184138


namespace part1_part2_l1841_184145

namespace RationalOp
  -- Define the otimes operation
  def otimes (a b : ℚ) : ℚ := a * b^2 + 2 * a * b + a

  -- Part 1: Prove (-2) ⊗ 4 = -50
  theorem part1 : otimes (-2) 4 = -50 := sorry

  -- Part 2: Given x ⊗ 3 = y ⊗ (-3), prove 8x - 2y + 5 = 5
  theorem part2 (x y : ℚ) (h : otimes x 3 = otimes y (-3)) : 8*x - 2*y + 5 = 5 := sorry
end RationalOp

end part1_part2_l1841_184145


namespace Ricardo_coin_difference_l1841_184164

theorem Ricardo_coin_difference (p : ℕ) (h₁ : 1 ≤ p) (h₂ : p ≤ 3029) :
    let max_value := 15150 - 4 * 1
    let min_value := 15150 - 4 * 3029
    max_value - min_value = 12112 := by
  sorry

end Ricardo_coin_difference_l1841_184164


namespace original_perimeter_l1841_184144

theorem original_perimeter (a b : ℝ) (h : a / 2 + b / 2 = 129 / 2) : 2 * (a + b) = 258 :=
by
  sorry

end original_perimeter_l1841_184144


namespace hiker_speeds_l1841_184146

theorem hiker_speeds:
  ∃ (d : ℝ), 
  (d > 5) ∧ ((70 / (d - 5)) = (110 / d)) ∧ (d - 5 = 8.75) :=
by
  sorry

end hiker_speeds_l1841_184146


namespace slower_train_speed_is_36_l1841_184107

def speed_of_slower_train (v : ℕ) : Prop :=
  let length_of_each_train := 100
  let distance_covered := length_of_each_train * 2
  let time_taken := 72
  let faster_train_speed := 46
  let relative_speed := (faster_train_speed - v) * (1000 / 3600)
  distance_covered = relative_speed * time_taken

theorem slower_train_speed_is_36 : ∃ v, speed_of_slower_train v ∧ v = 36 :=
by
  use 36
  unfold speed_of_slower_train
  -- Prove that the equation holds when v = 36
  sorry

end slower_train_speed_is_36_l1841_184107


namespace neg_one_to_zero_l1841_184125

theorem neg_one_to_zero : (-1 : ℤ)^0 = 1 := 
by 
  -- Expanding expressions and applying the exponent rule for non-zero numbers
  sorry

end neg_one_to_zero_l1841_184125


namespace smallest_mu_exists_l1841_184188

theorem smallest_mu_exists (a b c d : ℝ) (h : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d) :
  ∃ μ : ℝ, μ = (3 / 2) - (3 / (4 * Real.sqrt 2)) ∧ 
    (a^3 + b^3 + c^3 + d^3 ≥ a^2 * b + μ * b^2 * c + c^2 * d) :=
by
  sorry

end smallest_mu_exists_l1841_184188


namespace student_loses_one_mark_per_wrong_answer_l1841_184111

noncomputable def marks_lost_per_wrong_answer (x : ℝ) : Prop :=
  let total_questions := 60
  let correct_answers := 42
  let wrong_answers := total_questions - correct_answers
  let marks_per_correct := 4
  let total_marks := 150
  correct_answers * marks_per_correct - wrong_answers * x = total_marks

theorem student_loses_one_mark_per_wrong_answer : marks_lost_per_wrong_answer 1 :=
by
  sorry

end student_loses_one_mark_per_wrong_answer_l1841_184111


namespace total_muffins_correct_l1841_184197

-- Define the conditions
def boys_count := 3
def muffins_per_boy := 12
def girls_count := 2
def muffins_per_girl := 20

-- Define the question and answer
def total_muffins_for_sale : Nat :=
  boys_count * muffins_per_boy + girls_count * muffins_per_girl

theorem total_muffins_correct :
  total_muffins_for_sale = 76 := by
  sorry

end total_muffins_correct_l1841_184197


namespace ratio_quadrilateral_l1841_184132

theorem ratio_quadrilateral
  (ABCD_area : ℝ)
  (h_ABCD : ABCD_area = 40)
  (K L M N : Type)
  (AK KB : ℝ)
  (h_ratio : AK / KB = BL / LC ∧ BL / LC = CM / MD ∧ CM / MD = DN / NA)
  (KLMN_area : ℝ)
  (h_KLMN : KLMN_area = 25) :
  (AK / (AK + KB) = 1 / 4 ∨ AK / (AK + KB) = 3 / 4) :=
sorry

end ratio_quadrilateral_l1841_184132


namespace polygon_interior_angle_144_proof_l1841_184131

-- Definitions based on the conditions in the problem statement
def interior_angle (n : ℕ) : ℝ := 144
def sum_of_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

-- The problem statement as a Lean 4 theorem to prove n = 10
theorem polygon_interior_angle_144_proof : ∃ n : ℕ, interior_angle n = 144 ∧ sum_of_interior_angles n = n * 144 → n = 10 := by
  sorry

end polygon_interior_angle_144_proof_l1841_184131


namespace evaporated_water_l1841_184122

theorem evaporated_water 
  (E : ℝ)
  (h₁ : 0 < 10) -- initial mass is positive
  (h₂ : 10 * 0.3 + 10 * 0.7 = 3 + 7) -- Solution Y composition check
  (h₃ : (3 + 0.3 * E) / (10 - E + 0.7 * E) = 0.36) -- New solution composition
  : E = 0.9091 := 
sorry

end evaporated_water_l1841_184122


namespace car_speed_l1841_184154

theorem car_speed (v : ℝ) (h₁ : (1/75 * 3600) + 12 = 1/v * 3600) : v = 60 := 
by 
  sorry

end car_speed_l1841_184154


namespace correct_calculation_l1841_184159

theorem correct_calculation :
  (∀ (x y : ℝ), -3 * x - 3 * x ≠ 0) ∧
  (∀ (x : ℝ), x - 4 * x ≠ -3) ∧
  (∀ (x : ℝ), 2 * x + 3 * x^2 ≠ 5 * x^3) ∧
  (∀ (x y : ℝ), -4 * x * y + 3 * x * y = -x * y) :=
by
  sorry

end correct_calculation_l1841_184159


namespace total_amount_shared_l1841_184185

theorem total_amount_shared (A B C : ℕ) (h1 : A = 24) (h2 : 2 * A = 3 * B) (h3 : 8 * A = 4 * C) :
  A + B + C = 156 :=
sorry

end total_amount_shared_l1841_184185


namespace cube_sum_identity_l1841_184147

theorem cube_sum_identity (a b : ℝ) (h₁ : a + b = 12) (h₂ : a * b = 20) : a^3 + b^3 = 1008 := 
by 
 sorry

end cube_sum_identity_l1841_184147


namespace combined_money_l1841_184126

/-- Tom has a quarter the money of Nataly. Nataly has three times the money of Raquel.
     Sam has twice the money of Nataly. Raquel has $40. Prove that combined they have $430. -/
theorem combined_money : 
  ∀ (T R N S : ℕ), 
    (T = N / 4) ∧ 
    (N = 3 * R) ∧ 
    (S = 2 * N) ∧ 
    (R = 40) → 
    T + R + N + S = 430 := 
by
  sorry

end combined_money_l1841_184126


namespace ab_product_eq_2_l1841_184152

theorem ab_product_eq_2 (a b : ℝ) (h : (a + 1)^2 + (b + 2)^2 = 0) : a * b = 2 :=
by sorry

end ab_product_eq_2_l1841_184152


namespace arithmetic_seq_a12_l1841_184167

variable {a : ℕ → ℝ}

theorem arithmetic_seq_a12 :
  (∀ n, ∃ d, a (n + 1) = a n + d)
  ∧ a 5 + a 11 = 30
  ∧ a 4 = 7
  → a 12 = 23 :=
by
  sorry


end arithmetic_seq_a12_l1841_184167


namespace positive_difference_solutions_l1841_184127

theorem positive_difference_solutions (r₁ r₂ : ℝ) (h_r₁ : (r₁^2 - 5 * r₁ - 22) / (r₁ + 4) = 3 * r₁ + 8) (h_r₂ : (r₂^2 - 5 * r₂ - 22) / (r₂ + 4) = 3 * r₂ + 8) (h_r₁_ne : r₁ ≠ -4) (h_r₂_ne : r₂ ≠ -4) :
  |r₁ - r₂| = 3 / 2 := 
sorry


end positive_difference_solutions_l1841_184127


namespace lines_intersection_l1841_184158

theorem lines_intersection (a b : ℝ) : 
  (2 : ℝ) = (1/3 : ℝ) * (1 : ℝ) + a →
  (1 : ℝ) = (1/3 : ℝ) * (2 : ℝ) + b →
  a + b = 2 := 
by
  intros h₁ h₂
  sorry

end lines_intersection_l1841_184158


namespace first_instance_height_35_l1841_184195
noncomputable def projectile_height (t : ℝ) : ℝ := -5 * t^2 + 30 * t

theorem first_instance_height_35 {t : ℝ} (h : projectile_height t = 35) :
  t = 3 - Real.sqrt 2 :=
sorry

end first_instance_height_35_l1841_184195


namespace range_of_a_l1841_184181

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 2^x - a ≥ 0) ↔ (a ≤ 0) :=
by
  sorry

end range_of_a_l1841_184181


namespace original_weight_l1841_184133

namespace MarbleProblem

def remainingWeightAfterCuts (w : ℝ) : ℝ :=
  w * 0.70 * 0.70 * 0.85

theorem original_weight (w : ℝ) : remainingWeightAfterCuts w = 124.95 → w = 299.94 :=
by
  intros h
  sorry

end MarbleProblem

end original_weight_l1841_184133


namespace cups_per_serving_l1841_184149

-- Define the conditions
def total_cups : ℕ := 18
def servings : ℕ := 9

-- State the theorem to prove the answer
theorem cups_per_serving : total_cups / servings = 2 := by
  sorry

end cups_per_serving_l1841_184149


namespace min_total_cost_at_n_equals_1_l1841_184160

-- Define the conditions and parameters
variables (a : ℕ) -- The total construction area
variables (n : ℕ) -- The number of floors

-- Definitions based on the given problem conditions
def land_expropriation_cost : ℕ := 2388 * a
def construction_cost (n : ℕ) : ℕ :=
  if n = 0 then 0 else if n = 1 then 455 * a else (455 * n * a + 30 * (n-2) * (n-1) / 2 * a)

-- Total cost including land expropriation and construction costs
def total_cost (n : ℕ) : ℕ := land_expropriation_cost a + construction_cost a n

-- The minimum total cost occurs at n = 1
theorem min_total_cost_at_n_equals_1 :
  ∃ n, n = 1 ∧ total_cost a n = 2788 * a :=
by sorry

end min_total_cost_at_n_equals_1_l1841_184160


namespace max_distinct_subsets_l1841_184118

def T : Set ℕ := { x | 1 ≤ x ∧ x ≤ 999 }

theorem max_distinct_subsets (k : ℕ) (A : Fin k → Set ℕ) 
  (h : ∀ i j : Fin k, i < j → A i ∪ A j = T) : 
  k ≤ 1000 := 
sorry

end max_distinct_subsets_l1841_184118


namespace combined_average_score_clubs_l1841_184135

theorem combined_average_score_clubs
  (nA nB : ℕ) -- Number of members in each club
  (avgA avgB : ℝ) -- Average score of each club
  (hA : nA = 40)
  (hB : nB = 50)
  (hAvgA : avgA = 90)
  (hAvgB : avgB = 81) :
  (nA * avgA + nB * avgB) / (nA + nB) = 85 :=
by
  sorry -- Proof omitted

end combined_average_score_clubs_l1841_184135


namespace discount_correct_l1841_184196

variable {a : ℝ} (discount_percent : ℝ) (profit_percent : ℝ → ℝ)

noncomputable def calc_discount : ℝ :=
  discount_percent

theorem discount_correct :
  (discount_percent / 100) = (33 + 1 / 3) / 100 →
  profit_percent (discount_percent / 100) = (3 / 2) * (discount_percent / 100) →
  a * (1 - discount_percent / 100) * (1 + profit_percent (discount_percent / 100)) = a →
  discount_percent = 33 + 1 / 3 :=
by sorry

end discount_correct_l1841_184196


namespace best_of_five_advantageous_l1841_184106

theorem best_of_five_advantageous (p : ℝ) (h : p > 0.5) :
    let p1 := p^2 + 2 * p^2 * (1 - p)
    let p2 := p^3 + 3 * p^3 * (1 - p) + 6 * p^3 * (1 - p)^2
    p2 > p1 :=
by 
    let p1 := p^2 + 2 * p^2 * (1 - p)
    let p2 := p^3 + 3 * p^3 * (1 - p) + 6 * p^3 * (1 - p)^2
    sorry -- an actual proof would go here

end best_of_five_advantageous_l1841_184106


namespace students_with_certificates_l1841_184119

variable (C N : ℕ)

theorem students_with_certificates :
  (C + N = 120) ∧ (C = N + 36) → C = 78 :=
by
  sorry

end students_with_certificates_l1841_184119


namespace ice_cube_count_l1841_184199

theorem ice_cube_count (cubes_per_tray : ℕ) (tray_count : ℕ) (H1: cubes_per_tray = 9) (H2: tray_count = 8) :
  cubes_per_tray * tray_count = 72 :=
by
  sorry

end ice_cube_count_l1841_184199


namespace range_of_m_l1841_184117

theorem range_of_m {x m : ℝ} 
  (α : 2 / (x + 1) > 1) 
  (β : m ≤ x ∧ x ≤ 2) 
  (suff_condition : ∀ x, (2 / (x + 1) > 1) → (m ≤ x ∧ x ≤ 2)) :
  m ≤ -1 :=
sorry

end range_of_m_l1841_184117


namespace regression_equation_correct_l1841_184165

-- Defining the given data as constants
def x_data : List ℕ := [1, 2, 3, 4, 5, 6, 7]
def y_data : List ℕ := [891, 888, 351, 220, 200, 138, 112]

def sum_t_y : ℚ := 1586
def avg_t : ℚ := 0.37
def sum_t2_min7_avg_t2 : ℚ := 0.55

-- Defining the target regression equation
def target_regression (x : ℚ) : ℚ := 1000 / x + 30

-- Function to calculate the regression equation from data
noncomputable def calculate_regression (x_data y_data : List ℕ) : (ℚ → ℚ) :=
  let n : ℚ := x_data.length
  let avg_y : ℚ := y_data.sum / n
  let b : ℚ := (sum_t_y - n * avg_t * avg_y) / (sum_t2_min7_avg_t2)
  let a : ℚ := avg_y - b * avg_t
  fun x : ℚ => a + b / x

-- Theorem stating the regression equation matches the target regression equation
theorem regression_equation_correct :
  calculate_regression x_data y_data = target_regression :=
by
  sorry

end regression_equation_correct_l1841_184165


namespace calculate_rectangle_length_l1841_184137

theorem calculate_rectangle_length (side_of_square : ℝ) (width_of_rectangle : ℝ)
  (length_of_wire : ℝ) (perimeter_of_rectangle : ℝ) :
  side_of_square = 20 → 
  width_of_rectangle = 14 → 
  length_of_wire = 4 * side_of_square →
  perimeter_of_rectangle = length_of_wire →
  2 * (width_of_rectangle + length_of_rectangle) = perimeter_of_rectangle →
  length_of_rectangle = 26 :=
by
  intros
  sorry

end calculate_rectangle_length_l1841_184137


namespace falling_body_time_l1841_184163

theorem falling_body_time (g : ℝ) (h_g : g = 9.808) (d : ℝ) (t1 : ℝ) (h_d : d = 49.34) (h_t1 : t1 = 1.3) : 
  ∃ t : ℝ, (1 / 2 * g * (t + t1)^2 - 1 / 2 * g * t^2 = d) → t = 7.088 :=
by 
  use 7.088
  intros h
  sorry

end falling_body_time_l1841_184163


namespace subset_relation_l1841_184143

def P := {x : ℝ | x < 2}
def Q := {y : ℝ | y < 1}

theorem subset_relation : Q ⊆ P := 
by {
  sorry
}

end subset_relation_l1841_184143


namespace simplify_2A_minus_B_value_2A_minus_B_a_eq_neg1_b_eq_2_find_b_independent_of_a_l1841_184148

-- Definitions from the conditions
def A (a b : ℝ) : ℝ := 2 * a^2 - 5 * a * b + 3 * b
def B (a b : ℝ) : ℝ := 4 * a^2 + 6 * a * b + 8 * a

-- Part (1): Simplifying 2A - B
theorem simplify_2A_minus_B (a b : ℝ) : 
  2 * A a b - B a b = -16 * a * b + 6 * b - 8 * a := 
by
  sorry

-- Part (2): Finding 2A - B for specific a and b
theorem value_2A_minus_B_a_eq_neg1_b_eq_2 : 
  2 * A (-1) 2 - B (-1) 2 = 52 := 
by 
  sorry

-- Part (3): Finding b for which 2A - B is independent of a
theorem find_b_independent_of_a (a b : ℝ) (h : 2 * A a b - B a b = 6 * b) : 
  b = -1 / 2 := 
by
  sorry

end simplify_2A_minus_B_value_2A_minus_B_a_eq_neg1_b_eq_2_find_b_independent_of_a_l1841_184148


namespace final_digit_is_two_l1841_184104

-- Define initial conditions
def initial_ones : ℕ := 10
def initial_twos : ℕ := 10

-- Define the possible moves and the parity properties
def erase_identical (ones twos : ℕ) : ℕ × ℕ :=
  if ones ≥ 2 then (ones - 2, twos + 1)
  else (ones, twos - 1) -- for the case where two twos are removed

def erase_different (ones twos : ℕ) : ℕ × ℕ :=
  (ones, twos - 1)

-- Theorem stating that the final digit must be a two
theorem final_digit_is_two : 
∀ (ones twos : ℕ), ones = initial_ones → twos = initial_twos → 
(∃ n, ones + twos = n ∧ n = 1 ∧ (ones % 2 = 0)) → 
(∃ n, ones + twos = n ∧ n = 0 ∧ twos = 1) := 
by
  intros ones twos h_ones h_twos condition
  -- Constructing the proof should be done here
  sorry

end final_digit_is_two_l1841_184104


namespace charity_ticket_sales_l1841_184130

theorem charity_ticket_sales
  (x y p : ℕ)
  (h1 : x + y = 200)
  (h2 : x * p + y * (p / 2) = 3501)
  (h3 : x = 3 * y) :
  150 * 20 = 3000 :=
by
  sorry

end charity_ticket_sales_l1841_184130


namespace simplify_expression_l1841_184190

variable (a : ℝ) (ha : a ≠ 0)

theorem simplify_expression : (21 * a^3 - 7 * a) / (7 * a) = 3 * a^2 - 1 := by
  sorry

end simplify_expression_l1841_184190


namespace ratio_man_to_son_in_two_years_l1841_184166

-- Define current ages and the conditions
def son_current_age : ℕ := 24
def man_current_age : ℕ := son_current_age + 26

-- Define ages in two years
def son_age_in_two_years : ℕ := son_current_age + 2
def man_age_in_two_years : ℕ := man_current_age + 2

-- State the theorem
theorem ratio_man_to_son_in_two_years : 
  (man_age_in_two_years : ℚ) / (son_age_in_two_years : ℚ) = 2 :=
by sorry

end ratio_man_to_son_in_two_years_l1841_184166


namespace company_p_percentage_increase_l1841_184191

theorem company_p_percentage_increase :
  (460 - 400.00000000000006) / 400.00000000000006 * 100 = 15 := 
by
  sorry

end company_p_percentage_increase_l1841_184191


namespace problem_find_f_l1841_184194

noncomputable def f (x : ℝ) : ℝ := sorry

theorem problem_find_f {k : ℝ} :
  (∀ x : ℝ, x * (f (x + 1) - f x) = f x) →
  (∀ x y : ℝ, |f x - f y| ≤ |x - y|) →
  (∀ x : ℝ, 0 < x → f x = k * x) :=
by
  intro h1 h2
  apply sorry

end problem_find_f_l1841_184194


namespace numbers_from_five_threes_l1841_184153

theorem numbers_from_five_threes :
  (∃ (a b c d e : ℤ), (3*a + 3*b + 3*c + 3*d + 3*e = 11 ∨ 
                        3*a + 3*b + 3*c + 3*d + 3*e = 12 ∨ 
                        3*a + 3*b + 3*c + 3*d + 3*e = 13 ∨ 
                        3*a + 3*b + 3*c + 3*d + 3*e = 14 ∨ 
                        3*a + 3*b + 3*c + 3*d + 3*e = 15) ) :=
by
  -- Proof provided by the problem statement steps, using:
  -- 11 = (33/3)
  -- 12 = 3 * 3 + 3 + 3 - 3
  -- 13 = 3 * 3 + 3 + 3/3
  -- 14 = (33 + 3 * 3) / 3
  -- 15 = 3 + 3 + 3 + 3 + 3
  sorry

end numbers_from_five_threes_l1841_184153


namespace ellas_quadratic_equation_l1841_184121

theorem ellas_quadratic_equation (d e : ℤ) :
  (∀ x : ℤ, |x - 8| = 3 → (x = 11 ∨ x = 5)) →
  (∀ x : ℤ, (x = 11 ∨ x = 5) → x^2 + d * x + e = 0) →
  (d, e) = (-16, 55) :=
by
  intro h1 h2
  sorry

end ellas_quadratic_equation_l1841_184121
