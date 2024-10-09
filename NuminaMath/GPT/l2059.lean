import Mathlib

namespace total_spent_on_entertainment_l2059_205925

def cost_of_computer_game : ℕ := 66
def cost_of_one_movie_ticket : ℕ := 12
def number_of_movie_tickets : ℕ := 3

theorem total_spent_on_entertainment : cost_of_computer_game + cost_of_one_movie_ticket * number_of_movie_tickets = 102 := 
by sorry

end total_spent_on_entertainment_l2059_205925


namespace range_of_a_l2059_205987

variable (a : ℝ)

def p : Prop := ∃ x₀ : ℝ, x₀^2 - a * x₀ + 1 = 0

def q : Prop := ∀ x : ℝ, x ≥ 0 → x^2 - 2 * a * x + a^2 + 1 ≥ 1

theorem range_of_a : ¬(p a ∨ q a) → -2 < a ∧ a < 0 := by
  sorry

end range_of_a_l2059_205987


namespace brownies_maximum_l2059_205927

theorem brownies_maximum (m n : ℕ) (h1 : (m - 2) * (n - 2) = 2 * (2 * m + 2 * n - 4)) :
  m * n ≤ 144 :=
sorry

end brownies_maximum_l2059_205927


namespace myopia_relation_l2059_205979

def myopia_data := 
  [(1.00, 100), (0.50, 200), (0.25, 400), (0.20, 500), (0.10, 1000)]

noncomputable def myopia_function (x : ℝ) : ℝ :=
  100 / x

theorem myopia_relation (h₁ : 100 = (1.00 : ℝ) * 100)
    (h₂ : 100 = (0.50 : ℝ) * 200)
    (h₃ : 100 = (0.25 : ℝ) * 400)
    (h₄ : 100 = (0.20 : ℝ) * 500)
    (h₅ : 100 = (0.10 : ℝ) * 1000) :
  (∀ x > 0, myopia_function x = 100 / x) ∧ (myopia_function 250 = 0.4) :=
by
  sorry

end myopia_relation_l2059_205979


namespace vertices_of_equilateral_triangle_l2059_205921

noncomputable def a : ℝ := 52 / 3
noncomputable def b : ℝ := -13 / 3 - 15 * Real.sqrt 3 / 2

theorem vertices_of_equilateral_triangle (a b : ℝ)
  (h₀ : (0, 0) = (0, 0))
  (h₁ : (a, 15) = (52 / 3, 15))
  (h₂ : (b, 41) = (-13 / 3 - 15 * Real.sqrt 3 / 2, 41)) :
  a * b = -676 / 9 := 
by
  sorry

end vertices_of_equilateral_triangle_l2059_205921


namespace biology_physics_ratio_l2059_205990

theorem biology_physics_ratio (boys_bio : ℕ) (girls_bio : ℕ) (total_bio : ℕ) (total_phys : ℕ) 
  (h1 : boys_bio = 25) 
  (h2 : girls_bio = 3 * boys_bio) 
  (h3 : total_bio = boys_bio + girls_bio) 
  (h4 : total_phys = 200) : 
  total_bio / total_phys = 1 / 2 :=
by
  sorry

end biology_physics_ratio_l2059_205990


namespace A_B_days_together_l2059_205955

variable (W : ℝ) -- total work
variable (x : ℝ) -- days A and B worked together
variable (A_B_rate : ℝ) -- combined work rate of A and B
variable (A_rate : ℝ) -- work rate of A
variable (B_days : ℝ) -- days A worked alone after B left

-- Conditions:
axiom condition1 : A_B_rate = W / 40
axiom condition2 : A_rate = W / 80
axiom condition3 : B_days = 6
axiom condition4 : (x * A_B_rate + B_days * A_rate = W)

-- We want to prove that x = 37:
theorem A_B_days_together : x = 37 :=
by
  sorry

end A_B_days_together_l2059_205955


namespace cost_of_fencing_field_l2059_205910

def ratio (a b : ℕ) : Prop := ∃ k : ℕ, (b = k * a)

def assume_fields : Prop :=
  ∃ (x : ℚ), (ratio 3 4) ∧ (3 * 4 * x^2 = 9408) ∧ (0.25 > 0)

theorem cost_of_fencing_field :
  assume_fields → 98 = 98 := by
  sorry

end cost_of_fencing_field_l2059_205910


namespace alice_ride_average_speed_l2059_205905

theorem alice_ride_average_speed
    (d1 d2 : ℝ) 
    (s1 s2 : ℝ)
    (h_d1 : d1 = 40)
    (h_d2 : d2 = 20)
    (h_s1 : s1 = 8)
    (h_s2 : s2 = 40) :
    (d1 + d2) / (d1 / s1 + d2 / s2) = 10.909 :=
by
  simp [h_d1, h_d2, h_s1, h_s2]
  norm_num
  sorry

end alice_ride_average_speed_l2059_205905


namespace probability_two_balls_red_l2059_205959

variables (total_balls red_balls blue_balls green_balls picked_balls : ℕ)

def probability_of_both_red
  (h_total_balls : total_balls = 8)
  (h_red_balls : red_balls = 3)
  (h_blue_balls : blue_balls = 2)
  (h_green_balls : green_balls = 3)
  (h_picked_balls : picked_balls = 2) : ℚ :=
  (red_balls / total_balls) * ((red_balls - 1) / (total_balls - 1))

theorem probability_two_balls_red (h_total_balls : total_balls = 8)
  (h_red_balls : red_balls = 3)
  (h_blue_balls : blue_balls = 2)
  (h_green_balls : green_balls = 3)
  (h_picked_balls : picked_balls = 2)
  (h_prob : probability_of_both_red total_balls red_balls blue_balls green_balls picked_balls 
    h_total_balls h_red_balls h_blue_balls h_green_balls h_picked_balls = 3 / 28) : 
  probability_of_both_red total_balls red_balls blue_balls green_balls picked_balls 
    h_total_balls h_red_balls h_blue_balls h_green_balls h_picked_balls = 3 / 28 := 
sorry

end probability_two_balls_red_l2059_205959


namespace time_to_count_envelopes_l2059_205936

theorem time_to_count_envelopes (r : ℕ) : (r / 10 = 1) → (r * 60 / r = 60) ∧ (r * 90 / r = 90) :=
by sorry

end time_to_count_envelopes_l2059_205936


namespace number_of_candies_bought_on_Tuesday_l2059_205965

theorem number_of_candies_bought_on_Tuesday (T : ℕ) 
  (thursday_candies : ℕ := 5) 
  (friday_candies : ℕ := 2) 
  (candies_left : ℕ := 4) 
  (candies_eaten : ℕ := 6) 
  (total_initial_candies : T + thursday_candies + friday_candies = candies_left + candies_eaten) 
  : T = 3 := by
  sorry

end number_of_candies_bought_on_Tuesday_l2059_205965


namespace time_to_meet_in_minutes_l2059_205982

def distance_between_projectiles : ℕ := 1998
def speed_projectile_1 : ℕ := 444
def speed_projectile_2 : ℕ := 555

theorem time_to_meet_in_minutes : 
  (distance_between_projectiles / (speed_projectile_1 + speed_projectile_2)) * 60 = 120 := 
by
  sorry

end time_to_meet_in_minutes_l2059_205982


namespace blocks_difference_l2059_205984

def blocks_house := 89
def blocks_tower := 63

theorem blocks_difference : (blocks_house - blocks_tower = 26) :=
by sorry

end blocks_difference_l2059_205984


namespace intersection_with_complement_l2059_205904

-- Define the universal set U, sets A and B
def U : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {0, 2, 4}
def B : Set ℕ := {1, 4}

-- Define the complement of B with respect to U
def complement_B : Set ℕ := { x | x ∈ U ∧ x ∉ B }

-- The equivalent proof problem in Lean 4
theorem intersection_with_complement :
  A ∩ complement_B = {0, 2} :=
by
  sorry

end intersection_with_complement_l2059_205904


namespace direct_proportion_solution_l2059_205934

theorem direct_proportion_solution (m : ℝ) (h1 : m + 3 ≠ 0) (h2 : m^2 - 8 = 1) : m = 3 :=
sorry

end direct_proportion_solution_l2059_205934


namespace chickens_in_zoo_l2059_205926

theorem chickens_in_zoo (c e : ℕ) (h_legs : 2 * c + 4 * e = 66) (h_heads : c + e = 24) : c = 15 :=
by
  sorry

end chickens_in_zoo_l2059_205926


namespace wood_not_heavier_than_brick_l2059_205935

-- Define the weights of the wood and the brick
def block_weight_kg : ℝ := 8
def brick_weight_g : ℝ := 8000

-- Conversion function from kg to g
def kg_to_g (kg : ℝ) : ℝ := kg * 1000

-- State the proof problem
theorem wood_not_heavier_than_brick : ¬ (kg_to_g block_weight_kg > brick_weight_g) :=
by
  -- Begin the proof
  sorry

end wood_not_heavier_than_brick_l2059_205935


namespace find_x_l2059_205920

def operation_eur (x y : ℕ) : ℕ := 3 * x * y

theorem find_x (y x : ℕ) (h1 : y = 3) (h2 : operation_eur y (operation_eur x 5) = 540) : x = 4 :=
by
  sorry

end find_x_l2059_205920


namespace range_is_correct_l2059_205949

noncomputable def quadratic_function (x : ℝ) : ℝ := x^2 - 4 * x

def domain : Set ℝ := {x | -3 ≤ x ∧ x ≤ 3}

def range_of_function : Set ℝ := {y | ∃ x ∈ domain, quadratic_function x = y}

theorem range_is_correct : range_of_function = Set.Icc (-4) 21 :=
by {
  sorry
}

end range_is_correct_l2059_205949


namespace degree_of_product_l2059_205988

-- Definitions for the conditions
def isDegree (p : Polynomial ℝ) (n : ℕ) : Prop :=
  p.degree = n

variable {h j : Polynomial ℝ}

-- Given conditions
axiom h_deg : isDegree h 3
axiom j_deg : isDegree j 6

-- The theorem to prove
theorem degree_of_product : h.degree = 3 → j.degree = 6 → (Polynomial.degree (Polynomial.comp h (Polynomial.X ^ 4) * Polynomial.comp j (Polynomial.X ^ 3)) = 30) :=
by
  intros h3 j6
  sorry

end degree_of_product_l2059_205988


namespace probability_of_sunglasses_given_caps_l2059_205912

theorem probability_of_sunglasses_given_caps
  (s c sc : ℕ) 
  (h₀ : s = 60) 
  (h₁ : c = 40)
  (h₂ : sc = 20)
  (h₃ : sc = 1 / 3 * s) : 
  (sc / c) = 1 / 2 :=
by
  sorry

end probability_of_sunglasses_given_caps_l2059_205912


namespace sum_inverse_terms_l2059_205995

theorem sum_inverse_terms : 
  (∑' n : ℕ, if n = 0 then (0 : ℝ) else (1 / (n * (n + 3) : ℝ))) = 11 / 18 :=
by {
  -- proof to be filled in
  sorry
}

end sum_inverse_terms_l2059_205995


namespace find_fraction_l2059_205907

theorem find_fraction (f : ℝ) (h₁ : f * 50.0 - 4 = 6) : f = 0.2 :=
by
  sorry

end find_fraction_l2059_205907


namespace max_digit_product_l2059_205915

theorem max_digit_product (N : ℕ) (digits : List ℕ) (h1 : 0 < N) (h2 : digits.sum = 23) (h3 : digits.prod < 433) : 
  digits.prod ≤ 432 :=
sorry

end max_digit_product_l2059_205915


namespace plant_lamp_arrangements_l2059_205970

/-- Rachel has two identical basil plants and an aloe plant.
Additionally, she has two identical white lamps, two identical red lamps, and 
two identical blue lamps she can put each plant under 
(she can put more than one plant under a lamp, but each plant is under exactly one lamp). 
-/
theorem plant_lamp_arrangements : 
  let plants := ["basil", "basil", "aloe"]
  let lamps := ["white", "white", "red", "red", "blue", "blue"]
  ∃ n, n = 27 := by
  sorry

end plant_lamp_arrangements_l2059_205970


namespace determine_sum_of_squares_l2059_205948

theorem determine_sum_of_squares
  (x y z : ℝ)
  (h1 : x + y + z = 13)
  (h2 : x * y * z = 72)
  (h3 : 1/x + 1/y + 1/z = 3/4) :
  x^2 + y^2 + z^2 = 61 := 
sorry

end determine_sum_of_squares_l2059_205948


namespace sin_cos_identity_l2059_205981

variable {α : ℝ}

/-- Given 1 / sin(α) + 1 / cos(α) = √3, then sin(α) * cos(α) = -1 / 3 -/
theorem sin_cos_identity (h : 1 / Real.sin α + 1 / Real.cos α = Real.sqrt 3) : 
  Real.sin α * Real.cos α = -1 / 3 := 
sorry

end sin_cos_identity_l2059_205981


namespace series_2023_power_of_3_squared_20_equals_653_l2059_205986

def series (A : ℕ → ℕ) : Prop :=
  A 0 = 1 ∧ 
  ∀ n > 0, 
  A n = A (n / 2023) + A (n / 2023^2) + A (n / 2023^3)

theorem series_2023_power_of_3_squared_20_equals_653 (A : ℕ → ℕ) (h : series A) : A (2023 ^ (3^2) + 20) = 653 :=
by
  -- placeholder for proof
  sorry

end series_2023_power_of_3_squared_20_equals_653_l2059_205986


namespace parametric_to_ordinary_l2059_205923

theorem parametric_to_ordinary (θ : ℝ) (x y : ℝ) : 
  x = Real.cos θ ^ 2 →
  y = 2 * Real.sin θ ^ 2 →
  (x ∈ Set.Icc 0 1) → 
  2 * x + y - 2 = 0 :=
by
  intros hx hy h_range
  sorry

end parametric_to_ordinary_l2059_205923


namespace shirley_sold_10_boxes_l2059_205937

variable (cases boxes_per_case : ℕ)

-- Define the conditions
def number_of_cases := 5
def boxes_in_each_case := 2

-- Prove the total number of boxes is 10
theorem shirley_sold_10_boxes (H1 : cases = number_of_cases) (H2 : boxes_per_case = boxes_in_each_case) :
  cases * boxes_per_case = 10 := by
  sorry

end shirley_sold_10_boxes_l2059_205937


namespace car_speed_in_kmph_l2059_205941

def speed_mps : ℝ := 10  -- The speed of the car in meters per second
def conversion_factor : ℝ := 3.6  -- The conversion factor from m/s to km/h

theorem car_speed_in_kmph : speed_mps * conversion_factor = 36 := 
by
  sorry

end car_speed_in_kmph_l2059_205941


namespace points_above_line_l2059_205963

theorem points_above_line {t : ℝ} (hP : 1 + t - 1 > 0) (hQ : t^2 + (t - 1) - 1 > 0) : t > 1 :=
by
  sorry

end points_above_line_l2059_205963


namespace sides_ratio_of_arithmetic_sequence_l2059_205951

theorem sides_ratio_of_arithmetic_sequence (A B C : ℝ) (a b c : ℝ) 
  (h_arith_sequence : (A = B - (B - C)) ∧ (B = C + (C - A))) 
  (h_angle_B : B = 60)  
  (h_cosine_rule : a^2 + c^2 - b^2 = 2 * a * c * (Real.cos B)) :
  (1 / (a + b) + 1 / (b + c) = 3 / (a + b + c)) :=
sorry

end sides_ratio_of_arithmetic_sequence_l2059_205951


namespace boat_speed_in_still_water_l2059_205916

/-- In one hour, a boat goes 9 km along the stream and 5 km against the stream.
Prove that the speed of the boat in still water is 7 km/hr. -/
theorem boat_speed_in_still_water (B S : ℝ) 
  (h1 : B + S = 9) 
  (h2 : B - S = 5) : 
  B = 7 :=
by
  sorry

end boat_speed_in_still_water_l2059_205916


namespace average_of_remaining_two_numbers_l2059_205996

theorem average_of_remaining_two_numbers
  (a b c d e f : ℝ)
  (h1 : (a + b + c + d + e + f) / 6 = 3.9)
  (h2 : (a + b) / 2 = 3.4)
  (h3 : (c + d) / 2 = 3.85) :
  (e + f) / 2 = 4.45 :=
sorry

end average_of_remaining_two_numbers_l2059_205996


namespace remainder_mod_7_l2059_205968

theorem remainder_mod_7 : (9^7 + 8^8 + 7^9) % 7 = 3 :=
by sorry

end remainder_mod_7_l2059_205968


namespace cos_sin_equation_solution_l2059_205946

noncomputable def solve_cos_sin_equation (x : ℝ) (n : ℤ) : Prop :=
  let lhs := (Real.cos x) / (Real.sqrt 3)
  let rhs := Real.sqrt ((1 - (Real.cos (2*x)) - 2 * (Real.sin x)^3) / (6 * Real.sin x - 2))
  (lhs = rhs) ∧ (Real.cos x ≥ 0)

theorem cos_sin_equation_solution:
  (∃ (x : ℝ) (n : ℤ), solve_cos_sin_equation x n) ↔ 
  ∃ (n : ℤ), (x = (π / 2) + 2 * π * n) ∨ (x = (π / 6) + 2 * π * n) :=
by
  sorry

end cos_sin_equation_solution_l2059_205946


namespace smallest_norm_value_l2059_205911

theorem smallest_norm_value (w : ℝ × ℝ)
  (h : ‖(w.1 + 4, w.2 + 2)‖ = 10) :
  ‖w‖ = 10 - 2*Real.sqrt 5 := sorry

end smallest_norm_value_l2059_205911


namespace blue_pill_cost_l2059_205930

theorem blue_pill_cost
  (days : ℕ)
  (total_cost : ℤ)
  (cost_diff : ℤ)
  (daily_cost : ℤ)
  (y : ℤ) : 
  days = 21 →
  total_cost = 966 →
  cost_diff = 2 →
  daily_cost = total_cost / days →
  daily_cost = 46 →
  2 * y - cost_diff = daily_cost →
  y = 24 := 
by
  intros days_eq total_cost_eq cost_diff_eq daily_cost_eq d_cost_eq daily_eq_46;
  sorry

end blue_pill_cost_l2059_205930


namespace initial_pipes_count_l2059_205901

theorem initial_pipes_count (n : ℕ) (r : ℝ) :
  n * r = 1 / 16 → (n + 15) * r = 1 / 4 → n = 5 :=
by
  intro h1 h2
  sorry

end initial_pipes_count_l2059_205901


namespace find_a1_l2059_205967

theorem find_a1 (a : ℕ → ℕ) (h1 : a 5 = 14) (h2 : ∀ n, a (n+1) - a n = n + 1) : a 1 = 0 :=
by
  sorry

end find_a1_l2059_205967


namespace line_parallel_not_passing_through_point_l2059_205957

noncomputable def point_outside_line (A B C x0 y0 : ℝ) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ (A * x0 + B * y0 + C = k)

theorem line_parallel_not_passing_through_point 
  (A B C x0 y0 : ℝ) (h : point_outside_line A B C x0 y0) :
  ∃ k : ℝ, k ≠ 0 ∧ (∀ x y : ℝ, Ax + By + C + k = 0 → Ax_0 + By_0 + C + k ≠ 0) :=
sorry

end line_parallel_not_passing_through_point_l2059_205957


namespace permutations_PERCEPTION_l2059_205974

-- Define the word "PERCEPTION" and its letter frequencies
def word : String := "PERCEPTION"

def freq_P : Nat := 2
def freq_E : Nat := 2
def freq_R : Nat := 1
def freq_C : Nat := 1
def freq_T : Nat := 1
def freq_I : Nat := 1
def freq_O : Nat := 1
def freq_N : Nat := 1

-- Define the total number of letters in the word
def total_letters : Nat := 10

-- Calculate the number of permutations for the multiset
def permutations : Nat :=
  total_letters.factorial / (freq_P.factorial * freq_E.factorial)

-- Proof problem
theorem permutations_PERCEPTION :
  permutations = 907200 :=
by
  sorry

end permutations_PERCEPTION_l2059_205974


namespace calculate_fraction_l2059_205944

theorem calculate_fraction (x y : ℚ) (h1 : x = 5 / 6) (h2 : y = 6 / 5) : (1 / 3) * x^8 * y^9 = 2 / 5 := by
  sorry

end calculate_fraction_l2059_205944


namespace center_of_circumcircle_lies_on_AK_l2059_205932

variable {α β γ : Real} -- Angles in triangle ABC
variable (A B C L H K O : Point) -- Points in the configuration
variable (circumcircle_ABC : TriangularCircumcircle A B C) -- Circumcircle of triangle ABC

-- Definitions based on the given conditions
variable (is_angle_bisector : angle_bisector A B C L)
variable (is_height : height_from_point_to_line B A L H)
variable (intersects_circle_at_K : intersects_circumcircle A B L K circumcircle_ABC)
variable (is_circumcenter : circumcenter A B C O circumcircle_ABC)

theorem center_of_circumcircle_lies_on_AK
  (h_angle_bisector : is_angle_bisector)
  (h_height : is_height)
  (h_intersects_circle_at_K : intersects_circle_at_K)
  (h_circumcenter : is_circumcenter) 
    : lies_on_line O A K := 
sorry -- Proof is omitted

end center_of_circumcircle_lies_on_AK_l2059_205932


namespace complement_of_angle_l2059_205983

theorem complement_of_angle (α : ℝ) (h : α = 23 + 36 / 60) : 180 - α = 156.4 := 
by
  sorry

end complement_of_angle_l2059_205983


namespace opposite_number_subtraction_l2059_205933

variable (a b : ℝ)

theorem opposite_number_subtraction : -(a - b) = b - a := 
sorry

end opposite_number_subtraction_l2059_205933


namespace people_in_each_column_l2059_205943

theorem people_in_each_column
  (P : ℕ)
  (x : ℕ)
  (h1 : P = 16 * x)
  (h2 : P = 12 * 40) :
  x = 30 :=
sorry

end people_in_each_column_l2059_205943


namespace fifth_friend_paid_13_l2059_205940

noncomputable def fifth_friend_payment (a b c d e : ℝ) : Prop :=
a = (1/3) * (b + c + d + e) ∧
b = (1/4) * (a + c + d + e) ∧
c = (1/5) * (a + b + d + e) ∧
a + b + c + d + e = 120 ∧
e = 13

theorem fifth_friend_paid_13 : 
  ∃ (a b c d e : ℝ), fifth_friend_payment a b c d e := 
sorry

end fifth_friend_paid_13_l2059_205940


namespace part_I_period_part_I_monotonicity_interval_part_II_range_l2059_205942

noncomputable def f (x : ℝ) : ℝ :=
  4 * Real.cos x * Real.sin (x + Real.pi / 6) - 1

theorem part_I_period : ∀ x, f (x + Real.pi) = f x := by
  sorry

theorem part_I_monotonicity_interval (k : ℤ) :
  ∀ x, k * Real.pi + Real.pi / 6 ≤ x ∧ x ≤ k * Real.pi + 2 * Real.pi / 3 → f (x + Real.pi) = f x := by
  sorry

theorem part_II_range :
  ∀ x, -Real.pi / 6 ≤ x ∧ x ≤ Real.pi / 4 → f x ∈ Set.Icc (-1) 2 := by
  sorry

end part_I_period_part_I_monotonicity_interval_part_II_range_l2059_205942


namespace solution_one_solution_two_solution_three_l2059_205985

open Real

noncomputable def problem_one (a b : ℝ) (cosA : ℝ) : ℝ :=
if a = sqrt 6 ∧ b = 2 * 1 ∧ cosA = -1/4 then 1 else 0

theorem solution_one (a b : ℝ) (cosA : ℝ) :
  a = sqrt 6 → b = 2 * 1 → cosA = -1/4 → problem_one a b cosA = 1 := by
  intros ha hb hcos
  unfold problem_one
  simp [ha, hb, hcos]

noncomputable def problem_two (a b : ℝ) (cosA sinB : ℝ) : ℝ :=
if a = sqrt 6 ∧ b = 2 * 1 ∧ cosA = -1/4 ∧ sinB = sqrt 10 / 4 then sqrt 10 / 4 else 0

theorem solution_two (a b : ℝ) (cosA sinB : ℝ) :
  a = sqrt 6 → b = 2 * 1 → cosA = -1/4 → sinB = sqrt 10 / 4 → problem_two a b cosA sinB = sqrt 10 / 4 := by
  intros ha hb hcos hsinB
  unfold problem_two
  simp [ha, hb, hcos, hsinB]

noncomputable def problem_three (a b : ℝ) (cosA sinB sin2AminusB : ℝ) : ℝ :=
if a = sqrt 6 ∧ b = 2 * 1 ∧ cosA = -1/4 ∧ sinB = sqrt 10 / 4 ∧ sin2AminusB = sqrt 10 / 8 then sqrt 10 / 8 else 0

theorem solution_three (a b : ℝ) (cosA sinB sin2AminusB : ℝ) :
  a = sqrt 6 → b = 2 * 1 → cosA = -1/4 → sinB = sqrt 10 / 4 → sin2AminusB = sqrt 10 / 8 → problem_three a b cosA sinB sin2AminusB = sqrt 10 / 8 := by
  intros ha hb hcos hsinB hsin2AminusB
  unfold problem_three
  simp [ha, hb, hcos, hsinB, hsin2AminusB]

end solution_one_solution_two_solution_three_l2059_205985


namespace dandelion_average_l2059_205956

theorem dandelion_average :
  let Billy_initial := 36
  let George_initial := Billy_initial / 3
  let Billy_total := Billy_initial + 10
  let George_total := George_initial + 10
  let total := Billy_total + George_total
  let average := total / 2
  average = 34 :=
by
  -- placeholder for the proof
  sorry

end dandelion_average_l2059_205956


namespace coin_flip_sequences_l2059_205900

theorem coin_flip_sequences (n : ℕ) (h : n = 10) : (2 ^ n) = 1024 := by
  rw [h]
  -- 1024 is 2 ^ 10
  norm_num

end coin_flip_sequences_l2059_205900


namespace carrie_money_left_l2059_205962

/-- Carrie was given $91. She bought a sweater for $24, 
    a T-shirt for $6, a pair of shoes for $11,
    and a pair of jeans originally costing $30 with a 25% discount. 
    Prove that she has $27.50 left. -/
theorem carrie_money_left :
  let init_money := 91
  let sweater := 24
  let t_shirt := 6
  let shoes := 11
  let jeans := 30
  let discount := 25 / 100
  let jeans_discounted_price := jeans * (1 - discount)
  let total_cost := sweater + t_shirt + shoes + jeans_discounted_price
  let money_left := init_money - total_cost
  money_left = 27.50 :=
by
  intros
  sorry

end carrie_money_left_l2059_205962


namespace circle_area_from_equation_l2059_205992

theorem circle_area_from_equation :
  (∀ x y : ℝ, x^2 + y^2 - 4 * x + 6 * y = -9) →
  ∃ (r : ℝ), (r = 2) ∧
    (∃ (A : ℝ), A = π * r^2 ∧ A = 4 * π) :=
by {
  -- Conditions included as hypothesis
  sorry -- Proof to be provided here
}

end circle_area_from_equation_l2059_205992


namespace trapezoid_diagonal_is_8sqrt5_trapezoid_leg_is_4sqrt5_l2059_205939

namespace Trapezoid

def isosceles_trapezoid (AD BC : ℝ) := 
  AD = 20 ∧ BC = 12

def diagonal (AD BC : ℝ) (AC : ℝ) := 
  isosceles_trapezoid AD BC → AC = 8 * Real.sqrt 5

def leg (AD BC : ℝ) (CD : ℝ) := 
  isosceles_trapezoid AD BC → CD = 4 * Real.sqrt 5

theorem trapezoid_diagonal_is_8sqrt5 (AD BC AC : ℝ) : 
  diagonal AD BC AC :=
by
  intros
  sorry

theorem trapezoid_leg_is_4sqrt5 (AD BC CD : ℝ) : 
  leg AD BC CD :=
by
  intros
  sorry

end Trapezoid

end trapezoid_diagonal_is_8sqrt5_trapezoid_leg_is_4sqrt5_l2059_205939


namespace train_boxcars_capacity_l2059_205945

theorem train_boxcars_capacity :
  let red_boxcars := 3
  let blue_boxcars := 4
  let black_boxcars := 7
  let black_capacity := 4000
  let blue_capacity := black_capacity * 2
  let red_capacity := blue_capacity * 3
  (black_boxcars * black_capacity) + (blue_boxcars * blue_capacity) + (red_boxcars * red_capacity) = 132000 := by
  sorry

end train_boxcars_capacity_l2059_205945


namespace walking_speed_proof_l2059_205961

-- Definitions based on the problem's conditions
def rest_time_per_period : ℕ := 5
def distance_per_rest : ℕ := 10
def total_distance : ℕ := 50
def total_time : ℕ := 320

-- The man's walking speed
def walking_speed (distance : ℕ) (time : ℕ) : ℕ := distance / time

-- The main statement to be proved
theorem walking_speed_proof : 
  walking_speed total_distance ((total_time - ((total_distance / distance_per_rest) * rest_time_per_period)) / 60) = 10 := 
by
  sorry

end walking_speed_proof_l2059_205961


namespace length_stationary_l2059_205994

def speed : ℝ := 64.8
def time_pole : ℝ := 5
def time_stationary : ℝ := 25

def length_moving : ℝ := speed * time_pole
def length_combined : ℝ := speed * time_stationary

theorem length_stationary : length_combined - length_moving = 1296 :=
by
  sorry

end length_stationary_l2059_205994


namespace game_promises_total_hours_l2059_205914

open Real

noncomputable def total_gameplay_hours (T : ℝ) : Prop :=
  let boring_gameplay := 0.80 * T
  let enjoyable_gameplay := 0.20 * T
  let expansion_hours := 30
  (enjoyable_gameplay + expansion_hours = 50) → (T = 100)

theorem game_promises_total_hours (T : ℝ) : total_gameplay_hours T :=
  sorry

end game_promises_total_hours_l2059_205914


namespace cubic_sum_l2059_205975

theorem cubic_sum (a b c : ℝ) (h1 : a + b + c = 0) (h2 : a * b + a * c + b * c = 1) (h3 : a * b * c = -2) : 
  a^3 + b^3 + c^3 = -6 :=
sorry

end cubic_sum_l2059_205975


namespace coplanar_points_l2059_205966

theorem coplanar_points (a : ℝ) :
  ∀ (V : ℝ), V = 2 + a^3 → V = 0 → a = -((2:ℝ)^(1/3)) :=
by
  sorry

end coplanar_points_l2059_205966


namespace wrapping_paper_area_correct_l2059_205973

variable (w h : ℝ) -- Define the base length and height of the box.

-- Lean statement for the problem asserting that the area of the wrapping paper is \(2(w+h)^2\).
def wrapping_paper_area (w h : ℝ) : ℝ := 2 * (w + h) ^ 2

-- Theorem stating that the derived formula for the area of the wrapping paper is correct.
theorem wrapping_paper_area_correct (w h : ℝ) : wrapping_paper_area w h = 2 * (w + h) ^ 2 :=
by
  sorry -- Proof is omitted

end wrapping_paper_area_correct_l2059_205973


namespace problem_conditions_and_inequalities_l2059_205918

open Real

theorem problem_conditions_and_inequalities (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_eq : a + 2 * b = a * b) :
  (a + 2 * b ≥ 8) ∧ (2 * a + b ≥ 9) ∧ (a ^ 2 + 4 * b ^ 2 + 5 * a * b ≥ 72) ∧ ¬(logb 2 a + logb 2 b < 3) :=
by
  sorry

end problem_conditions_and_inequalities_l2059_205918


namespace even_function_l2059_205947

noncomputable def f : ℝ → ℝ :=
sorry

theorem even_function (f : ℝ → ℝ) (h1 : ∀ x, f x = f (-x)) 
  (h2 : ∀ x, -1 ≤ x ∧ x ≤ 0 → f x = x - 1) : f (1/2) = -3/2 :=
sorry

end even_function_l2059_205947


namespace find_g_53_l2059_205964

variable (g : ℝ → ℝ)

axiom functional_eq (x y : ℝ) : g (x * y) = y * g x
axiom g_one : g 1 = 10

theorem find_g_53 : g 53 = 530 :=
by
  sorry

end find_g_53_l2059_205964


namespace contrapositive_of_proposition_is_false_l2059_205993

theorem contrapositive_of_proposition_is_false (x y : ℝ) 
  (h₀ : (x + y > 0) → (x > 0 ∧ y > 0)) : 
  ¬ ((x ≤ 0 ∨ y ≤ 0) → (x + y ≤ 0)) :=
by
  sorry

end contrapositive_of_proposition_is_false_l2059_205993


namespace turtle_feeding_cost_l2059_205919

theorem turtle_feeding_cost (total_weight_pounds : ℝ) (food_per_half_pound : ℝ)
  (jar_food_ounces : ℝ) (jar_cost_dollars : ℝ) (total_cost : ℝ) : 
  total_weight_pounds = 30 →
  food_per_half_pound = 1 →
  jar_food_ounces = 15 →
  jar_cost_dollars = 2 →
  total_cost = 8 :=
by
  intros h_weight h_food h_jar_ounces h_jar_cost
  sorry

end turtle_feeding_cost_l2059_205919


namespace problem_remainder_3_l2059_205999

theorem problem_remainder_3 :
  88 % 5 = 3 :=
by
  sorry

end problem_remainder_3_l2059_205999


namespace larger_cube_volume_is_512_l2059_205991

def original_cube_volume := 64 -- volume in cubic feet
def scale_factor := 2 -- the factor by which the dimensions are scaled

def side_length (volume : ℕ) : ℕ := volume^(1/3) -- Assuming we have a function to compute cube root

def larger_cube_volume (original_volume : ℕ) (scale_factor : ℕ) : ℕ :=
  let original_side_length := side_length original_volume
  let larger_side_length := scale_factor * original_side_length
  larger_side_length ^ 3

theorem larger_cube_volume_is_512 :
  larger_cube_volume original_cube_volume scale_factor = 512 :=
sorry

end larger_cube_volume_is_512_l2059_205991


namespace minimum_value_l2059_205902

theorem minimum_value (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x + y + z = 6) :
  37.5 ≤ (9 / x + 25 / y + 49 / z) :=
sorry

end minimum_value_l2059_205902


namespace value_of_b_l2059_205903

theorem value_of_b (b : ℝ) (h : 4 * ((3.6 * b * 2.50) / (0.12 * 0.09 * 0.5)) = 3200.0000000000005) : b = 0.48 :=
by {
  sorry
}

end value_of_b_l2059_205903


namespace panthers_score_points_l2059_205954

theorem panthers_score_points (C P : ℕ) (h1 : C + P = 34) (h2 : C = P + 14) : P = 10 :=
by
  sorry

end panthers_score_points_l2059_205954


namespace right_triangle_condition_l2059_205913

theorem right_triangle_condition (a b c : ℝ) : (a^2 = b^2 - c^2) → (∃ B : ℝ, B = 90) := 
sorry

end right_triangle_condition_l2059_205913


namespace Morse_code_distinct_symbols_l2059_205978

-- Morse code sequences conditions
def MorseCodeSequence (n : ℕ) := n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4

-- Total number of distinct symbols calculation
def total_distinct_symbols : ℕ :=
  2 + 4 + 8 + 16

-- The theorem to prove
theorem Morse_code_distinct_symbols : total_distinct_symbols = 30 := by
  sorry

end Morse_code_distinct_symbols_l2059_205978


namespace find_positive_real_solution_l2059_205917

theorem find_positive_real_solution (x : ℝ) : 
  0 < x ∧ (1 / 2 * (4 * x^2 - 1) = (x^2 - 60 * x - 20) * (x^2 + 30 * x + 10)) ↔ 
  (x = 30 + Real.sqrt 919 ∨ x = -15 + Real.sqrt 216 ∧ 0 < -15 + Real.sqrt 216) :=
by sorry

end find_positive_real_solution_l2059_205917


namespace min_value_of_x3y2z_l2059_205952

noncomputable def min_value_of_polynomial (x y z : ℝ) : ℝ :=
  x^3 * y^2 * z

theorem min_value_of_x3y2z
  (x y z : ℝ)
  (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h : 1 / x + 1 / y + 1 / z = 9) :
  min_value_of_polynomial x y z = 1 / 46656 :=
sorry

end min_value_of_x3y2z_l2059_205952


namespace find_m_l2059_205969

theorem find_m (m : ℝ) : 
  (m^2 + 3 * m + 3 ≠ 0) ∧ (m^2 + 2 * m - 3 ≠ 0) ∧ 
  (m^2 + 3 * m + 3 = 1) → m = -2 := 
by
  sorry

end find_m_l2059_205969


namespace carrie_spent_l2059_205980

-- Define the cost of one t-shirt
def cost_per_tshirt : ℝ := 9.65

-- Define the number of t-shirts bought
def num_tshirts : ℝ := 12

-- Define the total cost function
def total_cost (cost_per_tshirt : ℝ) (num_tshirts : ℝ) : ℝ := cost_per_tshirt * num_tshirts

-- State the theorem which we need to prove
theorem carrie_spent :
  total_cost cost_per_tshirt num_tshirts = 115.80 :=
by
  sorry

end carrie_spent_l2059_205980


namespace planes_count_l2059_205931

-- Define the conditions as given in the problem.
def total_wings : ℕ := 90
def wings_per_plane : ℕ := 2

-- Define the number of planes calculation based on conditions.
def number_of_planes : ℕ := total_wings / wings_per_plane

-- Prove that the number of planes is 45.
theorem planes_count : number_of_planes = 45 :=
by 
  -- The proof steps are omitted as specified.
  sorry

end planes_count_l2059_205931


namespace highest_score_of_batsman_l2059_205960

theorem highest_score_of_batsman
  (avg : ℕ)
  (inn : ℕ)
  (diff_high_low : ℕ)
  (sum_high_low : ℕ)
  (avg_excl : ℕ)
  (inn_excl : ℕ)
  (h_l_avg : avg = 60)
  (h_l_inn : inn = 46)
  (h_l_diff : diff_high_low = 140)
  (h_l_sum : sum_high_low = 208)
  (h_l_avg_excl : avg_excl = 58)
  (h_l_inn_excl : inn_excl = 44) :
  ∃ H L : ℕ, H = 174 :=
by
  sorry

end highest_score_of_batsman_l2059_205960


namespace Ruby_apples_remaining_l2059_205972

def Ruby_original_apples : ℕ := 6357912
def Emily_takes_apples : ℕ := 2581435
def Ruby_remaining_apples (R E : ℕ) : ℕ := R - E

theorem Ruby_apples_remaining : Ruby_remaining_apples Ruby_original_apples Emily_takes_apples = 3776477 := by
  sorry

end Ruby_apples_remaining_l2059_205972


namespace no_common_multiples_of_3_l2059_205976

-- Define the sets X and Y
def SetX : Set ℤ := {n | 1 ≤ n ∧ n ≤ 24 ∧ n % 2 = 1}
def SetY : Set ℤ := {n | 0 ≤ n ∧ n ≤ 40 ∧ n % 2 = 0}

-- Define the condition for being a multiple of 3
def isMultipleOf3 (n : ℤ) : Prop := n % 3 = 0

-- Define the intersection of SetX and SetY that are multiples of 3
def intersectionMultipleOf3 : Set ℤ := {n | n ∈ SetX ∧ n ∈ SetY ∧ isMultipleOf3 n}

-- Prove that the set is empty
theorem no_common_multiples_of_3 : intersectionMultipleOf3 = ∅ := by
  sorry

end no_common_multiples_of_3_l2059_205976


namespace alex_silver_tokens_count_l2059_205909

-- Conditions
def initial_red_tokens := 90
def initial_blue_tokens := 80

def red_exchange (x : ℕ) (y : ℕ) : ℕ := 90 - 3 * x + y
def blue_exchange (x : ℕ) (y : ℕ) : ℕ := 80 + 2 * x - 4 * y

-- Boundaries where exchanges stop
def red_bound (x : ℕ) (y : ℕ) : Prop := red_exchange x y < 3
def blue_bound (x : ℕ) (y : ℕ) : Prop := blue_exchange x y < 4

-- Proof statement
theorem alex_silver_tokens_count (x y : ℕ) :
    red_bound x y → blue_bound x y → (x + y) = 52 :=
    by
    sorry

end alex_silver_tokens_count_l2059_205909


namespace minNumberOfGloves_l2059_205998

-- Define the number of participants
def numParticipants : ℕ := 43

-- Define the number of gloves needed per participant
def glovesPerParticipant : ℕ := 2

-- Define the total number of gloves
def totalGloves (participants glovesPerParticipant : ℕ) : ℕ := 
  participants * glovesPerParticipant

-- Theorem proving the minimum number of gloves required
theorem minNumberOfGloves : totalGloves numParticipants glovesPerParticipant = 86 :=
by
  sorry

end minNumberOfGloves_l2059_205998


namespace remainder_problem_l2059_205938

theorem remainder_problem (x y z : ℤ) 
  (hx : x % 15 = 11) (hy : y % 15 = 13) (hz : z % 15 = 14) : 
  (y + z - x) % 15 = 1 := 
by 
  sorry

end remainder_problem_l2059_205938


namespace find_added_number_l2059_205950

def original_number : ℕ := 5
def doubled : ℕ := 2 * original_number
def resultant (added : ℕ) : ℕ := 3 * (doubled + added)
def final_result : ℕ := 57

theorem find_added_number (added : ℕ) (h : resultant added = final_result) : added = 9 :=
sorry

end find_added_number_l2059_205950


namespace binomial_p_value_l2059_205997

noncomputable def binomial_expected_value (n : ℕ) (p : ℝ) : ℝ := n * p

theorem binomial_p_value (p : ℝ) : (binomial_expected_value 18 p = 9) → p = 1/2 :=
by
  intro h
  sorry

end binomial_p_value_l2059_205997


namespace recreation_percentage_this_week_l2059_205908

variable (W : ℝ) -- David's last week wages
variable (R_last_week : ℝ) -- Recreation spending last week
variable (W_this_week : ℝ) -- This week's wages
variable (R_this_week : ℝ) -- Recreation spending this week

-- Conditions
def wages_last_week : R_last_week = 0.4 * W := sorry
def wages_this_week : W_this_week = 0.95 * W := sorry
def recreation_spending_this_week : R_this_week = 1.1875 * R_last_week := sorry

-- Theorem to prove
theorem recreation_percentage_this_week :
  (R_this_week / W_this_week) = 0.5 := sorry

end recreation_percentage_this_week_l2059_205908


namespace kim_total_water_intake_l2059_205989

def quarts_to_ounces (q : ℝ) : ℝ := q * 32

theorem kim_total_water_intake :
  (quarts_to_ounces 1.5) + 12 = 60 := 
by
  -- proof step 
  sorry

end kim_total_water_intake_l2059_205989


namespace balazs_missed_number_l2059_205977

theorem balazs_missed_number (n k : ℕ) 
  (h1 : n * (n + 1) / 2 = 3000 + k)
  (h2 : 1 ≤ k)
  (h3 : k < n) : k = 3 := by
  sorry

end balazs_missed_number_l2059_205977


namespace problem_l2059_205929

theorem problem (a b : ℝ) (h1 : |a - 2| + (b + 1)^2 = 0) : a - b = 3 := by
  sorry

end problem_l2059_205929


namespace find_percentage_l2059_205922

theorem find_percentage (x p : ℝ) (h1 : x = 840) (h2 : 0.25 * x + 15 = p / 100 * 1500) : p = 15 := 
by
  sorry

end find_percentage_l2059_205922


namespace valentines_cards_count_l2059_205924

theorem valentines_cards_count (x y : ℕ) (h1 : x * y = x + y + 30) : x * y = 64 :=
by {
    sorry
}

end valentines_cards_count_l2059_205924


namespace car_repair_cost_l2059_205928

noncomputable def total_cost (first_mechanic_rate: ℝ) (first_mechanic_hours: ℕ) 
    (first_mechanic_days: ℕ) (second_mechanic_rate: ℝ) 
    (second_mechanic_hours: ℕ) (second_mechanic_days: ℕ) 
    (discount_first: ℝ) (discount_second: ℝ) 
    (parts_cost: ℝ) (sales_tax_rate: ℝ): ℝ :=
  let first_mechanic_cost := first_mechanic_rate * first_mechanic_hours * first_mechanic_days
  let second_mechanic_cost := second_mechanic_rate * second_mechanic_hours * second_mechanic_days
  let first_mechanic_discounted := first_mechanic_cost - (discount_first * first_mechanic_cost)
  let second_mechanic_discounted := second_mechanic_cost - (discount_second * second_mechanic_cost)
  let total_before_tax := first_mechanic_discounted + second_mechanic_discounted + parts_cost
  let sales_tax := sales_tax_rate * total_before_tax
  total_before_tax + sales_tax

theorem car_repair_cost :
  total_cost 60 8 14 75 6 10 0.15 0.10 3200 0.07 = 13869.34 := by
  sorry

end car_repair_cost_l2059_205928


namespace greatest_possible_value_of_a_l2059_205906

theorem greatest_possible_value_of_a 
  (x a : ℤ)
  (h : x^2 + a * x = -21)
  (ha_pos : 0 < a)
  (hx_int : x ∈ [-21, -7, -3, -1].toFinset): 
  a ≤ 22 := sorry

end greatest_possible_value_of_a_l2059_205906


namespace contrapositive_statement_l2059_205953

-- Condition definitions
def P (x : ℝ) := x^2 < 1
def Q (x : ℝ) := -1 < x ∧ x < 1
def not_Q (x : ℝ) := x ≤ -1 ∨ x ≥ 1
def not_P (x : ℝ) := x^2 ≥ 1

theorem contrapositive_statement (x : ℝ) : (x ≤ -1 ∨ x ≥ 1) → (x^2 ≥ 1) :=
by
  sorry

end contrapositive_statement_l2059_205953


namespace f_20_value_l2059_205958

noncomputable def f (n : ℕ) : ℚ := sorry

axiom f_initial : f 1 = 3 / 2
axiom f_eq : ∀ x y : ℕ, 
  f (x + y) = (1 + y / (x + 1)) * f x + (1 + x / (y + 1)) * f y + x^2 * y + x * y + x * y^2

theorem f_20_value : f 20 = 4305 := 
by {
  sorry 
}

end f_20_value_l2059_205958


namespace tangent_line_at_0_maximum_integer_value_of_a_l2059_205971

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.log (x + 1) - a*x + 2

-- Part (1)
-- Prove that the equation of the tangent line to f(x) at x = 0 is x + y - 2 = 0 when a = 2
theorem tangent_line_at_0 {a : ℝ} (h : a = 2) : ∀ x y : ℝ, (y = f x a) → (x = 0) → (y = 2 - x) :=
by 
  sorry

-- Part (2)
-- Prove that if f(x) + 2x + x log(x+1) ≥ 0 holds for all x ≥ 0, then the maximum integer value of a is 4
theorem maximum_integer_value_of_a 
  (h : ∀ x : ℝ, x ≥ 0 → f x a + 2 * x + x * Real.log (x + 1) ≥ 0) : a ≤ 4 :=
by
  sorry

end tangent_line_at_0_maximum_integer_value_of_a_l2059_205971
