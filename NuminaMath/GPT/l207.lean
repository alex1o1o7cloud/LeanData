import Mathlib

namespace total_shaded_area_l207_207690

theorem total_shaded_area (r R : ℝ) (h1 : π * R^2 = 100 * π) (h2 : r = R / 2) : 
    (1/4) * π * R^2 + (1/4) * π * r^2 = 31.25 * π :=
by
  sorry

end total_shaded_area_l207_207690


namespace semicircle_area_in_quarter_circle_l207_207047

theorem semicircle_area_in_quarter_circle (r : ℝ) (A : ℝ) (π : ℝ) (one : ℝ) :
    r = 1 / (Real.sqrt (2) + 1) →
    A = π * r^2 →
    120 * A / π = 20 :=
sorry

end semicircle_area_in_quarter_circle_l207_207047


namespace mary_visited_two_shops_l207_207217

-- Define the costs of items
def cost_shirt : ℝ := 13.04
def cost_jacket : ℝ := 12.27
def total_cost : ℝ := 25.31

-- Define the number of shops visited
def number_of_shops : ℕ := 2

-- Proof that Mary visited 2 shops given the conditions
theorem mary_visited_two_shops (h_shirt : cost_shirt = 13.04) (h_jacket : cost_jacket = 12.27) (h_total : cost_shirt + cost_jacket = total_cost) : number_of_shops = 2 :=
by
  sorry

end mary_visited_two_shops_l207_207217


namespace min_val_of_3x_add_4y_l207_207620

theorem min_val_of_3x_add_4y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 3 * y = 5 * x * y) : 
  (3 * x + 4 * y ≥ 5) ∧ (3 * x + 4 * y = 5 → x + 4 * y = 3) := 
by
  sorry

end min_val_of_3x_add_4y_l207_207620


namespace exists_n_such_that_not_square_l207_207315

theorem exists_n_such_that_not_square : ∃ n : ℕ, n > 1 ∧ ¬(∃ k : ℕ, k ^ 2 = 2 ^ (2 ^ n - 1) - 7) := 
sorry

end exists_n_such_that_not_square_l207_207315


namespace cos_half_angle_l207_207444

open Real

theorem cos_half_angle (α : ℝ) (h_sin : sin α = (4 / 9) * sqrt 2) (h_obtuse : π / 2 < α ∧ α < π) :
  cos (α / 2) = 1 / 3 :=
by
  sorry

end cos_half_angle_l207_207444


namespace solve_xyz_sum_l207_207851

theorem solve_xyz_sum :
  ∃ (x y z : ℕ), 0 < x ∧ 0 < y ∧ 0 < z ∧ (x+y+z)^3 - x^3 - y^3 - z^3 = 378 ∧ x+y+z = 9 :=
by
  sorry

end solve_xyz_sum_l207_207851


namespace find_multiplier_l207_207515

theorem find_multiplier (n k : ℤ) (h1 : n + 4 = 15) (h2 : 3 * n = k * (n + 4) + 3) : k = 2 :=
  sorry

end find_multiplier_l207_207515


namespace solve_for_x_l207_207859

theorem solve_for_x (x : ℚ) (h : 10 * x = x + 20) : x = 20 / 9 :=
  sorry

end solve_for_x_l207_207859


namespace range_of_a_l207_207648

noncomputable def f (a x : ℝ) := (1 / 3) * x^3 - x^2 - 3 * x - a

theorem range_of_a (a : ℝ) : 
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ z ≠ x ∧ f a x = 0 ∧ f a y = 0 ∧ f a z = 0) ↔ (-9 < a ∧ a < 5 / 3) :=
by apply sorry

end range_of_a_l207_207648


namespace problem1_problem2_l207_207739

variables (x a : ℝ)

-- Proposition definitions
def proposition_p (a : ℝ) (x : ℝ) : Prop :=
  a > 0 ∧ (-x^2 + 4*a*x - 3*a^2) > 0

def proposition_q (x : ℝ) : Prop :=
  (x - 3) / (x - 2) < 0

-- Problems
theorem problem1 : (proposition_p 1 x ∧ proposition_q x) ↔ 2 < x ∧ x < 3 :=
by sorry

theorem problem2 : (¬ ∃ x, proposition_p a x) → (∀ x, ¬ proposition_q x) →
  1 ≤ a ∧ a ≤ 2 :=
by sorry

end problem1_problem2_l207_207739


namespace d_minus_b_equals_757_l207_207356

theorem d_minus_b_equals_757 
  (a b c d : ℕ) 
  (h1 : a^5 = b^4) 
  (h2 : c^3 = d^2) 
  (h3 : c - a = 19) : 
  d - b = 757 := 
by 
  sorry

end d_minus_b_equals_757_l207_207356


namespace garage_sale_records_l207_207597

/--
Roberta started off with 8 vinyl records. Her friends gave her 12
records for her birthday and she bought some more at a garage
sale. It takes her 2 days to listen to 1 record. It will take her
100 days to listen to her record collection. Prove that she bought
30 records at the garage sale.
-/
theorem garage_sale_records :
  let initial_records := 8
  let gift_records := 12
  let days_per_record := 2
  let total_listening_days := 100
  let total_records := total_listening_days / days_per_record
  let records_before_sale := initial_records + gift_records
  let records_bought := total_records - records_before_sale
  records_bought = 30 := 
by
  -- Variable assumptions
  let initial_records := 8
  let gift_records := 12
  let days_per_record := 2
  let total_listening_days := 100

  -- Definitions
  let total_records := total_listening_days / days_per_record
  let records_before_sale := initial_records + gift_records
  let records_bought := total_records - records_before_sale

  -- Conclusion to prove
  show records_bought = 30
  sorry

end garage_sale_records_l207_207597


namespace max_pens_min_pens_l207_207441

def pen_prices : List ℕ := [2, 3, 4]
def total_money : ℕ := 31

/-- Given the conditions of the problem, prove the maximum number of pens -/
theorem max_pens  (hx : 31 = total_money) 
  (ha : pen_prices = [2, 3, 4])
  (at_least_one : ∀ p ∈ pen_prices, 1 ≤ p) :
  exists n : ℕ, n = 14 := by
  sorry

/-- Given the conditions of the problem, prove the minimum number of pens -/
theorem min_pens (hx : 31 = total_money) 
  (ha : pen_prices = [2, 3, 4])
  (at_least_one : ∀ p ∈ pen_prices, 1 ≤ p) :
  exists n : ℕ, n = 9 := by
  sorry

end max_pens_min_pens_l207_207441


namespace max_distance_travel_l207_207141

-- Each car can carry at most 24 barrels of gasoline
def max_gasoline_barrels : ℕ := 24

-- Each barrel allows a car to travel 60 kilometers
def distance_per_barrel : ℕ := 60

-- The maximum distance one car can travel one way on a full tank
def max_one_way_distance := max_gasoline_barrels * distance_per_barrel

-- Total trip distance for the furthest traveling car
def total_trip_distance := 2160

-- Distance the other car turns back
def turn_back_distance := 360

-- Formalize in Lean
theorem max_distance_travel :
  (∃ x : ℕ, x = turn_back_distance ∧ max_gasoline_barrels * distance_per_barrel = 360) ∧
  (∃ y : ℕ, y = max_one_way_distance * 3 - turn_back_distance * 6 ∧ y = total_trip_distance) :=
by
  sorry

end max_distance_travel_l207_207141


namespace cakes_remaining_l207_207463

theorem cakes_remaining (initial_cakes sold_cakes remaining_cakes: ℕ) (h₀ : initial_cakes = 167) (h₁ : sold_cakes = 108) (h₂ : remaining_cakes = initial_cakes - sold_cakes) : remaining_cakes = 59 :=
by
  rw [h₀, h₁] at h₂
  exact h₂

end cakes_remaining_l207_207463


namespace min_board_size_l207_207778

theorem min_board_size (n : ℕ) (total_area : ℕ) (domino_area : ℕ) 
  (h1 : total_area = 2008) 
  (h2 : domino_area = 2) 
  (h3 : ∀ domino_count : ℕ, domino_count = total_area / domino_area → (∃ m : ℕ, (m+1) * (m+1) ≥ domino_count * (2 + 4) → n = m)) :
  n = 77 :=
by
  sorry

end min_board_size_l207_207778


namespace difference_fraction_reciprocal_l207_207807

theorem difference_fraction_reciprocal :
  let f := (4 : ℚ) / 5
  let r := (5 : ℚ) / 4
  f - r = 9 / 20 :=
by
  sorry

end difference_fraction_reciprocal_l207_207807


namespace range_of_m_l207_207738

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, (x < m - 1 ∨ x > m + 1) → (x^2 - 2*x - 3 > 0)) ∧ 
  ¬(∀ x : ℝ, (x < m - 1 ∨ x > m + 1) ↔ (x^2 - 2*x - 3 > 0)) 
  ↔ 0 ≤ m ∧ m ≤ 2 :=
by 
  sorry

end range_of_m_l207_207738


namespace geometric_sequence_common_ratio_eq_one_third_l207_207337

variable {a_n : ℕ → ℝ}
variable {q : ℝ}

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = q * a n

theorem geometric_sequence_common_ratio_eq_one_third
  (h_geom : geometric_sequence a_n q)
  (h_increasing : ∀ n, a_n n < a_n (n + 1))
  (h_a1 : a_n 1 = -2)
  (h_recurrence : ∀ n, 3 * (a_n n + a_n (n + 2)) = 10 * a_n (n + 1)) :
  q = 1 / 3 :=
by
  sorry

end geometric_sequence_common_ratio_eq_one_third_l207_207337


namespace age_difference_l207_207822

-- Denote the ages of A, B, and C as a, b, and c respectively.
variables (a b c : ℕ)

-- The given condition
def condition : Prop := a + b = b + c + 12

-- Prove that C is 12 years younger than A.
theorem age_difference (h : condition a b c) : c = a - 12 :=
by {
  -- skip the actual proof here, as instructed
  sorry
}

end age_difference_l207_207822


namespace polynomial_horner_method_l207_207289

theorem polynomial_horner_method :
  let a_4 := 3
  let a_3 := 0
  let a_2 := -1
  let a_1 := 2
  let a_0 := 1
  let x := 2
  let v_0 := 3
  let v_1 := v_0 * x + a_3
  let v_2 := v_1 * x + a_2
  let v_3 := v_2 * x + a_1
  v_3 = 22 :=
by 
  let a_4 := 3
  let a_3 := 0
  let a_2 := -1
  let a_1 := 2
  let a_0 := 1
  let x := 2
  let v_0 := a_4
  let v_1 := v_0 * x + a_3
  let v_2 := v_1 * x + a_2
  let v_3 := v_2 * x + a_1
  sorry

end polynomial_horner_method_l207_207289


namespace set_inter_complement_U_B_l207_207064

-- Define sets U, A, B
def U : Set ℝ := {x | x ≤ -1 ∨ x ≥ 0}
def A : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}
def B : Set ℝ := {x | x^2 > 1}

-- Statement to prove
theorem set_inter_complement_U_B :
  A ∩ (Uᶜ \ B) = {x | 0 ≤ x ∧ x ≤ 1} :=
by
  sorry

end set_inter_complement_U_B_l207_207064


namespace solve_inequality_l207_207865

theorem solve_inequality :
  {x : ℝ | (3 * x + 1) * (2 * x - 1) < 0} = {x : ℝ | -1 / 3 < x ∧ x < 1 / 2} :=
  sorry

end solve_inequality_l207_207865


namespace correct_transformation_l207_207373

-- Conditions given in the problem
def cond_A (a : ℤ) : Prop := a + 3 = 9 → a = 3 + 9
def cond_B (x : ℤ) : Prop := 4 * x = 7 * x - 2 → 4 * x - 7 * x = 2
def cond_C (a : ℤ) : Prop := 2 * a - 2 = -6 → 2 * a = 6 + 2
def cond_D (x : ℤ) : Prop := 2 * x - 5 = 3 * x + 3 → 2 * x - 3 * x = 3 + 5

-- Prove that the transformation in condition D is correct
theorem correct_transformation : (∀ a : ℤ, ¬cond_A a) ∧ (∀ x : ℤ, ¬cond_B x) ∧ (∀ a : ℤ, ¬cond_C a) ∧ (∀ x : ℤ, cond_D x) :=
by {
  -- Proof is provided in the solution and skipped here
  sorry
}

end correct_transformation_l207_207373


namespace pencil_price_l207_207975

variable (P N : ℕ) -- This assumes the price of a pencil (P) and the price of a notebook (N) are natural numbers (non-negative integers).

-- Define the conditions
def conditions : Prop :=
  (P + N = 950) ∧ (N = P + 150)

-- The theorem to prove
theorem pencil_price (h : conditions P N) : P = 400 :=
by
  sorry

end pencil_price_l207_207975


namespace number_of_Ca_atoms_in_compound_l207_207209

theorem number_of_Ca_atoms_in_compound
  (n : ℤ)
  (total_weight : ℝ)
  (ca_weight : ℝ)
  (i_weight : ℝ)
  (n_i_atoms : ℤ)
  (molecular_weight : ℝ) :
  n_i_atoms = 2 →
  molecular_weight = 294 →
  ca_weight = 40.08 →
  i_weight = 126.90 →
  n * ca_weight + n_i_atoms * i_weight = molecular_weight →
  n = 1 :=
by
  sorry

end number_of_Ca_atoms_in_compound_l207_207209


namespace length_AB_eight_l207_207525

-- Define parabola and line
def parabola (x y : ℝ) : Prop := y^2 = 4 * x
def line (x y k : ℝ) : Prop := y = k * x - k

-- Define intersection points A and B
def intersects (p1 p2 : ℝ × ℝ) (k : ℝ) : Prop :=
  parabola p1.1 p1.2 ∧ line p1.1 p1.2 k ∧
  parabola p2.1 p2.2 ∧ line p2.1 p2.2 k

-- Define midpoint distance condition
def midpoint_condition (p1 p2 : ℝ × ℝ) : Prop :=
  let midpoint := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)
  midpoint.1 = 3

-- The main theorem statement
theorem length_AB_eight (k : ℝ) (A B : ℝ × ℝ) (h1 : intersects A B k)
  (h2 : midpoint_condition A B) : abs ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 64 := 
sorry

end length_AB_eight_l207_207525


namespace valentines_count_l207_207465

theorem valentines_count (x y : ℕ) (h : x * y = x + y + 42) : x * y = 88 := by
  sorry

end valentines_count_l207_207465


namespace no_such_graph_exists_l207_207732

noncomputable def vertex_degrees (n : ℕ) (deg : ℕ → ℕ) : Prop :=
  n ≥ 8 ∧
  ∃ (deg : ℕ → ℕ),
    (deg 0 = 4) ∧ (deg 1 = 5) ∧ ∀ i, 2 ≤ i ∧ i < n - 7 → deg i = i + 4 ∧
    (deg (n-7) = n-2) ∧ (deg (n-6) = n-2) ∧ (deg (n-5) = n-2) ∧
    (deg (n-4) = n-1) ∧ (deg (n-3) = n-1) ∧ (deg (n-2) = n-1)   

theorem no_such_graph_exists (n : ℕ) (deg : ℕ → ℕ) : 
  n ≥ 10 → ¬vertex_degrees n deg := 
by
  sorry

end no_such_graph_exists_l207_207732


namespace fourth_buoy_distance_with_current_l207_207215

-- Define the initial conditions
def first_buoy_distance : ℕ := 20
def second_buoy_additional_distance : ℕ := 24
def third_buoy_additional_distance : ℕ := 28
def common_difference_increment : ℕ := 4
def ocean_current_push_per_segment : ℕ := 3
def number_of_segments : ℕ := 3

-- Define the mathematical proof problem
theorem fourth_buoy_distance_with_current :
  let fourth_buoy_additional_distance := third_buoy_additional_distance + common_difference_increment
  let first_to_second_buoy := first_buoy_distance + second_buoy_additional_distance
  let second_to_third_buoy := first_to_second_buoy + third_buoy_additional_distance
  let distance_before_current := second_to_third_buoy + fourth_buoy_additional_distance
  let total_current_push := ocean_current_push_per_segment * number_of_segments
  let final_distance := distance_before_current - total_current_push
  final_distance = 95 := by
  sorry

end fourth_buoy_distance_with_current_l207_207215


namespace find_difference_l207_207752

variable (a b c d e f : ℝ)

-- Conditions
def cond1 : Prop := a - b = c + d + 9
def cond2 : Prop := a + b = c - d - 3
def cond3 : Prop := e = a^2 + b^2
def cond4 : Prop := f = c^2 + d^2
def cond5 : Prop := f - e = 5 * a + 2 * b + 3 * c + 4 * d

-- Problem Statement
theorem find_difference (h1 : cond1 a b c d) (h2 : cond2 a b c d) (h3 : cond3 a b e) (h4 : cond4 c d f) (h5 : cond5 a b c d e f) : a - c = 3 :=
sorry

end find_difference_l207_207752


namespace cos_315_deg_l207_207474

noncomputable def cos_315 : ℝ :=
  Real.cos (315 * Real.pi / 180)

theorem cos_315_deg : cos_315 = Real.sqrt 2 / 2 :=
by
  sorry

end cos_315_deg_l207_207474


namespace points_subtracted_per_wrong_answer_l207_207844

theorem points_subtracted_per_wrong_answer 
  (total_problems : ℕ) 
  (wrong_answers : ℕ) 
  (score : ℕ) 
  (points_per_right_answer : ℕ) 
  (correct_answers : ℕ)
  (subtracted_points : ℕ) 
  (expected_points : ℕ) 
  (points_subtracted : ℕ) :
  total_problems = 25 → 
  wrong_answers = 3 → 
  score = 85 → 
  points_per_right_answer = 4 → 
  correct_answers = total_problems - wrong_answers → 
  expected_points = correct_answers * points_per_right_answer → 
  subtracted_points = expected_points - score → 
  points_subtracted = subtracted_points / wrong_answers → 
  points_subtracted = 1 := 
by
  intros;
  sorry

end points_subtracted_per_wrong_answer_l207_207844


namespace value_of_f_5_l207_207220

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * x^3 - b * Real.sin x - 2

theorem value_of_f_5 (a b : ℝ) (hf : f a b (-5) = 17) : f a b 5 = -21 := by
  sorry

end value_of_f_5_l207_207220


namespace projectiles_initial_distance_l207_207066

theorem projectiles_initial_distance 
  (v₁ v₂ : ℝ) (t : ℝ) (d₁ d₂ d : ℝ) 
  (hv₁ : v₁ = 445 / 60) -- speed of first projectile in km/min
  (hv₂ : v₂ = 545 / 60) -- speed of second projectile in km/min
  (ht : t = 84) -- time to meet in minutes
  (hd₁ : d₁ = v₁ * t) -- distance traveled by the first projectile
  (hd₂ : d₂ = v₂ * t) -- distance traveled by the second projectile
  (hd : d = d₁ + d₂) -- total initial distance
  : d = 1385.6 :=
by 
  sorry

end projectiles_initial_distance_l207_207066


namespace average_of_solutions_l207_207858

theorem average_of_solutions (a b : ℝ) (h : ∃ x1 x2 : ℝ, a * x1 ^ 2 + 3 * a * x1 + b = 0 ∧ a * x2 ^ 2 + 3 * a * x2 + b = 0) :
  ((-3 : ℝ) / 2) = - 3 / 2 :=
by sorry

end average_of_solutions_l207_207858


namespace intersection_M_N_eq_02_l207_207062

open Set

def M : Set ℕ := {0, 1, 2}
def N : Set ℕ := {y | ∃ x ∈ M, y = 2 * x}

theorem intersection_M_N_eq_02 : M ∩ N = {0, 2} := 
by sorry

end intersection_M_N_eq_02_l207_207062


namespace sin_product_l207_207458

theorem sin_product (α : ℝ) (h : Real.tan α = 2) : Real.sin α * Real.sin (π / 2 - α) = 2 / 5 :=
by
  -- proof shorter placeholder
  sorry

end sin_product_l207_207458


namespace number_of_yellow_balls_l207_207410

-- Definitions based on conditions
def number_of_red_balls : ℕ := 10
def probability_red_ball := (1 : ℚ) / 3

-- Theorem stating the number of yellow balls
theorem number_of_yellow_balls :
  ∃ (y : ℕ), (number_of_red_balls : ℚ) / (number_of_red_balls + y) = probability_red_ball ∧ y = 20 :=
by
  sorry

end number_of_yellow_balls_l207_207410


namespace angle_RBC_10_degrees_l207_207455

noncomputable def compute_angle_RBC (angle_BRA angle_BAC angle_ABC : ℝ) : ℝ :=
  let angle_RBA := 180 - angle_BRA - angle_BAC
  angle_RBA - angle_ABC

theorem angle_RBC_10_degrees :
  ∀ (angle_BRA angle_BAC angle_ABC : ℝ), 
    angle_BRA = 72 → angle_BAC = 43 → angle_ABC = 55 → 
    compute_angle_RBC angle_BRA angle_BAC angle_ABC = 10 :=
by
  intros
  unfold compute_angle_RBC
  sorry

end angle_RBC_10_degrees_l207_207455


namespace smallest_hope_number_l207_207617

def is_square (n : ℕ) : Prop := ∃ (k : ℕ), n = k * k
def is_cube (n : ℕ) : Prop := ∃ (k : ℕ), n = k * k * k
def is_fifth_power (n : ℕ) : Prop := ∃ (k : ℕ), n = k * k * k * k * k

def is_hope_number (n : ℕ) : Prop :=
  is_square (n / 8) ∧ is_cube (n / 9) ∧ is_fifth_power (n / 25)

theorem smallest_hope_number : ∃ n, is_hope_number n ∧ n = 2^15 * 3^20 * 5^12 :=
by
  sorry

end smallest_hope_number_l207_207617


namespace nat_power_digit_condition_l207_207152

theorem nat_power_digit_condition (n k : ℕ) : 
  (10^(k-1) < n^n ∧ n^n < 10^k) → (10^(n-1) < k^k ∧ k^k < 10^n) → 
  (n = 1 ∧ k = 1) ∨ (n = 8 ∧ k = 8) ∨ (n = 9 ∧ k = 9) :=
by
  sorry

end nat_power_digit_condition_l207_207152


namespace greatest_of_consecutive_even_numbers_l207_207478

theorem greatest_of_consecutive_even_numbers (n : ℤ) (h : ((n - 4) + (n - 2) + n + (n + 2) + (n + 4)) / 5 = 35) : n + 4 = 39 :=
by
  sorry

end greatest_of_consecutive_even_numbers_l207_207478


namespace ratio_of_volume_to_surface_area_l207_207140

-- Definitions of the given conditions
def unit_cube_volume : ℕ := 1
def total_cubes : ℕ := 8
def volume := total_cubes * unit_cube_volume
def exposed_faces (center_cube_faces : ℕ) (side_cube_faces : ℕ) (top_cube_faces : ℕ) : ℕ :=
  center_cube_faces + 6 * side_cube_faces + top_cube_faces
def surface_area := exposed_faces 1 5 5
def ratio := volume / surface_area

-- The main theorem statement
theorem ratio_of_volume_to_surface_area : ratio = 2 / 9 := by
  sorry

end ratio_of_volume_to_surface_area_l207_207140


namespace find_a_l207_207208

theorem find_a (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : a + a^2 = 12) : a = 3 :=
by sorry

end find_a_l207_207208


namespace find_hours_hired_l207_207221

def hourly_rate : ℝ := 15
def tip_rate : ℝ := 0.20
def total_paid : ℝ := 54

theorem find_hours_hired (h : ℝ) : 15 * h + 0.20 * 15 * h = 54 → h = 3 :=
by
  sorry

end find_hours_hired_l207_207221


namespace probability_of_same_color_is_correct_l207_207042

-- Definitions from the problem conditions
def red_marbles := 6
def white_marbles := 7
def blue_marbles := 8
def total_marbles := red_marbles + white_marbles + blue_marbles -- 21

-- Calculate the probability of drawing 4 red marbles
def P_all_red := (red_marbles / total_marbles) * ((red_marbles - 1) / (total_marbles - 1)) * ((red_marbles - 2) / (total_marbles - 2)) * ((red_marbles - 3) / (total_marbles - 3))

-- Calculate the probability of drawing 4 white marbles
def P_all_white := (white_marbles / total_marbles) * ((white_marbles - 1) / (total_marbles - 1)) * ((white_marbles - 2) / (total_marbles - 2)) * ((white_marbles - 3) / (total_marbles - 3))

-- Calculate the probability of drawing 4 blue marbles
def P_all_blue := (blue_marbles / total_marbles) * ((blue_marbles - 1) / (total_marbles - 1)) * ((blue_marbles - 2) / (total_marbles - 2)) * ((blue_marbles - 3) / (total_marbles - 3))

-- Total probability of drawing 4 marbles of the same color
def P_all_same_color := P_all_red + P_all_white + P_all_blue

-- Proof that the total probability is equal to the given correct answer
theorem probability_of_same_color_is_correct : P_all_same_color = 240 / 11970 := by
  sorry

end probability_of_same_color_is_correct_l207_207042


namespace difference_of_squares_153_147_l207_207545

theorem difference_of_squares_153_147 : (153^2 - 147^2) = 1800 := by
  sorry

end difference_of_squares_153_147_l207_207545


namespace sides_and_diagonals_l207_207286

def number_of_sides_of_polygon (n : ℕ) :=
  180 * (n - 2) = 360 + (1 / 4 : ℤ) * 360

def number_of_diagonals_of_polygon (n : ℕ) : ℕ :=
  n * (n - 3) / 2

theorem sides_and_diagonals : 
  (∃ n : ℕ, number_of_sides_of_polygon n ∧ n = 12) ∧ number_of_diagonals_of_polygon 12 = 54 :=
by {
  -- Proof will be filled in later
  sorry
}

end sides_and_diagonals_l207_207286


namespace solve_quadratic_inequality_l207_207005

-- To express that a real number x is in the interval (0, 2)
def in_interval (x : ℝ) : Prop := 0 < x ∧ x < 2

theorem solve_quadratic_inequality :
  { x : ℝ | x^2 < 2 * x } = { x : ℝ | in_interval x } :=
by
  sorry

end solve_quadratic_inequality_l207_207005


namespace remove_least_candies_l207_207701

theorem remove_least_candies (total_candies : ℕ) (friends : ℕ) (candies_remaining : ℕ) : total_candies = 34 ∧ friends = 5 ∧ candies_remaining = 4 → (total_candies % friends = candies_remaining) :=
by
  intros h
  sorry

end remove_least_candies_l207_207701


namespace ratio_of_divisors_l207_207201

def M : Nat := 75 * 75 * 140 * 343

noncomputable def sumOfOddDivisors (n : Nat) : Nat := 
  -- Function that computes the sum of all odd divisors of n. (placeholder)
  sorry

noncomputable def sumOfEvenDivisors (n : Nat) : Nat := 
  -- Function that computes the sum of all even divisors of n. (placeholder)
  sorry

theorem ratio_of_divisors :
  let sumOdd := sumOfOddDivisors M
  let sumEven := sumOfEvenDivisors M
  sumOdd / sumEven = 1 / 6 := 
by
  sorry

end ratio_of_divisors_l207_207201


namespace interval_length_implies_difference_l207_207756

variable (c d : ℝ)

theorem interval_length_implies_difference (h1 : ∀ x : ℝ, c ≤ 3 * x + 5 ∧ 3 * x + 5 ≤ d) (h2 : (d - c) / 3 = 15) : d - c = 45 := 
sorry

end interval_length_implies_difference_l207_207756


namespace necessary_but_not_sufficient_condition_l207_207827

open Set

variable {α : Type*}

def M : Set ℝ := { x | 0 < x ∧ x ≤ 4 }
def N : Set ℝ := { x | 2 ≤ x ∧ x ≤ 3 }

theorem necessary_but_not_sufficient_condition :
  (N ⊆ M) ∧ (∀ a, a ∈ N → a ∈ M) ∧ (∃ a, a ∈ M ∧ a ∉ N) :=
by
  sorry

end necessary_but_not_sufficient_condition_l207_207827


namespace abs_diff_is_perfect_square_l207_207802

-- Define the conditions
variable (m n : ℤ) (h_odd_m : m % 2 = 1) (h_odd_n : n % 2 = 1)
variable (h_div : (n^2 - 1) ∣ (m^2 + 1 - n^2))

-- Theorem statement
theorem abs_diff_is_perfect_square : ∃ (k : ℤ), (m^2 + 1 - n^2) = k^2 :=
by
  sorry

end abs_diff_is_perfect_square_l207_207802


namespace smallest_x_250_multiple_1080_l207_207378

theorem smallest_x_250_multiple_1080 : (∃ x : ℕ, x > 0 ∧ (250 * x) % 1080 = 0) ∧ ¬(∃ y : ℕ, y > 0 ∧ y < 54 ∧ (250 * y) % 1080 = 0) :=
by
  sorry

end smallest_x_250_multiple_1080_l207_207378


namespace train_platform_ratio_l207_207736

noncomputable def speed_km_per_hr := 216 -- condition 1
noncomputable def crossing_time_sec := 60 -- condition 2
noncomputable def train_length_m := 1800 -- condition 3

noncomputable def speed_m_per_s := speed_km_per_hr * 1000 / 3600
noncomputable def total_distance_m := speed_m_per_s * crossing_time_sec
noncomputable def platform_length_m := total_distance_m - train_length_m
noncomputable def ratio := train_length_m / platform_length_m

theorem train_platform_ratio : ratio = 1 := by
    sorry

end train_platform_ratio_l207_207736


namespace measuring_cup_size_l207_207248

-- Defining the conditions
def total_flour := 8
def flour_needed := 6
def scoops_removed := 8 

-- Defining the size of the cup
def cup_size (x : ℚ) := 8 - scoops_removed * x = flour_needed

-- Stating the theorem
theorem measuring_cup_size : ∃ x : ℚ, cup_size x ∧ x = 1 / 4 :=
by {
    sorry
}

end measuring_cup_size_l207_207248


namespace range_of_f_gt_f_of_quadratic_l207_207381

-- Define the function f and its properties
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def is_increasing_on_pos (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x → x < y → f x < f y

-- Define the problem statement
theorem range_of_f_gt_f_of_quadratic (f : ℝ → ℝ)
  (h_odd : is_odd_function f)
  (h_inc : is_increasing_on_pos f) :
  {x : ℝ | f x > f (x^2 - 2*x + 2)} = {x : ℝ | 1 < x ∧ x < 2} :=
sorry

end range_of_f_gt_f_of_quadratic_l207_207381


namespace average_bacterial_count_closest_to_true_value_l207_207503

-- Define the conditions
variables (dilution_spread_plate_method : Prop)
          (count_has_randomness : Prop)
          (count_not_uniform : Prop)

-- State the theorem
theorem average_bacterial_count_closest_to_true_value
  (h1: dilution_spread_plate_method)
  (h2: count_has_randomness)
  (h3: count_not_uniform)
  : true := sorry

end average_bacterial_count_closest_to_true_value_l207_207503


namespace cheaperCandy_cost_is_5_l207_207587

def cheaperCandy (C : ℝ) : Prop :=
  let expensiveCandyCost := 20 * 8
  let cheaperCandyCost := 40 * C
  let totalWeight := 20 + 40
  let totalCost := 60 * 6
  expensiveCandyCost + cheaperCandyCost = totalCost

theorem cheaperCandy_cost_is_5 : cheaperCandy 5 :=
by
  unfold cheaperCandy
  -- SORRY is a placeholder for the proof steps, which are not required
  sorry 

end cheaperCandy_cost_is_5_l207_207587


namespace Dodo_is_sane_l207_207313

-- Declare the names of the characters
inductive Character
| Dodo : Character
| Lori : Character
| Eagle : Character

open Character

-- Definitions of sanity state
def sane (c : Character) : Prop := sorry
def insane (c : Character) : Prop := ¬ sane c

-- Conditions based on the problem statement
axiom Dodo_thinks_Lori_thinks_Eagle_not_sane : (sane Lori → insane Eagle)
axiom Lori_thinks_Dodo_not_sane : insane Dodo
axiom Eagle_thinks_Dodo_sane : sane Dodo

-- Theorem to prove Dodo is sane
theorem Dodo_is_sane : sane Dodo :=
by {
    sorry
}

end Dodo_is_sane_l207_207313


namespace min_value_expression_l207_207951

theorem min_value_expression (x : ℝ) (h : x > 2) : 
  ∃ y, y = x + 1 / (x - 2) ∧ y = 4 := 
sorry

end min_value_expression_l207_207951


namespace find_n_l207_207941

theorem find_n (n : ℝ) (h : (2 : ℂ) / (1 - I) = (1 : ℂ) + n * I) : n = 1 := by
  sorry

end find_n_l207_207941


namespace domain_of_function_l207_207857

theorem domain_of_function :
  {x : ℝ | x ≥ -1} \ {0} = {x : ℝ | (x ≥ -1 ∧ x < 0) ∨ x > 0} :=
by
  sorry

end domain_of_function_l207_207857


namespace selling_price_of_cycle_l207_207612

def cost_price : ℝ := 1400
def loss_percentage : ℝ := 18

theorem selling_price_of_cycle : 
    (cost_price - (loss_percentage / 100) * cost_price) = 1148 := 
by
  sorry

end selling_price_of_cycle_l207_207612


namespace find_x_plus_y_l207_207911

theorem find_x_plus_y (x y : ℝ) 
  (h1 : x + Real.cos y = 1004)
  (h2 : x + 1004 * Real.sin y = 1003)
  (h3 : 0 ≤ y ∧ y ≤ Real.pi / 2) : x + y = 1003 :=
sorry

end find_x_plus_y_l207_207911


namespace solve_for_x_l207_207046

theorem solve_for_x (n m x : ℕ) (h1 : 5 / 7 = n / 91) (h2 : 5 / 7 = (m + n) / 105) (h3 : 5 / 7 = (x - m) / 140) :
    x = 110 :=
sorry

end solve_for_x_l207_207046


namespace factorization_pq_difference_l207_207433

theorem factorization_pq_difference :
  ∃ (p q : ℤ), 25 * x^2 - 135 * x - 150 = (5 * x + p) * (5 * x + q) ∧ p - q = 36 := by
-- Given the conditions in the problem,
-- We assume ∃ integers p and q such that (5x + p)(5x + q) = 25x² - 135x - 150 and derive the difference p - q = 36.
  sorry

end factorization_pq_difference_l207_207433


namespace product_of_distances_l207_207961

-- Definitions based on the conditions
def curve (x y : ℝ) : Prop := x * y = 2

-- The theorem to prove
theorem product_of_distances (x y : ℝ) (h : curve x y) : abs x * abs y = 2 := by
  -- This is where the proof would go
  sorry

end product_of_distances_l207_207961


namespace find_x_solution_l207_207794

theorem find_x_solution :
  ∃ x, 2 ^ (x / 2) * (Real.sqrt ((2 + Real.sqrt 3) ^ x) + Real.sqrt ((2 - Real.sqrt 3) ^ x)) = 6 ∧
       x = 2 * Real.log 1.5 / Real.log 2 := by
  sorry

end find_x_solution_l207_207794


namespace ball_hits_ground_l207_207762

theorem ball_hits_ground (t : ℚ) : 
  (∃ t ≥ 0, (-4.9 * (t^2 : ℝ) + 5 * t + 10 = 0)) → t = 100 / 49 :=
by
  sorry

end ball_hits_ground_l207_207762


namespace xy_value_l207_207761

theorem xy_value (x y : ℝ) (h : x * (x + 2 * y) = x^2 + 10) : x * y = 5 :=
by
  sorry

end xy_value_l207_207761


namespace quadratic_root_solution_l207_207977

theorem quadratic_root_solution (k : ℤ) (a : ℤ) :
  (∀ x, x^2 + k * x - 10 = 0 → x = 2 ∨ x = a) →
  2 + a = -k →
  2 * a = -10 →
  k = 3 ∧ a = -5 :=
by
  sorry

end quadratic_root_solution_l207_207977


namespace geometric_sequence_product_identity_l207_207602

theorem geometric_sequence_product_identity 
  {a : ℕ → ℝ} (is_geometric_sequence : ∃ r, ∀ n, a (n+1) = a n * r)
  (h : a 3 * a 4 * a 6 * a 7 = 81):
  a 1 * a 9 = 9 :=
by
  sorry

end geometric_sequence_product_identity_l207_207602


namespace circle_units_diff_l207_207174

-- Define the context where we verify the claim about the circle

noncomputable def radius : ℝ := 3
noncomputable def circumference (r : ℝ) := 2 * Real.pi * r
noncomputable def area (r : ℝ) := Real.pi * r ^ 2

-- Lean Theorem statement that needs to be proved
theorem circle_units_diff (r : ℝ) (h₀ : r = radius) :
  circumference r ≠ area r :=
by sorry

end circle_units_diff_l207_207174


namespace max_band_members_l207_207909

theorem max_band_members (n : ℤ) (h1 : 30 * n % 21 = 9) (h2 : 30 * n < 1500) : 30 * n ≤ 1470 :=
by
  -- Proof to be filled in later
  sorry

end max_band_members_l207_207909


namespace find_line_equation_l207_207936

noncomputable def perpendicular_origin_foot := 
  ∃ l : ℝ → ℝ → Prop,
    (∀ (x y : ℝ), l x y ↔ y = 2 * x + 5) ∧
    l (-2) 1

theorem find_line_equation : 
  ∃ l : ℝ → ℝ → Prop,
    (∀ (x y : ℝ), l x y ↔ 2 * x - y + 5 = 0) ∧
    l (-2) 1 ∧
    ∀ p q : ℝ, p = 0 → q = 0 → ¬ (l p q)
:= sorry

end find_line_equation_l207_207936


namespace total_instruments_correct_l207_207489

def numberOfFlutesCharlie : ℕ := 1
def numberOfHornsCharlie : ℕ := 2
def numberOfHarpsCharlie : ℕ := 1
def numberOfDrumsCharlie : ℕ := 5

def numberOfFlutesCarli : ℕ := 3 * numberOfFlutesCharlie
def numberOfHornsCarli : ℕ := numberOfHornsCharlie / 2
def numberOfDrumsCarli : ℕ := 2 * numberOfDrumsCharlie
def numberOfHarpsCarli : ℕ := 0

def numberOfFlutesNick : ℕ := 2 * numberOfFlutesCarli - 1
def numberOfHornsNick : ℕ := numberOfHornsCharlie + numberOfHornsCarli
def numberOfDrumsNick : ℕ := 4 * numberOfDrumsCarli - 2
def numberOfHarpsNick : ℕ := 0

def numberOfFlutesDaisy : ℕ := numberOfFlutesNick * numberOfFlutesNick
def numberOfHornsDaisy : ℕ := (numberOfHornsNick - numberOfHornsCarli) / 2
def numberOfDrumsDaisy : ℕ := (numberOfDrumsCharlie + numberOfDrumsCarli + numberOfDrumsNick) / 3
def numberOfHarpsDaisy : ℕ := numberOfHarpsCharlie

def numberOfInstrumentsCharlie : ℕ := numberOfFlutesCharlie + numberOfHornsCharlie + numberOfHarpsCharlie + numberOfDrumsCharlie
def numberOfInstrumentsCarli : ℕ := numberOfFlutesCarli + numberOfHornsCarli + numberOfDrumsCarli
def numberOfInstrumentsNick : ℕ := numberOfFlutesNick + numberOfHornsNick + numberOfDrumsNick
def numberOfInstrumentsDaisy : ℕ := numberOfFlutesDaisy + numberOfHornsDaisy + numberOfHarpsDaisy + numberOfDrumsDaisy

def totalInstruments : ℕ := numberOfInstrumentsCharlie + numberOfInstrumentsCarli + numberOfInstrumentsNick + numberOfInstrumentsDaisy

theorem total_instruments_correct : totalInstruments = 113 := by
  sorry

end total_instruments_correct_l207_207489


namespace appropriate_chart_for_temperature_statistics_l207_207127

theorem appropriate_chart_for_temperature_statistics (chart_type : String) (is_line_chart : chart_type = "line chart") : chart_type = "line chart" :=
by
  sorry

end appropriate_chart_for_temperature_statistics_l207_207127


namespace marissa_lunch_calories_l207_207934

theorem marissa_lunch_calories :
  (1 * 400) + (5 * 20) + (5 * 50) = 750 :=
by
  sorry

end marissa_lunch_calories_l207_207934


namespace kaylin_age_32_l207_207446

-- Defining the ages of the individuals as variables
variables (Kaylin Sarah Eli Freyja Alfred Olivia : ℝ)

-- Defining the given conditions
def conditions : Prop := 
  (Kaylin = Sarah - 5) ∧
  (Sarah = 2 * Eli) ∧
  (Eli = Freyja + 9) ∧
  (Freyja = 2.5 * Alfred) ∧
  (Alfred = (3/4) * Olivia) ∧
  (Freyja = 9.5)

-- Main statement to prove
theorem kaylin_age_32 (h : conditions Kaylin Sarah Eli Freyja Alfred Olivia) : Kaylin = 32 :=
by
  sorry

end kaylin_age_32_l207_207446


namespace workers_complete_time_l207_207902

theorem workers_complete_time 
  (time_A time_B time_C : ℕ) 
  (hA : time_A = 10)
  (hB : time_B = 12) 
  (hC : time_C = 15) : 
  let rate_A := (1: ℚ) / time_A
  let rate_B := (1: ℚ) / time_B
  let rate_C := (1: ℚ) / time_C
  let total_rate := rate_A + rate_B + rate_C
  1 / total_rate = 4 := 
by
  sorry

end workers_complete_time_l207_207902


namespace cost_of_airplane_l207_207074

theorem cost_of_airplane (amount : ℝ) (change : ℝ) (h_amount : amount = 5) (h_change : change = 0.72) : 
  amount - change = 4.28 := 
by
  sorry

end cost_of_airplane_l207_207074


namespace total_points_scored_l207_207462

theorem total_points_scored (layla_score nahima_score : ℕ)
  (h1 : layla_score = 70)
  (h2 : layla_score = nahima_score + 28) :
  layla_score + nahima_score = 112 :=
by
  sorry

end total_points_scored_l207_207462


namespace total_pages_read_l207_207662

-- Define the reading rates
def ReneReadingRate : ℕ := 30  -- pages in 60 minutes
def LuluReadingRate : ℕ := 27  -- pages in 60 minutes
def CherryReadingRate : ℕ := 25  -- pages in 60 minutes

-- Total time in minutes
def totalTime : ℕ := 240  -- minutes

-- Define a function to calculate pages read in given time
def pagesRead (rate : ℕ) (time : ℕ) : ℕ :=
  rate * (time / 60)

-- Theorem to prove the total number of pages read
theorem total_pages_read :
  pagesRead ReneReadingRate totalTime +
  pagesRead LuluReadingRate totalTime +
  pagesRead CherryReadingRate totalTime = 328 :=
by
  -- Proof is not required, hence replaced with sorry
  sorry

end total_pages_read_l207_207662


namespace Megan_not_lead_actress_l207_207708

-- Define the conditions: total number of plays and lead actress percentage
def totalPlays : ℕ := 100
def leadActressPercentage : ℕ := 80

-- Define what we need to prove: the number of times Megan was not the lead actress
theorem Megan_not_lead_actress (totalPlays: ℕ) (leadActressPercentage: ℕ) : 
  (totalPlays * (100 - leadActressPercentage)) / 100 = 20 :=
by
  -- proof omitted
  sorry

end Megan_not_lead_actress_l207_207708


namespace discount_given_l207_207837

variables (initial_money : ℕ) (extra_fraction : ℕ) (additional_money_needed : ℕ)
variables (total_with_discount : ℕ) (discount_amount : ℕ)

def total_without_discount (initial_money : ℕ) (extra_fraction : ℕ) : ℕ :=
  initial_money + extra_fraction

def discount (initial_money : ℕ) (total_without_discount : ℕ) (total_with_discount : ℕ) : ℕ :=
  total_without_discount - total_with_discount

def discount_percentage (discount_amount : ℕ) (total_without_discount : ℕ) : ℚ :=
  (discount_amount : ℚ) / (total_without_discount : ℚ) * 100

theorem discount_given 
  (initial_money : ℕ := 500)
  (extra_fraction : ℕ := 200)
  (additional_money_needed : ℕ := 95)
  (total_without_discount₀ : ℕ := total_without_discount initial_money extra_fraction)
  (total_with_discount₀ : ℕ := initial_money + additional_money_needed)
  (discount_amount₀ : ℕ := discount initial_money total_without_discount₀ total_with_discount₀)
  : discount_percentage discount_amount₀ total_without_discount₀ = 15 :=
by sorry

end discount_given_l207_207837


namespace no_six_digit_numbers_exists_l207_207634

theorem no_six_digit_numbers_exists :
  ¬(∃ (N : Fin 6 → Fin 720), ∀ (a b c : Fin 6), a ≠ b → a ≠ c → b ≠ c →
  (∃ (i : Fin 6), N i == 720)) := sorry

end no_six_digit_numbers_exists_l207_207634


namespace travel_time_reduction_l207_207334

theorem travel_time_reduction : 
  let t_initial := 19.5
  let factor_1998 := 1.30
  let factor_1999 := 1.25
  let factor_2000 := 1.20
  t_initial / factor_1998 / factor_1999 / factor_2000 = 10 := by
  sorry

end travel_time_reduction_l207_207334


namespace motorboat_distance_l207_207121

variable (S v u : ℝ)
variable (V_m : ℝ := 2 * v + u)  -- Velocity of motorboat downstream
variable (V_b : ℝ := 3 * v - u)  -- Velocity of boat upstream

theorem motorboat_distance :
  ( L = (161 / 225) * S ∨ L = (176 / 225) * S) :=
by
  sorry

end motorboat_distance_l207_207121


namespace incorrect_major_premise_l207_207643

-- Define a structure for Line and Plane
structure Line : Type :=
  (name : String)

structure Plane : Type :=
  (name : String)

-- Define relationships: parallel and contains
def parallel (l1 l2 : Line) : Prop := sorry
def line_in_plane (l : Line) (p : Plane) : Prop := sorry
def parallel_line_plane (l : Line) (p : Plane) : Prop := sorry

-- Conditions
variables 
  (a b : Line) 
  (α : Plane)
  (H1 : line_in_plane a α) 
  (H2 : parallel_line_plane b α)

-- Major premise to disprove
def major_premise (l : Line) (p : Plane) : Prop :=
  ∀ (l_in : Line), line_in_plane l_in p → parallel l l_in

-- State the problem
theorem incorrect_major_premise : ¬major_premise b α :=
sorry

end incorrect_major_premise_l207_207643


namespace train_crossing_time_l207_207591

/-- Time for a train of length 1500 meters traveling at 108 km/h to cross an electric pole is 50 seconds -/
theorem train_crossing_time (length : ℕ) (speed_kmph : ℕ) 
    (h₁ : length = 1500) (h₂ : speed_kmph = 108) : 
    (length / ((speed_kmph * 1000) / 3600) = 50) :=
by
  sorry

end train_crossing_time_l207_207591


namespace find_inradius_of_scalene_triangle_l207_207211

noncomputable def side_a := 32
noncomputable def side_b := 40
noncomputable def side_c := 24
noncomputable def ic := 18
noncomputable def expected_inradius := 2 * Real.sqrt 17

theorem find_inradius_of_scalene_triangle (a b c : ℝ) (h : a = side_a) (h1 : b = side_b) (h2 : c = side_c) (ic_length : ℝ) (h3: ic_length = ic) : (Real.sqrt (ic_length ^ 2 - (b - ((a + b - c) / 2)) ^ 2)) = expected_inradius :=
by
  sorry

end find_inradius_of_scalene_triangle_l207_207211


namespace part1_part2_l207_207508

def f (x : ℝ) : ℝ := |x - 1| + |x + 1|

theorem part1 (x : ℝ) : f x ≥ 3 ↔ x ≤ -3 / 2 ∨ x ≥ 3 / 2 := 
  sorry

theorem part2 (a : ℝ) : (∀ x : ℝ, f x ≥ a^2 - a) ↔ -1 ≤ a ∧ a ≤ 2 :=
  sorry

end part1_part2_l207_207508


namespace num_students_left_l207_207316

variable (Joe_weight : ℝ := 45)
variable (original_avg_weight : ℝ := 30)
variable (new_avg_weight : ℝ := 31)
variable (final_avg_weight : ℝ := 30)
variable (diff_avg_weight : ℝ := 7.5)

theorem num_students_left (n : ℕ) (x : ℕ) (W : ℝ := n * original_avg_weight)
  (new_W : ℝ := W + Joe_weight) (A : ℝ := Joe_weight - diff_avg_weight) : 
  new_W = (n + 1) * new_avg_weight →
  W + Joe_weight - x * A = (n + 1 - x) * final_avg_weight →
  x = 2 :=
by
  sorry

end num_students_left_l207_207316


namespace analects_deductive_reasoning_l207_207362

theorem analects_deductive_reasoning :
  (∀ (P Q R S T U V : Prop), 
    (P → Q) → 
    (Q → R) → 
    (R → S) → 
    (S → T) → 
    (T → U) → 
    ((P → U) ↔ deductive_reasoning)) :=
sorry

end analects_deductive_reasoning_l207_207362


namespace total_wet_surface_area_is_62_l207_207076

-- Define the dimensions of the cistern
def length_cistern : ℝ := 8
def width_cistern : ℝ := 4
def depth_water : ℝ := 1.25

-- Define the calculation of the wet surface area
def bottom_surface_area : ℝ := length_cistern * width_cistern
def longer_side_surface_area : ℝ := length_cistern * depth_water * 2
def shorter_end_surface_area : ℝ := width_cistern * depth_water * 2

-- Sum up all wet surface areas
def total_wet_surface_area : ℝ := bottom_surface_area + longer_side_surface_area + shorter_end_surface_area

-- The theorem stating that the total wet surface area is 62 m²
theorem total_wet_surface_area_is_62 : total_wet_surface_area = 62 := by
  sorry

end total_wet_surface_area_is_62_l207_207076


namespace find_fourth_number_l207_207225

theorem find_fourth_number (a : ℕ → ℕ) (h1 : a 7 = 42) (h2 : a 9 = 110)
  (h3 : ∀ n, n ≥ 3 → a n = a (n - 1) + a (n - 2)) : a 4 = 10 := 
sorry

end find_fourth_number_l207_207225


namespace friends_receive_pens_l207_207394

-- Define the given conditions
def packs_kendra : ℕ := 4
def packs_tony : ℕ := 2
def pens_per_pack : ℕ := 3
def pens_kept_per_person : ℕ := 2

-- Define the proof problem
theorem friends_receive_pens :
  (packs_kendra * pens_per_pack + packs_tony * pens_per_pack - (pens_kept_per_person * 2)) = 14 :=
by sorry

end friends_receive_pens_l207_207394


namespace cost_comparison_for_30_pens_l207_207254

def cost_store_a (x : ℕ) : ℝ :=
  if x > 10 then 0.9 * x + 6
  else 1.5 * x

def cost_store_b (x : ℕ) : ℝ :=
  1.2 * x

theorem cost_comparison_for_30_pens :
  cost_store_a 30 < cost_store_b 30 :=
by
  have store_a_cost : cost_store_a 30 = 0.9 * 30 + 6 := by rfl
  have store_b_cost : cost_store_b 30 = 1.2 * 30 := by rfl
  rw [store_a_cost, store_b_cost]
  sorry

end cost_comparison_for_30_pens_l207_207254


namespace fraction_in_orange_tin_l207_207472

variables {C : ℕ} -- assume total number of cookies as a natural number

theorem fraction_in_orange_tin (h1 : 11 / 12 = (1 / 6) + (5 / 12) + w)
  (h2 : 1 - (11 / 12) = 1 / 12) :
  w = 1 / 3 :=
by
  sorry

end fraction_in_orange_tin_l207_207472


namespace sum_of_squares_edges_l207_207250

-- Define Points
structure Point where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define given conditions (4 vertices each on two parallel planes)
def A1 : Point := { x := 0, y := 0, z := 0 }
def A2 : Point := { x := 1, y := 0, z := 0 }
def A3 : Point := { x := 1, y := 1, z := 0 }
def A4 : Point := { x := 0, y := 1, z := 0 }

def B1 : Point := { x := 0, y := 0, z := 1 }
def B2 : Point := { x := 1, y := 0, z := 1 }
def B3 : Point := { x := 1, y := 1, z := 1 }
def B4 : Point := { x := 0, y := 1, z := 1 }

-- Function to calculate distance squared between two points
def dist_sq (p1 p2 : Point) : ℝ :=
  (p1.x - p2.x) ^ 2 + (p1.y - p2.y) ^ 2 + (p1.z - p2.z) ^ 2

-- The Theorem to be proven
theorem sum_of_squares_edges : dist_sq A1 B2 + dist_sq A2 B3 + dist_sq A3 B4 + dist_sq A4 B1 = 8 := by
  sorry

end sum_of_squares_edges_l207_207250


namespace equal_roots_of_quadratic_eq_l207_207974

theorem equal_roots_of_quadratic_eq (n : ℝ) : (∃ x : ℝ, (x^2 - x + n = 0) ∧ (Δ = 0)) ↔ n = 1 / 4 :=
by
  have h₁ : Δ = 0 := by sorry  -- The discriminant condition
  sorry  -- Placeholder for completing the theorem proof

end equal_roots_of_quadratic_eq_l207_207974


namespace soccer_balls_percentage_holes_l207_207259

variable (x : ℕ)

theorem soccer_balls_percentage_holes 
    (h1 : ∃ x, 0 ≤ x ∧ x ≤ 100)
    (h2 : 48 = 80 * (100 - x) / 100) : 
  x = 40 := sorry

end soccer_balls_percentage_holes_l207_207259


namespace investmentAmounts_l207_207401

variable (totalInvestment : ℝ) (bonds stocks mutualFunds : ℝ)

-- Given conditions
def conditions := 
  totalInvestment = 210000 ∧
  stocks = 2 * bonds ∧
  mutualFunds = 4 * stocks ∧
  bonds + stocks + mutualFunds = totalInvestment

-- Prove the investments
theorem investmentAmounts (h : conditions totalInvestment bonds stocks mutualFunds) :
  bonds = 19090.91 ∧ stocks = 38181.82 ∧ mutualFunds = 152727.27 :=
sorry

end investmentAmounts_l207_207401


namespace females_with_advanced_degrees_l207_207049

theorem females_with_advanced_degrees 
  (total_employees : ℕ) 
  (total_females : ℕ) 
  (total_advanced_degrees : ℕ) 
  (total_college_degrees : ℕ) 
  (males_with_college_degree : ℕ) 
  (h1 : total_employees = 180) 
  (h2 : total_females = 110) 
  (h3 : total_advanced_degrees = 90) 
  (h4 : total_college_degrees = 90) 
  (h5 : males_with_college_degree = 35) : 
  ∃ (females_with_advanced_degrees : ℕ), females_with_advanced_degrees = 55 := 
by {
  sorry
}

end females_with_advanced_degrees_l207_207049


namespace range_of_heights_l207_207418

theorem range_of_heights (max_height min_height : ℝ) (h_max : max_height = 175) (h_min : min_height = 100) :
  (max_height - min_height) = 75 :=
by
  -- Defer proof
  sorry

end range_of_heights_l207_207418


namespace train_speed_including_stoppages_l207_207791

noncomputable def trainSpeedExcludingStoppages : ℝ := 45
noncomputable def stoppageTimePerHour : ℝ := 20 / 60 -- 20 minutes per hour converted to hours
noncomputable def runningTimePerHour : ℝ := 1 - stoppageTimePerHour

theorem train_speed_including_stoppages (speed : ℝ) (stoppage : ℝ) (running_time : ℝ) : 
  speed = 45 → stoppage = 20 / 60 → running_time = 1 - stoppage → 
  (speed * running_time) / 1 = 30 :=
by sorry

end train_speed_including_stoppages_l207_207791


namespace wall_volume_is_128512_l207_207014

noncomputable def wall_volume (width : ℝ) (height : ℝ) (length : ℝ) : ℝ :=
  width * height * length

theorem wall_volume_is_128512 : 
  ∀ (w : ℝ) (h : ℝ) (l : ℝ), 
  h = 6 * w ∧ l = 7 * h ∧ w = 8 → 
  wall_volume w h l = 128512 := 
by
  sorry

end wall_volume_is_128512_l207_207014


namespace intersection_M_N_l207_207580

def M : Set ℝ := { x : ℝ | x^2 > 4 }
def N : Set ℝ := { x : ℝ | x = -3 ∨ x = -2 ∨ x = 2 ∨ x = 3 ∨ x = 4 }

theorem intersection_M_N : M ∩ N = { x : ℝ | x = -3 ∨ x = 3 ∨ x = 4 } :=
by
  sorry

end intersection_M_N_l207_207580


namespace min_value_l207_207957

theorem min_value (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : a + b = 1) : 
  ∃ m : ℝ, m = 3 + 2 * Real.sqrt 2 ∧ (∀ x y, x > 0 → y > 0 → x + y = 1 → (1 / x + 2 / y) ≥ m) := 
sorry

end min_value_l207_207957


namespace greg_needs_additional_amount_l207_207808

def total_cost : ℤ := 90
def saved_amount : ℤ := 57
def additional_amount_needed : ℤ := total_cost - saved_amount

theorem greg_needs_additional_amount :
  additional_amount_needed = 33 :=
by
  sorry

end greg_needs_additional_amount_l207_207808


namespace pens_at_end_l207_207768

-- Define the main variable
variable (x : ℝ)

-- Define the conditions as functions
def initial_pens (x : ℝ) := x
def mike_gives (x : ℝ) := 0.5 * x
def after_mike (x : ℝ) := x + (mike_gives x)
def after_cindy (x : ℝ) := 2 * (after_mike x)
def give_sharon (x : ℝ) := 0.25 * (after_cindy x)

-- Define the final number of pens
def final_pens (x : ℝ) := (after_cindy x) - (give_sharon x)

-- The theorem statement
theorem pens_at_end (x : ℝ) : final_pens x = 2.25 * x :=
by sorry

end pens_at_end_l207_207768


namespace minimize_quadratic_expression_l207_207269

theorem minimize_quadratic_expression : ∃ x : ℝ, (∀ y : ℝ, 3 * x^2 - 18 * x + 7 ≤ 3 * y^2 - 18 * y + 7) ∧ x = 3 :=
by
  sorry

end minimize_quadratic_expression_l207_207269


namespace find_intersection_point_l207_207586

-- Define the problem conditions and question in Lean
theorem find_intersection_point 
  (slope_l1 : ℝ) (slope_l2 : ℝ) (p : ℝ × ℝ) (P : ℝ × ℝ)
  (h_l1_slope : slope_l1 = 2) 
  (h_parallel : slope_l1 = slope_l2)
  (h_passes_through : p = (-1, 1)) :
  P = (0, 3) := sorry

end find_intersection_point_l207_207586


namespace proof_problem_l207_207190

theorem proof_problem (a b : ℝ) (h1 : a + b = 8) (h2 : a - b = 4) : a^2 - b^2 + 2 * a * b = 64 :=
sorry

end proof_problem_l207_207190


namespace smallest_set_handshakes_l207_207744

-- Define the number of people
def num_people : Nat := 36

-- Define a type for people
inductive Person : Type
| a : Fin num_people → Person

-- Define the handshake relationship
def handshake (p1 p2 : Person) : Prop :=
  match p1, p2 with
  | Person.a i, Person.a j => i.val = (j.val + 1) % num_people ∨ j.val = (i.val + 1) % num_people

-- Define the problem statement
theorem smallest_set_handshakes :
  ∃ s : Finset Person, (∀ p : Person, p ∈ s ∨ ∃ q ∈ s, handshake p q) ∧ s.card = 18 :=
sorry

end smallest_set_handshakes_l207_207744


namespace pool_length_calc_l207_207245

variable (total_water : ℕ) (drinking_cooking_water : ℕ) (shower_water : ℕ) (shower_count : ℕ)
variable (pool_width : ℕ) (pool_height : ℕ) (pool_volume : ℕ)

theorem pool_length_calc (h1 : total_water = 1000)
    (h2 : drinking_cooking_water = 100)
    (h3 : shower_water = 20)
    (h4 : shower_count = 15)
    (h5 : pool_width = 10)
    (h6 : pool_height = 6)
    (h7 : pool_volume = total_water - (drinking_cooking_water + shower_water * shower_count)) :
    pool_volume = 600 →
    pool_volume = 60 * length →
    length = 10 :=
by
  sorry

end pool_length_calc_l207_207245


namespace find_rate_of_interest_l207_207083

noncomputable def interest_rate (P R : ℝ) : Prop :=
  (400 = P * (1 + 4 * R / 100)) ∧ (500 = P * (1 + 6 * R / 100))

theorem find_rate_of_interest (R : ℝ) (P : ℝ) (h : interest_rate P R) :
  R = 25 :=
by
  sorry

end find_rate_of_interest_l207_207083


namespace part1_1_part1_2_part1_3_part2_l207_207760

def operation (a b c : ℝ) : Prop := a^c = b

theorem part1_1 : operation 3 81 4 :=
by sorry

theorem part1_2 : operation 4 1 0 :=
by sorry

theorem part1_3 : operation 2 (1 / 4) (-2) :=
by sorry

theorem part2 (x y z : ℝ) (h1 : operation 3 7 x) (h2 : operation 3 8 y) (h3 : operation 3 56 z) : x + y = z :=
by sorry

end part1_1_part1_2_part1_3_part2_l207_207760


namespace shortest_chord_eqn_of_circle_l207_207646

theorem shortest_chord_eqn_of_circle 
    (k x y : ℝ)
    (C_eq : x^2 + y^2 - 2*x - 24 = 0)
    (line_l : y = k * (x - 2) - 1) :
  y = x - 3 :=
by
  sorry

end shortest_chord_eqn_of_circle_l207_207646


namespace probability_black_then_red_l207_207492

/-- Definition of a standard deck -/
def standard_deck := {cards : Finset (Fin 52) // cards.card = 52}

/-- Definition of black cards in the deck -/
def black_cards := {cards : Finset (Fin 52) // cards.card = 26}

/-- Definition of red cards in the deck -/
def red_cards := {cards : Finset (Fin 52) // cards.card = 26}

/-- Probability of drawing the top card as black and the second card as red -/
def prob_black_then_red (deck : standard_deck) (black : black_cards) (red : red_cards) : ℚ :=
  (26 * 26) / (52 * 51)

theorem probability_black_then_red (deck : standard_deck) (black : black_cards) (red : red_cards) :
  prob_black_then_red deck black red = 13 / 51 :=
sorry

end probability_black_then_red_l207_207492


namespace solve_for_q_l207_207577

theorem solve_for_q (p q : ℚ) (h1 : 5 * p + 6 * q = 14) (h2 : 6 * p + 5 * q = 17) : q = -1 / 11 :=
by
  sorry

end solve_for_q_l207_207577


namespace product_of_two_numbers_l207_207728

theorem product_of_two_numbers :
  ∀ x y: ℝ, 
  ((x - y)^2) / ((x + y)^3) = 4 / 27 → 
  x + y = 5 * (x - y) + 3 → 
  x * y = 15.75 :=
by 
  intro x y
  sorry

end product_of_two_numbers_l207_207728


namespace area_product_equal_no_consecutive_integers_l207_207743

open Real

-- Define the areas of the triangles for quadrilateral ABCD
variables {A B C D O : Point} 
variables {S1 S2 S3 S4 : Real}  -- Areas of triangles ABO, BCO, CDO, DAO

-- Given conditions
variables (h_intersection : lies_on_intersection O AC BD)
variables (h_areas : S1 = 1 / 2 * (|AO| * |BM|) ∧ S2 = 1 / 2 * (|CO| * |BM|) ∧ S3 = 1 / 2 * (|CO| * |DN|) ∧ S4 = 1 / 2 * (|AO| * |DN|))

-- Theorem for part (a)
theorem area_product_equal : S1 * S3 = S2 * S4 :=
by sorry

-- Theorem for part (b)
theorem no_consecutive_integers : ¬∃ (n : ℕ), S1 = n ∧ S2 = n + 1 ∧ S3 = n + 2 ∧ S4 = n + 3 :=
by sorry

end area_product_equal_no_consecutive_integers_l207_207743


namespace sphere_volume_from_surface_area_l207_207523

theorem sphere_volume_from_surface_area (S : ℝ) (V : ℝ) (R : ℝ) (h1 : S = 36 * Real.pi) (h2 : S = 4 * Real.pi * R ^ 2) (h3 : V = (4 / 3) * Real.pi * R ^ 3) : V = 36 * Real.pi :=
by
  sorry

end sphere_volume_from_surface_area_l207_207523


namespace portion_of_larger_jar_full_l207_207688

noncomputable def smaller_jar_capacity (S L : ℝ) : Prop :=
  (1 / 5) * S = (1 / 4) * L

noncomputable def larger_jar_capacity (L : ℝ) : ℝ :=
  (1 / 5) * (5 / 4) * L

theorem portion_of_larger_jar_full (S L : ℝ) 
  (h1 : smaller_jar_capacity S L) : 
  (1 / 4) * L + (1 / 4) * L = (1 / 2) * L := 
sorry

end portion_of_larger_jar_full_l207_207688


namespace xiao_ming_cube_division_l207_207644

theorem xiao_ming_cube_division (large_edge small_cubes : ℕ)
  (large_edge_eq : large_edge = 4)
  (small_cubes_eq : small_cubes = 29)
  (total_volume : large_edge ^ 3 = 64) :
  ∃ (small_edge_1_cube : ℕ), small_edge_1_cube = 24 ∧ small_cubes = 29 ∧ 
  small_edge_1_cube + (small_cubes - small_edge_1_cube) * 8 = 64 := 
by
  -- We only need to assert the existence here as per the instruction.
  sorry

end xiao_ming_cube_division_l207_207644


namespace highest_slope_product_l207_207090

theorem highest_slope_product (m1 m2 : ℝ) (h1 : m1 = 5 * m2) 
    (h2 : abs ((m2 - m1) / (1 + m1 * m2)) = 1) : (m1 * m2) ≤ 1.8 :=
by
  sorry

end highest_slope_product_l207_207090


namespace greatest_possible_value_of_a_l207_207197

theorem greatest_possible_value_of_a :
  ∃ (a : ℕ), (∀ x : ℤ, x * (x + a) = -24 → x * (x + a) = -24) ∧ (∀ b : ℕ, (∀ x : ℤ, x * (x + b) = -24 → x * (x + b) = -24) → b ≤ a) ∧ a = 25 :=
sorry

end greatest_possible_value_of_a_l207_207197


namespace alex_silver_tokens_l207_207757

theorem alex_silver_tokens :
  ∃ x y : ℕ, 
    (100 - 3 * x + y ≤ 2) ∧ 
    (50 + 2 * x - 4 * y ≤ 3) ∧
    (x + y = 74) :=
by
  sorry

end alex_silver_tokens_l207_207757


namespace isosceles_triangle_largest_angle_l207_207176

theorem isosceles_triangle_largest_angle (α : ℝ) (β : ℝ)
  (h1 : 0 < α) (h2 : α = 30) (h3 : β = 30):
  ∃ γ : ℝ, γ = 180 - 2 * α ∧ γ = 120 := by
  sorry

end isosceles_triangle_largest_angle_l207_207176


namespace vivi_total_yards_l207_207113

theorem vivi_total_yards (spent_checkered spent_plain cost_per_yard : ℝ)
  (h1 : spent_checkered = 75)
  (h2 : spent_plain = 45)
  (h3 : cost_per_yard = 7.50) :
  (spent_checkered / cost_per_yard + spent_plain / cost_per_yard) = 16 :=
by 
  sorry

end vivi_total_yards_l207_207113


namespace geometric_progression_sum_of_cubes_l207_207755

theorem geometric_progression_sum_of_cubes :
  ∃ (a r : ℕ) (seq : Fin 6 → ℕ), (seq 0 = a) ∧ (seq 1 = a * r) ∧ (seq 2 = a * r^2) ∧ (seq 3 = a * r^3) ∧ (seq 4 = a * r^4) ∧ (seq 5 = a * r^5) ∧
  (∀ i, 0 ≤ seq i ∧ seq i < 100) ∧
  (seq 0 + seq 1 + seq 2 + seq 3 + seq 4 + seq 5 = 326) ∧
  (∃ T : ℕ, (∀ i, ∃ k, seq i = k^3 → k * k * k = seq i) ∧ T = 64) :=
sorry

end geometric_progression_sum_of_cubes_l207_207755


namespace test_total_points_l207_207862

def total_points (total_problems comp_problems : ℕ) (points_comp points_word : ℕ) : ℕ :=
  let word_problems := total_problems - comp_problems
  (comp_problems * points_comp) + (word_problems * points_word)

theorem test_total_points :
  total_points 30 20 3 5 = 110 := by
  sorry

end test_total_points_l207_207862


namespace prime_product_div_by_four_l207_207656

theorem prime_product_div_by_four 
  (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hpq1 : Nat.Prime (p * q + 1)) : 
  4 ∣ (2 * p + q) * (p + 2 * q) := 
sorry

end prime_product_div_by_four_l207_207656


namespace largest_multiple_of_11_less_than_minus_150_l207_207015

theorem largest_multiple_of_11_less_than_minus_150 : 
  ∃ n : ℤ, (n * 11 < -150) ∧ (∀ m : ℤ, (m * 11 < -150) →  n * 11 ≥ m * 11) ∧ (n * 11 = -154) :=
by
  sorry

end largest_multiple_of_11_less_than_minus_150_l207_207015


namespace alpha_beta_sum_two_l207_207272

theorem alpha_beta_sum_two (α β : ℝ) 
  (hα : α^3 - 3 * α^2 + 5 * α - 17 = 0)
  (hβ : β^3 - 3 * β^2 + 5 * β + 11 = 0) : 
  α + β = 2 :=
by
  sorry

end alpha_beta_sum_two_l207_207272


namespace olivia_insurance_premium_l207_207967

theorem olivia_insurance_premium :
  ∀ (P : ℕ) (base_premium accident_percentage ticket_cost : ℤ) (tickets accidents : ℕ),
    base_premium = 50 →
    accident_percentage = P →
    ticket_cost = 5 →
    tickets = 3 →
    accidents = 1 →
    (base_premium + (accidents * base_premium * P / 100) + (tickets * ticket_cost) = 70) →
    P = 10 :=
by
  intros P base_premium accident_percentage ticket_cost tickets accidents
  intro h1 h2 h3 h4 h5 h6
  sorry

end olivia_insurance_premium_l207_207967


namespace total_items_l207_207598

theorem total_items (B M C : ℕ) 
  (h1 : B = 58) 
  (h2 : B = M + 18) 
  (h3 : B = C - 27) : 
  B + M + C = 183 :=
by 
  sorry

end total_items_l207_207598


namespace number_of_girls_on_playground_l207_207889

theorem number_of_girls_on_playground (boys girls total : ℕ) 
  (h1 : boys = 44) (h2 : total = 97) (h3 : total = boys + girls) : 
  girls = 53 :=
by sorry

end number_of_girls_on_playground_l207_207889


namespace diamonds_count_l207_207501

-- Definitions based on the conditions given in the problem
def totalGems : Nat := 5155
def rubies : Nat := 5110
def diamonds (total rubies : Nat) : Nat := total - rubies

-- Statement of the proof problem
theorem diamonds_count : diamonds totalGems rubies = 45 := by
  sorry

end diamonds_count_l207_207501


namespace smallest_prime_linear_pair_l207_207460

def is_prime (n : ℕ) : Prop := ¬(∃ k > 1, k < n ∧ k ∣ n)

theorem smallest_prime_linear_pair :
  ∃ a b : ℕ, is_prime a ∧ is_prime b ∧ a + b = 180 ∧ a > b ∧ b = 7 := 
by
  sorry

end smallest_prime_linear_pair_l207_207460


namespace problem_l207_207905

theorem problem
  (a b : ℝ)
  (h : a^2 + b^2 - 2 * a + 6 * b + 10 = 0) :
  2 * a^100 - 3 * b⁻¹ = 3 := 
by {
  -- Proof steps go here
  sorry
}

end problem_l207_207905


namespace colton_share_l207_207310

-- Definitions
def footToInch (foot : ℕ) : ℕ := 12 * foot -- 1 foot equals 12 inches

-- Problem conditions
def coltonBurgerLength := footToInch 1 -- Colton bought a foot long burger
def sharedBurger (length : ℕ) : ℕ := length / 2 -- shared half with his brother

-- Equivalent proof problem statement
theorem colton_share : sharedBurger coltonBurgerLength = 6 := 
by sorry

end colton_share_l207_207310


namespace mary_age_l207_207935

theorem mary_age :
  ∃ M R : ℕ, (R = M + 30) ∧ (R + 20 = 2 * (M + 20)) ∧ (M = 10) :=
by
  sorry

end mary_age_l207_207935


namespace find_number_l207_207696

theorem find_number (x : ℝ) (h : 0.20 * x = 0.20 * 650 + 190) : x = 1600 := by 
  sorry

end find_number_l207_207696


namespace inequality_solution_set_l207_207277

theorem inequality_solution_set (x : ℝ) :
  2 * x^2 - x ≤ 0 → 0 ≤ x ∧ x ≤ 1 / 2 :=
sorry

end inequality_solution_set_l207_207277


namespace parabola_y_intercepts_l207_207836

theorem parabola_y_intercepts : ∃ y1 y2 : ℝ, (3 * y1^2 - 6 * y1 + 2 = 0) ∧ (3 * y2^2 - 6 * y2 + 2 = 0) ∧ (y1 ≠ y2) :=
by 
  sorry

end parabola_y_intercepts_l207_207836


namespace jan_drove_more_l207_207707

variables (d t s : ℕ)
variables (h h_ans : ℕ)
variables (ha_speed j_speed : ℕ)
variables (j d_plus : ℕ)

-- Ian's equation
def ian_distance (s t : ℕ) : ℕ := s * t

-- Han's additional conditions
def han_distance (s t : ℕ) (h_speed : ℕ)
    (d_plus : ℕ) : Prop :=
  d_plus + 120 = (s + h_speed) * (t + 2)

-- Jan's conditions and equation
def jan_distance (s t : ℕ) (j_speed : ℕ) : ℕ :=
  (s + j_speed) * (t + 3)

-- Proof statement
theorem jan_drove_more (d t s h_ans : ℕ)
    (h_speed j_speed : ℕ) (d_plus : ℕ)
    (h_dist_cond : han_distance s t h_speed d_plus)
    (j_dist_cond : jan_distance s t j_speed = h_ans) :
  h_ans = 195 :=
sorry

end jan_drove_more_l207_207707


namespace correct_statements_l207_207896

-- Definitions for each statement
def statement_1 := ∀ p q : ℤ, q ≠ 0 → (∃ n : ℤ, ∃ d : ℤ, p = n ∧ q = d ∧ (n, d) = (p, q))
def statement_2 := ∀ r : ℚ, (r > 0 ∨ r < 0) ∨ (∃ d : ℚ, d ≥ 0)
def statement_3 := ∀ x y : ℚ, abs x = abs y → x = y
def statement_4 := ∀ x : ℚ, (-x = x ∧ abs x = x) → x = 0
def statement_5 := ∀ x y : ℚ, abs x > abs y → x > y
def statement_6 := (∃ n : ℕ, n > 0) ∧ (∀ r : ℚ, r > 0 → ∃ q : ℚ, q > 0 ∧ q < r)

-- Main theorem: Prove that exactly 3 statements are correct
theorem correct_statements : 
  (statement_1 ∧ statement_4 ∧ statement_6) ∧ 
  (¬ statement_2 ∧ ¬ statement_3 ∧ ¬ statement_5) :=
by
  sorry

end correct_statements_l207_207896


namespace multiples_of_6_and_8_l207_207777

open Nat

theorem multiples_of_6_and_8 (n m k : ℕ) (h₁ : n = 33) (h₂ : m = 25) (h₃ : k = 8) :
  (n - k) + (m - k) = 42 :=
by
  sorry

end multiples_of_6_and_8_l207_207777


namespace probability_first_or_second_l207_207454

/-- Define the events and their probabilities --/
def prob_hit_first_sector : ℝ := 0.4
def prob_hit_second_sector : ℝ := 0.3
def prob_hit_first_or_second : ℝ := 0.7

/-- The proof that these probabilities add up as mutually exclusive events --/
theorem probability_first_or_second (P_A : ℝ) (P_B : ℝ) (P_A_or_B : ℝ) (hP_A : P_A = prob_hit_first_sector) (hP_B : P_B = prob_hit_second_sector) (hP_A_or_B : P_A_or_B = prob_hit_first_or_second) :
  P_A_or_B = P_A + P_B := 
  by
    rw [hP_A, hP_B, hP_A_or_B]
    sorry

end probability_first_or_second_l207_207454


namespace triangle_shortest_side_l207_207268

theorem triangle_shortest_side (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] 
    (base : Real) (base_angle : Real) (sum_other_sides : Real)
    (h1 : base = 80) 
    (h2 : base_angle = 60) 
    (h3 : sum_other_sides = 90) : 
    ∃ shortest_side : Real, shortest_side = 17 :=
by 
    sorry

end triangle_shortest_side_l207_207268


namespace gcd_lcm_relation_gcd3_lcm3_relation_lcm3_gcd3_relation_l207_207279

-- GCD as the greatest common divisor
def GCD (a b : ℕ) : ℕ := Nat.gcd a b

-- LCM as the least common multiple
def LCM (a b : ℕ) : ℕ := Nat.lcm a b

-- First proof problem in Lean 4
theorem gcd_lcm_relation (a b : ℕ) : GCD a b = (a * b) / (LCM a b) :=
  sorry

-- GCD function extended to three arguments
def GCD3 (a b c : ℕ) : ℕ := Nat.gcd (Nat.gcd a b) c

-- LCM function extended to three arguments
def LCM3 (a b c : ℕ) : ℕ := Nat.lcm (Nat.lcm a b) c

-- Second proof problem in Lean 4
theorem gcd3_lcm3_relation (a b c : ℕ) : GCD3 a b c = (a * b * c * LCM3 a b c) / (LCM a b * LCM b c * LCM c a) :=
  sorry

-- Third proof problem in Lean 4
theorem lcm3_gcd3_relation (a b c : ℕ) : LCM3 a b c = (a * b * c * GCD3 a b c) / (GCD a b * GCD b c * GCD c a) :=
  sorry

end gcd_lcm_relation_gcd3_lcm3_relation_lcm3_gcd3_relation_l207_207279


namespace range_of_a_minus_b_l207_207431

theorem range_of_a_minus_b (a b : ℝ) (h₁ : -1 < a) (h₂ : a < 2) (h₃ : -2 < b) (h₄ : b < 1) :
  -2 < a - b ∧ a - b < 4 :=
by
  sorry

end range_of_a_minus_b_l207_207431


namespace isosceles_triangle_largest_angle_l207_207610

/-- 
  Given an isosceles triangle where one of the angles is 20% smaller than a right angle,
  prove that the measure of one of the two largest angles is 54 degrees.
-/
theorem isosceles_triangle_largest_angle 
  (A B C : ℝ) 
  (triangle_ABC : A + B + C = 180)
  (isosceles_triangle : A = B ∨ A = C ∨ B = C)
  (smaller_angle : A = 0.80 * 90) :
  A = 54 ∨ B = 54 ∨ C = 54 :=
sorry

end isosceles_triangle_largest_angle_l207_207610


namespace part1_part2_part3_l207_207958

-- Definitions based on conditions
def fractional_eq (x a : ℝ) : Prop := (x - a) / (x - 2) - 5 / x = 1

-- Part (1): Proof statement for a == -1 if x == 5 is a root
theorem part1 (x : ℝ) (a : ℝ) (h : x = 5) (heq : fractional_eq x a) : a = -1 :=
sorry

-- Part (2): Proof statement for a == 2 if the equation has a double root
theorem part2 (a : ℝ) (h_double_root : ∀ x, fractional_eq x a → x = 0 ∨ x = 2) : a = 2 :=
sorry

-- Part (3): Proof statement for a == -3 or == 2 if the equation has no solution
theorem part3 (a : ℝ) (h_no_solution : ¬∃ x, fractional_eq x a) : a = -3 ∨ a = 2 :=
sorry

end part1_part2_part3_l207_207958


namespace trains_speed_ratio_l207_207297

-- Define the conditions
variables (V1 V2 L1 L2 : ℝ)
axiom time1 : L1 = 27 * V1
axiom time2 : L2 = 17 * V2
axiom timeTogether : L1 + L2 = 22 * (V1 + V2)

-- The theorem to prove the ratio of the speeds
theorem trains_speed_ratio : V1 / V2 = 7.8 :=
sorry

end trains_speed_ratio_l207_207297


namespace value_of_a_l207_207171

theorem value_of_a (a : ℝ) (x : ℝ) (h : 2 * x + 3 * a = -1) (hx : x = 1) : a = -1 :=
by
  sorry

end value_of_a_l207_207171


namespace radius_of_scrap_cookie_l207_207143

theorem radius_of_scrap_cookie
  (r_cookies : ℝ) (n_cookies : ℕ) (radius_layout : Prop)
  (circle_diameter_twice_width : Prop) :
  (r_cookies = 0.5 ∧ n_cookies = 9 ∧ radius_layout ∧ circle_diameter_twice_width)
  →
  (∃ r_scrap : ℝ, r_scrap = Real.sqrt 6.75) :=
by
  sorry

end radius_of_scrap_cookie_l207_207143


namespace cube_surface_area_with_holes_l207_207326

theorem cube_surface_area_with_holes 
    (edge_length : ℝ) 
    (hole_side_length : ℝ) 
    (num_faces : ℕ) 
    (parallel_edges : Prop)
    (holes_centered : Prop)
    (h_edge : edge_length = 5)
    (h_hole : hole_side_length = 2)
    (h_faces : num_faces = 6)
    (h_inside_area : parallel_edges ∧ holes_centered)
    : (150 - 24 + 96 = 222) :=
by
    sorry

end cube_surface_area_with_holes_l207_207326


namespace no_integer_solutions_for_mn_squared_eq_1980_l207_207792

theorem no_integer_solutions_for_mn_squared_eq_1980 :
  ¬ ∃ m n : ℤ, m^2 + n^2 = 1980 := 
sorry

end no_integer_solutions_for_mn_squared_eq_1980_l207_207792


namespace balloon_ratio_l207_207227

theorem balloon_ratio 
  (initial_blue : ℕ) (initial_purple : ℕ) (balloons_left : ℕ)
  (h1 : initial_blue = 303)
  (h2 : initial_purple = 453)
  (h3 : balloons_left = 378) :
  (balloons_left / (initial_blue + initial_purple) : ℚ) = (1 / 2 : ℚ) :=
by
  sorry

end balloon_ratio_l207_207227


namespace margo_total_distance_l207_207010

theorem margo_total_distance
  (t1 t2 : ℚ) (rate1 rate2 : ℚ)
  (h1 : t1 = 15 / 60)
  (h2 : t2 = 25 / 60)
  (r1 : rate1 = 5)
  (r2 : rate2 = 3) :
  (t1 * rate1 + t2 * rate2 = 2.5) :=
by
  sorry

end margo_total_distance_l207_207010


namespace sequence_general_term_and_sum_l207_207709

theorem sequence_general_term_and_sum (a_n : ℕ → ℕ) (b_n S_n : ℕ → ℕ) :
  (∀ n, a_n n = 2 ^ n) ∧ (∀ n, b_n n = a_n n * (Real.logb 2 (a_n n)) ∧
  S_n n = (n - 1) * 2 ^ (n + 1) + 2) :=
by
  sorry

end sequence_general_term_and_sum_l207_207709


namespace arithmetic_sequence_part_a_arithmetic_sequence_part_b_l207_207647

theorem arithmetic_sequence_part_a (e u k : ℕ) (n : ℕ) 
  (h1 : e = 1) 
  (h2 : u = 1000) 
  (h3 : k = 343) 
  (h4 : n = 100) : ¬ (∃ d m, e + (m - 1) * d = k ∧ u = e + (n - 1) * d ∧ 1 < m ∧ m < n) :=
by sorry

theorem arithmetic_sequence_part_b (e u k : ℝ) (n : ℕ) 
  (h1 : e = 81 * Real.sqrt 2 - 64 * Real.sqrt 3) 
  (h2 : u = 54 * Real.sqrt 2 - 28 * Real.sqrt 3)
  (h3 : k = 69 * Real.sqrt 2 - 48 * Real.sqrt 3)
  (h4 : n = 100) : (∃ d m, e + (m - 1) * d = k ∧ u = e + (n - 1) * d ∧ 1 < m ∧ m < n) :=
by sorry

end arithmetic_sequence_part_a_arithmetic_sequence_part_b_l207_207647


namespace range_of_x_l207_207637

noncomputable def f : ℝ → ℝ := sorry  -- f is an even function and decreasing on [0, +∞)

theorem range_of_x (x : ℝ) (h_even : ∀ x, f x = f (-x)) 
  (h_decreasing : ∀ x y, 0 ≤ x → x ≤ y → f x ≥ f y) 
  (h_condition : f (Real.log x) > f 1) : 
  1 / 10 < x ∧ x < 10 := 
sorry

end range_of_x_l207_207637


namespace cube_face_sum_l207_207875

theorem cube_face_sum (a d b e c f g : ℕ)
    (h1 : g = 2)
    (h2 : 2310 = 2 * 3 * 5 * 7 * 11)
    (h3 : (a + d) * (b + e) * (c + f) = 3 * 5 * 7 * 11):
    (a + d) + (b + e) + (c + f) = 47 :=
by
    sorry

end cube_face_sum_l207_207875


namespace train_b_leaves_after_train_a_l207_207320

noncomputable def time_difference := 2

theorem train_b_leaves_after_train_a 
  (speedA speedB distance t : ℝ) 
  (h1 : speedA = 30)
  (h2 : speedB = 38)
  (h3 : distance = 285)
  (h4 : distance = speedB * t)
  : time_difference = (distance - speedA * t) / speedA := 
by 
  sorry

end train_b_leaves_after_train_a_l207_207320


namespace comparison_1_comparison_2_l207_207172

noncomputable def expr1 := -(-((6: ℝ) / 7))
noncomputable def expr2 := -((abs (-((4: ℝ) / 5))))
noncomputable def expr3 := -((4: ℝ) / 5)
noncomputable def expr4 := -((2: ℝ) / 3)

theorem comparison_1 : expr1 > expr2 := sorry
theorem comparison_2 : expr3 < expr4 := sorry

end comparison_1_comparison_2_l207_207172


namespace sum_distinct_integers_l207_207085

theorem sum_distinct_integers (a b c d e : ℤ) (h1 : a ≠ b) (h2 : a ≠ c) (h3 : a ≠ d) (h4 : a ≠ e)
    (h5 : b ≠ c) (h6 : b ≠ d) (h7 : b ≠ e) (h8 : c ≠ d) (h9 : c ≠ e) (h10 : d ≠ e)
    (h : (5 - a) * (5 - b) * (5 - c) * (5 - d) * (5 - e) = 120) :
    a + b + c + d + e = 13 := by
  sorry

end sum_distinct_integers_l207_207085


namespace inequality_abc_l207_207185

theorem inequality_abc (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_prod : a * b * c = 1):
  (1 / (a^3 * (b + c)) + 1 / (b^3 * (c + a)) + 1 / (c^3 * (a + b))) ≥ (3 / 2) :=
by
  sorry

end inequality_abc_l207_207185


namespace find_length_of_bridge_l207_207698

noncomputable def length_of_train : ℝ := 165
noncomputable def speed_of_train_kmph : ℝ := 54
noncomputable def time_to_cross_bridge_seconds : ℝ := 67.66125376636536

noncomputable def speed_of_train_mps : ℝ :=
  speed_of_train_kmph * (1000 / 3600)

noncomputable def total_distance_covered : ℝ :=
  speed_of_train_mps * time_to_cross_bridge_seconds

noncomputable def length_of_bridge : ℝ :=
  total_distance_covered - length_of_train

theorem find_length_of_bridge : length_of_bridge = 849.92 := by
  sorry

end find_length_of_bridge_l207_207698


namespace playerA_winning_conditions_l207_207475

def playerA_has_winning_strategy (n : ℕ) : Prop :=
  (n % 4 = 0) ∨ (n % 4 = 3)

theorem playerA_winning_conditions (n : ℕ) (h : n ≥ 2) : 
  playerA_has_winning_strategy n ↔ (n % 4 = 0 ∨ n % 4 = 3) :=
by sorry

end playerA_winning_conditions_l207_207475


namespace cost_of_western_european_postcards_before_1980s_l207_207118

def germany_cost_1950s : ℝ := 5 * 0.07
def france_cost_1950s : ℝ := 8 * 0.05

def germany_cost_1960s : ℝ := 6 * 0.07
def france_cost_1960s : ℝ := 9 * 0.05

def germany_cost_1970s : ℝ := 11 * 0.07
def france_cost_1970s : ℝ := 10 * 0.05

def total_germany_cost : ℝ := germany_cost_1950s + germany_cost_1960s + germany_cost_1970s
def total_france_cost : ℝ := france_cost_1950s + france_cost_1960s + france_cost_1970s

def total_western_europe_cost : ℝ := total_germany_cost + total_france_cost

theorem cost_of_western_european_postcards_before_1980s :
  total_western_europe_cost = 2.89 := by
  sorry

end cost_of_western_european_postcards_before_1980s_l207_207118


namespace correct_operation_l207_207252

theorem correct_operation (a : ℕ) : a ^ 3 * a ^ 2 = a ^ 5 :=
by sorry

end correct_operation_l207_207252


namespace option_c_is_incorrect_l207_207522

/-- Define the temperature data -/
def temps : List Int := [-20, -10, 0, 10, 20, 30]

/-- Define the speed of sound data corresponding to the temperatures -/
def speeds : List Int := [318, 324, 330, 336, 342, 348]

/-- The speed of sound at 10 degrees Celsius -/
def speed_at_10 : Int := 336

/-- The incorrect claim in option C -/
def incorrect_claim : Prop := (speed_at_10 * 4 ≠ 1334)

/-- Prove that the claim in option C is incorrect -/
theorem option_c_is_incorrect : incorrect_claim :=
by {
  sorry
}

end option_c_is_incorrect_l207_207522


namespace max_value_of_angle_B_l207_207616

theorem max_value_of_angle_B (a b c : ℝ) (A B C : ℝ) 
  (h1: a + c = 2 * b)
  (h2: a^2 + b^2 - 2*a*b <= c^2 - 2*b*c - 2*a*c)
  (h3: A + B + C = π)
  (h4: 0 < A ∧ A < π) :  
  B ≤ π / 3 :=
sorry

end max_value_of_angle_B_l207_207616


namespace rope_cut_prob_l207_207759

theorem rope_cut_prob (x : ℝ) (hx : 0 < x) : 
  (∃ (a b : ℝ), a + b = 1 ∧ min a b ≤ max a b / x) → 
  (1 / (x + 1) * 2) = 2 / (x + 1) :=
sorry

end rope_cut_prob_l207_207759


namespace graph_passes_through_fixed_point_l207_207550

theorem graph_passes_through_fixed_point (a : ℝ) : (0, 2) ∈ { p : ℝ × ℝ | ∃ x, p = (x, a ^ x + 1) } :=
sorry

end graph_passes_through_fixed_point_l207_207550


namespace geometric_sequence_a9_l207_207983

theorem geometric_sequence_a9 (a : ℕ → ℝ) (q : ℝ) 
  (h1 : a 3 = 2) 
  (h2 : a 4 = 8 * a 7) 
  (h3 : ∀ n, a (n + 1) = a n * q) 
  (hq : q > 0) 
  : a 9 = 1 / 32 := 
by sorry

end geometric_sequence_a9_l207_207983


namespace person_B_completion_time_l207_207439

variables {A B : ℝ} (H : A + B = 1/6 ∧ (A + 10 * B = 1/6))

theorem person_B_completion_time :
    (1 / (1 - 2 * (A + B)) / B = 15) :=
by
  sorry

end person_B_completion_time_l207_207439


namespace fraction_product_simplification_l207_207125

theorem fraction_product_simplification : (3 / 4) * (4 / 5) * (5 / 6) * (6 / 7) * (7 / 8) = 3 / 8 :=
by
  sorry

end fraction_product_simplification_l207_207125


namespace rectangular_prism_volume_l207_207371

theorem rectangular_prism_volume (a b c V : ℝ) (h1 : a * b = 20) (h2 : b * c = 12) (h3 : a * c = 15) (hb : b = 5) : V = 75 :=
  sorry

end rectangular_prism_volume_l207_207371


namespace inheritance_amount_l207_207068

def federalTaxRate : ℝ := 0.25
def stateTaxRate : ℝ := 0.15
def totalTaxPaid : ℝ := 16500

theorem inheritance_amount :
  ∃ x : ℝ, (federalTaxRate * x + stateTaxRate * (1 - federalTaxRate) * x = totalTaxPaid) → x = 45500 := by
  sorry

end inheritance_amount_l207_207068


namespace total_cards_correct_l207_207416

-- Define the number of dozens each person has
def dozens_per_person : Nat := 9

-- Define the number of cards per dozen
def cards_per_dozen : Nat := 12

-- Define the number of people
def num_people : Nat := 4

-- Define the total number of Pokemon cards in all
def total_cards : Nat := dozens_per_person * cards_per_dozen * num_people

-- The statement to be proved
theorem total_cards_correct : total_cards = 432 := 
by 
  -- Proof omitted as requested
  sorry

end total_cards_correct_l207_207416


namespace fraction_comparison_l207_207325

noncomputable def one_seventh : ℚ := 1 / 7
noncomputable def decimal_0_point_14285714285 : ℚ := 14285714285 / 10^11
noncomputable def eps_1 : ℚ := 1 / (7 * 10^11)
noncomputable def eps_2 : ℚ := 1 / (7 * 10^12)

theorem fraction_comparison :
  one_seventh = decimal_0_point_14285714285 + eps_1 :=
sorry

end fraction_comparison_l207_207325


namespace range_of_a_l207_207557

variable {R : Type*} [LinearOrderedField R]

def setA (a : R) : Set R := {x | x^2 - 2*x + a ≤ 0}

def setB : Set R := {x | x^2 - 3*x + 2 ≤ 0}

theorem range_of_a (a : R) (h : setB ⊆ setA a) : a ≤ 0 := sorry

end range_of_a_l207_207557


namespace cream_ratio_l207_207255

noncomputable def John_creme_amount : ℚ := 3
noncomputable def Janet_initial_amount : ℚ := 8
noncomputable def Janet_creme_added : ℚ := 3
noncomputable def Janet_total_mixture : ℚ := Janet_initial_amount + Janet_creme_added
noncomputable def Janet_creme_ratio : ℚ := Janet_creme_added / Janet_total_mixture
noncomputable def Janet_drank_amount : ℚ := 3
noncomputable def Janet_drank_creme : ℚ := Janet_drank_amount * Janet_creme_ratio
noncomputable def Janet_creme_remaining : ℚ := Janet_creme_added - Janet_drank_creme

theorem cream_ratio :
  (John_creme_amount / Janet_creme_remaining) = (11 / 5) :=
by
  sorry

end cream_ratio_l207_207255


namespace jogger_ahead_distance_l207_207148

def jogger_speed_kmh : ℝ := 9
def train_speed_kmh : ℝ := 45
def train_length_m : ℝ := 120
def passing_time_s : ℝ := 31

theorem jogger_ahead_distance :
  let V_rel := (train_speed_kmh - jogger_speed_kmh) * (1000 / 3600)
  let Distance_train := V_rel * passing_time_s 
  Distance_train = 310 → 
  Distance_train = 190 + train_length_m :=
by
  intros
  sorry

end jogger_ahead_distance_l207_207148


namespace cos_double_angle_tan_sum_angles_l207_207605

variable (α β : ℝ)
variable (α_acute : 0 < α ∧ α < π / 2)
variable (β_acute : 0 < β ∧ β < π / 2)
variable (tan_alpha : Real.tan α = 4 / 3)
variable (sin_alpha_minus_beta : Real.sin (α - β) = - (Real.sqrt 5) / 5)

/- Prove that cos 2α = -7/25 given the conditions -/
theorem cos_double_angle :
  Real.cos (2 * α) = -7 / 25 :=
by
  sorry

/- Prove that tan (α + β) = -41/38 given the conditions -/
theorem tan_sum_angles :
  Real.tan (α + β) = -41 / 38 :=
by
  sorry

end cos_double_angle_tan_sum_angles_l207_207605


namespace ellipse_focal_distance_l207_207002

theorem ellipse_focal_distance : 
  ∀ (x y : ℝ), (x^2 / 36 + y^2 / 9 = 9) → (∃ c : ℝ, c = 2 * Real.sqrt 3) :=
by
  sorry

end ellipse_focal_distance_l207_207002


namespace max_area_triangle_bqc_l207_207157

noncomputable def triangle_problem : ℝ :=
  let a := 112.5
  let b := 56.25
  let c := 3
  a + b + c

theorem max_area_triangle_bqc : triangle_problem = 171.75 :=
by
  -- The proof would involve validating the steps to ensure the computations
  -- for the maximum area of triangle BQC match the expression 112.5 - 56.25 √3,
  -- and thus confirm that a = 112.5, b = 56.25, c = 3
  -- and verifying that a + b + c = 171.75.
  sorry

end max_area_triangle_bqc_l207_207157


namespace rectangle_area_l207_207765

theorem rectangle_area (x : ℝ) (h1 : x > 0) (h2 : x * 4 = 28) : x = 7 :=
sorry

end rectangle_area_l207_207765


namespace a_4_eq_28_l207_207452

def Sn (n : ℕ) : ℕ := 4 * n^2

def a_n (n : ℕ) : ℕ := Sn n - Sn (n - 1)

theorem a_4_eq_28 : a_n 4 = 28 :=
by
  sorry

end a_4_eq_28_l207_207452


namespace shaded_fraction_l207_207396

theorem shaded_fraction (rectangle_length rectangle_width : ℕ) (h_length : rectangle_length = 15) (h_width : rectangle_width = 20)
                        (total_area : ℕ := rectangle_length * rectangle_width)
                        (shaded_quarter : ℕ := total_area / 4)
                        (h_shaded_quarter : shaded_quarter = total_area / 5) :
  shaded_quarter / total_area = 1 / 5 :=
by
  sorry

end shaded_fraction_l207_207396


namespace problem_part1_problem_part2_l207_207278

open Real

theorem problem_part1 (α : ℝ) (h : (sin (π - α) * cos (2 * π - α)) / (tan (π - α) * sin (π / 2 + α) * cos (π / 2 - α)) = 1 / 2) :
  (cos α - 2 * sin α) / (3 * cos α + sin α) = 5 := sorry

theorem problem_part2 (α : ℝ) (h : tan α = -2) :
  1 - 2 * sin α * cos α + cos α ^ 2 = 2 / 5 := sorry

end problem_part1_problem_part2_l207_207278


namespace gcd_polynomial_multiple_528_l207_207885

-- Definition of the problem
theorem gcd_polynomial_multiple_528 (k : ℕ) : 
  gcd (3 * (528 * k) ^ 3 + (528 * k) ^ 2 + 4 * (528 * k) + 66) (528 * k) = 66 :=
by
  sorry

end gcd_polynomial_multiple_528_l207_207885


namespace find_correct_value_l207_207630

theorem find_correct_value (k : ℕ) (h1 : 173 * 240 = 41520) (h2 : 41520 / 48 = 865) : k * 48 = 173 * 240 → k = 865 :=
by
  intros h
  sorry

end find_correct_value_l207_207630


namespace certain_number_equals_l207_207569

theorem certain_number_equals (p q : ℚ) (h1 : 3 / p = 8) (h2 : 3 / q = 18) (h3 : p - q = 0.20833333333333334) : q = 1/6 := sorry

end certain_number_equals_l207_207569


namespace total_weekly_cost_correct_l207_207893

def daily_consumption (cups_per_day : ℕ) (ounces_per_cup : ℝ) : ℝ :=
  cups_per_day * ounces_per_cup

def weekly_consumption (cups_per_day : ℕ) (ounces_per_cup : ℝ) (days_per_week : ℕ) : ℝ :=
  daily_consumption cups_per_day ounces_per_cup * days_per_week

def weekly_cost (weekly_ounces : ℝ) (cost_per_ounce : ℝ) : ℝ :=
  weekly_ounces * cost_per_ounce

def person_A_weekly_cost : ℝ :=
  weekly_cost (weekly_consumption 3 0.4 7) 1.40

def person_B_weekly_cost : ℝ :=
  weekly_cost (weekly_consumption 1 0.6 7) 1.20

def person_C_weekly_cost : ℝ :=
  weekly_cost (weekly_consumption 2 0.5 5) 1.35

def james_weekly_cost : ℝ :=
  weekly_cost (weekly_consumption 2 0.5 7) 1.25

def total_weekly_cost : ℝ :=
  person_A_weekly_cost + person_B_weekly_cost + person_C_weekly_cost + james_weekly_cost

theorem total_weekly_cost_correct : total_weekly_cost = 32.30 := by
  unfold total_weekly_cost person_A_weekly_cost person_B_weekly_cost person_C_weekly_cost james_weekly_cost
  unfold weekly_cost weekly_consumption daily_consumption
  sorry

end total_weekly_cost_correct_l207_207893


namespace butter_mixture_price_l207_207619

theorem butter_mixture_price :
  let cost1 := 48 * 150
  let cost2 := 36 * 125
  let cost3 := 24 * 100
  let revenue1 := cost1 + cost1 * (20 / 100)
  let revenue2 := cost2 + cost2 * (30 / 100)
  let revenue3 := cost3 + cost3 * (50 / 100)
  let total_weight := 48 + 36 + 24
  (revenue1 + revenue2 + revenue3) / total_weight = 167.5 :=
by
  sorry

end butter_mixture_price_l207_207619


namespace area_region_eq_6_25_l207_207156

noncomputable def area_of_region : ℝ :=
  ∫ x in -0.5..4.5, (5 - |x - 2| - |x - 2|)

theorem area_region_eq_6_25 :
  area_of_region = 6.25 :=
sorry

end area_region_eq_6_25_l207_207156


namespace max_y_value_of_3x_plus_4_div_x_corresponds_value_of_x_l207_207830

noncomputable def y (x : ℝ) : ℝ := 3 * x + 4 / x
def max_value (x : ℝ) := y x ≤ -4 * Real.sqrt 3

theorem max_y_value_of_3x_plus_4_div_x (h : x < 0) : max_value x :=
sorry

theorem corresponds_value_of_x (x : ℝ) (h : x = -2 * Real.sqrt 3 / 3) : y x = -4 * Real.sqrt 3 :=
sorry

end max_y_value_of_3x_plus_4_div_x_corresponds_value_of_x_l207_207830


namespace A_empty_iff_a_gt_9_over_8_A_one_element_l207_207505

-- Definition of A based on a given condition
def A (a : ℝ) : Set ℝ := {x | a * x^2 - 3 * x + 2 = 0}

-- Problem 1: Prove that if A is empty, then a > 9/8
theorem A_empty_iff_a_gt_9_over_8 {a : ℝ} : 
  (A a = ∅) ↔ (a > 9 / 8) := 
sorry

-- Problem 2: Prove the elements in A when it contains only one element
theorem A_one_element {a : ℝ} : 
  (∃! x, x ∈ A a) ↔ (a = 0 ∧ (A a = {2 / 3})) ∨ (a = 9 / 8 ∧ (A a = {4 / 3})) := 
sorry

end A_empty_iff_a_gt_9_over_8_A_one_element_l207_207505


namespace pattern_expression_equality_l207_207349

theorem pattern_expression_equality (n : ℕ) : ((n - 1) * (n + 1)) + 1 = n^2 :=
  sorry

end pattern_expression_equality_l207_207349


namespace fence_perimeter_l207_207631

-- Definitions for the given conditions
def num_posts : ℕ := 36
def gap_between_posts : ℕ := 6

-- Defining the proof problem to show the perimeter equals 192 feet
theorem fence_perimeter (n : ℕ) (g : ℕ) (sq_field : ℕ → ℕ → Prop) : n = 36 ∧ g = 6 ∧ sq_field n g → 4 * ((n / 4 - 1) * g) = 192 :=
by
  intro h
  sorry

end fence_perimeter_l207_207631


namespace product_of_two_odd_numbers_not_always_composite_l207_207034

theorem product_of_two_odd_numbers_not_always_composite :
  ∃ (m n : ℕ), (¬ (2 ∣ m) ∧ ¬ (2 ∣ n)) ∧ (∀ d : ℕ, d ∣ (m * n) → d = 1 ∨ d = m * n) :=
by
  sorry

end product_of_two_odd_numbers_not_always_composite_l207_207034


namespace basketball_game_proof_l207_207013

-- Definition of the conditions
def num_teams (x : ℕ) : Prop := ∃ n : ℕ, n = x

def games_played (x : ℕ) (total_games : ℕ) : Prop := total_games = 28

def game_combinations (x : ℕ) : ℕ := (x * (x - 1)) / 2

-- Proof statement using the conditions
theorem basketball_game_proof (x : ℕ) (h1 : num_teams x) (h2 : games_played x 28) : 
  game_combinations x = 28 := by
  sorry

end basketball_game_proof_l207_207013


namespace mod_equiv_l207_207247

theorem mod_equiv :
  241 * 398 % 50 = 18 :=
by
  sorry

end mod_equiv_l207_207247


namespace problem_statement_l207_207697

noncomputable def verify_ratio (x y c d : ℝ) (h1 : 4 * x - 2 * y = c) (h2 : 6 * y - 12 * x = d) (h3 : d ≠ 0) : Prop :=
  c / d = -1/3

theorem problem_statement (x y c d : ℝ) (h1 : 4 * x - 2 * y = c) (h2 : 6 * y - 12 * x = d) (h3 : d ≠ 0) : verify_ratio x y c d h1 h2 h3 :=
  sorry

end problem_statement_l207_207697


namespace number_of_kiwis_l207_207989

/-
There are 500 pieces of fruit in a crate. One fourth of the fruits are apples,
20% are oranges, one fifth are strawberries, and the rest are kiwis.
Prove that the number of kiwis is 175.
-/

theorem number_of_kiwis (total_fruits apples oranges strawberries kiwis : ℕ)
  (h1 : total_fruits = 500)
  (h2 : apples = total_fruits / 4)
  (h3 : oranges = 20 * total_fruits / 100)
  (h4 : strawberries = total_fruits / 5)
  (h5 : kiwis = total_fruits - (apples + oranges + strawberries)) :
  kiwis = 175 :=
sorry

end number_of_kiwis_l207_207989


namespace discriminant_of_quadratic_l207_207107

-- Define the quadratic equation coefficients
def a : ℝ := 5
def b : ℝ := -11
def c : ℝ := 4

-- Prove the discriminant of the quadratic equation
theorem discriminant_of_quadratic :
    b^2 - 4 * a * c = 41 :=
by
  sorry

end discriminant_of_quadratic_l207_207107


namespace quadratic_square_binomial_l207_207308

theorem quadratic_square_binomial (k : ℝ) : 
  (∃ a : ℝ, (x : ℝ) → x^2 - 20 * x + k = (x + a)^2) ↔ k = 100 := 
by
  sorry

end quadratic_square_binomial_l207_207308


namespace real_roots_condition_l207_207039

-- Definitions based on conditions
def polynomial (x : ℝ) : ℝ := x^4 - 6 * x - 1
def is_root (a : ℝ) : Prop := polynomial a = 0

-- The statement we want to prove
theorem real_roots_condition (a b : ℝ) (ha: is_root a) (hb: is_root b) : 
  (a * b + 2 * a + 2 * b = 1.5 + Real.sqrt 3) := 
sorry

end real_roots_condition_l207_207039


namespace simplify_fraction_l207_207594

-- Define the numerator and denominator
def numerator := 5^4 + 5^2
def denominator := 5^3 - 5

-- Define the simplified fraction
def simplified_fraction := 65 / 12

-- The proof problem statement
theorem simplify_fraction : (numerator / denominator) = simplified_fraction := 
by 
   -- Proof will go here
   sorry

end simplify_fraction_l207_207594


namespace avg_difference_even_avg_difference_odd_l207_207887

noncomputable def avg (seq : List ℕ) : ℚ := (seq.sum : ℚ) / seq.length

def even_ints_20_to_60 := [20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60]
def even_ints_10_to_140 := [10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126, 128, 130, 132, 134, 136, 138, 140]

def odd_ints_21_to_59 := [21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59]
def odd_ints_11_to_139 := [11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63, 65, 67, 69, 71, 73, 75, 77, 79, 81, 83, 85, 87, 89, 91, 93, 95, 97, 99, 101, 103, 105, 107, 109, 111, 113, 115, 117, 119, 121, 123, 125, 127, 129, 131, 133, 135, 137, 139]

theorem avg_difference_even :
  avg even_ints_20_to_60 - avg even_ints_10_to_140 = -35 := sorry

theorem avg_difference_odd :
  avg odd_ints_21_to_59 - avg odd_ints_11_to_139 = -35 := sorry

end avg_difference_even_avg_difference_odd_l207_207887


namespace solve_for_x_l207_207740

-- Assume x is a positive integer
def pos_integer (x : ℕ) : Prop := 0 < x

-- Assume the equation holds for some x
def equation (x : ℕ) : Prop :=
  1^(x+2) + 2^(x+1) + 3^(x-1) + 4^x = 1170

-- Proposition stating that if x satisfies the equation then x must be 5
theorem solve_for_x (x : ℕ) (h1 : pos_integer x) (h2 : equation x) : x = 5 :=
by
  sorry

end solve_for_x_l207_207740


namespace mix_solutions_l207_207137

-- Definitions based on conditions
def solution_x_percentage : ℝ := 0.10
def solution_y_percentage : ℝ := 0.30
def volume_y : ℝ := 100
def desired_percentage : ℝ := 0.15

-- Problem statement rewrite with equivalent proof goal
theorem mix_solutions :
  ∃ Vx : ℝ, (Vx * solution_x_percentage + volume_y * solution_y_percentage) = (Vx + volume_y) * desired_percentage ∧ Vx = 300 :=
by
  sorry

end mix_solutions_l207_207137


namespace sqrt_four_l207_207236

theorem sqrt_four : {x : ℝ | x ^ 2 = 4} = {-2, 2} := by
  sorry

end sqrt_four_l207_207236


namespace constant_remainder_polynomial_division_l207_207342

theorem constant_remainder_polynomial_division (b : ℚ) :
  (∃ (r : ℚ), ∀ x : ℚ, r = (8 * x^3 - 9 * x^2 + b * x + 10) % (3 * x^2 - 2 * x + 5)) ↔ b = 118 / 9 :=
by
  sorry

end constant_remainder_polynomial_division_l207_207342


namespace balls_into_boxes_l207_207119

/-- There are 128 ways to distribute 7 distinguishable balls into 2 distinguishable boxes. -/
theorem balls_into_boxes : (2 : ℕ) ^ 7 = 128 := by
  sorry

end balls_into_boxes_l207_207119


namespace line_circle_no_intersection_l207_207375

theorem line_circle_no_intersection : 
  (∀ x y : ℝ, 3 * x + 4 * y = 12 → ¬ (x^2 + y^2 = 4)) :=
by
  sorry

end line_circle_no_intersection_l207_207375


namespace nautical_mile_to_land_mile_l207_207848

theorem nautical_mile_to_land_mile 
    (speed_one_sail : ℕ := 25) 
    (speed_two_sails : ℕ := 50) 
    (travel_time_one_sail : ℕ := 4) 
    (travel_time_two_sails : ℕ := 4)
    (total_distance : ℕ := 345) : 
    ∃ (x : ℚ), x = 1.15 ∧ 
    total_distance = travel_time_one_sail * speed_one_sail * x +
                    travel_time_two_sails * speed_two_sails * x := 
by
  sorry

end nautical_mile_to_land_mile_l207_207848


namespace Jenny_minutes_of_sleep_l207_207144

def hours_of_sleep : ℕ := 8
def minutes_per_hour : ℕ := 60

theorem Jenny_minutes_of_sleep : hours_of_sleep * minutes_per_hour = 480 := by
  sorry

end Jenny_minutes_of_sleep_l207_207144


namespace reciprocal_of_repeating_decimal_l207_207684

theorem reciprocal_of_repeating_decimal :
  let x := 0.36363636 -- simplified as .\overline{36}
  ∃ y : ℚ, x = 4 / 11 ∧ y = 1 / x ∧ y = 11 / 4 :=
by
  sorry

end reciprocal_of_repeating_decimal_l207_207684


namespace F_sum_l207_207303

noncomputable def f : ℝ → ℝ := sorry -- even function f(x)
noncomputable def F (x a c : ℝ) : ℝ := 
  let b := (a + c) / 2
  (x - b) * f (x - b) + 2016

theorem F_sum (a c : ℝ) : F a a c + F c a c = 4032 := 
by {
  sorry
}

end F_sum_l207_207303


namespace probability_of_draw_l207_207372

noncomputable def P_A_winning : ℝ := 0.4
noncomputable def P_A_not_losing : ℝ := 0.9

theorem probability_of_draw : P_A_not_losing - P_A_winning = 0.5 :=
by 
  sorry

end probability_of_draw_l207_207372


namespace cookie_sales_l207_207667

theorem cookie_sales (n M A : ℕ) 
  (hM : M = n - 9)
  (hA : A = n - 2)
  (h_sum : M + A < n)
  (hM_positive : M ≥ 1)
  (hA_positive : A ≥ 1) : 
  n = 10 := 
sorry

end cookie_sales_l207_207667


namespace rectangle_area_l207_207901

theorem rectangle_area (w l : ℕ) (h1 : l = 15) (h2 : (2 * l + 2 * w) / w = 5) : l * w = 150 :=
by
  -- We provide the conditions in the theorem's signature:
  -- l is the length which is 15 cm, given by h1
  -- The ratio of the perimeter to the width is 5:1, given by h2
  sorry

end rectangle_area_l207_207901


namespace geometric_progression_positions_l207_207443

theorem geometric_progression_positions (u1 q : ℝ) (m n p : ℕ)
  (h27 : 27 = u1 * q ^ (m - 1))
  (h8 : 8 = u1 * q ^ (n - 1))
  (h12 : 12 = u1 * q ^ (p - 1)) :
  m = 3 * p - 2 * n :=
sorry

end geometric_progression_positions_l207_207443


namespace max_stamps_l207_207758

def price_of_stamp : ℕ := 25  -- Price of one stamp in cents
def total_money : ℕ := 4000   -- Total money available in cents

theorem max_stamps : ∃ n : ℕ, price_of_stamp * n ≤ total_money ∧ (∀ m : ℕ, price_of_stamp * m ≤ total_money → m ≤ n) :=
by
  use 160
  sorry

end max_stamps_l207_207758


namespace intensity_on_Thursday_l207_207869

-- Step a) - Definitions from Conditions
def inversely_proportional (i b k : ℕ) : Prop := i * b = k

-- Translation of the proof problem
theorem intensity_on_Thursday (k b : ℕ) (h₁ : k = 24) (h₂ : b = 3) : ∃ i, inversely_proportional i b k ∧ i = 8 := 
by
  sorry

end intensity_on_Thursday_l207_207869


namespace total_selling_amount_l207_207063

-- Defining the given conditions
def total_metres_of_cloth := 200
def loss_per_metre := 6
def cost_price_per_metre := 66

-- Theorem statement to prove the total selling amount
theorem total_selling_amount : 
    (cost_price_per_metre - loss_per_metre) * total_metres_of_cloth = 12000 := 
by 
    sorry

end total_selling_amount_l207_207063


namespace zeros_of_geometric_sequence_quadratic_l207_207202

theorem zeros_of_geometric_sequence_quadratic (a b c : ℝ) (h_geometric : b^2 = a * c) (h_pos : a * c > 0) :
  ∀ x : ℝ, a * x^2 + b * x + c ≠ 0 := by
sorry

end zeros_of_geometric_sequence_quadratic_l207_207202


namespace competition_votes_l207_207388

/-- 
In a revival competition, if B's number of votes is 20/21 of A's, and B wins by
gaining at least 4 votes more than A, prove the possible valid votes counts.
-/
theorem competition_votes (x : ℕ) 
  (hx : x > 0) 
  (hx_mod_21 : x % 21 = 0) 
  (hB_wins : ∀ b : ℕ, b = (20 * x / 21) + 4 → b > x - 4) :
  (x = 147 ∧ 140 = 20 * x / 21) ∨ (x = 126 ∧ 120 = 20 * x / 21) := 
by 
  sorry

end competition_votes_l207_207388


namespace john_has_leftover_correct_l207_207642

-- Define the initial conditions
def initial_gallons : ℚ := 5
def given_away : ℚ := 18 / 7

-- Define the target result after subtraction
def remaining_gallons : ℚ := 17 / 7

-- The theorem statement
theorem john_has_leftover_correct :
  initial_gallons - given_away = remaining_gallons :=
by
  sorry

end john_has_leftover_correct_l207_207642


namespace smallest_angle_of_triangle_l207_207793

noncomputable def smallest_angle (a b : ℝ) (c : ℝ) (h_sum : a + b + c = 180) : ℝ :=
  min a (min b c)

theorem smallest_angle_of_triangle :
  smallest_angle 60 65 (180 - (60 + 65)) (by norm_num) = 55 :=
by
  -- The correct proof steps should be provided for the result
  sorry

end smallest_angle_of_triangle_l207_207793


namespace remaining_soup_feeds_adults_l207_207405

theorem remaining_soup_feeds_adults (C A k c : ℕ) 
    (hC : C= 10) 
    (hA : A = 5) 
    (hk : k = 8) 
    (hc : c = 20) : k - c / C * 10 * A = 30 := sorry

end remaining_soup_feeds_adults_l207_207405


namespace cost_per_pound_mixed_feed_correct_l207_207149

noncomputable def total_weight_of_feed : ℝ := 17
noncomputable def cost_per_pound_cheaper_feed : ℝ := 0.11
noncomputable def cost_per_pound_expensive_feed : ℝ := 0.50
noncomputable def weight_cheaper_feed : ℝ := 12.2051282051

noncomputable def total_cost_of_feed : ℝ :=
  (cost_per_pound_cheaper_feed * weight_cheaper_feed) + 
  (cost_per_pound_expensive_feed * (total_weight_of_feed - weight_cheaper_feed))

noncomputable def cost_per_pound_mixed_feed : ℝ :=
  total_cost_of_feed / total_weight_of_feed

theorem cost_per_pound_mixed_feed_correct : 
  cost_per_pound_mixed_feed = 0.22 :=
  by
    sorry

end cost_per_pound_mixed_feed_correct_l207_207149


namespace apps_minus_files_eq_seven_l207_207138

-- Definitions based on conditions
def initial_apps := 24
def initial_files := 9
def deleted_apps := initial_apps - 12
def deleted_files := initial_files - 5

-- Definitions based on the question and correct answer
def apps_left := 12
def files_left := 5

theorem apps_minus_files_eq_seven : apps_left - files_left = 7 := by
  sorry

end apps_minus_files_eq_seven_l207_207138


namespace points_per_round_l207_207914

-- Definitions based on conditions
def final_points (jane_points : ℕ) : Prop := jane_points = 60
def lost_points (jane_lost : ℕ) : Prop := jane_lost = 20
def rounds_played (jane_rounds : ℕ) : Prop := jane_rounds = 8

-- The theorem we want to prove
theorem points_per_round (jane_points jane_lost jane_rounds points_per_round : ℕ) 
  (h1 : final_points jane_points) 
  (h2 : lost_points jane_lost) 
  (h3 : rounds_played jane_rounds) : 
  points_per_round = ((jane_points + jane_lost) / jane_rounds) := 
sorry

end points_per_round_l207_207914


namespace least_number_of_cubes_is_10_l207_207223

noncomputable def volume_of_block (length width height : ℕ) : ℕ :=
  length * width * height

noncomputable def gcd_three_numbers (a b c : ℕ) : ℕ :=
  Nat.gcd a (Nat.gcd b c)

noncomputable def volume_of_cube (side : ℕ) : ℕ :=
  side ^ 3

noncomputable def least_number_of_cubes (length width height : ℕ) : ℕ := 
  volume_of_block length width height / volume_of_cube (gcd_three_numbers length width height)

theorem least_number_of_cubes_is_10 : least_number_of_cubes 15 30 75 = 10 := by
  sorry

end least_number_of_cubes_is_10_l207_207223


namespace next_performance_together_in_90_days_l207_207539

theorem next_performance_together_in_90_days :
  Nat.lcm (Nat.lcm 5 6) (Nat.lcm 9 10) = 90 := by
  sorry

end next_performance_together_in_90_days_l207_207539


namespace red_balls_in_box_l207_207271

theorem red_balls_in_box {n : ℕ} (h : n = 6) (p : (∃ (r : ℕ), r / 6 = 1 / 3)) : ∃ r, r = 2 :=
by
  sorry

end red_balls_in_box_l207_207271


namespace skate_cost_l207_207142

/- Define the initial conditions as Lean definitions -/
def admission_cost : ℕ := 5
def rental_cost : ℕ := 250 / 100  -- 2.50 dollars in cents for integer representation
def visits : ℕ := 26

/- Define the cost calculation as a Lean definition -/
def total_rental_cost (rental_cost : ℕ) (visits : ℕ) : ℕ := rental_cost * visits

/- Statement of the problem in Lean proof form -/
theorem skate_cost (C : ℕ) (h : total_rental_cost rental_cost visits = C) : C = 65 :=
by
  sorry

end skate_cost_l207_207142


namespace trigonometric_identity_l207_207567

open Real

theorem trigonometric_identity (θ : ℝ) (h₁ : 0 < θ ∧ θ < π/2) (h₂ : cos θ = sqrt 10 / 10) :
  (cos (2 * θ) / (sin (2 * θ) + (cos θ)^2)) = -8 / 7 := 
sorry

end trigonometric_identity_l207_207567


namespace thirty_thousand_times_thirty_thousand_l207_207578

-- Define the number thirty thousand
def thirty_thousand : ℕ := 30000

-- Define the product of thirty thousand times thirty thousand
def product_thirty_thousand : ℕ := thirty_thousand * thirty_thousand

-- State the theorem that this product equals nine hundred million
theorem thirty_thousand_times_thirty_thousand :
  product_thirty_thousand = 900000000 :=
sorry -- Proof goes here

end thirty_thousand_times_thirty_thousand_l207_207578


namespace solve_f_zero_k_eq_2_find_k_range_has_two_zeros_sum_of_reciprocals_l207_207305

-- Define the function f(x) based on the given conditions
def f (x k : ℝ) : ℝ := abs (x ^ 2 - 1) + x ^ 2 + k * x

-- Statement 1
theorem solve_f_zero_k_eq_2 :
  (∀ x : ℝ, f x 2 = 0 ↔ x = - (1 + Real.sqrt 3) / 2 ∨ x = -1 / 2) :=
sorry

-- Statement 2
theorem find_k_range_has_two_zeros (α β : ℝ) (hαβ : 0 < α ∧ α < β ∧ β < 2) :
  (∃ k : ℝ, f α k = 0 ∧ f β k = 0) ↔ - 7 / 2 < k ∧ k < -1 :=
sorry

-- Statement 3
theorem sum_of_reciprocals (α β : ℝ) (hαβ : 0 < α ∧ α < 1 ∧ 1 < β ∧ β < 2)
    (hα : f α (-1/α) = 0) (hβ : ∃ k : ℝ, f β k = 0) :
  (1 / α + 1 / β < 4) :=
sorry

end solve_f_zero_k_eq_2_find_k_range_has_two_zeros_sum_of_reciprocals_l207_207305


namespace minimum_value_of_m_l207_207651

-- Define a function to determine if a number is a perfect square
def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k * k = n

-- Define a function to determine if a number is a perfect cube
def is_perfect_cube (n : ℕ) : Prop := ∃ k : ℕ, k * k * k = n

-- Our main theorem statement
theorem minimum_value_of_m :
  ∃ m : ℕ, (600 < m ∧ m ≤ 800) ∧
           is_perfect_square (3 * m) ∧
           is_perfect_cube (5 * m) :=
sorry

end minimum_value_of_m_l207_207651


namespace woman_wait_time_l207_207335
noncomputable def time_for_man_to_catch_up (man_speed woman_speed distance: ℝ) : ℝ :=
  distance / man_speed

theorem woman_wait_time 
    (man_speed : ℝ)
    (woman_speed : ℝ)
    (wait_time_minutes : ℝ) 
    (woman_time : ℝ)
    (distance : ℝ)
    (man_time : ℝ) :
    man_speed = 5 -> 
    woman_speed = 15 -> 
    wait_time_minutes = 2 -> 
    woman_time = woman_speed * (1 / 60) * wait_time_minutes -> 
    woman_time = distance -> 
    man_speed * (1 / 60) = 0.0833 -> 
    man_time = distance / 0.0833 -> 
    man_time = 6 :=
by
  intros
  sorry

end woman_wait_time_l207_207335


namespace incorrect_statement_A_l207_207552

theorem incorrect_statement_A (p q : Prop) : (p ∨ q) → ¬ (p ∧ q) := by
  intros h
  cases h with
  | inl hp => sorry
  | inr hq => sorry

end incorrect_statement_A_l207_207552


namespace minimum_length_of_segment_PQ_l207_207653

theorem minimum_length_of_segment_PQ:
  (∀ x y : ℝ, (x^2 + y^2 - 2 * x + 4 * y + 1 = 0) → 
              (xy >= 2) → 
              (x - y >= 0) → 
              (y <= 1) → 
              ℝ) :=
sorry

end minimum_length_of_segment_PQ_l207_207653


namespace m_equals_p_of_odd_prime_and_integers_l207_207009

theorem m_equals_p_of_odd_prime_and_integers (p m : ℕ) (x y : ℕ) (hp : p > 1 ∧ ¬ (p % 2 = 0)) 
    (hx : x > 1) (hy : y > 1) 
    (h : (x ^ p + y ^ p) / 2 = ((x + y) / 2) ^ m): 
    m = p := 
by 
  sorry

end m_equals_p_of_odd_prime_and_integers_l207_207009


namespace rachel_problems_solved_each_minute_l207_207596

-- Definitions and conditions
def problems_solved_each_minute (x : ℕ) : Prop :=
  let problems_before_bed := 12 * x
  let problems_at_lunch := 16
  let total_problems := problems_before_bed + problems_at_lunch
  total_problems = 76

-- Theorem to be proved
theorem rachel_problems_solved_each_minute : ∃ x : ℕ, problems_solved_each_minute x ∧ x = 5 :=
by
  sorry

end rachel_problems_solved_each_minute_l207_207596


namespace milk_butterfat_problem_l207_207712

-- Define the values given in the problem
def b1 : ℝ := 0.35  -- butterfat percentage of initial milk
def v1 : ℝ := 8     -- volume of initial milk in gallons
def b2 : ℝ := 0.10  -- butterfat percentage of milk to be added
def bf : ℝ := 0.20  -- desired butterfat percentage of the final mixture

-- Define the proof statement
theorem milk_butterfat_problem :
  ∃ x : ℝ, (2.8 + 0.1 * x) / (v1 + x) = bf ↔ x = 12 :=
by {
  sorry
}

end milk_butterfat_problem_l207_207712


namespace calculate_a2_b2_c2_l207_207135

theorem calculate_a2_b2_c2 (a b c : ℝ) (h1 : a + b + c = 0) (h2 : a * b + a * c + b * c = -3) (h3 : a * b * c = 2) :
  a^2 + b^2 + c^2 = 6 :=
sorry

end calculate_a2_b2_c2_l207_207135


namespace sequence_formula_l207_207618

theorem sequence_formula (a : ℕ → ℕ) (h1 : a 1 = 1) (h_rec : ∀ n, a (n + 1) = 2 * a n + 1) :
  ∀ n, a n = 2^n - 1 :=
by
  sorry

end sequence_formula_l207_207618


namespace max_value_of_A_l207_207980

theorem max_value_of_A (a b c : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c) :
  (a^3 * (b + c) + b^3 * (c + a) + c^3 * (a + b)) / ((a + b + c)^4 - 79 * (a * b * c)^(4 / 3)) ≤ 3 :=
sorry

end max_value_of_A_l207_207980


namespace simplify_sqrt_expression_l207_207584

theorem simplify_sqrt_expression (t : ℝ) : (Real.sqrt (t^6 + t^4) = t^2 * Real.sqrt (t^2 + 1)) :=
by sorry

end simplify_sqrt_expression_l207_207584


namespace cos_150_eq_neg_sqrt3_div_2_l207_207155

theorem cos_150_eq_neg_sqrt3_div_2 : Real.cos (150 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  unfold Real.cos
  sorry

end cos_150_eq_neg_sqrt3_div_2_l207_207155


namespace interval_of_monotonic_increase_sum_greater_than_2e_l207_207799

noncomputable def f (a x : ℝ) : ℝ := a * x / (Real.log x)

theorem interval_of_monotonic_increase :
  ∀ (x : ℝ), (e < x → f 1 x > f 1 e) := 
sorry

theorem sum_greater_than_2e (x1 x2 : ℝ) (a : ℝ) (h1 : x1 ≠ x2) (hx1 : f 1 x1 = 1) (hx2 : f 1 x2 = 1) :
  x1 + x2 > 2 * Real.exp 1 :=
sorry

end interval_of_monotonic_increase_sum_greater_than_2e_l207_207799


namespace triangle_formation_segments_l207_207230

theorem triangle_formation_segments (a b c : ℝ) (h_sum : a + b + c = 1) (h_a : a < 1/2) (h_b : b < 1/2) (h_c : c < 1/2) : 
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a) := 
by
  sorry

end triangle_formation_segments_l207_207230


namespace journey_time_l207_207022

theorem journey_time 
  (d1 d2 T : ℝ)
  (h1 : d1 / 30 + (150 - d1) / 10 = T)
  (h2 : d1 / 30 + d2 / 30 + (150 - (d1 - d2)) / 30 = T)
  (h3 : (d1 - d2) / 10 + (150 - (d1 - d2)) / 30 = T) :
  T = 5 := 
sorry

end journey_time_l207_207022


namespace max_value_is_two_over_three_l207_207608

noncomputable def max_value_expr (x : ℝ) : ℝ := 2^x - 8^x

theorem max_value_is_two_over_three :
  ∃ (x : ℝ), max_value_expr x = 2 / 3 :=
sorry

end max_value_is_two_over_three_l207_207608


namespace tanya_bought_six_plums_l207_207323

theorem tanya_bought_six_plums (pears apples pineapples pieces_left : ℕ) 
  (h_pears : pears = 6) (h_apples : apples = 4) (h_pineapples : pineapples = 2) 
  (h_pieces_left : pieces_left = 9) (h_half_fell : pieces_left * 2 = total_fruit) :
  pears + apples + pineapples < total_fruit ∧ total_fruit - (pears + apples + pineapples) = 6 :=
by
  sorry

end tanya_bought_six_plums_l207_207323


namespace negation_of_diagonals_equal_l207_207132

def Rectangle : Type := sorry -- Let's assume there exists a type Rectangle
def diagonals_equal (r : Rectangle) : Prop := sorry -- Assume a function that checks if diagonals are equal

theorem negation_of_diagonals_equal :
  ¬(∀ r : Rectangle, diagonals_equal r) ↔ ∃ r : Rectangle, ¬diagonals_equal r :=
by
  sorry

end negation_of_diagonals_equal_l207_207132


namespace transform_circle_to_ellipse_l207_207583

theorem transform_circle_to_ellipse (x y x'' y'' : ℝ) (h_circle : x^2 + y^2 = 1)
  (hx_trans : x = x'' / 2) (hy_trans : y = y'' / 3) :
  (x''^2 / 4) + (y''^2 / 9) = 1 :=
by {
  sorry
}

end transform_circle_to_ellipse_l207_207583


namespace delta_solution_l207_207650

theorem delta_solution : ∃ Δ : ℤ, 4 * (-3) = Δ - 1 ∧ Δ = -11 :=
by
  -- Using the condition 4(-3) = Δ - 1, 
  -- we need to prove that Δ = -11
  sorry

end delta_solution_l207_207650


namespace total_weight_marble_purchased_l207_207389

theorem total_weight_marble_purchased (w1 w2 w3 : ℝ) (h1 : w1 = 0.33) (h2 : w2 = 0.33) (h3 : w3 = 0.08) :
  w1 + w2 + w3 = 0.74 := by
  sorry

end total_weight_marble_purchased_l207_207389


namespace find_y_intercept_l207_207559

def line_y_intercept (m x y : ℝ) (pt : ℝ × ℝ) : ℝ :=
  let y_intercept := pt.snd - m * pt.fst
  y_intercept

theorem find_y_intercept (m x y b : ℝ) (pt : ℝ × ℝ) (h1 : m = 2) (h2 : pt = (498, 998)) :
  line_y_intercept m x y pt = 2 :=
by
  sorry

end find_y_intercept_l207_207559


namespace prism_faces_even_or_odd_l207_207163

theorem prism_faces_even_or_odd (n : ℕ) (hn : 3 ≤ n) : ¬ (2 + n) % 2 = 1 :=
by
  sorry

end prism_faces_even_or_odd_l207_207163


namespace art_piece_increase_l207_207705

theorem art_piece_increase (initial_price : ℝ) (multiplier : ℝ) (future_increase : ℝ) (h1 : initial_price = 4000) (h2 : multiplier = 3) :
  future_increase = (multiplier * initial_price) - initial_price :=
by
  rw [h1, h2]
  norm_num
  sorry

end art_piece_increase_l207_207705


namespace complement_intersection_complement_l207_207797

-- Define the sets U, A, and B
def U : Set ℕ := {1, 2, 3, 4, 6}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 3, 4}

-- Define the statement of the proof problem
theorem complement_intersection_complement:
  (U \ (A ∩ B)) = {1, 4, 6} := by
  sorry

end complement_intersection_complement_l207_207797


namespace sum_of_squares_is_perfect_square_l207_207641

theorem sum_of_squares_is_perfect_square (n p k : ℤ) : 
  (∃ m : ℤ, n^2 + p^2 + k^2 = m^2) ↔ (n * k = (p / 2)^2) :=
by
  sorry

end sum_of_squares_is_perfect_square_l207_207641


namespace question_solution_l207_207480

theorem question_solution 
  (hA : -(-1) = abs (-1))
  (hB : ¬ (∃ n : ℤ, ∀ m : ℤ, n < m ∧ m < 0))
  (hC : (-2)^3 = -2^3)
  (hD : ∃ q : ℚ, q = 0) :
  ¬ (∀ q : ℚ, q > 0 ∨ q < 0) := 
by {
  sorry
}

end question_solution_l207_207480


namespace no_solution_ineq_range_a_l207_207282

theorem no_solution_ineq_range_a (a : ℝ) :
  (∀ x : ℝ, x^2 + a * x + 4 < 0 → false) ↔ (-4 ≤ a ∧ a ≤ 4) :=
by
  sorry

end no_solution_ineq_range_a_l207_207282


namespace valid_parametrizations_l207_207287

-- Definitions for the given points and directions
def pointA := (0, 4)
def dirA := (3, -1)

def pointB := (4/3, 0)
def dirB := (1, -3)

def pointC := (-2, 10)
def dirC := (-3, 9)

-- Line equation definition
def line (x y : ℝ) : Prop := y = -3 * x + 4

-- Proof statement
theorem valid_parametrizations :
  (line pointB.1 pointB.2 ∧ dirB.2 = -3 * dirB.1) ∧
  (line pointC.1 pointC.2 ∧ dirC.2 / dirC.1 = 3) :=
by
  sorry

end valid_parametrizations_l207_207287


namespace polynomial_expansion_l207_207011

variable (x : ℝ)

theorem polynomial_expansion :
  (7*x^2 + 3)*(5*x^3 + 4*x + 1) = 35*x^5 + 43*x^3 + 7*x^2 + 12*x + 3 := by
  sorry

end polynomial_expansion_l207_207011


namespace range_of_x_when_m_is_4_range_of_m_l207_207749

-- Define the conditions for p and q
def p (x : ℝ) : Prop := x^2 - 7 * x + 10 < 0
def q (x m : ℝ) : Prop := x^2 - 4 * m * x + 3 * m^2 < 0
def neg_p (x : ℝ) : Prop := x ≤ 2 ∨ x ≥ 5
def neg_q (x m : ℝ) : Prop := x ≤ m ∨ x ≥ 3 * m

-- Define the conditions for the values of m
def cond_m_pos (m : ℝ) : Prop := m > 0
def cond_sufficient (m : ℝ) : Prop := cond_m_pos m ∧ m ≤ 2 ∧ 3 * m ≥ 5

-- Problem 1
theorem range_of_x_when_m_is_4 (x : ℝ) : p x ∧ q x 4 → 4 < x ∧ x < 5 :=
sorry

-- Problem 2
theorem range_of_m (m : ℝ) : (∀ x : ℝ, neg_q x m → neg_p x) → 5 / 3 ≤ m ∧ m ≤ 2 :=
sorry

end range_of_x_when_m_is_4_range_of_m_l207_207749


namespace domain_of_p_l207_207674

def is_domain_of_p (x : ℝ) : Prop := x > 5

theorem domain_of_p :
  {x : ℝ | ∃ y : ℝ, y = 5*x + 2 ∧ ∃ z : ℝ, z = 2*x - 10 ∧
    z ≥ 0 ∧ z ≠ 0 ∧ p = 5*x + 2} = {x : ℝ | x > 5} :=
by
  sorry

end domain_of_p_l207_207674


namespace solve_quadratic_l207_207327

theorem solve_quadratic (x : ℝ) : (x^2 + x)^2 + (x^2 + x) - 6 = 0 ↔ x = -2 ∨ x = 1 :=
by
  sorry

end solve_quadratic_l207_207327


namespace rectangle_area_error_83_percent_l207_207764

theorem rectangle_area_error_83_percent (L W : ℝ) :
  let actual_area := L * W
  let measured_length := 1.14 * L
  let measured_width := 0.95 * W
  let calculated_area := measured_length * measured_width
  let error := calculated_area - actual_area
  (error / actual_area) * 100 = 8.3 := by
  sorry

end rectangle_area_error_83_percent_l207_207764


namespace deduction_from_third_l207_207249

-- Define the conditions
def avg_10_consecutive_eq_20 (x : ℝ) : Prop :=
  (x + (x + 1) + (x + 2) + (x + 3) + (x + 4) + (x + 5) + (x + 6) + (x + 7) + (x + 8) + (x + 9)) / 10 = 20

def new_avg_10_numbers_eq_15_5 (x y : ℝ) : Prop :=
  ((x - 9) + (x - 7) + (x + 2 - y) + (x - 3) + (x - 1) + (x + 1) + (x + 3) + (x + 5) + (x + 7) + (x + 9)) / 10 = 15.5

-- Define the theorem to be proved
theorem deduction_from_third (x y : ℝ) (h1 : avg_10_consecutive_eq_20 x) (h2 : new_avg_10_numbers_eq_15_5 x y) : y = 6 :=
sorry

end deduction_from_third_l207_207249


namespace combo_discount_is_50_percent_l207_207366

noncomputable def combo_discount_percentage
  (ticket_cost : ℕ) (combo_cost : ℕ) (ticket_discount : ℕ) (total_savings : ℕ) : ℕ :=
  let ticket_savings := ticket_cost * ticket_discount / 100
  let combo_savings := total_savings - ticket_savings
  (combo_savings * 100) / combo_cost

theorem combo_discount_is_50_percent:
  combo_discount_percentage 10 10 20 7 = 50 :=
by
  sorry

end combo_discount_is_50_percent_l207_207366


namespace cheenu_time_difference_l207_207561

-- Define the conditions in terms of Cheenu's activities

variable (boy_run_distance : ℕ) (boy_run_time : ℕ)
variable (midage_bike_distance : ℕ) (midage_bike_time : ℕ)
variable (old_walk_distance : ℕ) (old_walk_time : ℕ)

-- Define the problem with these variables
theorem cheenu_time_difference:
    boy_run_distance = 20 ∧ boy_run_time = 240 ∧
    midage_bike_distance = 30 ∧ midage_bike_time = 120 ∧
    old_walk_distance = 8 ∧ old_walk_time = 240 →
    (old_walk_time / old_walk_distance - midage_bike_time / midage_bike_distance) = 26 := by
    sorry

end cheenu_time_difference_l207_207561


namespace no_three_digit_number_l207_207231

theorem no_three_digit_number (N : ℕ) : 
  (100 ≤ N ∧ N < 1000 ∧ 
   (∀ k, k ∈ [1,2,3] → 5 < (N / 10^(k - 1) % 10)) ∧ 
   (N % 6 = 0) ∧ (N % 5 = 0)) → 
  false :=
by
sorry

end no_three_digit_number_l207_207231


namespace base10_to_base7_conversion_l207_207471

theorem base10_to_base7_conversion : 2023 = 5 * 7^3 + 6 * 7^2 + 2 * 7^1 + 0 * 7^0 :=
  sorry

end base10_to_base7_conversion_l207_207471


namespace largest_pillar_radius_l207_207115

-- Define the dimensions of the crate
def crate_length := 12
def crate_width := 8
def crate_height := 3

-- Define the condition that the pillar is a right circular cylinder
def is_right_circular_cylinder (r : ℝ) (h : ℝ) : Prop :=
  r > 0 ∧ h > 0

-- The theorem stating the radius of the largest volume pillar that can fit in the crate
theorem largest_pillar_radius (r h : ℝ) (cylinder_fits : is_right_circular_cylinder r h) :
  r = 1.5 := 
sorry

end largest_pillar_radius_l207_207115


namespace engineer_days_l207_207510

theorem engineer_days (x : ℕ) (k : ℕ) (d : ℕ) (n : ℕ) (m : ℕ) (e : ℕ)
  (h1 : k = 10) -- Length of the road in km
  (h2 : d = 15) -- Total days to complete the project
  (h3 : n = 30) -- Initial number of men
  (h4 : m = 2) -- Length of the road completed in x days
  (h5 : e = n + 30) -- New number of men
  (h6 : (4 : ℚ) / x = (8 : ℚ) / (d - x)) : x = 5 :=
by
  -- The proof would go here.
  sorry

end engineer_days_l207_207510


namespace nina_weekend_earnings_l207_207095

noncomputable def total_money_made (necklace_price bracelet_price earring_pair_price ensemble_price : ℕ)
                                   (necklaces_sold bracelets_sold individual_earrings_sold ensembles_sold : ℕ) : ℕ :=
  necklace_price * necklaces_sold +
  bracelet_price * bracelets_sold +
  earring_pair_price * (individual_earrings_sold / 2) +
  ensemble_price * ensembles_sold

theorem nina_weekend_earnings :
  total_money_made 25 15 10 45 5 10 20 2 = 465 :=
by
  sorry

end nina_weekend_earnings_l207_207095


namespace savings_correct_l207_207229

def initial_savings : ℕ := 1147240
def total_income : ℕ := (55000 + 45000 + 10000 + 17400) * 4
def total_expenses : ℕ := (40000 + 20000 + 5000 + 2000 + 2000) * 4
def final_savings : ℕ := initial_savings + total_income - total_expenses

theorem savings_correct : final_savings = 1340840 :=
by
  sorry

end savings_correct_l207_207229


namespace polynomial_no_negative_roots_l207_207866

theorem polynomial_no_negative_roots (x : ℝ) (h : x < 0) : x^4 - 4 * x^3 - 6 * x^2 - 3 * x + 9 ≠ 0 := 
by 
  sorry

end polynomial_no_negative_roots_l207_207866


namespace cone_base_radius_l207_207781

-- Definitions based on conditions
def sphere_radius : ℝ := 1
def cone_height : ℝ := 2

-- Problem statement
theorem cone_base_radius {r : ℝ} 
  (h1 : ∀ x y z : ℝ, (x = sphere_radius ∧ y = sphere_radius ∧ z = sphere_radius) → 
                     (x + y + z = 3 * sphere_radius)) 
  (h2 : ∃ (O O1 O2 O3 : ℝ), (O = 0) ∧ (O1 = 1) ∧ (O2 = 1) ∧ (O3 = 1)) 
  (h3 : ∀ x y z : ℝ, (x + y + z = 3 * sphere_radius) → 
                     (y = z) → (x = z) → y * z + x * z + x * y = 3 * sphere_radius ^ 2)
  (h4 : ∀ h : ℝ, h = cone_height) :
  r = (Real.sqrt 3 / 6) :=
sorry

end cone_base_radius_l207_207781


namespace correct_op_l207_207603

-- Declare variables and conditions
variables {a b : ℝ} {m n : ℤ}
variable (ha : a > 0)
variable (hb : b ≠ 0)

-- Define and state the theorem
theorem correct_op (ha : a > 0) (hb : b ≠ 0) : (b / a)^m = a^(-m) * b^m :=
sorry  -- Proof omitted

end correct_op_l207_207603


namespace hcf_of_two_numbers_l207_207206

-- Definitions based on conditions
def LCM (x y : ℕ) : ℕ := sorry  -- Assume some definition of LCM
def HCF (x y : ℕ) : ℕ := sorry  -- Assume some definition of HCF

-- Given conditions
axiom cond1 (x y : ℕ) : LCM x y = 600
axiom cond2 (x y : ℕ) : x * y = 18000

-- Statement to prove
theorem hcf_of_two_numbers (x y : ℕ) (h1 : LCM x y = 600) (h2 : x * y = 18000) : HCF x y = 30 :=
by {
  -- Proof omitted, hence we use sorry
  sorry
}

end hcf_of_two_numbers_l207_207206


namespace average_rate_of_reduction_l207_207264

theorem average_rate_of_reduction
  (original_price final_price : ℝ)
  (h1 : original_price = 200)
  (h2 : final_price = 128)
  : ∃ (x : ℝ), 0 ≤ x ∧ x < 1 ∧ 200 * (1 - x) * (1 - x) = 128 :=
by
  sorry

end average_rate_of_reduction_l207_207264


namespace certain_number_l207_207915

theorem certain_number (x certain_number : ℕ) (h1 : x = 3327) (h2 : 9873 + x = certain_number) : 
  certain_number = 13200 := 
by
  sorry

end certain_number_l207_207915


namespace find_sum_of_xy_l207_207175

theorem find_sum_of_xy (x y : ℝ) (hx_ne_y : x ≠ y) (hx_nonzero : x ≠ 0) (hy_nonzero : y ≠ 0)
  (h_equation : x^4 - 2018 * x^3 - 2018 * y^2 * x = y^4 - 2018 * y^3 - 2018 * y * x^2) :
  x + y = 2018 :=
sorry

end find_sum_of_xy_l207_207175


namespace coin_difference_l207_207645

theorem coin_difference : 
  ∀ (c : ℕ), c = 50 → 
  (∃ (n m : ℕ), 
    (n ≥ m) ∧ 
    (∃ (a b d e : ℕ), n = a + b + d + e ∧ 5 * a + 10 * b + 20 * d + 25 * e = c) ∧
    (∃ (p q r s : ℕ), m = p + q + r + s ∧ 5 * p + 10 * q + 20 * r + 25 * s = c) ∧ 
    (n - m = 8)) :=
by
  sorry

end coin_difference_l207_207645


namespace perimeter_of_isosceles_triangle_l207_207351

theorem perimeter_of_isosceles_triangle (a b : ℕ) (h_isosceles : (a = 3 ∧ b = 4) ∨ (a = 4 ∧ b = 3)) :
  ∃ p : ℕ, p = 10 ∨ p = 11 :=
by
  sorry

end perimeter_of_isosceles_triangle_l207_207351


namespace find_integer_closest_expression_l207_207868

theorem find_integer_closest_expression :
  let a := (7 + Real.sqrt 48) ^ 2023
  let b := (7 - Real.sqrt 48) ^ 2023
  ((a + b) ^ 2 - (a - b) ^ 2) = 4 :=
by
  sorry

end find_integer_closest_expression_l207_207868


namespace unique_intersection_value_l207_207853

theorem unique_intersection_value :
  (∀ (x y : ℝ), y = x^2 → y = 4 * x + k) → (k = -4) := 
by
  sorry

end unique_intersection_value_l207_207853


namespace percent_not_red_balls_l207_207860

theorem percent_not_red_balls (percent_cubes percent_red_balls : ℝ) 
  (h1 : percent_cubes = 0.3) (h2 : percent_red_balls = 0.25) : 
  (1 - percent_red_balls) * (1 - percent_cubes) = 0.525 :=
by
  sorry

end percent_not_red_balls_l207_207860


namespace Tn_lt_half_Sn_l207_207298

noncomputable def a_n (n : ℕ) : ℝ := (1/3)^(n-1)
noncomputable def b_n (n : ℕ) : ℝ := n * (1/3)^n
noncomputable def S_n (n : ℕ) : ℝ := 3/2 - 1/2 * (1/3)^(n-1)
noncomputable def T_n (n : ℕ) : ℝ := 3/4 - 1/4 * (1/3)^(n-1) - n/2 * (1/3)^n

theorem Tn_lt_half_Sn (n : ℕ) : T_n n < S_n n / 2 :=
by
  sorry

end Tn_lt_half_Sn_l207_207298


namespace pavan_total_distance_l207_207490

theorem pavan_total_distance:
  ∀ (D : ℝ),
  (∃ Time1 Time2,
    Time1 = (D / 2) / 30 ∧
    Time2 = (D / 2) / 25 ∧
    Time1 + Time2 = 11)
  → D = 150 :=
by
  intros D h
  sorry

end pavan_total_distance_l207_207490


namespace average_speed_l207_207937

-- Define the conditions as constants and theorems
def distance1 : ℝ := 240
def distance2 : ℝ := 420
def time_diff : ℝ := 3

theorem average_speed : ∃ v t : ℝ, distance1 = v * t ∧ distance2 = v * (t + time_diff) → v = 60 := 
by
  sorry

end average_speed_l207_207937


namespace system_no_five_distinct_solutions_system_four_distinct_solutions_l207_207353

theorem system_no_five_distinct_solutions (a : ℤ) :
  ¬ ∃ x₁ x₂ x₃ x₄ x₅ y₁ y₂ y₃ y₄ y₅ z₁ z₂ z₃ z₄ z₅ : ℤ,
    (x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₁ ≠ x₅ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₂ ≠ x₅ ∧ x₃ ≠ x₄ ∧ x₃ ≠ x₅ ∧ x₄ ≠ x₅) ∧
    (y₁ ≠ y₂ ∧ y₁ ≠ y₃ ∧ y₁ ≠ y₄ ∧ y₁ ≠ y₅ ∧ y₂ ≠ y₃ ∧ y₂ ≠ y₄ ∧ y₂ ≠ y₅ ∧ y₃ ≠ y₄ ∧ y₃ ≠ y₅ ∧ y₄ ≠ y₅) ∧
    (z₁ ≠ z₂ ∧ z₁ ≠ z₃ ∧ z₁ ≠ z₄ ∧ z₁ ≠ z₅ ∧ z₂ ≠ z₃ ∧ z₂ ≠ z₄ ∧ z₂ ≠ z₅ ∧ z₃ ≠ z₄ ∧ z₃ ≠ z₅ ∧ z₄ ≠ z₅) ∧
    (2 * y₁ * z₁ + x₁ - y₁ - z₁ = a) ∧ (2 * x₁ * z₁ - x₁ + y₁ - z₁ = a) ∧ (2 * x₁ * y₁ - x₁ - y₁ + z₁ = a) ∧
    (2 * y₂ * z₂ + x₂ - y₂ - z₂ = a) ∧ (2 * x₂ * z₂ - x₂ + y₂ - z₂ = a) ∧ (2 * x₂ * y₂ - x₂ - y₂ + z₂ = a) ∧
    (2 * y₃ * z₃ + x₃ - y₃ - z₃ = a) ∧ (2 * x₃ * z₃ - x₃ + y₃ - z₃ = a) ∧ (2 * x₃ * y₃ - x₃ - y₃ + z₃ = a) ∧
    (2 * y₄ * z₄ + x₄ - y₄ - z₄ = a) ∧ (2 * x₄ * z₄ - x₄ + y₄ - z₄ = a) ∧ (2 * x₄ * y₄ - x₄ - y₄ + z₄ = a) ∧
    (2 * y₅ * z₅ + x₅ - y₅ - z₅ = a) ∧ (2 * x₅ * z₅ - x₅ + y₅ - z₅ = a) ∧ (2 * x₅ * y₅ - x₅ - y₅ + z₅ = a) :=
sorry

theorem system_four_distinct_solutions (a : ℤ) :
  (∃ x₁ x₂ y₁ y₂ z₁ z₂ : ℤ,
    x₁ ≠ x₂ ∧ y₁ ≠ y₂ ∧ z₁ ≠ z₂ ∧
    (2 * y₁ * z₁ + x₁ - y₁ - z₁ = a) ∧ (2 * x₁ * z₁ - x₁ + y₁ - z₁ = a) ∧ (2 * x₁ * y₁ - x₁ - y₁ + z₁ = a) ∧
    (2 * y₂ * z₂ + x₂ - y₂ - z₂ = a) ∧ (2 * x₂ * z₂ - x₂ + y₂ - z₂ = a) ∧ (2 * x₂ * y₂ - x₂ - y₂ + z₂ = a)) ↔
  ∃ k : ℤ, k % 2 = 1 ∧ a = (k^2 - 1) / 8 :=
sorry

end system_no_five_distinct_solutions_system_four_distinct_solutions_l207_207353


namespace valid_triangle_inequality_l207_207916

theorem valid_triangle_inequality (a : ℝ) 
  (h1 : 4 + 6 > a) 
  (h2 : 4 + a > 6) 
  (h3 : 6 + a > 4) : 
  a = 5 :=
sorry

end valid_triangle_inequality_l207_207916


namespace part1_part2_l207_207263

open Set

def A : Set ℝ := {x | x^2 - 8 * x + 15 = 0}
def B (a : ℝ) : Set ℝ := {x | a * x = 1}

theorem part1 : B (1/5) ⊆ A ∧ ¬ A ⊆ B (1/5) := by
  sorry
  
theorem part2 (a : ℝ) : (B a ⊆ A) ↔ a ∈ ({0, 1/3, 1/5} : Set ℝ) := by
  sorry

end part1_part2_l207_207263


namespace solve_for_y_l207_207422

def star (x y : ℝ) : ℝ := 5 * x - 2 * y + 3 * x * y

theorem solve_for_y (y : ℝ) : star 2 y = 10 → y = 0 := by
  intro h
  sorry

end solve_for_y_l207_207422


namespace sequence_inequality_l207_207939

variable (a : ℕ → ℝ) (b : ℕ → ℝ)
variable (q : ℝ)
variable (n : ℕ)

noncomputable def is_geometric (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

def is_arithmetic (b : ℕ → ℝ) : Prop :=
  ∀ n, b (n + 1) - b n = b 1 - b 0

theorem sequence_inequality
  (ha : ∀ n, 0 < a n)
  (hg : is_geometric a q)
  (ha6_eq_b7 : a 6 = b 7)
  (hb : is_arithmetic b) :
  a 3 + a 9 ≥ b 4 + b 10 :=
by
  sorry

end sequence_inequality_l207_207939


namespace additional_seasons_is_one_l207_207056

-- Definitions for conditions
def episodes_per_season : Nat := 22
def episodes_last_season : Nat := episodes_per_season + 4
def episodes_in_9_seasons : Nat := 9 * episodes_per_season
def hours_per_episode : Nat := 1 / 2 -- Stored as half units

-- Given conditions
def total_hours_to_watch_after_last_season: Nat := 112 * 2 -- converted to half-hours
def time_watched_in_9_seasons: Nat := episodes_in_9_seasons * hours_per_episode
def additional_hours: Nat := total_hours_to_watch_after_last_season - time_watched_in_9_seasons

-- Theorem to prove
theorem additional_seasons_is_one : additional_hours / hours_per_episode = episodes_last_season -> 
      additional_hours / hours_per_episode / episodes_per_season = 1 :=
by
  sorry

end additional_seasons_is_one_l207_207056


namespace coefficient_of_expansion_l207_207256

theorem coefficient_of_expansion (m : ℝ) (h : m^3 * (Nat.choose 6 3) = -160) : m = -2 := by
  sorry

end coefficient_of_expansion_l207_207256


namespace unique_solution_l207_207364

theorem unique_solution (x y z t : ℕ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (ht : t > 0) :
  12^x + 13^y - 14^z = 2013^t → (x = 1 ∧ y = 3 ∧ z = 2 ∧ t = 1) :=
by
  intros h
  sorry

end unique_solution_l207_207364


namespace find_pastries_made_l207_207031

variable (cakes_made pastries_made total_cakes_sold total_pastries_sold extra_cakes more_cakes_than_pastries : ℕ)

def baker_conditions := (cakes_made = 157) ∧ 
                        (total_cakes_sold = 158) ∧ 
                        (total_pastries_sold = 147) ∧ 
                        (more_cakes_than_pastries = 11) ∧ 
                        (extra_cakes = total_cakes_sold - cakes_made) ∧ 
                        (pastries_made = cakes_made - more_cakes_than_pastries)

theorem find_pastries_made : 
  baker_conditions cakes_made pastries_made total_cakes_sold total_pastries_sold extra_cakes more_cakes_than_pastries → 
  pastries_made = 146 :=
by
  sorry

end find_pastries_made_l207_207031


namespace b_k_divisible_by_11_is_5_l207_207531

def b (n : ℕ) : ℕ :=
  -- Function to concatenate numbers from 1 to n
  let digits := List.join (List.map (λ x => Nat.digits 10 x) (List.range' 1 n.succ))
  digits.foldl (λ acc d => acc * 10 + d) 0

def g (n : ℕ) : ℤ :=
  let digits := Nat.digits 10 n
  digits.enum.foldl (λ acc ⟨i, d⟩ => if i % 2 = 0 then acc + Int.ofNat d else acc - Int.ofNat d) 0

def isDivisibleBy11 (n : ℕ) : Bool :=
  g n % 11 = 0

def count_b_k_divisible_by_11 : ℕ :=
  List.length (List.filter isDivisibleBy11 (List.map b (List.range' 1 51)))

theorem b_k_divisible_by_11_is_5 : count_b_k_divisible_by_11 = 5 := by
  sorry

end b_k_divisible_by_11_is_5_l207_207531


namespace shaded_area_percentage_l207_207670

-- Define the given conditions
def square_area := 6 * 6
def shaded_area_left := (1 / 2) * 2 * 6
def shaded_area_right := (1 / 2) * 4 * 6
def total_shaded_area := shaded_area_left + shaded_area_right

-- State the theorem
theorem shaded_area_percentage : (total_shaded_area / square_area) * 100 = 50 := by
  sorry

end shaded_area_percentage_l207_207670


namespace construct_segment_length_l207_207524

theorem construct_segment_length (a b : ℝ) (h : a > b) : 
  ∃ c : ℝ, c = (a^2 + b^2) / (a - b) :=
by
  sorry

end construct_segment_length_l207_207524


namespace park_is_square_l207_207970

-- Defining the concept of a square field
def square_field : ℕ := 4

-- Given condition: The sum of the right angles from the park and the square field
axiom angles_sum (park_angles : ℕ) : park_angles + square_field = 8

-- The theorem to be proven
theorem park_is_square (park_angles : ℕ) (h : park_angles + square_field = 8) : park_angles = 4 :=
by sorry

end park_is_square_l207_207970


namespace common_difference_l207_207638

-- Define the arithmetic sequence with general term
def arithmetic_seq (a₁ d : ℕ) (n : ℕ) : ℕ := a₁ + (n - 1) * d

theorem common_difference (a₁ a₅ a₄ d : ℕ) 
  (h₁ : a₁ + a₅ = 10)
  (h₂ : a₄ = 7)
  (h₅ : a₅ = a₁ + 4 * d)
  (h₄ : a₄ = a₁ + 3 * d) :
  d = 2 :=
by
  sorry

end common_difference_l207_207638


namespace value_of_m_l207_207045

noncomputable def A (m : ℝ) : Set ℝ := {3, m}
noncomputable def B (m : ℝ) : Set ℝ := {3 * m, 3}

theorem value_of_m (m : ℝ) (h : A m = B m) : m = 0 :=
by
  sorry

end value_of_m_l207_207045


namespace volume_of_prism_l207_207835

theorem volume_of_prism
  (a b c : ℝ)
  (h1 : a * b = 30)
  (h2 : a * c = 40)
  (h3 : b * c = 60) :
  a * b * c = 120 * Real.sqrt 5 := 
by
  sorry

end volume_of_prism_l207_207835


namespace product_has_trailing_zeros_l207_207032

theorem product_has_trailing_zeros (a b : ℕ) (h1 : a = 350) (h2 : b = 60) :
  ∃ (n : ℕ), (10^n ∣ a * b) ∧ n = 3 :=
by
  sorry

end product_has_trailing_zeros_l207_207032


namespace number_of_full_boxes_l207_207895

theorem number_of_full_boxes (peaches_in_basket baskets_eaten_peaches box_capacity : ℕ) (h1 : peaches_in_basket = 23) (h2 : baskets = 7) (h3 : eaten_peaches = 7) (h4 : box_capacity = 13) :
  (peaches_in_basket * baskets - eaten_peaches) / box_capacity = 11 :=
by
  sorry

end number_of_full_boxes_l207_207895


namespace hotel_total_towels_l207_207257

theorem hotel_total_towels :
  let rooms_A := 25
  let rooms_B := 30
  let rooms_C := 15
  let members_per_room_A := 5
  let members_per_room_B := 6
  let members_per_room_C := 4
  let towels_per_member_A := 3
  let towels_per_member_B := 2
  let towels_per_member_C := 4
  (rooms_A * members_per_room_A * towels_per_member_A) +
  (rooms_B * members_per_room_B * towels_per_member_B) +
  (rooms_C * members_per_room_C * towels_per_member_C) = 975
:= by
  sorry

end hotel_total_towels_l207_207257


namespace sequence_a_100_l207_207504

theorem sequence_a_100 (a : ℕ → ℤ) (h₁ : a 1 = 3) (h₂ : ∀ n : ℕ, a (n + 1) = a n - 2) : a 100 = -195 :=
by
  sorry

end sequence_a_100_l207_207504


namespace new_cost_after_decrease_l207_207833

def actual_cost : ℝ := 2400
def decrease_percentage : ℝ := 0.50
def decreased_amount (cost percentage : ℝ) : ℝ := percentage * cost
def new_cost (cost decreased : ℝ) : ℝ := cost - decreased

theorem new_cost_after_decrease :
  new_cost actual_cost (decreased_amount actual_cost decrease_percentage) = 1200 :=
by sorry

end new_cost_after_decrease_l207_207833


namespace find_large_no_l207_207678

theorem find_large_no (L S : ℤ) (h1 : L - S = 1365) (h2 : L = 6 * S + 15) : L = 1635 :=
by 
  sorry

end find_large_no_l207_207678


namespace merchant_profit_condition_l207_207094

theorem merchant_profit_condition (L : ℝ) (P : ℝ) (S : ℝ) (M : ℝ) :
  (P = 0.70 * L) →
  (S = 0.80 * M) →
  (S - P = 0.30 * S) →
  (M = 1.25 * L) := 
by
  intros h1 h2 h3
  sorry

end merchant_profit_condition_l207_207094


namespace quadratic_equivalence_statement_l207_207718

noncomputable def quadratic_in_cos (a b c x : ℝ) : Prop := 
  a * (Real.cos x)^2 + b * Real.cos x + c = 0

noncomputable def transform_to_cos2x (a b c : ℝ) : Prop := 
  (4*a^2) * (Real.cos (2*a))^2 + (2*a^2 + 4*a*c - 2*b^2) * Real.cos (2*a) + a^2 + 4*a*c - 2*b^2 + 4*c^2 = 0

theorem quadratic_equivalence_statement (a b c : ℝ) (h : quadratic_in_cos 4 2 (-1) a) :
  transform_to_cos2x 16 12 (-4) :=
sorry

end quadratic_equivalence_statement_l207_207718


namespace order_of_numbers_l207_207770

noncomputable def a : ℝ := 60.7
noncomputable def b : ℝ := 0.76
noncomputable def c : ℝ := Real.log 0.76

theorem order_of_numbers : (c < b) ∧ (b < a) :=
by
  have h1 : c = Real.log 0.76 := rfl
  have h2 : b = 0.76 := rfl
  have h3 : a = 60.7 := rfl
  have hc : c < 0 := sorry
  have hb : 0 < b := sorry
  have ha : 1 < a := sorry
  sorry 

end order_of_numbers_l207_207770


namespace find_initial_nickels_l207_207986

variable (initial_nickels current_nickels borrowed_nickels : ℕ)

def initial_nickels_equation (initial_nickels current_nickels borrowed_nickels : ℕ) : Prop :=
  initial_nickels - borrowed_nickels = current_nickels

theorem find_initial_nickels (h : initial_nickels_equation initial_nickels current_nickels borrowed_nickels) 
                             (h_current : current_nickels = 11) 
                             (h_borrowed : borrowed_nickels = 20) : 
                             initial_nickels = 31 :=
by
  sorry

end find_initial_nickels_l207_207986


namespace negation_of_proposition_l207_207798

theorem negation_of_proposition :
  (¬ ∀ (x : ℝ), |x| < 0) ↔ (∃ (x : ℝ), |x| ≥ 0) := 
sorry

end negation_of_proposition_l207_207798


namespace determine_original_volume_of_tank_l207_207881

noncomputable def salt_volume (x : ℝ) := 0.20 * x
noncomputable def new_volume_after_evaporation (x : ℝ) := (3 / 4) * x
noncomputable def new_volume_after_additions (x : ℝ) := (3 / 4) * x + 6 + 12
noncomputable def new_salt_after_addition (x : ℝ) := 0.20 * x + 12
noncomputable def resulting_salt_concentration (x : ℝ) := (0.20 * x + 12) / ((3 / 4) * x + 18)

theorem determine_original_volume_of_tank (x : ℝ) :
  resulting_salt_concentration x = 1 / 3 → x = 120 := 
by 
  sorry

end determine_original_volume_of_tank_l207_207881


namespace victoria_initial_money_l207_207293

-- Definitions based on conditions
def cost_rice := 2 * 20
def cost_flour := 3 * 25
def cost_soda := 150
def total_spent := cost_rice + cost_flour + cost_soda
def remaining_balance := 235

-- Theorem to prove
theorem victoria_initial_money (initial_money : ℕ) :
  initial_money = total_spent + remaining_balance :=
by
  sorry

end victoria_initial_money_l207_207293


namespace greatest_k_dividing_abcdef_l207_207668

theorem greatest_k_dividing_abcdef {a b c d e f : ℤ}
  (h : a^2 + b^2 + c^2 + d^2 + e^2 = f^2) :
  ∃ k, (∀ a b c d e f, a^2 + b^2 + c^2 + d^2 + e^2 = f^2 → k ∣ (a * b * c * d * e * f)) ∧ k = 24 :=
sorry

end greatest_k_dividing_abcdef_l207_207668


namespace maria_total_earnings_l207_207151

noncomputable def total_earnings : ℕ := 
  let tulips_day1 := 30
  let roses_day1 := 20
  let lilies_day1 := 15
  let sunflowers_day1 := 10
  let tulips_day2 := tulips_day1 * 2
  let roses_day2 := roses_day1 * 2
  let lilies_day2 := lilies_day1
  let sunflowers_day2 := sunflowers_day1 * 3
  let tulips_day3 := tulips_day2 / 10
  let roses_day3 := 16
  let lilies_day3 := lilies_day1 / 2
  let sunflowers_day3 := sunflowers_day2
  let price_tulip := 2
  let price_rose := 3
  let price_lily := 4
  let price_sunflower := 5
  let day1_earnings := tulips_day1 * price_tulip + roses_day1 * price_rose + lilies_day1 * price_lily + sunflowers_day1 * price_sunflower
  let day2_earnings := tulips_day2 * price_tulip + roses_day2 * price_rose + lilies_day2 * price_lily + sunflowers_day2 * price_sunflower
  let day3_earnings := tulips_day3 * price_tulip + roses_day3 * price_rose + lilies_day3 * price_lily + sunflowers_day3 * price_sunflower
  day1_earnings + day2_earnings + day3_earnings

theorem maria_total_earnings : total_earnings = 920 := 
by 
  unfold total_earnings
  sorry

end maria_total_earnings_l207_207151


namespace solution_set_of_inequality_l207_207467

theorem solution_set_of_inequality :
  ∀ x : ℝ, (x^2 - 3*x - 4 ≤ 0) ↔ (-1 ≤ x ∧ x ≤ 4) :=
by sorry

end solution_set_of_inequality_l207_207467


namespace A_equals_k_with_conditions_l207_207932

theorem A_equals_k_with_conditions (n m : ℕ) (h_n : 1 < n) (h_m : 1 < m) :
  ∃ k : ℤ, (1 : ℝ) < k ∧ (( (n + Real.sqrt (n^2 - 4)) / 2 ) ^ m = (k + Real.sqrt (k^2 - 4)) / 2) :=
sorry

end A_equals_k_with_conditions_l207_207932


namespace non_empty_solution_set_inequality_l207_207942

theorem non_empty_solution_set_inequality (a : ℝ) :
  (∃ x : ℝ, |x + 1| - |x - 3| < a) ↔ a > -4 := 
sorry

end non_empty_solution_set_inequality_l207_207942


namespace common_chord_eqn_circle_with_center_on_line_smallest_area_circle_l207_207294

noncomputable def circle1 (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 2*y - 8 = 0
noncomputable def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 2*x + 10*y - 24 = 0

theorem common_chord_eqn :
  ∀ x y : ℝ, (circle1 x y ∧ circle2 x y) ↔ (x - 2*y + 4 = 0) :=
sorry

noncomputable def A : ℝ × ℝ := (-4, 0)
noncomputable def B : ℝ × ℝ := (0, 2)
noncomputable def line_y_eq_neg_x (x y : ℝ) : Prop := y = -x

theorem circle_with_center_on_line :
  ∃ (x y : ℝ), line_y_eq_neg_x x y ∧ ((x + 3)^2 + (y - 3)^2 = 10) :=
sorry

theorem smallest_area_circle :
  ∃ (x y : ℝ), ((x + 2)^2 + (y - 1)^2 = 5) :=
sorry

end common_chord_eqn_circle_with_center_on_line_smallest_area_circle_l207_207294


namespace find_n_l207_207216

-- Define the function to sum the digits of a natural number n
def digit_sum (n : ℕ) : ℕ := 
  -- This is a dummy implementation for now
  -- Normally, we would implement the sum of the digits of n
  sorry 

-- The main theorem that we want to prove
theorem find_n : ∃ (n : ℕ), digit_sum n + n = 2011 ∧ n = 1991 :=
by
  -- Proof steps would go here, but we're skipping those with sorry.
  sorry

end find_n_l207_207216


namespace erasers_left_l207_207098

/-- 
There are initially 250 erasers in a box. Doris takes 75 erasers, Mark takes 40 
erasers, and Ellie takes 30 erasers out of the box. Prove that 105 erasers are 
left in the box.
-/
theorem erasers_left (initial_erasers : ℕ) (doris_takes : ℕ) (mark_takes : ℕ) (ellie_takes : ℕ)
  (h_initial : initial_erasers = 250)
  (h_doris : doris_takes = 75)
  (h_mark : mark_takes = 40)
  (h_ellie : ellie_takes = 30) :
  initial_erasers - doris_takes - mark_takes - ellie_takes = 105 :=
  by 
  sorry

end erasers_left_l207_207098


namespace inequality_condition_l207_207870

noncomputable def inequality_holds_for_all (a b c : ℝ) : Prop :=
  ∀ (x : ℝ), a * Real.sin x + b * Real.cos x + c > 0

theorem inequality_condition (a b c : ℝ) :
  inequality_holds_for_all a b c ↔ Real.sqrt (a^2 + b^2) < c :=
by sorry

end inequality_condition_l207_207870


namespace unique_root_when_abs_t_gt_2_l207_207469

theorem unique_root_when_abs_t_gt_2 (t : ℝ) (h : |t| > 2) :
  ∃! x : ℝ, x^3 - 3 * x = t ∧ |x| > 2 :=
sorry

end unique_root_when_abs_t_gt_2_l207_207469


namespace center_of_circle_l207_207459

-- Define the circle in polar coordinates
def circle_polar (ρ θ : ℝ) : Prop := ρ = 2 * Real.sin θ ∧ 0 ≤ θ ∧ θ < 2 * Real.pi

-- Define the center of the circle in polar coordinates
def center_polar (ρ θ : ℝ) : Prop := (ρ = 1 ∧ θ = Real.pi / 2) ∨ (ρ = 1 ∧ θ = 3 * Real.pi / 2)

-- The theorem states that the center of the given circle in polar coordinates is (1, π/2) or (1, 3π/2)
theorem center_of_circle : ∃ (ρ θ : ℝ), circle_polar ρ θ → center_polar ρ θ :=
by
  -- The center of the circle given the condition in polar coordinate system is (1, π/2) or (1, 3π/2)
  sorry

end center_of_circle_l207_207459


namespace resulting_total_mass_l207_207841

-- Define initial conditions
def initial_total_mass : ℝ := 12
def initial_white_paint_mass : ℝ := 0.8 * initial_total_mass
def initial_black_paint_mass : ℝ := initial_total_mass - initial_white_paint_mass

-- Required condition for the new mixture
def final_white_paint_percentage : ℝ := 0.9

-- Prove that the resulting total mass of paint is 24 kg
theorem resulting_total_mass (x : ℝ) (h1 : initial_total_mass = 12) 
                            (h2 : initial_white_paint_mass = 0.8 * initial_total_mass)
                            (h3 : initial_black_paint_mass = initial_total_mass - initial_white_paint_mass)
                            (h4 : final_white_paint_percentage = 0.9) 
                            (h5 : (initial_white_paint_mass + x) / (initial_total_mass + x) = final_white_paint_percentage) : 
                            initial_total_mass + x = 24 :=
by 
  -- Temporarily assume the proof without detailing the solution steps
  sorry

end resulting_total_mass_l207_207841


namespace find_a_l207_207311

noncomputable def f (a : ℝ) : ℝ → ℝ := fun x =>
  if x < 1 then 2^x + 1 else x^2 + a * x

theorem find_a (a : ℝ) (h : f a (f a 0) = 4 * a) : a = 2 := by
  sorry

end find_a_l207_207311


namespace no_entangled_two_digit_numbers_l207_207899

theorem no_entangled_two_digit_numbers :
  ∀ (a b : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ b ≤ 9 → 10 * a + b ≠ 2 * (a + b ^ 3) :=
by
  intros a b h
  rcases h with ⟨ha1, ha9, hb9⟩
  sorry

end no_entangled_two_digit_numbers_l207_207899


namespace gcd_of_45_135_225_is_45_l207_207843

theorem gcd_of_45_135_225_is_45 : Nat.gcd (Nat.gcd 45 135) 225 = 45 :=
by
  sorry

end gcd_of_45_135_225_is_45_l207_207843


namespace parabola_vertex_sum_l207_207737

theorem parabola_vertex_sum (p q r : ℝ) (h1 : ∀ x : ℝ, x = p * (x - 3)^2 + 2 → y) (h2 : p * (1 - 3)^2 + 2 = 6) :
  p + q + r = 6 :=
sorry

end parabola_vertex_sum_l207_207737


namespace sum_slopes_const_zero_l207_207655

-- Define variables and constants
variable (p : ℝ) (h : 0 < p)

-- Define parabola and circle equations
def parabola_C1 (x y : ℝ) : Prop := y^2 = 2 * p * x
def circle_C2 (x y : ℝ) : Prop := x^2 + y^2 = p^2

-- Condition: The line segment length from circle cut by directrix
def segment_length_condition : Prop := ∃ d : ℝ, d^2 + 3 = p^2

-- The main theorem to prove
theorem sum_slopes_const_zero
  (A : ℝ × ℝ)
  (F : ℝ × ℝ := (p / 2, 0))
  (M N : ℝ × ℝ)
  (line_n_through_A : ∀ x : ℝ, x = 1 / p - 1 + 1 / p → (1 / p - 1 + x) = 0)
  (intersection_prop: parabola_C1 p M.1 M.2 ∧ parabola_C1 p N.1 N.2) 
  (slope_MF : ℝ := (M.2 / (p / 2 - M.1)) ) 
  (slope_NF : ℝ := (N.2 / (p / 2 - N.1))) :
  slope_MF + slope_NF = 0 := 
sorry

end sum_slopes_const_zero_l207_207655


namespace expression_value_l207_207363

theorem expression_value : 200 * (200 - 8) - (200 * 200 + 8) = -1608 := 
by 
  -- We will put the proof here
  sorry

end expression_value_l207_207363


namespace new_students_joined_l207_207640

-- Define conditions
def initial_students : ℕ := 160
def end_year_students : ℕ := 120
def fraction_transferred_out : ℚ := 1 / 3
def total_students_at_start := end_year_students * 3 / 2

-- Theorem statement
theorem new_students_joined : (total_students_at_start - initial_students = 20) :=
by
  -- Placeholder for proof
  sorry

end new_students_joined_l207_207640


namespace find_d_l207_207053

-- Define the polynomial g(x)
def g (d : ℚ) (x : ℚ) : ℚ := d * x^4 + 11 * x^3 + 5 * d * x^2 - 28 * x + 72

-- The main proof statement
theorem find_d (hd : g d 4 = 0) : d = -83 / 42 := by
  sorry -- proof not needed as per prompt

end find_d_l207_207053


namespace find_r_l207_207281

-- Lean statement
theorem find_r (r : ℚ) (log_eq : Real.logb 81 (2 * r - 1) = -1 / 2) : r = 5 / 9 :=
by {
    sorry -- proof steps should not be included according to the requirements
}

end find_r_l207_207281


namespace square_eq_four_implies_two_l207_207354

theorem square_eq_four_implies_two (x : ℝ) (h : x^2 = 4) : x = 2 := 
sorry

end square_eq_four_implies_two_l207_207354


namespace find_third_number_l207_207419

theorem find_third_number (x : ℕ) (h : (6 + 16 + x) / 3 = 13) : x = 17 :=
by
  sorry

end find_third_number_l207_207419


namespace average_speed_comparison_l207_207084

theorem average_speed_comparison (u v w : ℝ) (hu : u > 0) (hv : v > 0) (hw : w > 0):
  (3 / (1 / u + 1 / v + 1 / w)) ≤ ((u + v + w) / 3) :=
sorry

end average_speed_comparison_l207_207084


namespace polygon_sides_eq_six_l207_207426

theorem polygon_sides_eq_six (n : ℕ) (h : 3 * n - (n * (n - 3)) / 2 = 6) : n = 6 := 
sorry

end polygon_sides_eq_six_l207_207426


namespace fraction_nonnegative_for_all_reals_l207_207368

theorem fraction_nonnegative_for_all_reals (x : ℝ) : 
  (x^2 + 2 * x + 1) / (x^2 + 4 * x + 8) ≥ 0 :=
by
  sorry

end fraction_nonnegative_for_all_reals_l207_207368


namespace combined_tickets_l207_207925

-- Definitions from the conditions
def dave_spent : Nat := 43
def dave_left : Nat := 55
def alex_spent : Nat := 65
def alex_left : Nat := 42

-- Theorem to prove that the combined starting tickets of Dave and Alex is 205
theorem combined_tickets : dave_spent + dave_left + alex_spent + alex_left = 205 := 
by
  sorry

end combined_tickets_l207_207925


namespace x_y_solution_l207_207030

variable (x y : ℕ)

noncomputable def x_wang_speed : ℕ := x - 6

theorem x_y_solution (hx : (5 : ℚ) / 6 * x = y) (hy : (2 : ℚ) / 3 * (x - 6) = y - 10) : x = 36 ∧ y = 30 :=
by {
  sorry
}

end x_y_solution_l207_207030


namespace fraction_less_than_40_percent_l207_207633

theorem fraction_less_than_40_percent (x : ℝ) (h1 : x * 180 = 48) (h2 : x < 0.4) : x = 4 / 15 :=
by
  sorry

end fraction_less_than_40_percent_l207_207633


namespace shooter_with_more_fluctuation_l207_207339

noncomputable def variance (scores : List ℕ) (mean : ℕ) : ℚ :=
  (List.sum (List.map (λ x => (x - mean) * (x - mean)) scores) : ℚ) / scores.length

theorem shooter_with_more_fluctuation :
  let scores_A := [7, 9, 8, 6, 10]
  let scores_B := [7, 8, 9, 8, 8]
  let mean := 8
  variance scores_A mean > variance scores_B mean :=
by
  sorry

end shooter_with_more_fluctuation_l207_207339


namespace import_tax_excess_amount_l207_207102

theorem import_tax_excess_amount (X : ℝ)
  (total_value : ℝ) (tax_paid : ℝ)
  (tax_rate : ℝ) :
  total_value = 2610 → tax_paid = 112.70 → tax_rate = 0.07 → 0.07 * (2610 - X) = 112.70 → X = 1000 :=
by
  intros h1 h2 h3 h4
  sorry

end import_tax_excess_amount_l207_207102


namespace negation_of_P_l207_207131

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ (m : ℕ), m ∣ n → m = 1 ∨ m = n)

def is_odd (n : ℕ) : Prop :=
  n % 2 = 1

def P : Prop := ∀ n : ℕ, is_prime n → is_odd n

theorem negation_of_P : ¬ P ↔ ∃ n : ℕ, is_prime n ∧ ¬ is_odd n :=
by sorry

end negation_of_P_l207_207131


namespace isosceles_triangle_of_condition_l207_207430

theorem isosceles_triangle_of_condition (A B C : ℝ) (a b c : ℝ)
  (h1 : a = 2 * b * Real.cos C)
  (h2 : A + B + C = Real.pi) :
  (B = C) ∨ (A = C) ∨ (A = B) := 
sorry

end isosceles_triangle_of_condition_l207_207430


namespace cos_value_l207_207849

theorem cos_value (α : ℝ) (h : Real.sin (π / 5 - α) = 1 / 4) : 
  Real.cos (2 * α + 3 * π / 5) = -7 / 8 := 
by
  sorry

end cos_value_l207_207849


namespace seonyeong_class_size_l207_207627

theorem seonyeong_class_size :
  (12 * 4 + 3) - 12 = 39 :=
by
  sorry

end seonyeong_class_size_l207_207627


namespace price_reduction_equation_l207_207193

theorem price_reduction_equation (x : ℝ) (P_initial : ℝ) (P_final : ℝ) 
  (h1 : P_initial = 560) (h2 : P_final = 315) : 
  P_initial * (1 - x)^2 = P_final :=
by
  rw [h1, h2]
  sorry

end price_reduction_equation_l207_207193


namespace area_of_smaller_circle_l207_207601

theorem area_of_smaller_circle (r R : ℝ) (PA AB : ℝ) 
  (h1 : R = 2 * r) (h2 : PA = 4) (h3 : AB = 4) :
  π * r^2 = 2 * π :=
by
  sorry

end area_of_smaller_circle_l207_207601


namespace sum_six_times_product_l207_207692

variable (a b x : ℝ)

theorem sum_six_times_product (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a + b = 6 * x) (h4 : 1/a + 1/b = 6) :
  x = a * b := sorry

end sum_six_times_product_l207_207692


namespace lines_do_not_intersect_l207_207717

theorem lines_do_not_intersect (b : ℝ) :
  ∀ s v : ℝ,
    (2 + 3 * s = 5 + 6 * v) →
    (1 + 4 * s = 3 + 3 * v) →
    (b + 5 * s = 1 + 2 * v) →
    b ≠ -4/5 :=
by
  intros s v h1 h2 h3
  sorry

end lines_do_not_intersect_l207_207717


namespace magnitude_quotient_l207_207200

open Complex

theorem magnitude_quotient : 
  abs ((1 + 2 * I) / (2 - I)) = 1 := 
by 
  sorry

end magnitude_quotient_l207_207200


namespace point_in_fourth_quadrant_l207_207990

def lies_in_fourth_quadrant (P : ℤ × ℤ) : Prop :=
  P.fst > 0 ∧ P.snd < 0

theorem point_in_fourth_quadrant : lies_in_fourth_quadrant (2023, -2024) :=
by
  -- Here is where the proof steps would go
  sorry

end point_in_fourth_quadrant_l207_207990


namespace cos_sq_alpha_cos_sq_beta_range_l207_207184

theorem cos_sq_alpha_cos_sq_beta_range
  (α β : ℝ)
  (h : 3 * (Real.sin α)^2 + 2 * (Real.sin β)^2 - 2 * Real.sin α = 0) :
  (Real.cos α)^2 + (Real.cos β)^2 ∈ Set.Icc (14 / 9) 2 :=
sorry

end cos_sq_alpha_cos_sq_beta_range_l207_207184


namespace red_peaches_per_basket_l207_207207

theorem red_peaches_per_basket (R : ℕ) (green_peaches_per_basket : ℕ) (number_of_baskets : ℕ) (total_peaches : ℕ) (h1 : green_peaches_per_basket = 4) (h2 : number_of_baskets = 15) (h3 : total_peaches = 345) : R = 19 :=
by
  sorry

end red_peaches_per_basket_l207_207207


namespace tommy_first_house_price_l207_207134

theorem tommy_first_house_price (C : ℝ) (P : ℝ) (loan_rate : ℝ) (interest_rate : ℝ)
  (term : ℝ) (property_tax_rate : ℝ) (insurance_cost : ℝ) 
  (price_ratio : ℝ) (monthly_payment : ℝ) :
  C = 500000 ∧ price_ratio = 1.25 ∧ P * price_ratio = C ∧
  loan_rate = 0.75 ∧ interest_rate = 0.035 ∧ term = 15 ∧
  property_tax_rate = 0.015 ∧ insurance_cost = 7500 → 
  P = 400000 :=
by sorry

end tommy_first_house_price_l207_207134


namespace expression_evaluation_l207_207682

theorem expression_evaluation :
  (0.86^3) - ((0.1^3) / (0.86^2)) + 0.086 + (0.1^2) = 0.730704 := 
by 
  sorry

end expression_evaluation_l207_207682


namespace factorize_quadratic_example_l207_207126

theorem factorize_quadratic_example (x : ℝ) :
  4 * x^2 - 8 * x + 4 = 4 * (x - 1)^2 :=
by
  sorry

end factorize_quadratic_example_l207_207126


namespace initial_mean_l207_207404

theorem initial_mean (M : ℝ) (h1 : 50 * (36.5 : ℝ) - 23 = 50 * (36.04 : ℝ) + 23)
: M = 36.04 :=
by
  sorry

end initial_mean_l207_207404


namespace intersection_and_perpendicular_line_l207_207819

theorem intersection_and_perpendicular_line :
  ∃ (x y : ℝ), (3 * x + y - 1 = 0) ∧ (x + 2 * y - 7 = 0) ∧ (2 * x - y + 6 = 0) :=
by
  sorry

end intersection_and_perpendicular_line_l207_207819


namespace triangular_pyramid_volume_l207_207165

theorem triangular_pyramid_volume (a b c : ℝ)
  (h1 : 1/2 * a * b = 1.5)
  (h2 : 1/2 * b * c = 2)
  (h3 : 1/2 * a * c = 6) :
  (1/6 * a * b * c = 2) :=
by {
  -- Here, we would provide the proof steps, but for now we leave it as sorry
  sorry
}

end triangular_pyramid_volume_l207_207165


namespace value_of_a_l207_207828

noncomputable def f (x : ℝ) : ℝ := x^2 + 10
noncomputable def g (x : ℝ) : ℝ := x^2 - 5

theorem value_of_a (a : ℝ) (h₁ : a > 0) (h₂ : f (g a) = 18) :
  a = Real.sqrt (5 + 2 * Real.sqrt 2) ∨ a = Real.sqrt (5 - 2 * Real.sqrt 2) := 
by
  sorry

end value_of_a_l207_207828


namespace probability_of_union_l207_207024

-- Define the range of two-digit numbers
def digit_count : ℕ := 90

-- Define events A and B
def event_a (n : ℕ) : Prop := n % 2 = 0
def event_b (n : ℕ) : Prop := n % 5 = 0

-- Define the probabilities P(A), P(B), and P(A ∩ B)
def P_A : ℚ := 45 / digit_count
def P_B : ℚ := 18 / digit_count
def P_A_and_B : ℚ := 9 / digit_count

-- Prove the final probability using inclusion-exclusion principle
theorem probability_of_union : P_A + P_B - P_A_and_B = 0.6 := by
  sorry

end probability_of_union_l207_207024


namespace equation_solution_l207_207725

theorem equation_solution :
  ∃ a b c d : ℤ, a > 0 ∧ (∀ x : ℝ, (64 * x^2 + 96 * x - 36) = (a * x + b)^2 + d) ∧ c = -36 ∧ a + b + c + d = -94 :=
by sorry

end equation_solution_l207_207725


namespace min_value_of_squared_sum_l207_207788

open Real

theorem min_value_of_squared_sum (x y z : ℝ) (h : x^3 + y^3 + z^3 - 3 * x * y * z = 8) :
  ∃ m, m = (x^2 + y^2 + z^2) ∧ m = 16 / 3 :=
by
  sorry

end min_value_of_squared_sum_l207_207788


namespace find_coprime_pairs_l207_207500

theorem find_coprime_pairs :
  ∀ (x y : ℕ), x > 0 → y > 0 → x.gcd y = 1 →
    (x ∣ y^2 + 210) →
    (y ∣ x^2 + 210) →
    (x = 1 ∧ y = 1) ∨ (x = 1 ∧ y = 211) ∨ 
    (∃ n : ℕ, n > 0 ∧ n = 1 ∧ n = 1 ∧ 
      (x = 212*n - n - 1 ∨ y = 212*n - n - 1)) := sorry

end find_coprime_pairs_l207_207500


namespace union_of_A_and_B_l207_207242

open Set

-- Definitions for the conditions
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 3, 4}

-- Statement of the theorem
theorem union_of_A_and_B : A ∪ B = {1, 2, 3, 4} :=
by
  sorry

end union_of_A_and_B_l207_207242


namespace original_number_l207_207878

theorem original_number (x : ℝ) (h1 : 1.5 * x = 135) : x = 90 :=
by
  sorry

end original_number_l207_207878


namespace equal_pair_b_l207_207825

def exprA1 := -3^2
def exprA2 := -2^3

def exprB1 := -6^3
def exprB2 := (-6)^3

def exprC1 := -6^2
def exprC2 := (-6)^2

def exprD1 := (-3 * 2)^2
def exprD2 := (-3) * 2^2

theorem equal_pair_b : exprB1 = exprB2 :=
by {
  -- proof steps should go here
  sorry
}

end equal_pair_b_l207_207825


namespace SallyCarrots_l207_207883

-- Definitions of the conditions
def FredGrew (F : ℕ) := F = 4
def TotalGrew (T : ℕ) := T = 10
def SallyGrew (S : ℕ) (F T : ℕ) := S + F = T

-- The theorem to be proved
theorem SallyCarrots : ∃ S : ℕ, FredGrew 4 ∧ TotalGrew 10 ∧ SallyGrew S 4 10 ∧ S = 6 :=
  sorry

end SallyCarrots_l207_207883


namespace number_of_20_paise_coins_l207_207162

theorem number_of_20_paise_coins (x y : ℕ) (h1 : x + y = 324) (h2 : 20 * x + 25 * y = 7000) : x = 220 :=
  sorry

end number_of_20_paise_coins_l207_207162


namespace carla_drank_total_amount_l207_207089

-- Define the conditions
def carla_water : ℕ := 15
def carla_soda := 3 * carla_water - 6
def total_liquid := carla_water + carla_soda

-- State the theorem
theorem carla_drank_total_amount : total_liquid = 54 := by
  sorry

end carla_drank_total_amount_l207_207089


namespace factor_of_change_l207_207543

-- Given conditions
def avg_marks_before : ℕ := 45
def avg_marks_after : ℕ := 90
def num_students : ℕ := 30

-- Prove the factor F by which marks are changed
theorem factor_of_change : ∃ F : ℕ, avg_marks_before * F = avg_marks_after := 
by
  use 2
  have h1 : 30 * avg_marks_before = 30 * 45 := rfl
  have h2 : 30 * avg_marks_after = 30 * 90 := rfl
  sorry

end factor_of_change_l207_207543


namespace circle_radius_is_2_chord_length_is_2sqrt3_l207_207283

-- Define the given conditions
def inclination_angle_line_incl60 : Prop := ∃ m, m = Real.sqrt 3
def circle_eq : Prop := ∀ x y, x^2 + y^2 - 4 * y = 0

-- Prove: radius of the circle
theorem circle_radius_is_2 (h : circle_eq) : radius = 2 := sorry

-- Prove: length of the chord cut by the line
theorem chord_length_is_2sqrt3 
  (h1 : inclination_angle_line_incl60) 
  (h2 : circle_eq) : chord_length = 2 * Real.sqrt 3 := sorry

end circle_radius_is_2_chord_length_is_2sqrt3_l207_207283


namespace equivalence_of_min_perimeter_and_cyclic_quadrilateral_l207_207191

-- Definitions for points P, Q, R, S on sides of quadrilateral ABCD
-- Function definitions for conditions and equivalence of stated problems

variable {A B C D P Q R S : Type*} 

def is_on_side (P : Type*) (A B : Type*) : Prop := sorry
def is_interior_point (P : Type*) (A B : Type*) : Prop := sorry
def is_convex_quadrilateral (A B C D : Type*) : Prop := sorry
def is_cyclic_quadrilateral (A B C D : Type*) : Prop := sorry
def has_circumcenter_interior (A B C D : Type*) : Prop := sorry
def has_minimal_perimeter (P Q R S : Type*) : Prop := sorry

theorem equivalence_of_min_perimeter_and_cyclic_quadrilateral 
  (h1 : is_convex_quadrilateral A B C D) 
  (hP : is_on_side P A B ∧ is_interior_point P A B) 
  (hQ : is_on_side Q B C ∧ is_interior_point Q B C) 
  (hR : is_on_side R C D ∧ is_interior_point R C D) 
  (hS : is_on_side S D A ∧ is_interior_point S D A) :
  (∃ P' Q' R' S', has_minimal_perimeter P' Q' R' S') ↔ (is_cyclic_quadrilateral A B C D ∧ has_circumcenter_interior A B C D) :=
sorry

end equivalence_of_min_perimeter_and_cyclic_quadrilateral_l207_207191


namespace sin_expression_l207_207391

noncomputable def f (a b x : ℝ) : ℝ := a * Real.sin x + b * Real.cos x

theorem sin_expression (a b x₀ : ℝ) (h₀ : a ≠ 0) (h₁ : b ≠ 0)
  (h₂ : ∀ x, f a b x = f a b (π / 6 - x)) 
  (h₃ : f a b x₀ = (8 / 5) * a) 
  (h₄ : b = Real.sqrt 3 * a) :
  Real.sin (2 * x₀ + π / 6) = 7 / 25 :=
by
  sorry

end sin_expression_l207_207391


namespace sufficient_but_not_necessary_condition_for_x_1_l207_207234

noncomputable def sufficient_but_not_necessary_condition (x : ℝ) : Prop :=
(x = 1 → (x = 1 ∨ x = 2)) ∧ ¬ ((x = 1 ∨ x = 2) → x = 1)

theorem sufficient_but_not_necessary_condition_for_x_1 :
  sufficient_but_not_necessary_condition 1 :=
by
  sorry

end sufficient_but_not_necessary_condition_for_x_1_l207_207234


namespace triangle_AC_5_sqrt_3_l207_207812

theorem triangle_AC_5_sqrt_3 
  (A B C : ℝ)
  (BC AC : ℝ)
  (h1 : 2 * Real.sin (A - B) + Real.cos (B + C) = 2)
  (h2 : BC = 5) :
  AC = 5 * Real.sqrt 3 :=
  sorry

end triangle_AC_5_sqrt_3_l207_207812


namespace arcsin_neg_sqrt_two_over_two_l207_207913

theorem arcsin_neg_sqrt_two_over_two : Real.arcsin (-Real.sqrt 2 / 2) = -Real.pi / 4 :=
  sorry

end arcsin_neg_sqrt_two_over_two_l207_207913


namespace red_markers_count_l207_207588

-- Define the given conditions
def blue_markers : ℕ := 1028
def total_markers : ℕ := 3343

-- Define the red_makers calculation based on the conditions
def red_markers (total_markers blue_markers : ℕ) : ℕ := total_markers - blue_markers

-- Prove that the number of red markers is 2315 given the conditions
theorem red_markers_count : red_markers total_markers blue_markers = 2315 := by
  -- We can skip the proof for this demonstration
  sorry

end red_markers_count_l207_207588


namespace quadratic_range_l207_207071

open Real

theorem quadratic_range (a : ℝ) (h : ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ a * x1^2 + 4 * x1 - 2 = 0 ∧ a * x2^2 + 4 * x2 - 2 = 0) : 
  a > -2 ∧ a ≠ 0 :=
by 
  sorry

end quadratic_range_l207_207071


namespace molecular_weight_one_mole_l207_207226

theorem molecular_weight_one_mole
  (molecular_weight_7_moles : ℝ)
  (mole_count : ℝ)
  (h : molecular_weight_7_moles = 126)
  (k : mole_count = 7)
  : molecular_weight_7_moles / mole_count = 18 := 
sorry

end molecular_weight_one_mole_l207_207226


namespace polynomial_real_root_l207_207178

variable {A B C D E : ℝ}

theorem polynomial_real_root
  (h : ∃ t : ℝ, t > 1 ∧ A * t^2 + (C - B) * t + (E - D) = 0) :
  ∃ x : ℝ, A * x^4 + B * x^3 + C * x^2 + D * x + E = 0 :=
by
  sorry

end polynomial_real_root_l207_207178


namespace volume_hemisphere_from_sphere_l207_207845

theorem volume_hemisphere_from_sphere (r : ℝ) (V_sphere : ℝ) (V_hemisphere : ℝ) 
  (h1 : V_sphere = 150 * Real.pi) 
  (h2 : V_sphere = (4 / 3) * Real.pi * r^3) : 
  V_hemisphere = 75 * Real.pi :=
by
  sorry

end volume_hemisphere_from_sphere_l207_207845


namespace find_a_2016_l207_207082

noncomputable def a (n : ℕ) : ℕ := sorry

axiom condition_1 : a 4 = 1
axiom condition_2 : a 11 = 9
axiom condition_3 : ∀ n : ℕ, a n + a (n+1) + a (n+2) = 15

theorem find_a_2016 : a 2016 = 5 := sorry

end find_a_2016_l207_207082


namespace survey_participants_l207_207497

-- Total percentage for option A and option B in bytes
def percent_A : ℝ := 0.50
def percent_B : ℝ := 0.30

-- Number of participants who chose option A
def participants_A : ℕ := 150

-- Target number of participants who chose option B (to be proved)
def participants_B : ℕ := 90

-- The theorem to prove the number of participants who chose option B
theorem survey_participants :
  (participants_B : ℝ) = participants_A * (percent_B / percent_A) :=
by
  sorry

end survey_participants_l207_207497


namespace entrance_fee_per_person_l207_207251

theorem entrance_fee_per_person :
  let ticket_price := 50.00
  let processing_fee_rate := 0.15
  let parking_fee := 10.00
  let total_cost := 135.00
  let known_cost := 2 * ticket_price + processing_fee_rate * (2 * ticket_price) + parking_fee
  ∃ entrance_fee_per_person, 2 * entrance_fee_per_person + known_cost = total_cost :=
by
  sorry

end entrance_fee_per_person_l207_207251


namespace part1_solution_part2_solution_l207_207786

noncomputable def part1_expr := (1 / (Real.sqrt 5 + 2)) - (Real.sqrt 3 - 1)^0 - Real.sqrt (9 - 4 * Real.sqrt 5)
theorem part1_solution : part1_expr = 2 := by
  sorry

noncomputable def part2_expr := 2 * Real.sqrt 3 * 612 * (7/2)
theorem part2_solution : part2_expr = 5508 * Real.sqrt 3 := by
  sorry

end part1_solution_part2_solution_l207_207786


namespace cesaro_sum_100_terms_l207_207491

noncomputable def cesaro_sum (A : List ℝ) : ℝ :=
  let n := A.length
  (List.sum A) / n

theorem cesaro_sum_100_terms :
  ∀ (A : List ℝ), A.length = 99 →
  cesaro_sum A = 1000 →
  cesaro_sum (1 :: A) = 991 :=
by
  intros A h1 h2
  sorry

end cesaro_sum_100_terms_l207_207491


namespace tickets_difference_l207_207716

def tickets_used_for_clothes : ℝ := 85
def tickets_used_for_accessories : ℝ := 45.5
def tickets_used_for_food : ℝ := 51
def tickets_used_for_toys : ℝ := 58

theorem tickets_difference : 
  (tickets_used_for_clothes + tickets_used_for_food + tickets_used_for_accessories) - tickets_used_for_toys = 123.5 := 
by
  sorry

end tickets_difference_l207_207716


namespace range_of_a_l207_207590

theorem range_of_a (a : ℝ) (h1 : a > 0) (h2 : ∃ x : ℝ, |x - 4| + |x + 3| < a) : a > 7 :=
sorry

end range_of_a_l207_207590


namespace john_allowance_spent_l207_207290

theorem john_allowance_spent (B t d : ℝ) (h1 : t = 0.25 * (B - d)) (h2 : d = 0.10 * (B - t)) :
  (t + d) / B = 0.31 := by
  sorry

end john_allowance_spent_l207_207290


namespace max_overlap_l207_207541

variable (A : Type) [Fintype A] [DecidableEq A]
variable (P1 P2 : A → Prop)

theorem max_overlap (hP1 : ∃ X : Finset A, (X.card : ℝ) / Fintype.card A = 0.20 ∧ ∀ a ∈ X, P1 a)
                    (hP2 : ∃ Y : Finset A, (Y.card : ℝ) / Fintype.card A = 0.70 ∧ ∀ a ∈ Y, P2 a) :
  ∃ Z : Finset A, (Z.card : ℝ) / Fintype.card A = 0.20 ∧ ∀ a ∈ Z, P1 a ∧ P2 a :=
sorry

end max_overlap_l207_207541


namespace find_integer_n_l207_207235

noncomputable def cubic_expr_is_pure_integer (n : ℤ) : Prop :=
  (729 * n ^ 6 - 540 * n ^ 4 + 240 * n ^ 2 - 64 : ℂ).im = 0

theorem find_integer_n :
  ∃! n : ℤ, cubic_expr_is_pure_integer n := 
sorry

end find_integer_n_l207_207235


namespace tangent_ln_at_origin_l207_207352

theorem tangent_ln_at_origin {k : ℝ} (h : ∀ x : ℝ, (k * x = Real.log x) → k = 1 / x) : k = 1 / Real.exp 1 :=
by
  sorry

end tangent_ln_at_origin_l207_207352


namespace correct_calculation_l207_207130

theorem correct_calculation (m n : ℝ) : 4 * m + 2 * n - (n - m) = 5 * m + n :=
by sorry

end correct_calculation_l207_207130


namespace expression_value_l207_207813

variable (m n : ℝ)

theorem expression_value (h : m - n = 1) : (m - n)^2 - 2 * m + 2 * n = -1 :=
by
  sorry

end expression_value_l207_207813


namespace zan_guo_gets_one_deer_l207_207440

noncomputable def a1 : ℚ := 5 / 3
noncomputable def sum_of_sequence (a1 : ℚ) (d : ℚ) : ℚ := 5 * a1 + (5 * 4 / 2) * d
noncomputable def d : ℚ := -1 / 3
noncomputable def a3 (a1 : ℚ) (d : ℚ) : ℚ := a1 + 2 * d

theorem zan_guo_gets_one_deer :
  a3 a1 d = 1 := by
  sorry

end zan_guo_gets_one_deer_l207_207440


namespace jovana_shells_l207_207635

theorem jovana_shells (initial_shells : ℕ) (added_shells : ℕ) (total_shells : ℕ) 
  (h_initial : initial_shells = 5) (h_added : added_shells = 12) :
  total_shells = 17 :=
by
  sorry

end jovana_shells_l207_207635


namespace size_of_former_apartment_l207_207168

open Nat

theorem size_of_former_apartment
  (former_rent_rate : ℕ)
  (new_apartment_cost : ℕ)
  (savings_per_year : ℕ)
  (split_factor : ℕ)
  (savings_per_month : ℕ)
  (share_new_rent : ℕ)
  (former_rent : ℕ)
  (apartment_size : ℕ)
  (h1 : former_rent_rate = 2)
  (h2 : new_apartment_cost = 2800)
  (h3 : savings_per_year = 1200)
  (h4 : split_factor = 2)
  (h5 : savings_per_month = savings_per_year / 12)
  (h6 : share_new_rent = new_apartment_cost / split_factor)
  (h7 : former_rent = share_new_rent + savings_per_month)
  (h8 : apartment_size = former_rent / former_rent_rate) :
  apartment_size = 750 :=
by
  sorry

end size_of_former_apartment_l207_207168


namespace num_four_digit_int_with_4_or_5_correct_l207_207782

def num_four_digit_int_with_4_or_5 : ℕ :=
  5416

theorem num_four_digit_int_with_4_or_5_correct (A B : ℕ) (hA : A = 9000) (hB : B = 3584) :
  num_four_digit_int_with_4_or_5 = A - B :=
by
  rw [hA, hB]
  sorry

end num_four_digit_int_with_4_or_5_correct_l207_207782


namespace ln_abs_a_even_iff_a_eq_zero_l207_207243

theorem ln_abs_a_even_iff_a_eq_zero (a : ℝ) :
  (∀ x : ℝ, Real.log (abs (x - a)) = Real.log (abs (-x - a))) ↔ (a = 0) :=
by
  sorry

end ln_abs_a_even_iff_a_eq_zero_l207_207243


namespace living_space_increase_l207_207408

theorem living_space_increase (a b x : ℝ) (h₁ : a = 10) (h₂ : b = 12.1) : a * (1 + x) ^ 2 = b :=
sorry

end living_space_increase_l207_207408


namespace reynald_volleyballs_l207_207606

def total_balls : ℕ := 145
def soccer_balls : ℕ := 20
def basketballs : ℕ := soccer_balls + 5
def tennis_balls : ℕ := 2 * soccer_balls
def baseballs : ℕ := soccer_balls + 10
def volleyballs : ℕ := total_balls - (soccer_balls + basketballs + tennis_balls + baseballs)

theorem reynald_volleyballs : volleyballs = 30 :=
by
  sorry

end reynald_volleyballs_l207_207606


namespace tan_half_difference_l207_207374

-- Given two angles a and b with the following conditions
variables (a b : ℝ)
axiom cos_cond : (Real.cos a + Real.cos b = 3 / 5)
axiom sin_cond : (Real.sin a + Real.sin b = 2 / 5)

-- Prove that tan ((a - b) / 2) = 2 / 3
theorem tan_half_difference (a b : ℝ) (cos_cond : Real.cos a + Real.cos b = 3 / 5) 
  (sin_cond : Real.sin a + Real.sin b = 2 / 5) : 
  Real.tan ((a - b) / 2) = 2 / 3 := 
sorry

end tan_half_difference_l207_207374


namespace convergent_inequalities_l207_207537

theorem convergent_inequalities (α : ℝ) (P Q : ℕ → ℤ) (h_convergent : ∀ n ≥ 1, abs (α - P n / Q n) < 1 / (2 * (Q n) ^ 2) ∨ abs (α - P (n - 1) / Q (n - 1)) < 1 / (2 * (Q (n - 1))^2))
  (h_continued_fraction : ∀ n ≥ 1, P (n-1) * Q n - P n * Q (n-1) = (-1)^(n-1)) :
  ∃ p q : ℕ, 0 < q ∧ abs (α - p / q) < 1 / (2 * q^2) :=
sorry

end convergent_inequalities_l207_207537


namespace first_discount_correct_l207_207050

noncomputable def first_discount (x : ℝ) : Prop :=
  let initial_price := 600
  let first_discounted_price := initial_price * (1 - x / 100)
  let final_price := first_discounted_price * (1 - 0.05)
  final_price = 456

theorem first_discount_correct : ∃ x : ℝ, first_discount x ∧ abs (x - 57.29) < 0.01 :=
by
  sorry

end first_discount_correct_l207_207050


namespace bowling_ball_weight_l207_207494

theorem bowling_ball_weight (b c : ℕ) 
  (h1 : 5 * b = 3 * c) 
  (h2 : 3 * c = 105) : 
  b = 21 := 
  sorry

end bowling_ball_weight_l207_207494


namespace second_largest_subtract_smallest_correct_l207_207894

-- Definition of the elements
def numbers : List ℕ := [10, 11, 12, 13, 14]

-- Conditions derived from the problem
def smallest_number : ℕ := 10
def second_largest_number : ℕ := 13

-- Lean theorem statement representing the problem
theorem second_largest_subtract_smallest_correct :
  (second_largest_number - smallest_number) = 3 := 
by
  sorry

end second_largest_subtract_smallest_correct_l207_207894


namespace clipping_per_friend_l207_207861

def GluePerClipping : Nat := 6
def TotalGlue : Nat := 126
def TotalFriends : Nat := 7

theorem clipping_per_friend :
  (TotalGlue / GluePerClipping) / TotalFriends = 3 := by
  sorry

end clipping_per_friend_l207_207861


namespace cistern_length_l207_207924

def cistern_conditions (L : ℝ) : Prop := 
  let width := 4
  let depth := 1.25
  let wet_surface_area := 42.5
  (L * width) + (2 * (L * depth)) + (2 * (width * depth)) = wet_surface_area

theorem cistern_length : 
  ∃ L : ℝ, cistern_conditions L ∧ L = 5 := sorry

end cistern_length_l207_207924


namespace multiple_of_spending_on_wednesday_l207_207872

-- Definitions based on the conditions
def monday_spending : ℤ := 60
def tuesday_spending : ℤ := 4 * monday_spending
def total_spending : ℤ := 600

-- Problem to prove
theorem multiple_of_spending_on_wednesday (x : ℤ) : 
  monday_spending + tuesday_spending + x * monday_spending = total_spending → 
  x = 5 := by
  sorry

end multiple_of_spending_on_wednesday_l207_207872


namespace exponent_multiplication_l207_207495

theorem exponent_multiplication (a : ℝ) (m n : ℝ) (h1 : a^m = 2) (h2 : a^n = 4) :
  a^(m + n) = 8 := by
  sorry

end exponent_multiplication_l207_207495


namespace minimum_distance_from_lattice_point_to_line_l207_207006

theorem minimum_distance_from_lattice_point_to_line :
  let distance (x y : ℤ) := |25 * x - 15 * y + 12| / (5 * Real.sqrt 34)
  ∃ (x y : ℤ), distance x y = Real.sqrt 34 / 85 :=
sorry

end minimum_distance_from_lattice_point_to_line_l207_207006


namespace train_speed_l207_207158

theorem train_speed (length_of_train : ℝ) (time_to_cross : ℝ) (conversion_factor : ℝ) :
  length_of_train = 200 → 
  time_to_cross = 24 → 
  conversion_factor = 3600 → 
  (length_of_train / 1000) / (time_to_cross / conversion_factor) = 30 := 
by
  sorry

end train_speed_l207_207158


namespace cylinder_height_and_diameter_l207_207814

/-- The surface area of a sphere is the same as the curved surface area of a right circular cylinder.
    The height and diameter of the cylinder are the same, and the radius of the sphere is 4 cm.
    Prove that the height and diameter of the cylinder are both 8 cm. --/
theorem cylinder_height_and_diameter (r_sphere : ℝ) (r_cylinder h_cylinder : ℝ)
  (h1 : r_sphere = 4)
  (h2 : 4 * π * r_sphere^2 = 2 * π * r_cylinder * h_cylinder)
  (h3 : h_cylinder = 2 * r_cylinder) :
  h_cylinder = 8 ∧ r_cylinder = 4 :=
by
  -- Proof to be completed
  sorry

end cylinder_height_and_diameter_l207_207814


namespace domain_of_function_l207_207036

theorem domain_of_function :
  {x : ℝ | 2 - x > 0 ∧ 1 + x > 0} = {x : ℝ | -1 < x ∧ x < 2} :=
by
  sorry

end domain_of_function_l207_207036


namespace radius_of_larger_circle_15_l207_207520

def radius_larger_circle (r1 r2 r3 r : ℝ) : Prop :=
  ∃ (A B C O : EuclideanSpace ℝ (Fin 2)), 
    dist A B = r1 + r2 ∧
    dist B C = r2 + r3 ∧
    dist A C = r1 + r3 ∧
    dist O A = r - r1 ∧
    dist O B = r - r2 ∧
    dist O C = r - r3 ∧
    (dist O A + r1 = r ∧
    dist O B + r2 = r ∧
    dist O C + r3 = r)

theorem radius_of_larger_circle_15 :
  radius_larger_circle 10 3 2 15 :=
by
  sorry

end radius_of_larger_circle_15_l207_207520


namespace chi_squared_test_expected_value_correct_l207_207629
open ProbabilityTheory

section Part1

def n : ℕ := 400
def a : ℕ := 60
def b : ℕ := 20
def c : ℕ := 180
def d : ℕ := 140
def alpha : ℝ := 0.005
def chi_critical : ℝ := 7.879

noncomputable def chi_squared : ℝ :=
  (n * (a * d - b * c) ^ 2) / ((a + b) * (c + d) * (a + c) * (b + d))

theorem chi_squared_test : chi_squared > chi_critical :=
  sorry

end Part1

section Part2

def reward_med : ℝ := 6  -- 60,000 yuan in 10,000 yuan unit
def reward_small : ℝ := 2  -- 20,000 yuan in 10,000 yuan unit
def total_support : ℕ := 12
def total_rewards : ℕ := 9

noncomputable def dist_table : List (ℝ × ℝ) :=
  [(180, 1 / 220),
   (220, 27 / 220),
   (260, 27 / 55),
   (300, 21 / 55)]

noncomputable def expected_value : ℝ :=
  dist_table.foldr (fun (xi : ℝ × ℝ) acc => acc + xi.1 * xi.2) 0

theorem expected_value_correct : expected_value = 270 :=
  sorry

end Part2

end chi_squared_test_expected_value_correct_l207_207629


namespace Hari_contribution_l207_207769

theorem Hari_contribution (H : ℕ) (Praveen_capital : ℕ := 3500) (months_Praveen : ℕ := 12) 
                          (months_Hari : ℕ := 7) (profit_ratio_P : ℕ := 2) (profit_ratio_H : ℕ := 3) : 
                          (Praveen_capital * months_Praveen) * profit_ratio_H = (H * months_Hari) * profit_ratio_P → 
                          H = 9000 :=
by
  sorry

end Hari_contribution_l207_207769


namespace birds_left_in_tree_l207_207687

-- Define the initial number of birds in the tree
def initialBirds : ℝ := 42.5

-- Define the number of birds that flew away
def birdsFlewAway : ℝ := 27.3

-- Theorem statement: Prove the number of birds left in the tree
theorem birds_left_in_tree : initialBirds - birdsFlewAway = 15.2 :=
by 
  sorry

end birds_left_in_tree_l207_207687


namespace parabola_transformation_zeros_sum_l207_207301

theorem parabola_transformation_zeros_sum :
  let y := fun x => (x - 3)^2 + 4
  let y_rotated := fun x => -(x - 3)^2 + 4
  let y_shifted_right := fun x => -(x - 7)^2 + 4
  let y_final := fun x => -(x - 7)^2 + 7
  ∃ a b, y_final a = 0 ∧ y_final b = 0 ∧ (a + b) = 14 :=
by
  sorry

end parabola_transformation_zeros_sum_l207_207301


namespace inequality_neg_reciprocal_l207_207330

theorem inequality_neg_reciprocal (a b : ℝ) (h1 : a < b) (h2 : b < 0) : 
  - (1 / a) < - (1 / b) :=
sorry

end inequality_neg_reciprocal_l207_207330


namespace num_ways_to_arrange_BANANA_l207_207548

theorem num_ways_to_arrange_BANANA : 
  let total_letters := 6
  let num_A := 3
  let num_N := 2
  let total_arrangements := Nat.factorial total_letters / (Nat.factorial num_A * Nat.factorial num_N)
  total_arrangements = 60 :=
by
  sorry

end num_ways_to_arrange_BANANA_l207_207548


namespace birds_joined_l207_207169

-- Definitions based on the identified conditions
def initial_birds : ℕ := 3
def initial_storks : ℕ := 2
def total_after_joining : ℕ := 10

-- Theorem statement that follows from the problem setup
theorem birds_joined :
  total_after_joining - (initial_birds + initial_storks) = 5 := by
  sorry

end birds_joined_l207_207169


namespace triangles_combined_area_is_96_l207_207004

noncomputable def combined_area_of_triangles : Prop :=
  let length_rectangle : ℝ := 6
  let width_rectangle : ℝ := 4
  let area_rectangle : ℝ := length_rectangle * width_rectangle
  let ratio_rectangle_to_first_triangle : ℝ := 2 / 5
  let area_first_triangle : ℝ := (5 / 2) * area_rectangle
  let x : ℝ := area_first_triangle / 5
  let base_second_triangle : ℝ := 8
  let height_second_triangle : ℝ := 9  -- calculated height based on the area ratio
  let area_second_triangle : ℝ := (base_second_triangle * height_second_triangle) / 2
  let combined_area : ℝ := area_first_triangle + area_second_triangle
  combined_area = 96

theorem triangles_combined_area_is_96 : combined_area_of_triangles := by
  sorry

end triangles_combined_area_is_96_l207_207004


namespace council_counts_l207_207484

theorem council_counts 
    (total_classes : ℕ := 20)
    (students_per_class : ℕ := 5)
    (total_students : ℕ := 100)
    (petya_class_council : ℕ × ℕ := (1, 4))  -- (boys, girls)
    (equal_boys_girls : 2 * 50 = total_students)  -- Equal number of boys and girls
    (more_girls_classes : ℕ := 15)
    (min_girls_each : ℕ := 3)
    (remaining_classes : ℕ := 4)
    (remaining_students : ℕ := 20)
    : (19, 1) = (19, 1) :=
by
    -- actual proof goes here
    sorry

end council_counts_l207_207484


namespace surface_area_ratio_l207_207103

-- Definitions based on conditions
def side_length (s : ℝ) := s > 0
def A_cube (s : ℝ) := 6 * s ^ 2
def A_rect (s : ℝ) := 2 * (2 * s) * (3 * s) + 2 * (2 * s) * (4 * s) + 2 * (3 * s) * (4 * s)

-- Theorem statement proving the ratio
theorem surface_area_ratio (s : ℝ) (h : side_length s) : A_cube s / A_rect s = 3 / 26 :=
by
  sorry

end surface_area_ratio_l207_207103


namespace truck_driver_gas_l207_207779

variables (miles_per_gallon distance_to_station gallons_to_add gallons_in_tank total_gallons_needed : ℕ)
variables (current_gas_in_tank : ℕ)
variables (h1 : miles_per_gallon = 3)
variables (h2 : distance_to_station = 90)
variables (h3 : gallons_to_add = 18)

theorem truck_driver_gas :
  current_gas_in_tank = 12 :=
by
  -- Prove that the truck driver already has 12 gallons of gas in his tank,
  -- given the conditions provided.
  sorry

end truck_driver_gas_l207_207779


namespace travel_time_without_paddles_l207_207466

variables (A B : Type) (v v_r S : ℝ)
noncomputable def time_to_travel (distance velocity : ℝ) := distance / velocity

-- Condition: The travel time from A to B is 3 times the travel time from B to A
axiom travel_condition : (time_to_travel S (v + v_r)) = 3 * (time_to_travel S (v - v_r))

-- Condition: We are considering travel from B to A by canoe without paddles
noncomputable def time_without_paddles := time_to_travel S v_r

-- Proving that without paddles it takes 3 times longer than usual (using canoes with paddles)
theorem travel_time_without_paddles :
  time_without_paddles S v_r = 3 * (time_to_travel S (v - v_r)) :=
sorry

end travel_time_without_paddles_l207_207466


namespace evaluate_expression_l207_207540

variable (m n p q s : ℝ)

theorem evaluate_expression :
  m / (n - (p + q * s)) = m / (n - p - q * s) :=
by
  sorry

end evaluate_expression_l207_207540


namespace length_of_CD_l207_207324

theorem length_of_CD (C D R S : ℝ) 
  (h1 : R = C + 3/8 * (D - C))
  (h2 : S = C + 4/11 * (D - C))
  (h3 : |S - R| = 3) :
  D - C = 264 := 
sorry

end length_of_CD_l207_207324


namespace hyperbola_asymptotes_correct_l207_207950

noncomputable def asymptotes_for_hyperbola : Prop :=
  ∀ (x y : ℂ),
    9 * (x : ℂ) ^ 2 - 4 * (y : ℂ) ^ 2 = -36 → 
    (y = (3 / 2) * (-Complex.I) * x) ∨ (y = -(3 / 2) * (-Complex.I) * x)

theorem hyperbola_asymptotes_correct :
  asymptotes_for_hyperbola := 
sorry

end hyperbola_asymptotes_correct_l207_207950


namespace anna_apple_ratio_l207_207658

-- Definitions based on conditions
def tuesday_apples : ℕ := 4
def wednesday_apples : ℕ := 2 * tuesday_apples
def total_apples : ℕ := 14

-- Theorem statement
theorem anna_apple_ratio :
  ∃ thursday_apples : ℕ, 
  thursday_apples = total_apples - (tuesday_apples + wednesday_apples) ∧
  (thursday_apples : ℚ) / tuesday_apples = 1 / 2 :=
by
  sorry

end anna_apple_ratio_l207_207658


namespace monotone_intervals_range_of_t_for_three_roots_l207_207671

def f (t x : ℝ) : ℝ := x^3 - 2 * x^2 + x + t

def f_prime (x : ℝ) : ℝ := 3 * x^2 - 4 * x + 1

-- 1. Monotonic intervals
theorem monotone_intervals (t : ℝ) :
  (∀ x, f_prime x > 0 → x < 1/3 ∨ x > 1) ∧
  (∀ x, f_prime x < 0 → 1/3 < x ∧ x < 1) :=
sorry

-- 2. Range of t for three real roots
theorem range_of_t_for_three_roots (t : ℝ) :
  (∃ a b : ℝ, f t a = 0 ∧ f t b = 0 ∧ a ≠ b ∧
   a = 1/3 ∧ b = 1 ∧
   -4/27 + t > 0 ∧ t < 0) :=
sorry

end monotone_intervals_range_of_t_for_three_roots_l207_207671


namespace avg_age_of_children_l207_207025

theorem avg_age_of_children 
  (participants : ℕ) (women : ℕ) (men : ℕ) (children : ℕ)
  (overall_avg_age : ℕ) (avg_age_women : ℕ) (avg_age_men : ℕ)
  (hp : participants = 50) (hw : women = 22) (hm : men = 18) (hc : children = 10)
  (ho : overall_avg_age = 20) (haw : avg_age_women = 24) (ham : avg_age_men = 19) :
  ∃ (avg_age_children : ℕ), avg_age_children = 13 :=
by
  -- Proof will be here.
  sorry

end avg_age_of_children_l207_207025


namespace race_time_A_l207_207987

noncomputable def time_for_A_to_cover_distance (distance : ℝ) (time_of_B : ℝ) (remaining_distance_for_B : ℝ) : ℝ :=
  let speed_of_B := distance / time_of_B
  let time_for_B_to_cover_remaining := remaining_distance_for_B / speed_of_B
  time_for_B_to_cover_remaining

theorem race_time_A (distance : ℝ) (time_of_B : ℝ) (remaining_distance_for_B : ℝ) :
  distance = 100 ∧ time_of_B = 25 ∧ remaining_distance_for_B = distance - 20 →
  time_for_A_to_cover_distance distance time_of_B remaining_distance_for_B = 20 :=
by
  intros h
  rcases h with ⟨h_distance, h_time_of_B, h_remaining_distance_for_B⟩
  rw [h_distance, h_time_of_B, h_remaining_distance_for_B]
  sorry

end race_time_A_l207_207987


namespace second_increase_is_40_l207_207962

variable (P : ℝ) (x : ℝ)

def second_increase (P : ℝ) (x : ℝ) : Prop :=
  1.30 * P * (1 + x / 100) = 1.82 * P

theorem second_increase_is_40 (P : ℝ) : ∃ x, second_increase P x ∧ x = 40 := by
  use 40
  sorry

end second_increase_is_40_l207_207962


namespace find_pos_ints_a_b_c_p_l207_207280

theorem find_pos_ints_a_b_c_p (a b c p : ℕ) (hp : Nat.Prime p) : 
  73 * p^2 + 6 = 9 * a^2 + 17 * b^2 + 17 * c^2 ↔
  (p = 2 ∧ a = 1 ∧ b = 4 ∧ c = 1) ∨ (p = 2 ∧ a = 1 ∧ b = 1 ∧ c = 4) :=
by
  sorry

end find_pos_ints_a_b_c_p_l207_207280


namespace possible_values_of_b_l207_207164

theorem possible_values_of_b (b : ℝ) : (¬ ∃ x : ℝ, x^2 + b * x + 1 ≤ 0) → -2 < b ∧ b < 2 :=
by
  intro h
  sorry

end possible_values_of_b_l207_207164


namespace circular_seating_count_l207_207659

theorem circular_seating_count :
  let D := 5 -- Number of Democrats
  let R := 5 -- Number of Republicans
  let total_politicians := D + R -- Total number of politicians
  let linear_arrangements := Nat.factorial total_politicians -- Total linear arrangements
  let unique_circular_arrangements := linear_arrangements / total_politicians -- Adjusting for circular rotations
  unique_circular_arrangements = 362880 :=
by
  sorry

end circular_seating_count_l207_207659


namespace chord_cos_theta_condition_l207_207470

open Real

-- Translation of the given conditions and proof problem
theorem chord_cos_theta_condition
  (a b x y θ : ℝ)
  (h1 : a^2 = b^2 + 2) :
  x * y = cos θ := 
sorry

end chord_cos_theta_condition_l207_207470


namespace debate_team_boys_l207_207021

theorem debate_team_boys (total_groups : ℕ) (members_per_group : ℕ) (num_girls : ℕ) (total_members : ℕ) :
  total_groups = 8 →
  members_per_group = 4 →
  num_girls = 4 →
  total_members = total_groups * members_per_group →
  total_members - num_girls = 28 :=
by
  sorry

end debate_team_boys_l207_207021


namespace find_b_l207_207265

theorem find_b (a b : ℝ) (k : ℝ) (h1 : a * b = k) (h2 : a + b = 40) (h3 : a - 2 * b = 10) (ha : a = 4) : b = 75 :=
  sorry

end find_b_l207_207265


namespace factors_multiple_of_120_l207_207993

theorem factors_multiple_of_120 (n : ℕ) (h : n = 2^12 * 3^15 * 5^9 * 7^5) :
  ∃ k : ℕ, k = 8100 ∧ ∀ d : ℕ, d ∣ n ∧ 120 ∣ d ↔ ∃ a b c d : ℕ, 3 ≤ a ∧ a ≤ 12 ∧ 1 ≤ b ∧ b ≤ 15 ∧ 1 ≤ c ∧ c ≤ 9 ∧ 0 ≤ d ∧ d ≤ 5 ∧ d = 2^a * 3^b * 5^c * 7^d :=
by
  sorry

end factors_multiple_of_120_l207_207993


namespace general_formula_no_arithmetic_sequence_l207_207146

-- Given condition
def Sn (a : ℕ → ℤ) (n : ℕ) : ℤ := 2 * a n - 3 * n

-- Theorem 1: General formula for the sequence a_n
theorem general_formula (a : ℕ → ℤ) (n : ℕ) (h : ∀ n, Sn a n = 2 * a n - 3 * n) : 
  a n = 3 * 2^n - 3 :=
sorry

-- Theorem 2: No three terms of the sequence form an arithmetic sequence
theorem no_arithmetic_sequence (a : ℕ → ℤ) (x y z : ℕ) (h : ∀ n, Sn a n = 2 * a n - 3 * n) (hx : x < y) (hy : y < z) :
  ¬ (a x + a z = 2 * a y) :=
sorry

end general_formula_no_arithmetic_sequence_l207_207146


namespace investment_difference_l207_207948

noncomputable def future_value_semi_annual (principal : ℝ) (annual_rate : ℝ) (years : ℝ) : ℝ :=
  principal * (1 + annual_rate / 2)^((years * 2))

noncomputable def future_value_monthly (principal : ℝ) (annual_rate : ℝ) (years : ℝ) : ℝ :=
  principal * (1 + annual_rate / 12)^((years * 12))

theorem investment_difference :
  let jose_investment := future_value_semi_annual 30000 0.03 3
  let patricia_investment := future_value_monthly 30000 0.025 3
  round (jose_investment) - round (patricia_investment) = 317 :=
by
  sorry

end investment_difference_l207_207948


namespace best_chart_for_temperature_changes_l207_207502

def Pie_chart := "Represent the percentage of parts in the whole."
def Line_chart := "Represent changes over time."
def Bar_chart := "Show the specific number of each item."

theorem best_chart_for_temperature_changes : 
  "The best statistical chart to use for understanding temperature changes throughout a day" = Line_chart :=
by
  sorry

end best_chart_for_temperature_changes_l207_207502


namespace integer_between_squares_l207_207506

theorem integer_between_squares (a b c d: ℝ) (h₀: 0 < a) (h₁: 0 < b) (h₂: 0 < c) (h₃: 0 < d) (h₄: c * d = 1) : 
  ∃ n : ℤ, ab ≤ n^2 ∧ n^2 ≤ (a + c) * (b + d) := 
by 
  sorry

end integer_between_squares_l207_207506


namespace minimum_distinct_values_is_145_l207_207593

-- Define the conditions
def n_series : ℕ := 2023
def unique_mode_occurrence : ℕ := 15

-- Define the minimum number of distinct values satisfying the conditions
def min_distinct_values (n : ℕ) (mode_count : ℕ) : ℕ :=
  if mode_count < n then 
    (n - mode_count + 13) / 14 + 1
  else
    1

-- The theorem restating the problem to be solved
theorem minimum_distinct_values_is_145 : 
  min_distinct_values n_series unique_mode_occurrence = 145 :=
by
  sorry

end minimum_distinct_values_is_145_l207_207593


namespace find_x_when_perpendicular_l207_207514

def a : ℝ × ℝ := (1, -2)
def b (x: ℝ) : ℝ × ℝ := (x, 1)
def are_perpendicular (a b: ℝ × ℝ) : Prop := a.1 * b.1 + a.2 * b.2 = 0

theorem find_x_when_perpendicular (x: ℝ) (h: are_perpendicular a (b x)) : x = 2 :=
by
  sorry

end find_x_when_perpendicular_l207_207514


namespace price_reduction_l207_207562

theorem price_reduction (original_price final_price : ℝ) (x : ℝ) 
  (h : original_price = 289) (h2 : final_price = 256) :
  289 * (1 - x) ^ 2 = 256 := sorry

end price_reduction_l207_207562


namespace simplify_expression_l207_207754

theorem simplify_expression : 0.4 * 0.5 + 0.3 * 0.2 = 0.26 := by
  sorry

end simplify_expression_l207_207754


namespace vector_parallel_find_k_l207_207897

theorem vector_parallel_find_k (k : ℝ) (a : ℝ × ℝ) (b : ℝ × ℝ) 
  (h₁ : a = (3 * k + 1, 2)) 
  (h₂ : b = (k, 1)) 
  (h₃ : ∃ c : ℝ, a = c • b) : k = -1 := 
by 
  sorry

end vector_parallel_find_k_l207_207897


namespace bruce_mango_purchase_l207_207772

theorem bruce_mango_purchase (m : ℕ) 
  (cost_grapes : 8 * 70 = 560)
  (cost_total : 560 + 55 * m = 1110) : 
  m = 10 :=
by
  sorry

end bruce_mango_purchase_l207_207772


namespace product_positions_8_2_100_100_l207_207104

def num_at_position : ℕ → ℕ → ℤ
| 0, _ => 0
| n, k => 
  let remainder := k % 3
  if remainder = 1 then 1 
  else if remainder = 2 then 2
  else -3

theorem product_positions_8_2_100_100 : 
  num_at_position 8 2 * num_at_position 100 100 = -3 :=
by
  unfold num_at_position
  -- unfold necessary definition steps
  sorry

end product_positions_8_2_100_100_l207_207104


namespace trig_expression_l207_207579

theorem trig_expression (α : ℝ) (h : Real.tan α = 2) : 
  (Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) + Real.cos α ^ 2 = 16 / 5 := 
  sorry

end trig_expression_l207_207579


namespace johny_journey_distance_l207_207319

def south_distance : ℕ := 40
def east_distance : ℕ := south_distance + 20
def north_distance : ℕ := 2 * east_distance
def total_distance : ℕ := south_distance + east_distance + north_distance

theorem johny_journey_distance :
  total_distance = 220 := by
  sorry

end johny_journey_distance_l207_207319


namespace factor_expression_l207_207891

theorem factor_expression :
  (8 * x^6 + 36 * x^4 - 5) - (2 * x^6 - 6 * x^4 + 5) = 2 * (3 * x^6 + 21 * x^4 - 5) :=
by
  sorry

end factor_expression_l207_207891


namespace area_OMVK_l207_207976

theorem area_OMVK :
  ∀ (S_OKCL S_ONAM S_ONBM S_ABCD S_OMVK : ℝ),
    S_OKCL = 6 →
    S_ONAM = 12 →
    S_ONBM = 24 →
    S_ABCD = 4 * (S_OKCL + S_ONAM) →
    S_OMVK = S_ABCD - S_OKCL - S_ONAM - S_ONBM →
    S_OMVK = 30 :=
by
  intros S_OKCL S_ONAM S_ONBM S_ABCD S_OMVK h_OKCL h_ONAM h_ONBM h_ABCD h_OMVK
  rw [h_OKCL, h_ONAM, h_ONBM] at *
  sorry

end area_OMVK_l207_207976


namespace swimmers_meetings_in_15_minutes_l207_207129

noncomputable def swimmers_pass_each_other_count 
    (pool_length : ℕ) (rate_swimmer1 : ℕ) (rate_swimmer2 : ℕ) (time_minutes : ℕ) : ℕ :=
sorry -- Definition of the function to count passing times

theorem swimmers_meetings_in_15_minutes :
  swimmers_pass_each_other_count 120 4 3 15 = 23 :=
sorry -- The proof is not required as per instruction.

end swimmers_meetings_in_15_minutes_l207_207129


namespace dog_running_direction_undeterminable_l207_207855

/-- Given the conditions:
 1. A dog is tied to a tree with a nylon cord of length 10 feet.
 2. The dog runs from one side of the tree to the opposite side with the cord fully extended.
 3. The dog runs approximately 30 feet.
 Prove that it is not possible to determine the specific starting direction of the dog.
-/
theorem dog_running_direction_undeterminable (r : ℝ) (full_length : r = 10) (distance_ran : ℝ) (approx_distance : distance_ran = 30) : (
  ∀ (d : ℝ), d < 2 * π * r → ¬∃ θ : ℝ, 0 ≤ θ ∧ θ < 2 * π ∧ (distance_ran = r * θ)
  ) :=
by
  sorry

end dog_running_direction_undeterminable_l207_207855


namespace same_speed_is_4_l207_207804

namespace SpeedProof

theorem same_speed_is_4 (x : ℝ) (h_jack_speed : x^2 - 11 * x - 22 = x - 10) (h_jill_speed : x^2 - 5 * x - 60 = (x - 10) * (x + 6)) :
  x = 14 → (x - 10) = 4 :=
by
  sorry

end SpeedProof

end same_speed_is_4_l207_207804


namespace multiply_polynomials_l207_207657

theorem multiply_polynomials (x : ℝ) : (x^4 + 50 * x^2 + 625) * (x^2 - 25) = x^6 - 15625 := by
  sorry

end multiply_polynomials_l207_207657


namespace gas_cost_per_gallon_l207_207734

theorem gas_cost_per_gallon (mpg : ℝ) (miles_per_day : ℝ) (days : ℝ) (total_cost : ℝ) : 
  mpg = 50 ∧ miles_per_day = 75 ∧ days = 10 ∧ total_cost = 45 → 
  (total_cost / ((miles_per_day * days) / mpg)) = 3 :=
by
  sorry

end gas_cost_per_gallon_l207_207734


namespace general_term_formula_l207_207195

variable (a : ℕ → ℤ) (S : ℕ → ℤ)
variable (n : ℕ)
variable (a1 d : ℤ)

-- Given conditions
axiom a2_eq : a 2 = 8
axiom S10_eq : S 10 = 185
axiom S_def : ∀ n, S n = n * (a 1 + a n) / 2
axiom a_def : ∀ n, a (n + 1) = a 1 + n * d

-- Prove the general term formula
theorem general_term_formula : a n = 3 * n + 2 := sorry

end general_term_formula_l207_207195


namespace tomatoes_sold_to_mr_wilson_l207_207403

theorem tomatoes_sold_to_mr_wilson :
  let T := 245.5
  let S_m := 125.5
  let N := 42
  let S_w := T - S_m - N
  S_w = 78 := 
by
  sorry

end tomatoes_sold_to_mr_wilson_l207_207403


namespace sophia_fraction_of_book_finished_l207_207079

variable (x : ℕ)

theorem sophia_fraction_of_book_finished (h1 : x + (x + 90) = 270) : (x + 90) / 270 = 2 / 3 := by
  sorry

end sophia_fraction_of_book_finished_l207_207079


namespace problem_proof_l207_207101

def f (a x : ℝ) := |a - x|

theorem problem_proof (a x x0 : ℝ) (h_a : a = 3 / 2) (h_x0 : x0 < 0) : 
  f a (x0 * x) ≥ x0 * f a x + f a (a * x0) :=
sorry

end problem_proof_l207_207101


namespace find_a_for_min_l207_207154

noncomputable def f (a x : ℝ) : ℝ := x^3 + 3 * a * x^2 - 6 * a * x + 2

theorem find_a_for_min {a x0 : ℝ} (hx0 : 1 < x0 ∧ x0 < 3) (h : ∀ x : ℝ, deriv (f a) x0 = 0) : a = -2 :=
by
  sorry

end find_a_for_min_l207_207154


namespace cake_pieces_kept_l207_207210

theorem cake_pieces_kept (total_pieces : ℕ) (two_fifths_eaten : ℕ) (extra_pieces_eaten : ℕ)
  (h1 : total_pieces = 35)
  (h2 : two_fifths_eaten = 2 * total_pieces / 5)
  (h3 : extra_pieces_eaten = 3)
  (correct_answer : ℕ)
  (h4 : correct_answer = total_pieces - (two_fifths_eaten + extra_pieces_eaten)) :
  correct_answer = 18 := by
  sorry

end cake_pieces_kept_l207_207210


namespace sector_area_angle_1_sector_max_area_l207_207312

-- The definition and conditions
variable (c : ℝ) (r l : ℝ)

-- 1) Proof that the area of the sector when the central angle is 1 radian is c^2 / 18
-- given 2r + l = c
theorem sector_area_angle_1 (h : 2 * r + l = c) (h1: l = r) :
  (1/2 * l * r = (c^2 / 18)) :=
by sorry

-- 2) Proof that the central angle that maximizes the area is 2 radians and the maximum area is c^2 / 16
-- given 2r + l = c
theorem sector_max_area (h : 2 * r + l = c) :
  ∃ l r, 2 * r = l ∧ 1/2 * l * r = (c^2 / 16) :=
by sorry

end sector_area_angle_1_sector_max_area_l207_207312


namespace percentage_decrease_hours_with_assistant_l207_207123

theorem percentage_decrease_hours_with_assistant :
  ∀ (B H H_new : ℝ), H_new = 0.9 * H → (H - H_new) / H * 100 = 10 :=
by
  intros B H H_new h_new_def
  sorry

end percentage_decrease_hours_with_assistant_l207_207123


namespace ellipse_solution_length_AB_l207_207839

noncomputable def ellipse_equation (a b : ℝ) (e : ℝ) (minor_axis : ℝ) : Prop :=
  a > b ∧ b > 0 ∧ e = 3 / 4 ∧ 2 * b = minor_axis ∧ minor_axis = 2 * Real.sqrt 7

theorem ellipse_solution (a b : ℝ) (e : ℝ) (minor_axis : ℝ) :
  ellipse_equation a b e minor_axis →
  (a^2 = 16 ∧ b^2 = 7 ∧ (1 / a^2) = 1 / 16 ∧ (1 / b^2) = 1 / 7) :=
by 
  intros h
  sorry

noncomputable def area_ratio (S1 S2 : ℝ) : Prop :=
  S1 / S2 = 9 / 13

theorem length_AB (S1 S2 : ℝ) :
  area_ratio S1 S2 →
  |S1 / S2| = |(9 * Real.sqrt 105) / 26| :=
by
  intros h
  sorry

end ellipse_solution_length_AB_l207_207839


namespace sixty_percent_of_fifty_minus_forty_percent_of_thirty_l207_207276

theorem sixty_percent_of_fifty_minus_forty_percent_of_thirty : 
  (0.6 * 50) - (0.4 * 30) = 18 :=
by
  sorry

end sixty_percent_of_fifty_minus_forty_percent_of_thirty_l207_207276


namespace susie_rooms_l207_207771

-- Define the conditions
def vacuum_time_per_room : ℕ := 20  -- in minutes
def total_vacuum_time : ℕ := 2 * 60  -- 2 hours in minutes

-- Define the number of rooms in Susie's house
def number_of_rooms (total_time room_time : ℕ) : ℕ := total_time / room_time

-- Prove that Susie has 6 rooms in her house
theorem susie_rooms : number_of_rooms total_vacuum_time vacuum_time_per_room = 6 :=
by
  sorry -- proof goes here

end susie_rooms_l207_207771


namespace expression_value_l207_207096

theorem expression_value (x : ℝ) (h : x = -2) : (x * x^2 * (1/x) = 4) :=
by
  rw [h]
  sorry

end expression_value_l207_207096


namespace train_capacity_l207_207751

theorem train_capacity (T : ℝ) (h : 2 * (T / 6) = 40) : T = 120 :=
sorry

end train_capacity_l207_207751


namespace pebble_difference_l207_207626

-- Definitions and conditions
variables (x : ℚ) -- we use rational numbers for exact division
def Candy := 2 * x
def Lance := 5 * x
def Sandy := 4 * x
def condition1 := Lance = Candy + 10

-- Theorem statement
theorem pebble_difference (h : condition1) : Lance + Sandy - Candy = 30 :=
sorry

end pebble_difference_l207_207626


namespace greatest_power_of_2_factor_l207_207877

theorem greatest_power_of_2_factor
    : ∃ k : ℕ, (2^k) ∣ (10^1503 - 4^752) ∧ ∀ m : ℕ, (2^(m+1)) ∣ (10^1503 - 4^752) → m < k :=
by
    sorry

end greatest_power_of_2_factor_l207_207877


namespace remainder_when_concat_numbers_1_to_54_div_55_l207_207723

def concat_numbers (n : ℕ) : ℕ :=
  let digits x := x.digits 10
  (List.range n).bind digits |> List.reverse |> List.foldl (λ acc x => acc * 10 + x) 0

theorem remainder_when_concat_numbers_1_to_54_div_55 :
  let M := concat_numbers 55
  M % 55 = 44 :=
by
  sorry

end remainder_when_concat_numbers_1_to_54_div_55_l207_207723


namespace rectangle_area_in_cm_l207_207238

theorem rectangle_area_in_cm (length_in_m : ℝ) (width_in_m : ℝ) 
  (h_length : length_in_m = 0.5) (h_width : width_in_m = 0.36) : 
  (100 * length_in_m) * (100 * width_in_m) = 1800 :=
by
  -- We skip the proof for now
  sorry

end rectangle_area_in_cm_l207_207238


namespace vertices_of_cube_l207_207613

-- Given condition: geometric shape is a cube
def is_cube (x : Type) : Prop := true -- This is a placeholder declaration that x is a cube.

-- Question: How many vertices does a cube have?
-- Proof problem: Prove that the number of vertices of a cube is 8.
theorem vertices_of_cube (x : Type) (h : is_cube x) : true := 
  sorry

end vertices_of_cube_l207_207613


namespace number_of_square_free_odds_l207_207907

def is_square_free (n : ℕ) : Prop :=
  ∀ m : ℕ, m > 1 → m * m ∣ n → false

theorem number_of_square_free_odds (n : ℕ) (h1 : 1 < n) (h2 : n < 200) (h3 : n % 2 = 1) :
  (is_square_free n) ↔ (n = 79) := by
  sorry

end number_of_square_free_odds_l207_207907


namespace calc_expression_l207_207574

theorem calc_expression : 
  |1 - Real.sqrt 2| - Real.sqrt 8 + (Real.sqrt 2 - 1)^0 = -Real.sqrt 2 :=
by
  sorry

end calc_expression_l207_207574


namespace fermat_little_theorem_l207_207938

theorem fermat_little_theorem (p : ℕ) (hp : Nat.Prime p) (a : ℕ) : a^p ≡ a [MOD p] :=
sorry

end fermat_little_theorem_l207_207938


namespace BaSO4_molecular_weight_l207_207921

noncomputable def Ba : ℝ := 137.327
noncomputable def S : ℝ := 32.065
noncomputable def O : ℝ := 15.999
noncomputable def BaSO4 : ℝ := Ba + S + 4 * O

theorem BaSO4_molecular_weight : BaSO4 = 233.388 := by
  sorry

end BaSO4_molecular_weight_l207_207921


namespace evaluate_expression_121point5_l207_207750

theorem evaluate_expression_121point5 :
  let x := (2 / 3 : ℝ)
  let y := (9 / 2 : ℝ)
  (1 / 3) * x^4 * y^5 = 121.5 :=
by
  let x := (2 / 3 : ℝ)
  let y := (9 / 2 : ℝ)
  sorry

end evaluate_expression_121point5_l207_207750


namespace num_sol_and_sum_sol_l207_207213

-- Definition of the main problem condition
def equation (x : ℝ) := (4 * x^2 - 9)^2 = 49

-- Proof problem statement
theorem num_sol_and_sum_sol :
  (∃ s : Finset ℝ, (∀ x, equation x ↔ x ∈ s) ∧ s.card = 4 ∧ s.sum id = 0) :=
sorry

end num_sol_and_sum_sol_l207_207213


namespace cost_of_3600_pens_l207_207347

theorem cost_of_3600_pens
  (pack_size : ℕ)
  (pack_cost : ℝ)
  (n_pens : ℕ)
  (pen_cost : ℝ)
  (total_cost : ℝ)
  (h1: pack_size = 150)
  (h2: pack_cost = 45)
  (h3: n_pens = 3600)
  (h4: pen_cost = pack_cost / pack_size)
  (h5: total_cost = n_pens * pen_cost) :
  total_cost = 1080 :=
sorry

end cost_of_3600_pens_l207_207347


namespace inverse_of_square_l207_207124

theorem inverse_of_square (A : Matrix (Fin 2) (Fin 2) ℝ) (hA_inv : A⁻¹ = ![![3, 4], ![-2, -2]]) :
  (A^2)⁻¹ = ![![1, 4], ![-2, -4]] :=
by
  sorry

end inverse_of_square_l207_207124


namespace volume_of_convex_polyhedron_l207_207029

variables {S1 S2 S : ℝ} {h : ℝ}

theorem volume_of_convex_polyhedron (S1 S2 S h : ℝ) :
  (h > 0) → (S1 ≥ 0) → (S2 ≥ 0) → (S ≥ 0) →
  ∃ V, V = (h / 6) * (S1 + S2 + 4 * S) :=
by {
  sorry
}

end volume_of_convex_polyhedron_l207_207029


namespace ladder_distance_from_wall_l207_207660

noncomputable def dist_from_wall (ladder_length : ℝ) (angle_deg : ℝ) : ℝ :=
  ladder_length * Real.cos (angle_deg * Real.pi / 180)

theorem ladder_distance_from_wall :
  ∀ (ladder_length : ℝ) (angle_deg : ℝ), ladder_length = 19 → angle_deg = 60 → dist_from_wall ladder_length angle_deg = 9.5 :=
by
  intros ladder_length angle_deg h1 h2
  sorry

end ladder_distance_from_wall_l207_207660


namespace intersection_eq_T_l207_207307

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_eq_T : S ∩ T = T := sorry

end intersection_eq_T_l207_207307


namespace max_knights_on_island_l207_207733

theorem max_knights_on_island :
  ∃ n x, (n * (n - 1) = 90) ∧ (x * (10 - x) = 24) ∧ (x ≤ n) ∧ (∀ y, y * (10 - y) = 24 → y ≤ x) := sorry

end max_knights_on_island_l207_207733


namespace journey_time_l207_207069

noncomputable def velocity_of_stream : ℝ := 4
noncomputable def speed_of_boat_in_still_water : ℝ := 14
noncomputable def distance_A_to_B : ℝ := 180
noncomputable def distance_B_to_C : ℝ := distance_A_to_B / 2
noncomputable def downstream_speed : ℝ := speed_of_boat_in_still_water + velocity_of_stream
noncomputable def upstream_speed : ℝ := speed_of_boat_in_still_water - velocity_of_stream

theorem journey_time : (distance_A_to_B / downstream_speed) + (distance_B_to_C / upstream_speed) = 19 := by
  sorry

end journey_time_l207_207069


namespace basketball_score_l207_207088

theorem basketball_score (score_game1 : ℕ) (score_game2 : ℕ) (score_game3 : ℕ) (score_game4 : ℕ) (score_total_games8 : ℕ) (score_total_games9 : ℕ) :
  score_game1 = 18 ∧ score_game2 = 22 ∧ score_game3 = 15 ∧ score_game4 = 20 ∧ 
  (score_game1 + score_game2 + score_game3 + score_game4) / 4 < score_total_games8 / 8 ∧ 
  score_total_games9 / 9 > 19 →
  score_total_games9 - score_total_games8 ≥ 21 :=
by
-- proof steps would be provided here based on the given solution
sorry

end basketball_score_l207_207088


namespace interview_passing_probability_l207_207888

def probability_of_passing_interview (p : ℝ) : ℝ :=
  p + (1 - p) * p + (1 - p) * (1 - p) * p

theorem interview_passing_probability : probability_of_passing_interview 0.7 = 0.973 :=
by
  -- proof steps to be filled
  sorry

end interview_passing_probability_l207_207888


namespace C_recurrence_S_recurrence_l207_207965

noncomputable def C (x : ℝ) : ℝ := 2 * Real.cos x
noncomputable def C_n (n : ℕ) (x : ℝ) : ℝ := 2 * Real.cos (n * x)
noncomputable def S_n (n : ℕ) (x : ℝ) : ℝ := Real.sin (n * x) / Real.sin x

theorem C_recurrence (n : ℕ) (x : ℝ) (hx : x ≠ 0) :
  C_n n x = C x * C_n (n - 1) x - C_n (n - 2) x := sorry

theorem S_recurrence (n : ℕ) (x : ℝ) (hx : x ≠ 0) :
  S_n n x = C x * S_n (n - 1) x - S_n (n - 2) x := sorry

end C_recurrence_S_recurrence_l207_207965


namespace sufficient_not_necessary_condition_l207_207092

theorem sufficient_not_necessary_condition (x y : ℝ) (hx : x > 0) (hy : y > 0) : 
  (x > 0 ∧ y > 0) → (x > 0 ∧ y > 0 ↔ (y/x + x/y ≥ 2)) :=
by sorry

end sufficient_not_necessary_condition_l207_207092


namespace possible_values_of_p_l207_207275

theorem possible_values_of_p (p : ℕ) (a b : ℕ) (h_fact : (x : ℤ) → x^2 - 5 * x + p = (x - a) * (x - b))
  (h1 : a + b = 5) (h2 : 1 ≤ a ∧ a ≤ 4) (h3 : 1 ≤ b ∧ b ≤ 4) : 
  p = 4 ∨ p = 6 :=
sorry

end possible_values_of_p_l207_207275


namespace athlete_groups_l207_207333

/-- A school has athletes divided into groups.
   - If there are 7 people per group, there will be 3 people left over.
   - If there are 8 people per group, there will be a shortage of 5 people.
The goal is to prove that the system of equations is valid --/
theorem athlete_groups (x y : ℕ) :
  7 * y = x - 3 ∧ 8 * y = x + 5 := 
by 
  sorry

end athlete_groups_l207_207333


namespace bank_transfer_amount_l207_207811

/-- Paul made two bank transfers. A service charge of 2% was added to each transaction.
The second transaction was reversed without the service charge. His account balance is now $307 if 
it was $400 before he made any transfers. Prove that the amount of the first bank transfer was 
$91.18. -/
theorem bank_transfer_amount (x : ℝ) (initial_balance final_balance : ℝ) (service_charge_rate : ℝ) 
  (second_transaction_reversed : Prop)
  (h_initial : initial_balance = 400)
  (h_final : final_balance = 307)
  (h_charge : service_charge_rate = 0.02)
  (h_reversal : second_transaction_reversed):
  initial_balance - (1 + service_charge_rate) * x = final_balance ↔
  x = 91.18 := 
by
  sorry

end bank_transfer_amount_l207_207811


namespace nancy_flooring_area_l207_207746

def area_of_rectangle (length : ℕ) (width : ℕ) : ℕ :=
  length * width

theorem nancy_flooring_area :
  let central_area_length := 10
  let central_area_width := 10
  let hallway_length := 6
  let hallway_width := 4
  let central_area := area_of_rectangle central_area_length central_area_width
  let hallway_area := area_of_rectangle hallway_length hallway_width
  let total_area := central_area + hallway_area
  total_area = 124 :=
by
  rfl  -- This is where the proof would go.

end nancy_flooring_area_l207_207746


namespace number_of_bricks_needed_l207_207988

theorem number_of_bricks_needed :
  ∀ (brick_length brick_width brick_height wall_length wall_height wall_width : ℝ),
  brick_length = 25 → 
  brick_width = 11.25 → 
  brick_height = 6 → 
  wall_length = 750 → 
  wall_height = 600 → 
  wall_width = 22.5 → 
  (wall_length * wall_height * wall_width) / (brick_length * brick_width * brick_height) = 6000 :=
by
  intros brick_length brick_width brick_height wall_length wall_height wall_width
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4, h5, h6]
  sorry

end number_of_bricks_needed_l207_207988


namespace ab_equality_l207_207304

theorem ab_equality (a b : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_div : 4 * a * b - 1 ∣ (4 * a ^ 2 - 1) ^ 2) : a = b := sorry

end ab_equality_l207_207304


namespace rhombus_diagonal_length_l207_207499

theorem rhombus_diagonal_length (d1 d2 : ℝ) (area : ℝ) 
(h_d2 : d2 = 18) (h_area : area = 126) (h_formula : area = (d1 * d2) / 2) : 
d1 = 14 :=
by
  -- We're skipping the proof steps.
  sorry

end rhombus_diagonal_length_l207_207499


namespace intersection_S_T_l207_207429

def S : Set ℝ := { x | (x - 2) * (x - 3) >= 0 }
def T : Set ℝ := { x | x > 0 }

theorem intersection_S_T :
  S ∩ T = { x | (0 < x ∧ x <= 2) ∨ (x >= 3) } := by
  sorry

end intersection_S_T_l207_207429


namespace new_person_weight_l207_207652

theorem new_person_weight (avg_increase : ℝ) (num_persons : ℕ) (old_weight : ℝ) : 
    avg_increase = 2.5 ∧ num_persons = 8 ∧ old_weight = 65 → 
    (old_weight + num_persons * avg_increase = 85) :=
by
  intro h
  sorry

end new_person_weight_l207_207652


namespace find_x_l207_207927

-- Define conditions
def simple_interest (x y : ℝ) : Prop :=
  x * y * 2 / 100 = 800

def compound_interest (x y : ℝ) : Prop :=
  x * ((1 + y / 100)^2 - 1) = 820

-- Prove x = 8000 given the conditions
theorem find_x (x y : ℝ) (h1 : simple_interest x y) (h2 : compound_interest x y) : x = 8000 :=
  sorry

end find_x_l207_207927


namespace tank_length_l207_207549

variable (rate : ℝ)
variable (time : ℝ)
variable (width : ℝ)
variable (depth : ℝ)
variable (volume : ℝ)
variable (length : ℝ)

-- Given conditions
axiom rate_cond : rate = 5 -- cubic feet per hour
axiom time_cond : time = 60 -- hours
axiom width_cond : width = 6 -- feet
axiom depth_cond : depth = 5 -- feet

-- Derived volume from the rate and time
axiom volume_cond : volume = rate * time

-- Definition of length from volume, width, and depth
axiom length_def : length = volume / (width * depth)

-- The proof problem to show
theorem tank_length : length = 10 := by
  -- conditions provided and we expect the length to be computed
  sorry

end tank_length_l207_207549


namespace primes_in_arithmetic_sequence_l207_207464

noncomputable def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem primes_in_arithmetic_sequence (p : ℕ) :
  is_prime p ∧ is_prime (p + 2) ∧ is_prime (p + 4) → p = 3 :=
by
  intro h
  sorry

end primes_in_arithmetic_sequence_l207_207464


namespace sum_of_reciprocals_factors_12_l207_207675

theorem sum_of_reciprocals_factors_12 : (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12) = 7/3 :=
by sorry

end sum_of_reciprocals_factors_12_l207_207675


namespace largest_divisor_of_expression_l207_207273

theorem largest_divisor_of_expression (n : ℤ) : ∃ k, ∀ n : ℤ, n^4 - n^2 = k * 12 :=
by sorry

end largest_divisor_of_expression_l207_207273


namespace triangle_sine_equality_l207_207946

theorem triangle_sine_equality {a b c : ℝ} {α β γ : ℝ} 
  (cos_rule : c^2 = a^2 + b^2 - 2 * a * b * Real.cos γ)
  (area : ∃ T : ℝ, T = (1 / 2) * a * b * Real.sin γ)
  (sin_addition_γ : Real.sin (γ + Real.pi / 6) = Real.sin γ * (Real.sqrt 3 / 2) + Real.cos γ * (1 / 2))
  (sin_addition_β : Real.sin (β + Real.pi / 6) = Real.sin β * (Real.sqrt 3 / 2) + Real.cos β * (1 / 2))
  (sin_addition_α : Real.sin (α + Real.pi / 6) = Real.sin α * (Real.sqrt 3 / 2) + Real.cos α * (1 / 2)) :
  c^2 + 2 * a * b * Real.sin (γ + Real.pi / 6) = b^2 + 2 * a * c * Real.sin (β + Real.pi / 6) ∧
  b^2 + 2 * a * c * Real.sin (β + Real.pi / 6) = a^2 + 2 * b * c * Real.sin (α + Real.pi / 6) :=
sorry

end triangle_sine_equality_l207_207946


namespace apples_distribution_l207_207424

variable (p b t : ℕ)

theorem apples_distribution (p_eq : p = 40) (b_eq : b = p + 8) (t_eq : t = (3 * b) / 8) :
  t = 18 := by
  sorry

end apples_distribution_l207_207424


namespace max_cube_sum_l207_207995

theorem max_cube_sum (x y z : ℝ) (h : x^2 + y^2 + z^2 = 9) : x^3 + y^3 + z^3 ≤ 27 :=
sorry

end max_cube_sum_l207_207995


namespace intersection_M_N_eq_l207_207361

open Set

theorem intersection_M_N_eq :
  let M := {x : ℝ | x - 2 > 0}
  let N := {y : ℝ | ∃ (x : ℝ), y = Real.sqrt (x^2 + 1)}
  M ∩ N = {x : ℝ | x > 2} :=
by
  sorry

end intersection_M_N_eq_l207_207361


namespace perpendicular_line_through_P_l207_207673

open Real

-- Define the point (1, 0)
def P : ℝ × ℝ := (1, 0)

-- Define the initial line x - 2y - 2 = 0
def initial_line (x y : ℝ) : Prop := x - 2 * y = 2

-- Define the desired line 2x + y - 2 = 0
def desired_line (x y : ℝ) : Prop := 2 * x + y = 2

-- State that the desired line passes through the point (1, 0) and is perpendicular to the initial line
theorem perpendicular_line_through_P :
  (∃ m b, b ∈ Set.univ ∧ (∀ x y, desired_line x y → y = m * x + b)) ∧ ∀ x y, 
  initial_line x y → x ≠ 0 → desired_line y (-x / 2) :=
sorry

end perpendicular_line_through_P_l207_207673


namespace license_plate_count_l207_207383

noncomputable def num_license_plates : Nat :=
  let num_digit_possibilities := 10
  let num_letter_possibilities := 26
  let num_letter_pairs := num_letter_possibilities * num_letter_possibilities
  let num_positions_for_block := 6
  num_positions_for_block * (num_digit_possibilities ^ 5) * num_letter_pairs

theorem license_plate_count :
  num_license_plates = 40560000 :=
by
  sorry

end license_plate_count_l207_207383


namespace problem1_problem2_l207_207992

-- Define the first problem: For positive real numbers a and b,
-- with the condition a + b = 2, show that the minimum value of 
-- (1 / (1 + a) + 4 / (1 + b)) is 9/4.
theorem problem1 (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_sum : a + b = 2) :
  1 / (1 + a) + 4 / (1 + b) ≥ 9 / 4 :=
sorry

-- Define the second problem: For any positive real numbers a and b,
-- prove that a^2 * b^2 + a^2 + b^2 ≥ a * b * (a + b + 1).
theorem problem2 (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) :
  a^2 * b^2 + a^2 + b^2 ≥ a * b * (a + b + 1) :=
sorry

end problem1_problem2_l207_207992


namespace sum_of_terms_l207_207357

variable (a : ℕ → ℕ)
variable (S : ℕ → ℕ)

theorem sum_of_terms (h : ∀ n, S n = n^2) : a 5 + a 6 + a 7 = 33 :=
by
  sorry

end sum_of_terms_l207_207357


namespace compare_f_values_l207_207842

noncomputable def f (x : Real) : Real := 
  Real.cos x + 2 * x * (1 / 2)  -- given f''(pi/6) = 1/2

theorem compare_f_values :
  f (-Real.pi / 3) < f (Real.pi / 3) :=
by
  sorry

end compare_f_values_l207_207842


namespace quadratic_roots_sum_product_l207_207507

theorem quadratic_roots_sum_product :
  ∀ (x1 x2 : ℝ), (x1^2 - 2 * x1 - 8 = 0 ∧ x2^2 - 2 * x2 - 8 = 0 ∧ x1 ≠ x2) →
    (x1 + x2) / (x1 * x2) = -1 / 4 :=
by
  sorry

end quadratic_roots_sum_product_l207_207507


namespace greatest_integer_difference_l207_207555

theorem greatest_integer_difference (x y : ℤ) (hx : 3 < x) (hx2 : x < 6) (hy : 6 < y) (hy2 : y < 10) :
  ∃ d : ℤ, d = y - x ∧ d = 5 :=
by
  sorry

end greatest_integer_difference_l207_207555


namespace find_counterfeit_80_coins_in_4_weighings_min_weighings_for_n_coins_l207_207978

theorem find_counterfeit_80_coins_in_4_weighings :
  ∃ f : Fin 80 → Bool, (∃ i, f i = true) ∧ (∃ i j, f i ≠ f j) := sorry

theorem min_weighings_for_n_coins (n : ℕ) (h : n > 0) :
  ∃ k : ℕ, 3^(k-1) < n ∧ n ≤ 3^k := sorry

end find_counterfeit_80_coins_in_4_weighings_min_weighings_for_n_coins_l207_207978


namespace number_of_discounted_tickets_l207_207611

def total_tickets : ℕ := 10
def full_price_ticket_cost : ℝ := 2.0
def discounted_ticket_cost : ℝ := 1.6
def total_spent : ℝ := 18.40

theorem number_of_discounted_tickets (F D : ℕ) : 
    F + D = total_tickets → 
    full_price_ticket_cost * ↑F + discounted_ticket_cost * ↑D = total_spent → 
    D = 4 :=
by
  intros h1 h2
  sorry

end number_of_discounted_tickets_l207_207611


namespace hyperbola_sufficiency_l207_207228

open Real

theorem hyperbola_sufficiency (k : ℝ) : 
  (9 - k < 0 ∧ k - 4 > 0) → 
  (∃ x y : ℝ, (x^2) / (9 - k) + (y^2) / (k - 4) = 1) :=
by
  intro hk
  sorry

end hyperbola_sufficiency_l207_207228


namespace transforming_sin_curve_l207_207332

theorem transforming_sin_curve :
  ∀ x : ℝ, (2 * Real.sin (x + (Real.pi / 3))) = (2 * Real.sin ((1/3) * x + (Real.pi / 3))) :=
by
  sorry

end transforming_sin_curve_l207_207332


namespace no_solution_m_4_l207_207109

theorem no_solution_m_4 (m : ℝ) : 
  (¬ ∃ x : ℝ, 2/x = m/(2*x + 1)) → m = 4 :=
by
  sorry

end no_solution_m_4_l207_207109


namespace length_of_segment_l207_207544

theorem length_of_segment (x : ℤ) (hx : |x - 3| = 4) : 
  let a := 7
  let b := -1
  a - b = 8 := by
    sorry

end length_of_segment_l207_207544


namespace penguin_seafood_protein_l207_207803

theorem penguin_seafood_protein
  (digest : ℝ) -- representing 30% 
  (digested : ℝ) -- representing 9 grams 
  (h : digest = 0.30) 
  (h1 : digested = 9) :
  ∃ x : ℝ, digested = digest * x ∧ x = 30 :=
by
  sorry

end penguin_seafood_protein_l207_207803


namespace remainder_3_pow_9_div_5_l207_207720

theorem remainder_3_pow_9_div_5 : (3^9) % 5 = 3 := by
  sorry

end remainder_3_pow_9_div_5_l207_207720


namespace triangle_ABC_proof_l207_207954

noncomputable def sin2C_eq_sqrt3sinC (C : ℝ) : Prop := Real.sin (2 * C) = Real.sqrt 3 * Real.sin C

theorem triangle_ABC_proof (C a b c : ℝ) 
  (H1 : sin2C_eq_sqrt3sinC C) 
  (H2 : 0 < Real.sin C)
  (H3 : b = 6) 
  (H4 : a + b + c = 6*Real.sqrt 3 + 6) :
  (C = π/6) ∧ (1/2 * a * b * Real.sin C = 6*Real.sqrt 3) :=
sorry

end triangle_ABC_proof_l207_207954


namespace semi_minor_axis_l207_207801

theorem semi_minor_axis (a c : ℝ) (h_a : a = 5) (h_c : c = 2) : 
  ∃ b : ℝ, b = Real.sqrt (a^2 - c^2) ∧ b = Real.sqrt 21 :=
by
  use Real.sqrt 21
  sorry

end semi_minor_axis_l207_207801


namespace min_value_of_expression_l207_207997

noncomputable def min_value_expression (a b c d : ℝ) : ℝ :=
  (a - 2)^2 + (b / a - 1)^2 + (c / b - 1)^2 + (d / c - 1)^2

theorem min_value_of_expression (a b c d : ℝ) (h1 : 2 ≤ a) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : c ≤ d) (h5 : d ≥ 2) :
  min_value_expression a b c d = 1 / 4 :=
sorry

end min_value_of_expression_l207_207997


namespace find_MT_square_l207_207054

-- Definitions and conditions
variables (P Q R S L O M N T U : Type*)
variables (x : ℝ)
variables (PL PQ PS QR RS LO : finset ℝ)
variable (side_length_PQRS : ℝ) (area_PLQ area_QMTL area_SNUL area_RNMUT : ℝ)
variables (LO_MT_perpendicular LO_NU_perpendicular : Prop)

-- Stating the problem
theorem find_MT_square :
  (side_length_PQRS = 3) →
  (PL ⊆ PQ) →
  (PO ⊆ PS) →
  (PL = PO) →
  (PL = x) →
  (U ∈ LO) →
  (T ∈ LO) →
  (LO_MT_perpendicular) →
  (LO_NU_perpendicular) →
  (area_PLQ = 1) →
  (area_QMTL = 1) →
  (area_SNUL = 2) →
  (area_RNMUT = 2) →
  (x^2 / 2 = 1) → 
  (PL * LO = 1) →
  MT^2 = 1 / 2 :=
sorry

end find_MT_square_l207_207054


namespace chipmunk_acorns_l207_207318

-- Define the conditions and goal for the proof
theorem chipmunk_acorns :
  ∃ x : ℕ, (∀ h_c h_s : ℕ, h_c = h_s + 4 → 3 * h_c = x ∧ 4 * h_s = x) → x = 48 :=
by {
  -- We assume the problem conditions as given
  sorry
}

end chipmunk_acorns_l207_207318


namespace total_water_capacity_l207_207831

-- Define the given conditions as constants
def numTrucks : ℕ := 5
def tanksPerTruck : ℕ := 4
def capacityPerTank : ℕ := 200

-- Define the claim as a theorem
theorem total_water_capacity :
  numTrucks * (tanksPerTruck * capacityPerTank) = 4000 :=
by
  sorry

end total_water_capacity_l207_207831


namespace expression_value_l207_207167

theorem expression_value (x y z : ℤ) (h1 : x = 25) (h2 : y = 30) (h3 : z = 7) :
  (x - (y - z)) - ((x - y) - (z - 1)) = 13 :=
by
  sorry

end expression_value_l207_207167


namespace faster_train_speed_correct_l207_207534

noncomputable def speed_of_faster_train (V_s_kmph : ℝ) (length_faster_train_m : ℝ) (time_s : ℝ) : ℝ :=
  let V_s_mps := V_s_kmph * (1000 / 3600)
  let V_r_mps := length_faster_train_m / time_s
  let V_f_mps := V_r_mps - V_s_mps
  V_f_mps * (3600 / 1000)

theorem faster_train_speed_correct : 
  speed_of_faster_train 36 90.0072 4 = 45.00648 := 
by
  sorry

end faster_train_speed_correct_l207_207534


namespace sequence_sum_l207_207086

open Nat

-- Define the sequence
def a : ℕ → ℕ
| 0     => 1
| (n+1) => a n + (n + 1)

-- Define the sum of reciprocals up to the 2016 term
def sum_reciprocals : ℕ → ℚ
| 0     => 1 / (a 0)
| (n+1) => sum_reciprocals n + 1 / (a (n+1))

-- Define the property we wish to prove
theorem sequence_sum :
  sum_reciprocals 2015 = 4032 / 2017 :=
sorry

end sequence_sum_l207_207086


namespace john_shower_duration_l207_207392

variable (days_per_week : ℕ := 7)
variable (weeks : ℕ := 4)
variable (total_days : ℕ := days_per_week * weeks)
variable (shower_frequency : ℕ := 2) -- every other day
variable (number_of_showers : ℕ := total_days / shower_frequency)
variable (total_gallons_used : ℕ := 280)
variable (gallons_per_shower : ℕ := total_gallons_used / number_of_showers)
variable (gallons_per_minute : ℕ := 2)

theorem john_shower_duration 
  (h_cond : total_gallons_used = number_of_showers * gallons_per_shower)
  (h_shower_eq : total_days / shower_frequency = number_of_showers)
  : gallons_per_shower / gallons_per_minute = 10 :=
by
  sorry

end john_shower_duration_l207_207392


namespace fixed_point_always_on_line_l207_207447

theorem fixed_point_always_on_line (a : ℝ) (h : a ≠ 0) :
  (a + 2) * 1 + (1 - a) * 1 - 3 = 0 :=
by
  sorry

end fixed_point_always_on_line_l207_207447


namespace cost_of_pen_l207_207377

-- define the conditions
def notebook_cost (pen_cost : ℝ) : ℝ := 3 * pen_cost
def total_cost (notebook_cost : ℝ) : ℝ := 4 * notebook_cost

-- theorem stating the problem we need to prove
theorem cost_of_pen (pen_cost : ℝ) (h1 : total_cost (notebook_cost pen_cost) = 18) : pen_cost = 1.5 :=
by
  -- proof to be constructed
  sorry

end cost_of_pen_l207_207377


namespace solve_system_of_equations_l207_207476

theorem solve_system_of_equations (x y : ℝ) (hx : x + y + Real.sqrt (x * y) = 28)
  (hy : x^2 + y^2 + x * y = 336) : (x = 4 ∧ y = 16) ∨ (x = 16 ∧ y = 4) :=
sorry

end solve_system_of_equations_l207_207476


namespace intersect_count_l207_207237

noncomputable def f (x : ℝ) : ℝ := sorry  -- Function f defined for all real x.
noncomputable def f_inv (y : ℝ) : ℝ := sorry  -- Inverse function of f.

theorem intersect_count : 
  (∃ a b : ℝ, a ≠ b ∧ f (a^2) = f (a^3) ∧ f (b^2) = f (b^3)) :=
by sorry

end intersect_count_l207_207237


namespace find_f_of_3_l207_207058

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ :=
  x^7 + a*x^5 + b*x - 5

theorem find_f_of_3 (a b : ℝ) (h : f (-3) a b = 5) : f 3 a b = -15 := by
  sorry

end find_f_of_3_l207_207058


namespace inequality_of_abc_l207_207621

theorem inequality_of_abc (a b c : ℝ) (hpos_a : 0 < a) (hpos_b : 0 < b) (hpos_c : 0 < c) (h_abc : a * b * c = 1) :
  1 / (a^3 * (b + c)) + 1 / (b^3 * (c + a)) + 1 / (c^3 * (a + b)) ≥ 3 / 2 :=
by {
  sorry
}

end inequality_of_abc_l207_207621


namespace task_completion_time_l207_207551

theorem task_completion_time (A B : ℝ) : 
  (14 * A / 80 + 10 * B / 96) = (20 * (A + B)) →
  (1 / (14 * A / 80 + 10 * B / 96)) = 480 / (84 * A + 50 * B) :=
by
  intros h
  sorry

end task_completion_time_l207_207551


namespace min_colors_required_l207_207592

-- Defining the color type
def Color := ℕ

-- Defining a 6x6 grid
def Grid := Fin 6 → Fin 6 → Color

-- Defining the conditions of the problem for a valid coloring
def is_valid_coloring (c : Grid) : Prop :=
  (∀ i j k, i ≠ j → c i k ≠ c j k) ∧ -- each row has all cells with different colors
  (∀ i j k, i ≠ j → c k i ≠ c k j) ∧ -- each column has all cells with different colors
  (∀ i j, i ≠ j → c i (i+j) ≠ c j (i+j)) ∧ -- each 45° diagonal has all different colors
  (∀ i j, i ≠ j → (i-j ≥ 0 → c (i-j) i ≠ c (i-j) j) ∧ (j-i ≥ 0 → c i (j-i) ≠ c j (j-i))) -- each 135° diagonal has all different colors

-- The formal statement of the math problem
theorem min_colors_required : ∃ (n : ℕ), (∀ c : Grid, is_valid_coloring c → n ≥ 7) :=
sorry

end min_colors_required_l207_207592


namespace cone_radius_from_melted_cylinder_l207_207694

theorem cone_radius_from_melted_cylinder :
  ∀ (r_cylinder h_cylinder r_cone h_cone : ℝ),
  r_cylinder = 8 ∧ h_cylinder = 2 ∧ h_cone = 6 ∧
  (π * r_cylinder^2 * h_cylinder = (1 / 3) * π * r_cone^2 * h_cone) →
  r_cone = 8 :=
by
  sorry

end cone_radius_from_melted_cylinder_l207_207694


namespace equation_of_line_through_point_with_given_slope_l207_207910

-- Define the condition that line L passes through point P(-2, 5) and has slope -3/4
def line_through_point_with_slope (x1 y1 m : ℚ) (x y : ℚ) : Prop :=
  y - y1 = m * (x - x1)

-- Define the specific point (-2, 5) and slope -3/4
def P : ℚ × ℚ := (-2, 5)
def m : ℚ := -3 / 4

-- The standard form equation of the line as the target
def standard_form (x y : ℚ) : Prop :=
  3 * x + 4 * y - 14 = 0

-- The theorem to prove
theorem equation_of_line_through_point_with_given_slope :
  ∀ x y : ℚ, line_through_point_with_slope (-2) 5 (-3 / 4) x y → standard_form x y :=
  by
    intros x y h
    sorry

end equation_of_line_through_point_with_given_slope_l207_207910


namespace equation_c_is_linear_l207_207871

-- Define the condition for being a linear equation with one variable
def is_linear_equation_with_one_variable (eq : ℝ → Prop) : Prop :=
  ∃ a b : ℝ, a ≠ 0 ∧ ∀ x : ℝ, eq x ↔ (a * x + b = 0)

-- The given equation to check is (x - 1) / 2 = 1, which simplifies to x = 3
def equation_c (x : ℝ) : Prop := (x - 1) / 2 = 1

-- Prove that the given equation is a linear equation with one variable
theorem equation_c_is_linear :
  is_linear_equation_with_one_variable equation_c :=
sorry

end equation_c_is_linear_l207_207871


namespace age_of_B_l207_207994

variable (a b : ℕ)

-- Conditions
def condition1 := a + 10 = 2 * (b - 10)
def condition2 := a = b + 5

-- The proof goal
theorem age_of_B (h1 : condition1 a b) (h2 : condition2 a b) : b = 35 := by
  sorry

end age_of_B_l207_207994


namespace saras_sister_ordered_notebooks_l207_207041

theorem saras_sister_ordered_notebooks (x : ℕ) 
  (initial_notebooks : ℕ := 4) 
  (lost_notebooks : ℕ := 2) 
  (current_notebooks : ℕ := 8) :
  initial_notebooks + x - lost_notebooks = current_notebooks → x = 6 :=
by
  intros h
  sorry

end saras_sister_ordered_notebooks_l207_207041


namespace min_sum_of_intercepts_l207_207991

-- Definitions based on conditions
def line (a b : ℝ) (x y : ℝ) : Prop := a * x + b * y = a * b
def point_on_line (a b : ℝ) : Prop := line a b 1 1

-- Main theorem statement
theorem min_sum_of_intercepts (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h_point : point_on_line a b) : 
  a + b >= 4 :=
sorry

end min_sum_of_intercepts_l207_207991


namespace find_natural_number_l207_207128

theorem find_natural_number (x : ℕ) (y z : ℤ) (hy : x = 2 * y^2 - 1) (hz : x^2 = 2 * z^2 - 1) : x = 1 ∨ x = 7 :=
sorry

end find_natural_number_l207_207128


namespace increase_average_by_runs_l207_207805

theorem increase_average_by_runs :
  let total_runs_10_matches : ℕ := 10 * 32
  let runs_scored_next_match : ℕ := 87
  let total_runs_11_matches : ℕ := total_runs_10_matches + runs_scored_next_match
  let new_average_11_matches : ℚ := total_runs_11_matches / 11
  let increased_average : ℚ := 32 + 5
  new_average_11_matches = increased_average :=
by
  sorry

end increase_average_by_runs_l207_207805


namespace ratio_boys_to_girls_l207_207461

-- Define the given conditions
def G : ℕ := 300
def T : ℕ := 780

-- State the proposition to be proven
theorem ratio_boys_to_girls (B : ℕ) (h : B + G = T) : B / G = 8 / 5 :=
by
  -- Proof placeholder
  sorry

end ratio_boys_to_girls_l207_207461


namespace positive_difference_of_R_coords_l207_207695

theorem positive_difference_of_R_coords :
    ∀ (xR yR : ℝ),
    ∃ (k : ℝ),
    (∀ (A B C R S : ℝ × ℝ), 
    A = (-1, 6) ∧ B = (1, 2) ∧ C = (7, 2) ∧ 
    R = (k, -0.5 * k + 5.5) ∧ S = (k, 2) ∧
    (0.5 * |7 - k| * |0.5 * k - 3.5| = 8)) → 
    |xR - yR| = 1 :=
by
  sorry

end positive_difference_of_R_coords_l207_207695


namespace potato_bag_weight_l207_207898

-- Defining the weight of the bag of potatoes as a variable W
variable (W : ℝ)

-- Given condition: The weight of the bag is described by the equation
def weight_condition (W : ℝ) := W = 12 / (W / 2)

-- Proving the weight of the bag of potatoes is 12 lbs:
theorem potato_bag_weight : weight_condition W → W = 12 :=
by
  sorry

end potato_bag_weight_l207_207898


namespace razorback_tshirt_sales_l207_207556

theorem razorback_tshirt_sales 
  (price_per_tshirt : ℕ) (total_money_made : ℕ)
  (h1 : price_per_tshirt = 16) (h2 : total_money_made = 720) :
  total_money_made / price_per_tshirt = 45 :=
by
  sorry

end razorback_tshirt_sales_l207_207556


namespace find_number_l207_207661

theorem find_number (x : ℝ) (h : x^2 + 50 = (x - 10)^2) : x = 2.5 :=
sorry

end find_number_l207_207661


namespace ivy_collectors_edition_dolls_l207_207816

-- Definitions from the conditions
def dina_dolls : ℕ := 60
def ivy_dolls : ℕ := dina_dolls / 2
def collectors_edition_dolls : ℕ := (2 * ivy_dolls) / 3

-- Assertion
theorem ivy_collectors_edition_dolls : collectors_edition_dolls = 20 := by
  sorry

end ivy_collectors_edition_dolls_l207_207816


namespace largest_number_in_sequence_l207_207742

noncomputable def largest_in_sequence (s : Fin 8 → ℝ) : ℝ :=
  max (s 0) (max (s 1) (max (s 2) (max (s 3) (max (s 4) (max (s 5) (max (s 6) (s 7)))))))

theorem largest_number_in_sequence (s : Fin 8 → ℝ)
  (h1 : ∀ i j : Fin 8, i < j → s i < s j)
  (h2 : ∃ i : Fin 5, (∃ d : ℝ, d = 4 ∨ d = 36) ∧ (∀ j : ℕ, j < 3 → s (i+j) + d = s (i+j+1)))
  (h3 : ∃ i : Fin 5, ∃ r : ℝ, (∀ j : ℕ, j < 3 → s (i+j) * r = s (i+j+1))) :
  largest_in_sequence s = 126 ∨ largest_in_sequence s = 6 :=
sorry

end largest_number_in_sequence_l207_207742


namespace xiaoxiao_types_faster_l207_207704

-- Defining the characters typed and time taken by both individuals
def characters_typed_taoqi : ℕ := 200
def time_taken_taoqi : ℕ := 5
def characters_typed_xiaoxiao : ℕ := 132
def time_taken_xiaoxiao : ℕ := 3

-- Calculating typing speeds
def speed_taoqi : ℕ := characters_typed_taoqi / time_taken_taoqi
def speed_xiaoxiao : ℕ := characters_typed_xiaoxiao / time_taken_xiaoxiao

-- Proving that 笑笑 types faster
theorem xiaoxiao_types_faster : speed_xiaoxiao > speed_taoqi := by
  -- Given calculations:
  -- speed_taoqi = 40
  -- speed_xiaoxiao = 44
  sorry

end xiaoxiao_types_faster_l207_207704


namespace ball_hits_ground_approx_time_l207_207437

noncomputable def ball_hits_ground_time (t : ℝ) : ℝ :=
-6 * t^2 - 12 * t + 60

theorem ball_hits_ground_approx_time :
  ∃ t : ℝ, |t - 2.32| < 0.01 ∧ ball_hits_ground_time t = 0 :=
sorry

end ball_hits_ground_approx_time_l207_207437


namespace range_of_a_l207_207402

-- Defining the function f(x) = x^2 + 2ax - 1
def f (x a : ℝ) : ℝ := x^2 + 2 * a * x - 1

-- Conditions: x1, x2 ∈ [1, +∞) and x1 < x2
variables (x1 x2 a : ℝ)
variables (h1 : 1 ≤ x1) (h2 : 1 ≤ x2) (h3 : x1 < x2)

-- Statement of the proof problem:
theorem range_of_a (hf_ineq : x2 * f x1 a - x1 * f x2 a < a * (x1 - x2)) : a ≤ 2 :=
sorry

end range_of_a_l207_207402


namespace carlos_earnings_l207_207724

theorem carlos_earnings (h1 : ∃ w, 18 * w = w * 18) (h2 : ∃ w, 30 * w = w * 30) (h3 : ∀ w, 30 * w - 18 * w = 54) : 
  ∃ w, 18 * w + 30 * w = 216 := 
sorry

end carlos_earnings_l207_207724


namespace bacteria_exceeds_day_l207_207817

theorem bacteria_exceeds_day :
  ∃ n : ℕ, 5 * 3^n > 200 ∧ ∀ m : ℕ, (m < n → 5 * 3^m ≤ 200) :=
sorry

end bacteria_exceeds_day_l207_207817


namespace opposite_of_neg_five_is_five_l207_207482

-- Define the condition for the opposite of a number
def is_opposite (x y : ℤ) : Prop := x + y = 0

-- State the theorem that the opposite of -5 is 5
theorem opposite_of_neg_five_is_five : is_opposite (-5) 5 :=
by
  -- Insert the proof here
  sorry

end opposite_of_neg_five_is_five_l207_207482


namespace maximize_net_income_l207_207302

noncomputable def net_income (x : ℕ) : ℤ :=
  if 60 ≤ x ∧ x ≤ 90 then 750 * x - 1700
  else if 90 < x ∧ x ≤ 300 then -3 * x * x + 1020 * x - 1700
  else 0

theorem maximize_net_income :
  (∀ x : ℕ, 60 ≤ x ∧ x ≤ 300 →
    net_income x ≤ net_income 170) ∧
  net_income 170 = 85000 := 
sorry

end maximize_net_income_l207_207302


namespace integral_result_l207_207114

open Real

theorem integral_result :
  (∫ x in (0:ℝ)..(π/2), (x^2 - 5 * x + 6) * sin (3 * x)) = (67 - 3 * π) / 27 := by
  sorry

end integral_result_l207_207114


namespace min_value_fracs_l207_207903

-- Define the problem and its conditions in Lean.
theorem min_value_fracs (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 2 * b = 1) : 
  (2 / a + 3 / b) ≥ 8 + 4 * Real.sqrt 3 :=
  sorry

end min_value_fracs_l207_207903


namespace smallest_k_l207_207852

theorem smallest_k (k : ℕ) (h₁ : k > 1) (h₂ : k % 17 = 1) (h₃ : k % 6 = 1) (h₄ : k % 2 = 1) : k = 103 :=
by sorry

end smallest_k_l207_207852


namespace floor_neg_seven_fourths_l207_207582

theorem floor_neg_seven_fourths : (Int.floor (-7 / 4 : ℚ) = -2) := 
by
  sorry

end floor_neg_seven_fourths_l207_207582


namespace sum_of_first_K_natural_numbers_is_perfect_square_l207_207111

noncomputable def values_K (K : ℕ) : Prop := 
  ∃ N : ℕ, (K * (K + 1)) / 2 = N^2 ∧ (N + K < 120)

theorem sum_of_first_K_natural_numbers_is_perfect_square :
  ∀ K : ℕ, values_K K ↔ (K = 1 ∨ K = 8 ∨ K = 49) := by
  sorry

end sum_of_first_K_natural_numbers_is_perfect_square_l207_207111


namespace strawberry_jelly_sales_l207_207773

def jelly_sales (grape strawberry raspberry plum : ℕ) : Prop :=
  grape = 2 * strawberry ∧
  raspberry = 2 * plum ∧
  raspberry = grape / 3 ∧
  plum = 6

theorem strawberry_jelly_sales {grape strawberry raspberry plum : ℕ}
    (h : jelly_sales grape strawberry raspberry plum) : 
    strawberry = 18 :=
by
  have h1 := h.1
  have h2 := h.2.1
  have h3 := h.2.2.1
  have h4 := h.2.2.2
  sorry

end strawberry_jelly_sales_l207_207773


namespace negation_proof_l207_207224

theorem negation_proof (x : ℝ) : ¬ (∃ x : ℝ, x^2 + 2*x + 2 ≤ 0) ↔ ∀ x : ℝ, x^2 + 2*x + 2 > 0 :=
by
  sorry

end negation_proof_l207_207224


namespace find_S_2013_l207_207546

variable {a : ℕ → ℤ} -- the arithmetic sequence
variable {S : ℕ → ℤ} -- the sum of the first n terms

-- Conditions
axiom a1_eq_neg2011 : a 1 = -2011
axiom sum_sequence : ∀ n, S n = (n * (a 1 + a n)) / 2
axiom condition_eq : (S 2012 / 2012) - (S 2011 / 2011) = 1

-- The Lean statement to prove that S 2013 = 2013
theorem find_S_2013 : S 2013 = 2013 := by
  sorry

end find_S_2013_l207_207546


namespace simplify_product_l207_207435

theorem simplify_product (x t : ℕ) : (x^2 * t^3) * (x^3 * t^4) = (x^5) * (t^7) := 
by 
  sorry

end simplify_product_l207_207435


namespace table_length_l207_207329

theorem table_length (area_m2 : ℕ) (width_cm : ℕ) (length_cm : ℕ) 
  (h_area : area_m2 = 54)
  (h_width : width_cm = 600)
  :
  length_cm = 900 := 
  sorry

end table_length_l207_207329


namespace first_comparison_second_comparison_l207_207906

theorem first_comparison (x y : ℕ) (h1 : x = 2^40) (h2 : y = 3^28) : x < y := 
by sorry

theorem second_comparison (a b : ℕ) (h3 : a = 31^11) (h4 : b = 17^14) : a < b := 
by sorry

end first_comparison_second_comparison_l207_207906


namespace smallest_b_value_l207_207558

theorem smallest_b_value (a b c : ℕ) (h0 : a > 0) (h1 : b > 0) (h2 : c > 0)
  (h3 : (31 : ℚ) / 72 = (a : ℚ) / 8 + (b : ℚ) / 9 - c) :
  b = 5 :=
sorry

end smallest_b_value_l207_207558


namespace correct_answer_l207_207973

theorem correct_answer (a b c : ℤ) 
  (h : (a - b) ^ 10 + (a - c) ^ 10 = 1) : 
  |a - b| + |b - c| + |c - a| = 2 := by 
  sorry

end correct_answer_l207_207973


namespace find_PR_in_triangle_l207_207189

theorem find_PR_in_triangle (P Q R M : ℝ) (PQ QR PM : ℝ):
  PQ = 7 →
  QR = 10 →
  PM = 5 →
  M = (Q + R) / 2 →
  PR = Real.sqrt 149 := 
sorry

end find_PR_in_triangle_l207_207189


namespace relationship_y1_y2_y3_l207_207239

theorem relationship_y1_y2_y3 
  (y_1 y_2 y_3 : ℝ)
  (h1 : y_1 = (-2)^2 + 2*(-2) + 2)
  (h2 : y_2 = (-1)^2 + 2*(-1) + 2)
  (h3 : y_3 = 2^2 + 2*2 + 2) :
  y_2 < y_1 ∧ y_1 < y_3 := 
sorry

end relationship_y1_y2_y3_l207_207239


namespace alexis_initial_budget_l207_207664

-- Define all the given conditions
def cost_shirt : Int := 30
def cost_pants : Int := 46
def cost_coat : Int := 38
def cost_socks : Int := 11
def cost_belt : Int := 18
def cost_shoes : Int := 41
def amount_left : Int := 16

-- Define the total expenses
def total_expenses : Int := cost_shirt + cost_pants + cost_coat + cost_socks + cost_belt + cost_shoes

-- Define the initial budget
def initial_budget : Int := total_expenses + amount_left

-- The proof statement
theorem alexis_initial_budget : initial_budget = 200 := by
  sorry

end alexis_initial_budget_l207_207664


namespace valid_arrangements_count_is_20_l207_207170

noncomputable def count_valid_arrangements : ℕ :=
  sorry

theorem valid_arrangements_count_is_20 :
  count_valid_arrangements = 20 :=
  by
    sorry

end valid_arrangements_count_is_20_l207_207170


namespace intersection_A_complement_is_2_4_l207_207721

-- Declare the universal set U, set A, and set B
def U : Set ℕ := { 1, 2, 3, 4, 5, 6, 7 }
def A : Set ℕ := { 2, 4, 5 }
def B : Set ℕ := { 1, 3, 5, 7 }

-- Define the complement of set B with respect to U
def complement_U_B : Set ℕ := { x ∈ U | x ∉ B }

-- Define the intersection of set A and the complement of set B
def intersection_A_complement_U_B : Set ℕ := { x ∈ A | x ∈ complement_U_B }

-- State the theorem
theorem intersection_A_complement_is_2_4 : 
  intersection_A_complement_U_B = { 2, 4 } := 
by
  sorry

end intersection_A_complement_is_2_4_l207_207721


namespace number_of_women_in_first_class_l207_207796

-- Definitions for the conditions
def total_passengers : ℕ := 180
def percentage_women : ℝ := 0.65
def percentage_women_first_class : ℝ := 0.15

-- The desired proof statement
theorem number_of_women_in_first_class :
  (round (total_passengers * percentage_women * percentage_women_first_class) = 18) :=
by
  sorry  

end number_of_women_in_first_class_l207_207796


namespace sun_radius_scientific_notation_l207_207719

theorem sun_radius_scientific_notation : 
  (369000 : ℝ) = 3.69 * 10^5 :=
by
  sorry

end sun_radius_scientific_notation_l207_207719


namespace tan_13pi_div_3_eq_sqrt_3_l207_207527

theorem tan_13pi_div_3_eq_sqrt_3 : Real.tan (13 * Real.pi / 3) = Real.sqrt 3 :=
  sorry

end tan_13pi_div_3_eq_sqrt_3_l207_207527


namespace remainder_3_pow_20_mod_5_l207_207713

theorem remainder_3_pow_20_mod_5 : (3 ^ 20) % 5 = 1 := by
  sorry

end remainder_3_pow_20_mod_5_l207_207713


namespace min_chemistry_teachers_l207_207815

/--
A school has 7 maths teachers, 6 physics teachers, and some chemistry teachers.
Each teacher can teach a maximum of 3 subjects.
The minimum number of teachers required is 6.
Prove that the minimum number of chemistry teachers required is 1.
-/
theorem min_chemistry_teachers (C : ℕ) (math_teachers : ℕ := 7) (physics_teachers : ℕ := 6) 
  (max_subjects_per_teacher : ℕ := 3) (min_teachers_required : ℕ := 6) :
  7 + 6 + C ≤ 6 * 3 → C = 1 := 
by
  sorry

end min_chemistry_teachers_l207_207815


namespace toothpaste_runs_out_in_two_days_l207_207981

noncomputable def toothpaste_capacity := 90
noncomputable def dad_usage_per_brushing := 4
noncomputable def mom_usage_per_brushing := 3
noncomputable def anne_usage_per_brushing := 2
noncomputable def brother_usage_per_brushing := 1
noncomputable def sister_usage_per_brushing := 1

noncomputable def dad_brushes_per_day := 4
noncomputable def mom_brushes_per_day := 4
noncomputable def anne_brushes_per_day := 4
noncomputable def brother_brushes_per_day := 4
noncomputable def sister_brushes_per_day := 2

noncomputable def total_daily_usage :=
  dad_usage_per_brushing * dad_brushes_per_day + 
  mom_usage_per_brushing * mom_brushes_per_day + 
  anne_usage_per_brushing * anne_brushes_per_day + 
  brother_usage_per_brushing * brother_brushes_per_day + 
  sister_usage_per_brushing * sister_brushes_per_day

theorem toothpaste_runs_out_in_two_days :
  toothpaste_capacity / total_daily_usage = 2 := by
  -- Proof omitted
  sorry

end toothpaste_runs_out_in_two_days_l207_207981


namespace hyperbola_eccentricity_l207_207847

theorem hyperbola_eccentricity (a : ℝ) (h : 0 < a) (h_asymptote : Real.tan (Real.pi / 6) = 1 / a) : 
  (a = Real.sqrt 3) → 
  (∃ e : ℝ, e = (2 * Real.sqrt 3) / 3) :=
by
  intros
  sorry

end hyperbola_eccentricity_l207_207847


namespace painted_cube_count_is_three_l207_207564

-- Define the colors of the faces
inductive Color
| Yellow
| Black
| White

-- Define a Cube with painted faces
structure Cube :=
(f1 f2 f3 f4 f5 f6 : Color)

-- Define rotational symmetry (two cubes are the same under rotation)
def equivalentUpToRotation (c1 c2 : Cube) : Prop := sorry -- Symmetry function

-- Define a property that counts the correct painting configuration
def paintedCubeCount : ℕ :=
  sorry -- Function to count correctly painted and uniquely identifiable cubes

theorem painted_cube_count_is_three :
  paintedCubeCount = 3 :=
sorry

end painted_cube_count_is_three_l207_207564


namespace six_digit_palindromes_count_l207_207886

theorem six_digit_palindromes_count :
  (∃ (a b c d : ℕ),
    1 ≤ a ∧ a ≤ 9 ∧ 
    0 ≤ b ∧ b ≤ 9 ∧ 
    0 ≤ c ∧ c ≤ 9 ∧ 
    0 ≤ d ∧ d ≤ 9) →
  9 * 10 * 10 * 10 = 9000 :=
by
  sorry

end six_digit_palindromes_count_l207_207886


namespace average_height_students_count_l207_207100

-- Definitions based on the conditions
def total_students : ℕ := 400
def short_students : ℕ := (2 * total_students) / 5
def extremely_tall_students : ℕ := total_students / 10
def tall_students : ℕ := 90
def average_height_students : ℕ := total_students - (short_students + tall_students + extremely_tall_students)

-- Theorem to prove
theorem average_height_students_count : average_height_students = 110 :=
by
  -- This proof is omitted, we are only stating the theorem.
  sorry

end average_height_students_count_l207_207100


namespace rectangle_area_l207_207649

theorem rectangle_area (A1 A2 : ℝ) (h1 : A1 = 40) (h2 : A2 = 10) :
    ∃ n : ℕ, n = 240 ∧ ∃ R : ℝ, R = 2 * Real.sqrt (40 / Real.pi) + 2 * Real.sqrt (10 / Real.pi) ∧ 
               (4 * Real.sqrt (10) / Real.sqrt (Real.pi)) * (6 * Real.sqrt (10) / Real.sqrt (Real.pi)) = n / Real.pi :=
by
  sorry

end rectangle_area_l207_207649


namespace terminal_side_half_angle_l207_207639

theorem terminal_side_half_angle {k : ℤ} {α : ℝ} 
  (h : 2 * k * π < α ∧ α < 2 * k * π + π / 2) : 
  (k * π < α / 2 ∧ α / 2 < k * π + π / 4) ∨ (k * π + π <= α / 2 ∧ α / 2 < (k + 1) * π + π / 4) :=
sorry

end terminal_side_half_angle_l207_207639


namespace certain_percentage_l207_207622

theorem certain_percentage (P : ℝ) : 
  0.15 * P * 0.50 * 4000 = 90 → P = 0.3 :=
by
  sorry

end certain_percentage_l207_207622


namespace complement_union_complement_intersection_l207_207407

open Set

def A : Set ℝ := {x | 3 ≤ x ∧ x < 7}
def B : Set ℝ := {x | 2 < x ∧ x < 10}

theorem complement_union (A B : Set ℝ) :
  (A ∪ B)ᶜ = { x : ℝ | x ≤ 2 ∨ x ≥ 10 } :=
by
  sorry

theorem complement_intersection (A B : Set ℝ) :
  (Aᶜ ∩ B) = { x : ℝ | (2 < x ∧ x < 3) ∨ (7 ≤ x ∧ x < 10) } :=
by
  sorry

end complement_union_complement_intersection_l207_207407


namespace find_t_l207_207360

-- conditions
def quadratic_eq (x : ℝ) : Prop := 25 * x^2 + 20 * x - 1000 = 0

-- statement to prove
theorem find_t (x : ℝ) (p t : ℝ) (h1 : p = 2/5) (h2 : t = 104/25) : 
  (quadratic_eq x) → (x + p)^2 = t :=
by
  intros
  sorry

end find_t_l207_207360


namespace todd_ate_cupcakes_l207_207538

def total_cupcakes_baked := 68
def packages := 6
def cupcakes_per_package := 6
def total_packaged_cupcakes := packages * cupcakes_per_package
def remaining_cupcakes := total_cupcakes_baked - total_packaged_cupcakes

theorem todd_ate_cupcakes : total_cupcakes_baked - remaining_cupcakes = 36 := by
  sorry

end todd_ate_cupcakes_l207_207538


namespace greatest_int_less_than_200_gcd_30_is_5_l207_207900

theorem greatest_int_less_than_200_gcd_30_is_5 : ∃ n, n < 200 ∧ gcd n 30 = 5 ∧ ∀ m, m < 200 ∧ gcd m 30 = 5 → m ≤ n :=
by
  sorry

end greatest_int_less_than_200_gcd_30_is_5_l207_207900


namespace combine_like_terms_l207_207570

theorem combine_like_terms (a : ℝ) : 2 * a + 3 * a = 5 * a := 
by sorry

end combine_like_terms_l207_207570


namespace average_speed_l207_207547

theorem average_speed (d1 d2 : ℝ) (t1 t2 : ℝ) (h1 : d1 = 90) (h2 : d2 = 75) (ht1 : t1 = 1) (ht2 : t2 = 1) :
  (d1 + d2) / (t1 + t2) = 82.5 :=
by
  sorry

end average_speed_l207_207547


namespace solve_E_l207_207784

-- Definitions based on the conditions provided
variables {A H S M C O E : ℕ}

-- Given conditions
def algebra_books := A
def geometry_books := H
def history_books := C
def S_algebra_books := S
def M_geometry_books := M
def O_history_books := O
def E_algebra_books := E

-- Prove that E = (AM + AO - SH - SC) / (M + O - H - C) given the conditions
theorem solve_E (h1: A ≠ H) (h2: A ≠ S) (h3: A ≠ M) (h4: A ≠ C) (h5: A ≠ O) (h6: A ≠ E)
                (h7: H ≠ S) (h8: H ≠ M) (h9: H ≠ C) (h10: H ≠ O) (h11: H ≠ E)
                (h12: S ≠ M) (h13: S ≠ C) (h14: S ≠ O) (h15: S ≠ E)
                (h16: M ≠ C) (h17: M ≠ O) (h18: M ≠ E)
                (h19: C ≠ O) (h20: C ≠ E)
                (h21: O ≠ E)
                (pos1: 0 < A) (pos2: 0 < H) (pos3: 0 < S) (pos4: 0 < M) (pos5: 0 < C)
                (pos6: 0 < O) (pos7: 0 < E) :
  E = (A * M + A * O - S * H - S * C) / (M + O - H - C) :=
sorry

end solve_E_l207_207784


namespace non_congruent_triangles_count_l207_207498

def is_triangle (a b c : ℕ) : Prop :=
  (a + b > c) ∧ (b + c > a) ∧ (c + a > b)

def count_non_congruent_triangles : ℕ :=
  let a_values := [1, 2]
  let b_values := [2, 3]
  let triangles := [(1, 2, 2), (2, 2, 2), (2, 2, 3)]
  triangles.length

theorem non_congruent_triangles_count : count_non_congruent_triangles = 3 :=
  by
    -- Proof would go here
    sorry

end non_congruent_triangles_count_l207_207498


namespace number_of_students_playing_soccer_l207_207923

variables (T B girls_total soccer_total G no_girls_soccer perc_boys_soccer : ℕ)

-- Conditions:
def total_students := T = 420
def boys_students := B = 312
def girls_students := G = 420 - 312
def girls_not_playing_soccer := no_girls_soccer = 63
def perc_boys_play_soccer := perc_boys_soccer = 82
def girls_playing_soccer := G - no_girls_soccer = 45

-- Proof Problem:
theorem number_of_students_playing_soccer (h1 : total_students T) (h2 : boys_students B) (h3 : girls_students G) (h4 : girls_not_playing_soccer no_girls_soccer) (h5 : girls_playing_soccer G no_girls_soccer) (h6 : perc_boys_play_soccer perc_boys_soccer) : soccer_total = 250 :=
by {
  -- The proof would be inserted here.
  sorry
}

end number_of_students_playing_soccer_l207_207923


namespace find_initial_strawberries_l207_207199

-- Define the number of strawberries after picking 35 more to be 63
def strawberries_after_picking := 63

-- Define the number of strawberries picked
def strawberries_picked := 35

-- Define the initial number of strawberries
def initial_strawberries := 28

-- State the theorem
theorem find_initial_strawberries (x : ℕ) (h : x + strawberries_picked = strawberries_after_picking) : x = initial_strawberries :=
by
  -- Proof omitted
  sorry

end find_initial_strawberries_l207_207199


namespace unique_solution_condition_l207_207722

theorem unique_solution_condition (a b c : ℝ) : 
  (∀ x : ℝ, 4 * x - 7 + a = (b + 1) * x + c) ↔ b ≠ 3 :=
by
  sorry

end unique_solution_condition_l207_207722


namespace sqrt_nested_eq_l207_207365

theorem sqrt_nested_eq (y : ℝ) (hy : 0 ≤ y) :
  Real.sqrt (y * Real.sqrt (y * Real.sqrt (y * Real.sqrt y))) = y ^ (9 / 4) :=
by
  sorry

end sqrt_nested_eq_l207_207365


namespace intersection_is_singleton_l207_207530

-- Definitions of sets M and N
def M : Set (ℝ × ℝ) := {p | p.1 + p.2 = 2}
def N : Set (ℝ × ℝ) := {p | p.1 - p.2 = 4}

-- The stated proposition we need to prove
theorem intersection_is_singleton :
  M ∩ N = {(3, -1)} :=
by {
  sorry
}

end intersection_is_singleton_l207_207530


namespace find_inlet_rate_l207_207072

-- definitions for the given conditions
def volume_cubic_feet : ℝ := 20
def conversion_factor : ℝ := 12^3
def volume_cubic_inches : ℝ := volume_cubic_feet * conversion_factor

def outlet_rate1 : ℝ := 9
def outlet_rate2 : ℝ := 8
def empty_time : ℕ := 2880

-- theorem that captures the proof problem
theorem find_inlet_rate (volume_cubic_inches : ℝ) (outlet_rate1 outlet_rate2 empty_time : ℝ) :
  ∃ (inlet_rate : ℝ), volume_cubic_inches = (outlet_rate1 + outlet_rate2 - inlet_rate) * empty_time ↔ inlet_rate = 5 := 
by
  sorry

end find_inlet_rate_l207_207072


namespace rectangle_area_l207_207343

theorem rectangle_area (x : ℝ) :
  let large_rectangle_area := (2 * x + 14) * (2 * x + 10)
  let hole_area := (4 * x - 6) * (2 * x - 4)
  let square_area := (x + 3) * (x + 3)
  large_rectangle_area - hole_area + square_area = -3 * x^2 + 82 * x + 125 := 
by
  sorry

end rectangle_area_l207_207343


namespace percentage_of_women_picnic_l207_207922

theorem percentage_of_women_picnic (E : ℝ) (h1 : 0.20 * 0.55 * E + W * 0.45 * E = 0.29 * E) : 
  W = 0.4 := 
  sorry

end percentage_of_women_picnic_l207_207922


namespace triangle_area_is_two_l207_207019

noncomputable def triangle_area (b c : ℝ) (angle_A : ℝ) : ℝ :=
  (1 / 2) * b * c * Real.sin angle_A

theorem triangle_area_is_two
  (A B C : ℝ) (a b c : ℝ)
  (hA : A = π / 4)
  (hCondition : b^2 * Real.sin C = 4 * Real.sqrt 2 * Real.sin B)
  (hBC : b * c = 4 * Real.sqrt 2) : 
  triangle_area b c A = 2 :=
by
  -- actual proof omitted
  sorry

end triangle_area_is_two_l207_207019


namespace derivative_f_eq_l207_207370

noncomputable def f (x : ℝ) : ℝ := (Real.exp (2 * x)) / x

theorem derivative_f_eq :
  (deriv f) = fun x ↦ ((2 * x - 1) * (Real.exp (2 * x))) / (x ^ 2) := by
  sorry

end derivative_f_eq_l207_207370


namespace total_hours_for_songs_l207_207449

def total_hours_worked_per_day := 10
def total_days_per_song := 10
def number_of_songs := 3

theorem total_hours_for_songs :
  total_hours_worked_per_day * total_days_per_song * number_of_songs = 300 :=
by
  sorry

end total_hours_for_songs_l207_207449


namespace first_wing_hall_rooms_l207_207715

theorem first_wing_hall_rooms
    (total_rooms : ℕ) (first_wing_floors : ℕ) (first_wing_halls_per_floor : ℕ)
    (second_wing_floors : ℕ) (second_wing_halls_per_floor : ℕ) (second_wing_rooms_per_hall : ℕ)
    (hotel_total_rooms : ℕ) (first_wing_total_rooms : ℕ) :
    hotel_total_rooms = total_rooms →
    first_wing_floors = 9 →
    first_wing_halls_per_floor = 6 →
    second_wing_floors = 7 →
    second_wing_halls_per_floor = 9 →
    second_wing_rooms_per_hall = 40 →
    hotel_total_rooms = first_wing_total_rooms + (second_wing_floors * second_wing_halls_per_floor * second_wing_rooms_per_hall) →
    first_wing_total_rooms = first_wing_floors * first_wing_halls_per_floor * 32 :=
by
  sorry

end first_wing_hall_rooms_l207_207715


namespace segment_length_OI_is_3_l207_207380

-- Define the points along the path
def point (n : ℕ) : ℝ × ℝ := (n, n)

-- Use the Pythagorean theorem to calculate the distance from point O to point I
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

-- Define the points O and I
def O : ℝ × ℝ := point 0
def I : ℝ × ℝ := point 3

-- The proposition to prove: 
-- The distance between points O and I is 3
theorem segment_length_OI_is_3 : distance O I = 3 := 
  sorry

end segment_length_OI_is_3_l207_207380


namespace find_fractions_l207_207139

noncomputable def fractions_to_sum_86_111 : Prop :=
  ∃ (a b d₁ d₂ : ℕ), 0 < a ∧ 0 < b ∧ d₁ ≤ 100 ∧ d₂ ≤ 100 ∧
  Nat.gcd a d₁ = 1 ∧ Nat.gcd b d₂ = 1 ∧
  (a: ℚ) / d₁ + (b: ℚ) / d₂ = 86 / 111

theorem find_fractions : fractions_to_sum_86_111 :=
  sorry

end find_fractions_l207_207139


namespace exists_real_k_l207_207420

theorem exists_real_k (c : Fin 1998 → ℕ)
  (h1 : 0 ≤ c 1)
  (h2 : ∀ (m n : ℕ), 0 < m → 0 < n → m + n < 1998 → c m + c n ≤ c (m + n) ∧ c (m + n) ≤ c m + c n + 1) :
  ∃ k : ℝ, ∀ n : Fin 1998, 1 ≤ n → c n = Int.floor (n * k) :=
by
  sorry

end exists_real_k_l207_207420


namespace min_troublemakers_29_l207_207145

noncomputable def min_troublemakers (n : ℕ) : ℕ :=
sorry

theorem min_troublemakers_29 : min_troublemakers 29 = 10 := 
sorry

end min_troublemakers_29_l207_207145


namespace red_tile_probability_l207_207864

def is_red_tile (n : ℕ) : Prop := n % 7 = 3

noncomputable def red_tiles_count : ℕ :=
  Nat.card {n : ℕ | n ≤ 70 ∧ is_red_tile n}

noncomputable def total_tiles_count : ℕ := 70

theorem red_tile_probability :
  (red_tiles_count : ℤ) / (total_tiles_count : ℤ) = (1 : ℤ) / 7 :=
sorry

end red_tile_probability_l207_207864


namespace cubic_equation_three_distinct_real_roots_l207_207110

noncomputable def f (a x : ℝ) : ℝ := x^3 - 3 * x^2 - a

theorem cubic_equation_three_distinct_real_roots (a : ℝ) :
  ∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃
  ∧ f a x₁ = 0 ∧ f a x₂ = 0 ∧ f a x₃ = 0 ↔ -4 < a ∧ a < 0 :=
sorry

end cubic_equation_three_distinct_real_roots_l207_207110


namespace simplify_expression_l207_207453

-- Defining each term in the sum
def term1  := 1 / ((1 / 3) ^ 1)
def term2  := 1 / ((1 / 3) ^ 2)
def term3  := 1 / ((1 / 3) ^ 3)

-- Sum of the terms
def terms_sum := term1 + term2 + term3

-- The simplification proof statement
theorem simplify_expression : 1 / terms_sum = 1 / 39 :=
by sorry

end simplify_expression_l207_207453


namespace breadth_of_rectangle_l207_207390

noncomputable def length (radius : ℝ) : ℝ := (1/4) * radius
noncomputable def side (sq_area : ℝ) : ℝ := Real.sqrt sq_area
noncomputable def radius (side : ℝ) : ℝ := side
noncomputable def breadth (rect_area length : ℝ) : ℝ := rect_area / length

theorem breadth_of_rectangle :
  breadth 200 (length (radius (side 1225))) = 200 / (1/4 * Real.sqrt 1225) :=
by
  sorry

end breadth_of_rectangle_l207_207390


namespace product_of_solutions_eq_neg_nine_product_of_solutions_l207_207727

theorem product_of_solutions_eq_neg_nine :
  ∀ (x : ℝ), (|x| = 3 * (|x| - 2)) → x = 3 ∨ x = -3 :=
by
  sorry

theorem product_of_solutions :
  (∀ (x : ℝ), (|x| = 3 * (|x| - 2)) → (∃ (a b : ℝ), x = a ∨ x = b ∧ a * b = -9)) :=
by
  sorry

end product_of_solutions_eq_neg_nine_product_of_solutions_l207_207727


namespace inequality_solution_l207_207260

theorem inequality_solution (x : ℝ) : 5 * x > 4 * x + 2 → x > 2 :=
by
  sorry

end inequality_solution_l207_207260


namespace f_is_even_if_g_is_odd_l207_207918

variable (g : ℝ → ℝ)

def is_odd (h : ℝ → ℝ) :=
  ∀ x : ℝ, h (-x) = -h x

def is_even (f : ℝ → ℝ) :=
  ∀ x : ℝ, f (-x) = f x

theorem f_is_even_if_g_is_odd (hg : is_odd g) :
  is_even (fun x => |g (x^4)|) :=
by
  sorry

end f_is_even_if_g_is_odd_l207_207918


namespace ball_hits_ground_approx_time_l207_207599

-- Conditions
def height (t : ℝ) : ℝ := -6.1 * t^2 + 4.5 * t + 10

-- Main statement to be proved
theorem ball_hits_ground_approx_time :
  ∃ t : ℝ, (height t = 0) ∧ (abs (t - 1.70) < 0.01) :=
sorry

end ball_hits_ground_approx_time_l207_207599


namespace count_ways_to_sum_2020_as_1s_and_2s_l207_207856

theorem count_ways_to_sum_2020_as_1s_and_2s : ∃ n, (∀ x y : ℕ, 4 * x + 5 * y = 2020 → x + y = n) → n = 102 :=
by
-- Mathematics proof needed.
sorry

end count_ways_to_sum_2020_as_1s_and_2s_l207_207856


namespace episodes_per_season_l207_207979

theorem episodes_per_season
  (days_to_watch : ℕ)
  (episodes_per_day : ℕ)
  (seasons : ℕ) :
  days_to_watch = 10 →
  episodes_per_day = 6 →
  seasons = 4 →
  (episodes_per_day * days_to_watch) / seasons = 15 :=
by
  intros
  sorry

end episodes_per_season_l207_207979


namespace pieces_brought_to_school_on_friday_l207_207850

def pieces_of_fruit_mark_had := 10
def pieces_eaten_first_four_days := 5
def pieces_kept_for_next_week := 2

theorem pieces_brought_to_school_on_friday :
  pieces_of_fruit_mark_had - pieces_eaten_first_four_days - pieces_kept_for_next_week = 3 :=
by
  sorry

end pieces_brought_to_school_on_friday_l207_207850


namespace sweets_neither_red_nor_green_l207_207057

theorem sweets_neither_red_nor_green (total_sweets red_sweets green_sweets : ℕ)
  (h1 : total_sweets = 285)
  (h2 : red_sweets = 49)
  (h3 : green_sweets = 59) : total_sweets - (red_sweets + green_sweets) = 177 :=
by
  sorry

end sweets_neither_red_nor_green_l207_207057


namespace quadratic_root_sum_m_n_l207_207783

theorem quadratic_root_sum_m_n (m n : ℤ) :
  (∃ x : ℤ, x^2 + m * x + 2 * n = 0 ∧ x = 2) → m + n = -2 :=
by
  sorry

end quadratic_root_sum_m_n_l207_207783


namespace sin_210_l207_207884

theorem sin_210 : Real.sin (210 * Real.pi / 180) = -1/2 := by
  sorry

end sin_210_l207_207884


namespace parabola_vertex_on_x_axis_l207_207774

theorem parabola_vertex_on_x_axis (c : ℝ) : 
  (∃ h k : ℝ, y = (x : ℝ)^2 - 12 * x + c ∧
   (h = -12 / 2) ∧
   (k = c - 144 / 4) ∧
   (k = 0)) ↔ c = 36 :=
by
  sorry

end parabola_vertex_on_x_axis_l207_207774


namespace katie_travel_distance_l207_207037

theorem katie_travel_distance (d_train d_bus d_bike d_car d_total d1 d2 d3 : ℕ)
  (h1 : d_train = 162)
  (h2 : d_bus = 124)
  (h3 : d_bike = 88)
  (h4 : d_car = 224)
  (h_total : d_total = d_train + d_bus + d_bike + d_car)
  (h1_distance : d1 = 96)
  (h2_distance : d2 = 108)
  (h3_distance : d3 = 130)
  (h1_prob : 30 = 30)
  (h2_prob : 50 = 50)
  (h3_prob : 20 = 20) :
  (d_total + d1 = 694) ∧
  (d_total + d2 = 706) ∧
  (d_total + d3 = 728) :=
sorry

end katie_travel_distance_l207_207037


namespace correct_sunset_time_proof_l207_207998

def Time := ℕ × ℕ  -- hours and minutes

def sunrise_time : Time := (7, 12)  -- 7:12 AM
def incorrect_daylight_duration : Time := (11, 15)  -- 11 hours 15 minutes as per newspaper

def add_time (t1 t2 : Time) : Time :=
  let (h1, m1) := t1
  let (h2, m2) := t2
  let minutes := m1 + m2
  let hours := h1 + h2 + minutes / 60
  (hours % 24, minutes % 60)

def correct_sunset_time : Time := (18, 27)  -- 18:27 in 24-hour format equivalent to 6:27 PM in 12-hour format

theorem correct_sunset_time_proof :
  add_time sunrise_time incorrect_daylight_duration = correct_sunset_time :=
by
  -- skipping the detailed proof for now
  sorry

end correct_sunset_time_proof_l207_207998


namespace distances_sum_in_triangle_l207_207038

variable (A B C O : Type)
variable [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace O]
variable (a b c P AO BO CO : ℝ)

def triangle_sides (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0

def triangle_perimeter (a b c : ℝ) (P : ℝ) : Prop :=
  P = a + b + c

def point_inside_triangle (O : Type) : Prop := 
  ∃ (A B C : Type), True -- Placeholder for the actual geometric condition

def distances_to_vertices (O : Type) (AO BO CO : ℝ) : Prop := 
  AO >= 0 ∧ BO >= 0 ∧ CO >= 0

theorem distances_sum_in_triangle
  (h1 : triangle_sides a b c)
  (h2 : triangle_perimeter a b c P)
  (h3 : point_inside_triangle O)
  (h4 : distances_to_vertices O AO BO CO) :
  P / 2 < AO + BO + CO ∧ AO + BO + CO < P :=
sorry

end distances_sum_in_triangle_l207_207038


namespace largest_4_digit_number_divisible_by_12_l207_207882

theorem largest_4_digit_number_divisible_by_12 : ∃ (n : ℕ), 1000 ≤ n ∧ n ≤ 9999 ∧ n % 12 = 0 ∧ ∀ m, 1000 ≤ m ∧ m ≤ 9999 ∧ m % 12 = 0 → m ≤ n := 
sorry

end largest_4_digit_number_divisible_by_12_l207_207882


namespace intersect_circle_line_l207_207876

theorem intersect_circle_line (k m : ℝ) : 
  (∃ (x y : ℝ), y = k * x + 2 * k ∧ x^2 + y^2 + m * x + 4 = 0) → m > 4 :=
by
  -- This statement follows from the conditions given in the problem
  -- You can use implicit for pure documentation
  -- We include a sorry here to skip the proof
  sorry

end intersect_circle_line_l207_207876


namespace prove_range_of_m_prove_m_value_l207_207919

def quadratic_roots (m : ℝ) (x1 x2 : ℝ) : Prop := 
  x1 * x1 - (2 * m - 3) * x1 + m * m = 0 ∧ 
  x2 * x2 - (2 * m - 3) * x2 + m * m = 0

def range_of_m (m : ℝ) : Prop := 
  m <= 3/4

def condition_on_m (m : ℝ) (x1 x2 : ℝ) : Prop :=
  x1 + x2 = -(x1 * x2)

theorem prove_range_of_m (m : ℝ) :
  (∃ x1 x2 : ℝ, quadratic_roots m x1 x2) → range_of_m m :=
sorry

theorem prove_m_value (m : ℝ) (x1 x2 : ℝ) :
  quadratic_roots m x1 x2 → condition_on_m m x1 x2 → m = -3 :=
sorry

end prove_range_of_m_prove_m_value_l207_207919


namespace proof_l207_207669

noncomputable def problem_statement (a b : ℝ) :=
  7 * (Real.sin a + Real.sin b) + 6 * (Real.cos a * Real.cos b - 1) = 0 →
  (Real.tan (a / 2) * Real.tan (b / 2) = 1 ∨ Real.tan (a / 2) * Real.tan (b / 2) = -1)

theorem proof : ∀ a b : ℝ, problem_statement a b := sorry

end proof_l207_207669


namespace smallest_b_for_fourth_power_l207_207288

noncomputable def is_fourth_power (n : ℕ) : Prop :=
  ∃ k : ℕ, k ^ 4 = n

theorem smallest_b_for_fourth_power :
  ∃ b : ℕ, (0 < b) ∧ (7 + 7 * b + 7 * b ^ 2 = (7 * 1 + 7 * 18 + 7 * 18 ^ 2)) 
  ∧ is_fourth_power (7 + 7 * b + 7 * b ^ 2) := sorry

end smallest_b_for_fourth_power_l207_207288


namespace correct_exponent_operation_l207_207052

theorem correct_exponent_operation (a b : ℝ) : 
  a^2 * a^3 = a^5 := 
by sorry

end correct_exponent_operation_l207_207052


namespace calculate_expression_l207_207179

theorem calculate_expression (f : ℕ → ℝ) (h1 : ∀ a b, f (a + b) = f a * f b) (h2 : f 1 = 2) : 
  (f 2 / f 1) + (f 4 / f 3) + (f 6 / f 5) = 6 := 
sorry

end calculate_expression_l207_207179


namespace average_cd_l207_207563

theorem average_cd (c d: ℝ) (h: (4 + 6 + 8 + c + d) / 5 = 18) : (c + d) / 2 = 36 :=
by sorry

end average_cd_l207_207563


namespace average_age_of_adults_l207_207026

theorem average_age_of_adults (n_total n_girls n_boys n_adults : ℕ) 
                              (avg_age_total avg_age_girls avg_age_boys avg_age_adults : ℕ)
                              (h1 : n_total = 60)
                              (h2 : avg_age_total = 18)
                              (h3 : n_girls = 30)
                              (h4 : avg_age_girls = 16)
                              (h5 : n_boys = 20)
                              (h6 : avg_age_boys = 17)
                              (h7 : n_adults = 10) :
                              avg_age_adults = 26 :=
sorry

end average_age_of_adults_l207_207026


namespace remainder_of_3_pow_100_mod_7_is_4_l207_207892

theorem remainder_of_3_pow_100_mod_7_is_4
  (h1 : 3^1 ≡ 3 [MOD 7])
  (h2 : 3^2 ≡ 2 [MOD 7])
  (h3 : 3^3 ≡ 6 [MOD 7])
  (h4 : 3^4 ≡ 4 [MOD 7])
  (h5 : 3^5 ≡ 5 [MOD 7])
  (h6 : 3^6 ≡ 1 [MOD 7]) :
  3^100 ≡ 4 [MOD 7] :=
by
  sorry

end remainder_of_3_pow_100_mod_7_is_4_l207_207892


namespace parallel_lines_condition_l207_207382

theorem parallel_lines_condition (a : ℝ) :
  (∀ x y : ℝ, (a * x + 2 * y - 1 = 0) → (x + (a + 1) * y + 4 = 0) → a = 1) ↔
  (∀ x y : ℝ, (a = 1 ∧ a * x + 2 * y - 1 = 0 → x + (a + 1) * y + 4 = 0) ∨
   (a ≠ 1 ∧ a = -2 ∧ a * x + 2 * y - 1 ≠ 0 → x + (a + 1) * y + 4 ≠ 0)) :=
by
  sorry

end parallel_lines_condition_l207_207382


namespace determine_N_l207_207266

theorem determine_N (N : ℕ) :
    995 + 997 + 999 + 1001 + 1003 = 5005 - N → N = 5 := 
by 
  sorry

end determine_N_l207_207266


namespace sin_sum_triangle_l207_207205

theorem sin_sum_triangle (α β γ : ℝ) (h : α + β + γ = Real.pi) : 
  Real.sin α + Real.sin β + Real.sin γ ≤ (3 * Real.sqrt 3) / 2 :=
sorry

end sin_sum_triangle_l207_207205


namespace weight_of_pants_l207_207787

def weight_socks := 2
def weight_underwear := 4
def weight_shirt := 5
def weight_shorts := 8
def total_allowed := 50

def weight_total (num_shirts num_shorts num_socks num_underwear : Nat) :=
  num_shirts * weight_shirt + num_shorts * weight_shorts + num_socks * weight_socks + num_underwear * weight_underwear

def items_in_wash := weight_total 2 1 3 4

theorem weight_of_pants :
  let weight_pants := total_allowed - items_in_wash
  weight_pants = 10 :=
by
  sorry

end weight_of_pants_l207_207787


namespace expected_participants_2008_l207_207731

theorem expected_participants_2008 (initial_participants : ℕ) (annual_increase_rate : ℝ) :
  initial_participants = 1000 ∧ annual_increase_rate = 1.25 →
  (initial_participants * annual_increase_rate ^ 3) = 1953.125 :=
by
  sorry

end expected_participants_2008_l207_207731


namespace find_a_l207_207450

-- Define the function f based on the given conditions
noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
if x < 0 then 2^x - a * x else -2^(-x) - a * x

-- Define the fact that f is an odd function
def is_odd_function (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f x = -f (-x)

-- State the main theorem that needs to be proven
theorem find_a (a : ℝ) :
  (is_odd_function (f a)) ∧ (f a 2 = 2) → a = -9 / 8 :=
by
  sorry

end find_a_l207_207450


namespace probability_A_level_l207_207432

theorem probability_A_level (p_B : ℝ) (p_C : ℝ) (h_B : p_B = 0.03) (h_C : p_C = 0.01) : 
  (1 - (p_B + p_C)) = 0.96 :=
by
  -- Proof is omitted
  sorry

end probability_A_level_l207_207432


namespace remaining_apples_l207_207355

-- Define the initial number of apples
def initialApples : ℕ := 356

-- Define the number of apples given away as a mixed number converted to a fraction
def applesGivenAway : ℚ := 272 + 3/5

-- Prove that the remaining apples after giving away are 83
theorem remaining_apples
  (initialApples : ℕ)
  (applesGivenAway : ℚ) :
  initialApples - applesGivenAway = 83 := 
sorry

end remaining_apples_l207_207355


namespace find_a5_div_a7_l207_207183

variable {a : ℕ → ℝ}
variable {q : ℝ}

-- The sequence {aₙ} is a positive geometric sequence.
axiom geo_seq (n : ℕ) : a (n + 1) = a n * q
axiom pos_seq (n : ℕ) : 0 < a n

-- Given conditions
axiom a2a8_eq_6 : a 2 * a 8 = 6
axiom a4_plus_a6_eq_5 : a 4 + a 6 = 5
axiom decreasing_seq (n : ℕ) : a (n + 1) < a n

theorem find_a5_div_a7 : a 5 / a 7 = 3 / 2 := 
sorry

end find_a5_div_a7_l207_207183


namespace total_amount_distributed_l207_207571

theorem total_amount_distributed (A : ℝ) :
  (∀ A, (A / 14 = A / 18 + 80) → A = 5040) :=
by
  sorry

end total_amount_distributed_l207_207571


namespace simplify_expr1_simplify_expr2_simplify_expr3_l207_207700

theorem simplify_expr1 (y : ℤ) (hy : y = 2) : -3 * y^2 - 6 * y + 2 * y^2 + 5 * y = -6 := 
by sorry

theorem simplify_expr2 (a : ℤ) (ha : a = -2) : 15 * a^2 * (-4 * a^2 + (6 * a - a^2) - 3 * a) = -1560 :=
by sorry

theorem simplify_expr3 (x y : ℤ) (h1 : x * y = 2) (h2 : x + y = 3) : (3 * x * y + 10 * y) + (5 * x - (2 * x * y + 2 * y - 3 * x)) = 26 :=
by sorry

end simplify_expr1_simplify_expr2_simplify_expr3_l207_207700


namespace Daisy_lunch_vs_breakfast_l207_207654

noncomputable def breakfast_cost : ℝ := 2.0 + 3.0 + 4.0 + 3.5
noncomputable def lunch_cost_before_service_charge : ℝ := 3.75 + 5.75 + 1.0
noncomputable def service_charge : ℝ := 0.10 * lunch_cost_before_service_charge
noncomputable def total_lunch_cost : ℝ := lunch_cost_before_service_charge + service_charge

theorem Daisy_lunch_vs_breakfast : total_lunch_cost - breakfast_cost = -0.95 := by
  sorry

end Daisy_lunch_vs_breakfast_l207_207654


namespace sum_of_reflected_midpoint_coords_l207_207081

theorem sum_of_reflected_midpoint_coords (P R : ℝ × ℝ) 
  (hP : P = (2, 1)) (hR : R = (12, 15)) :
  let M := ((P.1 + R.1) / 2, (P.2 + R.2) / 2)
  let P' := (-P.1, P.2)
  let R' := (-R.1, R.2)
  let M' := ((P'.1 + R'.1) / 2, (P'.2 + R'.2) / 2)
  M'.1 + M'.2 = 1 :=
by
  sorry

end sum_of_reflected_midpoint_coords_l207_207081


namespace arithmetic_sequence_correct_l207_207267

-- Define the conditions
def last_term_eq_num_of_terms (a l n : Int) : Prop := l = n
def common_difference (d : Int) : Prop := d = 5
def sum_of_sequence (n a S : Int) : Prop :=
  S = n * (2 * a + (n - 1) * 5) / 2

-- The target arithmetic sequence
def seq : List Int := [-7, -2, 3]
def first_term : Int := -7
def num_terms : Int := 3
def sum_of_seq : Int := -6

-- Proof statement
theorem arithmetic_sequence_correct :
  last_term_eq_num_of_terms first_term seq.length num_terms ∧
  common_difference 5 ∧
  sum_of_sequence seq.length first_term sum_of_seq →
  seq = [-7, -2, 3] :=
sorry

end arithmetic_sequence_correct_l207_207267


namespace f_properties_l207_207943

noncomputable def f : ℕ → ℕ := sorry

theorem f_properties (f : ℕ → ℕ) :
  (∀ x y : ℕ, x > 0 → y > 0 → f (x * y) = f x + f y) →
  (f 10 = 16) →
  (f 40 = 24) →
  (f 3 = 5) →
  (f 800 = 44) :=
by
  intros h1 h2 h3 h4
  sorry

end f_properties_l207_207943


namespace Kelsey_watched_537_videos_l207_207345

-- Definitions based on conditions
def total_videos : ℕ := 1222
def delilah_videos : ℕ := 78

-- Declaration of variables representing the number of videos each friend watched
variables (Kelsey Ekon Uma Ivan Lance : ℕ)

-- Conditions from the problem
def cond1 : Kelsey = 3 * Ekon := sorry
def cond2 : Ekon = Uma - 23 := sorry
def cond3 : Uma = 2 * Ivan := sorry
def cond4 : Lance = Ivan + 19 := sorry
def cond5 : delilah_videos = 78 := sorry
def cond6 := Kelsey + Ekon + Uma + Ivan + Lance + delilah_videos = total_videos

-- The theorem to prove
theorem Kelsey_watched_537_videos : Kelsey = 537 :=
  by
  sorry

end Kelsey_watched_537_videos_l207_207345


namespace carson_air_per_pump_l207_207413

-- Define the conditions
def total_air_needed : ℝ := 2 * 500 + 0.6 * 500 + 0.3 * 500

def total_pumps : ℕ := 29

-- Proof problem statement
theorem carson_air_per_pump : total_air_needed / total_pumps = 50 := by
  sorry

end carson_air_per_pump_l207_207413


namespace two_vectors_less_than_45_deg_angle_l207_207703

theorem two_vectors_less_than_45_deg_angle (n : ℕ) (h : n = 30) (v : Fin n → ℝ → ℝ → ℝ) :
  ∃ i j : Fin n, i ≠ j ∧ ∃ θ : ℝ, θ < (45 * Real.pi / 180) :=
  sorry

end two_vectors_less_than_45_deg_angle_l207_207703


namespace math_problem_l207_207834

-- Definitions for increasing function and periodic function
def increasing (f : ℝ → ℝ) : Prop := ∀ x y : ℝ, x < y → f x ≤ f y
def periodic (f : ℝ → ℝ) (T : ℝ) : Prop := ∀ x : ℝ, f (x + T) = f x

-- The main theorem statement
theorem math_problem (f g h : ℝ → ℝ) (T : ℝ) :
  (∀ x y : ℝ, x < y → f x + g x ≤ f y + g y) ∧ (∀ x y : ℝ, x < y → f x + h x ≤ f y + h y) ∧ (∀ x y : ℝ, x < y → g x + h x ≤ g y + h y) → 
  ¬(increasing g) ∧
  (∀ x : ℝ, f (x + T) + g (x + T) = f x + g x ∧ f (x + T) + h (x + T) = f x + h x ∧ g (x + T) + h (x + T) = g x + h x) → 
  increasing f ∧ increasing g ∧ increasing h :=
sorry

end math_problem_l207_207834


namespace range_of_m_value_of_x_l207_207532

noncomputable def a : ℝ := 3 / 2

noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log a

-- Statement for the range of m
theorem range_of_m :
  ∀ m : ℝ, f (3 * m - 2) < f (2 * m + 5) ↔ (2 / 3) < m ∧ m < 7 :=
by
  intro m
  sorry

-- Value of x
theorem value_of_x :
  ∃ x : ℝ, f (x - 2 / x) = Real.log (7 / 2) / Real.log (3 / 2) ∧ x > 0 ∧ x = 4 :=
by
  use 4
  sorry

end range_of_m_value_of_x_l207_207532


namespace find_y_perpendicular_l207_207604

theorem find_y_perpendicular (y : ℝ) (A B : ℝ × ℝ) (a : ℝ × ℝ)
  (hA : A = (-1, 2))
  (hB : B = (2, y))
  (ha : a = (2, 1))
  (h_perp : (B.1 - A.1) * a.1 + (B.2 - A.2) * a.2 = 0) :
  y = -4 :=
sorry

end find_y_perpendicular_l207_207604


namespace total_candies_l207_207409

def candies_in_boxes (num_boxes: Nat) (pieces_per_box: Nat) : Nat :=
  num_boxes * pieces_per_box

theorem total_candies :
  candies_in_boxes 3 6 + candies_in_boxes 5 8 + candies_in_boxes 4 10 = 98 := by
  sorry

end total_candies_l207_207409


namespace min_max_values_l207_207411

noncomputable def expression (x₁ x₂ x₃ x₄ : ℝ) : ℝ :=
  ( (x₁ ^ 2 / x₂) + (x₂ ^ 2 / x₃) + (x₃ ^ 2 / x₄) + (x₄ ^ 2 / x₁) ) /
  ( x₁ + x₂ + x₃ + x₄ )

theorem min_max_values
  (a b : ℝ) (x₁ x₂ x₃ x₄ : ℝ)
  (h₀ : 0 < a) (h₁ : a < b)
  (h₂ : a ≤ x₁) (h₃ : x₁ ≤ b)
  (h₄ : a ≤ x₂) (h₅ : x₂ ≤ b)
  (h₆ : a ≤ x₃) (h₇ : x₃ ≤ b)
  (h₈ : a ≤ x₄) (h₉ : x₄ ≤ b) :
  expression x₁ x₂ x₃ x₄ ≥ 1 / b ∧ expression x₁ x₂ x₃ x₄ ≤ 1 / a :=
  sorry

end min_max_values_l207_207411


namespace Marta_max_piles_l207_207328

theorem Marta_max_piles (a b c : ℕ) (ha : a = 42) (hb : b = 60) (hc : c = 90) : 
  Nat.gcd (Nat.gcd a b) c = 6 := by
  rw [ha, hb, hc]
  have h : Nat.gcd (Nat.gcd 42 60) 90 = Nat.gcd 6 90 := by sorry
  exact h    

end Marta_max_piles_l207_207328


namespace sum_fraction_series_eq_l207_207212

noncomputable def sum_fraction_series : ℝ :=
  ∑' n, (1 / (n * (n + 3)))

theorem sum_fraction_series_eq :
  sum_fraction_series = 11 / 18 :=
sorry

end sum_fraction_series_eq_l207_207212


namespace sections_capacity_l207_207080

theorem sections_capacity (total_people sections : ℕ) 
  (h1 : total_people = 984) 
  (h2 : sections = 4) : 
  total_people / sections = 246 := 
by
  sorry

end sections_capacity_l207_207080


namespace maria_savings_l207_207963

-- Conditions
def sweater_cost : ℕ := 30
def scarf_cost : ℕ := 20
def num_sweaters : ℕ := 6
def num_scarves : ℕ := 6
def savings : ℕ := 500

-- The proof statement
theorem maria_savings : savings - (num_sweaters * sweater_cost + num_scarves * scarf_cost) = 200 :=
by
  sorry

end maria_savings_l207_207963


namespace charlie_coins_worth_44_cents_l207_207944

-- Definitions based on the given conditions
def total_coins := 17
def p_eq_n_plus_2 (p n : ℕ) := p = n + 2

-- The main theorem stating the problem and the expected answer
theorem charlie_coins_worth_44_cents (p n : ℕ) (h1 : p + n = total_coins) (h2 : p_eq_n_plus_2 p n) :
  (7 * 5 + p * 1 = 44) :=
sorry

end charlie_coins_worth_44_cents_l207_207944


namespace problem_solution_l207_207624

noncomputable def alpha : ℝ := (3 + Real.sqrt 13) / 2
noncomputable def beta  : ℝ := (3 - Real.sqrt 13) / 2

theorem problem_solution : 7 * alpha ^ 4 + 10 * beta ^ 3 = 1093 :=
by
  -- Prove roots relation
  have hr1 : alpha * alpha - 3 * alpha - 1 = 0 := by sorry
  have hr2 : beta * beta - 3 * beta - 1 = 0 := by sorry
  -- Proceed to prove the required expression
  sorry

end problem_solution_l207_207624


namespace correct_number_of_paths_l207_207628

-- Define the number of paths for each segment.
def paths_A_to_B : ℕ := 2
def paths_B_to_D : ℕ := 2
def paths_D_to_C : ℕ := 2
def direct_path_A_to_C : ℕ := 1

-- Define the function to calculate the total paths from A to C.
def total_paths_A_to_C : ℕ :=
  (paths_A_to_B * paths_B_to_D * paths_D_to_C) + direct_path_A_to_C

-- Prove that the total number of paths from A to C is 9.
theorem correct_number_of_paths : total_paths_A_to_C = 9 := by
  -- This is where the proof would go, but it is not required for this task.
  sorry

end correct_number_of_paths_l207_207628


namespace general_solution_of_diff_eq_l207_207575

theorem general_solution_of_diff_eq {C1 C2 : ℝ} (y : ℝ → ℝ) (x : ℝ) :
  (∀ x, y x = C1 * Real.exp (-x) + C2 * Real.exp (-2 * x) + x^2 - 5 * x - 2) →
  (∀ x, (deriv (deriv y)) x + 3 * (deriv y) x + 2 * y x = 2 * x^2 - 4 * x - 17) :=
by
  intro hy
  sorry

end general_solution_of_diff_eq_l207_207575


namespace solve_for_y_l207_207691

theorem solve_for_y (y : ℕ) (h : 5 * (2^y) = 320) : y = 6 := 
by 
  sorry

end solve_for_y_l207_207691


namespace smallest_b_l207_207296

theorem smallest_b (b : ℝ) : b^2 - 16 * b + 63 ≤ 0 → (∃ b : ℝ, b = 7) :=
sorry

end smallest_b_l207_207296


namespace expression_equiv_l207_207854

theorem expression_equiv :
  (4 + 5) * (4^2 + 5^2) * (4^4 + 5^4) * (4^8 + 5^8) * (4^16 + 5^16) * (4^32 + 5^32) * (4^64 + 5^64) = 5^128 - 4^128 :=
by
  sorry

end expression_equiv_l207_207854


namespace area_of_large_rectangle_l207_207483

noncomputable def areaEFGH : ℕ :=
  let shorter_side := 3
  let longer_side := 2 * shorter_side
  let width_EFGH := shorter_side + shorter_side
  let length_EFGH := longer_side + longer_side
  width_EFGH * length_EFGH

theorem area_of_large_rectangle :
  areaEFGH = 72 := by
  sorry

end area_of_large_rectangle_l207_207483


namespace weight_increase_percentage_l207_207400

theorem weight_increase_percentage :
  ∀ (x : ℝ), (2 * x * 1.1 + 5 * x * 1.17 = 82.8) →
    ((82.8 - (2 * x + 5 * x)) / (2 * x + 5 * x)) * 100 = 15.06 := 
by 
  intro x 
  intro h
  sorry

end weight_increase_percentage_l207_207400


namespace second_odd_integer_l207_207681

theorem second_odd_integer (n : ℤ) (h : (n - 2) + (n + 2) = 128) : n = 64 :=
by
  sorry

end second_odd_integer_l207_207681


namespace five_digit_palindromes_count_l207_207468

theorem five_digit_palindromes_count : ∃ n : ℕ, n = 900 ∧
  ∀ (a b c : ℕ),
    1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 →
    ∃ (x : ℕ), x = a * 10^4 + b * 10^3 + c * 10^2 + b * 10 + a := sorry

end five_digit_palindromes_count_l207_207468


namespace greatest_integer_prime_l207_207093

def is_prime (n : ℤ) : Prop :=
  n > 1 ∧ ∀ m : ℤ, m > 0 → m < n → n % m ≠ 0

theorem greatest_integer_prime (x : ℤ) :
  is_prime (|8 * x ^ 2 - 56 * x + 21|) → ∀ y : ℤ, (is_prime (|8 * y ^ 2 - 56 * y + 21|) → y ≤ x) :=
by
  sorry

end greatest_integer_prime_l207_207093


namespace xiao_zhang_winning_probability_max_expected_value_l207_207685

-- Definitions for the conditions
variables (a b c : ℕ)
variable (h_sum : a + b + c = 6)

-- Main theorem statement 1: Probability of Xiao Zhang winning
theorem xiao_zhang_winning_probability (h_sum : a + b + c = 6) :
  (3 * a + 2 * b + c) / 36 = a / 6 * 3 / 6 + b / 6 * 2 / 6 + c / 6 * 1 / 6 :=
sorry

-- Main theorem statement 2: Maximum expected value of Xiao Zhang's score
theorem max_expected_value (h_sum : a + b + c = 6) :
  (3 * a + 4 * b + 3 * c) / 36 = (1 / 2 + b / 36) →  (a = 0 ∧ b = 6 ∧ c = 0) :=
sorry

end xiao_zhang_winning_probability_max_expected_value_l207_207685


namespace rachel_total_problems_l207_207966

theorem rachel_total_problems
    (problems_per_minute : ℕ)
    (minutes_before_bed : ℕ)
    (problems_next_day : ℕ) 
    (h1 : problems_per_minute = 5) 
    (h2 : minutes_before_bed = 12) 
    (h3 : problems_next_day = 16) : 
    problems_per_minute * minutes_before_bed + problems_next_day = 76 :=
by
  sorry

end rachel_total_problems_l207_207966


namespace truck_travel_yards_l207_207763

variables (b t : ℝ)

theorem truck_travel_yards : 
  (2 * (2 * b / 7) / (2 * t)) * 240 / 3 = (80 * b) / (7 * t) :=
by 
  sorry

end truck_travel_yards_l207_207763


namespace total_yellow_marbles_l207_207087

theorem total_yellow_marbles (mary_marbles : ℕ) (joan_marbles : ℕ) (h1 : mary_marbles = 9) (h2 : joan_marbles = 3) : mary_marbles + joan_marbles = 12 := 
by 
  sorry

end total_yellow_marbles_l207_207087


namespace proof_equilateral_inscribed_circle_l207_207214

variables {A B C : Type*}
variables (r : ℝ) (D : ℝ)

def is_equilateral_triangle (A B C : Type*) : Prop := 
  -- Define the equilateral condition, where all sides are equal
  true

def is_inscribed_circle_radius (D r : ℝ) : Prop := 
  -- Define the property that D is the center and r is the radius 
  true

def distance_center_to_vertex (D r x : ℝ) : Prop := 
  x = 3 * r

theorem proof_equilateral_inscribed_circle 
  (A B C : Type*) 
  (r D : ℝ) 
  (h1 : is_equilateral_triangle A B C) 
  (h2 : is_inscribed_circle_radius D r) : 
  distance_center_to_vertex D r (1 / 16) :=
by sorry

end proof_equilateral_inscribed_circle_l207_207214


namespace arrangement_with_one_ball_per_box_arrangement_with_one_empty_box_l207_207880

-- Definition of the first proof problem
theorem arrangement_with_one_ball_per_box:
  ∃ n : ℕ, n = 24 ∧ 
    -- Number of ways to arrange 4 different balls in 4 boxes such that each box has exactly one ball
    n = Nat.factorial 4 :=
by sorry

-- Definition of the second proof problem
theorem arrangement_with_one_empty_box:
  ∃ n : ℕ, n = 144 ∧ 
    -- Number of ways to arrange 4 different balls in 4 boxes such that exactly one box is empty
    n = Nat.choose 4 2 * Nat.factorial 3 :=
by sorry

end arrangement_with_one_ball_per_box_arrangement_with_one_empty_box_l207_207880


namespace range_of_m_l207_207666

theorem range_of_m (x y m : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + y = 3) (h4 : ∀ x y, x > 0 → y > 0 → x + y = 3 → (4 / (x + 1) + 16 / y > m^2 - 3 * m + 11)) : 1 < m ∧ m < 2 :=
by
  sorry

end range_of_m_l207_207666


namespace square_of_binomial_l207_207198

theorem square_of_binomial (a : ℝ) : 16 * x^2 + 32 * x + a = (4 * x + 4)^2 :=
by
  sorry

end square_of_binomial_l207_207198


namespace angle_CAB_in_regular_hexagon_l207_207147

-- Define a regular hexagon
structure regular_hexagon (A B C D E F : Type) :=
  (interior_angle : ℝ)
  (all_sides_equal : A = B ∧ B = C ∧ C = D ∧ D = E ∧ E = F)
  (all_angles_equal : interior_angle = 120)

-- Define the problem of finding the angle CAB
theorem angle_CAB_in_regular_hexagon 
  (A B C D E F : Type)
  (hex : regular_hexagon A B C D E F)
  (diagonal_AC : A = C)
  : ∃ (CAB : ℝ), CAB = 30 :=
sorry

end angle_CAB_in_regular_hexagon_l207_207147


namespace expression_equiv_l207_207926

theorem expression_equiv (x y : ℝ) (h : x ≠ 0 ∧ y ≠ 0) :
  ((x^4 + 1) / x^2) * ((y^4 + 1) / y^2) + ((x^4 - 1) / y^2) * ((y^4 - 1) / x^2) =
  2*x^2*y^2 + 2/(x^2*y^2) :=
by 
  sorry

end expression_equiv_l207_207926


namespace black_piece_is_option_C_l207_207486

-- Definitions for the problem conditions
def rectangular_prism (cubes : Nat) := cubes = 16
def block (small_cubes : Nat) := small_cubes = 4
def piece_containing_black_shape_is_partially_seen (rows : Nat) := rows = 2

-- Hypotheses and conditions
variable (rect_prism : Nat) (block1 block2 block3 block4 : Nat)
variable (visibility_block1 visibility_block2 visibility_block3 : Bool)
variable (visible_in_back_row : Bool)

-- Given conditions based on the problem statement
axiom h1 : rectangular_prism rect_prism
axiom h2 : block block1
axiom h3 : block block2
axiom h4 : block block3
axiom h5 : block block4
axiom h6 : visibility_block1 = true
axiom h7 : visibility_block2 = true
axiom h8 : visibility_block3 = true
axiom h9 : visible_in_back_row = true

-- Prove the configuration matches Option C
theorem black_piece_is_option_C :
  ∀ (config : Char), (config = 'C') :=
by
  intros
  -- Proof incomplete intentionally.
  sorry

end black_piece_is_option_C_l207_207486


namespace remainder_of_3_pow_800_mod_17_l207_207824

theorem remainder_of_3_pow_800_mod_17 : (3^800) % 17 = 1 := by
  sorry

end remainder_of_3_pow_800_mod_17_l207_207824


namespace mean_score_l207_207826

theorem mean_score (μ σ : ℝ)
  (h1 : 86 = μ - 7 * σ)
  (h2 : 90 = μ + 3 * σ) : μ = 88.8 := by
  -- Proof steps are not included as per requirements.
  sorry

end mean_score_l207_207826


namespace no_solution_exists_l207_207295

theorem no_solution_exists (x y : ℝ) : ¬ ((2 * x - 3 * y = 8) ∧ (6 * y - 4 * x = 9)) :=
sorry

end no_solution_exists_l207_207295


namespace birth_year_1849_l207_207246

theorem birth_year_1849 (x : ℕ) (h1 : 1850 ≤ x^2 - 2 * x + 1) (h2 : x^2 - 2 * x + 1 < 1900) (h3 : x^2 - x + 1 = x) : x = 44 ↔ x^2 - 2 * x + 1 = 1849 := 
sorry

end birth_year_1849_l207_207246


namespace problem_l207_207438

theorem problem 
  (x y : ℝ)
  (h1 : 3 * x + y = 7)
  (h2 : x + 3 * y = 8) : 
  10 * x ^ 2 + 13 * x * y + 10 * y ^ 2 = 113 := 
sorry

end problem_l207_207438


namespace valid_pairs_iff_l207_207020

noncomputable def valid_pairs (a b : ℝ) : Prop :=
  ∃ n : ℕ, n > 0 ∧ a * (⌊ b * n ⌋ : ℝ) = b * (⌊ a * n ⌋ : ℝ)

theorem valid_pairs_iff (a b : ℝ) : valid_pairs a b ↔
  (a = 0 ∨ b = 0 ∨ a = b ∨ (∃ (m n : ℤ), a = m ∧ b = n)) :=
by sorry

end valid_pairs_iff_l207_207020


namespace sin_inv_tan_eq_l207_207077

open Real

theorem sin_inv_tan_eq :
  let a := arcsin (4/5)
  let b := arctan 3
  sin (a + b) = (13 * sqrt 10) / 50 := 
by
  let a := arcsin (4/5)
  let b := arctan 3
  sorry

end sin_inv_tan_eq_l207_207077


namespace geometric_sequence_a_formula_l207_207614

noncomputable def a (n : ℕ) : ℤ :=
  if n = 1 then 1
  else if n = 2 then 2
  else n - 2

noncomputable def b (n : ℕ) : ℤ :=
  a (n + 1) - a n

theorem geometric_sequence (n : ℕ) (h : n ≥ 2) : 
  b n = (-1) * b (n - 1) := 
  sorry

theorem a_formula (n : ℕ) : 
  a n = (-1) ^ (n - 1) := 
  sorry

end geometric_sequence_a_formula_l207_207614


namespace remainder_zero_l207_207442

theorem remainder_zero (x : ℂ) 
  (h : x^5 + x^4 + x^3 + x^2 + x + 1 = 0) : 
  x^55 + x^44 + x^33 + x^22 + x^11 + 1 = 0 := 
by 
  sorry

end remainder_zero_l207_207442


namespace find_term_of_sequence_l207_207496

theorem find_term_of_sequence :
  ∀ (a d n : ℤ), a = -5 → d = -4 → (-4)*n + 1 = -401 → n = 100 :=
by
  intros a d n h₁ h₂ h₃
  sorry

end find_term_of_sequence_l207_207496


namespace arithmetic_sequence_sum_l207_207406

theorem arithmetic_sequence_sum (a : ℕ → ℕ) (d : ℕ) (h1 : a 1 = 2) (h2 : a 2 + a 3 = 13) :
  a 4 + a 5 + a 6 = 42 :=
sorry

end arithmetic_sequence_sum_l207_207406


namespace original_population_correct_l207_207748

def original_population_problem :=
  let original_population := 6731
  let final_population := 4725
  let initial_disappeared := 0.10 * original_population
  let remaining_after_disappearance := original_population - initial_disappeared
  let left_after_remaining := 0.25 * remaining_after_disappearance
  let remaining_after_leaving := remaining_after_disappearance - left_after_remaining
  let disease_affected := 0.05 * original_population
  let disease_died := 0.02 * disease_affected
  let disease_migrated := 0.03 * disease_affected
  let remaining_after_disease := remaining_after_leaving - (disease_died + disease_migrated)
  let moved_to_village := 0.04 * remaining_after_disappearance
  let total_after_moving := remaining_after_disease + moved_to_village
  let births := 0.008 * original_population
  let deaths := 0.01 * original_population
  let final_population_calculated := total_after_moving + (births - deaths)
  final_population_calculated = final_population

theorem original_population_correct :
  original_population_problem ↔ True :=
by
  sorry

end original_population_correct_l207_207748


namespace average_student_headcount_is_correct_l207_207387

noncomputable def average_student_headcount : ℕ :=
  let a := 11000
  let b := 10200
  let c := 10800
  let d := 11300
  (a + b + c + d) / 4

theorem average_student_headcount_is_correct :
  average_student_headcount = 10825 :=
by
  -- Proof will go here
  sorry

end average_student_headcount_is_correct_l207_207387


namespace f_a1_a3_a5_positive_l207_207150

theorem f_a1_a3_a5_positive (f : ℝ → ℝ) (a : ℕ → ℝ)
  (hf_odd : ∀ x, f (-x) = - f x)
  (hf_mono : ∀ x y, x < y → f x < f y)
  (ha_arith : ∃ d, ∀ n, a (n + 1) = a n + d)
  (ha3_pos : 0 < a 3) :
  0 < f (a 1) + f (a 3) + f (a 5) :=
sorry

end f_a1_a3_a5_positive_l207_207150


namespace largest_five_digit_number_with_product_120_l207_207512

def is_five_digit (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000

def prod_of_digits (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.foldr (· * ·) 1

def max_five_digit_prod_120 : ℕ := 85311

theorem largest_five_digit_number_with_product_120 :
  is_five_digit max_five_digit_prod_120 ∧ prod_of_digits max_five_digit_prod_120 = 120 :=
by
  sorry

end largest_five_digit_number_with_product_120_l207_207512


namespace abs_value_x_minus_2_plus_x_plus_3_ge_4_l207_207663

theorem abs_value_x_minus_2_plus_x_plus_3_ge_4 :
  ∀ x : ℝ, (|x - 2| + |x + 3| ≥ 4) ↔ (x ≤ - (5 / 2)) := 
sorry

end abs_value_x_minus_2_plus_x_plus_3_ge_4_l207_207663


namespace sufficient_but_not_necessary_condition_l207_207003

-- Define the condition
variable (a : ℝ)

-- Theorem statement: $a > 0$ is a sufficient but not necessary condition for $a^2 > 0$
theorem sufficient_but_not_necessary_condition : 
  (a > 0 → a^2 > 0) ∧ (¬ (a > 0) → a^2 > 0) :=
  by
    sorry

end sufficient_but_not_necessary_condition_l207_207003


namespace mary_total_cards_l207_207027

def mary_initial_cards := 33
def torn_cards := 6
def cards_given_by_sam := 23

theorem mary_total_cards : mary_initial_cards - torn_cards + cards_given_by_sam = 50 :=
  by
    sorry

end mary_total_cards_l207_207027


namespace triangle_inequality_property_l207_207412

noncomputable def perimeter (a b c : ℝ) : ℝ := a + b + c

noncomputable def circumradius (a b c : ℝ) (A B C: ℝ) : ℝ := 
  (a * b * c) / (4 * Real.sqrt (A * B * C))

noncomputable def inradius (a b c : ℝ) (A B C: ℝ) : ℝ := 
  Real.sqrt (A * B * C) * perimeter a b c

theorem triangle_inequality_property (a b c A B C : ℝ)
  (h₁ : ∀ {x}, x > 0)
  (h₂ : A ≠ B)
  (h₃ : B ≠ C)
  (h₄ : C ≠ A) :
  ¬ (perimeter a b c ≤ circumradius a b c A B C + inradius a b c A B C) ∧
  ¬ (perimeter a b c > circumradius a b c A B C + inradius a b c A B C) ∧
  ¬ (perimeter a b c / 6 < circumradius a b c A B C + inradius a b c A B C ∨ 
  circumradius a b c A B C + inradius a b c A B C < 6 * perimeter a b c) :=
sorry

end triangle_inequality_property_l207_207412


namespace johns_change_l207_207240

/-- Define the cost of Slurpees and amount given -/
def cost_per_slurpee : ℕ := 2
def amount_given : ℕ := 20
def slurpees_bought : ℕ := 6

/-- Define the total cost of the Slurpees -/
def total_cost : ℕ := cost_per_slurpee * slurpees_bought

/-- Define the change John gets -/
def change (amount_given total_cost : ℕ) : ℕ := amount_given - total_cost

/-- The statement for Lean 4 that proves the change John gets is $8 given the conditions -/
theorem johns_change : change amount_given total_cost = 8 :=
by 
  -- Rest of the proof omitted
  sorry

end johns_change_l207_207240


namespace intersection_S_T_l207_207299

def set_S : Set ℝ := { x | abs x < 5 }
def set_T : Set ℝ := { x | x^2 + 4*x - 21 < 0 }

theorem intersection_S_T :
  set_S ∩ set_T = { x | -5 < x ∧ x < 3 } :=
sorry

end intersection_S_T_l207_207299


namespace distributi_l207_207702

def number_of_distributions (spots : ℕ) (classes : ℕ) (min_spot_per_class : ℕ) : ℕ :=
  Nat.choose (spots - min_spot_per_class * classes + (classes - 1)) (classes - 1)

theorem distributi.on_of_10_spots (A B C : ℕ) (hA : A ≥ 1) (hB : B ≥ 1) (hC : C ≥ 1) 
(h_total : A + B + C = 10) : number_of_distributions 10 3 1 = 36 :=
by
  sorry

end distributi_l207_207702


namespace claire_gift_card_balance_l207_207322

/--
Claire has a $100 gift card to her favorite coffee shop.
A latte costs $3.75.
A croissant costs $3.50.
Claire buys one latte and one croissant every day for a week.
Claire buys 5 cookies, each costing $1.25.

Prove that the amount of money left on Claire's gift card after a week is $43.00.
-/
theorem claire_gift_card_balance :
  let initial_balance : ℝ := 100
  let latte_cost : ℝ := 3.75
  let croissant_cost : ℝ := 3.50
  let daily_expense : ℝ := latte_cost + croissant_cost
  let weekly_expense : ℝ := daily_expense * 7
  let cookie_cost : ℝ := 1.25
  let total_cookie_expense : ℝ := cookie_cost * 5
  let total_expense : ℝ := weekly_expense + total_cookie_expense
  let remaining_balance : ℝ := initial_balance - total_expense
  remaining_balance = 43 :=
by
  sorry

end claire_gift_card_balance_l207_207322


namespace find_y_l207_207929

theorem find_y (a b : ℝ) (y : ℝ) (h0 : b ≠ 0) (h1 : (3 * a)^(2 * b) = a^b * y^b) : y = 9 * a := by
  sorry

end find_y_l207_207929


namespace find_number_l207_207846

theorem find_number (x : ℝ) (h : 0.2 * x = 0.3 * 120 + 80) : x = 580 :=
by
  sorry

end find_number_l207_207846


namespace cole_trip_time_l207_207262

theorem cole_trip_time 
  (D : ℕ) -- The distance D from home to work
  (T_total : ℕ) -- The total round trip time in hours
  (S1 S2 : ℕ) -- The average speeds (S1, S2) in km/h
  (h1 : S1 = 80) -- The average speed from home to work
  (h2 : S2 = 120) -- The average speed from work to home
  (h3 : T_total = 2) -- The total round trip time is 2 hours
  : (D : ℝ) / 80 + (D : ℝ) / 120 = 2 →
    (T_work : ℝ) = (D : ℝ) / 80 →
    (T_work * 60) = 72 := 
by {
  sorry
}

end cole_trip_time_l207_207262


namespace john_remaining_income_l207_207542

/-- 
  Mr. John's monthly income is $2000, and he spends 5% of his income on public transport.
  Prove that after deducting his monthly transport fare, his remaining income is $1900.
-/
theorem john_remaining_income : 
  let income := 2000 
  let transport_percent := 5 
  let transport_fare := income * transport_percent / 100 
  income - transport_fare = 1900 := 
by 
  let income := 2000 
  let transport_percent := 5 
  let transport_fare := income * transport_percent / 100 
  have transport_fare_eq : transport_fare = 100 := by sorry
  have remaining_income_eq : income - transport_fare = 1900 := by sorry
  exact remaining_income_eq

end john_remaining_income_l207_207542


namespace third_number_eq_l207_207873

theorem third_number_eq :
  ∃ x : ℝ, (0.625 * 0.0729 * x) / (0.0017 * 0.025 * 8.1) = 382.5 ∧ x = 2.33075 := 
by
  sorry

end third_number_eq_l207_207873


namespace triangle_is_isosceles_l207_207097

theorem triangle_is_isosceles
  (α β γ x y z w : ℝ)
  (h1 : α + β + γ = 180)
  (h2 : α + β = x)
  (h3 : β + γ = y)
  (h4 : γ + α = z)
  (h5 : x + y + z + w = 360) : 
  (α = β ∧ β = γ) ∨ (α = γ ∧ γ = β) ∨ (β = α ∧ α = γ) := by
  sorry

end triangle_is_isosceles_l207_207097


namespace inequality_solution_l207_207044

noncomputable def operation (a b : ℝ) : ℝ := (a + 3 * b) - a * b

theorem inequality_solution (x : ℝ) : operation 5 x < 13 → x > -4 := by
  sorry

end inequality_solution_l207_207044


namespace smallest_perimeter_l207_207810

theorem smallest_perimeter (m n : ℕ) 
  (h1 : (m - 4) * (n - 4) = 8) 
  (h2 : ∀ k l : ℕ, (k - 4) * (l - 4) = 8 → 2 * k + 2 * l ≥ 2 * m + 2 * n) : 
  (m = 6 ∧ n = 8) ∨ (m = 8 ∧ n = 6) :=
sorry

end smallest_perimeter_l207_207810


namespace initial_strawberries_l207_207516

-- Define the conditions
def strawberries_eaten : ℝ := 42.0
def strawberries_left : ℝ := 36.0

-- State the theorem
theorem initial_strawberries :
  strawberries_eaten + strawberries_left = 78 :=
by
  sorry

end initial_strawberries_l207_207516


namespace meal_cost_l207_207838

variable (s c p : ℝ)

axiom cond1 : 5 * s + 8 * c + p = 5.00
axiom cond2 : 7 * s + 12 * c + p = 7.20
axiom cond3 : 4 * s + 6 * c + 2 * p = 6.00

theorem meal_cost : s + c + p = 1.90 :=
by
  sorry

end meal_cost_l207_207838


namespace second_wrongly_copied_number_l207_207181

theorem second_wrongly_copied_number 
  (avg_err : ℝ) 
  (total_nums : ℕ) 
  (sum_err : ℝ) 
  (first_err_corr : ℝ) 
  (correct_avg : ℝ) 
  (correct_num : ℝ) 
  (second_num_wrong : ℝ) :
  (avg_err = 40.2) → 
  (total_nums = 10) → 
  (sum_err = total_nums * avg_err) → 
  (first_err_corr = 16) → 
  (correct_avg = 40) → 
  (correct_num = 31) → 
  sum_err - first_err_corr + (correct_num - second_num_wrong) = total_nums * correct_avg → 
  second_num_wrong = 17 := 
by 
  intros h_avg h_total h_sum_err h_first_corr h_correct_avg h_correct_num h_corrected_sum 
  sorry

end second_wrongly_copied_number_l207_207181


namespace factor_expression_l207_207300

theorem factor_expression (x : ℝ) : 54 * x^5 - 135 * x^9 = 27 * x^5 * (2 - 5 * x^4) :=
by
  sorry

end factor_expression_l207_207300


namespace olympiad_scores_above_18_l207_207741

theorem olympiad_scores_above_18 
  (n : Nat) 
  (scores : Fin n → ℕ) 
  (h_diff_scores : ∀ i j : Fin n, i ≠ j → scores i ≠ scores j) 
  (h_score_sum : ∀ i j k : Fin n, i ≠ j ∧ i ≠ k ∧ j ≠ k → scores i < scores j + scores k) 
  (h_n : n = 20) : 
  ∀ i : Fin n, scores i > 18 := 
by 
  -- See the proof for the detailed steps.
  sorry

end olympiad_scores_above_18_l207_207741


namespace count_positive_integers_satisfying_conditions_l207_207055

theorem count_positive_integers_satisfying_conditions :
  let condition1 (n : ℕ) := (169 * n) ^ 25 > n ^ 75
  let condition2 (n : ℕ) := n ^ 75 > 3 ^ 150
  ∃ (count : ℕ), count = 3 ∧ (∀ (n : ℕ), (condition1 n) ∧ (condition2 n) → 9 < n ∧ n < 13) :=
by
  sorry

end count_positive_integers_satisfying_conditions_l207_207055


namespace trigonometric_product_identity_l207_207428

theorem trigonometric_product_identity : 
  let cos_40 : Real := Real.cos (Real.pi * 40 / 180)
  let sin_40 : Real := Real.sin (Real.pi * 40 / 180)
  let cos_50 : Real := Real.cos (Real.pi * 50 / 180)
  let sin_50 : Real := Real.sin (Real.pi * 50 / 180)
  (sin_50 = cos_40) → (cos_50 = sin_40) →
  (1 - cos_40⁻¹) * (1 + sin_50⁻¹) * (1 - sin_40⁻¹) * (1 + cos_50⁻¹) = 1 := by
  sorry

end trigonometric_product_identity_l207_207428


namespace basketball_team_count_l207_207204

theorem basketball_team_count :
  (∃ n : ℕ, n = (Nat.choose 13 4) ∧ n = 715) :=
by
  sorry

end basketball_team_count_l207_207204


namespace even_and_nonneg_range_l207_207829

theorem even_and_nonneg_range : 
  (∀ x : ℝ, abs x = abs (-x) ∧ (abs x ≥ 0)) ∧ (∀ x : ℝ, x^2 + abs x = ( (-x)^2) + abs (-x) ∧ (x^2 + abs x ≥ 0)) := sorry

end even_and_nonneg_range_l207_207829


namespace find_m_l207_207367

variables {m : ℝ}
def vec_a : ℝ × ℝ := (1, m)
def vec_b : ℝ × ℝ := (2, 5)
def vec_c : ℝ × ℝ := (m, 3)
def vec_a_plus_c := (1 + m, 3 + m)
def vec_a_minus_b := (1 - 2, m - 5)

theorem find_m (h : (1 + m) * (m - 5) = -1 * (m + 3)) : m = (3 + Real.sqrt 17) / 2 ∨ m = (3 - Real.sqrt 17) / 2 := 
sorry

end find_m_l207_207367


namespace trigonometric_relationship_l207_207931

variable (α β : ℝ)
variable (hα : 0 < α ∧ α < (π / 2))
variable (hβ : 0 < β ∧ β < π)
variable (h : Real.tan α = Real.cos β / (1 - Real.sin β))

theorem trigonometric_relationship : 
    2 * α - β = π / 2 :=
sorry

end trigonometric_relationship_l207_207931


namespace subset_implies_a_ge_2_l207_207159

theorem subset_implies_a_ge_2 (a : ℝ) :
  (∀ x, -1 ≤ x ∧ x ≤ 2 → x ≤ a) → a ≥ 2 :=
by sorry

end subset_implies_a_ge_2_l207_207159


namespace response_rate_percentage_l207_207112

theorem response_rate_percentage (responses_needed : ℕ) (questionnaires_mailed : ℕ)
  (h1 : responses_needed = 240) (h2 : questionnaires_mailed = 400) : 
  (responses_needed : ℝ) / (questionnaires_mailed : ℝ) * 100 = 60 := 
by 
  sorry

end response_rate_percentage_l207_207112


namespace total_students_l207_207331

theorem total_students (boys girls : ℕ) (h_ratio : 5 * girls = 7 * boys) (h_girls : girls = 140) :
  boys + girls = 240 :=
sorry

end total_students_l207_207331


namespace distinct_ints_sum_to_4r_l207_207338

theorem distinct_ints_sum_to_4r 
  (a b c d r : ℤ) 
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h_root : (r - a) * (r - b) * (r - c) * (r - d) = 4) : 
  4 * r = a + b + c + d := 
by sorry

end distinct_ints_sum_to_4r_l207_207338


namespace expected_winnings_correct_l207_207589

def probability_1 := (1:ℚ) / 4
def probability_2 := (1:ℚ) / 4
def probability_3 := (1:ℚ) / 6
def probability_4 := (1:ℚ) / 6
def probability_5 := (1:ℚ) / 8
def probability_6 := (1:ℚ) / 8

noncomputable def expected_winnings : ℚ :=
  (probability_1 + probability_3 + probability_5) * 2 +
  (probability_2 + probability_4) * 4 +
  probability_6 * (-6 + 4)

theorem expected_winnings_correct : expected_winnings = 1.67 := by
  sorry

end expected_winnings_correct_l207_207589


namespace bruce_money_left_to_buy_more_clothes_l207_207414

def calculate_remaining_money 
  (amount_given : ℝ) 
  (shirt_price : ℝ) (num_shirts : ℕ)
  (pants_price : ℝ)
  (sock_price : ℝ) (num_socks : ℕ)
  (belt_original_price : ℝ) (belt_discount : ℝ)
  (total_discount : ℝ) : ℝ := 
let shirts_cost := shirt_price * num_shirts
let socks_cost := sock_price * num_socks
let belt_price := belt_original_price * (1 - belt_discount)
let total_cost := shirts_cost + pants_price + socks_cost + belt_price
let discount_cost := total_cost * total_discount
let final_cost := total_cost - discount_cost
amount_given - final_cost

theorem bruce_money_left_to_buy_more_clothes 
  : calculate_remaining_money 71 5 5 26 3 2 12 0.25 0.10 = 11.60 := 
by
  sorry

end bruce_money_left_to_buy_more_clothes_l207_207414


namespace find_b_plus_m_l207_207133

def matrix_C (b : ℕ) : Matrix (Fin 3) (Fin 3) ℕ :=
  ![
    ![1, 3, b],
    ![0, 1, 5],
    ![0, 0, 1]
  ]

def matrix_RHS : Matrix (Fin 3) (Fin 3) ℕ :=
  ![
    ![1, 27, 3003],
    ![0, 1, 45],
    ![0, 0, 1]
  ]

theorem find_b_plus_m (b m : ℕ) (h : matrix_C b ^ m = matrix_RHS) : b + m = 306 := 
  sorry

end find_b_plus_m_l207_207133


namespace kelly_chickens_l207_207321

theorem kelly_chickens
  (chicken_egg_rate : ℕ)
  (chickens : ℕ)
  (egg_price_per_dozen : ℕ)
  (total_money : ℕ)
  (weeks : ℕ)
  (days_per_week : ℕ)
  (dozen : ℕ)
  (total_eggs_sold : ℕ)
  (total_days : ℕ)
  (total_eggs_laid : ℕ) : 
  chicken_egg_rate = 3 →
  egg_price_per_dozen = 5 →
  total_money = 280 →
  weeks = 4 →
  days_per_week = 7 →
  dozen = 12 →
  total_eggs_sold = total_money / egg_price_per_dozen * dozen →
  total_days = weeks * days_per_week →
  total_eggs_laid = chickens * chicken_egg_rate * total_days →
  total_eggs_sold = total_eggs_laid →
  chickens = 8 :=
by
  intros
  sorry

end kelly_chickens_l207_207321


namespace new_train_travel_distance_l207_207971

-- Definitions of the trains' travel distances
def older_train_distance : ℝ := 180
def new_train_additional_distance_ratio : ℝ := 0.50

-- Proof that the new train can travel 270 miles
theorem new_train_travel_distance
: new_train_additional_distance_ratio * older_train_distance + older_train_distance = 270 := 
by
  sorry

end new_train_travel_distance_l207_207971


namespace same_terminal_side_angle_l207_207933

theorem same_terminal_side_angle (θ : ℤ) : θ = -390 → ∃ k : ℤ, 0 ≤ θ + k * 360 ∧ θ + k * 360 < 360 ∧ θ + k * 360 = 330 :=
  by
    sorry

end same_terminal_side_angle_l207_207933


namespace average_incorrect_answers_is_correct_l207_207536

-- Definitions
def total_items : ℕ := 60
def liza_correct_answers : ℕ := (90 * total_items) / 100
def rose_correct_answers : ℕ := liza_correct_answers + 2
def max_correct_answers : ℕ := liza_correct_answers - 5

def liza_incorrect_answers : ℕ := total_items - liza_correct_answers
def rose_incorrect_answers : ℕ := total_items - rose_correct_answers
def max_incorrect_answers : ℕ := total_items - max_correct_answers

def average_incorrect_answers : ℚ :=
  (liza_incorrect_answers + rose_incorrect_answers + max_incorrect_answers) / 3

-- Theorem statement
theorem average_incorrect_answers_is_correct : average_incorrect_answers = 7 := by
  -- Proof goes here
  sorry

end average_incorrect_answers_is_correct_l207_207536


namespace greatest_ABCBA_l207_207735

/-
We need to prove that the greatest possible integer of the form AB,CBA 
that is both divisible by 11 and by 3, with A, B, and C being distinct digits, is 96569.
-/

theorem greatest_ABCBA (A B C : ℕ) 
  (h1 : A ≠ B ∧ B ≠ C ∧ A ≠ C) 
  (h2 : 1 ≤ A ∧ A ≤ 9 ∧ 0 ≤ B ∧ B ≤ 9 ∧ 0 ≤ C ∧ C ≤ 9) 
  (h3 : 10001 * A + 1010 * B + 100 * C < 100000) 
  (h4 : 2 * A - 2 * B + C ≡ 0 [MOD 11])
  (h5 : (2 * A + 2 * B + C) % 3 = 0) : 
  10001 * A + 1010 * B + 100 * C ≤ 96569 :=
sorry

end greatest_ABCBA_l207_207735


namespace value_of_expression_l207_207767

theorem value_of_expression : 2 * 2015 - 2015 = 2015 :=
by
  sorry

end value_of_expression_l207_207767


namespace does_not_pass_through_third_quadrant_l207_207274

theorem does_not_pass_through_third_quadrant :
  ¬ ∃ (x y : ℝ), 2 * x + 3 * y = 5 ∧ x < 0 ∧ y < 0 :=
by
  -- Proof goes here
  sorry

end does_not_pass_through_third_quadrant_l207_207274


namespace ab_over_a_minus_b_l207_207863

theorem ab_over_a_minus_b (a b : ℝ) (h : (1 / a) - (1 / b) = 1 / 3) : (a * b) / (a - b) = -3 := by
  sorry

end ab_over_a_minus_b_l207_207863


namespace cube_root_expression_l207_207075

variable (x : ℝ)

theorem cube_root_expression (h : x + 1 / x = 7) : x^3 + 1 / x^3 = 322 :=
  sorry

end cube_root_expression_l207_207075


namespace retailer_discount_percentage_l207_207116

noncomputable def market_price (P : ℝ) : ℝ := 36 * P
noncomputable def profit (CP : ℝ) : ℝ := CP * 0.1
noncomputable def selling_price (P : ℝ) : ℝ := 40 * P
noncomputable def total_revenue (CP Profit : ℝ) : ℝ := CP + Profit
noncomputable def discount (P S : ℝ) : ℝ := P - S
noncomputable def discount_percentage (D P : ℝ) : ℝ := (D / P) * 100

theorem retailer_discount_percentage (P CP Profit TR S D : ℝ) (h1 : CP = market_price P)
  (h2 : Profit = profit CP) (h3 : TR = total_revenue CP Profit)
  (h4 : TR = selling_price S) (h5 : S = TR / 40) (h6 : D = discount P S) :
  discount_percentage D P = 1 :=
by
  sorry

end retailer_discount_percentage_l207_207116


namespace find_length_l207_207187

-- Let's define the conditions given in the problem
variables (b l : ℝ)

-- Length is more than breadth by 200%
def length_eq_breadth_plus_200_percent (b l : ℝ) : Prop := l = 3 * b

-- Total cost and rate per square meter
def cost_eq_area_times_rate (total_cost rate area : ℝ) : Prop := total_cost = rate * area

-- Given values
def total_cost : ℝ := 529
def rate_per_sq_meter : ℝ := 3

-- We need to prove that the length l is approximately 23 meters
theorem find_length (h1 : length_eq_breadth_plus_200_percent b l) 
    (h2 : cost_eq_area_times_rate total_cost rate_per_sq_meter (3 * b^2)) : 
    abs (l - 23) < 1 :=
by
  sorry -- Proof to be filled

end find_length_l207_207187


namespace number_of_stones_l207_207017

theorem number_of_stones (hall_length_m : ℕ) (hall_breadth_m : ℕ)
  (stone_length_dm : ℕ) (stone_breadth_dm : ℕ)
  (hall_length_dm_eq : hall_length_m * 10 = 360)
  (hall_breadth_dm_eq : hall_breadth_m * 10 = 150)
  (stone_length_eq : stone_length_dm = 6)
  (stone_breadth_eq : stone_breadth_dm = 5) :
  ((hall_length_m * 10) * (hall_breadth_m * 10)) / (stone_length_dm * stone_breadth_dm) = 1800 :=
by
  sorry

end number_of_stones_l207_207017


namespace growth_pattern_equation_l207_207581

theorem growth_pattern_equation (x : ℕ) :
  1 + x + x^2 = 73 :=
sorry

end growth_pattern_equation_l207_207581


namespace graph_of_equation_is_two_lines_l207_207595

theorem graph_of_equation_is_two_lines :
  ∀ x y : ℝ, x^2 - 16*y^2 - 8*x + 16 = 0 ↔ (x = 4 + 4*y ∨ x = 4 - 4*y) :=
by
  sorry

end graph_of_equation_is_two_lines_l207_207595


namespace calculate_tan_product_l207_207964

theorem calculate_tan_product :
  let A := 30
  let B := 40
  (1 + Real.tan (A * Real.pi / 180)) * (1 + Real.tan (B * Real.pi / 180)) = 2.9 :=
by
  sorry

end calculate_tan_product_l207_207964


namespace each_bug_ate_1_5_flowers_l207_207488

-- Define the conditions given in the problem
def bugs : ℝ := 2.0
def flowers : ℝ := 3.0

-- The goal is to prove that the number of flowers each bug ate is 1.5
theorem each_bug_ate_1_5_flowers : (flowers / bugs) = 1.5 :=
by
  sorry

end each_bug_ate_1_5_flowers_l207_207488


namespace abs_add_gt_abs_sub_l207_207956

variables {a b : ℝ}

theorem abs_add_gt_abs_sub (h : a * b > 0) : |a + b| > |a - b| :=
sorry

end abs_add_gt_abs_sub_l207_207956


namespace MN_squared_l207_207061

theorem MN_squared (PQ QR RS SP : ℝ) (h1 : PQ = 15) (h2 : QR = 15) (h3 : RS = 20) (h4 : SP = 20) (angle_S : ℝ) (h5 : angle_S = 90)
(M N: ℝ) (Midpoint_M : M = (QR / 2)) (Midpoint_N : N = (SP / 2)) : 
MN^2 = 100 := by
  sorry

end MN_squared_l207_207061


namespace inequality_solution_l207_207099

theorem inequality_solution (x : ℝ) (h₁ : x ≠ 3) (h₂ : (x * (x + 1)) / ((x - 3)^2) ≥ 8) : 3 < x ∧ x ≤ 24/7 :=
sorry

end inequality_solution_l207_207099


namespace six_digit_numbers_with_at_least_one_zero_is_368559_l207_207972

def total_six_digit_numbers : ℕ := 9 * 10 * 10 * 10 * 10 * 10

def six_digit_numbers_no_zero : ℕ := 9 * 9 * 9 * 9 * 9 * 9

def six_digit_numbers_with_at_least_one_zero : ℕ :=
  total_six_digit_numbers - six_digit_numbers_no_zero

theorem six_digit_numbers_with_at_least_one_zero_is_368559 :
  six_digit_numbers_with_at_least_one_zero = 368559 := by
  sorry

end six_digit_numbers_with_at_least_one_zero_is_368559_l207_207972


namespace volumes_of_rotated_solids_l207_207904

theorem volumes_of_rotated_solids
  (π : ℝ)
  (b c a : ℝ)
  (h₁ : a^2 = b^2 + c^2)
  (v v₁ v₂ : ℝ)
  (hv : v = (1/3) * π * (b^2 * c^2) / a)
  (hv₁ : v₁ = (1/3) * π * c^2 * b)
  (hv₂ : v₂ = (1/3) * π * b^2 * c) :
  (1 / v^2) = (1 / v₁^2) + (1 / v₂^2) := 
by sorry

end volumes_of_rotated_solids_l207_207904


namespace pow_expression_eq_l207_207566

theorem pow_expression_eq : (-3)^(4^2) + 2^(3^2) = 43047233 := by
  sorry

end pow_expression_eq_l207_207566


namespace find_principal_sum_l207_207336

def compound_interest (P r : ℝ) (t : ℕ) : ℝ :=
  P * (1 + r)^t - P

def simple_interest (P r : ℝ) (t : ℕ) : ℝ :=
  P * r * t

theorem find_principal_sum (CI SI : ℝ) (t : ℕ)
  (h1 : CI = 11730) 
  (h2 : SI = 10200) 
  (h3 : t = 2) :
  ∃ P r, P = 17000 ∧
  compound_interest P r t = CI ∧
  simple_interest P r t = SI :=
by
  sorry

end find_principal_sum_l207_207336


namespace intersection_of_sets_l207_207436

open Set

def A : Set ℝ := { x | -1 < x ∧ x < 2 }
def B : Set ℝ := { x | 0 < x }

theorem intersection_of_sets : A ∩ B = { x | 0 < x ∧ x < 2 } :=
by sorry

end intersection_of_sets_l207_207436


namespace circle_center_radius_l207_207012

theorem circle_center_radius
    (x y : ℝ)
    (eq_circle : (x - 2)^2 + y^2 = 4) :
    (2, 0) = (2, 0) ∧ 2 = 2 :=
by
  sorry

end circle_center_radius_l207_207012


namespace williams_tips_august_l207_207573

variable (A : ℝ) (total_tips : ℝ)
variable (tips_August : ℝ) (average_monthly_tips_other_months : ℝ)

theorem williams_tips_august (h1 : tips_August = 0.5714285714285714 * total_tips)
                               (h2 : total_tips = 7 * average_monthly_tips_other_months) 
                               (h3 : total_tips = tips_August + 6 * average_monthly_tips_other_months) :
                               tips_August = 8 * average_monthly_tips_other_months :=
by
  sorry

end williams_tips_august_l207_207573


namespace complex_number_is_3i_quadratic_equation_roots_l207_207241

open Complex

-- Given complex number z satisfies 2z + |z| = 3 + 6i
-- We need to prove that z = 3i
theorem complex_number_is_3i (z : ℂ) (h : 2 * z + abs z = 3 + 6 * I) : z = 3 * I :=
sorry

-- Given that z = 3i is a root of the quadratic equation with real coefficients
-- Prove that b - c = -9
theorem quadratic_equation_roots (b c : ℝ) (h1 : 3 * I + -3 * I = -b)
  (h2 : 3 * I * -3 * I = c) : b - c = -9 :=
sorry

end complex_number_is_3i_quadratic_equation_roots_l207_207241


namespace lemonade_served_l207_207317

def glasses_per_pitcher : ℕ := 5
def number_of_pitchers : ℕ := 6
def total_glasses_served : ℕ := glasses_per_pitcher * number_of_pitchers

theorem lemonade_served : total_glasses_served = 30 :=
by
  -- proof goes here
  sorry

end lemonade_served_l207_207317


namespace max_a_plus_b_l207_207625

theorem max_a_plus_b (a b : ℝ) (h1 : 4 * a + 3 * b ≤ 10) (h2 : 3 * a + 6 * b ≤ 12) : a + b ≤ 14 / 5 := 
sorry

end max_a_plus_b_l207_207625


namespace quadrilateral_pyramid_volume_l207_207485

theorem quadrilateral_pyramid_volume (h Q : ℝ) : 
  ∃ V : ℝ, V = (2 / 3 : ℝ) * h * (Real.sqrt (h^2 + 4 * Q^2) - h^2) :=
by
  sorry

end quadrilateral_pyramid_volume_l207_207485


namespace find_k_l207_207120

noncomputable def f (a b c x : ℤ) : ℤ := a * x^2 + b * x + c

theorem find_k
  (a b c : ℤ)
  (k : ℤ)
  (h1 : f a b c 1 = 0)
  (h2 : 50 < f a b c 7)
  (h3 : f a b c 7 < 60)
  (h4 : 70 < f a b c 8)
  (h5 : f a b c 8 < 80)
  (h6 : 5000 * k < f a b c 100)
  (h7 : f a b c 100 < 5000 * (k + 1)) :
  k = 3 :=
sorry

end find_k_l207_207120


namespace seventh_root_of_unity_problem_l207_207073

theorem seventh_root_of_unity_problem (q : ℂ) (h : q^7 = 1) :
  (q = 1 → (q / (1 + q^2) + q^2 / (1 + q^4) + q^3 / (1 + q^6)) = 3 / 2) ∧ 
  (q ≠ 1 → (q / (1 + q^2) + q^2 / (1 + q^4) + q^3 / (1 + q^6)) = -2) :=
by
  sorry

end seventh_root_of_unity_problem_l207_207073


namespace count_solutions_cos2x_plus_3sin2x_eq_1_l207_207473

open Real

theorem count_solutions_cos2x_plus_3sin2x_eq_1 :
  ∀ x : ℝ, (-10 < x ∧ x < 45 → cos x ^ 2 + 3 * sin x ^ 2 = 1) → 
  ∃! n : ℕ, n = 18 := 
by
  intro x hEq
  sorry

end count_solutions_cos2x_plus_3sin2x_eq_1_l207_207473


namespace find_m_for_all_n_l207_207785

def sum_of_digits (k: ℕ) : ℕ :=
  k.digits 10 |>.sum

def A (k: ℕ) : ℕ :=
  -- Constructing the number A_k as described
  -- This is a placeholder for the actual implementation
  sorry

theorem find_m_for_all_n (n: ℕ) (hn: 0 < n) :
  ∃ m: ℕ, 0 < m ∧ n ∣ A m ∧ n ∣ m ∧ n ∣ sum_of_digits (A m) :=
sorry

end find_m_for_all_n_l207_207785


namespace find_d_k_l207_207840

open Matrix

noncomputable def matrix_A (d : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![3, 4], ![6, d]]

noncomputable def inv_matrix_A (d : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  let detA := 3 * d - 24
  (1 / detA) • ![![d, -4], ![-6, 3]]

theorem find_d_k (d k : ℝ) (h : inv_matrix_A d = k • matrix_A d) :
    (d, k) = (-3, 1/33) := by
  sorry

end find_d_k_l207_207840


namespace correct_random_error_causes_l207_207789

-- Definitions based on conditions
def is_random_error_cause (n : ℕ) : Prop :=
  n = 1 ∨ n = 2 ∨ n = 3

-- Theorem: Valid causes of random errors are options (1), (2), and (3)
theorem correct_random_error_causes :
  (is_random_error_cause 1) ∧ (is_random_error_cause 2) ∧ (is_random_error_cause 3) :=
by
  sorry

end correct_random_error_causes_l207_207789


namespace sum_is_945_l207_207879

def sum_of_integers_from_90_to_99 : ℕ :=
  90 + 91 + 92 + 93 + 94 + 95 + 96 + 97 + 98 + 99

theorem sum_is_945 : sum_of_integers_from_90_to_99 = 945 := 
by
  sorry

end sum_is_945_l207_207879


namespace range_of_a_l207_207607

noncomputable def f (a x : ℝ) : ℝ := x * Real.exp x + (1 / 2) * a * x^2 + a * x

theorem range_of_a (a : ℝ) : 
    (∀ x : ℝ, 2 * Real.exp (f a x) + Real.exp 1 + 2 ≥ 0) ↔ (0 ≤ a ∧ a ≤ 1) := 
sorry

end range_of_a_l207_207607


namespace complement_union_l207_207766

open Set

universe u

def U : Set ℕ := {1, 2, 3, 4}
def A : Set ℕ := {2, 4}
def B : Set ℕ := {1, 4}

theorem complement_union (U A B : Set ℕ) (hU : U = {1, 2, 3, 4}) (hA : A = {2, 4}) (hB : B = {1, 4}) :
  (U \ (A ∪ B)) = {3} :=
by
  simp [hU, hA, hB]
  sorry

end complement_union_l207_207766


namespace number_of_rectangles_containing_cell_l207_207745

theorem number_of_rectangles_containing_cell (m n p q : ℕ) (hp : 1 ≤ p ∧ p ≤ m) (hq : 1 ≤ q ∧ q ≤ n) :
    ∃ count : ℕ, count = p * q * (m - p + 1) * (n - q + 1) := 
    sorry

end number_of_rectangles_containing_cell_l207_207745


namespace sum_of_digits_divisible_by_9_l207_207427

theorem sum_of_digits_divisible_by_9 (N : ℕ) (a b c : ℕ) (hN : N < 10^1962)
  (h1 : N % 9 = 0)
  (ha : a = (N.digits 10).sum)
  (hb : b = (a.digits 10).sum)
  (hc : c = (b.digits 10).sum) :
  c = 9 :=
sorry

end sum_of_digits_divisible_by_9_l207_207427


namespace smallest_constant_N_l207_207623

-- Given that a, b, c are sides of a triangle and in arithmetic progression, prove that
-- (a^2 + b^2 + c^2) / (ab + bc + ca) ≥ 1.

theorem smallest_constant_N
  (a b c : ℝ)
  (habc : a + b > c ∧ a + c > b ∧ b + c > a) -- Triangle inequality
  (hap : ∃ d : ℝ, b = a + d ∧ c = a + 2 * d) -- Arithmetic progression
  : (a^2 + b^2 + c^2) / (a * b + b * c + c * a) ≥ 1 := 
sorry

end smallest_constant_N_l207_207623


namespace martha_black_butterflies_l207_207945

theorem martha_black_butterflies (total_butterflies blue_butterflies yellow_butterflies black_butterflies : ℕ) 
    (h1 : total_butterflies = 19)
    (h2 : blue_butterflies = 2 * yellow_butterflies)
    (h3 : blue_butterflies = 6) :
    black_butterflies = 10 :=
by
  -- Prove the theorem assuming the conditions are met
  sorry

end martha_black_butterflies_l207_207945


namespace fill_bucket_time_l207_207572

-- Problem statement:
-- Prove that the time taken to fill the bucket completely is 150 seconds
-- given that two-thirds of the bucket is filled in 100 seconds.

theorem fill_bucket_time (t : ℕ) (h : (2 / 3) * t = 100) : t = 150 :=
by
  -- Proof should be here
  sorry

end fill_bucket_time_l207_207572


namespace A_min_votes_for_victory_l207_207018

theorem A_min_votes_for_victory:
  ∀ (initial_votes_A initial_votes_B initial_votes_C total_votes remaining_votes min_votes_A: ℕ),
  initial_votes_A = 350 →
  initial_votes_B = 370 →
  initial_votes_C = 280 →
  total_votes = 1500 →
  remaining_votes = 500 →
  min_votes_A = 261 →
  initial_votes_A + min_votes_A > initial_votes_B + (remaining_votes - min_votes_A) :=
by
  intros _ _ _ _ _ _
  sorry

end A_min_votes_for_victory_l207_207018


namespace math_proof_problem_l207_207809

-- Define constants
def x := 2000000000000
def y := 1111111111111

-- Prove the main statement
theorem math_proof_problem :
  2 * (x - y) = 1777777777778 := 
  by
    sorry

end math_proof_problem_l207_207809


namespace geometric_sequence_problem_l207_207890

noncomputable def geom_sequence (a r : ℝ) (n : ℕ) : ℝ := a * r ^ n

theorem geometric_sequence_problem (a r : ℝ) (a4 a8 a6 a10 : ℝ) :
  a4 = geom_sequence a r 4 →
  a8 = geom_sequence a r 8 →
  a6 = geom_sequence a r 6 →
  a10 = geom_sequence a r 10 →
  a4 + a8 = -2 →
  a4^2 + 2 * a6^2 + a6 * a10 = 4 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end geometric_sequence_problem_l207_207890


namespace find_a_and_b_l207_207423

theorem find_a_and_b :
  ∃ a b : ℝ, 
    (∀ x : ℝ, (x^3 + 3*x^2 + 2*x > 0) ↔ (x > 0 ∨ -2 < x ∧ x < -1)) ∧
    (∀ x : ℝ, (x^2 + a*x + b ≤ 0) ↔ (-2 < x ∧ x ≤ 0 ∨ 0 < x ∧ x ≤ 2)) ∧ 
    a = -1 ∧ b = -2 := 
  sorry

end find_a_and_b_l207_207423


namespace solution_to_problem_l207_207117

theorem solution_to_problem (x : ℝ) (h : 12^(Real.log 7 / Real.log 12) = 10 * x + 3) : x = 2 / 5 :=
by sorry

end solution_to_problem_l207_207117


namespace Sandy_change_l207_207747

theorem Sandy_change (pants shirt sweater shoes total paid change : ℝ)
  (h1 : pants = 13.58) (h2 : shirt = 10.29) (h3 : sweater = 24.97) (h4 : shoes = 39.99) (h5 : total = pants + shirt + sweater + shoes) (h6 : paid = 100) (h7 : change = paid - total) :
  change = 11.17 := 
sorry

end Sandy_change_l207_207747


namespace motel_percentage_reduction_l207_207035

theorem motel_percentage_reduction
  (x y : ℕ) 
  (h : 40 * x + 60 * y = 1000) :
  ((1000 - (40 * (x + 10) + 60 * (y - 10))) / 1000) * 100 = 20 := 
by
  sorry

end motel_percentage_reduction_l207_207035


namespace p_q_2r_value_l207_207917

variable (p q r : ℝ) (f : ℝ → ℝ)

-- The conditions as definitions
def f_def : f = fun x => p * x^2 + q * x + r := by sorry
def f_at_0 : f 0 = 9 := by sorry
def f_at_1 : f 1 = 6 := by sorry

-- The theorem statement
theorem p_q_2r_value : p + q + 2 * r = 15 :=
by
  -- utilizing the given definitions 
  have h₁ : r = 9 := by sorry
  have h₂ : p + q + r = 6 := by sorry
  -- substitute into p + q + 2r
  sorry

end p_q_2r_value_l207_207917


namespace initial_rotations_l207_207513

-- Given conditions as Lean definitions
def rotations_per_block : ℕ := 200
def blocks_to_ride : ℕ := 8
def additional_rotations_needed : ℕ := 1000

-- Question translated to proof statement
theorem initial_rotations (rotations : ℕ) :
  rotations + additional_rotations_needed = rotations_per_block * blocks_to_ride → rotations = 600 :=
by
  intros h
  sorry

end initial_rotations_l207_207513


namespace salon_revenue_l207_207521

noncomputable def revenue (num_customers first_visit second_visit third_visit : ℕ) (first_charge second_charge : ℕ) : ℕ :=
  num_customers * first_charge + second_visit * second_charge + third_visit * second_charge

theorem salon_revenue : revenue 100 100 30 10 10 8 = 1320 :=
by
  unfold revenue
  -- The proof will continue here.
  sorry

end salon_revenue_l207_207521


namespace benny_pays_l207_207253

theorem benny_pays (cost_per_lunch : ℕ) (number_of_people : ℕ) (total_cost : ℕ) :
  cost_per_lunch = 8 → number_of_people = 3 → total_cost = number_of_people * cost_per_lunch → total_cost = 24 :=
by
  intros h₁ h₂ h₃
  rw [h₁, h₂] at h₃
  exact h₃

end benny_pays_l207_207253


namespace spring_length_increase_l207_207729

-- Define the weight (x) and length (y) data points
def weights : List ℝ := [0, 1, 2, 3, 4, 5]
def lengths : List ℝ := [20, 20.5, 21, 21.5, 22, 22.5]

-- Prove that for each increase of 1 kg in weight, the length of the spring increases by 0.5 cm
theorem spring_length_increase (h : weights.length = lengths.length) :
  ∀ i, i < weights.length - 1 → (lengths.get! (i+1) - lengths.get! i) = 0.5 :=
by
  -- Proof goes here, omitted for now
  sorry

end spring_length_increase_l207_207729


namespace inverse_proposition_l207_207699

theorem inverse_proposition (a b c : ℝ) : (a > b → a + c > b + c) → (a + c > b + c → a > b) :=
sorry

end inverse_proposition_l207_207699


namespace sum_of_four_digits_l207_207790

theorem sum_of_four_digits (EH OY AY OH : ℕ) (h1 : EH = 4 * OY) (h2 : AY = 4 * OH) : EH + OY + AY + OH = 150 :=
sorry

end sum_of_four_digits_l207_207790


namespace coordinate_difference_l207_207060

theorem coordinate_difference (m n : ℝ) (h : m = 4 * n + 5) :
  (4 * (n + 0.5) + 5) - m = 2 :=
by
  -- proof skipped
  sorry

end coordinate_difference_l207_207060


namespace common_terms_only_1_and_7_l207_207676

def sequence_a (n : ℕ) : ℕ :=
  if n = 0 then 1
  else if n = 1 then 2
  else 4 * sequence_a (n - 1) - sequence_a (n - 2)

def sequence_b (n : ℕ) : ℕ :=
  if n = 0 then 1
  else if n = 1 then 7
  else 6 * sequence_b (n - 1) - sequence_b (n - 2)

theorem common_terms_only_1_and_7 :
  ∀ n m : ℕ, (sequence_a n = sequence_b m) → (sequence_a n = 1 ∨ sequence_a n = 7) :=
by {
  sorry
}

end common_terms_only_1_and_7_l207_207676


namespace library_visitors_total_l207_207358

theorem library_visitors_total
  (visitors_monday : ℕ)
  (visitors_tuesday : ℕ)
  (average_visitors_remaining_days : ℕ)
  (remaining_days : ℕ)
  (total_visitors : ℕ)
  (hmonday : visitors_monday = 50)
  (htuesday : visitors_tuesday = 2 * visitors_monday)
  (haverage : average_visitors_remaining_days = 20)
  (hremaining_days : remaining_days = 5)
  (htotal : total_visitors =
    visitors_monday + visitors_tuesday + remaining_days * average_visitors_remaining_days) :
  total_visitors = 250 :=
by
  -- here goes the proof, marked as sorry for now
  sorry

end library_visitors_total_l207_207358


namespace crackers_per_box_l207_207776

-- Given conditions
variables (x : ℕ)
variable (darren_boxes : ℕ := 4)
variable (calvin_boxes : ℕ := 2 * darren_boxes - 1)
variable (total_crackers : ℕ := 264)

-- Using the given conditions, create the proof statement to show x = 24
theorem crackers_per_box:
  11 * x = total_crackers → x = 24 :=
by
  sorry

end crackers_per_box_l207_207776


namespace sarah_driving_distance_l207_207182

def sarah_car_mileage (miles_per_gallon : ℕ) (tank_capacity : ℕ) (initial_drive : ℕ) (refuel : ℕ) (remaining_fraction : ℚ) : Prop :=
  ∃ (total_drive : ℚ),
    (initial_drive / miles_per_gallon + refuel - (tank_capacity * remaining_fraction / 1)) * miles_per_gallon = total_drive ∧
    total_drive = 467

theorem sarah_driving_distance :
  sarah_car_mileage 28 16 280 6 (1 / 3) :=
by
  sorry

end sarah_driving_distance_l207_207182


namespace area_of_triangle_ABC_l207_207528

open Real

-- Defining the conditions as per the problem
def triangle_side_equality (AB AC : ℝ) : Prop := AB = AC
def angle_relation (angleBAC angleBTC : ℝ) : Prop := angleBAC = 2 * angleBTC
def side_length_BT (BT : ℝ) : Prop := BT = 70
def side_length_AT (AT : ℝ) : Prop := AT = 37

-- Proving the area of triangle ABC given the conditions
theorem area_of_triangle_ABC
  (AB AC : ℝ)
  (angleBAC angleBTC : ℝ)
  (BT AT : ℝ)
  (h1 : triangle_side_equality AB AC)
  (h2 : angle_relation angleBAC angleBTC)
  (h3 : side_length_BT BT)
  (h4 : side_length_AT AT) 
  : ∃ area : ℝ, area = 420 :=
sorry

end area_of_triangle_ABC_l207_207528


namespace find_integer_x_l207_207122

open Nat

noncomputable def isSquareOfPrime (n : ℤ) : Prop :=
  ∃ p : ℤ, Nat.Prime (Int.natAbs p) ∧ n = p * p

theorem find_integer_x :
  ∃ x : ℤ,
  (x = -360 ∨ x = -60 ∨ x = -48 ∨ x = -40 ∨ x = 8 ∨ x = 20 ∨ x = 32 ∨ x = 332) ∧
  isSquareOfPrime (x^2 + 28*x + 889) :=
sorry

end find_integer_x_l207_207122


namespace upper_limit_of_multiples_of_10_l207_207384

theorem upper_limit_of_multiples_of_10 (n : ℕ) (hn : 10 * n = 100) (havg : (10 * n + 10) / (n + 1) = 55) : 10 * n = 100 :=
by
  sorry

end upper_limit_of_multiples_of_10_l207_207384


namespace find_A_plus_B_l207_207867

def isFourDigitNumber (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

def isOdd (n : ℕ) : Prop :=
  n % 2 = 1

def isMultipleOf5 (n : ℕ) : Prop :=
  n % 5 = 0

def countFourDigitOddNumbers : ℕ :=
  ((9 : ℕ) * 10 * 10 * 5)

def countFourDigitMultiplesOf5 : ℕ :=
  ((9 : ℕ) * 10 * 10 * 2)

theorem find_A_plus_B : countFourDigitOddNumbers + countFourDigitMultiplesOf5 = 6300 := by
  sorry

end find_A_plus_B_l207_207867


namespace net_change_in_price_l207_207350

-- Define the initial price of the TV
def initial_price (P : ℝ) := P

-- Define the price after a 20% decrease
def decreased_price (P : ℝ) := 0.80 * P

-- Define the final price after a 50% increase on the decreased price
def final_price (P : ℝ) := 1.20 * P

-- Prove that the net change is 20% of the original price
theorem net_change_in_price (P : ℝ) : final_price P - initial_price P = 0.20 * P := by
  sorry

end net_change_in_price_l207_207350


namespace minimize_b_plus_4c_l207_207565

noncomputable def triangle := Type

variable {ABC : triangle}
variable (a b c : ℝ) -- sides of the triangle
variable (BAC : ℝ) -- angle BAC
variable (D : triangle → ℝ) -- angle bisector intersecting BC at D
variable (AD : ℝ) -- length of AD
variable (min_bc : ℝ) -- minimum value of b + 4c

-- Conditions
variable (h1 : BAC = 120)
variable (h2 : D ABC = 1)
variable (h3 : AD = 1)

-- Proof statement
theorem minimize_b_plus_4c (h1 : BAC = 120) (h2 : D ABC = 1) (h3 : AD = 1) : min_bc = 9 := 
sorry

end minimize_b_plus_4c_l207_207565


namespace carolyn_sum_correct_l207_207493

def initial_sequence := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

def carolyn_removes : List ℕ := [4, 8, 10, 9]

theorem carolyn_sum_correct : carolyn_removes.sum = 31 :=
by
  sorry

end carolyn_sum_correct_l207_207493


namespace items_left_in_cart_l207_207795

-- Define the initial items in the shopping cart
def initial_items : ℕ := 18

-- Define the items deleted from the shopping cart
def deleted_items : ℕ := 10

-- Theorem statement: Prove the remaining items are 8
theorem items_left_in_cart : initial_items - deleted_items = 8 :=
by
  -- Sorry marks the place where the proof would go.
  sorry

end items_left_in_cart_l207_207795


namespace max_value_of_gems_l207_207196

/-- Conditions -/
structure Gem :=
  (weight : ℕ)
  (value : ℕ)

def Gem1 : Gem := ⟨3, 9⟩
def Gem2 : Gem := ⟨6, 20⟩
def Gem3 : Gem := ⟨2, 5⟩

-- Laura can carry maximum of 21 pounds.
def max_weight : ℕ := 21

-- She is able to carry at least 15 of each type
def min_count := 15

/-- Prove that the maximum value Laura can carry is $69 -/
theorem max_value_of_gems : ∃ (n1 n2 n3 : ℕ), (n1 >= min_count) ∧ (n2 >= min_count) ∧ (n3 >= min_count) ∧ 
  (Gem1.weight * n1 + Gem2.weight * n2 + Gem3.weight * n3 ≤ max_weight) ∧ 
  (Gem1.value * n1 + Gem2.value * n2 + Gem3.value * n3 = 69) :=
sorry

end max_value_of_gems_l207_207196


namespace problem_l207_207399

theorem problem:
  ∀ k : Real, (2 - Real.sqrt 2 / 2 ≤ k ∧ k ≤ 2 + Real.sqrt 2 / 2) →
  (11 - 6 * Real.sqrt 2) / 4 ≤ (3 / 2 * (k - 1)^2 + 1 / 2) ∧ 
  (3 / 2 * (k - 1)^2 + 1 / 2 ≤ (11 + 6 * Real.sqrt 2) / 4) :=
by
  intros k hk
  sorry

end problem_l207_207399


namespace value_of_expression_l207_207261

theorem value_of_expression (x : ℤ) (h : x = 4) : (3 * x + 7) ^ 2 = 361 := by
  rw [h] -- Replace x with 4
  norm_num -- Simplify the expression
  done

end value_of_expression_l207_207261


namespace parabola_points_l207_207636

theorem parabola_points :
  {p : ℝ × ℝ | p.2 = p.1^2 - 1 ∧ p.2 = 3} = {(-2, 3), (2, 3)} :=
by
  sorry

end parabola_points_l207_207636


namespace ribbon_tape_needed_l207_207526

theorem ribbon_tape_needed 
  (total_length : ℝ) (num_boxes : ℕ) (ribbon_per_box : ℝ)
  (h1 : total_length = 82.04)
  (h2 : num_boxes = 28)
  (h3 : total_length / num_boxes = ribbon_per_box)
  : ribbon_per_box = 2.93 :=
sorry

end ribbon_tape_needed_l207_207526


namespace ratio_of_expenditures_l207_207033

variable (Rajan_income Balan_income Rajan_expenditure Balan_expenditure Rajan_savings Balan_savings: ℤ)
variable (ratio_incomes: ℚ)
variable (savings_amount: ℤ)

-- Given conditions
def conditions : Prop :=
  Rajan_income = 7000 ∧
  ratio_incomes = 7 / 6 ∧
  savings_amount = 1000 ∧
  Rajan_savings = Rajan_income - Rajan_expenditure ∧
  Balan_savings = Balan_income - Balan_expenditure ∧
  Rajan_savings = savings_amount ∧
  Balan_savings = savings_amount

-- The theorem we want to prove
theorem ratio_of_expenditures :
  conditions Rajan_income Balan_income Rajan_expenditure Balan_expenditure Rajan_savings Balan_savings ratio_incomes savings_amount →
  (Rajan_expenditure : ℚ) / (Balan_expenditure : ℚ) = 6 / 5 :=
by
  sorry

end ratio_of_expenditures_l207_207033


namespace juan_original_number_l207_207609

theorem juan_original_number (n : ℤ) 
  (h : ((2 * (n + 3) - 2) / 2) = 8) : 
  n = 6 := 
sorry

end juan_original_number_l207_207609


namespace revolutions_same_distance_l207_207457

theorem revolutions_same_distance (r R : ℝ) (revs_30 : ℝ) (dist_30 dist_10 : ℝ)
  (h_radius: r = 10) (H_radius: R = 30) (h_revs_30: revs_30 = 15) 
  (H_dist_30: dist_30 = 2 * Real.pi * R * revs_30) 
  (H_dist_10: dist_10 = 2 * Real.pi * r * 45) :
  dist_30 = dist_10 :=
by {
  sorry
}

end revolutions_same_distance_l207_207457


namespace fall_increase_l207_207680

noncomputable def percentage_increase_in_fall (x : ℝ) : ℝ :=
  x

theorem fall_increase :
  ∃ (x : ℝ), (1 + percentage_increase_in_fall x / 100) * (1 - 19 / 100) = 1 + 11.71 / 100 :=
by
  sorry

end fall_increase_l207_207680


namespace cost_per_kg_after_30_l207_207509

theorem cost_per_kg_after_30 (l m : ℝ) 
  (hl : l = 20) 
  (h1 : 30 * l + 3 * m = 663) 
  (h2 : 30 * l + 6 * m = 726) : 
  m = 21 :=
by
  -- Proof will be written here
  sorry

end cost_per_kg_after_30_l207_207509


namespace comb_12_9_eq_220_l207_207180

theorem comb_12_9_eq_220 : (Nat.choose 12 9) = 220 := by
  sorry

end comb_12_9_eq_220_l207_207180


namespace correct_propositions_count_l207_207800

theorem correct_propositions_count (x y : ℝ) :
  (x ≠ 0 ∨ y ≠ 0) → (x^2 + y^2 ≠ 0) ∧ -- original proposition
  (x^2 + y^2 ≠ 0) → (x ≠ 0 ∨ y ≠ 0) ∧ -- converse proposition
  (¬(x ≠ 0 ∨ y ≠ 0) ∨ x^2 + y^2 = 0) ∧ -- negation proposition
  (¬(x^2 + y^2 = 0) ∨ x ≠ 0 ∨ y ≠ 0) -- inverse proposition
  := by
  sorry

end correct_propositions_count_l207_207800


namespace pentomino_symmetry_count_l207_207173

noncomputable def num_symmetric_pentominoes : Nat :=
  15 -- This represents the given set of 15 different pentominoes

noncomputable def symmetric_pentomino_count : Nat :=
  -- Here we are asserting that the count of pentominoes with at least one vertical symmetry is 8
  8

theorem pentomino_symmetry_count :
  symmetric_pentomino_count = 8 :=
sorry

end pentomino_symmetry_count_l207_207173


namespace total_votes_l207_207397

theorem total_votes (V : ℝ) (h1 : 0.60 * V = V - 240) : V = 600 :=
sorry

end total_votes_l207_207397


namespace books_sold_correct_l207_207425

-- Definitions of the conditions
def initial_books : ℕ := 33
def remaining_books : ℕ := 7
def books_sold : ℕ := initial_books - remaining_books

-- The statement to be proven (with proof omitted)
theorem books_sold_correct : books_sold = 26 := by
  -- Proof omitted
  sorry

end books_sold_correct_l207_207425


namespace sum_of_other_endpoint_l207_207445

theorem sum_of_other_endpoint (x y : ℝ) :
  (10, -6) = ((x + 12) / 2, (y + 4) / 2) → x + y = -8 :=
by
  sorry

end sum_of_other_endpoint_l207_207445


namespace product_divisible_by_12_l207_207711

theorem product_divisible_by_12 (a b c d : ℤ) : 
  12 ∣ ((b - a) * (c - a) * (d - a) * (b - c) * (d - c) * (d - b)) :=
  sorry

end product_divisible_by_12_l207_207711


namespace track_and_field_analysis_l207_207219

theorem track_and_field_analysis :
  let male_athletes := 12
  let female_athletes := 8
  let tallest_height := 190
  let shortest_height := 160
  let avg_male_height := 175
  let avg_female_height := 165
  let total_athletes := male_athletes + female_athletes
  let sample_size := 10
  let prob_selected := 1 / 2
  let prop_male := male_athletes / total_athletes * sample_size
  let prop_female := female_athletes / total_athletes * sample_size
  let overall_avg_height := (male_athletes / total_athletes) * avg_male_height + (female_athletes / total_athletes) * avg_female_height
  (tallest_height - shortest_height = 30) ∧
  (sample_size / total_athletes = prob_selected) ∧
  (prop_male = 6 ∧ prop_female = 4) ∧
  (overall_avg_height = 171) →
  (A = true ∧ B = true ∧ C = false ∧ D = true) :=
by
  sorry

end track_and_field_analysis_l207_207219


namespace bad_carrots_count_l207_207780

-- Define the number of carrots each person picked and the number of good carrots
def carol_picked := 29
def mom_picked := 16
def good_carrots := 38

-- Define the total number of carrots picked and the total number of bad carrots
def total_carrots := carol_picked + mom_picked
def bad_carrots := total_carrots - good_carrots

-- State the theorem that the number of bad carrots is 7
theorem bad_carrots_count :
  bad_carrots = 7 :=
by
  sorry

end bad_carrots_count_l207_207780


namespace attendees_chose_water_l207_207194

theorem attendees_chose_water
  (total_attendees : ℕ)
  (juice_percentage water_percentage : ℝ)
  (attendees_juice : ℕ)
  (h1 : juice_percentage = 0.7)
  (h2 : water_percentage = 0.3)
  (h3 : attendees_juice = 140)
  (h4 : total_attendees * juice_percentage = attendees_juice)
  : total_attendees * water_percentage = 60 := by
  sorry

end attendees_chose_water_l207_207194


namespace find_a_l207_207969

noncomputable def f (x a : ℝ) : ℝ := -9 * x^2 - 6 * a * x + 2 * a - a^2

theorem find_a : (∃ a : ℝ, ((∀ x ∈ Set.Icc (-(1/3):ℝ) (1/3), f x a ≤ -3) ∧ (∃ x ∈ Set.Icc (-(1/3):ℝ) (1/3), f x a = -3)) ↔ a = Real.sqrt 6 + 2) :=
by
  sorry

end find_a_l207_207969


namespace minimum_total_length_of_removed_segments_l207_207067

-- Definitions based on conditions
def right_angled_triangle_sides : Nat × Nat × Nat := (3, 4, 5)

def large_square_side_length : Nat := 7

-- Statement of the problem to be proved
theorem minimum_total_length_of_removed_segments
  (triangles : Fin 4 → (Nat × Nat × Nat) := fun _ => right_angled_triangle_sides)
  (side_length_of_large_square : Nat := large_square_side_length) :
  ∃ (removed_length : Nat), removed_length = 7 :=
sorry

end minimum_total_length_of_removed_segments_l207_207067


namespace preimage_of_5_1_is_2_3_l207_207487

-- Define the mapping function
def f (p : ℝ × ℝ) : ℝ × ℝ := (p.1 + p.2, 2*p.1 - p.2)

-- Define the pre-image condition for (5, 1)
theorem preimage_of_5_1_is_2_3 : ∃ p : ℝ × ℝ, f p = (5, 1) ∧ p = (2, 3) :=
by
  -- Here we state that such a point p exists with the required properties.
  sorry

end preimage_of_5_1_is_2_3_l207_207487


namespace scientific_notation_of_22nm_l207_207434

theorem scientific_notation_of_22nm (h : 22 * 10^(-9) = 0.000000022) : 0.000000022 = 2.2 * 10^(-8) :=
sorry

end scientific_notation_of_22nm_l207_207434


namespace line_passing_through_quadrants_l207_207177

theorem line_passing_through_quadrants (a : ℝ) :
  (∀ x : ℝ, (3 * a - 1) * x - 1 ≠ 0) →
  (3 * a - 1 > 0) →
  a > 1 / 3 :=
by
  intro h1 h2
  -- proof to be filled
  sorry

end line_passing_through_quadrants_l207_207177


namespace number_of_girls_is_4_l207_207968

variable (x : ℕ)

def number_of_boys : ℕ := 12

def average_score_boys : ℕ := 84

def average_score_girls : ℕ := 92

def average_score_class : ℕ := 86

theorem number_of_girls_is_4 
  (h : average_score_class = 
    (average_score_boys * number_of_boys + average_score_girls * x) / (number_of_boys + x))
  : x = 4 := 
sorry

end number_of_girls_is_4_l207_207968


namespace greatest_coloring_integer_l207_207386

theorem greatest_coloring_integer (α β : ℝ) (h1 : 1 < α) (h2 : α < β) :
  ∃ r : ℕ, r = 2 ∧ ∀ (f : ℕ → ℕ), ∃ x y : ℕ, x ≠ y ∧ f x = f y ∧ α ≤ (x : ℝ) / (y : ℝ) ∧ (x : ℝ) / (y : ℝ) ≤ β := 
sorry

end greatest_coloring_integer_l207_207386


namespace appliance_costs_l207_207984

theorem appliance_costs (a b : ℕ) 
  (h1 : a + 2 * b = 2300) 
  (h2 : 2 * a + b = 2050) : 
  a = 600 ∧ b = 850 := 
by 
  sorry

end appliance_costs_l207_207984


namespace max_students_l207_207448

theorem max_students (pens pencils : ℕ) (h_pens : pens = 1340) (h_pencils : pencils = 1280) : Nat.gcd pens pencils = 20 := by
    sorry

end max_students_l207_207448


namespace domain_shift_l207_207519

theorem domain_shift (f : ℝ → ℝ) (dom_f : ∀ x, 1 ≤ x ∧ x ≤ 4 → f x = f x) :
  ∀ x, -1 ≤ x ∧ x ≤ 2 ↔ (1 ≤ x + 2 ∧ x + 2 ≤ 4) :=
by
  sorry

end domain_shift_l207_207519


namespace f_minus_5_eq_12_l207_207821

def f (x : ℝ) : ℝ := x^2 + 2*x - 3

theorem f_minus_5_eq_12 : f (-5) = 12 := 
by sorry

end f_minus_5_eq_12_l207_207821


namespace kelly_gave_away_games_l207_207309

theorem kelly_gave_away_games (initial_games : ℕ) (remaining_games : ℕ) (given_away_games : ℕ) 
  (h1 : initial_games = 183) 
  (h2 : remaining_games = 92) 
  (h3 : given_away_games = initial_games - remaining_games) : 
  given_away_games = 91 := 
by 
  sorry

end kelly_gave_away_games_l207_207309


namespace quadrilateral_angle_difference_l207_207706

theorem quadrilateral_angle_difference (h_ratio : ∀ (a b c d : ℕ), a = 3 * d ∧ b = 4 * d ∧ c = 5 * d ∧ d = 6 * d) 
  (h_sum : ∀ (a b c d : ℕ), a + b + c + d = 360) : 
  ∃ (x : ℕ), 6 * x - 3 * x = 60 := 
by 
  sorry

end quadrilateral_angle_difference_l207_207706


namespace correct_equation_for_gift_exchanges_l207_207258

theorem correct_equation_for_gift_exchanges
  (x : ℕ)
  (H : (x * (x - 1)) = 56) :
  x * (x - 1) = 56 := 
by 
  exact H

end correct_equation_for_gift_exchanges_l207_207258


namespace symmetric_point_origin_l207_207775

theorem symmetric_point_origin (x y : ℤ) (h : x = -2 ∧ y = 3) :
    (-x, -y) = (2, -3) :=
by
  cases h with
  | intro hx hy =>
  simp only [hx, hy]
  sorry

end symmetric_point_origin_l207_207775


namespace jana_distance_l207_207953

theorem jana_distance (time_to_walk_one_mile : ℝ) (time_to_walk : ℝ) :
  (time_to_walk_one_mile = 18) → (time_to_walk = 15) →
  ((time_to_walk / time_to_walk_one_mile) * 1 = 0.8) :=
  by
    intros h1 h2
    rw [h1, h2]
    -- Here goes the proof, but it is skipped as per requirements
    sorry

end jana_distance_l207_207953


namespace Jason_attended_36_games_l207_207959

noncomputable def games_attended (planned_this_month : ℕ) (planned_last_month : ℕ) (percentage_missed : ℕ) : ℕ :=
  let total_planned := planned_this_month + planned_last_month
  let missed_games := (percentage_missed * total_planned) / 100
  total_planned - missed_games

theorem Jason_attended_36_games :
  games_attended 24 36 40 = 36 :=
by
  sorry

end Jason_attended_36_games_l207_207959


namespace no_five_consecutive_divisible_by_2005_l207_207679

def seq (n : ℕ) : ℕ := 1 + 2^n + 3^n + 4^n + 5^n

theorem no_five_consecutive_divisible_by_2005 :
  ¬ (∃ m : ℕ, ∀ k : ℕ, k < 5 → (seq (m + k)) % 2005 = 0) :=
sorry

end no_five_consecutive_divisible_by_2005_l207_207679


namespace family_members_to_pay_l207_207291

theorem family_members_to_pay :
  (∃ (n : ℕ), 
    5 * 12 = 60 ∧ 
    60 * 2 = 120 ∧ 
    120 / 10 = 12 ∧ 
    12 * 2 = 24 ∧ 
    24 / 4 = n ∧ 
    n = 6) :=
by
  sorry

end family_members_to_pay_l207_207291


namespace coordinates_of_B_l207_207874

noncomputable def B_coordinates := 
  let A : ℝ × ℝ := (-1, -5)
  let a : ℝ × ℝ := (2, 3)
  let AB := (3 * a.1, 3 * a.2)
  let B := (A.1 + AB.1, A.2 + AB.2)
  B

theorem coordinates_of_B : B_coordinates = (5, 4) := 
by 
  sorry

end coordinates_of_B_l207_207874


namespace range_of_a_l207_207585

open Set

theorem range_of_a (a : ℝ) (M N : Set ℝ) (hM : ∀ x, x ∈ M ↔ x < 2) (hN : ∀ x, x ∈ N ↔ x < a) (hMN : M ⊆ N) : 2 ≤ a :=
by
  sorry

end range_of_a_l207_207585


namespace max_self_intersections_polyline_7_l207_207955

def max_self_intersections (n : ℕ) : ℕ :=
  if h : n > 2 then (n * (n - 3)) / 2 else 0

theorem max_self_intersections_polyline_7 :
  max_self_intersections 7 = 14 := 
sorry

end max_self_intersections_polyline_7_l207_207955


namespace yao_ming_mcgrady_probability_l207_207008

theorem yao_ming_mcgrady_probability
        (p : ℝ) (q : ℝ)
        (h1 : p = 0.8)
        (h2 : q = 0.7) :
        (2 * p * (1 - p)) * (2 * q * (1 - q)) = 0.1344 := 
by
  sorry

end yao_ming_mcgrady_probability_l207_207008


namespace part1_monotonicity_part2_range_a_l207_207912

noncomputable def f (a x : ℝ) : ℝ := Real.log x - a * x + 1

theorem part1_monotonicity (a : ℝ) :
  (∀ x > 0, (0 : ℝ) < x → 0 < 1 / x - a) ∨
  (a > 0 → ∀ x > 0, (0 : ℝ) < x ∧ x < 1 / a → 0 < 1 / x - a ∧ 1 / a < x → 1 / x - a < 0) := sorry

theorem part2_range_a (a : ℝ) :
  (∀ x > 0, Real.log x - a * x + 1 ≤ 0) → 1 ≤ a := sorry

end part1_monotonicity_part2_range_a_l207_207912


namespace smallest_lucky_number_theorem_specific_lucky_number_theorem_l207_207398

-- Definitions based on the given conditions
def is_lucky_number (M : ℕ) : Prop :=
  ∃ (A B : ℕ), (M = A * B) ∧
               (A ≥ B) ∧
               (A ≥ 10 ∧ A ≤ 99) ∧
               (B ≥ 10 ∧ B ≤ 99) ∧
               (A / 10 = B / 10) ∧
               (A % 10 + B % 10 = 6)

def smallest_lucky_number : ℕ :=
  165

def P (M A B : ℕ) := A + B
def Q (M A B : ℕ) := A - B

def specific_lucky_number (M A B : ℕ) : Prop :=
  M = A * B ∧ (P M A B) / (Q M A B) % 7 = 0

-- Theorems to prove
theorem smallest_lucky_number_theorem :
  ∃ M, is_lucky_number M ∧ M = smallest_lucky_number := by
  sorry

theorem specific_lucky_number_theorem :
  ∃ M A B, is_lucky_number M ∧ specific_lucky_number M A B ∧ M = 3968 := by
  sorry

end smallest_lucky_number_theorem_specific_lucky_number_theorem_l207_207398


namespace range_of_m_l207_207686

def f (x : ℝ) : ℝ := x^3 - 3 * x - 1

theorem range_of_m (m : ℝ) (h : ∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ f x1 = m ∧ f x2 = m ∧ f x3 = m) : -3 < m ∧ m < 1 := 
sorry

end range_of_m_l207_207686


namespace rectangle_area_given_diagonal_l207_207385

noncomputable def area_of_rectangle (x : ℝ) : ℝ :=
  1250 - x^2 / 2

theorem rectangle_area_given_diagonal (P : ℝ) (x : ℝ) (A : ℝ) :
  P = 100 → x^2 = (P / 2)^2 - 2 * A → A = area_of_rectangle x :=
by
  intros hP hx
  sorry

end rectangle_area_given_diagonal_l207_207385


namespace range_values_y_div_x_l207_207051

-- Given the equation of the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 4 * x - 6 * y + 12 = 0

-- Prove that the range of values for y / x is [ (6 - 2 * sqrt 3) / 3, (6 + 2 * sqrt 3) / 3 ]
theorem range_values_y_div_x :
  (∀ x y : ℝ, circle_eq x y → (∃ k : ℝ, y = k * x) → 
  ( (6 - 2 * Real.sqrt 3) / 3 ≤ y / x ∧ y / x ≤ (6 + 2 * Real.sqrt 3) / 3 )) :=
sorry

end range_values_y_div_x_l207_207051


namespace Papi_Calot_plants_l207_207672

theorem Papi_Calot_plants :
  let initial_potatoes_plants := 10 * 25
  let initial_carrots_plants := 15 * 30
  let initial_onions_plants := 12 * 20
  let total_potato_plants := initial_potatoes_plants + 20
  let total_carrot_plants := initial_carrots_plants + 30
  let total_onion_plants := initial_onions_plants + 10
  total_potato_plants = 270 ∧
  total_carrot_plants = 480 ∧
  total_onion_plants = 250 := by
  sorry

end Papi_Calot_plants_l207_207672


namespace largest_possible_value_of_N_l207_207161

theorem largest_possible_value_of_N :
  ∃ N : ℕ, (∀ d : ℕ, (d ∣ N) → (d = 1 ∨ d = N ∨ (∃ k : ℕ, d = 3 ∨ d=k ∨ d=441 / k))) ∧
            ((21 * 3) ∣ N) ∧
            (N = 441) :=
by
  sorry

end largest_possible_value_of_N_l207_207161


namespace greatest_possible_number_of_blue_chips_l207_207192

-- Definitions based on conditions
def total_chips : Nat := 72

-- Definition of the relationship between red and blue chips where p is a prime number
def is_prime (n : Nat) : Prop := Nat.Prime n

def satisfies_conditions (r b p : Nat) : Prop :=
  r + b = total_chips ∧ r = b + p ∧ is_prime p

-- The statement to prove
theorem greatest_possible_number_of_blue_chips (r b p : Nat) 
  (h : satisfies_conditions r b p) : b = 35 := 
sorry

end greatest_possible_number_of_blue_chips_l207_207192


namespace complete_the_square_l207_207415

theorem complete_the_square (x : ℝ) :
  (x^2 + 8*x + 9 = 0) ↔ ((x + 4)^2 = 7) :=
sorry

end complete_the_square_l207_207415


namespace each_wolf_kills_one_deer_l207_207108

-- Definitions based on conditions
def hunting_wolves : Nat := 4
def additional_wolves : Nat := 16
def wolves_per_pack : Nat := hunting_wolves + additional_wolves
def meat_per_wolf_per_day : Nat := 8
def days_between_hunts : Nat := 5
def meat_per_wolf : Nat := meat_per_wolf_per_day * days_between_hunts
def total_meat_required : Nat := wolves_per_pack * meat_per_wolf
def meat_per_deer : Nat := 200
def deer_needed : Nat := total_meat_required / meat_per_deer
def deer_per_wolf_needed : Nat := deer_needed / hunting_wolves

-- Lean statement to prove
theorem each_wolf_kills_one_deer (hunting_wolves : Nat := 4) (additional_wolves : Nat := 16) 
    (meat_per_wolf_per_day : Nat := 8) (days_between_hunts : Nat := 5) 
    (meat_per_deer : Nat := 200) : deer_per_wolf_needed = 1 := 
by
  -- Proof required here
  sorry

end each_wolf_kills_one_deer_l207_207108


namespace main_theorem_l207_207511

-- Define the function f(x)
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Condition: f is symmetric about x = 1
def symmetric_about_one (a b c : ℝ) : Prop := 
  ∀ x : ℝ, f a b c (1 - x) = f a b c (1 + x)

-- Main statement
theorem main_theorem (a b c : ℝ) (h₁ : 0 < a) (h₂ : symmetric_about_one a b c) :
  ∀ x : ℝ, f a b c (2^x) > f a b c (3^x) :=
sorry

end main_theorem_l207_207511


namespace sum_of_squares_l207_207730

theorem sum_of_squares (x y : ℝ) (h1 : x + y = 22) (h2 : x * y = 40) : x^2 + y^2 = 404 :=
  sorry

end sum_of_squares_l207_207730


namespace train_average_speed_l207_207292

theorem train_average_speed (x : ℝ) (h1 : x > 0) :
  let d1 := x
  let d2 := 2 * x
  let s1 := 50
  let s2 := 20
  let t1 := d1 / s1
  let t2 := d2 / s2
  let total_distance := d1 + d2
  let total_time := t1 + t2
  let average_speed := total_distance / total_time
  average_speed = 25 := 
by
  sorry

end train_average_speed_l207_207292


namespace inequality_relation_l207_207952

theorem inequality_relation (a b : ℝ) :
  (∃ a b : ℝ, a > b ∧ ¬(1/a < 1/b)) ∧ (∃ a b : ℝ, (1/a < 1/b) ∧ ¬(a > b)) :=
by {
  sorry
}

end inequality_relation_l207_207952


namespace average_chemistry_mathematics_l207_207479

-- Define the conditions 
variable {P C M : ℝ} -- Marks in Physics, Chemistry, and Mathematics

-- The given condition in the problem
theorem average_chemistry_mathematics (h : P + C + M = P + 130) : (C + M) / 2 = 65 := 
by
  -- This will be the main proof block (we use 'sorry' to omit the actual proof)
  sorry

end average_chemistry_mathematics_l207_207479


namespace park_area_l207_207203

variable (length width : ℝ)
variable (cost_per_meter total_cost : ℝ)
variable (ratio_length ratio_width : ℝ)
variable (x : ℝ)

def rectangular_park_ratio (length width : ℝ) (ratio_length ratio_width : ℝ) : Prop :=
  length / width = ratio_length / ratio_width

def fencing_cost (cost_per_meter total_cost : ℝ) (perimeter : ℝ) : Prop :=
  total_cost = cost_per_meter * perimeter

theorem park_area (length width : ℝ) (cost_per_meter total_cost : ℝ)
  (ratio_length ratio_width : ℝ) (x : ℝ)
  (h1 : rectangular_park_ratio length width ratio_length ratio_width)
  (h2 : cost_per_meter = 0.70)
  (h3 : total_cost = 175)
  (h4 : ratio_length = 3)
  (h5 : ratio_width = 2)
  (h6 : length = 3 * x)
  (h7 : width = 2 * x)
  (h8 : fencing_cost cost_per_meter total_cost (2 * (length + width))) :
  length * width = 3750 := by
  sorry

end park_area_l207_207203


namespace ages_when_john_is_50_l207_207710

variable (age_john age_alice age_mike : ℕ)

-- Given conditions:
-- John is 10 years old
def john_is_10 : age_john = 10 := by sorry

-- Alice is twice John's age
def alice_is_twice_john : age_alice = 2 * age_john := by sorry

-- Mike is 4 years younger than Alice
def mike_is_4_years_younger : age_mike = age_alice - 4 := by sorry

-- Prove that when John is 50 years old, Alice will be 60 years old, and Mike will be 56 years old
theorem ages_when_john_is_50 : age_john = 50 → age_alice = 60 ∧ age_mike = 56 := 
by 
  intro h
  sorry

end ages_when_john_is_50_l207_207710


namespace profit_percentage_before_decrease_l207_207477

-- Defining the conditions as Lean definitions
def newManufacturingCost : ℝ := 50
def oldManufacturingCost : ℝ := 80
def profitPercentageNew : ℝ := 0.5

-- Defining the problem as a theorem in Lean
theorem profit_percentage_before_decrease
  (P : ℝ)
  (hP : profitPercentageNew * P = P - newManufacturingCost) :
  ((P - oldManufacturingCost) / P) * 100 = 20 := 
by
  sorry

end profit_percentage_before_decrease_l207_207477


namespace y_intercept_of_line_l207_207518

theorem y_intercept_of_line : ∀ (x y : ℝ), (3 * x - 4 * y = 12) → (x = 0) → (y = -3) :=
by
  intros x y h_eq h_x0
  sorry

end y_intercept_of_line_l207_207518


namespace product_of_areas_eq_square_of_volume_l207_207451

variable (x y z : ℝ)

def area_xy : ℝ := x * y
def area_yz : ℝ := y * z
def area_zx : ℝ := z * x

theorem product_of_areas_eq_square_of_volume :
  (area_xy x y) * (area_yz y z) * (area_zx z x) = (x * y * z) ^ 2 :=
by
  sorry

end product_of_areas_eq_square_of_volume_l207_207451


namespace nancy_shoes_l207_207714

theorem nancy_shoes (boots_slippers_relation : ∀ (boots slippers : ℕ), slippers = boots + 9)
                    (heels_relation : ∀ (boots slippers heels : ℕ), heels = 3 * (boots + slippers)) :
                    ∃ (total_individual_shoes : ℕ), total_individual_shoes = 168 :=
by
  let boots := 6
  let slippers := boots + 9
  let total_pairs := boots + slippers
  let heels := 3 * total_pairs
  let total_pairs_shoes := boots + slippers + heels
  let total_individual_shoes := 2 * total_pairs_shoes
  use total_individual_shoes
  exact sorry

end nancy_shoes_l207_207714


namespace i_pow_2006_l207_207554

-- Definitions based on given conditions
def i : ℂ := Complex.I

-- Cyclic properties of i (imaginary unit)
axiom i_pow_1 : i^1 = i
axiom i_pow_2 : i^2 = -1
axiom i_pow_3 : i^3 = -i
axiom i_pow_4 : i^4 = 1
axiom i_pow_5 : i^5 = i

-- The proof statement
theorem i_pow_2006 : (i^2006 = -1) :=
by
  sorry

end i_pow_2006_l207_207554


namespace cube_problem_l207_207930

theorem cube_problem (n : ℕ) (h1 : n > 3) :
  (12 * (n - 4) = (n - 2)^3) → n = 5 :=
by {
  sorry
}

end cube_problem_l207_207930


namespace determinant_zero_l207_207070

theorem determinant_zero (α β : ℝ) :
  Matrix.det ![
    ![0, Real.sin α, -Real.cos α],
    ![-Real.sin α, 0, Real.sin β],
    ![Real.cos α, -Real.sin β, 0]
  ] = 0 :=
by sorry

end determinant_zero_l207_207070


namespace total_tshirts_bought_l207_207376

-- Given conditions
def white_packs : ℕ := 3
def white_tshirts_per_pack : ℕ := 6
def blue_packs : ℕ := 2
def blue_tshirts_per_pack : ℕ := 4

-- Theorem statement: Total number of T-shirts Dave bought
theorem total_tshirts_bought : white_packs * white_tshirts_per_pack + blue_packs * blue_tshirts_per_pack = 26 := by
  sorry

end total_tshirts_bought_l207_207376


namespace function_relationship_value_of_x_when_y_is_1_l207_207481

variable (x y : ℝ) (k : ℝ)

-- Conditions
def inverse_proportion (x y : ℝ) : Prop :=
  ∃ k : ℝ, y = k / (x - 3)

axiom condition_1 : inverse_proportion x y
axiom condition_2 : y = 5 ∧ x = 4

-- Statements to be proved
theorem function_relationship :
  ∃ k : ℝ, (y = k / (x - 3)) ∧ (y = 5 ∧ x = 4 → k = 5) :=
by
  sorry

theorem value_of_x_when_y_is_1 (hy : y = 1) :
  ∃ x : ℝ, (y = 5 / (x - 3)) ∧ x = 8 :=
by
  sorry

end function_relationship_value_of_x_when_y_is_1_l207_207481


namespace greatest_three_digit_multiple_of_17_is_986_l207_207417

def is_three_digit_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

def is_multiple_of_17 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 17 * k

def greatest_three_digit_multiple_of_17 : ℕ :=
  986

theorem greatest_three_digit_multiple_of_17_is_986 :
  ∀ n : ℕ, is_three_digit_number n → is_multiple_of_17 n → n ≤ greatest_three_digit_multiple_of_17 :=
by
  sorry

end greatest_three_digit_multiple_of_17_is_986_l207_207417


namespace compute_expression_l207_207600

theorem compute_expression :
  20 * (150 / 3 + 40 / 5 + 16 / 25 + 2) = 1212.8 :=
by
  -- skipping the proof steps
  sorry

end compute_expression_l207_207600


namespace zachary_cans_first_day_l207_207270

theorem zachary_cans_first_day :
  ∃ (first_day_cans : ℕ),
    ∃ (second_day_cans : ℕ),
      ∃ (third_day_cans : ℕ),
        ∃ (seventh_day_cans : ℕ),
          second_day_cans = 9 ∧
          third_day_cans = 14 ∧
          (∀ (n : ℕ), 2 ≤ n ∧ n < 7 → third_day_cans = second_day_cans + 5) →
          seventh_day_cans = 34 ∧
          first_day_cans = second_day_cans - 5 ∧
          first_day_cans = 4 :=

by
  sorry

end zachary_cans_first_day_l207_207270


namespace distinct_paths_to_B_and_C_l207_207028

def paths_to_red_arrows : ℕ × ℕ := (1, 2)
def paths_from_first_red_to_blue : ℕ := 3 * 2
def paths_from_second_red_to_blue : ℕ := 4 * 2
def total_paths_to_blue_arrows : ℕ := paths_from_first_red_to_blue + paths_from_second_red_to_blue

def paths_from_first_two_blue_to_green : ℕ := 5 * 4
def paths_from_third_and_fourth_blue_to_green : ℕ := 6 * 4
def total_paths_to_green_arrows : ℕ := paths_from_first_two_blue_to_green + paths_from_third_and_fourth_blue_to_green

def paths_to_B : ℕ := total_paths_to_green_arrows * 3
def paths_to_C : ℕ := total_paths_to_green_arrows * 4
def total_paths : ℕ := paths_to_B + paths_to_C

theorem distinct_paths_to_B_and_C :
  total_paths = 4312 := 
by
  -- all conditions can be used within this proof
  sorry

end distinct_paths_to_B_and_C_l207_207028


namespace fill_tub_in_seconds_l207_207683

theorem fill_tub_in_seconds 
  (faucet_rate : ℚ)
  (four_faucet_rate : ℚ := 4 * faucet_rate)
  (three_faucet_rate : ℚ := 3 * faucet_rate)
  (time_for_100_gallons_in_minutes : ℚ := 6)
  (time_for_100_gallons_in_seconds : ℚ := time_for_100_gallons_in_minutes * 60)
  (volume_100_gallons : ℚ := 100)
  (rate_per_three_faucets_in_gallons_per_second : ℚ := volume_100_gallons / time_for_100_gallons_in_seconds)
  (rate_per_faucet : ℚ := rate_per_three_faucets_in_gallons_per_second / 3)
  (rate_per_four_faucets : ℚ := 4 * rate_per_faucet)
  (volume_50_gallons : ℚ := 50)
  (expected_time_seconds : ℚ := volume_50_gallons / rate_per_four_faucets) :
  expected_time_seconds = 135 :=
sorry

end fill_tub_in_seconds_l207_207683


namespace single_point_graph_l207_207533

theorem single_point_graph (d : ℝ) :
  (∀ x y : ℝ, 3 * x^2 + y^2 + 6 * x - 8 * y + d = 0 → x = -1 ∧ y = 4) → d = 19 :=
by
  sorry

end single_point_graph_l207_207533


namespace brookdale_avg_temp_l207_207693

def highs : List ℤ := [51, 64, 60, 59, 48, 55]
def lows : List ℤ := [42, 49, 47, 43, 41, 44]

def average_temperature : ℚ :=
  let total_sum := highs.sum + lows.sum
  let count := (highs.length + lows.length : ℚ)
  total_sum / count

theorem brookdale_avg_temp :
  average_temperature = 49.4 :=
by
  -- The proof goes here
  sorry

end brookdale_avg_temp_l207_207693


namespace ellipse_foci_y_axis_range_l207_207560

noncomputable def is_ellipse_with_foci_on_y_axis (k : ℝ) : Prop :=
  (k > 5) ∧ (k < 10) ∧ (10 - k > k - 5)

theorem ellipse_foci_y_axis_range (k : ℝ) :
  is_ellipse_with_foci_on_y_axis k ↔ 5 < k ∧ k < 7.5 := 
by
  sorry

end ellipse_foci_y_axis_range_l207_207560


namespace jane_coffees_l207_207244

open Nat

theorem jane_coffees (b m c n : Nat) 
  (h1 : b + m + c = 6)
  (h2 : 75 * b + 60 * m + 100 * c = 100 * n) :
  c = 1 :=
by sorry

end jane_coffees_l207_207244


namespace sam_bought_nine_books_l207_207043

-- Definitions based on the conditions
def initial_money : ℕ := 79
def cost_per_book : ℕ := 7
def money_left : ℕ := 16

-- The amount spent on books
def money_spent_on_books : ℕ := initial_money - money_left

-- The number of books bought
def number_of_books (spent : ℕ) (cost : ℕ) : ℕ := spent / cost

-- Let x be the number of books bought and prove x = 9
theorem sam_bought_nine_books : number_of_books money_spent_on_books cost_per_book = 9 :=
by
  sorry

end sam_bought_nine_books_l207_207043


namespace solve_farm_l207_207233

def farm_problem (P H L T : ℕ) : Prop :=
  L = 4 * P + 2 * H ∧
  T = P + H ∧
  L = 3 * T + 36 →
  P = H + 36

-- Theorem statement
theorem solve_farm : ∃ P H L T : ℕ, farm_problem P H L T :=
by sorry

end solve_farm_l207_207233


namespace total_pets_combined_l207_207340

def teddy_dogs : ℕ := 7
def teddy_cats : ℕ := 8
def ben_dogs : ℕ := teddy_dogs + 9
def dave_cats : ℕ := teddy_cats + 13
def dave_dogs : ℕ := teddy_dogs - 5

def teddy_pets : ℕ := teddy_dogs + teddy_cats
def ben_pets : ℕ := ben_dogs
def dave_pets : ℕ := dave_cats + dave_dogs

def total_pets : ℕ := teddy_pets + ben_pets + dave_pets

theorem total_pets_combined : total_pets = 54 :=
by
  -- proof goes here
  sorry

end total_pets_combined_l207_207340


namespace gum_pieces_per_package_l207_207078

theorem gum_pieces_per_package :
  (∀ (packages pieces each_package : ℕ), packages = 9 ∧ pieces = 135 → each_package = pieces / packages → each_package = 15) := 
by
  intros packages pieces each_package
  sorry

end gum_pieces_per_package_l207_207078


namespace number_of_correct_statements_l207_207996

def line : Type := sorry
def plane : Type := sorry
def parallel (x y : line) : Prop := sorry
def perpendicular (x : line) (y : plane) : Prop := sorry
def subset (x : line) (y : plane) : Prop := sorry
def skew (x y : line) : Prop := sorry

variable (m n : line) -- two different lines
variable (alpha beta : plane) -- two different planes

theorem number_of_correct_statements :
  (¬parallel m alpha ∨ subset n alpha ∧ parallel m n) ∧
  (parallel m alpha ∧ perpendicular alpha beta ∧ perpendicular m n ∧ perpendicular n beta) ∧
  (subset m alpha ∧ subset n beta ∧ perpendicular m n) ∧
  (skew m n ∧ subset m alpha ∧ subset n beta ∧ parallel m beta ∧ parallel n alpha) :=
sorry

end number_of_correct_statements_l207_207996


namespace solve_eq1_solve_eq2_l207_207222

theorem solve_eq1 (x : ℤ) : x - 2 * (5 + x) = -4 → x = -6 := by
  sorry

theorem solve_eq2 (x : ℤ) : (2 * x - 1) / 2 = 1 - (3 - x) / 4 → x = 1 := by
  sorry

end solve_eq1_solve_eq2_l207_207222


namespace solve_for_x_l207_207048

theorem solve_for_x (x y : ℝ) (h₁ : y = 1 / (4 * x + 2)) (h₂ : y = 1 / 2) : x = 0 :=
by
  -- Placeholder for the proof
  sorry

end solve_for_x_l207_207048


namespace no_integer_a_for_integer_roots_l207_207285

theorem no_integer_a_for_integer_roots :
  ∀ a : ℤ, ¬ (∃ x : ℤ, x^2 - 2023 * x + 2022 * a + 1 = 0) := 
by
  intro a
  rintro ⟨x, hx⟩
  sorry

end no_integer_a_for_integer_roots_l207_207285


namespace no_prime_p_satisfies_l207_207001

noncomputable def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

theorem no_prime_p_satisfies (p : ℕ) (hp : Nat.Prime p) (hp1 : is_perfect_square (7 * p + 3 ^ p - 4)) : False :=
by
  sorry

end no_prime_p_satisfies_l207_207001


namespace sum_every_second_term_is_1010_l207_207949

def sequence_sum (n : ℕ) (a d : ℤ) : ℤ :=
  n * (2 * a + (n - 1) * d) / 2

theorem sum_every_second_term_is_1010 :
  ∃ (x1 : ℤ) (d : ℤ) (S : ℤ), 
  (sequence_sum 2020 x1 d = 6060) ∧
  (d = 2) ∧
  (S = (1010 : ℤ)) ∧ 
  (2 * S + 4040 = 6060) :=
  sorry

end sum_every_second_term_is_1010_l207_207949


namespace trig_identity_sin_cos_l207_207982

theorem trig_identity_sin_cos
  (a : ℝ)
  (h : Real.sin (Real.pi / 3 - a) = 1 / 3) :
  Real.cos (5 * Real.pi / 6 - a) = -1 / 3 :=
by
  sorry

end trig_identity_sin_cos_l207_207982


namespace no_rational_solution_5x2_plus_3y2_eq_1_l207_207105

theorem no_rational_solution_5x2_plus_3y2_eq_1 :
  ¬ ∃ (x y : ℚ), 5 * x^2 + 3 * y^2 = 1 := 
sorry

end no_rational_solution_5x2_plus_3y2_eq_1_l207_207105


namespace anand_income_l207_207529

theorem anand_income
  (x y : ℕ)
  (h1 : 5 * x - 3 * y = 800)
  (h2 : 4 * x - 2 * y = 800) : 
  5 * x = 2000 := 
sorry

end anand_income_l207_207529


namespace parallel_lines_l207_207568

theorem parallel_lines (a : ℝ) :
  (∀ x y, x + a^2 * y + 6 = 0 → (a - 2) * x + 3 * a * y + 2 * a = 0) ↔ (a = 0 ∨ a = -1) :=
by
  sorry

end parallel_lines_l207_207568


namespace find_sol_y_pct_l207_207689

-- Define the conditions
def sol_x_vol : ℕ := 200            -- Volume of solution x in milliliters
def sol_y_vol : ℕ := 600            -- Volume of solution y in milliliters
def sol_x_pct : ℕ := 10             -- Percentage of alcohol in solution x
def final_sol_pct : ℕ := 25         -- Percentage of alcohol in the final solution
def final_sol_vol := sol_x_vol + sol_y_vol -- Total volume of the final solution

-- Define the problem statement
theorem find_sol_y_pct (sol_x_vol sol_y_vol final_sol_vol : ℕ) 
  (sol_x_pct final_sol_pct : ℕ) : 
  (600 * 10 + sol_y_vol * 30) / 800 = 25 :=
by
  sorry

end find_sol_y_pct_l207_207689


namespace regression_estimate_l207_207016

theorem regression_estimate (x : ℝ) (h : x = 28) : 4.75 * x + 257 = 390 :=
by
  rw [h]
  norm_num

end regression_estimate_l207_207016


namespace find_x_l207_207947

theorem find_x (x : ℚ) (h : x ≠ 2 ∧ x ≠ 4/5) :
  (x^2 - 11*x + 24)/(x - 2) + (5*x^2 + 22*x - 48)/(5*x - 4) = -7 → x = -4/3 :=
by
  intro h1
  sorry

end find_x_l207_207947


namespace B_joined_after_8_months_l207_207160

-- Define the initial investments and time
def A_investment : ℕ := 36000
def B_investment : ℕ := 54000
def profit_ratio_A_B := 2 / 1

-- Define a proposition which states that B joined the business after x = 8 months
theorem B_joined_after_8_months (x : ℕ) (h : (A_investment * 12) / (B_investment * (12 - x)) = profit_ratio_A_B) : x = 8 :=
by
  sorry

end B_joined_after_8_months_l207_207160


namespace min_value_of_x_plus_2y_l207_207153

noncomputable def min_value_condition (x y : ℝ) : Prop :=
x > -1 ∧ y > 0 ∧ (1 / (x + 1) + 2 / y = 1)

theorem min_value_of_x_plus_2y (x y : ℝ) (h : min_value_condition x y) : x + 2 * y ≥ 8 :=
sorry

end min_value_of_x_plus_2y_l207_207153


namespace ratio_of_volumes_of_cones_l207_207928

theorem ratio_of_volumes_of_cones (r θ h1 h2 : ℝ) (hθ : 3 * θ + 4 * θ = 2 * π)
    (hr1 : r₁ = 3 * r / 7) (hr2 : r₂ = 4 * r / 7) :
    let V₁ := (1 / 3) * π * r₁^2 * h1
    let V₂ := (1 / 3) * π * r₂^2 * h2
    V₁ / V₂ = (9 : ℝ) / 16 := by
  sorry

end ratio_of_volumes_of_cones_l207_207928


namespace find_z_l207_207753

theorem find_z (z : ℚ) : (7 + 11 + 23) / 3 = (15 + z) / 2 → z = 37 / 3 :=
by
  sorry

end find_z_l207_207753


namespace jordans_greatest_average_speed_l207_207726

noncomputable def is_palindrome (n : ℕ) : Prop :=
  let s := n.digits 10
  s.reverse = s

theorem jordans_greatest_average_speed :
  ∃ (v : ℕ), 
  ∃ (d : ℕ), 
  ∃ (end_reading : ℕ), 
  is_palindrome 72327 ∧ 
  is_palindrome end_reading ∧ 
  72327 < end_reading ∧ 
  end_reading - 72327 = d ∧ 
  d ≤ 240 ∧ 
  end_reading ≤ 72327 + 240 ∧ 
  v = d / 4 ∧ 
  v = 50 :=
sorry

end jordans_greatest_average_speed_l207_207726


namespace initial_big_bottles_l207_207535

theorem initial_big_bottles (B : ℝ)
  (initial_small : ℝ := 6000)
  (sold_small : ℝ := 0.11)
  (sold_big : ℝ := 0.12)
  (remaining_total : ℝ := 18540) :
  (initial_small * (1 - sold_small) + B * (1 - sold_big) = remaining_total) → B = 15000 :=
by
  intro h
  sorry

end initial_big_bottles_l207_207535


namespace distinct_triangles_in_3x3_grid_l207_207065

theorem distinct_triangles_in_3x3_grid : 
  let num_points := 9 
  let total_combinations := Nat.choose num_points 3 
  let degenerate_cases := 8
  total_combinations - degenerate_cases = 76 := 
by
  sorry

end distinct_triangles_in_3x3_grid_l207_207065


namespace correct_algebraic_expression_l207_207306

theorem correct_algebraic_expression
  (A : String := "1 1/2 a")
  (B : String := "a × b")
  (C : String := "a ÷ b")
  (D : String := "2a") :
  D = "2a" :=
by {
  -- Explanation based on the conditions provided
  -- A: "1 1/2 a" is not properly formatted. Correct format involves improper fraction for multiplication.
  -- B: "a × b" should avoid using the multiplication sign explicitly.
  -- C: "a ÷ b" should be written as a fraction a/b.
  -- D: "2a" is correctly formatted.
  sorry
}

end correct_algebraic_expression_l207_207306


namespace minimum_students_to_share_birthday_l207_207284

theorem minimum_students_to_share_birthday (k : ℕ) (m : ℕ) (n : ℕ) (hcond1 : k = 366) (hcond2 : m = 2) (hineq : n > k * m) : n ≥ 733 := 
by
  -- since k = 366 and m = 2
  have hk : k = 366 := hcond1
  have hm : m = 2 := hcond2
  -- thus: n > 366 * 2
  have hn : n > 732 := by
    rw [hk, hm] at hineq
    exact hineq
  -- hence, n ≥ 733
  exact Nat.succ_le_of_lt hn

end minimum_students_to_share_birthday_l207_207284


namespace balance_equation_l207_207000

variable (G Y W B : ℝ)
variable (balance1 : 4 * G = 8 * B)
variable (balance2 : 3 * Y = 7.5 * B)
variable (balance3 : 8 * B = 6 * W)

theorem balance_equation : 5 * G + 3 * Y + 4 * W = 23.5 * B := by
  sorry

end balance_equation_l207_207000


namespace solve1_solve2_solve3_solve4_l207_207820

noncomputable section

-- Problem 1
theorem solve1 (x : ℝ) : x^2 + 2 * x = 0 ↔ x = 0 ∨ x = -2 := sorry

-- Problem 2
theorem solve2 (x : ℝ) : (x + 1)^2 - 144 = 0 ↔ x = 11 ∨ x = -13 := sorry

-- Problem 3
theorem solve3 (x : ℝ) : 3 * (x - 2)^2 = x * (x - 2) ↔ x = 2 ∨ x = 3 := sorry

-- Problem 4
theorem solve4 (x : ℝ) : x^2 + 5 * x - 1 = 0 ↔ x = (-5 + Real.sqrt 29) / 2 ∨ x = (-5 - Real.sqrt 29) / 2 := sorry

end solve1_solve2_solve3_solve4_l207_207820


namespace outlier_attribute_l207_207344

/-- Define the given attributes of the Dragon -/
def one_eyed := "одноокий"
def two_eared := "двуухий"
def three_tailed := "треххвостый"
def four_legged := "четырехлапый"
def five_spiked := "пятиглый"

/-- Define a predicate to check if an attribute contains doubled letters -/
def has_doubled_letters (s : String) : Bool :=
  let chars := s.toList
  chars.any (fun ch => chars.count ch > 1)

/-- Prove that "четырехлапый" (four-legged) does not fit the pattern of containing doubled letters -/
theorem outlier_attribute : ¬ has_doubled_letters four_legged :=
by
  -- Proof would be inserted here
  sorry

end outlier_attribute_l207_207344


namespace calculate_expression_l207_207421

theorem calculate_expression : (Real.sqrt 8 + Real.sqrt (1 / 2)) * Real.sqrt 32 = 20 := by
  sorry

end calculate_expression_l207_207421


namespace pto_shirts_total_cost_l207_207806

theorem pto_shirts_total_cost :
  let cost_Kindergartners : ℝ := 101 * 5.80
  let cost_FirstGraders : ℝ := 113 * 5.00
  let cost_SecondGraders : ℝ := 107 * 5.60
  let cost_ThirdGraders : ℝ := 108 * 5.25
  cost_Kindergartners + cost_FirstGraders + cost_SecondGraders + cost_ThirdGraders = 2317.00 := by
  sorry

end pto_shirts_total_cost_l207_207806


namespace triangle_perimeter_correct_l207_207999

def side_a : ℕ := 15
def side_b : ℕ := 8
def side_c : ℕ := 10
def perimeter (a b c : ℕ) : ℕ := a + b + c

theorem triangle_perimeter_correct :
  perimeter side_a side_b side_c = 33 := by
sorry

end triangle_perimeter_correct_l207_207999


namespace functions_with_inverses_l207_207832

-- Definitions for the conditions
def passes_Horizontal_Line_Test_A : Prop := false
def passes_Horizontal_Line_Test_B : Prop := true
def passes_Horizontal_Line_Test_C : Prop := true
def passes_Horizontal_Line_Test_D : Prop := false
def passes_Horizontal_Line_Test_E : Prop := false

-- Proof statement
theorem functions_with_inverses :
  (passes_Horizontal_Line_Test_A = false) ∧
  (passes_Horizontal_Line_Test_B = true) ∧
  (passes_Horizontal_Line_Test_C = true) ∧
  (passes_Horizontal_Line_Test_D = false) ∧
  (passes_Horizontal_Line_Test_E = false) →
  ([B, C] = which_functions_have_inverses) :=
sorry

end functions_with_inverses_l207_207832


namespace Danny_more_than_Larry_l207_207369

/-- Keith scored 3 points. --/
def Keith_marks : Nat := 3

/-- Larry scored 3 times as many marks as Keith. --/
def Larry_marks : Nat := 3 * Keith_marks

/-- The total marks scored by Keith, Larry, and Danny is 26. --/
def total_marks (D : Nat) : Prop := Keith_marks + Larry_marks + D = 26

/-- Prove the number of more marks Danny scored than Larry is 5. --/
theorem Danny_more_than_Larry (D : Nat) (h : total_marks D) : D - Larry_marks = 5 :=
sorry

end Danny_more_than_Larry_l207_207369


namespace difference_in_combined_area_l207_207553

-- Define the dimensions of the two rectangular sheets of paper
def paper1_length : ℝ := 11
def paper1_width : ℝ := 17
def paper2_length : ℝ := 8.5
def paper2_width : ℝ := 11

-- Define the areas of one side of each sheet
def area1 : ℝ := paper1_length * paper1_width -- 187
def area2 : ℝ := paper2_length * paper2_width -- 93.5

-- Define the combined areas of front and back of each sheet
def combined_area1 : ℝ := 2 * area1 -- 374
def combined_area2 : ℝ := 2 * area2 -- 187

-- Prove that the difference in combined area is 187
theorem difference_in_combined_area : combined_area1 - combined_area2 = 187 :=
by 
  -- Using the definitions above to simplify the goal
  sorry

end difference_in_combined_area_l207_207553


namespace max_lift_times_l207_207632

theorem max_lift_times (n : ℕ) :
  (2 * 30 * 10) = (2 * 25 * n) → n = 12 :=
by
  sorry

end max_lift_times_l207_207632


namespace problem1_problem2_l207_207040

variable {a b c : ℝ}

-- Conditions: a, b, c are positive numbers and a^(3/2) + b^(3/2) + c^(3/2) = 1
axiom pos_a : 0 < a
axiom pos_b : 0 < b
axiom pos_c : 0 < c
axiom sum_pow : a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Problem (1): Prove abc ≤ 1/9 given the conditions
theorem problem1 : abc ≤ 1 / 9 := by
  sorry

-- Problem (2): Prove (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ (1 / (2 * (abc) ^ (1/2))) given the conditions
theorem problem2 : (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * (abc) ^ (1/2)) := by
  sorry

end problem1_problem2_l207_207040


namespace hyperbola_center_l207_207232

-- Definitions based on conditions
def hyperbola (x y : ℝ) : Prop := ((4 * x + 8) ^ 2 / 16) - ((5 * y - 5) ^ 2 / 25) = 1

-- Theorem statement
theorem hyperbola_center : ∀ x y : ℝ, hyperbola x y → (x, y) = (-2, 1) := 
  by
    sorry

end hyperbola_center_l207_207232


namespace Maryann_frees_all_friends_in_42_minutes_l207_207091

-- Definitions for the problem conditions
def time_to_pick_cheap_handcuffs := 6
def time_to_pick_expensive_handcuffs := 8
def number_of_friends := 3

-- Define the statement we need to prove
theorem Maryann_frees_all_friends_in_42_minutes :
  (time_to_pick_cheap_handcuffs + time_to_pick_expensive_handcuffs) * number_of_friends = 42 :=
by
  sorry

end Maryann_frees_all_friends_in_42_minutes_l207_207091


namespace speed_increase_percentage_l207_207517

variable (T : ℚ)  -- usual travel time in minutes
variable (v : ℚ)  -- usual speed

-- Conditions
-- Ivan usually arrives at 9:00 AM, traveling for T minutes at speed v.
-- When Ivan leaves 40 minutes late and drives 1.6 times his usual speed, he arrives at 8:35 AM
def usual_arrival_time : ℚ := 9 * 60  -- 9:00 AM in minutes

def time_when_late : ℚ := (9 * 60) + 40 - (25 + 40)  -- 8:35 AM in minutes

def increased_speed := 1.6 * v -- 60% increase in speed

def time_taken_with_increased_speed := T - 65

theorem speed_increase_percentage :
  ((T / (T - 40)) = 1.3) :=
by
-- assume the equation for usual time T in terms of increased speed is known
-- Use provided conditions and solve the equation to derive the result.
  sorry

end speed_increase_percentage_l207_207517


namespace grandma_red_bacon_bits_l207_207920

def mushrooms := 3
def cherry_tomatoes := 2 * mushrooms
def pickles := 4 * cherry_tomatoes
def bacon_bits := 4 * pickles
def red_bacon_bits := bacon_bits / 3

theorem grandma_red_bacon_bits : red_bacon_bits = 32 := by
  sorry

end grandma_red_bacon_bits_l207_207920


namespace relative_error_comparison_l207_207393

theorem relative_error_comparison :
  let error1 := 0.05
  let length1 := 25
  let error2 := 0.25
  let length2 := 125
  (error1 / length1) = (error2 / length2) :=
by
  sorry

end relative_error_comparison_l207_207393


namespace girls_points_l207_207346

theorem girls_points (g b : ℕ) (total_points : ℕ) (points_g : ℕ) (points_b : ℕ) :
  b = 9 * g ∧
  total_points = 10 * g * (10 * g - 1) ∧
  points_g = 2 * g * (10 * g - 1) ∧
  points_b = 4 * points_g ∧
  total_points = points_g + points_b
  → points_g = 18 := 
by
  sorry

end girls_points_l207_207346


namespace largest_constant_inequality_l207_207186

theorem largest_constant_inequality (x y z : ℝ) (h₀ : 0 ≤ x) (h₁ : 0 ≤ y) (h₂ : 0 ≤ z) :
  (y*z + z*x + x*y)^2 * (x + y + z) ≥ 4 * x*y*z * (x^2 + y^2 + z^2) :=
sorry

end largest_constant_inequality_l207_207186


namespace range_of_real_number_a_l207_207908

theorem range_of_real_number_a (a : ℝ) : (∀ (x : ℝ), 0 < x → a < x + 1/x) → a < 2 := 
by
  sorry

end range_of_real_number_a_l207_207908


namespace towel_area_decrease_l207_207059

theorem towel_area_decrease (L B : ℝ) (hL : 0 < L) (hB : 0 < B) :
  let A := L * B
  let L' := 0.80 * L
  let B' := 0.90 * B
  let A' := L' * B'
  A' = 0.72 * A →
  ((A - A') / A) * 100 = 28 :=
by
  intros _ _ _ _
  sorry

end towel_area_decrease_l207_207059


namespace part_I_part_II_l207_207023

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x * Real.log x + a

theorem part_I (a : ℝ) (h : ∀ x > 0, f x a ≥ 0) : a ≥ (1 : ℝ) / Real.exp 1 :=
sorry

theorem part_II (a x1 x2 x : ℝ) (hx1 : 0 < x1) (hx2 : x1 < x2) (hx : x1 < x ∧ x < x2) :
  (f x a - f x1 a) / (x - x1) < (f x a - f x2 a) / (x - x2) :=
sorry

end part_I_part_II_l207_207023


namespace coordinates_of_P_l207_207188

-- Define the point P with given coordinates
structure Point where
  x : ℝ
  y : ℝ

-- Define the point P(3, 5)
def P : Point := ⟨3, 5⟩

-- Define a theorem stating that the coordinates of P are (3, 5)
theorem coordinates_of_P : P = ⟨3, 5⟩ :=
  sorry

end coordinates_of_P_l207_207188


namespace original_number_is_45_l207_207576

theorem original_number_is_45 (x y : ℕ) (h1 : x + y = 9) (h2 : 10 * y + x = 10 * x + y + 9) : 10 * x + y = 45 := by
  sorry

end original_number_is_45_l207_207576


namespace smallest_solution_l207_207106

def equation (x : ℝ) := (3 * x) / (x - 3) + (3 * x^2 - 27) / x = 14

theorem smallest_solution :
  ∀ x : ℝ, equation x → x = (3 - Real.sqrt 333) / 6 :=
sorry

end smallest_solution_l207_207106


namespace problem_solution_l207_207960

theorem problem_solution (x y : ℕ) (hxy : x + y + x * y = 104) (hx : 0 < x) (hy : 0 < y) (hx30 : x < 30) (hy30 : y < 30) : 
  x + y = 20 := 
sorry

end problem_solution_l207_207960


namespace part1_part2_l207_207985

-- Statement for part (1)
theorem part1 (m : ℝ) : 
  (∀ x1 x2 : ℝ, (m - 1) * x1^2 + 3 * x1 - 2 = 0 ∧ 
               (m - 1) * x2^2 + 3 * x2 - 2 = 0 ∧ x1 ≠ x2) ↔ m > -1/8 :=
sorry

-- Statement for part (2)
theorem part2 (m : ℝ) : 
  (∃ x : ℝ, (m - 1) * x^2 + 3 * x - 2 = 0 ∧ ∀ y : ℝ, (m - 1) * y^2 + 3 * y - 2 = 0 → y = x) ↔ 
  (m = 1 ∨ m = -1/8) :=
sorry

end part1_part2_l207_207985


namespace doubled_dimensions_volume_l207_207818

theorem doubled_dimensions_volume (original_volume : ℝ) (length_factor width_factor height_factor : ℝ) 
  (h : original_volume = 3) 
  (hl : length_factor = 2)
  (hw : width_factor = 2)
  (hh : height_factor = 2) : 
  original_volume * length_factor * width_factor * height_factor = 24 :=
by
  sorry

end doubled_dimensions_volume_l207_207818


namespace acute_triangle_inequality_l207_207395

theorem acute_triangle_inequality
  (A B C : ℝ)
  (a b c : ℝ)
  (R : ℝ)
  (h1 : 0 < A ∧ A < π/2)
  (h2 : 0 < B ∧ B < π/2)
  (h3 : 0 < C ∧ C < π/2)
  (h4 : A + B + C = π)
  (h5 : R = 1)
  (h6 : a = 2 * R * Real.sin A)
  (h7 : b = 2 * R * Real.sin B)
  (h8 : c = 2 * R * Real.sin C) :
  (a / (1 - Real.sin A)) + (b / (1 - Real.sin B)) + (c / (1 - Real.sin C)) ≥ 18 + 12 * Real.sqrt 3 :=
by
  sorry

end acute_triangle_inequality_l207_207395


namespace number_of_selected_in_interval_l207_207007

noncomputable def systematic_sampling_group := (420: ℕ)
noncomputable def selected_people := (21: ℕ)
noncomputable def interval_start := (241: ℕ)
noncomputable def interval_end := (360: ℕ)
noncomputable def sampling_interval := systematic_sampling_group / selected_people
noncomputable def interval_length := interval_end - interval_start + 1

theorem number_of_selected_in_interval :
  interval_length / sampling_interval = 6 :=
by
  -- Placeholder for the proof
  sorry

end number_of_selected_in_interval_l207_207007


namespace division_quotient_l207_207615

theorem division_quotient (dividend divisor remainder quotient : Nat) 
  (h_dividend : dividend = 109)
  (h_divisor : divisor = 12)
  (h_remainder : remainder = 1)
  (h_division_equation : dividend = divisor * quotient + remainder)
  : quotient = 9 := 
by
  sorry

end division_quotient_l207_207615


namespace strictly_increasing_intervals_l207_207218

-- Define the function y = cos^2(x + π/2)
noncomputable def y (x : ℝ) : ℝ := (Real.cos (x + Real.pi / 2))^2

-- Define the assertion
theorem strictly_increasing_intervals (k : ℤ) : 
  StrictMonoOn y (Set.Icc (k * Real.pi) (k * Real.pi + Real.pi / 2)) :=
sorry

end strictly_increasing_intervals_l207_207218


namespace unique_positive_b_discriminant_zero_l207_207359

theorem unique_positive_b_discriminant_zero (c : ℚ) : 
  (∃! b : ℚ, b > 0 ∧ (b^2 + 3*b + 1/b)^2 - 4*c = 0) ↔ c = -1/2 :=
sorry

end unique_positive_b_discriminant_zero_l207_207359


namespace scheduled_conference_games_l207_207341

-- Definitions based on conditions
def num_divisions := 3
def teams_per_division := 4
def games_within_division := 3
def games_across_divisions := 2

-- Proof statement
theorem scheduled_conference_games :
  let teams_in_division := teams_per_division
  let div_game_count := games_within_division * (teams_in_division * (teams_in_division - 1) / 2) 
  let total_within_division := div_game_count * num_divisions
  let cross_div_game_count := (teams_in_division * games_across_divisions * (num_divisions - 1) * teams_in_division * num_divisions) / 2
  total_within_division + cross_div_game_count = 102 := 
by {
  sorry
}

end scheduled_conference_games_l207_207341


namespace g_1986_l207_207166

def g : ℕ → ℤ := sorry

axiom g_def : ∀ n : ℕ, g n ≥ 0
axiom g_one : g 1 = 3
axiom g_func_eq : ∀ (a b : ℕ), g (a + b) = g a + g b - 3 * g (a * b)

theorem g_1986 : g 1986 = 0 :=
by
  sorry

end g_1986_l207_207166


namespace inequality_proof_l207_207456

theorem inequality_proof (m n : ℝ) (h₁ : m > 0) (h₂ : n > 0) (h₃ : m + n = 1) :
  (m + 1/m) * (n + 1/n) ≥ 25/4 :=
by
  sorry

end inequality_proof_l207_207456


namespace max_value_m_l207_207677

theorem max_value_m {m : ℝ} (h : ∀ x : ℝ, -Real.pi / 4 ≤ x ∧ x ≤ Real.pi / 4 → m ≤ Real.tan x + 1) : m = 2 :=
sorry

end max_value_m_l207_207677


namespace limes_given_l207_207314

theorem limes_given (original_limes now_limes : ℕ) (h1 : original_limes = 9) (h2 : now_limes = 5) : (original_limes - now_limes = 4) := 
by
  sorry

end limes_given_l207_207314


namespace sufficient_but_not_necessary_condition_l207_207940

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = - f x
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x
def is_even_function (h : ℝ → ℝ) : Prop := ∀ x, h (-x) = h x

-- This definition states that both f and g are either odd or even functions
def is_odd_or_even (f g : ℝ → ℝ) : Prop := 
  (is_odd f ∧ is_odd g) ∨ (is_even f ∧ is_even g)

theorem sufficient_but_not_necessary_condition (f g : ℝ → ℝ)
  (h : is_odd_or_even f g) : 
  ¬(is_odd f ∧ is_odd g) → is_even_function (f * g) :=
sorry

end sufficient_but_not_necessary_condition_l207_207940


namespace train_speed_is_60_l207_207136

noncomputable def train_speed_proof : Prop :=
  let train_length := 550 -- in meters
  let time_to_pass := 29.997600191984645 -- in seconds
  let man_speed_kmhr := 6 -- in km/hr
  let man_speed_ms := man_speed_kmhr * (1000 / 3600) -- converting km/hr to m/s
  let relative_speed_ms := train_length / time_to_pass -- relative speed in m/s
  let train_speed_ms := relative_speed_ms - man_speed_ms -- speed of the train in m/s
  let train_speed_kmhr := train_speed_ms * (3600 / 1000) -- converting m/s to km/hr
  train_speed_kmhr = 60 -- the speed of the train in km/hr

theorem train_speed_is_60 : train_speed_proof := by
  sorry

end train_speed_is_60_l207_207136


namespace cos_75_degree_l207_207348

theorem cos_75_degree (cos : ℝ → ℝ) (sin : ℝ → ℝ) :
    cos 75 = (Real.sqrt 6 - Real.sqrt 2) / 4 :=
by
  sorry

end cos_75_degree_l207_207348


namespace difference_between_place_and_face_value_l207_207823

def numeral : Nat := 856973

def digit_of_interest : Nat := 7

def place_value : Nat := 7 * 10

def face_value : Nat := 7

theorem difference_between_place_and_face_value : place_value - face_value = 63 :=
by
  sorry

end difference_between_place_and_face_value_l207_207823


namespace floor_plus_x_eq_205_l207_207665

theorem floor_plus_x_eq_205 (x : ℝ) (h : ⌊x⌋ + x = 20.5) : x = 10.5 :=
sorry

end floor_plus_x_eq_205_l207_207665


namespace number_of_men_in_third_group_l207_207379

theorem number_of_men_in_third_group (m w : ℝ) (x : ℕ) :
  3 * m + 8 * w = 6 * m + 2 * w →
  x * m + 5 * w = 0.9285714285714286 * (6 * m + 2 * w) →
  x = 4 :=
by
  intros h₁ h₂
  sorry

end number_of_men_in_third_group_l207_207379
