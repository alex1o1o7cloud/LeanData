import Mathlib

namespace range_of_a_l2000_200062

theorem range_of_a (a : ℝ) 
  (h : ∀ (f : ℝ → ℝ), 
    (∀ x ≤ a, f x = -x^2 - 2*x) ∧ 
    (∀ x > a, f x = -x) ∧ 
    ¬ ∃ M, ∀ x, f x ≤ M) : 
  a < -1 :=
by
  sorry

end range_of_a_l2000_200062


namespace ratio_students_above_8_to_8_years_l2000_200025

-- Definitions of the problem's known conditions
def total_students : ℕ := 125
def students_below_8_years : ℕ := 25
def students_of_8_years : ℕ := 60

-- Main proof inquiry
theorem ratio_students_above_8_to_8_years :
  ∃ (A : ℕ), students_below_8_years + students_of_8_years + A = total_students ∧
             A * 3 = students_of_8_years * 2 := 
sorry

end ratio_students_above_8_to_8_years_l2000_200025


namespace problem1_problem2_l2000_200081

-- Proof Problem 1
theorem problem1 (x : ℝ) : -x^2 + 4 * x + 5 < 0 ↔ x < -1 ∨ x > 5 :=
by sorry

-- Proof Problem 2
theorem problem2 (x a : ℝ) :
  if a = -1 then (x^2 + (1 - a) * x - a < 0 ↔ false) else
  if a > -1 then (x^2 + (1 - a) * x - a < 0 ↔ -1 < x ∧ x < a) else
  (x^2 + (1 - a) * x - a < 0 ↔ a < x ∧ x < -1) :=
by sorry

end problem1_problem2_l2000_200081


namespace restaurant_cooks_l2000_200078

variable (C W : ℕ)

theorem restaurant_cooks : 
  (C / W = 3 / 10) ∧ (C / (W + 12) = 3 / 14) → C = 9 :=
by sorry

end restaurant_cooks_l2000_200078


namespace net_wealth_after_transactions_l2000_200043

-- Define initial values and transactions
def initial_cash_A : ℕ := 15000
def initial_cash_B : ℕ := 20000
def initial_house_value : ℕ := 15000
def first_transaction_price : ℕ := 20000
def depreciation_rate : ℝ := 0.15

-- Post-depreciation house value
def depreciated_house_value : ℝ := initial_house_value * (1 - depreciation_rate)

-- Final amounts after transactions
def final_cash_A : ℝ := (initial_cash_A + first_transaction_price) - depreciated_house_value
def final_cash_B : ℝ := depreciated_house_value

-- Net changes in wealth
def net_change_wealth_A : ℝ := final_cash_A + depreciated_house_value - (initial_cash_A + initial_house_value)
def net_change_wealth_B : ℝ := final_cash_B - initial_cash_B

-- Our proof goal
theorem net_wealth_after_transactions :
  net_change_wealth_A = 5000 ∧ net_change_wealth_B = -7250 :=
by
  sorry

end net_wealth_after_transactions_l2000_200043


namespace least_blue_eyes_and_snack_l2000_200056

variable (total_students blue_eyes students_with_snack : ℕ)

theorem least_blue_eyes_and_snack (h1 : total_students = 35) 
                                 (h2 : blue_eyes = 14) 
                                 (h3 : students_with_snack = 22) :
  ∃ n, n = 1 ∧ 
        ∀ k, (k < n → 
                 ∃ no_snack_no_blue : ℕ, no_snack_no_blue = total_students - students_with_snack ∧
                      no_snack_no_blue = blue_eyes - k) := 
by
  sorry

end least_blue_eyes_and_snack_l2000_200056


namespace efficiency_ratio_l2000_200023

theorem efficiency_ratio (A B : ℝ) (h1 : A + B = 1 / 26) (h2 : B = 1 / 39) : A / B = 1 / 2 := 
by
  sorry

end efficiency_ratio_l2000_200023


namespace inequality_proof_l2000_200098

theorem inequality_proof (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  1 ≤ ((x + y) * (x^3 + y^3)) / (x^2 + y^2)^2 ∧ 
  ((x + y) * (x^3 + y^3)) / (x^2 + y^2)^2 ≤ 9 / 8 :=
sorry

end inequality_proof_l2000_200098


namespace fraction_to_decimal_l2000_200065

theorem fraction_to_decimal : (7 : ℝ) / 250 = 0.028 := 
sorry

end fraction_to_decimal_l2000_200065


namespace find_f_at_2_l2000_200059

variable {R : Type} [Ring R]

def f (a b x : R) : R := a * x ^ 3 + b * x - 3

theorem find_f_at_2 (a b : R) (h : f a b (-2) = 7) : f a b 2 = -13 := 
by 
  have h₁ : f a b (-2) + f a b 2 = -6 := sorry
  have h₂ : f a b 2 = -6 - f a b (-2) := sorry
  rw [h₂, h]
  norm_num

end find_f_at_2_l2000_200059


namespace kolya_is_wrong_l2000_200029

def pencils_problem_statement (at_least_four_blue : Prop) 
                              (at_least_five_green : Prop) 
                              (at_least_three_blue_and_four_green : Prop) 
                              (at_least_four_blue_and_four_green : Prop) : 
                              Prop :=
  ∃ (B G : ℕ), -- B represents the number of blue pencils, G represents the number of green pencils
    ((B ≥ 4) ∧ (G ≥ 4)) ∧ -- Vasya's statement (at least 4 blue), Petya's and Misha's combined statement (at least 4 green)
    at_least_four_blue ∧ -- Vasya's statement (there are at least 4 blue pencils)
    (at_least_five_green ↔ G ≥ 5) ∧ -- Kolya's statement (there are at least 5 green pencils)
    at_least_three_blue_and_four_green ∧ -- Petya's statement (at least 3 blue and 4 green)
    at_least_four_blue_and_four_green -- Misha's statement (at least 4 blue and 4 green)

theorem kolya_is_wrong (at_least_four_blue : Prop) 
                        (at_least_five_green : Prop) 
                        (at_least_three_blue_and_four_green : Prop) 
                        (at_least_four_blue_and_four_green : Prop) : 
                        pencils_problem_statement at_least_four_blue 
                                                  at_least_five_green 
                                                  at_least_three_blue_and_four_green 
                                                  at_least_four_blue_and_four_green :=
sorry

end kolya_is_wrong_l2000_200029


namespace sector_perimeter_l2000_200094

theorem sector_perimeter (R : ℝ) (α : ℝ) (A : ℝ) (P : ℝ) : 
  A = (1 / 2) * R^2 * α → 
  α = 4 → 
  A = 2 → 
  P = 2 * R + R * α → 
  P = 6 := 
by
  intros hArea hAlpha hA hP
  sorry

end sector_perimeter_l2000_200094


namespace mean_of_counts_is_7_l2000_200091

theorem mean_of_counts_is_7 (counts : List ℕ) (h : counts = [6, 12, 1, 12, 7, 3, 8]) :
  counts.sum / counts.length = 7 :=
by
  sorry

end mean_of_counts_is_7_l2000_200091


namespace find_f11_l2000_200071

-- Define the odd function properties
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

-- Define the functional equation property
def functional_eqn (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x + 2) = -f x

-- Define the specific values of the function on (0,2)
def specific_values (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, 0 < x ∧ x < 2 → f x = 2 * x^2

-- The main theorem that needs to be proved
theorem find_f11 (f : ℝ → ℝ) (h1 : is_odd_function f) (h2 : functional_eqn f) (h3 : specific_values f) : 
  f 11 = -2 :=
sorry

end find_f11_l2000_200071


namespace second_pipe_fill_time_l2000_200051

theorem second_pipe_fill_time (x : ℝ) :
  let rate1 := 1 / 8
  let rate2 := 1 / x
  let combined_rate := 1 / 4.8
  rate1 + rate2 = combined_rate → x = 12 :=
by
  intros
  sorry

end second_pipe_fill_time_l2000_200051


namespace initial_weight_l2000_200058

theorem initial_weight (lost_weight current_weight : ℕ) (h1 : lost_weight = 35) (h2 : current_weight = 34) :
  lost_weight + current_weight = 69 :=
sorry

end initial_weight_l2000_200058


namespace matrix_pow_101_l2000_200047

noncomputable def matrixA : Matrix (Fin 3) (Fin 3) ℝ :=
  ![
    ![0, 0, 1],
    ![1, 0, 0],
    ![0, 1, 0]
  ]

theorem matrix_pow_101 :
  matrixA ^ 101 =
  ![
    ![0, 1, 0],
    ![0, 0, 1],
    ![1, 0, 0]
  ] :=
sorry

end matrix_pow_101_l2000_200047


namespace union_of_A_and_B_l2000_200077

open Set

variable (A B : Set ℤ)

theorem union_of_A_and_B (hA : A = {0, 1}) (hB : B = {0, -1}) : A ∪ B = {-1, 0, 1} := by
  sorry

end union_of_A_and_B_l2000_200077


namespace rational_zero_quadratic_roots_l2000_200061

-- Part 1
theorem rational_zero (a b : ℚ) (h : a + b * Real.sqrt 5 = 0) : a = 0 ∧ b = 0 :=
sorry

-- Part 2
theorem quadratic_roots (k : ℝ) (h : k ≠ 0) (x1 x2 : ℝ)
  (h1 : 4 * k * x1^2 - 4 * k * x1 + k + 1 = 0)
  (h2 : 4 * k * x2^2 - 4 * k * x2 + k + 1 = 0)
  (h3 : x1 ≠ x2) 
  (h4 : x1^2 + x2^2 - 2 * x1 * x2 = 0.5) : k = -2 :=
sorry

end rational_zero_quadratic_roots_l2000_200061


namespace functional_eq_1996_l2000_200090

def f (x : ℝ) : ℝ := sorry

theorem functional_eq_1996 (f : ℝ → ℝ)
    (h : ∀ x y : ℝ, f (x^3 + y^3) = (x + y) * ((f x)^2 - (f x) * (f y) + (f y)^2)) :
    ∀ x : ℝ, f (1996 * x) = 1996 * f x := 
sorry

end functional_eq_1996_l2000_200090


namespace breadth_of_added_rectangle_l2000_200049

theorem breadth_of_added_rectangle 
  (s : ℝ) (b : ℝ) 
  (h_square_side : s = 8) 
  (h_perimeter_new_rectangle : 2 * s + 2 * (s + b) = 40) : 
  b = 4 :=
by
  sorry

end breadth_of_added_rectangle_l2000_200049


namespace cuboid_cutout_l2000_200097

theorem cuboid_cutout (x y : ℕ) (h1 : x * y = 36) (h2 : 0 < x) (h3 : x < 4) (h4 : 0 < y) (h5 : y < 15) :
  x + y = 15 :=
sorry

end cuboid_cutout_l2000_200097


namespace time_to_travel_downstream_l2000_200079

-- Definitions based on the conditions.
def speed_boat_still_water := 40 -- Speed of the boat in still water (km/hr)
def speed_stream := 5 -- Speed of the stream (km/hr)
def distance_downstream := 45 -- Distance to be traveled downstream (km)

-- The proof statement
theorem time_to_travel_downstream : (distance_downstream / (speed_boat_still_water + speed_stream)) = 1 :=
by
  -- This would be the place to include the proven steps, but it's omitted as per instructions.
  sorry

end time_to_travel_downstream_l2000_200079


namespace alexander_has_more_pencils_l2000_200040

-- Definitions based on conditions
def asaf_age := 50
def total_age := 140
def total_pencils := 220

-- Auxiliary definitions based on conditions
def alexander_age := total_age - asaf_age
def age_difference := alexander_age - asaf_age
def asaf_pencils := 2 * age_difference
def alexander_pencils := total_pencils - asaf_pencils

-- Statement to prove
theorem alexander_has_more_pencils :
  (alexander_pencils - asaf_pencils) = 60 := sorry

end alexander_has_more_pencils_l2000_200040


namespace find_multiplier_l2000_200004

theorem find_multiplier (x y n : ℤ) (h1 : 3 * x + y = 40) (h2 : 2 * x - y = 20) (h3 : y^2 = 16) :
  n * y^2 = 48 :=
by 
  -- proof goes here
  sorry

end find_multiplier_l2000_200004


namespace rug_area_correct_l2000_200006

def floor_length : ℕ := 10
def floor_width : ℕ := 8
def strip_width : ℕ := 2

def adjusted_length : ℕ := floor_length - 2 * strip_width
def adjusted_width : ℕ := floor_width - 2 * strip_width

def area_floor : ℕ := floor_length * floor_width
def area_rug : ℕ := adjusted_length * adjusted_width

theorem rug_area_correct : area_rug = 24 := by
  sorry

end rug_area_correct_l2000_200006


namespace quadratic_value_at_point_l2000_200005

theorem quadratic_value_at_point :
  ∃ a b c, 
    (∃ y, y = a * 2^2 + b * 2 + c ∧ y = 7) ∧
    (∃ y, y = a * 0^2 + b * 0 + c ∧ y = -7) ∧
    (∃ y, y = a * 5^2 + b * 5 + c ∧ y = -24.5) := 
sorry

end quadratic_value_at_point_l2000_200005


namespace solve_quadratic_l2000_200032

theorem solve_quadratic (x : ℝ) (h_pos : x > 0) (h_eq : 6 * x^2 + 9 * x - 24 = 0) : x = 4 / 3 :=
by
  sorry

end solve_quadratic_l2000_200032


namespace find_minimum_x2_x1_l2000_200027

noncomputable def f (x : ℝ) : ℝ := Real.exp (2 * x)
noncomputable def g (x : ℝ) : ℝ := Real.log x + 1 / 2

theorem find_minimum_x2_x1 (x1 : ℝ) :
  ∃ x2 : {r : ℝ // 0 < r}, f x1 = g x2 → (x2 - x1) ≥ 1 + Real.log 2 / 2 :=
by
  -- Proof
  sorry

end find_minimum_x2_x1_l2000_200027


namespace average_marks_correct_l2000_200002

/-- Define the marks scored by Shekar in different subjects -/
def marks_math : ℕ := 76
def marks_science : ℕ := 65
def marks_social_studies : ℕ := 82
def marks_english : ℕ := 67
def marks_biology : ℕ := 55

/-- Define the total marks scored by Shekar -/
def total_marks : ℕ := marks_math + marks_science + marks_social_studies + marks_english + marks_biology

/-- Define the number of subjects -/
def num_subjects : ℕ := 5

/-- Define the average marks scored by Shekar -/
def average_marks : ℕ := total_marks / num_subjects

theorem average_marks_correct : average_marks = 69 := by
  -- We need to show that the average marks is 69
  sorry

end average_marks_correct_l2000_200002


namespace total_distance_covered_l2000_200013

def teams_data : List (String × Nat × Nat) :=
  [("Green Bay High", 5, 150), 
   ("Blue Ridge Middle", 7, 200),
   ("Sunset Valley Elementary", 4, 100),
   ("Riverbend Prep", 6, 250)]

theorem total_distance_covered (team : String) (members relays : Nat) :
  (team, members, relays) ∈ teams_data →
    (team = "Green Bay High" → members * relays = 750) ∧
    (team = "Blue Ridge Middle" → members * relays = 1400) ∧
    (team = "Sunset Valley Elementary" → members * relays = 400) ∧
    (team = "Riverbend Prep" → members * relays = 1500) :=
  by
    intros; sorry -- Proof omitted

end total_distance_covered_l2000_200013


namespace find_n_l2000_200038

theorem find_n (x k m n : ℤ) 
  (h1 : x = 82 * k + 5)
  (h2 : x + n = 41 * m + 18) :
  n = 5 :=
by
  sorry

end find_n_l2000_200038


namespace negation_proposition_l2000_200095

theorem negation_proposition : 
  (¬ ∃ x_0 : ℝ, 2 * x_0 - 3 > 1) ↔ (∀ x : ℝ, 2 * x - 3 ≤ 1) :=
by
  sorry

end negation_proposition_l2000_200095


namespace locus_of_center_l2000_200060

-- Define point A
def PointA : ℝ × ℝ := (-2, 0)

-- Define the tangent line
def TangentLine : ℝ := 2

-- The condition to prove the locus equation
theorem locus_of_center (x₀ y₀ : ℝ) :
  (∃ r : ℝ, abs (x₀ - TangentLine) = r ∧ (x₀ + 2)^2 + y₀^2 = r^2) →
  y₀^2 = -8 * x₀ := by
  sorry

end locus_of_center_l2000_200060


namespace minimum_value_of_y_l2000_200068

theorem minimum_value_of_y (x : ℝ) (h : x > 2) : 
  ∃ y, y = x + 4 / (x - 2) ∧ y = 6 :=
by
  sorry

end minimum_value_of_y_l2000_200068


namespace product_of_a_and_b_l2000_200044

variable (a b : ℕ)

-- Conditions
def LCM(a b : ℕ) : ℕ := Nat.lcm a b
def HCF(a b : ℕ) : ℕ := Nat.gcd a b

-- Assertion: product of a and b
theorem product_of_a_and_b (h_lcm: LCM a b = 72) (h_hcf: HCF a b = 6) : a * b = 432 := by
  sorry

end product_of_a_and_b_l2000_200044


namespace day_of_week_after_10_pow_90_days_l2000_200055

theorem day_of_week_after_10_pow_90_days :
  let initial_day := "Friday"
  ∃ day_after_10_pow_90 : String,
  day_after_10_pow_90 = "Saturday" :=
by
  sorry

end day_of_week_after_10_pow_90_days_l2000_200055


namespace machines_working_together_l2000_200096

theorem machines_working_together (x : ℝ) :
  let R_time := x + 4
  let Q_time := x + 9
  let P_time := x + 12
  (1 / P_time + 1 / Q_time + 1 / R_time) = 1 / x ↔ x = 1 := 
by
  sorry

end machines_working_together_l2000_200096


namespace sum_of_integers_l2000_200084

theorem sum_of_integers (x y : ℕ) (h1 : x > y) (h2 : x - y = 14) (h3 : x * y = 180) :
  x + y = 2 * Int.sqrt 229 :=
sorry

end sum_of_integers_l2000_200084


namespace sugar_amount_l2000_200014

theorem sugar_amount (S F B : ℕ) (h1 : S = 5 * F / 4) (h2 : F = 10 * B) (h3 : F = 8 * (B + 60)) : S = 3000 := by
  sorry

end sugar_amount_l2000_200014


namespace part1_part2_part3_l2000_200007

-- Part 1
theorem part1 : (1 > -1) ∧ (1 < 2) ∧ (-(1/2) > -1) ∧ (-(1/2) < 2) := 
  by sorry

-- Part 2
theorem part2 (k : Real) : (3 < k) ∧ (k ≤ 4) := 
  by sorry

-- Part 3
theorem part3 (m : Real) : (2 < m) ∧ (m ≤ 3) := 
  by sorry

end part1_part2_part3_l2000_200007


namespace no_n_ge_1_such_that_sum_is_perfect_square_l2000_200048

theorem no_n_ge_1_such_that_sum_is_perfect_square :
  ¬ ∃ n : ℕ, n ≥ 1 ∧ ∃ k : ℕ, 2^n + 12^n + 2014^n = k^2 :=
by
  sorry

end no_n_ge_1_such_that_sum_is_perfect_square_l2000_200048


namespace find_m_l2000_200045

def line_eq (x y : ℝ) : Prop := x + 2 * y - 3 = 0

def circle_eq (x y m : ℝ) : Prop := x * x + y * y + x - 6 * y + m = 0

def perpendicular_vectors (x1 y1 x2 y2 : ℝ) : Prop := x1 * x2 + y1 * y2 = 0

theorem find_m (m : ℝ) :
  (∃ (x y : ℝ), line_eq x y ∧ line_eq (3 - 2 * y) y ∧ circle_eq x y m ∧ circle_eq (3 - 2 * y) y m) ∧
  (∃ (x1 y1 x2 y2 : ℝ), line_eq x1 y1 ∧ line_eq x2 y2 ∧ perpendicular_vectors x1 y1 x2 y2) → m = 3 :=
sorry

end find_m_l2000_200045


namespace lcm_4_6_9_l2000_200054

/-- The least common multiple (LCM) of 4, 6, and 9 is 36 -/
theorem lcm_4_6_9 : Nat.lcm (Nat.lcm 4 6) 9 = 36 :=
by
  -- sorry replaces the actual proof steps
  sorry

end lcm_4_6_9_l2000_200054


namespace star_sum_interior_angles_l2000_200057

theorem star_sum_interior_angles (n : ℕ) (h : n ≥ 6) :
  let S := 180 * n - 360
  S = 180 * (n - 2) :=
by
  let S := 180 * n - 360
  show S = 180 * (n - 2)
  sorry

end star_sum_interior_angles_l2000_200057


namespace dave_spent_102_dollars_l2000_200099

noncomputable def total_cost (books_animals books_space books_trains cost_per_book : ℕ) : ℕ :=
  (books_animals + books_space + books_trains) * cost_per_book

theorem dave_spent_102_dollars :
  total_cost 8 6 3 6 = 102 := by
  sorry

end dave_spent_102_dollars_l2000_200099


namespace find_m_values_l2000_200015

def has_unique_solution (m : ℝ) (A : Set ℝ) : Prop :=
  ∀ x1 x2, x1 ∈ A → x2 ∈ A → x1 = x2

theorem find_m_values :
  {m : ℝ | ∃ A : Set ℝ, has_unique_solution m A ∧ (A = {x | m * x^2 + 2 * x + 3 = 0})} = {0, 1/3} :=
by
  sorry

end find_m_values_l2000_200015


namespace roots_greater_than_one_implies_s_greater_than_zero_l2000_200076

theorem roots_greater_than_one_implies_s_greater_than_zero
  (b c : ℝ)
  (h : ∃ α β : ℝ, α > 0 ∧ β > 0 ∧ (1 + α) + (1 + β) = -b ∧ (1 + α) * (1 + β) = c) :
  b + c + 1 > 0 :=
sorry

end roots_greater_than_one_implies_s_greater_than_zero_l2000_200076


namespace new_person_weight_l2000_200011

theorem new_person_weight
  (avg_increase : ℝ) (original_person_weight : ℝ) (num_people : ℝ) (new_weight : ℝ)
  (h1 : avg_increase = 2.5)
  (h2 : original_person_weight = 85)
  (h3 : num_people = 8)
  (h4 : num_people * avg_increase = new_weight - original_person_weight):
    new_weight = 105 :=
by
  sorry

end new_person_weight_l2000_200011


namespace find_k_l2000_200089

-- Define the equation of line m
def line_m (x : ℝ) : ℝ := 2 * x + 8

-- Define the equation of line n with an unknown slope k
def line_n (k : ℝ) (x : ℝ) : ℝ := k * x - 9

-- Define the point of intersection
def intersection_point := (-4, 0)

-- The proof statement
theorem find_k : ∃ k : ℝ, k = -9 / 4 ∧ line_m (-4) = 0 ∧ line_n k (-4) = 0 :=
by
  exists (-9 / 4)
  simp [line_m, line_n, intersection_point]
  sorry

end find_k_l2000_200089


namespace freight_train_distance_l2000_200070

variable (travel_rate : ℕ) (initial_distance : ℕ) (time_minutes : ℕ) 

def total_distance_traveled (travel_rate : ℕ) (initial_distance : ℕ) (time_minutes : ℕ) : ℕ :=
  let traveled_distance := (time_minutes / travel_rate) 
  traveled_distance + initial_distance

theorem freight_train_distance :
  total_distance_traveled 2 5 90 = 50 :=
by
  sorry

end freight_train_distance_l2000_200070


namespace sequence_third_term_l2000_200018

theorem sequence_third_term (a m : ℤ) (h_a_neg : a < 0) (h_a1 : a + m = 2) (h_a2 : a^2 + m = 4) :
  (a^3 + m = 2) :=
by
  sorry

end sequence_third_term_l2000_200018


namespace elsie_money_l2000_200019

noncomputable def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r) ^ n

theorem elsie_money : 
  compound_interest 2500 0.04 20 = 5477.81 :=
by 
  sorry

end elsie_money_l2000_200019


namespace no_solutions_l2000_200010

theorem no_solutions {x y : ℤ} :
  (x ≠ 1) → (y ≠ 1) →
  ((x^7 - 1) / (x - 1) = y^5 - 1) →
  false :=
by sorry

end no_solutions_l2000_200010


namespace find_M_l2000_200064

theorem find_M (a b c M : ℚ) 
  (h1 : a + b + c = 100)
  (h2 : a - 10 = M)
  (h3 : b + 10 = M)
  (h4 : 10 * c = M) : 
  M = 1000 / 21 :=
sorry

end find_M_l2000_200064


namespace find_matrix_A_l2000_200037

theorem find_matrix_A (a b c d : ℝ) 
  (h1 : a - 3 * b = -1)
  (h2 : c - 3 * d = 3)
  (h3 : a + b = 3)
  (h4 : c + d = 3) :
  a = 2 ∧ b = 1 ∧ c = 3 ∧ d = 0 := by
  sorry

end find_matrix_A_l2000_200037


namespace repeating_decimal_to_fraction_l2000_200039

theorem repeating_decimal_to_fraction : ∀ (x : ℝ), x = 0.7 + 0.08 / (1-0.1) → x = 71 / 90 :=
by
  intros x hx
  sorry

end repeating_decimal_to_fraction_l2000_200039


namespace time_until_meeting_l2000_200074

theorem time_until_meeting (v1 v2 : ℝ) (t2 t1 : ℝ) 
    (h1 : v1 = 6) 
    (h2 : v2 = 4) 
    (h3 : t2 = 10)
    (h4 : v2 * t1 = v1 * (t1 - t2)) : t1 = 30 := 
sorry

end time_until_meeting_l2000_200074


namespace part1_part2_l2000_200021

-- Definition of the function
def f (x : ℝ) (m : ℝ) : ℝ := x^2 + m * x - 1

-- Theorem for part (1)
theorem part1 
  (m n : ℝ)
  (h1 : ∀ x : ℝ, f x m < 0 ↔ -2 < x ∧ x < n) : 
  m = 3 / 2 ∧ n = 1 / 2 :=
sorry

-- Theorem for part (2)
theorem part2 
  (m : ℝ)
  (h2 : ∀ x : ℝ, m ≤ x ∧ x ≤ m + 1 → f x m < 0) : 
  -Real.sqrt 2 / 2 < m ∧ m < 0 :=
sorry

end part1_part2_l2000_200021


namespace medium_stores_to_select_l2000_200017

-- Definitions based on conditions in a)
def total_stores := 1500
def ratio_large := 1
def ratio_medium := 5
def ratio_small := 9
def sample_size := 30
def medium_proportion := ratio_medium / (ratio_large + ratio_medium + ratio_small)

-- Main theorem to prove
theorem medium_stores_to_select : (sample_size * medium_proportion) = 10 :=
by sorry

end medium_stores_to_select_l2000_200017


namespace seed_mixture_ryegrass_l2000_200063

theorem seed_mixture_ryegrass (α : ℝ) :
  (0.4667 * 0.4 + 0.5333 * α = 0.32) -> α = 0.25 :=
by
  sorry

end seed_mixture_ryegrass_l2000_200063


namespace shirt_original_price_l2000_200053

theorem shirt_original_price {P : ℝ} :
  (P * 0.80045740423098913 * 0.8745 = 105) → P = 150 :=
by sorry

end shirt_original_price_l2000_200053


namespace tunnel_length_l2000_200009

-- Definitions as per the conditions
def train_length : ℚ := 2  -- 2 miles
def train_speed : ℚ := 40  -- 40 miles per hour

def speed_in_miles_per_minute (speed_mph : ℚ) : ℚ :=
  speed_mph / 60  -- Convert speed from miles per hour to miles per minute

def time_travelled_in_minutes : ℚ := 5  -- 5 minutes

-- Theorem statement to prove the length of the tunnel
theorem tunnel_length (h1 : train_length = 2) (h2 : train_speed = 40) :
  (speed_in_miles_per_minute train_speed * time_travelled_in_minutes) - train_length = 4 / 3 :=
by
  sorry  -- Proof not included

end tunnel_length_l2000_200009


namespace impossible_piles_of_three_l2000_200001

theorem impossible_piles_of_three (n : ℕ) (h1 : n = 1001)
  (h2 : ∀ p : ℕ, p > 1 → ∃ a b : ℕ, a + b = p - 1 ∧ a ≤ b) : 
  ¬ (∃ piles : List ℕ, ∀ pile ∈ piles, pile = 3 ∧ (piles.sum = n + piles.length)) :=
by
  sorry

end impossible_piles_of_three_l2000_200001


namespace divisor_probability_of_25_factorial_is_odd_and_multiple_of_5_l2000_200088

theorem divisor_probability_of_25_factorial_is_odd_and_multiple_of_5 :
  let prime_factors_25 := 2^22 * 3^10 * 5^6 * 7^3 * 11^2 * 13^1 * 17^1 * 19^1 * 23^1
  let total_divisors := (22+1) * (10+1) * (6+1) * (3+1) * (2+1) * (1+1) * (1+1) * (1+1)
  let odd_and_multiple_of_5_divisors := (6+1) * (3+1) * (2+1) * (1+1) * (1+1)
  (odd_and_multiple_of_5_divisors / total_divisors : ℚ) = 7 / 23 := 
sorry

end divisor_probability_of_25_factorial_is_odd_and_multiple_of_5_l2000_200088


namespace parallelogram_and_triangle_area_eq_l2000_200085

noncomputable def parallelogram_area (AB AD : ℝ) : ℝ :=
  AB * AD

noncomputable def right_triangle_area (DG FG : ℝ) : ℝ :=
  (DG * FG) / 2

variables (AB AD DG FG : ℝ)
variables (angleDFG : ℝ)

def parallelogram_ABCD (AB : ℝ) (AD : ℝ) (angleDFG : ℝ) (DG : ℝ) : Prop :=
  parallelogram_area AB AD = 24 ∧ angleDFG = 90 ∧ DG = 6

theorem parallelogram_and_triangle_area_eq (h1 : parallelogram_ABCD AB AD angleDFG DG)
    (h2 : parallelogram_area AB AD = right_triangle_area DG FG) : FG = 8 :=
by
  sorry

end parallelogram_and_triangle_area_eq_l2000_200085


namespace sum_of_m_and_n_l2000_200093

theorem sum_of_m_and_n (m n : ℝ) (h : m^2 + n^2 - 6 * m + 10 * n + 34 = 0) : m + n = -2 := 
sorry

end sum_of_m_and_n_l2000_200093


namespace inequality_always_holds_l2000_200003

noncomputable def range_for_inequality (k : ℝ) : Prop :=
  0 < k ∧ k ≤ 2 * Real.sqrt (2 + Real.sqrt 5)

theorem inequality_always_holds (x y k : ℝ) (h₁ : 0 < x) (h₂ : 0 < y) (h₃ : x + y = k) :
  (x + 1/x) * (y + 1/y) ≥ (k/2 + 2/k)^2 ↔ range_for_inequality k :=
sorry

end inequality_always_holds_l2000_200003


namespace art_club_artworks_l2000_200031

theorem art_club_artworks (students : ℕ) (artworks_per_student_per_quarter : ℕ)
  (quarters_per_year : ℕ) (years : ℕ) :
  students = 15 → artworks_per_student_per_quarter = 2 → 
  quarters_per_year = 4 → years = 2 → 
  (students * artworks_per_student_per_quarter * quarters_per_year * years) = 240 :=
by
  intros
  sorry

end art_club_artworks_l2000_200031


namespace circle_a_center_radius_circle_b_center_radius_circle_c_center_radius_l2000_200072

-- Part (a): Prove the center and radius for the given circle equation: (x-3)^2 + (y+2)^2 = 16
theorem circle_a_center_radius :
  (∃ (a b : ℤ) (R : ℕ), (∀ (x y : ℝ), (x - 3) ^ 2 + (y + 2) ^ 2 = 16 ↔ (x - a) ^ 2 + (y - b) ^ 2 = R^2) ∧ a = 3 ∧ b = -2 ∧ R = 4) :=
by {
  sorry
}

-- Part (b): Prove the center and radius for the given circle equation: x^2 + y^2 - 2(x - 3y) - 15 = 0
theorem circle_b_center_radius :
  (∃ (a b : ℤ) (R : ℕ), (∀ (x y : ℝ), x^2 + y^2 - 2 * (x - 3 * y) - 15 = 0 ↔ (x - a) ^ 2 + (y - b) ^ 2 = R^2) ∧ a = 1 ∧ b = -3 ∧ R = 5) :=
by {
  sorry
}

-- Part (c): Prove the center and radius for the given circle equation: x^2 + y^2 = x + y + 1/2
theorem circle_c_center_radius :
  (∃ (a b : ℚ) (R : ℚ), (∀ (x y : ℚ), x^2 + y^2 = x + y + 1/2 ↔ (x - a) ^ 2 + (y - b) ^ 2 = R^2) ∧ a = 1/2 ∧ b = 1/2 ∧ R = 1) :=
by {
  sorry
}

end circle_a_center_radius_circle_b_center_radius_circle_c_center_radius_l2000_200072


namespace intersection_equiv_l2000_200022

-- Define the sets M and N based on the given conditions
def M : Set ℤ := {-1, 0, 1, 2, 3}
def N : Set ℤ := {x | x < 0 ∨ x > 2}

-- The main proof statement
theorem intersection_equiv : M ∩ N = {-1, 3} :=
by
  sorry -- proof goes here

end intersection_equiv_l2000_200022


namespace first_chapter_pages_calculation_l2000_200086

-- Define the constants and conditions
def second_chapter_pages : ℕ := 11
def first_chapter_pages_more : ℕ := 37

-- Main proof problem
theorem first_chapter_pages_calculation : first_chapter_pages_more + second_chapter_pages = 48 := by
  sorry

end first_chapter_pages_calculation_l2000_200086


namespace algebraic_expression_value_l2000_200028

theorem algebraic_expression_value
  (x : ℝ)
  (h : 2 * x^2 + 3 * x + 1 = 10) :
  4 * x^2 + 6 * x + 1 = 19 := 
by
  sorry

end algebraic_expression_value_l2000_200028


namespace find_abc_sum_l2000_200016

theorem find_abc_sum :
  ∀ (a b c : ℝ),
    2 * |a + 3| + 4 - b = 0 →
    c^2 + 4 * b - 4 * c - 12 = 0 →
    a + b + c = 5 :=
by
  intros a b c h1 h2
  sorry

end find_abc_sum_l2000_200016


namespace mary_younger_than_albert_l2000_200020

variable (A M B : ℕ)

noncomputable def albert_age := 4 * B
noncomputable def mary_age := A / 2
noncomputable def betty_age := 4

theorem mary_younger_than_albert (h1 : A = 2 * M) (h2 : A = 4 * 4) (h3 : 4 = 4) :
  A - M = 8 :=
sorry

end mary_younger_than_albert_l2000_200020


namespace find_m_l2000_200000

theorem find_m (m : ℚ) : 
  (∃ m, (∀ x y z : ℚ, ((x, y) = (2, 9) ∨ (x, y) = (15, m) ∨ (x, y) = (35, 4)) ∧ 
  (∀ a b c d e f : ℚ, ((a, b) = (2, 9) ∨ (a, b) = (15, m) ∨ (a, b) = (35, 4)) → 
  ((b - d) / (a - c) = (f - d) / (e - c))) → m = 232 / 33)) :=
sorry

end find_m_l2000_200000


namespace total_distance_l2000_200030

theorem total_distance (D : ℕ) 
  (h1 : (1 / 2 * D : ℝ) + (1 / 4 * (1 / 2 * D : ℝ)) + 105 = D) : 
  D = 280 :=
by
  sorry

end total_distance_l2000_200030


namespace convert_speed_l2000_200050

theorem convert_speed (v_kmph : ℝ) (conversion_factor : ℝ) : 
  v_kmph = 252 → conversion_factor = 0.277778 → v_kmph * conversion_factor = 70 := by
  intros h1 h2
  rw [h1, h2]
  sorry

end convert_speed_l2000_200050


namespace relationship_among_x_y_z_l2000_200008

variable (a b c d : ℝ)

-- Conditions
variables (h1 : a < b)
variables (h2 : b < c)
variables (h3 : c < d)

-- Definitions of x, y, z
def x : ℝ := (a + b) * (c + d)
def y : ℝ := (a + c) * (b + d)
def z : ℝ := (a + d) * (b + c)

-- Theorem: Prove the relationship among x, y, z
theorem relationship_among_x_y_z (h1 : a < b) (h2 : b < c) (h3 : c < d) : x a b c d < y a b c d ∧ y a b c d < z a b c d := by
  sorry

end relationship_among_x_y_z_l2000_200008


namespace problem_1_problem_2_l2000_200087

theorem problem_1 (P_A P_B P_notA P_notB : ℚ) (hA: P_A = 1/2) (hB: P_B = 2/5) (hNotA: P_notA = 1/2) (hNotB: P_notB = 3/5) : 
  P_A * P_notB + P_B * P_notA = 1/2 := 
by 
  rw [hA, hB, hNotA, hNotB]
  -- exact calculations here
  sorry

theorem problem_2 (P_A P_B : ℚ) (hA: P_A = 1/2) (hB: P_B = 2/5) :
  (1 - (P_A * P_A * (1 - P_B) * (1 - P_B))) = 91/100 := 
by 
  rw [hA, hB]
  -- exact calculations here
  sorry

end problem_1_problem_2_l2000_200087


namespace number_of_divisors_180_l2000_200046

theorem number_of_divisors_180 : (∃ (n : ℕ), n = 180 ∧ (∀ (e1 e2 e3 : ℕ), 180 = 2^e1 * 3^e2 * 5^e3 → (e1 + 1) * (e2 + 1) * (e3 + 1) = 18)) :=
  sorry

end number_of_divisors_180_l2000_200046


namespace smallest_x_l2000_200035

theorem smallest_x (x : ℝ) (h : |4 * x + 12| = 40) : x = -13 :=
sorry

end smallest_x_l2000_200035


namespace number_of_nickels_is_three_l2000_200052

def coin_problem : Prop :=
  ∃ p n d q : ℕ,
    p + n + d + q = 12 ∧
    p + 5 * n + 10 * d + 25 * q = 128 ∧
    p ≥ 1 ∧ n ≥ 1 ∧ d ≥ 1 ∧ q ≥ 1 ∧
    q = 2 * d ∧
    n = 3

theorem number_of_nickels_is_three : coin_problem := 
by 
  sorry

end number_of_nickels_is_three_l2000_200052


namespace product_less_by_nine_times_l2000_200024

theorem product_less_by_nine_times (x y : ℕ) (hx : 10 ≤ x ∧ x < 100) (hy : y < 10) : 
  (x * y) * 10 - x * y = 9 * (x * y) := 
by
  sorry

end product_less_by_nine_times_l2000_200024


namespace question1_l2000_200069

def sequence1 (a : ℕ → ℕ) : Prop :=
   a 1 = 1 ∧ ∀ n, n ≥ 2 → a n = 3 * a (n - 1) + 1

noncomputable def a_n1 (n : ℕ) : ℕ := (3^n - 1) / 2

theorem question1 (a : ℕ → ℕ) (n : ℕ) : sequence1 a → a n = a_n1 n :=
by
  sorry

end question1_l2000_200069


namespace product_of_intersection_points_l2000_200067

-- Define the two circles in the plane
def circle1 (x y : ℝ) : Prop := x^2 - 4*x + y^2 - 8*y + 16 = 0
def circle2 (x y : ℝ) : Prop := x^2 - 6*x + y^2 - 8*y + 21 = 0

-- Define the intersection points property
def are_intersection_points (x y : ℝ) : Prop := circle1 x y ∧ circle2 x y

-- The theorem to be proved
theorem product_of_intersection_points : ∃ x y : ℝ, are_intersection_points x y ∧ x * y = 12 := 
by
  sorry

end product_of_intersection_points_l2000_200067


namespace isosceles_triangles_l2000_200092

theorem isosceles_triangles (a b c : ℝ) (hpos_a : 0 < a) (hpos_b : 0 < b) (hpos_c : 0 < c) (h_triangle : ∀ n : ℕ, (a^n + b^n > c^n ∧ b^n + c^n > a^n ∧ c^n + a^n > b^n)) :
  b = c := 
sorry

end isosceles_triangles_l2000_200092


namespace num_floors_each_building_l2000_200075

theorem num_floors_each_building
  (floors_each_building num_apartments_per_floor num_doors_per_apartment total_doors : ℕ)
  (h1 : floors_each_building = F)
  (h2 : num_apartments_per_floor = 6)
  (h3 : num_doors_per_apartment = 7)
  (h4 : total_doors = 1008)
  (eq1 : 2 * floors_each_building * num_apartments_per_floor * num_doors_per_apartment = total_doors) :
  F = 12 :=
sorry

end num_floors_each_building_l2000_200075


namespace circle_area_l2000_200080

theorem circle_area (C : ℝ) (hC : C = 24) : ∃ (A : ℝ), A = 144 / π :=
by
  sorry

end circle_area_l2000_200080


namespace smallest_y_value_in_set_l2000_200082

theorem smallest_y_value_in_set : ∀ y : ℕ, (0 < y) ∧ (y + 4 ≤ 8) → y = 4 :=
by
  intros y h
  have h1 : y + 4 ≤ 8 := h.2
  have h2 : 0 < y := h.1
  sorry

end smallest_y_value_in_set_l2000_200082


namespace yeast_cells_at_2_20_pm_l2000_200073

noncomputable def yeast_population (initial : Nat) (rate : Nat) (intervals : Nat) : Nat :=
  initial * rate ^ intervals

theorem yeast_cells_at_2_20_pm :
  let initial_population := 30
  let triple_rate := 3
  let intervals := 5 -- 20 minutes / 4 minutes per interval
  yeast_population initial_population triple_rate intervals = 7290 :=
by
  let initial_population := 30
  let triple_rate := 3
  let intervals := 5
  show yeast_population initial_population triple_rate intervals = 7290
  sorry

end yeast_cells_at_2_20_pm_l2000_200073


namespace subtraction_from_double_result_l2000_200026

theorem subtraction_from_double_result (x : ℕ) (h : 2 * x = 18) : x - 4 = 5 :=
sorry

end subtraction_from_double_result_l2000_200026


namespace coefficients_sum_binomial_coefficients_sum_l2000_200012

theorem coefficients_sum (x : ℝ) (h : (x + 2 / x)^6 = coeff_sum) : coeff_sum = 729 := 
sorry

theorem binomial_coefficients_sum (x : ℝ) (h : (x + 2 / x)^6 = binom_coeff_sum) : binom_coeff_sum = 64 := 
sorry

end coefficients_sum_binomial_coefficients_sum_l2000_200012


namespace multiplication_digit_sum_l2000_200083

theorem multiplication_digit_sum :
  let a := 879
  let b := 492
  let product := a * b
  let sum_of_digits := (4 + 3 + 2 + 4 + 6 + 8)
  product = 432468 ∧ sum_of_digits = 27 := by
  -- Step 1: Set up the given numbers
  let a := 879
  let b := 492

  -- Step 2: Calculate the product
  let product := a * b
  have product_eq : product = 432468 := by
    sorry

  -- Step 3: Sum the digits of the product
  let sum_of_digits := (4 + 3 + 2 + 4 + 6 + 8)
  have sum_of_digits_eq : sum_of_digits = 27 := by
    sorry

  -- Conclusion
  exact ⟨product_eq, sum_of_digits_eq⟩

end multiplication_digit_sum_l2000_200083


namespace playground_girls_count_l2000_200036

theorem playground_girls_count (boys : ℕ) (total_children : ℕ) 
  (h_boys : boys = 35) (h_total : total_children = 63) : 
  ∃ girls : ℕ, girls = 28 ∧ girls = total_children - boys := 
by 
  sorry

end playground_girls_count_l2000_200036


namespace positive_number_satisfying_condition_l2000_200066

theorem positive_number_satisfying_condition :
  ∃ x : ℝ, x > 0 ∧ x^2 = 64 ∧ x = 8 := by sorry

end positive_number_satisfying_condition_l2000_200066


namespace number_of_sandwiches_l2000_200041

-- Defining the conditions
def kinds_of_meat := 12
def kinds_of_cheese := 11
def kinds_of_bread := 5

-- Combinations calculation
def choose_one (n : Nat) := n
def choose_three (n : Nat) := Nat.choose n 3

-- Proof statement to show that the total number of sandwiches is 9900
theorem number_of_sandwiches : (choose_one kinds_of_meat) * (choose_three kinds_of_cheese) * (choose_one kinds_of_bread) = 9900 := by
  sorry

end number_of_sandwiches_l2000_200041


namespace C_paisa_for_A_rupee_l2000_200033

variable (A B C : ℝ)
variable (C_share : ℝ) (total_sum : ℝ)
variable (B_per_A : ℝ)

noncomputable def C_paisa_per_A_rupee (A B C C_share total_sum B_per_A : ℝ) : ℝ :=
  let C_paisa := C_share * 100
  C_paisa / A

theorem C_paisa_for_A_rupee : C_share = 32 ∧ total_sum = 164 ∧ B_per_A = 0.65 → 
  C_paisa_per_A_rupee A B C C_share total_sum B_per_A = 40 := by
  sorry

end C_paisa_for_A_rupee_l2000_200033


namespace arithmetic_sequence_geometric_mean_l2000_200034

theorem arithmetic_sequence_geometric_mean (d : ℝ) (k : ℕ) (a : ℕ → ℝ)
  (h1 : d ≠ 0)
  (h2 : a 1 = 9 * d)
  (h3 : a (k + 1) = a 1 + k * d)
  (h4 : a (2 * k + 1) = a 1 + (2 * k) * d)
  (h_gm : (a k) ^ 2 = a 1 * a (2 * k)) :
  k = 4 :=
sorry

end arithmetic_sequence_geometric_mean_l2000_200034


namespace next_birthday_monday_l2000_200042
open Nat

-- Define the basic structure and parameters of our problem
def is_leap_year (year : ℕ) : Prop := 
  (year % 4 = 0 ∧ year % 100 ≠ 0) ∨ (year % 400 = 0)

def day_of_week (start_day : ℕ) (year_diff : ℕ) (is_leap : ℕ → Prop) : ℕ :=
  (start_day + year_diff + (year_diff / 4) - (year_diff / 100) + (year_diff / 400)) % 7

-- Specify problem conditions
def initial_year := 2009
def initial_day := 5 -- 2009-06-18 is Friday, which is 5 if we start counting from Sunday as 0
def end_day := 1 -- target day is Monday, which is 1

-- Main theorem
theorem next_birthday_monday : ∃ year, year > initial_year ∧
  day_of_week initial_day (year - initial_year) is_leap_year = end_day := by
  use 2017
  -- The proof would go here, skipping with sorry
  sorry

end next_birthday_monday_l2000_200042
