import Mathlib

namespace floor_neg_seven_fourths_l2236_223626

theorem floor_neg_seven_fourths : (Int.floor (-7 / 4 : ℚ) = -2) := 
by
  sorry

end floor_neg_seven_fourths_l2236_223626


namespace ratio_shorter_to_longer_l2236_223639

theorem ratio_shorter_to_longer (total_length shorter_length longer_length : ℕ) (h1 : total_length = 40) 
(h2 : shorter_length = 16) (h3 : longer_length = total_length - shorter_length) : 
(shorter_length / Nat.gcd shorter_length longer_length) / (longer_length / Nat.gcd shorter_length longer_length) = 2 / 3 :=
by
  sorry

end ratio_shorter_to_longer_l2236_223639


namespace trigonometric_identity_l2236_223612

open Real

theorem trigonometric_identity (θ : ℝ) (h₁ : 0 < θ ∧ θ < π/2) (h₂ : cos θ = sqrt 10 / 10) :
  (cos (2 * θ) / (sin (2 * θ) + (cos θ)^2)) = -8 / 7 := 
sorry

end trigonometric_identity_l2236_223612


namespace simplify_fraction_l2236_223680

def a : ℕ := 2016
def b : ℕ := 2017

theorem simplify_fraction :
  (a^4 - 2 * a^3 * b + 3 * a^2 * b^2 - a * b^3 + 1) / (a^2 * b^2) = 1 - 1 / b^2 :=
by
  sorry

end simplify_fraction_l2236_223680


namespace evaluate_expression_l2236_223616

variable (m n p q s : ℝ)

theorem evaluate_expression :
  m / (n - (p + q * s)) = m / (n - p - q * s) :=
by
  sorry

end evaluate_expression_l2236_223616


namespace calc_expression_l2236_223615

theorem calc_expression : 
  |1 - Real.sqrt 2| - Real.sqrt 8 + (Real.sqrt 2 - 1)^0 = -Real.sqrt 2 :=
by
  sorry

end calc_expression_l2236_223615


namespace rectangle_area_l2236_223651

theorem rectangle_area
    (w l : ℕ)
    (h₁ : 28 = 2 * (l + w))
    (h₂ : w = 6) : l * w = 48 :=
by
  sorry

end rectangle_area_l2236_223651


namespace certain_number_equals_l2236_223621

theorem certain_number_equals (p q : ℚ) (h1 : 3 / p = 8) (h2 : 3 / q = 18) (h3 : p - q = 0.20833333333333334) : q = 1/6 := sorry

end certain_number_equals_l2236_223621


namespace evaluate_difference_of_squares_l2236_223662

theorem evaluate_difference_of_squares : 81^2 - 49^2 = 4160 := by
  sorry

end evaluate_difference_of_squares_l2236_223662


namespace lines_are_coplanar_l2236_223675

-- Define the first line
def line1 (t : ℝ) (m : ℝ) : ℝ × ℝ × ℝ :=
  (3 + 2 * t, 2 - m * t, 6 + t)

-- Define the second line
def line2 (u : ℝ) (m : ℝ) : ℝ × ℝ × ℝ :=
  (4 + m * u, 5 + 3 * u, 8 + 2 * u)

-- Define the vector connecting points on the lines when t=0 and u=0
def connecting_vector : ℝ × ℝ × ℝ :=
  (1, 3, 2)

-- Define the cross product of the direction vectors
def cross_product (m : ℝ) : ℝ × ℝ × ℝ :=
  ((-2 * m - 3), (m + 2), (6 + 2 * m))

-- Prove that lines are coplanar when m = -9/4
theorem lines_are_coplanar : ∃ k : ℝ, ∀ m : ℝ,
  cross_product m = (k * 1, k * 3, k * 2) → m = -9/4 :=
by
  sorry

end lines_are_coplanar_l2236_223675


namespace williams_tips_august_l2236_223603

variable (A : ℝ) (total_tips : ℝ)
variable (tips_August : ℝ) (average_monthly_tips_other_months : ℝ)

theorem williams_tips_august (h1 : tips_August = 0.5714285714285714 * total_tips)
                               (h2 : total_tips = 7 * average_monthly_tips_other_months) 
                               (h3 : total_tips = tips_August + 6 * average_monthly_tips_other_months) :
                               tips_August = 8 * average_monthly_tips_other_months :=
by
  sorry

end williams_tips_august_l2236_223603


namespace razorback_tshirt_sales_l2236_223627

theorem razorback_tshirt_sales 
  (price_per_tshirt : ℕ) (total_money_made : ℕ)
  (h1 : price_per_tshirt = 16) (h2 : total_money_made = 720) :
  total_money_made / price_per_tshirt = 45 :=
by
  sorry

end razorback_tshirt_sales_l2236_223627


namespace general_solution_of_diff_eq_l2236_223619

theorem general_solution_of_diff_eq {C1 C2 : ℝ} (y : ℝ → ℝ) (x : ℝ) :
  (∀ x, y x = C1 * Real.exp (-x) + C2 * Real.exp (-2 * x) + x^2 - 5 * x - 2) →
  (∀ x, (deriv (deriv y)) x + 3 * (deriv y) x + 2 * y x = 2 * x^2 - 4 * x - 17) :=
by
  intro hy
  sorry

end general_solution_of_diff_eq_l2236_223619


namespace solve_for_x_l2236_223693

theorem solve_for_x : ∃ x : ℤ, 24 - 5 = 3 + x ∧ x = 16 :=
by
  sorry

end solve_for_x_l2236_223693


namespace min_value_l2236_223696

theorem min_value (a b : ℝ) (ha_pos : 0 < a) (hb_pos : 0 < b) (h_sum : a + b = 1) : 
  ∃ x : ℝ, (x = 25) ∧ x ≤ (4 / a + 9 / b) :=
by
  sorry

end min_value_l2236_223696


namespace thirty_thousand_times_thirty_thousand_l2236_223600

-- Define the number thirty thousand
def thirty_thousand : ℕ := 30000

-- Define the product of thirty thousand times thirty thousand
def product_thirty_thousand : ℕ := thirty_thousand * thirty_thousand

-- State the theorem that this product equals nine hundred million
theorem thirty_thousand_times_thirty_thousand :
  product_thirty_thousand = 900000000 :=
sorry -- Proof goes here

end thirty_thousand_times_thirty_thousand_l2236_223600


namespace antonio_weight_l2236_223655

-- Let A be the weight of Antonio
variable (A : ℕ)

-- Conditions:
-- 1. Antonio's sister weighs A - 12 kilograms.
-- 2. The total weight of Antonio and his sister is 88 kilograms.

theorem antonio_weight (A: ℕ) (h1: A - 12 >= 0) (h2: A + (A - 12) = 88) : A = 50 := by
  sorry

end antonio_weight_l2236_223655


namespace area_triangle_ABC_given_conditions_l2236_223632

variable (a b c : ℝ) (A B C : ℝ)

noncomputable def area_of_triangle_ABC (a b c : ℝ) (A B C : ℝ) : ℝ :=
  1/2 * b * c * Real.sin A

theorem area_triangle_ABC_given_conditions
  (habc : a = 4)
  (hbc : b + c = 5)
  (htan : Real.tan B + Real.tan C + Real.sqrt 3 = Real.sqrt 3 * (Real.tan B * Real.tan C))
  : area_of_triangle_ABC a b c (Real.pi / 3) B C = 3 * Real.sqrt 3 / 4 := 
sorry

end area_triangle_ABC_given_conditions_l2236_223632


namespace largest_x_eq_neg5_l2236_223687

theorem largest_x_eq_neg5 (x : ℝ) (h : x ≠ 7) : (x^2 - 5*x - 84)/(x - 7) = 2/(x + 6) → x ≤ -5 := 
sorry

end largest_x_eq_neg5_l2236_223687


namespace max_integer_k_l2236_223640

noncomputable def f (x : ℝ) : ℝ := (1 + Real.log (x - 1)) / (x - 2)

theorem max_integer_k (x : ℝ) (k : ℕ) (hx : x > 2) :
  (∀ x, x > 2 → f x > (k : ℝ) / (x - 1)) ↔ k ≤ 3 :=
sorry

end max_integer_k_l2236_223640


namespace simplified_equation_has_solution_l2236_223674

theorem simplified_equation_has_solution (n : ℤ) :
  (∃ x y z : ℤ, x^2 + y^2 + z^2 - x * y - y * z - z * x = n) →
  (∃ x y : ℤ, x^2 + y^2 - x * y = n) :=
by
  intros h
  exact sorry

end simplified_equation_has_solution_l2236_223674


namespace find_intersection_point_l2236_223607

-- Define the problem conditions and question in Lean
theorem find_intersection_point 
  (slope_l1 : ℝ) (slope_l2 : ℝ) (p : ℝ × ℝ) (P : ℝ × ℝ)
  (h_l1_slope : slope_l1 = 2) 
  (h_parallel : slope_l1 = slope_l2)
  (h_passes_through : p = (-1, 1)) :
  P = (0, 3) := sorry

end find_intersection_point_l2236_223607


namespace john_remaining_income_l2236_223614

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

end john_remaining_income_l2236_223614


namespace simplify_sqrt_expression_l2236_223625

theorem simplify_sqrt_expression (t : ℝ) : (Real.sqrt (t^6 + t^4) = t^2 * Real.sqrt (t^2 + 1)) :=
by sorry

end simplify_sqrt_expression_l2236_223625


namespace problem_div_expansion_l2236_223652

theorem problem_div_expansion (m : ℝ) : ((2 * m^2 - m)^2) / (-m^2) = -4 * m^2 + 4 * m - 1 := 
by sorry

end problem_div_expansion_l2236_223652


namespace sarah_wide_reflections_l2236_223690

variables (tall_mirrors_sarah : ℕ) (tall_mirrors_ellie : ℕ) 
          (wide_mirrors_ellie : ℕ) (tall_count : ℕ) (wide_count : ℕ)
          (total_reflections : ℕ) (S : ℕ)

def reflections_in_tall_mirrors_sarah := 10 * tall_count
def reflections_in_tall_mirrors_ellie := 6 * tall_count
def reflections_in_wide_mirrors_ellie := 3 * wide_count
def total_reflections_no_wide_sarah := reflections_in_tall_mirrors_sarah + reflections_in_tall_mirrors_ellie + reflections_in_wide_mirrors_ellie

theorem sarah_wide_reflections :
  reflections_in_tall_mirrors_sarah = 30 →
  reflections_in_tall_mirrors_ellie = 18 →
  reflections_in_wide_mirrors_ellie = 15 →
  tall_count = 3 →
  wide_count = 5 →
  total_reflections = 88 →
  total_reflections = total_reflections_no_wide_sarah + 5 * S →
  S = 5 :=
sorry

end sarah_wide_reflections_l2236_223690


namespace correct_value_of_wrongly_read_number_l2236_223688

theorem correct_value_of_wrongly_read_number 
  (avg_wrong : ℝ) (n : ℕ) (wrong_value : ℝ) (avg_correct : ℝ) :
  avg_wrong = 5 →
  n = 10 →
  wrong_value = 26 →
  avg_correct = 6 →
  let sum_wrong := avg_wrong * n
  let correct_sum := avg_correct * n
  let difference := correct_sum - sum_wrong
  let correct_value := wrong_value + difference
  correct_value = 36 :=
by
  intros h_avg_wrong h_n h_wrong_value h_avg_correct
  let sum_wrong := avg_wrong * n
  let correct_sum := avg_correct * n
  let difference := correct_sum - sum_wrong
  let correct_value := wrong_value + difference
  sorry

end correct_value_of_wrongly_read_number_l2236_223688


namespace car_speed_l2236_223676

theorem car_speed (uses_one_gallon_per_30_miles : ∀ d : ℝ, d = 30 → d / 30 ≥ 1)
    (full_tank : ℝ := 10)
    (travel_time : ℝ := 5)
    (fraction_of_tank_used : ℝ := 0.8333333333333334)
    (speed : ℝ := 50) :
  let amount_of_gasoline_used := fraction_of_tank_used * full_tank
  let distance_traveled := amount_of_gasoline_used * 30
  speed = distance_traveled / travel_time :=
by
  sorry

end car_speed_l2236_223676


namespace average_rate_of_change_is_7_l2236_223694

-- Define the function
def f (x : ℝ) : ℝ := x^3 + 1

-- Define the interval
def a : ℝ := 1
def b : ℝ := 2

-- Define the proof problem
theorem average_rate_of_change_is_7 : 
  ((f b - f a) / (b - a)) = 7 :=
by 
  -- The proof would go here
  sorry

end average_rate_of_change_is_7_l2236_223694


namespace mira_additional_stickers_l2236_223689

-- Define the conditions
def mira_stickers : ℕ := 31
def row_size : ℕ := 7

-- Define the proof statement
theorem mira_additional_stickers (a : ℕ) (h : (31 + a) % 7 = 0) : 
  a = 4 := 
sorry

end mira_additional_stickers_l2236_223689


namespace hanoi_tower_l2236_223635

noncomputable def move_all_disks (n : ℕ) : Prop := 
  ∀ (A B C : Type), 
  (∃ (move : A → B), move = sorry) ∧ -- Only one disk can be moved
  (∃ (can_place : A → A → Prop), can_place = sorry) -- A disk cannot be placed on top of a smaller disk 
  → ∃ (u_n : ℕ), u_n = 2^n - 1 -- Formula for minimum number of steps

theorem hanoi_tower : ∀ n : ℕ, move_all_disks n :=
by sorry

end hanoi_tower_l2236_223635


namespace average_speed_l2236_223602

theorem average_speed (d1 d2 : ℝ) (t1 t2 : ℝ) (h1 : d1 = 90) (h2 : d2 = 75) (ht1 : t1 = 1) (ht2 : t2 = 1) :
  (d1 + d2) / (t1 + t2) = 82.5 :=
by
  sorry

end average_speed_l2236_223602


namespace total_cupcakes_correct_l2236_223645

def cupcakes_per_event : ℝ := 96.0
def num_events : ℝ := 8.0
def total_cupcakes : ℝ := cupcakes_per_event * num_events

theorem total_cupcakes_correct : total_cupcakes = 768.0 :=
by
  unfold total_cupcakes
  unfold cupcakes_per_event
  unfold num_events
  sorry

end total_cupcakes_correct_l2236_223645


namespace parallel_lines_l2236_223613

theorem parallel_lines (a : ℝ) :
  (∀ x y, x + a^2 * y + 6 = 0 → (a - 2) * x + 3 * a * y + 2 * a = 0) ↔ (a = 0 ∨ a = -1) :=
by
  sorry

end parallel_lines_l2236_223613


namespace initial_acorns_l2236_223695

theorem initial_acorns (T : ℝ) (h1 : 0.35 * T = 7) (h2 : 0.45 * T = 9) : T = 20 :=
sorry

end initial_acorns_l2236_223695


namespace greatest_integer_difference_l2236_223611

theorem greatest_integer_difference (x y : ℤ) (hx : 3 < x) (hx2 : x < 6) (hy : 6 < y) (hy2 : y < 10) :
  ∃ d : ℤ, d = y - x ∧ d = 5 :=
by
  sorry

end greatest_integer_difference_l2236_223611


namespace range_of_a_l2236_223628

variable {R : Type*} [LinearOrderedField R]

def setA (a : R) : Set R := {x | x^2 - 2*x + a ≤ 0}

def setB : Set R := {x | x^2 - 3*x + 2 ≤ 0}

theorem range_of_a (a : R) (h : setB ⊆ setA a) : a ≤ 0 := sorry

end range_of_a_l2236_223628


namespace growth_pattern_equation_l2236_223623

theorem growth_pattern_equation (x : ℕ) :
  1 + x + x^2 = 73 :=
sorry

end growth_pattern_equation_l2236_223623


namespace oranges_weight_l2236_223677

theorem oranges_weight (A O : ℕ) (h1 : O = 5 * A) (h2 : A + O = 12) : O = 10 := 
by 
  sorry

end oranges_weight_l2236_223677


namespace gain_percentage_l2236_223697

theorem gain_percentage (selling_price gain : ℕ) (h_sp : selling_price = 110) (h_gain : gain = 10) :
  (gain * 100) / (selling_price - gain) = 10 :=
by
  sorry

end gain_percentage_l2236_223697


namespace incorrect_statement_A_l2236_223605

theorem incorrect_statement_A (p q : Prop) : (p ∨ q) → ¬ (p ∧ q) := by
  intros h
  cases h with
  | inl hp => sorry
  | inr hq => sorry

end incorrect_statement_A_l2236_223605


namespace remaining_fruits_l2236_223691

theorem remaining_fruits (initial_apples initial_oranges initial_mangoes taken_apples twice_taken_apples taken_mangoes) : 
  initial_apples = 7 → 
  initial_oranges = 8 → 
  initial_mangoes = 15 → 
  taken_apples = 2 → 
  twice_taken_apples = 2 * taken_apples → 
  taken_mangoes = 2 * initial_mangoes / 3 → 
  initial_apples - taken_apples + initial_oranges - twice_taken_apples + initial_mangoes - taken_mangoes = 14 :=
by
  sorry

end remaining_fruits_l2236_223691


namespace find_A_l2236_223656

noncomputable def f (A B x : ℝ) : ℝ := A * x - 3 * B ^ 2
def g (B x : ℝ) : ℝ := B * x
variable (B : ℝ) (hB : B ≠ 0)

theorem find_A (h : f (A := A) B (g B 2) = 0) : A = 3 * B / 2 := by
  sorry

end find_A_l2236_223656


namespace maximum_value_existence_l2236_223659

open Real

theorem maximum_value_existence (a b c : ℝ) (h : 9 * a^2 + 4 * b^2 + 25 * c^2 = 1) :
    8 * a + 3 * b + 5 * c ≤ sqrt (373 / 36) := by
  sorry

end maximum_value_existence_l2236_223659


namespace max_value_x1_x2_l2236_223682

noncomputable def f (x : ℝ) := 1 - Real.sqrt (2 - 3 * x)
noncomputable def g (x : ℝ) := 2 * Real.log x

theorem max_value_x1_x2 (x1 x2 : ℝ) (h1 : x1 ≤ 2 / 3) (h2 : x2 > 0) (h3 : x1 - x2 = (1 - Real.sqrt (2 - 3 * x1)) - (2 * Real.log x2)) :
  x1 - x2 ≤ -25 / 48 :=
sorry

end max_value_x1_x2_l2236_223682


namespace julia_internet_speed_l2236_223646

theorem julia_internet_speed
  (songs : ℕ) (song_size : ℕ) (time_sec : ℕ)
  (h_songs : songs = 7200)
  (h_song_size : song_size = 5)
  (h_time_sec : time_sec = 1800) :
  songs * song_size / time_sec = 20 := by
  sorry

end julia_internet_speed_l2236_223646


namespace transformation_result_l2236_223668

theorem transformation_result (a b : ℝ) 
  (h1 : ∃ P : ℝ × ℝ, P = (a, b))
  (h2 : ∃ Q : ℝ × ℝ, Q = (b, a))
  (h3 : ∃ R : ℝ × ℝ, R = (2 - b, 10 - a))
  (h4 : (2 - b, 10 - a) = (-8, 2)) : 
  a - b = -2 := 
by 
  sorry

end transformation_result_l2236_223668


namespace choir_members_l2236_223657

theorem choir_members (k m n : ℕ) (h1 : n = k^2 + 11) (h2 : n = m * (m + 5)) : n ≤ 325 :=
by
  sorry -- A proof would go here, showing that n = 325 meets the criteria

end choir_members_l2236_223657


namespace trig_expression_l2236_223618

theorem trig_expression (α : ℝ) (h : Real.tan α = 2) : 
  (Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) + Real.cos α ^ 2 = 16 / 5 := 
  sorry

end trig_expression_l2236_223618


namespace smallest_integer_ends_3_divisible_5_l2236_223631

theorem smallest_integer_ends_3_divisible_5 : ∃ n : ℕ, (53 = n ∧ (n % 10 = 3) ∧ (n % 5 = 0) ∧ ∀ m : ℕ, (m % 10 = 3) ∧ (m % 5 = 0) → 53 ≤ m) :=
by {
  sorry
}

end smallest_integer_ends_3_divisible_5_l2236_223631


namespace intersection_M_N_l2236_223622

def M : Set ℝ := { x : ℝ | x^2 > 4 }
def N : Set ℝ := { x : ℝ | x = -3 ∨ x = -2 ∨ x = 2 ∨ x = 3 ∨ x = 4 }

theorem intersection_M_N : M ∩ N = { x : ℝ | x = -3 ∨ x = 3 ∨ x = 4 } :=
by
  sorry

end intersection_M_N_l2236_223622


namespace task_completion_time_l2236_223604

theorem task_completion_time (A B : ℝ) : 
  (14 * A / 80 + 10 * B / 96) = (20 * (A + B)) →
  (1 / (14 * A / 80 + 10 * B / 96)) = 480 / (84 * A + 50 * B) :=
by
  intros h
  sorry

end task_completion_time_l2236_223604


namespace exponent_proof_l2236_223661

theorem exponent_proof (n m : ℕ) (h1 : 4^n = 3) (h2 : 8^m = 5) : 2^(2*n + 3*m) = 15 :=
by
  -- Proof steps
  sorry

end exponent_proof_l2236_223661


namespace cheenu_time_difference_l2236_223601

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

end cheenu_time_difference_l2236_223601


namespace kayla_apples_correct_l2236_223699

-- Definition of Kylie and Kayla's apples
def total_apples : ℕ := 340
def kaylas_apples (k : ℕ) : ℕ := 4 * k + 10

-- The main statement to prove
theorem kayla_apples_correct :
  ∃ K : ℕ, K + kaylas_apples K = total_apples ∧ kaylas_apples K = 274 :=
sorry

end kayla_apples_correct_l2236_223699


namespace trajectory_equation_circle_equation_l2236_223678

-- Define the variables
variables {x y r : ℝ}

-- Prove the trajectory equation of the circle center P
theorem trajectory_equation (h1 : x^2 + r^2 = 2) (h2 : y^2 + r^2 = 3) : y^2 - x^2 = 1 :=
sorry

-- Prove the equation of the circle P given the distance to the line y = x
theorem circle_equation (h : (|x - y| / Real.sqrt 2) = (Real.sqrt 2) / 2) : 
  (x = y + 1 ∨ x = y - 1) → 
  ((y + 1)^2 + x^2 = 3 ∨ (y - 1)^2 + x^2 = 3) :=
sorry

end trajectory_equation_circle_equation_l2236_223678


namespace otimes_2_5_l2236_223685

def otimes (a b : ℝ) : ℝ := 4 * a + 3 * b

theorem otimes_2_5 : otimes 2 5 = 23 :=
by
  sorry

end otimes_2_5_l2236_223685


namespace fraction_ordering_l2236_223644

theorem fraction_ordering (x y : ℝ) (hx : x < 0) (hy : 0 < y ∧ y < 1) :
  (1 / x) < (y / x) ∧ (y / x) < (y^2 / x) :=
by
  sorry

end fraction_ordering_l2236_223644


namespace solution_set_of_absolute_value_inequality_l2236_223630

theorem solution_set_of_absolute_value_inequality :
  { x : ℝ | |x + 1| - |x - 2| > 1 } = { x : ℝ | 1 < x } :=
by 
  sorry

end solution_set_of_absolute_value_inequality_l2236_223630


namespace david_total_course_hours_l2236_223679

-- Definitions based on the conditions
def course_weeks : ℕ := 24
def three_hour_classes_per_week : ℕ := 2
def hours_per_three_hour_class : ℕ := 3
def four_hour_classes_per_week : ℕ := 1
def hours_per_four_hour_class : ℕ := 4
def homework_hours_per_week : ℕ := 4

-- Sum of weekly hours
def weekly_hours : ℕ := (three_hour_classes_per_week * hours_per_three_hour_class) +
                         (four_hour_classes_per_week * hours_per_four_hour_class) +
                         homework_hours_per_week

-- Total hours spent on the course
def total_hours : ℕ := weekly_hours * course_weeks

-- Prove that the total number of hours spent on the course is 336 hours
theorem david_total_course_hours : total_hours = 336 := by
  sorry

end david_total_course_hours_l2236_223679


namespace no_solution_eqn_l2236_223663

theorem no_solution_eqn (m : ℝ) :
  ¬ ∃ x : ℝ, (3 - 2 * x) / (x - 3) - (m * x - 2) / (3 - x) = -1 ↔ m = 1 :=
by
  sorry

end no_solution_eqn_l2236_223663


namespace min_value_expression_l2236_223667

theorem min_value_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  ∃ (c : ℝ), c = 216 ∧
    ∀ (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c), 
      ( (a^2 + 3*a + 2) * (b^2 + 3*b + 2) * (c^2 + 3*c + 2) / (a * b * c) ) ≥ 216 := 
sorry

end min_value_expression_l2236_223667


namespace clock_in_2023_hours_l2236_223658

theorem clock_in_2023_hours (current_time : ℕ) (h_current_time : current_time = 3) : 
  (current_time + 2023) % 12 = 10 := 
by {
  -- context: non-computational (time kept in modulo world and not real increments)
  sorry
}

end clock_in_2023_hours_l2236_223658


namespace distance_between_pathway_lines_is_5_l2236_223670

-- Define the conditions
def parallel_lines_distance (distance : ℤ) : Prop :=
  distance = 30

def pathway_length_between_lines (length : ℤ) : Prop :=
  length = 10

def pathway_line_length (length : ℤ) : Prop :=
  length = 60

-- Main proof problem
theorem distance_between_pathway_lines_is_5:
  ∀ (d : ℤ), parallel_lines_distance 30 → 
  pathway_length_between_lines 10 → 
  pathway_line_length 60 → 
  d = 5 := 
by
  sorry

end distance_between_pathway_lines_is_5_l2236_223670


namespace sum_of_cubes_four_consecutive_integers_l2236_223672

theorem sum_of_cubes_four_consecutive_integers (n : ℕ) (h : (n-1)^2 + n^2 + (n+1)^2 + (n+2)^2 = 11534) :
  (n-1)^3 + n^3 + (n+1)^3 + (n+2)^3 = 74836 :=
by
  sorry

end sum_of_cubes_four_consecutive_integers_l2236_223672


namespace range_of_a_l2236_223606

open Set

theorem range_of_a (a : ℝ) (M N : Set ℝ) (hM : ∀ x, x ∈ M ↔ x < 2) (hN : ∀ x, x ∈ N ↔ x < a) (hMN : M ⊆ N) : 2 ≤ a :=
by
  sorry

end range_of_a_l2236_223606


namespace initial_total_quantity_l2236_223692

theorem initial_total_quantity(milk_ratio water_ratio : ℕ) (W : ℕ) (x : ℕ) (h1 : milk_ratio = 3) (h2 : water_ratio = 1) (h3 : W = 100) (h4 : 3 * x / (x + 100) = 1 / 3) :
    4 * x = 50 :=
by
  sorry

end initial_total_quantity_l2236_223692


namespace arrange_snow_leopards_l2236_223673

theorem arrange_snow_leopards :
  let n := 9 -- number of leopards
  let factorial x := (Nat.factorial x) -- definition for factorial
  let tall_short_perm := 2 -- there are 2 ways to arrange the tallest and shortest leopards at the ends
  tall_short_perm * factorial (n - 2) = 10080 := by sorry

end arrange_snow_leopards_l2236_223673


namespace inscribed_square_ratios_l2236_223638

theorem inscribed_square_ratios (a b c x y : ℝ) (h_right_triangle : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_sides : a^2 + b^2 = c^2) 
  (h_leg_square : x = a) 
  (h_hyp_square : y = 5 / 18 * c) : 
  x / y = 18 / 13 := by
  sorry

end inscribed_square_ratios_l2236_223638


namespace find_principal_l2236_223641

theorem find_principal (R : ℝ) : ∃ P : ℝ, (P * (R + 5) * 10) / 100 - (P * R * 10) / 100 = 100 :=
by {
  use 200,
  sorry
}

end find_principal_l2236_223641


namespace ten_times_product_is_2010_l2236_223686

theorem ten_times_product_is_2010 (n : ℕ) (hn : 10 ≤ n ∧ n < 100) : 
  (∃ k : ℤ, 4.02 * (n : ℝ) = k) → (10 * k = 2010) :=
by
  sorry

end ten_times_product_is_2010_l2236_223686


namespace surface_area_difference_l2236_223642

theorem surface_area_difference
  (larger_cube_volume : ℝ)
  (num_smaller_cubes : ℝ)
  (smaller_cube_volume : ℝ)
  (h1 : larger_cube_volume = 125)
  (h2 : num_smaller_cubes = 125)
  (h3 : smaller_cube_volume = 1) :
  (6 * (smaller_cube_volume)^(2/3) * num_smaller_cubes) - (6 * (larger_cube_volume)^(2/3)) = 600 :=
by {
  sorry
}

end surface_area_difference_l2236_223642


namespace speed_of_man_in_still_water_l2236_223634

-- Define the parameters and conditions
def speed_in_still_water (v_m : ℝ) (v_s : ℝ) : Prop :=
    (v_m + v_s = 5) ∧  -- downstream condition
    (v_m - v_s = 7)    -- upstream condition

-- The theorem statement
theorem speed_of_man_in_still_water : 
  ∃ v_m v_s : ℝ, speed_in_still_water v_m v_s ∧ v_m = 6 := 
by
  sorry

end speed_of_man_in_still_water_l2236_223634


namespace transform_circle_to_ellipse_l2236_223624

theorem transform_circle_to_ellipse (x y x'' y'' : ℝ) (h_circle : x^2 + y^2 = 1)
  (hx_trans : x = x'' / 2) (hy_trans : y = y'' / 3) :
  (x''^2 / 4) + (y''^2 / 9) = 1 :=
by {
  sorry
}

end transform_circle_to_ellipse_l2236_223624


namespace square_area_from_diagonal_l2236_223643

theorem square_area_from_diagonal (d : ℝ) (h_d : d = 12) : ∃ (A : ℝ), A = 72 :=
by
  -- we will use the given diagonal to derive the result
  sorry

end square_area_from_diagonal_l2236_223643


namespace paintings_in_four_weeks_l2236_223654

def weekly_hours := 30
def hours_per_painting := 3
def weeks := 4

theorem paintings_in_four_weeks (w_hours : ℕ) (h_per_painting : ℕ) (n_weeks : ℕ) (result : ℕ) :
  w_hours = weekly_hours →
  h_per_painting = hours_per_painting →
  n_weeks = weeks →
  result = (w_hours / h_per_painting) * n_weeks →
  result = 40 :=
by
  intros
  sorry

end paintings_in_four_weeks_l2236_223654


namespace find_e_value_l2236_223681

theorem find_e_value : 
  ∃ e : ℝ, 12 / (-12 + 2 * e) = -11 - 2 * e ∧ e = 4 :=
by
  use 4
  sorry

end find_e_value_l2236_223681


namespace find_value_of_f_l2236_223649

noncomputable def f (ω φ : ℝ) (x : ℝ) := 2 * Real.sin (ω * x + φ)

theorem find_value_of_f (ω φ : ℝ) (h_symmetry : ∀ x : ℝ, f ω φ (π/4 + x) = f ω φ (π/4 - x)) :
  f ω φ (π/4) = 2 ∨ f ω φ (π/4) = -2 := 
sorry

end find_value_of_f_l2236_223649


namespace horse_catch_up_l2236_223664

theorem horse_catch_up :
  ∀ (x : ℕ), (240 * x = 150 * (x + 12)) → x = 20 :=
by
  intros x h
  have : 240 * x = 150 * x + 1800 := by sorry
  have : 240 * x - 150 * x = 1800 := by sorry
  have : 90 * x = 1800 := by sorry
  have : x = 1800 / 90 := by sorry
  have : x = 20 := by sorry
  exact this

end horse_catch_up_l2236_223664


namespace monthly_rent_of_shop_l2236_223684

theorem monthly_rent_of_shop
  (length width : ℕ) (rent_per_sqft : ℕ)
  (h_length : length = 20) (h_width : width = 18) (h_rent : rent_per_sqft = 48) :
  (length * width * rent_per_sqft) / 12 = 1440 := 
by
  sorry

end monthly_rent_of_shop_l2236_223684


namespace initial_big_bottles_l2236_223629

theorem initial_big_bottles (B : ℝ)
  (initial_small : ℝ := 6000)
  (sold_small : ℝ := 0.11)
  (sold_big : ℝ := 0.12)
  (remaining_total : ℝ := 18540) :
  (initial_small * (1 - sold_small) + B * (1 - sold_big) = remaining_total) → B = 15000 :=
by
  intro h
  sorry

end initial_big_bottles_l2236_223629


namespace finite_tasty_integers_l2236_223665

def is_terminating_decimal (a b : ℕ) : Prop :=
  ∃ (c : ℕ), (b = c * 2^a * 5^a)

def is_tasty (n : ℕ) : Prop :=
  n > 2 ∧ ∀ (a b : ℕ), a + b = n → (is_terminating_decimal a b ∨ is_terminating_decimal b a)

theorem finite_tasty_integers : 
  ∃ (N : ℕ), ∀ (n : ℕ), n > N → ¬ is_tasty n :=
sorry

end finite_tasty_integers_l2236_223665


namespace red_large_toys_count_l2236_223647

def percentage_red : ℝ := 0.25
def percentage_green : ℝ := 0.20
def percentage_blue : ℝ := 0.15
def percentage_yellow : ℝ := 0.25
def percentage_orange : ℝ := 0.15

def red_small : ℝ := 0.06
def red_medium : ℝ := 0.08
def red_large : ℝ := 0.07
def red_extra_large : ℝ := 0.04

def green_small : ℝ := 0.04
def green_medium : ℝ := 0.07
def green_large : ℝ := 0.05
def green_extra_large : ℝ := 0.04

def blue_small : ℝ := 0.06
def blue_medium : ℝ := 0.03
def blue_large : ℝ := 0.04
def blue_extra_large : ℝ := 0.02

def yellow_small : ℝ := 0.08
def yellow_medium : ℝ := 0.10
def yellow_large : ℝ := 0.05
def yellow_extra_large : ℝ := 0.02

def orange_small : ℝ := 0.09
def orange_medium : ℝ := 0.06
def orange_large : ℝ := 0.05
def orange_extra_large : ℝ := 0.05

def green_large_count : ℕ := 47

noncomputable def total_green_toys := green_large_count / green_large

noncomputable def total_toys := total_green_toys / percentage_green

noncomputable def red_large_toys := total_toys * red_large

theorem red_large_toys_count : red_large_toys = 329 := by
  sorry

end red_large_toys_count_l2236_223647


namespace find_p_of_five_l2236_223698

-- Define the cubic polynomial and the conditions
def cubic_poly (p : ℝ → ℝ) :=
  ∀ x, ∃ a b c d, p x = a * x^3 + b * x^2 + c * x + d

def satisfies_conditions (p : ℝ → ℝ) :=
  p 1 = 1 ^ 2 ∧
  p 2 = 2 ^ 2 ∧
  p 3 = 3 ^ 2 ∧
  p 4 = 4 ^ 2

-- Theorem statement to be proved
theorem find_p_of_five (p : ℝ → ℝ) (hcubic : cubic_poly p) (hconditions : satisfies_conditions p) : p 5 = 25 :=
by
  sorry

end find_p_of_five_l2236_223698


namespace Nicki_runs_30_miles_per_week_in_second_half_l2236_223648

/-
  Nicki ran 20 miles per week for the first half of the year.
  There are 26 weeks in each half of the year.
  She ran a total of 1300 miles for the year.
  Prove that Nicki ran 30 miles per week in the second half of the year.
-/

theorem Nicki_runs_30_miles_per_week_in_second_half (weekly_first_half : ℕ) (weeks_per_half : ℕ) (total_miles : ℕ) :
  weekly_first_half = 20 → weeks_per_half = 26 → total_miles = 1300 → 
  (total_miles - (weekly_first_half * weeks_per_half)) / weeks_per_half = 30 :=
by
  intros h1 h2 h3
  sorry

end Nicki_runs_30_miles_per_week_in_second_half_l2236_223648


namespace num_ways_to_arrange_BANANA_l2236_223609

theorem num_ways_to_arrange_BANANA : 
  let total_letters := 6
  let num_A := 3
  let num_N := 2
  let total_arrangements := Nat.factorial total_letters / (Nat.factorial num_A * Nat.factorial num_N)
  total_arrangements = 60 :=
by
  sorry

end num_ways_to_arrange_BANANA_l2236_223609


namespace painted_cube_count_is_three_l2236_223610

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

end painted_cube_count_is_three_l2236_223610


namespace minimize_b_plus_4c_l2236_223608

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

end minimize_b_plus_4c_l2236_223608


namespace parabola_vertex_value_of_a_l2236_223637

-- Define the conditions as given in the math problem
variables (a b c : ℤ)
def quadratic_fun (a b c : ℤ) (x : ℤ) : ℤ := a * x^2 + b * x + c

-- Given conditions about the vertex and a point on the parabola
def vertex_condition : Prop := (quadratic_fun a b c 2 = 3)
def point_condition : Prop := (quadratic_fun a b c 1 = 0)

-- Statement to prove
theorem parabola_vertex_value_of_a : vertex_condition a b c ∧ point_condition a b c → a = -3 :=
sorry

end parabola_vertex_value_of_a_l2236_223637


namespace number_of_customers_l2236_223683

theorem number_of_customers 
  (offices sandwiches_per_office total_sandwiches group_sandwiches_per_customer half_group_sandwiches : ℕ)
  (h1 : offices = 3)
  (h2 : sandwiches_per_office = 10)
  (h3 : total_sandwiches = 54)
  (h4 : group_sandwiches_per_customer = 4)
  (h5 : half_group_sandwiches = 54 - (3 * 10))
  : half_group_sandwiches = 24 → 2 * 12 = 24 :=
by
  sorry

end number_of_customers_l2236_223683


namespace total_chairs_calc_l2236_223633

-- Defining the condition of having 27 rows
def rows : ℕ := 27

-- Defining the condition of having 16 chairs per row
def chairs_per_row : ℕ := 16

-- Stating the theorem that the total number of chairs is 432
theorem total_chairs_calc : rows * chairs_per_row = 432 :=
by
  sorry

end total_chairs_calc_l2236_223633


namespace pow_expression_eq_l2236_223617

theorem pow_expression_eq : (-3)^(4^2) + 2^(3^2) = 43047233 := by
  sorry

end pow_expression_eq_l2236_223617


namespace right_triangle_perimeter_l2236_223653

theorem right_triangle_perimeter
  (a b c : ℝ)
  (h_right: a^2 + b^2 = c^2)
  (h_area: (1/2) * a * b = (1/2) * c) :
  a + b + c = 2 * (Real.sqrt 2 + 1) :=
sorry

end right_triangle_perimeter_l2236_223653


namespace friends_division_ways_l2236_223650

theorem friends_division_ways : (4 ^ 8 = 65536) :=
by
  sorry

end friends_division_ways_l2236_223650


namespace graph_passes_through_fixed_point_l2236_223620

theorem graph_passes_through_fixed_point (a : ℝ) : (0, 2) ∈ { p : ℝ × ℝ | ∃ x, p = (x, a ^ x + 1) } :=
sorry

end graph_passes_through_fixed_point_l2236_223620


namespace unique_sequence_and_a_2002_l2236_223636

-- Define the sequence (a_n)
noncomputable def a : ℕ → ℕ := -- define the correct sequence based on conditions
  -- we would define a such as in the constructive steps in the solution, but here's a placeholder
  sorry

-- Prove the uniqueness and finding a_2002
theorem unique_sequence_and_a_2002 :
  (∀ n : ℕ, ∃! (i j k : ℕ), n = a i + 2 * a j + 4 * a k) ∧ a 2002 = 1227132168 :=
by
  sorry

end unique_sequence_and_a_2002_l2236_223636


namespace books_total_correct_l2236_223669

-- Define the constants for the number of books obtained each day
def books_day1 : ℕ := 54
def books_day2_total : ℕ := 23
def books_day2_kept : ℕ := 12
def books_day3_multiplier : ℕ := 3

-- Calculate the total number of books obtained each day
def books_day3 := books_day3_multiplier * books_day2_total
def total_books := books_day1 + books_day2_kept + books_day3

-- The theorem to prove
theorem books_total_correct : total_books = 135 := by
  sorry

end books_total_correct_l2236_223669


namespace hot_air_balloon_height_l2236_223666

theorem hot_air_balloon_height (altitude_temp_decrease_per_1000m : ℝ) 
  (ground_temp : ℝ) (high_altitude_temp : ℝ) :
  altitude_temp_decrease_per_1000m = 6 →
  ground_temp = 8 →
  high_altitude_temp = -1 →
  ∃ (height : ℝ), height = 1500 :=
by
  intro h1 h2 h3
  have temp_change := ground_temp - high_altitude_temp
  have height := (temp_change / altitude_temp_decrease_per_1000m) * 1000
  exact Exists.intro height sorry -- height needs to be computed here

end hot_air_balloon_height_l2236_223666


namespace ratio_avg_eq_42_l2236_223660

theorem ratio_avg_eq_42 (a b c d : ℕ)
  (h1 : ∃ k : ℕ, a = 2 * k ∧ b = 3 * k ∧ c = 4 * k ∧ d = 5 * k)
  (h2 : (a + b + c + d) / 4 = 42) : a = 24 :=
by sorry

end ratio_avg_eq_42_l2236_223660


namespace solution_exists_l2236_223671

noncomputable def find_A_and_B : Prop :=
  ∃ A B : ℚ, 
    (A, B) = (75 / 16, 21 / 16) ∧ 
    ∀ x : ℚ, x ≠ 12 ∧ x ≠ -4 → 
    (6 * x + 3) / ((x - 12) * (x + 4)) = A / (x - 12) + B / (x + 4)

theorem solution_exists : find_A_and_B :=
sorry

end solution_exists_l2236_223671
