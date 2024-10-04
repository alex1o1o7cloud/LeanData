import Mathlib

namespace intersection_P_Q_l58_58166

open Set

def P : Set ℝ := {1, 2}
def Q : Set ℝ := {x | abs x < 2}

theorem intersection_P_Q : P ∩ Q = {1} :=
by
  sorry

end intersection_P_Q_l58_58166


namespace bus_total_people_l58_58173

def number_of_boys : ℕ := 50
def additional_girls (b : ℕ) : ℕ := (2 * b) / 5
def number_of_girls (b : ℕ) : ℕ := b + additional_girls b
def total_people (b g : ℕ) : ℕ := b + g + 3  -- adding 3 for the driver, assistant, and teacher

theorem bus_total_people : total_people number_of_boys (number_of_girls number_of_boys) = 123 :=
by
  sorry

end bus_total_people_l58_58173


namespace max_vehicles_div_by_100_l58_58329

noncomputable def max_vehicles_passing_sensor (n : ℕ) : ℕ :=
  2 * (20000 * n / (5 + 10 * n))

theorem max_vehicles_div_by_100 : 
  (∀ n : ℕ, (n > 0) → (∃ M : ℕ, M = max_vehicles_passing_sensor n ∧ M / 100 = 40)) :=
sorry

end max_vehicles_div_by_100_l58_58329


namespace acid_solution_l58_58327

theorem acid_solution (x y : ℝ) (h1 : 0.3 * x + 0.1 * y = 90)
  (h2 : x + y = 600) : x = 150 ∧ y = 450 :=
by
  sorry

end acid_solution_l58_58327


namespace calculate_expression_l58_58245

theorem calculate_expression : (3 / 4 - 1 / 8) ^ 5 = 3125 / 32768 :=
by
  sorry

end calculate_expression_l58_58245


namespace pudding_distribution_l58_58894

theorem pudding_distribution {puddings students : ℕ} (h1 : puddings = 315) (h2 : students = 218) : 
  ∃ (additional_puddings : ℕ), additional_puddings >= 121 ∧ ∃ (cups_per_student : ℕ), 
  (puddings + additional_puddings) ≥ students * cups_per_student :=
by
  sorry

end pudding_distribution_l58_58894


namespace range_of_f_l58_58313

noncomputable def f (x : ℝ) : ℝ := (1/3) ^ (x^2 - 2*x)

theorem range_of_f : Set.Ioo 0 3 ∪ {3} = { y | ∃ x, f x = y } :=
by
  sorry

end range_of_f_l58_58313


namespace volume_of_cube_with_diagonal_l58_58720

theorem volume_of_cube_with_diagonal (d : ℝ) (h : d = 5 * real.sqrt 3) : 
  ∃ (V : ℝ), V = 125 := 
by
  -- Definitions and conditions from the problem are used directly
  let s := d / real.sqrt 3
  sorry

end volume_of_cube_with_diagonal_l58_58720


namespace harry_sandy_midpoint_l58_58383

theorem harry_sandy_midpoint :
  ∃ (x y : ℤ), x = 9 ∧ y = -2 → ∃ (a b : ℤ), a = 1 ∧ b = 6 → ((9 + 1) / 2, (-2 + 6) / 2) = (5, 2) := 
by 
  sorry

end harry_sandy_midpoint_l58_58383


namespace eq_has_positive_integer_solution_l58_58605

theorem eq_has_positive_integer_solution (a : ℤ) :
  (∃ x : ℕ+, (x : ℤ) - 4 - 2 * (a * x - 1) = 2) → a = 0 :=
by
  sorry

end eq_has_positive_integer_solution_l58_58605


namespace integer_values_of_a_count_integer_values_of_a_l58_58084

theorem integer_values_of_a (a : ℤ) :
  (∃ x1 x2 : ℤ, x1 ≠ x2 ∧ (x1 * x1 + a * x1 + 9 * a = 0) ∧ (x2 * x2 + a * x2 + 9 * a = 0)) →
  a ∈ {0, -12, -64} :=
by
  sorry

theorem count_integer_values_of_a : 
  {a : ℤ | ∃ x1 x2 : ℤ, x1 ≠ x2 ∧ (x1 * x1 + a * x1 + 9 * a = 0) ∧ (x2 * x2 + a * x2 + 9 * a = 0)}.to_finset.card = 3 :=
by
  sorry

end integer_values_of_a_count_integer_values_of_a_l58_58084


namespace boarders_joined_l58_58754

theorem boarders_joined (initial_boarders : ℕ) (initial_day_scholars : ℕ) (final_ratio_num : ℕ) (final_ratio_denom : ℕ) (new_boarders : ℕ)
  (initial_ratio_boarders_to_day_scholars : initial_boarders * 16 = 7 * initial_day_scholars)
  (initial_boarders_eq : initial_boarders = 560)
  (final_ratio : (initial_boarders + new_boarders) * 2 = final_day_scholars)
  (day_scholars_eq : initial_day_scholars = 1280) : 
  new_boarders = 80 := by
  sorry

end boarders_joined_l58_58754


namespace find_b_n_find_T_n_l58_58960

-- Conditions
def S (n : ℕ) : ℕ := 3 * n^2 + 8 * n
def a (n : ℕ) : ℕ := S n - S (n - 1) -- provided n > 1
def b : ℕ → ℕ := sorry -- This is what we need to prove
def c (n : ℕ) : ℕ := (a n + 1)^(n + 1) / (b n + 2)^n  -- Definition of c_n
def T (n : ℕ) : ℕ := sorry -- The sum of the first n terms of c_n

-- Proof requirements
def proof_b_n := ∀ n : ℕ, b n = 3 * n + 1
def proof_T_n := ∀ n : ℕ, T n = 3 * n * 2^(n+2)

theorem find_b_n : proof_b_n := 
by sorry

theorem find_T_n : proof_T_n := 
by sorry

end find_b_n_find_T_n_l58_58960


namespace distinct_integer_values_of_a_l58_58086

theorem distinct_integer_values_of_a : 
  let eq_has_integer_solutions (a : ℤ) : Prop := 
    ∃ (x y : ℤ), (x^2 + a*x + 9*a = 0) ∧ (y^2 + a*y + 9*a = 0) in
  (finset.univ.filter eq_has_integer_solutions).card = 5 := 
sorry

end distinct_integer_values_of_a_l58_58086


namespace tom_coins_worth_l58_58589

-- Definitions based on conditions:
def total_coins : ℕ := 30
def value_difference_cents : ℕ := 90
def nickel_value_cents : ℕ := 5
def dime_value_cents : ℕ := 10

-- Main theorem statement:
theorem tom_coins_worth (n d : ℕ) (h1 : d = total_coins - n) 
    (h2 : (nickel_value_cents * n + dime_value_cents * d) - (dime_value_cents * n + nickel_value_cents * d) = value_difference_cents) : 
    (nickel_value_cents * n + dime_value_cents * d) = 180 :=
by
  sorry -- Proof omitted.

end tom_coins_worth_l58_58589


namespace perpendicular_lines_a_eq_3_l58_58666

theorem perpendicular_lines_a_eq_3 (a : ℝ) :
  let l₁ := (a + 1) * x + 2 * y + 6
  let l₂ := x + (a - 5) * y + a^2 - 1
  (a ≠ 5 → -((a + 1) / 2) * (1 / (5 - a)) = -1) → a = 3 := by
  intro l₁ l₂ h
  sorry

end perpendicular_lines_a_eq_3_l58_58666


namespace min_value_x4_y3_z2_l58_58165

theorem min_value_x4_y3_z2 (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : 1/x + 1/y + 1/z = 9) : 
  x^4 * y^3 * z^2 ≥ 1 / 9^9 :=
by 
  -- Proof goes here
  sorry

end min_value_x4_y3_z2_l58_58165


namespace slope_of_perpendicular_line_l58_58945

theorem slope_of_perpendicular_line (a b c : ℝ) (h : a ≠ 0 ∧ b ≠ 0) :
  ∃ m : ℝ, a * x - b * y = c → m = - (b / a) :=
by
  -- Here we state the definition and conditions provided in the problem
  -- And indicate what we want to prove (that the slope is -b/a in this case)
  sorry

end slope_of_perpendicular_line_l58_58945


namespace train_b_speed_l58_58208

variable (v : ℝ) -- the speed of Train B

theorem train_b_speed 
  (speedA : ℝ := 30) -- speed of Train A
  (head_start_hours : ℝ := 2) -- head start time in hours
  (overtake_distance : ℝ := 285) -- distance at which Train B overtakes Train A
  (train_a_travel_distance : ℝ := speedA * head_start_hours) -- distance Train A travels in the head start time
  (total_distance : ℝ := 345) -- total distance Train B travels to overtake Train A
  (train_a_travel_time : ℝ := overtake_distance / speedA) -- time taken by Train A to travel the overtake distance
  : v * train_a_travel_time = total_distance → v = 36.32 :=
by
  sorry

end train_b_speed_l58_58208


namespace sound_speed_temperature_l58_58891

theorem sound_speed_temperature (v : ℝ) (T : ℝ) (h1 : v = 0.4) (h2 : T = 15 * v^2) :
  T = 2.4 :=
by {
  sorry
}

end sound_speed_temperature_l58_58891


namespace decrease_percent_in_revenue_l58_58915

theorem decrease_percent_in_revenue
  (T C : ℝ) -- T = original tax, C = original consumption
  (h1 : 0 < T) -- ensuring that T is positive
  (h2 : 0 < C) -- ensuring that C is positive
  (new_tax : ℝ := 0.75 * T) -- new tax is 75% of original tax
  (new_consumption : ℝ := 1.10 * C) -- new consumption is 110% of original consumption
  (original_revenue : ℝ := T * C) -- original revenue
  (new_revenue : ℝ := (0.75 * T) * (1.10 * C)) -- new revenue
  (decrease_percent : ℝ := ((T * C - (0.75 * T) * (1.10 * C)) / (T * C)) * 100) -- decrease percent
  : decrease_percent = 17.5 :=
by
  sorry

end decrease_percent_in_revenue_l58_58915


namespace combined_flock_after_5_years_l58_58644

theorem combined_flock_after_5_years :
  let initial_flock : ℕ := 100
  let annual_net_gain : ℕ := 30 - 20
  let years : ℕ := 5
  let joined_flock : ℕ := 150
  let final_flock := initial_flock + annual_net_gain * years + joined_flock
  in final_flock = 300 :=
by
  sorry

end combined_flock_after_5_years_l58_58644


namespace count_six_digit_palindromes_l58_58495

def num_six_digit_palindromes : ℕ := 9000

theorem count_six_digit_palindromes :
  (∃ a b c d : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ 0 ≤ d ∧ d ≤ 9 ∧
     num_six_digit_palindromes = 9000) :=
sorry

end count_six_digit_palindromes_l58_58495


namespace assignment_problem_l58_58439

theorem assignment_problem (a b c : ℕ) (h1 : a = 10) (h2 : b = 20) (h3 : c = 30) :
  let a := b
  let b := c
  let c := a
  a = 20 ∧ b = 30 ∧ c = 20 :=
by
  sorry

end assignment_problem_l58_58439


namespace floor_add_self_eq_20_5_iff_l58_58786

theorem floor_add_self_eq_20_5_iff (s : ℝ) : (⌊s⌋₊ : ℝ) + s = 20.5 ↔ s = 10.5 :=
by
  sorry

end floor_add_self_eq_20_5_iff_l58_58786


namespace jordan_has_11_oreos_l58_58127

-- Define the conditions
def jamesOreos (x : ℕ) : ℕ := 3 + 2 * x
def totalOreos (jordanOreos : ℕ) : ℕ := 36

-- Theorem stating the problem that Jordan has 11 Oreos given the conditions
theorem jordan_has_11_oreos (x : ℕ) (h1 : jamesOreos x + x = totalOreos x) : x = 11 :=
by
  sorry

end jordan_has_11_oreos_l58_58127


namespace max_q_value_l58_58149

theorem max_q_value (A M C : ℕ) (h : A + M + C = 15) : 
  (A * M * C + A * M + M * C + C * A) ≤ 200 :=
sorry

end max_q_value_l58_58149


namespace determine_x_l58_58074

theorem determine_x (x : ℝ) (h : 9 * x^2 + 2 * x^2 + 3 * x^2 / 2 = 300) : x = 2 * Real.sqrt 6 :=
by sorry

end determine_x_l58_58074


namespace all_integers_appear_exactly_once_l58_58695

noncomputable def sequence_of_integers (a : ℕ → ℤ) : Prop :=
∀ n : ℕ, ∃ m : ℕ, a m > 0 ∧ ∃ m' : ℕ, a m' < 0

noncomputable def distinct_modulo_n (a : ℕ → ℤ) : Prop :=
∀ n : ℕ, (∀ i j : ℕ, i < j ∧ j < n → a i % n ≠ a j % n)

theorem all_integers_appear_exactly_once
  (a : ℕ → ℤ)
  (h_seq : sequence_of_integers a)
  (h_distinct : distinct_modulo_n a) :
  ∀ x : ℤ, ∃! i : ℕ, a i = x := 
sorry

end all_integers_appear_exactly_once_l58_58695


namespace find_value_l58_58659

theorem find_value (a b : ℝ) (h1 : a + b = 1) (h2 : a * b = 108) : a^2 * b + a * b^2 = 108 :=
by
  sorry

end find_value_l58_58659


namespace sophie_oranges_per_day_l58_58905

/-- Sophie and Hannah together eat a certain number of fruits in 30 days.
    Given Hannah eats 40 grapes every day, prove that Sophie eats 20 oranges every day. -/
theorem sophie_oranges_per_day (total_fruits : ℕ) (grapes_per_day : ℕ) (days : ℕ)
  (total_days_fruits : total_fruits = 1800) (hannah_grapes : grapes_per_day = 40) (days_count : days = 30) :
  (total_fruits - grapes_per_day * days) / days = 20 :=
by
  sorry

end sophie_oranges_per_day_l58_58905


namespace lathes_equal_parts_processed_15_minutes_l58_58730

variable (efficiencyA efficiencyB efficiencyC : ℝ)
variable (timeA timeB timeC : ℕ)

/-- Lathe A starts 10 minutes before lathe C -/
def start_time_relation_1 : Prop := timeA + 10 = timeC

/-- Lathe C starts 5 minutes before lathe B -/
def start_time_relation_2 : Prop := timeC + 5 = timeB

/-- After lathe B has been working for 10 minutes, B and C process the same number of parts -/
def parts_processed_relation_1 (efficiencyB efficiencyC : ℝ) : Prop :=
  10 * efficiencyB = (10 + 5) * efficiencyC

/-- After lathe C has been working for 30 minutes, A and C process the same number of parts -/
def parts_processed_relation_2 (efficiencyA efficiencyC : ℝ) : Prop :=
  (30 + 10) * efficiencyA = 30 * efficiencyC

/-- How many minutes after lathe B starts will it have processed the same number of standard parts as lathe A? -/
theorem lathes_equal_parts_processed_15_minutes
  (h₁ : start_time_relation_1 timeA timeC)
  (h₂ : start_time_relation_2 timeC timeB)
  (h₃ : parts_processed_relation_1 efficiencyB efficiencyC)
  (h₄ : parts_processed_relation_2 efficiencyA efficiencyC) :
  ∃ t : ℕ, (t = 15) ∧ ( (timeB + t) * efficiencyB = (timeA + (timeB + t - timeA)) * efficiencyA ) := sorry

end lathes_equal_parts_processed_15_minutes_l58_58730


namespace unique_geometric_sequence_l58_58269

theorem unique_geometric_sequence (a : ℝ) (q : ℝ) (a_n b_n : ℕ → ℝ) 
    (h1 : a > 0) 
    (h2 : a_n 1 = a) 
    (h3 : b_n 1 - a_n 1 = 1) 
    (h4 : b_n 2 - a_n 2 = 2) 
    (h5 : b_n 3 - a_n 3 = 3) 
    (h6 : ∀ n, a_n (n + 1) = a_n n * q) 
    (h7 : ∀ n, b_n (n + 1) = b_n n * q) : 
    a = 1 / 3 := sorry

end unique_geometric_sequence_l58_58269


namespace smallest_non_palindrome_power_of_13_is_2197_l58_58652

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString in
  s = s.reverse

theorem smallest_non_palindrome_power_of_13_is_2197 : ∃ n : ℕ, 13^n = 2197 ∧ ¬ is_palindrome (13^n) := 
by
  use 3
  have h1 : 13^3 = 2197 := by norm_num
  exact ⟨h1, by norm_num; sorry⟩

end smallest_non_palindrome_power_of_13_is_2197_l58_58652


namespace circumcircle_diameter_l58_58547

theorem circumcircle_diameter (a : ℝ) (c : ℝ) (B : ℝ) (S : ℝ) (h₁ : a = 2) (h₂ : B = 60 * (π / 180)) (h₃ : S = sqrt 3) :
  (2 / (sin (60 * (π / 180)))) = 4 * (sqrt 3) / 3 := 
sorry

end circumcircle_diameter_l58_58547


namespace hockey_games_in_season_l58_58459

-- Define the conditions
def games_per_month : Nat := 13
def season_months : Nat := 14

-- Define the total number of hockey games in the season
def total_games_in_season (games_per_month : Nat) (season_months : Nat) : Nat :=
  games_per_month * season_months

-- Define the theorem to prove
theorem hockey_games_in_season :
  total_games_in_season games_per_month season_months = 182 :=
by
  -- Proof omitted
  sorry

end hockey_games_in_season_l58_58459


namespace students_not_yes_for_either_subject_l58_58056

variable (total_students yes_m no_m unsure_m yes_r no_r unsure_r yes_only_m : ℕ)

theorem students_not_yes_for_either_subject :
  total_students = 800 →
  yes_m = 500 →
  no_m = 200 →
  unsure_m = 100 →
  yes_r = 400 →
  no_r = 100 →
  unsure_r = 300 →
  yes_only_m = 150 →
  ∃ students_not_yes, students_not_yes = total_students - (yes_only_m + (yes_m - yes_only_m) + (yes_r - (yes_m - yes_only_m))) ∧ students_not_yes = 400 :=
by
  intros ht yt1 nnm um ypr ynr ur yom
  sorry

end students_not_yes_for_either_subject_l58_58056


namespace method1_three_sessions_cost_method2_more_cost_effective_for_nine_sessions_method1_allows_more_sessions_l58_58476

/-- Method 1: Membership card costs 200 yuan + 10 yuan per swim session. -/
def method1_cost (num_sessions : ℕ) : ℕ := 200 + 10 * num_sessions

/-- Method 2: Each swim session costs 30 yuan. -/
def method2_cost (num_sessions : ℕ) : ℕ := 30 * num_sessions

/-- Problem (1): Total cost for 3 swim sessions using Method 1 is 230 yuan. -/
theorem method1_three_sessions_cost : method1_cost 3 = 230 := by
  sorry

/-- Problem (2): Method 2 is more cost-effective than Method 1 for 9 swim sessions. -/
theorem method2_more_cost_effective_for_nine_sessions : method2_cost 9 < method1_cost 9 := by
  sorry

/-- Problem (3): Method 1 allows more sessions than Method 2 within a budget of 600 yuan. -/
theorem method1_allows_more_sessions : (600 - 200) / 10 > 600 / 30 := by
  sorry

end method1_three_sessions_cost_method2_more_cost_effective_for_nine_sessions_method1_allows_more_sessions_l58_58476


namespace isosceles_triangle_angle_l58_58665

-- Define the triangle \triangle ABC with the given conditions
def Triangle (A B C : Point) :=
  ∃ (v1 v2 v3 : Fin 3 → ℝ), 
    A = v1 ∧ B = v2 ∧ C = v3 ∧ 
    dist A B = dist A C ∧ -- Condition: AB = AC
    abs (angle v1 v2 v3) + abs (angle v2 v3 v1) + abs (angle v3 v1 v2) = 180  -- Property: sum of angles of a triangle is 180°

-- Define the proposition to prove
def Prop (A B C : Point) : Prop :=
  ∀ (t : Triangle A B C), 
  angle A B C < 90

-- Rewrite the problem in Lean statement
theorem isosceles_triangle_angle (A B C : Point) (h : Triangle A B C) :
  Prop A B C :=
by
  sorry

end isosceles_triangle_angle_l58_58665


namespace largest_sum_of_watch_digits_l58_58621

theorem largest_sum_of_watch_digits : ∃ s : ℕ, s = 23 ∧ 
  (∀ h m : ℕ, h < 24 → m < 60 → s ≤ (h / 10 + h % 10 + m / 10 + m % 10)) :=
by
  sorry

end largest_sum_of_watch_digits_l58_58621


namespace find_brown_mms_second_bag_l58_58870

variable (x : ℕ)

-- Definitions based on the conditions
def BrownMmsFirstBag := 9
def BrownMmsThirdBag := 8
def BrownMmsFourthBag := 8
def BrownMmsFifthBag := 3
def AveBrownMmsPerBag := 8
def NumBags := 5

-- Condition specifying the average brown M&Ms per bag
axiom average_condition : AveBrownMmsPerBag = (BrownMmsFirstBag + x + BrownMmsThirdBag + BrownMmsFourthBag + BrownMmsFifthBag) / NumBags

-- Prove the number of brown M&Ms in the second bag
theorem find_brown_mms_second_bag : x = 12 := by
  sorry

end find_brown_mms_second_bag_l58_58870


namespace correct_transformation_C_l58_58608

-- Define the conditions as given in the problem
def condition_A (x : ℝ) : Prop := 4 + x = 3 ∧ x = 3 - 4
def condition_B (x : ℝ) : Prop := (1 / 3) * x = 0 ∧ x = 0
def condition_C (y : ℝ) : Prop := 5 * y = -4 * y + 2 ∧ 5 * y + 4 * y = 2
def condition_D (a : ℝ) : Prop := (1 / 2) * a - 1 = 3 * a ∧ a - 2 = 6 * a

-- The theorem to prove that condition_C is correctly transformed
theorem correct_transformation_C : condition_C 1 := 
by sorry

end correct_transformation_C_l58_58608


namespace cos_value_l58_58362

theorem cos_value (α : ℝ) (h : Real.sin (π / 5 - α) = 1 / 4) : 
  Real.cos (2 * α + 3 * π / 5) = -7 / 8 := 
by
  sorry

end cos_value_l58_58362


namespace gcd_5280_12155_l58_58904

theorem gcd_5280_12155 : Nat.gcd 5280 12155 = 55 := by
  sorry

end gcd_5280_12155_l58_58904


namespace possible_values_of_n_l58_58254

theorem possible_values_of_n (E M n : ℕ) (h1 : M + 3 = n * (E - 3)) (h2 : E + n = 3 * (M - n)) : 
  n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 7 :=
sorry

end possible_values_of_n_l58_58254


namespace max_value_of_q_l58_58139

theorem max_value_of_q (A M C : ℕ) (h_sum : A + M + C = 15) : 
  A * M * C + A * M + M * C + C * A ≤ 200 :=
sorry

end max_value_of_q_l58_58139


namespace smallest_non_palindrome_power_of_13_l58_58656

def is_smallest_non_palindrome_power_of_13 (n : ℕ) : Prop :=
  (∃ k : ℕ, n = 13^k) ∧ ¬ nat.is_palindrome n ∧ ∀ m, (∃ k : ℕ, m = 13^k) ∧ ¬ nat.is_palindrome m → n ≤ m

theorem smallest_non_palindrome_power_of_13 : is_smallest_non_palindrome_power_of_13 169 :=
by {
  sorry
}

end smallest_non_palindrome_power_of_13_l58_58656


namespace chickens_egg_production_l58_58713

/--
Roberto buys 4 chickens for $20 each. The chickens cost $1 in total per week to feed.
Roberto used to buy 1 dozen eggs (12 eggs) a week, spending $2 per dozen.
After 81 weeks, the total cost of raising chickens will be cheaper than buying the eggs.
Prove that each chicken produces 3 eggs per week.
-/
theorem chickens_egg_production:
  let chicken_cost := 20
  let num_chickens := 4
  let weekly_feed_cost := 1
  let weekly_eggs_cost := 2
  let dozen_eggs := 12
  let weeks := 81

  -- Cost calculations
  let total_chicken_cost := num_chickens * chicken_cost
  let total_feed_cost := weekly_feed_cost * weeks
  let total_raising_cost := total_chicken_cost + total_feed_cost
  let total_buying_cost := weekly_eggs_cost * weeks

  -- Ensure cost condition
  (total_raising_cost <= total_buying_cost) →
  
  -- Egg production calculation
  (dozen_eggs / num_chickens) = 3 :=
by
  intros
  sorry

end chickens_egg_production_l58_58713


namespace radius_of_O2016_l58_58683

-- Define the centers and radii of circles
variable (a : ℝ) (n : ℕ) (r : ℕ → ℝ)

-- Conditions
-- Radius of the first circle
def initial_radius := r 1 = 1 / (2 * a)
-- Sequence of the radius difference based on solution step
def radius_recursive := ∀ n > 1, r (n + 1) - r n = 1 / a

-- The final statement to be proven
theorem radius_of_O2016 (h1 : initial_radius a r) (h2 : radius_recursive a r) :
  r 2016 = 4031 / (2 * a) := 
by sorry

end radius_of_O2016_l58_58683


namespace construction_work_rate_l58_58336

theorem construction_work_rate (C : ℝ) 
  (h1 : ∀ t1 : ℝ, t1 = 10 → t1 * 8 = 80)
  (h2 : ∀ t2 : ℝ, t2 = 15 → t2 * C + 80 ≥ 300)
  (h3 : ∀ t : ℝ, t = 25 → ∀ t1 t2 : ℝ, t = t1 + t2 → t1 = 10 → t2 = 15)
  : C = 14.67 :=
by
  sorry

end construction_work_rate_l58_58336


namespace arith_seq_a1_a2_a3_sum_l58_58954

def arithmetic_seq (a : ℕ → ℤ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

theorem arith_seq_a1_a2_a3_sum (a : ℕ → ℤ) (h_seq : arithmetic_seq a)
  (h1 : a 1 = 2) (h_sum : a 1 + a 2 + a 3 = 18) :
  a 4 + a 5 + a 6 = 54 :=
sorry

end arith_seq_a1_a2_a3_sum_l58_58954


namespace water_left_after_experiment_l58_58069

theorem water_left_after_experiment (initial_water : ℝ) (used_water : ℝ) (result_water : ℝ) 
  (h1 : initial_water = 3) 
  (h2 : used_water = 9 / 4) 
  (h3 : result_water = 3 / 4) : 
  initial_water - used_water = result_water := by
  sorry

end water_left_after_experiment_l58_58069


namespace total_amount_paid_l58_58852

variable (W : ℝ) (P_refrigerator : ℝ) (P_oven : ℝ)

/-- Conditions -/
variable (h1 : P_refrigerator = 3 * W)
variable (h2 : P_oven = 500)
variable (h3 : 2 * W = 500)

/-- Statement to be proved -/
theorem total_amount_paid :
  W + P_refrigerator + P_oven = 1500 :=
sorry

end total_amount_paid_l58_58852


namespace range_of_a_l58_58985

theorem range_of_a (a : ℝ) : 
  (∀ x, (x ≤ 1 ∨ x ≥ 3) ↔ ((a ≤ x ∧ x ≤ a + 1) → (x ≤ 1 ∨ x ≥ 3))) → 
  (a ≤ 0 ∨ a ≥ 3) :=
by
  sorry

end range_of_a_l58_58985


namespace cody_tickets_l58_58243

theorem cody_tickets (initial_tickets spent_tickets won_tickets : ℕ) (h_initial : initial_tickets = 49) (h_spent : spent_tickets = 25) (h_won : won_tickets = 6) : initial_tickets - spent_tickets + won_tickets = 30 := 
by 
  sorry

end cody_tickets_l58_58243


namespace machine_no_repair_l58_58036

def nominal_mass (G_dev: ℝ) := G_dev / 0.1

theorem machine_no_repair (G_dev: ℝ) (σ: ℝ) (non_readable_dev_lt: ∀ (x: ℝ), x ∉ { measurements } → |x - nominal_mass(G_dev)| < G_dev) : 
  (G_dev = 39) ∧ (σ ≤ G_dev) ∧ (∀ (x: ℝ), x ∉ { measurements } → |x - nominal_mass(G_dev)| < G_dev) ∧ (G_dev ≤ 0.1 * nominal_mass(G_dev)) → 
  ¬ machine.requires_repair :=
by
  sorry

end machine_no_repair_l58_58036


namespace positive_diff_solutions_abs_eq_12_l58_58500

theorem positive_diff_solutions_abs_eq_12 : 
  ∀ (x1 x2 : ℤ), (|x1 - 4| = 12) ∧ (|x2 - 4| = 12) ∧ (x1 > x2) → (x1 - x2 = 24) :=
by
  sorry

end positive_diff_solutions_abs_eq_12_l58_58500


namespace employee_total_correct_l58_58001

variable (total_employees : ℝ)
variable (percentage_female : ℝ)
variable (percentage_male_literate : ℝ)
variable (percentage_total_literate : ℝ)
variable (number_female_literate : ℝ)
variable (percentage_male : ℝ := 1 - percentage_female)

variables (E : ℝ) (CF : ℝ) (M : ℝ) (total_literate : ℝ)

theorem employee_total_correct :
  percentage_female = 0.60 ∧
  percentage_male_literate = 0.50 ∧
  percentage_total_literate = 0.62 ∧
  number_female_literate = 546 ∧
  (total_employees = 1300) :=
by
  -- Change these variables according to the context or find a way to prove this
  let total_employees := 1300
  have Cf := number_female_literate / (percentage_female * total_employees)
  have total_male := percentage_male * total_employees
  have male_literate := percentage_male_literate * total_male
  have total_literate := percentage_total_literate * total_employees

  -- We replace "proof statements" with sorry here
  sorry

end employee_total_correct_l58_58001


namespace count_congruent_to_5_mod_7_l58_58109

theorem count_congruent_to_5_mod_7 (n : ℕ) :
  (∀ x : ℕ, 1 ≤ x ∧ x ≤ 300 ∧ x % 7 = 5) → ∃ count : ℕ, count = 43 := by
  sorry

end count_congruent_to_5_mod_7_l58_58109


namespace cylindrical_coords_of_point_l58_58641

theorem cylindrical_coords_of_point :
  ∃ (r θ z : ℝ), r > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi ∧
                 r = Real.sqrt (3^2 + 3^2) ∧
                 θ = Real.arctan (3 / 3) ∧
                 z = 4 ∧
                 (3, 3, 4) = (r * Real.cos θ, r * Real.sin θ, z) :=
by
  sorry

end cylindrical_coords_of_point_l58_58641


namespace train_speed_l58_58768

theorem train_speed :
  let train_length := 200 -- in meters
  let platform_length := 175.03 -- in meters
  let time_taken := 25 -- in seconds
  let total_distance := train_length + platform_length -- total distance in meters
  let speed_mps := total_distance / time_taken -- speed in meters per second
  let speed_kmph := speed_mps * 3.6 -- converting speed to kilometers per hour
  speed_kmph = 54.00432 := sorry

end train_speed_l58_58768


namespace maximum_rubles_received_l58_58706

def four_digit_number_of_form_20xx (n : ℕ) : Prop :=
  2000 ≤ n ∧ n < 2100

def divisible_by (d : ℕ) (n : ℕ) : Prop :=
  n % d = 0

theorem maximum_rubles_received :
  ∃ (n : ℕ), four_digit_number_of_form_20xx n ∧
             divisible_by 1 n ∧
             divisible_by 3 n ∧
             divisible_by 7 n ∧
             divisible_by 9 n ∧
             divisible_by 11 n ∧
             ¬ divisible_by 5 n ∧
             1 + 3 + 7 + 9 + 11 = 31 :=
sorry

end maximum_rubles_received_l58_58706


namespace customer_purchases_90_percent_l58_58892

variable (P Q : ℝ) 

theorem customer_purchases_90_percent (price_increase_expenditure_diff : 
  (1.25 * P * R / 100 * Q = 1.125 * P * Q)) : 
  R = 90 := 
by 
  sorry

end customer_purchases_90_percent_l58_58892


namespace number_of_bugs_l58_58415

def flowers_per_bug := 2
def total_flowers_eaten := 6

theorem number_of_bugs : total_flowers_eaten / flowers_per_bug = 3 := 
by sorry

end number_of_bugs_l58_58415


namespace zach_needs_more_money_l58_58049

theorem zach_needs_more_money
  (bike_cost : ℕ) (allowance : ℕ) (mowing_payment : ℕ) (babysitting_rate : ℕ) 
  (current_savings : ℕ) (babysitting_hours : ℕ) :
  bike_cost = 100 →
  allowance = 5 →
  mowing_payment = 10 →
  babysitting_rate = 7 →
  current_savings = 65 →
  babysitting_hours = 2 →
  (bike_cost - (current_savings + (allowance + mowing_payment + babysitting_hours * babysitting_rate))) = 6 :=
by
  sorry

end zach_needs_more_money_l58_58049


namespace smallest_perfect_square_greater_l58_58972

theorem smallest_perfect_square_greater (a : ℕ) (h : ∃ n : ℕ, a = n^2) : 
  ∃ m : ℕ, m^2 > a ∧ ∀ k : ℕ, k^2 > a → m^2 ≤ k^2 :=
  sorry

end smallest_perfect_square_greater_l58_58972


namespace sufficient_necessary_condition_l58_58066

theorem sufficient_necessary_condition (a : ℝ) :
  (∃ x : ℝ, 2 * x + 1 = a ∧ x > 2) ↔ a > 5 :=
by
  sorry

end sufficient_necessary_condition_l58_58066


namespace treadmill_discount_percentage_l58_58020

theorem treadmill_discount_percentage
  (p_t : ℝ) -- original price of the treadmill
  (t_p : ℝ) -- total amount paid for treadmill and plates
  (p_plate : ℝ) -- price of each plate
  (n_plate : ℕ) -- number of plates
  (h_t : p_t = 1350)
  (h_tp : t_p = 1045)
  (h_p_plate : p_plate = 50)
  (h_n_plate : n_plate = 2) :
  ((p_t - (t_p - n_plate * p_plate)) / p_t) * 100 = 30 :=
by
  sorry

end treadmill_discount_percentage_l58_58020


namespace pyramid_value_l58_58233

theorem pyramid_value (a b c d e f : ℕ) (h_b : b = 6) (h_d : d = 20) (h_prod1 : d = b * (20 / b)) (h_prod2 : e = (20 / b) * c) (h_prod3 : f = c * (72 / c)) : a = b * c → a = 54 :=
by 
  -- Assuming the proof would assert the calculations done in the solution.
  sorry

end pyramid_value_l58_58233


namespace below_sea_level_representation_l58_58820

def depth_below_sea_level (depth: Int → Prop) :=
  ∀ x, depth x → x < 0

theorem below_sea_level_representation (D: Int) (A: Int) 
  (hA: A = 9050) (hD: D = 10907) 
  (above_sea_level: Int → Prop) (below_sea_level: Int → Prop)
  (h1: ∀ y, above_sea_level y → y > 0):
  below_sea_level (-D) :=
by
  sorry

end below_sea_level_representation_l58_58820


namespace tyler_meals_l58_58463

def num_meals : ℕ := 
  let num_meats := 3
  let num_vegetable_combinations := Nat.choose 5 3
  let num_desserts := 5
  num_meats * num_vegetable_combinations * num_desserts

theorem tyler_meals :
  num_meals = 150 := by
  sorry

end tyler_meals_l58_58463


namespace Creekview_science_fair_l58_58642

/-- Given the total number of students at Creekview High School is 1500,
    900 of these students participate in a science fair, where three-quarters
    of the girls participate and two-thirds of the boys participate,
    prove that 900 girls participate in the science fair. -/
theorem Creekview_science_fair
  (g b : ℕ)
  (h1 : g + b = 1500)
  (h2 : (3 / 4) * g + (2 / 3) * b = 900) :
  (3 / 4) * g = 900 := by
sorry

end Creekview_science_fair_l58_58642


namespace y_work_days_24_l58_58055

-- Definitions of the conditions
def x_work_days := 36
def y_work_days (d : ℕ) := d
def y_worked_days := 12
def x_remaining_work_days := 18

-- Statement of the theorem
theorem y_work_days_24 : ∃ d : ℕ, (y_worked_days / y_work_days d + x_remaining_work_days / x_work_days = 1) ∧ d = 24 :=
  sorry

end y_work_days_24_l58_58055


namespace geometric_sequence_l58_58314

theorem geometric_sequence (a : ℝ) (h1 : a > 0)
  (h2 : ∃ r : ℝ, 210 * r = a ∧ a * r = 63 / 40) :
  a = 18.1875 :=
by
  sorry

end geometric_sequence_l58_58314


namespace problem_statement_l58_58105

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x - Real.log x

theorem problem_statement (a b : ℝ) (h1 : 0 < a) (h2 : ∀ x, 0 < x → f a b x ≥ f a b 1) : 
  Real.log a < -2 * b :=
by
  sorry

end problem_statement_l58_58105


namespace quadratic_expression_value_l58_58956

variables (α β : ℝ)
noncomputable def quadratic_root_sum (α β : ℝ) (h1 : α^2 + 2*α - 1 = 0) (h2 : β^2 + 2*β - 1 = 0) : Prop :=
  α + β = -2

theorem quadratic_expression_value (α β : ℝ) (h1 : α^2 + 2*α - 1 = 0) (h2 : β^2 + 2*β - 1 = 0) (h3 : α + β = -2) :
  α^2 + 3*α + β = -1 :=
sorry

end quadratic_expression_value_l58_58956


namespace no_base_131_cubed_l58_58272

open Nat

theorem no_base_131_cubed (n : ℕ) (k : ℕ) : 
  (4 ≤ n ∧ n ≤ 12) ∧ (1 * n^2 + 3 * n + 1 = k^3) → False := by
  sorry

end no_base_131_cubed_l58_58272


namespace negation_of_proposition_l58_58579

theorem negation_of_proposition :
  ¬(∀ n : ℤ, (∃ k : ℤ, n = 2 * k) → (∃ m : ℤ, n = 2 * m)) ↔ ∃ n : ℤ, (∃ k : ℤ, n = 2 * k) ∧ ¬(∃ m : ℤ, n = 2 * m) := 
sorry

end negation_of_proposition_l58_58579


namespace smallest_power_of_13_non_palindrome_l58_58654

-- Define a function to check if a number is a palindrome
def is_palindrome (n : ℕ) : Bool :=
  let s := n.to_string
  s = s.reverse

-- Define the main theorem stating the smallest power of 13 that is not a palindrome
theorem smallest_power_of_13_non_palindrome (n : ℕ) (hn : ∀ k : ℕ, 0 < k → 13^k = n → k = 2) : n = 169 :=
by
  have h1 : is_palindrome (13^1) = true := by sorry
  have h2 : is_palindrome (13^2) = false := by sorry
  have hm : ∀ m : ℕ, 0 < m → m < 2 → is_palindrome (13^m) = true := by sorry
  sorry

end smallest_power_of_13_non_palindrome_l58_58654


namespace not_must_be_even_number_of_even_scores_l58_58063

-- Define the conditions of the problem
def round_robin_tournament (teams games : ℕ) (scores : Fin teams → ℕ) : Prop :=
  teams = 14 ∧
  games = 91 ∧
  (∀ (i j : Fin teams), i ≠ j → scores i + scores j = 3 ∨ scores i + scores j = 4)

-- The problem we need to prove
theorem not_must_be_even_number_of_even_scores (scores : Fin 14 → ℕ) :
  round_robin_tournament 14 91 scores →
  ¬(∃ S, S = {n : ℕ | ∃ i, scores i = n ∧ n % 2 = 0} ∧ S.card % 2 = 0) := 
by {
  intro h,
  sorry
}

end not_must_be_even_number_of_even_scores_l58_58063


namespace cost_of_20_pounds_of_bananas_l58_58775

noncomputable def cost_of_bananas (rate : ℝ) (amount : ℝ) : ℝ :=
rate * amount / 4

theorem cost_of_20_pounds_of_bananas :
  cost_of_bananas 6 20 = 30 :=
by
  sorry

end cost_of_20_pounds_of_bananas_l58_58775


namespace customers_added_during_lunch_rush_l58_58931

noncomputable def initial_customers := 29.0
noncomputable def total_customers_after_lunch_rush := 83.0
noncomputable def expected_customers_added := 54.0

theorem customers_added_during_lunch_rush :
  (total_customers_after_lunch_rush - initial_customers) = expected_customers_added :=
by
  sorry

end customers_added_during_lunch_rush_l58_58931


namespace find_a_b_c_l58_58883

theorem find_a_b_c :
  ∃ a b c : ℕ, a = 1 ∧ b = 17 ∧ c = 2 ∧ (Nat.gcd a c = 1) ∧ a + b + c = 20 :=
by {
  -- the proof would go here
  sorry
}

end find_a_b_c_l58_58883


namespace janet_spends_more_on_piano_l58_58290

-- Condition definitions
def clarinet_hourly_rate : ℝ := 40
def clarinet_hours_per_week : ℝ := 3
def piano_hourly_rate : ℝ := 28
def piano_hours_per_week : ℝ := 5
def weeks_per_year : ℝ := 52

-- Calculations based on conditions
def weekly_cost_clarinet : ℝ := clarinet_hourly_rate * clarinet_hours_per_week
def weekly_cost_piano : ℝ := piano_hourly_rate * piano_hours_per_week
def weekly_difference : ℝ := weekly_cost_piano - weekly_cost_clarinet
def yearly_difference : ℝ := weekly_difference * weeks_per_year

theorem janet_spends_more_on_piano : yearly_difference = 1040 := by
  sorry 

end janet_spends_more_on_piano_l58_58290


namespace maximum_rubles_received_max_payment_possible_l58_58700

def is_four_digit (n : ℕ) : Prop :=
  2000 ≤ n ∧ n < 2100

def divisible_by (n d : ℕ) : Prop :=
  d ∣ n

def payment (n : ℕ) : ℕ :=
  if divisible_by n 1 then 1 else 0 +
  (if divisible_by n 3 then 3 else 0) +
  (if divisible_by n 5 then 5 else 0) +
  (if divisible_by n 7 then 7 else 0) +
  (if divisible_by n 9 then 9 else 0) +
  (if divisible_by n 11 then 11 else 0)

theorem maximum_rubles_received :
  ∀ (n : ℕ), is_four_digit n → payment n ≤ 31 :=
sorry

theorem max_payment_possible :
  ∃ (n : ℕ), is_four_digit n ∧ payment n = 31 :=
sorry

end maximum_rubles_received_max_payment_possible_l58_58700


namespace amelia_jet_bars_l58_58240

theorem amelia_jet_bars
    (required : ℕ) (sold_monday : ℕ) (sold_tuesday_less : ℕ) (total_sold : ℕ) (remaining : ℕ) :
    required = 90 →
    sold_monday = 45 →
    sold_tuesday_less = 16 →
    total_sold = sold_monday + (sold_monday - sold_tuesday_less) →
    remaining = required - total_sold →
    remaining = 16 :=
by
  intros
  sorry

end amelia_jet_bars_l58_58240


namespace probability_of_popped_white_is_12_over_17_l58_58122

noncomputable def probability_white_given_popped (white_kernels yellow_kernels : ℚ) (pop_white pop_yellow : ℚ) : ℚ :=
  let p_white_popped := white_kernels * pop_white
  let p_yellow_popped := yellow_kernels * pop_yellow
  let p_popped := p_white_popped + p_yellow_popped
  p_white_popped / p_popped

theorem probability_of_popped_white_is_12_over_17 :
  probability_white_given_popped (3/4) (1/4) (3/5) (3/4) = 12/17 :=
by
  sorry

end probability_of_popped_white_is_12_over_17_l58_58122


namespace value_of_m_l58_58532

theorem value_of_m (m : ℝ) (h₁ : m^2 - 9 * m + 19 = 1) (h₂ : 2 * m^2 - 7 * m - 9 ≤ 0) : m = 3 :=
sorry

end value_of_m_l58_58532


namespace prime_divisors_of_n_congruent_to_1_mod_4_l58_58471

theorem prime_divisors_of_n_congruent_to_1_mod_4
  (x y n : ℕ)
  (hx : x ≥ 3)
  (hn : n ≥ 2)
  (h_eq : x^2 + 5 = y^n) :
  ∀ p : ℕ, Prime p → p ∣ n → p ≡ 1 [MOD 4] :=
by
  sorry

end prime_divisors_of_n_congruent_to_1_mod_4_l58_58471


namespace exists_real_number_lt_neg_one_l58_58048

theorem exists_real_number_lt_neg_one : ∃ (x : ℝ), x < -1 := by
  sorry

end exists_real_number_lt_neg_one_l58_58048


namespace total_surface_area_of_prism_l58_58631

-- Define the conditions of the problem
def sphere_radius (R : ℝ) := R > 0
def prism_circumscribed_around_sphere (R : ℝ) := True  -- Placeholder as the concept assertion, actual geometry handling not needed here
def prism_height (R : ℝ) := 2 * R

-- Define the main theorem to be proved
theorem total_surface_area_of_prism (R : ℝ) (hR : sphere_radius R) (hCircumscribed : prism_circumscribed_around_sphere R) (hHeight : prism_height R = 2 * R) : 
  ∃ (S : ℝ), S = 12 * R^2 * Real.sqrt 3 :=
sorry

end total_surface_area_of_prism_l58_58631


namespace denote_depth_below_sea_level_l58_58836

theorem denote_depth_below_sea_level (above_sea_level : Int) (depth_haidou_1 : Int) :
  (above_sea_level > 0) ∧ (depth_haidou_1 < 0) → depth_haidou_1 = -10907 :=
by
  intros h
  sorry

end denote_depth_below_sea_level_l58_58836


namespace single_shot_percentage_decrease_l58_58024

theorem single_shot_percentage_decrease
  (initial_salary : ℝ)
  (final_salary : ℝ := initial_salary * 0.95 * 0.90 * 0.85) :
  ((1 - final_salary / initial_salary) * 100) = 27.325 := by
  sorry

end single_shot_percentage_decrease_l58_58024


namespace nathaniel_tickets_l58_58169

theorem nathaniel_tickets :
  ∀ (B S : ℕ),
  (7 * B + 4 * S + 11 = 128) →
  (B + S = 20) :=
by
  intros B S h
  sorry

end nathaniel_tickets_l58_58169


namespace num_students_in_research_study_group_prob_diff_classes_l58_58925

-- Define the number of students in each class and the number of students selected from class (2)
def num_students_class1 : ℕ := 18
def num_students_class2 : ℕ := 27
def selected_from_class2 : ℕ := 3

-- Prove the number of students in the research study group
theorem num_students_in_research_study_group : 
  (∃ (m : ℕ), (m / 18 = 3 / 27) ∧ (m + selected_from_class2 = 5)) := 
by
  sorry

-- Prove the probability that the students speaking in both activities come from different classes
theorem prob_diff_classes : 
  (12 / 25 = 12 / 25) :=
by
  sorry

end num_students_in_research_study_group_prob_diff_classes_l58_58925


namespace piglet_gifted_balloons_l58_58657

noncomputable def piglet_balloons_gifted (piglet_balloons : ℕ) : ℕ :=
  let winnie_balloons := 3 * piglet_balloons
  let owl_balloons := 4 * piglet_balloons
  let total_balloons := piglet_balloons + winnie_balloons + owl_balloons
  let burst_balloons := total_balloons - 60
  piglet_balloons - burst_balloons / 8

-- Prove that Piglet gifted 4 balloons given the conditions
theorem piglet_gifted_balloons :
  ∃ (piglet_balloons : ℕ), piglet_balloons = 8 ∧ piglet_balloons_gifted piglet_balloons = 4 := sorry

end piglet_gifted_balloons_l58_58657


namespace probability_of_even_distribution_l58_58399

noncomputable def probability_all_players_have_8_cards_after_dealing : ℝ :=
  let initial_cards := 48
  let players := 6
  let cards_per_player := initial_cards / players
  (5 / 6 : ℝ) -- We state the probability directly according to the problem solution.

theorem probability_of_even_distribution : 
  probability_all_players_have_8_cards_after_dealing = (5 / 6 : ℝ) :=
sorry

end probability_of_even_distribution_l58_58399


namespace max_heaps_660_stones_l58_58860

theorem max_heaps_660_stones :
  ∀ (heaps : List ℕ), (sum heaps = 660) → (∀ i j, i ≠ j → heaps[i] < 2 * heaps[j]) → heaps.length ≤ 30 :=
sorry

end max_heaps_660_stones_l58_58860


namespace find_min_max_l58_58260

noncomputable def f (x y : ℝ) : ℝ := Real.sin x + Real.sin y - Real.sin (x + y)

theorem find_min_max :
  (∀ (x y : ℝ), 0 ≤ x → 0 ≤ y → x + y ≤ 2 * Real.pi → 
    (0 ≤ f x y ∧ f x y ≤ 3 * Real.sqrt 3 / 2)) :=
sorry

end find_min_max_l58_58260


namespace werewolf_eats_per_week_l58_58238
-- First, we import the necessary libraries

-- We define the conditions using Lean definitions

-- The vampire drains 3 people a week
def vampire_drains_per_week : Nat := 3

-- The total population of the village
def village_population : Nat := 72

-- The number of weeks both can live off the population
def weeks : Nat := 9

-- Prove the number of people the werewolf eats per week (W) given the conditions
theorem werewolf_eats_per_week :
  ∃ W : Nat, vampire_drains_per_week * weeks + weeks * W = village_population ∧ W = 5 :=
by
  sorry

end werewolf_eats_per_week_l58_58238


namespace simplify_x_cubed_simplify_expr_l58_58938

theorem simplify_x_cubed (x : ℝ) : x * (x + 3) * (x + 5) = x^3 + 8 * x^2 + 15 * x := by
  sorry

theorem simplify_expr (x y : ℝ) : (5 * x + 2 * y) * (5 * x - 2 * y) - 5 * x * (5 * x - 3 * y) = -4 * y^2 + 15 * x * y := by
  sorry

end simplify_x_cubed_simplify_expr_l58_58938


namespace eight_mul_eleven_and_one_fourth_l58_58346

theorem eight_mul_eleven_and_one_fourth : 8 * (11 + (1 / 4 : ℝ)) = 90 := by
  sorry

end eight_mul_eleven_and_one_fourth_l58_58346


namespace rajas_salary_percentage_less_than_rams_l58_58868

-- Definitions from the problem conditions
def raja_salary : ℚ := sorry -- Placeholder, since Raja's salary doesn't need a fixed value
def ram_salary : ℚ := 1.25 * raja_salary

-- Theorem to be proved
theorem rajas_salary_percentage_less_than_rams :
  ∃ r : ℚ, (ram_salary - raja_salary) / ram_salary * 100 = 20 :=
by
  sorry

end rajas_salary_percentage_less_than_rams_l58_58868


namespace probability_six_distinct_one_repeat_l58_58046

def num_dice : ℕ := 7
def num_faces : ℕ := 6

noncomputable def total_outcomes : ℕ := num_faces ^ num_dice

noncomputable def repeated_num_choices : ℕ := num_faces
noncomputable def distinct_num_ways : ℕ := Nat.factorial num_faces.pred
noncomputable def dice_combinations : ℕ := Nat.choose num_dice 2

noncomputable def favorable_outcomes : ℕ := repeated_num_choices * distinct_num_ways * dice_combinations

noncomputable def probability : ℚ := favorable_outcomes / total_outcomes

theorem probability_six_distinct_one_repeat :
  probability = 5 / 186 := 
by
  sorry

end probability_six_distinct_one_repeat_l58_58046


namespace max_value_of_expression_l58_58153

theorem max_value_of_expression (A M C : ℕ) (h : A + M + C = 15) :
  A * M * C + A * M + M * C + C * A ≤ 200 :=
sorry

end max_value_of_expression_l58_58153


namespace volume_of_cube_l58_58721

theorem volume_of_cube (d : ℝ) (h : d = 5 * Real.sqrt 3) : ∃ (V : ℝ), V = 125 := by
  sorry

end volume_of_cube_l58_58721


namespace sum_of_coeffs_eq_neg30_l58_58507

noncomputable def expanded : Polynomial ℤ := 
  -(Polynomial.C 4 - Polynomial.X) * (Polynomial.X + 3 * (Polynomial.C 4 - Polynomial.X))

theorem sum_of_coeffs_eq_neg30 : (expanded.coeffs.sum) = -30 := 
  sorry

end sum_of_coeffs_eq_neg30_l58_58507


namespace parallelogram_coordinates_l58_58808

/-- Given points A, B, and C, prove the coordinates of point D for the parallelogram -/
theorem parallelogram_coordinates (A B C: (ℝ × ℝ)) 
  (hA : A = (3, 7)) 
  (hB : B = (4, 6))
  (hC : C = (1, -2)) :
  D = (0, -1) ∨ D = (2, -3) ∨ D = (6, 15) :=
sorry

end parallelogram_coordinates_l58_58808


namespace derivative_at_zero_l58_58275

def f (x : ℝ) : ℝ := (x + 1)^4

theorem derivative_at_zero : deriv f 0 = 4 :=
by
  sorry

end derivative_at_zero_l58_58275


namespace linear_function_increasing_and_composition_eq_implies_values_monotonic_gx_implies_m_range_l58_58261

-- Defining the first part of the problem
theorem linear_function_increasing_and_composition_eq_implies_values
  (a b : ℝ)
  (H1 : ∀ x y : ℝ, x < y → a * x + b < a * y + b)
  (H2 : ∀ x : ℝ, a * (a * x + b) + b = 16 * x + 5) :
  a = 4 ∧ b = 1 :=
by
  sorry

-- Defining the second part of the problem
theorem monotonic_gx_implies_m_range (m : ℝ)
  (H3 : ∀ x1 x2 : ℝ, 1 ≤ x1 → x1 < x2 → (x2 + m) * (4 * x2 + 1) > (x1 + m) * (4 * x1 + 1)) :
  -9 / 4 ≤ m :=
by
  sorry

end linear_function_increasing_and_composition_eq_implies_values_monotonic_gx_implies_m_range_l58_58261


namespace horner_method_value_v2_at_minus_one_l58_58498

noncomputable def f (x : ℝ) : ℝ :=
  x^6 - 5*x^5 + 6*x^4 - 3*x^3 + 1.8*x^2 + 0.35*x + 2

theorem horner_method_value_v2_at_minus_one :
  let a : ℝ := -1
  let v_0 := 1
  let v_1 := v_0 * a - 5
  let v_2 := v_1 * a + 6
  v_2 = 12 :=
by
  intros
  sorry

end horner_method_value_v2_at_minus_one_l58_58498


namespace smallest_non_palindrome_power_of_13_is_2197_l58_58651

-- Definition of palindrome
def is_palindrome (n : ℕ) : Prop :=
  let s := toString n
  s = s.reverse

-- Definition of power of 13
def power_of_13 (n : ℕ) : Prop :=
  ∃ k, n = 13^k

-- Theorem statement
theorem smallest_non_palindrome_power_of_13_is_2197 :
  ∀ n, power_of_13 n → ¬is_palindrome n → n ≥ 13 → n = 2197 :=
by
  sorry

end smallest_non_palindrome_power_of_13_is_2197_l58_58651


namespace find_x_ceil_mul_l58_58783

theorem find_x_ceil_mul (x : ℝ) (h : ⌈x⌉ * x = 75) : x = 8.333 := by
  sorry

end find_x_ceil_mul_l58_58783


namespace liquid_ratio_l58_58057

-- Defining the initial state of the container.
def initial_volume : ℝ := 37.5
def removed_and_replaced_volume : ℝ := 15

-- Defining the process steps.
def fraction_remaining (total: ℝ) (removed: ℝ) : ℝ := (total - removed) / total
def final_volume_A (initial: ℝ) (removed: ℝ) : ℝ := (fraction_remaining initial removed)^2 * initial

-- The given problem and its conclusion as a theorem.
theorem liquid_ratio (initial_V : ℝ) (remove_replace_V : ℝ) 
  (h1 : initial_V = 37.5) (h2 : remove_replace_V = 15) :
  let final_A := final_volume_A initial_V remove_replace_V in
  let final_B := initial_V - final_A in
  final_A / final_B = 9 / 16 :=
by
  sorry

end liquid_ratio_l58_58057


namespace Nikolai_faster_than_Gennady_l58_58741

theorem Nikolai_faster_than_Gennady
  (gennady_jump1 gennady_jump2 : ℕ) (nikolai_jump1 nikolai_jump2 nikolai_jump3 : ℕ) :
  gennady_jump1 = 6 → gennady_jump2 = 6 →
  nikolai_jump1 = 4 → nikolai_jump2 = 4 → nikolai_jump3 = 4 →
  2 * gennady_jump1 + gennady_jump2 = 3 * (nikolai_jump1 + nikolai_jump2 + nikolai_jump3) →
  let total_path := 2000 in
  (total_path % 4 = 0 ∧ total_path % 6 ≠ 0) →
  (total_path / 4 < (total_path + 4) / 6) :=
by
  intros
  sorry

end Nikolai_faster_than_Gennady_l58_58741


namespace gem_stone_necklaces_sold_l58_58413

-- Definitions and conditions
def bead_necklaces : ℕ := 7
def total_earnings : ℝ := 90
def price_per_necklace : ℝ := 9

-- Theorem to prove the number of gem stone necklaces sold
theorem gem_stone_necklaces_sold : 
  ∃ (G : ℕ), G * price_per_necklace = total_earnings - (bead_necklaces * price_per_necklace) ∧ G = 3 :=
by
  sorry

end gem_stone_necklaces_sold_l58_58413


namespace ab_cd_eq_neg190_over_9_l58_58967

theorem ab_cd_eq_neg190_over_9 (a b c d : ℝ)
  (h1 : a + b + c = 3)
  (h2 : a + b + d = -2)
  (h3 : a + c + d = 8)
  (h4 : b + c + d = -1) :
  a * b + c * d = -190 / 9 :=
by
  sorry

end ab_cd_eq_neg190_over_9_l58_58967


namespace clubs_popularity_order_l58_58586

theorem clubs_popularity_order (chess drama art science : ℚ)
  (h_chess: chess = 14/35) (h_drama: drama = 9/28) (h_art: art = 11/21) (h_science: science = 8/15) :
  science > art ∧ art > chess ∧ chess > drama :=
by {
  -- Place proof steps here (optional)
  sorry
}

end clubs_popularity_order_l58_58586


namespace inequality_does_not_hold_l58_58966

theorem inequality_does_not_hold (a b : ℝ) (h₁ : a < b) (h₂ : b < 0) :
  ¬ (1 / (a - 1) < 1 / b) :=
by
  sorry

end inequality_does_not_hold_l58_58966


namespace starting_number_of_range_l58_58458

theorem starting_number_of_range (multiples: ℕ) (end_of_range: ℕ) (span: ℕ)
  (h1: multiples = 991) (h2: end_of_range = 10000) (h3: span = multiples * 10) :
  end_of_range - span = 90 := 
by 
  sorry

end starting_number_of_range_l58_58458


namespace average_score_l58_58003

theorem average_score (s1 s2 s3 : ℕ) (n : ℕ) (h1 : s1 = 115) (h2 : s2 = 118) (h3 : s3 = 115) (h4 : n = 3) :
    (s1 + s2 + s3) / n = 116 :=
by
    sorry

end average_score_l58_58003


namespace petya_max_rubles_l58_58699

theorem petya_max_rubles :
  let is_valid_number := (λ n : ℕ, 2000 ≤ n ∧ n < 2100),
      is_divisible := (λ n d : ℕ, d ∣ n),
      rubles := (λ n, if is_divisible n 1 then 1 else 0 +
                       if is_divisible n 3 then 3 else 0 +
                       if is_divisible n 5 then 5 else 0 +
                       if is_divisible n 7 then 7 else 0 +
                       if is_divisible n 9 then 9 else 0 +
                       if is_divisible n 11 then 11 else 0)
  in ∃ n, is_valid_number n ∧ rubles n = 31 := 
sorry

end petya_max_rubles_l58_58699


namespace sine_central_angle_of_circular_sector_eq_4_5_l58_58606

theorem sine_central_angle_of_circular_sector_eq_4_5
  (R : Real)
  (α : Real)
  (h : π * R ^ 2 * Real.sin α = 2 * π * R ^ 2 * (1 - Real.cos α)) :
  Real.sin α = 4 / 5 := by
  sorry

end sine_central_angle_of_circular_sector_eq_4_5_l58_58606


namespace greatest_length_measures_exactly_l58_58222

theorem greatest_length_measures_exactly 
    (a b c : ℕ) 
    (ha : a = 700)
    (hb : b = 385)
    (hc : c = 1295) : 
    Nat.gcd (Nat.gcd a b) c = 35 := 
by
  sorry

end greatest_length_measures_exactly_l58_58222


namespace intersection_of_M_and_N_l58_58412

noncomputable def M : Set ℕ := { x | 1 < x ∧ x < 7 }
noncomputable def N : Set ℕ := { x | x % 3 ≠ 0 }

theorem intersection_of_M_and_N :
  M ∩ N = {2, 4, 5} := sorry

end intersection_of_M_and_N_l58_58412


namespace number_halfway_between_l58_58515

theorem number_halfway_between :
  ∃ x : ℚ, x = (1/12 + 1/14) / 2 ∧ x = 13 / 168 :=
sorry

end number_halfway_between_l58_58515


namespace find_EF_squared_l58_58426

noncomputable def square_side := 15
noncomputable def BE := 6
noncomputable def DF := 6
noncomputable def AE := 14
noncomputable def CF := 14

theorem find_EF_squared (A B C D E F : ℝ) (AB BC CD DA : ℝ := square_side) :
  (BE = 6) → (DF = 6) → (AE = 14) → (CF = 14) → EF^2 = 72 :=
by
  -- Definitions and conditions usage according to (a)
  sorry

end find_EF_squared_l58_58426


namespace upper_limit_of_multiples_of_10_l58_58320

theorem upper_limit_of_multiples_of_10 (n : ℕ) (hn : 10 * n = 100) (havg : (10 * n + 10) / (n + 1) = 55) : 10 * n = 100 :=
by
  sorry

end upper_limit_of_multiples_of_10_l58_58320


namespace find_y_l58_58560

def binary_op (a b c d : Int) : Int × Int := (a + d, b - c)

theorem find_y : ∃ y : Int, (binary_op 3 y 2 0) = (3, 4) ↔ y = 6 := by
  sorry

end find_y_l58_58560


namespace total_annual_cost_l58_58735

def daily_pills : ℕ := 2
def pill_cost : ℕ := 5
def medication_cost (daily_pills : ℕ) (pill_cost : ℕ) : ℕ := daily_pills * pill_cost
def insurance_coverage : ℚ := 0.80
def visit_cost : ℕ := 400
def visits_per_year : ℕ := 2
def annual_medication_cost (medication_cost : ℕ) (insurance_coverage : ℚ) : ℚ :=
  medication_cost * 365 * (1 - insurance_coverage)
def annual_visit_cost (visit_cost : ℕ) (visits_per_year : ℕ) : ℕ :=
  visit_cost * visits_per_year

theorem total_annual_cost : annual_medication_cost (medication_cost daily_pills pill_cost) insurance_coverage
  + annual_visit_cost visit_cost visits_per_year = 1530 := by
  sorry

end total_annual_cost_l58_58735


namespace solve_fraction_l58_58518

theorem solve_fraction (a b : ℝ) (hab : 3 * a = 2 * b) : (a + b) / b = 5 / 3 :=
by
  sorry

end solve_fraction_l58_58518


namespace hot_dogs_left_over_l58_58276

theorem hot_dogs_left_over : 25197629 % 6 = 5 := 
sorry

end hot_dogs_left_over_l58_58276


namespace complex_number_solution_l58_58648

theorem complex_number_solution : 
  ∃ (z : ℂ), (|z - 2| = |z + 4| ∧ |z + 4| = |z - 2i|) ∧ z = -1 + ⅈ :=
by
  sorry

end complex_number_solution_l58_58648


namespace anna_apple_ratio_l58_58491

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

end anna_apple_ratio_l58_58491


namespace tom_new_collection_l58_58461

theorem tom_new_collection (initial_stamps mike_gift : ℕ) (harry_gift : ℕ := 2 * mike_gift + 10) (sarah_gift : ℕ := 3 * mike_gift - 5) (total_gifts : ℕ := mike_gift + harry_gift + sarah_gift) (new_collection : ℕ := initial_stamps + total_gifts) 
  (h_initial_stamps : initial_stamps = 3000) (h_mike_gift : mike_gift = 17) :
  new_collection = 3107 := by
  sorry

end tom_new_collection_l58_58461


namespace bus_speed_including_stoppages_l58_58782

-- Definitions based on conditions
def speed_excluding_stoppages : ℝ := 50 -- kmph
def stoppage_time_per_hour : ℝ := 18 -- minutes

-- Lean statement of the problem
theorem bus_speed_including_stoppages :
  (speed_excluding_stoppages * (1 - stoppage_time_per_hour / 60)) = 35 := by
  sorry

end bus_speed_including_stoppages_l58_58782


namespace max_value_of_expression_l58_58152

theorem max_value_of_expression (A M C : ℕ) (h : A + M + C = 15) :
  A * M * C + A * M + M * C + C * A ≤ 200 :=
sorry

end max_value_of_expression_l58_58152


namespace smallest_angle_of_triangle_l58_58195

noncomputable def smallest_angle (a b : ℝ) (c : ℝ) (h_sum : a + b + c = 180) : ℝ :=
  min a (min b c)

theorem smallest_angle_of_triangle :
  smallest_angle 60 65 (180 - (60 + 65)) (by norm_num) = 55 :=
by
  -- The correct proof steps should be provided for the result
  sorry

end smallest_angle_of_triangle_l58_58195


namespace average_weight_increase_l58_58307

theorem average_weight_increase (A : ℝ) :
  let initial_weight := 8 * A
  let new_weight := initial_weight - 65 + 89
  let new_average := new_weight / 8
  let increase := new_average - A
  increase = (89 - 65) / 8 := 
by 
  sorry

end average_weight_increase_l58_58307


namespace factorization_correct_l58_58885

theorem factorization_correct (C D : ℤ) (h : 15 = C * D ∧ 48 = 8 * 6 ∧ -56 = -8 * D - 6 * C):
  C * D + C = 18 :=
  sorry

end factorization_correct_l58_58885


namespace correct_statements_l58_58936

namespace ProofProblem

def P1 : Prop := (-4) + (-5) = -9
def P2 : Prop := -5 - (-6) = 11
def P3 : Prop := -2 * (-10) = -20
def P4 : Prop := 4 / (-2) = -2

theorem correct_statements : P1 ∧ P4 ∧ ¬P2 ∧ ¬P3 := by
  -- proof to be filled in later
  sorry

end ProofProblem

end correct_statements_l58_58936


namespace jimmy_eats_7_cookies_l58_58249

def cookies_and_calories (c: ℕ) : Prop :=
  50 * c + 150 = 500

theorem jimmy_eats_7_cookies : cookies_and_calories 7 :=
by {
  -- This would be where the proof steps go, but we replace it with:
  sorry
}

end jimmy_eats_7_cookies_l58_58249


namespace calculate_annual_cost_l58_58734

noncomputable def doctor_visit_cost (visits_per_year: ℕ) (cost_per_visit: ℕ) : ℕ :=
  visits_per_year * cost_per_visit

noncomputable def medication_night_cost (pills_per_night: ℕ) (cost_per_pill: ℕ) : ℕ :=
  pills_per_night * cost_per_pill

noncomputable def insurance_coverage (medication_cost: ℕ) (coverage_percent: ℕ) : ℕ :=
  medication_cost * coverage_percent / 100

noncomputable def out_of_pocket_cost (total_cost: ℕ) (insurance_covered: ℕ) : ℕ :=
  total_cost - insurance_covered

noncomputable def annual_medication_cost (night_cost: ℕ) (nights_per_year: ℕ) : ℕ :=
  night_cost * nights_per_year

noncomputable def total_annual_cost (doctor_visit_total: ℕ) (medication_total: ℕ) : ℕ :=
  doctor_visit_total + medication_total

theorem calculate_annual_cost :
  let visits_per_year := 2 in
  let cost_per_visit := 400 in
  let pills_per_night := 2 in
  let cost_per_pill := 5 in
  let coverage_percent := 80 in
  let nights_per_year := 365 in
  let total_cost := total_annual_cost 
    (doctor_visit_cost visits_per_year cost_per_visit)
    (annual_medication_cost 
      (out_of_pocket_cost 
        (medication_night_cost pills_per_night cost_per_pill) 
        (insurance_coverage (medication_night_cost pills_per_night cost_per_pill) coverage_percent)
      ) 
      nights_per_year
    ) in
  total_cost = 1530 :=
begin
  sorry
end

end calculate_annual_cost_l58_58734


namespace find_angle_A_l58_58279

open Real

theorem find_angle_A (a b : ℝ) (B A : ℝ) 
  (ha : a = sqrt 2) 
  (hb : b = 2) 
  (hB : sin B + cos B = sqrt 2) :
  A = π / 6 := 
  sorry

end find_angle_A_l58_58279


namespace remainder_2345678901_div_101_l58_58596

theorem remainder_2345678901_div_101 : 2345678901 % 101 = 12 :=
sorry

end remainder_2345678901_div_101_l58_58596


namespace arbitrarily_large_ratios_l58_58296

open Nat

theorem arbitrarily_large_ratios (a : ℕ → ℕ) (h_distinct: ∀ m n, m ≠ n → a m ≠ a n)
  (h_no_100_ones: ∀ n, ¬ (∃ k, a n / 10^k % 10^100 = 10^100 - 1)):
  ∀ M : ℕ, ∃ n : ℕ, a n / n ≥ M :=
by
  sorry

end arbitrarily_large_ratios_l58_58296


namespace largest_value_among_given_numbers_l58_58607

theorem largest_value_among_given_numbers :
  let a := 2 * 0 * 2006
  let b := 2 * 0 + 6
  let c := 2 + 0 * 2006
  let d := 2 * (0 + 6)
  let e := 2006 * 0 + 0 * 6
  d >= a ∧ d >= b ∧ d >= c ∧ d >= e :=
by
  let a := 2 * 0 * 2006
  let b := 2 * 0 + 6
  let c := 2 + 0 * 2006
  let d := 2 * (0 + 6)
  let e := 2006 * 0 + 0 * 6
  sorry

end largest_value_among_given_numbers_l58_58607


namespace blake_change_l58_58633

-- Definitions based on conditions
def n_l : ℕ := 4
def n_c : ℕ := 6
def p_l : ℕ := 2
def p_c : ℕ := 4 * p_l
def amount_given : ℕ := 6 * 10

-- Total cost calculations derived from the conditions
def total_cost_lollipops : ℕ := n_l * p_l
def total_cost_chocolates : ℕ := n_c * p_c
def total_cost : ℕ := total_cost_lollipops + total_cost_chocolates

-- Calculating the change
def change : ℕ := amount_given - total_cost

-- Theorem stating the final answer
theorem blake_change : change = 4 := sorry

end blake_change_l58_58633


namespace perimeter_regular_polygon_l58_58487

-- Condition definitions
def is_regular_polygon (n : ℕ) (s : ℝ) : Prop := 
  n * s > 0

def exterior_angle (E : ℝ) (n : ℕ) : Prop := 
  E = 360 / n

def side_length (s : ℝ) : Prop :=
  s = 6

-- Theorem statement to prove the perimeter is 24 units
theorem perimeter_regular_polygon 
  (n : ℕ) (s E : ℝ)
  (h1 : is_regular_polygon n s)
  (h2 : exterior_angle E n)
  (h3 : side_length s)
  (h4 : E = 90) :
  4 * s = 24 :=
by
  sorry

end perimeter_regular_polygon_l58_58487


namespace dodecahedron_edge_coloring_l58_58749

-- Define the properties of the dodecahedron
structure Dodecahedron :=
  (faces : Fin 12)          -- 12 pentagonal faces
  (edges : Fin 30)         -- 30 edges
  (vertices : Fin 20)      -- 20 vertices
  (edge_faces : Fin 30 → Fin 2) -- Each edge contributes to two faces

-- Prove the number of valid edge colorations such that each face has an even number of red edges
theorem dodecahedron_edge_coloring : 
    (∃ num_colorings : ℕ, num_colorings = 2^11) :=
sorry

end dodecahedron_edge_coloring_l58_58749


namespace simplify_an_over_bn_l58_58946

noncomputable def a_n (n : ℕ) : ℚ :=
∑ k in Finset.range (n + 1), 1 / (Nat.choose n k)

noncomputable def b_n (n : ℕ) : ℚ :=
∑ k in Finset.range (n + 1), k / (Nat.choose n k)

theorem simplify_an_over_bn (n : ℕ) (h_pos : 0 < n) : (a_n n) / (b_n n) = 2 / n := by
  sorry

end simplify_an_over_bn_l58_58946


namespace distance_of_coming_down_stairs_l58_58386

noncomputable def totalTimeAscendingDescending (D : ℝ) : ℝ :=
  (D / 2) + ((D + 2) / 3)

theorem distance_of_coming_down_stairs : ∃ D : ℝ, totalTimeAscendingDescending D = 4 ∧ (D + 2) = 6 :=
by
  sorry

end distance_of_coming_down_stairs_l58_58386


namespace transformed_parabola_equation_l58_58719

theorem transformed_parabola_equation :
    ∀ (x : ℝ), let f := λ x, -2 * x ^ 2 in
    (f (x + 1) - 3) = -2 * x ^ 2 - 4 * x - 5 :=
by
  intro x
  let f := λ x, -2 * x ^ 2
  sorry

end transformed_parabola_equation_l58_58719


namespace zach_needs_more_money_l58_58050

theorem zach_needs_more_money
  (bike_cost : ℕ) (allowance : ℕ) (mowing_payment : ℕ) (babysitting_rate : ℕ) 
  (current_savings : ℕ) (babysitting_hours : ℕ) :
  bike_cost = 100 →
  allowance = 5 →
  mowing_payment = 10 →
  babysitting_rate = 7 →
  current_savings = 65 →
  babysitting_hours = 2 →
  (bike_cost - (current_savings + (allowance + mowing_payment + babysitting_hours * babysitting_rate))) = 6 :=
by
  sorry

end zach_needs_more_money_l58_58050


namespace total_money_collected_l58_58168

def hourly_wage : ℕ := 10 -- Marta's hourly wage 
def tips_collected : ℕ := 50 -- Tips collected by Marta
def hours_worked : ℕ := 19 -- Hours Marta worked

theorem total_money_collected : (hourly_wage * hours_worked + tips_collected = 240) :=
  sorry

end total_money_collected_l58_58168


namespace probability_B_l58_58979

variable (Ω : Type)

-- Define the events A, A', B
noncomputable def P {Ω : Type _} [MeasureSpace Ω] (a : event ℙ) := ℙ.MeasureSpace.measure a

variables (A B : event ℙ) (P : MeasureSpace ℙ)
variable hA : P A = 0.5
variable hB_given_A : P (B|A) = 0.9
variable A_compl : event ℙ := -A
variable hA_compl : P A_compl = 0.5
variable hB_given_A_compl : P (B|A_compl) = 0.05

theorem probability_B : P B = 0.475 :=
by
  sorry

end probability_B_l58_58979


namespace calc_101_cubed_expression_l58_58348

theorem calc_101_cubed_expression : 101^3 + 3 * (101^2) - 3 * 101 + 9 = 1060610 := 
by
  sorry

end calc_101_cubed_expression_l58_58348


namespace ian_says_1306_l58_58019

noncomputable def number_i_say := 4 * (4 * (4 * (4 * (4 * (4 * (4 * (4 * 1 - 2) - 2) - 2) - 2) - 2) - 2) - 2) - 2

theorem ian_says_1306 (n : ℕ) : 1 ≤ n ∧ n ≤ 2000 → n = 1306 :=
by sorry

end ian_says_1306_l58_58019


namespace john_drinks_42_quarts_per_week_l58_58558

def gallons_per_day : ℝ := 1.5
def quarts_per_gallon : ℝ := 4
def days_per_week : ℕ := 7

theorem john_drinks_42_quarts_per_week :
  gallons_per_day * quarts_per_gallon * days_per_week = 42 := sorry

end john_drinks_42_quarts_per_week_l58_58558


namespace prime_related_divisors_circle_l58_58209

variables (n : ℕ)

-- Definitions of prime-related and conditions for n
def is_prime (p: ℕ): Prop := p > 1 ∧ ∀ m : ℕ, m ∣ p → m = 1 ∨ m = p
def prime_related (a b : ℕ) : Prop := 
  ∃ p : ℕ, is_prime p ∧ (a = p * b ∨ b = p * a)

-- The main statement to be proven
theorem prime_related_divisors_circle (n : ℕ) : 
  (n ≥ 3) ∧ (∀ a b, a ≠ b → (a ∣ n ∧ b ∣ n) → prime_related a b) ↔ ¬ (
    ∃ (p : ℕ) (k : ℕ), is_prime p ∧ (n = p ^ k) ∨ 
    ∃ (m : ℕ), n = m ^ 2 ) :=
sorry

end prime_related_divisors_circle_l58_58209


namespace five_b_value_l58_58114

theorem five_b_value (a b : ℚ) 
  (h1 : 3 * a + 4 * b = 4) 
  (h2 : a = b - 3) : 
  5 * b = 65 / 7 := 
by
  sorry

end five_b_value_l58_58114


namespace curve_touch_all_Ca_l58_58780

theorem curve_touch_all_Ca (a : ℝ) (a_pos : a > 0) (x y : ℝ) :
  ( (y - a^2)^2 = x^2 * (a^2 - x^2) ) → (y = (3 / 4) * x^2) :=
by
  sorry

end curve_touch_all_Ca_l58_58780


namespace number_of_students_l58_58914

theorem number_of_students (n T : ℕ) (h1 : T = n * 90) 
(h2 : T - 120 = (n - 3) * 95) : n = 33 := 
by
sorry

end number_of_students_l58_58914


namespace below_sea_level_representation_l58_58824

def depth_below_sea_level (depth: Int → Prop) :=
  ∀ x, depth x → x < 0

theorem below_sea_level_representation (D: Int) (A: Int) 
  (hA: A = 9050) (hD: D = 10907) 
  (above_sea_level: Int → Prop) (below_sea_level: Int → Prop)
  (h1: ∀ y, above_sea_level y → y > 0):
  below_sea_level (-D) :=
by
  sorry

end below_sea_level_representation_l58_58824


namespace total_ounces_of_coffee_l58_58246

/-
Defining the given conditions
-/
def num_packages_10_oz : Nat := 5
def num_packages_5_oz : Nat := num_packages_10_oz + 2
def ounces_per_10_oz_pkg : Nat := 10
def ounces_per_5_oz_pkg : Nat := 5

/-
Statement to prove the total ounces of coffee
-/
theorem total_ounces_of_coffee :
  (num_packages_10_oz * ounces_per_10_oz_pkg + num_packages_5_oz * ounces_per_5_oz_pkg) = 85 := by
  sorry

end total_ounces_of_coffee_l58_58246


namespace jellybean_probability_l58_58479

theorem jellybean_probability :
  let total_jellybeans := 12
  let red_jellybeans := 5
  let blue_jellybeans := 2
  let yellow_jellybeans := 5
  let total_picks := 4
  let successful_outcomes := 10 * 7 
  let total_outcomes := Nat.choose 12 4 
  let required_probability := 14 / 99 
  successful_outcomes = 70 ∧ total_outcomes = 495 → 
  successful_outcomes / total_outcomes = required_probability := 
by 
  intros
  sorry

end jellybean_probability_l58_58479


namespace teaching_arrangements_l58_58123

theorem teaching_arrangements : 
  let teachers := ["A", "B", "C", "D", "E", "F"]
  let lessons := ["L1", "L2", "L3", "L4"]
  let valid_first_lesson := ["A", "B"]
  let valid_fourth_lesson := ["A", "C"]
  ∃ arrangements : ℕ, 
    (arrangements = 36) ∧
    (∀ (l1 l2 l3 l4 : String), (l1 ∈ valid_first_lesson) → (l4 ∈ valid_fourth_lesson) → 
      (l2 ≠ l1 ∧ l2 ≠ l4 ∧ l3 ≠ l1 ∧ l3 ≠ l4) ∧ 
      (List.length teachers - (if (l1 == "A") then 1 else 0) - (if (l4 == "A") then 1 else 0) = 4)) :=
by {
  -- This is just the theorem statement; no proof is required.
  sorry
}

end teaching_arrangements_l58_58123


namespace object_reaches_max_height_at_three_l58_58630

theorem object_reaches_max_height_at_three :
  ∀ (h : ℝ) (t : ℝ), h = -15 * (t - 3)^2 + 150 → t = 3 :=
by
  sorry

end object_reaches_max_height_at_three_l58_58630


namespace find_principal_l58_58750

-- Define the conditions
def simple_interest (P R T : ℕ) : ℕ :=
  (P * R * T) / 100

-- Given values
def SI : ℕ := 750
def R : ℕ := 6
def T : ℕ := 5

-- Proof statement
theorem find_principal : ∃ P : ℕ, simple_interest P R T = SI ∧ P = 2500 := by
  aesop

end find_principal_l58_58750


namespace original_cost_price_l58_58921

theorem original_cost_price (S P C : ℝ) (h1 : S = 260) (h2 : S = 1.20 * C) : C = 216.67 := sorry

end original_cost_price_l58_58921


namespace area_CDEB_correct_l58_58736

noncomputable def area_CDEB (A B C D F E : ℝ × ℝ) : ℝ := sorry

theorem area_CDEB_correct : 
  ∀ (A B C D F E : ℝ × ℝ),
  (dist A B = 2 * real.sqrt 5) →
  (dist B C = 1) →
  (dist C A = 5) →
  (colinear C D A) →
  (colinear F D A) →
  (dist C D = 1) →
  (dist B F = 2) →
  (dist C F = 3) →
  (colinear D F E) →
  (area_CDEB A B C D F E) = 22 / 35 :=
begin
  sorry
end

end area_CDEB_correct_l58_58736


namespace quadratic_vertex_coordinates_l58_58587

theorem quadratic_vertex_coordinates : ∀ x : ℝ,
  (∃ y : ℝ, y = 2 * x^2 - 4 * x + 5) →
  (1, 3) = (1, 3) :=
by
  intro x
  intro h
  sorry

end quadratic_vertex_coordinates_l58_58587


namespace prob_2_pow_x_in_1_2_eql_1_div_4_l58_58869

noncomputable def prob_2_pow_x_in_1_2 : ℝ :=
let A := { x : ℝ | 1 ≤ 2^x ∧ 2^x ≤ 2 }
let B := Icc (-2 : ℝ) 2 
(PMF.classicalOfFinSet B (by simp)).prob A

theorem prob_2_pow_x_in_1_2_eql_1_div_4 :
  prob_2_pow_x_in_1_2 = 1 / 4 := by
  sorry

end prob_2_pow_x_in_1_2_eql_1_div_4_l58_58869


namespace opposite_of_one_fourth_l58_58198

/-- The opposite of the fraction 1/4 is -1/4 --/
theorem opposite_of_one_fourth : - (1 / 4) = -1 / 4 :=
by
  sorry

end opposite_of_one_fourth_l58_58198


namespace construct_length_one_l58_58366

theorem construct_length_one
    (a : ℝ) 
    (h_a : a = Real.sqrt 2 + Real.sqrt 3 + Real.sqrt 5) : 
    ∃ (b : ℝ), b = 1 :=
by
    sorry

end construct_length_one_l58_58366


namespace condition_B_is_necessary_but_not_sufficient_l58_58690

-- Definitions of conditions A and B
def condition_A (x : ℝ) : Prop := 0 < x ∧ x < 5
def condition_B (x : ℝ) : Prop := abs (x - 2) < 3

-- The proof problem statement
theorem condition_B_is_necessary_but_not_sufficient : 
∀ x, condition_A x → condition_B x ∧ ¬(∀ x, condition_B x → condition_A x) := 
sorry

end condition_B_is_necessary_but_not_sufficient_l58_58690


namespace f_of_2_l58_58802

def f (x : ℝ) : ℝ := sorry

theorem f_of_2 : f 2 = 20 / 3 :=
    sorry

end f_of_2_l58_58802


namespace consecutive_numbers_l58_58729

theorem consecutive_numbers (x : ℕ) (h : (4 * x + 2) * (4 * x^2 + 6 * x + 6) = 3 * (4 * x^3 + 4 * x^2 + 18 * x + 8)) :
  x = 2 :=
sorry

end consecutive_numbers_l58_58729


namespace average_death_rate_l58_58977

-- Definitions of the given conditions
def birth_rate_two_seconds := 10
def net_increase_one_day := 345600
def seconds_per_day := 24 * 60 * 60 

-- Define the theorem to be proven
theorem average_death_rate :
  (birth_rate_two_seconds / 2) - (net_increase_one_day / seconds_per_day) = 1 :=
by 
  sorry

end average_death_rate_l58_58977


namespace determine_c_l58_58480

-- Define the points
def point1 : ℝ × ℝ := (-3, 1)
def point2 : ℝ × ℝ := (0, 4)

-- Define the direction vector calculation
def direction_vector : ℝ × ℝ := (point2.1 - point1.1, point2.2 - point1.2)

-- Define the target direction vector form
def target_direction_vector (c : ℝ) : ℝ × ℝ := (3, c)

-- Theorem stating that the calculated direction vector equals the target direction vector when c = 3
theorem determine_c : direction_vector = target_direction_vector 3 :=
by
  -- Proof omitted
  sorry

end determine_c_l58_58480


namespace solve_x_l58_58604

theorem solve_x (x : ℝ) (h : x ≠ 0) (h_eq : (5 * x) ^ 10 = (10 * x) ^ 5) : x = 2 / 5 :=
by
  sorry

end solve_x_l58_58604


namespace sample_size_is_80_l58_58714

-- Define the given conditions
variables (x : ℕ) (numA numB numC n : ℕ)

-- Conditions in Lean
def ratio_condition (x numA numB numC : ℕ) : Prop :=
  numA = 2 * x ∧ numB = 3 * x ∧ numC = 5 * x

def sample_condition (numA : ℕ) : Prop :=
  numA = 16

-- Definition of the proof problem
theorem sample_size_is_80 (x : ℕ) (numA numB numC n : ℕ)
  (h_ratio : ratio_condition x numA numB numC)
  (h_sample : sample_condition numA) : 
  n = 80 :=
by
-- The proof is omitted, just state the theorem
sorry

end sample_size_is_80_l58_58714


namespace greatest_multiple_of_four_cubed_less_than_2000_l58_58030

theorem greatest_multiple_of_four_cubed_less_than_2000 :
  ∃ x, (x > 0) ∧ (x % 4 = 0) ∧ (x^3 < 2000) ∧ ∀ y, (y > x) ∧ (y % 4 = 0) → y^3 ≥ 2000 :=
sorry

end greatest_multiple_of_four_cubed_less_than_2000_l58_58030


namespace find_a_l58_58364

theorem find_a (a : ℝ) (h : a ≠ 0) :
  (∀ x, -1 ≤ x ∧ x ≤ 4 → ax - a + 2 ≤ 7) →
  (∃ x, -1 ≤ x ∧ x ≤ 4 ∧ ax - a + 2 = 7) →
  (a = 5/3 ∨ a = -5/2) :=
by
  sorry

end find_a_l58_58364


namespace find_f2_l58_58660

-- A condition of the problem is the specific form of the function
def f (x : ℝ) (a b : ℝ) : ℝ := x^5 + a * x^3 + b * x - 8

-- Given condition
theorem find_f2 (a b : ℝ) (h : f (-2) a b = 3) : f 2 a b = -19 :=
by
  sorry

end find_f2_l58_58660


namespace alpha_beta_power_eq_sum_power_for_large_p_l58_58801

theorem alpha_beta_power_eq_sum_power_for_large_p (α β : ℂ) (p : ℕ) (hp : p ≥ 5)
  (hαβ : ∀ x : ℂ, 2 * x^4 - 6 * x^3 + 11 * x^2 - 6 * x - 4 = 0 → x = α ∨ x = β) :
  α^p + β^p = (α + β)^p :=
sorry

end alpha_beta_power_eq_sum_power_for_large_p_l58_58801


namespace min_airlines_needed_l58_58316

theorem min_airlines_needed 
  (towns : Finset ℕ) 
  (h_towns : towns.card = 21)
  (flights : Π (a : Finset ℕ), a.card = 5 → Finset (Finset ℕ))
  (h_flight : ∀ {a : Finset ℕ} (ha : a.card = 5), (flights a ha).card = 10):
  ∃ (n : ℕ), n = 21 :=
sorry

end min_airlines_needed_l58_58316


namespace solve_system_eq_l58_58183

theorem solve_system_eq (x y : ℝ) (h1 : x - y = 1) (h2 : 2 * x + 3 * y = 7) :
  x = 2 ∧ y = 1 := by
  sorry

end solve_system_eq_l58_58183


namespace sqrt_range_l58_58543

theorem sqrt_range (x : ℝ) (hx : 0 ≤ x - 1) : 1 ≤ x :=
by sorry

end sqrt_range_l58_58543


namespace proof_g_l58_58250

variable (x : ℤ)
def g (x : ℤ) := -7 * x ^ 4 - 5 * x ^ 3 + 6 * x ^ 2 - 9

theorem proof_g:
  7 * x ^ 4 - 4 * x ^ 2 + 2 + g x = -5 * x ^ 3 + 2 * x ^ 2 - 7 :=
by
  sorry

end proof_g_l58_58250


namespace interest_rate_per_annum_l58_58582

theorem interest_rate_per_annum :
  ∃ (r : ℝ), 338 = 312.50 * (1 + r) ^ 2 :=
by
  sorry

end interest_rate_per_annum_l58_58582


namespace joel_laps_count_l58_58613

def yvonne_laps : ℕ := 10

def younger_sister_laps : ℕ := yvonne_laps / 2

def joel_laps : ℕ := younger_sister_laps * 3

theorem joel_laps_count : joel_laps = 15 := by
  -- The proof is not required as per instructions
  sorry

end joel_laps_count_l58_58613


namespace positive_number_sum_square_l58_58443

theorem positive_number_sum_square (n : ℝ) (h : n^2 + n = 210) (h_pos : n > 0) : n = 14 :=
sorry

end positive_number_sum_square_l58_58443


namespace find_number_l58_58228

theorem find_number (x : ℝ) (h : 0.40 * x - 11 = 23) : x = 85 :=
sorry

end find_number_l58_58228


namespace mirka_number_l58_58414

noncomputable def original_number (a b : ℕ) : ℕ := 10 * a + b
noncomputable def reversed_number (a b : ℕ) : ℕ := 10 * b + a

theorem mirka_number (a b : ℕ) (h1 : 1 ≤ a ∧ a ≤ 4) (h2 : b = 2 * a) :
  original_number a b = 12 ∨ original_number a b = 24 ∨ original_number a b = 36 ∨ original_number a b = 48 :=
by
  sorry

end mirka_number_l58_58414


namespace remainder_2345678901_div_101_l58_58597

theorem remainder_2345678901_div_101 : 2345678901 % 101 = 12 :=
sorry

end remainder_2345678901_div_101_l58_58597


namespace find_positive_number_l58_58451

theorem find_positive_number (n : ℕ) (h : n^2 + n = 210) : n = 14 :=
sorry

end find_positive_number_l58_58451


namespace delivery_cost_l58_58686

theorem delivery_cost (base_fee : ℕ) (limit : ℕ) (extra_fee : ℕ) 
(item_weight : ℕ) (total_cost : ℕ) 
(h1 : base_fee = 13) (h2 : limit = 5) (h3 : extra_fee = 2) 
(h4 : item_weight = 7) (h5 : total_cost = 17) : 
  total_cost = base_fee + (item_weight - limit) * extra_fee := 
by
  sorry

end delivery_cost_l58_58686


namespace overall_rate_of_profit_is_25_percent_l58_58758

def cost_price_A : ℕ := 50
def selling_price_A : ℕ := 70
def cost_price_B : ℕ := 80
def selling_price_B : ℕ := 100
def cost_price_C : ℕ := 150
def selling_price_C : ℕ := 180

def profit (sp cp : ℕ) : ℕ := sp - cp

def total_cost_price : ℕ := cost_price_A + cost_price_B + cost_price_C
def total_selling_price : ℕ := selling_price_A + selling_price_B + selling_price_C
def total_profit : ℕ := profit selling_price_A cost_price_A +
                        profit selling_price_B cost_price_B +
                        profit selling_price_C cost_price_C

def overall_rate_of_profit : ℚ := (total_profit : ℚ) / (total_cost_price : ℚ) * 100

theorem overall_rate_of_profit_is_25_percent :
  overall_rate_of_profit = 25 :=
by sorry

end overall_rate_of_profit_is_25_percent_l58_58758


namespace intersection_A_B_l58_58963

def A : Set Real := { y | ∃ x : Real, y = Real.cos x }
def B : Set Real := { x | x^2 < 9 }

theorem intersection_A_B : A ∩ B = { y | -1 ≤ y ∧ y ≤ 1 } :=
by
  sorry

end intersection_A_B_l58_58963


namespace cab_driver_income_day3_l58_58922

theorem cab_driver_income_day3 :
  let income1 := 200
  let income2 := 150
  let income4 := 400
  let income5 := 500
  let avg_income := 400
  let total_income := avg_income * 5 
  total_income - (income1 + income2 + income4 + income5) = 750 := by
  sorry

end cab_driver_income_day3_l58_58922


namespace smallest_n_l58_58428

theorem smallest_n (x y : ℤ) (hx : x ≡ -2 [MOD 7]) (hy : y ≡ 2 [MOD 7]) :
  ∃ (n : ℕ), (n > 0) ∧ (x^2 + x * y + y^2 + ↑n ≡ 0 [MOD 7]) ∧ n = 3 := by
  sorry

end smallest_n_l58_58428


namespace max_value_of_q_l58_58138

theorem max_value_of_q (A M C : ℕ) (h_sum : A + M + C = 15) : 
  A * M * C + A * M + M * C + C * A ≤ 200 :=
sorry

end max_value_of_q_l58_58138


namespace average_of_remaining_numbers_l58_58431

theorem average_of_remaining_numbers 
  (numbers : List ℝ)
  (h_len : numbers.length = 15)
  (h_avg : (numbers.sum / 15) = 100)
  (h_remove : [80, 90, 95] ⊆ numbers) :
  ((numbers.sum - 80 - 90 - 95) / 12) = (1235 / 12) :=
sorry

end average_of_remaining_numbers_l58_58431


namespace proposition_truth_count_l58_58002

namespace Geometry

def is_obtuse_angle (A : Type) : Prop := sorry
def is_obtuse_triangle (ABC : Type) : Prop := sorry

def original_proposition (A : Type) (ABC : Type) : Prop :=
is_obtuse_angle A → is_obtuse_triangle ABC

def contrapositive_proposition (A : Type) (ABC : Type) : Prop :=
¬ (is_obtuse_triangle ABC) → ¬ (is_obtuse_angle A)

def converse_proposition (ABC : Type) (A : Type) : Prop :=
is_obtuse_triangle ABC → is_obtuse_angle A

def inverse_proposition (A : Type) (ABC : Type) : Prop :=
¬ (is_obtuse_angle A) → ¬ (is_obtuse_triangle ABC)

theorem proposition_truth_count (A : Type) (ABC : Type) :
  (original_proposition A ABC ∧ contrapositive_proposition A ABC ∧
  ¬ (converse_proposition ABC A) ∧ ¬ (inverse_proposition A ABC)) →
  ∃ n : ℕ, n = 2 :=
sorry

end Geometry

end proposition_truth_count_l58_58002


namespace john_cards_l58_58402

theorem john_cards (C : ℕ) (h1 : 15 * 2 + C * 2 = 70) : C = 20 :=
by
  sorry

end john_cards_l58_58402


namespace not_all_terms_positive_l58_58182

variable (a b c d : ℝ)
variable (e f g h : ℝ)

theorem not_all_terms_positive
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (he : e < 0) (hf : f < 0) (hg : g < 0) (hh : h < 0) :
  ¬ ((a * e + b * c > 0) ∧ (e * f + c * g > 0) ∧ (f * d + g * h > 0) ∧ (d * a + h * b > 0)) :=
sorry

end not_all_terms_positive_l58_58182


namespace woman_waits_time_until_man_catches_up_l58_58218

theorem woman_waits_time_until_man_catches_up
  (woman_speed : ℝ)
  (man_speed : ℝ)
  (wait_time : ℝ)
  (woman_slows_after : ℝ)
  (h_man_speed : man_speed = 5 / 60) -- man's speed in miles per minute
  (h_woman_speed : woman_speed = 25 / 60) -- woman's speed in miles per minute
  (h_wait_time : woman_slows_after = 5) -- the time in minutes after which the woman waits for man
  (h_woman_waits : wait_time = 25) : wait_time = (woman_slows_after * woman_speed) / man_speed :=
sorry

end woman_waits_time_until_man_catches_up_l58_58218


namespace max_rubles_l58_58709

theorem max_rubles (n : ℕ) (h1 : 2000 ≤ n) (h2 : n ≤ 2099) :
  (∃ k, n = 99 * k) → 
  31 ≤
    (if n % 1 = 0 then 1 else 0) +
    (if n % 3 = 0 then 3 else 0) +
    (if n % 5 = 0 then 5 else 0) +
    (if n % 7 = 0 then 7 else 0) +
    (if n % 9 = 0 then 9 else 0) +
    (if n % 11 = 0 then 11 else 0) :=
sorry

end max_rubles_l58_58709


namespace maximum_value_of_f_l58_58944

noncomputable def f (x : ℝ) : ℝ := (Real.sin x + Real.sqrt 3 * Real.cos x - 2 * Real.sin (3 * x))

theorem maximum_value_of_f :
  ∃ x : ℝ, f x = (16 * Real.sqrt 3) / 9 :=
sorry

end maximum_value_of_f_l58_58944


namespace slower_pipe_time_l58_58866

/-
One pipe can fill a tank four times as fast as another pipe. 
If together the two pipes can fill the tank in 40 minutes, 
how long will it take for the slower pipe alone to fill the tank?
-/

theorem slower_pipe_time (t : ℕ) (h1 : ∀ t, 1/t + 4/t = 1/40) : t = 200 :=
sorry

end slower_pipe_time_l58_58866


namespace line_through_P0_perpendicular_to_plane_l58_58609

-- Definitions of the given conditions
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def P0 : Point3D := { x := 3, y := 4, z := 2 }

def plane (x y z : ℝ) : Prop := 8 * x - 4 * y + 5 * z - 4 = 0

-- The proof problem statement
theorem line_through_P0_perpendicular_to_plane :
  ∃ t : ℝ, (P0.x + 8 * t = x ∧ P0.y - 4 * t = y ∧ P0.z + 5 * t = z) ↔
    (∃ t : ℝ, x = 3 + 8 * t ∧ y = 4 - 4 * t ∧ z = 2 + 5 * t) → 
    (∃ t : ℝ, (x - 3) / 8 = t ∧ (y - 4) / -4 = t ∧ (z - 2) / 5 = t) := sorry

end line_through_P0_perpendicular_to_plane_l58_58609


namespace rectangular_plot_breadth_l58_58887

theorem rectangular_plot_breadth :
  ∀ (l b : ℝ), (l = 3 * b) → (l * b = 588) → (b = 14) :=
by
  intros l b h1 h2
  sorry

end rectangular_plot_breadth_l58_58887


namespace big_container_capacity_l58_58059

-- Defining the conditions
def big_container_initial_fraction : ℚ := 0.30
def second_container_initial_fraction : ℚ := 0.50
def big_container_added_water : ℚ := 18
def second_container_added_water : ℚ := 12
def big_container_final_fraction : ℚ := 3 / 4
def second_container_final_fraction : ℚ := 0.90

-- Defining the capacity of the containers
variable (C_b C_s : ℚ)

-- Defining the equations based on the conditions
def big_container_equation : Prop :=
  big_container_initial_fraction * C_b + big_container_added_water = big_container_final_fraction * C_b

def second_container_equation : Prop :=
  second_container_initial_fraction * C_s + second_container_added_water = second_container_final_fraction * C_s

-- Proof statement to prove the capacity of the big container
theorem big_container_capacity : big_container_equation C_b → C_b = 40 :=
by
  intro H
  -- Skipping the proof steps
  sorry

end big_container_capacity_l58_58059


namespace smallest_non_palindromic_power_of_13_l58_58655

def is_palindrome (n : ℕ) : Prop := 
  let s := n.repr
  s = s.reverse

def is_power_of_13 (n : ℕ) : Prop := 
  ∃ k : ℕ, n = 13 ^ k

theorem smallest_non_palindromic_power_of_13 : ∃ n : ℕ, is_power_of_13 n ∧ ¬ is_palindrome n ∧ (∀ m : ℕ, is_power_of_13 m ∧ ¬ is_palindrome m → n ≤ m) :=
by
  use 169
  split
  · use 2
    exact rfl
  split
  · exact dec_trivial
  · intros m ⟨k, hk⟩ hpm
    sorry

end smallest_non_palindromic_power_of_13_l58_58655


namespace exists_nonconstant_poly_l58_58779

noncomputable def Q (x : ℤ) : ℤ := 420 * (x^2 - 1)^2

theorem exists_nonconstant_poly (n : ℤ) (h : n > 2) :
  ∃ Q : ℤ → ℤ, (Q ≠ (λ x, x)) ∧ (∀ k, 0 ≤ k ∧ k < n → k ∈ finset.range n → 
  let residues := (finset.image (λ x, Q x % n) (finset.range n)) in residues.card ≤ ⌊0.499 * n⌋) :=
sorry

end exists_nonconstant_poly_l58_58779


namespace parabola_transform_l58_58715

theorem parabola_transform :
  ∀ (x : ℝ),
    ∃ (y : ℝ),
      (y = -2 * x^2) →
      (∃ (y' : ℝ), y' = y - 1 ∧
      ∃ (x' : ℝ), x' = x - 3 ∧
      ∃ (y'' : ℝ), y'' = -2 * (x')^2 - 1) :=
by sorry

end parabola_transform_l58_58715


namespace bob_25_cent_coins_l58_58345

theorem bob_25_cent_coins (a b c : ℕ)
    (h₁ : a + b + c = 15)
    (h₂ : 15 + 4 * c = 27) : c = 3 := by
  sorry

end bob_25_cent_coins_l58_58345


namespace woman_speed_in_still_water_l58_58628

noncomputable def speed_in_still_water (V_c : ℝ) (t : ℝ) (d : ℝ) : ℝ :=
  let V_downstream := d / (t / 3600)
  V_downstream - V_c

theorem woman_speed_in_still_water :
  let V_c := 60
  let t := 9.99920006399488
  let d := 0.5 -- 500 meters converted to kilometers
  speed_in_still_water V_c t d = 120.01800180018 :=
by
  unfold speed_in_still_water
  sorry

end woman_speed_in_still_water_l58_58628


namespace max_value_expression_l58_58156

theorem max_value_expression (A M C : ℕ) (h₁ : A + M + C = 15) : 
  A * M * C + A * M + M * C + C * A ≤ 200 :=
  sorry

end max_value_expression_l58_58156


namespace ten_numbers_exists_l58_58847

theorem ten_numbers_exists :
  ∃ (a : Fin 10 → ℕ), 
    (∀ i j : Fin 10, i ≠ j → ¬ (a i ∣ a j))
    ∧ (∀ i j : Fin 10, i ≠ j → a i ^ 2 ∣ a j * a j) :=
sorry

end ten_numbers_exists_l58_58847


namespace more_sad_left_than_happy_l58_58771

-- Define initial conditions
def initial_sad_workers : ℕ := 36

-- Define the concept of sad and happy workers
def sad (n : ℕ) : Prop := n > 0
def happy (n : ℕ) : Prop := n > 0

-- Define the function of the game process
def game_process (initial : ℕ) : Σ (sad_out happy_out : ℕ), sad_out + happy_out = initial - 1 := 
⟨35, 0, by linarith⟩

-- Define the proof problem
theorem more_sad_left_than_happy (initial : ℕ) (game : Σ (sad_out happy_out : ℕ), sad_out + happy_out = initial - 1) :
  game.1 > game.2 := 
by 
-- Sorry because we are not providing the full proof
  sorry

-- Instantiate with initial_sad_workers
#eval more_sad_left_than_happy initial_sad_workers game_process

end more_sad_left_than_happy_l58_58771


namespace joel_laps_count_l58_58612

def yvonne_laps : ℕ := 10

def younger_sister_laps : ℕ := yvonne_laps / 2

def joel_laps : ℕ := younger_sister_laps * 3

theorem joel_laps_count : joel_laps = 15 := by
  -- The proof is not required as per instructions
  sorry

end joel_laps_count_l58_58612


namespace brandon_investment_percentage_l58_58554

noncomputable def jackson_initial_investment : ℕ := 500
noncomputable def brandon_initial_investment : ℕ := 500
noncomputable def jackson_final_investment : ℕ := 2000
noncomputable def difference_in_investments : ℕ := 1900
noncomputable def brandon_final_investment : ℕ := jackson_final_investment - difference_in_investments

theorem brandon_investment_percentage :
  (brandon_final_investment : ℝ) / (brandon_initial_investment : ℝ) * 100 = 20 := by
  sorry

end brandon_investment_percentage_l58_58554


namespace gcd_of_fraction_in_lowest_terms_l58_58932

theorem gcd_of_fraction_in_lowest_terms (n : ℤ) (h : n % 2 = 1) : Int.gcd (2 * n + 2) (3 * n + 2) = 1 := 
by 
  sorry

end gcd_of_fraction_in_lowest_terms_l58_58932


namespace return_trip_time_l58_58625

theorem return_trip_time (d p w : ℝ) (h1 : d = 84 * (p - w)) (h2 : d / (p + w) = d / p - 9) :
  (d / (p + w) = 63) ∨ (d / (p + w) = 12) :=
by
  sorry

end return_trip_time_l58_58625


namespace find_p_q_coprime_sum_l58_58901

theorem find_p_q_coprime_sum (x y n m: ℕ) (h_sum: x + y = 30)
  (h_prob: ((n/x) * (n-1)/(x-1) * (n-2)/(x-2)) * ((m/y) * (m-1)/(y-1) * (m-2)/(y-2)) = 18/25)
  : ∃ p q : ℕ, p.gcd q = 1 ∧ p + q = 1006 :=
by
  sorry

end find_p_q_coprime_sum_l58_58901


namespace find_x_prime_l58_58525

theorem find_x_prime (x : ℕ) (h1 : x > 0) (h2 : Prime (x^5 + x + 1)) : x = 1 := sorry

end find_x_prime_l58_58525


namespace charlie_received_495_l58_58618

theorem charlie_received_495 : 
  ∃ (A B C x : ℕ), 
    A + B + C = 1105 ∧ 
    A - 10 = 11 * x ∧ 
    B - 20 = 18 * x ∧ 
    C - 15 = 24 * x ∧ 
    C = 495 := 
by
  sorry

end charlie_received_495_l58_58618


namespace students_not_visiting_any_l58_58895

-- Define the given conditions as Lean definitions
def total_students := 52
def visited_botanical := 12
def visited_animal := 26
def visited_technology := 23
def visited_botanical_animal := 5
def visited_botanical_technology := 2
def visited_animal_technology := 4
def visited_all_three := 1

-- Translate the problem statement and proof goal
theorem students_not_visiting_any :
  total_students - (visited_botanical + visited_animal + visited_technology 
  - visited_botanical_animal - visited_botanical_technology 
  - visited_animal_technology + visited_all_three) = 1 :=
by
  -- The proof is omitted
  sorry

end students_not_visiting_any_l58_58895


namespace ln_of_x_sq_sub_2x_monotonic_l58_58437

noncomputable def ln_of_x_sq_sub_2x : ℝ → ℝ := fun x => Real.log (x^2 - 2*x)

theorem ln_of_x_sq_sub_2x_monotonic : ∀ x y : ℝ, (2 < x ∧ 2 < y ∧ x ≤ y) → ln_of_x_sq_sub_2x x ≤ ln_of_x_sq_sub_2x y :=
by
    intros x y h
    sorry

end ln_of_x_sq_sub_2x_monotonic_l58_58437


namespace max_q_value_l58_58144

theorem max_q_value (A M C : ℕ) (h : A + M + C = 15) : 
  (A * M * C + A * M + M * C + C * A) ≤ 200 :=
sorry

end max_q_value_l58_58144


namespace prob_of_selecting_blue_ball_l58_58248

noncomputable def prob_select_ball :=
  let prob_X := 1 / 3
  let prob_Y := 1 / 3
  let prob_Z := 1 / 3
  let prob_blue_X := 7 / 10
  let prob_blue_Y := 1 / 2
  let prob_blue_Z := 2 / 5
  prob_X * prob_blue_X + prob_Y * prob_blue_Y + prob_Z * prob_blue_Z

theorem prob_of_selecting_blue_ball :
  prob_select_ball = 8 / 15 :=
by
  -- Provide the proof here
  sorry

end prob_of_selecting_blue_ball_l58_58248


namespace intersection_A_B_l58_58091

def A : Set ℝ := { x | -1 < x ∧ x < 2 }
def B : Set ℝ := { x | x^2 - 2 * x < 0 }

theorem intersection_A_B : A ∩ B = { x | 0 < x ∧ x < 2 } :=
by
  -- We are going to skip the proof for now
  sorry

end intersection_A_B_l58_58091


namespace sum_of_radii_tangent_circles_l58_58760

theorem sum_of_radii_tangent_circles :
  ∃ (r1 r2 : ℝ), 
  (∀ r, (r = (6 + 2*Real.sqrt 6) ∨ r = (6 - 2*Real.sqrt 6)) → (r = r1 ∨ r = r2)) ∧ 
  ((r1 - 4)^2 + r1^2 = (r1 + 2)^2) ∧ 
  ((r2 - 4)^2 + r2^2 = (r2 + 2)^2) ∧ 
  (r1 + r2 = 12) :=
by
  sorry

end sum_of_radii_tangent_circles_l58_58760


namespace ratio_books_purchased_l58_58131

-- Definitions based on the conditions
def books_last_year : ℕ := 50
def books_before_purchase : ℕ := 100
def books_now : ℕ := 300

-- Let x be the multiple of the books purchased this year
def multiple_books_purchased_this_year (x : ℕ) : Prop :=
  books_now = books_before_purchase + books_last_year + books_last_year * x

-- Prove the ratio is 3:1
theorem ratio_books_purchased (x : ℕ) (h : multiple_books_purchased_this_year x) : x = 3 :=
  by sorry

end ratio_books_purchased_l58_58131


namespace bus_avg_speed_l58_58473

noncomputable def average_speed_of_bus 
  (bicycle_speed : ℕ) 
  (initial_distance_behind : ℕ) 
  (catch_up_time : ℕ) :
  ℕ :=
  (initial_distance_behind + bicycle_speed * catch_up_time) / catch_up_time

theorem bus_avg_speed 
  (bicycle_speed : ℕ) 
  (initial_distance_behind : ℕ) 
  (catch_up_time : ℕ) 
  (h_bicycle_speed : bicycle_speed = 15) 
  (h_initial_distance_behind : initial_distance_behind = 195)
  (h_catch_up_time : catch_up_time = 3) :
  average_speed_of_bus bicycle_speed initial_distance_behind catch_up_time = 80 :=
by
  sorry

end bus_avg_speed_l58_58473


namespace necessary_condition_for_inequality_l58_58889

theorem necessary_condition_for_inequality 
  (m : ℝ) : (∀ x : ℝ, x^2 - 2 * x + m > 0) → m > 0 :=
by 
  sorry

end necessary_condition_for_inequality_l58_58889


namespace solve_problem_l58_58643

noncomputable def problem_expression : ℝ :=
  4^(1/2) + Real.log (3^2) / Real.log 3

theorem solve_problem : problem_expression = 4 := by
  sorry

end solve_problem_l58_58643


namespace sqrt_two_squared_l58_58636

noncomputable def sqrt_two : Real := Real.sqrt 2

theorem sqrt_two_squared : (sqrt_two) ^ 2 = 2 :=
by
  sorry

end sqrt_two_squared_l58_58636


namespace not_all_pieces_found_l58_58935

theorem not_all_pieces_found (N : ℕ) (petya_tore : ℕ → ℕ) (vasya_tore : ℕ → ℕ) : 
  (∀ n, petya_tore n = n * 5 - n) →
  (∀ n, vasya_tore n = n * 9 - n) →
  1988 = N ∧ (N % 2 = 1) → false :=
by
  intros h_petya h_vasya h
  sorry

end not_all_pieces_found_l58_58935


namespace maximum_students_l58_58201

-- Definitions for conditions
def students (n : ℕ) := Fin n → Prop

-- Condition: Among any six students, there are two who are not friends
def not_friend_in_six (n : ℕ) (friend : Fin n → Fin n → Prop) : Prop :=
  ∀ (s : Finset (Fin n)), s.card = 6 → ∃ (a b : Fin n), a ∈ s ∧ b ∈ s ∧ ¬ friend a b

-- Condition: For any pair of students not friends, there is a student who is friends with both
def friend_of_two_not_friends (n : ℕ) (friend : Fin n → Fin n → Prop) : Prop :=
  ∀ (a b : Fin n), ¬ friend a b → ∃ (c : Fin n), c ≠ a ∧ c ≠ b ∧ friend c a ∧ friend c b

-- Theorem stating the main result
theorem maximum_students (n : ℕ) (friend : Fin n → Fin n → Prop) :
  not_friend_in_six n friend ∧ friend_of_two_not_friends n friend → n ≤ 25 := 
sorry

end maximum_students_l58_58201


namespace no_valid_partition_of_nat_l58_58568

-- Definitions of the sets A, B, and C as nonempty subsets of positive integers
variable (A B C : Set ℕ)

-- Definition to capture the key condition in the problem
def valid_partition (A B C : Set ℕ) : Prop :=
  (∀ x ∈ A, ∀ y ∈ B, (x^2 - x * y + y^2) ∈ C) 

-- The main theorem stating that such a partition is impossible
theorem no_valid_partition_of_nat : 
  (∃ A B C : Set ℕ, A ≠ ∅ ∧ B ≠ ∅ ∧ C ≠ ∅ ∧ (∀ x ∈ A, ∀ y ∈ B, (x^2 - x * y + y^2) ∈ C)) → False :=
by
  sorry

end no_valid_partition_of_nat_l58_58568


namespace number_of_six_digit_palindromes_l58_58497

def is_six_digit_palindrome (n : ℕ) : Prop :=
  ∃ a b c, a ≠ 0 ∧ 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ n = a * 100001 + b * 10010 + c * 1100

theorem number_of_six_digit_palindromes : ∃ p, p = 900 ∧ (∀ n, is_six_digit_palindrome n → n = p) :=
by
  sorry

end number_of_six_digit_palindromes_l58_58497


namespace volume_of_soil_extracted_l58_58282

-- Definition of the conditions
def Length : ℝ := 20
def Width : ℝ := 10
def Depth : ℝ := 8

-- Statement of the proof problem
theorem volume_of_soil_extracted : Length * Width * Depth = 1600 := by
  -- Proof skipped
  sorry

end volume_of_soil_extracted_l58_58282


namespace find_c_gen_formula_l58_58685

noncomputable def seq (a : ℕ → ℕ) (c : ℕ) : Prop :=
a 1 = 2 ∧
(∀ n, a (n + 1) = a n + c * n) ∧
(2 + c) * (2 + c) = 2 * (2 + 3 * c)

theorem find_c (a : ℕ → ℕ) : ∃ c, seq a c :=
by
  sorry

theorem gen_formula (a : ℕ → ℕ) (c : ℕ) (h : seq a c) : (∀ n, a n = n^2 - n + 2) :=
by
  sorry

end find_c_gen_formula_l58_58685


namespace geometric_sequence_sum_l58_58000

variable {a : ℕ → ℝ}

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
∀ n, a n = a 1 * q ^ n

theorem geometric_sequence_sum (h : geometric_sequence a 2) (h_sum : a 1 + a 2 = 3) :
  a 4 + a 5 = 24 := by
  sorry

end geometric_sequence_sum_l58_58000


namespace Ann_age_is_46_l58_58241

theorem Ann_age_is_46
  (a b : ℕ) 
  (h1 : a + b = 72)
  (h2 : b = (a / 3) + 2 * (a - b)) : a = 46 :=
by
  sorry

end Ann_age_is_46_l58_58241


namespace exists_matrices_B_C_not_exists_matrices_commute_l58_58751

-- Equivalent proof statement for part (a)
theorem exists_matrices_B_C (A : Matrix (Fin 2) (Fin 2) ℝ): 
  ∃ (B C : Matrix (Fin 2) (Fin 2) ℝ), A = B^2 + C^2 :=
by
  sorry

-- Equivalent proof statement for part (b)
theorem not_exists_matrices_commute (A : Matrix (Fin 2) (Fin 2) ℝ) 
  (hA : A = ![![0, 1], ![1, 0]]) :
  ¬∃ (B C: Matrix (Fin 2) (Fin 2) ℝ), A = B^2 + C^2 ∧ B * C = C * B :=
by
  sorry

end exists_matrices_B_C_not_exists_matrices_commute_l58_58751


namespace constant_term_expansion_l58_58034

theorem constant_term_expansion (x : ℝ) (hx : x ≠ 0) :
  ∃ k : ℝ, k = -21/2 ∧
  (∀ r : ℕ, (9 : ℕ).choose r * (x^(1/2))^(9-r) * ((-(1/(2*x)))^r) = k) :=
sorry

end constant_term_expansion_l58_58034


namespace bruce_total_payment_l58_58071

def cost_of_grapes (quantity rate : ℕ) : ℕ := quantity * rate
def cost_of_mangoes (quantity rate : ℕ) : ℕ := quantity * rate

theorem bruce_total_payment : 
  cost_of_grapes 8 70 + cost_of_mangoes 11 55 = 1165 :=
by 
  sorry

end bruce_total_payment_l58_58071


namespace max_value_expression_l58_58158

theorem max_value_expression (A M C : ℕ) (h₁ : A + M + C = 15) : 
  A * M * C + A * M + M * C + C * A ≤ 200 :=
  sorry

end max_value_expression_l58_58158


namespace sub_seq_arithmetic_l58_58393

variable (a : ℕ → ℝ) (d : ℝ)

def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

def sub_seq (a : ℕ → ℝ) (k : ℕ) : ℝ :=
  a (3 * k - 1)

theorem sub_seq_arithmetic (h : is_arithmetic_sequence a d) : is_arithmetic_sequence (sub_seq a) (3 * d) := 
sorry


end sub_seq_arithmetic_l58_58393


namespace coordinates_of_point_in_fourth_quadrant_l58_58190

theorem coordinates_of_point_in_fourth_quadrant 
  (P : ℝ × ℝ)
  (h₁ : P.1 > 0) -- P is in the fourth quadrant, so x > 0
  (h₂ : P.2 < 0) -- P is in the fourth quadrant, so y < 0
  (dist_x_axis : P.2 = -5) -- Distance from P to x-axis is 5 (absolute value of y)
  (dist_y_axis : P.1 = 3)  -- Distance from P to y-axis is 3 (absolute value of x)
  : P = (3, -5) :=
sorry

end coordinates_of_point_in_fourth_quadrant_l58_58190


namespace hotdogs_remainder_zero_l58_58388

theorem hotdogs_remainder_zero :
  25197624 % 6 = 0 :=
by
  sorry -- Proof not required

end hotdogs_remainder_zero_l58_58388


namespace percentage_answered_first_correctly_l58_58923

-- Defining the given conditions
def percentage_answered_second_correctly : ℝ := 0.25
def percentage_answered_neither_correctly : ℝ := 0.20
def percentage_answered_both_correctly : ℝ := 0.20

-- Lean statement for the proof problem
theorem percentage_answered_first_correctly :
  ∃ a : ℝ, a + percentage_answered_second_correctly - percentage_answered_both_correctly = 0.80 ∧ a = 0.75 := by
  sorry

end percentage_answered_first_correctly_l58_58923


namespace y_is_multiple_of_4_y_is_not_multiple_of_8_y_is_not_multiple_of_16_y_is_not_multiple_of_32_l58_58011

def y := 96 + 144 + 200 + 300 + 600 + 720 + 4800

theorem y_is_multiple_of_4 : y % 4 = 0 := 
by sorry

theorem y_is_not_multiple_of_8 : y % 8 ≠ 0 := 
by sorry

theorem y_is_not_multiple_of_16 : y % 16 ≠ 0 := 
by sorry

theorem y_is_not_multiple_of_32 : y % 32 ≠ 0 := 
by sorry

end y_is_multiple_of_4_y_is_not_multiple_of_8_y_is_not_multiple_of_16_y_is_not_multiple_of_32_l58_58011


namespace percentage_proof_l58_58993

theorem percentage_proof (n : ℝ) (h : 0.3 * 0.4 * n = 24) : 0.4 * 0.3 * n = 24 :=
sorry

end percentage_proof_l58_58993


namespace emily_subtracts_99_from_50sq_to_get_49sq_l58_58045

-- Define the identity for squares
theorem emily_subtracts_99_from_50sq_to_get_49sq :
  ∀ (x : ℕ), (49 : ℕ) = (50 - 1) → (x = 50 → 49^2 = 50^2 - 99) := by
  intro x h1 h2
  sorry

end emily_subtracts_99_from_50sq_to_get_49sq_l58_58045


namespace cake_volume_l58_58334

theorem cake_volume :
  let thickness := 1 / 2
  let diameter := 16
  let radius := diameter / 2
  let total_volume := Real.pi * radius^2 * thickness
  total_volume / 16 = 2 * Real.pi := by
    sorry

end cake_volume_l58_58334


namespace sad_employees_left_geq_cheerful_l58_58770

-- Define the initial number of sad employees
def initial_sad_employees : Nat := 36

-- Define the final number of remaining employees after the game
def final_remaining_employees : Nat := 1

-- Define the total number of employees hit and out of the game
def employees_out : Nat := initial_sad_employees - final_remaining_employees

-- Define the number of cheerful employees who have left
def cheerful_employees_left := employees_out

-- Define the number of sad employees who have left
def sad_employees_left := employees_out

-- The theorem stating the problem proof
theorem sad_employees_left_geq_cheerful:
    sad_employees_left ≥ cheerful_employees_left :=
by
  -- Proof is omitted
  sorry

end sad_employees_left_geq_cheerful_l58_58770


namespace max_value_expression_l58_58157

theorem max_value_expression (A M C : ℕ) (h₁ : A + M + C = 15) : 
  A * M * C + A * M + M * C + C * A ≤ 200 :=
  sorry

end max_value_expression_l58_58157


namespace number_of_six_digit_palindromes_l58_58496

def is_six_digit_palindrome (n : ℕ) : Prop :=
  ∃ a b c, a ≠ 0 ∧ 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ n = a * 100001 + b * 10010 + c * 1100

theorem number_of_six_digit_palindromes : ∃ p, p = 900 ∧ (∀ n, is_six_digit_palindrome n → n = p) :=
by
  sorry

end number_of_six_digit_palindromes_l58_58496


namespace purchasing_power_increase_l58_58535

theorem purchasing_power_increase (P M : ℝ) (h : 0 < P ∧ 0 < M) :
  let new_price := 0.80 * P
  let original_quantity := M / P
  let new_quantity := M / new_price
  new_quantity = 1.25 * original_quantity :=
by
  sorry

end purchasing_power_increase_l58_58535


namespace below_sea_level_representation_l58_58823

def depth_below_sea_level (depth: Int → Prop) :=
  ∀ x, depth x → x < 0

theorem below_sea_level_representation (D: Int) (A: Int) 
  (hA: A = 9050) (hD: D = 10907) 
  (above_sea_level: Int → Prop) (below_sea_level: Int → Prop)
  (h1: ∀ y, above_sea_level y → y > 0):
  below_sea_level (-D) :=
by
  sorry

end below_sea_level_representation_l58_58823


namespace Nikolai_faster_than_Gennady_l58_58742

theorem Nikolai_faster_than_Gennady
  (gennady_jump_time : ℕ)
  (nikolai_jump_time : ℕ)
  (jump_distance_gennady: ℕ)
  (jump_distance_nikolai: ℕ)
  (jump_count_gennady : ℕ)
  (jump_count_nikolai : ℕ)
  (total_distance : ℕ)
  (h1 : gennady_jump_time = nikolai_jump_time)
  (h2 : jump_distance_gennady = 6)
  (h3 : jump_distance_nikolai = 4)
  (h4 : jump_count_gennady = 2)
  (h5 : jump_count_nikolai = 3)
  (h6 : total_distance = 2000) :
  (total_distance % jump_distance_nikolai = 0) ∧ (total_distance % jump_distance_gennady ≠ 0) → 
  nikolai_jump_time < gennady_jump_time := 
sorry

end Nikolai_faster_than_Gennady_l58_58742


namespace proof_problem_l58_58693

noncomputable def f (x y k : ℝ) : ℝ := k * x + (1 / y)

theorem proof_problem
  (a b k : ℝ) (h1 : f a b k = f b a k) (h2 : a ≠ b) :
  f (a * b) 1 k = 0 :=
sorry

end proof_problem_l58_58693


namespace find_k_for_equation_l58_58550

theorem find_k_for_equation : 
  ∃ k : ℤ, -x^2 - (k + 7) * x - 8 = -(x - 2) * (x - 4) → k = -13 := 
by
  sorry

end find_k_for_equation_l58_58550


namespace intersection_volume_calculation_l58_58602

noncomputable def volume_of_intersection : ℝ :=
  let region1 := {p : ℝ × ℝ × ℝ | abs p.1 + abs p.2 + abs p.3 ≤ 2}
  let region2 := {p : ℝ × ℝ × ℝ | abs p.1 + abs p.2 + abs (p.3 - 2) ≤ 2}
  let intersection := region1 ∩ region2
  8/3

theorem intersection_volume_calculation :
  volume_of_intersection = 8 / 3 :=
begin
  sorry
end

end intersection_volume_calculation_l58_58602


namespace arithmetic_series_sum_l58_58908

theorem arithmetic_series_sum :
  let a := 2
  let d := 3
  let l := 56
  let n := 19
  let pairs_sum := (n-1) / 2 * (-3)
  let single_term := 56
  2 - 5 + 8 - 11 + 14 - 17 + 20 - 23 + 26 - 29 + 32 - 35 + 38 - 41 + 44 - 47 + 50 - 53 + 56 = 29 :=
by
  sorry

end arithmetic_series_sum_l58_58908


namespace jenna_less_than_bob_l58_58814

theorem jenna_less_than_bob :
  ∀ (bob jenna phil : ℕ),
  (bob = 60) →
  (phil = bob / 3) →
  (jenna = 2 * phil) →
  (bob - jenna = 20) :=
by
  intros bob jenna phil h1 h2 h3
  sorry

end jenna_less_than_bob_l58_58814


namespace zach_needs_more_money_zach_more_money_needed_l58_58052

/-!
# Zach's Bike Savings Problem
Zach needs $100 to buy a brand new bike.
Weekly allowance: $5.
Earnings from mowing the lawn: $10.
Earnings from babysitting: $7 per hour.
Zach has already saved $65.
He will receive weekly allowance on Friday.
He will mow the lawn and babysit for 2 hours this Saturday.
Prove that Zach needs $6 more to buy the bike.
-/

def zach_current_savings : ℕ := 65
def bike_cost : ℕ := 100
def weekly_allowance : ℕ := 5
def mowing_earnings : ℕ := 10
def babysitting_rate : ℕ := 7
def babysitting_hours : ℕ := 2

theorem zach_needs_more_money : zach_current_savings + (weekly_allowance + mowing_earnings + (babysitting_rate * babysitting_hours)) = 94 :=
by sorry

theorem zach_more_money_needed : bike_cost - (zach_current_savings + (weekly_allowance + mowing_earnings + (babysitting_rate * babysitting_hours))) = 6 :=
by sorry

end zach_needs_more_money_zach_more_money_needed_l58_58052


namespace volume_of_intersection_of_octahedra_l58_58603

theorem volume_of_intersection_of_octahedra :
  let region1 := {p : ℝ × ℝ × ℝ | abs p.1 + abs p.2 + abs p.3 ≤ 2}
  let region2 := {p : ℝ × ℝ × ℝ | abs p.1 + abs p.2 + abs (p.3 - 2) ≤ 2}
  region1 ∩ region2 = volume (region1 ∩ region2) = 16 / 3 := 
sorry

end volume_of_intersection_of_octahedra_l58_58603


namespace car_speed_first_hour_l58_58724

theorem car_speed_first_hour (x : ℕ) (h1 : 60 > 0) (h2 : 40 > 0) (h3 : 2 > 0) (avg_speed : 40 = (x + 60) / 2) : x = 20 := 
by
  sorry

end car_speed_first_hour_l58_58724


namespace lesser_fraction_of_sum_and_product_l58_58044

open Real

theorem lesser_fraction_of_sum_and_product (a b : ℚ)
  (h1 : a + b = 11 / 12)
  (h2 : a * b = 1 / 6) :
  min a b = 1 / 4 :=
sorry

end lesser_fraction_of_sum_and_product_l58_58044


namespace number_of_standing_demons_l58_58284

variable (N : ℕ)
variable (initial_knocked_down : ℕ)
variable (initial_standing : ℕ)
variable (current_knocked_down : ℕ)
variable (current_standing : ℕ)

axiom initial_condition : initial_knocked_down = (3 * initial_standing) / 2
axiom condition_after_changes : current_knocked_down = initial_knocked_down + 2
axiom condition_after_changes_2 : current_standing = initial_standing - 10
axiom final_condition : current_standing = (5 * current_knocked_down) / 4

theorem number_of_standing_demons : current_standing = 35 :=
sorry

end number_of_standing_demons_l58_58284


namespace original_price_l58_58723

theorem original_price (P : ℝ) (h_discount : 0.75 * P = 560): P = 746.68 :=
sorry

end original_price_l58_58723


namespace problem_statement_l58_58984

theorem problem_statement (a b c d n : Nat) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 0 < n) (h_eq : 7 * 4^n = a^2 + b^2 + c^2 + d^2) : 
  a ≥ 2^(n-1) ∧ b ≥ 2^(n-1) ∧ c ≥ 2^(n-1) ∧ d ≥ 2^(n-1) :=
sorry

end problem_statement_l58_58984


namespace demand_change_for_revenue_l58_58330

theorem demand_change_for_revenue (P D D' : ℝ)
  (h1 : D' = (1.10 * D) / 1.20)
  (h2 : P' = 1.20 * P)
  (h3 : P * D = P' * D') :
  (D' - D) / D * 100 = -8.33 := by
sorry

end demand_change_for_revenue_l58_58330


namespace determine_defective_coin_l58_58204

-- Define the properties of the coins
structure Coin :=
(denomination : ℕ)
(weight : ℕ)

-- Given coins
def c1 : Coin := ⟨1, 1⟩
def c2 : Coin := ⟨2, 2⟩
def c3 : Coin := ⟨3, 3⟩
def c5 : Coin := ⟨5, 5⟩

-- Assume one coin is defective
variable (defective : Coin)
variable (differing_weight : ℕ)
#check differing_weight

theorem determine_defective_coin :
  (∃ (defective : Coin), ∀ (c : Coin), 
    c ≠ defective → c.weight = c.denomination) → 
  ((c2.weight + c3.weight = c5.weight → defective = c1) ∧
   (c1.weight + c2.weight = c3.weight → defective = c5) ∧
   (c2.weight ≠ 2 → defective = c2) ∧
   (c3.weight ≠ 3 → defective = c3)) :=
by
  sorry

end determine_defective_coin_l58_58204


namespace total_books_l58_58971

-- Define the number of books Stu has
def Stu_books : ℕ := 9

-- Define the multiplier for Albert's books
def Albert_multiplier : ℕ := 4

-- Define the number of books Albert has
def Albert_books : ℕ := Albert_multiplier * Stu_books

-- Prove that the total number of books is 45
theorem total_books:
  Stu_books + Albert_books = 45 :=
by 
  -- This is where the proof steps would go, but we skip it for now 
  sorry

end total_books_l58_58971


namespace operation_two_three_l58_58351

def operation (a b : ℕ) : ℤ := 4 * a ^ 2 - 4 * b ^ 2

theorem operation_two_three : operation 2 3 = -20 :=
by
  sorry

end operation_two_three_l58_58351


namespace sum_of_digits_of_x_l58_58332

def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

theorem sum_of_digits_of_x (x : ℕ) (h1 : 100 ≤ x) (h2 : x ≤ 949)
  (h3 : is_palindrome x) (h4 : is_palindrome (x + 50)) :
  sum_of_digits x = 19 :=
sorry

end sum_of_digits_of_x_l58_58332


namespace abc_is_cube_of_integer_l58_58295

theorem abc_is_cube_of_integer (a b c : ℤ) (h : (a : ℚ) / b + (b : ℚ) / c + (c : ℚ) / a = 3) : ∃ k : ℤ, abc = k^3 := 
by
  sorry

end abc_is_cube_of_integer_l58_58295


namespace seq_geom_seq_of_geom_and_arith_l58_58803

theorem seq_geom_seq_of_geom_and_arith (a : ℕ → ℕ) (b : ℕ → ℕ) 
  (h1 : ∃ a₁ : ℕ, ∀ n : ℕ, a n = a₁ * 2^(n-1))
  (h2 : ∃ b₁ d : ℕ, d = 3 ∧ ∀ n : ℕ, b (n + 1) = b₁ + n * d ∧ b₁ > 0) :
  ∃ r : ℕ, r = 8 ∧ ∃ a₁ : ℕ, ∀ n : ℕ, a (b (n + 1)) = a₁ * r^n :=
by
  sorry

end seq_geom_seq_of_geom_and_arith_l58_58803


namespace nikolai_completes_faster_l58_58740

-- Given conditions: distances they can cover in the same time and total journey length 
def gennady_jump_distance := 2 * 6 -- 12 meters
def nikolai_jump_distance := 3 * 4 -- 12 meters
def total_distance := 2000 -- 2000 meters before turning back

-- Mathematical translation + Target proof: prove that Nikolai will complete the journey faster
theorem nikolai_completes_faster 
  (gennady_distance_per_time : gennady_jump_distance = 12)
  (nikolai_distance_per_time : nikolai_jump_distance = 12)
  (journey_length : total_distance = 2000) : 
  ( (2000 % 4 = 0) ∧ (2000 % 6 ≠ 0) ) -> true := 
by 
  intros,
  sorry

end nikolai_completes_faster_l58_58740


namespace books_total_l58_58969

def stuBooks : ℕ := 9
def albertBooks : ℕ := 4 * stuBooks
def totalBooks : ℕ := stuBooks + albertBooks

theorem books_total : totalBooks = 45 := by
  sorry

end books_total_l58_58969


namespace a_investment_l58_58219

theorem a_investment
  (b_investment : ℝ) (c_investment : ℝ) (c_share_profit : ℝ) (total_profit : ℝ)
  (h1 : b_investment = 45000)
  (h2 : c_investment = 50000)
  (h3 : c_share_profit = 36000)
  (h4 : total_profit = 90000) :
  ∃ A : ℝ, A = 30000 :=
by {
  sorry
}

end a_investment_l58_58219


namespace complex_product_l58_58689

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- Define the complex numbers z1 and z2
def z1 : ℂ := 1 - i
def z2 : ℂ := 3 + i

-- Statement of the problem
theorem complex_product : z1 * z2 = 4 - 2 * i := by
  sorry

end complex_product_l58_58689


namespace contradiction_assumption_l58_58310

-- Define the numbers x, y, z
variables (x y z : ℝ)

-- Define the assumption that all three numbers are non-positive
def all_non_positive (x y z : ℝ) : Prop := x ≤ 0 ∧ y ≤ 0 ∧ z ≤ 0

-- State the proposition to prove using the method of contradiction
theorem contradiction_assumption (h : all_non_positive x y z) : ¬ (x > 0 ∨ y > 0 ∨ z > 0) :=
by
  sorry

end contradiction_assumption_l58_58310


namespace max_value_of_q_l58_58140

theorem max_value_of_q (A M C : ℕ) (h_sum : A + M + C = 15) : 
  A * M * C + A * M + M * C + C * A ≤ 200 :=
sorry

end max_value_of_q_l58_58140


namespace max_value_of_expression_l58_58154

theorem max_value_of_expression (A M C : ℕ) (h : A + M + C = 15) :
  A * M * C + A * M + M * C + C * A ≤ 200 :=
sorry

end max_value_of_expression_l58_58154


namespace prism_surface_area_is_8pi_l58_58368

noncomputable def prismSphereSurfaceArea : ℝ :=
  let AB := 2
  let AC := 1
  let BAC := Real.pi / 3 -- angle 60 degrees in radians
  let volume := Real.sqrt 3
  let AA1 := 2
  let radius := Real.sqrt 2
  let surface_area := 4 * Real.pi * radius^2
  surface_area

theorem prism_surface_area_is_8pi : prismSphereSurfaceArea = 8 * Real.pi :=
  by
    sorry

end prism_surface_area_is_8pi_l58_58368


namespace sufficient_but_not_necessary_l58_58324

theorem sufficient_but_not_necessary (x : ℝ) (h : 2 < x ∧ x < 3) :
  x * (x - 5) < 0 ∧ ∃ y, y * (y - 5) < 0 ∧ (2 ≤ y ∧ y ≤ 3) → False :=
by
  sorry

end sufficient_but_not_necessary_l58_58324


namespace percent_of_150_is_60_l58_58756

def percent_is_correct (Part Whole : ℝ) : Prop :=
  (Part / Whole) * 100 = 250

theorem percent_of_150_is_60 :
  percent_is_correct 150 60 :=
by
  sorry

end percent_of_150_is_60_l58_58756


namespace intersection_point_l58_58353

-- Definitions of the lines
def line1 (x y : ℚ) : Prop := 8 * x - 5 * y = 10
def line2 (x y : ℚ) : Prop := 6 * x + 2 * y = 20

-- Theorem stating the intersection point
theorem intersection_point : line1 (60 / 23) (50 / 23) ∧ line2 (60 / 23) (50 / 23) :=
by {
  sorry
}

end intersection_point_l58_58353


namespace simplify_frac_l58_58873

variable (b c : ℕ)
variable (b_val : b = 2)
variable (c_val : c = 3)

theorem simplify_frac : (15 * b ^ 4 * c ^ 2) / (45 * b ^ 3 * c) = 2 :=
by
  rw [b_val, c_val]
  sorry

end simplify_frac_l58_58873


namespace concert_ticket_sales_l58_58897

theorem concert_ticket_sales (A C : ℕ) (total : ℕ) :
  (C = 3 * A) →
  (7 * A + 3 * C = 6000) →
  (total = A + C) →
  total = 1500 :=
by
  intros
  -- The proof is not required
  sorry

end concert_ticket_sales_l58_58897


namespace even_odd_product_l58_58013

theorem even_odd_product (n : ℕ) (i : Fin n → Fin n) (h_perm : ∀ j : Fin n, ∃ k : Fin n, i k = j) :
  (∃ l, l % 2 = 0) → 
  ∀ (k : Fin n), ¬(i k = k) → 
  (n % 2 = 0 → (∃ m : ℤ, m + 1 % 2 = 1) ∨ (∃ m : ℤ, m + 1 % 2 = 0)) ∧ 
  (n % 2 = 1 → (∃ m : ℤ, m + 1 % 2 = 0)) :=
by
  sorry

end even_odd_product_l58_58013


namespace sum_of_n_for_perfect_square_l58_58010

theorem sum_of_n_for_perfect_square (n : ℕ) (Sn : ℕ) 
  (hSn : Sn = n^2 + 20 * n + 12) 
  (hn : n > 0) :
  ∃ k : ℕ, k^2 = Sn → (sum_of_possible_n = 16) :=
by
  sorry

end sum_of_n_for_perfect_square_l58_58010


namespace ryan_fish_count_l58_58556

theorem ryan_fish_count
  (R : ℕ)
  (J : ℕ)
  (Jeffery_fish : ℕ)
  (h1 : Jeffery_fish = 60)
  (h2 : Jeffery_fish = 2 * R)
  (h3 : J + R + Jeffery_fish = 100)
  : R = 30 :=
by
  sorry

end ryan_fish_count_l58_58556


namespace clean_room_to_homework_ratio_l58_58292

-- Define the conditions
def timeHomework : ℕ := 30
def timeWalkDog : ℕ := timeHomework + 5
def timeTrash : ℕ := timeHomework / 6
def totalTimeAvailable : ℕ := 120
def remainingTime : ℕ := 35

-- Definition to calculate total time spent on other tasks
def totalTimeOnOtherTasks : ℕ := timeHomework + timeWalkDog + timeTrash

-- Definition to calculate the time to clean the room
def timeCleanRoom : ℕ := totalTimeAvailable - remainingTime - totalTimeOnOtherTasks

-- The theorem to prove the ratio
theorem clean_room_to_homework_ratio : (timeCleanRoom : ℚ) / (timeHomework : ℚ) = 1 / 2 :=
by
  -- Proof steps would go here
  sorry

end clean_room_to_homework_ratio_l58_58292


namespace triangles_formed_l58_58299

-- Define the combinatorial function for binomial coefficients.
def binom (n k : ℕ) : ℕ :=
  if h : k ≤ n then Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k)) else 0

-- Given conditions
def points_on_first_line := 6
def points_on_second_line := 8

-- Number of triangles calculation
def total_triangles :=
  binom points_on_first_line 2 * binom points_on_second_line 1 +
  binom points_on_first_line 1 * binom points_on_second_line 2

-- The final theorem to prove
theorem triangles_formed : total_triangles = 288 :=
by
  sorry

end triangles_formed_l58_58299


namespace gcd_21_eq_7_count_l58_58948

theorem gcd_21_eq_7_count : Nat.card {n : Fin 200 // Nat.gcd 21 n = 7} = 19 := 
by
  sorry

end gcd_21_eq_7_count_l58_58948


namespace solve_inequality_l58_58027

theorem solve_inequality (x : ℝ) (h : x ≠ -2 / 3) :
  3 - (1 / (3 * x + 2)) < 5 ↔ (x < -7 / 6 ∨ x > -2 / 3) := by
  sorry

end solve_inequality_l58_58027


namespace regular_polygon_perimeter_l58_58485

theorem regular_polygon_perimeter (s : ℕ) (E : ℕ) (n : ℕ) (P : ℕ)
  (h1 : s = 6)
  (h2 : E = 90)
  (h3 : E = 360 / n)
  (h4 : P = n * s) :
  P = 24 :=
by sorry

end regular_polygon_perimeter_l58_58485


namespace max_value_of_expression_l58_58150

theorem max_value_of_expression (A M C : ℕ) (h : A + M + C = 15) :
  A * M * C + A * M + M * C + C * A ≤ 200 :=
sorry

end max_value_of_expression_l58_58150


namespace total_money_l58_58629

variable (A B C: ℕ)
variable (h1: A + C = 200) 
variable (h2: B + C = 350)
variable (h3: C = 200)

theorem total_money : A + B + C = 350 :=
by
  sorry

end total_money_l58_58629


namespace max_value_expression_l58_58159

theorem max_value_expression (A M C : ℕ) (h₁ : A + M + C = 15) : 
  A * M * C + A * M + M * C + C * A ≤ 200 :=
  sorry

end max_value_expression_l58_58159


namespace find_x_l58_58528

open Nat

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem find_x (x : ℕ) (hx : x > 0) (hprime : is_prime (x^5 + x + 1)) : x = 1 := 
by 
  sorry

end find_x_l58_58528


namespace cube_numbers_not_all_even_cube_numbers_not_all_divisible_by_3_l58_58343

-- Define the initial state of the cube vertices
def initial_cube : ℕ → ℕ
| 0 => 1  -- The number at vertex 0 is 1
| _ => 0  -- The numbers at other vertices are 0

-- Define the edge addition operation
def edge_add (v1 v2 : ℕ → ℕ) (edge : ℕ × ℕ) : ℕ → ℕ :=
  λ x => if x = edge.1 ∨ x = edge.2 then v1 x + 1 else v1 x

-- Condition: one can add one to the numbers at the ends of any edge
axiom edge_op : ∀ (v : ℕ → ℕ) (e : ℕ × ℕ), ℕ → ℕ

-- Defining the problem in Lean
theorem cube_numbers_not_all_even :
  ¬ (∃ (v : ℕ → ℕ), ∀ x, v x % 2 = 0) :=
by
  -- Proof not required
  sorry

theorem cube_numbers_not_all_divisible_by_3 :
  ¬ (∃ (v : ℕ → ℕ), ∀ x, v x % 3 = 0) :=
by
  -- Proof not required
  sorry

end cube_numbers_not_all_even_cube_numbers_not_all_divisible_by_3_l58_58343


namespace max_value_of_expression_l58_58155

theorem max_value_of_expression (A M C : ℕ) (h : A + M + C = 15) :
  A * M * C + A * M + M * C + C * A ≤ 200 :=
sorry

end max_value_of_expression_l58_58155


namespace sunflower_seeds_more_than_half_on_day_three_l58_58865

-- Define the initial state and parameters
def initial_sunflower_seeds : ℚ := 0.4
def initial_other_seeds : ℚ := 0.6
def daily_added_sunflower_seeds : ℚ := 0.2
def daily_added_other_seeds : ℚ := 0.3
def daily_sunflower_eaten_factor : ℚ := 0.7
def daily_other_eaten_factor : ℚ := 0.4

-- Define the recurrence relations for sunflower seeds and total seeds
def sunflower_seeds (n : ℕ) : ℚ :=
  match n with
  | 0     => initial_sunflower_seeds
  | (n+1) => daily_sunflower_eaten_factor * sunflower_seeds n + daily_added_sunflower_seeds

def total_seeds (n : ℕ) : ℚ := 1 + (n : ℚ) * 0.5

-- Define the main theorem stating that on Tuesday (Day 3), sunflower seeds are more than half
theorem sunflower_seeds_more_than_half_on_day_three : sunflower_seeds 2 / total_seeds 2 > 0.5 :=
by
  -- Formal proof will go here
  sorry

end sunflower_seeds_more_than_half_on_day_three_l58_58865


namespace intersection_in_fourth_quadrant_l58_58192

theorem intersection_in_fourth_quadrant :
  (∃ x y : ℝ, y = -x ∧ y = 2 * x - 1 ∧ x = 1 ∧ y = -1) ∧ (1 > 0 ∧ -1 < 0) :=
by
  sorry

end intersection_in_fourth_quadrant_l58_58192


namespace correct_option_C_l58_58103

def f (x : ℝ) : ℝ := (2^x - 1) / (2^x + 1)

theorem correct_option_C : ∀ {x1 x2 : ℝ}, 0 < x1 → x1 < x2 → x1 * f x1 < x2 * f x2 :=
by
  sorry

end correct_option_C_l58_58103


namespace minimum_ratio_cone_cylinder_l58_58319

theorem minimum_ratio_cone_cylinder (r : ℝ) (h : ℝ) (a : ℝ) :
  (h = 4 * r) →
  (a^2 = r^2 * h^2 / (h - 2 * r)) →
  (∀ h > 0, ∃ V_cone V_cylinder, 
    V_cone = (1/3) * π * a^2 * h ∧ 
    V_cylinder = π * r^2 * (2 * r) ∧ 
    V_cone / V_cylinder = (4 / 3)) := 
sorry

end minimum_ratio_cone_cylinder_l58_58319


namespace function_property_l58_58692

theorem function_property (k a b : ℝ) (h : a ≠ b) (h_cond : k * a + 1 / b = k * b + 1 / a) : 
  k * (a * b) + 1 = 0 := 
by 
  sorry

end function_property_l58_58692


namespace total_people_on_bus_l58_58174

-- Definitions of the conditions
def num_boys : ℕ := 50
def num_girls : ℕ := (2 / 5 : ℚ) * num_boys
def num_students : ℕ := num_boys + num_girls.toNat
def num_non_students : ℕ := 3 -- Mr. Gordon, the driver, and the assistant

-- The theorem to be proven
theorem total_people_on_bus : num_students + num_non_students = 123 := by
  sorry

end total_people_on_bus_l58_58174


namespace boat_speed_in_still_water_l58_58585

variable (x : ℝ) -- speed of the boat in still water in km/hr
variable (current_rate : ℝ := 4) -- rate of the current in km/hr
variable (downstream_distance : ℝ := 4.8) -- distance traveled downstream in km
variable (downstream_time : ℝ := 18 / 60) -- time traveled downstream in hours

-- The main theorem stating that the speed of the boat in still water is 12 km/hr
theorem boat_speed_in_still_water : x = 12 :=
by
  -- Express the downstream speed and time relation
  have downstream_speed := x + current_rate
  have distance_relation := downstream_distance = downstream_speed * downstream_time
  -- Simplify and solve for x
  simp at distance_relation
  sorry

end boat_speed_in_still_water_l58_58585


namespace determine_properties_range_of_m_l58_58805

noncomputable def f (a x : ℝ) : ℝ := (a / (a - 1)) * (2^x - 2^(-x))

theorem determine_properties (a : ℝ) (h_a_pos : a > 0) (h_a_ne_one : a ≠ 1) :
  (∀ x : ℝ, f a (-x) = -f a x) ∧
  ((0 < a ∧ a < 1) → ∀ x1 x2 : ℝ, x1 < x2 → f a x1 > f a x2) ∧
  (a > 1 → ∀ x1 x2 : ℝ, x1 < x2 → f a x1 < f a x2) := 
sorry

theorem range_of_m (a m : ℝ) (h_a_pos : a > 0) (h_a_ne_one : a ≠ 1) (h_m_in_I : -1 < m ∧ m < 1) :
  f a (m - 1) + f a m < 0 ↔ 
  ((0 < a ∧ a < 1 → (1 / 2) < m ∧ m < 1) ∧
  (a > 1 → 0 < m ∧ m < (1 / 2))) := 
sorry

end determine_properties_range_of_m_l58_58805


namespace perimeter_regular_polygon_l58_58486

-- Condition definitions
def is_regular_polygon (n : ℕ) (s : ℝ) : Prop := 
  n * s > 0

def exterior_angle (E : ℝ) (n : ℕ) : Prop := 
  E = 360 / n

def side_length (s : ℝ) : Prop :=
  s = 6

-- Theorem statement to prove the perimeter is 24 units
theorem perimeter_regular_polygon 
  (n : ℕ) (s E : ℝ)
  (h1 : is_regular_polygon n s)
  (h2 : exterior_angle E n)
  (h3 : side_length s)
  (h4 : E = 90) :
  4 * s = 24 :=
by
  sorry

end perimeter_regular_polygon_l58_58486


namespace initial_position_is_minus_one_l58_58926

def initial_position_of_A (A B C : ℤ) : Prop :=
  B = A - 3 ∧ C = B + 5 ∧ C = 1 ∧ A = -1

theorem initial_position_is_minus_one (A B C : ℤ) (h1 : B = A - 3) (h2 : C = B + 5) (h3 : C = 1) : A = -1 :=
  by sorry

end initial_position_is_minus_one_l58_58926


namespace probability_three_or_more_same_l58_58988

-- Let us define the total number of outcomes when rolling 5 8-sided dice
def total_outcomes : ℕ := 8 ^ 5

-- Define the number of favorable outcomes where at least three dice show the same number
def favorable_outcomes : ℕ := 4208

-- Define the probability as a fraction
def probability : ℚ := favorable_outcomes / total_outcomes

-- Now we state the theorem that this probability simplifies to 1052/8192
theorem probability_three_or_more_same : probability = 1052 / 8192 :=
sorry

end probability_three_or_more_same_l58_58988


namespace max_value_of_q_l58_58143

theorem max_value_of_q (A M C : ℕ) (h_sum : A + M + C = 15) : 
  A * M * C + A * M + M * C + C * A ≤ 200 :=
sorry

end max_value_of_q_l58_58143


namespace distance_between_closest_points_of_circles_l58_58349

theorem distance_between_closest_points_of_circles :
  let circle1_center : ℝ × ℝ := (3, 3)
  let circle2_center : ℝ × ℝ := (20, 15)
  let circle1_radius : ℝ := 3
  let circle2_radius : ℝ := 15
  let distance_between_centers : ℝ := Real.sqrt ((20 - 3)^2 + (15 - 3)^2)
  distance_between_centers - (circle1_radius + circle2_radius) = 2.81 :=
by {
  sorry
}

end distance_between_closest_points_of_circles_l58_58349


namespace jordan_oreos_l58_58128

theorem jordan_oreos 
  (x : ℕ) 
  (h1 : let james := 3 + 2 * x in james + x = 36) : 
  x = 11 :=
by 
  -- Proof will go here
  sorry

end jordan_oreos_l58_58128


namespace line_through_circle_center_l58_58817

theorem line_through_circle_center (a : ℝ) :
  (∃ (x y : ℝ), 3 * x + y + a = 0 ∧ x^2 + y^2 + 2 * x - 4 * y = 0) ↔ (a = 1) :=
by
  sorry

end line_through_circle_center_l58_58817


namespace max_marks_mike_could_have_got_l58_58018

theorem max_marks_mike_could_have_got (p : ℝ) (m_s : ℝ) (d : ℝ) (M : ℝ) :
  p = 0.30 → m_s = 212 → d = 13 → 0.30 * M = (212 + 13) → M = 750 :=
by
  intros hp hms hd heq
  sorry

end max_marks_mike_could_have_got_l58_58018


namespace quilt_width_is_eight_l58_58850

def length := 7
def cost_per_square_foot := 40
def total_cost := 2240
def area := total_cost / cost_per_square_foot

theorem quilt_width_is_eight :
  area / length = 8 := by
  sorry

end quilt_width_is_eight_l58_58850


namespace min_hypotenuse_of_right_triangle_l58_58032

theorem min_hypotenuse_of_right_triangle (a b c : ℝ) 
  (h1 : a^2 + b^2 = c^2) 
  (h2 : a + b + c = 6) : 
  c = 6 * (Real.sqrt 2 - 1) :=
sorry

end min_hypotenuse_of_right_triangle_l58_58032


namespace domain_ln_l58_58789

theorem domain_ln (x : ℝ) : x^2 - x - 2 > 0 ↔ (x < -1 ∨ x > 2) := by
  sorry

end domain_ln_l58_58789


namespace surface_area_circumscribed_sphere_l58_58816

theorem surface_area_circumscribed_sphere (a b c : ℝ) (ha : a = 3) (hb : b = 4) (hc : c = 5) :
    4 * Real.pi * ((Real.sqrt (a^2 + b^2 + c^2) / 2)^2) = 50 * Real.pi :=
by
  rw [ha, hb, hc]
  -- prove the equality step-by-step
  sorry

end surface_area_circumscribed_sphere_l58_58816


namespace max_heaps_of_stones_l58_58864

noncomputable def stones : ℕ := 660
def max_heaps : ℕ := 30
def differs_less_than_twice (a b : ℕ) : Prop := a < 2 * b

theorem max_heaps_of_stones (h : ℕ) :
  (∀ i j : ℕ, i ≠ j → differs_less_than_twice (i+j) stones) → max_heaps = 30 :=
sorry

end max_heaps_of_stones_l58_58864


namespace savings_percentage_l58_58217

variables (I S : ℝ)
-- Conditions
-- A man saves a certain portion S of his income I during the first year.
-- He spends the remaining portion (I - S) on his personal expenses.
-- In the second year, his income increases by 50%, so his new income is 1.5I.
-- His savings increase by 100%, so his new savings are 2S.
-- His total expenditure in 2 years is double his expenditure in the first year.

def first_year_expenditure (I S : ℝ) : ℝ := I - S
def second_year_income (I : ℝ) : ℝ := 1.5 * I
def second_year_savings (S : ℝ) : ℝ := 2 * S
def second_year_expenditure (I S : ℝ) : ℝ := second_year_income I - second_year_savings S
def total_expenditure (I S : ℝ) : ℝ := first_year_expenditure I S + second_year_expenditure I S

theorem savings_percentage :
  total_expenditure I S = 2 * first_year_expenditure I S → S / I = 0.5 :=
by
  sorry

end savings_percentage_l58_58217


namespace find_second_number_l58_58187

theorem find_second_number (x : ℝ) 
    (h : (14 + x + 53) / 3 = (21 + 47 + 22) / 3 + 3) : 
    x = 32 := 
by 
    sorry

end find_second_number_l58_58187


namespace squares_with_equal_black_and_white_cells_l58_58680

open Nat

/-- Given a specific coloring of cells in a 5x5 grid, prove that there are
exactly 16 squares that have an equal number of black and white cells. --/
theorem squares_with_equal_black_and_white_cells :
  let gridSize := 5
  let number_of_squares_with_equal_black_and_white_cells := 16
  true := sorry

end squares_with_equal_black_and_white_cells_l58_58680


namespace max_piles_l58_58858

theorem max_piles (n : ℕ) (m : ℕ) : 
  (∀ x y : ℕ, x ∈ n ∧ y ∈ n → x < 2 * y → x > 0) → 
  ( ∑ i in n.to_finset, i) = m → 
  m = 660 →
  n.card = 30 :=
sorry

end max_piles_l58_58858


namespace standard_deviation_does_not_require_repair_l58_58038

-- Definitions based on conditions
def greatest_deviation (d : ℝ) := d = 39
def nominal_mass (M : ℝ) := 0.1 * M = 39
def unreadable_measurement_deviation (d : ℝ) := d < 39

-- Theorems to be proved
theorem standard_deviation (σ : ℝ) (d : ℝ) (M : ℝ) :
  greatest_deviation d →
  nominal_mass M →
  unreadable_measurement_deviation d →
  σ ≤ 39 :=
by
  sorry

theorem does_not_require_repair (σ : ℝ) :
  σ ≤ 39 → ¬(machine_requires_repair) :=
by
  sorry

-- Adding an assumption that if σ ≤ 39, the machine does not require repair
axiom machine_requires_repair : Prop

end standard_deviation_does_not_require_repair_l58_58038


namespace highest_certificate_probability_probability_exactly_two_l58_58879

/-- Probabilities of passing the theoretical exam for A, B, and C --/
def P_theoretical : ℕ → ℝ
| 1 := 4/5
| 2 := 3/4
| 3 := 2/3
| _ := 0

/-- Probabilities of passing the practical operation exam for A, B, and C --/
def P_practical : ℕ → ℝ
| 1 := 1/2
| 2 := 2/3
| 3 := 5/6
| _ := 0

/-- Probabilities of obtaining the "certificate of passing" for A, B, and C --/
def P_certificate (n : ℕ) : ℝ :=
  P_theoretical n * P_practical n

/-- Probabilities of passing both exams for A, B, and C --/
theorem highest_certificate_probability : 
  P_certificate 3 > P_certificate 2 ∧ P_certificate 2 > P_certificate 1 :=
by
  sorry

/-- Probability that exactly two out of A, B, and C obtain the "certificate of passing" --/
def P_exactly_two_pass : ℝ :=
  P_certificate 1 * P_certificate 2 * (1 - P_certificate 3) +
  P_certificate 1 * (1 - P_certificate 2) * P_certificate 3 +
  (1 - P_certificate 1) * P_certificate 2 * P_certificate 3

theorem probability_exactly_two : 
  P_exactly_two_pass = 11/30 :=
by
  sorry

end highest_certificate_probability_probability_exactly_two_l58_58879


namespace depth_notation_l58_58825

-- Definition of depth and altitude
def above_sea_level (h : ℝ) : Prop := h > 0 
def below_sea_level (h : ℝ) : Prop := h < 0 

-- Given conditions
axiom height_Dabaijing : above_sea_level 9050
axiom depth_Haidou1 : below_sea_level (-10907)

-- Proof goal
theorem depth_notation :
  ∀ (d : ℝ), above_sea_level 9050 → below_sea_level (-d) → d = 10907 :=
by
  intros d _ _
  exact sorry

end depth_notation_l58_58825


namespace find_root_and_m_l58_58668

theorem find_root_and_m {x : ℝ} {m : ℝ} (h : ∃ x1 x2 : ℝ, (x1 = 1) ∧ (x1 + x2 = -m) ∧ (x1 * x2 = 3)) :
  ∃ x2 : ℝ, (x2 = 3) ∧ (m = -4) :=
by
  obtain ⟨x1, x2, h1, h_sum, h_product⟩ := h
  have hx1 : x1 = 1 := h1
  rw [hx1] at h_product
  have hx2 : x2 = 3 := by linarith [h_product]
  have hm : m = -4 := by
    rw [hx1, hx2] at h_sum
    linarith
  exact ⟨x2, hx2, hm⟩

end find_root_and_m_l58_58668


namespace f_at_2_f_shifted_range_f_shifted_l58_58520

def f (x : ℝ) := x^2 - 2*x + 7

-- 1) Prove that f(2) = 7
theorem f_at_2 : f 2 = 7 := sorry

-- 2) Prove the expressions for f(x-1) and f(x+1)
theorem f_shifted (x : ℝ) : f (x-1) = x^2 - 4*x + 10 ∧ f (x+1) = x^2 + 6 := sorry

-- 3) Prove the range of f(x+1) is [6, +∞)
theorem range_f_shifted : ∀ x, f (x+1) ≥ 6 := sorry

end f_at_2_f_shifted_range_f_shifted_l58_58520


namespace factorize_poly1_factorize_poly2_l58_58722

-- Define y substitution for first problem
def poly1_y := fun (x : ℝ) => x^2 + 2*x
-- Define y substitution for second problem
def poly2_y := fun (x : ℝ) => x^2 - 4*x

-- Define the given polynomial expressions 
def poly1 := fun (x : ℝ) => (x^2 + 2*x)*(x^2 + 2*x + 2) + 1
def poly2 := fun (x : ℝ) => (x^2 - 4*x)*(x^2 - 4*x + 8) + 16

theorem factorize_poly1 (x : ℝ) : poly1 x = (x + 1) ^ 4 := sorry

theorem factorize_poly2 (x : ℝ) : poly2 x = (x - 2) ^ 4 := sorry

end factorize_poly1_factorize_poly2_l58_58722


namespace smallest_non_palindromic_power_of_13_l58_58650

def is_palindrome (n : ℕ) : Prop :=
  let s := n.to_string
  s = s.reverse

def is_power_of_13 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 13 ^ k

theorem smallest_non_palindromic_power_of_13
  (smallest_non_palindrome_power : ℕ)
  (h_p13 : is_power_of_13 smallest_non_palindrome_power)
  (h_palindrome : ¬ is_palindrome smallest_non_palindrome_power)
  (h_smallest : ∀ m : ℕ, is_power_of_13 m ∧ ¬ is_palindrome m → smallest_non_palindrome_power ≤ m)
  : smallest_non_palindrome_power = 2197 :=
sorry

end smallest_non_palindromic_power_of_13_l58_58650


namespace twelfth_term_l58_58893

-- Definitions based on the given conditions
def a_3_condition (a d : ℚ) : Prop := a + 2 * d = 10
def a_6_condition (a d : ℚ) : Prop := a + 5 * d = 20

-- The main theorem stating that the twelfth term is 40
theorem twelfth_term (a d : ℚ) (h1 : a_3_condition a d) (h2 : a_6_condition a d) :
  a + 11 * d = 40 :=
sorry

end twelfth_term_l58_58893


namespace employee_payments_l58_58207

theorem employee_payments :
  ∃ (A B C : ℤ), A = 900 ∧ B = 600 ∧ C = 500 ∧
    A + B + C = 2000 ∧
    A = 3 * B / 2 ∧
    C = 400 + 100 := 
by
  sorry

end employee_payments_l58_58207


namespace find_number_l58_58896

theorem find_number (x : ℝ) (h : x / 5 + 23 = 42) : x = 95 :=
by
  -- Proof placeholder
  sorry

end find_number_l58_58896


namespace problem1_problem2_l58_58638

-- Define the base types and expressions
variables (x m : ℝ)

-- Proofs of the given expressions
theorem problem1 : (x^7 / x^3) * x^4 = x^8 :=
by sorry

theorem problem2 : m * m^3 + ((-m^2)^3 / m^2) = 0 :=
by sorry

end problem1_problem2_l58_58638


namespace diana_age_l58_58639

open Classical

theorem diana_age :
  ∃ (D : ℚ), (∃ (C E : ℚ), C = 4 * D ∧ E = D + 5 ∧ C = E) ∧ D = 5/3 :=
by
  -- Definitions and conditions are encapsulated in the existential quantifiers and the proof concludes with D = 5/3.
  sorry

end diana_age_l58_58639


namespace simplify_abs_eq_l58_58545

variable {x : ℚ}

theorem simplify_abs_eq (hx : |1 - x| = 1 + |x|) : |x - 1| = 1 - x :=
by
  sorry

end simplify_abs_eq_l58_58545


namespace trajectory_equation_l58_58432

-- Define the condition that the distance to the coordinate axes is equal.
def equidistantToAxes (x y : ℝ) : Prop :=
  abs x = abs y

-- State the theorem that we need to prove.
theorem trajectory_equation (x y : ℝ) (h : equidistantToAxes x y) : y^2 = x^2 :=
by sorry

end trajectory_equation_l58_58432


namespace probability_no_adjacent_balls_is_correct_l58_58359

open Finset

noncomputable def probability_no_adjacent_bins : ℚ :=
  let total_ways := (choose 20 5 : ℕ) 
  let valid_ways := finset.sum (finset.range 6) (λ k, (-1 : ℤ)^k * (choose 5 k) * (choose 15 (5 - k))) 
  (valid_ways : ℚ) / (total_ways : ℚ)

theorem probability_no_adjacent_balls_is_correct :
  probability_no_adjacent_bins = (15504 / 18801 : ℚ) :=
sorry

end probability_no_adjacent_balls_is_correct_l58_58359


namespace number_of_tiles_is_47_l58_58748

theorem number_of_tiles_is_47 : 
  ∃ (n : ℕ), (n % 2 = 1) ∧ (n % 3 = 2) ∧ (n % 5 = 2) ∧ n = 47 :=
by
  sorry

end number_of_tiles_is_47_l58_58748


namespace tangent_line_to_curve_l58_58194

theorem tangent_line_to_curve (a : ℝ) : (∀ (x : ℝ), y = x → y = a + Real.log x) → a = 1 := 
sorry

end tangent_line_to_curve_l58_58194


namespace arithmetic_sequence_equal_sum_l58_58794

variable (a d : ℕ) -- defining first term and common difference as natural numbers
variable (n : ℕ) -- defining n as a natural number

noncomputable def sum_arithmetic_sequence (n: ℕ) (a d: ℕ): ℕ := (n * (2 * a + (n - 1) * d) ) / 2

theorem arithmetic_sequence_equal_sum (a d n : ℕ) :
  sum_arithmetic_sequence (10 * n) a d = sum_arithmetic_sequence (15 * n) a d - sum_arithmetic_sequence (10 * n) a d :=
by
  sorry

end arithmetic_sequence_equal_sum_l58_58794


namespace remainder_division_l58_58598
-- Import the necessary library

-- Define the number and the divisor
def number : ℕ := 2345678901
def divisor : ℕ := 101

-- State the theorem
theorem remainder_division : number % divisor = 23 :=
by sorry

end remainder_division_l58_58598


namespace cube_vertex_condition_l58_58797

noncomputable def possible_cube_vertex_counts (k : ℕ) : Prop :=
k ∈ {6, 7, 8}

theorem cube_vertex_condition (k : ℕ) (Hk : 2 ≤ k)
  (M : Finset (Fin 8))
  (hM: M.card = k)
  (h : ∀ x1 x2 ∈ M, ∃ y1 y2 ∈ M, (x1 ≠ x2) → (x1.valuation + x2.valuation + y1.valuation + y2.valuation = 12)) : 
  possible_cube_vertex_counts k :=
sorry

end cube_vertex_condition_l58_58797


namespace denote_depth_below_sea_level_l58_58838

theorem denote_depth_below_sea_level (above_sea_level : Int) (depth_haidou_1 : Int) :
  (above_sea_level > 0) ∧ (depth_haidou_1 < 0) → depth_haidou_1 = -10907 :=
by
  intros h
  sorry

end denote_depth_below_sea_level_l58_58838


namespace find_S25_l58_58664

variables (a : ℕ → ℝ) (S : ℕ → ℝ)

-- Definitions based on conditions
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop := ∀ n, a (n + 1) - a n = a 1 - a 0
def sum_of_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop := ∀ n, S n = (n / 2) * (2 * a 0 + (n - 1) * (a 1 - a 0))

-- Condition that given S_{15} - S_{10} = 1
axiom sum_difference : S 15 - S 10 = 1

-- Theorem we need to prove
theorem find_S25 (h_arith : is_arithmetic_sequence a) (h_sum : sum_of_first_n_terms a S) : S 25 = 5 :=
by
-- Placeholder for the actual proof
sorry

end find_S25_l58_58664


namespace no_positive_integral_solutions_l58_58041

theorem no_positive_integral_solutions (x y : ℕ) (h : x > 0) (k : y > 0) :
  x^4 * y^4 - 8 * x^2 * y^2 + 12 ≠ 0 :=
by
  sorry

end no_positive_integral_solutions_l58_58041


namespace integer_values_of_a_l58_58083

theorem integer_values_of_a : 
  ∃ (a : Set ℤ), (∀ x, x ∈ a → ∃ (y z : ℤ), x^2 + x * y + 9 * y = 0) ∧ (a.card = 6) :=
by
  sorry

end integer_values_of_a_l58_58083


namespace face_value_of_each_ticket_without_tax_l58_58467

theorem face_value_of_each_ticket_without_tax (total_people : ℕ) (total_cost : ℝ) (sales_tax : ℝ) (face_value : ℝ)
  (h1 : total_people = 25)
  (h2 : total_cost = 945)
  (h3 : sales_tax = 0.05)
  (h4 : total_cost = (1 + sales_tax) * face_value * total_people) :
  face_value = 36 := by
  sorry

end face_value_of_each_ticket_without_tax_l58_58467


namespace smallest_number_of_slices_l58_58867

-- Definition of the number of slices in each type of cheese package
def slices_of_cheddar : ℕ := 12
def slices_of_swiss : ℕ := 28

-- Predicate stating that the smallest number of slices of each type Randy could have bought is 84
theorem smallest_number_of_slices : Nat.lcm slices_of_cheddar slices_of_swiss = 84 := by
  sorry

end smallest_number_of_slices_l58_58867


namespace days_collected_money_l58_58017

-- Defining constants and parameters based on the conditions
def households_per_day : ℕ := 20
def money_per_pair : ℕ := 40
def total_money_collected : ℕ := 2000
def money_from_households : ℕ := (households_per_day / 2) * money_per_pair

-- The theorem that needs to be proven
theorem days_collected_money :
  (total_money_collected / money_from_households) = 5 :=
sorry -- Proof not provided

end days_collected_money_l58_58017


namespace garage_sale_items_l58_58242

theorem garage_sale_items (h : 34 = 13 + n + 1 + 14 - 14) : n = 22 := by
  sorry

end garage_sale_items_l58_58242


namespace find_number_l58_58300

theorem find_number (n : ℤ) (h : 7 * n = 3 * n + 12) : n = 3 :=
sorry

end find_number_l58_58300


namespace quadratic_condition_l58_58200

theorem quadratic_condition (a : ℝ) : 
  (∀ x : ℝ, a * x^2 - 2 * a * x + 3 > 0) ↔ (0 ≤ a ∧ a < 3) :=
sorry

end quadratic_condition_l58_58200


namespace remaining_pencils_total_l58_58293

-- Definitions corresponding to the conditions:
def J : ℝ := 300
def J_d : ℝ := 0.30 * J
def J_r : ℝ := J - J_d

def V : ℝ := 2 * J
def V_d : ℝ := 125
def V_r : ℝ := V - V_d

def S : ℝ := 450
def S_d : ℝ := 0.60 * S
def S_r : ℝ := S - S_d

-- Proving the remaining pencils add up to the required amount:
theorem remaining_pencils_total : J_r + V_r + S_r = 865 := by
  sorry

end remaining_pencils_total_l58_58293


namespace tangent_line_at_origin_increasing_intervals_l58_58375

-- Conditions and function definition
variable (a : ℝ) (h₁ : a ≠ -1)
def f (a : ℝ) : ℝ → ℝ := λ x, (x - 1) / (x + a) + Real.log (x + 1)

-- Proof problem 1: Tangent line equation at (0, f(0)) when a = 2
theorem tangent_line_at_origin (h₂ : a = 2) : 7 * x - 4 * y - 2 = 0 :=
sorry

-- Proof problem 2: Intervals of monotonic increase if f(x) has an extremum at x = 1
theorem increasing_intervals (h₃ : has_extremum_at (f a) 1) (ha : a = -3) :
  (-1 < x ∧ x ≤ 1) ∨ (7 ≤ x) :=
sorry

end tangent_line_at_origin_increasing_intervals_l58_58375


namespace crayons_total_l58_58546

theorem crayons_total (blue red green : ℕ) 
  (h1 : red = 4 * blue) 
  (h2 : green = 2 * red) 
  (h3 : blue = 3) : 
  blue + red + green = 39 := 
by
  sorry

end crayons_total_l58_58546


namespace number_in_pattern_l58_58070

theorem number_in_pattern (m n : ℕ) (h : 8 * m - 5 = 2023) (hn : n = 5) : m + n = 258 :=
by
  sorry

end number_in_pattern_l58_58070


namespace intersection_M_N_l58_58672

noncomputable def M : Set ℝ := { x | x^2 + x - 2 = 0 }
def N : Set ℝ := { x | x < 0 }

theorem intersection_M_N : M ∩ N = { -2 } := by
  sorry

end intersection_M_N_l58_58672


namespace find_function_p_t_additional_hours_l58_58622

variable (p0 : ℝ) (t k : ℝ)

-- Given condition: initial concentration decreased by 1/5 after one hour
axiom filtration_condition_1 : (p0 * ((4 : ℝ) / 5) ^ t = p0 * (Real.exp (-k * t)))
axiom filtration_condition_2 : (p0 * ((4 : ℝ) / 5) = p0 * (Real.exp (-k)))

-- Problem 1: Find the function p(t)
theorem find_function_p_t : ∃ k, ∀ t, p0 * ((4 : ℝ) / 5) ^ t = p0 * (Real.exp (-k * t)) := by
  sorry

-- Problem 2: Find the additional hours of filtration needed
theorem additional_hours (h : ∀ t, p0 * ((4 : ℝ) / 5) ^ t = p0 * (Real.exp (-k * t))) :
  ∀ t, p0 * ((4 : ℝ) / 5) ^ t ≤ (p0 / 1000) → t ≥ 30 := by
  sorry

end find_function_p_t_additional_hours_l58_58622


namespace find_a_m_range_c_l58_58377

noncomputable def f (x a : ℝ) := x^2 - 2*x + 2*a
def solution_set (f : ℝ → ℝ) (m : ℝ) := {x : ℝ | -2 ≤ x ∧ x ≤ m ∧ f x ≤ 0}

theorem find_a_m (a m : ℝ) : 
  (∀ x, f x a ≤ 0 ↔ -2 ≤ x ∧ x ≤ m) → a = -4 ∧ m = 4 := by
  sorry

theorem range_c (c : ℝ) : 
  (∀ x, (c - 4) * x^2 + 2 * (c - 4) * x - 1 < 0) → 13 / 4 < c ∧ c < 4 := by
  sorry

end find_a_m_range_c_l58_58377


namespace GCD_of_n_pow_13_sub_n_l58_58079

theorem GCD_of_n_pow_13_sub_n :
  ∀ n : ℤ, gcd (n^13 - n) 2730 = gcd (n^13 - n) n := sorry

end GCD_of_n_pow_13_sub_n_l58_58079


namespace range_g_l58_58355

noncomputable def g (x : ℝ) : ℝ := 1 / (x - 1)^2

theorem range_g : set.range (λ x, g x) = set.Ioi 0 := 
sorry

end range_g_l58_58355


namespace inequality_with_conditions_l58_58294

variable {a b c : ℝ}

theorem inequality_with_conditions (h : a * b + b * c + c * a = 1) :
  (|a - b| / |1 + c^2|) + (|b - c| / |1 + a^2|) ≥ (|c - a| / |1 + b^2|) :=
by
  sorry

end inequality_with_conditions_l58_58294


namespace function_property_l58_58691

theorem function_property (k a b : ℝ) (h : a ≠ b) (h_cond : k * a + 1 / b = k * b + 1 / a) : 
  k * (a * b) + 1 = 0 := 
by 
  sorry

end function_property_l58_58691


namespace inverse_of_B_squared_l58_58965

noncomputable def B_inv : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![2, -3, 0], ![0, -1, 0], ![0, 0, 5]]

theorem inverse_of_B_squared :
  (B_inv * B_inv) = ![![4, -3, 0], ![0, 1, 0], ![0, 0, 25]] := by
  sorry

end inverse_of_B_squared_l58_58965


namespace Priya_time_l58_58572

noncomputable def Suresh_rate : ℚ := 1 / 15
noncomputable def Ashutosh_rate : ℚ := 1 / 20
noncomputable def Priya_rate : ℚ := 1 / 25

noncomputable def Suresh_work : ℚ := 6 * Suresh_rate
noncomputable def Ashutosh_work : ℚ := 8 * Ashutosh_rate
noncomputable def total_work_done : ℚ := Suresh_work + Ashutosh_work
noncomputable def remaining_work : ℚ := 1 - total_work_done

theorem Priya_time : 
  remaining_work = Priya_rate * 5 := 
by sorry

end Priya_time_l58_58572


namespace find_f_prime_one_l58_58961

theorem find_f_prime_one (f : ℝ → ℝ) (f' : ℝ → ℝ) (h : ∀ x, f' x = 2 * f' 1 + 1 / x) (h_fx : ∀ x, f x = 2 * x * f' 1 + Real.log x) : f' 1 = -1 := 
by 
  sorry

end find_f_prime_one_l58_58961


namespace find_positive_number_l58_58450

theorem find_positive_number (n : ℕ) (h : n^2 + n = 210) : n = 14 :=
sorry

end find_positive_number_l58_58450


namespace line_through_fixed_point_and_equation_l58_58096

/-- Definitions and conditions -/
def circle (x y : ℝ) : Prop := x^2 + (y - 1)^2 = 5
def line (m x y : ℝ) : Prop := m * x - y + 1 - m = 0
def distance_between_points (A B : ℝ × ℝ) : ℝ := real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

/-- Main theorem statement -/

theorem line_through_fixed_point_and_equation {m x y : ℝ} {A B : ℝ × ℝ} :
  (∀ m, (∃ x y, line m x y ∧ x = 1 ∧ y = 1)) ∧
  (distance_between_points A B = real.sqrt 17 →
  ∃ m, line m x y ∧ (line m = λ x y, sqrt 3 * x - y + 1 - sqrt 3) ∨
                  (line m = λ x y, -sqrt 3 * x + y + 1 - sqrt 3)) :=
by
  split
  · -- proof that line l always passes through (1,1)
    sorry
  · -- proof to find the required line equation(s)
    sorry

end line_through_fixed_point_and_equation_l58_58096


namespace calculate_expression_l58_58637

theorem calculate_expression 
  (a1 : 84 + 4 / 19 = 1600 / 19) 
  (a2 : 105 + 5 / 19 = 2000 / 19) 
  (a3 : 1.375 = 11 / 8) 
  (a4 : 0.8 = 4 / 5) :
  84 * (4 / 19) * (11 / 8) + 105 * (5 / 19) * (4 / 5) = 200 := 
sorry

end calculate_expression_l58_58637


namespace sectorChordLength_correct_l58_58573

open Real

noncomputable def sectorChordLength (r α : ℝ) : ℝ :=
  2 * r * sin (α / 2)

theorem sectorChordLength_correct :
  ∃ (r α : ℝ), (1/2) * α * r^2 = 1 ∧ 2 * r + α * r = 4 ∧ sectorChordLength r α = 2 * sin 1 :=
by {
  sorry
}

end sectorChordLength_correct_l58_58573


namespace sum_of_reciprocals_l58_58614

theorem sum_of_reciprocals
  (m n p : ℕ)
  (HCF_mnp : Nat.gcd (Nat.gcd m n) p = 26)
  (LCM_mnp : Nat.lcm (Nat.lcm m n) p = 6930)
  (sum_mnp : m + n + p = 150) :
  (1 / (m : ℚ) + 1 / (n : ℚ) + 1 / (p : ℚ) = 1 / 320166) :=
by
  sorry

end sum_of_reciprocals_l58_58614


namespace second_train_speed_l58_58210

theorem second_train_speed
  (v : ℕ)
  (h1 : 8 * v - 8 * 11 = 160) :
  v = 31 :=
sorry

end second_train_speed_l58_58210


namespace max_value_amc_am_mc_ca_l58_58134

theorem max_value_amc_am_mc_ca (A M C : ℕ) 
  (h : A + M + C = 15) : 
  A * M * C + A * M + M * C + C * A ≤ 200 := 
sorry

end max_value_amc_am_mc_ca_l58_58134


namespace ball_reaches_20_feet_at_1_75_seconds_l58_58577

noncomputable def ball_height (t : ℝ) : ℝ :=
  60 - 9 * t - 8 * t ^ 2

theorem ball_reaches_20_feet_at_1_75_seconds :
  ∃ t : ℝ, ball_height t = 20 ∧ t = 1.75 ∧ t ≥ 0 :=
by {
  sorry
}

end ball_reaches_20_feet_at_1_75_seconds_l58_58577


namespace weight_of_b_l58_58189

theorem weight_of_b (A B C : ℝ) : 
  (A + B + C = 135) → 
  (A + B = 80) → 
  (B + C = 86) → 
  B = 31 :=
by {
  intros h1 h2 h3,
  sorry
}

end weight_of_b_l58_58189


namespace max_heaps_660_l58_58857

def max_heaps (heaps : List ℕ) (n : ℕ) : Prop :=
  ∑ h in heaps, h = n ∧ ∀ (a b : ℕ), a ∈ heaps → b ∈ heaps → a ≠ b → (a < 2 * b ∧ b < 2 * a)

theorem max_heaps_660 : ∃ heaps, max_heaps heaps 660 ∧ length heaps = 30 :=
begin
  sorry
end

end max_heaps_660_l58_58857


namespace find_x_l58_58792

theorem find_x (x : ℝ) (h1 : x > 0) (h2 : 1/2 * (2 * x) * x = 72) : x = 6 * Real.sqrt 2 :=
by
  sorry

end find_x_l58_58792


namespace octahedron_non_blue_probability_l58_58464

theorem octahedron_non_blue_probability :
  let total_faces := 8
  let blue_faces := 3
  let red_faces := 3
  let green_faces := 2
  let non_blue_faces := total_faces - blue_faces
  (non_blue_faces / total_faces : ℚ) = (5 / 8 : ℚ) :=
by
  sorry

end octahedron_non_blue_probability_l58_58464


namespace ratio_shorter_longer_l58_58058

theorem ratio_shorter_longer (total_length shorter_length longer_length : ℝ)
  (h1 : total_length = 21) 
  (h2 : shorter_length = 6) 
  (h3 : longer_length = total_length - shorter_length) 
  (h4 : shorter_length / longer_length = 2 / 5) : 
  shorter_length / longer_length = 2 / 5 :=
by sorry

end ratio_shorter_longer_l58_58058


namespace minimum_t_is_2_l58_58095

noncomputable def minimum_t_value (t : ℝ) : Prop :=
  let A := (-t, 0)
  let B := (t, 0)
  let C := (Real.sqrt 3, Real.sqrt 6)
  let r := 1
  ∃ P : ℝ × ℝ, 
    (P.1 - (Real.sqrt 3))^2 + (P.2 - (Real.sqrt 6))^2 = r^2 ∧ 
    (P.1 - A.1) * (P.1 - B.1) + (P.2 - A.2) * (P.2 - B.2) = 0

theorem minimum_t_is_2 : (∃ t : ℝ, t > 0 ∧ minimum_t_value t) → ∃ t : ℝ, t = 2 :=
sorry

end minimum_t_is_2_l58_58095


namespace spherical_coordinates_cone_l58_58081

open Real

-- Define spherical coordinates and the equation φ = c
def spherical_coordinates (ρ θ φ : ℝ) : Prop := 
  ∃ (c : ℝ), φ = c

-- Prove that φ = c describes a cone
theorem spherical_coordinates_cone (ρ θ : ℝ) (c : ℝ) :
  spherical_coordinates ρ θ c → ∃ ρ' θ', spherical_coordinates ρ' θ' c :=
by
  sorry

end spherical_coordinates_cone_l58_58081


namespace elvins_first_month_bill_l58_58504

theorem elvins_first_month_bill (F C : ℝ) 
  (h1 : F + C = 52)
  (h2 : F + 2 * C = 76) : 
  F + C = 52 :=
by
  sorry

end elvins_first_month_bill_l58_58504


namespace number_of_friends_l58_58424

def money_emma : ℕ := 8

def money_daya : ℕ := money_emma + (money_emma * 25 / 100)

def money_jeff : ℕ := (2 * money_daya) / 5

def money_brenda : ℕ := money_jeff + 4

def money_brenda_condition : Prop := money_brenda = 8

def friends_pooling_pizza : ℕ := 4

theorem number_of_friends (h : money_brenda_condition) : friends_pooling_pizza = 4 := by
  sorry

end number_of_friends_l58_58424


namespace Petya_rubles_maximum_l58_58705

theorem Petya_rubles_maximum :
  ∃ n, (2000 ≤ n ∧ n < 2100) ∧ (∀ d ∈ [1, 3, 5, 7, 9, 11], n % d = 0 → true) → 
  (1 + ite (n % 1 = 0) 1 0 + ite (n % 3 = 0) 3 0 + ite (n % 5 = 0) 5 0 + ite (n % 7 = 0) 7 0 + ite (n % 9 = 0) 9 0 + ite (n % 11 = 0) 11 0 = 31) :=
begin
  sorry
end

end Petya_rubles_maximum_l58_58705


namespace tomTotalWeightMoved_is_525_l58_58900

-- Tom's weight
def tomWeight : ℝ := 150

-- Weight in each hand
def weightInEachHand : ℝ := 1.5 * tomWeight

-- Weight vest
def weightVest : ℝ := 0.5 * tomWeight

-- Total weight moved
def totalWeightMoved : ℝ := (weightInEachHand * 2) + weightVest

theorem tomTotalWeightMoved_is_525 : totalWeightMoved = 525 := by
  sorry

end tomTotalWeightMoved_is_525_l58_58900


namespace rationalize_fraction_l58_58874

theorem rationalize_fraction :
  (5 : ℚ) / (Real.sqrt 50 + 3 * Real.sqrt 8 + Real.sqrt 18 + Real.sqrt 32) = 
  (5 * Real.sqrt 2) / 36 :=
by
  sorry

end rationalize_fraction_l58_58874


namespace expression_simplifies_l58_58420

variable {a b : ℚ}
variable (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a ≠ b)

theorem expression_simplifies : (a^2 - b^2) / (a * b) - (a * b - b^2) / (a * b - a^2) = a / b := by
  -- TODO: Proof goes here
  sorry

end expression_simplifies_l58_58420


namespace sequence_arithmetic_mean_l58_58367

theorem sequence_arithmetic_mean (a b c d e f g : ℝ)
  (h1 : b = (a + c) / 2)
  (h2 : c = (b + d) / 2)
  (h3 : d = (c + e) / 2)
  (h4 : e = (d + f) / 2)
  (h5 : f = (e + g) / 2) :
  d = (a + g) / 2 :=
sorry

end sequence_arithmetic_mean_l58_58367


namespace sum_of_nonzero_perfect_squares_l58_58661

theorem sum_of_nonzero_perfect_squares (p n : ℕ) (hp_prime : Nat.Prime p) 
    (hn_ge_p : n ≥ p) (h_perfect_square : ∃ k : ℕ, 1 + n * p = k^2) :
    ∃ (a : ℕ) (f : Fin p → ℕ), (∀ i, 0 < f i ∧ ∃ m, f i = m^2) ∧ (n + 1 = a + (Finset.univ.sum f)) :=
sorry

end sum_of_nonzero_perfect_squares_l58_58661


namespace max_value_expression_l58_58160

theorem max_value_expression (A M C : ℕ) (h₁ : A + M + C = 15) : 
  A * M * C + A * M + M * C + C * A ≤ 200 :=
  sorry

end max_value_expression_l58_58160


namespace solve_for_x_l58_58912

theorem solve_for_x (x : ℝ) (h : 5 / (4 + 1 / x) = 1) : x = 1 :=
by
  sorry

end solve_for_x_l58_58912


namespace variance_of_data_set_l58_58662

theorem variance_of_data_set :
  let data_set := [2, 3, 4, 5, 6]
  let mean := (2 + 3 + 4 + 5 + 6) / 5
  let variance := (1 / 5 : Real) * ((2 - mean)^2 + (3 - mean)^2 + (4 - mean)^2 + (5 - mean)^2 + (6 - mean)^2)
  variance = 2 :=
by
  sorry

end variance_of_data_set_l58_58662


namespace diameter_of_lake_l58_58544

-- Given conditions: the radius of the circular lake
def radius : ℝ := 7

-- The proof problem: proving the diameter of the lake is 14 meters
theorem diameter_of_lake : 2 * radius = 14 :=
by
  sorry

end diameter_of_lake_l58_58544


namespace valid_selections_one_female_l58_58203

-- Define the conditions
def GroupA_males : ℕ := 5
def GroupA_females : ℕ := 3
def GroupB_males : ℕ := 6
def GroupB_females : ℕ := 2
def students_selected_each_group : ℕ := 2

-- Define the selection problem and prove that the number of valid selections is 345
theorem valid_selections_one_female :
  let scenario1 := choose GroupA_males 1 * choose GroupA_females 1 * choose GroupB_males 2,
      scenario2 := choose GroupA_males 2 * choose GroupB_males 1 * choose GroupB_females 1
  in scenario1 + scenario2 = 345 := by
  sorry

end valid_selections_one_female_l58_58203


namespace find_s_of_2_l58_58559

def t (x : ℝ) : ℝ := 4 * x - 9
def s (x : ℝ) : ℝ := x^2 + 4 * x - 1

theorem find_s_of_2 : s (2) = 281 / 16 :=
by
  sorry

end find_s_of_2_l58_58559


namespace a_share_is_2500_l58_58067

theorem a_share_is_2500
  (x : ℝ)
  (h1 : 4 * x = 3 * x + 500)
  (h2 : 6 * x = 2 * 2 * x) : 5 * x = 2500 :=
by 
  sorry

end a_share_is_2500_l58_58067


namespace cos_pi_plus_alpha_l58_58264

theorem cos_pi_plus_alpha (α : ℝ) (h : Real.sin (π / 2 + α) = 1 / 3) : Real.cos (π + α) = - 1 / 3 :=
by
  sorry

end cos_pi_plus_alpha_l58_58264


namespace amount_of_p_l58_58469

theorem amount_of_p (p q r : ℝ) (h1 : q = (1 / 6) * p) (h2 : r = (1 / 6) * p) 
  (h3 : p = (q + r) + 32) : p = 48 :=
by
  sorry

end amount_of_p_l58_58469


namespace range_of_k_l58_58541

noncomputable def f (k x : ℝ) := (k * x + 7) / (k * x^2 + 4 * k * x + 3)

theorem range_of_k (k : ℝ) : (∀ x : ℝ, k * x^2 + 4 * k * x + 3 ≠ 0) ↔ 0 ≤ k ∧ k < 3 / 4 :=
by
  sorry

end range_of_k_l58_58541


namespace hilton_final_marbles_l58_58270

theorem hilton_final_marbles :
  let initial_marbles := 26
  let found_marbles := 6
  let lost_marbles := 10
  let gift_multiplication_factor := 2
  let marbles_after_find_and_lose := initial_marbles + found_marbles - lost_marbles
  let gift_marbles := gift_multiplication_factor * lost_marbles
  let final_marbles := marbles_after_find_and_lose + gift_marbles
  final_marbles = 42 :=
by
  -- Proof to be filled
  sorry

end hilton_final_marbles_l58_58270


namespace cup_of_coffee_price_l58_58978

def price_cheesecake : ℝ := 10
def price_set : ℝ := 12
def discount : ℝ := 0.75

theorem cup_of_coffee_price (C : ℝ) (h : price_set = discount * (C + price_cheesecake)) : C = 6 :=
by
  sorry

end cup_of_coffee_price_l58_58978


namespace milk_cost_is_3_l58_58849

def Banana_cost : ℝ := 2
def Sales_tax_rate : ℝ := 0.20
def Total_spent : ℝ := 6

theorem milk_cost_is_3 (Milk_cost : ℝ) :
  Total_spent = (Milk_cost + Banana_cost) + Sales_tax_rate * (Milk_cost + Banana_cost) → 
  Milk_cost = 3 :=
by
  simp [Banana_cost, Sales_tax_rate, Total_spent]
  sorry

end milk_cost_is_3_l58_58849


namespace modulus_product_eq_sqrt_5_l58_58436

open Complex

-- Define the given complex number.
def z : ℂ := 2 + I

-- Declare the product with I.
def z_product := z * I

-- State the theorem that the modulus of the product is sqrt(5).
theorem modulus_product_eq_sqrt_5 : abs z_product = Real.sqrt 5 := 
sorry

end modulus_product_eq_sqrt_5_l58_58436


namespace depth_notation_l58_58826

-- Definition of depth and altitude
def above_sea_level (h : ℝ) : Prop := h > 0 
def below_sea_level (h : ℝ) : Prop := h < 0 

-- Given conditions
axiom height_Dabaijing : above_sea_level 9050
axiom depth_Haidou1 : below_sea_level (-10907)

-- Proof goal
theorem depth_notation :
  ∀ (d : ℝ), above_sea_level 9050 → below_sea_level (-d) → d = 10907 :=
by
  intros d _ _
  exact sorry

end depth_notation_l58_58826


namespace range_of_k_l58_58818

theorem range_of_k
  (x y k : ℝ)
  (h1 : 3 * x + y = k + 1)
  (h2 : x + 3 * y = 3)
  (h3 : 0 < x + y)
  (h4 : x + y < 1) :
  -4 < k ∧ k < 0 :=
sorry

end range_of_k_l58_58818


namespace air_conditioning_price_november_l58_58548

noncomputable def price_in_november : ℝ :=
  let january_price := 470
  let february_price := january_price * (1 - 0.12)
  let march_price := february_price * (1 + 0.08)
  let april_price := march_price * (1 - 0.10)
  let june_price := april_price * (1 + 0.05)
  let august_price := june_price * (1 - 0.07)
  let october_price := august_price * (1 + 0.06)
  october_price * (1 - 0.15)

theorem air_conditioning_price_november : price_in_november = 353.71 := by
  sorry

end air_conditioning_price_november_l58_58548


namespace monotonicity_and_range_of_m_l58_58411

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  (1 - a) / 2 * x ^ 2 + a * x - Real.log x

theorem monotonicity_and_range_of_m (a m : ℝ) (h₀ : 2 < a) (h₁ : a < 3)
  (h₂ : ∀ (x1 x2 : ℝ), 1 ≤ x1 ∧ x1 ≤ 2 → 1 ≤ x2 ∧ x2 ≤ 2 -> ma + Real.log 2 > |f x1 a - f x2 a|):
  m ≥ 0 :=
sorry

end monotonicity_and_range_of_m_l58_58411


namespace valid_N_count_l58_58776

def sum_T (N_7 N_8 : ℕ) := N_7 + N_8

noncomputable def valid_Ns : ℕ := 22

theorem valid_N_count :
  ∃ N : ℕ, 1000 ≤ N ∧ N < 10000 ∧
           ∃ N_7 N_8 : ℕ, (N_7 = N.to_base 7) ∧ (N_8 = N.to_base 8) ∧
           sum_T N_7 N_8 % 1000 = 4 * N % 1000 ∧
           (∀ n, 1000 ≤ n ∧ n < 10000 ∧
              ∃ n_7 n_8 : ℕ, (n_7 = n.to_base 7) ∧ (n_8 = n.to_base 8) ∧
              sum_T n_7 n_8 % 1000 = 4 * n % 1000 → n ∈ finset.range valid_Ns) :=
sorry

end valid_N_count_l58_58776


namespace positive_number_sum_square_l58_58449

theorem positive_number_sum_square (n : ℕ) (h : n^2 + n = 210) : n = 14 :=
sorry

end positive_number_sum_square_l58_58449


namespace tangent_line_acute_probability_l58_58097

-- Define the function and interval
def f (x : ℝ) := x^2 + x

-- Derivative of the function
def f' (x : ℝ) := 2 * x + 1

noncomputable def acute_angle_probability : ℝ :=
  (1 - (-1/2)) / (1 - (-1))

theorem tangent_line_acute_probability :
  ∀ (a : ℝ), (a ∈ set.Icc (-1 : ℝ) (1 : ℝ)) → 
  (∃ p : ℝ, p = acute_angle_probability ∧ p = 3 / 4) := by
  -- Define probable regions and conditions, followed by its calculation
  sorry

end tangent_line_acute_probability_l58_58097


namespace quadratic_real_roots_l58_58949

theorem quadratic_real_roots (K : ℝ) :
  ∃ x : ℝ, K^2 * x^2 + (K^2 - 1) * x - 2 * K^2 = 0 :=
sorry

end quadratic_real_roots_l58_58949


namespace astronaut_revolutions_l58_58433

theorem astronaut_revolutions (n : ℤ) (R : ℝ) (hn : n > 2) :
    ∃ k : ℤ, k = n - 1 := 
sorry

end astronaut_revolutions_l58_58433


namespace partial_fraction_sum_inverse_l58_58164

theorem partial_fraction_sum_inverse (p q r A B C : ℝ)
  (hroots : (∀ s, s^3 - 20 * s^2 + 96 * s - 91 = (s - p) * (s - q) * (s - r)))
  (hA : ∀ s, s ≠ p ∧ s ≠ q ∧ s ≠ r → 
    1 / (s^3 - 20 * s^2 + 96 * s - 91) = A / (s - p) + B / (s - q) + C / (s - r)) :
  1 / A + 1 / B + 1 / C = 225 :=
sorry

end partial_fraction_sum_inverse_l58_58164


namespace probability_penny_dime_halfdollar_tails_is_1_over_8_l58_58033

def probability_penny_dime_halfdollar_tails : ℚ :=
  let total_outcomes := 2^5
  let successful_outcomes := 4
  successful_outcomes / total_outcomes

theorem probability_penny_dime_halfdollar_tails_is_1_over_8 :
  probability_penny_dime_halfdollar_tails = 1 / 8 :=
by
  sorry

end probability_penny_dime_halfdollar_tails_is_1_over_8_l58_58033


namespace probability_product_divisible_by_3_l58_58903

-- Define the problem setup
def die := {1, 2, 3, 4, 5, 6}

-- The event that a product is divisible by 3
def event_product_divisible_by_3 (rolls : list ℕ) : Prop :=
  (rolls.product % 3) = 0

-- The main theorem to prove
theorem probability_product_divisible_by_3 :
  probability (event_product_divisible_by_3 (rolls : list ℕ)) = 211 / 243 := 
sorry

end probability_product_divisible_by_3_l58_58903


namespace max_value_quadratic_function_l58_58539

noncomputable def quadratic_function (x : ℝ) : ℝ :=
  -3 * x^2 + 8

theorem max_value_quadratic_function : ∃(x : ℝ), quadratic_function x = 8 :=
by
  sorry

end max_value_quadratic_function_l58_58539


namespace earnings_bc_l58_58220

variable (A B C : ℕ)

theorem earnings_bc :
  A + B + C = 600 →
  A + C = 400 →
  C = 100 →
  B + C = 300 :=
by
  intros h1 h2 h3
  sorry

end earnings_bc_l58_58220


namespace circle_radius_inscribed_l58_58584

noncomputable def a : ℝ := 6
noncomputable def b : ℝ := 12
noncomputable def c : ℝ := 18

noncomputable def r : ℝ :=
  let term1 := 1/a
  let term2 := 1/b
  let term3 := 1/c
  let sqrt_term := Real.sqrt ((1/(a * b)) + (1/(a * c)) + (1/(b * c)))
  1 / ((term1 + term2 + term3) + 2 * sqrt_term)

theorem circle_radius_inscribed :
  r = 36 / 17 := 
by
  sorry

end circle_radius_inscribed_l58_58584


namespace find_c_l58_58845

theorem find_c (c : ℝ) (h1 : 0 < c) (h2 : c < 3) (h3 : abs (6 + 4 * c) = 14) : c = 2 :=
by {
  sorry
}

end find_c_l58_58845


namespace max_value_amc_am_mc_ca_l58_58133

theorem max_value_amc_am_mc_ca (A M C : ℕ) 
  (h : A + M + C = 15) : 
  A * M * C + A * M + M * C + C * A ≤ 200 := 
sorry

end max_value_amc_am_mc_ca_l58_58133


namespace actual_diameter_of_tissue_l58_58470

theorem actual_diameter_of_tissue (magnification: ℝ) (magnified_diameter: ℝ) :
  magnification = 1000 ∧ magnified_diameter = 1 → magnified_diameter / magnification = 0.001 :=
by
  intro h
  sorry

end actual_diameter_of_tissue_l58_58470


namespace specialPermutationCount_l58_58110

def countSpecialPerms (n : ℕ) : ℕ := 2 ^ (n - 1)

theorem specialPermutationCount (n : ℕ) : 
  (countSpecialPerms n = 2 ^ (n - 1)) := 
by 
  sorry

end specialPermutationCount_l58_58110


namespace intersection_complement_l58_58268

open Set

-- Definitions
def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def A : Set ℕ := {1, 2}
def B : Set ℕ := {2, 3, 4}

-- Theorem
theorem intersection_complement :
  A ∩ (U \ B) = {1} :=
sorry

end intersection_complement_l58_58268


namespace solution_x_l58_58226

noncomputable def find_x (x : ℝ) : Prop :=
  (Real.log (x^4))^2 = (Real.log x)^6

theorem solution_x (x : ℝ) : find_x x ↔ (x = 1 ∨ x = Real.exp 2 ∨ x = Real.exp (-2)) :=
sorry

end solution_x_l58_58226


namespace probability_of_roots_l58_58482

theorem probability_of_roots (k : ℝ) (h1 : 8 ≤ k) (h2 : k ≤ 13) :
  let a := k^2 - 2 * k - 35
  let b := 3 * k - 9
  let c := 2
  let discriminant := b^2 - 4 * a * c
  discriminant ≥ 0 → 
  (∃ x1 x2 : ℝ, 
    a * x1^2 + b * x1 + c = 0 ∧ 
    a * x2^2 + b * x2 + c = 0 ∧
    x1 ≤ 2 * x2) ↔ 
  ∃ p : ℝ, p = 0.6 := 
sorry

end probability_of_roots_l58_58482


namespace combined_weight_of_parcels_l58_58696

variable (x y z : ℕ)

theorem combined_weight_of_parcels : 
  (x + y = 132) ∧ (y + z = 135) ∧ (z + x = 140) → x + y + z = 204 :=
by 
  intros
  sorry

end combined_weight_of_parcels_l58_58696


namespace time_to_be_apart_l58_58918

noncomputable def speed_A : ℝ := 17.5
noncomputable def speed_B : ℝ := 15
noncomputable def initial_distance : ℝ := 65
noncomputable def final_distance : ℝ := 32.5

theorem time_to_be_apart (x : ℝ) :
  x = 1 ∨ x = 3 ↔ 
  (x * (speed_A + speed_B) = initial_distance - final_distance ∨ 
   x * (speed_A + speed_B) = initial_distance + final_distance) :=
sorry

end time_to_be_apart_l58_58918


namespace smallest_S_value_l58_58197

def num_list := {x : ℕ // 1 ≤ x ∧ x ≤ 9}

def S (a b c : num_list) (d e f : num_list) (g h i : num_list) : ℕ :=
  a.val * b.val * c.val + d.val * e.val * f.val + g.val * h.val * i.val

theorem smallest_S_value :
  ∃ a b c d e f g h i : num_list,
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧ a ≠ i ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧ b ≠ i ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧ c ≠ i ∧
  d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧ d ≠ i ∧
  e ≠ f ∧ e ≠ g ∧ e ≠ h ∧ e ≠ i ∧
  f ≠ g ∧ f ≠ h ∧ f ≠ i ∧
  g ≠ h ∧ g ≠ i ∧
  h ≠ i ∧
  S a b c d e f g h i = 214 :=
sorry

end smallest_S_value_l58_58197


namespace max_children_typeA_max_children_typeB_max_children_typeC_max_children_typeD_l58_58764

structure BusConfig where
  rows_section1 : ℕ
  seats_per_row_section1 : ℕ
  rows_section2 : ℕ
  seats_per_row_section2 : ℕ
  total_seats : ℕ
  max_children : ℕ

def typeA : BusConfig :=
  { rows_section1 := 6,
    seats_per_row_section1 := 4,
    rows_section2 := 4,
    seats_per_row_section2 := 3,
    total_seats := 36,
    max_children := 40 }

def typeB : BusConfig :=
  { rows_section1 := 6,
    seats_per_row_section1 := 4,
    rows_section2 := 6,
    seats_per_row_section2 := 5,
    total_seats := 54,
    max_children := 50 }

def typeC : BusConfig :=
  { rows_section1 := 8,
    seats_per_row_section1 := 4,
    rows_section2 := 2,
    seats_per_row_section2 := 2,
    total_seats := 36,
    max_children := 35 }

def typeD : BusConfig :=
  { rows_section1 := 6,
    seats_per_row_section1 := 3,
    rows_section2 := 6,
    seats_per_row_section2 := 3,
    total_seats := 36,
    max_children := 30 }

theorem max_children_typeA : min typeA.total_seats typeA.max_children = 36 := by
  sorry

theorem max_children_typeB : min typeB.total_seats typeB.max_children = 50 := by
  sorry

theorem max_children_typeC : min typeC.total_seats typeC.max_children = 35 := by
  sorry

theorem max_children_typeD : min typeD.total_seats typeD.max_children = 30 := by
  sorry

end max_children_typeA_max_children_typeB_max_children_typeC_max_children_typeD_l58_58764


namespace geometric_sequence_value_of_a_l58_58725

noncomputable def a : ℝ :=
sorry

theorem geometric_sequence_value_of_a
  (is_geometric_seq : ∀ (x y z : ℝ), z / y = y / x)
  (first_term : ℝ)
  (second_term : ℝ)
  (third_term : ℝ)
  (h1 : first_term = 140)
  (h2 : second_term = a)
  (h3 : third_term = 45 / 28)
  (pos_a : a > 0):
  a = 15 :=
sorry

end geometric_sequence_value_of_a_l58_58725


namespace find_n_from_digits_sum_l58_58181

theorem find_n_from_digits_sum (n : ℕ) (h1 : 777 = (9 * 1) + ((99 - 10 + 1) * 2) + (n - 99) * 3) : n = 295 :=
sorry

end find_n_from_digits_sum_l58_58181


namespace students_with_both_l58_58202

/-- There are 28 students in a class -/
def total_students : ℕ := 28

/-- Number of students with a cat -/
def students_with_cat : ℕ := 17

/-- Number of students with a dog -/
def students_with_dog : ℕ := 10

/-- Number of students with neither a cat nor a dog -/
def students_with_neither : ℕ := 5

/-- Number of students having both a cat and a dog -/
theorem students_with_both :
  students_with_cat + students_with_dog - (total_students - students_with_neither) = 4 :=
sorry

end students_with_both_l58_58202


namespace total_weight_of_bars_l58_58441

-- Definitions for weights of each gold bar
variables (C1 C2 C3 C4 C5 C6 C7 C8 C9 C10 C11 C12 C13 : ℝ)
variables (W1 W2 W3 W4 W5 W6 W7 W8 : ℝ)

-- Definitions for the weighings
axiom weight_C1_C2 : W1 = C1 + C2
axiom weight_C1_C3 : W2 = C1 + C3
axiom weight_C2_C3 : W3 = C2 + C3
axiom weight_C4_C5 : W4 = C4 + C5
axiom weight_C6_C7 : W5 = C6 + C7
axiom weight_C8_C9 : W6 = C8 + C9
axiom weight_C10_C11 : W7 = C10 + C11
axiom weight_C12_C13 : W8 = C12 + C13

-- Prove the total weight of all gold bars
theorem total_weight_of_bars :
  (C1 + C2 + C3 + C4 + C5 + C6 + C7 + C8 + C9 + C10 + C11 + C12 + C13)
  = (W1 + W2 + W3) / 2 + W4 + W5 + W6 + W7 + W8 :=
by sorry

end total_weight_of_bars_l58_58441


namespace girls_additional_laps_l58_58004

def distance_per_lap : ℚ := 1 / 6
def boys_laps : ℕ := 34
def boys_distance : ℚ := boys_laps * distance_per_lap
def girls_distance : ℚ := 9
def additional_distance : ℚ := girls_distance - boys_distance
def additional_laps (distance : ℚ) (lap_distance : ℚ) : ℚ := distance / lap_distance

theorem girls_additional_laps :
  additional_laps additional_distance distance_per_lap = 20 := 
by
  sorry

end girls_additional_laps_l58_58004


namespace xiaomings_possible_score_l58_58995

def average_score_class_A : ℤ := 87
def average_score_class_B : ℤ := 82

theorem xiaomings_possible_score (x : ℤ) :
  (average_score_class_B < x ∧ x < average_score_class_A) → x = 85 :=
by sorry

end xiaomings_possible_score_l58_58995


namespace percentage_of_boys_l58_58394

theorem percentage_of_boys (total_students : ℕ) (boy_ratio girl_ratio : ℕ) 
  (total_students_eq : total_students = 42)
  (ratio_eq : boy_ratio = 3 ∧ girl_ratio = 4) :
  (boy_ratio + girl_ratio) = 7 ∧ (total_students / 7 * boy_ratio * 100 / total_students : ℚ) = 42.86 :=
by
  sorry

end percentage_of_boys_l58_58394


namespace plane_equation_l58_58333

theorem plane_equation (A B C D x y z : ℤ) (h1 : A = 15) (h2 : B = -3) (h3 : C = 2) (h4 : D = -238) 
  (h5 : gcd (abs A) (gcd (abs B) (gcd (abs C) (abs D))) = 1) (h6 : A > 0) :
  A * x + B * y + C * z + D = 0 ↔ 15 * x - 3 * y + 2 * z - 238 = 0 :=
by
  sorry

end plane_equation_l58_58333


namespace tank_cost_correct_l58_58766

noncomputable def tankPlasteringCost (l w d cost_per_m2 : ℝ) : ℝ :=
  let long_walls_area := 2 * (l * d)
  let short_walls_area := 2 * (w * d)
  let bottom_area := l * w
  let total_area := long_walls_area + short_walls_area + bottom_area
  total_area * cost_per_m2

theorem tank_cost_correct :
  tankPlasteringCost 25 12 6 0.75 = 558 := by
  sorry

end tank_cost_correct_l58_58766


namespace angle_between_vectors_is_29_5_degrees_l58_58646

noncomputable def vec_a : ℝ × ℝ × ℝ := (3, -2, 2)
noncomputable def vec_b : ℝ × ℝ × ℝ := (1, -1, 1)

def dot_product (v1 v2 : ℝ × ℝ × ℝ) : ℝ :=
v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

def magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
Real.sqrt (v.1 ^ 2 + v.2 ^ 2 + v.3 ^ 2)

def cos_theta (v1 v2 : ℝ × ℝ × ℝ) : ℝ :=
dot_product v1 v2 / (magnitude v1 * magnitude v2)

def theta (v1 v2 : ℝ × ℝ × ℝ) : ℝ :=
Real.arccos (cos_theta v1 v2) * 180 / Real.pi

theorem angle_between_vectors_is_29_5_degrees :
  theta vec_a vec_b = 29.5 :=
sorry

end angle_between_vectors_is_29_5_degrees_l58_58646


namespace min_guests_at_banquet_l58_58341

-- Definitions based on conditions
def total_food : ℕ := 675
def vegetarian_food : ℕ := 195
def pescatarian_food : ℕ := 220
def carnivorous_food : ℕ := 260

def max_vegetarian_per_guest : ℚ := 3
def max_pescatarian_per_guest : ℚ := 2.5
def max_carnivorous_per_guest : ℚ := 4

-- Definition based on the question and the correct answer
def minimum_number_of_guests : ℕ := 218

-- Lean statement to prove the problem
theorem min_guests_at_banquet :
  195 / 3 + 220 / 2.5 + 260 / 4 = 218 :=
by sorry

end min_guests_at_banquet_l58_58341


namespace zach_needs_more_money_zach_more_money_needed_l58_58051

/-!
# Zach's Bike Savings Problem
Zach needs $100 to buy a brand new bike.
Weekly allowance: $5.
Earnings from mowing the lawn: $10.
Earnings from babysitting: $7 per hour.
Zach has already saved $65.
He will receive weekly allowance on Friday.
He will mow the lawn and babysit for 2 hours this Saturday.
Prove that Zach needs $6 more to buy the bike.
-/

def zach_current_savings : ℕ := 65
def bike_cost : ℕ := 100
def weekly_allowance : ℕ := 5
def mowing_earnings : ℕ := 10
def babysitting_rate : ℕ := 7
def babysitting_hours : ℕ := 2

theorem zach_needs_more_money : zach_current_savings + (weekly_allowance + mowing_earnings + (babysitting_rate * babysitting_hours)) = 94 :=
by sorry

theorem zach_more_money_needed : bike_cost - (zach_current_savings + (weekly_allowance + mowing_earnings + (babysitting_rate * babysitting_hours))) = 6 :=
by sorry

end zach_needs_more_money_zach_more_money_needed_l58_58051


namespace num_four_digit_integers_with_at_least_one_4_or_7_l58_58536

def count_four_digit_integers_with_4_or_7 : ℕ := 5416

theorem num_four_digit_integers_with_at_least_one_4_or_7 :
  let all_four_digit_integers := 9000
  let valid_digits_first := 7
  let valid_digits := 8
  let integers_without_4_or_7 := valid_digits_first * valid_digits * valid_digits * valid_digits
  all_four_digit_integers - integers_without_4_or_7 = count_four_digit_integers_with_4_or_7 :=
by
  -- Using known values from the problem statement
  let all_four_digit_integers := 9000
  let valid_digits_first := 7
  let valid_digits := 8
  let integers_without_4_or_7 := valid_digits_first * valid_digits * valid_digits * valid_digits
  show all_four_digit_integers - integers_without_4_or_7 = count_four_digit_integers_with_4_or_7
  sorry

end num_four_digit_integers_with_at_least_one_4_or_7_l58_58536


namespace triangle_height_l58_58574

theorem triangle_height (base height : ℝ) (area : ℝ) (h1 : base = 2) (h2 : area = 3) (area_formula : area = (base * height) / 2) : height = 3 :=
by
  sorry

end triangle_height_l58_58574


namespace complement_U_M_correct_l58_58673

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {4, 5}
def complement_U_M : Set ℕ := {1, 2, 3}

theorem complement_U_M_correct : U \ M = complement_U_M :=
  by sorry

end complement_U_M_correct_l58_58673


namespace nikolai_faster_than_gennady_l58_58739

-- The conditions of the problem translated to Lean definitions
def gennady_jump_length : ℕ := 6
def gennady_jumps_per_time : ℕ := 2
def nikolai_jump_length : ℕ := 4
def nikolai_jumps_per_time : ℕ := 3
def turn_around_distance : ℕ := 2000
def round_trip_distance : ℕ := 2 * turn_around_distance

-- The statement that Nikolai completes the journey faster than Gennady
theorem nikolai_faster_than_gennady :
  (nikolai_jumps_per_time * nikolai_jump_length) = (gennady_jumps_per_time * gennady_jump_length) →
  (round_trip_distance % nikolai_jump_length = 0) →
  (round_trip_distance % gennady_jump_length ≠ 0) →
  (round_trip_distance / nikolai_jump_length) + 1 < (round_trip_distance / gennady_jump_length) →
  "Nikolay completes the journey faster." :=
by
  intros h_eq_speed h_nikolai_divisible h_gennady_not_divisible h_time_compare
  sorry

end nikolai_faster_than_gennady_l58_58739


namespace geometric_series_first_term_l58_58340

noncomputable def first_term_geometric_series (r : ℝ) (S : ℝ) (a : ℝ) : Prop :=
  S = a / (1 - r)

theorem geometric_series_first_term
  (r : ℝ) (S : ℝ) (a : ℝ)
  (hr : r = 1/6)
  (hS : S = 54) :
  first_term_geometric_series r S a →
  a = 45 :=
by
  intros h
  -- The proof goes here
  sorry

end geometric_series_first_term_l58_58340


namespace below_sea_level_notation_l58_58844

theorem below_sea_level_notation (depth_above_sea_level : Int) (depth_below_sea_level: Int) 
  (h: depth_above_sea_level = 9050 ∧ depth_below_sea_level = -10907) : 
  depth_below_sea_level = -10907 :=
by 
  sorry

end below_sea_level_notation_l58_58844


namespace collinearity_B_K1_K2_l58_58396

noncomputable theory

open EuclideanGeometry

-- Definitions of key locations
variables {A B C B₁ B₂ K₁ K₂: Point} 

-- Predefined conditions
def non_isosceles_triangle (A B C : Point) : Prop := 
A ≠ B ∧ B ≠ C ∧ C ≠ A ∧ Triangle A B C

def internal_angle_bisector_intersect (A B C B₁ : Point) : Prop := 
Bisects (inside Angle_B_ABC) A C B₁

def external_angle_bisector_intersect (A B C B₂ : Point) : Prop := 
Bisects (outside Angle_B_ABC) A C B₂

def tangent_from_incircle_touch {A B C B₁ K₁ B₂ K₂ : Point} 
    (I : InCircle non_isosceles_triangle ABC) 
    (h₁ : internal_angle_bisector_intersect A B C B₁) 
    (h₂ : external_angle_bisector_intersect A B C B₂) : Prop := 
    TangentsFromPoint B₁ I K₁ ∧ TangentsFromPoint B₂ I K₂

-- Main theorem to prove
theorem collinearity_B_K1_K2 
    {A B C B₁ B₂ K₁ K₂: Point} 
    (HABC : non_isosceles_triangle A B C) 
    (H₁ : internal_angle_bisector_intersect A B C B₁) 
    (H₂ : external_angle_bisector_intersect A B C B₂) 
    (touching : tangent_from_incircle_touch (incircle HABC) H₁ H₂) :
     Collinear B K₁ K₂ := 
sorry

end collinearity_B_K1_K2_l58_58396


namespace tangent_line_through_point_l58_58762

theorem tangent_line_through_point (t : ℝ) :
    (∃ l : ℝ → ℝ, (∃ m : ℝ, (∀ x, l x = 2 * m * x - m^2) ∧ (t = m - 2 * m + 2 * m * m) ∧ m = 1/2) ∧ l t = 0)
    → t = 1/4 :=
by
  sorry

end tangent_line_through_point_l58_58762


namespace range_of_a_if_p_true_l58_58856

theorem range_of_a_if_p_true : 
  (∃ (x : ℝ), 1 ≤ x ∧ x ≤ 9 ∧ x^2 - a * x + 36 ≤ 0) → a ≥ 12 :=
sorry

end range_of_a_if_p_true_l58_58856


namespace max_rubles_can_receive_l58_58711

-- Define the four-digit number 2079
def number := 2079

-- Check divisibility conditions
def divisible_by_1 := number % 1 = 0
def divisible_by_3 := number % 3 = 0
def divisible_by_5 := number % 5 = 0
def divisible_by_7 := number % 7 = 0
def divisible_by_9 := number % 9 = 0
def divisible_by_11 := number % 11 = 0

-- Define the sum of rubles under the conditions described
def sum_rubles := if divisible_by_1 then 1 else 0 + if divisible_by_3 then 3 else 0 + if divisible_by_7 then 7 else 0 + if divisible_by_9 then 9 else 0 + if divisible_by_11 then 11 else 0

-- The final proof statement
theorem max_rubles_can_receive : sum_rubles = 31 :=
by
  unfold sum_rubles
  -- Confirm the result by simplification, let's skip the proof as required
  sorry

end max_rubles_can_receive_l58_58711


namespace max_student_count_l58_58980

theorem max_student_count
  (x1 x2 x3 x4 x5 : ℝ)
  (h1 : (x1 + x2 + x3 + x4 + x5) / 5 = 7)
  (h2 : ((x1 - 7) ^ 2 + (x2 - 7) ^ 2 + (x3 - 7) ^ 2 + (x4 - 7) ^ 2 + (x5 - 7) ^ 2) / 5 = 4)
  (h3 : ∀ i j, i ≠ j → List.nthLe [x1, x2, x3, x4, x5] i sorry ≠ List.nthLe [x1, x2, x3, x4, x5] j sorry) :
  max x1 (max x2 (max x3 (max x4 x5))) = 10 := 
sorry

end max_student_count_l58_58980


namespace range_of_a_l58_58576

open Real

noncomputable def f (a x : ℝ) : ℝ := log x + (1 / 2) * x ^ 2 + a * x

theorem range_of_a (a : ℝ) : (∃ x > 0, deriv (f a) x = 3) ↔ a < 1 := by
  sorry

end range_of_a_l58_58576


namespace airfare_price_for_BD_l58_58773

theorem airfare_price_for_BD (AB AC AD CD BC : ℝ) (hAB : AB = 2000) (hAC : AC = 1600) (hAD : AD = 2500) 
    (hCD : CD = 900) (hBC : BC = 1200) (proportional_pricing : ∀ x y : ℝ, x * (y / x) = y) : 
    ∃ BD : ℝ, BD = 1500 :=
by
  sorry

end airfare_price_for_BD_l58_58773


namespace find_set_B_l58_58667

set_option pp.all true

variable (A : Set ℤ) (B : Set ℤ)

theorem find_set_B (hA : A = {-2, 0, 1, 3})
                    (hB : B = {x | -x ∈ A ∧ 1 - x ∉ A}) :
  B = {-3, -1, 2} :=
by
  sorry

end find_set_B_l58_58667


namespace find_x_l58_58527

open Nat

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem find_x (x : ℕ) (hx : x > 0) (hprime : is_prime (x^5 + x + 1)) : x = 1 := 
by 
  sorry

end find_x_l58_58527


namespace arithmetic_seq_a7_value_l58_58398

theorem arithmetic_seq_a7_value {a : ℕ → ℝ} (h_positive : ∀ n, 0 < a n)
    (h_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1))
    (h_eq : 3 * a 6 - (a 7) ^ 2 + 3 * a 8 = 0) : a 7 = 6 :=
  sorry

end arithmetic_seq_a7_value_l58_58398


namespace greatest_multiple_of_four_cubed_less_than_2000_l58_58031

theorem greatest_multiple_of_four_cubed_less_than_2000 :
  ∃ x, (x > 0) ∧ (x % 4 = 0) ∧ (x^3 < 2000) ∧ ∀ y, (y > x) ∧ (y % 4 = 0) → y^3 ≥ 2000 :=
sorry

end greatest_multiple_of_four_cubed_less_than_2000_l58_58031


namespace same_solution_k_value_l58_58088

theorem same_solution_k_value 
  (x : ℝ)
  (k : ℝ)
  (m : ℝ)
  (h₁ : 2 * x + 4 = 4 * (x - 2))
  (h₂ : k * x + m = 2 * x - 1) 
  (h₃ : k = 17) : 
  k = 17 ∧ m = -91 :=
by
  sorry

end same_solution_k_value_l58_58088


namespace different_sets_l58_58408

theorem different_sets (a b c : ℤ) (h1 : 0 < a) (h2 : a < c - 1) (h3 : 1 < b) (h4 : b < c)
  (rk : ∀ (k : ℤ), 0 ≤ k ∧ k ≤ a → ∃ (r : ℤ), 0 ≤ r ∧ r < c ∧ k * b % c = r) :
  {r | ∃ k, 0 ≤ k ∧ k ≤ a ∧ r = k * b % c} ≠ {k | 0 ≤ k ∧ k ≤ a} :=
sorry

end different_sets_l58_58408


namespace second_train_speed_l58_58902

theorem second_train_speed (v : ℝ) :
  (∃ t : ℝ, 20 * t = v * t + 75 ∧ 20 * t + v * t = 675) → v = 16 :=
by
  sorry

end second_train_speed_l58_58902


namespace probability_three_blue_jellybeans_l58_58919

theorem probability_three_blue_jellybeans:
  let total_jellybeans := 20
  let blue_jellybeans := 10
  let red_jellybeans := 10
  let draws := 3
  let q := (1 / 2) * (9 / 19) * (4 / 9)
  q = 2 / 19 :=
sorry

end probability_three_blue_jellybeans_l58_58919


namespace membership_percentage_change_l58_58930

-- Definitions required based on conditions
def membersFallChange (initialMembers : ℝ) : ℝ := initialMembers * 1.07
def membersSpringChange (fallMembers : ℝ) : ℝ := fallMembers * 0.81
def membersSummerChange (springMembers : ℝ) : ℝ := springMembers * 1.15

-- Prove the total change in percentage from fall to the end of summer
theorem membership_percentage_change :
  let initialMembers := 100
  let fallMembers := membersFallChange initialMembers
  let springMembers := membersSpringChange fallMembers
  let summerMembers := membersSummerChange springMembers
  ((summerMembers - initialMembers) / initialMembers) * 100 = -0.33 := by
  sorry

end membership_percentage_change_l58_58930


namespace triangles_in_figure_l58_58274

-- Definitions for the figure
def number_of_triangles : ℕ :=
  -- The number of triangles in a figure composed of a rectangle with three vertical lines and two horizontal lines
  50

-- The theorem we want to prove
theorem triangles_in_figure : number_of_triangles = 50 :=
by
  sorry

end triangles_in_figure_l58_58274


namespace trigonometric_identity_l58_58100

open Real

theorem trigonometric_identity (θ : ℝ) (h₁ : 0 < θ ∧ θ < π/2) (h₂ : cos θ = sqrt 10 / 10) :
  (cos (2 * θ) / (sin (2 * θ) + (cos θ)^2)) = -8 / 7 := 
sorry

end trigonometric_identity_l58_58100


namespace sum_of_three_consecutive_even_numbers_l58_58387

theorem sum_of_three_consecutive_even_numbers (m : ℤ) (h : ∃ k, m = 2 * k) : 
  m + (m + 2) + (m + 4) = 3 * m + 6 :=
by
  sorry

end sum_of_three_consecutive_even_numbers_l58_58387


namespace infinitely_many_solutions_b_value_l58_58252

theorem infinitely_many_solutions_b_value :
  ∀ (x : ℝ) (b : ℝ), (5 * (4 * x - b) = 3 * (5 * x + 15)) ↔ b = -9 := 
by
  intro x b
  sorry

end infinitely_many_solutions_b_value_l58_58252


namespace purchasing_plans_count_l58_58312

theorem purchasing_plans_count :
  ∃ (x y : ℕ), (4 * y + 6 * x = 40)  ∧ (y ≥ 0) ∧ (x ≥ 0) ∧ (∃! (x y : ℕ), (4 * y + 6 * x = 40)  ∧ (y ≥ 0) ∧ (x ≥ 0)) := sorry

end purchasing_plans_count_l58_58312


namespace inverse_proportion_decreasing_l58_58176

theorem inverse_proportion_decreasing (k : ℝ) (x : ℝ) (hx : x > 0) :
  (y = (k - 1) / x) → (k > 1) :=
by
  sorry

end inverse_proportion_decreasing_l58_58176


namespace sum_of_coefficients_l58_58510

theorem sum_of_coefficients (d : ℤ) : 
  let expr := -(4 - d) * (d + 3 * (4 - d))
  let expanded_form := -2 * d ^ 2 + 20 * d - 48
  let sum_of_coeffs := -2 + 20 - 48
  sum_of_coeffs = -30 :=
by
  -- The proof will go here, skipping for now.
  sorry

end sum_of_coefficients_l58_58510


namespace lowest_cost_per_ton_l58_58430

-- Define the conditions given in the problem statement
variable (x : ℝ) (y : ℝ)

-- Define the annual production range
def production_range (x : ℝ) : Prop := x ≥ 150 ∧ x ≤ 250

-- Define the relationship between total annual production cost and annual production
def production_cost_relation (x y : ℝ) : Prop := y = (x^2 / 10) - 30 * x + 4000

-- State the main theorem: the annual production when the cost per ton is the lowest is 200 tons
theorem lowest_cost_per_ton (x : ℝ) (y : ℝ) (h1 : production_range x) (h2 : production_cost_relation x y) : x = 200 :=
sorry

end lowest_cost_per_ton_l58_58430


namespace calculate_expression_l58_58073

theorem calculate_expression : 
  2 * (3 + 1) * (3^2 + 1) * (3^4 + 1) * (3^8 + 1) * (3^16 + 1) * (3^32 + 1) * (3^64 + 1) + 1 = 3^128 :=
sorry

end calculate_expression_l58_58073


namespace lines_of_first_character_l58_58557

-- Definitions for the number of lines each character has
def L3 : Nat := 2

def L2 : Nat := 3 * L3 + 6

def L1 : Nat := L2 + 8

-- The theorem we are proving
theorem lines_of_first_character : L1 = 20 :=
by
  -- The proof would go here
  sorry

end lines_of_first_character_l58_58557


namespace max_rubles_can_receive_l58_58710

-- Define the four-digit number 2079
def number := 2079

-- Check divisibility conditions
def divisible_by_1 := number % 1 = 0
def divisible_by_3 := number % 3 = 0
def divisible_by_5 := number % 5 = 0
def divisible_by_7 := number % 7 = 0
def divisible_by_9 := number % 9 = 0
def divisible_by_11 := number % 11 = 0

-- Define the sum of rubles under the conditions described
def sum_rubles := if divisible_by_1 then 1 else 0 + if divisible_by_3 then 3 else 0 + if divisible_by_7 then 7 else 0 + if divisible_by_9 then 9 else 0 + if divisible_by_11 then 11 else 0

-- The final proof statement
theorem max_rubles_can_receive : sum_rubles = 31 :=
by
  unfold sum_rubles
  -- Confirm the result by simplification, let's skip the proof as required
  sorry

end max_rubles_can_receive_l58_58710


namespace depth_notation_l58_58829

-- Definition of depth and altitude
def above_sea_level (h : ℝ) : Prop := h > 0 
def below_sea_level (h : ℝ) : Prop := h < 0 

-- Given conditions
axiom height_Dabaijing : above_sea_level 9050
axiom depth_Haidou1 : below_sea_level (-10907)

-- Proof goal
theorem depth_notation :
  ∀ (d : ℝ), above_sea_level 9050 → below_sea_level (-d) → d = 10907 :=
by
  intros d _ _
  exact sorry

end depth_notation_l58_58829


namespace larger_number_is_1891_l58_58357

def is_prime (n : ℕ) : Prop :=
  ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem larger_number_is_1891 :
  ∃ L S : ℕ, (L - S = 1355) ∧ (L = 6 * S + 15) ∧ is_prime (sum_of_digits L) ∧ sum_of_digits L ≠ 12
  :=
sorry

end larger_number_is_1891_l58_58357


namespace imaginary_part_of_fraction_l58_58434

theorem imaginary_part_of_fraction (i : ℂ) (h : i^2 = -1) : ( (i^2) / (2 * i - 1) ).im = (2 / 5) :=
by
  sorry

end imaginary_part_of_fraction_l58_58434


namespace proof_a_plus_2b_equal_7_l58_58846

theorem proof_a_plus_2b_equal_7 (a b : ℕ) (h1 : 82 * 1000 + a * 10 + 7 + 6 * b = 190) (h2 : 1 ≤ a) (h3 : a < 10) (h4 : 1 ≤ b) (h5 : b < 10) : 
  a + 2 * b = 7 :=
by sorry

end proof_a_plus_2b_equal_7_l58_58846


namespace find_x_prime_l58_58526

theorem find_x_prime (x : ℕ) (h1 : x > 0) (h2 : Prime (x^5 + x + 1)) : x = 1 := sorry

end find_x_prime_l58_58526


namespace find_other_number_l58_58321

/--
Given two numbers A and B, where:
    * The reciprocal of the HCF of A and B is \( \frac{1}{13} \).
    * The reciprocal of the LCM of A and B is \( \frac{1}{312} \).
    * A = 24
Prove that B = 169.
-/
theorem find_other_number 
  (A B : ℕ) 
  (h1 : A = 24)
  (h2 : (Nat.gcd A B) = 13)
  (h3 : (Nat.lcm A B) = 312) : 
  B = 169 := 
by 
  sorry

end find_other_number_l58_58321


namespace like_terms_sum_l58_58113

theorem like_terms_sum (m n : ℕ) (h1 : m = 3) (h2 : 4 = n + 2) : m + n = 5 :=
by
  sorry

end like_terms_sum_l58_58113


namespace sum_a2012_a2013_l58_58688

-- Define the geometric sequence and its conditions
def geometric_sequence (a : ℕ → ℚ) (q : ℚ) : Prop := 
  ∀ n : ℕ, a (n + 1) = a n * q

-- Parameters for the problem
variable (a : ℕ → ℚ)
variable (q : ℚ)
variable (h_seq : geometric_sequence a q)
variable (h_q : 1 < q)
variable (h_eq : ∀ x : ℚ, 4 * x^2 - 8 * x + 3 = 0 → x = a 2010 ∨ x = a 2011)

-- Statement to prove
theorem sum_a2012_a2013 : a 2012 + a 2013 = 18 :=
by
  sorry

end sum_a2012_a2013_l58_58688


namespace max_value_amc_am_mc_ca_l58_58137

theorem max_value_amc_am_mc_ca (A M C : ℕ) 
  (h : A + M + C = 15) : 
  A * M * C + A * M + M * C + C * A ≤ 200 := 
sorry

end max_value_amc_am_mc_ca_l58_58137


namespace problem_statement_l58_58361

theorem problem_statement {m n : ℝ} 
  (h1 : (n + 2 * m) / (1 + m ^ 2) = -1 / 2) 
  (h2 : -(1 + n) + 2 * (m + 2) = 0) : 
  (m / n = -1) := 
sorry

end problem_statement_l58_58361


namespace tan_gt_x_plus_one_third_cubed_l58_58021

theorem tan_gt_x_plus_one_third_cubed (x : ℝ) (h1 : 0 < x) (h2 : x < π / 2) : 
  tan x > x + (1 / 3) * x^3 :=
sorry

end tan_gt_x_plus_one_third_cubed_l58_58021


namespace machine_no_repair_needed_l58_58037

theorem machine_no_repair_needed (M : ℕ) (σ : ℕ) (greatest_deviation : ℕ) 
                                  (nominal_weight : ℕ)
                                  (h1 : greatest_deviation = 39)
                                  (h2 : greatest_deviation ≤ (0.1 * nominal_weight))
                                  (h3 : ∀ d, d < 39) : 
                                  σ ≤ greatest_deviation :=
by
  sorry

end machine_no_repair_needed_l58_58037


namespace intimate_interval_proof_l58_58012

-- Define the functions f and g
def f (x : ℝ) : ℝ := x^2 - 3 * x + 4
def g (x : ℝ) : ℝ := 2 * x - 3

-- Define the concept of intimate functions over an interval
def are_intimate_functions (a b : ℝ) : Prop :=
  ∀ x, a ≤ x ∧ x ≤ b → |f x - g x| ≤ 1

-- Prove that the interval [2, 3] is a subset of [a, b]
theorem intimate_interval_proof (a b : ℝ) (h : are_intimate_functions a b) :
  2 ≤ b ∧ a ≤ 3 :=
sorry

end intimate_interval_proof_l58_58012


namespace find_positive_number_l58_58452

theorem find_positive_number (n : ℕ) (h : n^2 + n = 210) : n = 14 :=
sorry

end find_positive_number_l58_58452


namespace radius_of_smaller_base_l58_58309

theorem radius_of_smaller_base (C1 C2 : ℝ) (r : ℝ) (l : ℝ) (A : ℝ) 
    (h1 : C2 = 3 * C1) 
    (h2 : l = 3) 
    (h3 : A = 84 * Real.pi) 
    (h4 : C1 = 2 * Real.pi * r) 
    (h5 : C2 = 2 * Real.pi * (3 * r)) :
    r = 7 := 
by
  -- proof steps here
  sorry

end radius_of_smaller_base_l58_58309


namespace sand_art_l58_58555

theorem sand_art (len_blue_rect : ℕ) (area_blue_rect : ℕ) (side_red_square : ℕ) (sand_per_sq_inch : ℕ) (h1 : len_blue_rect = 7) (h2 : area_blue_rect = 42) (h3 : side_red_square = 5) (h4 : sand_per_sq_inch = 3) :
  (area_blue_rect * sand_per_sq_inch) + (side_red_square * side_red_square * sand_per_sq_inch) = 201 :=
by
  sorry

end sand_art_l58_58555


namespace both_firms_participate_number_of_firms_participate_social_optimality_l58_58917

-- Definitions for general conditions
variable (α V IC : ℝ)
variable (hα : 0 < α ∧ α < 1)

-- Condition for both firms to participate
def condition_to_participate (V : ℝ) (α : ℝ) (IC : ℝ) : Prop :=
  V * α * (1 - 0.5 * α) ≥ IC

-- Part (a): Under what conditions will both firms participate?
theorem both_firms_participate (α V IC : ℝ) (hα : 0 < α ∧ α < 1) :
  condition_to_participate V α IC → (V * α * (1 - 0.5 * α) ≥ IC) :=
by sorry

-- Part (b): Given V=16, α=0.5, and IC=5, determine the number of firms participating
theorem number_of_firms_participate :
  (condition_to_participate 16 0.5 5) :=
by sorry

-- Part (c): To determine if the number of participating firms is socially optimal
def total_profit (α V IC : ℝ) (both : Bool) :=
  if both then 2 * (α * (1 - α) * V + 0.5 * α^2 * V - IC)
  else α * V - IC

theorem social_optimality :
   (total_profit 0.5 16 5 true ≠ max (total_profit 0.5 16 5 true) (total_profit 0.5 16 5 false)) :=
by sorry

end both_firms_participate_number_of_firms_participate_social_optimality_l58_58917


namespace mod_problem_l58_58678

theorem mod_problem (x : ℤ) (h1 : 5 * x ≡ 10 [ZMOD 25]) (h2 : 4 * x ≡ 21 [ZMOD 25]) : (x^2 ≡ 21 [ZMOD 25]) :=
sorry

end mod_problem_l58_58678


namespace find_q_value_l58_58540

theorem find_q_value 
  (p q r : ℕ) 
  (hp : 0 < p) 
  (hq : 0 < q) 
  (hr : 0 < r) 
  (h : p + 1 / (q + 1 / r : ℚ) = 25 / 19) : 
  q = 3 :=
by 
  sorry

end find_q_value_l58_58540


namespace kaleb_can_buy_toys_l58_58405

def kaleb_initial_money : ℕ := 12
def money_spent_on_game : ℕ := 8
def money_saved : ℕ := 2
def toy_cost : ℕ := 2

theorem kaleb_can_buy_toys :
  (kaleb_initial_money - money_spent_on_game - money_saved) / toy_cost = 1 :=
by
  sorry

end kaleb_can_buy_toys_l58_58405


namespace perimeter_of_square_l58_58237

theorem perimeter_of_square (a : ℤ) (h : a * a = 36) : 4 * a = 24 := 
by
  sorry

end perimeter_of_square_l58_58237


namespace remainder_division_l58_58599
-- Import the necessary library

-- Define the number and the divisor
def number : ℕ := 2345678901
def divisor : ℕ := 101

-- State the theorem
theorem remainder_division : number % divisor = 23 :=
by sorry

end remainder_division_l58_58599


namespace probability_Q_eq_1_l58_58623

open Complex

theorem probability_Q_eq_1 :
  let W := {2 * I, -2 * I, (1 + I) / 2, (-1 + I) / 2, (1 - I) / 2, (-1 - I) / 2, 1, -1}
  let Q := ∏ k in Finset.range 16, Finset.choose W 16
  ∃ (c d q : ℕ), q.prime ∧ c % q ≠ 0 ∧ probability (Q = 1) = (c : ℝ) / (q : ℝ)^d ∧ c + d + q = 65 :=
begin
  sorry
end

end probability_Q_eq_1_l58_58623


namespace ellipse_x_intercept_l58_58774

theorem ellipse_x_intercept :
  let F_1 := (0,3)
  let F_2 := (4,0)
  let ellipse := { P : ℝ × ℝ | (dist P F_1) + (dist P F_2) = 7 }
  ∃ x : ℝ, x ≠ 0 ∧ (x, 0) ∈ ellipse ∧ x = 56 / 11 :=
by
  sorry

end ellipse_x_intercept_l58_58774


namespace hypotenuse_length_l58_58234

theorem hypotenuse_length
  (a b c : ℝ)
  (h1 : a + b + c = 40)
  (h2 : (1 / 2) * a * b = 24)
  (h3 : a^2 + b^2 = c^2) :
  c = 18.8 :=
by sorry

end hypotenuse_length_l58_58234


namespace Petya_rubles_maximum_l58_58704

theorem Petya_rubles_maximum :
  ∃ n, (2000 ≤ n ∧ n < 2100) ∧ (∀ d ∈ [1, 3, 5, 7, 9, 11], n % d = 0 → true) → 
  (1 + ite (n % 1 = 0) 1 0 + ite (n % 3 = 0) 3 0 + ite (n % 5 = 0) 5 0 + ite (n % 7 = 0) 7 0 + ite (n % 9 = 0) 9 0 + ite (n % 11 = 0) 11 0 = 31) :=
begin
  sorry
end

end Petya_rubles_maximum_l58_58704


namespace no_solution_for_inequalities_l58_58501

theorem no_solution_for_inequalities (x : ℝ) : ¬(4 * x ^ 2 + 7 * x - 2 < 0 ∧ 3 * x - 1 > 0) :=
by
  sorry

end no_solution_for_inequalities_l58_58501


namespace min_value_frac_l58_58372

variable (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : a + b = 1)

theorem min_value_frac : (1 / a + 4 / b) = 9 :=
by sorry

end min_value_frac_l58_58372


namespace smallest_non_palindrome_power_of_13_l58_58653

def is_palindrome (n : ℕ) : Bool :=
  let s := (toString n).toList
  s == s.reverse

def is_power_of_13 (n : ℕ) : Bool :=
  ∃ k : ℕ, n = 13 ^ k

theorem smallest_non_palindrome_power_of_13 : ∃ n : ℕ, n = 13 ∧ 
  (¬ is_palindrome n) ∧ (is_power_of_13 n) ∧ (∀ m : ℕ, m < n → ¬ is_palindrome m ∨ ¬ is_power_of_13 m) :=
    sorry

end smallest_non_palindrome_power_of_13_l58_58653


namespace find_k_l58_58379

-- Define the sets A and B
def A (k : ℕ) : Set ℕ := {1, 2, k}
def B : Set ℕ := {2, 5}

-- Given that the union of sets A and B is {1, 2, 3, 5}, prove that k = 3.
theorem find_k (k : ℕ) (h : A k ∪ B = {1, 2, 3, 5}) : k = 3 :=
by
  sorry

end find_k_l58_58379


namespace sum_of_coefficients_expansion_l58_58506

theorem sum_of_coefficients_expansion (d : ℝ) :
  let expr := -(4 - d) * (d + 3 * (4 - d))
  in (polynomial.sum_of_coeffs expr) = -30 :=
by
  let expr := -(4 - d) * (d + 3 * (4 - d))
  have h_expr : expr = -2 * d^2 + 20 * d - 48, sorry
  have h_coeffs_sum : polynomial.sum_of_coeffs (-2 * d^2 + 20 * d - 48) = -30, sorry
  rw h_expr
  exact h_coeffs_sum

end sum_of_coefficients_expansion_l58_58506


namespace tank_depth_is_six_l58_58068

-- Definitions derived from the conditions
def tank_length : ℝ := 25
def tank_width : ℝ := 12
def plastering_cost_per_sq_meter : ℝ := 0.45
def total_cost : ℝ := 334.8

-- Compute the surface area to be plastered
def surface_area (d : ℝ) : ℝ := (tank_length * tank_width) + 2 * (tank_length * d) + 2 * (tank_width * d)

-- Equation relating the plastering cost to the surface area
def cost_equation (d : ℝ) : ℝ := plastering_cost_per_sq_meter * (surface_area d)

-- The mathematical result we need to prove
theorem tank_depth_is_six : ∃ d : ℝ, cost_equation d = total_cost ∧ d = 6 := by
  sorry

end tank_depth_is_six_l58_58068


namespace sufficient_but_not_necessary_l58_58093

variable (m : ℝ)

def P : Prop := ∀ x : ℝ, x^2 - 4*x + 3*m > 0
def Q : Prop := ∀ x : ℝ, 3*x^2 + 4*x + m ≥ 0

theorem sufficient_but_not_necessary : (P m → Q m) ∧ ¬(Q m → P m) :=
by
  sorry

end sufficient_but_not_necessary_l58_58093


namespace max_heaps_660_l58_58861

-- Define the conditions and goal
theorem max_heaps_660 (h : ∀ (a b : ℕ), a ∈ heaps → b ∈ heaps → a ≤ b → b < 2 * a) :
  ∃ heaps : finset ℕ, heaps.sum id = 660 ∧ heaps.card = 30 :=
by
  -- Initial definitions
  have : ∀ (heaps : finset ℕ), heaps.sum id = 660 → heaps.card ≤ 30,
  sorry
  -- Construct existence of heaps with the required conditions
  refine ⟨{15, 15, 16, 16, 17, 17, 18, 18, ..., 29, 29}.to_finset, _, _⟩,
  sorry

end max_heaps_660_l58_58861


namespace jenna_less_than_bob_l58_58811

def bob_amount : ℕ := 60
def phil_amount : ℕ := (1 / 3) * bob_amount
def jenna_amount : ℕ := 2 * phil_amount

theorem jenna_less_than_bob : bob_amount - jenna_amount = 20 := by
  sorry

end jenna_less_than_bob_l58_58811


namespace sum_of_numbers_l58_58311

theorem sum_of_numbers (a b : ℝ) 
  (h1 : a^2 - b^2 = 6) 
  (h2 : (a - 2)^2 - (b - 2)^2 = 18): 
  a + b = -2 := 
by 
  sorry

end sum_of_numbers_l58_58311


namespace sum_of_two_numbers_l58_58180

theorem sum_of_two_numbers (L S : ℕ) (hL : L = 22) (hExceeds : L = S + 10) : L + S = 34 := by
  sorry

end sum_of_two_numbers_l58_58180


namespace probability_first_4_second_club_third_2_l58_58206

theorem probability_first_4_second_club_third_2 :
  let deck_size := 52
  let prob_4_first := 4 / deck_size
  let deck_minus_first_card := deck_size - 1
  let prob_club_second := 13 / deck_minus_first_card
  let deck_minus_two_cards := deck_minus_first_card - 1
  let prob_2_third := 4 / deck_minus_two_cards
  prob_4_first * prob_club_second * prob_2_third = 1 / 663 :=
by
  sorry

end probability_first_4_second_club_third_2_l58_58206


namespace least_positive_linear_combination_l58_58108

theorem least_positive_linear_combination :
  ∃ x y z : ℤ, 0 < 24 * x + 20 * y + 12 * z ∧ ∀ n : ℤ, (∃ x y z : ℤ, n = 24 * x + 20 * y + 12 * z) → 0 < n → 4 ≤ n :=
by
  sorry

end least_positive_linear_combination_l58_58108


namespace initial_toys_count_l58_58015

theorem initial_toys_count (T : ℕ) (h : 10 * T + 300 = 580) : T = 28 :=
by
  sorry

end initial_toys_count_l58_58015


namespace inequality_solution_l58_58502

theorem inequality_solution : {x : ℝ | -2 < (x^2 - 12 * x + 20) / (x^2 - 4 * x + 8) ∧ (x^2 - 12 * x + 20) / (x^2 - 4 * x + 8) < 2} = {x : ℝ | 5 < x} := 
sorry

end inequality_solution_l58_58502


namespace find_x_cube_plus_reciprocal_cube_l58_58390

variable {x : ℝ}

theorem find_x_cube_plus_reciprocal_cube (hx : x + 1/x = 10) : x^3 + 1/x^3 = 970 :=
sorry

end find_x_cube_plus_reciprocal_cube_l58_58390


namespace lunch_break_duration_l58_58016

theorem lunch_break_duration (m a : ℝ) (L : ℝ) :
  (9 - L) * (m + a) = 0.6 → 
  (7 - L) * a = 0.3 → 
  (5 - L) * m = 0.1 → 
  L = 42 / 60 :=
by sorry

end lunch_break_duration_l58_58016


namespace train_speed_l58_58627

theorem train_speed (length : ℕ) (time : ℕ) (h_length : length = 1200) (h_time : time = 15) :
  (length / time) = 80 := by
  sorry

end train_speed_l58_58627


namespace hilton_final_marbles_l58_58271

def initial_marbles : ℕ := 26
def found_marbles : ℕ := 6
def lost_marbles : ℕ := 10
def given_marbles := 2 * lost_marbles

theorem hilton_final_marbles (initial_marbles : ℕ) (found_marbles : ℕ) (lost_marbles : ℕ)
  (given_marbles : ℕ) : 
  initial_marbles = 26 →
  found_marbles = 6 →
  lost_marbles = 10 →
  given_marbles = 2 * lost_marbles →
  (initial_marbles + found_marbles - lost_marbles + given_marbles) = 42 :=
by
  intros,
  sorry

end hilton_final_marbles_l58_58271


namespace convex_ngon_sides_l58_58620

theorem convex_ngon_sides (n : ℕ) (h : (n * (n - 3)) / 2 = 27) : n = 9 :=
by
  -- Proof omitted
  sorry

end convex_ngon_sides_l58_58620


namespace die_roll_probability_div_3_l58_58489

noncomputable def probability_divisible_by_3 : ℚ :=
  1 - ((2 : ℚ) / 3) ^ 8

theorem die_roll_probability_div_3 :
  probability_divisible_by_3 = 6305 / 6561 :=
by
  sorry

end die_roll_probability_div_3_l58_58489


namespace problem_l58_58658

open Set

def U : Set ℕ := {0, 1, 2, 3, 4, 5, 6}
def A : Set ℕ := {1, 3, 5}
def B : Set ℕ := {2, 4, 5}
def C : Set ℕ := {1, 3}

theorem problem : A ∩ (U \ B) = C := by
  sorry

end problem_l58_58658


namespace xy_in_N_l58_58986

def M : Set ℤ := {x | ∃ n : ℤ, x = 3 * n + 1}
def N : Set ℤ := {y | ∃ n : ℤ, y = 3 * n - 1}

theorem xy_in_N (x y : ℤ) (hx : x ∈ M) (hy : y ∈ N) : x * y ∈ N := by
  -- hint: use any knowledge and axioms from Mathlib to aid your proof
  sorry

end xy_in_N_l58_58986


namespace exponent_multiplication_l58_58347

theorem exponent_multiplication :
  (5^0.2 * 10^0.4 * 10^0.1 * 10^0.5 * 5^0.8) = 50 := by
  sorry

end exponent_multiplication_l58_58347


namespace scientific_notation_correct_l58_58819

theorem scientific_notation_correct :
  27600 = 2.76 * 10^4 :=
sorry

end scientific_notation_correct_l58_58819


namespace average_daily_sales_l58_58416

def pens_sold_day_one : ℕ := 96
def pens_sold_next_days : ℕ := 44
def total_days : ℕ := 13

theorem average_daily_sales : (pens_sold_day_one + 12 * pens_sold_next_days) / total_days = 48 := 
by 
  sorry

end average_daily_sales_l58_58416


namespace problem_solution_l58_58987

-- Definitions of sets A and B
def A : Set ℝ := {y | ∃ x : ℝ, y = x^2 }
def B : Set ℝ := {-2, -1, 1, 2}

-- Complement of set A in reals
def C_A : Set ℝ := {x | x < 0}

-- Lean theorem statement
theorem problem_solution : (C_A ∩ B) = {-2, -1} :=
by sorry

end problem_solution_l58_58987


namespace sum_of_possible_values_of_N_l58_58438

theorem sum_of_possible_values_of_N :
  (∃ N : ℝ, N * (N - 7) = 12) → (∃ N₁ N₂ : ℝ, (N₁ * (N₁ - 7) = 12 ∧ N₂ * (N₂ - 7) = 12) ∧ N₁ + N₂ = 7) :=
by
  sorry

end sum_of_possible_values_of_N_l58_58438


namespace rulers_added_initially_46_finally_71_l58_58317

theorem rulers_added_initially_46_finally_71 : 
  ∀ (initial final added : ℕ), initial = 46 → final = 71 → added = final - initial → added = 25 :=
by
  intros initial final added h_initial h_final h_added
  rw [h_initial, h_final] at h_added
  exact h_added

end rulers_added_initially_46_finally_71_l58_58317


namespace f_2006_eq_1_l58_58035

noncomputable def f : ℤ → ℤ := sorry
axiom odd_function : ∀ x : ℤ, f (-x) = -f x
axiom period_3 : ∀ x : ℤ, f (3 * (x + 1)) = f (3 * x + 1)
axiom f_at_1 : f 1 = -1

theorem f_2006_eq_1 : f 2006 = 1 := by
  sorry

end f_2006_eq_1_l58_58035


namespace range_of_c_l58_58409

def p (c : ℝ) := (0 < c) ∧ (c < 1)
def q (c : ℝ) := (1 - 2 * c < 0)

theorem range_of_c (c : ℝ) : (p c ∨ q c) ∧ ¬ (p c ∧ q c) ↔ (0 < c ∧ c ≤ 1/2) ∨ (1 < c) :=
by sorry

end range_of_c_l58_58409


namespace percent_employed_in_town_l58_58553

theorem percent_employed_in_town (E : ℝ) : 
  (0.14 * E) + 55 = E → E = 64 :=
by
  intro h
  have h1: 0.14 * E + 55 = E := h
  -- Proof step here, but we put sorry to skip the proof
  sorry

end percent_employed_in_town_l58_58553


namespace cos_half_pi_plus_alpha_l58_58800

theorem cos_half_pi_plus_alpha (α : ℝ) (h : Real.sin (π - α) = 1 / 3) : Real.cos (π / 2 + α) = - (1 / 3) :=
by
  sorry

end cos_half_pi_plus_alpha_l58_58800


namespace symmetric_point_l58_58881

theorem symmetric_point (x0 y0 : ℝ) (P : ℝ × ℝ) (line : ℝ → ℝ) 
  (hP : P = (-1, 3)) (hline : ∀ x, line x = x) :
  ((x0, y0) = (3, -1)) ↔
    ( ∃ M : ℝ × ℝ, M = ((x0 - -1) / 2, (y0 + 3) / 2) ∧ M.1 = M.2 ) ∧ 
    ( ∃ l : ℝ, l = (y0 - 3) / (x0 + 1) ∧ l = -1 ) :=
by
  sorry

end symmetric_point_l58_58881


namespace range_of_a_l58_58267

theorem range_of_a (a : ℝ) (h : ∃ α β : ℝ, (α + β = -(a^2 - 1)) ∧ (α * β = a - 2) ∧ (1 < α ∧ β < 1) ∨ (α < 1 ∧ 1 < β)) :
  -2 < a ∧ a < 1 :=
sorry

end range_of_a_l58_58267


namespace line_intercepts_l58_58064

theorem line_intercepts (x y : ℝ) (P : ℝ × ℝ) (h1 : P = (1, 4)) (h2 : ∃ k : ℝ, (x + y = k ∨ 4 * x - y = 0) ∧ 
  ∃ intercepts_p : ℝ × ℝ, intercepts_p = (k / 2, k / 2)) :
  ∃ k : ℝ, (x + y - k = 0 ∧ k = 5) ∨ (4 * x - y = 0) :=
sorry

end line_intercepts_l58_58064


namespace shopkeeper_profit_percentage_l58_58236

theorem shopkeeper_profit_percentage 
  (cost_price : ℝ := 100) 
  (loss_due_to_theft_percent : ℝ := 30) 
  (overall_loss_percent : ℝ := 23) 
  (remaining_goods_value : ℝ := 70) 
  (overall_loss_value : ℝ := 23) 
  (selling_price : ℝ := 77) 
  (profit_percentage : ℝ) 
  (h1 : remaining_goods_value = cost_price * (1 - loss_due_to_theft_percent / 100)) 
  (h2 : overall_loss_value = cost_price * (overall_loss_percent / 100)) 
  (h3 : selling_price = cost_price - overall_loss_value) 
  (h4 : remaining_goods_value + remaining_goods_value * profit_percentage / 100 = selling_price) :
  profit_percentage = 10 := 
by 
  sorry

end shopkeeper_profit_percentage_l58_58236


namespace annual_income_of_A_l58_58196

variable (Cm : ℝ)
variable (Bm : ℝ)
variable (Am : ℝ)
variable (Aa : ℝ)

-- Given conditions
axiom h1 : Cm = 12000
axiom h2 : Bm = Cm + 0.12 * Cm
axiom h3 : (Am / Bm) = 5 / 2

-- Statement to prove
theorem annual_income_of_A : Aa = 403200 := by
  sorry

end annual_income_of_A_l58_58196


namespace product_xyz_one_l58_58529

theorem product_xyz_one (x y z : ℝ) (h1 : x + 1/y = 2) (h2 : y + 1/z = 2) (h3 : z + 1/x = 2) : x * y * z = 1 := 
by {
    sorry
}

end product_xyz_one_l58_58529


namespace cos_triple_sum_div_l58_58115

theorem cos_triple_sum_div {A B C : ℝ} (h : Real.cos A + Real.cos B + Real.cos C = 0) : 
  (Real.cos (3 * A) + Real.cos (3 * B) + Real.cos (3 * C)) / (Real.cos A * Real.cos B * Real.cos C) = 12 :=
by
  sorry

end cos_triple_sum_div_l58_58115


namespace total_days_2000_to_2003_correct_l58_58111

-- Define the days in each type of year
def days_in_leap_year : ℕ := 366
def days_in_common_year : ℕ := 365

-- Define each year and its corresponding number of days
def year_2000 := days_in_leap_year
def year_2001 := days_in_common_year
def year_2002 := days_in_common_year
def year_2003 := days_in_common_year

-- Calculate the total number of days from 2000 to 2003
def total_days_2000_to_2003 : ℕ := year_2000 + year_2001 + year_2002 + year_2003

theorem total_days_2000_to_2003_correct : total_days_2000_to_2003 = 1461 := 
by
  unfold total_days_2000_to_2003 year_2000 year_2001 year_2002 year_2003 
        days_in_leap_year days_in_common_year 
  exact rfl

end total_days_2000_to_2003_correct_l58_58111


namespace total_books_l58_58970

-- Define the number of books Stu has
def Stu_books : ℕ := 9

-- Define the multiplier for Albert's books
def Albert_multiplier : ℕ := 4

-- Define the number of books Albert has
def Albert_books : ℕ := Albert_multiplier * Stu_books

-- Prove that the total number of books is 45
theorem total_books:
  Stu_books + Albert_books = 45 :=
by 
  -- This is where the proof steps would go, but we skip it for now 
  sorry

end total_books_l58_58970


namespace desired_cost_of_mixture_l58_58060

theorem desired_cost_of_mixture 
  (w₈ : ℝ) (c₈ : ℝ) -- weight and cost per pound of the $8 candy
  (w₅ : ℝ) (c₅ : ℝ) -- weight and cost per pound of the $5 candy
  (total_w : ℝ) (desired_cost : ℝ) -- total weight and desired cost per pound of the mixture
  (h₁ : w₈ = 30) (h₂ : c₈ = 8) 
  (h₃ : w₅ = 60) (h₄ : c₅ = 5)
  (h₅ : total_w = w₈ + w₅)
  (h₆ : desired_cost = (w₈ * c₈ + w₅ * c₅) / total_w) :
  desired_cost = 6 := 
by
  sorry

end desired_cost_of_mixture_l58_58060


namespace inequality_proof_l58_58523

theorem inequality_proof (a b : ℝ) (h : a > b ∧ b > 0) : 
  (a - b)^2 / (8 * a) < (a + b) / 2 - Real.sqrt (a * b) ∧ (a + b) / 2 - Real.sqrt (a * b) < (a - b)^2 / (8 * b) := 
by 
  sorry

end inequality_proof_l58_58523


namespace least_n_froods_l58_58981

theorem least_n_froods (n : ℕ) : (∃ n, n ≥ 30 ∧ (n * (n + 1)) / 2 > 15 * n) ∧ (∀ m < 30, (m * (m + 1)) / 2 ≤ 15 * m) :=
sorry

end least_n_froods_l58_58981


namespace parabola_shift_l58_58718

theorem parabola_shift (x : ℝ) : 
  let y := -2 * x^2 
  let y1 := -2 * (x + 1)^2 
  let y2 := y1 - 3 
  y2 = -2 * x^2 - 4 * x - 5 := 
by 
  sorry

end parabola_shift_l58_58718


namespace exists_intersecting_line_l58_58098

/-- Represents a segment as a pair of endpoints in a 2D plane. -/
structure Segment where
  x : ℝ
  y1 : ℝ
  y2 : ℝ

open Segment

/-- Given several parallel segments with the property that for any three of these segments, 
there exists a line that intersects all three of them, prove that 
there is a line that intersects all the segments. -/
theorem exists_intersecting_line (segments : List Segment)
  (h : ∀ s1 s2 s3 : Segment, s1 ∈ segments → s2 ∈ segments → s3 ∈ segments → 
       ∃ a b : ℝ, (s1.y1 <= a * s1.x + b) ∧ (a * s1.x + b <= s1.y2) ∧ 
                   (s2.y1 <= a * s2.x + b) ∧ (a * s2.x + b <= s2.y2) ∧ 
                   (s3.y1 <= a * s3.x + b) ∧ (a * s3.x + b <= s3.y2)) :
  ∃ a b : ℝ, ∀ s : Segment, s ∈ segments → (s.y1 <= a * s.x + b) ∧ (a * s.x + b <= s.y2) := 
sorry

end exists_intersecting_line_l58_58098


namespace remaining_time_for_P_l58_58465

theorem remaining_time_for_P 
  (P_rate : ℝ) (Q_rate : ℝ) (together_time : ℝ) (remaining_time_minutes : ℝ)
  (hP_rate : P_rate = 1 / 3) 
  (hQ_rate : Q_rate = 1 / 18) 
  (h_together_time : together_time = 2) 
  (h_remaining_time_minutes : remaining_time_minutes = 40) :
  (((P_rate + Q_rate) * together_time) + P_rate * (remaining_time_minutes / 60)) = 1 :=
by  rw [hP_rate, hQ_rate, h_together_time, h_remaining_time_minutes]
    admit

end remaining_time_for_P_l58_58465


namespace lcm_of_two_numbers_l58_58743

theorem lcm_of_two_numbers (a b : ℕ) (h_prod : a * b = 145862784) (h_hcf : Nat.gcd a b = 792) : Nat.lcm a b = 184256 :=
by {
  sorry
}

end lcm_of_two_numbers_l58_58743


namespace remainder_76_pow_77_mod_7_l58_58229

theorem remainder_76_pow_77_mod_7 : (76 ^ 77) % 7 = 6 := 
by 
  sorry 

end remainder_76_pow_77_mod_7_l58_58229


namespace average_s_t_l58_58580

theorem average_s_t (s t : ℝ) 
  (h : (1 + 3 + 7 + s + t) / 5 = 12) : 
  (s + t) / 2 = 24.5 :=
by
  sorry

end average_s_t_l58_58580


namespace solve_equation_l58_58315

theorem solve_equation (x : ℝ) :
  ((x - 2)^2 - 4 = 0) ↔ (x = 4 ∨ x = 0) :=
by
  sorry

end solve_equation_l58_58315


namespace shaded_area_in_rectangle_is_correct_l58_58738

noncomputable def percentage_shaded_area : ℝ :=
  let side_length_congruent_squares := 10
  let side_length_small_square := 5
  let rect_length := 20
  let rect_width := 15
  let rect_area := rect_length * rect_width
  let overlap_congruent_squares := side_length_congruent_squares * rect_width
  let overlap_small_square := (side_length_small_square / 2) * side_length_small_square
  let total_shaded_area := overlap_congruent_squares + overlap_small_square
  (total_shaded_area / rect_area) * 100

theorem shaded_area_in_rectangle_is_correct :
  percentage_shaded_area = 54.17 :=
sorry

end shaded_area_in_rectangle_is_correct_l58_58738


namespace find_third_number_l58_58877

-- Definitions
def A : ℕ := 600
def B : ℕ := 840
def LCM : ℕ := 50400
def HCF : ℕ := 60

-- Theorem to be proven
theorem find_third_number (C : ℕ) (h_lcm : Nat.lcm (Nat.lcm A B) C = LCM) (h_hcf : Nat.gcd (Nat.gcd A B) C = HCF) : C = 6 :=
by -- proof
  sorry

end find_third_number_l58_58877


namespace sales_professionals_count_l58_58492

theorem sales_professionals_count :
  (∀ (C : ℕ) (MC : ℕ) (M : ℕ), C = 500 → MC = 10 → M = 5 → C / M / MC = 10) :=
by
  intros C MC M hC hMC hM
  sorry

end sales_professionals_count_l58_58492


namespace ticket_cost_calculation_l58_58404

theorem ticket_cost_calculation :
  let adult_price := 12
  let child_price := 10
  let num_adults := 3
  let num_children := 3
  let total_cost := (num_adults * adult_price) + (num_children * child_price)
  total_cost = 66 := 
by
  rfl -- or add sorry to skip proof

end ticket_cost_calculation_l58_58404


namespace find_weight_of_b_l58_58188

theorem find_weight_of_b (A B C : ℝ)
  (h1 : (A + B + C) / 3 = 45)
  (h2 : (A + B) / 2 = 40)
  (h3 : (B + C) / 2 = 43) : B = 31 :=
sorry

end find_weight_of_b_l58_58188


namespace points_earned_l58_58397

-- Definitions of the types of enemies and their point values
def points_A := 10
def points_B := 15
def points_C := 20

-- Number of each type of enemies in the level
def num_A_total := 3
def num_B_total := 2
def num_C_total := 3

-- Number of each type of enemies defeated
def num_A_defeated := num_A_total -- 3 Type A enemies
def num_B_defeated := 1 -- Half of 2 Type B enemies
def num_C_defeated := 1 -- 1 Type C enemy

-- Calculation of total points earned
def total_points : ℕ :=
  num_A_defeated * points_A + num_B_defeated * points_B + num_C_defeated * points_C

-- Proof that the total points earned is 65
theorem points_earned : total_points = 65 := by
  -- Placeholder for the proof, which calculates the total points
  sorry

end points_earned_l58_58397


namespace translation_correctness_l58_58042

theorem translation_correctness :
  ( ∀ (x : ℝ), ((x + 4)^2 - 5) = ((x + 4)^2 - 5) ) :=
by
  sorry

end translation_correctness_l58_58042


namespace two_f_of_x_l58_58277

noncomputable def f (x : ℝ) : ℝ := 3 / (3 + x)

theorem two_f_of_x (x : ℝ) (h : x > 0) : 2 * f x = 18 / (9 + x) :=
  sorry

end two_f_of_x_l58_58277


namespace max_q_value_l58_58147

theorem max_q_value (A M C : ℕ) (h : A + M + C = 15) : 
  (A * M * C + A * M + M * C + C * A) ≤ 200 :=
sorry

end max_q_value_l58_58147


namespace third_diff_n_cube_is_const_6_third_diff_general_form_is_6_l58_58712

-- Define the first finite difference function
def delta (f : ℕ → ℤ) (n : ℕ) : ℤ := f (n + 1) - f n

-- Define the second finite difference using the first
def delta2 (f : ℕ → ℤ) (n : ℕ) : ℤ := delta (delta f) n

-- Define the third finite difference using the second
def delta3 (f : ℕ → ℤ) (n : ℕ) : ℤ := delta (delta2 f) n

-- Prove the third finite difference of n^3 is 6
theorem third_diff_n_cube_is_const_6 :
  delta3 (fun (n : ℕ) => (n : ℤ)^3) = fun _ => 6 := 
by
  sorry

-- Prove the third finite difference of the general form function is 6
theorem third_diff_general_form_is_6 (a b c : ℤ) :
  delta3 (fun (n : ℕ) => (n : ℤ)^3 + a * (n : ℤ)^2 + b * (n : ℤ) + c) = fun _ => 6 := 
by
  sorry

end third_diff_n_cube_is_const_6_third_diff_general_form_is_6_l58_58712


namespace tire_mileage_problem_l58_58338

/- Definitions -/
def total_miles : ℕ := 45000
def enhancement_ratio : ℚ := 1.2
def total_tire_miles : ℚ := 180000

/- Question as theorem -/
theorem tire_mileage_problem
  (x y : ℚ)
  (h1 : y = enhancement_ratio * x)
  (h2 : 4 * x + y = total_tire_miles) :
  (x = 34615 ∧ y = 41538) :=
sorry

end tire_mileage_problem_l58_58338


namespace general_formula_for_sequences_c_seq_is_arithmetic_fn_integer_roots_l58_58107

noncomputable def a_seq (n : ℕ) : ℕ :=
  if h : n > 0 then n else 1

noncomputable def b_seq (n : ℕ) : ℚ :=
  if h : n > 0 then n * (n - 1) / 4 else 0

noncomputable def c_seq (n : ℕ) : ℚ :=
  a_seq n ^ 2 - 4 * b_seq n

theorem general_formula_for_sequences (n : ℕ) (h : n > 0) :
  a_seq n = n ∧ b_seq n = (n * (n - 1)) / 4 :=
sorry

theorem c_seq_is_arithmetic (n : ℕ) (h : n > 0) : 
  ∀ m : ℕ, (h2 : m > 0) -> c_seq (m+1) - c_seq m = 1 :=
sorry

theorem fn_integer_roots (n : ℕ) : 
  ∃ k : ℤ, n = k ^ 2 ∧ k ≠ 0 :=
sorry

end general_formula_for_sequences_c_seq_is_arithmetic_fn_integer_roots_l58_58107


namespace bilion_wins_1000000_dollars_l58_58737

theorem bilion_wins_1000000_dollars :
  ∃ (p : ℕ), (p = 1000000) ∧ (p % 3 = 1) → p = 1000000 :=
by
  sorry

end bilion_wins_1000000_dollars_l58_58737


namespace axis_of_symmetry_eq_l58_58871

theorem axis_of_symmetry_eq : 
  ∃ k : ℤ, (λ x => 2 * Real.cos (2 * x)) = (λ x => 2 * Real.sin (2 * (x + π / 3) - π / 6)) ∧
            x = (1/2) * k * π ∧ x = -π / 2 := 
by
  sorry

end axis_of_symmetry_eq_l58_58871


namespace sid_fraction_left_l58_58872

noncomputable def fraction_left (original total_spent remaining additional : ℝ) : ℝ :=
  (remaining - additional) / original

theorem sid_fraction_left 
  (original : ℝ := 48) 
  (spent_computer : ℝ := 12) 
  (spent_snacks : ℝ := 8) 
  (remaining : ℝ := 28) 
  (additional : ℝ := 4) :
  fraction_left original (spent_computer + spent_snacks) remaining additional = 1 / 2 :=
by
  sorry

end sid_fraction_left_l58_58872


namespace below_sea_level_notation_l58_58841

theorem below_sea_level_notation (depth_above_sea_level : Int) (depth_below_sea_level: Int) 
  (h: depth_above_sea_level = 9050 ∧ depth_below_sea_level = -10907) : 
  depth_below_sea_level = -10907 :=
by 
  sorry

end below_sea_level_notation_l58_58841


namespace problem_part_1_solution_set_of_f_when_a_is_3_problem_part_2_range_of_a_l58_58571

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := abs (2 * x - a) + a

theorem problem_part_1_solution_set_of_f_when_a_is_3 :
  {x : ℝ | 0 ≤ x ∧ x ≤ 3} = {x : ℝ | f x 3 ≤ 6} :=
by
  sorry

def g (x : ℝ) : ℝ := abs (2 * x - 3)

theorem problem_part_2_range_of_a :
  {a : ℝ | 4 ≤ a} = {a : ℝ | ∀ x : ℝ, f x a + g x ≥ 5} :=
by
  sorry

end problem_part_1_solution_set_of_f_when_a_is_3_problem_part_2_range_of_a_l58_58571


namespace symmetric_about_y_l58_58795

theorem symmetric_about_y (m n : ℤ) (h1 : 2 * n - m = -14) (h2 : m = 4) : (m + n) ^ 2023 = -1 := by
  sorry

end symmetric_about_y_l58_58795


namespace positive_number_sum_square_l58_58444

theorem positive_number_sum_square (n : ℝ) (h : n^2 + n = 210) (h_pos : n > 0) : n = 14 :=
sorry

end positive_number_sum_square_l58_58444


namespace proportion_sets_l58_58772

-- Define unit lengths for clarity
def length (n : ℕ) := n 

-- Define the sets of line segments
def setA := (length 4, length 5, length 6, length 7)
def setB := (length 3, length 4, length 5, length 8)
def setC := (length 5, length 15, length 3, length 9)
def setD := (length 8, length 4, length 1, length 3)

-- Define a condition for a set to form a proportion
def is_proportional (a b c d : ℕ) : Prop :=
  a * d = b * c

-- Main theorem: setC forms a proportion while others don't
theorem proportion_sets : is_proportional 5 15 3 9 ∧ 
                         ¬ is_proportional 4 5 6 7 ∧ 
                         ¬ is_proportional 3 4 5 8 ∧ 
                         ¬ is_proportional 8 4 1 3 := by
  sorry

end proportion_sets_l58_58772


namespace smallest_solution_l58_58791

noncomputable def equation (x : ℝ) : Prop :=
  (1 / (x - 3) + 1 / (x - 5) + 1 / (x - 6) = 4 / (x - 4))

def valid_x (x : ℝ) : Prop :=
  x ≠ 3 ∧ x ≠ 4 ∧ x ≠ 5 ∧ x ≠ 6

theorem smallest_solution (x : ℝ) (h1 : equation x) (h2 : valid_x x) : x = 16 := sorry

end smallest_solution_l58_58791


namespace total_number_of_people_on_bus_l58_58171

theorem total_number_of_people_on_bus (boys girls : ℕ)
    (driver assistant teacher : ℕ) 
    (h1 : boys = 50)
    (h2 : girls = boys + (2 * boys / 5))
    (h3 : driver = 1)
    (h4 : assistant = 1)
    (h5 : teacher = 1) :
    (boys + girls + driver + assistant + teacher = 123) :=
by
    sorry

end total_number_of_people_on_bus_l58_58171


namespace polygon_sides_eq_eight_l58_58975

theorem polygon_sides_eq_eight (n : ℕ) (h1 : (n - 2) * 180 = 3 * 360) : n = 8 :=
sorry

end polygon_sides_eq_eight_l58_58975


namespace monthly_rent_l58_58130

-- Definition
def total_amount_saved := 2225
def extra_amount_needed := 775
def deposit := 500

-- Total amount required
def total_amount_required := total_amount_saved + extra_amount_needed
def total_rent_plus_deposit (R : ℝ) := 2 * R + deposit

-- The statement to prove
theorem monthly_rent (R : ℝ) : total_rent_plus_deposit R = total_amount_required → R = 1250 :=
by
  intros h
  exact sorry -- Proof is omitted.

end monthly_rent_l58_58130


namespace below_sea_level_notation_l58_58840

theorem below_sea_level_notation (depth_above_sea_level : Int) (depth_below_sea_level: Int) 
  (h: depth_above_sea_level = 9050 ∧ depth_below_sea_level = -10907) : 
  depth_below_sea_level = -10907 :=
by 
  sorry

end below_sea_level_notation_l58_58840


namespace solution_problem_l58_58982

open Real

noncomputable def parametric_curve_x (t : ℝ) : ℝ := 2 * cos t
noncomputable def parametric_curve_y (t : ℝ) : ℝ := sin t

def polar_line_equation (ρ θ : ℝ) : Prop :=
  ρ * cos (θ + π / 3) = -√3 / 2

def cartesian_curve_equation (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 = 1

def cartesian_line_equation (x y : ℝ) : Prop :=
  x - √3 * y + √3 = 0

def length_of_segment_AB (A B : ℝ × ℝ) : ℝ :=
  sqrt (1 + (√3 / 3)^2) * (abs ((fst A + fst B) - 4 * 0)) -- A == (x1, y1) and B == (x2, y2)

theorem solution_problem 
  (t : ℝ)
  (ρ θ : ℝ) 
  (A B : ℝ × ℝ) :
  cartesian_curve_equation (parametric_curve_x t) (parametric_curve_y t) ∧
  polar_line_equation ρ θ ∧
  cartesian_line_equation (fst A) (snd A) ∧
  cartesian_line_equation (fst B) (snd B)
  → length_of_segment_AB A B = 32 / 7 :=
by sorry

end solution_problem_l58_58982


namespace vanya_number_l58_58593

theorem vanya_number (m n : ℕ) (hm : m < 10) (hn : n < 10) (h : (10 * n + m)^2 = 4 * (10 * m + n)) : 
  10 * m + n = 81 :=
by sorry

end vanya_number_l58_58593


namespace o_hara_triple_example_l58_58731

-- definitions
def is_OHara_triple (a b x : ℕ) : Prop :=
  (Real.sqrt a) + (Real.sqrt b) = x

-- conditions
def a : ℕ := 81
def b : ℕ := 49
def x : ℕ := 16

-- statement
theorem o_hara_triple_example : is_OHara_triple a b x :=
by
  sorry

end o_hara_triple_example_l58_58731


namespace find_integers_k_l58_58784

theorem find_integers_k (k : ℤ) : 
  (k = 15 ∨ k = 30) ↔ 
  (k ≥ 3 ∧ ∃ m n : ℤ, 1 < m ∧ m < k ∧ 1 < n ∧ n < k ∧ 
                       Int.gcd m k = 1 ∧ Int.gcd n k = 1 ∧ 
                       m + n > k ∧ k ∣ (m - 1) * (n - 1)) :=
by
  sorry -- Proof goes here

end find_integers_k_l58_58784


namespace below_sea_level_notation_l58_58842

theorem below_sea_level_notation (depth_above_sea_level : Int) (depth_below_sea_level: Int) 
  (h: depth_above_sea_level = 9050 ∧ depth_below_sea_level = -10907) : 
  depth_below_sea_level = -10907 :=
by 
  sorry

end below_sea_level_notation_l58_58842


namespace max_heaps_660_l58_58863

noncomputable def max_heaps (total_stones : ℕ) : ℕ := 
  let heaps := list.range' 15 15 ++ list.range' 15 15
  heaps.length

theorem max_heaps_660 : max_heaps 660 = 30 :=
  by
    sorry

end max_heaps_660_l58_58863


namespace area_of_triangle_l58_58983

def triangle (α β γ : Type) : (α ≃ β) ≃ γ ≃ Prop := sorry

variables (α β γ : Type) (AB AC AM : ℝ)
variables (ha : AB = 9) (hb : AC = 17) (hc : AM = 12)

theorem area_of_triangle (α β γ : Type) (AB AC AM : ℝ)
  (ha : AB = 9) (hb : AC = 17) (hc : AM = 12) : 
  ∃ A : ℝ, A = 74 :=
sorry

end area_of_triangle_l58_58983


namespace finite_points_outside_unit_circle_l58_58406

noncomputable def centroid (x y z : ℝ × ℝ) : ℝ × ℝ := 
  ((x.1 + y.1 + z.1) / 3, (x.2 + y.2 + z.2) / 3)

theorem finite_points_outside_unit_circle
  (A₁ B₁ C₁ D₁ : ℝ × ℝ)
  (A : ℕ → ℝ × ℝ)
  (B : ℕ → ℝ × ℝ)
  (C : ℕ → ℝ × ℝ)
  (D : ℕ → ℝ × ℝ)
  (hA : ∀ n, A (n + 1) = centroid (B n) (C n) (D n))
  (hB : ∀ n, B (n + 1) = centroid (A n) (C n) (D n))
  (hC : ∀ n, C (n + 1) = centroid (A n) (B n) (D n))
  (hD : ∀ n, D (n + 1) = centroid (A n) (B n) (C n))
  (h₀ : A 1 = A₁ ∧ B 1 = B₁ ∧ C 1 = C₁ ∧ D 1 = D₁)
  : ∃ N : ℕ, ∀ n > N, (A n).1 * (A n).1 + (A n).2 * (A n).2 ≤ 1 :=
sorry

end finite_points_outside_unit_circle_l58_58406


namespace positive_number_square_sum_eq_210_l58_58456

theorem positive_number_square_sum_eq_210 (n : ℕ) (h : n^2 + n = 210) : n = 14 :=
sorry

end positive_number_square_sum_eq_210_l58_58456


namespace tangent_length_from_A_to_circle_l58_58684

noncomputable def point_A_polar : (ℝ × ℝ) := (6, Real.pi)
noncomputable def circle_eq_polar (θ : ℝ) : ℝ := -4 * Real.cos θ

theorem tangent_length_from_A_to_circle : 
  ∃ (length : ℝ), length = 2 * Real.sqrt 3 ∧ 
  (∃ (ρ θ : ℝ), point_A_polar = (6, Real.pi) ∧ ρ = circle_eq_polar θ) :=
sorry

end tangent_length_from_A_to_circle_l58_58684


namespace probability_heads_not_less_than_tails_is_11_over_16_l58_58462

open ProbabilityTheory

noncomputable def num_of_heads_not_less_than_tails (tosses : list bool) : Prop :=
  tosses.count tt >= tosses.count ff

def all_outcomes_4_tosses : finset (list bool) :=
  finset.univ.image (λ b: fin 4 → bool, list.of_fn b)

def desired_outcomes : finset (list bool) :=
  finset.filter (λ outcome, num_of_heads_not_less_than_tails outcome) all_outcomes_4_tosses

theorem probability_heads_not_less_than_tails_is_11_over_16 :
  (desired_outcomes.card : ℚ) / (all_outcomes_4_tosses.card : ℚ) = 11 / 16 :=
by
  sorry

end probability_heads_not_less_than_tails_is_11_over_16_l58_58462


namespace optionA_optionC_optionD_l58_58806

noncomputable def f (x : ℝ) := (3 : ℝ) ^ x / (1 + (3 : ℝ) ^ x)

theorem optionA : ∀ x : ℝ, f (-x) + f x = 1 := by
  sorry

theorem optionC : ∀ y : ℝ, (∃ x : ℝ, f x = y) ↔ (y > 0 ∧ y < 1) := by
  sorry

theorem optionD : ∀ x : ℝ, f (2 * x - 3) + f (x - 3) > 1 ↔ x > 2 := by
  sorry

end optionA_optionC_optionD_l58_58806


namespace jenna_less_than_bob_l58_58812

def bob_amount : ℕ := 60
def phil_amount : ℕ := (1 / 3) * bob_amount
def jenna_amount : ℕ := 2 * phil_amount

theorem jenna_less_than_bob : bob_amount - jenna_amount = 20 := by
  sorry

end jenna_less_than_bob_l58_58812


namespace depth_below_sea_notation_l58_58831

variables (alt_above_sea : ℝ) (depth_below_sea : ℝ)

def notation_above_sea : ℝ := alt_above_sea

def notation_below_sea (d : ℝ) : ℝ := -d

theorem depth_below_sea_notation : alt_above_sea = 9050 → notation_above_sea = 9050 → depth_below_sea = 10907 → notation_below_sea depth_below_sea = -10907 :=
by
  intros h1 h2 h3
  rw [h3, notation_below_sea]
  exact eq.symm sorry

end depth_below_sea_notation_l58_58831


namespace even_function_a_value_l58_58118

theorem even_function_a_value (a : ℝ) :
  (∀ x : ℝ, (x + 1) * (x - a) = (-x + 1) * (-x - a)) → a = 1 :=
by
  sorry

end even_function_a_value_l58_58118


namespace parabola_chord_ratio_is_3_l58_58575

noncomputable def parabola_chord_ratio (p : ℝ) (h : p > 0) : ℝ :=
  let focus_x := p / 2
  let a_x := (3 * p) / 2
  let b_x := p / 6
  let af := a_x + (p / 2)
  let bf := b_x + (p / 2)
  af / bf

theorem parabola_chord_ratio_is_3 (p : ℝ) (h : p > 0) : parabola_chord_ratio p h = 3 := by
  sorry

end parabola_chord_ratio_is_3_l58_58575


namespace lengths_of_broken_lines_eq_l58_58285

noncomputable theory

open EuclideanGeometry

variable {R : Type*} [Real.R R]

/--
Given:
1. Inside the acute angle X O Y, points M and N are chosen such that \(\angle X O N = \angle Y O M\).
2. A point Q is chosen on segment O X such that \(\angle N Q O = \angle M Q X\).
3. A point P is chosen on segment O Y such that \(\angle N P O = \angle M P Y\).

Prove that:
The lengths of the broken lines M P N and M Q N are equal.
-/
theorem lengths_of_broken_lines_eq 
  {O X Y M N Q P : EuclideanGeometry.Point R}
  (h1 : ∠X O N = ∠Y O M)
  (h2 : Q ∈ OpenSegment O X)
  (h3 : ∠N Q O = ∠M Q X)
  (h4 : P ∈ OpenSegment O Y)
  (h5 : ∠N P O = ∠M P Y) :
  EuclideanGeometry.length (segment M P) + EuclideanGeometry.length (segment P N) =
  EuclideanGeometry.length (segment M Q) + EuclideanGeometry.length (segment Q N) :=
sorry

end lengths_of_broken_lines_eq_l58_58285


namespace sequence_formula_l58_58807

theorem sequence_formula (a : ℕ → ℤ) (S : ℕ → ℤ)
  (h1 : a 1 = 1)
  (h2 : a 2 = 4)
  (h3 : ∀ n : ℕ, n > 0 → a (n + 2) + 2 * a n = 3 * a (n + 1)) :
  (∀ n, a n = 3 * 2^(n-1) - 2) ∧ (S 4 > 21 - 2 * 4) :=
by
  sorry

end sequence_formula_l58_58807


namespace quadratic_complete_square_l58_58076

theorem quadratic_complete_square (x : ℝ) (m t : ℝ) :
  (4 * x^2 - 16 * x - 448 = 0) → ((x + m) ^ 2 = t) → (t = 116) :=
by
  sorry

end quadratic_complete_square_l58_58076


namespace number_of_positive_integer_solutions_l58_58224

theorem number_of_positive_integer_solutions :
  ∃ n : ℕ, n = 84 ∧ (∀ x y z t : ℕ, 0 < x ∧ 0 < y ∧ 0 < z ∧ 0 < t ∧ x + y + z + t = 10 → true) :=
sorry

end number_of_positive_integer_solutions_l58_58224


namespace average_eq_5_times_non_zero_l58_58753

theorem average_eq_5_times_non_zero (x : ℝ) (h1 : x ≠ 0) (h2 : (x + x^2) / 2 = 5 * x) : x = 9 := 
by sorry

end average_eq_5_times_non_zero_l58_58753


namespace Sandy_original_number_l58_58570

theorem Sandy_original_number (x : ℝ) (h : (3 * x + 20)^2 = 2500) : x = 10 :=
by
  sorry

end Sandy_original_number_l58_58570


namespace max_heaps_l58_58862

theorem max_heaps (stone_count : ℕ) (h1 : stone_count = 660) (heaps : list ℕ) 
  (h2 : ∀ a b ∈ heaps, a <= b → b < 2 * a): heaps.length <= 30 :=
sorry

end max_heaps_l58_58862


namespace custom_op_12_7_l58_58974

def custom_op (a b : ℤ) := (a + b) * (a - b)

theorem custom_op_12_7 : custom_op 12 7 = 95 := by
  sorry

end custom_op_12_7_l58_58974


namespace solution_set_inequality_l58_58959

noncomputable def f (x : ℝ) : ℝ := sorry

axiom odd_function : ∀ x : ℝ, f (-x) = -f x
axiom increasing_on_positive : ∀ {x y : ℝ}, 0 < x → x < y → f x < f y
axiom f_one : f 1 = 0

theorem solution_set_inequality :
  {x : ℝ | (f x) / x < 0} = {x : ℝ | x < -1} ∪ {x | 0 < x ∧ x < 1} := sorry

end solution_set_inequality_l58_58959


namespace average_marks_mathematics_chemistry_l58_58765

theorem average_marks_mathematics_chemistry (M P C B : ℕ) 
    (h1 : M + P = 80) 
    (h2 : C + B = 120) 
    (h3 : C = P + 20) 
    (h4 : B = M - 15) : 
    (M + C) / 2 = 50 :=
by
  sorry

end average_marks_mathematics_chemistry_l58_58765


namespace f_g_of_2_eq_4_l58_58676

def f (x : ℝ) : ℝ := x^2 - 2*x + 1
def g (x : ℝ) : ℝ := 2*x - 5

theorem f_g_of_2_eq_4 : f (g 2) = 4 := by
  sorry

end f_g_of_2_eq_4_l58_58676


namespace find_S25_l58_58663

variable (a : ℕ → ℚ) (S : ℕ → ℚ)

-- Conditions: arithmetic sequence {a_n} and sum of the first n terms S_n with S15 - S10 = 1
def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

def sum_of_first_n_terms (a : ℕ → ℚ) (S : ℕ → ℚ) : Prop :=
  ∀ n : ℕ, S n = ∑ i in finset.range(n), a (i + 1)

axiom condition : S 15 - S 10 = 1

-- Question: Prove that S25 = 5
theorem find_S25 (h1 : arithmetic_sequence a) (h2 : sum_of_first_n_terms a S) : S 25 = 5 :=
by
  sorry

end find_S25_l58_58663


namespace find_sale4_l58_58478

variable (sale1 sale2 sale3 sale5 sale6 avg : ℕ)
variable (total_sales : ℕ := 6 * avg)
variable (known_sales : ℕ := sale1 + sale2 + sale3 + sale5 + sale6)
variable (sale4 : ℕ := total_sales - known_sales)

theorem find_sale4 (h1 : sale1 = 6235) (h2 : sale2 = 6927) (h3 : sale3 = 6855)
                   (h5 : sale5 = 6562) (h6 : sale6 = 5191) (h_avg : avg = 6500) :
  sale4 = 7225 :=
by 
  sorry

end find_sale4_l58_58478


namespace prime_divisor_problem_l58_58512

theorem prime_divisor_problem (d r : ℕ) (h1 : d > 1) (h2 : Prime d)
  (h3 : 1274 % d = r) (h4 : 1841 % d = r) (h5 : 2866 % d = r) : d - r = 6 :=
by
  sorry

end prime_divisor_problem_l58_58512


namespace bus_total_people_l58_58172

def number_of_boys : ℕ := 50
def additional_girls (b : ℕ) : ℕ := (2 * b) / 5
def number_of_girls (b : ℕ) : ℕ := b + additional_girls b
def total_people (b g : ℕ) : ℕ := b + g + 3  -- adding 3 for the driver, assistant, and teacher

theorem bus_total_people : total_people number_of_boys (number_of_girls number_of_boys) = 123 :=
by
  sorry

end bus_total_people_l58_58172


namespace third_square_area_difference_l58_58890

def side_length (p : ℕ) : ℕ :=
  p / 4

def area (s : ℕ) : ℕ :=
  s * s

theorem third_square_area_difference
  (p1 p2 p3 : ℕ)
  (h1 : p1 = 60)
  (h2 : p2 = 48)
  (h3 : p3 = 36)
  : area (side_length p3) = area (side_length p1) - area (side_length p2) :=
by
  sorry

end third_square_area_difference_l58_58890


namespace complement_of_M_l58_58381

open Set

-- Define the universal set
def U : Set ℝ := univ

-- Define the set M
def M : Set ℝ := {y | ∃ x : ℝ, y = x^2 - 1}

-- The theorem stating the complement of M in U
theorem complement_of_M : (U \ M) = {y | y < -1} :=
by
  sorry

end complement_of_M_l58_58381


namespace prime_ge_7_divides_30_l58_58389

theorem prime_ge_7_divides_30 (p : ℕ) (hp : p ≥ 7) (hp_prime : Nat.Prime p) : 30 ∣ (p^2 - 1) := by
  sorry

end prime_ge_7_divides_30_l58_58389


namespace n_cubed_minus_9n_plus_27_not_div_by_81_l58_58301

theorem n_cubed_minus_9n_plus_27_not_div_by_81 (n : ℤ) : ¬ 81 ∣ (n^3 - 9 * n + 27) :=
sorry

end n_cubed_minus_9n_plus_27_not_div_by_81_l58_58301


namespace denote_depth_below_sea_level_l58_58835

theorem denote_depth_below_sea_level (above_sea_level : Int) (depth_haidou_1 : Int) :
  (above_sea_level > 0) ∧ (depth_haidou_1 < 0) → depth_haidou_1 = -10907 :=
by
  intros h
  sorry

end denote_depth_below_sea_level_l58_58835


namespace proof_2720000_scientific_l58_58876

def scientific_notation (n : ℕ) : ℝ := 
  2.72 * 10^6 

theorem proof_2720000_scientific :
  scientific_notation 2720000 = 2.72 * 10^6 := by
  sorry

end proof_2720000_scientific_l58_58876


namespace cube_loop_probability_l58_58061

-- Define the number of faces and alignments for a cube
def total_faces := 6
def stripe_orientations_per_face := 2

-- Define the total possible stripe combinations
def total_stripe_combinations := stripe_orientations_per_face ^ total_faces

-- Define the combinations for both vertical and horizontal loops
def vertical_and_horizontal_loop_combinations := 64

-- Define the probability space
def probability_at_least_one_each := vertical_and_horizontal_loop_combinations / total_stripe_combinations

-- The main theorem to state the probability of having at least one vertical and one horizontal loop
theorem cube_loop_probability : probability_at_least_one_each = 1 := by
  sorry

end cube_loop_probability_l58_58061


namespace area_of_region_W_l58_58023

structure Rhombus (P Q R T : Type) :=
  (side_length : ℝ)
  (angle_Q : ℝ)

def Region_W
  (P Q R T : Type)
  (r : Rhombus P Q R T)
  (h_side : r.side_length = 5)
  (h_angle : r.angle_Q = 90) : ℝ :=
6.25

theorem area_of_region_W
  {P Q R T : Type}
  (r : Rhombus P Q R T)
  (h_side : r.side_length = 5)
  (h_angle : r.angle_Q = 90) :
  Region_W P Q R T r h_side h_angle = 6.25 :=
sorry

end area_of_region_W_l58_58023


namespace intersection_M_N_l58_58561

def M : Set ℝ := {x | x^2 - x ≤ 0}
def N : Set ℝ := {x | -1 < x ∧ x < 1}

theorem intersection_M_N : (M ∩ N) = {x | 0 ≤ x ∧ x < 1} :=
by {
  sorry
}

end intersection_M_N_l58_58561


namespace polynomial_divisible_by_x_minus_4_l58_58939

theorem polynomial_divisible_by_x_minus_4 (m : ℤ) :
  (∀ x, 6 * x ^ 3 - 12 * x ^ 2 + m * x - 24 = 0 → x = 4) ↔ m = -42 :=
by
  sorry

end polynomial_divisible_by_x_minus_4_l58_58939


namespace smallest_m_for_no_real_solution_l58_58102

theorem smallest_m_for_no_real_solution : 
  (∀ x : ℝ, ∀ m : ℝ, (m * x^2 - 3 * x + 1 = 0) → false) ↔ (m ≥ 3) :=
by
  sorry

end smallest_m_for_no_real_solution_l58_58102


namespace sum_of_roots_l58_58112

theorem sum_of_roots (x : ℝ) :
  (x + 2) * (x - 3) = 16 →
  ∃ a b : ℝ, (a ≠ x ∧ b ≠ x ∧ (x - a) * (x - b) = 0) ∧
             (a + b = 1) :=
by
  intro h
  sorry

end sum_of_roots_l58_58112


namespace Cody_book_series_total_count_l58_58247

theorem Cody_book_series_total_count :
  ∀ (weeks: ℕ) (books_first_week: ℕ) (books_second_week: ℕ) (books_per_week_after: ℕ),
    weeks = 7 ∧ books_first_week = 6 ∧ books_second_week = 3 ∧ books_per_week_after = 9 →
    (books_first_week + books_second_week + (weeks - 2) * books_per_week_after) = 54 :=
by
  sorry

end Cody_book_series_total_count_l58_58247


namespace max_rubles_earned_l58_58728

theorem max_rubles_earned :
  ∀ (cards_with_1 cards_with_2 : ℕ), 
  cards_with_1 = 2013 ∧ cards_with_2 = 2013 →
  ∃ (max_moves : ℕ), max_moves = 5 :=
by
  intros cards_with_1 cards_with_2 h
  sorry

end max_rubles_earned_l58_58728


namespace area_of_triangle_ABC_l58_58752

/--
Given a triangle \(ABC\) with points \(D\) and \(E\) on sides \(BC\) and \(AC\) respectively,
where \(BD = 4\), \(DE = 2\), \(EC = 6\), and \(BF = FC = 3\),
proves that the area of triangle \( \triangle ABC \) is \( 18\sqrt{3} \).
-/
theorem area_of_triangle_ABC :
  ∀ (ABC D E : Type) (BD DE EC BF FC : ℝ),
    BD = 4 → DE = 2 → EC = 6 → BF = 3 → FC = 3 → 
    ∃ area, area = 18 * Real.sqrt 3 :=
by
  intros ABC D E BD DE EC BF FC hBD hDE hEC hBF hFC
  sorry

end area_of_triangle_ABC_l58_58752


namespace find_y_l58_58996

theorem find_y (x y : ℝ) (h1 : (100 + 200 + 300 + x) / 4 = 250) (h2 : (300 + 150 + 100 + x + y) / 5 = 200) : y = 50 :=
by
  sorry

end find_y_l58_58996


namespace positive_number_square_sum_eq_210_l58_58457

theorem positive_number_square_sum_eq_210 (n : ℕ) (h : n^2 + n = 210) : n = 14 :=
sorry

end positive_number_square_sum_eq_210_l58_58457


namespace jina_total_mascots_l58_58401

-- Definitions and Conditions
def num_teddies := 5
def num_bunnies := 3 * num_teddies
def num_koala_bears := 1
def additional_teddies := 2 * num_bunnies

-- Total mascots calculation
def total_mascots := num_teddies + num_bunnies + num_koala_bears + additional_teddies

theorem jina_total_mascots : total_mascots = 51 := by
  sorry

end jina_total_mascots_l58_58401


namespace mindy_mork_earnings_ratio_l58_58934

theorem mindy_mork_earnings_ratio (M K : ℝ) (h1 : 0.20 * M + 0.30 * K = 0.225 * (M + K)) : M / K = 3 :=
by
  sorry

end mindy_mork_earnings_ratio_l58_58934


namespace no_real_roots_abs_eq_l58_58884

theorem no_real_roots_abs_eq (x : ℝ) : 
  |2*x - 5| + |3*x - 7| + |5*x - 11| = 2015/2016 → false :=
by sorry

end no_real_roots_abs_eq_l58_58884


namespace james_needs_to_sell_12_coins_l58_58126

theorem james_needs_to_sell_12_coins:
  ∀ (num_coins : ℕ) (initial_price new_price : ℝ),
  num_coins = 20 ∧ initial_price = 15 ∧ new_price = initial_price + (2 / 3) * initial_price →
  (num_coins * initial_price) / new_price = 12 :=
by
  intros num_coins initial_price new_price h
  obtain ⟨hc1, hc2, hc3⟩ := h
  sorry

end james_needs_to_sell_12_coins_l58_58126


namespace geom_seq_sum_l58_58121

theorem geom_seq_sum (a : ℕ → ℝ) (q : ℝ) (h1 : a 1 = 3)
  (h2 : a 1 + a 2 + a 3 = 21)
  (h3 : ∀ n, a (n + 1) = a n * q) : a 4 + a 5 + a 6 = 168 :=
sorry

end geom_seq_sum_l58_58121


namespace convert_point_cylindrical_to_rectangular_l58_58350

noncomputable def cylindrical_to_rectangular_coordinates (r θ z : ℝ) : ℝ × ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ, z)

theorem convert_point_cylindrical_to_rectangular :
  cylindrical_to_rectangular_coordinates 6 (5 * Real.pi / 3) (-3) = (3, -3 * Real.sqrt 3, -3) :=
by
  sorry

end convert_point_cylindrical_to_rectangular_l58_58350


namespace jordan_Oreos_count_l58_58129

variable (J : ℕ)
variable (OreosTotal : ℕ)
variable (JamesOreos : ℕ)

axiom James_Oreos_condition : JamesOreos = 2 * J + 3
axiom Oreos_total_condition : J + JamesOreos = OreosTotal
axiom Oreos_total_value : OreosTotal = 36

theorem jordan_Oreos_count : J = 11 :=
by 
  unfold OreosTotal JamesOreos
  sorry

end jordan_Oreos_count_l58_58129


namespace sum_of_two_numbers_l58_58179

theorem sum_of_two_numbers (L S : ℕ) (hL : L = 22) (hExceeds : L = S + 10) : L + S = 34 := by
  sorry

end sum_of_two_numbers_l58_58179


namespace ratio_of_ian_to_jessica_l58_58421

/-- 
Rodney has 35 dollars more than Ian. 
Jessica has 100 dollars. 
Jessica has 15 dollars more than Rodney. 
Prove that the ratio of Ian's money to Jessica's money is 1/2.
-/
theorem ratio_of_ian_to_jessica (I R J : ℕ) (h1 : R = I + 35) (h2 : J = 100) (h3 : J = R + 15) :
  I / J = 1 / 2 :=
by
  sorry

end ratio_of_ian_to_jessica_l58_58421


namespace simplify_expression_1_simplify_expression_2_l58_58227

-- Problem 1
theorem simplify_expression_1 (a b : ℤ) : a + 2 * b + 3 * a - 2 * b = 4 * a :=
by
  sorry

-- Problem 2
theorem simplify_expression_2 (m n : ℤ) (h_m : m = 2) (h_n : n = 1) :
  (2 * m ^ 2 - 3 * m * n + 8) - (5 * m * n - 4 * m ^ 2 + 8) = 8 :=
by
  sorry

end simplify_expression_1_simplify_expression_2_l58_58227


namespace decimal_equivalent_of_quarter_cubed_l58_58322

theorem decimal_equivalent_of_quarter_cubed :
    (1 / 4 : ℝ) ^ 3 = 0.015625 := 
by
    sorry

end decimal_equivalent_of_quarter_cubed_l58_58322


namespace required_vases_l58_58929

def vase_capacity_roses : Nat := 6
def vase_capacity_tulips : Nat := 8
def vase_capacity_lilies : Nat := 4

def remaining_roses : Nat := 20
def remaining_tulips : Nat := 15
def remaining_lilies : Nat := 5

def vases_for_roses : Nat := (remaining_roses + vase_capacity_roses - 1) / vase_capacity_roses
def vases_for_tulips : Nat := (remaining_tulips + vase_capacity_tulips - 1) / vase_capacity_tulips
def vases_for_lilies : Nat := (remaining_lilies + vase_capacity_lilies - 1) / vase_capacity_lilies

def total_vases_needed : Nat := vases_for_roses + vases_for_tulips + vases_for_lilies

theorem required_vases : total_vases_needed = 8 := by
  sorry

end required_vases_l58_58929


namespace three_sum_xyz_l58_58391

theorem three_sum_xyz (x y z : ℝ) 
  (h1 : y + z = 18 - 4 * x) 
  (h2 : x + z = 22 - 4 * y) 
  (h3 : x + y = 15 - 4 * z) : 
  3 * x + 3 * y + 3 * z = 55 / 2 := 
  sorry

end three_sum_xyz_l58_58391


namespace domain_f_l58_58352

noncomputable def f (x : ℝ) : ℝ := (x - 2) ^ (1 / 2) + 1 / (x - 3)

theorem domain_f :
  {x : ℝ | x ≥ 2 ∧ x ≠ 3 } = {x : ℝ | (2 ≤ x ∧ x < 3) ∨ (3 < x)} :=
by
  sorry

end domain_f_l58_58352


namespace max_min_f_l58_58514

noncomputable def f (x : ℝ) : ℝ := 
  5 * Real.cos x ^ 2 - 6 * Real.sin (2 * x) + 20 * Real.sin x - 30 * Real.cos x + 7

theorem max_min_f :
  (∃ x : ℝ, f x = 16 + 10 * Real.sqrt 13) ∧
  (∃ x : ℝ, f x = 16 - 10 * Real.sqrt 13) :=
sorry

end max_min_f_l58_58514


namespace algebraic_expression_value_l58_58804

theorem algebraic_expression_value (x y : ℝ) (h : x^4 + 6*x^2*y + 9*y^2 + 2*x^2 + 6*y + 4 = 7) :
(x^4 + 6*x^2*y + 9*y^2 - 2*x^2 - 6*y - 1 = -2) ∨ (x^4 + 6*x^2*y + 9*y^2 - 2*x^2 - 6*y - 1 = 14) :=
sorry

end algebraic_expression_value_l58_58804


namespace rectangle_width_l58_58521

theorem rectangle_width (r l w : ℝ) (h_r : r = Real.sqrt 12) (h_l : l = 3 * Real.sqrt 2)
  (h_area_eq: Real.pi * r^2 = l * w) : w = 2 * Real.sqrt 2 * Real.pi :=
by
  sorry

end rectangle_width_l58_58521


namespace frosting_cans_needed_l58_58078

theorem frosting_cans_needed :
  let daily_cakes := 10
  let days := 5
  let total_cakes := daily_cakes * days
  let eaten_cakes := 12
  let remaining_cakes := total_cakes - eaten_cakes
  let cans_per_cake := 2
  let total_cans := remaining_cakes * cans_per_cake
  total_cans = 76 := 
by
  sorry

end frosting_cans_needed_l58_58078


namespace martha_weight_l58_58635

theorem martha_weight :
  ∀ (Bridget_weight : ℕ) (difference : ℕ) (Martha_weight : ℕ),
  Bridget_weight = 39 → difference = 37 →
  Bridget_weight = Martha_weight + difference →
  Martha_weight = 2 :=
by
  intros Bridget_weight difference Martha_weight hBridget hDifference hRelation
  sorry

end martha_weight_l58_58635


namespace k_less_than_zero_l58_58125

variable (k : ℝ)

def function_decreases (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ > f x₂

theorem k_less_than_zero (h : function_decreases (λ x => k * x - 5)) : k < 0 :=
sorry

end k_less_than_zero_l58_58125


namespace avg_speed_is_20_l58_58886

-- Define the total distance and total time
def total_distance : ℕ := 100
def total_time : ℕ := 5

-- Define the average speed calculation
def average_speed (distance : ℕ) (time : ℕ) : ℕ := distance / time

-- The theorem to prove the average speed given the distance and time
theorem avg_speed_is_20 : average_speed total_distance total_time = 20 :=
by
  sorry

end avg_speed_is_20_l58_58886


namespace maximum_rubles_received_max_payment_possible_l58_58701

def is_four_digit (n : ℕ) : Prop :=
  2000 ≤ n ∧ n < 2100

def divisible_by (n d : ℕ) : Prop :=
  d ∣ n

def payment (n : ℕ) : ℕ :=
  if divisible_by n 1 then 1 else 0 +
  (if divisible_by n 3 then 3 else 0) +
  (if divisible_by n 5 then 5 else 0) +
  (if divisible_by n 7 then 7 else 0) +
  (if divisible_by n 9 then 9 else 0) +
  (if divisible_by n 11 then 11 else 0)

theorem maximum_rubles_received :
  ∀ (n : ℕ), is_four_digit n → payment n ≤ 31 :=
sorry

theorem max_payment_possible :
  ∃ (n : ℕ), is_four_digit n ∧ payment n = 31 :=
sorry

end maximum_rubles_received_max_payment_possible_l58_58701


namespace necessary_but_not_sufficient_l58_58747

theorem necessary_but_not_sufficient (x : ℝ) (h : x < 4) : x < 0 ∨ true :=
by
  sorry

end necessary_but_not_sufficient_l58_58747


namespace real_number_a_value_l58_58410

open Set

variable {a : ℝ}

theorem real_number_a_value (A B : Set ℝ) (hA : A = {-1, 1, 3}) (hB : B = {a + 2, a^2 + 4}) (hAB : A ∩ B = {3}) : a = 1 := 
by 
-- Step proof will be here
sorry

end real_number_a_value_l58_58410


namespace ratio_of_men_to_women_l58_58591

def num_cannoneers : ℕ := 63
def num_people : ℕ := 378
def num_women (C : ℕ) : ℕ := 2 * C
def num_men (total : ℕ) (women : ℕ) : ℕ := total - women

theorem ratio_of_men_to_women : 
  let C := num_cannoneers
  let total := num_people
  let W := num_women C
  let M := num_men total W
  M / W = 2 :=
by
  sorry

end ratio_of_men_to_women_l58_58591


namespace distinct_real_roots_l58_58163

noncomputable def g (x d : ℝ) : ℝ := x^2 + 4*x + d

theorem distinct_real_roots (d : ℝ) :
  (∃! x : ℝ, g (g x d) d = 0) ↔ d = 0 :=
sorry

end distinct_real_roots_l58_58163


namespace colored_pictures_count_l58_58910

def initial_pictures_count : ℕ := 44 + 44
def pictures_left : ℕ := 68

theorem colored_pictures_count : initial_pictures_count - pictures_left = 20 := by
  -- Definitions and proof will go here
  sorry

end colored_pictures_count_l58_58910


namespace find_k_l58_58809

-- Define the vector operations and properties

def vector_add (a b : ℝ × ℝ) : ℝ × ℝ := (a.1 + b.1, a.2 + b.2)
def vector_smul (k : ℝ) (a : ℝ × ℝ) : ℝ × ℝ := (k * a.1, k * a.2)
def vectors_parallel (a b : ℝ × ℝ) : Prop := 
  (a.1 * b.2 = a.2 * b.1)

-- Given vectors
def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (-3, 2)

-- Statement of the problem
theorem find_k (k : ℝ) : 
  vectors_parallel (vector_add (vector_smul k a) b) (vector_add a (vector_smul (-3) b)) 
  → k = -1 / 3 :=
by
  sorry

end find_k_l58_58809


namespace maximum_rubles_received_l58_58707

def four_digit_number_of_form_20xx (n : ℕ) : Prop :=
  2000 ≤ n ∧ n < 2100

def divisible_by (d : ℕ) (n : ℕ) : Prop :=
  n % d = 0

theorem maximum_rubles_received :
  ∃ (n : ℕ), four_digit_number_of_form_20xx n ∧
             divisible_by 1 n ∧
             divisible_by 3 n ∧
             divisible_by 7 n ∧
             divisible_by 9 n ∧
             divisible_by 11 n ∧
             ¬ divisible_by 5 n ∧
             1 + 3 + 7 + 9 + 11 = 31 :=
sorry

end maximum_rubles_received_l58_58707


namespace days_to_complete_work_l58_58474

-- Conditions
def work_rate_A : ℚ := 1 / 8
def work_rate_B : ℚ := 1 / 16
def combined_work_rate := work_rate_A + work_rate_B

-- Statement to prove
theorem days_to_complete_work : 1 / combined_work_rate = 16 / 3 := by
  sorry

end days_to_complete_work_l58_58474


namespace problem_solution_l58_58632

theorem problem_solution
  (y1 y2 y3 y4 y5 y6 y7 : ℝ)
  (h1 : y1 + 3*y2 + 5*y3 + 7*y4 + 9*y5 + 11*y6 + 13*y7 = 0)
  (h2 : 3*y1 + 5*y2 + 7*y3 + 9*y4 + 11*y5 + 13*y6 + 15*y7 = 10)
  (h3 : 5*y1 + 7*y2 + 9*y3 + 11*y4 + 13*y5 + 15*y6 + 17*y7 = 104) :
  7*y1 + 9*y2 + 11*y3 + 13*y4 + 15*y5 + 17*y6 + 19*y7 = 282 := by
  sorry

end problem_solution_l58_58632


namespace prob_first_diamond_second_ace_or_face_l58_58259

theorem prob_first_diamond_second_ace_or_face :
  let deck_size := 52
  let first_card_diamonds := 13 / deck_size
  let prob_ace_after_diamond := 4 / (deck_size - 1)
  let prob_face_after_diamond := 12 / (deck_size - 1)
  first_card_diamonds * (prob_ace_after_diamond + prob_face_after_diamond) = 68 / 867 :=
by
  let deck_size := 52
  let first_card_diamonds := 13 / deck_size
  let prob_ace_after_diamond := 4 / (deck_size - 1)
  let prob_face_after_diamond := 12 / (deck_size - 1)
  sorry

end prob_first_diamond_second_ace_or_face_l58_58259


namespace sequence_non_zero_l58_58094

theorem sequence_non_zero :
  ∀ n : ℕ, ∃ a : ℕ → ℤ,
  (a 1 = 1) ∧
  (a 2 = 2) ∧
  (∀ n : ℕ, (a (n+1) % 2 = 1 ∧ a n % 2 = 1) → (a (n+2) = 5 * a (n+1) - 3 * a n)) ∧
  (∀ n : ℕ, (a (n+1) % 2 = 0 ∧ a n % 2 = 0) → (a (n+2) = a (n+1) - a n)) ∧
  (a n ≠ 0) :=
by
  sorry

end sequence_non_zero_l58_58094


namespace bill_picked_apples_l58_58344

-- Definitions from conditions
def children := 2
def apples_per_child_per_teacher := 3
def favorite_teachers := 2
def apples_per_pie := 10
def pies_baked := 2
def apples_left := 24

-- Number of apples given to teachers
def apples_for_teachers := children * apples_per_child_per_teacher * favorite_teachers

-- Number of apples used for pies
def apples_for_pies := pies_baked * apples_per_pie

-- The final theorem to be stated
theorem bill_picked_apples :
  apples_for_teachers + apples_for_pies + apples_left = 56 := 
sorry

end bill_picked_apples_l58_58344


namespace below_sea_level_representation_l58_58822

def depth_below_sea_level (depth: Int → Prop) :=
  ∀ x, depth x → x < 0

theorem below_sea_level_representation (D: Int) (A: Int) 
  (hA: A = 9050) (hD: D = 10907) 
  (above_sea_level: Int → Prop) (below_sea_level: Int → Prop)
  (h1: ∀ y, above_sea_level y → y > 0):
  below_sea_level (-D) :=
by
  sorry

end below_sea_level_representation_l58_58822


namespace sum_gcd_lcm_63_2898_l58_58745

theorem sum_gcd_lcm_63_2898 : Nat.gcd 63 2898 + Nat.lcm 63 2898 = 182575 :=
by
  sorry

end sum_gcd_lcm_63_2898_l58_58745


namespace zero_points_of_function_l58_58378

/-- Assume g(x) is x^3 * ln(x) and we want to determine the range of values of m
such that f(x) = x^3 * ln x + m has 2 zero points. -/
theorem zero_points_of_function (m : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1 > 0 ∧ x2 > 0 ∧ x1^3 * Real.log x1 + m = 0 ∧ x2^3 * Real.log x2 + m = 0) ↔
  m < (1 / (3 * Real.exp 1)) :=
sorry

end zero_points_of_function_l58_58378


namespace trigonometric_inequality_l58_58022

theorem trigonometric_inequality (a b x : ℝ) :
  (Real.sin x + a * Real.cos x) * (Real.sin x + b * Real.cos x) ≤ 1 + ( (a + b) / 2 )^2 :=
by
  sorry

end trigonometric_inequality_l58_58022


namespace value_of_y_l58_58214

theorem value_of_y : 
  ∀ y : ℚ, y = (2010^2 - 2010 + 1 : ℚ) / 2010 → y = (2009 + 1 / 2010 : ℚ) := by
  sorry

end value_of_y_l58_58214


namespace depth_below_sea_notation_l58_58834

variables (alt_above_sea : ℝ) (depth_below_sea : ℝ)

def notation_above_sea : ℝ := alt_above_sea

def notation_below_sea (d : ℝ) : ℝ := -d

theorem depth_below_sea_notation : alt_above_sea = 9050 → notation_above_sea = 9050 → depth_below_sea = 10907 → notation_below_sea depth_below_sea = -10907 :=
by
  intros h1 h2 h3
  rw [h3, notation_below_sea]
  exact eq.symm sorry

end depth_below_sea_notation_l58_58834


namespace jake_peaches_count_l58_58007

-- Define Jill's peaches
def jill_peaches : ℕ := 5

-- Define Steven's peaches based on the condition that Steven has 18 more peaches than Jill
def steven_peaches : ℕ := jill_peaches + 18

-- Define Jake's peaches based on the condition that Jake has 6 fewer peaches than Steven
def jake_peaches : ℕ := steven_peaches - 6

-- The theorem to prove that Jake has 17 peaches
theorem jake_peaches_count : jake_peaches = 17 := by
  sorry

end jake_peaches_count_l58_58007


namespace trigonometric_proof_l58_58964

theorem trigonometric_proof (α : ℝ) (h : 3 * Real.sin α + Real.cos α = 0) : 
  1 / (Real.cos α ^ 2 + 2 * Real.sin α * Real.cos α) = 10 / 3 :=
by sorry

end trigonometric_proof_l58_58964


namespace greatest_value_of_x_l58_58029

theorem greatest_value_of_x
  (x : ℕ)
  (h1 : x % 4 = 0) -- x is a multiple of 4
  (h2 : x > 0) -- x is positive
  (h3 : x^3 < 2000) -- x^3 < 2000
  : x ≤ 12 :=
by
  sorry

end greatest_value_of_x_l58_58029


namespace polynomial_possible_integer_roots_l58_58927

theorem polynomial_possible_integer_roots (b1 b2 : ℤ) :
  ∀ x : ℤ, (x ∣ 18) ↔ (x^3 + b2 * x^2 + b1 * x + 18 = 0) → 
  x = -18 ∨ x = -9 ∨ x = -6 ∨ x = -3 ∨ x = -2 ∨ x = -1 ∨ x = 1 ∨ x = 2 ∨ x = 3 ∨ x = 6 ∨ x = 9 ∨ x = 18 :=
by {
  sorry
}


end polynomial_possible_integer_roots_l58_58927


namespace average_of_possible_values_of_x_l58_58538

theorem average_of_possible_values_of_x (x : ℝ) (h : (2 * x^2 + 3) = 21) : (x = 3 ∨ x = -3) → (3 + -3) / 2 = 0 := by
  sorry

end average_of_possible_values_of_x_l58_58538


namespace vanya_number_l58_58592

theorem vanya_number (m n : ℕ) (hm : m < 10) (hn : n < 10) (h : (10 * n + m)^2 = 4 * (10 * m + n)) : 
  10 * m + n = 81 :=
by sorry

end vanya_number_l58_58592


namespace decreasing_function_positive_l58_58265

variable {f : ℝ → ℝ}

axiom decreasing (h : ℝ → ℝ) : ∀ x1 x2, x1 < x2 → h x1 > h x2

theorem decreasing_function_positive (h_decreasing: ∀ x1 x2: ℝ, x1 < x2 → f x1 > f x2)
    (h_condition: ∀ x: ℝ, f x / (deriv^[2] f x) + x < 1) :
  ∀ x : ℝ, f x > 0 := 
by
  sorry

end decreasing_function_positive_l58_58265


namespace regular_polygon_perimeter_l58_58484

theorem regular_polygon_perimeter (s : ℕ) (E : ℕ) (n : ℕ) (P : ℕ)
  (h1 : s = 6)
  (h2 : E = 90)
  (h3 : E = 360 / n)
  (h4 : P = n * s) :
  P = 24 :=
by sorry

end regular_polygon_perimeter_l58_58484


namespace num_positive_integers_l58_58087

theorem num_positive_integers (n : ℕ) :
    (0 < n ∧ n < 40 ∧ ∃ k : ℕ, k > 0 ∧ n = 40 * k / (k + 1)) ↔ 
    (n = 20 ∨ n = 30 ∨ n = 32 ∨ n = 35 ∨ n = 36 ∨ n = 38 ∨ n = 39) :=
sorry

end num_positive_integers_l58_58087


namespace solve_for_x_l58_58422

theorem solve_for_x (x : ℚ) : (x + 4) / (x - 3) = (x - 2) / (x + 2) -> x = -2 / 11 := by
  sorry

end solve_for_x_l58_58422


namespace parabola_directrix_l58_58790

theorem parabola_directrix (x y : ℝ) (h : y = 4 * (x - 1)^2 + 3) : y = 11 / 4 :=
sorry

end parabola_directrix_l58_58790


namespace friend_time_to_read_book_l58_58385

-- Define the conditions and variables
def my_reading_time : ℕ := 240 -- 4 hours in minutes
def speed_ratio : ℕ := 2 -- I read at half the speed of my friend

-- Define the variable for my friend's reading time which we need to find
def friend_reading_time : ℕ := my_reading_time / speed_ratio

-- The theorem statement that given the conditions, the friend's reading time is 120 minutes
theorem friend_time_to_read_book : friend_reading_time = 120 := sorry

end friend_time_to_read_book_l58_58385


namespace range_of_m_range_of_x_l58_58376

-- Define the function f(x) = m*x^2 - m*x - 6 + m
def f (m x : ℝ) : ℝ := m*x^2 - m*x - 6 + m

-- Proof for the first statement
theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 3 → f m x < 0) ↔ m < 6 / 7 := 
sorry

-- Proof for the second statement
theorem range_of_x (x : ℝ) :
  (∀ m : ℝ, -2 ≤ m ∧ m ≤ 2 → f m x < 0) ↔ -1 < x ∧ x < 2 :=
sorry

end range_of_m_range_of_x_l58_58376


namespace space_shuttle_speed_conversion_l58_58466

-- Define the given conditions
def speed_km_per_sec : ℕ := 6  -- Speed in km/s
def seconds_per_hour : ℕ := 3600  -- Seconds in an hour

-- Define the computed speed in km/hr
def expected_speed_km_per_hr : ℕ := 21600  -- Expected speed in km/hr

-- The main theorem statement to be proven
theorem space_shuttle_speed_conversion : speed_km_per_sec * seconds_per_hour = expected_speed_km_per_hr := by
  sorry

end space_shuttle_speed_conversion_l58_58466


namespace triangle_is_isosceles_right_l58_58280

theorem triangle_is_isosceles_right (a b c : ℝ) (A B C : ℝ) (h1 : b = a * Real.sin C) (h2 : c = a * Real.cos B) : 
  A = π / 2 ∧ b = c := 
sorry

end triangle_is_isosceles_right_l58_58280


namespace fifth_boat_more_than_average_l58_58325

theorem fifth_boat_more_than_average :
  let total_people := 2 + 4 + 3 + 5 + 6
  let num_boats := 5
  let average_people := total_people / num_boats
  let fifth_boat := 6
  (fifth_boat - average_people) = 2 :=
by
  sorry

end fifth_boat_more_than_average_l58_58325


namespace tangent_line_equation_l58_58080

theorem tangent_line_equation (e x y : ℝ) (h_curve : y = x^3 / e) (h_point : x = e ∧ y = e^2) :
  3 * e * x - y - 2 * e^2 = 0 :=
sorry

end tangent_line_equation_l58_58080


namespace proj_v_onto_w_l58_58516

open Real

noncomputable def v : ℝ × ℝ := (8, -4)
noncomputable def w : ℝ × ℝ := (2, 3)

noncomputable def dot_product (a b : ℝ × ℝ) : ℝ :=
  a.1 * b.1 + a.2 * b.2

noncomputable def projection (v w : ℝ × ℝ) : ℝ × ℝ :=
  let coeff := dot_product v w / dot_product w w
  (coeff * w.1, coeff * w.2)

theorem proj_v_onto_w :
  projection v w = (8 / 13, 12 / 13) :=
by
  sorry

end proj_v_onto_w_l58_58516


namespace Joel_laps_count_l58_58610

-- Definitions of conditions
def Yvonne_laps := 10
def sister_laps := Yvonne_laps / 2
def Joel_laps := sister_laps * 3

-- Statement to be proved
theorem Joel_laps_count : Joel_laps = 15 := by
  -- currently, proof is not required, so we defer it with 'sorry'
  sorry

end Joel_laps_count_l58_58610


namespace remainder_of_sum_is_five_l58_58746

theorem remainder_of_sum_is_five (a b c d : ℕ) (ha : a % 15 = 11) (hb : b % 15 = 12) (hc : c % 15 = 13) (hd : d % 15 = 14) :
  (a + b + c + d) % 15 = 5 :=
by
  sorry

end remainder_of_sum_is_five_l58_58746


namespace books_total_l58_58968

def stuBooks : ℕ := 9
def albertBooks : ℕ := 4 * stuBooks
def totalBooks : ℕ := stuBooks + albertBooks

theorem books_total : totalBooks = 45 := by
  sorry

end books_total_l58_58968


namespace calculation_eq_990_l58_58326

theorem calculation_eq_990 : (0.0077 * 3.6) / (0.04 * 0.1 * 0.007) = 990 :=
by
  sorry

end calculation_eq_990_l58_58326


namespace intersection_point_l58_58998

theorem intersection_point : ∃ (x y : ℝ), y = 3 - x ∧ y = 3 * x - 5 ∧ x = 2 ∧ y = 1 :=
by
  sorry

end intersection_point_l58_58998


namespace sum_of_two_numbers_l58_58178

theorem sum_of_two_numbers (a b : ℕ) (h1 : a - b = 10) (h2 : a = 22) : a + b = 34 :=
sorry

end sum_of_two_numbers_l58_58178


namespace batsman_average_20th_l58_58920

noncomputable def average_after_20th (A : ℕ) : ℕ :=
  let total_runs_19 := 19 * A
  let total_runs_20 := total_runs_19 + 85
  let new_average := (total_runs_20) / 20
  new_average
  
theorem batsman_average_20th (A : ℕ) (h1 : 19 * A + 85 = 20 * (A + 4)) : average_after_20th A = 9 := by
  sorry

end batsman_average_20th_l58_58920


namespace find_dividend_l58_58281

theorem find_dividend (dividend divisor quotient : ℕ) 
  (h_sum : dividend + divisor + quotient = 103)
  (h_quotient : quotient = 3)
  (h_divisor : divisor = dividend / quotient) : 
  dividend = 75 :=
by
  rw [h_quotient, h_divisor] at h_sum
  sorry

end find_dividend_l58_58281


namespace combined_flock_size_l58_58645

def original_ducks := 100
def killed_per_year := 20
def born_per_year := 30
def years_passed := 5
def another_flock := 150

theorem combined_flock_size :
  original_ducks + years_passed * (born_per_year - killed_per_year) + another_flock = 300 :=
by
  sorry

end combined_flock_size_l58_58645


namespace smallest_palindromic_integer_is_21_l58_58356

noncomputable def is_palindrome_base (n : ℕ) (b : ℕ) : Prop :=
  let digits := (Nat.digits b n).reverse
  digits = Nat.digits b n

def smallest_palindromic_integer : ℕ :=
  (List.range 1000).find (λ n, n > 20 ∧ is_palindrome_base n 2 ∧ is_palindrome_base n 4).get_or_else 0

theorem smallest_palindromic_integer_is_21 :
  smallest_palindromic_integer = 21 :=
by
  sorry

end smallest_palindromic_integer_is_21_l58_58356


namespace circumference_proportionality_l58_58429

theorem circumference_proportionality (r : ℝ) (C : ℝ) (k : ℝ) (π : ℝ)
  (h1 : C = k * r)
  (h2 : C = 2 * π * r) :
  k = 2 * π :=
sorry

end circumference_proportionality_l58_58429


namespace solve_inequality_system_l58_58303

theorem solve_inequality_system
  (x : ℝ)
  (h1 : 3 * (x - 1) < 5 * x + 11)
  (h2 : 2 * x > (9 - x) / 4) :
  x > 1 :=
sorry

end solve_inequality_system_l58_58303


namespace no_equalities_l58_58875

def f1 (x : ℤ) : ℤ := x * (x - 2007)
def f2 (x : ℤ) : ℤ := (x - 1) * (x - 2006)
def f1004 (x : ℤ) : ℤ := (x - 1003) * (x - 1004)

theorem no_equalities (x : ℤ) (h : 0 ≤ x ∧ x ≤ 2007) :
  ¬(f1 x = f2 x ∨ f1 x = f1004 x ∨ f2 x = f1004 x) :=
by
  sorry

end no_equalities_l58_58875


namespace no_net_coin_change_l58_58339

noncomputable def probability_no_coin_change_each_round : ℚ :=
  (1 / 3) ^ 5

theorem no_net_coin_change :
  probability_no_coin_change_each_round = 1 / 243 := by
  sorry

end no_net_coin_change_l58_58339


namespace variance_of_dataset_l58_58392

theorem variance_of_dataset (a : ℝ) 
  (h1 : (4 + a + 5 + 3 + 8) / 5 = a) :
  (1 / 5) * ((4 - a) ^ 2 + (a - a) ^ 2 + (5 - a) ^ 2 + (3 - a) ^ 2 + (8 - a) ^ 2) = 14 / 5 :=
by
  sorry

end variance_of_dataset_l58_58392


namespace ratio_e_to_f_l58_58533

theorem ratio_e_to_f {a b c d e f : ℝ}
  (h1 : a / b = 1 / 3)
  (h2 : b / c = 2)
  (h3 : c / d = 1 / 2)
  (h4 : d / e = 3)
  (h5 : a * b * c / (d * e * f) = 0.75) :
  e / f = 0.5 :=
sorry

end ratio_e_to_f_l58_58533


namespace janet_spending_difference_l58_58289

-- Definitions for the conditions
def clarinet_hourly_rate : ℝ := 40
def clarinet_hours_per_week : ℝ := 3
def piano_hourly_rate : ℝ := 28
def piano_hours_per_week : ℝ := 5
def weeks_per_year : ℕ := 52

-- The theorem to be proven
theorem janet_spending_difference :
  (piano_hourly_rate * piano_hours_per_week * weeks_per_year - clarinet_hourly_rate * clarinet_hours_per_week * weeks_per_year) = 1040 :=
by
  sorry

end janet_spending_difference_l58_58289


namespace percent_defective_units_shipped_l58_58283

theorem percent_defective_units_shipped (h1 : 8 / 100 * 4 / 100 = 32 / 10000) :
  (32 / 10000) * 100 = 0.32 := 
sorry

end percent_defective_units_shipped_l58_58283


namespace double_root_values_l58_58257

theorem double_root_values (c : ℝ) :
  (∃ a : ℝ, (a^5 - 5 * a + c = 0) ∧ (5 * a^4 - 5 = 0)) ↔ (c = 4 ∨ c = -4) :=
by
  sorry

end double_root_values_l58_58257


namespace length_of_room_calculation_l58_58788

variable (broadness_of_room : ℝ) (width_of_carpet : ℝ) (total_cost : ℝ) (rate_per_sq_meter : ℝ) (area_of_carpet : ℝ) (length_of_room : ℝ)

theorem length_of_room_calculation (h1 : broadness_of_room = 9) 
    (h2 : width_of_carpet = 0.75) 
    (h3 : total_cost = 1872) 
    (h4 : rate_per_sq_meter = 12) 
    (h5 : area_of_carpet = total_cost / rate_per_sq_meter)
    (h6 : area_of_carpet = length_of_room * width_of_carpet) 
    : length_of_room = 208 := 
by 
    sorry

end length_of_room_calculation_l58_58788


namespace real_set_x_eq_l58_58785

theorem real_set_x_eq :
  {x : ℝ | ⌊x * ⌊x⌋⌋ = 45} = {x : ℝ | 7.5 ≤ x ∧ x < 7.6667} :=
by
  -- The proof would be provided here, but we're skipping it with sorry
  sorry

end real_set_x_eq_l58_58785


namespace evaluate_sum_l58_58369

theorem evaluate_sum (a b c : ℝ) 
  (h : (a / (36 - a) + b / (49 - b) + c / (81 - c) = 9)) :
  (6 / (36 - a) + 7 / (49 - b) + 9 / (81 - c) = 5.047) :=
by
  sorry

end evaluate_sum_l58_58369


namespace janet_spending_difference_l58_58288

-- Definitions for the conditions
def clarinet_hourly_rate : ℝ := 40
def clarinet_hours_per_week : ℝ := 3
def piano_hourly_rate : ℝ := 28
def piano_hours_per_week : ℝ := 5
def weeks_per_year : ℕ := 52

-- The theorem to be proven
theorem janet_spending_difference :
  (piano_hourly_rate * piano_hours_per_week * weeks_per_year - clarinet_hourly_rate * clarinet_hours_per_week * weeks_per_year) = 1040 :=
by
  sorry

end janet_spending_difference_l58_58288


namespace cone_radius_l58_58513

theorem cone_radius (CSA : ℝ) (l : ℝ) (r : ℝ) (h_CSA : CSA = 989.6016858807849) (h_l : l = 15) :
    r = 21 :=
by
  sorry

end cone_radius_l58_58513


namespace total_hours_worked_l58_58244

theorem total_hours_worked :
  (∃ (hours_per_day : ℕ) (days : ℕ), hours_per_day = 3 ∧ days = 6) →
  (∃ (total_hours : ℕ), total_hours = 18) :=
by
  intros
  sorry

end total_hours_worked_l58_58244


namespace find_principal_l58_58120

theorem find_principal (x y : ℝ) : 
  (2 * x * y / 100 = 400) → 
  (2 * x * y + x * y^2 / 100 = 41000) → 
  x = 4000 := 
by
  sorry

end find_principal_l58_58120


namespace max_money_received_back_l58_58331

def total_money_before := 3000
def value_chip_20 := 20
def value_chip_100 := 100
def chips_lost_total := 16
def chips_lost_diff_1 (x y : ℕ) := x = y + 2
def chips_lost_diff_2 (x y : ℕ) := x = y - 2

theorem max_money_received_back :
  ∃ (x y : ℕ), 
  (chips_lost_diff_1 x y ∨ chips_lost_diff_2 x y) ∧ 
  (x + y = chips_lost_total) ∧
  total_money_before - (x * value_chip_20 + y * value_chip_100) = 2120 :=
sorry

end max_money_received_back_l58_58331


namespace joseph_total_cost_l58_58851

variable (cost_refrigerator cost_water_heater cost_oven : ℝ)

-- Conditions
axiom h1 : cost_refrigerator = 3 * cost_water_heater
axiom h2 : cost_oven = 500
axiom h3 : cost_oven = 2 * cost_water_heater

-- Theorem
theorem joseph_total_cost : cost_refrigerator + cost_water_heater + cost_oven = 1500 := by
  sorry

end joseph_total_cost_l58_58851


namespace small_monkey_dolls_cheaper_than_large_l58_58940

theorem small_monkey_dolls_cheaper_than_large (S : ℕ) 
  (h1 : 300 / 6 = 50) 
  (h2 : 300 / S = 75) 
  (h3 : 75 - 50 = 25) : 
  6 - S = 2 := 
sorry

end small_monkey_dolls_cheaper_than_large_l58_58940


namespace original_price_of_article_l58_58221

theorem original_price_of_article (new_price : ℝ) (reduction_percentage : ℝ) (original_price : ℝ) 
  (h_reduction : reduction_percentage = 56/100) (h_new_price : new_price = 4400) :
  original_price = 10000 :=
sorry

end original_price_of_article_l58_58221


namespace max_rubles_l58_58708

theorem max_rubles (n : ℕ) (h1 : 2000 ≤ n) (h2 : n ≤ 2099) :
  (∃ k, n = 99 * k) → 
  31 ≤
    (if n % 1 = 0 then 1 else 0) +
    (if n % 3 = 0 then 3 else 0) +
    (if n % 5 = 0 then 5 else 0) +
    (if n % 7 = 0 then 7 else 0) +
    (if n % 9 = 0 then 9 else 0) +
    (if n % 11 = 0 then 11 else 0) :=
sorry

end max_rubles_l58_58708


namespace problem_f_prime_at_zero_l58_58519

noncomputable def f (x : ℝ) : ℝ := x * (x + 1) * (x + 2) * (x + 3) * (x + 4) * (x + 5) + 6

theorem problem_f_prime_at_zero : deriv f 0 = 120 :=
by
  -- Proof omitted
  sorry

end problem_f_prime_at_zero_l58_58519


namespace balloons_total_l58_58878

theorem balloons_total (a b : ℕ) (h1 : a = 47) (h2 : b = 13) : a + b = 60 := 
by
  -- Since h1 and h2 provide values for a and b respectively,
  -- the result can be proved using these values.
  sorry

end balloons_total_l58_58878


namespace max_value_amc_am_mc_ca_l58_58132

theorem max_value_amc_am_mc_ca (A M C : ℕ) 
  (h : A + M + C = 15) : 
  A * M * C + A * M + M * C + C * A ≤ 200 := 
sorry

end max_value_amc_am_mc_ca_l58_58132


namespace min_value_expression_l58_58957

theorem min_value_expression (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (habc : a * b + a * c = 4) :
  ∃ m, m = 4 ∧ m ≤ 2 / a + 2 / (b + c) + 8 / (a + b + c) :=
by
  sorry

end min_value_expression_l58_58957


namespace max_value_of_q_l58_58142

theorem max_value_of_q (A M C : ℕ) (h_sum : A + M + C = 15) : 
  A * M * C + A * M + M * C + C * A ≤ 200 :=
sorry

end max_value_of_q_l58_58142


namespace denote_depth_below_sea_level_l58_58837

theorem denote_depth_below_sea_level (above_sea_level : Int) (depth_haidou_1 : Int) :
  (above_sea_level > 0) ∧ (depth_haidou_1 < 0) → depth_haidou_1 = -10907 :=
by
  intros h
  sorry

end denote_depth_below_sea_level_l58_58837


namespace min_boxes_needed_to_form_cube_l58_58674

-- Definitions based on problem conditions
def width : ℕ := 18
def length : ℕ := 12
def height : ℕ := 9

-- Least common multiple of the given dimensions
def lcm_dimensions : ℕ := Nat.lcm (Nat.lcm width length) height

-- Volume of the cube whose side is the LCM of the dimensions
def volume_cube : ℕ := lcm_dimensions ^ 3

-- Volume of one cuboid-shaped box
def volume_box : ℕ := width * length * height

-- Number of boxes needed to fill the cube
def number_boxes : ℕ := volume_cube / volume_box

-- Theorem: Proving that the number of boxes required is 24
theorem min_boxes_needed_to_form_cube : number_boxes = 24 := by
  -- Placeholder for the actual proof
  sorry

end min_boxes_needed_to_form_cube_l58_58674


namespace box_combination_is_correct_l58_58781

variables (C A S T t u : ℕ)

theorem box_combination_is_correct
    (h1 : 3 * S % t = C)
    (h2 : 2 * A + C = T)
    (h3 : 2 * C + A + u = T) :
  (1000 * C + 100 * A + 10 * S + T = 7252) :=
sorry

end box_combination_is_correct_l58_58781


namespace triangle_perimeter_l58_58193

-- Definitions for the conditions
def side_length1 : ℕ := 3
def side_length2 : ℕ := 6
def equation (x : ℤ) := x^2 - 6 * x + 8 = 0

-- Perimeter calculation given the sides form a triangle
theorem triangle_perimeter (x : ℤ) (h₁ : equation x) (h₂ : 3 + 6 > x) (h₃ : 3 + x > 6) (h₄ : 6 + x > 3) :
  3 + 6 + x = 13 :=
by sorry

end triangle_perimeter_l58_58193


namespace Vanya_number_thought_of_l58_58595

theorem Vanya_number_thought_of :
  ∃ m n : ℕ, m < 10 ∧ n < 10 ∧ (10 * m + n = 81 ∧ (10 * n + m)^2 = 4 * (10 * m + n)) :=
sorry

end Vanya_number_thought_of_l58_58595


namespace turnip_total_correct_l58_58562

def turnips_left (melanie benny sarah david m_sold d_sold : ℕ) : ℕ :=
  let melanie_left := melanie - m_sold
  let david_left := david - d_sold
  benny + sarah + melanie_left + david_left

theorem turnip_total_correct :
  turnips_left 139 113 195 87 32 15 = 487 :=
by
  sorry

end turnip_total_correct_l58_58562


namespace tangents_collinear_F_minimum_area_triangle_l58_58522

noncomputable def ellipse_condition : Prop :=
  ∀ (x y : ℝ), x^2 / 4 + y^2 / 3 = 1

noncomputable def point_P_on_line (P : ℝ × ℝ) : Prop :=
  P.1 = 4

noncomputable def tangent_condition (P A B : ℝ × ℝ) (ellipse : ℝ × ℝ → Prop) : Prop :=
  -- Tangent lines meet the ellipse equation at points A and B
  ellipse A ∧ ellipse B

noncomputable def collinear (A F B : ℝ × ℝ) : Prop :=
  (A.2 - F.2) * (B.1 - F.1) = (B.2 - F.2) * (A.1 - F.1)

noncomputable def minimum_area (P A B : ℝ × ℝ) : ℝ :=
  1 / 2 * abs ((A.1 * B.2 + B.1 * P.2 + P.1 * A.2) - (A.2 * B.1 + B.2 * P.1 + P.2 * A.1))

theorem tangents_collinear_F (F : ℝ × ℝ) (P A B : ℝ × ℝ) (ellipse : ℝ × ℝ → Prop)
  (h_ellipse_F : F = (1, 0))
  (h_point_P_on_line : point_P_on_line P)
  (h_tangent_condition : tangent_condition P A B ellipse)
  (h_ellipse_def : ellipse_condition) :
  collinear A F B :=
sorry

theorem minimum_area_triangle (F P A B : ℝ × ℝ) (ellipse : ℝ × ℝ → Prop)
  (h_ellipse_F : F = (1, 0))
  (h_point_P_on_line : point_P_on_line P)
  (h_tangent_condition : tangent_condition P A B ellipse)
  (h_ellipse_def : ellipse_condition) :
  minimum_area P A B = 9 / 2 :=
sorry

end tangents_collinear_F_minimum_area_triangle_l58_58522


namespace candies_per_block_l58_58025

theorem candies_per_block (candies_per_house : ℕ) (houses_per_block : ℕ) (h1 : candies_per_house = 7) (h2 : houses_per_block = 5) :
  candies_per_house * houses_per_block = 35 :=
by 
  -- Placeholder for the formal proof
  sorry

end candies_per_block_l58_58025


namespace blue_eyes_blonde_hair_logic_l58_58006

theorem blue_eyes_blonde_hair_logic :
  ∀ (a b c d : ℝ), 
  (a / (a + b) > (a + c) / (a + b + c + d)) →
  (a / (a + c) > (a + b) / (a + b + c + d)) :=
by
  intro a b c d h
  sorry

end blue_eyes_blonde_hair_logic_l58_58006


namespace correct_option_C_l58_58104

noncomputable def f (x : ℝ) : ℝ := (2^x - 1) / (2^x + 1)

theorem correct_option_C : ∀ (x1 x2 : ℝ), 0 < x1 → x1 < x2 → x1 * f x1 < x2 * f x2 :=
by
  intro x1 x2 hx1 hx12
  sorry

end correct_option_C_l58_58104


namespace rationalize_expression_l58_58991

theorem rationalize_expression :
  ( ∀ (x y z a b c : ℝ), 
      x = sqrt 3 ∧ y = sqrt 7 ∧ z = sqrt 5 ∧ a = sqrt 11 ∧ b = sqrt 6 ∧ c = sqrt 8 
      → (x / y * z / a * b / c) = 3 * sqrt 385 / 154) :=
begin
  rintros _ _ _ _ _ _ ⟨hx, hy, hz, ha, hb, hc⟩,
  sorry,
end

end rationalize_expression_l58_58991


namespace quadratic_root_relation_l58_58583

theorem quadratic_root_relation (m n p q : ℝ) (s₁ s₂ : ℝ) 
  (h1 : s₁ + s₂ = -p) 
  (h2 : s₁ * s₂ = q) 
  (h3 : 3 * s₁ + 3 * s₂ = -m) 
  (h4 : 9 * s₁ * s₂ = n) 
  (h_m : m ≠ 0) 
  (h_n : n ≠ 0) 
  (h_p : p ≠ 0) 
  (h_q : q ≠ 0) :
  n = 9 * q :=
by
  sorry

end quadratic_root_relation_l58_58583


namespace max_net_income_is_50000_l58_58395

def tax_rate (y : ℝ) : ℝ :=
  10 * y ^ 2

def net_income (y : ℝ) : ℝ :=
  1000 * y - tax_rate y

theorem max_net_income_is_50000 :
  ∃ y : ℝ, (net_income y = 25000 ∧ 1000 * y = 50000) :=
by
  use 50
  sorry

end max_net_income_is_50000_l58_58395


namespace distinct_stone_arrangements_l58_58403

-- Define the set of 12 unique stones
def stones := Finset.range 12

-- Define the number of unique placements without considering symmetries
def placements : ℕ := stones.card.factorial

-- Define the number of symmetries (6 rotations and 6 reflections)
def symmetries : ℕ := 12

-- The total number of distinct configurations accounting for symmetries
def distinct_arrangements : ℕ := placements / symmetries

-- The main theorem stating the number of distinct arrangements
theorem distinct_stone_arrangements : distinct_arrangements = 39916800 := by 
  sorry

end distinct_stone_arrangements_l58_58403


namespace count_six_digit_palindromes_l58_58494

def num_six_digit_palindromes : ℕ := 9000

theorem count_six_digit_palindromes :
  (∃ a b c d : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ 0 ≤ d ∧ d ≤ 9 ∧
     num_six_digit_palindromes = 9000) :=
sorry

end count_six_digit_palindromes_l58_58494


namespace cooks_choice_l58_58062

theorem cooks_choice (n k : ℕ) (A B : Fin n) (hA : A.val < n) (hB : B.val < n) :
  n = 10 →
  k = 3 →
  (∀ (S : Finset (Fin n)), S.card = k → A ∈ S → B ∉ S) → 
  Finset.card (Finset.filter (λ S, A ∈ S ∧ B ∉ S) (Finset.powerset_len k (Finset.univ : Finset (Fin n)))) = 112 :=
by
  sorry

end cooks_choice_l58_58062


namespace unique_rhombus_property_not_in_rectangle_l58_58215

-- Definitions of properties for a rhombus and a rectangle
def is_rhombus (sides_equal : Prop) (opposite_sides_parallel : Prop) (opposite_angles_equal : Prop)
  (diagonals_perpendicular_and_bisect : Prop) : Prop :=
  sides_equal ∧ opposite_sides_parallel ∧ opposite_angles_equal ∧ diagonals_perpendicular_and_bisect

def is_rectangle (opposite_sides_equal_and_parallel : Prop) (all_angles_right : Prop)
  (diagonals_equal_and_bisect : Prop) : Prop :=
  opposite_sides_equal_and_parallel ∧ all_angles_right ∧ diagonals_equal_and_bisect

-- Proof objective: Prove that the unique property of a rhombus is the perpendicular and bisecting nature of its diagonals
theorem unique_rhombus_property_not_in_rectangle :
  ∀ (sides_equal opposite_sides_parallel opposite_angles_equal
      diagonals_perpendicular_and_bisect opposite_sides_equal_and_parallel
      all_angles_right diagonals_equal_and_bisect : Prop),
  is_rhombus sides_equal opposite_sides_parallel opposite_angles_equal diagonals_perpendicular_and_bisect →
  is_rectangle opposite_sides_equal_and_parallel all_angles_right diagonals_equal_and_bisect →
  diagonals_perpendicular_and_bisect ∧ ¬diagonals_equal_and_bisect :=
by
  sorry

end unique_rhombus_property_not_in_rectangle_l58_58215


namespace domain_of_f_monotonicity_of_f_l58_58092

noncomputable def f (a x : ℝ) := Real.log (a ^ x - 1) / Real.log a

theorem domain_of_f (a : ℝ) (h₀ : 0 < a) (h₁ : a ≠ 1) :
  (a > 1 → ∀ x : ℝ, f a x ∈ Set.Ioi 0) ∧ (0 < a ∧ a < 1 → ∀ x : ℝ, f a x ∈ Set.Iio 0) :=
sorry

theorem monotonicity_of_f (a : ℝ) (h₀ : 0 < a) (h₁ : a ≠ 1) :
  (a > 1 → StrictMono (f a)) ∧ (0 < a ∧ a < 1 → StrictMono (f a)) :=
sorry

end domain_of_f_monotonicity_of_f_l58_58092


namespace height_of_Brixton_l58_58286

theorem height_of_Brixton
  (I Z B Zr : ℕ)
  (h1 : I = Z + 4)
  (h2 : Z = B - 8)
  (h3 : Zr = B)
  (h4 : (I + Z + B + Zr) / 4 = 61) :
  B = 64 := by
  sorry

end height_of_Brixton_l58_58286


namespace sum_of_two_numbers_l58_58177

theorem sum_of_two_numbers (a b : ℕ) (h1 : a - b = 10) (h2 : a = 22) : a + b = 34 :=
sorry

end sum_of_two_numbers_l58_58177


namespace allocation_ways_l58_58419

/-- Defining the number of different balls and boxes -/
def num_balls : ℕ := 4
def num_boxes : ℕ := 3

/-- The theorem asserting the number of ways to place the balls into the boxes -/
theorem allocation_ways : (num_boxes ^ num_balls) = 81 := by
  sorry

end allocation_ways_l58_58419


namespace max_heaps_l58_58859

/-- Conditions and requirements for dividing a pile of stones into specified heaps -/
theorem max_heaps (stone_count heap_count : ℕ) (sizes : fin heap_count → ℕ) : 
  stone_count = 660 → heap_count = 30 → 
  (∀ i j, i < j → sizes i ≤ sizes j ∧ sizes j < 2 * sizes i) → 
  (∑ i, sizes i) = stone_count :=
by sorry

end max_heaps_l58_58859


namespace sequence_periodic_l58_58235

theorem sequence_periodic (a : ℕ → ℕ) (h : ∀ n > 2, a (n + 1) = (a n ^ n + a (n - 1)) % 10) :
  ∃ n₀, ∀ k, a (n₀ + k) = a (n₀ + k + 4) :=
by {
  sorry
}

end sequence_periodic_l58_58235


namespace red_card_events_l58_58517

-- Definitions based on the conditions
inductive Person
| A | B | C | D

inductive Card
| Red | Black | Blue | White

-- Definition of the events
def event_A_receives_red (distribution : Person → Card) : Prop :=
  distribution Person.A = Card.Red

def event_B_receives_red (distribution : Person → Card) : Prop :=
  distribution Person.B = Card.Red

-- The relationship between the two events
def mutually_exclusive_but_not_opposite (distribution : Person → Card) : Prop :=
  (event_A_receives_red distribution → ¬ event_B_receives_red distribution) ∧
  (event_B_receives_red distribution → ¬ event_A_receives_red distribution)

-- The formal theorem statement
theorem red_card_events (distribution : Person → Card) :
  mutually_exclusive_but_not_opposite distribution :=
sorry

end red_card_events_l58_58517


namespace solution_volume_l58_58472

theorem solution_volume (x : ℝ) (h1 : (0.16 * x) / (x + 13) = 0.0733333333333333) : x = 11 :=
by sorry

end solution_volume_l58_58472


namespace choose_one_from_ten_l58_58537

theorem choose_one_from_ten :
  Nat.choose 10 1 = 10 :=
by
  sorry

end choose_one_from_ten_l58_58537


namespace solve_for_x_l58_58225

theorem solve_for_x (x : ℕ) 
  (h : 225 + 2 * 15 * 4 + 16 = x) : x = 361 := 
by 
  sorry

end solve_for_x_l58_58225


namespace max_value_amc_am_mc_ca_l58_58136

theorem max_value_amc_am_mc_ca (A M C : ℕ) 
  (h : A + M + C = 15) : 
  A * M * C + A * M + M * C + C * A ≤ 200 := 
sorry

end max_value_amc_am_mc_ca_l58_58136


namespace opposite_of_fraction_reciprocal_of_fraction_absolute_value_of_fraction_l58_58256

def improper_fraction : ℚ := -4/3

theorem opposite_of_fraction : -improper_fraction = 4/3 :=
by sorry

theorem reciprocal_of_fraction : (improper_fraction⁻¹) = -3/4 :=
by sorry

theorem absolute_value_of_fraction : |improper_fraction| = 4/3 :=
by sorry

end opposite_of_fraction_reciprocal_of_fraction_absolute_value_of_fraction_l58_58256


namespace tan_7pi_over_6_eq_1_over_sqrt_3_l58_58072

theorem tan_7pi_over_6_eq_1_over_sqrt_3 : 
  ∀ θ : ℝ, θ = (7 * Real.pi) / 6 → Real.tan θ = 1 / Real.sqrt 3 :=
by
  intros θ hθ
  rw [hθ]
  sorry  -- Proof to be completed

end tan_7pi_over_6_eq_1_over_sqrt_3_l58_58072


namespace negation_of_prop_l58_58106

variable (x : ℝ)
def prop (x : ℝ) := x ∈ Set.Ici 0 → Real.exp x ≥ 1

theorem negation_of_prop :
  (¬ ∀ x ∈ Set.Ici 0, Real.exp x ≥ 1) = ∃ x ∈ Set.Ici 0, Real.exp x < 1 :=
by
  sorry

end negation_of_prop_l58_58106


namespace points_on_line_any_real_n_l58_58552

theorem points_on_line_any_real_n (m n : ℝ) 
  (h1 : m = 2 * n + 5) 
  (h2 : m + 1 = 2 * (n + 0.5) + 5) : 
  True :=
by
  sorry

end points_on_line_any_real_n_l58_58552


namespace range_of_m_l58_58374

open Set

variable (f : ℝ → ℝ) (m : ℝ)

theorem range_of_m (h1 : ∀ x1 x2 : ℝ, x1 < x2 → f x1 > f x2)
  (h2 : f (2 * m) > f (1 + m)) : m < 1 :=
by {
  -- The proof would go here.
  sorry
}

end range_of_m_l58_58374


namespace teacups_count_l58_58933

theorem teacups_count (total_people teacup_capacity : ℕ) (H1 : total_people = 63) (H2 : teacup_capacity = 9) : total_people / teacup_capacity = 7 :=
by
  sorry

end teacups_count_l58_58933


namespace new_average_l58_58716

theorem new_average (n : ℕ) (average : ℝ) (new_average : ℝ) 
  (h1 : n = 10)
  (h2 : average = 80)
  (h3 : new_average = (2 * average * n) / n) : 
  new_average = 160 := 
by 
  simp [h1, h2, h3]
  sorry

end new_average_l58_58716


namespace prob_win_all_6_games_prob_win_exactly_5_out_of_6_games_l58_58185

noncomputable def prob_win_single_game : ℚ := 7 / 10
noncomputable def prob_lose_single_game : ℚ := 3 / 10

theorem prob_win_all_6_games : (prob_win_single_game ^ 6) = 117649 / 1000000 :=
by
  sorry

theorem prob_win_exactly_5_out_of_6_games : (6 * (prob_win_single_game ^ 5) * prob_lose_single_game) = 302526 / 1000000 :=
by
  sorry

end prob_win_all_6_games_prob_win_exactly_5_out_of_6_games_l58_58185


namespace greatest_value_of_x_l58_58028

theorem greatest_value_of_x
  (x : ℕ)
  (h1 : x % 4 = 0) -- x is a multiple of 4
  (h2 : x > 0) -- x is positive
  (h3 : x^3 < 2000) -- x^3 < 2000
  : x ≤ 12 :=
by
  sorry

end greatest_value_of_x_l58_58028


namespace mango_price_reduction_l58_58563

theorem mango_price_reduction (P R : ℝ) (M : ℕ)
  (hP_orig : 110 * P = 366.67)
  (hM : M * P = 360)
  (hR_red : (M + 12) * R = 360) :
  ((P - R) / P) * 100 = 10 :=
by sorry

end mango_price_reduction_l58_58563


namespace positive_number_square_sum_eq_210_l58_58454

theorem positive_number_square_sum_eq_210 (n : ℕ) (h : n^2 + n = 210) : n = 14 :=
sorry

end positive_number_square_sum_eq_210_l58_58454


namespace find_positive_number_l58_58453

theorem find_positive_number (n : ℕ) (h : n^2 + n = 210) : n = 14 :=
sorry

end find_positive_number_l58_58453


namespace find_k_l58_58117

noncomputable def k := 3

theorem find_k :
  (∀ x : ℝ, (Real.sin x ^ k) * (Real.sin (k * x)) + (Real.cos x ^ k) * (Real.cos (k * x)) = Real.cos (2 * x) ^ k) ↔ k = 3 :=
sorry

end find_k_l58_58117


namespace number_of_integer_values_of_a_l58_58085

theorem number_of_integer_values_of_a (a : ℤ) : 
  (∃ x : ℤ, x^2 + a * x + 9 * a = 0) ↔ 
  (∃ (a_values : Finset ℤ), a_values.card = 6 ∧ ∀ a ∈ a_values, ∃ x : ℤ, x^2 + a * x + 9 * a = 0) :=
by
  sorry

end number_of_integer_values_of_a_l58_58085


namespace school_year_length_l58_58565

theorem school_year_length
  (children : ℕ)
  (juice_boxes_per_child_per_day : ℕ)
  (days_per_week : ℕ)
  (total_juice_boxes : ℕ)
  (w : ℕ)
  (h1 : children = 3)
  (h2 : juice_boxes_per_child_per_day = 1)
  (h3 : days_per_week = 5)
  (h4 : total_juice_boxes = 375)
  (h5 : total_juice_boxes = children * juice_boxes_per_child_per_day * days_per_week * w)
  : w = 25 :=
by
  sorry

end school_year_length_l58_58565


namespace sum_of_coeffs_eq_neg30_l58_58508

noncomputable def expanded : Polynomial ℤ := 
  -(Polynomial.C 4 - Polynomial.X) * (Polynomial.X + 3 * (Polynomial.C 4 - Polynomial.X))

theorem sum_of_coeffs_eq_neg30 : (expanded.coeffs.sum) = -30 := 
  sorry

end sum_of_coeffs_eq_neg30_l58_58508


namespace sum_of_first_8_terms_l58_58997

theorem sum_of_first_8_terms (a : ℝ) (h : 15 * a = 1) : 
  (a + 2 * a + 4 * a + 8 * a + 16 * a + 32 * a + 64 * a + 128 * a) = 17 :=
by
  sorry

end sum_of_first_8_terms_l58_58997


namespace necessary_but_not_sufficient_l58_58880

theorem necessary_but_not_sufficient (x y : ℕ) : x + y = 3 → (x = 1 ∧ y = 2) ↔ (¬ (x = 0 ∧ y = 3)) := by
  sorry

end necessary_but_not_sufficient_l58_58880


namespace region_volume_is_two_thirds_l58_58600

noncomputable def volume_of_region : ℝ :=
  let region := {p : ℝ × ℝ × ℝ | |p.1| + |p.2| + |p.3| ≤ 2 ∧ |p.1| + |p.2| + |p.3 - 2| ≤ 2}
  -- Assuming volume function calculates the volume of the region
  volume region

theorem region_volume_is_two_thirds :
  volume_of_region = 2 / 3 :=
by
  sorry

end region_volume_is_two_thirds_l58_58600


namespace jenna_less_than_bob_l58_58813

theorem jenna_less_than_bob :
  ∀ (bob jenna phil : ℕ),
  (bob = 60) →
  (phil = bob / 3) →
  (jenna = 2 * phil) →
  (bob - jenna = 20) :=
by
  intros bob jenna phil h1 h2 h3
  sorry

end jenna_less_than_bob_l58_58813


namespace min_value_of_2x_plus_y_l58_58810

theorem min_value_of_2x_plus_y 
  (x y : ℝ) 
  (hx : x > 0) 
  (hy : y > 0) 
  (h : 1 / (x + 1) + 1 / (x + 2 * y) = 1) : 
  (2 * x + y) = 1 / 2 + Real.sqrt 3 := 
sorry

end min_value_of_2x_plus_y_l58_58810


namespace a3_plus_a4_value_l58_58090

theorem a3_plus_a4_value
  (a_0 a_1 a_2 a_3 a_4 a_5 : ℝ)
  (h : (1 - 2*x)^5 = a_0 + a_1*(1 + x) + a_2*(1 + x)^2 + a_3*(1 + x)^3 + a_4*(1 + x)^4 + a_5*(1 + x)^5) :
  a_3 + a_4 = -480 := 
sorry

end a3_plus_a4_value_l58_58090


namespace complex_eq_l58_58951

theorem complex_eq (a b : ℝ) (i : ℂ) (hi : i^2 = -1) (h : (a + 2 * i) / i = b + i) : a + b = 1 :=
sorry

end complex_eq_l58_58951


namespace total_people_on_bus_l58_58175

-- Definitions of the conditions
def num_boys : ℕ := 50
def num_girls : ℕ := (2 / 5 : ℚ) * num_boys
def num_students : ℕ := num_boys + num_girls.toNat
def num_non_students : ℕ := 3 -- Mr. Gordon, the driver, and the assistant

-- The theorem to be proven
theorem total_people_on_bus : num_students + num_non_students = 123 := by
  sorry

end total_people_on_bus_l58_58175


namespace solve_system_eq_l58_58992

theorem solve_system_eq (x y : ℝ) :
  x^2 * y - x * y^2 - 5 * x + 5 * y + 3 = 0 ∧
  x^3 * y - x * y^3 - 5 * x^2 + 5 * y^2 + 15 = 0 ↔
  x = 4 ∧ y = 1 :=
sorry

end solve_system_eq_l58_58992


namespace center_of_circle_in_second_quadrant_l58_58278

theorem center_of_circle_in_second_quadrant (a b : ℝ) 
  (h1 : a < 0) 
  (h2 : b > 0) : 
  ∃ (q : ℕ), q = 2 := 
by 
  sorry

end center_of_circle_in_second_quadrant_l58_58278


namespace find_B_from_period_l58_58493

theorem find_B_from_period (A B C D : ℝ) (h : B ≠ 0) (period_condition : 2 * |2 * π / B| = 4 * π) : B = 1 := sorry

end find_B_from_period_l58_58493


namespace smallest_n_multiple_of_7_l58_58427

theorem smallest_n_multiple_of_7 (x y n : ℤ) (h1 : x + 2 ≡ 0 [ZMOD 7]) (h2 : y - 2 ≡ 0 [ZMOD 7]) :
  x^2 + x * y + y^2 + n ≡ 0 [ZMOD 7] → n = 3 :=
by
  sorry

end smallest_n_multiple_of_7_l58_58427


namespace save_percentage_l58_58481

theorem save_percentage (I S : ℝ) 
  (h1 : 1.5 * I - 2 * S + (I - S) = 2 * (I - S))
  (h2 : I ≠ 0) : 
  S / I = 0.5 :=
by sorry

end save_percentage_l58_58481


namespace money_raised_by_full_price_tickets_l58_58488

theorem money_raised_by_full_price_tickets (f h : ℕ) (p revenue total_tickets : ℕ) 
  (full_price : p = 20) (total_cost : f * p + h * (p / 2) = revenue) 
  (ticket_count : f + h = total_tickets) (total_revenue : revenue = 2750)
  (ticket_number : total_tickets = 180) : f * p = 1900 := 
by
  sorry

end money_raised_by_full_price_tickets_l58_58488


namespace remainder_when_divided_by_x_minus_4_l58_58906

noncomputable def f (x : ℝ) : ℝ := x^4 - 9 * x^3 + 21 * x^2 + x - 18

theorem remainder_when_divided_by_x_minus_4 : f 4 = 2 :=
by
  sorry

end remainder_when_divided_by_x_minus_4_l58_58906


namespace four_edge_trips_count_l58_58435

-- Defining points and edges of the cube
inductive Point
| A | B | C | D | E | F | G | H

open Point

-- Edges of the cube are connections between points
def Edge (p1 p2 : Point) : Prop :=
  ∃ (edges : List (Point × Point)), 
    edges = [(A, B), (A, D), (A, E), (B, C), (B, E), (B, F), (C, D), (C, F), (C, G), (D, E), (D, F), (D, H), (E, F), (E, H), (F, G), (F, H), (G, H)] ∧ 
    ((p1, p2) ∈ edges ∨ (p2, p1) ∈ edges)

-- Define the proof statement
theorem four_edge_trips_count : 
  ∃ (num_paths : ℕ), num_paths = 12 :=
sorry

end four_edge_trips_count_l58_58435


namespace below_sea_level_representation_l58_58821

def depth_below_sea_level (depth: Int → Prop) :=
  ∀ x, depth x → x < 0

theorem below_sea_level_representation (D: Int) (A: Int) 
  (hA: A = 9050) (hD: D = 10907) 
  (above_sea_level: Int → Prop) (below_sea_level: Int → Prop)
  (h1: ∀ y, above_sea_level y → y > 0):
  below_sea_level (-D) :=
by
  sorry

end below_sea_level_representation_l58_58821


namespace probability_triplet_1_2_3_in_10_rolls_l58_58212

noncomputable def probability_of_triplet (n : ℕ) : ℝ :=
  let A0 := (6^10 : ℝ)
  let A1 := (8 * 6^7 : ℝ)
  let A2 := (15 * 6^4 : ℝ)
  let A3 := (4 * 6 : ℝ)
  let total := A0
  let p := (A0 - (total - (A1 - A2 + A3))) / total
  p

theorem probability_triplet_1_2_3_in_10_rolls : 
  abs (probability_of_triplet 10 - 0.0367) < 0.0001 :=
by
  sorry

end probability_triplet_1_2_3_in_10_rolls_l58_58212


namespace parametric_line_eq_l58_58717

theorem parametric_line_eq (t : ℝ) :
  ∃ t : ℝ, ∃ x : ℝ, ∃ y : ℝ, 
  (x = 3 * t + 5) ∧ (y = 6 * t - 7) → y = 2 * x - 17 :=
by
  sorry

end parametric_line_eq_l58_58717


namespace train_A_distance_travelled_l58_58744

/-- Let Train A and Train B start from opposite ends of a 200-mile route at the same time.
Train A has a constant speed of 20 miles per hour, and Train B has a constant speed of 200 miles / 6 hours (which is approximately 33.33 miles per hour).
Prove that Train A had traveled 75 miles when it met Train B. --/
theorem train_A_distance_travelled:
  ∀ (T : Type) (start_time : T) (distance : ℝ) (speed_A : ℝ) (speed_B : ℝ) (meeting_time : ℝ),
  distance = 200 ∧ speed_A = 20 ∧ speed_B = 33.33 ∧ meeting_time = 200 / (speed_A + speed_B) → 
  (speed_A * meeting_time = 75) :=
by
  sorry

end train_A_distance_travelled_l58_58744


namespace max_tan2alpha_l58_58360

variable (α β : ℝ)
variable (hα : 0 < α ∧ α < Real.pi / 2)
variable (hβ : 0 < β ∧ β < Real.pi / 2)
variable (h : Real.tan (α + β) = 2 * Real.tan β)

theorem max_tan2alpha : 
    Real.tan (2 * α) = 4 * Real.sqrt 2 / 7 := 
by 
  sorry

end max_tan2alpha_l58_58360


namespace evaluate_x2_plus_y2_l58_58955

theorem evaluate_x2_plus_y2 (x y : ℝ) (h₁ : 3 * x + 2 * y = 20) (h₂ : 4 * x + 2 * y = 26) : x^2 + y^2 = 37 := by
  sorry

end evaluate_x2_plus_y2_l58_58955


namespace triangle_base_length_l58_58305

theorem triangle_base_length (height : ℝ) (area : ℝ) (base : ℝ) 
  (h_height : height = 6) (h_area : area = 9) 
  (h_formula : area = (1/2) * base * height) : 
  base = 3 :=
by
  sorry

end triangle_base_length_l58_58305


namespace positive_number_square_sum_eq_210_l58_58455

theorem positive_number_square_sum_eq_210 (n : ℕ) (h : n^2 + n = 210) : n = 14 :=
sorry

end positive_number_square_sum_eq_210_l58_58455


namespace increase_80_by_50_percent_l58_58230

theorem increase_80_by_50_percent :
  let initial_number : ℕ := 80
  let increase_percentage : ℝ := 0.5
  initial_number + (initial_number * increase_percentage) = 120 :=
by
  sorry

end increase_80_by_50_percent_l58_58230


namespace find_a_perpendicular_lines_l58_58373

theorem find_a_perpendicular_lines (a : ℝ) :
    (∀ x y : ℝ, a * x - y + 2 * a = 0 → (2 * a - 1) * x + a * y + a = 0) →
    (a = 0 ∨ a = 1) :=
by
  intro h
  sorry

end find_a_perpendicular_lines_l58_58373


namespace maximum_value_l58_58323

-- Define the variables as positive real numbers
variables (a b c : ℝ)

-- Define the conditions
def condition (a b c : ℝ) : Prop := a > 0 ∧ b > 0 ∧ c > 0 ∧ a^2 + b^2 + c^2 = 2*a*b*c + 1

-- Define the expression
def expr (a b c : ℝ) : ℝ := (a - 2*b*c) * (b - 2*c*a) * (c - 2*a*b)

-- The theorem stating that under the given conditions, the expression has a maximum value of 1/8
theorem maximum_value : ∀ (a b c : ℝ), condition a b c → expr a b c ≤ 1/8 :=
by
  sorry

end maximum_value_l58_58323


namespace tangent_line_at_P0_is_parallel_l58_58796

noncomputable def curve (x : ℝ) : ℝ := x^3 + x - 2

def tangent_slope (x : ℝ) : ℝ := 3 * x^2 + 1

theorem tangent_line_at_P0_is_parallel (x y : ℝ) (h_curve : y = curve x) (h_slope : tangent_slope x = 4) :
  (x, y) = (-1, -4) :=
sorry

end tangent_line_at_P0_is_parallel_l58_58796


namespace range_of_m_l58_58531

theorem range_of_m (m : ℝ) :
  (¬(∀ x : ℝ, x^2 - m * x + 1 > 0 → -2 < m ∧ m < 2)) ∧
  (∃ x : ℝ, x^2 < 9 - m^2) ∧
  (-3 < m ∧ m < 3) →
  ((-3 < m ∧ m ≤ -2) ∨ (2 ≤ m ∧ m < 3)) :=
by sorry

end range_of_m_l58_58531


namespace restaurant_cost_l58_58342

theorem restaurant_cost (total_people kids adult_cost : ℕ)
  (h1 : total_people = 12)
  (h2 : kids = 7)
  (h3 : adult_cost = 3) :
  total_people - kids * adult_cost = 15 := by
  sorry

end restaurant_cost_l58_58342


namespace fraction_product_l58_58937

theorem fraction_product : (2 * (-4)) / (9 * 5) = -8 / 45 :=
  by sorry

end fraction_product_l58_58937


namespace age_of_B_l58_58053

-- Define the ages based on the conditions
def A (x : ℕ) : ℕ := 2 * x + 2
def B (x : ℕ) : ℕ := 2 * x
def C (x : ℕ) : ℕ := x

-- The main statement to be proved
theorem age_of_B (x : ℕ) (h : A x + B x + C x = 72) : B 14 = 28 :=
by
  -- we need the proof here but we will put sorry for now
  sorry

end age_of_B_l58_58053


namespace intersection_M_N_l58_58973

def M := { x : ℝ | x^2 - 2 * x < 0 }
def N := { x : ℝ | abs x < 1 }

theorem intersection_M_N : M ∩ N = { x : ℝ | 0 < x ∧ x < 1 } :=
by
  sorry

end intersection_M_N_l58_58973


namespace max_value_of_expression_l58_58151

theorem max_value_of_expression (A M C : ℕ) (h : A + M + C = 15) :
  A * M * C + A * M + M * C + C * A ≤ 200 :=
sorry

end max_value_of_expression_l58_58151


namespace series_sum_eq_three_halves_l58_58077

noncomputable def series (k : ℕ) : ℝ := k * (k + 1) / (2 * 3^k)

theorem series_sum_eq_three_halves :
  has_sum (λ k, series k) (3 / 2) :=
sorry

end series_sum_eq_three_halves_l58_58077


namespace range_of_a_l58_58119

theorem range_of_a (a : ℝ) :
  (∀ x ≥ 0, ∃ y ∈ Set.Ici a, y = (x^2 + 2*x + a) / (x + 1)) ↔ a ≤ 2 :=
by
  sorry

end range_of_a_l58_58119


namespace probability_of_consecutive_triplets_l58_58460

def total_ways_to_select_3_days (n : ℕ) : ℕ :=
  Nat.choose n 3

def number_of_consecutive_triplets (n : ℕ) : ℕ :=
  n - 2

theorem probability_of_consecutive_triplets :
  let total_ways := total_ways_to_select_3_days 10
  let consecutive_triplets := number_of_consecutive_triplets 10
  (consecutive_triplets : ℚ) / total_ways = 1 / 15 :=
by
  sorry

end probability_of_consecutive_triplets_l58_58460


namespace max_value_of_expression_l58_58255

theorem max_value_of_expression (x y z : ℤ) 
  (h1 : x * y + x + y = 20) 
  (h2 : y * z + y + z = 6) 
  (h3 : x * z + x + z = 2) : 
  x^2 + y^2 + z^2 ≤ 84 :=
sorry

end max_value_of_expression_l58_58255


namespace tangerine_and_orange_percentage_l58_58732

-- Given conditions
def initial_apples := 9
def initial_oranges := 5
def initial_tangerines := 17
def initial_grapes := 12
def initial_kiwis := 7

def removed_oranges := 2
def removed_tangerines := 10
def removed_grapes := 4
def removed_kiwis := 3

def added_oranges := 3
def added_tangerines := 6

-- Computed values based on the initial conditions and changes
def remaining_apples := initial_apples
def remaining_oranges := initial_oranges - removed_oranges + added_oranges
def remaining_tangerines := initial_tangerines - removed_tangerines + added_tangerines
def remaining_grapes := initial_grapes - removed_grapes
def remaining_kiwis := initial_kiwis - removed_kiwis

def total_remaining_fruits := remaining_apples + remaining_oranges + remaining_tangerines + remaining_grapes + remaining_kiwis
def total_citrus_fruits := remaining_oranges + remaining_tangerines

-- Statement to prove
def citrus_percentage := (total_citrus_fruits : ℚ) / total_remaining_fruits * 100

theorem tangerine_and_orange_percentage : citrus_percentage = 47.5 := by
  sorry

end tangerine_and_orange_percentage_l58_58732


namespace sequence_remainder_mod_10_l58_58947

def T : ℕ → ℕ := sorry -- Since the actual recursive definition is part of solution steps, we abstract it.
def remainder (n k : ℕ) : ℕ := n % k

theorem sequence_remainder_mod_10 (n : ℕ) (h: n = 2023) : remainder (T n) 10 = 6 :=
by 
  sorry

end sequence_remainder_mod_10_l58_58947


namespace machine_does_not_require_repair_l58_58039

noncomputable def nominal_mass := 390 -- The nominal mass M is 390 grams

def greatest_deviation_preserved := 39 -- The greatest deviation among preserved measurements is 39 grams

def deviation_unread_measurements (x : ℕ) : Prop := x < 39 -- Deviations of unread measurements are less than 39 grams

def all_deviations_no_more_than := ∀ (x : ℕ), x ≤ 39 -- All deviations are no more than 39 grams

theorem machine_does_not_require_repair 
  (M : ℕ) 
  (h_nominal_mass : M = nominal_mass)
  (h_greatest_deviation : greatest_deviation_preserved ≤ 0.1 * M)
  (h_unread_deviations : ∀ (x : ℕ), deviation_unread_measurements x) 
  (h_all_deviations : all_deviations_no_more_than):
  true := -- Prove the machine does not require repair
sorry

end machine_does_not_require_repair_l58_58039


namespace mutually_exclusive_not_complementary_l58_58898

def event_odd (n : ℕ) : Prop := n = 1 ∨ n = 3 ∨ n = 5
def event_greater_than_5 (n : ℕ) : Prop := n = 6

theorem mutually_exclusive_not_complementary :
  (∀ n : ℕ, event_odd n → ¬ event_greater_than_5 n) ∧
  (∃ n : ℕ, ¬ event_odd n ∧ ¬ event_greater_than_5 n) :=
by
  sorry

end mutually_exclusive_not_complementary_l58_58898


namespace M_intersection_N_l58_58962

-- Define the sets M and N
def M : Set ℝ := {x | 0 ≤ x ∧ x ≤ 1}
def N : Set ℝ := {x | -1 < x ∧ x < 1}

-- Define the proof problem
theorem M_intersection_N : M ∩ N = {x | 0 ≤ x ∧ x < 1} := by
  sorry

end M_intersection_N_l58_58962


namespace subset_relation_l58_58854

def M : Set ℝ := {x | x < 9}
def N : Set ℝ := {x | x^2 < 9}

theorem subset_relation : N ⊆ M := by
  sorry

end subset_relation_l58_58854


namespace find_n_l58_58941

theorem find_n (n : ℕ) (h : n * n.factorial + n.factorial = 720) : n = 5 :=
sorry

end find_n_l58_58941


namespace translate_line_downwards_l58_58888

theorem translate_line_downwards :
  ∀ (x : ℝ), (∀ (y : ℝ), (y = 2 * x + 1) → (y - 2 = 2 * x - 1)) :=
by
  intros x y h
  rw [h]
  sorry

end translate_line_downwards_l58_58888


namespace no_solution_condition_l58_58670

theorem no_solution_condition (m : ℝ) : (∀ x : ℝ, (3 * x - m) / (x - 2) ≠ 1) → m = 6 :=
by
  sorry

end no_solution_condition_l58_58670


namespace pencils_ratio_l58_58211

theorem pencils_ratio (T S Ti : ℕ) 
  (h1 : T = 6 * S)
  (h2 : T = 12)
  (h3 : Ti = 16) : Ti / S = 8 := by
  sorry

end pencils_ratio_l58_58211


namespace parallel_lines_slope_equal_intercepts_lines_l58_58534

theorem parallel_lines_slope (m : ℝ) :
  (∀ x y, (2 * x - y - 3 = 0 ∧ x - m * y + 1 - 3 * m = 0) → 2 = (1 / m)) → m = 1 / 2 :=
by
  intro h
  sorry

theorem equal_intercepts_lines (m : ℝ) :
  (m ≠ 0 → (∀ x y, (x - m * y + 1 - 3 * m = 0) → (1 - 3 * m) / m = 3 * m - 1)) →
  (m = -1 ∨ m = 1 / 3) →
  ∀ x y, (x - m * y + 1 - 3 * m = 0) →
  (x + y + 4 = 0 ∨ 3 * x - y = 0) :=
by
  intro h hm
  sorry

end parallel_lines_slope_equal_intercepts_lines_l58_58534


namespace sum_of_g1_values_l58_58855

noncomputable def g : Polynomial ℝ := sorry

theorem sum_of_g1_values :
  (∀ x : ℝ, x ≠ 0 → g.eval (x-1) + g.eval x + g.eval (x+1) = (g.eval x)^2 / (4036 * x)) →
  g.degree ≠ 0 →
  g.eval 1 = 12108 :=
by
  sorry

end sum_of_g1_values_l58_58855


namespace cos_A_is_one_l58_58549

-- Definitions as per Lean's requirement
variable (A B C D : Type) [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D]

-- Declaring the conditions are given
variables (α : ℝ) (cos_A : ℝ)
variables (AB CD AD BC : ℝ)
def is_convex_quadrilateral (A B C D : Type) : Prop := 
  sorry -- This would be a formal definition of convex quadrilateral

-- The conditions are specified in Lean terms
variables (h1 : is_convex_quadrilateral A B C D)
variables (h2 : α = 0) -- α = 0 implies cos(α) = 1
variables (h3 : AB = 240)
variables (h4 : CD = 240)
variables (h5 : AD ≠ BC)
variables (h6 : AB + CD + AD + BC = 960)

-- The proof statement to indicate that cos(α) = 1 under the given conditions
theorem cos_A_is_one : cos_A = 1 :=
by
  sorry -- Proof not included as per the instruction

end cos_A_is_one_l58_58549


namespace depth_below_sea_notation_l58_58830

variables (alt_above_sea : ℝ) (depth_below_sea : ℝ)

def notation_above_sea : ℝ := alt_above_sea

def notation_below_sea (d : ℝ) : ℝ := -d

theorem depth_below_sea_notation : alt_above_sea = 9050 → notation_above_sea = 9050 → depth_below_sea = 10907 → notation_below_sea depth_below_sea = -10907 :=
by
  intros h1 h2 h3
  rw [h3, notation_below_sea]
  exact eq.symm sorry

end depth_below_sea_notation_l58_58830


namespace arith_seq_general_formula_l58_58617

noncomputable def increasing_arith_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, d > 0 ∧ ∀ n : ℕ, a (n + 1) = a n + d

theorem arith_seq_general_formula (a : ℕ → ℤ) (d : ℤ)
  (h_arith : increasing_arith_sequence a)
  (h_a1 : a 1 = 1)
  (h_a3 : a 3 = (a 2)^2 - 4) :
  ∀ n, a n = 3 * n - 2 :=
sorry

end arith_seq_general_formula_l58_58617


namespace max_q_value_l58_58146

theorem max_q_value (A M C : ℕ) (h : A + M + C = 15) : 
  (A * M * C + A * M + M * C + C * A) ≤ 200 :=
sorry

end max_q_value_l58_58146


namespace Vanya_number_thought_of_l58_58594

theorem Vanya_number_thought_of :
  ∃ m n : ℕ, m < 10 ∧ n < 10 ∧ (10 * m + n = 81 ∧ (10 * n + m)^2 = 4 * (10 * m + n)) :=
sorry

end Vanya_number_thought_of_l58_58594


namespace oil_bill_increase_l58_58043

theorem oil_bill_increase :
  ∀ (F x : ℝ), 
    (F / 120 = 5 / 4) → 
    ((F + x) / 120 = 3 / 2) → 
    x = 30 :=
by
  intros F x h1 h2
  -- proof
  sorry

end oil_bill_increase_l58_58043


namespace find_three_digit_number_l58_58767

theorem find_three_digit_number (a b c : ℕ) (h1 : a + b + c = 16)
    (h2 : 100 * b + 10 * a + c = 100 * a + 10 * b + c - 360)
    (h3 : 100 * a + 10 * c + b = 100 * a + 10 * b + c + 54) :
    100 * a + 10 * b + c = 628 :=
by
  sorry

end find_three_digit_number_l58_58767


namespace units_digit_smallest_n_l58_58297

theorem units_digit_smallest_n (n : ℕ) (h1 : 7 * n ≥ 10^2015) (h2 : 7 * (n - 1) < 10^2015) : (n % 10) = 6 :=
sorry

end units_digit_smallest_n_l58_58297


namespace depth_notation_l58_58827

-- Definition of depth and altitude
def above_sea_level (h : ℝ) : Prop := h > 0 
def below_sea_level (h : ℝ) : Prop := h < 0 

-- Given conditions
axiom height_Dabaijing : above_sea_level 9050
axiom depth_Haidou1 : below_sea_level (-10907)

-- Proof goal
theorem depth_notation :
  ∀ (d : ℝ), above_sea_level 9050 → below_sea_level (-d) → d = 10907 :=
by
  intros d _ _
  exact sorry

end depth_notation_l58_58827


namespace inequality_for_pos_a_b_c_d_l58_58258

theorem inequality_for_pos_a_b_c_d
  (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a + b) * (b + c) * (c + d) * (d + a) * (1 + (abcd ^ (1/4)))^4
  ≥ 16 * abcd * (1 + a) * (1 + b) * (1 + c) * (1 + d) :=
by
  sorry

end inequality_for_pos_a_b_c_d_l58_58258


namespace fraction_transform_l58_58909

theorem fraction_transform (x : ℕ) (h : 9 * (537 - x) = 463 + x) : x = 437 :=
by
  sorry

end fraction_transform_l58_58909


namespace probability_f_geq_1_l58_58671

noncomputable def f (x : ℝ) : ℝ := 3 * x^2 - x - 1

theorem probability_f_geq_1 : 
  ∀ x ∈ Icc (-1 : ℝ) 2, 
  let len_total := (2 - (-1)) in -- total length is 3
  let len_sub_interval := (1 - (-2 / 3)) in -- interval length for f(x) >= 1
  let probability := len_sub_interval / len_total in
  probability = 5 / 9 :=
begin
  sorry
end

end probability_f_geq_1_l58_58671


namespace expected_losses_correct_l58_58569

def game_probabilities : List (ℕ × ℝ) := [
  (5, 0.6), (10, 0.75), (15, 0.4), (12, 0.85), (20, 0.5),
  (30, 0.2), (10, 0.9), (25, 0.7), (35, 0.65), (10, 0.8)
]

def expected_losses : ℝ :=
  (1 - 0.6) + (1 - 0.75) + (1 - 0.4) + (1 - 0.85) +
  (1 - 0.5) + (1 - 0.2) + (1 - 0.9) + (1 - 0.7) +
  (1 - 0.65) + (1 - 0.8)

theorem expected_losses_correct :
  expected_losses = 3.55 :=
by {
  -- Skipping the actual proof and inserting a sorry as instructed
  sorry
}

end expected_losses_correct_l58_58569


namespace all_equal_l58_58953

variable (a : ℕ → ℝ)

axiom h1 : a 1 - 3 * a 2 + 2 * a 3 ≥ 0
axiom h2 : a 2 - 3 * a 3 + 2 * a 4 ≥ 0
axiom h3 : a 3 - 3 * a 4 + 2 * a 5 ≥ 0
axiom h4 : ∀ n, 4 ≤ n ∧ n ≤ 98 → a n - 3 * a (n + 1) + 2 * a (n + 2) ≥ 0
axiom h99 : a 99 - 3 * a 100 + 2 * a 1 ≥ 0
axiom h100 : a 100 - 3 * a 1 + 2 * a 2 ≥ 0

theorem all_equal : ∀ i j : ℕ, 1 ≤ i ∧ i ≤ 100 ∧ 1 ≤ j ∧ j ≤ 100 → a i = a j := by
  sorry

end all_equal_l58_58953


namespace solve_equation_l58_58423

theorem solve_equation (x : ℝ) : (x + 2)^2 - 5 * (x + 2) = 0 ↔ (x = -2 ∨ x = 3) :=
by sorry

end solve_equation_l58_58423


namespace find_a_l58_58848

theorem find_a (a : ℤ) : 
  (∀ K : ℤ, K ≠ 27 → (27 - K) ∣ (a - K^3)) ↔ (a = 3^9) :=
by
  sorry

end find_a_l58_58848


namespace find_original_number_l58_58065

theorem find_original_number (x : ℤ) : 4 * (3 * x + 29) = 212 → x = 8 :=
by
  intro h
  sorry

end find_original_number_l58_58065


namespace find_C_l58_58588

theorem find_C (A B C D : ℕ) (h_diff : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D)
  (h_eq : 4000 + 100 * A + 50 + B + (1000 * C + 200 + 10 * D + 7) = 7070) : C = 2 :=
sorry

end find_C_l58_58588


namespace max_positive_root_satisfies_range_l58_58778

noncomputable def max_positive_root_in_range (b c d : ℝ) (hb : |b| ≤ 1) (hc : |c| ≤ 1.5) (hd : |d| ≤ 1) : Prop :=
  ∃ s : ℝ, 2.5 ≤ s ∧ s < 3 ∧ ∃ x : ℝ, x > 0 ∧ x^3 + b * x^2 + c * x + d = 0

theorem max_positive_root_satisfies_range (b c d : ℝ) (hb : |b| ≤ 1) (hc : |c| ≤ 1.5) (hd : |d| ≤ 1) :
  max_positive_root_in_range b c d hb hc hd := sorry

end max_positive_root_satisfies_range_l58_58778


namespace line_passes_through_fixed_point_l58_58363

theorem line_passes_through_fixed_point (p q : ℝ) (h : 3 * p - 2 * q = 1) :
  p * (-3 / 2) + 3 * (1 / 6) + q = 0 :=
by 
  sorry

end line_passes_through_fixed_point_l58_58363


namespace depth_notation_l58_58828

-- Definition of depth and altitude
def above_sea_level (h : ℝ) : Prop := h > 0 
def below_sea_level (h : ℝ) : Prop := h < 0 

-- Given conditions
axiom height_Dabaijing : above_sea_level 9050
axiom depth_Haidou1 : below_sea_level (-10907)

-- Proof goal
theorem depth_notation :
  ∀ (d : ℝ), above_sea_level 9050 → below_sea_level (-d) → d = 10907 :=
by
  intros d _ _
  exact sorry

end depth_notation_l58_58828


namespace pick_peanut_cluster_percentage_l58_58499

def total_chocolates := 100
def typeA_caramels := 5
def typeB_caramels := 6
def typeC_caramels := 4
def typeD_nougats := 2 * typeA_caramels
def typeE_nougats := 2 * typeB_caramels
def typeF_truffles := typeA_caramels + 6
def typeG_truffles := typeB_caramels + 6
def typeH_truffles := typeC_caramels + 6

def total_non_peanut_clusters := 
  typeA_caramels + typeB_caramels + typeC_caramels + typeD_nougats + typeE_nougats + typeF_truffles + typeG_truffles + typeH_truffles

def number_peanut_clusters := total_chocolates - total_non_peanut_clusters

def percent_peanut_clusters := (number_peanut_clusters * 100) / total_chocolates

theorem pick_peanut_cluster_percentage : percent_peanut_clusters = 30 := 
by {
  sorry
}

end pick_peanut_cluster_percentage_l58_58499


namespace Joel_laps_count_l58_58611

-- Definitions of conditions
def Yvonne_laps := 10
def sister_laps := Yvonne_laps / 2
def Joel_laps := sister_laps * 3

-- Statement to be proved
theorem Joel_laps_count : Joel_laps = 15 := by
  -- currently, proof is not required, so we defer it with 'sorry'
  sorry

end Joel_laps_count_l58_58611


namespace four_m0_as_sum_of_primes_l58_58101

theorem four_m0_as_sum_of_primes (m0 : ℕ) (h1 : m0 > 1) 
  (h2 : ∀ n : ℕ, ∃ p : ℕ, Prime p ∧ n ≤ p ∧ p ≤ 2 * n) 
  (h3 : ∀ p1 p2 : ℕ, Prime p1 → Prime p2 → (2 * m0 ≠ p1 + p2)) : 
  ∃ p1 p2 p3 p4 : ℕ, Prime p1 ∧ Prime p2 ∧ Prime p3 ∧ Prime p4 ∧ (4 * m0 = p1 + p2 + p3 + p4) ∨ (∃ p1 p2 p3 : ℕ, Prime p1 ∧ Prime p2 ∧ Prime p3 ∧ 4 * m0 = p1 + p2 + p3) :=
by sorry

end four_m0_as_sum_of_primes_l58_58101


namespace janet_spends_more_on_piano_l58_58291

-- Condition definitions
def clarinet_hourly_rate : ℝ := 40
def clarinet_hours_per_week : ℝ := 3
def piano_hourly_rate : ℝ := 28
def piano_hours_per_week : ℝ := 5
def weeks_per_year : ℝ := 52

-- Calculations based on conditions
def weekly_cost_clarinet : ℝ := clarinet_hourly_rate * clarinet_hours_per_week
def weekly_cost_piano : ℝ := piano_hourly_rate * piano_hours_per_week
def weekly_difference : ℝ := weekly_cost_piano - weekly_cost_clarinet
def yearly_difference : ℝ := weekly_difference * weeks_per_year

theorem janet_spends_more_on_piano : yearly_difference = 1040 := by
  sorry 

end janet_spends_more_on_piano_l58_58291


namespace length_of_platform_l58_58216

variables (t L T_p T_s : ℝ)
def train_length := 200  -- length of the train in meters
def platform_cross_time := 50  -- time in seconds to cross the platform
def pole_cross_time := 42  -- time in seconds to cross the signal pole

theorem length_of_platform :
  T_p = platform_cross_time ->
  T_s = pole_cross_time ->
  t = train_length ->
  (L = 38) :=
by
  intros hp hsp ht
  sorry  -- proof goes here

end length_of_platform_l58_58216


namespace largest_ball_radius_l58_58337

def torus_inner_radius : ℝ := 2
def torus_outer_radius : ℝ := 4
def circle_center : ℝ × ℝ × ℝ := (3, 0, 1)
def circle_radius : ℝ := 1

theorem largest_ball_radius : ∃ r : ℝ, r = 9 / 4 ∧
  (∃ (sphere_center : ℝ × ℝ × ℝ) (torus_center : ℝ × ℝ × ℝ),
  (sphere_center = (0, 0, r)) ∧
  (torus_center = (3, 0, 1)) ∧
  (dist (0, 0, r) (3, 0, 1) = r + 1)) := sorry

end largest_ball_radius_l58_58337


namespace floodDamageInUSD_l58_58924

def floodDamageAUD : ℝ := 45000000
def exchangeRateAUDtoUSD : ℝ := 1.2

theorem floodDamageInUSD : floodDamageAUD * (1 / exchangeRateAUDtoUSD) = 37500000 := 
by 
  sorry

end floodDamageInUSD_l58_58924


namespace func_positive_range_l58_58082

theorem func_positive_range (a : ℝ) : 
  (∀ x : ℝ, (5 - a) * x^2 - 6 * x + a + 5 > 0) → (-4 < a ∧ a < 4) := 
by 
  sorry

end func_positive_range_l58_58082


namespace range_of_m_if_p_range_of_m_if_p_and_q_l58_58799

variable (m : ℝ)

def proposition_p (m : ℝ) : Prop :=
  (3 - m > m - 1) ∧ (m - 1 > 0)

def proposition_q (m : ℝ) : Prop :=
  m^2 - 9 / 4 < 0

theorem range_of_m_if_p (m : ℝ) (hp : proposition_p m) : 1 < m ∧ m < 2 :=
  sorry

theorem range_of_m_if_p_and_q (m : ℝ) (hp : proposition_p m) (hq : proposition_q m) : 1 < m ∧ m < 3 / 2 :=
  sorry

end range_of_m_if_p_range_of_m_if_p_and_q_l58_58799


namespace denote_depth_below_sea_level_l58_58839

theorem denote_depth_below_sea_level (above_sea_level : Int) (depth_haidou_1 : Int) :
  (above_sea_level > 0) ∧ (depth_haidou_1 < 0) → depth_haidou_1 = -10907 :=
by
  intros h
  sorry

end denote_depth_below_sea_level_l58_58839


namespace depth_below_sea_notation_l58_58833

variables (alt_above_sea : ℝ) (depth_below_sea : ℝ)

def notation_above_sea : ℝ := alt_above_sea

def notation_below_sea (d : ℝ) : ℝ := -d

theorem depth_below_sea_notation : alt_above_sea = 9050 → notation_above_sea = 9050 → depth_below_sea = 10907 → notation_below_sea depth_below_sea = -10907 :=
by
  intros h1 h2 h3
  rw [h3, notation_below_sea]
  exact eq.symm sorry

end depth_below_sea_notation_l58_58833


namespace num_six_digit_asc_digits_l58_58273

theorem num_six_digit_asc_digits : 
  ∃ n : ℕ, n = (Nat.choose 9 3) ∧ n = 84 := 
by
  sorry

end num_six_digit_asc_digits_l58_58273


namespace complex_number_solution_l58_58649

open Complex

noncomputable def findComplexNumber (z : ℂ) : Prop :=
  abs (z - 2) = abs (z + 4) ∧ abs (z - 2) = abs (z - Complex.I * 2)

theorem complex_number_solution :
  ∃ z : ℂ, findComplexNumber z ∧ z = -1 + Complex.I :=
by
  sorry

end complex_number_solution_l58_58649


namespace min_distance_sum_l58_58616

theorem min_distance_sum
  (A B C D E P : ℝ)
  (h_collinear : B = A + 2 ∧ C = B + 2 ∧ D = C + 3 ∧ E = D + 4)
  (h_bisector : P = (A + E) / 2) :
  (A - P)^2 + (B - P)^2 + (C - P)^2 + (D - P)^2 + (E - P)^2 = 77.25 :=
by
  sorry

end min_distance_sum_l58_58616


namespace max_value_expression_l58_58161

theorem max_value_expression (A M C : ℕ) (h₁ : A + M + C = 15) : 
  A * M * C + A * M + M * C + C * A ≤ 200 :=
  sorry

end max_value_expression_l58_58161


namespace triangle_area_correct_l58_58005

theorem triangle_area_correct :
  ∀ (A B C : ℝ) (AB AC : ℝ) (angle_B : ℝ),
  AB = sqrt 3 →
  AC = 1 →
  angle_B = π / 6 →
  let s := (AB + AC + sqrt (AB^2 + AC^2 - 2 * AB * AC * cos angle_B)) / 2 in
  let area := sqrt (s * (s - AB) * (s - AC) * (s - sqrt (AB^2 + AC^2 - 2 * AB * AC * cos angle_B))) in
  area = sqrt 3 / 4 :=
by intros; sorry

end triangle_area_correct_l58_58005


namespace solve_system_of_equations_l58_58380

theorem solve_system_of_equations :
  ∃ y : ℝ, (2 * 2 + y = 0) ∧ (2 + y = 3) :=
by
  sorry

end solve_system_of_equations_l58_58380


namespace arithmetic_sqrt_9_l58_58186

def arithmetic_sqrt (x : ℕ) : ℕ :=
  if h : 0 ≤ x then Nat.sqrt x else 0

theorem arithmetic_sqrt_9 : arithmetic_sqrt 9 = 3 :=
by {
  sorry
}

end arithmetic_sqrt_9_l58_58186


namespace smallest_yellow_marbles_l58_58167

theorem smallest_yellow_marbles :
  ∃ n : ℕ, (n ≡ 0 [MOD 20]) ∧
           (∃ b : ℕ, b = n / 4) ∧
           (∃ r : ℕ, r = n / 5) ∧
           (∃ g : ℕ, g = 10) ∧
           (∃ y : ℕ, y = n - (b + r + g) ∧ y = 1) :=
sorry

end smallest_yellow_marbles_l58_58167


namespace total_number_of_people_on_bus_l58_58170

theorem total_number_of_people_on_bus (boys girls : ℕ)
    (driver assistant teacher : ℕ) 
    (h1 : boys = 50)
    (h2 : girls = boys + (2 * boys / 5))
    (h3 : driver = 1)
    (h4 : assistant = 1)
    (h5 : teacher = 1) :
    (boys + girls + driver + assistant + teacher = 123) :=
by
    sorry

end total_number_of_people_on_bus_l58_58170


namespace sum_of_coefficients_expansion_l58_58505

theorem sum_of_coefficients_expansion (d : ℝ) :
  let expr := -(4 - d) * (d + 3 * (4 - d))
  in (polynomial.sum_of_coeffs expr) = -30 :=
by
  let expr := -(4 - d) * (d + 3 * (4 - d))
  have h_expr : expr = -2 * d^2 + 20 * d - 48, sorry
  have h_coeffs_sum : polynomial.sum_of_coeffs (-2 * d^2 + 20 * d - 48) = -30, sorry
  rw h_expr
  exact h_coeffs_sum

end sum_of_coefficients_expansion_l58_58505


namespace smallest_positive_value_l58_58907

theorem smallest_positive_value (x : ℝ) (hx : x > 0) (h : x / 7 + 2 / (7 * x) = 1) : 
  x = (7 - Real.sqrt 41) / 2 :=
sorry

end smallest_positive_value_l58_58907


namespace team_a_completion_rate_l58_58682

theorem team_a_completion_rate :
  ∃ x : ℝ, (9000 / x - 9000 / (1.5 * x) = 15) ∧ x = 200 :=
by {
  sorry
}

end team_a_completion_rate_l58_58682


namespace fraction_multiplication_l58_58679

-- Given fractions a and b
def a := (1 : ℚ) / 4
def b := (1 : ℚ) / 8

-- The first product result
def result1 := a * b

-- The final product result when multiplied by 4
def result2 := result1 * 4

-- The theorem to prove
theorem fraction_multiplication : result2 = (1 : ℚ) / 8 := by
  sorry

end fraction_multiplication_l58_58679


namespace max_value_amc_am_mc_ca_l58_58135

theorem max_value_amc_am_mc_ca (A M C : ℕ) 
  (h : A + M + C = 15) : 
  A * M * C + A * M + M * C + C * A ≤ 200 := 
sorry

end max_value_amc_am_mc_ca_l58_58135


namespace both_firms_participate_condition_both_firms_will_participate_social_nonoptimal_participation_l58_58916

section RD

variables (V IC α : ℝ) (0 < α ∧ α < 1)

-- Condition for part (a)
def participation_condition : Prop :=
  α * V * (1 - 0.5 * α) ≥ IC

-- Part (b) Definition
def firms_participate_when : Prop :=
  V = 16 ∧ α = 0.5 ∧ IC = 5

-- Part (c) Definition
def social_optimal : Prop :=
  let total_profit_both := 2 * (α * (1 - α) * V + 0.5 * α^2 * V - IC) in
  let total_profit_one := α * V - IC in
  total_profit_one > total_profit_both

-- Theorem for part (a)
theorem both_firms_participate_condition : participation_condition V IC α :=
sorry

-- Theorem for part (b)
theorem both_firms_will_participate (h : firms_participate_when V IC α) : participation_condition 16 5 0.5 :=
sorry

-- Theorem for part (c)
theorem social_nonoptimal_participation (h : firms_participate_when V IC α) : social_optimal 16 IC 0.5 :=
sorry

end RD

end both_firms_participate_condition_both_firms_will_participate_social_nonoptimal_participation_l58_58916


namespace age_difference_l58_58726

variable (A B C D : ℕ)

theorem age_difference (h₁ : A + B > B + C) (h₂ : C = A - 15) : (A + B) - (B + C) = 15 :=
by
  sorry

end age_difference_l58_58726


namespace solve_eq_roots_l58_58026

noncomputable def solve_equation (x : ℝ) : Prop :=
  (7 * x + 2) / (3 * x^2 + 7 * x - 6) = (3 * x) / (3 * x - 2)

theorem solve_eq_roots (x : ℝ) (h₁ : x ≠ 2 / 3) :
  solve_equation x ↔ (x = (-1 + Real.sqrt 7) / 3 ∨ x = (-1 - Real.sqrt 7) / 3) :=
by
  sorry

end solve_eq_roots_l58_58026


namespace carlson_fraction_l58_58298

-- Define variables
variables (n m k p T : ℝ)

theorem carlson_fraction (h1 : k = 0.6 * n)
                         (h2 : p = 2.5 * m)
                         (h3 : T = n * m + k * p) :
                         k * p / T = 3 / 5 := by
  -- Omitted proof
  sorry

end carlson_fraction_l58_58298


namespace production_equation_l58_58759

-- Define the conditions as per the problem
variables (workers : ℕ) (x : ℕ) 

-- The number of total workers is fixed
def total_workers := 44

-- Production rates per worker
def bodies_per_worker := 50
def bottoms_per_worker := 120

-- The problem statement as a Lean theorem
theorem production_equation (h : workers = total_workers) (hx : x ≤ workers) :
  2 * bottoms_per_worker * (total_workers - x) = bodies_per_worker * x :=
by
  sorry

end production_equation_l58_58759


namespace amount_of_money_around_circumference_l58_58483

-- Define the given conditions
def horizontal_coins : ℕ := 6
def vertical_coins : ℕ := 4
def coin_value_won : ℕ := 100

-- The goal is to prove the total amount of money around the circumference
theorem amount_of_money_around_circumference : 
  (2 * (horizontal_coins - 2) + 2 * (vertical_coins - 2) + 4) * coin_value_won = 1600 :=
by
  sorry

end amount_of_money_around_circumference_l58_58483


namespace find_ratio_l58_58075

theorem find_ratio (x y c d : ℝ) (h1 : 8 * x - 6 * y = c) (h2 : 12 * y - 18 * x = d) (hx : x ≠ 0) (hy : y ≠ 0) (hd : d ≠ 0) : c / d = -1 := by
  sorry

end find_ratio_l58_58075


namespace tailoring_business_days_l58_58400

theorem tailoring_business_days
  (shirts_per_day : ℕ)
  (fabric_per_shirt : ℕ)
  (pants_per_day : ℕ)
  (fabric_per_pant : ℕ)
  (total_fabric : ℕ)
  (h1 : shirts_per_day = 3)
  (h2 : fabric_per_shirt = 2)
  (h3 : pants_per_day = 5)
  (h4 : fabric_per_pant = 5)
  (h5 : total_fabric = 93) :
  (total_fabric / (shirts_per_day * fabric_per_shirt + pants_per_day * fabric_per_pant)) = 3 :=
by {
  sorry
}

end tailoring_business_days_l58_58400


namespace range_g_l58_58354

variable (x : ℝ)
noncomputable def g (x : ℝ) : ℝ := 1 / (x - 1)^2

theorem range_g (y : ℝ) : 
  (∃ x, g x = y) ↔ y > 0 :=
by
  sorry

end range_g_l58_58354


namespace positive_number_sum_square_l58_58447

theorem positive_number_sum_square (n : ℕ) (h : n^2 + n = 210) : n = 14 :=
sorry

end positive_number_sum_square_l58_58447


namespace min_f_in_interval_l58_58530

open Real

noncomputable def f (ω x : ℝ) : ℝ := sin (ω * x) - 2 * sqrt 3 * sin (ω * x / 2) ^ 2 + sqrt 3

theorem min_f_in_interval (ω : ℝ) (hω : ω > 0) :
  (∀ x, 0 <= x ∧ x <= π / 2 → f 1 x >= f 1 (π / 3)) :=
by sorry

end min_f_in_interval_l58_58530


namespace measure_of_angle_C_maximum_area_of_triangle_l58_58681

noncomputable def triangle (A B C a b c : ℝ) : Prop :=
  a = 2 * Real.sin A ∧
  b = 2 * Real.sin B ∧
  c = 2 * Real.sin C ∧
  2 * (Real.sin A ^ 2 - Real.sin C ^ 2) = (Real.sqrt 2 * a - b) * Real.sin B

theorem measure_of_angle_C :
  ∀ (A B C a b c : ℝ),
  triangle A B C a b c →
  C = π / 4 :=
by
  intros A B C a b c h
  sorry

theorem maximum_area_of_triangle :
  ∀ (A B C a b c : ℝ),
  triangle A B C a b c →
  C = π / 4 →
  1 / 2 * a * b * Real.sin C = (Real.sqrt 2 / 2 + 1 / 2) :=
by
  intros A B C a b c h hC
  sorry

end measure_of_angle_C_maximum_area_of_triangle_l58_58681


namespace value_taken_away_l58_58116

theorem value_taken_away (n x : ℕ) (h1 : n = 4) (h2 : 2 * n + 20 = 8 * n - x) : x = 4 :=
by
  sorry

end value_taken_away_l58_58116


namespace combined_size_UK_India_US_l58_58777

theorem combined_size_UK_India_US (U : ℝ)
    (Canada : ℝ := 1.5 * U)
    (Russia : ℝ := (1 + 1/3) * Canada)
    (China : ℝ := (1 / 1.7) * Russia)
    (Brazil : ℝ := (2 / 3) * U)
    (Australia : ℝ := (1 / 2) * Brazil)
    (UK : ℝ := 2 * Australia)
    (India : ℝ := (1 / 4) * Russia)
    (India' : ℝ := 6 * UK)
    (h_India : India = India') :
  UK + India = 7 / 6 * U := 
by
  -- Proof details
  sorry

end combined_size_UK_India_US_l58_58777


namespace curve_has_axis_of_symmetry_l58_58882

theorem curve_has_axis_of_symmetry (x y : ℝ) :
  (x^2 - x * y + y^2 + x - y - 1 = 0) ↔ (x+y = 0) :=
sorry

end curve_has_axis_of_symmetry_l58_58882


namespace range_of_m_l58_58263

def proposition_p (m : ℝ) : Prop := ∀ x : ℝ, 2^x - m + 1 > 0
def proposition_q (m : ℝ) : Prop := 5 - 2*m > 1

theorem range_of_m (m : ℝ) (hp : proposition_p m) (hq : proposition_q m) : m ≤ 1 :=
sorry

end range_of_m_l58_58263


namespace hyperbola_center_l58_58942

theorem hyperbola_center (x y : ℝ) :
    9 * x^2 - 18 * x - 16 * y^2 + 64 * y - 143 = 0 →
    (x, y) = (1, 2) :=
sorry

end hyperbola_center_l58_58942


namespace ratio_second_to_first_l58_58769

-- Define the given conditions and variables
variables 
  (total_water : ℕ := 1200)
  (neighborhood1_usage : ℕ := 150)
  (neighborhood4_usage : ℕ := 350)
  (x : ℕ) -- water usage by second neighborhood

-- Define the usage by third neighborhood in terms of the second neighborhood usage
def neighborhood3_usage := x + 100

-- Define remaining water usage after substracting neighborhood 4 usage
def remaining_water := total_water - neighborhood4_usage

-- The sum of water used by neighborhoods
def total_usage_neighborhoods := neighborhood1_usage + neighborhood3_usage x + x

theorem ratio_second_to_first (h : total_usage_neighborhoods x = remaining_water) :
  (x : ℚ) / neighborhood1_usage = 2 := 
by
  sorry

end ratio_second_to_first_l58_58769


namespace triangle_area_not_twice_parallelogram_l58_58306

theorem triangle_area_not_twice_parallelogram (b h : ℝ) :
  (1 / 2) * b * h ≠ 2 * b * h :=
sorry

end triangle_area_not_twice_parallelogram_l58_58306


namespace positive_number_sum_square_l58_58445

theorem positive_number_sum_square (n : ℝ) (h : n^2 + n = 210) (h_pos : n > 0) : n = 14 :=
sorry

end positive_number_sum_square_l58_58445


namespace total_chocolate_bar_count_l58_58239

def large_box_count : ℕ := 150
def small_box_count_per_large_box : ℕ := 45
def chocolate_bar_count_per_small_box : ℕ := 35

theorem total_chocolate_bar_count :
  large_box_count * small_box_count_per_large_box * chocolate_bar_count_per_small_box = 236250 :=
by
  sorry

end total_chocolate_bar_count_l58_58239


namespace fewer_gallons_for_plants_correct_l58_58687

-- Define the initial conditions
def initial_water : ℕ := 65
def water_per_car : ℕ := 7
def total_cars : ℕ := 2
def water_for_cars : ℕ := water_per_car * total_cars
def water_remaining_after_cars : ℕ := initial_water - water_for_cars
def water_for_plates_clothes : ℕ := 24
def water_remaining_before_plates_clothes : ℕ := water_for_plates_clothes * 2
def water_for_plants : ℕ := water_remaining_after_cars - water_remaining_before_plates_clothes

-- Define the query statement
def fewer_gallons_for_plants : Prop := water_per_car - water_for_plants = 4

-- Proof skeleton
theorem fewer_gallons_for_plants_correct : fewer_gallons_for_plants :=
by sorry

end fewer_gallons_for_plants_correct_l58_58687


namespace problem_statement_l58_58162

noncomputable theory

open_locale classical

variables {A B C P A' B' C' : Point}

axiom Point_on_circumcircle (A B C P : Point) : Prop
axiom Parallel (l1 l2 : Line) : Prop
axiom Line_through_points (P1 P2 : Point) : Line

def circumcircle_of_triangle (A B C : Point) : Circle :=
  sorry -- Placeholder for the definition of a circumcircle

def points_on_circle (C : Circle) (P : Point) : Prop :=
  sorry -- Placeholder for the definition of a point being on a circle

def parallel_lines (P1 P2 Q1 Q2 : Point) : Prop :=
  Parallel (Line_through_points P1 P2) (Line_through_points Q1 Q2)

theorem problem_statement
  (hp : Point_on_circumcircle A B C P)
  (hpa : parallel_lines P A' B C)
  (hpb : parallel_lines P B' C A)
  (hpc : parallel_lines P C' A B) :
  parallel_lines A A' B B' ∧ parallel_lines A A' C C' :=
sorry

end problem_statement_l58_58162


namespace joseph_total_power_cost_l58_58853

theorem joseph_total_power_cost:
  let oven_cost := 500 in
  let wh_cost := oven_cost / 2 in
  let fr_cost := 3 * wh_cost in
  let total_cost := oven_cost + wh_cost + fr_cost in
  total_cost = 1500 :=
by
  -- Definitions
  let oven_cost := 500
  let wh_cost := oven_cost / 2
  let fr_cost := 3 * wh_cost
  let total_cost := oven_cost + wh_cost + fr_cost
  -- Main goal
  sorry

end joseph_total_power_cost_l58_58853


namespace strawberries_in_each_handful_l58_58994

theorem strawberries_in_each_handful (x : ℕ) (h : (x - 1) * (75 / x) = 60) : x = 5 :=
sorry

end strawberries_in_each_handful_l58_58994


namespace value_of_5_l58_58124

def q' (q : ℤ) : ℤ := 3 * q - 3

theorem value_of_5'_prime : q' (q' 5) = 33 :=
by
  sorry

end value_of_5_l58_58124


namespace jason_optimal_reroll_probability_l58_58008

-- Define the probability function based on the three dice roll problem
def probability_of_rerolling_two_dice : ℚ := 
  -- As per the problem, the computed and fixed probability is 7/64.
  7 / 64

-- Prove that Jason's optimal strategy leads to rerolling exactly two dice with a probability of 7/64.
theorem jason_optimal_reroll_probability : probability_of_rerolling_two_dice = 7 / 64 := 
  sorry

end jason_optimal_reroll_probability_l58_58008


namespace sqrt_infinite_series_eq_two_l58_58191

theorem sqrt_infinite_series_eq_two (m : ℝ) (hm : 0 < m) :
  (m ^ 2 = 2 + m) → m = 2 :=
by {
  sorry
}

end sqrt_infinite_series_eq_two_l58_58191


namespace number_of_students_in_the_course_l58_58989

variable (T : ℝ)

theorem number_of_students_in_the_course
  (h1 : (1/5) * T + (1/4) * T + (1/2) * T + 40 = T) :
  T = 800 :=
sorry

end number_of_students_in_the_course_l58_58989


namespace positive_number_sum_square_l58_58446

theorem positive_number_sum_square (n : ℕ) (h : n^2 + n = 210) : n = 14 :=
sorry

end positive_number_sum_square_l58_58446


namespace sum_of_roots_l58_58099

theorem sum_of_roots (α β : ℝ)
  (hα : α^3 - 3*α^2 + 5*α - 4 = 0)
  (hβ : β^3 - 3*β^2 + 5*β - 2 = 0) :
  α + β = 2 :=
sorry

end sum_of_roots_l58_58099


namespace sum_of_coefficients_l58_58509

theorem sum_of_coefficients (d : ℤ) : 
  let expr := -(4 - d) * (d + 3 * (4 - d))
  let expanded_form := -2 * d ^ 2 + 20 * d - 48
  let sum_of_coeffs := -2 + 20 - 48
  sum_of_coeffs = -30 :=
by
  -- The proof will go here, skipping for now.
  sorry

end sum_of_coefficients_l58_58509


namespace trailing_zeroes_500_fact_l58_58251

-- Define a function to count multiples of a given number in a range
def countMultiples (n m : Nat) : Nat :=
  m / n

-- Define a function to count trailing zeroes in the factorial
def trailingZeroesFactorial (n : Nat) : Nat :=
  countMultiples 5 n + countMultiples (5^2) n + countMultiples (5^3) n + countMultiples (5^4) n

theorem trailing_zeroes_500_fact : trailingZeroesFactorial 500 = 124 :=
by
  sorry

end trailing_zeroes_500_fact_l58_58251


namespace farmer_feed_full_price_l58_58231

theorem farmer_feed_full_price
  (total_spent : ℕ)
  (chicken_feed_discount_percent : ℕ)
  (chicken_feed_percent : ℕ)
  (goat_feed_percent : ℕ)
  (total_spent_val : total_spent = 35)
  (chicken_feed_discount_percent_val : chicken_feed_discount_percent = 50)
  (chicken_feed_percent_val : chicken_feed_percent = 40)
  (goat_feed_percent_val : goat_feed_percent = 60) :
  (total_spent * chicken_feed_percent / 100 * 2) + (total_spent * goat_feed_percent / 100) = 49 := 
by
  -- Placeholder for proof.
  sorry

end farmer_feed_full_price_l58_58231


namespace equal_real_roots_a_value_l58_58542

theorem equal_real_roots_a_value (a : ℝ) :
  a ≠ 0 →
  let b := -4
  let c := 3
  b * b - 4 * a * c = 0 →
  a = 4 / 3 :=
by
  intros h_nonzero h_discriminant
  sorry

end equal_real_roots_a_value_l58_58542


namespace lines_per_page_l58_58009

theorem lines_per_page
  (total_words : ℕ)
  (words_per_line : ℕ)
  (words_left : ℕ)
  (pages_filled : ℚ) :
  total_words = 400 →
  words_per_line = 10 →
  words_left = 100 →
  pages_filled = 1.5 →
  (total_words - words_left) / words_per_line / pages_filled = 20 := 
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end lines_per_page_l58_58009


namespace pages_per_day_difference_l58_58425

theorem pages_per_day_difference :
  let songhee_pages := 288
  let songhee_days := 12
  let eunju_pages := 243
  let eunju_days := 9
  let songhee_per_day := songhee_pages / songhee_days
  let eunju_per_day := eunju_pages / eunju_days
  eunju_per_day - songhee_per_day = 3 := by
  sorry

end pages_per_day_difference_l58_58425


namespace tan_alpha_second_quadrant_l58_58675

theorem tan_alpha_second_quadrant (α : ℝ) 
(h_cos : Real.cos α = -4/5) 
(h_quadrant : π/2 < α ∧ α < π) : 
  Real.tan α = -3/4 :=
by
  sorry

end tan_alpha_second_quadrant_l58_58675


namespace new_water_intake_recommendation_l58_58733

noncomputable def current_consumption : ℝ := 25
noncomputable def increase_percentage : ℝ := 0.75
noncomputable def increased_amount : ℝ := increase_percentage * current_consumption
noncomputable def new_recommended_consumption : ℝ := current_consumption + increased_amount

theorem new_water_intake_recommendation :
  new_recommended_consumption = 43.75 := 
by 
  sorry

end new_water_intake_recommendation_l58_58733


namespace volume_of_intersecting_octahedra_l58_58601

def absolute (x : ℝ) : ℝ := abs x

noncomputable def volume_of_region : ℝ :=
  let region1 (x y z : ℝ) := absolute x + absolute y + absolute z ≤ 2
  let region2 (x y z : ℝ) := absolute x + absolute y + absolute (z - 2) ≤ 2
  -- The region is the intersection of these two inequalities
  -- However, we calculate its volume directly
  (2 / 3 : ℝ)

theorem volume_of_intersecting_octahedra :
  (volume_of_region : ℝ) = (2 / 3 : ℝ) :=
sorry

end volume_of_intersecting_octahedra_l58_58601


namespace probability_not_late_probability_late_and_misses_bus_l58_58199

variable (P_Sam_late : ℚ)
variable (P_miss_bus_given_late : ℚ)

theorem probability_not_late (h1 : P_Sam_late = 5/9) :
  1 - P_Sam_late = 4/9 := by
  rw [h1]
  norm_num

theorem probability_late_and_misses_bus (h1 : P_Sam_late = 5/9) (h2 : P_miss_bus_given_late = 1/3) :
  P_Sam_late * P_miss_bus_given_late = 5/27 := by
  rw [h1, h2]
  norm_num

#check probability_not_late
#check probability_late_and_misses_bus

end probability_not_late_probability_late_and_misses_bus_l58_58199


namespace isosceles_triangle_aacute_l58_58262

theorem isosceles_triangle_aacute (a b c : ℝ) (h1 : a = b) (h2 : a + b + c = 180) (h3 : c = 108)
  : ∃ x y z : ℝ, x + y + z = 180 ∧ x < 90 ∧ y < 90 ∧ z < 90 ∧ x > 0 ∧ y > 0 ∧ z > 0 :=
by {
  sorry
}

end isosceles_triangle_aacute_l58_58262


namespace sum_log2_geom_seq_first_7_terms_l58_58976

-- Define the geometric sequence {a_n}
def geometric_seq (a r : ℝ) (n : ℕ) : ℝ := a * r^n

-- Define the sequence {log_2 a_n}
def log2_geom_seq (a r : ℝ) (n : ℕ) : ℕ → ℝ :=
  λ n, Real.log 2 (geometric_seq a r n)

-- Main theorem to be proved
theorem sum_log2_geom_seq_first_7_terms (a r : ℝ) (h_pos : ∀ n : ℕ, geometric_seq a r n > 0)
  (h_cond : geometric_seq a r 2 * geometric_seq a r 4 = 4) :
  ∑ i in (Finset.range 7), log2_geom_seq a r i = 7 :=
sorry

end sum_log2_geom_seq_first_7_terms_l58_58976


namespace maximum_rubles_received_l58_58703

theorem maximum_rubles_received : ∃ (n : ℕ), -- There exists a natural number n such that
  (n ≥ 2000 ∧ n < 2100) ∧                -- condition 1: n is between 2000 and 2100
  ((∃ k1, n = 3^3 * 7 * 11 * k1) ∨         -- condition 2: n is of the form 3^3 * 7 * 11 * k1
   (∃ k2, n = 5 * k2)) ∧                  -- or n is of the form 5 * k2
  let payouts :=                          -- definition: distinguish the payouts for divisibility
    (if n % 1 = 0 then 1 else 0) +        -- payout for divisibility by 1
    (if n % 3 = 0 then 3 else 0) +        -- payout for divisibility by 3
    (if n % 5 = 0 then 5 else 0) +        -- payout for divisibility by 5
    (if n % 7 = 0 then 7 else 0) +        -- payout for divisibility by 7
    (if n % 9 = 0 then 9 else 0) +        -- payout for divisibility by 9
    (if n % 11 = 0 then 11 else 0) in     -- payout for divisibility by 11
  payouts = 31                            -- condition 3: sum of payouts equals 31
  ∧ n = 2079 := sorry                   -- answer: n equals 2079

end maximum_rubles_received_l58_58703


namespace lower_rent_amount_l58_58990

-- Define the conditions and proof goal
variable (T R : ℕ)
variable (L : ℕ)

-- Condition 1: Total rent is $1000
def total_rent (T R : ℕ) (L : ℕ) := 60 * R + L * (T - R)

-- Condition 2: Reduction by 20% when 10 rooms are swapped
def reduced_rent (T R : ℕ) (L : ℕ) := 60 * (R - 10) + L * (T - R + 10)

-- Proof that the lower rent amount is $40 given the conditions
theorem lower_rent_amount (h1 : total_rent T R L = 1000)
                         (h2 : reduced_rent T R L = 800) : L = 40 :=
by
  sorry

end lower_rent_amount_l58_58990


namespace last_8_digits_of_product_l58_58213

theorem last_8_digits_of_product :
  let p := 11 * 101 * 1001 * 10001 * 1000001 * 111
  (p % 100000000) = 87654321 :=
by
  let p := 11 * 101 * 1001 * 10001 * 1000001 * 111
  have : p % 100000000 = 87654321 := sorry
  exact this

end last_8_digits_of_product_l58_58213


namespace crayons_problem_l58_58418

theorem crayons_problem
  (S M L : ℕ)
  (hS_condition : (3 / 5 : ℚ) * S = 60)
  (hM_condition : (1 / 4 : ℚ) * M = 98)
  (hL_condition : (4 / 7 : ℚ) * L = 168) :
  S = 100 ∧ M = 392 ∧ L = 294 ∧ ((2 / 5 : ℚ) * S + (3 / 4 : ℚ) * M + (3 / 7 : ℚ) * L = 460) := 
by
  sorry

end crayons_problem_l58_58418


namespace max_value_of_q_l58_58141

theorem max_value_of_q (A M C : ℕ) (h_sum : A + M + C = 15) : 
  A * M * C + A * M + M * C + C * A ≤ 200 :=
sorry

end max_value_of_q_l58_58141


namespace sale_prices_correct_l58_58626

-- Define the cost prices and profit percentages
def cost_price_A : ℕ := 320
def profit_percentage_A : ℕ := 50

def cost_price_B : ℕ := 480
def profit_percentage_B : ℕ := 70

def cost_price_C : ℕ := 600
def profit_percentage_C : ℕ := 40

-- Define the expected sale prices
def sale_price_A : ℕ := 480
def sale_price_B : ℕ := 816
def sale_price_C : ℕ := 840

-- Define a function to compute sale price
def compute_sale_price (cost_price : ℕ) (profit_percentage : ℕ) : ℕ :=
  cost_price + (profit_percentage * cost_price) / 100

-- The proof statement
theorem sale_prices_correct :
  compute_sale_price cost_price_A profit_percentage_A = sale_price_A ∧
  compute_sale_price cost_price_B profit_percentage_B = sale_price_B ∧
  compute_sale_price cost_price_C profit_percentage_C = sale_price_C :=
by {
  sorry
}

end sale_prices_correct_l58_58626


namespace remainder_of_2n_l58_58913

theorem remainder_of_2n (n : ℕ) (h : n % 4 = 3) : (2 * n) % 4 = 2 := 
sorry

end remainder_of_2n_l58_58913


namespace find_k_l58_58669

theorem find_k (k : ℝ) : (1 - 1.5 * k = (k - 2.5) / 3) → k = 1 :=
by
  intro h
  sorry

end find_k_l58_58669


namespace gina_expenditure_l58_58950

noncomputable def gina_total_cost : ℝ :=
  let regular_classes_cost := 12 * 450
  let lab_classes_cost := 6 * 550
  let textbooks_cost := 3 * 150
  let online_resources_cost := 4 * 95
  let facilities_fee := 200
  let lab_fee := 6 * 75
  let total_cost := regular_classes_cost + lab_classes_cost + textbooks_cost + online_resources_cost + facilities_fee + lab_fee
  let scholarship_amount := 0.5 * regular_classes_cost
  let discount_amount := 0.25 * lab_classes_cost
  let adjusted_cost := total_cost - scholarship_amount - discount_amount
  let interest := 0.04 * adjusted_cost
  adjusted_cost + interest

theorem gina_expenditure : gina_total_cost = 5881.20 :=
by
  sorry

end gina_expenditure_l58_58950


namespace max_q_value_l58_58148

theorem max_q_value (A M C : ℕ) (h : A + M + C = 15) : 
  (A * M * C + A * M + M * C + C * A) ≤ 200 :=
sorry

end max_q_value_l58_58148


namespace find_b_and_c_l58_58382

variable (U : Set ℝ) -- Define the universal set U
variable (A : Set ℝ) -- Define the set A
variables (b c : ℝ) -- Variables for coefficients

-- Conditions that U = {2, 3, 5} and A = { x | x^2 + bx + c = 0 }
def cond_universal_set := U = {2, 3, 5}
def cond_set_A := A = { x | x^2 + b * x + c = 0 }

-- Condition for the complement of A w.r.t U being {2}
def cond_complement := (U \ A) = {2}

-- The statement to be proved
theorem find_b_and_c : 
  cond_universal_set U →
  cond_set_A A b c →
  cond_complement U A →
  b = -8 ∧ c = 15 :=
by
  intros
  sorry

end find_b_and_c_l58_58382


namespace sum_of_possible_radii_l58_58761

theorem sum_of_possible_radii :
  ∀ r : ℝ, (r - 4)^2 + r^2 = (r + 2)^2 → r = 6 + 2 * Real.sqrt 6 ∨ r = 6 - 2 * Real.sqrt 6 → (6 + 2 * Real.sqrt 6) + (6 - 2 * Real.sqrt 6) = 12 :=
by
  intros r h tangency condition
  sorry

end sum_of_possible_radii_l58_58761


namespace petya_max_rubles_l58_58698

theorem petya_max_rubles :
  let is_valid_number := (λ n : ℕ, 2000 ≤ n ∧ n < 2100),
      is_divisible := (λ n d : ℕ, d ∣ n),
      rubles := (λ n, if is_divisible n 1 then 1 else 0 +
                       if is_divisible n 3 then 3 else 0 +
                       if is_divisible n 5 then 5 else 0 +
                       if is_divisible n 7 then 7 else 0 +
                       if is_divisible n 9 then 9 else 0 +
                       if is_divisible n 11 then 11 else 0)
  in ∃ n, is_valid_number n ∧ rubles n = 31 := 
sorry

end petya_max_rubles_l58_58698


namespace maximum_rubles_received_l58_58702

theorem maximum_rubles_received : ∃ (n : ℕ), -- There exists a natural number n such that
  (n ≥ 2000 ∧ n < 2100) ∧                -- condition 1: n is between 2000 and 2100
  ((∃ k1, n = 3^3 * 7 * 11 * k1) ∨         -- condition 2: n is of the form 3^3 * 7 * 11 * k1
   (∃ k2, n = 5 * k2)) ∧                  -- or n is of the form 5 * k2
  let payouts :=                          -- definition: distinguish the payouts for divisibility
    (if n % 1 = 0 then 1 else 0) +        -- payout for divisibility by 1
    (if n % 3 = 0 then 3 else 0) +        -- payout for divisibility by 3
    (if n % 5 = 0 then 5 else 0) +        -- payout for divisibility by 5
    (if n % 7 = 0 then 7 else 0) +        -- payout for divisibility by 7
    (if n % 9 = 0 then 9 else 0) +        -- payout for divisibility by 9
    (if n % 11 = 0 then 11 else 0) in     -- payout for divisibility by 11
  payouts = 31                            -- condition 3: sum of payouts equals 31
  ∧ n = 2079 := sorry                   -- answer: n equals 2079

end maximum_rubles_received_l58_58702


namespace bars_not_sold_l58_58503

-- Definitions for the conditions
def cost_per_bar : ℕ := 3
def total_bars : ℕ := 9
def money_made : ℕ := 18

-- The theorem we need to prove
theorem bars_not_sold : total_bars - (money_made / cost_per_bar) = 3 := sorry

end bars_not_sold_l58_58503


namespace square_side_length_equals_nine_l58_58205

-- Definitions based on the conditions
def rectangle_length : ℕ := 10
def rectangle_width : ℕ := 8
def rectangle_perimeter (length width : ℕ) : ℕ := 2 * length + 2 * width
def side_length_of_square (perimeter : ℕ) : ℕ := perimeter / 4

-- The theorem we want to prove
theorem square_side_length_equals_nine : 
  side_length_of_square (rectangle_perimeter rectangle_length rectangle_width) = 9 :=
by
  -- proof goes here
  sorry

end square_side_length_equals_nine_l58_58205


namespace wage_constraint_l58_58899

/-- Wage constraints for hiring carpenters and tilers given a budget -/
theorem wage_constraint (x y : ℕ) (h_carpenter_wage : 50 * x + 40 * y = 2000) : 5 * x + 4 * y = 200 := by
  sorry

end wage_constraint_l58_58899


namespace all_perfect_squares_l58_58793

theorem all_perfect_squares (a b c : ℕ) (h_gcd : Nat.gcd (Nat.gcd a b) c = 1) 
  (h_eq : a ^ 2 + b ^ 2 + c ^ 2 = 2 * (a * b + b * c + c * a)) : 
  ∃ (k l m : ℕ), a = k ^ 2 ∧ b = l ^ 2 ∧ c = m ^ 2 :=
sorry

end all_perfect_squares_l58_58793


namespace frog_climb_time_l58_58477

-- Define the problem as an assertion within Lean.
theorem frog_climb_time 
  (well_depth : ℕ) (climb_up : ℕ) (slide_down : ℕ) (time_per_meter: ℕ) (climb_start_time : ℕ) 
  (time_to_slide_multiplier: ℚ)
  (time_to_second_position: ℕ) 
  (final_distance: ℕ) 
  (total_time: ℕ)
  (h_start : well_depth = 12)
  (h_climb_up: climb_up = 3)
  (h_slide_down : slide_down = 1)
  (h_time_per_meter : time_per_meter = 1)
  (h_time_to_slide_multiplier: time_to_slide_multiplier = 1/3)
  (h_time_to_second_position : climb_start_time = 8 * 60 /\ time_to_second_position = 8 * 60 + 17)
  (h_final_distance : final_distance = 3)
  (h_total_time: total_time = 22) :
  
  ∃ (t: ℕ), 
    t = total_time := 
by
  sorry

end frog_climb_time_l58_58477


namespace total_amount_paid_l58_58304

def p1 := 20
def p2 := p1 + 2
def p3 := p2 + 3
def p4 := p3 + 4

theorem total_amount_paid : p1 + p2 + p3 + p4 = 96 :=
by
  sorry

end total_amount_paid_l58_58304


namespace monogramming_cost_per_stocking_l58_58564

noncomputable def total_stockings : ℕ := (5 * 5) + 4
noncomputable def price_per_stocking : ℝ := 20 - (0.10 * 20)
noncomputable def total_cost_of_stockings : ℝ := total_stockings * price_per_stocking
noncomputable def total_cost : ℝ := 1035
noncomputable def total_monogramming_cost : ℝ := total_cost - total_cost_of_stockings

theorem monogramming_cost_per_stocking :
  (total_monogramming_cost / total_stockings) = 17.69 :=
by
  sorry

end monogramming_cost_per_stocking_l58_58564


namespace nine_wolves_nine_sheep_seven_days_l58_58757

theorem nine_wolves_nine_sheep_seven_days
    (wolves_sheep_seven_days : ∀ {n : ℕ}, 7 * n / 7 = n) :
    9 * 9 / 9 = 7 := by
  sorry

end nine_wolves_nine_sheep_seven_days_l58_58757


namespace coefficient_x3_in_expansion_l58_58551

-- Define the expansion of the binomial expression
def expansion (x : ℝ) : ℝ := (1 + x) * (2 + x)^5

-- Statement to be proved
theorem coefficient_x3_in_expansion : (expansion x).coeff 3 = 120 :=
sorry

end coefficient_x3_in_expansion_l58_58551


namespace prob_rel_prime_is_1721_l58_58590

def is_rel_prime (a b : ℕ) : Prop := Nat.gcd a b = 1
def pairs := {p : ℕ × ℕ // 1 ≤ p.1 ∧ p.1 < p.2 ∧ p.2 ≤ 7}
def non_rel_prime_pairs := {pr : pairs // ¬ is_rel_prime pr.1.1 pr.1.2}

def prob_rel_prime : ℚ :=
(21 - (non_rel_prime_pairs.to_finset.card : ℕ)) / 21

theorem prob_rel_prime_is_1721 : prob_rel_prime = 17 / 21 :=
by
  sorry

end prob_rel_prime_is_1721_l58_58590


namespace ratio_is_one_half_l58_58287

noncomputable def ratio_dresses_with_pockets (D : ℕ) (total_pockets : ℕ) (pockets_two : ℕ) (pockets_three : ℕ) :=
  ∃ (P : ℕ), D = 24 ∧ total_pockets = 32 ∧
  (P / 3) * 2 + (2 * P / 3) * 3 = total_pockets ∧ 
  P / D = 1 / 2

theorem ratio_is_one_half :
  ratio_dresses_with_pockets 24 32 2 3 :=
by 
  sorry

end ratio_is_one_half_l58_58287


namespace man_overtime_hours_correctness_l58_58054

def man_worked_overtime_hours (r h_r t : ℕ): ℕ :=
  let regular_pay := r * h_r
  let overtime_pay := t - regular_pay
  let overtime_rate := 2 * r
  overtime_pay / overtime_rate

theorem man_overtime_hours_correctness : man_worked_overtime_hours 3 40 186 = 11 := by
  sorry

end man_overtime_hours_correctness_l58_58054


namespace zero_plus_one_plus_two_plus_three_not_eq_zero_mul_one_mul_two_mul_three_l58_58440

theorem zero_plus_one_plus_two_plus_three_not_eq_zero_mul_one_mul_two_mul_three :
  (0 + 1 + 2 + 3) ≠ (0 * 1 * 2 * 3) :=
by
  sorry

end zero_plus_one_plus_two_plus_three_not_eq_zero_mul_one_mul_two_mul_three_l58_58440


namespace small_cuboid_length_is_five_l58_58328

-- Define initial conditions
def large_cuboid_length : ℝ := 18
def large_cuboid_width : ℝ := 15
def large_cuboid_height : ℝ := 2
def num_small_cuboids : ℕ := 6
def small_cuboid_width : ℝ := 6
def small_cuboid_height : ℝ := 3

-- Theorem to prove the length of the smaller cuboid
theorem small_cuboid_length_is_five (small_cuboid_length : ℝ) 
  (h1 : large_cuboid_length * large_cuboid_width * large_cuboid_height 
          = num_small_cuboids * (small_cuboid_length * small_cuboid_width * small_cuboid_height)) :
  small_cuboid_length = 5 := by
  sorry

end small_cuboid_length_is_five_l58_58328


namespace sqrt_product_eq_l58_58640

theorem sqrt_product_eq :
  (Int.sqrt (2 ^ 2 * 3 ^ 4) : ℤ) = 18 :=
sorry

end sqrt_product_eq_l58_58640


namespace remainder_addition_l58_58047

theorem remainder_addition (m : ℕ) (k : ℤ) (h : m = 9 * k + 4) : (m + 2025) % 9 = 4 := by
  sorry

end remainder_addition_l58_58047


namespace total_stickers_purchased_l58_58697

-- Definitions for the number of sheets and stickers per sheet for each folder
def num_sheets_per_folder := 10
def stickers_per_sheet_red := 3
def stickers_per_sheet_green := 2
def stickers_per_sheet_blue := 1

-- Theorem stating that the total number of stickers is 60
theorem total_stickers_purchased : 
  num_sheets_per_folder * (stickers_per_sheet_red + stickers_per_sheet_green + stickers_per_sheet_blue) = 60 := 
  by
  -- Skipping the proof
  sorry

end total_stickers_purchased_l58_58697


namespace inverse_matrix_equation_of_line_l_l58_58727

noncomputable def M : Matrix (Fin 2) (Fin 2) ℚ := ![![1, 2], ![3, 4]]
noncomputable def M_inv : Matrix (Fin 2) (Fin 2) ℚ := ![![-2, 1], ![3/2, -1/2]]

theorem inverse_matrix :
  M⁻¹ = M_inv :=
by
  sorry

def transformed_line (x y : ℚ) : Prop := 2 * (x + 2 * y) - (3 * x + 4 * y) = 4 

theorem equation_of_line_l (x y : ℚ) :
  transformed_line x y → x + 4 = 0 :=
by
  sorry

end inverse_matrix_equation_of_line_l_l58_58727


namespace find_smallest_pqrs_sum_l58_58014

variables (p q r s : ℕ) -- We use natural numbers to ensure p, q, r, s are positive.

theorem find_smallest_pqrs_sum :
  let A := !![4, 0; 0, 3],
      B := !![p, q; r, s],
      C := !![24, 16; -30, -19] in
  (A ⬝ B = B ⬝ C → p > 0 ∧ q > 0 ∧ r > 0 ∧ s > 0 → p + q + r + s = 14) :=
sorry

end find_smallest_pqrs_sum_l58_58014


namespace totalOwlsOnFence_l58_58619

-- Define the conditions given in the problem
def initialOwls : Nat := 3
def joinedOwls : Nat := 2

-- Define the total number of owls
def totalOwls : Nat := initialOwls + joinedOwls

-- State the theorem we want to prove
theorem totalOwlsOnFence : totalOwls = 5 := by
  sorry

end totalOwlsOnFence_l58_58619


namespace henry_money_l58_58615

-- Define the conditions
def initial : ℕ := 11
def birthday : ℕ := 18
def spent : ℕ := 10

-- Define the final amount
def final_amount : ℕ := initial + birthday - spent

-- State the theorem
theorem henry_money : final_amount = 19 := by
  -- Skipping the proof
  sorry

end henry_money_l58_58615


namespace minimum_period_of_f_l58_58578

noncomputable def f (x : ℝ) : ℝ := (real.sqrt 3 * real.sin x + real.cos x) * (real.sqrt 3 * real.cos x - real.sin x)

theorem minimum_period_of_f : ∀ (x : ℝ), f(x + π) = f(x) := by
  sorry

end minimum_period_of_f_l58_58578


namespace speed_of_man_in_still_water_l58_58232

variable (v_m v_s : ℝ)

theorem speed_of_man_in_still_water
  (h1 : (v_m + v_s) * 4 = 24)
  (h2 : (v_m - v_s) * 5 = 20) :
  v_m = 5 := 
sorry

end speed_of_man_in_still_water_l58_58232


namespace find_center_of_ellipse_l58_58958

-- Defining the equation of the ellipse
def ellipse (x y : ℝ) : Prop := 2*x^2 + 2*x*y + y^2 + 2*x + 2*y - 4 = 0

-- The coordinates of the center
def center_of_ellipse : ℝ × ℝ := (0, -1)

-- The theorem asserting the center of the ellipse
theorem find_center_of_ellipse (x y : ℝ) (h : ellipse x y) : (x, y) = center_of_ellipse :=
sorry

end find_center_of_ellipse_l58_58958


namespace find_first_offset_l58_58787

theorem find_first_offset 
  (area : ℝ) (diagonal : ℝ) (offset2 : ℝ) (offset1 : ℝ) 
  (h_area : area = 210) 
  (h_diagonal : diagonal = 28)
  (h_offset2 : offset2 = 6) :
  offset1 = 9 :=
by
  sorry

end find_first_offset_l58_58787


namespace find_x_plus_y_l58_58524

theorem find_x_plus_y (x y : ℝ) (h1 : x + Real.cos y = 2010) (h2 : x + 2010 * Real.sin y = 2009) (hy : 0 ≤ y ∧ y ≤ Real.pi / 2) :
  x + y = 2009 :=
by
  sorry

end find_x_plus_y_l58_58524


namespace inv_mod_35_l58_58511

theorem inv_mod_35 : ∃ x : ℕ, 5 * x ≡ 1 [MOD 35] :=
by
  use 29
  sorry

end inv_mod_35_l58_58511


namespace sum_of_A_and_B_zero_l58_58407

theorem sum_of_A_and_B_zero
  (A B C : ℝ)
  (h1 : A ≠ B)
  (h2 : C ≠ 0)
  (f g : ℝ → ℝ)
  (h3 : ∀ x, f x = A * x + B + C)
  (h4 : ∀ x, g x = B * x + A - C)
  (h5 : ∀ x, f (g x) - g (f x) = 2 * C) : A + B = 0 :=
sorry

end sum_of_A_and_B_zero_l58_58407


namespace necessary_and_sufficient_condition_l58_58952

noncomputable def f (a x : ℝ) : ℝ := a * x - x^2

theorem necessary_and_sufficient_condition (a : ℝ) (h : 0 < a) :
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → f a x ≤ 1) ↔ (0 < a ∧ a ≤ 2) := by
  sorry

end necessary_and_sufficient_condition_l58_58952


namespace parallelogram_area_l58_58581

theorem parallelogram_area
  (a b : ℕ)
  (h1 : a + b = 15)
  (h2 : 2 * a = 3 * b) :
  2 * a = 18 :=
by
  -- Proof is omitted; the statement shows what needs to be proven
  sorry

end parallelogram_area_l58_58581


namespace stacy_grew_more_l58_58184

variable (initial_height_stacy current_height_stacy brother_growth stacy_growth_more : ℕ)

-- Conditions
def stacy_initial_height : initial_height_stacy = 50 := by sorry
def stacy_current_height : current_height_stacy = 57 := by sorry
def brother_growth_last_year : brother_growth = 1 := by sorry

-- Compute Stacy's growth
def stacy_growth : ℕ := current_height_stacy - initial_height_stacy

-- Prove the difference in growth
theorem stacy_grew_more :
  stacy_growth - brother_growth = stacy_growth_more → stacy_growth_more = 6 := 
by sorry

end stacy_grew_more_l58_58184


namespace tank_capacity_l58_58911

noncomputable def leak_rate (C : ℝ) := C / 6
noncomputable def inlet_rate := 240
noncomputable def net_emptying_rate (C : ℝ) := C / 8

theorem tank_capacity : ∀ (C : ℝ), 
  (inlet_rate - leak_rate C = net_emptying_rate C) → 
  C = 5760 / 7 :=
by 
  sorry

end tank_capacity_l58_58911


namespace find_value_less_than_twice_l58_58417

def value_less_than_twice_another (x y v : ℕ) : Prop :=
  y = 2 * x - v ∧ x + y = 51 ∧ y = 33

theorem find_value_less_than_twice (x y v : ℕ) (h : value_less_than_twice_another x y v) : v = 3 := by
  sorry

end find_value_less_than_twice_l58_58417


namespace eunji_class_total_students_l58_58490

variable (A B : Finset ℕ) (universe_students : Finset ℕ)

axiom students_play_instrument_a : A.card = 24
axiom students_play_instrument_b : B.card = 17
axiom students_play_both_instruments : (A ∩ B).card = 8
axiom no_students_without_instruments : A ∪ B = universe_students

theorem eunji_class_total_students : universe_students.card = 33 := by
  sorry

end eunji_class_total_students_l58_58490


namespace digits_1_left_of_6_count_l58_58999

theorem digits_1_left_of_6_count :
  let digits := {1, 2, 3, 4, 5, 6}
  let is_six_digit_unique (n : ℕ) : Prop := 
    ∃ l : List ℕ, l.nodup ∧ (l.perm digits.to_list) ∧ (1 ∈ l) ∧ (6 ∈ l) ∧
    (n = l.foldl (λ acc d, acc * 10 + d) 0)
  let count_1_left_of_6 (l : List ℕ) : Prop := list_index l 1 < list_index l 6
  (number_of_six_digit_integers_with_property digits is_six_digit_unique count_1_left_of_6 = 360) :=
begin
  sorry
end

end digits_1_left_of_6_count_l58_58999


namespace percent_both_correct_proof_l58_58468

-- Define the problem parameters
def totalTestTakers := 100
def percentFirstCorrect := 80
def percentSecondCorrect := 75
def percentNeitherCorrect := 5

-- Define the target proof statement
theorem percent_both_correct_proof :
  percentFirstCorrect + percentSecondCorrect - percentFirstCorrect + percentNeitherCorrect = 60 := 
by 
  sorry

end percent_both_correct_proof_l58_58468


namespace find_complex_number_l58_58647

theorem find_complex_number (z : ℂ) (h1 : complex.abs (z - 2) = complex.abs (z + 4))
  (h2 : complex.abs (z + 4) = complex.abs (z - 2 * complex.I)) : z = -1 + complex.I :=
by
  sorry

end find_complex_number_l58_58647


namespace find_constant_t_l58_58943

theorem find_constant_t : ∃ t : ℝ, 
  (∀ x : ℝ, (3 * x^2 - 4 * x + 5) * (2 * x^2 + t * x + 8) = 6 * x^4 + (-26) * x^3 + 58 * x^2 + (-76) * x + 40) ↔ t = -6 :=
by {
  sorry
}

end find_constant_t_l58_58943


namespace man_l58_58763

theorem man's_speed_with_current (v : ℝ) (current_speed : ℝ) (against_current_speed : ℝ) 
  (h_current_speed : current_speed = 5) (h_against_current_speed : against_current_speed = 12) 
  (h_v : v - current_speed = against_current_speed) : 
  v + current_speed = 22 := 
by
  sorry

end man_l58_58763


namespace range_of_a_l58_58755

variable (a : ℝ)
def A (a : ℝ) : Set ℝ := {x | -2 - a < x ∧ x < a ∧ a > 0}

def p (a : ℝ) := 1 ∈ A a
def q (a : ℝ) := 2 ∈ A a

theorem range_of_a (h1 : p a ∨ q a) (h2 : ¬ (p a ∧ q a)) : 1 < a ∧ a ≤ 2 := sorry

end range_of_a_l58_58755


namespace highest_score_batsman_l58_58223

variable (H L : ℕ)

theorem highest_score_batsman :
  (60 * 46) = (58 * 44 + H + L) ∧ (H - L = 190) → H = 199 :=
by
  intros h
  sorry

end highest_score_batsman_l58_58223


namespace proof_problem_l58_58694

noncomputable def f (x y k : ℝ) : ℝ := k * x + (1 / y)

theorem proof_problem
  (a b k : ℝ) (h1 : f a b k = f b a k) (h2 : a ≠ b) :
  f (a * b) 1 k = 0 :=
sorry

end proof_problem_l58_58694


namespace production_line_improvement_better_than_financial_investment_l58_58475

noncomputable def improved_mean_rating (initial_mean : ℝ) := initial_mean + 0.05

noncomputable def combined_mean_rating (mean_unimproved : ℝ) (mean_improved : ℝ) : ℝ :=
  (mean_unimproved * 200 + mean_improved * 200) / 400

noncomputable def combined_variance (variance : ℝ) (combined_mean : ℝ) : ℝ :=
  (2 * variance) - combined_mean ^ 2

noncomputable def increased_returns (grade_a_price : ℝ) (grade_b_price : ℝ) 
  (proportion_upgraded : ℝ) (units_per_day : ℕ) (days_per_year : ℕ) : ℝ :=
  (grade_a_price - grade_b_price) * proportion_upgraded * units_per_day * days_per_year - 200000000

noncomputable def financial_returns (initial_investment : ℝ) (annual_return_rate : ℝ) : ℝ :=
  initial_investment * (1 + annual_return_rate) - initial_investment

theorem production_line_improvement_better_than_financial_investment 
  (initial_mean : ℝ := 9.98) 
  (initial_variance : ℝ := 0.045) 
  (grade_a_price : ℝ := 2000) 
  (grade_b_price : ℝ := 1200) 
  (proportion_upgraded : ℝ := 3 / 8) 
  (units_per_day : ℕ := 200) 
  (days_per_year : ℕ := 365) 
  (initial_investment : ℝ := 200000000) 
  (annual_return_rate : ℝ := 0.082) : 
  combined_mean_rating initial_mean (improved_mean_rating initial_mean) = 10.005 ∧ 
  combined_variance initial_variance (combined_mean_rating initial_mean (improved_mean_rating initial_mean)) = 0.045625 ∧ 
  increased_returns grade_a_price grade_b_price proportion_upgraded units_per_day days_per_year > financial_returns initial_investment annual_return_rate := 
by {
  sorry
}

end production_line_improvement_better_than_financial_investment_l58_58475


namespace initial_amount_invested_l58_58089

-- Conditions
def initial_investment : ℝ := 367.36
def annual_interest_rate : ℝ := 0.08
def accumulated_amount : ℝ := 500
def years : ℕ := 4

-- Required to prove that the initial investment satisfies the given equation
theorem initial_amount_invested :
  initial_investment * (1 + annual_interest_rate) ^ years = accumulated_amount :=
by
  sorry

end initial_amount_invested_l58_58089


namespace positive_number_sum_square_l58_58448

theorem positive_number_sum_square (n : ℕ) (h : n^2 + n = 210) : n = 14 :=
sorry

end positive_number_sum_square_l58_58448


namespace smaller_cube_volume_l58_58928

theorem smaller_cube_volume
  (d : ℝ) (s : ℝ) (V : ℝ)
  (h1 : d = 12)  -- condition: diameter of the sphere equals the edge length of the larger cube
  (h2 : d = s * Real.sqrt 3)  -- condition: space diagonal of the smaller cube equals the diameter of the sphere
  (h3 : s = 12 / Real.sqrt 3)  -- condition: side length of the smaller cube
  (h4 : V = s^3)  -- condition: volume of the cube with side length s
  : V = 192 * Real.sqrt 3 :=  -- proving the volume of the smaller cube
sorry

end smaller_cube_volume_l58_58928


namespace depth_below_sea_notation_l58_58832

variables (alt_above_sea : ℝ) (depth_below_sea : ℝ)

def notation_above_sea : ℝ := alt_above_sea

def notation_below_sea (d : ℝ) : ℝ := -d

theorem depth_below_sea_notation : alt_above_sea = 9050 → notation_above_sea = 9050 → depth_below_sea = 10907 → notation_below_sea depth_below_sea = -10907 :=
by
  intros h1 h2 h3
  rw [h3, notation_below_sea]
  exact eq.symm sorry

end depth_below_sea_notation_l58_58832


namespace num_teachers_l58_58318

-- This statement involves defining the given conditions and stating the theorem to be proved.
theorem num_teachers (parents students total_people : ℕ) (h_parents : parents = 73) (h_students : students = 724) (h_total : total_people = 1541) :
  total_people - (parents + students) = 744 :=
by
  -- Including sorry to skip the proof, as required.
  sorry

end num_teachers_l58_58318


namespace fraction_ordering_l58_58677

theorem fraction_ordering (x y : ℝ) (hx : x < 0) (hy : 0 < y ∧ y < 1) :
  (1 / x) < (y / x) ∧ (y / x) < (y^2 / x) :=
by
  sorry

end fraction_ordering_l58_58677


namespace isosceles_triangle_vertex_angle_l58_58798

noncomputable def vertex_angle_of_isosceles (a b : ℝ) : ℝ :=
  if a = b then 40 else 100

theorem isosceles_triangle_vertex_angle (a : ℝ) (interior_angle : ℝ)
  (h_isosceles : a = 40 ∨ a = interior_angle ∧ interior_angle = 40 ∨ interior_angle = 100) :
  vertex_angle_of_isosceles a interior_angle = 40 ∨ vertex_angle_of_isosceles a interior_angle = 100 := 
by
  sorry

end isosceles_triangle_vertex_angle_l58_58798


namespace cos_2beta_proof_l58_58370

theorem cos_2beta_proof (α β : ℝ)
  (h1 : Real.sin (α - β) = 3 / 5)
  (h2 : Real.sin (α + β) = -3 / 5)
  (h3 : α - β ∈ Set.Ioo (π / 2) π)
  (h4 : α + β ∈ Set.Ioo (3 * π / 2) (2 * π)) :
  Real.cos (2 * β) = -7 / 25 :=
by
  sorry

end cos_2beta_proof_l58_58370


namespace inequality_proof_l58_58567

variable (a b c : ℝ)

#check (0 < a ∧ 0 < b ∧ 0 < c ∧ abc * (a + b + c) = ab + bc + ca) →
  5 * (a + b + c) ≥ 7 + 8 * abc

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (h : abc * (a + b + c) = ab + bc + ca) : 
  5 * (a + b + c) ≥ 7 + 8 * abc :=
sorry

end inequality_proof_l58_58567


namespace max_q_value_l58_58145

theorem max_q_value (A M C : ℕ) (h : A + M + C = 15) : 
  (A * M * C + A * M + M * C + C * A) ≤ 200 :=
sorry

end max_q_value_l58_58145


namespace courtyard_length_is_60_l58_58384

noncomputable def stone_length : ℝ := 2.5
noncomputable def stone_breadth : ℝ := 2.0
noncomputable def num_stones : ℕ := 198
noncomputable def courtyard_breadth : ℝ := 16.5

theorem courtyard_length_is_60 :
  ∃ (courtyard_length : ℝ), courtyard_length = 60 ∧
  num_stones * (stone_length * stone_breadth) = courtyard_length * courtyard_breadth :=
sorry

end courtyard_length_is_60_l58_58384


namespace Sam_drinks_l58_58253

theorem Sam_drinks (juice_don : ℚ) (fraction_sam : ℚ) 
  (h1 : juice_don = 3 / 7) (h2 : fraction_sam = 4 / 5) : 
  (fraction_sam * juice_don = 12 / 35) :=
by
  sorry

end Sam_drinks_l58_58253


namespace problem1_solution_problem2_solution_l58_58302

theorem problem1_solution (x : ℝ) : x^2 - x - 6 > 0 ↔ x < -2 ∨ x > 3 := sorry

theorem problem2_solution (x : ℝ) : -2*x^2 + x + 1 < 0 ↔ x < -1/2 ∨ x > 1 := sorry

end problem1_solution_problem2_solution_l58_58302


namespace blake_change_l58_58634

theorem blake_change :
  ∃ (change : ℕ), 
    let lollipop_cost := 2 in
    let lollipops := 4 in
    let chocolate_cost := lollipop_cost * 4 in
    let chocolates := 6 in
    let total_cost := (lollipop_cost * lollipops) + (chocolate_cost * chocolates) in
    let amount_given := 10 * 6 in
    change = amount_given - total_cost := 
by
  use 4
  sorry

end blake_change_l58_58634


namespace lcm_of_54_96_120_150_l58_58358

theorem lcm_of_54_96_120_150 : Nat.lcm 54 (Nat.lcm 96 (Nat.lcm 120 150)) = 21600 := by
  sorry

end lcm_of_54_96_120_150_l58_58358


namespace below_sea_level_notation_l58_58843

theorem below_sea_level_notation (depth_above_sea_level : Int) (depth_below_sea_level: Int) 
  (h: depth_above_sea_level = 9050 ∧ depth_below_sea_level = -10907) : 
  depth_below_sea_level = -10907 :=
by 
  sorry

end below_sea_level_notation_l58_58843


namespace find_multiple_l58_58624

-- Definitions of the divisor, original number, and remainders given in the problem conditions.
def D : ℕ := 367
def remainder₁ : ℕ := 241
def remainder₂ : ℕ := 115

-- Statement of the problem.
theorem find_multiple (N m k l : ℕ) :
  (N = k * D + remainder₁) →
  (m * N = l * D + remainder₂) →
  ∃ m, m > 0 ∧ 241 * m - 115 % 367 = 0 ∧ m = 2 :=
by
  sorry

end find_multiple_l58_58624


namespace tan_eq_example_l58_58371

theorem tan_eq_example (x : ℝ) (hx : Real.tan (3 * x) * Real.tan (5 * x) = Real.tan (7 * x) * Real.tan (9 * x)) : x = 30 * Real.pi / 180 :=
  sorry

end tan_eq_example_l58_58371


namespace total_red_cards_l58_58335

def num_standard_decks : ℕ := 3
def num_special_decks : ℕ := 2
def num_custom_decks : ℕ := 2
def red_cards_standard_deck : ℕ := 26
def red_cards_special_deck : ℕ := 30
def red_cards_custom_deck : ℕ := 20

theorem total_red_cards : num_standard_decks * red_cards_standard_deck +
                          num_special_decks * red_cards_special_deck +
                          num_custom_decks * red_cards_custom_deck = 178 :=
by
  -- Calculation omitted
  sorry

end total_red_cards_l58_58335


namespace inequality_of_abc_l58_58566

variable {a b c : ℝ}

theorem inequality_of_abc 
    (h : 0 < a ∧ 0 < b ∧ 0 < c)
    (h₁ : abc * (a + b + c) = ab + bc + ca) :
    5 * (a + b + c) ≥ 7 + 8 * abc :=
sorry

end inequality_of_abc_l58_58566


namespace problem1_problem2_l58_58266

theorem problem1 (m : ℝ) (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = m - |x - 2|) 
  (h2 : ∀ x, f (x + 2) ≥ 0 → -1 ≤ x ∧ x ≤ 1) : 
  m = 1 := 
sorry

theorem problem2 (a b c : ℝ) 
  (h : 1 / a + 1 / (2 * b) + 1 / (3 * c) = 1) : 
  a + 2 * b + 3 * c ≥ 9 := 
sorry

end problem1_problem2_l58_58266


namespace number_of_good_numbers_lt_1000_l58_58815

def is_good_number (n : ℕ) : Prop :=
  let sum := n + (n + 1) + (n + 2)
  sum % 10 < 10 ∧
  (sum / 10) % 10 < 10 ∧
  (sum / 100) % 10 < 10 ∧
  (sum < 1000)

theorem number_of_good_numbers_lt_1000 : ∃ n : ℕ, n = 48 ∧
  (forall k, k < 1000 → k < 1000 → is_good_number k → k = 48) := sorry

end number_of_good_numbers_lt_1000_l58_58815


namespace reflected_ray_bisects_circle_circumference_l58_58365

open Real

noncomputable def equation_of_line_reflected_ray : Prop :=
  ∃ (m b : ℝ), (m = 2 / (-3 + 1)) ∧ (b = (3/(-5 + 5)) + 1) ∧ ((-5, -3) = (-5, (-5*m + b))) ∧ ((1, 1) = (1, (1*m + b)))

theorem reflected_ray_bisects_circle_circumference :
  equation_of_line_reflected_ray ↔ ∃ a b c : ℝ, (a = 2) ∧ (b = -3) ∧ (c = 1) ∧ (a*x + b*y + c = 0) :=
by
  sorry

end reflected_ray_bisects_circle_circumference_l58_58365


namespace weight_of_A_l58_58308

theorem weight_of_A
  (W_A W_B W_C W_D W_E : ℕ)
  (H_A H_B H_C H_D : ℕ)
  (Age_A Age_B Age_C Age_D : ℕ)
  (hw1 : (W_A + W_B + W_C) / 3 = 84)
  (hh1 : (H_A + H_B + H_C) / 3 = 170)
  (ha1 : (Age_A + Age_B + Age_C) / 3 = 30)
  (hw2 : (W_A + W_B + W_C + W_D) / 4 = 80)
  (hh2 : (H_A + H_B + H_C + H_D) / 4 = 172)
  (ha2 : (Age_A + Age_B + Age_C + Age_D) / 4 = 28)
  (hw3 : (W_B + W_C + W_D + W_E) / 4 = 79)
  (hh3 : (H_B + H_C + H_D + H_E) / 4 = 173)
  (ha3 : (Age_B + Age_C + Age_D + (Age_A - 3)) / 4 = 27)
  (hw4 : W_E = W_D + 7)
  : W_A = 79 := 
sorry

end weight_of_A_l58_58308


namespace max_value_y_l58_58040

noncomputable def y (x : ℝ) : ℝ := x * (3 - 2 * x)

theorem max_value_y : ∃ x, 0 < x ∧ x < (3:ℝ) / 2 ∧ y x = 9 / 8 :=
by
  sorry

end max_value_y_l58_58040


namespace positive_number_sum_square_l58_58442

theorem positive_number_sum_square (n : ℝ) (h : n^2 + n = 210) (h_pos : n > 0) : n = 14 :=
sorry

end positive_number_sum_square_l58_58442
