import Mathlib

namespace alice_weekly_walk_distance_l202_202006

theorem alice_weekly_walk_distance :
  let miles_to_school_per_day := 10
  let miles_home_per_day := 12
  let days_per_week := 5
  let weekly_total_miles := (miles_to_school_per_day * days_per_week) + (miles_home_per_day * days_per_week)
  weekly_total_miles = 110 :=
by
  sorry

end alice_weekly_walk_distance_l202_202006


namespace orange_balls_count_l202_202562

theorem orange_balls_count :
  ∀ (total red blue orange pink : ℕ), 
  total = 50 → red = 20 → blue = 10 → 
  total = red + blue + orange + pink → 3 * orange = pink → 
  orange = 5 :=
by
  intros total red blue orange pink h_total h_red h_blue h_total_eq h_ratio
  sorry

end orange_balls_count_l202_202562


namespace intersection_of_A_and_B_l202_202511

def A : Set ℝ := {y | y > 1}
def B : Set ℝ := {x | Real.log x ≥ 0}
def Intersect : Set ℝ := {x | x > 1}

theorem intersection_of_A_and_B : A ∩ B = Intersect :=
by
  sorry

end intersection_of_A_and_B_l202_202511


namespace minimum_time_needed_l202_202112

-- Define the task times
def review_time : ℕ := 30
def rest_time : ℕ := 30
def boil_water_time : ℕ := 15
def homework_time : ℕ := 25

-- Define the minimum time required (Xiao Ming can boil water while resting)
theorem minimum_time_needed : review_time + rest_time + homework_time = 85 := by
  -- The proof is omitted with sorry
  sorry

end minimum_time_needed_l202_202112


namespace non_raining_hours_l202_202104

-- Definitions based on the conditions.
def total_hours := 9
def rained_hours := 4

-- Problem statement: Prove that the non-raining hours equals to 5 given total_hours and rained_hours.
theorem non_raining_hours : (total_hours - rained_hours = 5) :=
by
  -- The proof is omitted with "sorry" to indicate the missing proof.
  sorry

end non_raining_hours_l202_202104


namespace total_volume_of_water_l202_202883

-- Define the conditions
def volume_of_hemisphere : ℕ := 4
def number_of_hemispheres : ℕ := 2734

-- Define the total volume
def total_volume : ℕ := volume_of_hemisphere * number_of_hemispheres

-- State the theorem
theorem total_volume_of_water : total_volume = 10936 :=
by
  -- Proof placeholder
  sorry

end total_volume_of_water_l202_202883


namespace convex_k_gons_count_l202_202995

noncomputable def number_of_convex_k_gons (n k : ℕ) : ℕ :=
  if h : n ≥ 2 * k then
    n * Nat.factorial (n - k - 1) / (k * Nat.factorial k * Nat.factorial (n - 2 * k))
  else
    0

theorem convex_k_gons_count (n k : ℕ) (h : n ≥ 2 * k) :
  number_of_convex_k_gons n k = n * Nat.factorial (n - k - 1) / (k * Nat.factorial k * Nat.factorial (n - 2 * k)) :=
by
  sorry

end convex_k_gons_count_l202_202995


namespace solve_for_x_l202_202549

theorem solve_for_x (x : ℝ) (h : 3^(3 * x - 2) = (1 : ℝ) / 27) : x = -(1 : ℝ) / 3 :=
sorry

end solve_for_x_l202_202549


namespace arithmetic_sequence_formula_geometric_sequence_formula_sum_of_sequence_l202_202862

theorem arithmetic_sequence_formula (a : ℕ → ℕ) (d : ℕ) (h1 : d > 0) 
  (h2 : a 1 + a 4 + a 7 = 12) (h3 : a 1 * a 4 * a 7 = 28) :
  ∀ n, a n = n :=
sorry

theorem geometric_sequence_formula (b : ℕ → ℕ) (a : ℕ → ℕ) 
  (h1 : b 1 = 16) (h2 : a 2 * b 2 = 4) :
  ∀ n, b n = 2^(n + 3) :=
sorry

theorem sum_of_sequence (a : ℕ → ℕ) (b : ℕ → ℕ) (c : ℕ → ℕ) (T : ℕ → ℕ)
  (h1 : ∀ n, a n = n) (h2 : ∀ n, b n = 2^(n + 3)) 
  (h3 : ∀ n, c n = a n * b n) :
  ∀ n, T n = 8 * (2^n * (n + 1) - 1) :=
sorry

end arithmetic_sequence_formula_geometric_sequence_formula_sum_of_sequence_l202_202862


namespace triangle_interior_angles_l202_202929

theorem triangle_interior_angles (E1 E2 E3 : ℝ) (I1 I2 I3 : ℝ) (x : ℝ)
  (h1 : E1 = 12 * x) 
  (h2 : E2 = 13 * x) 
  (h3 : E3 = 15 * x)
  (h4 : E1 + E2 + E3 = 360) 
  (h5 : I1 = 180 - E1) 
  (h6 : I2 = 180 - E2) 
  (h7 : I3 = 180 - E3) :
  I1 = 72 ∧ I2 = 63 ∧ I3 = 45 :=
by
  sorry

end triangle_interior_angles_l202_202929


namespace no_such_function_l202_202790

theorem no_such_function :
  ¬ (∃ f : ℕ → ℕ, ∀ n ≥ 2, f (f (n - 1)) = f (n + 1) - f (n)) :=
sorry

end no_such_function_l202_202790


namespace kasun_family_children_count_l202_202867

theorem kasun_family_children_count 
    (m : ℝ) (x : ℕ) (y : ℝ)
    (h1 : (m + 50 + x * y + 10) / (3 + x) = 22)
    (h2 : (m + x * y + 10) / (2 + x) = 18) :
    x = 5 :=
by
  sorry

end kasun_family_children_count_l202_202867


namespace expand_expression_l202_202496

variable {R : Type} [CommRing R]
variables (x y : R)

theorem expand_expression :
  5 * (3 * x^3 - 4 * x * y + x^2 - y^2) = 15 * x^3 - 20 * x * y + 5 * x^2 - 5 * y^2 :=
by
  sorry

end expand_expression_l202_202496


namespace ellipse_x1_x2_squared_sum_eq_4_l202_202301

theorem ellipse_x1_x2_squared_sum_eq_4
  (x₁ y₁ x₂ y₂ : ℝ)
  (a b : ℝ)
  (ha : a = 2)
  (hb : b = 1)
  (hM : x₁^2 / a^2 + y₁^2 = 1)
  (hN : x₂^2 / a^2 + y₂^2 = 1)
  (h_slope_product : (y₁ / x₁) * (y₂ / x₂) = -1 / 4) :
  x₁^2 + x₂^2 = 4 :=
by
  sorry

end ellipse_x1_x2_squared_sum_eq_4_l202_202301


namespace find_volume_from_vessel_c_l202_202497

noncomputable def concentration_vessel_a : ℝ := 0.45
noncomputable def concentration_vessel_b : ℝ := 0.30
noncomputable def concentration_vessel_c : ℝ := 0.10
noncomputable def volume_vessel_a : ℝ := 4
noncomputable def volume_vessel_b : ℝ := 5
noncomputable def resultant_concentration : ℝ := 0.26

theorem find_volume_from_vessel_c (x : ℝ) : 
    concentration_vessel_a * volume_vessel_a + concentration_vessel_b * volume_vessel_b + concentration_vessel_c * x = 
    resultant_concentration * (volume_vessel_a + volume_vessel_b + x) → 
    x = 6 :=
by
  sorry

end find_volume_from_vessel_c_l202_202497


namespace min_area_of_B_l202_202670

noncomputable def setA := { p : ℝ × ℝ | abs (p.1 - 2) + abs (p.2 - 3) ≤ 1 }

noncomputable def setB (D E F : ℝ) := { p : ℝ × ℝ | p.1^2 + p.2^2 + D * p.1 + E * p.2 + F ≤ 0 ∧ D^2 + E^2 - 4 * F > 0 }

theorem min_area_of_B (D E F : ℝ) (h : setA ⊆ setB D E F) : 
  ∃ r : ℝ, (∀ p ∈ setB D E F, p.1^2 + p.2^2 ≤ r^2) ∧ (π * r^2 = 2 * π) :=
sorry

end min_area_of_B_l202_202670


namespace value_of_x_plus_y_l202_202349

theorem value_of_x_plus_y (x y : ℝ) (h1 : 1 / x + 1 / y = 4) (h2 : 1 / x - 1 / y = -6) : x + y = -4 / 5 :=
sorry

end value_of_x_plus_y_l202_202349


namespace choose_starters_with_twins_l202_202667

theorem choose_starters_with_twins :
  let total_players := 12
  let num_starters := 5
  let twins_num := 2
  let total_ways := Nat.choose total_players num_starters
  let without_twins := Nat.choose (total_players - twins_num) num_starters
  total_ways - without_twins = 540 := 
by
  let total_players := 12
  let num_starters := 5
  let twins_num := 2
  let total_ways := Nat.choose total_players num_starters
  let without_twins := Nat.choose (total_players - twins_num) num_starters
  exact Nat.sub_eq_of_eq_add sorry -- here we will need the exact proof steps which we skip

end choose_starters_with_twins_l202_202667


namespace lateral_area_of_given_cone_l202_202864

noncomputable def lateral_area_cone (r h : ℝ) : ℝ :=
  let l := Real.sqrt (r^2 + h^2)
  (Real.pi * r * l)

theorem lateral_area_of_given_cone :
  lateral_area_cone 3 4 = 15 * Real.pi :=
by
  -- sorry to skip the proof
  sorry

end lateral_area_of_given_cone_l202_202864


namespace tetrahedron_cube_volume_ratio_l202_202880

theorem tetrahedron_cube_volume_ratio (a : ℝ) :
  let V_tetrahedron := (a * Real.sqrt 2)^3 * Real.sqrt 2 / 12
  let V_cube := a^3
  (V_tetrahedron / V_cube) = 1 / 3 :=
by
  sorry

end tetrahedron_cube_volume_ratio_l202_202880


namespace asha_wins_probability_l202_202226

variable (p_lose p_tie p_win : ℚ)

theorem asha_wins_probability 
  (h_lose : p_lose = 3 / 7) 
  (h_tie : p_tie = 1 / 7) 
  (h_total : p_win + p_lose + p_tie = 1) : 
  p_win = 3 / 7 := by
  sorry

end asha_wins_probability_l202_202226


namespace smallest_n_condition_l202_202781

theorem smallest_n_condition (n : ℕ) : (4 * n) ∣ (n^2) ∧ (5 * n) ∣ (u^3) → n = 100 :=
by
  sorry

end smallest_n_condition_l202_202781


namespace a_n_geometric_sequence_b_n_general_term_l202_202600

theorem a_n_geometric_sequence (t : ℝ) (h : t ≠ 0 ∧ t ≠ 1) :
  (∀ n, ∃ r : ℝ, a_n = t^n) :=
sorry

theorem b_n_general_term (t : ℝ) (h1 : t ≠ 0 ∧ t ≠ 1) (h2 : ∀ n, a_n = t^n)
  (h3 : ∃ q : ℝ, q = (2 * t^2 + t) / 2) :
  (∀ n, b_n = (t^(n + 1) * (2 * t + 1)^(n - 1)) / 2^(n - 2)) :=
sorry

end a_n_geometric_sequence_b_n_general_term_l202_202600


namespace evaluate_star_property_l202_202165

noncomputable def star (a b : ℕ) : ℕ := b ^ a

theorem evaluate_star_property (a b c m : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hm : 0 < m) :
  (star a b ≠ star b a) ∧
  (star a (star b c) ≠ star (star a b) c) ∧
  (star a (b ^ m) ≠ star (star a m) b) ∧
  ((star a b) ^ m ≠ star a (m * b)) :=
by
  sorry

end evaluate_star_property_l202_202165


namespace letter_puzzle_l202_202704

theorem letter_puzzle (A B : ℕ) (hA : 1 ≤ A ∧ A ≤ 9) (hB : 1 ≤ B ∧ B ≤ 9) (h_diff : A ≠ B) :
  A^B = 10 * B + A ↔ (A = 2 ∧ B = 5) ∨ (A = 6 ∧ B = 2) ∨ (A = 4 ∧ B = 3) :=
by
  sorry

end letter_puzzle_l202_202704


namespace range_of_u_l202_202405

variable (a b u : ℝ)

theorem range_of_u (h1 : a > 0) (h2 : b > 0) (h3 : (1 / a) + (9 / b) = 1) : u ≤ 16 :=
by
  sorry

end range_of_u_l202_202405


namespace intervals_of_increase_l202_202420

def f (x : ℝ) : ℝ := 2*x^3 - 6*x^2 + 7

theorem intervals_of_increase : 
  ∀ x : ℝ, (x < 0 ∨ x > 2) → (6*x^2 - 12*x > 0) :=
by
  -- Placeholder for proof
  sorry

end intervals_of_increase_l202_202420


namespace joey_total_study_time_l202_202302

def hours_weekdays (hours_per_night : Nat) (nights_per_week : Nat) : Nat :=
  hours_per_night * nights_per_week

def hours_weekends (hours_per_day : Nat) (days_per_weekend : Nat) : Nat :=
  hours_per_day * days_per_weekend

def total_weekly_study_time (weekday_hours : Nat) (weekend_hours : Nat) : Nat :=
  weekday_hours + weekend_hours

def total_study_time_in_weeks (weekly_hours : Nat) (weeks : Nat) : Nat :=
  weekly_hours * weeks

theorem joey_total_study_time :
  let hours_per_night := 2
  let nights_per_week := 5
  let hours_per_day := 3
  let days_per_weekend := 2
  let weeks := 6
  hours_weekdays hours_per_night nights_per_week +
  hours_weekends hours_per_day days_per_weekend = 16 →
  total_study_time_in_weeks 16 weeks = 96 :=
by 
  intros h1 h2 h3 h4 h5
  have weekday_hours := hours_weekdays h1 h2
  have weekend_hours := hours_weekends h3 h4
  have total_weekly := total_weekly_study_time weekday_hours weekend_hours
  sorry

end joey_total_study_time_l202_202302


namespace volume_between_spheres_l202_202327

theorem volume_between_spheres (r_small r_large : ℝ) (h_small : r_small = 4) (h_large : r_large = 8) : 
  (4 / 3) * Real.pi * (r_large ^ 3) - (4 / 3) * Real.pi * (r_small ^ 3) = (1792 / 3) * Real.pi := 
by
  rw [h_small, h_large]
  sorry

end volume_between_spheres_l202_202327


namespace multiple_of_six_and_nine_l202_202271

-- Definitions: x is a multiple of 6, y is a multiple of 9.
def is_multiple_of_six (x : ℤ) : Prop := ∃ m : ℤ, x = 6 * m
def is_multiple_of_nine (y : ℤ) : Prop := ∃ n : ℤ, y = 9 * n

-- Assertions: Given the conditions, prove the following.
theorem multiple_of_six_and_nine (x y : ℤ)
  (hx : is_multiple_of_six x) (hy : is_multiple_of_nine y) :
  ((∃ k : ℤ, x - y = 3 * k) ∧
   (∃ m n : ℤ, x = 6 * m ∧ y = 9 * n ∧ (2 * m - 3 * n) % 3 ≠ 0)) :=
by
  sorry

end multiple_of_six_and_nine_l202_202271


namespace arithmetic_progression_x_value_l202_202668

theorem arithmetic_progression_x_value :
  ∃ x : ℝ, (2 * x - 1) + ((5 * x + 6) - (3 * x + 4)) = (3 * x + 4) + ((3 * x + 4) - (2 * x - 1)) ∧ x = 3 :=
by
  sorry

end arithmetic_progression_x_value_l202_202668


namespace solve_for_y_l202_202655

theorem solve_for_y 
  (x y : ℝ) 
  (h1 : 2 * x - 3 * y = 9) 
  (h2 : x + y = 8) : 
  y = 1.4 := 
sorry

end solve_for_y_l202_202655


namespace inequality_solution_set_is_correct_l202_202065

noncomputable def inequality_solution_set (x : ℝ) : Prop :=
  (3 * x - 1) / (2 - x) ≥ 1

theorem inequality_solution_set_is_correct :
  { x : ℝ | inequality_solution_set x } = { x : ℝ | 3 / 4 ≤ x ∧ x < 2 } :=
by sorry

end inequality_solution_set_is_correct_l202_202065


namespace brown_gumdrops_after_replacement_l202_202239

theorem brown_gumdrops_after_replacement
  (total_gumdrops : ℕ)
  (percent_blue : ℚ)
  (percent_brown : ℚ)
  (percent_red : ℚ)
  (percent_yellow : ℚ)
  (num_green : ℕ)
  (replace_half_blue_with_brown : ℕ) :
  total_gumdrops = 120 →
  percent_blue = 0.30 →
  percent_brown = 0.20 →
  percent_red = 0.15 →
  percent_yellow = 0.10 →
  num_green = 30 →
  replace_half_blue_with_brown = 18 →
  ((percent_brown * ↑total_gumdrops) + replace_half_blue_with_brown) = 42 :=
by sorry

end brown_gumdrops_after_replacement_l202_202239


namespace Aiyanna_more_than_Alyssa_Brady_fewer_than_Aiyanna_Brady_more_than_Alyssa_l202_202573

-- Defining the number of cookies each person had
def Alyssa_cookies : ℕ := 1523
def Aiyanna_cookies : ℕ := 3720
def Brady_cookies : ℕ := 2265

-- Proving the statements
theorem Aiyanna_more_than_Alyssa : Aiyanna_cookies - Alyssa_cookies = 2197 := by
  sorry

theorem Brady_fewer_than_Aiyanna : Aiyanna_cookies - Brady_cookies = 1455 := by
  sorry

theorem Brady_more_than_Alyssa : Brady_cookies - Alyssa_cookies = 742 := by
  sorry

end Aiyanna_more_than_Alyssa_Brady_fewer_than_Aiyanna_Brady_more_than_Alyssa_l202_202573


namespace number_of_ways_to_construct_cube_l202_202700

theorem number_of_ways_to_construct_cube :
  let num_white_cubes := 5
  let num_blue_cubes := 3
  let cube_size := (2, 2, 2)
  let num_rotations := 24
  let num_constructions := 4
  ∃ (num_constructions : ℕ), num_constructions = 4 :=
sorry

end number_of_ways_to_construct_cube_l202_202700


namespace new_person_weight_is_90_l202_202570

-- Define the weight of the replaced person
def replaced_person_weight : ℝ := 40

-- Define the increase in average weight when the new person replaces the replaced person
def increase_in_average_weight : ℝ := 10

-- Define the increase in total weight as 5 times the increase in average weight
def increase_in_total_weight (n : ℕ) (avg_increase : ℝ) : ℝ := n * avg_increase

-- Define the weight of the new person
def new_person_weight (replaced_w : ℝ) (total_increase : ℝ) : ℝ := replaced_w + total_increase

-- Prove that the weight of the new person is 90 kg
theorem new_person_weight_is_90 :
  new_person_weight replaced_person_weight (increase_in_total_weight 5 increase_in_average_weight) = 90 := 
by 
  -- sorry will skip the proof, as required
  sorry

end new_person_weight_is_90_l202_202570


namespace no_rational_roots_of_odd_l202_202394

theorem no_rational_roots_of_odd (m n : ℤ) (hm : m % 2 = 1) (hn : n % 2 = 1) : ¬ ∃ x : ℚ, x^2 + 2 * m * x + 2 * n = 0 :=
sorry

end no_rational_roots_of_odd_l202_202394


namespace solve_quadratic_inequality_l202_202846

open Set Real

noncomputable def quadratic_inequality (x : ℝ) : Prop := -9 * x^2 + 6 * x + 8 > 0

theorem solve_quadratic_inequality :
  {x : ℝ | -9 * x^2 + 6 * x + 8 > 0} = {x : ℝ | -2/3 < x ∧ x < 4/3} :=
by
  sorry

end solve_quadratic_inequality_l202_202846


namespace abigail_monthly_saving_l202_202684

-- Definitions based on the conditions
def total_saving := 48000
def months_in_year := 12

-- The statement to be proved
theorem abigail_monthly_saving : total_saving / months_in_year = 4000 :=
by sorry

end abigail_monthly_saving_l202_202684


namespace mary_flour_requirement_l202_202163

theorem mary_flour_requirement (total_flour : ℕ) (added_flour : ℕ) (remaining_flour : ℕ) 
  (h1 : total_flour = 7) 
  (h2 : added_flour = 2) 
  (h3 : remaining_flour = total_flour - added_flour) : 
  remaining_flour = 5 :=
sorry

end mary_flour_requirement_l202_202163


namespace max_pasture_area_maximization_l202_202767

noncomputable def max_side_length (fence_cost_per_foot : ℕ) (total_cost : ℕ) : ℕ :=
  let total_length := total_cost / fence_cost_per_foot
  let x := total_length / 4
  2 * x

theorem max_pasture_area_maximization :
  max_side_length 8 1920 = 120 :=
by
  sorry

end max_pasture_area_maximization_l202_202767


namespace range_of_m_l202_202689

open Set

theorem range_of_m (m : ℝ) : 
  (∀ x, (m + 1 ≤ x ∧ x ≤ 2 * m - 1) → (-2 < x ∧ x ≤ 5)) → 
  m ∈ Iic (3 : ℝ) :=
by
  intros h
  sorry

end range_of_m_l202_202689


namespace total_matches_correct_total_points_earthlings_correct_total_players_is_square_l202_202657

-- Definitions
variables (t a : ℕ)

-- Part (a): Total number of matches
def total_matches : ℕ := (t + a) * (t + a - 1) / 2

-- Part (b): Total points of the Earthlings
def total_points_earthlings : ℕ := (t * (t - 1)) / 2 + (a * (a - 1)) / 2

-- Part (c): Total number of players is a perfect square
def is_total_players_square : Prop := ∃ k : ℕ, (t + a) = k * k

-- Lean statements
theorem total_matches_correct : total_matches t a = (t + a) * (t + a - 1) / 2 := 
by sorry

theorem total_points_earthlings_correct : total_points_earthlings t a = (t * (t - 1)) / 2 + (a * (a - 1)) / 2 := 
by sorry

theorem total_players_is_square : is_total_players_square t a := by sorry

end total_matches_correct_total_points_earthlings_correct_total_players_is_square_l202_202657


namespace infinite_series_sum_l202_202978

theorem infinite_series_sum :
  (∑' n : ℕ, (n + 1) * (1 / 5) ^ (n + 1)) = 5 / 16 :=
sorry

end infinite_series_sum_l202_202978


namespace michael_hours_worked_l202_202233

def michael_hourly_rate := 7
def michael_overtime_rate := 2 * michael_hourly_rate
def work_hours := 40
def total_earnings := 320

theorem michael_hours_worked :
  (total_earnings = michael_hourly_rate * work_hours + michael_overtime_rate * (42 - work_hours)) :=
sorry

end michael_hours_worked_l202_202233


namespace arithmetic_sequence_general_formula_l202_202245

noncomputable def a_n (n : ℕ) : ℝ :=
sorry

theorem arithmetic_sequence_general_formula (h1 : (a_n 2 + a_n 6) / 2 = 5)
                                            (h2 : (a_n 3 + a_n 7) / 2 = 7) :
  a_n n = 2 * (n : ℝ) - 3 :=
sorry

end arithmetic_sequence_general_formula_l202_202245


namespace find_positive_real_solutions_l202_202634

theorem find_positive_real_solutions (x : ℝ) (h1 : 0 < x) 
(h2 : 3 / 5 * (2 * x ^ 2 - 2) = (x ^ 2 - 40 * x - 8) * (x ^ 2 + 20 * x + 4)) :
    x = (40 + Real.sqrt 1636) / 2 ∨ x = (-20 + Real.sqrt 388) / 2 := by
  sorry

end find_positive_real_solutions_l202_202634


namespace farmer_flax_acres_l202_202376

-- Definitions based on conditions
def total_acres : ℕ := 240
def extra_sunflower_acres : ℕ := 80

-- Problem statement
theorem farmer_flax_acres (F : ℕ) (S : ℕ) 
    (h1 : F + S = total_acres) 
    (h2 : S = F + extra_sunflower_acres) : 
    F = 80 :=
by
    -- Proof goes here
    sorry

end farmer_flax_acres_l202_202376


namespace find_original_number_l202_202501

theorem find_original_number (x : ℚ) (h : 5 * ((3 * x + 6) / 2) = 100) : x = 34 / 3 := sorry

end find_original_number_l202_202501


namespace range_j_l202_202875

def h (x : ℝ) : ℝ := 4 * x - 3
def j (x : ℝ) : ℝ := h (h (h x))

theorem range_j : ∀ x, 0 ≤ x ∧ x ≤ 3 → -63 ≤ j x ∧ j x ≤ 129 :=
by
  intro x
  intro hx
  sorry

end range_j_l202_202875


namespace digital_earth_storage_technology_matured_l202_202228

-- Definitions of conditions as technology properties
def NanoStorageTechnology : Prop := 
  -- Assume it has matured (based on solution analysis)
  sorry

def LaserHolographicStorageTechnology : Prop :=
  -- Assume it has matured (based on solution analysis)
  sorry

def ProteinStorageTechnology : Prop :=
  -- Assume it has matured (based on solution analysis)
  sorry

def DistributedStorageTechnology : Prop :=
  -- Assume it has matured (based on solution analysis)
  sorry

def VirtualStorageTechnology : Prop :=
  -- Assume it has not matured or is not relevant
  sorry

def SpatialStorageTechnology : Prop :=
  -- Assume it has not matured or is not relevant
  sorry

def VisualizationStorageTechnology : Prop :=
  -- Assume it has not matured or is not relevant
  sorry

-- Lean statement to prove the combination
theorem digital_earth_storage_technology_matured : 
  NanoStorageTechnology ∧ LaserHolographicStorageTechnology ∧ ProteinStorageTechnology ∧ DistributedStorageTechnology :=
by {
  sorry
}

end digital_earth_storage_technology_matured_l202_202228


namespace hydrogen_atoms_in_compound_l202_202675

-- Define atoms and their weights
def C_weight : ℕ := 12
def H_weight : ℕ := 1
def O_weight : ℕ := 16

-- Number of each atom in the compound and total molecular weight
def num_C : ℕ := 4
def num_O : ℕ := 1
def total_weight : ℕ := 65

-- Total mass of carbon and oxygen in the compound
def mass_C_O : ℕ := (num_C * C_weight) + (num_O * O_weight)

-- Mass and number of hydrogen atoms in the compound
def mass_H : ℕ := total_weight - mass_C_O
def num_H : ℕ := mass_H / H_weight

theorem hydrogen_atoms_in_compound : num_H = 1 := by
  sorry

end hydrogen_atoms_in_compound_l202_202675


namespace intersection_A_B_l202_202733

def A := {x : ℝ | x > 3}
def B := {x : ℝ | (x - 1) * (x - 4) < 0}

theorem intersection_A_B : A ∩ B = {x : ℝ | 3 < x ∧ x < 4} :=
by
  sorry

end intersection_A_B_l202_202733


namespace intersection_A_B_l202_202299

-- Definition of sets A and B based on given conditions
def A : Set ℝ := {y | ∃ x : ℝ, y = x^2 + 2 * x - 3 }
def B : Set ℝ := {y | ∃ x : ℝ, x < 0 ∧ y = x + 1 / x }

-- Proving the intersection of sets A and B
theorem intersection_A_B : A ∩ B = {y | -4 ≤ y ∧ y ≤ -2} := 
by
  sorry

end intersection_A_B_l202_202299


namespace log_equation_positive_x_l202_202794

theorem log_equation_positive_x (x : ℝ) (hx : 0 < x) (hx1 : x ≠ 1) : 
  (Real.log x / Real.log 2) * (Real.log 9 / Real.log x) = Real.log 9 / Real.log 2 :=
by sorry

end log_equation_positive_x_l202_202794


namespace smallest_log_log_x0_l202_202417

noncomputable def f (x : ℝ) : ℝ := Real.log x - 6 + 2 * x

theorem smallest_log_log_x0 (x₀ : ℝ) (h₀ : f x₀ = 0) (h_dom : 2 < x₀ ∧ x₀ < Real.exp 1) :
  min (min (Real.log x₀) (Real.log (Real.sqrt x₀))) (min (Real.log (Real.log x₀)) ((Real.log x₀)^2)) = Real.log (Real.log x₀) :=
sorry

end smallest_log_log_x0_l202_202417


namespace complete_contingency_table_chi_sq_test_result_expected_value_X_l202_202268

noncomputable def probability_set := {x : ℚ // x ≥ 0 ∧ x ≤ 1}

variable (P : probability_set → probability_set)

-- Conditions from the problem
def P_A_given_not_B : probability_set := ⟨2 / 5, by norm_num⟩
def P_B_given_not_A : probability_set := ⟨5 / 8, by norm_num⟩
def P_B : probability_set := ⟨3 / 4, by norm_num⟩

-- Definitions related to counts and probabilities
def total_students : ℕ := 200
def male_students := P_A_given_not_B.val * total_students
def female_students := total_students - male_students
def score_exceeds_85 := P_B.val * total_students
def score_not_exceeds_85 := total_students - score_exceeds_85

-- Expected counts based on given probabilities
def male_score_not_exceeds_85 := P_A_given_not_B.val * score_not_exceeds_85
def female_score_not_exceeds_85 := score_not_exceeds_85 - male_score_not_exceeds_85
def male_score_exceeds_85 := male_students - male_score_not_exceeds_85
def female_score_exceeds_85 := female_students - female_score_not_exceeds_85

-- Chi-squared test independence 
def chi_squared := (total_students * (male_score_not_exceeds_85 * female_score_exceeds_85 - female_score_not_exceeds_85 * male_score_exceeds_85) ^ 2) / 
                    (male_students * female_students * score_not_exceeds_85 * score_exceeds_85)
def is_related : Prop := chi_squared > 10.828

-- Expected distributions and expectation of X
def P_X_0 := (1 / 4) ^ 2 * (1 / 3) ^ 2
def P_X_1 := 2 * (3 / 4) * (1 / 4) * (1 / 3) ^ 2 + 2 * (2 / 3) * (1 / 3) * (1 / 4) ^ 2
def P_X_2 := (3 / 4) ^ 2 * (1 / 3) ^ 2 + (1 / 4) ^ 2 * (2 / 3) ^ 2 + 2 * (2 / 3) * (1 / 3) * (3 / 4) * (1 / 4)
def P_X_3 := (3 / 4) ^ 2 * 2 * (2 / 3) * (1 / 3) + 2 * (3 / 4) * (1 / 4) * (2 / 3) ^ 2
def P_X_4 := (3 / 4) ^ 2 * (2 / 3) ^ 2
def expectation_X := 0 * P_X_0 + 1 * P_X_1 + 2 * P_X_2 + 3 * P_X_3 + 4 * P_X_4

-- Lean theorem statements for answers using the above definitions
theorem complete_contingency_table :
  male_score_not_exceeds_85 + female_score_not_exceeds_85 = score_not_exceeds_85 ∧
  male_score_exceeds_85 + female_score_exceeds_85 = score_exceeds_85 ∧
  male_students + female_students = total_students := sorry

theorem chi_sq_test_result :
  is_related = true := sorry

theorem expected_value_X :
  expectation_X = 17 / 6 := sorry

end complete_contingency_table_chi_sq_test_result_expected_value_X_l202_202268


namespace exists_even_function_b_l202_202431

-- Define the function f(x) = 2x^2 - b*x
def f (b x : ℝ) : ℝ := 2 * x^2 - b * x

-- Define the condition for f being an even function: f(-x) = f(x)
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

-- The main theorem stating the existence of a b in ℝ such that f is an even function
theorem exists_even_function_b :
  ∃ b : ℝ, is_even_function (f b) :=
by
  sorry

end exists_even_function_b_l202_202431


namespace employee_Y_base_pay_l202_202479

theorem employee_Y_base_pay (P : ℝ) (h1 : 1.2 * P + P * 1.1 + P * 1.08 + P = P * 4.38)
                            (h2 : 2 * 1.5 * 1.2 * P = 3.6 * P)
                            (h3 : P * 4.38 + 100 + 3.6 * P = 1800) :
  P = 213.03 :=
by
  sorry

end employee_Y_base_pay_l202_202479


namespace tetradecagon_edge_length_correct_l202_202041

-- Define the parameters of the problem
def regular_tetradecagon_perimeter (n : ℕ := 14) : ℕ := 154

-- Define the length of one edge
def edge_length (P : ℕ) (n : ℕ) : ℕ := P / n

-- State the theorem
theorem tetradecagon_edge_length_correct :
  edge_length (regular_tetradecagon_perimeter 14) 14 = 11 := by
  sorry

end tetradecagon_edge_length_correct_l202_202041


namespace three_digit_numbers_divisible_by_5_l202_202532

theorem three_digit_numbers_divisible_by_5 : 
  let first_term := 100
  let last_term := 995
  let common_difference := 5 
  (last_term - first_term) / common_difference + 1 = 180 :=
by
  sorry

end three_digit_numbers_divisible_by_5_l202_202532


namespace relationship_a_b_c_l202_202171

noncomputable def a := Real.log 3 / Real.log (1/2)
noncomputable def b := Real.log (1/2) / Real.log 3
noncomputable def c := Real.exp (0.3 * Real.log 2)

theorem relationship_a_b_c : 
  a < b ∧ b < c := 
by {
  sorry
}

end relationship_a_b_c_l202_202171


namespace peanut_count_l202_202460

-- Definitions
def initial_peanuts : Nat := 10
def added_peanuts : Nat := 8

-- Theorem to prove
theorem peanut_count : (initial_peanuts + added_peanuts) = 18 := 
by
  -- Proof placeholder
  sorry

end peanut_count_l202_202460


namespace race_distance_l202_202415

theorem race_distance (D : ℝ) (h1 : ∀ t : ℝ, t = 30 → D / 30 = D / t)
                      (h2 : ∀ t : ℝ, t = 45 → D / 45 = D / t)
                      (h3 : ∀ d : ℝ, d = 33.333333333333336 → D - (D / 45) * 30 = d) :
  D = 100 :=
sorry

end race_distance_l202_202415


namespace set_equality_l202_202103

theorem set_equality : 
  { x : ℕ | ∃ k : ℕ, 6 - x = k ∧ 8 % k = 0 } = { 2, 4, 5 } :=
by
  sorry

end set_equality_l202_202103


namespace solve_system_l202_202356

theorem solve_system :
  ∃ x y : ℝ, (x^3 + y^3) * (x^2 + y^2) = 64 ∧ x + y = 2 ∧ 
  ((x = 1 + Real.sqrt (5 / 3) ∧ y = 1 - Real.sqrt (5 / 3)) ∨ 
   (x = 1 - Real.sqrt (5 / 3) ∧ y = 1 + Real.sqrt (5 / 3))) :=
by
  sorry

end solve_system_l202_202356


namespace probability_area_l202_202444

noncomputable def probability_x_y_le_five (x y : ℝ) : ℚ :=
  if 0 ≤ x ∧ x ≤ 4 ∧ 0 ≤ y ∧ y ≤ 8 ∧ x + y ≤ 5 then 1 else 0

theorem probability_area {P : ℚ} :
  (∀ (x y : ℝ), 0 ≤ x ∧ x ≤ 4 ∧ 0 ≤ y ∧ y ≤ 8 → P = probability_x_y_le_five x y / (4 * 8)) →
  P = 5 / 16 :=
by
  sorry

end probability_area_l202_202444


namespace drying_time_short_haired_dog_l202_202750

theorem drying_time_short_haired_dog (x : ℕ) (h1 : ∀ y, y = 2 * x) (h2 : 6 * x + 9 * (2 * x) = 240) : x = 10 :=
by
  sorry

end drying_time_short_haired_dog_l202_202750


namespace sales_value_minimum_l202_202242

theorem sales_value_minimum (V : ℝ) (base_salary new_salary : ℝ) (commission_rate sales_needed old_salary : ℝ)
    (h_base_salary : base_salary = 45000 )
    (h_new_salary : new_salary = base_salary + 0.15 * V * sales_needed)
    (h_sales_needed : sales_needed = 266.67)
    (h_old_salary : old_salary = 75000) :
    new_salary ≥ old_salary ↔ V ≥ 750 := 
by
  sorry

end sales_value_minimum_l202_202242


namespace average_monthly_income_is_2125_l202_202380

noncomputable def calculate_average_monthly_income (expenses_3_months: ℕ) (expenses_4_months: ℕ) (expenses_5_months: ℕ) (savings_per_year: ℕ) : ℕ :=
  (expenses_3_months * 3 + expenses_4_months * 4 + expenses_5_months * 5 + savings_per_year) / 12

theorem average_monthly_income_is_2125 :
  calculate_average_monthly_income 1700 1550 1800 5200 = 2125 :=
by
  sorry

end average_monthly_income_is_2125_l202_202380


namespace class_B_more_uniform_l202_202046

def x_A : ℝ := 80
def x_B : ℝ := 80
def S2_A : ℝ := 240
def S2_B : ℝ := 180

theorem class_B_more_uniform (h1 : x_A = 80) (h2 : x_B = 80) (h3 : S2_A = 240) (h4 : S2_B = 180) : 
  S2_B < S2_A :=
by {
  exact sorry
}

end class_B_more_uniform_l202_202046


namespace volume_of_increased_box_l202_202191

theorem volume_of_increased_box {l w h : ℝ} (vol : l * w * h = 4860) (sa : l * w + w * h + l * h = 930) (sum_dim : l + w + h = 56) :
  (l + 2) * (w + 3) * (h + 1) = 5964 :=
by
  sorry

end volume_of_increased_box_l202_202191


namespace find_b_l202_202881

-- Definitions
variable (k : ℤ) (b : ℤ)
def x := 3 * k
def y := 4 * k
def z := 7 * k

-- Conditions
axiom ratio : x / y = 3 / 4 ∧ y / z = 4 / 7
axiom equation : y = 15 * b - 5

-- Theorem statement
theorem find_b : ∃ b : ℤ, 4 * k = 15 * b - 5 ∧ b = 3 :=
by
  sorry

end find_b_l202_202881


namespace average_trees_planted_l202_202298

theorem average_trees_planted 
  (A : ℕ) 
  (B : ℕ) 
  (C : ℕ) 
  (h1 : A = 35) 
  (h2 : B = A + 6) 
  (h3 : C = A - 3) : 
  (A + B + C) / 3 = 36 :=
  by
  sorry

end average_trees_planted_l202_202298


namespace ribbon_arrangement_count_correct_l202_202127

-- Definitions for the problem conditions
inductive Color
| red
| yellow
| blue

-- The color sequence from top to bottom
def color_sequence : List Color := [Color.red, Color.blue, Color.yellow, Color.yellow]

-- A function to count the valid arrangements
def count_valid_arrangements (sequence : List Color) : Nat :=
  -- Since we need to prove, we're bypassing the actual implementation with sorry
  sorry

-- The proof statement
theorem ribbon_arrangement_count_correct : count_valid_arrangements color_sequence = 12 :=
by
  sorry

end ribbon_arrangement_count_correct_l202_202127


namespace find_width_of_lawn_l202_202260

noncomputable def width_of_lawn
    (length : ℕ)
    (cost : ℕ)
    (cost_per_sq_m : ℕ)
    (road_width : ℕ) : ℕ :=
  let total_area := cost / cost_per_sq_m
  let road_area_length := road_width * length
  let eq_area := total_area - road_area_length
  eq_area / road_width

theorem find_width_of_lawn :
  width_of_lawn 110 4800 3 10 = 50 :=
by
  sorry

end find_width_of_lawn_l202_202260


namespace equation_of_parallel_line_l202_202996

theorem equation_of_parallel_line {x y : ℝ} :
  (∃ b : ℝ, ∀ (P : ℝ × ℝ), P = (1, 0) → (2 * P.1 + P.2 + b = 0)) ↔ 
  (∃ b : ℝ, b = -2 ∧ ∀ (P : ℝ × ℝ), P = (1, 0) → (2 * P.1 + P.2 - 2 = 0)) := 
by 
  sorry

end equation_of_parallel_line_l202_202996


namespace maximum_radius_l202_202714

open Set Real

-- Definitions of sets M, N, and D_r.
def M : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.snd ≥ 1 / 4 * p.fst^2}

def N : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.snd ≤ -1 / 4 * p.fst^2 + p.fst + 7}

def D_r (x₀ y₀ r : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.fst - x₀)^2 + (p.snd - y₀)^2 ≤ r^2}

-- Theorem statement for the largest r
theorem maximum_radius {x₀ y₀ : ℝ} (H : D_r x₀ y₀ r ⊆ M ∩ N) :
  r = sqrt ((25 - 5 * sqrt 5) / 2) :=
sorry

end maximum_radius_l202_202714


namespace problem_solution_l202_202820

variable (x y : ℝ)

theorem problem_solution
  (h1 : (x + y)^2 = 64)
  (h2 : x * y = 15) :
  (x - y)^2 = 4 := 
by
  sorry

end problem_solution_l202_202820


namespace greatest_visible_unit_cubes_from_corner_l202_202748

theorem greatest_visible_unit_cubes_from_corner
  (n : ℕ) (units : ℕ) 
  (cube_volume : ∀ x, x = 1000)
  (face_size : ∀ x, x = 10) :
  (units = 274) :=
by sorry

end greatest_visible_unit_cubes_from_corner_l202_202748


namespace john_remaining_money_l202_202653

theorem john_remaining_money (q : ℝ) : 
  let drink_cost := 5 * q
  let medium_pizza_cost := 3 * 2 * q
  let large_pizza_cost := 2 * 3 * q
  let dessert_cost := 4 * (1 / 2) * q
  let total_cost := drink_cost + medium_pizza_cost + large_pizza_cost + dessert_cost
  let initial_money := 60
  initial_money - total_cost = 60 - 19 * q :=
by
  sorry

end john_remaining_money_l202_202653


namespace intersection_nonempty_iff_m_lt_one_l202_202537

open Set Real

variable {m : ℝ}

theorem intersection_nonempty_iff_m_lt_one 
  (A : Set ℝ) (B : Set ℝ) (U : Set ℝ := univ) 
  (hA : A = {x | x + m >= 0}) 
  (hB : B = {x | -1 < x ∧ x < 5}) : 
  (U \ A ∩ B ≠ ∅) ↔ m < 1 := by
  sorry

end intersection_nonempty_iff_m_lt_one_l202_202537


namespace average_age_before_new_students_l202_202020

theorem average_age_before_new_students
  (N : ℕ) (A : ℚ) 
  (hN : N = 8) 
  (new_avg : (A - 4) = ((A * N) + (32 * 8)) / (N + 8)) 
  : A = 40 := 
by
  sorry

end average_age_before_new_students_l202_202020


namespace minimize_d_and_distance_l202_202135

-- Define point and geometric shapes
structure Point :=
  (x : ℝ)
  (y : ℝ)

def Parabola (P : Point) : Prop := P.x^2 = 4 * P.y
def Circle (P1 : Point) : Prop := (P1.x - 2)^2 + (P1.y + 1)^2 = 1

-- Define the point P and point P1
variable (P : Point)
variable (P1 : Point)

-- Condition: P is on the parabola
axiom on_parabola : Parabola P

-- Condition: P1 is on the circle
axiom on_circle : Circle P1

-- Theorem: coordinates of P when the function d + distance(P, P1) is minimized
theorem minimize_d_and_distance :
  P = { x := 2 * Real.sqrt 2 - 2, y := 3 - 2 * Real.sqrt 2 } :=
sorry

end minimize_d_and_distance_l202_202135


namespace simplify_frac_l202_202730

variable (m : ℝ)

theorem simplify_frac : m^2 ≠ 9 → (3 / (m^2 - 9) + m / (9 - m^2)) = - (1 / (m + 3)) :=
by
  intro h
  sorry

end simplify_frac_l202_202730


namespace prime_numbers_satisfy_equation_l202_202184

noncomputable def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem prime_numbers_satisfy_equation :
  ∃ (p q r : ℕ), is_prime p ∧ is_prime q ∧ is_prime r ∧ (p + q^2 = r^4) ∧ 
  (p = 7) ∧ (q = 3) ∧ (r = 2) :=
by
  sorry

end prime_numbers_satisfy_equation_l202_202184


namespace mass_percentage_C_in_C6HxO6_indeterminate_l202_202606

-- Definition of conditions
def mass_percentage_C_in_C6H8O6 : ℚ := 40.91 / 100
def molar_mass_C : ℚ := 12.01
def molar_mass_H : ℚ := 1.01
def molar_mass_O : ℚ := 16.00

-- Formula for molar mass of C6H8O6
def molar_mass_C6H8O6 : ℚ := 6 * molar_mass_C + 8 * molar_mass_H + 6 * molar_mass_O

-- Mass of carbon in C6H8O6 is 40.91% of the total molar mass
def mass_of_C_in_C6H8O6 : ℚ := mass_percentage_C_in_C6H8O6 * molar_mass_C6H8O6

-- Hypothesis: mass percentage of carbon in C6H8O6 is given
axiom hyp_mass_percentage_C_in_C6H8O6 : mass_of_C_in_C6H8O6 = 72.06

-- Proof that we need the value of x to determine the mass percentage of C in C6HxO6
theorem mass_percentage_C_in_C6HxO6_indeterminate (x : ℚ) :
  (molar_mass_C6H8O6 = 176.14) → (mass_of_C_in_C6H8O6 = 72.06) → False :=
by
  sorry

end mass_percentage_C_in_C6HxO6_indeterminate_l202_202606


namespace lg_sum_eq_lg_double_diff_l202_202647

theorem lg_sum_eq_lg_double_diff (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) 
  (h_harmonic : 2 / y = 1 / x + 1 / z) : 
  Real.log (x + z) + Real.log (x - 2 * y + z) = 2 * Real.log (x - z) := 
by
  sorry

end lg_sum_eq_lg_double_diff_l202_202647


namespace solution_l202_202313

variable (x y z : ℝ)
variable (hx : x > 0)
variable (hy : y > 0)
variable (hz : z > 0)

-- Condition 1: 20/x + 6/y = 1
axiom eq1 : 20 / x + 6 / y = 1

-- Condition 2: 4/x + 2/y = 2/9
axiom eq2 : 4 / x + 2 / y = 2 / 9

-- What we need to prove: 1/z = 1/x + 1/y
axiom eq3 : 1 / x + 1 / y = 1 / z

theorem solution : z = 14.4 := by
  -- Omitted proof, just the statement
  sorry

end solution_l202_202313


namespace A_can_complete_work_in_4_days_l202_202779

-- Definitions based on conditions
def work_done_in_one_day (days : ℕ) : ℚ := 1 / days

def combined_work_done_in_two_days (a b c : ℕ) : ℚ :=
  work_done_in_one_day a + work_done_in_one_day b + work_done_in_one_day c

-- Theorem statement based on the problem
theorem A_can_complete_work_in_4_days (A B C : ℕ) 
  (hB : B = 8) (hC : C = 8) 
  (h_combined : combined_work_done_in_two_days A B C = work_done_in_one_day 2) :
  A = 4 :=
sorry

end A_can_complete_work_in_4_days_l202_202779


namespace rectangle_area_l202_202097

theorem rectangle_area (L B : ℕ) (h1 : L - B = 23) (h2 : 2 * (L + B) = 266) : L * B = 4290 := 
by 
  sorry

end rectangle_area_l202_202097


namespace sufficient_condition_l202_202831

theorem sufficient_condition (A B : Set α) (h : A ⊆ B) (x : α) : x ∈ A → x ∈ B :=
  by
    intro h1
    apply h
    exact h1

end sufficient_condition_l202_202831


namespace find_x_l202_202481

theorem find_x (x : ℕ) (h : x + 1 = 6) : x = 5 :=
sorry

end find_x_l202_202481


namespace find_x_and_y_l202_202772

variables (x y : ℝ)

def arithmetic_mean_condition : Prop := (8 + 15 + x + y + 22 + 30) / 6 = 15
def relationship_condition : Prop := y = x + 6

theorem find_x_and_y (h1 : arithmetic_mean_condition x y) (h2 : relationship_condition x y) : 
  x = 4.5 ∧ y = 10.5 :=
by
  sorry

end find_x_and_y_l202_202772


namespace nth_equation_pattern_l202_202262

theorem nth_equation_pattern (n : ℕ) : (n + 1) * (n^2 - n + 1) - 1 = n^3 :=
by
  sorry

end nth_equation_pattern_l202_202262


namespace alice_meeting_distance_l202_202515

noncomputable def distanceAliceWalks (t : ℝ) : ℝ :=
  6 * t

theorem alice_meeting_distance :
  ∃ t : ℝ, 
    distanceAliceWalks t = 
      (900 * Real.sqrt 2 - Real.sqrt 630000) / 11 ∧
    (5 * t) ^ 2 =
      (6 * t) ^ 2 + 150 ^ 2 - 2 * 6 * t * 150 * Real.cos (Real.pi / 4) :=
sorry

end alice_meeting_distance_l202_202515


namespace negation_prop_l202_202706

theorem negation_prop (p : Prop) : 
  (∀ (x : ℝ), x > 2 → x^2 - 1 > 0) → (¬(∀ (x : ℝ), x > 2 → x^2 - 1 > 0) ↔ (∃ (x : ℝ), x > 2 ∧ x^2 - 1 ≤ 0)) :=
by 
  sorry

end negation_prop_l202_202706


namespace amount_given_to_second_set_of_families_l202_202203

theorem amount_given_to_second_set_of_families
  (total_spent : ℝ) (amount_first_set : ℝ) (amount_last_set : ℝ)
  (h_total_spent : total_spent = 900)
  (h_amount_first_set : amount_first_set = 325)
  (h_amount_last_set : amount_last_set = 315) :
  total_spent - amount_first_set - amount_last_set = 260 :=
by
  -- sorry is placed to skip the proof
  sorry

end amount_given_to_second_set_of_families_l202_202203


namespace minimum_value_of_expression_l202_202927

noncomputable def min_value (a b : ℝ) : ℝ :=
  a^2 + (1 / (a * b)) + (1 / (a * (a - b)))

theorem minimum_value_of_expression (a b : ℝ) (h1 : a > b) (h2 : b > 0) : min_value a b >= 4 := by
  sorry

end minimum_value_of_expression_l202_202927


namespace monomial_k_add_n_l202_202601

variable (k n : ℤ)

-- Conditions
def is_monomial_coefficient (k : ℤ) : Prop := -k = 5
def is_monomial_degree (n : ℤ) : Prop := n + 1 = 7

-- Theorem to prove
theorem monomial_k_add_n (hk : is_monomial_coefficient k) (hn : is_monomial_degree n) : k + n = 1 :=
by
  sorry

end monomial_k_add_n_l202_202601


namespace train_passing_time_l202_202935

noncomputable def first_train_length : ℝ := 270
noncomputable def first_train_speed_kmh : ℝ := 108
noncomputable def second_train_length : ℝ := 360
noncomputable def second_train_speed_kmh : ℝ := 72

noncomputable def convert_speed_to_mps (speed_kmh : ℝ) : ℝ := speed_kmh * (1000 / 3600)

noncomputable def first_train_speed_mps : ℝ := convert_speed_to_mps first_train_speed_kmh
noncomputable def second_train_speed_mps : ℝ := convert_speed_to_mps second_train_speed_kmh

noncomputable def relative_speed_mps : ℝ := first_train_speed_mps + second_train_speed_mps
noncomputable def total_distance : ℝ := first_train_length + second_train_length
noncomputable def time_to_pass : ℝ := total_distance / relative_speed_mps

theorem train_passing_time : time_to_pass = 12.6 :=
by 
  sorry

end train_passing_time_l202_202935


namespace find_number_l202_202680

theorem find_number (x : ℤ) (h1 : x - 2 + 4 = 9) : x = 7 :=
by
  sorry

end find_number_l202_202680


namespace find_first_term_l202_202456

theorem find_first_term (a : ℚ) (n : ℕ) (T : ℕ → ℚ)
  (hT : ∀ n, T n = n * (2 * a + 5 * (n - 1)) / 2)
  (h_const : ∃ c : ℚ, ∀ n > 0, T (4 * n) / T n = c) :
  a = 5 / 2 := 
sorry

end find_first_term_l202_202456


namespace triangle_right_angle_AB_solution_l202_202851

theorem triangle_right_angle_AB_solution (AC BC AB : ℝ) (hAC : AC = 6) (hBC : BC = 8) :
  (AC^2 + BC^2 = AB^2 ∨ AB^2 + AC^2 = BC^2) ↔ (AB = 10 ∨ AB = 2 * Real.sqrt 7) :=
by
  sorry

end triangle_right_angle_AB_solution_l202_202851


namespace circle_circumference_ratio_l202_202393

theorem circle_circumference_ratio (A₁ A₂ : ℝ) (h : A₁ / A₂ = 16 / 25) :
  ∃ C₁ C₂ : ℝ, (C₁ / C₂ = 4 / 5) :=
by
  -- Definitions and calculations to be done here
  sorry

end circle_circumference_ratio_l202_202393


namespace avg_megabyte_usage_per_hour_l202_202137

theorem avg_megabyte_usage_per_hour (megabytes : ℕ) (days : ℕ) (hours : ℕ) (avg_mbps : ℕ)
  (h1 : megabytes = 27000)
  (h2 : days = 15)
  (h3 : hours = days * 24)
  (h4 : avg_mbps = megabytes / hours) : 
  avg_mbps = 75 := by
  sorry

end avg_megabyte_usage_per_hour_l202_202137


namespace barbie_earrings_l202_202290

theorem barbie_earrings (total_earrings_alissa : ℕ) (alissa_triple_given : ℕ → ℕ) 
  (given_earrings_double_bought : ℕ → ℕ) (pairs_of_earrings : ℕ) : 
  total_earrings_alissa = 36 → 
  alissa_triple_given (total_earrings_alissa / 3) = total_earrings_alissa → 
  given_earrings_double_bought (total_earrings_alissa / 3) = total_earrings_alissa →
  pairs_of_earrings = 12 :=
by
  intros h1 h2 h3
  sorry

end barbie_earrings_l202_202290


namespace solution_exists_l202_202225

noncomputable def verifySolution (x y z : ℝ) : Prop := 
  x^2 - y = (z - 1)^2 ∧
  y^2 - z = (x - 1)^2 ∧
  z^2 - x = (y- 1)^2 

theorem solution_exists (x y z : ℝ) (h : verifySolution x y z) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) :
  (x = 1 ∧ y = 1 ∧ z = 1) ∨ 
  (x, y, z) = (-2.93122, 2.21124, 0.71998) ∨ 
  (x, y, z) = (2.21124, 0.71998, -2.93122) ∨ 
  (x, y, z) = (0.71998, -2.93122, 2.21124) :=
sorry

end solution_exists_l202_202225


namespace stratified_sampling_correct_l202_202154

variables (total_employees senior_employees mid_level_employees junior_employees sample_size : ℕ)
          (sampling_ratio : ℚ)
          (senior_sample mid_sample junior_sample : ℕ)

-- Conditions
def company_conditions := 
  total_employees = 450 ∧ 
  senior_employees = 45 ∧ 
  mid_level_employees = 135 ∧ 
  junior_employees = 270 ∧ 
  sample_size = 30 ∧ 
  sampling_ratio = 1 / 15

-- Proof goal
theorem stratified_sampling_correct : 
  company_conditions total_employees senior_employees mid_level_employees junior_employees sample_size sampling_ratio →
  senior_sample = senior_employees * sampling_ratio ∧ 
  mid_sample = mid_level_employees * sampling_ratio ∧ 
  junior_sample = junior_employees * sampling_ratio ∧
  senior_sample + mid_sample + junior_sample = sample_size :=
by sorry

end stratified_sampling_correct_l202_202154


namespace math_problem_l202_202286

theorem math_problem : 1999^2 - 2000 * 1998 = 1 := 
by
  sorry

end math_problem_l202_202286


namespace carriages_per_train_l202_202939

variable (c : ℕ)

theorem carriages_per_train :
  (∃ c : ℕ, (25 + 10) * c * 3 = 420) → c = 4 :=
by
  sorry

end carriages_per_train_l202_202939


namespace average_calls_per_day_l202_202543

def calls_Monday : ℕ := 35
def calls_Tuesday : ℕ := 46
def calls_Wednesday : ℕ := 27
def calls_Thursday : ℕ := 61
def calls_Friday : ℕ := 31

def total_calls : ℕ := calls_Monday + calls_Tuesday + calls_Wednesday + calls_Thursday + calls_Friday
def number_of_days : ℕ := 5

theorem average_calls_per_day : (total_calls / number_of_days) = 40 := 
by 
  -- calculations and proof steps go here.
  sorry

end average_calls_per_day_l202_202543


namespace probability_girls_same_color_l202_202591

open Classical

noncomputable def probability_same_color_marbles : ℚ :=
(3/6) * (2/5) * (1/4) + (3/6) * (2/5) * (1/4)

theorem probability_girls_same_color :
  probability_same_color_marbles = 1/20 := by
  sorry

end probability_girls_same_color_l202_202591


namespace binary_110101_is_53_l202_202860

def binary_to_decimal (n : Nat) : Nat :=
  let digits := [1, 1, 0, 1, 0, 1]  -- Define binary digits from the problem statement
  digits.reverse.foldr (λ d (acc, pow) => (acc + d * (2^pow), pow + 1)) (0, 0) |>.fst

theorem binary_110101_is_53 : binary_to_decimal 110101 = 53 := by
  sorry

end binary_110101_is_53_l202_202860


namespace negation_of_existence_proposition_l202_202599

theorem negation_of_existence_proposition :
  ¬ (∃ x : ℝ, x^2 + 2*x - 8 = 0) ↔ ∀ x : ℝ, x^2 + 2*x - 8 ≠ 0 := by
  sorry

end negation_of_existence_proposition_l202_202599


namespace tangent_line_at_one_l202_202336

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x

theorem tangent_line_at_one : ∀ (x y : ℝ), y = 2 * Real.exp 1 * x - Real.exp 1 → 
  ∃ m b : ℝ, (∀ x: ℝ, f x = m * x + b) ∧ (m = 2 * Real.exp 1) ∧ (b = -Real.exp 1) :=
by
  sorry

end tangent_line_at_one_l202_202336


namespace rectangle_cut_l202_202584

def dimensions_ratio (x y : ℕ) : Prop := ∃ (r : ℚ), x = r * y

theorem rectangle_cut (k m n : ℕ) (hk : ℝ) (hm : ℝ) (hn : ℝ) 
  (h1 : k + m + n = 10) 
  (h2 : k * 9 / 10 = hk)
  (h3 : m * 9 / 10 = hm)
  (h4 : n * 9 / 10 = hn)
  (h5 : hk + hm + hn = 9) :
  ∃ (k' m' n' : ℕ), 
    dimensions_ratio k k' ∧ 
    dimensions_ratio m m' ∧
    dimensions_ratio n n' ∧
    k ≠ m ∧ m ≠ n ∧ k ≠ n :=
sorry

end rectangle_cut_l202_202584


namespace distance_to_place_l202_202166

variables {r c1 c2 t D : ℝ}

theorem distance_to_place (h : t = (D / (r - c1)) + (D / (r + c2))) :
  D = t * (r^2 - c1 * c2) / (2 * r + c2 - c1) :=
by
  have h1 : D * (r + c2) / (r - c1) * (r - c1) = D * (r + c2) := by sorry
  have h2 : D * (r - c1) / (r + c2) * (r + c2) = D * (r - c1) := by sorry
  have h3 : D * (r + c2) = D * (r + c2) := by sorry
  have h4 : D * (r - c1) = D * (r - c1) := by sorry
  have h5 : t * (r - c1) * (r + c2) = D * (r + c2) + D * (r - c1) := by sorry
  have h6 : t * (r^2 - c1 * c2) = D * (2 * r + c2 - c1) := by sorry
  have h7 : D = t * (r^2 - c1 * c2) / (2 * r + c2 - c1) := by sorry
  exact h7

end distance_to_place_l202_202166


namespace min_a_b_l202_202894

theorem min_a_b (a b : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 45 * a + b = 2021) : a + b = 85 :=
sorry

end min_a_b_l202_202894


namespace sufficient_but_not_necessary_l202_202465

theorem sufficient_but_not_necessary (x : ℝ) (h : x > 4) :
  (x ^ 2 - 5 * x + 4 ≥ 0 ∧ ¬(∀ x, (x ^ 2 - 5 * x + 4 ≥ 0 → x > 4))) :=
by
  sorry

end sufficient_but_not_necessary_l202_202465


namespace original_wage_before_increase_l202_202323

theorem original_wage_before_increase (W : ℝ) 
  (h1 : W * 1.4 = 35) : W = 25 := by
  sorry

end original_wage_before_increase_l202_202323


namespace value_of_a6_in_arithmetic_sequence_l202_202814

/-- In the arithmetic sequence {a_n}, if a_2 and a_{10} are the two roots of the equation
    x^2 + 12x - 8 = 0, prove that the value of a_6 is -6. -/
theorem value_of_a6_in_arithmetic_sequence :
  ∃ a_2 a_10 : ℤ, (a_2 + a_10 = -12 ∧
  (2: ℤ) * ((a_2 + a_10) / (2 * 1)) = a_2 + a_10 ) → 
  ∃ a_6: ℤ, a_6 = -6 :=
by
  sorry

end value_of_a6_in_arithmetic_sequence_l202_202814


namespace remainder_of_sum_of_powers_div_2_l202_202453

theorem remainder_of_sum_of_powers_div_2 : 
  (1^1 + 2^2 + 3^3 + 4^4 + 5^5 + 6^6 + 7^7 + 8^8 + 9^9) % 2 = 1 :=
by 
  sorry

end remainder_of_sum_of_powers_div_2_l202_202453


namespace find_f3_l202_202249

theorem find_f3 {f : ℝ → ℝ} (h : ∀ x : ℝ, f (2 * x + 1) = 3 * x - 5) : f 3 = -2 :=
by
  sorry

end find_f3_l202_202249


namespace find_tangent_c_l202_202640

theorem find_tangent_c (c : ℝ) : 
  (∀ x y : ℝ, y = 3 * x + c → y^2 = 12 * x) → (c = 1) :=
by
  intros h
  sorry

end find_tangent_c_l202_202640


namespace average_speed_round_trip_l202_202631

theorem average_speed_round_trip :
  ∀ (D : ℝ), 
  D > 0 → 
  let upstream_speed := 6 
  let downstream_speed := 5 
  (2 * D) / ((D / upstream_speed) + (D / downstream_speed)) = 60 / 11 :=
by
  intro D hD
  let upstream_speed := 6
  let downstream_speed := 5
  have h : (2 * D) / ((D / upstream_speed) + (D / downstream_speed)) = 60 / 11 := sorry
  exact h

end average_speed_round_trip_l202_202631


namespace no_such_functions_exist_l202_202884

open Function

theorem no_such_functions_exist : ¬ (∃ (f g : ℝ → ℝ), ∀ x : ℝ, f (g x) = x^2 ∧ g (f x) = x^3) := 
sorry

end no_such_functions_exist_l202_202884


namespace cube_volume_increase_l202_202305

variable (a : ℝ)

theorem cube_volume_increase (a : ℝ) : (2 * a)^3 - a^3 = 7 * a^3 :=
by
  sorry

end cube_volume_increase_l202_202305


namespace rectangle_area_l202_202922

theorem rectangle_area (y : ℝ) (h_rect : (5 - (-3)) * (y - (-1)) = 48) (h_pos : 0 < y) : y = 5 :=
by
  sorry

end rectangle_area_l202_202922


namespace probability_of_two_in_decimal_rep_of_eight_over_eleven_l202_202335

theorem probability_of_two_in_decimal_rep_of_eight_over_eleven : 
  (∃ B : List ℕ, (B = [7, 2]) ∧ (1 = (B.count 2) / (B.length)) ∧ 
  (0 + B.sum + 1) / 11 = 8 / 11) := sorry

end probability_of_two_in_decimal_rep_of_eight_over_eleven_l202_202335


namespace ef_plus_e_l202_202277

-- Define the polynomial expression
def polynomial_expr (y : ℤ) := 15 * y^2 - 82 * y + 48

-- Define the factorized form
def factorized_form (E F : ℤ) (y : ℤ) := (E * y - 16) * (F * y - 3)

-- Define the main statement to prove
theorem ef_plus_e : ∃ E F : ℤ, E * F + E = 20 ∧ ∀ y : ℤ, polynomial_expr y = factorized_form E F y :=
by {
  sorry
}

end ef_plus_e_l202_202277


namespace prop_D_l202_202402

variable (a b : ℝ)

theorem prop_D (a b : ℝ) (h : a > |b|) : a^2 > b^2 :=
  by
    sorry

end prop_D_l202_202402


namespace number_and_its_square_root_l202_202775

theorem number_and_its_square_root (x : ℝ) (h : x + 10 * Real.sqrt x = 39) : x = 9 :=
sorry

end number_and_its_square_root_l202_202775


namespace least_five_digit_perfect_square_and_cube_l202_202180

theorem least_five_digit_perfect_square_and_cube : ∃ n : ℕ, (10000 ≤ n ∧ n < 100000) ∧ (∃ k1 : ℕ, n = k1 ^ 2) ∧ (∃ k2 : ℕ, n = k2 ^ 3) ∧ n = 15625 :=
by
  sorry

end least_five_digit_perfect_square_and_cube_l202_202180


namespace find_a_plus_c_l202_202330

theorem find_a_plus_c (a b c d : ℝ) (h1 : ab + bc + cd + da = 40) (h2 : b + d = 8) : a + c = 5 :=
by
  sorry

end find_a_plus_c_l202_202330


namespace matt_total_vibrations_l202_202216

noncomputable def vibrations_lowest : ℕ := 1600
noncomputable def vibrations_highest : ℕ := vibrations_lowest + (6 * vibrations_lowest / 10)
noncomputable def time_seconds : ℕ := 300
noncomputable def total_vibrations : ℕ := vibrations_highest * time_seconds

theorem matt_total_vibrations :
  total_vibrations = 768000 := by
  sorry

end matt_total_vibrations_l202_202216


namespace stadium_capacity_l202_202438

theorem stadium_capacity 
  (C : ℕ)
  (entry_fee : ℕ := 20)
  (three_fourth_full_fees : ℕ := 3 / 4 * C * entry_fee)
  (full_fees : ℕ := C * entry_fee)
  (fee_difference : ℕ := full_fees - three_fourth_full_fees)
  (h : fee_difference = 10000) :
  C = 2000 :=
by
  sorry

end stadium_capacity_l202_202438


namespace inequality_8xyz_l202_202219

theorem inequality_8xyz (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x + y + z = 1) : 
  (1 - x) * (1 - y) * (1 - z) > 8 * x * y * z := 
  by sorry

end inequality_8xyz_l202_202219


namespace transformation_correct_l202_202619

theorem transformation_correct (a b : ℝ) (h₁ : 3 * a = 2 * b) (h₂ : a ≠ 0) (h₃ : b ≠ 0) :
  a / 2 = b / 3 :=
sorry

end transformation_correct_l202_202619


namespace gcd_max_possible_value_l202_202214

theorem gcd_max_possible_value (x y : ℤ) (h_coprime : Int.gcd x y = 1) : 
  ∃ d, d = Int.gcd (x + 2015 * y) (y + 2015 * x) ∧ d = 4060224 :=
by
  sorry

end gcd_max_possible_value_l202_202214


namespace calc1_calc2_calc3_calc4_calc5_calc6_l202_202499

theorem calc1 : 320 + 16 * 27 = 752 :=
by
  -- Proof goes here
  sorry

theorem calc2 : 1500 - 125 * 8 = 500 :=
by
  -- Proof goes here
  sorry

theorem calc3 : 22 * 22 - 84 = 400 :=
by
  -- Proof goes here
  sorry

theorem calc4 : 25 * 8 * 9 = 1800 :=
by
  -- Proof goes here
  sorry

theorem calc5 : (25 + 38) * 15 = 945 :=
by
  -- Proof goes here
  sorry

theorem calc6 : (62 + 12) * 38 = 2812 :=
by
  -- Proof goes here
  sorry

end calc1_calc2_calc3_calc4_calc5_calc6_l202_202499


namespace car_travel_speed_l202_202578

theorem car_travel_speed (v : ℝ) : 
  (1 / 60) * 3600 + 5 = (1 / v) * 3600 → v = 65 := 
by
  intros h
  sorry

end car_travel_speed_l202_202578


namespace mabel_tomatoes_l202_202649

theorem mabel_tomatoes (x : ℕ)
  (plant_1_bore : ℕ)
  (plant_2_bore : ℕ := x + 4)
  (total_first_two_plants : ℕ := x + plant_2_bore)
  (plant_3_bore : ℕ := 3 * total_first_two_plants)
  (plant_4_bore : ℕ := 3 * total_first_two_plants)
  (total_tomatoes : ℕ)
  (h1 : total_first_two_plants = 2 * x + 4)
  (h2 : plant_3_bore = 3 * (2 * x + 4))
  (h3 : plant_4_bore = 3 * (2 * x + 4))
  (h4 : total_tomatoes = x + plant_2_bore + plant_3_bore + plant_4_bore)
  (h5 : total_tomatoes = 140) :
   x = 8 :=
by
  sorry

end mabel_tomatoes_l202_202649


namespace bridge_weight_excess_l202_202183

theorem bridge_weight_excess :
  ∀ (Kelly_weight Megan_weight Mike_weight : ℕ),
  Kelly_weight = 34 →
  Kelly_weight = 85 * Megan_weight / 100 →
  Mike_weight = Megan_weight + 5 →
  (Kelly_weight + Megan_weight + Mike_weight) - 100 = 19 :=
by
  intros Kelly_weight Megan_weight Mike_weight
  intros h1 h2 h3
  sorry

end bridge_weight_excess_l202_202183


namespace ratio_of_larger_to_smaller_l202_202637

theorem ratio_of_larger_to_smaller (x y : ℝ) (hx_pos : 0 < x) (hy_pos : 0 < y) (hx_gt_hy : x > y) (h_eq : x + y = 7 * (x - y)) : x / y = 4 / 3 := 
by
  sorry

end ratio_of_larger_to_smaller_l202_202637


namespace first_chinese_supercomputer_is_milkyway_l202_202012

-- Define the names of the computers
inductive ComputerName
| Universe
| Taihu
| MilkyWay
| Dawn

-- Define a structure to hold the properties of the computer
structure Computer :=
  (name : ComputerName)
  (introduction_year : Nat)
  (calculations_per_second : Nat)

-- Define the properties of the specific computer in the problem
def first_chinese_supercomputer := 
  Computer.mk ComputerName.MilkyWay 1983 100000000

-- The theorem to be proven
theorem first_chinese_supercomputer_is_milkyway :
  first_chinese_supercomputer.name = ComputerName.MilkyWay :=
by
  -- Provide the conditions that lead to the conclusion (proof steps will be added here)
  sorry

end first_chinese_supercomputer_is_milkyway_l202_202012


namespace jericho_altitude_300_l202_202373

def jericho_altitude (below_sea_level : Int) : Prop :=
  below_sea_level = -300

theorem jericho_altitude_300 (below_sea_level : Int)
  (h1 : below_sea_level = -300) : jericho_altitude below_sea_level :=
by
  sorry

end jericho_altitude_300_l202_202373


namespace subset_P1_P2_l202_202446

def P1 (a : ℝ) : Set ℝ := {x | x^2 + a*x + 1 > 0}
def P2 (a : ℝ) : Set ℝ := {x | x^2 + a*x + 2 > 0}

theorem subset_P1_P2 (a : ℝ) : P1 a ⊆ P2 a :=
by intros x hx; sorry

end subset_P1_P2_l202_202446


namespace divisibility_of_3_pow_p_minus_2_pow_p_minus_1_l202_202641

theorem divisibility_of_3_pow_p_minus_2_pow_p_minus_1 (p : ℕ) (hp : Nat.Prime p) (hp_gt_3 : p > 3) : 
  (3^p - 2^p - 1) % (42 * p) = 0 := 
by
  sorry

end divisibility_of_3_pow_p_minus_2_pow_p_minus_1_l202_202641


namespace fraction_of_passengers_from_Africa_l202_202951

theorem fraction_of_passengers_from_Africa :
  (1/4 + 1/8 + 1/6 + A + 36/96 = 1) → (96 - 36) = (11/24 * 96) → 
  A = 1/12 :=
by
  sorry

end fraction_of_passengers_from_Africa_l202_202951


namespace quadratic_form_ratio_l202_202059

theorem quadratic_form_ratio (x y u v : ℤ) (h : ∃ k : ℤ, k * (u^2 + 3*v^2) = x^2 + 3*y^2) :
  ∃ a b : ℤ, (x^2 + 3*y^2) / (u^2 + 3*v^2) = a^2 + 3*b^2 := sorry

end quadratic_form_ratio_l202_202059


namespace frog_arrangements_l202_202320

theorem frog_arrangements :
  let total_frogs := 7
  let green_frogs := 2
  let red_frogs := 3
  let blue_frogs := 2
  let valid_sequences := 4
  let green_permutations := Nat.factorial green_frogs
  let red_permutations := Nat.factorial red_frogs
  let blue_permutations := Nat.factorial blue_frogs
  let total_permutations := valid_sequences * (green_permutations * red_permutations * blue_permutations)
  total_frogs = green_frogs + red_frogs + blue_frogs → 
  green_frogs = 2 ∧ red_frogs = 3 ∧ blue_frogs = 2 →
  valid_sequences = 4 →
  total_permutations = 96 := 
by
  -- Given conditions lead to the calculation of total permutations 
  sorry

end frog_arrangements_l202_202320


namespace mass_of_15_moles_is_9996_9_l202_202082

/-- Calculation of the molar mass of potassium aluminum sulfate dodecahydrate -/
def KAl_SO4_2_12H2O_molar_mass : ℝ :=
  let K := 39.10
  let Al := 26.98
  let S := 32.07
  let O := 16.00
  let H := 1.01
  K + Al + 2 * S + (8 + 24) * O + 24 * H

/-- Mass calculation for 15 moles of potassium aluminum sulfate dodecahydrate -/
def mass_of_15_moles_KAl_SO4_2_12H2O : ℝ :=
  15 * KAl_SO4_2_12H2O_molar_mass

/-- Proof statement that the mass of 15 moles of potassium aluminum sulfate dodecahydrate is 9996.9 grams -/
theorem mass_of_15_moles_is_9996_9 : mass_of_15_moles_KAl_SO4_2_12H2O = 9996.9 := by
  -- assume KAl_SO4_2_12H2O_molar_mass = 666.46 (from the problem solution steps)
  sorry

end mass_of_15_moles_is_9996_9_l202_202082


namespace smallest_whole_number_gt_total_sum_l202_202176

-- Declarations of the fractions involved
def term1 : ℚ := 3 + 1/3
def term2 : ℚ := 4 + 1/6
def term3 : ℚ := 5 + 1/12
def term4 : ℚ := 6 + 1/8

-- Definition of the entire sum
def total_sum : ℚ := term1 + term2 + term3 + term4

-- Statement of the theorem
theorem smallest_whole_number_gt_total_sum : 
  ∀ n : ℕ, (n > total_sum) → (∀ m : ℕ, (m >= 0) → (m > total_sum) → (n ≤ m)) → n = 19 := by
  sorry -- the proof is omitted

end smallest_whole_number_gt_total_sum_l202_202176


namespace oil_already_put_in_engine_l202_202811

def oil_per_cylinder : ℕ := 8
def cylinders : ℕ := 6
def additional_needed_oil : ℕ := 32

theorem oil_already_put_in_engine :
  (oil_per_cylinder * cylinders) - additional_needed_oil = 16 := by
  sorry

end oil_already_put_in_engine_l202_202811


namespace increase_in_area_l202_202563

theorem increase_in_area :
  let original_side := 6
  let increase := 1
  let new_side := original_side + increase
  let original_area := original_side * original_side
  let new_area := new_side * new_side
  let area_increase := new_area - original_area
  area_increase = 13 :=
by
  let original_side := 6
  let increase := 1
  let new_side := original_side + increase
  let original_area := original_side * original_side
  let new_area := new_side * new_side
  let area_increase := new_area - original_area
  sorry

end increase_in_area_l202_202563


namespace largest_binomial_coefficient_term_largest_coefficient_term_remainder_mod_7_l202_202425

-- Definitions and conditions
def binomial_expansion (x : ℕ) (n : ℕ) : ℕ := (x + (3 * x^2))^n

-- Problem statements
theorem largest_binomial_coefficient_term :
  ∀ x n,
  (x + 3 * x^2)^n = (x + 3 * x^2)^7 →
  (2^n = 128) →
  ∃ t : ℕ, t = 2835 * x^11 := 
by sorry

theorem largest_coefficient_term :
  ∀ x n,
  (x + 3 * x^2)^n = (x + 3 * x^2)^7 →
  exists t, t = 5103 * x^13 :=
by sorry

theorem remainder_mod_7 :
  ∀ x n,
  x = 3 →
  n = 2016 →
  (x + (3 * x^2))^n % 7 = 1 :=
by sorry

end largest_binomial_coefficient_term_largest_coefficient_term_remainder_mod_7_l202_202425


namespace sample_size_9_l202_202845

variable (X : Nat)

theorem sample_size_9 (h : 36 % X = 0 ∧ 36 % (X + 1) ≠ 0) : X = 9 := 
sorry

end sample_size_9_l202_202845


namespace cyclist_speed_ratio_l202_202554

-- Define the conditions
def speeds_towards_each_other (v1 v2 : ℚ) : Prop :=
  v1 + v2 = 25

def speeds_apart_with_offset (v1 v2 : ℚ) : Prop :=
  v1 - v2 = 10 / 3

-- The proof problem to show the required ratio of speeds
theorem cyclist_speed_ratio (v1 v2 : ℚ) (h1 : speeds_towards_each_other v1 v2) (h2 : speeds_apart_with_offset v1 v2) :
  v1 / v2 = 17 / 13 :=
sorry

end cyclist_speed_ratio_l202_202554


namespace rectangular_solid_length_l202_202124

theorem rectangular_solid_length (w h : ℕ) (surface_area : ℕ) (l : ℕ) 
  (hw : w = 4) (hh : h = 1) (hsa : surface_area = 58) 
  (h_surface_area_formula : surface_area = 2 * l * w + 2 * l * h + 2 * w * h) : 
  l = 5 :=
by
  rw [hw, hh, hsa] at h_surface_area_formula
  sorry

end rectangular_solid_length_l202_202124


namespace container_marbles_volume_l202_202824

theorem container_marbles_volume {V₁ V₂ m₁ m₂ : ℕ} 
  (h₁ : V₁ = 24) (h₂ : m₁ = 75) (h₃ : V₂ = 72) :
  m₂ = 225 :=
by
  have proportion := (m₁ : ℚ) / V₁
  have proportion2 := (m₂ : ℚ) / V₂
  have h4 := proportion = proportion2
  sorry

end container_marbles_volume_l202_202824


namespace first_book_length_l202_202432

-- Statement of the problem
theorem first_book_length
  (x : ℕ) -- Number of pages in the first book
  (total_pages : ℕ)
  (days_in_two_weeks : ℕ)
  (pages_per_day : ℕ)
  (second_book_pages : ℕ := 100) :
  pages_per_day = 20 ∧ days_in_two_weeks = 14 ∧ total_pages = 280 ∧ total_pages = pages_per_day * days_in_two_weeks ∧ total_pages = x + second_book_pages → x = 180 :=
by
  sorry

end first_book_length_l202_202432


namespace right_triangle_side_length_l202_202792

theorem right_triangle_side_length (c a b : ℕ) (h1 : c = 5) (h2 : a = 3) (h3 : c^2 = a^2 + b^2) : b = 4 :=
  by
  sorry

end right_triangle_side_length_l202_202792


namespace find_ctg_half_l202_202869

noncomputable def ctg (x : ℝ) := 1 / (Real.tan x)

theorem find_ctg_half
  (x : ℝ)
  (h : Real.sin x - Real.cos x = (1 + 2 * Real.sqrt 2) / 3) :
  ctg (x / 2) = Real.sqrt 2 / 2 ∨ ctg (x / 2) = 3 - 2 * Real.sqrt 2 :=
by
  sorry

end find_ctg_half_l202_202869


namespace extreme_point_l202_202555

noncomputable def f (x : ℝ) : ℝ := (x^4 / 4) - (x^3 / 3)
noncomputable def f_prime (x : ℝ) : ℝ := deriv f x

theorem extreme_point (x : ℝ) : f_prime 1 = 0 ∧
  (∀ y, y < 1 → f_prime y < 0) ∧
  (∀ z, z > 1 → f_prime z > 0) :=
by
  sorry

end extreme_point_l202_202555


namespace workman_problem_l202_202610

theorem workman_problem 
  {A B : Type}
  (W : ℕ)
  (RA RB : ℝ)
  (h1 : RA = (1/2) * RB)
  (h2 : RA + RB = W / 14)
  : W / RB = 21 :=
by
  sorry

end workman_problem_l202_202610


namespace cyclic_sum_fraction_ge_one_l202_202455

theorem cyclic_sum_fraction_ge_one (a b c : ℝ) 
  (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0)
  (hineq : (a/(b+c+1) + b/(c+a+1) + c/(a+b+1)) ≤ 1) :
  (1/(b+c+1) + 1/(c+a+1) + 1/(a+b+1)) ≥ 1 :=
by sorry

end cyclic_sum_fraction_ge_one_l202_202455


namespace exists_infinite_irregular_set_l202_202143

def is_irregular (A : Set ℤ) :=
  ∀ ⦃x y : ℤ⦄, x ∈ A → y ∈ A → x ≠ y → ∀ ⦃k : ℤ⦄, x + k * (y - x) ≠ x ∧ x + k * (y - x) ≠ y

theorem exists_infinite_irregular_set : ∃ A : Set ℤ, Set.Infinite A ∧ is_irregular A :=
sorry

end exists_infinite_irregular_set_l202_202143


namespace intersect_complement_A_B_eq_l202_202131

open Set

variable (U : Set ℝ)
variable (A : Set ℝ)
variable (B : Set ℝ)

noncomputable def complement_A : Set ℝ := U \ A
noncomputable def intersection_complement_A_B : Set ℝ := complement_A U A ∩ B

theorem intersect_complement_A_B_eq : 
  U = univ ∧ A = {x : ℝ | x + 1 < 0} ∧ B = {x : ℝ | x - 3 < 0} →
  intersection_complement_A_B U A B = Icc (-1 : ℝ) 3 :=
by
  intro h
  sorry

end intersect_complement_A_B_eq_l202_202131


namespace ken_gave_manny_10_pencils_l202_202422

theorem ken_gave_manny_10_pencils (M : ℕ) 
  (ken_pencils : ℕ := 50)
  (ken_kept : ℕ := 20)
  (ken_distributed : ℕ := ken_pencils - ken_kept)
  (nilo_pencils : ℕ := M + 10)
  (distribution_eq : M + nilo_pencils = ken_distributed) : 
  M = 10 :=
by
  sorry

end ken_gave_manny_10_pencils_l202_202422


namespace probability_multiple_of_3_when_die_rolled_twice_l202_202832

theorem probability_multiple_of_3_when_die_rolled_twice :
  let total_outcomes := 36
  let favorable_outcomes := 12
  (12 / 36 : ℚ) = 1 / 3 :=
by
  sorry

end probability_multiple_of_3_when_die_rolled_twice_l202_202832


namespace algebraic_expression_decrease_l202_202463

theorem algebraic_expression_decrease (x y : ℝ) :
  let original_expr := 2 * x^2 * y
  let new_expr := 2 * ((1 / 2) * x) ^ 2 * ((1 / 2) * y)
  let decrease := ((original_expr - new_expr) / original_expr) * 100
  decrease = 87.5 := by
  sorry

end algebraic_expression_decrease_l202_202463


namespace king_luis_courtiers_are_odd_l202_202123

theorem king_luis_courtiers_are_odd (n : ℕ) 
  (h : ∀ i : ℕ, i < n → ∃ j : ℕ, j < n ∧ i ≠ j) : 
  ¬ Even n := 
sorry

end king_luis_courtiers_are_odd_l202_202123


namespace cost_effective_plan1_l202_202039

/-- 
Plan 1 involves purchasing a 80 yuan card and a subsequent fee of 10 yuan per session.
Plan 2 involves a fee of 20 yuan per session without purchasing the card.
We want to prove that Plan 1 is more cost-effective than Plan 2 for any number of sessions x > 8.
-/
theorem cost_effective_plan1 (x : ℕ) (h : x > 8) : 
  10 * x + 80 < 20 * x :=
sorry

end cost_effective_plan1_l202_202039


namespace greatest_divisor_remainders_l202_202434

theorem greatest_divisor_remainders (d : ℤ) :
  d > 0 → (1657 % d = 10) → (2037 % d = 7) → d = 1 :=
by
  intros hdg h1657 h2037
  sorry

end greatest_divisor_remainders_l202_202434


namespace proof_problem_l202_202679

variable (α β : ℝ)

def interval_αβ : Prop := 
  α ∈ Set.Icc (-Real.pi / 2) (Real.pi / 2) ∧ 
  β ∈ Set.Icc (-Real.pi / 2) (Real.pi / 2)

def condition : Prop := α * Real.sin α - β * Real.sin β > 0

theorem proof_problem (h1 : interval_αβ α β) (h2 : condition α β) : α ^ 2 > β ^ 2 := 
sorry

end proof_problem_l202_202679


namespace evaluate_expression_l202_202522

theorem evaluate_expression (a : ℚ) (h : a = 3/2) : 
  ((5 * a^2 - 13 * a + 4) * (2 * a - 3)) = 0 := by
  sorry

end evaluate_expression_l202_202522


namespace max_sum_xy_l202_202879

theorem max_sum_xy (x y : ℤ) (h1 : x^2 + y^2 = 64) (h2 : x ≥ 0) (h3 : y ≥ 0) : x + y ≤ 8 :=
by sorry

end max_sum_xy_l202_202879


namespace find_coefficients_l202_202965

theorem find_coefficients (a1 a2 : ℚ) :
  (4 * a1 + 5 * a2 = 9) ∧ (-a1 + 3 * a2 = 4) ↔ (a1 = 181 / 136) ∧ (a2 = 25 / 68) := 
sorry

end find_coefficients_l202_202965


namespace K1K2_eq_one_over_four_l202_202889

theorem K1K2_eq_one_over_four
  (K1 : ℝ) (hK1 : K1 ≠ 0)
  (K2 : ℝ)
  (x1 y1 x2 y2 : ℝ)
  (hx1y1 : x1^2 - 4 * y1^2 = 4)
  (hx2y2 : x2^2 - 4 * y2^2 = 4)
  (hx0 : x0 = (x1 + x2) / 2)
  (hy0 : y0 = (y1 + y2) / 2)
  (K1_eq : K1 = (y1 - y2) / (x1 - x2))
  (K2_eq : K2 = y0 / x0) :
  K1 * K2 = 1 / 4 :=
sorry

end K1K2_eq_one_over_four_l202_202889


namespace batman_game_cost_l202_202683

theorem batman_game_cost (total_spent superman_cost : ℝ) 
  (H1 : total_spent = 18.66) (H2 : superman_cost = 5.06) :
  total_spent - superman_cost = 13.60 :=
by
  sorry

end batman_game_cost_l202_202683


namespace inequality_solution_l202_202177

theorem inequality_solution (a x : ℝ) (h : |a + 1| < 3) :
  (-4 < a ∧ a < -2 ∧ (x > -1 ∨ x < 1 + a)) ∨ 
  (a = -2 ∧ (x ∈ Set.univ \ {-1})) ∨ 
  (-2 < a ∧ a < 2 ∧ (x > 1 + a ∨ x < -1)) :=
by sorry

end inequality_solution_l202_202177


namespace percent_increase_correct_l202_202895

noncomputable def last_year_ticket_price : ℝ := 85
noncomputable def last_year_tax_rate : ℝ := 0.10
noncomputable def this_year_ticket_price : ℝ := 102
noncomputable def this_year_tax_rate : ℝ := 0.12
noncomputable def student_discount_rate : ℝ := 0.15

noncomputable def last_year_total_cost : ℝ := last_year_ticket_price * (1 + last_year_tax_rate)
noncomputable def discounted_ticket_price_this_year : ℝ := this_year_ticket_price * (1 - student_discount_rate)
noncomputable def total_cost_this_year : ℝ := discounted_ticket_price_this_year * (1 + this_year_tax_rate)

noncomputable def percent_increase : ℝ := ((total_cost_this_year - last_year_total_cost) / last_year_total_cost) * 100

theorem percent_increase_correct :
  abs (percent_increase - 3.854) < 0.001 := sorry

end percent_increase_correct_l202_202895


namespace chickens_do_not_lay_eggs_l202_202955

theorem chickens_do_not_lay_eggs (total_chickens : ℕ) 
  (roosters : ℕ) (hens : ℕ) (hens_lay_eggs : ℕ) (hens_do_not_lay_eggs : ℕ) 
  (chickens_do_not_lay_eggs : ℕ) :
  total_chickens = 80 →
  roosters = total_chickens / 4 →
  hens = total_chickens - roosters →
  hens_lay_eggs = 3 * hens / 4 →
  hens_do_not_lay_eggs = hens - hens_lay_eggs →
  chickens_do_not_lay_eggs = hens_do_not_lay_eggs + roosters →
  chickens_do_not_lay_eggs = 35 :=
by
  intros h0 h1 h2 h3 h4 h5
  sorry

end chickens_do_not_lay_eggs_l202_202955


namespace vector_parallel_dot_product_l202_202114

theorem vector_parallel_dot_product (x : ℝ) 
  (a : ℝ × ℝ) (b : ℝ × ℝ)
  (h1 : a = (x, 1))
  (h2 : b = (4, 2))
  (h3 : x / 4 = 1 / 2) : 
  (a.1 * (b.1 - a.1) + a.2 * (b.2 - a.2)) = 5 := 
by 
  sorry

end vector_parallel_dot_product_l202_202114


namespace total_cost_rental_l202_202272

theorem total_cost_rental :
  let rental_fee := 20.99
  let charge_per_mile := 0.25
  let miles_driven := 299
  let total_cost := rental_fee + charge_per_mile * miles_driven
  total_cost = 95.74 := by
{
  sorry
}

end total_cost_rental_l202_202272


namespace Frank_seeds_per_orange_l202_202236

noncomputable def Betty_oranges := 15
noncomputable def Bill_oranges := 12
noncomputable def total_oranges := Betty_oranges + Bill_oranges
noncomputable def Frank_oranges := 3 * total_oranges
noncomputable def oranges_per_tree := 5
noncomputable def Philip_oranges := 810
noncomputable def number_of_trees := Philip_oranges / oranges_per_tree
noncomputable def seeds_per_orange := number_of_trees / Frank_oranges

theorem Frank_seeds_per_orange :
  seeds_per_orange = 2 :=
by
  sorry

end Frank_seeds_per_orange_l202_202236


namespace cost_of_article_l202_202500

variable (C : ℝ) 
variable (G : ℝ)
variable (H1 : G = 380 - C)
variable (H2 : 1.05 * G = 420 - C)

theorem cost_of_article : C = 420 :=
by
  sorry

end cost_of_article_l202_202500


namespace mean_of_remaining_three_numbers_l202_202173

theorem mean_of_remaining_three_numbers 
    (a b c d : ℝ)
    (h₁ : (a + b + c + d) / 4 = 92)
    (h₂ : d = 120)
    (h₃ : b = 60) : 
    (a + b + c) / 3 = 82.6666666666 := 
by 
    -- This state suggests adding the constraints added so far for the proof:
    sorry

end mean_of_remaining_three_numbers_l202_202173


namespace trigonometric_identity_proof_l202_202232

theorem trigonometric_identity_proof :
  ( (Real.cos (40 * Real.pi / 180) + Real.sin (50 * Real.pi / 180) * (1 + Real.sqrt 3 * Real.tan (10 * Real.pi / 180)))
  / (Real.sin (70 * Real.pi / 180) * Real.sqrt (1 + Real.cos (40 * Real.pi / 180))) ) =
  Real.sqrt 2 :=
by
  sorry

end trigonometric_identity_proof_l202_202232


namespace solve_bx2_ax_1_lt_0_l202_202347

noncomputable def quadratic_inequality_solution (a b : ℝ) (x : ℝ) : Prop :=
  x^2 + a * x + b > 0

theorem solve_bx2_ax_1_lt_0 (a b : ℝ) :
  (∀ x : ℝ, quadratic_inequality_solution a b x ↔ (x < -2 ∨ x > -1/2)) →
  (∀ x : ℝ, (x = -2 ∨ x = -1/2) → x^2 + a * x + b = 0) →
  (b * x^2 + a * x + 1 < 0) ↔ (-2 < x ∧ x < -1/2) :=
by
  sorry

end solve_bx2_ax_1_lt_0_l202_202347


namespace max_height_reached_l202_202092

def h (t : ℝ) : ℝ := -20 * t^2 + 100 * t + 36

theorem max_height_reached : ∃ t : ℝ, h t = 161 :=
by
  sorry

end max_height_reached_l202_202092


namespace simplify_expression_l202_202022

theorem simplify_expression : |(-5^2 - 6 * 2)| = 37 := by
  sorry

end simplify_expression_l202_202022


namespace union_complement_A_eq_l202_202921

open Set

universe u

noncomputable def U : Set ℝ := univ
def A : Set ℝ := { x | x < 2 }
def B : Set ℝ := { y | ∃ (x : ℝ), y = x^2 + 1 }

theorem union_complement_A_eq :
  A ∪ ((U \ B : Set ℝ) : Set ℝ) = { x | x < 2 } := by
  sorry

end union_complement_A_eq_l202_202921


namespace solve_for_n_l202_202519

theorem solve_for_n (n : ℕ) : (3^n * 3^n * 3^n * 3^n = 81^2) → n = 2 :=
by
  sorry

end solve_for_n_l202_202519


namespace equilateral_triangle_sum_l202_202162

theorem equilateral_triangle_sum (side_length : ℚ) (h_eq : side_length = 13 / 12) :
  3 * side_length = 13 / 4 :=
by
  -- Proof omitted
  sorry

end equilateral_triangle_sum_l202_202162


namespace ratio_population_XZ_l202_202773

variable (Population : Type) [Field Population]
variable (Z : Population) -- Population of City Z
variable (Y : Population) -- Population of City Y
variable (X : Population) -- Population of City X

-- Conditions
def population_Y : Y = 2 * Z := sorry
def population_X : X = 7 * Y := sorry

-- Theorem stating the ratio of populations
theorem ratio_population_XZ : (X / Z) = 14 := by
  -- The proof will use the conditions population_Y and population_X
  sorry

end ratio_population_XZ_l202_202773


namespace correct_time_fraction_l202_202478

theorem correct_time_fraction : 
  (∀ hour : ℕ, hour < 24 → true) →
  (∀ minute : ℕ, minute < 60 → (minute ≠ 16)) →
  (fraction_of_correct_time = 59 / 60) :=
by
  intros h_hour h_minute
  sorry

end correct_time_fraction_l202_202478


namespace value_of_a_l202_202556

theorem value_of_a (a : ℝ) (h : (2 : ℝ)^a = (1 / 2 : ℝ)) : a = -1 := 
sorry

end value_of_a_l202_202556


namespace supplement_of_complementary_angle_l202_202527

theorem supplement_of_complementary_angle (α β : ℝ) 
  (h1 : α + β = 90) (h2 : α = 30) : 180 - β = 120 :=
by sorry

end supplement_of_complementary_angle_l202_202527


namespace dilation_image_l202_202224

open Complex

noncomputable def dilation_center := (1 : ℂ) + (3 : ℂ) * I
noncomputable def scale_factor := -3
noncomputable def initial_point := -I
noncomputable def target_point := (4 : ℂ) + (15 : ℂ) * I

theorem dilation_image :
  let c := dilation_center
  let k := scale_factor
  let z := initial_point
  let z_prime := target_point
  z_prime = c + k * (z - c) := 
  by
    sorry

end dilation_image_l202_202224


namespace sequence_equality_l202_202906

theorem sequence_equality (a : ℕ → ℤ) (h : ∀ n, a (n + 2) ^ 2 + a (n + 1) * a n ≤ a (n + 2) * (a (n + 1) + a n)) :
  ∃ N : ℕ, ∀ n ≥ N, a (n + 2) = a n :=
by sorry

end sequence_equality_l202_202906


namespace ellipse_properties_l202_202032

theorem ellipse_properties (h k a b : ℝ) (θ : ℝ)
  (h_def : h = -2)
  (k_def : k = 3)
  (a_def : a = 6)
  (b_def : b = 4)
  (θ_def : θ = 45) :
  h + k + a + b = 11 :=
by
  sorry

end ellipse_properties_l202_202032


namespace BrotherUpperLimit_l202_202037

variable (w : ℝ) -- Arun's weight
variable (b : ℝ) -- Upper limit of Arun's weight according to his brother's opinion

-- Conditions as per the problem
def ArunOpinion (w : ℝ) := 64 < w ∧ w < 72
def BrotherOpinion (w b : ℝ) := 60 < w ∧ w < b
def MotherOpinion (w : ℝ) := w ≤ 67

-- The average of probable weights
def AverageWeight (weights : Set ℝ) (avg : ℝ) := (∀ w ∈ weights, 64 < w ∧ w ≤ 67) ∧ avg = 66

-- The main theorem to be proven
theorem BrotherUpperLimit (hA : ArunOpinion w) (hB : BrotherOpinion w b) (hM : MotherOpinion w) (hAvg : AverageWeight {w | 64 < w ∧ w ≤ 67} 66) : b = 67 := by
  sorry

end BrotherUpperLimit_l202_202037


namespace socks_selection_l202_202314

/-!
  # Socks Selection Problem
  Prove the total number of ways to choose a pair of socks of different colors
  given:
  1. there are 5 white socks,
  2. there are 4 brown socks,
  3. there are 3 blue socks,
  is equal to 47.
-/

theorem socks_selection : 
  let white_socks := 5
  let brown_socks := 4
  let blue_socks := 3
  5 * 4 + 4 * 3 + 5 * 3 = 47 :=
by
  let white_socks := 5
  let brown_socks := 4
  let blue_socks := 3
  sorry

end socks_selection_l202_202314


namespace powerjet_pumps_250_gallons_in_30_minutes_l202_202153

theorem powerjet_pumps_250_gallons_in_30_minutes :
  let rate : ℝ := 500
  let time_in_hours : ℝ := 1 / 2
  rate * time_in_hours = 250 :=
by
  sorry

end powerjet_pumps_250_gallons_in_30_minutes_l202_202153


namespace other_acute_angle_of_right_triangle_l202_202920

theorem other_acute_angle_of_right_triangle (a : ℝ) (h₀ : 0 < a ∧ a < 90) (h₁ : a = 20) :
  ∃ b, b = 90 - a ∧ b = 70 := by
    sorry

end other_acute_angle_of_right_triangle_l202_202920


namespace equivalent_proof_problem_l202_202066

theorem equivalent_proof_problem (x : ℤ) (h : (x + 2) * (x - 2) = 1221) :
    (x = 35 ∨ x = -35) ∧ ((x + 1) * (x - 1) = 1224) :=
sorry

end equivalent_proof_problem_l202_202066


namespace sum_of_two_squares_l202_202560

theorem sum_of_two_squares (n : ℕ) (h : ∀ m, m = n → n = 2 ∨ (n = 2 * 10 + m) → n % 8 = m) :
  (∃ a b : ℕ, n = a^2 + b^2) ↔ n = 2 := by
  sorry

end sum_of_two_squares_l202_202560


namespace cyrus_shots_percentage_l202_202715

theorem cyrus_shots_percentage (total_shots : ℕ) (missed_shots : ℕ) (made_shots : ℕ)
  (h_total : total_shots = 20)
  (h_missed : missed_shots = 4)
  (h_made : made_shots = total_shots - missed_shots) :
  (made_shots / total_shots : ℚ) * 100 = 80 := by
  sorry

end cyrus_shots_percentage_l202_202715


namespace ninety_eight_times_ninety_eight_l202_202598

theorem ninety_eight_times_ninety_eight : 98 * 98 = 9604 :=
by
  sorry

end ninety_eight_times_ninety_eight_l202_202598


namespace polynomial_expansion_identity_l202_202774

theorem polynomial_expansion_identity
  (a a1 a3 a4 a5 : ℝ)
  (h : (a - x)^5 = a + a1 * x + 80 * x^2 + a3 * x^3 + a4 * x^4 + a5 * x^5) :
  a + a1 + 80 + a3 + a4 + a5 = 1 := 
sorry

end polynomial_expansion_identity_l202_202774


namespace extremum_and_equal_values_l202_202169

theorem extremum_and_equal_values {f : ℝ → ℝ} {a b x_0 x_1 : ℝ} 
    (hf : ∀ x, f x = (x - 1)^3 - a * x + b)
    (h'x0 : deriv f x_0 = 0)
    (hfx1_eq_fx0 : f x_1 = f x_0)
    (hx1_ne_x0 : x_1 ≠ x_0) :
  x_1 + 2 * x_0 = 3 := sorry

end extremum_and_equal_values_l202_202169


namespace water_level_after_opening_valve_l202_202836

-- Define the initial conditions and final height to be proved
def initial_water_height_cm : ℝ := 40
def initial_oil_height_cm : ℝ := 40
def water_density : ℝ := 1000
def oil_density : ℝ := 700
def final_water_height_cm : ℝ := 34

-- The proof that the final height of water after equilibrium will be 34 cm
theorem water_level_after_opening_valve :
  ∀ (h_w h_o : ℝ),
  (water_density * h_w = oil_density * h_o) ∧ (h_w + h_o = initial_water_height_cm + initial_oil_height_cm) →
  h_w = final_water_height_cm :=
by
  -- Here goes the proof, skipped with sorry
  sorry

end water_level_after_opening_valve_l202_202836


namespace arrangement_count_27_arrangement_count_26_l202_202604

open Int

def valid_arrangement_count (n : ℕ) : ℕ :=
  if n = 27 then 14 else if n = 26 then 105 else 0

theorem arrangement_count_27 : valid_arrangement_count 27 = 14 :=
  by
    sorry

theorem arrangement_count_26 : valid_arrangement_count 26 = 105 :=
  by
    sorry

end arrangement_count_27_arrangement_count_26_l202_202604


namespace horizontal_asymptote_exists_x_intercepts_are_roots_l202_202648

noncomputable def given_function (x : ℝ) : ℝ :=
  (15 * x^4 + 6 * x^3 + 7 * x^2 + 4 * x + 5) / (5 * x^5 + 3 * x^3 + 9 * x^2 + 2 * x + 4)

theorem horizontal_asymptote_exists :
  ∃ L : ℝ, ∀ x : ℝ, (∃ M : ℝ, M > 0 ∧ (∀ x > M, abs (given_function x - L) < 1)) ∧ L = 0 := 
sorry

theorem x_intercepts_are_roots :
  ∀ y, y = 0 ↔ ∃ x : ℝ, x ≠ 0 ∧ 15 * x^4 + 6 * x^3 + 7 * x^2 + 4 * x + 5 = 0 :=
sorry

end horizontal_asymptote_exists_x_intercepts_are_roots_l202_202648


namespace quadratic_roots_real_distinct_l202_202762

theorem quadratic_roots_real_distinct (k : ℝ) (h : k < 0) :
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x1^2 + x1 + k - 1 = 0) ∧ (x2^2 + x2 + k - 1 = 0) :=
by
  sorry

end quadratic_roots_real_distinct_l202_202762


namespace Rachel_plant_arrangement_l202_202568

-- We define Rachel's plants and lamps
inductive Plant : Type
| basil1
| basil2
| aloe
| cactus

inductive Lamp : Type
| white1
| white2
| red1
| red2

def arrangements (plants : List Plant) (lamps : List Lamp) : Nat :=
  -- This would be the function counting all valid arrangements
  -- I'm skipping the implementation
  sorry

def Rachel_arrangement_count : Nat :=
  arrangements [Plant.basil1, Plant.basil2, Plant.aloe, Plant.cactus]
                [Lamp.white1, Lamp.white2, Lamp.red1, Lamp.red2]

theorem Rachel_plant_arrangement : Rachel_arrangement_count = 22 := by
  sorry

end Rachel_plant_arrangement_l202_202568


namespace prove_range_of_a_l202_202424

noncomputable def f (x a : ℝ) := x^2 + (a + 1) * x + Real.log (abs (a + 2))

def is_increasing (f : ℝ → ℝ) (interval : Set ℝ) :=
 ∀ ⦃x y⦄, x ∈ interval → y ∈ interval → x ≤ y → f x ≤ f y

def g (x a : ℝ) := (a + 1) * x
def is_decreasing (g : ℝ → ℝ) :=
 ∀ ⦃x y⦄, x ≤ y → g y ≤ g x

def proposition_p (a : ℝ) : Prop :=
  is_increasing (f a) (Set.Ici ((a + 1)^2))

def proposition_q (a : ℝ) : Prop :=
  is_decreasing (g a)

theorem prove_range_of_a (a : ℝ) (h : ¬ (proposition_p a ↔ proposition_q a)) :
  a > -3 / 2 :=
sorry

end prove_range_of_a_l202_202424


namespace average_vegetables_per_week_l202_202492

theorem average_vegetables_per_week (P Vp S W : ℕ) (h1 : P = 200) (h2 : Vp = 2) (h3 : S = 25) (h4 : W = 2) :
  (P / Vp) / S / W = 2 :=
by
  sorry

end average_vegetables_per_week_l202_202492


namespace alex_total_earnings_l202_202795

def total_earnings (hours_w1 hours_w2 wage : ℕ) : ℕ :=
  (hours_w1 + hours_w2) * wage

theorem alex_total_earnings
  (hours_w1 hours_w2 wage : ℕ)
  (h1 : hours_w1 = 28)
  (h2 : hours_w2 = hours_w1 - 10)
  (h3 : wage * 10 = 80) :
  total_earnings hours_w1 hours_w2 wage = 368 :=
by
  sorry

end alex_total_earnings_l202_202795


namespace set_B_roster_method_l202_202234

def A : Set ℤ := {-2, 2, 3, 4}
def B : Set ℤ := {x | ∃ t ∈ A, x = t^2}

theorem set_B_roster_method : B = {4, 9, 16} :=
by
  sorry

end set_B_roster_method_l202_202234


namespace number_of_schools_is_8_l202_202509

-- Define the number of students trying out and not picked per school
def students_trying_out := 65.0
def students_not_picked := 17.0
def students_picked := students_trying_out - students_not_picked

-- Define the total number of students who made the teams
def total_students_made_teams := 384.0

-- Define the number of schools
def number_of_schools := total_students_made_teams / students_picked

theorem number_of_schools_is_8 : number_of_schools = 8 := by
  -- Proof omitted
  sorry

end number_of_schools_is_8_l202_202509


namespace graph_symmetric_l202_202280

noncomputable def f (x : ℝ) : ℝ := sorry

theorem graph_symmetric (f : ℝ → ℝ) :
  (∀ x y, y = f x ↔ (∃ y₁, y₁ = f (2 - x) ∧ y = - (1 / (y₁ + 1)))) →
  ∀ x, f x = 1 / (x - 3) := 
by
  intro h x
  sorry

end graph_symmetric_l202_202280


namespace negation_proposition_l202_202672

theorem negation_proposition :
  (¬ (x ≠ 3 ∧ x ≠ 2) → ¬ (x ^ 2 - 5 * x + 6 ≠ 0)) =
  ((x = 3 ∨ x = 2) → (x ^ 2 - 5 * x + 6 = 0)) :=
by
  sorry

end negation_proposition_l202_202672


namespace sum_xyz_l202_202361

theorem sum_xyz (x y z : ℝ) (h1 : x + y = 1) (h2 : y + z = 1) (h3 : z + x = 1) : x + y + z = 3 / 2 :=
  sorry

end sum_xyz_l202_202361


namespace num_ways_to_use_100_yuan_l202_202893

noncomputable def x : ℕ → ℝ
| 0       => 0
| 1       => 1
| 2       => 3
| (n + 3) => x (n + 2) + 2 * x (n + 1)

theorem num_ways_to_use_100_yuan :
  x 100 = (1 / 3) * (2 ^ 101 + 1) :=
sorry

end num_ways_to_use_100_yuan_l202_202893


namespace sum_of_squares_l202_202902

theorem sum_of_squares (a b c d : ℝ) (h1 : b = a + 1) (h2 : c = a + 2) (h3 : d = a + 3) :
  a^2 + b^2 = c^2 + d^2 := by
  sorry

end sum_of_squares_l202_202902


namespace total_team_points_l202_202974

theorem total_team_points :
  let A := 2
  let B := 9
  let C := 4
  let D := -3
  let E := 7
  let F := 0
  let G := 5
  let H := -2
  (A + B + C + D + E + F + G + H = 22) :=
by
  let A := 2
  let B := 9
  let C := 4
  let D := -3
  let E := 7
  let F := 0
  let G := 5
  let H := -2
  sorry

end total_team_points_l202_202974


namespace total_soda_consumption_l202_202149

variables (c_soda b_soda c_consumed b_consumed b_remaining carol_final bob_final total_consumed : ℕ)

-- Define the conditions
def carol_soda_size : ℕ := 20
def bob_soda_25_percent_more : ℕ := carol_soda_size + carol_soda_size * 25 / 100
def carol_consumed : ℕ := carol_soda_size * 80 / 100
def bob_consumed : ℕ := bob_soda_25_percent_more * 80 / 100
def carol_remaining : ℕ := carol_soda_size - carol_consumed
def bob_remaining : ℕ := bob_soda_25_percent_more - bob_consumed
def bob_gives_carol : ℕ := bob_remaining / 2 + 3
def carol_final_consumption : ℕ := carol_consumed + bob_gives_carol
def bob_final_consumption : ℕ := bob_consumed - bob_gives_carol
def total_soda_consumed : ℕ := carol_final_consumption + bob_final_consumption

-- The theorem to prove the total amount of soda consumed by Carol and Bob together is 36 ounces
theorem total_soda_consumption : total_soda_consumed = 36 := by {
  sorry
}

end total_soda_consumption_l202_202149


namespace total_employees_in_buses_l202_202990

-- Definitions from conditions
def busCapacity : ℕ := 150
def percentageFull1 : ℕ := 60
def percentageFull2 : ℕ := 70

-- Proving the total number of employees
theorem total_employees_in_buses : 
  (percentageFull1 * busCapacity / 100) + (percentageFull2 * busCapacity / 100) = 195 := 
by
  sorry

end total_employees_in_buses_l202_202990


namespace option_A_is_correct_l202_202581

-- Define propositions p and q
variables (p q : Prop)

-- Option A
def isOptionACorrect: Prop := (¬p ∨ ¬q) → (¬p ∧ ¬q)

theorem option_A_is_correct: isOptionACorrect p q := sorry

end option_A_is_correct_l202_202581


namespace solve_percentage_increase_length_l202_202517

def original_length (L : ℝ) : Prop := true
def original_breadth (B : ℝ) : Prop := true

def new_breadth (B' : ℝ) (B : ℝ) : Prop := B' = 1.25 * B

def new_length (L' : ℝ) (L : ℝ) (x : ℝ) : Prop := L' = L * (1 + x / 100)

def original_area (L : ℝ) (B : ℝ) (A : ℝ) : Prop := A = L * B

def new_area (A' : ℝ) (A : ℝ) : Prop := A' = 1.375 * A

def percentage_increase_length (x : ℝ) : Prop := x = 10

theorem solve_percentage_increase_length (L B A A' L' B' x : ℝ)
  (hL : original_length L)
  (hB : original_breadth B)
  (hB' : new_breadth B' B)
  (hL' : new_length L' L x)
  (hA : original_area L B A)
  (hA' : new_area A' A)
  (h_eqn : L' * B' = A') :
  percentage_increase_length x :=
by
  sorry

end solve_percentage_increase_length_l202_202517


namespace log_relationship_l202_202035

theorem log_relationship (a b c : ℝ) 
  (ha : a = Real.log 3 / Real.log 2) 
  (hb : b = Real.log 4 / Real.log 3) 
  (hc : c = Real.log 5 / Real.log 4) : 
  c < b ∧ b < a :=
by 
  sorry

end log_relationship_l202_202035


namespace no_n_nat_powers_l202_202060

theorem no_n_nat_powers (n : ℕ) : ∀ n : ℕ, ¬∃ m k : ℕ, k ≥ 2 ∧ n * (n + 1) = m ^ k := 
by 
  sorry

end no_n_nat_powers_l202_202060


namespace line_equation_under_transformation_l202_202665

noncomputable def T1_matrix : Matrix (Fin 2) (Fin 2) ℝ := ![
  ![0, -1],
  ![1, 0]
]

noncomputable def T2_matrix : Matrix (Fin 2) (Fin 2) ℝ := ![
  ![2, 0],
  ![0, 3]
]

noncomputable def NM_matrix : Matrix (Fin 2) (Fin 2) ℝ := ![
  ![0, -2],
  ![3, 0]
]

theorem line_equation_under_transformation :
  ∀ x y : ℝ, (∃ x' y' : ℝ, NM_matrix.mulVec ![x, y] = ![x', y'] ∧ x' = y') → 3 * x + 2 * y = 0 :=
by sorry

end line_equation_under_transformation_l202_202665


namespace obtain_2015_in_4_operations_obtain_2015_in_3_operations_l202_202161

-- Define what an operation is
def operation (cards : List ℕ) : List ℕ :=
  sorry  -- Implementation of this is unnecessary for the statement

-- Check if 2015 can be obtained in 4 operations
def can_obtain_2015_in_4_operations (initial_cards : List ℕ) : Prop :=
  ∃ cards, (operation^[4] initial_cards) = cards ∧ 2015 ∈ cards

-- Check if 2015 can be obtained in 3 operations
def can_obtain_2015_in_3_operations (initial_cards : List ℕ) : Prop :=
  ∃ cards, (operation^[3] initial_cards) = cards ∧ 2015 ∈ cards

theorem obtain_2015_in_4_operations :
  can_obtain_2015_in_4_operations [1, 2] :=
sorry

theorem obtain_2015_in_3_operations :
  can_obtain_2015_in_3_operations [1, 2] :=
sorry

end obtain_2015_in_4_operations_obtain_2015_in_3_operations_l202_202161


namespace identical_digits_satisfy_l202_202561

theorem identical_digits_satisfy (n : ℕ) (hn : n ≥ 2) (x y z : ℕ) :
  (∃ (x y z : ℕ),
     (∃ (x y z : ℕ), 
         x = 3 ∧ y = 2 ∧ z = 1) ∨
     (∃ (x y z : ℕ), 
         x = 6 ∧ y = 8 ∧ z = 4) ∨
     (∃ (x y z : ℕ), 
         x = 8 ∧ y = 3 ∧ z = 7)) :=
by sorry

end identical_digits_satisfy_l202_202561


namespace angle_acb_after_rotations_is_30_l202_202222

noncomputable def initial_angle : ℝ := 60
noncomputable def rotation_clockwise_540 : ℝ := -540
noncomputable def rotation_counterclockwise_90 : ℝ := 90
noncomputable def final_angle : ℝ := 30

theorem angle_acb_after_rotations_is_30 
  (initial_angle : ℝ)
  (rotation_clockwise_540 : ℝ)
  (rotation_counterclockwise_90 : ℝ) :
  final_angle = 30 :=
sorry

end angle_acb_after_rotations_is_30_l202_202222


namespace miles_round_trip_time_l202_202745

theorem miles_round_trip_time : 
  ∀ (d : ℝ), d = 57 →
  ∀ (t : ℝ), t = 40 →
  ∀ (x : ℝ), x = 4 →
  10 = ((2 * d * x) / t) * 2 := 
by
  intros d hd t ht x hx
  rw [hd, ht, hx]
  sorry

end miles_round_trip_time_l202_202745


namespace product_of_two_consecutive_even_numbers_is_divisible_by_8_l202_202946

theorem product_of_two_consecutive_even_numbers_is_divisible_by_8 (n : ℤ) : (4 * n * (n + 1)) % 8 = 0 :=
sorry

end product_of_two_consecutive_even_numbers_is_divisible_by_8_l202_202946


namespace find_line_equation_l202_202545

-- Define the first line equation
def line1 (x y : ℝ) : Prop := 2 * x - y - 5 = 0

-- Define the second line equation
def line2 (x y : ℝ) : Prop := x + y + 2 = 0

-- Define the parallel line equation with a variable constant term
def line_parallel (x y m : ℝ) : Prop := 3 * x + y + m = 0

-- State the intersection point
def intersect_point (x y : ℝ) : Prop := line1 x y ∧ line2 x y

-- The desired equation of the line passing through the intersection point
theorem find_line_equation (x y : ℝ) (h : intersect_point x y) : ∃ m, line_parallel x y m := by
  sorry

end find_line_equation_l202_202545


namespace male_students_count_l202_202391

theorem male_students_count
  (average_all_students : ℕ → ℕ → ℚ → Prop)
  (average_male_students : ℕ → ℚ → Prop)
  (average_female_students : ℕ → ℚ → Prop)
  (F : ℕ)
  (total_average : average_all_students (F + M) (83 * M + 92 * F) 90)
  (male_average : average_male_students M 83)
  (female_average : average_female_students 28 92) :
  ∃ (M : ℕ), M = 8 :=
by {
  sorry
}

end male_students_count_l202_202391


namespace blood_drug_concentration_at_13_hours_l202_202195

theorem blood_drug_concentration_at_13_hours :
  let peak_time := 3
  let test_interval := 2
  let decrease_rate := 0.4
  let target_rate := 0.01024
  let time_to_reach_target := (fun n => (2 * n + 1))
  peak_time + test_interval * 5 = 13 :=
sorry

end blood_drug_concentration_at_13_hours_l202_202195


namespace fraction_twins_l202_202952

variables (P₀ I E P_f f : ℕ) (x : ℚ)

def initial_population := P₀ = 300000
def immigrants := I = 50000
def emigrants := E = 30000
def pregnant_fraction := f = 1 / 8
def final_population := P_f = 370000

theorem fraction_twins :
  initial_population P₀ ∧ immigrants I ∧ emigrants E ∧ pregnant_fraction f ∧ final_population P_f →
  x = 1 / 4 :=
by
  sorry

end fraction_twins_l202_202952


namespace every_integer_as_sum_of_squares_l202_202411

theorem every_integer_as_sum_of_squares (n : ℤ) : ∃ x y z : ℕ, 0 < x ∧ 0 < y ∧ 0 < z ∧ n = (x^2 : ℤ) + (y^2 : ℤ) - (z^2 : ℤ) :=
by sorry

end every_integer_as_sum_of_squares_l202_202411


namespace solution_80_percent_needs_12_ounces_l202_202307

theorem solution_80_percent_needs_12_ounces:
  ∀ (x y: ℝ), (x + y = 40) → (0.30 * x + 0.80 * y = 0.45 * 40) → (y = 12) :=
by
  intros x y h1 h2
  sorry

end solution_80_percent_needs_12_ounces_l202_202307


namespace compare_abc_l202_202283

noncomputable def a : ℝ := Real.log 4 / Real.log 3
noncomputable def b : ℝ := (1 / 3 : ℝ) ^ (1 / 3 : ℝ)
noncomputable def c : ℝ := (3 : ℝ) ^ (-1 / 4 : ℝ)

theorem compare_abc : b < c ∧ c < a :=
by
  sorry

end compare_abc_l202_202283


namespace find_a13_l202_202351

variable (a_n : ℕ → ℝ)
variable (d : ℝ)
variable (h_arith : ∀ n, a_n (n + 1) = a_n n + d)
variable (h_geo : a_n 9 ^ 2 = a_n 1 * a_n 5)
variable (h_sum : a_n 1 + 3 * a_n 5 + a_n 9 = 20)

theorem find_a13 (h_non_zero_d : d ≠ 0):
  a_n 13 = 28 :=
sorry

end find_a13_l202_202351


namespace abs_inequality_solution_l202_202028

theorem abs_inequality_solution (x : ℝ) : 
  (|x - 2| + |x + 3| < 8) ↔ -9 / 2 < x ∧ x < 7 / 2 :=
by
  sorry

end abs_inequality_solution_l202_202028


namespace circle_tangent_problem_solution_l202_202371

noncomputable def circle_tangent_problem
(radius : ℝ)
(center : ℝ × ℝ)
(point_A : ℝ × ℝ)
(distance_OA : ℝ)
(segment_BC : ℝ) : ℝ :=
  let r := radius
  let O := center
  let A := point_A
  let OA := distance_OA
  let BC := segment_BC
  let AT := Real.sqrt (OA^2 - r^2)
  2 * AT - BC

-- Definitions for the conditions
def radius : ℝ := 8
def center : ℝ × ℝ := (0, 0)
def point_A : ℝ × ℝ := (17, 0)
def distance_OA : ℝ := 17
def segment_BC : ℝ := 12

-- Statement of the problem as an example theorem
theorem circle_tangent_problem_solution :
  circle_tangent_problem radius center point_A distance_OA segment_BC = 18 :=
by
  -- We would provide the proof here. The proof steps are not required as per the instructions.
  sorry

end circle_tangent_problem_solution_l202_202371


namespace geometric_sequence_common_ratio_l202_202215

theorem geometric_sequence_common_ratio (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h1 : a 3 = 2 * S 2 + 1) (h2 : a 4 = 2 * S 3 + 1) :
  ∃ q : ℝ, (q = 3) :=
by
  -- Proof will go here.
  sorry

end geometric_sequence_common_ratio_l202_202215


namespace exists_good_set_l202_202682

variable (M : Set ℕ) [DecidableEq M] [Fintype M]
variable (f : Finset ℕ → ℕ)

theorem exists_good_set :
  ∃ T : Finset ℕ, T.card = 10 ∧ (∀ k ∈ T, f (T.erase k) ≠ k) := by
  sorry

end exists_good_set_l202_202682


namespace division_of_cubics_l202_202344

theorem division_of_cubics (c d : ℕ) (h1 : c = 7) (h2 : d = 3) : 
  (c^3 + d^3) / (c^2 - c * d + d^2) = 10 := by
  sorry

end division_of_cubics_l202_202344


namespace solution_l202_202121

theorem solution (A B C : ℚ) (h1 : A + B = 10) (h2 : 2 * A = 3 * B + 5) (h3 : A * B * C = 120) :
  A = 7 ∧ B = 3 ∧ C = 40 / 7 := by
  sorry

end solution_l202_202121


namespace odd_function_behavior_on_interval_l202_202842

theorem odd_function_behavior_on_interval
  (f : ℝ → ℝ)
  (h_odd : ∀ x, f (-x) = -f x)
  (h_increasing : ∀ x₁ x₂, 1 ≤ x₁ → x₁ < x₂ → x₂ ≤ 4 → f x₁ < f x₂)
  (h_max : ∀ x, 1 ≤ x → x ≤ 4 → f x ≤ 5) :
  (∀ x, -4 ≤ x → x ≤ -1 → f (-4) ≤ f x ∧ f x ≤ f (-1)) ∧ f (-4) = -5 :=
sorry

end odd_function_behavior_on_interval_l202_202842


namespace convex_polygon_with_arith_prog_angles_l202_202635

theorem convex_polygon_with_arith_prog_angles 
  (n : ℕ) 
  (angles : Fin n → ℝ)
  (is_convex : ∀ i, angles i < 180)
  (arithmetic_progression : ∃ a d, d = 3 ∧ ∀ i, angles i = a + i * d)
  (largest_angle : ∃ i, angles i = 150)
  : n = 24 :=
sorry

end convex_polygon_with_arith_prog_angles_l202_202635


namespace class_contribution_Miss_Evans_class_contribution_Mr_Smith_class_contribution_Mrs_Johnson_l202_202911

theorem class_contribution_Miss_Evans :
  let total_contribution : ℝ := 90
  let class_funds_Evans : ℝ := 14
  let num_students_Evans : ℕ := 19
  let individual_contribution_Evans : ℝ := (total_contribution - class_funds_Evans) / num_students_Evans
  individual_contribution_Evans = 4 := 
sorry

theorem class_contribution_Mr_Smith :
  let total_contribution : ℝ := 90
  let class_funds_Smith : ℝ := 20
  let num_students_Smith : ℕ := 15
  let individual_contribution_Smith : ℝ := (total_contribution - class_funds_Smith) / num_students_Smith
  individual_contribution_Smith = 4.67 := 
sorry

theorem class_contribution_Mrs_Johnson :
  let total_contribution : ℝ := 90
  let class_funds_Johnson : ℝ := 30
  let num_students_Johnson : ℕ := 25
  let individual_contribution_Johnson : ℝ := (total_contribution - class_funds_Johnson) / num_students_Johnson
  individual_contribution_Johnson = 2.40 := 
sorry

end class_contribution_Miss_Evans_class_contribution_Mr_Smith_class_contribution_Mrs_Johnson_l202_202911


namespace find_a_value_l202_202212

theorem find_a_value (a : ℕ) (h : a^3 = 21 * 25 * 45 * 49) : a = 105 :=
sorry

end find_a_value_l202_202212


namespace expression_value_l202_202282

theorem expression_value (a : ℝ) (h : a = 1/3) : 
  (4 * a⁻¹ - 2 * a⁻¹ / 3) / a^2 = 90 := by
  sorry

end expression_value_l202_202282


namespace trajectory_eq_l202_202407

theorem trajectory_eq :
  ∀ (x y : ℝ), abs x * abs y = 1 → (x * y = 1 ∨ x * y = -1) :=
by
  intro x y h
  sorry

end trajectory_eq_l202_202407


namespace count_perfect_cubes_between_bounds_l202_202404

theorem count_perfect_cubes_between_bounds :
  let lower_bound := 3^6 + 1
  let upper_bound := 3^12 + 1
  -- the number of perfect cubes k^3 such that 3^6 + 1 < k^3 < 3^12 + 1 inclusive is 72
  (730 < k * k * k ∧ k * k * k <= 531442 ∧ 10 <= k ∧ k <= 81 → k = 72) :=
by
  let lower_bound : ℕ := 3^6 + 1
  let upper_bound : ℕ := 3^12 + 1
  sorry

end count_perfect_cubes_between_bounds_l202_202404


namespace angle_F_measure_l202_202909

theorem angle_F_measure (α β γ : ℝ) (hD : α = 84) (hAngleSum : α + β + γ = 180) (hBeta : β = 4 * γ + 18) :
  γ = 15.6 := by
  sorry

end angle_F_measure_l202_202909


namespace find_a_in_triangle_l202_202211

theorem find_a_in_triangle
  (A B C : ℝ)
  (a b c : ℝ)
  (h1 : c = 3)
  (h2 : C = Real.pi / 3)
  (h3 : Real.sin B = 2 * Real.sin A)
  (h4 : a = 3) :
  a = Real.sqrt 3 := by
  sorry

end find_a_in_triangle_l202_202211


namespace min_value_condition_l202_202343

theorem min_value_condition 
  (a b : ℝ) 
  (h1 : 4 * a + b = 1) 
  (h2 : a > 0) 
  (h3 : b > 0) : 
  ∃ x : ℝ, ∀ y : ℝ, (y = 1 - 4 * x → x = 16) := 
sorry

end min_value_condition_l202_202343


namespace Tim_income_percentage_less_than_Juan_l202_202401

-- Definitions for the problem
variables (T M J : ℝ)

-- Conditions based on the problem
def condition1 : Prop := M = 1.60 * T
def condition2 : Prop := M = 0.80 * J

-- Goal statement
theorem Tim_income_percentage_less_than_Juan :
  condition1 T M ∧ condition2 M J → T = 0.50 * J :=
by sorry

end Tim_income_percentage_less_than_Juan_l202_202401


namespace solve_for_x_l202_202230

theorem solve_for_x (x : ℕ) (hx : 1000^4 = 10^x) : x = 12 := 
by
  sorry

end solve_for_x_l202_202230


namespace train_length_l202_202408

theorem train_length (speed_first_train speed_second_train : ℝ) (length_second_train : ℝ) (cross_time : ℝ) (L1 : ℝ) : 
  speed_first_train = 100 ∧ 
  speed_second_train = 60 ∧ 
  length_second_train = 300 ∧ 
  cross_time = 18 → 
  L1 = 420 :=
by
  sorry

end train_length_l202_202408


namespace simplify_exponentiation_l202_202582

-- Define the exponents and the base
variables (t : ℕ)

-- Define the expression and expected result
def expr := t^5 * t^2
def expected := t^7

-- State the proof goal
theorem simplify_exponentiation : expr = expected := 
by sorry

end simplify_exponentiation_l202_202582


namespace sum_first_13_terms_l202_202956

variable {a_n : ℕ → ℝ} (S : ℕ → ℝ)
variable (a_1 d : ℝ)

-- Arithmetic sequence properties
axiom arithmetic_sequence (n : ℕ) : a_n n = a_1 + (n - 1) * d

-- Sum of the first n terms
axiom sum_of_terms (n : ℕ) : S n = n / 2 * (2 * a_1 + (n - 1) * d)

-- Given condition
axiom sum_specific_terms : a_n 2 + a_n 7 + a_n 12 = 30

-- Theorem to prove
theorem sum_first_13_terms : S 13 = 130 := sorry

end sum_first_13_terms_l202_202956


namespace max_metro_lines_l202_202318

theorem max_metro_lines (lines : ℕ) 
  (stations_per_line : ℕ) 
  (max_interchange : ℕ) 
  (max_lines_per_interchange : ℕ) :
  (stations_per_line >= 4) → 
  (max_interchange <= 3) → 
  (max_lines_per_interchange <= 2) → 
  (∀ s_1 s_2, ∃ t_1 t_2, t_1 ≤ max_interchange ∧ t_2 ≤ max_interchange ∧
     (s_1 = t_1 ∨ s_2 = t_1 ∨ s_1 = t_2 ∨ s_2 = t_2)) → 
  lines ≤ 10 :=
by
  sorry

end max_metro_lines_l202_202318


namespace problem1_problem2_l202_202150

-- Proof Problem 1: Prove that when \( k = 5 \), \( x^2 - 5x + 4 > 0 \) holds for \( \{x \mid x < 1 \text{ or } x > 4\} \).
theorem problem1 (x : ℝ) (h : x^2 - 5 * x + 4 > 0) : x < 1 ∨ x > 4 :=
sorry

-- Proof Problem 2: Prove that the range of values for \( k \) such that \( x^2 - kx + 4 > 0 \) holds for all real numbers \( x \) is \( (-4, 4) \).
theorem problem2 (k : ℝ) : (∀ x : ℝ, x^2 - k * x + 4 > 0) ↔ -4 < k ∧ k < 4 :=
sorry

end problem1_problem2_l202_202150


namespace average_salary_correct_l202_202725

def salary_A : ℕ := 8000
def salary_B : ℕ := 5000
def salary_C : ℕ := 16000
def salary_D : ℕ := 7000
def salary_E : ℕ := 9000

def total_salary : ℕ := salary_A + salary_B + salary_C + salary_D + salary_E
def number_of_people : ℕ := 5

def average_salary : ℕ := total_salary / number_of_people

theorem average_salary_correct : average_salary = 8800 := by
  sorry

end average_salary_correct_l202_202725


namespace pyramid_side_length_difference_l202_202038

theorem pyramid_side_length_difference (x : ℕ) (h1 : 1 + x^2 + (x + 1)^2 + (x + 2)^2 = 30) : x = 2 :=
by
  sorry

end pyramid_side_length_difference_l202_202038


namespace no_pairs_probability_l202_202309

-- Define the number of socks and initial conditions
def pairs_of_socks : ℕ := 3
def total_socks : ℕ := pairs_of_socks * 2

-- Probabilistic outcome space for no pairs in first three draws
def probability_no_pairs_in_first_three_draws : ℚ :=
  (4/5) * (1/2)

-- Theorem stating that probability of no matching pairs in the first three draws is 2/5
theorem no_pairs_probability : probability_no_pairs_in_first_three_draws = 2/5 := by
  sorry

end no_pairs_probability_l202_202309


namespace quadratic_equation_l202_202589

theorem quadratic_equation (a b c x1 x2 : ℝ) (hx1 : a * x1^2 + b * x1 + c = 0) (hx2 : a * x2^2 + b * x2 + c = 0) :
  ∃ y : ℝ, c * y^2 + b * y + a = 0 := 
sorry

end quadratic_equation_l202_202589


namespace polygon_sides_eight_l202_202743

theorem polygon_sides_eight {n : ℕ} (h : (n - 2) * 180 = 3 * 360) : n = 8 :=
by
  sorry

end polygon_sides_eight_l202_202743


namespace shelves_of_mystery_books_l202_202976

theorem shelves_of_mystery_books (total_books : ℕ) (picture_shelves : ℕ) (books_per_shelf : ℕ) (M : ℕ) 
  (h_total_books : total_books = 54) 
  (h_picture_shelves : picture_shelves = 4) 
  (h_books_per_shelf : books_per_shelf = 6)
  (h_mystery_books : total_books - picture_shelves * books_per_shelf = M * books_per_shelf) :
  M = 5 :=
by
  sorry

end shelves_of_mystery_books_l202_202976


namespace hula_hoop_radius_l202_202739

theorem hula_hoop_radius (d : ℝ) (hd : d = 14) : d / 2 = 7 :=
by
  rw [hd]
  norm_num

end hula_hoop_radius_l202_202739


namespace determine_a_l202_202100

theorem determine_a (a x y : ℝ) (h : (a + 1) * x^(|a|) + y = -8) (h_linear : ∀ x y, (a + 1) * x^(|a|) + y = -8 → x ^ 1 = x): a = 1 :=
by 
  sorry

end determine_a_l202_202100


namespace count_defective_pens_l202_202056

theorem count_defective_pens
  (total_pens : ℕ) (prob_non_defective : ℚ)
  (h1 : total_pens = 12)
  (h2 : prob_non_defective = 0.5454545454545454) :
  ∃ (D : ℕ), D = 1 := by
  sorry

end count_defective_pens_l202_202056


namespace students_shared_cost_l202_202586

theorem students_shared_cost (P n : ℕ) (h_price_range: 100 ≤ P ∧ P ≤ 120)
  (h_div1: P % n = 0) (h_div2: P % (n - 2) = 0) (h_extra_cost: P / n + 1 = P / (n - 2)) : n = 14 := by
  sorry

end students_shared_cost_l202_202586


namespace divisor_of_5025_is_5_l202_202372

/--
  Given an original number n which is 5026,
  and a resulting number after subtracting 1 from n,
  prove that the divisor of the resulting number is 5.
-/
theorem divisor_of_5025_is_5 (n : ℕ) (h₁ : n = 5026) (d : ℕ) (h₂ : (n - 1) % d = 0) : d = 5 :=
sorry

end divisor_of_5025_is_5_l202_202372


namespace tangent_line_b_value_l202_202072

theorem tangent_line_b_value (b : ℝ) : 
  (∃ pt : ℝ × ℝ, (pt.1)^2 + (pt.2)^2 = 25 ∧ pt.1 - pt.2 + b = 0)
  ↔ b = 5 * Real.sqrt 2 ∨ b = -5 * Real.sqrt 2 :=
by
  sorry

end tangent_line_b_value_l202_202072


namespace sandbox_volume_l202_202170

def length : ℕ := 312
def width : ℕ := 146
def depth : ℕ := 75
def volume (l w d : ℕ) : ℕ := l * w * d

theorem sandbox_volume : volume length width depth = 3429000 := by
  sorry

end sandbox_volume_l202_202170


namespace transformed_parabolas_combined_l202_202488

theorem transformed_parabolas_combined (a b c : ℝ) :
  let f (x : ℝ) := a * (x - 3) ^ 2 + b * (x - 3) + c
  let g (x : ℝ) := -a * (x + 4) ^ 2 - b * (x + 4) - c
  ∀ x, (f x + g x) = -14 * a * x - 19 * a - 7 * b :=
by
  -- This is a placeholder for the actual proof using the conditions
  sorry

end transformed_parabolas_combined_l202_202488


namespace mo_tea_cups_l202_202933

theorem mo_tea_cups (n t : ℤ) 
  (h1 : 2 * n + 5 * t = 26) 
  (h2 : 5 * t = 2 * n + 14) :
  t = 4 :=
sorry

end mo_tea_cups_l202_202933


namespace popton_school_bus_total_toes_l202_202999

-- Define the number of toes per hand for each race
def toes_per_hand_hoopit : ℕ := 3
def toes_per_hand_neglart : ℕ := 2
def toes_per_hand_zentorian : ℕ := 4

-- Define the number of hands for each race
def hands_per_hoopit : ℕ := 4
def hands_per_neglart : ℕ := 5
def hands_per_zentorian : ℕ := 6

-- Define the number of students from each race on the bus
def num_hoopits : ℕ := 7
def num_neglarts : ℕ := 8
def num_zentorians : ℕ := 5

-- Calculate the total number of toes on the bus
def total_toes_on_bus : ℕ :=
  num_hoopits * (toes_per_hand_hoopit * hands_per_hoopit) +
  num_neglarts * (toes_per_hand_neglart * hands_per_neglart) +
  num_zentorians * (toes_per_hand_zentorian * hands_per_zentorian)

-- Theorem stating the number of toes on the bus
theorem popton_school_bus_total_toes : total_toes_on_bus = 284 :=
by
  sorry

end popton_school_bus_total_toes_l202_202999


namespace ratio_problem_l202_202094

theorem ratio_problem 
  (A B C : ℚ) 
  (h : A / B = 3 / 2 ∧ B / C = 2 / 5 ∧ A / C = 3 / 5) : 
  (4 * A + 3 * B) / (5 * C - 2 * A) = 15 / 19 := 
by 
  sorry

end ratio_problem_l202_202094


namespace notAlwaysTriangleInSecondQuadrantAfterReflection_l202_202597

structure Point where
  x : ℝ
  y : ℝ

structure Triangle where
  P : Point
  Q : Point
  R : Point

def isInSecondQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

def reflectionOverYEqualsX (p : Point) : Point :=
  { x := p.y, y := p.x }

def reflectTriangleOverYEqualsX (T : Triangle) : Triangle :=
  { P := reflectionOverYEqualsX T.P,
    Q := reflectionOverYEqualsX T.Q,
    R := reflectionOverYEqualsX T.R }

def triangleInSecondQuadrant (T : Triangle) : Prop :=
  isInSecondQuadrant T.P ∧ isInSecondQuadrant T.Q ∧ isInSecondQuadrant T.R

theorem notAlwaysTriangleInSecondQuadrantAfterReflection
  (T : Triangle)
  (h : triangleInSecondQuadrant T)
  : ¬ (triangleInSecondQuadrant (reflectTriangleOverYEqualsX T)) := 
sorry -- Proof not required

end notAlwaysTriangleInSecondQuadrantAfterReflection_l202_202597


namespace meadow_grazing_days_l202_202384

theorem meadow_grazing_days 
    (a b x : ℝ) 
    (h1 : a + 6 * b = 27 * 6 * x)
    (h2 : a + 9 * b = 23 * 9 * x)
    : ∃ y : ℝ, (a + y * b = 21 * y * x) ∧ y = 12 := 
by
    sorry

end meadow_grazing_days_l202_202384


namespace union_A_B_range_of_a_l202_202409

-- Definitions of sets A, B, and C
def A : Set ℝ := { x | 3 ≤ x ∧ x ≤ 9 }
def B : Set ℝ := { x | 2 < x ∧ x < 5 }
def C (a : ℝ) : Set ℝ := { x | x > a }

-- Problem 1: Proving A ∪ B = { x | 2 < x ≤ 9 }
theorem union_A_B : A ∪ B = { x | 2 < x ∧ x ≤ 9 } :=
sorry

-- Problem 2: Proving the range of 'a' given B ∩ C = ∅
theorem range_of_a (a : ℝ) (h : B ∩ C a = ∅) : a ≥ 5 :=
sorry

end union_A_B_range_of_a_l202_202409


namespace probability_black_ball_l202_202966

theorem probability_black_ball :
  let P_red := 0.41
  let P_white := 0.27
  let P_black := 1 - P_red - P_white
  P_black = 0.32 :=
by
  sorry

end probability_black_ball_l202_202966


namespace initial_price_of_TV_l202_202931

theorem initial_price_of_TV (T : ℤ) (phone_price_increase : ℤ) (total_amount : ℤ) 
    (h1 : phone_price_increase = (400: ℤ) + (40 * 400 / 100)) 
    (h2 : total_amount = T + (2 * T / 5) + phone_price_increase) 
    (h3 : total_amount = 1260) : 
    T = 500 := by
  sorry

end initial_price_of_TV_l202_202931


namespace opposite_of_neg_abs_is_positive_two_l202_202572

theorem opposite_of_neg_abs_is_positive_two : -(abs (-2)) = -2 :=
by sorry

end opposite_of_neg_abs_is_positive_two_l202_202572


namespace find_triplets_find_triplets_non_negative_l202_202498

theorem find_triplets :
  ∀ (x y z : ℕ), (x > 0) ∧ (y > 0) ∧ (z > 0) →
    x^2 + y^2 + 1 = 2^z →
    (x = 1 ∧ y = 0 ∧ z = 1) ∨ (x = 0 ∧ y = 1 ∧ z = 1) :=
by
  sorry

theorem find_triplets_non_negative :
  ∀ (x y z : ℕ), x^2 + y^2 + 1 = 2^z →
    (x = 1 ∧ y = 0 ∧ z = 1) ∨ (x = 0 ∧ y = 1 ∧ z = 1) ∨ (x = 0 ∧ y = 0 ∧ z = 0) :=
by
  sorry

end find_triplets_find_triplets_non_negative_l202_202498


namespace div_by_16_l202_202023

theorem div_by_16 (n : ℕ) : 
  ((2*n - 1)^3 - (2*n)^2 + 2*n + 1) % 16 = 0 :=
sorry

end div_by_16_l202_202023


namespace average_carnations_l202_202988

theorem average_carnations (c1 c2 c3 n : ℕ) (h1 : c1 = 9) (h2 : c2 = 14) (h3 : c3 = 13) (h4 : n = 3) :
  (c1 + c2 + c3) / n = 12 :=
by
  sorry

end average_carnations_l202_202988


namespace range_of_a_l202_202159

variable {x a : ℝ}

def p (x : ℝ) := 2*x^2 - 3*x + 1 ≤ 0
def q (x : ℝ) (a : ℝ) := (x - a) * (x - a - 1) ≤ 0

theorem range_of_a (h : ¬ p x → ¬ q x a) : 0 ≤ a ∧ a ≤ 1/2 := by
  sorry

end range_of_a_l202_202159


namespace negation_of_proposition_p_is_false_l202_202948

variable (p : Prop)

theorem negation_of_proposition_p_is_false
  (h : ¬p) : ¬(¬p) :=
by
  sorry

end negation_of_proposition_p_is_false_l202_202948


namespace solve_quadratic_l202_202192

theorem solve_quadratic : 
  ∀ x : ℝ, (x - 1) ^ 2 = 64 → (x = 9 ∨ x = -7) :=
by
  sorry

end solve_quadratic_l202_202192


namespace quadratic_coefficient_conversion_l202_202158

theorem quadratic_coefficient_conversion :
  ∀ x : ℝ, (3 * x^2 - 1 = 5 * x) → (3 * x^2 - 5 * x - 1 = 0) :=
by
  intros x h
  rw [←sub_eq_zero, ←h]
  ring

end quadratic_coefficient_conversion_l202_202158


namespace triangle_is_acute_l202_202789

-- Define the condition that the angles have a ratio of 2:3:4
def angle_ratio_cond (a b c : ℝ) : Prop :=
  a / b = 2 / 3 ∧ b / c = 3 / 4

-- Define the sum of the angles in a triangle
def angle_sum_cond (a b c : ℝ) : Prop :=
  a + b + c = 180

-- The proof problem stating that triangle with angles in ratio 2:3:4 is acute
theorem triangle_is_acute (a b c : ℝ) (h_ratio : angle_ratio_cond a b c) (h_sum : angle_sum_cond a b c) : 
  a < 90 ∧ b < 90 ∧ c < 90 := 
by
  sorry

end triangle_is_acute_l202_202789


namespace total_pets_remaining_l202_202193

def initial_counts := (7, 6, 4, 5, 3)  -- (puppies, kittens, rabbits, guinea pigs, chameleons)
def morning_sales := (1, 2, 1, 0, 0)  -- (puppies, kittens, rabbits, guinea pigs, chameleons)
def afternoon_sales := (1, 1, 2, 3, 0)  -- (puppies, kittens, rabbits, guinea pigs, chameleons)
def returns := (0, 1, 0, 1, 1)  -- (puppies, kittens, rabbits, guinea pigs, chameleons)

def calculate_remaining (initial_counts morning_sales afternoon_sales returns : Nat × Nat × Nat × Nat × Nat) : Nat :=
  let (p0, k0, r0, g0, c0) := initial_counts
  let (p1, k1, r1, g1, c1) := morning_sales
  let (p2, k2, r2, g2, c2) := afternoon_sales
  let (p3, k3, r3, g3, c3) := returns
  let remaining_puppies := p0 - p1 - p2 + p3
  let remaining_kittens := k0 - k1 - k2 + k3
  let remaining_rabbits := r0 - r1 - r2 + r3
  let remaining_guinea_pigs := g0 - g1 - g2 + g3
  let remaining_chameleons := c0 - c1 - c2 + c3
  remaining_puppies + remaining_kittens + remaining_rabbits + remaining_guinea_pigs + remaining_chameleons

theorem total_pets_remaining : calculate_remaining initial_counts morning_sales afternoon_sales returns = 15 := 
by
  simp [initial_counts, morning_sales, afternoon_sales, returns, calculate_remaining]
  sorry

end total_pets_remaining_l202_202193


namespace quadratic_roots_shifted_l202_202977

theorem quadratic_roots_shifted (a b c : ℝ) (r s : ℝ) 
  (h1 : 4 * r ^ 2 + 2 * r - 9 = 0) 
  (h2 : 4 * s ^ 2 + 2 * s - 9 = 0) :
  c = 51 / 4 := by
  sorry

end quadratic_roots_shifted_l202_202977


namespace trapezoid_perimeter_l202_202728

noncomputable def perimeter_trapezoid 
  (AB CD AD BC : ℝ) 
  (h_AB_CD_parallel : AB = CD) 
  (h_AD_perpendicular : AD = 4 * Real.sqrt 2)
  (h_BC_perpendicular : BC = 4 * Real.sqrt 2)
  (h_AB_eq : AB = 10)
  (h_CD_eq : CD = 18)
  (h_height : Real.sqrt (AD ^ 2 - 1) = 4) 
  : ℝ :=
AB + BC + CD + AD

theorem trapezoid_perimeter
  (AB CD AD BC : ℝ)
  (h_AB_CD_parallel : AB = CD) 
  (h_AD_perpendicular : AD = 4 * Real.sqrt 2)
  (h_BC_perpendicular : BC = 4 * Real.sqrt 2)
  (h_AB_eq : AB = 10)
  (h_CD_eq : CD = 18)
  (h_height : Real.sqrt (AD ^ 2 - 1) = 4) 
  : perimeter_trapezoid AB CD AD BC h_AB_CD_parallel h_AD_perpendicular h_BC_perpendicular h_AB_eq h_CD_eq h_height = 28 + 8 * Real.sqrt 2 :=
by
  sorry

end trapezoid_perimeter_l202_202728


namespace realNumbersGreaterThan8IsSet_l202_202630

-- Definitions based on conditions:
def verySmallNumbers : Type := {x : ℝ // sorry} -- Need to define what very small numbers would be
def interestingBooks : Type := sorry -- Need to define what interesting books would be
def realNumbersGreaterThan8 : Set ℝ := { x : ℝ | x > 8 }
def tallPeople : Type := sorry -- Need to define what tall people would be

-- Main theorem: Real numbers greater than 8 can form a set
theorem realNumbersGreaterThan8IsSet : Set ℝ :=
  realNumbersGreaterThan8

end realNumbersGreaterThan8IsSet_l202_202630


namespace cone_height_l202_202763

theorem cone_height (h : ℝ) (r : ℝ) 
  (volume_eq : (1/3) * π * r^2 * h = 19683 * π) 
  (isosceles_right_triangle : h = r) : 
  h = 39.0 :=
by
  -- The proof will go here
  sorry

end cone_height_l202_202763


namespace tree_graph_probability_127_l202_202374

theorem tree_graph_probability_127 :
  let n := 5
  let p := 125
  let q := 1024
  q ^ (1/10) + p = 127 :=
by
  sorry

end tree_graph_probability_127_l202_202374


namespace bricks_in_wall_is_720_l202_202485

/-- 
Two bricklayers have varying speeds: one could build a wall in 12 hours and 
the other in 15 hours if working alone. Their efficiency decreases by 12 bricks
per hour when they work together. The contractor placed them together on this 
project and the wall was completed in 6 hours.
Prove that the number of bricks in the wall is 720.
-/
def number_of_bricks_in_wall (y : ℕ) : Prop :=
  let rate1 := y / 12
  let rate2 := y / 15
  let combined_rate := rate1 + rate2 - 12
  6 * combined_rate = y

theorem bricks_in_wall_is_720 : ∃ y : ℕ, number_of_bricks_in_wall y ∧ y = 720 :=
  by sorry

end bricks_in_wall_is_720_l202_202485


namespace range_of_a_l202_202786

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x^2 + 2*a*x + a > 0) → 0 < a ∧ a < 1 :=
by
  sorry

end range_of_a_l202_202786


namespace total_households_in_apartment_complex_l202_202043

theorem total_households_in_apartment_complex :
  let buildings := 25
  let floors_per_building := 10
  let households_per_floor := 8
  buildings * floors_per_building * households_per_floor = 2000 :=
by
  sorry

end total_households_in_apartment_complex_l202_202043


namespace volume_cone_equals_cylinder_minus_surface_area_l202_202437

theorem volume_cone_equals_cylinder_minus_surface_area (r h : ℝ) :
  let V_cyl := π * r^2 * h
  let V_cone := (1 / 3) * π * r^2 * h
  let S_lateral_cyl := 2 * π * r * h
  V_cone = V_cyl - (1 / 3) * S_lateral_cyl * r := by
  let V_cyl := π * r^2 * h
  let V_cone := (1 / 3) * π * r^2 * h
  let S_lateral_cyl := 2 * π * r * h
  sorry

end volume_cone_equals_cylinder_minus_surface_area_l202_202437


namespace largest_five_digit_number_with_product_l202_202742

theorem largest_five_digit_number_with_product :
  ∃ (x : ℕ), (x = 98752) ∧ (∀ (d : List ℕ), (x.digits 10 = d) → (d.prod = 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1)) ∧ (x < 100000) ∧ (x ≥ 10000) :=
by
  sorry

end largest_five_digit_number_with_product_l202_202742


namespace non_neg_solutions_l202_202737

theorem non_neg_solutions (x y z : ℕ) :
  (x^3 = 2 * y^2 - z) →
  (y^3 = 2 * z^2 - x) →
  (z^3 = 2 * x^2 - y) →
  (x = 0 ∧ y = 0 ∧ z = 0) ∨ (x = 1 ∧ y = 1 ∧ z = 1) :=
by {
  sorry
}

end non_neg_solutions_l202_202737


namespace f_f_of_2_l202_202870

def f (x : ℤ) : ℤ := 4 * x ^ 3 - 3 * x + 1

theorem f_f_of_2 : f (f 2) = 78652 := 
by
  sorry

end f_f_of_2_l202_202870


namespace sqrt_6_between_2_and_3_l202_202107

theorem sqrt_6_between_2_and_3 : 2 < Real.sqrt 6 ∧ Real.sqrt 6 < 3 :=
by
  sorry

end sqrt_6_between_2_and_3_l202_202107


namespace quadratic_rewrite_l202_202304

theorem quadratic_rewrite :
  ∃ d e f : ℤ, (4 * (x : ℝ)^2 - 24 * x + 35 = (d * x + e)^2 + f) ∧ (d * e = -12) :=
by
  sorry

end quadratic_rewrite_l202_202304


namespace Ginger_sold_10_lilacs_l202_202958

variable (R L G : ℕ)

def condition1 := R = 3 * L
def condition2 := G = L / 2
def condition3 := L + R + G = 45

theorem Ginger_sold_10_lilacs
    (h1 : condition1 R L)
    (h2 : condition2 G L)
    (h3 : condition3 L R G) :
  L = 10 := 
  sorry

end Ginger_sold_10_lilacs_l202_202958


namespace sum_series_l202_202052

theorem sum_series (s : ℕ → ℝ) 
  (h : ∀ n : ℕ, s n = (n+1) / (4 : ℝ)^(n+1)) : 
  tsum s = (4 / 9 : ℝ) :=
sorry

end sum_series_l202_202052


namespace median_line_eqn_l202_202751

theorem median_line_eqn (A B C : ℝ × ℝ)
  (hA : A = (3, 7)) (hB : B = (5, -1)) (hC : C = (-2, -5)) :
  ∃ m b : ℝ, (4, -3, -7) = (m, b, 0) :=
by sorry

end median_line_eqn_l202_202751


namespace seq_general_formula_l202_202238

def seq (a : ℕ → ℝ) : Prop :=
  ∀ n, a n > 0 ∧ a n ^ 2 - (2 * a (n + 1) - 1) * a n - 2 * a (n + 1) = 0

theorem seq_general_formula {a : ℕ → ℝ} (h1 : a 1 = 1) (h2 : seq a) :
  ∀ n, a n = 1 / 2 ^ (n - 1) :=
by
  sorry

end seq_general_formula_l202_202238


namespace steven_ships_boxes_l202_202758

-- Translate the conditions into Lean definitions and state the theorem
def truck_weight_limit : ℕ := 2000
def truck_count : ℕ := 3
def pair_weight : ℕ := 10 + 40
def boxes_per_pair : ℕ := 2

theorem steven_ships_boxes :
  ((truck_weight_limit / pair_weight) * boxes_per_pair * truck_count) = 240 := by
  sorry

end steven_ships_boxes_l202_202758


namespace president_vice_president_count_l202_202962

/-- The club consists of 24 members, split evenly with 12 boys and 12 girls. 
    There are also two classes, each containing 6 boys and 6 girls. 
    Prove that the number of ways to choose a president and a vice-president 
    if they must be of the same gender and from different classes is 144. -/
theorem president_vice_president_count :
  ∃ n : ℕ, n = 144 ∧ 
  (∀ (club : Finset ℕ) (boys girls : Finset ℕ) 
     (class1_boys class1_girls class2_boys class2_girls : Finset ℕ),
     club.card = 24 →
     boys.card = 12 → girls.card = 12 →
     class1_boys.card = 6 → class1_girls.card = 6 →
     class2_boys.card = 6 → class2_girls.card = 6 →
     (∃ president vice_president : ℕ,
     president ∈ club ∧ vice_president ∈ club ∧
     ((president ∈ boys ∧ vice_president ∈ boys) ∨ 
      (president ∈ girls ∧ vice_president ∈ girls)) ∧
     ((president ∈ class1_boys ∧ vice_president ∈ class2_boys) ∨
      (president ∈ class2_boys ∧ vice_president ∈ class1_boys) ∨
      (president ∈ class1_girls ∧ vice_president ∈ class2_girls) ∨
      (president ∈ class2_girls ∧ vice_president ∈ class1_girls)) →
     n = 144)) :=
by
  sorry

end president_vice_president_count_l202_202962


namespace expression_range_l202_202388

open Real -- Open the real number namespace

theorem expression_range (x y : ℝ) (h : (x - 1)^2 + (y - 4)^2 = 1) : 
  0 ≤ (x * y - x) / (x^2 + (y - 1)^2) ∧ (x * y - x) / (x^2 + (y - 1)^2) ≤ 12 / 25 :=
sorry -- Proof to be filled in.

end expression_range_l202_202388


namespace number_of_lightsabers_in_order_l202_202918

-- Let's define the given conditions
def metal_arcs_per_lightsaber : ℕ := 2
def cost_per_metal_arc : ℕ := 400
def apparatus_production_rate : ℕ := 20 -- lightsabers per hour
def combined_app_expense_rate : ℕ := 300 -- units per hour
def total_order_cost : ℕ := 65200
def lightsaber_cost : ℕ := metal_arcs_per_lightsaber * cost_per_metal_arc + (combined_app_expense_rate / apparatus_production_rate)

-- Define the main theorem to prove
theorem number_of_lightsabers_in_order : 
  (total_order_cost / lightsaber_cost) = 80 :=
by
  sorry

end number_of_lightsabers_in_order_l202_202918


namespace wood_blocks_after_days_l202_202651

-- Defining the known conditions
def blocks_per_tree : Nat := 3
def trees_per_day : Nat := 2
def days : Nat := 5

-- Stating the theorem to prove the total number of blocks of wood after 5 days
theorem wood_blocks_after_days : blocks_per_tree * trees_per_day * days = 30 :=
by
  sorry

end wood_blocks_after_days_l202_202651


namespace find_number_l202_202882

theorem find_number (x : ℤ) (h : 4 * x - 7 = 13) : x = 5 := 
sorry

end find_number_l202_202882


namespace marbles_total_l202_202930

theorem marbles_total (r b g y : ℝ) 
  (h1 : r = 1.30 * b)
  (h2 : g = 1.50 * r)
  (h3 : y = 0.80 * g) :
  r + b + g + y = 4.4692 * r :=
by
  sorry

end marbles_total_l202_202930


namespace intersection_A_B_l202_202070

-- Define the sets A and B based on given conditions
def A : Set ℝ := {x | -2 < x ∧ x < 4}
def B : Set ℝ := {x | -1 < x}

-- The statement to prove
theorem intersection_A_B : (A ∩ B) = {x | -1 < x ∧ x < 4} :=
  sorry

end intersection_A_B_l202_202070


namespace committee_with_one_boy_one_girl_prob_l202_202480

def total_members := 30
def boys := 12
def girls := 18
def committee_size := 6

theorem committee_with_one_boy_one_girl_prob :
  let total_ways := Nat.choose total_members committee_size
  let all_boys_ways := Nat.choose boys committee_size
  let all_girls_ways := Nat.choose girls committee_size
  let prob_all_boys_or_all_girls := (all_boys_ways + all_girls_ways) / total_ways
  let desired_prob := 1 - prob_all_boys_or_all_girls
  desired_prob = 19145 / 19793 :=
by
  sorry

end committee_with_one_boy_one_girl_prob_l202_202480


namespace division_of_fractions_l202_202828

theorem division_of_fractions :
  (5 / 6 : ℚ) / (11 / 12) = 10 / 11 := by
  sorry

end division_of_fractions_l202_202828


namespace pair_with_15_is_47_l202_202660

theorem pair_with_15_is_47 (numbers : Set ℕ) (k : ℕ) 
  (h : numbers = {49, 29, 9, 40, 22, 15, 53, 33, 13, 47}) 
  (pair_sum_eq_k : ∀ (a b : ℕ), a ∈ numbers → b ∈ numbers → (a, b) ≠ (15, 15) → a + b = k) : 
  ∃ (k : ℕ), 15 + 47 = k := 
sorry

end pair_with_15_is_47_l202_202660


namespace solve_exponential_equation_l202_202877

theorem solve_exponential_equation :
  ∃ x, (2:ℝ)^(2*x) - 8 * (2:ℝ)^x + 12 = 0 ↔ x = 1 ∨ x = 1 + Real.log 3 / Real.log 2 :=
by
  sorry

end solve_exponential_equation_l202_202877


namespace Susie_possible_values_l202_202907

theorem Susie_possible_values (n : ℕ) (h1 : n > 43) (h2 : 2023 % n = 43) : 
  (∃ count : ℕ, count = 19 ∧ ∀ n, n > 43 ∧ 2023 % n = 43 → 1980 ∣ (2023 - 43)) :=
sorry

end Susie_possible_values_l202_202907


namespace pieces_left_to_place_l202_202616

noncomputable def total_pieces : ℕ := 300
noncomputable def reyn_pieces : ℕ := 25
noncomputable def rhys_pieces : ℕ := 2 * reyn_pieces
noncomputable def rory_pieces : ℕ := 3 * reyn_pieces
noncomputable def placed_pieces : ℕ := reyn_pieces + rhys_pieces + rory_pieces
noncomputable def remaining_pieces : ℕ := total_pieces - placed_pieces

theorem pieces_left_to_place : remaining_pieces = 150 :=
by sorry

end pieces_left_to_place_l202_202616


namespace ellipse_meets_sine_more_than_8_points_l202_202691

noncomputable def ellipse_intersects_sine_curve_more_than_8_times (a b : ℝ) (h k : ℝ) :=
  ∃ p : ℕ, p > 8 ∧ 
  ∃ (x y : ℝ), 
    (∃ (i : ℕ), y = Real.sin x ∧ 
    (x - h)^2 / a^2 + (y - k)^2 / b^2 = 1)

theorem ellipse_meets_sine_more_than_8_points : 
  ∀ (a b h k : ℝ), ellipse_intersects_sine_curve_more_than_8_times a b h k := 
by sorry

end ellipse_meets_sine_more_than_8_points_l202_202691


namespace find_k_range_l202_202759

theorem find_k_range (k : ℝ) : 
  (∃ x y : ℝ, y = -2 * x + 3 * k + 14 ∧ x - 4 * y = -3 * k - 2 ∧ x > 0 ∧ y < 0) ↔ -6 < k ∧ k < -2 :=
by
  sorry

end find_k_range_l202_202759


namespace problem_solution_l202_202240

theorem problem_solution :
  ∃ (b₂ b₃ b₄ b₅ b₆ b₇ : ℤ),
    (0 ≤ b₂ ∧ b₂ < 2) ∧
    (0 ≤ b₃ ∧ b₃ < 3) ∧
    (0 ≤ b₄ ∧ b₄ < 4) ∧
    (0 ≤ b₅ ∧ b₅ < 5) ∧
    (0 ≤ b₆ ∧ b₆ < 6) ∧
    (0 ≤ b₇ ∧ b₇ < 8) ∧
    (6 / 7 = b₂ / 2 + b₃ / 6 + b₄ / 24 + b₅ / 120 + b₆ / 720 + b₇ / 5040) ∧
    (b₂ + b₃ + b₄ + b₅ + b₆ + b₇ = 11) :=
sorry

end problem_solution_l202_202240


namespace Adam_marbles_l202_202848

variable (Adam Greg : Nat)

theorem Adam_marbles (h1 : Greg = 43) (h2 : Greg = Adam + 14) : Adam = 29 := 
by
  sorry

end Adam_marbles_l202_202848


namespace subset_single_element_l202_202251

-- Define the set X
def X : Set ℝ := { x | x > -1 }

-- The proof statement
-- We need to prove that {0} ⊆ X
theorem subset_single_element : {0} ⊆ X :=
sorry

end subset_single_element_l202_202251


namespace intersection_points_l202_202044

noncomputable def hyperbola : Set (ℝ × ℝ) := { p | ∃ (x y : ℝ), p = (x, y) ∧ x^2 / 9 - y^2 = 1 }

noncomputable def line : Set (ℝ × ℝ) := { p | ∃ (x y : ℝ), p = (x, y) ∧ y = (1 / 3) * (x + 1) }

theorem intersection_points :
  ∃! (p : ℝ × ℝ), p ∈ hyperbola ∧ p ∈ line :=
sorry

end intersection_points_l202_202044


namespace composite_expression_l202_202174

theorem composite_expression (n : ℕ) : ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ (a * b = 6 * 2^(2^(4 * n)) + 1) :=
by
  sorry

end composite_expression_l202_202174


namespace inequality_solution_l202_202993

theorem inequality_solution (x : ℝ) : (x / (x + 1) + (x + 3) / (2 * x) ≥ 2) ↔ (0 < x ∧ x ≤ 1) ∨ x = -3 :=
by
sorry

end inequality_solution_l202_202993


namespace div_by_5_l202_202139

theorem div_by_5 (n : ℕ) (hn : 0 < n) : (2^(4*n+1) + 3) % 5 = 0 := 
by sorry

end div_by_5_l202_202139


namespace fraction_to_decimal_l202_202369

theorem fraction_to_decimal : (3 : ℚ) / 40 = 0.075 :=
sorry

end fraction_to_decimal_l202_202369


namespace max_value_of_f_on_interval_l202_202629

noncomputable def f (x : ℝ) : ℝ := -x^2 + 4 * x - 2

theorem max_value_of_f_on_interval : ∀ x, 0 ≤ x ∧ x ≤ 1 → f x ≤ 1 :=
by
  sorry

end max_value_of_f_on_interval_l202_202629


namespace spirit_concentration_l202_202281

theorem spirit_concentration (vol_a vol_b vol_c : ℕ) (conc_a conc_b conc_c : ℝ)
(h_a : conc_a = 0.45) (h_b : conc_b = 0.30) (h_c : conc_c = 0.10)
(h_vola : vol_a = 4) (h_volb : vol_b = 5) (h_volc : vol_c = 6) : 
  (conc_a * vol_a + conc_b * vol_b + conc_c * vol_c) / (vol_a + vol_b + vol_c) * 100 = 26 := 
by
  sorry

end spirit_concentration_l202_202281


namespace percentage_very_satisfactory_l202_202034

-- Definitions based on conditions
def total_parents : ℕ := 120
def needs_improvement_count : ℕ := 6
def excellent_percentage : ℕ := 15
def satisfactory_remaining_percentage : ℕ := 80

-- Theorem statement
theorem percentage_very_satisfactory 
  (total_parents : ℕ) 
  (needs_improvement_count : ℕ) 
  (excellent_percentage : ℕ) 
  (satisfactory_remaining_percentage : ℕ) 
  (result : ℕ) : result = 16 :=
by
  sorry

end percentage_very_satisfactory_l202_202034


namespace inequality_solution_set_l202_202209

theorem inequality_solution_set (x : ℝ) : (x + 2) * (1 - x) > 0 ↔ -2 < x ∧ x < 1 :=
by
  sorry

end inequality_solution_set_l202_202209


namespace simplify_evaluate_expr_l202_202771

noncomputable def expr (x : ℝ) : ℝ := 
  ( ( (x^2 - 3) / (x + 2) - x + 2 ) / ( (x^2 - 4) / (x^2 + 4*x + 4) ) )

theorem simplify_evaluate_expr : 
  expr (Real.sqrt 2 + 1) = Real.sqrt 2 + 1 := by
  sorry

end simplify_evaluate_expr_l202_202771


namespace fixed_monthly_costs_l202_202190

theorem fixed_monthly_costs
  (production_cost_per_component : ℕ)
  (shipping_cost_per_component : ℕ)
  (components_per_month : ℕ)
  (lowest_price_per_component : ℕ)
  (total_revenue : ℕ)
  (total_variable_cost : ℕ)
  (F : ℕ) :
  production_cost_per_component = 80 →
  shipping_cost_per_component = 5 →
  components_per_month = 150 →
  lowest_price_per_component = 195 →
  total_variable_cost = components_per_month * (production_cost_per_component + shipping_cost_per_component) →
  total_revenue = components_per_month * lowest_price_per_component →
  total_revenue = total_variable_cost + F →
  F = 16500 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end fixed_monthly_costs_l202_202190


namespace area_triangle_ABF_proof_area_triangle_AFD_proof_l202_202130

variable (A B C D M F : Type)
variable (area_square : Real) (midpoint_D_CM : Prop) (lies_on_line_BC : Prop)

-- Given conditions
axiom area_ABCD_300 : area_square = 300
axiom M_midpoint_DC : midpoint_D_CM
axiom F_on_line_BC : lies_on_line_BC

-- Define areas for the triangles
def area_triangle_ABF : Real := 300
def area_triangle_AFD : Real := 150

-- Prove that given the conditions, the area of triangle ABF is 300 cm²
theorem area_triangle_ABF_proof : area_square = 300 ∧ midpoint_D_CM ∧ lies_on_line_BC → area_triangle_ABF = 300 :=
by
  intro h
  sorry

-- Prove that given the conditions, the area of triangle AFD is 150 cm²
theorem area_triangle_AFD_proof : area_square = 300 ∧ midpoint_D_CM ∧ lies_on_line_BC → area_triangle_AFD = 150 :=
by
  intro h
  sorry

end area_triangle_ABF_proof_area_triangle_AFD_proof_l202_202130


namespace Roberto_outfit_count_l202_202390

theorem Roberto_outfit_count :
  let trousers := 5
  let shirts := 6
  let jackets := 3
  let ties := 2
  trousers * shirts * jackets * ties = 180 :=
by
  sorry

end Roberto_outfit_count_l202_202390


namespace range_of_a_l202_202834

-- Define the function f
def f (a x : ℝ) : ℝ := a * x^3 - 3 * x^2 + 1

-- Define the conditions: f has a unique zero point x₀ and x₀ < 0
def unique_zero_point (a : ℝ) : Prop :=
  ∃! x₀ : ℝ, f a x₀ = 0 ∧ x₀ < 0

-- The theorem we need to prove
theorem range_of_a (a : ℝ) : unique_zero_point a → a > 2 :=
sorry

end range_of_a_l202_202834


namespace find_a_and_b_l202_202898

theorem find_a_and_b (a b : ℚ) (h : ∀ (n : ℕ), 1 / ((2 * n - 1) * (2 * n + 1)) = a / (2 * n - 1) + b / (2 * n + 1)) : 
  a = 1/2 ∧ b = -1/2 := 
by 
  sorry

end find_a_and_b_l202_202898


namespace license_plate_count_l202_202474

def license_plate_combinations : Nat :=
  26 * Nat.choose 25 2 * Nat.choose 4 2 * 720

theorem license_plate_count :
  license_plate_combinations = 33696000 :=
by
  unfold license_plate_combinations
  sorry

end license_plate_count_l202_202474


namespace triangle_area_l202_202531

theorem triangle_area (a b c p : ℕ) (h_ratio : a = 5 * p) (h_ratio2 : b = 12 * p) (h_ratio3 : c = 13 * p) (h_perimeter : a + b + c = 300) : 
  (1 / 4) * Real.sqrt ((a + b + c) * (a + b - c) * (a + c - b) * (b + c - a)) = 3000 := 
by 
  sorry

end triangle_area_l202_202531


namespace tangent_circles_radii_l202_202746

noncomputable def radii_of_tangent_circles (R r : ℝ) (h : R > r) : Set ℝ :=
  { x | x = (R * r) / ((Real.sqrt R + Real.sqrt r)^2) ∨ x = (R * r) / ((Real.sqrt R - Real.sqrt r)^2) }

theorem tangent_circles_radii (R r : ℝ) (h : R > r) :
  ∃ x, x ∈ radii_of_tangent_circles R r h := sorry

end tangent_circles_radii_l202_202746


namespace sum_remainder_product_remainder_l202_202830

open Nat

-- Define the modulus conditions
variables (x y z : ℕ)
def condition1 : Prop := x % 15 = 11
def condition2 : Prop := y % 15 = 13
def condition3 : Prop := z % 15 = 14

-- Proof statement for the sum remainder
theorem sum_remainder (h1 : condition1 x) (h2 : condition2 y) (h3 : condition3 z) : (x + y + z) % 15 = 8 :=
by
  sorry

-- Proof statement for the product remainder
theorem product_remainder (h1 : condition1 x) (h2 : condition2 y) (h3 : condition3 z) : (x * y * z) % 15 = 2 :=
by
  sorry

end sum_remainder_product_remainder_l202_202830


namespace math_problem_l202_202866

variables {R : Type*} [Ring R] (x y z : R)

theorem math_problem (h : x * y + y * z + z * x = 0) : 
  3 * x * y * z + x^2 * (y + z) + y^2 * (z + x) + z^2 * (x + y) = 0 :=
by 
  sorry

end math_problem_l202_202866


namespace repeating_decimal_to_fraction_l202_202928

theorem repeating_decimal_to_fraction : (6 + 81 / 99) = 75 / 11 := 
by 
  sorry

end repeating_decimal_to_fraction_l202_202928


namespace ratio_of_bases_l202_202144

-- Definitions for an isosceles trapezoid
def isosceles_trapezoid (s t : ℝ) := ∃ (a b c d : ℝ), s = d ∧ s = a ∧ t = b ∧ (a + c = b + d)

-- Main theorem statement based on conditions and required ratio
theorem ratio_of_bases (s t : ℝ) (h1 : isosceles_trapezoid s t)
  (h2 : s = s) (h3 : t = t) : s / t = 3 / 5 :=
by { sorry }

end ratio_of_bases_l202_202144


namespace vertex_of_parabola_l202_202507

theorem vertex_of_parabola :
  (∃ x y : ℝ, y = (x - 6)^2 + 3 ↔ (x = 6 ∧ y = 3)) :=
sorry

end vertex_of_parabola_l202_202507


namespace neg_prop_l202_202064

theorem neg_prop : ∃ (a : ℝ), ∀ (x : ℝ), (a * x^2 - 3 * x + 2 = 0) → x ≤ 0 :=
sorry

end neg_prop_l202_202064


namespace velma_more_than_veronica_l202_202083

-- Defining the distances each flashlight can be seen
def veronica_distance : ℕ := 1000
def freddie_distance : ℕ := 3 * veronica_distance
def velma_distance : ℕ := 5 * freddie_distance - 2000

-- The proof problem: Prove that Velma's flashlight can be seen 12000 feet farther than Veronica's flashlight.
theorem velma_more_than_veronica : velma_distance - veronica_distance = 12000 := by
  sorry

end velma_more_than_veronica_l202_202083


namespace total_days_on_jury_duty_l202_202819

-- Define the conditions
def jury_selection_days : ℕ := 2
def trial_duration_factor : ℕ := 4
def deliberation_days : ℕ := 6
def deliberation_hours_per_day : ℕ := 16
def hours_per_day : ℕ := 24

-- Calculate the trial duration in days
def trial_days : ℕ := trial_duration_factor * jury_selection_days

-- Calculate the total deliberation time in days
def deliberation_total_hours : ℕ := deliberation_days * deliberation_hours_per_day
def deliberation_days_converted : ℕ := deliberation_total_hours / hours_per_day

-- Statement that John spends a total of 14 days on jury duty
theorem total_days_on_jury_duty : jury_selection_days + trial_days + deliberation_days_converted = 14 :=
sorry

end total_days_on_jury_duty_l202_202819


namespace binary_division_remainder_l202_202899

theorem binary_division_remainder (n : ℕ) (h_n : n = 0b110110011011) : n % 8 = 3 :=
by {
  -- This sorry statement skips the actual proof
  sorry
}

end binary_division_remainder_l202_202899


namespace minimize_material_used_l202_202991

theorem minimize_material_used (r h : ℝ) (V : ℝ) (S : ℝ) 
  (volume_formula : π * r^2 * h = V) (volume_given : V = 27 * π) :
  ∃ r, r = 3 :=
by
  sorry

end minimize_material_used_l202_202991


namespace S_inter_T_eq_T_l202_202362

-- Define the sets S and T
def S := { s : ℤ | ∃ n : ℤ, s = 2 * n + 1 }
def T := { t : ℤ | ∃ n : ℤ, t = 4 * n + 1 }

-- Prove that S ∩ T = T
theorem S_inter_T_eq_T : S ∩ T = T := 
by
  sorry

end S_inter_T_eq_T_l202_202362


namespace angle_equivalence_l202_202291

theorem angle_equivalence :
  ∃ k : ℤ, -495 + 360 * k = 225 :=
sorry

end angle_equivalence_l202_202291


namespace marys_next_birthday_l202_202489

theorem marys_next_birthday (d s m : ℝ) (h1 : s = 0.7 * d) (h2 : m = 1.3 * s) (h3 : m + s + d = 25.2) : m + 1 = 9 :=
by
  sorry

end marys_next_birthday_l202_202489


namespace max_points_in_equilateral_property_set_l202_202181

theorem max_points_in_equilateral_property_set (Γ : Finset (ℝ × ℝ)) :
  (∀ (A B : (ℝ × ℝ)), A ∈ Γ → B ∈ Γ → 
    ∃ C : (ℝ × ℝ), C ∈ Γ ∧ 
    dist A B = dist B C ∧ dist B C = dist C A ∧ dist C A = dist A B) → Γ.card ≤ 3 :=
by
  intro h
  sorry

end max_points_in_equilateral_property_set_l202_202181


namespace monotonically_increasing_interval_l202_202430

noncomputable def f (x : ℝ) : ℝ :=
  Real.cos x ^ 2 + Real.sqrt 3 * Real.sin x * Real.cos x

theorem monotonically_increasing_interval :
  ∀ x, 0 < x ∧ x ≤ π / 6 → ∀ y, x ≤ y ∧ y < π / 2 → f x ≤ f y :=
by
  intro x hx y hy
  sorry

end monotonically_increasing_interval_l202_202430


namespace copies_per_person_l202_202317

-- Definitions derived from the conditions
def pages_per_contract : ℕ := 20
def total_pages_copied : ℕ := 360
def number_of_people : ℕ := 9

-- Theorem stating the result based on the conditions
theorem copies_per_person : (total_pages_copied / pages_per_contract) / number_of_people = 2 := by
  sorry

end copies_per_person_l202_202317


namespace sample_var_interpretation_l202_202809

theorem sample_var_interpretation (squared_diffs : Fin 10 → ℝ) :
  (10 = 10) ∧ (∀ i, squared_diffs i = (i - 20)^2) →
  (∃ n: ℕ, n = 10 ∧ ∃ μ: ℝ, μ = 20) :=
by
  intro h
  sorry

end sample_var_interpretation_l202_202809


namespace area_of_triangle_arithmetic_sides_l202_202140

theorem area_of_triangle_arithmetic_sides 
  (a : ℝ) (h : a > 0) (h_sin : Real.sin (2 * Real.pi / 3) = Real.sqrt 3 / 2) :
  let s₁ := a - 2
  let s₂ := a
  let s₃ := a + 2
  ∃ (a b c : ℝ), 
    a = s₁ ∧ b = s₂ ∧ c = s₃ ∧ 
    Real.sin (2 * Real.pi / 3) = Real.sqrt 3 / 2 → 
    (1/2 * s₁ * s₂ * Real.sin (2 * Real.pi / 3) = 15 * Real.sqrt 3 / 4) :=
by
  sorry

end area_of_triangle_arithmetic_sides_l202_202140


namespace smallest_n_for_terminating_decimal_l202_202168

-- Theorem follows the tuple of (question, conditions, correct answer)
theorem smallest_n_for_terminating_decimal (n : ℕ) (h : ∃ k : ℕ, n + 75 = 2^k ∨ n + 75 = 5^k ∨ n + 75 = (2^k * 5^k)) :
  n = 50 :=
by
  sorry -- Proof is omitted

end smallest_n_for_terminating_decimal_l202_202168


namespace economic_rationale_education_policy_l202_202476

theorem economic_rationale_education_policy
  (countries : Type)
  (foreign_citizens : Type)
  (universities : Type)
  (free_or_nominal_fee : countries → Prop)
  (international_agreements : countries → Prop)
  (aging_population : countries → Prop)
  (economic_benefits : countries → Prop)
  (credit_concessions : countries → Prop)
  (reciprocity_education : countries → Prop)
  (educated_youth_contributions : countries → Prop)
  :
  (∀ c : countries, free_or_nominal_fee c ↔
    (international_agreements c ∧ (credit_concessions c ∨ reciprocity_education c)) ∨
    (aging_population c ∧ economic_benefits c ∧ educated_youth_contributions c)) := 
sorry

end economic_rationale_education_policy_l202_202476


namespace f_2016_is_1_l202_202178

noncomputable def f : ℤ → ℤ := sorry

axiom h1 : f 1 = 1
axiom h2 : f 2015 ≠ 1
axiom h3 : ∀ a b : ℤ, f (a + b) ≤ max (f a) (f b)
axiom h4 : ∀ x : ℤ, f x = f (-x)

theorem f_2016_is_1 : f 2016 = 1 := 
by 
  sorry

end f_2016_is_1_l202_202178


namespace triangle_abc_l202_202445

/-!
# Problem Statement
In triangle ABC with side lengths a, b, and c opposite to vertices A, B, and C respectively, we are given that ∠A = 2 * ∠B. We need to prove that a² = b * (b + c).
-/

variables (A B C : Type) -- Define vertices of the triangle
variables (α β γ : ℝ) -- Define angles at vertices A, B, and C respectively.

-- Define sides of the triangle
variables (a b c x y : ℝ) -- Define sides opposite to the corresponding angles

-- Main statement to prove in Lean 4
theorem triangle_abc (h1 : α = 2 * β) (h2 : a = b * (2 * β)) :
  a^2 = b * (b + c) :=
sorry

end triangle_abc_l202_202445


namespace chandler_weeks_to_save_l202_202440

theorem chandler_weeks_to_save :
  let birthday_money := 50 + 35 + 15 + 20
  let weekly_earnings := 18
  let bike_cost := 650
  ∃ x : ℕ, (birthday_money + x * weekly_earnings) ≥ bike_cost ∧ (birthday_money + (x - 1) * weekly_earnings) < bike_cost := 
by
  sorry

end chandler_weeks_to_save_l202_202440


namespace max_x_on_circle_l202_202375

theorem max_x_on_circle : 
  ∀ x y : ℝ,
  (x - 10)^2 + (y - 30)^2 = 100 → x ≤ 20 :=
by
  intros x y h
  sorry

end max_x_on_circle_l202_202375


namespace distance_closer_to_R_after_meeting_l202_202008

def distance_between_R_and_S : ℕ := 80
def rate_of_man_from_R : ℕ := 5
def initial_rate_of_man_from_S : ℕ := 4

theorem distance_closer_to_R_after_meeting 
  (t : ℕ) 
  (x : ℕ) 
  (h1 : t ≠ 0) 
  (h2 : distance_between_R_and_S = 80) 
  (h3 : rate_of_man_from_R = 5) 
  (h4 : initial_rate_of_man_from_S = 4) 
  (h5 : (rate_of_man_from_R * t) 
        + (t * initial_rate_of_man_from_S 
        + ((t - 1) * t / 2)) = distance_between_R_and_S) :
  x = 20 :=
sorry

end distance_closer_to_R_after_meeting_l202_202008


namespace sequence_general_term_l202_202552

theorem sequence_general_term (a : ℕ → ℕ) 
  (h₀ : a 1 = 4) 
  (h₁ : ∀ n : ℕ, a (n + 1) = 2 * a n + n^2) : 
  ∀ n : ℕ, a n = 5 * 2^n - n^2 - 2*n - 3 :=
by
  sorry

end sequence_general_term_l202_202552


namespace student_score_variance_l202_202101

noncomputable def variance_student_score : ℝ :=
  let number_of_questions := 25
  let probability_correct := 0.8
  let score_correct := 4
  let variance_eta := number_of_questions * probability_correct * (1 - probability_correct)
  let variance_xi := (score_correct ^ 2) * variance_eta
  variance_xi

theorem student_score_variance : variance_student_score = 64 := by
  sorry

end student_score_variance_l202_202101


namespace solution_proof_l202_202289

variable (A B C : ℕ+) (x y : ℚ)
variable (h1 : A > B) (h2 : B > C) (h3 : A = B * (1 + x / 100)) (h4 : B = C * (1 + y / 100))

theorem solution_proof : x = 100 * ((A / (C * (1 + y / 100))) - 1) :=
by
  sorry

end solution_proof_l202_202289


namespace adult_ticket_cost_l202_202007

/--
Tickets at a local theater cost a certain amount for adults and 2 dollars for kids under twelve.
Given that 175 tickets were sold and the profit was 750 dollars, and 75 kid tickets were sold,
prove that an adult ticket costs 6 dollars.
-/
theorem adult_ticket_cost
  (kid_ticket_price : ℕ := 2)
  (kid_tickets_sold : ℕ := 75)
  (total_tickets_sold : ℕ := 175)
  (total_profit : ℕ := 750)
  (adult_tickets_sold : ℕ := total_tickets_sold - kid_tickets_sold)
  (adult_ticket_revenue : ℕ := total_profit - kid_ticket_price * kid_tickets_sold)
  (adult_ticket_cost : ℕ := adult_ticket_revenue / adult_tickets_sold) :
  adult_ticket_cost = 6 :=
by
  sorry

end adult_ticket_cost_l202_202007


namespace Ivan_bought_10_cards_l202_202943

-- Define variables and conditions
variables (x : ℕ) -- Number of Uno Giant Family Cards bought
def original_price : ℕ := 12
def discount_per_card : ℕ := 2
def discounted_price := original_price - discount_per_card
def total_paid : ℕ := 100

-- Lean 4 theorem statement
theorem Ivan_bought_10_cards (h : discounted_price * x = total_paid) : x = 10 := by
  -- proof goes here
  sorry

end Ivan_bought_10_cards_l202_202943


namespace Hazel_shirts_proof_l202_202315

variable (H : ℕ)

def shirts_received_by_Razel (h_shirts : ℕ) : ℕ :=
  2 * h_shirts

def total_shirts (h_shirts : ℕ) (r_shirts : ℕ) : ℕ :=
  h_shirts + r_shirts

theorem Hazel_shirts_proof
  (h_shirts : ℕ)
  (r_shirts : ℕ)
  (total : ℕ)
  (H_nonneg : 0 ≤ h_shirts)
  (R_twice_H : r_shirts = shirts_received_by_Razel h_shirts)
  (T_total : total = total_shirts h_shirts r_shirts)
  (total_is_18 : total = 18) :
  h_shirts = 6 :=
by
  sorry

end Hazel_shirts_proof_l202_202315


namespace min_value_expression_l202_202989

theorem min_value_expression (k x y z : ℝ) (hk : 0 < k) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  ∃ x_min y_min z_min : ℝ, (0 < x_min) ∧ (0 < y_min) ∧ (0 < z_min) ∧
  (∀ x y z : ℝ, 0 < x → 0 < y → 0 < z →
    k * (4 * z / (2 * x + y) + 4 * x / (y + 2 * z) + y / (x + z))
    ≥ 3 * k) ∧
  k * (4 * z_min / (2 * x_min + y_min) + 4 * x_min / (y_min + 2 * z_min) + y_min / (x_min + z_min)) = 3 * k :=
by sorry

end min_value_expression_l202_202989


namespace cricket_target_run_l202_202839

theorem cricket_target_run (run_rate1 run_rate2 : ℝ) (overs1 overs2 : ℕ) (T : ℝ) 
  (h1 : run_rate1 = 3.2) (h2 : overs1 = 10) (h3 : run_rate2 = 25) (h4 : overs2 = 10) :
  T = (run_rate1 * overs1) + (run_rate2 * overs2) → T = 282 :=
by
  sorry

end cricket_target_run_l202_202839


namespace fraction_defined_iff_l202_202609

theorem fraction_defined_iff (x : ℝ) : (x + 2 ≠ 0) ↔ (x ≠ -2) :=
by sorry

end fraction_defined_iff_l202_202609


namespace min_x_y_l202_202079

theorem min_x_y
  (x y : ℝ)
  (h1 : 0 < x)
  (h2 : 0 < y)
  (h3 : x + 2 * y + x * y - 7 = 0) :
  x + y ≥ 3 := by
  sorry

end min_x_y_l202_202079


namespace solve_quadratic_equation_solve_linear_equation_l202_202175

-- Equation (1)
theorem solve_quadratic_equation :
  ∀ x : ℝ, x^2 - 8 * x + 1 = 0 → (x = 4 + Real.sqrt 15 ∨ x = 4 - Real.sqrt 15) :=
by
  sorry

-- Equation (2)
theorem solve_linear_equation :
  ∀ x : ℝ, 3 * x * (x - 1) = 2 - 2 * x → (x = 1 ∨ x = -2/3) :=
by
  sorry

end solve_quadratic_equation_solve_linear_equation_l202_202175


namespace inequality_holds_for_all_x_l202_202413

variable (a x : ℝ)

theorem inequality_holds_for_all_x (h : a ∈ Set.Ioc (-2 : ℝ) 4): ∀ x : ℝ, (x^2 - a*x + 9 > 0) :=
sorry

end inequality_holds_for_all_x_l202_202413


namespace initial_speeds_l202_202872

/-- Motorcyclists Vasya and Petya ride at constant speeds around a circular track 1 km long.
    Petya overtakes Vasya every 2 minutes. Then Vasya doubles his speed and now he himself 
    overtakes Petya every 2 minutes. What were the initial speeds of Vasya and Petya? 
    Answer: 1000 and 1500 meters per minute.
-/

theorem initial_speeds (V_v V_p : ℕ) (track_length : ℕ) (time_interval : ℕ) 
  (h1 : track_length = 1000)
  (h2 : time_interval = 2)
  (h3 : V_p - V_v = track_length / time_interval)
  (h4 : 2 * V_v - V_p = track_length / time_interval):
  V_v = 1000 ∧ V_p = 1500 :=
by
  sorry

end initial_speeds_l202_202872


namespace ratio_area_shaded_triangle_l202_202749

variables (PQ PX QR QY YR : ℝ)
variables {A : ℝ}

def midpoint_QR (QR QY YR : ℝ) : Prop := QR = QY + YR ∧ QY = YR

def fraction_PQ_PX (PQ PX : ℝ) : Prop := PX = (3 / 4) * PQ

noncomputable def area_square (PQ : ℝ) : ℝ := PQ * PQ

noncomputable def area_triangle (base height : ℝ) : ℝ := (1 / 2) * base * height

theorem ratio_area_shaded_triangle
  (PQ PX QR QY YR : ℝ)
  (h_mid : midpoint_QR QR QY YR)
  (h_frac : fraction_PQ_PX PQ PX)
  (hQY_QR2 : QY = QR / 2)
  (hYR_QR2 : YR = QR / 2) :
  A = 5 / 16 :=
sorry

end ratio_area_shaded_triangle_l202_202749


namespace find_a_for_unique_solution_l202_202905

theorem find_a_for_unique_solution :
  ∃ a : ℝ, (∀ x : ℝ, 0 ≤ x^2 - a * x + a ∧ x^2 - a * x + a ≤ 1) ↔ a = 2 :=
by
  sorry

end find_a_for_unique_solution_l202_202905


namespace min_value_of_a2_b2_l202_202674

theorem min_value_of_a2_b2 (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : a + b = 4) : 
  ∃ m : ℝ, (∀ x y, x > 0 → y > 0 → x + y = 4 → x^2 + y^2 ≥ m) ∧ m = 8 :=
by
  sorry

end min_value_of_a2_b2_l202_202674


namespace number_of_female_students_l202_202325

-- Given conditions
variables (F : ℕ)

-- The average score of all students (90)
def avg_all_students := 90
-- The total number of male students (8)
def num_male_students := 8
-- The average score of male students (87)
def avg_male_students := 87
-- The average score of female students (92)
def avg_female_students := 92

-- We want to prove the following statement
theorem number_of_female_students :
  num_male_students * avg_male_students + F * avg_female_students = (num_male_students + F) * avg_all_students →
  F = 12 :=
sorry

end number_of_female_students_l202_202325


namespace rectangle_area_proof_l202_202525

variable (x y : ℕ) -- Declaring the variables to represent length and width of the rectangle.

-- Declaring the conditions as hypotheses.
def condition1 := (x + 3) * (y - 1) = x * y
def condition2 := (x - 3) * (y + 2) = x * y
def condition3 := (x + 4) * (y - 2) = x * y

-- The theorem to prove the area is 36 given the above conditions.
theorem rectangle_area_proof (h1 : condition1 x y) (h2 : condition2 x y) (h3 : condition3 x y) : x * y = 36 :=
by
  sorry

end rectangle_area_proof_l202_202525


namespace sequence_diff_exists_l202_202359

theorem sequence_diff_exists (x : ℕ → ℕ) (h1 : x 1 = 1) (h2 : ∀ n : ℕ, 1 ≤ n → x n < x (n + 1) ∧ x (n + 1) ≤ 2 * n) :
  ∀ k : ℕ, ∃ r s : ℕ, x r - x s = k :=
by
  sorry

end sequence_diff_exists_l202_202359


namespace count_sums_of_two_cubes_lt_400_l202_202627

theorem count_sums_of_two_cubes_lt_400 : 
  ∃ (s : Finset ℕ), 
    (∀ n ∈ s, ∃ a b, 1 ≤ a ∧ a ≤ 7 ∧ 1 ≤ b ∧ b ≤ 7 ∧ n = a^3 + b^3 ∧ (Odd a ∨ Odd b) ∧ n < 400) ∧
    s.card = 15 :=
by 
  sorry

end count_sums_of_two_cubes_lt_400_l202_202627


namespace even_m_n_l202_202047

variable {m n : ℕ}

theorem even_m_n
  (h_m : ∃ k : ℕ, m = 2 * k + 1)
  (h_n : ∃ k : ℕ, n = 2 * k + 1) :
  Even ((m - n) ^ 2) ∧ Even ((m - n - 4) ^ 2) ∧ Even (2 * m * n + 4) :=
by
  sorry

end even_m_n_l202_202047


namespace min_trams_spy_sees_l202_202588

/-- 
   Vasya stood at a bus stop for some time and saw 1 bus and 2 trams.
   Buses run every hour.
   After Vasya left, a spy stood at the bus stop for 10 hours and saw 10 buses.
   Given these conditions, the minimum number of trams that the spy could have seen is 5.
-/
theorem min_trams_spy_sees (bus_interval tram_interval : ℕ) 
  (vasya_buses vasya_trams spy_buses spy_hours min_trams : ℕ) 
  (h1 : bus_interval = 1)
  (h2 : vasya_buses = 1)
  (h3 : vasya_trams = 2)
  (h4 : spy_buses = spy_hours)
  (h5 : spy_buses = 10)
  (h6 : spy_hours = 10)
  (h7 : ∀ t : ℕ, t * tram_interval ≤ 2 → 2 * bus_interval ≤ 2)
  (h8 : min_trams = 5) :
  min_trams = 5 := 
sorry

end min_trams_spy_sees_l202_202588


namespace correctLikeTermsPair_l202_202529

def areLikeTerms (term1 term2 : String) : Bool :=
  -- Define the criteria for like terms (variables and their respective powers)
  sorry

def pairA : (String × String) := ("-2x^3", "-2x")
def pairB : (String × String) := ("-1/2ab", "18ba")
def pairC : (String × String) := ("x^2y", "-xy^2")
def pairD : (String × String) := ("4m", "4mn")

theorem correctLikeTermsPair :
  areLikeTerms pairA.1 pairA.2 = false ∧
  areLikeTerms pairB.1 pairB.2 = true ∧
  areLikeTerms pairC.1 pairC.2 = false ∧
  areLikeTerms pairD.1 pairD.2 = false :=
sorry

end correctLikeTermsPair_l202_202529


namespace curve_C_cartesian_eq_line_l_general_eq_max_area_triangle_PAB_l202_202642

-- Definitions for the conditions
def curve_C_polar (ρ θ : ℝ) := ρ = 4 * Real.sin θ
def line_l_parametric (x y t : ℝ) := 
  x = (Real.sqrt 3 / 2) * t ∧ 
  y = 1 + (1 / 2) * t

-- Theorem statements
theorem curve_C_cartesian_eq : ∀ x y : ℝ,
  (∃ (ρ θ : ℝ), curve_C_polar ρ θ ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) →
  x^2 + (y - 2)^2 = 4 :=
by sorry

theorem line_l_general_eq : ∀ x y t : ℝ,
  line_l_parametric x y t →
  x - (Real.sqrt 3) * y + Real.sqrt 3 = 0 :=
by sorry

theorem max_area_triangle_PAB : ∀ (P A B : ℝ × ℝ),
  (∃ (θ : ℝ), P = ⟨2 * Real.cos θ, 2 + 2 * Real.sin θ⟩ ∧
   (∃ t : ℝ, line_l_parametric A.1 A.2 t) ∧
   (∃ t' : ℝ, line_l_parametric B.1 B.2 t') ∧
   A ≠ B) →
  (1/2) * Real.sqrt 13 * (2 + Real.sqrt 3 / 2) = (4 * Real.sqrt 13 + Real.sqrt 39) / 4 :=
by sorry

end curve_C_cartesian_eq_line_l_general_eq_max_area_triangle_PAB_l202_202642


namespace g_50_equals_zero_l202_202303

noncomputable def g : ℝ → ℝ := sorry

theorem g_50_equals_zero (h : ∀ (x y : ℝ), 0 < x → 0 < y → x * g y - y * g x = g ((x + y) / y)) : g 50 = 0 :=
sorry

end g_50_equals_zero_l202_202303


namespace f_fixed_point_l202_202813

-- Definitions and conditions based on the problem statement
def g (n : ℕ) : ℕ := sorry
def f (n : ℕ) : ℕ := sorry

-- Helper functions for the repeated application of f
noncomputable def f_iter (n x : ℕ) : ℕ := 
    Nat.iterate f (x^2023) n

axiom g_bijective : Function.Bijective g
axiom f_repeated : ∀ x : ℕ, f_iter x x = x
axiom f_div_g : ∀ (x y : ℕ), x ∣ y → f x ∣ g y

-- Main theorem statement
theorem f_fixed_point : ∀ x : ℕ, f x = x := by
  sorry

end f_fixed_point_l202_202813


namespace total_tickets_l202_202947

-- Definitions based on given conditions
def initial_tickets : ℕ := 49
def spent_tickets : ℕ := 25
def additional_tickets : ℕ := 6

-- Proof statement (only statement, proof is not required)
theorem total_tickets : (initial_tickets - spent_tickets + additional_tickets = 30) :=
  sorry

end total_tickets_l202_202947


namespace binary_to_decimal_110011_l202_202243

theorem binary_to_decimal_110011 : 
  (1 * 2^0 + 1 * 2^1 + 0 * 2^2 + 0 * 2^3 + 1 * 2^4 + 1 * 2^5) = 51 :=
by
  sorry

end binary_to_decimal_110011_l202_202243


namespace correct_structure_l202_202741

-- Definitions for the conditions regarding flowchart structures
def loop_contains_conditional : Prop := ∀ (loop : Prop), ∃ (conditional : Prop), conditional ∧ loop
def unique_flowchart_for_boiling_water : Prop := ∀ (flowcharts : Prop), ∃! (boiling_process : Prop), flowcharts ∧ boiling_process
def conditional_does_not_contain_sequential : Prop := ∀ (conditional : Prop), ∃ (sequential : Prop), ¬ (conditional ∧ sequential)
def conditional_must_contain_loop : Prop := ∀ (conditional : Prop), ∃ (loop : Prop), conditional ∧ loop

-- The proof statement
theorem correct_structure (A B C D : Prop) (hA : A = loop_contains_conditional) 
  (hB : B = unique_flowchart_for_boiling_water) 
  (hC : C = conditional_does_not_contain_sequential) 
  (hD : D = conditional_must_contain_loop) : 
  A = loop_contains_conditional ∧ ¬ B ∧ ¬ C ∧ ¬ D :=
by {
  sorry
}

end correct_structure_l202_202741


namespace count_linear_eqs_l202_202045

-- Define each equation as conditions
def eq1 (x y : ℝ) := 3 * x - y = 2
def eq2 (x : ℝ) := x + 1 / x + 2 = 0
def eq3 (x : ℝ) := x^2 - 2 * x - 3 = 0
def eq4 (x : ℝ) := x = 0
def eq5 (x : ℝ) := 3 * x - 1 ≥ 5
def eq6 (x : ℝ) := 1 / 2 * x = 1 / 2
def eq7 (x : ℝ) := (2 * x + 1) / 3 = 1 / 6 * x

-- Proof statement: there are exactly 3 linear equations
theorem count_linear_eqs : 
  (∃ x y, eq1 x y) ∧ eq4 0 ∧ (∃ x, eq6 x) ∧ (∃ x, eq7 x) ∧ 
  ¬ (∃ x, eq2 x) ∧ ¬ (∃ x, eq3 x) ∧ ¬ (∃ x, eq5 x) → 
  3 = 3 :=
sorry

end count_linear_eqs_l202_202045


namespace stratified_sampling_sample_size_l202_202246

-- Definitions based on conditions
def total_employees : ℕ := 120
def male_employees : ℕ := 90
def female_employees_in_sample : ℕ := 3

-- Proof statement
theorem stratified_sampling_sample_size : total_employees = 120 ∧ male_employees = 90 ∧ female_employees_in_sample = 3 → 
  (female_employees_in_sample + female_employees_in_sample * (male_employees / (total_employees - male_employees))) = 12 :=
sorry

end stratified_sampling_sample_size_l202_202246


namespace linear_function_diff_l202_202379

noncomputable def g : ℝ → ℝ := sorry

theorem linear_function_diff (h_linear : ∀ x y z w : ℝ, (g y - g x) / (y - x) = (g w - g z) / (w - z))
                            (h_condition : g 8 - g 1 = 21) : 
  g 16 - g 1 = 45 := 
by 
  sorry

end linear_function_diff_l202_202379


namespace sequence_general_formula_l202_202090

theorem sequence_general_formula (a : ℕ → ℕ)
    (h1 : a 1 = 1)
    (h2 : a 2 = 2)
    (h3 : ∀ n, a (n + 2) = a n + 2) :
    ∀ n, a n = n := by
  sorry

end sequence_general_formula_l202_202090


namespace digital_earth_functionalities_l202_202089

def digital_earth_allows_internet_navigation : Prop := 
  ∀ (f : String), f ∈ ["Receive distance education", "Shop online", "Seek medical advice online"]

def digital_earth_does_not_allow_physical_travel : Prop := 
  ¬ (∀ (f : String), f ∈ ["Travel around the world"])

theorem digital_earth_functionalities :
  digital_earth_allows_internet_navigation ∧ digital_earth_does_not_allow_physical_travel →
  ∀(f : String), f ∈ ["Receive distance education", "Shop online", "Seek medical advice online"] :=
by
  sorry

end digital_earth_functionalities_l202_202089


namespace symmetric_point_exists_l202_202508

-- Define the point P and line equation.
structure Point (α : Type*) := (x : α) (y : α)
def P : Point ℝ := ⟨5, -2⟩
def line_eq (x y : ℝ) : Prop := x - y + 5 = 0

-- Define a function for the line PQ being perpendicular to the given line.
def is_perpendicular (P Q : Point ℝ) : Prop :=
  (Q.y - P.y) / (Q.x - P.x) = -1

-- Define a function for the midpoint of PQ lying on the given line.
def midpoint_on_line (P Q : Point ℝ) : Prop :=
  line_eq ((P.x + Q.x) / 2) ((P.y + Q.y) / 2)

-- Define the symmetry function based on the provided conditions.
def is_symmetric (Q : Point ℝ) : Prop :=
  is_perpendicular P Q ∧ midpoint_on_line P Q

-- State the main theorem to be proved: there exists a point Q that satisfies the 
-- conditions and is symmetric to P with respect to the given line.
theorem symmetric_point_exists : ∃ Q : Point ℝ, is_symmetric Q ∧ Q = ⟨-7, 10⟩ :=
by
  sorry

end symmetric_point_exists_l202_202508


namespace ratio_a_to_c_l202_202833

variables {a b c d : ℚ}

theorem ratio_a_to_c
  (h1 : a / b = 5 / 2)
  (h2 : c / d = 4 / 1)
  (h3 : d / b = 3 / 10) :
  a / c = 25 / 12 :=
sorry

end ratio_a_to_c_l202_202833


namespace polynomial_non_negative_l202_202940

theorem polynomial_non_negative (x : ℝ) : x^8 + x^6 - 4*x^4 + x^2 + 1 ≥ 0 := 
sorry

end polynomial_non_negative_l202_202940


namespace flat_fee_rate_l202_202541

-- Definitions for the variables
variable (F n : ℝ)

-- Conditions based on the problem statement
axiom mark_cost : F + 4.6 * n = 310
axiom lucy_cost : F + 6.2 * n = 410

-- Problem Statement
theorem flat_fee_rate : F = 22.5 ∧ n = 62.5 :=
by
  sorry

end flat_fee_rate_l202_202541


namespace total_suitcases_correct_l202_202825

-- Conditions as definitions
def num_siblings : Nat := 4
def suitcases_per_sibling : Nat := 2
def num_parents : Nat := 2
def suitcases_per_parent : Nat := 3

-- Total suitcases calculation
def total_suitcases :=
  (num_siblings * suitcases_per_sibling) + (num_parents * suitcases_per_parent)

-- Statement to prove
theorem total_suitcases_correct : total_suitcases = 14 :=
by
  sorry

end total_suitcases_correct_l202_202825


namespace m_value_quadratic_l202_202477

theorem m_value_quadratic (m : ℝ)
  (h1 : |m - 2| = 2)
  (h2 : m - 4 ≠ 0) :
  m = 0 :=
sorry

end m_value_quadratic_l202_202477


namespace Harkamal_purchase_grapes_l202_202566

theorem Harkamal_purchase_grapes
  (G : ℕ) -- The number of kilograms of grapes
  (cost_grapes_per_kg : ℕ := 70)
  (kg_mangoes : ℕ := 9)
  (cost_mangoes_per_kg : ℕ := 55)
  (total_paid : ℕ := 1195) :
  70 * G + 55 * 9 = 1195 → G = 10 := 
by
  sorry

end Harkamal_purchase_grapes_l202_202566


namespace determine_m_l202_202095

theorem determine_m (a b c m : ℤ) 
  (h1 : c = -4 * a - 2 * b)
  (h2 : 70 < 4 * (8 * a + b) ∧ 4 * (8 * a + b) < 80)
  (h3 : 110 < 5 * (9 * a + b) ∧ 5 * (9 * a + b) < 120)
  (h4 : 2000 * m < (2500 * a + 50 * b + c) ∧ (2500 * a + 50 * b + c) < 2000 * (m + 1)) :
  m = 5 := sorry

end determine_m_l202_202095


namespace total_books_on_shelves_l202_202776

def num_shelves : ℕ := 520
def books_per_shelf : ℝ := 37.5

theorem total_books_on_shelves : num_shelves * books_per_shelf = 19500 :=
by
  sorry

end total_books_on_shelves_l202_202776


namespace bottle_caps_sum_l202_202526

theorem bottle_caps_sum : 
  let starting_caps := 91
  let found_caps := 88
  starting_caps + found_caps = 179 :=
by
  sorry

end bottle_caps_sum_l202_202526


namespace correct_operations_result_l202_202677

theorem correct_operations_result (n : ℕ) 
  (h1 : n / 8 - 12 = 32) : (n * 8 + 12 = 2828) :=
sorry

end correct_operations_result_l202_202677


namespace x_minus_y_eq_2_l202_202198

theorem x_minus_y_eq_2 (x y : ℝ) (h1 : 2 * x + 3 * y = 9) (h2 : 3 * x + 2 * y = 11) : x - y = 2 :=
sorry

end x_minus_y_eq_2_l202_202198


namespace whale_tongue_weight_difference_l202_202971

noncomputable def tongue_weight_blue_whale_kg : ℝ := 2700
noncomputable def tongue_weight_fin_whale_kg : ℝ := 1800
noncomputable def kg_to_pounds : ℝ := 2.20462
noncomputable def ton_to_pounds : ℝ := 2000

noncomputable def tongue_weight_blue_whale_tons := (tongue_weight_blue_whale_kg * kg_to_pounds) / ton_to_pounds
noncomputable def tongue_weight_fin_whale_tons := (tongue_weight_fin_whale_kg * kg_to_pounds) / ton_to_pounds
noncomputable def weight_difference_tons := tongue_weight_blue_whale_tons - tongue_weight_fin_whale_tons

theorem whale_tongue_weight_difference :
  weight_difference_tons = 0.992079 :=
by
  sorry

end whale_tongue_weight_difference_l202_202971


namespace average_weight_of_remaining_carrots_l202_202736

noncomputable def total_weight_30_carrots : ℕ := 5940
noncomputable def total_weight_3_carrots : ℕ := 540
noncomputable def carrots_count_30 : ℕ := 30
noncomputable def carrots_count_3_removed : ℕ := 3
noncomputable def carrots_count_remaining : ℕ := 27
noncomputable def average_weight_of_removed_carrots : ℕ := 180

theorem average_weight_of_remaining_carrots :
  (total_weight_30_carrots - total_weight_3_carrots) / carrots_count_remaining = 200 :=
  by
  sorry

end average_weight_of_remaining_carrots_l202_202736


namespace eccentricity_of_hyperbola_l202_202397

noncomputable def hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) (c : ℝ)
  (hc : c^2 = a^2 + b^2) : ℝ :=
  (1 + Real.sqrt 5) / 2

theorem eccentricity_of_hyperbola (a b c e : ℝ)
  (ha : a > 0) (hb : b > 0) (h_hyperbola : c^2 = a^2 + b^2)
  (h_eccentricity : e = (1 + Real.sqrt 5) / 2) :
  e = hyperbola_eccentricity a b ha hb c h_hyperbola :=
by
  sorry

end eccentricity_of_hyperbola_l202_202397


namespace no_positive_integers_satisfy_l202_202435

theorem no_positive_integers_satisfy (a b : ℕ) (ha : a > 0) (hb : b > 0) : 
  ¬ (3 * a^2 = b^2 + 1) := 
sorry

end no_positive_integers_satisfy_l202_202435


namespace find_a_l202_202300

open Set

noncomputable def A (a : ℝ) : Set ℝ := {x | x^2 - a*x + a^2 - 19 = 0}
def B : Set ℝ := {x | x^2 - 5*x + 6 = 0}
def C : Set ℝ := {x | x^2 + 2*x - 8 = 0}

theorem find_a (a : ℝ) :
  ∅ ⊂ (A a ∩ B) ∧ A a ∩ C = ∅ → a = -2 :=
by
  sorry

end find_a_l202_202300


namespace place_two_after_three_digit_number_l202_202199

theorem place_two_after_three_digit_number (h t u : ℕ) 
  (Hh : h < 10) (Ht : t < 10) (Hu : u < 10) : 
  (100 * h + 10 * t + u) * 10 + 2 = 1000 * h + 100 * t + 10 * u + 2 := 
by
  sorry

end place_two_after_three_digit_number_l202_202199


namespace each_person_towel_day_l202_202678

def total_people (families : ℕ) (members_per_family : ℕ) : ℕ :=
  families * members_per_family

def total_towels (loads : ℕ) (towels_per_load : ℕ) : ℕ :=
  loads * towels_per_load

def towels_per_day (total_towels : ℕ) (days : ℕ) : ℕ :=
  total_towels / days

def towels_per_person_per_day (towels_per_day : ℕ) (total_people : ℕ) : ℕ :=
  towels_per_day / total_people

theorem each_person_towel_day
  (families : ℕ) (members_per_family : ℕ) (days : ℕ) (loads : ℕ) (towels_per_load : ℕ)
  (h_family : families = 3) (h_members : members_per_family = 4) (h_days : days = 7)
  (h_loads : loads = 6) (h_towels_per_load : towels_per_load = 14) :
  towels_per_person_per_day (towels_per_day (total_towels loads towels_per_load) days) (total_people families members_per_family) = 1 :=
by {
  -- Import necessary assumptions
  sorry
}

end each_person_towel_day_l202_202678


namespace problem1_problem2_l202_202810

-- Problem 1: Prove that (a/(a - b)) + (b/(b - a)) = 1
theorem problem1 (a b : ℝ) (h : a ≠ b) : (a / (a - b)) + (b / (b - a)) = 1 := 
sorry

-- Problem 2: Prove that (a^2 / (b^2 * c)) * (- (b * c^2) / (2 * a)) / (a / b) = -c
theorem problem2 (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) : 
  (a^2 / (b^2 * c)) * (- (b * c^2) / (2 * a)) / (a / b) = -c :=
sorry

end problem1_problem2_l202_202810


namespace find_y_l202_202513

theorem find_y (t : ℚ) (x y : ℚ) (h1 : x = 3 - 2 * t) (h2 : y = 3 * t + 10) (hx : x = -4) : y = 41 / 2 :=
by
  sorry

end find_y_l202_202513


namespace tan_alpha_plus_pi_over_4_l202_202887

noncomputable def vec_a (α : ℝ) : ℝ × ℝ := (Real.cos (2 * α), Real.sin α)
noncomputable def vec_b (α : ℝ) : ℝ × ℝ := (1, 2 * Real.sin α - 1)

def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

theorem tan_alpha_plus_pi_over_4 (α : ℝ) (h1 : 0 < α) (h2 : α < Real.pi)
    (h3 : dot_product (vec_a α) (vec_b α) = 0) :
    Real.tan (α + Real.pi / 4) = -1 := sorry

end tan_alpha_plus_pi_over_4_l202_202887


namespace value_of_f_g6_minus_g_f6_l202_202110

def f (x : ℝ) : ℝ := x^2 - 3 * x + 4
def g (x : ℝ) : ℝ := x + 4

theorem value_of_f_g6_minus_g_f6 : f (g 6) - g (f 6) = 48 :=
by
  sorry

end value_of_f_g6_minus_g_f6_l202_202110


namespace borrowing_methods_l202_202855

theorem borrowing_methods (A_has_3_books : True) (B_borrows_at_least_one_book : True) :
  (∃ (methods : ℕ), methods = 7) :=
by
  existsi 7
  sorry

end borrowing_methods_l202_202855


namespace total_candy_given_l202_202622

def candy_given_total (a b c : ℕ) : ℕ := a + b + c

def first_10_friends_candy (n : ℕ) := 10 * n

def next_7_friends_candy (n : ℕ) := 7 * (2 * n)

def remaining_friends_candy := 50

theorem total_candy_given (n : ℕ) (h1 : first_10_friends_candy 12 = 120)
  (h2 : next_7_friends_candy 12 = 168) (h3 : remaining_friends_candy = 50) :
  candy_given_total 120 168 50 = 338 := by
  sorry

end total_candy_given_l202_202622


namespace rectangle_segments_sum_l202_202804

theorem rectangle_segments_sum :
  let EF := 6
  let FG := 8
  let n := 210
  let diagonal_length := Real.sqrt (EF^2 + FG^2)
  let segment_length (k : ℕ) : ℝ := diagonal_length * (n - k) / n
  let sum_segments := 2 * (Finset.sum (Finset.range 210) segment_length) - diagonal_length
  sum_segments = 2080 := by
  sorry

end rectangle_segments_sum_l202_202804


namespace mutually_prime_sum_l202_202292

open Real

theorem mutually_prime_sum (A B C : ℤ) (h_prime : Int.gcd A (Int.gcd B C) = 1)
    (h_eq : A * log 5 / log 200 + B * log 2 / log 200 = C) : A + B + C = 6 := 
sorry

end mutually_prime_sum_l202_202292


namespace f_identically_zero_l202_202800

open Real

-- Define the function f and its properties
noncomputable def f : ℝ → ℝ := sorry

-- Given conditions
axiom func_eqn (a b : ℝ) : f (a * b) = a * f b + b * f a 
axiom func_bounded (x : ℝ) : |f x| ≤ 1

-- Goal: Prove that f is identically zero
theorem f_identically_zero : ∀ x : ℝ, f x = 0 := 
by
  sorry

end f_identically_zero_l202_202800


namespace rate_per_sqm_is_correct_l202_202621

-- Definitions of the problem conditions
def room_length : ℝ := 10
def room_width : ℝ := 7
def room_height : ℝ := 5

def door_width : ℝ := 1
def door_height : ℝ := 3

def window1_width : ℝ := 2
def window1_height : ℝ := 1.5
def window2_width : ℝ := 1
def window2_height : ℝ := 1.5

def number_of_doors : ℕ := 2
def number_of_window2 : ℕ := 2

def total_cost : ℝ := 474

-- Our goal is to prove this rate
def expected_rate_per_sqm : ℝ := 3

-- Wall area calculations
def wall_area : ℝ :=
  2 * (room_length * room_height) + 2 * (room_width * room_height)

def doors_area : ℝ :=
  number_of_doors * (door_width * door_height)

def window1_area : ℝ :=
  window1_width * window1_height

def window2_area : ℝ :=
  number_of_window2 * (window2_width * window2_height)

def total_unpainted_area : ℝ :=
  doors_area + window1_area + window2_area

def paintable_area : ℝ :=
  wall_area - total_unpainted_area

-- Proof goal
theorem rate_per_sqm_is_correct : total_cost / paintable_area = expected_rate_per_sqm :=
by
  sorry

end rate_per_sqm_is_correct_l202_202621


namespace dorothy_profit_l202_202482

-- Define the conditions
def expense := 53
def number_of_doughnuts := 25
def price_per_doughnut := 3

-- Define revenue and profit calculations
def revenue := number_of_doughnuts * price_per_doughnut
def profit := revenue - expense

-- Prove the profit calculation
theorem dorothy_profit : profit = 22 := by
  sorry

end dorothy_profit_l202_202482


namespace solution_set_of_inequality_l202_202553

theorem solution_set_of_inequality :
  {x : ℝ | |x^2 - 2| < 2} = {x : ℝ | (x > -2 ∧ x < 0) ∨ (x > 0 ∧ x < 2)} :=
by
  sorry

end solution_set_of_inequality_l202_202553


namespace jordan_machine_solution_l202_202185

theorem jordan_machine_solution (x : ℝ) (h : 2 * x + 3 - 5 = 27) : x = 14.5 :=
sorry

end jordan_machine_solution_l202_202185


namespace number_of_rows_in_theater_l202_202261

theorem number_of_rows_in_theater 
  (x : ℕ)
  (h1 : ∀ (students : ℕ), students = 30 → ∃ row : ℕ, row < x ∧ ∃ a b : ℕ, a ≠ b ∧ row = a ∧ row = b)
  (h2 : ∀ (students : ℕ), students = 26 → ∃ empties : ℕ, empties ≥ 3 ∧ x - students = empties)
  : x = 29 :=
by
  sorry

end number_of_rows_in_theater_l202_202261


namespace b_profit_l202_202650

noncomputable def profit_share (x t : ℝ) : ℝ :=
  let total_profit := 31500
  let a_investment := 3 * x
  let a_period := 2 * t
  let b_investment := x
  let b_period := t
  let profit_ratio_a := a_investment * a_period
  let profit_ratio_b := b_investment * b_period
  let total_ratio := profit_ratio_a + profit_ratio_b
  let b_share := profit_ratio_b / total_ratio
  b_share * total_profit

theorem b_profit (x t : ℝ) : profit_share x t = 4500 :=
by
  sorry

end b_profit_l202_202650


namespace wine_problem_solution_l202_202770

theorem wine_problem_solution (x : ℝ) (h1 : 0 ≤ x ∧ x ≤ 200) (h2 : (200 - x) * (180 - x) / 200 = 144) : x = 20 := 
by
  sorry

end wine_problem_solution_l202_202770


namespace arithmetic_problem_l202_202467

noncomputable def arithmetic_progression (a₁ d : ℝ) (n : ℕ) := a₁ + (n - 1) * d

noncomputable def sum_terms (a₁ d : ℝ) (n : ℕ) : ℝ :=
  n * (2 * a₁ + (n - 1) * d) / 2

theorem arithmetic_problem (a₁ d : ℝ)
  (h₁ : a₁ + (a₁ + 2 * d) = 5)
  (h₂ : 4 * (2 * a₁ + 3 * d) / 2 = 20) :
  (sum_terms a₁ d 8 - 2 * sum_terms a₁ d 4) / (sum_terms a₁ d 6 - sum_terms a₁ d 4 - sum_terms a₁ d 2) = 10 := by
  sorry

end arithmetic_problem_l202_202467


namespace wire_length_between_poles_l202_202847

theorem wire_length_between_poles :
  let x_dist := 20
  let y_dist := (18 / 2) - 8
  (x_dist ^ 2 + y_dist ^ 2 = 401) :=
by
  sorry

end wire_length_between_poles_l202_202847


namespace good_walker_catch_up_l202_202339

theorem good_walker_catch_up :
  ∀ x y : ℕ, 
    (x = (100:ℕ) + y) ∧ (x = ((100:ℕ)/(60:ℕ) : ℚ) * y) := 
by
  sorry

end good_walker_catch_up_l202_202339


namespace math_problem_l202_202067

variable {p q r x y : ℝ}

theorem math_problem (h1 : p / q = 6 / 7)
                     (h2 : p / r = 8 / 9)
                     (h3 : q / r = x / y) :
                     x = 28 ∧ y = 27 ∧ 2 * p + q = (19 / 6) * p := 
by 
  sorry

end math_problem_l202_202067


namespace find_k_l202_202014

-- The expression in terms of x, y, and k
def expression (k x y : ℝ) :=
  4 * x^2 - 6 * k * x * y + (3 * k^2 + 2) * y^2 - 4 * x - 4 * y + 6

-- The mathematical statement to be proved
theorem find_k : ∃ k : ℝ, (∀ x y : ℝ, expression k x y ≥ 0) ∧ (∃ (x y : ℝ), expression k x y = 0) :=
sorry

end find_k_l202_202014


namespace line_through_point_equal_intercepts_l202_202449

-- Definitions based on conditions
def passes_through (p : ℝ × ℝ) (l : ℝ → ℝ → Prop) : Prop :=
  l p.1 p.2

def equal_intercepts (l : ℝ → ℝ → Prop) : Prop :=
  ∃ a, a ≠ 0 ∧ (∀ x y, l x y ↔ x + y = a) ∨ (∀ x y, l x y ↔ y = 2 * x)

-- Theorem statement based on the problem
theorem line_through_point_equal_intercepts :
  ∃ l, passes_through (1, 2) l ∧ equal_intercepts l ∧
  (∀ x y, l x y ↔ 2 * x - y = 0) ∨ (∀ x y, l x y ↔ x + y - 3 = 0) :=
sorry

end line_through_point_equal_intercepts_l202_202449


namespace tangent_line_eq_l202_202577

open Real

noncomputable def f (x : ℝ) : ℝ := exp x * log x

theorem tangent_line_eq (x y : ℝ) (h : x = 1 ∧ y = 0) :
  ∃ m b, (∀ t, y = m * (t - 1) + b) ∧ (f x = y) ∧ (m = exp 1) ∧ (b = -exp 1) :=
by
  sorry

end tangent_line_eq_l202_202577


namespace relatively_prime_solutions_l202_202352

theorem relatively_prime_solutions  (x y : ℤ) (h_rel_prime : gcd x y = 1) : 
  2 * (x^3 - x) = 5 * (y^3 - y) ↔ 
  (x = 0 ∧ (y = 1 ∨ y = -1)) ∨ 
  (x = 1 ∧ y = 0) ∨
  (x = -1 ∧ y = 0) ∨
  (x = 4 ∧ (y = 3 ∨ y = -3)) ∨ 
  (x = -4 ∧ (y = -3 ∨ y = 3)) ∨
  (x = 1 ∧ y = -1) ∨
  (x = -1 ∧ y = 1) ∨
  (x = 0 ∧ y = 0) :=
by sorry

end relatively_prime_solutions_l202_202352


namespace machining_defect_probability_l202_202885

theorem machining_defect_probability :
  let defect_rate_process1 := 0.03
  let defect_rate_process2 := 0.05
  let non_defective_rate_process1 := 1 - defect_rate_process1
  let non_defective_rate_process2 := 1 - defect_rate_process2
  let non_defective_rate := non_defective_rate_process1 * non_defective_rate_process2
  let defective_rate := 1 - non_defective_rate
  defective_rate = 0.0785 :=
by
  sorry

end machining_defect_probability_l202_202885


namespace white_area_of_painting_l202_202197

theorem white_area_of_painting (s : ℝ) (total_gray_area : ℝ) (gray_area_squares : ℕ)
  (h1 : ∀ t, t = 3 * s) -- The frame is 3 times the smaller square's side length.
  (h2 : total_gray_area = 62) -- The gray area is 62 cm^2.
  (h3 : gray_area_squares = 31) -- The gray area is composed of 31 smaller squares.
  : ∃ white_area, white_area = 10 := 
  sorry

end white_area_of_painting_l202_202197


namespace count_implications_l202_202221

def r : Prop := sorry
def s : Prop := sorry

def statement_1 := ¬r ∧ ¬s
def statement_2 := ¬r ∧ s
def statement_3 := r ∧ ¬s
def statement_4 := r ∧ s

def neg_rs : Prop := r ∨ s

theorem count_implications : (statement_2 → neg_rs) ∧ 
                             (statement_3 → neg_rs) ∧ 
                             (statement_4 → neg_rs) ∧ 
                             (¬(statement_1 → neg_rs)) -> 
                             3 = 3 := by
  sorry

end count_implications_l202_202221


namespace midpoint_of_interception_l202_202528

theorem midpoint_of_interception (x1 x2 y1 y2 : ℝ) 
  (h1 : y1^2 = 4 * x1) 
  (h2 : y2^2 = 4 * x2) 
  (h3 : y1 = x1 - 1) 
  (h4 : y2 = x2 - 1) : 
  ( (x1 + x2) / 2, (y1 + y2) / 2 ) = (3, 2) :=
by 
  sorry

end midpoint_of_interception_l202_202528


namespace find_CB_l202_202662

variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (a b : V)

-- Given condition
-- D divides AB in the ratio 1:3 such that CA = a and CD = b

def D_divides_AB (A B D : V) : Prop := ∃ (k : ℝ), k = 1 / 4 ∧ A + k • (B - A) = D

theorem find_CB (CA CD : V) (A B D : V) (h1 : CA = A) (h2 : CD = B)
  (h3 : D_divides_AB A B D) : (B - A) = -3 • CA + 4 • CD :=
sorry

end find_CB_l202_202662


namespace arithmetic_sequence_sum_20_l202_202695

open BigOperators

def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
∀ n, a (n + 1) = a n + (a 1 - a 0)

theorem arithmetic_sequence_sum_20 {a : ℕ → ℤ} (h_arith : is_arithmetic_sequence a)
    (h1 : a 0 + a 1 + a 2 = -24)
    (h18 : a 17 + a 18 + a 19 = 78) :
    ∑ i in Finset.range 20, a i = 180 :=
sorry

end arithmetic_sequence_sum_20_l202_202695


namespace pelican_count_in_shark_bite_cove_l202_202812

theorem pelican_count_in_shark_bite_cove
  (num_sharks_pelican_bay : ℕ)
  (num_pelicans_shark_bite_cove : ℕ)
  (num_pelicans_moved : ℕ) :
  num_sharks_pelican_bay = 60 →
  num_sharks_pelican_bay = 2 * num_pelicans_shark_bite_cove →
  num_pelicans_moved = num_pelicans_shark_bite_cove / 3 →
  num_pelicans_shark_bite_cove - num_pelicans_moved = 20 :=
by
  sorry

end pelican_count_in_shark_bite_cove_l202_202812


namespace sum_of_consecutive_multiples_of_4_l202_202133

theorem sum_of_consecutive_multiples_of_4 (n : ℝ) (h : 4 * n + (4 * n + 8) = 140) :
  4 * n + (4 * n + 4) + (4 * n + 8) = 210 :=
sorry

end sum_of_consecutive_multiples_of_4_l202_202133


namespace solve_for_x_l202_202345

theorem solve_for_x (x : ℝ) (h : 0.05 * x + 0.07 * (25 + x) = 15.1) : x = 111.25 :=
by
  sorry

end solve_for_x_l202_202345


namespace n_cube_plus_5n_divisible_by_6_l202_202172

theorem n_cube_plus_5n_divisible_by_6 (n : ℤ) : 6 ∣ (n^3 + 5 * n) := 
sorry

end n_cube_plus_5n_divisible_by_6_l202_202172


namespace problem1_problem2_problem3_l202_202040

-- First problem
theorem problem1 : 24 - |(-2)| + (-16) - 8 = -2 := by
  sorry

-- Second problem
theorem problem2 : (-2) * (3 / 2) / (-3 / 4) * 4 = 4 := by
  sorry

-- Third problem
theorem problem3 : -1^2016 - (1 - 0.5) / 3 * (2 - (-3)^2) = 1 / 6 := by
  sorry

end problem1_problem2_problem3_l202_202040


namespace cost_price_6500_l202_202142

variable (CP SP : ℝ)

-- Condition 1: The selling price is 30% more than the cost price.
def selling_price (CP : ℝ) : ℝ := CP * 1.3

-- Condition 2: The selling price is Rs. 8450.
axiom selling_price_8450 : selling_price CP = 8450

-- Prove that the cost price of the computer table is Rs. 6500.
theorem cost_price_6500 : CP = 6500 :=
by
  sorry

end cost_price_6500_l202_202142


namespace kendra_minivans_l202_202557

theorem kendra_minivans (afternoon: ℕ) (evening: ℕ) (h1: afternoon = 4) (h2: evening = 1) : afternoon + evening = 5 :=
by sorry

end kendra_minivans_l202_202557


namespace a2b_sub_ab2_eq_neg16sqrt5_l202_202399

noncomputable def a : ℝ := 4 + 2 * Real.sqrt 5
noncomputable def b : ℝ := 4 - 2 * Real.sqrt 5

theorem a2b_sub_ab2_eq_neg16sqrt5 : a^2 * b - a * b^2 = -16 * Real.sqrt 5 :=
by
  sorry

end a2b_sub_ab2_eq_neg16sqrt5_l202_202399


namespace max_checkers_on_chessboard_l202_202726

theorem max_checkers_on_chessboard : 
  ∃ (w b : ℕ), (∀ r c : ℕ, r < 8 ∧ c < 8 → w = 2 * b) ∧ (8 * (w + b) = 48) ∧ (w + b) * 8 ≤ 64 :=
by sorry

end max_checkers_on_chessboard_l202_202726


namespace total_nails_polished_l202_202908

-- Defining the number of girls
def num_girls : ℕ := 5

-- Defining the number of fingers and toes per person
def num_fingers_per_person : ℕ := 10
def num_toes_per_person : ℕ := 10

-- Defining the total number of nails per person
def nails_per_person : ℕ := num_fingers_per_person + num_toes_per_person

-- The theorem stating that the total number of nails polished for 5 girls is 100 nails
theorem total_nails_polished : num_girls * nails_per_person = 100 := by
  sorry

end total_nails_polished_l202_202908


namespace cost_of_6_bottle_caps_l202_202656

-- Define the cost of each bottle cap
def cost_per_bottle_cap : ℕ := 2

-- Define how many bottle caps we are buying
def number_of_bottle_caps : ℕ := 6

-- Define the total cost of the bottle caps
def total_cost : ℕ := 12

-- The proof statement to prove that the total cost is as expected
theorem cost_of_6_bottle_caps :
  cost_per_bottle_cap * number_of_bottle_caps = total_cost :=
by
  sorry

end cost_of_6_bottle_caps_l202_202656


namespace rational_solutions_of_quadratic_l202_202319

theorem rational_solutions_of_quadratic (k : ℕ) (h_positive : k > 0) :
  (∃ p q : ℚ, p * p + 30 * p * q + k * (q * q) = 0) ↔ k = 9 ∨ k = 15 :=
sorry

end rational_solutions_of_quadratic_l202_202319


namespace proper_sampling_method_l202_202396

-- Definitions for conditions
def large_bulbs : ℕ := 120
def medium_bulbs : ℕ := 60
def small_bulbs : ℕ := 20
def sample_size : ℕ := 25

-- Definition for the proper sampling method to use
def sampling_method : String := "Stratified sampling"

-- Theorem statement to prove the sampling method
theorem proper_sampling_method :
  ∃ method : String, 
  method = sampling_method ∧
  sampling_method = "Stratified sampling" := by
    sorry

end proper_sampling_method_l202_202396


namespace perimeter_of_shaded_region_l202_202502

noncomputable def circle_center : Type := sorry -- Define the object type for circle's center
noncomputable def radius_length : ℝ := 10 -- Define the radius length as 10
noncomputable def central_angle : ℝ := 270 -- Define the central angle corresponding to the arc RS

-- Function to calculate the perimeter of the shaded region
noncomputable def perimeter_shaded_region (radius : ℝ) (angle : ℝ) : ℝ :=
  2 * radius + (angle / 360) * 2 * Real.pi * radius

-- Theorem stating that the perimeter of the shaded region is 20 + 15π given the conditions
theorem perimeter_of_shaded_region : 
  perimeter_shaded_region radius_length central_angle = 20 + 15 * Real.pi :=
by
  -- skipping the actual proof
  sorry

end perimeter_of_shaded_region_l202_202502


namespace gcd_3570_4840_l202_202421

-- Define the numbers
def num1 : Nat := 3570
def num2 : Nat := 4840

-- Define the problem statement
theorem gcd_3570_4840 : Nat.gcd num1 num2 = 10 := by
  sorry

end gcd_3570_4840_l202_202421


namespace rectangle_diagonals_equal_rhombus_not_l202_202732

/-- Define the properties for a rectangle -/
structure Rectangle :=
  (sides_parallel : Prop)
  (diagonals_equal : Prop)
  (diagonals_bisect : Prop)
  (angles_equal : Prop)

/-- Define the properties for a rhombus -/
structure Rhombus :=
  (sides_parallel : Prop)
  (diagonals_equal : Prop)
  (diagonals_bisect : Prop)
  (angles_equal : Prop)

/-- The property that distinguishes a rectangle from a rhombus is that the diagonals are equal. -/
theorem rectangle_diagonals_equal_rhombus_not
  (R : Rectangle)
  (H : Rhombus)
  (hR1 : R.sides_parallel)
  (hR2 : R.diagonals_equal)
  (hR3 : R.diagonals_bisect)
  (hR4 : R.angles_equal)
  (hH1 : H.sides_parallel)
  (hH2 : ¬H.diagonals_equal)
  (hH3 : H.diagonals_bisect)
  (hH4 : H.angles_equal) :
  (R.diagonals_equal) := by
  sorry

end rectangle_diagonals_equal_rhombus_not_l202_202732


namespace quadratic_residues_count_l202_202533

theorem quadratic_residues_count (p : ℕ) (hp : Nat.Prime p) (hp2 : p % 2 = 1) :
  ∃ (q_residues : Finset (ZMod p)), q_residues.card = (p - 1) / 2 ∧
  ∃ (nq_residues : Finset (ZMod p)), nq_residues.card = (p - 1) / 2 ∧
  ∀ d ∈ q_residues, ∃ x y : ZMod p, x^2 = d ∧ y^2 = d ∧ x ≠ y :=
by
  sorry

end quadratic_residues_count_l202_202533


namespace number_of_shirts_proof_l202_202521

def regular_price := 50
def discount_percentage := 20
def total_paid := 240

def sale_price (rp : ℕ) (dp : ℕ) : ℕ := rp * (100 - dp) / 100

def number_of_shirts (tp : ℕ) (sp : ℕ) : ℕ := tp / sp

theorem number_of_shirts_proof : 
  number_of_shirts total_paid (sale_price regular_price discount_percentage) = 6 :=
by 
  sorry

end number_of_shirts_proof_l202_202521


namespace temperature_difference_on_day_xianning_l202_202854

theorem temperature_difference_on_day_xianning 
  (highest_temp : ℝ) (lowest_temp : ℝ) 
  (h_highest : highest_temp = 2) (h_lowest : lowest_temp = -3) : 
  highest_temp - lowest_temp = 5 := 
by
  sorry

end temperature_difference_on_day_xianning_l202_202854


namespace find_a_l202_202576

open Set

-- Define set A
def A : Set ℝ := {-1, 1, 3}

-- Define set B in terms of a
def B (a : ℝ) : Set ℝ := {a + 2, a^2 + 4}

-- State the theorem
theorem find_a (a : ℝ) (h : A ∩ B a = {3}) : a = 1 :=
sorry

end find_a_l202_202576


namespace problem_statement_l202_202673

def digit_sum (k : ℕ) : ℕ :=
  k.digits 10 |>.sum

theorem problem_statement :
  ∀ n : ℕ, (∃ a b : ℕ, n = digit_sum a ∧ n = digit_sum b ∧ n = digit_sum (a + b)) ↔ (∃ k : ℕ, n = 9 * k) :=
by
  sorry

end problem_statement_l202_202673


namespace combinedTotalSandcastlesAndTowers_l202_202652

def markSandcastles : Nat := 20
def towersPerMarkSandcastle : Nat := 10
def jeffSandcastles : Nat := 3 * markSandcastles
def towersPerJeffSandcastle : Nat := 5

theorem combinedTotalSandcastlesAndTowers :
  (markSandcastles + markSandcastles * towersPerMarkSandcastle) +
  (jeffSandcastles + jeffSandcastles * towersPerJeffSandcastle) = 580 :=
by
  sorry

end combinedTotalSandcastlesAndTowers_l202_202652


namespace Emily_total_points_l202_202087

-- Definitions of the points scored in each round
def round1_points := 16
def round2_points := 32
def round3_points := -27
def round4_points := 92
def round5_points := 4

-- Total points calculation in Lean
def total_points := round1_points + round2_points + round3_points + round4_points + round5_points

-- Lean statement to prove total points at the end of the game
theorem Emily_total_points : total_points = 117 :=
by 
  -- Unfold the definition of total_points and simplify
  unfold total_points round1_points round2_points round3_points round4_points round5_points
  -- Simplify the expression
  sorry

end Emily_total_points_l202_202087


namespace probability_sin_cos_in_range_l202_202797

noncomputable def probability_sin_cos_interval : ℝ :=
  let interval_length := (Real.pi / 2 + Real.pi / 6)
  let valid_length := (Real.pi / 2 - 0)
  valid_length / interval_length

theorem probability_sin_cos_in_range :
  probability_sin_cos_interval = 3 / 4 :=
sorry

end probability_sin_cos_in_range_l202_202797


namespace christopher_age_l202_202717

theorem christopher_age (G C : ℕ) (h1 : C = 2 * G) (h2 : C - 9 = 5 * (G - 9)) : C = 24 := 
by
  sorry

end christopher_age_l202_202717


namespace abc_inequality_l202_202863

theorem abc_inequality (a b c : ℝ) (h1 : a ≥ -1) (h2 : b ≥ -1) (h3 : c ≥ -1) (h4 : a^3 + b^3 + c^3 = 1) : 
  a + b + c + a^2 + b^2 + c^2 ≤ 4 := 
sorry

end abc_inequality_l202_202863


namespace Clarence_total_oranges_l202_202897

def Clarence_oranges_initial := 5
def oranges_from_Joyce := 3

theorem Clarence_total_oranges : Clarence_oranges_initial + oranges_from_Joyce = 8 := by
  sorry

end Clarence_total_oranges_l202_202897


namespace min_rounds_needed_l202_202003

-- Defining the number of players
def num_players : ℕ := 10

-- Defining the number of matches each player plays per round
def matches_per_round (n : ℕ) : ℕ := n / 2

-- Defining the scoring system
def win_points : ℝ := 1
def draw_points : ℝ := 0.5
def loss_points : ℝ := 0

-- Defining the total number of rounds needed for a clear winner to emerge
def min_rounds_for_winner : ℕ := 7

-- Theorem stating the minimum number of rounds required
theorem min_rounds_needed :
  ∀ (n : ℕ), n = num_players → (∃ r : ℕ, r = min_rounds_for_winner) :=
by
  intros n hn
  existsi min_rounds_for_winner
  sorry

end min_rounds_needed_l202_202003


namespace remainder_of_m_div_1000_l202_202603

   -- Define the set T
   def T : Set ℕ := {n | 1 ≤ n ∧ n ≤ 12}

   -- Define the computation of m
   noncomputable def m : ℕ := (3^12 - 2 * 2^12 + 1) / 2

   -- Statement for the proof problem
   theorem remainder_of_m_div_1000 : m % 1000 = 625 := by
     sorry
   
end remainder_of_m_div_1000_l202_202603


namespace population_present_l202_202285

theorem population_present (P : ℝ) (h : P * (1.1)^3 = 79860) : P = 60000 :=
sorry

end population_present_l202_202285


namespace option_D_correct_option_A_incorrect_option_B_incorrect_option_C_incorrect_l202_202608

-- Define the variables
variables (m : ℤ)

-- State the conditions as hypotheses
theorem option_D_correct (m : ℤ) : 
  (m * (m - 1) = m^2 - m) :=
by {
    -- Proof sketch (not implemented):
    -- Use distributive property to demonstrate that both sides are equal.
    sorry
}

theorem option_A_incorrect (m : ℤ) : 
  ¬ (m^4 + m^3 = m^7) :=
by {
    -- Proof sketch (not implemented):
    -- Demonstrate that exponents can't be added this way when bases are added.
    sorry
}

theorem option_B_incorrect (m : ℤ) : 
  ¬ ((m^4)^3 = m^7) :=
by {
    -- Proof sketch (not implemented):
    -- Show that raising m^4 to the power of 3 results in m^12.
    sorry
}

theorem option_C_incorrect (m : ℤ) : 
  ¬ (2 * m^5 / m^3 = m^2) :=
by {
    -- Proof sketch (not implemented):
    -- Show that dividing results in 2m^2.
    sorry
}

end option_D_correct_option_A_incorrect_option_B_incorrect_option_C_incorrect_l202_202608


namespace fraction_neg_range_l202_202071

theorem fraction_neg_range (x : ℝ) : (x ≠ 0 ∧ x < 1) ↔ (x - 1 < 0 ∧ x^2 > 0) := by
  sorry

end fraction_neg_range_l202_202071


namespace carla_games_won_l202_202676

theorem carla_games_won (F C : ℕ) (h1 : F + C = 30) (h2 : F = C / 2) : C = 20 :=
by
  sorry

end carla_games_won_l202_202676


namespace gcd_polynomial_multiple_of_532_l202_202078

theorem gcd_polynomial_multiple_of_532 (a : ℤ) (h : ∃ k : ℤ, a = 532 * k) :
  Int.gcd (5 * a ^ 3 + 2 * a ^ 2 + 6 * a + 76) a = 76 :=
by
  sorry

end gcd_polynomial_multiple_of_532_l202_202078


namespace remainder_845307_div_6_l202_202414

theorem remainder_845307_div_6 :
  let n := 845307
  ∃ r : ℕ, n % 6 = r ∧ r = 3 :=
by
  let n := 845307
  have h_div_2 : ¬(n % 2 = 0) := by sorry
  have h_div_3 : n % 3 = 0 := by sorry
  exact ⟨3, by sorry, rfl⟩

end remainder_845307_div_6_l202_202414


namespace polygon_sides_l202_202274

theorem polygon_sides (n : ℕ) (sum_of_angles : ℕ) (missing_angle : ℕ) 
  (h1 : sum_of_angles = 3240) 
  (h2 : missing_angle * n / (n - 1) = 2 * sum_of_angles) : 
  n = 20 := 
sorry

end polygon_sides_l202_202274


namespace sixth_equation_l202_202535

theorem sixth_equation :
  (6 + 7 + 8 + 9 + 10 + 11 + 12 + 13 + 14 + 15 + 16 = 121) :=
by
  sorry

end sixth_equation_l202_202535


namespace fifteen_pow_mn_eq_PnQm_l202_202427

-- Definitions
def P (m : ℕ) := 3^m
def Q (n : ℕ) := 5^n

-- Theorem statement
theorem fifteen_pow_mn_eq_PnQm (m n : ℕ) : 15^(m * n) = (P m)^n * (Q n)^m :=
by
  -- Placeholder for the proof, which isn't required
  sorry

end fifteen_pow_mn_eq_PnQm_l202_202427


namespace solve_quartic_equation_l202_202752

theorem solve_quartic_equation :
  (∃ x : ℝ, x > 0 ∧ 
    (1 / 3) * (4 * x ^ 2 - 3) = (x ^ 2 - 60 * x - 12) * (x ^ 2 + 30 * x + 6) ∧ 
    ∃ y1 y2 : ℝ, y1 + y2 = 60 ∧ (x^2 - 60 * x - 12 = 0)) → 
    x = 30 + Real.sqrt 912 :=
sorry

end solve_quartic_equation_l202_202752


namespace sheets_in_backpack_l202_202755

-- Definitions for the conditions
def total_sheets := 91
def desk_sheets := 50

-- Theorem statement with the goal
theorem sheets_in_backpack (total_sheets : ℕ) (desk_sheets : ℕ) (h1 : total_sheets = 91) (h2 : desk_sheets = 50) : 
  ∃ backpack_sheets : ℕ, backpack_sheets = total_sheets - desk_sheets ∧ backpack_sheets = 41 :=
by
  -- The proof is omitted here
  sorry

end sheets_in_backpack_l202_202755


namespace max_value_k_eq_1_range_k_no_zeros_l202_202358

-- Define the function f(x)
noncomputable def f (x : ℝ) (k : ℝ) : ℝ := Real.log (x - 1) - k * (x - 1) + 1

-- Note: 'by' and 'sorry' are placeholders to skip the proof; actual proofs are not required.

-- Proof Problem 1: Prove that when k = 1, the maximum value of f(x) is 0.
theorem max_value_k_eq_1 : ∀ x : ℝ, 1 < x → f x 1 ≤ 0 := 
by
  sorry

-- Proof Problem 2: Prove that k ∈ (1, +∞) is the range such that f(x) has no zeros.
theorem range_k_no_zeros : ∀ k : ℝ, (∀ x : ℝ, 1 < x → f x k ≠ 0) → 1 < k :=
by
  sorry

end max_value_k_eq_1_range_k_no_zeros_l202_202358


namespace smallest_value_of_m_plus_n_l202_202900

theorem smallest_value_of_m_plus_n :
  ∃ m n : ℕ, 1 < m ∧ 
  (∃ l : ℝ, l = (m^2 - 1 : ℝ) / (m * n) ∧ l = 1 / 2021) ∧
  m + n = 85987 := 
sorry

end smallest_value_of_m_plus_n_l202_202900


namespace min_value_a_plus_b_l202_202913

theorem min_value_a_plus_b (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : (2 / a) + (2 / b) = 1) :
  a + b >= 8 :=
sorry

end min_value_a_plus_b_l202_202913


namespace trains_crossing_time_l202_202926

noncomputable def length_first_train : ℝ := 120
noncomputable def length_second_train : ℝ := 160
noncomputable def speed_first_train_kmph : ℝ := 60
noncomputable def speed_second_train_kmph : ℝ := 40
noncomputable def kmph_to_mps (v : ℝ) : ℝ := v * 1000 / 3600

noncomputable def speed_first_train : ℝ := kmph_to_mps speed_first_train_kmph
noncomputable def speed_second_train : ℝ := kmph_to_mps speed_second_train_kmph
noncomputable def relative_speed : ℝ := speed_first_train + speed_second_train
noncomputable def total_distance : ℝ := length_first_train + length_second_train
noncomputable def crossing_time : ℝ := total_distance / relative_speed

theorem trains_crossing_time :
  crossing_time = 10.08 := by
  sorry

end trains_crossing_time_l202_202926


namespace distance_between_parallel_lines_l202_202263

/-- Given two parallel lines y=2x and y=2x+5, the distance between them is √5. -/
theorem distance_between_parallel_lines :
  let A := -2
  let B := 1
  let C1 := 0
  let C2 := -5
  let distance := (|C2 - C1|: ℝ) / Real.sqrt (A ^ 2 + B ^ 2)
  distance = Real.sqrt 5 := by
  -- Assuming calculations as done in the original solution
  sorry

end distance_between_parallel_lines_l202_202263


namespace find_locus_of_P_l202_202389

theorem find_locus_of_P:
  ∃ x y: ℝ, (x - 1)^2 + y^2 = 9 ∧ y ≠ 0 ∧
          ((x + 2)^2 + y^2 + (x - 4)^2 + y^2 = 36) :=
sorry

end find_locus_of_P_l202_202389


namespace complete_work_in_12_days_l202_202719

def Ravi_rate_per_day : ℚ := 1 / 24
def Prakash_rate_per_day : ℚ := 1 / 40
def Suresh_rate_per_day : ℚ := 1 / 60
def combined_rate_per_day : ℚ := Ravi_rate_per_day + Prakash_rate_per_day + Suresh_rate_per_day

theorem complete_work_in_12_days : 
  (1 / combined_rate_per_day) = 12 := 
by
  sorry

end complete_work_in_12_days_l202_202719


namespace opposite_of_2023_l202_202829

theorem opposite_of_2023 : ∃ x : ℤ, 2023 + x = 0 ∧ x = -2023 := 
by
  sorry

end opposite_of_2023_l202_202829


namespace friend_owns_10_bikes_l202_202612

theorem friend_owns_10_bikes (ignatius_bikes : ℕ) (tires_per_bike : ℕ) (unicycle_tires : ℕ) (tricycle_tires : ℕ) (friend_total_tires : ℕ) :
  ignatius_bikes = 4 →
  tires_per_bike = 2 →
  unicycle_tires = 1 →
  tricycle_tires = 3 →
  friend_total_tires = 3 * (ignatius_bikes * tires_per_bike) →
  (friend_total_tires - (unicycle_tires + tricycle_tires)) / tires_per_bike = 10 :=
by
  sorry

end friend_owns_10_bikes_l202_202612


namespace totalLemonProductionIn5Years_l202_202571

-- Definition of a normal lemon tree's production rate
def normalLemonProduction : ℕ := 60

-- Definition of the percentage increase for Jim's lemon trees (50%)
def percentageIncrease : ℕ := 50

-- Calculate Jim's lemon tree production per year
def jimLemonProduction : ℕ := normalLemonProduction * (100 + percentageIncrease) / 100

-- Calculate the total number of trees in Jim's grove
def treesInGrove : ℕ := 50 * 30

-- Calculate the total lemon production by Jim's grove in one year
def annualLemonProduction : ℕ := treesInGrove * jimLemonProduction

-- Calculate the total lemon production by Jim's grove in 5 years
def fiveYearLemonProduction : ℕ := 5 * annualLemonProduction

-- Theorem: Prove that the total lemon production in 5 years is 675000
theorem totalLemonProductionIn5Years : fiveYearLemonProduction = 675000 := by
  -- Proof needs to be filled in
  sorry

end totalLemonProductionIn5Years_l202_202571


namespace f_2_values_l202_202690

theorem f_2_values (f : ℝ → ℝ)
  (h : ∀ x y : ℝ, |f x - f y| = |x - y|)
  (hf1 : f 1 = 3) :
  f 2 = 2 ∨ f 2 = 4 :=
sorry

end f_2_values_l202_202690


namespace comprehensive_survey_l202_202685

def suitable_for_census (s: String) : Prop := 
  s = "Surveying the heights of all classmates in the class"

theorem comprehensive_survey : suitable_for_census "Surveying the heights of all classmates in the class" :=
by
  sorry

end comprehensive_survey_l202_202685


namespace total_cranes_folded_l202_202264

-- Definitions based on conditions
def hyerinCranesPerDay : ℕ := 16
def hyerinDays : ℕ := 7
def taeyeongCranesPerDay : ℕ := 25
def taeyeongDays : ℕ := 6

-- Definition of total number of cranes folded by Hyerin and Taeyeong
def totalCranes : ℕ :=
  (hyerinCranesPerDay * hyerinDays) + (taeyeongCranesPerDay * taeyeongDays)

-- Proof statement
theorem total_cranes_folded : totalCranes = 262 := by 
  sorry

end total_cranes_folded_l202_202264


namespace montague_fraction_l202_202747

noncomputable def fraction_montague (M C : ℝ) : Prop :=
  M + C = 1 ∧
  (0.70 * C) / (0.20 * M + 0.70 * C) = 7 / 11

theorem montague_fraction : ∃ M C : ℝ, fraction_montague M C ∧ M = 2 / 3 :=
by sorry

end montague_fraction_l202_202747


namespace directrix_of_parabola_l202_202472

def parabola_eq (x : ℝ) : ℝ := -4 * x^2 + 4

theorem directrix_of_parabola : 
  ∃ y : ℝ, y = 65 / 16 :=
by
  sorry

end directrix_of_parabola_l202_202472


namespace unique_real_solution_k_l202_202768

theorem unique_real_solution_k (k : ℝ) :
  ∃! x : ℝ, (3 * x + 8) * (x - 6) = -62 + k * x ↔ k = -10 + 12 * Real.sqrt 1.5 ∨ k = -10 - 12 * Real.sqrt 1.5 := by
  sorry

end unique_real_solution_k_l202_202768


namespace fraction_of_science_liking_students_l202_202019

open Real

theorem fraction_of_science_liking_students (total_students math_fraction english_fraction no_fav_students math_students english_students fav_students remaining_students science_students fraction_science) :
  total_students = 30 ∧
  math_fraction = 1/5 ∧
  english_fraction = 1/3 ∧
  no_fav_students = 12 ∧
  math_students = total_students * math_fraction ∧
  english_students = total_students * english_fraction ∧
  fav_students = total_students - no_fav_students ∧
  remaining_students = fav_students - (math_students + english_students) ∧
  science_students = remaining_students ∧
  fraction_science = science_students / remaining_students →
  fraction_science = 1 :=
by
  sorry

end fraction_of_science_liking_students_l202_202019


namespace domain_translation_l202_202210

theorem domain_translation (f : ℝ → ℝ) :
  (∀ x : ℝ, 0 < 3 * x + 2 ∧ 3 * x + 2 < 1 → (∃ y : ℝ, f (3 * x + 2) = y)) →
  (∀ x : ℝ, ∃ y : ℝ, f (2 * x - 1) = y ↔ (3 / 2) < x ∧ x < 3) :=
sorry

end domain_translation_l202_202210


namespace intersection_product_of_circles_l202_202524

theorem intersection_product_of_circles :
  (∀ x y : ℝ, (x^2 + 2 * x + y^2 + 4 * y + 5 = 0) ∧ (x^2 + 6 * x + y^2 + 4 * y + 9 = 0) →
  x * y = 2) :=
sorry

end intersection_product_of_circles_l202_202524


namespace line_passes_fixed_point_l202_202433

theorem line_passes_fixed_point (a b : ℝ) (h : a + 2 * b = 1) : 
  a * (1/2) + 3 * (-1/6) + b = 0 :=
by
  sorry

end line_passes_fixed_point_l202_202433


namespace village_population_l202_202036

theorem village_population (P : ℝ) (h : 0.8 * P = 32000) : P = 40000 := by
  sorry

end village_population_l202_202036


namespace number_divisible_by_7_last_digits_l202_202558

theorem number_divisible_by_7_last_digits :
  ∀ d : ℕ, d ≤ 9 → ∃ n : ℕ, n % 7 = 0 ∧ n % 10 = d :=
by
  sorry

end number_divisible_by_7_last_digits_l202_202558


namespace passenger_drop_ratio_l202_202370

theorem passenger_drop_ratio (initial_passengers passengers_at_first passengers_at_second final_passengers x : ℕ)
  (h0 : initial_passengers = 288)
  (h1 : passengers_at_first = initial_passengers - (initial_passengers / 3) + 280)
  (h2 : passengers_at_second = passengers_at_first - x + 12)
  (h3 : final_passengers = 248)
  (h4 : passengers_at_second = final_passengers) :
  x / passengers_at_first = 1 / 2 :=
by
  sorry

end passenger_drop_ratio_l202_202370


namespace decreasing_interval_eqn_l202_202287

def f (a x : ℝ) : ℝ := x^2 - 2 * a * x + 2

theorem decreasing_interval_eqn {a : ℝ} : (∀ x : ℝ, x < 6 → deriv (f a) x < 0) ↔ a ≥ 6 :=
sorry

end decreasing_interval_eqn_l202_202287


namespace average_employees_per_week_l202_202987

variable (x : ℕ)

theorem average_employees_per_week (h1 : x + 200 > x)
                                   (h2 : x < 200)
                                   (h3 : 2 * 200 = 400) :
  (x + 200 + x + 200 + 200 + 400) / 4 = 250 := by
  sorry

end average_employees_per_week_l202_202987


namespace seq_general_formula_l202_202957

open Nat

def seq (a : ℕ+ → ℝ) : Prop :=
  a 1 = 1 ∧ ∀ n : ℕ+, a (n + 1) = 2 * a n / (2 + a n)

theorem seq_general_formula (a : ℕ+ → ℝ) (h : seq a) :
  ∀ n : ℕ+, a n = 2 / (n + 1) :=
by
  sorry

end seq_general_formula_l202_202957


namespace series_sum_eq_one_sixth_l202_202791

noncomputable def a (n : ℕ) : ℝ := 2^n / (7^(2^n) + 1)

theorem series_sum_eq_one_sixth :
  (∑' (n : ℕ), a n) = 1 / 6 :=
sorry

end series_sum_eq_one_sixth_l202_202791


namespace freezer_temperature_l202_202002

theorem freezer_temperature 
  (refrigeration_temp : ℝ)
  (freezer_temp_diff : ℝ)
  (h1 : refrigeration_temp = 4)
  (h2 : freezer_temp_diff = 22)
  : (refrigeration_temp - freezer_temp_diff) = -18 :=
by 
  sorry

end freezer_temperature_l202_202002


namespace solve_fraction_equation_l202_202217

theorem solve_fraction_equation :
  ∀ x : ℝ, (3 / (2 * x - 2) + 1 / (1 - x) = 3) → x = 7 / 6 :=
by
  sorry

end solve_fraction_equation_l202_202217


namespace cylinder_original_radius_inch_l202_202567

theorem cylinder_original_radius_inch (r : ℝ) :
  (∃ r : ℝ, (π * (r + 4)^2 * 3 = π * r^2 * 15) ∧ (r > 0)) →
  r = 1 + Real.sqrt 5 :=
by 
  sorry

end cylinder_original_radius_inch_l202_202567


namespace average_waiting_time_l202_202157

-- Define the problem conditions
def light_period : ℕ := 3  -- Total cycle time in minutes
def green_time : ℕ := 1    -- Green light duration in minutes
def red_time : ℕ := 2      -- Red light duration in minutes

-- Define the probabilities of each light state
def P_G : ℚ := green_time / light_period
def P_R : ℚ := red_time / light_period

-- Define the expected waiting times given each state
def E_T_G : ℚ := 0
def E_T_R : ℚ := red_time / 2

-- Calculate the expected waiting time using the law of total expectation
def E_T : ℚ := E_T_G * P_G + E_T_R * P_R

-- Convert the expected waiting time to seconds
def E_T_seconds : ℚ := E_T * 60

-- Prove that the expected waiting time in seconds is 40 seconds
theorem average_waiting_time : E_T_seconds = 40 := by
  sorry

end average_waiting_time_l202_202157


namespace product_of_integers_cubes_sum_to_35_l202_202025

-- Define the conditions
def integers_sum_of_cubes (a b : ℤ) : Prop :=
  a^3 + b^3 = 35

-- Define the theorem that the product of integers whose cubes sum to 35 is 6
theorem product_of_integers_cubes_sum_to_35 :
  ∃ a b : ℤ, integers_sum_of_cubes a b ∧ a * b = 6 :=
by
  sorry

end product_of_integers_cubes_sum_to_35_l202_202025


namespace opposite_of_neg_two_is_two_l202_202220

theorem opposite_of_neg_two_is_two : -(-2) = 2 :=
by
  sorry

end opposite_of_neg_two_is_two_l202_202220


namespace geometric_sequence_a7_l202_202886

theorem geometric_sequence_a7 (a : ℕ → ℝ) (q : ℝ)
  (h1 : a 1 + a 2 = 3)
  (h2 : a 2 + a 3 = 6)
  (h_geometric : ∀ n, a (n + 1) = q * a n) :
  a 7 = 64 := by
  sorry

end geometric_sequence_a7_l202_202886


namespace power_of_powers_l202_202466

theorem power_of_powers :
  (3^2)^4 = 6561 := 
by 
  sorry

end power_of_powers_l202_202466


namespace annual_salary_is_20_l202_202559

-- Define the conditions
variable (months_worked : ℝ) (total_received : ℝ) (turban_price : ℝ)
variable (S : ℝ)

-- Actual values from the problem
axiom h1 : months_worked = 9 / 12
axiom h2 : total_received = 55
axiom h3 : turban_price = 50

-- Define the statement to prove
theorem annual_salary_is_20 : S = 20 := by
  -- Conditions derived from the problem
  have cash_received := total_received - turban_price
  have fraction_of_salary := months_worked * S
  -- Given the servant worked 9 months and received Rs. 55 including Rs. 50 turban
  have : cash_received = fraction_of_salary := by sorry
  -- Solving the equation 3/4 S = 5 for S
  have : S = 20 := by sorry
  sorry -- Final proof step

end annual_salary_is_20_l202_202559


namespace integers_satisfy_equation_l202_202340

theorem integers_satisfy_equation (x y z : ℤ) (h1 : x = z) (h2 : y - 1 = x) : x * (x - y) + y * (y - z) + z * (z - x) = 1 :=
by
  sorry

end integers_satisfy_equation_l202_202340


namespace quadratic_inequality_solution_l202_202364

theorem quadratic_inequality_solution (x : ℝ) (h : x^2 - 9*x + 14 < 0) : 2 < x ∧ x < 7 :=
by
  sorry

end quadratic_inequality_solution_l202_202364


namespace total_income_l202_202392

variable (I : ℝ)

/-- A person distributed 20% of his income to his 3 children each. -/
def distributed_children (I : ℝ) : ℝ := 3 * 0.20 * I

/-- He deposited 30% of his income to his wife's account. -/
def deposited_wife (I : ℝ) : ℝ := 0.30 * I

/-- The total percentage of his income that was given away is 90%. -/
def total_given_away (I : ℝ) : ℝ := distributed_children I + deposited_wife I 

/-- The remaining income after giving away 90%. -/
def remaining_income (I : ℝ) : ℝ := I - total_given_away I

/-- He donated 5% of the remaining income to the orphan house. -/
def donated_orphan_house (remaining : ℝ) : ℝ := 0.05 * remaining

/-- Finally, he has $40,000 left, which is 95% of the remaining income. -/
def final_amount (remaining : ℝ) : ℝ := 0.95 * remaining

theorem total_income (I : ℝ) (h : final_amount (remaining_income I) = 40000) :
  I = 421052.63 := 
  sorry

end total_income_l202_202392


namespace julia_total_cost_l202_202708

theorem julia_total_cost
  (snickers_cost : ℝ := 1.5)
  (mm_cost : ℝ := 2 * snickers_cost)
  (pepsi_cost : ℝ := 2 * mm_cost)
  (bread_cost : ℝ := 3 * pepsi_cost)
  (snickers_qty : ℕ := 2)
  (mm_qty : ℕ := 3)
  (pepsi_qty : ℕ := 4)
  (bread_qty : ℕ := 5)
  (money_given : ℝ := 5 * 20) :
  ((snickers_qty * snickers_cost) + (mm_qty * mm_cost) + (pepsi_qty * pepsi_cost) + (bread_qty * bread_cost)) > money_given := 
by
  sorry

end julia_total_cost_l202_202708


namespace find_second_number_l202_202590

theorem find_second_number (x y : ℤ) (h1 : x = -63) (h2 : (2 + y + x) / 3 = 5) : y = 76 :=
sorry

end find_second_number_l202_202590


namespace product_of_roots_cubic_eq_l202_202322

theorem product_of_roots_cubic_eq (α : Type _) [Field α] :
  (∃ (r1 r2 r3 : α), (r1 * r2 * r3 = 6) ∧ (r1 + r2 + r3 = 6) ∧ (r1 * r2 + r1 * r3 + r2 * r3 = 11)) :=
by
  sorry

end product_of_roots_cubic_eq_l202_202322


namespace bird_counts_remaining_l202_202005

theorem bird_counts_remaining
  (peregrine_falcons pigeons crows sparrows : ℕ)
  (chicks_per_pigeon chicks_per_crow chicks_per_sparrow : ℕ)
  (peregrines_eat_pigeons_percent peregrines_eat_crows_percent peregrines_eat_sparrows_percent : ℝ)
  (initial_peregrine_falcons : peregrine_falcons = 12)
  (initial_pigeons : pigeons = 80)
  (initial_crows : crows = 25)
  (initial_sparrows : sparrows = 15)
  (chicks_per_pigeon_cond : chicks_per_pigeon = 8)
  (chicks_per_crow_cond : chicks_per_crow = 5)
  (chicks_per_sparrow_cond : chicks_per_sparrow = 3)
  (peregrines_eat_pigeons_percent_cond : peregrines_eat_pigeons_percent = 0.4)
  (peregrines_eat_crows_percent_cond : peregrines_eat_crows_percent = 0.25)
  (peregrines_eat_sparrows_percent_cond : peregrines_eat_sparrows_percent = 0.1)
  : 
  (peregrine_falcons = 12) ∧
  (pigeons = 48) ∧
  (crows = 19) ∧
  (sparrows = 14) :=
by
  sorry

end bird_counts_remaining_l202_202005


namespace tan_alpha_eq_2_implies_sin_2alpha_inverse_l202_202027

theorem tan_alpha_eq_2_implies_sin_2alpha_inverse (α : ℝ) (h : Real.tan α = 2) :
  1 / Real.sin (2 * α) = 5 / 4 :=
sorry

end tan_alpha_eq_2_implies_sin_2alpha_inverse_l202_202027


namespace impossible_condition_l202_202057

noncomputable def f (x : ℝ) : ℝ := |2^x - 1|

theorem impossible_condition (a b c : ℝ) (h : f a > f b ∧ f b > f c) : ¬ (b < a ∧ a < c) :=
by
  sorry

end impossible_condition_l202_202057


namespace perpendicular_condition_l202_202816

-- Condition definition
def is_perpendicular (a : ℝ) : Prop :=
  let line1_slope := -1
  let line2_slope := - (a / 2)
  (line1_slope * line2_slope = -1)

-- Statement of the theorem
theorem perpendicular_condition (a : ℝ) :
  is_perpendicular a ↔ a = -2 :=
sorry

end perpendicular_condition_l202_202816


namespace total_score_l202_202821

theorem total_score (score_cap : ℝ) (score_val : ℝ) (score_imp : ℝ) (wt_cap : ℝ) (wt_val : ℝ) (wt_imp : ℝ) (total_weight : ℝ) :
  score_cap = 8 → score_val = 9 → score_imp = 7 → wt_cap = 5 → wt_val = 3 → wt_imp = 2 → total_weight = 10 →
  ((score_cap * (wt_cap / total_weight)) + (score_val * (wt_val / total_weight)) + (score_imp * (wt_imp / total_weight))) = 8.1 := 
by
  intros
  sorry

end total_score_l202_202821


namespace percentage_of_paycheck_went_to_taxes_l202_202892

-- Definitions
def original_paycheck : ℝ := 125
def savings : ℝ := 20
def spend_percentage : ℝ := 0.80
def save_percentage : ℝ := 0.20

-- Statement that needs to be proved
theorem percentage_of_paycheck_went_to_taxes (T : ℝ) :
  (0.20 * (1 - T / 100) * original_paycheck = savings) → T = 20 := 
by
  sorry

end percentage_of_paycheck_went_to_taxes_l202_202892


namespace power_mean_inequality_l202_202259

theorem power_mean_inequality (a b : ℝ) (n : ℕ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hn : 0 < n) :
  (a^n + b^n) / 2 ≥ ((a + b) / 2)^n := 
by
  sorry

end power_mean_inequality_l202_202259


namespace geometric_sequence_sum_l202_202873

variable {a : ℕ → ℝ} -- Sequence terms
variable {S : ℕ → ℝ} -- Sum of the first n terms

-- Conditions
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) := ∀ n, a (n+1) = a n * q
def sum_of_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) := ∀ n, S n = a 0 * (1 - (a n)) / (1 - a 1)
def is_arithmetic_sequence (x y z : ℝ) := 2 * y = x + z
def term_1_equals_1 (a : ℕ → ℝ) := a 0 = 1

-- Question: Prove that given the conditions, S_5 = 31
theorem geometric_sequence_sum (q : ℝ) (h_geom : is_geometric_sequence a q) 
  (h_sum : sum_of_first_n_terms a S) (h_arith : is_arithmetic_sequence (4 * a 0) (2 * a 1) (a 2)) 
  (h_a1 : term_1_equals_1 a) : S 5 = 31 :=
sorry

end geometric_sequence_sum_l202_202873


namespace remaining_thumbtacks_in_each_can_l202_202120

-- Definitions based on the conditions:
def total_thumbtacks : ℕ := 450
def num_cans : ℕ := 3
def thumbtacks_per_board_tested : ℕ := 1
def total_boards_tested : ℕ := 120

-- Lean 4 Statement

theorem remaining_thumbtacks_in_each_can :
  ∀ (initial_thumbtacks_per_can remaining_thumbtacks_per_can : ℕ),
  initial_thumbtacks_per_can = (total_thumbtacks / num_cans) →
  remaining_thumbtacks_per_can = (initial_thumbtacks_per_can - (thumbtacks_per_board_tested * total_boards_tested)) →
  remaining_thumbtacks_per_can = 30 :=
by
  sorry

end remaining_thumbtacks_in_each_can_l202_202120


namespace Danny_more_wrappers_than_bottle_caps_l202_202306

theorem Danny_more_wrappers_than_bottle_caps
  (initial_wrappers : ℕ)
  (initial_bottle_caps : ℕ)
  (found_wrappers : ℕ)
  (found_bottle_caps : ℕ) :
  initial_wrappers = 67 →
  initial_bottle_caps = 35 →
  found_wrappers = 18 →
  found_bottle_caps = 15 →
  (initial_wrappers + found_wrappers) - (initial_bottle_caps + found_bottle_caps) = 35 :=
by
  intros h1 h2 h3 h4
  sorry

end Danny_more_wrappers_than_bottle_caps_l202_202306


namespace mushroom_children_count_l202_202051

variables {n : ℕ} {A V S R : ℕ}

-- Conditions:
def condition1 (n : ℕ) (A : ℕ) (V : ℕ) : Prop :=
  ∀ (k : ℕ), k < n → V + A / 2 = k

def condition2 (S : ℕ) (A : ℕ) (R : ℕ) (V : ℕ) : Prop :=
  S + A = R + V + A

-- Proof statement
theorem mushroom_children_count (n : ℕ) (A : ℕ) (V : ℕ) (S : ℕ) (R : ℕ) :
  condition1 n A V → condition2 S A R V → n = 6 :=
by
  intros hcondition1 hcondition2
  sorry

end mushroom_children_count_l202_202051


namespace min_sum_of_a_and_b_l202_202514

theorem min_sum_of_a_and_b (a b : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a > 4 * b) : a + b ≥ 6 :=
by
  sorry

end min_sum_of_a_and_b_l202_202514


namespace complex_number_sum_equals_one_l202_202853

variable {a b c d : ℝ}
variable {ω : ℂ}

theorem complex_number_sum_equals_one
  (ha : a ≠ -1) 
  (hb : b ≠ -1) 
  (hc : c ≠ -1) 
  (hd : d ≠ -1) 
  (hω : ω^4 = 1) 
  (hω_ne : ω ≠ 1)
  (h_eq : (1 / (a + ω)) + (1 / (b + ω)) + (1 / (c + ω)) + (1 / (d + ω)) = 4 / ω)
  : (1 / (a + 1)) + (1 / (b + 1)) + (1 / (c + 1)) + (1 / (d + 1)) = 1 :=
by sorry

end complex_number_sum_equals_one_l202_202853


namespace range_of_ab_l202_202160

noncomputable def circle_equation (x y : ℝ) : Prop := (x^2 + y^2 + 2*x - 4*y + 1 = 0)

noncomputable def line_equation (a b x y : ℝ) : Prop := (2*a*x - b*y - 2 = 0)

def symmetric_with_respect_to (center_x center_y a b : ℝ) : Prop :=
  line_equation a b center_x center_y  -- check if the line passes through the center

theorem range_of_ab (a b : ℝ) (h_symm : symmetric_with_respect_to (-1) 2 a b) : 
  ∃ ab_max : ℝ, ab_max = 1/4 ∧ ∀ ab : ℝ, ab = (a * b) → ab ≤ ab_max :=
sorry

end range_of_ab_l202_202160


namespace find_a_c_pair_l202_202964

-- Given conditions in the problem
variable (a c : ℝ)

-- First condition: The quadratic equation has exactly one solution
def quadratic_eq_has_one_solution : Prop :=
  let discriminant := (30:ℝ)^2 - 4 * a * c
  discriminant = 0

-- Second condition: Sum of a and c
def sum_eq_41 : Prop := a + c = 41

-- Third condition: a is less than c
def a_lt_c : Prop := a < c

-- State the proof problem
theorem find_a_c_pair (a c : ℝ) (h1 : quadratic_eq_has_one_solution a c) (h2 : sum_eq_41 a c) (h3 : a_lt_c a c) : (a, c) = (6.525, 34.475) :=
sorry

end find_a_c_pair_l202_202964


namespace simplify_sqrt_expression_l202_202720

theorem simplify_sqrt_expression : 2 * Real.sqrt 3 - Real.sqrt 12 + Real.sqrt 75 = 5 * Real.sqrt 3 :=
by
  sorry

end simplify_sqrt_expression_l202_202720


namespace slope_of_tangent_line_l202_202069

theorem slope_of_tangent_line (f : ℝ → ℝ) (f_deriv : ∀ x, deriv f x = f x) (h_tangent : ∃ x₀, f x₀ = x₀ * deriv f x₀ ∧ (0 < f x₀)) :
  ∃ k, k = Real.exp 1 :=
by
  sorry

end slope_of_tangent_line_l202_202069


namespace find_a_l202_202696

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  Real.log x + (a - 1) * x

theorem find_a {a : ℝ} : 
  (∀ x : ℝ, 0 < x → f x a ≤ x^2 * Real.exp x - Real.log x - 4 * x - 1) → 
  a ≤ -2 :=
sorry

end find_a_l202_202696


namespace range_of_a_l202_202387

theorem range_of_a (x : ℝ) (h : 1 < x) : ∀ a, (∀ x, 1 < x → x + 1 / (x - 1) ≥ a) → a ≤ 3 :=
by
sorry

end range_of_a_l202_202387


namespace maximum_xy_value_l202_202963

theorem maximum_xy_value :
  ∃ (x y : ℕ), 7 * x + 4 * y = 140 ∧ x * y = 168 :=
by
  sorry

end maximum_xy_value_l202_202963


namespace find_alpha_l202_202949

theorem find_alpha (α : ℝ) (h1 : Real.tan α = -1) (h2 : 0 < α ∧ α ≤ Real.pi) : α = 3 * Real.pi / 4 :=
sorry

end find_alpha_l202_202949


namespace find_x_l202_202227

theorem find_x (x : ℕ) (h : 2^x - 2^(x-2) = 3 * 2^(12)) : x = 14 :=
sorry

end find_x_l202_202227


namespace solve_eq_l202_202030

theorem solve_eq (x : ℝ) : x^6 - 19*x^3 = 216 ↔ (x = 3 ∨ x = -2) :=
by
  sorry

end solve_eq_l202_202030


namespace ada_original_seat_l202_202235

theorem ada_original_seat (seats: Fin 6 → Option String)
  (Bea_init Ceci_init Dee_init Edie_init Fran_init: Fin 6) 
  (Bea_fin Ceci_fin Fran_fin: Fin 6) 
  (Ada_fin: Fin 6)
  (Bea_moves_right: Bea_fin = Bea_init + 3)
  (Ceci_stays: Ceci_fin = Ceci_init)
  (Dee_switches_with_Edie: ∃ Dee_fin Edie_fin: Fin 6, Dee_fin = Edie_init ∧ Edie_fin = Dee_init)
  (Fran_moves_left: Fran_fin = Fran_init - 1)
  (Ada_end_seat: Ada_fin = 0 ∨ Ada_fin = 5):
  ∃ Ada_init: Fin 6, Ada_init = 2 + Ada_fin + 1 → Ada_init = 3 := 
by 
  sorry

end ada_original_seat_l202_202235


namespace sally_picked_peaches_l202_202942

variable (p_initial p_current p_picked : ℕ)

theorem sally_picked_peaches (h1 : p_initial = 13) (h2 : p_current = 55) :
  p_picked = p_current - p_initial → p_picked = 42 :=
by
  intros
  sorry

end sally_picked_peaches_l202_202942


namespace increasing_sequence_k_range_l202_202938

theorem increasing_sequence_k_range (k : ℝ) (a : ℕ → ℝ) (h : ∀ n : ℕ, a n = n^2 + k * n) :
  (∀ n : ℕ, a (n + 1) > a n) → (k ≥ -3) :=
  sorry

end increasing_sequence_k_range_l202_202938


namespace jack_sugar_final_l202_202550

-- Conditions
def initial_sugar := 65
def sugar_used := 18
def sugar_bought := 50

-- Question and proof goal
theorem jack_sugar_final : initial_sugar - sugar_used + sugar_bought = 97 := by
  sorry

end jack_sugar_final_l202_202550


namespace average_speed_with_stoppages_l202_202312

/--The average speed of the bus including stoppages is 20 km/hr, 
  given that the bus stops for 40 minutes per hour and 
  has an average speed of 60 km/hr excluding stoppages.--/
theorem average_speed_with_stoppages 
  (avg_speed_without_stoppages : ℝ)
  (stoppage_time_per_hour : ℕ) 
  (running_time_per_hour : ℕ) 
  (avg_speed_with_stoppages : ℝ) 
  (h1 : avg_speed_without_stoppages = 60) 
  (h2 : stoppage_time_per_hour = 40) 
  (h3 : running_time_per_hour = 20) 
  (h4 : running_time_per_hour + stoppage_time_per_hour = 60):
  avg_speed_with_stoppages = 20 := 
sorry

end average_speed_with_stoppages_l202_202312


namespace fewest_tiles_to_cover_region_l202_202049

namespace TileCoverage

def tile_width : ℕ := 2
def tile_length : ℕ := 6
def region_width_feet : ℕ := 3
def region_length_feet : ℕ := 4

def region_width_inches : ℕ := region_width_feet * 12
def region_length_inches : ℕ := region_length_feet * 12

def region_area : ℕ := region_width_inches * region_length_inches
def tile_area : ℕ := tile_width * tile_length

def fewest_tiles_needed : ℕ := region_area / tile_area

theorem fewest_tiles_to_cover_region :
  fewest_tiles_needed = 144 :=
sorry

end TileCoverage

end fewest_tiles_to_cover_region_l202_202049


namespace proposition_verification_l202_202205

-- Definitions and Propositions
def prop1 : Prop := (∀ x, x = 1 → x^2 - 3 * x + 2 = 0) ∧ (∃ x, x ≠ 1 ∧ x^2 - 3 * x + 2 = 0)
def prop2 : Prop := (∀ x, ¬ (x^2 - 3 * x + 2 = 0 → x = 1) → (x ≠ 1 → x^2 - 3 * x + 2 ≠ 0))
def prop3 : Prop := ¬ (∃ x > 0, x^2 + x + 1 < 0) → (∀ x ≤ 0, x^2 + x + 1 ≥ 0)
def prop4 : Prop := ¬ (∃ p q : Prop, (p ∨ q) → ¬p ∧ ¬q)

-- Final theorem statement
theorem proposition_verification : prop1 ∧ prop2 ∧ ¬prop3 ∧ ¬prop4 := by 
  sorry

end proposition_verification_l202_202205


namespace tissue_magnification_l202_202916

theorem tissue_magnification (d_image d_actual : ℝ) (h_image : d_image = 0.3) (h_actual : d_actual = 0.0003) :
  (d_image / d_actual) = 1000 :=
by
  sorry

end tissue_magnification_l202_202916


namespace min_removed_numbers_l202_202780

theorem min_removed_numbers : 
  ∃ S : Finset ℤ, 
    (∀ x ∈ S, 1 ≤ x ∧ x ≤ 1982) ∧ 
    (∀ a b c : ℤ, a ∈ S → b ∈ S → c ∈ S → c ≠ a * b) ∧
    ∀ T : Finset ℤ, 
      ((∀ y ∈ T, 1 ≤ y ∧ y ≤ 1982) ∧ 
       (∀ p q r : ℤ, p ∈ T → q ∈ T → r ∈ T → r ≠ p * q) → 
       T.card ≥ 1982 - 43) :=
sorry

end min_removed_numbers_l202_202780


namespace eggs_total_l202_202248

-- Definitions of the conditions in Lean
def num_people : ℕ := 3
def omelets_per_person : ℕ := 3
def eggs_per_omelet : ℕ := 4

-- The claim we need to prove
theorem eggs_total : (num_people * omelets_per_person) * eggs_per_omelet = 36 :=
by
  sorry

end eggs_total_l202_202248


namespace integer_solutions_count_l202_202284

theorem integer_solutions_count (x : ℤ) :
  (75 ^ 60 * x ^ 60 > x ^ 120 ∧ x ^ 120 > 3 ^ 240) → ∃ n : ℕ, n = 65 :=
by
  sorry

end integer_solutions_count_l202_202284


namespace sum_of_squares_l202_202464

theorem sum_of_squares (x y : ℝ) (h₁ : x + y = 40) (h₂ : x * y = 120) : x^2 + y^2 = 1360 :=
by
  sorry

end sum_of_squares_l202_202464


namespace num_outfits_l202_202654

def num_shirts := 6
def num_ties := 4
def num_pants := 3
def outfits : ℕ := num_shirts * num_pants * (num_ties + 1)

theorem num_outfits: outfits = 90 :=
by 
  -- sorry will be removed when proof is provided
  sorry

end num_outfits_l202_202654


namespace positive_even_integers_less_than_1000_not_divisible_by_3_or_11_l202_202857

theorem positive_even_integers_less_than_1000_not_divisible_by_3_or_11 :
  ∃ n : ℕ, n = 108 ∧
    (∀ m : ℕ, 0 < m → 2 ∣ m → m < 1000 → (¬ (3 ∣ m) ∧ ¬ (11 ∣ m) ↔ m ≤ n)) :=
sorry

end positive_even_integers_less_than_1000_not_divisible_by_3_or_11_l202_202857


namespace range_of_m_l202_202954

theorem range_of_m {m : ℝ} : (-1 ≤ 1 - m ∧ 1 - m ≤ 1) ↔ (0 ≤ m ∧ m ≤ 2) := 
sorry

end range_of_m_l202_202954


namespace min_floodgates_to_reduce_level_l202_202495

-- Definitions for the conditions given in the problem
def num_floodgates : ℕ := 10
def a (v : ℝ) := 30 * v
def w (v : ℝ) := 2 * v

def time_one_gate : ℝ := 30
def time_two_gates : ℝ := 10
def time_target : ℝ := 3

-- Prove that the minimum number of floodgates \(n\) that must be opened to achieve the goal
theorem min_floodgates_to_reduce_level (v : ℝ) (n : ℕ) :
  (a v + time_target * v) ≤ (n * time_target * w v) → n ≥ 6 :=
by
  sorry

end min_floodgates_to_reduce_level_l202_202495


namespace calculate_expression_l202_202295

theorem calculate_expression : 103^3 - 3 * 103^2 + 3 * 103 - 1 = 1061208 := by
  sorry

end calculate_expression_l202_202295


namespace fraction_equiv_l202_202108

theorem fraction_equiv (m n : ℚ) (h : m / n = 3 / 4) : (m + n) / n = 7 / 4 :=
sorry

end fraction_equiv_l202_202108


namespace solve_for_x_l202_202360

theorem solve_for_x (x : ℤ) (h : (-1) * 2 * x * 4 = 24) : x = -3 := by
  sorry

end solve_for_x_l202_202360


namespace geometric_sum_of_first_four_terms_eq_120_l202_202697

theorem geometric_sum_of_first_four_terms_eq_120
  (a : ℕ → ℝ)
  (r : ℝ)
  (h_geom : ∀ n, a (n + 1) = a n * r)
  (ha2 : a 2 = 9)
  (ha5 : a 5 = 243) :
  a 1 * (1 - r^4) / (1 - r) = 120 := 
sorry

end geometric_sum_of_first_four_terms_eq_120_l202_202697


namespace garden_area_l202_202353

-- Definitions for the conditions
def perimeter : ℕ := 36
def width : ℕ := 10

-- Define the length using the perimeter and width
def length : ℕ := (perimeter - 2 * width) / 2

-- Define the area using the length and width
def area : ℕ := length * width

-- The theorem to prove the area is 80 square feet given the conditions
theorem garden_area : area = 80 :=
by 
  -- Here we use sorry to skip the proof
  sorry

end garden_area_l202_202353


namespace packets_of_chips_l202_202448

variable (P R M : ℕ)

theorem packets_of_chips (h1: P > 0) (h2: R > 0) (h3: M > 0) :
  ((10 * M * P) / R) = (10 * M * P) / R :=
sorry

end packets_of_chips_l202_202448


namespace arctan_sum_pi_div_two_l202_202912

theorem arctan_sum_pi_div_two :
  let α := Real.arctan (3 / 4)
  let β := Real.arctan (4 / 3)
  α + β = Real.pi / 2 := by
  sorry

end arctan_sum_pi_div_two_l202_202912


namespace no_triangular_sides_of_specific_a_b_l202_202723

theorem no_triangular_sides_of_specific_a_b (a b c : ℕ) (h1 : a = 10^100 + 1002) (h2 : b = 1001) (h3 : ∃ n : ℕ, c = n^2) : ¬ (a + b > c ∧ a + c > b ∧ b + c > a) :=
by sorry

end no_triangular_sides_of_specific_a_b_l202_202723


namespace minimum_a_div_x_l202_202109

theorem minimum_a_div_x (a x y : ℕ) (h1 : 100 < a) (h2 : 100 < x) (h3 : 100 < y) (h4 : y^2 - 1 = a^2 * (x^2 - 1)) :
  2 ≤ a / x :=
by sorry

end minimum_a_div_x_l202_202109


namespace negation_proposition_l202_202368

theorem negation_proposition (p : Prop) (h : ∀ x : ℝ, 2 * x^2 + 1 > 0) : ¬p ↔ ∃ x : ℝ, 2 * x^2 + 1 ≤ 0 :=
sorry

end negation_proposition_l202_202368


namespace graph_does_not_pass_through_quadrant_II_l202_202941

noncomputable def linear_function (x : ℝ) : ℝ := 3 * x - 4

def passes_through_quadrant_I (x : ℝ) : Prop := x > 0 ∧ linear_function x > 0
def passes_through_quadrant_II (x : ℝ) : Prop := x < 0 ∧ linear_function x > 0
def passes_through_quadrant_III (x : ℝ) : Prop := x < 0 ∧ linear_function x < 0
def passes_through_quadrant_IV (x : ℝ) : Prop := x > 0 ∧ linear_function x < 0

theorem graph_does_not_pass_through_quadrant_II :
  ¬(∃ x : ℝ, passes_through_quadrant_II x) :=
sorry

end graph_does_not_pass_through_quadrant_II_l202_202941


namespace total_distance_traveled_l202_202206

theorem total_distance_traveled:
  let speed1 := 30
  let time1 := 4
  let speed2 := 35
  let time2 := 5
  let speed3 := 25
  let time3 := 6
  let total_time := 20
  let time1_3 := time1 + time2 + time3
  let time4 := total_time - time1_3
  let speed4 := 40

  let distance1 := speed1 * time1
  let distance2 := speed2 * time2
  let distance3 := speed3 * time3
  let distance4 := speed4 * time4

  let total_distance := distance1 + distance2 + distance3 + distance4

  total_distance = 645 :=
  sorry

end total_distance_traveled_l202_202206


namespace probability_not_sit_next_to_each_other_l202_202734

noncomputable def total_ways_to_choose_two_chairs_excluding_broken : ℕ := 28

noncomputable def unfavorable_outcomes : ℕ := 6

theorem probability_not_sit_next_to_each_other :
  (1 - (unfavorable_outcomes / total_ways_to_choose_two_chairs_excluding_broken) = (11 / 14)) :=
by sorry

end probability_not_sit_next_to_each_other_l202_202734


namespace cost_effectiveness_l202_202138

-- Define the variables and conditions
def num_employees : ℕ := 30
def ticket_price : ℝ := 80
def group_discount_rate : ℝ := 0.8
def women_discount_rate : ℝ := 0.5

-- Define the costs for each scenario
def cost_with_group_discount : ℝ := num_employees * ticket_price * group_discount_rate

def cost_with_women_discount (x : ℕ) : ℝ :=
  ticket_price * women_discount_rate * x + ticket_price * (num_employees - x)

-- Formalize the equivalence of cost and comparison logic
theorem cost_effectiveness (x : ℕ) (h : 0 ≤ x ∧ x ≤ num_employees) :
  if x < 12 then cost_with_women_discount x > cost_with_group_discount
  else if x = 12 then cost_with_women_discount x = cost_with_group_discount
  else cost_with_women_discount x < cost_with_group_discount :=
by sorry

end cost_effectiveness_l202_202138


namespace subtraction_result_l202_202033

theorem subtraction_result : 3.05 - 5.678 = -2.628 := 
by
  sorry

end subtraction_result_l202_202033


namespace perfect_square_problem_l202_202119

-- Define the given conditions and question
theorem perfect_square_problem 
  (a b c : ℕ) 
  (h_pos: a > 0 ∧ b > 0 ∧ c > 0)
  (h_cond: 0 < a^2 + b^2 - a * b * c ∧ a^2 + b^2 - a * b * c ≤ c + 1) : 
  ∃ k : ℕ, k^2 = a^2 + b^2 - a * b * c := 
sorry

end perfect_square_problem_l202_202119


namespace correct_transformation_l202_202721

theorem correct_transformation (a b : ℝ) (hb : b ≠ 0) : (a / b = 2 * a / 2 * b) :=
by
  sorry

end correct_transformation_l202_202721


namespace original_employee_salary_l202_202055

-- Given conditions
def emily_original_salary : ℝ := 1000000
def emily_new_salary : ℝ := 850000
def number_of_employees : ℕ := 10
def employee_new_salary : ℝ := 35000

-- Prove the original salary of each employee
theorem original_employee_salary :
  (emily_original_salary - emily_new_salary) / number_of_employees = employee_new_salary - 20000 := 
by
  sorry

end original_employee_salary_l202_202055


namespace force_of_water_pressure_on_plate_l202_202799

noncomputable def force_on_plate_under_water (γ : ℝ) (g : ℝ) (a b : ℝ) : ℝ :=
  γ * g * (b^2 - a^2) / 2

theorem force_of_water_pressure_on_plate :
  let γ : ℝ := 1000 -- kg/m^3
  let g : ℝ := 9.81  -- m/s^2
  let a : ℝ := 0.5   -- top depth
  let b : ℝ := 2.5   -- bottom depth
  force_on_plate_under_water γ g a b = 29430 := sorry

end force_of_water_pressure_on_plate_l202_202799


namespace chime_date_is_march_22_2003_l202_202410

-- Definitions
def clock_chime (n : ℕ) : ℕ := n % 12

def half_hour_chimes (half_hours : ℕ) : ℕ := half_hours
def hourly_chimes (hours : List ℕ) : ℕ := hours.map clock_chime |>.sum

-- Problem conditions and result
def initial_chimes_and_half_hours : ℕ := half_hour_chimes 9
def initial_hourly_chimes : ℕ := hourly_chimes [4, 5, 6, 7, 8, 9, 10, 11, 0]
def chimes_on_february_28_2003 : ℕ := initial_chimes_and_half_hours + initial_hourly_chimes

def half_hour_chimes_per_day : ℕ := half_hour_chimes 24
def hourly_chimes_per_day : ℕ := hourly_chimes (List.range 12 ++ List.range 12)
def total_chimes_per_day : ℕ := half_hour_chimes_per_day + hourly_chimes_per_day

def remaining_chimes_needed : ℕ := 2003 - chimes_on_february_28_2003
def full_days_needed : ℕ := remaining_chimes_needed / total_chimes_per_day
def additional_chimes_needed : ℕ := remaining_chimes_needed % total_chimes_per_day

-- Lean theorem statement
theorem chime_date_is_march_22_2003 :
    (full_days_needed = 21) → (additional_chimes_needed < total_chimes_per_day) → 
    true :=
by
  sorry

end chime_date_is_march_22_2003_l202_202410


namespace parabola_focus_l202_202452

open Real

theorem parabola_focus (a : ℝ) (h k : ℝ) (x y : ℝ) (f : ℝ) :
  (a = -1/4) → (h = 0) → (k = 0) → 
  (f = (1 / (4 * a))) →
  (y = a * (x - h) ^ 2 + k) → 
  (y = -1 / 4 * x ^ 2) → f = -1 := by
  intros h_a h_h h_k h_f parabola_eq _
  rw [h_a, h_h, h_k] at *
  sorry

end parabola_focus_l202_202452


namespace zero_function_unique_l202_202613

theorem zero_function_unique 
  (f : ℝ → ℝ) 
  (h : ∀ (x y : ℝ), f (x ^ (42 ^ 42) + y) = f (x ^ 3 + 2 * y) + f (x ^ 12)) : 
  ∀ x : ℝ, f x = 0 :=
by
  sorry

end zero_function_unique_l202_202613


namespace joska_has_higher_probability_l202_202029

open Nat

def num_4_digit_with_all_diff_digits := 10 * 9 * 8 * 7
def total_4_digit_combinations := 10^4
def num_4_digit_with_repeated_digits := total_4_digit_combinations - num_4_digit_with_all_diff_digits

-- Calculate probabilities
noncomputable def prob_joska := (num_4_digit_with_all_diff_digits : ℝ) / (total_4_digit_combinations : ℝ)
noncomputable def prob_gabor := (num_4_digit_with_repeated_digits : ℝ) / (total_4_digit_combinations : ℝ)

theorem joska_has_higher_probability :
  prob_joska > prob_gabor :=
  by
    sorry

end joska_has_higher_probability_l202_202029


namespace quadratic_distinct_real_roots_l202_202777

theorem quadratic_distinct_real_roots (c : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ x^2 - 2*x + c = 0 ∧ y^2 - 2*y + c = 0) ↔ c < 1 :=
by
  sorry

end quadratic_distinct_real_roots_l202_202777


namespace jason_investing_months_l202_202269

noncomputable def initial_investment (total_amount earned_amount_per_month : ℕ) := total_amount / 3
noncomputable def months_investing (initial_investment earned_amount_per_month : ℕ) := (2 * initial_investment) / earned_amount_per_month

theorem jason_investing_months (total_amount earned_amount_per_month : ℕ) 
  (h1 : total_amount = 90) 
  (h2 : earned_amount_per_month = 12) 
  : months_investing (initial_investment total_amount earned_amount_per_month) earned_amount_per_month = 5 := 
by
  sorry

end jason_investing_months_l202_202269


namespace exponent_problem_l202_202979

variable (x m n : ℝ)
variable (h1 : x^m = 3)
variable (h2 : x^n = 5)

theorem exponent_problem : x^(2 * m - 3 * n) = 9 / 125 :=
by 
  sorry

end exponent_problem_l202_202979


namespace variance_is_0_02_l202_202151

def data_points : List ℝ := [9.8, 9.9, 10.1, 10, 10.2]

noncomputable def mean (l : List ℝ) : ℝ :=
  l.sum / l.length

noncomputable def variance (l : List ℝ) : ℝ :=
  let m := mean l
  (l.map (λ x => (x - m) ^ 2)).sum / l.length

theorem variance_is_0_02 : variance data_points = 0.02 :=
by
  sorry

end variance_is_0_02_l202_202151


namespace solve_for_x_l202_202237

theorem solve_for_x (x : ℝ) (h : (x / 5) / 3 = 15 / (x / 3)) : x = 15 * Real.sqrt 3 ∨ x = -15 * Real.sqrt 3 :=
by
  sorry

end solve_for_x_l202_202237


namespace smallest_root_abs_eq_six_l202_202888

theorem smallest_root_abs_eq_six : 
  (∃ x : ℝ, (abs (x - 1)) / (x^2) = 6 ∧ ∀ y : ℝ, (abs (y - 1)) / (y^2) = 6 → y ≥ x) → x = -1 / 2 := by
  sorry

end smallest_root_abs_eq_six_l202_202888


namespace largest_cube_volume_l202_202442

theorem largest_cube_volume (width length height : ℕ) (h₁ : width = 15) (h₂ : length = 12) (h₃ : height = 8) :
  ∃ V, V = 512 := by
  use 8^3
  sorry

end largest_cube_volume_l202_202442


namespace seq_positive_integers_no_m_exists_l202_202213

-- Definition of the sequence
def seq (n : ℕ) : ℕ :=
  Nat.recOn n
    1
    (λ n a_n => 3 * a_n + 2 * (2 * a_n * a_n - 1).sqrt)

-- Axiomatize the properties involved in the recurrence relation
axiom rec_sqrt_property (n : ℕ) : ∃ k : ℕ, (2 * seq n * seq n - 1) = k * k

-- Proof statement for the sequence of positive integers
theorem seq_positive_integers (n : ℕ) : seq n > 0 := sorry

-- Proof statement for non-existence of m such that 2015 divides seq(m)
theorem no_m_exists (m : ℕ) : ¬ (2015 ∣ seq m) := sorry

end seq_positive_integers_no_m_exists_l202_202213


namespace students_in_class_l202_202583

theorem students_in_class (b g : ℕ) 
  (h1 : b + g = 20)
  (h2 : (b : ℚ) / 20 = (3 : ℚ) / 4 * (g : ℚ) / 20) : 
  b = 12 ∧ g = 8 :=
by
  sorry

end students_in_class_l202_202583


namespace no_family_of_lines_exists_l202_202607

theorem no_family_of_lines_exists (k : ℕ → ℝ) (a : ℕ → ℝ) (b : ℕ → ℝ) :
  (∀ n, (1 : ℝ) = k n * (1 : ℝ) + (1 - k n)) ∧
  (∀ n, k (n + 1) = a n - b n ∧ a n = 1 - 1 / k n ∧ b n = 1 - k n) ∧
  (∀ n, k n * k (n + 1) ≥ 0) →
  False :=
by
  sorry

end no_family_of_lines_exists_l202_202607


namespace roots_sum_equality_l202_202878

theorem roots_sum_equality {a b c : ℝ} {x₁ x₂ x₃ x₄ y₁ y₂ y₃ y₄ : ℝ} :
  (∀ x, x ^ 4 + a * x ^ 3 + b * x ^ 2 + c * x - 1 = 0 → x = x₁ ∨ x = x₂ ∨ x = x₃ ∨ x = x₄) →
  (∀ x, x ^ 4 + a * x ^ 3 + b * x ^ 2 + c * x - 2 = 0 → x = y₁ ∨ x = y₂ ∨ x = y₃ ∨ x = y₄) →
  x₁ + x₂ = x₃ + x_₄ →
  y₁ + y₂ = y₃ + y₄ :=
sorry

end roots_sum_equality_l202_202878


namespace sandwich_cost_l202_202713

theorem sandwich_cost (soda_cost sandwich_cost total_cost : ℝ) (h1 : soda_cost = 0.87) (h2 : total_cost = 10.46) (h3 : 4 * soda_cost + 2 * sandwich_cost = total_cost) :
  sandwich_cost = 3.49 :=
by
  sorry

end sandwich_cost_l202_202713


namespace sin_cos_expr1_sin_cos_expr2_l202_202808

variable {x : ℝ}
variable (hx : Real.tan x = 2)

theorem sin_cos_expr1 : (2 / 3) * (Real.sin x)^2 + (1 / 4) * (Real.cos x)^2 = 7 / 12 := by
  sorry

theorem sin_cos_expr2 : 2 * (Real.sin x)^2 - (Real.sin x) * (Real.cos x) + (Real.cos x)^2 = 7 / 5 := by
  sorry

end sin_cos_expr1_sin_cos_expr2_l202_202808


namespace distribution_difference_l202_202565

theorem distribution_difference 
  (total_amnt : ℕ)
  (p_amnt : ℕ) 
  (q_amnt : ℕ) 
  (r_amnt : ℕ)
  (s_amnt : ℕ)
  (h_total : total_amnt = 1000)
  (h_p : p_amnt = 2 * q_amnt)
  (h_s : s_amnt = 4 * r_amnt)
  (h_qr : q_amnt = r_amnt) :
  s_amnt - p_amnt = 250 := 
sorry

end distribution_difference_l202_202565


namespace sarah_house_units_digit_l202_202698

-- Sarah's house number has two digits
def is_two_digit (n : ℕ) : Prop := n >= 10 ∧ n < 100

-- The four statements about Sarah's house number
def is_multiple_of_5 (n : ℕ) : Prop := n % 5 = 0
def is_odd (n : ℕ) : Prop := n % 2 = 1
def is_divisible_by_3 (n : ℕ) : Prop := n % 3 = 0
def has_digit_7 (n : ℕ) : Prop := n / 10 = 7 ∨ n % 10 = 7

-- Exactly three out of the four statements are true
def exactly_three_true (n : ℕ) : Prop :=
  (is_multiple_of_5 n ∧ is_odd n ∧ is_divisible_by_3 n ∧ ¬has_digit_7 n) ∨
  (is_multiple_of_5 n ∧ is_odd n ∧ ¬is_divisible_by_3 n ∧ has_digit_7 n) ∨
  (is_multiple_of_5 n ∧ ¬is_odd n ∧ is_divisible_by_3 n ∧ has_digit_7 n) ∨
  (¬is_multiple_of_5 n ∧ is_odd n ∧ is_divisible_by_3 n ∧ has_digit_7 n)

-- Main statement
theorem sarah_house_units_digit : ∃ n : ℕ, is_two_digit n ∧ exactly_three_true n ∧ n % 10 = 5 :=
by
  sorry

end sarah_house_units_digit_l202_202698


namespace sum_of_coefficients_l202_202164

theorem sum_of_coefficients (a b c d : ℝ) (f : ℝ → ℝ)
    (h1 : ∀ x, f (x + 2) = 2*x^3 + 5*x^2 + 3*x + 6)
    (h2 : ∀ x, f x = a*x^3 + b*x^2 + c*x + d) :
  a + b + c + d = 6 :=
by sorry

end sum_of_coefficients_l202_202164


namespace rhombus_longest_diagonal_l202_202980

theorem rhombus_longest_diagonal (area : ℝ) (ratio : ℝ) (h_area : area = 192) (h_ratio : ratio = 4 / 3) :
  ∃ d1 d2 : ℝ, d1 / d2 = 4 / 3 ∧ (d1 * d2) / 2 = 192 ∧ d1 = 16 * Real.sqrt 2 :=
by
  sorry

end rhombus_longest_diagonal_l202_202980


namespace power_of_7_mod_10_l202_202416

theorem power_of_7_mod_10 (k : ℕ) (h : 7^4 ≡ 1 [MOD 10]) : 7^150 ≡ 9 [MOD 10] :=
sorry

end power_of_7_mod_10_l202_202416


namespace number_of_solutions_is_zero_l202_202596

theorem number_of_solutions_is_zero : 
  ∀ x : ℝ, (x ≠ 0 ∧ x ≠ 5) → (3 * x^2 - 15 * x) / (x^2 - 5 * x) ≠ x - 2 :=
by
  sorry

end number_of_solutions_is_zero_l202_202596


namespace stone_reaches_bottom_l202_202614

structure StoneInWater where
  σ : ℝ   -- Density of stone in g/cm³
  d : ℝ   -- Depth of lake in cm
  g : ℝ   -- Acceleration due to gravity in cm/sec²
  σ₁ : ℝ  -- Density of water in g/cm³

noncomputable def time_and_velocity (siw : StoneInWater) : ℝ × ℝ :=
  let g₁ := ((siw.σ - siw.σ₁) / siw.σ) * siw.g
  let t := Real.sqrt ((2 * siw.d) / g₁)
  let v := g₁ * t
  (t, v)

theorem stone_reaches_bottom (siw : StoneInWater)
  (hσ : siw.σ = 2.1)
  (hd : siw.d = 850)
  (hg : siw.g = 980.8)
  (hσ₁ : siw.σ₁ = 1.0) :
  time_and_velocity siw = (1.82, 935) :=
by
  sorry

end stone_reaches_bottom_l202_202614


namespace cos_double_angle_l202_202081

open Real

-- Define the given conditions
variables {θ : ℝ}
axiom θ_in_interval : 0 < θ ∧ θ < π / 2
axiom sin_minus_cos : sin θ - cos θ = sqrt 2 / 2

-- Create a theorem that reflects the proof problem
theorem cos_double_angle : cos (2 * θ) = - sqrt 3 / 2 :=
by
  sorry

end cos_double_angle_l202_202081


namespace tangent_ellipse_hyperbola_l202_202967

theorem tangent_ellipse_hyperbola {m : ℝ} :
    (∀ x y : ℝ, x^2 + 9*y^2 = 9 → x^2 - m*(y + 1)^2 = 1 → false) →
    m = 72 :=
sorry

end tangent_ellipse_hyperbola_l202_202967


namespace solve_for_x_l202_202838

theorem solve_for_x : ∃ x : ℝ, (1 / 6 + 6 / x = 15 / x + 1 / 15) ∧ x = 90 :=
by
  sorry

end solve_for_x_l202_202838


namespace compute_x_l202_202890

theorem compute_x :
  (∑' n : ℕ, (1 / (3^n)) * (1 / (3^n) * (-1)^n)) = (∑' n : ℕ, 1 / (9^n)) →
  (∑' n : ℕ, (1 / (3^n)) * (1 / (3^n) * (-1)^n)) = 1 / (1 - (1 / 9)) →
  9 = 9 :=
by
  sorry

end compute_x_l202_202890


namespace x_squared_plus_inverse_squared_l202_202256

theorem x_squared_plus_inverse_squared (x : ℝ) (h : x + 1/x = 3.5) : x^2 + (1/x)^2 = 10.25 :=
by sorry

end x_squared_plus_inverse_squared_l202_202256


namespace Alice_fills_needed_l202_202167

def cups_needed : ℚ := 15/4
def cup_capacity : ℚ := 1/3
def fills_needed : ℚ := 12

theorem Alice_fills_needed : (cups_needed / cup_capacity).ceil = fills_needed := by
  -- Proof is omitted with sorry
  sorry

end Alice_fills_needed_l202_202167


namespace triangle_base_l202_202765

theorem triangle_base (h : ℝ) (A : ℝ) (b : ℝ) (h_eq : h = 10) (A_eq : A = 46) (area_eq : A = (b * h) / 2) : b = 9.2 :=
by
  -- sorry to be replaced with the actual proof
  sorry

end triangle_base_l202_202765


namespace expression_c_is_negative_l202_202735

noncomputable def A : ℝ := -4.2
noncomputable def B : ℝ := 2.3
noncomputable def C : ℝ := -0.5
noncomputable def D : ℝ := 3.4
noncomputable def E : ℝ := -1.8

theorem expression_c_is_negative : D / B * C < 0 := 
by
  -- proof goes here
  sorry

end expression_c_is_negative_l202_202735


namespace max_ab_value_l202_202914

theorem max_ab_value (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 2 * b = 4) : ab ≤ 2 :=
sorry

end max_ab_value_l202_202914


namespace Mary_forgot_pigs_l202_202024

theorem Mary_forgot_pigs (Mary_thinks : ℕ) (actual_animals : ℕ) (double_counted_sheep : ℕ)
  (H_thinks : Mary_thinks = 60) (H_actual : actual_animals = 56)
  (H_double_counted : double_counted_sheep = 7) :
  ∃ pigs_forgot : ℕ, pigs_forgot = 3 :=
by
  let counted_animals := Mary_thinks - double_counted_sheep
  have H_counted_correct : counted_animals = 53 := by sorry -- 60 - 7 = 53
  have pigs_forgot := actual_animals - counted_animals
  have H_pigs_forgot : pigs_forgot = 3 := by sorry -- 56 - 53 = 3
  exact ⟨pigs_forgot, H_pigs_forgot⟩

end Mary_forgot_pigs_l202_202024


namespace faye_initial_coloring_books_l202_202903

theorem faye_initial_coloring_books (gave_away1 gave_away2 remaining : ℝ) 
    (h1 : gave_away1 = 34.0) (h2 : gave_away2 = 3.0) (h3 : remaining = 11.0) :
    gave_away1 + gave_away2 + remaining = 48.0 := 
by
  sorry

end faye_initial_coloring_books_l202_202903


namespace converse_opposite_l202_202992

theorem converse_opposite (x y : ℝ) : (x + y = 0) → (y = -x) :=
by
  sorry

end converse_opposite_l202_202992


namespace smallest_number_multiple_of_45_and_exceeds_100_by_multiple_of_7_l202_202385

theorem smallest_number_multiple_of_45_and_exceeds_100_by_multiple_of_7 :
  ∃ n : ℕ, n % 45 = 0 ∧ (n - 100) % 7 = 0 ∧ n = 135 :=
sorry

end smallest_number_multiple_of_45_and_exceeds_100_by_multiple_of_7_l202_202385


namespace inequality_f_solution_minimum_g_greater_than_f_l202_202798

noncomputable def f (x : ℝ) := abs (x - 2) - abs (x + 1)

theorem inequality_f_solution : {x : ℝ | f x > 1} = {x | x < 0} :=
sorry

noncomputable def g (a x : ℝ) := (a * x^2 - x + 1) / x

theorem minimum_g_greater_than_f (a : ℝ) (h : 0 < a) : 
  (∀ x : ℝ, 0 < x → g a x > f x) ↔ 1 ≤ a :=
sorry

end inequality_f_solution_minimum_g_greater_than_f_l202_202798


namespace total_spent_l202_202592

theorem total_spent (cost_other_toys : ℕ) (cost_lightsaber : ℕ) 
  (h1 : cost_other_toys = 1000) 
  (h2 : cost_lightsaber = 2 * cost_other_toys) : 
  cost_lightsaber + cost_other_toys = 3000 :=
  by
    sorry

end total_spent_l202_202592


namespace sales_overlap_l202_202085

-- Define the conditions
def bookstore_sale_days : List ℕ := [2, 6, 10, 14, 18, 22, 26, 30]
def shoe_store_sale_days : List ℕ := [1, 8, 15, 22, 29]

-- Define the statement to prove
theorem sales_overlap : (bookstore_sale_days ∩ shoe_store_sale_days).length = 1 := 
by
  sorry

end sales_overlap_l202_202085


namespace binomial_19_10_l202_202338

theorem binomial_19_10 :
  ∀ (binom : ℕ → ℕ → ℕ),
  binom 17 7 = 19448 → binom 17 9 = 24310 →
  binom 19 10 = 92378 :=
by
  intros
  sorry

end binomial_19_10_l202_202338


namespace remainder_444_444_mod_13_l202_202265

theorem remainder_444_444_mod_13 :
  444 ≡ 3 [MOD 13] →
  3^3 ≡ 1 [MOD 13] →
  444^444 ≡ 1 [MOD 13] := by
  intros h1 h2
  sorry

end remainder_444_444_mod_13_l202_202265


namespace binom_150_1_l202_202727

-- Definition of the binomial coefficient
def binom (n k : ℕ) : ℕ := n.choose k

theorem binom_150_1 : binom 150 1 = 150 := by
  -- The proof is skipped and marked as sorry
  sorry

end binom_150_1_l202_202727


namespace total_bill_is_60_l202_202296

def num_adults := 6
def num_children := 2
def cost_adult := 6
def cost_child := 4
def cost_soda := 2

theorem total_bill_is_60 : num_adults * cost_adult + num_children * cost_child + (num_adults + num_children) * cost_soda = 60 := by
  sorry

end total_bill_is_60_l202_202296


namespace complement_union_l202_202337

def U : Set Nat := {1, 2, 3, 4}
def A : Set Nat := {1, 2}
def B : Set Nat := {2, 3}

theorem complement_union : U \ (A ∪ B) = {4} := by
  sorry

end complement_union_l202_202337


namespace remainder_of_3_pow_101_plus_5_mod_11_l202_202048

theorem remainder_of_3_pow_101_plus_5_mod_11 : (3 ^ 101 + 5) % 11 = 8 := by
  -- The theorem statement includes the condition that (3^101 + 5) mod 11 equals 8.
  -- The proof will make use of repetitive behavior and modular arithmetic.
  sorry

end remainder_of_3_pow_101_plus_5_mod_11_l202_202048


namespace geometric_sequence_fifth_term_l202_202823

variables (a r : ℝ) (h1 : a * r ^ 2 = 12 / 5) (h2 : a * r ^ 6 = 48)

theorem geometric_sequence_fifth_term : a * r ^ 4 = 12 / 5 := by
  sorry

end geometric_sequence_fifth_term_l202_202823


namespace smallest_total_books_l202_202086

-- Definitions based on conditions
def physics_books (x : ℕ) := 3 * x
def chemistry_books (x : ℕ) := 2 * x
def biology_books (x : ℕ) := (3 / 2 : ℚ) * x

-- Total number of books
def total_books (x : ℕ) := physics_books x + chemistry_books x + biology_books x

-- Statement of the theorem
theorem smallest_total_books :
  ∃ x : ℕ, total_books x = 15 ∧ 
           (∀ y : ℕ, y < x → total_books y % 1 ≠ 0) :=
sorry

end smallest_total_books_l202_202086


namespace determine_range_of_x_l202_202042

theorem determine_range_of_x (x : ℝ) (h₁ : 1/x < 3) (h₂ : 1/x > -2) : x > 1/3 ∨ x < -1/2 :=
sorry

end determine_range_of_x_l202_202042


namespace probability_of_not_shorter_than_one_meter_l202_202710

noncomputable def probability_of_event_A : ℝ := 
  let length_of_rope : ℝ := 3
  let event_A_probability : ℝ := 1 / 3
  event_A_probability

theorem probability_of_not_shorter_than_one_meter (l : ℝ) (h_l : l = 3) : 
    probability_of_event_A = 1 / 3 :=
sorry

end probability_of_not_shorter_than_one_meter_l202_202710


namespace triangles_fit_in_pan_l202_202329

theorem triangles_fit_in_pan (pan_length pan_width triangle_base triangle_height : ℝ)
  (h1 : pan_length = 15) (h2 : pan_width = 24) (h3 : triangle_base = 3) (h4 : triangle_height = 4) :
  (pan_length * pan_width) / (1/2 * triangle_base * triangle_height) = 60 :=
by
  sorry

end triangles_fit_in_pan_l202_202329


namespace original_five_digit_number_l202_202658

theorem original_five_digit_number :
  ∃ N y x : ℕ, (N = 10 * y + x) ∧ (N + y = 54321) ∧ (N = 49383) :=
by
  -- The proof script goes here
  sorry

end original_five_digit_number_l202_202658


namespace don_raise_l202_202253

variable (D R : ℝ)

theorem don_raise 
  (h1 : R = 0.08 * D)
  (h2 : 840 = 0.08 * 10500)
  (h3 : (D + R) - (10500 + 840) = 540) : 
  R = 880 :=
by sorry

end don_raise_l202_202253


namespace gcd_q_r_min_value_l202_202273

theorem gcd_q_r_min_value (p q r : ℕ) (hp : p > 0) (hq : q > 0) (hr : r > 0)
  (h1 : Nat.gcd p q = 210) (h2 : Nat.gcd p r = 770) : Nat.gcd q r = 10 :=
sorry

end gcd_q_r_min_value_l202_202273


namespace n_fifth_minus_n_divisible_by_30_l202_202278

theorem n_fifth_minus_n_divisible_by_30 (n : ℤ) : 30 ∣ (n^5 - n) :=
sorry

end n_fifth_minus_n_divisible_by_30_l202_202278


namespace find_k_l202_202738

theorem find_k (k : ℝ) : 
  (∀ α β : ℝ, (α * β = 15 ∧ α + β = -k ∧ (α + 3 + β + 3 = k)) → k = 3) :=
by 
  sorry

end find_k_l202_202738


namespace initial_ratio_of_milk_water_l202_202105

theorem initial_ratio_of_milk_water (M W : ℝ) (H1 : M + W = 85) (H2 : M / (W + 5) = 3) : M / W = 27 / 7 :=
by sorry

end initial_ratio_of_milk_water_l202_202105


namespace monotonically_increasing_intervals_min_and_max_values_l202_202363

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt 2) * Real.sin (2 * x + Real.pi / 4) + 1

theorem monotonically_increasing_intervals :
  ∀ k : ℤ, ∀ x : ℝ, 
    -3 * Real.pi / 8 + k * Real.pi ≤ x ∧ x ≤ Real.pi / 8 + k * Real.pi → 
    f (x + 1) ≥ f x := sorry

theorem min_and_max_values :
  ∃ min max, 
    (∀ x ∈ Set.Icc (-Real.pi / 4) (Real.pi / 4), f x ≥ min ∧ f x ≤ max) ∧ 
    (min = 0) ∧ 
    (max = Real.sqrt 2 + 1) := sorry

end monotonically_increasing_intervals_min_and_max_values_l202_202363


namespace unused_combinations_eq_40_l202_202454

-- Defining the basic parameters
def num_resources : ℕ := 6
def total_combinations : ℕ := 2 ^ num_resources
def used_combinations : ℕ := 23

-- Calculating the number of unused combinations
theorem unused_combinations_eq_40 : total_combinations - 1 - used_combinations = 40 := by
  sorry

end unused_combinations_eq_40_l202_202454


namespace trajectory_of_moving_circle_l202_202575

-- Define the conditions
def passes_through (M : ℝ × ℝ) (A : ℝ × ℝ) : Prop :=
  M = A

def tangent_to_line (M : ℝ × ℝ) (l : ℝ) : Prop :=
  M.1 = -l

noncomputable def equation_of_trajectory (M : ℝ × ℝ) : Prop :=
  M.2 ^ 2 = 12 * M.1

theorem trajectory_of_moving_circle 
  (M : ℝ × ℝ)
  (A : ℝ × ℝ)
  (l : ℝ)
  (h1 : passes_through M (3, 0))
  (h2 : tangent_to_line M 3)
  : equation_of_trajectory M := 
sorry

end trajectory_of_moving_circle_l202_202575


namespace sqrt_inequality_l202_202669

theorem sqrt_inequality : (Real.sqrt 6 + Real.sqrt 7) > (2 * Real.sqrt 2 + Real.sqrt 5) :=
by {
  sorry
}

end sqrt_inequality_l202_202669


namespace perpendicular_bisector_c_value_l202_202129

theorem perpendicular_bisector_c_value :
  (∃ c : ℝ, ∀ x y : ℝ, 
    2 * x - y = c ↔ x = 5 ∧ y = 8) → c = 2 := 
by
  sorry

end perpendicular_bisector_c_value_l202_202129


namespace find_first_divisor_l202_202817

theorem find_first_divisor (x : ℕ) (k m : ℕ) (h₁ : 282 = k * x + 3) (h₂ : 282 = 9 * m + 3) : x = 31 :=
sorry

end find_first_divisor_l202_202817


namespace even_n_of_even_Omega_P_l202_202569

-- Define the Omega function
def Omega (N : ℕ) : ℕ := 
  N.factors.length

-- Define the polynomial function P
def P (x : ℕ) (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  List.prod (List.map (λ i => x + a i) (List.range n))

theorem even_n_of_even_Omega_P (a : ℕ → ℕ) (n : ℕ)
  (H : ∀ k > 0, Even (Omega (P k a n))) : Even n :=
by
  sorry

end even_n_of_even_Omega_P_l202_202569


namespace twelve_percent_greater_than_80_l202_202766

theorem twelve_percent_greater_than_80 (x : ℝ) (h : x = 80 + 0.12 * 80) : x = 89.6 :=
by
  sorry

end twelve_percent_greater_than_80_l202_202766


namespace minimum_value_of_g_l202_202617

noncomputable def g (a b x : ℝ) : ℝ :=
  max (|x + a|) (|x + b|)

theorem minimum_value_of_g (a b : ℝ) (h : a < b) :
  ∃ x : ℝ, g a b x = (b - a) / 2 :=
by
  use - (a + b) / 2
  sorry

end minimum_value_of_g_l202_202617


namespace expression_equals_a5_l202_202536

theorem expression_equals_a5 (a : ℝ) : a^4 * a = a^5 := 
by sorry

end expression_equals_a5_l202_202536


namespace total_lambs_l202_202984

-- Defining constants
def Merry_lambs : ℕ := 10
def Brother_lambs : ℕ := Merry_lambs + 3

-- Proving the total number of lambs
theorem total_lambs : Merry_lambs + Brother_lambs = 23 :=
  by
    -- The actual proof is omitted and a placeholder is put instead
    sorry

end total_lambs_l202_202984


namespace total_profit_or_loss_is_negative_175_l202_202398

theorem total_profit_or_loss_is_negative_175
    (price_A price_B selling_price : ℝ)
    (profit_A loss_B : ℝ)
    (h1 : selling_price = 2100)
    (h2 : profit_A = 0.2)
    (h3 : loss_B = 0.2)
    (hA : price_A * (1 + profit_A) = selling_price)
    (hB : price_B * (1 - loss_B) = selling_price) :
    (selling_price + selling_price) - (price_A + price_B) = -175 := 
by 
  -- The proof is omitted
  sorry

end total_profit_or_loss_is_negative_175_l202_202398


namespace calc_a8_l202_202783

variable {a : ℕ+ → ℕ}

-- Conditions
axiom recur_relation : ∀ (p q : ℕ+), a (p + q) = a p * a q
axiom initial_condition : a 2 = 2

-- Proof statement
theorem calc_a8 : a 8 = 16 := by
  sorry

end calc_a8_l202_202783


namespace find_a11_l202_202231

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}

-- Given conditions
axiom cond1 : ∀ n : ℕ, n > 0 → 4 * S n = 2 * a n - n^2 + 7 * n

-- Theorem stating the proof problem
theorem find_a11 :
  a 11 = -2 :=
sorry

end find_a11_l202_202231


namespace quadratic_function_properties_l202_202972

theorem quadratic_function_properties :
  ∃ a : ℝ, ∃ f : ℝ → ℝ,
    (∀ x : ℝ, f x = a * (x + 1) ^ 2 - 2) ∧
    (f 1 = 10) ∧
    (f (-1) = -2) ∧
    (∀ x : ℝ, x > -1 → f x ≥ f (-1))
:=
by
  sorry

end quadratic_function_properties_l202_202972


namespace distinct_divisor_sum_l202_202910

theorem distinct_divisor_sum (n : ℕ) (x : ℕ) (h : x < n.factorial) :
  ∃ (k : ℕ) (d : Fin k → ℕ), (k ≤ n) ∧ (∀ i j, i ≠ j → d i ≠ d j) ∧ (∀ i, d i ∣ n.factorial) ∧ (x = Finset.sum Finset.univ d) :=
sorry

end distinct_divisor_sum_l202_202910


namespace necessary_but_not_sufficient_condition_l202_202088

theorem necessary_but_not_sufficient_condition (x : ℝ) : (|x - 1| < 1 → x^2 - 5 * x < 0) ∧ (¬(x^2 - 5 * x < 0 → |x - 1| < 1)) :=
by
  sorry

end necessary_but_not_sufficient_condition_l202_202088


namespace job_positions_growth_rate_l202_202724

theorem job_positions_growth_rate (x : ℝ) :
  1501 * (1 + x) ^ 2 = 1815 := sorry

end job_positions_growth_rate_l202_202724


namespace girls_in_wind_band_not_string_band_l202_202611

def M_G : ℕ := 100
def F_G : ℕ := 80
def M_O : ℕ := 80
def F_O : ℕ := 100
def total_students : ℕ := 230
def boys_in_both : ℕ := 60

theorem girls_in_wind_band_not_string_band : (F_G - (total_students - (M_G + F_G + M_O + F_O - boys_in_both - boys_in_both))) = 10 :=
by
  sorry

end girls_in_wind_band_not_string_band_l202_202611


namespace tan_theta_value_l202_202324

noncomputable def tan_theta (θ : ℝ) : ℝ :=
  if (0 < θ) ∧ (θ < 2 * Real.pi) ∧ (Real.cos (θ / 2) = 1 / 3) then
    (2 * (2 * Real.sqrt 2) / (1 - (2 * Real.sqrt 2) ^ 2))
  else
    0 -- added default value for well-definedness

theorem tan_theta_value (θ : ℝ) (h₀: 0 < θ) (h₁ : θ < 2 * Real.pi) (h₂ : Real.cos (θ / 2) = 1 / 3) : 
  tan_theta θ = -4 * Real.sqrt 2 / 7 :=
by
  sorry

end tan_theta_value_l202_202324


namespace inequality_for_positive_reals_l202_202279

open Real

theorem inequality_for_positive_reals 
  (a b c : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c) :
  a^3 * b + b^3 * c + c^3 * a ≥ a * b * c * (a + b + c) :=
sorry

end inequality_for_positive_reals_l202_202279


namespace right_triangle_hypotenuse_l202_202796

noncomputable def triangle_hypotenuse (a b c : ℝ) : Prop :=
(a + b + c = 40) ∧
(a * b = 48) ∧
(a^2 + b^2 = c^2) ∧
(c = 18.8)

theorem right_triangle_hypotenuse :
  ∃ (a b c : ℝ), triangle_hypotenuse a b c :=
by
  sorry

end right_triangle_hypotenuse_l202_202796


namespace distinct_real_roots_l202_202937

def otimes (a b : ℝ) : ℝ := b^2 - a * b

theorem distinct_real_roots (m x : ℝ) :
  otimes (m - 2) x = m -> ∃ x1 x2 : ℝ, x1 ≠ x2 ∧
  (x^2 - (m - 2) * x - m = 0) := by
  sorry

end distinct_real_roots_l202_202937


namespace sqrt_expression_equality_l202_202013

theorem sqrt_expression_equality :
  Real.sqrt ((5 - 3 * Real.sqrt 2) ^ 2) + Real.sqrt ((5 + 3 * Real.sqrt 2) ^ 2) = 10 :=
by
  sorry

end sqrt_expression_equality_l202_202013


namespace ratio_of_fish_cat_to_dog_l202_202718

theorem ratio_of_fish_cat_to_dog (fish_dog : ℕ) (cost_per_fish : ℕ) (total_spent : ℕ)
  (h1 : fish_dog = 40)
  (h2 : cost_per_fish = 4)
  (h3 : total_spent = 240) :
  (total_spent / cost_per_fish - fish_dog) / fish_dog = 1 / 2 := by
  sorry

end ratio_of_fish_cat_to_dog_l202_202718


namespace number_of_drawings_on_first_page_l202_202858

-- Let D be the number of drawings on the first page.
variable (D : ℕ)

-- Conditions:
-- 1. D is the number of drawings on the first page.
-- 2. The number of drawings increases by 5 after every page.
-- 3. The total number of drawings in the first five pages is 75.

theorem number_of_drawings_on_first_page (h : D + (D + 5) + (D + 10) + (D + 15) + (D + 20) = 75) :
    D = 5 :=
by
  sorry

end number_of_drawings_on_first_page_l202_202858


namespace partition_subset_sum_l202_202516

variable {p k : ℕ}

def V_p (p : ℕ) := {k : ℕ | p ∣ (k * (k + 1) / 2) ∧ k ≥ 2 * p - 1}

theorem partition_subset_sum (p : ℕ) (hp : Nat.Prime p) (k : ℕ) : k ∈ V_p p := sorry

end partition_subset_sum_l202_202516


namespace problem_solution_l202_202080

-- Definitions of the conditions as Lean statements:
def condition1 (t : ℝ) : Prop :=
  (1 + Real.sin t) * (1 - Real.cos t) = 1

def condition2 (t : ℝ) (a b c : ℕ) : Prop :=
  (1 - Real.sin t) * (1 + Real.cos t) = (a / b) - Real.sqrt c

def areRelativelyPrime (a b : ℕ) : Prop :=
  Nat.gcd a b = 1

-- The proof problem statement:
theorem problem_solution (t : ℝ) (a b c : ℕ) (h1 : condition1 t) (h2 : condition2 t a b c) (h3 : areRelativelyPrime a b) (ha_pos : 0 < a) (hb_pos : 0 < b) (hc_pos : 0 < c) : a + b + c = 2 := 
sorry

end problem_solution_l202_202080


namespace stormi_mowing_charge_l202_202383

theorem stormi_mowing_charge (cars_washed : ℕ) (car_wash_price : ℕ) (lawns_mowed : ℕ) (bike_cost : ℕ) (money_needed_more : ℕ) 
  (total_from_cars : ℕ := cars_washed * car_wash_price)
  (total_earned : ℕ := bike_cost - money_needed_more)
  (earned_from_lawns : ℕ := total_earned - total_from_cars) :
  cars_washed = 3 → car_wash_price = 10 → lawns_mowed = 2 → bike_cost = 80 → money_needed_more = 24 → earned_from_lawns / lawns_mowed = 13 := 
by
  sorry

end stormi_mowing_charge_l202_202383


namespace abs_eq_zero_sum_is_neg_two_l202_202659

theorem abs_eq_zero_sum_is_neg_two (x y : ℝ) (h : |x - 1| + |y + 3| = 0) : x + y = -2 := 
by 
  sorry

end abs_eq_zero_sum_is_neg_two_l202_202659


namespace average_weight_decrease_l202_202983

theorem average_weight_decrease 
  (weight_old_student : ℝ := 92) 
  (weight_new_student : ℝ := 72) 
  (number_of_students : ℕ := 5) : 
  (weight_old_student - weight_new_student) / ↑number_of_students = 4 :=
by 
  sorry

end average_weight_decrease_l202_202983


namespace hyperbola_constants_l202_202628

theorem hyperbola_constants (h k a c b : ℝ) : 
  h = -3 ∧ k = 1 ∧ a = 2 ∧ c = 5 ∧ b = Real.sqrt 21 → 
  h + k + a + b = 0 + Real.sqrt 21 :=
by
  intro hka
  sorry

end hyperbola_constants_l202_202628


namespace problem_condition_l202_202587

theorem problem_condition (x y : ℝ) (h : x^2 + y^2 - x * y = 1) : 
  x + y ≥ -2 ∧ x^2 + y^2 ≤ 2 :=
by
  sorry

end problem_condition_l202_202587


namespace PQRS_product_l202_202506

noncomputable def P : ℝ := (Real.sqrt 2023 + Real.sqrt 2024)
noncomputable def Q : ℝ := (-Real.sqrt 2023 - Real.sqrt 2024)
noncomputable def R : ℝ := (Real.sqrt 2023 - Real.sqrt 2024)
noncomputable def S : ℝ := (Real.sqrt 2024 - Real.sqrt 2023)

theorem PQRS_product : (P * Q * R * S) = 1 := 
by 
  sorry

end PQRS_product_l202_202506


namespace distance_between_countries_l202_202293

theorem distance_between_countries (total_distance : ℕ) (spain_germany : ℕ) (spain_other : ℕ) :
  total_distance = 7019 →
  spain_germany = 1615 →
  spain_other = total_distance - spain_germany →
  spain_other = 5404 :=
by
  intros h_total_distance h_spain_germany h_spain_other
  rw [h_total_distance, h_spain_germany] at h_spain_other
  exact h_spain_other

end distance_between_countries_l202_202293


namespace find_min_max_l202_202924

noncomputable def f (x y : ℝ) : ℝ := Real.sin x + Real.sin y - Real.sin (x + y)

theorem find_min_max :
  (∀ (x y : ℝ), 0 ≤ x → 0 ≤ y → x + y ≤ 2 * Real.pi → 
    (0 ≤ f x y ∧ f x y ≤ 3 * Real.sqrt 3 / 2)) :=
sorry

end find_min_max_l202_202924


namespace proportional_function_decreases_l202_202365

-- Define the function y = -2x
def proportional_function (x : ℝ) : ℝ := -2 * x

-- State the theorem to prove that y decreases as x increases
theorem proportional_function_decreases (x y : ℝ) (h : y = proportional_function x) :
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → proportional_function x₁ > proportional_function x₂ := 
sorry

end proportional_function_decreases_l202_202365


namespace convert_to_scientific_notation_9600000_l202_202346

theorem convert_to_scientific_notation_9600000 :
  9600000 = 9.6 * 10^6 := 
sorry

end convert_to_scientific_notation_9600000_l202_202346


namespace only_solution_l202_202595

def phi : ℕ → ℕ := sorry  -- Euler's totient function
def d : ℕ → ℕ := sorry    -- Divisor function

theorem only_solution (n : ℕ) (h1 : n ∣ (phi n)^(d n) + 1) (h2 : ¬ d n ^ 5 ∣ n ^ (phi n) - 1) : n = 2 :=
sorry

end only_solution_l202_202595


namespace proof_problem_l202_202615

-- Defining lines l1, l2, l3
def l1 (x y : ℝ) : Prop := 3 * x + 4 * y = 2
def l2 (x y : ℝ) : Prop := 2 * x + y = -2
def l3 (x y : ℝ) : Prop := x - 2 * y = 1

-- Point P being the intersection of l1 and l2
def P : ℝ × ℝ := (-2, 2)

-- Definition of the first required line passing through P and the origin
def line_through_P_and_origin (x y : ℝ) : Prop := x + y = 0

-- Definition of the second required line passing through P and perpendicular to l3
def required_line (x y : ℝ) : Prop := 2 * x + y + 2 = 0

-- The theorem to prove
theorem proof_problem :
  (∃ x y, l1 x y ∧ l2 x y ∧ (x, y) = P) →
  (∀ x y, (x, y) = P → line_through_P_and_origin x y) ∧
  (∀ x y, (x, y) = P → required_line x y) :=
by
  sorry

end proof_problem_l202_202615


namespace ratio_seconds_l202_202876

theorem ratio_seconds (x : ℕ) (h : 12 / x = 6 / 240) : x = 480 :=
sorry

end ratio_seconds_l202_202876


namespace cost_of_article_l202_202483

theorem cost_of_article (C G : ℝ) (h1 : C + G = 348) (h2 : C + 1.05 * G = 350) : C = 308 :=
by
  sorry

end cost_of_article_l202_202483


namespace find_an_l202_202484

variable (a : ℕ → ℤ) (S : ℕ → ℤ)
variable (a₁ d : ℤ)

-- Conditions
def S4 : Prop := S 4 = 0
def a5 : Prop := a 5 = 5
def Sn (n : ℕ) : Prop := S n = n * (2 * a₁ + (n - 1) * d) / 2
def an (n : ℕ) : Prop := a n = a₁ + (n - 1) * d

-- Theorem statement
theorem find_an (S4_hyp : S 4 = 0) (a5_hyp : a 5 = 5) (Sn_hyp : ∀ n, S n = n * (2 * a₁ + (n - 1) * d) / 2) (an_hyp : ∀ n, a n = a₁ + (n - 1) * d) :
  ∀ n, a n = 2 * n - 5 :=
by 
  intros n

  -- Proof is omitted, added here for logical conclusion completeness
  sorry

end find_an_l202_202484


namespace Rachel_books_total_l202_202207

-- Define the conditions
def mystery_shelves := 6
def picture_shelves := 2
def scifi_shelves := 3
def bio_shelves := 4
def books_per_shelf := 9

-- Define the total number of books
def total_books := 
  mystery_shelves * books_per_shelf + 
  picture_shelves * books_per_shelf + 
  scifi_shelves * books_per_shelf + 
  bio_shelves * books_per_shelf

-- Statement of the problem
theorem Rachel_books_total : total_books = 135 := 
by
  -- Proof can be added here
  sorry

end Rachel_books_total_l202_202207


namespace area_square_15_cm_l202_202645

-- Define the side length of the square
def side_length : ℝ := 15

-- Define the area calculation for a square given the side length
def area_of_square (s : ℝ) : ℝ := s * s

-- The theorem statement translating the problem to Lean
theorem area_square_15_cm :
  area_of_square side_length = 225 :=
by
  -- We need to provide proof here, but 'sorry' is used to skip the proof as per instructions
  sorry

end area_square_15_cm_l202_202645


namespace fraction_from_condition_l202_202096

theorem fraction_from_condition (x f : ℝ) (h : 0.70 * x = f * x + 110) (hx : x = 300) : f = 1 / 3 :=
by
  sorry

end fraction_from_condition_l202_202096


namespace possible_to_form_square_l202_202475

def shape_covers_units : ℕ := 4

theorem possible_to_form_square (shape : ℕ) : ∃ n : ℕ, ∃ k : ℕ, n * n = shape * k :=
by
  use 4
  use 4
  sorry

end possible_to_form_square_l202_202475


namespace positional_relationship_l202_202923

-- Definitions of skew_lines and parallel_lines
def skew_lines (a b : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, ¬ (a x y ∨ b x y) 

def parallel_lines (a c : ℝ → ℝ → Prop) : Prop :=
  ∃ k : ℝ, ∀ x y, c x y = a (k * x) (k * y)

-- Theorem statement
theorem positional_relationship (a b c : ℝ → ℝ → Prop) 
  (h1 : skew_lines a b) 
  (h2 : parallel_lines a c) : 
  skew_lines c b ∨ (∃ x y, c x y ∧ b x y) :=
sorry

end positional_relationship_l202_202923


namespace corn_harvest_l202_202961

theorem corn_harvest (rows : ℕ) (stalks_per_row : ℕ) (stalks_per_bushel : ℕ) (total_bushels : ℕ) :
  rows = 5 →
  stalks_per_row = 80 →
  stalks_per_bushel = 8 →
  total_bushels = (rows * stalks_per_row) / stalks_per_bushel →
  total_bushels = 50 :=
by
  intro h1 h2 h3 h4
  rw [h1, h2, h3, mul_comm 5 80] at h4
  norm_num at h4
  exact h4

end corn_harvest_l202_202961


namespace arithmetic_sequence_ninth_term_l202_202470

-- Definitions and Conditions
variables {a : ℕ → ℝ} (S : ℕ → ℝ)

-- Problem Statement
theorem arithmetic_sequence_ninth_term
  (h1 : a 3 = 4)
  (h2 : S 11 = 110)
  (h3 : ∀ n, S n = (n * (a 1 + a n)) / 2) :
  a 9 = 16 :=
sorry

end arithmetic_sequence_ninth_term_l202_202470


namespace H_double_prime_coordinates_l202_202247

/-- Define the points of the parallelogram EFGH and their reflections. --/
structure Point := (x : ℝ) (y : ℝ)

def E : Point := ⟨3, 4⟩
def F : Point := ⟨5, 7⟩
def G : Point := ⟨7, 4⟩
def H : Point := ⟨5, 1⟩

/-- Reflection of a point across the x-axis changes the y-coordinate sign. --/
def reflect_x (p : Point) : Point :=
  ⟨p.x, -p.y⟩

/-- Reflection of a point across y=x-1 involves translation and reflection across y=x. --/
def reflect_y_x_minus_1 (p : Point) : Point :=
  let translated := Point.mk p.x (p.y + 1)
  let reflected := Point.mk translated.y translated.x
  Point.mk reflected.x (reflected.y - 1)

def H' : Point := reflect_x H
def H'' : Point := reflect_y_x_minus_1 H'

theorem H_double_prime_coordinates : H'' = ⟨0, 4⟩ :=
by
  sorry

end H_double_prime_coordinates_l202_202247


namespace ratio_arithmetic_sequence_last_digit_l202_202754

def is_ratio_arithmetic_sequence (a : ℕ → ℕ) (d : ℕ) :=
  ∀ n : ℕ, n > 0 → (a (n + 2) * a n) = (a (n + 1) ^ 2) * d

theorem ratio_arithmetic_sequence_last_digit :
  ∃ a : ℕ → ℕ, is_ratio_arithmetic_sequence a 1 ∧ a 1 = 1 ∧ a 2 = 1 ∧ a 3 = 2 ∧
  (a 2009 / a 2006) % 10 = 6 :=
sorry

end ratio_arithmetic_sequence_last_digit_l202_202754


namespace smallest_three_digit_multiple_of_17_l202_202671

theorem smallest_three_digit_multiple_of_17 : 
    ∃ (n : ℤ), n ≥ 100 ∧ n < 1000 ∧ (∃ (k : ℤ), n = 17 * k) ∧ n = 102 :=
sorry

end smallest_three_digit_multiple_of_17_l202_202671


namespace Felix_can_lift_150_pounds_l202_202000

theorem Felix_can_lift_150_pounds : ∀ (weightFelix weightBrother : ℝ),
  (weightBrother = 2 * weightFelix) →
  (3 * weightBrother = 600) →
  (Felix_can_lift = 1.5 * weightFelix) →
  Felix_can_lift = 150 :=
by
  intros weightFelix weightBrother h1 h2 h3
  sorry

end Felix_can_lift_150_pounds_l202_202000


namespace quadratic_passes_through_neg3_n_l202_202471

-- Definition of the quadratic function with given conditions
def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Conditions provided in the problem
variables {a b c : ℝ}
axiom max_at_neg2 : ∀ x, quadratic a b c x ≤ 8
axiom value_at_neg2 : quadratic a b c (-2) = 8
axiom passes_through_1_4 : quadratic a b c 1 = 4

-- Statement to prove
theorem quadratic_passes_through_neg3_n : quadratic a b c (-3) = 68 / 9 :=
sorry

end quadratic_passes_through_neg3_n_l202_202471


namespace train_b_speed_l202_202968

/-- Given:
    1. Length of train A: 150 m
    2. Length of train B: 150 m
    3. Speed of train A: 54 km/hr
    4. Time taken to cross train B: 12 seconds
    Prove: The speed of train B is 36 km/hr
-/
theorem train_b_speed (l_A l_B : ℕ) (V_A : ℕ) (t : ℕ) (h1 : l_A = 150) (h2 : l_B = 150) (h3 : V_A = 54) (h4 : t = 12) :
  ∃ V_B : ℕ, V_B = 36 := sorry

end train_b_speed_l202_202968


namespace divisibility_by_30_l202_202788

theorem divisibility_by_30 (p : ℕ) (hp_prime : Nat.Prime p) (hp_ge_3 : p ≥ 3) : 30 ∣ (p^3 - 1) ↔ p % 15 = 1 := 
  sorry

end divisibility_by_30_l202_202788


namespace derivative_at_zero_l202_202827

noncomputable def f : ℝ → ℝ
| x => if x = 0 then 0 else Real.arcsin (x^2 * Real.cos (1 / (9 * x))) + (2 / 3) * x

theorem derivative_at_zero : HasDerivAt f (2 / 3) 0 := sorry

end derivative_at_zero_l202_202827


namespace find_x6_l202_202357

-- Definition of the variables xi for i = 1, ..., 10.
variables {x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 : ℝ}

-- Given conditions as equations.
axiom eq1 : (x2 + x4) / 2 = 3
axiom eq2 : (x4 + x6) / 2 = 5
axiom eq3 : (x6 + x8) / 2 = 7
axiom eq4 : (x8 + x10) / 2 = 9
axiom eq5 : (x10 + x2) / 2 = 1

axiom eq6 : (x1 + x3) / 2 = 2
axiom eq7 : (x3 + x5) / 2 = 4
axiom eq8 : (x5 + x7) / 2 = 6
axiom eq9 : (x7 + x9) / 2 = 8
axiom eq10 : (x9 + x1) / 2 = 10

-- The theorem to prove.
theorem find_x6 : x6 = 1 :=
by
  sorry

end find_x6_l202_202357


namespace proportion_of_mothers_full_time_jobs_l202_202378

theorem proportion_of_mothers_full_time_jobs
  (P : ℝ) (W : ℝ) (F : ℝ → Prop) (M : ℝ)
  (hwomen : W = 0.4 * P)
  (hfathers_full_time : ∀ p, F p → p = 0.75)
  (hno_full_time : P - (W + 0.75 * (P - W)) = 0.19 * P) :
  M = 0.9 :=
by
  sorry

end proportion_of_mothers_full_time_jobs_l202_202378


namespace area_ratio_correct_l202_202076

noncomputable def ratio_area_MNO_XYZ (s t u : ℝ) (S_XYZ : ℝ) : ℝ := 
  let S_XMO := s * (1 - u) * S_XYZ
  let S_YNM := t * (1 - s) * S_XYZ
  let S_OZN := u * (1 - t) * S_XYZ
  S_XYZ - S_XMO - S_YNM - S_OZN

theorem area_ratio_correct (s t u : ℝ) (h1 : s + t + u = 3 / 4) 
  (h2 : s^2 + t^2 + u^2 = 3 / 8) : 
  ratio_area_MNO_XYZ s t u 1 = 13 / 32 := 
by
  -- Proof omitted
  sorry

end area_ratio_correct_l202_202076


namespace comic_books_exclusive_count_l202_202244

theorem comic_books_exclusive_count 
  (shared_comics : ℕ) 
  (total_andrew_comics : ℕ) 
  (john_exclusive_comics : ℕ) 
  (h_shared_comics : shared_comics = 15) 
  (h_total_andrew_comics : total_andrew_comics = 22) 
  (h_john_exclusive_comics : john_exclusive_comics = 10) : 
  (total_andrew_comics - shared_comics + john_exclusive_comics = 17) := by 
  sorry

end comic_books_exclusive_count_l202_202244


namespace smallest_x_undefined_l202_202753

theorem smallest_x_undefined : ∃ x : ℝ, (10 * x^2 - 90 * x + 20 = 0) ∧ x = 1 / 4 :=
by sorry

end smallest_x_undefined_l202_202753


namespace symmetry_line_intersection_l202_202255

theorem symmetry_line_intersection 
  (k : ℝ) (k_pos : k > 0) (k_ne_one : k ≠ 1)
  (k1 : ℝ) (h_sym : ∀ (P : ℝ × ℝ), (P.2 = k1 * P.1 + 1) ↔ P.2 - 1 = k * (P.1 + 1) + 1)
  (H : ∀ M : ℝ × ℝ, (M.2 = k * M.1 + 1) → (M.1^2 / 4 + M.2^2 = 1)) :
  (k * k1 = 1) ∧ (∀ k : ℝ, ∃ P : ℝ × ℝ, (P.fst = 0) ∧ (P.snd = -5 / 3)) :=
sorry

end symmetry_line_intersection_l202_202255


namespace regular_polygon_sides_and_interior_angle_l202_202539

theorem regular_polygon_sides_and_interior_angle (n : ℕ) (H : (n - 2) * 180 = 3 * 360 + 180) :
  n = 9 ∧ (n - 2) * 180 / n = 140 :=
by
-- This marks the start of the proof, but the proof is omitted.
sorry

end regular_polygon_sides_and_interior_angle_l202_202539


namespace sum_of_x_intersections_is_zero_l202_202436

-- Definition of an even function
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

-- Definition for the x-coordinates of the intersection points with x-axis
def intersects_x_axis (f : ℝ → ℝ) (x_coords : List ℝ) : Prop :=
  (∀ x ∈ x_coords, f x = 0) ∧ (x_coords.length = 4)

-- Main theorem
theorem sum_of_x_intersections_is_zero 
  (f : ℝ → ℝ)
  (x_coords : List ℝ)
  (h1 : is_even_function f)
  (h2 : intersects_x_axis f x_coords) : 
  x_coords.sum = 0 :=
sorry

end sum_of_x_intersections_is_zero_l202_202436


namespace part1_part2_l202_202579

noncomputable def f : ℝ → ℝ := sorry

variable (x y : ℝ)
variable (hx0 : 0 < x)
variable (hy0 : 0 < y)
variable (hx12 : x < 1 → f x > 0)
variable (hf_half : f (1 / 2) = 1)
variable (hf_mul : f (x * y) = f x + f y)

theorem part1 : (∀ x1 x2, 0 < x1 → 0 < x2 → x1 < x2 → f x1 > f x2) := sorry

theorem part2 : (∀ x, 3 < x → x < 4 → f (x - 3) > f (1 / x) - 2) := sorry

end part1_part2_l202_202579


namespace probability_even_sum_5_balls_drawn_l202_202267

theorem probability_even_sum_5_balls_drawn :
  let total_ways := (Nat.choose 12 5)
  let favorable_ways := (Nat.choose 6 0) * (Nat.choose 6 5) + 
                        (Nat.choose 6 2) * (Nat.choose 6 3) + 
                        (Nat.choose 6 4) * (Nat.choose 6 1)
  favorable_ways / total_ways = 1 / 2 :=
by sorry

end probability_even_sum_5_balls_drawn_l202_202267


namespace find_remainder_l202_202128

theorem find_remainder (x y P Q : ℕ) (hx : 0 < x) (hy : 0 < y) (h : x^4 + y^4 = (P + 13) * (x + y) + Q) : Q = 8 :=
sorry

end find_remainder_l202_202128


namespace cheryl_used_material_l202_202822

theorem cheryl_used_material 
  (a b c l : ℚ) 
  (ha : a = 3 / 8) 
  (hb : b = 1 / 3) 
  (hl : l = 15 / 40) 
  (Hc: c = a + b): 
  (c - l = 1 / 3) := 
by 
  -- proof will be deferred to Lean's syntax for user to fill in.
  sorry

end cheryl_used_material_l202_202822


namespace maximum_gold_coins_l202_202850

theorem maximum_gold_coins (n k : ℕ) (h1 : n = 13 * k + 3) (h2 : n < 150) : n ≤ 146 :=
by
  sorry

end maximum_gold_coins_l202_202850


namespace degrees_to_radians_150_l202_202118

theorem degrees_to_radians_150 :
  (150 : ℝ) * (Real.pi / 180) = (5 * Real.pi) / 6 :=
by
  sorry

end degrees_to_radians_150_l202_202118


namespace percentage_calculation_l202_202712

theorem percentage_calculation :
  ( (2 / 3 * 2432 / 3 + 1 / 6 * 3225) / 450 * 100 ) = 239.54 := 
sorry

end percentage_calculation_l202_202712


namespace power_of_two_plus_one_div_by_power_of_three_l202_202661

theorem power_of_two_plus_one_div_by_power_of_three (n : ℕ) : 3^(n + 1) ∣ (2^(3^n) + 1) :=
sorry

end power_of_two_plus_one_div_by_power_of_three_l202_202661


namespace ryan_final_tokens_l202_202861

-- Conditions
def initial_tokens : ℕ := 36
def pacman_fraction : ℚ := 2 / 3
def candy_crush_fraction : ℚ := 1 / 2
def skiball_tokens : ℕ := 7
def friend_borrowed_tokens : ℕ := 5
def friend_returned_tokens : ℕ := 8
def laser_tag_tokens : ℕ := 3
def parents_purchase_factor : ℕ := 10

-- Final Answer
theorem ryan_final_tokens : initial_tokens - 24  - 6 - skiball_tokens + friend_returned_tokens + (parents_purchase_factor * skiball_tokens) - laser_tag_tokens = 75 :=
by sorry

end ryan_final_tokens_l202_202861


namespace find_wrong_quotient_l202_202106

-- Define the conditions
def correct_divisor : Nat := 21
def correct_quotient : Nat := 24
def mistaken_divisor : Nat := 12
def dividend : Nat := correct_divisor * correct_quotient

-- State the theorem to prove the wrong quotient
theorem find_wrong_quotient : (dividend / mistaken_divisor) = 42 := by
  sorry

end find_wrong_quotient_l202_202106


namespace solve_equation_l202_202546

theorem solve_equation : ∀ x : ℝ, (2 * x - 1)^2 - (1 - 3 * x)^2 = 5 * (1 - x) * (x + 1) → x = 5 / 2 :=
by
  intro x
  intro h
  sorry

end solve_equation_l202_202546


namespace gcd_consecutive_digits_l202_202145

theorem gcd_consecutive_digits (a : ℕ) (b : ℕ) (c : ℕ) (d : ℕ) 
  (h₁ : b = a + 1) (h₂ : c = a + 2) (h₃ : d = a + 3) :
  ∃ g, g = gcd (1000 * a + 100 * b + 10 * c + d - (1000 * d + 100 * c + 10 * b + a)) 3096 :=
by {
  sorry
}

end gcd_consecutive_digits_l202_202145


namespace theater_price_balcony_l202_202419

theorem theater_price_balcony 
  (price_orchestra : ℕ) (total_tickets : ℕ) (total_revenue : ℕ) (extra_balcony_tickets : ℕ) (price_balcony : ℕ) 
  (h1 : price_orchestra = 12) 
  (h2 : total_tickets = 380) 
  (h3 : total_revenue = 3320) 
  (h4 : extra_balcony_tickets = 240) 
  (h5 : ∃ (O : ℕ), O + (O + extra_balcony_tickets) = total_tickets ∧ (price_orchestra * O) + (price_balcony * (O + extra_balcony_tickets)) = total_revenue) : 
  price_balcony = 8 := 
by
  sorry

end theater_price_balcony_l202_202419


namespace exponential_function_pass_through_point_l202_202341

theorem exponential_function_pass_through_point
  (a : ℝ) (h₀ : a > 0) (h₁ : a ≠ 1) :
  (a^(1 - 1) + 1 = 2) :=
by
  sorry

end exponential_function_pass_through_point_l202_202341


namespace odds_against_horse_C_winning_l202_202756

theorem odds_against_horse_C_winning (odds_A : ℚ) (odds_B : ℚ) (odds_C : ℚ) 
  (cond1 : odds_A = 5 / 2) 
  (cond2 : odds_B = 3 / 1) 
  (race_condition : odds_C = 1 - ((2 / (5 + 2)) + (1 / (3 + 1))))
  : odds_C / (1 - odds_C) = 15 / 13 := 
sorry

end odds_against_horse_C_winning_l202_202756


namespace student_correct_sums_l202_202950

theorem student_correct_sums (x wrong total : ℕ) (h1 : wrong = 2 * x) (h2 : total = x + wrong) (h3 : total = 54) : x = 18 :=
by
  sorry

end student_correct_sums_l202_202950


namespace max_rubles_l202_202004

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

end max_rubles_l202_202004


namespace birds_in_store_l202_202450

/-- 
A pet store had a total of 180 animals, consisting of birds, dogs, and cats. 
Among the birds, 64 talked, and 13 didn't. If there were 40 dogs in the store 
and the number of birds that talked was four times the number of cats, 
prove that there were 124 birds in total.
-/
theorem birds_in_store (total_animals : ℕ) (talking_birds : ℕ) (non_talking_birds : ℕ) 
  (dogs : ℕ) (cats : ℕ) 
  (h1 : total_animals = 180)
  (h2 : talking_birds = 64)
  (h3 : non_talking_birds = 13)
  (h4 : dogs = 40)
  (h5 : talking_birds = 4 * cats) : 
  talking_birds + non_talking_birds + dogs + cats = 180 ∧ 
  talking_birds + non_talking_birds = 124 :=
by
  -- We are skipping the proof itself and focusing on the theorem statement
  sorry

end birds_in_store_l202_202450


namespace tiles_needed_l202_202574

def floor9ₓ12_ft : Type := {l : ℕ × ℕ // l = (9, 12)}
def tile4ₓ6_inch : Type := {l : ℕ × ℕ // l = (4, 6)}

theorem tiles_needed (floor : floor9ₓ12_ft) (tile : tile4ₓ6_inch) : 
  ∃ tiles : ℕ, tiles = 648 :=
sorry

end tiles_needed_l202_202574


namespace first_discount_percentage_l202_202148

theorem first_discount_percentage (x : ℝ) :
  let initial_price := 26.67
  let final_price := 15.0
  let second_discount := 0.25
  (initial_price * (1 - x / 100) * (1 - second_discount) = final_price) → x = 25 :=
by
  intros
  sorry

end first_discount_percentage_l202_202148


namespace yan_distance_ratio_l202_202426

theorem yan_distance_ratio (w x y : ℝ) (h1 : w > 0) (h2 : x > 0) (h3 : y > 0)
(h4 : y / w = x / w + (x + y) / (5 * w)) : x / y = 2 / 3 :=
by
  sorry

end yan_distance_ratio_l202_202426


namespace population_relation_l202_202297

-- Conditions: average life expectancies
def life_expectancy_gondor : ℝ := 64
def life_expectancy_numenor : ℝ := 92
def combined_life_expectancy (g n : ℕ) : ℝ := 85

-- Proof Problem: Given the conditions, prove the population relation
theorem population_relation (g n : ℕ) (h1 : life_expectancy_gondor * g + life_expectancy_numenor * n = combined_life_expectancy g n * (g + n)) : n = 3 * g :=
by
  sorry

end population_relation_l202_202297


namespace next_perfect_square_l202_202540

theorem next_perfect_square (n : ℤ) (hn : Even n) (x : ℤ) (hx : x = n^2) : 
  ∃ y : ℤ, y = x + 2 * n + 1 ∧ (∃ m : ℤ, y = m^2) ∧ m > n :=
by
  sorry

end next_perfect_square_l202_202540


namespace min_abs_val_of_36_power_minus_5_power_l202_202632

theorem min_abs_val_of_36_power_minus_5_power :
  ∃ (m n : ℕ), |(36^m : ℤ) - (5^n : ℤ)| = 11 := sorry

end min_abs_val_of_36_power_minus_5_power_l202_202632


namespace Mikaela_savings_l202_202530

theorem Mikaela_savings
  (hourly_rate : ℕ)
  (first_month_hours : ℕ)
  (additional_hours_second_month : ℕ)
  (spending_fraction : ℚ)
  (earnings_first_month := hourly_rate * first_month_hours)
  (hours_second_month := first_month_hours + additional_hours_second_month)
  (earnings_second_month := hourly_rate * hours_second_month)
  (total_earnings := earnings_first_month + earnings_second_month)
  (amount_spent := spending_fraction * total_earnings)
  (amount_saved := total_earnings - amount_spent) :
  hourly_rate = 10 →
  first_month_hours = 35 →
  additional_hours_second_month = 5 →
  spending_fraction = 4 / 5 →
  amount_saved = 150 :=
by
  sorry

end Mikaela_savings_l202_202530


namespace pages_read_on_fourth_day_l202_202050

-- condition: Hallie reads the whole book in 4 days, read specific pages each day
variable (total_pages first_day_pages second_day_pages third_day_pages fourth_day_pages : ℕ)

-- Given conditions
def conditions : Prop :=
  first_day_pages = 63 ∧
  second_day_pages = 2 * first_day_pages ∧
  third_day_pages = second_day_pages + 10 ∧
  total_pages = 354 ∧
  first_day_pages + second_day_pages + third_day_pages + fourth_day_pages = total_pages

-- Prove Hallie read 29 pages on the fourth day
theorem pages_read_on_fourth_day (h : conditions total_pages first_day_pages second_day_pages third_day_pages fourth_day_pages) :
  fourth_day_pages = 29 := sorry

end pages_read_on_fourth_day_l202_202050


namespace symmetry_origin_l202_202468

def f (x : ℝ) : ℝ := x^3 + x

theorem symmetry_origin : ∀ x : ℝ, f (-x) = -f x :=
by
  sorry

end symmetry_origin_l202_202468


namespace find_c_squared_ab_l202_202147

theorem find_c_squared_ab (a b c : ℝ) (h1 : a^2 * (b + c) = 2008) (h2 : b^2 * (a + c) = 2008) (h3 : a ≠ b) : 
  c^2 * (a + b) = 2008 :=
sorry

end find_c_squared_ab_l202_202147


namespace cube_divisibility_l202_202122

theorem cube_divisibility (a : ℤ) (k : ℤ) (h₁ : a > 1) 
(h₂ : (a - 1)^3 + a^3 + (a + 1)^3 = k^3) : 4 ∣ a := 
by
  sorry

end cube_divisibility_l202_202122


namespace maximum_ratio_l202_202156

-- Define the conditions
def is_two_digit_positive_integer (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99

def mean_is_45 (x y : ℕ) : Prop :=
  (x + y) / 2 = 45

-- State the theorem
theorem maximum_ratio (x y : ℕ) (hx : is_two_digit_positive_integer x) (hy : is_two_digit_positive_integer y) (h_mean : mean_is_45 x y) : 
  ∃ (k: ℕ), (x / y = k) ∧ k = 8 :=
sorry

end maximum_ratio_l202_202156


namespace monotonic_increasing_interval_l202_202982

noncomputable def f (x : ℝ) : ℝ := sorry

theorem monotonic_increasing_interval :
  (∀ x Δx : ℝ, 0 < x → 0 < Δx → 
  (f (x + Δx) - f x) / Δx = (2 / (Real.sqrt (x + Δx) + Real.sqrt x)) - (1 / (x^2 + x * Δx))) →
  ∀ x : ℝ, 1 < x → (∃ ε > 0, ∀ y, x < y ∧ y < x + ε → f y > f x) :=
by
  intro hyp
  sorry

end monotonic_increasing_interval_l202_202982


namespace find_triangle_areas_l202_202803

variables (A B C D : Point)
variables (S_ABC S_ACD S_ABD S_BCD : ℝ)

def quadrilateral_area (S_ABC S_ACD S_ABD S_BCD : ℝ) : Prop :=
  S_ABC + S_ACD + S_ABD + S_BCD = 25

def conditions (S_ABC S_ACD S_ABD S_BCD : ℝ) : Prop :=
  (S_ABC = 2 * S_BCD) ∧ (S_ABD = 3 * S_ACD)

theorem find_triangle_areas
  (S_ABC S_ACD S_ABD S_BCD : ℝ) :
  quadrilateral_area S_ABC S_ACD S_ABD S_BCD →
  conditions S_ABC S_ACD S_ABD S_BCD →
  S_ABC = 10 ∧ S_ACD = 5 ∧ S_ABD = 15 ∧ S_BCD = 10 :=
by
  sorry

end find_triangle_areas_l202_202803


namespace cost_of_each_soda_l202_202386

theorem cost_of_each_soda (total_cost sandwiches_cost : ℝ) (number_of_sodas : ℕ)
  (h_total_cost : total_cost = 6.46)
  (h_sandwiches_cost : sandwiches_cost = 2 * 1.49) :
  total_cost - sandwiches_cost = 4 * 0.87 := by
  sorry

end cost_of_each_soda_l202_202386


namespace min_value_x_2y_l202_202196

theorem min_value_x_2y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2 * y + 2 * x * y = 8) : x + 2 * y ≥ 4 :=
sorry

end min_value_x_2y_l202_202196


namespace solve_for_s_l202_202201

-- Definition of the given problem conditions
def parallelogram_sides_60_angle_sqrt_area (s : ℝ) :=
  ∃ (area : ℝ), (area = 27 * Real.sqrt 3) ∧
  (3 * s * s * Real.sqrt 3 = area)

-- Proof statement to demonstrate the equivalence of the theoretical and computed value of s
theorem solve_for_s (s : ℝ) : parallelogram_sides_60_angle_sqrt_area s → s = 3 :=
by
  intro h
  sorry

end solve_for_s_l202_202201


namespace line_intersection_x_value_l202_202594

theorem line_intersection_x_value :
  let line1 (x : ℝ) := 3 * x + 14
  let line2 (x : ℝ) (y : ℝ) := 5 * x - 2 * y = 40
  ∃ x : ℝ, ∃ y : ℝ, (line1 x = y) ∧ (line2 x y) ∧ (x = -68) :=
by
  sorry

end line_intersection_x_value_l202_202594


namespace Greg_harvested_acres_l202_202241

-- Defining the conditions
def Sharon_harvested : ℝ := 0.1
def Greg_harvested (additional: ℝ) (Sharon: ℝ) : ℝ := Sharon + additional

-- Proving the statement
theorem Greg_harvested_acres : Greg_harvested 0.3 Sharon_harvested = 0.4 :=
by
  sorry

end Greg_harvested_acres_l202_202241


namespace red_beads_count_is_90_l202_202366

-- Define the arithmetic sequence for red beads
def red_bead_count (n : ℕ) : ℕ := 2 * n

-- The sum of the first n terms in our sequence
def sum_red_beads (n : ℕ) : ℕ := n * (n + 1)

-- Verify the number of terms n such that the sum of red beads remains under 100
def valid_num_terms : ℕ := Nat.sqrt 99

-- Calculate total number of red beads on the necklace
def total_red_beads : ℕ := sum_red_beads valid_num_terms

theorem red_beads_count_is_90 (num_beads : ℕ) (valid : num_beads = 99) : 
  total_red_beads = 90 :=
by
  -- Proof skipped
  sorry

end red_beads_count_is_90_l202_202366


namespace quadratic_coefficient_a_l202_202778

theorem quadratic_coefficient_a (a b c : ℝ) :
  (2 = 9 * a - 3 * b + c) ∧
  (2 = 9 * a + 3 * b + c) ∧
  (-6 = 4 * a + 2 * b + c) →
  a = 8 / 5 :=
by
  sorry

end quadratic_coefficient_a_l202_202778


namespace problem1_problem2_problem3_problem4_l202_202868

-- Problem 1
theorem problem1 : -9 + 5 - 11 + 16 = 1 :=
by
  sorry

-- Problem 2
theorem problem2 : -9 + 5 - (-6) - 18 / (-3) = 8 :=
by
  sorry

-- Problem 3
theorem problem3 : -2^2 - ((-3) * (-4 / 3) - (-2)^3) = -16 :=
by
  sorry

-- Problem 4
theorem problem4 : (59 - (7 / 9 - 11 / 12 + 1 / 6) * (-6)^2) / (-7)^2 = 58 / 49 :=
by
  sorry

end problem1_problem2_problem3_problem4_l202_202868


namespace words_per_page_l202_202585

theorem words_per_page (p : ℕ) :
  (136 * p) % 203 = 184 % 203 ∧ p ≤ 100 → p = 73 :=
sorry

end words_per_page_l202_202585


namespace find_value_of_a2_plus_b2_plus_c2_l202_202182

variables (a b c : ℝ)

-- Define the conditions
def conditions := (a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) ∧ (a + b + c = 0) ∧ (a^3 + b^3 + c^3 = a^5 + b^5 + c^5)

-- State the theorem we need to prove
theorem find_value_of_a2_plus_b2_plus_c2 (h : conditions a b c) : a^2 + b^2 + c^2 = 6 / 5 :=
  sorry

end find_value_of_a2_plus_b2_plus_c2_l202_202182


namespace z_in_fourth_quadrant_l202_202849

noncomputable def z : ℂ := (3 * Complex.I - 2) / (Complex.I - 1) * Complex.I

theorem z_in_fourth_quadrant : z.re < 0 ∧ z.im > 0 := by
  sorry

end z_in_fourth_quadrant_l202_202849


namespace min_value_a_4b_l202_202136

theorem min_value_a_4b (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a * b = a + b) :
  a + 4 * b = 9 :=
sorry

end min_value_a_4b_l202_202136


namespace total_listening_days_l202_202731

theorem total_listening_days (x y z t : ℕ) (h1 : x = 8) (h2 : y = 12) (h3 : z = 30) (h4 : t = 2) :
  (x + y + z) * t = 100 :=
by
  sorry

end total_listening_days_l202_202731


namespace Mona_grouped_with_one_player_before_in_second_group_l202_202350

/-- Mona plays in groups with four other players, joined 9 groups, and grouped with 33 unique players. 
    One of the groups included 2 players she had grouped with before. 
    Prove that the number of players she had grouped with before in the second group is 1. -/
theorem Mona_grouped_with_one_player_before_in_second_group 
    (total_groups : ℕ) (group_size : ℕ) (unique_players : ℕ) 
    (repeat_players_in_group1 : ℕ) : 
    (total_groups = 9) → (group_size = 5) → (unique_players = 33) → (repeat_players_in_group1 = 2) 
        → ∃ repeat_players_in_group2 : ℕ, repeat_players_in_group2 = 1 :=
by
    sorry

end Mona_grouped_with_one_player_before_in_second_group_l202_202350


namespace expected_red_pairs_correct_l202_202186

-- Define the number of red cards and the total number of cards
def red_cards : ℕ := 25
def total_cards : ℕ := 50

-- Calculate the probability that one red card is followed by another red card in a circle of total_cards
def prob_adj_red : ℚ := (red_cards - 1) / (total_cards - 1)

-- The expected number of pairs of adjacent red cards
def expected_adj_red_pairs : ℚ := red_cards * prob_adj_red

-- The theorem to be proved: the expected number of adjacent red pairs is 600/49
theorem expected_red_pairs_correct : expected_adj_red_pairs = 600 / 49 :=
by
  -- Placeholder for the proof
  sorry

end expected_red_pairs_correct_l202_202186


namespace find_certain_number_l202_202493

theorem find_certain_number (n x : ℤ) (h1 : 9 - n / x = 7 + 8 / x) (h2 : x = 6) : n = 8 := by
  sorry

end find_certain_number_l202_202493


namespace train_speed_kph_l202_202666

-- Define conditions as inputs
def train_time_to_cross_pole : ℝ := 6 -- seconds
def train_length : ℝ := 100 -- meters

-- Conversion factor from meters per second to kilometers per hour
def mps_to_kph : ℝ := 3.6

-- Define and state the theorem to be proved
theorem train_speed_kph : (train_length / train_time_to_cross_pole) * mps_to_kph = 50 :=
by
  sorry

end train_speed_kph_l202_202666


namespace find_smallest_solution_l202_202852

theorem find_smallest_solution : ∃ x : ℝ, x = Real.sqrt 119 ∧ (Int.floor (x^2) - Int.floor x ^ 2 = 19) := by
  sorry

end find_smallest_solution_l202_202852


namespace find_larger_number_l202_202102

theorem find_larger_number (a b : ℝ) (h1 : a + b = 40) (h2 : a - b = 10) : a = 25 := by
  sorry

end find_larger_number_l202_202102


namespace children_neither_blue_nor_red_is_20_l202_202843

-- Definitions
def num_children : ℕ := 45
def num_adults : ℕ := num_children / 3
def num_adults_blue : ℕ := num_adults / 3
def num_adults_red : ℕ := 4
def num_adults_other_colors : ℕ := num_adults - num_adults_blue - num_adults_red
def num_children_red : ℕ := 15
def num_remaining_children : ℕ := num_children - num_children_red
def num_children_other_colors : ℕ := num_remaining_children / 2
def num_children_blue : ℕ := 2 * num_adults_blue
def num_children_neither_blue_nor_red : ℕ := num_children - num_children_red - num_children_blue

-- Theorem statement
theorem children_neither_blue_nor_red_is_20 : num_children_neither_blue_nor_red = 20 :=
  by
  sorry

end children_neither_blue_nor_red_is_20_l202_202843


namespace alpha_values_m_range_l202_202208

noncomputable section

open Real

def f (x : ℝ) (α : ℝ) : ℝ := 2^(x + cos α) - 2^(-x + cos α)

-- Problem 1: Set of values for α
theorem alpha_values (h : f 1 α = 3/4) : ∃ k : ℤ, α = 2 * k * π + π :=
sorry

-- Problem 2: Range of values for real number m
theorem m_range (h0 : 0 ≤ θ ∧ θ ≤ π / 2) 
  (h1 : ∀ (m : ℝ), f (m * cos θ) (-1) + f (1 - m) (-1) > 0) : 
  ∀ (m : ℝ), m < 1 :=
sorry

end alpha_values_m_range_l202_202208


namespace incenter_closest_to_median_l202_202342

variables (a b c : ℝ) (s_a s_b s_c d_a d_b d_c : ℝ)

noncomputable def median_length (a b c : ℝ) : ℝ := 
  Real.sqrt ((2 * b^2 + 2 * c^2 - a^2) / 4)

noncomputable def distance_to_median (x y median_length : ℝ) : ℝ := 
  (y - x) / (2 * median_length)

theorem incenter_closest_to_median
  (h₀ : a = 4) (h₁ : b = 5) (h₂ : c = 8) 
  (h₃ : s_a = median_length a b c)
  (h₄ : s_b = median_length b a c)
  (h₅ : s_c = median_length c a b)
  (h₆ : d_a = distance_to_median b c s_a)
  (h₇ : d_b = distance_to_median a c s_b)
  (h₈ : d_c = distance_to_median a b s_c) : 
  d_a = d_c := 
sorry

end incenter_closest_to_median_l202_202342


namespace tank_fill_time_with_leak_l202_202782

theorem tank_fill_time_with_leak 
  (pump_fill_time : ℕ) (leak_empty_time : ℕ) (effective_fill_time : ℕ)
  (hp : pump_fill_time = 5)
  (hl : leak_empty_time = 10)
  (he : effective_fill_time = 10) : effective_fill_time = 10 :=
by
  sorry

end tank_fill_time_with_leak_l202_202782


namespace sum_of_c_d_l202_202011

theorem sum_of_c_d (c d : ℝ) (g : ℝ → ℝ) 
(hg : ∀ x, g x = (x + 5) / (x^2 + c * x + d)) 
(hasymp : ∀ x, (x = 2 ∨ x = -3) → x^2 + c * x + d = 0) : 
c + d = -5 := 
by 
  sorry

end sum_of_c_d_l202_202011


namespace probability_heads_at_least_10_in_12_flips_l202_202053

theorem probability_heads_at_least_10_in_12_flips :
  let total_outcomes := 2^12
  let favorable_outcomes := (Nat.choose 12 10) + (Nat.choose 12 11) + (Nat.choose 12 12)
  let probability := (favorable_outcomes : ℚ) / (total_outcomes : ℚ)
  probability = 79 / 4096 := by
  sorry

end probability_heads_at_least_10_in_12_flips_l202_202053


namespace intersection_M_N_l202_202722

def setM : Set ℝ := {x | x^2 - 1 ≤ 0}
def setN : Set ℝ := {x | x^2 - 3 * x > 0}

theorem intersection_M_N :
  {x | -1 ≤ x ∧ x < 0} = setM ∩ setN :=
by
  sorry

end intersection_M_N_l202_202722


namespace min_value_expression_l202_202126

open Real

theorem min_value_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) 
  (hxyz : x^2 + y^2 + z^2 = 1) : 
  (∃ (c : ℝ), c = 3 * sqrt 3 / 2 ∧ c ≤ (x / (1 - x^2) + y / (1 - y^2) + z / (1 - z^2))) :=
by
  sorry

end min_value_expression_l202_202126


namespace distance_between_points_l202_202332

theorem distance_between_points :
  let p1 := (3, -5)
  let p2 := (-4, 4)
  dist p1 p2 = Real.sqrt 130 := by
  sorry

noncomputable def dist (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

end distance_between_points_l202_202332


namespace bonus_trigger_sales_amount_l202_202270

theorem bonus_trigger_sales_amount (total_sales S : ℝ) (h1 : 0.09 * total_sales = 1260)
  (h2 : 0.03 * (total_sales - S) = 120) : S = 10000 :=
sorry

end bonus_trigger_sales_amount_l202_202270


namespace smallest_positive_real_x_l202_202785

theorem smallest_positive_real_x :
  ∃ (x : ℝ), x > 0 ∧ ⌊x^2⌋ - x * ⌊x⌋ = 8 ∧ x = 89 / 9 :=
by
  sorry

end smallest_positive_real_x_l202_202785


namespace total_steps_l202_202871

theorem total_steps (up_steps down_steps : ℕ) (h1 : up_steps = 567) (h2 : down_steps = 325) : up_steps + down_steps = 892 := by
  sorry

end total_steps_l202_202871


namespace geometric_sequence_sum_l202_202458

-- Define the sequence and state the conditions
def geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ n, a (n + 1) / a n = a 1 / a 0

-- The mathematical problem rewritten in Lean 4 statement
theorem geometric_sequence_sum (a : ℕ → ℝ) (s : ℕ → ℝ)
  (h1 : geometric_sequence a)
  (h2 : s 2 = 7)
  (h3 : s 6 = 91)
  : ∃ s_4 : ℝ, s_4 = 28 :=
by
  sorry

end geometric_sequence_sum_l202_202458


namespace sin_cos_theta_l202_202826

-- Define the problem conditions and the question as a Lean statement
theorem sin_cos_theta (θ : ℝ) (h : Real.tan (θ + Real.pi / 2) = 2) : Real.sin θ * Real.cos θ = -2 / 5 := by
  sorry

end sin_cos_theta_l202_202826


namespace train_length_l202_202975

theorem train_length (L : ℕ) 
  (h_tree : L / 120 = L / 200 * 200) 
  (h_platform : (L + 800) / 200 = L / 120) : 
  L = 1200 :=
by
  sorry

end train_length_l202_202975


namespace total_pokemon_cards_l202_202793

-- Definitions based on conditions
def dozen := 12
def amount_per_person := 9 * dozen
def num_people := 4

-- Proposition to prove
theorem total_pokemon_cards :
  num_people * amount_per_person = 432 :=
by sorry

end total_pokemon_cards_l202_202793


namespace find_unknown_number_l202_202688

-- Definitions

-- Declaring that we have an inserted number 'a' between 3 and unknown number 'b'
variable (a b : ℕ)

-- Conditions provided in the problem
def arithmetic_sequence_condition (a b : ℕ) : Prop := 
  a - 3 = b - a

def geometric_sequence_condition (a b : ℕ) : Prop :=
  (a - 6) / 3 = b / (a - 6)

-- The theorem statement equivalent to the problem
theorem find_unknown_number (h1 : arithmetic_sequence_condition a b) (h2 : geometric_sequence_condition a b) : b = 27 :=
sorry

end find_unknown_number_l202_202688


namespace min_sum_xyz_l202_202618

theorem min_sum_xyz (x y z : ℝ) 
  (hx : x ≥ 4) (hy : y ≥ 5) (hz : z ≥ 6) 
  (hxyz : x^2 + y^2 + z^2 ≥ 90) : 
  x + y + z ≥ 16 := 
sorry

end min_sum_xyz_l202_202618


namespace multiple_of_Jills_age_l202_202807

theorem multiple_of_Jills_age (m : ℤ) : 
  ∀ (J R F : ℤ),
  J = 20 →
  F = 40 →
  R = m * J + 5 →
  (R + 15) - (J + 15) = (F + 15) - 30 →
  m = 2 :=
by
  intros J R F hJ hF hR hDiff
  sorry

end multiple_of_Jills_age_l202_202807


namespace width_of_room_l202_202494

theorem width_of_room (C r l : ℝ) (hC : C = 18700) (hr : r = 850) (hl : l = 5.5) : 
  ∃ w, C / r / l = w ∧ w = 4 :=
by
  use 4
  sorry

end width_of_room_l202_202494


namespace quadratic_has_two_real_roots_l202_202451

-- Define the condition that the discriminant must be non-negative
def discriminant_nonneg (a b c : ℝ) : Prop := b * b - 4 * a * c ≥ 0

-- Define our specific quadratic equation conditions: x^2 - 2x + m = 0
theorem quadratic_has_two_real_roots (m : ℝ) :
  discriminant_nonneg 1 (-2) m → m ≤ 1 :=
by
  sorry

end quadratic_has_two_real_roots_l202_202451


namespace tangent_line_at_origin_l202_202959

noncomputable def f (x : ℝ) := Real.log (1 + x) + x * Real.exp (-x)

theorem tangent_line_at_origin : 
  ∀ (x : ℝ), (1 : ℝ) * x + (0 : ℝ) = 2 * x := 
sorry

end tangent_line_at_origin_l202_202959


namespace second_number_multiple_of_seven_l202_202457

theorem second_number_multiple_of_seven (x : ℕ) (h : gcd (gcd 105 x) 2436 = 7) : 7 ∣ x :=
sorry

end second_number_multiple_of_seven_l202_202457


namespace dino_second_gig_hourly_rate_l202_202702

theorem dino_second_gig_hourly_rate (h1 : 20 * 10 = 200)
  (h2 : 5 * 40 = 200) (h3 : 500 + 500 = 1000) : 
  let total_income := 1000 
  let income_first_gig := 200 
  let income_third_gig := 200 
  let income_second_gig := total_income - income_first_gig - income_third_gig 
  let hours_second_gig := 30 
  let hourly_rate := income_second_gig / hours_second_gig 
  hourly_rate = 20 := 
by 
  sorry

end dino_second_gig_hourly_rate_l202_202702


namespace sum_of_fourth_powers_l202_202091

theorem sum_of_fourth_powers (n : ℤ) (h1 : n > 0) (h2 : (n - 1)^2 + n^2 + (n + 1)^2 = 9458) :
  (n - 1)^4 + n^4 + (n + 1)^4 = 30212622 :=
by sorry

end sum_of_fourth_powers_l202_202091


namespace interest_rate_l202_202901

theorem interest_rate (P CI SI: ℝ) (r: ℝ) : P = 5100 → CI = P * (1 + r)^2 - P → SI = P * r * 2 → (CI - SI = 51) → r = 0.1 :=
by
  intros
  -- skipping the proof
  sorry

end interest_rate_l202_202901


namespace tan_sum_eq_tan_prod_l202_202125

noncomputable def tan (x : Real) : Real :=
  Real.sin x / Real.cos x

theorem tan_sum_eq_tan_prod (α β γ : Real) (h : tan α + tan β + tan γ = tan α * tan β * tan γ) :
  ∃ k : Int, α + β + γ = k * Real.pi :=
by
  sorry

end tan_sum_eq_tan_prod_l202_202125


namespace sara_total_cents_l202_202009

-- Define the conditions as constants
def quarters : ℕ := 11
def value_per_quarter : ℕ := 25

-- Define the total amount formula based on the conditions
def total_cents (q : ℕ) (v : ℕ) : ℕ := q * v

-- The theorem to be proven
theorem sara_total_cents : total_cents quarters value_per_quarter = 275 :=
by
  -- Proof goes here
  sorry

end sara_total_cents_l202_202009


namespace zero_knights_l202_202815

noncomputable def knights_count (n : ℕ) : ℕ := sorry

theorem zero_knights (n : ℕ) (half_lairs : n ≥ 205) :
  knights_count 410 = 0 :=
sorry

end zero_knights_l202_202815


namespace person_is_not_sane_l202_202639

-- Definitions
def Person : Type := sorry
def sane : Person → Prop := sorry
def human : Person → Prop := sorry
def vampire : Person → Prop := sorry
def declares (p : Person) (s : String) : Prop := sorry

-- Conditions
axiom transylvanian_declares_vampire (p : Person) : declares p "I am a vampire"
axiom sane_human_never_claims_vampire (p : Person) : sane p → human p → ¬ declares p "I am a vampire"
axiom sane_vampire_never_admits_vampire (p : Person) : sane p → vampire p → ¬ declares p "I am a vampire"
axiom insane_human_might_claim_vampire (p : Person) : ¬ sane p → human p → declares p "I am a vampire"
axiom insane_vampire_might_admit_vampire (p : Person) : ¬ sane p → vampire p → declares p "I am a vampire"

-- Proof statement
theorem person_is_not_sane (p : Person) : declares p "I am a vampire" → ¬ sane p :=
by
  intros h
  sorry

end person_is_not_sane_l202_202639


namespace num_apartments_per_floor_l202_202580

-- Definitions used in the proof
def num_buildings : ℕ := 2
def floors_per_building : ℕ := 12
def doors_per_apartment : ℕ := 7
def total_doors_needed : ℕ := 1008

-- Lean statement to proof the number of apartments per floor
theorem num_apartments_per_floor : 
  (total_doors_needed / (doors_per_apartment * num_buildings * floors_per_building)) = 6 :=
by
  sorry

end num_apartments_per_floor_l202_202580


namespace parabola_centroid_locus_l202_202194

/-- Let P_0 be a parabola defined by the equation y = m * x^2. 
    Let A and B be points on P_0 such that the tangents at A and B are perpendicular. 
    Let G be the centroid of the triangle formed by A, B, and the vertex of P_0.
    Let P_n be the nth derived parabola.
    Prove that the equation of P_n is y = 3^n * m * x^2 + (1 / (4 * m)) * (1 - (1 / 3)^n). -/
theorem parabola_centroid_locus (n : ℕ) (m : ℝ) 
  (h_pos_m : 0 < m) :
  ∃ P_n : ℝ → ℝ, 
    ∀ x : ℝ, P_n x = 3^n * m * x^2 + (1 / (4 * m)) * (1 - (1 / 3)^n) :=
sorry

end parabola_centroid_locus_l202_202194


namespace collete_age_ratio_l202_202503

theorem collete_age_ratio (Ro R C : ℕ) (h1 : R = 2 * Ro) (h2 : Ro = 8) (h3 : R - C = 12) :
  C / Ro = 1 / 2 := by
sorry

end collete_age_ratio_l202_202503


namespace percentage_increase_l202_202636

noncomputable def percentMoreThan (a b : ℕ) : ℕ :=
  ((a - b) * 100) / b

theorem percentage_increase (x y z : ℕ) (h1 : z = 300) (h2 : x = 5 * y / 4) (h3 : x + y + z = 1110) :
  percentMoreThan y z = 20 := by
  sorry

end percentage_increase_l202_202636


namespace haruto_ratio_is_1_to_2_l202_202252

def haruto_tomatoes_ratio (total_tomatoes : ℕ) (eaten_by_birds : ℕ) (remaining_tomatoes : ℕ) : ℚ :=
  let picked_tomatoes := total_tomatoes - eaten_by_birds
  let given_to_friend := picked_tomatoes - remaining_tomatoes
  given_to_friend / picked_tomatoes

theorem haruto_ratio_is_1_to_2 : haruto_tomatoes_ratio 127 19 54 = 1 / 2 :=
by
  -- We'll skip the proof details as instructed
  sorry

end haruto_ratio_is_1_to_2_l202_202252


namespace power_cycle_i_pow_2012_l202_202017

-- Define the imaginary unit i as a complex number
def i : ℂ := Complex.I

-- Define the periodic properties of i
theorem power_cycle (n : ℕ) : Complex := 
  match n % 4 with
  | 0 => 1
  | 1 => i
  | 2 => -1
  | 3 => -i
  | _ => 0 -- this case should never happen

-- Using the periodic properties
theorem i_pow_2012 : (i ^ 2012) = 1 := by
  sorry

end power_cycle_i_pow_2012_l202_202017


namespace geometric_sequence_value_l202_202997

theorem geometric_sequence_value 
  (a : ℕ → ℝ)
  (r : ℝ)
  (h_geom : ∀ n, a (n + 1) = r * a n)
  (h_condition : a 4 * a 6 * a 8 * a 10 * a 12 = 32) :
  (a 10 ^ 2) / (a 12) = 2 :=
sorry

end geometric_sequence_value_l202_202997


namespace compute_product_l202_202744

theorem compute_product (s : ℂ) (h1 : s^7 = 1) (h2 : s ≠ 1) :
  (s - 1) * (s^2 - 1) * (s^3 - 1) * (s^4 - 1) * (s^5 - 1) * (s^6 - 1) = 10 :=
sorry

end compute_product_l202_202744


namespace rachel_class_choices_l202_202490

theorem rachel_class_choices : (Nat.choose 8 3) = 56 :=
by
  sorry

end rachel_class_choices_l202_202490


namespace y_in_terms_of_w_l202_202960

theorem y_in_terms_of_w (y w : ℝ) (h1 : y = 3^2 - 1) (h2 : w = 2) : y = 4 * w :=
by
  sorry

end y_in_terms_of_w_l202_202960


namespace second_team_pieces_l202_202625

-- Definitions for the conditions
def total_pieces_required : ℕ := 500
def pieces_first_team : ℕ := 189
def pieces_third_team : ℕ := 180

-- The number of pieces the second team made
def pieces_second_team : ℕ := total_pieces_required - (pieces_first_team + pieces_third_team)

-- The theorem we are proving
theorem second_team_pieces : pieces_second_team = 131 := by
  unfold pieces_second_team
  norm_num
  sorry

end second_team_pieces_l202_202625


namespace xyz_ratio_l202_202969

theorem xyz_ratio (k x y z : ℝ) (h1 : x + k * y + 3 * z = 0)
                                (h2 : 3 * x + k * y - 2 * z = 0)
                                (h3 : 2 * x + 4 * y - 3 * z = 0)
                                (x_ne_zero : x ≠ 0)
                                (y_ne_zero : y ≠ 0)
                                (z_ne_zero : z ≠ 0) :
  (k = 11) → (x * z) / (y ^ 2) = 10 := by
  sorry

end xyz_ratio_l202_202969


namespace range_of_f_l202_202428

def f (x : ℤ) := x + 1

theorem range_of_f : 
  (∀ x ∈ ({-1, 0, 1, 2} : Set ℤ), f x ∈ ({0, 1, 2, 3} : Set ℤ)) ∧ 
  (∀ y ∈ ({0, 1, 2, 3} : Set ℤ), ∃ x ∈ ({-1, 0, 1, 2} : Set ℤ), f x = y) := 
by 
  sorry

end range_of_f_l202_202428


namespace perimeter_gt_sixteen_l202_202638

theorem perimeter_gt_sixteen (a b : ℝ) (h : a * b > 2 * a + 2 * b) : 2 * (a + b) > 16 :=
by
  sorry

end perimeter_gt_sixteen_l202_202638


namespace teaching_arrangements_l202_202229

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

end teaching_arrangements_l202_202229


namespace acetone_mass_percentage_O_l202_202994

-- Definition of atomic masses
def atomic_mass_C := 12.01
def atomic_mass_H := 1.008
def atomic_mass_O := 16.00

-- Definition of the molar mass of acetone
def molar_mass_acetone := (3 * atomic_mass_C) + (6 * atomic_mass_H) + atomic_mass_O

-- Definition of mass percentage of oxygen in acetone
def mass_percentage_O_acetone := (atomic_mass_O / molar_mass_acetone) * 100

theorem acetone_mass_percentage_O : mass_percentage_O_acetone = 27.55 := by sorry

end acetone_mass_percentage_O_l202_202994


namespace defective_units_percentage_l202_202288

variables (D : ℝ)

-- 4% of the defective units are shipped for sale
def percent_defective_shipped : ℝ := 0.04

-- 0.24% of the units produced are defective units that are shipped for sale
def percent_total_defective_shipped : ℝ := 0.0024

-- The theorem to prove: the percentage of the units produced that are defective is 0.06
theorem defective_units_percentage (h : percent_defective_shipped * D = percent_total_defective_shipped) : D = 0.06 :=
sorry

end defective_units_percentage_l202_202288


namespace intersecting_lines_a_b_sum_zero_l202_202844

theorem intersecting_lines_a_b_sum_zero
    (a b : ℝ)
    (h₁ : ∀ z : ℝ × ℝ, z = (3, -3) → z.1 = (1 / 3) * z.2 + a)
    (h₂ : ∀ z : ℝ × ℝ, z = (3, -3) → z.2 = (1 / 3) * z.1 + b)
    :
    a + b = 0 := by
  sorry

end intersecting_lines_a_b_sum_zero_l202_202844


namespace greatest_possible_large_chips_l202_202348

theorem greatest_possible_large_chips 
  (s l : ℕ) 
  (p : ℕ) 
  (h1 : s + l = 72) 
  (h2 : s = l + p) 
  (h_prime : Prime p) : 
  l ≤ 35 :=
sorry

end greatest_possible_large_chips_l202_202348


namespace num_ordered_pairs_squares_diff_30_l202_202062

theorem num_ordered_pairs_squares_diff_30 :
  ∃ (n : ℕ), n = 0 ∧
  ∀ (m n: ℕ), 0 < m ∧ 0 < n ∧ m ≥ n ∧ m^2 - n^2 = 30 → false :=
by
  sorry

end num_ordered_pairs_squares_diff_30_l202_202062


namespace find_new_person_age_l202_202934

variables (A X : ℕ) -- A is the original average age, X is the age of the new person

def original_total_age (A : ℕ) := 10 * A
def new_total_age (A X : ℕ) := 10 * (A - 3)

theorem find_new_person_age (A : ℕ) (h : new_total_age A X = original_total_age A - 45 + X) : X = 15 :=
by
  sorry

end find_new_person_age_l202_202934


namespace right_triangle_side_81_exists_arithmetic_progression_l202_202664

theorem right_triangle_side_81_exists_arithmetic_progression :
  ∃ (a d : ℕ), a > 0 ∧ d > 0 ∧ (a - d)^2 + a^2 = (a + d)^2 ∧ (3*d = 81 ∨ 4*d = 81 ∨ 5*d = 81) :=
sorry

end right_triangle_side_81_exists_arithmetic_progression_l202_202664


namespace scientific_notation_correct_l202_202510

theorem scientific_notation_correct :
  52000000 = 5.2 * 10^7 :=
sorry

end scientific_notation_correct_l202_202510


namespace find_S6_l202_202504

variable {a : ℕ → ℝ} 
variable {S : ℕ → ℝ}

/-- sum_of_first_n_terms_of_geometric_sequence -/
def sum_of_first_n_terms_of_geometric_sequence (S : ℕ → ℝ) : Prop :=
  ∃ a1 r, ∀ n, S n = a1 * (1 - r^(n+1)) / (1 - r)

-- Given conditions
axiom geom_seq_positive_terms : ∀ n, a n > 0
axiom sum_S2 : S 2 = 3
axiom sum_S4 : S 4 = 15

theorem find_S6 : S 6 = 63 := by
  sorry

end find_S6_l202_202504


namespace mod_remainder_l202_202116

theorem mod_remainder (n : ℤ) (h : n % 3 = 2) : (7 * n) % 3 = 2 := by
  sorry

end mod_remainder_l202_202116


namespace Marnie_can_make_9_bracelets_l202_202534

def number_of_beads : Nat :=
  (5 * 50) + (2 * 100)

def beads_per_bracelet : Nat := 50

def total_bracelets (total_beads : Nat) (beads_per_bracelet : Nat) : Nat :=
  total_beads / beads_per_bracelet

theorem Marnie_can_make_9_bracelets :
  total_bracelets number_of_beads beads_per_bracelet = 9 :=
by
  -- proof goes here
  sorry

end Marnie_can_make_9_bracelets_l202_202534


namespace patrons_per_golf_cart_l202_202518

theorem patrons_per_golf_cart (patrons_from_cars patrons_from_bus golf_carts total_patrons patrons_per_cart : ℕ) 
  (h1 : patrons_from_cars = 12)
  (h2 : patrons_from_bus = 27)
  (h3 : golf_carts = 13)
  (h4 : total_patrons = patrons_from_cars + patrons_from_bus)
  (h5 : patrons_per_cart = total_patrons / golf_carts) : 
  patrons_per_cart = 3 := 
by
  sorry

end patrons_per_golf_cart_l202_202518


namespace sufficient_but_not_necessary_l202_202412

theorem sufficient_but_not_necessary (x : ℝ) : ((0 < x) → (|x-1| - |x| ≤ 1)) ∧ ((|x-1| - |x| ≤ 1) → True) ∧ ¬((|x-1| - |x| ≤ 1) → (0 < x)) := sorry

end sufficient_but_not_necessary_l202_202412


namespace find_integers_in_range_l202_202973

theorem find_integers_in_range :
  ∀ x : ℤ,
  (20 ≤ x ∧ x ≤ 50 ∧ (6 * x + 5) % 10 = 19) ↔
  x = 24 ∨ x = 29 ∨ x = 34 ∨ x = 39 ∨ x = 44 ∨ x = 49 :=
by sorry

end find_integers_in_range_l202_202973


namespace check_numbers_has_property_P_l202_202694

def has_property_P (n : ℤ) : Prop :=
  ∃ x y z : ℤ, n = x^3 + y^3 + z^3 - 3 * x * y * z

theorem check_numbers_has_property_P :
  has_property_P 1 ∧ has_property_P 5 ∧ has_property_P 2014 ∧ ¬has_property_P 2013 :=
by
  sorry

end check_numbers_has_property_P_l202_202694


namespace verify_condition_C_l202_202111

variable (x y z : ℤ)

-- Given conditions
def condition_C : Prop := x = y ∧ y = z + 1

-- The theorem/proof problem
theorem verify_condition_C (h : condition_C x y z) : (x - y)^2 + (y - z)^2 + (z - x)^2 = 2 := 
by 
  sorry

end verify_condition_C_l202_202111


namespace cafe_purchase_l202_202395

theorem cafe_purchase (s d : ℕ) (h_d : d ≥ 2) (h_cost : 5 * s + 125 * d = 4000) :  s + d = 11 :=
    -- Proof steps go here
    sorry

end cafe_purchase_l202_202395


namespace amc_problem_l202_202986

theorem amc_problem (a b : ℕ) (h : ∀ n : ℕ, 0 < n → a^n + n ∣ b^n + n) : a = b :=
sorry

end amc_problem_l202_202986


namespace product_simplification_l202_202687

variables {a b c : ℝ}

theorem product_simplification (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  ((a + b + c)⁻¹ * (a⁻¹ + b⁻¹ + c⁻¹) * (ab + bc + ac) * ((ab)⁻¹ + (bc)⁻¹ + (ac)⁻¹)) = 
  ((ab + bc + ac)^2) / (abc) := 
sorry

end product_simplification_l202_202687


namespace mixed_number_calculation_l202_202354

/-
  We need to define a proof that shows:
  75 * (2 + 3/7 - 5 * (1/3)) / (3 + 1/5 + 2 + 1/6) = -208 + 7/9
-/
theorem mixed_number_calculation :
  75 * ((17 / 7) - (16 / 3)) / ((16 / 5) + (13 / 6)) = -208 + 7 / 9 := by
  sorry

end mixed_number_calculation_l202_202354


namespace problem_statement_l202_202865

def arithmetic_seq (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

variable (a : ℕ → ℝ)
variable (d : ℝ)

-- defining conditions
axiom a1_4_7 : a 1 + a 4 + a 7 = 39
axiom a2_5_8 : a 2 + a 5 + a 8 = 33
axiom is_arithmetic : arithmetic_seq a d

theorem problem_statement : a 5 + a 8 + a 11 = 15 :=
by sorry

end problem_statement_l202_202865


namespace toms_total_cost_l202_202334

theorem toms_total_cost :
  let costA := 4 * 15
  let costB := 3 * 12
  let discountB := 0.20 * costB
  let costBDiscounted := costB - discountB
  let costC := 2 * 18
  costA + costBDiscounted + costC = 124.80 := 
by
  sorry

end toms_total_cost_l202_202334


namespace bird_stork_difference_l202_202626

theorem bird_stork_difference :
  let initial_birds := 3
  let initial_storks := 4
  let additional_birds := 2
  let total_birds := initial_birds + additional_birds
  total_birds - initial_storks = 1 := 
by
  let initial_birds := 3
  let initial_storks := 4
  let additional_birds := 2
  let total_birds := initial_birds + additional_birds
  show total_birds - initial_storks = 1
  sorry

end bird_stork_difference_l202_202626


namespace tan_pi_over_12_minus_tan_pi_over_6_l202_202061

theorem tan_pi_over_12_minus_tan_pi_over_6 :
  (Real.tan (Real.pi / 12) - Real.tan (Real.pi / 6)) = 7 - 4 * Real.sqrt 3 :=
  sorry

end tan_pi_over_12_minus_tan_pi_over_6_l202_202061


namespace find_magical_points_on_specific_quad_find_t_for_unique_magical_point_l202_202520

-- Define what it means to be a "magical point"
def is_magical_point (m : ℝ) (A : ℝ × ℝ) : Prop :=
  A = (m, 2 * m)

-- Specialize for the specific quadratic function y = x^2 - x - 4
def on_specific_quadratic (m : ℝ) (A : ℝ × ℝ) : Prop :=
  A = (m, m^2 - m - 4)

-- Theorem for part 1: Find the magical points on y = x^2 - x - 4
theorem find_magical_points_on_specific_quad (m : ℝ) (A : ℝ × ℝ) :
  is_magical_point m A ∧ on_specific_quadratic m A →
  (A = (4, 8) ∨ A = (-1, -2)) :=
sorry

-- Define the quadratic function for part 2
def on_general_quadratic (t m : ℝ) (A : ℝ × ℝ) : Prop :=
  A = (m, t * m^2 + (t-2) * m - 4)

-- Theorem for part 2: Find the t values for unique magical points
theorem find_t_for_unique_magical_point (t m : ℝ) (A : ℝ × ℝ) :
  ( ∀ m, is_magical_point m A ∧ on_general_quadratic t m A → 
    (t * m^2 + (t-4) * m - 4 = 0) ) → 
  ( ∃! m, is_magical_point m A ∧ on_general_quadratic t m A ) →
  t = -4 :=
sorry

end find_magical_points_on_specific_quad_find_t_for_unique_magical_point_l202_202520


namespace symmetric_line_eq_l202_202016

-- Definitions for the given line equations
def l1 (x y : ℝ) : Prop := 3 * x - y - 3 = 0
def l2 (x y : ℝ) : Prop := x + y - 1 = 0
def l3 (x y : ℝ) : Prop := x - 3 * y - 1 = 0

-- The theorem to prove
theorem symmetric_line_eq (x y : ℝ) (h1: l1 x y) (h2: l2 x y) : l3 x y :=
sorry

end symmetric_line_eq_l202_202016


namespace range_of_a_l202_202646

open Set

noncomputable def setA (a : ℝ) : Set ℝ := {x : ℝ | x^2 - 2*x + a ≥ 0}

theorem range_of_a (a : ℝ) : (1 ∉ setA a) → a < 1 :=
sorry

end range_of_a_l202_202646


namespace min_value_inequality_l202_202321

theorem min_value_inequality (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : 3 * a + 2 * b = 1) : 
  ∃ (m : ℝ), m = 25 ∧ (∀ x y, (x > 0) → (y > 0) → (3 * x + 2 * y = 1) → (3 / x + 2 / y) ≥ m) :=
sorry

end min_value_inequality_l202_202321


namespace math_competition_l202_202693

theorem math_competition :
  let Sammy_score := 20
  let Gab_score := 2 * Sammy_score
  let Cher_score := 2 * Gab_score
  let Total_score := Sammy_score + Gab_score + Cher_score
  let Opponent_score := 85
  Total_score - Opponent_score = 55 :=
by
  sorry

end math_competition_l202_202693


namespace probability_of_consonant_initials_l202_202026

def number_of_students : Nat := 30
def alphabet_size : Nat := 26
def redefined_vowels : Finset Char := {'A', 'E', 'I', 'O', 'U', 'Y'}
def number_of_vowels : Nat := redefined_vowels.card
def number_of_consonants : Nat := alphabet_size - number_of_vowels

theorem probability_of_consonant_initials :
  (number_of_consonants : ℝ) / (number_of_students : ℝ) = 2/3 := 
by
  -- Proof goes here
  sorry

end probability_of_consonant_initials_l202_202026


namespace determine_abcd_l202_202152

theorem determine_abcd (a b c d : ℕ) (h₀ : 0 ≤ a ∧ a ≤ 9) (h₁ : 0 ≤ b ∧ b ≤ 9) 
    (h₂ : 0 ≤ c ∧ c ≤ 9) (h₃ : 0 ≤ d ∧ d ≤ 9) 
    (h₄ : (10 * a + b) / 99 + (1000 * a + 100 * b + 10 * c + d) / 9999 = 27 / 37) :
    1000 * a + 100 * b + 10 * c + d = 3644 :=
by
  sorry

end determine_abcd_l202_202152


namespace expression_divisible_by_19_l202_202459

theorem expression_divisible_by_19 (n : ℕ) (h : n > 0) : 
  19 ∣ (5^(2*n - 1) + 3^(n - 2) * 2^(n - 1)) := 
by 
  sorry

end expression_divisible_by_19_l202_202459


namespace polynomial_smallest_e_l202_202932

theorem polynomial_smallest_e :
  ∃ (a b c d e : ℤ), (a*x^4 + b*x^3 + c*x^2 + d*x + e = 0 ∧ a ≠ 0 ∧ e > 0 ∧ (x + 3) * (x - 6) * (x - 10) * (2 * x + 1) = 0) 
  ∧ e = 180 :=
by
  sorry

end polynomial_smallest_e_l202_202932


namespace extra_men_needed_l202_202757

theorem extra_men_needed
  (total_length : ℕ) (total_days : ℕ) (initial_men : ℕ)
  (completed_days : ℕ) (completed_work : ℕ) (remaining_work : ℕ)
  (remaining_days : ℕ) (total_man_days_needed : ℕ)
  (number_of_men_needed : ℕ) (extra_men_needed : ℕ)
  (h1 : total_length = 10)
  (h2 : total_days = 60)
  (h3 : initial_men = 30)
  (h4 : completed_days = 20)
  (h5 : completed_work = 2)
  (h6 : remaining_work = total_length - completed_work)
  (h7 : remaining_days = total_days - completed_days)
  (h8 : total_man_days_needed = remaining_work * (completed_days * initial_men) / completed_work)
  (h9 : number_of_men_needed = total_man_days_needed / remaining_days)
  (h10 : extra_men_needed = number_of_men_needed - initial_men)
  : extra_men_needed = 30 :=
by sorry

end extra_men_needed_l202_202757


namespace total_increase_by_five_l202_202308

-- Let B be the number of black balls
variable (B : ℕ)
-- Let W be the number of white balls
variable (W : ℕ)
-- Initially the total number of balls
def T := B + W
-- If the number of black balls is increased to 5 times the original, the total becomes twice the original
axiom h1 : 5 * B + W = 2 * (B + W)
-- If the number of white balls is increased to 5 times the original 
def k : ℕ := 5
-- The new total number of balls 
def new_total := B + k * W

-- Prove that the new total is 4 times the original total.
theorem total_increase_by_five : new_total = 4 * T :=
by
sorry

end total_increase_by_five_l202_202308


namespace solve_for_q_l202_202447

theorem solve_for_q 
  (n m q : ℕ)
  (h1 : 5 / 6 = n / 60)
  (h2 : 5 / 6 = (m + n) / 90)
  (h3 : 5 / 6 = (q - m) / 150) : 
  q = 150 :=
sorry

end solve_for_q_l202_202447


namespace sum_of_v_values_is_zero_l202_202074

def v (x : ℝ) : ℝ := sorry

theorem sum_of_v_values_is_zero
  (h_odd : ∀ x : ℝ, v (-x) = -v x) :
  v (-3.14) + v (-1.57) + v (1.57) + v (3.14) = 0 :=
by
  sorry

end sum_of_v_values_is_zero_l202_202074


namespace find_remainder_l202_202593

theorem find_remainder (G : ℕ) (Q1 Q2 R1 : ℕ) (hG : G = 127) (h1 : 1661 = G * Q1 + R1) (h2 : 2045 = G * Q2 + 13) : R1 = 10 :=
by
  sorry

end find_remainder_l202_202593


namespace A_should_shoot_air_l202_202806

-- Define the problem conditions
def hits_A : ℝ := 0.3
def hits_B : ℝ := 1
def hits_C : ℝ := 0.5

-- Define turns
inductive Turn
| A | B | C

-- Define the strategic choice
inductive Strategy
| aim_C | aim_B | shoot_air

-- Define the outcome structure
structure DuelOutcome where
  winner : Option Turn
  probability : ℝ

-- Noncomputable definition given the context of probabilistic reasoning
noncomputable def maximize_survival : Strategy := 
sorry

-- Main theorem to prove the optimal strategy
theorem A_should_shoot_air : maximize_survival = Strategy.shoot_air := 
sorry

end A_should_shoot_air_l202_202806


namespace fraction_of_income_from_tips_l202_202859

variable (S T I : ℝ)
variable (h : T = (5 / 4) * S)

theorem fraction_of_income_from_tips (h : T = (5 / 4) * S) (I : ℝ) (w : I = S + T) : (T / I) = 5 / 9 :=
by
  -- The proof goes here
  sorry

end fraction_of_income_from_tips_l202_202859


namespace original_price_of_good_l202_202218

theorem original_price_of_good (P : ℝ) (h1 : 0.684 * P = 6840) : P = 10000 :=
sorry

end original_price_of_good_l202_202218


namespace orange_ring_weight_l202_202917

theorem orange_ring_weight :
  ∀ (p w t o : ℝ), 
  p = 0.33 → w = 0.42 → t = 0.83 → t - (p + w) = o → 
  o = 0.08 :=
by
  intro p w t o hp hw ht h
  rw [hp, hw, ht] at h
  -- Additional steps would go here, but
  sorry -- Skipping the proof as instructed

end orange_ring_weight_l202_202917


namespace remainder_of_67_pow_67_plus_67_mod_68_l202_202837

theorem remainder_of_67_pow_67_plus_67_mod_68 : (67^67 + 67) % 68 = 66 := by
  -- Add the conditions and final proof step
  sorry

end remainder_of_67_pow_67_plus_67_mod_68_l202_202837


namespace always_two_real_roots_find_m_l202_202188

-- Given quadratic equation: x^2 - 4mx + 3m^2 = 0
-- Definitions for the problem
def quadratic_eq (m x : ℝ) : Prop := x^2 - 4 * m * x + 3 * m^2 = 0

-- Q1: Prove that this equation always has two real roots.
theorem always_two_real_roots (m : ℝ) : ∃ x₁ x₂ : ℝ, quadratic_eq m x₁ ∧ quadratic_eq m x₂ :=
by
  sorry

-- Q2: If m > 0 and the difference between the two real roots is 2, find the value of m.
theorem find_m (m : ℝ) (h₁ : m > 0) (h₂ : ∃ x₁ x₂ : ℝ, quadratic_eq m x₁ ∧ quadratic_eq m x₂ ∧ |x₁ - x₂| = 2) : m = 1 :=
by
  sorry

end always_two_real_roots_find_m_l202_202188


namespace find_side_length_l202_202382

noncomputable def cos (x : ℝ) := Real.cos x

theorem find_side_length
  (A : ℝ) (c : ℝ) (b : ℝ) (a : ℝ)
  (hA : A = Real.pi / 3)
  (hc : c = Real.sqrt 3)
  (hb : b = 2 * Real.sqrt 3) :
  a = 3 := 
sorry

end find_side_length_l202_202382


namespace fixed_point_line_l202_202018

theorem fixed_point_line (k : ℝ) :
  ∃ A : ℝ × ℝ, (3 + k) * A.1 - 2 * A.2 + 1 - k = 0 ∧ (A = (1, 2)) :=
by
  let A : ℝ × ℝ := (1, 2)
  use A
  sorry

end fixed_point_line_l202_202018


namespace shaded_cubes_count_l202_202487

theorem shaded_cubes_count :
  let faces := 6
  let shaded_on_one_face := 5
  let corner_cubes := 8
  let center_cubes := 2 * 1 -- center cubes shared among opposite faces
  let total_shaded_cubes := corner_cubes + center_cubes
  faces = 6 → shaded_on_one_face = 5 → corner_cubes = 8 → center_cubes = 2 →
  total_shaded_cubes = 10 := 
by
  intros _ _ _ _ 
  sorry

end shaded_cubes_count_l202_202487


namespace basketball_team_heights_l202_202075

theorem basketball_team_heights :
  ∃ (second tallest third fourth shortest : ℝ),
  (tallest = 80.5 ∧
   second = tallest - 6.25 ∧
   third = second - 3.75 ∧
   fourth = third - 5.5 ∧
   shortest = fourth - 4.8 ∧
   second = 74.25 ∧
   third = 70.5 ∧
   fourth = 65 ∧
   shortest = 60.2) := sorry

end basketball_team_heights_l202_202075


namespace option_one_better_than_option_two_l202_202919

/-- Define the probability of winning in the first lottery option (drawing two red balls from a box
containing 4 red balls and 2 white balls). -/
def probability_option_one : ℚ := 2 / 5

/-- Define the probability of winning in the second lottery option (rolling two dice and having at least one die show a four). -/
def probability_option_two : ℚ := 11 / 36

/-- Prove that the probability of winning in the first lottery option is greater than the probability of winning in the second lottery option. -/
theorem option_one_better_than_option_two : probability_option_one > probability_option_two :=
by sorry

end option_one_better_than_option_two_l202_202919


namespace solve_686_l202_202547

theorem solve_686 : ∃ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ x^2 + y^2 + z^2 = 686 := 
by
  sorry

end solve_686_l202_202547


namespace value_of_n_l202_202705

theorem value_of_n (n : ℝ) : (∀ (x y : ℝ), x^2 + y^2 - 2 * n * x + 2 * n * y + 2 * n^2 - 8 = 0 → (x + 1)^2 + (y - 1)^2 = 2) → n = 1 :=
by
  sorry

end value_of_n_l202_202705


namespace quadratic_b_value_l202_202981

theorem quadratic_b_value (b m : ℝ) (h_b_pos : 0 < b) (h_quad_form : ∀ x, x^2 + b * x + 108 = (x + m)^2 - 4)
  (h_m_pos_sqrt : m = 4 * Real.sqrt 7 ∨ m = -4 * Real.sqrt 7) : b = 8 * Real.sqrt 7 :=
by
  sorry

end quadratic_b_value_l202_202981


namespace average_speed_l202_202202

def initial_odometer_reading : ℕ := 20
def final_odometer_reading : ℕ := 200
def travel_duration : ℕ := 6

theorem average_speed :
  (final_odometer_reading - initial_odometer_reading) / travel_duration = 30 := by
  sorry

end average_speed_l202_202202


namespace janet_spending_difference_l202_202896

-- Defining hourly rates and weekly hours for each type of lessons
def clarinet_hourly_rate := 40
def clarinet_weekly_hours := 3
def piano_hourly_rate := 28
def piano_weekly_hours := 5
def violin_hourly_rate := 35
def violin_weekly_hours := 2
def singing_hourly_rate := 45
def singing_weekly_hours := 1

-- Calculating weekly costs
def clarinet_weekly_cost := clarinet_hourly_rate * clarinet_weekly_hours
def piano_weekly_cost := piano_hourly_rate * piano_weekly_hours
def violin_weekly_cost := violin_hourly_rate * violin_weekly_hours
def singing_weekly_cost := singing_hourly_rate * singing_weekly_hours
def combined_weekly_cost := piano_weekly_cost + violin_weekly_cost + singing_weekly_cost

-- Calculating annual costs with 52 weeks in a year
def weeks_per_year := 52
def clarinet_annual_cost := clarinet_weekly_cost * weeks_per_year
def combined_annual_cost := combined_weekly_cost * weeks_per_year

-- Proving the final statement
theorem janet_spending_difference :
  combined_annual_cost - clarinet_annual_cost = 7020 := by sorry

end janet_spending_difference_l202_202896


namespace non_negative_solutions_l202_202760

theorem non_negative_solutions (x : ℕ) (h : 1 + x ≥ 2 * x - 1) : x = 0 ∨ x = 1 ∨ x = 2 := 
by {
  sorry
}

end non_negative_solutions_l202_202760


namespace sum_faces_edges_vertices_l202_202998

def faces : Nat := 6
def edges : Nat := 12
def vertices : Nat := 8

theorem sum_faces_edges_vertices : faces + edges + vertices = 26 :=
by
  sorry

end sum_faces_edges_vertices_l202_202998


namespace student_missed_20_l202_202179

theorem student_missed_20 {n : ℕ} (S_correct : ℕ) (S_incorrect : ℕ) 
    (h1 : S_correct = n * (n + 1) / 2)
    (h2 : S_incorrect = S_correct - 20) : 
    S_incorrect = n * (n + 1) / 2 - 20 := 
sorry

end student_missed_20_l202_202179


namespace similar_triangles_legs_sum_l202_202443

theorem similar_triangles_legs_sum (a b : ℕ) (h1 : a * b = 18) (h2 : a^2 + b^2 = 25) (bigger_area : ℕ) (smaller_area : ℕ) (hypotenuse : ℕ) 
  (h_similar : bigger_area = 225) 
  (h_smaller_area : smaller_area = 9) 
  (h_hypotenuse : hypotenuse = 5) 
  (h_non_3_4_5 : ¬ (a = 3 ∧ b = 4 ∨ a = 4 ∧ b = 3)) : 
  5 * (a + b) = 45 := 
by sorry

end similar_triangles_legs_sum_l202_202443


namespace distinct_L_shapes_l202_202204

-- Definitions of conditions
def num_convex_shapes : Nat := 10
def L_shapes_per_convex : Nat := 2
def corner_L_shapes : Nat := 4

-- Total number of distinct "L" shapes
def total_L_shapes : Nat :=
  num_convex_shapes * L_shapes_per_convex + corner_L_shapes

theorem distinct_L_shapes :
  total_L_shapes = 24 :=
by
  -- Proof is omitted
  sorry

end distinct_L_shapes_l202_202204


namespace purely_imaginary_m_complex_division_a_plus_b_l202_202418

-- Problem 1: Prove that m=-2 for z to be purely imaginary
theorem purely_imaginary_m (m : ℝ) (h : ∀ z : ℂ, z = (m - 1) * (m + 2) + (m - 1) * I → z.im = z.im) : m = -2 :=
sorry

-- Problem 2: Prove a+b = 13/10 with given conditions
theorem complex_division_a_plus_b (a b : ℝ) (m : ℝ) (h_m : m = 2) 
  (h_z : z = 4 + I) (h_eq : (z + I) / (z - I) = a + b * I) : a + b = 13 / 10 :=
sorry

end purely_imaginary_m_complex_division_a_plus_b_l202_202418


namespace tangent_line_circle_intersection_l202_202406

open Real

noncomputable def is_tangent (θ : ℝ) : Prop :=
  abs (4 * tan θ) / sqrt ((tan θ) ^ 2 + 1) = 2

theorem tangent_line_circle_intersection (θ : ℝ) (hθ : 0 ≤ θ ∧ θ < π) :
  is_tangent θ ↔ θ = π / 6 ∨ θ = 5 * π / 6 :=
sorry

end tangent_line_circle_intersection_l202_202406


namespace fraction_students_say_dislike_actually_like_l202_202663

theorem fraction_students_say_dislike_actually_like (total_students : ℕ) (like_dancing_fraction : ℚ) 
  (like_dancing_say_dislike_fraction : ℚ) (dislike_dancing_say_dislike_fraction : ℚ) : 
  (∃ frac : ℚ, frac = 40.7 / 100) :=
by
  let total_students := (200 : ℕ)
  let like_dancing_fraction := (70 / 100 : ℚ)
  let like_dancing_say_dislike_fraction := (25 / 100 : ℚ)
  let dislike_dancing_say_dislike_fraction := (85 / 100 : ℚ)
  
  let total_like_dancing := total_students * like_dancing_fraction
  let total_dislike_dancing :=  total_students * (1 - like_dancing_fraction)
  let like_dancing_say_dislike := total_like_dancing * like_dancing_say_dislike_fraction
  let dislike_dancing_say_dislike := total_dislike_dancing * dislike_dancing_say_dislike_fraction
  let total_say_dislike := like_dancing_say_dislike + dislike_dancing_say_dislike
  let fraction_say_dislike_actually_like := like_dancing_say_dislike / total_say_dislike
  
  existsi fraction_say_dislike_actually_like
  sorry

end fraction_students_say_dislike_actually_like_l202_202663


namespace A_investment_is_correct_l202_202761

-- Definitions based on the given conditions
def B_investment : ℝ := 8000
def C_investment : ℝ := 10000
def P_B : ℝ := 1000
def diff_P_A_P_C : ℝ := 500

-- Main statement we need to prove
theorem A_investment_is_correct (A_investment : ℝ) 
  (h1 : B_investment = 8000) 
  (h2 : C_investment = 10000)
  (h3 : P_B = 1000)
  (h4 : diff_P_A_P_C = 500)
  (h5 : A_investment = B_investment * (P_B / 1000) * 1.5) :
  A_investment = 12000 :=
sorry

end A_investment_is_correct_l202_202761


namespace sum_of_coefficients_is_225_l202_202564

theorem sum_of_coefficients_is_225 :
  let C4 := 1
  let C41 := 4
  let C42 := 6
  let C43 := 4
  (C4 + C41 + C42 + C43)^2 = 225 :=
by
  sorry

end sum_of_coefficients_is_225_l202_202564


namespace swallow_distance_flew_l202_202333

/-- The TGV departs from Paris at 150 km/h toward Marseille, which is 800 km away, while an intercité departs from Marseille at 50 km/h toward Paris at the same time. A swallow perched on the TGV takes off at that moment, flying at 200 km/h toward Marseille. We aim to prove that the distance flown by the swallow when the two trains meet is 800 km. -/
theorem swallow_distance_flew :
  let distance := 800 -- distance between Paris and Marseille in km
  let speed_TGV := 150 -- speed of TGV in km/h
  let speed_intercite := 50 -- speed of intercité in km/h
  let speed_swallow := 200 -- speed of swallow in km/h
  let combined_speed := speed_TGV + speed_intercite
  let time_to_meet := distance / combined_speed
  let distance_swallow_traveled := speed_swallow * time_to_meet
  distance_swallow_traveled = 800 := 
by
  sorry

end swallow_distance_flew_l202_202333


namespace total_highlighters_l202_202439

def num_pink_highlighters := 9
def num_yellow_highlighters := 8
def num_blue_highlighters := 5

theorem total_highlighters : 
  num_pink_highlighters + num_yellow_highlighters + num_blue_highlighters = 22 :=
by
  sorry

end total_highlighters_l202_202439


namespace find_f_values_l202_202441

def func_property1 (f : ℕ → ℕ) : Prop := 
  ∀ a b : ℕ, a ≠ b → a * f a + b * f b > a * f b + b * f a

def func_property2 (f : ℕ → ℕ) : Prop := 
  ∀ n : ℕ, f (f n) = 3 * n

theorem find_f_values (f : ℕ → ℕ) (h1 : func_property1 f) (h2 : func_property2 f) : 
  f 1 + f 6 + f 28 = 66 :=
sorry

end find_f_values_l202_202441


namespace cost_of_pencils_and_pens_l202_202461

theorem cost_of_pencils_and_pens (a b : ℝ) (h1 : 4 * a + b = 2.60) (h2 : a + 3 * b = 2.15) : 3 * a + 2 * b = 2.63 :=
sorry

end cost_of_pencils_and_pens_l202_202461


namespace probability_all_switches_on_is_correct_l202_202254

-- Mechanical declaration of the problem
structure SwitchState :=
  (state : Fin 2003 → Bool)

noncomputable def probability_all_on (initial : SwitchState) : ℚ :=
  let satisfying_confs := 2
  let total_confs := 2 ^ 2003
  let p := satisfying_confs / total_confs
  p

-- Definition of the term we want to prove
theorem probability_all_switches_on_is_correct :
  ∀ (initial : SwitchState), probability_all_on initial = 1 / 2 ^ 2002 :=
  sorry

end probability_all_switches_on_is_correct_l202_202254


namespace number_of_men_l202_202874

variable (W D X : ℝ)

theorem number_of_men (M_eq_2W : M = 2 * W)
  (wages_40_women : 21600 = 40 * W * D)
  (men_wages : 14400 = X * M * 20) :
  X = (2 / 3) * D :=
  by
  sorry

end number_of_men_l202_202874


namespace analogical_reasoning_l202_202491

theorem analogical_reasoning {a b c : ℝ} (h1 : c ≠ 0) : 
  (a + b) * c = a * c + b * c → (a + b) / c = a / c + b / c := 
by 
  sorry

end analogical_reasoning_l202_202491


namespace intersection_A_complement_UB_l202_202486

-- Definitions of the sets U, A, and B
def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def A : Set ℕ := {2, 3, 5, 6}
def B : Set ℕ := {x ∈ U | x^2 - 5 * x ≥ 0}

-- Complement of B w.r.t. U
def complement_U_B : Set ℕ := {x ∈ U | ¬ (x ∈ B)}

-- The statement we want to prove
theorem intersection_A_complement_UB : A ∩ complement_U_B = {2, 3} := by
  sorry

end intersection_A_complement_UB_l202_202486


namespace nails_to_buy_l202_202155

-- Define the initial number of nails Tom has
def initial_nails : ℝ := 247

-- Define the number of nails found in the toolshed
def toolshed_nails : ℝ := 144

-- Define the number of nails found in a drawer
def drawer_nails : ℝ := 0.5

-- Define the number of nails given by the neighbor
def neighbor_nails : ℝ := 58.75

-- Define the total number of nails needed for the project
def total_needed_nails : ℝ := 625.25

-- Define the total number of nails Tom already has
def total_existing_nails : ℝ := 
  initial_nails + toolshed_nails + drawer_nails + neighbor_nails

-- Prove that Tom needs to buy 175 more nails
theorem nails_to_buy :
  total_needed_nails - total_existing_nails = 175 := by
  sorry

end nails_to_buy_l202_202155


namespace parabola_directrix_l202_202633

theorem parabola_directrix (x y : ℝ) (h : y = 16 * x^2) : y = -1/64 :=
sorry

end parabola_directrix_l202_202633


namespace outfits_without_matching_color_l202_202551

theorem outfits_without_matching_color (red_shirts green_shirts pairs_pants green_hats red_hats : ℕ) 
  (h_red_shirts : red_shirts = 5) 
  (h_green_shirts : green_shirts = 5) 
  (h_pairs_pants : pairs_pants = 6) 
  (h_green_hats : green_hats = 8) 
  (h_red_hats : red_hats = 8) : 
  (red_shirts * pairs_pants * green_hats) + (green_shirts * pairs_pants * red_hats) = 480 := 
by 
  sorry

end outfits_without_matching_color_l202_202551


namespace average_speed_of_bus_trip_l202_202462

theorem average_speed_of_bus_trip 
  (v d : ℝ) 
  (h1 : d = 560)
  (h2 : ∀ v > 0, ∀ Δv > 0, (d / v) - (d / (v + Δv)) = 2)
  (h3 : Δv = 10): 
  v = 50 := 
by 
  sorry

end average_speed_of_bus_trip_l202_202462


namespace trebled_resultant_l202_202538

theorem trebled_resultant (n : ℕ) (h : n = 20) : 3 * ((2 * n) + 5) = 135 := 
by
  sorry

end trebled_resultant_l202_202538


namespace apples_left_over_l202_202891

-- Defining the number of apples collected by Liam, Mia, and Noah
def liam_apples := 53
def mia_apples := 68
def noah_apples := 22

-- The total number of apples collected
def total_apples := liam_apples + mia_apples + noah_apples

-- Proving that the remainder when the total number of apples is divided by 10 is 3
theorem apples_left_over : total_apples % 10 = 3 := by
  -- Placeholder for proof
  sorry

end apples_left_over_l202_202891


namespace MN_intersection_correct_l202_202644

-- Define the sets M and N
def setM : Set ℝ := {y | ∃ x ∈ (Set.univ : Set ℝ), y = x^2 + 2*x - 3}
def setN : Set ℝ := {x | |x - 2| ≤ 3}

-- Reformulated sets
def setM_reformulated : Set ℝ := {y | y ≥ -4}
def setN_reformulated : Set ℝ := {x | -1 ≤ x ∧ x ≤ 5}

-- The intersection set
def MN_intersection : Set ℝ := {y | -1 ≤ y ∧ y ≤ 5}

-- The theorem stating the intersection of M and N equals MN_intersection
theorem MN_intersection_correct :
  {y | ∃ x ∈ setN_reformulated, y = x^2 + 2*x - 3} = MN_intersection :=
sorry  -- Proof not required as per instruction

end MN_intersection_correct_l202_202644


namespace paula_paint_coverage_l202_202605

-- Define the initial conditions
def initial_capacity : ℕ := 36
def lost_cans : ℕ := 4
def reduced_capacity : ℕ := 28

-- Define the proof problem
theorem paula_paint_coverage :
  (initial_capacity - reduced_capacity = lost_cans * (initial_capacity / reduced_capacity)) →
  (reduced_capacity / (initial_capacity / reduced_capacity) = 14) :=
by
  sorry

end paula_paint_coverage_l202_202605


namespace adjusted_distance_buoy_fourth_l202_202331

theorem adjusted_distance_buoy_fourth :
  let a1 := 20  -- distance to the first buoy
  let d := 4    -- common difference (distance between consecutive buoys)
  let ocean_current_effect := 3  -- effect of ocean current
  
  -- distances from the beach to buoys based on their sequence
  let a2 := a1 + d 
  let a3 := a2 + d
  let a4 := a3 + d
  
  -- distance to the fourth buoy without external factors
  let distance_to_fourth_buoy := a1 + 3 * d
  
  -- adjusted distance considering the ocean current
  let adjusted_distance := distance_to_fourth_buoy - ocean_current_effect
  adjusted_distance = 29 := 
by
  let a1 := 20
  let d := 4
  let ocean_current_effect := 3
  let a2 := a1 + d
  let a3 := a2 + d
  let a4 := a3 + d
  let distance_to_fourth_buoy := a1 + 3 * d
  let adjusted_distance := distance_to_fourth_buoy - ocean_current_effect
  sorry

end adjusted_distance_buoy_fourth_l202_202331


namespace find_x_squared_plus_y_squared_find_xy_l202_202063

variable {x y : ℝ}

theorem find_x_squared_plus_y_squared (h1 : (x - y)^2 = 4) (h2 : (x + y)^2 = 64) : x^2 + y^2 = 34 :=
sorry

theorem find_xy (h1 : (x - y)^2 = 4) (h2 : (x + y)^2 = 64) : x * y = 15 :=
sorry

end find_x_squared_plus_y_squared_find_xy_l202_202063


namespace max_real_solution_under_100_l202_202729

theorem max_real_solution_under_100 (k a b c r : ℕ) (h0 : ∃ (m n p : ℕ), a = k^m ∧ b = k^n ∧ c = k^p)
  (h1 : r < 100) (h2 : b^2 = 4 * a * c) (h3 : r = b / (2 * a)) : r ≤ 64 :=
sorry

end max_real_solution_under_100_l202_202729


namespace cathy_wins_probability_l202_202953

theorem cathy_wins_probability : 
  (∑' (n : ℕ), (1 / 6 : ℚ)^3 * (5 / 6)^(3 * n)) = 1 / 91 
:= by sorry

end cathy_wins_probability_l202_202953


namespace total_money_raised_l202_202275

-- Given conditions:
def tickets_sold : Nat := 25
def ticket_price : ℚ := 2
def num_donations_15 : Nat := 2
def donation_15 : ℚ := 15
def donation_20 : ℚ := 20

-- Theorem statement proving the total amount raised is $100
theorem total_money_raised
  (h1 : tickets_sold = 25)
  (h2 : ticket_price = 2)
  (h3 : num_donations_15 = 2)
  (h4 : donation_15 = 15)
  (h5 : donation_20 = 20) :
  (tickets_sold * ticket_price + num_donations_15 * donation_15 + donation_20) = 100 := 
by
  sorry

end total_money_raised_l202_202275


namespace factorize_expression_l202_202310

variable (x : ℝ)

theorem factorize_expression : x^2 + x = x * (x + 1) :=
by
  sorry

end factorize_expression_l202_202310


namespace count_pairs_divisible_by_nine_l202_202099

open Nat

theorem count_pairs_divisible_by_nine (n : ℕ) (h : n = 528) :
  ∃ (count : ℕ), count = n ∧
  ∀ (a b : ℕ), 1 ≤ a ∧ a < b ∧ b ≤ 100 ∧ (a^2 + b^2 + a * b) % 9 = 0 ↔
  count = 528 :=
by
  sorry

end count_pairs_divisible_by_nine_l202_202099


namespace total_ingredients_l202_202001

theorem total_ingredients (b f s : ℕ) (h_ratio : 2 * f = 5 * f) (h_flour : f = 15) : b + f + s = 30 :=
by 
  sorry

end total_ingredients_l202_202001


namespace range_of_a_l202_202258

-- Define set A
def setA (x a : ℝ) : Prop := 2 * a ≤ x ∧ x ≤ a^2 + 1

-- Define set B
def setB (x a : ℝ) : Prop := (x - 2) * (x - (3 * a + 1)) ≤ 0

-- Theorem statement
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, setA x a → setB x a) ↔ (1 ≤ a ∧ a ≤ 3) ∨ (a = -1) :=
sorry

end range_of_a_l202_202258


namespace recurring_subtraction_l202_202355

theorem recurring_subtraction (x y : ℚ) (h1 : x = 35 / 99) (h2 : y = 7 / 9) : x - y = -14 / 33 := by
  sorry

end recurring_subtraction_l202_202355


namespace log_mult_l202_202769

theorem log_mult : 
  (Real.log 9 / Real.log 2) * (Real.log 4 / Real.log 3) = 4 :=
by 
  sorry

end log_mult_l202_202769


namespace find_coordinates_of_B_l202_202367

-- Define the conditions from the problem
def point_A (a : ℝ) : ℝ × ℝ := (a - 1, a + 1)
def point_B (a : ℝ) : ℝ × ℝ := (a + 3, a - 5)

-- The proof problem: The coordinates of B are (4, -4)
theorem find_coordinates_of_B (a : ℝ) (h : point_A a = (0, a + 1)) : point_B a = (4, -4) := by
  -- This is skipping the proof part.
  sorry

end find_coordinates_of_B_l202_202367


namespace new_average_l202_202945

theorem new_average (n : ℕ) (average : ℝ) (new_average : ℝ) 
  (h1 : n = 10)
  (h2 : average = 80)
  (h3 : new_average = (2 * average * n) / n) : 
  new_average = 160 := 
by 
  simp [h1, h2, h3]
  sorry

end new_average_l202_202945


namespace product_of_x_values_product_of_all_possible_x_values_l202_202311

theorem product_of_x_values (x : ℚ) (h : abs ((18 : ℚ) / x - 4) = 3) :
  x = 18 ∨ x = 18 / 7 :=
sorry

theorem product_of_all_possible_x_values (x1 x2 : ℚ) (h1 : abs ((18 : ℚ) / x1 - 4) = 3) (h2 : abs ((18 : ℚ) / x2 - 4) = 3) :
  x1 * x2 = 324 / 7 :=
sorry

end product_of_x_values_product_of_all_possible_x_values_l202_202311


namespace darius_scores_less_l202_202400

variable (D M Ma : ℕ)

-- Conditions
def condition1 := D = 10
def condition2 := Ma = D + 3
def condition3 := D + M + Ma = 38

-- Theorem to prove
theorem darius_scores_less (D M Ma : ℕ) (h1 : condition1 D) (h2 : condition2 D Ma) (h3 : condition3 D M Ma) : M - D = 5 :=
by
  sorry

end darius_scores_less_l202_202400


namespace quadratic_real_roots_k_leq_one_l202_202681

theorem quadratic_real_roots_k_leq_one (k : ℝ) : 
  (∃ x : ℝ, k * x^2 + 4 * x + 4 = 0) ↔ k ≤ 1 :=
by
  sorry

end quadratic_real_roots_k_leq_one_l202_202681


namespace ellipse_standard_equation_l202_202699

theorem ellipse_standard_equation (a b : ℝ) (h1 : 2 * a = 2 * (2 * b)) (h2 : (2, 0) ∈ {p : ℝ × ℝ | (p.1^2 / a^2) + (p.2^2 / b^2) = 1} ∨ (2, 0) ∈ {p : ℝ × ℝ | (p.2^2 / a^2) + (p.1^2 / b^2) = 1}) :
  (∃ a b : ℝ, (a > b ∧ a > 0 ∧ b > 0 ∧ (2 * a = 2 * (2 * b)) ∧ (2, 0) ∈ {p : ℝ × ℝ | (p.1^2 / a^2) + (p.2^2 / b^2) = 1} ∧ (∀ x y : ℝ, (x, y) ∈ {p : ℝ × ℝ | (p.1^2 / a^2) + (p.2^2 / b^2) = 1} → (x^2 / 4 + y^2 / 1 = 1)) ∨ (x^2 / 16 + y^2 / 4 = 1))) :=
  sorry

end ellipse_standard_equation_l202_202699


namespace joey_read_percentage_l202_202423

theorem joey_read_percentage : 
  ∀ (total_pages read_after_break : ℕ), 
  total_pages = 30 → read_after_break = 9 → 
  ( (total_pages - read_after_break : ℕ) / (total_pages : ℕ) * 100 ) = 70 :=
by
  intros total_pages read_after_break h_total h_after
  sorry

end joey_read_percentage_l202_202423


namespace distance_between_parallel_lines_l202_202936

theorem distance_between_parallel_lines (a d : ℝ) (d_pos : 0 ≤ d) (a_pos : 0 ≤ a) :
  {d_ | d_ = d + a ∨ d_ = |d - a|} = {d + a, abs (d - a)} :=
by
  sorry

end distance_between_parallel_lines_l202_202936


namespace mary_money_left_l202_202764

theorem mary_money_left (p : ℝ) : 50 - (4 * p + 2 * p + 4 * p) = 50 - 10 * p := 
by 
  sorry

end mary_money_left_l202_202764


namespace fifty_percent_of_x_l202_202692

variable (x : ℝ)

theorem fifty_percent_of_x (h : 0.40 * x = 160) : 0.50 * x = 200 :=
by
  sorry

end fifty_percent_of_x_l202_202692


namespace carlos_books_in_june_l202_202716

def books_in_july : ℕ := 28
def books_in_august : ℕ := 30
def goal_books : ℕ := 100

theorem carlos_books_in_june :
  let books_in_july_august := books_in_july + books_in_august
  let books_needed_june := goal_books - books_in_july_august
  books_needed_june = 42 := 
by
  sorry

end carlos_books_in_june_l202_202716


namespace no_two_ways_for_z_l202_202058

theorem no_two_ways_for_z (z : ℤ) (x y x' y' : ℕ) 
  (hx : x ≤ y) (hx' : x' ≤ y') : ¬ (z = x! + y! ∧ z = x'! + y'! ∧ (x ≠ x' ∨ y ≠ y')) :=
by
  sorry

end no_two_ways_for_z_l202_202058


namespace babblian_word_count_l202_202077

theorem babblian_word_count (n : ℕ) (h1 : n = 6) : ∃ m, m = 258 := by
  sorry

end babblian_word_count_l202_202077


namespace exists_unique_root_in_interval_l202_202620

noncomputable def f (x : ℝ) : ℝ := 2^x + x - 2

theorem exists_unique_root_in_interval : 
  ∃! x : ℝ, 0 < x ∧ x < 1 ∧ f x = 0 :=
sorry

end exists_unique_root_in_interval_l202_202620


namespace geometric_sequence_sum_l202_202701

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) 
  (h_geo : ∀ n, a (n + 1) = q * a n)
  (h1 : a 1 + a 2 + a 3 = 7)
  (h2 : a 2 + a 3 + a 4 = 14) :
  a 4 + a 5 + a 6 = 56 :=
sorry

end geometric_sequence_sum_l202_202701


namespace proof_l202_202377

-- Define the universal set U.
def U : Set ℕ := {x | x > 0 ∧ x < 9}

-- Define set M.
def M : Set ℕ := {1, 2, 3}

-- Define set N.
def N : Set ℕ := {3, 4, 5, 6}

-- The complement of M with respect to U.
def compl_U_M : Set ℕ := {x ∈ U | x ∉ M}

-- The intersection of complement of M and N.
def result : Set ℕ := compl_U_M ∩ N

-- The theorem to be proven.
theorem proof : result = {4, 5, 6} := by
  -- This is where the proof would go.
  sorry

end proof_l202_202377


namespace triangle_area_l202_202784

/-
A triangle with side lengths in the ratio 4:5:6 is inscribed in a circle of radius 5.
We need to prove that the area of the triangle is 250/9.
-/

theorem triangle_area (x : ℝ) (r : ℝ) (h_r : r = 5) (h_ratio : 6 * x = 2 * r) :
  (1 / 2) * (4 * x) * (5 * x) = 250 / 9 := by 
  -- Proof goes here.
  sorry

end triangle_area_l202_202784


namespace fraction_second_year_students_l202_202031

theorem fraction_second_year_students
  (total_students : ℕ)
  (third_year_students : ℕ)
  (second_year_students : ℕ)
  (h1 : third_year_students = total_students * 30 / 100)
  (h2 : second_year_students = total_students * 10 / 100) :
  (second_year_students : ℚ) / (total_students - third_year_students) = 1 / 7 := by
  sorry

end fraction_second_year_students_l202_202031


namespace cars_meet_cars_apart_l202_202223

section CarsProblem

variable (distance : ℕ) (speedA speedB : ℕ) (distanceToMeet distanceApart : ℕ)

def meetTime := distance / (speedA + speedB)
def apartTime1 := (distance - distanceApart) / (speedA + speedB)
def apartTime2 := (distance + distanceApart) / (speedA + speedB)

theorem cars_meet (h1: distance = 450) (h2: speedA = 115) (h3: speedB = 85):
  meetTime distance speedA speedB = 9 / 4 := by
  sorry

theorem cars_apart (h1: distance = 450) (h2: speedA = 115) (h3: speedB = 85) (h4: distanceApart = 50):
  apartTime1 distance speedA speedB distanceApart = 2 ∧ apartTime2 distance speedA speedB distanceApart = 5 / 2 := by
  sorry

end CarsProblem

end cars_meet_cars_apart_l202_202223


namespace tan_angle_addition_l202_202707

theorem tan_angle_addition (x : Real) (h1 : Real.tan x = 3) (h2 : Real.tan (Real.pi / 3) = Real.sqrt 3) : 
  Real.tan (x + Real.pi / 3) = (3 + Real.sqrt 3) / (1 - 3 * Real.sqrt 3) :=
by 
  sorry

end tan_angle_addition_l202_202707


namespace basis_of_R3_l202_202294

def e1 : ℝ × ℝ × ℝ := (1, 0, 0)
def e2 : ℝ × ℝ × ℝ := (0, 1, 0)
def e3 : ℝ × ℝ × ℝ := (0, 0, 1)

theorem basis_of_R3 :
  ∀ (u : ℝ × ℝ × ℝ), ∃ (α β γ : ℝ), u = α • e1 + β • e2 + γ • e3 ∧ 
  (∀ (a b c : ℝ), a • e1 + b • e2 + c • e3 = (0, 0, 0) → a = 0 ∧ b = 0 ∧ c = 0) :=
by
  sorry

end basis_of_R3_l202_202294


namespace relationship_among_abc_l202_202542

noncomputable def a : ℝ := Real.log (1/4) / Real.log 2
noncomputable def b : ℝ := 2.1^(1/3)
noncomputable def c : ℝ := (4/5)^2

theorem relationship_among_abc : a < c ∧ c < b :=
by
  -- Definitions
  have ha : a = Real.log (1/4) / Real.log 2 := rfl
  have hb : b = 2.1^(1/3) := rfl
  have hc : c = (4/5)^2 := rfl
  sorry

end relationship_among_abc_l202_202542


namespace min_value_x_plus_2y_l202_202381

theorem min_value_x_plus_2y (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + 2 * y = x * y) :
  x + 2 * y ≥ 8 :=
sorry

end min_value_x_plus_2y_l202_202381


namespace park_available_spaces_l202_202054

theorem park_available_spaces :
  let section_A_benches := 30
  let section_A_capacity_per_bench := 4
  let section_B_benches := 20
  let section_B_capacity_per_bench := 5
  let section_C_benches := 15
  let section_C_capacity_per_bench := 6
  let section_A_people := 50
  let section_B_people := 40
  let section_C_people := 45
  let section_A_total_capacity := section_A_benches * section_A_capacity_per_bench
  let section_B_total_capacity := section_B_benches * section_B_capacity_per_bench
  let section_C_total_capacity := section_C_benches * section_C_capacity_per_bench
  let section_A_available := section_A_total_capacity - section_A_people
  let section_B_available := section_B_total_capacity - section_B_people
  let section_C_available := section_C_total_capacity - section_C_people
  let total_available_spaces := section_A_available + section_B_available + section_C_available
  total_available_spaces = 175 := 
by
  let section_A_benches := 30
  let section_A_capacity_per_bench := 4
  let section_B_benches := 20
  let section_B_capacity_per_bench := 5
  let section_C_benches := 15
  let section_C_capacity_per_bench := 6
  let section_A_people := 50
  let section_B_people := 40
  let section_C_people := 45
  let section_A_total_capacity := section_A_benches * section_A_capacity_per_bench
  let section_B_total_capacity := section_B_benches * section_B_capacity_per_bench
  let section_C_total_capacity := section_C_benches * section_C_capacity_per_bench
  let section_A_available := section_A_total_capacity - section_A_people
  let section_B_available := section_B_total_capacity - section_B_people
  let section_C_available := section_C_total_capacity - section_C_people
  let total_available_spaces := section_A_available + section_B_available + section_C_available
  sorry

end park_available_spaces_l202_202054


namespace factorization_correct_l202_202686

theorem factorization_correct (x y : ℝ) : x^2 - 4 * y^2 = (x - 2 * y) * (x + 2 * y) :=
by sorry

end factorization_correct_l202_202686


namespace probability_of_exactly_three_heads_l202_202200

open Nat

noncomputable def binomial (n k : ℕ) : ℕ := (n.choose k)
noncomputable def probability_three_heads_in_eight_tosses : ℚ :=
  let total_outcomes := 2 ^ 8
  let favorable_outcomes := binomial 8 3
  favorable_outcomes / total_outcomes

theorem probability_of_exactly_three_heads :
  probability_three_heads_in_eight_tosses = 7 / 32 :=
by 
  sorry

end probability_of_exactly_three_heads_l202_202200


namespace ratio_female_to_male_l202_202257

variable {f m c : ℕ}

/-- 
  The following conditions are given:
  - The average age of female members is 35 years.
  - The average age of male members is 30 years.
  - The average age of children members is 10 years.
  - The average age of the entire membership is 25 years.
  - The number of children members is equal to the number of male members.
  We need to show that the ratio of female to male members is 1.
-/
theorem ratio_female_to_male (h1 : c = m)
  (h2 : 35 * f + 40 * m = 25 * (f + 2 * m)) :
  f = m :=
by sorry

end ratio_female_to_male_l202_202257


namespace no_solution_l202_202015

theorem no_solution (x : ℝ) : ¬ (3 * x - 2 < (x + 2)^2 ∧ (x + 2)^2 < 9 * x - 5) :=
by
  sorry

end no_solution_l202_202015


namespace rounds_on_sunday_l202_202643

theorem rounds_on_sunday (round_time total_time saturday_rounds : ℕ) (h1 : round_time = 30)
(h2 : total_time = 780) (h3 : saturday_rounds = 11) : 
(total_time - saturday_rounds * round_time) / round_time = 15 := by
  sorry

end rounds_on_sunday_l202_202643


namespace rahim_average_price_l202_202113

def books_shop1 : ℕ := 50
def cost_shop1 : ℕ := 1000
def books_shop2 : ℕ := 40
def cost_shop2 : ℕ := 800

def total_books : ℕ := books_shop1 + books_shop2
def total_cost : ℕ := cost_shop1 + cost_shop2
def average_price_per_book : ℕ := total_cost / total_books

theorem rahim_average_price :
  average_price_per_book = 20 := by
  sorry

end rahim_average_price_l202_202113


namespace arithmetic_sequence_sum_l202_202548

theorem arithmetic_sequence_sum (a : ℕ → ℕ) (S : ℕ → ℕ) (d : ℤ) :
  (∀ n, a n = a 1 + (n - 1) * d) → 
  (∀ n, S n = n * (a 1 + a n) / 2) → 
  (a 3 + 4 = a 2 + a 7) → 
  S 11 = 44 :=
by 
  sorry

end arithmetic_sequence_sum_l202_202548


namespace abs_diff_eq_l202_202841

-- Define the conditions
variables (x y : ℝ)
axiom h1 : x + y = 30
axiom h2 : x * y = 162

-- Define the problem to prove
theorem abs_diff_eq : |x - y| = 6 * Real.sqrt 7 :=
by sorry

end abs_diff_eq_l202_202841


namespace cost_of_math_book_l202_202316

-- The definitions based on the conditions from the problem
def total_books : ℕ := 90
def math_books : ℕ := 54
def history_books := total_books - math_books -- 36
def cost_history_book : ℝ := 5
def total_cost : ℝ := 396

-- The theorem we want to prove: the cost of each math book
theorem cost_of_math_book (M : ℝ) : (math_books * M + history_books * cost_history_book = total_cost) → M = 4 := 
by 
  sorry

end cost_of_math_book_l202_202316


namespace tan_alpha_value_l202_202544

open Real

theorem tan_alpha_value
  (α : ℝ)
  (h₀ : 0 < α)
  (h₁ : α < π / 2)
  (h₂ : cos (2 * α) = (2 * sqrt 5 / 5) * sin (α + π / 4)) :
  tan α = 1 / 3 :=
sorry

end tan_alpha_value_l202_202544


namespace solutionY_materialB_correct_l202_202703

open Real

-- Definitions and conditions from step a
def solutionX_materialA : ℝ := 0.20
def solutionX_materialB : ℝ := 0.80
def solutionY_materialA : ℝ := 0.30
def mixture_materialA : ℝ := 0.22
def solutionX_in_mixture : ℝ := 0.80
def solutionY_in_mixture : ℝ := 0.20

-- The conjecture to prove
theorem solutionY_materialB_correct (B_Y : ℝ) 
  (h1 : solutionX_materialA = 0.20)
  (h2 : solutionX_materialB = 0.80) 
  (h3 : solutionY_materialA = 0.30) 
  (h4 : mixture_materialA = 0.22)
  (h5 : solutionX_in_mixture = 0.80)
  (h6 : solutionY_in_mixture = 0.20) :
  B_Y = 1 - solutionY_materialA := by 
  sorry

end solutionY_materialB_correct_l202_202703


namespace lily_catches_up_mary_in_60_minutes_l202_202473

theorem lily_catches_up_mary_in_60_minutes
  (mary_speed : ℝ) (lily_speed : ℝ) (initial_distance : ℝ)
  (h_mary_speed : mary_speed = 4)
  (h_lily_speed : lily_speed = 6)
  (h_initial_distance : initial_distance = 2) :
  ∃ t : ℝ, t = 60 := by
  sorry

end lily_catches_up_mary_in_60_minutes_l202_202473


namespace net_change_in_price_l202_202084

theorem net_change_in_price (P : ℝ) : 
  ((P * 0.75) * 1.2 = P * 0.9) → 
  ((P * 0.9 - P) / P = -0.1) :=
by
  intro h
  sorry

end net_change_in_price_l202_202084


namespace trailing_zeros_50_factorial_l202_202141

def factorial_trailing_zeros (n : Nat) : Nat :=
  n / 5 + n / 25 -- Count the number of trailing zeros given the algorithm used in solution steps

theorem trailing_zeros_50_factorial : factorial_trailing_zeros 50 = 12 :=
by 
  -- Proof goes here
  sorry

end trailing_zeros_50_factorial_l202_202141


namespace bush_height_l202_202093

theorem bush_height (h : ℕ → ℕ) (h0 : h 5 = 81) (h1 : ∀ n, h (n + 1) = 3 * h n) :
  h 2 = 3 := 
sorry

end bush_height_l202_202093


namespace number_of_men_is_15_l202_202021

-- Define the conditions
def number_of_people : Prop :=
  ∃ (M W B : ℕ), M = 8 ∧ W = 8 ∧ B = 8 ∧ 8 * M = 120

-- Define the final statement to be proven
theorem number_of_men_is_15 (h: number_of_people) : ∃ M : ℕ, M = 15 :=
by
  obtain ⟨M, W, B, hM, hW, hB, htotal⟩ := h
  use M
  rw [hM] at htotal
  have hM15 : M = 15 := by linarith
  exact hM15

end number_of_men_is_15_l202_202021


namespace cube_surface_area_l202_202818

theorem cube_surface_area (a : ℝ) : 
    let edge_length := 3 * a
    let face_area := edge_length^2
    let total_surface_area := 6 * face_area
    total_surface_area = 54 * a^2 := 
by sorry

end cube_surface_area_l202_202818


namespace problem_1_problem_2_l202_202711

noncomputable def f (x k : ℝ) : ℝ := (2 * k * x) / (x * x + 6 * k)

theorem problem_1 (k m : ℝ) (hk : k > 0)
  (hsol : ∀ x, (f x k) > m ↔ x < -3 ∨ x > -2) :
  ∀ x, 5 * m * x ^ 2 + k * x + 3 > 0 ↔ -1 < x ∧ x < 3 / 2 :=
sorry

theorem problem_2 (k : ℝ) (hk : k > 0)
  (hsol : ∃ (x : ℝ), x > 3 ∧ (f x k) > 1) :
  k > 6 :=
sorry

end problem_1_problem_2_l202_202711


namespace cone_prism_ratio_is_pi_over_16_l202_202925

noncomputable def cone_prism_volume_ratio 
  (prism_length : ℝ) (prism_width : ℝ) (prism_height : ℝ) 
  (cone_base_radius : ℝ) (cone_height : ℝ)
  (h_length : prism_length = 3) (h_width : prism_width = 4) (h_height : prism_height = 5)
  (h_radius_cone : cone_base_radius = 1.5) (h_cone_height : cone_height = 5) : ℝ :=
  (1/3) * Real.pi * cone_base_radius^2 * cone_height / (prism_length * prism_width * prism_height)

theorem cone_prism_ratio_is_pi_over_16
  (prism_length : ℝ) (prism_width : ℝ) (prism_height : ℝ)
  (cone_base_radius : ℝ) (cone_height : ℝ)
  (h_length : prism_length = 3) (h_width : prism_width = 4) (h_height : prism_height = 5)
  (h_radius_cone : cone_base_radius = 1.5) (h_cone_height : cone_height = 5) :
  cone_prism_volume_ratio prism_length prism_width prism_height cone_base_radius cone_height
    h_length h_width h_height h_radius_cone h_cone_height = Real.pi / 16 := 
by
  sorry

end cone_prism_ratio_is_pi_over_16_l202_202925


namespace ratio_of_numbers_l202_202134

theorem ratio_of_numbers (A B D M : ℕ) 
  (h1 : A + B + D = M)
  (h2 : Nat.gcd A B = D)
  (h3 : Nat.lcm A B = M)
  (h4 : A ≥ B) : A / B = 3 / 2 :=
by
  sorry

end ratio_of_numbers_l202_202134


namespace keystone_arch_larger_angle_l202_202802

def isosceles_trapezoid_larger_angle (n : ℕ) : Prop :=
  n = 10 → ∃ (x : ℝ), x = 99

theorem keystone_arch_larger_angle :
  isosceles_trapezoid_larger_angle 10 :=
by
  sorry

end keystone_arch_larger_angle_l202_202802


namespace right_triangle_shorter_leg_l202_202904
-- Import all necessary libraries

-- Define the problem
theorem right_triangle_shorter_leg (a b c : ℕ) (h1 : a < b) (h2 : b < c) (h3 : c = 65) (h4 : a^2 + b^2 = c^2) :
  a = 25 :=
sorry

end right_triangle_shorter_leg_l202_202904


namespace students_at_end_of_year_l202_202328

def n_start := 10
def n_left := 4
def n_new := 42

theorem students_at_end_of_year : n_start - n_left + n_new = 48 := by
  sorry

end students_at_end_of_year_l202_202328


namespace lemonade_lemons_per_glass_l202_202856

def number_of_glasses : ℕ := 9
def total_lemons : ℕ := 18
def lemons_per_glass : ℕ := 2

theorem lemonade_lemons_per_glass :
  total_lemons / number_of_glasses = lemons_per_glass :=
by
  sorry

end lemonade_lemons_per_glass_l202_202856


namespace friend_wants_to_take_5_marbles_l202_202801

theorem friend_wants_to_take_5_marbles
  (total_marbles : ℝ)
  (clear_marbles : ℝ)
  (black_marbles : ℝ)
  (other_marbles : ℝ)
  (friend_marbles : ℝ)
  (h1 : clear_marbles = 0.4 * total_marbles)
  (h2 : black_marbles = 0.2 * total_marbles)
  (h3 : other_marbles = total_marbles - clear_marbles - black_marbles)
  (h4 : friend_marbles = 2)
  (friend_total_marbles : ℝ)
  (h5 : friend_marbles = 0.4 * friend_total_marbles) :
  friend_total_marbles = 5 := by
  sorry

end friend_wants_to_take_5_marbles_l202_202801


namespace arithmetic_sequence_inequality_l202_202403

noncomputable def arithmetic_seq (a d : ℕ) (n : ℕ) : ℕ := a + (n - 1) * d

noncomputable def sum_arithmetic_seq (a d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

theorem arithmetic_sequence_inequality
  (a d : ℕ)
  (i j k l : ℕ)
  (hi : i ≤ j)
  (hj : j ≤ k)
  (hk : k ≤ l)
  (hij: i + l = j + k)
  : (arithmetic_seq a d i) * (arithmetic_seq a d l) ≤ (arithmetic_seq a d j) * (arithmetic_seq a d k) :=
sorry

end arithmetic_sequence_inequality_l202_202403


namespace total_distance_proof_l202_202117

-- Define the conditions
def first_half_time := 20
def second_half_time := 30
def average_time_per_kilometer := 5

-- Calculate the total time
def total_time := first_half_time + second_half_time

-- State the proof problem: prove that the total distance is 10 kilometers
theorem total_distance_proof : 
  (total_time / average_time_per_kilometer) = 10 :=
  by sorry

end total_distance_proof_l202_202117


namespace pi_div_two_minus_alpha_in_third_quadrant_l202_202469

theorem pi_div_two_minus_alpha_in_third_quadrant (α : ℝ) (k : ℤ) (h : ∃ k : ℤ, (π + 2 * k * π < α) ∧ (α < 3 * π / 2 + 2 * k * π)) : 
  ∃ k : ℤ, (π + 2 * k * π < (π / 2 - α)) ∧ ((π / 2 - α) < 3 * π / 2 + 2 * k * π) :=
sorry

end pi_div_two_minus_alpha_in_third_quadrant_l202_202469


namespace total_men_employed_l202_202623

/--
A work which could be finished in 11 days was finished 3 days earlier 
after 10 more men joined. Prove that the total number of men employed 
to finish the work earlier is 37.
-/
theorem total_men_employed (x : ℕ) (h1 : 11 * x = 8 * (x + 10)) : x = 27 ∧ 27 + 10 = 37 := by
  sorry

end total_men_employed_l202_202623


namespace rahul_matches_played_l202_202835

-- Define the conditions of the problem
variable (m : ℕ) -- number of matches Rahul has played so far
variable (runs_before : ℕ := 51 * m) -- total runs before today's match
variable (runs_today : ℕ := 69) -- runs scored today
variable (new_average : ℕ := 54) -- new batting average after today's match

-- The equation derived from the conditions
def batting_average_equation : Prop :=
  new_average * (m + 1) = runs_before + runs_today

-- The problem: prove that m = 5 given the conditions
theorem rahul_matches_played (h : batting_average_equation m) : m = 5 :=
  sorry

end rahul_matches_played_l202_202835


namespace even_suff_not_nec_l202_202944

theorem even_suff_not_nec (f g : ℝ → ℝ) 
  (hf_even : ∀ x : ℝ, f (-x) = f x)
  (hg_even : ∀ x : ℝ, g (-x) = g x) :
  ∀ x : ℝ, (f x + g x) = ((f + g) x) ∧ (∀ h : ℝ → ℝ, ∃ f g : ℝ → ℝ, h = f + g ∧ ∀ x : ℝ, (h (-x) = h x) ↔ (f (-x) = f x ∧ g (-x) = g x)) :=
by 
  sorry

end even_suff_not_nec_l202_202944


namespace sum_a_b_l202_202187

variable {a b : ℝ}

theorem sum_a_b (hab : a * b = 5) (hrecip : 1 / (a^2) + 1 / (b^2) = 0.6) : a + b = 5 ∨ a + b = -5 :=
sorry

end sum_a_b_l202_202187


namespace at_least_one_inequality_holds_l202_202132

theorem at_least_one_inequality_holds (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x + y > 2) : 
  (1 + x) / y < 2 ∨ (1 + y) / x < 2 :=
by
  sorry

end at_least_one_inequality_holds_l202_202132


namespace pen_cost_l202_202840

theorem pen_cost
  (p q : ℕ)
  (h1 : 6 * p + 5 * q = 380)
  (h2 : 3 * p + 8 * q = 298) :
  p = 47 :=
sorry

end pen_cost_l202_202840


namespace student_chose_number_l202_202970

theorem student_chose_number (x : ℤ) (h : 2 * x - 148 = 110) : x = 129 := 
by
  sorry

end student_chose_number_l202_202970


namespace theater_total_revenue_l202_202098

theorem theater_total_revenue :
  let seats := 400
  let capacity := 0.8
  let ticket_price := 30
  let days := 3
  seats * capacity * ticket_price * days = 28800 := by
  sorry

end theater_total_revenue_l202_202098


namespace trajectory_equation_minimum_AB_l202_202985

/-- Let a moving circle \( C \) passes through the point \( F(0, 1) \).
    The center of the circle \( C \), denoted as \( (x, y) \), is above the \( x \)-axis and the
    distance from \( (x, y) \) to \( F \) is greater than its distance to the \( x \)-axis by 1.
    We aim to prove that the trajectory of the center is \( x^2 = 4y \). -/
theorem trajectory_equation {x y : ℝ} (h : y > 0) (hCF : Real.sqrt (x^2 + (y - 1)^2) - y = 1) : 
  x^2 = 4 * y :=
sorry

/-- Suppose \( A \) and \( B \) are two distinct points on the curve \( x^2 = 4y \). 
    The tangents at \( A \) and \( B \) intersect at \( P \), and \( AP \perp BP \). 
    Then the minimum value of \( |AB| \) is 4. -/
theorem minimum_AB {x₁ x₂ : ℝ} 
  (h₁ : y₁ = (x₁^2) / 4) (h₂ : y₂ = (x₂^2) / 4)
  (h_perp : x₁ * x₂ = -4) : 
  ∃ (d : ℝ), d ≥ 0 ∧ d = 4 :=
sorry

end trajectory_equation_minimum_AB_l202_202985


namespace complex_mul_im_unit_l202_202250

theorem complex_mul_im_unit (i : ℂ) (h : i^2 = -1) : i * (1 - i) = 1 + i := by
  sorry

end complex_mul_im_unit_l202_202250


namespace soccer_most_students_l202_202624

def sports := ["hockey", "basketball", "soccer", "volleyball", "badminton"]
def num_students (sport : String) : Nat :=
  match sport with
  | "hockey" => 30
  | "basketball" => 35
  | "soccer" => 50
  | "volleyball" => 20
  | "badminton" => 25
  | _ => 0

theorem soccer_most_students : ∀ sport ∈ sports, num_students "soccer" ≥ num_students sport := by
  sorry

end soccer_most_students_l202_202624


namespace brendan_cuts_84_yards_in_week_with_lawnmower_l202_202073

-- Brendan cuts 8 yards per day
def yards_per_day : ℕ := 8

-- The lawnmower increases his efficiency by fifty percent
def efficiency_increase (yards : ℕ) : ℕ :=
  yards + (yards / 2)

-- Calculate total yards cut in 7 days with the lawnmower
def total_yards_in_week (days : ℕ) (daily_yards : ℕ) : ℕ :=
  days * daily_yards

-- Prove the total yards cut in 7 days with the lawnmower is 84
theorem brendan_cuts_84_yards_in_week_with_lawnmower :
  total_yards_in_week 7 (efficiency_increase yards_per_day) = 84 :=
by
  sorry

end brendan_cuts_84_yards_in_week_with_lawnmower_l202_202073


namespace container_capacity_in_liters_l202_202915

-- Defining the conditions
def portions : Nat := 10
def portion_size_ml : Nat := 200

-- Statement to prove
theorem container_capacity_in_liters : (portions * portion_size_ml / 1000 = 2) :=
by 
  sorry

end container_capacity_in_liters_l202_202915


namespace algebraic_expression_evaluation_l202_202326

theorem algebraic_expression_evaluation
  (x y p q : ℝ)
  (h1 : x + y = 0)
  (h2 : p * q = 1) : (x + y) - 2 * (p * q) = -2 :=
by
  sorry

end algebraic_expression_evaluation_l202_202326


namespace spending_difference_l202_202115

def chocolate_price : ℝ := 7
def candy_bar_price : ℝ := 2
def discount_rate : ℝ := 0.15
def sales_tax_rate : ℝ := 0.08
def gum_price : ℝ := 3

def discounted_chocolate_price : ℝ := chocolate_price * (1 - discount_rate)
def total_before_tax : ℝ := candy_bar_price + gum_price
def tax_amount : ℝ := total_before_tax * sales_tax_rate
def total_after_tax : ℝ := total_before_tax + tax_amount

theorem spending_difference : 
  discounted_chocolate_price - candy_bar_price = 3.95 :=
by 
  -- Apply the necessary calculations
  have discount_chocolate : ℝ := discounted_chocolate_price
  have candy_bar : ℝ := candy_bar_price
  calc
    discounted_chocolate_price - candy_bar_price = _ := sorry

end spending_difference_l202_202115


namespace digits_of_2048_in_base_9_l202_202010

def digits_base9 (n : ℕ) : ℕ :=
if n < 9 then 1 else 1 + digits_base9 (n / 9)

theorem digits_of_2048_in_base_9 : digits_base9 2048 = 4 :=
by sorry

end digits_of_2048_in_base_9_l202_202010


namespace tangent_line_at_x_neg1_l202_202709

-- Definition of the curve.
def curve (x : ℝ) : ℝ := 2*x - x^3

-- Definition of the point of tangency.
def point_of_tangency_x : ℝ := -1

-- Definition of the point of tangency.
def point_of_tangency_y : ℝ := curve point_of_tangency_x

-- Definition of the derivative of the curve.
def derivative (x : ℝ) : ℝ := -3*x^2 + 2

-- Slope of the tangent at the point of tangency.
def slope_at_tangency : ℝ := derivative point_of_tangency_x

-- Equation of the tangent line function.
def tangent_line (x y : ℝ) := x + y + 2 = 0

theorem tangent_line_at_x_neg1 :
  tangent_line point_of_tangency_x point_of_tangency_y :=
by
  -- Here we will perform the proof, which is omitted for the purposes of this task.
  sorry

end tangent_line_at_x_neg1_l202_202709


namespace abc_eq_bc_l202_202787

theorem abc_eq_bc (a b c : ℝ) (hpos : 0 < a ∧ 0 < b ∧ 0 < c) 
(h : 4 * a * b * c * (a + b + c) = (a + b)^2 * (a + c)^2) :
  a * (a + b + c) = b * c :=
by 
  sorry

end abc_eq_bc_l202_202787


namespace quadratic_eq_coeffs_l202_202523

theorem quadratic_eq_coeffs (x : ℝ) : 
  ∃ a b c : ℝ, 3 * x^2 + 1 - 6 * x = a * x^2 + b * x + c ∧ a = 3 ∧ b = -6 ∧ c = 1 :=
by sorry

end quadratic_eq_coeffs_l202_202523


namespace part1_part2_part3a_part3b_l202_202146

open Real

variable (θ : ℝ) (m : ℝ)

-- Conditions
axiom theta_domain : 0 < θ ∧ θ < 2 * π
axiom quadratic_eq : ∀ x : ℝ, 2 * x^2 - (sqrt 3 + 1) * x + m = 0
axiom roots_eq_theta : ∀ x : ℝ, (x = sin θ ∨ x = cos θ)

-- Proof statements
theorem part1 : 1 - cos θ ≠ 0 → 1 - tan θ ≠ 0 → 
  (sin θ / (1 - cos θ) + cos θ / (1 - tan θ)) = (3 + 5 * sqrt 3) / 4 := sorry

theorem part2 : sin θ * cos θ = m / 2 → m = sqrt 3 / 4 := sorry

theorem part3a : sin θ = sqrt 3 / 2 ∧ cos θ = 1 / 2 → θ = π / 3 := sorry

theorem part3b : sin θ = 1 / 2 ∧ cos θ = sqrt 3 / 2 → θ = π / 6 := sorry

end part1_part2_part3a_part3b_l202_202146


namespace find_b_l202_202505

theorem find_b (a b : ℝ) (h₁ : ∀ x y, y = 0.75 * x + 1 → (4, b) = (x, y))
                (h₂ : k = 0.75) : b = 4 :=
by sorry

end find_b_l202_202505


namespace moles_of_CO2_formed_l202_202602

-- Definitions based on the conditions provided
def moles_HNO3 := 2
def moles_NaHCO3 := 2
def balanced_eq (HNO3 NaHCO3 NaNO3 CO2 H2O : ℕ) : Prop :=
  HNO3 = NaHCO3 ∧ NaNO3 = NaHCO3 ∧ CO2 = NaHCO3 ∧ H2O = NaHCO3

-- Lean Proposition: Prove that 2 moles of CO2 are formed
theorem moles_of_CO2_formed :
  balanced_eq moles_HNO3 moles_NaHCO3 moles_HNO3 moles_HNO3 moles_HNO3 →
  ∃ CO2, CO2 = 2 :=
by
  sorry

end moles_of_CO2_formed_l202_202602


namespace percentage_of_first_to_second_l202_202429

theorem percentage_of_first_to_second (X : ℝ) (first second : ℝ) :
  first = 0.06 * X →
  second = 0.30 * X →
  (first / second) * 100 = 20 :=
by
  intros h1 h2
  sorry

end percentage_of_first_to_second_l202_202429


namespace expected_total_rainfall_10_days_l202_202805

theorem expected_total_rainfall_10_days :
  let P_sun := 0.5
  let P_rain3 := 0.3
  let P_rain6 := 0.2
  let daily_rain := (P_sun * 0) + (P_rain3 * 3) + (P_rain6 * 6)
  daily_rain * 10 = 21 :=
by
  sorry

end expected_total_rainfall_10_days_l202_202805


namespace range_of_function_l202_202276

-- Given conditions 
def independent_variable_range (x : ℝ) : Prop := x ≥ 2

-- Proof statement (no proof only statement with "sorry")
theorem range_of_function (x : ℝ) (y : ℝ) (h : y = Real.sqrt (x - 2)) : independent_variable_range x :=
by sorry

end range_of_function_l202_202276


namespace sale_percent_saved_l202_202266

noncomputable def percent_saved (P : ℝ) : ℝ := (3 * P) / (6 * P) * 100

theorem sale_percent_saved :
  ∀ (P : ℝ), P > 0 → percent_saved P = 50 :=
by
  intros P hP
  unfold percent_saved
  have hP_nonzero : 6 * P ≠ 0 := by linarith
  field_simp [hP_nonzero]
  norm_num
  sorry

end sale_percent_saved_l202_202266


namespace johns_disposable_income_increase_l202_202512

noncomputable def percentage_increase_of_johns_disposable_income
  (weekly_income_before : ℝ) (weekly_income_after : ℝ)
  (tax_rate_before : ℝ) (tax_rate_after : ℝ)
  (monthly_expense : ℝ) : ℝ :=
  let disposable_income_before := (weekly_income_before * (1 - tax_rate_before) * 4 - monthly_expense)
  let disposable_income_after := (weekly_income_after * (1 - tax_rate_after) * 4 - monthly_expense)
  (disposable_income_after - disposable_income_before) / disposable_income_before * 100

theorem johns_disposable_income_increase :
  percentage_increase_of_johns_disposable_income 60 70 0.15 0.18 100 = 24.62 :=
  by
  sorry

end johns_disposable_income_increase_l202_202512


namespace diamond_associative_l202_202189

def diamond (a b : ℕ) : ℕ := a ^ (b / a)

theorem diamond_associative (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c):
  diamond a (diamond b c) = diamond (diamond a b) c :=
sorry

end diamond_associative_l202_202189


namespace min_value_of_quadratic_l202_202740

theorem min_value_of_quadratic (m : ℝ) (x : ℝ) (hx1 : 3 ≤ x) (hx2 : x < 4) (h : x^2 - 4 * x ≥ m) : 
  m ≤ -3 :=
sorry

end min_value_of_quadratic_l202_202740


namespace ratio_a_b_l202_202068

-- Definitions of the arithmetic sequences
open Classical

noncomputable def sequence1 (a y b : ℕ) : ℕ → ℕ
| 0 => a
| 1 => y
| 2 => b
| 3 => 14
| _ => 0 -- only the first four terms are given for sequence1

noncomputable def sequence2 (x y : ℕ) : ℕ → ℕ
| 0 => 2
| 1 => x
| 2 => 6
| 3 => y
| _ => 0 -- only the first four terms are given for sequence2

theorem ratio_a_b (a y b x : ℕ) (h1 : sequence1 a y b 0 = a) (h2 : sequence1 a y b 1 = y) 
  (h3 : sequence1 a y b 2 = b) (h4 : sequence1 a y b 3 = 14)
  (h5 : sequence2 x y 0 = 2) (h6 : sequence2 x y 1 = x) 
  (h7 : sequence2 x y 2 = 6) (h8 : sequence2 x y 3 = y) :
  (a:ℚ) / b = 2 / 3 :=
sorry

end ratio_a_b_l202_202068
