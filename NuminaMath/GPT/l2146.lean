import Mathlib

namespace NUMINAMATH_GPT_largest_three_digit_divisible_by_13_l2146_214633

theorem largest_three_digit_divisible_by_13 :
  ∃ n, (n ≤ 999 ∧ n ≥ 100 ∧ 13 ∣ n) ∧ (∀ m, m ≤ 999 ∧ m ≥ 100 ∧ 13 ∣ m → m ≤ 987) :=
by
  sorry

end NUMINAMATH_GPT_largest_three_digit_divisible_by_13_l2146_214633


namespace NUMINAMATH_GPT_optimal_selling_price_maximizes_profit_l2146_214680

/-- The purchase price of a certain product is 40 yuan. -/
def cost_price : ℝ := 40

/-- At a selling price of 50 yuan, 50 units can be sold. -/
def initial_selling_price : ℝ := 50
def initial_quantity_sold : ℝ := 50

/-- If the selling price increases by 1 yuan, the sales volume decreases by 1 unit. -/
def price_increase_effect (x : ℝ) : ℝ := initial_selling_price + x
def quantity_decrease_effect (x : ℝ) : ℝ := initial_quantity_sold - x

/-- The revenue function. -/
def revenue (x : ℝ) : ℝ := (price_increase_effect x) * (quantity_decrease_effect x)

/-- The cost function. -/
def cost (x : ℝ) : ℝ := cost_price * (quantity_decrease_effect x)

/-- The profit function. -/
def profit (x : ℝ) : ℝ := revenue x - cost x

/-- The proof that the optimal selling price to maximize profit is 70 yuan. -/
theorem optimal_selling_price_maximizes_profit : price_increase_effect 20 = 70 :=
by
  sorry

end NUMINAMATH_GPT_optimal_selling_price_maximizes_profit_l2146_214680


namespace NUMINAMATH_GPT_first_bag_brown_mms_l2146_214674

theorem first_bag_brown_mms :
  ∀ (x : ℕ),
  (12 + 8 + 8 + 3 + x) / 5 = 8 → x = 9 :=
by
  intros x h
  sorry

end NUMINAMATH_GPT_first_bag_brown_mms_l2146_214674


namespace NUMINAMATH_GPT_electric_guitar_count_l2146_214639

theorem electric_guitar_count (E A : ℤ) (h1 : E + A = 9) (h2 : 479 * E + 339 * A = 3611) (hE_nonneg : E ≥ 0) (hA_nonneg : A ≥ 0) : E = 4 :=
by
  sorry

end NUMINAMATH_GPT_electric_guitar_count_l2146_214639


namespace NUMINAMATH_GPT_intersection_of_sets_l2146_214650

def setA (x : ℝ) : Prop := x^2 - 4 * x - 5 > 0

def setB (x : ℝ) : Prop := 4 - x^2 > 0

theorem intersection_of_sets :
  {x : ℝ | setA x} ∩ {x : ℝ | setB x} = {x : ℝ | -2 < x ∧ x < -1} :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_sets_l2146_214650


namespace NUMINAMATH_GPT_tickets_used_63_l2146_214612

def rides_ferris_wheel : ℕ := 5
def rides_bumper_cars : ℕ := 4
def cost_per_ride : ℕ := 7
def total_rides : ℕ := rides_ferris_wheel + rides_bumper_cars
def total_tickets_used : ℕ := total_rides * cost_per_ride

theorem tickets_used_63 : total_tickets_used = 63 := by
  unfold total_tickets_used
  unfold total_rides
  unfold rides_ferris_wheel
  unfold rides_bumper_cars
  unfold cost_per_ride
  -- proof goes here
  sorry

end NUMINAMATH_GPT_tickets_used_63_l2146_214612


namespace NUMINAMATH_GPT_paving_stone_size_l2146_214697

theorem paving_stone_size (length_courtyard width_courtyard : ℕ) (num_paving_stones : ℕ) (area_courtyard : ℕ) (s : ℕ)
  (h₁ : length_courtyard = 30) 
  (h₂ : width_courtyard = 18)
  (h₃ : num_paving_stones = 135)
  (h₄ : area_courtyard = length_courtyard * width_courtyard)
  (h₅ : area_courtyard = num_paving_stones * s * s) :
  s = 2 := 
by
  sorry

end NUMINAMATH_GPT_paving_stone_size_l2146_214697


namespace NUMINAMATH_GPT_max_value_of_y_l2146_214676

open Real

theorem max_value_of_y (x : ℝ) (h1 : 0 < x) (h2 : x < sqrt 3) : x * sqrt (3 - x^2) ≤ 9 / 4 :=
sorry

end NUMINAMATH_GPT_max_value_of_y_l2146_214676


namespace NUMINAMATH_GPT_twenty_percent_of_x_l2146_214696

noncomputable def x := 1800 / 1.2

theorem twenty_percent_of_x (h : 1.2 * x = 1800) : 0.2 * x = 300 :=
by
  -- The proof would go here, but we'll replace it with sorry.
  sorry

end NUMINAMATH_GPT_twenty_percent_of_x_l2146_214696


namespace NUMINAMATH_GPT_proof_smallest_integer_proof_sum_of_integers_l2146_214670

def smallest_integer (n : Int) : Prop :=
  ∃ (a b c d e : Int), a = n ∧ b = n + 2 ∧ c = n + 4 ∧ d = n + 6 ∧ e = n + 8 ∧ a + e = 204 ∧ n = 98

def sum_of_integers (n : Int) : Prop :=
  ∃ (a b c d e : Int), a = n ∧ b = n + 2 ∧ c = n + 4 ∧ d = n + 6 ∧ e = n + 8 ∧ a + e = 204 ∧ a + b + c + d + e = 510

theorem proof_smallest_integer : ∃ n : Int, smallest_integer n := by
  sorry

theorem proof_sum_of_integers : ∃ n : Int, sum_of_integers n := by
  sorry

end NUMINAMATH_GPT_proof_smallest_integer_proof_sum_of_integers_l2146_214670


namespace NUMINAMATH_GPT_distance_walked_is_18_miles_l2146_214673

-- Defining the variables for speed, time, and distance
variables (x t d : ℕ)

-- Declaring the conditions given in the problem
def walked_distance_at_usual_rate : Prop :=
  d = x * t

def walked_distance_at_increased_rate : Prop :=
  d = (x + 1) * (3 * t / 4)

def walked_distance_at_decreased_rate : Prop :=
  d = (x - 1) * (t + 3)

-- The proof problem statement to show the distance walked is 18 miles
theorem distance_walked_is_18_miles
  (hx : walked_distance_at_usual_rate x t d)
  (hz : walked_distance_at_increased_rate x t d)
  (hy : walked_distance_at_decreased_rate x t d) :
  d = 18 := by
  sorry

end NUMINAMATH_GPT_distance_walked_is_18_miles_l2146_214673


namespace NUMINAMATH_GPT_athlete_heartbeats_l2146_214640

def heart_beats_per_minute : ℕ := 120
def running_pace_minutes_per_mile : ℕ := 6
def race_distance_miles : ℕ := 30
def total_heartbeats : ℕ := 21600

theorem athlete_heartbeats :
  (running_pace_minutes_per_mile * race_distance_miles * heart_beats_per_minute) = total_heartbeats :=
by
  sorry

end NUMINAMATH_GPT_athlete_heartbeats_l2146_214640


namespace NUMINAMATH_GPT_probability_of_drawing_white_ball_l2146_214641

theorem probability_of_drawing_white_ball 
  (total_balls : ℕ) (white_balls : ℕ) 
  (h_total : total_balls = 9) (h_white : white_balls = 4) : 
  (white_balls : ℚ) / total_balls = 4 / 9 := 
by 
  sorry

end NUMINAMATH_GPT_probability_of_drawing_white_ball_l2146_214641


namespace NUMINAMATH_GPT_pentagon_area_sol_l2146_214672

theorem pentagon_area_sol (a b : ℤ) (h1 : 0 < b) (h2 : b < a) (h3 : a * (3 * b + a) = 792) : a + b = 45 :=
sorry

end NUMINAMATH_GPT_pentagon_area_sol_l2146_214672


namespace NUMINAMATH_GPT_percentage_of_men_l2146_214644

theorem percentage_of_men (E M W : ℝ) 
  (h1 : M + W = E)
  (h2 : 0.5 * M + 0.1666666666666669 * W = 0.4 * E)
  (h3 : W = E - M) : 
  (M / E = 0.70) :=
by
  sorry

end NUMINAMATH_GPT_percentage_of_men_l2146_214644


namespace NUMINAMATH_GPT_find_two_digit_number_l2146_214630

def tens_digit (n: ℕ) := n / 10
def unit_digit (n: ℕ) := n % 10
def is_required_number (n: ℕ) : Prop :=
  tens_digit n + 2 = unit_digit n ∧ n < 30 ∧ 10 ≤ n

theorem find_two_digit_number (n : ℕ) :
  is_required_number n → n = 13 ∨ n = 24 :=
by
  -- Proof placeholder
  sorry

end NUMINAMATH_GPT_find_two_digit_number_l2146_214630


namespace NUMINAMATH_GPT_youngest_child_age_l2146_214667

theorem youngest_child_age (x : ℕ) 
  (h : x + (x + 4) + (x + 8) + (x + 12) + (x + 16) + (x + 20) + (x + 24) = 112) : 
  x = 4 := by
  sorry

end NUMINAMATH_GPT_youngest_child_age_l2146_214667


namespace NUMINAMATH_GPT_regression_shows_positive_correlation_l2146_214621

-- Define the regression equations as constants
def reg_eq_A (x : ℝ) : ℝ := -2.1 * x + 1.8
def reg_eq_B (x : ℝ) : ℝ := 1.2 * x + 1.5
def reg_eq_C (x : ℝ) : ℝ := -0.5 * x + 2.1
def reg_eq_D (x : ℝ) : ℝ := -0.6 * x + 3

-- Define the condition for positive correlation
def positive_correlation (b : ℝ) : Prop := b > 0

-- The theorem statement to prove
theorem regression_shows_positive_correlation : 
  positive_correlation 1.2 := 
by
  sorry

end NUMINAMATH_GPT_regression_shows_positive_correlation_l2146_214621


namespace NUMINAMATH_GPT_paths_from_A_to_B_l2146_214618

def path_count_A_to_B : Nat :=
  let red_to_blue_ways := [2, 3]  -- 2 ways to first blue, 3 ways to second blue
  let blue_to_green_ways_first := 4 * 2  -- Each of the 2 green arrows from first blue, 4 ways each
  let blue_to_green_ways_second := 5 * 2 -- Each of the 2 green arrows from second blue, 5 ways each
  let green_to_B_ways_first := 2 * blue_to_green_ways_first  -- Each of the first green, 2 ways each
  let green_to_B_ways_second := 3 * blue_to_green_ways_second  -- Each of the second green, 3 ways each
  green_to_B_ways_first + green_to_B_ways_second  -- Total paths from green arrows to B

theorem paths_from_A_to_B : path_count_A_to_B = 46 := by
  sorry

end NUMINAMATH_GPT_paths_from_A_to_B_l2146_214618


namespace NUMINAMATH_GPT_hcf_462_5_1_l2146_214608

theorem hcf_462_5_1 (a b c : ℕ) (h₁ : a = 462) (h₂ : b = 5) (h₃ : c = 2310) (h₄ : Nat.lcm a b = c) : Nat.gcd a b = 1 := by
  sorry

end NUMINAMATH_GPT_hcf_462_5_1_l2146_214608


namespace NUMINAMATH_GPT_solve_inequality_l2146_214619

theorem solve_inequality (x : Real) : 
  x^2 - 48 * x + 576 ≤ 16 ↔ 20 ≤ x ∧ x ≤ 28 :=
by
  sorry

end NUMINAMATH_GPT_solve_inequality_l2146_214619


namespace NUMINAMATH_GPT_inequality_proof_l2146_214629

variables {a1 a2 a3 b1 b2 b3 : ℝ}

theorem inequality_proof (h1 : 0 < a1) (h2 : 0 < a2) (h3 : 0 < a3) 
                         (h4 : 0 < b1) (h5 : 0 < b2) (h6 : 0 < b3):
  (a1 * b2 + a2 * b1 + a2 * b3 + a3 * b2 + a3 * b1 + a1 * b3)^2 
  ≥ 4 * (a1 * a2 + a2 * a3 + a3 * a1) * (b1 * b2 + b2 * b3 + b3 * b1) := 
sorry

end NUMINAMATH_GPT_inequality_proof_l2146_214629


namespace NUMINAMATH_GPT_sphere_radius_vol_eq_area_l2146_214606

noncomputable def volume (r : ℝ) : ℝ := (4/3) * Real.pi * r ^ 3
noncomputable def surface_area (r : ℝ) : ℝ := 4 * Real.pi * r ^ 2

theorem sphere_radius_vol_eq_area (r : ℝ) :
  volume r = surface_area r → r = 3 :=
by
  sorry

end NUMINAMATH_GPT_sphere_radius_vol_eq_area_l2146_214606


namespace NUMINAMATH_GPT_go_games_l2146_214610

theorem go_games (total_go_balls : ℕ) (go_balls_per_game : ℕ) (h_total : total_go_balls = 901) (h_game : go_balls_per_game = 53) : (total_go_balls / go_balls_per_game) = 17 := by
  sorry

end NUMINAMATH_GPT_go_games_l2146_214610


namespace NUMINAMATH_GPT_sequence_arithmetic_l2146_214637

-- Define the sequence and sum conditions
variable (a : ℕ → ℝ) (S : ℕ → ℝ) (p : ℝ)

-- We are given that the sum of the first n terms is Sn = n * p * a_n
axiom sum_condition (n : ℕ) (hpos : n > 0) : S n = n * p * a n

-- Also, given that a_1 ≠ a_2
axiom a1_ne_a2 : a 1 ≠ a 2

-- Define what we need to prove
theorem sequence_arithmetic (n : ℕ) (hn : n ≥ 2) :
  ∃ (a2 : ℝ), p = 1/2 ∧ a n = (n-1) * a2 :=
by
  sorry

end NUMINAMATH_GPT_sequence_arithmetic_l2146_214637


namespace NUMINAMATH_GPT_intersection_S_T_l2146_214628

def setS (x : ℝ) : Prop := (x - 1) * (x - 3) ≥ 0
def setT (x : ℝ) : Prop := x > 0

theorem intersection_S_T : {x : ℝ | setS x} ∩ {x : ℝ | setT x} = {x : ℝ | (0 < x ∧ x ≤ 1) ∨ (3 ≤ x)} := 
sorry

end NUMINAMATH_GPT_intersection_S_T_l2146_214628


namespace NUMINAMATH_GPT_ellipse_condition_l2146_214607

theorem ellipse_condition (x y m : ℝ) :
  (1 < m ∧ m < 3) → (∀ x y, (∃ k1 k2: ℝ, k1 > 0 ∧ k2 > 0 ∧ k1 ≠ k2 ∧ (x^2 / k1 + y^2 / k2 = 1 ↔ (1 < m ∧ m < 3 ∧ m ≠ 2)))) :=
by 
  sorry

end NUMINAMATH_GPT_ellipse_condition_l2146_214607


namespace NUMINAMATH_GPT_trig_identity_solution_l2146_214624

theorem trig_identity_solution
  (α : ℝ) (β : ℝ)
  (h1 : Real.tan α = 1 / 2)
  (h2 : Real.tan β = -1 / 3) :
  (3 * Real.sin α * Real.cos β - Real.sin β * Real.cos α) / (Real.cos α * Real.cos β + 2 * Real.sin α * Real.sin β) = 11 / 4 :=
by
  sorry

end NUMINAMATH_GPT_trig_identity_solution_l2146_214624


namespace NUMINAMATH_GPT_find_first_number_l2146_214663

/-- The lcm of two numbers is 2310 and hcf (gcd) is 26. One of the numbers is 286. What is the other number? --/
theorem find_first_number (A : ℕ) 
  (h_lcm : Nat.lcm A 286 = 2310) 
  (h_gcd : Nat.gcd A 286 = 26) : 
  A = 210 := 
by
  sorry

end NUMINAMATH_GPT_find_first_number_l2146_214663


namespace NUMINAMATH_GPT_hyperbola_equation_through_point_l2146_214666

theorem hyperbola_equation_through_point
  (hyp_passes_through : ∀ (x y : ℝ), (x, y) = (1, 1) → ∃ (a b t : ℝ), (y^2 / a^2 - x^2 / b^2 = t))
  (asymptotes : ∀ (x y : ℝ), (y / x = Real.sqrt 2 ∨ y / x = -Real.sqrt 2) → ∃ (a b t : ℝ), (a = b * Real.sqrt 2)) :
  ∃ (a b t : ℝ), (2 * (1:ℝ)^2 - (1:ℝ)^2 = 1) :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_equation_through_point_l2146_214666


namespace NUMINAMATH_GPT_johan_painted_green_fraction_l2146_214681

theorem johan_painted_green_fraction :
  let total_rooms := 10
  let walls_per_room := 8
  let purple_walls := 32
  let purple_rooms := purple_walls / walls_per_room
  let green_rooms := total_rooms - purple_rooms
  (green_rooms : ℚ) / total_rooms = 3 / 5 := by
  sorry

end NUMINAMATH_GPT_johan_painted_green_fraction_l2146_214681


namespace NUMINAMATH_GPT_ketchup_bottles_count_l2146_214617

def ratio_ketchup_mustard_mayo : Nat × Nat × Nat := (3, 3, 2)
def num_mayo_bottles : Nat := 4

theorem ketchup_bottles_count 
  (r : Nat × Nat × Nat)
  (m : Nat)
  (h : r = ratio_ketchup_mustard_mayo)
  (h2 : m = num_mayo_bottles) :
  ∃ k : Nat, k = 6 := by
sorry

end NUMINAMATH_GPT_ketchup_bottles_count_l2146_214617


namespace NUMINAMATH_GPT_temperature_conversion_l2146_214638

noncomputable def celsius_to_fahrenheit (c : ℝ) : ℝ :=
  (c * (9 / 5)) + 32

theorem temperature_conversion (c : ℝ) (hf : c = 60) :
  celsius_to_fahrenheit c = 140 :=
by {
  rw [hf, celsius_to_fahrenheit];
  norm_num
}

end NUMINAMATH_GPT_temperature_conversion_l2146_214638


namespace NUMINAMATH_GPT_polygon_interior_exterior_relation_l2146_214636

theorem polygon_interior_exterior_relation (n : ℕ) (h1 : (n-2) * 180 = 2 * 360) : n = 6 :=
by sorry

end NUMINAMATH_GPT_polygon_interior_exterior_relation_l2146_214636


namespace NUMINAMATH_GPT_sqrt_expression_is_869_l2146_214692

theorem sqrt_expression_is_869 :
  (31 * 30 * 29 * 28 + 1) = 869 := 
sorry

end NUMINAMATH_GPT_sqrt_expression_is_869_l2146_214692


namespace NUMINAMATH_GPT_proof_problem_l2146_214603

/-- Definition of the problem -/
def problem_statement : Prop :=
  ∃(a b c : ℝ) (A B C : ℝ) (D : ℝ),
    -- Conditions:
    ((b ^ 2 = a * c) ∧
     (2 * Real.cos (A - C) - 2 * Real.cos B = 1) ∧
     (D = 5) ∧
     -- Questions:
     (B = Real.pi / 3) ∧
     (∀ (AC CD : ℝ), (a = b ∧ b = c) → -- Equilateral triangle
       (AC * CD = (1/2) * (5 * AC - AC ^ 2) ∧
       (0 < AC * CD ∧ AC * CD ≤ 25/8))))

-- Lean 4 statement
theorem proof_problem : problem_statement := sorry

end NUMINAMATH_GPT_proof_problem_l2146_214603


namespace NUMINAMATH_GPT_smallest_n_for_partition_condition_l2146_214605

theorem smallest_n_for_partition_condition :
  ∃ n : ℕ, n = 4 ∧ ∀ T, (T = {i : ℕ | 2 ≤ i ∧ i ≤ n}) →
  (∀ A B, (T = A ∪ B ∧ A ∩ B = ∅) →
   (∃ a b c, (a ∈ A ∨ a ∈ B) ∧ (b ∈ A ∨ b ∈ B) ∧ (a + b = c))) := sorry

end NUMINAMATH_GPT_smallest_n_for_partition_condition_l2146_214605


namespace NUMINAMATH_GPT_hamburgers_leftover_l2146_214655

-- Define the number of hamburgers made and served
def hamburgers_made : ℕ := 9
def hamburgers_served : ℕ := 3

-- Prove the number of leftover hamburgers
theorem hamburgers_leftover : hamburgers_made - hamburgers_served = 6 := 
by
  sorry

end NUMINAMATH_GPT_hamburgers_leftover_l2146_214655


namespace NUMINAMATH_GPT_positive_root_exists_iff_p_range_l2146_214687

theorem positive_root_exists_iff_p_range (p : ℝ) :
  (∃ x : ℝ, x > 0 ∧ x^4 + 4 * p * x^3 + x^2 + 4 * p * x + 4 = 0) ↔ 
  p ∈ Set.Iio (-Real.sqrt 2 / 2) ∪ Set.Ioi (Real.sqrt 2 / 2) :=
by
  sorry

end NUMINAMATH_GPT_positive_root_exists_iff_p_range_l2146_214687


namespace NUMINAMATH_GPT_percent_problem_l2146_214669

variable (x : ℝ)

theorem percent_problem (h : 0.30 * 0.15 * x = 27) : 0.15 * 0.30 * x = 27 :=
by sorry

end NUMINAMATH_GPT_percent_problem_l2146_214669


namespace NUMINAMATH_GPT_general_equation_M_range_distance_D_to_l_l2146_214651

noncomputable def parametric_to_general (θ : ℝ) : Prop :=
  let x := Real.cos θ
  let y := 2 * Real.sin θ
  x^2 + y^2 / 4 = 1

noncomputable def distance_range (θ : ℝ) : Prop :=
  let x := Real.cos θ
  let y := 2 * Real.sin θ
  let l := x + y - 4
  let d := |x + 2 * y - 4| / Real.sqrt 2
  let min_dist := (4 * Real.sqrt 2 - Real.sqrt 10) / 2
  let max_dist := (4 * Real.sqrt 2 + Real.sqrt 10) / 2
  min_dist ≤ d ∧ d ≤ max_dist

theorem general_equation_M (θ : ℝ) : parametric_to_general θ := sorry

theorem range_distance_D_to_l (θ : ℝ) : distance_range θ := sorry

end NUMINAMATH_GPT_general_equation_M_range_distance_D_to_l_l2146_214651


namespace NUMINAMATH_GPT_Patriots_won_30_games_l2146_214601

def Tigers_won_more_games_than_Eagles (games_tigers games_eagles : ℕ) : Prop :=
games_tigers > games_eagles

def Patriots_won_more_than_Cubs_less_than_Mounties (games_patriots games_cubs games_mounties : ℕ) : Prop :=
games_cubs < games_patriots ∧ games_patriots < games_mounties

def Cubs_won_more_than_20_games (games_cubs : ℕ) : Prop :=
games_cubs > 20

theorem Patriots_won_30_games (games_tigers games_eagles games_patriots games_cubs games_mounties : ℕ)  :
  Tigers_won_more_games_than_Eagles games_tigers games_eagles →
  Patriots_won_more_than_Cubs_less_than_Mounties games_patriots games_cubs games_mounties →
  Cubs_won_more_than_20_games games_cubs →
  ∃ games_patriots, games_patriots = 30 := 
by
  sorry

end NUMINAMATH_GPT_Patriots_won_30_games_l2146_214601


namespace NUMINAMATH_GPT_fractions_proper_or_improper_l2146_214604

theorem fractions_proper_or_improper : 
  ∀ (a b : ℚ), (∃ p q : ℚ, a = p / q ∧ p < q) ∨ (∃ r s : ℚ, a = r / s ∧ r ≥ s) :=
by 
  sorry

end NUMINAMATH_GPT_fractions_proper_or_improper_l2146_214604


namespace NUMINAMATH_GPT_relatively_prime_ratios_l2146_214632

theorem relatively_prime_ratios (r s : ℕ) (h_coprime: Nat.gcd r s = 1) 
  (h_cond: (r : ℝ) / s = 2 * (Real.sqrt 2 + Real.sqrt 10) / (5 * Real.sqrt (3 + Real.sqrt 5))) :
  r = 4 ∧ s = 5 :=
by
  sorry

end NUMINAMATH_GPT_relatively_prime_ratios_l2146_214632


namespace NUMINAMATH_GPT_interest_rate_correct_l2146_214695

noncomputable def annual_interest_rate : ℝ :=
  4^(1/10) - 1

theorem interest_rate_correct (P A₁₀ A₁₅ : ℝ) (h₁ : P = 6000) (h₂ : A₁₀ = 24000) (h₃ : A₁₅ = 48000) :
  (P * (1 + annual_interest_rate)^10 = A₁₀) ∧ (P * (1 + annual_interest_rate)^15 = A₁₅) :=
by
  sorry

end NUMINAMATH_GPT_interest_rate_correct_l2146_214695


namespace NUMINAMATH_GPT_women_with_fair_hair_percentage_l2146_214679

theorem women_with_fair_hair_percentage
  (A : ℝ) (B : ℝ)
  (hA : A = 0.40)
  (hB : B = 0.25) :
  A * B = 0.10 := 
by
  rw [hA, hB]
  norm_num

end NUMINAMATH_GPT_women_with_fair_hair_percentage_l2146_214679


namespace NUMINAMATH_GPT_min_length_GH_l2146_214690

theorem min_length_GH :
  let ellipse (x y : ℝ) := (x^2 / 4) + y^2 = 1
  let A := (-2, 0)
  let B := (2, 0)
  ∀ P G H : ℝ × ℝ,
    (P.1^2 / 4 + P.2^2 = 1) →
    P.2 > 0 →
    (G.2 = 3) →
    (H.2 = 3) →
    ∃ k : ℝ, k > 0 ∧ G.1 = 3 / k - 2 ∧ H.1 = -12 * k + 2 →
    |G.1 - H.1| = 8 :=
sorry

end NUMINAMATH_GPT_min_length_GH_l2146_214690


namespace NUMINAMATH_GPT_eval_expression_l2146_214623

open Real

theorem eval_expression :
  (0.8^5 - (0.5^6 / 0.8^4) + 0.40 + 0.5^3 - log 0.3 + sin (π / 6)) = 2.51853302734375 :=
  sorry

end NUMINAMATH_GPT_eval_expression_l2146_214623


namespace NUMINAMATH_GPT_white_tshirts_l2146_214645

theorem white_tshirts (packages shirts_per_package : ℕ) (h1 : packages = 71) (h2 : shirts_per_package = 6) : packages * shirts_per_package = 426 := 
by 
  sorry

end NUMINAMATH_GPT_white_tshirts_l2146_214645


namespace NUMINAMATH_GPT_angle_between_lines_is_arctan_one_third_l2146_214609

theorem angle_between_lines_is_arctan_one_third
  (l1 : ∀ x y : ℝ, 2 * x - y + 1 = 0)
  (l2 : ∀ x y : ℝ, x - y - 2 = 0)
  : ∃ θ : ℝ, θ = Real.arctan (1 / 3) := 
sorry

end NUMINAMATH_GPT_angle_between_lines_is_arctan_one_third_l2146_214609


namespace NUMINAMATH_GPT_range_of_m_l2146_214682

theorem range_of_m (m : ℝ) :
  (∃ x : ℝ, 2^|x| + m = 0) → m ≤ -1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l2146_214682


namespace NUMINAMATH_GPT_average_annual_growth_rate_sales_revenue_2018_l2146_214625

-- Define the conditions as hypotheses
def initial_sales := 200000
def final_sales := 800000
def years := 2
def growth_rate := 1.0 -- representing 100%

theorem average_annual_growth_rate (x : ℝ) :
  (initial_sales : ℝ) * (1 + x)^years = final_sales → x = 1 :=
by
  intro h1
  sorry

theorem sales_revenue_2018 (x : ℝ) (revenue_2017 : ℝ) :
  x = 1 → revenue_2017 = final_sales → revenue_2017 * (1 + x) = 1600000 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_average_annual_growth_rate_sales_revenue_2018_l2146_214625


namespace NUMINAMATH_GPT_trajectory_of_center_of_moving_circle_l2146_214653

noncomputable def center_trajectory (x y : ℝ) : Prop :=
  0 < y ∧ y ≤ 1 ∧ x^2 = 4 * (y - 1)

theorem trajectory_of_center_of_moving_circle (x y : ℝ) :
  0 ≤ y ∧ y ≤ 2 ∧ x^2 + y^2 = 4 ∧ 0 < y → center_trajectory x y :=
by
  sorry

end NUMINAMATH_GPT_trajectory_of_center_of_moving_circle_l2146_214653


namespace NUMINAMATH_GPT_smallest_positive_integer_problem_l2146_214694

theorem smallest_positive_integer_problem
  (n : ℕ) 
  (h1 : 50 ∣ n) 
  (h2 : (∃ e1 e2 e3 : ℕ, n = 2^e1 * 5^e2 * 3^e3 ∧ (e1 + 1) * (e2 + 1) * (e3 + 1) = 100)) 
  (h3 : ∀ m : ℕ, (50 ∣ m) → ((∃ e1 e2 e3 : ℕ, m = 2^e1 * 5^e2 * 3^e3 ∧ (e1 + 1) * (e2 + 1) * (e3 + 1) = 100) → (n ≤ m))) :
  n / 50 = 8100 := 
sorry

end NUMINAMATH_GPT_smallest_positive_integer_problem_l2146_214694


namespace NUMINAMATH_GPT_circles_intersect_line_l2146_214684

theorem circles_intersect_line (m c : ℝ)
  (hA : (1 : ℝ) - 3 + c = 0)
  (hB : 1 = -(m - 1) / (-4)) :
  m + c = -1 :=
by
  sorry

end NUMINAMATH_GPT_circles_intersect_line_l2146_214684


namespace NUMINAMATH_GPT_find_a_of_pure_imaginary_z_l2146_214631

-- Definition of a pure imaginary number
def pure_imaginary (z : ℂ) : Prop := z.re = 0

-- Main theorem statement
theorem find_a_of_pure_imaginary_z (a : ℝ) (z : ℂ) (hz : pure_imaginary z) (h : (2 - I) * z = 4 + 2 * a * I) : a = 4 :=
by
  sorry

end NUMINAMATH_GPT_find_a_of_pure_imaginary_z_l2146_214631


namespace NUMINAMATH_GPT_min_days_to_find_poisoned_apple_l2146_214660

theorem min_days_to_find_poisoned_apple (n : ℕ) (n_pos : 0 < n) : 
  ∀ k : ℕ, 2^k ≥ 2021 → k ≥ 11 :=
  sorry

end NUMINAMATH_GPT_min_days_to_find_poisoned_apple_l2146_214660


namespace NUMINAMATH_GPT_mass_of_three_packages_l2146_214643

noncomputable def total_mass {x y z : ℝ} (h1 : x + y = 112) (h2 : y + z = 118) (h3 : z + x = 120) : ℝ := 
  x + y + z

theorem mass_of_three_packages {x y z : ℝ} (h1 : x + y = 112) (h2 : y + z = 118) (h3 : z + x = 120) : total_mass h1 h2 h3 = 175 :=
by
  sorry

end NUMINAMATH_GPT_mass_of_three_packages_l2146_214643


namespace NUMINAMATH_GPT_simplify_expression_l2146_214665

theorem simplify_expression (w : ℝ) : 2 * w + 3 - 4 * w - 5 + 6 * w + 7 - 8 * w - 9 = -4 * w - 4 :=
by
  -- Proof steps would go here
  sorry

end NUMINAMATH_GPT_simplify_expression_l2146_214665


namespace NUMINAMATH_GPT_correct_equation_is_x2_sub_10x_add_9_l2146_214646

-- Define the roots found by Student A and Student B
def roots_A := (8, 2)
def roots_B := (-9, -1)

-- Define the incorrect equation by student A from given roots
def equation_A (x : ℝ) := x^2 - 10 * x + 16

-- Define the incorrect equation by student B from given roots
def equation_B (x : ℝ) := x^2 + 10 * x + 9

-- Define the correct quadratic equation
def correct_quadratic_equation (x : ℝ) := x^2 - 10 * x + 9

-- Theorem stating that the correct quadratic equation balances the errors of both students
theorem correct_equation_is_x2_sub_10x_add_9 :
  ∃ (eq_correct : ℝ → ℝ), 
    eq_correct = correct_quadratic_equation :=
by
  -- proof will go here
  sorry

end NUMINAMATH_GPT_correct_equation_is_x2_sub_10x_add_9_l2146_214646


namespace NUMINAMATH_GPT_noemi_initial_money_l2146_214686

variable (money_lost_roulette : ℕ := 400)
variable (money_lost_blackjack : ℕ := 500)
variable (money_left : ℕ)
variable (money_started : ℕ)

axiom money_left_condition : money_left > 0
axiom total_loss_condition : money_lost_roulette + money_lost_blackjack = 900

theorem noemi_initial_money (h1 : money_lost_roulette = 400) (h2 : money_lost_blackjack = 500)
    (h3 : money_started - 900 = money_left) (h4 : money_left > 0) :
    money_started > 900 := by
  sorry

end NUMINAMATH_GPT_noemi_initial_money_l2146_214686


namespace NUMINAMATH_GPT_shapes_identification_l2146_214661

theorem shapes_identification :
  (∃ x y: ℝ, (x - 1/2)^2 + y^2 = 1/4) ∧ (∃ t: ℝ, x = -t ∧ y = 2 + t → x + y + 1 = 0) :=
by
  sorry

end NUMINAMATH_GPT_shapes_identification_l2146_214661


namespace NUMINAMATH_GPT_max_area_right_triangle_l2146_214656

def right_triangle_max_area (l : ℝ) (p : ℝ) (h : ℝ) : ℝ :=
  l + p + h

noncomputable def maximal_area (x y : ℝ) : ℝ :=
  (1/2) * x * y

theorem max_area_right_triangle (x y : ℝ) (h : ℝ) (hp : h = Real.sqrt (x^2 + y^2)) (hp2: x + y + h = 60) :
  maximal_area 30 30 = 450 :=
by
  sorry

end NUMINAMATH_GPT_max_area_right_triangle_l2146_214656


namespace NUMINAMATH_GPT_vector_magnitude_l2146_214647

theorem vector_magnitude (a b : ℝ × ℝ) (a_def : a = (2, 1)) (b_def : b = (-2, 4)) :
  ‖(a.1 - b.1, a.2 - b.2)‖ = 5 := by
  sorry

end NUMINAMATH_GPT_vector_magnitude_l2146_214647


namespace NUMINAMATH_GPT_minimum_value_at_x_eq_3_l2146_214642

theorem minimum_value_at_x_eq_3 (b : ℝ) : 
  ∃ m : ℝ, (∀ x : ℝ, 3 * x^2 - 18 * x + b ≥ m) ∧ (3 * 3^2 - 18 * 3 + b = m) :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_at_x_eq_3_l2146_214642


namespace NUMINAMATH_GPT_P_works_alone_l2146_214634

theorem P_works_alone (P : ℝ) (hP : 2 * (1 / P + 1 / 15) + 0.6 * (1 / P) = 1) : P = 3 :=
by sorry

end NUMINAMATH_GPT_P_works_alone_l2146_214634


namespace NUMINAMATH_GPT_sequence_finite_l2146_214658

def sequence_terminates (a_0 : ℕ) : Prop :=
  ∀ (a : ℕ → ℕ), (a 0 = a_0) ∧ 
                  (∀ n, ((a n > 5) ∧ (a n % 10 ≤ 5) → a (n + 1) = a n / 10)) ∧
                  (∀ n, ((a n > 5) ∧ (a n % 10 > 5) → a (n + 1) = 9 * a n)) → 
                  ∃ n, a n ≤ 5 

theorem sequence_finite (a_0 : ℕ) : sequence_terminates a_0 :=
sorry

end NUMINAMATH_GPT_sequence_finite_l2146_214658


namespace NUMINAMATH_GPT_find_integer_l2146_214677

theorem find_integer (n : ℤ) (h1 : n ≥ 50) (h2 : n ≤ 100) (h3 : n % 7 = 0) (h4 : n % 9 = 3) (h5 : n % 6 = 3) : n = 84 := 
by 
  sorry

end NUMINAMATH_GPT_find_integer_l2146_214677


namespace NUMINAMATH_GPT_jame_initial_gold_bars_l2146_214685

theorem jame_initial_gold_bars (X : ℝ) (h1 : X * 0.1 + 0.5 * (X * 0.9) = 0.5 * (X * 0.9) - 27) :
  X = 60 :=
by
-- Placeholder for proof
sorry

end NUMINAMATH_GPT_jame_initial_gold_bars_l2146_214685


namespace NUMINAMATH_GPT_reported_length_correct_l2146_214678

def length_in_yards := 80
def conversion_factor := 3 -- 1 yard is 3 feet
def length_in_feet := 240

theorem reported_length_correct :
  length_in_feet = length_in_yards * conversion_factor :=
by rfl

end NUMINAMATH_GPT_reported_length_correct_l2146_214678


namespace NUMINAMATH_GPT_diameter_of_circle_l2146_214615

theorem diameter_of_circle (a b : ℕ) (r : ℝ) (h_a : a = 6) (h_b : b = 8) (h_triangle : a^2 + b^2 = r^2) : r = 10 :=
by 
  rw [h_a, h_b] at h_triangle
  sorry

end NUMINAMATH_GPT_diameter_of_circle_l2146_214615


namespace NUMINAMATH_GPT_probability_increase_l2146_214664

theorem probability_increase:
  let P_win1 := 0.30
  let P_lose1 := 0.70
  let P_win2 := 0.50
  let P_lose2 := 0.50
  let P_win3 := 0.40
  let P_lose3 := 0.60
  let P_win4 := 0.25
  let P_lose4 := 0.75
  let P_win_all := P_win1 * P_win2 * P_win3 * P_win4
  let P_lose_all := P_lose1 * P_lose2 * P_lose3 * P_lose4
  (P_lose_all - P_win_all) / P_win_all = 9.5 :=
by
  sorry

end NUMINAMATH_GPT_probability_increase_l2146_214664


namespace NUMINAMATH_GPT_part1_part2_l2146_214613

-- Define the function f(x) = ln(x) / x
noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

-- Part 1: Prove the range of k such that f(x) < k * x for all x
theorem part1 (k : ℝ) : (∀ x : ℝ, x > 0 → f x < k * x) ↔ k > 1 / (2 * Real.exp 1) :=
by sorry

-- Part 2: Define the function g(x) = f(x) - k * x and prove the range of k for which g(x) has two zeros in the interval [1/e, e^2]
noncomputable def g (x k : ℝ) : ℝ := f x - k * x

theorem part2 (k : ℝ) : (∃ x1 x2 : ℝ, 1 / Real.exp 1 ≤ x1 ∧ x1 ≤ Real.exp 2 ∧
                                 1 / Real.exp 1 ≤ x2 ∧ x2 ≤ Real.exp 2 ∧
                                 g x1 k = 0 ∧ g x2 k = 0 ∧ x1 ≠ x2)
                               ↔ 2 / (Real.exp 4) ≤ k ∧ k < 1 / (2 * Real.exp 1) :=
by sorry

end NUMINAMATH_GPT_part1_part2_l2146_214613


namespace NUMINAMATH_GPT_equivalence_of_statements_l2146_214648

theorem equivalence_of_statements 
  (Q P : Prop) :
  (Q → ¬ P) ↔ (P → ¬ Q) := sorry

end NUMINAMATH_GPT_equivalence_of_statements_l2146_214648


namespace NUMINAMATH_GPT_ratio_of_pants_to_shirts_l2146_214668

noncomputable def cost_shirt : ℝ := 6
noncomputable def cost_pants : ℝ := 8
noncomputable def num_shirts : ℝ := 10
noncomputable def total_cost : ℝ := 100

noncomputable def num_pants : ℝ :=
  (total_cost - (num_shirts * cost_shirt)) / cost_pants

theorem ratio_of_pants_to_shirts : num_pants / num_shirts = 1 / 2 := by
  sorry

end NUMINAMATH_GPT_ratio_of_pants_to_shirts_l2146_214668


namespace NUMINAMATH_GPT_height_inradius_ratio_is_7_l2146_214620

-- Definitions of geometric entities and given conditions.
variable (h r : ℝ)
variable (cos_theta : ℝ)
variable (cos_theta_eq : cos_theta = 1 / 6)

-- Theorem statement: Ratio of height to inradius is 7 given the cosine condition.
theorem height_inradius_ratio_is_7
  (h r : ℝ)
  (cos_theta : ℝ)
  (cos_theta_eq : cos_theta = 1 / 6)
  (prism_def : true) -- Added to mark the geometric nature properly
: h / r = 7 :=
sorry  -- Placeholder for the actual proof.

end NUMINAMATH_GPT_height_inradius_ratio_is_7_l2146_214620


namespace NUMINAMATH_GPT_english_textbook_cost_l2146_214693

variable (cost_english_book : ℝ)

theorem english_textbook_cost :
  let geography_book_cost := 10.50
  let num_books := 35
  let total_order_cost := 630
  (num_books * cost_english_book + num_books * geography_book_cost = total_order_cost) →
  cost_english_book = 7.50 :=
by {
sorry
}

end NUMINAMATH_GPT_english_textbook_cost_l2146_214693


namespace NUMINAMATH_GPT_algebraic_expression_opposite_l2146_214602

theorem algebraic_expression_opposite (a b x : ℝ) (h : b^2 * x^2 + |a| = -(b^2 * x^2 + |a|)) : a * b = 0 :=
by 
  sorry

end NUMINAMATH_GPT_algebraic_expression_opposite_l2146_214602


namespace NUMINAMATH_GPT_sum_of_relatively_prime_integers_l2146_214698

theorem sum_of_relatively_prime_integers (n : ℕ) (h : n ≥ 7) :
  ∃ a b : ℕ, n = a + b ∧ a > 1 ∧ b > 1 ∧ Nat.gcd a b = 1 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_relatively_prime_integers_l2146_214698


namespace NUMINAMATH_GPT_ladybugs_total_total_ladybugs_is_5_l2146_214675

def num_ladybugs (x y : ℕ) : ℕ :=
  x + y

theorem ladybugs_total (x y n : ℕ) 
    (h_spot_calc_1: 6 * x + 4 * y = 30 ∨ 6 * x + 4 * y = 26)
    (h_total_spots_30: (6 * x + 4 * y = 30) ↔ 3 * x + 2 * y = 15)
    (h_total_spots_26: (6 * x + 4 * y = 26) ↔ 3 * x + 2 * y = 13)
    (h_truth_only_one: 
       (6 * x + 4 * y = 30 ∧ ¬(6 * x + 4 * y = 26)) ∨
       (¬(6 * x + 4 * y = 30) ∧ 6 * x + 4 * y = 26))
    : n = x + y :=
by 
  sorry

theorem total_ladybugs_is_5 : ∃ x y : ℕ, num_ladybugs x y = 5 :=
  ⟨3, 2, rfl⟩

end NUMINAMATH_GPT_ladybugs_total_total_ladybugs_is_5_l2146_214675


namespace NUMINAMATH_GPT_max_correct_answers_l2146_214600

theorem max_correct_answers (a b c : ℕ) (h1 : a + b + c = 80) (h2 : 5 * a - 2 * c = 150) : a ≤ 44 :=
by
  sorry

end NUMINAMATH_GPT_max_correct_answers_l2146_214600


namespace NUMINAMATH_GPT_star_comm_star_assoc_star_id_exists_star_not_dist_add_l2146_214671

def star (x y : ℝ) : ℝ := (x + 2) * (y + 2) - 2

-- Statement 1: Commutativity
theorem star_comm : ∀ x y : ℝ, star x y = star y x := 
by sorry

-- Statement 2: Associativity
theorem star_assoc : ∀ x y z : ℝ, star (star x y) z = star x (star y z) := 
by sorry

-- Statement 3: Identity Element
theorem star_id_exists : ∃ e : ℝ, ∀ x : ℝ, star x e = x := 
by sorry

-- Statement 4: Distributivity Over Addition
theorem star_not_dist_add : ∃ x y z : ℝ, star x (y + z) ≠ star x y + star x z := 
by sorry

end NUMINAMATH_GPT_star_comm_star_assoc_star_id_exists_star_not_dist_add_l2146_214671


namespace NUMINAMATH_GPT_claire_photos_l2146_214614

theorem claire_photos (C L R : ℕ) 
  (h1 : L = 3 * C) 
  (h2 : R = C + 12)
  (h3 : L = R) : C = 6 := 
by
  sorry

end NUMINAMATH_GPT_claire_photos_l2146_214614


namespace NUMINAMATH_GPT_product_of_roots_l2146_214635

-- Define the quadratic function
def quadratic (x : ℝ) : ℝ := x^2 - 9 * x + 20

-- The main statement for the Lean theorem
theorem product_of_roots : (∃ x₁ x₂ : ℝ, quadratic x₁ = 0 ∧ quadratic x₂ = 0 ∧ x₁ * x₂ = 20) :=
by
  sorry

end NUMINAMATH_GPT_product_of_roots_l2146_214635


namespace NUMINAMATH_GPT_georgie_ghost_enter_exit_diff_window_l2146_214649

theorem georgie_ghost_enter_exit_diff_window (n : ℕ) (h : n = 8) :
    (∃ enter exit, enter ≠ exit ∧ 1 ≤ enter ∧ enter ≤ n ∧ 1 ≤ exit ∧ exit ≤ n) ∧
    (∃ W : ℕ, W = (n * (n - 1))) :=
sorry

end NUMINAMATH_GPT_georgie_ghost_enter_exit_diff_window_l2146_214649


namespace NUMINAMATH_GPT_track_length_l2146_214652

theorem track_length
  (meet1_dist : ℝ)
  (meet2_sally_additional_dist : ℝ)
  (constant_speed : ∀ (b_speed s_speed : ℝ), b_speed = s_speed)
  (opposite_start : true)
  (brenda_first_meet : meet1_dist = 100)
  (sally_second_meet : meet2_sally_additional_dist = 200) :
  ∃ L : ℝ, L = 200 :=
by
  sorry

end NUMINAMATH_GPT_track_length_l2146_214652


namespace NUMINAMATH_GPT_evaluate_imaginary_expression_l2146_214626

theorem evaluate_imaginary_expression (i : ℂ) (h_i2 : i^2 = -1) (h_i4 : i^4 = 1) :
  i^14 + i^19 + i^24 + i^29 + 3 * i^34 + 2 * i^39 = -3 - 2 * i :=
by sorry

end NUMINAMATH_GPT_evaluate_imaginary_expression_l2146_214626


namespace NUMINAMATH_GPT_power_function_value_l2146_214627

noncomputable def f (x : ℝ) (α : ℝ) : ℝ := x ^ α

theorem power_function_value (α : ℝ) (h : 2 ^ α = (Real.sqrt 2) / 2) : f 4 α = 1 / 2 := 
by 
  sorry

end NUMINAMATH_GPT_power_function_value_l2146_214627


namespace NUMINAMATH_GPT_problem_bound_l2146_214691

theorem problem_bound (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) (h4 : x + y + z = 1) :
  0 ≤ x * y + y * z + z * x - 2 * x * y * z ∧ x * y + y * z + z * x - 2 * x * y * z ≤ 7 / 27 :=
by 
  sorry

end NUMINAMATH_GPT_problem_bound_l2146_214691


namespace NUMINAMATH_GPT_travel_distance_l2146_214699

-- Define the average speed of the car
def speed : ℕ := 68

-- Define the duration of the trip in hours
def time : ℕ := 12

-- Define the distance formula for constant speed
def distance (speed time : ℕ) : ℕ := speed * time

-- Proof statement
theorem travel_distance : distance speed time = 756 := by
  -- Provide a placeholder for the proof
  sorry

end NUMINAMATH_GPT_travel_distance_l2146_214699


namespace NUMINAMATH_GPT_five_natural_numbers_increase_15_times_l2146_214616

noncomputable def prod_of_decreased_factors_is_15_times_original (a1 a2 a3 a4 a5 : ℕ) : Prop :=
  (a1 - 3) * (a2 - 3) * (a3 - 3) * (a4 - 3) * (a5 - 3) = 15 * (a1 * a2 * a3 * a4 * a5)

theorem five_natural_numbers_increase_15_times {a1 a2 a3 a4 a5 : ℕ} :
  a1 * a2 * a3 * a4 * a5 = 48 → prod_of_decreased_factors_is_15_times_original a1 a2 a3 a4 a5 :=
by
  sorry

end NUMINAMATH_GPT_five_natural_numbers_increase_15_times_l2146_214616


namespace NUMINAMATH_GPT_triangle_right_hypotenuse_l2146_214611

theorem triangle_right_hypotenuse (c : ℝ) (a : ℝ) (h₀ : c = 4) (h₁ : 0 < a) (h₂ : a^2 + b^2 = c^2) :
  a ≤ 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_triangle_right_hypotenuse_l2146_214611


namespace NUMINAMATH_GPT_rational_quotient_of_arith_geo_subseq_l2146_214657

theorem rational_quotient_of_arith_geo_subseq (A d : ℝ) (h_d_nonzero : d ≠ 0)
    (h_contains_geo : ∃ (q : ℝ) (k m n : ℕ), q ≠ 1 ∧ q ≠ 0 ∧ 
        A + k * d = (A + m * d) * q ∧ A + m * d = (A + n * d) * q)
    : ∃ (r : ℚ), A / d = r :=
  sorry

end NUMINAMATH_GPT_rational_quotient_of_arith_geo_subseq_l2146_214657


namespace NUMINAMATH_GPT_evaluate_expression_l2146_214688

/-- Given conditions: -/
def a : ℕ := 3998
def b : ℕ := 3999

theorem evaluate_expression :
  b^3 - 2 * a * b^2 - 2 * a^2 * b + (b - 2)^3 = 95806315 :=
  sorry

end NUMINAMATH_GPT_evaluate_expression_l2146_214688


namespace NUMINAMATH_GPT_trapezoid_base_solutions_l2146_214683

theorem trapezoid_base_solutions (A h : ℕ) (d : ℕ) (bd : ℕ → Prop)
  (hA : A = 1800) (hH : h = 60) (hD : d = 10) (hBd : ∀ (x : ℕ), bd x ↔ ∃ (k : ℕ), x = d * k) :
  ∃ m n : ℕ, bd (10 * m) ∧ bd (10 * n) ∧ 10 * (m + n) = 60 ∧ m + n = 6 :=
by
  simp [hA, hH, hD, hBd]
  sorry

end NUMINAMATH_GPT_trapezoid_base_solutions_l2146_214683


namespace NUMINAMATH_GPT_giant_slide_wait_is_15_l2146_214662

noncomputable def wait_time_for_giant_slide
  (hours_at_carnival : ℕ) 
  (roller_coaster_wait : ℕ)
  (tilt_a_whirl_wait : ℕ)
  (rides_roller_coaster : ℕ)
  (rides_tilt_a_whirl : ℕ)
  (rides_giant_slide : ℕ) : ℕ :=
  
  (hours_at_carnival * 60 - (roller_coaster_wait * rides_roller_coaster + tilt_a_whirl_wait * rides_tilt_a_whirl)) / rides_giant_slide

theorem giant_slide_wait_is_15 :
  wait_time_for_giant_slide 4 30 60 4 1 4 = 15 := 
sorry

end NUMINAMATH_GPT_giant_slide_wait_is_15_l2146_214662


namespace NUMINAMATH_GPT_triangle_side_sum_l2146_214654

def sum_of_remaining_sides_of_triangle (A B C : ℝ) (a b c : ℝ) (α β γ : ℝ) : Prop :=
  α = 40 ∧ β = 50 ∧ γ = 180 - α - β ∧ c = 8 * Real.sqrt 3 →
  (a + b) = 34.3

theorem triangle_side_sum (A B C : ℝ) (a b c : ℝ) (α β γ : ℝ) :
  sum_of_remaining_sides_of_triangle A B C a b c α β γ :=
sorry

end NUMINAMATH_GPT_triangle_side_sum_l2146_214654


namespace NUMINAMATH_GPT_roberto_raise_percentage_l2146_214659

theorem roberto_raise_percentage
    (starting_salary : ℝ)
    (previous_salary : ℝ)
    (current_salary : ℝ)
    (h1 : starting_salary = 80000)
    (h2 : previous_salary = starting_salary * 1.40)
    (h3 : current_salary = 134400) :
    ((current_salary - previous_salary) / previous_salary) * 100 = 20 :=
by sorry

end NUMINAMATH_GPT_roberto_raise_percentage_l2146_214659


namespace NUMINAMATH_GPT_original_price_of_tshirt_l2146_214689

theorem original_price_of_tshirt :
  ∀ (P : ℝ), 
    (∀ discount quantity_sold revenue : ℝ, discount = 8 ∧ quantity_sold = 130 ∧ revenue = 5590 ∧
      revenue = quantity_sold * (P - discount)) → P = 51 := 
by
  intros P
  intro h
  sorry

end NUMINAMATH_GPT_original_price_of_tshirt_l2146_214689


namespace NUMINAMATH_GPT_distance_from_point_to_directrix_l2146_214622

noncomputable def parabola_point_distance_to_directrix (x y p : ℝ) : ℝ :=
  x + p / 2

theorem distance_from_point_to_directrix (x y p dist_to_directrix : ℝ)
  (hx : y^2 = 2 * p * x)
  (hdist_def : dist_to_directrix = parabola_point_distance_to_directrix x y p) :
  dist_to_directrix = 9 / 4 :=
by
  sorry

end NUMINAMATH_GPT_distance_from_point_to_directrix_l2146_214622
