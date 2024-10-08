import Mathlib

namespace min_value_quadratic_l44_44947

theorem min_value_quadratic (x : ℝ) : ∃ x, x = -7 ∧ (x^2 + 14 * x + 24 = -25) := sorry

end min_value_quadratic_l44_44947


namespace new_mean_l44_44388

-- Define the conditions
def mean_of_numbers (n : ℕ) (mean : ℝ) : ℝ := n * mean
def added_to_each (n : ℕ) (addend : ℝ) : ℝ := n * addend

-- The proof problem
theorem new_mean (n : ℕ) (mean addend : ℝ) (h1 : mean_of_numbers n mean = 600) (h2 : added_to_each n addend = 150) (h3 : n = 15) (h4 : mean = 40) (h5 : addend = 10) :
  (mean_of_numbers n mean + added_to_each n addend) / n = 50 :=
by
  sorry

end new_mean_l44_44388


namespace add_base_6_l44_44665

theorem add_base_6 (a b c : ℕ) (h₀ : a = 3 * 6^3 + 4 * 6^2 + 2 * 6 + 1)
                    (h₁ : b = 4 * 6^3 + 5 * 6^2 + 2 * 6 + 5)
                    (h₂ : c = 1 * 6^4 + 2 * 6^3 + 3 * 6^2 + 5 * 6 + 0) : 
  a + b = c :=
by  
  sorry

end add_base_6_l44_44665


namespace packed_tents_and_food_truck_arrangements_minimum_transportation_cost_l44_44849

-- Define the conditions
def total_items : ℕ := 320
def tents_more_than_food : ℕ := 80
def total_trucks : ℕ := 8
def type_A_tent_capacity : ℕ := 40
def type_A_food_capacity : ℕ := 10
def type_B_tent_capacity : ℕ := 20
def type_B_food_capacity : ℕ := 20
def type_A_cost : ℕ := 4000
def type_B_cost : ℕ := 3600

-- Questions to prove:
theorem packed_tents_and_food:
  ∃ t f : ℕ, t + f = total_items ∧ t = f + tents_more_than_food ∧ t = 200 ∧ f = 120 :=
sorry

theorem truck_arrangements:
  ∃ A B : ℕ, A + B = total_trucks ∧
    (A * type_A_tent_capacity + B * type_B_tent_capacity = 200) ∧
    (A * type_A_food_capacity + B * type_B_food_capacity = 120) ∧
    ((A = 2 ∧ B = 6) ∨ (A = 3 ∧ B = 5) ∨ (A = 4 ∧ B = 4)) :=
sorry

theorem minimum_transportation_cost:
  ∃ A B : ℕ, A = 2 ∧ B = 6 ∧ A * type_A_cost + B * type_B_cost = 29600 :=
sorry

end packed_tents_and_food_truck_arrangements_minimum_transportation_cost_l44_44849


namespace budget_equality_year_l44_44726

theorem budget_equality_year :
  ∃ n : ℕ, 540000 + 30000 * n = 780000 - 10000 * n ∧ 1990 + n = 1996 :=
by
  sorry

end budget_equality_year_l44_44726


namespace number_of_days_l44_44402

theorem number_of_days (d : ℝ) (h : 2 * d = 1.5 * d + 3) : d = 6 :=
by
  sorry

end number_of_days_l44_44402


namespace family_of_four_children_has_at_least_one_boy_and_one_girl_l44_44543

noncomputable section

def probability_at_least_one_boy_one_girl : ℚ :=
  1 - (1 / 16 + 1 / 16)

theorem family_of_four_children_has_at_least_one_boy_and_one_girl :
  probability_at_least_one_boy_one_girl = 7 / 8 := by
  sorry

end family_of_four_children_has_at_least_one_boy_and_one_girl_l44_44543


namespace yellow_ball_range_l44_44838

-- Definitions
def probability_condition (x : ℕ) : Prop :=
  (20 / 100 : ℝ) ≤ (4 * x / ((x + 2) * (x + 1))) ∧ (4 * x / ((x + 2) * (x + 1))) ≤ (33 / 100)

theorem yellow_ball_range (x : ℕ) : probability_condition x ↔ 9 ≤ x ∧ x ≤ 16 := 
by
  sorry

end yellow_ball_range_l44_44838


namespace lcm_smallest_value_l44_44361

/-- The smallest possible value of lcm(k, l) for positive 5-digit integers k and l such that gcd(k, l) = 5 is 20010000. -/
theorem lcm_smallest_value (k l : ℕ) (h1 : 10000 ≤ k ∧ k < 100000) (h2 : 10000 ≤ l ∧ l < 100000) (h3 : Nat.gcd k l = 5) : Nat.lcm k l = 20010000 :=
sorry

end lcm_smallest_value_l44_44361


namespace hyperbola_range_k_l44_44297

theorem hyperbola_range_k (k : ℝ) : (4 + k) * (1 - k) < 0 ↔ k ∈ (Set.Iio (-4) ∪ Set.Ioi 1) := 
by
  sorry

end hyperbola_range_k_l44_44297


namespace problem_statement_l44_44494

theorem problem_statement (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a / (b + 2 * c) + b / (c + 2 * a) + c / (a + 2 * b) > 1 / 2) :=
by
  sorry

end problem_statement_l44_44494


namespace winning_jackpot_is_event_l44_44603

-- Definitions based on the conditions
def has_conditions (experiment : String) : Prop :=
  experiment = "A" ∨ experiment = "B" ∨ experiment = "C" ∨ experiment = "D"

def has_outcomes (experiment : String) : Prop :=
  experiment = "D"

def is_event (experiment : String) : Prop :=
  has_conditions experiment ∧ has_outcomes experiment

-- Statement to prove
theorem winning_jackpot_is_event : is_event "D" :=
by
  -- Trivial step to show that D meets both conditions and outcomes
  exact sorry

end winning_jackpot_is_event_l44_44603


namespace reach_any_natural_number_l44_44597

theorem reach_any_natural_number (n : ℕ) : ∃ (f : ℕ → ℕ), f 0 = 1 ∧ (∀ k, f (k + 1) = 3 * f k + 1 ∨ f (k + 1) = f k / 2) ∧ (∃ m, f m = n) := by
  sorry

end reach_any_natural_number_l44_44597


namespace minimum_value_frac_inverse_l44_44357

theorem minimum_value_frac_inverse (a b c : ℝ) (h : a + b + c = 3) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (1 / (a + b)) + (1 / c) ≥ 4 / 3 :=
by
  sorry

end minimum_value_frac_inverse_l44_44357


namespace probability_no_rain_five_days_probability_drought_alert_approx_l44_44113

theorem probability_no_rain_five_days (p : ℚ) (h : p = 1/3) :
  (p ^ 5) = 1 / 243 :=
by
  -- Add assumptions and proceed
  sorry

theorem probability_drought_alert_approx (p : ℚ) (h : p = 1/3) :
  4 * (p ^ 2) = 4 / 9 :=
by
  -- Add assumptions and proceed
  sorry

end probability_no_rain_five_days_probability_drought_alert_approx_l44_44113


namespace smallest_positive_integer_a_l44_44804

theorem smallest_positive_integer_a (a : ℕ) (h1 : 0 < a) (h2 : ∃ b : ℕ, 3150 * a = b^2) : a = 14 :=
by
  sorry

end smallest_positive_integer_a_l44_44804


namespace jo_thinking_number_l44_44877

theorem jo_thinking_number 
  (n : ℕ) 
  (h1 : n < 100) 
  (h2 : n % 8 = 7) 
  (h3 : n % 7 = 4) 
  : n = 95 :=
sorry

end jo_thinking_number_l44_44877


namespace tina_wins_before_first_loss_l44_44319

-- Definitions based on conditions
variable (W : ℕ) -- The number of wins before Tina's first loss

-- Conditions
def win_before_first_loss : W = 10 := by sorry

def total_wins (W : ℕ) := W + 2 * W -- After her first loss, she doubles her wins and loses again
def total_losses : ℕ := 2 -- She loses twice

def career_record_condition (W : ℕ) : Prop := total_wins W - total_losses = 28

-- Proof Problem (Statement)
theorem tina_wins_before_first_loss : career_record_condition W → W = 10 :=
by sorry

end tina_wins_before_first_loss_l44_44319


namespace closest_points_to_A_l44_44698

noncomputable def distance_squared (x y : ℝ) : ℝ :=
  x^2 + (y + 3)^2

def hyperbola (x y : ℝ) : Prop :=
  x^2 - y^2 = 9

theorem closest_points_to_A :
  ∃ (x y : ℝ),
    hyperbola x y ∧
    (distance_squared x y = distance_squared (-3 * Real.sqrt 5 / 2) (-3/2) ∨
     distance_squared x y = distance_squared (3 * Real.sqrt 5 / 2) (-3/2)) :=
sorry

end closest_points_to_A_l44_44698


namespace candy_initial_count_l44_44164

theorem candy_initial_count (candy_given_first candy_given_second candy_given_third candy_bought candy_eaten candy_left initial_candy : ℕ) 
    (h1 : candy_given_first = 18) 
    (h2 : candy_given_second = 12)
    (h3 : candy_given_third = 25)
    (h4 : candy_bought = 10)
    (h5 : candy_eaten = 7)
    (h6 : candy_left = 16)
    (h_initial : candy_left + candy_eaten = initial_candy - candy_bought - candy_given_first - candy_given_second - candy_given_third):
    initial_candy = 68 := 
by 
  sorry

end candy_initial_count_l44_44164


namespace graph_paper_problem_l44_44605

theorem graph_paper_problem :
  let line_eq := ∀ x y : ℝ, 7 * x + 268 * y = 1876
  ∃ (n : ℕ), 
  (∀ x y : ℕ, 0 < x ∧ x ≤ 268 ∧ 0 < y ∧ y ≤ 7 ∧ (7 * (x:ℝ) + 268 * (y:ℝ)) < 1876) →
  n = 801 :=
by
  sorry

end graph_paper_problem_l44_44605


namespace age_difference_l44_44611

-- Defining the necessary variables and their types
variables (A B : ℕ)

-- Given conditions: 
axiom B_current_age : B = 38
axiom future_age_relationship : A + 10 = 2 * (B - 10)

-- Proof goal statement
theorem age_difference : A - B = 8 :=
by
  sorry

end age_difference_l44_44611


namespace range_of_x_l44_44140

def odot (a b : ℝ) : ℝ := a * b + 2 * a + b

theorem range_of_x :
  {x : ℝ | odot x (x - 2) < 0} = {x : ℝ | -2 < x ∧ x < 1} := 
by sorry

end range_of_x_l44_44140


namespace right_triangle_side_lengths_l44_44122

theorem right_triangle_side_lengths (a b c : ℝ) (varrho r : ℝ) (h_varrho : varrho = 8) (h_r : r = 41) : 
  (a = 80 ∧ b = 18 ∧ c = 82) ∨ (a = 18 ∧ b = 80 ∧ c = 82) :=
by
  sorry

end right_triangle_side_lengths_l44_44122


namespace vertical_coordinate_intersection_l44_44445

def original_function (x : ℝ) := x^2 + 2 * x + 1

def shifted_function (x : ℝ) := (x + 3)^2 + 3

theorem vertical_coordinate_intersection :
  shifted_function 0 = 12 :=
by
  sorry

end vertical_coordinate_intersection_l44_44445


namespace coplanar_values_l44_44147

namespace CoplanarLines

-- Define parametric equations of the lines
def line1 (t : ℝ) (m : ℝ) : ℝ × ℝ × ℝ := (3 + 2 * t, 2 - t, 5 + m * t)
def line2 (u : ℝ) (m : ℝ) : ℝ × ℝ × ℝ := (4 - m * u, 5 + 3 * u, 6 + 2 * u)

-- Define coplanarity condition
def coplanar_condition (m : ℝ) : Prop :=
  ∃ t u : ℝ, line1 t m = line2 u m

-- Theorem to prove the specific values of m for coplanarity
theorem coplanar_values (m : ℝ) : coplanar_condition m ↔ (m = -13/9 ∨ m = 1) :=
sorry

end CoplanarLines

end coplanar_values_l44_44147


namespace bus_time_one_way_l44_44185

-- define conditions
def walk_time_one_way := 5 -- 5 minutes for one walk
def total_annual_travel_time_hours := 365 -- 365 hours per year
def work_days_per_year := 365 -- works every day

-- convert annual travel time from hours to minutes
def total_annual_travel_time_minutes := total_annual_travel_time_hours * 60

-- calculate total daily travel time
def total_daily_travel_time := total_annual_travel_time_minutes / work_days_per_year

-- walking time per day
def total_daily_walking_time := (walk_time_one_way * 4)

-- total bus travel time per day
def total_daily_bus_time := total_daily_travel_time - total_daily_walking_time

-- one-way bus time
theorem bus_time_one_way : total_daily_bus_time / 2 = 20 := by
  sorry

end bus_time_one_way_l44_44185


namespace inequality_proof_l44_44421

theorem inequality_proof 
(x1 x2 y1 y2 z1 z2 : ℝ) 
(hx1 : x1 > 0) 
(hx2 : x2 > 0) 
(hineq1 : x1 * y1 - z1^2 > 0) 
(hineq2 : x2 * y2 - z2^2 > 0)
: 
  8 / ((x1 + x2)*(y1 + y2) - (z1 + z2)^2) <= 
  1 / (x1 * y1 - z1^2) + 
  1 / (x2 * y2 - z2^2) := 
sorry

end inequality_proof_l44_44421


namespace buildings_collapsed_l44_44213

theorem buildings_collapsed (B : ℕ) (h₁ : 2 * B = X) (h₂ : 4 * B = Y) (h₃ : 8 * B = Z) (h₄ : B + 2 * B + 4 * B + 8 * B = 60) : B = 4 :=
by
  sorry

end buildings_collapsed_l44_44213


namespace number_of_terms_in_arithmetic_sequence_is_20_l44_44427

theorem number_of_terms_in_arithmetic_sequence_is_20
  (a : ℕ → ℤ)
  (common_difference : ℤ)
  (h1 : common_difference = 2)
  (even_num_terms : ℕ)
  (h2 : ∃ k, even_num_terms = 2 * k)
  (sum_odd_terms sum_even_terms : ℤ)
  (h3 : sum_odd_terms = 15)
  (h4 : sum_even_terms = 35)
  (h5 : ∀ n, a n = a 0 + n * common_difference) :
  even_num_terms = 20 :=
by
  sorry

end number_of_terms_in_arithmetic_sequence_is_20_l44_44427


namespace find_y_l44_44007

variable (h : ℕ) -- integral number of hours

-- Distance between A and B
def distance_AB : ℕ := 60

-- Speed and distance walked by woman starting at A
def speed_A : ℕ := 3
def distance_A (h : ℕ) : ℕ := speed_A * h

-- Speed and distance walked by woman starting at B
def speed_B_1st_hour : ℕ := 2
def distance_B (h : ℕ) : ℕ := (h * (h + 3)) / 2

-- Meeting point equation
def meeting_point_eqn (h : ℕ) : Prop := (distance_A h) + (distance_B h) = distance_AB

-- Requirement: y miles nearer to A whereas y = distance_AB - 2 * distance_B (since B meets closer to A by y miles)
def y_nearer_A (h : ℕ) : ℕ := distance_AB - 2 * (distance_A h)

-- Prove y = 6 for the specific value of h
theorem find_y : ∃ (h : ℕ), meeting_point_eqn h ∧ y_nearer_A h = 6 := by
  sorry

end find_y_l44_44007


namespace no_minimum_of_f_over_M_l44_44293

/-- Define the domain M for the function y = log(3 - 4x + x^2) -/
def domain_M (x : ℝ) : Prop := (x > 3 ∨ x < 1)

/-- Define the function f(x) = 2x + 2 - 3 * 4^x -/
noncomputable def f (x : ℝ) : ℝ := 2 * x + 2 - 3 * 4^x

/-- The theorem statement:
    Prove that f(x) does not have a minimum value for x in the domain M -/
theorem no_minimum_of_f_over_M : ¬ ∃ x ∈ {x | domain_M x}, ∀ y ∈ {x | domain_M x}, f x ≤ f y := sorry

end no_minimum_of_f_over_M_l44_44293


namespace store_profit_in_february_l44_44615

variable (C : ℝ)

def initialSellingPrice := C * 1.20
def secondSellingPrice := initialSellingPrice C * 1.25
def finalSellingPrice := secondSellingPrice C * 0.88

theorem store_profit_in_february
  (initialSellingPrice_eq : initialSellingPrice C = C * 1.20)
  (secondSellingPrice_eq : secondSellingPrice C = initialSellingPrice C * 1.25)
  (finalSellingPrice_eq : finalSellingPrice C = secondSellingPrice C * 0.88)
  : finalSellingPrice C - C = 0.32 * C :=
sorry

end store_profit_in_february_l44_44615


namespace minimum_value_x_plus_y_l44_44686

theorem minimum_value_x_plus_y (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x + 2 * y = x * y) :
  x + y = 3 + 2 * Real.sqrt 2 :=
sorry

end minimum_value_x_plus_y_l44_44686


namespace total_fish_caught_l44_44362

theorem total_fish_caught (leo_fish : ℕ) (agrey_fish : ℕ) (h1 : leo_fish = 40) (h2 : agrey_fish = leo_fish + 20) :
  leo_fish + agrey_fish = 100 :=
by
  sorry

end total_fish_caught_l44_44362


namespace mode_I_swaps_mode_II_swaps_l44_44227

-- Define the original and target strings
def original_sign := "MEGYEI TAKARÉKPÉNZTÁR R. T."
def target_sign := "TATÁR GYERMEK A PÉNZT KÉRI."

-- Define a function for adjacent swaps needed to convert original_sign to target_sign
def adjacent_swaps (orig : String) (target : String) : ℕ := sorry

-- Define a function for any distant swaps needed to convert original_sign to target_sign
def distant_swaps (orig : String) (target : String) : ℕ := sorry

-- The theorems we want to prove
theorem mode_I_swaps : adjacent_swaps original_sign target_sign = 85 := sorry

theorem mode_II_swaps : distant_swaps original_sign target_sign = 11 := sorry

end mode_I_swaps_mode_II_swaps_l44_44227


namespace bella_total_roses_l44_44614

-- Define the constants and conditions
def dozen := 12
def roses_from_parents := 2 * dozen
def friends := 10
def roses_per_friend := 2
def total_roses := roses_from_parents + (roses_per_friend * friends)

-- Prove that the total number of roses Bella received is 44
theorem bella_total_roses : total_roses = 44 := 
by
  sorry

end bella_total_roses_l44_44614


namespace least_possible_k_l44_44051

-- Define the conditions
def prime_factor_form (k : ℕ) : Prop :=
  ∃ a b c : ℕ, a > 0 ∧ b > 0 ∧ c > 0 ∧ k = 2^a * 3^b * 5^c

def divisible_by_1680 (k : ℕ) : Prop :=
  (k ^ 4) % 1680 = 0

-- Define the proof problem
theorem least_possible_k (k : ℕ) (h_div : divisible_by_1680 k) (h_prime : prime_factor_form k) : k = 210 :=
by
  -- Statement of the problem, proof to be filled
  sorry

end least_possible_k_l44_44051


namespace union_of_A_and_B_l44_44003

def A := {x : ℝ | ∃ y : ℝ, y = Real.log (x - 3)}
def B := {y : ℝ | ∃ (x : ℝ), y = Real.exp x}

theorem union_of_A_and_B : A ∪ B = {x : ℝ | x > 0} := by
sorry

end union_of_A_and_B_l44_44003


namespace benny_seashells_l44_44114

-- Define the initial number of seashells Benny found
def seashells_found : ℝ := 66.5

-- Define the percentage of seashells Benny gave away
def percentage_given_away : ℝ := 0.75

-- Calculate the number of seashells Benny gave away
def seashells_given_away : ℝ := percentage_given_away * seashells_found

-- Calculate the number of seashells Benny now has
def seashells_left : ℝ := seashells_found - seashells_given_away

-- Prove that Benny now has 16.625 seashells
theorem benny_seashells : seashells_left = 16.625 :=
by
  sorry

end benny_seashells_l44_44114


namespace longest_side_l44_44864

theorem longest_side (l w : ℝ) 
  (h1 : 2 * l + 2 * w = 240)
  (h2 : l * w = 2880) :
  l = 86.835 ∨ w = 86.835 :=
sorry

end longest_side_l44_44864


namespace valid_license_plates_count_l44_44023

-- Defining the total number of choices for letters and digits
def num_letter_choices := 26
def num_digit_choices := 10

-- Function to calculate the total number of valid license plates
def total_license_plates := num_letter_choices ^ 3 * num_digit_choices ^ 4

-- The proof statement
theorem valid_license_plates_count : total_license_plates = 175760000 := 
by 
  -- The placeholder for the proof
  sorry

end valid_license_plates_count_l44_44023


namespace Math_Proof_Problem_l44_44485

noncomputable def problem : ℝ := (1005^3) / (1003 * 1004) - (1003^3) / (1004 * 1005)

theorem Math_Proof_Problem : ⌊ problem ⌋ = 8 :=
by
  sorry

end Math_Proof_Problem_l44_44485


namespace janet_initial_number_l44_44448

-- Define the conditions using Lean definitions
def janetProcess (x : ℕ) : ℕ :=
  (2 * (x + 7)) - 4

-- The theorem that expresses the statement of the problem: If the final result of the process is 28, then x = 9
theorem janet_initial_number (x : ℕ) (h : janetProcess x = 28) : x = 9 :=
sorry

end janet_initial_number_l44_44448


namespace angle_bisector_length_is_5_l44_44699

open Real

noncomputable def triangleAngleBisectorLength (a b c : ℝ) : ℝ :=
  sqrt (a * b * (1 - (c * c) / ((a + b) * (a + b))))

theorem angle_bisector_length_is_5 :
  ∀ (A B C : ℝ), A = 20 ∧ C = 40 ∧ (b - c = 5) →
  triangleAngleBisectorLength a (2 * a * cos (A * π / 180) + 5) (2 * a * cos (A * π / 180)) = 5 :=
  by
  -- you can skip this part with sorry
  sorry

end angle_bisector_length_is_5_l44_44699


namespace ellipse_product_l44_44512

noncomputable def a (b : ℝ) := b + 4
noncomputable def AB (a: ℝ) := 2 * a
noncomputable def CD (b: ℝ) := 2 * b

theorem ellipse_product:
  (∀ (a b : ℝ), a = b + 4 → a^2 - b^2 = 64) →
  (∃ (a b : ℝ), (AB a) * (CD b) = 240) :=
by
  intros h
  use 10, 6
  simp [AB, CD]
  sorry

end ellipse_product_l44_44512


namespace ferris_wheel_rides_l44_44461

theorem ferris_wheel_rides :
  let people_per_20_minutes := 70
  let operation_duration_hours := 6
  let minutes_per_hour := 60
  let operation_duration_minutes := operation_duration_hours * minutes_per_hour
  let times_per_hour := minutes_per_hour / 20
  let total_people_per_hour := times_per_hour * people_per_20_minutes
  let total_people := total_people_per_hour * operation_duration_hours
  total_people = 1260 :=
by
  let people_per_20_minutes := 70
  let operation_duration_hours := 6
  let minutes_per_hour := 60
  let operation_duration_minutes := operation_duration_hours * minutes_per_hour
  let times_per_hour := minutes_per_hour / 20
  let total_people_per_hour := times_per_hour * people_per_20_minutes
  let total_people := total_people_per_hour * operation_duration_hours
  have : total_people = 1260 := by sorry
  exact this

end ferris_wheel_rides_l44_44461


namespace percent_preferred_apples_l44_44435

def frequencies : List ℕ := [75, 80, 45, 100, 50]
def frequency_apples : ℕ := 75
def total_frequency : ℕ := frequency_apples + frequencies[1] + frequencies[2] + frequencies[3] + frequencies[4]

theorem percent_preferred_apples :
  (frequency_apples * 100) / total_frequency = 21 := by
  -- Proof steps go here
  sorry

end percent_preferred_apples_l44_44435


namespace orchids_initially_three_l44_44309

-- Define initial number of roses and provided number of orchids in the vase
def initial_roses : ℕ := 9
def added_orchids (O : ℕ) : ℕ := 13
def added_roses : ℕ := 3
def difference := 10

-- Define initial number of orchids that we need to prove
def initial_orchids (O : ℕ) : Prop :=
  added_orchids O - added_roses = difference →
  O = 3

theorem orchids_initially_three :
  initial_orchids O :=
sorry

end orchids_initially_three_l44_44309


namespace find_n_l44_44468

theorem find_n (n : ℕ) (M : ℕ) (A : ℕ) 
  (hM : M = n - 11) 
  (hA : A = n - 2) 
  (hM_ge_one : M ≥ 1) 
  (hA_ge_one : A ≥ 1) 
  (hM_plus_A_lt_n : M + A < n) : 
  n = 12 := 
by 
  sorry

end find_n_l44_44468


namespace number_of_students_l44_44701

theorem number_of_students (n : ℕ)
  (h1 : ∃ n, (175 * n) / n = 175)
  (h2 : 175 * n - 40 = 173 * n) :
  n = 20 :=
sorry

end number_of_students_l44_44701


namespace number_of_red_squares_in_19th_row_l44_44086

-- Define the number of squares in the n-th row
def number_of_squares (n : ℕ) : ℕ := 3 * n - 1

-- Define the number of red squares in the n-th row
def red_squares (n : ℕ) : ℕ := (number_of_squares n) / 2

-- The theorem stating the problem
theorem number_of_red_squares_in_19th_row : red_squares 19 = 28 := by
  -- Proof goes here
  sorry

end number_of_red_squares_in_19th_row_l44_44086


namespace power_calculation_l44_44324

theorem power_calculation (a : ℝ) (m n : ℝ) (h1 : a^m = 2) (h2 : a^n = 5) : a^(3*m + 2*n) = 200 := by
  sorry

end power_calculation_l44_44324


namespace solution_set_inequality_k_l44_44479

theorem solution_set_inequality_k (k : ℚ) :
  (∀ x : ℚ, 3 * x - (2 * k - 3) < 4 * x + 3 * k + 6 ↔ x > 1) → k = -4/5 :=
by
  sorry

end solution_set_inequality_k_l44_44479


namespace complex_multiply_cis_l44_44955

open Complex

theorem complex_multiply_cis :
  (4 * (cos (25 * Real.pi / 180) + sin (25 * Real.pi / 180) * I)) *
  (-3 * (cos (48 * Real.pi / 180) + sin (48 * Real.pi / 180) * I)) =
  12 * (cos (253 * Real.pi / 180) + sin (253 * Real.pi / 180) * I) :=
sorry

end complex_multiply_cis_l44_44955


namespace two_zeros_of_cubic_polynomial_l44_44717

theorem two_zeros_of_cubic_polynomial (m : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ -x1^3 + 3*x1 + m = 0 ∧ -x2^3 + 3*x2 + m = 0) →
  (m = -2 ∨ m = 2) :=
by
  sorry

end two_zeros_of_cubic_polynomial_l44_44717


namespace sin_double_angle_value_l44_44892

open Real

theorem sin_double_angle_value (x : ℝ) (h : sin (x + π / 4) = - 5 / 13) : sin (2 * x) = - 119 / 169 := 
sorry

end sin_double_angle_value_l44_44892


namespace power_simplification_l44_44777

noncomputable def sqrt2_six : ℝ := 6 ^ (1 / 2)
noncomputable def sqrt3_six : ℝ := 6 ^ (1 / 3)

theorem power_simplification :
  (sqrt2_six / sqrt3_six) = 6 ^ (1 / 6) :=
  sorry

end power_simplification_l44_44777


namespace original_ghee_quantity_l44_44782

theorem original_ghee_quantity (x : ℝ) (H1 : 0.60 * x + 10 = ((1 + 0.40 * x) * 0.80)) :
  x = 10 :=
sorry

end original_ghee_quantity_l44_44782


namespace find_integer_n_l44_44314

theorem find_integer_n : ∃ (n : ℤ), (-90 ≤ n ∧ n ≤ 90) ∧ (Real.sin (n * Real.pi / 180) = Real.cos (456 * Real.pi / 180)) ∧ n = -6 := 
by
  sorry

end find_integer_n_l44_44314


namespace find_a_l44_44385

noncomputable def A (a : ℝ) : Set ℝ := {a + 2, (a + 1) ^ 2, a ^ 2 + 3 * a + 3}

theorem find_a (a : ℝ) (h : 1 ∈ A a) : a = 0 :=
by
  sorry

end find_a_l44_44385


namespace solve_arithmetic_sequence_l44_44098

variable {a : ℕ → ℝ}
variable {d a1 a2 a3 a10 a11 a6 a7 : ℝ}

axiom arithmetic_seq (n : ℕ) : a (n + 1) = a1 + n * d

def arithmetic_condition (h : a 2 + a 3 + a 10 + a 11 = 32) : Prop :=
  a 6 + a 7 = 16

theorem solve_arithmetic_sequence (h : a 2 + a 3 + a 10 + a 11 = 32) : a 6 + a 7 = 16 :=
  by
    -- Proof will go here
    sorry

end solve_arithmetic_sequence_l44_44098


namespace common_external_tangent_b_l44_44742

def circle1_center := (1, 3)
def circle1_radius := 3
def circle2_center := (10, 6)
def circle2_radius := 7

theorem common_external_tangent_b :
  ∃ (b : ℝ), ∀ (m : ℝ), m = 3 / 4 ∧ b = 9 / 4 := sorry

end common_external_tangent_b_l44_44742


namespace triangle_area_example_l44_44176

noncomputable def area_triangle (BC AB : ℝ) (B : ℝ) : ℝ :=
  (1 / 2) * BC * AB * Real.sin B

theorem triangle_area_example
  (BC AB : ℝ) (B : ℝ)
  (hBC : BC = 2)
  (hAB : AB = 3)
  (hB : B = Real.pi / 3) :
  area_triangle BC AB B = (3 * Real.sqrt 3) / 2 :=
by
  sorry

end triangle_area_example_l44_44176


namespace functional_equation_zero_l44_44907

open Function

theorem functional_equation_zero (f : ℕ+ → ℝ) 
  (h : ∀ (m n : ℕ+), n ≥ m → f (n + m) + f (n - m) = f (3 * n)) :
  ∀ n : ℕ+, f n = 0 := sorry

end functional_equation_zero_l44_44907


namespace ratio_of_pq_l44_44692

def is_pure_imaginary (z : Complex) : Prop :=
  z.re = 0

theorem ratio_of_pq (p q : ℝ) (hp : p ≠ 0) (hq : q ≠ 0)
  (H : is_pure_imaginary ((Complex.ofReal 3 - Complex.ofReal 4 * Complex.I) * (Complex.ofReal p + Complex.ofReal q * Complex.I))) :
  p / q = -4 / 3 :=
by
  sorry

end ratio_of_pq_l44_44692


namespace max_cables_cut_l44_44504

/-- 
Prove that given 200 computers connected by 345 cables initially forming a single cluster, after 
cutting cables to form 8 clusters, the maximum possible number of cables that could have been 
cut is 153.
--/
theorem max_cables_cut (computers : ℕ) (initial_cables : ℕ) (final_clusters : ℕ) (initial_clusters : ℕ) 
  (minimal_cables : ℕ) (cuts : ℕ) : 
  computers = 200 ∧ initial_cables = 345 ∧ final_clusters = 8 ∧ initial_clusters = 1 ∧ 
  minimal_cables = computers - final_clusters ∧ 
  cuts = initial_cables - minimal_cables →
  cuts = 153 := 
sorry

end max_cables_cut_l44_44504


namespace product_of_primes_l44_44630

theorem product_of_primes : 5 * 7 * 997 = 34895 :=
by
  sorry

end product_of_primes_l44_44630


namespace train_length_l44_44921

theorem train_length (L : ℝ) (h1 : L + 110 / 15 = (L + 250) / 20) : L = 310 := 
sorry

end train_length_l44_44921


namespace trapezium_area_proof_l44_44713

def trapeziumArea (a b h : ℕ) : ℕ :=
  (1 / 2) * (a + b) * h

theorem trapezium_area_proof :
  let a := 20
  let b := 18
  let h := 14
  trapeziumArea a b h = 266 := by
  sorry

end trapezium_area_proof_l44_44713


namespace sums_same_remainder_exists_l44_44125

theorem sums_same_remainder_exists (n : ℕ) (h : n > 0) (a : Fin (2 * n) → Fin (2 * n)) (ha_permutation : Function.Bijective a) :
  ∃ (i j : Fin (2 * n)), i ≠ j ∧ ((a i + i) % (2 * n) = (a j + j) % (2 * n)) :=
by sorry

end sums_same_remainder_exists_l44_44125


namespace haleys_current_height_l44_44573

-- Define the conditions
def growth_rate : ℕ := 3
def years : ℕ := 10
def future_height : ℕ := 50

-- Define the proof problem
theorem haleys_current_height : (future_height - growth_rate * years) = 20 :=
by {
  -- This is where the actual proof would go
  sorry
}

end haleys_current_height_l44_44573


namespace jakes_present_weight_l44_44397

theorem jakes_present_weight (J S : ℕ) (h1 : J - 32 = 2 * S) (h2 : J + S = 212) : J = 152 :=
by
  sorry

end jakes_present_weight_l44_44397


namespace negation_example_l44_44102

theorem negation_example :
  (¬ ∀ x : ℝ, x^2 ≥ 0) ↔ (∃ x : ℝ, x^2 < 0) := sorry

end negation_example_l44_44102


namespace arithmetic_sequence_n_l44_44817

theorem arithmetic_sequence_n (a1 d an n : ℕ) (h1 : a1 = 1) (h2 : d = 3) (h3 : an = 298) (h4 : an = a1 + (n - 1) * d) : n = 100 :=
by
  sorry

end arithmetic_sequence_n_l44_44817


namespace dave_non_working_games_l44_44184

def total_games : ℕ := 10
def price_per_game : ℕ := 4
def total_earnings : ℕ := 32

theorem dave_non_working_games : (total_games - (total_earnings / price_per_game)) = 2 := by
  sorry

end dave_non_working_games_l44_44184


namespace M_intersection_N_l44_44405

-- Define the sets M and N
def M : Set ℝ := {x | 0 ≤ x ∧ x ≤ 1}
def N : Set ℝ := {x | -1 < x ∧ x < 1}

-- Define the proof problem
theorem M_intersection_N : M ∩ N = {x | 0 ≤ x ∧ x < 1} := by
  sorry

end M_intersection_N_l44_44405


namespace value_of_a_l44_44106

theorem value_of_a (x a : ℤ) (h : x = 4) (h_eq : 5 * (x - 1) - 3 * a = -3) : a = 6 :=
by {
  sorry
}

end value_of_a_l44_44106


namespace minimum_value_of_sum_l44_44392

variable (x y : ℝ)

theorem minimum_value_of_sum (hx : x > 0) (hy : y > 0) : ∃ x y, x > 0 ∧ y > 0 ∧ (x + 2 * y) = 9 :=
sorry

end minimum_value_of_sum_l44_44392


namespace Terry_driving_speed_is_40_l44_44710

-- Conditions
def distance_home_to_workplace : ℕ := 60
def total_time_driving : ℕ := 3

-- Computation for total distance
def total_distance := distance_home_to_workplace * 2

-- Desired speed computation
def driving_speed := total_distance / total_time_driving

-- Problem statement to prove
theorem Terry_driving_speed_is_40 : driving_speed = 40 :=
by 
  sorry -- proof not required as per instructions

end Terry_driving_speed_is_40_l44_44710


namespace trains_cross_time_l44_44954

theorem trains_cross_time (length1 length2 : ℕ) (time1 time2 : ℕ) 
  (speed1 speed2 relative_speed total_length : ℚ) 
  (h1 : length1 = 120) (h2 : length2 = 150) 
  (h3 : time1 = 10) (h4 : time2 = 15) 
  (h5 : speed1 = length1 / time1) (h6 : speed2 = length2 / time2) 
  (h7 : relative_speed = speed1 - speed2) 
  (h8 : total_length = length1 + length2) : 
  (total_length / relative_speed = 135) := 
by sorry

end trains_cross_time_l44_44954


namespace simplify_polynomial_l44_44562

theorem simplify_polynomial (x : ℝ) : 
  (3 - 5 * x - 7 * x^2 + 9 + 11 * x - 13 * x^2 - 15 + 17 * x + 19 * x^2) = (-x^2 + 23 * x - 3) := 
by
  sorry

end simplify_polynomial_l44_44562


namespace original_quantity_of_ghee_l44_44433

theorem original_quantity_of_ghee (Q : ℝ) (h1 : 0.6 * Q = 9) (h2 : 0.4 * Q = 6) (h3 : 0.4 * Q = 0.2 * (Q + 10)) : Q = 10 :=
by sorry

end original_quantity_of_ghee_l44_44433


namespace fish_caught_l44_44498

theorem fish_caught (x y : ℕ) 
  (h1 : y - 2 = 4 * (x + 2))
  (h2 : y - 6 = 2 * (x + 6)) :
  x = 4 ∧ y = 26 :=
by
  sorry

end fish_caught_l44_44498


namespace negation_of_p_equiv_l44_44307

-- Define the initial proposition p
def p : Prop := ∃ x : ℝ, x^2 - 5*x - 6 < 0

-- State the theorem for the negation of p
theorem negation_of_p_equiv : ¬p ↔ ∀ x : ℝ, x^2 - 5*x - 6 ≥ 0 :=
by
  sorry

end negation_of_p_equiv_l44_44307


namespace root_shifted_is_root_of_quadratic_with_integer_coeffs_l44_44679

theorem root_shifted_is_root_of_quadratic_with_integer_coeffs
  (a b c t : ℤ)
  (h : a ≠ 0)
  (h_root : a * t^2 + b * t + c = 0) :
  ∃ (x : ℤ), a * x^2 + (4 * a + b) * x + (4 * a + 2 * b + c) = 0 :=
by {
  sorry
}

end root_shifted_is_root_of_quadratic_with_integer_coeffs_l44_44679


namespace find_r_s_l44_44165

theorem find_r_s (r s : ℚ) :
  (-3)^5 - 2*(-3)^4 + 3*(-3)^3 - r*(-3)^2 + s*(-3) - 8 = 0 ∧
  2^5 - 2*(2^4) + 3*(2^3) - r*(2^2) + s*2 - 8 = 0 →
  (r, s) = (-482/15, -1024/15) :=
by
  sorry

end find_r_s_l44_44165


namespace age_of_third_boy_l44_44384

theorem age_of_third_boy (a b c : ℕ) (h1 : a = 9) (h2 : b = 9) (h_sum : a + b + c = 29) : c = 11 :=
by
  sorry

end age_of_third_boy_l44_44384


namespace eric_age_l44_44434

theorem eric_age (B E : ℕ) (h1 : B = E + 4) (h2 : B + E = 28) : E = 12 :=
by
  sorry

end eric_age_l44_44434


namespace pills_needed_for_week_l44_44302

def pill_mg : ℕ := 50 -- Each pill has 50 mg of Vitamin A.
def recommended_daily_mg : ℕ := 200 -- The recommended daily serving of Vitamin A is 200 mg.
def days_in_week : ℕ := 7 -- There are 7 days in a week.

theorem pills_needed_for_week : (recommended_daily_mg / pill_mg) * days_in_week = 28 := 
by 
  sorry

end pills_needed_for_week_l44_44302


namespace zoe_remaining_pictures_l44_44238

-- Definitions based on the conditions
def total_pictures : Nat := 88
def colored_pictures : Nat := 20

-- Proof statement
theorem zoe_remaining_pictures : total_pictures - colored_pictures = 68 := by
  sorry

end zoe_remaining_pictures_l44_44238


namespace fuse_length_must_be_80_l44_44792

-- Define the basic conditions
def distanceToSafeArea : ℕ := 400
def personSpeed : ℕ := 5
def fuseBurnSpeed : ℕ := 1

-- Calculate the time required to reach the safe area
def timeToSafeArea (distance speed : ℕ) : ℕ := distance / speed

-- Calculate the minimum length of the fuse based on the time to reach the safe area
def minFuseLength (time burnSpeed : ℕ) : ℕ := time * burnSpeed

-- The main problem statement: The fuse must be at least 80 meters long.
theorem fuse_length_must_be_80:
  minFuseLength (timeToSafeArea distanceToSafeArea personSpeed) fuseBurnSpeed = 80 :=
by
  sorry

end fuse_length_must_be_80_l44_44792


namespace function_f_not_all_less_than_half_l44_44879

theorem function_f_not_all_less_than_half (p q : ℝ) (f : ℝ → ℝ) (h1 : ∀ x : ℝ, f x = x^2 + p*x + q) :
  ¬ (|f 1| < 1 / 2 ∧ |f 2| < 1 / 2 ∧ |f 3| < 1 / 2) :=
sorry

end function_f_not_all_less_than_half_l44_44879


namespace product_divisible_by_5_l44_44339

theorem product_divisible_by_5 (a b : ℕ) (ha : a > 0) (hb : b > 0)
  (h : ∃ k, a * b = 5 * k) : a % 5 = 0 ∨ b % 5 = 0 :=
by
  sorry

end product_divisible_by_5_l44_44339


namespace sum_of_first_n_terms_sequence_l44_44329

open Nat

def sequence_term (i : ℕ) : ℚ :=
  if i = 0 then 0 else 1 / (i * (i + 1) / 2 : ℕ)

def sum_of_sequence (n : ℕ) : ℚ :=
  (Finset.range (n+1)).sum fun i => sequence_term i

theorem sum_of_first_n_terms_sequence (n : ℕ) : sum_of_sequence n = 2 * n / (n + 1) := by
  sorry

end sum_of_first_n_terms_sequence_l44_44329


namespace find_number_l44_44608

variable (number : ℤ)

theorem find_number (h : number - 44 = 15) : number = 59 := 
sorry

end find_number_l44_44608


namespace part1_part2_l44_44418

noncomputable def f (a x : ℝ) : ℝ := (a * Real.exp x - a - x) * Real.exp x

theorem part1 (a : ℝ) (h0 : a ≥ 0) (h1 : ∀ x : ℝ, f a x ≥ 0) : a = 1 := 
sorry

theorem part2 (h1 : ∀ x : ℝ, f 1 x ≥ 0) :
  ∃! x0 : ℝ, (∀ x : ℝ, x0 = x → 
  (f 1 x0) = (f 1 x)) ∧ (0 < f 1 x0 ∧ f 1 x0 < 1/4) :=
sorry

end part1_part2_l44_44418


namespace hotel_charge_percentage_l44_44025

theorem hotel_charge_percentage (G R P : ℝ) 
  (hR : R = 1.60 * G) 
  (hP : P = 0.80 * G) : 
  ((R - P) / R) * 100 = 50 := by
  sorry

end hotel_charge_percentage_l44_44025


namespace semicircle_radius_l44_44088

theorem semicircle_radius (P : ℝ) (r : ℝ) (h₁ : P = π * r + 2 * r) (h₂ : P = 198) :
  r = 198 / (π + 2) :=
sorry

end semicircle_radius_l44_44088


namespace sector_area_l44_44323

theorem sector_area (r : ℝ) (α : ℝ) (h_r : r = 2) (h_α : α = π / 4) :
  1/2 * r^2 * α = π / 2 :=
by
  subst h_r
  subst h_α
  sorry

end sector_area_l44_44323


namespace scientific_notation_of_750000_l44_44962

theorem scientific_notation_of_750000 : 750000 = 7.5 * 10^5 :=
by
  sorry

end scientific_notation_of_750000_l44_44962


namespace ratio_cookies_to_pie_l44_44046

def num_surveyed_students : ℕ := 800
def num_students_preferred_cookies : ℕ := 280
def num_students_preferred_pie : ℕ := 160

theorem ratio_cookies_to_pie : num_students_preferred_cookies / num_students_preferred_pie = 7 / 4 := by
  sorry

end ratio_cookies_to_pie_l44_44046


namespace part1_tangent_line_max_min_values_l44_44858

def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a * x^2
def f' (a : ℝ) (x : ℝ) : ℝ := 3 * x^2 + 2 * a * x
def tangent_line_at (a : ℝ) (x y : ℝ) : ℝ := 9 * x + y - 4

theorem part1 (a : ℝ) : f' a 1 = -9 → a = -6 :=
by
  sorry

theorem tangent_line (a : ℝ) (x y : ℝ) : a = -6 → f a 1 = -5 → tangent_line_at a 1 (-5) = 0 :=
by
  sorry

def interval := Set.Icc (-5 : ℝ) 5

theorem max_min_values (a : ℝ) : a = -6 →
  (∀ x ∈ interval, f a (-5) = -275 ∨ f a 0 = 0 ∨ f a 4 = -32 ∨ f a 5 = -25) →
  (∀ x ∈ interval, f a x ≤ 0 ∧ f a x ≥ -275) :=
by
  sorry

end part1_tangent_line_max_min_values_l44_44858


namespace evaluate_expression_l44_44221

theorem evaluate_expression : -30 + 5 * (9 / (3 + 3)) = -22.5 := sorry

end evaluate_expression_l44_44221


namespace sales_worth_l44_44414

variable (S : ℝ)
def old_remuneration (S : ℝ) : ℝ := 0.05 * S
def new_remuneration (S : ℝ) : ℝ := 1300 + 0.025 * (S - 4000)

theorem sales_worth :
  new_remuneration S = old_remuneration S + 600 → S = 24000 :=
by
  intro h
  sorry

end sales_worth_l44_44414


namespace min_distance_to_line_l44_44945

theorem min_distance_to_line (m n : ℝ) (h : 4 * m + 3 * n = 10)
  : m^2 + n^2 ≥ 4 :=
sorry

end min_distance_to_line_l44_44945


namespace measles_cases_1993_l44_44152

theorem measles_cases_1993 :
  ∀ (cases_1970 cases_1986 cases_2000 : ℕ)
    (rate1 rate2 : ℕ),
  cases_1970 = 600000 →
  cases_1986 = 30000 →
  cases_2000 = 600 →
  rate1 = 35625 →
  rate2 = 2100 →
  cases_1986 - 7 * rate2 = 15300 :=
by {
  sorry
}

end measles_cases_1993_l44_44152


namespace product_of_ratios_l44_44042

theorem product_of_ratios 
  (x1 y1 x2 y2 x3 y3 : ℝ)
  (hx1 : x1^3 - 3 * x1 * y1^2 = 2005)
  (hy1 : y1^3 - 3 * x1^2 * y1 = 2004)
  (hx2 : x2^3 - 3 * x2 * y2^2 = 2005)
  (hy2 : y2^3 - 3 * x2^2 * y2 = 2004)
  (hx3 : x3^3 - 3 * x3 * y3^2 = 2005)
  (hy3 : y3^3 - 3 * x3^2 * y3 = 2004) :
  (1 - x1/y1) * (1 - x2/y2) * (1 - x3/y3) = 1/1002 := 
sorry

end product_of_ratios_l44_44042


namespace solve_for_y_l44_44915

theorem solve_for_y (y : ℝ) (h : (2 / y) + (3 / y) / (6 / y) = 1.5) : y = 2 :=
sorry

end solve_for_y_l44_44915


namespace average_age_when_youngest_born_l44_44108

theorem average_age_when_youngest_born (n : ℕ) (avg_age current_youngest_age total_age_when_youngest_born : ℝ) 
  (h1 : n = 7) (h2 : avg_age = 30) (h3 : current_youngest_age = 8) (h4 : total_age_when_youngest_born = (n * avg_age - n * current_youngest_age)) : 
  total_age_when_youngest_born / n = 22 :=
by
  sorry

end average_age_when_youngest_born_l44_44108


namespace find_expression_for_f_l44_44841

noncomputable def f (x a b : ℝ) : ℝ := (x + a) * (b * x + 2 * a)

-- Assuming a, b ∈ ℝ, f(x) is even, and range of f(x) is (-∞, 2]
theorem find_expression_for_f (a b : ℝ) (h1 : ∀ x : ℝ, f x a b = f (-x) a b) (h2 : ∀ y : ℝ, ∃ x : ℝ, f x a b = y → y ≤ 2):
  f x a b = -x^2 + 2 :=
by 
  sorry

end find_expression_for_f_l44_44841


namespace paul_books_left_l44_44475
-- Add the necessary imports

-- Define the initial conditions
def initial_books : ℕ := 115
def books_sold : ℕ := 78

-- Statement of the problem as a theorem
theorem paul_books_left : (initial_books - books_sold) = 37 := by
  -- Proof omitted
  sorry

end paul_books_left_l44_44475


namespace leak_time_l44_44951

theorem leak_time (A L : ℝ) (PipeA_filling_rate : A = 1 / 6) (Combined_rate : A - L = 1 / 10) : 
  1 / L = 15 :=
by
  sorry

end leak_time_l44_44951


namespace benches_count_l44_44857

theorem benches_count (num_people_base6 : ℕ) (people_per_bench : ℕ) (num_people_base10 : ℕ) (num_benches : ℕ) :
  num_people_base6 = 204 ∧ people_per_bench = 2 ∧ num_people_base10 = 76 ∧ num_benches = 38 →
  (num_people_base10 = 2 * 6^2 + 0 * 6^1 + 4 * 6^0) ∧
  (num_benches = num_people_base10 / people_per_bench) :=
by
  sorry

end benches_count_l44_44857


namespace frac_add_eq_l44_44757

theorem frac_add_eq : (2 / 5) + (3 / 10) = 7 / 10 := 
by
  sorry

end frac_add_eq_l44_44757


namespace electricity_consumption_l44_44180

variable (x y : ℝ)

-- y = 0.55 * x
def electricity_fee := 0.55 * x

-- if y = 40.7 then x should be 74
theorem electricity_consumption :
  (∃ x, electricity_fee x = 40.7) → (x = 74) :=
by
  sorry

end electricity_consumption_l44_44180


namespace cost_of_cherries_l44_44090

theorem cost_of_cherries (total_spent amount_for_grapes amount_for_cherries : ℝ)
  (h1 : total_spent = 21.93)
  (h2 : amount_for_grapes = 12.08)
  (h3 : amount_for_cherries = total_spent - amount_for_grapes) :
  amount_for_cherries = 9.85 :=
sorry

end cost_of_cherries_l44_44090


namespace six_star_three_l44_44464

-- Define the mathematical operation.
def operation (r t : ℝ) : ℝ := sorry

axiom condition_1 (r : ℝ) : operation r 0 = r^2
axiom condition_2 (r t : ℝ) : operation r t = operation t r
axiom condition_3 (r t : ℝ) : operation (r + 1) t = operation r t + 2 * t + 1

-- Prove that 6 * 3 = 75 given the conditions.
theorem six_star_three : operation 6 3 = 75 := by
  sorry

end six_star_three_l44_44464


namespace large_circle_diameter_l44_44863

theorem large_circle_diameter (r : ℝ) (R : ℝ) (R' : ℝ) :
  r = 2 ∧ R = 2 * r ∧ R' = R + r → 2 * R' = 12 :=
by
  intros h
  sorry

end large_circle_diameter_l44_44863


namespace minimize_total_cost_l44_44634

noncomputable def event_probability_without_measures : ℚ := 0.3
noncomputable def loss_if_event_occurs : ℚ := 4000000
noncomputable def cost_measure_A : ℚ := 450000
noncomputable def prob_event_not_occurs_measure_A : ℚ := 0.9
noncomputable def cost_measure_B : ℚ := 300000
noncomputable def prob_event_not_occurs_measure_B : ℚ := 0.85

noncomputable def total_cost_no_measures : ℚ :=
  event_probability_without_measures * loss_if_event_occurs

noncomputable def total_cost_measure_A : ℚ :=
  cost_measure_A + (1 - prob_event_not_occurs_measure_A) * loss_if_event_occurs

noncomputable def total_cost_measure_B : ℚ :=
  cost_measure_B + (1 - prob_event_not_occurs_measure_B) * loss_if_event_occurs

noncomputable def total_cost_measures_A_and_B : ℚ :=
  cost_measure_A + cost_measure_B + (1 - prob_event_not_occurs_measure_A) * (1 - prob_event_not_occurs_measure_B) * loss_if_event_occurs

theorem minimize_total_cost :
  min (min total_cost_no_measures total_cost_measure_A) (min total_cost_measure_B total_cost_measures_A_and_B) = total_cost_measures_A_and_B :=
by sorry

end minimize_total_cost_l44_44634


namespace marathon_yards_l44_44528

theorem marathon_yards (miles_per_marathon : ℕ) (extra_yards_per_marathon : ℕ) (yards_per_mile : ℕ) (num_marathons : ℕ)
  (total_miles : ℕ) (total_yards : ℕ) 
  (H1 : miles_per_marathon = 26) 
  (H2 : extra_yards_per_marathon = 395) 
  (H3 : yards_per_mile = 1760) 
  (H4 : num_marathons = 15) 
  (H5 : total_miles = num_marathons * miles_per_marathon + (num_marathons * extra_yards_per_marathon) / yards_per_mile)
  (H6 : total_yards = (num_marathons * extra_yards_per_marathon) % yards_per_mile)
  (H7 : 0 ≤ total_yards ∧ total_yards < yards_per_mile) 
  : total_yards = 645 :=
sorry

end marathon_yards_l44_44528


namespace initial_percentage_of_alcohol_l44_44831

theorem initial_percentage_of_alcohol 
  (P: ℝ)
  (h_condition1 : 18 * P / 100 = 21 * 17.14285714285715 / 100) : 
  P = 20 :=
by 
  sorry

end initial_percentage_of_alcohol_l44_44831


namespace find_k_l44_44922

theorem find_k (k r s : ℝ) 
  (h1 : r + s = -k) 
  (h2 : r * s = 12) 
  (h3 : (r + 7) + (s + 7) = k) : 
  k = 7 := by 
  sorry

end find_k_l44_44922


namespace ricky_time_difference_l44_44676

noncomputable def old_man_time_per_mile : ℚ := 300 / 8
noncomputable def young_man_time_per_mile : ℚ := 160 / 12
noncomputable def time_difference : ℚ := old_man_time_per_mile - young_man_time_per_mile

theorem ricky_time_difference :
  time_difference = 24 := by
sorry

end ricky_time_difference_l44_44676


namespace sunflower_is_taller_l44_44631

def sister_height_ft : Nat := 4
def sister_height_in : Nat := 3
def sunflower_height_ft : Nat := 6

def feet_to_inches (ft : Nat) : Nat := ft * 12

def sister_height := feet_to_inches sister_height_ft + sister_height_in
def sunflower_height := feet_to_inches sunflower_height_ft

def height_difference : Nat := sunflower_height - sister_height

theorem sunflower_is_taller : height_difference = 21 :=
by
  -- proof has to be provided:
  sorry

end sunflower_is_taller_l44_44631


namespace f_inequality_l44_44028

-- Define the function f.
def f (x : ℝ) : ℝ := x^2 - x + 13

-- The main theorem to prove the given inequality.
theorem f_inequality (x m : ℝ) (h : |x - m| < 1) : |f x - f m| < 2*(|m| + 1) :=
by
  sorry

end f_inequality_l44_44028


namespace two_a7_minus_a8_l44_44535

variable (a : ℕ → ℝ) -- Assuming the arithmetic sequence {a_n} is a sequence of real numbers

-- Definitions and conditions of the problem
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) - a n = a 1 - a 0

axiom a1_plus_3a6_plus_a11 : a 1 + 3 * (a 6) + a 11 = 120

-- The theorem to be proved
theorem two_a7_minus_a8 (h : is_arithmetic_sequence a) : 2 * a 7 - a 8 = 24 := 
sorry

end two_a7_minus_a8_l44_44535


namespace tall_mirror_passes_l44_44232

theorem tall_mirror_passes (T : ℕ)
    (s_tall_ref : ℕ)
    (s_wide_ref : ℕ)
    (e_tall_ref : ℕ)
    (e_wide_ref : ℕ)
    (wide_passes : ℕ)
    (total_reflections : ℕ)
    (H1 : s_tall_ref = 10)
    (H2 : s_wide_ref = 5)
    (H3 : e_tall_ref = 6)
    (H4 : e_wide_ref = 3)
    (H5 : wide_passes = 5)
    (H6 : s_tall_ref * T + s_wide_ref * wide_passes + e_tall_ref * T + e_wide_ref * wide_passes = 88) : 
    T = 3 := 
by sorry

end tall_mirror_passes_l44_44232


namespace max_profit_achieved_at_180_l44_44736

-- Definitions:
def cost (x : ℝ) : ℝ := 0.1 * x^2 - 11 * x + 3000  -- Condition 1
def selling_price_per_unit : ℝ := 25  -- Condition 2

-- Statement to prove that the maximum profit is achieved at x = 180
theorem max_profit_achieved_at_180 :
  ∃ (S : ℝ), ∀ (x : ℝ),
    S = -0.1 * (x - 180)^2 + 240 → S = 25 * 180 - cost 180 :=
by
  sorry

end max_profit_achieved_at_180_l44_44736


namespace complement_U_A_l44_44197

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def A : Set ℕ := {3, 4, 5}

theorem complement_U_A :
  U \ A = {1, 2, 6} := by
  sorry

end complement_U_A_l44_44197


namespace fewer_spoons_l44_44727

/--
Stephanie initially planned to buy 15 pieces of each type of silverware.
There are 4 types of silverware.
This totals to 60 pieces initially planned to be bought.
She only bought 44 pieces in total.
Show that she decided to purchase 4 fewer spoons.
-/
theorem fewer_spoons
  (initial_total : ℕ := 60)
  (final_total : ℕ := 44)
  (types : ℕ := 4)
  (pieces_per_type : ℕ := 15) :
  (initial_total - final_total) / types = 4 := 
by
  -- since initial_total = 60, final_total = 44, and types = 4
  -- we need to prove (60 - 44) / 4 = 4
  sorry

end fewer_spoons_l44_44727


namespace prime_exponent_condition_l44_44364

theorem prime_exponent_condition (p a n : ℕ) (hp : Nat.Prime p) (ha : 0 < a) (hn : 0 < n)
  (h : 2^p + 3^p = a^n) : n = 1 :=
sorry

end prime_exponent_condition_l44_44364


namespace total_wheels_in_parking_lot_l44_44254

def num_cars : ℕ := 14
def num_bikes : ℕ := 10
def wheels_per_car : ℕ := 4
def wheels_per_bike : ℕ := 2

theorem total_wheels_in_parking_lot :
  (num_cars * wheels_per_car) + (num_bikes * wheels_per_bike) = 76 :=
by
  sorry

end total_wheels_in_parking_lot_l44_44254


namespace probability_of_same_type_l44_44070

-- Definitions for the given conditions
def total_books : ℕ := 12 + 9
def novels : ℕ := 12
def biographies : ℕ := 9

-- Define the number of ways to pick any two books
def total_ways_to_pick_two_books : ℕ := Nat.choose total_books 2

-- Define the number of ways to pick two novels
def ways_to_pick_two_novels : ℕ := Nat.choose novels 2

-- Define the number of ways to pick two biographies
def ways_to_pick_two_biographies : ℕ := Nat.choose biographies 2

-- Define the number of ways to pick two books of the same type
def ways_to_pick_two_books_of_same_type : ℕ := ways_to_pick_two_novels + ways_to_pick_two_biographies

-- Calculate the probability
noncomputable def probability_same_type (total_ways ways_same_type : ℕ) : ℚ :=
  ways_same_type / total_ways

theorem probability_of_same_type :
  probability_same_type total_ways_to_pick_two_books ways_to_pick_two_books_of_same_type = 17 / 35 := by
  sorry

end probability_of_same_type_l44_44070


namespace cost_of_potatoes_l44_44661

theorem cost_of_potatoes
  (per_person_potatoes : ℕ → ℕ → ℕ)
  (amount_of_people : ℕ)
  (bag_cost : ℕ)
  (bag_weight : ℕ)
  (people : ℕ)
  (cost : ℕ) :
  (per_person_potatoes people amount_of_people = 60) →
  (60 / bag_weight = 3) →
  (3 * bag_cost = cost) →
  cost = 15 :=
by
  sorry

end cost_of_potatoes_l44_44661


namespace isosceles_triangle_perimeter_l44_44303

def is_isosceles (a b c : ℝ) : Prop :=
  a = b ∨ b = c ∨ c = a

def is_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

noncomputable def perimeter (a b c : ℝ) : ℝ :=
  a + b + c

theorem isosceles_triangle_perimeter (a b c : ℝ) (h1 : is_isosceles a b c) (h2 : is_triangle a b c) (h3 : a = 4 ∨ a = 9) (h4 : b = 4 ∨ b = 9) :
  perimeter a b c = 22 :=
  sorry

end isosceles_triangle_perimeter_l44_44303


namespace largest_modulus_z_l44_44969

open Complex

noncomputable def z_largest_value (a b c z : ℂ) : ℝ :=
  (1 + Real.sqrt 5) / 2

theorem largest_modulus_z (a b c z : ℂ) (r : ℝ) (hr_pos : 0 < r)
  (hmod_a : Complex.abs a = r) (hmod_b : Complex.abs b = r) (hmod_c : Complex.abs c = r)
  (heqn : a * z ^ 2 + b * z + c = 0) :
  Complex.abs z ≤ z_largest_value a b c z :=
sorry

end largest_modulus_z_l44_44969


namespace exponential_ordering_l44_44446

noncomputable def a := (0.4:ℝ)^(0.3:ℝ)
noncomputable def b := (0.3:ℝ)^(0.4:ℝ)
noncomputable def c := (0.3:ℝ)^(-0.2:ℝ)

theorem exponential_ordering : b < a ∧ a < c := by
  sorry

end exponential_ordering_l44_44446


namespace total_birds_and_storks_l44_44417

theorem total_birds_and_storks (initial_birds initial_storks additional_storks : ℕ) 
  (h1 : initial_birds = 3) 
  (h2 : initial_storks = 4) 
  (h3 : additional_storks = 6) 
  : initial_birds + initial_storks + additional_storks = 13 := 
  by sorry

end total_birds_and_storks_l44_44417


namespace midpoint_3d_l44_44651

/-- Midpoint calculation in 3D space -/
theorem midpoint_3d (x1 y1 z1 x2 y2 z2 : ℝ) : 
  (x1, y1, z1) = (2, -3, 6) → 
  (x2, y2, z2) = (8, 5, -4) → 
  ((x1 + x2) / 2, (y1 + y2) / 2, (z1 + z2) / 2) = (5, 1, 1) := 
by
  intros
  sorry

end midpoint_3d_l44_44651


namespace proof_problem_l44_44010

noncomputable def M : Set ℝ := { x | x ≥ 2 }
noncomputable def a : ℝ := Real.pi

theorem proof_problem : a ∈ M ∧ {a} ⊂ M :=
by
  sorry

end proof_problem_l44_44010


namespace rectangle_perimeter_l44_44313

theorem rectangle_perimeter (breadth length : ℝ) (h1 : length = 3 * breadth) (h2 : length * breadth = 147) : 2 * length + 2 * breadth = 56 :=
by
  sorry

end rectangle_perimeter_l44_44313


namespace hyperbola_eccentricity_l44_44360

theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (2 : ℝ) = 2 →
  a^2 = 2 * b^2 →
  (c : ℝ) = Real.sqrt (a^2 + b^2) →
  Real.sqrt (a^2 + b^2) = Real.sqrt (3 / 2 * a^2) →
  (e : ℝ) = c / a →
  e = Real.sqrt (6) / 2 :=
by
  sorry

end hyperbola_eccentricity_l44_44360


namespace gcd_1722_966_l44_44541

theorem gcd_1722_966 : Nat.gcd 1722 966 = 42 :=
  sorry

end gcd_1722_966_l44_44541


namespace sqrt_t6_plus_t4_eq_t2_sqrt_t2_plus_1_l44_44443

theorem sqrt_t6_plus_t4_eq_t2_sqrt_t2_plus_1 (t : ℝ) : 
  Real.sqrt (t^6 + t^4) = t^2 * Real.sqrt (t^2 + 1) :=
sorry

end sqrt_t6_plus_t4_eq_t2_sqrt_t2_plus_1_l44_44443


namespace dave_winfield_home_runs_l44_44953

theorem dave_winfield_home_runs (W : ℕ) (h : 755 = 2 * W - 175) : W = 465 :=
by
  sorry

end dave_winfield_home_runs_l44_44953


namespace maria_high_school_students_l44_44673

variable (M D : ℕ)

theorem maria_high_school_students (h1 : M = 4 * D) (h2 : M - D = 1800) : M = 2400 :=
by
  sorry

end maria_high_school_students_l44_44673


namespace sister_ages_l44_44816

theorem sister_ages (x y : ℕ) (h1 : x = y + 4) (h2 : x^3 - y^3 = 988) : y = 7 ∧ x = 11 :=
by
  sorry

end sister_ages_l44_44816


namespace sum_local_values_l44_44994

theorem sum_local_values :
  let local_value_2 := 2000
  let local_value_3 := 300
  let local_value_4 := 40
  let local_value_5 := 5
  local_value_2 + local_value_3 + local_value_4 + local_value_5 = 2345 :=
by
  sorry

end sum_local_values_l44_44994


namespace regular_21_gon_symmetry_calculation_l44_44505

theorem regular_21_gon_symmetry_calculation:
  let L := 21
  let R := 360 / 21
  L + R = 38 :=
by
  sorry

end regular_21_gon_symmetry_calculation_l44_44505


namespace carrots_total_l44_44002
-- import the necessary library

-- define the conditions as given
def sandy_carrots : Nat := 6
def sam_carrots : Nat := 3

-- state the problem as a theorem to be proven
theorem carrots_total : sandy_carrots + sam_carrots = 9 := by
  sorry

end carrots_total_l44_44002


namespace sum_a_b_when_pow_is_max_l44_44004

theorem sum_a_b_when_pow_is_max (a b : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 1) (h_pow : a^b < 500) 
(h_max : ∀ (a' b' : ℕ), (a' > 0) -> (b' > 1) -> (a'^b' < 500) -> a^b >= a'^b') : a + b = 24 := by
  sorry

end sum_a_b_when_pow_is_max_l44_44004


namespace remainder_div_2DD_l44_44424

theorem remainder_div_2DD' (P D D' Q R Q' R' : ℕ) 
  (h1 : P = Q * D + R) 
  (h2 : Q = 2 * D' * Q' + R') :
  P % (2 * D * D') = D * R' + R :=
sorry

end remainder_div_2DD_l44_44424


namespace not_divisible_by_n_plus_4_l44_44778

theorem not_divisible_by_n_plus_4 (n : ℕ) (h : 0 < n) : ¬ (n + 4 ∣ n^2 + 8 * n + 15) := 
sorry

end not_divisible_by_n_plus_4_l44_44778


namespace distinct_socks_pairs_l44_44563

theorem distinct_socks_pairs (n : ℕ) (h : n = 9) : (Nat.choose n 2) = 36 := by
  rw [h]
  norm_num
  sorry

end distinct_socks_pairs_l44_44563


namespace first_term_of_geometric_series_l44_44669

theorem first_term_of_geometric_series (r : ℚ) (S : ℚ) (a : ℚ) (h1 : r = 1/5) (h2 : S = 100) (h3 : S = a / (1 - r)) : a = 80 := 
by
  sorry

end first_term_of_geometric_series_l44_44669


namespace ratio_b_to_c_l44_44353

variable (a b c k : ℕ)

-- Conditions
def condition1 : Prop := a = b + 2
def condition2 : Prop := b = k * c
def condition3 : Prop := a + b + c = 32
def condition4 : Prop := b = 12

-- Question: Prove that ratio of b to c is 2:1
theorem ratio_b_to_c
  (h1 : condition1 a b)
  (h2 : condition2 b k c)
  (h3 : condition3 a b c)
  (h4 : condition4 b) :
  b = 2 * c := 
sorry

end ratio_b_to_c_l44_44353


namespace total_bottles_remaining_is_14090_l44_44666

-- Define the constants
def total_small_bottles : ℕ := 5000
def total_big_bottles : ℕ := 12000
def small_bottles_sold_percentage : ℕ := 15
def big_bottles_sold_percentage : ℕ := 18

-- Define the remaining bottles
def calc_remaining_bottles (total_bottles sold_percentage : ℕ) : ℕ :=
  total_bottles - (sold_percentage * total_bottles / 100)

-- Define the remaining small and big bottles
def remaining_small_bottles : ℕ := calc_remaining_bottles total_small_bottles small_bottles_sold_percentage
def remaining_big_bottles : ℕ := calc_remaining_bottles total_big_bottles big_bottles_sold_percentage

-- Define the total remaining bottles
def total_remaining_bottles : ℕ := remaining_small_bottles + remaining_big_bottles

-- State the theorem
theorem total_bottles_remaining_is_14090 : total_remaining_bottles = 14090 := by
  sorry

end total_bottles_remaining_is_14090_l44_44666


namespace question_1_question_2_l44_44741

def f (x : ℝ) : ℝ := |x + 1| - |x - 4|

theorem question_1 (m : ℝ) :
  (∀ x : ℝ, f x ≤ -m^2 + 6 * m) ↔ (1 ≤ m ∧ m ≤ 5) :=
by
  sorry

theorem question_2 (a b c : ℝ) (h1 : 3 * a + 4 * b + 5 * c = 5) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a^2 + b^2 + c^2 ≥ 1 / 2 :=
by
  sorry

end question_1_question_2_l44_44741


namespace new_cube_edge_length_l44_44870

theorem new_cube_edge_length
  (a1 a2 a3 : ℝ)
  (h1 : a1 = 3) 
  (h2 : a2 = 4) 
  (h3 : a3 = 5) :
  (a1^3 + a2^3 + a3^3)^(1/3) = 6 := by
sorry

end new_cube_edge_length_l44_44870


namespace height_of_Brixton_l44_44350

theorem height_of_Brixton
  (I Z B Zr : ℕ)
  (h1 : I = Z + 4)
  (h2 : Z = B - 8)
  (h3 : Zr = B)
  (h4 : (I + Z + B + Zr) / 4 = 61) :
  B = 64 := by
  sorry

end height_of_Brixton_l44_44350


namespace inequality_proof_l44_44674

variable (a b c d : ℝ)

theorem inequality_proof (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : d > 0) (h5 : a + b + c + d = 1) : 
  (1 / (4 * a + 3 * b + c) + 1 / (3 * a + b + 4 * d) + 1 / (a + 4 * c + 3 * d) + 1 / (4 * b + 3 * c + d)) ≥ 2 :=
by
  sorry

end inequality_proof_l44_44674


namespace james_new_fuel_cost_l44_44919

def original_cost : ℕ := 200
def price_increase_rate : ℕ := 20
def extra_tank_factor : ℕ := 2

theorem james_new_fuel_cost :
  let new_price := original_cost + (price_increase_rate * original_cost / 100)
  let total_cost := extra_tank_factor * new_price
  total_cost = 480 :=
by
  sorry

end james_new_fuel_cost_l44_44919


namespace infinite_set_divisor_l44_44728

open Set

noncomputable def exists_divisor (A : Set ℕ) : Prop :=
  ∃ (d : ℕ), d > 1 ∧ ∀ (a : ℕ), a ∈ A → d ∣ a

theorem infinite_set_divisor (A : Set ℕ) (hA1 : ∀ (b : Finset ℕ), (↑b ⊆ A) → ∃ (d : ℕ), d > 1 ∧ ∀ (a : ℕ), a ∈ b → d ∣ a) :
  exists_divisor A :=
sorry

end infinite_set_divisor_l44_44728


namespace Doris_spent_6_l44_44670

variable (D : ℝ)

theorem Doris_spent_6 (h0 : 24 - (D + D / 2) = 15) : D = 6 :=
by
  sorry

end Doris_spent_6_l44_44670


namespace f_m_eq_five_l44_44253

def f (x : ℝ) (a : ℝ) : ℝ :=
  x^3 + a * x + 3

axiom f_neg_m : ∀ (m a : ℝ), f (-m) a = 1

theorem f_m_eq_five (m a : ℝ) (h : f (-m) a = 1) : f m a = 5 :=
  by sorry

end f_m_eq_five_l44_44253


namespace central_angle_of_cone_l44_44920

theorem central_angle_of_cone (A : ℝ) (l : ℝ) (r : ℝ) (θ : ℝ)
  (hA : A = (1 / 2) * 2 * Real.pi * r)
  (hl : l = 1)
  (ha : A = (3 / 8) * Real.pi) :
  θ = (3 / 4) * Real.pi :=
by
  sorry

end central_angle_of_cone_l44_44920


namespace jack_pays_back_l44_44557

-- Define the principal amount P and interest rate r
def principal_amount : ℝ := 1200
def interest_rate : ℝ := 0.1

-- Define the interest and the total amount Jack has to pay back
def interest : ℝ := interest_rate * principal_amount
def total_amount : ℝ := principal_amount + interest

-- State the theorem to prove that the total amount Jack pays back is 1320
theorem jack_pays_back : total_amount = 1320 := by
  sorry

end jack_pays_back_l44_44557


namespace Jaymee_is_22_l44_44734

-- Define Shara's age
def Shara_age : ℕ := 10

-- Define Jaymee's age according to the problem conditions
def Jaymee_age : ℕ := 2 + 2 * Shara_age

-- The proof statement to show that Jaymee's age is 22
theorem Jaymee_is_22 : Jaymee_age = 22 := by 
  -- The proof is omitted according to the instructions.
  sorry

end Jaymee_is_22_l44_44734


namespace max_value_of_expression_l44_44080

theorem max_value_of_expression (A M C : ℕ) (h : A + M + C = 15) :
  A * M * C + A * M + M * C + C * A ≤ 200 :=
sorry

end max_value_of_expression_l44_44080


namespace car_price_is_5_l44_44695

variable (numCars : ℕ) (totalEarnings legoCost carCost : ℕ)

-- Conditions
axiom h1 : numCars = 3
axiom h2 : totalEarnings = 45
axiom h3 : legoCost = 30
axiom h4 : totalEarnings - legoCost = 15
axiom h5 : (totalEarnings - legoCost) / numCars = carCost

-- The proof problem statement
theorem car_price_is_5 : carCost = 5 :=
  by
    -- Here the proof steps would be filled in, but are not required for this task.
    sorry

end car_price_is_5_l44_44695


namespace average_visitors_per_day_l44_44553

theorem average_visitors_per_day (average_sunday : ℕ) (average_other : ℕ) (days_in_month : ℕ) (begins_with_sunday : Bool) :
  average_sunday = 600 → average_other = 240 → days_in_month = 30 → begins_with_sunday = true → (8640 / 30 = 288) :=
by
  intros h1 h2 h3 h4
  sorry

end average_visitors_per_day_l44_44553


namespace acute_triangle_l44_44218

theorem acute_triangle (a b c : ℝ) (h : a^π + b^π = c^π) : a^2 + b^2 > c^2 := sorry

end acute_triangle_l44_44218


namespace average_of_remaining_two_numbers_l44_44081

theorem average_of_remaining_two_numbers 
  (a b c d e f : ℝ)
  (h1 : (a + b + c + d + e + f) / 6 = 4.60)
  (h2 : (a + b) / 2 = 3.4)
  (h3 : (c + d) / 2 = 3.8) :
  ((e + f) / 2) = 6.6 :=
sorry

end average_of_remaining_two_numbers_l44_44081


namespace angle_P_measure_l44_44912

theorem angle_P_measure (P Q : ℝ) (h1 : P + Q = 180) (h2 : P = 5 * Q) : P = 150 := by
  sorry

end angle_P_measure_l44_44912


namespace sum_of_possible_values_of_a_l44_44883

theorem sum_of_possible_values_of_a :
  ∀ (a b c d : ℝ), a > b → b > c → c > d → a + b + c + d = 50 → 
  (a - b = 4 ∧ b - d = 7 ∧ a - c = 5 ∧ c - d = 6 ∧ b - c = 2 ∨
   a - b = 5 ∧ b - d = 6 ∧ a - c = 4 ∧ c - d = 7 ∧ b - c = 2) →
  (a = 17.75 ∨ a = 18.25) →
  a + 18.25 + 17.75 - a = 36 :=
by sorry

end sum_of_possible_values_of_a_l44_44883


namespace pipe_fill_time_without_leak_l44_44166

theorem pipe_fill_time_without_leak (T : ℝ) (h1 : (1 / 9 : ℝ) = 1 / T - 1 / 4.5) : T = 3 := 
by
  sorry

end pipe_fill_time_without_leak_l44_44166


namespace perpendicular_condition_l44_44171

-- Definitions of lines
def line1 (x y : ℝ) : Prop := x + y = 0
def line2 (x y : ℝ) (a : ℝ) : Prop := x - a * y = 0

-- Theorem: Prove that a = 1 is a necessary and sufficient condition for the lines
-- line1 and line2 to be perpendicular.
theorem perpendicular_condition (a : ℝ) : 
  (∀ x y : ℝ, line1 x y → line2 x y a) ↔ (a = 1) :=
sorry

end perpendicular_condition_l44_44171


namespace ratio_proof_l44_44963

noncomputable def ratio_of_segment_lengths (a b : ℝ) (points : Finset (ℝ × ℝ)) : Prop :=
  points.card = 5 ∧
  ∃ (dists : Finset ℝ), 
    dists = {a, a, a, a, a, b, 3 * a} ∧
    ∀ (p1 p2 : ℝ × ℝ), p1 ∈ points → p2 ∈ points → p1 ≠ p2 →
      (dist p1 p2 ∈ dists)

theorem ratio_proof (a b : ℝ) (points : Finset (ℝ × ℝ)) (h : ratio_of_segment_lengths a b points) : 
  b / a = 2.8 :=
sorry

end ratio_proof_l44_44963


namespace neg_p_l44_44859

variable (x : ℝ)

def p : Prop := ∃ x : ℝ, Real.exp x - x - 1 ≤ 0

theorem neg_p : ¬p ↔ (∀ x : ℝ, Real.exp x - x - 1 > 0) :=
by
  sorry

end neg_p_l44_44859


namespace sum_series_eq_1_div_300_l44_44719

noncomputable def sum_series : ℝ :=
  ∑' n, (6 * (n:ℝ) + 1) / ((6 * (n:ℝ) - 1) ^ 2 * (6 * (n:ℝ) + 5) ^ 2)

theorem sum_series_eq_1_div_300 : sum_series = 1 / 300 :=
  sorry

end sum_series_eq_1_div_300_l44_44719


namespace max_value_ln_x_plus_x_l44_44585

theorem max_value_ln_x_plus_x (x : ℝ) (h1 : 1 ≤ x) (h2 : x ≤ Real.exp 1) : 
  ∃ y, y = Real.log x + x ∧ y ≤ Real.log (Real.exp 1) + Real.exp 1 :=
sorry

end max_value_ln_x_plus_x_l44_44585


namespace min_value_z_l44_44099

variable (x y : ℝ)

theorem min_value_z : ∃ (x y : ℝ), 2 * x + 3 * y = 9 :=
sorry

end min_value_z_l44_44099


namespace value_of_M_l44_44279

theorem value_of_M (M : ℝ) (h : 0.2 * M = 500) : M = 2500 :=
by
  sorry

end value_of_M_l44_44279


namespace least_multiple_of_13_gt_450_l44_44604

theorem least_multiple_of_13_gt_450 : ∃ (n : ℕ), (455 = 13 * n) ∧ 455 > 450 ∧ ∀ m : ℕ, (13 * m > 450) → 455 ≤ 13 * m :=
by
  sorry

end least_multiple_of_13_gt_450_l44_44604


namespace smallest_base_b_l44_44006

theorem smallest_base_b (b : ℕ) : (b ≥ 1) → (b^2 ≤ 82) → (82 < b^3) → b = 5 := by
  sorry

end smallest_base_b_l44_44006


namespace arithmetic_expression_evaluation_l44_44423

theorem arithmetic_expression_evaluation :
  (3 + 9) ^ 2 + (3 ^ 2) * (9 ^ 2) = 873 :=
by
  -- Proof is skipped, using sorry for now.
  sorry

end arithmetic_expression_evaluation_l44_44423


namespace exists_line_l_l44_44506

-- Define the parabola and line l1
def parabola (P : ℝ × ℝ) : Prop := P.2^2 = 8 * P.1
def line_l1 (P : ℝ × ℝ) : Prop := P.1 + 5 * P.2 - 5 = 0

-- Define the problem statement
theorem exists_line_l :
  ∃ l : ℝ × ℝ → Prop, 
    ((∃ A B : ℝ × ℝ, parabola A ∧ parabola B ∧ A ≠ B ∧ l A ∧ l B) ∧
    (∃ M : ℝ × ℝ, M = (1, 4/5) ∧ line_l1 M) ∧
    (∀ A B : ℝ × ℝ, l A ∧ l B → (A.2 - B.2) / (A.1 - B.1) = 5)) ∧
    (∀ P : ℝ × ℝ, l P ↔ 25 * P.1 - 5 * P.2 - 21 = 0) :=
sorry

end exists_line_l_l44_44506


namespace find_y_l44_44960

theorem find_y (x y : ℝ) (hA : {2, Real.log x} = {a | a = 2 ∨ a = Real.log x})
                (hB : {x, y} = {a | a = x ∨ a = y})
                (hInt : {a | a = 2 ∨ a = Real.log x} ∩ {a | a = x ∨ a = y} = {0}) :
  y = 0 :=
  sorry

end find_y_l44_44960


namespace fraction_of_number_l44_44156

variable (N : ℝ) (F : ℝ)

theorem fraction_of_number (h1 : 0.5 * N = F * N + 2) (h2 : N = 8.0) : F = 0.25 := by
  sorry

end fraction_of_number_l44_44156


namespace Tammy_runs_10_laps_per_day_l44_44188

theorem Tammy_runs_10_laps_per_day
  (total_distance_per_week : ℕ)
  (track_length : ℕ)
  (days_per_week : ℕ)
  (h1 : total_distance_per_week = 3500)
  (h2 : track_length = 50)
  (h3 : days_per_week = 7) :
  (total_distance_per_week / track_length) / days_per_week = 10 := by
  sorry

end Tammy_runs_10_laps_per_day_l44_44188


namespace supplement_of_angle_with_given_complement_l44_44474

theorem supplement_of_angle_with_given_complement (θ : ℝ) (h : 90 - θ = 50) : 180 - θ = 140 :=
by sorry

end supplement_of_angle_with_given_complement_l44_44474


namespace salary_increase_percentage_l44_44170

variable {P : ℝ} (initial_salary : P > 0)

def salary_after_first_year (P: ℝ) : ℝ :=
  P * 1.12

def salary_after_second_year (P: ℝ) : ℝ :=
  (salary_after_first_year P) * 1.12

def salary_after_third_year (P: ℝ) : ℝ :=
  (salary_after_second_year P) * 1.15

theorem salary_increase_percentage (P: ℝ) (h: P > 0) : 
  (salary_after_third_year P - P) / P * 100 = 44 :=
by 
  sorry

end salary_increase_percentage_l44_44170


namespace circle_area_irrational_if_rational_diameter_l44_44077

noncomputable def pi : ℝ := Real.pi

theorem circle_area_irrational_if_rational_diameter (d : ℚ) :
  ¬ ∃ (A : ℝ), A = pi * (d / 2)^2 ∧ (∃ (q : ℚ), A = q) :=
by
  sorry

end circle_area_irrational_if_rational_diameter_l44_44077


namespace negation_of_p_l44_44076

def proposition_p := ∃ x : ℝ, x ≥ 1 ∧ x^2 - x < 0

theorem negation_of_p : (∀ x : ℝ, x ≥ 1 → x^2 - x ≥ 0) :=
by
  sorry

end negation_of_p_l44_44076


namespace reward_function_conditions_l44_44888

theorem reward_function_conditions :
  (∀ x : ℝ, (10 ≤ x ∧ x ≤ 1000) → (y = x / 150 + 2 → y ≤ 90 ∧ y ≤ x / 5) → False) ∧
  (∃ a : ℕ, (∀ x : ℝ, (10 ≤ x ∧ x ≤ 1000) → (y = (10 * x - 3 * a) / (x + 2) → y ≤ 9 ∧ y ≤ x / 5)) ∧ (a = 328)) :=
by
  sorry

end reward_function_conditions_l44_44888


namespace complex_mul_eq_l44_44511

/-- Proof that the product of two complex numbers (1 + i) and (2 + i) is equal to (1 + 3i) -/
theorem complex_mul_eq (i : ℂ) (h_i_squared : i^2 = -1) : (1 + i) * (2 + i) = 1 + 3 * i :=
by
  -- The actual proof logic goes here.
  sorry

end complex_mul_eq_l44_44511


namespace total_spokes_in_garage_l44_44548

-- Definitions based on the problem conditions
def num_bicycles : ℕ := 4
def spokes_per_wheel : ℕ := 10
def wheels_per_bicycle : ℕ := 2

-- The goal is to prove the total number of spokes
theorem total_spokes_in_garage : (num_bicycles * wheels_per_bicycle * spokes_per_wheel) = 80 :=
by
    sorry

end total_spokes_in_garage_l44_44548


namespace tetrahedron_labeling_impossible_l44_44214

/-- Suppose each vertex of a tetrahedron needs to be labeled with an integer from 1 to 4, each integer being used exactly once.
We need to prove that there are no such arrangements in which the sum of the numbers on the vertices of each face is the same for all four faces.
Arrangements that can be rotated into each other are considered identical. -/
theorem tetrahedron_labeling_impossible :
  ∀ (label : Fin 4 → Fin 5) (h_unique : ∀ v1 v2 : Fin 4, v1 ≠ v2 → label v1 ≠ label v2),
  ∃ (sum_faces : ℕ), sum_faces = 7 ∧ sum_faces % 3 = 1 → False :=
by
  sorry

end tetrahedron_labeling_impossible_l44_44214


namespace compute_difference_of_squares_l44_44168

theorem compute_difference_of_squares :
  (23 + 15) ^ 2 - (23 - 15) ^ 2 = 1380 := by
  sorry

end compute_difference_of_squares_l44_44168


namespace complete_square_h_l44_44342

theorem complete_square_h (x h : ℝ) :
  (∃ a k : ℝ, 3 * x^2 + 9 * x + 20 = a * (x - h)^2 + k) → h = -3 / 2 :=
by
  sorry

end complete_square_h_l44_44342


namespace team_a_wins_at_least_2_l44_44127

def team_a_wins_at_least (total_games lost_games : ℕ) (points : ℕ) (won_points draw_points lost_points : ℕ) : Prop :=
  ∃ (won_games : ℕ), 
    total_games = won_games + (total_games - lost_games - won_games) + lost_games ∧
    won_games * won_points + (total_games - lost_games - won_games) * draw_points > points ∧
    won_games ≥ 2

theorem team_a_wins_at_least_2 :
  team_a_wins_at_least 5 1 7 3 1 0 :=
by
  -- Proof goes here
  sorry

end team_a_wins_at_least_2_l44_44127


namespace u2008_is_5898_l44_44431

-- Define the sequence as given in the problem.
def u (n : ℕ) : ℕ := sorry  -- The nth term of the sequence defined in the problem.

-- The main theorem stating u_{2008} = 5898.
theorem u2008_is_5898 : u 2008 = 5898 := sorry

end u2008_is_5898_l44_44431


namespace largest_decimal_number_l44_44199

theorem largest_decimal_number :
  max (0.9123 : ℝ) (max (0.9912 : ℝ) (max (0.9191 : ℝ) (max (0.9301 : ℝ) (0.9091 : ℝ)))) = 0.9912 :=
by
  sorry

end largest_decimal_number_l44_44199


namespace composite_divisor_bound_l44_44187

theorem composite_divisor_bound (n : ℕ) (hn : ¬Prime n ∧ 1 < n) : 
  ∃ a : ℕ, 1 < a ∧ a ≤ Int.sqrt (n : ℤ) ∧ a ∣ n :=
sorry

end composite_divisor_bound_l44_44187


namespace problem_statement_l44_44497

def U := Set ℝ
def M := { x : ℝ | x^2 - 4 * x - 5 < 0 }
def N := { x : ℝ | 1 ≤ x }
def comp_U_N := { x : ℝ | x < 1 }
def intersection := { x : ℝ | -1 < x ∧ x < 1 }

theorem problem_statement : M ∩ comp_U_N = intersection := sorry

end problem_statement_l44_44497


namespace least_three_digit_multiple_of_3_4_9_is_108_l44_44454

theorem least_three_digit_multiple_of_3_4_9_is_108 :
  ∃ (n : ℕ), (100 ≤ n) ∧ (n % 3 = 0) ∧ (n % 4 = 0) ∧ (n % 9 = 0) ∧ (n = 108) :=
by
  sorry

end least_three_digit_multiple_of_3_4_9_is_108_l44_44454


namespace angles_in_triangle_l44_44304

theorem angles_in_triangle (A B C : ℝ) (h1 : A + B + C = 180) (h2 : 2 * B = 3 * A) (h3 : 5 * A = 2 * C) :
  B = 54 ∧ C = 90 :=
by
  sorry

end angles_in_triangle_l44_44304


namespace gcd_55555555_111111111_l44_44489

/-- Let \( m = 55555555 \) and \( n = 111111111 \).
We want to prove that the greatest common divisor (gcd) of \( m \) and \( n \) is 1. -/
theorem gcd_55555555_111111111 :
  let m := 55555555
  let n := 111111111
  Nat.gcd m n = 1 :=
by
  sorry

end gcd_55555555_111111111_l44_44489


namespace initial_average_production_l44_44836

theorem initial_average_production (n : ℕ) (today_production : ℕ) 
  (new_average : ℕ) (initial_average : ℕ) :
  n = 1 → today_production = 60 → new_average = 55 → initial_average = (new_average * (n + 1) - today_production) → initial_average = 50 :=
by
  intros h1 h2 h3 h4
  -- Insert further proof here
  sorry

end initial_average_production_l44_44836


namespace garden_area_difference_l44_44066

theorem garden_area_difference:
  (let length_rect := 60
   let width_rect := 20
   let perimeter_rect := 2 * (length_rect + width_rect)
   let side_square := perimeter_rect / 4
   let area_rect := length_rect * width_rect
   let area_square := side_square * side_square
   area_square - area_rect = 400) := 
by
  sorry

end garden_area_difference_l44_44066


namespace lock_combination_l44_44067

-- Define the digits as distinct
def distinct_digits (V E N U S I A R : ℕ) : Prop :=
  V ≠ E ∧ V ≠ N ∧ V ≠ U ∧ V ≠ S ∧ V ≠ I ∧ V ≠ A ∧ V ≠ R ∧
  E ≠ N ∧ E ≠ U ∧ E ≠ S ∧ E ≠ I ∧ E ≠ A ∧ E ≠ R ∧
  N ≠ U ∧ N ≠ S ∧ N ≠ I ∧ N ≠ A ∧ N ≠ R ∧
  U ≠ S ∧ U ≠ I ∧ U ≠ A ∧ U ≠ R ∧
  S ≠ I ∧ S ≠ A ∧ S ≠ R ∧
  I ≠ A ∧ I ≠ R ∧
  A ≠ R

-- Define the base 12 addition for the equation
def base12_addition (V E N U S I A R : ℕ) : Prop :=
  let VENUS := V * 12^4 + E * 12^3 + N * 12^2 + U * 12^1 + S
  let IS := I * 12^1 + S
  let NEAR := N * 12^3 + E * 12^2 + A * 12^1 + R
  let SUN := S * 12^2 + U * 12^1 + N
  VENUS + IS + NEAR = SUN

-- The theorem statement
theorem lock_combination :
  ∃ (V E N U S I A R : ℕ),
    distinct_digits V E N U S I A R ∧
    base12_addition V E N U S I A R ∧
    (S * 12^2 + U * 12^1 + N) = 655 := 
sorry

end lock_combination_l44_44067


namespace bacon_suggestion_count_l44_44395

theorem bacon_suggestion_count (B : ℕ) (h1 : 408 = B + 366) : B = 42 :=
by
  sorry

end bacon_suggestion_count_l44_44395


namespace original_price_of_computer_l44_44212

theorem original_price_of_computer :
  ∃ (P : ℝ), (1.30 * P = 377) ∧ (2 * P = 580) ∧ (P = 290) :=
by
  existsi (290 : ℝ)
  sorry

end original_price_of_computer_l44_44212


namespace geometric_sequence_arithmetic_condition_l44_44851

variable {a_n : ℕ → ℝ}
variable {q : ℝ}

-- Conditions of the problem
def is_geometric_sequence (a_n : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a_n (n + 1) = a_n n * q

def positive_terms (a_n : ℕ → ℝ) : Prop :=
  ∀ n, a_n n > 0

def arithmetic_sequence_cond (a_n : ℕ → ℝ) : Prop :=
  a_n 2 - (1 / 2) * a_n 3 = (1 / 2) * a_n 3 - a_n 1

-- Problem: Prove the required ratio equals the given value
theorem geometric_sequence_arithmetic_condition
  (h_geo: is_geometric_sequence a_n q)
  (h_pos: positive_terms a_n)
  (h_arith: arithmetic_sequence_cond a_n)
  (h_q_ne_one: q ≠ 1) :
  (a_n 4 + a_n 5) / (a_n 3 + a_n 4) = (1 + Real.sqrt 5) / 2 :=
sorry

end geometric_sequence_arithmetic_condition_l44_44851


namespace tan_of_45_deg_l44_44175

theorem tan_of_45_deg : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_of_45_deg_l44_44175


namespace units_digit_47_pow_47_l44_44078

theorem units_digit_47_pow_47 : (47 ^ 47) % 10 = 3 :=
by sorry

end units_digit_47_pow_47_l44_44078


namespace negation_of_universal_proposition_l44_44336

theorem negation_of_universal_proposition (x : ℝ) :
  ¬ (∀ m : ℝ, 0 ≤ m ∧ m ≤ 1 → x + 1 / x ≥ 2^m) ↔ ∃ m : ℝ, (0 ≤ m ∧ m ≤ 1) ∧ (x + 1 / x < 2^m) := by
  sorry

end negation_of_universal_proposition_l44_44336


namespace tangent_line_x_squared_l44_44411

theorem tangent_line_x_squared (P : ℝ × ℝ) (hP : P = (1, -1)) :
  ∃ (a : ℝ), a = 1 + Real.sqrt 2 ∨ a = 1 - Real.sqrt 2 ∧
    ((∀ x : ℝ, (2 * (1 + Real.sqrt 2) * x - (3 + 2 * Real.sqrt 2)) = P.2 → 
      P.2 = (2 * (1 + Real.sqrt 2) * P.1 - (3 + 2 * Real.sqrt 2))) ∨
    (∀ x : ℝ, (2 * (1 - Real.sqrt 2) * x - (3 - 2 * Real.sqrt 2)) = P.2 → 
      P.2 = (2 * (1 - Real.sqrt 2) * P.1 - (3 - 2 * Real.sqrt 2)))) := by
  sorry

end tangent_line_x_squared_l44_44411


namespace kellys_apples_l44_44843

def apples_kelly_needs_to_pick := 49
def total_apples := 105

theorem kellys_apples :
  ∃ x : ℕ, x + apples_kelly_needs_to_pick = total_apples ∧ x = 56 :=
sorry

end kellys_apples_l44_44843


namespace solve_for_x_l44_44189

theorem solve_for_x (x : ℝ) (y : ℝ) (h : y = 1) (h1 : y = 1 / (4 * x^2 + 2 * x + 1)) : 
  x = 0 ∨ x = -1/2 :=
by
  sorry

end solve_for_x_l44_44189


namespace unit_price_first_purchase_l44_44129

theorem unit_price_first_purchase (x y : ℝ) (h1 : x * y = 500000) 
    (h2 : 1.4 * x * (y + 10000) = 770000) : x = 5 :=
by
  -- Proof details here
  sorry

end unit_price_first_purchase_l44_44129


namespace geometric_series_first_term_l44_44639

theorem geometric_series_first_term (r a S : ℝ) (hr : r = 1 / 8) (hS : S = 60) (hS_formula : S = a / (1 - r)) : 
  a = 105 / 2 := by
  rw [hr, hS] at hS_formula
  sorry

end geometric_series_first_term_l44_44639


namespace garden_fencing_l44_44250

/-- A rectangular garden has a length of 50 yards and the width is half the length.
    Prove that the total amount of fencing needed to enclose the garden is 150 yards. -/
theorem garden_fencing : 
  ∀ (length width : ℝ), 
  length = 50 ∧ width = length / 2 → 
  2 * (length + width) = 150 :=
by
  intros length width
  rintro ⟨h1, h2⟩
  sorry

end garden_fencing_l44_44250


namespace fourth_vertex_parallelogram_coordinates_l44_44798

def fourth_vertex_of_parallelogram (A B C : ℝ × ℝ) :=
  ∃ D : ℝ × ℝ, (D = (11, 4) ∨ D = (-1, 12) ∨ D = (3, -12))

theorem fourth_vertex_parallelogram_coordinates :
  fourth_vertex_of_parallelogram (1, 0) (5, 8) (7, -4) :=
by
  sorry

end fourth_vertex_parallelogram_coordinates_l44_44798


namespace determine_m_l44_44378

theorem determine_m (a b : ℝ) (m : ℝ) :
  (a^2 + 2 * a * b - b^2) - (a^2 + m * a * b + 2 * b^2) = (2 - m) * a * b - 3 * b^2 →
  (∀ a b : ℝ, (2 - m) * a * b = 0) →
  m = 2 :=
by
  sorry

end determine_m_l44_44378


namespace triangle_first_side_l44_44627

theorem triangle_first_side (x : ℕ) (h1 : 10 + 15 + x = 32) : x = 7 :=
by
  sorry

end triangle_first_side_l44_44627


namespace determine_k_for_one_real_solution_l44_44706

theorem determine_k_for_one_real_solution (k : ℝ):
  (∃ x : ℝ, 9 * x^2 + k * x + 49 = 0 ∧ (∀ y : ℝ, 9 * y^2 + k * y + 49 = 0 → y = x)) → k = 42 :=
sorry

end determine_k_for_one_real_solution_l44_44706


namespace completing_square_l44_44649

-- Define the theorem statement
theorem completing_square (x : ℝ) : 
  x^2 - 2 * x = 2 -> (x - 1)^2 = 3 :=
by sorry

end completing_square_l44_44649


namespace product_of_consecutive_integers_is_square_l44_44990

theorem product_of_consecutive_integers_is_square (x : ℤ) : 
  x * (x + 1) * (x + 2) * (x + 3) + 1 = (x^2 + 3 * x + 1) ^ 2 :=
by
  sorry

end product_of_consecutive_integers_is_square_l44_44990


namespace polynomial_in_y_l44_44478

theorem polynomial_in_y {x y : ℝ} (h₁ : x^3 - 6 * x^2 + 11 * x - 6 = 0) (h₂ : y = x + 1/x) :
  x^2 * (y^2 + y - 6) = 0 :=
sorry

end polynomial_in_y_l44_44478


namespace workers_combined_time_l44_44403

theorem workers_combined_time (g_rate a_rate c_rate : ℝ)
  (hg : g_rate = 1 / 70)
  (ha : a_rate = 1 / 30)
  (hc : c_rate = 1 / 42) :
  1 / (g_rate + a_rate + c_rate) = 14 :=
by
  sorry

end workers_combined_time_l44_44403


namespace find_y_l44_44292

theorem find_y (x y : ℤ) (h1 : x^2 - 5 * x + 8 = y + 6) (h2 : x = -8) : y = 106 := by
  sorry

end find_y_l44_44292


namespace max_product_of_functions_l44_44517

theorem max_product_of_functions (f h : ℝ → ℝ) (hf : ∀ x, -5 ≤ f x ∧ f x ≤ 3) (hh : ∀ x, -3 ≤ h x ∧ h x ≤ 4) :
  ∃ x, f x * h x = 20 :=
by {
  sorry
}

end max_product_of_functions_l44_44517


namespace eleven_billion_in_scientific_notation_l44_44578

namespace ScientificNotation

def Yi : ℝ := 10 ^ 8

theorem eleven_billion_in_scientific_notation : (11 * (10 : ℝ) ^ 9) = (1.1 * (10 : ℝ) ^ 10) :=
by 
  sorry

end ScientificNotation

end eleven_billion_in_scientific_notation_l44_44578


namespace Donny_spends_28_on_Thursday_l44_44428

theorem Donny_spends_28_on_Thursday :
  let monday_savings := 15
  let tuesday_savings := 28
  let wednesday_savings := 13
  let total_savings := monday_savings + tuesday_savings + wednesday_savings
  let thursday_spending := total_savings / 2
  thursday_spending = 28 :=
by 
  let monday_savings := 15
  let tuesday_savings := 28
  let wednesday_savings := 13
  let total_savings := monday_savings + tuesday_savings + wednesday_savings
  let thursday_spending := total_savings / 2
  sorry

end Donny_spends_28_on_Thursday_l44_44428


namespace translation_correct_l44_44610

-- Define the first line l1
def l1 (x : ℝ) : ℝ := 2 * x - 2

-- Define the second line l2
def l2 (x : ℝ) : ℝ := 2 * x

-- State the theorem
theorem translation_correct :
  ∀ x : ℝ, l2 x = l1 x + 2 :=
by
  intro x
  unfold l1 l2
  sorry

end translation_correct_l44_44610


namespace percent_increase_is_equivalent_l44_44703

variable {P : ℝ}

theorem percent_increase_is_equivalent 
  (h1 : 1.0 + 15.0 / 100.0 = 1.15)
  (h2 : 1.15 * (1.0 + 25.0 / 100.0) = 1.4375)
  (h3 : 1.4375 * (1.0 + 10.0 / 100.0) = 1.58125) :
  (1.58125 - 1) * 100 = 58.125 :=
by
  sorry

end percent_increase_is_equivalent_l44_44703


namespace calc_fraction_l44_44110

variable {x y : ℝ}

theorem calc_fraction (h : x + y = x * y - 1) (hx : x ≠ 0) (hy : y ≠ 0) : 
  (1 / x) + (1 / y) = 1 - 1 / (x * y) := 
by 
  sorry

end calc_fraction_l44_44110


namespace range_of_a_l44_44887

-- Definitions of conditions
def is_odd_function {A : Type} [AddGroup A] (f : A → A) : Prop :=
  ∀ x, f (-x) = -f x

def is_monotonically_decreasing {A : Type} [LinearOrderedAddCommGroup A] (f : A → A) : Prop :=
  ∀ x y, x < y → f y ≤ f x

-- Main statement
theorem range_of_a 
  (f : ℝ → ℝ)
  (h_odd : is_odd_function f)
  (h_monotone_dec : is_monotonically_decreasing f)
  (h_domain : ∀ x, -7 < x ∧ x < 7 → -7 < f x ∧ f x < 7)
  (h_cond : ∀ a, f (1 - a) + f (2 * a - 5) < 0): 
  ∀ a, 4 < a → a < 6 :=
sorry

end range_of_a_l44_44887


namespace smallest_y_value_l44_44420

theorem smallest_y_value (y : ℚ) (h : y / 7 + 2 / (7 * y) = 1 / 3) : y = 2 / 3 :=
sorry

end smallest_y_value_l44_44420


namespace sin_half_angle_l44_44582

theorem sin_half_angle
  (theta : ℝ)
  (h1 : Real.sin theta = 3 / 5)
  (h2 : 5 * Real.pi / 2 < theta ∧ theta < 3 * Real.pi) :
  Real.sin (theta / 2) = - (3 * Real.sqrt 10 / 10) :=
by
  sorry

end sin_half_angle_l44_44582


namespace find_original_price_each_stocking_l44_44358

open Real

noncomputable def original_stocking_price (total_stockings total_cost_per_stocking discounted_cost monogramming_cost total_cost : ℝ) : ℝ :=
  let stocking_cost_before_monogramming := total_cost - (total_stockings * monogramming_cost)
  let original_price := stocking_cost_before_monogramming / (total_stockings * discounted_cost)
  original_price

theorem find_original_price_each_stocking :
  original_stocking_price 9 122.22 0.9 5 1035 = 122.22 := by
  sorry

end find_original_price_each_stocking_l44_44358


namespace golden_section_length_l44_44095

noncomputable def golden_section_point (a b : ℝ) := a / (a + b) = b / a

theorem golden_section_length (A B P : ℝ) (h : golden_section_point A P) (hAP_gt_PB : A > P) (hAB : A + P = 2) : 
  A = Real.sqrt 5 - 1 :=
by
  -- Proof goes here
  sorry

end golden_section_length_l44_44095


namespace vector_opposite_direction_and_magnitude_l44_44373

theorem vector_opposite_direction_and_magnitude
  (a : ℝ × ℝ) (b : ℝ × ℝ) 
  (h1 : a = (-1, 2)) 
  (h2 : ∃ k : ℝ, k < 0 ∧ b = k • a) 
  (hb : ‖b‖ = Real.sqrt 5) :
  b = (1, -2) :=
sorry

end vector_opposite_direction_and_magnitude_l44_44373


namespace cottonCandyToPopcornRatio_l44_44793

variable (popcornEarningsPerDay : ℕ) (netEarnings : ℕ) (rentCost : ℕ) (ingredientCost : ℕ)

theorem cottonCandyToPopcornRatio
  (h_popcorn : popcornEarningsPerDay = 50)
  (h_net : netEarnings = 895)
  (h_rent : rentCost = 30)
  (h_ingredient : ingredientCost = 75)
  (h : ∃ C : ℕ, 5 * C + 5 * popcornEarningsPerDay - rentCost - ingredientCost = netEarnings) :
  ∃ r : ℕ, r = 3 :=
by
  sorry

end cottonCandyToPopcornRatio_l44_44793


namespace Amy_total_crumbs_eq_3z_l44_44762

variable (T C z : ℕ)

-- Given conditions
def total_crumbs_Arthur := T * C = z
def trips_Amy := 2 * T
def crumbs_per_trip_Amy := 3 * C / 2

-- Problem statement
theorem Amy_total_crumbs_eq_3z (h : total_crumbs_Arthur T C z) :
  (trips_Amy T) * (crumbs_per_trip_Amy C) = 3 * z :=
sorry

end Amy_total_crumbs_eq_3z_l44_44762


namespace households_with_only_bike_l44_44033

theorem households_with_only_bike
  (N : ℕ) (H_no_car_or_bike : ℕ) (H_car_bike : ℕ) (H_car : ℕ)
  (hN : N = 90)
  (h_no_car_or_bike : H_no_car_or_bike = 11)
  (h_car_bike : H_car_bike = 16)
  (h_car : H_car = 44) :
  ∃ (H_bike_only : ℕ), H_bike_only = 35 :=
by {
  sorry
}

end households_with_only_bike_l44_44033


namespace sqrt_plus_inv_sqrt_eq_l44_44249

noncomputable def sqrt_plus_inv_sqrt (x : ℝ) : ℝ :=
  Real.sqrt x + 1 / Real.sqrt x

theorem sqrt_plus_inv_sqrt_eq (x : ℝ) (h₁ : 0 < x) (h₂ : x + 1 / x = 50) :
  sqrt_plus_inv_sqrt x = 2 * Real.sqrt 13 := by
  sorry

end sqrt_plus_inv_sqrt_eq_l44_44249


namespace find_precy_age_l44_44462

-- Defining the given conditions as Lean definitions
def alex_current_age : ℕ := 15
def alex_age_in_3_years : ℕ := alex_current_age + 3
def alex_age_a_year_ago : ℕ := alex_current_age - 1
axiom precy_current_age : ℕ
axiom in_3_years : alex_age_in_3_years = 3 * (precy_current_age + 3)
axiom a_year_ago : alex_age_a_year_ago = 7 * (precy_current_age - 1)

-- Stating the equivalent proof problem
theorem find_precy_age : precy_current_age = 3 :=
by
  sorry

end find_precy_age_l44_44462


namespace age_sum_proof_l44_44708

theorem age_sum_proof (a b c : ℕ) (h1 : a - (b + c) = 16) (h2 : a^2 - (b + c)^2 = 1632) : a + b + c = 102 :=
by
  sorry

end age_sum_proof_l44_44708


namespace directrix_of_parabola_l44_44442

theorem directrix_of_parabola :
  (∀ x y : ℝ, x^2 = 4 * y → ∃ y₀ : ℝ, y₀ = -1 ∧ ∀ y' : ℝ, y' = y₀) :=
by
  sorry

end directrix_of_parabola_l44_44442


namespace brick_length_correct_l44_44138

-- Define the constants
def courtyard_length_meters : ℝ := 25
def courtyard_width_meters : ℝ := 18
def courtyard_area_meters : ℝ := courtyard_length_meters * courtyard_width_meters
def bricks_number : ℕ := 22500
def brick_width_cm : ℕ := 10

-- We want to prove the length of each brick
def brick_length_cm : ℕ := 20

-- Convert courtyard area to square centimeters
def courtyard_area_cm : ℝ := courtyard_area_meters * 10000

-- Define the proof statement
theorem brick_length_correct :
  courtyard_area_cm = (brick_length_cm * brick_width_cm) * bricks_number :=
by
  sorry

end brick_length_correct_l44_44138


namespace best_k_k_l44_44560

theorem best_k_k' (v w x y z : ℝ) (hv : 0 < v) (hw : 0 < w) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  1 < (v / (v + w) + w / (w + x) + x / (x + y) + y / (y + z) + z / (z + v)) ∧ 
  (v / (v + w) + w / (w + x) + x / (x + y) + y / (y + z) + z / (z + v)) < 4 :=
sorry

end best_k_k_l44_44560


namespace P_subsetneq_M_l44_44174

def M := {x : ℝ | x > 1}
def P := {x : ℝ | x^2 - 6*x + 9 = 0}

theorem P_subsetneq_M : P ⊂ M := by
  sorry

end P_subsetneq_M_l44_44174


namespace injured_player_age_l44_44137

noncomputable def average_age_full_team := 22
noncomputable def number_of_players := 11
noncomputable def average_age_remaining_players := 21
noncomputable def number_of_remaining_players := 10
noncomputable def total_age_full_team := number_of_players * average_age_full_team
noncomputable def total_age_remaining_players := number_of_remaining_players * average_age_remaining_players

theorem injured_player_age :
  (number_of_players * average_age_full_team) -
  (number_of_remaining_players * average_age_remaining_players) = 32 :=
by
  sorry

end injured_player_age_l44_44137


namespace man_l44_44415

-- Define the conditions
def speed_downstream : ℕ := 8
def speed_upstream : ℕ := 4

-- Define the man's rate in still water
def rate_in_still_water : ℕ := (speed_downstream + speed_upstream) / 2

-- The target theorem
theorem man's_rate_in_still_water : rate_in_still_water = 6 := by
  -- The statement is set up. Proof to be added later.
  sorry

end man_l44_44415


namespace fraction_equality_l44_44677

theorem fraction_equality (a b : ℝ) (h : a / b = 2 / 3) : a / (a + b) = 2 / 5 :=
by
  sorry

end fraction_equality_l44_44677


namespace sum_of_two_numbers_eq_l44_44017

theorem sum_of_two_numbers_eq (x y : ℝ) (h1 : x * y = 16) (h2 : 1 / x = 3 * (1 / y)) : x + y = (16 * Real.sqrt 3) / 3 :=
by sorry

end sum_of_two_numbers_eq_l44_44017


namespace vector_parallel_l44_44426

variables {t : ℝ}

def a : ℝ × ℝ := (1, 3)
def b : ℝ × ℝ := (3, t)

theorem vector_parallel (h : (1 : ℝ) / (3 : ℝ) = (3 : ℝ) / t) : t = 9 :=
by 
  sorry

end vector_parallel_l44_44426


namespace product_of_two_numbers_ratio_l44_44581

theorem product_of_two_numbers_ratio {x y : ℝ}
  (h1 : x + y = (5/3) * (x - y))
  (h2 : x * y = 5 * (x - y)) :
  x * y = 56.25 := sorry

end product_of_two_numbers_ratio_l44_44581


namespace find_x_value_l44_44195

theorem find_x_value {C S x : ℝ}
  (h1 : C = 100 * (1 + x / 100))
  (h2 : S - C = 10 / 9)
  (h3 : S = 100 * (1 + x / 100)):
  x = 10 :=
by
  sorry

end find_x_value_l44_44195


namespace number_of_penny_piles_l44_44449

theorem number_of_penny_piles
    (piles_of_quarters : ℕ := 4) 
    (piles_of_dimes : ℕ := 6)
    (piles_of_nickels : ℕ := 9)
    (total_value_in_dollars : ℝ := 21)
    (coins_per_pile : ℕ := 10)
    (quarter_value : ℝ := 0.25)
    (dime_value : ℝ := 0.10)
    (nickel_value : ℝ := 0.05)
    (penny_value : ℝ := 0.01) :
    (total_value_in_dollars - ((piles_of_quarters * coins_per_pile * quarter_value) +
                               (piles_of_dimes * coins_per_pile * dime_value) +
                               (piles_of_nickels * coins_per_pile * nickel_value))) /
                               (coins_per_pile * penny_value) = 5 := 
by
  sorry

end number_of_penny_piles_l44_44449


namespace factor_quadratic_l44_44061

theorem factor_quadratic (y : ℝ) : 16 * y^2 - 40 * y + 25 = (4 * y - 5)^2 := 
by 
  sorry

end factor_quadratic_l44_44061


namespace star_polygon_n_value_l44_44797

theorem star_polygon_n_value (n : ℕ) (A B : ℕ → ℝ) (h1 : ∀ i, A i = B i - 20)
    (h2 : 360 = n * 20) : n = 18 :=
by {
  sorry
}

end star_polygon_n_value_l44_44797


namespace triangle_inequality_internal_point_l44_44267

theorem triangle_inequality_internal_point {A B C P : Type} 
  (x y z p q r : ℝ) 
  (h_distances_from_vertices : x > 0 ∧ y > 0 ∧ z > 0) 
  (h_distances_from_sides : p > 0 ∧ q > 0 ∧ r > 0)
  (h_x_y_z_triangle_ineq : x + y > z ∧ y + z > x ∧ z + x > y)
  (h_p_q_r_triangle_ineq : p + q > r ∧ q + r > p ∧ r + p > q) :
  x * y * z ≥ (q + r) * (r + p) * (p + q) :=
sorry

end triangle_inequality_internal_point_l44_44267


namespace competition_inequality_l44_44211

variable (a b k : ℕ)

-- Conditions
variable (h1 : b % 2 = 1) 
variable (h2 : b ≥ 3)
variable (h3 : ∀ (J1 J2 : ℕ), J1 ≠ J2 → ∃ num_students : ℕ, num_students ≤ a ∧ num_students ≤ k)

theorem competition_inequality (h1: b % 2 = 1) (h2: b ≥ 3) (h3: ∀ (J1 J2 : ℕ), J1 ≠ J2 → ∃ num_students : ℕ, num_students ≤ a ∧ num_students ≤ k) :
  (k: ℝ) / (a: ℝ) ≥ (b-1: ℝ) / (2*b: ℝ) := sorry

end competition_inequality_l44_44211


namespace simplify_exponents_l44_44277

variable (x : ℝ)

theorem simplify_exponents (x : ℝ) : (x^5) * (x^2) = x^(7) :=
by
  sorry

end simplify_exponents_l44_44277


namespace watermelons_eaten_l44_44011

theorem watermelons_eaten (original left : ℕ) (h1 : original = 4) (h2 : left = 1) :
  original - left = 3 :=
by {
  -- Providing the proof steps is not necessary as per the instructions
  sorry
}

end watermelons_eaten_l44_44011


namespace claudia_filled_5oz_glasses_l44_44200

theorem claudia_filled_5oz_glasses :
  ∃ (n : ℕ), n = 6 ∧ 4 * 8 + 15 * 4 + n * 5 = 122 :=
by
  sorry

end claudia_filled_5oz_glasses_l44_44200


namespace length_of_bridge_l44_44493

theorem length_of_bridge (L_train : ℕ) (v_km_hr : ℕ) (t : ℕ) 
  (h_L_train : L_train = 150)
  (h_v_km_hr : v_km_hr = 45)
  (h_t : t = 30) : 
  ∃ L_bridge : ℕ, L_bridge = 225 :=
by 
  sorry

end length_of_bridge_l44_44493


namespace copper_price_l44_44359

theorem copper_price (c : ℕ) (hzinc : ℕ) (zinc_weight : ℕ) (brass_weight : ℕ) (price_brass : ℕ) (used_copper : ℕ) :
  hzinc = 30 →
  zinc_weight = brass_weight - used_copper →
  brass_weight = 70 →
  price_brass = 45 →
  used_copper = 30 →
  (used_copper * c + zinc_weight * hzinc) = brass_weight * price_brass →
  c = 65 :=
by
  sorry

end copper_price_l44_44359


namespace odd_n_cube_minus_n_div_by_24_l44_44365

theorem odd_n_cube_minus_n_div_by_24 (n : ℤ) (h_odd : n % 2 = 1) : 24 ∣ (n^3 - n) :=
sorry

end odd_n_cube_minus_n_div_by_24_l44_44365


namespace average_salary_rest_of_workers_l44_44946

theorem average_salary_rest_of_workers
  (avg_salary_all : ℝ)
  (num_all_workers : ℕ)
  (avg_salary_techs : ℝ)
  (num_techs : ℕ)
  (avg_salary_rest : ℝ)
  (num_rest : ℕ) :
  avg_salary_all = 8000 →
  num_all_workers = 21 →
  avg_salary_techs = 12000 →
  num_techs = 7 →
  num_rest = num_all_workers - num_techs →
  avg_salary_rest = (avg_salary_all * num_all_workers - avg_salary_techs * num_techs) / num_rest →
  avg_salary_rest = 6000 :=
by
  intros h_avg_all h_num_all h_avg_techs h_num_techs h_num_rest h_avg_rest
  sorry

end average_salary_rest_of_workers_l44_44946


namespace exists_x_gg_eq_3_l44_44594

noncomputable def g (x : ℝ) : ℝ :=
if x < -3 then -0.5 * x^2 + 3
else if x < 2 then 1
else 0.5 * x^2 - 1.5 * x + 3

theorem exists_x_gg_eq_3 : ∃ x : ℝ, x = -5 ∨ x = 5 ∧ g (g x) = 3 :=
by
  sorry

end exists_x_gg_eq_3_l44_44594


namespace find_a_l44_44155

def diamond (a b : ℝ) : ℝ := 3 * a - b^2

theorem find_a (a : ℝ) (h : diamond a 6 = 15) : a = 17 :=
by
  sorry

end find_a_l44_44155


namespace arithmetic_mean_of_fractions_l44_44802

theorem arithmetic_mean_of_fractions :
  let a := (9 : ℝ) / 12
  let b := (5 : ℝ) / 6
  let c := (11 : ℝ) / 12
  (a + c) / 2 = b := 
by
  sorry

end arithmetic_mean_of_fractions_l44_44802


namespace polyhedron_volume_correct_l44_44743

-- Definitions of geometric shapes and their properties
def is_isosceles_right_triangle (A : Type) (a b c : ℝ) := 
  a = b ∧ c = a * Real.sqrt 2

def is_square (B : Type) (side : ℝ) := 
  side = 2

def is_equilateral_triangle (G : Type) (side : ℝ) := 
  side = Real.sqrt 8

noncomputable def polyhedron_volume (A E F B C D G : Type) (a b c d e f g : ℝ) := 
  let cube_volume := 8
  let tetrahedron_volume := 2 * Real.sqrt 2 / 3
  cube_volume - tetrahedron_volume

theorem polyhedron_volume_correct (A E F B C D G : Type) (a b c d e f g : ℝ) :
  (is_isosceles_right_triangle A a b c) →
  (is_isosceles_right_triangle E a b c) →
  (is_isosceles_right_triangle F a b c) →
  (is_square B d) →
  (is_square C e) →
  (is_square D f) →
  (is_equilateral_triangle G g) →
  a = 2 → d = 2 → e = 2 → f = 2 → g = Real.sqrt 8 →
  polyhedron_volume A E F B C D G a b c d e f g =
    8 - (2 * Real.sqrt 2 / 3) :=
by
  intros hA hE hF hB hC hD hG ha hd he hf hg
  sorry

end polyhedron_volume_correct_l44_44743


namespace pet_store_cages_l44_44671

theorem pet_store_cages (initial_puppies sold_puppies puppies_per_cage : ℕ) (h₁ : initial_puppies = 78)
(h₂ : sold_puppies = 30) (h₃ : puppies_per_cage = 8) : (initial_puppies - sold_puppies) / puppies_per_cage = 6 :=
by
  -- assumptions: initial_puppies = 78, sold_puppies = 30, puppies_per_cage = 8
  -- goal: (initial_puppies - sold_puppies) / puppies_per_cage = 6
  sorry

end pet_store_cages_l44_44671


namespace line_intersects_ellipse_max_chord_length_l44_44606

theorem line_intersects_ellipse (m : ℝ) : 
  (∃ x y : ℝ, (y = (3/2 : ℝ) * x + m) ∧ (x^2 / 4 + y^2 / 9 = 1)) ↔ 
  (-3 * Real.sqrt 2 ≤ m ∧ m ≤ 3 * Real.sqrt 2) := 
by sorry

theorem max_chord_length : 
  (∃ m : ℝ, (m = 0) ∧ 
    (∀ x y x1 y1 : ℝ, (y = (3/2 : ℝ) * x + m) ∧ (x^2 / 4 + y^2 / 9 = 1) ∧ 
     (y1 = (3/2 : ℝ) * x1 + m) ∧ (x1^2 / 4 + y1^2 / 9 = 1) ∧ 
     (x ≠ x1 ∨ y ≠ y1) → 
     (Real.sqrt (13 / 9) * Real.sqrt (18 - m^2) = Real.sqrt 26))) := 
by sorry

end line_intersects_ellipse_max_chord_length_l44_44606


namespace smallest_four_digit_number_l44_44181

noncomputable def smallest_four_digit_solution : ℕ := 1011

theorem smallest_four_digit_number (x : ℕ) (h1 : 5 * x ≡ 25 [MOD 20]) (h2 : 3 * x + 10 ≡ 19 [MOD 7]) (h3 : x + 3 ≡ 2 * x [MOD 12]) :
  x = smallest_four_digit_solution :=
by
  sorry

end smallest_four_digit_number_l44_44181


namespace find_side_c_of_triangle_ABC_l44_44815

theorem find_side_c_of_triangle_ABC
  (a b : ℝ)
  (cosA : ℝ)
  (c : ℝ) :
  a = 4 * Real.sqrt 5 →
  b = 5 →
  cosA = 3 / 5 →
  c^2 - 3 * c - 55 = 0 →
  c = 11 := by
  intros ha hb hcosA hquadratic
  sorry

end find_side_c_of_triangle_ABC_l44_44815


namespace denis_dartboard_score_l44_44194

theorem denis_dartboard_score :
  ∀ P1 P2 P3 P4 : ℕ,
  P1 = 30 → 
  P2 = 38 → 
  P3 = 41 → 
  P1 + P2 + P3 + P4 = 4 * ((P1 + P2 + P3 + P4) / 4) → 
  P4 = 34 :=
by
  intro P1 P2 P3 P4 hP1 hP2 hP3 hTotal
  have hSum := hP1.symm ▸ hP2.symm ▸ hP3.symm ▸ hTotal
  sorry

end denis_dartboard_score_l44_44194


namespace gold_copper_ratio_l44_44299

theorem gold_copper_ratio (G C : ℕ) (h : 19 * G + 9 * C = 17 * (G + C)) : G = 4 * C :=
by
  sorry

end gold_copper_ratio_l44_44299


namespace pears_sales_l44_44337

variable (x : ℝ)
variable (morning_sales : ℝ := x)
variable (afternoon_sales : ℝ := 2 * x)
variable (evening_sales : ℝ := 3 * afternoon_sales)
variable (total_sales : ℝ := morning_sales + afternoon_sales + evening_sales)

theorem pears_sales :
  (total_sales = 510) →
  (afternoon_sales = 113.34) :=
by
  sorry

end pears_sales_l44_44337


namespace probability_of_hitting_target_at_least_once_l44_44787

noncomputable def prob_hit_target_once : ℚ := 2/3

noncomputable def prob_miss_target_once : ℚ := 1 - prob_hit_target_once

noncomputable def prob_miss_target_three_times : ℚ := prob_miss_target_once ^ 3

noncomputable def prob_hit_target_at_least_once : ℚ := 1 - prob_miss_target_three_times

theorem probability_of_hitting_target_at_least_once :
  prob_hit_target_at_least_once = 26 / 27 := 
sorry

end probability_of_hitting_target_at_least_once_l44_44787


namespace function_decreasing_range_k_l44_44058

theorem function_decreasing_range_k : 
  ∀ k : ℝ, (∀ x : ℝ, 1 ≤ x → ∀ y : ℝ, 1 ≤ y → x ≤ y → (k * x ^ 2 + (3 * k - 2) * x - 5) ≥ (k * y ^ 2 + (3 * k - 2) * y - 5)) ↔ (k ∈ Set.Iic 0) :=
by sorry

end function_decreasing_range_k_l44_44058


namespace problem_l44_44030

-- Helper definition for point on a line
def point_on_line (x y : ℝ) (a b : ℝ) : Prop := y = a * x + b

-- Given condition: Point P(1, 3) lies on the line y = 2x + b
def P_on_l (b : ℝ) : Prop := point_on_line 1 3 2 b

-- The proof problem: Proving (2, 5) also lies on the line y = 2x + b where b is the constant found using P
theorem problem (b : ℝ) (h: P_on_l b) : point_on_line 2 5 2 b :=
by
  sorry

end problem_l44_44030


namespace find_a_l44_44370

def A : Set ℝ := {0, 2}
def B (a : ℝ) : Set ℝ := {1, a ^ 2}

theorem find_a (a : ℝ) (h : A ∪ B a = {0, 1, 2, 4}) : a = 2 ∨ a = -2 :=
by
  sorry

end find_a_l44_44370


namespace gauss_family_mean_age_l44_44890

theorem gauss_family_mean_age :
  let ages := [8, 8, 8, 8, 16, 17]
  let num_children := 6
  let sum_ages := 65
  (sum_ages : ℚ) / (num_children : ℚ) = 65 / 6 :=
by
  sorry

end gauss_family_mean_age_l44_44890


namespace find_a_range_l44_44983

theorem find_a_range (a : ℝ) :
  (∃ x : ℝ, x > 0 ∧ 2 * x^2 - 2 * x + a - 3 = 0) ∧ 
  (∃ y : ℝ, y > 0 ∧ y ≠ x ∧ 2 * y^2 - 2 * y + a - 3 = 0) 
  ↔ 3 < a ∧ a < 7 / 2 := 
sorry

end find_a_range_l44_44983


namespace distance_between_A_and_B_is_45_kilometers_l44_44721

variable (speedA speedB : ℝ)
variable (distanceAB : ℝ)

noncomputable def problem_conditions := 
  speedA = 1.2 * speedB ∧
  ∃ (distanceMalfunction : ℝ), distanceMalfunction = 5 ∧
  ∃ (timeFixingMalfunction : ℝ), timeFixingMalfunction = (distanceAB / 6) / speedB ∧
  ∃ (increasedSpeedB : ℝ), increasedSpeedB = 1.6 * speedB ∧
  ∃ (timeA timeB timeB_new : ℝ),
    timeA = (distanceAB / speedA) ∧
    timeB = (distanceMalfunction / speedB) + timeFixingMalfunction + (distanceAB - distanceMalfunction) / increasedSpeedB ∧
    timeA = timeB

theorem distance_between_A_and_B_is_45_kilometers
  (speedA speedB distanceAB : ℝ) 
  (cond : problem_conditions speedA speedB distanceAB) :
  distanceAB = 45 :=
sorry

end distance_between_A_and_B_is_45_kilometers_l44_44721


namespace problem_1_solution_set_problem_2_range_of_T_l44_44093

noncomputable def f (x : ℝ) : ℝ := |2 * x + 1| - |x - 2|

theorem problem_1_solution_set :
  {x : ℝ | f x > 2} = {x | x < -5 ∨ 1 < x} :=
by 
  -- to be proven
  sorry

theorem problem_2_range_of_T (T : ℝ) :
  (∀ x : ℝ, f x ≥ -T^2 - 2.5 * T - 1) →
  (T ≤ -3 ∨ T ≥ 0.5) :=
by
  -- to be proven
  sorry

end problem_1_solution_set_problem_2_range_of_T_l44_44093


namespace range_of_a_l44_44872

namespace InequalityProblem

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, (1 < x ∧ x < 2) → (x - 1)^2 < Real.log x / Real.log a) ↔ (1 < a ∧ a ≤ 2) :=
by
  sorry

end InequalityProblem

end range_of_a_l44_44872


namespace quadratic_no_real_solutions_l44_44799

theorem quadratic_no_real_solutions (a : ℝ) (h₀ : 0 < a) (h₁ : a^3 = 6 * (a + 1)) : 
  ∀ x : ℝ, ¬ (x^2 + a * x + a^2 - 6 = 0) :=
by
  sorry

end quadratic_no_real_solutions_l44_44799


namespace rectangle_width_is_3_l44_44534

-- Define the given conditions
def length_square : ℝ := 9
def length_rectangle : ℝ := 27

-- Calculate the area based on the given conditions
def area_square : ℝ := length_square * length_square

-- Define the area equality condition
def area_equality (width_rectangle : ℝ) : Prop :=
  area_square = length_rectangle * width_rectangle

-- The theorem stating the width of the rectangle
theorem rectangle_width_is_3 (width_rectangle: ℝ) :
  area_equality width_rectangle → width_rectangle = 3 :=
by
  -- Skipping the proof itself as instructed
  intro h
  sorry

end rectangle_width_is_3_l44_44534


namespace value_independent_of_b_value_for_d_zero_l44_44653

theorem value_independent_of_b
  (c b d h : ℝ)
  (x1 x2 x3 x4 y1 y2 y3 y4 : ℝ)
  (h1 : x1 = b - d - h)
  (h2 : x2 = b - d)
  (h3 : x3 = b + d)
  (h4 : x4 = b + d + h)
  (hy1 : y1 = c * x1^2)
  (hy2 : y2 = c * x2^2)
  (hy3 : y3 = c * x3^2)
  (hy4 : y4 = c * x4^2) :
  (y1 + y4 - y2 - y3) = 2 * c * h * (2 * d + h) :=
by
  sorry

theorem value_for_d_zero
  (c b h : ℝ)
  (x1 x2 x3 x4 y1 y2 y3 y4 : ℝ)
  (d : ℝ := 0)
  (h1 : x1 = b - h)
  (h2 : x2 = b)
  (h3 : x3 = b)
  (h4 : x4 = b + h)
  (hy1 : y1 = c * x1^2)
  (hy2 : y2 = c * x2^2)
  (hy3 : y3 = c * x3^2)
  (hy4 : y4 = c * x4^2) :
  (y1 + y4 - y2 - y3) = 2 * c * h^2 :=
by
  sorry

end value_independent_of_b_value_for_d_zero_l44_44653


namespace reciprocal_of_sum_of_fraction_l44_44345

theorem reciprocal_of_sum_of_fraction (y : ℚ) (h : y = 6 + 1/6) : 1 / y = 6 / 37 := by
  sorry

end reciprocal_of_sum_of_fraction_l44_44345


namespace pencils_sold_is_correct_l44_44820

-- Define the conditions
def first_two_students_pencils : Nat := 2 * 2
def next_six_students_pencils : Nat := 6 * 3
def last_two_students_pencils : Nat := 2 * 1
def total_pencils_sold : Nat := first_two_students_pencils + next_six_students_pencils + last_two_students_pencils

-- Prove that all pencils sold equals 24
theorem pencils_sold_is_correct : total_pencils_sold = 24 :=
by 
  -- Add the statement to be proved here
  sorry

end pencils_sold_is_correct_l44_44820


namespace edward_initial_amount_l44_44871

theorem edward_initial_amount (spent received final_amount : ℤ) 
  (h_spent : spent = 17) 
  (h_received : received = 10) 
  (h_final : final_amount = 7) : 
  ∃ initial_amount : ℤ, (initial_amount - spent + received = final_amount) ∧ (initial_amount = 14) :=
by
  sorry

end edward_initial_amount_l44_44871


namespace math_equivalence_l44_44830

theorem math_equivalence (m n : ℤ) (h : |m - 2023| + (n + 2024)^2 = 0) : (m + n) ^ 2023 = -1 := 
by
  sorry

end math_equivalence_l44_44830


namespace max_height_reached_by_rocket_l44_44052

def h (t : ℝ) : ℝ := -12 * t^2 + 72 * t + 36

theorem max_height_reached_by_rocket : ∃ t : ℝ, h t = 144 ∧ ∀ t' : ℝ, h t' ≤ 144 := sorry

end max_height_reached_by_rocket_l44_44052


namespace z_amount_per_rupee_l44_44885

theorem z_amount_per_rupee (x y z : ℝ) 
  (h1 : ∀ rupees_x, y = 0.45 * rupees_x)
  (h2 : y = 36)
  (h3 : x + y + z = 156)
  (h4 : ∀ rupees_x, x = rupees_x) :
  ∃ a : ℝ, z = a * x ∧ a = 0.5 := 
by
  -- Placeholder for the actual proof
  sorry

end z_amount_per_rupee_l44_44885


namespace original_flow_rate_l44_44780

theorem original_flow_rate (x : ℝ) (h : 2 = 0.6 * x - 1) : x = 5 :=
by
  sorry

end original_flow_rate_l44_44780


namespace factorize_diff_squares_1_factorize_diff_squares_2_factorize_common_term_l44_44789

-- Proof Problem 1
theorem factorize_diff_squares_1 (x y : ℝ) :
  4 * x^2 - 9 * y^2 = (2 * x + 3 * y) * (2 * x - 3 * y) :=
sorry

-- Proof Problem 2
theorem factorize_diff_squares_2 (a b : ℝ) :
  -16 * a^2 + 25 * b^2 = (5 * b + 4 * a) * (5 * b - 4 * a) :=
sorry

-- Proof Problem 3
theorem factorize_common_term (x y : ℝ) :
  x^3 * y - x * y^3 = x * y * (x + y) * (x - y) :=
sorry

end factorize_diff_squares_1_factorize_diff_squares_2_factorize_common_term_l44_44789


namespace cylinder_volume_l44_44209

theorem cylinder_volume (r h : ℝ) (hrh : 2 * Real.pi * r * h = 100 * Real.pi) (h_diag : 4 * r^2 + h^2 = 200) :
  Real.pi * r^2 * h = 250 * Real.pi :=
sorry

end cylinder_volume_l44_44209


namespace exponentiation_problem_l44_44936

theorem exponentiation_problem (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (h : 2^a * 2^b = 8) : (2^a)^b = 4 := 
sorry

end exponentiation_problem_l44_44936


namespace taco_cost_l44_44207

theorem taco_cost (T E : ℝ) (h1 : 2 * T + 3 * E = 7.80) (h2 : 3 * T + 5 * E = 12.70) : T = 0.90 := 
by 
  sorry

end taco_cost_l44_44207


namespace money_left_after_purchase_l44_44035

-- The costs and amounts for each item
def bread_cost : ℝ := 2.35
def num_bread : ℝ := 4
def peanut_butter_cost : ℝ := 3.10
def num_peanut_butter : ℝ := 2
def honey_cost : ℝ := 4.50
def num_honey : ℝ := 1

-- The coupon discount and budget
def coupon_discount : ℝ := 2
def budget : ℝ := 20

-- Calculate the total cost before applying the coupon
def total_before_coupon : ℝ := num_bread * bread_cost + num_peanut_butter * peanut_butter_cost + num_honey * honey_cost

-- Calculate the total cost after applying the coupon
def total_after_coupon : ℝ := total_before_coupon - coupon_discount

-- Calculate the money left over after the purchase
def money_left_over : ℝ := budget - total_after_coupon

-- The theorem to be proven
theorem money_left_after_purchase : money_left_over = 1.90 :=
by
  -- The proof of this theorem will involve the specific calculations and will be filled in later
  sorry

end money_left_after_purchase_l44_44035


namespace extra_men_needed_l44_44190

theorem extra_men_needed (total_length : ℝ) (total_days : ℕ) (initial_men : ℕ) (completed_length : ℝ) (days_passed : ℕ) 
  (remaining_length := total_length - completed_length)
  (remaining_days := total_days - days_passed)
  (current_rate := completed_length / days_passed)
  (required_rate := remaining_length / remaining_days)
  (rate_increase := required_rate / current_rate)
  (total_men_needed := initial_men * rate_increase)
  (extra_men_needed := ⌈total_men_needed⌉ - initial_men) :
  total_length = 15 → 
  total_days = 300 → 
  initial_men = 35 → 
  completed_length = 2.5 → 
  days_passed = 100 → 
  extra_men_needed = 53 :=
by
-- Prove that given the conditions, the number of extra men needed is 53
sorry

end extra_men_needed_l44_44190


namespace jason_cutting_hours_l44_44471

-- Definitions derived from conditions
def time_to_cut_one_lawn : ℕ := 30  -- minutes
def lawns_per_day := 8 -- number of lawns Jason cuts each day
def days := 2 -- number of days (Saturday and Sunday)
def minutes_in_an_hour := 60 -- conversion factor from minutes to hours

-- The proof problem
theorem jason_cutting_hours : 
  (time_to_cut_one_lawn * lawns_per_day * days) / minutes_in_an_hour = 8 := sorry

end jason_cutting_hours_l44_44471


namespace arithmetic_contains_geometric_progression_l44_44104

theorem arithmetic_contains_geometric_progression (a d : ℕ) (h_pos : d > 0) :
  ∃ (a' : ℕ) (r : ℕ), a' = a ∧ r = 1 + d ∧ (∀ k : ℕ, ∃ n : ℕ, a' * r^k = a + (n-1)*d) :=
by
  sorry

end arithmetic_contains_geometric_progression_l44_44104


namespace digit_sum_of_4_digit_number_l44_44848

theorem digit_sum_of_4_digit_number (abcd : ℕ) (H1 : 1000 ≤ abcd ∧ abcd < 10000) (erased_digit: ℕ) (H2: erased_digit < 10) (H3 : 100*(abcd / 1000) + 10*(abcd % 1000 / 100) + (abcd % 100 / 10) + erased_digit = 6031): 
    (abcd / 1000 + abcd % 1000 / 100 + abcd % 100 / 10 + abcd % 10 = 20) :=
sorry

end digit_sum_of_4_digit_number_l44_44848


namespace find_n_l44_44280

open Nat

-- Define the binomial coefficient function
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Given condition for the proof
def condition (n : ℕ) : Prop := binom (n + 1) 7 - binom n 7 = binom n 8

-- The statement to prove
theorem find_n (n : ℕ) (h : condition n) : n = 14 :=
sorry

end find_n_l44_44280


namespace lines_intersect_ellipse_at_2_or_4_points_l44_44441

noncomputable def ellipse_eq (x y : ℝ) : Prop := (x^2) / 4 + (y^2) / 9 = 1

def line_intersects_ellipse (line : ℝ → ℝ → Prop) (x y : ℝ) : Prop :=
  ellipse_eq x y ∧ line x y

def number_of_intersections (line1 line2 : ℝ → ℝ → Prop) (n : ℕ) : Prop :=
  ∃ pts : Finset (ℝ × ℝ), (∀ pt ∈ pts, (line_intersects_ellipse line1 pt.1 pt.2 ∨
                                        line_intersects_ellipse line2 pt.1 pt.2)) ∧
                           pts.card = n ∧ 
                           (∀ pt ∈ pts, line1 pt.1 pt.2 ∨ line2 pt.1 pt.2) ∧
                           (∀ (pt1 pt2 : ℝ × ℝ), pt1 ∈ pts → pt2 ∈ pts → pt1 ≠ pt2 → pt1 ≠ pt2)

theorem lines_intersect_ellipse_at_2_or_4_points 
  (line1 line2 : ℝ → ℝ → Prop)
  (h1 : ∃ x1 y1, line1 x1 y1 ∧ ellipse_eq x1 y1)
  (h2 : ∃ x2 y2, line2 x2 y2 ∧ ellipse_eq x2 y2)
  (h3: ¬ ∀ x y, line1 x y ∧ ellipse_eq x y → false)
  (h4: ¬ ∀ x y, line2 x y ∧ ellipse_eq x y → false) :
  ∃ n : ℕ, (n = 2 ∨ n = 4) ∧ number_of_intersections line1 line2 n := sorry

end lines_intersect_ellipse_at_2_or_4_points_l44_44441


namespace find_polynomials_l44_44286

-- Define our polynomial P(x)
def polynomial_condition (P : Polynomial ℝ) : Prop :=
  ∀ x : ℝ, (x-1) * P.eval (x+1) - (x+2) * P.eval x = 0

-- State the theorem
theorem find_polynomials (P : Polynomial ℝ) :
  polynomial_condition P ↔ ∃ a : ℝ, P = Polynomial.C a * (Polynomial.X^3 - Polynomial.X) :=
by
  sorry

end find_polynomials_l44_44286


namespace num_factors_of_M_l44_44136

def M : ℕ := 2^4 * 3^3 * 7^2

theorem num_factors_of_M : ∃ n, n = 60 ∧ (∀ d e f : ℕ, 0 ≤ d ∧ d ≤ 4 ∧ 0 ≤ e ∧ e ≤ 3 ∧ 0 ≤ f ∧ f ≤ 2 → (2^d * 3^e * 7^f ∣ M) ∧ ∃ k, k = 5 * 4 * 3 ∧ k = n) :=
by
  sorry

end num_factors_of_M_l44_44136


namespace greatest_value_x_l44_44037

theorem greatest_value_x (x : ℝ) : 
  (x ≠ 9) → 
  (x^2 - 5 * x - 84) / (x - 9) = 4 / (x + 6) →
  x ≤ -2 :=
by
  sorry

end greatest_value_x_l44_44037


namespace sum_of_arithmetic_sequence_l44_44992

noncomputable def arithmetic_sequence (a : ℕ → ℝ) :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem sum_of_arithmetic_sequence (a : ℕ → ℝ) 
  (h_seq : arithmetic_sequence a)
  (h_a5 : a 5 = 3)
  (h_a7 : a 7 = 7) :
  a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 = 35 :=
  sorry

end sum_of_arithmetic_sequence_l44_44992


namespace daniel_dolls_l44_44939

theorem daniel_dolls (normal_price discount_price: ℕ) 
  (normal_dolls: ℕ) 
  (saved_money: ℕ := normal_dolls * normal_price):
  normal_price = 4 →
  normal_dolls = 15 →
  discount_price = 3 →
  saved_money = normal_dolls * normal_price →
  saved_money / discount_price = 20 :=
by
  sorry

end daniel_dolls_l44_44939


namespace number_of_cookies_l44_44381

def total_cake := 22
def total_chocolate := 16
def total_groceries := 42

theorem number_of_cookies :
  ∃ C : ℕ, total_groceries = total_cake + total_chocolate + C ∧ C = 4 := 
by
  sorry

end number_of_cookies_l44_44381


namespace find_m_l44_44881

noncomputable def polynomial (x : ℝ) (m : ℝ) := 4 * x^2 - 3 * x + 5 - 2 * m * x^2 - x + 1

theorem find_m (m : ℝ) : 
  ∀ x : ℝ, (4 * x^2 - 3 * x + 5 - 2 * m * x^2 - x + 1 = (4 - 2 * m) * x^2 - 4 * x + 6)
  → (4 - 2 * m = 0) → (m = 2) :=
by
  intros x h1 h2
  sorry

end find_m_l44_44881


namespace sampling_methods_correct_l44_44005

def company_sales_outlets (A B C D : ℕ) : Prop :=
  A = 150 ∧ B = 120 ∧ C = 180 ∧ D = 150 ∧ A + B + C + D = 600

def investigation_samples (total_samples large_outlets region_C_sample : ℕ) : Prop :=
  total_samples = 100 ∧ large_outlets = 20 ∧ region_C_sample = 7

def appropriate_sampling_methods (investigation1_method investigation2_method : String) : Prop :=
  investigation1_method = "Stratified sampling" ∧ investigation2_method = "Simple random sampling"

theorem sampling_methods_correct :
  company_sales_outlets 150 120 180 150 →
  investigation_samples 100 20 7 →
  appropriate_sampling_methods "Stratified sampling" "Simple random sampling" :=
by
  intros h1 h2
  sorry

end sampling_methods_correct_l44_44005


namespace number_is_12_l44_44305

theorem number_is_12 (x : ℝ) (h : 4 * x - 3 = 9 * (x - 7)) : x = 12 :=
by
  sorry

end number_is_12_l44_44305


namespace complete_the_square_l44_44693

theorem complete_the_square (d e f : ℤ) (h1 : 0 < d)
    (h2 : ∀ x : ℝ, 100 * x^2 + 60 * x - 90 = 0 ↔ (d * x + e)^2 = f) :
  d + e + f = 112 := by
  sorry

end complete_the_square_l44_44693


namespace area_region_inside_but_outside_l44_44625

noncomputable def area_diff (side_large side_small : ℝ) : ℝ :=
  (side_large ^ 2) - (side_small ^ 2)

theorem area_region_inside_but_outside (h_large : 10 > 0) (h_small : 4 > 0) :
  area_diff 10 4 = 84 :=
by
  -- The proof steps would go here
  sorry

end area_region_inside_but_outside_l44_44625


namespace greatest_divisor_consistent_remainder_l44_44549

noncomputable def gcd_of_differences : ℕ :=
  Nat.gcd (Nat.gcd 1050 28770) 71670

theorem greatest_divisor_consistent_remainder :
  gcd_of_differences = 30 :=
by
  -- The proof can be filled in here...
  sorry

end greatest_divisor_consistent_remainder_l44_44549


namespace rotation_90_ccw_l44_44410

-- Define the complex number before the rotation
def initial_complex : ℂ := -4 - 2 * Complex.I

-- Define the resulting complex number after a 90-degree counter-clockwise    rotation
def result_complex : ℂ := 2 - 4 * Complex.I

-- State the theorem to be proved
theorem rotation_90_ccw (z : ℂ) (h : z = initial_complex) :
  Complex.I * z = result_complex :=
by sorry

end rotation_90_ccw_l44_44410


namespace shifted_function_expression_l44_44490

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ :=
  Real.sin (ω * x + Real.pi / 3)

theorem shifted_function_expression (ω : ℝ) (h : ℝ) (x : ℝ) (h_positive : ω > 0) (h_period : Real.pi = 2 * Real.pi / ω) :
  f ω (x + h) = Real.cos (2 * x) :=
by
  -- We assume h = π/12, ω = 2
  have ω_val : ω = 2 := by sorry
  have h_val : h = Real.pi / 12 := by sorry
  rw [ω_val, h_val]
  sorry

end shifted_function_expression_l44_44490


namespace angle_sum_property_l44_44045

theorem angle_sum_property 
  (P Q R S : Type) 
  (alpha beta : ℝ)
  (h1 : alpha = 3 * x)
  (h2 : beta = 2 * x)
  (h3 : alpha + beta = 90) :
  x = 18 :=
by
  sorry

end angle_sum_property_l44_44045


namespace smallest_r_l44_44753

variables (p q r s : ℤ)

-- Define the conditions
def condition1 : Prop := p + 3 = q - 1
def condition2 : Prop := p + 3 = r + 5
def condition3 : Prop := p + 3 = s - 2

-- Prove that r is the smallest
theorem smallest_r (h1 : condition1 p q) (h2 : condition2 p r) (h3 : condition3 p s) : r < p ∧ r < q ∧ r < s :=
sorry

end smallest_r_l44_44753


namespace complex_product_l44_44193

theorem complex_product (a b c d : ℤ) (i : ℂ) (h : i^2 = -1) :
  (6 - 7 * i) * (3 + 6 * i) = 60 + 15 * i :=
  by
    -- proof statements would go here
    sorry

end complex_product_l44_44193


namespace number_of_correct_statements_l44_44084

noncomputable def f (x : ℝ) : ℝ := 1 / 2 * Real.sin (2 * x)

def statement_1 : Prop := ∃ T > 0, ∀ x, f (x + T) = f x ∧ T = 2 * Real.pi
def statement_2 : Prop := ∀ x y, -Real.pi / 4 ≤ x ∧ x ≤ y ∧ y ≤ Real.pi / 4 → f x ≤ f y
def statement_3 : Prop := ∀ y, -Real.pi / 6 ≤ y ∧ y ≤ Real.pi / 3 → -Real.sqrt 3 / 4 ≤ f y ∧ f y ≤ Real.sqrt 3 / 4
def statement_4 : Prop := ∀ x, f x = (1 / 2 * Real.sin (2 * x + Real.pi / 4) - Real.pi / 8)

theorem number_of_correct_statements : 
  (¬ statement_1 ∧ statement_2 ∧ ¬ statement_3 ∧ ¬ statement_4) = true :=
sorry

end number_of_correct_statements_l44_44084


namespace problem_l44_44769

theorem problem (a₅ b₅ a₆ b₆ a₇ b₇ : ℤ) (S₇ S₅ T₆ T₄ : ℤ)
  (h1 : a₅ = b₅)
  (h2 : a₆ = b₆)
  (h3 : S₇ - S₅ = 4 * (T₆ - T₄)) :
  (a₇ + a₅) / (b₇ + b₅) = -1 :=
sorry

end problem_l44_44769


namespace find_larger_number_l44_44225

theorem find_larger_number (hc_f : ℕ) (factor1 factor2 : ℕ)
(h_hcf : hc_f = 63)
(h_factor1 : factor1 = 11)
(h_factor2 : factor2 = 17)
(lcm := hc_f * factor1 * factor2)
(A := hc_f * factor1)
(B := hc_f * factor2) :
max A B = 1071 := by
  sorry

end find_larger_number_l44_44225


namespace point_in_second_quadrant_l44_44766

variable (m : ℝ)

/-- 
If point P(m-1, 3) is in the second quadrant, 
then a possible value of m is -1
--/
theorem point_in_second_quadrant (h1 : (m - 1 < 0)) : m = -1 :=
by sorry

end point_in_second_quadrant_l44_44766


namespace ratio_of_triangle_to_square_l44_44466

theorem ratio_of_triangle_to_square (s : ℝ) (hs : 0 < s) :
  let A_square := s^2
  let A_triangle := (1/2) * s * (s/2)
  A_triangle / A_square = 1/4 :=
by
  sorry

end ratio_of_triangle_to_square_l44_44466


namespace rahul_deepak_age_ratio_l44_44001

-- Define the conditions
variables (R D : ℕ)
axiom deepak_age : D = 33
axiom rahul_future_age : R + 6 = 50

-- Define the theorem to prove the ratio
theorem rahul_deepak_age_ratio : R / D = 4 / 3 :=
by
  -- Placeholder for proof
  sorry

end rahul_deepak_age_ratio_l44_44001


namespace episodes_lost_per_season_l44_44957

theorem episodes_lost_per_season (s1 s2 : ℕ) (e : ℕ) (remaining : ℕ) (total_seasons : ℕ) (total_episodes_before : ℕ) (total_episodes_lost : ℕ)
  (h1 : s1 = 12) (h2 : s2 = 14) (h3 : e = 16) (h4 : remaining = 364) 
  (h5 : total_seasons = s1 + s2) (h6 : total_episodes_before = s1 * e + s2 * e) 
  (h7 : total_episodes_lost = total_episodes_before - remaining) :
  total_episodes_lost / total_seasons = 2 := by
  sorry

end episodes_lost_per_season_l44_44957


namespace ara_current_height_l44_44819

theorem ara_current_height (original_height : ℚ) (shea_growth_ratio : ℚ) (ara_growth_ratio : ℚ) (shea_current_height : ℚ) (h1 : shea_growth_ratio = 0.25) (h2 : ara_growth_ratio = 0.75) (h3 : shea_current_height = 75) (h4 : shea_current_height = original_height * (1 + shea_growth_ratio)) : 
  original_height * (1 + ara_growth_ratio * shea_growth_ratio) = 71.25 := 
by
  sorry

end ara_current_height_l44_44819


namespace find_last_number_l44_44514

theorem find_last_number (A B C D : ℝ) (h1 : A + B + C = 18) (h2 : B + C + D = 9) (h3 : A + D = 13) : D = 2 :=
by
sorry

end find_last_number_l44_44514


namespace restore_original_price_l44_44371

-- Defining the original price of the jacket
def original_price (P : ℝ) := P

-- Defining the price after each step of reduction
def price_after_first_reduction (P : ℝ) := P * (1 - 0.25)
def price_after_second_reduction (P : ℝ) := price_after_first_reduction P * (1 - 0.20)
def price_after_third_reduction (P : ℝ) := price_after_second_reduction P * (1 - 0.10)

-- Express the condition to restore the original price
theorem restore_original_price (P : ℝ) (x : ℝ) : 
  original_price P = price_after_third_reduction P * (1 + x) → 
  x = 0.85185185 := 
by
  sorry

end restore_original_price_l44_44371


namespace f_2011_is_zero_l44_44790

noncomputable def f : ℝ → ℝ := sorry

-- Conditions
axiom odd_f : ∀ x : ℝ, f (-x) = -f x
axiom periodic_f : ∀ x : ℝ, f (x + 2) = f (x) + f (1)

-- Theorem stating the mathematically equivalent proof problem
theorem f_2011_is_zero : f (2011) = 0 :=
sorry

end f_2011_is_zero_l44_44790


namespace minimum_value_a_plus_4b_l44_44786

theorem minimum_value_a_plus_4b (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h : (1 / a) + (1 / b) = 1) : a + 4 * b ≥ 9 :=
sorry

end minimum_value_a_plus_4b_l44_44786


namespace sneakers_sold_l44_44009

theorem sneakers_sold (total_shoes sandals boots : ℕ) (h1 : total_shoes = 17) (h2 : sandals = 4) (h3 : boots = 11) :
  total_shoes - (sandals + boots) = 2 :=
by
  -- proof steps will be included here
  sorry

end sneakers_sold_l44_44009


namespace hyperbola_equation_of_focus_and_asymptote_l44_44160

theorem hyperbola_equation_of_focus_and_asymptote :
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ (2 * a) ^ 2 + (2 * b) ^ 2 = 25 ∧ b / a = 2 ∧ 
  (∀ x y : ℝ, (y = 2 * x + 10) → (x = -5) ∧ (y = 0)) ∧ 
  (∀ x y : ℝ, (x ^ 2 / 5 - y ^ 2 / 20 = 1)) :=
by
  sorry

end hyperbola_equation_of_focus_and_asymptote_l44_44160


namespace total_shoes_in_box_l44_44662

theorem total_shoes_in_box (pairs : ℕ) (prob_matching : ℚ) (h1 : pairs = 7) (h2 : prob_matching = 1 / 13) : 
  ∃ (n : ℕ), n = 2 * pairs ∧ n = 14 :=
by 
  sorry

end total_shoes_in_box_l44_44662


namespace weights_divide_three_piles_l44_44321

theorem weights_divide_three_piles (n : ℕ) (h : n > 3) :
  (∃ (k : ℕ), n = 3 * k ∨ n = 3 * k + 2) ↔
  (∃ (A B C : Finset ℕ), A ∪ B ∪ C = Finset.range (n + 1) ∧
   A ∩ B = ∅ ∧ A ∩ C = ∅ ∧ B ∩ C = ∅ ∧
   A.sum id = (n * (n + 1)) / 6 ∧ B.sum id = (n * (n + 1)) / 6 ∧ C.sum id = (n * (n + 1)) / 6) :=
sorry

end weights_divide_three_piles_l44_44321


namespace cousin_points_correct_l44_44846

-- Conditions translated to definitions
def paul_points : ℕ := 3103
def total_points : ℕ := 5816

-- Dependent condition to get cousin's points
def cousin_points : ℕ := total_points - paul_points

-- The goal of our proof problem
theorem cousin_points_correct : cousin_points = 2713 :=
by
    sorry

end cousin_points_correct_l44_44846


namespace fewest_tiles_needed_to_cover_rectangle_l44_44647

noncomputable def height_of_equilateral_triangle (side_length : ℝ) : ℝ :=
  (Real.sqrt 3 / 2) * side_length

noncomputable def area_of_equilateral_triangle (side_length : ℝ) : ℝ :=
  (1 / 2) * side_length * height_of_equilateral_triangle side_length

noncomputable def area_of_floor_in_square_inches (length_in_feet : ℝ) (width_in_feet : ℝ) : ℝ :=
  length_in_feet * width_in_feet * (12 * 12)

noncomputable def number_of_tiles_required (floor_area : ℝ) (tile_area : ℝ) : ℝ :=
  floor_area / tile_area

theorem fewest_tiles_needed_to_cover_rectangle :
  number_of_tiles_required (area_of_floor_in_square_inches 3 4) (area_of_equilateral_triangle 2) = 997 := 
by
  sorry

end fewest_tiles_needed_to_cover_rectangle_l44_44647


namespace quadratic_inequality_l44_44590

-- Define the quadratic function and conditions
variables {a b c x0 y1 y2 y3 : ℝ}
variables (A : (a * x0^2 + b * x0 + c = 0))
variables (B : (a * (-2)^2 + b * (-2) + c = 0))
variables (C : (a + b + c) * (4 * a + 2 * b + c) < 0)
variables (D : a > 0)
variables (E1 : y1 = a * (-1)^2 + b * (-1) + c)
variables (E2 : y2 = a * (- (sqrt 2) / 2)^2 + b * (- (sqrt 2) / 2) + c)
variables (E3 : y3 = a * 1^2 + b * 1 + c)

-- Prove that y3 > y1 > y2
theorem quadratic_inequality : y3 > y1 ∧ y1 > y2 := by 
  sorry

end quadratic_inequality_l44_44590


namespace gem_stone_necklaces_sold_l44_44132

-- Definitions and conditions
def bead_necklaces : ℕ := 7
def total_earnings : ℝ := 90
def price_per_necklace : ℝ := 9

-- Theorem to prove the number of gem stone necklaces sold
theorem gem_stone_necklaces_sold : 
  ∃ (G : ℕ), G * price_per_necklace = total_earnings - (bead_necklaces * price_per_necklace) ∧ G = 3 :=
by
  sorry

end gem_stone_necklaces_sold_l44_44132


namespace problem_statement_l44_44700

/-- Define the sequence of numbers spoken by Jo and Blair. -/
def next_number (n : ℕ) : ℕ :=
if n % 2 = 1 then (n + 1) / 2 else n / 2

/-- Helper function to compute the 21st number said. -/
noncomputable def twenty_first_number : ℕ :=
(21 + 1) / 2

/-- Statement of the problem in Lean 4. -/
theorem problem_statement : twenty_first_number = 11 := by
  sorry

end problem_statement_l44_44700


namespace purely_imaginary_solution_l44_44551

noncomputable def complex_number_is_purely_imaginary (m : ℝ) : Prop :=
  (m^2 - 2 * m - 3 = 0) ∧ (m + 1 ≠ 0)

theorem purely_imaginary_solution (m : ℝ) (h : complex_number_is_purely_imaginary m) : m = 3 := by
  sorry

end purely_imaginary_solution_l44_44551


namespace number_of_permutations_l44_44867

noncomputable def num_satisfying_permutations : ℕ :=
  Nat.choose 15 7

theorem number_of_permutations : num_satisfying_permutations = 6435 := by
  sorry

end number_of_permutations_l44_44867


namespace cos_alpha_in_second_quadrant_l44_44439

theorem cos_alpha_in_second_quadrant (α : ℝ) (hα : π / 2 < α ∧ α < π) (h_tan : Real.tan α = -1 / 2) :
  Real.cos α = -2 * Real.sqrt 5 / 5 :=
by
  sorry

end cos_alpha_in_second_quadrant_l44_44439


namespace page_added_twice_is_33_l44_44513

noncomputable def sum_first_n (n : ℕ) : ℕ :=
  n * (n + 1) / 2

theorem page_added_twice_is_33 :
  ∃ n : ℕ, ∃ m : ℕ, sum_first_n n + m = 1986 ∧ 1 ≤ m ∧ m ≤ n → m = 33 := 
by {
  sorry
}

end page_added_twice_is_33_l44_44513


namespace scientific_notation_correct_l44_44167

def distance_moon_km : ℕ := 384000

def scientific_notation (n : ℕ) : ℝ := 3.84 * 10^5

theorem scientific_notation_correct : scientific_notation distance_moon_km = 3.84 * 10^5 := by
  sorry

end scientific_notation_correct_l44_44167


namespace f_at_4_l44_44183

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry
noncomputable def g_inv : ℝ → ℝ := sorry

axiom h1 : ∀ x : ℝ, f (x-1) = g_inv (x-3)
axiom h2 : ∀ x : ℝ, g_inv (g x) = x
axiom h3 : ∀ x : ℝ, g (g_inv x) = x
axiom h4 : g 5 = 2005

theorem f_at_4 : f 4 = 2008 :=
by
  sorry

end f_at_4_l44_44183


namespace savings_with_discount_l44_44235

theorem savings_with_discount :
  let original_price := 3.00
  let discount_rate := 0.30
  let discounted_price := original_price * (1 - discount_rate)
  let number_of_notebooks := 7
  let total_cost_without_discount := number_of_notebooks * original_price
  let total_cost_with_discount := number_of_notebooks * discounted_price
  total_cost_without_discount - total_cost_with_discount = 6.30 :=
by
  sorry

end savings_with_discount_l44_44235


namespace no_eight_consecutive_sums_in_circle_l44_44711

theorem no_eight_consecutive_sums_in_circle :
  ¬ ∃ (arrangement : Fin 8 → ℕ) (sums : Fin 8 → ℤ),
      (∀ i, 1 ≤ arrangement i ∧ arrangement i ≤ 8) ∧
      (∀ i, sums i = arrangement i + arrangement (⟨(i + 1) % 8, sorry⟩)) ∧
      (∃ (n : ℤ), 
        (sums 0 = n - 3) ∧ 
        (sums 1 = n - 2) ∧ 
        (sums 2 = n - 1) ∧ 
        (sums 3 = n) ∧ 
        (sums 4 = n + 1) ∧ 
        (sums 5 = n + 2) ∧ 
        (sums 6 = n + 3) ∧ 
        (sums 7 = n + 4)) := 
sorry

end no_eight_consecutive_sums_in_circle_l44_44711


namespace exists_odd_k_l44_44092

noncomputable def f (n : ℕ) : ℕ :=
sorry

theorem exists_odd_k : 
  (∀ m n : ℕ, f (m * n) = f m * f n) → 
  (∀ m n : ℕ, (m + n) ∣ (f m + f n)) → 
  ∃ k : ℕ, (k % 2 = 1) ∧ (∀ n : ℕ, f n = n ^ k) :=
sorry

end exists_odd_k_l44_44092


namespace objects_meeting_time_l44_44480

theorem objects_meeting_time 
  (initial_velocity : ℝ) (g : ℝ) (t_delay : ℕ) (t_meet : ℝ) 
  (hv : initial_velocity = 120) 
  (hg : g = 9.8) 
  (ht : t_delay = 5)
  : t_meet = 14.74 :=
sorry

end objects_meeting_time_l44_44480


namespace intercepts_of_line_l44_44809

theorem intercepts_of_line :
  (∀ x y : ℝ, (x = 4 ∨ y = -3) → (x / 4 - y / 3 = 1)) ∧ (∀ x y : ℝ, (x / 4 = 1 ∧ y = 0) ∧ (x = 0 ∧ y / 3 = -1)) :=
by
  sorry

end intercepts_of_line_l44_44809


namespace daily_coffee_machine_cost_l44_44550

def coffee_machine_cost := 200 -- $200
def discount := 20 -- $20
def daily_coffee_cost := 2 * 4 -- $8/day
def days_to_pay_off := 36 -- 36 days

theorem daily_coffee_machine_cost :
  (days_to_pay_off * daily_coffee_cost - (coffee_machine_cost - discount)) / days_to_pay_off = 3 := 
by
  -- Using the given conditions: 
  -- coffee_machine_cost = 200
  -- discount = 20
  -- daily_coffee_cost = 8
  -- days_to_pay_off = 36
  sorry

end daily_coffee_machine_cost_l44_44550


namespace initial_volume_of_solution_is_six_l44_44012

theorem initial_volume_of_solution_is_six
  (V : ℝ)
  (h1 : 0.30 * V + 2.4 = 0.50 * (V + 2.4)) :
  V = 6 :=
by
  sorry

end initial_volume_of_solution_is_six_l44_44012


namespace solve_for_q_l44_44039

variable (k h q : ℝ)

-- Conditions given in the problem
axiom cond1 : (3 / 4) = (k / 48)
axiom cond2 : (3 / 4) = ((h + 36) / 60)
axiom cond3 : (3 / 4) = ((q - 9) / 80)

-- Our goal is to state that q = 69
theorem solve_for_q : q = 69 :=
by
  -- the proof goes here
  sorry

end solve_for_q_l44_44039


namespace all_terms_are_positive_integers_terms_product_square_l44_44823

def seq (x : ℕ → ℕ) : Prop :=
  x 1 = 1 ∧
  x 2 = 4 ∧
  ∀ n > 1, x n = Nat.sqrt (x (n - 1) * x (n + 1) + 1)

theorem all_terms_are_positive_integers (x : ℕ → ℕ) (h : seq x) : ∀ n, x n > 0 :=
sorry

theorem terms_product_square (x : ℕ → ℕ) (h : seq x) : ∀ n ≥ 1, ∃ k, 2 * x n * x (n + 1) + 1 = k ^ 2 :=
sorry

end all_terms_are_positive_integers_terms_product_square_l44_44823


namespace parkway_girls_not_playing_soccer_l44_44961

theorem parkway_girls_not_playing_soccer (total_students boys soccer_students : ℕ) 
    (percent_boys_playing_soccer : ℕ) 
    (h1 : total_students = 420)
    (h2 : boys = 312)
    (h3 : soccer_students = 250)
    (h4 : percent_boys_playing_soccer = 86) :
   (total_students - boys - (soccer_students - soccer_students * percent_boys_playing_soccer / 100)) = 73 :=
by sorry

end parkway_girls_not_playing_soccer_l44_44961


namespace simplify_expression_l44_44393

variable (b : ℝ)

theorem simplify_expression (b : ℝ) : (3 * b - 3 - 5 * b) / 3 = - (2 / 3) * b - 1 :=
by
  sorry

end simplify_expression_l44_44393


namespace hyperbola_eccentricity_l44_44142

theorem hyperbola_eccentricity (a b c : ℝ) (h_a : a > 0) (h_b : b > 0)
  (h_hyperbola: ∀ x y : ℝ, (x^2 / a^2) - (y^2 / b^2) = 1)
  (h_asymptotes_l1: ∀ x : ℝ, y = (b / a) * x)
  (h_asymptotes_l2: ∀ x : ℝ, y = -(b / a) * x)
  (h_focus: c^2 = a^2 + b^2)
  (h_symmetric: ∀ m : ℝ, m = -c / 2 ∧ (m, (b * c) / (2 * a)) ∈ { p : ℝ × ℝ | p.2 = -(b / a) * p.1 }) :
  (c / a) = 2 := sorry

end hyperbola_eccentricity_l44_44142


namespace num_integers_satisfying_conditions_l44_44835

theorem num_integers_satisfying_conditions : 
  ∃ n : ℕ, 
    (120 < n) ∧ (n < 250) ∧ (n % 5 = n % 7) :=
sorry

axiom num_integers_with_conditions : ℕ
@[simp] lemma val_num_integers_with_conditions : num_integers_with_conditions = 25 :=
sorry

end num_integers_satisfying_conditions_l44_44835


namespace visible_steps_on_escalator_l44_44263

variable (steps_visible : ℕ) -- The number of steps visible on the escalator
variable (al_steps : ℕ := 150) -- Al walks down 150 steps
variable (bob_steps : ℕ := 75) -- Bob walks up 75 steps
variable (al_speed : ℕ := 3) -- Al's walking speed
variable (bob_speed : ℕ := 1) -- Bob's walking speed
variable (escalator_speed : ℚ) -- The speed of the escalator

theorem visible_steps_on_escalator : steps_visible = 120 :=
by
  -- Define times taken by Al and Bob
  let al_time := al_steps / al_speed
  let bob_time := bob_steps / bob_speed

  -- Define effective speeds considering escalator speed 'escalator_speed'
  let al_effective_speed := al_speed - escalator_speed
  let bob_effective_speed := bob_speed + escalator_speed

  -- Calculate the total steps walked if the escalator was stopped (same total steps)
  have al_total_steps := al_effective_speed * al_time
  have bob_total_steps := bob_effective_speed * bob_time

  -- Set up the equation
  have eq := al_total_steps = bob_total_steps

  -- Substitute and solve for escalator_speed
  sorry

end visible_steps_on_escalator_l44_44263


namespace pencil_price_units_l44_44774

def pencil_price_in_units (pencil_price : ℕ) : ℚ := pencil_price / 10000

theorem pencil_price_units 
  (price_of_pencil : ℕ) 
  (h1 : price_of_pencil = 5000 - 20) : 
  pencil_price_in_units price_of_pencil = 0.5 := 
by
  sorry

end pencil_price_units_l44_44774


namespace sqrt_product_simplification_l44_44222

theorem sqrt_product_simplification (q : ℝ) : 
  Real.sqrt (15 * q) * Real.sqrt (10 * q^3) * Real.sqrt (14 * q^5) = 10 * q^4 * Real.sqrt (21 * q) := 
by 
  sorry

end sqrt_product_simplification_l44_44222


namespace trigonometric_identity_l44_44230

theorem trigonometric_identity
  (α : Real)
  (h : 3 * Real.sin α + Real.cos α = 0) :
  1 / (Real.cos α ^ 2 + 2 * Real.sin α * Real.cos α) = 10 / 3 :=
by
  sorry

end trigonometric_identity_l44_44230


namespace intersection_A_B_l44_44589

section
  def A : Set ℤ := {-2, 0, 1}
  def B : Set ℤ := {x | x^2 > 1}
  theorem intersection_A_B : A ∩ B = {-2} := 
  by
    sorry
end

end intersection_A_B_l44_44589


namespace susan_bought_36_items_l44_44438

noncomputable def cost_per_pencil : ℝ := 0.25
noncomputable def cost_per_pen : ℝ := 0.80
noncomputable def pencils_bought : ℕ := 16
noncomputable def total_spent : ℝ := 20.0

theorem susan_bought_36_items :
  ∃ (pens_bought : ℕ), pens_bought * cost_per_pen + pencils_bought * cost_per_pencil = total_spent ∧ pencils_bought + pens_bought = 36 := 
sorry

end susan_bought_36_items_l44_44438


namespace polynomial_factorization_l44_44523

theorem polynomial_factorization (x : ℝ) :
  (x^2 + 4*x + 3) * (x^2 + 8*x + 15) + (x^2 + 6*x - 8) = (x^2 + 6*x + 1) * (x^2 + 6*x + 37) :=
  sorry

end polynomial_factorization_l44_44523


namespace find_f_pi_over_4_l44_44576

variable (f : ℝ → ℝ)
variable (h : ∀ x, f x = f (Real.pi / 4) * Real.cos x + Real.sin x)

theorem find_f_pi_over_4 : f (Real.pi / 4) = 1 := by
  sorry

end find_f_pi_over_4_l44_44576


namespace star_15_star_eq_neg_15_l44_44412

-- Define the operations as given
def y_star (y : ℤ) := 9 - y
def star_y (y : ℤ) := y - 9

-- The theorem stating the required proof
theorem star_15_star_eq_neg_15 : star_y (y_star 15) = -15 :=
by
  sorry

end star_15_star_eq_neg_15_l44_44412


namespace smallest_x_y_sum_l44_44079

theorem smallest_x_y_sum (x y : ℕ) (h1 : x ≠ y) (h2 : x > 0) (h3 : y > 0) (h4 : 1 / (x : ℚ) + 1 / (y : ℚ) = 1 / 10) : x + y = 45 :=
by
  sorry

end smallest_x_y_sum_l44_44079


namespace translation_equivalence_l44_44613

def f₁ (x : ℝ) : ℝ := 4 * (x + 3)^2 - 4
def f₂ (x : ℝ) : ℝ := 4 * (x - 3)^2 + 4

theorem translation_equivalence :
  (∀ x : ℝ, f₁ (x + 6) = 4 * (x + 9)^2 + 4) ∧
  (∀ x : ℝ, f₁ x  - 8 = 4 * (x + 3)^2 - 4) :=
by sorry

end translation_equivalence_l44_44613


namespace remainder_product_div_6_l44_44956

theorem remainder_product_div_6 :
  (3 * 7 * 13 * 17 * 23 * 27 * 33 * 37 * 43 * 47 * 53 * 57 * 63 * 67 * 73 * 77 * 83 * 87 * 93 * 97 
   * 103 * 107 * 113 * 117 * 123 * 127 * 133 * 137 * 143 * 147 * 153 * 157 * 163 * 167 * 173 
   * 177 * 183 * 187 * 193 * 197) % 6 = 3 := 
by 
  -- basic info about modulo arithmetic and properties of sequences
  sorry

end remainder_product_div_6_l44_44956


namespace volume_of_cylindrical_block_l44_44723

variable (h_cylindrical : ℕ) (combined_value : ℝ)

theorem volume_of_cylindrical_block (h_cylindrical : ℕ) (combined_value : ℝ):
  h_cylindrical = 3 → combined_value / 5 * h_cylindrical = 15.42 := by
suffices combined_value / 5 = 5.14 from sorry
suffices 5.14 * 3 = 15.42 from sorry
suffices h_cylindrical = 3 from sorry
suffices 25.7 = combined_value from sorry
sorry

end volume_of_cylindrical_block_l44_44723


namespace cubic_ineq_solution_l44_44972

theorem cubic_ineq_solution (x : ℝ) :
  (4 < x ∧ x < 4 + 2 * Real.sqrt 3) ∨ (x > 4 + 2 * Real.sqrt 3) → (x^3 - 12 * x^2 + 44 * x - 16 > 0) :=
by
  sorry

end cubic_ineq_solution_l44_44972


namespace find_a_b_find_m_l44_44950

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := a * x^3 + b * x^2

theorem find_a_b (a b : ℝ) (h₁ : f 1 a b = 4)
  (h₂ : 3 * a + 2 * b = 9) : a = 1 ∧ b = 3 :=
by
  sorry

theorem find_m (m : ℝ) (h : ∀ x, (m ≤ x ∧ x ≤ m + 1) → (3 * x^2 + 6 * x > 0)) :
  m ≥ 0 ∨ m ≤ -3 :=
by
  sorry

end find_a_b_find_m_l44_44950


namespace total_payment_l44_44724

-- Define the basic conditions
def hours_first_day : ℕ := 10
def hours_second_day : ℕ := 8
def hours_third_day : ℕ := 15
def hourly_wage : ℕ := 10
def number_of_workers : ℕ := 2

-- Define the proof problem
theorem total_payment : 
  (hours_first_day + hours_second_day + hours_third_day) * hourly_wage * number_of_workers = 660 := 
by
  sorry

end total_payment_l44_44724


namespace sum_of_corners_is_164_l44_44616

section CheckerboardSum

-- Define the total number of elements in the 9x9 grid
def num_elements := 81

-- Define the positions of the corners
def top_left : ℕ := 1
def top_right : ℕ := 9
def bottom_left : ℕ := 73
def bottom_right : ℕ := 81

-- Define the sum of the corners
def corner_sum : ℕ := top_left + top_right + bottom_left + bottom_right

-- State the theorem
theorem sum_of_corners_is_164 : corner_sum = 164 :=
by
  exact sorry

end CheckerboardSum

end sum_of_corners_is_164_l44_44616


namespace daughter_age_in_3_years_l44_44014

theorem daughter_age_in_3_years (mother_age_now : ℕ) (h1 : mother_age_now = 41)
  (h2 : ∃ daughter_age_5_years_ago : ℕ, mother_age_now - 5 = 2 * daughter_age_5_years_ago) :
  ∃ daughter_age_in_3_years : ℕ, daughter_age_in_3_years = 26 :=
by {
  sorry
}

end daughter_age_in_3_years_l44_44014


namespace value_of_x_abs_not_positive_l44_44678

theorem value_of_x_abs_not_positive {x : ℝ} : |4 * x - 6| = 0 → x = 3 / 2 :=
by
  sorry

end value_of_x_abs_not_positive_l44_44678


namespace discriminant_nonnegative_l44_44813

theorem discriminant_nonnegative {x : ℤ} (a : ℝ) (h₁ : x^2 * (49 - 40 * x^2) ≥ 0) :
  a = 0 ∨ a = 1 ∨ a = -1 ∨ a = 5/2 ∨ a = -5/2 := sorry

end discriminant_nonnegative_l44_44813


namespace total_pages_l44_44689

def Johnny_word_count : ℕ := 195
def Madeline_word_count : ℕ := 2 * Johnny_word_count
def Timothy_word_count : ℕ := Madeline_word_count + 50
def Samantha_word_count : ℕ := 3 * Madeline_word_count
def Ryan_word_count : ℕ := Johnny_word_count + 100
def Words_per_page : ℕ := 235

def pages_needed (words : ℕ) : ℕ :=
  if words % Words_per_page = 0 then words / Words_per_page else words / Words_per_page + 1

theorem total_pages :
  pages_needed Johnny_word_count +
  pages_needed Madeline_word_count +
  pages_needed Timothy_word_count +
  pages_needed Samantha_word_count +
  pages_needed Ryan_word_count = 12 :=
  by sorry

end total_pages_l44_44689


namespace number_of_points_l44_44645

theorem number_of_points (a b : ℤ) : (|a| = 3 ∧ |b| = 2) → ∃! (P : ℤ × ℤ), P = (a, b) :=
by sorry

end number_of_points_l44_44645


namespace simplified_identity_l44_44584

theorem simplified_identity :
  (12 : ℚ) * ( (1/3 : ℚ) + (1/4) + (1/6) + (1/12) )⁻¹ = 72 / 5 :=
  sorry

end simplified_identity_l44_44584


namespace original_average_is_24_l44_44228

theorem original_average_is_24
  (A : ℝ)
  (h1 : ∀ n : ℕ, n = 7 → 35 * A = 7 * 120) :
  A = 24 :=
by
  sorry

end original_average_is_24_l44_44228


namespace Annika_hiking_rate_is_correct_l44_44043

def AnnikaHikingRate
  (distance_partial_east distance_total_east : ℕ)
  (time_back_to_start : ℕ)
  (equality_rate : Nat) : Prop :=
  distance_partial_east = 2750 / 1000 ∧
  distance_total_east = 3500 / 1000 ∧
  time_back_to_start = 51 ∧
  equality_rate = 34

theorem Annika_hiking_rate_is_correct :
  ∃ R : ℕ, ∀ d1 d2 t,
  AnnikaHikingRate d1 d2 t R → R = 34 :=
by
  sorry

end Annika_hiking_rate_is_correct_l44_44043


namespace find_x_average_is_3_l44_44444

theorem find_x_average_is_3 (x : ℝ) (h : (2 + 4 + 1 + 3 + x) / 5 = 3) : x = 5 :=
sorry

end find_x_average_is_3_l44_44444


namespace positive_integer_power_of_two_l44_44226

theorem positive_integer_power_of_two (n : ℕ) (hn : 0 < n) :
  (∃ m : ℤ, (2^n - 1) ∣ (m^2 + 9)) ↔ (∃ k : ℕ, n = 2^k) :=
by
  sorry

end positive_integer_power_of_two_l44_44226


namespace find_k_l44_44986

theorem find_k (k : ℝ) :
    (1 - 7) * (k - 3) = (3 - k) * (7 - 1) → k = 6.5 :=
by
sorry

end find_k_l44_44986


namespace annual_parking_savings_l44_44219

theorem annual_parking_savings :
  let weekly_rate := 10
  let monthly_rate := 40
  let weeks_in_year := 52
  let months_in_year := 12
  let annual_weekly_cost := weekly_rate * weeks_in_year
  let annual_monthly_cost := monthly_rate * months_in_year
  let savings := annual_weekly_cost - annual_monthly_cost
  savings = 40 := by
{
  sorry
}

end annual_parking_savings_l44_44219


namespace ratio_of_second_to_third_l44_44255

theorem ratio_of_second_to_third (A B C : ℕ) (h1 : A + B + C = 98) (h2 : A * 3 = B * 2) (h3 : B = 30) :
  B * 8 = C * 5 :=
by
  sorry

end ratio_of_second_to_third_l44_44255


namespace hyperbola_center_l44_44131

theorem hyperbola_center :
  (∃ h k : ℝ,
    (∀ x y : ℝ, ((4 * x - 8) / 9)^2 - ((5 * y + 5) / 7)^2 = 1 ↔ (x - h)^2 / (81 / 16) - (y - k)^2 / (49 / 25) = 1) ∧
    (h = 2) ∧ (k = -1)) :=
sorry

end hyperbola_center_l44_44131


namespace min_cost_29_disks_l44_44268

theorem min_cost_29_disks
  (price_single : ℕ := 20) 
  (price_pack_10 : ℕ := 111) 
  (price_pack_25 : ℕ := 265) :
  ∃ cost : ℕ, cost ≥ (price_pack_10 + price_pack_10 + price_pack_10) 
              ∧ cost ≤ (price_pack_25 + price_single * 4) 
              ∧ cost = 333 := 
by
  sorry

end min_cost_29_disks_l44_44268


namespace probability_of_exactly_two_dice_showing_3_l44_44828

-- Definition of the problem conditions
def n_dice : ℕ := 5
def sides : ℕ := 5
def prob_showing_3 : ℚ := 1/5
def prob_not_showing_3 : ℚ := 4/5
def way_to_choose_2_of_5 : ℕ := Nat.choose 5 2

-- Lean proof problem statement
theorem probability_of_exactly_two_dice_showing_3 : 
  (10 : ℚ) * (prob_showing_3 ^ 2) * (prob_not_showing_3 ^ 3) = 640 / 3125 := 
by sorry

end probability_of_exactly_two_dice_showing_3_l44_44828


namespace option_A_option_C_l44_44810

/-- Definition of the set M such that M = {a | a = x^2 - y^2, x, y ∈ ℤ} -/
def M := {a : ℤ | ∃ x y : ℤ, a = x^2 - y^2}

/-- Definition of the set B such that B = {b | b = 2n + 1, n ∈ ℕ} -/
def B := {b : ℤ | ∃ n : ℕ, b = 2 * n + 1}

theorem option_A (a1 a2 : ℤ) (ha1 : a1 ∈ M) (ha2 : a2 ∈ M) : a1 * a2 ∈ M := sorry

theorem option_C : B ⊆ M := sorry

end option_A_option_C_l44_44810


namespace hexagons_formed_square_z_l44_44425

theorem hexagons_formed_square_z (a b s z : ℕ) (hexagons_congruent : a = 9 ∧ b = 16 ∧ s = 12 ∧ z = 4): 
(z = 4) := by
  sorry

end hexagons_formed_square_z_l44_44425


namespace movie_friends_l44_44310

noncomputable def movie_only (M P G MP MG PG MPG : ℕ) : Prop :=
  let total_M := 20
  let total_P := 20
  let total_G := 5
  let total_students := 31
  (MP = 4) ∧ 
  (MG = 2) ∧ 
  (PG = 0) ∧ (MPG = 2) ∧ 
  (M + MP + MG + MPG = total_M) ∧ 
  (P + MP + PG + MPG = total_P) ∧ 
  (G + MG + PG + MPG = total_G) ∧ 
  (M + P + G + MP + MG + PG + MPG = total_students) ∧ 
  (M = 12)

theorem movie_friends (M P G MP MG PG MPG : ℕ) : movie_only M P G MP MG PG MPG := 
by 
  sorry

end movie_friends_l44_44310


namespace problem_statement_l44_44612

-- Definitions
def MagnitudeEqual : Prop := (2.4 : ℝ) = (2.40 : ℝ)
def CountUnit2_4 : Prop := (0.1 : ℝ) = 2.4 / 24
def CountUnit2_40 : Prop := (0.01 : ℝ) = 2.40 / 240

-- Theorem statement
theorem problem_statement : MagnitudeEqual ∧ CountUnit2_4 ∧ CountUnit2_40 → True := by
  intros
  sorry

end problem_statement_l44_44612


namespace quadratic_roots_difference_l44_44484

theorem quadratic_roots_difference (a b : ℝ) :
  (5 * a^2 - 30 * a + 45 = 0) ∧ (5 * b^2 - 30 * b + 45 = 0) → (a - b)^2 = 0 :=
by
  sorry

end quadratic_roots_difference_l44_44484


namespace more_students_than_guinea_pigs_l44_44865

theorem more_students_than_guinea_pigs (students_per_classroom guinea_pigs_per_classroom classrooms : ℕ)
  (h1 : students_per_classroom = 24) 
  (h2 : guinea_pigs_per_classroom = 3) 
  (h3 : classrooms = 6) : 
  (students_per_classroom * classrooms) - (guinea_pigs_per_classroom * classrooms) = 126 := 
by
  sorry

end more_students_than_guinea_pigs_l44_44865


namespace sets_are_equal_l44_44244

theorem sets_are_equal :
  let M := {x | ∃ k : ℤ, x = 2 * k + 1}
  let N := {x | ∃ k : ℤ, x = 4 * k + 1 ∨ x = 4 * k - 1}
  M = N :=
by
  sorry

end sets_are_equal_l44_44244


namespace primes_divisible_by_3_percentage_is_12_5_l44_44473

-- Definition of the primes less than 20
def primes_less_than_20 : List Nat := [2, 3, 5, 7, 11, 13, 17, 19]

-- Definition of the prime numbers from the list that are divisible by 3
def primes_divisible_by_3 : List Nat := primes_less_than_20.filter (λ p => p % 3 = 0)

-- Total number of primes less than 20
def total_primes_less_than_20 : Nat := primes_less_than_20.length

-- Total number of primes less than 20 that are divisible by 3
def total_primes_divisible_by_3 : Nat := primes_divisible_by_3.length

-- The percentage of prime numbers less than 20 that are divisible by 3
noncomputable def percentage_primes_divisible_by_3 : Float := 
  (total_primes_divisible_by_3.toFloat / total_primes_less_than_20.toFloat) * 100

theorem primes_divisible_by_3_percentage_is_12_5 :
  percentage_primes_divisible_by_3 = 12.5 := by
  sorry

end primes_divisible_by_3_percentage_is_12_5_l44_44473


namespace original_combined_price_l44_44599

theorem original_combined_price (C S : ℝ)
  (hC_new : (C + 0.25 * C) = 12.5)
  (hS_new : (S + 0.50 * S) = 13.5) :
  (C + S) = 19 := by
  -- sorry makes sure to skip the proof
  sorry

end original_combined_price_l44_44599


namespace chord_length_proof_tangent_lines_through_M_l44_44391

noncomputable def circle_C (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 4

noncomputable def line_l (x y : ℝ) : Prop := 2*x - y + 4 = 0

noncomputable def point_M : (ℝ × ℝ) := (3, 1)

noncomputable def chord_length : ℝ := 4 * Real.sqrt (5) / 5

noncomputable def tangent_line_1 (x y : ℝ) : Prop := 3*x - 4*y - 5 = 0
noncomputable def tangent_line_2 (x : ℝ) : Prop := x = 3

theorem chord_length_proof :
  ∀ x y : ℝ, circle_C x y → line_l x y → chord_length = 4 * Real.sqrt (5) / 5 :=
by sorry

theorem tangent_lines_through_M :
  ∀ x y : ℝ, circle_C x y → (tangent_line_1 x y ∨ tangent_line_2 x) :=
by sorry

end chord_length_proof_tangent_lines_through_M_l44_44391


namespace find_even_odd_functions_l44_44646

variable {X : Type} [AddGroup X]

def even_function (f : X → X) : Prop :=
∀ x, f (-x) = f x

def odd_function (f : X → X) : Prop :=
∀ x, f (-x) = -f x

theorem find_even_odd_functions
  (f g : X → X)
  (h_even : even_function f)
  (h_odd : odd_function g)
  (h_eq : ∀ x, f x + g x = 0) :
  (∀ x, f x = 0) ∧ (∀ x, g x = 0) :=
sorry

end find_even_odd_functions_l44_44646


namespace second_group_people_l44_44635

theorem second_group_people (x : ℕ) (K : ℕ) (hK : K > 0) :
  (96 - 16 = K * (x + 16) + 6) → (x = 58 ∨ x = 21) :=
by
  intro h
  sorry

end second_group_people_l44_44635


namespace cos_difference_identity_cos_phi_value_l44_44008

variables (α β θ φ : ℝ)
variables (a b : ℝ × ℝ)

-- Part I
theorem cos_difference_identity (hα : 0 ≤ α ∧ α ≤ 2 * Real.pi) (hβ : 0 ≤ β ∧ β ≤ 2 * Real.pi) : 
  Real.cos (α - β) = Real.cos α * Real.cos β + Real.sin α * Real.sin β :=
sorry

-- Part II
theorem cos_phi_value (hθ : 0 < θ ∧ θ < Real.pi / 2) (hφ : 0 < φ ∧ φ < Real.pi / 2)
  (ha : a = (Real.sin θ, -2)) (hb : b = (1, Real.cos θ)) (dot_ab_zero : a.1 * b.1 + a.2 * b.2 = 0)
  (h_sin_diff : Real.sin (theta - phi) = Real.sqrt 10 / 10) :
  Real.cos φ = Real.sqrt 2 / 2 :=
sorry

end cos_difference_identity_cos_phi_value_l44_44008


namespace smallest_sum_ending_2050306_l44_44704

/--
Given nine consecutive natural numbers starting at n,
prove that the smallest sum of these nine numbers ending in 2050306 is 22050306.
-/
theorem smallest_sum_ending_2050306 
  (n : ℕ) 
  (hn : ∃ m : ℕ, 9 * m = (n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) + (n + 6) + (n + 7) + (n + 8)) ∧ 
                 (9 * m) % 10^7 = 2050306) :
  (n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) + (n + 6) + (n + 7) + (n + 8)) = 22050306 := 
sorry

end smallest_sum_ending_2050306_l44_44704


namespace number_of_books_in_box_l44_44126

theorem number_of_books_in_box (total_weight : ℕ) (weight_per_book : ℕ) 
  (h1 : total_weight = 42) (h2 : weight_per_book = 3) : total_weight / weight_per_book = 14 :=
by sorry

end number_of_books_in_box_l44_44126


namespace problem1_l44_44516

theorem problem1 (a : ℝ) (h : Real.sqrt a + 1 / Real.sqrt a = 3) :
  (a ^ 2 + 1 / a ^ 2 + 3) / (4 * a + 1 / (4 * a)) = 10 * Real.sqrt 5 := sorry

end problem1_l44_44516


namespace equation_of_line_l44_44320

-- Define the parabola equation
def parabola (x : ℝ) : ℝ := x^2 + 4 * x + 4

-- Define the line equation with parameters m and b
def line (m b x : ℝ) : ℝ := m * x + b

-- Define the point of intersection with the parabola on the line x = k
def intersection_point_parabola (k : ℝ) : ℝ := parabola k

-- Define the point of intersection with the line on the line x = k
def intersection_point_line (m b k : ℝ) : ℝ := line m b k

-- Define the vertical distance between the points on x = k
def vertical_distance (k m b : ℝ) : ℝ :=
  abs ((parabola k) - (line m b k))

-- Define the condition that vertical distance is exactly 4 units
def intersection_distance_condition (k m b : ℝ) : Prop :=
  vertical_distance k m b = 4

-- The line passes through point (2, 8)
def passes_through_point (m b : ℝ) : Prop :=
  line m b 2 = 8

-- Non-zero y-intercept condition
def non_zero_intercept (b : ℝ) : Prop := 
  b ≠ 0

-- The final theorem stating the required equation of the line
theorem equation_of_line (m b : ℝ) (h1 : ∃ k, intersection_distance_condition k m b)
  (h2 : passes_through_point m b) (h3 : non_zero_intercept b) : 
  (m = 12 ∧ b = -16) :=
by
  sorry

end equation_of_line_l44_44320


namespace least_positive_integer_solution_l44_44601

theorem least_positive_integer_solution :
  ∃ x : ℕ, (x + 7391) % 12 = 167 % 12 ∧ x = 8 :=
by 
  sorry

end least_positive_integer_solution_l44_44601


namespace fibonacci_units_digit_l44_44899

def fibonacci (n : ℕ) : ℕ :=
match n with
| 0     => 4
| 1     => 3
| (n+2) => fibonacci (n+1) + fibonacci n

def units_digit (n : ℕ) : ℕ :=
n % 10

theorem fibonacci_units_digit : units_digit (fibonacci (fibonacci 10)) = 3 := by
  sorry

end fibonacci_units_digit_l44_44899


namespace back_wheel_revolutions_l44_44432

theorem back_wheel_revolutions
  (r_front : ℝ) (r_back : ℝ) (rev_front : ℝ) (r_front_eq : r_front = 3)
  (r_back_eq : r_back = 0.5) (rev_front_eq : rev_front = 50) :
  let C_front := 2 * Real.pi * r_front
  let D_front := C_front * rev_front
  let C_back := 2 * Real.pi * r_back
  let rev_back := D_front / C_back
  rev_back = 300 := by
  sorry

end back_wheel_revolutions_l44_44432


namespace proof_l44_44430

variable {S : Type} 
variable (op : S → S → S)

-- Condition given in the problem
def condition (a b : S) : Prop :=
  op (op a b) a = b

-- Statement to be proven
theorem proof (h : ∀ a b : S, condition op a b) :
  ∀ a b : S, op a (op b a) = b :=
by
  intros a b
  sorry

end proof_l44_44430


namespace find_second_group_of_men_l44_44476

noncomputable def work_rate_of_man := ℝ
noncomputable def work_rate_of_woman := ℝ

variables (m w : ℝ)

-- Condition 1: 3 men and 8 women complete the task in the same time as x men and 2 women.
axiom condition1 (x : ℝ) : 3 * m + 8 * w = x * m + 2 * w

-- Condition 2: 2 men and 3 women complete half the task in the same time as 3 men and 8 women completing the whole task.
axiom condition2 : 2 * m + 3 * w = 0.5 * (3 * m + 8 * w)

theorem find_second_group_of_men (x : ℝ) (m w : ℝ) (h1 : 0.5 * m = w)
  (h2 : 3 * m + 8 * w = x * m + 2 * w) : x = 6 :=
by {
  sorry
}

end find_second_group_of_men_l44_44476


namespace peter_large_glasses_l44_44296

theorem peter_large_glasses (cost_small cost_large total_money small_glasses change num_large_glasses : ℕ)
    (h1 : cost_small = 3)
    (h2 : cost_large = 5)
    (h3 : total_money = 50)
    (h4 : small_glasses = 8)
    (h5 : change = 1)
    (h6 : total_money - change = 49)
    (h7 : small_glasses * cost_small = 24)
    (h8 : 49 - 24 = 25)
    (h9 : 25 / cost_large = 5) :
  num_large_glasses = 5 :=
by
  sorry

end peter_large_glasses_l44_44296


namespace youngest_is_dan_l44_44000

notation "alice" => 21
notation "bob" => 18
notation "clare" => 22
notation "dan" => 16
notation "eve" => 28

theorem youngest_is_dan :
  let a := alice
  let b := bob
  let c := clare
  let d := dan
  let e := eve
  a + b = 39 ∧
  b + c = 40 ∧
  c + d = 38 ∧
  d + e = 44 ∧
  a + b + c + d + e = 105 →
  min (min (min (min a b) c) d) e = d :=
by {
  sorry
}

end youngest_is_dan_l44_44000


namespace non_working_games_l44_44744

def total_games : ℕ := 30
def working_games : ℕ := 17

theorem non_working_games :
  total_games - working_games = 13 := 
by 
  sorry

end non_working_games_l44_44744


namespace additional_increment_charge_cents_l44_44572

-- Conditions as definitions
def first_increment_charge_cents : ℝ := 3.10
def total_charge_8_minutes_cents : ℝ := 18.70
def total_minutes : ℝ := 8
def increments_per_minute : ℝ := 5
def total_increments : ℝ := total_minutes * increments_per_minute
def remaining_increments : ℝ := total_increments - 1
def remaining_charge_cents : ℝ := total_charge_8_minutes_cents - first_increment_charge_cents

-- Proof problem: What is the charge for each additional 1/5 of a minute?
theorem additional_increment_charge_cents : remaining_charge_cents / remaining_increments = 0.40 := by
  sorry

end additional_increment_charge_cents_l44_44572


namespace equilateral_triangle_area_l44_44756

theorem equilateral_triangle_area (h : ℝ) 
  (height_eq : h = 2 * Real.sqrt 3) :
  ∃ (A : ℝ), A = 4 * Real.sqrt 3 :=
by
  sorry

end equilateral_triangle_area_l44_44756


namespace smallest_composite_no_prime_factors_less_than_20_is_529_l44_44085

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ¬is_prime n

def smallest_prime_factor_greater_than_20 (n : ℕ) : Prop :=
  is_prime n ∧ n > 20 ∧ ∀ p : ℕ, is_prime p ∧ p > 20 → p >= n

def smallest_composite_with_no_prime_factors_less_than_20 (n : ℕ) : Prop :=
  is_composite n ∧ ∀ p : ℕ, is_prime p ∧ p < 20 → ¬ p ∣ n

theorem smallest_composite_no_prime_factors_less_than_20_is_529 :
  smallest_composite_with_no_prime_factors_less_than_20 529 :=
by
  sorry

end smallest_composite_no_prime_factors_less_than_20_is_529_l44_44085


namespace total_words_read_l44_44633

/-- Proof Problem Statement:
  Given the following conditions:
  - Henri has 8 hours to watch movies and read.
  - He watches one movie for 3.5 hours.
  - He watches another movie for 1.5 hours.
  - He watches two more movies with durations of 1.25 hours and 0.75 hours, respectively.
  - He reads for the remaining time after watching movies.
  - For the first 30 minutes of reading, he reads at a speed of 12 words per minute.
  - For the following 20 minutes, his reading speed decreases to 8 words per minute.
  - In the last remaining minutes, his reading speed increases to 15 words per minute.
  Prove that the total number of words Henri reads during his free time is 670.
--/
theorem total_words_read : 8 * 60 - (7 * 60) = 60 ∧
  (30 * 12) + (20 * 8) + ((60 - 30 - 20) * 15) = 670 :=
by
  sorry

end total_words_read_l44_44633


namespace carla_marbles_start_l44_44059

-- Conditions defined as constants
def marblesBought : ℝ := 489.0
def marblesTotalNow : ℝ := 2778.0

-- Theorem statement
theorem carla_marbles_start (marblesBought marblesTotalNow: ℝ) :
  marblesTotalNow - marblesBought = 2289.0 := by
  sorry

end carla_marbles_start_l44_44059


namespace inequality_1_le_x3y3_over_x2y2_squared_le_9_over_8_l44_44477

theorem inequality_1_le_x3y3_over_x2y2_squared_le_9_over_8
  (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
    1 ≤ (x + y) * (x^3 + y^3) / (x^2 + y^2)^2 ∧ (x + y) * (x^3 + y^3) / (x^2 + y^2)^2 ≤ 9 / 8 := 
sorry

end inequality_1_le_x3y3_over_x2y2_squared_le_9_over_8_l44_44477


namespace project_completion_days_l44_44898

theorem project_completion_days (A B C : ℝ) (h1 : 1/A + 1/B = 1/2) (h2 : 1/B + 1/C = 1/4) (h3 : 1/C + 1/A = 1/2.4) : A = 3 :=
by
sorry

end project_completion_days_l44_44898


namespace solve_equation_l44_44730

theorem solve_equation (x : ℝ) :
  (1 / (x ^ 2 + 14 * x - 10)) + (1 / (x ^ 2 + 3 * x - 10)) + (1 / (x ^ 2 - 16 * x - 10)) = 0
  ↔ (x = 5 ∨ x = -2 ∨ x = 2 ∨ x = -5) :=
sorry

end solve_equation_l44_44730


namespace fraction_subtraction_identity_l44_44020

theorem fraction_subtraction_identity (x y : ℕ) (hx : x = 3) (hy : y = 4) : (1 / (x : ℚ) - 1 / (y : ℚ) = 1 / 12) :=
by
  sorry

end fraction_subtraction_identity_l44_44020


namespace number_of_friends_l44_44917

-- Define the conditions
def kendra_packs : ℕ := 7
def tony_packs : ℕ := 5
def pens_per_kendra_pack : ℕ := 4
def pens_per_tony_pack : ℕ := 6
def pens_kendra_keep : ℕ := 3
def pens_tony_keep : ℕ := 3

-- Define the theorem to be proved
theorem number_of_friends 
  (packs_k : ℕ := kendra_packs)
  (packs_t : ℕ := tony_packs)
  (pens_per_pack_k : ℕ := pens_per_kendra_pack)
  (pens_per_pack_t : ℕ := pens_per_tony_pack)
  (kept_k : ℕ := pens_kendra_keep)
  (kept_t : ℕ := pens_tony_keep) :
  packs_k * pens_per_pack_k + packs_t * pens_per_pack_t - (kept_k + kept_t) = 52 :=
by
  sorry

end number_of_friends_l44_44917


namespace expr_value_at_neg2_l44_44487

variable (a b : ℝ)

def expr (x : ℝ) : ℝ := a * x^3 + b * x - 7

theorem expr_value_at_neg2 :
  (expr a b 2 = -19) → (expr a b (-2) = 5) :=
by 
  intro h
  sorry

end expr_value_at_neg2_l44_44487


namespace f_45_g_10_l44_44034

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

axiom f_condition1 (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : f (x * y) = f x / y
axiom g_condition2 (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : g (x + y) = g x + g y
axiom f_15 : f 15 = 10
axiom g_5 : g 5 = 3

theorem f_45 : f 45 = 10 / 3 := sorry
theorem g_10 : g 10 = 6 := sorry

end f_45_g_10_l44_44034


namespace exists_three_distinct_nats_sum_prod_squares_l44_44660

theorem exists_three_distinct_nats_sum_prod_squares :
  ∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
  (∃ (x : ℕ), a + b + c = x^2) ∧ 
  (∃ (y : ℕ), a * b * c = y^2) :=
sorry

end exists_three_distinct_nats_sum_prod_squares_l44_44660


namespace card_2_in_box_Q_l44_44862

theorem card_2_in_box_Q (P Q : Finset ℕ) (hP : P.card = 3) (hQ : Q.card = 5) 
  (hdisjoint : Disjoint P Q) (huniv : P ∪ Q = (Finset.range 9).erase 0)
  (hsum_eq : P.sum id = Q.sum id) :
  2 ∈ Q := 
sorry

end card_2_in_box_Q_l44_44862


namespace radio_price_position_l44_44525

def price_positions (n : ℕ) (total_items : ℕ) (rank_lowest : ℕ) : Prop :=
  rank_lowest = total_items - n + 1

theorem radio_price_position :
  ∀ (n total_items rank_lowest : ℕ),
    total_items = 34 →
    rank_lowest = 21 →
    price_positions n total_items rank_lowest →
    n = 14 :=
by
  intros n total_items rank_lowest h_total h_rank h_pos
  rw [h_total, h_rank] at h_pos
  sorry

end radio_price_position_l44_44525


namespace find_n_l44_44021

theorem find_n (x n : ℝ) (h₁ : x = 1) (h₂ : 5 / (n + 1 / x) = 1) : n = 4 :=
sorry

end find_n_l44_44021


namespace idiom_describes_random_event_l44_44394

-- Define the idioms as propositions.
def FishingForMoonInWater : Prop := ∀ (x : Type), x -> False
def CastlesInTheAir : Prop := ∀ (y : Type), y -> False
def WaitingByStumpForHare : Prop := ∃ (z : Type), True
def CatchingTurtleInJar : Prop := ∀ (w : Type), w -> False

-- Define the main theorem to state that WaitingByStumpForHare describes a random event.
theorem idiom_describes_random_event : WaitingByStumpForHare :=
  sorry

end idiom_describes_random_event_l44_44394


namespace oreos_total_l44_44259

variable (Jordan : ℕ)
variable (James : ℕ := 4 * Jordan + 7)

theorem oreos_total (h : James = 43) : 43 + Jordan = 52 :=
sorry

end oreos_total_l44_44259


namespace sum_gcf_lcm_l44_44842

theorem sum_gcf_lcm (a b : ℕ) (h_a : a = 8) (h_b : b = 12) :
  Nat.gcd a b + Nat.lcm a b = 28 :=
by
  rw [h_a, h_b]
  -- here you'd typically use known theorems about gcd and lcm for specific numbers
  sorry

end sum_gcf_lcm_l44_44842


namespace arithmetic_geometric_sequence_l44_44847

theorem arithmetic_geometric_sequence
    (a : ℕ → ℕ)
    (b : ℕ → ℕ)
    (h_arith_seq : ∀ n, a (n + 1) - a n = a 1 - a 0) -- Definition of arithmetic sequence
    (h_geom_seq : ∀ n, b (n + 1) / b n = b 1 / b 0) -- Definition of geometric sequence
    (h_a3_a11 : a 3 + a 11 = 8) -- Condition a_3 + a_11 = 8
    (h_b7_a7 : b 7 = a 7) -- Condition b_7 = a_7
    : b 6 * b 8 = 16 := -- Prove that b_6 * b_8 = 16
sorry

end arithmetic_geometric_sequence_l44_44847


namespace find_function_expression_l44_44499

noncomputable def f (x : ℝ) : ℝ := x^2 - 5*x + 7

theorem find_function_expression (x : ℝ) :
  (∀ x : ℝ, f (x + 2) = x^2 - x + 1) →
  f x = x^2 - 5*x + 7 :=
by
  intro h
  sorry

end find_function_expression_l44_44499


namespace terminal_side_equiv_l44_44993

theorem terminal_side_equiv (θ : ℝ) (hθ : θ = 23 * π / 3) : 
  ∃ k : ℤ, θ = 2 * π * k + 5 * π / 3 := by
  sorry

end terminal_side_equiv_l44_44993


namespace remainder_when_7n_divided_by_4_l44_44347

theorem remainder_when_7n_divided_by_4 (n : ℤ) (h : n % 4 = 3) : (7 * n) % 4 = 1 := 
by
  sorry

end remainder_when_7n_divided_by_4_l44_44347


namespace route_time_difference_l44_44159

-- Define the conditions of the problem
def first_route_time : ℕ := 
  let time_uphill := 6
  let time_path := 2 * time_uphill
  let time_finish := (time_uphill + time_path) / 3
  time_uphill + time_path + time_finish

def second_route_time : ℕ := 
  let time_flat_path := 14
  let time_finish := 2 * time_flat_path
  time_flat_path + time_finish

-- Prove the question
theorem route_time_difference : second_route_time - first_route_time = 18 :=
by
  sorry

end route_time_difference_l44_44159


namespace power_sum_inequality_l44_44714

theorem power_sum_inequality (k l m : ℕ) : 
  2 ^ (k + l) + 2 ^ (k + m) + 2 ^ (l + m) ≤ 2 ^ (k + l + m + 1) + 1 := 
by 
  sorry

end power_sum_inequality_l44_44714


namespace lower_bound_third_inequality_l44_44716

theorem lower_bound_third_inequality (x : ℤ) 
  (h1 : 3 < x ∧ x < 10)
  (h2 : 5 < x ∧ x < 18)
  (h3 : 8 > x ∧ x > 0)
  (h4 : x + 1 < 9) :
  x = 7 → ∃ l < 7, ∀ y, l < y ∧ y < 9 → y = x := 
sorry

end lower_bound_third_inequality_l44_44716


namespace total_action_figures_l44_44275

-- Definitions based on conditions
def initial_figures : ℕ := 8
def figures_per_set : ℕ := 5
def added_sets : ℕ := 2
def total_added_figures : ℕ := added_sets * figures_per_set
def total_figures : ℕ := initial_figures + total_added_figures

-- Theorem statement with conditions and expected result
theorem total_action_figures : total_figures = 18 := by
  sorry

end total_action_figures_l44_44275


namespace inequality_positives_l44_44242

theorem inequality_positives (x1 x2 x3 x4 x5 : ℝ) (hx1 : 0 < x1) (hx2 : 0 < x2) (hx3 : 0 < x3) (hx4 : 0 < x4) (hx5 : 0 < x5) : 
  (x1 + x2 + x3 + x4 + x5)^2 ≥ 4 * (x1 * x2 + x3 * x4 + x5 * x1 + x2 * x3 + x4 * x5) :=
sorry

end inequality_positives_l44_44242


namespace perfect_cube_factors_count_l44_44509

-- Define the given prime factorization
def prime_factorization_8820 : Prop :=
  ∃ a b c d : ℕ, 0 ≤ a ∧ a ≤ 2 ∧ 0 ≤ b ∧ b ≤ 2 ∧ 0 ≤ c ∧ c ≤ 1 ∧ 0 ≤ d ∧ d ≤ 2 ∧
  (2 ^ a) * (3 ^ b) * (5 ^ c) * (7 ^ d) = 8820

-- Prove the statement about positive integer factors that are perfect cubes
theorem perfect_cube_factors_count : prime_factorization_8820 → (∃ n : ℕ, n = 1) :=
by
  sorry

end perfect_cube_factors_count_l44_44509


namespace brandon_cards_l44_44124

theorem brandon_cards (b m : ℕ) 
  (h1 : m = b + 8) 
  (h2 : 14 = m / 2) : 
  b = 20 := by
  sorry

end brandon_cards_l44_44124


namespace max_a_plus_2b_plus_c_l44_44501

open Real

theorem max_a_plus_2b_plus_c
  (A : Set ℝ := {x | |x + 1| ≤ 4})
  (T : ℝ := 3)
  (a b c : ℝ)
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_T : a^2 + b^2 + c^2 = T) :
  a + 2 * b + c ≤ 3 * sqrt 2 :=
by
  -- Proof is omitted
  sorry

end max_a_plus_2b_plus_c_l44_44501


namespace xy_value_l44_44396

theorem xy_value (x y : ℝ) (h : x * (x + y) = x^2 + 12) : x * y = 12 := by
  sorry

end xy_value_l44_44396


namespace problem1_problem2_l44_44465

theorem problem1 (n : ℕ) (hn : 0 < n) : (3^(2*n+1) + 2^(n+2)) % 7 = 0 := 
sorry

theorem problem2 (n : ℕ) (hn : 0 < n) : (3^(2*n+2) + 2^(6*n+1)) % 11 = 0 := 
sorry

end problem1_problem2_l44_44465


namespace find_value_of_expression_l44_44655

-- Conditions translated to Lean 4 definitions
variable (a b : ℝ)
axiom h1 : (a^2 * b^3) / 5 = 1000
axiom h2 : a * b = 2

-- The theorem stating what we need to prove
theorem find_value_of_expression :
  (a^3 * b^2) / 3 = 2 / 705 :=
by
  sorry

end find_value_of_expression_l44_44655


namespace second_worker_time_l44_44981

theorem second_worker_time 
  (first_worker_rate : ℝ)
  (combined_rate : ℝ)
  (x : ℝ)
  (h1 : first_worker_rate = 1 / 6)
  (h2 : combined_rate = 1 / 2.4) :
  (1 / 6) + (1 / x) = combined_rate → x = 4 := 
by 
  intros h
  sorry

end second_worker_time_l44_44981


namespace solve_inequality_l44_44130

theorem solve_inequality (x : ℝ) : -4 * x - 8 > 0 → x < -2 := sorry

end solve_inequality_l44_44130


namespace main_inequality_l44_44264

theorem main_inequality (m : ℝ) : (∀ x : ℝ, |2 * x - m| ≤ |3 * x + 6|) ↔ m = -4 := by
  sorry

end main_inequality_l44_44264


namespace no_integer_solutions_l44_44488

theorem no_integer_solutions :
   ¬ ∃ (x y z : ℤ), x^4 + y^4 + z^4 = 2 * x^2 * y^2 + 2 * y^2 * z^2 + 2 * z^2 * x^2 + 24 :=
by
  sorry

end no_integer_solutions_l44_44488


namespace opposite_of_neg_one_fourth_l44_44852

def opposite_of (x : ℝ) : ℝ := -x

theorem opposite_of_neg_one_fourth :
  opposite_of (-1/4) = 1/4 :=
by
  sorry

end opposite_of_neg_one_fourth_l44_44852


namespace max_variance_l44_44934

theorem max_variance (p : ℝ) (h₀ : 0 < p) (h₁ : p < 1) : 
  ∃ q, p * (1 - p) ≤ q ∧ q = 1 / 4 :=
by
  existsi (1 / 4)
  sorry

end max_variance_l44_44934


namespace max_area_of_triangle_l44_44163

theorem max_area_of_triangle (a b c : ℝ) 
  (h1 : ∀ (a b c : ℝ), S = a^2 - (b - c)^2)
  (h2 : b + c = 8) : 
  S ≤ 64 / 17 :=
sorry

end max_area_of_triangle_l44_44163


namespace jordans_score_l44_44971

theorem jordans_score 
  (N : ℕ) 
  (first_19_avg : ℚ) 
  (total_avg : ℚ)
  (total_score_19 : ℚ) 
  (total_score_20 : ℚ) 
  (jordan_score : ℚ) 
  (h1 : N = 19)
  (h2 : first_19_avg = 74)
  (h3 : total_avg = 76)
  (h4 : total_score_19 = N * first_19_avg)
  (h5 : total_score_20 = (N + 1) * total_avg)
  (h6 : jordan_score = total_score_20 - total_score_19) :
  jordan_score = 114 :=
by {
  -- the proof will be filled in, but for now we use sorry
  sorry
}

end jordans_score_l44_44971


namespace min_value_y_l44_44173

theorem min_value_y (x : ℝ) (h : x > 5 / 4) : 
  ∃ y, y = 4*x - 1 + 1 / (4*x - 5) ∧ y ≥ 6 :=
by
  sorry

end min_value_y_l44_44173


namespace min_value_of_n_l44_44861

theorem min_value_of_n (n : ℕ) (k : ℚ) (h1 : k > 0.9999) 
    (h2 : 4 * n * (n - 1) * (1 - k) = 1) : 
    n = 51 :=
sorry

end min_value_of_n_l44_44861


namespace second_horse_revolutions_l44_44210

noncomputable def circumference (radius : ℝ) : ℝ := 2 * Real.pi * radius
noncomputable def distance_traveled (circumference : ℝ) (revolutions : ℕ) : ℝ := circumference * (revolutions : ℝ)
noncomputable def revolutions_needed (distance : ℝ) (circumference : ℝ) : ℕ := ⌊distance / circumference⌋₊

theorem second_horse_revolutions :
  let r1 := 30
  let r2 := 10
  let revolutions1 := 40
  let c1 := circumference r1
  let c2 := circumference r2
  let d1 := distance_traveled c1 revolutions1
  (revolutions_needed d1 c2) = 120 :=
by
  sorry

end second_horse_revolutions_l44_44210


namespace edith_books_total_l44_44932

-- Define the conditions
def novels := 80
def writing_books := novels * 2

-- Theorem statement
theorem edith_books_total : novels + writing_books = 240 :=
by sorry

end edith_books_total_l44_44932


namespace marching_band_total_weight_l44_44144

noncomputable def total_weight : ℕ :=
  let trumpet_weight := 5
  let clarinet_weight := 5
  let trombone_weight := 10
  let tuba_weight := 20
  let drum_weight := 15
  let trumpets := 6
  let clarinets := 9
  let trombones := 8
  let tubas := 3
  let drummers := 2
  (trumpets + clarinets) * trumpet_weight + trombones * trombone_weight + tubas * tuba_weight + drummers * drum_weight

theorem marching_band_total_weight : total_weight = 245 := by
  sorry

end marching_band_total_weight_l44_44144


namespace sales_volume_increase_30_units_every_5_yuan_initial_sales_volume_750_units_daily_sales_volume_at_540_yuan_l44_44375

def price_reduction_table : List (ℕ × ℕ) := 
  [(5, 780), (10, 810), (15, 840), (20, 870), (25, 900), (30, 930), (35, 960)]

theorem sales_volume_increase_30_units_every_5_yuan :
  ∀ reduction volume1 volume2, (reduction + 5, volume1) ∈ price_reduction_table →
  (reduction + 10, volume2) ∈ price_reduction_table → volume2 - volume1 = 30 := sorry

theorem initial_sales_volume_750_units :
  (5, 780) ∈ price_reduction_table → (10, 810) ∈ price_reduction_table →
  (0, 750) ∉ price_reduction_table → 780 - 30 = 750 := sorry

theorem daily_sales_volume_at_540_yuan :
  ∀ P₀ P₁ volume, P₀ = 600 → P₁ = 540 → 
  (5, 780) ∈ price_reduction_table → (10, 810) ∈ price_reduction_table →
  (15, 840) ∈ price_reduction_table → (20, 870) ∈ price_reduction_table →
  (25, 900) ∈ price_reduction_table → (30, 930) ∈ price_reduction_table →
  (35, 960) ∈ price_reduction_table →
  volume = 750 + (P₀ - P₁) / 5 * 30 → volume = 1110 := sorry

end sales_volume_increase_30_units_every_5_yuan_initial_sales_volume_750_units_daily_sales_volume_at_540_yuan_l44_44375


namespace ellipse_standard_equation_parabola_standard_equation_l44_44618

-- Ellipse with major axis length 10 and eccentricity 4/5
theorem ellipse_standard_equation (a c b : ℝ) (h₀ : a = 5) (h₁ : c = 4) (h₂ : b = 3) :
  (x^2 / a^2) + (y^2 / b^2) = 1 := by sorry

-- Parabola with vertex at the origin and directrix y = 2
theorem parabola_standard_equation (p : ℝ) (h₀ : p = 4) :
  x^2 = -8 * y := by sorry

end ellipse_standard_equation_parabola_standard_equation_l44_44618


namespace grandma_vasya_cheapest_option_l44_44889

/-- Constants and definitions for the cost calculations --/
def train_ticket_cost : ℕ := 200
def collected_berries_kg : ℕ := 5
def market_berries_cost_per_kg : ℕ := 150
def sugar_cost_per_kg : ℕ := 54
def jam_made_per_kg_combination : ℕ := 15 / 10  -- representing 1.5 kg (as ratio 15/10)
def ready_made_jam_cost_per_kg : ℕ := 220

/-- Compute the cost per kg of jam for different methods --/
def cost_per_kg_jam_option1 : ℕ := (train_ticket_cost / collected_berries_kg + sugar_cost_per_kg)
def cost_per_kg_jam_option2 : ℕ := market_berries_cost_per_kg + sugar_cost_per_kg
def cost_per_kg_jam_option3 : ℕ := ready_made_jam_cost_per_kg

/-- Numbers converted to per 1.5 kg --/
def cost_for_1_5_kg (cost_per_kg: ℕ) : ℕ := cost_per_kg * (15 / 10)

/-- Theorem stating option 1 is the cheapest --/
theorem grandma_vasya_cheapest_option :
  cost_for_1_5_kg cost_per_kg_jam_option1 ≤ cost_for_1_5_kg cost_per_kg_jam_option2 ∧
  cost_for_1_5_kg cost_per_kg_jam_option1 ≤ cost_for_1_5_kg cost_per_kg_jam_option3 :=
by sorry

end grandma_vasya_cheapest_option_l44_44889


namespace avg_velocity_2_to_2_1_l44_44984

def motion_eq (t : ℝ) : ℝ := 3 + t^2

theorem avg_velocity_2_to_2_1 : 
  (motion_eq 2.1 - motion_eq 2) / (2.1 - 2) = 4.1 :=
by
  sorry

end avg_velocity_2_to_2_1_l44_44984


namespace problem_statement_l44_44568

variable (g : ℝ)

-- Definition of the operation
def my_op (g y : ℝ) : ℝ := g^2 + 2 * y

-- The statement we want to prove
theorem problem_statement : my_op g (my_op g g) = g^4 + 4 * g^3 + 6 * g^2 + 4 * g :=
by
  sorry

end problem_statement_l44_44568


namespace negation_of_forall_geq_l44_44216

theorem negation_of_forall_geq {x : ℝ} : ¬ (∀ x : ℝ, x^2 - x ≥ 0) ↔ ∃ x : ℝ, x^2 - x < 0 :=
by
  sorry

end negation_of_forall_geq_l44_44216


namespace total_birds_on_fence_l44_44248

theorem total_birds_on_fence (initial_birds additional_birds storks : ℕ) 
  (h1 : initial_birds = 6) 
  (h2 : additional_birds = 4) 
  (h3 : storks = 8) :
  initial_birds + additional_birds + storks = 18 :=
by
  sorry

end total_birds_on_fence_l44_44248


namespace smallest_visible_sum_of_3x3x3_cube_is_90_l44_44642

theorem smallest_visible_sum_of_3x3x3_cube_is_90 
: ∀ (dices: Fin 27 → Fin 6 → ℕ),
    (∀ i j k, dices (3*i+j) k = 7 - dices (3*i+j) (5-k)) → 
    (∃ s, s = 90 ∧
    s = (8 * (dices 0 0 + dices 0 1 + dices 0 2)) + 
        (12 * (dices 0 0 + dices 0 1)) +
        (6 * (dices 0 0))) := sorry

end smallest_visible_sum_of_3x3x3_cube_is_90_l44_44642


namespace students_with_dog_and_cat_only_l44_44139

theorem students_with_dog_and_cat_only
  (U : Finset (ℕ)) -- Universe of students
  (D C B : Finset (ℕ)) -- Sets of students with dogs, cats, and birds respectively
  (hU : U.card = 50)
  (hD : D.card = 30)
  (hC : C.card = 35)
  (hB : B.card = 10)
  (hIntersection : (D ∩ C ∩ B).card = 5) :
  ((D ∩ C) \ B).card = 25 := 
sorry

end students_with_dog_and_cat_only_l44_44139


namespace box_upper_surface_area_l44_44878

theorem box_upper_surface_area (L W H : ℕ) 
    (h1 : L * W = 120) 
    (h2 : L * H = 72) 
    (h3 : L * W * H = 720) : 
    L * W = 120 := 
by 
  sorry

end box_upper_surface_area_l44_44878


namespace arccos_cos_solution_l44_44369

theorem arccos_cos_solution (x : ℝ) (h₀ : 0 ≤ x) (h₁ : x ≤ (Real.pi / 2)) (h₂ : Real.arccos (Real.cos x) = 2 * x) : 
    x = 0 :=
by 
  sorry

end arccos_cos_solution_l44_44369


namespace locus_of_points_where_tangents_are_adjoint_lines_l44_44680

theorem locus_of_points_where_tangents_are_adjoint_lines 
  (p : ℝ) (y x : ℝ)
  (h_parabola : y^2 = 2 * p * x) :
  y^2 = - (p / 2) * x :=
sorry

end locus_of_points_where_tangents_are_adjoint_lines_l44_44680


namespace age_difference_l44_44923

variable (S R : ℝ)

theorem age_difference (h1 : S = 38.5) (h2 : S / R = 11 / 9) : S - R = 7 :=
by
  sorry

end age_difference_l44_44923


namespace exists_polynomial_h_l44_44697

variable {R : Type} [CommRing R] [IsDomain R] [CharZero R]

noncomputable def f (x : R) : ℝ := sorry -- define the polynomial f(x) here
noncomputable def g (x : R) : ℝ := sorry -- define the polynomial g(x) here

theorem exists_polynomial_h (m n : ℕ) (a : ℕ → ℝ) (b : ℕ → ℝ) (h_mn : m + n > 0)
  (h_fg_squares : ∀ x : ℝ, (∃ k : ℤ, f x = k^2) ↔ (∃ l : ℤ, g x = l^2)) :
  ∃ h : ℝ → ℝ, ∀ x : ℝ, f x * g x = (h x)^2 :=
sorry

end exists_polynomial_h_l44_44697


namespace solution_of_inequality_is_correct_l44_44192

-- Inequality condition (x-1)/(2x+1) ≤ 0
def inequality (x : ℝ) : Prop := (x - 1) / (2 * x + 1) ≤ 0 

-- Conditions
def condition1 (x : ℝ) : Prop := (x - 1) * (2 * x + 1) ≤ 0
def condition2 (x : ℝ) : Prop := 2 * x + 1 ≠ 0

-- Combined condition
def combined_condition (x : ℝ) : Prop := condition1 x ∧ condition2 x

-- Solution set
def solution_set : Set ℝ := { x | -1/2 < x ∧ x ≤ 1 }

-- Theorem statement
theorem solution_of_inequality_is_correct :
  ∀ x : ℝ, inequality x ↔ combined_condition x ∧ x ∈ solution_set :=
by
  sorry

end solution_of_inequality_is_correct_l44_44192


namespace range_of_a_l44_44696

theorem range_of_a (a : ℝ) 
  (h : ∀ x y, (a * x^2 - 3 * x + 2 = 0) ∧ (a * y^2 - 3 * y + 2 = 0) → x = y) :
  a = 0 ∨ a ≥ 9/8 :=
sorry

end range_of_a_l44_44696


namespace school_problem_proof_l44_44868

noncomputable def solve_school_problem (B G x y z : ℕ) :=
  B + G = 300 ∧
  B * y = x * G ∧
  G = (x * 300) / 100 →
  z = 300 - 3 * x - (300 * x) / (x + y)

theorem school_problem_proof (B G x y z : ℕ) :
  solve_school_problem B G x y z :=
by
  sorry

end school_problem_proof_l44_44868


namespace measure_one_kg_grain_l44_44814

/-- Proving the possibility of measuring exactly 1 kg of grain
    using a balance scale, one 3 kg weight, and three weighings. -/
theorem measure_one_kg_grain :
  ∃ (weighings : ℕ) (balance_scale : ℕ → ℤ) (weight_3kg : ℤ → Prop),
  weighings = 3 ∧
  (∀ w, weight_3kg w ↔ w = 3) ∧
  ∀ n m, balance_scale n = 0 ∧ balance_scale m = 1 → true :=
sorry

end measure_one_kg_grain_l44_44814


namespace base_length_of_isosceles_triangle_l44_44989

theorem base_length_of_isosceles_triangle (a b : ℕ) (h1 : a = 8) (h2 : b + 2 * a = 26) : b = 10 :=
by
  have h3 : 2 * 8 = 16 := by norm_num
  rw [h1] at h2
  rw [h3] at h2
  linarith

end base_length_of_isosceles_triangle_l44_44989


namespace total_apples_and_pears_l44_44832

theorem total_apples_and_pears (x y : ℤ) 
  (h1 : x = 3 * (y / 2 + 1)) 
  (h2 : x = 5 * (y / 4 - 3)) : 
  x + y = 39 :=
sorry

end total_apples_and_pears_l44_44832


namespace sequence_u5_value_l44_44751

theorem sequence_u5_value (u : ℕ → ℝ) 
  (h_rec : ∀ n, u (n + 2) = 2 * u (n + 1) + u n)
  (h_u3 : u 3 = 9) 
  (h_u6 : u 6 = 128) : 
  u 5 = 53 := 
sorry

end sequence_u5_value_l44_44751


namespace value_of_x2_minus_y2_l44_44316

theorem value_of_x2_minus_y2 (x y : ℚ) (h1 : x + y = 9 / 17) (h2 : x - y = 1 / 19) : x^2 - y^2 = 9 / 323 :=
by
  -- the proof would go here
  sorry

end value_of_x2_minus_y2_l44_44316


namespace simplify_sqrt_24_l44_44763

theorem simplify_sqrt_24 : Real.sqrt 24 = 2 * Real.sqrt 6 :=
sorry

end simplify_sqrt_24_l44_44763


namespace edward_earnings_l44_44521

theorem edward_earnings
    (total_lawns : ℕ := 17)
    (forgotten_lawns : ℕ := 9)
    (total_earnings : ℕ := 32) :
    (total_earnings / (total_lawns - forgotten_lawns) = 4) :=
by
  sorry

end edward_earnings_l44_44521


namespace hotel_P_charge_less_than_G_l44_44270

open Real

variable (G R P : ℝ)

-- Given conditions
def charge_R_eq_2G : Prop := R = 2 * G
def charge_P_eq_R_minus_55percent : Prop := P = R - 0.55 * R

-- Goal: Prove the percentage by which P's charge is less than G's charge is 10%
theorem hotel_P_charge_less_than_G : charge_R_eq_2G G R → charge_P_eq_R_minus_55percent R P → P = 0.9 * G := by
  intros h1 h2
  sorry

end hotel_P_charge_less_than_G_l44_44270


namespace consecutive_integers_solution_l44_44737

theorem consecutive_integers_solution :
  ∃ (n : ℕ), n > 0 ∧ n * (n + 1) + 91 = n^2 + (n + 1)^2 ∧ n + 1 = 10 :=
by
  sorry

end consecutive_integers_solution_l44_44737


namespace expression_evaluation_l44_44795

theorem expression_evaluation (a b : ℤ) (h1 : a = 4) (h2 : b = -2) : -a - b^4 + a * b = -28 := 
by 
  sorry

end expression_evaluation_l44_44795


namespace peaches_picked_up_l44_44826

variable (initial_peaches : ℕ) (final_peaches : ℕ)

theorem peaches_picked_up :
  initial_peaches = 13 →
  final_peaches = 55 →
  final_peaches - initial_peaches = 42 :=
by
  intros
  sorry

end peaches_picked_up_l44_44826


namespace find_k4_l44_44569

theorem find_k4
  (a_n : ℕ → ℝ)
  (d : ℝ)
  (h1 : ∀ n, a_n (n + 1) = a_n n + d)
  (h2 : d ≠ 0)
  (h3 : ∃ r : ℝ, a_n 2^2 = a_n 1 * a_n 6)
  (h4 : a_n 1 = a_n k_1)
  (h5 : a_n 2 = a_n k_2)
  (h6 : a_n 6 = a_n k_3)
  (h_k1 : k_1 = 1)
  (h_k2 : k_2 = 2)
  (h_k3 : k_3 = 6) 
  : ∃ k_4 : ℕ, k_4 = 22 := sorry

end find_k4_l44_44569


namespace gcd_of_given_lengths_l44_44265

def gcd_of_lengths_is_eight : Prop :=
  let lengths := [48, 64, 80, 120]
  ∃ d, d = 8 ∧ (∀ n ∈ lengths, d ∣ n)

theorem gcd_of_given_lengths : gcd_of_lengths_is_eight := 
  sorry

end gcd_of_given_lengths_l44_44265


namespace escher_consecutive_probability_l44_44068

open Classical

noncomputable def probability_Escher_consecutive (total_pieces escher_pieces: ℕ): ℚ :=
  if total_pieces < escher_pieces then 0 else (Nat.factorial (total_pieces - escher_pieces) * Nat.factorial escher_pieces) / Nat.factorial (total_pieces - 1)

theorem escher_consecutive_probability :
  probability_Escher_consecutive 12 4 = 1 / 41 :=
by
  sorry

end escher_consecutive_probability_l44_44068


namespace parallelogram_area_l44_44308

theorem parallelogram_area (base height : ℝ) (h_base : base = 14) (h_height : height = 24) :
  base * height = 336 :=
by 
  rw [h_base, h_height]
  sorry

end parallelogram_area_l44_44308


namespace quadratic_function_points_l44_44455

theorem quadratic_function_points:
  (∀ x y, (y = x^2 + x - 1) → ((x = -2 → y = 1) ∧ (x = 0 → y = -1) ∧ (x = 2 → y = 5))) →
  (-1 < 1 ∧ 1 < 5) :=
by
  intro h
  have h1 := h (-2) 1 (by ring)
  have h2 := h 0 (-1) (by ring)
  have h3 := h 2 5 (by ring)
  exact And.intro (by linarith) (by linarith)

end quadratic_function_points_l44_44455


namespace value_of_a_l44_44761

theorem value_of_a (a : ℝ) : (1 / (Real.log 3 / Real.log a) + 1 / (Real.log 4 / Real.log a) + 1 / (Real.log 5 / Real.log a) = 1) → a = 60 :=
by
  sorry

end value_of_a_l44_44761


namespace complex_root_condition_l44_44075

open Complex

theorem complex_root_condition (u v : ℂ) 
    (h1 : 3 * abs (u + 1) * abs (v + 1) ≥ abs (u * v + 5 * u + 5 * v + 1))
    (h2 : abs (u + v) = abs (u * v + 1)) :
    u = 1 ∨ v = 1 :=
sorry

end complex_root_condition_l44_44075


namespace max_a_condition_l44_44118

theorem max_a_condition (a : ℝ) :
  (∀ x : ℝ, x < a → |x| > 2) ∧ (∃ x : ℝ, |x| > 2 ∧ ¬ (x < a)) →
  a ≤ -2 :=
by 
  sorry

end max_a_condition_l44_44118


namespace smallest_Y_l44_44291

theorem smallest_Y (S : ℕ) (h1 : (∀ d ∈ S.digits 10, d = 0 ∨ d = 1)) (h2 : 18 ∣ S) : 
  (∃ (Y : ℕ), Y = S / 18 ∧ ∀ (S' : ℕ), (∀ d ∈ S'.digits 10, d = 0 ∨ d = 1) → 18 ∣ S' → S' / 18 ≥ Y) → 
  Y = 6172839500 :=
sorry

end smallest_Y_l44_44291


namespace part_I_part_II_l44_44998

def S (n : ℕ) : ℕ := 2 ^ n - 1

def a (n : ℕ) : ℕ := 2 ^ (n - 1)

def T (n : ℕ) : ℕ := (n - 1) * 2 ^ n + 1

theorem part_I (S : ℕ → ℕ) (hS : ∀ n, S n = 2^n - 1) :
  ∀ n, ∃ a : ℕ → ℕ, a n = 2^(n-1) :=
by
  sorry

theorem part_II (a : ℕ → ℕ) (ha : ∀ n, a n = 2^(n-1)) :
  ∀ n, ∃ T : ℕ → ℕ, T n = (n - 1) * 2 ^ n + 1 :=
by
  sorry

end part_I_part_II_l44_44998


namespace cos_2pi_minus_alpha_tan_alpha_minus_7pi_l44_44387

open Real

variables (α : ℝ)
variables (h1 : sin (π + α) = -1 / 3) (h2 : π / 2 < α ∧ α < π)

-- Statement for the problem (Ⅰ)
theorem cos_2pi_minus_alpha :
  cos (2 * π - α) = -2 * sqrt 2 / 3 :=
sorry

-- Statement for the problem (Ⅱ)
theorem tan_alpha_minus_7pi :
  tan (α - 7 * π) = -sqrt 2 / 4 :=
sorry

end cos_2pi_minus_alpha_tan_alpha_minus_7pi_l44_44387


namespace total_value_of_treats_l44_44587

def hotel_cost_per_night : ℕ := 4000
def number_of_nights : ℕ := 2
def car_cost : ℕ := 30000
def house_multiplier : ℕ := 4

theorem total_value_of_treats : 
  (number_of_nights * hotel_cost_per_night) + car_cost + (house_multiplier * car_cost) = 158000 := 
by
  sorry

end total_value_of_treats_l44_44587


namespace smallest_n_containing_375_consecutively_l44_44925

theorem smallest_n_containing_375_consecutively :
  ∃ (m n : ℕ), m < n ∧ Nat.gcd m n = 1 ∧ (n = 8) ∧ (∀ (d : ℕ), d < 1000 →
  ∃ (k : ℕ), k * d % n = m ∧ (d / 100) % 10 = 3 ∧ (d / 10) % 10 = 7 ∧ d % 10 = 5) :=
sorry

end smallest_n_containing_375_consecutively_l44_44925


namespace a_left_after_working_days_l44_44745

variable (x : ℕ)  -- x represents the days A worked 

noncomputable def A_work_rate := (1 : ℚ) / 21
noncomputable def B_work_rate := (1 : ℚ) / 28
noncomputable def B_remaining_work := (3 : ℚ) / 4
noncomputable def combined_work_rate := A_work_rate + B_work_rate

theorem a_left_after_working_days 
  (h : combined_work_rate * x + B_remaining_work = 1) : x = 3 :=
by 
  sorry

end a_left_after_working_days_l44_44745


namespace find_number_l44_44943

theorem find_number (x : ℝ) 
  (h : 0.4 * x + (0.3 * 0.2) = 0.26) : x = 0.5 := 
by
  sorry

end find_number_l44_44943


namespace g_of_neg3_l44_44893

def g (x : ℝ) : ℝ := x^2 + 2 * x

theorem g_of_neg3 : g (-3) = 3 :=
by
  sorry

end g_of_neg3_l44_44893


namespace complement_union_l44_44995

open Set Real

noncomputable def S : Set ℝ := {x | x > -2}
noncomputable def T : Set ℝ := {x | x^2 + 3*x - 4 ≤ 0}

theorem complement_union (x : ℝ): x ∈ (univ \ S) ∪ T ↔ x ≤ 1 :=
by
  sorry

end complement_union_l44_44995


namespace equilateral_triangle_of_equal_heights_and_inradius_l44_44452

theorem equilateral_triangle_of_equal_heights_and_inradius 
  {a b c h1 h2 h3 r : ℝ} (h1_eq : h1 = 2 * r * (a * b * c) / a) 
  (h2_eq : h2 = 2 * r * (a * b * c) / b) 
  (h3_eq : h3 = 2 * r * (a * b * c) / c) 
  (sum_heights_eq : h1 + h2 + h3 = 9 * r) : a = b ∧ b = c ∧ c = a :=
by
  sorry

end equilateral_triangle_of_equal_heights_and_inradius_l44_44452


namespace interior_diagonal_length_l44_44050

variables (a b c : ℝ)

-- Conditions
def surface_area_eq : Prop := 2 * (a * b + b * c + c * a) = 22
def edge_length_eq : Prop := 4 * (a + b + c) = 24

-- Question to be proved
theorem interior_diagonal_length :
  surface_area_eq a b c → edge_length_eq a b c → (Real.sqrt (a^2 + b^2 + c^2) = Real.sqrt 14) :=
by
  intros h1 h2
  sorry

end interior_diagonal_length_l44_44050


namespace part1_part2_i_part2_ii_l44_44806

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := k * Real.log x + 1 / Real.exp x

theorem part1 (k : ℝ) (h : ¬ MonotoneOn (f k) (Set.Icc 2 3)) :
  3 / Real.exp 3 < k ∧ k < 2 / Real.exp 2 :=
sorry

variables {x1 x2 : ℝ}
variable (k : ℝ)
variable (h0 : 0 < x1)
variable (h1 : x1 < x2)
variable (h2 : k = x1 / Real.exp x1 ∧ k = x2 / Real.exp x2)

theorem part2_i :
  e / Real.exp x2 - e / Real.exp x1 > -Real.log (x2 / x1) ∧ -Real.log (x2 / x1) > 1 - x2 / x1 :=
sorry

theorem part2_ii : |f k x1 - f k x2| < 1 :=
sorry

end part1_part2_i_part2_ii_l44_44806


namespace trig_problems_l44_44348

variable {A B C : ℝ}
variable {a b c : ℝ}

-- The main theorem statement to prove the magnitude of angle B and find b under given conditions.
theorem trig_problems
  (h₁ : (2 * a - c) * Real.cos B = b * Real.cos C)
  (h₂ : a = Real.sqrt 3)
  (h₃ : c = Real.sqrt 3) :
  Real.cos B = 1 / 2 ∧ b = Real.sqrt 3 := by
sorry

end trig_problems_l44_44348


namespace second_discount_percentage_l44_44891

-- Define the initial conditions.
def listed_price : ℝ := 200
def first_discount_rate : ℝ := 0.20
def final_sale_price : ℝ := 144

-- Calculate the price after the first discount.
def first_discount_amount := first_discount_rate * listed_price
def price_after_first_discount := listed_price - first_discount_amount

-- Define the second discount amount.
def second_discount_amount := price_after_first_discount - final_sale_price

-- Define the theorem to prove the second discount rate.
theorem second_discount_percentage : 
  (second_discount_amount / price_after_first_discount) * 100 = 10 :=
by 
  sorry -- Proof placeholder

end second_discount_percentage_l44_44891


namespace fraction_computation_l44_44876

theorem fraction_computation (p q s u : ℚ)
  (hpq : p / q = 5 / 2)
  (hsu : s / u = 7 / 11) :
  (5 * p * s - 3 * q * u) / (7 * q * u - 4 * p * s) = 109 / 14 := 
by
  sorry

end fraction_computation_l44_44876


namespace conference_min_duration_l44_44482

theorem conference_min_duration : Nat.gcd 9 11 = 1 ∧ Nat.gcd 9 12 = 3 ∧ Nat.gcd 11 12 = 1 ∧ Nat.lcm 9 (Nat.lcm 11 12) = 396 := by
  sorry

end conference_min_duration_l44_44482


namespace probability_correct_l44_44767

noncomputable def probability_parallel_not_coincident : ℚ :=
  let total_points := 6
  let lines := total_points.choose 2
  let total_ways := lines * lines
  let parallel_not_coincident_pairs := 12
  parallel_not_coincident_pairs / total_ways

theorem probability_correct :
  probability_parallel_not_coincident = 4 / 75 := by
  sorry

end probability_correct_l44_44767


namespace max_positive_integers_l44_44208

theorem max_positive_integers (f : Fin 2018 → ℤ) (h : ∀ i : Fin 2018, f i > f (i - 1) + f (i - 2)) : 
  ∃ n: ℕ, n = 2016 ∧ (∀ i : ℕ, i < 2018 → f i > 0) ∧ (∀ i : ℕ, i < 2 → f i < 0) := 
sorry

end max_positive_integers_l44_44208


namespace find_x_if_friendly_l44_44177

theorem find_x_if_friendly (x : ℚ) :
    (∃ m n : ℚ, m + n = 66 ∧ m = 7 * x ∧ n = -18) →
    x = 12 :=
by
  sorry

end find_x_if_friendly_l44_44177


namespace simplifyExpression_l44_44201

theorem simplifyExpression (a b c d : Int) (ha : a = -2) (hb : b = -6) (hc : c = -3) (hd : d = 2) :
  (a + b - c - d = -2 - 6 + 3 - 2) :=
by {
  sorry
}

end simplifyExpression_l44_44201


namespace ryan_chinese_learning_hours_l44_44312

variable (hours_english : ℕ)
variable (days : ℕ)
variable (total_hours : ℕ)

theorem ryan_chinese_learning_hours (h1 : hours_english = 6) 
                                    (h2 : days = 5) 
                                    (h3 : total_hours = 65) : 
                                    total_hours - (hours_english * days) / days = 7 := by
  sorry

end ryan_chinese_learning_hours_l44_44312


namespace range_of_a_l44_44941

-- Define the conditions
def line1 (a x y : ℝ) : Prop := a * x + y - 4 = 0
def line2 (x y : ℝ) : Prop := x - y - 2 = 0
def first_quadrant (x y : ℝ) : Prop := x > 0 ∧ y > 0

-- The main theorem to state
theorem range_of_a (a : ℝ) : 
  (∃ x y : ℝ, line1 a x y ∧ line2 x y ∧ first_quadrant x y) ↔ -1 < a ∧ a < 2 :=
by
  sorry

end range_of_a_l44_44941


namespace alok_age_l44_44372

theorem alok_age (B A C : ℕ) (h1 : B = 6 * A) (h2 : B + 10 = 2 * (C + 10)) (h3 : C = 10) : A = 5 :=
by
  -- proof would go here
  sorry

end alok_age_l44_44372


namespace exists_lattice_midpoint_among_five_points_l44_44234

-- Definition of lattice points
structure LatticePoint where
  x : ℤ
  y : ℤ

open LatticePoint

-- The theorem we want to prove
theorem exists_lattice_midpoint_among_five_points (A B C D E : LatticePoint) :
    ∃ P Q : LatticePoint, P ≠ Q ∧ (P.x + Q.x) % 2 = 0 ∧ (P.y + Q.y) % 2 = 0 := 
  sorry

end exists_lattice_midpoint_among_five_points_l44_44234


namespace coin_flip_sequences_count_l44_44979

theorem coin_flip_sequences_count : 
  let total_flips := 10;
  let heads_fixed := 2;
  (2 : ℕ) ^ (total_flips - heads_fixed) = 256 := 
by 
  sorry

end coin_flip_sequences_count_l44_44979


namespace find_n_l44_44944

theorem find_n (n : ℤ) : 
  50 < n ∧ n < 120 ∧ (n % 8 = 0) ∧ (n % 7 = 3) ∧ (n % 9 = 3) → n = 192 :=
by
  sorry

end find_n_l44_44944


namespace sum_of_products_l44_44145

theorem sum_of_products {a b c : ℝ}
  (h1 : a ^ 2 + b ^ 2 + c ^ 2 = 138)
  (h2 : a + b + c = 20) :
  a * b + b * c + c * a = 131 := 
by
  sorry

end sum_of_products_l44_44145


namespace exponent_tower_divisibility_l44_44811

theorem exponent_tower_divisibility (h1 h2 : ℕ) (Hh1 : h1 ≥ 3) (Hh2 : h2 ≥ 3) : 
  (2 ^ (5 ^ (2 ^ (5 ^ h1))) + 4 ^ (5 ^ (4 ^ (5 ^ h2)))) % 2008 = 0 := by
  sorry

end exponent_tower_divisibility_l44_44811


namespace number_of_triplets_l44_44032

theorem number_of_triplets (N : ℕ) (a b c : ℕ) 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
  (h4 : 2017 ≥ 10 * a) (h5 : 10 * a ≥ 100 * b) (h6 : 100 * b ≥ 1000 * c) : 
  N = 574 := 
sorry

end number_of_triplets_l44_44032


namespace triangle_inequalities_l44_44274

theorem triangle_inequalities (a b c : ℝ) :
  (∀ n : ℕ, a^n + b^n > c^n ∧ a^n + c^n > b^n ∧ b^n + c^n > a^n) →
  (a = b ∧ a > c) ∨ (a = b ∧ b = c) :=
by
  sorry

end triangle_inequalities_l44_44274


namespace greatest_k_for_200k_divides_100_factorial_l44_44592

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem greatest_k_for_200k_divides_100_factorial :
  let x := factorial 100
  let k_max := 12
  ∃ k : ℕ, y = 200 ^ k ∧ y ∣ x ∧ k = k_max :=
sorry

end greatest_k_for_200k_divides_100_factorial_l44_44592


namespace probability_solution_l44_44366

noncomputable def binom_10_7 := Nat.choose 10 7
noncomputable def binom_10_6 := Nat.choose 10 6

theorem probability_solution (p q : ℝ) (h₁ : q = 1 - p) (h₂ : binom_10_7 = 120) (h₃ : binom_10_6 = 210)
  (h₄ : 120 * p ^ 7 * q ^ 3 = 210 * p ^ 6 * q ^ 4) : p = 7 / 11 := 
sorry

end probability_solution_l44_44366


namespace polygon_divided_into_7_triangles_l44_44284

theorem polygon_divided_into_7_triangles (n : ℕ) (h : n - 2 = 7) : n = 9 :=
by
  sorry

end polygon_divided_into_7_triangles_l44_44284


namespace problem_equivalence_l44_44755

variable {x y z w : ℝ}

theorem problem_equivalence (h : (x - y) * (z - w) / ((y - z) * (w - x)) = 3 / 7) :
  (x - z) * (y - w) / ((x - y) * (z - w)) = -4 / 3 := 
sorry

end problem_equivalence_l44_44755


namespace remainder_when_four_times_number_minus_nine_divided_by_eight_l44_44026

theorem remainder_when_four_times_number_minus_nine_divided_by_eight
  (n : ℤ) (h : n % 8 = 3) : (4 * n - 9) % 8 = 3 := by
  sorry

end remainder_when_four_times_number_minus_nine_divided_by_eight_l44_44026


namespace hexagon_interior_angle_Q_l44_44691

theorem hexagon_interior_angle_Q 
  (A B C D E F : ℕ)
  (hA : A = 135) (hB : B = 150) (hC : C = 120) (hD : D = 130) (hE : E = 100)
  (hex_angle_sum : A + B + C + D + E + F = 720) :
  F = 85 :=
by
  rw [hA, hB, hC, hD, hE] at hex_angle_sum
  sorry

end hexagon_interior_angle_Q_l44_44691


namespace probability_exactly_one_second_class_product_l44_44252

open Nat

/-- Proof problem -/
theorem probability_exactly_one_second_class_product :
  let n := 100 -- total products
  let k := 4   -- number of selected products
  let first_class := 90 -- first-class products
  let second_class := 10 -- second-class products
  let C (n k : ℕ) := Nat.choose n k
  (C second_class 1 * C first_class 3 : ℚ) / C n k = 
  (Nat.choose second_class 1 * Nat.choose first_class 3 : ℚ) / Nat.choose n k :=
by
  -- Mathematically equivalent proof
  sorry

end probability_exactly_one_second_class_product_l44_44252


namespace geom_series_correct_sum_l44_44456

-- Define the geometric series sum
noncomputable def geom_series_sum (a r : ℚ) (n : ℕ) :=
  a * (1 - r ^ n) / (1 - r)

-- Given conditions
def a := (1 : ℚ) / 4
def r := (1 : ℚ) / 4
def n := 8

-- Correct answer sum
def correct_sum := (65535 : ℚ) / 196608

-- Proof problem statement
theorem geom_series_correct_sum : geom_series_sum a r n = correct_sum := 
  sorry

end geom_series_correct_sum_l44_44456


namespace weight_of_rod_l44_44750

theorem weight_of_rod (w₆ : ℝ) (h₁ : w₆ = 6.1) : 
  w₆ / 6 * 12 = 12.2 := by
  sorry

end weight_of_rod_l44_44750


namespace y1_gt_y2_l44_44071

theorem y1_gt_y2 (y : ℤ → ℤ) (h_eq : ∀ x, y x = 8 * x - 1)
  (y1 y2 : ℤ) (h_y1 : y 3 = y1) (h_y2 : y 2 = y2) : y1 > y2 :=
by
  -- proof
  sorry

end y1_gt_y2_l44_44071


namespace find_fraction_l44_44036

theorem find_fraction (F : ℝ) (N : ℝ) (X : ℝ)
  (h1 : 0.85 * F = 36)
  (h2 : N = 70.58823529411765)
  (h3 : F = 42.35294117647059) :
  X * N = 42.35294117647059 → X = 0.6 :=
by
  sorry

end find_fraction_l44_44036


namespace Union_A_B_eq_l44_44518

noncomputable def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 4}
noncomputable def B : Set ℝ := {x | -2 < x ∧ x < 2}

theorem Union_A_B_eq : A ∪ B = {x | -2 < x ∧ x ≤ 4} :=
by
  sorry

end Union_A_B_eq_l44_44518


namespace maxAdditionalTiles_l44_44591

-- Board definition
structure Board where
  width : Nat
  height : Nat
  cells : List (Nat × Nat) -- List of cells occupied by tiles

def initialBoard : Board := 
  ⟨10, 9, [(1,1), (1,2), (2,1), (2,2), (3,1), (3,2), (4,1), (4,2), (5,1), (5,2),
            (6,1), (6,2), (7,1), (7,2)]⟩

-- Function to count cells occupied
def occupiedCells (b : Board) : Nat :=
  b.cells.length

-- Function to calculate total cells in a board
def totalCells (b : Board) : Nat :=
  b.width * b.height

-- Function to calculate additional 2x1 tiles that can be placed
def additionalTiles (board : Board) : Nat :=
  (totalCells board - occupiedCells board) / 2

theorem maxAdditionalTiles : additionalTiles initialBoard = 36 := by
  sorry

end maxAdditionalTiles_l44_44591


namespace negation_proof_l44_44638

-- Definitions based on conditions
def Line : Type := sorry  -- Define a type for lines (using sorry for now)
def Plane : Type := sorry  -- Define a type for planes (using sorry for now)

-- Condition definition
def is_perpendicular (l : Line) (α : Plane) : Prop := sorry  -- Define what it means for a plane to be perpendicular to a line (using sorry for now)

-- Given condition
axiom condition : ∀ (l : Line), ∃ (α : Plane), is_perpendicular l α

-- Statement to prove
theorem negation_proof : (∃ (l : Line), ∀ (α : Plane), ¬is_perpendicular l α) :=
sorry

end negation_proof_l44_44638


namespace hcf_of_two_numbers_l44_44491

theorem hcf_of_two_numbers 
  (x y : ℕ) 
  (h1 : x + y = 45)
  (h2 : Nat.lcm x y = 120)
  (h3 : (1/x : ℚ) + (1/y : ℚ) = 11/120) : 
  Nat.gcd x y = 1 := 
sorry

end hcf_of_two_numbers_l44_44491


namespace factorization_divisibility_l44_44637

theorem factorization_divisibility (n : ℤ) : 120 ∣ (n^5 - 5 * n^3 + 4 * n) :=
by
  sorry

end factorization_divisibility_l44_44637


namespace greatest_product_l44_44333

theorem greatest_product (x : ℤ) (h : x + (2020 - x) = 2020) : x * (2020 - x) ≤ 1020100 :=
sorry

end greatest_product_l44_44333


namespace find_n_l44_44869

theorem find_n (q d : ℕ) (hq : q = 25) (hd : d = 10) (h : 30 * q + 20 * d = 5 * q + n * d) : n = 83 := by
  sorry

end find_n_l44_44869


namespace fraction_to_decimal_l44_44351

theorem fraction_to_decimal : (58 / 125 : ℚ) = 0.464 := 
by {
  -- proof omitted
  sorry
}

end fraction_to_decimal_l44_44351


namespace sin_45_eq_sqrt2_div_2_l44_44326

theorem sin_45_eq_sqrt2_div_2 :
  Real.sin (π / 4) = Real.sqrt 2 / 2 := 
by
  sorry

end sin_45_eq_sqrt2_div_2_l44_44326


namespace Cauchy_solution_on_X_l44_44949

section CauchyEquation

variable (f : ℝ → ℝ)

def is_morphism (f : ℝ → ℝ) := ∀ x y : ℝ, f (x + y) = f x + f y

theorem Cauchy_solution_on_X :
  (∀ a b : ℤ, ∀ c d : ℤ, a + b * Real.sqrt 2 = c + d * Real.sqrt 2 → a = c ∧ b = d) →
  is_morphism f →
  ∃ x y : ℝ, ∀ a b : ℤ,
    f (a + b * Real.sqrt 2) = a * x + b * y :=
by
  intros h1 h2
  let x := f 1
  let y := f (Real.sqrt 2)
  exists x, y
  intros a b
  sorry

end CauchyEquation

end Cauchy_solution_on_X_l44_44949


namespace linear_system_solution_l44_44827

theorem linear_system_solution (x y m : ℝ) (h1 : x + 2 * y = m) (h2 : 2 * x - 3 * y = 4) (h3 : x + y = 7) : 
  m = 9 :=
sorry

end linear_system_solution_l44_44827


namespace frac_mul_square_l44_44048

theorem frac_mul_square 
  : (8/9)^2 * (1/3)^2 = 64/729 := 
by 
  sorry

end frac_mul_square_l44_44048


namespace total_workers_is_28_l44_44619

noncomputable def avg_salary_total : ℝ := 750
noncomputable def num_type_a : ℕ := 5
noncomputable def avg_salary_type_a : ℝ := 900
noncomputable def num_type_b : ℕ := 4
noncomputable def avg_salary_type_b : ℝ := 800
noncomputable def avg_salary_type_c : ℝ := 700

theorem total_workers_is_28 :
  ∃ (W : ℕ) (C : ℕ),
  W * avg_salary_total = num_type_a * avg_salary_type_a + num_type_b * avg_salary_type_b + C * avg_salary_type_c ∧
  W = num_type_a + num_type_b + C ∧
  W = 28 :=
by
  sorry

end total_workers_is_28_l44_44619


namespace range_of_g_l44_44896

noncomputable def g (x : ℝ) : ℝ := (Real.cos x)^4 + (Real.sin x)^2

theorem range_of_g : Set.Icc (3 / 4) 1 = Set.range g :=
by
  sorry

end range_of_g_l44_44896


namespace people_per_column_in_second_arrangement_l44_44570
-- Import the necessary libraries

-- Define the conditions as given in the problem
def number_of_people_first_arrangement : ℕ := 30 * 16
def number_of_columns_second_arrangement : ℕ := 8

-- Define the problem statement with proof
theorem people_per_column_in_second_arrangement :
  (number_of_people_first_arrangement / number_of_columns_second_arrangement) = 60 :=
by
  -- Skip the proof here
  sorry

end people_per_column_in_second_arrangement_l44_44570


namespace largest_inscribed_equilateral_triangle_area_l44_44064

noncomputable def inscribed_triangle_area (r : ℝ) : ℝ :=
  let s := r * (3 / Real.sqrt 3)
  let h := (Real.sqrt 3 / 2) * s
  (1 / 2) * s * h

theorem largest_inscribed_equilateral_triangle_area :
  inscribed_triangle_area 10 = 75 * Real.sqrt 3 :=
by
  simp [inscribed_triangle_area]
  sorry

end largest_inscribed_equilateral_triangle_area_l44_44064


namespace func1_max_min_func2_max_min_l44_44519

noncomputable def func1 (x : ℝ) : ℝ := 2 * Real.sin x - 3
noncomputable def func2 (x : ℝ) : ℝ := (7/4 : ℝ) + Real.sin x - (Real.sin x) ^ 2

theorem func1_max_min : (∀ x : ℝ, func1 x ≤ -1) ∧ (∃ x : ℝ, func1 x = -1) ∧ (∀ x : ℝ, func1 x ≥ -5) ∧ (∃ x : ℝ, func1 x = -5)  :=
by
  sorry

theorem func2_max_min : (∀ x : ℝ, func2 x ≤ 2) ∧ (∃ x : ℝ, func2 x = 2) ∧ (∀ x : ℝ, func2 x ≥ 7 / 4) ∧ (∃ x : ℝ, func2 x = 7 / 4) :=
by
  sorry

end func1_max_min_func2_max_min_l44_44519


namespace sum_of_reciprocals_of_shifted_roots_l44_44537

theorem sum_of_reciprocals_of_shifted_roots (p q r : ℝ)
  (h1 : p^3 - 2 * p^2 - p + 3 = 0)
  (h2 : q^3 - 2 * q^2 - q + 3 = 0)
  (h3 : r^3 - 2 * r^2 - r + 3 = 0) :
  (1 / (p - 2)) + (1 / (q - 2)) + (1 / (r - 2)) = -3 :=
by
  sorry

end sum_of_reciprocals_of_shifted_roots_l44_44537


namespace number_of_persons_in_group_l44_44128

theorem number_of_persons_in_group 
    (n : ℕ)
    (h1 : average_age_before - average_age_after = 3)
    (h2 : person_replaced_age = 40)
    (h3 : new_person_age = 10)
    (h4 : total_age_decrease = 3 * n):
  n = 10 := 
sorry

end number_of_persons_in_group_l44_44128


namespace no_integers_with_cube_sum_l44_44935

theorem no_integers_with_cube_sum (a b : ℤ) (h1 : a^3 + b^3 = 4099) (h2 : Prime 4099) : false :=
sorry

end no_integers_with_cube_sum_l44_44935


namespace only_B_forms_triangle_l44_44773

/-- Check if a set of line segments can form a triangle --/
def can_form_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem only_B_forms_triangle :
  ¬ can_form_triangle 2 6 3 ∧
  can_form_triangle 6 7 8 ∧
  ¬ can_form_triangle 1 7 9 ∧
  ¬ can_form_triangle (3 / 2) 4 (5 / 2) :=
by
  sorry

end only_B_forms_triangle_l44_44773


namespace positive_integer_count_l44_44330

theorem positive_integer_count (n : ℕ) :
  ∃ (count : ℕ), (count = 122) ∧ 
  (∀ (k : ℕ), 27 < k ∧ k < 150 → ((150 * k)^40 > k^80 ∧ k^80 > 3^240)) :=
sorry

end positive_integer_count_l44_44330


namespace subset_condition_l44_44918

theorem subset_condition (a : ℝ) :
  (∀ x : ℝ, |2 * x - 1| < 1 → x^2 - 2 * a * x + a^2 - 1 > 0) →
  (a ≤ -1 ∨ a ≥ 2) :=
by
  sorry

end subset_condition_l44_44918


namespace expand_and_simplify_l44_44805

theorem expand_and_simplify (x : ℝ) : (x - 3) * (x + 7) + x = x^2 + 5 * x - 21 := 
by 
  sorry

end expand_and_simplify_l44_44805


namespace kerosene_consumption_reduction_l44_44776

variable (P C : ℝ)

/-- In the new budget, with the price of kerosene oil rising by 25%, 
    we need to prove that consumption must be reduced by 20% to maintain the same expenditure. -/
theorem kerosene_consumption_reduction (h : 1.25 * P * C_new = P * C) : C_new = 0.8 * C := by
  sorry

end kerosene_consumption_reduction_l44_44776


namespace circle_sector_cones_sum_radii_l44_44117

theorem circle_sector_cones_sum_radii :
  let r := 5
  let a₁ := 1
  let a₂ := 2
  let a₃ := 3
  let total_area := π * r * r
  let θ₁ := (a₁ / (a₁ + a₂ + a₃)) * 2 * π
  let θ₂ := (a₂ / (a₁ + a₂ + a₃)) * 2 * π
  let θ₃ := (a₃ / (a₁ + a₂ + a₃)) * 2 * π
  let r₁ := (a₁ / (a₁ + a₂ + a₃)) * r
  let r₂ := (a₂ / (a₁ + a₂ + a₃)) * r
  let r₃ := (a₃ / (a₁ + a₂ + a₃)) * r
  r₁ + r₂ + r₃ = 5 :=
by {
  sorry
}

end circle_sector_cones_sum_radii_l44_44117


namespace only_zero_function_satisfies_inequality_l44_44559

noncomputable def f (x : ℝ) : ℝ := sorry

theorem only_zero_function_satisfies_inequality (α β : ℝ) (hα : α ≠ 0) (hβ : β ≠ 0) :
  (∀ x y : ℝ, 0 < x → 0 < y →
    f x * f y ≥ (y^α / (x^α + x^β)) * (f x)^2 + (x^β / (y^α + y^β)) * (f y)^2) →
  ∀ x : ℝ, 0 < x → f x = 0 :=
sorry

end only_zero_function_satisfies_inequality_l44_44559


namespace solve_for_x_l44_44049

theorem solve_for_x (x : ℝ) (h : (x * (x ^ (5 / 2))) ^ (1 / 4) = 4) : 
  x = 4 ^ (8 / 7) :=
sorry

end solve_for_x_l44_44049


namespace percentage_increase_area_rectangle_l44_44672

theorem percentage_increase_area_rectangle (L W : ℝ) :
  let new_length := 1.20 * L
  let new_width := 1.20 * W
  let original_area := L * W
  let new_area := new_length * new_width
  let percentage_increase := ((new_area - original_area) / original_area) * 100
  percentage_increase = 44 := by
  sorry

end percentage_increase_area_rectangle_l44_44672


namespace ratio_of_new_circumference_to_increase_in_area_l44_44368

theorem ratio_of_new_circumference_to_increase_in_area
  (r k : ℝ) (h_k : 0 < k) :
  (2 * π * (r + k)) / (π * (2 * r * k + k ^ 2)) = 2 * (r + k) / (2 * r * k + k ^ 2) :=
by
  sorry

end ratio_of_new_circumference_to_increase_in_area_l44_44368


namespace ten_thousand_times_ten_thousand_l44_44089

theorem ten_thousand_times_ten_thousand :
  10000 * 10000 = 100000000 :=
by
  sorry

end ten_thousand_times_ten_thousand_l44_44089


namespace hyperbola_foci_l44_44338

/-- Define a hyperbola -/
def hyperbola_eq (x y : ℝ) : Prop := 4 * y^2 - 25 * x^2 = 100

/-- Definition of the foci of the hyperbola -/
def foci_coords (c : ℝ) : Prop := c = Real.sqrt 29

/-- Proof that the foci of the hyperbola 4y^2 - 25x^2 = 100 are (0, -sqrt(29)) and (0, sqrt(29)) -/
theorem hyperbola_foci (x y : ℝ) (c : ℝ) (hx : hyperbola_eq x y) (hc : foci_coords c) :
  (x = 0 ∧ (y = -c ∨ y = c)) :=
sorry

end hyperbola_foci_l44_44338


namespace p_is_necessary_but_not_sufficient_for_q_l44_44758

variable (x : ℝ)

def p : Prop := -1 ≤ x ∧ x ≤ 5
def q : Prop := (x - 5) * (x + 1) < 0

theorem p_is_necessary_but_not_sufficient_for_q : (∀ x, p x → q x) ∧ ¬ (∀ x, q x → p x) := 
sorry

end p_is_necessary_but_not_sufficient_for_q_l44_44758


namespace min_distance_between_parallel_lines_l44_44571

theorem min_distance_between_parallel_lines
  (m c_1 c_2 : ℝ)
  (h_parallel : ∀ x : ℝ, m * x + c_1 = m * x + c_2 → false) :
  ∃ D : ℝ, D = (|c_2 - c_1|) / (Real.sqrt (1 + m^2)) :=
by
  sorry

end min_distance_between_parallel_lines_l44_44571


namespace find_x_l44_44974

theorem find_x (x y : ℝ) (hx : 0 < x) (hy : 0 < y) 
(h_eq : 7 * x^3 + 14 * x^2 * y = x^4 + 2 * x^3 * y) :
  x = 7 :=
by
  sorry

end find_x_l44_44974


namespace real_return_l44_44322

theorem real_return (n i r: ℝ) (h₁ : n = 0.21) (h₂ : i = 0.10) : 
  (1 + r) = (1 + n) / (1 + i) → r = 0.10 :=
by
  intro h₃
  sorry

end real_return_l44_44322


namespace calls_on_friday_l44_44508

noncomputable def total_calls_monday := 35
noncomputable def total_calls_tuesday := 46
noncomputable def total_calls_wednesday := 27
noncomputable def total_calls_thursday := 61
noncomputable def average_calls_per_day := 40
noncomputable def number_of_days := 5
noncomputable def total_calls_week := average_calls_per_day * number_of_days

theorem calls_on_friday : 
  total_calls_week - (total_calls_monday + total_calls_tuesday + total_calls_wednesday + total_calls_thursday) = 31 :=
by
  sorry

end calls_on_friday_l44_44508


namespace probability_correct_l44_44975

/-- 
The set of characters in "HMMT2005".
-/
def characters : List Char := ['H', 'M', 'M', 'T', '2', '0', '0', '5']

/--
The number of ways to choose 4 positions out of 8.
-/
def choose_4_from_8 : ℕ := Nat.choose 8 4

/-- 
The factorial of an integer n.
-/
def factorial : ℕ → ℕ
| 0     => 1
| (n+1) => (n+1) * factorial n

/-- 
The number of ways to arrange "HMMT".
-/
def arrangements_hmmt : ℕ := choose_4_from_8 * (factorial 4 / factorial 2)

/-- 
The number of ways to arrange "2005".
-/
def arrangements_2005 : ℕ := choose_4_from_8 * (factorial 4 / factorial 2)

/-- 
The number of arrangements where both "HMMT" and "2005" appear.
-/
def arrangements_both : ℕ := choose_4_from_8

/-- 
The total number of possible arrangements of "HMMT2005".
-/
def total_arrangements : ℕ := factorial 8 / (factorial 2 * factorial 2)

/-- 
The number of desirable arrangements using inclusion-exclusion.
-/
def desirable_arrangements : ℕ := arrangements_hmmt + arrangements_2005 - arrangements_both

/-- 
The probability of being able to read either "HMMT" or "2005" 
in a random arrangement of "HMMT2005".
-/
def probability : ℚ := (desirable_arrangements : ℚ) / (total_arrangements : ℚ)

/-- 
Prove that the computed probability is equal to 23/144.
-/
theorem probability_correct : probability = 23 / 144 := sorry

end probability_correct_l44_44975


namespace prime_sum_square_mod_3_l44_44082

theorem prime_sum_square_mod_3 (p : Fin 100 → ℕ) (h_prime : ∀ i, Nat.Prime (p i)) (h_distinct : Function.Injective p) :
  let N := (Finset.univ : Finset (Fin 100)).sum (λ i => (p i)^2)
  N % 3 = 1 := by
  sorry

end prime_sum_square_mod_3_l44_44082


namespace range_of_a_l44_44437

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, |x + 3| + |x - 1| ≥ a^2 - 3 * a) ↔ -1 ≤ a ∧ a ≤ 4 :=
by
  sorry

end range_of_a_l44_44437


namespace does_not_round_to_72_56_l44_44681

-- Definitions for the numbers in question
def numA := 72.558
def numB := 72.563
def numC := 72.55999
def numD := 72.564
def numE := 72.555

-- Function to round a number to the nearest hundredth
def round_nearest_hundredth (x : Float) : Float :=
  (Float.round (x * 100) / 100 : Float)

-- Lean statement for the equivalent proof problem
theorem does_not_round_to_72_56 :
  round_nearest_hundredth numA = 72.56 ∧
  round_nearest_hundredth numB = 72.56 ∧
  round_nearest_hundredth numC = 72.56 ∧
  round_nearest_hundredth numD = 72.56 ∧
  round_nearest_hundredth numE ≠ 72.56 :=
by
  sorry

end does_not_round_to_72_56_l44_44681


namespace unit_place_3_pow_34_l44_44667

theorem unit_place_3_pow_34 : Nat.mod (3^34) 10 = 9 :=
by
  sorry

end unit_place_3_pow_34_l44_44667


namespace square_perimeter_l44_44966

-- We define a structure for a square with an area as a condition.
structure Square (s : ℝ) :=
(area_eq : s ^ 2 = 400)

-- The theorem states that given the area of the square is 400 square meters,
-- the perimeter of the square is 80 meters.
theorem square_perimeter (s : ℝ) (sq : Square s) : 4 * s = 80 :=
by
  -- proof omitted
  sorry

end square_perimeter_l44_44966


namespace fractions_not_equal_to_seven_over_five_l44_44459

theorem fractions_not_equal_to_seven_over_five :
  (7 / 5 ≠ 1 + (4 / 20)) ∧ (7 / 5 ≠ 1 + (3 / 15)) ∧ (7 / 5 ≠ 1 + (2 / 6)) :=
by
  sorry

end fractions_not_equal_to_seven_over_five_l44_44459


namespace correct_card_ordering_l44_44565

structure CardOrder where
  left : String
  middle : String
  right : String

def is_right_of (a b : String) : Prop := (a = "club" ∧ (b = "heart" ∨ b = "diamond")) ∨ (a = "8" ∧ b = "4")

def is_left_of (a b : String) : Prop := a = "5" ∧ b = "heart"

def correct_order : CardOrder :=
  { left := "5 of diamonds", middle := "4 of hearts", right := "8 of clubs" }

theorem correct_card_ordering : 
  ∀ order : CardOrder, 
  is_right_of order.right order.middle ∧ is_right_of order.right order.left ∧ is_left_of order.left order.middle 
  → order = correct_order := 
by
  intro order
  intro h
  sorry

end correct_card_ordering_l44_44565


namespace min_seats_to_occupy_l44_44530

theorem min_seats_to_occupy (n : ℕ) (h_n : n = 150) : 
  ∃ (k : ℕ), k = 90 ∧ ∀ m : ℕ, m ≥ k → ∀ i : ℕ, i < n → ∃ j : ℕ, (j < n) ∧ ((j = i + 1) ∨ (j = i - 1)) :=
sorry

end min_seats_to_occupy_l44_44530


namespace total_students_l44_44729

theorem total_students (S : ℕ) (h1 : S / 2 / 2 = 250) : S = 1000 :=
by
  sorry

end total_students_l44_44729


namespace find_number_l44_44948

theorem find_number (x : ℤ) (h : 3 * (x + 8) = 36) : x = 4 :=
by {
  sorry
}

end find_number_l44_44948


namespace max_value_isosceles_triangle_l44_44725

theorem max_value_isosceles_triangle (a b c : ℝ) (h_isosceles : b = c) :
  ∃ B, (∀ (a b c : ℝ), b = c → (b + c) / a ≤ B) ∧ B = 2 :=
by
  sorry

end max_value_isosceles_triangle_l44_44725


namespace cubical_block_weight_l44_44406

-- Given conditions
variables (s : ℝ) (volume_ratio : ℝ) (weight2 : ℝ)
variable (h : volume_ratio = 8)
variable (h_weight : weight2 = 40)

-- The problem statement
theorem cubical_block_weight (weight1 : ℝ) :
  volume_ratio * weight1 = weight2 → weight1 = 5 :=
by
  -- Assume volume ratio as 8, weight of the second cube as 40 pounds
  have h1 : volume_ratio = 8 := h
  have h2 : weight2 = 40 := h_weight
  -- sorry is here to indicate we are skipping the proof
  sorry

end cubical_block_weight_l44_44406


namespace monotonic_intervals_range_of_m_l44_44377

noncomputable def f (x : ℝ) : ℝ := Real.sin (Real.pi / 3 - 2 * x)
noncomputable def g (x : ℝ) (m : ℝ) : ℝ := x^2 - 2 * x + m - 3

theorem monotonic_intervals :
  ∀ k : ℤ,
    (
      (∀ x, -Real.pi / 12 + k * Real.pi ≤ x ∧ x ≤ 5 * Real.pi / 12 + k * Real.pi → ∃ (d : ℝ), f x = d)
      ∧
      (∀ x, 5 * Real.pi / 12 + k * Real.pi ≤ x ∧ x ≤ 11 * Real.pi / 12 + k * Real.pi → ∃ (i : ℝ), f x = i)
    ) := sorry

theorem range_of_m (m : ℝ) :
  (∀ x1 : ℝ, Real.pi / 12 ≤ x1 ∧ x1 ≤ Real.pi / 2 → ∃ x2 : ℝ, -2 ≤ x2 ∧ x2 ≤ m ∧ f x1 = g x2 m) ↔ -1 ≤ m ∧ m ≤ 3 := sorry

end monotonic_intervals_range_of_m_l44_44377


namespace find_ck_l44_44121

-- Definitions based on the conditions
def arithmetic_sequence (d : ℕ) (n : ℕ) : ℕ :=
  1 + (n - 1) * d

def geometric_sequence (r : ℕ) (n : ℕ) : ℕ :=
  r^(n - 1)

def combined_sequence (d r : ℕ) (n : ℕ) : ℕ :=
  arithmetic_sequence d n + geometric_sequence r n

-- Given conditions
variable {d r k : ℕ}
variable (hd : combined_sequence d r (k-1) = 250)
variable (hk : combined_sequence d r (k+1) = 1250)

-- The theorem statement
theorem find_ck : combined_sequence d r k = 502 :=
  sorry

end find_ck_l44_44121


namespace find_c_of_parabola_l44_44123

theorem find_c_of_parabola 
  (a b c : ℝ)
  (h_eq : ∀ y, -3 = a * (y - 1)^2 + b * (y - 1) - 3)
  (h1 : -1 = a * (3 - 1)^2 + b * (3 - 1) - 3) :
  c = -5/2 := by
  sorry

end find_c_of_parabola_l44_44123


namespace find_d_square_plus_5d_l44_44694

theorem find_d_square_plus_5d (a b c d : ℤ) (h₁: a^2 + 2 * a = 65) (h₂: b^2 + 3 * b = 125) (h₃: c^2 + 4 * c = 205) (h₄: d = 5 + 6) :
  d^2 + 5 * d = 176 :=
by
  rw [h₄]
  sorry

end find_d_square_plus_5d_l44_44694


namespace factorization_a_squared_minus_3a_l44_44609

theorem factorization_a_squared_minus_3a (a : ℝ) : a^2 - 3 * a = a * (a - 3) := 
by 
  sorry

end factorization_a_squared_minus_3a_l44_44609


namespace bugs_eat_same_flowers_l44_44626

theorem bugs_eat_same_flowers (num_bugs : ℕ) (total_flowers : ℕ) (flowers_per_bug : ℕ) 
  (h1 : num_bugs = 3) (h2 : total_flowers = 6) (h3 : flowers_per_bug = total_flowers / num_bugs) : 
  flowers_per_bug = 2 :=
by
  sorry

end bugs_eat_same_flowers_l44_44626


namespace Shekar_average_marks_l44_44712

theorem Shekar_average_marks 
  (math_marks : ℕ := 76)
  (science_marks : ℕ := 65)
  (social_studies_marks : ℕ := 82)
  (english_marks : ℕ := 67)
  (biology_marks : ℕ := 95)
  (num_subjects : ℕ := 5) :
  (math_marks + science_marks + social_studies_marks + english_marks + biology_marks) / num_subjects = 77 := 
sorry

end Shekar_average_marks_l44_44712


namespace total_feet_l44_44492

theorem total_feet (heads hens : ℕ) (h1 : heads = 46) (h2 : hens = 22) : 
  ∃ feet : ℕ, feet = 140 := 
by 
  sorry

end total_feet_l44_44492


namespace pencils_left_l44_44839

def initial_pencils := 4527
def given_to_dorothy := 1896
def given_to_samuel := 754
def given_to_alina := 307
def total_given := given_to_dorothy + given_to_samuel + given_to_alina
def remaining_pencils := initial_pencils - total_given

theorem pencils_left : remaining_pencils = 1570 := by
  sorry

end pencils_left_l44_44839


namespace number_of_friends_l44_44929

-- Definitions based on the given problem conditions
def total_candy := 420
def candy_per_friend := 12

-- Proof statement in Lean 4
theorem number_of_friends : total_candy / candy_per_friend = 35 := by
  sorry

end number_of_friends_l44_44929


namespace williams_tips_fraction_l44_44247

theorem williams_tips_fraction
  (A : ℝ) -- average tips for months other than August
  (h : ∀ A, A > 0) -- assuming some positivity constraint for non-degenerate mean
  (h_august : A ≠ 0) -- assuming average can’t be zero
  (august_tips : ℝ := 10 * A)
  (other_months_tips : ℝ := 6 * A)
  (total_tips : ℝ := 16 * A) :
  (august_tips / total_tips) = (5 / 8) := 
sorry

end williams_tips_fraction_l44_44247


namespace vacation_days_l44_44376

-- A plane ticket costs $24 for each person
def plane_ticket_cost : ℕ := 24

-- A hotel stay costs $12 for each person per day
def hotel_stay_cost_per_day : ℕ := 12

-- Total vacation cost is $120
def total_vacation_cost : ℕ := 120

-- The number of days they are planning to stay is 3
def number_of_days : ℕ := 3

-- Prove that given the conditions, the number of days (d) they plan to stay satisfies the total vacation cost
theorem vacation_days (d : ℕ) (plane_ticket_cost hotel_stay_cost_per_day total_vacation_cost : ℕ) 
  (h1 : plane_ticket_cost = 24)
  (h2 : hotel_stay_cost_per_day = 12) 
  (h3 : total_vacation_cost = 120) 
  (h4 : 2 * plane_ticket_cost + (2 * hotel_stay_cost_per_day) * d = total_vacation_cost)
  : d = 3 := sorry

end vacation_days_l44_44376


namespace Priya_driving_speed_l44_44968

/-- Priya's driving speed calculation -/
theorem Priya_driving_speed
  (time_XZ : ℝ) (rate_back : ℝ) (time_ZY : ℝ)
  (midway_condition : time_XZ = 5)
  (speed_back_condition : rate_back = 60)
  (time_back_condition : time_ZY = 2.0833333333333335) :
  ∃ speed_XZ : ℝ, speed_XZ = 50 :=
by
  have distance_ZY : ℝ := rate_back * time_ZY
  have distance_XZ : ℝ := 2 * distance_ZY
  have speed_XZ : ℝ := distance_XZ / time_XZ
  existsi speed_XZ
  sorry

end Priya_driving_speed_l44_44968


namespace homes_termite_ridden_but_not_collapsing_fraction_l44_44272

variable (H : Type) -- Representing Homes on Gotham Street

def termite_ridden_fraction : ℚ := 1 / 3
def collapsing_fraction_given_termite_ridden : ℚ := 7 / 10

theorem homes_termite_ridden_but_not_collapsing_fraction :
  (termite_ridden_fraction * (1 - collapsing_fraction_given_termite_ridden)) = 1 / 10 :=
by
  sorry

end homes_termite_ridden_but_not_collapsing_fraction_l44_44272


namespace graph_n_plus_k_odd_l44_44162

-- Definitions and assumptions
variable {V : Type} [Fintype V] [DecidableEq V] (G : SimpleGraph V)
variable (n k : ℕ)
variable (hG : Fintype.card V = n)
variable (hCond : ∀ (S : Finset V), S.card = k → (G.commonNeighborsFinset S).card % 2 = 1)

-- Goal
theorem graph_n_plus_k_odd :
  (n + k) % 2 = 1 :=
sorry

end graph_n_plus_k_odd_l44_44162


namespace find_x_l44_44718

theorem find_x (x : ℝ) (y : ℝ) : (∀ y, 10 * x * y - 15 * y + 2 * x - 3 = 0) → x = 3 / 2 := by
  intros h
  -- At this point, you would include the necessary proof steps, but for now we skip it.
  sorry

end find_x_l44_44718


namespace find_f_7_l44_44109

noncomputable def f (x : ℝ) : ℝ := 2*x^4 - 17*x^3 + 26*x^2 - 24*x - 60

theorem find_f_7 : f 7 = 17 :=
  by
  -- The proof steps will go here
  sorry

end find_f_7_l44_44109


namespace solve_problem_l44_44552

theorem solve_problem :
  ∃ a b c d e f : ℤ,
  (208208 = 8^5 * a + 8^4 * b + 8^3 * c + 8^2 * d + 8 * e + f) ∧
  (0 ≤ a ∧ a ≤ 7) ∧ (0 ≤ b ∧ b ≤ 7) ∧ (0 ≤ c ∧ c ≤ 7) ∧
  (0 ≤ d ∧ d ≤ 7) ∧ (0 ≤ e ∧ e ≤ 7) ∧ (0 ≤ f ∧ f ≤ 7) ∧
  (a * b * c + d * e * f = 72) :=
by
  sorry

end solve_problem_l44_44552


namespace find_sum_of_a_and_c_l44_44982

variable (a b c d : ℝ)

theorem find_sum_of_a_and_c (h1 : a * b + b * c + c * d + d * a = 48) (h2 : b + d = 6) :
  a + c = 8 := by sorry

end find_sum_of_a_and_c_l44_44982


namespace chord_length_perpendicular_bisector_of_radius_l44_44038

theorem chord_length_perpendicular_bisector_of_radius (r : ℝ) (h : r = 15) :
  ∃ (CD : ℝ), CD = 15 * Real.sqrt 3 :=
by
  sorry

end chord_length_perpendicular_bisector_of_radius_l44_44038


namespace inequality_solution_l44_44988

theorem inequality_solution (x : ℝ) : (4 + 2 * x > -6) → (x > -5) :=
by sorry

end inequality_solution_l44_44988


namespace min_value_of_f_l44_44246

-- Define the function f
def f (a b c x y z : ℤ) : ℤ := a * x + b * y + c * z

-- Define the gcd function for three integers
def gcd3 (a b c : ℕ) : ℕ := Nat.gcd (Nat.gcd a b) c

-- Define the main theorem to prove
theorem min_value_of_f (a b c : ℕ) (a_pos : a > 0) (b_pos : b > 0) (c_pos : c > 0) :
  ∃ (x y z : ℤ), f a b c x y z = gcd3 a b c := 
by
  sorry

end min_value_of_f_l44_44246


namespace g_at_3_l44_44824

def g (x : ℝ) : ℝ := x^3 - 2 * x^2 + x

theorem g_at_3 : g 3 = 12 := by
  sorry

end g_at_3_l44_44824


namespace find_x_y_l44_44901

theorem find_x_y (x y : ℝ) (h1 : x + Real.cos y = 2023) (h2 : x + 2023 * Real.sin y = 2022) (h3 : 0 ≤ y ∧ y ≤ Real.pi / 2) :
  x + y = 2022 :=
sorry

end find_x_y_l44_44901


namespace application_methods_count_l44_44116

theorem application_methods_count (n_graduates m_universities : ℕ) (h_graduates : n_graduates = 5) (h_universities : m_universities = 3) :
  (m_universities ^ n_graduates) = 243 :=
by
  rw [h_graduates, h_universities]
  show 3 ^ 5 = 243
  sorry

end application_methods_count_l44_44116


namespace area_of_fourth_rectangle_l44_44771

theorem area_of_fourth_rectangle (a b c d : ℕ) (x y z w : ℕ)
  (h1 : a = x * y)
  (h2 : b = x * w)
  (h3 : c = z * w)
  (h4 : d = y * w)
  (h5 : (x + z) * (y + w) = a + b + c + d) : d = 15 :=
sorry

end area_of_fourth_rectangle_l44_44771


namespace eval_expression_correct_l44_44593

noncomputable def eval_expression : ℝ := (-64)^(4/3)

theorem eval_expression_correct : eval_expression = 256 := by
  sorry

end eval_expression_correct_l44_44593


namespace youngest_brother_age_l44_44041

theorem youngest_brother_age 
  (x : ℤ) 
  (h1 : ∃ (a b c : ℤ), a = x ∧ b = x + 1 ∧ c = x + 2 ∧ a + b + c = 96) : 
  x = 31 :=
by sorry

end youngest_brother_age_l44_44041


namespace find_solution_l44_44873

def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

def sum_first_p_squares (p : ℕ) : ℕ := p * (p + 1) * (2 * p + 1) / 6

theorem find_solution : ∃ (n p : ℕ), p.Prime ∧ sum_first_n n = 3 * sum_first_p_squares p ∧ (n, p) = (5, 2) := 
by
  sorry

end find_solution_l44_44873


namespace three_digit_sum_of_factorials_l44_44245

theorem three_digit_sum_of_factorials : 
  ∃ (n : ℕ), (100 ≤ n ∧ n < 1000) ∧ (n = 145) ∧ 
  (∃ (d1 d2 d3 : ℕ), n = d1 * 100 + d2 * 10 + d3 ∧ 
    1 ≤ d1 ∧ d1 < 10 ∧ 1 ≤ d2 ∧ d2 < 10 ∧ 1 ≤ d3 ∧ d3 < 10 ∧ 
    (d1 * d1.factorial + d2 * d2.factorial + d3 * d3.factorial = n)) :=
  by
  sorry

end three_digit_sum_of_factorials_l44_44245


namespace evaluate_expression_l44_44502

theorem evaluate_expression : (↑7 ^ (1/4) / ↑7 ^ (1/6)) = (↑7 ^ (1/12)) :=
by
  sorry

end evaluate_expression_l44_44502


namespace hired_is_B_l44_44845

-- Define the individuals
inductive Person : Type
| A : Person
| B : Person
| C : Person
| D : Person

open Person

-- Define the statements made by each person
def statement (p : Person) (hired : Person) : Prop :=
  match p with
  | A => hired = C
  | B => hired ≠ B
  | C => hired = D
  | D => hired ≠ D

-- The main theorem is to prove B is hired given the conditions
theorem hired_is_B :
  (∃! p : Person, ∃ t : Person → Prop,
    (∀ h : Person, t h ↔ h = p) ∧
    (∃ q : Person, statement q q ∧ ∀ r : Person, r ≠ q → ¬statement r q) ∧
    t B) :=
by
  sorry

end hired_is_B_l44_44845


namespace entire_show_length_l44_44577

def first_segment (S T : ℕ) : ℕ := 2 * (S + T)
def second_segment (T : ℕ) : ℕ := 2 * T
def third_segment : ℕ := 10

theorem entire_show_length : 
  first_segment (second_segment third_segment) third_segment + 
  second_segment third_segment + 
  third_segment = 90 :=
by
  sorry

end entire_show_length_l44_44577


namespace intersection_eq_l44_44542

def A : Set ℤ := {-2, -1, 3, 4}
def B : Set ℤ := {-1, 2, 3}

theorem intersection_eq : A ∩ B = {-1, 3} := 
by
  sorry

end intersection_eq_l44_44542


namespace find_a_l44_44374

theorem find_a (a : ℚ) : (∃ b : ℚ, 4 * (x : ℚ)^2 + 14 * x + a = (2 * x + b)^2) → a = 49 / 4 :=
by
  sorry

end find_a_l44_44374


namespace problem_statement_l44_44715

variable {f : ℝ → ℝ}

-- Assume the conditions provided in the problem statement.
def continuous_on_ℝ (f : ℝ → ℝ) : Prop := Continuous f
def condition_x_f_prime (f : ℝ → ℝ) (h : ℝ → ℝ) : Prop := ∀ x : ℝ, x * h x < 0

-- The main theorem statement based on the conditions and the correct answer.
theorem problem_statement (hf : continuous_on_ℝ f) (hf' : ∀ x : ℝ, x * (deriv f x) < 0) :
  f (-1) + f 1 < 2 * f 0 :=
sorry

end problem_statement_l44_44715


namespace cost_of_adult_ticket_is_15_l44_44243

variable (A : ℕ) -- Cost of an adult ticket
variable (total_tickets : ℕ) (cost_child_ticket : ℕ) (total_revenue : ℕ)
variable (adult_tickets_sold : ℕ)

theorem cost_of_adult_ticket_is_15
  (h1 : total_tickets = 522)
  (h2 : cost_child_ticket = 8)
  (h3 : total_revenue = 5086)
  (h4 : adult_tickets_sold = 130) 
  (h5 : (total_tickets - adult_tickets_sold) * cost_child_ticket + adult_tickets_sold * A = total_revenue) :
  A = 15 :=
by
  sorry

end cost_of_adult_ticket_is_15_l44_44243


namespace gcd_of_1230_and_990_l44_44991

theorem gcd_of_1230_and_990 : Nat.gcd 1230 990 = 30 :=
by
  sorry

end gcd_of_1230_and_990_l44_44991


namespace angle_in_fourth_quadrant_l44_44654

variable (α : ℝ)

def is_in_first_quadrant (α : ℝ) : Prop := 0 < α ∧ α < 90

def is_in_fourth_quadrant (θ : ℝ) : Prop := 270 < θ ∧ θ < 360

theorem angle_in_fourth_quadrant (h : is_in_first_quadrant α) : is_in_fourth_quadrant (360 - α) := sorry

end angle_in_fourth_quadrant_l44_44654


namespace compare_negative_fractions_l44_44596

theorem compare_negative_fractions :
  (-5 : ℝ) / 6 < (-4 : ℝ) / 5 :=
sorry

end compare_negative_fractions_l44_44596


namespace sum_of_slopes_range_l44_44747

theorem sum_of_slopes_range (p b : ℝ) (hpb : 2 * p > b) (hp : p > 0) 
  (K1 K2 : ℝ) (A B : ℝ × ℝ) (hA : A.2^2 = 2 * p * A.1) (hB : B.2^2 = 2 * p * B.1)
  (hl1 : A.2 = A.1 + b) (hl2 : B.2 = B.1 + b) 
  (hA_pos : A.2 > 0) (hB_pos : B.2 > 0) :
  4 < K1 + K2 :=
sorry

end sum_of_slopes_range_l44_44747


namespace total_area_of_removed_triangles_l44_44833

theorem total_area_of_removed_triangles (x r s : ℝ) (h1 : (x - r)^2 + (x - s)^2 = 15^2) :
  4 * (1/2 * r * s) = 112.5 :=
by
  sorry

end total_area_of_removed_triangles_l44_44833


namespace find_stiffnesses_l44_44707

def stiffnesses (m g x1 x2 k1 k2 : ℝ) : Prop :=
  (m = 3) ∧ (g = 10) ∧ (x1 = 0.4) ∧ (x2 = 0.075) ∧
  (k1 * k2 / (k1 + k2) * x1 = m * g) ∧
  ((k1 + k2) * x2 = m * g)

theorem find_stiffnesses (k1 k2 : ℝ) :
  stiffnesses 3 10 0.4 0.075 k1 k2 → 
  k1 = 300 ∧ k2 = 100 := 
sorry

end find_stiffnesses_l44_44707


namespace boat_travel_distance_downstream_l44_44522

-- Definitions of the given conditions
def boatSpeedStillWater : ℕ := 10 -- km/hr
def streamSpeed : ℕ := 8 -- km/hr
def timeDownstream : ℕ := 3 -- hours

-- Effective speed downstream
def effectiveSpeedDownstream : ℕ := boatSpeedStillWater + streamSpeed

-- Goal: Distance traveled downstream equals 54 km
theorem boat_travel_distance_downstream :
  effectiveSpeedDownstream * timeDownstream = 54 := 
by
  -- Since only the statement is needed, we use sorry to indicate the proof is skipped
  sorry

end boat_travel_distance_downstream_l44_44522


namespace arithmetic_sequence_equality_l44_44600

theorem arithmetic_sequence_equality {a b c : ℝ} (h₁ : a ≠ 0) (h₂ : b ≠ 0) (h₃ : c ≠ 0) (a20 : a ≠ c) (a2012 : b ≠ c) 
(h₄ : ∀ (i : ℕ), ∃ d : ℝ, a_n = a + i * d) : 
  1992 * a * c - 1811 * b * c - 181 * a * b = 0 := 
by {
  sorry
}

end arithmetic_sequence_equality_l44_44600


namespace algebraic_expression_value_l44_44705

theorem algebraic_expression_value (x : ℝ) (hx : x = 2 * Real.cos 45 + 1) :
  (1 / (x - 1) - (x - 3) / (x ^ 2 - 2 * x + 1)) / (2 / (x - 1)) = Real.sqrt 2 / 2 :=
by
  sorry

end algebraic_expression_value_l44_44705


namespace state_a_selection_percentage_l44_44398

-- Definitions based on the conditions
variables {P : ℕ} -- percentage of candidates selected in State A

theorem state_a_selection_percentage 
  (candidates : ℕ) 
  (state_b_percentage : ℕ) 
  (extra_selected_in_b : ℕ) 
  (total_selected_in_b : ℕ) 
  (total_selected_in_a : ℕ)
  (appeared_in_each_state : ℕ) 
  (H1 : appeared_in_each_state = 8200)
  (H2 : state_b_percentage = 7)
  (H3 : extra_selected_in_b = 82)
  (H4 : total_selected_in_b = (state_b_percentage * appeared_in_each_state) / 100)
  (H5 : total_selected_in_a = total_selected_in_b - extra_selected_in_b)
  (H6 : total_selected_in_a = (P * appeared_in_each_state) / 100)
  : P = 6 :=
by {
  sorry
}

end state_a_selection_percentage_l44_44398


namespace total_ways_to_choose_gifts_l44_44217

/-- The 6 pairs of zodiac signs -/
def zodiac_pairs : Set (Set String) :=
  {{"Rat", "Ox"}, {"Tiger", "Rabbit"}, {"Dragon", "Snake"}, {"Horse", "Sheep"}, {"Monkey", "Rooster"}, {"Dog", "Pig"}}

/-- The preferences of Students A, B, and C -/
def A_likes : Set String := {"Ox", "Horse"}
def B_likes : Set String := {"Ox", "Dog", "Sheep"}
def C_likes : Set String := {"Rat", "Ox", "Tiger", "Rabbit", "Dragon", "Snake", "Horse", "Sheep", "Monkey", "Rooster", "Dog", "Pig"}

theorem total_ways_to_choose_gifts : 
  True := 
by
  -- We prove that the number of ways is 16
  sorry

end total_ways_to_choose_gifts_l44_44217


namespace polynomial_product_roots_l44_44628

theorem polynomial_product_roots (a b c : ℝ) : 
  (∀ x, (x - (Real.sin (Real.pi / 6))) * (x - (Real.sin (Real.pi / 3))) * (x - (Real.sin (5 * Real.pi / 6))) = x^3 + a * x^2 + b * x + c) → 
  a * b * c = Real.sqrt 3 / 2 :=
by
  sorry

end polynomial_product_roots_l44_44628


namespace polynomial_roots_r_eq_18_l44_44223

theorem polynomial_roots_r_eq_18
  (a b c : ℂ) 
  (h_roots : Polynomial.roots (Polynomial.C (0 : ℂ) * Polynomial.X^3 + Polynomial.C (5 : ℂ) * Polynomial.X^2 + Polynomial.C (2 : ℂ) * Polynomial.X + Polynomial.C (-8 : ℂ)) = {a, b, c}) 
  (h_ab_roots : Polynomial.roots (Polynomial.C (0 : ℂ) * Polynomial.X^3 + Polynomial.C p * Polynomial.X^2 + Polynomial.C q * Polynomial.X + Polynomial.C r) = {2 * a + b, 2 * b + c, 2 * c + a}) :
  r = 18 := sorry

end polynomial_roots_r_eq_18_l44_44223


namespace mila_social_media_hours_l44_44503

/-- 
Mila spends 6 hours on his phone every day. 
Half of this time is spent on social media. 
Prove that Mila spends 21 hours on social media in a week.
-/
theorem mila_social_media_hours 
  (hours_per_day : ℕ)
  (phone_time_per_day : hours_per_day = 6)
  (daily_social_media_fraction : ℕ)
  (fractional_time : daily_social_media_fraction = hours_per_day / 2)
  (days_per_week : ℕ)
  (days_in_week : days_per_week = 7) :
  (daily_social_media_fraction * days_per_week = 21) :=
sorry

end mila_social_media_hours_l44_44503


namespace tangerine_count_l44_44547

def initial_tangerines : ℕ := 10
def added_tangerines : ℕ := 6

theorem tangerine_count : initial_tangerines + added_tangerines = 16 :=
by
  sorry

end tangerine_count_l44_44547


namespace cylinder_surface_area_l44_44791

theorem cylinder_surface_area
  (l : ℝ) (r : ℝ) (unfolded_square_side : ℝ) (base_circumference : ℝ)
  (hl : unfolded_square_side = 2 * π)
  (hl_gen : l = 2 * π)
  (hc : base_circumference = 2 * π)
  (hr : r = 1) :
  2 * π * r * (r + l) = 2 * π + 4 * π^2 :=
by
  sorry

end cylinder_surface_area_l44_44791


namespace permutations_divisibility_l44_44788

theorem permutations_divisibility (n : ℕ) (a b : Fin n → ℕ) 
  (h_n : 2 < n)
  (h_a_perm : ∀ i, ∃ j, a j = i)
  (h_b_perm : ∀ i, ∃ j, b j = i) :
  ∃ (i j : Fin n), i ≠ j ∧ n ∣ (a i * b i - a j * b j) :=
by sorry

end permutations_divisibility_l44_44788


namespace find_product_in_geometric_sequence_l44_44300

def geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ n, a (n + 1) / a n = a 1 / a 0

theorem find_product_in_geometric_sequence (a : ℕ → ℝ) 
  (h_geom : geometric_sequence a)
  (h_prod : a 1 * a 7 * a 13 = 8) : 
  a 3 * a 11 = 4 :=
by sorry

end find_product_in_geometric_sequence_l44_44300


namespace simplify_expression_l44_44087

theorem simplify_expression : 0.72 * 0.43 + 0.12 * 0.34 = 0.3504 := by
  sorry

end simplify_expression_l44_44087


namespace ratio_of_silver_to_gold_l44_44783

-- Definitions for balloon counts
def gold_balloons : Nat := 141
def black_balloons : Nat := 150
def total_balloons : Nat := 573

-- Define the number of silver balloons S
noncomputable def silver_balloons : Nat :=
  total_balloons - gold_balloons - black_balloons

-- The goal is to prove the ratio of silver to gold balloons is 2
theorem ratio_of_silver_to_gold :
  (silver_balloons / gold_balloons) = 2 := by
  sorry

end ratio_of_silver_to_gold_l44_44783


namespace suitable_for_lottery_method_B_l44_44544

def total_items_A : Nat := 3000
def samples_A : Nat := 600

def total_items_B (n: Nat) : Nat := 2 * 15
def samples_B : Nat := 6

def total_items_C : Nat := 2 * 15
def samples_C : Nat := 6

def total_items_D : Nat := 3000
def samples_D : Nat := 10

def is_lottery_suitable (total_items : Nat) (samples : Nat) (different_factories : Bool) : Bool :=
  total_items <= 30 && samples <= total_items && !different_factories

theorem suitable_for_lottery_method_B : 
  is_lottery_suitable (total_items_B 2) samples_B false = true :=
  sorry

end suitable_for_lottery_method_B_l44_44544


namespace complex_third_quadrant_l44_44429

-- Define the imaginary unit i.
def i : ℂ := Complex.I 

-- Define the complex number z = i * (1 + i).
def z : ℂ := i * (1 + i)

-- Prove that z lies in the third quadrant.
theorem complex_third_quadrant : z.re < 0 ∧ z.im < 0 := 
by
  sorry

end complex_third_quadrant_l44_44429


namespace problem_statement_l44_44233

theorem problem_statement (g : ℝ → ℝ) (m k : ℝ) (h₀ : ∀ x, g x = 5 * x - 3)
  (h₁ : 0 < k) (h₂ : 0 < m)
  (h₃ : ∀ x, |g x - 2| < k ↔ |x - 1| < m) : m ≤ k / 5 :=
sorry

end problem_statement_l44_44233


namespace movie_store_additional_movie_needed_l44_44709

theorem movie_store_additional_movie_needed (movies shelves : ℕ) (h_movies : movies = 999) (h_shelves : shelves = 5) : 
  (shelves - (movies % shelves)) % shelves = 1 :=
by
  sorry

end movie_store_additional_movie_needed_l44_44709


namespace square_difference_division_l44_44904

theorem square_difference_division (a b : ℕ) (h₁ : a = 121) (h₂ : b = 112) :
  (a^2 - b^2) / 9 = 233 :=
by
  sorry

end square_difference_division_l44_44904


namespace gold_coins_count_l44_44495

theorem gold_coins_count (n c : ℕ) (h1 : n = 8 * (c - 3))
                                     (h2 : n = 5 * c + 4)
                                     (h3 : c ≥ 10) : n = 54 :=
by
  sorry

end gold_coins_count_l44_44495


namespace solution_set_of_f_prime_gt_zero_l44_44335

noncomputable def f (x : ℝ) : ℝ := x^2 - 2*x - 4 * Real.log x

theorem solution_set_of_f_prime_gt_zero :
  {x : ℝ | 0 < x ∧ 2*x - 2 - (4 / x) > 0} = {x : ℝ | 2 < x} :=
by
  sorry

end solution_set_of_f_prime_gt_zero_l44_44335


namespace perpendicular_lines_l44_44602

theorem perpendicular_lines (a : ℝ) : 
  (∀ (x y : ℝ), (1 - 2 * a) * x - 2 * y + 3 = 0 → 3 * x + y + 2 * a = 0) → 
  a = 1 / 6 :=
by
  sorry

end perpendicular_lines_l44_44602


namespace gas_cost_per_gallon_is_4_l44_44884

noncomputable def cost_per_gallon (miles_per_gallon : ℝ) (total_miles : ℝ) (total_cost : ℝ) : ℝ :=
  total_cost / (total_miles / miles_per_gallon)

theorem gas_cost_per_gallon_is_4 :
  cost_per_gallon 32 432 54 = 4 := by
  sorry

end gas_cost_per_gallon_is_4_l44_44884


namespace molecular_weight_of_9_moles_l44_44469

theorem molecular_weight_of_9_moles (molecular_weight : ℕ) (moles : ℕ) (h₁ : molecular_weight = 1098) (h₂ : moles = 9) :
  molecular_weight * moles = 9882 :=
by {
  sorry
}

end molecular_weight_of_9_moles_l44_44469


namespace remainder_of_exponentiation_l44_44886

theorem remainder_of_exponentiation (n : ℕ) : (3 ^ (2 * n) + 8) % 8 = 1 := 
by sorry

end remainder_of_exponentiation_l44_44886


namespace volume_of_pyramid_in_cube_l44_44276

structure Cube :=
(side_length : ℝ)

noncomputable def base_triangle_area (side_length : ℝ) : ℝ :=
(1/2) * side_length * side_length

noncomputable def pyramid_volume (triangle_area : ℝ) (height : ℝ) : ℝ :=
(1/3) * triangle_area * height

theorem volume_of_pyramid_in_cube (c : Cube) (h : c.side_length = 2) : 
  pyramid_volume (base_triangle_area c.side_length) c.side_length = 4/3 :=
by {
  sorry
}

end volume_of_pyramid_in_cube_l44_44276


namespace prove_M_squared_l44_44970

noncomputable def M (x : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![ -1, 2], ![ (5/2:ℝ), x]]

def eigenvalue_condition (x : ℝ) : Prop :=
  let A := M x
  ∃ v : ℝ, (A - (-2) • (1 : Matrix (Fin 2) (Fin 2) ℝ)).det = 0

theorem prove_M_squared (x : ℝ) (h : eigenvalue_condition x) :
  (M x * M x) = ![![ 6, -9], ![ - (45/4:ℝ), 69/4]] :=
sorry

end prove_M_squared_l44_44970


namespace expression_not_computable_by_square_difference_l44_44500

theorem expression_not_computable_by_square_difference (x : ℝ) :
  ¬ ((x + 1) * (1 + x) = (x + 1) * (x - 1) ∨
     (x + 1) * (1 + x) = (-x + 1) * (-x - 1) ∨
     (x + 1) * (1 + x) = (x + 1) * (-x + 1)) :=
by
  sorry

end expression_not_computable_by_square_difference_l44_44500


namespace initial_number_of_people_l44_44191

theorem initial_number_of_people (X : ℕ) (h : ((X - 10) + 15 = 17)) : X = 12 :=
by
  sorry

end initial_number_of_people_l44_44191


namespace seq_solution_l44_44447

-- Definitions: Define the sequence {a_n} according to the given conditions
def seq (a : ℕ → ℤ) : Prop :=
  a 1 = 0 ∧ ∀ n ≥ 2, a n - 2 * a (n - 1) = n ^ 2 - 3

-- Main statement: Prove that for all n, the sequence satisfies the derived formula
theorem seq_solution (a : ℕ → ℤ) (h : seq a) : ∀ n, a n = 2 ^ (n + 2) - n ^ 2 - 4 * n - 3 :=
sorry

end seq_solution_l44_44447


namespace smallest_Norwegian_l44_44996

def is_Norwegian (n : ℕ) : Prop :=
  ∃ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ n = a * b * c ∧ a + b + c = 2022

theorem smallest_Norwegian :
  ∀ n : ℕ, is_Norwegian n → 1344 ≤ n := by
  sorry

end smallest_Norwegian_l44_44996


namespace odd_function_def_l44_44018

noncomputable def f (x : ℝ) : ℝ :=
if x < 0 then x * (x - 1)
else -x * (x + 1)

theorem odd_function_def {x : ℝ} (h : x > 0) :
  f x = -x * (x + 1) :=
by
  sorry

end odd_function_def_l44_44018


namespace deformable_to_triangle_l44_44481

-- We define a planar polygon with n rods connected by hinges
structure PlanarPolygon (n : ℕ) :=
  (rods : Fin n → ℝ)
  (connections : Fin n → Fin n → Prop)

-- Define the conditions for the rods being rigid and connections (hinges)
def rigid_rod (n : ℕ) : PlanarPolygon n → Prop := λ poly => 
  ∀ i j, poly.connections i j → poly.rods i = poly.rods j

-- Defining the theorem for deformation into a triangle
theorem deformable_to_triangle (n : ℕ) (p : PlanarPolygon n) : 
  (n > 4) ↔ ∃ q : PlanarPolygon 3, true :=
by
  sorry

end deformable_to_triangle_l44_44481


namespace seq_sum_terms_l44_44588

def S (n : ℕ) : ℕ := 3^n - 2

def a (n : ℕ) : ℕ :=
  if n = 1 then 1
  else 2 * 3^(n - 1)

theorem seq_sum_terms (n : ℕ) : 
  a n = if n = 1 then 1 else 2 * 3^(n-1) :=
sorry

end seq_sum_terms_l44_44588


namespace correct_assignment_l44_44794

-- Definition of conditions
def is_variable_free (e : String) : Prop := -- a simplistic placeholder
  e ∈ ["A", "B", "C", "D", "x"]

def valid_assignment (lhs : String) (rhs : String) : Prop :=
  is_variable_free lhs ∧ ¬(is_variable_free rhs)

-- The statement of the proof problem
theorem correct_assignment : valid_assignment "A" "A * A + A - 2" :=
by
  sorry

end correct_assignment_l44_44794


namespace largest_number_Ahn_can_get_l44_44545

theorem largest_number_Ahn_can_get :
  ∃ (n : ℕ), (100 ≤ n ∧ n ≤ 999) ∧ (∀ m, (100 ≤ m ∧ m ≤ 999) → 3 * (500 - m) ≤ 1200) := sorry

end largest_number_Ahn_can_get_l44_44545


namespace raspberry_pies_l44_44738

theorem raspberry_pies (total_pies : ℕ) (r_peach : ℕ) (r_strawberry : ℕ) (r_raspberry : ℕ) (r_sum : ℕ) :
    total_pies = 36 → r_peach = 2 → r_strawberry = 5 → r_raspberry = 3 → r_sum = (r_peach + r_strawberry + r_raspberry) →
    (total_pies : ℝ) / (r_sum : ℝ) * (r_raspberry : ℝ) = 10.8 :=
by
    -- This theorem is intended to state the problem.
    sorry

end raspberry_pies_l44_44738


namespace cost_per_chicken_l44_44735

-- Definitions for conditions
def totalBirds : ℕ := 15
def ducks : ℕ := totalBirds / 3
def chickens : ℕ := totalBirds - ducks
def feed_cost : ℕ := 20

-- Theorem stating the cost per chicken
theorem cost_per_chicken : (feed_cost / chickens) = 2 := by
  sorry

end cost_per_chicken_l44_44735


namespace sqrt_subtraction_l44_44271

theorem sqrt_subtraction :
  (Real.sqrt (49 + 81)) - (Real.sqrt (36 - 9)) = (Real.sqrt 130) - (3 * Real.sqrt 3) :=
sorry

end sqrt_subtraction_l44_44271


namespace volume_of_solid_bounded_by_planes_l44_44910

theorem volume_of_solid_bounded_by_planes (a : ℝ) : 
  ∃ v, v = (a ^ 3) / 6 :=
by 
  sorry

end volume_of_solid_bounded_by_planes_l44_44910


namespace decagon_interior_angle_measure_l44_44153

-- Define the type for a regular polygon
structure RegularPolygon (n : Nat) :=
  (interior_angle_sum : Nat := (n - 2) * 180)
  (side_count : Nat := n)
  (regularity : Prop := True)  -- All angles are equal

-- Define the degree measure of an interior angle of a regular polygon
def interiorAngle (p : RegularPolygon 10) : Nat :=
  (p.interior_angle_sum) / p.side_count

-- The theorem to be proved
theorem decagon_interior_angle_measure : 
  ∀ (p : RegularPolygon 10), interiorAngle p = 144 := by
  -- The proof will be here, but for now, we use sorry
  sorry

end decagon_interior_angle_measure_l44_44153


namespace find_a2015_l44_44470

variable (a : ℕ → ℝ)

-- Conditions
axiom h1 : a 1 = 1
axiom h2 : a 2 = 3
axiom h3 : ∀ n : ℕ, n > 0 → a (n + 1) - a n ≤ 2 ^ n
axiom h4 : ∀ n : ℕ, n > 0 → a (n + 2) - a n ≥ 3 * 2 ^ n

-- Theorem stating the solution
theorem find_a2015 : a 2015 = 2 ^ 2015 - 1 :=
by sorry

end find_a2015_l44_44470


namespace min_ring_cuts_l44_44355

/-- Prove that the minimum number of cuts needed to pay the owner daily with an increasing 
    number of rings for 11 days, given a chain of 11 rings, is 2. -/
theorem min_ring_cuts {days : ℕ} {rings : ℕ} : days = 11 → rings = 11 → (∃ cuts : ℕ, cuts = 2) :=
by intros; sorry

end min_ring_cuts_l44_44355


namespace expand_polynomials_l44_44598

def p (z : ℤ) := 3 * z^3 + 4 * z^2 - 2 * z + 1
def q (z : ℤ) := 2 * z^2 - 3 * z + 5
def r (z : ℤ) := 10 * z^5 - 8 * z^4 + 11 * z^3 + 5 * z^2 - 10 * z + 5

theorem expand_polynomials (z : ℤ) : (p z) * (q z) = r z :=
by sorry

end expand_polynomials_l44_44598


namespace no_positive_integer_satisfies_inequality_l44_44083

theorem no_positive_integer_satisfies_inequality :
  ∀ x : ℕ, 0 < x → ¬ (15 < -3 * (x : ℤ) + 18) := by
  sorry

end no_positive_integer_satisfies_inequality_l44_44083


namespace corey_lowest_score_l44_44999

theorem corey_lowest_score
  (e1 e2 e3 e4 : ℕ)
  (h1 : e1 = 84)
  (h2 : e2 = 67)
  (max_score : ∀ (e : ℕ), e ≤ 100)
  (avg_at_least_75 : (e1 + e2 + e3 + e4) / 4 ≥ 75) :
  e3 ≥ 49 ∨ e4 ≥ 49 :=
by
  sorry

end corey_lowest_score_l44_44999


namespace polynomial_factorization_l44_44343

theorem polynomial_factorization (a b c : ℝ) :
  a * (b - c)^3 + b * (c - a)^3 + c * (a - b)^3 + (a - b)^2 * (b - c)^2 * (c - a)^2
  = (a - b) * (b - c) * (c - a) * (a + b + c + a * b * c) :=
sorry

end polynomial_factorization_l44_44343


namespace ABCD_eq_neg1_l44_44413

noncomputable def A := (Real.sqrt 2013 + Real.sqrt 2012)
noncomputable def B := (- Real.sqrt 2013 - Real.sqrt 2012)
noncomputable def C := (Real.sqrt 2013 - Real.sqrt 2012)
noncomputable def D := (Real.sqrt 2012 - Real.sqrt 2013)

theorem ABCD_eq_neg1 : A * B * C * D = -1 :=
by sorry

end ABCD_eq_neg1_l44_44413


namespace vertical_asymptotes_A_plus_B_plus_C_l44_44977

noncomputable def A : ℤ := -6
noncomputable def B : ℤ := 5
noncomputable def C : ℤ := 12

theorem vertical_asymptotes_A_plus_B_plus_C :
  (x + 1) * (x - 3) * (x - 4) = x^3 + A*x^2 + B*x + C ∧ A + B + C = 11 := by
  sorry

end vertical_asymptotes_A_plus_B_plus_C_l44_44977


namespace derivative_of_odd_is_even_l44_44800

variable (f : ℝ → ℝ) (g : ℝ → ℝ)

-- Assume f is an odd function
axiom f_odd : ∀ x, f (-x) = -f x

-- Assume g is the derivative of f
axiom g_derivative : ∀ x, g x = deriv f x

-- Goal: Prove that g is an even function, i.e., g(-x) = g(x)
theorem derivative_of_odd_is_even : ∀ x, g (-x) = g x :=
by
  sorry

end derivative_of_odd_is_even_l44_44800


namespace find_number_l44_44135

theorem find_number (n : ℕ) (h : Nat.factorial 4 / Nat.factorial (4 - n) = 24) : n = 3 :=
by
  sorry

end find_number_l44_44135


namespace diagonals_in_nine_sided_polygon_l44_44524

-- Given a regular polygon with 9 sides
def regular_polygon_sides : ℕ := 9

-- To find the number of diagonals in a polygon, we use the formula
noncomputable def number_of_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

-- We need to prove this particular instance where the number of sides is 9
theorem diagonals_in_nine_sided_polygon : number_of_diagonals regular_polygon_sides = 27 := 
by sorry

end diagonals_in_nine_sided_polygon_l44_44524


namespace abs_inequality_solution_l44_44062

theorem abs_inequality_solution (x : ℝ) : 
  (|5 - 2*x| >= 3) ↔ (x ≤ 1 ∨ x ≥ 4) := sorry

end abs_inequality_solution_l44_44062


namespace intersection_result_l44_44416

def A : Set ℝ := {x | |x - 2| ≤ 2}

def B : Set ℝ := {y | ∃ x ∈ A, y = -2 * x + 2}

def intersection : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}

theorem intersection_result : (A ∩ B) = intersection :=
by
  sorry

end intersection_result_l44_44416


namespace hoopit_toes_l44_44702

theorem hoopit_toes (h : ℕ) : 
  (7 * (4 * h) + 8 * (2 * 5) = 164) -> h = 3 :=
by
  sorry

end hoopit_toes_l44_44702


namespace two_colonies_reach_limit_same_time_l44_44937

theorem two_colonies_reach_limit_same_time
  (doubles_in_size : ∀ (n : ℕ), n = n * 2)
  (reaches_limit_in_25_days : ∃ N : ℕ, ∀ t : ℕ, t = 25 → N = N * 2^t) :
  ∀ t : ℕ, t = 25 := sorry

end two_colonies_reach_limit_same_time_l44_44937


namespace choose_two_out_of_three_l44_44382

-- Define the number of vegetables as n and the number to choose as k
def n : ℕ := 3
def k : ℕ := 2

-- The combination formula C(n, k) == n! / (k! * (n - k)!)
def combination (n k : ℕ) : ℕ := n.choose k

-- Problem statement: Prove that the number of ways to choose 2 out of 3 vegetables is 3
theorem choose_two_out_of_three : combination n k = 3 :=
by
  sorry

end choose_two_out_of_three_l44_44382


namespace sqrt_two_irrational_l44_44179

theorem sqrt_two_irrational : ¬ ∃ (a b : ℕ), b ≠ 0 ∧ (a / b) ^ 2 = 2 :=
by
  sorry

end sqrt_two_irrational_l44_44179


namespace hose_filling_time_l44_44659

theorem hose_filling_time :
  ∀ (P A B C : ℝ), 
  (P / 3 = A + B) →
  (P / 5 = A + C) →
  (P / 4 = B + C) →
  (P / (A + B + C) = 2.55) :=
by
  intros P A B C hAB hAC hBC
  sorry

end hose_filling_time_l44_44659


namespace sandwiches_cost_l44_44380

theorem sandwiches_cost (sandwiches sodas : ℝ) 
  (cost_sandwich : ℝ := 2.44)
  (cost_soda : ℝ := 0.87)
  (num_sodas : ℕ := 4)
  (total_cost : ℝ := 8.36)
  (total_soda_cost : ℝ := cost_soda * num_sodas)
  (total_sandwich_cost : ℝ := total_cost - total_soda_cost):
  sandwiches = (total_sandwich_cost / cost_sandwich) → sandwiches = 2 := by 
  sorry

end sandwiches_cost_l44_44380


namespace find_c_l44_44073

-- Let a, b, c, d, and e be positive consecutive integers.
variables {a b c d e : ℕ}

-- Conditions: 
def conditions (a b c d e : ℕ) : Prop :=
  a < b ∧ b < c ∧ c < d ∧ d < e ∧ 
  a + b = e - 1 ∧
  a * b = d + 1

-- Proof statement
theorem find_c (h : conditions a b c d e) : c = 4 :=
by sorry

end find_c_l44_44073


namespace total_spent_on_concert_tickets_l44_44112

theorem total_spent_on_concert_tickets : 
  let price_per_ticket := 4
  let number_of_tickets := 3 + 5
  let discount_threshold := 5
  let discount_rate := 0.10
  let service_fee_per_ticket := 2
  let initial_cost := number_of_tickets * price_per_ticket
  let discount := if number_of_tickets > discount_threshold then discount_rate * initial_cost else 0
  let discounted_cost := initial_cost - discount
  let service_fee := number_of_tickets * service_fee_per_ticket
  let total_cost := discounted_cost + service_fee
  total_cost = 44.8 :=
by
  sorry

end total_spent_on_concert_tickets_l44_44112


namespace line_parabola_intersection_one_point_l44_44976

theorem line_parabola_intersection_one_point (k : ℝ) :
  (∃ y : ℝ, (-3 * y^2 - 4 * y + 7 = k) ∧ ∀ y1 y2 : ℝ, ( 3 * y1^2 + 4 * y1 + (k - 7) = 0 → 3 * y2^2 + 4 * y2 + (k - 7) = 0 → y1 = y2)) ↔ (k = 25 / 3) :=
by
  sorry

end line_parabola_intersection_one_point_l44_44976


namespace count_valid_numbers_l44_44683

def digits_set : List ℕ := [0, 2, 4, 7, 8, 9]

def divisible_by_4 (n : ℕ) : Prop :=
  n % 4 = 0

def divisible_by_3 (n : ℕ) : Prop :=
  n % 3 = 0

def sum_digits (digits : List ℕ) : ℕ :=
  List.sum digits

def last_two_digits_divisibility (last_two_digits : ℕ) : Prop :=
  last_two_digits % 4 = 0

def number_is_valid (digits : List ℕ) : Prop :=
  sum_digits digits % 3 = 0

theorem count_valid_numbers :
  let possible_digits := [0, 2, 4, 7, 8, 9]
  let positions := 5
  let combinations := Nat.pow (List.length possible_digits) (positions - 1)
  let last_digit_choices := [0, 4, 8]
  3888 = 3 * combinations :=
sorry

end count_valid_numbers_l44_44683


namespace range_of_p_l44_44251

theorem range_of_p (p : ℝ) (a_n b_n : ℕ → ℝ)
  (ha : ∀ n, a_n n = -n + p)
  (hb : ∀ n, b_n n = 3^(n-4))
  (C_n : ℕ → ℝ)
  (hC : ∀ n, C_n n = if a_n n ≥ b_n n then a_n n else b_n n)
  (hc : ∀ n : ℕ, n ≥ 1 → C_n n > C_n 4) :
  4 < p ∧ p < 7 :=
sorry

end range_of_p_l44_44251


namespace penelope_mandm_candies_l44_44258

theorem penelope_mandm_candies (m n : ℕ) (r : ℝ) :
  (m / n = 5 / 3) → (n = 15) → (m = 25) :=
by
  sorry

end penelope_mandm_candies_l44_44258


namespace smallest_crate_side_l44_44658

/-- 
A crate measures some feet by 8 feet by 12 feet on the inside. 
A stone pillar in the shape of a right circular cylinder must fit into the crate for shipping so that 
it rests upright when the crate sits on at least one of its six sides. 
The radius of the pillar is 7 feet. 
Prove that the length of the crate's smallest side is 8 feet.
-/
theorem smallest_crate_side (x : ℕ) (hx : x >= 14) : min (min x 8) 12 = 8 :=
by {
  sorry
}

end smallest_crate_side_l44_44658


namespace percent_of_x_is_y_l44_44057

variable {x y : ℝ}

theorem percent_of_x_is_y
  (h : 0.5 * (x - y) = 0.4 * (x + y)) :
  y = (1 / 9) * x :=
sorry

end percent_of_x_is_y_l44_44057


namespace probability_of_green_ball_l44_44874

theorem probability_of_green_ball :
  let P_X := 0.2
  let P_Y := 0.5
  let P_Z := 0.3
  let P_green_given_X := 5 / 10
  let P_green_given_Y := 3 / 10
  let P_green_given_Z := 8 / 10
  P_green_given_X * P_X + P_green_given_Y * P_Y + P_green_given_Z * P_Z = 0.49 :=
by {
  sorry
}

end probability_of_green_ball_l44_44874


namespace consecutive_log_sum_l44_44178

theorem consecutive_log_sum : 
  ∃ c d: ℤ, (c + 1 = d) ∧ (c < Real.logb 5 125) ∧ (Real.logb 5 125 < d) ∧ (c + d = 5) :=
sorry

end consecutive_log_sum_l44_44178


namespace parallel_lines_m_eq_neg2_l44_44532

def l1_equation (m : ℝ) (x y: ℝ) : Prop :=
  (m+1) * x + y - 1 = 0

def l2_equation (m : ℝ) (x y: ℝ) : Prop :=
  2 * x + m * y - 1 = 0

theorem parallel_lines_m_eq_neg2 (m : ℝ) :
  (∀ x y : ℝ, l1_equation m x y) →
  (∀ x y : ℝ, l2_equation m x y) →
  (m ≠ 1) →
  (m = -2) :=
sorry

end parallel_lines_m_eq_neg2_l44_44532


namespace part_a_part_b_l44_44134

variable {A B C A₁ B₁ C₁ : Prop}
variables {a b c a₁ b₁ c₁ S S₁ : ℝ}

-- Assume basic conditions of triangles
variable (h1 : IsTriangle A B C)
variable (h2 : IsTriangleWithCentersAndSquares A B C A₁ B₁ C₁ a b c a₁ b₁ c₁ S S₁)
variable (h3 : IsExternalSquaresConstructed A B C A₁ B₁ C₁)

-- Part (a)
theorem part_a : a₁^2 + b₁^2 + c₁^2 = a^2 + b^2 + c^2 + 6 * S := 
sorry

-- Part (b)
theorem part_b : S₁ - S = (a^2 + b^2 + c^2) / 8 := 
sorry

end part_a_part_b_l44_44134


namespace contradiction_assumption_l44_44340

-- Define the numbers x, y, z
variables (x y z : ℝ)

-- Define the assumption that all three numbers are non-positive
def all_non_positive (x y z : ℝ) : Prop := x ≤ 0 ∧ y ≤ 0 ∧ z ≤ 0

-- State the proposition to prove using the method of contradiction
theorem contradiction_assumption (h : all_non_positive x y z) : ¬ (x > 0 ∨ y > 0 ∨ z > 0) :=
by
  sorry

end contradiction_assumption_l44_44340


namespace q_l44_44204

-- Definitions for the problem conditions
def slips := 50
def numbers := 12
def slips_per_number := 5
def drawn_slips := 5
def binom := Nat.choose -- Lean function for binomial coefficients

-- Define the probabilities p' and q'
def p' := 12 / (binom slips drawn_slips)
def favorable_q' := (binom numbers 2) * (binom slips_per_number 3) * (binom slips_per_number 2)
def q' := favorable_q' / (binom slips drawn_slips)

-- The statement we need to prove
theorem q'_over_p'_equals_550 : q' / p' = 550 :=
by sorry

end q_l44_44204


namespace new_mean_rent_l44_44722

theorem new_mean_rent
  (num_friends : ℕ)
  (avg_rent : ℕ)
  (original_rent_increased : ℕ)
  (increase_percentage : ℝ)
  (new_mean_rent : ℕ) :
  num_friends = 4 →
  avg_rent = 800 →
  original_rent_increased = 1400 →
  increase_percentage = 0.2 →
  new_mean_rent = 870 :=
by
  intros h1 h2 h3 h4
  sorry

end new_mean_rent_l44_44722


namespace geometric_sequence_product_l44_44624

variable {α : Type*} [LinearOrderedField α]
variable (a : ℕ → α)

theorem geometric_sequence_product (h : a 7 * a 12 = 5) : 
  a 8 * a 9 * a 10 * a 11 = 25 :=
by
  -- Proof goes here
  sorry

end geometric_sequence_product_l44_44624


namespace sufficient_but_not_necessary_l44_44407

-- Definitions for lines a and b, and planes alpha and beta
variables {a b : Type} {α β : Type}

-- predicate for line a being in plane α
def line_in_plane (a : Type) (α : Type) : Prop := sorry

-- predicate for line b being perpendicular to plane β
def line_perpendicular_plane (b : Type) (β : Type) : Prop := sorry

-- predicate for plane α being parallel to plane β
def plane_parallel_plane (α : Type) (β : Type) : Prop := sorry

-- predicate for line a being perpendicular to line b
def line_perpendicular_line (a : Type) (b : Type) : Prop := sorry

-- Proof of the statement: The condition of line a being in plane α, line b being perpendicular to plane β,
-- and plane α being parallel to plane β is sufficient but not necessary for line a being perpendicular to line b.
theorem sufficient_but_not_necessary
  (a b : Type) (α β : Type)
  (h1 : line_in_plane a α)
  (h2 : line_perpendicular_plane b β)
  (h3 : plane_parallel_plane α β) :
  line_perpendicular_line a b :=
sorry

end sufficient_but_not_necessary_l44_44407


namespace repeated_pair_exists_l44_44409

theorem repeated_pair_exists (a : Fin 99 → Fin 10)
  (h1 : ∀ n : Fin 98, a n = 1 → a (n + 1) ≠ 2)
  (h2 : ∀ n : Fin 98, a n = 3 → a (n + 1) ≠ 4) :
  ∃ k l : Fin 98, k ≠ l ∧ a k = a l ∧ a (k + 1) = a (l + 1) :=
sorry

end repeated_pair_exists_l44_44409


namespace calculate_expression_l44_44930

theorem calculate_expression :
  -2^3 * (-3)^2 / (9 / 8) - abs (1 / 2 - 3 / 2) = -65 :=
by
  sorry

end calculate_expression_l44_44930


namespace total_amount_paid_is_correct_l44_44905

-- Define constants based on conditions
def cost_per_night_per_person : ℕ := 40
def number_of_people : ℕ := 3
def number_of_nights : ℕ := 3

-- Define the total cost calculation
def total_cost := cost_per_night_per_person * number_of_people * number_of_nights

-- The proof problem statement
theorem total_amount_paid_is_correct :
  total_cost = 360 :=
by
  sorry

end total_amount_paid_is_correct_l44_44905


namespace math_expr_evaluation_l44_44107

theorem math_expr_evaluation :
  3 + 15 / 3 - 2^2 + 1 = 5 :=
by
  -- The proof will be filled here
  sorry

end math_expr_evaluation_l44_44107


namespace probability_of_drawing_red_ball_l44_44720

theorem probability_of_drawing_red_ball (total_balls red_balls : ℕ) (h_total : total_balls = 10) (h_red : red_balls = 7) : (red_balls : ℚ) / total_balls = 7 / 10 :=
by
  sorry

end probability_of_drawing_red_ball_l44_44720


namespace students_behind_minyoung_l44_44860

-- Definition of the initial conditions
def total_students : ℕ := 35
def students_in_front_of_minyoung : ℕ := 27

-- The question we want to prove
theorem students_behind_minyoung : (total_students - (students_in_front_of_minyoung + 1) = 7) := 
by 
  sorry

end students_behind_minyoung_l44_44860


namespace Jason_total_money_l44_44575

theorem Jason_total_money :
  let quarter_value := 0.25
  let dime_value := 0.10
  let nickel_value := 0.05
  let initial_quarters := 49
  let initial_dimes := 32
  let initial_nickels := 18
  let additional_quarters := 25
  let additional_dimes := 15
  let additional_nickels := 10
  let initial_money := initial_quarters * quarter_value + initial_dimes * dime_value + initial_nickels * nickel_value
  let additional_money := additional_quarters * quarter_value + additional_dimes * dime_value + additional_nickels * nickel_value
  initial_money + additional_money = 24.60 :=
by
  sorry

end Jason_total_money_l44_44575


namespace prob_at_least_one_palindrome_correct_l44_44579

-- Define a function to represent the probability calculation.
def probability_at_least_one_palindrome : ℚ :=
  let prob_digit_palindrome : ℚ := 1 / 100
  let prob_letter_palindrome : ℚ := 1 / 676
  let prob_both_palindromes : ℚ := (1 / 100) * (1 / 676)
  (prob_digit_palindrome + prob_letter_palindrome - prob_both_palindromes)

-- The theorem we are stating based on the given problem and solution:
theorem prob_at_least_one_palindrome_correct : probability_at_least_one_palindrome = 427 / 2704 :=
by
  -- We assume this step for now as we are just stating the theorem
  sorry

end prob_at_least_one_palindrome_correct_l44_44579


namespace price_difference_l44_44044

theorem price_difference (total_cost shirt_price : ℝ) (h1 : total_cost = 80.34) (h2 : shirt_price = 36.46) :
  (total_cost - shirt_price) - shirt_price = 7.42 :=
by
  sorry

end price_difference_l44_44044


namespace age_of_jerry_l44_44829

variable (M J : ℕ)

theorem age_of_jerry (h1 : M = 2 * J - 5) (h2 : M = 19) : J = 12 := by
  sorry

end age_of_jerry_l44_44829


namespace exist_2022_good_numbers_with_good_sum_l44_44818

def is_good (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k * (k + 1)

theorem exist_2022_good_numbers_with_good_sum :
  ∃ (a : Fin 2022 → ℕ), (∀ i j : Fin 2022, i ≠ j → a i ≠ a j) ∧ (∀ i : Fin 2022, is_good (a i)) ∧ is_good (Finset.univ.sum a) :=
sorry

end exist_2022_good_numbers_with_good_sum_l44_44818


namespace kaleb_final_score_l44_44834

variable (score_first_half : ℝ) (bonus_special_q : ℝ) (bonus_streak : ℝ) (score_second_half : ℝ) (penalty_speed_round : ℝ) (penalty_lightning_round : ℝ)

-- Given conditions from the problem statement
def kaleb_initial_scores (score_first_half score_second_half : ℝ) := 
  score_first_half = 43 ∧ score_second_half = 23

def kaleb_bonuses (score_first_half bonus_special_q bonus_streak : ℝ) :=
  bonus_special_q = 0.20 * score_first_half ∧ bonus_streak = 0.05 * score_first_half

def kaleb_penalties (score_second_half penalty_speed_round penalty_lightning_round : ℝ) := 
  penalty_speed_round = 0.10 * score_second_half ∧ penalty_lightning_round = 0.08 * score_second_half

-- The final score adjusted with all bonuses and penalties
def kaleb_adjusted_score (score_first_half score_second_half bonus_special_q bonus_streak penalty_speed_round penalty_lightning_round : ℝ) : ℝ := 
  score_first_half + bonus_special_q + bonus_streak + score_second_half - penalty_speed_round - penalty_lightning_round

theorem kaleb_final_score :
  kaleb_initial_scores score_first_half score_second_half ∧
  kaleb_bonuses score_first_half bonus_special_q bonus_streak ∧
  kaleb_penalties score_second_half penalty_speed_round penalty_lightning_round →
  kaleb_adjusted_score score_first_half score_second_half bonus_special_q bonus_streak penalty_speed_round penalty_lightning_round = 72.61 :=
by
  intros
  sorry

end kaleb_final_score_l44_44834


namespace expression_is_product_l44_44752

def not_sum (a x : Int) : Prop :=
  ¬(a + x = -7 * x)

def not_difference (a x : Int) : Prop :=
  ¬(a - x = -7 * x)

def not_quotient (a x : Int) : Prop :=
  ¬(a / x = -7 * x)

theorem expression_is_product (x : Int) : 
  not_sum (-7) x ∧ not_difference (-7) x ∧ not_quotient (-7) x → (-7 * x = -7 * x) :=
by sorry

end expression_is_product_l44_44752


namespace lowest_fraction_done_in_an_hour_by_two_people_l44_44115

def a_rate : ℚ := 1 / 4
def b_rate : ℚ := 1 / 5
def c_rate : ℚ := 1 / 6

theorem lowest_fraction_done_in_an_hour_by_two_people : 
  min (min (a_rate + b_rate) (a_rate + c_rate)) (b_rate + c_rate) = 11 / 30 := 
by
  sorry

end lowest_fraction_done_in_an_hour_by_two_people_l44_44115


namespace inverse_function_of_1_div_2_pow_eq_log_base_1_div_2_l44_44278

noncomputable def inverse_of_half_pow (x : ℝ) : ℝ := Real.log x / Real.log (1 / 2)

theorem inverse_function_of_1_div_2_pow_eq_log_base_1_div_2 (x : ℝ) (hx : 0 < x) :
  inverse_of_half_pow x = Real.log x / Real.log (1 / 2) :=
by
  sorry

end inverse_function_of_1_div_2_pow_eq_log_base_1_div_2_l44_44278


namespace find_value_perpendicular_distances_l44_44019

variable {R a b c D E F : ℝ}
variable {ABC : Triangle}

-- Assume the distances from point P on the circumcircle of triangle ABC
-- to the sides BC, CA, and AB respectively.
axiom D_def : D = R * a / (2 * R)
axiom E_def : E = R * b / (2 * R)
axiom F_def : F = R * c / (2 * R)

theorem find_value_perpendicular_distances
    (a b c R : ℝ) (D E F : ℝ) 
    (hD : D = R * a / (2 * R)) 
    (hE : E = R * b / (2 * R)) 
    (hF : F = R * c / (2 * R)) : 
    a^2 * D^2 + b^2 * E^2 + c^2 * F^2 = (a^4 + b^4 + c^4) / (4 * R^2) :=
by
  sorry

end find_value_perpendicular_distances_l44_44019


namespace probability_blue_face_eq_one_third_l44_44146

-- Define the necessary conditions
def numberOfFaces : Nat := 12
def numberOfBlueFaces : Nat := 4

-- Define the term representing the probability
def probabilityOfBlueFace : ℚ := numberOfBlueFaces / numberOfFaces

-- The theorem to prove that the probability is 1/3
theorem probability_blue_face_eq_one_third :
  probabilityOfBlueFace = (1 : ℚ) / 3 :=
  by
  sorry

end probability_blue_face_eq_one_third_l44_44146


namespace opposite_of_six_is_neg_six_l44_44239

-- Define the condition that \( a \) is the opposite of \( 6 \)
def is_opposite_of_six (a : Int) : Prop := a = -6

-- Prove that \( a = -6 \) given that \( a \) is the opposite of \( 6 \)
theorem opposite_of_six_is_neg_six (a : Int) (h : is_opposite_of_six a) : a = -6 :=
by
  sorry

end opposite_of_six_is_neg_six_l44_44239


namespace find_dividend_l44_44965

noncomputable def dividend (divisor quotient remainder : ℕ) : ℕ :=
  (divisor * quotient) + remainder

theorem find_dividend :
  ∀ (divisor quotient remainder : ℕ), 
  divisor = 16 → 
  quotient = 8 → 
  remainder = 4 → 
  dividend divisor quotient remainder = 132 :=
by
  intros divisor quotient remainder hdiv hquo hrem
  sorry

end find_dividend_l44_44965


namespace has_zero_in_intervals_l44_44074

noncomputable def f (x : ℝ) : ℝ := (1 / 3) * x - Real.log x
noncomputable def f' (x : ℝ) : ℝ := (1 / 3) - (1 / x)

theorem has_zero_in_intervals : 
  (∃ x : ℝ, 0 < x ∧ x < 3 ∧ f x = 0) ∧ (∃ x : ℝ, 3 < x ∧ f x = 0) :=
sorry

end has_zero_in_intervals_l44_44074


namespace root_expr_calculation_l44_44690

theorem root_expr_calculation : (3 + Real.sqrt 10) * (Real.sqrt 2 - Real.sqrt 5) = -2 * Real.sqrt 2 - Real.sqrt 5 := 
by 
  sorry

end root_expr_calculation_l44_44690


namespace sum_of_angles_l44_44282

theorem sum_of_angles (A B C x y : ℝ) 
  (hA : A = 34) 
  (hB : B = 80) 
  (hC : C = 30)
  (pentagon_angles_sum : A + B + (360 - x) + 90 + (120 - y) = 540) : 
  x + y = 144 :=
by
  sorry

end sum_of_angles_l44_44282


namespace consecutive_integers_sum_l44_44047

theorem consecutive_integers_sum (n : ℤ) (h : n * (n + 1) = 20412) : n + (n + 1) = 287 :=
sorry

end consecutive_integers_sum_l44_44047


namespace find_integers_k_l44_44334

theorem find_integers_k (k : ℤ) : 
  (k = 15 ∨ k = 30) ↔ 
  (k ≥ 3 ∧ ∃ m n : ℤ, 1 < m ∧ m < k ∧ 1 < n ∧ n < k ∧ 
                       Int.gcd m k = 1 ∧ Int.gcd n k = 1 ∧ 
                       m + n > k ∧ k ∣ (m - 1) * (n - 1)) :=
by
  sorry -- Proof goes here

end find_integers_k_l44_44334


namespace multiple_of_5_add_multiple_of_10_l44_44855

theorem multiple_of_5_add_multiple_of_10 (p q : ℤ) (hp : ∃ m : ℤ, p = 5 * m) (hq : ∃ n : ℤ, q = 10 * n) : ∃ k : ℤ, p + q = 5 * k :=
by
  sorry

end multiple_of_5_add_multiple_of_10_l44_44855


namespace crate_weight_l44_44928

variable (C : ℝ)
variable (carton_weight : ℝ := 3)
variable (total_weight : ℝ := 96)
variable (num_crates : ℝ := 12)
variable (num_cartons : ℝ := 16)

theorem crate_weight :
  (num_crates * C + num_cartons * carton_weight = total_weight) → (C = 4) :=
by
  sorry

end crate_weight_l44_44928


namespace percentage_vanaspati_after_adding_ghee_l44_44119

theorem percentage_vanaspati_after_adding_ghee :
  ∀ (original_quantity new_pure_ghee percentage_ghee percentage_vanaspati : ℝ),
    original_quantity = 30 →
    percentage_ghee = 0.5 →
    percentage_vanaspati = 0.5 →
    new_pure_ghee = 20 →
    (percentage_vanaspati * original_quantity) /
    (original_quantity + new_pure_ghee) * 100 = 30 :=
by
  intros original_quantity new_pure_ghee percentage_ghee percentage_vanaspati
  sorry

end percentage_vanaspati_after_adding_ghee_l44_44119


namespace find_x0_l44_44344

noncomputable def f (x : ℝ) : ℝ := x * Real.log x
noncomputable def f' (x : ℝ) : ℝ := Real.log x + 1

theorem find_x0 (x_0 : ℝ) (h : f' x_0 = 2) : x_0 = Real.exp 1 :=
by
  sorry

end find_x0_l44_44344


namespace rooms_equation_l44_44486

theorem rooms_equation (x : ℕ) (h₁ : ∃ n, n = 6 * (x - 1)) (h₂ : ∃ m, m = 5 * x + 4) :
  6 * (x - 1) = 5 * x + 4 :=
sorry

end rooms_equation_l44_44486


namespace circle_center_coordinates_l44_44472

theorem circle_center_coordinates :
  ∃ c : ℝ × ℝ, (c = (1, -2)) ∧ 
  (∀ x y : ℝ, (x^2 + y^2 - 2*x + 4*y - 4 = 0 ↔ (x - 1)^2 + (y + 2)^2 = 9)) :=
by
  sorry

end circle_center_coordinates_l44_44472


namespace intersection_of_A_and_B_l44_44902

def A := Set.Ioo 1 3
def B := Set.Ioo 2 4

theorem intersection_of_A_and_B : A ∩ B = Set.Ioo 2 3 :=
by
  sorry

end intersection_of_A_and_B_l44_44902


namespace total_students_in_school_l44_44220

theorem total_students_in_school : 
  ∀ (number_of_deaf_students number_of_blind_students : ℕ), 
  (number_of_deaf_students = 180) → 
  (number_of_deaf_students = 3 * number_of_blind_students) → 
  (number_of_deaf_students + number_of_blind_students = 240) :=
by 
  sorry

end total_students_in_school_l44_44220


namespace inequality_solution_l44_44853

theorem inequality_solution (x : ℝ) : x^2 + x - 20 < 0 ↔ -5 < x ∧ x < 4 := 
by
  sorry

end inequality_solution_l44_44853


namespace visit_orders_l44_44807

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

noncomputable def num_permutations_cities (pohang busan geoncheon gimhae gyeongju : Type) : ℕ :=
  factorial 4

theorem visit_orders (pohang busan geoncheon gimhae gyeongju : Type) :
  num_permutations_cities pohang busan geoncheon gimhae gyeongju = 24 :=
by
  unfold num_permutations_cities
  norm_num
  sorry

end visit_orders_l44_44807


namespace negation_of_universal_proposition_l44_44111

theorem negation_of_universal_proposition :
  ¬ (∀ x : ℝ, 1 < x ∧ x < 2 → x^2 > 1) ↔ ∃ x : ℝ, 1 < x ∧ x < 2 ∧ x^2 ≤ 1 := 
sorry

end negation_of_universal_proposition_l44_44111


namespace arithmetic_sequence_condition_l44_44754

theorem arithmetic_sequence_condition {a : ℕ → ℤ} 
  (h_arith : ∃ d, ∀ n, a (n + 1) = a n + d)
  (m p q : ℕ) (hpq_pos : 0 < p) (hq_pos : 0 < q) (hm_pos : 0 < m) : 
  (p + q = 2 * m) → (a p + a q = 2 * a m) ∧ ¬((a p + a q = 2 * a m) → (p + q = 2 * m)) :=
by 
  sorry

end arithmetic_sequence_condition_l44_44754


namespace parabola_x_intercepts_l44_44903

theorem parabola_x_intercepts :
  ∃! y : ℝ, -3 * y^2 + 2 * y + 4 = y := 
by
  sorry

end parabola_x_intercepts_l44_44903


namespace represent_in_scientific_notation_l44_44318

def million : ℕ := 10^6
def rural_residents : ℝ := 42.39 * million

theorem represent_in_scientific_notation :
  42.39 * 10^6 = 4.239 * 10^7 :=
by
  -- The proof is omitted.
  sorry

end represent_in_scientific_notation_l44_44318


namespace greatest_int_less_than_neg_19_div_3_l44_44564

theorem greatest_int_less_than_neg_19_div_3 : ∃ n : ℤ, n = -7 ∧ n < (-19 / 3 : ℚ) ∧ (-19 / 3 : ℚ) < n + 1 := 
by
  sorry

end greatest_int_less_than_neg_19_div_3_l44_44564


namespace Ella_jellybeans_l44_44657

-- Definitions based on conditions from part (a)
def Dan_volume := 10
def Dan_jellybeans := 200
def scaling_factor := 3

-- Prove that Ella's box holds 5400 jellybeans
theorem Ella_jellybeans : scaling_factor^3 * Dan_jellybeans = 5400 := 
by
  sorry

end Ella_jellybeans_l44_44657


namespace flag_arrangement_modulo_1000_l44_44400

theorem flag_arrangement_modulo_1000 :
  let red_flags := 8
  let white_flags := 8
  let black_flags := 1
  let total_flags := red_flags + white_flags + black_flags
  let number_of_gaps := total_flags + 1
  let valid_arrangements := (Nat.choose number_of_gaps white_flags) * (number_of_gaps - 2)
  valid_arrangements % 1000 = 315 :=
by
  sorry

end flag_arrangement_modulo_1000_l44_44400


namespace angle_B_in_triangle_is_pi_over_6_l44_44172

theorem angle_B_in_triangle_is_pi_over_6
  (a b c : ℝ)
  (A B C : ℝ)
  (h₁ : a > 0) (h₂ : b > 0) (h₃ : c > 0)
  (h₄ : A + B + C = π)
  (h₅ : b * (Real.cos C) / (Real.cos B) + c = (2 * Real.sqrt 3 / 3) * a) :
  B = π / 6 :=
by sorry

end angle_B_in_triangle_is_pi_over_6_l44_44172


namespace volume_conversion_l44_44100

theorem volume_conversion (v_feet : ℕ) (h : v_feet = 250) : (v_feet / 27 : ℚ) = 250 / 27 := by
  sorry

end volume_conversion_l44_44100


namespace parabola_through_origin_l44_44148

theorem parabola_through_origin {a b c : ℝ} :
  (c = 0 ↔ ∀ x, (0, 0) = (x, a * x^2 + b * x + c)) :=
sorry

end parabola_through_origin_l44_44148


namespace sheelas_total_net_monthly_income_l44_44419

noncomputable def totalNetMonthlyIncome
    (PrimaryJobIncome : ℝ)
    (FreelanceIncome : ℝ)
    (FreelanceIncomeTaxRate : ℝ)
    (AnnualInterestIncome : ℝ)
    (InterestIncomeTaxRate : ℝ) : ℝ :=
    let PrimaryJobMonthlyIncome := 5000 / 0.20
    let FreelanceIncomeTax := FreelanceIncome * FreelanceIncomeTaxRate
    let NetFreelanceIncome := FreelanceIncome - FreelanceIncomeTax
    let InterestIncomeTax := AnnualInterestIncome * InterestIncomeTaxRate
    let NetAnnualInterestIncome := AnnualInterestIncome - InterestIncomeTax
    let NetMonthlyInterestIncome := NetAnnualInterestIncome / 12
    PrimaryJobMonthlyIncome + NetFreelanceIncome + NetMonthlyInterestIncome

theorem sheelas_total_net_monthly_income :
    totalNetMonthlyIncome 25000 3000 0.10 2400 0.05 = 27890 := 
by
    sorry

end sheelas_total_net_monthly_income_l44_44419


namespace frequency_count_l44_44684

theorem frequency_count (n : ℕ) (f : ℝ) (h1 : n = 1000) (h2 : f = 0.4) : n * f = 400 := by
  sorry

end frequency_count_l44_44684


namespace most_suitable_for_comprehensive_survey_l44_44580

-- Definitions of the survey options
inductive SurveyOption
| A
| B
| C
| D

-- Condition definitions based on the problem statement
def comprehensive_survey (option : SurveyOption) : Prop :=
  option = SurveyOption.B

-- The theorem stating that the most suitable survey is option B
theorem most_suitable_for_comprehensive_survey : ∀ (option : SurveyOption), comprehensive_survey option ↔ option = SurveyOption.B :=
by
  intro option
  sorry

end most_suitable_for_comprehensive_survey_l44_44580


namespace length_of_plot_l44_44546

open Real

variable (breadth : ℝ) (length : ℝ)
variable (b : ℝ)

axiom H1 : length = b + 40
axiom H2 : 26.5 * (4 * b + 80) = 5300

theorem length_of_plot : length = 70 :=
by
  -- To prove: The length of the plot is 70 meters.
  exact sorry

end length_of_plot_l44_44546


namespace valid_differences_of_squares_l44_44103

theorem valid_differences_of_squares (n : ℕ) (h : 2 * n + 1 < 150) :
    (2 * n + 1 = 129 ∨ 2 * n +1 = 147) :=
by
  sorry

end valid_differences_of_squares_l44_44103


namespace smallest_prime_after_seven_consecutive_nonprimes_l44_44821

theorem smallest_prime_after_seven_consecutive_nonprimes :
  ∃ p, p > 96 ∧ Nat.Prime p ∧ ∀ n, 90 ≤ n ∧ n ≤ 96 → ¬Nat.Prime n :=
by
  sorry

end smallest_prime_after_seven_consecutive_nonprimes_l44_44821


namespace sum_sequence_conjecture_l44_44206

theorem sum_sequence_conjecture (a : ℕ → ℚ) (S : ℕ → ℚ) :
  (∀ n : ℕ+, a n = (8 * n) / ((2 * n - 1) ^ 2 * (2 * n + 1) ^ 2)) →
  (∀ n : ℕ+, S n = (S n + a (n + 1))) →
  (∀ n : ℕ+, S 1 = 8 / 9) →
  (∀ n : ℕ+, S n = ((2 * n + 1) ^ 2 - 1) / (2 * n + 1) ^ 2) :=
by {
  sorry
}

end sum_sequence_conjecture_l44_44206


namespace apple_selling_price_l44_44149

theorem apple_selling_price (CP SP Loss : ℝ) (h₀ : CP = 18) (h₁ : Loss = (1/6) * CP) (h₂ : SP = CP - Loss) : SP = 15 :=
  sorry

end apple_selling_price_l44_44149


namespace tickets_sold_l44_44389

theorem tickets_sold (student_tickets non_student_tickets student_ticket_price non_student_ticket_price total_revenue : ℕ)
  (h1 : student_ticket_price = 5)
  (h2 : non_student_ticket_price = 8)
  (h3 : total_revenue = 930)
  (h4 : student_tickets = 90)
  (h5 : non_student_tickets = 60) :
  student_tickets + non_student_tickets = 150 := 
by 
  sorry

end tickets_sold_l44_44389


namespace nuts_needed_for_cookies_l44_44931

-- Given conditions
def total_cookies : Nat := 120
def fraction_nuts : Rat := 1 / 3
def fraction_chocolate : Rat := 0.25
def nuts_per_cookie : Nat := 3

-- Translated conditions as helpful functions
def cookies_with_nuts : Nat := Nat.floor (fraction_nuts * total_cookies)
def cookies_with_chocolate : Nat := Nat.floor (fraction_chocolate * total_cookies)
def cookies_with_both : Nat := total_cookies - cookies_with_nuts - cookies_with_chocolate
def total_cookies_with_nuts : Nat := cookies_with_nuts + cookies_with_both
def total_nuts_needed : Nat := total_cookies_with_nuts * nuts_per_cookie

-- Proof problem: proving that total nuts needed is 270
theorem nuts_needed_for_cookies : total_nuts_needed = 270 :=
by
  sorry

end nuts_needed_for_cookies_l44_44931


namespace product_of_distinct_solutions_l44_44980

theorem product_of_distinct_solutions (x y : ℝ) (h₁ : x ≠ y) (h₂ : x ≠ 0) (h₃ : y ≠ 0) (h₄ : x - 2 / x = y - 2 / y) :
  x * y = -2 :=
sorry

end product_of_distinct_solutions_l44_44980


namespace other_ticket_price_l44_44205

theorem other_ticket_price (total_tickets : ℕ) (total_sales : ℝ) (cheap_tickets : ℕ) (cheap_price : ℝ) (expensive_tickets : ℕ) (expensive_price : ℝ) :
  total_tickets = 380 →
  total_sales = 1972.50 →
  cheap_tickets = 205 →
  cheap_price = 4.50 →
  expensive_tickets = 380 - 205 →
  205 * 4.50 + expensive_tickets * expensive_price = 1972.50 →
  expensive_price = 6.00 :=
by
  intros
  -- proof will be filled here
  sorry

end other_ticket_price_l44_44205


namespace negation_of_universal_statement_l44_44844

theorem negation_of_universal_statement :
  (¬ (∀ x : ℝ, |x| + x^2 ≥ 0)) ↔ (∃ x : ℝ, |x| + x^2 < 0) :=
by
  sorry

end negation_of_universal_statement_l44_44844


namespace total_cost_of_books_l44_44803

theorem total_cost_of_books
  (C1 : ℝ) (C2 : ℝ)
  (h1 : C1 = 315)
  (h2 : 0.85 * C1 = 1.19 * C2) :
  C1 + C2 = 2565 :=
by 
  sorry

end total_cost_of_books_l44_44803


namespace simplify_expr_l44_44237

variable {x y : ℝ}

theorem simplify_expr (hx : x ≠ 0) (hy : y ≠ 0) :
  ((x^3 + 1) / x) * ((y^3 + 1) / y) - ((x^3 - 1) / y) * ((y^3 - 1) / x) = 2 * x^2 + 2 * y^2 :=
by sorry

end simplify_expr_l44_44237


namespace martha_apples_l44_44740

theorem martha_apples (initial_apples jane_apples extra_apples final_apples : ℕ)
  (h1 : initial_apples = 20)
  (h2 : jane_apples = 5)
  (h3 : extra_apples = 2)
  (h4 : final_apples = 4) :
  initial_apples - jane_apples - (jane_apples + extra_apples) - final_apples = final_apples := 
by
  sorry

end martha_apples_l44_44740


namespace ten_numbers_exists_l44_44013

theorem ten_numbers_exists :
  ∃ (a : Fin 10 → ℕ), 
    (∀ i j : Fin 10, i ≠ j → ¬ (a i ∣ a j))
    ∧ (∀ i j : Fin 10, i ≠ j → a i ^ 2 ∣ a j * a j) :=
sorry

end ten_numbers_exists_l44_44013


namespace value_of_m_minus_n_l44_44101

theorem value_of_m_minus_n (m n : ℝ) (i : ℂ) (h1 : i * i = -1) (h2 : (m : ℂ) / (1 + i) = 1 - n * i) : m - n = 1 :=
sorry

end value_of_m_minus_n_l44_44101


namespace photographer_choice_l44_44055

theorem photographer_choice : 
  (Nat.choose 7 4) + (Nat.choose 7 5) = 56 := 
by 
  sorry

end photographer_choice_l44_44055


namespace solve_for_q_l44_44069

-- Define the conditions
variables (p q : ℝ)
axiom condition1 : 3 * p + 4 * q = 8
axiom condition2 : 4 * p + 3 * q = 13

-- State the goal to prove q = -1
theorem solve_for_q : q = -1 :=
by
  sorry

end solve_for_q_l44_44069


namespace min_jumps_to_visit_all_points_and_return_l44_44854

theorem min_jumps_to_visit_all_points_and_return (n : ℕ) (h : n = 2016) : 
  ∀ jumps : ℕ, (∀ p : Fin n, ∃ k : ℕ, p = (2 * k) % n ∨ p = (3 * k) % n) → 
  jumps = 2017 :=
by 
  intros jumps h
  sorry

end min_jumps_to_visit_all_points_and_return_l44_44854


namespace ants_harvest_time_l44_44759

theorem ants_harvest_time :
  ∃ h : ℕ, (∀ h : ℕ, 24 - 4 * h = 12) ∧ h = 3 := sorry

end ants_harvest_time_l44_44759


namespace max_three_cell_corners_l44_44269

-- Define the grid size
def grid_height : ℕ := 7
def grid_width : ℕ := 14

-- Define the concept of a three-cell corner removal
def three_cell_corner (region : ℕ) : ℕ := region / 3

-- Define the problem statement in Lean
theorem max_three_cell_corners : three_cell_corner (grid_height * grid_width) = 32 := by
  sorry

end max_three_cell_corners_l44_44269


namespace phillip_remaining_money_l44_44231

def initial_money : ℝ := 95
def cost_oranges : ℝ := 14
def cost_apples : ℝ := 25
def cost_candy : ℝ := 6
def cost_eggs : ℝ := 12
def cost_milk : ℝ := 8
def discount_apples_rate : ℝ := 0.15
def discount_milk_rate : ℝ := 0.10

def discounted_cost_apples : ℝ := cost_apples * (1 - discount_apples_rate)
def discounted_cost_milk : ℝ := cost_milk * (1 - discount_milk_rate)

def total_spent : ℝ := cost_oranges + discounted_cost_apples + cost_candy + cost_eggs + discounted_cost_milk

def remaining_money : ℝ := initial_money - total_spent

theorem phillip_remaining_money : remaining_money = 34.55 := by
  -- Proof here
  sorry

end phillip_remaining_money_l44_44231


namespace sea_creatures_lost_l44_44527

theorem sea_creatures_lost (sea_stars seashells snails items_left : ℕ) 
  (h1 : sea_stars = 34) 
  (h2 : seashells = 21) 
  (h3 : snails = 29) 
  (h4 : items_left = 59) : 
  sea_stars + seashells + snails - items_left = 25 :=
by
  sorry

end sea_creatures_lost_l44_44527


namespace pete_books_ratio_l44_44349

theorem pete_books_ratio 
  (M_last : ℝ) (P_last P_this_year M_this_year : ℝ)
  (h1 : P_last = 2 * M_last)
  (h2 : M_this_year = 1.5 * M_last)
  (h3 : P_last + P_this_year = 300)
  (h4 : M_this_year = 75) :
  P_this_year / P_last = 2 :=
by
  sorry

end pete_books_ratio_l44_44349


namespace problem_statement_l44_44229

open Set

-- Definitions based on the problem's conditions
def U : Set ℕ := { x | 0 < x ∧ x ≤ 8 }
def S : Set ℕ := {1, 2, 4, 5}
def T : Set ℕ := {3, 5, 7}
def complement_U_T : Set ℕ := U \ T

-- The Lean 4 statement to prove
theorem problem_statement : S ∩ complement_U_T = {1, 2, 4} :=
by sorry

end problem_statement_l44_44229


namespace sum_of_ages_five_years_from_now_l44_44015

noncomputable def viggo_age_when_brother_was_2 (brother_age: ℕ) : ℕ :=
  10 + 2 * brother_age

noncomputable def current_viggo_age (viggo_age_at_2: ℕ) (current_brother_age: ℕ) : ℕ :=
  viggo_age_at_2 + (current_brother_age - 2)

def sister_age (viggo_age: ℕ) : ℕ :=
  viggo_age + 5

noncomputable def cousin_age (viggo_age: ℕ) (brother_age: ℕ) (sister_age: ℕ) : ℕ :=
  ((viggo_age + brother_age + sister_age) / 3)

noncomputable def future_ages_sum (viggo_age: ℕ) (brother_age: ℕ) (sister_age: ℕ) (cousin_age: ℕ) : ℕ :=
  viggo_age + 5 + brother_age + 5 + sister_age + 5 + cousin_age + 5

theorem sum_of_ages_five_years_from_now :
  let current_brother_age := 10
  let viggo_age_at_2 := viggo_age_when_brother_was_2 2
  let current_viggo_age := current_viggo_age viggo_age_at_2 current_brother_age
  let current_sister_age := sister_age current_viggo_age
  let current_cousin_age := cousin_age current_viggo_age current_brother_age current_sister_age
  future_ages_sum current_viggo_age current_brother_age current_sister_age current_cousin_age = 99 := sorry

end sum_of_ages_five_years_from_now_l44_44015


namespace sufficient_condition_for_proposition_l44_44964

theorem sufficient_condition_for_proposition (a : ℝ) : 
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 - a ≤ 0) ↔ (a ≥ 5) :=
by 
  sorry

end sufficient_condition_for_proposition_l44_44964


namespace speed_of_first_car_l44_44341

theorem speed_of_first_car (v : ℝ) 
  (h1 : ∀ v, v > 0 → (first_speed = 1.25 * v))
  (h2 : 720 = (v + 1.25 * v) * 4) : 
  first_speed = 100 := 
by
  sorry

end speed_of_first_car_l44_44341


namespace proved_problem_l44_44837

theorem proved_problem (x y p n k : ℕ) (h_eq : x^n + y^n = p^k)
  (h1 : n > 1)
  (h2 : n % 2 = 1)
  (h3 : Nat.Prime p)
  (h4 : p % 2 = 1) :
  ∃ l : ℕ, n = p^l :=
by sorry

end proved_problem_l44_44837


namespace first_year_after_2022_with_digit_sum_5_l44_44289

def sum_of_digits (n : ℕ) : ℕ :=
  (toString n).foldl (λ acc c => acc + c.toNat - '0'.toNat) 0

theorem first_year_after_2022_with_digit_sum_5 :
  ∃ y : ℕ, y > 2022 ∧ sum_of_digits y = 5 ∧ ∀ z : ℕ, z > 2022 ∧ z < y → sum_of_digits z ≠ 5 :=
sorry

end first_year_after_2022_with_digit_sum_5_l44_44289


namespace average_weight_of_arun_l44_44390

variable (weight : ℝ)

def arun_constraint := 61 < weight ∧ weight < 72
def brother_constraint := 60 < weight ∧ weight < 70
def mother_constraint := weight ≤ 64
def father_constraint := 62 < weight ∧ weight < 73
def sister_constraint := 59 < weight ∧ weight < 68

theorem average_weight_of_arun : 
  (∃ w : ℝ, arun_constraint w ∧ brother_constraint w ∧ mother_constraint w ∧ father_constraint w ∧ sister_constraint w) →
  (63.5 = (63 + 64) / 2) := 
by
  sorry

end average_weight_of_arun_l44_44390


namespace k_value_for_root_multiplicity_l44_44825

theorem k_value_for_root_multiplicity (k : ℝ) :
  (∃ x : ℝ, (x - 1) / (x - 3) = k / (x - 3) ∧ (x-3 = 0)) → k = 2 :=
by
  sorry

end k_value_for_root_multiplicity_l44_44825


namespace peanut_butter_candy_count_l44_44746

-- Definitions derived from the conditions
def grape_candy (banana_candy : ℕ) := banana_candy + 5
def peanut_butter_candy (grape_candy : ℕ) := 4 * grape_candy

-- Given condition for the banana jar
def banana_candy := 43

-- The main theorem statement
theorem peanut_butter_candy_count : peanut_butter_candy (grape_candy banana_candy) = 192 :=
by
  sorry

end peanut_butter_candy_count_l44_44746


namespace derivative_of_y_l44_44240

noncomputable def y (x : ℝ) : ℝ :=
  1/2 * Real.tanh x + 1/(4 * Real.sqrt 2) * Real.log ((1 + Real.sqrt 2 * Real.tanh x) / (1 - Real.sqrt 2 * Real.tanh x))

theorem derivative_of_y (x : ℝ) : 
  (deriv y x) = 1/(Real.cosh x ^ 2 * (1 - Real.sinh x ^ 2)) := 
by
  sorry

end derivative_of_y_l44_44240


namespace complete_residue_system_infinitely_many_positive_integers_l44_44065

def is_complete_residue_system (n m : ℕ) : Prop :=
  ∀ i j, 1 ≤ i ∧ i ≤ m → 1 ≤ j ∧ j ≤ m → i ≠ j → (i^n % m ≠ j^n % m)

theorem complete_residue_system_infinitely_many_positive_integers (m : ℕ) (h_pos : 0 < m) :
  ∃ᶠ n in at_top, is_complete_residue_system n m :=
sorry

end complete_residue_system_infinitely_many_positive_integers_l44_44065


namespace num_prime_numbers_with_units_digit_7_num_prime_numbers_less_than_100_with_units_digit_7_l44_44422

def is_prime (n : ℕ) : Prop := Nat.Prime n
def ends_with_7 (n : ℕ) : Prop := n % 10 = 7

theorem num_prime_numbers_with_units_digit_7 (n : ℕ) (h1 : n < 100) (h2 : ends_with_7 n) : is_prime n :=
by sorry

theorem num_prime_numbers_less_than_100_with_units_digit_7 : 
  ∃ (l : List ℕ), (∀ x ∈ l, x < 100 ∧ ends_with_7 x ∧ is_prime x) ∧ l.length = 6 :=
by sorry

end num_prime_numbers_with_units_digit_7_num_prime_numbers_less_than_100_with_units_digit_7_l44_44422


namespace inequality_solution_sets_l44_44641

noncomputable def solve_inequality (m : ℝ) : Set ℝ :=
  if m = 0 then Set.Iic (-2)
  else if m > 0 then Set.Iic (-2) ∪ Set.Ici (1 / m)
  else if (-(1/2) < m ∧ m < 0) then Set.Ioo (1 / m) (-2)
  else if m = -(1 / 2) then ∅
  else Set.Ioo (-2) (1 / m)

theorem inequality_solution_sets (m : ℝ) :
  solve_inequality m = 
    if m = 0 then Set.Iic (-2)
    else if m > 0 then Set.Iic (-2) ∪ Set.Ici (1 / m)
    else if (-(1/2) < m ∧ m < 0) then Set.Ioo (1 / m) (-2)
    else if m = -(1 / 2) then ∅
    else Set.Ioo (-2) (1 / m) :=
sorry

end inequality_solution_sets_l44_44641


namespace center_of_circle_in_second_quadrant_l44_44875

theorem center_of_circle_in_second_quadrant (a b : ℝ) 
  (h1 : a < 0) 
  (h2 : b > 0) : 
  ∃ (q : ℕ), q = 2 := 
by 
  sorry

end center_of_circle_in_second_quadrant_l44_44875


namespace backpack_pencil_case_combinations_l44_44158

theorem backpack_pencil_case_combinations (backpacks pencil_cases : Fin 2) : 
  (backpacks * pencil_cases) = 4 :=
by 
  sorry

end backpack_pencil_case_combinations_l44_44158


namespace specific_heat_capacity_l44_44440

variable {k x p S V α ν R μ : Real}
variable (p x V α : Real) (hp : p = α * V)
variable (hk : k * x = p * S)
variable (hα : α = k / (S^2))

theorem specific_heat_capacity 
  (hk : k * x = p * S) 
  (hp : p = α * V)
  (hα : α = k / (S^2)) 
  (hR : R > 0) 
  (hν : ν > 0) 
  (hμ : μ > 0)
  : (2 * R / μ) = 4155 := 
sorry

end specific_heat_capacity_l44_44440


namespace inequality_abc_l44_44940

theorem inequality_abc (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (1/a) + (1/b) ≥ 4/(a + b) :=
by
  sorry

end inequality_abc_l44_44940


namespace proof_problem_l44_44328

open Real

def p : Prop := ∀ a : ℝ, a^2017 > -1 → a > -1
def q : Prop := ∀ x : ℝ, x^2 * tan (x^2) > 0

theorem proof_problem : p ∨ q :=
sorry

end proof_problem_l44_44328


namespace remainder_of_7_pow_51_mod_8_l44_44060

theorem remainder_of_7_pow_51_mod_8 : (7^51 % 8) = 7 := sorry

end remainder_of_7_pow_51_mod_8_l44_44060


namespace gcd_of_powers_of_three_l44_44540

theorem gcd_of_powers_of_three :
  let a := 3^1001 - 1
  let b := 3^1012 - 1
  gcd a b = 177146 := by
  sorry

end gcd_of_powers_of_three_l44_44540


namespace blossom_room_area_l44_44801

theorem blossom_room_area
  (ft_to_in : ℕ)
  (length_ft : ℕ)
  (width_ft : ℕ)
  (ft_to_in_def : ft_to_in = 12)
  (length_width_def : length_ft = 10)
  (room_square : length_ft = width_ft) :
  (length_ft * ft_to_in) * (width_ft * ft_to_in) = 14400 := 
by
  -- ft_to_in is the conversion factor from feet to inches
  -- length_ft and width_ft are both 10 according to length_width_def and room_square
  -- So, we have (10 * 12) * (10 * 12) = 14400
  sorry

end blossom_room_area_l44_44801


namespace lucky_lucy_l44_44510

theorem lucky_lucy (a b c d e : ℤ)
  (ha : a = 2)
  (hb : b = 4)
  (hc : c = 6)
  (hd : d = 8)
  (he : a + b - c + d - e = a + (b - (c + (d - e)))) :
  e = 8 :=
by
  rw [ha, hb, hc, hd] at he
  exact eq_of_sub_eq_zero (by linarith)

end lucky_lucy_l44_44510


namespace sally_bread_consumption_l44_44617

/-
Given:
    1) Sally eats 2 sandwiches on Saturday
    2) Sally eats 1 sandwich on Sunday
    3) Each sandwich uses 2 pieces of bread

Prove:
    Sally eats 6 pieces of bread across Saturday and Sunday
-/
theorem sally_bread_consumption (sandwiches_sat : Nat) (sandwiches_sun : Nat) (bread_per_sandwich : Nat)
    (H1 : sandwiches_sat = 2) (H2 : sandwiches_sun = 1) (H3 : bread_per_sandwich = 2) :
    2 * bread_per_sandwich + 1 * bread_per_sandwich = 6 := by
  sorry

end sally_bread_consumption_l44_44617


namespace find_fraction_value_l44_44325

variable (a b : ℝ)
variable (h1 : b > a)
variable (h2 : a > 0)
variable (h3 : a / b + b / a = 4)

theorem find_fraction_value (a b : ℝ) (h1 : b > a) (h2 : a > 0) (h3 : a / b + b / a = 4) : (a + b) / (a - b) = Real.sqrt 3 := by
  sorry

end find_fraction_value_l44_44325


namespace price_reduction_equation_l44_44607

theorem price_reduction_equation (x : ℝ) : 25 * (1 - x)^2 = 16 :=
by
  sorry

end price_reduction_equation_l44_44607


namespace prob_of_B1_selected_prob_of_D1_in_team_l44_44290

noncomputable def total_teams : ℕ := 20

noncomputable def teams_with_B1 : ℕ := 8

noncomputable def teams_with_D1 : ℕ := 12

theorem prob_of_B1_selected : (teams_with_B1 : ℚ) / total_teams = 2 / 5 := by
  sorry

theorem prob_of_D1_in_team : (teams_with_D1 : ℚ) / total_teams = 3 / 5 := by
  sorry

end prob_of_B1_selected_prob_of_D1_in_team_l44_44290


namespace stickers_difference_l44_44311

theorem stickers_difference (X : ℕ) :
  let Cindy_initial := X
  let Dan_initial := X
  let Cindy_after := Cindy_initial - 15
  let Dan_after := Dan_initial + 18
  Dan_after - Cindy_after = 33 := by
  sorry

end stickers_difference_l44_44311


namespace sum_when_max_power_less_500_l44_44256

theorem sum_when_max_power_less_500 :
  ∃ (a b : ℕ), a > 0 ∧ b > 1 ∧ a^b < 500 ∧
  (∀ (a' b' : ℕ), a' > 0 → b' > 1 → a'^b' < 500 → a^b ≥ a'^b') ∧ (a + b = 24) :=
  sorry

end sum_when_max_power_less_500_l44_44256


namespace number_of_ordered_pairs_l44_44496

theorem number_of_ordered_pairs :
  ∃ n : ℕ, n = 89 ∧ (∃ (x y : ℕ), 0 < x ∧ 0 < y ∧ x < y ∧ 2 * x * y = 8 ^ 30 * (x + y)) := sorry

end number_of_ordered_pairs_l44_44496


namespace area_inside_C_outside_A_B_l44_44287

/-- Define the radii of circles A, B, and C --/
def radius_A : ℝ := 1
def radius_B : ℝ := 1
def radius_C : ℝ := 2

/-- Define the condition of tangency and overlap --/
def circles_tangent_at_one_point (r1 r2 : ℝ) : Prop :=
  r1 = r2 

def circle_C_tangent_to_A_B (rA rB rC : ℝ) : Prop :=
  rA = 1 ∧ rB = 1 ∧ rC = 2 ∧ circles_tangent_at_one_point rA rB

/-- Statement to be proved: The area inside circle C but outside circles A and B is 2π --/
theorem area_inside_C_outside_A_B (h : circle_C_tangent_to_A_B radius_A radius_B radius_C) : 
  π * radius_C^2 - π * (radius_A^2 + radius_B^2) = 2 * π :=
by
  sorry

end area_inside_C_outside_A_B_l44_44287


namespace even_function_f_l44_44812

noncomputable def f (a b c x : ℝ) := a * Real.cos x + b * x^2 + c

theorem even_function_f (a b c : ℝ) (h1 : f a b c 1 = 1) : f a b c (-1) = f a b c 1 := by
  sorry

end even_function_f_l44_44812


namespace division_of_decimals_l44_44916

theorem division_of_decimals : (0.05 / 0.002) = 25 :=
by
  -- Proof will be filled here
  sorry

end division_of_decimals_l44_44916


namespace fraction_is_determined_l44_44987

theorem fraction_is_determined (y x : ℕ) (h1 : y * 3 = x - 1) (h2 : (y + 4) * 2 = x) : 
  y = 7 ∧ x = 22 :=
by
  sorry

end fraction_is_determined_l44_44987


namespace total_oranges_l44_44024

def oranges_from_first_tree : Nat := 80
def oranges_from_second_tree : Nat := 60
def oranges_from_third_tree : Nat := 120

theorem total_oranges : oranges_from_first_tree + oranges_from_second_tree + oranges_from_third_tree = 260 :=
by
  sorry

end total_oranges_l44_44024


namespace ratio_of_areas_l44_44186

theorem ratio_of_areas (x y l : ℝ)
  (h1 : 2 * (x + 3 * y) = 2 * (l + y))
  (h2 : 2 * x + l = 3 * y) :
  (x * 3 * y) / (l * y) = 3 / 7 :=
by
  -- Proof will be provided here
  sorry

end ratio_of_areas_l44_44186


namespace calculate_expression_l44_44933

theorem calculate_expression : 287 * 287 + 269 * 269 - (2 * 287 * 269) = 324 :=
by
  sorry

end calculate_expression_l44_44933


namespace number_of_unlocked_cells_l44_44215

-- Establish the conditions from the problem description.
def total_cells : ℕ := 2004

-- Helper function to determine if a number is a perfect square.
def is_perfect_square (n : ℕ) : Prop :=
  ∃ (k : ℕ), k * k = n

-- Counting the number of perfect squares in the range from 1 to total_cells.
def perfect_squares_up_to (n : ℕ) : ℕ :=
  (Nat.sqrt n)

-- The theorem that needs to be proved.
theorem number_of_unlocked_cells : perfect_squares_up_to total_cells = 44 :=
by
  sorry

end number_of_unlocked_cells_l44_44215


namespace base_length_of_parallelogram_l44_44822

theorem base_length_of_parallelogram (A : ℕ) (H : ℕ) (Base : ℕ) (hA : A = 576) (hH : H = 48) (hArea : A = Base * H) : 
  Base = 12 := 
by 
  -- We skip the proof steps since we only need to provide the Lean theorem statement.
  sorry

end base_length_of_parallelogram_l44_44822


namespace consecutive_integers_divisor_l44_44458

theorem consecutive_integers_divisor {m n : ℕ} (hm : m < n) (a : ℕ) :
  ∃ i j : ℕ, i ≠ j ∧ (a + i) * (a + j) % (m * n) = 0 :=
by
  sorry

end consecutive_integers_divisor_l44_44458


namespace regular_polygon_num_sides_l44_44664

theorem regular_polygon_num_sides (angle : ℝ) (h : angle = 45) : 
  (∃ n : ℕ, n = 360 / angle ∧ n ≠ 0) → n = 8 :=
by
  sorry

end regular_polygon_num_sides_l44_44664


namespace horner_method_complexity_l44_44574

variable {α : Type*} [Field α]

/-- Evaluating a polynomial of degree n using Horner's method requires exactly n multiplications
    and n additions, and 0 exponentiations.  -/
theorem horner_method_complexity (n : ℕ) (a : Fin (n + 1) → α) (x₀ : α) :
  ∃ (muls adds exps : ℕ), 
    (muls = n) ∧ (adds = n) ∧ (exps = 0) :=
by
  sorry

end horner_method_complexity_l44_44574


namespace triangle_area_ab_l44_44840

theorem triangle_area_ab (a b : ℝ) (ha : 0 < a) (hb : 0 < b) 
    (hline : ∀ x y : ℝ, 2 * a * x + 3 * b * y = 12) (harea : (1/2) * (6 / a) * (4 / b) = 9) : 
    a * b = 4 / 3 :=
by 
  sorry

end triangle_area_ab_l44_44840


namespace sum_first_15_odd_integers_l44_44586

theorem sum_first_15_odd_integers : 
  let a := 1
  let n := 15
  let d := 2
  let l := a + (n-1) * d
  let S := n / 2 * (a + l)
  S = 225 :=
by
  sorry

end sum_first_15_odd_integers_l44_44586


namespace shiela_bottles_l44_44856

theorem shiela_bottles (num_stars : ℕ) (stars_per_bottle : ℕ) (num_bottles : ℕ) 
  (h1 : num_stars = 45) (h2 : stars_per_bottle = 5) : num_bottles = 9 :=
sorry

end shiela_bottles_l44_44856


namespace kicks_before_break_l44_44288

def total_kicks : ℕ := 98
def kicks_after_break : ℕ := 36
def kicks_needed_to_goal : ℕ := 19

theorem kicks_before_break :
  total_kicks - (kicks_after_break + kicks_needed_to_goal) = 43 := 
by
  -- proof wanted
  sorry

end kicks_before_break_l44_44288


namespace even_function_f_l44_44295

noncomputable def f (x : ℝ) : ℝ :=
if x < 0 then 2^x - 1 else sorry

theorem even_function_f (h_even : ∀ x : ℝ, f x = f (-x)) : f 1 = -1 / 2 := by
  -- proof development skipped
  sorry

end even_function_f_l44_44295


namespace log_x_inequality_l44_44629

noncomputable def log_x_over_x (x : ℝ) := (Real.log x) / x

theorem log_x_inequality {x : ℝ} (h1 : 1 < x) (h2 : x < 2) : 
  (log_x_over_x x) ^ 2 < log_x_over_x x ∧ log_x_over_x x < log_x_over_x (x * x) :=
by
  sorry

end log_x_inequality_l44_44629


namespace find_linear_function_l44_44554

theorem find_linear_function (α : ℝ) (hα : α > 0)
  (f : ℕ+ → ℝ)
  (h : ∀ (k m : ℕ+), α * (m : ℝ) ≤ (k : ℝ) ∧ (k : ℝ) < (α + 1) * (m : ℝ) → f (k + m) = f k + f m)
: ∃ (b : ℝ), ∀ (n : ℕ+), f n = b * (n : ℝ) :=
sorry

end find_linear_function_l44_44554


namespace arccos_sin_eq_l44_44150

open Real

-- Definitions from the problem conditions
noncomputable def radians := π / 180

-- The theorem we need to prove
theorem arccos_sin_eq : arccos (sin 3) = 3 - (π / 2) :=
by
  sorry

end arccos_sin_eq_l44_44150


namespace translated_upwards_2_units_l44_44958

theorem translated_upwards_2_units (x : ℝ) : (x + 2 > 0) → (x > -2) :=
by 
  intros h
  exact sorry

end translated_upwards_2_units_l44_44958


namespace chicken_nuggets_order_l44_44952

theorem chicken_nuggets_order (cost_per_box : ℕ) (nuggets_per_box : ℕ) (total_amount_paid : ℕ) 
  (h1 : cost_per_box = 4) (h2 : nuggets_per_box = 20) (h3 : total_amount_paid = 20) : 
  total_amount_paid / cost_per_box * nuggets_per_box = 100 :=
by
  -- This is where the proof would go
  sorry

end chicken_nuggets_order_l44_44952


namespace dark_squares_exceed_light_squares_by_one_l44_44531

theorem dark_squares_exceed_light_squares_by_one 
  (m n : ℕ) (h_m : m = 9) (h_n : n = 9) (h_total_squares : m * n = 81) :
  let dark_squares := 5 * 5 + 4 * 4
  let light_squares := 5 * 4 + 4 * 5
  dark_squares - light_squares = 1 :=
by {
  sorry
}

end dark_squares_exceed_light_squares_by_one_l44_44531


namespace probability_X1_lt_X2_lt_X3_is_1_6_l44_44029

noncomputable def probability_X1_lt_X2_lt_X3 (n : ℕ) (h : n ≥ 3) : ℚ :=
if h : n ≥ 3 then
  1/6
else
  0

theorem probability_X1_lt_X2_lt_X3_is_1_6 (n : ℕ) (h : n ≥ 3) :
  probability_X1_lt_X2_lt_X3 n h = 1/6 :=
sorry

end probability_X1_lt_X2_lt_X3_is_1_6_l44_44029


namespace total_legs_walking_on_ground_l44_44367

def horses : ℕ := 16
def men : ℕ := 16

def men_walking := men / 2
def men_riding := men / 2

def legs_per_man := 2
def legs_per_horse := 4

def legs_for_men_walking := men_walking * legs_per_man
def legs_for_horses := horses * legs_per_horse

theorem total_legs_walking_on_ground : legs_for_men_walking + legs_for_horses = 80 := 
by
  sorry

end total_legs_walking_on_ground_l44_44367


namespace smaller_angle_at_3_20_correct_l44_44796

noncomputable def smaller_angle_at_3_20 : Float :=
  let degrees_per_minute_for_minute_hand := 360 / 60
  let degrees_per_minute_for_hour_hand := 360 / (60 * 12)
  let initial_hour_hand_position := 90.0  -- 3 o'clock position
  let minute_past_three := 20
  let minute_hand_movement := minute_past_three * degrees_per_minute_for_minute_hand
  let hour_hand_movement := minute_past_three * degrees_per_minute_for_hour_hand
  let current_hour_hand_position := initial_hour_hand_position + hour_hand_movement
  let angle_between_hands := minute_hand_movement - current_hour_hand_position
  if angle_between_hands < 0 then
    -angle_between_hands
  else
    angle_between_hands

theorem smaller_angle_at_3_20_correct : smaller_angle_at_3_20 = 20.0 := by
  sorry

end smaller_angle_at_3_20_correct_l44_44796


namespace complete_square_eq_l44_44749

theorem complete_square_eq (b c : ℤ) (h : ∃ b c : ℤ, (∀ x : ℝ, (x - 5)^2 = b * x + c) ∧ b + c = 5) :
  b + c = 5 :=
sorry

end complete_square_eq_l44_44749


namespace rational_square_l44_44063

theorem rational_square (a b c : ℚ) (h : a ≠ b ∧ b ≠ c ∧ c ≠ a) : ∃ r : ℚ, (1 / (a - b)^2) + (1 / (b - c)^2) + (1 / (c - a)^2) = r^2 := 
by 
  sorry

end rational_square_l44_44063


namespace min_side_length_l44_44768

noncomputable def side_length_min : ℝ := 30

theorem min_side_length (s r : ℝ) (hs₁ : s^2 ≥ 900) (hr₁ : π * r^2 ≥ 100) (hr₂ : 2 * r ≤ s) :
  s ≥ side_length_min :=
by
  sorry

end min_side_length_l44_44768


namespace f_at_neg_one_l44_44399

def f (x : ℝ) : ℝ := sorry

theorem f_at_neg_one (h : ∀ x : ℝ, f (x - 1) = x^2 + 1) : f (-1) = 1 :=
by sorry

end f_at_neg_one_l44_44399


namespace base_problem_l44_44536

theorem base_problem (c d : Nat) (pos_c : c > 0) (pos_d : d > 0) (h : 5 * c + 8 = 8 * d + 5) : c + d = 15 :=
sorry

end base_problem_l44_44536


namespace buratino_loss_l44_44913

def buratino_dollars_lost (x y : ℕ) : ℕ := 5 * y - 3 * x

theorem buratino_loss :
  ∃ (x y : ℕ), x + y = 50 ∧ 3 * y - 2 * x = 0 ∧ buratino_dollars_lost x y = 10 :=
by {
  sorry
}

end buratino_loss_l44_44913


namespace apples_difference_l44_44386

theorem apples_difference 
  (father_apples : ℕ := 8)
  (mother_apples : ℕ := 13)
  (jungkook_apples : ℕ := 7)
  (brother_apples : ℕ := 5) :
  max father_apples (max mother_apples (max jungkook_apples brother_apples)) - 
  min father_apples (min mother_apples (min jungkook_apples brother_apples)) = 8 :=
by
  sorry

end apples_difference_l44_44386


namespace find_linear_function_and_unit_price_l44_44733

def linear_function (k b x : ℝ) : ℝ := k * x + b

def profit (cost_price : ℝ) (selling_price : ℝ) (sales_volume : ℝ) : ℝ := 
  (selling_price - cost_price) * sales_volume

theorem find_linear_function_and_unit_price
  (x1 y1 x2 y2 x3 y3 : ℝ)
  (h1 : x1 = 20) (h2 : y1 = 200)
  (h3 : x2 = 25) (h4 : y2 = 150)
  (h5 : x3 = 30) (h6 : y3 = 100)
  (cost_price := 10) (desired_profit := 2160) :
  ∃ k b x : ℝ, 
    (linear_function k b x1 = y1) ∧ 
    (linear_function k b x2 = y2) ∧ 
    (profit cost_price x (linear_function k b x) = desired_profit) ∧ 
    (linear_function k b x = -10 * x + 400) ∧ 
    (x = 22) :=
by
  sorry

end find_linear_function_and_unit_price_l44_44733


namespace max_streetlights_l44_44198

theorem max_streetlights {road_length streetlight_length : ℝ} 
  (h1 : road_length = 1000)
  (h2 : streetlight_length = 1)
  (fully_illuminated : ∀ (n : ℕ), (n * streetlight_length) < road_length)
  : ∃ max_n, max_n = 1998 ∧ (∀ n, n > max_n → (∃ i, streetlight_length * i > road_length)) :=
sorry

end max_streetlights_l44_44198


namespace no_quadruples_sum_2013_l44_44620

theorem no_quadruples_sum_2013 :
  ¬ ∃ (a b c d : ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
  a + b + c + d = 2013 ∧
  2013 % a = 0 ∧ 2013 % b = 0 ∧ 2013 % c = 0 ∧ 2013 % d = 0 :=
by
  sorry

end no_quadruples_sum_2013_l44_44620


namespace complement_U_A_l44_44784

-- Define the universal set U and the subset A
def U : Set ℕ := {1, 2, 3, 4}
def A : Set ℕ := {1, 3, 4}

-- Define the complement of A relative to the universal set U
def complement (U A : Set ℕ) : Set ℕ := { x | x ∈ U ∧ x ∉ A }

-- The theorem we want to prove
theorem complement_U_A : complement U A = {2} := by
  sorry

end complement_U_A_l44_44784


namespace steve_popsicle_sticks_l44_44959

theorem steve_popsicle_sticks (S Sid Sam : ℕ) (h1 : Sid = 2 * S) (h2 : Sam = 3 * Sid) (h3 : S + Sid + Sam = 108) : S = 12 :=
by
  sorry

end steve_popsicle_sticks_l44_44959


namespace original_price_of_article_l44_44460

theorem original_price_of_article
  (P S : ℝ) 
  (h1 : S = 1.4 * P) 
  (h2 : S - P = 560) 
  : P = 1400 :=
by
  sorry

end original_price_of_article_l44_44460


namespace cone_base_circumference_l44_44453

theorem cone_base_circumference (V : ℝ) (h : ℝ) (r : ℝ) (C : ℝ) 
  (hV : V = 18 * Real.pi)
  (hh : h = 6) 
  (hV_cone : V = (1/3) * Real.pi * r^2 * h) :
  C = 2 * Real.pi * r → C = 6 * Real.pi :=
by 
  -- We assume as conditions are only mentioned
  sorry

end cone_base_circumference_l44_44453


namespace proof_of_area_weighted_sum_of_distances_l44_44317

def area_weighted_sum_of_distances
  (a b a1 a2 a3 a4 b1 b2 b3 b4 : ℝ) 
  (t1 t2 t3 t4 t : ℝ)
  (z1 z2 z3 z4 z : ℝ) 
  (h1 : t1 = a1 * b1)
  (h2 : t2 = (a - a1) * b1)
  (h3 : t3 = a3 * b3)
  (h4 : t4 = (a - a3) * b3)
  (rect_area : t = a * b)
  : Prop :=
  t1 * z1 + t2 * z2 + t3 * z3 + t4 * z4 = t * z

theorem proof_of_area_weighted_sum_of_distances
  (a b a1 a2 a3 a4 b1 b2 b3 b4 : ℝ)
  (t1 t2 t3 t4 t : ℝ)
  (z1 z2 z3 z4 z : ℝ)
  (h1 : t1 = a1 * b1)
  (h2 : t2 = (a - a1) * b1)
  (h3 : t3 = a3 * b3)
  (h4 : t4 = (a - a3) * b3)
  (rect_area : t = a * b)
  : area_weighted_sum_of_distances a b a1 a2 a3 a4 b1 b2 b3 b4 t1 t2 t3 t4 t z1 z2 z3 z4 z h1 h2 h3 h4 rect_area :=
  sorry

end proof_of_area_weighted_sum_of_distances_l44_44317


namespace hyperbola_and_line_properties_l44_44927

open Real

def hyperbola (x y : ℝ) (a b : ℝ) : Prop := (x^2 / a^2) - (y^2 / b^2) = 1
def asymptote1 (x y : ℝ) : Prop := y = sqrt 3 * x
def asymptote2 (x y : ℝ) : Prop := y = -sqrt 3 * x
def line (x y t : ℝ) : Prop := y = x + t

theorem hyperbola_and_line_properties :
  ∃ a b t : ℝ,
  a > 0 ∧ b > 0 ∧ a = 1 ∧ b^2 = 3 ∧
  (∀ x y, hyperbola x y a b ↔ (x^2 - y^2 / 3 = 1)) ∧
  (∀ x y, asymptote1 x y ↔ y = sqrt 3 * x) ∧
  (∀ x y, asymptote2 x y ↔ y = -sqrt 3 * x) ∧
  (∀ x y, (line x y t ↔ (y = x + sqrt 3) ∨ (y = x - sqrt 3))) := sorry

end hyperbola_and_line_properties_l44_44927


namespace parabola_through_point_l44_44760

theorem parabola_through_point (a b : ℝ) (ha : 0 < a) :
  ∃ f : ℝ → ℝ, (∀ x, f x = -a*x^2 + b*x + 1) ∧ f 0 = 1 :=
by
  -- We are given a > 0
  -- We need to show there exists a parabola of the form y = -a*x^2 + b*x + 1 passing through (0,1)
  sorry

end parabola_through_point_l44_44760


namespace find_a_plus_2b_l44_44538

noncomputable def f (a b x : ℝ) : ℝ := a * x^3 + 3 * x^2 - 6 * a * x + b

noncomputable def f' (a b x : ℝ) : ℝ := 3 * a * x^2 + 6 * x - 6 * a

theorem find_a_plus_2b (a b : ℝ) 
  (h1 : f' a b 2 = 0)
  (h2 : f a b 2 = 9) : a + 2 * b = -24 := 
by sorry

end find_a_plus_2b_l44_44538


namespace has_local_maximum_l44_44882

noncomputable def func (x : ℝ) : ℝ := (1 / 3) * x ^ 3 - 4 * x + 4

theorem has_local_maximum :
  ∃ x, x = -2 ∧ func x = 28 / 3 :=
by
  sorry

end has_local_maximum_l44_44882


namespace b_is_geometric_T_sum_l44_44785

noncomputable def a (n : ℕ) : ℝ := 1/2 + (n-1) * (1/2)
noncomputable def S (n : ℕ) : ℝ := n * (1/2) + (n * (n-1) / 2) * (1/2)
noncomputable def b (n : ℕ) : ℝ := 4 ^ (a n)
noncomputable def c (n : ℕ) : ℝ := a n + b n
noncomputable def T (n : ℕ) : ℝ := (n * (n+1) / 4) + 2^(n+1) - 2

theorem b_is_geometric : ∀ n : ℕ, (n > 0) → b (n+1) / b n = 2 := by
  sorry

theorem T_sum : ∀ n : ℕ, T n = (n * (n + 1) / 4) + 2^(n + 1) - 2 := by
  sorry

end b_is_geometric_T_sum_l44_44785


namespace compute_fraction_value_l44_44379

theorem compute_fraction_value : 2 + 3 / (4 + 5 / 6) = 76 / 29 := by
  sorry

end compute_fraction_value_l44_44379


namespace range_of_m_l44_44973

-- Definitions based on conditions
def inequality_holds (m : ℝ) : Prop :=
  ∀ x : ℝ, 0 < x → (m * Real.exp x / x ≥ 6 - 4 * x)

-- The statement to be proved
theorem range_of_m (m : ℝ) : inequality_holds m → m ≥ 2 * Real.exp (-(1 / 2)) :=
by
  sorry

end range_of_m_l44_44973


namespace correct_option_D_l44_44779

variables (a b c : ℤ)

theorem correct_option_D : -2 * a + 3 * (b - 1) = -2 * a + 3 * b - 3 := 
by
  sorry

end correct_option_D_l44_44779


namespace hyperbola_center_l44_44942

theorem hyperbola_center (x y : ℝ) :
    9 * x^2 - 18 * x - 16 * y^2 + 64 * y - 143 = 0 →
    (x, y) = (1, 2) :=
sorry

end hyperbola_center_l44_44942


namespace bucket_full_weight_l44_44636

variable (p q : ℝ)

theorem bucket_full_weight (p q : ℝ) (x y: ℝ) (h1 : x + 3/4 * y = p) (h2 : x + 1/3 * y = q) :
  x + y = (8 * p - 7 * q) / 5 :=
by
  sorry

end bucket_full_weight_l44_44636


namespace ten_faucets_fill_time_l44_44770

theorem ten_faucets_fill_time (rate : ℕ → ℕ → ℝ) (gallons : ℕ) (minutes : ℝ) :
  rate 5 9 = 150 / 5 ∧
  rate 10 135 = 75 / 30 * rate 10 9 / 0.9 * 60 →
  9 * 60 / 30 * 75 / 10 * 60 = 135 :=
sorry

end ten_faucets_fill_time_l44_44770


namespace max_probability_sum_15_l44_44261

-- Context and Definitions based on conditions
def S : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

-- The assertion to be proved:
theorem max_probability_sum_15 (n : ℕ) (h : n ∈ S) :
  n = 7 :=
by
  sorry

end max_probability_sum_15_l44_44261


namespace smallest_M_bound_l44_44072

theorem smallest_M_bound {f : ℕ → ℝ} (hf1 : f 1 = 2) 
  (hf2 : ∀ n : ℕ, f (n + 1) ≥ f n ∧ f n ≥ (n / (n + 1)) * f (2 * n)) : 
  ∃ M : ℕ, (∀ n : ℕ, f n < M) ∧ M = 10 :=
by
  sorry

end smallest_M_bound_l44_44072


namespace minimum_value_expression_l44_44880

theorem minimum_value_expression :
  ∃ x y : ℝ, ∀ x y : ℝ, 3 * x ^ 2 + 4 * x * y + 2 * y ^ 2 - 6 * x + 8 * y + 9 ≥ -10 :=
by
  sorry

end minimum_value_expression_l44_44880


namespace experiment_success_probability_l44_44352

/-- 
There are three boxes, each containing 10 balls. 
- The first box contains 7 balls marked 'A' and 3 balls marked 'B'.
- The second box contains 5 red balls and 5 white balls.
- The third box contains 8 red balls and 2 white balls.

The experiment consists of:
1. Drawing a ball from the first box.
2. If a ball marked 'A' is drawn, drawing from the second box.
3. If a ball marked 'B' is drawn, drawing from the third box.
The experiment is successful if the second ball drawn is red.

Prove that the probability of the experiment being successful is 0.59.
-/
theorem experiment_success_probability (P : ℝ) : 
  P = 0.59 :=
sorry

end experiment_success_probability_l44_44352


namespace Rebecca_group_count_l44_44467

def groupEggs (total_eggs number_of_eggs_per_group total_groups : Nat) : Prop :=
  total_groups = total_eggs / number_of_eggs_per_group

theorem Rebecca_group_count :
  groupEggs 8 2 4 :=
by
  sorry

end Rebecca_group_count_l44_44467


namespace shaded_triangle_area_l44_44463

theorem shaded_triangle_area (b h : ℝ) (hb : b = 2) (hh : h = 3) : 
  (1 / 2 * b * h) = 3 := 
by
  rw [hb, hh]
  norm_num

end shaded_triangle_area_l44_44463


namespace express_y_in_terms_of_x_l44_44621

variable (x y p : ℝ)

-- Conditions
def condition1 := x = 1 + 3^p
def condition2 := y = 1 + 3^(-p)

-- The theorem to be proven
theorem express_y_in_terms_of_x (h1 : condition1 x p) (h2 : condition2 y p) : y = x / (x - 1) :=
sorry

end express_y_in_terms_of_x_l44_44621


namespace find_n_l44_44556

theorem find_n (a1 a2 : ℕ) (s2 s1 : ℕ) (n : ℕ) :
    a1 = 12 →
    a2 = 3 →
    s2 = 3 * s1 →
    ∃ n : ℕ, a1 / (1 - a2/a1) = 16 ∧
             a1 / (1 - (a2 + n) / a1) = s2 →
             n = 6 :=
by
  intros
  sorry

end find_n_l44_44556


namespace final_books_is_correct_l44_44765

def initial_books : ℝ := 35.5
def books_bought : ℝ := 12.3
def books_given_to_friends : ℝ := 7.2
def books_donated : ℝ := 20.8

theorem final_books_is_correct :
  (initial_books + books_bought - books_given_to_friends - books_donated) = 19.8 := by
  sorry

end final_books_is_correct_l44_44765


namespace intersection_of_A_and_B_l44_44451

open Set

variable {α : Type} [PartialOrder α]

noncomputable def A := { x : ℝ | -1 < x ∧ x < 1 }
noncomputable def B := { x : ℝ | 0 < x }

theorem intersection_of_A_and_B :
  A ∩ B = { x : ℝ | 0 < x ∧ x < 1 } :=
by sorry

end intersection_of_A_and_B_l44_44451


namespace value_after_increase_l44_44097

def original_number : ℝ := 400
def percentage_increase : ℝ := 0.20

theorem value_after_increase : original_number * (1 + percentage_increase) = 480 := by
  sorry

end value_after_increase_l44_44097


namespace rabbits_ate_three_potatoes_l44_44656

variable (initial_potatoes remaining_potatoes eaten_potatoes : ℕ)

-- Definitions from the conditions
def mary_initial_potatoes : initial_potatoes = 8 := sorry
def mary_remaining_potatoes : remaining_potatoes = 5 := sorry

-- The goal to prove
theorem rabbits_ate_three_potatoes :
  initial_potatoes - remaining_potatoes = 3 := sorry

end rabbits_ate_three_potatoes_l44_44656


namespace range_of_4a_minus_2b_l44_44908

theorem range_of_4a_minus_2b (a b : ℝ) (h1 : 0 ≤ a - b) (h2 : a - b ≤ 1) (h3 : 2 ≤ a + b) (h4 : a + b ≤ 4) :
  2 ≤ 4 * a - 2 * b ∧ 4 * a - 2 * b ≤ 7 := 
sorry

end range_of_4a_minus_2b_l44_44908


namespace train_speed_kph_l44_44236

-- Definitions based on conditions
def time_seconds : ℕ := 9
def length_meters : ℕ := 135
def conversion_factor : ℕ := 36 -- 3.6 represented as an integer by multiplying both sides by 10

-- The proof statement
theorem train_speed_kph : (length_meters * conversion_factor / 10 / time_seconds = 54) :=
by
  sorry

end train_speed_kph_l44_44236


namespace valid_license_plates_count_l44_44731

def validLicensePlates : Nat :=
  26 * 26 * 26 * 10 * 9 * 8

theorem valid_license_plates_count :
  validLicensePlates = 15818400 :=
by
  sorry

end valid_license_plates_count_l44_44731


namespace erica_total_earnings_l44_44281

def fishPrice : Nat := 20
def pastCatch : Nat := 80
def todayCatch : Nat := 2 * pastCatch
def pastEarnings := pastCatch * fishPrice
def todayEarnings := todayCatch * fishPrice
def totalEarnings := pastEarnings + todayEarnings

theorem erica_total_earnings : totalEarnings = 4800 := by
  sorry

end erica_total_earnings_l44_44281


namespace sum_of_remainders_mod_13_l44_44273

theorem sum_of_remainders_mod_13 :
  ∀ (a b c d e : ℤ),
    a ≡ 3 [ZMOD 13] →
    b ≡ 5 [ZMOD 13] →
    c ≡ 7 [ZMOD 13] →
    d ≡ 9 [ZMOD 13] →
    e ≡ 11 [ZMOD 13] →
    (a + b + c + d + e) % 13 = 9 :=
by
  intros a b c d e ha hb hc hd he
  sorry

end sum_of_remainders_mod_13_l44_44273


namespace algorithm_can_contain_all_structures_l44_44408

def sequential_structure : Prop := sorry
def conditional_structure : Prop := sorry
def loop_structure : Prop := sorry

def algorithm_contains_structure (str : Prop) : Prop := sorry

theorem algorithm_can_contain_all_structures :
  algorithm_contains_structure sequential_structure ∧
  algorithm_contains_structure conditional_structure ∧
  algorithm_contains_structure loop_structure := sorry

end algorithm_can_contain_all_structures_l44_44408


namespace initial_amount_of_money_l44_44241

variable (X : ℕ) -- Initial amount of money Lily had in her account

-- Conditions
def spent_on_shirt : ℕ := 7
def spent_in_second_shop : ℕ := 3 * spent_on_shirt
def remaining_after_purchases : ℕ := 27

-- Proof problem: prove that the initial amount of money X is 55 given the conditions
theorem initial_amount_of_money (h : X - spent_on_shirt - spent_in_second_shop = remaining_after_purchases) : X = 55 :=
by
  -- Placeholder to indicate that steps will be worked out in Lean
  sorry

end initial_amount_of_money_l44_44241


namespace sum_minimal_area_k_l44_44688

def vertices_triangle_min_area (k : ℤ) : Prop :=
  let x1 := 1
  let y1 := 7
  let x2 := 13
  let y2 := 16
  let x3 := 5
  ((y1 - k) * (x2 - x1) ≠ (x1 - x3) * (y2 - y1))

def minimal_area_sum_k : ℤ :=
  9 + 11

theorem sum_minimal_area_k :
  ∃ k1 k2 : ℤ, vertices_triangle_min_area k1 ∧ vertices_triangle_min_area k2 ∧ k1 + k2 = 20 := 
sorry

end sum_minimal_area_k_l44_44688


namespace geometric_seq_fraction_l44_44687

theorem geometric_seq_fraction (a : ℕ → ℝ) (q : ℝ) (h1 : ∀ n, a (n + 1) = q * a n) 
  (h2 : (a 1 + 3 * a 3) / (a 2 + 3 * a 4) = 1 / 2) : 
  (a 4 * a 6 + a 6 * a 8) / (a 6 * a 8 + a 8 * a 10) = 1 / 16 :=
by
  sorry

end geometric_seq_fraction_l44_44687


namespace find_a_l44_44775

theorem find_a (x y a : ℝ) (h1 : x = 2) (h2 : y = 1) (h3 : a * x - y = 3) : a = 2 :=
by
  sorry

end find_a_l44_44775


namespace mosquito_feedings_to_death_l44_44154

theorem mosquito_feedings_to_death 
  (drops_per_feeding : ℕ := 20) 
  (drops_per_liter : ℕ := 5000) 
  (lethal_blood_loss_liters : ℝ := 3) 
  (drops_per_feeding_liters : ℝ := drops_per_feeding / drops_per_liter) 
  (lethal_feedings : ℝ := lethal_blood_loss_liters / drops_per_feeding_liters) :
  lethal_feedings = 750 := 
by
  sorry

end mosquito_feedings_to_death_l44_44154


namespace pitbull_chihuahua_weight_ratio_l44_44967

theorem pitbull_chihuahua_weight_ratio
  (C P G : ℕ)
  (h1 : G = 307)
  (h2 : G = 3 * P + 10)
  (h3 : C + P + G = 439) :
  P / C = 3 :=
by {
  sorry
}

end pitbull_chihuahua_weight_ratio_l44_44967


namespace find_sum_12_terms_of_sequence_l44_44558

variable {a : ℕ → ℕ}

def geometric_sequence (a : ℕ → ℕ) (k : ℕ) : Prop :=
  ∀ n : ℕ, a n * a (n + 1) * a (n + 2) = k

def is_periodic_sequence (a : ℕ → ℕ) (period : ℕ) : Prop :=
  ∀ n : ℕ, a n = a (n + period)

noncomputable def given_sequence : ℕ → ℕ
| 0       => 1
| 1       => 2
| (n + 2) => (given_sequence n * given_sequence (n + 1) / 4) -- This should ensure periodic sequence of period 3 given a common product of 8 and simplifying the product equation.

theorem find_sum_12_terms_of_sequence :
  geometric_sequence given_sequence 8 ∧ given_sequence 0 = 1 ∧ given_sequence 1 = 2 →
  (Finset.range 12).sum given_sequence = 28 :=
by
  sorry

end find_sum_12_terms_of_sequence_l44_44558


namespace union_complement_subset_range_l44_44526

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | 0 < 2 * x + a ∧ 2 * x + a ≤ 3}
def B : Set ℝ := {x | 2 * x ^ 2 - 3 * x - 2 < 0}

-- Define the complement of B
def complement_R (s : Set ℝ) : Set ℝ := {x | x ∉ s}

-- 1. The proof problem for A ∪ (complement of B) when a = 1
theorem union_complement (a : ℝ) (h : a = 1) :
  { x : ℝ | (-1/2 < x ∧ x ≤ 1) ∨ (x ≥ 2 ∨ x ≤ -1/2) } = 
  { x : ℝ | x ≤ 1 ∨ x ≥ 2 } :=
by
  sorry

-- 2. The proof problem for A ⊆ B to find the range of a
theorem subset_range (a : ℝ) :
  (∀ x, A a x → B x) ↔ -1 < a ∧ a ≤ 1 :=
by
  sorry

end union_complement_subset_range_l44_44526


namespace find_multiplier_l44_44283

theorem find_multiplier (A N : ℕ) (h : A = 32) (eqn : N * (A + 4) - 4 * (A - 4) = A) : N = 4 :=
sorry

end find_multiplier_l44_44283


namespace find_h_l44_44906

theorem find_h: 
  ∃ h k, (∀ x, 2 * x ^ 2 + 6 * x + 11 = 2 * (x - h) ^ 2 + k) ∧ h = -3 / 2 :=
by
  sorry

end find_h_l44_44906


namespace march_first_is_sunday_l44_44346

theorem march_first_is_sunday (days_in_march : ℕ) (num_wednesdays : ℕ) (num_saturdays : ℕ) 
  (h1 : days_in_march = 31) (h2 : num_wednesdays = 4) (h3 : num_saturdays = 4) : 
  ∃ d : ℕ, d = 0 := 
by 
  sorry

end march_first_is_sunday_l44_44346


namespace palm_trees_total_l44_44404

theorem palm_trees_total
  (forest_palm_trees : ℕ := 5000)
  (desert_palm_trees : ℕ := forest_palm_trees - (3 * forest_palm_trees / 5)) :
  desert_palm_trees + forest_palm_trees = 7000 :=
by
  sorry

end palm_trees_total_l44_44404


namespace pizza_slices_left_l44_44515

theorem pizza_slices_left (total_slices john_ate : ℕ) 
  (initial_slices : total_slices = 12) 
  (john_slices : john_ate = 3) 
  (sam_ate : ¬¬(2 * john_ate = 6)) : 
  ∃ slices_left, slices_left = 3 :=
by
  sorry

end pizza_slices_left_l44_44515


namespace total_boxes_l44_44668

theorem total_boxes (r_cost y_cost : ℝ) (avg_cost : ℝ) (R Y : ℕ) (hc_r : r_cost = 1.30) (hc_y : y_cost = 2.00) 
                    (hc_avg : avg_cost = 1.72) (hc_R : R = 4) (hc_Y : Y = 4) : 
  R + Y = 8 :=
by
  sorry

end total_boxes_l44_44668


namespace trapezium_parallel_side_length_l44_44022

theorem trapezium_parallel_side_length (a h area x : ℝ) (h1 : a = 20) (h2 : h = 15) (h3 : area = 285) :
  area = 1/2 * (a + x) * h → x = 18 :=
by
  -- placeholder for the proof
  sorry

end trapezium_parallel_side_length_l44_44022


namespace simplify_and_evaluate_l44_44997

theorem simplify_and_evaluate (x y : ℤ) (h1 : x = -1) (h2 : y = -2) :
  ((x + y) ^ 2 - (3 * x - y) * (3 * x + y) - 2 * y ^ 2) / (-2 * x) = -2 :=
by 
  sorry

end simplify_and_evaluate_l44_44997


namespace sqrt_one_over_four_eq_pm_half_l44_44157

theorem sqrt_one_over_four_eq_pm_half : Real.sqrt (1 / 4) = 1 / 2 ∨ Real.sqrt (1 / 4) = - (1 / 2) := by
  sorry

end sqrt_one_over_four_eq_pm_half_l44_44157


namespace factorize_expression_l44_44520

theorem factorize_expression (x : ℝ) :
  (x + 1)^4 + (x + 3)^4 - 272 = 2 * (x^2 + 4*x + 19) * (x + 5) * (x - 1) :=
  sorry

end factorize_expression_l44_44520


namespace mean_value_theorem_for_integrals_l44_44978

variable {a b : ℝ} (f : ℝ → ℝ)

theorem mean_value_theorem_for_integrals (h_cont : ContinuousOn f (Set.Icc a b)) :
  ∃ ξ ∈ Set.Icc a b, ∫ x in a..b, f x = f ξ * (b - a) :=
sorry

end mean_value_theorem_for_integrals_l44_44978


namespace marie_age_l44_44306

theorem marie_age (L M O : ℕ) (h1 : L = 4 * M) (h2 : O = M + 8) (h3 : L = O) : M = 8 / 3 := by
  sorry

end marie_age_l44_44306


namespace average_temperature_l44_44354

theorem average_temperature (T : Fin 5 → ℝ) (h : T = ![52, 67, 55, 59, 48]) :
    (1 / 5) * (T 0 + T 1 + T 2 + T 3 + T 4) = 56.2 := by
  sorry

end average_temperature_l44_44354


namespace equilateral_triangle_area_l44_44327

theorem equilateral_triangle_area (A B C P : ℝ × ℝ)
  (hABC : ∃ a b c : ℝ, a = b ∧ b = c ∧ a = dist A B ∧ b = dist B C ∧ c = dist C A)
  (hPA : dist P A = 10)
  (hPB : dist P B = 8)
  (hPC : dist P C = 12) :
  ∃ (area : ℝ), area = 104 :=
by
  sorry

end equilateral_triangle_area_l44_44327


namespace domain_f_2x_minus_1_l44_44739

theorem domain_f_2x_minus_1 (f : ℝ → ℝ) : 
  (∀ x, 1 ≤ x ∧ x ≤ 2 → 2 ≤ x + 1 ∧ x + 1 ≤ 3) → 
  (∀ z, 2 ≤ 2 * z - 1 ∧ 2 * z - 1 ≤ 3 → ∃ x, 3/2 ≤ x ∧ x ≤ 2 ∧ 2 * x - 1 = z) := 
sorry

end domain_f_2x_minus_1_l44_44739


namespace frog_jumps_10_inches_more_than_grasshopper_frog_jumps_10_inches_farther_than_grasshopper_l44_44266

-- Definitions of conditions
def grasshopper_jump : ℕ := 19
def mouse_jump_frog (frog_jump : ℕ) : ℕ := frog_jump + 20
def mouse_jump_grasshopper : ℕ := grasshopper_jump + 30

-- The proof problem statement
theorem frog_jumps_10_inches_more_than_grasshopper (frog_jump : ℕ) :
  mouse_jump_frog frog_jump = mouse_jump_grasshopper → frog_jump = 29 :=
by
  sorry

-- The ultimate question in the problem
theorem frog_jumps_10_inches_farther_than_grasshopper : 
  (∃ (frog_jump : ℕ), frog_jump = 29) → (frog_jump - grasshopper_jump = 10) :=
by
  sorry

end frog_jumps_10_inches_more_than_grasshopper_frog_jumps_10_inches_farther_than_grasshopper_l44_44266


namespace find_f1_and_f1_l44_44356

theorem find_f1_and_f1' (f : ℝ → ℝ) (f' : ℝ → ℝ) (h_deriv : ∀ x, deriv f x = f' x)
  (h_eq : ∀ x, f x = 2 * x * f' 1 + Real.log x) : f 1 + f' 1 = -3 :=
by sorry

end find_f1_and_f1_l44_44356


namespace dorothy_total_sea_glass_l44_44105

def Blanche_red : ℕ := 3
def Rose_red : ℕ := 9
def Rose_blue : ℕ := 11

def Dorothy_red : ℕ := 2 * (Blanche_red + Rose_red)
def Dorothy_blue : ℕ := 3 * Rose_blue

theorem dorothy_total_sea_glass : Dorothy_red + Dorothy_blue = 57 :=
by
  sorry

end dorothy_total_sea_glass_l44_44105


namespace number_of_initials_is_10000_l44_44567

-- Define the set of letters A through J as a finite set
inductive Letter : Type
| A | B | C | D | E | F | G | H | I | J

open Letter

-- Define a function to count the number of different four-letter sets of initials
def count_initials : Nat :=
  10 ^ 4

-- The theorem to prove: the number of different four-letter sets of initials is 10000
theorem number_of_initials_is_10000 : count_initials = 10000 := by
  sorry

end number_of_initials_is_10000_l44_44567


namespace equalize_rice_move_amount_l44_44663

open Real

noncomputable def containerA_kg : Real := 12
noncomputable def containerA_g : Real := 400
noncomputable def containerB_g : Real := 7600

noncomputable def total_rice_in_A_g : Real := containerA_kg * 1000 + containerA_g
noncomputable def total_rice_in_A_and_B_g : Real := total_rice_in_A_g + containerB_g
noncomputable def equalized_rice_per_container_g : Real := total_rice_in_A_and_B_g / 2

noncomputable def amount_to_move_g : Real := total_rice_in_A_g - equalized_rice_per_container_g
noncomputable def amount_to_move_kg : Real := amount_to_move_g / 1000

theorem equalize_rice_move_amount :
  amount_to_move_kg = 2.4 :=
by
  sorry

end equalize_rice_move_amount_l44_44663


namespace solve_z_l44_44161

variable (z : ℂ) -- Define the variable z in the complex number system
variable (i : ℂ) -- Define the variable i in the complex number system

-- State the conditions: 2 - 3i * z = 4 + 5i * z and i^2 = -1
axiom cond1 : 2 - 3 * i * z = 4 + 5 * i * z
axiom cond2 : i^2 = -1

-- The theorem to prove: z = i / 4
theorem solve_z : z = i / 4 :=
by
  sorry

end solve_z_l44_44161


namespace part_a_part_b_l44_44054

variable {A : Type*} [Ring A]

def B (A : Type*) [Ring A] : Set A :=
  {a | a^2 = 1}

variable (a : A) (b : B A)

theorem part_a (a : A) (b : A) (h : b ∈ B A) : a * b - b * a = b * a * b - a := by
  sorry

theorem part_b (A : Type*) [Ring A] (h : ∀ x : A, x^2 = 0 -> x = 0) : Group (B A) := by
  sorry

end part_a_part_b_l44_44054


namespace count_integers_divisible_by_2_3_5_7_l44_44561

theorem count_integers_divisible_by_2_3_5_7 :
  ∃ n : ℕ, (∀ k : ℕ, k < 500 → (k % 2 = 0 ∧ k % 3 = 0 ∧ k % 5 = 0 ∧ k % 7 = 0) → k ≠ n → k < 500 ∧ k > 0) ∧
  (n = 2) :=
by
  sorry

end count_integers_divisible_by_2_3_5_7_l44_44561


namespace sequence_a_n_2013_l44_44808

theorem sequence_a_n_2013 (a : ℕ → ℤ) (h1 : a 1 = 3) (h2 : a 2 = 6)
  (h : ∀ n, a (n + 2) = a (n + 1) - a n) :
  a 2013 = 3 :=
sorry

end sequence_a_n_2013_l44_44808


namespace tank_capacity_l44_44566

theorem tank_capacity
  (x : ℝ) -- define x as the full capacity of the tank in gallons
  (h1 : (5/6) * x - (2/3) * x = 15) -- first condition
  (h2 : (2/3) * x = y) -- second condition, though not actually needed
  : x = 90 := 
by sorry

end tank_capacity_l44_44566


namespace factor_x4_plus_81_l44_44331

theorem factor_x4_plus_81 (x : ℝ) : (x^2 + 6 * x + 9) * (x^2 - 6 * x + 9) = x^4 + 81 := 
by 
   sorry

end factor_x4_plus_81_l44_44331


namespace flooring_cost_correct_l44_44675

noncomputable def cost_of_flooring (l w h_t b_t c : ℝ) : ℝ :=
  let area_rectangle := l * w
  let area_triangle := (b_t * h_t) / 2
  let area_to_be_floored := area_rectangle - area_triangle
  area_to_be_floored * c

theorem flooring_cost_correct :
  cost_of_flooring 10 7 3 4 900 = 57600 :=
by
  sorry

end flooring_cost_correct_l44_44675


namespace solution_set_of_inequality_l44_44143

theorem solution_set_of_inequality :
  {x : ℝ | (x + 1) / x ≤ 3} = {x : ℝ | x < 0} ∪ {x : ℝ | x ≥ 1 / 2} :=
by
  sorry

end solution_set_of_inequality_l44_44143


namespace chord_ratio_l44_44450

theorem chord_ratio (A B C D P : Type) (AP BP CP DP : ℝ)
  (h1 : AP = 4) (h2 : CP = 9)
  (h3 : AP * BP = CP * DP) : BP / DP = 9 / 4 := 
by 
  sorry

end chord_ratio_l44_44450


namespace average_value_function_example_l44_44866

def average_value_function (f : ℝ → ℝ) (a b : ℝ) :=
  ∃ x0 : ℝ, a < x0 ∧ x0 < b ∧ f x0 = (f b - f a) / (b - a)

theorem average_value_function_example :
  average_value_function (λ x => x^2 - m * x - 1) (-1) (1) → 
  ∃ m : ℝ, 0 < m ∧ m < 2 :=
by
  intros h
  sorry

end average_value_function_example_l44_44866


namespace find_m_l44_44096

theorem find_m (y x m : ℝ) (h1 : 2 - 3 * (1 - y) = 2 * y) (h2 : y = x) (h3 : m * (x - 3) - 2 = -8) : m = 3 :=
sorry

end find_m_l44_44096


namespace triangle_area_l44_44298

theorem triangle_area (a b c : ℝ) (h1: a = 15) (h2: c = 17) (h3: a^2 + b^2 = c^2) :
  (1 / 2) * a * b = 60 :=
by
  sorry

end triangle_area_l44_44298


namespace max_revenue_l44_44332

variable (x y : ℝ)

-- Conditions
def ads_time_constraint := x + y ≤ 300
def ads_cost_constraint := 500 * x + 200 * y ≤ 90000
def revenue := 0.3 * x + 0.2 * y

-- Question: Prove that the maximum revenue is 70 million yuan
theorem max_revenue (h_time : ads_time_constraint (x := 100) (y := 200))
                    (h_cost : ads_cost_constraint (x := 100) (y := 200)) :
  revenue (x := 100) (y := 200) = 70 := 
sorry

end max_revenue_l44_44332


namespace find_S40_l44_44652

noncomputable def geometric_sequence_sum (a r : ℝ) (n : ℕ) : ℝ := a * (r^n - 1) / (r - 1)

theorem find_S40 (a r : ℝ) (S : ℕ → ℝ)
  (h1 : ∀ n, S n = geometric_sequence_sum a r n)
  (h2 : S 10 = 10)
  (h3 : S 30 = 70) :
  S 40 = 150 ∨ S 40 = 110 := 
sorry

end find_S40_l44_44652


namespace composite_square_perimeter_l44_44262

theorem composite_square_perimeter (p1 p2 : ℝ) (h1 : p1 = 40) (h2 : p2 = 100) : 
  let s1 := p1 / 4
  let s2 := p2 / 4
  (p1 + p2 - 2 * s1) = 120 := 
by
  -- proof goes here
  sorry

end composite_square_perimeter_l44_44262


namespace cube_volume_ratio_l44_44897

theorem cube_volume_ratio
  (a : ℕ) (b : ℕ)
  (h₁ : a = 5)
  (h₂ : b = 24)
  : (a^3 : ℚ) / (b^3 : ℚ) = 125 / 13824 := by
  sorry

end cube_volume_ratio_l44_44897


namespace average_gas_mileage_round_trip_l44_44260

theorem average_gas_mileage_round_trip :
  let distance_to_city := 150
  let mpg_sedan := 25
  let mpg_rental := 15
  let total_distance := 2 * distance_to_city
  let gas_used_outbound := distance_to_city / mpg_sedan
  let gas_used_return := distance_to_city / mpg_rental
  let total_gas_used := gas_used_outbound + gas_used_return
  let avg_gas_mileage := total_distance / total_gas_used
  avg_gas_mileage = 18.75 := by
{
  sorry
}

end average_gas_mileage_round_trip_l44_44260


namespace find_Q_l44_44623

theorem find_Q (m n Q p : ℝ) (h1 : m = 6 * n + 5)
    (h2 : p = 0.3333333333333333)
    (h3 : m + Q = 6 * (n + p) + 5) : Q = 2 := 
by
  sorry

end find_Q_l44_44623


namespace jam_fraction_left_after_dinner_l44_44781

noncomputable def jam_left_after_dinner (initial: ℚ) (lunch_fraction: ℚ) (dinner_fraction: ℚ) : ℚ :=
  initial - (initial * lunch_fraction) - ((initial - (initial * lunch_fraction)) * dinner_fraction)

theorem jam_fraction_left_after_dinner :
  jam_left_after_dinner 1 (1/3) (1/7) = (4/7) :=
by
  sorry

end jam_fraction_left_after_dinner_l44_44781


namespace polynomial_expansion_a5_l44_44924

theorem polynomial_expansion_a5 :
  (x - 1) ^ 8 = (1 : ℤ) + a₁ * x + a₂ * x^2 + a₃ * x^3 + a₄ * x^4 + a₅ * x^5 + a₆ * x^6 + a₇ * x^7 + a₈ * x^8 →
  a₅ = -56 :=
by
  intro h
  -- The proof is omitted.
  sorry

end polynomial_expansion_a5_l44_44924


namespace ratio_of_part_to_whole_l44_44632

theorem ratio_of_part_to_whole (N : ℝ) (h1 : (1/3) * (2/5) * N = 15) (h2 : (40/100) * N = 180) :
  (15 / N) = (1 / 7.5) :=
by
  sorry

end ratio_of_part_to_whole_l44_44632


namespace sticker_distribution_probability_l44_44315

theorem sticker_distribution_probability :
  let p := 32
  let q := 50050
  p + q = 50082 :=
sorry

end sticker_distribution_probability_l44_44315


namespace peter_has_142_nickels_l44_44053

-- Define the conditions
def nickels (n : ℕ) : Prop :=
  40 < n ∧ n < 400 ∧
  n % 4 = 2 ∧
  n % 5 = 2 ∧
  n % 7 = 2

-- The theorem to prove the number of nickels
theorem peter_has_142_nickels : ∃ (n : ℕ), nickels n ∧ n = 142 :=
by {
  sorry
}

end peter_has_142_nickels_l44_44053


namespace max_value_E_X_E_Y_l44_44383

open MeasureTheory

-- Defining the random variables and their ranges
variables {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω)
variable (X : Ω → ℝ) (Y : Ω → ℝ)

-- Condition: 2 ≤ X ≤ 3
def condition1 : Prop := ∀ ω, 2 ≤ X ω ∧ X ω ≤ 3

-- Condition: XY = 1
def condition2 : Prop := ∀ ω, X ω * Y ω = 1

-- The theorem statement
theorem max_value_E_X_E_Y (h1 : condition1 X) (h2 : condition2 X Y) : 
  ∃ E_X E_Y, (E_X = ∫ ω, X ω ∂μ) ∧ (E_Y = ∫ ω, Y ω ∂μ) ∧ (E_X * E_Y = 25 / 24) := 
sorry

end max_value_E_X_E_Y_l44_44383


namespace three_digit_permuted_mean_l44_44894

theorem three_digit_permuted_mean (N : ℕ) :
  (∃ x y z : ℕ, N = 100 * x + 10 * y + z ∧ x < 10 ∧ y < 10 ∧ z < 10 ∧
    (N = 111 ∨ N = 222 ∨ N = 333 ∨ N = 444 ∨ N = 555 ∨ N = 666 ∨ N = 777 ∨ N = 888 ∨ N = 999 ∨
     N = 407 ∨ N = 518 ∨ N = 629 ∨ N = 370 ∨ N = 481 ∨ N = 592)) ↔
    (∃ x y z : ℕ, N = 100 * x + 10 * y + z ∧ x < 10 ∧ y < 10 ∧ z < 10 ∧ 7 * x = 3 * y + 4 * z) := by
sorry

end three_digit_permuted_mean_l44_44894


namespace semicircle_radius_correct_l44_44640

noncomputable def semicircle_radius (P : ℝ) : ℝ := P / (Real.pi + 2)

theorem semicircle_radius_correct (h :127 =113): semicircle_radius 113 = 113 / (Real.pi + 2) :=
by
  sorry

end semicircle_radius_correct_l44_44640


namespace find_multiple_l44_44648

theorem find_multiple (a b m : ℤ) (h1 : b = 7) (h2 : b - a = 2) 
  (h3 : a * b = m * (a + b) + 11) : m = 2 :=
by {
  sorry
}

end find_multiple_l44_44648


namespace equidistant_cyclist_l44_44650

-- Definition of key parameters
def speed_car := 60  -- in km/h
def speed_cyclist := 18  -- in km/h
def speed_pedestrian := 6  -- in km/h
def distance_AC := 10  -- in km
def angle_ACB := 60  -- in degrees
def time_car_start := (7, 58)  -- 7:58 AM
def time_cyclist_start := (8, 0)  -- 8:00 AM
def time_pedestrian_start := (6, 44) -- 6:44 AM
def time_solution := (8, 6)  -- 8:06 AM

-- Time difference function
def time_diff (t1 t2 : Nat × Nat) : Nat :=
  (t2.1 - t1.1) * 60 + (t2.2 - t1.2)  -- time difference in minutes

-- Convert minutes to hours
noncomputable def minutes_to_hours (m : Nat) : ℝ :=
  m / 60.0

-- Distances traveled by car, cyclist, and pedestrian by the given time
noncomputable def distance_car (t1 t2 : Nat × Nat) : ℝ :=
  speed_car * (minutes_to_hours (time_diff t1 t2) + 2 / 60.0)

noncomputable def distance_cyclist (t1 t2 : Nat × Nat) : ℝ :=
  speed_cyclist * minutes_to_hours (time_diff t1 t2)

noncomputable def distance_pedestrian (t1 t2 : Nat × Nat) : ℝ :=
  speed_pedestrian * (minutes_to_hours (time_diff t1 t2) + 136 / 60.0)

-- Verification statement
theorem equidistant_cyclist :
  distance_car time_car_start time_solution = distance_pedestrian time_pedestrian_start time_solution → 
  distance_cyclist time_cyclist_start time_solution = 
  distance_car time_car_start time_solution ∧
  distance_cyclist time_cyclist_start time_solution = 
  distance_pedestrian time_pedestrian_start time_solution :=
by
  -- Given conditions and the correctness to be shown
  sorry

end equidistant_cyclist_l44_44650


namespace sphere_diagonal_property_l44_44133

variable {A B C D : ℝ}

-- conditions provided
variable (radius : ℝ) (x y z : ℝ)
variable (h_radius : radius = 1)
variable (h_non_coplanar : ¬(is_coplanar A B C D))
variable (h_AB_CD : dist A B = x ∧ dist C D = x)
variable (h_BC_DA : dist B C = y ∧ dist D A = y)
variable (h_CA_BD : dist C A = z ∧ dist B D = z)

theorem sphere_diagonal_property :
  x^2 + y^2 + z^2 = 8 := 
sorry

end sphere_diagonal_property_l44_44133


namespace tangent_slope_at_1_l44_44224

def f (x : ℝ) : ℝ := x^3 + x^2 + 1

theorem tangent_slope_at_1 : (deriv f 1) = 5 := by
  sorry

end tangent_slope_at_1_l44_44224


namespace angleC_is_36_l44_44196

theorem angleC_is_36 
  (p q r : ℝ)  -- fictitious types for lines, as Lean needs a type here
  (A B C : ℝ)  -- Angles as Real numbers
  (hpq : p = q)  -- Line p is parallel to line q (represented equivalently for Lean)
  (h : A = 1/4 * B)
  (hr : B + C = 180)
  (vert_opposite : C = A) :
  C = 36 := 
by
  sorry

end angleC_is_36_l44_44196


namespace at_least_240_students_l44_44539

-- Define the total number of students
def total_students : ℕ := 1200

-- Define the 80th percentile score
def percentile_80_score : ℕ := 103

-- Define the number of students below the 80th percentile
def students_below_80th_percentile : ℕ := total_students * 80 / 100

-- Define the number of students with at least the 80th percentile score
def students_at_least_80th_percentile : ℕ := total_students - students_below_80th_percentile

-- The theorem to prove
theorem at_least_240_students : students_at_least_80th_percentile ≥ 240 :=
by
  -- Placeholder proof, to be filled in as the actual proof
  sorry

end at_least_240_students_l44_44539


namespace number_of_persons_l44_44301

theorem number_of_persons
    (total_amount : ℕ) 
    (amount_per_person : ℕ) 
    (h1 : total_amount = 42900) 
    (h2 : amount_per_person = 1950) :
    total_amount / amount_per_person = 22 :=
by
  sorry

end number_of_persons_l44_44301


namespace solve_pair_N_n_l44_44483

def is_solution_pair (N n : ℕ) : Prop :=
  N ^ 2 = 1 + n * (N + n)

def fibonacci : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => fibonacci n + fibonacci (n + 1)

theorem solve_pair_N_n (N n : ℕ) (i : ℕ) :
  is_solution_pair N n ↔ N = fibonacci (i + 1) ∧ n = fibonacci i := sorry

end solve_pair_N_n_l44_44483


namespace number_divisibility_l44_44027

def A_n (n : ℕ) : ℕ := (10^(3^n) - 1) / 9

theorem number_divisibility (n : ℕ) :
  (3^n ∣ A_n n) ∧ ¬ (3^(n + 1) ∣ A_n n) := by
  sorry

end number_divisibility_l44_44027


namespace chord_intersects_inner_circle_l44_44202

noncomputable def probability_chord_intersects_inner_circle
  (r1 r2 : ℝ) (h1 : r1 = 2) (h2 : r2 = 5) : ℝ :=
0.098

theorem chord_intersects_inner_circle :
  probability_chord_intersects_inner_circle 2 5 rfl rfl = 0.098 :=
sorry

end chord_intersects_inner_circle_l44_44202


namespace coupon_percentage_l44_44533

theorem coupon_percentage (P i d final_price total_price discount_amount percentage: ℝ)
  (h1 : P = 54) (h2 : i = 20) (h3 : d = 0.20 * i) 
  (h4 : total_price = P - d) (h5 : final_price = 45) 
  (h6 : discount_amount = total_price - final_price) 
  (h7 : percentage = (discount_amount / total_price) * 100) : 
  percentage = 10 := 
by
  sorry

end coupon_percentage_l44_44533


namespace largest_integer_base8_square_l44_44151

theorem largest_integer_base8_square :
  ∃ (N : ℕ), (N^2 >= 8^3) ∧ (N^2 < 8^4) ∧ (N = 63 ∧ N % 8 = 7) := sorry

end largest_integer_base8_square_l44_44151


namespace roots_expression_value_l44_44900

theorem roots_expression_value {a b : ℝ} 
  (h₁ : a^2 + a - 3 = 0) 
  (h₂ : b^2 + b - 3 = 0) 
  (ha_ne_hb : a ≠ b) : 
  a * b - 2023 * a - 2023 * b = 2020 :=
by 
  sorry

end roots_expression_value_l44_44900


namespace orange_juice_serving_size_l44_44772

theorem orange_juice_serving_size (n_servings : ℕ) (c_concentrate : ℕ) (v_concentrate : ℕ) (c_water_per_concentrate : ℕ)
    (v_cans : ℕ) (expected_serving_size : ℕ) 
    (h1 : n_servings = 200)
    (h2 : c_concentrate = 60)
    (h3 : v_concentrate = 5)
    (h4 : c_water_per_concentrate = 3)
    (h5 : v_cans = 5)
    (h6 : expected_serving_size = 6) : 
   (c_concentrate * v_concentrate + c_concentrate * c_water_per_concentrate * v_cans) / n_servings = expected_serving_size := 
by 
  sorry

end orange_juice_serving_size_l44_44772


namespace neg_of_univ_prop_l44_44644

theorem neg_of_univ_prop :
  (∃ (x₀ : ℝ), 0 ≤ x₀ ∧ x₀^3 + x₀ < 0) ↔ ¬ (∀ (x : ℝ), 0 ≤ x → x^3 + x ≥ 0) := by
sorry

end neg_of_univ_prop_l44_44644


namespace shoes_remaining_l44_44094

theorem shoes_remaining (monthly_goal : ℕ) (sold_last_week : ℕ) (sold_this_week : ℕ) (remaining_shoes : ℕ) :
  monthly_goal = 80 →
  sold_last_week = 27 →
  sold_this_week = 12 →
  remaining_shoes = monthly_goal - sold_last_week - sold_this_week →
  remaining_shoes = 41 := by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  exact h4

end shoes_remaining_l44_44094


namespace polynomial_equivalence_l44_44285

-- Define the polynomial T in terms of x.
def T (x : ℝ) : ℝ := (x-2)^5 + 5 * (x-2)^4 + 10 * (x-2)^3 + 10 * (x-2)^2 + 5 * (x-2) + 1

-- Define the target polynomial.
def target (x : ℝ) : ℝ := (x-1)^5

-- State the theorem that T is equivalent to target.
theorem polynomial_equivalence (x : ℝ) : T x = target x :=
by
  sorry

end polynomial_equivalence_l44_44285


namespace range_of_a_l44_44257

open Real

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x^2 + a * x + 1 ≥ 0) ↔ -2 ≤ a ∧ a ≤ 2 :=
by
  sorry

end range_of_a_l44_44257


namespace time_in_1867_minutes_correct_l44_44643

def current_time := (3, 15) -- (hours, minutes)
def minutes_in_hour := 60
def total_minutes := 1867
def hours_after := total_minutes / minutes_in_hour
def remainder_minutes := total_minutes % minutes_in_hour
def result_time := ((current_time.1 + hours_after) % 24, current_time.2 + remainder_minutes)
def expected_time := (22, 22) -- 10:22 p.m. in 24-hour format

theorem time_in_1867_minutes_correct : result_time = expected_time := 
by
    -- No proof is required according to the instructions.
    sorry

end time_in_1867_minutes_correct_l44_44643


namespace average_price_per_book_l44_44595

theorem average_price_per_book
  (amount_spent_first_shop : ℕ)
  (amount_spent_second_shop : ℕ)
  (books_first_shop : ℕ)
  (books_second_shop : ℕ)
  (total_amount_spent : ℕ := amount_spent_first_shop + amount_spent_second_shop)
  (total_books_bought : ℕ := books_first_shop + books_second_shop)
  (average_price : ℕ := total_amount_spent / total_books_bought) :
  amount_spent_first_shop = 520 → amount_spent_second_shop = 248 →
  books_first_shop = 42 → books_second_shop = 22 →
  average_price = 12 :=
by
  intros
  sorry

end average_price_per_book_l44_44595


namespace value_of_M_in_equation_l44_44401

theorem value_of_M_in_equation :
  ∀ {M : ℕ}, (32 = 2^5) ∧ (8 = 2^3) → (32^3 * 8^4 = 2^M) → M = 27 :=
by
  intros M h1 h2
  sorry

end value_of_M_in_equation_l44_44401


namespace no_value_of_n_l44_44748

noncomputable def t1 (n : ℕ) : ℚ :=
3 * n * (n + 2)

noncomputable def t2 (n : ℕ) : ℚ :=
(3 * n^2 + 19 * n) / 2

theorem no_value_of_n (n : ℕ) (h : n > 0) : t1 n ≠ t2 n :=
by {
  sorry
}

end no_value_of_n_l44_44748


namespace Fred_last_week_l44_44294

-- Definitions from conditions
def Fred_now := 40
def Fred_earned := 21

-- The theorem we need to prove
theorem Fred_last_week :
  Fred_now - Fred_earned = 19 :=
by
  sorry

end Fred_last_week_l44_44294


namespace area_of_path_is_675_l44_44938

def rectangular_field_length : ℝ := 75
def rectangular_field_width : ℝ := 55
def path_width : ℝ := 2.5

def area_of_path : ℝ :=
  let new_length := rectangular_field_length + 2 * path_width
  let new_width := rectangular_field_width + 2 * path_width
  let area_with_path := new_length * new_width
  let area_of_grass_field := rectangular_field_length * rectangular_field_width
  area_with_path - area_of_grass_field

theorem area_of_path_is_675 : area_of_path = 675 := by
  sorry

end area_of_path_is_675_l44_44938


namespace total_votes_election_l44_44529

theorem total_votes_election
  (pct_candidate1 pct_candidate2 pct_candidate3 pct_candidate4 : ℝ)
  (votes_candidate4 total_votes : ℝ)
  (h1 : pct_candidate1 = 0.42)
  (h2 : pct_candidate2 = 0.30)
  (h3 : pct_candidate3 = 0.20)
  (h4 : pct_candidate4 = 0.08)
  (h5 : votes_candidate4 = 720)
  (h6 : votes_candidate4 = pct_candidate4 * total_votes) :
  total_votes = 9000 :=
sorry

end total_votes_election_l44_44529


namespace max_volume_pyramid_l44_44091

theorem max_volume_pyramid 
  (AB AC : ℝ)
  (sin_BAC : ℝ)
  (angle_cond : ∀ (SA SB SC : ℝ), SA = SB ∧ SB = SC ∧ SC = SA → ∀ θ, θ ≤ 60 → true)
  (h : ℝ)
  (V : ℝ)
  (AB_eq : AB = 3)
  (AC_eq : AC = 5)
  (sin_BAC_eq : sin_BAC = 3/5)
  (height_cond : h = (5 * Real.sqrt 3) / 2)
  (volume_cond : V = (1/3) * (1/2 * 3 * 5 * (3/5)) * h) :
  V = (5 * Real.sqrt 174) / 4 := sorry

end max_volume_pyramid_l44_44091


namespace tutors_meeting_schedule_l44_44056

/-- In a school, five tutors, Jaclyn, Marcelle, Susanna, Wanda, and Thomas, 
are scheduled to work in the library. Their schedules are as follows: 
Jaclyn works every fifth school day, Marcelle works every sixth school day, 
Susanna works every seventh school day, Wanda works every eighth school day, 
and Thomas works every ninth school day. Today, all five tutors are working 
in the library. Prove that the least common multiple of 5, 6, 7, 8, and 9 is 2520 days. 
-/
theorem tutors_meeting_schedule : Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 9))) = 2520 := 
by
  sorry

end tutors_meeting_schedule_l44_44056


namespace oranges_to_pears_l44_44457

-- Define the equivalence relation between oranges and pears
def equivalent_weight (orange pear : ℕ) : Prop := 4 * pear = 3 * orange

-- Given:
-- 1. 4 oranges weigh the same as 3 pears
-- 2. Jimmy has 36 oranges
-- Prove that 27 pears are required to balance the weight of 36 oranges
theorem oranges_to_pears (orange pear : ℕ) (h : equivalent_weight 1 1) :
  (4 * pear = 3 * orange) → equivalent_weight 36 27 :=
by
  sorry

end oranges_to_pears_l44_44457


namespace expand_expression_l44_44914

variable (y : ℤ)

theorem expand_expression : 12 * (3 * y - 4) = 36 * y - 48 := 
by
  sorry

end expand_expression_l44_44914


namespace sum_S_15_22_31_l44_44622

-- Define the sequence \{a_n\} with the sum of the first n terms S_n
def S : ℕ → ℤ
| 0 => 0
| n + 1 => S n + (-1: ℤ)^n * (4 * (n + 1) - 3)

-- The statement to prove: S_{15} + S_{22} - S_{31} = -76
theorem sum_S_15_22_31 : S 15 + S 22 - S 31 = -76 :=
sorry

end sum_S_15_22_31_l44_44622


namespace equivalent_expression_l44_44926

theorem equivalent_expression : 8^8 * 4^4 / 2^28 = 16 := by
  -- Here, we're stating the equivalency directly
  sorry

end equivalent_expression_l44_44926


namespace profit_function_is_correct_marginal_profit_function_is_correct_profit_function_max_value_marginal_profit_function_max_value_profit_and_marginal_profit_max_not_equal_l44_44040

noncomputable def R (x : ℕ) : ℝ := 3000 * x - 20 * x^2
noncomputable def C (x : ℕ) : ℝ := 500 * x + 4000
noncomputable def p (x : ℕ) : ℝ := R x - C x
noncomputable def Mp (x : ℕ) : ℝ := p (x + 1) - p x

theorem profit_function_is_correct : ∀ x, p x = -20 * x^2 + 2500 * x - 4000 := 
by 
  intro x
  sorry

theorem marginal_profit_function_is_correct : ∀ x, 0 < x ∧ x ≤ 100 → Mp x = -40 * x + 2480 := 
by 
  intro x
  sorry

theorem profit_function_max_value : ∃ x, (x = 62 ∨ x = 63) ∧ p x = 74120 :=
by 
  sorry

theorem marginal_profit_function_max_value : ∃ x, x = 1 ∧ Mp x = 2440 :=
by 
  sorry

theorem profit_and_marginal_profit_max_not_equal : ¬ (∃ x y, (x = 62 ∨ x = 63) ∧ y = 1 ∧ p x = Mp y) :=
by 
  sorry

end profit_function_is_correct_marginal_profit_function_is_correct_profit_function_max_value_marginal_profit_function_max_value_profit_and_marginal_profit_max_not_equal_l44_44040


namespace initial_amount_of_A_l44_44682

variable (a b c : ℕ)

-- Conditions
axiom condition1 : a - b - c = 32
axiom condition2 : b + c = 48
axiom condition3 : a + b + c = 128

-- The goal is to prove that A had 80 cents initially.
theorem initial_amount_of_A : a = 80 :=
by
  -- We need to skip the proof here
  sorry

end initial_amount_of_A_l44_44682


namespace minimum_abs_phi_l44_44909

noncomputable def f (ω φ x : ℝ) : ℝ := 2 * Real.sin (ω * x + φ)

theorem minimum_abs_phi 
  (ω φ b : ℝ)
  (hω : ω > 0)
  (hb : 0 < b ∧ b < 2)
  (h_intersections : f ω φ (π / 6) = b ∧ f ω φ (5 * π / 6) = b ∧ f ω φ (7 * π / 6) = b)
  (h_minimum : f ω φ (3 * π / 2) = -2) : 
  |φ| = π / 2 :=
sorry

end minimum_abs_phi_l44_44909


namespace translation_graph_pass_through_point_l44_44016

theorem translation_graph_pass_through_point :
  (∃ a : ℝ, (∀ x y : ℝ, y = -2 * x + 1 - 3 → y = 3 → x = a) → a = -5/2) :=
sorry

end translation_graph_pass_through_point_l44_44016


namespace tom_days_to_finish_l44_44764

noncomputable def days_to_finish_show
  (episodes : Nat) 
  (minutes_per_episode : Nat) 
  (hours_per_day : Nat) : Nat :=
  let total_minutes := episodes * minutes_per_episode
  let total_hours := total_minutes / 60
  total_hours / hours_per_day

theorem tom_days_to_finish :
  days_to_finish_show 90 20 2 = 15 :=
by
  -- the proof steps go here
  sorry

end tom_days_to_finish_l44_44764


namespace tank_holds_21_liters_l44_44685

def tank_capacity (S L : ℝ) : Prop :=
  (L = 2 * S + 3) ∧
  (L = 4) ∧
  (2 * S + 5 * L = 21)

theorem tank_holds_21_liters :
  ∃ S L : ℝ, tank_capacity S L :=
by
  use 1/2, 4
  unfold tank_capacity
  simp
  sorry

end tank_holds_21_liters_l44_44685


namespace proof_equiv_l44_44911

noncomputable def M : Set ℝ := { y | ∃ x : ℝ, y = 2 ^ Real.sqrt (3 + 2 * x - x ^ 2) }
noncomputable def N : Set ℝ := { x | ∃ y : ℝ, y = Real.log (x - 2) }
def I : Set ℝ := Set.univ
def complement_N : Set ℝ := I \ N

theorem proof_equiv : M ∩ complement_N = { y | 1 ≤ y ∧ y ≤ 2 } :=
sorry

end proof_equiv_l44_44911


namespace max_chords_intersecting_line_l44_44583

theorem max_chords_intersecting_line (A : Fin 2017 → Type) :
  ∃ k : ℕ, (k ≤ 2016 ∧ ∃ m : ℕ, (m = k * (2016 - k) + 2016) ∧ m = 1018080) :=
sorry

end max_chords_intersecting_line_l44_44583


namespace sum_of_primes_between_30_and_50_l44_44436

-- Define a prime number
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- List of prime numbers between 30 and 50
def prime_numbers_between_30_and_50 : List ℕ := [31, 37, 41, 43, 47]

-- Sum of prime numbers between 30 and 50
def sum_prime_numbers_between_30_and_50 : ℕ :=
  prime_numbers_between_30_and_50.sum

-- Theorem: The sum of prime numbers between 30 and 50 is 199
theorem sum_of_primes_between_30_and_50 :
  sum_prime_numbers_between_30_and_50 = 199 := by
    sorry

end sum_of_primes_between_30_and_50_l44_44436


namespace extra_profit_is_60000_l44_44182

theorem extra_profit_is_60000 (base_house_cost special_house_cost base_house_price special_house_price : ℝ) :
  (special_house_cost = base_house_cost + 100000) →
  (special_house_price = 1.5 * base_house_price) →
  (base_house_price = 320000) →
  (special_house_price - base_house_price - 100000 = 60000) :=
by
  -- Definitions and conditions
  intro h1 h2 h3
  -- Placeholder for the eventual proof
  sorry

end extra_profit_is_60000_l44_44182


namespace other_coin_value_l44_44141

-- Condition definitions
def total_coins : ℕ := 36
def dime_count : ℕ := 26
def total_value_dollars : ℝ := 3.10
def dime_value : ℝ := 0.10

-- Derived definitions
def total_dimes_value : ℝ := dime_count * dime_value
def remaining_value : ℝ := total_value_dollars - total_dimes_value
def other_coin_count : ℕ := total_coins - dime_count

-- Proof statement
theorem other_coin_value : (remaining_value / other_coin_count) = 0.05 := by
  sorry

end other_coin_value_l44_44141


namespace super_k_teams_l44_44203

theorem super_k_teams (n : ℕ) (h : n * (n - 1) / 2 = 45) : n = 10 :=
sorry

end super_k_teams_l44_44203


namespace vertex_of_parabola_l44_44732

-- Define the statement of the problem
theorem vertex_of_parabola :
  ∀ (a h k : ℝ), (∀ x : ℝ, 3 * (x - 5) ^ 2 + 4 = a * (x - h) ^ 2 + k) → (h, k) = (5, 4) :=
by
  sorry

end vertex_of_parabola_l44_44732


namespace renovation_project_truck_load_l44_44850

theorem renovation_project_truck_load (sand : ℝ) (dirt : ℝ) (cement : ℝ)
  (h1 : sand = 0.17) (h2 : dirt = 0.33) (h3 : cement = 0.17) :
  sand + dirt + cement = 0.67 :=
by
  sorry

end renovation_project_truck_load_l44_44850


namespace a_finishes_work_in_four_days_l44_44895

theorem a_finishes_work_in_four_days (x : ℝ) 
  (B_work_rate : ℝ) 
  (work_done_together : ℝ) 
  (work_done_by_B_alone : ℝ) : 
  B_work_rate = 1 / 16 → 
  work_done_together = 2 * (1 / x + 1 / 16) → 
  work_done_by_B_alone = 6 * (1 / 16) → 
  work_done_together + work_done_by_B_alone = 1 → 
  x = 4 :=
by
  intros hB hTogether hBAlone hTotal
  sorry

end a_finishes_work_in_four_days_l44_44895


namespace length_of_platform_l44_44169

/--
Problem statement:
A train 450 m long running at 108 km/h crosses a platform in 25 seconds.
Prove that the length of the platform is 300 meters.

Given:
- The train is 450 meters long.
- The train's speed is 108 km/h.
- The train crosses the platform in 25 seconds.

To prove:
The length of the platform is 300 meters.
-/
theorem length_of_platform :
  let train_length := 450
  let train_speed := 108 * (1000 / 3600) -- converting km/h to m/s
  let crossing_time := 25
  let total_distance_covered := train_speed * crossing_time
  let platform_length := total_distance_covered - train_length
  platform_length = 300 := by
  sorry

end length_of_platform_l44_44169


namespace opposite_of_three_minus_one_l44_44363

theorem opposite_of_three_minus_one : -(3 - 1) = -2 := 
by
  sorry

end opposite_of_three_minus_one_l44_44363


namespace colored_ints_square_diff_l44_44031

-- Define a coloring function c as a total function from ℤ to a finite set {0, 1, 2}
def c : ℤ → Fin 3 := sorry

-- Lean 4 statement for the problem
theorem colored_ints_square_diff : 
  ∃ a b : ℤ, a ≠ b ∧ c a = c b ∧ ∃ k : ℤ, a - b = k ^ 2 :=
sorry

end colored_ints_square_diff_l44_44031


namespace smallest_x_value_l44_44985

theorem smallest_x_value (x : ℤ) (h : 3 * x^2 - 4 < 20) : x = -2 :=
sorry

end smallest_x_value_l44_44985


namespace cans_purchased_l44_44507

theorem cans_purchased (S Q E : ℝ) (h1 : Q ≠ 0) (h2 : S > 0) :
  (10 * E * S) / Q = (10 * (E : ℝ) * (S : ℝ)) / (Q : ℝ) := by 
  sorry

end cans_purchased_l44_44507


namespace smallest_w_correct_l44_44120

-- Define the conditions
def is_factor (a b : ℕ) : Prop := ∃ k, a = b * k

-- Given conditions
def cond1 (w : ℕ) : Prop := is_factor (2^6) (1152 * w)
def cond2 (w : ℕ) : Prop := is_factor (3^4) (1152 * w)
def cond3 (w : ℕ) : Prop := is_factor (5^3) (1152 * w)
def cond4 (w : ℕ) : Prop := is_factor (7^2) (1152 * w)
def cond5 (w : ℕ) : Prop := is_factor (11) (1152 * w)
def is_positive (w : ℕ) : Prop := w > 0

-- The smallest possible value of w given all conditions
def smallest_w : ℕ := 16275

-- Proof statement
theorem smallest_w_correct : 
  ∀ (w : ℕ), cond1 w ∧ cond2 w ∧ cond3 w ∧ cond4 w ∧ cond5 w ∧ is_positive w ↔ w = smallest_w := sorry

end smallest_w_correct_l44_44120


namespace shallow_depth_of_pool_l44_44555

theorem shallow_depth_of_pool (w l D V : ℝ) (h₀ : w = 9) (h₁ : l = 12) (h₂ : D = 4) (h₃ : V = 270) :
  (0.5 * (d + D) * w * l = V) → d = 1 :=
by
  intros h_equiv
  sorry

end shallow_depth_of_pool_l44_44555
