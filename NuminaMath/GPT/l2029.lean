import Mathlib

namespace largest_number_2013_l2029_202987

theorem largest_number_2013 (x y : ℕ) (h1 : x + y = 2013)
    (h2 : y = 5 * (x / 100 + 1)) : max x y = 1913 := by
  sorry

end largest_number_2013_l2029_202987


namespace inverse_proportion_relation_l2029_202931

theorem inverse_proportion_relation (x₁ x₂ y₁ y₂ : ℝ) 
  (h1 : y₁ = 2 / x₁) 
  (h2 : y₂ = 2 / x₂) 
  (h3 : x₁ < x₂) 
  (h4 : x₂ < 0) : 
  y₂ < y₁ ∧ y₁ < 0 := 
sorry

end inverse_proportion_relation_l2029_202931


namespace find_number_satisfies_l2029_202924

noncomputable def find_number (m : ℤ) (n : ℤ) : Prop :=
  (m % n = 2) ∧ (3 * m % n = 1)

theorem find_number_satisfies (m : ℤ) : ∃ n : ℤ, find_number m n ∧ n = 5 :=
by
  sorry

end find_number_satisfies_l2029_202924


namespace last_number_aryana_counts_l2029_202939

theorem last_number_aryana_counts (a d : ℤ) (h_start : a = 72) (h_diff : d = -11) :
  ∃ n : ℕ, (a + n * d > 0) ∧ (a + (n + 1) * d ≤ 0) ∧ a + n * d = 6 := by
  sorry

end last_number_aryana_counts_l2029_202939


namespace negation_of_exists_gt0_and_poly_gt0_l2029_202918

theorem negation_of_exists_gt0_and_poly_gt0 :
  (¬ (∃ x₀ : ℝ, x₀ > 0 ∧ x₀^2 - 5 * x₀ + 6 > 0)) ↔ 
  (∀ x : ℝ, x > 0 → x^2 - 5 * x + 6 ≤ 0) :=
by sorry

end negation_of_exists_gt0_and_poly_gt0_l2029_202918


namespace positive_integer_pair_solution_l2029_202961

theorem positive_integer_pair_solution :
  ∃ a b : ℕ, (a > 0) ∧ (b > 0) ∧ 
    ¬ (7 ∣ (a * b * (a + b))) ∧ 
    (7^7 ∣ ((a + b)^7 - a^7 - b^7)) ∧ 
    (a, b) = (18, 1) :=
by {
  sorry
}

end positive_integer_pair_solution_l2029_202961


namespace percentage_workday_in_meetings_l2029_202950

theorem percentage_workday_in_meetings :
  let workday_minutes := 10 * 60
  let first_meeting := 30
  let second_meeting := 2 * first_meeting
  let third_meeting := first_meeting + second_meeting
  let total_meeting_minutes := first_meeting + second_meeting + third_meeting
  (total_meeting_minutes * 100) / workday_minutes = 30 :=
by
  sorry

end percentage_workday_in_meetings_l2029_202950


namespace find_oranges_l2029_202979

def A : ℕ := 3
def B : ℕ := 1

theorem find_oranges (O : ℕ) : A + B + O + (A + 4) + 10 * B + 2 * (A + 4) = 39 → O = 4 :=
by 
  intros h
  sorry

end find_oranges_l2029_202979


namespace final_velocity_l2029_202975

variable (u a t : ℝ)

-- Defining the conditions
def initial_velocity := u = 0
def acceleration := a = 1.2
def time := t = 15

-- Statement of the theorem
theorem final_velocity : initial_velocity u ∧ acceleration a ∧ time t → (u + a * t = 18) := by
  sorry

end final_velocity_l2029_202975


namespace solve_for_x_l2029_202952

theorem solve_for_x (x : ℝ) (hx : x^(1/10) * (x^(3/2))^(1/10) = 3) : x = 9 :=
sorry

end solve_for_x_l2029_202952


namespace total_pages_in_book_l2029_202991

-- Define the given conditions
def chapters : Nat := 41
def days : Nat := 30
def pages_per_day : Nat := 15

-- Define the statement to be proven
theorem total_pages_in_book : (days * pages_per_day) = 450 := by
  sorry

end total_pages_in_book_l2029_202991


namespace sum_odd_product_even_l2029_202972

theorem sum_odd_product_even (a b : ℤ) (h1 : ∃ k : ℤ, a = 2 * k) 
                             (h2 : ∃ m : ℤ, b = 2 * m + 1) 
                             (h3 : ∃ n : ℤ, a + b = 2 * n + 1) : 
  ∃ p : ℤ, a * b = 2 * p := 
  sorry

end sum_odd_product_even_l2029_202972


namespace average_student_headcount_l2029_202985

theorem average_student_headcount :
  let count_0304 := 10500
  let count_0405 := 10700
  let count_0506 := 11300
  let total_count := count_0304 + count_0405 + count_0506
  let number_of_terms := 3
  let average := total_count / number_of_terms
  average = 10833 :=
by
  sorry

end average_student_headcount_l2029_202985


namespace pure_imaginary_value_l2029_202955

theorem pure_imaginary_value (a : ℝ) 
  (h1 : (a^2 - 3 * a + 2) = 0) 
  (h2 : (a - 2) ≠ 0) : a = 1 := sorry

end pure_imaginary_value_l2029_202955


namespace average_speed_of_car_l2029_202990

noncomputable def averageSpeed : ℚ := 
  let speed1 := 45     -- kph
  let distance1 := 15  -- km
  let speed2 := 55     -- kph
  let distance2 := 30  -- km
  let speed3 := 65     -- kph
  let time3 := 35 / 60 -- hours
  let speed4 := 52     -- kph
  let time4 := 20 / 60 -- hours
  let distance3 := speed3 * time3
  let distance4 := speed4 * time4
  let totalDistance := distance1 + distance2 + distance3 + distance4
  let time1 := distance1 / speed1
  let time2 := distance2 / speed2
  let totalTime := time1 + time2 + time3 + time4
  totalDistance / totalTime

theorem average_speed_of_car :
  abs (averageSpeed - 55.85) < 0.01 := 
  sorry

end average_speed_of_car_l2029_202990


namespace smallest_possible_value_of_M_l2029_202953

theorem smallest_possible_value_of_M :
  ∀ (a b c d e f : ℕ), a > 0 → b > 0 → c > 0 → d > 0 → e > 0 → f > 0 →
  a + b + c + d + e + f = 4020 →
  (∃ M : ℕ, M = max (a + b) (max (b + c) (max (c + d) (max (d + e) (e + f)))) ∧
    (∀ (M' : ℕ), (∀ (a b c d e f : ℕ), a > 0 → b > 0 → c > 0 → d > 0 → e > 0 → f > 0 →
      a + b + c + d + e + f = 4020 →
      M' = max (a + b) (max (b + c) (max (c + d) (max (d + e) (e + f)))) → M' ≥ 804) → M = 804)) := by
  sorry

end smallest_possible_value_of_M_l2029_202953


namespace flower_problem_l2029_202904

def totalFlowers (n_rows n_per_row : Nat) : Nat :=
  n_rows * n_per_row

def flowersCut (total percent_cut : Nat) : Nat :=
  total * percent_cut / 100

def flowersRemaining (total cut : Nat) : Nat :=
  total - cut

theorem flower_problem :
  let n_rows := 50
  let n_per_row := 400
  let percent_cut := 60
  let total := totalFlowers n_rows n_per_row
  let cut := flowersCut total percent_cut
  flowersRemaining total cut = 8000 :=
by
  sorry

end flower_problem_l2029_202904


namespace matrix_to_system_solution_l2029_202995

theorem matrix_to_system_solution :
  ∀ (x y : ℝ),
  (2 * x + y = 5) ∧ (x - 2 * y = 0) →
  3 * x - y = 5 :=
by
  sorry

end matrix_to_system_solution_l2029_202995


namespace rentExpenses_l2029_202958

noncomputable def monthlySalary : ℝ := 23000
noncomputable def milkExpenses : ℝ := 1500
noncomputable def groceriesExpenses : ℝ := 4500
noncomputable def educationExpenses : ℝ := 2500
noncomputable def petrolExpenses : ℝ := 2000
noncomputable def miscellaneousExpenses : ℝ := 5200
noncomputable def savings : ℝ := 2300

-- Calculating total non-rent expenses
noncomputable def totalNonRentExpenses : ℝ :=
  milkExpenses + groceriesExpenses + educationExpenses + petrolExpenses + miscellaneousExpenses

-- The rent expenses theorem
theorem rentExpenses : totalNonRentExpenses + savings + 5000 = monthlySalary :=
by sorry

end rentExpenses_l2029_202958


namespace turtles_remaining_proof_l2029_202978

noncomputable def turtles_original := 50
noncomputable def turtles_additional := 7 * turtles_original - 6
noncomputable def turtles_total_before_frightened := turtles_original + turtles_additional
noncomputable def turtles_frightened := (3 / 7) * turtles_total_before_frightened
noncomputable def turtles_remaining := turtles_total_before_frightened - turtles_frightened

theorem turtles_remaining_proof : turtles_remaining = 226 := by
  sorry

end turtles_remaining_proof_l2029_202978


namespace minimum_value_expression_l2029_202922

noncomputable def expression (a b c d : ℝ) : ℝ :=
  (a + b) / c + (a + c) / d + (b + d) / a + (c + d) / b

theorem minimum_value_expression (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  expression a b c d ≥ 8 :=
by
  -- Proof goes here
  sorry

end minimum_value_expression_l2029_202922


namespace price_of_orange_is_60_l2029_202956

theorem price_of_orange_is_60
  (x a o : ℕ)
  (h1 : 40 * a + x * o = 540)
  (h2 : a + o = 10)
  (h3 : 40 * a + x * (o - 5) = 240) :
  x = 60 :=
by
  sorry

end price_of_orange_is_60_l2029_202956


namespace Luca_weight_loss_per_year_l2029_202960

def Barbi_weight_loss_per_month : Real := 1.5
def months_in_a_year : Nat := 12
def Luca_years : Nat := 11
def extra_weight_Luca_lost : Real := 81

theorem Luca_weight_loss_per_year :
  (Barbi_weight_loss_per_month * months_in_a_year + extra_weight_Luca_lost) / Luca_years = 9 := by
  sorry

end Luca_weight_loss_per_year_l2029_202960


namespace solution_set_f_x_sq_gt_2f_x_plus_1_l2029_202999

noncomputable def f : ℝ → ℝ := sorry

theorem solution_set_f_x_sq_gt_2f_x_plus_1
  (h_domain : ∀ x, 0 < x → ∃ y, f y = f x)
  (h_func_equation : ∀ x y, 0 < x → 0 < y → f (x + y) = f x * f y)
  (h_greater_than_2 : ∀ x, 1 < x → f x > 2)
  (h_f2 : f 2 = 4) :
  ∀ x, x^2 > x + 2 → x > 2 :=
by
  intros x h
  sorry

end solution_set_f_x_sq_gt_2f_x_plus_1_l2029_202999


namespace sum_of_squares_first_15_l2029_202946

def sum_of_squares (n : ℕ) : ℕ := (n * (n + 1) * (2 * n + 1)) / 6

theorem sum_of_squares_first_15 : sum_of_squares 15 = 3720 :=
by
  sorry

end sum_of_squares_first_15_l2029_202946


namespace ratio_Sachin_Rahul_l2029_202945

-- Definitions: Sachin's age (S) is 63, and Sachin is younger than Rahul by 18 years.
def Sachin_age : ℕ := 63
def Rahul_age : ℕ := Sachin_age + 18

-- The problem: Prove the ratio of Sachin's age to Rahul's age is 7/9.
theorem ratio_Sachin_Rahul : (Sachin_age : ℚ) / (Rahul_age : ℚ) = 7 / 9 :=
by 
  -- The proof will go here, but we are skipping the proof as per the instructions.
  sorry

end ratio_Sachin_Rahul_l2029_202945


namespace swimmers_meet_l2029_202988

def time_to_meet (pool_length speed1 speed2 time: ℕ) : ℕ :=
  (time * (speed1 + speed2)) / pool_length

theorem swimmers_meet
  (pool_length : ℕ)
  (speed1 : ℕ)
  (speed2 : ℕ)
  (total_time : ℕ) :
  total_time = 12 * 60 →
  pool_length = 90 →
  speed1 = 3 →
  speed2 = 2 →
  time_to_meet pool_length speed1 speed2 total_time = 20 := by
  sorry

end swimmers_meet_l2029_202988


namespace solve_for_x_l2029_202926

theorem solve_for_x (x : ℝ) : 
  x - 3 * x + 5 * x = 150 → x = 50 :=
by
  intro h
  -- sorry to skip the proof
  sorry

end solve_for_x_l2029_202926


namespace total_money_taken_l2029_202982

def individual_bookings : ℝ := 12000
def group_bookings : ℝ := 16000
def returned_due_to_cancellations : ℝ := 1600

def total_taken (individual_bookings : ℝ) (group_bookings : ℝ) (returned_due_to_cancellations : ℝ) : ℝ :=
  (individual_bookings + group_bookings) - returned_due_to_cancellations

theorem total_money_taken :
  total_taken individual_bookings group_bookings returned_due_to_cancellations = 26400 := by
  sorry

end total_money_taken_l2029_202982


namespace probability_bob_wins_l2029_202992

theorem probability_bob_wins (P_lose : ℝ) (P_tie : ℝ) (h1 : P_lose = 5/8) (h2 : P_tie = 1/8) :
  (1 - P_lose - P_tie) = 1/4 :=
by
  sorry

end probability_bob_wins_l2029_202992


namespace domain_of_w_l2029_202907

theorem domain_of_w :
  {x : ℝ | x + (x - 1)^(1/3) + (8 - x)^(1/3) ≥ 0} = {x : ℝ | x ≥ 0} :=
by {
  sorry
}

end domain_of_w_l2029_202907


namespace find_ordered_pair_l2029_202917

theorem find_ordered_pair (a b : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0)
  (h₃ : (x : ℝ) → x^2 + 2 * a * x + b = 0 → x = a ∨ x = b) :
  (a, b) = (1, -3) :=
sorry

end find_ordered_pair_l2029_202917


namespace function_monotonic_decreasing_interval_l2029_202969

noncomputable def f (x : ℝ) := Real.sin (2 * x + Real.pi / 6)

theorem function_monotonic_decreasing_interval :
  ∀ x ∈ Set.Icc (Real.pi / 6) (2 * Real.pi / 3), 
  ∀ y ∈ Set.Icc (Real.pi / 6) (2 * Real.pi / 3), 
  (x ≤ y → f y ≤ f x) :=
by
  sorry

end function_monotonic_decreasing_interval_l2029_202969


namespace max_value_expression_l2029_202913

noncomputable def target_expr (x y z : ℝ) : ℝ :=
  (x^2 - x * y + y^2) * (y^2 - y * z + z^2) * (z^2 - z * x + x^2)

theorem max_value_expression (x y z : ℝ) (h : x + y + z = 3) (hxy : x = y) (hxz : 0 ≤ x) (hyz : 0 ≤ y) (hzz : 0 ≤ z) :
  target_expr x y z ≤ 9 / 4 := by
  sorry

end max_value_expression_l2029_202913


namespace minimize_potato_cost_l2029_202929

def potatoes_distribution (x1 x2 x3 : ℚ) : Prop :=
  x1 ≥ 0 ∧ x2 ≥ 0 ∧ x3 ≥ 0 ∧
  x1 + x2 + x3 = 12 ∧
  x1 + 4 * x2 + 3 * x3 ≤ 40 ∧
  x1 ≤ 10 ∧ x2 ≤ 8 ∧ x3 ≤ 6 ∧
  4 * x1 + 3 * x2 + 1 * x3 = (74 / 3)

theorem minimize_potato_cost :
  ∃ x1 x2 x3 : ℚ, potatoes_distribution x1 x2 x3 ∧ x1 = (2/3) ∧ x2 = (16/3) ∧ x3 = 6 :=
by
  sorry

end minimize_potato_cost_l2029_202929


namespace type_2004_A_least_N_type_B_diff_2004_l2029_202977

def game_type_A (N : ℕ) : Prop :=
  ∀ n, (1 ≤ n ∧ n ≤ N) → (n % 2 = 0 → false) 

def game_type_B (N : ℕ) : Prop :=
  ∃ n, (1 ≤ n ∧ n ≤ N) ∧ (n % 2 = 0 → true)


theorem type_2004_A : game_type_A 2004 :=
sorry

theorem least_N_type_B_diff_2004 : ∀ N, N > 2004 → game_type_B N → N = 2048 :=
sorry

end type_2004_A_least_N_type_B_diff_2004_l2029_202977


namespace number_of_birds_seen_l2029_202901

theorem number_of_birds_seen (dozens_seen : ℕ) (birds_per_dozen : ℕ) (h₀ : dozens_seen = 8) (h₁ : birds_per_dozen = 12) : dozens_seen * birds_per_dozen = 96 :=
by sorry

end number_of_birds_seen_l2029_202901


namespace max_segment_perimeter_l2029_202983

def isosceles_triangle (base height : ℝ) := true -- A realistic definition can define properties of an isosceles triangle

def equal_area_segments (triangle : isosceles_triangle 10 12) (n : ℕ) := true -- A realist definition can define cutting into equal area segments

noncomputable def perimeter_segment (base height : ℝ) (k : ℕ) (n : ℕ) : ℝ :=
  1 + Real.sqrt (height^2 + (base / n * k)^2) + Real.sqrt (height^2 + (base / n * (k + 1))^2)

theorem max_segment_perimeter (base height : ℝ) (n : ℕ) (h_base : base = 10) (h_height : height = 12) (h_segments : n = 10) :
  ∃ k, k ∈ Finset.range n ∧ perimeter_segment base height k n = 31.62 :=
by
  sorry

end max_segment_perimeter_l2029_202983


namespace symmetric_points_addition_l2029_202916

theorem symmetric_points_addition (m n : ℤ) (h₁ : m = 2) (h₂ : n = -3) : m + n = -1 := by
  rw [h₁, h₂]
  norm_num

end symmetric_points_addition_l2029_202916


namespace arthur_walked_total_miles_l2029_202930

def blocks_east := 8
def blocks_north := 15
def blocks_west := 3
def block_length := 1/2

def total_blocks := blocks_east + blocks_north + blocks_west
def total_miles := total_blocks * block_length

theorem arthur_walked_total_miles : total_miles = 13 := by
  sorry

end arthur_walked_total_miles_l2029_202930


namespace maxwell_distance_traveled_l2029_202941

theorem maxwell_distance_traveled
  (distance_between_homes : ℕ)
  (maxwell_speed : ℕ)
  (brad_speed : ℕ)
  (meeting_time : ℕ)
  (h1 : distance_between_homes = 72)
  (h2 : maxwell_speed = 6)
  (h3 : brad_speed = 12)
  (h4 : meeting_time = distance_between_homes / (maxwell_speed + brad_speed)) :
  maxwell_speed * meeting_time = 24 :=
by
  sorry

end maxwell_distance_traveled_l2029_202941


namespace find_radius_yz_l2029_202980

-- Define the setup for the centers of the circles and their radii
def circle_with_center (c : Type*) (radius : ℝ) : Prop := sorry
def tangent_to (c₁ c₂ : Type*) : Prop := sorry

-- Given conditions
variable (O X Y Z : Type*)
variable (r : ℝ)
variable (Xe_radius : circle_with_center X 1)
variable (O_radius : circle_with_center O 2)
variable (XtangentO : tangent_to X O)
variable (YtangentO : tangent_to Y O)
variable (YtangentX : tangent_to Y X)
variable (YtangentZ : tangent_to Y Z)
variable (ZtangentO : tangent_to Z O)
variable (ZtangentX : tangent_to Z X)
variable (ZtangentY : tangent_to Z Y)

-- The theorem to prove
theorem find_radius_yz :
  r = 8 / 9 := sorry

end find_radius_yz_l2029_202980


namespace product_not_divisible_by_prime_l2029_202936

theorem product_not_divisible_by_prime (p a b : ℕ) (hp : Prime p) (ha : 1 ≤ a) (hpa : a < p) (hb : 1 ≤ b) (hpb : b < p) : ¬ (p ∣ (a * b)) :=
by
  sorry

end product_not_divisible_by_prime_l2029_202936


namespace customer_purchases_90_percent_l2029_202962

variable (P Q : ℝ) 

theorem customer_purchases_90_percent (price_increase_expenditure_diff : 
  (1.25 * P * R / 100 * Q = 1.125 * P * Q)) : 
  R = 90 := 
by 
  sorry

end customer_purchases_90_percent_l2029_202962


namespace car_storm_distance_30_l2029_202957

noncomputable def car_position (t : ℝ) : ℝ × ℝ :=
  (0, 3/4 * t)

noncomputable def storm_center (t : ℝ) : ℝ × ℝ :=
  (150 - (3/4 / Real.sqrt 2) * t, -(3/4 / Real.sqrt 2) * t)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

theorem car_storm_distance_30 :
  ∃ (t : ℝ), distance (car_position t) (storm_center t) = 30 :=
sorry

end car_storm_distance_30_l2029_202957


namespace opposite_of_one_half_l2029_202998

theorem opposite_of_one_half : -((1:ℚ)/2) = -1/2 := by
  -- Skipping the proof using sorry
  sorry

end opposite_of_one_half_l2029_202998


namespace solve_equation_l2029_202925

noncomputable def equation (x : ℝ) : Prop := x * (x - 2) + x - 2 = 0

theorem solve_equation : ∀ x, equation x ↔ (x = 2 ∨ x = -1) :=
by sorry

end solve_equation_l2029_202925


namespace total_chapters_eq_l2029_202993

-- Definitions based on conditions
def days : ℕ := 664
def chapters_per_day : ℕ := 332

-- Theorem to prove the total number of chapters in the book is 220448
theorem total_chapters_eq : (chapters_per_day * days = 220448) :=
by
  sorry

end total_chapters_eq_l2029_202993


namespace remainder_of_3_pow_600_mod_19_l2029_202920

theorem remainder_of_3_pow_600_mod_19 :
  (3 ^ 600) % 19 = 11 :=
sorry

end remainder_of_3_pow_600_mod_19_l2029_202920


namespace ceilings_left_to_paint_l2029_202996

theorem ceilings_left_to_paint
    (floors : ℕ)
    (rooms_per_floor : ℕ)
    (ceilings_painted_this_week : ℕ)
    (hallways_per_floor : ℕ)
    (hallway_ceilings_per_hallway : ℕ)
    (ceilings_painted_ratio : ℚ)
    : floors = 4
    → rooms_per_floor = 7
    → ceilings_painted_this_week = 12
    → hallways_per_floor = 1
    → hallway_ceilings_per_hallway = 1
    → ceilings_painted_ratio = 1 / 4
    → (floors * rooms_per_floor + floors * hallways_per_floor * hallway_ceilings_per_hallway 
        - ceilings_painted_this_week 
        - (ceilings_painted_ratio * ceilings_painted_this_week + floors * hallway_ceilings_per_hallway) = 13) :=
by
  intros
  sorry

end ceilings_left_to_paint_l2029_202996


namespace pipe_length_difference_l2029_202921

theorem pipe_length_difference (total_length shorter_piece : ℕ) (h1 : total_length = 68) (h2 : shorter_piece = 28) : 
  total_length - shorter_piece * 2 = 12 := 
sorry

end pipe_length_difference_l2029_202921


namespace Pat_height_l2029_202963

noncomputable def Pat_first_day_depth := 40 -- in cm
noncomputable def Mat_second_day_depth := 3 * Pat_first_day_depth -- Mat digs 3 times the depth on the second day
noncomputable def Pat_third_day_depth := Mat_second_day_depth - Pat_first_day_depth -- Pat digs the same amount on the third day
noncomputable def Total_depth_after_third_day := Mat_second_day_depth + Pat_third_day_depth -- Total depth after third day's digging
noncomputable def Depth_above_Pat_head := 50 -- The depth above Pat's head

theorem Pat_height : Total_depth_after_third_day - Depth_above_Pat_head = 150 := by
  sorry

end Pat_height_l2029_202963


namespace remaining_distance_l2029_202934

theorem remaining_distance (S u : ℝ) (h1 : S / (2 * u) + 24 = S) (h2 : S * u / 2 + 15 = S) : ∃ x : ℝ, x = 8 :=
by
  -- Proof steps would go here
  sorry

end remaining_distance_l2029_202934


namespace quadratic_has_single_solution_l2029_202981

theorem quadratic_has_single_solution (q : ℚ) (h : q ≠ 0) :
  (∀ x : ℚ, q * x^2 - 16 * x + 9 = 0 → q = 64 / 9) := by
  sorry

end quadratic_has_single_solution_l2029_202981


namespace crow_eating_time_l2029_202966

theorem crow_eating_time (n : ℕ) (h : ∀ t : ℕ, t = (n / 5) → t = 4) : (4 + (4 / 5) = 4.8) :=
by
  sorry

end crow_eating_time_l2029_202966


namespace exists_A_for_sqrt_d_l2029_202948

def is_not_perfect_square (d : ℕ) : Prop := ∀ m : ℕ, m * m ≠ d

def s (d n : ℕ) : ℕ := 
  -- count number of 1's in the first n digits of binary representation of √d
  sorry 

theorem exists_A_for_sqrt_d (d : ℕ) (h : is_not_perfect_square d) :
  ∃ A : ℕ, ∀ n ≥ A, s d n > Int.sqrt (2 * n) - 2 :=
sorry

end exists_A_for_sqrt_d_l2029_202948


namespace sport_flavoring_to_water_ratio_l2029_202903

/-- The ratio by volume of flavoring to corn syrup to water in the 
standard formulation is 1:12:30. The sport formulation has a ratio 
of flavoring to corn syrup three times as great as in the standard formulation. 
A large bottle of the sport formulation contains 4 ounces of corn syrup and 
60 ounces of water. Prove that the ratio of the amount of flavoring to water 
in the sport formulation compared to the standard formulation is 1:2. -/
theorem sport_flavoring_to_water_ratio 
    (standard_flavoring : ℝ) 
    (standard_corn_syrup : ℝ) 
    (standard_water : ℝ) : 
  standard_flavoring = 1 → standard_corn_syrup = 12 → 
  standard_water = 30 → 
  ∃ sport_flavoring : ℝ, 
  ∃ sport_corn_syrup : ℝ, 
  ∃ sport_water : ℝ, 
  sport_corn_syrup = 4 ∧ 
  sport_water = 60 ∧ 
  (sport_flavoring / sport_water) = (standard_flavoring / standard_water) / 2 :=
by
  sorry

end sport_flavoring_to_water_ratio_l2029_202903


namespace derivative_at_minus_one_l2029_202908
open Real

def f (x : ℝ) : ℝ := (1 + x) * (2 + x^2)^(1 / 2) * (3 + x^3)^(1 / 3)

theorem derivative_at_minus_one : deriv f (-1) = sqrt 3 * 2^(1 / 3) :=
by sorry

end derivative_at_minus_one_l2029_202908


namespace number_of_shelves_l2029_202943

-- Given conditions
def booksBeforeTrip : ℕ := 56
def booksBought : ℕ := 26
def avgBooksPerShelf : ℕ := 20
def booksLeftOver : ℕ := 2
def totalBooks : ℕ := booksBeforeTrip + booksBought

-- Statement to prove
theorem number_of_shelves :
  totalBooks - booksLeftOver = 80 →
  80 / avgBooksPerShelf = 4 := by
  intros h
  sorry

end number_of_shelves_l2029_202943


namespace probability_two_white_balls_l2029_202905

noncomputable def probability_of_two_white_balls (total_balls white_balls black_balls: ℕ) : ℚ :=
  if white_balls + black_balls = total_balls ∧ total_balls = 15 ∧ white_balls = 7 ∧ black_balls = 8 then
    (white_balls / total_balls) * ((white_balls - 1) / (total_balls - 1))
  else 0

theorem probability_two_white_balls : 
  probability_of_two_white_balls 15 7 8 = 1/5
:= sorry

end probability_two_white_balls_l2029_202905


namespace neg_abs_neg_three_l2029_202910

theorem neg_abs_neg_three : -|(-3)| = -3 := 
by
  sorry

end neg_abs_neg_three_l2029_202910


namespace donuts_left_l2029_202971

def initial_donuts : ℕ := 50
def after_bill_eats (initial : ℕ) : ℕ := initial - 2
def after_secretary_takes (remaining_after_bill : ℕ) : ℕ := remaining_after_bill - 4
def coworkers_take (remaining_after_secretary : ℕ) : ℕ := remaining_after_secretary / 2
def final_donuts (initial : ℕ) : ℕ :=
  let remaining_after_bill := after_bill_eats initial
  let remaining_after_secretary := after_secretary_takes remaining_after_bill
  remaining_after_secretary - coworkers_take remaining_after_secretary

theorem donuts_left : final_donuts 50 = 22 := by
  sorry

end donuts_left_l2029_202971


namespace geo_seq_value_l2029_202970

variable (a : ℕ → ℝ)
variable (a_2 : a 2 = 2) 
variable (a_4 : a 4 = 8)
variable (geo_prop : a 2 * a 6 = (a 4) ^ 2)

theorem geo_seq_value : a 6 = 32 := 
by 
  sorry

end geo_seq_value_l2029_202970


namespace total_trolls_l2029_202973

theorem total_trolls (P B T : ℕ) (hP : P = 6) (hB : B = 4 * P - 6) (hT : T = B / 2) : P + B + T = 33 := by
  sorry

end total_trolls_l2029_202973


namespace rectangle_area_diff_l2029_202928

theorem rectangle_area_diff :
  ∀ (l w : ℕ), (2 * l + 2 * w = 60) → (∃ A_max A_min : ℕ, 
    A_max = (l * (30 - l)) ∧ A_min = (min (1 * (30 - 1)) (29 * (30 - 29))) ∧ (A_max - A_min = 196)) :=
by
  intros l w h
  use 15 * 15, min (1 * 29) (29 * 1)
  sorry

end rectangle_area_diff_l2029_202928


namespace train_speed_with_coaches_l2029_202976

theorem train_speed_with_coaches (V₀ : ℝ) (V₉ V₁₆ : ℝ) (k : ℝ) :
  V₀ = 30 → V₁₆ = 14 → V₉ = 30 - k * (9: ℝ) ^ (1/2: ℝ) ∧ V₁₆ = 30 - k * (16: ℝ) ^ (1/2: ℝ) →
  V₉ = 18 :=
by sorry

end train_speed_with_coaches_l2029_202976


namespace find_n_l2029_202942

theorem find_n (n : ℕ) : (8 : ℝ)^(1/3) = (2 : ℝ)^n → n = 1 := by
  sorry

end find_n_l2029_202942


namespace vinegar_final_percentage_l2029_202949

def vinegar_percentage (volume1 volume2 : ℕ) (percent1 percent2 : ℚ) : ℚ :=
  let vinegar1 := volume1 * percent1 / 100
  let vinegar2 := volume2 * percent2 / 100
  (vinegar1 + vinegar2) / (volume1 + volume2) * 100

theorem vinegar_final_percentage:
  vinegar_percentage 128 128 8 13 = 10.5 :=
  sorry

end vinegar_final_percentage_l2029_202949


namespace village_male_population_l2029_202944

theorem village_male_population (total_population parts male_parts : ℕ) (h1 : total_population = 600) (h2 : parts = 4) (h3 : male_parts = 2) :
  male_parts * (total_population / parts) = 300 :=
by
  -- We are stating the problem as per the given conditions
  sorry

end village_male_population_l2029_202944


namespace line_tangent_to_ellipse_l2029_202923

theorem line_tangent_to_ellipse (m : ℝ) : 
  (∀ x y : ℝ, y = mx + 2 → x^2 + 9 * y^2 = 9 → ∃ u, y = u) → m^2 = 1 / 3 := 
by
  intro h
  sorry

end line_tangent_to_ellipse_l2029_202923


namespace max_distance_from_B_to_P_l2029_202965

structure Point where
  x : ℝ
  y : ℝ

def A : Point := { x := -4, y := 1 }
def P : Point := { x := 3, y := -1 }

def line_l (m : ℝ) (pt : Point) : Prop :=
  (2 * m + 1) * pt.x - (m - 1) * pt.y - m - 5 = 0

theorem max_distance_from_B_to_P :
  ∃ B : Point, A = { x := -4, y := 1 } → 
               (∀ m : ℝ, line_l m B) →
               ∃ d, d = 5 + Real.sqrt 10 :=
sorry

end max_distance_from_B_to_P_l2029_202965


namespace more_girls_than_boys_l2029_202947

variables (boys girls : ℕ)

def ratio_condition : Prop := (3 * girls = 4 * boys)
def total_students_condition : Prop := (boys + girls = 42)

theorem more_girls_than_boys (h1 : ratio_condition boys girls) (h2 : total_students_condition boys girls) :
  (girls - boys = 6) :=
sorry

end more_girls_than_boys_l2029_202947


namespace arithmetic_geometric_sequence_a1_l2029_202914

theorem arithmetic_geometric_sequence_a1 (a : ℕ → ℚ)
  (h1 : a 1 + a 6 = 11)
  (h2 : a 3 * a 4 = 32 / 9) :
  a 1 = 32 / 3 ∨ a 1 = 1 / 3 :=
sorry

end arithmetic_geometric_sequence_a1_l2029_202914


namespace boat_distance_along_stream_l2029_202927

theorem boat_distance_along_stream
  (distance_against_stream : ℝ)
  (speed_still_water : ℝ)
  (time : ℝ)
  (v_s : ℝ)
  (H1 : distance_against_stream = 5)
  (H2 : speed_still_water = 6)
  (H3 : time = 1)
  (H4 : speed_still_water - v_s = distance_against_stream / time) :
  (speed_still_water + v_s) * time = 7 :=
by
  -- Sorry to skip proof
  sorry

end boat_distance_along_stream_l2029_202927


namespace rest_area_milepost_l2029_202909

theorem rest_area_milepost : 
  let fifth_exit := 30
  let fifteenth_exit := 210
  (3 / 5) * (fifteenth_exit - fifth_exit) + fifth_exit = 138 := 
by 
  let fifth_exit := 30
  let fifteenth_exit := 210
  sorry

end rest_area_milepost_l2029_202909


namespace lunks_needed_for_apples_l2029_202915

theorem lunks_needed_for_apples :
  (∀ l k a : ℕ, (4 * k = 2 * l) ∧ (3 * a = 5 * k ) → ∃ l', l' = (24 * l / 4)) :=
by
  intros l k a h
  obtain ⟨h1, h2⟩ := h
  have k_for_apples := 3 * a / 5
  have l_for_kunks := 4 * k / 2
  sorry

end lunks_needed_for_apples_l2029_202915


namespace acute_angle_vector_range_l2029_202937

theorem acute_angle_vector_range (m : ℝ) (a b : ℝ × ℝ) 
  (h1 : a = (1, 2)) 
  (h2 : b = (4, m)) 
  (acute : (a.1 * b.1 + a.2 * b.2) > 0) : 
  (m > -2) ∧ (m ≠ 8) := 
by 
  sorry

end acute_angle_vector_range_l2029_202937


namespace negation_of_proposition_l2029_202994

theorem negation_of_proposition :
  (¬ (∀ x : ℝ, x ≥ 1 → x^2 ≥ 1)) ↔ (∃ x : ℝ, x ≥ 1 ∧ x^2 < 1) := 
sorry

end negation_of_proposition_l2029_202994


namespace derivative_at_pi_div_3_l2029_202951

noncomputable def f (x : ℝ) : ℝ := x * Real.cos x - Real.sin x

theorem derivative_at_pi_div_3 : 
  deriv f (Real.pi / 3) = - (Real.sqrt 3 * Real.pi / 6) :=
by
  sorry

end derivative_at_pi_div_3_l2029_202951


namespace total_pages_in_scifi_section_l2029_202935

theorem total_pages_in_scifi_section : 
  let books := 8
  let pages_per_book := 478
  books * pages_per_book = 3824 := 
by
  sorry

end total_pages_in_scifi_section_l2029_202935


namespace fewest_fence_posts_l2029_202967

def fence_posts (length_wide short_side long_side : ℕ) (post_interval : ℕ) : ℕ :=
  let wide_side_posts := (long_side / post_interval) + 1
  let short_side_posts := (short_side / post_interval)
  wide_side_posts + 2 * short_side_posts

theorem fewest_fence_posts : fence_posts 40 10 100 10 = 19 :=
  by
    -- The proof will be completed here
    sorry

end fewest_fence_posts_l2029_202967


namespace factor_expression_l2029_202968

theorem factor_expression (a b c : ℝ) :
  a * (b - c)^4 + b * (c - a)^4 + c * (a - b)^4 =
  (a - b)^2 * (b - c)^2 * (c - a)^2 * (a + b + c) :=
sorry

end factor_expression_l2029_202968


namespace second_term_is_correct_l2029_202919

noncomputable def arithmetic_sequence_second_term (a d : ℤ) (h1 : a + 9 * d = 15) (h2 : a + 10 * d = 18) : ℤ :=
  a + d

theorem second_term_is_correct (a d : ℤ) (h1 : a + 9 * d = 15) (h2 : a + 10 * d = 18) :
  arithmetic_sequence_second_term a d h1 h2 = -9 :=
sorry

end second_term_is_correct_l2029_202919


namespace find_c_d_l2029_202933

noncomputable def g (c d x : ℝ) : ℝ := c * x^3 + 5 * x^2 + d * x + 7

theorem find_c_d : ∃ (c d : ℝ), 
  (g c d 2 = 11) ∧ (g c d (-3) = 134) ∧ c = -35 / 13 ∧ d = 16 / 13 :=
  by
  sorry

end find_c_d_l2029_202933


namespace parallelogram_area_example_l2029_202959

noncomputable def area_parallelogram (A B C D : (ℝ × ℝ)) : ℝ := 
  0.5 * |(A.1 * B.2 + B.1 * C.2 + C.1 * D.2 + D.1 * A.2) - (A.2 * B.1 + B.2 * C.1 + C.2 * D.1 + D.2 * A.1)|

theorem parallelogram_area_example : 
  let A := (0, 0)
  let B := (20, 0)
  let C := (25, 7)
  let D := (5, 7)
  area_parallelogram A B C D = 140 := 
by
  sorry

end parallelogram_area_example_l2029_202959


namespace range_of_a_l2029_202902

noncomputable def setA : Set ℝ := {x | x^2 + 4 * x = 0}
noncomputable def setB (a : ℝ) : Set ℝ := {x | x^2 + a * x + a = 0}

theorem range_of_a :
  ∀ a : ℝ, (setA ∪ setB a) = setA ↔ 0 ≤ a ∧ a < 4 :=
by sorry

end range_of_a_l2029_202902


namespace max_valid_words_for_AU_language_l2029_202906

noncomputable def maxValidWords : ℕ :=
  2^14 - 128

theorem max_valid_words_for_AU_language 
  (letters : Finset (String)) (validLengths : Set ℕ) (noConcatenation : Prop) :
  letters = {"a", "u"} ∧ validLengths = {n | 1 ≤ n ∧ n ≤ 13} ∧ noConcatenation →
  maxValidWords = 16256 :=
by
  sorry

end max_valid_words_for_AU_language_l2029_202906


namespace find_constant_a_range_of_f_l2029_202911

noncomputable def f (a x : ℝ) : ℝ :=
  2 * a * (Real.sin x)^2 + 2 * (Real.sin x) * (Real.cos x) - a

theorem find_constant_a (h : f a 0 = -Real.sqrt 3) : a = Real.sqrt 3 := by
  sorry

theorem range_of_f (a : ℝ) (h : a = Real.sqrt 3) (x : ℝ) (hx : 0 ≤ x ∧ x ≤ Real.pi / 2) :
  f a x ∈ Set.Icc (-Real.sqrt 3) 2 := by
  sorry

end find_constant_a_range_of_f_l2029_202911


namespace product_of_fractions_is_25_div_324_l2029_202997

noncomputable def product_of_fractions : ℚ := 
  (10 / 6) * (4 / 20) * (20 / 12) * (16 / 32) * 
  (40 / 24) * (8 / 40) * (60 / 36) * (32 / 64)

theorem product_of_fractions_is_25_div_324 : product_of_fractions = 25 / 324 := 
  sorry

end product_of_fractions_is_25_div_324_l2029_202997


namespace range_of_a_for_three_distinct_zeros_l2029_202964

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^3 - 3 * x + a

theorem range_of_a_for_three_distinct_zeros : 
  ∀ a : ℝ, (∀ x y : ℝ, x ≠ y → f x a = 0 → f y a = 0 → (f (1:ℝ) a < 0 ∧ f (-1:ℝ) a > 0)) ↔ (-2 < a ∧ a < 2) := 
by
  sorry

end range_of_a_for_three_distinct_zeros_l2029_202964


namespace lemons_required_for_new_recipe_l2029_202954

noncomputable def lemons_needed_to_make_gallons (lemons_original : ℕ) (gallons_original : ℕ) (additional_lemons : ℕ) (additional_gallons : ℕ) (gallons_new : ℕ) : ℝ :=
  let lemons_per_gallon := (lemons_original : ℝ) / (gallons_original : ℝ)
  let additional_lemons_per_gallon := (additional_lemons : ℝ) / (additional_gallons : ℝ)
  let total_lemons_per_gallon := lemons_per_gallon + additional_lemons_per_gallon
  total_lemons_per_gallon * (gallons_new : ℝ)

theorem lemons_required_for_new_recipe : lemons_needed_to_make_gallons 36 48 2 6 18 = 19.5 :=
by
  sorry

end lemons_required_for_new_recipe_l2029_202954


namespace initial_pennies_in_each_compartment_l2029_202938

theorem initial_pennies_in_each_compartment (x : ℕ) (h : 12 * (x + 6) = 96) : x = 2 :=
by sorry

end initial_pennies_in_each_compartment_l2029_202938


namespace max_plus_min_value_of_y_eq_neg4_l2029_202912

noncomputable def y (x : ℝ) : ℝ := (2 * (Real.sin x) ^ 2 + Real.sin (3 * x / 2) - 4) / ((Real.sin x) ^ 2 + 2 * (Real.cos x) ^ 2)

theorem max_plus_min_value_of_y_eq_neg4 (M m : ℝ) (hM : ∃ x : ℝ, y x = M) (hm : ∃ x : ℝ, y x = m) :
  M + m = -4 := sorry

end max_plus_min_value_of_y_eq_neg4_l2029_202912


namespace range_of_a_l2029_202932

theorem range_of_a : (∀ x : ℝ, x^2 + (a-1)*x + 1 > 0) ↔ (-1 < a ∧ a < 3) := by
  sorry

end range_of_a_l2029_202932


namespace line_intersection_points_l2029_202974

def line_intersects_axes (x y : ℝ) : Prop :=
  (4 * y - 5 * x = 20)

theorem line_intersection_points :
  ∃ p1 p2, line_intersects_axes p1.1 p1.2 ∧ line_intersects_axes p2.1 p2.2 ∧
    (p1 = (-4, 0) ∧ p2 = (0, 5)) :=
by
  sorry

end line_intersection_points_l2029_202974


namespace quadratic_real_roots_range_l2029_202986

-- Given conditions and definitions
def discriminant (a b c : ℝ) : ℝ :=
  b^2 - 4 * a * c

def equation_has_real_roots (a b c : ℝ) : Prop :=
  discriminant a b c ≥ 0

-- Problem translated into a Lean statement
theorem quadratic_real_roots_range (m : ℝ) :
  equation_has_real_roots 1 (-2) (-m) ↔ m ≥ -1 :=
by
  sorry

end quadratic_real_roots_range_l2029_202986


namespace prove_option_d_l2029_202984

-- Definitions of conditions
variables (a b : ℝ)
variable (h_nonzero : a ≠ 0 ∧ b ≠ 0)
variable (h_lt : a < b)

-- The theorem to be proved
theorem prove_option_d : a^3 < b^3 :=
sorry

end prove_option_d_l2029_202984


namespace range_of_b_l2029_202900

noncomputable def f : ℝ → ℝ
| x => if x < -1/2 then (2*x + 1) / (x^2) else x + 1

def g (x : ℝ) : ℝ := x^2 - 4*x - 4

theorem range_of_b (a b : ℝ) (h : f a + g b = 0) : -1 <= b ∧ b <= 5 :=
sorry

end range_of_b_l2029_202900


namespace amount_borrowed_eq_4137_84_l2029_202940

noncomputable def compound_interest (initial : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  initial * (1 + rate/100) ^ time

theorem amount_borrowed_eq_4137_84 :
  ∃ P : ℝ, 
    (compound_interest (compound_interest (compound_interest P 6 3) 8 4) 10 2 = 8110) 
    ∧ (P = 4137.84) :=
by
  sorry

end amount_borrowed_eq_4137_84_l2029_202940


namespace option_d_can_form_triangle_l2029_202989

noncomputable def satisfies_triangle_inequality (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem option_d_can_form_triangle : satisfies_triangle_inequality 2 3 4 :=
by {
  -- Using the triangle inequality theorem to check
  sorry
}

end option_d_can_form_triangle_l2029_202989
