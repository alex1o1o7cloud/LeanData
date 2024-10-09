import Mathlib

namespace carpet_breadth_l1809_180903

theorem carpet_breadth
  (b : ℝ)
  (h1 : ∀ b, ∃ l, l = 1.44 * b)
  (h2 : 4082.4 = 45 * ((1.40 * l) * (1.25 * b)))
  : b = 6.08 :=
by
  sorry

end carpet_breadth_l1809_180903


namespace find_b_perpendicular_lines_l1809_180974

variable (b : ℝ)

theorem find_b_perpendicular_lines :
  (2 * b + (-4) * 3 + 7 * (-1) = 0) → b = 19 / 2 := 
by
  intro h
  sorry

end find_b_perpendicular_lines_l1809_180974


namespace general_term_sequence_l1809_180993

noncomputable def a (t : ℝ) (n : ℕ) : ℝ :=
if h : t ≠ 1 then (2 * (t^n - 1) / n) - 1 else 0

theorem general_term_sequence (t : ℝ) (n : ℕ) (hn : n ≠ 0) (h : t ≠ 1) :
  a t (n+1) = (2 * (t^(n+1) - 1) / (n+1)) - 1 := 
sorry

end general_term_sequence_l1809_180993


namespace texas_california_plate_diff_l1809_180913

def california_plates := 26^3 * 10^3
def texas_plates := 26^3 * 10^4
def plates_difference := texas_plates - california_plates

theorem texas_california_plate_diff :
  plates_difference = 158184000 :=
by sorry

end texas_california_plate_diff_l1809_180913


namespace solve_trig_eq_l1809_180977

theorem solve_trig_eq (k : ℤ) : 
  ∃ x, 12 * Real.sin x - 5 * Real.cos x = 13 ∧ x = Real.pi / 2 + Real.arctan (5 / 12) + 2 * k * Real.pi :=
sorry

end solve_trig_eq_l1809_180977


namespace min_value_of_expr_l1809_180939

theorem min_value_of_expr {a b : ℝ} (ha : a > 0) (hb : b > 0) (h : a + b = (1 / a) + (1 / b)) :
  ∃ x : ℝ, x = (1 / a) + (2 / b) ∧ x = 2 * Real.sqrt 2 :=
sorry

end min_value_of_expr_l1809_180939


namespace probability_at_8_10_probability_at_8_10_through_5_6_probability_at_8_10_within_circle_l1809_180986

-- Definitions based on the conditions laid out in the problem
def fly_paths (n_right n_up : ℕ) : ℕ :=
  (Nat.factorial (n_right + n_up)) / ((Nat.factorial n_right) * (Nat.factorial n_up))

-- Probability for part a
theorem probability_at_8_10 : 
  (fly_paths 8 10) / (2 ^ 18) = (Nat.choose 18 8 : ℚ) / 2 ^ 18 := 
sorry

-- Probability for part b
theorem probability_at_8_10_through_5_6 :
  ((fly_paths 5 6) * (fly_paths 1 0) * (fly_paths 2 4)) / (2 ^ 18) = (6930 : ℚ) / 2 ^ 18 :=
sorry

-- Probability for part c
theorem probability_at_8_10_within_circle :
  (2 * fly_paths 2 7 * fly_paths 6 3 + 2 * fly_paths 3 6 * fly_paths 5 3 + (fly_paths 4 6) ^ 2) / (2 ^ 18) = 
  (2 * Nat.choose 9 2 * Nat.choose 9 6 + 2 * Nat.choose 9 3 * Nat.choose 9 5 + (Nat.choose 9 4) ^ 2 : ℚ) / 2 ^ 18 :=
sorry

end probability_at_8_10_probability_at_8_10_through_5_6_probability_at_8_10_within_circle_l1809_180986


namespace correct_calculation_l1809_180947

variable (n : ℕ)
variable (h1 : 63 + n = 70)

theorem correct_calculation : 36 * n = 252 :=
by
  -- Here we will need the Lean proof, which we skip using sorry
  sorry

end correct_calculation_l1809_180947


namespace equation_of_circle_passing_through_points_l1809_180989

theorem equation_of_circle_passing_through_points :
  (∃ D E F : ℝ, 
    (∀ x y : ℝ, x = 0 → y = 0 → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (∀ x y : ℝ, x = 4 → y = 0 → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (∀ x y : ℝ, x = -1 → y = 1 → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (x^2 + y^2 + D * x + E * y + F = 0 ↔ x^2 + y^2 - 4 * x - 6 * y = 0)) :=
sorry

end equation_of_circle_passing_through_points_l1809_180989


namespace louie_share_of_pie_l1809_180988

def fraction_of_pie_taken_home (total_pie : ℚ) (shares : ℚ) : ℚ :=
  2 * (total_pie / shares)

theorem louie_share_of_pie : fraction_of_pie_taken_home (8 / 9) 4 = 4 / 9 := 
by 
  sorry

end louie_share_of_pie_l1809_180988


namespace unique_prime_sum_and_diff_l1809_180922

noncomputable def is_prime (n : ℕ) : Prop := Nat.Prime n

noncomputable def is_sum_of_two_primes (p : ℕ) : Prop :=
  ∃ q1 q2 : ℕ, is_prime q1 ∧ is_prime q2 ∧ p = q1 + q2

noncomputable def is_diff_of_two_primes (p : ℕ) : Prop :=
  ∃ q3 q4 : ℕ, is_prime q3 ∧ is_prime q4 ∧ q3 > q4 ∧ p = q3 - q4

theorem unique_prime_sum_and_diff :
  ∀ p : ℕ, is_prime p ∧ is_sum_of_two_primes p ∧ is_diff_of_two_primes p ↔ p = 5 := 
by
  sorry

end unique_prime_sum_and_diff_l1809_180922


namespace club_members_remainder_l1809_180907

theorem club_members_remainder (N : ℕ) (h1 : 50 < N) (h2 : N < 80)
  (h3 : N % 5 = 0) (h4 : N % 8 = 0 ∨ N % 7 = 0) :
  N % 9 = 6 ∨ N % 9 = 7 := by
  sorry

end club_members_remainder_l1809_180907


namespace boy_overall_average_speed_l1809_180965

noncomputable def total_distance : ℝ := 100
noncomputable def distance1 : ℝ := 15
noncomputable def speed1 : ℝ := 12

noncomputable def distance2 : ℝ := 20
noncomputable def speed2 : ℝ := 8

noncomputable def distance3 : ℝ := 10
noncomputable def speed3 : ℝ := 25

noncomputable def distance4 : ℝ := 15
noncomputable def speed4 : ℝ := 18

noncomputable def distance5 : ℝ := 20
noncomputable def speed5 : ℝ := 10

noncomputable def distance6 : ℝ := 20
noncomputable def speed6 : ℝ := 22

noncomputable def time1 : ℝ := distance1 / speed1
noncomputable def time2 : ℝ := distance2 / speed2
noncomputable def time3 : ℝ := distance3 / speed3
noncomputable def time4 : ℝ := distance4 / speed4
noncomputable def time5 : ℝ := distance5 / speed5
noncomputable def time6 : ℝ := distance6 / speed6

noncomputable def total_time : ℝ := time1 + time2 + time3 + time4 + time5 + time6

noncomputable def overall_average_speed : ℝ := total_distance / total_time

theorem boy_overall_average_speed : overall_average_speed = 100 / (15 / 12 + 20 / 8 + 10 / 25 + 15 / 18 + 20 / 10 + 20 / 22) :=
by
  sorry

end boy_overall_average_speed_l1809_180965


namespace mangoes_total_l1809_180936

theorem mangoes_total (M A : ℕ) 
  (h1 : A = 4 * M) 
  (h2 : A = 60) :
  A + M = 75 :=
by
  sorry

end mangoes_total_l1809_180936


namespace calculate_f_2015_l1809_180953

noncomputable def f : ℝ → ℝ := sorry

-- Define the odd function property
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

-- Define the periodic function property with period 4
def periodic_4 (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x + 4) = f x

-- Define the given condition for the interval (0, 2)
def interval_condition (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, 0 < x ∧ x < 2 → f x = 2 * x ^ 2

theorem calculate_f_2015
  (odd_f : odd_function f)
  (periodic_f : periodic_4 f)
  (interval_f : interval_condition f) :
  f 2015 = -2 :=
sorry

end calculate_f_2015_l1809_180953


namespace kristy_gave_to_brother_l1809_180934

def total_cookies : Nat := 22
def kristy_ate : Nat := 2
def first_friend_took : Nat := 3
def second_friend_took : Nat := 5
def third_friend_took : Nat := 5
def cookies_left : Nat := 6

theorem kristy_gave_to_brother :
  kristy_ate + first_friend_took + second_friend_took + third_friend_took = 15 ∧
  total_cookies - cookies_left - (kristy_ate + first_friend_took + second_friend_took + third_friend_took) = 1 :=
by
  sorry

end kristy_gave_to_brother_l1809_180934


namespace sam_final_amount_l1809_180982

def initial_dimes : ℕ := 9
def initial_quarters : ℕ := 5
def initial_nickels : ℕ := 3

def dad_dimes : ℕ := 7
def dad_quarters : ℕ := 2

def mom_nickels : ℕ := 1
def mom_dimes : ℕ := 2

def dime_value : ℕ := 10
def quarter_value : ℕ := 25
def nickel_value : ℕ := 5

def initial_amount : ℕ := (initial_dimes * dime_value) + (initial_quarters * quarter_value) + (initial_nickels * nickel_value)
def dad_amount : ℕ := (dad_dimes * dime_value) + (dad_quarters * quarter_value)
def mom_amount : ℕ := (mom_nickels * nickel_value) + (mom_dimes * dime_value)

def final_amount : ℕ := initial_amount + dad_amount - mom_amount

theorem sam_final_amount : final_amount = 325 := by
  sorry

end sam_final_amount_l1809_180982


namespace circumscribed_triangle_area_relation_l1809_180925

theorem circumscribed_triangle_area_relation
    (a b c: ℝ) (h₀: a = 8) (h₁: b = 15) (h₂: c = 17)
    (triangle_area: ℝ) (circle_area: ℝ) (X Y Z: ℝ)
    (hZ: Z > X) (hXY: X < Y)
    (triangle_area_calc: triangle_area = 60)
    (circle_area_calc: circle_area = π * (c / 2)^2) :
    X + Y = Z := by
  sorry

end circumscribed_triangle_area_relation_l1809_180925


namespace average_speed_l1809_180991

theorem average_speed (D : ℝ) (hD : D > 0) :
  let t1 := (D / 3) / 80
  let t2 := (D / 3) / 15
  let t3 := (D / 3) / 48
  let total_time := t1 + t2 + t3
  let avg_speed := D / total_time
  avg_speed = 30 :=
by
  sorry

end average_speed_l1809_180991


namespace movie_theatre_total_seats_l1809_180920

theorem movie_theatre_total_seats (A C : ℕ) 
  (hC : C = 188) 
  (hRevenue : 6 * A + 4 * C = 1124) 
  : A + C = 250 :=
by
  sorry

end movie_theatre_total_seats_l1809_180920


namespace find_a_value_l1809_180967

theorem find_a_value : (15^2 * 8^3 / 256 = 450) :=
by
  sorry

end find_a_value_l1809_180967


namespace sqrt_nine_eq_three_l1809_180912

theorem sqrt_nine_eq_three : Real.sqrt 9 = 3 :=
by
  sorry

end sqrt_nine_eq_three_l1809_180912


namespace mrs_heine_dogs_l1809_180998

-- Define the number of biscuits per dog
def biscuits_per_dog : ℕ := 3

-- Define the total number of biscuits
def total_biscuits : ℕ := 6

-- Define the number of dogs
def number_of_dogs : ℕ := 2

-- Define the proof statement
theorem mrs_heine_dogs : total_biscuits / biscuits_per_dog = number_of_dogs :=
by
  sorry

end mrs_heine_dogs_l1809_180998


namespace work_day_percentage_l1809_180975

theorem work_day_percentage 
  (work_day_hours : ℕ) 
  (first_meeting_minutes : ℕ) 
  (second_meeting_factor : ℕ) 
  (h_work_day : work_day_hours = 10) 
  (h_first_meeting : first_meeting_minutes = 60) 
  (h_second_meeting_factor : second_meeting_factor = 2) :
  ((first_meeting_minutes + second_meeting_factor * first_meeting_minutes) / (work_day_hours * 60) : ℚ) * 100 = 30 :=
sorry

end work_day_percentage_l1809_180975


namespace annual_interest_rate_is_correct_l1809_180917

theorem annual_interest_rate_is_correct :
  ∃ r : ℝ, r = 0.0583 ∧
  (200 * (1 + r)^2 = 224) :=
by
  sorry

end annual_interest_rate_is_correct_l1809_180917


namespace middle_number_divisible_by_4_l1809_180923

noncomputable def three_consecutive_cubes_is_cube (x y : ℕ) : Prop :=
  (x-1)^3 + x^3 + (x+1)^3 = y^3

theorem middle_number_divisible_by_4 (x y : ℕ) (h : three_consecutive_cubes_is_cube x y) : 4 ∣ x :=
sorry

end middle_number_divisible_by_4_l1809_180923


namespace diagonal_AC_length_l1809_180959

theorem diagonal_AC_length (AB BC CD DA : ℝ) (angle_ADC : ℝ) (h_AB : AB = 12) (h_BC : BC = 12) 
(h_CD : CD = 13) (h_DA : DA = 13) (h_angle_ADC : angle_ADC = 60) : 
  AC = 13 := 
sorry

end diagonal_AC_length_l1809_180959


namespace oranges_cost_l1809_180943

def cost_for_multiple_dozens (price_per_dozen: ℝ) (dozens: ℝ) : ℝ := 
    price_per_dozen * dozens

theorem oranges_cost (price_for_4_dozens: ℝ) (price_for_5_dozens: ℝ) :
  price_for_4_dozens = 28.80 →
  price_for_5_dozens = cost_for_multiple_dozens (28.80 / 4) 5 →
  price_for_5_dozens = 36 :=
by
  intros h1 h2
  sorry

end oranges_cost_l1809_180943


namespace reciprocal_of_neg6_l1809_180952

theorem reciprocal_of_neg6 : 1 / (-6 : ℝ) = -1 / 6 := 
sorry

end reciprocal_of_neg6_l1809_180952


namespace problem_I_solution_problem_II_solution_l1809_180963

noncomputable def f (x : ℝ) : ℝ := |3 * x - 2| + |x - 2|

-- Problem (I): Solve the inequality f(x) <= 8
theorem problem_I_solution (x : ℝ) : 
  f x ≤ 8 ↔ -1 ≤ x ∧ x ≤ 3 :=
sorry

-- Problem (II): Find the range of the real number m
theorem problem_II_solution (x m : ℝ) : 
  f x ≥ (m^2 - m + 2) * |x| ↔ (0 ≤ m ∧ m ≤ 1) :=
sorry

end problem_I_solution_problem_II_solution_l1809_180963


namespace packets_of_candy_bought_l1809_180927

theorem packets_of_candy_bought
    (candies_per_day_weekday : ℕ)
    (candies_per_day_weekend : ℕ)
    (days_weekday : ℕ)
    (days_weekend : ℕ)
    (weeks : ℕ)
    (candies_per_packet : ℕ)
    (total_candies : ℕ)
    (packets_bought : ℕ) :
    candies_per_day_weekday = 2 →
    candies_per_day_weekend = 1 →
    days_weekday = 5 →
    days_weekend = 2 →
    weeks = 3 →
    candies_per_packet = 18 →
    total_candies = (candies_per_day_weekday * days_weekday + candies_per_day_weekend * days_weekend) * weeks →
    packets_bought = total_candies / candies_per_packet →
    packets_bought = 2 :=
by
  intros
  sorry

end packets_of_candy_bought_l1809_180927


namespace div_n_by_8_eq_2_8089_l1809_180911

theorem div_n_by_8_eq_2_8089
  (n : ℕ)
  (h : n = 16^2023) :
  n / 8 = 2^8089 := by
  sorry

end div_n_by_8_eq_2_8089_l1809_180911


namespace angle_sum_of_octagon_and_triangle_l1809_180933

-- Define the problem setup
def is_interior_angle_of_regular_polygon (n : ℕ) (angle : ℝ) : Prop :=
  angle = 180 * (n - 2) / n

def is_regular_octagon_angle (angle : ℝ) : Prop :=
  is_interior_angle_of_regular_polygon 8 angle

def is_equilateral_triangle_angle (angle : ℝ) : Prop :=
  is_interior_angle_of_regular_polygon 3 angle

-- The statement of the problem
theorem angle_sum_of_octagon_and_triangle :
  ∃ angle_ABC angle_ABD : ℝ,
    is_regular_octagon_angle angle_ABC ∧
    is_equilateral_triangle_angle angle_ABD ∧
    angle_ABC + angle_ABD = 195 :=
sorry

end angle_sum_of_octagon_and_triangle_l1809_180933


namespace find_n_l1809_180944

open Nat

-- Defining the production rates for conditions.
structure Production := 
  (workers : ℕ)
  (gadgets : ℕ)
  (gizmos : ℕ)
  (hours : ℕ)

def condition1 : Production := { workers := 150, gadgets := 450, gizmos := 300, hours := 1 }
def condition2 : Production := { workers := 100, gadgets := 400, gizmos := 500, hours := 2 }
def condition3 : Production := { workers := 75, gadgets := 900, gizmos := 900, hours := 4 }

-- Statement: Finding the value of n.
theorem find_n :
  (75 * ((condition2.gadgets / condition2.workers) * (condition3.hours / condition2.hours))) = 600 := by
  sorry

end find_n_l1809_180944


namespace find_circle_center_l1809_180926

noncomputable def midpoint_line (a b : ℝ) : ℝ :=
  (a + b) / 2

noncomputable def circle_center (x y : ℝ) : Prop :=
  6 * x - 5 * y = midpoint_line 40 (-20) ∧ 3 * x + 2 * y = 0

theorem find_circle_center : circle_center (20 / 27) (-10 / 9) :=
by
  -- Here would go the proof steps, but we skip it
  sorry

end find_circle_center_l1809_180926


namespace A_alone_days_l1809_180964

noncomputable def days_for_A (r_A r_B r_C : ℝ) : ℝ :=
  1 / r_A

theorem A_alone_days
  (r_A r_B r_C : ℝ) 
  (h1 : r_A + r_B = 1 / 3)
  (h2 : r_B + r_C = 1 / 6)
  (h3 : r_A + r_C = 1 / 4) :
  days_for_A r_A r_B r_C = 4.8 := by
  sorry

end A_alone_days_l1809_180964


namespace negation_is_correct_l1809_180929

-- Define the condition: we have two integers a and b
variables (a b : ℤ)

-- Original proposition: If the sum of two integers is even, then both integers are even.
def original_proposition := (a + b) % 2 = 0 → (a % 2 = 0) ∧ (b % 2 = 0)

-- Negation of the proposition: There exist two integers such that their sum is even and not both are even.
def negation_of_proposition := (a + b) % 2 = 0 ∧ ¬((a % 2 = 0) ∧ (b % 2 = 0))

theorem negation_is_correct :
  ¬ original_proposition a b = negation_of_proposition a b :=
by
  sorry

end negation_is_correct_l1809_180929


namespace find_angle_C_l1809_180919

variable (A B C : ℝ)
variable (a b c : ℝ)

theorem find_angle_C (hA : A = 39) 
                     (h_condition : (a^2 - b^2)*(a^2 + a*c - b^2) = b^2 * c^2) : 
                     C = 115 :=
sorry

end find_angle_C_l1809_180919


namespace snakes_in_breeding_ball_l1809_180941

theorem snakes_in_breeding_ball (x : ℕ) (h : 3 * x + 12 = 36) : x = 8 :=
by sorry

end snakes_in_breeding_ball_l1809_180941


namespace acute_triangle_cannot_divide_into_two_obtuse_l1809_180909

def is_acute_triangle (A B C : ℝ) : Prop :=
  A < 90 ∧ B < 90 ∧ C < 90

def is_obtuse_triangle (A B C : ℝ) : Prop :=
  A > 90 ∨ B > 90 ∨ C > 90

theorem acute_triangle_cannot_divide_into_two_obtuse (A B C A1 B1 C1 A2 B2 C2 : ℝ) 
  (h_acute : is_acute_triangle A B C) 
  (h_divide : A + B + C = 180 ∧ A1 + B1 + C1 = 180 ∧ A2 + B2 + C2 = 180)
  (h_sum : A1 + A2 = A ∧ B1 + B2 = B ∧ C1 + C2 = C) :
  ¬ (is_obtuse_triangle A1 B1 C1 ∧ is_obtuse_triangle A2 B2 C2) :=
sorry

end acute_triangle_cannot_divide_into_two_obtuse_l1809_180909


namespace pascal_triangle_probability_l1809_180937

-- Define the probability problem in Lean 4
theorem pascal_triangle_probability :
  let total_elements := ((20 * (20 + 1)) / 2)
  let ones_count := (1 + 2 * 19)
  let twos_count := (2 * (19 - 2 + 1))
  (ones_count + twos_count) / total_elements = 5 / 14 :=
by
  let total_elements := ((20 * (20 + 1)) / 2)
  let ones_count := (1 + 2 * 19)
  let twos_count := (2 * (19 - 2 + 1))
  have h1 : total_elements = 210 := by sorry
  have h2 : ones_count = 39 := by sorry
  have h3 : twos_count = 36 := by sorry
  have h4 : (39 + 36) / 210 = 5 / 14 := by sorry
  exact h4

end pascal_triangle_probability_l1809_180937


namespace violet_children_count_l1809_180961

theorem violet_children_count 
  (family_pass_cost : ℕ := 120)
  (adult_ticket_cost : ℕ := 35)
  (child_ticket_cost : ℕ := 20)
  (separate_ticket_total_cost : ℕ := 155)
  (adult_count : ℕ := 1) : 
  ∃ c : ℕ, 35 + 20 * c = 155 ∧ c = 6 :=
by
  sorry

end violet_children_count_l1809_180961


namespace find_missing_id_l1809_180980

theorem find_missing_id
  (total_students : ℕ)
  (sample_size : ℕ)
  (known_ids : Finset ℕ)
  (k : ℕ)
  (missing_id : ℕ) : 
  total_students = 52 ∧ 
  sample_size = 4 ∧ 
  known_ids = {3, 29, 42} ∧ 
  k = total_students / sample_size ∧ 
  missing_id = 16 :=
by
  sorry

end find_missing_id_l1809_180980


namespace cos_double_angle_l1809_180981

theorem cos_double_angle (θ : ℝ) (h : Real.cos θ = 1 / 3) : Real.cos (2 * θ) = -7 / 9 :=
by
  sorry

end cos_double_angle_l1809_180981


namespace pool_fill_time_l1809_180928

theorem pool_fill_time
  (faster_pipe_time : ℝ) (slower_pipe_factor : ℝ)
  (H1 : faster_pipe_time = 9) 
  (H2 : slower_pipe_factor = 1.25) : 
  (faster_pipe_time * (1 + slower_pipe_factor) / (faster_pipe_time + faster_pipe_time/slower_pipe_factor)) = 5 :=
by
  sorry

end pool_fill_time_l1809_180928


namespace quadratic_h_value_l1809_180968

theorem quadratic_h_value (p q r h : ℝ) (hq : p*x^2 + q*x + r = 5*(x - 3)^2 + 15):
  let new_quadratic := 4* (p*x^2 + q*x + r)
  let m := 20
  let k := 60
  new_quadratic = m * (x - h) ^ 2 + k → h = 3 := by
  sorry

end quadratic_h_value_l1809_180968


namespace problem_l1809_180902

noncomputable def discriminant (p q : ℝ) : ℝ := p^2 - 4 * q
noncomputable def quadratic (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

theorem problem (p q : ℝ) (hq : q = -2 * p - 5) :
  (quadratic 1 p (q + 1) 2 = 0) →
  q = -2 * p - 5 ∧
  discriminant p q > 0 ∧
  (discriminant p (q + 1) = 0 → 
    (p = -4 ∧ q = 3 ∧ ∀ x : ℝ, quadratic 1 p q x = 0 ↔ (x = 1 ∨ x = 3))) :=
by
  intro hroot_eq
  sorry

end problem_l1809_180902


namespace welders_started_on_other_project_l1809_180954

theorem welders_started_on_other_project
  (r : ℝ) (x : ℝ) (W : ℝ)
  (h1 : 16 * r * 8 = W)
  (h2 : (16 - x) * r * 24 = W - 16 * r) :
  x = 11 :=
by
  sorry

end welders_started_on_other_project_l1809_180954


namespace range_of_m_l1809_180979

-- Define the points and hyperbola condition
section ProofProblem

variables (m y₁ y₂ : ℝ)

-- Given conditions
def point_A_hyperbola : Prop := y₁ = -3 - m
def point_B_hyperbola : Prop := y₂ = (3 + m) / 2
def y1_greater_than_y2 : Prop := y₁ > y₂

-- The theorem to prove
theorem range_of_m (h1 : point_A_hyperbola m y₁) (h2 : point_B_hyperbola m y₂) (h3 : y1_greater_than_y2 y₁ y₂) : m < -3 :=
by { sorry }

end ProofProblem

end range_of_m_l1809_180979


namespace Smarties_remainder_l1809_180960

theorem Smarties_remainder (m : ℕ) (h : m % 11 = 5) : (4 * m) % 11 = 9 :=
by
  sorry

end Smarties_remainder_l1809_180960


namespace numeral_diff_local_face_value_l1809_180915

theorem numeral_diff_local_face_value (P : ℕ) :
  7 * (10 ^ P - 1) = 693 → P = 2 ∧ (N = 700) :=
by
  intro h
  -- The actual proof is not required hence we insert sorry
  sorry

end numeral_diff_local_face_value_l1809_180915


namespace heat_of_neutralization_combination_l1809_180916

-- Define instruments
inductive Instrument
| Balance
| MeasuringCylinder
| Beaker
| Burette
| Thermometer
| TestTube
| AlcoholLamp

def correct_combination : List Instrument :=
  [Instrument.MeasuringCylinder, Instrument.Beaker, Instrument.Thermometer]

theorem heat_of_neutralization_combination :
  correct_combination = [Instrument.MeasuringCylinder, Instrument.Beaker, Instrument.Thermometer] :=
sorry

end heat_of_neutralization_combination_l1809_180916


namespace probability_A_given_B_probability_A_or_B_l1809_180994

-- Definitions of the given conditions
def PA : ℝ := 0.2
def PB : ℝ := 0.18
def PAB : ℝ := 0.12

-- Theorem to prove the probability that city A also experiences rain when city B is rainy
theorem probability_A_given_B : PA * PB = PAB -> PA = 2 / 3 := by
  sorry

-- Theorem to prove the probability that at least one of the two cities experiences rain
theorem probability_A_or_B (PA PB PAB : ℝ) : (PA + PB - PAB) = 0.26 := by
  sorry

end probability_A_given_B_probability_A_or_B_l1809_180994


namespace percentage_of_400_equals_100_l1809_180906

def part : ℝ := 100
def whole : ℝ := 400

theorem percentage_of_400_equals_100 : (part / whole) * 100 = 25 := by
  sorry

end percentage_of_400_equals_100_l1809_180906


namespace inequality_of_ab_l1809_180996

theorem inequality_of_ab (a b : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : a ≠ b) :
  Real.sqrt (a * b) < (a - b) / (Real.log a - Real.log b) ∧ 
  (a - b) / (Real.log a - Real.log b) < (a + b) / 2 :=
by
  sorry

end inequality_of_ab_l1809_180996


namespace mul_inv_mod_35_l1809_180976

theorem mul_inv_mod_35 : (8 * 22) % 35 = 1 := 
  sorry

end mul_inv_mod_35_l1809_180976


namespace jerry_total_shingles_l1809_180985

def roof_length : ℕ := 20
def roof_width : ℕ := 40
def num_roofs : ℕ := 3
def shingles_per_square_foot : ℕ := 8

def area_of_one_side (length width : ℕ) : ℕ :=
  length * width

def total_area_one_roof (area_one_side : ℕ) : ℕ :=
  area_one_side * 2

def total_area_three_roofs (total_area_one_roof : ℕ) : ℕ :=
  total_area_one_roof * num_roofs

def total_shingles_needed (total_area_all_roofs shingles_per_square_foot : ℕ) : ℕ :=
  total_area_all_roofs * shingles_per_square_foot

theorem jerry_total_shingles :
  total_shingles_needed (total_area_three_roofs (total_area_one_roof (area_of_one_side roof_length roof_width))) shingles_per_square_foot = 38400 :=
by
  sorry

end jerry_total_shingles_l1809_180985


namespace find_bases_l1809_180995

theorem find_bases {F1 F2 : ℝ} (R1 R2 : ℕ) 
                   (hR1 : R1 = 9)
                   (hR2 : R2 = 6)
                   (hF1_R1 : F1 = 0.484848 * 9^2 / (9^2 - 1))
                   (hF2_R1 : F2 = 0.848484 * 9^2 / (9^2 - 1))
                   (hF1_R2 : F1 = 0.353535 * 6^2 / (6^2 - 1))
                   (hF2_R2 : F2 = 0.535353 * 6^2 / (6^2 - 1))
                   : R1 + R2 = 15 :=
by
  sorry

end find_bases_l1809_180995


namespace Vasya_and_Petya_no_mistake_exists_l1809_180918

def is_prime (n : ℕ) : Prop := ∃ d, d > 1 ∧ d < n ∧ n % d = 0

theorem Vasya_and_Petya_no_mistake_exists :
  ∃ x : ℝ, (∃ p : ℕ, is_prime p ∧ 10 * x = p) ∧ 
           (∃ q : ℕ, is_prime q ∧ 15 * x = q) :=
sorry

end Vasya_and_Petya_no_mistake_exists_l1809_180918


namespace a_n_formula_T_n_formula_l1809_180945

variable (a : Nat → Int) (b : Nat → Int)
variable (S : Nat → Int) (T : Nat → Int)
variable (d a_1 : Int)

-- Conditions:
axiom a_seq_arith : ∀ n, a (n + 1) = a n + d
axiom S_arith : ∀ n, S n = n * (a 1 + a n) / 2
axiom S_10 : S 10 = 110
axiom geo_seq : (a 2) ^ 2 = a 1 * a 4
axiom b_def : ∀ n, b n = 1 / ((a n - 1) * (a n + 1))

-- Goals: 
-- 1. Find the general formula for the terms of sequence {a_n}
theorem a_n_formula : ∀ n, a n = 2 * n := sorry

-- 2. Find the sum of the first n terms T_n of the sequence {b_n} given b_n
theorem T_n_formula : ∀ n, T n = 1 / 2 - 1 / (4 * n + 2) := sorry

end a_n_formula_T_n_formula_l1809_180945


namespace walkway_area_l1809_180900

theorem walkway_area (l w : ℕ) (walkway_width : ℕ) (total_length total_width pool_area walkway_area : ℕ)
  (hl : l = 20) 
  (hw : w = 8)
  (hww : walkway_width = 1)
  (htl : total_length = l + 2 * walkway_width)
  (htw : total_width = w + 2 * walkway_width)
  (hpa : pool_area = l * w)
  (hta : (total_length * total_width) = pool_area + walkway_area) :
  walkway_area = 60 := 
  sorry

end walkway_area_l1809_180900


namespace new_numbers_are_reciprocals_l1809_180921

variable {x y : ℝ}

theorem new_numbers_are_reciprocals (h : (1 / x) + (1 / y) = 1) : 
  (x - 1 = 1 / (y - 1)) ∧ (y - 1 = 1 / (x - 1)) := 
by
  sorry

end new_numbers_are_reciprocals_l1809_180921


namespace slices_per_person_l1809_180955

theorem slices_per_person
  (number_of_coworkers : ℕ)
  (number_of_pizzas : ℕ)
  (number_of_slices_per_pizza : ℕ)
  (total_slices : ℕ)
  (slices_per_person : ℕ) :
  number_of_coworkers = 12 →
  number_of_pizzas = 3 →
  number_of_slices_per_pizza = 8 →
  total_slices = number_of_pizzas * number_of_slices_per_pizza →
  slices_per_person = total_slices / number_of_coworkers →
  slices_per_person = 2 :=
by intros; sorry

end slices_per_person_l1809_180955


namespace problem_statement_l1809_180966

theorem problem_statement (a n : ℕ) (h_a : a ≥ 1) (h_n : n ≥ 1) :
  (∃ k : ℕ, (a + 1)^n - a^n = k * n) ↔ n = 1 := by
  sorry

end problem_statement_l1809_180966


namespace vegetable_plot_area_l1809_180997

variable (V W : ℝ)

theorem vegetable_plot_area (h1 : (1/2) * V + (1/3) * W = 13) (h2 : (1/2) * W + (1/3) * V = 12) : V = 18 :=
by
  sorry

end vegetable_plot_area_l1809_180997


namespace paul_money_duration_l1809_180905

theorem paul_money_duration (mowing_income weed_eating_income weekly_spending money_last: ℕ) 
    (h1: mowing_income = 44) 
    (h2: weed_eating_income = 28) 
    (h3: weekly_spending = 9) 
    (h4: money_last = 8) 
    : (mowing_income + weed_eating_income) / weekly_spending = money_last := 
by
  sorry

end paul_money_duration_l1809_180905


namespace triangle_side_count_l1809_180962

theorem triangle_side_count :
  {b c : ℕ} → b ≤ 5 → 5 ≤ c → c - b < 5 → ∃ t : ℕ, t = 15 :=
by
  sorry

end triangle_side_count_l1809_180962


namespace integer_roots_and_composite_l1809_180948

theorem integer_roots_and_composite (a b : ℤ) (h1 : ∃ x1 x2 : ℤ, x1 * x2 = 1 - b ∧ x1 + x2 = -a) (h2 : b ≠ 1) : 
  ∃ (m n : ℕ), m > 1 ∧ n > 1 ∧ m * n = (a^2 + b^2) := 
sorry

end integer_roots_and_composite_l1809_180948


namespace cyclist_average_speed_l1809_180924

theorem cyclist_average_speed (v : ℝ) 
  (h1 : 8 / v + 10 / 8 = 18 / 8.78) : v = 10 :=
by
  sorry

end cyclist_average_speed_l1809_180924


namespace complement_intersection_l1809_180984

/-- Given the universal set U={1,2,3,4,5},
    A={2,3,4}, and B={1,2,3}, 
    Prove the complement of (A ∩ B) in U is {1,4,5}. -/
theorem complement_intersection 
    (U : Set ℕ) (A : Set ℕ) (B : Set ℕ) 
    (hU : U = {1, 2, 3, 4, 5})
    (hA : A = {2, 3, 4})
    (hB : B = {1, 2, 3}) :
    U \ (A ∩ B) = {1, 4, 5} :=
by
  -- proof goes here
  sorry

end complement_intersection_l1809_180984


namespace dot_product_to_linear_form_l1809_180904

noncomputable def proof_problem (r a : ℝ × ℝ) (m : ℝ) : Prop :=
  let A := a.1
  let B := a.2
  let C := -m
  (r.1 * a.1 + r.2 * a.2 = m) → (A * r.1 + B * r.2 + C = 0)

-- The theorem statement
theorem dot_product_to_linear_form (r a : ℝ × ℝ) (m : ℝ) :
  proof_problem r a m :=
sorry

end dot_product_to_linear_form_l1809_180904


namespace length_AE_l1809_180970

-- The given conditions:
def isosceles_triangle (A B C : Type*) (AB BC : ℝ) (h : AB = BC) : Prop := true

def angles_and_lengths (A D C E : Type*) (angle_ADC angle_AEC AD CE DC : ℝ) 
  (h_angles : angle_ADC = 60 ∧ angle_AEC = 60)
  (h_lengths : AD = 13 ∧ CE = 13 ∧ DC = 9) : Prop := true

variables {A B C D E : Type*} (AB BC AD CE DC : ℝ)
  (h_isosceles_triangle : isosceles_triangle A B C AB BC (by sorry))
  (h_angles_and_lengths : angles_and_lengths A D C E 60 60 AD CE DC 
    (by split; norm_num) (by repeat {split}; norm_num))

-- The proof problem:
theorem length_AE : ∃ AE : ℝ, AE = 4 :=
  by sorry

end length_AE_l1809_180970


namespace common_ratio_geometric_series_l1809_180946

theorem common_ratio_geometric_series :
  let a := (7 : ℚ) / 8
  let b := - (5 : ℚ) / 12
  (b / a) = - (10 : ℚ) / 21 :=
by
  -- definitions
  let a := (7 : ℚ) / 8
  let b := - (5 : ℚ) / 12
  -- assertion
  have ratio := b / a
  sorry

end common_ratio_geometric_series_l1809_180946


namespace profit_margin_comparison_l1809_180983

theorem profit_margin_comparison
    (cost_price_A : ℝ) (selling_price_A : ℝ)
    (cost_price_B : ℝ) (selling_price_B : ℝ)
    (h1 : cost_price_A = 1600)
    (h2 : selling_price_A = 0.9 * 2000)
    (h3 : cost_price_B = 320)
    (h4 : selling_price_B = 0.8 * 460) :
    ((selling_price_B - cost_price_B) / cost_price_B) > ((selling_price_A - cost_price_A) / cost_price_A) := 
by
    sorry

end profit_margin_comparison_l1809_180983


namespace Veronica_to_Half_Samir_Ratio_l1809_180931

-- Mathematical conditions 
def Samir_stairs : ℕ := 318
def Total_stairs : ℕ := 495
def Half_Samir_stairs : ℚ := Samir_stairs / 2

-- Definition for Veronica's stairs as a multiple of half Samir's stairs
def Veronica_stairs (R: ℚ) : ℚ := R * Half_Samir_stairs

-- Lean statement to prove the ratio
theorem Veronica_to_Half_Samir_Ratio (R : ℚ) (H1 : Veronica_stairs R + Samir_stairs = Total_stairs) : R = 1.1132 := 
by
  sorry

end Veronica_to_Half_Samir_Ratio_l1809_180931


namespace range_of_fx_over_x_l1809_180932

variable (f : ℝ → ℝ)

noncomputable def is_odd (f : ℝ → ℝ) :=
  ∀ x, f (-x) = -f x

noncomputable def is_increasing_on (f : ℝ → ℝ) (s : Set ℝ) :=
  ∀ ⦃x y⦄, x ∈ s → y ∈ s → x < y → f x < f y

theorem range_of_fx_over_x (odd_f : is_odd f)
                           (increasing_f_pos : is_increasing_on f {x : ℝ | x > 0})
                           (hf1 : f (-1) = 0) :
  {x | f x / x < 0} = {x | -1 < x ∧ x < 0} ∪ {x | 0 < x ∧ x < 1} :=
sorry

end range_of_fx_over_x_l1809_180932


namespace not_proportional_eqn_exists_l1809_180956

theorem not_proportional_eqn_exists :
  ∀ (x y : ℝ), (4 * x + 2 * y = 8) → ¬ ((∃ k : ℝ, x = k * y) ∨ (∃ k : ℝ, x * y = k)) :=
by
  intros x y h
  sorry

end not_proportional_eqn_exists_l1809_180956


namespace least_value_of_a_plus_b_l1809_180930

def a_and_b (a b : ℕ) : Prop :=
  (Nat.gcd (a + b) 330 = 1) ∧ 
  (a^a % b^b = 0) ∧ 
  (¬ (a % b = 0))

theorem least_value_of_a_plus_b :
  ∃ (a b : ℕ), a_and_b a b ∧ a + b = 105 :=
sorry

end least_value_of_a_plus_b_l1809_180930


namespace light_glow_duration_l1809_180969

-- Define the conditions
def total_time_seconds : ℕ := 4969
def glow_times : ℚ := 292.29411764705884

-- Prove the equivalent statement
theorem light_glow_duration :
  (total_time_seconds / glow_times) = 17 := by
  sorry

end light_glow_duration_l1809_180969


namespace atomic_weight_of_iodine_is_correct_l1809_180957

noncomputable def atomic_weight_iodine (atomic_weight_nitrogen : ℝ) (atomic_weight_hydrogen : ℝ) (molecular_weight_compound : ℝ) : ℝ :=
  molecular_weight_compound - (atomic_weight_nitrogen + 4 * atomic_weight_hydrogen)

theorem atomic_weight_of_iodine_is_correct :
  atomic_weight_iodine 14.01 1.008 145 = 126.958 :=
by
  unfold atomic_weight_iodine
  norm_num

end atomic_weight_of_iodine_is_correct_l1809_180957


namespace jimmy_needs_4_packs_of_bread_l1809_180992

theorem jimmy_needs_4_packs_of_bread
  (num_sandwiches : ℕ)
  (slices_per_sandwich : ℕ)
  (slices_per_pack : ℕ)
  (initial_slices : ℕ)
  (h1 : num_sandwiches = 8)
  (h2 : slices_per_sandwich = 2)
  (h3 : slices_per_pack = 4)
  (h4 : initial_slices = 0) :
  (num_sandwiches * slices_per_sandwich) / slices_per_pack = 4 := by
  sorry

end jimmy_needs_4_packs_of_bread_l1809_180992


namespace cost_of_1000_pieces_of_gum_l1809_180940

theorem cost_of_1000_pieces_of_gum
  (cost_per_piece : ℕ)
  (num_pieces : ℕ)
  (discount_threshold : ℕ)
  (discount_rate : ℚ)
  (conversion_rate : ℕ)
  (h_cost : cost_per_piece = 2)
  (h_pieces : num_pieces = 1000)
  (h_threshold : discount_threshold = 500)
  (h_discount : discount_rate = 0.90)
  (h_conversion : conversion_rate = 100)
  (h_more_than_threshold : num_pieces > discount_threshold) :
  (num_pieces * cost_per_piece * discount_rate) / conversion_rate = 18 := 
sorry

end cost_of_1000_pieces_of_gum_l1809_180940


namespace ordering_of_powers_l1809_180951

theorem ordering_of_powers :
  2^30 < 10^10 ∧ 10^10 < 5^15 :=
by sorry

end ordering_of_powers_l1809_180951


namespace trigonometric_identity_l1809_180973

theorem trigonometric_identity :
  1 / Real.sin (70 * Real.pi / 180) - Real.sqrt 2 / Real.cos (70 * Real.pi / 180) = 
  -2 * (Real.sin (25 * Real.pi / 180) / Real.sin (40 * Real.pi / 180)) :=
sorry

end trigonometric_identity_l1809_180973


namespace area_T_l1809_180935

variable (T : Set (ℝ × ℝ)) -- T is a region in the plane
variable (A : Matrix (Fin 2) (Fin 2) ℝ) -- A is a 2x2 matrix
variable (detA : ℝ) -- detA is the determinant of A

-- assumptions
axiom area_T : ∃ (area : ℝ), area = 9
axiom matrix_A : A = ![![3, 2], ![-1, 4]]
axiom determinant_A : detA = 14

-- statement to prove
theorem area_T' : ∃ area_T' : ℝ, area_T' = 126 :=
sorry

end area_T_l1809_180935


namespace number_of_girls_l1809_180901

theorem number_of_girls (B G : ℕ) 
  (h1 : B = G + 124) 
  (h2 : B + G = 1250) : G = 563 :=
by
  sorry

end number_of_girls_l1809_180901


namespace center_of_circle_l1809_180999

theorem center_of_circle (x y : ℝ) :
  x^2 + y^2 - 2 * x - 6 * y + 1 = 0 →
  (1, 3) = (1, 3) :=
by
  intros h
  sorry

end center_of_circle_l1809_180999


namespace jane_stick_length_l1809_180908

variable (P U S J F : ℕ)
variable (h1 : P = 30)
variable (h2 : U = P - 7)
variable (h3 : U = S / 2)
variable (h4 : F = 2 * 12)
variable (h5 : J = S - F)

theorem jane_stick_length : J = 22 := by
  sorry

end jane_stick_length_l1809_180908


namespace water_intake_proof_l1809_180990

variable {quarts_per_bottle : ℕ} {bottles_per_day : ℕ} {extra_ounces_per_day : ℕ} 
variable {days_per_week : ℕ} {ounces_per_quart : ℕ} 

def total_weekly_water_intake 
    (quarts_per_bottle : ℕ) 
    (bottles_per_day : ℕ) 
    (extra_ounces_per_day : ℕ) 
    (ounces_per_quart : ℕ) 
    (days_per_week : ℕ) 
    (correct_answer : ℕ) : Prop :=
    (quarts_per_bottle * ounces_per_quart * bottles_per_day + extra_ounces_per_day) * days_per_week = correct_answer

theorem water_intake_proof : 
    total_weekly_water_intake 3 2 20 32 7 812 := 
by
    sorry

end water_intake_proof_l1809_180990


namespace fraction_notation_correct_reading_decimal_correct_l1809_180950

-- Define the given conditions
def fraction_notation (num denom : ℕ) : Prop :=
  num / denom = num / denom  -- Essentially stating that in fraction notation, it holds

def reading_decimal (n : ℚ) (s : String) : Prop :=
  if n = 90.58 then s = "ninety point five eight" else false -- Defining the reading rule for this specific case

-- State the theorem using the defined conditions
theorem fraction_notation_correct : fraction_notation 8 9 := 
by 
  sorry

theorem reading_decimal_correct : reading_decimal 90.58 "ninety point five eight" :=
by 
  sorry

end fraction_notation_correct_reading_decimal_correct_l1809_180950


namespace smallest_n_with_divisors_l1809_180978

-- Definitions of the divisors
def d_total (a b c : ℕ) : ℕ := (a + 1) * (b + 1) * (c + 1)
def d_even (a b c : ℕ) : ℕ := a * (b + 1) * (c + 1)
def d_odd (b c : ℕ) : ℕ := (b + 1) * (c + 1)

-- Math problem and proving smallest n
theorem smallest_n_with_divisors (a b c : ℕ) (n : ℕ) (h_1 : d_odd b c = 8) (h_2 : d_even a b c = 16) : n = 60 :=
  sorry

end smallest_n_with_divisors_l1809_180978


namespace total_selling_price_l1809_180938

theorem total_selling_price 
  (n : ℕ) (p : ℕ) (c : ℕ) 
  (h_n : n = 85) (h_p : p = 15) (h_c : c = 85) : 
  (c + p) * n = 8500 :=
by
  sorry

end total_selling_price_l1809_180938


namespace cuberoot_sum_l1809_180949

-- Prove that the sum c + d = 60 for the simplified form of the given expression.
theorem cuberoot_sum :
  let c := 15
  let d := 45
  c + d = 60 :=
by
  sorry

end cuberoot_sum_l1809_180949


namespace line_BC_eq_circumscribed_circle_eq_l1809_180958

noncomputable def A : ℝ × ℝ := (3, 0)
noncomputable def B : ℝ × ℝ := (0, -1)
noncomputable def altitude_line (x y : ℝ) : Prop := x + y + 1 = 0
noncomputable def equation_line_BC (x y : ℝ) : Prop := 3 * x - y - 1 = 0
noncomputable def circumscribed_circle (x y : ℝ) : Prop := (x - 5 / 2)^2 + (y + 7 / 2)^2 = 50 / 4

theorem line_BC_eq :
  ∃ x y : ℝ, altitude_line x y →
             B = (x, y) →
             equation_line_BC x y :=
by sorry

theorem circumscribed_circle_eq :
  ∃ x y : ℝ, altitude_line x y →
             (x - 3)^2 + y^2 = (5 / 2)^2 →
             circumscribed_circle x y :=
by sorry

end line_BC_eq_circumscribed_circle_eq_l1809_180958


namespace problem1_problem2_l1809_180914

-- Definitions based on the given conditions
def A (a b : ℝ) : ℝ := 2 * a^2 + 3 * a * b - 2 * a - 1
def B (a b : ℝ) : ℝ := a^2 + a * b - 1

-- Statement for problem (1)
theorem problem1 (a b : ℝ) : 
  4 * A a b - (3 * A a b - 2 * B a b) = 4 * a^2 + 5 * a * b - 2 * a - 3 :=
by sorry

-- Statement for problem (2)
theorem problem2 (a b : ℝ) (h : ∀ a, A a b - 2 * B a b = k) : 
  b = 2 :=
by sorry

end problem1_problem2_l1809_180914


namespace ratio_correct_l1809_180987

theorem ratio_correct : 
    (2^17 * 3^19) / (6^18) = 3 / 2 :=
by sorry

end ratio_correct_l1809_180987


namespace max_visible_sum_is_128_l1809_180971

-- Define the structure of the problem
structure Cube :=
  (faces : Fin 6 → Nat)
  (bottom_face : Nat)
  (all_faces : ∀ i : Fin 6, i ≠ ⟨0, by decide⟩ → faces i = bottom_face → False)

-- Define the problem conditions
noncomputable def problem_conditions : Prop :=
  let cubes := [Cube.mk (fun i => [1, 3, 5, 7, 9, 11].get i) 1 sorry,
                Cube.mk (fun i => [1, 3, 5, 7, 9, 11].get i) 1 sorry,
                Cube.mk (fun i => [1, 3, 5, 7, 9, 11].get i) 1 sorry,
                Cube.mk (fun i => [1, 3, 5, 7, 9, 11].get i) 1 sorry]
  -- Cube stacking in two layers, with two cubes per layer
  
  true

-- Define the theorem to be proved
theorem max_visible_sum_is_128 (h : problem_conditions) : 
  ∃ (total_sum : Nat), total_sum = 128 := 
sorry

end max_visible_sum_is_128_l1809_180971


namespace max_distance_on_highway_l1809_180972

-- Assume there are definitions for the context of this problem
def mpg_highway : ℝ := 12.2
def gallons : ℝ := 24
def max_distance (mpg : ℝ) (gal : ℝ) : ℝ := mpg * gal

theorem max_distance_on_highway :
  max_distance mpg_highway gallons = 292.8 :=
sorry

end max_distance_on_highway_l1809_180972


namespace transform_map_ABCD_to_A_l1809_180910

structure Point :=
(x : ℤ)
(y : ℤ)

structure Rectangle :=
(A : Point)
(B : Point)
(C : Point)
(D : Point)

def transform180 (p : Point) : Point :=
  ⟨-p.x, -p.y⟩

def rect_transform180 (rect : Rectangle) : Rectangle :=
  { A := transform180 rect.A,
    B := transform180 rect.B,
    C := transform180 rect.C,
    D := transform180 rect.D }

def ABCD := Rectangle.mk ⟨-3, 2⟩ ⟨-1, 2⟩ ⟨-1, 5⟩ ⟨-3, 5⟩
def A'B'C'D' := Rectangle.mk ⟨3, -2⟩ ⟨1, -2⟩ ⟨1, -5⟩ ⟨3, -5⟩

theorem transform_map_ABCD_to_A'B'C'D' :
  rect_transform180 ABCD = A'B'C'D' :=
by
  -- This is where the proof would go.
  sorry

end transform_map_ABCD_to_A_l1809_180910


namespace average_speed_of_train_l1809_180942

theorem average_speed_of_train (x : ℝ) (h₀ : x > 0) :
  let time_1 := x / 40
  let time_2 := 2 * x / 20
  let total_time := time_1 + time_2
  let total_distance := 6 * x
  let avg_speed := total_distance / total_time
  avg_speed = 48 := by
  let time_1 := x / 40
  let time_2 := 2 * x / 20
  let total_time := time_1 + time_2
  let total_distance := 6 * x
  let avg_speed := total_distance / total_time
  sorry

end average_speed_of_train_l1809_180942
