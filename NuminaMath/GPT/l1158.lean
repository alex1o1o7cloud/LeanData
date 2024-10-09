import Mathlib

namespace solve_for_x_l1158_115822

theorem solve_for_x (x : ℝ) (h : (1 / 5) + (5 / x) = (12 / x) + (1 / 12)) : x = 60 := by
  sorry

end solve_for_x_l1158_115822


namespace power_of_power_l1158_115896

theorem power_of_power {a : ℝ} : (a^2)^3 = a^6 := 
by
  sorry

end power_of_power_l1158_115896


namespace value_of_p_l1158_115862

theorem value_of_p (x y p : ℝ) 
  (h1 : 3 * x - 2 * y = 4 - p) 
  (h2 : 4 * x - 3 * y = 2 + p) 
  (h3 : x > y) : 
  p < -1 := 
sorry

end value_of_p_l1158_115862


namespace at_least_one_neg_l1158_115832

theorem at_least_one_neg (a b c d : ℝ) (h1 : a + b = 1) (h2 : c + d = 1) (h3 : ac + bd > 1) : 
  a < 0 ∨ b < 0 ∨ c < 0 ∨ d < 0 :=
sorry

end at_least_one_neg_l1158_115832


namespace cost_of_one_dozen_pens_l1158_115889

theorem cost_of_one_dozen_pens
  (p q : ℕ)
  (h1 : 3 * p + 5 * q = 240)
  (h2 : p = 5 * q) :
  12 * p = 720 :=
by
  sorry

end cost_of_one_dozen_pens_l1158_115889


namespace range_of_a_l1158_115833

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 2 * x^2 + (a - 1) * x + 1 / 2 > 0) → (-1 < a ∧ a < 3) :=
by
  intro h
  sorry

end range_of_a_l1158_115833


namespace cost_fly_D_to_E_l1158_115875

-- Definitions for the given conditions
def distance_DE : ℕ := 4750
def cost_per_km_plane : ℝ := 0.12
def booking_fee_plane : ℝ := 150

-- The proof statement about the total cost
theorem cost_fly_D_to_E : (distance_DE * cost_per_km_plane + booking_fee_plane = 720) :=
by sorry

end cost_fly_D_to_E_l1158_115875


namespace root_of_equation_l1158_115841

theorem root_of_equation : 
  ∀ (f : ℝ → ℝ), (∀ x : ℝ, f x = (x - 1) / x) →
  f (4 * (1 / 2)) = (1 / 2) :=
by
  sorry

end root_of_equation_l1158_115841


namespace no_fraternity_member_is_club_member_l1158_115854

variable {U : Type} -- Domain of discourse, e.g., the set of all people at the school
variables (Club Member Student Honest Fraternity : U → Prop)

theorem no_fraternity_member_is_club_member
  (h1 : ∀ x, Club x → Student x)
  (h2 : ∀ x, Club x → ¬ Honest x)
  (h3 : ∀ x, Fraternity x → Honest x) :
  ∀ x, Fraternity x → ¬ Club x := 
sorry

end no_fraternity_member_is_club_member_l1158_115854


namespace eccentricity_range_l1158_115856

section EllipseEccentricity

variables {F1 F2 : ℝ × ℝ}
variable (M : ℝ × ℝ)

-- Conditions from a)
def is_orthogonal (v1 v2 : ℝ × ℝ) : Prop :=
  v1.1 * v2.1 + v1.2 * v2.2 = 0

def is_inside_ellipse (F1 F2 M : ℝ × ℝ) : Prop :=
  is_orthogonal (M.1 - F1.1, M.2 - F1.2) (M.1 - F2.1, M.2 - F2.2) ∧ 
  -- other conditions to assert M is inside could be defined but this is unspecified
  true

-- Statement from c)
theorem eccentricity_range {a b c e : ℝ}
  (h : ∀ (M: ℝ × ℝ), is_orthogonal (M.1 - F1.1, M.2 - F1.2) (M.1 - F2.1, M.2 - F2.2) → is_inside_ellipse F1 F2 M)
  (h1 : c^2 < a^2 - c^2)
  (h2 : e^2 = c^2 / a^2) :
  0 < e ∧ e < (Real.sqrt 2) / 2 := 
sorry

end EllipseEccentricity

end eccentricity_range_l1158_115856


namespace find_c_l1158_115897

noncomputable def parabola_equation (a b c y : ℝ) : ℝ :=
  a * y^2 + b * y + c

theorem find_c (a b c : ℝ) (h_vertex : (-4, 2) = (-4, 2)) (h_point : (-2, 4) = (-2, 4)) :
  ∃ c : ℝ, parabola_equation a b c 0 = -2 :=
  by {
    use -2,
    sorry
  }

end find_c_l1158_115897


namespace unique_function_satisfying_conditions_l1158_115871

noncomputable def f : (ℝ → ℝ) := sorry

axiom condition1 : f 1 = 1
axiom condition2 : ∀ x y : ℝ, f (x * y + f x) = x * f y + f x

theorem unique_function_satisfying_conditions : ∀ x : ℝ, f x = x := sorry

end unique_function_satisfying_conditions_l1158_115871


namespace find_atomic_weight_of_Na_l1158_115852

def atomic_weight_of_Na_is_correct : Prop :=
  ∃ (atomic_weight_of_Na : ℝ),
    (atomic_weight_of_Na + 35.45 + 16.00 = 74) ∧ (atomic_weight_of_Na = 22.55)

theorem find_atomic_weight_of_Na : atomic_weight_of_Na_is_correct :=
by
  sorry

end find_atomic_weight_of_Na_l1158_115852


namespace smallest_divisible_by_15_16_18_l1158_115895

def factors_of_15 : Prop := 15 = 3 * 5
def factors_of_16 : Prop := 16 = 2^4
def factors_of_18 : Prop := 18 = 2 * 3^2

theorem smallest_divisible_by_15_16_18 (h1: factors_of_15) (h2: factors_of_16) (h3: factors_of_18) : 
  ∃ n, n > 0 ∧ n % 15 = 0 ∧ n % 16 = 0 ∧ n % 18 = 0 ∧ n = 720 :=
by
  sorry

end smallest_divisible_by_15_16_18_l1158_115895


namespace factorize_expr_l1158_115891

theorem factorize_expr (x : ℝ) : x^3 - 16 * x = x * (x + 4) * (x - 4) :=
sorry

end factorize_expr_l1158_115891


namespace volume_ratio_of_cubes_l1158_115872

def cube_volume (a : ℝ) : ℝ := a ^ 3

theorem volume_ratio_of_cubes :
  cube_volume 3 / cube_volume 18 = 1 / 216 :=
by
  sorry

end volume_ratio_of_cubes_l1158_115872


namespace positive_number_sum_square_l1158_115802

theorem positive_number_sum_square (n : ℕ) (h : n^2 + n = 210) : n = 14 :=
sorry

end positive_number_sum_square_l1158_115802


namespace solve_for_x_l1158_115837

theorem solve_for_x (x : ℝ) :
  (∀ y : ℝ, 10 * x * y - 15 * y + 2 * x - 3 = 0) → x = 3 / 2 :=
by
  intro h
  sorry

end solve_for_x_l1158_115837


namespace belle_rawhide_bones_per_evening_l1158_115846

theorem belle_rawhide_bones_per_evening 
  (cost_rawhide_bone : ℝ)
  (cost_dog_biscuit : ℝ)
  (num_dog_biscuits_per_evening : ℕ)
  (total_weekly_cost : ℝ)
  (days_per_week : ℕ)
  (rawhide_bones_per_evening : ℕ)
  (h1 : cost_rawhide_bone = 1)
  (h2 : cost_dog_biscuit = 0.25)
  (h3 : num_dog_biscuits_per_evening = 4)
  (h4 : total_weekly_cost = 21)
  (h5 : days_per_week = 7)
  (h6 : rawhide_bones_per_evening * cost_rawhide_bone * (days_per_week : ℝ) = total_weekly_cost - num_dog_biscuits_per_evening * cost_dog_biscuit * (days_per_week : ℝ)) :
  rawhide_bones_per_evening = 2 := 
sorry

end belle_rawhide_bones_per_evening_l1158_115846


namespace determine_k_l1158_115834

theorem determine_k (k : ℤ) : (∀ n : ℤ, gcd (4 * n + 1) (k * n + 1) = 1) ↔ 
  (∃ m : ℕ, k = 4 + 2 ^ m ∨ k = 4 - 2 ^ m) :=
by
  sorry

end determine_k_l1158_115834


namespace average_viewing_times_correct_l1158_115880

-- Define the viewing times for each family member per week
def Evelyn_week1 : ℕ := 10
def Evelyn_week2 : ℕ := 8
def Evelyn_week3 : ℕ := 6

def Eric_week1 : ℕ := 8
def Eric_week2 : ℕ := 6
def Eric_week3 : ℕ := 5

def Kate_week2_episodes : ℕ := 12
def minutes_per_episode : ℕ := 40
def Kate_week2 : ℕ := (Kate_week2_episodes * minutes_per_episode) / 60
def Kate_week3 : ℕ := 4

def John_week2 : ℕ := (Kate_week2_episodes * minutes_per_episode) / 60
def John_week3 : ℕ := 8

-- Calculate the averages
def average (total : ℚ) (weeks : ℚ) : ℚ := total / weeks

-- Define the total viewing time for each family member
def Evelyn_total : ℕ := Evelyn_week1 + Evelyn_week2 + Evelyn_week3
def Eric_total : ℕ := Eric_week1 + Eric_week2 + Eric_week3
def Kate_total : ℕ := 0 + Kate_week2 + Kate_week3
def John_total : ℕ := 0 + John_week2 + John_week3

-- Define the expected averages
def Evelyn_expected_avg : ℚ := 8
def Eric_expected_avg : ℚ := 19 / 3
def Kate_expected_avg : ℚ := 4
def John_expected_avg : ℚ := 16 / 3

-- The theorem to prove that the calculated averages are correct
theorem average_viewing_times_correct :
  average Evelyn_total 3 = Evelyn_expected_avg ∧
  average Eric_total 3 = Eric_expected_avg ∧
  average Kate_total 3 = Kate_expected_avg ∧
  average John_total 3 = John_expected_avg :=
by sorry

end average_viewing_times_correct_l1158_115880


namespace train_a_distance_at_meeting_l1158_115886

-- Define the problem conditions as constants
def distance := 75 -- distance between start points of Train A and B
def timeA := 3 -- time taken by Train A to complete the trip in hours
def timeB := 2 -- time taken by Train B to complete the trip in hours

-- Calculate the speeds
def speedA := distance / timeA -- speed of Train A in miles per hour
def speedB := distance / timeB -- speed of Train B in miles per hour

-- Calculate the combined speed and time to meet
def combinedSpeed := speedA + speedB
def timeToMeet := distance / combinedSpeed

-- Define the distance traveled by Train A at the time of meeting
def distanceTraveledByTrainA := speedA * timeToMeet

-- Theorem stating Train A has traveled 30 miles when it met Train B
theorem train_a_distance_at_meeting : distanceTraveledByTrainA = 30 := by
  sorry

end train_a_distance_at_meeting_l1158_115886


namespace abs_x_gt_1_iff_x_sq_minus1_gt_0_l1158_115829

theorem abs_x_gt_1_iff_x_sq_minus1_gt_0 (x : ℝ) : (|x| > 1) ↔ (x^2 - 1 > 0) := by
  sorry

end abs_x_gt_1_iff_x_sq_minus1_gt_0_l1158_115829


namespace solve_for_s_l1158_115845

theorem solve_for_s (s : ℝ) :
  (s^2 - 6 * s + 8) / (s^2 - 9 * s + 14) = (s^2 - 3 * s - 18) / (s^2 - 2 * s - 24) →
  s = -5 / 4 :=
by {
  sorry
}

end solve_for_s_l1158_115845


namespace expectation_of_X_l1158_115878

-- Conditions:
-- Defect rate of the batch of products is 0.05
def defect_rate : ℚ := 0.05

-- 5 items are randomly selected for quality inspection
def n : ℕ := 5

-- The probability of obtaining a qualified product in each trial
def P : ℚ := 1 - defect_rate

-- Question:
-- The random variable X, representing the number of qualified products, follows a binomial distribution.
-- Expectation of X
def expectation_X : ℚ := n * P

-- Prove that the mathematical expectation E(X) is equal to 4.75
theorem expectation_of_X :
  expectation_X = 4.75 := 
sorry

end expectation_of_X_l1158_115878


namespace exists_pair_satisfying_system_l1158_115855

theorem exists_pair_satisfying_system (m : ℝ) :
  (∃ x y : ℝ, y = m * x + 5 ∧ y = (3 * m - 2) * x + 7) ↔ m ≠ 1 :=
by
  sorry

end exists_pair_satisfying_system_l1158_115855


namespace percentage_of_number_l1158_115830

theorem percentage_of_number (n : ℝ) (h : (1 / 4) * (1 / 3) * (2 / 5) * n = 16) : 0.4 * n = 192 :=
by 
  sorry

end percentage_of_number_l1158_115830


namespace additional_people_needed_l1158_115836

-- Definition of the conditions
def person_hours (people: ℕ) (hours: ℕ) : ℕ := people * hours

-- Assertion that 8 people can paint the fence in 3 hours
def eight_people_three_hours : Prop := person_hours 8 3 = 24

-- Definition of the additional people required
def additional_people (initial_people required_people: ℕ) : ℕ := required_people - initial_people

-- Main theorem stating the problem
theorem additional_people_needed : eight_people_three_hours → additional_people 8 12 = 4 :=
by
  sorry

end additional_people_needed_l1158_115836


namespace calculate_expression_l1158_115825

theorem calculate_expression (a b : ℝ) : (a - b) * (a + b) * (a^2 - b^2) = a^4 - 2 * a^2 * b^2 + b^4 := 
by
  sorry

end calculate_expression_l1158_115825


namespace math_problem_l1158_115804

theorem math_problem (x : ℝ) :
  (x^3 - 8*x^2 + 16*x > 64) ∧ (x^2 - 4*x + 5 > 0) → x > 4 :=
by
  sorry

end math_problem_l1158_115804


namespace comb_identity_a_l1158_115883

theorem comb_identity_a (r m k : ℕ) (h : 0 ≤ k ∧ k ≤ m ∧ m ≤ r) :
  Nat.choose r m * Nat.choose m k = Nat.choose r k * Nat.choose (r - k) (m - k) :=
sorry

end comb_identity_a_l1158_115883


namespace number_of_cars_parked_l1158_115866

-- Definitions for the given conditions
def total_area (length width : ℕ) : ℕ := length * width
def usable_area (total : ℕ) : ℕ := (8 * total) / 10
def cars_parked (usable : ℕ) (area_per_car : ℕ) : ℕ := usable / area_per_car

-- Given conditions
def length : ℕ := 400
def width : ℕ := 500
def area_per_car : ℕ := 10
def expected_cars : ℕ := 16000 -- correct answer from solution

-- Define a proof statement
theorem number_of_cars_parked : cars_parked (usable_area (total_area length width)) area_per_car = expected_cars := by
  sorry

end number_of_cars_parked_l1158_115866


namespace marbles_ratio_l1158_115869

theorem marbles_ratio (miriam_current_marbles miriam_initial_marbles marbles_brother marbles_sister marbles_total_given marbles_savanna : ℕ)
  (h1 : miriam_current_marbles = 30)
  (h2 : marbles_brother = 60)
  (h3 : marbles_sister = 2 * marbles_brother)
  (h4 : miriam_initial_marbles = 300)
  (h5 : marbles_total_given = miriam_initial_marbles - miriam_current_marbles)
  (h6 : marbles_savanna = marbles_total_given - (marbles_brother + marbles_sister)) :
  (marbles_savanna : ℚ) / miriam_current_marbles = 3 :=
by
  sorry

end marbles_ratio_l1158_115869


namespace optimal_roof_angle_no_friction_l1158_115850

theorem optimal_roof_angle_no_friction {g x : ℝ} (hg : 0 < g) (hx : 0 < x) :
  ∃ α : ℝ, α = 45 :=
by
  sorry

end optimal_roof_angle_no_friction_l1158_115850


namespace abc_equal_l1158_115827

theorem abc_equal (a b c : ℝ) (h : a^2 + b^2 + c^2 = a * b + b * c + c * a) : a = b ∧ b = c :=
by
  sorry

end abc_equal_l1158_115827


namespace remove_terms_for_desired_sum_l1158_115847

theorem remove_terms_for_desired_sum :
  let series_sum := (1/3) + (1/5) + (1/7) + (1/9) + (1/11) + (1/13)
  series_sum - (1/11 + 1/13) = 11/20 :=
by
  sorry

end remove_terms_for_desired_sum_l1158_115847


namespace triangle_sides_l1158_115893

theorem triangle_sides (a : ℕ) (h : a > 0) : 
  (a + 1) + (a + 2) > (a + 3) ∧ (a + 1) + (a + 3) > (a + 2) ∧ (a + 2) + (a + 3) > (a + 1) := 
by 
  sorry

end triangle_sides_l1158_115893


namespace proof_x_plus_y_l1158_115870

variables (x y : ℝ)

-- Definitions for the given conditions
def cond1 (x y : ℝ) : Prop := 2 * |x| + x + y = 18
def cond2 (x y : ℝ) : Prop := x + 2 * |y| - y = 14

theorem proof_x_plus_y (x y : ℝ) (h1 : cond1 x y) (h2 : cond2 x y) : x + y = 14 := by
  sorry

end proof_x_plus_y_l1158_115870


namespace find_temp_M_l1158_115810

section TemperatureProof

variables (M T W Th F : ℕ)

-- Conditions
def avg_temp_MTWT := (M + T + W + Th) / 4 = 48
def avg_temp_TWThF := (T + W + Th + F) / 4 = 40
def temp_F := F = 10

-- Proof
theorem find_temp_M (h1 : avg_temp_MTWT M T W Th)
                    (h2 : avg_temp_TWThF T W Th F)
                    (h3 : temp_F F)
                    : M = 42 :=
sorry

end TemperatureProof

end find_temp_M_l1158_115810


namespace range_of_m_l1158_115839

variable {x m : ℝ}

def condition_p (x : ℝ) : Prop := |x - 3| ≤ 2
def condition_q (x m : ℝ) : Prop := (x - m + 1) * (x - m - 1) ≤ 0

theorem range_of_m (m : ℝ) :
  (∀ x, ¬(condition_p x) → ¬(condition_q x m)) ∧ ¬(∀ x, ¬(condition_q x m) → ¬(condition_p x)) →
  2 < m ∧ m < 4 := 
sorry

end range_of_m_l1158_115839


namespace johns_speed_l1158_115840

theorem johns_speed :
  ∀ (v : ℝ), 
    (∀ (t : ℝ), 24 = 30 * (t + 4 / 60) → 24 = v * (t - 8 / 60)) → 
    v = 40 :=
by
  intros
  sorry

end johns_speed_l1158_115840


namespace additional_discount_A_is_8_l1158_115814

-- Define the problem conditions
def full_price_A : ℝ := 125
def full_price_B : ℝ := 130
def discount_B : ℝ := 0.10
def price_difference : ℝ := 2

-- Define the unknown additional discount of store A
def discount_A (x : ℝ) : Prop :=
  full_price_A - (full_price_A * (x / 100)) = (full_price_B - (full_price_B * discount_B)) - price_difference

-- Theorem stating that the additional discount offered by store A is 8%
theorem additional_discount_A_is_8 : discount_A 8 :=
by
  -- Proof can be filled in here
  sorry

end additional_discount_A_is_8_l1158_115814


namespace total_students_l1158_115812

theorem total_students (m f : ℕ) (h_ratio : 3 * f = 7 * m) (h_males : m = 21) : m + f = 70 :=
by
  sorry

end total_students_l1158_115812


namespace infinite_positive_integer_solutions_l1158_115894

theorem infinite_positive_integer_solutions : ∃ (a b c : ℕ), (∃ k : ℕ, k > 0 ∧ a = k * (k^3 + 1990) ∧ b = (k^3 + 1990) ∧ c = (k^3 + 1990)) ∧ (a^3 + 1990 * b^3) = c^4 :=
sorry

end infinite_positive_integer_solutions_l1158_115894


namespace molecular_weight_CaO_l1158_115805

def atomic_weight_Ca : Float := 40.08
def atomic_weight_O : Float := 16.00

def molecular_weight (atoms : List (String × Float)) : Float :=
  atoms.foldr (fun (_, w) acc => w + acc) 0.0

theorem molecular_weight_CaO :
  molecular_weight [("Ca", atomic_weight_Ca), ("O", atomic_weight_O)] = 56.08 :=
by
  sorry

end molecular_weight_CaO_l1158_115805


namespace find_general_term_arithmetic_sequence_l1158_115819

-- Definitions needed
variable {a_n : ℕ → ℚ}
variable {S_n : ℕ → ℚ}

-- The main theorem to prove
theorem find_general_term_arithmetic_sequence 
  (h1 : a_n 4 - a_n 2 = 4)
  (h2 : S_n 3 = 9)
  (h3 : ∀ n : ℕ, S_n n = n / 2 * (2 * (a_n 1) + (n - 1) * (a_n 2 - a_n 1))) :
  (∀ n : ℕ, a_n n = 2 * n - 1) :=
by
  sorry

end find_general_term_arithmetic_sequence_l1158_115819


namespace cricket_scores_l1158_115821

-- Define the conditions
variable (X : ℝ) (A B C D E average10 average6 : ℝ)
variable (matches10 matches6 : ℕ)

-- Set the given constants
axiom average_runs_10 : average10 = 38.9
axiom matches_10 : matches10 = 10
axiom average_runs_6 : average6 = 42
axiom matches_6 : matches6 = 6

-- Define the equations based on the conditions
axiom eq1 : X = average10 * matches10
axiom eq2 : A + B + C + D = X - (average6 * matches6)
axiom eq3 : E = (A + B + C + D) / 4

-- The target statement
theorem cricket_scores : X = 389 ∧ A + B + C + D = 137 ∧ E = 34.25 :=
  by
    sorry

end cricket_scores_l1158_115821


namespace stone_length_l1158_115867

theorem stone_length (hall_length_m : ℕ) (hall_breadth_m : ℕ) (number_of_stones : ℕ) (stone_width_dm : ℕ) 
    (length_in_dm : 10 > 0) :
    hall_length_m = 36 → hall_breadth_m = 15 → number_of_stones = 2700 → stone_width_dm = 5 →
    ∀ L : ℕ, 
    (10 * hall_length_m) * (10 * hall_breadth_m) = number_of_stones * (L * stone_width_dm) → 
    L = 4 :=
by
  intros h1 h2 h3 h4
  simp at *
  sorry

end stone_length_l1158_115867


namespace number_value_l1158_115857

theorem number_value (x : ℝ) (h : x = 3 * (1/x * -x) + 5) : x = 2 :=
by
  sorry

end number_value_l1158_115857


namespace unique_solution_exists_l1158_115892

theorem unique_solution_exists :
  ∃ (a b c d e : ℕ),
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ e ≠ 0 ∧
  a + b = 1/7 * (c + d + e) ∧
  a + c = 1/5 * (b + d + e) ∧
  (a, b, c, d, e) = (1, 2, 3, 9, 9) :=
by {
  sorry
}

end unique_solution_exists_l1158_115892


namespace reduced_price_l1158_115885

variable (P R : ℝ)
variable (price_reduction : R = 0.75 * P)
variable (buy_more_oil : 700 / R = 700 / P + 5)

theorem reduced_price (non_zero_P : P ≠ 0) (non_zero_R : R ≠ 0) : R = 35 := 
by
  sorry

end reduced_price_l1158_115885


namespace length_BC_l1158_115800

theorem length_BC {A B C : ℝ} (r1 r2 : ℝ) (AB : ℝ) (h1 : r1 = 8) (h2 : r2 = 5) (h3 : AB = r1 + r2) :
  C = B + (65 : ℝ) / 3 :=
by
  -- Problem set-up and solving comes here if needed
  sorry

end length_BC_l1158_115800


namespace newspaper_price_l1158_115824

-- Define the conditions as variables
variables 
  (P : ℝ)                    -- Price per edition for Wednesday, Thursday, and Friday
  (total_cost : ℝ := 28)     -- Total cost over 8 weeks
  (sunday_cost : ℝ := 2)     -- Cost of Sunday edition
  (weeks : ℕ := 8)           -- Number of weeks
  (wednesday_thursday_friday_editions : ℕ := 3 * weeks) -- Total number of editions for Wednesday, Thursday, and Friday over 8 weeks

-- Math proof problem statement
theorem newspaper_price : 
  (total_cost - weeks * sunday_cost) / wednesday_thursday_friday_editions = 0.5 :=
  sorry

end newspaper_price_l1158_115824


namespace smallest_nat_number_l1158_115843

theorem smallest_nat_number (x : ℕ) (h1 : 5 ∣ x) (h2 : 7 ∣ x) (h3 : x % 3 = 1) : x = 70 :=
sorry

end smallest_nat_number_l1158_115843


namespace range_of_a_l1158_115881

-- Definitions of propositions p and q

def p (a : ℝ) : Prop := ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 - a ≥ 0
def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0

-- Theorem stating the range of values for a given p ∧ q is true

theorem range_of_a (a : ℝ) : (p a ∧ q a) → (a ≤ -2 ∨ a = 1) :=
by sorry

end range_of_a_l1158_115881


namespace patrick_purchased_pencils_l1158_115863

theorem patrick_purchased_pencils 
  (S : ℝ) -- selling price of one pencil
  (C : ℝ) -- cost price of one pencil
  (P : ℕ) -- number of pencils purchased
  (h1 : C = 1.3333333333333333 * S) -- condition 1: cost of pencils is 1.3333333 times the selling price
  (h2 : (P : ℝ) * C - (P : ℝ) * S = 20 * S) -- condition 2: loss equals selling price of 20 pencils
  : P = 60 := 
sorry

end patrick_purchased_pencils_l1158_115863


namespace leo_trousers_count_l1158_115890

theorem leo_trousers_count (S T : ℕ) (h1 : 5 * S + 9 * T = 140) (h2 : S = 10) : T = 10 :=
by
  sorry

end leo_trousers_count_l1158_115890


namespace combine_like_terms_substitute_expression_complex_expression_l1158_115888

-- Part 1
theorem combine_like_terms (a b : ℝ) : 
  10 * (a - b)^2 - 12 * (a - b)^2 + 9 * (a - b)^2 = 7 * (a - b)^2 :=
by
  sorry

-- Part 2
theorem substitute_expression (x y : ℝ) (h1 : x^2 - 2 * y = -5) : 
  4 * x^2 - 8 * y + 24 = 4 :=
by
  sorry

-- Part 3
theorem complex_expression (a b c d : ℝ) 
  (h1 : a - 2 * b = 1009.5) 
  (h2 : 2 * b - c = -2024.6666)
  (h3 : c - d = 1013.1666) : 
  (a - c) + (2 * b - d) - (2 * b - c) = -2 :=
by
  sorry

end combine_like_terms_substitute_expression_complex_expression_l1158_115888


namespace simplify_and_evaluate_l1158_115858

-- Defining the conditions
def a : Int := -3
def b : Int := -2

-- Defining the expression
def expr (a b : Int) : Int := (3 * a^2 * b + 2 * a * b^2) - (2 * (a^2 * b - 1) + 3 * a * b^2 + 2)

-- Stating the theorem/proof problem
theorem simplify_and_evaluate : expr a b = -6 := by
  sorry

end simplify_and_evaluate_l1158_115858


namespace sufficient_not_necessary_condition_l1158_115864

variable {a : ℝ}

theorem sufficient_not_necessary_condition (ha : a > 1 / a^2) :
  a^2 > 1 / a ∧ ∃ a, a^2 > 1 / a ∧ ¬(a > 1 / a^2) :=
by
  sorry

end sufficient_not_necessary_condition_l1158_115864


namespace evaluate_expression_at_2_l1158_115803

theorem evaluate_expression_at_2 : (3^2 - 2^3) = 1 := 
by
  sorry

end evaluate_expression_at_2_l1158_115803


namespace distance_between_joe_and_gracie_l1158_115818

open Complex

noncomputable def joe_point : ℂ := 2 + 3 * I
noncomputable def gracie_point : ℂ := -2 + 2 * I
noncomputable def distance := abs (joe_point - gracie_point)

theorem distance_between_joe_and_gracie :
  distance = Real.sqrt 17 := by
  sorry

end distance_between_joe_and_gracie_l1158_115818


namespace min_distance_to_line_l1158_115815

theorem min_distance_to_line : 
  let A := 5
  let B := -3
  let C := 4
  let d (x₀ y₀ : ℤ) := (abs (A * x₀ + B * y₀ + C) : ℝ) / (Real.sqrt (A ^ 2 + B ^ 2))
  ∃ (x₀ y₀ : ℤ), d x₀ y₀ = Real.sqrt 34 / 85 := 
by 
  sorry

end min_distance_to_line_l1158_115815


namespace area_of_given_rhombus_l1158_115849

open Real

noncomputable def area_of_rhombus_with_side_and_angle (side : ℝ) (angle : ℝ) : ℝ :=
  let half_diag1 := side * cos (angle / 2)
  let half_diag2 := side * sin (angle / 2)
  let diag1 := 2 * half_diag1
  let diag2 := 2 * half_diag2
  (diag1 * diag2) / 2

theorem area_of_given_rhombus :
  area_of_rhombus_with_side_and_angle 25 40 = 201.02 :=
by
  sorry

end area_of_given_rhombus_l1158_115849


namespace find_surface_area_of_ball_l1158_115838

noncomputable def surface_area_of_ball : ℝ :=
  let tetrahedron_edge := 4
  let tetrahedron_volume := (1 / 3) * (Real.sqrt 3 / 4) * tetrahedron_edge ^ 2 * (Real.sqrt 16 - 16 / 3)
  let water_volume := (7 / 8) * tetrahedron_volume
  let remaining_volume := (1 / 8) * tetrahedron_volume
  let remaining_edge := 2
  let ball_radius := Real.sqrt 6 / 6
  let surface_area := 4 * Real.pi * ball_radius ^ 2
  surface_area

theorem find_surface_area_of_ball :
  let tetrahedron_edge := 4
  let tetrahedron_volume := (1 / 3) * (Real.sqrt 3 / 4) * tetrahedron_edge ^ 2 * (Real.sqrt 16 - 16 / 3)
  let water_volume := (7 / 8) * tetrahedron_volume
  let remaining_volume := (1 / 8) * tetrahedron_volume
  let remaining_edge := 2
  let ball_radius := Real.sqrt 6 / 6
  let surface_area := 4 * Real.pi * ball_radius ^ 2
  surface_area = (2 / 3) * Real.pi :=
by
  let tetrahedron_edge := 4
  let tetrahedron_volume := (1 / 3) * (Real.sqrt 3 / 4) * tetrahedron_edge ^ 2 * (Real.sqrt 16 - 16 / 3)
  let water_volume := (7 / 8) * tetrahedron_volume
  let remaining_volume := (1 / 8) * tetrahedron_volume
  let remaining_edge := 2
  let ball_radius := Real.sqrt 6 / 6
  let surface_area := 4 * Real.pi * ball_radius ^ 2
  sorry

end find_surface_area_of_ball_l1158_115838


namespace solve_for_N_l1158_115842

theorem solve_for_N (N : ℤ) (h1 : N < 0) (h2 : 2 * N * N + N = 15) : N = -3 :=
sorry

end solve_for_N_l1158_115842


namespace period_tan_2x_3_l1158_115882

noncomputable def period_of_tan_transformed : Real :=
  let period_tan := Real.pi
  let coeff := 2/3
  (period_tan / coeff : Real)

theorem period_tan_2x_3 : period_of_tan_transformed = 3 * Real.pi / 2 :=
  sorry

end period_tan_2x_3_l1158_115882


namespace total_copies_in_half_hour_l1158_115848

-- Define the rates of the copy machines
def rate_machine1 : ℕ := 35
def rate_machine2 : ℕ := 65

-- Define the duration of time in minutes
def time_minutes : ℕ := 30

-- Define the total number of copies made by both machines in the given duration
def total_copies_made : ℕ := rate_machine1 * time_minutes + rate_machine2 * time_minutes

-- Prove that the total number of copies made is 3000
theorem total_copies_in_half_hour : total_copies_made = 3000 := by
  -- The proof is skipped with sorry for the demonstration purpose
  sorry

end total_copies_in_half_hour_l1158_115848


namespace range_of_a_l1158_115811

-- Define the function f
def f (a x : ℝ) : ℝ := -x^3 + a * x^2 - x - 1

-- Define the derivative of f
def f_prime (a x : ℝ) : ℝ := -3 * x^2 + 2 * a * x - 1

-- State the theorem
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f_prime a x ≤ 0) ↔ -Real.sqrt 3 ≤ a ∧ a ≤ Real.sqrt 3 :=
sorry

end range_of_a_l1158_115811


namespace ways_to_stand_on_staircase_l1158_115835

theorem ways_to_stand_on_staircase (A B C : Type) (steps : Fin 7) : 
  ∃ ways : Nat, ways = 336 := by sorry

end ways_to_stand_on_staircase_l1158_115835


namespace ratio_unit_price_l1158_115823

theorem ratio_unit_price (v p : ℝ) (hv : v > 0) (hp : p > 0) :
  let vX := 1.25 * v
  let pX := 0.85 * p
  (pX / vX) / (p / v) = 17 / 25 := by
{
  sorry
}

end ratio_unit_price_l1158_115823


namespace calc_expr_eq_l1158_115879

-- Define the polynomial and expression
def expr (x : ℝ) : ℝ := x * (x * (x * (3 - 2 * x) - 4) + 8) + 3 * x^2

theorem calc_expr_eq (x : ℝ) : expr x = -2 * x^4 + 3 * x^3 - x^2 + 8 * x := 
by
  sorry

end calc_expr_eq_l1158_115879


namespace arithmetic_mean_of_fractions_l1158_115853

theorem arithmetic_mean_of_fractions :
  let a := (3 : ℚ) / 4
  let b := (5 : ℚ) / 8
  (a + b) / 2 = 11 / 16 :=
by 
  let a := (3 : ℚ) / 4
  let b := (5 : ℚ) / 8
  show (a + b) / 2 = 11 / 16
  sorry

end arithmetic_mean_of_fractions_l1158_115853


namespace solve_for_x_l1158_115817

theorem solve_for_x (x : ℝ) (h : 9 / (5 + x / 0.75) = 1) : x = 3 :=
by {
  sorry
}

end solve_for_x_l1158_115817


namespace lice_checks_time_in_hours_l1158_115887

-- Define the number of students in each grade
def kindergarteners : ℕ := 26
def first_graders : ℕ := 19
def second_graders : ℕ := 20
def third_graders : ℕ := 25

-- Define the time each check takes (in minutes)
def time_per_check : ℕ := 2

-- Define the conversion factor from minutes to hours
def minutes_per_hour : ℕ := 60

-- The theorem states that the total time in hours is 3
theorem lice_checks_time_in_hours : 
  ((kindergarteners + first_graders + second_graders + third_graders) * time_per_check) / minutes_per_hour = 3 := 
by
  sorry

end lice_checks_time_in_hours_l1158_115887


namespace smallest_x_value_l1158_115831

theorem smallest_x_value (x : ℝ) (h : 3 * (8 * x^2 + 10 * x + 12) = x * (8 * x - 36)) : x = -3 :=
sorry

end smallest_x_value_l1158_115831


namespace PB_length_l1158_115844

/-- In a square ABCD with area 1989 cm², with the center O, and
a point P inside such that ∠OPB = 45° and PA : PB = 5 : 14,
prove that PB = 42 cm. -/
theorem PB_length (s PA PB : ℝ) (h₁ : s^2 = 1989) 
(h₂ : PA / PB = 5 / 14) 
(h₃ : 25 * (PA / PB)^2 + 196 * (PB / PA)^2 = s^2) :
  PB = 42 := 
by sorry

end PB_length_l1158_115844


namespace range_of_a_l1158_115859

theorem range_of_a
  (x : ℝ) (hx : 0 ≤ x ∧ x ≤ π / 6)
  (y : ℝ) (hy : 0 < y)
  (h : (y / 4 - 2 * (Real.cos x)^2) ≥ a * (Real.sin x) - 9 / y) :
  a ≤ 3 :=
sorry

end range_of_a_l1158_115859


namespace triplet_solution_l1158_115868

theorem triplet_solution (x y z : ℝ) 
  (h1 : y = (x^3 + 12 * x) / (3 * x^2 + 4))
  (h2 : z = (y^3 + 12 * y) / (3 * y^2 + 4))
  (h3 : x = (z^3 + 12 * z) / (3 * z^2 + 4)) :
  (x = 0 ∧ y = 0 ∧ z = 0) ∨ 
  (x = 2 ∧ y = 2 ∧ z = 2) ∨ 
  (x = -2 ∧ y = -2 ∧ z = -2) :=
sorry

end triplet_solution_l1158_115868


namespace solve_system_of_equations_l1158_115801

theorem solve_system_of_equations :
  ∃ x y : ℤ, (2 * x + 7 * y = -6) ∧ (2 * x - 5 * y = 18) ∧ (x = 4) ∧ (y = -2) := 
by
  -- Proof will go here
  sorry

end solve_system_of_equations_l1158_115801


namespace tony_bread_slices_left_l1158_115898

def num_slices_left (initial_slices : ℕ) (d1 : ℕ) (d2 : ℕ) : ℕ :=
  initial_slices - (d1 + d2)

theorem tony_bread_slices_left :
  num_slices_left 22 (5 * 2) (2 * 2) = 8 := by
  sorry

end tony_bread_slices_left_l1158_115898


namespace count_interesting_quadruples_l1158_115806

def interesting_quadruples (a b c d : ℤ) : Prop :=
  1 ≤ a ∧ a < b ∧ b < c ∧ c < d ∧ d ≤ 15 ∧ a + 2 * d > b + 2 * c 

theorem count_interesting_quadruples : 
  (∃ n : ℤ, n = 582 ∧ ∀ a b c d : ℤ, interesting_quadruples a b c d → n = 582) :=
sorry

end count_interesting_quadruples_l1158_115806


namespace james_marbles_left_l1158_115851

theorem james_marbles_left (initial_marbles : ℕ) (total_bags : ℕ) (marbles_per_bag : ℕ) (bags_given_away : ℕ) : 
  initial_marbles = 28 → total_bags = 4 → marbles_per_bag = initial_marbles / total_bags → bags_given_away = 1 → 
  initial_marbles - marbles_per_bag * bags_given_away = 21 :=
by
  intros h_initial h_total h_each h_given
  sorry

end james_marbles_left_l1158_115851


namespace factor_polynomial_l1158_115820

theorem factor_polynomial :
  ∀ x : ℝ, 
  (x^2 + 4 * x + 3) * (x^2 + 8 * x + 15) + (x^2 + 6 * x - 8) 
  = (x^2 + 6 * x + 1) * (x^2 + 6 * x + 37) :=
by
  intros x
  sorry

end factor_polynomial_l1158_115820


namespace value_of_b_minus_d_squared_l1158_115816

theorem value_of_b_minus_d_squared
  (a b c d : ℤ)
  (h1 : a - b - c + d = 13)
  (h2 : a + b - c - d = 9) :
  (b - d) ^ 2 = 4 :=
sorry

end value_of_b_minus_d_squared_l1158_115816


namespace census_entirety_is_population_l1158_115884

-- Define the options as a type
inductive CensusOptions
| Part
| Whole
| Individual
| Population

-- Define the condition: the entire object under investigation in a census
def entirety_of_objects_under_investigation : CensusOptions := CensusOptions.Population

-- Prove that the entirety of objects under investigation in a census is called Population
theorem census_entirety_is_population :
  entirety_of_objects_under_investigation = CensusOptions.Population :=
sorry

end census_entirety_is_population_l1158_115884


namespace square_perimeter_l1158_115826

def perimeter_of_square (side_length : ℝ) : ℝ :=
  4 * side_length

theorem square_perimeter (side_length : ℝ) (h : side_length = 5) : perimeter_of_square side_length = 20 := by
  sorry

end square_perimeter_l1158_115826


namespace find_theta_l1158_115807

-- Define the angles
variables (VEK KEW EVG θ : ℝ)

-- State the conditions as hypotheses
def conditions (VEK KEW EVG θ : ℝ) := 
  VEK = 70 ∧
  KEW = 40 ∧
  EVG = 110

-- State the theorem
theorem find_theta (VEK KEW EVG θ : ℝ)
  (h : conditions VEK KEW EVG θ) : 
  θ = 40 :=
by {
  sorry
}

end find_theta_l1158_115807


namespace percentage_decrease_is_14_percent_l1158_115877

-- Definitions based on conditions
def original_price_per_pack : ℚ := 7 / 3
def new_price_per_pack : ℚ := 8 / 4

-- Statement to prove that percentage decrease is 14%
theorem percentage_decrease_is_14_percent :
  ((original_price_per_pack - new_price_per_pack) / original_price_per_pack) * 100 = 14 := by
  sorry

end percentage_decrease_is_14_percent_l1158_115877


namespace not_geometric_sequence_of_transformed_l1158_115873

theorem not_geometric_sequence_of_transformed (a b c : ℝ) (q : ℝ) (hq : q ≠ 1) 
  (h_geometric : b = a * q ∧ c = b * q) :
  ¬ (∃ q' : ℝ, 1 - b = (1 - a) * q' ∧ 1 - c = (1 - b) * q') :=
by
  sorry

end not_geometric_sequence_of_transformed_l1158_115873


namespace eccentricity_condition_l1158_115899

theorem eccentricity_condition (m : ℝ) (h : 0 < m) : 
  (m < (4 / 3) ∨ m > (3 / 4)) ↔ ((1 - m) > (1 / 4) ∨ ((m - 1) / m) > (1 / 4)) :=
by
  sorry

end eccentricity_condition_l1158_115899


namespace systematic_sampling_first_group_l1158_115813

theorem systematic_sampling_first_group 
  (total_students sample_size group_size group_number drawn_number : ℕ)
  (h1 : total_students = 160)
  (h2 : sample_size = 20)
  (h3 : total_students = sample_size * group_size)
  (h4 : group_number = 16)
  (h5 : drawn_number = 126) 
  : (drawn_lots_first_group : ℕ) 
      = ((drawn_number - ((group_number - 1) * group_size + 1)) + 1) :=
sorry


end systematic_sampling_first_group_l1158_115813


namespace complement_intersection_l1158_115828

open Set

def A : Set ℝ := {x | x > 0}
def B : Set ℝ := {-2, -1, 1, 2}

theorem complement_intersection :
  compl A ∩ B = {-2, -1} :=
by
  sorry

end complement_intersection_l1158_115828


namespace sean_div_julie_eq_two_l1158_115808

def sum_n (n : ℕ) := n * (n + 1) / 2

def sean_sum := 2 * sum_n 500

def julie_sum := sum_n 500

theorem sean_div_julie_eq_two : sean_sum / julie_sum = 2 := 
by sorry

end sean_div_julie_eq_two_l1158_115808


namespace determinant_zero_l1158_115874

def matrix_determinant (x y z : ℝ) : ℝ :=
  Matrix.det ![
    ![1, x, y + z],
    ![1, x + y, z],
    ![1, x + z, y]
  ]

theorem determinant_zero (x y z : ℝ) : matrix_determinant x y z = 0 := 
by
  sorry

end determinant_zero_l1158_115874


namespace ten_percent_of_x_is_17_85_l1158_115865

-- Define the conditions and the proof statement
theorem ten_percent_of_x_is_17_85 :
  ∃ x : ℝ, (3 - (1/4) * 2 - (1/3) * 3 - (1/7) * x = 27) ∧ (0.10 * x = 17.85) := sorry

end ten_percent_of_x_is_17_85_l1158_115865


namespace inequality_abc_lt_l1158_115876

variable (a b c : ℝ)

theorem inequality_abc_lt:
  c > b → b > a → a^2 * b + b^2 * c + c^2 * a < a * b^2 + b * c^2 + c * a^2 :=
by
  intros h1 h2
  sorry

end inequality_abc_lt_l1158_115876


namespace sum_of_ages_is_26_l1158_115861

def Yoongi_aunt_age := 38
def Yoongi_age := Yoongi_aunt_age - 23
def Hoseok_age := Yoongi_age - 4
def sum_of_ages := Yoongi_age + Hoseok_age

theorem sum_of_ages_is_26 : sum_of_ages = 26 :=
by
  sorry

end sum_of_ages_is_26_l1158_115861


namespace rational_solutions_zero_l1158_115809

theorem rational_solutions_zero (x y z : ℚ) (h : x^3 + 3*y^3 + 9*z^3 - 9*x*y*z = 0) : x = 0 ∧ y = 0 ∧ z = 0 :=
by 
  sorry

end rational_solutions_zero_l1158_115809


namespace necessary_condition_not_sufficient_condition_main_l1158_115860

example (x : ℝ) : (x^2 - 3 * x > 0) → (x > 4) ∨ (x < 0 ∧ x > 0) := by
  sorry

theorem necessary_condition (x : ℝ) :
  (x^2 - 3 * x > 0) → (x > 4) :=
by
  sorry

theorem not_sufficient_condition (x : ℝ) :
  ¬ (x > 4) → (x^2 - 3 * x > 0) :=
by
  sorry

theorem main (x : ℝ) :
  (x^2 - 3 * x > 0) ↔ ¬ (x > 4) :=
by
  sorry

end necessary_condition_not_sufficient_condition_main_l1158_115860
