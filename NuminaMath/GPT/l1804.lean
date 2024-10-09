import Mathlib

namespace alpha_lt_beta_of_acute_l1804_180402

open Real

theorem alpha_lt_beta_of_acute (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2)
  (h : 2 * sin α = sin α * cos β + cos α * sin β) : α < β :=
by
  sorry

end alpha_lt_beta_of_acute_l1804_180402


namespace parking_spots_l1804_180434

def numberOfLevels := 5
def openSpotsOnLevel1 := 4
def openSpotsOnLevel2 := openSpotsOnLevel1 + 7
def openSpotsOnLevel3 := openSpotsOnLevel2 + 6
def openSpotsOnLevel4 := 14
def openSpotsOnLevel5 := openSpotsOnLevel4 + 5
def totalOpenSpots := openSpotsOnLevel1 + openSpotsOnLevel2 + openSpotsOnLevel3 + openSpotsOnLevel4 + openSpotsOnLevel5

theorem parking_spots :
  openSpotsOnLevel5 = 19 ∧ totalOpenSpots = 65 := by
  sorry

end parking_spots_l1804_180434


namespace qin_jiushao_operations_required_l1804_180499

def polynomial (x : ℝ) : ℝ := 5 * x^5 + 4 * x^4 + 3 * x^3 + 2 * x^2 + x + 1

theorem qin_jiushao_operations_required : 
  (∃ x : ℝ, polynomial x = (((((5 * x + 4) * x + 3) * x + 2) * x + 1) * x + 1)) →
  (∃ m a : ℕ, m = 5 ∧ a = 5) := by
  sorry

end qin_jiushao_operations_required_l1804_180499


namespace gcd_16_12_eq_4_l1804_180423

theorem gcd_16_12_eq_4 : Nat.gcd 16 12 = 4 := by
  -- Skipping proof using sorry
  sorry

end gcd_16_12_eq_4_l1804_180423


namespace constant_sequence_l1804_180401

theorem constant_sequence (a : ℕ → ℕ) (h : ∀ i j, 1 ≤ i ∧ i ≤ 2016 ∧ 1 ≤ j ∧ j ≤ 2016 → (i + j) ∣ (i * a i + j * a j)) :
  ∀ i j, 1 ≤ i ∧ i ≤ 2016 ∧ 1 ≤ j ∧ j ≤ 2016 → a i = a j :=
by
  sorry

end constant_sequence_l1804_180401


namespace matches_start_with_l1804_180424

-- Let M be the number of matches Nate started with
variables (M : ℕ)

-- Given conditions
def dropped_creek (dropped : ℕ) := dropped = 10
def eaten_by_dog (eaten : ℕ) := eaten = 2 * 10
def matches_left (final_matches : ℕ) := final_matches = 40

-- Prove that the number of matches Nate started with is 70
theorem matches_start_with 
  (h1 : dropped_creek 10)
  (h2 : eaten_by_dog 20)
  (h3 : matches_left 40) 
  : M = 70 :=
sorry

end matches_start_with_l1804_180424


namespace solve_for_x_l1804_180455

theorem solve_for_x (x : ℚ) (h : (1 / 3 - 1 / 4 = 4 / x)) : x = 48 := by
  sorry

end solve_for_x_l1804_180455


namespace find_integer_solutions_l1804_180428

theorem find_integer_solutions :
  {p : ℤ × ℤ | 2 * p.1^3 + p.1 * p.2 = 7} = {(-7, -99), (-1, -9), (1, 5), (7, -97)} :=
by
  -- Proof not required
  sorry

end find_integer_solutions_l1804_180428


namespace triangle_angles_l1804_180474

noncomputable def angle_triangle (E : ℝ) :=
if E = 45 then (90, 45, 45) else if E = 36 then (72, 72, 36) else (0, 0, 0)

theorem triangle_angles (E : ℝ) :
  (∃ E, E = 45 → angle_triangle E = (90, 45, 45))
  ∨
  (∃ E, E = 36 → angle_triangle E = (72, 72, 36)) :=
by
    sorry

end triangle_angles_l1804_180474


namespace correct_propositions_l1804_180454

-- Definitions of propositions
def prop1 (f : ℝ → ℝ) : Prop :=
  f (-2) ≠ f (2) → ∀ x : ℝ, f (-x) ≠ f (x)

def prop2 : Prop :=
  ∀ n : ℕ, n = 0 ∨ n = 1 → (∀ x : ℝ, x ≠ 0 → x ^ n ≠ 0)

def prop3 : Prop :=
  ∀ a b : ℝ, (a ≠ 0 ∧ b ≠ 0) → (a * b ≠ 0) ∧ (a * b = 0 → a = 0 ∨ b = 0)

def prop4 (a b c d : ℝ) : Prop :=
  ∀ x : ℝ, ∃ k : ℝ, k = d → (3 * a * x ^ 2 + 2 * b * x + c ≠ 0 ∧ b ^ 2 - 3 * a * c ≥ 0)

-- Final proof statement
theorem correct_propositions (f : ℝ → ℝ) (a b c d : ℝ) :
  prop1 f ∧ ¬prop2 ∧ prop3 ∧ ¬prop4 a b c d :=
sorry

end correct_propositions_l1804_180454


namespace ratio_of_legs_of_triangles_l1804_180471

theorem ratio_of_legs_of_triangles (s a b : ℝ) (h1 : 0 < s)
  (h2 : a = s / 2)
  (h3 : b = (s * Real.sqrt 7) / 2) :
  b / a = Real.sqrt 7 := by
  sorry

end ratio_of_legs_of_triangles_l1804_180471


namespace original_paint_intensity_l1804_180462

theorem original_paint_intensity (I : ℝ) (h1 : 0.5 * I + 0.5 * 20 = 15) : I = 10 :=
sorry

end original_paint_intensity_l1804_180462


namespace problem_l1804_180478

noncomputable def f (x : ℝ) : ℝ := 5 * x - 7
noncomputable def g (x : ℝ) : ℝ := x / 5 + 3

theorem problem : ∀ x : ℝ, f (g x) - g (f x) = 6.4 :=
by
  intro x
  sorry

end problem_l1804_180478


namespace carrie_phone_charges_l1804_180492

def total_miles (d1 d2 d3 d4 : ℕ) : ℕ :=
  d1 + d2 + d3 + d4

def charges_needed (total_miles charge_miles : ℕ) : ℕ :=
  total_miles / charge_miles + if total_miles % charge_miles = 0 then 0 else 1

theorem carrie_phone_charges :
  let d1 := 135
  let d2 := 135 + 124
  let d3 := 159
  let d4 := 189
  let charge_miles := 106
  charges_needed (total_miles d1 d2 d3 d4) charge_miles = 7 :=
by
  sorry

end carrie_phone_charges_l1804_180492


namespace general_formula_sequence_l1804_180408

-- Define the sequence as an arithmetic sequence with the given first term and common difference
def arithmetic_sequence (a_1 d : ℕ) (n : ℕ) : ℕ := a_1 + (n - 1) * d

-- Define given values
def a_1 : ℕ := 1
def d : ℕ := 2

-- State the theorem to be proved
theorem general_formula_sequence :
  ∀ n : ℕ, n > 0 → arithmetic_sequence a_1 d n = 2 * n - 1 :=
by
  intro n hn
  sorry

end general_formula_sequence_l1804_180408


namespace other_factor_of_936_mul_w_l1804_180435

theorem other_factor_of_936_mul_w (w : ℕ) (h_w_pos : 0 < w)
  (h_factors_936w : ∃ k, 936 * w = k * (3^3)) 
  (h_factors_936w_2 : ∃ m, 936 * w = m * (10^2))
  (h_w : w = 120):
  ∃ n, n = 45 :=
by
  sorry

end other_factor_of_936_mul_w_l1804_180435


namespace greatest_possible_a_l1804_180409

theorem greatest_possible_a (a : ℕ) :
  (∃ x : ℤ, x^2 + a * x = -28) → a = 29 :=
sorry

end greatest_possible_a_l1804_180409


namespace minute_first_catch_hour_l1804_180481

theorem minute_first_catch_hour :
  ∃ (t : ℚ), t = 60 * (1 + (5 / 11)) :=
sorry

end minute_first_catch_hour_l1804_180481


namespace solve_quadratic1_solve_quadratic2_l1804_180436

-- Equation 1
theorem solve_quadratic1 (x : ℝ) :
  (x = 4 + 3 * Real.sqrt 2 ∨ x = 4 - 3 * Real.sqrt 2) ↔ x ^ 2 - 8 * x - 2 = 0 := by
  sorry

-- Equation 2
theorem solve_quadratic2 (x : ℝ) :
  (x = 3 / 2 ∨ x = -1) ↔ 2 * x ^ 2 - x - 3 = 0 := by
  sorry

end solve_quadratic1_solve_quadratic2_l1804_180436


namespace linear_function_not_in_fourth_quadrant_l1804_180498

theorem linear_function_not_in_fourth_quadrant (a b : ℝ) (h : a = 2 ∧ b = 1) :
  ∀ (x : ℝ), (2 * x + 1 < 0 → x > 0) := 
sorry

end linear_function_not_in_fourth_quadrant_l1804_180498


namespace length_of_AB_l1804_180422

noncomputable def hyperbola_conditions (a b : ℝ) (hac : a > 0) (hbc : b = 2 * a) :=
  ∃ x y : ℝ, (x^2 / a^2) - (y^2 / b^2) = 1

def circle_intersection_condition (A B : ℝ × ℝ) :=
  ∃ (x1 y1 x2 y2 : ℝ), 
  (A = (x1, y1)) ∧ (B = (x2, y2)) ∧ ((x1 - 2)^2 + (y1 - 3)^2 = 1 ∧ y1 = 2 * x1) ∧
  ((x2 - 2)^2 + (y2 - 3)^2 = 1 ∧ y2 = 2 * x2)

theorem length_of_AB {a b : ℝ} (hac : a > 0) (hb : b = 2 * a) :
  (hyperbola_conditions a b hac hb) →
  ∃ (A B : ℝ × ℝ), circle_intersection_condition A B → 
  dist A B = (4 * Real.sqrt 5) / 5 :=
by
  sorry

end length_of_AB_l1804_180422


namespace possible_values_of_m_l1804_180475

-- Defining sets A and B based on the given conditions
def set_A : Set ℝ := { x | x^2 - 2 * x - 3 = 0 }
def set_B (m : ℝ) : Set ℝ := { x | m * x + 1 = 0 }

-- The main theorem statement
theorem possible_values_of_m (m : ℝ) :
  (set_A ∪ set_B m = set_A) ↔ (m = 0 ∨ m = -1 / 3 ∨ m = 1) := by
  sorry

end possible_values_of_m_l1804_180475


namespace polynomial_horner_method_l1804_180444

-- Define the polynomial f
def f (x : ℕ) :=
  7 * x ^ 7 + 6 * x ^ 6 + 5 * x ^ 5 + 4 * x ^ 4 + 3 * x ^ 3 + 2 * x ^ 2 + x

-- Define x as given in the condition
def x : ℕ := 3

-- State that f(x) = 262 when x = 3
theorem polynomial_horner_method : f x = 262 :=
  by
  sorry

end polynomial_horner_method_l1804_180444


namespace company_fund_initial_amount_l1804_180465

-- Let n be the number of employees in the company.
variable (n : ℕ)

-- Conditions from the problem.
def initial_fund := 60 * n - 10
def adjusted_fund := 50 * n + 150
def employees_count := 16

-- Given the conditions, prove that the initial fund amount was $950.
theorem company_fund_initial_amount
    (h1 : adjusted_fund n = initial_fund n)
    (h2 : n = employees_count) : 
    initial_fund n = 950 := by
  sorry

end company_fund_initial_amount_l1804_180465


namespace wheel_rotation_angle_l1804_180495

-- Define the conditions
def radius : ℝ := 20
def arc_length : ℝ := 40

-- Define the theorem stating the desired proof problem
theorem wheel_rotation_angle (r : ℝ) (l : ℝ) (h_r : r = radius) (h_l : l = arc_length) :
  l / r = 2 := 
by sorry

end wheel_rotation_angle_l1804_180495


namespace certain_number_exists_l1804_180430

theorem certain_number_exists :
  ∃ x : ℤ, 55 * x % 7 = 6 ∧ x % 7 = 1 := by
  sorry

end certain_number_exists_l1804_180430


namespace area_of_rectangle_l1804_180443

theorem area_of_rectangle (x y : ℝ)
    (h1 : x * y = (x + 3) * (y - 1))
    (h2 : x * y = (x - 4) * (y + 3 / 2)) :
    x * y = 108 := by
  sorry

end area_of_rectangle_l1804_180443


namespace smallest_y_for_perfect_cube_l1804_180450

theorem smallest_y_for_perfect_cube (x y : ℕ) (x_def : x = 11 * 36 * 54) : 
  (∃ y : ℕ, y > 0 ∧ ∀ (n : ℕ), (x * y = n^3 ↔ y = 363)) := 
by 
  sorry

end smallest_y_for_perfect_cube_l1804_180450


namespace older_friend_is_38_l1804_180421

-- Define the conditions
def younger_friend_age (x : ℕ) : Prop := 
  ∃ (y : ℕ), (y = x + 2 ∧ x + y = 74)

-- Define the age of the older friend
def older_friend_age (x : ℕ) : ℕ := x + 2

-- State the theorem
theorem older_friend_is_38 : ∃ x, younger_friend_age x ∧ older_friend_age x = 38 :=
by
  sorry

end older_friend_is_38_l1804_180421


namespace probability_both_correct_given_any_correct_l1804_180448

-- Defining the probabilities
def P_A : ℚ := 3 / 5
def P_B : ℚ := 1 / 3

-- Defining the events and their products
def P_AnotB : ℚ := P_A * (1 - P_B)
def P_notAB : ℚ := (1 - P_A) * P_B
def P_AB : ℚ := P_A * P_B

-- Calculated Probability of C
def P_C : ℚ := P_AnotB + P_notAB + P_AB

-- The proof statement
theorem probability_both_correct_given_any_correct : (P_AB / P_C) = 3 / 11 :=
by
  sorry

end probability_both_correct_given_any_correct_l1804_180448


namespace existence_of_k_good_function_l1804_180446

def is_k_good_function (f : ℕ+ → ℕ+) (k : ℕ) : Prop :=
  ∀ m n : ℕ+, m ≠ n → Nat.gcd (f m + n) (f n + m) ≤ k

theorem existence_of_k_good_function (k : ℕ) :
  (∃ f : ℕ+ → ℕ+, is_k_good_function f k) ↔ k ≥ 2 := sorry

end existence_of_k_good_function_l1804_180446


namespace triangle_inequality_l1804_180466

theorem triangle_inequality (a b c : ℝ) (h : a + b > c) (h2 : b + c > a) (h3 : c + a > b) : 
  a^2 * (b + c - a) + b^2 * (c + a - b) + c^2 * (a + b - c) ≤ 3 * a * b * c := 
sorry

end triangle_inequality_l1804_180466


namespace Hilltown_Volleyball_Club_Members_l1804_180470

-- Definitions corresponding to the conditions
def knee_pad_cost : ℕ := 6
def uniform_cost : ℕ := 14
def total_expenditure : ℕ := 4000

-- Definition of total cost per member
def cost_per_member : ℕ := 2 * (knee_pad_cost + uniform_cost)

-- Proof statement
theorem Hilltown_Volleyball_Club_Members :
  total_expenditure % cost_per_member = 0 ∧ total_expenditure / cost_per_member = 100 := by
    sorry

end Hilltown_Volleyball_Club_Members_l1804_180470


namespace total_cost_of_apples_l1804_180482

theorem total_cost_of_apples (cost_per_kg : ℝ) (packaging_fee : ℝ) (weight : ℝ) :
  cost_per_kg = 15.3 →
  packaging_fee = 0.25 →
  weight = 2.5 →
  (weight * (cost_per_kg + packaging_fee) = 38.875) :=
by
  intros h1 h2 h3
  sorry

end total_cost_of_apples_l1804_180482


namespace line_intercepts_l1804_180442

-- Definitions
def point_on_axis (a b : ℝ) : Prop := a = b
def passes_through_point (a b : ℝ) (x y : ℝ) : Prop := a * x + b * y = 1

theorem line_intercepts (a b x y : ℝ) (hx : x = -1) (hy : y = 2) (intercept_property : point_on_axis a b) (point_property : passes_through_point a b x y) :
  (2 * x + y = 0) ∨ (x + y - 1 = 0) :=
sorry

end line_intercepts_l1804_180442


namespace rectangle_area_proof_l1804_180410

def rectangle_area (L W : ℝ) : ℝ := L * W

theorem rectangle_area_proof (L W : ℝ) (h1 : L + W = 23) (h2 : L^2 + W^2 = 289) : rectangle_area L W = 120 := by
  sorry

end rectangle_area_proof_l1804_180410


namespace integer_solution_unique_l1804_180429

variable (x y : ℤ)

def nested_sqrt_1964_times (x : ℤ) : ℤ := 
  sorry -- (This should define the function for nested sqrt 1964 times, but we'll use sorry to skip the proof)

theorem integer_solution_unique : 
  nested_sqrt_1964_times x = y → x = 0 ∧ y = 0 :=
by
  intros h
  sorry -- Proof of the theorem goes here

end integer_solution_unique_l1804_180429


namespace angle_measure_l1804_180405

theorem angle_measure (x : ℝ) 
  (h1 : 90 - x = (2 / 5) * (180 - x)) :
  x = 30 :=
by
  sorry

end angle_measure_l1804_180405


namespace find_f_neg_eight_l1804_180406

-- Conditions based on the given problem
variable (f : ℤ → ℤ)
axiom func_property : ∀ x y : ℤ, f (x + y) = f x + f y + x * y + 1
axiom f1_is_one : f 1 = 1

-- Main theorem
theorem find_f_neg_eight : f (-8) = 19 := by
  sorry

end find_f_neg_eight_l1804_180406


namespace circle_center_radius_l1804_180445

theorem circle_center_radius :
  ∃ (h : ℝ × ℝ) (r : ℝ),
    (h = (1, -3)) ∧ (r = 2) ∧ ∀ x y : ℝ, 
    (x - h.1)^2 + (y - h.2)^2 = 4 → x^2 + y^2 - 2*x + 6*y + 6 = 0 :=
sorry

end circle_center_radius_l1804_180445


namespace cos_300_eq_half_l1804_180489

theorem cos_300_eq_half : Real.cos (300 * Real.pi / 180) = 1 / 2 :=
by
  sorry

end cos_300_eq_half_l1804_180489


namespace find_n_l1804_180407

theorem find_n (n : ℕ) : 
  Nat.lcm n 12 = 48 ∧ Nat.gcd n 12 = 8 → n = 32 := 
by 
  sorry

end find_n_l1804_180407


namespace minimum_value_exists_l1804_180479

noncomputable def min_value (a b c : ℝ) : ℝ :=
  a / (3 * b^2) + b / (4 * c^3) + c / (5 * a^4)

theorem minimum_value_exists :
  ∀ (a b c : ℝ), a > 0 → b > 0 → c > 0 → abc = 1 → min_value a b c ≥ 1 :=
by
  sorry

end minimum_value_exists_l1804_180479


namespace sum_f_values_l1804_180496

theorem sum_f_values (a b c d e f g : ℕ) 
  (h1: 100 * a * b = 100 * d)
  (h2: c * d * e = 100 * d)
  (h3: b * d * f = 100 * d)
  (h4: b * f = 100)
  (h5: 100 * d = 100) : 
  100 + 50 + 25 + 20 + 10 + 5 + 4 + 2 + 1 = 217 :=
by
  sorry

end sum_f_values_l1804_180496


namespace needed_people_l1804_180413

theorem needed_people (n t t' k m : ℕ) (h1 : n = 6) (h2 : t = 8) (h3 : t' = 3) 
    (h4 : k = n * t) (h5 : k = m * t') : m - n = 10 :=
by
  sorry

end needed_people_l1804_180413


namespace katya_sum_greater_than_masha_l1804_180411

theorem katya_sum_greater_than_masha (a b c d : ℝ) (h : a + b + c + d = 2021) :
  ((a+1)*(b+1) + (b+1)*(c+1) + (c+1)*(d+1) + (d+1)*(a+1)) - (a*b + b*c + c*d + d*a) = 4046 := by
  sorry

end katya_sum_greater_than_masha_l1804_180411


namespace binom_12_6_l1804_180425

theorem binom_12_6 : Nat.choose 12 6 = 924 :=
by
  sorry

end binom_12_6_l1804_180425


namespace triangle_side_length_l1804_180486

theorem triangle_side_length {x : ℝ} (h1 : 6 + x + x = 20) : x = 7 :=
by 
  sorry

end triangle_side_length_l1804_180486


namespace time_to_cross_platform_l1804_180485

-- Definitions for the length of the train, the length of the platform, and the speed of the train
def length_train : ℕ := 750
def length_platform : ℕ := 750
def speed_train_kmh : ℕ := 90

-- Conversion constants
def meters_per_kilometer : ℕ := 1000
def seconds_per_hour : ℕ := 3600

-- Convert speed from km/hr to m/s
def speed_train_ms : ℚ := speed_train_kmh * meters_per_kilometer / seconds_per_hour

-- Total distance the train covers to cross the platform
def total_distance : ℕ := length_train + length_platform

-- Proof problem: To prove that the time taken to cross the platform is 60 seconds
theorem time_to_cross_platform : total_distance / speed_train_ms = 60 := by
  sorry

end time_to_cross_platform_l1804_180485


namespace arkansas_tshirts_sold_l1804_180447

theorem arkansas_tshirts_sold (A T : ℕ) (h1 : A + T = 163) (h2 : 98 * A = 8722) : A = 89 := by
  -- We state the problem and add 'sorry' to skip the actual proof
  sorry

end arkansas_tshirts_sold_l1804_180447


namespace mary_change_l1804_180416

def cost_of_berries : ℝ := 7.19
def cost_of_peaches : ℝ := 6.83
def amount_paid : ℝ := 20.00

theorem mary_change : amount_paid - (cost_of_berries + cost_of_peaches) = 5.98 := by
  sorry

end mary_change_l1804_180416


namespace chucks_team_final_score_l1804_180487

variable (RedTeamScore : ℕ) (scoreDifference : ℕ)

-- Given conditions
def red_team_score := RedTeamScore = 76
def score_difference := scoreDifference = 19

-- Question: What was the final score of Chuck's team?
def chucks_team_score (RedTeamScore scoreDifference : ℕ) : ℕ := 
  RedTeamScore + scoreDifference

-- Proof statement
theorem chucks_team_final_score : red_team_score 76 ∧ score_difference 19 → chucks_team_score 76 19 = 95 :=
by
  sorry

end chucks_team_final_score_l1804_180487


namespace ladder_base_distance_l1804_180472

theorem ladder_base_distance
  (ladder_length : ℝ)
  (wall_height : ℝ)
  (base_distance : ℝ)
  (h1 : ladder_length = 13)
  (h2 : wall_height = 12)
  (h3 : ladder_length^2 = wall_height^2 + base_distance^2) :
  base_distance = 5 :=
sorry

end ladder_base_distance_l1804_180472


namespace cone_volume_l1804_180404

theorem cone_volume (slant_height : ℝ) (central_angle_deg : ℝ) (volume : ℝ) :
  slant_height = 1 ∧ central_angle_deg = 120 ∧ volume = (2 * Real.sqrt 2 / 81) * Real.pi →
  ∃ r h, h = Real.sqrt (slant_height^2 - r^2) ∧
    r = (1/3) ∧
    h = (2 * Real.sqrt 2 / 3) ∧
    volume = (1/3) * Real.pi * r^2 * h := 
by
  sorry

end cone_volume_l1804_180404


namespace smallest_boxes_l1804_180415

theorem smallest_boxes (n : Nat) (h₁ : n % 5 = 0) (h₂ : n % 24 = 0) : n = 120 := 
  sorry

end smallest_boxes_l1804_180415


namespace sum_of_areas_B_D_l1804_180484

theorem sum_of_areas_B_D (area_large_square : ℝ) (area_small_square : ℝ) (B D : ℝ) 
  (h1 : area_large_square = 9) 
  (h2 : area_small_square = 1)
  (h3 : B + D = 4) : 
  B + D = 4 := 
by
  sorry

end sum_of_areas_B_D_l1804_180484


namespace inequality_div_two_l1804_180459

theorem inequality_div_two (x y : ℝ) (h : x > y) : x / 2 > y / 2 := sorry

end inequality_div_two_l1804_180459


namespace count_CONES_paths_l1804_180449

def diagram : List (List Char) :=
  [[' ', ' ', 'C', ' ', ' ', ' '],
   [' ', 'C', 'O', 'C', ' ', ' '],
   ['C', 'O', 'N', 'O', 'C', ' '],
   [' ', 'N', 'E', 'N', ' ', ' '],
   [' ', ' ', 'S', ' ', ' ', ' ']]

def is_adjacent (pos1 pos2 : (Nat × Nat)) : Bool :=
  (pos1.1 = pos2.1 ∨ pos1.1 + 1 = pos2.1 ∨ pos1.1 = pos2.1 + 1) ∧
  (pos1.2 = pos2.2 ∨ pos1.2 + 1 = pos2.2 ∨ pos1.2 = pos2.2 + 1)

def valid_paths (diagram : List (List Char)) : Nat :=
  -- Implementation of counting paths that spell "CONES" skipped
  sorry

theorem count_CONES_paths (d : List (List Char)) 
  (h : d = [[' ', ' ', 'C', ' ', ' ', ' '],
            [' ', 'C', 'O', 'C', ' ', ' '],
            ['C', 'O', 'N', 'O', 'C', ' '],
            [' ', 'N', 'E', 'N', ' ', ' '],
            [' ', ' ', 'S', ' ', ' ', ' ']]): valid_paths d = 6 := 
by
  sorry

end count_CONES_paths_l1804_180449


namespace value_of_b_is_one_l1804_180464

open Complex

theorem value_of_b_is_one (a b : ℝ) (h : (1 + I) / (1 - I) = a + b * I) : b = 1 := 
by
  sorry

end value_of_b_is_one_l1804_180464


namespace expand_product_l1804_180458

theorem expand_product (x : ℝ) : (x + 4) * (x - 9) = x^2 - 5*x - 36 :=
by
  sorry

end expand_product_l1804_180458


namespace expression_divisible_by_11_l1804_180433

theorem expression_divisible_by_11 (n : ℕ) : (3 ^ (2 * n + 2) + 2 ^ (6 * n + 1)) % 11 = 0 :=
sorry

end expression_divisible_by_11_l1804_180433


namespace exists_root_in_interval_l1804_180467

theorem exists_root_in_interval
    (a b c x₁ x₂ : ℝ)
    (h₁ : a * x₁^2 + b * x₁ + c = 0)
    (h₂ : -a * x₂^2 + b * x₂ + c = 0) :
    ∃ x₃ : ℝ, (a / 2) * x₃^2 + b * x₃ + c = 0 ∧ ((x₁ ≤ x₃ ∧ x₃ ≤ x₂) ∨ (x₁ ≥ x₃ ∧ x₃ ≥ x₂)) :=
sorry

end exists_root_in_interval_l1804_180467


namespace right_triangle_sides_l1804_180483

theorem right_triangle_sides (m n : ℝ) (x : ℝ) (a b c : ℝ)
  (h1 : 2 * x < m + n) 
  (h2 : a = Real.sqrt (2 * m * n) - m)
  (h3 : b = Real.sqrt (2 * m * n) - n)
  (h4 : c = m + n - Real.sqrt (2 * m * n))
  (h5 : a^2 + b^2 = c^2)
  (h6 : 4 * x^2 = (m - 2 * x)^2 + (n - 2 * x)^2) :
  a = Real.sqrt (2 * m * n) - m ∧ b = Real.sqrt (2 * m * n) - n ∧ c = m + n - Real.sqrt (2 * m * n) :=
by
  sorry

end right_triangle_sides_l1804_180483


namespace smallest_n_for_terminating_decimal_l1804_180441

theorem smallest_n_for_terminating_decimal :
  ∃ (n : ℕ), (∀ m : ℕ, (n = m → m > 0 → ∃ (a b : ℕ), n + 103 = 2^a * 5^b)) 
    ∧ n = 22 :=
sorry

end smallest_n_for_terminating_decimal_l1804_180441


namespace range_of_m_l1804_180480

open Set Real

def A : Set ℝ := { x | x^2 - 5 * x + 6 = 0 }
def B (m : ℝ) : Set ℝ := { x | x^2 - (m + 3) * x + m^2 = 0 }

theorem range_of_m (m : ℝ) :
  (A ∪ (univ \ B m)) = univ ↔ m ∈ Iio (-1) ∪ Ici 3 :=
sorry

end range_of_m_l1804_180480


namespace cos_diff_expression_eq_half_l1804_180453

theorem cos_diff_expression_eq_half :
  (Real.cos (Real.pi * 24 / 180) * Real.cos (Real.pi * 36 / 180) -
   Real.cos (Real.pi * 66 / 180) * Real.cos (Real.pi * 54 / 180)) = 1 / 2 := by
sorry

end cos_diff_expression_eq_half_l1804_180453


namespace t_lt_s_l1804_180457

noncomputable def t : ℝ := Real.sqrt 11 - 3
noncomputable def s : ℝ := Real.sqrt 7 - Real.sqrt 5

theorem t_lt_s : t < s :=
by
  sorry

end t_lt_s_l1804_180457


namespace volume_of_prism_l1804_180490

theorem volume_of_prism (l w h : ℝ) (h1 : l * w = 15) (h2 : w * h = 10) (h3 : l * h = 6) :
  l * w * h = 30 :=
by
  sorry

end volume_of_prism_l1804_180490


namespace number_of_people_l1804_180437

theorem number_of_people (x : ℕ) : 
  (x % 10 = 1) ∧
  (x % 9 = 1) ∧
  (x % 8 = 1) ∧
  (x % 7 = 1) ∧
  (x % 6 = 1) ∧
  (x % 5 = 1) ∧
  (x % 4 = 1) ∧
  (x % 3 = 1) ∧
  (x % 2 = 1) ∧
  (x < 5000) →
  x = 2521 :=
sorry

end number_of_people_l1804_180437


namespace infinitely_many_m_l1804_180461

theorem infinitely_many_m (r : ℕ) (n : ℕ) (h_r : r > 1) (h_n : n > 0) : 
  ∃ m, m = 4 * r ^ 4 ∧ ¬Prime (n^4 + m) :=
by
  sorry

end infinitely_many_m_l1804_180461


namespace pencils_problem_l1804_180439

theorem pencils_problem (x : ℕ) :
  2 * x + 6 * 3 + 2 * 1 = 24 → x = 2 :=
by
  sorry

end pencils_problem_l1804_180439


namespace measure_angle_BAC_l1804_180417

-- Define the elements in the problem
def triangle (A B C : Type) := (A ≠ B) ∧ (B ≠ C) ∧ (C ≠ A)

-- Define the lengths and angles
variables {A B C X Y : Type}

-- Define the conditions given in the problem
def conditions (AX XY YB BC : ℝ) (angleABC : ℝ) : Prop :=
  AX = XY ∧ XY = YB ∧ YB = BC ∧ angleABC = 100

-- The Lean 4 statement (proof outline is not required)
theorem measure_angle_BAC {A B C X Y : Type} (hT : triangle A B C)
  (AX XY YB BC : ℝ) (angleABC : ℝ) (hC : conditions AX XY YB BC angleABC) :
  ∃ (t : ℝ), t = 25 :=
sorry
 
end measure_angle_BAC_l1804_180417


namespace range_of_b_l1804_180460

noncomputable def f (a x : ℝ) : ℝ := 
  Real.log x - (1 / 2) * a * x^2 - 2 * x

theorem range_of_b (a : ℝ) (b : ℝ) :
  (∀ x : ℝ, 0 < x ∧ x ≤ 1 → a ∈ Set.Ico (-1 : ℝ) (0 : ℝ) → f a x < b) ↔ b > -3 / 2 :=
by
  sorry

end range_of_b_l1804_180460


namespace isabella_original_hair_length_l1804_180469

-- Define conditions from the problem
def isabella_current_hair_length : ℕ := 9
def hair_cut_length : ℕ := 9

-- The proof problem to show original hair length equals 18 inches
theorem isabella_original_hair_length 
  (hc : isabella_current_hair_length = 9)
  (ht : hair_cut_length = 9) : 
  isabella_current_hhair_length + hair_cut_length = 18 := 
sorry

end isabella_original_hair_length_l1804_180469


namespace opposite_of_neg_2023_l1804_180426

theorem opposite_of_neg_2023 : ∃ x : ℤ, -2023 + x = 0 ∧ x = 2023 := 
by 
  use 2023
  constructor
  · exact rfl
  · exact rfl

end opposite_of_neg_2023_l1804_180426


namespace total_amount_earned_l1804_180431

-- Conditions
def avg_price_pair_rackets : ℝ := 9.8
def num_pairs_sold : ℕ := 60

-- Proof statement
theorem total_amount_earned :
  avg_price_pair_rackets * num_pairs_sold = 588 := by
    sorry

end total_amount_earned_l1804_180431


namespace train_speed_second_part_l1804_180400

-- Define conditions
def distance_first_part (x : ℕ) := x
def speed_first_part := 40
def distance_second_part (x : ℕ) := 2 * x
def total_distance (x : ℕ) := 5 * x
def average_speed := 40

-- Define the problem
theorem train_speed_second_part (x : ℕ) (v : ℕ) (h1 : total_distance x = 5 * x)
  (h2 : total_distance x / average_speed = distance_first_part x / speed_first_part + distance_second_part x / v) :
  v = 20 :=
  sorry

end train_speed_second_part_l1804_180400


namespace sqrt_pos_condition_l1804_180477

theorem sqrt_pos_condition (x : ℝ) : (1 - x) ≥ 0 ↔ x ≤ 1 := 
by 
  sorry

end sqrt_pos_condition_l1804_180477


namespace original_price_correct_l1804_180438

noncomputable def original_price (selling_price : ℝ) (gain_percent : ℝ) : ℝ :=
  selling_price / (1 + gain_percent / 100)

theorem original_price_correct :
  original_price 35 75 = 20 :=
by
  sorry

end original_price_correct_l1804_180438


namespace solve_for_z_l1804_180440

theorem solve_for_z (x y : ℝ) (z : ℝ) (h : 2 / x - 1 / y = 3 / z) : 
  z = (2 * y - x) / 3 :=
by
  sorry

end solve_for_z_l1804_180440


namespace sides_of_polygon_l1804_180432

-- Define the conditions
def polygon_sum_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

-- State the mathematically equivalent proof problem
theorem sides_of_polygon (n : ℕ) (h : polygon_sum_interior_angles n = 1260) : n = 9 := by
  sorry

end sides_of_polygon_l1804_180432


namespace find_m_l1804_180420

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (m^2 - m - 5) * x^(m + 1)

theorem find_m (m : ℝ) :
  (∀ x > 0, f m x < 0) → m = -2 := by
  sorry

end find_m_l1804_180420


namespace number_of_jerseys_sold_l1804_180414

-- Definitions based on conditions
def revenue_per_jersey : ℕ := 115
def revenue_per_tshirt : ℕ := 25
def tshirts_sold : ℕ := 113
def jersey_cost_difference : ℕ := 90

-- Main condition: Prove the number of jerseys sold is 113
theorem number_of_jerseys_sold : ∀ (J : ℕ), 
  (revenue_per_jersey = revenue_per_tshirt + jersey_cost_difference) →
  (J * revenue_per_jersey = tshirts_sold * revenue_per_tshirt) →
  J = 113 :=
by
  intros J h1 h2
  sorry

end number_of_jerseys_sold_l1804_180414


namespace total_shaded_area_l1804_180491

theorem total_shaded_area (S T U : ℝ) 
  (h1 : 12 / S = 4)
  (h2 : S / T = 2)
  (h3 : T / U = 2) :
  1 * (S * S) + 4 * (T * T) + 8 * (U * U) = 22.5 := by
sorry

end total_shaded_area_l1804_180491


namespace adjacent_books_probability_l1804_180452

def chinese_books : ℕ := 2
def math_books : ℕ := 2
def physics_books : ℕ := 1
def total_books : ℕ := chinese_books + math_books + physics_books

theorem adjacent_books_probability :
  (total_books = 5) →
  (chinese_books = 2) →
  (math_books = 2) →
  (physics_books = 1) →
  (∃ p : ℚ, p = 1 / 5) :=
by
  intros h1 h2 h3 h4
  -- Proof omitted.
  exact ⟨1 / 5, rfl⟩

end adjacent_books_probability_l1804_180452


namespace find_r_x_l1804_180463

open Nat

theorem find_r_x (r n : ℕ) (x : ℕ) (h_r_le_70 : r ≤ 70) (repr_x : x = (10 * r + 6) * (r ^ (2 * n) - 1) / (r ^ 2 - 1))
  (repr_x2 : x^2 = (r ^ (4 * n) - 1) / (r - 1)) :
  (r = 7 ∧ x = 26) :=
by
  sorry

end find_r_x_l1804_180463


namespace find_circle_eqn_range_of_slope_l1804_180473

noncomputable def circle_eqn_through_points (M N : ℝ × ℝ) (line : ℝ → ℝ → Prop) :=
  ∃ (C : ℝ × ℝ) (r : ℝ),
    C ∈ {P : ℝ × ℝ | line P.1 P.2} ∧
    dist C M = dist C N ∧
    (∀ (P : ℝ × ℝ), dist P C = r ↔ (P = M ∨ P = N))

noncomputable def circle_standard_eqn (C : ℝ × ℝ) (r : ℝ) :=
  ∀ (P : ℝ × ℝ), dist P C = r ↔ (P.1 - C.1)^2 + P.2^2 = r^2

theorem find_circle_eqn (M N : ℝ × ℝ) (line : ℝ → ℝ → Prop)
  (h : circle_eqn_through_points M N line) :
  ∃ r : ℝ, circle_standard_eqn (1, 0) r ∧ r = 5 := 
  sorry

theorem range_of_slope (k : ℝ) :
  0 < k → 8 * k^2 - 15 * k > 0 → k > (15 / 8) :=
  sorry

end find_circle_eqn_range_of_slope_l1804_180473


namespace remainder_problem_l1804_180419

theorem remainder_problem (f y z : ℤ) (k m n : ℤ) 
  (h1 : f % 5 = 3) 
  (h2 : y % 5 = 4)
  (h3 : z % 7 = 6)
  (h4 : (f + y) % 15 = 7)
  : (f + y + z) % 35 = 3 ∧ (f + y + z) % 105 = 3 :=
by
  sorry

end remainder_problem_l1804_180419


namespace sequence_ab_sum_l1804_180493

theorem sequence_ab_sum (s a b : ℝ) (h1 : 16 * s = 4) (h2 : 1024 * s = a) (h3 : a * s = b) : a + b = 320 := by
  sorry

end sequence_ab_sum_l1804_180493


namespace intersection_of_sets_l1804_180418

def A : Set ℝ := { x | x < 2 }
def B : Set ℝ := { y | ∃ x : ℝ, y = 2^x - 1 }
def C : Set ℝ := { m | -1 < m ∧ m < 2 }

theorem intersection_of_sets : A ∩ B = C := 
by sorry

end intersection_of_sets_l1804_180418


namespace remainder_of_product_and_square_is_zero_mod_7_remainder_of_9_pow_100_mod_8_l1804_180488

-- Part (a): Remainder of (1989 * 1990 * 1991 + 1992^2) when divided by 7 is 0.
theorem remainder_of_product_and_square_is_zero_mod_7 :
  (1989 * 1990 * 1991 + 1992^2) % 7 = 0 :=
sorry

-- Part (b): Remainder of 9^100 when divided by 8 is 1.
theorem remainder_of_9_pow_100_mod_8 :
  9^100 % 8 = 1 :=
sorry

end remainder_of_product_and_square_is_zero_mod_7_remainder_of_9_pow_100_mod_8_l1804_180488


namespace line_intersects_midpoint_l1804_180427

theorem line_intersects_midpoint (c : ℤ) : 
  (∃x y : ℤ, 2 * x - y = c ∧ x = (1 + 5) / 2 ∧ y = (3 + 11) / 2) → c = -1 := by
  sorry

end line_intersects_midpoint_l1804_180427


namespace expand_product_l1804_180476

-- Define x as a variable within the real numbers
variable (x : ℝ)

-- Statement of the theorem
theorem expand_product : (x + 3) * (x - 4) = x^2 - x - 12 := 
by 
  sorry

end expand_product_l1804_180476


namespace sqrt_expr_evaluation_l1804_180451

theorem sqrt_expr_evaluation :
  (Real.sqrt (5 + 4 * Real.sqrt 3) - Real.sqrt (5 - 4 * Real.sqrt 3)) = 2 * Real.sqrt 2 :=
  sorry

end sqrt_expr_evaluation_l1804_180451


namespace yen_per_cad_l1804_180468

theorem yen_per_cad (yen cad : ℝ) (h : yen / cad = 5000 / 60) : yen = 83 := by
  sorry

end yen_per_cad_l1804_180468


namespace lifespan_of_bat_l1804_180412

variable (B H F T : ℝ)

theorem lifespan_of_bat (h₁ : H = B - 6)
                        (h₂ : F = 4 * H)
                        (h₃ : T = 2 * B)
                        (h₄ : B + H + F + T = 62) :
  B = 11.5 :=
by
  sorry

end lifespan_of_bat_l1804_180412


namespace percentage_weight_loss_measured_l1804_180497

variable (W : ℝ)

def weight_after_loss (W : ℝ) := 0.85 * W
def weight_with_clothes (W : ℝ) := weight_after_loss W * 1.02

theorem percentage_weight_loss_measured (W : ℝ) :
  ((W - weight_with_clothes W) / W) * 100 = 13.3 := by
  sorry

end percentage_weight_loss_measured_l1804_180497


namespace circle_passing_through_points_l1804_180494

open Real

theorem circle_passing_through_points :
  ∃ D E F : ℝ, (∀ x y : ℝ, (x = 0 ∧ y = 0) ∨ (x = 4 ∧ y = 0) ∨ (x = -1 ∧ y = 1) →
  x^2 + y^2 + D * x + E * y + F = 0) ∧
  x^2 + y^2 - 4 * x - 6 * y = 0 :=
sorry

end circle_passing_through_points_l1804_180494


namespace E_union_F_eq_univ_l1804_180456

-- Define the given conditions
def E : Set ℝ := { x | x^2 - 5 * x - 6 > 0 }
def F (a : ℝ) : Set ℝ := { x | x - 5 < a }
def I : Set ℝ := Set.univ
axiom a_gt_6 : ∃ a : ℝ, a > 6 ∧ 11 ∈ F a

-- State the theorem
theorem E_union_F_eq_univ (a : ℝ) (h₁ : a > 6) (h₂ : 11 ∈ F a) : E ∪ F a = I := by
  sorry

end E_union_F_eq_univ_l1804_180456


namespace katrina_cookies_sale_l1804_180403

/-- 
Katrina has 120 cookies in the beginning.
She sells 36 cookies in the morning.
She sells 16 cookies in the afternoon.
She has 11 cookies left to take home at the end of the day.
Prove that she sold 57 cookies during the lunch rush.
-/
theorem katrina_cookies_sale :
  let total_cookies := 120
  let morning_sales := 36
  let afternoon_sales := 16
  let cookies_left := 11
  let cookies_sold_lunch_rush := total_cookies - morning_sales - afternoon_sales - cookies_left
  cookies_sold_lunch_rush = 57 :=
by
  sorry

end katrina_cookies_sale_l1804_180403
