import Mathlib

namespace greatest_integer_a_l78_7880

theorem greatest_integer_a (a : ℤ) : a * a < 44 → a ≤ 6 :=
by
  intros h
  sorry

end greatest_integer_a_l78_7880


namespace total_strings_needed_l78_7828

def basses := 3
def strings_per_bass := 4
def guitars := 2 * basses
def strings_per_guitar := 6
def eight_string_guitars := guitars - 3
def strings_per_eight_string_guitar := 8

theorem total_strings_needed :
  (basses * strings_per_bass) + (guitars * strings_per_guitar) + (eight_string_guitars * strings_per_eight_string_guitar) = 72 := by
  sorry

end total_strings_needed_l78_7828


namespace f_monotonic_intervals_f_extreme_values_l78_7810

def f (x : ℝ) : ℝ := x^3 - 12 * x

-- Monotonicity intervals
theorem f_monotonic_intervals (x : ℝ) : 
  (x < -2 → deriv f x > 0) ∧ 
  (-2 < x ∧ x < 2 → deriv f x < 0) ∧ 
  (2 < x → deriv f x > 0) := 
sorry

-- Extreme values
theorem f_extreme_values :
  f (-2) = 16 ∧ f (2) = -16 :=
sorry

end f_monotonic_intervals_f_extreme_values_l78_7810


namespace problem_part1_problem_part2_l78_7801

variable {θ m : ℝ}
variable {h₀ : θ ∈ Ioo 0 (Real.pi / 2)}
variable {h₁ : Real.sin θ + Real.cos θ = (Real.sqrt 3 + 1) / 2}
variable {h₂ : Real.sin θ * Real.cos θ = m / 2}

theorem problem_part1 :
  (Real.sin θ / (1 - 1 / Real.tan θ) + Real.cos θ / (1 - Real.tan θ)) = (Real.sqrt 3 + 1) / 2 :=
sorry

theorem problem_part2 :
  m = Real.sqrt 3 / 2 ∧ (θ = Real.pi / 6 ∨ θ = Real.pi / 3) :=
sorry

end problem_part1_problem_part2_l78_7801


namespace determine_b_l78_7826

def imaginary_unit : Type := {i : ℂ // i^2 = -1}

theorem determine_b (i : imaginary_unit) (b : ℝ) : 
  (2 - i.val) * 4 * i.val = 4 - b * i.val → b = -8 :=
by
  sorry

end determine_b_l78_7826


namespace initial_investment_l78_7841

variable (P1 P2 π1 π2 : ℝ)

-- Given conditions
axiom h1 : π1 = 100
axiom h2 : π2 = 120

-- Revenue relation after the first transaction
axiom h3 : P2 = P1 + π1

-- Consistent profit relationship across transactions
axiom h4 : π2 = 0.2 * P2

-- To be proved
theorem initial_investment (P1 : ℝ) (h1 : π1 = 100) (h2 : π2 = 120) (h3 : P2 = P1 + π1) (h4 : π2 = 0.2 * P2) :
  P1 = 500 :=
sorry

end initial_investment_l78_7841


namespace complement_intersection_l78_7811

theorem complement_intersection (A B U : Set ℕ) (hA : A = {4, 5, 7}) (hB : B = {3, 4, 7, 8}) (hU : U = A ∪ B) :
  U \ (A ∩ B) = {3, 5, 8} :=
by
  sorry

end complement_intersection_l78_7811


namespace scheduling_arrangements_correct_l78_7869

-- Define the set of employees
inductive Employee
| A | B | C | D | E | F deriving DecidableEq

open Employee

-- Define the days of the festival
inductive Day
| May31 | June1 | June2 deriving DecidableEq

open Day

def canWork (e : Employee) (d : Day) : Prop :=
match e, d with
| A, May31 => False
| B, June2 => False
| _, _ => True

def schedulingArrangements : ℕ :=
  -- Calculations go here, placeholder for now
  sorry

theorem scheduling_arrangements_correct : schedulingArrangements = 42 := 
  sorry

end scheduling_arrangements_correct_l78_7869


namespace neg_prop_of_exists_x_gt_0_xsq_sub_x_leq_0_l78_7836

theorem neg_prop_of_exists_x_gt_0_xsq_sub_x_leq_0 :
  ¬ (∃ x : ℝ, x > 0 ∧ x^2 - x ≤ 0) ↔ ∀ x : ℝ, x ≤ 0 → x^2 - x > 0 :=
by
    sorry

end neg_prop_of_exists_x_gt_0_xsq_sub_x_leq_0_l78_7836


namespace cos_pi_minus_alpha_correct_l78_7807

noncomputable def cos_pi_minus_alpha (α : ℝ) (P : ℝ × ℝ) : ℝ :=
  let x := P.1
  let y := P.2
  let h := Real.sqrt (x^2 + y^2)
  let cos_alpha := x / h
  let cos_pi_minus_alpha := -cos_alpha
  cos_pi_minus_alpha

theorem cos_pi_minus_alpha_correct :
  cos_pi_minus_alpha α (-1, 2) = Real.sqrt 5 / 5 :=
by
  sorry

end cos_pi_minus_alpha_correct_l78_7807


namespace compare_abc_l78_7844

noncomputable def a : ℝ := (2 / 5) ^ (3 / 5)
noncomputable def b : ℝ := (2 / 5) ^ (2 / 5)
noncomputable def c : ℝ := (3 / 5) ^ (2 / 5)

theorem compare_abc : a < b ∧ b < c := sorry

end compare_abc_l78_7844


namespace rhombus_perimeter_is_80_l78_7879

-- Definitions of the conditions
def rhombus_diagonals_ratio : Prop := ∃ (d1 d2 : ℝ), d1 / d2 = 3 / 4 ∧ d1 + d2 = 56

-- The goal is to prove that given the conditions, the perimeter of the rhombus is 80
theorem rhombus_perimeter_is_80 (h : rhombus_diagonals_ratio) : ∃ (p : ℝ), p = 80 :=
by
  sorry  -- The actual proof steps would go here

end rhombus_perimeter_is_80_l78_7879


namespace smallest_whole_number_inequality_l78_7804

theorem smallest_whole_number_inequality (x : ℕ) (h : 3 * x + 4 > 11 - 2 * x) : x ≥ 2 :=
sorry

end smallest_whole_number_inequality_l78_7804


namespace initial_weight_of_alloy_is_16_l78_7881

variable (Z C : ℝ)
variable (h1 : Z / C = 5 / 3)
variable (h2 : (Z + 8) / C = 3)
variable (A : ℝ := Z + C)

theorem initial_weight_of_alloy_is_16 (h1 : Z / C = 5 / 3) (h2 : (Z + 8) / C = 3) : A = 16 := by
  sorry

end initial_weight_of_alloy_is_16_l78_7881


namespace merill_has_30_marbles_l78_7893

variable (M E : ℕ)

-- Conditions
def merill_twice_as_many_as_elliot : Prop := M = 2 * E
def together_five_fewer_than_selma : Prop := M + E = 45

theorem merill_has_30_marbles (h1 : merill_twice_as_many_as_elliot M E) (h2 : together_five_fewer_than_selma M E) : M = 30 := 
by
  sorry

end merill_has_30_marbles_l78_7893


namespace relationship_between_a_and_b_l78_7886

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x + 1

-- Given conditions
variables (b a : ℝ)
variables (hx : 0 < b) (ha : 0 < a)
variables (x : ℝ) (hb : |x - 1| < b) (hf : |f x - 4| < a)

-- The theorem statement
theorem relationship_between_a_and_b
  (hf_x : ∀ x : ℝ, |x - 1| < b -> |f x - 4| < a) :
  a - 3 * b ≥ 0 :=
sorry

end relationship_between_a_and_b_l78_7886


namespace family_children_count_l78_7812

theorem family_children_count (x y : ℕ) 
  (sister_condition : x = y - 1) 
  (brother_condition : y = 2 * (x - 1)) : 
  x + y = 7 := 
sorry

end family_children_count_l78_7812


namespace triangle_angles_21_equal_triangles_around_square_l78_7857

theorem triangle_angles_21_equal_triangles_around_square
    (theta alpha beta gamma : ℝ)
    (h1 : 4 * theta + 90 = 360)
    (h2 : alpha + beta + 90 = 180)
    (h3 : alpha + beta + gamma = 180)
    (h4 : gamma + 90 = 180)
    : theta = 67.5 ∧ alpha = 67.5 ∧ beta = 22.5 ∧ gamma = 90 :=
by
  sorry

end triangle_angles_21_equal_triangles_around_square_l78_7857


namespace diameter_other_endpoint_l78_7865

def center : ℝ × ℝ := (1, -2)
def endpoint1 : ℝ × ℝ := (4, 3)
def expected_endpoint2 : ℝ × ℝ := (7, -7)

theorem diameter_other_endpoint (c : ℝ × ℝ) (e1 e2 : ℝ × ℝ) (h₁ : c = center) (h₂ : e1 = endpoint1) : e2 = expected_endpoint2 :=
by
  sorry

end diameter_other_endpoint_l78_7865


namespace first_place_team_wins_l78_7883

-- Define the conditions in Lean 4
variable (joe_won : ℕ := 1) (joe_draw : ℕ := 3) (fp_draw : ℕ := 2) (joe_points : ℕ := 3 * joe_won + joe_draw)
variable (fp_points : ℕ := joe_points + 2)

 -- Define the proof problem
theorem first_place_team_wins : 3 * (fp_points - fp_draw) / 3 = 2 := by
  sorry

end first_place_team_wins_l78_7883


namespace inequality_1_plus_a_1_plus_b_1_plus_c_geq_1_minus_d_squared_l78_7890

theorem inequality_1_plus_a_1_plus_b_1_plus_c_geq_1_minus_d_squared 
  (a b c : ℝ)
  (h_sum : a + b + c = 0)
  (d : ℝ) 
  (h_d : d = max (abs a) (max (abs b) (abs c))) : 
  abs ((1 + a) * (1 + b) * (1 + c)) ≥ 1 - d^2 :=
by 
  sorry

end inequality_1_plus_a_1_plus_b_1_plus_c_geq_1_minus_d_squared_l78_7890


namespace lcm_1540_2310_l78_7878

theorem lcm_1540_2310 : Nat.lcm 1540 2310 = 4620 :=
by sorry

end lcm_1540_2310_l78_7878


namespace students_in_all_three_workshops_l78_7859

-- Define the students counts and other conditions
def num_students : ℕ := 25
def num_dance : ℕ := 12
def num_chess : ℕ := 15
def num_robotics : ℕ := 11
def num_at_least_two : ℕ := 12

-- Define the proof statement
theorem students_in_all_three_workshops : 
  ∃ c : ℕ, c = 1 ∧ 
    (∃ a b d : ℕ, 
      a + b + c + d = num_at_least_two ∧
      num_students ≥ num_dance + num_chess + num_robotics - a - b - d - 2 * c
    ) := 
by
  sorry

end students_in_all_three_workshops_l78_7859


namespace sum_of_altitudes_l78_7817

theorem sum_of_altitudes (a b c : ℝ) (h1 : a = 9) (h2 : b = 12) (h3 : c = 15) (h4 : a^2 + b^2 = c^2) : a + b = 21 :=
by
  -- Using the provided hypotheses, the proof would ensure a + b = 21.
  sorry

end sum_of_altitudes_l78_7817


namespace square_side_length_false_l78_7896

theorem square_side_length_false (perimeter : ℝ) (side_length : ℝ) (h1 : perimeter = 8) (h2 : side_length = 4) :
  ¬(4 * side_length = perimeter) :=
by
  sorry

end square_side_length_false_l78_7896


namespace lowest_score_on_one_of_last_two_tests_l78_7898

-- define conditions
variables (score1 score2 : ℕ) (total_score average desired_score : ℕ)

-- Shauna's scores on the first two tests are 82 and 75
def shauna_score1 := 82
def shauna_score2 := 75

-- Shauna wants to average 85 over 4 tests
def desired_average := 85
def number_of_tests := 4

-- total points needed for desired average
def total_points_needed := desired_average * number_of_tests

-- total points from first two tests
def total_first_two_tests := shauna_score1 + shauna_score2

-- total points needed on last two tests
def points_needed_last_two_tests := total_points_needed - total_first_two_tests

-- Prove the lowest score on one of the last two tests
theorem lowest_score_on_one_of_last_two_tests : 
  (∃ (score3 score4 : ℕ), score3 + score4 = points_needed_last_two_tests ∧ score3 ≤ 100 ∧ score4 ≤ 100 ∧ (score3 ≥ 83 ∨ score4 ≥ 83)) :=
sorry

end lowest_score_on_one_of_last_two_tests_l78_7898


namespace smallest_common_multiple_l78_7882

theorem smallest_common_multiple : Nat.lcm 18 35 = 630 := by
  sorry

end smallest_common_multiple_l78_7882


namespace geometric_progression_vertex_l78_7818

theorem geometric_progression_vertex (a b c d : ℝ) (q : ℝ)
  (h1 : b = 1)
  (h2 : c = 2)
  (h3 : q = c / b)
  (h4 : a = b / q)
  (h5 : d = c * q) :
  a + d = 9 / 2 :=
sorry

end geometric_progression_vertex_l78_7818


namespace solve_system_of_equations_l78_7848

theorem solve_system_of_equations : 
  ∃ (x y : ℚ), 4 * x - 3 * y = -2 ∧ 5 * x + 2 * y = 8 ∧ x = 20 / 23 ∧ y = 42 / 23 :=
by
  sorry

end solve_system_of_equations_l78_7848


namespace problem_1_problem_2_l78_7803

-- Definitions of the given probabilities
def prob_A : ℚ := 2/3
def prob_B : ℚ := 1/4
def prob_C : ℚ := 2/5

-- Independence implies that the probabilities of combined events are products of individual probabilities.
-- To avoid unnecessary complications, we assume independence holds true without proof.
axiom independence : ∀ A B C : Prop, (A ∧ B ∧ C) ↔ (A ∧ B) ∧ C

-- Problem statement for part (1)
theorem problem_1 : prob_A * prob_B * prob_C = 1/15 := by
  sorry

-- Helper definitions for probabilities of not visiting
def not_prob_A : ℚ := 1 - prob_A
def not_prob_B : ℚ := 1 - prob_B
def not_prob_C : ℚ := 1 - prob_C

-- Problem statement for part (2)
theorem problem_2 : (prob_A * not_prob_B * not_prob_C + not_prob_A * prob_B * not_prob_C + not_prob_A * not_prob_B * prob_C) = 9/20 := by
  sorry

end problem_1_problem_2_l78_7803


namespace total_bathing_suits_l78_7847

theorem total_bathing_suits 
  (a : ℕ) (b : ℕ) (c : ℕ) (d : ℕ) (e : ℕ)
  (ha : a = 8500) (hb : b = 12750) (hc : c = 5900) (hd : d = 7250) (he : e = 1100) :
  a + b + c + d + e = 35500 :=
by
  sorry

end total_bathing_suits_l78_7847


namespace winnie_keeps_lollipops_l78_7820

theorem winnie_keeps_lollipops :
  let cherry := 36
  let wintergreen := 125
  let grape := 8
  let shrimp_cocktail := 241
  let total_lollipops := cherry + wintergreen + grape + shrimp_cocktail
  let friends := 13
  total_lollipops % friends = 7 :=
by
  sorry

end winnie_keeps_lollipops_l78_7820


namespace consecutive_ints_square_l78_7860

theorem consecutive_ints_square (a b : ℤ) (h : b = a + 1) : 
  a^2 + b^2 + (a * b)^2 = (a * b + 1)^2 := 
by sorry

end consecutive_ints_square_l78_7860


namespace checkered_fabric_cost_l78_7876

variable (P : ℝ) (cost_per_yard : ℝ) (total_yards : ℕ)
variable (x : ℝ) (C : ℝ)

theorem checkered_fabric_cost :
  P = 45 ∧ cost_per_yard = 7.50 ∧ total_yards = 16 →
  C = cost_per_yard * (total_yards - x) →
  7.50 * (16 - x) = 45 →
  C = 75 :=
by
  intro h1 h2 h3
  sorry

end checkered_fabric_cost_l78_7876


namespace ratio_of_speeds_l78_7895

/-- Define the conditions -/
def distance_AB : ℝ := 540 -- Distance between city A and city B is 540 km
def time_Eddy : ℝ := 3     -- Eddy takes 3 hours to travel to city B
def distance_AC : ℝ := 300 -- Distance between city A and city C is 300 km
def time_Freddy : ℝ := 4   -- Freddy takes 4 hours to travel to city C

/-- Define the average speeds -/
noncomputable def avg_speed_Eddy : ℝ := distance_AB / time_Eddy
noncomputable def avg_speed_Freddy : ℝ := distance_AC / time_Freddy

/-- The statement to prove -/
theorem ratio_of_speeds : avg_speed_Eddy / avg_speed_Freddy = 12 / 5 :=
by sorry

end ratio_of_speeds_l78_7895


namespace trajectory_eq_ellipse_l78_7839

theorem trajectory_eq_ellipse :
  (∀ M : ℝ × ℝ, (∀ r : ℝ, (M.1 - 4)^2 + M.2^2 = r^2 ∧ (M.1 + 4)^2 + M.2^2 = (10 - r)^2) → false) →
  ∀ x y : ℝ, (x^2 / 25 + y^2 / 9 = 1) :=
by
  sorry

end trajectory_eq_ellipse_l78_7839


namespace gcd_of_ratios_l78_7864

noncomputable def gcd_of_two_ratios (A B : ℕ) : ℕ :=
  if h : A % B = 0 then B else gcd B (A % B)

theorem gcd_of_ratios (A B : ℕ) (k : ℕ) (h1 : Nat.lcm A B = 180) (h2 : A = 2 * k) (h3 : B = 3 * k) : gcd_of_two_ratios A B = 30 :=
  by
    sorry

end gcd_of_ratios_l78_7864


namespace round_trip_average_mileage_l78_7845

theorem round_trip_average_mileage 
  (d1 d2 : ℝ) (m1 m2 : ℝ)
  (h1 : d1 = 150) (h2 : d2 = 150)
  (h3 : m1 = 40) (h4 : m2 = 25) :
  (d1 + d2) / ((d1 / m1) + (d2 / m2)) = 30.77 :=
by
  sorry

end round_trip_average_mileage_l78_7845


namespace evaluate_expression_l78_7867

theorem evaluate_expression : 
  1 + 2 / (3 + 4 / (5 + 6 / 7)) = 233 / 151 := 
by 
  sorry

end evaluate_expression_l78_7867


namespace tenth_equation_sum_of_cubes_l78_7873

theorem tenth_equation_sum_of_cubes :
  (1^3 + 2^3 + 3^3 + 4^3 + 5^3 + 6^3 + 7^3 + 8^3 + 9^3 + 10^3) = 55^2 := 
by sorry

end tenth_equation_sum_of_cubes_l78_7873


namespace problem_statement_l78_7808

theorem problem_statement (m : ℝ) (h : m + 1/m = 10) : m^2 + 1/m^2 + 6 = 104 := by
  sorry

end problem_statement_l78_7808


namespace sin_2017pi_over_6_l78_7800

theorem sin_2017pi_over_6 : Real.sin (2017 * Real.pi / 6) = 1 / 2 := 
by 
  -- Proof to be filled in later
  sorry

end sin_2017pi_over_6_l78_7800


namespace area_of_rectangle_l78_7832

noncomputable def area_proof : ℝ :=
  let a := 294
  let b := 147
  let c := 3
  a + b * Real.sqrt c

theorem area_of_rectangle (ABCD : ℝ × ℝ) (E : ℝ) (F : ℝ) (BE : ℝ) (AB' : ℝ) : 
  BE = 21 ∧ BE = 2 * CF → AB' = 7 → 
  (ABCD.1 * ABCD.2 = 294 + 147 * Real.sqrt 3 ∧ (294 + 147 + 3 = 444)) :=
sorry

end area_of_rectangle_l78_7832


namespace trigonometric_identity_l78_7806

theorem trigonometric_identity (θ : ℝ) (hθ1 : θ ∈ Set.Ico 0 Real.pi) (hθ2 : Real.cos θ * (Real.sin θ + Real.cos θ) = 1) :
  θ = 0 ∨ θ = Real.pi / 4 :=
sorry

end trigonometric_identity_l78_7806


namespace length_of_arc_l78_7837

theorem length_of_arc (S : ℝ) (α : ℝ) (hS : S = 4) (hα : α = 2) : 
  ∃ l : ℝ, l = 4 :=
by
  sorry

end length_of_arc_l78_7837


namespace shaded_region_area_l78_7838

noncomputable def area_of_shaded_region (a b c d : ℝ) (area_rect : ℝ) : ℝ :=
  let dg : ℝ := (a * d) / (c + d)
  let area_triangle : ℝ := 0.5 * dg * b
  area_rect - area_triangle

theorem shaded_region_area :
  area_of_shaded_region 12 5 12 4 (4 * 5) = 85 / 8 :=
by
  simp [area_of_shaded_region]
  sorry

end shaded_region_area_l78_7838


namespace total_peaches_in_each_basket_l78_7815

-- Define the given conditions
def red_peaches : ℕ := 7
def green_peaches : ℕ := 3

-- State the theorem
theorem total_peaches_in_each_basket : red_peaches + green_peaches = 10 :=
by
  -- Proof goes here, which we skip for now
  sorry

end total_peaches_in_each_basket_l78_7815


namespace exponent_identity_l78_7871

theorem exponent_identity (m : ℕ) : 5 ^ m = 5 * (25 ^ 4) * (625 ^ 3) ↔ m = 21 := by
  sorry

end exponent_identity_l78_7871


namespace triangle_is_isosceles_l78_7853

variable (A B C a b c : ℝ)
variable (sin : ℝ → ℝ)

theorem triangle_is_isosceles (h1 : a * sin A - b * sin B = 0) :
  a = b :=
by
  sorry

end triangle_is_isosceles_l78_7853


namespace spider_paths_l78_7854

theorem spider_paths : (Nat.choose (7 + 3) 3) = 210 := 
by
  sorry

end spider_paths_l78_7854


namespace flyDistanceCeiling_l78_7872

variable (P : ℝ × ℝ × ℝ)
variable (x : ℝ)
variable (y : ℝ)
variable (z : ℝ)

-- Defining the conditions
def isAtRightAngles (P : ℝ × ℝ × ℝ) : Prop :=
  P = (0, 0, 0)

def distanceFromWall1 (x : ℝ) : Prop :=
  x = 2

def distanceFromWall2 (y : ℝ) : Prop :=
  y = 5

def distanceFromPointP (x y z : ℝ) : Prop :=
  7 = Real.sqrt (x^2 + y^2 + z^2)

-- Proving the distance from the ceiling
theorem flyDistanceCeiling (P : ℝ × ℝ × ℝ) (x y z : ℝ) :
  isAtRightAngles P →
  distanceFromWall1 x →
  distanceFromWall2 y →
  distanceFromPointP x y z →
  z = 2 * Real.sqrt 5 := 
sorry

end flyDistanceCeiling_l78_7872


namespace li_family_cinema_cost_l78_7827

theorem li_family_cinema_cost :
  let standard_ticket_price := 10
  let child_discount := 0.4
  let senior_discount := 0.3
  let handling_fee := 5
  let num_adults := 2
  let num_children := 1
  let num_seniors := 1
  let child_ticket_price := (1 - child_discount) * standard_ticket_price
  let senior_ticket_price := (1 - senior_discount) * standard_ticket_price
  let total_ticket_cost := num_adults * standard_ticket_price + num_children * child_ticket_price + num_seniors * senior_ticket_price
  let final_cost := total_ticket_cost + handling_fee
  final_cost = 38 :=
by
  sorry

end li_family_cinema_cost_l78_7827


namespace copper_to_zinc_ratio_l78_7843

theorem copper_to_zinc_ratio (total_weight_brass : ℝ) (weight_zinc : ℝ) (weight_copper : ℝ) 
  (h1 : total_weight_brass = 100) (h2 : weight_zinc = 70) (h3 : weight_copper = total_weight_brass - weight_zinc) : 
  weight_copper / weight_zinc = 3 / 7 :=
by
  sorry

end copper_to_zinc_ratio_l78_7843


namespace complex_expression_evaluation_l78_7868

theorem complex_expression_evaluation (z : ℂ) (h : z^2 + z + 1 = 0) :
  z^101 + z^102 + z^103 + z^104 + z^105 = -1 := 
sorry

end complex_expression_evaluation_l78_7868


namespace triangle_concurrency_l78_7899

-- Define Triangle Structure
structure Triangle (α : Type*) :=
(A B C : α)

-- Define Medians, Angle Bisectors, and Altitudes Concurrency Conditions
noncomputable def medians_concurrent {α : Type*} [MetricSpace α] (T : Triangle α) : Prop := sorry
noncomputable def angle_bisectors_concurrent {α : Type*} [MetricSpace α] (T : Triangle α) : Prop := sorry
noncomputable def altitudes_concurrent {α : Type*} [MetricSpace α] (T : Triangle α) : Prop := sorry

-- Main Theorem Statement
theorem triangle_concurrency {α : Type*} [MetricSpace α] (T : Triangle α) :
  medians_concurrent T ∧ angle_bisectors_concurrent T ∧ altitudes_concurrent T :=
by 
  -- Proof outline: Prove each concurrency condition
  sorry

end triangle_concurrency_l78_7899


namespace Tyler_CDs_after_giveaway_and_purchase_l78_7813

theorem Tyler_CDs_after_giveaway_and_purchase :
  (∃ cds_initial cds_giveaway_fraction cds_bought cds_final, 
     cds_initial = 21 ∧ 
     cds_giveaway_fraction = 1 / 3 ∧ 
     cds_bought = 8 ∧ 
     cds_final = cds_initial - (cds_initial * cds_giveaway_fraction) + cds_bought ∧
     cds_final = 22) := 
sorry

end Tyler_CDs_after_giveaway_and_purchase_l78_7813


namespace budget_left_equals_16_l78_7822

def initial_budget : ℤ := 200
def expense_shirt : ℤ := 30
def expense_pants : ℤ := 46
def expense_coat : ℤ := 38
def expense_socks : ℤ := 11
def expense_belt : ℤ := 18
def expense_shoes : ℤ := 41

def total_expenses : ℤ := 
  expense_shirt + expense_pants + expense_coat + expense_socks + expense_belt + expense_shoes

def budget_left : ℤ := initial_budget - total_expenses

theorem budget_left_equals_16 : 
  budget_left = 16 := by
  sorry

end budget_left_equals_16_l78_7822


namespace floor_problem_2020_l78_7894

-- Define the problem statement
theorem floor_problem_2020:
  2020 ^ 2021 - (Int.floor ((2020 ^ 2021 : ℝ) / 2021) * 2021) = 2020 :=
sorry

end floor_problem_2020_l78_7894


namespace solution_set_inequality_l78_7805

theorem solution_set_inequality (x : ℝ) : ((x - 1) * (x + 2) < 0) ↔ (-2 < x ∧ x < 1) := by
  sorry

end solution_set_inequality_l78_7805


namespace Exponent_Equality_l78_7821

theorem Exponent_Equality : 2^8 * 2^32 = 256^5 :=
by
  sorry

end Exponent_Equality_l78_7821


namespace leibo_orange_price_l78_7855

variable (x y m : ℝ)

theorem leibo_orange_price :
  (3 * x + 2 * y = 78) ∧ (2 * x + 3 * y = 72) ∧ (18 * m + 12 * (100 - m) ≤ 1440) → (x = 18) ∧ (y = 12) ∧ (m ≤ 40) :=
by
  intros h
  sorry

end leibo_orange_price_l78_7855


namespace right_triangle_area_l78_7824

theorem right_triangle_area (a b : ℕ) (h1 : a = 36) (h2 : b = 48) : (1 / 2 : ℚ) * (a * b) = 864 := 
by 
  sorry

end right_triangle_area_l78_7824


namespace power_of_11_in_expression_l78_7823

-- Define the mathematical context
def prime_factors_count (n : ℕ) (a b c : ℕ) : ℕ :=
  n + a + b

-- Given conditions
def count_factors_of_2 : ℕ := 22
def count_factors_of_7 : ℕ := 5
def total_prime_factors : ℕ := 29

-- Theorem stating that power of 11 in the expression is 2
theorem power_of_11_in_expression : 
  ∃ n : ℕ, prime_factors_count n count_factors_of_2 count_factors_of_7 = total_prime_factors ∧ n = 2 :=
by
  sorry

end power_of_11_in_expression_l78_7823


namespace discount_calculation_l78_7891

-- Definitions based on the given conditions
def cost_magazine : Float := 0.85
def cost_pencil : Float := 0.50
def amount_spent : Float := 1.00

-- Define the total cost before discount
def total_cost_before_discount : Float := cost_magazine + cost_pencil

-- Goal: Prove that the discount is $0.35
theorem discount_calculation : total_cost_before_discount - amount_spent = 0.35 := by
  -- Proof (to be filled in later)
  sorry

end discount_calculation_l78_7891


namespace last_digit_base5_89_l78_7816

theorem last_digit_base5_89 : 
  ∃ (b : ℕ), (89 : ℕ) = b * 5 + 4 :=
by
  -- The theorem above states that there exists an integer b, such that when we compute 89 in base 5, 
  -- its last digit is 4.
  sorry

end last_digit_base5_89_l78_7816


namespace chocolate_distribution_l78_7852

theorem chocolate_distribution :
  let total_chocolate := 60 / 7
  let piles := 5
  let eaten_piles := 1
  let friends := 2
  let one_pile := total_chocolate / piles
  let remaining_chocolate := total_chocolate - eaten_piles * one_pile
  let chocolate_per_friend := remaining_chocolate / friends
  chocolate_per_friend = 24 / 7 :=
by
  sorry

end chocolate_distribution_l78_7852


namespace total_books_correct_l78_7870

-- Define the number of books each person has
def booksKeith : Nat := 20
def booksJason : Nat := 21
def booksMegan : Nat := 15

-- Define the total number of books they have together
def totalBooks : Nat := booksKeith + booksJason + booksMegan

-- Prove that the total number of books is 56
theorem total_books_correct : totalBooks = 56 := by
  sorry

end total_books_correct_l78_7870


namespace range_of_a_l78_7831

theorem range_of_a (a : ℝ) : (∀ x : ℝ, ax^2 + ax + 1 > 0) ↔ (0 ≤ a ∧ a < 4) :=
by
  sorry

end range_of_a_l78_7831


namespace contrapositive_example_l78_7888

theorem contrapositive_example (x : ℝ) :
  (x < -1 ∨ x ≥ 1) → (x^2 ≥ 1) :=
sorry

end contrapositive_example_l78_7888


namespace product_12_3460_l78_7825

theorem product_12_3460 : 12 * 3460 = 41520 :=
by
  sorry

end product_12_3460_l78_7825


namespace cathy_remaining_money_l78_7849

noncomputable def remaining_money (initial : ℝ) (dad : ℝ) (book : ℝ) (cab_percentage : ℝ) (food_percentage : ℝ) : ℝ :=
  let money_mom := 2 * dad
  let total_money := initial + dad + money_mom
  let remaining_after_book := total_money - book
  let cab_cost := cab_percentage * remaining_after_book
  let food_budget := food_percentage * total_money
  let dinner_cost := 0.5 * food_budget
  remaining_after_book - cab_cost - dinner_cost

theorem cathy_remaining_money :
  remaining_money 12 25 15 0.03 0.4 = 52.44 :=
by
  sorry

end cathy_remaining_money_l78_7849


namespace radio_advertiser_savings_l78_7814

def total_store_price : ℚ := 299.99
def ad_payment : ℚ := 55.98
def payments_count : ℚ := 5
def shipping_handling : ℚ := 12.99

def total_ad_price : ℚ := payments_count * ad_payment + shipping_handling

def savings_in_dollars : ℚ := total_store_price - total_ad_price
def savings_in_cents : ℚ := savings_in_dollars * 100

theorem radio_advertiser_savings :
  savings_in_cents = 710 := by
  sorry

end radio_advertiser_savings_l78_7814


namespace intersecting_lines_a_value_l78_7819

theorem intersecting_lines_a_value :
  ∀ t a b : ℝ, (b = 12) ∧ (b = 2 * a + t) ∧ (t = 4) → a = 4 :=
by
  intros t a b h
  obtain ⟨hb1, hb2, ht⟩ := h
  sorry

end intersecting_lines_a_value_l78_7819


namespace molecular_weight_C7H6O2_l78_7877

noncomputable def molecular_weight_one_mole (w_9moles : ℕ) (m_9moles : ℕ) : ℕ :=
  m_9moles / w_9moles

theorem molecular_weight_C7H6O2 :
  molecular_weight_one_mole 9 1098 = 122 := by
  sorry

end molecular_weight_C7H6O2_l78_7877


namespace fraction_playing_in_field_l78_7834

def class_size : ℕ := 50
def students_painting : ℚ := 3/5
def students_left_in_classroom : ℕ := 10

theorem fraction_playing_in_field :
  (class_size - students_left_in_classroom - students_painting * class_size) / class_size = 1/5 :=
by
  sorry

end fraction_playing_in_field_l78_7834


namespace project_completion_time_l78_7802

theorem project_completion_time (rate_a rate_b rate_c : ℝ) (total_work : ℝ) (quit_time : ℝ) 
  (ha : rate_a = 1 / 20) 
  (hb : rate_b = 1 / 30) 
  (hc : rate_c = 1 / 40) 
  (htotal : total_work = 1)
  (hquit : quit_time = 18) : 
  ∃ T : ℝ, T = 18 :=
by {
  sorry
}

end project_completion_time_l78_7802


namespace multiples_six_or_eight_not_both_l78_7856

def countMultiples (n m : ℕ) : ℕ := n / m

def LCM (a b : ℕ) : ℕ := a * b / Nat.gcd a b

theorem multiples_six_or_eight_not_both : 
  let multiplesSix := countMultiples 200 6
  let multiplesEight := countMultiples 200 8
  let commonMultiple := countMultiples 200 (LCM 6 8)
  multiplesSix - commonMultiple + multiplesEight - commonMultiple = 42 := 
by
  sorry

end multiples_six_or_eight_not_both_l78_7856


namespace solve_abs_eq_l78_7884

theorem solve_abs_eq (x : ℝ) (h : |x + 2| = |x - 3|) : x = 1 / 2 :=
sorry

end solve_abs_eq_l78_7884


namespace symmetry_condition_l78_7809

theorem symmetry_condition (a : ℝ) :
  (∀ x : ℝ, |x + 1| + |x - a| = |(2 - x) + 1| + |(2 - x) - a|) ↔ a = 3 :=
by
  sorry

end symmetry_condition_l78_7809


namespace power_equality_l78_7889

theorem power_equality (p : ℕ) : 16^10 = 4^p → p = 20 :=
by
  intro h
  -- proof goes here
  sorry

end power_equality_l78_7889


namespace parabola_focus_coordinates_l78_7863

theorem parabola_focus_coordinates :
  (∃ f : ℝ × ℝ, f = (0, 2) ∧ ∀ x y : ℝ, y = (1/8) * x^2 ↔ f = (0, 2)) :=
sorry

end parabola_focus_coordinates_l78_7863


namespace work_completion_l78_7875

theorem work_completion (A B C D : ℝ) :
  (A = 1 / 5) →
  (A + C = 2 / 5) →
  (B + C = 1 / 4) →
  (A + D = 1 / 3.6) →
  (B + C + D = 1 / 2) →
  B = 1 / 20 :=
by
  sorry

end work_completion_l78_7875


namespace intersection_empty_implies_m_leq_neg1_l78_7851

theorem intersection_empty_implies_m_leq_neg1 (m : ℝ) :
  (∀ (x y: ℝ), (x < m) → (y = x^2 + 2*x) → y < -1) →
  m ≤ -1 :=
by
  intro h
  sorry

end intersection_empty_implies_m_leq_neg1_l78_7851


namespace book_club_boys_count_l78_7850

theorem book_club_boys_count (B G : ℕ) 
  (h1 : B + G = 30) 
  (h2 : B + (1 / 3 : ℝ) * G = 18) :
  B = 12 :=
by
  have h3 : 3 • B + G = 54 := sorry
  have h4 : 3 • B + G - (B + G) = 54 - 30 := sorry
  have h5 : 2 • B = 24 := sorry
  have h6 : B = 12 := sorry
  exact h6

end book_club_boys_count_l78_7850


namespace jennifer_money_left_l78_7887

def money_left (initial_amount sandwich_fraction museum_fraction book_fraction : ℚ) : ℚ :=
  initial_amount - (initial_amount * sandwich_fraction + initial_amount * museum_fraction + initial_amount * book_fraction)

theorem jennifer_money_left :
  money_left 150 (1/5) (1/6) (1/2) = 20 := by
  -- Proof goes here
  sorry

end jennifer_money_left_l78_7887


namespace sum_of_squares_consecutive_nat_l78_7897

theorem sum_of_squares_consecutive_nat (n : ℕ) (h : n = 26) : (n - 1) ^ 2 + n ^ 2 + (n + 1) ^ 2 = 2030 :=
by
  sorry

end sum_of_squares_consecutive_nat_l78_7897


namespace adoption_complete_in_7_days_l78_7829

-- Define the initial number of puppies
def initial_puppies := 9

-- Define the number of puppies brought in later
def additional_puppies := 12

-- Define the number of puppies adopted per day
def adoption_rate := 3

-- Define the total number of puppies
def total_puppies : Nat := initial_puppies + additional_puppies

-- Define the number of days required to adopt all puppies
def adoption_days : Nat := total_puppies / adoption_rate

-- Prove that the number of days to adopt all puppies is 7
theorem adoption_complete_in_7_days : adoption_days = 7 := by
  -- The exact implementation of the proof is not necessary,
  -- so we use sorry to skip the proof.
  sorry

end adoption_complete_in_7_days_l78_7829


namespace minimum_total_trips_l78_7835

theorem minimum_total_trips :
  ∃ (x y : ℕ), (31 * x + 32 * y = 5000) ∧ (x + y = 157) :=
by
  sorry

end minimum_total_trips_l78_7835


namespace find_a_l78_7862

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if h : x < 0 then a ^ x - 1 else 2 * x ^ 2

theorem find_a (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : ∀ m n : ℝ, f a m ≤ f a n ↔ m ≤ n)
  (h4 : f a a = 5 * a - 2) : a = 2 :=
sorry

end find_a_l78_7862


namespace possible_k_value_l78_7866

theorem possible_k_value (a n k : ℕ) (h1 : n > 1) (h2 : 10^(n-1) ≤ a ∧ a < 10^n)
    (h3 : b = a * (10^n + 1)) (h4 : k = b / a^2) (h5 : b = a * 10 ^n + a) :
  k = 7 := 
sorry

end possible_k_value_l78_7866


namespace interval_is_correct_l78_7842

def total_population : ℕ := 2000
def sample_size : ℕ := 40
def interval_between_segments (N : ℕ) (n : ℕ) : ℕ := N / n

theorem interval_is_correct : interval_between_segments total_population sample_size = 50 :=
by
  sorry

end interval_is_correct_l78_7842


namespace fewest_students_possible_l78_7858

theorem fewest_students_possible (N : ℕ) :
  (N % 5 = 2) ∧ (N % 6 = 3) ∧ (N % 8 = 4) ↔ N = 59 :=
by
  sorry

end fewest_students_possible_l78_7858


namespace sum_of_inscribed_angles_l78_7885

-- Define the circle and its division into arcs.
def circle_division (O : Type) (total_arcs : ℕ) := total_arcs = 16

-- Define the inscribed angles x and y.
def inscribed_angle (O : Type) (arc_subtended : ℕ) := arc_subtended

-- Define the conditions for angles x and y subtending 3 and 5 arcs respectively.
def angle_x := inscribed_angle ℝ 3
def angle_y := inscribed_angle ℝ 5

-- Theorem stating the sum of the inscribed angles x and y.
theorem sum_of_inscribed_angles 
  (O : Type)
  (total_arcs : ℕ)
  (h1 : circle_division O total_arcs)
  (h2 : inscribed_angle O angle_x = 3)
  (h3 : inscribed_angle O angle_y = 5) :
  33.75 + 56.25 = 90 :=
by
  sorry

end sum_of_inscribed_angles_l78_7885


namespace income_expenses_opposite_l78_7861

def income_denotation (income : Int) : Int := income

theorem income_expenses_opposite :
  income_denotation 5 = 5 →
  income_denotation (-5) = -5 :=
by
  intro h
  sorry

end income_expenses_opposite_l78_7861


namespace largest_angle_bounds_triangle_angles_l78_7874

theorem largest_angle_bounds (A B C : ℝ) (angle_A angle_B angle_C : ℝ)
  (h_triangle : angle_A + angle_B + angle_C = 180)
  (h_tangent : angle_B + 2 * angle_C = 90) :
  90 ≤ angle_A ∧ angle_A < 135 :=
sorry

theorem triangle_angles (A B C : ℝ) (angle_A angle_B angle_C : ℝ)
  (h_triangle : angle_A + angle_B + angle_C = 180)
  (h_tangent_B : angle_B + 2 * angle_C = 90)
  (h_tangent_C : angle_C + 2 * angle_B = 90) :
  angle_A = 120 ∧ angle_B = 30 ∧ angle_C = 30 :=
sorry

end largest_angle_bounds_triangle_angles_l78_7874


namespace problem_distribution_l78_7840

theorem problem_distribution:
  let num_problems := 6
  let num_friends := 15
  (num_friends ^ num_problems) = 11390625 :=
by sorry

end problem_distribution_l78_7840


namespace angle_in_third_quadrant_l78_7830

theorem angle_in_third_quadrant (α : ℝ) (h1 : Real.sin α * Real.cos α > 0) (h2 : Real.sin α * Real.tan α < 0) : 
  (π < α ∧ α < 3 * π / 2) :=
by
  sorry

end angle_in_third_quadrant_l78_7830


namespace interest_rate_simple_and_compound_l78_7846

theorem interest_rate_simple_and_compound (P T: ℝ) (SI CI R: ℝ) 
  (simple_interest_eq: SI = (P * R * T) / 100)
  (compound_interest_eq: CI = P * ((1 + R / 100) ^ T - 1)) 
  (hP : P = 3000) (hT : T = 2) (hSI : SI = 300) (hCI : CI = 307.50) :
  R = 5 :=
by
  sorry

end interest_rate_simple_and_compound_l78_7846


namespace students_in_dexters_high_school_l78_7833

variables (D S N : ℕ)

theorem students_in_dexters_high_school :
  (D = 4 * S) ∧
  (D + S + N = 3600) ∧
  (N = S - 400) →
  D = 8000 / 3 := 
sorry

end students_in_dexters_high_school_l78_7833


namespace square_areas_l78_7892

theorem square_areas (z : ℂ) 
  (h1 : ¬ (2 : ℂ) * z^2 = z)
  (h2 : ¬ (3 : ℂ) * z^3 = z)
  (sz : (3 * z^3 - z) = (I * (2 * z^2 - z)) ∨ (3 * z^3 - z) = (-I * (2 * z^2 - z))) :
  ∃ (areas : Finset ℝ), areas = {85, 4500} :=
by {
  sorry
}

end square_areas_l78_7892
