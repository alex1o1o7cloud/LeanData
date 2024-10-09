import Mathlib

namespace circle_tangent_line_standard_equation_l1152_115257

-- Problem Statement:
-- Prove that the standard equation of the circle with center at (1,1)
-- and tangent to the line x + y = 4 is (x - 1)^2 + (y - 1)^2 = 2
theorem circle_tangent_line_standard_equation :
  (forall (x y : ℝ), (x + y = 4) -> (x - 1)^2 + (y - 1)^2 = 2) := by
  sorry

end circle_tangent_line_standard_equation_l1152_115257


namespace one_number_greater_than_one_l1152_115225

theorem one_number_greater_than_one 
  (a b c : ℝ) 
  (h_pos_a : 0 < a) 
  (h_pos_b : 0 < b) 
  (h_pos_c : 0 < c)
  (h_prod : a * b * c = 1) 
  (h_sum : a + b + c > (1 / a) + (1 / b) + (1 / c)) 
  : (1 < a ∨ 1 < b ∨ 1 < c) ∧ ¬(1 < a ∧ 1 < b ∧ 1 < c) :=
by
  sorry

end one_number_greater_than_one_l1152_115225


namespace boys_meet_time_is_correct_l1152_115240

structure TrackMeetProblem where
  (track_length : ℕ) -- Track length in meters
  (speed_first_boy_kmh : ℚ) -- Speed of the first boy in km/hr
  (speed_second_boy_kmh : ℚ) -- Speed of the second boy in km/hr

noncomputable def time_to_meet (p : TrackMeetProblem) : ℚ :=
  let speed_first_boy_ms := (p.speed_first_boy_kmh * 1000) / 3600
  let speed_second_boy_ms := (p.speed_second_boy_kmh * 1000) / 3600
  let relative_speed := speed_first_boy_ms + speed_second_boy_ms
  (p.track_length : ℚ) / relative_speed

theorem boys_meet_time_is_correct (p : TrackMeetProblem) : 
  p.track_length = 4800 → 
  p.speed_first_boy_kmh = 61.3 → 
  p.speed_second_boy_kmh = 97.5 → 
  time_to_meet p = 108.8 := by
  intros
  sorry  

end boys_meet_time_is_correct_l1152_115240


namespace fewer_mpg_in_city_l1152_115216

theorem fewer_mpg_in_city
  (highway_miles : ℕ)
  (city_miles : ℕ)
  (city_mpg : ℕ)
  (highway_mpg : ℕ)
  (tank_size : ℝ) :
  highway_miles = 462 →
  city_miles = 336 →
  city_mpg = 32 →
  tank_size = 336 / 32 →
  highway_mpg = 462 / tank_size →
  (highway_mpg - city_mpg) = 12 :=
by
  intros h_highway_miles h_city_miles h_city_mpg h_tank_size h_highway_mpg
  sorry

end fewer_mpg_in_city_l1152_115216


namespace find_numbers_between_70_and_80_with_gcd_6_l1152_115293

theorem find_numbers_between_70_and_80_with_gcd_6 :
  ∃ n, 70 ≤ n ∧ n ≤ 80 ∧ Nat.gcd n 30 = 6 ∧ (n = 72 ∨ n = 78) :=
by
  sorry

end find_numbers_between_70_and_80_with_gcd_6_l1152_115293


namespace line_equation_l1152_115237

theorem line_equation (A : ℝ × ℝ) (hA : A = (1, 4)) 
  (h_intercept_sum : ∃ b c, b + c = 0 ∧ (∀ x y, A.1 * x + A.2 * y = 1 ∨ A.1 * x + A.2 * y = -1)) :
  (∃ m n, m = 4 ∧ n = -1 ∧ (∀ x y, m * x + n * y = 0)) ∨ 
  (∃ p q r, p = 1 ∧ q = -1 ∧ r = 3 ∧ (∀ x y, p * x + q * y + r = 0)) :=
by
  sorry

end line_equation_l1152_115237


namespace red_lights_l1152_115283

theorem red_lights (total_lights yellow_lights blue_lights red_lights : ℕ)
  (h1 : total_lights = 95)
  (h2 : yellow_lights = 37)
  (h3 : blue_lights = 32)
  (h4 : red_lights = total_lights - (yellow_lights + blue_lights)) :
  red_lights = 26 := by
  sorry

end red_lights_l1152_115283


namespace value_of_expression_l1152_115266

theorem value_of_expression : (0.3 : ℝ)^2 + 0.1 = 0.19 := 
by sorry

end value_of_expression_l1152_115266


namespace equation_one_solution_equation_two_no_solution_l1152_115284

theorem equation_one_solution (x : ℝ) (hx1 : x ≠ 3) : (2 * x + 9) / (3 - x) = (4 * x - 7) / (x - 3) ↔ x = -1 / 3 := 
by 
    sorry

theorem equation_two_no_solution (x : ℝ) (hx2 : x ≠ 1) (hx3 : x ≠ -1) : 
    (x + 1) / (x - 1) - 4 / (x ^ 2 - 1) = 1 → False := 
by 
    sorry

end equation_one_solution_equation_two_no_solution_l1152_115284


namespace if_a_eq_b_then_ac_eq_bc_l1152_115295

theorem if_a_eq_b_then_ac_eq_bc (a b c : ℝ) : a = b → ac = bc :=
sorry

end if_a_eq_b_then_ac_eq_bc_l1152_115295


namespace circumference_proportionality_l1152_115298

theorem circumference_proportionality (r : ℝ) (C : ℝ) (k : ℝ) (π : ℝ)
  (h1 : C = k * r)
  (h2 : C = 2 * π * r) :
  k = 2 * π :=
sorry

end circumference_proportionality_l1152_115298


namespace log_base_5_domain_correct_l1152_115220

def log_base_5_domain : Set ℝ := {x : ℝ | x > 0}

theorem log_base_5_domain_correct : (∀ x : ℝ, x > 0 ↔ x ∈ log_base_5_domain) :=
by sorry

end log_base_5_domain_correct_l1152_115220


namespace smallest_natural_b_for_root_exists_l1152_115255

-- Define the problem's conditions
def quadratic_eqn (b : ℕ) := ∀ x : ℝ, x^2 + (b : ℝ) * x + 25 = 0

def discriminant (a b c : ℝ) := b^2 - 4 * a * c

-- Define the main problem statement
theorem smallest_natural_b_for_root_exists :
  ∃ b : ℕ, (discriminant 1 b 25 ≥ 0) ∧ (∀ b' : ℕ, b' < b → discriminant 1 b' 25 < 0) ∧ b = 10 :=
by
  sorry

end smallest_natural_b_for_root_exists_l1152_115255


namespace find_s_l1152_115281

theorem find_s (s : ℝ) (t : ℝ) (h1 : t = 4) (h2 : t = 12 * s^2 + 2 * s) : s = 0.5 ∨ s = -2 / 3 :=
by
  sorry

end find_s_l1152_115281


namespace grandfather_age_l1152_115275

theorem grandfather_age :
  ∃ (a b : ℕ), a < 10 ∧ b < 10 ∧ 10 * a + b = a + b^2 ∧ 10 * a + b = 89 :=
by
  sorry

end grandfather_age_l1152_115275


namespace larger_number_is_23_l1152_115269

theorem larger_number_is_23 (x y : ℕ) (h_sum : x + y = 40) (h_diff : x - y = 6) : x = 23 :=
sorry

end larger_number_is_23_l1152_115269


namespace metallic_sheet_width_l1152_115226

theorem metallic_sheet_width 
  (length_of_cut_square : ℝ) (original_length_of_sheet : ℝ) (volume_of_box : ℝ) (w : ℝ)
  (h1 : length_of_cut_square = 5) 
  (h2 : original_length_of_sheet = 48) 
  (h3 : volume_of_box = 4940) : 
  (38 * (w - 10) * 5 = 4940) → w = 36 :=
by
  intros
  sorry

end metallic_sheet_width_l1152_115226


namespace count_whole_numbers_between_cuberoots_l1152_115248

theorem count_whole_numbers_between_cuberoots : 
  ∃ (n : ℕ), n = 7 ∧ 
      ∀ x : ℝ, (3 < x ∧ x < 4 → ∃ k : ℕ, k = 4) ∧ 
                (9 < x ∧ x ≤ 10 → ∃ k : ℕ, k = 10) :=
sorry

end count_whole_numbers_between_cuberoots_l1152_115248


namespace fixed_monthly_fee_l1152_115204

theorem fixed_monthly_fee (f h : ℝ) 
  (feb_bill : f + h = 18.72)
  (mar_bill : f + 3 * h = 33.78) :
  f = 11.19 :=
by
  sorry

end fixed_monthly_fee_l1152_115204


namespace circle_bisect_line_l1152_115262

theorem circle_bisect_line (a : ℝ) :
  (∃ x y, (x - a) ^ 2 + (y + 1) ^ 2 = 3 ∧ 5 * x + 4 * y - a = 0) →
  a = 1 :=
by
  sorry

end circle_bisect_line_l1152_115262


namespace find_growth_rate_l1152_115212

noncomputable def donation_first_day : ℝ := 10000
noncomputable def donation_third_day : ℝ := 12100
noncomputable def growth_rate (x : ℝ) : Prop :=
  (donation_first_day * (1 + x) ^ 2 = donation_third_day)

theorem find_growth_rate : ∃ x : ℝ, growth_rate x ∧ x = 0.1 :=
by
  sorry

end find_growth_rate_l1152_115212


namespace min_value_of_expression_l1152_115244

noncomputable def minimum_value_expression : ℝ :=
  let f (a b : ℝ) := a^4 + b^4 + 16 / (a^2 + b^2)^2
  4

theorem min_value_of_expression (a b : ℝ) (h : 0 < a ∧ 0 < b) : 
  let f := a^4 + b^4 + 16 / (a^2 + b^2)^2
  ∃ c : ℝ, f = c ∧ c = 4 :=
sorry

end min_value_of_expression_l1152_115244


namespace team_structure_ways_l1152_115236

open Nat

noncomputable def combinatorial_structure (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem team_structure_ways :
  let total_members := 13
  let team_lead_choices := total_members
  let remaining_after_lead := total_members - 1
  let project_manager_choices := combinatorial_structure remaining_after_lead 3
  let remaining_after_pm1 := remaining_after_lead - 3
  let subordinate_choices_pm1 := combinatorial_structure remaining_after_pm1 3
  let remaining_after_pm2 := remaining_after_pm1 - 3
  let subordinate_choices_pm2 := combinatorial_structure remaining_after_pm2 3
  let remaining_after_pm3 := remaining_after_pm2 - 3
  let subordinate_choices_pm3 := combinatorial_structure remaining_after_pm3 3
  let total_ways := team_lead_choices * project_manager_choices * subordinate_choices_pm1 * subordinate_choices_pm2 * subordinate_choices_pm3
  total_ways = 4804800 :=
by
  sorry

end team_structure_ways_l1152_115236


namespace factor_expression_l1152_115280

theorem factor_expression (x : ℝ) : x * (x + 3) + 2 * (x + 3) = (x + 2) * (x + 3) :=
by
  sorry

end factor_expression_l1152_115280


namespace solution_set_of_x_l1152_115211

noncomputable def satisfies_condition (x : ℝ) : Prop :=
  ⌊⌊3 * x⌋ - 1/3⌋ = ⌊x + 3⌋

theorem solution_set_of_x :
  { x : ℝ | satisfies_condition x } = { x : ℝ | 5/3 ≤ x ∧ x < 7/3 } :=
by
  sorry

end solution_set_of_x_l1152_115211


namespace real_root_exists_l1152_115252

theorem real_root_exists (p : ℝ) : ∃ x : ℝ, x^4 + 2*p*x^3 + x^3 + 2*p*x + 1 = 0 :=
sorry

end real_root_exists_l1152_115252


namespace option_A_option_B_option_C_option_D_verify_options_l1152_115258

open Real

-- Option A: Prove the maximum value of x(6-x) given 0 < x < 6 is 9.
theorem option_A (x : ℝ) (h1 : 0 < x) (h2 : x < 6) : 
  ∃ (max_value : ℝ), max_value = 9 ∧ ∀(y : ℝ), 0 < y ∧ y < 6 → y * (6 - y) ≤ max_value :=
sorry

-- Option B: Prove the minimum value of x^2 + 1/(x^2 + 3) for x in ℝ is not -1.
theorem option_B (x : ℝ) : ¬(∃ (min_value : ℝ), min_value = -1 ∧ ∀(y : ℝ), (y ^ 2) + 1 / (y ^ 2 + 3) ≥ min_value) :=
sorry

-- Option C: Prove the maximum value of xy given x + 2y + xy = 6 and x, y > 0 is 2.
theorem option_C (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x + 2 * y + x * y = 6) : 
  ∃ (max_value : ℝ), max_value = 2 ∧ ∀(u v : ℝ), 0 < u ∧ 0 < v ∧ u + 2 * v + u * v = 6 → u * v ≤ max_value :=
sorry

-- Option D: Prove the minimum value of 2x + y given x + 4y + 4 = xy and x, y > 0 is 17.
theorem option_D (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x + 4 * y + 4 = x * y) : 
  ∃ (min_value : ℝ), min_value = 17 ∧ ∀(u v : ℝ), 0 < u ∧ 0 < v ∧ u + 4 * v + 4 = u * v → 2 * u + v ≥ min_value :=
sorry

-- Combine to verify which options are correct
theorem verify_options (A_correct B_correct C_correct D_correct : Prop) :
  A_correct = true ∧ B_correct = false ∧ C_correct = true ∧ D_correct = true :=
sorry

end option_A_option_B_option_C_option_D_verify_options_l1152_115258


namespace imaginary_part_of_complex_num_l1152_115215

-- Define the complex number and the imaginary part condition
def complex_num : ℂ := ⟨1, 2⟩

-- Define the theorem to prove the imaginary part is 2
theorem imaginary_part_of_complex_num : complex_num.im = 2 :=
by
  -- The proof steps would go here
  sorry

end imaginary_part_of_complex_num_l1152_115215


namespace average_first_50_even_numbers_l1152_115218

-- Condition: The sequence starts from 2.
-- Condition: The sequence consists of the first 50 even numbers.
def first50EvenNumbers : List ℤ := List.range' 2 100

theorem average_first_50_even_numbers : (first50EvenNumbers.sum / 50 = 51) :=
by
  sorry

end average_first_50_even_numbers_l1152_115218


namespace certain_number_is_14_l1152_115251

theorem certain_number_is_14 
  (a b n : ℕ) 
  (h1 : ∃ k1, a = k1 * n) 
  (h2 : ∃ k2, b = k2 * n) 
  (h3 : b = a + 11 * n) 
  (h4 : b = a + 22 * 7) : n = 14 := 
by 
  sorry

end certain_number_is_14_l1152_115251


namespace number_of_sets_count_number_of_sets_l1152_115292

theorem number_of_sets (P : Set ℕ) :
  ({1, 2} ⊆ P ∧ P ⊆ {1, 2, 3, 4}) → (P = {1, 2} ∨ P = {1, 2, 3} ∨ P = {1, 2, 4}) :=
sorry

theorem count_number_of_sets :
  ∃ (Ps : Finset (Set ℕ)), 
  (∀ P ∈ Ps, {1, 2} ⊆ P ∧ P ⊆ {1, 2, 3, 4}) ∧ Ps.card = 3 :=
sorry

end number_of_sets_count_number_of_sets_l1152_115292


namespace gerald_remaining_pfennigs_l1152_115274

-- Definitions of Gerald's initial money and the costs of items
def farthings : Nat := 54
def groats : Nat := 8
def florins : Nat := 17
def meat_pie_cost : Nat := 120
def sausage_roll_cost : Nat := 75

-- Conversion rates
def farthings_to_pfennigs (f : Nat) : Nat := f / 6
def groats_to_pfennigs (g : Nat) : Nat := g * 4
def florins_to_pfennigs (f : Nat) : Nat := f * 40

-- Total pfennigs Gerald has
def total_pfennigs : Nat :=
  farthings_to_pfennigs farthings + groats_to_pfennigs groats + florins_to_pfennigs florins

-- Total cost of both items
def total_cost : Nat := meat_pie_cost + sausage_roll_cost

-- Gerald's remaining pfennigs after purchase
def remaining_pfennigs : Nat := total_pfennigs - total_cost

theorem gerald_remaining_pfennigs :
  remaining_pfennigs = 526 :=
by
  sorry

end gerald_remaining_pfennigs_l1152_115274


namespace employees_use_public_transportation_l1152_115200

theorem employees_use_public_transportation 
  (total_employees : ℕ)
  (percentage_drive : ℕ)
  (half_of_non_drivers_take_transport : ℕ)
  (h1 : total_employees = 100)
  (h2 : percentage_drive = 60)
  (h3 : half_of_non_drivers_take_transport = 1 / 2) 
  : (total_employees - percentage_drive * total_employees / 100) / 2 = 20 := 
  by
  sorry

end employees_use_public_transportation_l1152_115200


namespace partial_fraction_decomposition_l1152_115232

noncomputable def polynomial : Polynomial ℝ := Polynomial.X^3 - 24 * Polynomial.X^2 + 143 * Polynomial.X - 210

theorem partial_fraction_decomposition (A B C p q r : ℝ) (h1 : Polynomial.roots polynomial = {p, q, r}) 
  (h2 : ∀ s : ℝ, 1 / (s^3 - 24 * s^2 + 143 * s - 210) = A / (s - p) + B / (s - q) + C / (s - r)) :
  1 / A + 1 / B + 1 / C = 243 :=
by
  sorry

end partial_fraction_decomposition_l1152_115232


namespace bags_filled_l1152_115246

def bags_filled_on_certain_day (x : ℕ) : Prop :=
  let bags := x + 3
  let total_cans := 8 * bags
  total_cans = 72

theorem bags_filled {x : ℕ} (h : bags_filled_on_certain_day x) : x = 6 :=
  sorry

end bags_filled_l1152_115246


namespace loci_of_square_view_l1152_115290

-- Definitions based on the conditions in a)
def square (A B C D : Point) : Prop := -- Formalize what it means to be a square
sorry

def region1 (P : Point) (A B : Point) : Prop := -- Formalize the definition of region 1
sorry

def region2 (P : Point) (B C : Point) : Prop := -- Formalize the definition of region 2
sorry

-- Additional region definitions (3 through 9)
-- ...

def visible_side (P A B : Point) : Prop := -- Definition of a visible side from a point
sorry

def visible_diagonal (P A C : Point) : Prop := -- Definition of a visible diagonal from a point
sorry

def loci_of_angles (angle : ℝ) : Set Point := -- Definition of loci for a given angle
sorry

-- Main problem statement with the question and conditions as hypotheses
theorem loci_of_square_view (A B C D P : Point) (angle : ℝ) :
    square A B C D →
    (∀ P, (visible_side P A B ∨ visible_side P B C ∨ visible_side P C D ∨ visible_side P D A → 
             P ∈ loci_of_angles angle) ∧ 
         ((region1 P A B ∨ region2 P B C) → visible_diagonal P A C)) →
    -- Additional conditions here
    True :=
-- Prove that the loci is as described in the solution
sorry

end loci_of_square_view_l1152_115290


namespace ball_bounce_height_l1152_115222

theorem ball_bounce_height :
  ∃ k : ℕ, 800 * (1 / 2 : ℝ)^k < 2 ∧ k ≥ 9 :=
by
  sorry

end ball_bounce_height_l1152_115222


namespace find_ratio_a6_b6_l1152_115202

noncomputable def S (n : ℕ) : ℕ := sorry
noncomputable def T (n : ℕ) : ℕ := sorry
noncomputable def a (n : ℕ) : ℕ := sorry
noncomputable def b (n : ℕ) : ℕ := sorry

theorem find_ratio_a6_b6 
  (H1 : ∀ n: ℕ, n > 0 → (S n / T n : ℚ) = n / (2 * n + 1)) :
  (a 6 / b 6 : ℚ) = 11 / 23 :=
sorry

end find_ratio_a6_b6_l1152_115202


namespace S2_side_length_656_l1152_115291

noncomputable def S1_S2_S3_side_lengths (l1 l2 a b c : ℕ) (total_length : ℕ) : Prop :=
  l1 + l2 + a + b + c = total_length

theorem S2_side_length_656 :
  ∃ (l1 l2 a c : ℕ), S1_S2_S3_side_lengths l1 l2 a 656 c 3322 :=
by
  sorry

end S2_side_length_656_l1152_115291


namespace range_of_a_l1152_115296

theorem range_of_a (a : ℝ) : (∀ x : ℝ, |x + 1| + |x - 2| > a) → a < 3 :=
by
  sorry

end range_of_a_l1152_115296


namespace decagon_diagonals_l1152_115286

-- Define the number of sides of the polygon
def n : ℕ := 10

-- Calculate the number of diagonals in an n-sided polygon
def number_of_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

-- State the theorem that the number of diagonals in a decagon is 35
theorem decagon_diagonals : number_of_diagonals n = 35 := by
  sorry

end decagon_diagonals_l1152_115286


namespace min_value_of_a_l1152_115273

theorem min_value_of_a (a b c : ℝ) (h₁ : a > 0) (h₂ : ∃ p q : ℝ, 0 < p ∧ p < 2 ∧ 0 < q ∧ q < 2 ∧ 
  ∀ x, ax^2 + bx + c = a * (x - p) * (x - q)) (h₃ : 25 * a + 10 * b + 4 * c ≥ 4) (h₄ : c ≥ 1) : 
  a ≥ 16 / 25 :=
sorry

end min_value_of_a_l1152_115273


namespace union_A_B_intersection_complementA_B_range_of_a_l1152_115214

-- Definition of the universal set U, sets A and B
def U : Set ℝ := Set.univ
def A : Set ℝ := {x | 1 ≤ x ∧ x < 5}
def B : Set ℝ := {x | 2 < x ∧ x < 8}

-- Complement of A in the universal set U
def complement_A : Set ℝ := {x | x < 1 ∨ x ≥ 5}

-- Definition of set C parametrized by a
def C (a : ℝ) : Set ℝ := {x | -a < x ∧ x ≤ a + 3}

-- Prove that A ∪ B is {x | 1 ≤ x < 8}
theorem union_A_B : A ∪ B = {x | 1 ≤ x ∧ x < 8} :=
sorry

-- Prove that (complement_U A) ∩ B = {x | 5 ≤ x < 8}
theorem intersection_complementA_B : (complement_A ∩ B) = {x | 5 ≤ x ∧ x < 8} :=
sorry

-- Prove the range of values for a if C ∩ A = C
theorem range_of_a (a : ℝ) : (C a ∩ A = C a) → a ≤ -1 :=
sorry

end union_A_B_intersection_complementA_B_range_of_a_l1152_115214


namespace problem_solved_by_at_least_one_student_l1152_115271

theorem problem_solved_by_at_least_one_student (P_A P_B : ℝ) 
  (hA : P_A = 0.8) 
  (hB : P_B = 0.9) :
  (1 - (1 - P_A) * (1 - P_B) = 0.98) :=
by
  have pAwrong := 1 - P_A
  have pBwrong := 1 - P_B
  have both_wrong := pAwrong * pBwrong
  have one_right := 1 - both_wrong
  sorry

end problem_solved_by_at_least_one_student_l1152_115271


namespace exist_m_squared_plus_9_mod_2_pow_n_minus_1_l1152_115279

theorem exist_m_squared_plus_9_mod_2_pow_n_minus_1 (n : ℕ) (hn : n > 0) :
  (∃ m : ℤ, (m^2 + 9) % (2^n - 1) = 0) ↔ ∃ k : ℕ, n = 2^k :=
by
  sorry

end exist_m_squared_plus_9_mod_2_pow_n_minus_1_l1152_115279


namespace int_solutions_to_inequalities_l1152_115276

theorem int_solutions_to_inequalities :
  { x : ℤ | -5 * x ≥ 3 * x + 15 } ∩
  { x : ℤ | -3 * x ≤ 9 } ∩
  { x : ℤ | 7 * x ≤ -14 } = { -3, -2 } :=
by {
  sorry
}

end int_solutions_to_inequalities_l1152_115276


namespace class_heights_mode_median_l1152_115243

def mode (l : List ℕ) : ℕ := sorry
def median (l : List ℕ) : ℕ := sorry

theorem class_heights_mode_median 
  (A : List ℕ) -- Heights of students from Class A
  (B : List ℕ) -- Heights of students from Class B
  (hA : A = [170, 170, 169, 171, 171, 171])
  (hB : B = [168, 170, 170, 172, 169, 170]) :
  mode A = 171 ∧ median B = 170 := sorry

end class_heights_mode_median_l1152_115243


namespace ratio_of_money_with_Gopal_and_Krishan_l1152_115223

theorem ratio_of_money_with_Gopal_and_Krishan 
  (R G K : ℕ) 
  (h1 : R = 735) 
  (h2 : K = 4335) 
  (h3 : R * 17 = G * 7) :
  G * 4335 = 1785 * K :=
by
  sorry

end ratio_of_money_with_Gopal_and_Krishan_l1152_115223


namespace sequence_general_formula_l1152_115234

theorem sequence_general_formula (a : ℕ → ℚ) (h₀ : a 1 = 3 / 5)
    (h₁ : ∀ n : ℕ, a (n + 1) = a n / (2 * a n + 1)) :
  ∀ n : ℕ, a n = 3 / (6 * n - 1) := 
by sorry

end sequence_general_formula_l1152_115234


namespace mixed_groups_count_l1152_115209

theorem mixed_groups_count 
    (total_children : ℕ)
    (total_groups : ℕ)
    (group_size : ℕ)
    (total_games : ℕ)
    (boy_vs_boy_games : ℕ)
    (girl_vs_girl_games : ℕ)
    (total_mixed_games : ℕ)
    (games_per_group : ℕ)
    (mixed_group_games_per_group : ℕ)
    (mixed_groups : ℕ) :
    total_children = 90 →
    total_groups = 30 →
    group_size = 3 →
    total_games = 90 →
    boy_vs_boy_games = 30 →
    girl_vs_girl_games = 14 →
    total_mixed_games = 46 →
    games_per_group = 3 →
    mixed_group_games_per_group = 2 →
    mixed_groups = total_mixed_games / mixed_group_games_per_group →
    mixed_groups = 23 :=
by
    intros
    sorry

end mixed_groups_count_l1152_115209


namespace sin_double_angle_l1152_115249

theorem sin_double_angle (α : ℝ) (h : Real.tan α = 3 / 4) : Real.sin (2 * α) = 24 / 25 := by
  sorry

end sin_double_angle_l1152_115249


namespace work_completion_l1152_115267

theorem work_completion (original_men planned_days absent_men remaining_men completion_days : ℕ) :
  original_men = 180 → 
  planned_days = 55 →
  absent_men = 15 →
  remaining_men = original_men - absent_men →
  remaining_men * completion_days = original_men * planned_days →
  completion_days = 60 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end work_completion_l1152_115267


namespace min_packs_to_buy_120_cans_l1152_115230

/-- Prove that the minimum number of packs needed to buy exactly 120 cans of soda,
with packs available in sizes of 8, 15, and 30 cans, is 4. -/
theorem min_packs_to_buy_120_cans : 
  ∃ n, n = 4 ∧ ∀ x y z: ℕ, 8 * x + 15 * y + 30 * z = 120 → x + y + z ≥ n :=
sorry

end min_packs_to_buy_120_cans_l1152_115230


namespace proof_sum_q_p_x_l1152_115282

def p (x : ℝ) : ℝ := |x| - 3
def q (x : ℝ) : ℝ := -|x|

-- define the list of x values
def x_values : List ℝ := [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]

-- define q_p_x to apply q to p of each x
def q_p_x : List ℝ := x_values.map (λ x => q (p x))

-- define the sum of q(p(x)) for given x values
def sum_q_p_x : ℝ := q_p_x.sum

theorem proof_sum_q_p_x : sum_q_p_x = -15 := by
  -- steps of solution
  sorry

end proof_sum_q_p_x_l1152_115282


namespace find_ages_l1152_115263

-- Define that f is a polynomial with integer coefficients
noncomputable def f : ℤ → ℤ := sorry

-- Given conditions
axiom f_at_7 : f 7 = 77
axiom f_at_b : ∃ b : ℕ, f b = 85
axiom f_at_c : ∃ c : ℕ, f c = 0

-- Define what we need to prove
theorem find_ages : ∃ b c : ℕ, (b - 7 ∣ 8) ∧ (c - b ∣ 85) ∧ (c - 7 ∣ 77) ∧ (b = 9) ∧ (c = 14) :=
sorry

end find_ages_l1152_115263


namespace probability_target_hit_l1152_115256

theorem probability_target_hit (P_A P_B : ℚ) (h1 : P_A = 1/2) (h2 : P_B = 1/3) : 
  (1 - (1 - P_A) * (1 - P_B)) = 2/3 :=
by
  sorry

end probability_target_hit_l1152_115256


namespace rounded_product_less_than_original_l1152_115203

theorem rounded_product_less_than_original
  (x y z : ℝ) (hx_pos : 0 < x) (hy_pos : 0 < y) (hz_pos : 0 < z) (hxy : x > 2 * y) :
  (x + z) * (y - z) < x * y :=
by
  sorry

end rounded_product_less_than_original_l1152_115203


namespace factorize_expression_l1152_115239

theorem factorize_expression (a b : ℝ) : 2 * a^2 - 8 * b^2 = 2 * (a + 2 * b) * (a - 2 * b) :=
by
  sorry

end factorize_expression_l1152_115239


namespace vertex_closest_point_l1152_115254

theorem vertex_closest_point (a : ℝ) (x y : ℝ) :
  (x^2 = 2 * y) ∧ (y ≥ 0) ∧ ((y^2 + 2 * (1 - a) * y + a^2) ≤ 0) → a ≤ 1 :=
by 
  sorry

end vertex_closest_point_l1152_115254


namespace circle_line_chord_length_l1152_115259

theorem circle_line_chord_length :
  ∀ (k m : ℝ), (∀ x y : ℝ, x^2 + y^2 = 4 → y = k * x + m → ∃ (a : ℝ), a = 2) →
    |m| = Real.sqrt 3 :=
by 
  intros k m h
  sorry

end circle_line_chord_length_l1152_115259


namespace tetrahedron_inequality_l1152_115288

theorem tetrahedron_inequality
  (a b c d h_a h_b h_c h_d V : ℝ)
  (ha : V = 1/3 * a * h_a)
  (hb : V = 1/3 * b * h_b)
  (hc : V = 1/3 * c * h_c)
  (hd : V = 1/3 * d * h_d) :
  (a + b + c + d) * (h_a + h_b + h_c + h_d) >= 48 * V := 
  by sorry

end tetrahedron_inequality_l1152_115288


namespace chastity_lollipops_l1152_115294

theorem chastity_lollipops (initial_money lollipop_cost gummy_cost left_money total_gummies total_spent lollipops : ℝ)
  (h1 : initial_money = 15)
  (h2 : lollipop_cost = 1.50)
  (h3 : gummy_cost = 2)
  (h4 : left_money = 5)
  (h5 : total_gummies = 2)
  (h6 : total_spent = initial_money - left_money)
  (h7 : total_spent = 10)
  (h8 : total_gummies * gummy_cost = 4)
  (h9 : total_spent - (total_gummies * gummy_cost) = 6)
  (h10 : lollipops = (total_spent - (total_gummies * gummy_cost)) / lollipop_cost) :
  lollipops = 4 := 
sorry

end chastity_lollipops_l1152_115294


namespace max_min_diff_of_c_l1152_115233

-- Definitions and conditions
variables (a b c : ℝ)
def condition1 := a + b + c = 6
def condition2 := a^2 + b^2 + c^2 = 18

-- Theorem statement
theorem max_min_diff_of_c (h1 : condition1 a b c) (h2 : condition2 a b c) :
  ∃ (c_max c_min : ℝ), c_max = 6 ∧ c_min = -2 ∧ (c_max - c_min = 8) :=
by
  sorry

end max_min_diff_of_c_l1152_115233


namespace problem1_problem2_problem3_problem4_problem5_problem6_problem7_problem8_l1152_115261

-- Problem 1: 1 / 0.25 = 4
theorem problem1 : 1 / 0.25 = 4 :=
by sorry

-- Problem 2: 0.25 / 0.1 = 2.5
theorem problem2 : 0.25 / 0.1 = 2.5 :=
by sorry

-- Problem 3: 1.2 / 1.2 = 1
theorem problem3 : 1.2 / 1.2 = 1 :=
by sorry

-- Problem 4: 4.01 * 1 = 4.01
theorem problem4 : 4.01 * 1 = 4.01 :=
by sorry

-- Problem 5: 0.25 * 2 = 0.5
theorem problem5 : 0.25 * 2 = 0.5 :=
by sorry

-- Problem 6: 0 / 2.76 = 0
theorem problem6 : 0 / 2.76 = 0 :=
by sorry

-- Problem 7: 0.8 / 1.25 = 0.64
theorem problem7 : 0.8 / 1.25 = 0.64 :=
by sorry

-- Problem 8: 3.5 * 2.7 = 9.45
theorem problem8 : 3.5 * 2.7 = 9.45 :=
by sorry

end problem1_problem2_problem3_problem4_problem5_problem6_problem7_problem8_l1152_115261


namespace quadratic_cubic_expression_l1152_115270

theorem quadratic_cubic_expression
  (r s : ℝ)
  (h_eq : ∀ x : ℝ, 3 * x^2 - 4 * x - 12 = 0 → x = r ∨ x = s) :
  (9 * r^3 - 9 * s^3) / (r - s) = 52 :=
by 
  sorry

end quadratic_cubic_expression_l1152_115270


namespace complex_number_equality_l1152_115253

-- Define the conditions a, b ∈ ℝ and a + i = 1 - bi
theorem complex_number_equality (a b : ℝ) (i : ℂ) (h : a + i = 1 - b * i) : (a + b * i) ^ 8 = 16 :=
  sorry

end complex_number_equality_l1152_115253


namespace lowest_possible_price_l1152_115285

-- Definitions based on the provided conditions
def regular_discount_range : Set Real := {x | 0.10 ≤ x ∧ x ≤ 0.30}
def additional_discount : Real := 0.20
def retail_price : Real := 35.00

-- Problem statement transformed into Lean
theorem lowest_possible_price :
  ∃ d ∈ regular_discount_range, (retail_price * (1 - d)) * (1 - additional_discount) = 19.60 :=
by
  sorry

end lowest_possible_price_l1152_115285


namespace circle_equation_l1152_115208

theorem circle_equation (x y : ℝ) (h1 : (1 - 1)^2 + (1 - 1)^2 = 2) (h2 : (0 - 1)^2 + (0 - 1)^2 = r_sq) :
  (x - 1)^2 + (y - 1)^2 = 2 :=
sorry

end circle_equation_l1152_115208


namespace minimum_value_of_quadratic_l1152_115287

def quadratic_polynomial (x : ℝ) : ℝ := 2 * x^2 - 16 * x + 22

theorem minimum_value_of_quadratic : ∃ x : ℝ, quadratic_polynomial x = -10 :=
by 
  use 4
  { sorry }

end minimum_value_of_quadratic_l1152_115287


namespace outdoor_chairs_count_l1152_115250

theorem outdoor_chairs_count (indoor_tables outdoor_tables : ℕ) (chairs_per_indoor_table : ℕ) 
  (total_chairs : ℕ) (h1: indoor_tables = 9) (h2: outdoor_tables = 11) 
  (h3: chairs_per_indoor_table = 10) (h4: total_chairs = 123) : 
  (total_chairs - indoor_tables * chairs_per_indoor_table) / outdoor_tables = 3 :=
by 
  sorry

end outdoor_chairs_count_l1152_115250


namespace prime_mod_30_not_composite_l1152_115299

theorem prime_mod_30_not_composite (p : ℕ) (h_prime : Prime p) (h_gt_30 : p > 30) : 
  ¬ ∃ (x : ℕ), (x > 1 ∧ ∃ (a b : ℕ), x = a * b ∧ a > 1 ∧ b > 1) ∧ (0 < x ∧ x < 30 ∧ ∃ (k : ℕ), p = 30 * k + x) :=
by
  sorry

end prime_mod_30_not_composite_l1152_115299


namespace proof_problem_l1152_115210

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g (x : ℝ) : ℝ := (x + 1) * f x

axiom domain_f : ∀ x : ℝ, true
axiom even_f : ∀ x : ℝ, f (2 * x - 1) = f (-(2 * x - 1))
axiom mono_g_neg_inf_minus_1 : ∀ x y : ℝ, x ≤ y → x ≤ -1 → y ≤ -1 → g x ≤ g y

-- Proof Problem Statement
theorem proof_problem :
  (∀ x y : ℝ, x ≤ y → -1 ≤ x → -1 ≤ y → g x ≤ g y) ∧
  (∀ a b : ℝ, g a + g b > 0 → a + b + 2 > 0) :=
by
  sorry

end proof_problem_l1152_115210


namespace quadratic_unique_solution_l1152_115260

theorem quadratic_unique_solution (a c : ℝ) :
  (∀ x : ℝ, a * x^2 + 36 * x + c = 0 ↔ x = (-36) / (2*a))  -- The quadratic equation has exactly one solution
  → a + c = 37  -- Given condition
  → a < c      -- Given condition
  → (a, c) = ( (37 - Real.sqrt 73) / 2, (37 + Real.sqrt 73) / 2 ) :=  -- Correct answer
by
  sorry

end quadratic_unique_solution_l1152_115260


namespace final_sale_price_l1152_115224

def initial_price : ℝ := 450
def first_discount : ℝ := 0.25
def second_discount : ℝ := 0.10
def third_discount : ℝ := 0.05

def price_after_first_discount (initial : ℝ) (discount : ℝ) : ℝ :=
  initial * (1 - discount)
  
def price_after_second_discount (price_first : ℝ) (discount : ℝ) : ℝ :=
  price_first * (1 - discount)
  
def price_after_third_discount (price_second : ℝ) (discount : ℝ) : ℝ :=
  price_second * (1 - discount)

theorem final_sale_price :
  price_after_third_discount
    (price_after_second_discount
      (price_after_first_discount initial_price first_discount)
      second_discount)
    third_discount = 288.5625 := 
sorry

end final_sale_price_l1152_115224


namespace value_of_expression_l1152_115221

theorem value_of_expression : (2 + 4 + 6) - (1 + 3 + 5) = 3 := 
by 
  sorry

end value_of_expression_l1152_115221


namespace min_value_2a_3b_6c_l1152_115227

theorem min_value_2a_3b_6c (a b c : ℝ) (hpos_a : 0 < a) (hpos_b : 0 < b) (hpos_c : 0 < c) (habc : a * b * c = 27) :
  2 * a + 3 * b + 6 * c ≥ 27 :=
sorry

end min_value_2a_3b_6c_l1152_115227


namespace solve_for_x_l1152_115201

theorem solve_for_x (x : ℤ) (h : 3 * x - 7 = 11) : x = 6 :=
by
  sorry

end solve_for_x_l1152_115201


namespace bags_sold_in_afternoon_l1152_115229

theorem bags_sold_in_afternoon (bags_morning : ℕ) (weight_per_bag : ℕ) (total_weight : ℕ) 
  (h1 : bags_morning = 29) (h2 : weight_per_bag = 7) (h3 : total_weight = 322) : 
  total_weight - bags_morning * weight_per_bag / weight_per_bag = 17 := 
by 
  sorry

end bags_sold_in_afternoon_l1152_115229


namespace find_triangle_angles_l1152_115242

theorem find_triangle_angles (a b h_a h_b : ℝ) (A B C : ℝ) :
  a ≤ h_a → b ≤ h_b →
  h_a ≤ b → h_b ≤ a →
  ∃ x y z : ℝ, (x = 90 ∧ y = 45 ∧ z = 45) ∧ 
  (x + y + z = 180) :=
by
  sorry

end find_triangle_angles_l1152_115242


namespace students_not_pass_l1152_115289

theorem students_not_pass (total_students : ℕ) (percentage_passed : ℕ) (students_passed : ℕ) (students_not_passed : ℕ) :
  total_students = 804 →
  percentage_passed = 75 →
  students_passed = total_students * percentage_passed / 100 →
  students_not_passed = total_students - students_passed →
  students_not_passed = 201 :=
by
  intros h1 h2 h3 h4
  sorry

end students_not_pass_l1152_115289


namespace find_expression_value_l1152_115297

theorem find_expression_value (x y : ℚ) (h₁ : 3 * x + y = 6) (h₂ : x + 3 * y = 8) :
  9 * x ^ 2 + 15 * x * y + 9 * y ^ 2 = 1629 / 16 := 
sorry

end find_expression_value_l1152_115297


namespace predicted_temperature_l1152_115245

-- Define the observation data points
def data_points : List (ℕ × ℝ) :=
  [(20, 25), (30, 27.5), (40, 29), (50, 32.5), (60, 36)]

-- Define the linear regression equation with constant k
def regression (x : ℕ) (k : ℝ) : ℝ :=
  0.25 * x + k

-- Proof statement
theorem predicted_temperature (k : ℝ) (h : regression 40 k = 30) : regression 80 k = 40 :=
by
  sorry

end predicted_temperature_l1152_115245


namespace factor_expression_l1152_115205

theorem factor_expression (a b c : ℝ) :
  ((a^3 - b^3)^3 + (b^3 - c^3)^3 + (c^3 - a^3)^3) / ((a - b)^3 + (b - c)^3 + (c - a)^3) =
  (a^2 + a * b + b^2) * (b^2 + b * c + c^2) * (c^2 + c * a + a^2) :=
by
  sorry

end factor_expression_l1152_115205


namespace fraction_value_l1152_115268

theorem fraction_value
  (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (cond1 : (a + b + c) / (a + b - c) = 7)
  (cond2 : (a + b + c) / (a + c - b) = 1.75) :
  (a + b + c) / (b + c - a) = 3.5 :=
by
  sorry

end fraction_value_l1152_115268


namespace integral_of_reciprocal_l1152_115264

theorem integral_of_reciprocal (a b : ℝ) (h_eq : a = 1) (h_eb : b = Real.exp 1) : ∫ x in a..b, 1/x = 1 :=
by 
  rw [h_eq, h_eb]
  sorry

end integral_of_reciprocal_l1152_115264


namespace find_x_l1152_115278

def vector := (ℝ × ℝ)

def collinear (u v : vector) : Prop :=
  ∃ k : ℝ, u = (k * v.1, k * v.2)

def a : vector := (1, 2)
def b (x : ℝ) : vector := (x, 1)
def a_minus_b (x : ℝ) : vector := ((1 - x), 1)

theorem find_x (x : ℝ) (h : collinear a (a_minus_b x)) : x = 1/2 :=
by
  sorry

end find_x_l1152_115278


namespace prime_solution_l1152_115219

theorem prime_solution (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (h : 5 * p + 3 * q = 91) : p = 17 ∧ q = 2 :=
by
  sorry

end prime_solution_l1152_115219


namespace coefficient_of_x4_l1152_115206

theorem coefficient_of_x4 (n : ℕ) (f : ℕ → ℕ → ℝ)
  (h1 : (2 : ℕ) ^ n = 256) :
  (f 8 4) * (2 : ℕ) ^ 4 = 1120 :=
by
  sorry

end coefficient_of_x4_l1152_115206


namespace graph_does_not_pass_second_quadrant_l1152_115247

noncomputable def y_function (a b : ℝ) (x : ℝ) : ℝ := a^x + b

theorem graph_does_not_pass_second_quadrant (a b : ℝ) (h1 : a > 1) (h2 : b < -1) : 
  ∀ x y : ℝ, (y = y_function a b x) → ¬(x < 0 ∧ y > 0) := by
  sorry

end graph_does_not_pass_second_quadrant_l1152_115247


namespace fraction_of_dark_tiles_is_correct_l1152_115277

def num_tiles_in_block : ℕ := 64
def num_dark_tiles : ℕ := 18
def expected_fraction_dark_tiles : ℚ := 9 / 32

theorem fraction_of_dark_tiles_is_correct :
  (num_dark_tiles : ℚ) / num_tiles_in_block = expected_fraction_dark_tiles := by
sorry

end fraction_of_dark_tiles_is_correct_l1152_115277


namespace eq_g_of_f_l1152_115213

def f (x : ℝ) : ℝ := 3 * x - 5
def g (x : ℝ) : ℝ := 6 * x - 29

theorem eq_g_of_f (x : ℝ) : 2 * (f x) - 19 = g x :=
by 
  sorry

end eq_g_of_f_l1152_115213


namespace constant_S13_l1152_115272

noncomputable def S (a d : ℝ) (n : ℕ) : ℝ := n * (2 * a + (n - 1) * d) / 2

theorem constant_S13 (a d p : ℝ) 
  (h : a + a + 3 * d + a + 7 * d = p) : 
  S a d 13 = 13 * p / 18 :=
by
  unfold S
  sorry

end constant_S13_l1152_115272


namespace border_material_correct_l1152_115235

noncomputable def pi_approx := (22 : ℚ) / 7

def circle_radius (area : ℚ) (pi_value : ℚ) : ℚ :=
  (area * (7 / 22)).sqrt

def circumference (radius : ℚ) (pi_value : ℚ) : ℚ :=
  2 * pi_value * radius

def total_border_material (area : ℚ) (pi_value : ℚ) (extra : ℚ) : ℚ :=
  circumference (circle_radius area pi_value) pi_value + extra

theorem border_material_correct :
  total_border_material 616 pi_approx 3 = 91 :=
by
  sorry

end border_material_correct_l1152_115235


namespace binomial_coefficient_ratio_l1152_115207

theorem binomial_coefficient_ratio (n k : ℕ) (h₁ : n = 4 * k + 3) (h₂ : n = 3 * k + 5) : n + k = 13 :=
by
  sorry

end binomial_coefficient_ratio_l1152_115207


namespace quadratic_rewrite_constants_l1152_115231

theorem quadratic_rewrite_constants (a b c : ℤ) 
    (h1 : -4 * (x - 2) ^ 2 + 144 = -4 * x ^ 2 + 16 * x + 128) 
    (h2 : a = -4)
    (h3 : b = -2)
    (h4 : c = 144) 
    : a + b + c = 138 := by
  sorry

end quadratic_rewrite_constants_l1152_115231


namespace possible_polynomials_l1152_115241

noncomputable def f (x : ℝ) : ℝ := x^2

theorem possible_polynomials (g : ℝ → ℝ) :
  (∀ x, f (g x) = 9 * x^2 - 6 * x + 1) → 
  (∀ x, (g x = 3 * x - 1) ∨ (g x = -(3 * x - 1))) := 
by
  intros h x
  sorry

end possible_polynomials_l1152_115241


namespace goods_train_speed_l1152_115228

theorem goods_train_speed (length_train length_platform distance time : ℕ) (conversion_factor : ℚ) : 
  length_train = 250 → 
  length_platform = 270 → 
  distance = length_train + length_platform → 
  time = 26 → 
  conversion_factor = 3.6 →
  (distance / time : ℚ) * conversion_factor = 72 :=
by
  intros h_lt h_lp h_d h_t h_cf
  rw [h_lt, h_lp] at h_d
  rw [h_t, h_cf]
  sorry

end goods_train_speed_l1152_115228


namespace work_days_l1152_115238

theorem work_days (m r d : ℕ) (h : 2 * m * d = 2 * (m + r) * (md / (m + r))) : d = md / (m + r) :=
by
  sorry

end work_days_l1152_115238


namespace negation_statement_l1152_115217

theorem negation_statement (x : ℝ) (hx0 : x ≠ 0) (hx1 : x ≠ 1) : x^2 - x ≠ 0 :=
by sorry

end negation_statement_l1152_115217


namespace sum_of_squares_l1152_115265

def b1 : ℚ := 10 / 32
def b2 : ℚ := 0
def b3 : ℚ := -5 / 32
def b4 : ℚ := 0
def b5 : ℚ := 1 / 32

theorem sum_of_squares : b1^2 + b2^2 + b3^2 + b4^2 + b5^2 = 63 / 512 :=
by
  sorry

end sum_of_squares_l1152_115265
