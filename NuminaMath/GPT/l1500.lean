import Mathlib

namespace NUMINAMATH_GPT_probability_A_given_B_l1500_150002

def roll_outcomes : ℕ := 6^3 -- Total number of possible outcomes when rolling three dice

def P_AB : ℚ := 60 / 216 -- Probability of both events A and B happening

def P_B : ℚ := 91 / 216 -- Probability of event B happening

theorem probability_A_given_B : (P_AB / P_B) = (60 / 91) := by
  sorry

end NUMINAMATH_GPT_probability_A_given_B_l1500_150002


namespace NUMINAMATH_GPT_geo_seq_a12_equal_96_l1500_150057

def is_geometric (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

theorem geo_seq_a12_equal_96
  (a : ℕ → ℝ) (q : ℝ)
  (h0 : 1 < q)
  (h1 : is_geometric a q)
  (h2 : a 3 * a 7 = 72)
  (h3 : a 2 + a 8 = 27) :
  a 12 = 96 :=
sorry

end NUMINAMATH_GPT_geo_seq_a12_equal_96_l1500_150057


namespace NUMINAMATH_GPT_quadratic_root_condition_l1500_150068

theorem quadratic_root_condition (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1 > 1 ∧ x2 < 1 ∧ x1^2 + 2*a*x1 + 1 = 0 ∧ x2^2 + 2*a*x2 + 1 = 0) →
  a < -1 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_root_condition_l1500_150068


namespace NUMINAMATH_GPT_riverview_problem_l1500_150029

theorem riverview_problem (h c : Nat) (p : Nat := 4 * h) (s : Nat := 5 * c) (d : Nat := 4 * p) :
  (p + h + s + c + d = 52 → false) :=
by {
  sorry
}

end NUMINAMATH_GPT_riverview_problem_l1500_150029


namespace NUMINAMATH_GPT_constant_term_in_expansion_l1500_150069

theorem constant_term_in_expansion (x : ℂ) : 
  (2 - (3 / x)) * (x ^ 2 + 2 / x) ^ 5 = 0 := 
sorry

end NUMINAMATH_GPT_constant_term_in_expansion_l1500_150069


namespace NUMINAMATH_GPT_bob_corn_stalks_per_row_l1500_150087

noncomputable def corn_stalks_per_row
  (rows : ℕ)
  (bushels : ℕ)
  (stalks_per_bushel : ℕ) :
  ℕ :=
  (bushels * stalks_per_bushel) / rows

theorem bob_corn_stalks_per_row
  (rows : ℕ)
  (bushels : ℕ)
  (stalks_per_bushel : ℕ) :
  rows = 5 → bushels = 50 → stalks_per_bushel = 8 → corn_stalks_per_row rows bushels stalks_per_bushel = 80 :=
by
  intros h1 h2 h3
  subst h1
  subst h2
  subst h3
  unfold corn_stalks_per_row
  rfl

end NUMINAMATH_GPT_bob_corn_stalks_per_row_l1500_150087


namespace NUMINAMATH_GPT_initial_investment_l1500_150021

noncomputable def compound_interest (P : ℝ) (r : ℝ) (n t : ℕ) : ℝ :=
  P * (1 + r / n)^(n * t)

theorem initial_investment (A : ℝ) (r : ℝ) (n t : ℕ) (P : ℝ) :
  A = 3630.0000000000005 → r = 0.10 → n = 1 → t = 2 → P = 3000 →
  A = compound_interest P r n t :=
by
  intros hA hr hn ht hP
  rw [compound_interest, hA, hr, hP]
  sorry

end NUMINAMATH_GPT_initial_investment_l1500_150021


namespace NUMINAMATH_GPT_triangle_shape_l1500_150028

-- Defining the conditions:
variables (A B C a b c : ℝ)
variable (h1 : c - a * Real.cos B = (2 * a - b) * Real.cos A)

-- Defining the property to prove:
theorem triangle_shape : 
  (A = Real.pi / 2 ∨ A = B ∨ B = C ∨ C = A + B) :=
sorry

end NUMINAMATH_GPT_triangle_shape_l1500_150028


namespace NUMINAMATH_GPT_length_of_faster_train_l1500_150091

theorem length_of_faster_train (speed_faster_train : ℝ) (speed_slower_train : ℝ) (elapsed_time : ℝ) (relative_speed : ℝ) (length_train : ℝ)
  (h1 : speed_faster_train = 50) 
  (h2 : speed_slower_train = 32) 
  (h3 : elapsed_time = 15) 
  (h4 : relative_speed = (speed_faster_train - speed_slower_train) * (1000 / 3600)) 
  (h5 : length_train = relative_speed * elapsed_time) :
  length_train = 75 :=
sorry

end NUMINAMATH_GPT_length_of_faster_train_l1500_150091


namespace NUMINAMATH_GPT_spherical_to_rectangular_correct_l1500_150099

noncomputable def spherical_to_rectangular (ρ θ φ : ℝ) : ℝ × ℝ × ℝ :=
  (ρ * Real.sin φ * Real.cos θ, ρ * Real.sin φ * Real.sin θ, ρ * Real.cos φ)

theorem spherical_to_rectangular_correct :
  spherical_to_rectangular 3 (Real.pi / 2) (Real.pi / 3) = (0, (3 * Real.sqrt 3) / 2, 3 / 2) :=
by
  sorry

end NUMINAMATH_GPT_spherical_to_rectangular_correct_l1500_150099


namespace NUMINAMATH_GPT_maximum_marks_l1500_150013

-- Definitions based on the conditions
def passing_percentage : ℝ := 0.5
def student_marks : ℝ := 200
def marks_to_pass : ℝ := student_marks + 20

-- Lean 4 statement for the proof problem
theorem maximum_marks (M : ℝ) 
  (h1 : marks_to_pass = 220)
  (h2 : passing_percentage * M = marks_to_pass) :
  M = 440 :=
sorry

end NUMINAMATH_GPT_maximum_marks_l1500_150013


namespace NUMINAMATH_GPT_complement_of_angle_l1500_150052

def complement_angle (deg : ℕ) (min : ℕ) : ℕ × ℕ :=
  if deg < 90 then 
    let total_min := (90 * 60)
    let angle_min := (deg * 60) + min
    let comp_min := total_min - angle_min
    (comp_min / 60, comp_min % 60) -- degrees and remaining minutes
  else 
    (0, 0) -- this case handles if the angle is not less than complement allowable range

-- Definitions based on the problem
def given_angle_deg : ℕ := 57
def given_angle_min : ℕ := 13

-- Complement calculation
def comp (deg : ℕ) (min : ℕ) : ℕ × ℕ := complement_angle deg min

-- Expected result of the complement
def expected_comp : ℕ × ℕ := (32, 47)

-- Theorem to prove the complement of 57°13' is 32°47'
theorem complement_of_angle : comp given_angle_deg given_angle_min = expected_comp := by
  sorry

end NUMINAMATH_GPT_complement_of_angle_l1500_150052


namespace NUMINAMATH_GPT_part1_part2_part3_l1500_150041

-- Part 1: Simplifying the Expression
theorem part1 (a b : ℝ) : 
  3 * (a - b)^2 - 6 * (a - b)^2 + 2 * (a - b)^2 = -(a - b)^2 :=
by sorry

-- Part 2: Finding the Value of an Expression
theorem part2 (x y : ℝ) (h : x^2 - 2 * y = 4) : 
  3 * x^2 - 6 * y - 21 = -9 :=
by sorry

-- Part 3: Evaluating a Compound Expression
theorem part3 (a b c d : ℝ) (h1 : a - 2 * b = 6) (h2 : 2 * b - c = -8) (h3 : c - d = 9) : 
  (a - c) + (2 * b - d) - (2 * b - c) = 7 :=
by sorry

end NUMINAMATH_GPT_part1_part2_part3_l1500_150041


namespace NUMINAMATH_GPT_ratio_out_of_state_to_in_state_l1500_150050

/-
Given:
- total job applications Carly sent is 600
- job applications sent to companies in her state is 200

Prove:
- The ratio of job applications sent to companies in other states to the number sent to companies in her state is 2:1.
-/

def total_applications : ℕ := 600
def in_state_applications : ℕ := 200
def out_of_state_applications : ℕ := total_applications - in_state_applications

theorem ratio_out_of_state_to_in_state :
  (out_of_state_applications / in_state_applications) = 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_out_of_state_to_in_state_l1500_150050


namespace NUMINAMATH_GPT_calculate_purple_pants_l1500_150096

def total_shirts : ℕ := 5
def total_pants : ℕ := 24
def plaid_shirts : ℕ := 3
def non_plaid_non_purple_items : ℕ := 21

theorem calculate_purple_pants : total_pants - (non_plaid_non_purple_items - (total_shirts - plaid_shirts)) = 5 :=
by 
  sorry

end NUMINAMATH_GPT_calculate_purple_pants_l1500_150096


namespace NUMINAMATH_GPT_greatest_integer_b_l1500_150058

theorem greatest_integer_b (b : ℤ) : (∀ x : ℝ, x^2 + (b : ℝ) * x + 7 ≠ 0) → b ≤ 5 :=
by sorry

end NUMINAMATH_GPT_greatest_integer_b_l1500_150058


namespace NUMINAMATH_GPT_initial_number_of_men_l1500_150039

variable (M : ℕ) (A : ℕ)
variable (change_in_age: ℕ := 16)
variable (age_increment: ℕ := 2)

theorem initial_number_of_men :
  ((A + age_increment) * M = A * M + change_in_age) → M = 8 :=
by
  intros h_1
  sorry

end NUMINAMATH_GPT_initial_number_of_men_l1500_150039


namespace NUMINAMATH_GPT_find_f_600_l1500_150008

variable (f : ℝ → ℝ)
variable (h1 : ∀ (x y : ℝ), x > 0 → y > 0 → f (x * y) = f x / y)
variable (h2 : f 500 = 3)

theorem find_f_600 : f 600 = 5 / 2 :=
by
  sorry

end NUMINAMATH_GPT_find_f_600_l1500_150008


namespace NUMINAMATH_GPT_theater_ticket_sales_l1500_150059

theorem theater_ticket_sales (A K : ℕ) (h1 : A + K = 275) (h2 :  12 * A + 5 * K = 2150) : K = 164 := by
  sorry

end NUMINAMATH_GPT_theater_ticket_sales_l1500_150059


namespace NUMINAMATH_GPT_xiao_ming_math_score_l1500_150038

noncomputable def math_score (C M E : ℕ) : ℕ :=
  let A := 94
  let N := 3
  let total_score := A * N
  let T_CE := (A - 1) * 2
  total_score - T_CE

theorem xiao_ming_math_score (C M E : ℕ)
    (h1 : (C + M + E) / 3 = 94)
    (h2 : (C + E) / 2 = 93) :
  math_score C M E = 96 := by
  sorry

end NUMINAMATH_GPT_xiao_ming_math_score_l1500_150038


namespace NUMINAMATH_GPT_pam_walked_1683_miles_l1500_150074

noncomputable def pam_miles_walked 
    (pedometer_limit : ℕ)
    (initial_reading : ℕ)
    (flips : ℕ)
    (final_reading : ℕ)
    (steps_per_mile : ℕ)
    : ℕ :=
  (pedometer_limit + 1) * flips + final_reading / steps_per_mile

theorem pam_walked_1683_miles
    (pedometer_limit : ℕ := 49999)
    (initial_reading : ℕ := 0)
    (flips : ℕ := 50)
    (final_reading : ℕ := 25000)
    (steps_per_mile : ℕ := 1500) 
    : pam_miles_walked pedometer_limit initial_reading flips final_reading steps_per_mile = 1683 := 
  sorry

end NUMINAMATH_GPT_pam_walked_1683_miles_l1500_150074


namespace NUMINAMATH_GPT_area_half_l1500_150055

theorem area_half (width height : ℝ) (h₁ : width = 25) (h₂ : height = 16) :
  (width * height) / 2 = 200 :=
by
  -- The formal proof is skipped here
  sorry

end NUMINAMATH_GPT_area_half_l1500_150055


namespace NUMINAMATH_GPT_total_cost_of_refueling_l1500_150049

theorem total_cost_of_refueling 
  (smaller_tank_capacity : ℤ)
  (larger_tank_capacity : ℤ)
  (num_smaller_planes : ℤ)
  (num_larger_planes : ℤ)
  (fuel_cost_per_liter : ℤ)
  (service_charge_per_plane : ℤ)
  (total_cost : ℤ) :
  smaller_tank_capacity = 60 →
  larger_tank_capacity = 90 →
  num_smaller_planes = 2 →
  num_larger_planes = 2 →
  fuel_cost_per_liter = 50 →
  service_charge_per_plane = 100 →
  total_cost = (num_smaller_planes * smaller_tank_capacity + num_larger_planes * larger_tank_capacity) * (fuel_cost_per_liter / 100) + (num_smaller_planes + num_larger_planes) * service_charge_per_plane →
  total_cost = 550 :=
by
  intros
  sorry

end NUMINAMATH_GPT_total_cost_of_refueling_l1500_150049


namespace NUMINAMATH_GPT_percentage_supports_policy_l1500_150044

theorem percentage_supports_policy (men women : ℕ) (men_favor women_favor : ℝ) (total_population : ℕ) (total_supporters : ℕ) (percentage_supporters : ℝ)
  (h1 : men = 200) 
  (h2 : women = 800)
  (h3 : men_favor = 0.70)
  (h4 : women_favor = 0.75)
  (h5 : total_population = men + women)
  (h6 : total_supporters = (men_favor * men) + (women_favor * women))
  (h7 : percentage_supporters = (total_supporters / total_population) * 100) :
  percentage_supporters = 74 := 
by
  sorry

end NUMINAMATH_GPT_percentage_supports_policy_l1500_150044


namespace NUMINAMATH_GPT_private_schools_in_district_B_l1500_150030

theorem private_schools_in_district_B :
  let total_schools := 50
  let public_schools := 25
  let parochial_schools := 16
  let private_schools := 9
  let district_A_schools := 18
  let district_B_schools := 17
  let district_C_schools := total_schools - district_A_schools - district_B_schools
  let schools_per_kind_in_C := district_C_schools / 3
  let private_schools_in_C := schools_per_kind_in_C
  let remaining_private_schools := private_schools - private_schools_in_C
  remaining_private_schools = 4 :=
by
  let total_schools := 50
  let public_schools := 25
  let parochial_schools := 16
  let private_schools := 9
  let district_A_schools := 18
  let district_B_schools := 17
  let district_C_schools := total_schools - district_A_schools - district_B_schools
  let schools_per_kind_in_C := district_C_schools / 3
  let private_schools_in_C := schools_per_kind_in_C
  let remaining_private_schools := private_schools - private_schools_in_C
  sorry

end NUMINAMATH_GPT_private_schools_in_district_B_l1500_150030


namespace NUMINAMATH_GPT_largest_tile_side_length_l1500_150026

theorem largest_tile_side_length (w l : ℕ) (hw : w = 120) (hl : l = 96) : 
  ∃ s, s = Nat.gcd w l ∧ s = 24 :=
by
  sorry

end NUMINAMATH_GPT_largest_tile_side_length_l1500_150026


namespace NUMINAMATH_GPT_solve_eq1_solve_eq2_l1500_150089

-- For Equation (1)
theorem solve_eq1 (x : ℝ) : x^2 - 4*x - 6 = 0 → x = 2 + Real.sqrt 10 ∨ x = 2 - Real.sqrt 10 :=
sorry

-- For Equation (2)
theorem solve_eq2 (x : ℝ) : (x / (x - 1) - 1 = 3 / (x^2 - 1)) → x ≠ 1 ∧ x ≠ -1 → x = 2 :=
sorry

end NUMINAMATH_GPT_solve_eq1_solve_eq2_l1500_150089


namespace NUMINAMATH_GPT_division_problem_l1500_150006

theorem division_problem : (4 * 5) / 10 = 2 :=
by sorry

end NUMINAMATH_GPT_division_problem_l1500_150006


namespace NUMINAMATH_GPT_tiling_implies_divisibility_l1500_150079

def is_divisible_by (a b : Nat) : Prop := ∃ k : Nat, a = k * b

noncomputable def can_be_tiled (m n a b : Nat) : Prop :=
  a * b > 0 ∧ -- positivity condition for rectangle dimensions
  (∃ f_horiz : Fin (a * b) → Fin m, 
   ∃ g_vert : Fin (a * b) → Fin n, 
   True) -- A placeholder to denote tiling condition.

theorem tiling_implies_divisibility (m n a b : Nat)
  (hmn_pos : 0 < m ∧ 0 < n ∧ 0 < a ∧ 0 < b)
  (h_tiling : can_be_tiled m n a b) :
  is_divisible_by a m ∨ is_divisible_by b n :=
by
  sorry

end NUMINAMATH_GPT_tiling_implies_divisibility_l1500_150079


namespace NUMINAMATH_GPT_diagonal_plane_angle_l1500_150098

theorem diagonal_plane_angle
  (α : Real)
  (a : Real)
  (plane_square_angle_with_plane : Real)
  (diagonal_plane_angle : Real) 
  (h1 : plane_square_angle_with_plane = α) :
  diagonal_plane_angle = Real.arcsin (Real.sin α / Real.sqrt 2) :=
sorry

end NUMINAMATH_GPT_diagonal_plane_angle_l1500_150098


namespace NUMINAMATH_GPT_needle_intersection_probability_l1500_150075

noncomputable def needle_probability (a l : ℝ) (h : l < a) : ℝ :=
  (2 * l) / (a * Real.pi)

theorem needle_intersection_probability (a l : ℝ) (h : l < a) :
  needle_probability a l h = 2 * l / (a * Real.pi) :=
by
  -- This is the statement to be proved
  sorry

end NUMINAMATH_GPT_needle_intersection_probability_l1500_150075


namespace NUMINAMATH_GPT_prob_contact_l1500_150090

variables (p : ℝ)
def prob_no_contact : ℝ := (1 - p) ^ 40

theorem prob_contact : 1 - prob_no_contact p = 1 - (1 - p) ^ 40 := by
  sorry

end NUMINAMATH_GPT_prob_contact_l1500_150090


namespace NUMINAMATH_GPT_inequality_solution_sum_of_squares_geq_sum_of_products_l1500_150032

-- Problem 1
theorem inequality_solution (x : ℝ) : (0 < x ∧ x < 2/3) ↔ (x + 2) / (2 - 3 * x) > 1 :=
by
  sorry

-- Problem 2
theorem sum_of_squares_geq_sum_of_products (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  a^2 + b^2 + c^2 ≥ a * b + b * c + c * a :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_sum_of_squares_geq_sum_of_products_l1500_150032


namespace NUMINAMATH_GPT_original_monthly_bill_l1500_150022

-- Define the necessary conditions
def increased_bill (original: ℝ): ℝ := original + 0.3 * original
def total_bill_after_increase : ℝ := 78

-- The proof we need to construct
theorem original_monthly_bill (X : ℝ) (H : increased_bill X = total_bill_after_increase) : X = 60 :=
by {
    sorry -- Proof is not required, only statement
}

end NUMINAMATH_GPT_original_monthly_bill_l1500_150022


namespace NUMINAMATH_GPT_coordinates_of_A_equidistant_BC_l1500_150070

theorem coordinates_of_A_equidistant_BC :
  ∃ z : ℚ, (∀ A B C : ℚ × ℚ × ℚ, A = (0, 0, z) ∧ B = (7, 0, -15) ∧ C = (2, 10, -12) →
  (dist A B = dist A C)) ↔ z = -(13/3) :=
by sorry

end NUMINAMATH_GPT_coordinates_of_A_equidistant_BC_l1500_150070


namespace NUMINAMATH_GPT_quadratic_inequality_solution_l1500_150080

theorem quadratic_inequality_solution (x : ℝ) : (x^2 - 4 * x - 21 < 0) ↔ (-3 < x ∧ x < 7) :=
sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_l1500_150080


namespace NUMINAMATH_GPT_min_value_expression_l1500_150072

noncomputable section

variables {x y : ℝ}

theorem min_value_expression (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_sum : x + y = 1) :
  (∀ x y : ℝ, 0 < x ∧ 0 < y ∧ x + y = 1 ∧ 
    (∃ min_val : ℝ, min_val = (x^2 / (x + 2) + y^2 / (y + 1)) ∧ min_val = 1 / 4)) :=
  sorry

end NUMINAMATH_GPT_min_value_expression_l1500_150072


namespace NUMINAMATH_GPT_volume_of_right_prism_with_trapezoid_base_l1500_150016

variable (S1 S2 H a b h: ℝ)

theorem volume_of_right_prism_with_trapezoid_base 
  (hS1 : S1 = a * H) 
  (hS2 : S2 = b * H) 
  (h_trapezoid : a ≠ b) : 
  1 / 2 * (S1 + S2) * h = (1 / 2 * (a + b) * h) * H :=
by 
  sorry

end NUMINAMATH_GPT_volume_of_right_prism_with_trapezoid_base_l1500_150016


namespace NUMINAMATH_GPT_jose_birds_left_l1500_150020

-- Define initial conditions
def chickens_initial : Nat := 28
def ducks : Nat := 18
def turkeys : Nat := 15
def chickens_sold : Nat := 12

-- Calculate remaining chickens
def chickens_left : Nat := chickens_initial - chickens_sold

-- Calculate total birds left
def total_birds_left : Nat := chickens_left + ducks + turkeys

-- Theorem statement to prove the number of birds left
theorem jose_birds_left : total_birds_left = 49 :=
by
  -- This is where the proof would typically go
  sorry

end NUMINAMATH_GPT_jose_birds_left_l1500_150020


namespace NUMINAMATH_GPT_problem1_l1500_150073

theorem problem1 (m n : ℕ) (h1 : 3 ^ m = 4) (h2 : 3 ^ (m + 4 * n) = 324) : 2016 ^ n = 2016 := 
by 
  sorry

end NUMINAMATH_GPT_problem1_l1500_150073


namespace NUMINAMATH_GPT_tomatoes_harvest_ratio_l1500_150088

noncomputable def tomatoes_ratio (w t f : ℕ) (g r : ℕ) : ℕ × ℕ :=
  if (w = 400) ∧ ((w + t + f) = 2000) ∧ ((g = 700) ∧ (r = 700) ∧ ((g + r) = f)) ∧ (t = 200) then 
    (2, 1)
  else 
    sorry

theorem tomatoes_harvest_ratio : 
  ∀ (w t f : ℕ) (g r : ℕ), 
  (w = 400) → 
  (w + t + f = 2000) → 
  (g = 700) → 
  (r = 700) → 
  (g + r = f) → 
  (t = 200) →
  tomatoes_ratio w t f g r = (2, 1) :=
by {
  -- insert proof here
  sorry
}

end NUMINAMATH_GPT_tomatoes_harvest_ratio_l1500_150088


namespace NUMINAMATH_GPT_solve_for_x_l1500_150053

theorem solve_for_x (x : ℕ) (h : 5 * (2 ^ x) = 320) : x = 6 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l1500_150053


namespace NUMINAMATH_GPT_positive_integer_divisibility_l1500_150036

theorem positive_integer_divisibility (n : ℕ) (h_pos : n > 0) (h_div : (n^2 + 1) ∣ (n + 1)) : n = 1 := 
sorry

end NUMINAMATH_GPT_positive_integer_divisibility_l1500_150036


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_l1500_150063

theorem necessary_but_not_sufficient (x : ℝ) : (x > -1) ↔ (∀ y : ℝ, (2 * y > 2) → (-1 < y)) :=
sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_l1500_150063


namespace NUMINAMATH_GPT_appointment_on_tuesday_duration_l1500_150093

theorem appointment_on_tuesday_duration :
  let rate := 20
  let monday_appointments := 5
  let monday_each_duration := 1.5
  let thursday_appointments := 2
  let thursday_each_duration := 2
  let saturday_duration := 6
  let weekly_earnings := 410
  let known_earnings := (monday_appointments * monday_each_duration * rate) + (thursday_appointments * thursday_each_duration * rate) + (saturday_duration * rate)
  let tuesday_earnings := weekly_earnings - known_earnings
  (tuesday_earnings / rate = 3) :=
by
  -- let rate := 20
  -- let monday_appointments := 5
  -- let monday_each_duration := 1.5
  -- let thursday_appointments := 2
  -- let thursday_each_duration := 2
  -- let saturday_duration := 6
  -- let weekly_earnings := 410
  -- let known_earnings := (monday_appointments * monday_each_duration * rate) + (thursday_appointments * thursday_each_duration * rate) + (saturday_duration * rate)
  -- let tuesday_earnings := weekly_earnings - known_earnings
  -- exact tuesday_earnings / rate = 3
  sorry

end NUMINAMATH_GPT_appointment_on_tuesday_duration_l1500_150093


namespace NUMINAMATH_GPT_helpers_cakes_l1500_150077

theorem helpers_cakes (S : ℕ) (helpers large_cakes small_cakes : ℕ)
  (h1 : helpers = 10)
  (h2 : large_cakes = 2)
  (h3 : small_cakes = 700)
  (h4 : 1 * helpers * large_cakes = 20)
  (h5 : 2 * helpers * S = small_cakes) :
  S = 35 :=
by
  sorry

end NUMINAMATH_GPT_helpers_cakes_l1500_150077


namespace NUMINAMATH_GPT_fair_attendance_l1500_150060

theorem fair_attendance :
  let this_year := 600
  let next_year := 2 * this_year
  let total_people := 2800
  let last_year := total_people - this_year - next_year
  (1200 - last_year = 200) ∧ (last_year = 1000) := by
  sorry

end NUMINAMATH_GPT_fair_attendance_l1500_150060


namespace NUMINAMATH_GPT_find_angle_A_range_area_of_triangle_l1500_150027

variables {A B C : ℝ}
variables {a b c : ℝ}
variables {S : ℝ}

theorem find_angle_A (h1 : b^2 + c^2 = a^2 - b * c) : A = (2 : ℝ) * Real.pi / 3 :=
by sorry

theorem range_area_of_triangle (h1 : b^2 + c^2 = a^2 - b * c)
(h2 : b * Real.sin A = 4 * Real.sin B) 
(h3 : Real.log b + Real.log c ≥ 1 - 2 * Real.cos (B + C)) 
(h4 : A = (2 : ℝ) * Real.pi / 3) :
(Real.sqrt 3 / 4 : ℝ) ≤ (1 / 2) * b * c * Real.sin A ∧
(1 / 2) * b * c * Real.sin A ≤ (4 * Real.sqrt 3 / 3 : ℝ) :=
by sorry

end NUMINAMATH_GPT_find_angle_A_range_area_of_triangle_l1500_150027


namespace NUMINAMATH_GPT_train_length_eq_l1500_150065

theorem train_length_eq 
  (speed_kmh : ℝ) (time_sec : ℝ) 
  (h_speed_kmh : speed_kmh = 126)
  (h_time_sec : time_sec = 6.856594329596489) : 
  ((speed_kmh * 1000 / 3600) * time_sec) = 239.9808045358781 :=
by
  -- We skip the proof with sorry, as per instructions
  sorry

end NUMINAMATH_GPT_train_length_eq_l1500_150065


namespace NUMINAMATH_GPT_one_over_x_plus_one_over_y_eq_fifteen_l1500_150083

theorem one_over_x_plus_one_over_y_eq_fifteen
  (x y : ℝ)
  (h1 : xy > 0)
  (h2 : 1 / xy = 5)
  (h3 : (x + y) / 5 = 0.6) : 
  (1 / x) + (1 / y) = 15 := 
by
  sorry

end NUMINAMATH_GPT_one_over_x_plus_one_over_y_eq_fifteen_l1500_150083


namespace NUMINAMATH_GPT_sin_6_cos_6_theta_proof_l1500_150000

noncomputable def sin_6_cos_6_theta (θ : ℝ) (h : Real.cos (2 * θ) = 1 / 4) : ℝ :=
  Real.sin θ ^ 6 + Real.cos θ ^ 6

theorem sin_6_cos_6_theta_proof (θ : ℝ) (h : Real.cos (2 * θ) = 1 / 4) : 
  sin_6_cos_6_theta θ h = 19 / 64 :=
by
  sorry

end NUMINAMATH_GPT_sin_6_cos_6_theta_proof_l1500_150000


namespace NUMINAMATH_GPT_move_point_right_l1500_150097

theorem move_point_right 
  (x y : ℤ)
  (h : (x, y) = (2, -1)) :
  (x + 3, y) = (5, -1) := 
by
  sorry

end NUMINAMATH_GPT_move_point_right_l1500_150097


namespace NUMINAMATH_GPT_triangle_equilateral_l1500_150076

noncomputable def is_equilateral (a b c : ℝ) : Prop :=
a = b ∧ b = c

theorem triangle_equilateral (A B C a b c : ℝ) (hB : B = 60) (hb : b^2 = a * c) (hcos : b^2 = a^2 + c^2 - a * c):
  is_equilateral a b c :=
by
  sorry

end NUMINAMATH_GPT_triangle_equilateral_l1500_150076


namespace NUMINAMATH_GPT_ellipse_foci_k_value_l1500_150005

theorem ellipse_foci_k_value 
    (k : ℝ) 
    (h1 : 5 * (0:ℝ)^2 + k * (2:ℝ)^2 = 5): 
    k = 1 := 
by 
  sorry

end NUMINAMATH_GPT_ellipse_foci_k_value_l1500_150005


namespace NUMINAMATH_GPT_circle_radius_l1500_150018

theorem circle_radius (P Q : ℝ) (h1 : P = π * r^2) (h2 : Q = 2 * π * r) (h3 : P / Q = 15) : r = 30 :=
by
  sorry

end NUMINAMATH_GPT_circle_radius_l1500_150018


namespace NUMINAMATH_GPT_product_of_terms_in_geometric_sequence_l1500_150054

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ n m : ℕ, a (n + m) = a n * a m

noncomputable def roots_of_quadratic (a b c : ℝ) (r1 r2 : ℝ) : Prop :=
r1 * r2 = c

theorem product_of_terms_in_geometric_sequence
  (a : ℕ → ℝ)
  (h1 : geometric_sequence a)
  (h2 : roots_of_quadratic 1 (-4) 3 (a 5) (a 7)) :
  a 2 * a 10 = 3 :=
sorry

end NUMINAMATH_GPT_product_of_terms_in_geometric_sequence_l1500_150054


namespace NUMINAMATH_GPT_running_time_square_field_l1500_150051

theorem running_time_square_field
  (side : ℕ)
  (running_speed_kmh : ℕ)
  (perimeter : ℕ := 4 * side)
  (running_speed_ms : ℕ := (running_speed_kmh * 1000) / 3600)
  (time : ℕ := perimeter / running_speed_ms) 
  (h_side : side = 35)
  (h_speed : running_speed_kmh = 9) :
  time = 56 := 
by
  sorry

end NUMINAMATH_GPT_running_time_square_field_l1500_150051


namespace NUMINAMATH_GPT_fraction_decomposition_l1500_150024

noncomputable def p (n : ℕ) : ℚ :=
  (n + 1) / 2

noncomputable def q (n : ℕ) : ℚ :=
  n * p n

theorem fraction_decomposition (n : ℕ) (h : ∃ k : ℕ, n = 5 + 2*k) :
  (2 / n : ℚ) = (1 / p n) + (1 / q n) :=
by
  sorry

end NUMINAMATH_GPT_fraction_decomposition_l1500_150024


namespace NUMINAMATH_GPT_complex_modulus_l1500_150071

open Complex

noncomputable def modulus_of_complex : ℂ :=
  (1 - 2 * Complex.I) * (1 - 2 * Complex.I) / Complex.I

theorem complex_modulus : Complex.abs modulus_of_complex = 5 :=
  sorry

end NUMINAMATH_GPT_complex_modulus_l1500_150071


namespace NUMINAMATH_GPT_find_x_l1500_150092

noncomputable def arithmetic_sequence (x : ℝ) : Prop := 
  (x + 1) - (1/3) = 4 * x - (x + 1)

theorem find_x :
  ∃ x : ℝ, arithmetic_sequence x ∧ x = 5 / 6 :=
by
  use 5 / 6
  unfold arithmetic_sequence
  sorry

end NUMINAMATH_GPT_find_x_l1500_150092


namespace NUMINAMATH_GPT_error_difference_l1500_150043

noncomputable def total_income_without_error (T: ℝ) : ℝ :=
  T + 110000

noncomputable def total_income_with_error (T: ℝ) : ℝ :=
  T + 1100000

noncomputable def mean_without_error (T: ℝ) : ℝ :=
  (T + 110000) / 500

noncomputable def mean_with_error (T: ℝ) : ℝ :=
  (T + 1100000) / 500

theorem error_difference (T: ℝ) :
  mean_with_error T - mean_without_error T = 1980 :=
by
  sorry

end NUMINAMATH_GPT_error_difference_l1500_150043


namespace NUMINAMATH_GPT_solve_equation_l1500_150078

theorem solve_equation (x : ℚ) :
  (x^2 + 3 * x + 4) / (x + 5) = x + 6 ↔ x = -13 / 4 :=
by
  sorry

end NUMINAMATH_GPT_solve_equation_l1500_150078


namespace NUMINAMATH_GPT_irreducible_polynomial_l1500_150084

open Polynomial

theorem irreducible_polynomial (n : ℕ) : Irreducible ((X^2 + X)^(2^n) + 1 : ℤ[X]) := sorry

end NUMINAMATH_GPT_irreducible_polynomial_l1500_150084


namespace NUMINAMATH_GPT_total_volume_of_barrel_l1500_150019

-- Define the total volume of the barrel and relevant conditions.
variable (x : ℝ) -- total volume of the barrel

-- State the given condition about the barrel's honey content.
def condition := (0.7 * x - 0.3 * x = 30)

-- Goal to prove:
theorem total_volume_of_barrel : condition x → x = 75 :=
by
  sorry

end NUMINAMATH_GPT_total_volume_of_barrel_l1500_150019


namespace NUMINAMATH_GPT_midpoint_trace_quarter_circle_l1500_150056

theorem midpoint_trace_quarter_circle (L : ℝ) (hL : 0 < L):
  ∃ (C : ℝ) (M : ℝ × ℝ → ℝ), 
    (∀ (x y : ℝ), x^2 + y^2 = L^2 → M (x, y) = C) ∧ 
    (C = (1/2) * L) ∧ 
    (∀ (x y : ℝ), M (x, y) = (x/2)^2 + (y/2)^2) → 
    ∀ (x y : ℝ), x^2 + y^2 = L^2 → (x/2)^2 + (y/2)^2 = (1/2 * L)^2 := 
by
  sorry

end NUMINAMATH_GPT_midpoint_trace_quarter_circle_l1500_150056


namespace NUMINAMATH_GPT_candies_per_person_l1500_150067

theorem candies_per_person (a b people total_candies candies_per_person : ℕ)
  (h1: a = 17)
  (h2: b = 19)
  (h3: people = 9)
  (h4: total_candies = a + b)
  (h5: candies_per_person = total_candies / people) :
  candies_per_person = 4 :=
by sorry

end NUMINAMATH_GPT_candies_per_person_l1500_150067


namespace NUMINAMATH_GPT_find_point_C_l1500_150012

-- Definitions of the conditions
def line_eq (x y : ℝ) : Prop := x - 2 * y - 1 = 0
def parabola_eq (x y : ℝ) : Prop := y^2 = 4 * x
def on_parabola (C : ℝ × ℝ) : Prop := parabola_eq C.1 C.2
def perpendicular_at_C (A B C : ℝ × ℝ) : Prop :=
  (A.1 - C.1) * (B.1 - C.1) + (A.2 - C.2) * (B.2 - C.2) = 0

-- Points A and B satisfy both the line and parabola equations
def intersection_points (A B : ℝ × ℝ) : Prop :=
  line_eq A.1 A.2 ∧ parabola_eq A.1 A.2 ∧
  line_eq B.1 B.2 ∧ parabola_eq B.1 B.2

-- Statement to be proven
theorem find_point_C (A B : ℝ × ℝ) (hA : intersection_points A B) :
  ∃ C : ℝ × ℝ, on_parabola C ∧ perpendicular_at_C A B C ∧
    (C = (1, -2) ∨ C = (9, -6)) :=
by
  sorry

end NUMINAMATH_GPT_find_point_C_l1500_150012


namespace NUMINAMATH_GPT_cost_of_drapes_l1500_150004

theorem cost_of_drapes (D: ℝ) (h1 : 3 * 40 = 120) (h2 : D * 3 + 120 = 300) : D = 60 :=
  sorry

end NUMINAMATH_GPT_cost_of_drapes_l1500_150004


namespace NUMINAMATH_GPT_sum_of_0_75_of_8_and_2_l1500_150035

theorem sum_of_0_75_of_8_and_2 : 0.75 * 8 + 2 = 8 := by
  sorry

end NUMINAMATH_GPT_sum_of_0_75_of_8_and_2_l1500_150035


namespace NUMINAMATH_GPT_largest_of_nine_consecutive_integers_l1500_150034

theorem largest_of_nine_consecutive_integers (sum_eq_99: ∃ (n : ℕ), 99 = (n - 4) + (n - 3) + (n - 2) + (n - 1) + n + (n + 1) + (n + 2) + (n + 3) + (n + 4)) : 
  ∃ n : ℕ, n = 15 :=
by
  sorry

end NUMINAMATH_GPT_largest_of_nine_consecutive_integers_l1500_150034


namespace NUMINAMATH_GPT_train_passes_man_in_approx_21_seconds_l1500_150037

noncomputable def train_length : ℝ := 385
noncomputable def train_speed_kmph : ℝ := 60
noncomputable def man_speed_kmph : ℝ := 6

-- Convert speeds to m/s
noncomputable def kmph_to_mps (kmph : ℝ) : ℝ := kmph * (1000 / 3600)
noncomputable def train_speed_mps : ℝ := kmph_to_mps train_speed_kmph
noncomputable def man_speed_mps : ℝ := kmph_to_mps man_speed_kmph

-- Calculate relative speed
noncomputable def relative_speed_mps : ℝ := train_speed_mps + man_speed_mps

-- Calculate time
noncomputable def time_to_pass : ℝ := train_length / relative_speed_mps

theorem train_passes_man_in_approx_21_seconds : abs (time_to_pass - 21) < 1 :=
by
  sorry

end NUMINAMATH_GPT_train_passes_man_in_approx_21_seconds_l1500_150037


namespace NUMINAMATH_GPT_h_2023_eq_4052_l1500_150086

theorem h_2023_eq_4052 (h : ℕ → ℕ) (h1 : h 1 = 2) (h2 : h 2 = 2) 
    (h3 : ∀ n ≥ 3, h n = h (n-1) - h (n-2) + 2 * n) : h 2023 = 4052 := 
by
  -- Use conditions as given
  sorry

end NUMINAMATH_GPT_h_2023_eq_4052_l1500_150086


namespace NUMINAMATH_GPT_new_cost_after_decrease_l1500_150082

theorem new_cost_after_decrease (C new_C : ℝ) (hC : C = 1100) (h_decrease : new_C = 0.76 * C) : new_C = 836 :=
-- To be proved based on the given conditions
sorry

end NUMINAMATH_GPT_new_cost_after_decrease_l1500_150082


namespace NUMINAMATH_GPT_solve_for_x_l1500_150081

theorem solve_for_x : ∃ x : ℝ, 4 * x + 6 * x = 360 - 9 * (x - 4) ∧ x = 396 / 19 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l1500_150081


namespace NUMINAMATH_GPT_find_a_given_even_l1500_150064

def f (x a : ℝ) : ℝ := (x + a) * (x - 4)

theorem find_a_given_even (a : ℝ) :
  (∀ x : ℝ, f x a = f (-x) a) → a = 4 :=
by
  unfold f
  sorry

end NUMINAMATH_GPT_find_a_given_even_l1500_150064


namespace NUMINAMATH_GPT_car_city_mileage_l1500_150040

theorem car_city_mileage (h c t : ℝ) 
  (h_eq : h * t = 462)
  (c_eq : (h - 15) * t = 336) 
  (c_def : c = h - 15) : 
  c = 40 := 
by 
  sorry

end NUMINAMATH_GPT_car_city_mileage_l1500_150040


namespace NUMINAMATH_GPT_good_coloring_count_l1500_150014

noncomputable def c_n (n : ℕ) : ℤ :=
  1 / 2 * (3^(n + 1) + (-1)^(n + 1))

theorem good_coloring_count (n : ℕ) : 
  ∃ c : ℕ → ℤ, c n = c_n n := sorry

end NUMINAMATH_GPT_good_coloring_count_l1500_150014


namespace NUMINAMATH_GPT_find_y_l1500_150062

theorem find_y (y : ℝ) (h : (17.28 / 12) / (3.6 * y) = 2) : y = 0.2 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_y_l1500_150062


namespace NUMINAMATH_GPT_binomial_12_3_equals_220_l1500_150009

theorem binomial_12_3_equals_220 : Nat.choose 12 3 = 220 := by
  sorry

end NUMINAMATH_GPT_binomial_12_3_equals_220_l1500_150009


namespace NUMINAMATH_GPT_tips_fraction_l1500_150095

-- Define the conditions
variables (S T : ℝ) (h : T = (2 / 4) * S)

-- The statement to be proved
theorem tips_fraction : (T / (S + T)) = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_tips_fraction_l1500_150095


namespace NUMINAMATH_GPT_solve_system_l1500_150085

theorem solve_system (x y z : ℝ) :
  x^2 = y^2 + z^2 ∧
  x^2024 = y^2024 + z^2024 ∧
  x^2025 = y^2025 + z^2025 ↔
  (y = x ∧ z = 0) ∨
  (y = -x ∧ z = 0) ∨
  (y = 0 ∧ z = x) ∨
  (y = 0 ∧ z = -x) :=
by {
  sorry -- The detailed proof will be filled here.
}

end NUMINAMATH_GPT_solve_system_l1500_150085


namespace NUMINAMATH_GPT_B_subset_A_iff_l1500_150025

namespace MathProofs

def A (x : ℝ) : Prop := -2 < x ∧ x < 5

def B (x : ℝ) (m : ℝ) : Prop := m + 1 ≤ x ∧ x ≤ 2 * m - 1

theorem B_subset_A_iff (m : ℝ) :
  (∀ x : ℝ, B x m → A x) ↔ m < 3 :=
by
  sorry

end MathProofs

end NUMINAMATH_GPT_B_subset_A_iff_l1500_150025


namespace NUMINAMATH_GPT_ratio_paislee_to_calvin_l1500_150047

theorem ratio_paislee_to_calvin (calvin_points paislee_points : ℕ) (h1 : calvin_points = 500) (h2 : paislee_points = 125) : paislee_points / calvin_points = 1 / 4 := by
  sorry

end NUMINAMATH_GPT_ratio_paislee_to_calvin_l1500_150047


namespace NUMINAMATH_GPT_mn_sum_l1500_150066

theorem mn_sum {m n : ℤ} (h : ∀ x : ℤ, (x + 8) * (x - 1) = x^2 + m * x + n) : m + n = -1 :=
by
  sorry

end NUMINAMATH_GPT_mn_sum_l1500_150066


namespace NUMINAMATH_GPT_find_value_of_expression_l1500_150007

theorem find_value_of_expression
  (x y : ℝ)
  (h : x^2 - 2*x + y^2 - 6*y + 10 = 0) :
  x^2 * y^2 + 2 * x * y + 1 = 16 :=
sorry

end NUMINAMATH_GPT_find_value_of_expression_l1500_150007


namespace NUMINAMATH_GPT_intersection_A_B_l1500_150017

open Set

-- Define sets A and B with given conditions
def A : Set ℕ := {0, 1, 2, 3}
def B : Set ℕ := {x | ∃ a ∈ A, x = 3 * a}

-- Prove the intersection of sets A and B
theorem intersection_A_B : A ∩ B = {0, 3} := 
by
  sorry

end NUMINAMATH_GPT_intersection_A_B_l1500_150017


namespace NUMINAMATH_GPT_brick_piles_l1500_150033

theorem brick_piles (x y z : ℤ) :
  2 * (x - 100) = y + 100 ∧
  x + z = 6 * (y - z) →
  x = 170 ∧ y = 40 :=
by
  sorry

end NUMINAMATH_GPT_brick_piles_l1500_150033


namespace NUMINAMATH_GPT_cost_of_previous_hay_l1500_150045

theorem cost_of_previous_hay
    (x : ℤ)
    (previous_hay_bales : ℤ)
    (better_quality_hay_cost : ℤ)
    (additional_amount_needed : ℤ)
    (better_quality_hay_bales : ℤ)
    (new_total_cost : ℤ) :
    previous_hay_bales = 10 ∧ 
    better_quality_hay_cost = 18 ∧ 
    additional_amount_needed = 210 ∧ 
    better_quality_hay_bales = 2 * previous_hay_bales ∧ 
    new_total_cost = better_quality_hay_bales * better_quality_hay_cost ∧ 
    new_total_cost - additional_amount_needed = 10 * x → 
    x = 15 := by
  sorry

end NUMINAMATH_GPT_cost_of_previous_hay_l1500_150045


namespace NUMINAMATH_GPT_find_math_marks_l1500_150042

theorem find_math_marks (subjects : ℕ)
  (english_marks : ℕ)
  (physics_marks : ℕ)
  (chemistry_marks : ℕ)
  (biology_marks : ℕ)
  (average_marks : ℝ)
  (math_marks : ℕ) :
  subjects = 5 →
  english_marks = 96 →
  physics_marks = 99 →
  chemistry_marks = 100 →
  biology_marks = 98 →
  average_marks = 98.2 →
  math_marks = 98 :=
by
  intros h_subjects h_english h_physics h_chemistry h_biology h_average
  sorry

end NUMINAMATH_GPT_find_math_marks_l1500_150042


namespace NUMINAMATH_GPT_sin_beta_l1500_150046

variable (α β : ℝ)
variable (hα1 : 0 < α) (hα2 : α < Real.pi / 2)
variable (hβ1 : 0 < β) (hβ2: β < Real.pi / 2)
variable (h1 : Real.cos α = 5 / 13)
variable (h2 : Real.sin (α - β) = 4 / 5)

theorem sin_beta (α β : ℝ) (hα1 : 0 < α) (hα2 : α < Real.pi / 2) 
  (hβ1 : 0 < β) (hβ2 : β < Real.pi / 2) 
  (h1 : Real.cos α = 5 / 13) 
  (h2 : Real.sin (α - β) = 4 / 5) : 
  Real.sin β = 16 / 65 := 
by 
  sorry

end NUMINAMATH_GPT_sin_beta_l1500_150046


namespace NUMINAMATH_GPT_solution_l1500_150031

open Real

variables (a b c A B C : ℝ)

-- Condition: In ΔABC, the sides opposite to angles A, B, and C are a, b, and c respectively
-- Condition: Given equation relating sides and angles in ΔABC
axiom eq1 : a * sin C / (1 - cos A) = sqrt 3 * c
-- Condition: b + c = 10
axiom eq2 : b + c = 10
-- Condition: Area of ΔABC
axiom eq3 : (1 / 2) * b * c * sin A = 4 * sqrt 3

-- The final statement to prove
theorem solution :
    (A = π / 3) ∧ (a = 2 * sqrt 13) :=
by
    sorry

end NUMINAMATH_GPT_solution_l1500_150031


namespace NUMINAMATH_GPT_greatest_nat_not_sum_of_two_composites_l1500_150061

def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ m k : ℕ, m > 1 ∧ k > 1 ∧ n = m * k

theorem greatest_nat_not_sum_of_two_composites :
  ¬ ∃ a b : ℕ, is_composite a ∧ is_composite b ∧ 11 = a + b ∧
  (∀ n : ℕ, n > 11 → ¬ ∃ x y : ℕ, is_composite x ∧ is_composite y ∧ n = x + y) :=
sorry

end NUMINAMATH_GPT_greatest_nat_not_sum_of_two_composites_l1500_150061


namespace NUMINAMATH_GPT_jerry_remaining_debt_l1500_150023

theorem jerry_remaining_debt :
  ∀ (paid_two_months_ago paid_last_month total_debt: ℕ),
  paid_two_months_ago = 12 →
  paid_last_month = paid_two_months_ago + 3 →
  total_debt = 50 →
  total_debt - (paid_two_months_ago + paid_last_month) = 23 :=
by
  intros paid_two_months_ago paid_last_month total_debt h1 h2 h3
  sorry

end NUMINAMATH_GPT_jerry_remaining_debt_l1500_150023


namespace NUMINAMATH_GPT_point_c_third_quadrant_l1500_150010

variable (a b : ℝ)

-- Definition of the conditions
def condition_1 : Prop := b = -1
def condition_2 : Prop := a = -3

-- Definition to check if a point is in the third quadrant
def is_third_quadrant (a b : ℝ) : Prop := a < 0 ∧ b < 0

-- The main statement to be proven
theorem point_c_third_quadrant (h1 : condition_1 b) (h2 : condition_2 a) :
  is_third_quadrant a b :=
by
  -- Proof of the theorem (to be completed)
  sorry

end NUMINAMATH_GPT_point_c_third_quadrant_l1500_150010


namespace NUMINAMATH_GPT_number_of_extreme_points_l1500_150003

-- Define the function's derivative
def f_derivative (x : ℝ) : ℝ := (x + 1)^2 * (x - 1) * (x - 2)

-- State the theorem
theorem number_of_extreme_points : ∃ n : ℕ, n = 2 ∧ 
  (∀ x, (f_derivative x = 0 → ((f_derivative (x - ε) > 0 ∧ f_derivative (x + ε) < 0) ∨ 
                             (f_derivative (x - ε) < 0 ∧ f_derivative (x + ε) > 0))) → 
   (x = 1 ∨ x = 2)) :=
sorry

end NUMINAMATH_GPT_number_of_extreme_points_l1500_150003


namespace NUMINAMATH_GPT_fraction_subtraction_l1500_150048

theorem fraction_subtraction (a b : ℕ) (h₁ : a = 18) (h₂ : b = 14) :
  (↑a / ↑b - ↑b / ↑a) = (32 / 63) := by
  sorry

end NUMINAMATH_GPT_fraction_subtraction_l1500_150048


namespace NUMINAMATH_GPT_problem1_problem2_l1500_150011

def box (n : ℕ) : ℕ := (10^n - 1) / 9

theorem problem1 (m : ℕ) :
  let b := box (3^m)
  b % (3^m) = 0 ∧ b % (3^(m+1)) ≠ 0 :=
  sorry

theorem problem2 (n : ℕ) :
  (n % 27 = 0) ↔ (box n % 27 = 0) :=
  sorry

end NUMINAMATH_GPT_problem1_problem2_l1500_150011


namespace NUMINAMATH_GPT_larger_number_l1500_150001

variables (x y : ℕ)

theorem larger_number (h1 : x + y = 47) (h2 : x - y = 7) : x = 27 :=
sorry

end NUMINAMATH_GPT_larger_number_l1500_150001


namespace NUMINAMATH_GPT_value_of_a_l1500_150015

/-- Given that 0.5% of a is 85 paise, prove that the value of a is 170 rupees. --/
theorem value_of_a (a : ℝ) (h : 0.005 * a = 85) : a = 170 := 
  sorry

end NUMINAMATH_GPT_value_of_a_l1500_150015


namespace NUMINAMATH_GPT_integer_solutions_l1500_150094

theorem integer_solutions :
  { (x, y) : ℤ × ℤ |
       y^2 + y = x^4 + x^3 + x^2 + x } =
  { (-1, -1), (-1, 0), (0, -1), (0, 0), (2, 5), (2, -6) } :=
by
  sorry

end NUMINAMATH_GPT_integer_solutions_l1500_150094
