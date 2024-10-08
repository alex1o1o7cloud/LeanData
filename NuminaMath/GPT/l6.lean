import Mathlib

namespace greatest_x_l6_6929

theorem greatest_x (x : ℕ) (h : x^2 < 32) : x ≤ 5 := 
sorry

end greatest_x_l6_6929


namespace find_fixed_point_on_ellipse_l6_6149

theorem find_fixed_point_on_ellipse (a b c : ℝ) (h_gt_zero : a > b ∧ b > 0)
    (h_ellipse : ∀ (P : ℝ × ℝ), (P.1 ^ 2 / a ^ 2) + (P.2 ^ 2 / b ^ 2) = 1)
    (A1 A2 : ℝ × ℝ)
    (h_A1 : A1 = (-a, 0))
    (h_A2 : A2 = (a, 0))
    (MC : ℝ) (h_MC : MC = (a^2 + b^2) / c) :
  ∃ (M : ℝ × ℝ), M = (MC, 0) := 
sorry

end find_fixed_point_on_ellipse_l6_6149


namespace find_x_value_l6_6987

noncomputable def check_x (x : ℝ) : Prop :=
  (0 < x) ∧ (Real.sqrt (12 * x) * Real.sqrt (5 * x) * Real.sqrt (6 * x) * Real.sqrt (10 * x) = 10)

theorem find_x_value (x : ℝ) (h : check_x x) : x = 1 / 6 :=
by 
  sorry

end find_x_value_l6_6987


namespace distribute_pictures_l6_6003

/-
Tiffany uploaded 34 pictures from her phone, 55 from her camera,
and 12 from her tablet to Facebook. If she sorted the pics into 7 different albums
with the same amount of pics in each album, how many pictures were in each of the albums?
-/

theorem distribute_pictures :
  let phone_pics := 34
  let camera_pics := 55
  let tablet_pics := 12
  let total_pics := phone_pics + camera_pics + tablet_pics
  let albums := 7
  ∃ k r, (total_pics = k * albums + r) ∧ (r < albums) := by
  sorry

end distribute_pictures_l6_6003


namespace jack_paycheck_l6_6385

theorem jack_paycheck (P : ℝ) (h1 : 0.15 * 150 + 0.25 * (P - 150) + 30 + 70 / 100 * (P - (0.15 * 150 + 0.25 * (P - 150) + 30)) * 30 / 100 = 50) : P = 242.22 :=
sorry

end jack_paycheck_l6_6385


namespace no_positive_integer_solution_l6_6189

def is_solution (x y z t : ℕ) : Prop :=
  x^2 + 5 * y^2 = z^2 ∧ 5 * x^2 + y^2 = t^2

theorem no_positive_integer_solution :
  ¬ ∃ (x y z t : ℕ), 0 < x ∧ 0 < y ∧ 0 < z ∧ 0 < t ∧ is_solution x y z t :=
by
  sorry

end no_positive_integer_solution_l6_6189


namespace solution_set_all_real_solution_set_empty_exists_at_least_one_solution_l6_6811

-- Definitions for the inequality ax^2 - 2ax + 2a - 3 < 0
def quadratic_expr (a x : ℝ) : ℝ := a * x^2 - 2 * a * x + 2 * a - 3

-- Requirement (1): The solution set is ℝ
theorem solution_set_all_real (a : ℝ) (h : a ≤ 0) : 
  ∀ x : ℝ, quadratic_expr a x < 0 :=
by sorry

-- Requirement (2): The solution set is ∅
theorem solution_set_empty (a : ℝ) (h : a ≥ 3) : 
  ¬∃ x : ℝ, quadratic_expr a x < 0 :=
by sorry

-- Requirement (3): There is at least one real solution
theorem exists_at_least_one_solution (a : ℝ) (h : a < 3) : 
  ∃ x : ℝ, quadratic_expr a x < 0 :=
by sorry

end solution_set_all_real_solution_set_empty_exists_at_least_one_solution_l6_6811


namespace xavier_yvonne_not_zelda_prob_l6_6489

def Px : ℚ := 1 / 4
def Py : ℚ := 2 / 3
def Pz : ℚ := 5 / 8

theorem xavier_yvonne_not_zelda_prob : 
  (Px * Py * (1 - Pz) = 1 / 16) :=
by 
  sorry

end xavier_yvonne_not_zelda_prob_l6_6489


namespace no_solution_in_positive_integers_l6_6439

theorem no_solution_in_positive_integers
    (x y : ℕ)
    (h : x > 0 ∧ y > 0) :
    x^2006 - 4 * y^2006 - 2006 ≠ 4 * y^2007 + 2007 * y :=
by
  sorry

end no_solution_in_positive_integers_l6_6439


namespace complement_of_M_in_U_l6_6024

namespace SetComplements

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {1, 3, 5}
def complement_U_M : Set ℕ := U \ M

theorem complement_of_M_in_U :
  complement_U_M = {2, 4, 6} :=
by
  sorry

end SetComplements

end complement_of_M_in_U_l6_6024


namespace initial_people_in_line_l6_6132

theorem initial_people_in_line (x : ℕ) (h1 : x + 22 = 83) : x = 61 :=
by sorry

end initial_people_in_line_l6_6132


namespace quadratic_sum_solutions_l6_6194

noncomputable def sum_of_solutions (a b c : ℝ) : ℝ := 
  (-b/a)

theorem quadratic_sum_solutions : 
  ∀ x : ℝ, sum_of_solutions 1 (-9) (-45) = 9 := 
by
  intro x
  sorry

end quadratic_sum_solutions_l6_6194


namespace jumpy_implies_not_green_l6_6889

variables (Lizard : Type)
variables (IsJumpy IsGreen CanSing CanDance : Lizard → Prop)

-- Conditions given in the problem
axiom jumpy_implies_can_sing : ∀ l, IsJumpy l → CanSing l
axiom green_implies_cannot_dance : ∀ l, IsGreen l → ¬ CanDance l
axiom cannot_dance_implies_cannot_sing : ∀ l, ¬ CanDance l → ¬ CanSing l

theorem jumpy_implies_not_green (l : Lizard) : IsJumpy l → ¬ IsGreen l :=
by
  sorry

end jumpy_implies_not_green_l6_6889


namespace determine_head_start_l6_6104

def head_start (v : ℝ) (s : ℝ) : Prop :=
  let a_speed := 2 * v
  let distance := 142
  distance / a_speed = (distance - s) / v

theorem determine_head_start (v : ℝ) : head_start v 71 :=
  by
    sorry

end determine_head_start_l6_6104


namespace right_triangle_x_value_l6_6300

theorem right_triangle_x_value (BM MA BC CA: ℝ) (M_is_altitude: BM + MA = BC + CA)
  (x: ℝ) (h: ℝ) (d: ℝ) (M: BM = x) (CB: BC = h) (CA: CA = d) :
  x = (2 * h * d - d ^ 2 / 4) / (2 * d + 2 * h) := by
  sorry

end right_triangle_x_value_l6_6300


namespace tan_alpha_plus_pi_cos_alpha_minus_pi_div_two_sin_alpha_plus_3pi_div_two_l6_6172

open Real

theorem tan_alpha_plus_pi (α : ℝ)
  (h1 : sin α = 3 / 5)
  (h2 : π / 2 < α ∧ α < π) :
  tan (α + π) = -3 / 4 :=
sorry

theorem cos_alpha_minus_pi_div_two_sin_alpha_plus_3pi_div_two (α : ℝ)
  (h1 : sin α = 3 / 5)
  (h2 : π / 2 < α ∧ α < π) :
  cos (α - π / 2) * sin (α + 3 * π / 2) = 12 / 25 :=
sorry

end tan_alpha_plus_pi_cos_alpha_minus_pi_div_two_sin_alpha_plus_3pi_div_two_l6_6172


namespace original_price_of_petrol_l6_6702

theorem original_price_of_petrol (P : ℝ) :
  (∃ P, 
    ∀ (GA GB GC : ℝ),
    0.8 * P = 0.8 * P ∧
    GA = 200 / P ∧
    GB = 300 / P ∧
    GC = 400 / P ∧
    200 = (GA + 8) * 0.8 * P ∧
    300 = (GB + 15) * 0.8 * P ∧
    400 = (GC + 22) * 0.8 * P) → 
  P = 6.25 :=
by
  sorry

end original_price_of_petrol_l6_6702


namespace area_of_shape_l6_6622

def points := [(0, 1), (1, 2), (3, 2), (4, 1), (2, 0)]

theorem area_of_shape : 
  let I := 6 -- Number of interior points
  let B := 5 -- Number of boundary points
  ∃ (A : ℝ), A = I + B / 2 - 1 ∧ A = 7.5 := 
  by
    use 7.5
    simp
    sorry

end area_of_shape_l6_6622


namespace quadratic_eq_roots_are_coeffs_l6_6488

theorem quadratic_eq_roots_are_coeffs :
  ∃ (a b : ℝ), (a = r_1) → (b = r_2) →
  (r_1 + r_2 = -a) → (r_1 * r_2 = b) →
  r_1 = 1 ∧ r_2 = -2 ∧ (x^2 + x - 2 = 0):=
by
  sorry

end quadratic_eq_roots_are_coeffs_l6_6488


namespace eleven_pow_four_l6_6325

theorem eleven_pow_four : 11 ^ 4 = 14641 := 
by sorry

end eleven_pow_four_l6_6325


namespace stratified_sampling_male_athletes_l6_6656

theorem stratified_sampling_male_athletes : 
  ∀ (total_males total_females total_to_sample : ℕ), 
    total_males = 20 → 
    total_females = 10 → 
    total_to_sample = 6 → 
    20 * (total_to_sample / (total_males + total_females)) = 4 :=
by
  intros total_males total_females total_to_sample h_males h_females h_sample
  rw [h_males, h_females, h_sample]
  sorry

end stratified_sampling_male_athletes_l6_6656


namespace solve_for_a_l6_6182

theorem solve_for_a (x a : ℝ) (h : x = -2) (hx : 2 * x + 3 * a = 0) : a = 4 / 3 :=
by
  sorry

end solve_for_a_l6_6182


namespace roundness_1000000_l6_6065

-- Definitions based on the conditions in the problem
def prime_factors (n : ℕ) : List (ℕ × ℕ) :=
  if n = 1 then []
  else [(2, 6), (5, 6)] -- Example specifically for 1,000,000

def roundness (n : ℕ) : ℕ :=
  (prime_factors n).map Prod.snd |>.sum

-- The main theorem
theorem roundness_1000000 : roundness 1000000 = 12 := by
  sorry

end roundness_1000000_l6_6065


namespace diameter_of_lid_is_2_inches_l6_6966

noncomputable def π : ℝ := 3.14
def C : ℝ := 6.28

theorem diameter_of_lid_is_2_inches (d : ℝ) : d = C / π → d = 2 :=
by
  intro h
  sorry

end diameter_of_lid_is_2_inches_l6_6966


namespace eggs_removed_l6_6851

theorem eggs_removed (initial remaining : ℕ) (h1 : initial = 27) (h2 : remaining = 20) : initial - remaining = 7 :=
by
  sorry

end eggs_removed_l6_6851


namespace coloring_two_corners_removed_l6_6730

noncomputable def coloring_count (total_ways : Nat) (ways_without_corner_a : Nat) : Nat :=
  total_ways - 2 * (total_ways - ways_without_corner_a) / 2 + 
  (ways_without_corner_a - (total_ways - ways_without_corner_a) / 2)

theorem coloring_two_corners_removed : coloring_count 120 96 = 78 := by
  sorry

end coloring_two_corners_removed_l6_6730


namespace pioneers_club_attendance_l6_6378

theorem pioneers_club_attendance :
  ∃ (A B : (Fin 11)), A ≠ B ∧
  (∃ (clubs_A clubs_B : Finset (Fin 5)), clubs_A = clubs_B) :=
by
  sorry

end pioneers_club_attendance_l6_6378


namespace comparison_of_large_exponents_l6_6060

theorem comparison_of_large_exponents : 2^1997 > 5^850 := sorry

end comparison_of_large_exponents_l6_6060


namespace complex_proof_problem_l6_6657

theorem complex_proof_problem (i : ℂ) (h1 : i^2 = -1) :
  (i^2 + i^3 + i^4) / (1 - i) = (1 / 2) - (1 / 2) * i :=
by
  -- Proof will be provided here
  sorry

end complex_proof_problem_l6_6657


namespace mushrooms_gigi_cut_l6_6056

-- Definitions based on conditions in part a)
def pieces_per_mushroom := 4
def kenny_sprinkled := 38
def karla_sprinkled := 42
def pieces_remaining := 8

-- The total number of pieces is the sum of Kenny's, Karla's, and the remaining pieces.
def total_pieces := kenny_sprinkled + karla_sprinkled + pieces_remaining

-- The number of mushrooms GiGi cut up at the beginning is total_pieces divided by pieces_per_mushroom.
def mushrooms_cut := total_pieces / pieces_per_mushroom

-- The theorem to be proved.
theorem mushrooms_gigi_cut (h1 : pieces_per_mushroom = 4)
                           (h2 : kenny_sprinkled = 38)
                           (h3 : karla_sprinkled = 42)
                           (h4 : pieces_remaining = 8)
                           (h5 : total_pieces = kenny_sprinkled + karla_sprinkled + pieces_remaining)
                           (h6 : mushrooms_cut = total_pieces / pieces_per_mushroom) :
  mushrooms_cut = 22 :=
by
  sorry

end mushrooms_gigi_cut_l6_6056


namespace sum_first_40_terms_l6_6295

-- Given: The sum of the first 10 terms of a geometric sequence is 9
axiom S_10 : ℕ → ℕ
axiom sum_S_10 : S_10 10 = 9 

-- Given: The sum of the terms from the 11th to the 20th is 36
axiom S_20 : ℕ → ℕ
axiom sum_S_20 : S_20 20 - S_10 10 = 36

-- Let Sn be the sum of the first n terms in the geometric sequence
def Sn (n : ℕ) : ℕ := sorry

-- Prove: The sum of the first 40 terms is 144
theorem sum_first_40_terms : Sn 40 = 144 := sorry

end sum_first_40_terms_l6_6295


namespace asymptote_of_hyperbola_l6_6239

theorem asymptote_of_hyperbola (h : (∀ x y : ℝ, y^2 / 3 - x^2 / 2 = 1)) : 
  (∀ x : ℝ, ∃ y : ℝ, y = (sqrt6 / 2) * x ∨ y = - (sqrt6 / 2) * x) :=
sorry

end asymptote_of_hyperbola_l6_6239


namespace robin_initial_gum_is_18_l6_6456

-- Defining the conditions as given in the problem
def given_gum : ℝ := 44
def total_gum : ℝ := 62

-- Statement to prove that the initial number of pieces of gum Robin had is 18
theorem robin_initial_gum_is_18 : total_gum - given_gum = 18 := by
  -- Proof goes here
  sorry

end robin_initial_gum_is_18_l6_6456


namespace absolute_difference_probability_l6_6521

-- Define the conditions
def num_red_marbles : ℕ := 1500
def num_black_marbles : ℕ := 2000
def total_marbles : ℕ := num_red_marbles + num_black_marbles

def P_s : ℚ :=
  let ways_to_choose_2_red := (num_red_marbles * (num_red_marbles - 1)) / 2
  let ways_to_choose_2_black := (num_black_marbles * (num_black_marbles - 1)) / 2
  let total_favorable_outcomes := ways_to_choose_2_red + ways_to_choose_2_black
  total_favorable_outcomes / (total_marbles * (total_marbles - 1) / 2)

def P_d : ℚ :=
  (num_red_marbles * num_black_marbles) / (total_marbles * (total_marbles - 1) / 2)

-- Prove the statement
theorem absolute_difference_probability : |P_s - P_d| = 1 / 50 := by
  sorry

end absolute_difference_probability_l6_6521


namespace simplify_expression_l6_6507

variable {R : Type*} [CommRing R]

theorem simplify_expression (x y : R) : (x + 2 * y) * (x - 2 * y) - y * (3 - 4 * y) = x^2 - 3 * y :=
by
  sorry

end simplify_expression_l6_6507


namespace triangle_area_l6_6653

noncomputable def vector (x y z : ℝ) : ℝ × ℝ × ℝ :=
(x, y, z)

noncomputable def cross_product (v w : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
(v.2.1 * w.2.2 - v.2.2 * w.2.1,
 v.2.2 * w.1 - v.1 * w.2.2,
 v.1 * w.2.1 - v.2.1 * w.1)

noncomputable def magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
Real.sqrt (v.1^2 + v.2.1^2 + v.2.2^2)

theorem triangle_area :
  let A := vector 2 1 (-1)
  let B := vector 3 0 3
  let C := vector 7 3 2
  let AB := (B.1 - A.1, B.2.1 - A.2.1, B.2.2 - A.2.2)
  let AC := (C.1 - A.1, C.2.1 - A.2.1, C.2.2 - A.2.2)
  0.5 * magnitude (cross_product AB AC) = (1 / 2) * Real.sqrt 459 :=
by
  -- All the steps needed to prove the theorem here
  sorry

end triangle_area_l6_6653


namespace megan_roles_other_than_lead_l6_6946

def total_projects : ℕ := 800

def theater_percentage : ℚ := 50 / 100
def films_percentage : ℚ := 30 / 100
def television_percentage : ℚ := 20 / 100

def theater_lead_percentage : ℚ := 55 / 100
def theater_support_percentage : ℚ := 30 / 100
def theater_ensemble_percentage : ℚ := 10 / 100
def theater_cameo_percentage : ℚ := 5 / 100

def films_lead_percentage : ℚ := 70 / 100
def films_support_percentage : ℚ := 20 / 100
def films_minor_percentage : ℚ := 7 / 100
def films_cameo_percentage : ℚ := 3 / 100

def television_lead_percentage : ℚ := 60 / 100
def television_support_percentage : ℚ := 25 / 100
def television_recurring_percentage : ℚ := 10 / 100
def television_guest_percentage : ℚ := 5 / 100

theorem megan_roles_other_than_lead :
  let theater_projects := total_projects * theater_percentage
  let films_projects := total_projects * films_percentage
  let television_projects := total_projects * television_percentage

  let theater_other_roles := (theater_projects * theater_support_percentage) + 
                             (theater_projects * theater_ensemble_percentage) + 
                             (theater_projects * theater_cameo_percentage)

  let films_other_roles := (films_projects * films_support_percentage) + 
                           (films_projects * films_minor_percentage) + 
                           (films_projects * films_cameo_percentage)

  let television_other_roles := (television_projects * television_support_percentage) + 
                                (television_projects * television_recurring_percentage) + 
                                (television_projects * television_guest_percentage)
  
  theater_other_roles + films_other_roles + television_other_roles = 316 :=
by
  sorry

end megan_roles_other_than_lead_l6_6946


namespace probability_of_two_digit_number_l6_6145

def total_elements_in_set : ℕ := 961
def two_digit_elements_in_set : ℕ := 60

theorem probability_of_two_digit_number :
  (two_digit_elements_in_set : ℚ) / total_elements_in_set = 60 / 961 := by
  sorry

end probability_of_two_digit_number_l6_6145


namespace triangle_inequality_l6_6031

theorem triangle_inequality (a b c : ℝ) (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a) : True :=
  sorry

end triangle_inequality_l6_6031


namespace andy_demerits_for_joke_l6_6911

def max_demerits := 50
def demerits_late_per_instance := 2
def instances_late := 6
def remaining_demerits := 23
def total_demerits := max_demerits - remaining_demerits
def demerits_late := demerits_late_per_instance * instances_late
def demerits_joke := total_demerits - demerits_late

theorem andy_demerits_for_joke : demerits_joke = 15 := by
  sorry

end andy_demerits_for_joke_l6_6911


namespace count_non_squares_or_cubes_l6_6958

theorem count_non_squares_or_cubes (n : ℕ) (h₀ : 1 ≤ n ∧ n ≤ 200) : 
  ∃ c, c = 182 ∧ 
  (∃ k, k^2 = n ∨ ∃ m, m^3 = n) → false :=
by
  sorry

end count_non_squares_or_cubes_l6_6958


namespace edge_length_of_inscribed_cube_in_sphere_l6_6718

noncomputable def edge_length_of_cube_in_sphere (surface_area_sphere : ℝ) (π_cond : surface_area_sphere = 36 * Real.pi) : ℝ :=
  let x := 2 * Real.sqrt 3
  x

theorem edge_length_of_inscribed_cube_in_sphere (surface_area_sphere : ℝ) (π_cond : surface_area_sphere = 36 * Real.pi) :
  edge_length_of_cube_in_sphere surface_area_sphere π_cond = 2 * Real.sqrt 3 :=
by
  sorry

end edge_length_of_inscribed_cube_in_sphere_l6_6718


namespace lowest_temperature_l6_6518

-- Define the temperatures in the four cities.
def temp_Harbin := -20
def temp_Beijing := -10
def temp_Hangzhou := 0
def temp_Jinhua := 2

-- The proof statement asserting the lowest temperature.
theorem lowest_temperature :
  min temp_Harbin (min temp_Beijing (min temp_Hangzhou temp_Jinhua)) = -20 :=
by
  -- Proof omitted
  sorry

end lowest_temperature_l6_6518


namespace foil_covered_prism_width_l6_6465

theorem foil_covered_prism_width 
    (l w h : ℕ) 
    (h_w_eq_2l : w = 2 * l)
    (h_w_eq_2h : w = 2 * h)
    (h_volume : l * w * h = 128) 
    (h_foiled_width : q = w + 2) :
  q = 10 := 
sorry

end foil_covered_prism_width_l6_6465


namespace similar_triangle_angles_l6_6477

theorem similar_triangle_angles (α β γ : ℝ) (h1 : α + β + γ = Real.pi) (h2 : α + β/2 + γ/2 = Real.pi):
  ∃ (k : ℝ), α = k ∧ β = 2 * k ∧ γ = 4 * k ∧ k = Real.pi / 7 := 
sorry

end similar_triangle_angles_l6_6477


namespace ploughing_problem_l6_6631

theorem ploughing_problem
  (hours_per_day_group1 : ℕ)
  (days_group1 : ℕ)
  (bulls_group1 : ℕ)
  (total_fields_group2 : ℕ)
  (hours_per_day_group2 : ℕ)
  (days_group2 : ℕ)
  (bulls_group2 : ℕ)
  (fields_group1 : ℕ)
  (fields_group2 : ℕ) :
    hours_per_day_group1 = 10 →
    days_group1 = 3 →
    bulls_group1 = 10 →
    hours_per_day_group2 = 8 →
    days_group2 = 2 →
    bulls_group2 = 30 →
    fields_group2 = 32 →
    480 * fields_group1 = 300 * fields_group2 →
    fields_group1 = 20 := by
  sorry

end ploughing_problem_l6_6631


namespace discount_received_l6_6930

theorem discount_received (original_cost : ℝ) (amt_spent : ℝ) (discount : ℝ) 
  (h1 : original_cost = 467) (h2 : amt_spent = 68) : 
  discount = 399 :=
by
  sorry

end discount_received_l6_6930


namespace total_first_year_students_l6_6201

theorem total_first_year_students (males : ℕ) (sample_size : ℕ) (female_in_sample : ℕ) (N : ℕ)
  (h1 : males = 570)
  (h2 : sample_size = 110)
  (h3 : female_in_sample = 53)
  (h4 : N = ((sample_size - female_in_sample) * males) / (sample_size - (sample_size - female_in_sample)))
  : N = 1100 := 
by
  sorry

end total_first_year_students_l6_6201


namespace dealer_sold_70_hondas_l6_6080

theorem dealer_sold_70_hondas
  (total_cars: ℕ)
  (percent_audi percent_toyota percent_acura percent_honda : ℝ)
  (total_audi := total_cars * percent_audi)
  (total_toyota := total_cars * percent_toyota)
  (total_acura := total_cars * percent_acura)
  (total_honda := total_cars * percent_honda )
  (h1 : total_cars = 200)
  (h2 : percent_audi = 0.15)
  (h3 : percent_toyota = 0.22)
  (h4 : percent_acura = 0.28)
  (h5 : percent_honda = 1 - (percent_audi + percent_toyota + percent_acura))
  : total_honda = 70 := 
  by
  sorry

end dealer_sold_70_hondas_l6_6080


namespace value_of_a_l6_6346

theorem value_of_a (a : ℝ) :
  ((abs ((1) - (2) + a)) = 1) ↔ (a = 0 ∨ a = 2) :=
by
  sorry

end value_of_a_l6_6346


namespace ashton_sheets_l6_6590
-- Import the entire Mathlib to bring in the necessary library

-- Defining the conditions and proving the statement
theorem ashton_sheets (t j a : ℕ) (h1 : t = j + 10) (h2 : j = 32) (h3 : j + a = t + 30) : a = 40 := by
  -- Sorry placeholder for the proof
  sorry

end ashton_sheets_l6_6590


namespace hyperbola_eccentricity_l6_6788

theorem hyperbola_eccentricity (a b c : ℝ) (h_asymptotes : b / a = 3 / 4 ∨ a / b = 3 / 4) :
  (c / a = 5 / 4) ∨ (c / a = 5 / 3) :=
by
  -- Proof omitted
  sorry

end hyperbola_eccentricity_l6_6788


namespace expand_binomial_trinomial_l6_6153

theorem expand_binomial_trinomial (x y z : ℝ) :
  (x + 12) * (3 * y + 4 * z + 15) = 3 * x * y + 4 * x * z + 15 * x + 36 * y + 48 * z + 180 :=
by sorry

end expand_binomial_trinomial_l6_6153


namespace overlapping_squares_proof_l6_6589

noncomputable def overlapping_squares_area (s : ℝ) : ℝ :=
  let AB := s
  let MN := s
  let areaMN := s^2
  let intersection_area := areaMN / 4
  intersection_area

theorem overlapping_squares_proof (s : ℝ) :
  overlapping_squares_area s = s^2 / 4 := by
    -- proof would go here
    sorry

end overlapping_squares_proof_l6_6589


namespace length_PD_l6_6101

theorem length_PD (PA PB PC PD : ℝ) (hPA : PA = 5) (hPB : PB = 3) (hPC : PC = 4) :
  PD = 4 * Real.sqrt 2 :=
by
  sorry

end length_PD_l6_6101


namespace ratio_x_y_l6_6021

theorem ratio_x_y (x y : ℤ) (h : (8 * x - 5 * y) * 3 = (11 * x - 3 * y) * 2) :
  x / y = 9 / 2 := by
  sorry

end ratio_x_y_l6_6021


namespace cacti_average_height_l6_6822

variables {Cactus1 Cactus2 Cactus3 Cactus4 Cactus5 Cactus6 : ℕ}
variables (condition1 : Cactus1 = 14)
variables (condition3 : Cactus3 = 7)
variables (condition6 : Cactus6 = 28)
variables (condition2 : Cactus2 = 14)
variables (condition4 : Cactus4 = 14)
variables (condition5 : Cactus5 = 14)

theorem cacti_average_height : 
  (Cactus1 + Cactus2 + Cactus3 + Cactus4 + Cactus5 + Cactus6 : ℕ) = 91 → 
  (91 : ℝ) / 6 = (15.2 : ℝ) :=
by
  sorry

end cacti_average_height_l6_6822


namespace distinct_rationals_count_l6_6671

theorem distinct_rationals_count : ∃ N : ℕ, (N = 40) ∧ ∀ k : ℚ, (|k| < 100) → (∃ x : ℤ, 3 * x^2 + k * x + 8 = 0) :=
by
  sorry

end distinct_rationals_count_l6_6671


namespace contrapositive_of_a_gt_1_then_a_sq_gt_1_l6_6545

theorem contrapositive_of_a_gt_1_then_a_sq_gt_1 : 
  (∀ a : ℝ, a > 1 → a^2 > 1) → (∀ a : ℝ, a^2 ≤ 1 → a ≤ 1) :=
by 
  sorry

end contrapositive_of_a_gt_1_then_a_sq_gt_1_l6_6545


namespace find_d_l6_6905

theorem find_d :
  ∃ d : ℝ, ∀ x : ℝ, x * (4 * x - 3) < d ↔ - (9/4 : ℝ) < x ∧ x < (3/2 : ℝ) ∧ d = 27 / 2 :=
by
  sorry

end find_d_l6_6905


namespace value_of_c_l6_6224

theorem value_of_c (c : ℝ) :
  (∀ x y : ℝ, (x, y) = ((2 + 8) / 2, (6 + 10) / 2) → x + y = c) → c = 13 :=
by
  -- Placeholder for proof
  sorry

end value_of_c_l6_6224


namespace rationalize_denominator_theorem_l6_6473

noncomputable def rationalize_denominator : Prop :=
  let num := 5
  let den := 2 + Real.sqrt 5
  let conj := 2 - Real.sqrt 5
  let expr := (num * conj) / (den * conj)
  expr = -10 + 5 * Real.sqrt 5

theorem rationalize_denominator_theorem : rationalize_denominator :=
  sorry

end rationalize_denominator_theorem_l6_6473


namespace seq_formula_l6_6744

def S (n : ℕ) (a : ℕ → ℤ) : ℤ := 2 * a n + 1

theorem seq_formula (a : ℕ → ℤ) (S_n : ℕ → ℤ)
  (hS : ∀ n, S_n n = S n a) :
  a = fun n => -2^(n-1) := by
  sorry

end seq_formula_l6_6744


namespace original_weight_of_marble_l6_6821

variable (W: ℝ) 

theorem original_weight_of_marble (h: 0.80 * 0.82 * 0.72 * W = 85.0176): W = 144 := 
by
  sorry

end original_weight_of_marble_l6_6821


namespace probability_of_5_blue_marbles_l6_6632

/--
Jane has a bag containing 9 blue marbles and 6 red marbles. 
She draws a marble, records its color, returns it to the bag, and repeats this process 8 times. 
We aim to prove that the probability that she draws exactly 5 blue marbles is \(0.279\).
-/
theorem probability_of_5_blue_marbles :
  let blue_probability := 9 / 15 
  let red_probability := 6 / 15
  let single_combination_prob := (blue_probability^5) * (red_probability^3)
  let combinations := (Nat.choose 8 5)
  let total_probability := combinations * single_combination_prob
  (Float.round (total_probability.toFloat * 1000) / 1000) = 0.279 :=
by
  sorry

end probability_of_5_blue_marbles_l6_6632


namespace solve_for_y_l6_6173

-- Define the main theorem to be proven
theorem solve_for_y (y : ℤ) (h : 7 * (4 * y + 3) - 5 = -3 * (2 - 8 * y) + 5 * y) : y = 22 :=
by
  sorry

end solve_for_y_l6_6173


namespace find_rectangle_width_l6_6757

variable (length_square : ℕ) (length_rectangle : ℕ) (width_rectangle : ℕ)

-- Given conditions
def square_side_length := 700
def rectangle_length := 400
def square_perimeter := 4 * square_side_length
def rectangle_perimeter := square_perimeter / 2
def rectangle_perimeter_eq := 2 * length_rectangle + 2 * width_rectangle

-- Statement to prove
theorem find_rectangle_width :
  (square_perimeter = 2800) →
  (rectangle_perimeter = 1400) →
  (length_rectangle = 400) →
  (rectangle_perimeter_eq = 1400) →
  (width_rectangle = 300) :=
by
  intros
  sorry

end find_rectangle_width_l6_6757


namespace necessary_but_not_sufficient_condition_l6_6953

open Real

-- Define α as an internal angle of a triangle
def is_internal_angle (α : ℝ) : Prop := (0 < α ∧ α < π)

-- Given conditions
axiom α : ℝ
axiom h1 : is_internal_angle α

-- Prove: if (α ≠ π / 6) then (sin α ≠ 1 / 2) is a necessary but not sufficient condition 
theorem necessary_but_not_sufficient_condition : 
  (α ≠ π / 6) ∧ ¬((α ≠ π / 6) → (sin α ≠ 1 / 2)) ∧ ((sin α ≠ 1 / 2) → (α ≠ π / 6)) :=
by
  sorry

end necessary_but_not_sufficient_condition_l6_6953


namespace ratio_a_b_l6_6582

variables {x y a b : ℝ}

theorem ratio_a_b (h1 : 8 * x - 6 * y = a)
                  (h2 : 12 * y - 18 * x = b)
                  (hx : x ≠ 0)
                  (hy : y ≠ 0)
                  (hb : b ≠ 0) :
  a / b = -4 / 9 :=
sorry

end ratio_a_b_l6_6582


namespace ylona_initial_bands_l6_6678

variable (B J Y : ℕ)  -- Represents the initial number of rubber bands for Bailey, Justine, and Ylona respectively

-- Define the conditions
axiom h1 : J = B + 10
axiom h2 : J = Y - 2
axiom h3 : B - 4 = 8

-- Formulate the statement
theorem ylona_initial_bands : Y = 24 :=
by
  sorry

end ylona_initial_bands_l6_6678


namespace max_necklaces_with_beads_l6_6039

noncomputable def necklace_problem : Prop :=
  ∃ (necklaces : ℕ),
    let green_beads := 200
    let white_beads := 100
    let orange_beads := 50
    let beads_per_pattern_green := 3
    let beads_per_pattern_white := 1
    let beads_per_pattern_orange := 1
    necklaces = orange_beads ∧
    green_beads / beads_per_pattern_green >= necklaces ∧
    white_beads / beads_per_pattern_white >= necklaces ∧
    orange_beads / beads_per_pattern_orange >= necklaces

theorem max_necklaces_with_beads : necklace_problem :=
  sorry

end max_necklaces_with_beads_l6_6039


namespace product_closest_value_l6_6784

theorem product_closest_value (a b : ℝ) (ha : a = 0.000321) (hb : b = 7912000) :
  abs ((a * b) - 2523) < min (abs ((a * b) - 2500)) (min (abs ((a * b) - 2700)) (min (abs ((a * b) - 3100)) (abs ((a * b) - 2000)))) := by
  sorry

end product_closest_value_l6_6784


namespace monotonic_increasing_interval_l6_6613

noncomputable def sqrt (x : ℝ) : ℝ := Real.sqrt x

theorem monotonic_increasing_interval :
  ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 1 → ∀ (x1 x2 : ℝ), 0 ≤ x1 ∧ x1 ≤ 1 → 0 ≤ x2 ∧ x2 ≤ 1 → x1 ≤ x2 → sqrt (- x1 ^ 2 + 2 * x1) ≤ sqrt (- x2 ^ 2 + 2 * x2) :=
sorry

end monotonic_increasing_interval_l6_6613


namespace seventh_observation_is_eight_l6_6982

theorem seventh_observation_is_eight
  (s₆ : ℕ)
  (a₆ : ℕ)
  (s₇ : ℕ)
  (a₇ : ℕ)
  (h₁ : s₆ = 6 * a₆)
  (h₂ : a₆ = 15)
  (h₃ : s₇ = 7 * a₇)
  (h₄ : a₇ = 14) :
  s₇ - s₆ = 8 :=
by
  -- Place proof here
  sorry

end seventh_observation_is_eight_l6_6982


namespace vector_subtraction_l6_6773

-- Define the given vectors
def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (3, 1)

-- State the theorem that the vector subtraction b - a equals (2, -1)
theorem vector_subtraction : b - a = (2, -1) :=
by
  -- Proof is omitted and replaced with sorry
  sorry

end vector_subtraction_l6_6773


namespace number_of_hens_is_50_l6_6998

def number_goats : ℕ := 45
def number_camels : ℕ := 8
def number_keepers : ℕ := 15
def extra_feet : ℕ := 224

def total_heads (number_hens number_goats number_camels number_keepers : ℕ) : ℕ :=
  number_hens + number_goats + number_camels + number_keepers

def total_feet (number_hens number_goats number_camels number_keepers : ℕ) : ℕ :=
  2 * number_hens + 4 * number_goats + 4 * number_camels + 2 * number_keepers

theorem number_of_hens_is_50 (H : ℕ) :
  total_feet H number_goats number_camels number_keepers = (total_heads H number_goats number_camels number_keepers) + extra_feet → H = 50 :=
sorry

end number_of_hens_is_50_l6_6998


namespace solve_quadratic_equation_l6_6494

theorem solve_quadratic_equation :
  ∃ x : ℝ, 2 * x^2 = 4 * x - 1 ∧ (x = (2 + Real.sqrt 2) / 2 ∨ x = (2 - Real.sqrt 2) / 2) :=
by
  sorry

end solve_quadratic_equation_l6_6494


namespace polygon_sides_l6_6790

-- Define the given conditions
def sum_interior_angles (n : ℕ) : ℕ := (n - 2) * 180
def sum_exterior_angles : ℕ := 360

-- Define the theorem
theorem polygon_sides (n : ℕ) (h : sum_interior_angles n = 3 * sum_exterior_angles + 180) : n = 9 :=
sorry

end polygon_sides_l6_6790


namespace denise_removed_bananas_l6_6532

theorem denise_removed_bananas (initial_bananas remaining_bananas : ℕ) 
  (h_initial : initial_bananas = 46) (h_remaining : remaining_bananas = 41) : 
  initial_bananas - remaining_bananas = 5 :=
by
  sorry

end denise_removed_bananas_l6_6532


namespace necessary_but_not_sufficient_condition_l6_6963

noncomputable def condition_sufficiency (m : ℝ) : Prop :=
  ∀ (x : ℝ), x^2 + m*x + 1 > 0

theorem necessary_but_not_sufficient_condition (m : ℝ) : m < 2 → (¬ condition_sufficiency m ∨ condition_sufficiency m) :=
by
  sorry

end necessary_but_not_sufficient_condition_l6_6963


namespace polygon_sides_l6_6707

theorem polygon_sides {R : ℝ} (hR : R > 0) : 
  (∃ n : ℕ, n > 2 ∧ (1/2) * n * R^2 * Real.sin (2 * Real.pi / n) = 4 * R^2) → 
  ∃ n : ℕ, n = 15 :=
by
  sorry

end polygon_sides_l6_6707


namespace num_rectangles_in_5x5_grid_l6_6748

def count_rectangles (n : ℕ) : ℕ :=
  let choose2 := n * (n - 1) / 2
  choose2 * choose2

theorem num_rectangles_in_5x5_grid : count_rectangles 5 = 100 :=
  sorry

end num_rectangles_in_5x5_grid_l6_6748


namespace exponent_identity_l6_6777

variable (x : ℝ) (m n : ℝ)
axiom h1 : x^m = 6
axiom h2 : x^n = 9

theorem exponent_identity : x^(2 * m - n) = 4 :=
by
  sorry

end exponent_identity_l6_6777


namespace problem_statement_l6_6926

theorem problem_statement (m n : ℤ) (h : 3 * m - n = 1) : 9 * m ^ 2 - n ^ 2 - 2 * n = 1 := 
by sorry

end problem_statement_l6_6926


namespace find_B_l6_6419

noncomputable def g (A B C D x : ℝ) : ℝ :=
  A * x^3 + B * x^2 + C * x + D

theorem find_B (A C D : ℝ) (h1 : ∀ x, g A (-2) C D x = A * (x + 2) * (x - 1) * (x - 2)) 
  (h2 : g A (-2) C D 0 = -8) : 
  (-2 : ℝ) = -2 := 
by
  simp [g] at h2
  sorry

end find_B_l6_6419


namespace faster_speed_l6_6317

theorem faster_speed (S : ℝ) (actual_speed : ℝ := 10) (extra_distance : ℝ := 20) (actual_distance : ℝ := 20) :
  actual_distance / actual_speed = (actual_distance + extra_distance) / S → S = 20 :=
by
  sorry

end faster_speed_l6_6317


namespace Bill_threw_more_sticks_l6_6029

-- Definitions based on the given conditions
def Ted_sticks : ℕ := 10
def Ted_rocks : ℕ := 10
def Ted_double_Bill_rocks (R : ℕ) : Prop := Ted_rocks = 2 * R
def Bill_total_objects (S R : ℕ) : Prop := S + R = 21

-- The theorem stating Bill throws 6 more sticks than Ted
theorem Bill_threw_more_sticks (S R : ℕ) (h1 : Ted_double_Bill_rocks R) (h2 : Bill_total_objects S R) : S - Ted_sticks = 6 :=
by
  -- Definitions and conditions are loaded here
  sorry

end Bill_threw_more_sticks_l6_6029


namespace find_other_number_l6_6524

-- Define the conditions
variable (B : ℕ)
variable (HCF : ℕ → ℕ → ℕ)
variable (LCM : ℕ → ℕ → ℕ)

axiom hcf_cond : HCF 24 B = 15
axiom lcm_cond : LCM 24 B = 312

-- The theorem statement
theorem find_other_number (B : ℕ) (HCF : ℕ → ℕ → ℕ) (LCM : ℕ → ℕ → ℕ) 
  (hcf_cond : HCF 24 B = 15) (lcm_cond : LCM 24 B = 312) : 
  B = 195 :=
sorry

end find_other_number_l6_6524


namespace geometric_series_first_term_l6_6999

theorem geometric_series_first_term (r : ℚ) (S : ℚ) (a : ℚ) (h_r : r = 1 / 4) (h_S : S = 80) (h_sum : S = a / (1 - r)) : a = 60 :=
by 
  sorry

end geometric_series_first_term_l6_6999


namespace find_a8_a12_sum_l6_6810

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem find_a8_a12_sum
  (a : ℕ → ℝ) 
  (h_geom : geometric_sequence a) 
  (h1 : a 2 + a 6 = 3) 
  (h2 : a 6 + a 10 = 12) : 
  a 8 + a 12 = 24 :=
sorry

end find_a8_a12_sum_l6_6810


namespace angle_between_hands_at_3_27_l6_6692

noncomputable def minute_hand_angle (m : ℕ) : ℝ :=
  (m / 60.0) * 360.0

noncomputable def hour_hand_angle (h : ℕ) (m : ℕ) : ℝ :=
  ((h + m / 60.0) / 12.0) * 360.0

theorem angle_between_hands_at_3_27 : 
  minute_hand_angle 27 - hour_hand_angle 3 27 = 58.5 :=
by
  rw [minute_hand_angle, hour_hand_angle]
  simp
  sorry

end angle_between_hands_at_3_27_l6_6692


namespace number_of_female_students_l6_6259

theorem number_of_female_students
  (F : ℕ) -- number of female students
  (T : ℕ) -- total number of students
  (h1 : T = F + 8) -- total students = female students + 8 male students
  (h2 : 90 * T = 85 * 8 + 92 * F) -- equation from the sum of scores
  : F = 20 :=
sorry

end number_of_female_students_l6_6259


namespace minimum_value_inequality_l6_6223

theorem minimum_value_inequality (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x * y * z = 64) :
  ∃ x y z, 0 < x ∧ 0 < y ∧ 0 < z ∧ x * y * z = 64 ∧ (x^2 + 8 * x * y + 4 * y^2 + 4 * z^2) = 384 := 
sorry

end minimum_value_inequality_l6_6223


namespace alcohol_percentage_correct_in_mixed_solution_l6_6883

-- Define the ratios of alcohol to water
def ratio_A : ℚ := 21 / 25
def ratio_B : ℚ := 2 / 5

-- Define the mixing ratio of solutions A and B
def mix_ratio_A : ℚ := 5 / 11
def mix_ratio_B : ℚ := 6 / 11

-- Define the function to compute the percentage of alcohol in the mixed solution
def alcohol_percentage_mixed : ℚ := 
  (mix_ratio_A * ratio_A + mix_ratio_B * ratio_B) * 100

-- The theorem to be proven
theorem alcohol_percentage_correct_in_mixed_solution : 
  alcohol_percentage_mixed = 60 :=
by
  sorry

end alcohol_percentage_correct_in_mixed_solution_l6_6883


namespace loraine_total_wax_l6_6552

-- Conditions
def large_animal_wax := 4
def small_animal_wax := 2
def small_animal_count := 12 / small_animal_wax
def large_animal_count := small_animal_count / 3
def total_wax := 12 + (large_animal_count * large_animal_wax)

-- The proof problem
theorem loraine_total_wax : total_wax = 20 := by
  sorry

end loraine_total_wax_l6_6552


namespace car_A_faster_than_car_B_l6_6580

noncomputable def car_A_speed := 
  let t_A1 := 50 / 60 -- time for the first 50 miles at 60 mph
  let t_A2 := 50 / 40 -- time for the next 50 miles at 40 mph
  let t_A := t_A1 + t_A2 -- total time for Car A
  100 / t_A -- average speed of Car A

noncomputable def car_B_speed := 
  let t_B := 1 + (1 / 4) + 1 -- total time for Car B, including a 15-minute stop
  100 / t_B -- average speed of Car B

theorem car_A_faster_than_car_B : car_A_speed > car_B_speed := 
by sorry

end car_A_faster_than_car_B_l6_6580


namespace base8_to_base10_problem_l6_6957

theorem base8_to_base10_problem (c d : ℕ) (h : 543 = 3*8^2 + c*8 + d) : (c * d) / 12 = 5 / 4 :=
by 
  sorry

end base8_to_base10_problem_l6_6957


namespace mean_absolute_temperature_correct_l6_6514

noncomputable def mean_absolute_temperature (temps : List ℝ) : ℝ :=
  (temps.map (λ x => |x|)).sum / temps.length

theorem mean_absolute_temperature_correct :
  mean_absolute_temperature [-6, -3, -3, -6, 0, 4, 3] = 25 / 7 :=
by
  sorry

end mean_absolute_temperature_correct_l6_6514


namespace union_A_B_eq_A_union_B_l6_6696

def A : Set ℝ := { x | -1 ≤ x ∧ x ≤ 4 }
def B : Set ℝ := { x | x > 3 / 2 }

theorem union_A_B_eq_A_union_B :
  (A ∪ B) = { x | -1 ≤ x } :=
by
  sorry

end union_A_B_eq_A_union_B_l6_6696


namespace new_average_after_exclusion_l6_6952

theorem new_average_after_exclusion (S : ℕ) (h1 : S = 27 * 5) (excluded : ℕ) (h2 : excluded = 35) : (S - excluded) / 4 = 25 :=
by
  sorry

end new_average_after_exclusion_l6_6952


namespace solve_inequality1_solve_inequality_system_l6_6816

-- Define the first condition inequality
def inequality1 (x : ℝ) : Prop := 
  (2 * x - 1) / 3 - (9 * x + 2) / 6 ≤ 1

-- Theorem for the first inequality proving x >= -2
theorem solve_inequality1 {x : ℝ} (h : inequality1 x) : x ≥ -2 := 
sorry

-- Define the first condition for the system of inequalities
def inequality2 (x : ℝ) : Prop := 
  x - 3 * (x - 2) ≥ 4

-- Define the second condition for the system of inequalities
def inequality3 (x : ℝ) : Prop := 
  (2 * x - 1) / 5 < (x + 1) / 2

-- Theorem for the system of inequalities proving -7 < x ≤ 1
theorem solve_inequality_system {x : ℝ} (h1 : inequality2 x) (h2 : inequality3 x) : -7 < x ∧ x ≤ 1 := 
sorry

end solve_inequality1_solve_inequality_system_l6_6816


namespace divisible_by_72_l6_6996

theorem divisible_by_72 (a b : ℕ) (h1 : 0 ≤ a ∧ a < 10) (h2 : 0 ≤ b ∧ b < 10) :
  (b = 2 ∧ a = 3) → (a * 10000 + 6 * 1000 + 7 * 100 + 9 * 10 + b) % 72 = 0 :=
by
  sorry

end divisible_by_72_l6_6996


namespace rectangle_area_l6_6022

theorem rectangle_area :
  ∃ (L B : ℝ), (L - B = 23) ∧ (2 * (L + B) = 206) ∧ (L * B = 2520) :=
sorry

end rectangle_area_l6_6022


namespace initial_spinach_volume_l6_6120

theorem initial_spinach_volume (S : ℝ) (h1 : 0.20 * S + 6 + 4 = 18) : S = 40 :=
by
  sorry

end initial_spinach_volume_l6_6120


namespace isosceles_triangle_area_l6_6635

theorem isosceles_triangle_area (a b h : ℝ) (h_eq : h = a / (2 * Real.sqrt 3)) :
  (1 / 2 * a * h) = (a^2 * Real.sqrt 3) / 12 :=
by
  -- Define the necessary parameters and conditions
  let area := (1 / 2) * a * h
  have h := h_eq
  -- Substitute and prove the calculated area
  sorry

end isosceles_triangle_area_l6_6635


namespace fourth_term_arithmetic_sequence_l6_6755

theorem fourth_term_arithmetic_sequence (a d : ℝ) (h : 2 * a + 2 * d = 12) : a + d = 6 := 
by
  sorry

end fourth_term_arithmetic_sequence_l6_6755


namespace dave_final_tickets_l6_6165

-- Define the initial number of tickets and operations
def initial_tickets : ℕ := 25
def tickets_spent_on_beanie : ℕ := 22
def tickets_won_after : ℕ := 15

-- Define the final number of tickets function
def final_tickets (initial : ℕ) (spent : ℕ) (won : ℕ) : ℕ :=
  initial - spent + won

-- Theorem stating that Dave would end up with 18 tickets given the conditions
theorem dave_final_tickets : final_tickets initial_tickets tickets_spent_on_beanie tickets_won_after = 18 :=
by
  -- Proof to be filled in
  sorry

end dave_final_tickets_l6_6165


namespace line_exists_symmetric_diagonals_l6_6991

-- Define the initial conditions
def Circle_C1 (x y : ℝ) : Prop := x^2 + y^2 + 6 * x = 0
def Line_l1 (x y : ℝ) : Prop := y = 2 * x + 1

-- Define the symmetric circle C about the line l1
def Symmetric_Circle (x y : ℝ) : Prop := (x - 1)^2 + (y + 2)^2 = 9

-- Define the origion and intersection points
def Point_O : (ℝ × ℝ) := (0, 0)
def Point_Intersection (l : ℝ → ℝ) (A B : ℝ × ℝ) : Prop := ∃ x_A y_A x_B y_B : ℝ,
  l x_A = y_A ∧ l x_B = y_B ∧ Symmetric_Circle x_A y_A ∧ Symmetric_Circle x_B y_B

-- Define diagonal equality condition
def Diagonals_Equal (O A S B : ℝ × ℝ) : Prop := 
  let (xO, yO) := O
  let (xA, yA) := A
  let (xS, yS) := S
  let (xB, yB) := B
  (xA - xO)^2 + (yA - yO)^2 = (xB - xS)^2 + (yB - yS)^2

-- Prove existence of line where diagonals are equal and find the equation
theorem line_exists_symmetric_diagonals :
  ∃ l : ℝ → ℝ, (l (-1) = 0) ∧
    (∃ (A B S : ℝ × ℝ), Point_Intersection l A B ∧ Diagonals_Equal Point_O A S B) ∧
    (∀ x : ℝ, l x = x + 1) :=
by
  sorry

end line_exists_symmetric_diagonals_l6_6991


namespace at_least_one_variety_has_27_apples_l6_6206

theorem at_least_one_variety_has_27_apples (total_apples : ℕ) (varieties : ℕ) 
  (h_total : total_apples = 105) (h_varieties : varieties = 4) : 
  ∃ v : ℕ, v ≥ 27 := 
sorry

end at_least_one_variety_has_27_apples_l6_6206


namespace agent_commission_calculation_l6_6169

-- Define the conditions
def total_sales : ℝ := 250
def commission_rate : ℝ := 0.05

-- Define the commission calculation function
def calculate_commission (sales : ℝ) (rate : ℝ) : ℝ :=
  sales * rate

-- Proposition stating the desired commission
def agent_commission_is_correct : Prop :=
  calculate_commission total_sales commission_rate = 12.5

-- State the proof problem
theorem agent_commission_calculation : agent_commission_is_correct :=
by sorry

end agent_commission_calculation_l6_6169


namespace functional_equation_solution_l6_6843

theorem functional_equation_solution :
  ∃ f : ℝ → ℝ,
  (f 1 = 1 ∧ (∀ x y : ℝ, f (x * y + f x) = x * f y + f x)) ∧ f (1/2) = 1/2 :=
by
  sorry

end functional_equation_solution_l6_6843


namespace max_A_min_A_l6_6445

-- Define the problem and its conditions and question

def A_max (B : ℕ) (h1 : B > 22222222) (h2 : (B / 100000000 : ℕ) ≠ 0) (h3 : Nat.gcd B 18 = 1) : ℕ :=
  let b := B % 10
  10^8 * b + (B - b) / 10

def A_min (B : ℕ) (h1 : B > 22222222) (h2 : (B / 100000000 : ℕ) ≠ 0) (h3 : Nat.gcd B 18 = 1) : ℕ :=
  let b := B % 10
  10^8 * b + (B - b) / 10

theorem max_A (B : ℕ) (h1 : B > 22222222) (h2 : (B / 100000000 : ℕ) ≠ 0) (h3 : Nat.gcd B 18 = 1) :
  A_max B h1 h2 h3 = 999999998 := sorry

theorem min_A (B : ℕ) (h1 : B > 22222222) (h2 : (B / 100000000 : ℕ) ≠ 0) (h3 : Nat.gcd B 18 = 1) :
  A_min B h1 h2 h3 = 122222224 := sorry

end max_A_min_A_l6_6445


namespace log_tan_ratio_l6_6366

noncomputable def sin_add (α β : ℝ) : ℝ := Real.sin (α + β)
noncomputable def sin_sub (α β : ℝ) : ℝ := Real.sin (α - β)
noncomputable def tan_ratio (α β : ℝ) : ℝ := Real.tan α / Real.tan β

theorem log_tan_ratio (α β : ℝ)
  (h1 : sin_add α β = 1 / 2)
  (h2 : sin_sub α β = 1 / 3) :
  Real.logb 5 (tan_ratio α β) = 1 := by
sorry

end log_tan_ratio_l6_6366


namespace total_days_2010_to_2013_l6_6205

theorem total_days_2010_to_2013 :
  let year2010_days := 365
  let year2011_days := 365
  let year2012_days := 366
  let year2013_days := 365
  year2010_days + year2011_days + year2012_days + year2013_days = 1461 := by
  sorry

end total_days_2010_to_2013_l6_6205


namespace aftershave_lotion_volume_l6_6703

theorem aftershave_lotion_volume (V : ℝ) (h1 : 0.30 * V = 0.1875 * (V + 30)) : V = 50 := 
by 
-- sorry is added to indicate proof is omitted.
sorry

end aftershave_lotion_volume_l6_6703


namespace average_first_three_numbers_l6_6470

theorem average_first_three_numbers (A B C D : ℝ) 
  (hA : A = 33) 
  (hD : D = 18)
  (hBCD : (B + C + D) / 3 = 15) : 
  (A + B + C) / 3 = 20 := 
by 
  sorry

end average_first_three_numbers_l6_6470


namespace qiuqiu_servings_l6_6537

-- Define the volume metrics
def bottles : ℕ := 1
def cups_per_bottle_kangkang : ℕ := 4
def foam_expansion : ℕ := 3
def foam_fraction : ℚ := 1 / 2

-- Calculate the effective cup volume under Qiuqiu's serving method
def beer_fraction_per_cup_qiuqiu : ℚ := 1 / 2 + (1 / foam_expansion) * foam_fraction

-- Calculate the number of cups Qiuqiu can serve from one bottle
def qiuqiu_cups_from_bottle : ℚ := cups_per_bottle_kangkang / beer_fraction_per_cup_qiuqiu

-- The theorem statement
theorem qiuqiu_servings :
  qiuqiu_cups_from_bottle = 6 := by
  sorry

end qiuqiu_servings_l6_6537


namespace area_fraction_of_rhombus_in_square_l6_6943

theorem area_fraction_of_rhombus_in_square :
  let n := 7                 -- grid size
  let side_length := n - 1   -- side length of the square
  let square_area := side_length^2 -- area of the square
  let rhombus_side := Real.sqrt 2 -- side length of the rhombus
  let rhombus_area := 2      -- area of the rhombus
  (rhombus_area / square_area) = 1 / 18 := sorry

end area_fraction_of_rhombus_in_square_l6_6943


namespace one_percent_as_decimal_l6_6403

theorem one_percent_as_decimal : (1 / 100 : ℝ) = 0.01 := by
  sorry

end one_percent_as_decimal_l6_6403


namespace diagonal_of_rectangular_prism_l6_6712

noncomputable def diagonal_length (l w h : ℝ) : ℝ :=
  Real.sqrt (l^2 + w^2 + h^2)

theorem diagonal_of_rectangular_prism :
  diagonal_length 15 25 20 = 25 * Real.sqrt 2 :=
by
  sorry

end diagonal_of_rectangular_prism_l6_6712


namespace union_sets_l6_6254

def setA : Set ℝ := { x | -1 ≤ x ∧ x < 3 }

def setB : Set ℝ := { x | x^2 - 7 * x + 10 ≤ 0 }

theorem union_sets : setA ∪ setB = { x | -1 ≤ x ∧ x ≤ 5 } :=
by
  sorry

end union_sets_l6_6254


namespace triple_root_possible_values_l6_6586

-- Definitions and conditions
def polynomial (x : ℤ) (b3 b2 b1 : ℤ) := x^4 + b3 * x^3 + b2 * x^2 + b1 * x + 24

-- The proof problem
theorem triple_root_possible_values 
  (r b3 b2 b1 : ℤ)
  (h_triple_root : polynomial r b3 b2 b1 = (x * (x - 1) * (x - 2)) * (x - r) ) :
  r = -2 ∨ r = -1 ∨ r = 1 ∨ r = 2 :=
by
  sorry

end triple_root_possible_values_l6_6586


namespace paint_house_l6_6136

theorem paint_house (n s h : ℕ) (h_pos : 0 < h)
    (rate_eq : ∀ (x : ℕ), 0 < x → ∃ t : ℕ, x * t = n * h) :
    (n + s) * (nh / (n + s)) = n * h := 
sorry

end paint_house_l6_6136


namespace range_of_m_l6_6857

theorem range_of_m (x1 x2 m : Real) (h_eq : ∀ x : Real, x^2 - 2*x + m + 2 = 0)
  (h_abs : |x1| + |x2| ≤ 3)
  (h_real : ∀ x : Real, ∃ y : Real, x^2 - 2*x + m + 2 = 0) : -13 / 4 ≤ m ∧ m ≤ -1 :=
by
  sorry

end range_of_m_l6_6857


namespace side_length_of_square_l6_6152

theorem side_length_of_square (s : ℝ) (h : s^2 = 6 * (4 * s)) : s = 24 :=
by sorry

end side_length_of_square_l6_6152


namespace total_fruit_in_buckets_l6_6778

theorem total_fruit_in_buckets (A B C : ℕ) 
  (h1 : A = B + 4)
  (h2 : B = C + 3)
  (h3 : C = 9) :
  A + B + C = 37 := by
  sorry

end total_fruit_in_buckets_l6_6778


namespace sandy_hours_per_day_l6_6033

theorem sandy_hours_per_day (total_hours : ℕ) (days : ℕ) (H : total_hours = 45 ∧ days = 5) : total_hours / days = 9 :=
by
  sorry

end sandy_hours_per_day_l6_6033


namespace inequality_proof_l6_6193

theorem inequality_proof (x y z : ℝ) (hx : -1 < x) (hy : -1 < y) (hz : -1 < z) :
    (1 + x^2) / (1 + y + z^2) + (1 + y^2) / (1 + z + x^2) + (1 + z^2) / (1 + x + y^2) ≥ 2 :=
sorry

end inequality_proof_l6_6193


namespace steve_speed_back_home_l6_6674

-- Definitions based on conditions
def distance := 20 -- distance from house to work in km
def total_time := 6 -- total time on the road in hours
def speed_to_work (v : ℝ) := v -- speed to work in km/h
def speed_back_home (v : ℝ) := 2 * v -- speed back home in km/h

-- Theorem to assert the proof
theorem steve_speed_back_home (v : ℝ) (h : distance / v + distance / (2 * v) = total_time) :
  speed_back_home v = 10 := by
  -- Proof goes here but we just state sorry to skip it
  sorry

end steve_speed_back_home_l6_6674


namespace female_democrats_count_l6_6409

variable (F M : ℕ)
def total_participants : Prop := F + M = 720
def female_democrats (D_F : ℕ) : Prop := D_F = 1 / 2 * F
def male_democrats (D_M : ℕ) : Prop := D_M = 1 / 4 * M
def total_democrats (D_F D_M : ℕ) : Prop := D_F + D_M = 1 / 3 * 720

theorem female_democrats_count
  (F M D_F D_M : ℕ)
  (h1 : total_participants F M)
  (h2 : female_democrats F D_F)
  (h3 : male_democrats M D_M)
  (h4 : total_democrats D_F D_M) :
  D_F = 120 :=
sorry

end female_democrats_count_l6_6409


namespace lines_intersection_points_l6_6955

theorem lines_intersection_points :
  let line1 (x y : ℝ) := 2 * y - 3 * x = 4
  let line2 (x y : ℝ) := 3 * x + y = 5
  let line3 (x y : ℝ) := 6 * x - 4 * y = 8
  ∃ p1 p2 : (ℝ × ℝ),
    (line1 p1.1 p1.2 ∧ line2 p1.1 p1.2) ∧
    (line2 p2.1 p2.2 ∧ line3 p2.1 p2.2) ∧
    (p1 = (2, 5)) ∧ (p2 = (14/9, 1/3)) :=
by
  sorry

end lines_intersection_points_l6_6955


namespace complex_div_i_l6_6053

open Complex

theorem complex_div_i (z : ℂ) (hz : z = -2 - i) : z / i = -1 + 2 * i :=
by
  sorry

end complex_div_i_l6_6053


namespace function_no_real_zeros_l6_6615

variable (a b c : ℝ)

-- Conditions: a, b, c form a geometric sequence and ac > 0
def geometric_sequence (a b c : ℝ) : Prop := b^2 = a * c
def positive_product (a c : ℝ) : Prop := a * c > 0

theorem function_no_real_zeros (h_geom : geometric_sequence a b c) (h_pos : positive_product a c) :
  ∀ x : ℝ, a * x^2 + b * x + c ≠ 0 := 
by
  sorry

end function_no_real_zeros_l6_6615


namespace remaining_customers_after_some_left_l6_6554

-- Define the initial conditions and question (before proving it)
def initial_customers := 8
def new_customers := 99
def total_customers_after_new := 104

-- Define the hypothesis based on the total customers after new customers added
theorem remaining_customers_after_some_left (x : ℕ) (h : x + new_customers = total_customers_after_new) : x = 5 :=
by {
  -- Proof omitted
  sorry
}

end remaining_customers_after_some_left_l6_6554


namespace find_constants_l6_6785

theorem find_constants (P Q R : ℚ) 
  (h : ∀ x : ℚ, x ≠ 4 → x ≠ 2 → (5 * x + 1) / ((x - 4) * (x - 2) ^ 2) = P / (x - 4) + Q / (x - 2) + R / (x - 2)^2) :
  P = 21 / 4 ∧ Q = 15 ∧ R = -11 / 2 :=
by
  sorry

end find_constants_l6_6785


namespace probability_of_two_eights_l6_6683

-- Define a function that calculates the factorial of a number
noncomputable def fact (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * fact (n - 1)

-- Definition of binomial coefficient
noncomputable def binom (n k : ℕ) : ℕ :=
  fact n / (fact k * fact (n - k))

-- Probability of exactly two dice showing 8 out of eight 8-sided dice
noncomputable def prob_exactly_two_eights : ℚ :=
  binom 8 2 * ((1 / 8 : ℚ) ^ 2) * ((7 / 8 : ℚ) ^ 6)

-- Main theorem statement
theorem probability_of_two_eights :
  prob_exactly_two_eights = 0.196 := by
  sorry

end probability_of_two_eights_l6_6683


namespace max_possible_N_l6_6436

theorem max_possible_N (cities roads N : ℕ) (h1 : cities = 1000) (h2 : roads = 2017) (h3 : N = roads - (cities - 1 + 7 - 1)) :
  N = 1009 :=
by {
  sorry
}

end max_possible_N_l6_6436


namespace intersection_M_N_l6_6410

def M := { x : ℝ | x < 2011 }
def N := { x : ℝ | 0 < x ∧ x < 1 }

theorem intersection_M_N :
  M ∩ N = { x : ℝ | 0 < x ∧ x < 1 } := 
by 
  sorry

end intersection_M_N_l6_6410


namespace average_length_of_two_strings_l6_6406

theorem average_length_of_two_strings (a b : ℝ) (h1 : a = 3.2) (h2 : b = 4.8) :
  (a + b) / 2 = 4.0 :=
by
  sorry

end average_length_of_two_strings_l6_6406


namespace some_number_is_ten_l6_6177

theorem some_number_is_ten (x : ℕ) (h : 5 ^ 29 * 4 ^ 15 = 2 * x ^ 29) : x = 10 :=
by
  sorry

end some_number_is_ten_l6_6177


namespace num_ways_placing_2015_bishops_l6_6004

-- Define the concept of placing bishops on a 2 x n chessboard without mutual attacks
def max_bishops (n : ℕ) : ℕ := n

-- Define the calculation of the number of ways to place these bishops
def num_ways_to_place_bishops (n : ℕ) : ℕ := 2 ^ n

-- The proof statement for our specific problem
theorem num_ways_placing_2015_bishops :
  num_ways_to_place_bishops 2015 = 2 ^ 2015 :=
by
  sorry

end num_ways_placing_2015_bishops_l6_6004


namespace intersection_A_B_l6_6715

-- Definitions of sets A and B
def A : Set ℕ := {2, 3, 5, 7}
def B : Set ℕ := {1, 2, 3, 5, 8}

-- Prove that the intersection of sets A and B is {2, 3, 5}
theorem intersection_A_B :
  A ∩ B = {2, 3, 5} :=
sorry

end intersection_A_B_l6_6715


namespace lacrosse_more_than_football_l6_6807

-- Define the constants and conditions
def total_bottles := 254
def football_players := 11
def bottles_per_football_player := 6
def soccer_bottles := 53
def rugby_bottles := 49

-- Calculate the number of bottles needed by each team
def football_bottles := football_players * bottles_per_football_player
def other_teams_bottles := football_bottles + soccer_bottles + rugby_bottles
def lacrosse_bottles := total_bottles - other_teams_bottles

-- The theorem to be proven
theorem lacrosse_more_than_football : lacrosse_bottles - football_bottles = 20 :=
by
  sorry

end lacrosse_more_than_football_l6_6807


namespace radius_increase_by_100_percent_l6_6109

theorem radius_increase_by_100_percent (A A' r r' : ℝ) (π : ℝ)
  (h1 : A = π * r^2) -- initial area of the circle
  (h2 : A' = 4 * A) -- new area is 4 times the original area
  (h3 : A' = π * r'^2) -- new area formula with new radius
  : r' = 2 * r :=
by
  sorry

end radius_increase_by_100_percent_l6_6109


namespace propP_necessary_but_not_sufficient_l6_6186

open Function Real

variable (f : ℝ → ℝ)

-- Conditions: differentiable function f and the proposition Q
def diff_and_propQ (h_deriv : Differentiable ℝ f) : Prop :=
∀ x : ℝ, abs (deriv f x) < 2018

-- Proposition P
def propP : Prop :=
∀ x1 x2 : ℝ, x1 ≠ x2 → abs ((f x1 - f x2) / (x1 - x2)) < 2018

-- Final statement
theorem propP_necessary_but_not_sufficient (h_deriv : Differentiable ℝ f) (hQ : diff_and_propQ f h_deriv) : 
  (∀ x1 x2 : ℝ, x1 ≠ x2 → abs ((f x1 - f x2) / (x1 - x2)) < 2018) ∧ 
  ¬(∀ x : ℝ, abs (deriv f x) < 2018 ↔ ∀ x1 x2 : ℝ, x1 ≠ x2 → abs ((f x1 - f x2) / (x1 - x2)) < 2018) :=
by
  sorry

end propP_necessary_but_not_sufficient_l6_6186


namespace probability_of_pink_l6_6133

theorem probability_of_pink (B P : ℕ) (h1 : (B : ℚ) / (B + P) = 6 / 7) (h2 : (B^2 : ℚ) / (B + P)^2 = 36 / 49) : 
  (P : ℚ) / (B + P) = 1 / 7 :=
by
  sorry

end probability_of_pink_l6_6133


namespace fraction_sum_of_roots_l6_6852

theorem fraction_sum_of_roots (x1 x2 : ℝ) (h1 : 5 * x1^2 - 3 * x1 - 2 = 0) (h2 : 5 * x2^2 - 3 * x2 - 2 = 0) (hx : x1 ≠ x2) :
  (1 / x1 + 1 / x2 = -3 / 2) :=
by
  sorry

end fraction_sum_of_roots_l6_6852


namespace compare_abc_l6_6612

noncomputable def a : ℝ := 4 ^ (Real.log 2 / Real.log 3)
noncomputable def b : ℝ := 4 ^ (Real.log 6 / (2 * Real.log 3))
noncomputable def c : ℝ := 2 ^ (Real.sqrt 5)

theorem compare_abc : c > b ∧ b > a := 
by
  sorry

end compare_abc_l6_6612


namespace coplanar_lines_l6_6619

noncomputable def line1 (s k : ℝ) : ℝ × ℝ × ℝ :=
  (2 + s, 5 - k * s, 3 + k * s)

noncomputable def line2 (t : ℝ) : ℝ × ℝ × ℝ :=
  (2 * t, 4 + 2 * t, 6 - 2 * t)

theorem coplanar_lines (k : ℝ) :
  (exists s t : ℝ, line1 s k = line2 t) ∨ line1 1 k = (1, -k, k) ∧ line2 1 = (2, 2, -2) → k = -1 :=
by sorry

end coplanar_lines_l6_6619


namespace cards_arrangement_count_is_10_l6_6130

-- Define the problem in Lean statement terms
def valid_arrangements_count : ℕ :=
  -- number of arrangements of seven cards where one card can be removed 
  -- leaving the remaining six cards in either ascending or descending order
  10

-- Theorem stating that the number of valid arrangements is 10
theorem cards_arrangement_count_is_10 : valid_arrangements_count = 10 :=
by
  -- Proof is omitted (the explanation above corresponds to the omitted proof details)
  sorry

end cards_arrangement_count_is_10_l6_6130


namespace original_number_is_16_l6_6768

theorem original_number_is_16 (x : ℤ) (h1 : 3 * (2 * x + 5) = 111) : x = 16 :=
by
  sorry

end original_number_is_16_l6_6768


namespace probability_event_B_l6_6202

-- Define the type of trial outcomes, we're considering binary outcomes for simplicity
inductive Outcome
| win : Outcome
| lose : Outcome

open Outcome

def all_possible_outcomes := [
  [win, win, win],
  [win, win, win, lose],
  [win],
  [win],
  [lose],
  [win, win, lose, lose],
  [win, lose],
  [win, lose, win, lose, win],
  [win],
  [lose],
  [lose],
  [lose],
  [lose, win, win],
  [win, lose, lose, win],
  [lose, win, lose, lose],
  [win],
  [win],
  [lose],
  [lose],
  [lose, lose],
  [lose],
  [lose],
  [],
  [lose, lose, lose, lose]
]

-- Event A is winning a prize
def event_A := [
  [win, win, win],
  [win, win, win, lose],
  [win, win, lose, lose],
  [win, lose, win, lose, win],
  [win, lose, lose, win]
]

-- Event B is satisfying the condition \(a + b + c + d \leq 2\)
def event_B := [
  [lose],
  [win, lose],
  [lose, win],
  [win],
  [lose, lose],
  [lose, win, lose],
  [lose, lose, win],
  [lose, win, win],
  [win, lose, lose],
  [lose, lose, lose],
  []
]

-- Proof that the probability of event B equals 11/16
theorem probability_event_B : (event_B.length / all_possible_outcomes.length) = 11 / 16 := by
  sorry

end probability_event_B_l6_6202


namespace num_people_is_8_l6_6669

-- Define the known conditions
def bill_amt : ℝ := 314.16
def person_amt : ℝ := 34.91
def total_amt : ℝ := 314.19

-- Prove that the number of people is 8
theorem num_people_is_8 : ∃ num_people : ℕ, num_people = total_amt / person_amt ∧ num_people = 8 :=
by
  sorry

end num_people_is_8_l6_6669


namespace rectangle_width_l6_6283

-- The Lean statement only with given conditions and the final proof goal
theorem rectangle_width (w l : ℕ) (P : ℕ) (h1 : l = w - 3) (h2 : P = 2 * w + 2 * l) (h3 : P = 54) :
  w = 15 :=
by
  sorry

end rectangle_width_l6_6283


namespace find_f_2024_l6_6713

noncomputable def f (x : ℝ) (a b c : ℝ) : ℝ := x^3 + a * x^2 + b * x + c

theorem find_f_2024 (a b c : ℝ)
  (h1 : f 2021 a b c = 2021)
  (h2 : f 2022 a b c = 2022)
  (h3 : f 2023 a b c = 2023) :
  f 2024 a b c = 2030 := sorry

end find_f_2024_l6_6713


namespace line_AB_l6_6880

-- Statements for circles and intersection
def circle_C1 (x y: ℝ) : Prop := x^2 + y^2 = 1
def circle_C2 (x y: ℝ) : Prop := (x - 1)^2 + (y + 1)^2 = 1

-- Points A and B are defined as the intersection points of circles C1 and C2
axiom A (x y: ℝ) : circle_C1 x y ∧ circle_C2 x y
axiom B (x y: ℝ) : circle_C1 x y ∧ circle_C2 x y

-- The goal is to prove that the line passing through points A and B has the equation x - y = 0
theorem line_AB (x y: ℝ) : circle_C1 x y → circle_C2 x y → (x - y = 0) :=
by
  sorry

end line_AB_l6_6880


namespace power_subtraction_l6_6134

theorem power_subtraction : 2^4 - 2^3 = 2^3 := by
  sorry

end power_subtraction_l6_6134


namespace map_distance_ratio_l6_6867

theorem map_distance_ratio (actual_distance_km : ℝ) (map_distance_cm : ℝ) (h_actual_distance : actual_distance_km = 5) (h_map_distance : map_distance_cm = 2) :
  map_distance_cm / (actual_distance_km * 100000) = 1 / 250000 :=
by
  -- Given the actual distance in kilometers and map distance in centimeters, prove the scale ratio
  -- skip the proof
  sorry

end map_distance_ratio_l6_6867


namespace parallel_vectors_implies_x_value_l6_6593

variable (x : ℝ)

def vec_a : ℝ × ℝ := (1, 2)
def vec_b (x : ℝ) : ℝ × ℝ := (x, 1)
def scalar_mul (c : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (c * v.1, c * v.2)
def vec_add (v1 v2 : ℝ × ℝ) : ℝ × ℝ := (v1.1 + v2.1, v1.2 + v2.2)
def vec_sub (v1 v2 : ℝ × ℝ) : ℝ × ℝ := (v1.1 - v2.1, v1.2 - v2.2)

theorem parallel_vectors_implies_x_value :
  (∃ k : ℝ, vec_add vec_a (scalar_mul 2 (vec_b x)) = scalar_mul k (vec_sub (scalar_mul 2 vec_a) (scalar_mul 2 (vec_b x)))) →
  x = 1 / 2 :=
by
  sorry

end parallel_vectors_implies_x_value_l6_6593


namespace car_speed_second_hour_l6_6075

theorem car_speed_second_hour
  (v1 : ℕ) (avg_speed : ℕ) (time : ℕ) (v2 : ℕ)
  (h1 : v1 = 90)
  (h2 : avg_speed = 70)
  (h3 : time = 2) :
  v2 = 50 :=
by
  sorry

end car_speed_second_hour_l6_6075


namespace nested_function_evaluation_l6_6694

def f (x : ℕ) : ℕ := x + 3
def g (x : ℕ) : ℕ := x / 2
def f_inv (x : ℕ) : ℕ := x - 3
def g_inv (x : ℕ) : ℕ := 2 * x

theorem nested_function_evaluation : 
  f (g_inv (f_inv (g_inv (f_inv (g (f 15)))))) = 21 := 
by 
  sorry

end nested_function_evaluation_l6_6694


namespace house_height_l6_6257

theorem house_height
  (tree_height : ℕ) (tree_shadow : ℕ)
  (house_shadow : ℕ) (h : ℕ) :
  tree_height = 15 →
  tree_shadow = 18 →
  house_shadow = 72 →
  (h / tree_height) = (house_shadow / tree_shadow) →
  h = 60 :=
by
  intros h1 h2 h3 h4
  have h5 : h / 15 = 72 / 18 := by
    rw [h1, h2, h3] at h4
    exact h4
  sorry

end house_height_l6_6257


namespace log_sum_range_l6_6833

theorem log_sum_range (x y : ℝ) (hx_pos : x > 0) (hy_pos : y > 0) (hx_ne_one : x ≠ 1) (hy_ne_one : y ≠ 1) :
  (Real.log y / Real.log x + Real.log x / Real.log y) ∈ Set.union (Set.Iic (-2)) (Set.Ici 2) :=
sorry

end log_sum_range_l6_6833


namespace intersection_proof_complement_proof_range_of_m_condition_l6_6115

def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 4}
def B : Set ℝ := {x | -3 < x ∧ x < 1}
def C (m : ℝ) : Set ℝ := {x | 2 - m ≤ x ∧ x ≤ 2 + m}

theorem intersection_proof : A ∩ B = {x | -2 ≤ x ∧ x < 1} := sorry

theorem complement_proof : (Set.univ \ B) = {x | x ≤ -3 ∨ x ≥ 1} := sorry

theorem range_of_m_condition (m : ℝ) : (A ∪ C m = A) → (m ≤ 2) := sorry

end intersection_proof_complement_proof_range_of_m_condition_l6_6115


namespace gcd_390_455_546_l6_6387

theorem gcd_390_455_546 :
  Nat.gcd (Nat.gcd 390 455) 546 = 13 := 
sorry

end gcd_390_455_546_l6_6387


namespace remainder_3_pow_19_mod_10_l6_6140

theorem remainder_3_pow_19_mod_10 : (3^19) % 10 = 7 := 
by 
  sorry

end remainder_3_pow_19_mod_10_l6_6140


namespace quadratic_inequality_range_l6_6956

theorem quadratic_inequality_range (a : ℝ) :
  (∀ x : ℝ, (a - 2) * x^2 + 2 * (a - 2) * x - 4 < 0) ↔ a ∈ Set.Ioc (-2 : ℝ) 2 :=
sorry

end quadratic_inequality_range_l6_6956


namespace y_coord_vertex_C_l6_6941

/-- The coordinates of vertices A, B, and D are given as A(0,0), B(0,1), and D(3,1).
 Vertex C is directly above vertex B. The quadrilateral ABCD has a vertical line of symmetry 
 and the area of quadrilateral ABCD is 18 square units.
 Prove that the y-coordinate of vertex C is 11. -/
theorem y_coord_vertex_C (h : ℝ) 
  (A : ℝ × ℝ := (0, 0)) 
  (B : ℝ × ℝ := (0, 1)) 
  (D : ℝ × ℝ := (3, 1)) 
  (C : ℝ × ℝ := (0, h)) 
  (symmetry : C.fst = B.fst) 
  (area : 18 = 3 * 1 + (1 / 2) * 3 * (h - 1)) :
  h = 11 := 
by
  sorry

end y_coord_vertex_C_l6_6941


namespace superchess_no_attacks_l6_6658

open Finset

theorem superchess_no_attacks (board_size : ℕ) (num_pieces : ℕ)  (attack_limit : ℕ) 
  (h_board_size : board_size = 100) (h_num_pieces : num_pieces = 20) 
  (h_attack_limit : attack_limit = 20) : 
  ∃ (placements : Finset (ℕ × ℕ)), placements.card = num_pieces ∧
  ∀ {p1 p2 : ℕ × ℕ}, p1 ≠ p2 → p1 ∈ placements → p2 ∈ placements → 
  ¬(∃ (attack_positions : Finset (ℕ × ℕ)), attack_positions.card ≤ attack_limit ∧ 
  ∃ piece_pos : ℕ × ℕ, piece_pos ∈ placements ∧ attack_positions ⊆ placements ∧ p1 ∈ attack_positions ∧ p2 ∈ attack_positions) :=
sorry

end superchess_no_attacks_l6_6658


namespace distance_from_dormitory_to_city_l6_6798

theorem distance_from_dormitory_to_city (D : ℝ)
  (h1 : (1 / 5) * D + (2 / 3) * D + 4 = D) : D = 30 := by
  sorry

end distance_from_dormitory_to_city_l6_6798


namespace Melanie_gumballs_sale_l6_6146

theorem Melanie_gumballs_sale (gumballs : ℕ) (price_per_gumball : ℕ) (total_price : ℕ) :
  gumballs = 4 →
  price_per_gumball = 8 →
  total_price = gumballs * price_per_gumball →
  total_price = 32 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  assumption

end Melanie_gumballs_sale_l6_6146


namespace pos_sol_eq_one_l6_6179

theorem pos_sol_eq_one (n : ℕ) (hn : 1 < n) :
  ∀ x : ℝ, 0 < x → (x ^ n - n * x + n - 1 = 0) → x = 1 := by
  -- The proof goes here
  sorry

end pos_sol_eq_one_l6_6179


namespace find_factor_l6_6214

-- Defining the given conditions
def original_number : ℕ := 7
def resultant (x: ℕ) : ℕ := 2 * x + 9
def condition (x f: ℕ) : Prop := (resultant x) * f = 69

-- The problem statement
theorem find_factor : ∃ f: ℕ, condition original_number f ∧ f = 3 :=
by
  sorry

end find_factor_l6_6214


namespace polygon_sides_l6_6847

theorem polygon_sides (x : ℝ) (hx : 0 < x) (h : x + 5 * x = 180) : 12 = 360 / x :=
by {
  -- Steps explaining: x should be the exterior angle then proof follows.
  sorry
}

end polygon_sides_l6_6847


namespace total_crew_members_l6_6055

def num_islands : ℕ := 3
def ships_per_island : ℕ := 12
def crew_per_ship : ℕ := 24

theorem total_crew_members : num_islands * ships_per_island * crew_per_ship = 864 := by
  sorry

end total_crew_members_l6_6055


namespace roots_expression_value_l6_6008

theorem roots_expression_value (x1 x2 : ℝ) (h1 : x1 + x2 = 5) (h2 : x1 * x2 = 2) :
  2 * x1 - x1 * x2 + 2 * x2 = 8 :=
by
  sorry

end roots_expression_value_l6_6008


namespace athul_downstream_distance_l6_6601

-- Define the conditions
def upstream_distance : ℝ := 16
def upstream_time : ℝ := 4
def speed_of_stream : ℝ := 1
def downstream_time : ℝ := 4

-- Translate the conditions into properties and prove the downstream distance
theorem athul_downstream_distance (V : ℝ) 
  (h1 : upstream_distance = (V - speed_of_stream) * upstream_time) :
  (V + speed_of_stream) * downstream_time = 24 := 
by
  -- Given the conditions, the proof would be filled here
  sorry

end athul_downstream_distance_l6_6601


namespace number_of_diagonal_intersections_of_convex_n_gon_l6_6143

theorem number_of_diagonal_intersections_of_convex_n_gon (n : ℕ) (h : 4 ≤ n) :
  (∀ P : Π m, m = n ↔ m ≥ 4, ∃ i : ℕ, i = n * (n - 1) * (n - 2) * (n - 3) / 24) := 
by
  sorry

end number_of_diagonal_intersections_of_convex_n_gon_l6_6143


namespace death_rate_is_three_l6_6571

-- Let birth_rate be the average birth rate in people per two seconds
def birth_rate : ℕ := 6
-- Let net_population_increase be the net population increase per day
def net_population_increase : ℕ := 129600
-- Let seconds_per_day be the total number of seconds in a day
def seconds_per_day : ℕ := 86400

noncomputable def death_rate_per_two_seconds : ℕ :=
  let net_increase_per_second := net_population_increase / seconds_per_day
  let birth_rate_per_second := birth_rate / 2
  2 * (birth_rate_per_second - net_increase_per_second)

theorem death_rate_is_three :
  death_rate_per_two_seconds = 3 := by
  sorry

end death_rate_is_three_l6_6571


namespace integers_even_condition_l6_6598

-- Definitions based on conditions
def is_even (n : ℤ) : Prop := n % 2 = 0

def exactly_one_even (a b c : ℤ) : Prop :=
(is_even a ∧ ¬ is_even b ∧ ¬ is_even c) ∨
(¬ is_even a ∧ is_even b ∧ ¬ is_even c) ∨
(¬ is_even a ∧ ¬ is_even b ∧ is_even c)

-- Proof statement
theorem integers_even_condition (a b c : ℤ) (h : ¬ exactly_one_even a b c) :
  (¬ is_even a ∧ ¬ is_even b ∧ ¬ is_even c) ∨
  (is_even a ∧ is_even b) ∨
  (is_even a ∧ is_even c) ∨
  (is_even b ∧ is_even c) :=
sorry

end integers_even_condition_l6_6598


namespace molecular_weight_N2O3_l6_6399

variable (atomic_weight_N : ℝ) (atomic_weight_O : ℝ)
variable (n_N_atoms : ℝ) (n_O_atoms : ℝ)
variable (expected_molecular_weight : ℝ)

theorem molecular_weight_N2O3 :
  atomic_weight_N = 14.01 →
  atomic_weight_O = 16.00 →
  n_N_atoms = 2 →
  n_O_atoms = 3 →
  expected_molecular_weight = 76.02 →
  (n_N_atoms * atomic_weight_N + n_O_atoms * atomic_weight_O = expected_molecular_weight) :=
by
  intros
  sorry

end molecular_weight_N2O3_l6_6399


namespace find_a_and_b_l6_6285

theorem find_a_and_b (a b : ℝ) (A B : Set ℝ) 
  (hA : A = {2, 3}) 
  (hB : B = {x | x^2 + a * x + b = 0}) 
  (h_intersection : A ∩ B = {2}) 
  (h_union : A ∪ B = A) : 
  (a + b = 0) ∨ (a + b = 1) := 
sorry

end find_a_and_b_l6_6285


namespace total_spent_on_date_l6_6592

-- Constants representing costs
def ticket_cost : ℝ := 10.00
def combo_meal_cost : ℝ := 11.00
def candy_cost : ℝ := 2.50

-- Numbers of items to buy
def num_tickets : ℝ := 2
def num_candies : ℝ := 2

-- Total cost calculation
def total_cost : ℝ := (ticket_cost * num_tickets) + (candy_cost * num_candies) + combo_meal_cost

-- Prove that the total cost is $36.00
theorem total_spent_on_date : total_cost = 36.00 := by
  sorry

end total_spent_on_date_l6_6592


namespace smallest_D_l6_6701

theorem smallest_D {A B C D : ℕ} (h1 : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D) 
  (h2 : (A * 100 + B * 10 + C) * B = D * 1000 + C * 100 + B * 10 + D) : 
  D = 1 :=
sorry

end smallest_D_l6_6701


namespace calories_per_serving_is_120_l6_6651

-- Define the conditions
def servings : ℕ := 3
def halfCalories : ℕ := 180
def totalCalories : ℕ := 2 * halfCalories

-- Define the target value
def caloriesPerServing : ℕ := totalCalories / servings

-- The proof goal
theorem calories_per_serving_is_120 : caloriesPerServing = 120 :=
by 
  sorry

end calories_per_serving_is_120_l6_6651


namespace none_satisfied_l6_6129

-- Define the conditions
variables {a b c x y z : ℝ}
  
-- Theorem that states that none of the given inequalities are satisfied strictly
theorem none_satisfied (h1 : x ≤ a) (h2 : y ≤ b) (h3 : z ≤ c) :
  ¬(x^2 * y + y^2 * z + z^2 * x < a^2 * b + b^2 * c + c^2 * a) ∧
  ¬(x^3 + y^3 + z^3 < a^3 + b^3 + c^3) :=
  by
    sorry

end none_satisfied_l6_6129


namespace graph_transformation_matches_B_l6_6191

noncomputable def f (x : ℝ) : ℝ :=
  if (-3 : ℝ) ≤ x ∧ x ≤ 0 then -2 - x
  else if 0 ≤ x ∧ x ≤ 2 then Real.sqrt (4 - (x - 2)^2) - 2
  else if 2 ≤ x ∧ x ≤ 3 then 2 * (x - 2)
  else 0 -- Define this part to handle cases outside the given range.

noncomputable def g (x : ℝ) : ℝ :=
  f ((1 - x) / 2)

theorem graph_transformation_matches_B :
  g = some_graph_function_B := 
sorry

end graph_transformation_matches_B_l6_6191


namespace correct_operation_l6_6848

theorem correct_operation (a : ℕ) :
  (a^2 * a^3 = a^5) ∧ ¬((a^2)^3 = a^5) ∧ ¬(a^6 / a^2 = a^3) ∧ ¬(3 * a^2 - 2 * a = a^2) :=
by
  sorry

end correct_operation_l6_6848


namespace man_is_older_by_24_l6_6051

-- Define the conditions as per the given problem
def present_age_son : ℕ := 22
def present_age_man (M : ℕ) : Prop := M + 2 = 2 * (present_age_son + 2)

-- State the problem: Prove that the man is 24 years older than his son
theorem man_is_older_by_24 (M : ℕ) (h : present_age_man M) : M - present_age_son = 24 := 
sorry

end man_is_older_by_24_l6_6051


namespace max_volume_of_cuboid_l6_6474

theorem max_volume_of_cuboid (x y z : ℝ) (h1 : 4 * (x + y + z) = 60) : 
  x * y * z ≤ 125 :=
by
  sorry

end max_volume_of_cuboid_l6_6474


namespace ratatouille_cost_per_quart_l6_6910

def eggplants := 88 * 0.22
def zucchini := 60.8 * 0.15
def tomatoes := 73.6 * 0.25
def onions := 43.2 * 0.07
def basil := (16 / 4) * 2.70
def bell_peppers := 12 * 0.20

def total_cost := eggplants + zucchini + tomatoes + onions + basil + bell_peppers
def yield := 4.5

def cost_per_quart := total_cost / yield

theorem ratatouille_cost_per_quart : cost_per_quart = 14.02 := 
by
  unfold cost_per_quart total_cost eggplants zucchini tomatoes onions basil bell_peppers 
  sorry

end ratatouille_cost_per_quart_l6_6910


namespace no_rain_either_day_l6_6965

noncomputable def P_A := 0.62
noncomputable def P_B := 0.54
noncomputable def P_A_and_B := 0.44
noncomputable def P_A_or_B := P_A + P_B - P_A_and_B -- Applying Inclusion-Exclusion principle.
noncomputable def P_A_and_B_complement := 1 - P_A_or_B -- Complement of P(A ∪ B).

theorem no_rain_either_day :
  P_A_and_B_complement = 0.28 :=
by
  unfold P_A_and_B_complement P_A_or_B
  unfold P_A P_B P_A_and_B
  simp
  sorry

end no_rain_either_day_l6_6965


namespace car_and_bus_speeds_l6_6685

-- Definitions of given conditions
def car_speed : ℕ := 44
def bus_speed : ℕ := 52

-- Definition of total distance after 4 hours
def total_distance (car_speed bus_speed : ℕ) := 4 * car_speed + 4 * bus_speed

-- Definition of fact that cars started from the same point and traveled in opposite directions
def cars_from_same_point (car_speed bus_speed : ℕ) := car_speed + bus_speed

theorem car_and_bus_speeds :
  total_distance car_speed (car_speed + 8) = 384 :=
by
  -- Proof constructed based on the conditions given
  sorry

end car_and_bus_speeds_l6_6685


namespace angle_skew_lines_range_l6_6563

theorem angle_skew_lines_range (θ : ℝ) (h1 : 0 < θ) (h2 : θ ≤ 90) : 0 < θ ∧ θ ≤ 90 :=
by sorry

end angle_skew_lines_range_l6_6563


namespace fraction_of_quarters_1840_1849_equals_4_over_15_l6_6804

noncomputable def fraction_of_states_from_1840s (total_states : ℕ) (states_from_1840s : ℕ) : ℚ := 
  states_from_1840s / total_states

theorem fraction_of_quarters_1840_1849_equals_4_over_15 :
  fraction_of_states_from_1840s 30 8 = 4 / 15 := 
by
  sorry

end fraction_of_quarters_1840_1849_equals_4_over_15_l6_6804


namespace quadratic_binomial_square_l6_6105

theorem quadratic_binomial_square (a : ℚ) :
  (∃ r s : ℚ, (ax^2 + 22*x + 9 = (r*x + s)^2) ∧ s = 3 ∧ r = 11 / 3) → a = 121 / 9 := 
by 
  sorry

end quadratic_binomial_square_l6_6105


namespace certain_number_divisible_by_9_l6_6102

theorem certain_number_divisible_by_9 : ∃ N : ℕ, (∀ k : ℕ, (0 ≤ k ∧ k < 1110 → N + 9 * k ≤ 10000 ∧ (N + 9 * k) % 9 = 0)) ∧ N = 27 :=
by
  -- Given conditions:
  -- Numbers are in an arithmetic sequence with common difference 9.
  -- Total count of such numbers is 1110.
  -- The last number ≤ 10000 that is divisible by 9 is 9999.
  let L := 9999
  let n := 1110
  let d := 9
  -- First term in the sequence:
  let a := L - (n - 1) * d
  exists 27
  -- Proof of the conditions would follow here ...
  sorry

end certain_number_divisible_by_9_l6_6102


namespace train_length_l6_6836

/-- Given a train traveling at 72 km/hr passing a pole in 8 seconds,
     prove that the length of the train in meters is 160. -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) (speed_m_s : ℝ) (distance_m : ℝ) :
  speed_kmh = 72 → 
  time_s = 8 → 
  speed_m_s = (speed_kmh * 1000) / 3600 → 
  distance_m = speed_m_s * time_s → 
  distance_m = 160 :=
by
  sorry

end train_length_l6_6836


namespace regular_polygon_perimeter_l6_6377

theorem regular_polygon_perimeter (s : ℝ) (exterior_angle : ℝ) 
  (h1 : s = 7) (h2 : exterior_angle = 45) : 
  8 * s = 56 :=
by
  sorry

end regular_polygon_perimeter_l6_6377


namespace average_people_per_boat_correct_l6_6709

-- Define number of boats and number of people
def num_boats := 3.0
def num_people := 5.0

-- Definition for average people per boat
def avg_people_per_boat := num_people / num_boats

-- Theorem to prove the average number of people per boat is 1.67
theorem average_people_per_boat_correct : avg_people_per_boat = 1.67 := by
  sorry

end average_people_per_boat_correct_l6_6709


namespace Roberta_spent_on_shoes_l6_6418

-- Define the conditions as per the problem statement
variables (S B L : ℝ) (h1 : B = S - 17) (h2 : L = B / 4) (h3 : 158 - (S + B + L) = 78)

-- State the theorem to be proved
theorem Roberta_spent_on_shoes : S = 45 :=
by
  -- use variables and conditions
  have := h1
  have := h2
  have := h3
  sorry -- Proof steps can be filled later

end Roberta_spent_on_shoes_l6_6418


namespace sector_area_l6_6043

theorem sector_area (α r : ℝ) (hα : α = π / 3) (hr : r = 2) : 
  1 / 2 * α * r^2 = 2 * π / 3 := 
by 
  rw [hα, hr] 
  simp 
  sorry

end sector_area_l6_6043


namespace min_value_expression_l6_6170

noncomputable def min_expression (a b c : ℝ) : ℝ :=
  (a + b) / c + (b + c) / a + (c + a) / b + 3

theorem min_value_expression {a b c : ℝ} (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a ≠ b) (h5 : b ≠ c) (h6 : c ≠ a) : 
  ∃ x, min_expression a b c = x ∧ x ≥ 9 := 
sorry

end min_value_expression_l6_6170


namespace repaved_before_today_l6_6639

variable (total_repaved today_repaved : ℕ)

theorem repaved_before_today (h1 : total_repaved = 4938) (h2 : today_repaved = 805) :
  total_repaved - today_repaved = 4133 :=
by 
  -- variables are integers and we are performing a subtraction
  sorry

end repaved_before_today_l6_6639


namespace max_value_h3_solve_for_h_l6_6218

-- Definition part for conditions
def quadratic_function (h : ℝ) (x : ℝ) : ℝ :=
  -(x - h) ^ 2

-- Part (1): When h = 3, proving the maximum value of the function within 2 ≤ x ≤ 5 is 0.
theorem max_value_h3 : ∀ x : ℝ, 2 ≤ x ∧ x ≤ 5 → quadratic_function 3 x ≤ 0 :=
by
  sorry

-- Part (2): If the maximum value of the function is -1, then the value of h is 6 or 1.
theorem solve_for_h (h : ℝ) : 
  (∀ x : ℝ, 2 ≤ x ∧ x ≤ 5 → quadratic_function h x ≤ -1) ↔ h = 6 ∨ h = 1 :=
by
  sorry

end max_value_h3_solve_for_h_l6_6218


namespace line_does_not_pass_through_fourth_quadrant_l6_6724

theorem line_does_not_pass_through_fourth_quadrant
  (A B C : ℝ) (hAB : A * B < 0) (hBC : B * C < 0) :
  ¬ ∃ x y : ℝ, x > 0 ∧ y < 0 ∧ A * x + B * y + C = 0 :=
by
  sorry

end line_does_not_pass_through_fourth_quadrant_l6_6724


namespace sales_tax_is_5_percent_l6_6817

theorem sales_tax_is_5_percent :
  let cost_tshirt := 8
  let cost_sweater := 18
  let cost_jacket := 80
  let discount := 0.10
  let num_tshirts := 6
  let num_sweaters := 4
  let num_jackets := 5
  let total_cost_with_tax := 504
  let total_cost_before_discount := (num_jackets * cost_jacket)
  let discount_amount := discount * total_cost_before_discount
  let discounted_cost_jackets := total_cost_before_discount - discount_amount
  let total_cost_before_tax := (num_tshirts * cost_tshirt) + (num_sweaters * cost_sweater) + discounted_cost_jackets
  let sales_tax := (total_cost_with_tax - total_cost_before_tax)
  let sales_tax_percentage := (sales_tax / total_cost_before_tax) * 100
  sales_tax_percentage = 5 := by
  sorry

end sales_tax_is_5_percent_l6_6817


namespace probability_of_winning_five_tickets_l6_6028

def probability_of_winning_one_ticket := 1 / 10000000
def number_of_tickets_bought := 5

theorem probability_of_winning_five_tickets : 
  (number_of_tickets_bought * probability_of_winning_one_ticket) = 5 / 10000000 :=
by
  sorry

end probability_of_winning_five_tickets_l6_6028


namespace chastity_leftover_money_l6_6160

theorem chastity_leftover_money (n_lollipops : ℕ) (price_lollipop : ℝ) (n_gummies : ℕ) (price_gummy : ℝ) (initial_money : ℝ) :
  n_lollipops = 4 →
  price_lollipop = 1.50 →
  n_gummies = 2 →
  price_gummy = 2 →
  initial_money = 15 →
  initial_money - ((n_lollipops * price_lollipop) + (n_gummies * price_gummy)) = 5 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  norm_num
  sorry

end chastity_leftover_money_l6_6160


namespace average_weight_of_section_B_l6_6316

theorem average_weight_of_section_B
  (num_students_A : ℕ) (num_students_B : ℕ)
  (avg_weight_A : ℝ) (avg_weight_class : ℝ)
  (total_students : ℕ := num_students_A + num_students_B)
  (total_weight_class : ℝ := total_students * avg_weight_class)
  (total_weight_A : ℝ := num_students_A * avg_weight_A)
  (total_weight_B : ℝ := total_weight_class - total_weight_A)
  (avg_weight_B : ℝ := total_weight_B / num_students_B) :
  num_students_A = 50 →
  num_students_B = 40 →
  avg_weight_A = 50 →
  avg_weight_class = 58.89 →
  avg_weight_B = 70.0025 :=
by intros; sorry

end average_weight_of_section_B_l6_6316


namespace walnut_trees_planted_l6_6827

-- Define the initial number of walnut trees
def initial_walnut_trees : ℕ := 22

-- Define the total number of walnut trees after planting
def total_walnut_trees_after : ℕ := 55

-- The Lean statement to prove the number of walnut trees planted today
theorem walnut_trees_planted : (total_walnut_trees_after - initial_walnut_trees = 33) :=
by
  sorry

end walnut_trees_planted_l6_6827


namespace total_initial_seashells_l6_6304

-- Definitions for the conditions
def Henry_seashells := 11
def Paul_seashells := 24

noncomputable def Leo_initial_seashells (total_seashells : ℕ) :=
  (total_seashells - (Henry_seashells + Paul_seashells)) * 4 / 3

theorem total_initial_seashells 
  (total_seashells_now : ℕ)
  (leo_shared_fraction : ℕ → ℕ)
  (h : total_seashells_now = 53) : 
  Henry_seashells + Paul_seashells + leo_shared_fraction 53 = 59 :=
by
  let L := Leo_initial_seashells 53
  have L_initial : L = 24 := by sorry
  exact sorry

end total_initial_seashells_l6_6304


namespace freds_average_book_cost_l6_6292

theorem freds_average_book_cost :
  ∀ (initial_amount spent_amount num_books remaining_amount avg_cost : ℕ),
    initial_amount = 236 →
    remaining_amount = 14 →
    num_books = 6 →
    spent_amount = initial_amount - remaining_amount →
    avg_cost = spent_amount / num_books →
    avg_cost = 37 :=
by
  intros initial_amount spent_amount num_books remaining_amount avg_cost h_init h_rem h_books h_spent h_avg
  sorry

end freds_average_book_cost_l6_6292


namespace price_per_vanilla_cookie_l6_6203

theorem price_per_vanilla_cookie (P : ℝ) (h1 : 220 + 70 * P = 360) : P = 2 := 
by 
  sorry

end price_per_vanilla_cookie_l6_6203


namespace johns_sixth_quiz_score_l6_6048

theorem johns_sixth_quiz_score
  (score1 score2 score3 score4 score5 : ℕ)
  (h1 : score1 = 85)
  (h2 : score2 = 90)
  (h3 : score3 = 88)
  (h4 : score4 = 92)
  (h5 : score5 = 95)
  : (∃ score6 : ℕ, (score1 + score2 + score3 + score4 + score5 + score6) / 6 = 90) :=
by
  use 90
  sorry

end johns_sixth_quiz_score_l6_6048


namespace negation_of_proposition_l6_6980

variable (f : ℕ+ → ℕ)

theorem negation_of_proposition :
  (¬ ∀ n : ℕ+, f n ≤ n) ↔ (∃ n : ℕ+, f n > n) :=
by sorry

end negation_of_proposition_l6_6980


namespace parabola_intersection_l6_6217

theorem parabola_intersection:
  (∀ x y1 y2 : ℝ, (y1 = 3 * x^2 - 6 * x + 6) ∧ (y2 = -2 * x^2 - 4 * x + 6) → y1 = y2 → x = 0 ∨ x = 2 / 5) ∧
  (∀ a c : ℝ, a = 0 ∧ c = 2 / 5 ∧ c ≥ a → c - a = 2 / 5) :=
by sorry

end parabola_intersection_l6_6217


namespace phillip_spent_on_oranges_l6_6360

theorem phillip_spent_on_oranges 
  (M : ℕ) (A : ℕ) (C : ℕ) (L : ℕ) (O : ℕ)
  (hM : M = 95) (hA : A = 25) (hC : C = 6) (hL : L = 50)
  (h_total_spending : O + A + C = M - L) : 
  O = 14 := 
sorry

end phillip_spent_on_oranges_l6_6360


namespace sin_double_angle_plus_pi_div_two_l6_6673

open Real

theorem sin_double_angle_plus_pi_div_two (θ : ℝ) (h₀ : 0 < θ) (h₁ : θ < π / 2) (h₂ : sin θ = 1 / 3) :
  sin (2 * θ + π / 2) = 7 / 9 :=
by
  sorry

end sin_double_angle_plus_pi_div_two_l6_6673


namespace sum_le_six_l6_6458

theorem sum_le_six (a b : ℤ) (h1 : a ≠ -1) (h2 : b ≠ -1) 
    (h3 : ∃ (r s : ℤ), r * s = a + b ∧ r + s = ab) : a + b ≤ 6 :=
sorry

end sum_le_six_l6_6458


namespace base_six_conversion_addition_l6_6904

def base_six_to_base_ten (n : ℕ) : ℕ :=
  4 * 6^0 + 1 * 6^1 + 2 * 6^2

theorem base_six_conversion_addition : base_six_to_base_ten 214 + 15 = 97 :=
by
  sorry

end base_six_conversion_addition_l6_6904


namespace maria_age_l6_6049

variable (M J : Nat)

theorem maria_age (h1 : J = M + 12) (h2 : M + J = 40) : M = 14 := by
  sorry

end maria_age_l6_6049


namespace eggs_in_each_basket_l6_6648

theorem eggs_in_each_basket :
  ∃ x : ℕ, x ∣ 30 ∧ x ∣ 42 ∧ x ≥ 5 ∧ x = 6 :=
by
  sorry

end eggs_in_each_basket_l6_6648


namespace honey_barrel_problem_l6_6482

theorem honey_barrel_problem
  (x y : ℝ)
  (h1 : x + y = 56)
  (h2 : x / 2 + y = 34) :
  x = 44 ∧ y = 12 :=
by
  sorry

end honey_barrel_problem_l6_6482


namespace city_schools_count_l6_6690

theorem city_schools_count (a b c : ℕ) (schools : ℕ) : 
  b = 40 → c = 51 → b < a → a < c → 
  (a > b ∧ a < c ∧ (a - 1) * 3 < (c - b + 1) * 3 + 1) → 
  schools = (c - 1) / 3 :=
by
  sorry

end city_schools_count_l6_6690


namespace comb_product_l6_6733

theorem comb_product :
  (Nat.choose 10 3) * (Nat.choose 8 3) * 2 = 13440 :=
by
  sorry

end comb_product_l6_6733


namespace n_squared_plus_n_divisible_by_2_l6_6245

theorem n_squared_plus_n_divisible_by_2 (n : ℤ) : 2 ∣ (n^2 + n) :=
sorry

end n_squared_plus_n_divisible_by_2_l6_6245


namespace convert_base_8_to_base_10_l6_6727

def to_base_10 (n : ℕ) (b : ℕ) (digits : List ℕ) : ℕ :=
  digits.foldr (λ digit acc => acc * b + digit) 0

theorem convert_base_8_to_base_10 : 
  to_base_10 10 8 [6, 4, 2] = 166 := by
  sorry

end convert_base_8_to_base_10_l6_6727


namespace int_solution_count_l6_6830

def g (n : ℤ) : ℤ :=
  ⌈97 * n / 98⌉ - ⌊98 * n / 99⌋

theorem int_solution_count :
  (∃! n : ℤ, 1 + ⌊98 * n / 99⌋ = ⌈97 * n / 98⌉) :=
sorry

end int_solution_count_l6_6830


namespace C_increases_as_n_increases_l6_6426

noncomputable def C (e R r n : ℝ) : ℝ := (e * n^2) / (R + n * r)

theorem C_increases_as_n_increases
  (e R r : ℝ) (e_pos : 0 < e) (R_pos : 0 < R) (r_pos : 0 < r) :
  ∀ n : ℝ, 0 < n → ∃ M : ℝ, ∀ N : ℝ, N > n → C e R r N > M :=
by
  sorry

end C_increases_as_n_increases_l6_6426


namespace coins_left_zero_when_divided_by_9_l6_6780

noncomputable def smallestCoinCount (n: ℕ) : Prop :=
  n % 8 = 6 ∧ n % 7 = 5

theorem coins_left_zero_when_divided_by_9 (n : ℕ) (h : smallestCoinCount n) (h_min: ∀ m : ℕ, smallestCoinCount m → n ≤ m) :
  n % 9 = 0 :=
sorry

end coins_left_zero_when_divided_by_9_l6_6780


namespace markdown_calculation_l6_6167

noncomputable def markdown_percentage (P S : ℝ) (h_inc : P = S * 1.1494) : ℝ :=
  1 - (1 / 1.1494)

theorem markdown_calculation (P S : ℝ) (h_sale : S = P * (1 - markdown_percentage P S sorry / 100)) (h_inc : P = S * 1.1494) :
  markdown_percentage P S h_inc = 12.99 := 
sorry

end markdown_calculation_l6_6167


namespace calculate_k_l6_6665

variable (A B C D k : ℕ)

def workers_time : Prop :=
  (1 / (A : ℚ) + 1 / (B : ℚ) + 1 / (C : ℚ) + 1 / (D : ℚ)) = 
  (1 / (A - 8 : ℚ)) ∧
  (1 / (A : ℚ) + 1 / (B : ℚ) + 1 / (C : ℚ) + 1 / (D : ℚ)) = 
  (1 / (B - 2 : ℚ)) ∧
  (1 / (A : ℚ) + 1 / (B : ℚ) + 1 / (C : ℚ) + 1 / (D : ℚ)) = 
  (3 / (C : ℚ))

theorem calculate_k (h : workers_time A B C D) : k = 16 :=
  sorry

end calculate_k_l6_6665


namespace john_saves_money_l6_6576

def original_spending (coffees_per_day: ℕ) (price_per_coffee: ℕ) : ℕ :=
  coffees_per_day * price_per_coffee

def new_price (original_price: ℕ) (increase_percentage: ℕ) : ℕ :=
  original_price + (original_price * increase_percentage / 100)

def new_coffees_per_day (original_coffees_per_day: ℕ) (reduction_fraction: ℕ) : ℕ :=
  original_coffees_per_day / reduction_fraction

def current_spending (new_coffees_per_day: ℕ) (new_price_per_coffee: ℕ) : ℕ :=
  new_coffees_per_day * new_price_per_coffee

theorem john_saves_money
  (coffees_per_day : ℕ := 4)
  (price_per_coffee : ℕ := 2)
  (increase_percentage : ℕ := 50)
  (reduction_fraction : ℕ := 2) :
  original_spending coffees_per_day price_per_coffee
  - current_spending (new_coffees_per_day coffees_per_day reduction_fraction)
                     (new_price price_per_coffee increase_percentage)
  = 2 := by
{
  sorry
}

end john_saves_money_l6_6576


namespace find_positive_integers_l6_6231

theorem find_positive_integers (m n : ℕ) (hm : m > 0) (hn : n > 0) :
  (1 / m + 1 / n - 1 / (m * n) = 2 / 5) ↔ 
  (m = 3 ∧ n = 10) ∨ (m = 10 ∧ n = 3) ∨ (m = 4 ∧ n = 5) ∨ (m = 5 ∧ n = 4) :=
by sorry

end find_positive_integers_l6_6231


namespace grazing_area_proof_l6_6357

noncomputable def grazing_area (s r : ℝ) : ℝ :=
  let A_circle := 3.14 * r^2
  let A_sector := (300 / 360) * A_circle
  let A_triangle := (1.732 / 4) * s^2
  let A_triangle_part := A_triangle / 3
  let A_grazing := A_sector - A_triangle_part
  3 * A_grazing

theorem grazing_area_proof : grazing_area 5 7 = 136.59 :=
  by
  sorry

end grazing_area_proof_l6_6357


namespace range_of_3a_minus_2b_l6_6862

theorem range_of_3a_minus_2b (a b : ℝ) (h1 : 1 ≤ a - b ∧ a - b ≤ 2) (h2 : 2 ≤ a + b ∧ a + b ≤ 4) :
  7 / 2 ≤ 3 * a - 2 * b ∧ 3 * a - 2 * b ≤ 7 :=
sorry

end range_of_3a_minus_2b_l6_6862


namespace number_of_different_duty_schedules_l6_6708

-- Define a structure for students
inductive Student
| A | B | C

-- Define days of the week excluding Sunday as all duties are from Monday to Saturday
inductive Day
| Monday | Tuesday | Wednesday | Thursday | Friday | Saturday

-- Define the conditions in Lean
def condition_A_does_not_take_Monday (schedules : Day → Student) : Prop :=
  schedules Day.Monday ≠ Student.A

def condition_B_does_not_take_Saturday (schedules : Day → Student) : Prop :=
  schedules Day.Saturday ≠ Student.B

-- Define the function to count valid schedules
noncomputable def count_valid_schedules : ℕ :=
  sorry  -- This would be the computation considering combinatorics

-- Theorem statement to prove the correct answer
theorem number_of_different_duty_schedules 
    (schedules : Day → Student)
    (h1 : condition_A_does_not_take_Monday schedules)
    (h2 : condition_B_does_not_take_Saturday schedules)
    : count_valid_schedules = 42 :=
sorry

end number_of_different_duty_schedules_l6_6708


namespace geometric_sequence_ratio_28_l6_6244

noncomputable def geometric_sequence_sum_ratio (a1 : ℝ) (q : ℝ) (S : ℕ → ℝ) :=
  S 6 / S 3 = 28

theorem geometric_sequence_ratio_28 (a1 : ℝ) (q : ℝ) (S : ℕ → ℝ) 
  (h_GS : ∀ n, S n = a1 * (1 - q^n) / (1 - q)) 
  (h_increasing : ∀ n m, n < m → a1 * q^n < a1 * q^m) 
  (h_mean : 2 * 6 * a1 * q^6 = a1 * q^7 + a1 * q^8) : 
  geometric_sequence_sum_ratio a1 q S := 
by {
  -- Proof should be completed here
  sorry
}

end geometric_sequence_ratio_28_l6_6244


namespace jose_share_of_profit_correct_l6_6642

noncomputable def jose_share_of_profit (total_profit : ℝ) : ℝ :=
  let tom_investment_time := 30000 * 12
  let jose_investment_time := 45000 * 10
  let angela_investment_time := 60000 * 8
  let rebecca_investment_time := 75000 * 6
  let total_investment_time := tom_investment_time + jose_investment_time + angela_investment_time + rebecca_investment_time
  (jose_investment_time / total_investment_time) * total_profit

theorem jose_share_of_profit_correct : 
  ∀ (total_profit : ℝ), total_profit = 72000 -> jose_share_of_profit total_profit = 18620.69 := 
by
  intro total_profit
  sorry

end jose_share_of_profit_correct_l6_6642


namespace total_original_cost_of_books_l6_6423

noncomputable def original_cost_price_in_eur (selling_prices : List ℝ) (profit_margin : ℝ) (exchange_rate : ℝ) : ℝ :=
  let original_cost_prices := selling_prices.map (λ price => price / (1 + profit_margin))
  let total_original_cost_usd := original_cost_prices.sum
  total_original_cost_usd * exchange_rate

theorem total_original_cost_of_books : original_cost_price_in_eur [240, 260, 280, 300, 320] 0.20 0.85 = 991.67 :=
  sorry

end total_original_cost_of_books_l6_6423


namespace find_g5_l6_6990

variable (g : ℝ → ℝ)

-- Formal definition of the condition for the function g in the problem statement.
def functional_eq_condition :=
  ∀ x : ℝ, g x + 3 * g (2 - x) = 4 * x^2

-- The main statement to prove g(5) = 1 under the given condition.
theorem find_g5 (h : functional_eq_condition g) :
  g 5 = 1 :=
sorry

end find_g5_l6_6990


namespace probability_dice_sum_perfect_square_l6_6517

def is_perfect_square (n : ℕ) : Prop :=
  n = 4 ∨ n = 9 ∨ n = 16

noncomputable def probability_perfect_square : ℚ :=
  12 / 64

theorem probability_dice_sum_perfect_square :
  -- Two standard 8-sided dice are rolled
  -- The probability that the sum rolled is a perfect square is 3/16
  (∃ dice1 dice2 : ℕ, 1 ≤ dice1 ∧ dice1 ≤ 8 ∧ 1 ≤ dice2 ∧ dice2 ≤ 8) →
  probability_perfect_square = 3 / 16 :=
sorry

end probability_dice_sum_perfect_square_l6_6517


namespace sum_of_pairwise_rel_prime_integers_l6_6005

def is_pairwise_rel_prime (a b c : ℕ) : Prop :=
  Nat.gcd a b = 1 ∧ Nat.gcd b c = 1 ∧ Nat.gcd a c = 1

theorem sum_of_pairwise_rel_prime_integers 
  (a b c : ℕ) (h1 : a > 1) (h2 : b > 1) (h3 : c > 1) 
  (h_prod : a * b * c = 343000) (h_rel_prime : is_pairwise_rel_prime a b c) : 
  a + b + c = 476 := 
sorry

end sum_of_pairwise_rel_prime_integers_l6_6005


namespace prob_A_prob_B_l6_6728

variable (a b : ℝ) -- Declare variables a and b as real numbers
variable (h_ab : a + b = 1) -- Declare the condition a + b = 1
variable (h_pos_a : 0 < a) -- Declare a is a positive real number
variable (h_pos_b : 0 < b) -- Declare b is a positive real number

-- Prove that 1/a + 1/b ≥ 4 under the given conditions
theorem prob_A (h_ab : a + b = 1) (h_pos_a : 0 < a) (h_pos_b : 0 < b) : 
  (1 / a) + (1 / b) ≥ 4 :=
by
  sorry

-- Prove that a^2 + b^2 ≥ 1/2 under the given conditions
theorem prob_B (h_ab : a + b = 1) (h_pos_a : 0 < a) (h_pos_b : 0 < b) : 
  a^2 + b^2 ≥ 1 / 2 :=
by
  sorry

end prob_A_prob_B_l6_6728


namespace simplify_and_evaluate_l6_6211

theorem simplify_and_evaluate :
  ∀ (a b : ℤ), a = -1 → b = 4 →
  (a + b)^2 - 2 * a * (a - b) + (a + 2 * b) * (a - 2 * b) = -64 :=
by
  intros a b ha hb
  rw [ha, hb]
  sorry

end simplify_and_evaluate_l6_6211


namespace find_number_l6_6034

theorem find_number (x : ℤ) (n : ℤ) (h1 : x = 88320) (h2 : x + 1315 + n - 1569 = 11901) : n = -75165 :=
by 
  sorry

end find_number_l6_6034


namespace factor_expression_l6_6602

-- Problem Statement
theorem factor_expression (x y : ℝ) : 60 * x ^ 2 + 40 * y = 20 * (3 * x ^ 2 + 2 * y) :=
by
  -- Proof to be provided
  sorry

end factor_expression_l6_6602


namespace tangent_line_to_ellipse_l6_6536

variable (a b x y x₀ y₀ : ℝ)

-- Definitions
def is_ellipse (x y a b : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

def point_on_ellipse (x₀ y₀ a b : ℝ) : Prop :=
  x₀^2 / a^2 + y₀^2 / b^2 = 1

-- Theorem
theorem tangent_line_to_ellipse
  (h₁ : point_on_ellipse x₀ y₀ a b) :
    (x₀ * x) / (a^2) + (y₀ * y) / (b^2) = 1 :=
sorry

end tangent_line_to_ellipse_l6_6536


namespace man_older_than_son_l6_6197

theorem man_older_than_son (S M : ℕ) (h1 : S = 18) (h2 : M + 2 = 2 * (S + 2)) : M - S = 20 :=
by
  sorry

end man_older_than_son_l6_6197


namespace product_identity_l6_6273

theorem product_identity :
  (1 + 1 / Nat.factorial 1) * (1 + 1 / Nat.factorial 2) * (1 + 1 / Nat.factorial 3) *
  (1 + 1 / Nat.factorial 4) * (1 + 1 / Nat.factorial 5) * (1 + 1 / Nat.factorial 6) *
  (1 + 1 / Nat.factorial 7) = 5041 / 5040 := sorry

end product_identity_l6_6273


namespace f_strictly_increasing_intervals_l6_6462

noncomputable def f (x : Real) : Real :=
  x * Real.sin x + Real.cos x

noncomputable def f' (x : Real) : Real :=
  x * Real.cos x

theorem f_strictly_increasing_intervals :
  ∀ (x : Real), (-π < x ∧ x < -π / 2 ∨ 0 < x ∧ x < π / 2) → f' x > 0 :=
by
  intros x h
  sorry

end f_strictly_increasing_intervals_l6_6462


namespace max_min_of_f_find_a_and_theta_l6_6345

noncomputable def f (x θ a : ℝ) : ℝ :=
  Real.sin (x + θ) + a * Real.cos (x + 2 * θ)

theorem max_min_of_f (a θ : ℝ) (h1 : a = Real.sqrt 2) (h2 : θ = π / 4) :
  (∀ x ∈ Set.Icc 0 π, -1 ≤ f x θ a ∧ f x θ a ≤ (Real.sqrt 2) / 2) := sorry

theorem find_a_and_theta (a θ : ℝ) (h1 : f (π / 2) θ a = 0) (h2 : f π θ a = 1) :
  a = -1 ∧ θ = -π / 6 := sorry

end max_min_of_f_find_a_and_theta_l6_6345


namespace regular_polygon_sides_l6_6123

theorem regular_polygon_sides (n : ℕ) (h : n ≥ 3) (interior_angle : ℝ) : 
  interior_angle = 144 → n = 10 :=
by
  intro h1
  sorry

end regular_polygon_sides_l6_6123


namespace number_of_people_in_team_l6_6498

variable (x : ℕ) -- Number of people in the team

-- Conditions as definitions
def average_age_all (x : ℕ) : ℝ := 25
def leader_age : ℝ := 45
def average_age_without_leader (x : ℕ) : ℝ := 23

-- Proof problem statement
theorem number_of_people_in_team (h1 : (x : ℝ) * average_age_all x = x * (average_age_without_leader x - 1) + leader_age) : x = 11 := by
  sorry

end number_of_people_in_team_l6_6498


namespace smallest_value_of_x_l6_6879

theorem smallest_value_of_x :
  ∃ x, (12 * x^2 - 58 * x + 70 = 0) ∧ x = 7 / 3 :=
by
  sorry

end smallest_value_of_x_l6_6879


namespace flower_beds_l6_6676

theorem flower_beds (seeds_per_bed total_seeds flower_beds : ℕ) 
  (h1 : seeds_per_bed = 10) (h2 : total_seeds = 60) : 
  flower_beds = total_seeds / seeds_per_bed := by
  rw [h1, h2]
  sorry

end flower_beds_l6_6676


namespace x_intercept_of_perpendicular_line_l6_6236

theorem x_intercept_of_perpendicular_line (x y : ℝ) (h1 : 5 * x - 3 * y = 9) (y_intercept : ℝ) 
  (h2 : y_intercept = 4) : x = 20 / 3 :=
sorry

end x_intercept_of_perpendicular_line_l6_6236


namespace problem_statement_l6_6263

theorem problem_statement 
  (x y z : ℝ) 
  (hx1 : x ≠ 1) 
  (hy1 : y ≠ 1) 
  (hz1 : z ≠ 1) 
  (hxyz : x * y * z = 1) : 
  x^2 / (x - 1)^2 + y^2 / (y - 1)^2 + z^2 / (z - 1)^2 ≥ 1 :=
sorry

end problem_statement_l6_6263


namespace max_value_pq_qr_rs_sp_l6_6566

def max_pq_qr_rs_sp (p q r s : ℕ) : ℕ :=
  p * q + q * r + r * s + s * p

theorem max_value_pq_qr_rs_sp :
  ∀ (p q r s : ℕ), (p = 1 ∨ p = 5 ∨ p = 3 ∨ p = 6) → 
                    (q = 1 ∨ q = 5 ∨ q = 3 ∨ q = 6) →
                    (r = 1 ∨ r = 5 ∨ r = 3 ∨ r = 6) → 
                    (s = 1 ∨ s = 5 ∨ s = 3 ∨ s = 6) →
                    p ≠ q → p ≠ r → p ≠ s → q ≠ r → q ≠ s → r ≠ s → 
                    max_pq_qr_rs_sp p q r s ≤ 56 := by
  sorry

end max_value_pq_qr_rs_sp_l6_6566


namespace selection_structure_count_is_three_l6_6126

def requiresSelectionStructure (problem : ℕ) : Bool :=
  match problem with
  | 1 => true
  | 2 => false
  | 3 => true
  | 4 => true
  | _ => false

def countSelectionStructure : ℕ :=
  (if requiresSelectionStructure 1 then 1 else 0) +
  (if requiresSelectionStructure 2 then 1 else 0) +
  (if requiresSelectionStructure 3 then 1 else 0) +
  (if requiresSelectionStructure 4 then 1 else 0)

theorem selection_structure_count_is_three : countSelectionStructure = 3 :=
  by
    sorry

end selection_structure_count_is_three_l6_6126


namespace travel_time_without_walking_l6_6801

-- Definitions based on the problem's conditions
def walking_time_without_escalator (x y : ℝ) : Prop := 75 * x = y
def walking_time_with_escalator (x k y : ℝ) : Prop := 30 * (x + k) = y

-- Main theorem: Time taken to travel the distance with the escalator alone
theorem travel_time_without_walking (x k y : ℝ) (h1 : walking_time_without_escalator x y) (h2 : walking_time_with_escalator x k y) : y / k = 50 :=
by
  sorry

end travel_time_without_walking_l6_6801


namespace remainder_when_7n_div_by_3_l6_6282

theorem remainder_when_7n_div_by_3 (n : ℤ) (h : n % 3 = 2) : (7 * n) % 3 = 2 := 
sorry

end remainder_when_7n_div_by_3_l6_6282


namespace proof_ratio_QP_over_EF_l6_6803

noncomputable def rectangle_theorem : Prop :=
  ∃ (A B C D E F G P Q : ℝ × ℝ),
    -- Coordinates of the rectangle vertices
    A = (0, 4) ∧ B = (5, 4) ∧ C = (5, 0) ∧ D = (0, 0) ∧
    -- Coordinates of points E, F, and G on the sides of the rectangle
    E = (4, 4) ∧ F = (2, 0) ∧ G = (5, 1) ∧
    -- Coordinates of intersection points P and Q
    P = (20 / 7, 12 / 7) ∧ Q = (40 / 13, 28 / 13) ∧
    -- Ratio of distances PQ and EF
    (dist P Q)/(dist E F) = 10 / 91

theorem proof_ratio_QP_over_EF : rectangle_theorem :=
sorry

end proof_ratio_QP_over_EF_l6_6803


namespace price_of_pants_l6_6195

theorem price_of_pants (P : ℝ) 
  (h1 : (3 / 4) * P + P + (P + 10) = 340)
  (h2 : ∃ P, (3 / 4) * P + P + (P + 10) = 340) : 
  P = 120 :=
sorry

end price_of_pants_l6_6195


namespace ratio_B_to_A_l6_6740

-- Definitions for conditions
def w_B : ℕ := 275 -- weight of element B in grams
def w_X : ℕ := 330 -- total weight of compound X in grams

-- Statement to prove
theorem ratio_B_to_A : (w_B:ℚ) / (w_X - w_B) = 5 :=
by 
  sorry

end ratio_B_to_A_l6_6740


namespace calculate_expression_l6_6723

theorem calculate_expression : |(-5 : ℤ)| + (1 / 3 : ℝ)⁻¹ - (Real.pi - 2) ^ 0 = 7 := by
  sorry

end calculate_expression_l6_6723


namespace expand_and_simplify_l6_6092

theorem expand_and_simplify (x : ℝ) : (17 * x - 9) * 3 * x = 51 * x^2 - 27 * x := 
by 
  sorry

end expand_and_simplify_l6_6092


namespace max_marks_l6_6222

theorem max_marks {M : ℝ} (h : 0.90 * M = 550) : M = 612 :=
sorry

end max_marks_l6_6222


namespace minimum_value_fraction_l6_6919

theorem minimum_value_fraction (a b : ℝ) (h1 : a > 1) (h2 : b > 2) (h3 : 2 * a + b - 6 = 0) :
  (1 / (a - 1) + 2 / (b - 2)) = 4 := 
  sorry

end minimum_value_fraction_l6_6919


namespace light_match_first_l6_6163

-- Define the conditions
def dark_room : Prop := true
def has_candle : Prop := true
def has_kerosene_lamp : Prop := true
def has_ready_to_use_stove : Prop := true
def has_single_match : Prop := true

-- Define the main question as a theorem
theorem light_match_first (h1 : dark_room) (h2 : has_candle) (h3 : has_kerosene_lamp) (h4 : has_ready_to_use_stove) (h5 : has_single_match) : true :=
by
  sorry

end light_match_first_l6_6163


namespace multiple_of_shorter_piece_l6_6925

theorem multiple_of_shorter_piece :
  ∃ (m : ℕ), 
  (35 + (m * 35 + 15) = 120) ∧
  (m = 2) :=
by
  sorry

end multiple_of_shorter_piece_l6_6925


namespace g_f_neg3_l6_6404

def f (x : ℤ) : ℤ := x^3 - 1
def g (x : ℤ) : ℤ := 3 * x^2 + 3 * x + 1

theorem g_f_neg3 : g (f (-3)) = 2285 :=
by
  -- provide the proof here
  sorry

end g_f_neg3_l6_6404


namespace pq_sum_l6_6213

def single_digit (n : ℕ) : Prop := n < 10

theorem pq_sum (P Q : ℕ) (hP : single_digit P) (hQ : single_digit Q)
  (hSum : P * 100 + Q * 10 + Q + P * 110 + Q + Q * 111 = 876) : P + Q = 5 :=
by 
  -- Here we assume the expected outcome based on the problem solution
  sorry

end pq_sum_l6_6213


namespace solve_box_dimensions_l6_6072

theorem solve_box_dimensions (m n r : ℕ) (h1 : m ≤ n) (h2 : n ≤ r) (h3 : m ≥ 1) (h4 : n ≥ 1) (h5 : r ≥ 1) :
  let k₀ := (m - 2) * (n - 2) * (r - 2)
  let k₁ := 2 * ((m - 2) * (n - 2) + (m - 2) * (r - 2) + (n - 2) * (r - 2))
  let k₂ := 4 * ((m - 2) + (n - 2) + (r - 2))
  (k₀ + k₂ - k₁ = 1985) ↔ ((m = 5 ∧ n = 7 ∧ r = 663) ∨ 
                            (m = 5 ∧ n = 5 ∧ r = 1981) ∨
                            (m = 3 ∧ n = 3 ∧ r = 1981) ∨
                            (m = 1 ∧ n = 7 ∧ r = 399) ∨
                            (m = 1 ∧ n = 3 ∧ r = 1987)) :=
sorry

end solve_box_dimensions_l6_6072


namespace batsman_average_17th_innings_l6_6826

theorem batsman_average_17th_innings:
  ∀ (A : ℝ), 
  (16 * A + 85 = 17 * (A + 3)) →
  (A + 3 = 37) :=
by
  intros A h
  sorry

end batsman_average_17th_innings_l6_6826


namespace find_a_b_l6_6600

noncomputable def f (a b c x : ℝ) : ℝ := a * x^3 + b * x + c

theorem find_a_b (a b c : ℝ) (h1 : (12 * a + b = 0)) (h2 : (4 * a + b = -3)) :
  a = 3 / 8 ∧ b = -9 / 2 := by
  sorry

end find_a_b_l6_6600


namespace sums_of_adjacent_cells_l6_6976

theorem sums_of_adjacent_cells (N : ℕ) (h : N ≥ 2) :
  ∃ (f : ℕ → ℕ → ℝ), (∀ i j, 1 ≤ i ∧ i < N → 1 ≤ j ∧ j < N → 
    (∃ (s : ℕ), 1 ≤ s ∧ s ≤ 2*(N-1)*N ∧ s = f i j + f (i + 1) j) ∧
    (∃ (s : ℕ), 1 ≤ s ∧ s ≤ 2*(N-1)*N ∧ s = f i j + f i (j + 1))) := sorry

end sums_of_adjacent_cells_l6_6976


namespace total_number_of_cars_l6_6907

theorem total_number_of_cars (T A R : ℕ)
  (h1 : T - A = 37)
  (h2 : R ≥ 41)
  (h3 : ∀ x, x ≤ 59 → A = x + 37) :
  T = 133 :=
by
  sorry

end total_number_of_cars_l6_6907


namespace no_real_roots_l6_6141

-- Define the coefficients of the quadratic equation
def a : ℝ := 1
def b : ℝ := 2
def c : ℝ := 4

-- Define the quadratic equation
def quadratic_eqn (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the discriminant
def discriminant : ℝ := b^2 - 4 * a * c

-- State the theorem: The quadratic equation has no real roots because the discriminant is negative
theorem no_real_roots : discriminant < 0 := by
  unfold discriminant
  unfold a b c
  sorry

end no_real_roots_l6_6141


namespace resistance_per_band_is_10_l6_6934

noncomputable def resistance_per_band := 10
def total_squat_weight := 30
def dumbbell_weight := 10
def number_of_bands := 2

theorem resistance_per_band_is_10 :
  (total_squat_weight - dumbbell_weight) / number_of_bands = resistance_per_band := 
by
  sorry

end resistance_per_band_is_10_l6_6934


namespace geometric_sequence_fifth_term_l6_6147

theorem geometric_sequence_fifth_term (α : ℕ → ℝ) (h : α 4 * α 5 * α 6 = 27) : α 5 = 3 :=
sorry

end geometric_sequence_fifth_term_l6_6147


namespace smallest_marbles_l6_6834

theorem smallest_marbles
  : ∃ n : ℕ, ((n % 8 = 5) ∧ (n % 7 = 2) ∧ (n = 37) ∧ (37 % 9 = 1)) :=
by
  sorry

end smallest_marbles_l6_6834


namespace evaluate_expression_l6_6168

theorem evaluate_expression : (4 + 6 + 7) / 3 - 2 / 3 = 5 := by
  sorry

end evaluate_expression_l6_6168


namespace tangent_line_at_one_m_positive_if_equal_vals_ineq_if_equal_vals_l6_6091

noncomputable def f (x m : ℝ) : ℝ := (Real.exp (x - 1) - 0.5 * x^2 + x - m * Real.log x)

theorem tangent_line_at_one (m : ℝ) :
  ∃ (y : ℝ → ℝ), (∀ x, y x = (1 - m) * x + m + 0.5) ∧ y 1 = f 1 m ∧ (tangent_slope : ℝ) = 1 - m ∧
    ∀ x, y x = f x m + y 0 :=
sorry

theorem m_positive_if_equal_vals (m x₁ x₂ : ℝ) (h₁ : x₁ ≠ x₂) (h₂ : f x₁ m = f x₂ m) :
  m > 0 :=
sorry

theorem ineq_if_equal_vals (m x₁ x₂ : ℝ) (h₁ : x₁ ≠ x₂) (h₂ : f x₁ m = f x₂ m) :
  2 * m > Real.exp (Real.log x₁ + Real.log x₂) :=
sorry

end tangent_line_at_one_m_positive_if_equal_vals_ineq_if_equal_vals_l6_6091


namespace amount_paid_for_peaches_l6_6878

noncomputable def cost_of_berries : ℝ := 7.19
noncomputable def change_received : ℝ := 5.98
noncomputable def total_bill : ℝ := 20

theorem amount_paid_for_peaches :
  total_bill - change_received - cost_of_berries = 6.83 :=
by
  sorry

end amount_paid_for_peaches_l6_6878


namespace infinite_values_prime_divisor_l6_6052

noncomputable def largestPrimeDivisor (n : ℕ) : ℕ :=
  sorry

theorem infinite_values_prime_divisor :
  ∃ᶠ n in at_top, largestPrimeDivisor (n^2 + n + 1) = largestPrimeDivisor ((n+1)^2 + (n+1) + 1) :=
sorry

end infinite_values_prime_divisor_l6_6052


namespace solve_system_l6_6655

theorem solve_system (x y z : ℝ) 
  (h1 : x^3 - y = 6)
  (h2 : y^3 - z = 6)
  (h3 : z^3 - x = 6) :
  x = 2 ∧ y = 2 ∧ z = 2 :=
by sorry

end solve_system_l6_6655


namespace shifted_function_is_correct_l6_6110

def original_function (x : ℝ) : ℝ :=
  (x - 1)^2 + 2

def shifted_up_function (x : ℝ) : ℝ :=
  original_function x + 3

def shifted_right_function (x : ℝ) : ℝ :=
  shifted_up_function (x - 4)

theorem shifted_function_is_correct : ∀ x : ℝ, shifted_right_function x = (x - 5)^2 + 5 := 
by
  sorry

end shifted_function_is_correct_l6_6110


namespace time_to_fill_box_correct_l6_6183

def total_toys := 50
def mom_rate := 5
def mia_rate := 3

def time_to_fill_box (total_toys mom_rate mia_rate : ℕ) : ℚ :=
  let net_rate_per_cycle := mom_rate - mia_rate
  let cycles := ((total_toys - 1) / net_rate_per_cycle) + 1
  let total_seconds := cycles * 30
  total_seconds / 60

theorem time_to_fill_box_correct : time_to_fill_box total_toys mom_rate mia_rate = 12.5 :=
by
  sorry

end time_to_fill_box_correct_l6_6183


namespace distance_travelled_l6_6266

variables (D : ℝ) (h_eq : D / 10 = (D + 20) / 14)

theorem distance_travelled : D = 50 :=
by sorry

end distance_travelled_l6_6266


namespace total_children_l6_6872

theorem total_children (sons daughters : ℕ) (h1 : sons = 3) (h2 : daughters = 6 * sons) : (sons + daughters) = 21 :=
by
  sorry

end total_children_l6_6872


namespace option_a_is_correct_l6_6575

theorem option_a_is_correct (a b : ℝ) :
  (a - b) * (-a - b) = b^2 - a^2 :=
sorry

end option_a_is_correct_l6_6575


namespace perpendicular_planes_parallel_l6_6628

-- Define the lines m and n, and planes alpha and beta
def Line := Unit
def Plane := Unit

-- Define perpendicular and parallel relations
def perpendicular (l : Line) (p : Plane) : Prop := sorry
def parallel (p₁ p₂ : Plane) : Prop := sorry

-- The main theorem statement: If m ⊥ α and m ⊥ β, then α ∥ β
theorem perpendicular_planes_parallel (m : Line) (α β : Plane)
  (h₁ : perpendicular m α) (h₂ : perpendicular m β) : parallel α β :=
sorry

end perpendicular_planes_parallel_l6_6628


namespace minimum_squares_and_perimeter_l6_6970

theorem minimum_squares_and_perimeter 
  (length width : ℕ) 
  (h_length : length = 90) 
  (h_width : width = 42) 
  (h_gcd : Nat.gcd length width = 6) 
  : 
  ((length / Nat.gcd length width) * (width / Nat.gcd length width) = 105) ∧ 
  (105 * (4 * Nat.gcd length width) = 2520) := 
by 
  sorry

end minimum_squares_and_perimeter_l6_6970


namespace sum_of_207_instances_of_33_difference_when_25_instances_of_112_are_subtracted_from_3000_difference_between_product_and_sum_of_12_and_13_l6_6178

-- 1. Prove that 33 * 207 = 6831
theorem sum_of_207_instances_of_33 : 33 * 207 = 6831 := by
    sorry

-- 2. Prove that 3000 - 112 * 25 = 200
theorem difference_when_25_instances_of_112_are_subtracted_from_3000 : 3000 - 112 * 25 = 200 := by
    sorry

-- 3. Prove that 12 * 13 - (12 + 13) = 131
theorem difference_between_product_and_sum_of_12_and_13 : 12 * 13 - (12 + 13) = 131 := by
    sorry

end sum_of_207_instances_of_33_difference_when_25_instances_of_112_are_subtracted_from_3000_difference_between_product_and_sum_of_12_and_13_l6_6178


namespace complement_A_U_l6_6361

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4}

-- Define the set A
def A : Set ℕ := {1, 3}

-- Define the complement of A with respect to U
def C_U_A : Set ℕ := U \ A

-- Theorem: The complement of A with respect to U is {2, 4}
theorem complement_A_U : C_U_A = {2, 4} := by
  sorry

end complement_A_U_l6_6361


namespace sum_of_corners_of_9x9_grid_l6_6944

theorem sum_of_corners_of_9x9_grid : 
    let topLeft := 1
    let topRight := 9
    let bottomLeft := 73
    let bottomRight := 81
    topLeft + topRight + bottomLeft + bottomRight = 164 :=
by {
  sorry
}

end sum_of_corners_of_9x9_grid_l6_6944


namespace cone_generatrix_length_theorem_l6_6265

noncomputable def cone_generatrix_length 
  (diameter : ℝ) 
  (unfolded_side_area : ℝ) 
  (h_diameter : diameter = 6)
  (h_area : unfolded_side_area = 18 * Real.pi) : 
  ℝ :=
6

theorem cone_generatrix_length_theorem 
  (diameter : ℝ) 
  (unfolded_side_area : ℝ) 
  (h_diameter : diameter = 6)
  (h_area : unfolded_side_area = 18 * Real.pi) :
  cone_generatrix_length diameter unfolded_side_area h_diameter h_area = 6 :=
sorry

end cone_generatrix_length_theorem_l6_6265


namespace functional_eq_solutions_l6_6251

theorem functional_eq_solutions
  (f : ℚ → ℚ)
  (h0 : f 0 = 0)
  (h1 : ∀ x y : ℚ, f (f x + f y) = x + y) :
  ∀ x : ℚ, f x = x ∨ f x = -x := 
sorry

end functional_eq_solutions_l6_6251


namespace total_fencing_costs_l6_6604

theorem total_fencing_costs (c1 c2 c3 c4 l1 l2 l3 : ℕ) 
    (h_c1 : c1 = 79) (h_c2 : c2 = 92) (h_c3 : c3 = 85) (h_c4 : c4 = 96)
    (h_l1 : l1 = 5) (h_l2 : l2 = 7) (h_l3 : l3 = 9) :
    (c1 + c2 + c3 + c4) * l1 = 1760 ∧ 
    (c1 + c2 + c3 + c4) * l2 = 2464 ∧ 
    (c1 + c2 + c3 + c4) * l3 = 3168 := 
by {
    sorry -- Proof to be constructed
}

end total_fencing_costs_l6_6604


namespace solve_system_l6_6596

theorem solve_system :
  (∃ x y : ℝ, 4 * x + y = 5 ∧ 2 * x - 3 * y = 13) ↔ (x = 2 ∧ y = -3) :=
by
  sorry

end solve_system_l6_6596


namespace smallest_n_inverse_mod_1176_l6_6711

theorem smallest_n_inverse_mod_1176 : ∃ n : ℕ, n > 1 ∧ Nat.Coprime n 1176 ∧ (∀ m : ℕ, m > 1 ∧ Nat.Coprime m 1176 → n ≤ m) ∧ n = 5 := by
  sorry

end smallest_n_inverse_mod_1176_l6_6711


namespace calculate_f_one_l6_6430

def f (x : ℝ) : ℝ := x^2 + |x - 2|

theorem calculate_f_one : f 1 = 2 := by
  sorry

end calculate_f_one_l6_6430


namespace simplify_expression_l6_6275

theorem simplify_expression (b : ℝ) (hb : b = -1) : 
  (3 * b⁻¹ + (2 * b⁻¹) / 3) / b = 11 / 3 :=
by
  sorry

end simplify_expression_l6_6275


namespace money_needed_l6_6061

def phone_cost : ℕ := 1300
def mike_fraction : ℚ := 0.4

theorem money_needed : mike_fraction * phone_cost + 780 = phone_cost := by
  sorry

end money_needed_l6_6061


namespace exists_multiple_l6_6030

theorem exists_multiple (n : ℕ) (a : Fin (n + 1) → ℕ) 
  (h : ∀ i, a i > 0) 
  (h2 : ∀ i, a i ≤ 2 * n) : 
  ∃ i j : Fin (n + 1), i ≠ j ∧ (a i ∣ a j ∨ a j ∣ a i) :=
by
sorry

end exists_multiple_l6_6030


namespace period_start_time_l6_6278

theorem period_start_time (end_time : ℕ) (rained_hours : ℕ) (not_rained_hours : ℕ) (total_hours : ℕ) (start_time : ℕ) 
  (h1 : end_time = 17) -- 5 pm as 17 in 24-hour format 
  (h2 : rained_hours = 2)
  (h3 : not_rained_hours = 6)
  (h4 : total_hours = rained_hours + not_rained_hours)
  (h5 : total_hours = 8)
  (h6 : start_time = end_time - total_hours)
  : start_time = 9 :=
sorry

end period_start_time_l6_6278


namespace combine_exponent_remains_unchanged_l6_6949

-- Define combining like terms condition
def combining_like_terms (terms : List (ℕ × String)) : List (ℕ × String) := sorry

-- Define the problem statement
theorem combine_exponent_remains_unchanged (terms : List (ℕ × String)) : 
  (combining_like_terms terms).map Prod.snd = terms.map Prod.snd :=
sorry

end combine_exponent_remains_unchanged_l6_6949


namespace mark_has_3_tanks_l6_6924

-- Define conditions
def pregnant_fish_per_tank : ℕ := 4
def young_per_fish : ℕ := 20
def total_young : ℕ := 240

-- Theorem statement that Mark has 3 tanks
theorem mark_has_3_tanks : (total_young / (pregnant_fish_per_tank * young_per_fish)) = 3 :=
by
  sorry

end mark_has_3_tanks_l6_6924


namespace find_pairs_l6_6686

-- Definitions for the conditions in the problem
def is_positive (x : ℝ) : Prop := x > 0

def equations (x y : ℝ) : Prop :=
  (Real.log (x^2 + y^2) / Real.log 10 = 2) ∧ 
  (Real.log x / Real.log 2 - 4 = Real.log 3 / Real.log 2 - Real.log y / Real.log 2)

-- Lean 4 Statement
theorem find_pairs (x y : ℝ) : 
  is_positive x ∧ is_positive y ∧ equations x y → (x, y) = (8, 6) ∨ (x, y) = (6, 8) :=
by
  sorry

end find_pairs_l6_6686


namespace median_of_first_fifteen_positive_integers_l6_6522

theorem median_of_first_fifteen_positive_integers : ∃ m : ℝ, m = 7.5 :=
by
  let seq := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
  let median := (seq[6] + seq[7]) / 2
  use median
  sorry

end median_of_first_fifteen_positive_integers_l6_6522


namespace project_presentation_period_length_l6_6469

theorem project_presentation_period_length
  (students : ℕ)
  (presentation_time_per_student : ℕ)
  (number_of_periods : ℕ)
  (total_students : students = 32)
  (time_per_student : presentation_time_per_student = 5)
  (periods_needed : number_of_periods = 4) :
  (32 * 5) / 4 = 40 := 
by {
  sorry
}

end project_presentation_period_length_l6_6469


namespace jane_needs_9_more_days_l6_6336

def jane_rate : ℕ := 16
def mark_rate : ℕ := 20
def mark_days : ℕ := 3
def total_vases : ℕ := 248

def vases_by_mark_in_3_days : ℕ := mark_rate * mark_days
def vases_by_jane_and_mark_in_3_days : ℕ := (jane_rate + mark_rate) * mark_days
def remaining_vases_after_3_days : ℕ := total_vases - vases_by_jane_and_mark_in_3_days
def days_jane_needs_alone : ℕ := (remaining_vases_after_3_days + jane_rate - 1) / jane_rate

theorem jane_needs_9_more_days :
  days_jane_needs_alone = 9 :=
by
  sorry

end jane_needs_9_more_days_l6_6336


namespace optimal_green_tiles_l6_6085

variable (n_red n_orange n_yellow n_green n_blue n_indigo : ℕ)

def conditions (n_red n_orange n_yellow n_green n_blue n_indigo : ℕ) :=
  n_indigo ≥ n_red + n_orange + n_yellow + n_green + n_blue ∧
  n_blue ≥ n_red + n_orange + n_yellow + n_green ∧
  n_green ≥ n_red + n_orange + n_yellow ∧
  n_yellow ≥ n_red + n_orange ∧
  n_orange ≥ n_red ∧
  n_red + n_orange + n_yellow + n_green + n_blue + n_indigo = 100

theorem optimal_green_tiles : 
  conditions n_red n_orange n_yellow n_green n_blue n_indigo → 
  n_green = 13 := by
    sorry

end optimal_green_tiles_l6_6085


namespace max_balls_of_clay_l6_6513

theorem max_balls_of_clay (radius cube_side_length : ℝ) (V_cube : ℝ) (V_ball : ℝ) (num_balls : ℕ) :
  radius = 3 ->
  cube_side_length = 10 ->
  V_cube = cube_side_length ^ 3 ->
  V_ball = (4 / 3) * π * radius ^ 3 ->
  num_balls = ⌊ V_cube / V_ball ⌋ ->
  num_balls = 8 :=
by
  sorry

end max_balls_of_clay_l6_6513


namespace max_height_l6_6247

-- Define the parabolic function h(t) representing the height of the soccer ball.
def h (t : ℝ) : ℝ := -20 * t^2 + 100 * t + 11

-- State that the maximum height of the soccer ball is 136 feet.
theorem max_height : ∃ t : ℝ, h t = 136 :=
by
  sorry

end max_height_l6_6247


namespace joe_initial_paint_l6_6198
-- Use necessary imports

-- Define the hypothesis
def initial_paint_gallons (g : ℝ) :=
  (1 / 4) * g + (1 / 7) * (3 / 4) * g = 128.57

-- Define the theorem
theorem joe_initial_paint (P : ℝ) (h : initial_paint_gallons P) : P = 360 :=
  sorry

end joe_initial_paint_l6_6198


namespace find_odd_natural_numbers_l6_6677

-- Definition of a friendly number
def is_friendly (n : ℕ) : Prop :=
  ∀ i, (n / 10^i) % 10 = (n / 10^(i + 1)) % 10 + 1 ∨ (n / 10^i) % 10 = (n / 10^(i + 1)) % 10 - 1

-- Given condition: n is divisible by 64m
def is_divisible_by_64m (n m : ℕ) : Prop :=
  64 * m ∣ n

-- Proof problem statement
theorem find_odd_natural_numbers (m : ℕ) (hm1 : m % 2 = 1) :
  (5 ∣ m → ¬ ∃ n, is_friendly n ∧ is_divisible_by_64m n m) ∧ 
  (¬ 5 ∣ m → ∃ n, is_friendly n ∧ is_divisible_by_64m n m) :=
by
  sorry

end find_odd_natural_numbers_l6_6677


namespace TimSpentThisMuch_l6_6411

/-- Tim's lunch cost -/
def lunchCost : ℝ := 50.50

/-- Tip percentage -/
def tipPercent : ℝ := 0.20

/-- Calculate the tip amount -/
def tipAmount := tipPercent * lunchCost

/-- Calculate the total amount spent -/
def totalAmountSpent := lunchCost + tipAmount

/-- Prove that the total amount spent is as expected -/
theorem TimSpentThisMuch : totalAmountSpent = 60.60 :=
  sorry

end TimSpentThisMuch_l6_6411


namespace unique_solution_implies_d_999_l6_6384

variable (a b c d x y : ℤ)

theorem unique_solution_implies_d_999
  (h1 : a < b)
  (h2 : b < c)
  (h3 : c < d)
  (h4 : 3 * x + y = 3005)
  (h5 : y = |x-a| + |x-b| + |x-c| + |x-d|)
  (h6 : ∃! x, 3 * x + |x-a| + |x-b| + |x-c| + |x-d| = 3005) :
  d = 999 :=
sorry

end unique_solution_implies_d_999_l6_6384


namespace simplified_polynomial_l6_6918

theorem simplified_polynomial : ∀ (x : ℝ), (3 * x + 2) * (3 * x - 2) - (3 * x - 1) ^ 2 = 6 * x - 5 := by
  sorry

end simplified_polynomial_l6_6918


namespace find_a_value_l6_6747

theorem find_a_value 
  (A : Set ℝ := {x | x^2 - 4 ≤ 0})
  (B : Set ℝ := {x | 2 * x + a ≤ 0})
  (intersection : A ∩ B = {x | -2 ≤ x ∧ x ≤ 1}) : a = -2 :=
by
  sorry

end find_a_value_l6_6747


namespace largest_N_exists_l6_6461

noncomputable def parabola_properties (a T : ℤ) :=
    (∀ (x y : ℤ), y = a * x * (x - 2 * T) → (x = 0 ∨ x = 2 * T) → y = 0) ∧ 
    (∀ (v : ℤ × ℤ), v = (2 * T + 1, 28) → 28 = a * (2 * T + 1))

theorem largest_N_exists : 
    ∃ (a T : ℤ), T ≠ 0 ∧ (∀ (P : ℤ × ℤ), P = (0, 0) ∨ P = (2 * T, 0) ∨ P = (2 * T + 1, 28)) 
    ∧ (s = T - a * T^2) ∧ s = 60 :=
sorry

end largest_N_exists_l6_6461


namespace find_f_prime_at_1_l6_6307

variable (f : ℝ → ℝ)

-- Initial condition
variable (h : ∀ x, f x = x^2 + deriv f 2 * (Real.log x - x))

-- The goal is to prove that f'(1) = 2
theorem find_f_prime_at_1 : deriv f 1 = 2 :=
by
  sorry

end find_f_prime_at_1_l6_6307


namespace g_pi_over_4_eq_neg_sqrt2_over_4_l6_6978

noncomputable def g (x : Real) : Real := 
  Real.sqrt (5 * (Real.sin x)^4 + 4 * (Real.cos x)^2) - 
  Real.sqrt (6 * (Real.cos x)^4 + 4 * (Real.sin x)^2)

theorem g_pi_over_4_eq_neg_sqrt2_over_4 :
  g (Real.pi / 4) = - (Real.sqrt 2) / 4 := 
sorry

end g_pi_over_4_eq_neg_sqrt2_over_4_l6_6978


namespace min_value_fraction_geq_3_div_2_l6_6371

theorem min_value_fraction_geq_3_div_2 (a : ℕ → ℝ) (m n : ℕ) (q : ℝ) (h1 : q > 0) 
  (h2 : ∀ k, a (k + 2) = q * a (k + 1)) (h3 : a 2016 = a 2015 + 2 * a 2014) 
  (h4 : a m * a n = 16 * (a 1) ^ 2) :
  (∃ q, q = 2 ∧ m + n = 6) → 4 / m + 1 / n ≥ 3 / 2 :=
by sorry

end min_value_fraction_geq_3_div_2_l6_6371


namespace correct_calculation_l6_6249

theorem correct_calculation (x : ℤ) (h : 20 + x = 60) : 34 - x = -6 := by
  sorry

end correct_calculation_l6_6249


namespace increasing_interval_of_g_l6_6688

noncomputable def f (x : ℝ) : ℝ :=
  (Real.cos (Real.pi / 3 - 2 * x)) -
  2 * (Real.sin (Real.pi / 4 + x) * Real.sin (Real.pi / 4 - x))

noncomputable def g (x : ℝ) : ℝ :=
  f (x + Real.pi / 12)

theorem increasing_interval_of_g :
  ∀ x ∈ Set.Icc (-Real.pi / 12) (Real.pi / 2),
  ∃ a b, a = -Real.pi / 12 ∧ b = Real.pi / 4 ∧
      (∀ x y, (a ≤ x ∧ x ≤ y ∧ y ≤ b) → g x ≤ g y) :=
sorry

end increasing_interval_of_g_l6_6688


namespace marked_price_each_article_l6_6429

noncomputable def pair_price : ℝ := 50
noncomputable def discount : ℝ := 0.60
noncomputable def marked_price_pair : ℝ := 50 / 0.40
noncomputable def marked_price_each : ℝ := marked_price_pair / 2

theorem marked_price_each_article : 
  marked_price_each = 62.50 := by
  sorry

end marked_price_each_article_l6_6429


namespace gold_coins_l6_6424

theorem gold_coins (n c : Nat) : 
  n = 9 * (c - 2) → n = 6 * c + 3 → n = 45 :=
by 
  intros h1 h2 
  sorry

end gold_coins_l6_6424


namespace max_brownie_pieces_l6_6933

theorem max_brownie_pieces (base height piece_width piece_height : ℕ) 
    (h_base : base = 30) (h_height : height = 24)
    (h_piece_width : piece_width = 3) (h_piece_height : piece_height = 4) :
  (base / piece_width) * (height / piece_height) = 60 :=
by sorry

end max_brownie_pieces_l6_6933


namespace average_output_l6_6348

theorem average_output (t1 t2 t_total : ℝ) (c1 c2 c_total : ℕ) 
                        (h1 : c1 = 60) (h2 : c2 = 60) 
                        (rate1 : ℝ := 15) (rate2 : ℝ := 60) :
  t1 = c1 / rate1 ∧ t2 = c2 / rate2 ∧ t_total = t1 + t2 ∧ c_total = c1 + c2 → 
  (c_total / t_total = 24) := 
by 
  sorry

end average_output_l6_6348


namespace amount_left_after_spending_l6_6791

-- Definitions based on conditions
def initial_amount : ℕ := 204
def amount_spent_on_toy (initial : ℕ) : ℕ := initial / 2
def remaining_after_toy (initial : ℕ) : ℕ := initial - amount_spent_on_toy initial
def amount_spent_on_book (remaining : ℕ) : ℕ := remaining / 2
def remaining_after_book (remaining : ℕ) : ℕ := remaining - amount_spent_on_book remaining

-- Proof statement
theorem amount_left_after_spending : 
  remaining_after_book (remaining_after_toy initial_amount) = 51 :=
sorry

end amount_left_after_spending_l6_6791


namespace bamboo_break_height_l6_6914

theorem bamboo_break_height (x : ℝ) (h₁ : 0 < x) (h₂ : x < 9) (h₃ : x^2 + 3^2 = (9 - x)^2) : x = 4 :=
by
  sorry

end bamboo_break_height_l6_6914


namespace jenna_costume_l6_6233

def cost_of_skirts (skirt_count : ℕ) (material_per_skirt : ℕ) : ℕ :=
  skirt_count * material_per_skirt

def cost_of_bodice (shirt_material : ℕ) (sleeve_material_per : ℕ) (sleeve_count : ℕ) : ℕ :=
  shirt_material + (sleeve_material_per * sleeve_count)

def total_material (skirt_material : ℕ) (bodice_material : ℕ) : ℕ :=
  skirt_material + bodice_material

def total_cost (total_material : ℕ) (cost_per_sqft : ℕ) : ℕ :=
  total_material * cost_per_sqft

theorem jenna_costume : 
  cost_of_skirts 3 48 + cost_of_bodice 2 5 2 = 156 → total_cost 156 3 = 468 :=
by
  sorry

end jenna_costume_l6_6233


namespace other_number_in_product_l6_6054

theorem other_number_in_product (w : ℕ) (n : ℕ) (hw_pos : 0 < w) (n_factor : Nat.lcm (2^5) (Nat.gcd  864 w) = 2^5 * 3^3) (h_w : w = 144) : n = 6 :=
by
  -- proof would go here
  sorry

end other_number_in_product_l6_6054


namespace range_f_x1_x2_l6_6402

noncomputable def f (c x : ℝ) : ℝ := 2 * x ^ 3 - 3 * x ^ 2 + c * x + 1

theorem range_f_x1_x2 (c x1 x2 : ℝ) (h1 : 0 < x1) (h2 : 0 < x2) (h3 : x1 < x2) 
  (h4 : 36 - 24 * c > 0) (h5 : ∀ x, f c x = 2 * x ^ 3 - 3 * x ^ 2 + c * x + 1) :
  1 < f c x1 / x2 ∧ f c x1 / x2 < 5 / 2 :=
sorry

end range_f_x1_x2_l6_6402


namespace problem_statement_l6_6802

-- Definitions for conditions
def cond_A : Prop := ∃ B : ℝ, B = 45 ∨ B = 135
def cond_B : Prop := ∃ C : ℝ, C = 90
def cond_C : Prop := false
def cond_D : Prop := ∃ B : ℝ, 0 < B ∧ B < 60

-- Prove that only cond_A has two possibilities
theorem problem_statement : cond_A ∧ ¬cond_B ∧ ¬cond_C ∧ ¬cond_D :=
by 
  -- Lean proof goes here
  sorry

end problem_statement_l6_6802


namespace find_a2_l6_6457

variable (x : ℝ)
variable (a₀ a₁ a₂ a₃ : ℝ)
axiom condition : ∀ x, x^3 = a₀ + a₁ * (x - 2) + a₂ * (x - 2)^2 + a₃ * (x - 2)^3

theorem find_a2 : a₂ = 6 :=
by
  -- The proof that involves verifying the Taylor series expansion will come here
  sorry

end find_a2_l6_6457


namespace triangle_sides_l6_6508

theorem triangle_sides
  (D E : ℝ) (DE EF : ℝ)
  (h1 : Real.cos (2 * D - E) + Real.sin (D + E) = 2)
  (hDE : DE = 6) :
  EF = 3 :=
by
  -- Proof is omitted
  sorry

end triangle_sides_l6_6508


namespace drummer_difference_l6_6714

def flute_players : Nat := 5
def trumpet_players : Nat := 3 * flute_players
def trombone_players : Nat := trumpet_players - 8
def clarinet_players : Nat := 2 * flute_players
def french_horn_players : Nat := trombone_players + 3
def total_seats_needed : Nat := 65
def total_seats_taken : Nat := flute_players + trumpet_players + trombone_players + clarinet_players + french_horn_players
def drummers : Nat := total_seats_needed - total_seats_taken

theorem drummer_difference : drummers - trombone_players = 11 := by
  sorry

end drummer_difference_l6_6714


namespace find_g_neg_six_l6_6220

theorem find_g_neg_six (g : ℤ → ℤ)
  (h1 : g 1 - 1 > 0)
  (h2 : ∀ x y : ℤ, g x * g y + x + y + x * y = g (x + y) + x * g y + y * g x)
  (h3 : ∀ x : ℤ, 3 * g (x + 1) = g x + 2 * x + 3) :
  g (-6) = -20 :=
sorry

end find_g_neg_six_l6_6220


namespace circumradius_eq_exradius_opposite_BC_l6_6324

-- Definitions of points and triangles
variable {A B C : Point}
variable (O I D : Point)
variable {α β γ : Angle}

-- Definitions of circumcenter, incenter, altitude, and collinearity
def is_circumcenter (O : Point) (A B C : Point) : Prop := sorry
def is_incenter (I : Point) (A B C : Point) : Prop := sorry
def is_altitude (A D B C : Point) : Prop := sorry
def collinear (O D I : Point) : Prop := sorry

-- Definitions of circumradius and exradius
def circumradius (A B C : Point) : ℝ := sorry
def exradius_opposite_BC (A B C : Point) : ℝ := sorry

-- Main theorem statement
theorem circumradius_eq_exradius_opposite_BC
  (h_circ : is_circumcenter O A B C)
  (h_incenter : is_incenter I A B C)
  (h_altitude : is_altitude A D B C)
  (h_collinear : collinear O D I) : 
  circumradius A B C = exradius_opposite_BC A B C :=
sorry

end circumradius_eq_exradius_opposite_BC_l6_6324


namespace part1_part2_part3_l6_6388

open Real

-- Definition of "$k$-derived point"
def k_derived_point (P : ℝ × ℝ) (k : ℝ) : ℝ × ℝ := (P.1 + k * P.2, k * P.1 + P.2)

-- Problem statements to prove
theorem part1 :
  k_derived_point (-2, 3) 2 = (4, -1) :=
sorry

theorem part2 (P : ℝ × ℝ) (h : k_derived_point P 3 = (9, 11)) :
  P = (3, 2) :=
sorry

theorem part3 (b k : ℝ) (h1 : b > 0) (h2 : |k * b| ≥ 5 * b) :
  k ≥ 5 ∨ k ≤ -5 :=
sorry

end part1_part2_part3_l6_6388


namespace speed_of_man_in_still_water_l6_6322

variable (v_m v_s : ℝ)

-- Conditions
def downstream_distance : ℝ := 51
def upstream_distance : ℝ := 18
def time : ℝ := 3

-- Equations based on the conditions
def downstream_speed_eq : Prop := downstream_distance = (v_m + v_s) * time
def upstream_speed_eq : Prop := upstream_distance = (v_m - v_s) * time

-- The theorem to prove
theorem speed_of_man_in_still_water : downstream_speed_eq v_m v_s ∧ upstream_speed_eq v_m v_s → v_m = 11.5 :=
by
  intro h
  sorry

end speed_of_man_in_still_water_l6_6322


namespace set_operation_empty_l6_6018

-- Definition of the universal set I, and sets P and Q with the given properties
variable {I : Set ℕ} -- Universal set
variable {P Q : Set ℕ} -- Non-empty sets with P ⊂ Q ⊂ I
variable (hPQ : P ⊂ Q) (hQI : Q ⊂ I)

-- Prove the set operation expression that results in the empty set
theorem set_operation_empty :
  ∃ (P Q : Set ℕ), P ⊂ Q ∧ Q ⊂ I ∧ P ≠ ∅ ∧ Q ≠ ∅ → 
  P ∩ (I \ Q) = ∅ :=
by
  sorry

end set_operation_empty_l6_6018


namespace father_age_three_times_xiaojun_after_years_l6_6162

theorem father_age_three_times_xiaojun_after_years (years_passed : ℕ) (xiaojun_current_age : ℕ) (father_current_age : ℕ) 
  (h1 : xiaojun_current_age = 5) (h2 : father_current_age = 31) (h3 : years_passed = 8) :
  father_current_age + years_passed = 3 * (xiaojun_current_age + years_passed) := by
  sorry

end father_age_three_times_xiaojun_after_years_l6_6162


namespace probability_three_or_more_same_l6_6082

-- Let us define the total number of outcomes when rolling 5 8-sided dice
def total_outcomes : ℕ := 8 ^ 5

-- Define the number of favorable outcomes where at least three dice show the same number
def favorable_outcomes : ℕ := 4208

-- Define the probability as a fraction
def probability : ℚ := favorable_outcomes / total_outcomes

-- Now we state the theorem that this probability simplifies to 1052/8192
theorem probability_three_or_more_same : probability = 1052 / 8192 :=
sorry

end probability_three_or_more_same_l6_6082


namespace tan_triple_angle_l6_6820

theorem tan_triple_angle (θ : ℝ) (h : Real.tan θ = 3) : Real.tan (3 * θ) = 9 / 13 :=
by
  sorry

end tan_triple_angle_l6_6820


namespace final_result_always_4_l6_6390

-- The function that performs the operations described in the problem
def transform (x : Nat) : Nat :=
  let step1 := 2 * x
  let step2 := step1 + 3
  let step3 := step2 * 5
  let step4 := step3 + 7
  let last_digit := step4 % 10
  let step6 := last_digit + 18
  step6 / 5

-- The theorem statement claiming that for any single-digit number x, the result of transform x is always 4
theorem final_result_always_4 (x : Nat) (h : x < 10) : transform x = 4 := by
  sorry

end final_result_always_4_l6_6390


namespace parallel_lines_condition_l6_6603

theorem parallel_lines_condition (a : ℝ) : 
    (∀ x y : ℝ, 2 * x + a * y + 2 ≠ (a - 1) * x + y - 2) ↔ a = 2 := 
sorry

end parallel_lines_condition_l6_6603


namespace parabola_point_distance_to_focus_l6_6551

theorem parabola_point_distance_to_focus :
  ∀ (x y : ℝ), (y^2 = 12 * x) → (∃ (xf : ℝ), xf = 3 ∧ 0 ≤ y) → (∃ (d : ℝ), d = 7) → x = 4 :=
by
  intros x y parabola_focus distance_to_focus distance
  sorry

end parabola_point_distance_to_focus_l6_6551


namespace sum_of_percentages_l6_6793

theorem sum_of_percentages : (20 / 100 : ℝ) * 40 + (25 / 100 : ℝ) * 60 = 23 := 
by 
  -- Sorry skips the proof
  sorry

end sum_of_percentages_l6_6793


namespace largest_divisor_of_odd_product_l6_6298

theorem largest_divisor_of_odd_product (n : ℕ) (h : Even n ∧ n > 0) :
  ∃ m, m > 0 ∧ (∀ k, (n+1)*(n+3)*(n+7)*(n+9)*(n+11) % k = 0 ↔ k ≤ 15) := by
  -- Proof goes here
  sorry

end largest_divisor_of_odd_product_l6_6298


namespace right_triangle_hypotenuse_l6_6617

theorem right_triangle_hypotenuse (a b : ℕ) (h₁ : a = 3) (h₂ : b = 5) : 
  ∃ h : ℝ, h = Real.sqrt (a^2 + b^2) ∧ h = Real.sqrt 34 := 
by
  sorry

end right_triangle_hypotenuse_l6_6617


namespace length_of_train_l6_6438

noncomputable def speed_kmh_to_ms (speed_kmh : ℝ) : ℝ :=
  speed_kmh * (1000 / 3600)

noncomputable def total_distance (speed_m_s : ℝ) (time_s : ℝ) : ℝ :=
  speed_m_s * time_s

noncomputable def train_length (total_distance : ℝ) (bridge_length : ℝ) : ℝ :=
  total_distance - bridge_length

theorem length_of_train
  (speed_kmh : ℝ)
  (time_s : ℝ)
  (bridge_length : ℝ)
  (speed_in_kmh : speed_kmh = 45)
  (time_in_seconds : time_s = 30)
  (length_of_bridge : bridge_length = 220.03) :
  train_length (total_distance (speed_kmh_to_ms speed_kmh) time_s) bridge_length = 154.97 :=
by
  sorry

end length_of_train_l6_6438


namespace prove_q_l6_6119

-- Assume the conditions
variable (p q : Prop)
variable (hpq : p ∨ q) -- "p or q" is true
variable (hnp : ¬p)    -- "not p" is true

-- The theorem to prove q is true
theorem prove_q : q :=
by {
  sorry
}

end prove_q_l6_6119


namespace cylinder_height_relation_l6_6444

theorem cylinder_height_relation (r1 r2 h1 h2 V1 V2 : ℝ) 
  (h_volumes_equal : V1 = V2)
  (h_r2_gt_r1 : r2 = 1.1 * r1)
  (h_volume_first : V1 = π * r1^2 * h1)
  (h_volume_second : V2 = π * r2^2 * h2) : 
  h1 = 1.21 * h2 :=
by 
  sorry

end cylinder_height_relation_l6_6444


namespace problem_a_problem_b_l6_6906

-- Problem a
theorem problem_a (p q : ℕ) (h1 : ∃ n : ℤ, 2 * p - q = n^2) (h2 : ∃ m : ℤ, 2 * p + q = m^2) : ∃ k : ℤ, q = 2 * k :=
sorry

-- Problem b
theorem problem_b (m : ℕ) (h1 : ∃ n : ℕ, 2 * m - 4030 = n^2) (h2 : ∃ k : ℕ, 2 * m + 4030 = k^2) : (m = 2593 ∨ m = 12097 ∨ m = 81217 ∨ m = 2030113) :=
sorry

end problem_a_problem_b_l6_6906


namespace miles_walked_on_Tuesday_l6_6641

theorem miles_walked_on_Tuesday (monday_miles total_miles : ℕ) (hmonday : monday_miles = 9) (htotal : total_miles = 18) :
  total_miles - monday_miles = 9 :=
by
  sorry

end miles_walked_on_Tuesday_l6_6641


namespace math_problem_l6_6185

-- Definitions of conditions
def cond1 (x a y b z c : ℝ) : Prop := x / a + y / b + z / c = 1
def cond2 (x a y b z c : ℝ) : Prop := a / x + b / y + c / z = 0

-- Theorem statement
theorem math_problem (x a y b z c : ℝ)
  (h1 : cond1 x a y b z c) (h2 : cond2 x a y b z c) :
  (x^2 / a^2) + (y^2 / b^2) + (z^2 / c^2) = 1 :=
by
  sorry

end math_problem_l6_6185


namespace vector_BC_l6_6315

def vector_sub (u v : ℝ × ℝ) : ℝ × ℝ :=
  (u.1 - v.1, u.2 - v.2)

theorem vector_BC (BA CA BC : ℝ × ℝ) (BA_def : BA = (1, 2)) (CA_def : CA = (4, 5)) (BC_def : BC = vector_sub BA CA) : BC = (-3, -3) :=
by
  subst BA_def
  subst CA_def
  subst BC_def
  sorry

end vector_BC_l6_6315


namespace students_in_class_l6_6511

theorem students_in_class (S : ℕ) 
  (h1 : (1 / 4) * (9 / 10 : ℚ) * S = 9) : S = 40 :=
sorry

end students_in_class_l6_6511


namespace chess_piece_max_visitable_squares_l6_6081

-- Define initial board properties and movement constraints
structure ChessBoard :=
  (rows : ℕ)
  (columns : ℕ)
  (movement : ℕ)
  (board_size : rows * columns = 225)

-- Define condition for unique visitation
def can_visit (movement : ℕ) (board_size : ℕ) : Prop :=
  ∃ (max_squares : ℕ), (max_squares ≤ board_size) ∧ (max_squares = 196)

-- Main theorem statement 
theorem chess_piece_max_visitable_squares (cb : ChessBoard) : 
  can_visit 196 225 :=
by sorry

end chess_piece_max_visitable_squares_l6_6081


namespace mutually_exclusive_but_not_complementary_l6_6945

open Classical

namespace CardDistribution

inductive Card
| red | yellow | blue | white

inductive Person
| A | B | C | D

def Event_A_gets_red (distrib: Person → Card) : Prop :=
  distrib Person.A = Card.red

def Event_D_gets_red (distrib: Person → Card) : Prop :=
  distrib Person.D = Card.red

theorem mutually_exclusive_but_not_complementary :
  ∀ (distrib: Person → Card),
  (Event_A_gets_red distrib → ¬Event_D_gets_red distrib) ∧
  ¬(∀ (distrib: Person → Card), Event_A_gets_red distrib ∨ Event_D_gets_red distrib) := 
by
  sorry

end CardDistribution

end mutually_exclusive_but_not_complementary_l6_6945


namespace diff_of_squares_l6_6112

theorem diff_of_squares (a b : ℕ) : 
  (∃ x y : ℤ, a = x^2 - y^2) ∨ (∃ x y : ℤ, b = x^2 - y^2) ∨ (∃ x y : ℤ, a + b = x^2 - y^2) :=
sorry

end diff_of_squares_l6_6112


namespace largest_number_among_l6_6984

theorem largest_number_among (π: ℝ) (sqrt_2: ℝ) (neg_2: ℝ) (three: ℝ)
  (h1: 3.14 ≤ π)
  (h2: 1 < sqrt_2 ∧ sqrt_2 < 2)
  (h3: neg_2 < 1)
  (h4: 3 < π) :
  (neg_2 < sqrt_2) ∧ (sqrt_2 < 3) ∧ (3 < π) :=
by {
  sorry
}

end largest_number_among_l6_6984


namespace cars_left_in_parking_lot_l6_6595

-- Define constants representing the initial number of cars and cars that went out.
def initial_cars : ℕ := 24
def first_out : ℕ := 8
def second_out : ℕ := 6

-- State the theorem to prove the remaining cars in the parking lot.
theorem cars_left_in_parking_lot : 
  initial_cars - first_out - second_out = 10 := 
by {
  -- Here, 'sorry' is used to indicate the proof is omitted.
  sorry
}

end cars_left_in_parking_lot_l6_6595


namespace smallest_number_with_2020_divisors_l6_6502

theorem smallest_number_with_2020_divisors :
  ∃ n : ℕ, 
  (∀ n : ℕ, (∃ (p : ℕ) (α : ℕ), n = p^α) → 
  ∃ (p1 p2 p3 p4 : ℕ) (α1 α2 α3 α4 : ℕ), 
  n = p1^α1 * p2^α2 * p3^α3 * p4^α4 ∧ 
  (α1 + 1) * (α2 + 1) * (α3 + 1) * (α4 + 1) = 2020) → 
  n = 2^100 * 3^4 * 5 * 7 :=
sorry

end smallest_number_with_2020_divisors_l6_6502


namespace correct_input_statement_l6_6383

-- Definitions based on the conditions
def input_format_A : Prop := sorry
def input_format_B : Prop := sorry
def input_format_C : Prop := sorry
def output_format_D : Prop := sorry

-- The main statement we need to prove
theorem correct_input_statement : input_format_A ∧ ¬ input_format_B ∧ ¬ input_format_C ∧ ¬ output_format_D := 
by sorry

end correct_input_statement_l6_6383


namespace intersection_A_B_l6_6975

def A : Set ℤ := {0, 1, 2}
def B : Set ℤ := {x | x - 2 < 0}

theorem intersection_A_B : A ∩ B = {0, 1} :=
by
  sorry

end intersection_A_B_l6_6975


namespace log2_3_value_l6_6252

variables (a b log2 log3 : ℝ)

-- Define the conditions
axiom h1 : a = log2 + log3
axiom h2 : b = 1 + log2

-- Define the logarithmic requirement to be proved
theorem log2_3_value : log2 * log3 = (a - b + 1) / (b - 1) :=
sorry

end log2_3_value_l6_6252


namespace seating_arrangements_l6_6332

def valid_seating_arrangements (total_seats : ℕ) (people : ℕ) : ℕ :=
  if total_seats = 8 ∧ people = 3 then 12 else 0

theorem seating_arrangements (total_seats people : ℕ) (h1 : total_seats = 8) (h2 : people = 3) :
  valid_seating_arrangements total_seats people = 12 :=
by
  rw [valid_seating_arrangements, h1, h2]
  simp
  done

end seating_arrangements_l6_6332


namespace second_caterer_cheaper_l6_6453

theorem second_caterer_cheaper (x : ℕ) :
  (∀ n : ℕ, n < x → 150 + 18 * n ≤ 250 + 15 * n) ∧ (150 + 18 * x > 250 + 15 * x) ↔ x = 34 :=
by sorry

end second_caterer_cheaper_l6_6453


namespace min_value_of_a_l6_6442

theorem min_value_of_a (a : ℝ) : 
  (∀ x > 1, x + a / (x - 1) ≥ 5) → a ≥ 4 :=
sorry

end min_value_of_a_l6_6442


namespace number_of_cows_on_farm_l6_6769

theorem number_of_cows_on_farm :
  (∀ (cows_per_week : ℤ) (six_cows_milk : ℤ) (total_milk : ℤ) (weeks : ℤ),
    cows_per_week = 6 → 
    six_cows_milk = 108 →
    total_milk = 2160 →
    weeks = 5 →
    (total_milk / (six_cows_milk / cows_per_week * weeks)) = 24) :=
by
  intros cows_per_week six_cows_milk total_milk weeks h1 h2 h3 h4
  have h_cow_milk_per_week : six_cows_milk / cows_per_week = 18 := by sorry
  have h_cow_milk_per_five_weeks : (six_cows_milk / cows_per_week) * weeks = 90 := by sorry
  have h_total_cows : total_milk / ((six_cows_milk / cows_per_week) * weeks) = 24 := by sorry
  exact h_total_cows

end number_of_cows_on_farm_l6_6769


namespace smallest_y_divisible_l6_6196

theorem smallest_y_divisible (y : ℕ) : 
  (y % 3 = 2) ∧ (y % 5 = 4) ∧ (y % 7 = 6) → y = 104 :=
by
  sorry

end smallest_y_divisible_l6_6196


namespace contingency_table_proof_l6_6670

noncomputable def probability_of_mistake (K_squared : ℝ) : ℝ :=
if K_squared > 3.841 then 0.05 else 1.0 -- placeholder definition to be refined

theorem contingency_table_proof :
  probability_of_mistake 4.013 ≤ 0.05 :=
by sorry

end contingency_table_proof_l6_6670


namespace average_speed_second_bus_l6_6738

theorem average_speed_second_bus (x : ℝ) (h1 : x > 0) :
  (12 / x) - (12 / (1.2 * x)) = 3 / 60 :=
by
  sorry

end average_speed_second_bus_l6_6738


namespace g_at_4_l6_6921

def g (x : ℝ) : ℝ := 5 * x + 6

theorem g_at_4 : g 4 = 26 :=
by
  sorry

end g_at_4_l6_6921


namespace power_inequality_l6_6199

variable (a b c : ℝ)

theorem power_inequality (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  2 * (a^3 + b^3 + c^3) ≥ a * b^2 + a^2 * b + b * c^2 + b^2 * c + a * c^2 + a^2 * c :=
by sorry

end power_inequality_l6_6199


namespace construction_company_order_l6_6468

def concrete_weight : ℝ := 0.17
def bricks_weight : ℝ := 0.17
def stone_weight : ℝ := 0.5
def total_weight : ℝ := 0.84

theorem construction_company_order :
  concrete_weight + bricks_weight + stone_weight = total_weight :=
by
  -- The proof would go here but is omitted per instructions.
  sorry

end construction_company_order_l6_6468


namespace problem_l6_6828

noncomputable def f : ℝ → ℝ := sorry

theorem problem (x : ℝ) (h : ∀ x : ℝ, f (4 * x) = 4) : f (2 * x) = 4 :=
by
  sorry

end problem_l6_6828


namespace max_perimeter_of_triangle_l6_6717

theorem max_perimeter_of_triangle (A B C a b c p : ℝ) 
  (h_angle_A : A = 2 * Real.pi / 3)
  (h_a : a = 3)
  (h_perimeter : p = a + b + c) 
  (h_sine_law : b = 2 * Real.sqrt 3 * Real.sin B ∧ c = 2 * Real.sqrt 3 * Real.sin C) :
  p ≤ 3 + 2 * Real.sqrt 3 :=
sorry

end max_perimeter_of_triangle_l6_6717


namespace percentage_of_Y_pay_X_is_paid_correct_l6_6067

noncomputable def percentage_of_Y_pay_X_is_paid
  (total_pay : ℝ) (Y_pay : ℝ) : ℝ :=
  let X_pay := total_pay - Y_pay
  (X_pay / Y_pay) * 100

theorem percentage_of_Y_pay_X_is_paid_correct :
  percentage_of_Y_pay_X_is_paid 700 318.1818181818182 = 120 := 
by
  unfold percentage_of_Y_pay_X_is_paid
  sorry

end percentage_of_Y_pay_X_is_paid_correct_l6_6067


namespace sugar_content_of_mixture_l6_6735

theorem sugar_content_of_mixture 
  (volume_juice1 : ℝ) (conc_juice1 : ℝ)
  (volume_juice2 : ℝ) (conc_juice2 : ℝ) 
  (total_volume : ℝ) (total_sugar : ℝ) 
  (resulting_sugar_content : ℝ) :
  volume_juice1 = 2 →
  conc_juice1 = 0.1 →
  volume_juice2 = 3 →
  conc_juice2 = 0.15 →
  total_volume = volume_juice1 + volume_juice2 →
  total_sugar = (conc_juice1 * volume_juice1) + (conc_juice2 * volume_juice2) →
  resulting_sugar_content = (total_sugar / total_volume) * 100 →
  resulting_sugar_content = 13 :=
by
  intros
  sorry

end sugar_content_of_mixture_l6_6735


namespace evalExpression_at_3_2_l6_6726

def evalExpression (x y : ℕ) : ℕ := 3 * x^y + 4 * y^x

theorem evalExpression_at_3_2 : evalExpression 3 2 = 59 := by
  sorry

end evalExpression_at_3_2_l6_6726


namespace solution_to_system_of_inequalities_l6_6350

variable {x y : ℝ}

theorem solution_to_system_of_inequalities :
  11 * (-1/3 : ℝ)^2 + 8 * (-1/3 : ℝ) * (2/3 : ℝ) + 8 * (2/3 : ℝ)^2 ≤ 3 ∧
  (-1/3 : ℝ) - 4 * (2/3 : ℝ) ≤ -3 :=
by
  sorry

end solution_to_system_of_inequalities_l6_6350


namespace parabola_equation_l6_6174

def is_parabola (a b c x y : ℝ) : Prop :=
  y = a*x^2 + b*x + c

def has_vertex (h k a b c : ℝ) : Prop :=
  b = -2 * a * h ∧ c = k + a * h^2 

def contains_point (a b c x y : ℝ) : Prop :=
  y = a*x^2 + b*x + c

theorem parabola_equation (a b c : ℝ) :
  has_vertex 3 (-2) a b c ∧ contains_point a b c 5 6 → 
  a = 2 ∧ b = -12 ∧ c = 16 := by
  sorry

end parabola_equation_l6_6174


namespace bug_crawl_distance_l6_6994

-- Define the conditions
def initial_position : ℤ := -2
def first_move : ℤ := -6
def second_move : ℤ := 5

-- Define the absolute difference function (distance on a number line)
def abs_diff (a b : ℤ) : ℤ :=
  abs (b - a)

-- Define the total distance crawled function
def total_distance (p1 p2 p3 : ℤ) : ℤ :=
  abs_diff p1 p2 + abs_diff p2 p3

-- Prove that total distance starting at -2, moving to -6, and then to 5 is 15 units
theorem bug_crawl_distance : total_distance initial_position first_move second_move = 15 := by
  sorry

end bug_crawl_distance_l6_6994


namespace at_least_1991_red_points_l6_6412

theorem at_least_1991_red_points (P : Fin 997 → ℝ × ℝ) :
  ∃ (R : Finset (ℝ × ℝ)), 1991 ≤ R.card ∧ (∀ (i j : Fin 997), i ≠ j → ((P i + P j) / 2) ∈ R) :=
sorry

end at_least_1991_red_points_l6_6412


namespace mean_noon_temperature_l6_6792

def temperatures : List ℝ := [79, 78, 82, 86, 88, 90, 88, 90, 89]

theorem mean_noon_temperature :
  (List.sum temperatures) / (temperatures.length) = 770 / 9 := by
  sorry

end mean_noon_temperature_l6_6792


namespace flashlight_price_percentage_l6_6232

theorem flashlight_price_percentage 
  (hoodie_price boots_price total_spent flashlight_price : ℝ)
  (discount_rate : ℝ)
  (h1 : hoodie_price = 80)
  (h2 : boots_price = 110)
  (h3 : discount_rate = 0.10)
  (h4 : total_spent = 195) 
  (h5 : total_spent = hoodie_price + ((1 - discount_rate) * boots_price) + flashlight_price) : 
  (flashlight_price / hoodie_price) * 100 = 20 :=
by
  sorry

end flashlight_price_percentage_l6_6232


namespace range_of_f_l6_6212

-- Define the function f
def f (x : ℕ) : ℕ := 3 * x - 1

-- Define the domain
def domain : Set ℕ := {x | 1 ≤ x ∧ x ≤ 4}

-- Define the range
def range : Set ℕ := {2, 5, 8, 11}

-- Lean 4 theorem statement
theorem range_of_f : 
  {y | ∃ x ∈ domain, y = f x} = range :=
by
  sorry

end range_of_f_l6_6212


namespace largest_of_w_l6_6594

variable {x y z w : ℝ}

namespace MathProof

theorem largest_of_w
  (h1 : x + 3 = y - 1)
  (h2 : x + 3 = z + 5)
  (h3 : x + 3 = w - 2) :  
  w > y ∧ w > x ∧ w > z :=
by
  sorry

end MathProof

end largest_of_w_l6_6594


namespace gcd_1443_999_l6_6961

theorem gcd_1443_999 : Nat.gcd 1443 999 = 111 := by
  sorry

end gcd_1443_999_l6_6961


namespace doritos_ratio_l6_6881

noncomputable def bags_of_chips : ℕ := 80
noncomputable def bags_per_pile : ℕ := 5
noncomputable def piles : ℕ := 4

theorem doritos_ratio (D T : ℕ) (h1 : T = bags_of_chips)
  (h2 : D = piles * bags_per_pile) :
  (D : ℚ) / T = 1 / 4 := by
  sorry

end doritos_ratio_l6_6881


namespace dans_car_mpg_l6_6636

noncomputable def milesPerGallon (distance money gas_price : ℝ) : ℝ :=
  distance / (money / gas_price)

theorem dans_car_mpg :
  let gas_price := 4
  let distance := 432
  let money := 54
  milesPerGallon distance money gas_price = 32 :=
by
  simp [milesPerGallon]
  sorry

end dans_car_mpg_l6_6636


namespace mod_remainder_l6_6279

theorem mod_remainder :
  ((85^70 + 19^32)^16) % 21 = 16 := by
  -- Given conditions
  have h1 : 85^70 % 21 = 1 := sorry
  have h2 : 19^32 % 21 = 4 := sorry
  -- Conclusion
  sorry

end mod_remainder_l6_6279


namespace factor_expression_l6_6303

theorem factor_expression (y : ℝ) : 16 * y^3 + 8 * y^2 = 8 * y^2 * (2 * y + 1) :=
by
  sorry

end factor_expression_l6_6303


namespace geo_seq_sum_l6_6813

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geo_seq_sum (a : ℕ → ℝ) (h : geometric_sequence a) (h1 : a 0 + a 1 = 30) (h4 : a 3 + a 4 = 120) :
  a 6 + a 7 = 480 :=
sorry

end geo_seq_sum_l6_6813


namespace fraction_simplification_l6_6992

theorem fraction_simplification
  (a b c x : ℝ)
  (hb : b ≠ 0)
  (hxc : c ≠ 0)
  (h : x = a / b)
  (ha : a ≠ c * b) :
  (a + c * b) / (a - c * b) = (x + c) / (x - c) :=
by
  sorry

end fraction_simplification_l6_6992


namespace chord_length_perpendicular_bisector_l6_6258

theorem chord_length_perpendicular_bisector (r : ℝ) (h : r = 10) :
  ∃ (CD : ℝ), CD = 10 * Real.sqrt 3 :=
by
  -- The proof is omitted.
  sorry

end chord_length_perpendicular_bisector_l6_6258


namespace equal_roots_quadratic_eq_l6_6915

theorem equal_roots_quadratic_eq (m n : ℝ) (h : m^2 - 4 * n = 0) : m = 2 ∧ n = 1 :=
by
  sorry

end equal_roots_quadratic_eq_l6_6915


namespace paco_cookies_l6_6176

theorem paco_cookies (initial_cookies: ℕ) (eaten_cookies: ℕ) (final_cookies: ℕ) (bought_cookies: ℕ) 
  (h1 : initial_cookies = 40)
  (h2 : eaten_cookies = 2)
  (h3 : final_cookies = 75)
  (h4 : initial_cookies - eaten_cookies + bought_cookies = final_cookies) :
  bought_cookies = 37 :=
by
  rw [h1, h2, h3] at h4
  sorry

end paco_cookies_l6_6176


namespace length_of_bridge_l6_6623

def length_of_train : ℝ := 135  -- Length of the train in meters
def speed_of_train_km_per_hr : ℝ := 45  -- Speed of the train in km/hr
def speed_of_train_m_per_s : ℝ := 12.5  -- Speed of the train in m/s
def time_to_cross_bridge : ℝ := 30  -- Time to cross the bridge in seconds
def distance_covered : ℝ := speed_of_train_m_per_s * time_to_cross_bridge  -- Total distance covered

theorem length_of_bridge :
  distance_covered - length_of_train = 240 :=
by
  sorry

end length_of_bridge_l6_6623


namespace tan_3theta_l6_6974

-- Let θ be an angle such that tan θ = 3.
variable (θ : ℝ)
noncomputable def tan_theta : ℝ := 3

-- Claim: tan(3 * θ) = 9/13
theorem tan_3theta :
  Real.tan (3 * θ) = 9 / 13 :=
by
  sorry

end tan_3theta_l6_6974


namespace line_equation_135_deg_l6_6422

theorem line_equation_135_deg (A : ℝ × ℝ) (theta : ℝ) (l : ℝ → ℝ → Prop) :
  A = (1, -2) →
  theta = 135 →
  (∀ x y, l x y ↔ y = -(x - 1) - 2) →
  ∀ x y, l x y ↔ x + y + 1 = 0 :=
by
  intros hA hTheta hl_form
  sorry

end line_equation_135_deg_l6_6422


namespace find_interest_rate_of_initial_investment_l6_6297

def initial_investment : ℝ := 1400
def additional_investment : ℝ := 700
def total_investment : ℝ := 2100
def additional_interest_rate : ℝ := 0.08
def target_total_income_rate : ℝ := 0.06
def target_total_income : ℝ := target_total_income_rate * total_investment

theorem find_interest_rate_of_initial_investment (r : ℝ) :
  (initial_investment * r + additional_investment * additional_interest_rate = target_total_income) → 
  (r = 0.05) :=
by
  sorry

end find_interest_rate_of_initial_investment_l6_6297


namespace handshake_count_l6_6094

theorem handshake_count : 
  let n := 5  -- number of representatives per company
  let c := 5  -- number of companies
  let total_people := n * c  -- total number of people
  let handshakes_per_person := total_people - n  -- each person shakes hands with 20 others
  (total_people * handshakes_per_person) / 2 = 250 := 
by
  sorry

end handshake_count_l6_6094


namespace problem_part1_problem_part2_l6_6917

variable (a : ℝ)

def quadratic_solution_set_1 := {x : ℝ | x^2 + 2*x + a = 0}
def quadratic_solution_set_2 := {x : ℝ | a*x^2 + 2*x + 2 = 0}

theorem problem_part1 :
  (quadratic_solution_set_1 a = ∅ ∨ quadratic_solution_set_2 a = ∅) ∧ ¬ (quadratic_solution_set_1 a = ∅ ∧ quadratic_solution_set_2 a = ∅) →
  (1/2 < a ∧ a ≤ 1) :=
sorry

theorem problem_part2 :
  quadratic_solution_set_1 a ∪ quadratic_solution_set_2 a ≠ ∅ →
  a ≤ 1 :=
sorry

end problem_part1_problem_part2_l6_6917


namespace sum_of_consecutive_negatives_l6_6306

theorem sum_of_consecutive_negatives (n : ℤ) (h1 : n * (n + 1) = 2720) (h2 : n < 0) : 
  n + (n + 1) = -103 :=
by
  sorry

end sum_of_consecutive_negatives_l6_6306


namespace mixed_groups_count_l6_6661

-- Defining the conditions
def total_children : ℕ := 300
def groups_count : ℕ := 100
def group_size : ℕ := 3
def photographs_per_group : ℕ := group_size
def total_photographs : ℕ := groups_count * photographs_per_group
def boy_boy_photos : ℕ := 100
def girl_girl_photos : ℕ := 56
def mixed_photos : ℕ := total_photographs - boy_boy_photos - girl_girl_photos
def mixed_groups : ℕ := mixed_photos / 2

theorem mixed_groups_count : mixed_groups = 72 := by
  -- skipping the proof
  sorry

end mixed_groups_count_l6_6661


namespace geometric_sequence_a5_eq_8_l6_6549

variable (a : ℕ → ℝ)
variable (n : ℕ)

-- Conditions
axiom pos (n : ℕ) : a n > 0
axiom prod_eq (a3 a7 : ℝ) : a 3 * a 7 = 64

-- Statement to prove
theorem geometric_sequence_a5_eq_8
  (pos : ∀ n, a n > 0)
  (prod_eq : a 3 * a 7 = 64) :
  a 5 = 8 :=
sorry

end geometric_sequence_a5_eq_8_l6_6549


namespace intersection_of_function_and_inverse_l6_6845

theorem intersection_of_function_and_inverse (c k : ℤ) (f : ℤ → ℤ)
  (hf : ∀ x:ℤ, f x = 4 * x + c) 
  (hf_inv : ∀ y:ℤ, (∃ x:ℤ, f x = y) → (∃ x:ℤ, f y = x))
  (h_intersection : ∀ k:ℤ, f 2 = k ∧ f k = 2 ) 
  : k = 2 :=
sorry

end intersection_of_function_and_inverse_l6_6845


namespace train_cross_pole_time_l6_6242

-- Defining the given conditions
def speed_km_hr : ℕ := 54
def length_m : ℕ := 135

-- Conversion of speed from km/hr to m/s
def speed_m_s : ℤ := (54 * 1000) / 3600

-- Statement to be proved
theorem train_cross_pole_time : (length_m : ℤ) / speed_m_s = 9 := by
  sorry

end train_cross_pole_time_l6_6242


namespace company_p_employees_december_l6_6476

theorem company_p_employees_december :
  let january_employees := 434.7826086956522
  let percent_more := 0.15
  let december_employees := january_employees + (percent_more * january_employees)
  december_employees = 500 :=
by
  sorry

end company_p_employees_december_l6_6476


namespace room_perimeter_l6_6413

theorem room_perimeter (b l : ℝ) (h1 : l = 3 * b) (h2 : l * b = 12) : 2 * (l + b) = 16 :=
by sorry

end room_perimeter_l6_6413


namespace point_on_line_has_correct_y_l6_6310

theorem point_on_line_has_correct_y (a : ℝ) : (2 * 3 + a - 7 = 0) → a = 1 :=
by 
  sorry

end point_on_line_has_correct_y_l6_6310


namespace complement_of_A_eq_interval_l6_6083

def U : Set ℝ := Set.univ
def A : Set ℝ := {x | x ≥ 1} ∪ {x | x < 0}
def complement_U_A : Set ℝ := {x | 0 ≤ x ∧ x < 1}

theorem complement_of_A_eq_interval : (U \ A) = complement_U_A := by
  sorry

end complement_of_A_eq_interval_l6_6083


namespace transport_cost_6725_l6_6318

variable (P : ℝ) (T : ℝ)

theorem transport_cost_6725
  (h1 : 0.80 * P = 17500)
  (h2 : 1.10 * P = 24475)
  (h3 : 17500 + T + 250 = 24475) :
  T = 6725 := 
sorry

end transport_cost_6725_l6_6318


namespace train_speed_l6_6896

theorem train_speed (length : ℝ) (time : ℝ) (conversion_factor: ℝ)
  (h_length : length = 100) 
  (h_time : time = 5) 
  (h_conversion : conversion_factor = 3.6) :
  (length / time * conversion_factor) = 72 :=
by
  sorry

end train_speed_l6_6896


namespace tile_ratio_l6_6516

/-- Given the initial configuration and extension method, the ratio of black tiles to white tiles in the new design is 22/27. -/
theorem tile_ratio (initial_black : ℕ) (initial_white : ℕ) (border_black : ℕ) (border_white : ℕ) (total_tiles : ℕ)
  (h1 : initial_black = 10)
  (h2 : initial_white = 15)
  (h3 : border_black = 12)
  (h4 : border_white = 12)
  (h5 : total_tiles = 49) :
  (initial_black + border_black) / (initial_white + border_white) = 22 / 27 := 
by {
  /- 
     Here we would provide the proof steps if needed.
     This is a theorem stating that the ratio of black to white tiles 
     in the new design is 22 / 27 given the initial conditions.
  -/
  sorry 
}

end tile_ratio_l6_6516


namespace max_valid_subset_cardinality_l6_6882

def set_S : Finset ℕ := Finset.range 1998 \ {0}

def is_valid_subset (A : Finset ℕ) : Prop :=
  ∀ (x y : ℕ), x ≠ y → x ∈ A → y ∈ A → (x + y) % 117 ≠ 0

theorem max_valid_subset_cardinality :
  ∃ (A : Finset ℕ), is_valid_subset A ∧ 995 = A.card :=
sorry

end max_valid_subset_cardinality_l6_6882


namespace count_4_tuples_l6_6255

theorem count_4_tuples (p : ℕ) [hp : Fact (Nat.Prime p)] : 
  Nat.card {abcd : ℕ × ℕ × ℕ × ℕ // (0 < abcd.1 ∧ abcd.1 < p) ∧ 
                                     (0 < abcd.2.1 ∧ abcd.2.1 < p) ∧ 
                                     (0 < abcd.2.2.1 ∧ abcd.2.2.1 < p) ∧ 
                                     (0 < abcd.2.2.2 ∧ abcd.2.2.2 < p) ∧ 
                                     ((abcd.1 * abcd.2.2.2 - abcd.2.1 * abcd.2.2.1) % p = 0)} = (p - 1) * (p - 1) * (p - 1) :=
by
  sorry

end count_4_tuples_l6_6255


namespace bridge_length_l6_6326

theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (time_to_cross_bridge : ℝ) 
  (train_speed_m_s : train_speed_kmh * (1000 / 3600) = 15) : 
  train_length = 110 → train_speed_kmh = 54 → time_to_cross_bridge = 16.13204276991174 → 
  ((train_speed_kmh * (1000 / 3600)) * time_to_cross_bridge - train_length = 131.9806415486761) :=
by
  intros h1 h2 h3
  sorry

end bridge_length_l6_6326


namespace milton_books_l6_6706

variable (z b : ℕ)

theorem milton_books (h₁ : z + b = 80) (h₂ : b = 4 * z) : z = 16 :=
by
  sorry

end milton_books_l6_6706


namespace sammy_mistakes_l6_6497

def bryan_score : ℕ := 20
def jen_score : ℕ := bryan_score + 10
def sammy_score : ℕ := jen_score - 2
def total_points : ℕ := 35
def mistakes : ℕ := total_points - sammy_score

theorem sammy_mistakes : mistakes = 7 := by
  sorry

end sammy_mistakes_l6_6497


namespace employee_discount_percentage_l6_6846

theorem employee_discount_percentage (wholesale_cost retail_price employee_price discount_percentage : ℝ) 
  (h1 : wholesale_cost = 200)
  (h2 : retail_price = wholesale_cost * 1.2)
  (h3 : employee_price = 204)
  (h4 : discount_percentage = ((retail_price - employee_price) / retail_price) * 100) :
  discount_percentage = 15 :=
by
  sorry

end employee_discount_percentage_l6_6846


namespace matthew_hotdogs_needed_l6_6666

def total_hotdogs (ella_hotdogs emma_hotdogs : ℕ) (luke_multiplier hunter_multiplier : ℕ) : ℕ :=
  let total_sisters := ella_hotdogs + emma_hotdogs
  let luke := luke_multiplier * total_sisters
  let hunter := hunter_multiplier * total_sisters / 2  -- because 1.5 = 3/2 and hunter_multiplier = 3
  total_sisters + luke + hunter

theorem matthew_hotdogs_needed :
  total_hotdogs 2 2 2 3 = 18 := 
by
  -- This proof is correct given the calculations above
  sorry

end matthew_hotdogs_needed_l6_6666


namespace product_positivity_l6_6954

theorem product_positivity (a b c d e f : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) (he : e ≠ 0) (hf : f ≠ 0)
  (h : ∀ {x y z u v w : ℝ}, x * y * z > 0 → ¬(u * v * w > 0) ∧ ¬(x * v * u > 0) ∧ ¬(y * z * u > 0) ∧ ¬(y * v * x > 0)) :
  b * d * e > 0 ∧ a * c * d < 0 ∧ a * c * e < 0 ∧ b * d * f < 0 ∧ b * e * f < 0 := 
by
  sorry

end product_positivity_l6_6954


namespace math_problem_l6_6294

theorem math_problem :
    3 * 3^4 - (27 ^ 63 / 27 ^ 61) = -486 :=
by
  sorry

end math_problem_l6_6294


namespace leonardo_initial_money_l6_6408

theorem leonardo_initial_money (chocolate_cost : ℝ) (borrowed_amount : ℝ) (needed_amount : ℝ)
  (h_chocolate_cost : chocolate_cost = 5)
  (h_borrowed_amount : borrowed_amount = 0.59)
  (h_needed_amount : needed_amount = 0.41) :
  chocolate_cost + borrowed_amount + needed_amount - (chocolate_cost - borrowed_amount) = 4.41 :=
by
  rw [h_chocolate_cost, h_borrowed_amount, h_needed_amount]
  norm_num
  -- Continue with the proof, eventually obtaining the value 4.41
  sorry

end leonardo_initial_money_l6_6408


namespace inequality_transformation_l6_6293

theorem inequality_transformation (f : ℝ → ℝ) (a b : ℝ) (h1 : ∀ x, f x = 2 * x + 3) (h2 : a > 0) (h3 : b > 0) :
  (∀ x, |f x + 5| < a → |x + 3| < b) ↔ b ≤ a / 2 :=
sorry

end inequality_transformation_l6_6293


namespace original_proposition_false_implies_negation_true_l6_6529

-- Define the original proposition and its negation
def original_proposition (x y : ℝ) : Prop := (x + y > 0) → (x > 0 ∧ y > 0)
def negation (x y : ℝ) : Prop := ¬ original_proposition x y

-- Theorem statement
theorem original_proposition_false_implies_negation_true (x y : ℝ) : ¬ original_proposition x y → negation x y :=
by
  -- Since ¬ original_proposition x y implies the negation is true
  intro h
  exact h

end original_proposition_false_implies_negation_true_l6_6529


namespace students_class_division_l6_6496

theorem students_class_division (n : ℕ) (h1 : n % 15 = 0) (h2 : n % 24 = 0) : n = 120 :=
sorry

end students_class_division_l6_6496


namespace solve_x_squared_eq_sixteen_l6_6431

theorem solve_x_squared_eq_sixteen : ∃ (x1 x2 : ℝ), (x1 = -4 ∧ x2 = 4) ∧ ∀ x : ℝ, x^2 = 16 → (x = x1 ∨ x = x2) :=
by
  sorry

end solve_x_squared_eq_sixteen_l6_6431


namespace max_area_rectangle_l6_6922

theorem max_area_rectangle (l w : ℕ) (h : 2 * l + 2 * w = 40) : l * w ≤ 100 :=
sorry

end max_area_rectangle_l6_6922


namespace strong_2013_l6_6019

theorem strong_2013 :
  ∃ x : ℕ, x > 0 ∧ (x ^ (2013 * x) + 1) % (2 ^ 2013) = 0 :=
sorry

end strong_2013_l6_6019


namespace difference_nickels_is_8q_minus_20_l6_6286

variable (q : ℤ)

-- Define the number of quarters for Alice and Bob
def alice_quarters : ℤ := 7 * q - 3
def bob_quarters : ℤ := 3 * q + 7

-- Define the worth of a quarter in nickels
def quarter_to_nickels (quarters : ℤ) : ℤ := 2 * quarters

-- Define the difference in quarters
def difference_quarters : ℤ := alice_quarters q - bob_quarters q

-- Define the difference in their amount of money in nickels
def difference_nickels (q : ℤ) : ℤ := quarter_to_nickels (difference_quarters q)

theorem difference_nickels_is_8q_minus_20 : difference_nickels q = 8 * q - 20 := by
  sorry

end difference_nickels_is_8q_minus_20_l6_6286


namespace number_of_clips_after_k_steps_l6_6753

theorem number_of_clips_after_k_steps (k : ℕ) : 
  ∃ (c : ℕ), c = 2^(k-1) + 1 :=
by sorry

end number_of_clips_after_k_steps_l6_6753


namespace smallest_of_six_consecutive_even_numbers_l6_6640

theorem smallest_of_six_consecutive_even_numbers (h : ∃ n : ℤ, (n - 4) + (n - 2) + n + (n + 2) + (n + 4) + (n + 6) = 390) : ∃ m : ℤ, m = 60 :=
by
  have ex : ∃ n : ℤ, 6 * n + 6 = 390 := by sorry
  obtain ⟨n, hn⟩ := ex
  use (n - 4)
  sorry

end smallest_of_six_consecutive_even_numbers_l6_6640


namespace A_share_is_9000_l6_6432

noncomputable def A_share_in_gain (x : ℝ) : ℝ :=
  let total_gain := 27000
  let A_investment_time := 12 * x
  let B_investment_time := 6 * 2 * x
  let C_investment_time := 4 * 3 * x
  let total_investment_time := A_investment_time + B_investment_time + C_investment_time
  total_gain * A_investment_time / total_investment_time

theorem A_share_is_9000 (x : ℝ) : A_share_in_gain x = 27000 / 3 :=
by
  sorry

end A_share_is_9000_l6_6432


namespace at_least_one_solves_l6_6871

-- Given probabilities
def pA : ℝ := 0.8
def pB : ℝ := 0.6

-- Probability that at least one solves the problem
def prob_at_least_one_solves : ℝ := 1 - ((1 - pA) * (1 - pB))

-- Statement: Prove that the probability that at least one solves the problem is 0.92
theorem at_least_one_solves : prob_at_least_one_solves = 0.92 :=
by
  -- Proof steps would go here
  sorry

end at_least_one_solves_l6_6871


namespace remaining_cubes_count_l6_6107

-- Define the initial number of cubes
def initial_cubes : ℕ := 64

-- Define the holes in the bottom layer
def holes_in_bottom_layer : ℕ := 6

-- Define the number of cubes removed per hole
def cubes_removed_per_hole : ℕ := 3

-- Define the calculation for missing cubes
def missing_cubes : ℕ := holes_in_bottom_layer * cubes_removed_per_hole

-- Define the calculation for remaining cubes
def remaining_cubes : ℕ := initial_cubes - missing_cubes

-- The theorem to prove
theorem remaining_cubes_count : remaining_cubes = 46 := by
  sorry

end remaining_cubes_count_l6_6107


namespace trajectory_equation_no_such_point_l6_6296

-- Conditions for (I): The ratio of the distances is given
def ratio_condition (P : ℝ × ℝ) : Prop :=
  let M := (1, 0)
  let N := (4, 0)
  2 * Real.sqrt ((P.1 - M.1)^2 + P.2^2) = Real.sqrt ((P.1 - N.1)^2 + P.2^2)

-- Proof of (I): Find the trajectory equation of point P
theorem trajectory_equation : 
  ∀ P : ℝ × ℝ, ratio_condition P → P.1^2 + P.2^2 = 4 :=
by
  sorry

-- Conditions for (II): Given points A, B, C
def points_condition (P : ℝ × ℝ) : Prop :=
  let A := (-2, -2)
  let B := (-2, 6)
  let C := (-4, 2)
  (P.1 + 2)^2 + (P.2 + 2)^2 + 
  (P.1 + 2)^2 + (P.2 - 6)^2 + 
  (P.1 + 4)^2 + (P.2 - 2)^2 = 36

-- Proof of (II): Determine the non-existence of point P
theorem no_such_point (P : ℝ × ℝ) : 
  P.1^2 + P.2^2 = 4 → ¬ points_condition P :=
by
  sorry

end trajectory_equation_no_such_point_l6_6296


namespace freshman_count_630_l6_6070

-- Define the variables and conditions
variables (f o j s : ℕ)
variable (total_students : ℕ)

-- Define the ratios given in the problem
def freshmen_to_sophomore : Prop := f = (5 * o) / 4
def sophomore_to_junior : Prop := j = (8 * o) / 7
def junior_to_senior : Prop := s = (7 * j) / 9

-- Total number of students condition
def total_students_condition : Prop := f + o + j + s = total_students

theorem freshman_count_630
  (h1 : freshmen_to_sophomore f o)
  (h2 : sophomore_to_junior o j)
  (h3 : junior_to_senior j s)
  (h4 : total_students_condition f o j s 2158) :
  f = 630 :=
sorry

end freshman_count_630_l6_6070


namespace total_apples_eq_l6_6568

-- Define the conditions for the problem
def baskets : ℕ := 37
def apples_per_basket : ℕ := 17

-- Define the theorem to prove the total number of apples
theorem total_apples_eq : baskets * apples_per_basket = 629 :=
by
  sorry

end total_apples_eq_l6_6568


namespace inverse_modulo_1000000_l6_6151

def A : ℕ := 123456
def B : ℕ := 769230
def N : ℕ := 1053

theorem inverse_modulo_1000000 : (A * B * N) % 1000000 = 1 := 
  by 
  sorry

end inverse_modulo_1000000_l6_6151


namespace interestDifference_l6_6040

noncomputable def simpleInterest (P R T : ℝ) : ℝ :=
  (P * R * T) / 100

noncomputable def compoundInterest (P R T : ℝ) : ℝ :=
  P * (1 + R / 100)^T - P

theorem interestDifference (P R T : ℝ) (hP : P = 500) (hR : R = 20) (hT : T = 2) :
  compoundInterest P R T - simpleInterest P R T = 120 := by
  sorry

end interestDifference_l6_6040


namespace train_crosses_platform_in_39_seconds_l6_6007

-- Definitions based on the problem's conditions
def train_length : ℕ := 450
def time_to_cross_signal : ℕ := 18
def platform_length : ℕ := 525

-- The speed of the train
def train_speed : ℕ := train_length / time_to_cross_signal

-- The total distance the train has to cover
def total_distance : ℕ := train_length + platform_length

-- The time it takes for the train to cross the platform
def time_to_cross_platform : ℕ := total_distance / train_speed

-- The theorem we need to prove
theorem train_crosses_platform_in_39_seconds :
  time_to_cross_platform = 39 := by
  sorry

end train_crosses_platform_in_39_seconds_l6_6007


namespace range_of_x_l6_6446

noncomputable def f (x : ℝ) : ℝ := x * (2^x - 1 / 2^x)

theorem range_of_x (x : ℝ) (h : f (x - 1) > f x) : x < 1 / 2 :=
by sorry

end range_of_x_l6_6446


namespace remainder_of_S_mod_1000_l6_6271

def digit_contribution (d pos : ℕ) : ℕ := (d * d) * pos

def sum_of_digits_with_no_repeats : ℕ :=
  let thousands := (16 + 25 + 36 + 49 + 64 + 81) * (9 * 8 * 7) * 1000
  let hundreds := (16 + 25 + 36 + 49 + 64 + 81) * (8 * 7 * 6) * 100
  let tens := (0 + 1 + 4 + 9 + 16 + 25 + 36 + 49 + 64 + 81 - (16 + 25 + 36 + 49 + 64 + 81)) * 6 * 5 * 10
  let units := (0 + 1 + 4 + 9 + 16 + 25 + 36 + 49 + 64 + 81 - (16 + 25 + 36 + 49 + 64 + 81)) * 6 * 5 * 1
  thousands + hundreds + tens + units

theorem remainder_of_S_mod_1000 : (sum_of_digits_with_no_repeats % 1000) = 220 :=
  by
  sorry

end remainder_of_S_mod_1000_l6_6271


namespace steven_sixth_quiz_score_l6_6116

theorem steven_sixth_quiz_score :
  ∃ x : ℕ, (75 + 80 + 85 + 90 + 100 + x) / 6 = 95 ∧ x = 140 :=
by
  sorry

end steven_sixth_quiz_score_l6_6116


namespace intersection_of_A_and_B_l6_6610

open Set

noncomputable def A : Set ℝ := { x | (x - 2) / (x + 5) < 0 }
noncomputable def B : Set ℝ := { x | x^2 - 2 * x - 3 ≥ 0 }

theorem intersection_of_A_and_B : A ∩ B = { x : ℝ | -5 < x ∧ x ≤ -1 } :=
sorry

end intersection_of_A_and_B_l6_6610


namespace cos_expression_l6_6135

-- Define the condition for the line l and its relationship
def slope_angle_of_line_l (α : ℝ) : Prop :=
  ∃ l : ℝ, l = 2

-- Given the tangent condition for α
def tan_alpha (α : ℝ) : Prop :=
  Real.tan α = 2

theorem cos_expression (α : ℝ) (h1 : slope_angle_of_line_l α) (h2 : tan_alpha α) :
  Real.cos (2015 * Real.pi / 2 - 2 * α) = -4/5 :=
by sorry

end cos_expression_l6_6135


namespace total_trip_cost_l6_6948

def distance_AC : ℝ := 4000
def distance_AB : ℝ := 4250
def bus_rate : ℝ := 0.10
def plane_rate : ℝ := 0.15
def boarding_fee : ℝ := 150

theorem total_trip_cost :
  let distance_BC := Real.sqrt (distance_AB ^ 2 - distance_AC ^ 2)
  let flight_cost := distance_AB * plane_rate + boarding_fee
  let bus_cost := distance_BC * bus_rate
  flight_cost + bus_cost = 931.15 :=
by
  sorry

end total_trip_cost_l6_6948


namespace chimpanzee_count_l6_6499

def total_chimpanzees (moving_chimps : ℕ) (staying_chimps : ℕ) : ℕ :=
  moving_chimps + staying_chimps

theorem chimpanzee_count : total_chimpanzees 18 27 = 45 :=
by
  sorry

end chimpanzee_count_l6_6499


namespace find_value_l6_6452

theorem find_value (m n : ℤ) (h : 2 * m + n - 2 = 0) : 2 * m + n + 1 = 3 :=
by { sorry }

end find_value_l6_6452


namespace solve_equations_l6_6037

theorem solve_equations (x : ℝ) (h1 : x^2 - 9 = 0) (h2 : (-x)^3 = (-8)^2) : x = 3 ∨ x = -3 ∨ x = -4 :=
by 
  sorry

end solve_equations_l6_6037


namespace diagonal_BD_eq_diagonal_AD_eq_l6_6553

structure Point where
  x : ℝ
  y : ℝ

def A : Point := ⟨-1, 2⟩
def C : Point := ⟨5, 4⟩
def line_AB (p : Point) : Prop := p.x - p.y + 3 = 0

theorem diagonal_BD_eq :
  (∃ M : Point, M = ⟨2, 3⟩) →
  (∃ BD : Point → Prop, (BD = fun p => 3*p.x + p.y - 9 = 0)) :=
by
  sorry

theorem diagonal_AD_eq :
  (∃ M : Point, M = ⟨2, 3⟩) →
  (∃ AD : Point → Prop, (AD = fun p => p.x + 7*p.y - 13 = 0)) :=
by
  sorry

end diagonal_BD_eq_diagonal_AD_eq_l6_6553


namespace Solomon_collected_66_l6_6319

-- Definitions
variables (J S L : ℕ) -- J for Juwan, S for Solomon, L for Levi

-- Conditions
axiom C1 : S = 3 * J
axiom C2 : L = J / 2
axiom C3 : J + S + L = 99

-- Theorem to prove
theorem Solomon_collected_66 : S = 66 :=
by
  sorry

end Solomon_collected_66_l6_6319


namespace quadratic_inequality_solution_l6_6090

theorem quadratic_inequality_solution
  (x : ℝ)
  (h : x^2 - 5 * x + 6 < 0) :
  2 < x ∧ x < 3 :=
by sorry

end quadratic_inequality_solution_l6_6090


namespace bisection_approximation_interval_l6_6861

noncomputable def bisection_accuracy (a b : ℝ) (n : ℕ) : ℝ := (b - a) / 2^n

theorem bisection_approximation_interval 
  (a b : ℝ) (n : ℕ) (accuracy : ℝ) (h1 : a = 2) (h2 : b = 3) (h3 : accuracy = 0.01) 
  (h4 : 2^n ≥ 100) : bisection_accuracy a b n ≤ accuracy :=
sorry

end bisection_approximation_interval_l6_6861


namespace num_two_wheelers_wheels_eq_two_l6_6697

variable (num_two_wheelers num_four_wheelers total_wheels : ℕ)

def total_wheels_eq : Prop :=
  2 * num_two_wheelers + 4 * num_four_wheelers = total_wheels

theorem num_two_wheelers_wheels_eq_two (h1 : num_four_wheelers = 13)
                                        (h2 : total_wheels = 54)
                                        (h_total_eq : total_wheels_eq num_two_wheelers num_four_wheelers total_wheels) :
  2 * num_two_wheelers = 2 :=
by
  unfold total_wheels_eq at h_total_eq
  sorry

end num_two_wheelers_wheels_eq_two_l6_6697


namespace moles_of_naoh_combined_number_of_moles_of_naoh_combined_l6_6353

-- Define the reaction equation and given conditions
def reaction_equation := "2 NaOH + Cl₂ → NaClO + NaCl + H₂O"

-- Given conditions
def moles_chlorine : ℕ := 2
def moles_water_produced : ℕ := 2
def moles_naoh_needed_for_one_mole_water : ℕ := 2

-- Stoichiometric relationship from the reaction equation
def moles_naoh_per_mole_water : ℕ := 2

-- Theorem to prove the number of moles of NaOH combined
theorem moles_of_naoh_combined (moles_water_produced : ℕ)
  (moles_naoh_per_mole_water : ℕ) : ℕ :=
  moles_water_produced * moles_naoh_per_mole_water

-- Statement of the theorem
theorem number_of_moles_of_naoh_combined : moles_of_naoh_combined 2 2 = 4 :=
by sorry

end moles_of_naoh_combined_number_of_moles_of_naoh_combined_l6_6353


namespace pigeons_on_branches_and_under_tree_l6_6870

theorem pigeons_on_branches_and_under_tree (x y : ℕ) 
  (h1 : y - 1 = (x + 1) / 2)
  (h2 : x - 1 = y + 1) : x = 7 ∧ y = 5 :=
by
  sorry

end pigeons_on_branches_and_under_tree_l6_6870


namespace carnations_in_last_three_bouquets_l6_6936

/--
Trevor buys six bouquets of carnations.
In the first bouquet, there are 9.5 carnations.
In the second bouquet, there are 14.25 carnations.
In the third bouquet, there are 18.75 carnations.
The average number of carnations in all six bouquets is 16.
Prove that the total number of carnations in the fourth, fifth, and sixth bouquets combined is 53.5.
-/
theorem carnations_in_last_three_bouquets:
  let bouquet1 := 9.5
  let bouquet2 := 14.25
  let bouquet3 := 18.75
  let total_bouquets := 6
  let average_per_bouquet := 16
  let total_carnations := average_per_bouquet * total_bouquets
  let remaining_carnations := total_carnations - (bouquet1 + bouquet2 + bouquet3)
  remaining_carnations = 53.5 :=
by
  sorry

end carnations_in_last_three_bouquets_l6_6936


namespace printer_paper_last_days_l6_6977

def packs : Nat := 2
def sheets_per_pack : Nat := 240
def prints_per_day : Nat := 80
def total_sheets : Nat := packs * sheets_per_pack
def number_of_days : Nat := total_sheets / prints_per_day

theorem printer_paper_last_days :
  number_of_days = 6 :=
by
  sorry

end printer_paper_last_days_l6_6977


namespace thirteen_y_minus_x_l6_6095

theorem thirteen_y_minus_x (x y : ℤ) (hx1 : x = 11 * y + 4) (hx2 : 2 * x = 8 * (3 * y) + 3) : 13 * y - x = 1 :=
by
  sorry

end thirteen_y_minus_x_l6_6095


namespace range_of_angle_A_l6_6010

theorem range_of_angle_A (a b : ℝ) (A : ℝ) (h₁ : a = 2) (h₂ : b = 2 * Real.sqrt 2) 
  (h_triangle : 0 < A ∧ A ≤ Real.pi / 4) :
  (0 < A ∧ A ≤ Real.pi / 4) :=
by
  sorry

end range_of_angle_A_l6_6010


namespace percentage_of_students_in_60_to_69_range_is_20_l6_6569

theorem percentage_of_students_in_60_to_69_range_is_20 :
  let scores := [4, 8, 6, 5, 2]
  let total_students := scores.sum
  let students_in_60_to_69 := 5
  (students_in_60_to_69 * 100 / total_students) = 20 := by
  sorry

end percentage_of_students_in_60_to_69_range_is_20_l6_6569


namespace reciprocal_of_subtraction_l6_6534

-- Defining the conditions
def x : ℚ := 1 / 9
def y : ℚ := 2 / 3

-- Defining the main theorem statement
theorem reciprocal_of_subtraction : (1 / (y - x)) = 9 / 5 :=
by
  sorry

end reciprocal_of_subtraction_l6_6534


namespace simple_interest_calculation_l6_6512

variable (P : ℝ) (R : ℝ) (T : ℝ)

def simple_interest (P R T : ℝ) : ℝ := P * R * T

theorem simple_interest_calculation (hP : P = 10000) (hR : R = 0.09) (hT : T = 1) :
    simple_interest P R T = 900 := by
  rw [hP, hR, hT]
  sorry

end simple_interest_calculation_l6_6512


namespace parabola_coefficient_c_l6_6567

def parabola (b c x : ℝ) : ℝ := x^2 + b * x + c

theorem parabola_coefficient_c (b c : ℝ) (h1 : parabola b c 1 = -1) (h2 : parabola b c 3 = 9) : 
  c = -3 := 
by
  sorry

end parabola_coefficient_c_l6_6567


namespace club_planning_committee_l6_6069

theorem club_planning_committee : Nat.choose 20 3 = 1140 := 
by sorry

end club_planning_committee_l6_6069


namespace simplify_and_evaluate_expression_l6_6045

noncomputable def given_expression (a : ℝ) : ℝ :=
  (a - 1 - (2 * a - 1) / (a + 1)) / ((a^2 - 4 * a + 4) / (a + 1))

theorem simplify_and_evaluate_expression (a : ℝ) (h : a = 2 + Real.sqrt 3) :
  given_expression a = (2 * Real.sqrt 3 + 3) / 3 :=
by
  sorry

end simplify_and_evaluate_expression_l6_6045


namespace simplified_expression_correct_l6_6823

def simplify_expression (x : ℝ) : ℝ :=
  4 * (x ^ 2 - 5 * x) - 5 * (2 * x ^ 2 + 3 * x)

theorem simplified_expression_correct (x : ℝ) :
  simplify_expression x = -6 * x ^ 2 - 35 * x :=
by
  sorry

end simplified_expression_correct_l6_6823


namespace brick_width_l6_6795

theorem brick_width (L W : ℕ) (l : ℕ) (b : ℕ) (n : ℕ) (A B : ℕ) 
    (courtyard_area_eq : A = L * W * 10000)
    (brick_area_eq : B = l * b)
    (total_bricks_eq : A = n * B)
    (courtyard_dims : L = 30 ∧ W = 16)
    (brick_len : l = 20)
    (num_bricks : n = 24000) :
    b = 10 := by
  sorry

end brick_width_l6_6795


namespace solve_ode_l6_6415

noncomputable def x (t : ℝ) : ℝ :=
  -((1 : ℝ) / 18) * Real.exp (-t) +
  (25 / 54) * Real.exp (5 * t) -
  (11 / 27) * Real.exp (-4 * t)

theorem solve_ode :
  ∀ t : ℝ, 
    (deriv^[2] x t) - (deriv x t) - 20 * x t = Real.exp (-t) ∧
    x 0 = 0 ∧
    (deriv x 0) = 4 :=
by
  sorry

end solve_ode_l6_6415


namespace parabola_trajectory_l6_6981

theorem parabola_trajectory :
  ∀ P : ℝ × ℝ, (dist P (0, -1) + 1 = dist P (0, 3)) ↔ (P.1 ^ 2 = -8 * P.2) := by
  sorry

end parabola_trajectory_l6_6981


namespace smallest_t_for_given_roots_l6_6591

-- Define the polynomial with integer coefficients and specific roots
def poly (x : ℝ) : ℝ := (x + 3) * (x - 4) * (x - 6) * (2 * x - 1)

-- Define the main theorem statement
theorem smallest_t_for_given_roots :
  ∃ (t : ℤ), 0 < t ∧ t = 72 := by
  -- polynomial expansion skipped, proof will come here
  sorry

end smallest_t_for_given_roots_l6_6591


namespace two_legged_birds_count_l6_6654

-- Definitions and conditions
variables {x y z : ℕ}
variables (heads_eq : x + y + z = 200) (legs_eq : 2 * x + 3 * y + 4 * z = 558)

-- The statement to prove
theorem two_legged_birds_count : x = 94 :=
sorry

end two_legged_birds_count_l6_6654


namespace emus_count_l6_6787

theorem emus_count (E : ℕ) (heads : ℕ) (legs : ℕ) 
  (h_heads : ∀ e : ℕ, heads = e) 
  (h_legs : ∀ e : ℕ, legs = 2 * e)
  (h_total : heads + legs = 60) : 
  E = 20 :=
by sorry

end emus_count_l6_6787


namespace roots_of_quadratic_l6_6734

theorem roots_of_quadratic (α β : ℝ) (h1 : α^2 - 4*α - 5 = 0) (h2 : β^2 - 4*β - 5 = 0) :
  3*α^4 + 10*β^3 = 2593 := 
by
  sorry

end roots_of_quadratic_l6_6734


namespace sufficient_not_necessary_condition_for_positive_quadratic_l6_6463

variables {a b c : ℝ}

theorem sufficient_not_necessary_condition_for_positive_quadratic 
  (ha : a > 0)
  (hb : b^2 - 4 * a * c < 0) :
  (∀ x : ℝ, a * x ^ 2 + b * x + c > 0) 
  ∧ ¬ (∀ x : ℝ, ∃ a b c : ℝ, a > 0 ∧ b^2 - 4 * a * c ≥ 0 ∧ (a * x ^ 2 + b * x + c > 0)) :=
by
  sorry

end sufficient_not_necessary_condition_for_positive_quadratic_l6_6463


namespace evaluate_expression_l6_6359

theorem evaluate_expression : 21^2 - 19^2 + 17^2 - 15^2 + 13^2 - 11^2 + 9^2 - 7^2 + 5^2 - 3^2 + 1^2 = 221 :=
by
  sorry

end evaluate_expression_l6_6359


namespace value_of_x_after_z_doubled_l6_6633

theorem value_of_x_after_z_doubled (x y z : ℕ) (hz : z = 48) (hz_d : z_d = 2 * z) (hy : y = z / 4) (hx : x = y / 3) :
  x = 8 := by
  -- Proof goes here (skipped as instructed)
  sorry

end value_of_x_after_z_doubled_l6_6633


namespace total_peaches_l6_6570

-- Definitions of conditions
def initial_peaches : ℕ := 13
def picked_peaches : ℕ := 55

-- Proof problem statement
theorem total_peaches : initial_peaches + picked_peaches = 68 :=
by
  -- Including sorry to skip the actual proof
  sorry

end total_peaches_l6_6570


namespace arithmetic_sequence_general_term_geometric_sequence_inequality_l6_6888

-- Sequence {a_n} and its sum S_n
def S (a : ℕ → ℤ) (n : ℕ) : ℤ := (Finset.range n).sum a

-- Sequence {b_n}
def b (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  2 * (S a (n + 1) - S a n) * (S a n) - n * (S a (n + 1) + S a n)

-- Arithmetic sequence and related conditions
theorem arithmetic_sequence_general_term (a : ℕ → ℤ) (d : ℤ)
  (h1 : ∀ n, a n = a 0 + n * d) (h2 : ∀ n, b a n = 0) :
  (∀ n, a n = 0) ∨ (∀ n, a n = n) :=
sorry

-- Conditions for sequences and finding the set of positive integers n
theorem geometric_sequence_inequality (a : ℕ → ℤ)
  (h1 : a 1 = 1) (h2 : a 2 = 3)
  (h3 : ∀ n, a (2 * n - 1) = 2^(n-1))
  (h4 : ∀ n, a (2 * n) = 3 * 2^(n-1)) :
  {n : ℕ | b a (2 * n) < b a (2 * n - 1)} = {1, 2, 3, 4, 5, 6} :=
sorry

end arithmetic_sequence_general_term_geometric_sequence_inequality_l6_6888


namespace correct_points_per_answer_l6_6188

noncomputable def points_per_correct_answer (total_questions : ℕ) 
  (answered_correctly : ℕ) (final_score : ℝ) (penalty_per_incorrect : ℝ)
  (total_incorrect : ℕ := total_questions - answered_correctly) 
  (points_subtracted : ℝ := total_incorrect * penalty_per_incorrect) 
  (earned_points : ℝ := final_score + points_subtracted) : ℝ := 
    earned_points / answered_correctly

theorem correct_points_per_answer :
  points_per_correct_answer 120 104 100 0.25 = 1 := 
by 
  sorry

end correct_points_per_answer_l6_6188


namespace largest_integer_less_than_80_with_remainder_3_when_divided_by_5_l6_6228

theorem largest_integer_less_than_80_with_remainder_3_when_divided_by_5 : 
  ∃ x : ℤ, x < 80 ∧ x % 5 = 3 ∧ (∀ y : ℤ, y < 80 ∧ y % 5 = 3 → y ≤ x) :=
sorry

end largest_integer_less_than_80_with_remainder_3_when_divided_by_5_l6_6228


namespace alissa_total_amount_spent_correct_l6_6741
-- Import necessary Lean library

-- Define the costs of individual items
def football_cost : ℝ := 8.25
def marbles_cost : ℝ := 6.59
def puzzle_cost : ℝ := 12.10
def action_figure_cost : ℝ := 15.29
def board_game_cost : ℝ := 23.47

-- Define the discount rate and the sales tax rate
def discount_rate : ℝ := 0.10
def sales_tax_rate : ℝ := 0.06

-- Define the total cost before discount
def total_cost_before_discount : ℝ :=
  football_cost + marbles_cost + puzzle_cost + action_figure_cost + board_game_cost

-- Define the discount amount
def discount : ℝ := total_cost_before_discount * discount_rate

-- Define the total cost after discount
def total_cost_after_discount : ℝ := total_cost_before_discount - discount

-- Define the sales tax amount
def sales_tax : ℝ := total_cost_after_discount * sales_tax_rate

-- Define the total amount spent
def total_amount_spent : ℝ := total_cost_after_discount + sales_tax

-- Prove that the total amount spent is $62.68
theorem alissa_total_amount_spent_correct : total_amount_spent = 62.68 := 
  by 
    sorry

end alissa_total_amount_spent_correct_l6_6741


namespace min_abs_sum_half_l6_6434

theorem min_abs_sum_half :
  ∀ (f g : ℝ → ℝ),
  (∀ x, f x = Real.sin (x + Real.pi / 3)) →
  (∀ x, g x = Real.sin (2 * x + Real.pi / 3)) →
  (∀ x1 x2 : ℝ, g x1 * g x2 = -1 ∧ x1 ≠ x2 → abs ((x1 + x2) / 2) = Real.pi / 6) := by
-- Definitions and conditions are set, now we can state the theorem.
  sorry

end min_abs_sum_half_l6_6434


namespace solve_problem_l6_6643

noncomputable def problem_statement (x : ℝ) : Prop :=
  1 + Real.sin x - Real.cos (5 * x) - Real.sin (7 * x) = 2 * Real.cos (3 * x / 2) ^ 2

theorem solve_problem (x : ℝ) :
  problem_statement x ↔
  (∃ k : ℤ, x = (Real.pi / 8) * (2 * k + 1)) ∨ 
  (∃ n : ℤ, x = (Real.pi / 4) * (4 * n - 1)) :=
by
  sorry

end solve_problem_l6_6643


namespace number_of_people_got_on_train_l6_6272

theorem number_of_people_got_on_train (initial_people : ℕ) (people_got_off : ℕ) (final_people : ℕ) (x : ℕ) 
  (h_initial : initial_people = 78) 
  (h_got_off : people_got_off = 27) 
  (h_final : final_people = 63) 
  (h_eq : final_people = initial_people - people_got_off + x) : x = 12 :=
by 
  sorry

end number_of_people_got_on_train_l6_6272


namespace solution_set_for_inequality_l6_6722

theorem solution_set_for_inequality :
  {x : ℝ | -x^2 + 2 * x + 3 ≥ 0} = {x : ℝ | -1 ≤ x ∧ x ≤ 3} :=
sorry

end solution_set_for_inequality_l6_6722


namespace problem_solution_l6_6114

theorem problem_solution :
  (2^2 + 4^2 + 6^2) / (1^2 + 3^2 + 5^2) - (1^2 + 3^2 + 5^2) / (2^2 + 4^2 + 6^2) = 1911 / 1960 :=
by sorry

end problem_solution_l6_6114


namespace max_small_boxes_l6_6121

-- Define the dimensions of the larger box in meters
def large_box_length : ℝ := 6
def large_box_width : ℝ := 5
def large_box_height : ℝ := 4

-- Define the dimensions of the smaller box in meters
def small_box_length : ℝ := 0.60
def small_box_width : ℝ := 0.50
def small_box_height : ℝ := 0.40

-- Calculate the volume of the larger box
def large_box_volume : ℝ := large_box_length * large_box_width * large_box_height

-- Calculate the volume of the smaller box
def small_box_volume : ℝ := small_box_length * small_box_width * small_box_height

-- State the theorem to prove the maximum number of smaller boxes that can fit in the larger box
theorem max_small_boxes : large_box_volume / small_box_volume = 1000 :=
by
  sorry

end max_small_boxes_l6_6121


namespace product_of_base8_digits_of_8654_l6_6767

theorem product_of_base8_digits_of_8654 : 
  let base10 := 8654
  let base8_rep := [2, 0, 7, 1, 6] -- Representing 8654(10) to 20716(8)
  (base8_rep.prod = 0) :=
  sorry

end product_of_base8_digits_of_8654_l6_6767


namespace smallest_constant_obtuse_triangle_l6_6433

theorem smallest_constant_obtuse_triangle (a b c : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c) :
  (a^2 > b^2 + c^2) → (b^2 + c^2) / (a^2) ≥ 1 / 2 :=
by 
  sorry

end smallest_constant_obtuse_triangle_l6_6433


namespace katie_flour_l6_6058

theorem katie_flour (x : ℕ) (h1 : x + (x + 2) = 8) : x = 3 := 
by
  sorry

end katie_flour_l6_6058


namespace abs_neg_2023_l6_6144

theorem abs_neg_2023 : |(-2023)| = 2023 := by
  sorry

end abs_neg_2023_l6_6144


namespace bella_age_l6_6720

theorem bella_age (B : ℕ) 
  (h1 : (B + 9) + B + B / 2 = 27) 
  : B = 6 :=
by sorry

end bella_age_l6_6720


namespace most_representative_sample_l6_6328

/-- Options for the student sampling methods -/
inductive SamplingMethod
| NinthGradeStudents : SamplingMethod
| FemaleStudents : SamplingMethod
| BasketballStudents : SamplingMethod
| StudentsWithIDEnding5 : SamplingMethod

/-- Definition of representativeness for each SamplingMethod -/
def isMostRepresentative (method : SamplingMethod) : Prop :=
  method = SamplingMethod.StudentsWithIDEnding5

/-- Prove that the students with ID ending in 5 is the most representative sampling method -/
theorem most_representative_sample : isMostRepresentative SamplingMethod.StudentsWithIDEnding5 :=
  by
  sorry

end most_representative_sample_l6_6328


namespace problem_statement_l6_6562

theorem problem_statement : 
  10 - 1.05 / (5.2 * 14.6 - (9.2 * 5.2 + 5.4 * 3.7 - 4.6 * 1.5)) = 9.93 :=
by sorry

end problem_statement_l6_6562


namespace slope_of_parallel_line_l6_6710

/-- A line is described by the equation 3x - 6y = 12. The slope of a line 
    parallel to this line is 1/2. -/
theorem slope_of_parallel_line (x y : ℝ) (h : 3 * x - 6 * y = 12) : 
  ∃ m : ℝ, m = 1/2 := by
  sorry

end slope_of_parallel_line_l6_6710


namespace cost_for_23_days_l6_6887

structure HostelStay where
  charge_first_week : ℝ
  charge_additional_week : ℝ

def cost_of_stay (days : ℕ) (hostel : HostelStay) : ℝ :=
  let first_week_days := min days 7
  let remaining_days := days - first_week_days
  let additional_full_weeks := remaining_days / 7 
  let additional_days := remaining_days % 7
  (first_week_days * hostel.charge_first_week) + 
  (additional_full_weeks * 7 * hostel.charge_additional_week) + 
  (additional_days * hostel.charge_additional_week)

theorem cost_for_23_days :
  cost_of_stay 23 { charge_first_week := 18.00, charge_additional_week := 11.00 } = 302.00 :=
by
  sorry

end cost_for_23_days_l6_6887


namespace find_a_l6_6913

-- Definitions for conditions
def line_equation (a : ℝ) (x y : ℝ) := a * x - y - 1 = 0
def angle_of_inclination (θ : ℝ) := θ = Real.pi / 3

-- The main theorem statement
theorem find_a (a : ℝ) (θ : ℝ) (h1 : angle_of_inclination θ) (h2 : a = Real.tan θ) : a = Real.sqrt 3 :=
 by
   -- skipping the proof
   sorry

end find_a_l6_6913


namespace common_root_values_l6_6493

def has_common_root (p x : ℝ) : Prop :=
  (x^2 - (p+1)*x + (p+1) = 0) ∧ (2*x^2 + (p-2)*x - p - 7 = 0)

theorem common_root_values :
  (has_common_root 3 2) ∧ (has_common_root (-3/2) (-1)) :=
by {
  sorry
}

end common_root_values_l6_6493


namespace store_total_profit_l6_6971

theorem store_total_profit
  (purchase_price : ℕ)
  (selling_price_total : ℕ)
  (max_selling_price : ℕ)
  (profit : ℕ)
  (N : ℕ)
  (selling_price_per_card : ℕ)
  (h1 : purchase_price = 21)
  (h2 : selling_price_total = 1457)
  (h3 : max_selling_price = 2 * purchase_price)
  (h4 : selling_price_per_card * N = selling_price_total)
  (h5 : selling_price_per_card ≤ max_selling_price)
  (h_profit : profit = (selling_price_per_card - purchase_price) * N)
  : profit = 470 :=
sorry

end store_total_profit_l6_6971


namespace factorize_quadratic_l6_6301

theorem factorize_quadratic (x : ℝ) : x^2 - 6 * x + 9 = (x - 3)^2 :=
by {
  sorry  -- Proof goes here
}

end factorize_quadratic_l6_6301


namespace find_m_l6_6758

theorem find_m (m : ℝ) (a b : ℝ) (r s : ℝ) (S1 S2 : ℝ)
  (h1 : a = 10)
  (h2 : b = 10)
  (h3 : 10 * r = 5)
  (h4 : S1 = 20)
  (h5 : 10 * s = 5 + m)
  (h6 : S2 = 100 / (5 - m))
  (h7 : S2 = 3 * S1) :
  m = 10 / 3 := by
  sorry

end find_m_l6_6758


namespace sequence_solution_l6_6017

theorem sequence_solution (a : ℕ → ℝ) (h1 : a 1 = 2) (h2 : a 2 = 1)
  (h_rec : ∀ n ≥ 2, 2 / a n = 1 / a (n + 1) + 1 / a (n - 1)) :
  ∀ n, a n = 2 / n :=
by
  sorry

end sequence_solution_l6_6017


namespace bottles_remaining_l6_6335

-- Define the initial number of bottles.
def initial_bottles : ℝ := 45.0

-- Define the number of bottles Maria drank.
def maria_drinks : ℝ := 14.0

-- Define the number of bottles Maria's sister drank.
def sister_drinks : ℝ := 8.0

-- The value that needs to be proved.
def bottles_left : ℝ := initial_bottles - maria_drinks - sister_drinks

-- The theorem statement.
theorem bottles_remaining :
  bottles_left = 23.0 :=
by
  sorry

end bottles_remaining_l6_6335


namespace symmetric_points_sum_l6_6890

theorem symmetric_points_sum (a b : ℝ) (h1 : B = (-A)) (h2 : A = (1, a)) (h3 : B = (b, 2)) : a + b = -3 := by
  sorry

end symmetric_points_sum_l6_6890


namespace smallest_n_l6_6287

theorem smallest_n (n : ℕ) : 17 * n ≡ 136 [MOD 5] → n = 3 := 
by sorry

end smallest_n_l6_6287


namespace each_friend_pays_l6_6492

def hamburgers_cost : ℝ := 5 * 3
def fries_cost : ℝ := 4 * 1.20
def soda_cost : ℝ := 5 * 0.50
def spaghetti_cost : ℝ := 1 * 2.70
def total_cost : ℝ := hamburgers_cost + fries_cost + soda_cost + spaghetti_cost
def num_friends : ℝ := 5

theorem each_friend_pays :
  total_cost / num_friends = 5 :=
by
  sorry

end each_friend_pays_l6_6492


namespace evaluate_m_l6_6858

theorem evaluate_m (m : ℕ) : 2 ^ m = (64 : ℝ) ^ (1 / 3) → m = 2 :=
by
  sorry

end evaluate_m_l6_6858


namespace vector_orthogonality_solution_l6_6672

theorem vector_orthogonality_solution :
  let a := (3, -2)
  let b := (x, 1)
  (a.1 * b.1 + a.2 * b.2 = 0) →
  x = 2 / 3 :=
by
  intro h
  sorry

end vector_orthogonality_solution_l6_6672


namespace partI_solution_set_partII_range_of_a_l6_6400

namespace MathProof

-- Define the function f(x)
def f (x a : ℝ) : ℝ := abs (x - a) - abs (x + 3)

-- Part (Ⅰ) Proof Problem
theorem partI_solution_set (x : ℝ) : 
  f x (-1) ≤ 1 ↔ -5/2 ≤ x :=
sorry

-- Part (Ⅱ) Proof Problem
theorem partII_range_of_a (a : ℝ) : 
  (∀ x, 0 ≤ x ∧ x ≤ 3 → f x a ≤ 4) ↔ -7 ≤ a ∧ a ≤ 7 :=
sorry

end MathProof

end partI_solution_set_partII_range_of_a_l6_6400


namespace anna_has_2_fewer_toys_than_amanda_l6_6806

-- Define the variables for the number of toys each person has
variables (A B : ℕ)

-- Define the conditions
def conditions (M : ℕ) : Prop :=
  M = 20 ∧ A = 3 * M ∧ A + M + B = 142

-- The theorem to prove
theorem anna_has_2_fewer_toys_than_amanda (M : ℕ) (h : conditions A B M) : B - A = 2 :=
sorry

end anna_has_2_fewer_toys_than_amanda_l6_6806


namespace quadratic_roots_solution_l6_6289

noncomputable def quadratic_roots_differ_by_2 (p q : ℝ) (hq_pos : 0 < q) (hp_pos : 0 < p) : Prop :=
  let root1 := (-p + Real.sqrt (p^2 - 4*q)) / 2
  let root2 := (-p - Real.sqrt (p^2 - 4*q)) / 2
  abs (root1 - root2) = 2

theorem quadratic_roots_solution (p q : ℝ) (hq_pos : 0 < q) (hp_pos : 0 < p) :
  quadratic_roots_differ_by_2 p q hq_pos hp_pos →
  p = 2 * Real.sqrt (q + 1) :=
sorry

end quadratic_roots_solution_l6_6289


namespace range_of_m_l6_6111

-- Define the propositions
def p (m : ℝ) : Prop := m ≤ 2
def q (m : ℝ) : Prop := 0 < m ∧ m < 1

-- Problem statement to derive m's range
theorem range_of_m (m : ℝ) (h1: ¬ (p m ∧ q m)) (h2: p m ∨ q m) : m ≤ 0 ∨ (1 ≤ m ∧ m ≤ 2) := 
sorry

end range_of_m_l6_6111


namespace dice_probability_abs_diff_2_l6_6693

theorem dice_probability_abs_diff_2 :
  let total_outcomes := 36
  let favorable_outcomes := 8
  let probability := favorable_outcomes / total_outcomes
  probability = 2 / 9 :=
by
  sorry

end dice_probability_abs_diff_2_l6_6693


namespace total_students_in_class_l6_6842

-- Definitions based on the conditions
def volleyball_participants : Nat := 22
def basketball_participants : Nat := 26
def both_participants : Nat := 4

-- The theorem statement
theorem total_students_in_class : volleyball_participants + basketball_participants - both_participants = 44 :=
by
  -- Sorry to skip the proof
  sorry

end total_students_in_class_l6_6842


namespace part1_part2_l6_6210

-- Part (1) Lean 4 statement
theorem part1 {x : ℕ} (h : 0 < x ∧ 4 * (x + 2) < 18 + 2 * x) : x = 1 ∨ x = 2 ∨ x = 3 ∨ x = 4 :=
sorry

-- Part (2) Lean 4 statement
theorem part2 (x : ℝ) (h1 : 5 * x + 2 ≥ 4 * x + 1) (h2 : (x + 1) / 4 > (x - 3) / 2 + 1) : -1 ≤ x ∧ x < 3 :=
sorry

end part1_part2_l6_6210


namespace system_of_equations_solution_l6_6928

theorem system_of_equations_solution :
  ∀ (x : Fin 100 → ℝ), 
  (x 0 + x 1 + x 2 = 0) ∧ 
  (x 1 + x 2 + x 3 = 0) ∧ 
  -- Continue for all other equations up to
  (x 98 + x 99 + x 0 = 0) ∧ 
  (x 99 + x 0 + x 1 = 0)
  → ∀ (i : Fin 100), x i = 0 :=
by
  intros x h
  -- We can insert the detailed solving steps here
  sorry

end system_of_equations_solution_l6_6928


namespace max_electronic_thermometers_l6_6766

-- Definitions
def budget : ℕ := 300
def price_mercury : ℕ := 3
def price_electronic : ℕ := 10
def total_students : ℕ := 53

-- The theorem statement
theorem max_electronic_thermometers : 
  (∃ x : ℕ, x ≤ total_students ∧ 10 * x + 3 * (total_students - x) ≤ budget ∧ 
            ∀ y : ℕ, y ≤ total_students ∧ 10 * y + 3 * (total_students - y) ≤ budget → y ≤ x) :=
sorry

end max_electronic_thermometers_l6_6766


namespace limit_of_sequence_z_l6_6281

open Nat Real

noncomputable def sequence_z (n : ℕ) : ℝ :=
  -3 + (-1)^n / (n^2 : ℝ)

theorem limit_of_sequence_z :
  ∀ ε > 0, ∃ N : ℕ, ∀ n > N, abs (sequence_z n + 3) < ε :=
by
  sorry

end limit_of_sequence_z_l6_6281


namespace log_defined_for_powers_of_a_if_integer_exponents_log_undefined_if_only_positive_indices_l6_6687

variable (a : ℝ) (b : ℝ)

-- Conditions
axiom base_pos (h : a > 0) : a ≠ 1
axiom integer_exponents_only (h : ∃ n : ℤ, b = a^n) : True
axiom positive_indices_only (h : ∃ n : ℕ, b = a^n) : 0 < b ∧ b < 1 → False

-- Theorem: If we only knew integer exponents, the logarithm of any number b in base a is defined for powers of a.
theorem log_defined_for_powers_of_a_if_integer_exponents (h : ∃ n : ℤ, b = a^n) : True :=
by sorry

-- Theorem: If we only knew positive exponents, the logarithm of any number b in base a is undefined for all 0 < b < 1
theorem log_undefined_if_only_positive_indices : (∃ n : ℕ, b = a^n) → (0 < b ∧ b < 1 → False) :=
by sorry

end log_defined_for_powers_of_a_if_integer_exponents_log_undefined_if_only_positive_indices_l6_6687


namespace square_side_length_eq_8_over_pi_l6_6942

noncomputable def side_length_square : ℝ := 8 / Real.pi

theorem square_side_length_eq_8_over_pi :
  ∀ (s : ℝ),
  (4 * s = (Real.pi * (s / Real.sqrt 2) ^ 2) / 2) →
  s = side_length_square :=
by
  intro s h
  sorry

end square_side_length_eq_8_over_pi_l6_6942


namespace unique_solution_zmod_11_l6_6491

theorem unique_solution_zmod_11 : 
  ∀ (n : ℕ), 
  (2 ≤ n → 
  (∀ x : ZMod n, (x^2 - 3 * x + 5 = 0) → (∃! x : ZMod n, x^2 - (3 : ZMod n) * x + (5 : ZMod n) = 0)) → 
  n = 11) := 
by
  sorry

end unique_solution_zmod_11_l6_6491


namespace chocolate_bars_per_box_l6_6561

-- Definitions for the given conditions
def total_chocolate_bars : ℕ := 849
def total_boxes : ℕ := 170

-- The statement to prove
theorem chocolate_bars_per_box : total_chocolate_bars / total_boxes = 5 :=
by 
  -- Proof is omitted here
  sorry

end chocolate_bars_per_box_l6_6561


namespace perimeter_of_sector_l6_6637

theorem perimeter_of_sector (r : ℝ) (area : ℝ) (perimeter : ℝ) 
  (hr : r = 1) (ha : area = π / 3) : perimeter = (2 * π / 3) + 2 :=
by
  -- You can start the proof here
  sorry

end perimeter_of_sector_l6_6637


namespace problem_l6_6393

theorem problem (a b : ℝ) (h1 : abs a = 4) (h2 : b^2 = 9) (h3 : a / b > 0) : a - b = 1 ∨ a - b = -1 := 
sorry

end problem_l6_6393


namespace magic_square_l6_6386

-- Define a 3x3 grid with positions a, b, c and unknowns x, y, z, t, u, v
variables (a b c x y z t u v : ℝ)

-- State the theorem: there exists values for x, y, z, t, u, v
-- such that the sums in each row, column, and both diagonals are the same
theorem magic_square (h1: x = (b + 3*c - 2*a) / 2)
  (h2: y = a + b - c)
  (h3: z = (b + c) / 2)
  (h4: t = 2*c - a)
  (h5: u = b + c - a)
  (h6: v = (2*a + b - c) / 2) :
  x + a + b = y + z + t ∧
  y + z + t = u ∧
  z + t + u = b + z + c ∧
  t + u + v = a + u + c ∧
  x + t + v = u + y + c ∧
  by sorry :=
sorry

end magic_square_l6_6386


namespace positive_difference_of_squares_l6_6009

theorem positive_difference_of_squares (a b : ℕ) (h1 : a + b = 60) (h2 : a - b = 18) : a^2 - b^2 = 1080 :=
by
  sorry

end positive_difference_of_squares_l6_6009


namespace range_of_reciprocals_l6_6478

theorem range_of_reciprocals (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : a + b = 1) :
  ∃ c ∈ Set.Ici (9 : ℝ), (c = (1/a + 4/b)) :=
by
  sorry

end range_of_reciprocals_l6_6478


namespace rhombus_area_l6_6076

theorem rhombus_area (side_length : ℝ) (d1_diff_d2 : ℝ) 
  (h_side_length : side_length = Real.sqrt 104) 
  (h_d1_diff_d2 : d1_diff_d2 = 10) : 
  (1 / 2) * (2 * Real.sqrt 104 - d1_diff_d2) * (d1_diff_d2 + 2 * Real.sqrt 104) = 79.17 :=
by
  sorry

end rhombus_area_l6_6076


namespace probability_complement_l6_6483

theorem probability_complement (P_A : ℝ) (h : P_A = 0.992) : 1 - P_A = 0.008 := by
  sorry

end probability_complement_l6_6483


namespace range_of_a_l6_6227

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, -5 ≤ x ∧ x ≤ 0 → x^2 + 2 * x - 3 + a ≤ 0) ↔ a ≤ -12 :=
by
  sorry

end range_of_a_l6_6227


namespace empty_solution_set_of_inequalities_l6_6414

theorem empty_solution_set_of_inequalities (a : ℝ) : 
  (∀ x : ℝ, ¬ ((2 * x < 5 - 3 * x) ∧ ((x - 1) / 2 > a))) ↔ (0 ≤ a) := 
by
  sorry

end empty_solution_set_of_inequalities_l6_6414


namespace find_number_of_spiders_l6_6154

theorem find_number_of_spiders (S : ℕ) (h1 : (1 / 2) * S = 5) : S = 10 := sorry

end find_number_of_spiders_l6_6154


namespace tank_capacity_l6_6663

theorem tank_capacity (w c: ℚ) (h1: w / c = 1 / 3) (h2: (w + 5) / c = 2 / 5) :
  c = 37.5 := 
by
  sorry

end tank_capacity_l6_6663


namespace solve_for_b_l6_6142

noncomputable def system_has_solution (b : ℝ) : Prop :=
  ∃ (a : ℝ) (x y : ℝ),
    y = -b - x^2 ∧
    x^2 + y^2 + 8 * a^2 = 4 + 4 * a * (x + y)

theorem solve_for_b (b : ℝ) : system_has_solution b ↔ b ≤ 2 * Real.sqrt 2 + 1 / 4 := 
by 
  sorry

end solve_for_b_l6_6142


namespace intersection_of_A_and_B_l6_6337

-- Conditions: definitions of sets A and B
def set_A : Set ℝ := {x | -1 ≤ x ∧ x < 2}
def set_B : Set ℝ := {x | x < 1}

-- The proof goal: A ∩ B = {x | -1 ≤ x ∧ x < 1}
theorem intersection_of_A_and_B : set_A ∩ set_B = {x | -1 ≤ x ∧ x < 1} := by
  sorry

end intersection_of_A_and_B_l6_6337


namespace proof_problem_l6_6609

def setA : Set ℝ := {x | -1 < x ∧ x ≤ 5}
def setB : Set ℝ := {x | -1 < x ∧ x < 3}

def complementB : Set ℝ := {x | x ≥ 3 ∨ x ≤ -1}
def intersection : Set ℝ := {x | 3 ≤ x ∧ x ≤ 5}

theorem proof_problem :
  (setA ∩ complementB) = intersection := 
by
  sorry

end proof_problem_l6_6609


namespace smallest_positive_integer_l6_6088
-- Import the required library

-- State the problem in Lean
theorem smallest_positive_integer (x : ℕ) (h : 5 * x ≡ 17 [MOD 31]) : x = 13 :=
sorry

end smallest_positive_integer_l6_6088


namespace therapy_charge_l6_6884

-- Let F be the charge for the first hour and A be the charge for each additional hour
-- Two conditions are:
-- 1. F = A + 40
-- 2. F + 4A = 375

-- We need to prove that the total charge for 2 hours of therapy is 174
theorem therapy_charge (A F : ℕ) (h1 : F = A + 40) (h2 : F + 4 * A = 375) :
  F + A = 174 :=
by
  sorry

end therapy_charge_l6_6884


namespace triangle_inequality_for_f_l6_6646

noncomputable def f (x m : ℝ) : ℝ :=
  x^3 - 3 * x + m

theorem triangle_inequality_for_f (a b c m : ℝ) (h₀ : 0 ≤ a) (h₁ : a ≤ 2) (h₂ : 0 ≤ b) (h₃ : b ≤ 2) (h₄ : 0 ≤ c) (h₅ : c ≤ 2) 
(h₆ : 6 < m) :
  ∃ u v w, u = f a m ∧ v = f b m ∧ w = f c m ∧ u + v > w ∧ u + w > v ∧ v + w > u := 
sorry

end triangle_inequality_for_f_l6_6646


namespace arithmetic_sequence_sum_l6_6885

/-- Let {a_n} be an arithmetic sequence with a positive common difference d.
  Given that a_1 + a_2 + a_3 = 15 and a_1 * a_2 * a_3 = 80, we aim to show that
  a_11 + a_12 + a_13 = 105. -/
theorem arithmetic_sequence_sum
  (a : ℕ → ℚ)
  (d : ℚ)
  (h1 : ∀ n, a (n + 1) = a n + d)
  (h2 : d > 0)
  (h3 : a 1 + a 2 + a 3 = 15)
  (h4 : a 1 * a 2 * a 3 = 80) :
  a 11 + a 12 + a 13 = 105 :=
sorry

end arithmetic_sequence_sum_l6_6885


namespace find_f_half_l6_6920

noncomputable def f : ℝ → ℝ := sorry

theorem find_f_half (x : ℝ) (h₀ : 0 ≤ x ∧ x ≤ Real.pi / 2) (h₁ : f (Real.sin x) = x) : 
  f (1 / 2) = Real.pi / 6 :=
sorry

end find_f_half_l6_6920


namespace problem_statement_l6_6979

theorem problem_statement
  (g : ℝ → ℝ)
  (p q r s : ℝ)
  (h_roots : ∃ n1 n2 n3 n4 : ℕ, 
                ∀ x, g x = (x + 2 * n1) * (x + 2 * n2) * (x + 2 * n3) * (x + 2 * n4))
  (h_pqrs : p + q + r + s = 2552)
  (h_g : ∀ x, g x = x^4 + p * x^3 + q * x^2 + r * x + s) :
  s = 3072 :=
by
  sorry

end problem_statement_l6_6979


namespace divide_subtract_multiply_l6_6824

theorem divide_subtract_multiply :
  (-5) / ((1/4) - (1/3)) * 12 = 720 := by
  sorry

end divide_subtract_multiply_l6_6824


namespace find_interest_rate_l6_6876

theorem find_interest_rate
  (P : ℝ) (CI : ℝ) (T : ℝ) (n : ℕ)
  (comp_int_formula : CI = P * ((1 + (r / (n : ℝ))) ^ (n * T)) - P) :
  r = 0.099 :=
by
  have h : CI = 788.13 := sorry
  have hP : P = 5000 := sorry
  have hT : T = 1.5 := sorry
  have hn : (n : ℝ) = 2 := sorry
  sorry

end find_interest_rate_l6_6876


namespace total_carrots_l6_6425

-- Definitions from conditions in a)
def JoanCarrots : ℕ := 29
def JessicaCarrots : ℕ := 11

-- Theorem that encapsulates the problem
theorem total_carrots : JoanCarrots + JessicaCarrots = 40 := by
  sorry

end total_carrots_l6_6425


namespace sequence_bound_l6_6262

noncomputable def sequenceProperties (a : ℕ → ℝ) (c : ℝ) : Prop :=
  (∀ i, 0 ≤ a i ∧ a i ≤ c) ∧ (∀ i j, i ≠ j → abs (a i - a j) ≥ 1 / (i + j))

theorem sequence_bound (a : ℕ → ℝ) (c : ℝ) (h : sequenceProperties a c) : 
  c ≥ 1 :=
by
  sorry

end sequence_bound_l6_6262


namespace customers_not_wanting_change_l6_6486

-- Given Conditions
def cars_initial := 4
def cars_additional := 6
def cars_total := cars_initial + cars_additional
def tires_per_car := 4
def half_change_customers := 2
def tires_for_half_change_customers := 2 * 2 -- 2 cars, 2 tires each
def tires_left := 20

-- Theorem to Prove
theorem customers_not_wanting_change : 
  (cars_total * tires_per_car) - (tires_left + tires_for_half_change_customers) = 
  4 * tires_per_car -> 
  cars_total - ((tires_left + tires_for_half_change_customers) / tires_per_car) - half_change_customers = 4 :=
by
  sorry

end customers_not_wanting_change_l6_6486


namespace calculate_possible_change_l6_6695

structure ChangeProblem where
  (change : ℕ)
  (h1 : change < 100)
  (h2 : ∃ (q : ℕ), change = 25 * q + 10 ∧ q ≤ 3)
  (h3 : ∃ (d : ℕ), change = 10 * d + 20 ∧ d ≤ 9)

theorem calculate_possible_change (p1 p2 p3 p4 : ChangeProblem) :
  p1.change + p2.change + p3.change = 180 :=
by
  sorry

end calculate_possible_change_l6_6695


namespace athlete_difference_l6_6875

-- Define the conditions
def initial_athletes : ℕ := 300
def rate_of_leaving : ℕ := 28
def time_of_leaving : ℕ := 4
def rate_of_arriving : ℕ := 15
def time_of_arriving : ℕ := 7

-- Define intermediary calculations
def number_leaving : ℕ := rate_of_leaving * time_of_leaving
def remaining_athletes : ℕ := initial_athletes - number_leaving
def number_arriving : ℕ := rate_of_arriving * time_of_arriving
def total_sunday_night : ℕ := remaining_athletes + number_arriving

-- Theorem statement
theorem athlete_difference : initial_athletes - total_sunday_night = 7 :=
by
  sorry

end athlete_difference_l6_6875


namespace supermarket_problem_l6_6044

-- Define that type A costs x yuan and type B costs y yuan
def cost_price_per_item (x y : ℕ) : Prop :=
  (10 * x + 8 * y = 880) ∧ (2 * x + 5 * y = 380)

-- Define purchasing plans with the conditions described
def purchasing_plans (a : ℕ) : Prop :=
  ∀ a : ℕ, 24 ≤ a ∧ a ≤ 26

theorem supermarket_problem : 
  (∃ x y, cost_price_per_item x y ∧ x = 40 ∧ y = 60) ∧ 
  (∃ n, purchasing_plans n ∧ n = 3) :=
by
  sorry

end supermarket_problem_l6_6044


namespace nearest_multiple_to_457_divisible_by_11_l6_6023

theorem nearest_multiple_to_457_divisible_by_11 : ∃ n : ℤ, (n % 11 = 0) ∧ (abs (457 - n) = 5) :=
by
  sorry

end nearest_multiple_to_457_divisible_by_11_l6_6023


namespace coefficient_a_eq_2_l6_6370

theorem coefficient_a_eq_2 (a : ℝ) (h : (a^3 * (4 : ℝ)) = 32) : a = 2 :=
by {
  -- Proof will need to be filled in here
  sorry
}

end coefficient_a_eq_2_l6_6370


namespace unique_g_zero_l6_6351

theorem unique_g_zero (g : ℝ → ℝ) (h : ∀ x y : ℝ, g (x + y) = g (x) + g (y) - 1) : g 0 = 1 :=
by
  sorry

end unique_g_zero_l6_6351


namespace boy_current_age_l6_6629

theorem boy_current_age (x : ℕ) (h : 5 ≤ x) (age_statement : x = 2 * (x - 5)) : x = 10 :=
by
  sorry

end boy_current_age_l6_6629


namespace mahmoud_gets_at_least_two_heads_l6_6675

def probability_of_at_least_two_heads := 1 - ((1/2)^5 + 5 * (1/2)^5)

theorem mahmoud_gets_at_least_two_heads (n : ℕ) (hn : n = 5) :
  probability_of_at_least_two_heads = 13 / 16 :=
by
  simp only [probability_of_at_least_two_heads, hn]
  sorry

end mahmoud_gets_at_least_two_heads_l6_6675


namespace james_pre_injury_miles_600_l6_6699

-- Define the conditions
def james_pre_injury_miles (x : ℝ) : Prop :=
  ∃ goal_increase : ℝ, ∃ days : ℝ, ∃ weekly_increase : ℝ,
  goal_increase = 1.2 * x ∧
  days = 280 ∧
  weekly_increase = 3 ∧
  (days / 7) * weekly_increase = (goal_increase - x)

-- Define the main theorem to be proved
theorem james_pre_injury_miles_600 : james_pre_injury_miles 600 :=
sorry

end james_pre_injury_miles_600_l6_6699


namespace sum_of_roots_eq_14_l6_6358

theorem sum_of_roots_eq_14 (x : ℝ) :
  (x - 7) ^ 2 = 16 → ∃ r1 r2 : ℝ, (x = r1 ∨ x = r2) ∧ r1 + r2 = 14 :=
by
  intro h
  sorry

end sum_of_roots_eq_14_l6_6358


namespace parabola_focus_l6_6783

theorem parabola_focus (x y : ℝ) :
  (∃ x, y = 4 * x^2 + 8 * x - 5) →
  (x, y) = (-1, -8.9375) :=
by
  sorry

end parabola_focus_l6_6783


namespace tank_capacity_l6_6909

variable (C : ℝ)  -- total capacity of the tank

-- The tank is 5/8 full initially
axiom h1 : (5/8) * C + 15 = (19/24) * C

theorem tank_capacity : C = 90 :=
by
  sorry

end tank_capacity_l6_6909


namespace number_exceeds_fraction_l6_6435

theorem number_exceeds_fraction (x : ℝ) (h : x = (3/8) * x + 15) : x = 24 :=
sorry

end number_exceeds_fraction_l6_6435


namespace width_of_sheet_of_paper_l6_6327

theorem width_of_sheet_of_paper (W : ℝ) (h1 : ∀ (W : ℝ), W > 0) (length_paper : ℝ) (margin : ℝ)
  (width_picture_area : ∀ (W : ℝ), W - 2 * margin = (W - 3)) 
  (area_picture : ℝ) (length_picture_area : ℝ) :
  length_paper = 10 ∧ margin = 1.5 ∧ area_picture = 38.5 ∧ length_picture_area = 7 →
  W = 8.5 :=
by
  sorry

end width_of_sheet_of_paper_l6_6327


namespace simplify_expression_l6_6700

theorem simplify_expression :
  10 / (2 / 0.3) / (0.3 / 0.04) / (0.04 / 0.05) = 0.25 :=
by
  sorry

end simplify_expression_l6_6700


namespace denomination_of_remaining_notes_eq_500_l6_6939

-- Definitions of the given conditions:
def total_money : ℕ := 10350
def total_notes : ℕ := 126
def n_50_notes : ℕ := 117

-- The theorem stating what we need to prove
theorem denomination_of_remaining_notes_eq_500 :
  ∃ (X : ℕ), X = 500 ∧ total_money = (n_50_notes * 50 + (total_notes - n_50_notes) * X) :=
by
sorry

end denomination_of_remaining_notes_eq_500_l6_6939


namespace evaluate_expression_l6_6148

theorem evaluate_expression (x : ℝ) (h : |7 - 8 * (x - 12)| - |5 - 11| = 73) : x = 3 :=
  sorry

end evaluate_expression_l6_6148


namespace abc_cube_geq_abc_sum_l6_6535

theorem abc_cube_geq_abc_sum (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (a ^ a * b ^ b * c ^ c) ^ 3 ≥ (a * b * c) ^ (a + b + c) :=
by
  sorry

end abc_cube_geq_abc_sum_l6_6535


namespace geometric_sequence_seventh_term_l6_6819

theorem geometric_sequence_seventh_term (a r: ℤ) (h1 : a = 3) (h2 : a * r ^ 5 = 729) : a * r ^ 6 = 2187 :=
by sorry

end geometric_sequence_seventh_term_l6_6819


namespace solve_equation_l6_6541

theorem solve_equation :
  ∀ x : ℝ, 81 * (1 - x) ^ 2 = 64 ↔ x = 1 / 9 ∨ x = 17 / 9 :=
by
  sorry

end solve_equation_l6_6541


namespace value_of_sum_l6_6347

theorem value_of_sum (a x y : ℝ) (h1 : 17 * x + 19 * y = 6 - a) (h2 : 13 * x - 7 * y = 10 * a + 1) : 
  x + y = 1 / 3 := 
sorry

end value_of_sum_l6_6347


namespace min_sum_x_y_l6_6362

theorem min_sum_x_y (x y : ℕ) (h1 : x ≠ y) (h2 : x > 0 ∧ y > 0) (h3 : (1 : ℚ)/x + (1 : ℚ)/y = 1/12) : x + y = 49 :=
sorry

end min_sum_x_y_l6_6362


namespace evaluation_l6_6032
-- Import the entire Mathlib library

-- Define the operations triangle and nabla
def triangle (a b : ℕ) : ℕ := 3 * a + 2 * b
def nabla (a b : ℕ) : ℕ := 2 * a + 3 * b

-- The proof statement
theorem evaluation : triangle 2 (nabla 3 4) = 42 :=
by
  -- Provide a placeholder for the proof
  sorry

end evaluation_l6_6032


namespace spheres_max_min_dist_l6_6515

variable {R_1 R_2 d : ℝ}

noncomputable def max_min_dist (R_1 R_2 d : ℝ) (sep : d > R_1 + R_2) :
  ℝ × ℝ :=
(d + R_1 + R_2, d - R_1 - R_2)

theorem spheres_max_min_dist {R_1 R_2 d : ℝ} (sep : d > R_1 + R_2) :
  max_min_dist R_1 R_2 d sep = (d + R_1 + R_2, d - R_1 - R_2) := by
sorry

end spheres_max_min_dist_l6_6515


namespace least_possible_coins_l6_6100

theorem least_possible_coins : 
  ∃ b : ℕ, b % 7 = 3 ∧ b % 4 = 2 ∧ ∀ n : ℕ, (n % 7 = 3 ∧ n % 4 = 2) → b ≤ n :=
sorry

end least_possible_coins_l6_6100


namespace scientific_notation_140000000_l6_6291

theorem scientific_notation_140000000 :
  140000000 = 1.4 * 10^8 := 
sorry

end scientific_notation_140000000_l6_6291


namespace correct_product_of_a_and_b_l6_6531

theorem correct_product_of_a_and_b
    (a b : ℕ)
    (h_a_pos : 0 < a)
    (h_b_pos : 0 < b)
    (h_a_two_digits : 10 ≤ a ∧ a < 100)
    (a' : ℕ)
    (h_a' : a' = (a % 10) * 10 + (a / 10))
    (h_product_erroneous : a' * b = 198) :
  a * b = 198 :=
sorry

end correct_product_of_a_and_b_l6_6531


namespace difficult_vs_easy_l6_6253

theorem difficult_vs_easy (x y z : ℕ) (h1 : x + y + z = 100) (h2 : x + 3 * y + 2 * z = 180) :
  x - y = 20 :=
by sorry

end difficult_vs_easy_l6_6253


namespace volume_ratio_l6_6059

def cube_volume (side_length : ℝ) : ℝ :=
  side_length ^ 3

theorem volume_ratio : 
  let a := (4 : ℝ) / 12   -- 4 inches converted to feet
  let b := (2 : ℝ)       -- 2 feet
  cube_volume a / cube_volume b = 1 / 216 :=
by
  sorry

end volume_ratio_l6_6059


namespace gcd_problem_l6_6389

def gcd3 (x y z : ℕ) : ℕ := Int.gcd x (Int.gcd y z)

theorem gcd_problem (a b c : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (h4 : gcd3 (a^2 - 1) (b^2 - 1) (c^2 - 1) = 1) :
  gcd3 (a * b + c) (b * c + a) (c * a + b) = gcd3 a b c :=
by
  sorry

end gcd_problem_l6_6389


namespace area_converted_2018_l6_6226

theorem area_converted_2018 :
  let a₁ := 8 -- initial area in ten thousand hectares
  let q := 1.1 -- common ratio
  let a₆ := a₁ * q^5 -- area converted in 2018
  a₆ = 8 * 1.1^5 :=
sorry

end area_converted_2018_l6_6226


namespace polygon_area_is_nine_l6_6416

-- Definitions of vertices and coordinates.
def vertexA := (0, 0)
def vertexD := (3, 0)
def vertexP := (3, 3)
def vertexM := (0, 3)

-- Area of the polygon formed by the vertices A, D, P, M.
def polygonArea (A D P M : ℕ × ℕ) : ℕ :=
  (D.1 - A.1) * (P.2 - A.2)

-- Statement of the theorem.
theorem polygon_area_is_nine : polygonArea vertexA vertexD vertexP vertexM = 9 := by
  sorry

end polygon_area_is_nine_l6_6416


namespace algebraic_expression_value_l6_6950

theorem algebraic_expression_value (x : ℝ) (h : x ^ 2 - 3 * x = 4) : 2 * x ^ 2 - 6 * x - 3 = 5 :=
by
  sorry

end algebraic_expression_value_l6_6950


namespace find_older_friend_age_l6_6314

theorem find_older_friend_age (A B C : ℕ) 
  (h1 : A - B = 2) 
  (h2 : A - C = 5) 
  (h3 : A + B + C = 110) : 
  A = 39 := 
by 
  sorry

end find_older_friend_age_l6_6314


namespace seeds_planted_on_wednesday_l6_6855

theorem seeds_planted_on_wednesday
  (total_seeds : ℕ) (seeds_thursday : ℕ) (seeds_wednesday : ℕ)
  (h_total : total_seeds = 22) (h_thursday : seeds_thursday = 2) :
  seeds_wednesday = 20 ↔ total_seeds - seeds_thursday = seeds_wednesday :=
by
  -- the proof would go here
  sorry

end seeds_planted_on_wednesday_l6_6855


namespace calculate_product_sum_l6_6764

theorem calculate_product_sum :
  17 * (17/18) + 35 * (35/36) = 50 + 1/12 :=
by sorry

end calculate_product_sum_l6_6764


namespace least_positive_integer_to_multiple_of_5_l6_6627

theorem least_positive_integer_to_multiple_of_5 : ∃ (n : ℕ), n > 0 ∧ (725 + n) % 5 = 0 ∧ ∀ m : ℕ, m > 0 ∧ (725 + m) % 5 = 0 → n ≤ m :=
by
  sorry

end least_positive_integer_to_multiple_of_5_l6_6627


namespace derek_history_test_l6_6794

theorem derek_history_test :
  let ancient_questions := 20
  let medieval_questions := 25
  let modern_questions := 35
  let total_questions := ancient_questions + medieval_questions + modern_questions

  let derek_ancient_correct := 0.60 * ancient_questions
  let derek_medieval_correct := 0.56 * medieval_questions
  let derek_modern_correct := 0.70 * modern_questions

  let derek_total_correct := derek_ancient_correct + derek_medieval_correct + derek_modern_correct

  let passing_score := 0.65 * total_questions
  (derek_total_correct < passing_score) →
  passing_score - derek_total_correct = 2
  := by
  sorry

end derek_history_test_l6_6794


namespace correct_factorization_l6_6659

theorem correct_factorization :
  (∀ a b : ℝ, ¬ (a^2 + b^2 = (a + b) * (a - b))) ∧
  (∀ a : ℝ, ¬ (a^4 - 1 = (a^2 + 1) * (a^2 - 1))) ∧
  (∀ x : ℝ, ¬ (x^2 + 2 * x + 4 = (x + 2)^2)) ∧
  (∀ x : ℝ, x^2 - 3 * x + 2 = (x - 1) * (x - 2)) :=
by
  sorry

end correct_factorization_l6_6659


namespace arithmetic_sequence_tenth_term_l6_6260

/- 
  Define the arithmetic sequence in terms of its properties 
  and prove that the 10th term is 18.
-/

theorem arithmetic_sequence_tenth_term (a : ℕ → ℤ) (h_arith : ∀ n : ℕ, a (n + 1) - a n = a 1 - a 0)
  (h_a2 : a 2 = 2) (h_a5 : a 5 = 8) : a 10 = 18 := 
by 
  sorry

end arithmetic_sequence_tenth_term_l6_6260


namespace original_price_of_cupcakes_l6_6087

theorem original_price_of_cupcakes
  (revenue : ℕ := 32) 
  (cookies_sold : ℕ := 8) 
  (cupcakes_sold : ℕ := 16) 
  (cookie_price: ℕ := 2)
  (half_price_of_cookie: ℕ := 1) :
  (x : ℕ) → (16 * (x / 2)) + (8 * 1) = 32 → x = 3 := 
by
  sorry

end original_price_of_cupcakes_l6_6087


namespace total_sweaters_calculated_l6_6770

def monday_sweaters := 8
def tuesday_sweaters := monday_sweaters + 2
def wednesday_sweaters := tuesday_sweaters - 4
def thursday_sweaters := tuesday_sweaters - 4
def friday_sweaters := monday_sweaters / 2

def total_sweaters := monday_sweaters + tuesday_sweaters + wednesday_sweaters + thursday_sweaters + friday_sweaters

theorem total_sweaters_calculated : total_sweaters = 34 := 
by sorry

end total_sweaters_calculated_l6_6770


namespace fraction_ordering_l6_6421

theorem fraction_ordering :
  (8 / 25 : ℚ) < 6 / 17 ∧ 6 / 17 < 10 / 27 ∧ 8 / 25 < 10 / 27 :=
by
  sorry

end fraction_ordering_l6_6421


namespace problem1_l6_6012

theorem problem1 : 
  ∀ a b : ℤ, a = 1 → b = -3 → (a - b)^2 - 2 * a * (a + 3 * b) + (a + 2 * b) * (a - 2 * b) = -3 :=
by
  intros a b h1 h2
  rw [h1, h2]
  sorry

end problem1_l6_6012


namespace contestant_advancing_probability_l6_6891

noncomputable def probability_correct : ℝ := 0.8
noncomputable def probability_incorrect : ℝ := 1 - probability_correct

def sequence_pattern (q1 q2 q3 q4 : Bool) : Bool :=
  -- Pattern INCORRECT, CORRECT, CORRECT, CORRECT
  q1 == false ∧ q2 == true ∧ q3 == true ∧ q4 == true

def probability_pattern (p_corr p_incorr : ℝ) : ℝ :=
  p_incorr * p_corr * p_corr * p_corr

theorem contestant_advancing_probability :
  (probability_pattern probability_correct probability_incorrect = 0.1024) :=
by
  -- Proof required here
  sorry

end contestant_advancing_probability_l6_6891


namespace solve_congruence_l6_6343

open Nat

theorem solve_congruence (x : ℕ) (h : x^2 + x - 6 ≡ 0 [MOD 143]) : 
  x = 2 ∨ x = 41 ∨ x = 101 ∨ x = 140 :=
by
  sorry

end solve_congruence_l6_6343


namespace ceil_floor_difference_l6_6960

theorem ceil_floor_difference : 
  (Int.ceil ((15 : ℚ) / 8 * ((-34 : ℚ) / 4)) - Int.floor (((15 : ℚ) / 8) * Int.ceil ((-34 : ℚ) / 4)) = 0) :=
by 
  sorry

end ceil_floor_difference_l6_6960


namespace dice_product_sum_impossible_l6_6330

theorem dice_product_sum_impossible (d1 d2 d3 d4 : ℕ) (h1 : 1 ≤ d1 ∧ d1 ≤ 6) (h2 : 1 ≤ d2 ∧ d2 ≤ 6) (h3 : 1 ≤ d3 ∧ d3 ≤ 6) (h4 : 1 ≤ d4 ∧ d4 ≤ 6) (hprod : d1 * d2 * d3 * d4 = 180) :
  (d1 + d2 + d3 + d4 ≠ 14) ∧ (d1 + d2 + d3 + d4 ≠ 17) :=
by
  sorry

end dice_product_sum_impossible_l6_6330


namespace ash_cloud_ratio_l6_6745

theorem ash_cloud_ratio
  (distance_ashes_shot_up : ℕ)
  (radius_ash_cloud : ℕ)
  (h1 : distance_ashes_shot_up = 300)
  (h2 : radius_ash_cloud = 2700) :
  (2 * radius_ash_cloud) / distance_ashes_shot_up = 18 :=
by
  sorry

end ash_cloud_ratio_l6_6745


namespace jane_buys_bagels_l6_6781

variable (b m : ℕ)
variable (h1 : b + m = 7)
variable (h2 : 65 * b + 40 * m % 100 = 80)
variable (h3 : 40 * b + 40 * m % 100 = 0)

theorem jane_buys_bagels : b = 4 := by sorry

end jane_buys_bagels_l6_6781


namespace shanghai_world_expo_l6_6354

theorem shanghai_world_expo (n : ℕ) (total_cost : ℕ) 
  (H1 : total_cost = 4000)
  (H2 : n ≤ 30 → total_cost = n * 120)
  (H3 : n > 30 → total_cost = n * (120 - 2 * (n - 30)) ∧ (120 - 2 * (n - 30)) ≥ 90) :
  n = 40 := 
sorry

end shanghai_world_expo_l6_6354


namespace necessary_condition_l6_6775

theorem necessary_condition (A B C D : Prop) (h1 : A > B → C < D) : A > B → C < D := by
  exact h1 -- This is just a placeholder for the actual hypothesis, a required assumption in our initial problem statement

end necessary_condition_l6_6775


namespace Ms_Thompsons_statement_contrapositive_of_Ms_Thompsons_statement_l6_6472

-- Define P and Q as propositions where P indicates submission of all required essays and Q indicates failing the course.
variable (P Q : Prop)

-- Ms. Thompson's statement translated to logical form.
theorem Ms_Thompsons_statement : ¬P → Q := sorry

-- The goal is to prove that if a student did not fail the course, then they submitted all the required essays.
theorem contrapositive_of_Ms_Thompsons_statement (h : ¬Q) : P := 
by {
  -- Proof will go here
  sorry 
}

end Ms_Thompsons_statement_contrapositive_of_Ms_Thompsons_statement_l6_6472


namespace part1_part2_l6_6299

def pointA : (ℝ × ℝ) := (1, 2)
def pointB : (ℝ × ℝ) := (-2, 3)
def pointC : (ℝ × ℝ) := (8, -5)

-- Definitions of the vectors
def OA : (ℝ × ℝ) := pointA
def OB : (ℝ × ℝ) := pointB
def OC : (ℝ × ℝ) := pointC
def AB : (ℝ × ℝ) := (pointB.1 - pointA.1, pointB.2 - pointA.2)

-- Part 1: Proving the values of x and y
theorem part1 : ∃ (x y : ℝ), OC = (x * OA.1 + y * OB.1, x * OA.2 + y * OB.2) ∧ x = 2 ∧ y = -3 :=
by
  sorry

-- Part 2: Proving the value of m when vectors are parallel
theorem part2 : ∃ (m : ℝ), ∃ k : ℝ, AB = (k * (m + 8), k * (2 * m - 5)) ∧ m = 1 :=
by
  sorry

end part1_part2_l6_6299


namespace length_of_ae_l6_6789

def consecutive_points_on_line (a b c d e : ℝ) : Prop :=
  ∃ (ab bc cd de : ℝ), 
  ab = 5 ∧ 
  bc = 2 * cd ∧ 
  de = 4 ∧ 
  a + ab = b ∧ 
  b + bc = c ∧ 
  c + cd = d ∧ 
  d + de = e ∧
  a + ab + bc = c -- ensuring ac = 11

theorem length_of_ae (a b c d e : ℝ) 
  (h1 : consecutive_points_on_line a b c d e) 
  (h2 : a + 5 = b)
  (h3 : b + 2 * (c - b) = c)
  (h4 : d - c = 3)
  (h5 : d + 4 = e)
  (h6 : a + 5 + 2 * (c - b) = c) :
  e - a = 18 :=
sorry

end length_of_ae_l6_6789


namespace f_positive_for_all_x_f_increasing_solution_set_inequality_l6_6840

namespace ProofProblem

-- Define the function f and its properties
def f : ℝ → ℝ := sorry

axiom f_zero_ne_zero : f 0 ≠ 0
axiom f_one_eq_two : f 1 = 2
axiom f_pos_when_pos : ∀ x : ℝ, x > 0 → f x > 1
axiom f_add_mul : ∀ a b : ℝ, f (a + b) = f a * f b

-- Problem 1: Prove that f(x) > 0 for all x ∈ ℝ
theorem f_positive_for_all_x : ∀ x : ℝ, f x > 0 := sorry

-- Problem 2: Prove that f(x) is increasing on ℝ
theorem f_increasing : ∀ x y : ℝ, x < y → f x < f y := sorry

-- Problem 3: Find the solution set of the inequality f(3-2x) > 4
theorem solution_set_inequality : { x : ℝ | f (3 - 2 * x) > 4 } = { x | x < 1 / 2 } := sorry

end ProofProblem

end f_positive_for_all_x_f_increasing_solution_set_inequality_l6_6840


namespace sum_first_23_natural_numbers_l6_6460

theorem sum_first_23_natural_numbers :
  (23 * (23 + 1)) / 2 = 276 := 
by
  sorry

end sum_first_23_natural_numbers_l6_6460


namespace problem1_problem2_l6_6519

def f (x b : ℝ) : ℝ := |x - b| + |x + b|

theorem problem1 (x : ℝ) : (∀ y, y = 1 → f x y ≤ x + 2) ↔ (0 ≤ x ∧ x ≤ 2) :=
sorry

theorem problem2 (a b : ℝ) (h : a ≠ 0) : (∀ y, y = 1 → f y b ≥ (|a + 1| - |2 * a - 1|) / |a|) ↔ (b ≤ -3 / 2 ∨ b ≥ 3 / 2) :=
sorry

end problem1_problem2_l6_6519


namespace marks_difference_l6_6863

theorem marks_difference (A B C D E : ℕ) 
  (h1 : (A + B + C) / 3 = 48) 
  (h2 : (A + B + C + D) / 4 = 47) 
  (h3 : E > D) 
  (h4 : (B + C + D + E) / 4 = 48) 
  (h5 : A = 43) : 
  E - D = 3 := 
sorry

end marks_difference_l6_6863


namespace max_value_ab_bc_cd_l6_6800

theorem max_value_ab_bc_cd (a b c d : ℝ) (h_nonneg : a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0)
  (h_sum : a + b + c + d = 120) : ab + bc + cd ≤ 3600 :=
by {
  sorry
}

end max_value_ab_bc_cd_l6_6800


namespace find_n_value_l6_6490

theorem find_n_value (n : ℕ) (h : ∃ k : ℤ, n^2 + 5 * n + 13 = k^2) : n = 4 :=
by
  sorry

end find_n_value_l6_6490


namespace expected_games_is_correct_l6_6729

def prob_A_wins : ℚ := 2 / 3
def prob_B_wins : ℚ := 1 / 3
def max_games : ℕ := 6

noncomputable def expected_games : ℚ :=
  2 * (prob_A_wins^2 + prob_B_wins^2) +
  4 * (prob_A_wins * prob_B_wins * (prob_A_wins^2 + prob_B_wins^2)) +
  6 * (prob_A_wins * prob_B_wins)^2

theorem expected_games_is_correct : expected_games = 266 / 81 := by
  sorry

end expected_games_is_correct_l6_6729


namespace train_trip_length_l6_6150

theorem train_trip_length (x D : ℝ) (h1 : D > 0) (h2 : x > 0) 
(h3 : 2 + 3 * (D - 2 * x) / (2 * x) + 1 = (x + 240) / x + 1 + 3 * (D - 2 * x - 120) / (2 * x) - 0.5) 
(h4 : 3 + 3 * (D - 2 * x) / (2 * x) = 7) :
  D = 640 :=
by
  sorry

end train_trip_length_l6_6150


namespace rhombus_area_l6_6221

theorem rhombus_area (x y : ℝ)
  (h1 : x^2 + y^2 = 113) 
  (h2 : x = y + 8) : 
  1 / 2 * (2 * y) * (2 * (y + 4)) = 97 := 
by 
  -- Assume x and y are the half-diagonals of the rhombus
  sorry

end rhombus_area_l6_6221


namespace roses_ordered_l6_6505

theorem roses_ordered (tulips carnations roses : ℕ) (cost_per_flower total_expenses : ℕ)
  (h1 : tulips = 250)
  (h2 : carnations = 375)
  (h3 : cost_per_flower = 2)
  (h4 : total_expenses = 1890)
  (h5 : total_expenses = (tulips + carnations + roses) * cost_per_flower) :
  roses = 320 :=
by 
  -- Using the mathematical equivalence and conditions provided
  sorry

end roses_ordered_l6_6505


namespace complex_quadrant_l6_6382

theorem complex_quadrant (x y: ℝ) (h : x = 1 ∧ y = 2) : x > 0 ∧ y > 0 :=
by
  sorry

end complex_quadrant_l6_6382


namespace total_distance_walked_l6_6578

theorem total_distance_walked (t1 t2 : ℝ) (r : ℝ) (total_distance : ℝ)
  (h1 : t1 = 15 / 60)  -- Convert 15 minutes to hours
  (h2 : t2 = 25 / 60)  -- Convert 25 minutes to hours
  (h3 : r = 3)         -- Average speed in miles per hour
  (h4 : total_distance = r * (t1 + t2))
  : total_distance = 2 :=
by
  -- here is where the proof would go
  sorry

end total_distance_walked_l6_6578


namespace largest_5_digit_congruent_15_mod_24_l6_6455

theorem largest_5_digit_congruent_15_mod_24 : ∃ x, 10000 ≤ x ∧ x < 100000 ∧ x % 24 = 15 ∧ x = 99999 := by
  sorry

end largest_5_digit_congruent_15_mod_24_l6_6455


namespace lcm_inequality_l6_6527

theorem lcm_inequality (m n : ℕ) (h : n > m) :
  Nat.lcm m n + Nat.lcm (m + 1) (n + 1) ≥ 2 * m * n / Real.sqrt (n - m) := 
sorry

end lcm_inequality_l6_6527


namespace chromium_percentage_new_alloy_l6_6749

-- Conditions as definitions
def first_alloy_chromium_percentage : ℝ := 12
def second_alloy_chromium_percentage : ℝ := 8
def first_alloy_weight : ℝ := 10
def second_alloy_weight : ℝ := 30

-- Final proof statement
theorem chromium_percentage_new_alloy : 
  ((first_alloy_chromium_percentage / 100 * first_alloy_weight +
    second_alloy_chromium_percentage / 100 * second_alloy_weight) /
  (first_alloy_weight + second_alloy_weight)) * 100 = 9 :=
by
  sorry

end chromium_percentage_new_alloy_l6_6749


namespace bus_cost_proof_l6_6579

-- Define conditions
def train_cost (bus_cost : ℚ) : ℚ := bus_cost + 6.85
def discount_rate : ℚ := 0.15
def service_fee : ℚ := 1.25
def combined_cost : ℚ := 10.50

-- Formula for the total cost after discount
def discounted_train_cost (bus_cost : ℚ) : ℚ := (train_cost bus_cost) * (1 - discount_rate)
def total_cost (bus_cost : ℚ) : ℚ := discounted_train_cost bus_cost + bus_cost + service_fee

-- Lean 4 statement asserting the cost of the bus ride before service fee
theorem bus_cost_proof : ∃ (B : ℚ), total_cost B = combined_cost ∧ B = 1.85 :=
sorry

end bus_cost_proof_l6_6579


namespace symmetric_point_l6_6180

theorem symmetric_point (x y : ℝ) (hx : x = -2) (hy : y = 3) (a b : ℝ) (hne : y = x + 1)
  (halfway : (a = (x + (-2)) / 2) ∧ (b = (y + 3) / 2) ∧ (2 * b = 2 * a + 2) ∧ (2 * b = 1)):
  (a, b) = (0, 1) :=
by
  sorry

end symmetric_point_l6_6180


namespace find_crew_members_l6_6520

noncomputable def passengers_initial := 124
noncomputable def passengers_texas := passengers_initial - 58 + 24
noncomputable def passengers_nc := passengers_texas - 47 + 14
noncomputable def total_people_virginia := 67

theorem find_crew_members (passengers_initial passengers_texas passengers_nc total_people_virginia : ℕ) :
  passengers_initial = 124 →
  passengers_texas = passengers_initial - 58 + 24 →
  passengers_nc = passengers_texas - 47 + 14 →
  total_people_virginia = 67 →
  ∃ crew_members : ℕ, total_people_virginia = passengers_nc + crew_members ∧ crew_members = 10 :=
by
  sorry

end find_crew_members_l6_6520


namespace ratio_after_girls_leave_l6_6338

-- Define the initial conditions
def initial_conditions (B G : ℕ) : Prop :=
  B = G ∧ B + G = 32

-- Define the event of girls leaving
def girls_leave (G : ℕ) : ℕ :=
  G - 8

-- Define the final ratio of boys to girls
def final_ratio (B G : ℕ) : ℕ :=
  B / (girls_leave G)

-- Prove the final ratio is 2:1
theorem ratio_after_girls_leave (B G : ℕ) (h : initial_conditions B G) :
  final_ratio B G = 2 :=
by
  sorry

end ratio_after_girls_leave_l6_6338


namespace unknown_number_lcm_hcf_l6_6269

theorem unknown_number_lcm_hcf (a b : ℕ) 
  (lcm_ab : Nat.lcm a b = 192) 
  (hcf_ab : Nat.gcd a b = 16) 
  (known_number : a = 64) :
  b = 48 :=
by
  sorry -- Proof is omitted as per instruction

end unknown_number_lcm_hcf_l6_6269


namespace tolya_is_older_by_either_4_or_22_years_l6_6139

-- Definitions of the problem conditions
def sum_of_digits (n : ℕ) : ℕ := n.digits 10 |>.sum

def kolya_conditions (y : ℕ) : Prop :=
  1985 ≤ y ∧ y + sum_of_digits y = 2013

def tolya_conditions (y : ℕ) : Prop :=
  1985 ≤ y ∧ y + sum_of_digits y = 2014

-- The problem statement
theorem tolya_is_older_by_either_4_or_22_years (k_birth t_birth : ℕ) 
  (hk : kolya_conditions k_birth) (ht : tolya_conditions t_birth) :
  t_birth - k_birth = 4 ∨ t_birth - k_birth = 22 :=
sorry

end tolya_is_older_by_either_4_or_22_years_l6_6139


namespace total_birds_in_marsh_l6_6108

-- Define the number of geese and ducks as constants.
def geese : Nat := 58
def ducks : Nat := 37

-- The theorem that we need to prove.
theorem total_birds_in_marsh : geese + ducks = 95 :=
by
  -- Here, we add the sorry keyword to skip the proof part.
  sorry

end total_birds_in_marsh_l6_6108


namespace hyperbola_asymptote_l6_6543

theorem hyperbola_asymptote (a : ℝ) (h : a > 0) (hyp_eq : ∀ x y : ℝ, (x^2) / (a^2) - (y^2) / 81 = 1 → y = 3 * x) : a = 3 := 
by
  sorry

end hyperbola_asymptote_l6_6543


namespace neither_5_nor_6_nice_1200_l6_6968

def is_k_nice (N k : ℕ) : Prop := N % k = 1

def count_k_nice_up_to (k n : ℕ) : ℕ :=
(n + (k - 1)) / k

def count_neither_5_nor_6_nice_up_to (n : ℕ) : ℕ :=
  let count_5_nice := count_k_nice_up_to 5 n
  let count_6_nice := count_k_nice_up_to 6 n
  let count_5_and_6_nice := count_k_nice_up_to 30 n
  n - (count_5_nice + count_6_nice - count_5_and_6_nice)

theorem neither_5_nor_6_nice_1200 : count_neither_5_nor_6_nice_up_to 1200 = 800 := 
by
  sorry

end neither_5_nor_6_nice_1200_l6_6968


namespace cylinder_volume_ratio_l6_6899

noncomputable def ratio_of_volumes (r h V_small V_large : ℝ) : ℝ := V_large / V_small

theorem cylinder_volume_ratio (r : ℝ) (h : ℝ) 
  (original_height : ℝ := 3 * r)
  (height_small : ℝ := r / 4)
  (height_large : ℝ := 3 * r - height_small)
  (A_small : ℝ := 2 * π * r * (r + height_small))
  (A_large : ℝ := 2 * π * r * (r + height_large))
  (V_small : ℝ := π * r^2 * height_small) 
  (V_large : ℝ := π * r^2 * height_large) :
  A_large = 3 * A_small → 
  ratio_of_volumes r height_small V_small V_large = 11 := by 
  sorry

end cylinder_volume_ratio_l6_6899


namespace plane_speeds_l6_6874

-- Define the speeds of the planes
def speed_slower (x : ℕ) := x
def speed_faster (x : ℕ) := 2 * x

-- Define the distances each plane travels in 3 hours
def distance_slower (x : ℕ) := 3 * speed_slower x
def distance_faster (x : ℕ) := 3 * speed_faster x

-- Define the total distance
def total_distance (x : ℕ) := distance_slower x + distance_faster x

-- Prove the speeds given the total distance
theorem plane_speeds (x : ℕ) (h : total_distance x = 2700) : speed_slower x = 300 ∧ speed_faster x = 600 :=
by {
  sorry
}

end plane_speeds_l6_6874


namespace correct_average_marks_l6_6352

theorem correct_average_marks (n : ℕ) (average initial_wrong current_correct : ℕ) 
  (h_n : n = 10) 
  (h_avg : average = 100) 
  (h_wrong : initial_wrong = 60)
  (h_correct : current_correct = 10) : 
  (average * n - initial_wrong + current_correct) / n = 95 := 
by
  -- This is where the proof would go
  sorry

end correct_average_marks_l6_6352


namespace range_of_a_l6_6797

noncomputable def range_of_a_condition (a : ℝ) : Prop :=
  ∃ x : ℝ, |x + 1| + |x - a| ≤ 2

theorem range_of_a : ∀ a : ℝ, range_of_a_condition a → (-3 : ℝ) ≤ a ∧ a ≤ 1 :=
by
  intros a h
  sorry

end range_of_a_l6_6797


namespace car_winning_probability_l6_6776

noncomputable def probability_of_winning (P_X P_Y P_Z : ℚ) : ℚ :=
  P_X + P_Y + P_Z

theorem car_winning_probability :
  let P_X := (1 : ℚ) / 6
  let P_Y := (1 : ℚ) / 10
  let P_Z := (1 : ℚ) / 8
  probability_of_winning P_X P_Y P_Z = 47 / 120 :=
by
  sorry

end car_winning_probability_l6_6776


namespace area_of_border_l6_6967

theorem area_of_border (height_painting width_painting border_width : ℕ)
    (area_painting framed_height framed_width : ℕ)
    (H1 : height_painting = 12)
    (H2 : width_painting = 15)
    (H3 : border_width = 3)
    (H4 : area_painting = height_painting * width_painting)
    (H5 : framed_height = height_painting + 2 * border_width)
    (H6 : framed_width = width_painting + 2 * border_width)
    (area_framed : ℕ)
    (H7 : area_framed = framed_height * framed_width) :
    area_framed - area_painting = 198 := 
sorry

end area_of_border_l6_6967


namespace son_age_l6_6501

theorem son_age (M S : ℕ) (h1 : M = S + 24) (h2 : M + 2 = 2 * (S + 2)) : S = 22 :=
by
  sorry

end son_age_l6_6501


namespace range_of_m_l6_6363

theorem range_of_m
  (m : ℝ)
  (h1 : (m - 1) * (3 - m) ≠ 0) 
  (h2 : 3 - m > 0) 
  (h3 : m - 1 > 0) 
  (h4 : 3 - m ≠ m - 1) :
  1 < m ∧ m < 3 ∧ m ≠ 2 :=
sorry

end range_of_m_l6_6363


namespace quadratic_two_distinct_real_roots_l6_6050

theorem quadratic_two_distinct_real_roots : 
  ∀ x : ℝ, ∃ a b c : ℝ, (∀ x : ℝ, (x+1)*(x-1) = 2*x + 3 → x^2 - 2*x - 4 = 0) ∧ 
  (a = 1) ∧ (b = -2) ∧ (c = -4) ∧ (b^2 - 4*a*c > 0) :=
by
  sorry

end quadratic_two_distinct_real_roots_l6_6050


namespace inequality_ln_l6_6574

theorem inequality_ln (x : ℝ) (h₁ : x > -1) (h₂ : x ≠ 0) :
    (2 * abs x) / (2 + x) < abs (Real.log (1 + x)) ∧ abs (Real.log (1 + x)) < (abs x) / Real.sqrt (1 + x) :=
by
  sorry

end inequality_ln_l6_6574


namespace part_one_part_two_l6_6274

noncomputable def f (x : ℝ) : ℝ := (x + 1) * Real.log x - x + 1

noncomputable def f' (x : ℝ) : ℝ := Real.log x + 1 / x

theorem part_one (x a : ℝ) (hx : x > 0) (ineq : x * f' x ≤ x^2 + a * x + 1) : a ∈ Set.Ici (-1) :=
by sorry

theorem part_two (x : ℝ) (hx : x > 0) : (x - 1) * f x ≥ 0 :=
by sorry

end part_one_part_two_l6_6274


namespace green_to_blue_ratio_l6_6772

-- Definition of the problem conditions
variable (G B R : ℕ)
variable (H1 : 2 * G = R)
variable (H2 : B = 80)
variable (H3 : R = 1280)

-- Theorem statement: the ratio of the green car's speed to the blue car's speed is 8:1
theorem green_to_blue_ratio (G B R : ℕ) (H1 : 2 * G = R) (H2 : B = 80) (H3 : R = 1280) :
  G / B = 8 :=
by
  sorry

end green_to_blue_ratio_l6_6772


namespace volume_of_prism_l6_6200

-- Define the variables a, b, c and the conditions
variables (a b c : ℝ)

-- Given conditions
theorem volume_of_prism (h1 : a * b = 48) (h2 : b * c = 49) (h3 : a * c = 50) :
  a * b * c = 343 :=
by {
  sorry
}

end volume_of_prism_l6_6200


namespace soda_ratio_l6_6546

theorem soda_ratio (total_sodas diet_sodas regular_sodas : ℕ) (h1 : total_sodas = 64) (h2 : diet_sodas = 28) (h3 : regular_sodas = total_sodas - diet_sodas) : regular_sodas / Nat.gcd regular_sodas diet_sodas = 9 ∧ diet_sodas / Nat.gcd regular_sodas diet_sodas = 7 :=
by
  sorry

end soda_ratio_l6_6546


namespace min_value_of_quadratic_l6_6897

theorem min_value_of_quadratic (y1 y2 y3 : ℝ) (h1 : 0 < y1) (h2 : 0 < y2) (h3 : 0 < y3) (h_eq : 2 * y1 + 3 * y2 + 4 * y3 = 75) :
  y1^2 + 2 * y2^2 + 3 * y3^2 ≥ 5625 / 29 :=
sorry

end min_value_of_quadratic_l6_6897


namespace degree_of_expression_l6_6964

open Polynomial

noncomputable def expr1 : Polynomial ℤ := (monomial 5 3 - monomial 3 2 + 4) * (monomial 12 2 - monomial 8 1 + monomial 6 5 - 15)
noncomputable def expr2 : Polynomial ℤ := (monomial 3 2 - 4) ^ 6
noncomputable def final_expr : Polynomial ℤ := expr1 - expr2

theorem degree_of_expression : degree final_expr = 18 := by
  sorry

end degree_of_expression_l6_6964


namespace vendor_has_maaza_l6_6097

theorem vendor_has_maaza (liters_pepsi : ℕ) (liters_sprite : ℕ) (total_cans : ℕ) (gcd_pepsi_sprite : ℕ) (cans_pepsi : ℕ) (cans_sprite : ℕ) (cans_maaza : ℕ) (liters_per_can : ℕ) (total_liters_maaza : ℕ) :
  liters_pepsi = 144 →
  liters_sprite = 368 →
  total_cans = 133 →
  gcd_pepsi_sprite = Nat.gcd liters_pepsi liters_sprite →
  gcd_pepsi_sprite = 16 →
  cans_pepsi = liters_pepsi / gcd_pepsi_sprite →
  cans_sprite = liters_sprite / gcd_pepsi_sprite →
  cans_maaza = total_cans - (cans_pepsi + cans_sprite) →
  liters_per_can = gcd_pepsi_sprite →
  total_liters_maaza = cans_maaza * liters_per_can →
  total_liters_maaza = 1616 :=
by
  sorry

end vendor_has_maaza_l6_6097


namespace min_distance_circle_tangent_l6_6555

theorem min_distance_circle_tangent
  (P : ℝ × ℝ)
  (hP: 3 * P.1 + 4 * P.2 = 11) :
  ∃ d : ℝ, d = 11 / 5 := 
sorry

end min_distance_circle_tangent_l6_6555


namespace non_divisible_by_twenty_l6_6959

theorem non_divisible_by_twenty (k : ℤ) (h : ∃ m : ℤ, k * (k + 1) * (k + 2) = 5 * m) :
  ¬ (∃ l : ℤ, k * (k + 1) * (k + 2) = 20 * l) := sorry

end non_divisible_by_twenty_l6_6959


namespace suff_but_not_nec_l6_6375

def M (a : ℝ) : Prop := ∀ x : ℝ, x^2 + a * x + 1 > 0
def N (a : ℝ) : Prop := ∃ x : ℝ, (a - 3) * x + 1 = 0

theorem suff_but_not_nec (a : ℝ) : M a → N a ∧ ¬(N a → M a) := by
  sorry

end suff_but_not_nec_l6_6375


namespace actual_area_of_region_l6_6209

-- Problem Definitions
def map_scale : ℕ := 300000
def map_area_cm_squared : ℕ := 24

-- The actual area calculation should be 216 km²
theorem actual_area_of_region :
  let scale_factor_distance := map_scale
  let scale_factor_area := scale_factor_distance ^ 2
  let actual_area_cm_squared := map_area_cm_squared * scale_factor_area
  let actual_area_km_squared := actual_area_cm_squared / 10^10
  actual_area_km_squared = 216 := 
by
  sorry

end actual_area_of_region_l6_6209


namespace monthly_payment_l6_6204

noncomputable def house_price := 280
noncomputable def deposit := 40
noncomputable def mortgage_years := 10
noncomputable def months_per_year := 12

theorem monthly_payment (house_price deposit : ℕ) (mortgage_years months_per_year : ℕ) :
  (house_price - deposit) / mortgage_years / months_per_year = 2 :=
by
  sorry

end monthly_payment_l6_6204


namespace evaluate_expression_l6_6229

theorem evaluate_expression : (3^2 - 3) + (4^2 - 4) - (5^2 - 5) + (6^2 - 6) = 28 :=
by 
  sorry

end evaluate_expression_l6_6229


namespace max_subset_count_l6_6739

-- Define the problem conditions in Lean 4
def is_valid_subset (T : Finset ℕ) : Prop :=
  ∀ (a b : ℕ), a ∈ T → b ∈ T → a ≠ b → ¬ (a + b) % 5 = 0

theorem max_subset_count :
  ∃ (T : Finset ℕ), (is_valid_subset T) ∧ T.card = 18 := by
  sorry

end max_subset_count_l6_6739


namespace inner_hexagon_area_l6_6611

-- Define necessary conditions in Lean 4
variable (a b c d e f : ℕ)
variable (a1 a2 a3 a4 a5 a6 : ℕ)

-- Congruent equilateral triangles conditions forming a hexagon
axiom congruent_equilateral_triangles_overlap : 
  a1 = 1 ∧ a2 = 1 ∧ a3 = 9 ∧ a4 = 9 ∧ a5 = 16 ∧ a6 = 16

-- We want to show that the area of the inner hexagon is 38
theorem inner_hexagon_area : 
  a1 = 1 ∧ a2 = 1 ∧ a3 = 9 ∧ a4 = 9 ∧ a5 = 16 ∧ a6 = 16 → a = 38 :=
by
  intro h
  sorry

end inner_hexagon_area_l6_6611


namespace assumption_for_contradiction_l6_6624

theorem assumption_for_contradiction (a b : ℕ) (a_pos : 0 < a) (b_pos : 0 < b) (h : 5 ∣ a * b) : 
  ¬ (¬ (5 ∣ a) ∧ ¬ (5 ∣ b)) := 
sorry

end assumption_for_contradiction_l6_6624


namespace number_of_monomials_l6_6246

def isMonomial (expr : String) : Bool :=
  match expr with
  | "-(2 / 3) * a^3 * b" => true
  | "(x * y) / 2" => true
  | "-4" => true
  | "0" => true
  | _ => false

def countMonomials (expressions : List String) : Nat :=
  expressions.foldl (fun acc expr => if isMonomial expr then acc + 1 else acc) 0

theorem number_of_monomials : countMonomials ["-(2 / 3) * a^3 * b", "(x * y) / 2", "-4", "-(2 / a)", "0", "x - y"] = 4 :=
by
  sorry

end number_of_monomials_l6_6246


namespace initial_tickets_count_l6_6447

def spent_tickets : ℕ := 5
def additional_tickets : ℕ := 10
def current_tickets : ℕ := 16

theorem initial_tickets_count (initial_tickets : ℕ) :
  initial_tickets - spent_tickets + additional_tickets = current_tickets ↔ initial_tickets = 11 :=
by
  sorry

end initial_tickets_count_l6_6447


namespace inequality_holds_for_triangle_sides_l6_6774

theorem inequality_holds_for_triangle_sides (a : ℝ) : 
  (∀ (x y z : ℕ), x + y > z ∧ y + z > x ∧ z + x > y → (x^2 + y^2 + z^2 ≤ a * (x * y + y * z + z * x))) ↔ (1 ≤ a ∧ a ≤ 6 / 5) :=
by sorry

end inequality_holds_for_triangle_sides_l6_6774


namespace bulb_works_longer_than_4000_hours_l6_6782

noncomputable def P_X := 0.5
noncomputable def P_Y := 0.3
noncomputable def P_Z := 0.2

noncomputable def P_4000_given_X := 0.59
noncomputable def P_4000_given_Y := 0.65
noncomputable def P_4000_given_Z := 0.70

noncomputable def P_4000 := 
  P_X * P_4000_given_X + P_Y * P_4000_given_Y + P_Z * P_4000_given_Z

theorem bulb_works_longer_than_4000_hours : P_4000 = 0.63 :=
by
  sorry

end bulb_works_longer_than_4000_hours_l6_6782


namespace problem_equivalent_l6_6997

theorem problem_equivalent :
  ∀ m n : ℤ, |m - n| = n - m ∧ |m| = 4 ∧ |n| = 3 → m + n = -1 ∨ m + n = -7 :=
by
  intros m n h
  have h1 : |m - n| = n - m := h.1
  have h2 : |m| = 4 := h.2.1
  have h3 : |n| = 3 := h.2.2
  sorry

end problem_equivalent_l6_6997


namespace base_number_exponent_l6_6171

theorem base_number_exponent (x : ℝ) (h : ((x^4) * 3.456789) ^ 12 = y) (has_24_digits : true) : x = 10^12 :=
  sorry

end base_number_exponent_l6_6171


namespace blueberries_count_l6_6475

theorem blueberries_count (total_berries raspberries blackberries blueberries : ℕ)
  (h1 : total_berries = 42)
  (h2 : raspberries = total_berries / 2)
  (h3 : blackberries = total_berries / 3)
  (h4 : blueberries = total_berries - raspberries - blackberries) :
  blueberries = 7 :=
sorry

end blueberries_count_l6_6475


namespace index_card_area_l6_6557

theorem index_card_area 
  (a b : ℕ)
  (ha : a = 5)
  (hb : b = 7)
  (harea : (a - 2) * b = 21) :
  (a * (b - 2) = 25) :=
by
  sorry

end index_card_area_l6_6557


namespace evaluate_expression_l6_6664

theorem evaluate_expression (a b : ℤ) (h_a : a = 4) (h_b : b = -3) : -a - b^3 + a * b = 11 :=
by
  rw [h_a, h_b]
  sorry

end evaluate_expression_l6_6664


namespace symmetric_points_sum_l6_6155

theorem symmetric_points_sum
  (a b : ℝ)
  (h1 : a = -3)
  (h2 : b = 2) :
  a + b = -1 := by
  sorry

end symmetric_points_sum_l6_6155


namespace sahil_selling_price_l6_6215

-- Defining the conditions as variables
def purchase_price : ℕ := 14000
def repair_cost : ℕ := 5000
def transportation_charges : ℕ := 1000
def profit_percentage : ℕ := 50

-- Defining the total cost
def total_cost : ℕ := purchase_price + repair_cost + transportation_charges

-- Calculating the profit amount
def profit : ℕ := (profit_percentage * total_cost) / 100

-- Calculating the selling price
def selling_price : ℕ := total_cost + profit

-- The Lean statement to prove the selling price is Rs 30,000
theorem sahil_selling_price : selling_price = 30000 :=
by 
  simp [total_cost, profit, selling_price]
  sorry

end sahil_selling_price_l6_6215


namespace fourth_grade_planted_89_l6_6621

-- Define the number of trees planted by the fifth grade
def fifth_grade_trees : Nat := 114

-- Define the condition that the fifth grade planted twice as many trees as the third grade
def third_grade_trees : Nat := fifth_grade_trees / 2

-- Define the condition that the fourth grade planted 32 more trees than the third grade
def fourth_grade_trees : Nat := third_grade_trees + 32

-- Theorem to prove the number of trees planted by the fourth grade is 89
theorem fourth_grade_planted_89 : fourth_grade_trees = 89 := by
  sorry

end fourth_grade_planted_89_l6_6621


namespace ones_digit_of_73_pow_355_l6_6368

theorem ones_digit_of_73_pow_355 : (73 ^ 355) % 10 = 7 := 
  sorry

end ones_digit_of_73_pow_355_l6_6368


namespace advertisements_shown_l6_6216

theorem advertisements_shown (advertisement_duration : ℕ) (cost_per_minute : ℕ) (total_cost : ℕ) :
  advertisement_duration = 3 →
  cost_per_minute = 4000 →
  total_cost = 60000 →
  total_cost / (advertisement_duration * cost_per_minute) = 5 :=
by
  sorry

end advertisements_shown_l6_6216


namespace weight_of_piece_l6_6241

theorem weight_of_piece (x d : ℝ) (h1 : x - d = 300) (h2 : x + d = 500) : x = 400 := 
by
  sorry

end weight_of_piece_l6_6241


namespace integer_solution_pairs_l6_6308

theorem integer_solution_pairs (a b : ℕ) (h_pos : a > 0 ∧ b > 0):
  (∃ k : ℕ, k > 0 ∧ a^2 = k * (2 * a * b^2 - b^3 + 1)) ↔ 
  (∃ l : ℕ, l > 0 ∧ ((a = 2 * l ∧ b = 1) ∨ (a = l ∧ b = 2 * l) ∨ (a = 8 * l^4 - l ∧ b = 2 * l))) :=
sorry

end integer_solution_pairs_l6_6308


namespace exponent_solver_l6_6626

theorem exponent_solver (x : ℕ) : 3^x + 3^x + 3^x + 3^x = 19683 → x = 7 := sorry

end exponent_solver_l6_6626


namespace product_of_consecutive_integers_l6_6506

theorem product_of_consecutive_integers (n : ℕ) : 24 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
sorry

end product_of_consecutive_integers_l6_6506


namespace cost_price_per_meter_l6_6479

-- Definitions
def selling_price : ℝ := 9890
def meters_sold : ℕ := 92
def profit_per_meter : ℝ := 24

-- Theorem
theorem cost_price_per_meter : (selling_price - profit_per_meter * meters_sold) / meters_sold = 83.5 :=
by
  sorry

end cost_price_per_meter_l6_6479


namespace point_not_in_plane_l6_6000

def is_in_plane (p0 : ℝ × ℝ × ℝ) (n : ℝ × ℝ × ℝ) (p : ℝ × ℝ × ℝ) : Prop :=
  let (x0, y0, z0) := p0
  let (nx, ny, nz) := n
  let (x, y, z) := p
  (nx * (x - x0) + ny * (y - y0) + nz * (z - z0)) = 0

theorem point_not_in_plane :
  ¬ is_in_plane (1, 2, 3) (1, 1, 1) (-2, 5, 4) :=
by
  sorry

end point_not_in_plane_l6_6000


namespace bertha_gave_away_balls_l6_6732

def balls_initial := 2
def balls_worn_out := 20 / 10
def balls_lost := 20 / 5
def balls_purchased := (20 / 4) * 3
def balls_after_20_games_without_giveaway := balls_initial - balls_worn_out - balls_lost + balls_purchased
def balls_after_20_games := 10

theorem bertha_gave_away_balls : balls_after_20_games_without_giveaway - balls_after_20_games = 1 := by
  sorry

end bertha_gave_away_balls_l6_6732


namespace george_speed_l6_6900

theorem george_speed : 
  ∀ (d_tot d_1st : ℝ) (v_tot v_1st : ℝ) (v_2nd : ℝ),
    d_tot = 1 ∧ d_1st = 1 / 2 ∧ v_tot = 3 ∧ v_1st = 2 ∧ ((d_tot / v_tot) = (d_1st / v_1st + d_1st / v_2nd)) →
    v_2nd = 6 :=
by
  -- Proof here
  sorry

end george_speed_l6_6900


namespace sam_fish_count_l6_6765

/-- Let S be the number of fish Sam has. -/
def num_fish_sam : ℕ := sorry

/-- Joe has 8 times as many fish as Sam, which gives 8S fish. -/
def num_fish_joe (S : ℕ) : ℕ := 8 * S

/-- Harry has 4 times as many fish as Joe, hence 32S fish. -/
def num_fish_harry (S : ℕ) : ℕ := 32 * S

/-- Harry has 224 fish. -/
def harry_fish : ℕ := 224

/-- Prove that Sam has 7 fish given the conditions above. -/
theorem sam_fish_count : num_fish_harry num_fish_sam = harry_fish → num_fish_sam = 7 := by
  sorry

end sam_fish_count_l6_6765


namespace curves_intersect_at_four_points_l6_6938

theorem curves_intersect_at_four_points (a : ℝ) :
  (∃ x y : ℝ, (x^2 + y^2 = a^2 ∧ y = -x^2 + a ) ∧ 
   (0 = x ∧ y = a) ∧ 
   (∃ t : ℝ, x = t ∧ (y = 1 ∧ x^2 = a - 1))) ↔ a = 2 := 
by
  sorry

end curves_intersect_at_four_points_l6_6938


namespace frosting_cupcakes_in_10_minutes_l6_6763

def speed_Cagney := 1 / 20 -- Cagney frosts 1 cupcake every 20 seconds
def speed_Lacey := 1 / 30 -- Lacey frosts 1 cupcake every 30 seconds
def speed_Jamie := 1 / 15 -- Jamie frosts 1 cupcake every 15 seconds

def combined_speed := speed_Cagney + speed_Lacey + speed_Jamie -- Combined frosting rate (cupcakes per second)

def total_seconds := 10 * 60 -- 10 minutes converted to seconds

def number_of_cupcakes := combined_speed * total_seconds -- Total number of cupcakes frosted in 10 minutes

theorem frosting_cupcakes_in_10_minutes :
  number_of_cupcakes = 90 := by
  sorry

end frosting_cupcakes_in_10_minutes_l6_6763


namespace price_per_bottle_is_half_l6_6894

theorem price_per_bottle_is_half (P : ℚ) 
  (Remy_bottles_morning : ℕ) (Nick_bottles_morning : ℕ) 
  (Total_sales_evening : ℚ) (Evening_more : ℚ) : 
  Remy_bottles_morning = 55 → 
  Nick_bottles_morning = Remy_bottles_morning - 6 → 
  Total_sales_evening = 55 → 
  Evening_more = 3 → 
  104 * P + 3 = 55 → 
  P = 1 / 2 := 
by
  intros h_remy_55 h_nick_remy h_total_55 h_evening_3 h_sales_eq
  sorry

end price_per_bottle_is_half_l6_6894


namespace beginner_trigonometry_probability_l6_6808

def BC := ℝ
def AC := ℝ
def IC := ℝ
def BT := ℝ
def AT := ℝ
def IT := ℝ
def T := 5000

theorem beginner_trigonometry_probability :
  ∀ (BC AC IC BT AT IT : ℝ),
  (BC + AC + IC = 0.60 * T) →
  (BT + AT + IT = 0.40 * T) →
  (BC + BT = 0.45 * T) →
  (AC + AT = 0.35 * T) →
  (IC + IT = 0.20 * T) →
  (BC = 1.25 * BT) →
  (IC + AC = 1.20 * (IT + AT)) →
  (BT / T = 1/5) :=
by
  intros
  sorry

end beginner_trigonometry_probability_l6_6808


namespace john_share_l6_6752

theorem john_share
  (total_amount : ℝ)
  (john_ratio jose_ratio binoy_ratio : ℝ)
  (total_amount_eq : total_amount = 6000)
  (ratios_eq : john_ratio = 2 ∧ jose_ratio = 4 ∧ binoy_ratio = 6) :
  (john_ratio / (john_ratio + jose_ratio + binoy_ratio)) * total_amount = 1000 :=
by
  -- Here we would derive the proof, but just use sorry for the moment.
  sorry

end john_share_l6_6752


namespace initial_total_perimeter_l6_6323

theorem initial_total_perimeter (n : ℕ) (a : ℕ) (m : ℕ)
  (h1 : n = 2 * m)
  (h2 : 40 = 2 * a * m)
  (h3 : 4 * n - 6 * m = 4 * n - 40) :
  4 * n = 280 :=
by sorry

end initial_total_perimeter_l6_6323


namespace sum_seven_terms_l6_6746

-- Define the arithmetic sequence and sum of first n terms
variable {a : ℕ → ℝ} -- The arithmetic sequence a_n
variable {S : ℕ → ℝ} -- The sum of the first n terms S_n

-- Define the conditions
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def sum_of_arithmetic_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, S n = (n : ℝ) / 2 * (a 1 + a n)

-- Given condition: a_4 = 4
def a_4_eq_4 (a : ℕ → ℝ) : Prop :=
  a 4 = 4

-- Proposition we want to prove: S_7 = 28 given a_4 = 4
theorem sum_seven_terms (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (ha : is_arithmetic_sequence a)
  (hS : sum_of_arithmetic_sequence a S)
  (h : a_4_eq_4 a) : 
  S 7 = 28 := 
sorry

end sum_seven_terms_l6_6746


namespace cos_pi_minus_alpha_l6_6662

theorem cos_pi_minus_alpha (α : ℝ) (h1 : Real.sin α = 5 / 13) (h2 : 0 < α ∧ α < π / 2) :
  Real.cos (π - α) = -12 / 13 :=
sorry

end cos_pi_minus_alpha_l6_6662


namespace equation_of_line_containing_BC_l6_6016

theorem equation_of_line_containing_BC (A B C : ℝ × ℝ)
  (hA : A = (1, 2))
  (altitude_from_CA : ∀ x y : ℝ, 2 * x - 3 * y + 1 = 0)
  (altitude_from_BA : ∀ x y : ℝ, x + y = 1)
  (eq_BC : ∃ k b : ℝ, ∀ x y : ℝ, y = k * x + b) :
  ∃ k b : ℝ, (∀ x y : ℝ, y = k * x + b) ∧ 2 * x + 3 * y + 7 = 0 :=
sorry

end equation_of_line_containing_BC_l6_6016


namespace wholesale_cost_is_200_l6_6638

variable (W R E : ℝ)

def retail_price (W : ℝ) : ℝ := 1.20 * W

def employee_price (R : ℝ) : ℝ := 0.75 * R

-- Main theorem stating that given the retail and employee price formulas and the employee paid amount,
-- the wholesale cost W is equal to 200.
theorem wholesale_cost_is_200
  (hR : R = retail_price W)
  (hE : E = employee_price R)
  (heq : E = 180) :
  W = 200 :=
by
  sorry

end wholesale_cost_is_200_l6_6638


namespace find_b_plus_m_l6_6759

-- Definitions of the constants and functions based on the given conditions.
variables (m b : ℝ)

-- The first line equation passing through (5, 8).
def line1 := 8 = m * 5 + 3

-- The second line equation passing through (5, 8).
def line2 := 8 = 4 * 5 + b

-- The goal statement we need to prove.
theorem find_b_plus_m (h1 : line1 m) (h2 : line2 b) : b + m = -11 :=
sorry

end find_b_plus_m_l6_6759


namespace total_profit_l6_6510

-- Define the relevant variables and conditions
variables (x y : ℝ) -- Cost prices of the two music players

-- Given conditions
axiom cost_price_first : x * 1.2 = 132
axiom cost_price_second : y * 1.1 = 132

theorem total_profit : 132 + 132 - y - x = 34 :=
by
  -- The proof body is not required
  sorry

end total_profit_l6_6510


namespace perpendicular_os_bc_l6_6290

variable {A B C O S : Type}

noncomputable def acute_triangle (A B C : Type) := true -- Placeholder definition for acute triangle.

noncomputable def circumcenter (O : Type) (A B C : Type) := true -- Placeholder definition for circumcenter.

noncomputable def line_intersects_circumcircle_second_time (AC : Type) (circ : Type) (S : Type) := true -- Placeholder def.

-- Define the problem in Lean
theorem perpendicular_os_bc
  (ABC_is_acute : acute_triangle A B C)
  (O_is_circumcenter : circumcenter O A B C)
  (AC_intersects_AOB_circumcircle_at_S : line_intersects_circumcircle_second_time (A → C) (A → B → O) S) :
  true := -- Place for the proof that OS ⊥ BC
sorry

end perpendicular_os_bc_l6_6290


namespace find_y_l6_6443

-- Hypotheses
variable (x y : ℤ)

-- Given conditions
def condition1 : Prop := x = 4
def condition2 : Prop := x + y = 0

-- The goal is to prove y = -4 given the conditions
theorem find_y (h1 : condition1 x) (h2 : condition2 x y) : y = -4 := by
  sorry

end find_y_l6_6443


namespace endangered_animal_population_after_3_years_l6_6937

-- Given conditions and definitions
def population (m : ℕ) (n : ℕ) : ℝ := m * (0.90 ^ n)

theorem endangered_animal_population_after_3_years :
  population 8000 3 = 5832 :=
by
  sorry

end endangered_animal_population_after_3_years_l6_6937


namespace tyler_common_ratio_l6_6940

theorem tyler_common_ratio (a r : ℝ) 
  (h1 : a / (1 - r) = 10)
  (h2 : (a + 4) / (1 - r) = 15) : 
  r = 1 / 5 :=
by
  sorry

end tyler_common_ratio_l6_6940


namespace find_X_l6_6128

theorem find_X (X : ℕ) : 
  (∃ k : ℕ, X = 26 * k + k) ∧ (∃ m : ℕ, X = 29 * m + m) → (X = 270 ∨ X = 540) :=
by
  sorry

end find_X_l6_6128


namespace gumball_water_wednesday_l6_6583

theorem gumball_water_wednesday :
  ∀ (total_weekly_water monday_thursday_saturday_water tuesday_friday_sunday_water : ℕ),
  total_weekly_water = 60 →
  monday_thursday_saturday_water = 9 →
  tuesday_friday_sunday_water = 8 →
  total_weekly_water - (monday_thursday_saturday_water * 3 + tuesday_friday_sunday_water * 3) = 9 :=
by
  intros total_weekly_water monday_thursday_saturday_water tuesday_friday_sunday_water
  sorry

end gumball_water_wednesday_l6_6583


namespace ben_and_sara_tie_fraction_l6_6417

theorem ben_and_sara_tie_fraction (ben_wins sara_wins : ℚ) (h1 : ben_wins = 2 / 5) (h2 : sara_wins = 1 / 4) : 
  1 - (ben_wins + sara_wins) = 7 / 20 :=
by
  rw [h1, h2]
  norm_num

end ben_and_sara_tie_fraction_l6_6417


namespace unique_k_for_triangle_inequality_l6_6311

theorem unique_k_for_triangle_inequality (k : ℕ) (h : 0 < k) :
  (∀ (a b c : ℝ), 0 < a → 0 < b → 0 < c → k * (a * b + b * c + c * a) > 5 * (a * b + b * b + c * c) → a + b > c ∧ b + c > a ∧ c + a > b) ↔ (k = 6) :=
by
  sorry

end unique_k_for_triangle_inequality_l6_6311


namespace avg_salary_of_employees_is_1500_l6_6993

-- Definitions for conditions
def num_employees : ℕ := 20
def num_people_incl_manager : ℕ := 21
def manager_salary : ℝ := 4650
def salary_increase : ℝ := 150

-- Definition for average salary of employees excluding the manager
def avg_salary_employees (A : ℝ) : Prop :=
    21 * (A + salary_increase) = 20 * A + manager_salary

-- The target proof statement
theorem avg_salary_of_employees_is_1500 :
  ∃ A : ℝ, avg_salary_employees A ∧ A = 1500 := by
  -- Proof goes here
  sorry

end avg_salary_of_employees_is_1500_l6_6993


namespace average_age_after_leaves_is_27_l6_6540

def average_age_of_remaining_people (initial_avg_age : ℕ) (initial_people_count : ℕ) 
    (age_leave1 : ℕ) (age_leave2 : ℕ) (remaining_people_count : ℕ) : ℕ :=
  let initial_total_age := initial_avg_age * initial_people_count
  let new_total_age := initial_total_age - (age_leave1 + age_leave2)
  new_total_age / remaining_people_count

theorem average_age_after_leaves_is_27 :
  average_age_of_remaining_people 25 6 20 22 4 = 27 :=
by
  -- Proof is skipped
  sorry

end average_age_after_leaves_is_27_l6_6540


namespace determine_a_l6_6230

theorem determine_a
  (h : ∀ x : ℝ, x > 0 → (x - a + 2) * (x^2 - a * x - 2) ≥ 0) : 
  a = 1 :=
sorry

end determine_a_l6_6230


namespace triangle_angles_sum_l6_6001

theorem triangle_angles_sum (x : ℝ) (h : 40 + 3 * x + (x + 10) = 180) : x = 32.5 := by
  sorry

end triangle_angles_sum_l6_6001


namespace arithmetic_sequence_min_value_Sn_l6_6854

-- Define the sequence a_n and the sum S_n
variable (a : ℕ → ℝ) (S : ℕ → ℝ)

-- The given condition
axiom condition : ∀ n : ℕ, n > 0 → (2 * S n / n) + n = 2 * a n + 1

-- Arithmetic sequence proof
theorem arithmetic_sequence : ∀ n : ℕ, n > 0 → a (n + 1) = a n + 1 :=
by sorry

-- Minimum value of S_n when a_4, a_7, a_9 are geometric
theorem min_value_Sn (G : ℝ) (h : a 4 * a 9 = a 7 ^ 2) : ∃ n : ℕ, S n = -78 :=
by sorry

end arithmetic_sequence_min_value_Sn_l6_6854


namespace find_total_price_l6_6682

noncomputable def total_price (p : ℝ) : Prop := 0.20 * p = 240

theorem find_total_price (p : ℝ) (h : total_price p) : p = 1200 :=
by sorry

end find_total_price_l6_6682


namespace at_least_one_admitted_prob_l6_6405

theorem at_least_one_admitted_prob (pA pB : ℝ) (hA : pA = 0.6) (hB : pB = 0.7) (independent : ∀ (P Q : Prop), P ∧ Q → P ∧ Q):
  (1 - ((1 - pA) * (1 - pB))) = 0.88 :=
by
  rw [hA, hB]
  -- more steps would follow in a complete proof
  sorry

end at_least_one_admitted_prob_l6_6405


namespace ron_pay_cuts_l6_6397

-- Define percentages as decimals
def cut_1 : ℝ := 0.05
def cut_2 : ℝ := 0.10
def cut_3 : ℝ := 0.15
def overall_cut : ℝ := 0.27325

-- Define the total number of pay cuts
def total_pay_cuts : ℕ := 3

noncomputable def verify_pay_cuts (cut_1 cut_2 cut_3 overall_cut : ℝ) (total_pay_cuts : ℕ) : Prop :=
  (((1 - cut_1) * (1 - cut_2) * (1 - cut_3) = (1 - overall_cut)) ∧ (total_pay_cuts = 3))

theorem ron_pay_cuts 
  (cut_1 : ℝ := 0.05)
  (cut_2 : ℝ := 0.10)
  (cut_3 : ℝ := 0.15)
  (overall_cut : ℝ := 0.27325)
  (total_pay_cuts : ℕ := 3) 
  : verify_pay_cuts cut_1 cut_2 cut_3 overall_cut total_pay_cuts :=
by sorry

end ron_pay_cuts_l6_6397


namespace sectors_not_equal_l6_6988

theorem sectors_not_equal (a1 a2 a3 a4 a5 a6 : ℕ) :
  ¬(∃ k : ℕ, (∀ n : ℕ, n = k) ↔
    ∃ m, (a1 + m) = k ∧ (a2 + m) = k ∧ (a3 + m) = k ∧ 
         (a4 + m) = k ∧ (a5 + m) = k ∧ (a6 + m) = k) :=
sorry

end sectors_not_equal_l6_6988


namespace goose_eggs_hatching_l6_6138

theorem goose_eggs_hatching (x : ℝ) :
  (∃ n_hatched : ℝ, 3 * (2 * n_hatched / 20) = 110 ∧ x = n_hatched / 550) →
  x = 2 / 3 :=
by
  intro h
  sorry

end goose_eggs_hatching_l6_6138


namespace minimum_balls_l6_6504

/-- Given that tennis balls are stored in big boxes containing 25 balls each 
    and small boxes containing 20 balls each, and the least number of balls 
    that can be left unboxed is 5, prove that the least number of 
    freshly manufactured balls is 105.
-/
theorem minimum_balls (B S : ℕ) : 
  ∃ (n : ℕ), 25 * B + 20 * S = n ∧ n % 25 = 5 ∧ n % 20 = 5 ∧ n = 105 := 
sorry

end minimum_balls_l6_6504


namespace east_high_school_students_l6_6893

theorem east_high_school_students (S : ℝ) 
  (h1 : 0.52 * S * 0.125 = 26) :
  S = 400 :=
by
  -- The proof is omitted for this exercise
  sorry

end east_high_school_students_l6_6893


namespace river_width_l6_6339

def boat_width : ℕ := 3
def num_boats : ℕ := 8
def space_between_boats : ℕ := 2
def riverbank_space : ℕ := 2

theorem river_width : 
  let boat_space := num_boats * boat_width
  let between_boat_space := (num_boats - 1) * space_between_boats
  let riverbank_space_total := 2 * riverbank_space
  boat_space + between_boat_space + riverbank_space_total = 42 :=
by
  sorry

end river_width_l6_6339


namespace repeating_decimal_sum_l6_6812

theorem repeating_decimal_sum (x : ℚ) (hx : x = 0.417) :
  let num := 46
  let denom := 111
  let sum := num + denom
  sum = 157 :=
by
  sorry

end repeating_decimal_sum_l6_6812


namespace work_days_together_l6_6750

theorem work_days_together (A B : Type) (R_A R_B : ℝ) 
  (h1 : R_A = 1/2 * R_B) (h2 : R_B = 1 / 27) : 
  (1 / (R_A + R_B)) = 18 :=
by
  sorry

end work_days_together_l6_6750


namespace right_triangle_ratio_l6_6523

noncomputable def hypotenuse (a b : ℝ) : ℝ := Real.sqrt (a^2 + b^2)

noncomputable def radius_circumscribed_circle (hypo : ℝ) : ℝ := hypo / 2

noncomputable def radius_inscribed_circle (a b hypo : ℝ) : ℝ := (a + b - hypo) / 2

theorem right_triangle_ratio 
  (a b : ℝ) 
  (ha : a = 6) 
  (hb : b = 8) 
  (right_angle : a^2 + b^2 = hypotenuse a b ^ 2) :
  radius_inscribed_circle a b (hypotenuse a b) / radius_circumscribed_circle (hypotenuse a b) = 2 / 5 :=
by {
  -- The proof to be filled in
  sorry
}

end right_triangle_ratio_l6_6523


namespace socorro_training_days_l6_6969

def total_hours := 5
def minutes_per_hour := 60
def total_training_minutes := total_hours * minutes_per_hour

def minutes_multiplication_per_day := 10
def minutes_division_per_day := 20
def daily_training_minutes := minutes_multiplication_per_day + minutes_division_per_day

theorem socorro_training_days:
  total_training_minutes / daily_training_minutes = 10 :=
by
  -- proof omitted
  sorry

end socorro_training_days_l6_6969


namespace fractions_are_integers_l6_6096

theorem fractions_are_integers (x y : ℕ) 
    (h : ∃ k : ℤ, (x^2 - 1) / (y + 1) + (y^2 - 1) / (x + 1) = k) :
    ∃ u v : ℤ, (x^2 - 1) = u * (y + 1) ∧ (y^2 - 1) = v * (x + 1) := 
by
  sorry

end fractions_are_integers_l6_6096


namespace max_product_min_quotient_l6_6451

theorem max_product_min_quotient :
  let nums := [-5, -3, -1, 2, 4]
  let a := max (max (-5 * -3) (-5 * -1)) (max (-3 * -1) (max (2 * 4) (max (2 * -1) (4 * -1))))
  let b := min (min (4 / -1) (2 / -3)) (min (2 / -5) (min (4 / -3) (-5 / -3)))
  a = 15 ∧ b = -4 → a / b = -15 / 4 :=
by
  sorry

end max_product_min_quotient_l6_6451


namespace arithmetic_sequence_S12_l6_6025

theorem arithmetic_sequence_S12 (S : ℕ → ℝ) (a : ℕ → ℝ) 
  (h1 : ∀ n : ℕ, S n = n * (a 1 + a n) / 2)
  (hS4 : S 4 = 25) (hS8 : S 8 = 100) : S 12 = 225 :=
by
  sorry

end arithmetic_sequence_S12_l6_6025


namespace swimming_speed_in_still_water_l6_6585

theorem swimming_speed_in_still_water (v : ℝ) (current_speed : ℝ) (time : ℝ) (distance : ℝ) (effective_speed : current_speed = 10) (time_to_return : time = 6) (distance_to_return : distance = 12) (speed_eq : v - current_speed = distance / time) : v = 12 :=
by
  sorry

end swimming_speed_in_still_water_l6_6585


namespace unique_pair_exists_for_each_n_l6_6073

theorem unique_pair_exists_for_each_n (n : ℕ) (h : n > 0) : 
  ∃! (a b : ℕ), a > 0 ∧ b > 0 ∧ n = (a + b - 1) * (a + b - 2) / 2 + a :=
sorry

end unique_pair_exists_for_each_n_l6_6073


namespace Matilda_fathers_chocolate_bars_l6_6268

theorem Matilda_fathers_chocolate_bars
  (total_chocolates : ℕ) (sisters : ℕ) (chocolates_each : ℕ) (given_to_father_each : ℕ) 
  (chocolates_given : ℕ) (given_to_mother : ℕ) (eaten_by_father : ℕ) : 
  total_chocolates = 20 → 
  sisters = 4 → 
  chocolates_each = total_chocolates / (sisters + 1) → 
  given_to_father_each = chocolates_each / 2 → 
  chocolates_given = (sisters + 1) * given_to_father_each → 
  given_to_mother = 3 → 
  eaten_by_father = 2 → 
  chocolates_given - given_to_mother - eaten_by_father = 5 :=
by
  intros h_total h_sisters h_chocolates_each h_given_to_father_each h_chocolates_given h_given_to_mother h_eaten_by_father
  sorry

end Matilda_fathers_chocolate_bars_l6_6268


namespace distribute_balls_in_boxes_l6_6892

theorem distribute_balls_in_boxes :
  let balls := 5
  let boxes := 4
  (4 ^ balls) = 1024 :=
by
  sorry

end distribute_balls_in_boxes_l6_6892


namespace simplify_expression_l6_6320

-- Definitions of intermediate calculations
def a : ℤ := 3 + 5 + 6 - 2
def b : ℚ := a * 2 / 4
def c : ℤ := 3 * 4 + 6 - 4
def d : ℚ := c / 3

-- The statement to be proved
theorem simplify_expression : b + d = 32 / 3 := by
  sorry

end simplify_expression_l6_6320


namespace triangle_side_length_l6_6599

-- Defining basic properties and known lengths of the similar triangles
def GH : ℝ := 8
def HI : ℝ := 16
def YZ : ℝ := 24
def XY : ℝ := 12

-- Defining the similarity condition for triangles GHI and XYZ
def triangles_similar : Prop := 
  -- The similarity of the triangles implies proportionality of the sides
  (XY / GH = YZ / HI)

-- The theorem statement to prove
theorem triangle_side_length (h_sim : triangles_similar) : XY = 12 :=
by
  -- assuming the similarity condition and known lengths
  sorry -- This will be the detailed proof

end triangle_side_length_l6_6599


namespace equal_distribution_arithmetic_seq_l6_6761

theorem equal_distribution_arithmetic_seq :
  ∃ (a1 d : ℚ), (a1 + (a1 + d) = (a1 + 2 * d) + (a1 + 3 * d) + (a1 + 4 * d)) ∧ 
                (5 * a1 + 10 / 2 * d = 5) ∧ 
                (a1 = 4 / 3) :=
by
  sorry

end equal_distribution_arithmetic_seq_l6_6761


namespace value_of_a_l6_6581

theorem value_of_a (x a : ℝ) (h1 : 0 < x) (h2 : x < 1 / a) (h3 : ∀ x, x * (1 - a * x) ≤ 1 / 12) : a = 3 :=
sorry

end value_of_a_l6_6581


namespace fraction_arithmetic_proof_l6_6668

theorem fraction_arithmetic_proof :
  (7 / 6) + (5 / 4) - (3 / 2) = 11 / 12 :=
by sorry

end fraction_arithmetic_proof_l6_6668


namespace cost_of_20_pounds_of_bananas_l6_6041

noncomputable def cost_of_bananas (rate : ℝ) (amount : ℝ) : ℝ :=
rate * amount / 4

theorem cost_of_20_pounds_of_bananas :
  cost_of_bananas 6 20 = 30 :=
by
  sorry

end cost_of_20_pounds_of_bananas_l6_6041


namespace arc_length_150_deg_max_area_sector_l6_6620

noncomputable def alpha := 150 * (Real.pi / 180)
noncomputable def r := 6
noncomputable def perimeter := 24

-- 1. Proving the arc length when α = 150° and r = 6
theorem arc_length_150_deg : alpha * r = 5 * Real.pi := by
  sorry

-- 2. Proving the maximum area and corresponding alpha given the perimeter of 24
theorem max_area_sector : ∃ (α : ℝ), α = 2 ∧ (1 / 2) * ((perimeter - 2 * r) * r) = 36 := by
  sorry

end arc_length_150_deg_max_area_sector_l6_6620


namespace lines_parallel_a_eq_sqrt2_l6_6166

theorem lines_parallel_a_eq_sqrt2 (a : ℝ) (h1 : 1 ≠ 0) :
  (∀ a ≠ 0, ((- (1 / (2 * a))) = (- a / 2)) → a = Real.sqrt 2) :=
by
  sorry

end lines_parallel_a_eq_sqrt2_l6_6166


namespace find_sum_fusion_number_l6_6573

def sum_fusion_number (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 4 * (2 * k + 1)

theorem find_sum_fusion_number (n : ℕ) :
  n = 2020 ↔ sum_fusion_number n :=
sorry

end find_sum_fusion_number_l6_6573


namespace equivalence_condition_l6_6106

universe u

variables {U : Type u} (A B : Set U)

theorem equivalence_condition :
  (∃ (C : Set U), A ⊆ C ∧ B ⊆ Cᶜ) ↔ (A ∩ B = ∅) :=
sorry

end equivalence_condition_l6_6106


namespace number_division_l6_6391

theorem number_division (x : ℕ) (h : x / 5 = 80 + x / 6) : x = 2400 :=
sorry

end number_division_l6_6391


namespace coaches_meet_together_l6_6035

theorem coaches_meet_together (e s n a : ℕ)
  (h₁ : e = 5) (h₂ : s = 3) (h₃ : n = 9) (h₄ : a = 8) :
  Nat.lcm (Nat.lcm e s) (Nat.lcm n a) = 360 :=
by
  sorry

end coaches_meet_together_l6_6035


namespace average_speed_round_trip_l6_6267

theorem average_speed_round_trip (d : ℝ) (h_d_pos : d > 0) : 
  let t1 := d / 80
  let t2 := d / 120
  let d_total := 2 * d
  let t_total := t1 + t2
  let v_avg := d_total / t_total
  v_avg = 96 :=
by
  sorry

end average_speed_round_trip_l6_6267


namespace no_infinite_positive_integer_sequence_l6_6973

theorem no_infinite_positive_integer_sequence (a : ℕ → ℕ) :
  ¬(∀ n, a (n - 1) ^ 2 ≥ 2 * a n * a (n + 2)) :=
sorry

end no_infinite_positive_integer_sequence_l6_6973


namespace find_N_l6_6427

-- Definitions based on conditions from the problem
def remainder := 6
def dividend := 86
def divisor (Q : ℕ) := 5 * Q
def number_added_to_thrice_remainder (N : ℕ) := 3 * remainder + N
def quotient (Q : ℕ) := Q

-- The condition that relates dividend, divisor, quotient, and remainder
noncomputable def division_equation (Q : ℕ) := dividend = divisor Q * Q + remainder

-- Now, prove the condition
theorem find_N : ∃ N Q : ℕ, division_equation Q ∧ divisor Q = number_added_to_thrice_remainder N ∧ N = 2 :=
by
  sorry

end find_N_l6_6427


namespace perpendicular_OP_CD_l6_6036

variables {Point : Type}

-- Definitions of all the points involved
variables (A B C D P O : Point)
-- Definitions for distances / lengths
variables (dist : Point → Point → ℝ)
-- Definitions for relationships
variables (circumcenter : Point → Point → Point → Point)
variables (perpendicular : Point → Point → Point → Point → Prop)

-- Segment meet condition
variables (meet_at : Point → Point → Point → Prop)

-- Assuming the given conditions
theorem perpendicular_OP_CD 
  (meet : meet_at A C P)
  (meet' : meet_at B D P)
  (h1 : dist P A = dist P D)
  (h2 : dist P B = dist P C)
  (hO : circumcenter P A B = O) :
  perpendicular O P C D :=
sorry

end perpendicular_OP_CD_l6_6036


namespace fractional_part_frustum_l6_6544

noncomputable def base_edge : ℝ := 24
noncomputable def original_altitude : ℝ := 18
noncomputable def smaller_altitude : ℝ := original_altitude / 3

noncomputable def volume_pyramid (base_edge : ℝ) (altitude : ℝ) : ℝ :=
  (1 / 3) * (base_edge ^ 2) * altitude

noncomputable def volume_original : ℝ := volume_pyramid base_edge original_altitude
noncomputable def similarity_ratio : ℝ := (smaller_altitude / original_altitude) ^ 3
noncomputable def volume_smaller : ℝ := similarity_ratio * volume_original
noncomputable def volume_frustum : ℝ := volume_original - volume_smaller

noncomputable def fractional_volume_frustum : ℝ := volume_frustum / volume_original

theorem fractional_part_frustum : fractional_volume_frustum = 26 / 27 := by
  sorry

end fractional_part_frustum_l6_6544


namespace tangent_line_equation_l6_6471

-- Define the function
def f (x : ℝ) : ℝ := x^2

-- Define the point of tangency
def x0 : ℝ := 2

-- Define the value of function at the point of tangency
def y0 : ℝ := f x0

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 2 * x

-- State the theorem
theorem tangent_line_equation : ∃ (m b : ℝ), m = f' x0 ∧ b = y0 - m * x0 ∧ ∀ x, (y = m * x + b) ↔ (x = 2 → y = f x - f' x0 * (x - 2)) :=
by
  sorry

end tangent_line_equation_l6_6471


namespace percentage_error_in_side_l6_6860

theorem percentage_error_in_side
  (s s' : ℝ) -- the actual and measured side lengths
  (h : (s' * s' - s * s) / (s * s) * 100 = 41.61) : 
  ((s' - s) / s) * 100 = 19 :=
sorry

end percentage_error_in_side_l6_6860


namespace smallest_a_l6_6248

theorem smallest_a (a x : ℤ) (hx : x^2 + a * x = 30) (ha_pos : a > 0) (product_gt_30 : ∃ x₁ x₂ : ℤ, x₁ * x₂ = 30 ∧ x₁ + x₂ = -a ∧ x₁ * x₂ > 30) : a = 11 :=
sorry

end smallest_a_l6_6248


namespace profit_correct_A_B_l6_6190

noncomputable def profit_per_tire_A (batch_cost_A1 batch_cost_A2 cost_per_tire_A1 cost_per_tire_A2 sell_price_tire_A1 sell_price_tire_A2 produced_A : ℕ) : ℚ :=
  let cost_first_5000 := batch_cost_A1 + (cost_per_tire_A1 * 5000)
  let revenue_first_5000 := sell_price_tire_A1 * 5000
  let profit_first_5000 := revenue_first_5000 - cost_first_5000
  let cost_remaining := batch_cost_A2 + (cost_per_tire_A2 * (produced_A - 5000))
  let revenue_remaining := sell_price_tire_A2 * (produced_A - 5000)
  let profit_remaining := revenue_remaining - cost_remaining
  let total_profit := profit_first_5000 + profit_remaining
  total_profit / produced_A

noncomputable def profit_per_tire_B (batch_cost_B cost_per_tire_B sell_price_tire_B produced_B : ℕ) : ℚ :=
  let cost := batch_cost_B + (cost_per_tire_B * produced_B)
  let revenue := sell_price_tire_B * produced_B
  let profit := revenue - cost
  profit / produced_B

theorem profit_correct_A_B
  (batch_cost_A1 : ℕ := 22500) 
  (batch_cost_A2 : ℕ := 20000) 
  (cost_per_tire_A1 : ℕ := 8) 
  (cost_per_tire_A2 : ℕ := 6) 
  (sell_price_tire_A1 : ℕ := 20) 
  (sell_price_tire_A2 : ℕ := 18) 
  (produced_A : ℕ := 15000)
  (batch_cost_B : ℕ := 24000) 
  (cost_per_tire_B : ℕ := 7) 
  (sell_price_tire_B : ℕ := 19) 
  (produced_B : ℕ := 10000) :
  profit_per_tire_A batch_cost_A1 batch_cost_A2 cost_per_tire_A1 cost_per_tire_A2 sell_price_tire_A1 sell_price_tire_A2 produced_A = 9.17 ∧
  profit_per_tire_B batch_cost_B cost_per_tire_B sell_price_tire_B produced_B = 9.60 :=
by
  sorry

end profit_correct_A_B_l6_6190


namespace solve_pears_and_fruits_l6_6207

noncomputable def pears_and_fruits_problem : Prop :=
  ∃ (x y : ℕ), x + y = 1000 ∧ (11 * x) * (1/9 : ℚ) + (4 * y) * (1/7 : ℚ) = 999

theorem solve_pears_and_fruits :
  pears_and_fruits_problem := by
  sorry

end solve_pears_and_fruits_l6_6207


namespace arithmetic_geom_seq_a1_over_d_l6_6113

theorem arithmetic_geom_seq_a1_over_d (a1 a2 a3 a4 d : ℝ) (hne : d ≠ 0)
  (hgeom1 : (a1 + 2*d)^2 = a1 * (a1 + 3*d))
  (hgeom2 : (a1 + d)^2 = a1 * (a1 + 3*d)) :
  (a1 / d = -4) ∨ (a1 / d = 1) :=
by
  sorry

end arithmetic_geom_seq_a1_over_d_l6_6113


namespace quadratic_roots_eq_k_quadratic_inequality_k_range_l6_6835

theorem quadratic_roots_eq_k (k : ℝ) (h1 : k > 0)
  (h2 : ∀ x : ℝ, 2 < x → x < 3 → k*x^2 - 2*x + 6*k < 0)
  (h3: (2 + 3) = (2/k)) : k = 2/5 :=
by sorry

theorem quadratic_inequality_k_range (k : ℝ) 
  (h1 : k > 0)
  (h2 : ∀ x : ℝ, 2 < x → x < 3 → k*x^2 - 2*x + 6*k < 0) 
: 0 < k ∧ k <= 2/5 :=
by sorry

end quadratic_roots_eq_k_quadratic_inequality_k_range_l6_6835


namespace necessary_condition_for_x_greater_than_2_l6_6089

-- Define the real number x
variable (x : ℝ)

-- The proof statement
theorem necessary_condition_for_x_greater_than_2 : (x > 2) → (x > 1) :=
by sorry

end necessary_condition_for_x_greater_than_2_l6_6089


namespace digit_864_div_5_appending_zero_possibilities_l6_6219

theorem digit_864_div_5_appending_zero_possibilities :
  ∀ X : ℕ, (X * 1000 + 864) % 5 ≠ 0 :=
by sorry

end digit_864_div_5_appending_zero_possibilities_l6_6219


namespace max_possible_percentage_l6_6349

theorem max_possible_percentage (p_wi : ℝ) (p_fs : ℝ) (h_wi : p_wi = 0.4) (h_fs : p_fs = 0.7) :
  ∃ p_both : ℝ, p_both = min p_wi p_fs ∧ p_both = 0.4 :=
by
  sorry

end max_possible_percentage_l6_6349


namespace find_largest_N_l6_6577

noncomputable def largest_N : ℕ :=
  by
    -- This proof needs to demonstrate the solution based on constraints.
    -- Proof will be filled here.
    sorry

theorem find_largest_N :
  largest_N = 44 := 
  by
    -- Proof to establish the largest N will be completed here.
    sorry

end find_largest_N_l6_6577


namespace solve_for_x_l6_6428

def f (x : ℝ) : ℝ := 3 * x - 5

theorem solve_for_x : ∃ x, 2 * f x - 16 = f (x - 6) ∧ x = 1 := by
  exists 1
  sorry

end solve_for_x_l6_6428


namespace quadratic_roots_l6_6042

theorem quadratic_roots (m : ℝ) : 
  (m > 0 → ∃ a b : ℝ, a ≠ b ∧ (a^2 + a - 2 = m) ∧ (b^2 + b - 2 = m)) ∧ 
  ¬(m = 0 ∧ ∃ a : ℝ, (a^2 + a - 2 = m) ∧ (a^2 + a - 2 = m)) ∧ 
  ¬(m < 0 ∧ ¬ ∃ a b : ℝ, (a^2 + a - 2 = m) ∧ (b^2 + b - 2 = m) ) ∧ 
  ¬(∀ m, ∃ a : ℝ, (a^2 + a - 2 = m)) :=
by 
  sorry

end quadratic_roots_l6_6042


namespace remainder_sum_l6_6484

theorem remainder_sum (x y z : ℕ) (h1 : x % 15 = 11) (h2 : y % 15 = 13) (h3 : z % 15 = 9) :
  ((2 * (x % 15) + (y % 15) + (z % 15)) % 15) = 14 :=
by
  sorry

end remainder_sum_l6_6484


namespace boat_downstream_distance_l6_6898

theorem boat_downstream_distance (V_b V_s : ℝ) (t_downstream t_upstream : ℝ) (d_upstream : ℝ) 
  (h1 : t_downstream = 8) (h2 : t_upstream = 15) (h3 : d_upstream = 75) (h4 : V_s = 3.75) 
  (h5 : V_b - V_s = (d_upstream / t_upstream)) : (V_b + V_s) * t_downstream = 100 :=
by
  sorry

end boat_downstream_distance_l6_6898


namespace sequence_value_l6_6002

theorem sequence_value (a : ℕ → ℤ) (h1 : ∀ p q : ℕ, 0 < p → 0 < q → a (p + q) = a p + a q)
                       (h2 : a 2 = -6) : a 10 = -30 :=
by
  sorry

end sequence_value_l6_6002


namespace compare_answers_l6_6679

def num : ℕ := 384
def correct_answer : ℕ := (5 * num) / 16
def students_answer : ℕ := (5 * num) / 6
def difference : ℕ := students_answer - correct_answer

theorem compare_answers : difference = 200 := 
by
  sorry

end compare_answers_l6_6679


namespace paving_stones_needed_l6_6464

-- Definition for the dimensions of the paving stone and the courtyard
def paving_stone_length : ℝ := 2.5
def paving_stone_width : ℝ := 2
def courtyard_length : ℝ := 30
def courtyard_width : ℝ := 16.5

-- Compute areas
def paving_stone_area : ℝ := paving_stone_length * paving_stone_width
def courtyard_area : ℝ := courtyard_length * courtyard_width

-- The theorem to prove that the number of paving stones needed is 99
theorem paving_stones_needed :
  (courtyard_area / paving_stone_area) = 99 :=
by
  sorry

end paving_stones_needed_l6_6464


namespace eq_30_apples_n_7_babies_min_3_max_6_l6_6340

theorem eq_30_apples_n_7_babies_min_3_max_6 (x : ℕ) 
    (h1 : 30 = x + 7 * 4)
    (h2 : 21 ≤ 30) 
    (h3 : 30 ≤ 42) 
    (h4 : x = 2) :
  x = 2 :=
by
  sorry

end eq_30_apples_n_7_babies_min_3_max_6_l6_6340


namespace muffin_banana_ratio_l6_6374

variable {R : Type} [LinearOrderedField R]

-- Define the costs of muffins and bananas
variables {m b : R}

-- Susie's cost
def susie_cost (m b : R) := 4 * m + 5 * b

-- Calvin's cost for three times Susie's items
def calvin_cost_tripled (m b : R) := 12 * m + 15 * b

-- Calvin's actual cost
def calvin_cost_actual (m b : R) := 2 * m + 12 * b

theorem muffin_banana_ratio (m b : R) (h : calvin_cost_tripled m b = calvin_cost_actual m b) : m = (3 / 10) * b :=
by sorry

end muffin_banana_ratio_l6_6374


namespace smallest_digit_not_found_in_units_place_of_odd_number_l6_6015

def odd_digits : Set ℕ := {1, 3, 5, 7, 9}
def even_digits : Set ℕ := {0, 2, 4, 6, 8}
def smallest_digit_not_in_odd_digits : ℕ := 0

theorem smallest_digit_not_found_in_units_place_of_odd_number : smallest_digit_not_in_odd_digits ∈ even_digits ∧ smallest_digit_not_in_odd_digits ∉ odd_digits :=
by
  -- Step by step proof would start here (but is omitted with sorry)
  sorry

end smallest_digit_not_found_in_units_place_of_odd_number_l6_6015


namespace problem_statement_l6_6927

-- Define complex number i
noncomputable def i : ℂ := Complex.I

-- Define x as per the problem statement
noncomputable def x : ℂ := (1 + i * Real.sqrt 3) / 2

-- The main proposition to prove
theorem problem_statement : (1 / (x^2 - x)) = -1 :=
  sorry

end problem_statement_l6_6927


namespace cyclist_speed_l6_6815

theorem cyclist_speed
  (V : ℝ)
  (H1 : ∃ t_p : ℝ, V * t_p = 96 ∧ t_p = (96 / (V - 1)) - 2)
  (H2 : V > 1.25 * (V - 1)) :
  V = 16 :=
by
  sorry

end cyclist_speed_l6_6815


namespace seashells_second_day_l6_6237

theorem seashells_second_day (x : ℕ) (h1 : 5 + x + 2 * (5 + x) = 36) : x = 7 :=
by
  sorry

end seashells_second_day_l6_6237


namespace election_winner_percentage_l6_6550

theorem election_winner_percentage :
    let votes_candidate1 := 2500
    let votes_candidate2 := 5000
    let votes_candidate3 := 15000
    let total_votes := votes_candidate1 + votes_candidate2 + votes_candidate3
    let winning_votes := votes_candidate3
    (winning_votes / total_votes) * 100 = 75 := 
by 
    sorry

end election_winner_percentage_l6_6550


namespace unique_solution_l6_6099

noncomputable def func_prop (f : ℝ → ℝ) : Prop :=
  (∀ x ≥ 1, f x ≤ 2 * (x + 1)) ∧
  (∀ x ≥ 1, f (x + 1) = (f x)^2 / x - 1 / x)

theorem unique_solution (f : ℝ → ℝ) :
  func_prop f → ∀ x ≥ 1, f x = x + 1 :=
by
  sorry

end unique_solution_l6_6099


namespace evaluate_expression_l6_6078

theorem evaluate_expression (a b : ℕ) (h1 : a = 3) (h2 : b = 2) : (a^b)^b + (b^a)^a = 593 := by
  sorry

end evaluate_expression_l6_6078


namespace simplify_sqrt_20_minus_sqrt_5_plus_sqrt_one_fifth_simplify_fraction_of_sqrt_12_plus_sqrt_18_minus_sqrt_half_times_sqrt_3_l6_6864

-- Problem (1)
theorem simplify_sqrt_20_minus_sqrt_5_plus_sqrt_one_fifth :
  (Real.sqrt 20 - Real.sqrt 5 + Real.sqrt (1 / 5) = 6 * Real.sqrt 5 / 5) :=
by
  sorry

-- Problem (2)
theorem simplify_fraction_of_sqrt_12_plus_sqrt_18_minus_sqrt_half_times_sqrt_3 :
  (Real.sqrt 12 + Real.sqrt 18) / Real.sqrt 3 - 2 * Real.sqrt (1 / 2) * Real.sqrt 3 = 2 :=
by
  sorry

end simplify_sqrt_20_minus_sqrt_5_plus_sqrt_one_fifth_simplify_fraction_of_sqrt_12_plus_sqrt_18_minus_sqrt_half_times_sqrt_3_l6_6864


namespace vincent_correct_answer_l6_6364

theorem vincent_correct_answer (y : ℕ) (h : (y - 7) / 5 = 23) : (y - 5) / 7 = 17 :=
by
  sorry

end vincent_correct_answer_l6_6364


namespace half_angle_quadrants_l6_6157

theorem half_angle_quadrants (α : ℝ) (k : ℤ) (hα : ∃ k : ℤ, (π/2 + k * 2 * π < α ∧ α < π + k * 2 * π)) : 
  ∃ k : ℤ, (π/4 + k * π < α/2 ∧ α/2 < π/2 + k * π) := 
sorry

end half_angle_quadrants_l6_6157


namespace part1_part2_l6_6901

-- Part (1) prove maximum value of 4 - 2x - 1/x when x > 0 is 0.
theorem part1 (x : ℝ) (h : 0 < x) : 
  4 - 2 * x - (2 / x) ≤ 0 :=
sorry

-- Part (2) prove minimum value of 1/a + 1/b when a + 2b = 1 and a > 0, b > 0 is 3 + 2 * sqrt 2.
theorem part2 (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + 2 * b = 1) :
  (1 / a) + (1 / b) ≥ 3 + 2 * Real.sqrt 2 :=
sorry

end part1_part2_l6_6901


namespace locus_of_points_l6_6356

def point := (ℝ × ℝ)

variables (F_1 F_2 : point) (r k : ℝ)

def distance (P Q : point) : ℝ :=
  ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)^(1/2)

def on_circle (P : point) (center : point) (radius : ℝ) : Prop :=
  distance P center = radius

theorem locus_of_points
  (P : point)
  (r1 r2 PF1 PF2 : ℝ)
  (h_pF1 : r1 = distance P F_1)
  (h_pF2 : PF2 = distance P F_2)
  (h_outside_circle : PF2 = r2 + r)
  (h_inside_circle : PF2 = r - r2)
  (h_k : r1 + PF2 = k) :
  (∀ P, distance P F_1 + distance P F_2 = k →
  ( ∃ e_ellipse : Prop, on_circle P F_2 r → e_ellipse) ∨ 
  ( ∃ h_hyperbola : Prop, on_circle P F_2 r → h_hyperbola)) :=
by
  sorry

end locus_of_points_l6_6356


namespace arrangement_of_athletes_l6_6762

def num_arrangements (n : ℕ) (available_tracks_for_A : ℕ) (permutations_remaining : ℕ) : ℕ :=
  n * available_tracks_for_A * permutations_remaining

theorem arrangement_of_athletes :
  num_arrangements 2 3 24 = 144 :=
by
  sorry

end arrangement_of_athletes_l6_6762


namespace ratio_netbooks_is_one_third_l6_6125

open Nat

def total_computers (total : ℕ) : Prop := total = 72
def laptops_sold (laptops : ℕ) (total : ℕ) : Prop := laptops = total / 2
def desktops_sold (desktops : ℕ) : Prop := desktops = 12
def netbooks_sold (netbooks : ℕ) (total laptops desktops : ℕ) : Prop :=
  netbooks = total - (laptops + desktops)
def ratio_netbooks_total (netbooks total : ℕ) : Prop :=
  netbooks * 3 = total

theorem ratio_netbooks_is_one_third
  (total laptops desktops netbooks : ℕ)
  (h_total : total_computers total)
  (h_laptops : laptops_sold laptops total)
  (h_desktops : desktops_sold desktops)
  (h_netbooks : netbooks_sold netbooks total laptops desktops) :
  ratio_netbooks_total netbooks total :=
by
  sorry

end ratio_netbooks_is_one_third_l6_6125


namespace bananas_bought_l6_6407

def cost_per_banana : ℝ := 5.00
def total_cost : ℝ := 20.00

theorem bananas_bought : total_cost / cost_per_banana = 4 :=
by {
   sorry
}

end bananas_bought_l6_6407


namespace candy_totals_l6_6588

-- Definitions of the conditions
def sandra_bags := 2
def sandra_pieces_per_bag := 6

def roger_bags1 := 11
def roger_bags2 := 3

def emily_bags1 := 4
def emily_bags2 := 7
def emily_bags3 := 5

-- Definitions of total pieces of candy
def sandra_total_candy := sandra_bags * sandra_pieces_per_bag
def roger_total_candy := roger_bags1 + roger_bags2
def emily_total_candy := emily_bags1 + emily_bags2 + emily_bags3

-- The proof statement
theorem candy_totals :
  sandra_total_candy = 12 ∧ roger_total_candy = 14 ∧ emily_total_candy = 16 :=
by
  -- Here we would provide the proof but we'll use sorry to skip it
  sorry

end candy_totals_l6_6588


namespace marble_count_l6_6912

theorem marble_count (x : ℕ) 
  (h1 : ∀ (Liam Mia Noah Olivia: ℕ), Mia = 3 * Liam ∧ Noah = 4 * Mia ∧ Olivia = 2 * Noah)
  (h2 : Liam + Mia + Noah + Olivia = 156)
  : x = 4 :=
by sorry

end marble_count_l6_6912


namespace points_per_member_l6_6103

theorem points_per_member
    (total_members : ℕ)
    (absent_members : ℕ)
    (total_points : ℕ)
    (present_members : ℕ)
    (points_per_member : ℕ)
    (h1 : total_members = 5)
    (h2 : absent_members = 2)
    (h3 : total_points = 18)
    (h4 : present_members = total_members - absent_members)
    (h5 : points_per_member = total_points / present_members) :
  points_per_member = 6 :=
by
  sorry

end points_per_member_l6_6103


namespace tangent_circle_radius_l6_6923

theorem tangent_circle_radius (r1 r2 d : ℝ) (h1 : r2 = 2) (h2 : d = 5) (tangent : abs (r1 - r2) = d ∨ r1 + r2 = d) :
  r1 = 3 ∨ r1 = 7 :=
by
  sorry

end tangent_circle_radius_l6_6923


namespace abs_value_difference_l6_6277

theorem abs_value_difference (x y : ℤ) (h1 : |x| = 7) (h2 : |y| = 9) (h3 : |x + y| = -(x + y)) :
  x - y = 16 ∨ x - y = -16 :=
sorry

end abs_value_difference_l6_6277


namespace exists_prime_and_cube_root_l6_6560

theorem exists_prime_and_cube_root (n : ℕ) (hn : 0 < n) :
  ∃ (p m : ℕ), p.Prime ∧ p % 6 = 5 ∧ ¬p ∣ n ∧ n ≡ m^3 [MOD p] :=
sorry

end exists_prime_and_cube_root_l6_6560


namespace theo_cookies_l6_6062

theorem theo_cookies (cookies_per_time times_per_day total_cookies total_months : ℕ) (h1 : cookies_per_time = 13) (h2 : times_per_day = 3) (h3 : total_cookies = 2340) (h4 : total_months = 3) : (total_cookies / total_months) / (cookies_per_time * times_per_day) = 20 := 
by
  -- Placeholder for the proof
  sorry

end theo_cookies_l6_6062


namespace brandon_initial_skittles_l6_6825

theorem brandon_initial_skittles (initial_skittles : ℕ) (loss : ℕ) (final_skittles : ℕ) (h1 : final_skittles = 87) (h2 : loss = 9) (h3 : final_skittles = initial_skittles - loss) : initial_skittles = 96 :=
sorry

end brandon_initial_skittles_l6_6825


namespace increased_contact_area_effect_l6_6684

-- Define the conditions as assumptions
theorem increased_contact_area_effect (k : ℝ) (A₁ A₂ : ℝ) (dTdx : ℝ) (Q₁ Q₂ : ℝ) :
  (A₂ > A₁) →
  (Q₁ = -k * A₁ * dTdx) →
  (Q₂ = -k * A₂ * dTdx) →
  (Q₂ > Q₁) →
  ∃ increased_sensation : Prop, increased_sensation :=
by 
  exfalso
  sorry

end increased_contact_area_effect_l6_6684


namespace complex_root_cubic_l6_6814

theorem complex_root_cubic (a b q r : ℝ) (h_b_ne_zero : b ≠ 0)
  (h_root : (Polynomial.C a + Polynomial.C b * Polynomial.C I) * 
             (Polynomial.C a - Polynomial.C b * Polynomial.C I) * 
             (Polynomial.C (-2 * a)) 
             = Polynomial.X^3 + Polynomial.C q * Polynomial.X + Polynomial.C r) :
  q = b^2 - 3 * a^2 :=
sorry

end complex_root_cubic_l6_6814


namespace price_difference_is_300_cents_l6_6983

noncomputable def list_price : ℝ := 59.99
noncomputable def tech_bargains_price : ℝ := list_price - 15
noncomputable def digital_deal_price : ℝ := 0.7 * list_price
noncomputable def price_difference : ℝ := tech_bargains_price - digital_deal_price
noncomputable def price_difference_in_cents : ℝ := price_difference * 100

theorem price_difference_is_300_cents :
  price_difference_in_cents = 300 := by
  sorry

end price_difference_is_300_cents_l6_6983


namespace twelve_times_y_plus_three_half_quarter_l6_6071

theorem twelve_times_y_plus_three_half_quarter (y : ℝ) : 
  (1 / 2) * (1 / 4) * (12 * y + 3) = (3 * y) / 2 + 3 / 8 :=
by sorry

end twelve_times_y_plus_three_half_quarter_l6_6071


namespace lemon_pie_degrees_l6_6841

-- Defining the constants
def total_students : ℕ := 45
def chocolate_pie : ℕ := 15
def apple_pie : ℕ := 10
def blueberry_pie : ℕ := 9

-- Defining the remaining students
def remaining_students := total_students - (chocolate_pie + apple_pie + blueberry_pie)

-- Half of the remaining students prefer cherry pie and half prefer lemon pie
def students_prefer_cherry : ℕ := remaining_students / 2
def students_prefer_lemon : ℕ := remaining_students / 2

-- Defining the degree measure function
def degrees (students : ℕ) := (students * 360) / total_students

-- Proof statement
theorem lemon_pie_degrees : degrees students_prefer_lemon = 48 := by
  sorry  -- proof omitted

end lemon_pie_degrees_l6_6841


namespace family_b_initial_members_l6_6481

variable (x : ℕ)

theorem family_b_initial_members (h : 6 + (x - 1) + 9 + 12 + 5 + 9 = 48) : x = 8 :=
by
  sorry

end family_b_initial_members_l6_6481


namespace arithmetic_mean_frac_l6_6192

theorem arithmetic_mean_frac (y b : ℝ) (h : y ≠ 0) : 
  (1 / 2 : ℝ) * ((y + b) / y + (2 * y - b) / y) = 1.5 := 
by 
  sorry

end arithmetic_mean_frac_l6_6192


namespace Mrs_Heine_treats_l6_6309

theorem Mrs_Heine_treats :
  let dogs := 2
  let cats := 1
  let parrots := 3
  let biscuits_per_dog := 3
  let treats_per_cat := 2
  let sticks_per_parrot := 1
  let total_treats := dogs * biscuits_per_dog + cats * treats_per_cat + parrots * sticks_per_parrot
  total_treats = 11 :=
by
  let dogs := 2
  let cats := 1
  let parrots := 3
  let biscuits_per_dog := 3
  let treats_per_cat := 2
  let sticks_per_parrot := 1
  let total_treats := dogs * biscuits_per_dog + cats * treats_per_cat + parrots * sticks_per_parrot
  show total_treats = 11
  sorry

end Mrs_Heine_treats_l6_6309


namespace sum_of_coordinates_inv_graph_l6_6731

variable {f : ℝ → ℝ}
variable (hf : f 2 = 12)

theorem sum_of_coordinates_inv_graph :
  ∃ (x y : ℝ), y = f⁻¹ x / 3 ∧ x = 12 ∧ y = 2 / 3 ∧ x + y = 38 / 3 := by
  sorry

end sum_of_coordinates_inv_graph_l6_6731


namespace expression_value_l6_6850

/--
Prove that for a = 51 and b = 15, the expression (a + b)^2 - (a^2 + b^2) equals 1530.
-/
theorem expression_value (a b : ℕ) (h1 : a = 51) (h2 : b = 15) : (a + b)^2 - (a^2 + b^2) = 1530 := by
  rw [h1, h2]
  sorry

end expression_value_l6_6850


namespace strawberries_jam_profit_l6_6020

noncomputable def betty_strawberries : ℕ := 25
noncomputable def matthew_strawberries : ℕ := betty_strawberries + 30
noncomputable def natalie_strawberries : ℕ := matthew_strawberries / 3  -- Integer division rounds down
noncomputable def total_strawberries : ℕ := betty_strawberries + matthew_strawberries + natalie_strawberries
noncomputable def strawberries_per_jar : ℕ := 12
noncomputable def jars_of_jam : ℕ := total_strawberries / strawberries_per_jar  -- Integer division rounds down
noncomputable def money_per_jar : ℕ := 6
noncomputable def total_money_made : ℕ := jars_of_jam * money_per_jar

theorem strawberries_jam_profit :
  total_money_made = 48 := by
  sorry

end strawberries_jam_profit_l6_6020


namespace roots_of_polynomial_l6_6026

theorem roots_of_polynomial :
  {x : ℝ | x^10 - 5*x^8 + 4*x^6 - 64*x^4 + 320*x^2 - 256 = 0} = {x | x = 1 ∨ x = -1 ∨ x = 2 ∨ x = -2} :=
by
  sorry

end roots_of_polynomial_l6_6026


namespace domain_of_sqrt_expr_l6_6916

theorem domain_of_sqrt_expr (x : ℝ) : x ≥ 3 ∧ x < 8 ↔ x ∈ Set.Ico 3 8 :=
by
  sorry

end domain_of_sqrt_expr_l6_6916


namespace compute_100p_plus_q_l6_6011

theorem compute_100p_plus_q
  (p q : ℝ)
  (h1 : ∀ x : ℝ, (x + p) * (x + q) * (x + 15) = 0 → 
                  x ≠ -4 → x ≠ -15 → x ≠ -p → x ≠ -q)
  (h2 : ∀ x : ℝ, (x + 2 * p) * (x + 4) * (x + 9) = 0 → 
                  x ≠ -q → x ≠ -15 → (x = -4 ∨ x = -9))
  : 100 * p + q = -191 := 
sorry

end compute_100p_plus_q_l6_6011


namespace Mikey_leaves_l6_6466

theorem Mikey_leaves (initial_leaves : ℕ) (leaves_blew_away : ℕ) 
  (h1 : initial_leaves = 356) 
  (h2 : leaves_blew_away = 244) : 
  initial_leaves - leaves_blew_away = 112 :=
by
  -- proof steps would go here
  sorry

end Mikey_leaves_l6_6466


namespace new_temperature_l6_6932

-- Define the initial temperature
variable (t : ℝ)

-- Define the temperature drop
def temperature_drop : ℝ := 2

-- State the theorem
theorem new_temperature (t : ℝ) (temperature_drop : ℝ) : t - temperature_drop = t - 2 :=
by
  sorry

end new_temperature_l6_6932


namespace perfect_square_trinomial_l6_6284

theorem perfect_square_trinomial (x : ℝ) : (x + 9)^2 = x^2 + 18 * x + 81 := by
  sorry

end perfect_square_trinomial_l6_6284


namespace increasing_on_interval_solution_set_l6_6117

noncomputable def f (x : ℝ) : ℝ := x / (x ^ 2 + 1)

/- Problem 1 -/
theorem increasing_on_interval : ∀ x1 x2 : ℝ, -1 < x1 → x1 < x2 → x2 < 1 → f x1 < f x2 :=
by
  sorry

/- Problem 2 -/
theorem solution_set : ∀ x : ℝ, f (2 * x - 1) + f x < 0 ↔ 0 < x ∧ x < 1 / 3 :=
by
  sorry

end increasing_on_interval_solution_set_l6_6117


namespace perpendicular_lines_condition_l6_6068

variable {A1 B1 C1 A2 B2 C2 : ℝ}

theorem perpendicular_lines_condition :
  (∀ x y : ℝ, A1 * x + B1 * y + C1 = 0) ∧ (∀ x y : ℝ, A2 * x + B2 * y + C2 = 0) → 
  (A1 * A2) / (B1 * B2) = -1 := 
sorry

end perpendicular_lines_condition_l6_6068


namespace complement_intersection_l6_6721

def set_P : Set ℝ := {x | x^2 - 2 * x ≥ 0}
def set_Q : Set ℝ := {x | 1 < x ∧ x ≤ 2}

theorem complement_intersection (P Q : Set ℝ) (hP : P = set_P) (hQ : Q = set_Q) :
  (Pᶜ ∩ Q) = {x | 1 < x ∧ x < 2} :=
by
  sorry

end complement_intersection_l6_6721


namespace inequality_proof_l6_6525

variables {a b c : ℝ}

theorem inequality_proof (h1 : a > b) (h2 : b > c) (h3 : a + b + c = 0) : a * c < b * c :=
by
  sorry

end inequality_proof_l6_6525


namespace ending_number_divisible_by_3_l6_6074

theorem ending_number_divisible_by_3 : 
∃ n : ℕ, (∀ k : ℕ, (10 + k * 3) ≤ n → (10 + k * 3) % 3 = 0) ∧ 
       (∃ c : ℕ, c = 12 ∧ (n - 10) / 3 + 1 = c) ∧ 
       n = 45 := 
sorry

end ending_number_divisible_by_3_l6_6074


namespace value_range_of_a_l6_6539

variable (a : ℝ)
variable (suff_not_necess : ∀ x, x ∈ ({3, a} : Set ℝ) → 2 * x^2 - 5 * x - 3 ≥ 0)

theorem value_range_of_a :
  (a ≤ -1/2 ∨ a > 3) :=
sorry

end value_range_of_a_l6_6539


namespace paula_bought_fewer_cookies_l6_6312
-- Import the necessary libraries

-- Definitions
def paul_cookies : ℕ := 45
def total_cookies : ℕ := 87

-- Theorem statement
theorem paula_bought_fewer_cookies : ∃ (paula_cookies : ℕ), paul_cookies + paula_cookies = total_cookies ∧ paul_cookies - paula_cookies = 3 := by
  sorry

end paula_bought_fewer_cookies_l6_6312


namespace cricket_problem_solved_l6_6779

noncomputable def cricket_problem : Prop :=
  let run_rate_10 := 3.2
  let target := 252
  let required_rate := 5.5
  let overs_played := 10
  let total_overs := 50
  let runs_scored := run_rate_10 * overs_played
  let runs_remaining := target - runs_scored
  let overs_remaining := total_overs - overs_played
  (runs_remaining / overs_remaining = required_rate)

theorem cricket_problem_solved : cricket_problem :=
by
  sorry

end cricket_problem_solved_l6_6779


namespace limes_remaining_l6_6500

-- Definitions based on conditions
def initial_limes : ℕ := 9
def limes_given_to_Sara : ℕ := 4

-- Theorem to prove
theorem limes_remaining : initial_limes - limes_given_to_Sara = 5 :=
by
  -- Sorry keyword to skip the actual proof
  sorry

end limes_remaining_l6_6500


namespace bus_capacity_l6_6379

theorem bus_capacity :
  ∀ (left_seats right_seats people_per_seat back_seat : ℕ),
  left_seats = 15 →
  right_seats = left_seats - 3 →
  people_per_seat = 3 →
  back_seat = 11 →
  (left_seats * people_per_seat) + 
  (right_seats * people_per_seat) + 
  back_seat = 92 := by
  intros left_seats right_seats people_per_seat back_seat 
  intros h1 h2 h3 h4 
  sorry

end bus_capacity_l6_6379


namespace jake_weight_loss_l6_6448

variable {J K L : Nat}

theorem jake_weight_loss
  (h1 : J + K = 290)
  (h2 : J = 196)
  (h3 : J - L = 2 * K) : L = 8 :=
by
  sorry

end jake_weight_loss_l6_6448


namespace ecuadorian_number_unique_l6_6127

def is_Ecuadorian (n : ℕ) : Prop :=
  let a := n / 100
  let b := (n / 10) % 10
  let c := n % 10
  n < 1000 ∧ c ≠ 0 ∧ n % 36 = 0 ∧ (n - (100 * c + 10 * b + a) > 0) ∧ (n - (100 * c + 10 * b + a)) % 36 = 0

theorem ecuadorian_number_unique (n : ℕ) : 
  is_Ecuadorian n → n = 864 :=
sorry

end ecuadorian_number_unique_l6_6127


namespace total_dress_designs_l6_6818

theorem total_dress_designs:
  let colors := 5
  let patterns := 4
  let sleeve_lengths := 2
  colors * patterns * sleeve_lengths = 40 := 
by
  sorry

end total_dress_designs_l6_6818


namespace one_greater_l6_6331

theorem one_greater (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a * b * c = 1) 
  (h5 : a + b + c > 1 / a + 1 / b + 1 / c) : 
  (1 < a ∧ b ≤ 1 ∧ c ≤ 1) ∨ (1 < b ∧ a ≤ 1 ∧ c ≤ 1) ∨ (1 < c ∧ a ≤ 1 ∧ b ≤ 1) :=
sorry

end one_greater_l6_6331


namespace missing_number_is_eight_l6_6485

theorem missing_number_is_eight (x : ℤ) : (4 + 3) + (x - 3 - 1) = 11 → x = 8 := by
  intro h
  sorry

end missing_number_is_eight_l6_6485


namespace number_of_students_l6_6725

theorem number_of_students (n : ℕ) (h1 : n < 50) (h2 : n % 8 = 5) (h3 : n % 6 = 4) : n = 13 :=
by
  sorry

end number_of_students_l6_6725


namespace prob_A_not_losing_prob_A_not_winning_l6_6027

-- Definitions based on the conditions
def prob_winning : ℝ := 0.41
def prob_tie : ℝ := 0.27

-- The probability of A not losing
def prob_not_losing : ℝ := prob_winning + prob_tie

-- The probability of A not winning
def prob_not_winning : ℝ := 1 - prob_winning

-- Proof problems
theorem prob_A_not_losing : prob_not_losing = 0.68 := by
  sorry

theorem prob_A_not_winning : prob_not_winning = 0.59 := by
  sorry

end prob_A_not_losing_prob_A_not_winning_l6_6027


namespace solution_set_of_inequality_l6_6616

theorem solution_set_of_inequality :
  { x : ℝ | |x + 1| + |x - 4| ≥ 7 } = { x : ℝ | x ≤ -2 ∨ x ≥ 5 } := sorry

end solution_set_of_inequality_l6_6616


namespace points_on_line_l6_6972

theorem points_on_line (b m n : ℝ) (hA : m = -(-5) + b) (hB : n = -(4) + b) :
  m > n :=
by
  sorry

end points_on_line_l6_6972


namespace pythagorean_theorem_l6_6986

theorem pythagorean_theorem (a b c : ℝ) : (a^2 + b^2 = c^2) ↔ (a^2 + b^2 = c^2) :=
by sorry

end pythagorean_theorem_l6_6986


namespace ratio_of_part_diminished_by_10_to_whole_number_l6_6962

theorem ratio_of_part_diminished_by_10_to_whole_number (N : ℝ) (x : ℝ) (h1 : 1/5 * N + 4 = x * N - 10) (h2 : N = 280) :
  x = 1 / 4 :=
by
  rw [h2] at h1
  sorry

end ratio_of_part_diminished_by_10_to_whole_number_l6_6962


namespace sum_of_midpoints_l6_6038

theorem sum_of_midpoints 
  (a b c d e f : ℝ)
  (h1 : a + b + c = 15)
  (h2 : d + e + f = 15) :
  ((a + b) / 2 + (a + c) / 2 + (b + c) / 2 = 15) ∧ 
  ((d + e) / 2 + (d + f) / 2 + (e + f) / 2 = 15) :=
by
  sorry

end sum_of_midpoints_l6_6038


namespace smallest_s_for_347_l6_6369

open Nat

theorem smallest_s_for_347 (r s : ℕ) (hr_pos : 0 < r) (hs_pos : 0 < s) 
  (h_rel_prime : Nat.gcd r s = 1) (h_r_lt_s : r < s) 
  (h_contains_347 : ∃ k : ℕ, ∃ y : ℕ, 10 ^ k * r - s * y = 347): 
  s = 653 := 
by sorry

end smallest_s_for_347_l6_6369


namespace polyhedron_value_l6_6606

theorem polyhedron_value (T H V E : ℕ) (h t : ℕ) 
  (F : ℕ) (h_eq : h = 10) (t_eq : t = 10)
  (F_eq : F = 20)
  (edges_eq : E = (3 * t + 6 * h) / 2)
  (vertices_eq : V = E - F + 2)
  (T_value : T = 2) (H_value : H = 2) :
  100 * H + 10 * T + V = 227 := by
  sorry

end polyhedron_value_l6_6606


namespace max_value_X2_plus_2XY_plus_3Y2_l6_6013

theorem max_value_X2_plus_2XY_plus_3Y2 
  (x y : ℝ) 
  (h₁ : 0 < x) (h₂ : 0 < y) 
  (h₃ : x^2 - 2 * x * y + 3 * y^2 = 10) : 
  x^2 + 2 * x * y + 3 * y^2 ≤ 30 + 20 * Real.sqrt 3 :=
sorry

end max_value_X2_plus_2XY_plus_3Y2_l6_6013


namespace unit_cost_of_cranberry_juice_l6_6689

theorem unit_cost_of_cranberry_juice (total_cost : ℕ) (ounces : ℕ) (h1 : total_cost = 84) (h2 : ounces = 12) :
  total_cost / ounces = 7 :=
by
  sorry

end unit_cost_of_cranberry_juice_l6_6689


namespace knights_in_exchange_l6_6467

noncomputable def count_knights (total_islanders : ℕ) (odd_statements : ℕ) (even_statements : ℕ) : ℕ :=
if total_islanders % 2 = 0 ∧ odd_statements = total_islanders ∧ even_statements = total_islanders then
    total_islanders / 2
else
    0

theorem knights_in_exchange : count_knights 30 30 30 = 15 :=
by
    -- proof part will go here but is not required.
    sorry

end knights_in_exchange_l6_6467


namespace combined_work_rate_l6_6341

theorem combined_work_rate (W : ℝ) 
  (A_rate : ℝ := W / 10) 
  (B_rate : ℝ := W / 5) : 
  A_rate + B_rate = 3 * W / 10 := 
by
  sorry

end combined_work_rate_l6_6341


namespace color_points_l6_6367

def is_white (p : ℤ × ℤ) : Prop := (p.1 % 2 = 1) ∧ (p.2 % 2 = 1)
def is_black (p : ℤ × ℤ) : Prop := (p.1 % 2 = 0) ∧ (p.2 % 2 = 0)
def is_red (p : ℤ × ℤ) : Prop := (p.1 % 2 = 1 ∧ p.2 % 2 = 0) ∨ (p.1 % 2 = 0 ∧ p.2 % 2 = 1)

theorem color_points :
  (∀ n : ℤ, ∃ (p : ℤ × ℤ), (p.2 = n) ∧ is_white p ∧
                             is_black ⟨p.1, n * 2⟩ ∧
                             is_red ⟨p.1, n * 2 + 1⟩) ∧ 
  (∀ (A B C : ℤ × ℤ), 
    is_white A → is_red B → is_black C → 
    ∃ D : ℤ × ℤ, is_red D ∧ 
    (A.1 + C.1 - B.1 = D.1 ∧
     A.2 + C.2 - B.2 = D.2)) := sorry

end color_points_l6_6367


namespace find_positive_difference_l6_6547

theorem find_positive_difference 
  (p1 p2 : ℝ × ℝ) (q1 q2 : ℝ × ℝ) 
  (h_p1 : p1 = (0, 8)) (h_p2 : p2 = (4, 0))
  (h_q1 : q1 = (0, 5)) (h_q2 : q2 = (10, 0))
  (y : ℝ) (hy : y = 20) :
  let m_p := (p2.2 - p1.2) / (p2.1 - p1.1)
  let b_p := p1.2 - m_p * p1.1
  let x_p := (y - b_p) / m_p
  let m_q := (q2.2 - q1.2) / (q2.1 - q1.1)
  let b_q := q1.2 - m_q * q1.1
  let x_q := (y - b_q) / m_q
  abs (x_p - x_q) = 24 :=
by
  sorry

end find_positive_difference_l6_6547


namespace area_of_enclosed_shape_l6_6063

noncomputable def enclosed_area : ℝ :=
  ∫ x in (0 : ℝ)..(2 : ℝ), (4 * x - x^3)

theorem area_of_enclosed_shape : enclosed_area = 4 := by
  sorry

end area_of_enclosed_shape_l6_6063


namespace sqrt_E_nature_l6_6856

def E (x : ℤ) : ℤ :=
  let a := x
  let b := x + 1
  let c := a * b
  let d := b * c
  a^2 + b^2 + c^2 + d^2

theorem sqrt_E_nature : ∀ x : ℤ, (∃ n : ℤ, n^2 = E x) ∧ (∃ m : ℤ, m^2 ≠ E x) :=
  by
  sorry

end sqrt_E_nature_l6_6856


namespace sum_of_first_six_terms_l6_6334

def geometric_sequence (a : ℕ → ℤ) :=
  a 1 = 1 ∧ ∀ n, n ≥ 2 → a n = -2 * a (n - 1)

def sum_first_six_terms (a : ℕ → ℤ) : ℤ := 
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6

theorem sum_of_first_six_terms (a : ℕ → ℤ) 
  (h : geometric_sequence a) :
  sum_first_six_terms a = -21 :=
sorry

end sum_of_first_six_terms_l6_6334


namespace total_cookies_l6_6832

def mona_cookies : ℕ := 20
def jasmine_cookies : ℕ := mona_cookies - 5
def rachel_cookies : ℕ := jasmine_cookies + 10

theorem total_cookies : mona_cookies + jasmine_cookies + rachel_cookies = 60 := 
by
  have h1 : jasmine_cookies = 15 := by sorry
  have h2 : rachel_cookies = 25 := by sorry
  have h3 : mona_cookies = 20 := by sorry
  sorry

end total_cookies_l6_6832


namespace find_non_divisible_and_product_l6_6908

-- Define the set of numbers
def numbers : List Nat := [3543, 3552, 3567, 3579, 3581]

-- Function to get the digits of a number
def digits (n : Nat) : List Nat := n.digits 10

-- Function to sum the digits
def sum_of_digits (n : Nat) : Nat := (digits n).sum

-- Function to check divisibility by 3
def divisible_by_3 (n : Nat) : Bool := sum_of_digits n % 3 = 0

-- Find the units digit of a number
def units_digit (n : Nat) : Nat := n % 10

-- Find the tens digit of a number
def tens_digit (n : Nat) : Nat := (n / 10) % 10

-- The problem statement
theorem find_non_divisible_and_product :
  ∃ n ∈ numbers, ¬ divisible_by_3 n ∧ units_digit n * tens_digit n = 8 :=
by
  sorry

end find_non_divisible_and_product_l6_6908


namespace free_throws_count_l6_6459

-- Given conditions:
variables (a b x : ℕ) -- α is an abbreviation for natural numbers

-- Condition: number of points from all shots
axiom points_condition : 2 * a + 3 * b + x = 79
-- Condition: three-point shots are twice the points of two-point shots
axiom three_point_condition : 3 * b = 4 * a
-- Condition: number of free throws is one more than the number of two-point shots
axiom free_throw_condition : x = a + 1

-- Prove that the number of free throws is 12
theorem free_throws_count : x = 12 :=
by {
  sorry
}

end free_throws_count_l6_6459


namespace max_area_of_triangle_ABC_l6_6760

noncomputable def max_area_triangle_ABC: ℝ :=
  let QA := 3
  let QB := 4
  let QC := 5
  let BC := 6
  -- Given these conditions, prove the maximum area of triangle ABC
  19

theorem max_area_of_triangle_ABC 
  (QA QB QC BC : ℝ) 
  (h1 : QA = 3) 
  (h2 : QB = 4) 
  (h3 : QC = 5) 
  (h4 : BC = 6) 
  (h5 : QB * QB + BC * BC = QC * QC) -- The right angle condition at Q
  : max_area_triangle_ABC = 19 :=
by sorry

end max_area_of_triangle_ABC_l6_6760


namespace andrew_friends_brought_food_l6_6528

theorem andrew_friends_brought_food (slices_per_friend total_slices : ℕ) (h1 : slices_per_friend = 4) (h2 : total_slices = 16) :
  total_slices / slices_per_friend = 4 :=
by
  sorry

end andrew_friends_brought_food_l6_6528


namespace right_triangle_bc_is_3_l6_6264

-- Define the setup: a right triangle with given side lengths
structure RightTriangle :=
  (AB AC BC : ℝ)
  (right_angle : AB^2 = AC^2 + BC^2)
  (AB_val : AB = 5)
  (AC_val : AC = 4)

-- The goal is to prove that BC = 3 given the conditions
theorem right_triangle_bc_is_3 (T : RightTriangle) : T.BC = 3 :=
  sorry

end right_triangle_bc_is_3_l6_6264


namespace factor_difference_of_cubes_l6_6558

theorem factor_difference_of_cubes (t : ℝ) : 
  t^3 - 125 = (t - 5) * (t^2 + 5 * t + 25) :=
sorry

end factor_difference_of_cubes_l6_6558


namespace candies_per_friend_l6_6902

theorem candies_per_friend (initial_candies : ℕ) (additional_candies : ℕ) (friends : ℕ) 
  (h_initial : initial_candies = 10)
  (h_additional : additional_candies = 4)
  (h_friends : friends = 7) : initial_candies + additional_candies = 14 ∧ 14 / friends = 2 :=
by
  sorry

end candies_per_friend_l6_6902


namespace k_2_sufficient_but_not_necessary_l6_6809

def vector_a : ℝ × ℝ := (2, 1)
def vector_b (k : ℝ) : ℝ × ℝ := (1, k^2 - 1) - (2, 1)

def perpendicular (x y : ℝ × ℝ) : Prop := x.1 * y.1 + x.2 * y.2 = 0

theorem k_2_sufficient_but_not_necessary (k : ℝ) :
  k = 2 → perpendicular vector_a (vector_b k) ∧ ∃ k, not (k = 2) ∧ perpendicular vector_a (vector_b k) :=
by
  sorry

end k_2_sufficient_but_not_necessary_l6_6809


namespace continuous_tape_length_l6_6771

theorem continuous_tape_length :
  let num_sheets := 15
  let sheet_length_cm := 25
  let overlap_cm := 0.5 
  let total_length_without_overlap := num_sheets * sheet_length_cm
  let num_overlaps := num_sheets - 1
  let total_overlap_length := num_overlaps * overlap_cm
  let total_length_cm := total_length_without_overlap - total_overlap_length
  let total_length_m := total_length_cm / 100
  total_length_m = 3.68 := 
by {
  sorry
}

end continuous_tape_length_l6_6771


namespace find_k_l6_6175

theorem find_k 
  (h : ∀ x k : ℝ, x^2 + (k^2 - 4)*x + k - 1 = 0 → ∃ x₁ x₂ : ℝ, x₁ + x₂ = 0):
  ∃ (k : ℝ), k = -2 :=
sorry

end find_k_l6_6175


namespace new_students_weights_correct_l6_6302

-- Definitions of the initial conditions
def initial_student_count : ℕ := 29
def initial_avg_weight : ℚ := 28
def total_initial_weight := initial_student_count * initial_avg_weight
def new_student_counts : List ℕ := [30, 31, 32, 33]
def new_avg_weights : List ℚ := [27.2, 27.8, 27.6, 28]

-- Weights of the four new students
def W1 : ℚ := 4
def W2 : ℚ := 45.8
def W3 : ℚ := 21.4
def W4 : ℚ := 40.8

-- The proof statement
theorem new_students_weights_correct :
  total_initial_weight = 812 ∧
  W1 = 4 ∧
  W2 = 45.8 ∧
  W3 = 21.4 ∧
  W4 = 40.8 ∧
  (total_initial_weight + W1) = 816 ∧
  (total_initial_weight + W1) / new_student_counts.head! = new_avg_weights.head! ∧
  (total_initial_weight + W1 + W2) = 861.8 ∧
  (total_initial_weight + W1 + W2) / new_student_counts.tail.head! = new_avg_weights.tail.head! ∧
  (total_initial_weight + W1 + W2 + W3) = 883.2 ∧
  (total_initial_weight + W1 + W2 + W3) / new_student_counts.tail.tail.head! = new_avg_weights.tail.tail.head! ∧
  (total_initial_weight + W1 + W2 + W3 + W4) = 924 ∧
  (total_initial_weight + W1 + W2 + W3 + W4) / new_student_counts.tail.tail.tail.head! = new_avg_weights.tail.tail.tail.head! :=
by
  sorry

end new_students_weights_correct_l6_6302


namespace card_count_l6_6837

theorem card_count (x y : ℕ) (h1 : x + y + 2 = 10) (h2 : 3 * x + 4 * y + 10 = 39) : x = 3 :=
by {
  sorry
}

end card_count_l6_6837


namespace johns_total_earnings_l6_6449

noncomputable def total_earnings_per_week (baskets_monday : ℕ) (baskets_thursday : ℕ) (small_crabs_per_basket : ℕ) (large_crabs_per_basket : ℕ) (price_small_crab : ℕ) (price_large_crab : ℕ) : ℕ :=
  let small_crabs := baskets_monday * small_crabs_per_basket
  let large_crabs := baskets_thursday * large_crabs_per_basket
  (small_crabs * price_small_crab) + (large_crabs * price_large_crab)

theorem johns_total_earnings :
  total_earnings_per_week 3 4 4 5 3 5 = 136 :=
by
  sorry

end johns_total_earnings_l6_6449


namespace village_distance_l6_6584

theorem village_distance
  (d : ℝ)
  (uphill_speed : ℝ) (downhill_speed : ℝ)
  (total_time : ℝ)
  (h1 : uphill_speed = 15)
  (h2 : downhill_speed = 30)
  (h3 : total_time = 4) :
  d = 40 :=
by
  sorry

end village_distance_l6_6584


namespace problem_180_l6_6159

variables (P Q : Prop)

theorem problem_180 (h : P → Q) : ¬ (P ∨ ¬Q) :=
sorry

end problem_180_l6_6159


namespace find_number_l6_6548

theorem find_number (x : ℚ) (h : 0.5 * x = (3/5) * x - 10) : x = 100 := 
sorry

end find_number_l6_6548


namespace num_students_l6_6652

theorem num_students (n : ℕ) 
    (average_marks_wrong : ℕ)
    (wrong_mark : ℕ)
    (correct_mark : ℕ)
    (average_marks_correct : ℕ) :
    average_marks_wrong = 100 →
    wrong_mark = 90 →
    correct_mark = 10 →
    average_marks_correct = 92 →
    n = 10 :=
by
  intros h1 h2 h3 h4
  sorry

end num_students_l6_6652


namespace part_a_part_b_l6_6873

-- Part (a): Proving at most one integer solution for general k
theorem part_a (k : ℤ) : 
  ∀ (x1 x2 : ℤ), (x1^3 - 24*x1 + k = 0 ∧ x2^3 - 24*x2 + k = 0) → x1 = x2 :=
sorry

-- Part (b): Proving exactly one integer solution for k = -2016
theorem part_b :
  ∃! (x : ℤ), x^3 + 24*x - 2016 = 0 :=
sorry

end part_a_part_b_l6_6873


namespace existence_of_epsilon_and_u_l6_6542

theorem existence_of_epsilon_and_u (n : ℕ) (h : 0 < n) :
  ∀ k ≥ 1, ∃ ε : ℝ, (0 < ε ∧ ε < 1 / k) ∧
  (∀ (a : Fin n → ℝ), (∀ i, 0 < a i) → ∃ u > 0, ∀ i, ε < (u * a i - ⌊u * a i⌋) ∧ (u * a i - ⌊u * a i⌋) < 1 / k) :=
by {
  sorry
}

end existence_of_epsilon_and_u_l6_6542


namespace ratio_fraction_l6_6634

variable (X Y Z : ℝ)
variable (k : ℝ) (hk : k > 0)

-- Given conditions
def ratio_condition := (3 * Y = 2 * X) ∧ (6 * Y = 2 * Z)

-- Statement
theorem ratio_fraction (h : ratio_condition X Y Z) : 
  (2 * X + 3 * Y) / (5 * Z - 2 * X) = 1 / 2 := by
  sorry

end ratio_fraction_l6_6634


namespace leftover_space_desks_bookcases_l6_6533

theorem leftover_space_desks_bookcases 
  (number_of_desks : ℕ) (number_of_bookcases : ℕ)
  (wall_length : ℝ) (desk_length : ℝ) (bookcase_length : ℝ) (space_between : ℝ)
  (equal_number : number_of_desks = number_of_bookcases)
  (wall_length_eq : wall_length = 15)
  (desk_length_eq : desk_length = 2)
  (bookcase_length_eq : bookcase_length = 1.5)
  (space_between_eq : space_between = 0.5) :
  ∃ k : ℝ, k = 3 := 
by
  sorry

end leftover_space_desks_bookcases_l6_6533


namespace intersection_of_complements_l6_6743

open Set

variable (U A B : Set Nat)
variable (hU : U = {1, 2, 3, 4, 5, 6, 7, 8})
variable (hA : A = {2, 3, 4, 5})
variable (hB : B = {2, 4, 6, 8})

theorem intersection_of_complements :
  A ∩ (U \ B) = {3, 5} :=
by
  rw [hU, hA, hB]
  sorry

end intersection_of_complements_l6_6743


namespace current_tree_height_in_inches_l6_6859

-- Constants
def initial_height_ft : ℝ := 10
def growth_percentage : ℝ := 0.50
def feet_to_inches : ℝ := 12

-- Conditions
def growth_ft : ℝ := growth_percentage * initial_height_ft
def current_height_ft : ℝ := initial_height_ft + growth_ft

-- Question/Answer equivalence
theorem current_tree_height_in_inches :
  (current_height_ft * feet_to_inches) = 180 :=
by 
  sorry

end current_tree_height_in_inches_l6_6859


namespace parabola_constant_term_l6_6704

theorem parabola_constant_term 
  (b c : ℝ)
  (h1 : 2 = 2 * (1 : ℝ)^2 + b * (1 : ℝ) + c)
  (h2 : 2 = 2 * (3 : ℝ)^2 + b * (3 : ℝ) + c) : 
  c = 8 :=
by
  sorry

end parabola_constant_term_l6_6704


namespace expected_value_of_biased_die_l6_6564

-- Define the probabilities
def P1 : ℚ := 1/10
def P2 : ℚ := 1/10
def P3 : ℚ := 2/10
def P4 : ℚ := 2/10
def P5 : ℚ := 2/10
def P6 : ℚ := 2/10

-- Define the outcomes
def X1 : ℚ := 1
def X2 : ℚ := 2
def X3 : ℚ := 3
def X4 : ℚ := 4
def X5 : ℚ := 5
def X6 : ℚ := 6

-- Define the expected value calculation according to the probabilities and outcomes
def expected_value : ℚ := P1 * X1 + P2 * X2 + P3 * X3 + P4 * X4 + P5 * X5 + P6 * X6

-- The theorem we want to prove
theorem expected_value_of_biased_die : expected_value = 3.9 := by
  -- We skip the proof here with sorry for now
  sorry

end expected_value_of_biased_die_l6_6564


namespace proof_smallest_lcm_1_to_12_l6_6538

noncomputable def smallest_lcm_1_to_12 : ℕ := 27720

theorem proof_smallest_lcm_1_to_12 :
  ∀ n : ℕ, (∀ i : ℕ, 1 ≤ i ∧ i ≤ 12 → i ∣ n) ↔ n = 27720 :=
by
  sorry

end proof_smallest_lcm_1_to_12_l6_6538


namespace find_children_and_coins_l6_6077

def condition_for_child (k m remaining_coins : ℕ) : Prop :=
  ∃ (received_coins : ℕ), (received_coins = k + remaining_coins / 7 ∧ received_coins * 7 = 7 * k + remaining_coins)

def valid_distribution (n m : ℕ) : Prop :=
  ∀ k (hk : 1 ≤ k ∧ k ≤ n),
  ∃ remaining_coins,
    condition_for_child k m remaining_coins

theorem find_children_and_coins :
  ∃ n m, valid_distribution n m ∧ n = 6 ∧ m = 36 :=
sorry

end find_children_and_coins_l6_6077


namespace correct_option_l6_6737

def U : Set ℝ := Set.univ
def M : Set ℝ := {x | x < 1}
def N : Set ℝ := {x | 0 < x ∧ x < 1}

theorem correct_option : M ∪ (U \ N) = U :=
by
  sorry

end correct_option_l6_6737


namespace max_value_of_expression_l6_6394

theorem max_value_of_expression (a b : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : 2 * a + b = 1) : 
  2 * Real.sqrt (a * b) - 4 * a ^ 2 - b ^ 2 ≤ (Real.sqrt 2 - 1) / 2 :=
sorry

end max_value_of_expression_l6_6394


namespace complement_of_P_subset_Q_l6_6440

-- Definitions based on conditions
def P : Set ℝ := {x | x < 1}
def Q : Set ℝ := {x | x > -1}

-- Theorem statement to prove the correct option C
theorem complement_of_P_subset_Q : {x | ¬ (x < 1)} ⊆ {x | x > -1} :=
by {
  sorry
}

end complement_of_P_subset_Q_l6_6440


namespace factorization_sum_l6_6903

theorem factorization_sum (a b c : ℤ) 
  (h1 : ∀ x : ℝ, (x + a) * (x + b) = x^2 + 13 * x + 40)
  (h2 : ∀ x : ℝ, (x - b) * (x - c) = x^2 - 19 * x + 88) :
  a + b + c = 24 := 
sorry

end factorization_sum_l6_6903


namespace distance_between_stations_l6_6985

-- distance calculation definitions
def distance (rate time : ℝ) := rate * time

-- problem conditions as definitions
def rate_slow := 20 -- km/hr
def rate_fast := 25 -- km/hr
def extra_distance := 50 -- km

-- final statement
theorem distance_between_stations :
  ∃ (D : ℝ) (T : ℝ),
    (distance rate_slow T = D) ∧
    (distance rate_fast T = D + extra_distance) ∧
    (D + (D + extra_distance) = 450) :=
by
  sorry

end distance_between_stations_l6_6985


namespace B_finishes_in_4_days_l6_6869

theorem B_finishes_in_4_days
  (A_days : ℕ) (B_days : ℕ) (working_days_together : ℕ) 
  (A_rate : ℝ) (B_rate : ℝ) (combined_rate : ℝ) (work_done : ℝ) (remaining_work : ℝ)
  (B_rate_alone : ℝ) (days_B: ℝ) :
  A_days = 5 →
  B_days = 10 →
  working_days_together = 2 →
  A_rate = 1 / A_days →
  B_rate = 1 / B_days →
  combined_rate = A_rate + B_rate →
  work_done = combined_rate * working_days_together →
  remaining_work = 1 - work_done →
  B_rate_alone = 1 / B_days →
  days_B = remaining_work / B_rate_alone →
  days_B = 4 := 
by
  intros
  sorry

end B_finishes_in_4_days_l6_6869


namespace slices_all_three_toppings_l6_6736

def slices_with_all_toppings (total_slices pepperoni_slices mushroom_slices olive_slices : ℕ) : ℕ := 
  (12 : ℕ)

theorem slices_all_three_toppings
  (total_slices : ℕ)
  (pepperoni_slices : ℕ)
  (mushroom_slices : ℕ)
  (olive_slices : ℕ)
  (h : total_slices = 24)
  (h1 : pepperoni_slices = 12)
  (h2 : mushroom_slices = 14)
  (h3 : olive_slices = 16)
  (hc : total_slices ≥ 0)
  (hc1 : pepperoni_slices ≥ 0)
  (hc2 : mushroom_slices ≥ 0)
  (hc3 : olive_slices ≥ 0) :
  slices_with_all_toppings total_slices pepperoni_slices mushroom_slices olive_slices = 2 :=
  sorry

end slices_all_three_toppings_l6_6736


namespace cost_of_tax_free_items_l6_6935

theorem cost_of_tax_free_items : 
  ∀ (total_spent : ℝ) (sales_tax : ℝ) (tax_rate : ℝ) (taxable_cost : ℝ),
  total_spent = 25 ∧ sales_tax = 0.30 ∧ tax_rate = 0.05 ∧ sales_tax = tax_rate * taxable_cost → 
  total_spent - taxable_cost = 19 :=
by
  intros total_spent sales_tax tax_rate taxable_cost
  intro h
  sorry

end cost_of_tax_free_items_l6_6935


namespace distinct_colored_triangle_l6_6751

open Finset

variables {n k : ℕ} (hn : 0 < n) (hk : 3 ≤ k)
variables (K : SimpleGraph (Fin n))
variables (color : Edge (Fin n) → Fin k)
variables (connected_subgraph : ∀ i : Fin k, ∀ u v : Fin n, u ≠ v → (∃ p : Walk (Fin n) u v, ∀ {e}, e ∈ p.edges → color e = i))

theorem distinct_colored_triangle :
  ∃ (A B C : Fin n), A ≠ B ∧ B ≠ C ∧ C ≠ A ∧
  color (A, B) ≠ color (B, C) ∧
  color (B, C) ≠ color (C, A) ∧
  color (C, A) ≠ color (A, B) :=
sorry

end distinct_colored_triangle_l6_6751


namespace total_students_l6_6243

theorem total_students
  (T : ℝ) 
  (h1 : 0.20 * T = 168)
  (h2 : 0.30 * T = 252) : T = 840 :=
sorry

end total_students_l6_6243


namespace smallest_difference_l6_6705

theorem smallest_difference (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_prod : a * b * c = 362880) (h_order : a < b ∧ b < c) : c - a = 92 := 
sorry

end smallest_difference_l6_6705


namespace necessary_and_sufficient_condition_l6_6614

theorem necessary_and_sufficient_condition (x : ℝ) (h : x > 0) : (x + 1/x ≥ 2) ↔ (x > 0) :=
sorry

end necessary_and_sufficient_condition_l6_6614


namespace sasha_age_l6_6321

theorem sasha_age :
  ∃ a : ℕ, 
    (M = 2 * a - 3) ∧
    (M = a + (a - 3)) ∧
    (a = 3) :=
by
  sorry

end sasha_age_l6_6321


namespace sum_of_sequence_l6_6122

theorem sum_of_sequence (avg : ℕ → ℕ → ℕ) (n : ℕ) (total_sum : ℕ) 
  (condition : avg 16 272 = 17) : 
  total_sum = 272 := 
by 
  sorry

end sum_of_sequence_l6_6122


namespace no_real_solution_x_4_plus_x_plus1_4_plus_x_plus2_4_eq_x_plus3_4_plus_10_l6_6240

theorem no_real_solution_x_4_plus_x_plus1_4_plus_x_plus2_4_eq_x_plus3_4_plus_10 :
  ¬ ∃ x : ℝ, x^4 + (x + 1)^4 + (x + 2)^4 = (x + 3)^4 + 10 :=
by {
  sorry
}

end no_real_solution_x_4_plus_x_plus1_4_plus_x_plus2_4_eq_x_plus3_4_plus_10_l6_6240


namespace largest_integral_solution_l6_6526

theorem largest_integral_solution : ∃ x : ℤ, (1 / 4 < x / 7 ∧ x / 7 < 3 / 5) ∧ ∀ y : ℤ, (1 / 4 < y / 7 ∧ y / 7 < 3 / 5) → y ≤ x := sorry

end largest_integral_solution_l6_6526


namespace problem_solution_l6_6719

def diamond (x y k : ℝ) : ℝ := x^2 - k * y

theorem problem_solution (h : ℝ) (k : ℝ) (hk : k = 3) : 
  diamond h (diamond h h k) k = -2 * h^2 + 9 * h :=
by
  rw [hk, diamond, diamond]
  sorry

end problem_solution_l6_6719


namespace find_function_l6_6530

theorem find_function (f : ℝ → ℝ) (h : ∀ x : ℝ, x ≠ 0 → f x + 2 * f (1 / x) = 3 * x) : 
  ∀ x : ℝ, x ≠ 0 → f x = -x + 2 / x := 
by
  sorry

end find_function_l6_6530


namespace solution_for_g0_l6_6401

variable (g : ℝ → ℝ)

def functional_eq_condition := ∀ x y : ℝ, g (x + y) = g x + g y - 1

theorem solution_for_g0 (h : functional_eq_condition g) : g 0 = 1 :=
by {
  sorry
}

end solution_for_g0_l6_6401


namespace evaluate_seventy_five_squared_minus_twenty_five_squared_l6_6288

theorem evaluate_seventy_five_squared_minus_twenty_five_squared :
  75^2 - 25^2 = 5000 :=
by
  sorry

end evaluate_seventy_five_squared_minus_twenty_five_squared_l6_6288


namespace isolate_urea_decomposing_bacteria_valid_option_l6_6006

variable (KH2PO4 Na2HPO4 MgSO4_7H2O urea glucose agar water : Type)
variable (urea_decomposing_bacteria : Type)
variable (CarbonSource : Type → Prop)
variable (NitrogenSource : Type → Prop)
variable (InorganicSalt : Type → Prop)
variable (bacteria_can_synthesize_urease : urea_decomposing_bacteria → Prop)

axiom KH2PO4_is_inorganic_salt : InorganicSalt KH2PO4
axiom Na2HPO4_is_inorganic_salt : InorganicSalt Na2HPO4
axiom MgSO4_7H2O_is_inorganic_salt : InorganicSalt MgSO4_7H2O
axiom urea_is_nitrogen_source : NitrogenSource urea

theorem isolate_urea_decomposing_bacteria_valid_option :
  (InorganicSalt KH2PO4) ∧
  (InorganicSalt Na2HPO4) ∧
  (InorganicSalt MgSO4_7H2O) ∧
  (NitrogenSource urea) ∧
  (CarbonSource glucose) → (∃ bacteria : urea_decomposing_bacteria, bacteria_can_synthesize_urease bacteria) := sorry

end isolate_urea_decomposing_bacteria_valid_option_l6_6006


namespace shop_earnings_correct_l6_6831

theorem shop_earnings_correct :
  let cola_price := 3
  let juice_price := 1.5
  let water_price := 1
  let cola_sold := 15
  let juice_sold := 12
  let water_sold := 25
  cola_price * cola_sold + juice_price * juice_sold + water_price * water_sold = 88 := 
by 
  sorry

end shop_earnings_correct_l6_6831


namespace shaded_area_l6_6156

theorem shaded_area 
  (side_of_square : ℝ)
  (arc_radius : ℝ)
  (side_length_eq_sqrt_two : side_of_square = Real.sqrt 2)
  (radius_eq_one : arc_radius = 1) :
  let square_area := 4
  let sector_area := 3 * Real.pi
  let shaded_area := square_area + sector_area
  shaded_area = 4 + 3 * Real.pi :=
by
  sorry

end shaded_area_l6_6156


namespace perpendicular_angles_l6_6208

theorem perpendicular_angles (α β : ℝ) (k : ℤ) : 
  (∃ k : ℤ, β - α = k * 360 + 90 ∨ β - α = k * 360 - 90) →
  β = k * 360 + α + 90 ∨ β = k * 360 + α - 90 :=
by
  sorry

end perpendicular_angles_l6_6208


namespace marble_boxes_l6_6495

theorem marble_boxes (m : ℕ) : 
  (720 % m = 0) ∧ (m > 1) ∧ (720 / m > 1) ↔ m = 28 := 
sorry

end marble_boxes_l6_6495


namespace quadratic_solution_value_l6_6625

open Real

theorem quadratic_solution_value (a b : ℝ) (h1 : 2 + b = -a) (h2 : 2 * b = -6) :
  (2 * a + b)^2023 = -1 :=
sorry

end quadratic_solution_value_l6_6625


namespace effective_rate_proof_l6_6995

noncomputable def nominal_rate : ℝ := 0.08
noncomputable def compounding_periods : ℕ := 2
noncomputable def effective_annual_rate (i : ℝ) (n : ℕ) : ℝ := (1 + i / n) ^ n - 1

theorem effective_rate_proof :
  effective_annual_rate nominal_rate compounding_periods = 0.0816 :=
by
  sorry

end effective_rate_proof_l6_6995


namespace perfect_squares_as_difference_l6_6093

theorem perfect_squares_as_difference (N : ℕ) (hN : N = 20000) : 
  (∃ (n : ℕ), n = 71 ∧ 
    ∀ m < N, 
      (∃ a b : ℤ, 
        a^2 = m ∧
        b^2 = m + ((b + 1)^2 - b^2) - 1 ∧ 
        (b + 1)^2 - b^2 = 2 * b + 1)) :=
by 
  sorry

end perfect_squares_as_difference_l6_6093


namespace total_plums_picked_l6_6064

-- Conditions
def Melanie_plums : ℕ := 4
def Dan_plums : ℕ := 9
def Sally_plums : ℕ := 3

-- Proof statement
theorem total_plums_picked : Melanie_plums + Dan_plums + Sally_plums = 16 := by
  sorry

end total_plums_picked_l6_6064


namespace total_voters_in_districts_l6_6079

theorem total_voters_in_districts : 
  ∀ (D1 D2 D3 : ℕ),
  (D1 = 322) →
  (D2 = D3 - 19) →
  (D3 = 2 * D1) →
  (D1 + D2 + D3 = 1591) :=
by
  intros D1 D2 D3 h1 h2 h3
  sorry

end total_voters_in_districts_l6_6079


namespace B_pow_2024_l6_6597

noncomputable def B : Matrix (Fin 3) (Fin 3) ℝ :=
  ![
    ![Real.cos (Real.pi / 4), 0, -Real.sin (Real.pi / 4)],
    ![0, 1, 0],
    ![Real.sin (Real.pi / 4), 0, Real.cos (Real.pi / 4)]
  ]

theorem B_pow_2024 :
  B ^ 2024 = ![
    ![-1, 0, 0],
    ![0, 1, 0],
    ![0, 0, -1]
  ] :=
by
  sorry

end B_pow_2024_l6_6597


namespace polygon_interior_exterior_angles_l6_6234

theorem polygon_interior_exterior_angles (n : ℕ) :
  (n - 2) * 180 = 360 + 720 → n = 8 := 
by {
  sorry
}

end polygon_interior_exterior_angles_l6_6234


namespace find_A_minus_C_l6_6365

/-- There are three different natural numbers A, B, and C. 
    When A + B = 84, B + C = 60, and A = 6B, find the value of A - C. -/
theorem find_A_minus_C (A B C : ℕ) 
  (h1 : A + B = 84) 
  (h2 : B + C = 60) 
  (h3 : A = 6 * B) 
  (h4 : A ≠ B) 
  (h5 : A ≠ C) 
  (h6 : B ≠ C) :
  A - C = 24 :=
sorry

end find_A_minus_C_l6_6365


namespace min_rubles_for_1001_l6_6644

def min_rubles_needed (n : ℕ) : ℕ :=
  let side_cells := (n + 1) * 4
  let inner_cells := (n - 1) * (n - 1)
  let total := inner_cells * 4 + side_cells
  total / 2 -- since each side is shared by two cells

theorem min_rubles_for_1001 : min_rubles_needed 1001 = 503000 := by
  sorry

end min_rubles_for_1001_l6_6644


namespace total_students_correct_l6_6716

-- Given conditions
def number_of_buses : ℕ := 95
def number_of_seats_per_bus : ℕ := 118

-- Definition for the total number of students
def total_number_of_students : ℕ := number_of_buses * number_of_seats_per_bus

-- Problem statement
theorem total_students_correct :
  total_number_of_students = 11210 :=
by
  -- Proof is omitted, hence we use sorry.
  sorry

end total_students_correct_l6_6716


namespace largest_multiple_of_7_smaller_than_neg_50_l6_6895

theorem largest_multiple_of_7_smaller_than_neg_50 : ∃ n, (∃ k : ℤ, n = 7 * k) ∧ n < -50 ∧ ∀ m, (∃ j : ℤ, m = 7 * j) ∧ m < -50 → m ≤ n :=
by
  sorry

end largest_multiple_of_7_smaller_than_neg_50_l6_6895


namespace profit_amount_l6_6280

-- Conditions: Selling Price and Profit Percentage
def SP : ℝ := 850
def P_percent : ℝ := 37.096774193548384

-- Theorem: The profit amount is $230
theorem profit_amount : (SP / (1 + P_percent / 100)) * P_percent / 100 = 230 := by
  -- sorry will be replaced with the proof
  sorry

end profit_amount_l6_6280


namespace solve_quadratic_l6_6395

theorem solve_quadratic (x : ℝ) : x^2 - 6*x + 5 = 0 ↔ x = 1 ∨ x = 5 := by
  sorry

end solve_quadratic_l6_6395


namespace red_tulips_l6_6565

theorem red_tulips (white_tulips : ℕ) (bouquets : ℕ)
  (hw : white_tulips = 21)
  (hb : bouquets = 7)
  (div_prop : ∀ n, white_tulips % n = 0 ↔ bouquets % n = 0) : 
  ∃ red_tulips : ℕ, red_tulips = 7 :=
by
  sorry

end red_tulips_l6_6565


namespace semicircle_problem_l6_6066

open Real

theorem semicircle_problem (r : ℝ) (N : ℕ)
  (h1 : True) -- condition 1: There are N small semicircles each with radius r.
  (h2 : True) -- condition 2: The diameter of the large semicircle is 2Nr.
  (h3 : (N * (π * r^2) / 2) / ((π * (N^2 * r^2) / 2) - (N * (π * r^2) / 2)) = (1 : ℝ) / 12) -- given ratio A / B = 1 / 12 
  : N = 13 :=
sorry

end semicircle_problem_l6_6066


namespace distance_between_points_on_parabola_l6_6256

theorem distance_between_points_on_parabola (x1 y1 x2 y2 : ℝ) 
  (h_parabola : ∀ (x : ℝ), 4 * ((x^2)/4) = x^2) 
  (h_focus : F = (0, 1))
  (h_line : y1 = k * x1 + 1 ∧ y2 = k * x2 + 1)
  (h_intersects : x1^2 = 4 * y1 ∧ x2^2 = 4 * y2)
  (h_y_sum : y1 + y2 = 6) :
  |dist (x1, y1) (x2, y2)| = 8 := sorry

end distance_between_points_on_parabola_l6_6256


namespace maximum_x_plus_7y_exists_Q_locus_l6_6667

noncomputable def Q_locus (x y : ℝ) : Prop :=
  (x - 1)^2 + (y - 1)^2 = 2

theorem maximum_x_plus_7y (M : ℝ × ℝ) (h : Q_locus M.fst M.snd) : 
  ∃ max_value, max_value = 18 :=
  sorry

theorem exists_Q_locus (x y : ℝ) : 
  (∃ (Q : ℝ × ℝ), Q_locus Q.fst Q.snd) :=
  sorry

end maximum_x_plus_7y_exists_Q_locus_l6_6667


namespace walkway_area_correct_l6_6691

-- Define the dimensions and conditions
def bed_width : ℝ := 4
def bed_height : ℝ := 3
def walkway_width : ℝ := 2
def num_rows : ℕ := 4
def num_columns : ℕ := 3
def num_beds : ℕ := num_rows * num_columns

-- Total dimensions of garden including walkways
def total_width : ℝ := (num_columns * bed_width) + ((num_columns + 1) * walkway_width)
def total_height : ℝ := (num_rows * bed_height) + ((num_rows + 1) * walkway_width)

-- Areas
def total_garden_area : ℝ := total_width * total_height
def total_bed_area : ℝ := (bed_width * bed_height) * num_beds

-- Correct answer we want to prove
def walkway_area : ℝ := total_garden_area - total_bed_area

theorem walkway_area_correct : walkway_area = 296 := by
  sorry

end walkway_area_correct_l6_6691


namespace complex_coordinate_l6_6398

theorem complex_coordinate (i : ℂ) (h : i * i = -1) : i * (1 - i) = 1 + i :=
by sorry

end complex_coordinate_l6_6398


namespace cylinder_radius_and_remaining_space_l6_6046

theorem cylinder_radius_and_remaining_space 
  (cone_radius : ℝ) (cone_height : ℝ) 
  (cylinder_radius : ℝ) (cylinder_height : ℝ) :
  cone_radius = 8 →
  cone_height = 20 →
  cylinder_height = 2 * cylinder_radius →
  (20 - 2 * cylinder_radius) / cylinder_radius = 20 / 8 →
  (cylinder_radius = 40 / 9 ∧ (cone_height - cylinder_height) = 100 / 9) :=
by
  intros cone_radius_8 cone_height_20 cylinder_height_def similarity_eq
  sorry

end cylinder_radius_and_remaining_space_l6_6046


namespace ratio_of_area_of_shaded_square_l6_6607

theorem ratio_of_area_of_shaded_square 
  (large_square : Type) 
  (smaller_squares : Finset large_square) 
  (area_large_square : ℝ) 
  (area_smaller_square : ℝ) 
  (h_division : smaller_squares.card = 25)
  (h_equal_area : ∀ s ∈ smaller_squares, area_smaller_square = (area_large_square / 25))
  (shaded_square : Finset large_square)
  (h_shaded_sub : shaded_square ⊆ smaller_squares)
  (h_shaded_card : shaded_square.card = 5) :
  (5 * area_smaller_square) / area_large_square = 1 / 5 := 
by
  sorry

end ratio_of_area_of_shaded_square_l6_6607


namespace sum_inequality_l6_6124

open Real

theorem sum_inequality (a b c : ℝ) (h : a + b + c = 3) :
  (1 / (5 * a^2 - 4 * a + 11) + 1 / (5 * b^2 - 4 * b + 11) + 1 / (5 * c^2 - 4 * c + 11)) ≤ 1 / 4 :=
by
  sorry

end sum_inequality_l6_6124


namespace percent_students_both_correct_l6_6276

def percent_answered_both_questions (total_students first_correct second_correct neither_correct : ℕ) : ℕ :=
  let at_least_one_correct := total_students - neither_correct
  let total_individual_correct := first_correct + second_correct
  total_individual_correct - at_least_one_correct

theorem percent_students_both_correct
  (total_students : ℕ)
  (first_question_correct : ℕ)
  (second_question_correct : ℕ)
  (neither_question_correct : ℕ) 
  (h_total_students : total_students = 100)
  (h_first_correct : first_question_correct = 80)
  (h_second_correct : second_question_correct = 55)
  (h_neither_correct : neither_question_correct = 20) :
  percent_answered_both_questions total_students first_question_correct second_question_correct neither_question_correct = 55 :=
by
  rw [h_total_students, h_first_correct, h_second_correct, h_neither_correct]
  sorry


end percent_students_both_correct_l6_6276


namespace neither_plaid_nor_purple_l6_6660

-- Definitions and given conditions:
def total_shirts := 5
def total_pants := 24
def plaid_shirts := 3
def purple_pants := 5

-- Proof statement:
theorem neither_plaid_nor_purple : 
  (total_shirts - plaid_shirts) + (total_pants - purple_pants) = 21 := 
by 
  -- Mark proof steps with sorry
  sorry

end neither_plaid_nor_purple_l6_6660


namespace B_wins_four_rounds_prob_is_0_09_C_wins_three_rounds_prob_is_0_162_l6_6838

namespace GoGame

-- Define the players: A, B, C
inductive Player
| A
| B
| C

open Player

-- Define the probabilities as given
def P_A_beats_B : ℝ := 0.4
def P_B_beats_C : ℝ := 0.5
def P_C_beats_A : ℝ := 0.6

-- Define the game rounds and logic
def probability_B_winning_four_rounds 
  (P_A_beats_B : ℝ) (P_B_beats_C : ℝ) (P_C_beats_A : ℝ) : ℝ :=
(1 - P_A_beats_B)^2 * P_B_beats_C^2

def probability_C_winning_three_rounds 
  (P_A_beats_B : ℝ) (P_B_beats_C : ℝ) (P_C_beats_A : ℝ) : ℝ :=
  P_A_beats_B * P_C_beats_A^2 * P_B_beats_C + 
  (1 - P_A_beats_B) * P_B_beats_C^2 * P_C_beats_A

-- Proof statements
theorem B_wins_four_rounds_prob_is_0_09 : 
  probability_B_winning_four_rounds P_A_beats_B P_B_beats_C P_C_beats_A = 0.09 :=
by
  sorry

theorem C_wins_three_rounds_prob_is_0_162 : 
  probability_C_winning_three_rounds P_A_beats_B P_B_beats_C P_C_beats_A = 0.162 :=
by
  sorry

end GoGame

end B_wins_four_rounds_prob_is_0_09_C_wins_three_rounds_prob_is_0_162_l6_6838


namespace johns_age_l6_6886

theorem johns_age (j d : ℕ) (h1 : j = d - 30) (h2 : j + d = 80) : j = 25 :=
by
  sorry

end johns_age_l6_6886


namespace smallest_x_l6_6344

theorem smallest_x (x : ℕ) (h : 67 * 89 * x % 35 = 0) : x = 35 := 
by sorry

end smallest_x_l6_6344


namespace change_correct_l6_6947

def cost_gum : ℕ := 350
def cost_protractor : ℕ := 500
def amount_paid : ℕ := 1000

theorem change_correct : amount_paid - (cost_gum + cost_protractor) = 150 := by
  sorry

end change_correct_l6_6947


namespace intersection_A_B_l6_6839

-- Define the set A
def A : Set ℕ := {1, 2, 3, 4, 5}

-- Define the set B as the set of natural numbers greater than 2.5
def B : Set ℕ := {x : ℕ | 2 * x > 5}

-- Prove that the intersection of A and B is {3, 4, 5}
theorem intersection_A_B : A ∩ B = {3, 4, 5} :=
by sorry

end intersection_A_B_l6_6839


namespace problem_proof_l6_6333

-- Define positive integers and the conditions given in the problem
variables {p q r s : ℕ}

-- The product of the four integers is 7!
axiom product_of_integers : p * q * r * s = 5040  -- 7! = 5040

-- The equations defining the relationships
axiom equation1 : p * q + p + q = 715
axiom equation2 : q * r + q + r = 209
axiom equation3 : r * s + r + s = 143

-- The goal is to prove p - s = 10
theorem problem_proof : p - s = 10 :=
sorry

end problem_proof_l6_6333


namespace problem1_solutionset_problem2_minvalue_l6_6556

noncomputable def f (x : ℝ) : ℝ := 45 * abs (2 * x - 1)
noncomputable def g (x : ℝ) : ℝ := f x + f (x - 1)

theorem problem1_solutionset :
  {x : ℝ | 0 < x ∧ x < 2 / 3} = {x : ℝ | f x + abs (x + 1) < 2} :=
by
  sorry

theorem problem2_minvalue (a : ℝ) (m n : ℝ) (h : m + n = a ∧ m > 0 ∧ n > 0) :
  a = 2 → (4 / m + 1 / n) ≥ 9 / 2 :=
by
  sorry

end problem1_solutionset_problem2_minvalue_l6_6556


namespace complex_division_l6_6799

theorem complex_division (z1 z2 : ℂ) (h1 : z1 = 1 + 1 * Complex.I) (h2 : z2 = 0 + 2 * Complex.I) :
  z2 / z1 = 1 + Complex.I :=
by
  sorry

end complex_division_l6_6799


namespace factor_expression_l6_6225

theorem factor_expression (b : ℝ) : 
  (8 * b^4 - 100 * b^3 + 18) - (3 * b^4 - 11 * b^3 + 18) = b^3 * (5 * b - 89) :=
by
  sorry

end factor_expression_l6_6225


namespace prob_win_3_1_correct_l6_6342

-- Defining the probability for winning a game
def prob_win_game : ℚ := 2 / 3

-- Defining the probability for losing a game
def prob_lose_game : ℚ := 1 - prob_win_game

-- A function to calculate the probability of winning the match with a 3:1 score
def prob_win_3_1 : ℚ :=
  let combinations := 3 -- Number of ways to lose exactly 1 game in the first 3 games (C_3^1)
  let win_prob := prob_win_game ^ 3 -- Probability for winning 3 games
  let lose_prob := prob_lose_game -- Probability for losing 1 game
  combinations * win_prob * lose_prob

-- The theorem that states the probability that player A wins with a score of 3:1
theorem prob_win_3_1_correct : prob_win_3_1 = 8 / 27 := by
  sorry

end prob_win_3_1_correct_l6_6342


namespace total_cost_of_items_l6_6487

variable (M R F : ℝ)
variable (h1 : 10 * M = 24 * R)
variable (h2 : F = 2 * R)
variable (h3 : F = 21)

theorem total_cost_of_items : 4 * M + 3 * R + 5 * F = 237.3 :=
by
  sorry

end total_cost_of_items_l6_6487


namespace election_1002nd_k_election_1001st_k_l6_6235

variable (k : ℕ)

noncomputable def election_in_1002nd_round_max_k : Prop :=
  ∀ (n : ℕ), (n = 2002) → (k ≤ n - 1) → k = 2001 → -- The conditions include the number of candidates 'n', and specifying that 'k' being the maximum initially means k ≤ 2001.
  true

noncomputable def election_in_1001st_round_max_k : Prop :=
  ∀ (n : ℕ), (n = 2002) → (k ≤ n - 1) → k = 1 → -- Similarly, these conditions specify the initial maximum placement as 1 when elected in 1001st round.
  true

-- Definitions specifying the problem to identify max k for given rounds
theorem election_1002nd_k : election_in_1002nd_round_max_k k := sorry

theorem election_1001st_k : election_in_1001st_round_max_k k := sorry

end election_1002nd_k_election_1001st_k_l6_6235


namespace Keith_initial_picked_l6_6829

-- Definitions based on the given conditions
def Mike_picked := 12
def Keith_gave_away := 46
def remaining_pears := 13

-- Question: Prove that Keith initially picked 47 pears.
theorem Keith_initial_picked :
  ∃ K : ℕ, K = 47 ∧ (K - Keith_gave_away + Mike_picked = remaining_pears) :=
sorry

end Keith_initial_picked_l6_6829


namespace percent_increase_l6_6181

theorem percent_increase (N : ℝ) (h : (1 / 7) * N = 1) : 
  N = 7 ∧ (N - (4 / 7)) / (4 / 7) * 100 = 1125.0000000000002 := 
by 
  sorry

end percent_increase_l6_6181


namespace polygon_properties_l6_6057

theorem polygon_properties
  (n : ℕ)
  (h_exterior_angle : 360 / 20 = n)
  (h_n_sides : n = 18) :
  (180 * (n - 2) = 2880) ∧ (n * (n - 3) / 2 = 135) :=
by
  sorry

end polygon_properties_l6_6057


namespace bc_sum_eq_twelve_l6_6098

theorem bc_sum_eq_twelve (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hb_lt : b < 12) (hc_lt : c < 12) 
  (h_eq : (12 * a + b) * (12 * a + c) = 144 * a * (a + 1) + b * c) : b + c = 12 :=
by
  sorry

end bc_sum_eq_twelve_l6_6098


namespace pond_length_l6_6754

theorem pond_length (V W D L : ℝ) (hV : V = 1600) (hW : W = 10) (hD : D = 8) :
  L = 20 ↔ V = L * W * D :=
by
  sorry

end pond_length_l6_6754


namespace gcd_60_75_l6_6608

theorem gcd_60_75 : Nat.gcd 60 75 = 15 := by
  sorry

end gcd_60_75_l6_6608


namespace ferris_wheel_seats_l6_6420

theorem ferris_wheel_seats (total_people : ℕ) (people_per_seat : ℕ) (h1 : total_people = 16) (h2 : people_per_seat = 4) : (total_people / people_per_seat) = 4 := by
  sorry

end ferris_wheel_seats_l6_6420


namespace sum_of_coordinates_l6_6396

noncomputable def endpoint_x (x : ℤ) := (-3 + x) / 2 = 2
noncomputable def endpoint_y (y : ℤ) := (-15 + y) / 2 = -5

theorem sum_of_coordinates : ∃ x y : ℤ, endpoint_x x ∧ endpoint_y y ∧ x + y = 12 :=
by
  sorry

end sum_of_coordinates_l6_6396


namespace angle_with_same_terminal_side_315_l6_6587

def same_terminal_side (α β : ℝ) : Prop :=
  ∃ k : ℤ, α = k * 360 + β

theorem angle_with_same_terminal_side_315:
  same_terminal_side (-45) 315 :=
by
  sorry

end angle_with_same_terminal_side_315_l6_6587


namespace arithmetic_expression_evaluation_l6_6187

theorem arithmetic_expression_evaluation : 
  2000 - 80 + 200 - 120 = 2000 := by
  sorry

end arithmetic_expression_evaluation_l6_6187


namespace polynomial_bound_l6_6559

noncomputable def P (x : ℝ) : ℝ := sorry  -- Placeholder for the polynomial P(x)

theorem polynomial_bound (n : ℕ) (hP : ∀ y : ℝ, 0 ≤ y ∧ y ≤ 1 → |P y| ≤ 1) :
  P (-1 / n) ≤ 2^(n + 1) - 1 :=
sorry

end polynomial_bound_l6_6559


namespace max_m_l6_6014

noncomputable def f (x a : ℝ) : ℝ := 2 ^ |x + a|

theorem max_m (a m : ℝ) (H1 : ∀ x, f (3 + x) a = f (3 - x) a) 
(H2 : ∀ x y, x ≤ y → y ≤ m → f x a ≥ f y a) : 
  m = 3 :=
by
  sorry

end max_m_l6_6014


namespace circle_eq_of_points_value_of_m_l6_6118

-- Define the points on the circle
def P : ℝ × ℝ := (0, -4)
def Q : ℝ × ℝ := (2, 0)
def R : ℝ × ℝ := (3, -1)

-- Statement 1: The equation of the circle passing through P, Q, and R
theorem circle_eq_of_points (C : ℝ × ℝ → Prop) :
  (C P ∧ C Q ∧ C R) ↔ ∀ x y : ℝ, C (x, y) ↔ (x - 1)^2 + (y + 2)^2 = 5 := sorry

-- Define the line intersecting the circle and the chord length condition |AB| = 4
def line_l (m : ℝ) (x y : ℝ) : Prop := m * x + y - 1 = 0

-- Statement 2: The value of m such that the chord length |AB| is 4
theorem value_of_m (m : ℝ) :
  (∃ A B : ℝ × ℝ, line_l m A.1 A.2 ∧ line_l m B.1 B.2 ∧
  ((A.1 - B.1)^2 + (A.2 - B.2)^2 = 16)) → m = 4 / 3 := sorry

end circle_eq_of_points_value_of_m_l6_6118


namespace probability_at_least_one_hit_l6_6865

theorem probability_at_least_one_hit (pA pB pC : ℝ) (hA : pA = 0.7) (hB : pB = 0.5) (hC : pC = 0.4) : 
  (1 - ((1 - pA) * (1 - pB) * (1 - pC))) = 0.91 :=
by
  sorry

end probability_at_least_one_hit_l6_6865


namespace eccentricity_of_ellipse_l6_6270

theorem eccentricity_of_ellipse 
  (a b : ℝ) (e : ℝ) (h1 : a > b) (h2 : b > 0) 
  (h3 : ∃ (x y : ℝ), x = 0 ∧ y > 0 ∧ (9 * b^2 = 16/7 * a^2)) :
  e = Real.sqrt (10) / 6 :=
sorry

end eccentricity_of_ellipse_l6_6270


namespace area_of_triangle_PF1F2_l6_6376

noncomputable def ellipse := {P : ℝ × ℝ // (4 * P.1^2) / 49 + (P.2^2) / 6 = 1}

noncomputable def area_triangle (P F1 F2 : ℝ × ℝ) :=
  1 / 2 * abs ((F1.1 - P.1) * (F2.2 - P.2) - (F1.2 - P.2) * (F2.1 - P.1))

theorem area_of_triangle_PF1F2 :
  ∀ (F1 F2 : ℝ × ℝ) (P : ellipse), 
    (dist P.1 F1 = 4) →
    (dist P.1 F2 = 3) →
    (dist F1 F2 = 5) →
    area_triangle P.1 F1 F2 = 6 :=
by sorry

end area_of_triangle_PF1F2_l6_6376


namespace shaded_fraction_is_half_l6_6844

-- Define the number of rows and columns in the grid
def num_rows : ℕ := 8
def num_columns : ℕ := 8

-- Define the number of shaded triangles based on the pattern explained
def shaded_rows : List ℕ := [1, 3, 5, 7]
def num_shaded_rows : ℕ := 4
def triangles_per_row : ℕ := num_columns
def num_shaded_triangles : ℕ := num_shaded_rows * triangles_per_row

-- Define the total number of triangles
def total_triangles : ℕ := num_rows * num_columns

-- Define the fraction of shaded triangles
def shaded_fraction : ℚ := num_shaded_triangles / total_triangles

-- Prove the shaded fraction is 1/2
theorem shaded_fraction_is_half : shaded_fraction = 1 / 2 :=
by
  -- Provide the calculations
  sorry

end shaded_fraction_is_half_l6_6844


namespace carriages_people_equation_l6_6618

theorem carriages_people_equation (x : ℕ) :
  3 * (x - 2) = 2 * x + 9 :=
sorry

end carriages_people_equation_l6_6618


namespace range_of_c_value_of_c_given_perimeter_l6_6509

variables (a b c : ℝ)

-- Question 1: Proving the range of values for c
theorem range_of_c (h1 : a + b = 3 * c - 2) (h2 : a - b = 2 * c - 6) :
  1 < c ∧ c < 6 :=
sorry

-- Question 2: Finding the value of c for a given perimeter
theorem value_of_c_given_perimeter (h1 : a + b = 3 * c - 2) (h2 : a - b = 2 * c - 6) (h3 : a + b + c = 18) :
  c = 5 :=
sorry

end range_of_c_value_of_c_given_perimeter_l6_6509


namespace larger_group_men_count_l6_6756

-- Define the conditions
def total_man_days (men : ℕ) (days : ℕ) : ℕ := men * days

-- Define the total work for 36 men in 18 days
def work_by_36_men_in_18_days : ℕ := total_man_days 36 18

-- Define the number of days the larger group takes
def days_for_larger_group : ℕ := 8

-- Problem Statement: Prove that if 36 men take 18 days to complete the work, and a larger group takes 8 days, then the larger group consists of 81 men.
theorem larger_group_men_count : 
  ∃ (M : ℕ), total_man_days M days_for_larger_group = work_by_36_men_in_18_days ∧ M = 81 := 
by
  -- Here would go the proof steps
  sorry

end larger_group_men_count_l6_6756


namespace spaceship_initial_people_count_l6_6392

/-- For every 100 additional people that board a spaceship, its speed is halved.
     The speed of the spaceship with a certain number of people on board is 500 km per hour.
     The speed of the spaceship when there are 400 people on board is 125 km/hr.
     Prove that the number of people on board when the spaceship was moving at 500 km/hr is 200. -/
theorem spaceship_initial_people_count (speed : ℕ → ℕ) (n : ℕ) :
  (∀ k, speed (k + 100) = speed k / 2) →
  speed n = 500 →
  speed 400 = 125 →
  n = 200 :=
by
  intro half_speed speed_500 speed_400
  sorry

end spaceship_initial_people_count_l6_6392


namespace smallest_denominator_is_168_l6_6184

theorem smallest_denominator_is_168 (a b : ℕ) (h1: Nat.gcd a 600 = 1) (h2: Nat.gcd b 700 = 1) :
  ∃ k, Nat.gcd (7 * a + 6 * b) 4200 = k ∧ k = 25 ∧ (4200 / k) = 168 :=
sorry

end smallest_denominator_is_168_l6_6184


namespace min_value_prime_factorization_l6_6796

/-- Let x and y be positive integers and assume 5 * x ^ 7 = 13 * y ^ 11.
  If x has a prime factorization of the form a ^ c * b ^ d, then the minimum possible value of a + b + c + d is 31. -/
theorem min_value_prime_factorization (x y a b c d : ℕ) (hx_pos : x > 0) (hy_pos: y > 0) (ha_pos : a > 0) (hb_pos : b > 0) (hc_pos: c > 0) (hd_pos: d > 0)
    (h_eq : 5 * x ^ 7 = 13 * y ^ 11) (h_fact : x = a^c * b^d) : a + b + c + d = 31 :=
by
  sorry

end min_value_prime_factorization_l6_6796


namespace smaller_prime_factor_l6_6698

theorem smaller_prime_factor (a b : ℕ) (prime_a : Nat.Prime a) (prime_b : Nat.Prime b) (distinct : a ≠ b)
  (product : a * b = 316990099009901) :
  min a b = 4002001 :=
  sorry

end smaller_prime_factor_l6_6698


namespace volume_is_85_l6_6680

/-!
# Proof Problem
Prove that the total volume of Carl's and Kate's cubes is 85, given the conditions,
Carl has 3 cubes each with a side length of 3, and Kate has 4 cubes each with a side length of 1.
-/

-- Definitions for the problem conditions:
def volume_of_cube (s : ℕ) : ℕ := s^3

def total_volume (n : ℕ) (s : ℕ) : ℕ := n * volume_of_cube s

-- Given conditions
def carls_cubes_volume : ℕ := total_volume 3 3
def kates_cubes_volume : ℕ := total_volume 4 1

-- The total volume of Carl's and Kate's cubes:
def total_combined_volume : ℕ := carls_cubes_volume + kates_cubes_volume

-- Prove the total volume is 85
theorem volume_is_85 : total_combined_volume = 85 :=
by sorry

end volume_is_85_l6_6680


namespace complex_number_real_imaginary_opposite_l6_6866

theorem complex_number_real_imaginary_opposite (a : ℝ) (i : ℂ) (comp : z = (1 - a * i) * i):
  (z.re = -z.im) → a = 1 :=
by 
  sorry

end complex_number_real_imaginary_opposite_l6_6866


namespace mrs_jensens_preschool_l6_6630

theorem mrs_jensens_preschool (total_students students_with_both students_with_neither students_with_green_eyes students_with_red_hair : ℕ) 
(h1 : total_students = 40) 
(h2 : students_with_red_hair = 3 * students_with_green_eyes) 
(h3 : students_with_both = 8) 
(h4 : students_with_neither = 4) :
students_with_green_eyes = 12 := 
sorry

end mrs_jensens_preschool_l6_6630


namespace max_abs_cubic_at_least_one_fourth_l6_6853

def cubic_polynomial (p q r x : ℝ) : ℝ := x^3 + p * x^2 + q * x + r

theorem max_abs_cubic_at_least_one_fourth (p q r : ℝ) :
  ∃ x ∈ Set.Icc (-1 : ℝ) 1, |cubic_polynomial p q r x| ≥ 1 / 4 :=
by
  sorry

end max_abs_cubic_at_least_one_fourth_l6_6853


namespace sum_of_square_areas_l6_6238

variable (WX XZ : ℝ)

theorem sum_of_square_areas (hW : WX = 15) (hX : XZ = 20) : WX^2 + XZ^2 = 625 := by
  sorry

end sum_of_square_areas_l6_6238


namespace equivalent_weeks_l6_6250

def hoursPerDay := 24
def daysPerWeek := 7
def hoursPerWeek := daysPerWeek * hoursPerDay
def totalHours := 2016

theorem equivalent_weeks : totalHours / hoursPerWeek = 12 := 
by
  sorry

end equivalent_weeks_l6_6250


namespace remainder_of_sum_l6_6380

theorem remainder_of_sum (f y z : ℤ) 
  (hf : f % 5 = 3)
  (hy : y % 5 = 4)
  (hz : z % 7 = 6) : 
  (f + y + z) % 35 = 13 :=
by
  sorry

end remainder_of_sum_l6_6380


namespace find_smallest_integer_y_l6_6805

theorem find_smallest_integer_y : ∃ y : ℤ, (8 / 12 : ℚ) < (y / 15) ∧ ∀ z : ℤ, z < y → ¬ ((8 / 12 : ℚ) < (z / 15)) :=
by
  sorry

end find_smallest_integer_y_l6_6805


namespace incorrect_transformation_D_l6_6786

theorem incorrect_transformation_D (x : ℝ) (hx1 : x + 1 ≠ 0) : 
  (2 - x) / (x + 1) ≠ (x - 2) / (1 + x) := 
by 
  sorry

end incorrect_transformation_D_l6_6786


namespace vertical_distance_from_top_to_bottom_l6_6454

-- Conditions
def ring_thickness : ℕ := 2
def largest_ring_diameter : ℕ := 18
def smallest_ring_diameter : ℕ := 4

-- Additional definitions based on the problem context
def count_rings : ℕ := (largest_ring_diameter - smallest_ring_diameter) / ring_thickness + 1
def inner_diameters_sum : ℕ := count_rings * (largest_ring_diameter - ring_thickness + smallest_ring_diameter) / 2
def vertical_distance : ℕ := inner_diameters_sum + 2 * ring_thickness

-- The problem statement to prove
theorem vertical_distance_from_top_to_bottom :
  vertical_distance = 76 := by
  sorry

end vertical_distance_from_top_to_bottom_l6_6454


namespace second_polygon_sides_l6_6261

theorem second_polygon_sides 
  (s : ℝ) -- side length of the second polygon
  (n1 n2 : ℕ) -- n1 = number of sides of the first polygon, n2 = number of sides of the second polygon
  (h1 : n1 = 40) -- first polygon has 40 sides
  (h2 : ∀ s1 s2 : ℝ, s1 = 3 * s2 → n1 * s1 = n2 * s2 → n2 = 120)
  : n2 = 120 := 
by
  sorry

end second_polygon_sides_l6_6261


namespace ab_plus_cd_value_l6_6868

theorem ab_plus_cd_value (a b c d : ℝ)
  (h1 : a + b + c = 5)
  (h2 : a + b + d = 1)
  (h3 : a + c + d = 12)
  (h4 : b + c + d = 7) :
  a * b + c * d = 176 / 9 := 
sorry

end ab_plus_cd_value_l6_6868


namespace blue_notebook_cost_l6_6989

theorem blue_notebook_cost
  (total_spent : ℕ)
  (total_notebooks : ℕ)
  (red_notebooks : ℕ)
  (red_notebook_cost : ℕ)
  (green_notebooks : ℕ)
  (green_notebook_cost : ℕ)
  (blue_notebook_cost : ℕ)
  (h₀ : total_spent = 37)
  (h₁ : total_notebooks = 12)
  (h₂ : red_notebooks = 3)
  (h₃ : red_notebook_cost = 4)
  (h₄ : green_notebooks = 2)
  (h₅ : green_notebook_cost = 2)
  (h₆ : total_spent = red_notebooks * red_notebook_cost + green_notebooks * green_notebook_cost + blue_notebook_cost * (total_notebooks - red_notebooks - green_notebooks)) :
  blue_notebook_cost = 3 := by
  sorry

end blue_notebook_cost_l6_6989


namespace cos_double_angle_l6_6355

variable (θ : ℝ)

theorem cos_double_angle (h : Real.tan (θ + Real.pi / 4) = 3) : Real.cos (2 * θ) = 3 / 5 :=
sorry

end cos_double_angle_l6_6355


namespace baby_grasshoppers_l6_6158

-- Definition for the number of grasshoppers on the plant
def grasshoppers_on_plant : ℕ := 7

-- Definition for the total number of grasshoppers found
def total_grasshoppers : ℕ := 31

-- The theorem to prove the number of baby grasshoppers under the plant
theorem baby_grasshoppers : 
  (total_grasshoppers - grasshoppers_on_plant) = 24 := 
by
  sorry

end baby_grasshoppers_l6_6158


namespace smallest_integer_neither_prime_nor_square_with_prime_factors_ge_60_l6_6450

def is_not_prime (n : ℕ) := ¬ Prime n
def is_not_square (n : ℕ) := ∀ m : ℕ, m * m ≠ n
def no_prime_factors_less_than (n k : ℕ) := ∀ p : ℕ, Prime p → p < k → ¬ p ∣ n
def smallest_integer_prop (n : ℕ) := is_not_prime n ∧ is_not_square n ∧ no_prime_factors_less_than n 60

theorem smallest_integer_neither_prime_nor_square_with_prime_factors_ge_60 : ∃ n : ℕ, smallest_integer_prop n ∧ n = 4087 :=
by
  sorry

end smallest_integer_neither_prime_nor_square_with_prime_factors_ge_60_l6_6450


namespace curve_is_line_l6_6084

noncomputable def polar_to_cartesian (r : ℝ) (θ : ℝ) : ℝ × ℝ :=
  let x := r * Real.cos θ
  let y := r * Real.sin θ
  (x, y)

theorem curve_is_line (r : ℝ) (θ : ℝ) :
  r = 1 / (Real.sin θ + Real.cos θ) ↔ ∃ (x y : ℝ), (x, y) = polar_to_cartesian r θ ∧ (x + y)^2 = 1 :=
by 
  sorry

end curve_is_line_l6_6084


namespace area_bounded_region_l6_6650

theorem area_bounded_region :
  (∃ (x y : ℝ), y^2 + 2 * x * y + 50 * |x| = 500) →
  ∃ (area : ℝ), area = 1250 :=
by
  sorry

end area_bounded_region_l6_6650


namespace tom_hockey_games_l6_6951

theorem tom_hockey_games (g_this_year g_last_year : ℕ) 
  (h1 : g_this_year = 4)
  (h2 : g_last_year = 9) 
  : g_this_year + g_last_year = 13 := 
by
  sorry

end tom_hockey_games_l6_6951


namespace sequence_integral_terms_l6_6161

theorem sequence_integral_terms (x : ℕ → ℝ) (h1 : ∀ n, x n ≠ 0)
  (h2 : ∀ n > 2, x n = (x (n - 2) * x (n - 1)) / (2 * x (n - 2) - x (n - 1))) :
  (∀ n, ∃ k : ℤ, x n = k) → x 1 = x 2 :=
by
  sorry

end sequence_integral_terms_l6_6161


namespace parallel_vectors_x_value_l6_6086

-- Define the vectors a and b
def a : ℝ × ℝ := (2, 3)
def b (x : ℝ) : ℝ × ℝ := (6, x)

-- Define what it means for vectors to be parallel (they are proportional)
def parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

-- Theorem to prove
theorem parallel_vectors_x_value :
  ∀ x : ℝ, parallel a (b x) → x = 9 :=
by
  intros x h
  sorry

end parallel_vectors_x_value_l6_6086


namespace option_A_incorrect_option_B_incorrect_option_C_incorrect_option_D_correct_l6_6373

theorem option_A_incorrect : ¬(Real.sqrt 2 + Real.sqrt 6 = Real.sqrt 8) :=
by sorry

theorem option_B_incorrect : ¬(6 * Real.sqrt 3 - 2 * Real.sqrt 3 = 4) :=
by sorry

theorem option_C_incorrect : ¬(4 * Real.sqrt 2 * 2 * Real.sqrt 3 = 6 * Real.sqrt 6) :=
by sorry

theorem option_D_correct : (1 / (2 - Real.sqrt 3) = 2 + Real.sqrt 3) :=
by sorry

end option_A_incorrect_option_B_incorrect_option_C_incorrect_option_D_correct_l6_6373


namespace division_problem_l6_6381

theorem division_problem : 160 / (10 + 11 * 2) = 5 := 
  by 
    sorry

end division_problem_l6_6381


namespace ellipse_foci_y_axis_l6_6849

theorem ellipse_foci_y_axis (k : ℝ) (h_eq : ∀ x y : ℝ, x^2 + k * y^2 = 2)
  (h_foci : ∀ x y : ℝ, x^2 ≤ 2 ∧ k * y^2 ≤ 2) :
  0 < k ∧ k < 1 :=
  sorry

end ellipse_foci_y_axis_l6_6849


namespace solve_for_x_l6_6329

theorem solve_for_x (m n : ℝ) (hm : m ≠ 0) (hn : n ≠ 0) (h_neq : m ≠ n) :
  ∃ x : ℝ, (x + 2 * m)^2 - (x - 3 * n)^2 = 9 * (m + n)^2 ↔
  x = (5 * m^2 + 18 * m * n + 18 * n^2) / (10 * m + 6 * n) := sorry

end solve_for_x_l6_6329


namespace total_fiscal_revenue_scientific_notation_l6_6645

theorem total_fiscal_revenue_scientific_notation : 
  ∃ a n, (1073 * 10^8 : ℝ) = a * 10^n ∧ (1 ≤ |a| ∧ |a| < 10) ∧ a = 1.07 ∧ n = 11 :=
by
  use 1.07, 11
  simp
  sorry

end total_fiscal_revenue_scientific_notation_l6_6645


namespace ball_hits_ground_time_l6_6131

theorem ball_hits_ground_time :
  ∃ t : ℝ, -20 * t^2 + 30 * t + 60 = 0 ∧ t = (3 + Real.sqrt 57) / 4 :=
by 
  sorry

end ball_hits_ground_time_l6_6131


namespace prime_p_geq_7_div_240_l6_6503

theorem prime_p_geq_7_div_240 (p : ℕ) (hp : Nat.Prime p) (hge7 : p ≥ 7) : 240 ∣ p^4 - 1 := 
sorry

end prime_p_geq_7_div_240_l6_6503


namespace probability_interval_contains_q_l6_6480

theorem probability_interval_contains_q (P_C P_D : ℝ) (q : ℝ)
    (hC : P_C = 5 / 7) (hD : P_D = 3 / 4) :
    (5 / 28 ≤ q ∧ q ≤ 5 / 7) ↔ (max (P_C + P_D - 1) 0 ≤ q ∧ q ≤ min P_C P_D) :=
by
  sorry

end probability_interval_contains_q_l6_6480


namespace amount_received_by_A_is_4_over_3_l6_6305

theorem amount_received_by_A_is_4_over_3
  (a d : ℚ)
  (h1 : a - 2 * d + a - d = a + (a + d) + (a + 2 * d))
  (h2 : 5 * a = 5) :
  a - 2 * d = 4 / 3 :=
by
  sorry

end amount_received_by_A_is_4_over_3_l6_6305


namespace one_more_square_possible_l6_6572

def grid_size : ℕ := 29
def total_cells : ℕ := grid_size * grid_size
def number_of_squares_removed : ℕ := 99
def cells_per_square : ℕ := 4
def total_removed_cells : ℕ := number_of_squares_removed * cells_per_square
def remaining_cells : ℕ := total_cells - total_removed_cells

theorem one_more_square_possible :
  remaining_cells ≥ cells_per_square :=
sorry

end one_more_square_possible_l6_6572


namespace solve_for_a_l6_6605

-- Define the sets M and N as given in the problem
def M : Set ℝ := {x : ℝ | x^2 + 6 * x - 16 = 0}
def N (a : ℝ) : Set ℝ := {x : ℝ | x * a - 3 = 0}

-- Define the proof statement
theorem solve_for_a (a : ℝ) : (N a ⊆ M) ↔ (a = 0 ∨ a = 3/2 ∨ a = -3/8) :=
by
  -- The proof would go here
  sorry

end solve_for_a_l6_6605


namespace raised_bed_section_area_l6_6137

theorem raised_bed_section_area :
  let length := 220
  let width := 120
  let total_area := length * width
  let half_area := total_area / 2
  let fraction_for_raised_beds := 2 / 3
  let area_of_raised_beds := half_area * fraction_for_raised_beds
  area_of_raised_beds = 8800 :=
by 
  let length := 220
  let width := 120
  let total_area := length * width
  let half_area := total_area / 2
  let fraction_for_raised_beds := 2 / 3
  let area_of_raised_beds := half_area * fraction_for_raised_beds
  show area_of_raised_beds = 8800
  sorry

end raised_bed_section_area_l6_6137


namespace lcm_of_numbers_l6_6313

theorem lcm_of_numbers (a b c d : ℕ) (h1 : a = 8) (h2 : b = 24) (h3 : c = 36) (h4 : d = 54) :
  Nat.lcm (Nat.lcm a b) (Nat.lcm c d) = 216 := 
by 
  sorry

end lcm_of_numbers_l6_6313


namespace computation_result_l6_6164

def a : ℕ := 3
def b : ℕ := 5
def c : ℕ := 7

theorem computation_result :
  (a + b + c) ^ 2 + (a ^ 2 + b ^ 2 + c ^ 2) = 308 := by
  sorry

end computation_result_l6_6164


namespace jeff_total_jars_l6_6931

theorem jeff_total_jars (x : ℕ) : 
  16 * x + 28 * x + 40 * x + 52 * x = 2032 → 4 * x = 56 :=
by
  intro h
  -- additional steps to solve the problem would go here.
  sorry

end jeff_total_jars_l6_6931


namespace chad_total_spend_on_ice_l6_6647

-- Define the given conditions
def num_people : ℕ := 15
def pounds_per_person : ℕ := 2
def pounds_per_bag : ℕ := 1
def price_per_pack : ℕ := 300 -- Price in cents to avoid floating-point issues
def bags_per_pack : ℕ := 10

-- The main statement to prove
theorem chad_total_spend_on_ice : 
  (num_people * pounds_per_person * 100 / (pounds_per_bag * bags_per_pack) * price_per_pack / 100 = 9) :=
by sorry

end chad_total_spend_on_ice_l6_6647


namespace oleg_can_find_adjacent_cells_divisible_by_4_l6_6437

theorem oleg_can_find_adjacent_cells_divisible_by_4 :
  ∀ (grid : Fin 22 → Fin 22 → ℕ),
  (∀ i j, 1 ≤ grid i j ∧ grid i j ≤ 22 * 22) →
  ∃ i j k l, ((i = k ∧ (j = l + 1 ∨ j = l - 1)) ∨ ((i = k + 1 ∨ i = k - 1) ∧ j = l)) ∧ ((grid i j + grid k l) % 4 = 0) :=
by
  sorry

end oleg_can_find_adjacent_cells_divisible_by_4_l6_6437


namespace negation_of_exists_cube_pos_l6_6047

theorem negation_of_exists_cube_pos :
  (¬ (∃ x : ℝ, x^3 > 0)) ↔ (∀ x : ℝ, x^3 ≤ 0) :=
by
  sorry

end negation_of_exists_cube_pos_l6_6047


namespace graph_is_two_lines_l6_6877

theorem graph_is_two_lines (x y : ℝ) : (x^2 - 25 * y^2 - 10 * x + 50 = 0) ↔
  (x = 5 + 5 * y) ∨ (x = 5 - 5 * y) :=
by
  sorry

end graph_is_two_lines_l6_6877


namespace probability_of_correct_match_l6_6372

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def total_possible_arrangements : ℕ :=
  factorial 4

def correct_arrangements : ℕ :=
  1

def probability_correct_match : ℚ :=
  correct_arrangements / total_possible_arrangements

theorem probability_of_correct_match : probability_correct_match = 1 / 24 :=
by
  -- Proof is omitted
  sorry

end probability_of_correct_match_l6_6372


namespace probability_at_least_75_cents_l6_6649

theorem probability_at_least_75_cents (p n d q c50 : Prop) 
  (Hp : p = tt ∨ p = ff)
  (Hn : n = tt ∨ n = ff)
  (Hd : d = tt ∨ d = ff)
  (Hq : q = tt ∨ q = ff)
  (Hc50 : c50 = tt ∨ c50 = ff) :
  (1 / 2 : ℝ) = 
  ((if c50 = tt then (if q = tt then 1 else 0) else 0) + 
  (if c50 = tt then 2^3 else 0)) / 2^5 :=
by sorry

end probability_at_least_75_cents_l6_6649


namespace shaded_region_area_is_15_l6_6681

noncomputable def area_of_shaded_region : ℝ :=
  let radius := 1
  let area_of_one_circle := Real.pi * (radius ^ 2)
  4 * area_of_one_circle + 3 * (4 - area_of_one_circle)

theorem shaded_region_area_is_15 : 
  abs (area_of_shaded_region - 15) < 1 :=
by
  exact sorry

end shaded_region_area_is_15_l6_6681


namespace mean_equal_l6_6742

theorem mean_equal (y : ℚ) :
  (5 + 10 + 20) / 3 = (15 + y) / 2 → y = 25 / 3 := 
by
  sorry

end mean_equal_l6_6742


namespace complex_number_calculation_l6_6441

theorem complex_number_calculation (z : ℂ) (hz : z = 1 - I) : (z^2 / (z - 1)) = 2 := by
  sorry

end complex_number_calculation_l6_6441
