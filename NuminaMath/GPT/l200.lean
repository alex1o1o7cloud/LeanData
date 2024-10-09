import Mathlib

namespace product_of_integer_with_100_l200_20098

theorem product_of_integer_with_100 (x : ℝ) (h : 10 * x = x + 37.89) : 100 * x = 421 :=
by
  -- insert the necessary steps to solve the problem
  sorry

end product_of_integer_with_100_l200_20098


namespace not_p_and_not_q_true_l200_20084

variable (p q: Prop)

theorem not_p_and_not_q_true (h1: ¬ (p ∧ q)) (h2: ¬ (p ∨ q)) : ¬ p ∧ ¬ q :=
by
  sorry

end not_p_and_not_q_true_l200_20084


namespace line_passes_through_fixed_point_l200_20043

theorem line_passes_through_fixed_point (k : ℝ) : (k * 2 - 1 + 1 - 2 * k = 0) :=
by
  sorry

end line_passes_through_fixed_point_l200_20043


namespace cortney_downloads_all_files_in_2_hours_l200_20092

theorem cortney_downloads_all_files_in_2_hours :
  let speed := 2 -- internet speed in megabits per minute
  let file1 := 80 -- file size in megabits
  let file2 := 90 -- file size in megabits
  let file3 := 70 -- file size in megabits
  let time1 := file1 / speed -- time to download first file in minutes
  let time2 := file2 / speed -- time to download second file in minutes
  let time3 := file3 / speed -- time to download third file in minutes
  let total_time_minutes := time1 + time2 + time3
  let total_time_hours := total_time_minutes / 60
  total_time_hours = 2 :=
by
  sorry

end cortney_downloads_all_files_in_2_hours_l200_20092


namespace toy_discount_price_l200_20059

theorem toy_discount_price (original_price : ℝ) (discount_rate : ℝ) (price_after_first_discount : ℝ) (price_after_second_discount : ℝ) : 
  original_price = 200 → 
  discount_rate = 0.1 →
  price_after_first_discount = original_price * (1 - discount_rate) →
  price_after_second_discount = price_after_first_discount * (1 - discount_rate) →
  price_after_second_discount = 162 :=
by
  intros h1 h2 h3 h4
  sorry

end toy_discount_price_l200_20059


namespace seating_arrangement_l200_20083

theorem seating_arrangement (boys girls : ℕ) (h_boys : boys = 4) (h_girls : girls = 3) : 
  (∃ n : ℕ, n = (boys.factorial * girls.factorial) + (girls.factorial * boys.factorial) ∧ n = 288) :=
by 
  sorry

end seating_arrangement_l200_20083


namespace sum_remainder_l200_20005

theorem sum_remainder (a b c d : ℤ) (h1 : a % 53 = 33) (h2 : b % 53 = 11) 
                       (h3 : c % 53 = 49) (h4 : d % 53 = 2) :
  (a + b + c + d) % 53 = 42 :=
sorry

end sum_remainder_l200_20005


namespace part_a_solutions_l200_20057

theorem part_a_solutions (x : ℝ) : (⌊x⌋^2 - x = -0.99) ↔ (x = 0.99 ∨ x = 1.99) :=
sorry

end part_a_solutions_l200_20057


namespace area_of_rectangle_l200_20015

theorem area_of_rectangle (w l : ℕ) (hw : w = 10) (hl : l = 2) : (w * l) = 20 :=
by
  sorry

end area_of_rectangle_l200_20015


namespace arithmetic_sequence_a5_value_l200_20050

theorem arithmetic_sequence_a5_value 
  (a : ℕ → ℝ) (d : ℝ)
  (h_seq : ∀ n, a n = a 1 + (n - 1) * d)
  (h_a2 : a 2 = 1)
  (h_a8 : a 8 = 2 * a 6 + a 4) : 
  a 5 = -1 / 2 :=
by
  sorry

end arithmetic_sequence_a5_value_l200_20050


namespace sum_nat_numbers_l200_20000

/-- 
If S is the set of all natural numbers n such that 0 ≤ n ≤ 200, n ≡ 7 [MOD 11], 
and n ≡ 5 [MOD 7], then the sum of elements in S is 351.
-/
theorem sum_nat_numbers (S : Finset ℕ) 
  (hs : ∀ n, n ∈ S ↔ n ≤ 200 ∧ n % 11 = 7 ∧ n % 7 = 5) 
  : S.sum id = 351 := 
sorry 

end sum_nat_numbers_l200_20000


namespace angle_B_equiv_60_l200_20024

noncomputable def triangle_condition (a b c : ℝ) (A B C : ℝ) : Prop :=
  2 * b * Real.cos B = a * Real.cos C + c * Real.cos A

theorem angle_B_equiv_60 
  (a b c A B C : ℝ)
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : 0 < A) (h5 : A < π)
  (h6 : 0 < B) (h7 : B < π)
  (h8 : 0 < C) (h9 : C < π)
  (h_triangle : A + B + C = π)
  (h_arith : triangle_condition a b c A B C) : 
  B = π / 3 :=
by
  sorry

end angle_B_equiv_60_l200_20024


namespace algebraic_expression_value_l200_20048

theorem algebraic_expression_value (m : ℝ) (h : (2018 + m) * (2020 + m) = 2) : (2018 + m)^2 + (2020 + m)^2 = 8 :=
by
  sorry

end algebraic_expression_value_l200_20048


namespace probability_of_yellow_ball_l200_20051

theorem probability_of_yellow_ball 
  (red_balls : ℕ) 
  (yellow_balls : ℕ) 
  (blue_balls : ℕ) 
  (total_balls : ℕ)
  (h1 : red_balls = 2)
  (h2 : yellow_balls = 5)
  (h3 : blue_balls = 4)
  (h4 : total_balls = red_balls + yellow_balls + blue_balls) :
  (yellow_balls / total_balls : ℚ) = 5 / 11 :=
by 
  rw [h1, h2, h3] at h4  -- Substitute the ball counts into the total_balls definition.
  norm_num at h4  -- Simplify to verify the total is indeed 11.
  rw [h2, h4] -- Use the number of yellow balls and total number of balls to state the ratio.
  norm_num -- Normalize the fraction to show it equals 5/11.

#check probability_of_yellow_ball

end probability_of_yellow_ball_l200_20051


namespace solve_inequality_system_l200_20082

theorem solve_inequality_system (x : ℝ) :
  (x + 2 < 3 * x) ∧ ((5 - x) / 2 + 1 < 0) → (x > 7) :=
by
  sorry

end solve_inequality_system_l200_20082


namespace rationalize_denominator_correct_l200_20049

noncomputable def rationalize_denominator : Prop :=
  1 / (Real.sqrt 3 - 1) = (Real.sqrt 3 + 1) / 2

theorem rationalize_denominator_correct : rationalize_denominator :=
by
  sorry

end rationalize_denominator_correct_l200_20049


namespace solve_trig_equation_l200_20034

theorem solve_trig_equation (x : ℝ) : 
  (∃ (k : ℤ), x = (Real.pi / 16) * (4 * k + 1)) ↔ 2 * (Real.cos (4 * x) - Real.sin x * Real.cos (3 * x)) = Real.sin (4 * x) + Real.sin (2 * x) :=
by
  -- The full proof detail goes here.
  sorry

end solve_trig_equation_l200_20034


namespace ratio_of_ages_l200_20061

theorem ratio_of_ages (a b : ℕ) (h1 : a - 10 = (b - 10) / 2) (h2 : a + b = 35) : a / gcd a b = 3 ∧ b / gcd a b = 4 :=
by
  sorry

end ratio_of_ages_l200_20061


namespace mittens_per_box_l200_20058

theorem mittens_per_box (boxes : ℕ) (scarves_per_box : ℕ) (total_clothing : ℕ) (h_boxes : boxes = 7) (h_scarves : scarves_per_box = 3) (h_total : total_clothing = 49) : 
  let total_scarves := boxes * scarves_per_box
  let total_mittens := total_clothing - total_scarves
  total_mittens / boxes = 4 :=
by
  sorry

end mittens_per_box_l200_20058


namespace ladder_base_distance_l200_20066

theorem ladder_base_distance (h l : ℝ) (h_eq : h = 13) (l_eq : l = 12) :
  ∃ b : ℝ, (h^2 = l^2 + b^2) ∧ b = 5 :=
by
  sorry

end ladder_base_distance_l200_20066


namespace largest_integer_less_than_100_with_remainder_4_l200_20072

theorem largest_integer_less_than_100_with_remainder_4 (k n : ℤ) (h1 : k = 7 * n + 4) (h2 : k < 100) : k ≤ 95 :=
sorry

end largest_integer_less_than_100_with_remainder_4_l200_20072


namespace speed_of_water_l200_20088

theorem speed_of_water (v : ℝ) (swim_speed_still_water : ℝ)
  (distance : ℝ) (time : ℝ)
  (h1 : swim_speed_still_water = 4) 
  (h2 : distance = 14) 
  (h3 : time = 7) 
  (h4 : 4 - v = distance / time) : 
  v = 2 := 
sorry

end speed_of_water_l200_20088


namespace first_term_of_geometric_series_l200_20016

theorem first_term_of_geometric_series (a r S : ℝ) (h₁ : r = 1/4) (h₂ : S = 80) (h₃ : S = a / (1 - r)) : a = 60 :=
by
  sorry

end first_term_of_geometric_series_l200_20016


namespace simplify_expression_at_zero_l200_20028

-- Define the expression f(x)
def f (x : ℚ) : ℚ := (2 * x + 4) / (x^2 - 6 * x + 9) / ((2 * x - 1) / (x - 3) - 1)

-- State that for the given value x = 0, the simplified expression equals -2/3
theorem simplify_expression_at_zero :
  f 0 = -2 / 3 :=
by
  sorry

end simplify_expression_at_zero_l200_20028


namespace simple_interest_amount_is_58_l200_20067

noncomputable def principal (CI : ℝ) (r : ℝ) (t : ℝ) : ℝ :=
  CI / ((1 + r / 100)^t - 1)

noncomputable def simple_interest (P : ℝ) (r : ℝ) (t : ℝ) : ℝ :=
  P * r * t / 100

theorem simple_interest_amount_is_58 (CI : ℝ) (r : ℝ) (t : ℝ) (P : ℝ) :
  CI = 59.45 -> r = 5 -> t = 2 -> P = principal CI r t ->
  simple_interest P r t = 58 :=
by
  sorry

end simple_interest_amount_is_58_l200_20067


namespace num_comics_bought_l200_20091

def initial_comic_books : ℕ := 14
def current_comic_books : ℕ := 13
def comic_books_sold (initial : ℕ) : ℕ := initial / 2
def comics_bought (initial current : ℕ) : ℕ :=
  current - (initial - comic_books_sold initial)

theorem num_comics_bought :
  comics_bought initial_comic_books current_comic_books = 6 :=
by
  sorry

end num_comics_bought_l200_20091


namespace frustum_volume_l200_20045

noncomputable def volume_of_frustum (V₁ V₂ : ℝ) : ℝ :=
  V₁ - V₂

theorem frustum_volume : 
  let base_edge_original := 15
  let height_original := 10
  let base_edge_smaller := 9
  let height_smaller := 6
  let base_area_original := base_edge_original ^ 2
  let base_area_smaller := base_edge_smaller ^ 2
  let V_original := (1 / 3 : ℝ) * base_area_original * height_original
  let V_smaller := (1 / 3 : ℝ) * base_area_smaller * height_smaller
  volume_of_frustum V_original V_smaller = 588 := 
by
  sorry

end frustum_volume_l200_20045


namespace problem_solution_l200_20011

noncomputable def proof_problem (x1 x2 x3 x4 x5 : ℝ) (h1 : 0 < x1) (h2 : 0 < x2) (h3 : 0 < x3) (h4 : 0 < x4) (h5 : 0 < x5) : Prop :=
  ((x1^2 - x3 * x5) * (x2^2 - x3 * x5) ≤ 0) ∧
  ((x2^2 - x4 * x1) * (x3^2 - x4 * x1) ≤ 0) ∧
  ((x3^2 - x5 * x2) * (x4^2 - x5 * x2) ≤ 0) ∧
  ((x4^2 - x1 * x3) * (x5^2 - x1 * x3) ≤ 0) ∧
  ((x5^2 - x2 * x4) * (x1^2 - x2 * x4) ≤ 0) → 
  x1 = x2 ∧ x2 = x3 ∧ x3 = x4 ∧ x4 = x5

theorem problem_solution (x1 x2 x3 x4 x5 : ℝ) (h1 : 0 < x1) (h2 : 0 < x2) (h3 : 0 < x3) (h4 : 0 < x4) (h5 : 0 < x5) :
  proof_problem x1 x2 x3 x4 x5 h1 h2 h3 h4 h5 :=
  by
    sorry

end problem_solution_l200_20011


namespace min_value_expression_l200_20064

theorem min_value_expression (a b: ℝ) (h : 2 * a + b = 1) : (a - 1) ^ 2 + (b - 1) ^ 2 = 4 / 5 :=
sorry

end min_value_expression_l200_20064


namespace first_term_geometric_sequence_l200_20089

theorem first_term_geometric_sequence (a r : ℚ) 
    (h1 : a * r^2 = 8) 
    (h2 : a * r^4 = 27 / 4) : 
    a = 256 / 27 :=
by sorry

end first_term_geometric_sequence_l200_20089


namespace g_at_12_l200_20006

def g (n : ℤ) : ℤ := n^2 + 2*n + 23

theorem g_at_12 : g 12 = 191 := by
  -- proof skipped
  sorry

end g_at_12_l200_20006


namespace find_standard_equation_of_ellipse_l200_20055

noncomputable def ellipse_equation (a c b : ℝ) : Prop :=
  ∃ x y : ℝ, (x^2 / a^2 + y^2 / b^2 = 1) ∨ (y^2 / a^2 + x^2 / b^2 = 1)

theorem find_standard_equation_of_ellipse (h1 : 2 * a = 12) (h2 : c / a = 1 / 3) :
  ellipse_equation 6 2 4 :=
by
  -- We are proving that given the conditions, the standard equation of the ellipse is as stated
  sorry

end find_standard_equation_of_ellipse_l200_20055


namespace total_weight_of_snacks_l200_20021

-- Definitions for conditions
def weight_peanuts := 0.1
def weight_raisins := 0.4
def weight_almonds := 0.3

-- Theorem statement
theorem total_weight_of_snacks : weight_peanuts + weight_raisins + weight_almonds = 0.8 := by
  sorry

end total_weight_of_snacks_l200_20021


namespace max_sum_pyramid_on_hexagonal_face_l200_20076

structure hexagonal_prism :=
(faces_initial : ℕ)
(vertices_initial : ℕ)
(edges_initial : ℕ)

structure pyramid_added :=
(faces_total : ℕ)
(vertices_total : ℕ)
(edges_total : ℕ)
(total_sum : ℕ)

theorem max_sum_pyramid_on_hexagonal_face (h : hexagonal_prism) :
  (h = ⟨8, 12, 18⟩) →
  ∃ p : pyramid_added, 
    p = ⟨13, 13, 24, 50⟩ :=
by
  sorry

end max_sum_pyramid_on_hexagonal_face_l200_20076


namespace least_amount_of_money_l200_20087

variable (Money : Type) [LinearOrder Money]
variable (Anne Bo Coe Dan El : Money)

-- Conditions from the problem
axiom anne_less_than_bo : Anne < Bo
axiom dan_less_than_bo : Dan < Bo
axiom coe_less_than_anne : Coe < Anne
axiom coe_less_than_el : Coe < El
axiom coe_less_than_dan : Coe < Dan
axiom dan_less_than_anne : Dan < Anne

theorem least_amount_of_money : (∀ x, x = Anne ∨ x = Bo ∨ x = Coe ∨ x = Dan ∨ x = El → Coe < x) :=
by
  sorry

end least_amount_of_money_l200_20087


namespace BD_is_diameter_of_circle_l200_20030

variables {A B C D X Y : Type} [MetricSpace A] [MetricSpace B] 
  [MetricSpace C] [MetricSpace D] [MetricSpace X] [MetricSpace Y]

-- Assume these four points lie on a circle with certain ordering
variables (circ : Circle A B C D)

-- Given conditions
variables (h1 : circ.AB < circ.AD)
variables (h2 : circ.BC > circ.CD)

-- Points X and Y are where angle bisectors meet the circle again
variables (h3 : circ.bisects_angle_BAD_at X)
variables (h4 : circ.bisects_angle_BCD_at Y)

-- Hexagon sides with four equal lengths
variables (hex_equal : circ.hexagon_sides_equal_length A B X C D Y)

-- Prove that BD is a diameter
theorem BD_is_diameter_of_circle : circ.is_diameter BD := 
by
  sorry

end BD_is_diameter_of_circle_l200_20030


namespace wand_cost_l200_20033

-- Conditions based on the problem
def initialWands := 3
def salePrice (x : ℝ) := x + 5
def totalCollected := 130
def soldWands := 2

-- Proof statement
theorem wand_cost (x : ℝ) : 
  2 * salePrice x = totalCollected → x = 60 := 
by 
  sorry

end wand_cost_l200_20033


namespace abs_lt_one_sufficient_not_necessary_l200_20075

theorem abs_lt_one_sufficient_not_necessary (x : ℝ) : (|x| < 1) -> (x < 1) ∧ ¬(x < 1 -> |x| < 1) :=
by
  sorry

end abs_lt_one_sufficient_not_necessary_l200_20075


namespace proof_M1M2_product_l200_20038

theorem proof_M1M2_product : 
  (∀ x, (45 * x - 34) / (x^2 - 4 * x + 3) = M_1 / (x - 1) + M_2 / (x - 3)) →
  M_1 * M_2 = -1111 / 4 := 
by
  sorry

end proof_M1M2_product_l200_20038


namespace total_pieces_l200_20096

def pieces_from_friend : ℕ := 123
def pieces_from_brother : ℕ := 136
def pieces_needed : ℕ := 117

theorem total_pieces :
  pieces_from_friend + pieces_from_brother + pieces_needed = 376 :=
by
  unfold pieces_from_friend pieces_from_brother pieces_needed
  sorry

end total_pieces_l200_20096


namespace intersection_of_sets_l200_20078

theorem intersection_of_sets :
  let A := {-2, -1, 0, 1, 2}
  let B := {x | -2 < x ∧ x ≤ 2}
  A ∩ B = {-1, 0, 1, 2} :=
by
  sorry

end intersection_of_sets_l200_20078


namespace distance_around_track_l200_20002

-- Define the conditions
def total_mileage : ℝ := 10
def distance_to_high_school : ℝ := 3
def round_trip_distance : ℝ := 2 * distance_to_high_school

-- State the question and the desired proof problem
theorem distance_around_track : 
  total_mileage - round_trip_distance = 4 := 
by
  sorry

end distance_around_track_l200_20002


namespace range_of_k_l200_20095

noncomputable def h (x : ℝ) (k : ℝ) : ℝ := 2 * x - k / x + k / 3

theorem range_of_k (k : ℝ) :
  (∀ x : ℝ, 1 < x → 2 + k / x^2 > 0) ↔ k ≥ -2 :=
by
  sorry

end range_of_k_l200_20095


namespace downstream_speed_l200_20022

theorem downstream_speed 
  (upstream_speed : ℕ) 
  (still_water_speed : ℕ) 
  (hm_upstream : upstream_speed = 27) 
  (hm_still_water : still_water_speed = 31) 
  : (still_water_speed + (still_water_speed - upstream_speed)) = 35 :=
by
  sorry

end downstream_speed_l200_20022


namespace truth_probability_l200_20040

theorem truth_probability (P_A : ℝ) (P_A_and_B : ℝ) (P_B : ℝ) 
  (hA : P_A = 0.70) (hA_and_B : P_A_and_B = 0.42) : 
  P_A * P_B = P_A_and_B → P_B = 0.6 :=
by
  sorry

end truth_probability_l200_20040


namespace whale_plankton_feeding_frenzy_l200_20019

theorem whale_plankton_feeding_frenzy
  (x y : ℕ)
  (h1 : x + 5 * y = 54)
  (h2 : 9 * x + 36 * y = 450) :
  y = 4 :=
sorry

end whale_plankton_feeding_frenzy_l200_20019


namespace no_integer_solutions_l200_20032

theorem no_integer_solutions :
  ¬ ∃ (x y : ℤ), x^3 + 21 * y^2 + 5 = 0 :=
by {
  sorry
}

end no_integer_solutions_l200_20032


namespace parabola1_right_of_parabola2_l200_20068

def parabola1 (x : ℝ) : ℝ := x^2 - x + 3
def parabola2 (x : ℝ) : ℝ := x^2 + x + 5

theorem parabola1_right_of_parabola2 :
  ∃ x1 x2 : ℝ, x1 > x2 ∧ parabola1 x1 < parabola2 x2 :=
by
  sorry

end parabola1_right_of_parabola2_l200_20068


namespace perpendicular_vectors_eq_l200_20069

theorem perpendicular_vectors_eq {x : ℝ} (h : (x - 5) * 2 + 3 * x = 0) : x = 2 :=
sorry

end perpendicular_vectors_eq_l200_20069


namespace micah_water_l200_20035

theorem micah_water (x : ℝ) (h1 : 3 * x + x = 6) : x = 1.5 :=
sorry

end micah_water_l200_20035


namespace original_bananas_total_l200_20060

theorem original_bananas_total (willie_bananas : ℝ) (charles_bananas : ℝ) : willie_bananas = 48.0 → charles_bananas = 35.0 → willie_bananas + charles_bananas = 83.0 :=
by
  intros h1 h2
  rw [h1, h2]
  norm_num

end original_bananas_total_l200_20060


namespace sprint_team_total_miles_l200_20046

theorem sprint_team_total_miles (number_of_people : ℝ) (miles_per_person : ℝ) 
  (h1 : number_of_people = 150.0) (h2 : miles_per_person = 5.0) : 
  number_of_people * miles_per_person = 750.0 :=
by
  rw [h1, h2]
  norm_num

end sprint_team_total_miles_l200_20046


namespace range_of_a_iff_l200_20037

noncomputable def range_of_a (a : ℝ) : Prop :=
  ∀ x : ℝ, |x| + |x - 1| ≤ a → a ≥ 1

theorem range_of_a_iff (a : ℝ) :
  (∃ x : ℝ, |x| + |x - 1| ≤ a) ↔ (a ≥ 1) :=
by sorry

end range_of_a_iff_l200_20037


namespace compute_value_l200_20063

def Δ (p q : ℕ) : ℕ := p^3 - q

theorem compute_value : Δ (5^Δ 2 7) (4^Δ 4 8) = 125 - 4^56 := by
  sorry

end compute_value_l200_20063


namespace flour_needed_l200_20013

-- Define the given conditions
def F_total : ℕ := 9
def F_added : ℕ := 3

-- State the main theorem to be proven
theorem flour_needed : (F_total - F_added) = 6 := by
  sorry -- Placeholder for the proof

end flour_needed_l200_20013


namespace smallest_integer_in_set_l200_20053

theorem smallest_integer_in_set (n : ℤ) (h : n+4 < 3 * ((n + (n + 1) + (n + 2) + (n + 3) + (n + 4)) / 5)) : n ≥ 0 :=
by sorry

end smallest_integer_in_set_l200_20053


namespace show_length_50_l200_20003

def Gina_sSis_three_as_often (G S : ℕ) : Prop := G = 3 * S
def sister_total_shows (G S : ℕ) : Prop := G + S = 24
def Gina_total_minutes (G : ℕ) (minutes : ℕ) : Prop := minutes = 900
def length_of_each_show (minutes shows length : ℕ) : Prop := length = minutes / shows

theorem show_length_50 (G S : ℕ) (length : ℕ) :
  Gina_sSis_three_as_often G S →
  sister_total_shows G S →
  Gina_total_minutes G 900 →
  length_of_each_show 900 G length →
  length = 50 :=
by
  intros h1 h2 h3 h4
  sorry

end show_length_50_l200_20003


namespace avg_age_adults_l200_20004

-- Given conditions
def num_members : ℕ := 50
def avg_age_members : ℕ := 20
def num_girls : ℕ := 25
def num_boys : ℕ := 20
def num_adults : ℕ := 5
def avg_age_girls : ℕ := 18
def avg_age_boys : ℕ := 22

-- Prove that the average age of the adults is 22 years
theorem avg_age_adults :
  (num_members * avg_age_members - num_girls * avg_age_girls - num_boys * avg_age_boys) / num_adults = 22 :=
by 
  sorry

end avg_age_adults_l200_20004


namespace trapezoid_problem_case1_trapezoid_problem_case2_trapezoid_problem_case3_l200_20012

variable {a b : ℝ}
variable {M N : ℝ}

/-- Trapezoid problem statements -/
theorem trapezoid_problem_case1 (h : a < 2 * b) : M - N = a - 2 * b := 
sorry

theorem trapezoid_problem_case2 (h : a = 2 * b) : M - N = 0 := 
sorry

theorem trapezoid_problem_case3 (h : a > 2 * b) : M - N = 2 * b - a := 
sorry

end trapezoid_problem_case1_trapezoid_problem_case2_trapezoid_problem_case3_l200_20012


namespace total_packs_l200_20090

theorem total_packs (cards_per_person : ℕ) (cards_per_pack : ℕ) (people_count : ℕ) (cards_per_person_eq : cards_per_person = 540) (cards_per_pack_eq : cards_per_pack = 20) (people_count_eq : people_count = 4) :
  (cards_per_person / cards_per_pack) * people_count = 108 :=
by
  sorry

end total_packs_l200_20090


namespace find_n_tan_eq_l200_20018

theorem find_n_tan_eq (n : ℝ) (h1 : -180 < n) (h2 : n < 180) (h3 : Real.tan (n * Real.pi / 180) = Real.tan (678 * Real.pi / 180)) : 
  n = 138 := 
sorry

end find_n_tan_eq_l200_20018


namespace sum_xyz_le_two_l200_20065

theorem sum_xyz_le_two (x y z : ℝ) (h : 2 * x + y^2 + z^2 ≤ 2) : x + y + z ≤ 2 :=
sorry

end sum_xyz_le_two_l200_20065


namespace vertex_of_quadratic_l200_20014

-- Define the quadratic function
def quadratic_function (x : ℝ) : ℝ := -3 * x^2 - 6 * x + 5

-- State the theorem for vertex coordinates
theorem vertex_of_quadratic :
  (∀ x : ℝ, quadratic_function (- (-6) / (2 * -3)) = quadratic_function 1)
  → (1, quadratic_function 1) = (1, 8) :=
by
  intros h
  sorry

end vertex_of_quadratic_l200_20014


namespace gcd_2048_2101_eq_1_l200_20062

theorem gcd_2048_2101_eq_1 : Int.gcd 2048 2101 = 1 := sorry

end gcd_2048_2101_eq_1_l200_20062


namespace trigonometric_identity_l200_20093

theorem trigonometric_identity (α : ℝ) :
    1 - 1/4 * (Real.sin (2 * α)) ^ 2 + Real.cos (2 * α) = (Real.cos α) ^ 2 + (Real.cos α) ^ 4 :=
by
  sorry

end trigonometric_identity_l200_20093


namespace highest_place_value_734_48_l200_20056

theorem highest_place_value_734_48 : 
  (∃ k, 10^4 = k ∧ k * 10^4 ≤ 734 * 48 ∧ 734 * 48 < (k + 1) * 10^4) := 
sorry

end highest_place_value_734_48_l200_20056


namespace inequality_holds_l200_20001

theorem inequality_holds : ∀ (n : ℕ), (n - 1)^(n + 1) * (n + 1)^(n - 1) < n^(2 * n) :=
by sorry

end inequality_holds_l200_20001


namespace find_a_l200_20080

variables {a b c : ℂ}

-- Given conditions
variables (h1 : a + b + c = 5) 
variables (h2 : a * b + b * c + c * a = 5) 
variables (h3 : a * b * c = 5)
variables (h4 : a.im = 0) -- a is real

theorem find_a : a = 4 :=
by
  sorry

end find_a_l200_20080


namespace total_chocolate_bars_l200_20020

theorem total_chocolate_bars (small_boxes : ℕ) (bars_per_box : ℕ) 
  (h1 : small_boxes = 17) (h2 : bars_per_box = 26) 
  : small_boxes * bars_per_box = 442 :=
by sorry

end total_chocolate_bars_l200_20020


namespace sin_240_eq_neg_sqrt3_over_2_l200_20017

theorem sin_240_eq_neg_sqrt3_over_2 : Real.sin (240 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  sorry

end sin_240_eq_neg_sqrt3_over_2_l200_20017


namespace odd_integer_95th_l200_20036

theorem odd_integer_95th : (2 * 95 - 1) = 189 := 
by
  -- The proof would go here
  sorry

end odd_integer_95th_l200_20036


namespace zach_cookies_left_l200_20029

/- Defining the initial conditions on cookies baked each day -/
def cookies_monday : ℕ := 32
def cookies_tuesday : ℕ := cookies_monday / 2
def cookies_wednesday : ℕ := 3 * cookies_tuesday - 4 - 3
def cookies_thursday : ℕ := 2 * cookies_monday - 10 + 5
def cookies_friday : ℕ := cookies_wednesday - 6 - 4
def cookies_saturday : ℕ := cookies_monday + cookies_friday - 10

/- Aggregating total cookies baked throughout the week -/
def total_baked : ℕ := cookies_monday + cookies_tuesday + cookies_wednesday +
                      cookies_thursday + cookies_friday + cookies_saturday

/- Defining cookies lost each day -/
def daily_parents_eat : ℕ := 2 * 6
def neighbor_friday_eat : ℕ := 8
def friends_thursday_eat : ℕ := 3 * 2

def total_lost : ℕ := 4 + 3 + 10 + 6 + 4 + 10 + daily_parents_eat + neighbor_friday_eat + friends_thursday_eat

/- Calculating cookies left at end of six days -/
def cookies_left : ℕ := total_baked - total_lost

/- Proof objective -/
theorem zach_cookies_left : cookies_left = 200 := by
  sorry

end zach_cookies_left_l200_20029


namespace probability_of_pink_l200_20052

variable (B P : ℕ) -- number of blue and pink gumballs
variable (h_total : B + P > 0) -- there is at least one gumball in the jar
variable (h_prob_two_blue : (B / (B + P)) * (B / (B + P)) = 16 / 49) -- the probability of drawing two blue gumballs in a row

theorem probability_of_pink : (P / (B + P)) = 3 / 7 :=
sorry

end probability_of_pink_l200_20052


namespace valid_p_values_l200_20023

theorem valid_p_values (p : ℕ) (h : p = 3 ∨ p = 4 ∨ p = 5 ∨ p = 12) :
  0 < (4 * p + 34) / (3 * p - 8) ∧ (4 * p + 34) % (3 * p - 8) = 0 :=
by
  sorry

end valid_p_values_l200_20023


namespace point_in_second_quadrant_l200_20042

theorem point_in_second_quadrant (a : ℝ) : 
  ∃ q : ℕ, q = 2 ∧ (-1, a^2 + 1).1 < 0 ∧ 0 < (-1, a^2 + 1).2 :=
by
  sorry

end point_in_second_quadrant_l200_20042


namespace Isabella_exchange_l200_20099

/-
Conditions:
1. Isabella exchanged d U.S. dollars to receive (8/5)d Canadian dollars.
2. After spending 80 Canadian dollars, she had d + 20 Canadian dollars left.
3. Sum of the digits of d is 14.
-/

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.foldr (.+.) 0

theorem Isabella_exchange (d : ℕ) (h : (8 * d / 5) - 80 = d + 20) : sum_of_digits d = 14 :=
by sorry

end Isabella_exchange_l200_20099


namespace calc_mod_residue_l200_20085

theorem calc_mod_residue :
  (245 * 15 - 18 * 8 + 5) % 17 = 0 := by
  sorry

end calc_mod_residue_l200_20085


namespace highest_power_of_2_dividing_15_pow_4_minus_9_pow_4_l200_20031

theorem highest_power_of_2_dividing_15_pow_4_minus_9_pow_4 :
  (∃ k, 15^4 - 9^4 = 2^k * m ∧ ¬ ∃ m', m = 2 * m') ∧ (k = 5) :=
by
  sorry

end highest_power_of_2_dividing_15_pow_4_minus_9_pow_4_l200_20031


namespace age_problem_l200_20097

theorem age_problem (F : ℝ) (M : ℝ) (Y : ℝ)
  (hF : F = 40.00000000000001)
  (hM : M = (2/5) * F)
  (hY : M + Y = (1/2) * (F + Y)) :
  Y = 8.000000000000002 :=
sorry

end age_problem_l200_20097


namespace shaded_area_l200_20025

open Real

theorem shaded_area (AH HF GF : ℝ) (AH_eq : AH = 12) (HF_eq : HF = 16) (GF_eq : GF = 4) 
  (DG : ℝ) (DG_eq : DG = 3) (area_triangle_DGF : ℝ) (area_triangle_DGF_eq : area_triangle_DGF = 6) :
  let area_square : ℝ := 4 * 4
  let shaded_area : ℝ := area_square - area_triangle_DGF
  shaded_area = 10 := by
    sorry

end shaded_area_l200_20025


namespace equation_of_lamps_l200_20073

theorem equation_of_lamps (n k : ℕ) (N M : ℕ) (h1 : 0 < n) (h2 : 0 < k) (h3 : k ≥ n) (h4 : (k - n) % 2 = 0) : 
  N = 2^(k - n) * M := 
sorry

end equation_of_lamps_l200_20073


namespace solve_double_inequality_l200_20047

theorem solve_double_inequality (x : ℝ) :
  (-1 < (x^2 - 20 * x + 21) / (x^2 - 4 * x + 5) ∧
   (x^2 - 20 * x + 21) / (x^2 - 4 * x + 5) < 1) ↔ (2 < x ∨ 26 < x) := 
sorry

end solve_double_inequality_l200_20047


namespace total_money_divided_l200_20010

theorem total_money_divided (A B C T : ℝ) 
    (h1 : A = (2/5) * (B + C)) 
    (h2 : B = (1/5) * (A + C)) 
    (h3 : A = 600) :
    T = A + B + C →
    T = 2100 :=
by 
  sorry

end total_money_divided_l200_20010


namespace num_ways_distribute_plants_correct_l200_20027

def num_ways_to_distribute_plants : Nat :=
  let basil := 2
  let aloe := 1
  let cactus := 1
  let white_lamps := 2
  let red_lamp := 1
  let blue_lamp := 1
  let plants := basil + aloe + cactus
  let lamps := white_lamps + red_lamp + blue_lamp
  4
  
theorem num_ways_distribute_plants_correct :
  num_ways_to_distribute_plants = 4 :=
by
  sorry -- Proof of the correctness of the distribution

end num_ways_distribute_plants_correct_l200_20027


namespace total_rope_in_inches_l200_20086

-- Definitions for conditions
def feet_last_week : ℕ := 6
def feet_less : ℕ := 4
def inches_per_foot : ℕ := 12

-- Condition: rope bought this week
def feet_this_week := feet_last_week - feet_less

-- Condition: total rope bought in feet
def total_feet := feet_last_week + feet_this_week

-- Condition: total rope bought in inches
def total_inches := total_feet * inches_per_foot

-- Theorem statement
theorem total_rope_in_inches : total_inches = 96 := by
  sorry

end total_rope_in_inches_l200_20086


namespace find_a1000_l200_20054

noncomputable def seq (a : ℕ → ℤ) : Prop :=
a 1 = 1009 ∧
a 2 = 1010 ∧
(∀ n ≥ 1, a n + a (n + 1) + a (n + 2) = 2 * n)

theorem find_a1000 (a : ℕ → ℤ) (h : seq a) : a 1000 = 1675 :=
sorry

end find_a1000_l200_20054


namespace range_of_a_l200_20081

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if h : x ≤ 1 then (a - 2) * x - 1 else Real.log x / Real.log a

theorem range_of_a (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) ↔ 2 < a ∧ a ≤ 3 :=
by {
  sorry
}

end range_of_a_l200_20081


namespace total_bill_l200_20009

-- Definitions from conditions
def num_people : ℕ := 3
def amount_per_person : ℕ := 45

-- Mathematical proof problem statement
theorem total_bill : num_people * amount_per_person = 135 := by
  sorry

end total_bill_l200_20009


namespace simplify_expression_l200_20074

theorem simplify_expression (p : ℝ) : ((7 * p - 3) - 3 * p * 2) * 2 + (5 - 2 / 2) * (8 * p - 12) = 34 * p - 54 := by
  sorry

end simplify_expression_l200_20074


namespace problem1_inner_problem2_inner_l200_20079

-- Problem 1
theorem problem1_inner {m n : ℤ} (hm : |m| = 5) (hn : |n| = 4) (opposite_signs : m * n < 0) :
  m^2 - m * n + n = 41 ∨ m^2 - m * n + n = 49 :=
sorry

-- Problem 2
theorem problem2_inner {a b c d x : ℝ} (opposite_ab : a + b = 0) (reciprocals_cd : c * d = 1) (hx : |x| = 5) (hx_pos : x > 0) :
  3 * (a + b) - 2 * (c * d) + x = 3 :=
sorry

end problem1_inner_problem2_inner_l200_20079


namespace solve_system_l200_20070

theorem solve_system :
  ∃ x y z : ℝ, (8 * (x^3 + y^3 + z^3) = 73) ∧
              (2 * (x^2 + y^2 + z^2) = 3 * (x * y + y * z + z * x)) ∧
              (x * y * z = 1) ∧
              (x, y, z) = (1, 2, 0.5) ∨ (x, y, z) = (1, 0.5, 2) ∨
              (x, y, z) = (2, 1, 0.5) ∨ (x, y, z) = (2, 0.5, 1) ∨
              (x, y, z) = (0.5, 1, 2) ∨ (x, y, z) = (0.5, 2, 1) :=
by
  sorry

end solve_system_l200_20070


namespace geometric_sum_a4_a6_l200_20008

-- Definitions based on the conditions
def geometric_sequence (a : ℕ → ℝ) := ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sum_a4_a6 (a : ℕ → ℝ) (h_geom : geometric_sequence a) (h_pos : ∀ n, a n > 0) 
(h_cond : a 3 * a 5 + a 2 * a 10 + 2 * a 4 * a 6 = 100) : a 4 + a 6 = 10 :=
by
  sorry

end geometric_sum_a4_a6_l200_20008


namespace smallest_number_of_coins_l200_20044

theorem smallest_number_of_coins :
  ∃ (n : ℕ), (∀ (a : ℕ), 5 ≤ a ∧ a < 100 → 
    ∃ (c : ℕ → ℕ), (a = 5 * c 0 + 10 * c 1 + 25 * c 2) ∧ 
    (c 0 + c 1 + c 2 = n)) ∧ n = 9 :=
by
  sorry

end smallest_number_of_coins_l200_20044


namespace g_neg_two_is_zero_l200_20007

theorem g_neg_two_is_zero {f g : ℤ → ℤ} 
  (h_odd: ∀ x: ℤ, f (-x) + (-x) = -(f x + x)) 
  (hf_two: f 2 = 1) 
  (hg_def: ∀ x: ℤ, g x = f x + 1):
  g (-2) = 0 := 
sorry

end g_neg_two_is_zero_l200_20007


namespace translation_coordinates_l200_20094

theorem translation_coordinates :
  ∀ (x y : ℤ) (a : ℤ), 
  (x, y) = (3, -4) → a = 5 → (x - a, y) = (-2, -4) :=
by
  sorry

end translation_coordinates_l200_20094


namespace intersection_PQ_l200_20039

def P := {x : ℝ | x < 1}
def Q := {x : ℝ | x^2 < 4}
def PQ_intersection := {x : ℝ | -2 < x ∧ x < 1}

theorem intersection_PQ : P ∩ Q = PQ_intersection := by
  sorry

end intersection_PQ_l200_20039


namespace sequence_of_arrows_from_425_to_427_l200_20026

theorem sequence_of_arrows_from_425_to_427 :
  ∀ (arrows : ℕ → ℕ), (∀ n, arrows (n + 4) = arrows n) →
  (arrows 425, arrows 426, arrows 427) = (arrows 1, arrows 2, arrows 3) :=
by
  intros arrows h_period
  have h1 : arrows 425 = arrows 1 := by 
    sorry
  have h2 : arrows 426 = arrows 2 := by 
    sorry
  have h3 : arrows 427 = arrows 3 := by 
    sorry
  sorry

end sequence_of_arrows_from_425_to_427_l200_20026


namespace chocolate_cost_is_3_l200_20041

-- Definitions based on the conditions
def dan_has_5_dollars : Prop := true
def cost_candy_bar : ℕ := 2
def cost_chocolate : ℕ := cost_candy_bar + 1

-- Theorem to prove
theorem chocolate_cost_is_3 : cost_chocolate = 3 :=
by {
  -- This is where the proof steps would go
  sorry
}

end chocolate_cost_is_3_l200_20041


namespace range_of_m_l200_20077

open Set Real

-- Define over the real numbers ℝ
noncomputable def A : Set ℝ := { x : ℝ | x^2 - 2*x - 3 ≤ 0 }
noncomputable def B (m : ℝ) : Set ℝ := { x : ℝ | x^2 - 2*m*x + m^2 - 4 ≤ 0 }
noncomputable def CRB (m : ℝ) : Set ℝ := { x : ℝ | x < m - 2 ∨ x > m + 2 }

-- Main theorem statement
theorem range_of_m (m : ℝ) (h : A ⊆ CRB m) : m < -3 ∨ m > 5 :=
sorry

end range_of_m_l200_20077


namespace sum_reciprocal_geo_seq_l200_20071

theorem sum_reciprocal_geo_seq {a_5 a_6 a_7 a_8 : ℝ}
  (h_sum : a_5 + a_6 + a_7 + a_8 = 15 / 8)
  (h_prod : a_6 * a_7 = -9 / 8) :
  (1 / a_5) + (1 / a_6) + (1 / a_7) + (1 / a_8) = -5 / 3 := by
  sorry

end sum_reciprocal_geo_seq_l200_20071
