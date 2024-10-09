import Mathlib

namespace value_of_x_squared_add_reciprocal_squared_l1000_100020

theorem value_of_x_squared_add_reciprocal_squared (x : ℝ) (h : 47 = x^4 + 1 / x^4) : x^2 + 1 / x^2 = 7 :=
by
  sorry

end value_of_x_squared_add_reciprocal_squared_l1000_100020


namespace a_n_general_term_b_n_general_term_l1000_100044

noncomputable def seq_a (n : ℕ) : ℕ :=
  2 * n - 1

theorem a_n_general_term (n : ℕ) (Sn : ℕ → ℕ) (S_property : ∀ n : ℕ, 4 * Sn n = (seq_a n) ^ 2 + 2 * seq_a n + 1) :
  seq_a n = 2 * n - 1 :=
sorry

noncomputable def geom_seq (q : ℕ) (n : ℕ) : ℕ :=
  q ^ (n - 1)

theorem b_n_general_term (n m q : ℕ) (a1 am am3 : ℕ) (b_property : ∀ n : ℕ, geom_seq q n = q ^ (n - 1))
  (a_property : ∀ n : ℕ, seq_a n = 2 * n - 1)
  (b1_condition : geom_seq q 1 = seq_a 1) (bm_condition : geom_seq q m = seq_a m)
  (bm1_condition : geom_seq q (m + 1) = seq_a (m + 3)) :
  q = 3 ∨ q = 7 ∧ (∀ n : ℕ, geom_seq q n = 3 ^ (n - 1) ∨ geom_seq q n = 7 ^ (n - 1)) :=
sorry

end a_n_general_term_b_n_general_term_l1000_100044


namespace prob_equals_two_yellow_marbles_l1000_100011

noncomputable def probability_two_yellow_marbles : ℚ :=
  let total_marbles : ℕ := 3 + 4 + 8
  let yellow_marbles : ℕ := 4
  let first_draw_prob : ℚ := yellow_marbles / total_marbles
  let second_total_marbles : ℕ := total_marbles - 1
  let second_yellow_marbles : ℕ := yellow_marbles - 1
  let second_draw_prob : ℚ := second_yellow_marbles / second_total_marbles
  first_draw_prob * second_draw_prob

theorem prob_equals_two_yellow_marbles :
  probability_two_yellow_marbles = 2 / 35 :=
by
  sorry

end prob_equals_two_yellow_marbles_l1000_100011


namespace purely_imaginary_complex_number_l1000_100027

theorem purely_imaginary_complex_number (a : ℝ) (i : ℂ)
  (h₁ : i * i = -1)
  (h₂ : ∃ z : ℂ, z = (a + i) / (1 - i) ∧ z.im ≠ 0 ∧ z.re = 0) :
  a = 1 :=
sorry

end purely_imaginary_complex_number_l1000_100027


namespace train_speed_l1000_100097

theorem train_speed (distance time : ℕ) (h1 : distance = 180) (h2 : time = 9) : distance / time = 20 := by
  sorry

end train_speed_l1000_100097


namespace complement_of_M_in_U_l1000_100048

open Set

noncomputable def U : Set ℕ := {0, 1, 2, 3, 4, 5}
noncomputable def M : Set ℕ := {0, 1}

theorem complement_of_M_in_U : (U \ M) = {2, 3, 4, 5} :=
by
  -- The proof is omitted here.
  sorry

end complement_of_M_in_U_l1000_100048


namespace geometric_sequence_properties_l1000_100099

noncomputable def geometric_sequence (a2 a5 : ℕ) (n : ℕ) : ℕ :=
  3 ^ (n - 1)

noncomputable def sum_first_n_terms (n : ℕ) : ℕ :=
  (3^n - 1) / 2

def T10_sum_of_sequence : ℚ := 10/11

theorem geometric_sequence_properties :
  (geometric_sequence 3 81 2 = 3) ∧
  (geometric_sequence 3 81 5 = 81) ∧
  (sum_first_n_terms 2 = (geometric_sequence 3 81 1 + geometric_sequence 3 81 2)) ∧
  (sum_first_n_terms 5 = (geometric_sequence 3 81 1 + geometric_sequence 3 81 2 + geometric_sequence 3 81 3 + geometric_sequence 3 81 4 + geometric_sequence 3 81 5)) ∧
  T10_sum_of_sequence = 10/11 :=
by
  sorry

end geometric_sequence_properties_l1000_100099


namespace min_cost_and_ways_l1000_100073

-- Define the cost of each package
def cost_A : ℕ := 10
def cost_B : ℕ := 5

-- Define a function to calculate the total cost given the number of each package
def total_cost (nA nB : ℕ) : ℕ := nA * cost_A + nB * cost_B

-- Define the number of friends
def num_friends : ℕ := 4

-- Prove the minimum cost is 15 yuan and there are 28 ways
theorem min_cost_and_ways :
  (∃ nA nB : ℕ, total_cost nA nB = 15 ∧ (
    (nA = 1 ∧ nB = 1 ∧ 12 = 12) ∨ 
    (nA = 0 ∧ nB = 3 ∧ 12 = 12) ∨
    (nA = 0 ∧ nB = 3 ∧ 4 = 4) → 28 = 28)) :=
sorry

end min_cost_and_ways_l1000_100073


namespace pizza_slices_left_l1000_100058

-- Lean definitions for conditions
def total_slices : ℕ := 24
def slices_eaten_dinner : ℕ := total_slices / 3
def slices_after_dinner : ℕ := total_slices - slices_eaten_dinner

def slices_eaten_yves : ℕ := slices_after_dinner / 5
def slices_after_yves : ℕ := slices_after_dinner - slices_eaten_yves

def slices_eaten_oldest_siblings : ℕ := 3 * 3
def slices_after_oldest_siblings : ℕ := slices_after_yves - slices_eaten_oldest_siblings

def num_remaining_siblings : ℕ := 7 - 3
def slices_eaten_remaining_siblings : ℕ := num_remaining_siblings * 2
def slices_final : ℕ := if slices_after_oldest_siblings < slices_eaten_remaining_siblings then 0 else slices_after_oldest_siblings - slices_eaten_remaining_siblings

-- Proposition to prove
theorem pizza_slices_left : slices_final = 0 := by sorry

end pizza_slices_left_l1000_100058


namespace chi_square_confidence_l1000_100036

theorem chi_square_confidence (chi_square : ℝ) (df : ℕ) (critical_value : ℝ) :
  chi_square = 6.825 ∧ df = 1 ∧ critical_value = 6.635 → confidence_level = 0.99 := 
by
  sorry

end chi_square_confidence_l1000_100036


namespace students_spend_185_minutes_in_timeout_l1000_100042

variable (tR tF tS t_total : ℕ)

-- Conditions
def running_timeouts : ℕ := 5
def food_timeouts : ℕ := 5 * running_timeouts - 1
def swearing_timeouts : ℕ := food_timeouts / 3
def total_timeouts : ℕ := running_timeouts + food_timeouts + swearing_timeouts
def timeout_duration : ℕ := 5

-- Total time spent in time-out
def total_timeout_minutes : ℕ := total_timeouts * timeout_duration

theorem students_spend_185_minutes_in_timeout :
  total_timeout_minutes = 185 :=
by
  -- The answer is directly given by the conditions and the correct answer identified.
  sorry

end students_spend_185_minutes_in_timeout_l1000_100042


namespace no_solution_for_xx_plus_yy_eq_9z_l1000_100069

theorem no_solution_for_xx_plus_yy_eq_9z (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : 
  ¬ (x^x + y^y = 9^z) :=
sorry

end no_solution_for_xx_plus_yy_eq_9z_l1000_100069


namespace fraction_of_number_is_one_fifth_l1000_100086

theorem fraction_of_number_is_one_fifth (N : ℕ) (f : ℚ) 
    (hN : N = 90) 
    (h : 3 + (1 / 2) * (1 / 3) * f * N = (1 / 15) * N) : 
  f = 1 / 5 := by 
  sorry

end fraction_of_number_is_one_fifth_l1000_100086


namespace required_raise_percentage_l1000_100010

theorem required_raise_percentage (S : ℝ) (hS : S > 0) : 
  ((S - (0.85 * S - 50)) / (0.85 * S - 50) = 0.1875) :=
by
  -- Proof of this theorem can be carried out here
  sorry

end required_raise_percentage_l1000_100010


namespace total_number_of_ways_is_144_l1000_100003

def count_ways_to_place_letters_on_grid : Nat :=
  16 * 9

theorem total_number_of_ways_is_144 :
  count_ways_to_place_letters_on_grid = 144 :=
  by
    sorry

end total_number_of_ways_is_144_l1000_100003


namespace circle_hyperbola_intersection_l1000_100098

def hyperbola_equation (x y a : ℝ) : Prop := x^2 - y^2 = a^2
def circle_equation (x y c d r : ℝ) : Prop := (x - c)^2 + (y - d)^2 = r^2

theorem circle_hyperbola_intersection (a r : ℝ) (P Q R S : ℝ × ℝ):
  (∃ c d: ℝ, 
    circle_equation P.1 P.2 c d r ∧ 
    circle_equation Q.1 Q.2 c d r ∧ 
    circle_equation R.1 R.2 c d r ∧ 
    circle_equation S.1 S.2 c d r ∧ 
    hyperbola_equation P.1 P.2 a ∧ 
    hyperbola_equation Q.1 Q.2 a ∧ 
    hyperbola_equation R.1 R.2 a ∧ 
    hyperbola_equation S.1 S.2 a
  ) →
  (P.1^2 + P.2^2) + (Q.1^2 + Q.2^2) + (R.1^2 + R.2^2) + (S.1^2 + S.2^2) = 4 * r^2 :=
by
  sorry

end circle_hyperbola_intersection_l1000_100098


namespace num_int_solutions_l1000_100056

theorem num_int_solutions (x : ℤ) : 
  (x^4 - 39 * x^2 + 140 < 0) ↔ (x = 3 ∨ x = -3 ∨ x = 4 ∨ x = -4 ∨ x = 5 ∨ x = -5) := 
sorry

end num_int_solutions_l1000_100056


namespace crayons_initially_l1000_100037

theorem crayons_initially (crayons_left crayons_lost : ℕ) (h_left : crayons_left = 134) (h_lost : crayons_lost = 345) :
  crayons_left + crayons_lost = 479 :=
by
  sorry

end crayons_initially_l1000_100037


namespace box_contains_1600_calories_l1000_100029

theorem box_contains_1600_calories :
  let cookies_per_bag := 20
  let bags_per_box := 4
  let calories_per_cookie := 20
  let total_cookies := cookies_per_bag * bags_per_box
  let total_calories := total_cookies * calories_per_cookie
  total_calories = 1600 :=
by
  let cookies_per_bag := 20
  let bags_per_box := 4
  let calories_per_cookie := 20
  let total_cookies := cookies_per_bag * bags_per_box
  let total_calories := total_cookies * calories_per_cookie
  show total_calories = 1600
  sorry

end box_contains_1600_calories_l1000_100029


namespace choose_president_and_secretary_same_gender_l1000_100047

theorem choose_president_and_secretary_same_gender :
  let total_members := 25
  let boys := 15
  let girls := 10
  ∃ (total_ways : ℕ), total_ways = (boys * (boys - 1)) + (girls * (girls - 1)) := sorry

end choose_president_and_secretary_same_gender_l1000_100047


namespace sum_of_first_six_terms_geometric_sequence_l1000_100022

theorem sum_of_first_six_terms_geometric_sequence :
  let a := (1 : ℚ) / 4
  let r := (1 : ℚ) / 4
  let n := 6
  let S_n := a * (1 - r ^ n) / (1 - r)
  S_n = 1365 / 4096 := by
  sorry

end sum_of_first_six_terms_geometric_sequence_l1000_100022


namespace exists_identical_coordinates_l1000_100049

theorem exists_identical_coordinates
  (O O' : ℝ × ℝ)
  (Ox Oy O'x' O'y' : ℝ → ℝ)
  (units_different : ∃ u v : ℝ, u ≠ v)
  (O_ne_O' : O ≠ O')
  (Ox_not_parallel_O'x' : ∃ θ : ℝ, θ ≠ 0 ∧ θ ≠ π) :
  ∃ S : ℝ × ℝ, (S.1 = Ox S.1 ∧ S.2 = Oy S.2) ∧ (S.1 = O'x' S.1 ∧ S.2 = O'y' S.2) :=
sorry

end exists_identical_coordinates_l1000_100049


namespace sym_coords_origin_l1000_100082

theorem sym_coords_origin (a b : ℝ) (h : |a - 3| + (b + 4)^2 = 0) :
  (-a, -b) = (-3, 4) :=
sorry

end sym_coords_origin_l1000_100082


namespace max_k_range_minus_five_l1000_100057

theorem max_k_range_minus_five :
  ∃ k : ℝ, (∀ x : ℝ, x^2 + 5 * x + k = -5) → k = 5 / 4 :=
by
  sorry

end max_k_range_minus_five_l1000_100057


namespace cuts_needed_l1000_100071

-- Define the length of the wood in centimeters
def wood_length_cm : ℕ := 400

-- Define the length of each stake in centimeters
def stake_length_cm : ℕ := 50

-- Define the expected number of cuts needed
def expected_cuts : ℕ := 7

-- The main theorem stating the equivalence
theorem cuts_needed (wood_length stake_length : ℕ) (h1 : wood_length = 400) (h2 : stake_length = 50) :
  (wood_length / stake_length) - 1 = expected_cuts :=
sorry

end cuts_needed_l1000_100071


namespace start_time_is_10_am_l1000_100016

-- Definitions related to the problem statements
def distance_AB : ℝ := 600
def speed_A_to_B : ℝ := 70
def speed_B_to_A : ℝ := 80
def meeting_time : ℝ := 14  -- using 24-hour format, 2 pm as 14

-- Prove that the starting time is 10 am given the conditions
theorem start_time_is_10_am (t : ℝ) :
  (speed_A_to_B * t + speed_B_to_A * t = distance_AB) →
  (meeting_time - t = 10) :=
sorry

end start_time_is_10_am_l1000_100016


namespace positive_integers_divide_n_plus_7_l1000_100064

theorem positive_integers_divide_n_plus_7 (n : ℕ) (hn_pos : 0 < n) : n ∣ n + 7 ↔ n = 1 ∨ n = 7 :=
by 
  sorry

end positive_integers_divide_n_plus_7_l1000_100064


namespace find_expression_l1000_100083

variables {x y : ℝ}

theorem find_expression
  (h1: 3 * x + y = 5)
  (h2: x + 3 * y = 6)
  : 10 * x^2 + 13 * x * y + 10 * y^2 = 97 :=
by
  sorry

end find_expression_l1000_100083


namespace eggs_per_omelet_l1000_100079

theorem eggs_per_omelet:
  let small_children_tickets := 53
  let older_children_tickets := 35
  let adult_tickets := 75
  let senior_tickets := 37
  let smallChildrenOmelets := small_children_tickets * 0.5
  let olderChildrenOmelets := older_children_tickets
  let adultOmelets := adult_tickets * 2
  let seniorOmelets := senior_tickets * 1.5
  let extra_omelets := 25
  let total_omelets := smallChildrenOmelets + olderChildrenOmelets + adultOmelets + seniorOmelets + extra_omelets
  let total_eggs := 584
  total_eggs / total_omelets = 2 := 
by
  sorry

end eggs_per_omelet_l1000_100079


namespace regression_prediction_l1000_100081

theorem regression_prediction
  (slope : ℝ) (centroid_x centroid_y : ℝ) (b : ℝ)
  (h_slope : slope = 1.23)
  (h_centroid : centroid_x = 4 ∧ centroid_y = 5)
  (h_intercept : centroid_y = slope * centroid_x + b)
  (x : ℝ) (h_x : x = 10) :
  centroid_y = 5 →
  slope = 1.23 →
  x = 10 →
  b = 5 - 1.23 * 4 →
  (slope * x + b) = 12.38 :=
by
  intros
  sorry

end regression_prediction_l1000_100081


namespace quadrilateral_is_parallelogram_l1000_100072

theorem quadrilateral_is_parallelogram
  (AB BC CD DA : ℝ)
  (K L M N : ℝ)
  (H₁ : K = (AB + BC) / 2)
  (H₂ : L = (BC + CD) / 2)
  (H₃ : M = (CD + DA) / 2)
  (H₄ : N = (DA + AB) / 2)
  (H : K + M + L + N = (AB + BC + CD + DA) / 2)
  : ∃ P Q R S : ℝ, P ≠ Q ∧ Q ≠ R ∧ R ≠ S ∧ S ≠ P ∧ 
    (P + R = AB) ∧ (Q + S = CD)  := 
sorry

end quadrilateral_is_parallelogram_l1000_100072


namespace find_a_in_terms_of_x_l1000_100041

theorem find_a_in_terms_of_x (a b x : ℝ) (h₁ : a ≠ b) (h₂ : a^3 - b^3 = 22 * x^3) (h₃ : a - b = 2 * x) : 
  a = x * (1 + (Real.sqrt (40 / 3)) / 2) ∨ a = x * (1 - (Real.sqrt (40 / 3)) / 2) :=
by
  sorry

end find_a_in_terms_of_x_l1000_100041


namespace max_value_3x_plus_4y_l1000_100046

theorem max_value_3x_plus_4y (x y : ℝ) : x^2 + y^2 = 14 * x + 6 * y + 6 → 3 * x + 4 * y ≤ 73 :=
sorry

end max_value_3x_plus_4y_l1000_100046


namespace quilt_patch_cost_is_correct_l1000_100005

noncomputable def quilt_area : ℕ := 16 * 20

def patch_area : ℕ := 4

def first_10_patch_cost : ℕ := 10

def discount_patch_cost : ℕ := 5

def total_patches (quilt_area patch_area : ℕ) : ℕ := quilt_area / patch_area

def cost_for_first_10 (first_10_patch_cost : ℕ) : ℕ := 10 * first_10_patch_cost

def cost_for_discounted (total_patches first_10_patch_cost discount_patch_cost : ℕ) : ℕ :=
  (total_patches - 10) * discount_patch_cost

def total_cost (cost_for_first_10 cost_for_discounted : ℕ) : ℕ :=
  cost_for_first_10 + cost_for_discounted

theorem quilt_patch_cost_is_correct :
  total_cost (cost_for_first_10 first_10_patch_cost)
             (cost_for_discounted (total_patches quilt_area patch_area) first_10_patch_cost discount_patch_cost) = 450 :=
by
  sorry

end quilt_patch_cost_is_correct_l1000_100005


namespace total_cost_of_apples_l1000_100066

variable (num_apples_per_bag cost_per_bag num_apples : ℕ)
#check num_apples_per_bag = 50
#check cost_per_bag = 8
#check num_apples = 750

theorem total_cost_of_apples : 
  (num_apples_per_bag = 50) → 
  (cost_per_bag = 8) → 
  (num_apples = 750) → 
  (num_apples / num_apples_per_bag * cost_per_bag = 120) :=
by
  intros
  sorry

end total_cost_of_apples_l1000_100066


namespace length_of_nylon_cord_l1000_100062

-- Definitions based on the conditions
def tree : ℝ := 0 -- Tree as the center point (assuming a 0 for simplicity)
def distance_ran : ℝ := 30 -- Dog ran approximately 30 feet

-- The theorem to prove
theorem length_of_nylon_cord : (distance_ran / 2) = 15 := by
  -- Assuming the dog ran along the diameter of the circle
  -- and the length of the cord is the radius of that circle.
  sorry

end length_of_nylon_cord_l1000_100062


namespace maximum_f_value_l1000_100040

noncomputable def otimes (a b : ℝ) : ℝ :=
if a ≤ b then a else b

noncomputable def f (x : ℝ) : ℝ :=
otimes (3 * x^2 + 6) (23 - x^2)

theorem maximum_f_value : ∃ x : ℝ, ∀ y : ℝ, f x ≤ f y ∧ f x = 4 :=
sorry

end maximum_f_value_l1000_100040


namespace five_consecutive_product_div_24_l1000_100070

theorem five_consecutive_product_div_24 (n : ℤ) : 
  24 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) := 
sorry

end five_consecutive_product_div_24_l1000_100070


namespace three_xy_eq_24_l1000_100059

variable {x y : ℝ}

theorem three_xy_eq_24 (h : x * (x + 3 * y) = x^2 + 24) : 3 * x * y = 24 :=
sorry

end three_xy_eq_24_l1000_100059


namespace brownies_left_is_zero_l1000_100091

-- Definitions of the conditions
def total_brownies : ℝ := 24
def tina_lunch : ℝ := 1.5 * 5
def tina_dinner : ℝ := 0.5 * 5
def tina_total : ℝ := tina_lunch + tina_dinner
def husband_total : ℝ := 0.75 * 5
def guests_total : ℝ := 2.5 * 2
def daughter_total : ℝ := 2 * 3

-- Formulate the proof statement
theorem brownies_left_is_zero :
    total_brownies - (tina_total + husband_total + guests_total + daughter_total) = 0 := by
  sorry

end brownies_left_is_zero_l1000_100091


namespace train_speed_and_length_l1000_100008

theorem train_speed_and_length 
  (x y : ℝ)
  (h1 : 60 * x = 1000 + y)
  (h2 : 40 * x = 1000 - y) :
  x = 20 ∧ y = 200 :=
by
  sorry

end train_speed_and_length_l1000_100008


namespace smaller_number_eq_l1000_100074

variable (m n t s : ℝ)
variable (h_ratio : m / n = t)
variable (h_sum : m + n = s)
variable (h_t_gt_one : t > 1)

theorem smaller_number_eq : n = s / (1 + t) :=
by sorry

end smaller_number_eq_l1000_100074


namespace father_cards_given_l1000_100014

-- Defining the conditions
def Janessa_initial_cards : Nat := 4
def eBay_cards : Nat := 36
def bad_cards : Nat := 4
def dexter_cards : Nat := 29
def janessa_kept_cards : Nat := 20

-- Proving the number of cards father gave her
theorem father_cards_given : ∃ n : Nat, n = 13 ∧ (Janessa_initial_cards + eBay_cards - bad_cards + n = dexter_cards + janessa_kept_cards) := 
by
  sorry

end father_cards_given_l1000_100014


namespace log_function_passes_through_point_l1000_100001

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  Real.log (x - 1) / Real.log a - 1

theorem log_function_passes_through_point {a : ℝ} (h1 : a > 0) (h2 : a ≠ 1) :
  f a 2 = -1 :=
by
  -- To complete the proof, one would argue about the properties of logarithms in specific bases.
  sorry

end log_function_passes_through_point_l1000_100001


namespace area_of_region_is_12_l1000_100025

def region_area : ℝ :=
  let f1 (x : ℝ) : ℝ := |x - 2|
  let f2 (x : ℝ) : ℝ := 5 - |x + 1|
  let valid_region (x y : ℝ) : Prop := f1 x ≤ y ∧ y ≤ f2 x
  12

theorem area_of_region_is_12 :
  ∃ (area : ℝ), region_area = 12 := by
  use 12
  sorry

end area_of_region_is_12_l1000_100025


namespace bowls_remaining_l1000_100043

def initial_bowls : ℕ := 250

def customers_purchases : List (ℕ × ℕ) :=
  [(5, 7), (10, 15), (15, 22), (5, 36), (7, 46), (8, 0)]

def reward_ranges (bought : ℕ) : ℕ :=
  if bought >= 5 && bought <= 9 then 1
  else if bought >= 10 && bought <= 19 then 3
  else if bought >= 20 && bought <= 29 then 6
  else if bought >= 30 && bought <= 39 then 8
  else if bought >= 40 then 12
  else 0

def total_free_bowls : ℕ :=
  List.foldl (λ acc (n, b) => acc + n * reward_ranges b) 0 customers_purchases

theorem bowls_remaining :
  initial_bowls - total_free_bowls = 1 := by
  sorry

end bowls_remaining_l1000_100043


namespace book_arrangement_l1000_100096

theorem book_arrangement (math_books : ℕ) (english_books : ℕ) (science_books : ℕ)
  (math_different : math_books = 4) 
  (english_different : english_books = 5) 
  (science_different : science_books = 2) :
  (Nat.factorial 3) * (Nat.factorial math_books) * (Nat.factorial english_books) * (Nat.factorial science_books) = 34560 := 
by
  sorry

end book_arrangement_l1000_100096


namespace clock_correct_time_fraction_l1000_100026

/-- A 12-hour digital clock problem:
A 12-hour digital clock displays the hour and minute of a day.
Whenever it is supposed to display a '1' or a '2', it mistakenly displays a '9'.
The fraction of the day during which the clock shows the correct time is 7/24.
-/
theorem clock_correct_time_fraction : (7 : ℚ) / 24 = 7 / 24 :=
by sorry

end clock_correct_time_fraction_l1000_100026


namespace time_difference_is_16_point_5_l1000_100038

noncomputable def time_difference : ℝ :=
  let danny_to_steve : ℝ := 33
  let steve_to_danny := 2 * danny_to_steve -- Steve takes twice the time as Danny
  let emma_to_houses : ℝ := 40
  let danny_halfway := danny_to_steve / 2 -- Halfway point for Danny
  let steve_halfway := steve_to_danny / 2 -- Halfway point for Steve
  let emma_halfway := emma_to_houses / 2 -- Halfway point for Emma
  -- Additional times to the halfway point
  let steve_additional := steve_halfway - danny_halfway
  let emma_additional := emma_halfway - danny_halfway
  -- The final result is the maximum of these times
  max steve_additional emma_additional

theorem time_difference_is_16_point_5 : time_difference = 16.5 :=
  by
  sorry

end time_difference_is_16_point_5_l1000_100038


namespace johns_new_weekly_earnings_l1000_100017

-- Definition of the initial weekly earnings
def initial_weekly_earnings := 40

-- Definition of the percent increase in earnings
def percent_increase := 100

-- Definition for the final weekly earnings after the raise
def final_weekly_earnings (initial_earnings : Nat) (percentage : Nat) := 
  initial_earnings + (initial_earnings * percentage / 100)

-- Theorem stating John’s final weekly earnings after the raise
theorem johns_new_weekly_earnings : final_weekly_earnings initial_weekly_earnings percent_increase = 80 :=
  by
  sorry

end johns_new_weekly_earnings_l1000_100017


namespace coloring_equilateral_triangle_l1000_100004

theorem coloring_equilateral_triangle :
  ∀ (A B C : Type) (color : A → Type) (d : A → A → ℝ),
  (∀ x y, d x y = 1 → color x = color y) :=
by sorry

end coloring_equilateral_triangle_l1000_100004


namespace find_x_of_orthogonal_vectors_l1000_100039

variable (x : ℝ)

def a : ℝ × ℝ × ℝ := (2, -3, 1)
def b : ℝ × ℝ × ℝ := (-4, 2, x)

theorem find_x_of_orthogonal_vectors (h : (2 * -4 + -3 * 2 + 1 * x) = 0) : x = 14 := by
  sorry

end find_x_of_orthogonal_vectors_l1000_100039


namespace minimum_value_expression_l1000_100067

theorem minimum_value_expression (a : ℝ) (h : a > 0) : 
  a + (a + 4) / a ≥ 5 :=
sorry

end minimum_value_expression_l1000_100067


namespace power_of_two_l1000_100006

theorem power_of_two (m n : ℕ) (h_m_pos : 0 < m) (h_n_pos : 0 < n) 
  (h_prime : Prime (m^(4^n + 1) - 1)) : 
  ∃ t : ℕ, n = 2^t :=
sorry

end power_of_two_l1000_100006


namespace negation_of_quadratic_statement_l1000_100045

variable {x a b : ℝ}

theorem negation_of_quadratic_statement (h : x = a ∨ x = b) : x^2 - (a + b) * x + ab = 0 := sorry

end negation_of_quadratic_statement_l1000_100045


namespace frosting_problem_equivalent_l1000_100007

/-
Problem:
Cagney can frost a cupcake every 15 seconds.
Lacey can frost a cupcake every 40 seconds.
Mack can frost a cupcake every 25 seconds.
Prove that together they can frost 79 cupcakes in 10 minutes.
-/

def cupcakes_frosted_together_in_10_minutes (rate_cagney rate_lacey rate_mack : ℕ) (time_minutes : ℕ) : ℕ :=
  let time_seconds := time_minutes * 60
  let rate_cagney := 1 / 15
  let rate_lacey := 1 / 40
  let rate_mack := 1 / 25
  let combined_rate := rate_cagney + rate_lacey + rate_mack
  combined_rate * time_seconds

theorem frosting_problem_equivalent:
  cupcakes_frosted_together_in_10_minutes 1 1 1 10 = 79 := by
  sorry

end frosting_problem_equivalent_l1000_100007


namespace fencing_required_l1000_100012

theorem fencing_required {length width : ℝ} 
  (uncovered_side : length = 20)
  (field_area : length * width = 50) :
  2 * width + length = 25 :=
by
  sorry

end fencing_required_l1000_100012


namespace safety_rent_a_car_cost_per_mile_l1000_100034

/-
Problem:
Prove that the cost per mile for Safety Rent-a-Car is 0.177 dollars, given that the total cost of renting an intermediate-size car for 150 miles is the same for Safety Rent-a-Car and City Rentals, with their respective pricing schemes.
-/

theorem safety_rent_a_car_cost_per_mile :
  let x := 21.95
  let y := 18.95
  let z := 0.21
  (x + 150 * real_safety_per_mile) = (y + 150 * z) ↔ real_safety_per_mile = 0.177 :=
by
  sorry

end safety_rent_a_car_cost_per_mile_l1000_100034


namespace part_a_part_b_l1000_100021

-- Assuming existence of function S satisfying certain properties
variable (S : Type → Type → Type → ℝ)

-- Part (a)
theorem part_a (A B C : Type) : 
  S A B C = -S B A C ∧ S A B C = S B C A :=
sorry

-- Part (b)
theorem part_b (A B C D : Type) : 
  S A B C = S D A B + S D B C + S D C A :=
sorry

end part_a_part_b_l1000_100021


namespace integral_cos_2x_eq_half_l1000_100033

theorem integral_cos_2x_eq_half :
  ∫ x in (0:ℝ)..(Real.pi / 4), Real.cos (2 * x) = 1 / 2 := by
sorry

end integral_cos_2x_eq_half_l1000_100033


namespace minimum_perimeter_triangle_l1000_100090

noncomputable def minimum_perimeter (a b c : ℝ) (cos_C : ℝ) (ha : a + b = 10) (hroot : 2 * cos_C^2 - 3 * cos_C - 2 = 0) 
  : ℝ :=
  a + b + c

theorem minimum_perimeter_triangle (a b c : ℝ) (cos_C : ℝ)
  (ha : a + b = 10)
  (hroot : 2 * cos_C^2 - 3 * cos_C - 2 = 0)
  (cos_C_valid : cos_C = -1/2) :
  (minimum_perimeter a b c cos_C ha hroot) = 10 + 5 * Real.sqrt 3 :=
sorry

end minimum_perimeter_triangle_l1000_100090


namespace remaining_marbles_l1000_100054

theorem remaining_marbles (initial_marbles : ℕ) (num_customers : ℕ) (marble_range : List ℕ)
  (h_initial : initial_marbles = 2500)
  (h_customers : num_customers = 50)
  (h_range : marble_range = List.range' 1 50)
  (disjoint_range : ∀ (a b : ℕ), a ∈ marble_range → b ∈ marble_range → a ≠ b → a + b ≤ 50) :
  initial_marbles - (num_customers * (50 + 1) / 2) = 1225 :=
by
  sorry

end remaining_marbles_l1000_100054


namespace distance_between_intersections_l1000_100088

theorem distance_between_intersections :
  let a := 3
  let b := 2
  let c := -7
  let x1 := (-1 + Real.sqrt 22) / 3
  let x2 := (-1 - Real.sqrt 22) / 3
  let distance := abs (x1 - x2)
  let p := 88  -- 2^2 * 22 = 88
  let q := 9   -- 3^2 = 9
  distance = 2 * Real.sqrt 22 / 3 →
  p - q = 79 :=
by
  sorry

end distance_between_intersections_l1000_100088


namespace arithmetic_sequence_property_l1000_100018

theorem arithmetic_sequence_property 
  (a : ℕ → ℤ) 
  (h₁ : ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d)
  (h₂ : a 1 + 2 * a 8 + a 15 = 96) :
  2 * a 9 - a 10 = 24 := 
sorry

end arithmetic_sequence_property_l1000_100018


namespace intersection_point_parallel_line_through_intersection_l1000_100085

-- Definitions for the problem
def l1 (x y : ℝ) : Prop := x + 8 * y + 7 = 0
def l2 (x y : ℝ) : Prop := 2 * x + y - 1 = 0
def l3 (x y : ℝ) : Prop := x + y + 1 = 0
def parallel (x y c : ℝ) : Prop := x + y + c = 0
def point (x y : ℝ) : Prop := x = 1 ∧ y = -1

-- (1) Proof that the intersection point of l1 and l2 is (1, -1)
theorem intersection_point : ∃ (x y : ℝ), l1 x y ∧ l2 x y ∧ point x y :=
by 
  sorry

-- (2) Proof that the line passing through the intersection point of l1 and l2
-- which is parallel to l3 is x + y = 0
theorem parallel_line_through_intersection : ∃ (c : ℝ), parallel 1 (-1) c ∧ c = 0 :=
by 
  sorry

end intersection_point_parallel_line_through_intersection_l1000_100085


namespace consecutive_numbers_count_l1000_100093

-- Definitions and conditions
variables (n : ℕ) (x : ℕ)
axiom avg_condition : (2 * 33 = 2 * x + n - 1)
axiom highest_num_condition : (x + (n - 1) = 36)

-- Thm statement
theorem consecutive_numbers_count : n = 7 :=
by
  sorry

end consecutive_numbers_count_l1000_100093


namespace dogwood_trees_initial_count_l1000_100052

theorem dogwood_trees_initial_count 
  (dogwoods_today : ℕ) 
  (dogwoods_tomorrow : ℕ) 
  (final_dogwoods : ℕ)
  (total_planted : ℕ := dogwoods_today + dogwoods_tomorrow)
  (initial_dogwoods := final_dogwoods - total_planted)
  (h : dogwoods_today = 41)
  (h1 : dogwoods_tomorrow = 20)
  (h2 : final_dogwoods = 100) : 
  initial_dogwoods = 39 := 
by sorry

end dogwood_trees_initial_count_l1000_100052


namespace linear_eq_substitution_l1000_100078

theorem linear_eq_substitution (x y : ℝ) (h1 : 3 * x - 4 * y = 2) (h2 : x = 2 * y - 1) :
  3 * (2 * y - 1) - 4 * y = 2 :=
by
  sorry

end linear_eq_substitution_l1000_100078


namespace integer_roots_condition_l1000_100077

theorem integer_roots_condition (a : ℝ) (h_pos : 0 < a) :
  (∀ x y : ℤ, (a ^ 2 * x ^ 2 + a * x + 1 - 13 * a ^ 2 = 0) ∧ (a ^ 2 * y ^ 2 + a * y + 1 - 13 * a ^ 2 = 0)) ↔
  (a = 1 ∨ a = 1/3 ∨ a = 1/4) :=
by sorry

end integer_roots_condition_l1000_100077


namespace no_such_integers_exists_l1000_100035

theorem no_such_integers_exists :
  ∀ (P : ℕ → ℕ), (∀ x, P x = x ^ 2000 - x ^ 1000 + 1) →
  ¬(∃ (a : Fin 8002 → ℕ), Function.Injective a ∧
  (∀ i j k : Fin 8002, i ≠ j → j ≠ k → i ≠ k → a i * a j * a k ∣ P (a i) * P (a j) * P (a k))) := 
by
  intro P hP notExists
  have contra : ∃ (a : Fin 8002 → ℕ), Function.Injective a ∧
    (∀ i j k : Fin 8002, i ≠ j → j ≠ k → i ≠ k → a i * a j * a k ∣ P (a i) * P (a j) * P (a k)) := notExists
  sorry

end no_such_integers_exists_l1000_100035


namespace last_years_rate_per_mile_l1000_100076

-- Definitions from the conditions
variables (m : ℕ) (x : ℕ)

-- Condition 1: This year, walkers earn $2.75 per mile
def amount_per_mile_this_year : ℝ := 2.75

-- Condition 2: Last year's winner collected $44
def last_years_total_amount : ℕ := 44

-- Condition 3: Elroy will walk 5 more miles than last year's winner
def elroy_walks_more_miles (m : ℕ) : ℕ := m + 5

-- The main goal is to prove that last year's rate per mile was $4 given the conditions
theorem last_years_rate_per_mile (h1 : last_years_total_amount = m * x)
  (h2 : last_years_total_amount = (elroy_walks_more_miles m) * amount_per_mile_this_year) :
  x = 4 :=
by {
  sorry
}

end last_years_rate_per_mile_l1000_100076


namespace product_is_correct_l1000_100050

noncomputable def IKS := 521
noncomputable def KSI := 215
def product := 112015

theorem product_is_correct : IKS * KSI = product :=
by
  -- Proof yet to be constructed
  sorry

end product_is_correct_l1000_100050


namespace bus_travel_fraction_l1000_100023

theorem bus_travel_fraction :
  ∃ D : ℝ, D = 30.000000000000007 ∧
            (1 / 3) * D + 2 + (18 / 30) * D = D ∧
            (18 / 30) = (3 / 5) :=
by
  sorry

end bus_travel_fraction_l1000_100023


namespace percentage_sales_other_l1000_100084

theorem percentage_sales_other (p_pens p_pencils p_markers p_other : ℕ)
(h_pens : p_pens = 25)
(h_pencils : p_pencils = 30)
(h_markers : p_markers = 20)
(h_other : p_other = 100 - (p_pens + p_pencils + p_markers)): p_other = 25 :=
by
  rw [h_pens, h_pencils, h_markers] at h_other
  exact h_other


end percentage_sales_other_l1000_100084


namespace probability_non_defective_second_draw_l1000_100089

theorem probability_non_defective_second_draw 
  (total_products : ℕ)
  (defective_products : ℕ)
  (first_draw_defective : Bool)
  (second_draw_non_defective_probability : ℚ) : 
  total_products = 100 → 
  defective_products = 3 → 
  first_draw_defective = true → 
  second_draw_non_defective_probability = 97 / 99 :=
by
  intros h_total h_defective h_first_draw
  subst h_total
  subst h_defective
  subst h_first_draw
  sorry

end probability_non_defective_second_draw_l1000_100089


namespace union_card_ge_165_l1000_100032

open Finset

variable (A : Finset ℕ) (A_i : Fin (11) → Finset ℕ)
variable (hA : A.card = 225)
variable (hA_i_card : ∀ i, (A_i i).card = 45)
variable (hA_i_intersect : ∀ i j, i < j → ((A_i i) ∩ (A_i j)).card = 9)

theorem union_card_ge_165 : (Finset.biUnion Finset.univ A_i).card ≥ 165 := by sorry

end union_card_ge_165_l1000_100032


namespace zach_needs_more_money_zach_more_money_needed_l1000_100030

/-!
# Zach's Bike Savings Problem
Zach needs $100 to buy a brand new bike.
Weekly allowance: $5.
Earnings from mowing the lawn: $10.
Earnings from babysitting: $7 per hour.
Zach has already saved $65.
He will receive weekly allowance on Friday.
He will mow the lawn and babysit for 2 hours this Saturday.
Prove that Zach needs $6 more to buy the bike.
-/

def zach_current_savings : ℕ := 65
def bike_cost : ℕ := 100
def weekly_allowance : ℕ := 5
def mowing_earnings : ℕ := 10
def babysitting_rate : ℕ := 7
def babysitting_hours : ℕ := 2

theorem zach_needs_more_money : zach_current_savings + (weekly_allowance + mowing_earnings + (babysitting_rate * babysitting_hours)) = 94 :=
by sorry

theorem zach_more_money_needed : bike_cost - (zach_current_savings + (weekly_allowance + mowing_earnings + (babysitting_rate * babysitting_hours))) = 6 :=
by sorry

end zach_needs_more_money_zach_more_money_needed_l1000_100030


namespace units_digit_quotient_eq_one_l1000_100002

theorem units_digit_quotient_eq_one :
  (2^2023 + 3^2023) / 5 % 10 = 1 := by
  sorry

end units_digit_quotient_eq_one_l1000_100002


namespace abs_w_unique_l1000_100060

theorem abs_w_unique (w : ℂ) (h : w^2 - 6 * w + 40 = 0) : ∃! x : ℝ, x = Complex.abs w ∧ x = Real.sqrt 40 := by
  sorry

end abs_w_unique_l1000_100060


namespace parking_lot_wheels_l1000_100094

-- Define the conditions
def num_cars : Nat := 10
def num_bikes : Nat := 2
def wheels_per_car : Nat := 4
def wheels_per_bike : Nat := 2

-- Define the total number of wheels
def total_wheels : Nat := (num_cars * wheels_per_car) + (num_bikes * wheels_per_bike)

-- State the theorem
theorem parking_lot_wheels : total_wheels = 44 :=
by
  sorry

end parking_lot_wheels_l1000_100094


namespace factory_a_min_hours_l1000_100019

theorem factory_a_min_hours (x : ℕ) :
  (550 * x + (700 - 55 * x) / 45 * 495 ≤ 7260) → (8 ≤ x) :=
by
  sorry

end factory_a_min_hours_l1000_100019


namespace solution_to_g_inv_2_l1000_100065

noncomputable def g (x : ℝ) (c d : ℝ) : ℝ := 1 / (c * x + d)

theorem solution_to_g_inv_2 (c d : ℝ) (hc : c ≠ 0) (hd : d ≠ 0) :
    ∃ x : ℝ, g x c d = 2 ↔ x = (1 - 2 * d) / (2 * c) :=
by
  sorry

end solution_to_g_inv_2_l1000_100065


namespace sheila_tue_thu_hours_l1000_100015

def hours_mwf : ℕ := 8
def days_mwf : ℕ := 3
def total_hours_mwf : ℕ := hours_mwf * days_mwf

def weekly_earnings : ℕ := 360
def hourly_rate : ℕ := 10
def earnings_mwf : ℕ := total_hours_mwf * hourly_rate

def earnings_tue_thu : ℕ := weekly_earnings - earnings_mwf
def hours_tue_thu : ℕ := earnings_tue_thu / hourly_rate

theorem sheila_tue_thu_hours : hours_tue_thu = 12 := by
  -- proof omitted
  sorry

end sheila_tue_thu_hours_l1000_100015


namespace distance_to_Tianbo_Mountain_l1000_100063

theorem distance_to_Tianbo_Mountain : ∀ (x y : ℝ), 
  (x ≠ 0) ∧ 
  (y = 3) ∧ 
  (∀ v, v = (4 * y + x) * ((2 * x - 8) / v)) ∧ 
  (2 * (y * x) = 8 * y + x^2 - 4 * x) 
  → 
  (x + y = 9) := 
by
  sorry

end distance_to_Tianbo_Mountain_l1000_100063


namespace sum_of_consecutive_integers_product_2730_eq_42_l1000_100031

theorem sum_of_consecutive_integers_product_2730_eq_42 :
  ∃ x : ℤ, x * (x + 1) * (x + 2) = 2730 ∧ x + (x + 1) + (x + 2) = 42 :=
by
  sorry

end sum_of_consecutive_integers_product_2730_eq_42_l1000_100031


namespace muffin_cost_ratio_l1000_100095

theorem muffin_cost_ratio (m b : ℝ) 
  (h1 : 5 * m + 4 * b = 20)
  (h2 : 3 * (5 * m + 4 * b) = 60)
  (h3 : 3 * m + 18 * b = 60) :
  m / b = 13 / 4 :=
by
  sorry

end muffin_cost_ratio_l1000_100095


namespace triangle_area_is_9_l1000_100053

-- Define the vertices of the triangle
def x1 : ℝ := 1
def y1 : ℝ := 2
def x2 : ℝ := 4
def y2 : ℝ := 5
def x3 : ℝ := 6
def y3 : ℝ := 1

-- Define the area calculation formula for the triangle
def triangle_area (x1 y1 x2 y2 x3 y3 : ℝ) : ℝ :=
  0.5 * abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

-- The proof statement
theorem triangle_area_is_9 :
  triangle_area x1 y1 x2 y2 x3 y3 = 9 :=
by
  sorry

end triangle_area_is_9_l1000_100053


namespace quadratic_decomposition_l1000_100024

theorem quadratic_decomposition (a b c : ℝ) :
  (∀ x : ℝ, 6 * x^2 + 72 * x + 432 = a * (x + b)^2 + c) → a + b + c = 228 :=
sorry

end quadratic_decomposition_l1000_100024


namespace cannot_form_set_of_good_friends_of_wang_ming_l1000_100075

def is_well_defined_set (description : String) : Prop := sorry  -- Placeholder for the formal definition.

theorem cannot_form_set_of_good_friends_of_wang_ming :
  ¬ is_well_defined_set "Good friends of Wang Ming" :=
sorry

end cannot_form_set_of_good_friends_of_wang_ming_l1000_100075


namespace lcm_105_360_eq_2520_l1000_100055

theorem lcm_105_360_eq_2520 :
  Nat.lcm 105 360 = 2520 :=
by
  have h1 : 105 = 3 * 5 * 7 := by norm_num
  have h2 : 360 = 2^3 * 3^2 * 5 := by norm_num
  rw [h1, h2]
  sorry

end lcm_105_360_eq_2520_l1000_100055


namespace num_multiples_of_three_in_ap_l1000_100061

variable (a : ℕ → ℚ)  -- Defining the arithmetic sequence

def first_term (a1 : ℚ) := a 1 = a1
def eighth_term (a8 : ℚ) := a 8 = a8
def general_term (d : ℚ) := ∀ n : ℕ, a n = 9 + (n - 1) * d
def multiple_of_three (n : ℕ) := ∃ k : ℕ, a n = 3 * k

theorem num_multiples_of_three_in_ap 
  (a : ℕ → ℚ)
  (h1 : first_term a 9)
  (h2 : eighth_term a 12) :
  ∃ n : ℕ, n = 288 ∧ ∃ l : ℕ → Prop, ∀ k : ℕ, l k → multiple_of_three a (k * 7 + 1) :=
sorry

end num_multiples_of_three_in_ap_l1000_100061


namespace tan_arithmetic_seq_value_l1000_100068

variable {a : ℕ → ℝ}
variable (d : ℝ)

-- Define the arithmetic sequence
def arithmetic_seq (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n : ℕ, a n = a 0 + n * d

-- Given conditions and the final proof goal
theorem tan_arithmetic_seq_value (h_arith : arithmetic_seq a d)
    (h_sum : a 0 + a 6 + a 12 = Real.pi) :
    Real.tan (a 1 + a 11) = -Real.sqrt 3 := sorry

end tan_arithmetic_seq_value_l1000_100068


namespace union_M_N_is_U_l1000_100009

-- Defining the universal set as the set of real numbers
def U : Set ℝ := Set.univ

-- Defining the set M
def M : Set ℝ := {x | x > 0}

-- Defining the set N
def N : Set ℝ := {x | x^2 >= x}

-- Stating the theorem that M ∪ N = U
theorem union_M_N_is_U : M ∪ N = U :=
  sorry

end union_M_N_is_U_l1000_100009


namespace b101_mod_49_l1000_100028

-- Definitions based on conditions
def b (n : ℕ) : ℕ := 5^n + 7^n

-- The formal statement of the proof problem
theorem b101_mod_49 : b 101 % 49 = 12 := by
  sorry

end b101_mod_49_l1000_100028


namespace matrix_no_solution_neg_two_l1000_100092

-- Define the matrix and vector equation
def matrix_equation (a x y : ℝ) : Prop :=
  (a * x + 2 * y = a + 2) ∧ (2 * x + a * y = 2 * a)

-- Define the condition for no solution
def no_solution_condition (a : ℝ) : Prop :=
  (a/2 = 2/a) ∧ (a/2 ≠ (a + 2) / (2 * a))

-- Theorem stating that a = -2 is the necessary condition for no solution
theorem matrix_no_solution_neg_two (a : ℝ) : no_solution_condition a → a = -2 := by
  sorry

end matrix_no_solution_neg_two_l1000_100092


namespace fraction_is_one_over_three_l1000_100087

variable (x : ℚ) -- Let the fraction x be a rational number
variable (num : ℚ) -- Let the number be a rational number

theorem fraction_is_one_over_three (h1 : num = 45) (h2 : x * num - 5 = 10) : x = 1 / 3 := by
  sorry

end fraction_is_one_over_three_l1000_100087


namespace problem_statement_l1000_100051

theorem problem_statement (x : ℝ) (hx : x + 1/x = Real.sqrt 5) : 
  x^12 - 7 * x^8 + x^4 = 343 :=
sorry

end problem_statement_l1000_100051


namespace rectangle_extraction_l1000_100013

theorem rectangle_extraction (m : ℤ) (h1 : m > 12) : 
  ∃ (x y : ℤ), x ≤ y ∧ x * y > m ∧ x * (y - 1) < m :=
by
  sorry

end rectangle_extraction_l1000_100013


namespace MEMOrable_rectangle_count_l1000_100080

section MEMOrable_rectangles

variables (K L : ℕ) (hK : K > 0) (hL : L > 0) 

/-- In a 2K x 2L board, if the ant starts at (1,1) and ends at (2K, 2L),
    and some squares may remain unvisited forming a MEMOrable rectangle,
    then the number of such MEMOrable rectangles is (K(K+1)L(L+1))/2. -/
theorem MEMOrable_rectangle_count :
  ∃ (n : ℕ), n = K * (K + 1) * L * (L + 1) / 2 :=
by
  sorry

end MEMOrable_rectangles

end MEMOrable_rectangle_count_l1000_100080


namespace gain_percent_l1000_100000

-- Let C be the cost price of one chocolate
-- Let S be the selling price of one chocolate
-- Given: 35 * C = 21 * S
-- Prove: The gain percent is 66.67%

theorem gain_percent (C S : ℝ) (h : 35 * C = 21 * S) : (S - C) / C * 100 = 200 / 3 :=
by sorry

end gain_percent_l1000_100000
