import Mathlib

namespace total_conference_games_scheduled_l404_40454

-- Definitions of the conditions
def num_divisions : ℕ := 2
def teams_per_division : ℕ := 6
def intradivision_games_per_pair : ℕ := 3
def interdivision_games_per_pair : ℕ := 2

-- The statement to prove the total number of conference games
theorem total_conference_games_scheduled : 
  (num_divisions * (teams_per_division * (teams_per_division - 1) * intradivision_games_per_pair) / 2) 
  + (teams_per_division * teams_per_division * interdivision_games_per_pair) = 162 := 
by
  sorry

end total_conference_games_scheduled_l404_40454


namespace selection_ways_l404_40451

/-- 
A math interest group in a vocational school consists of 4 boys and 3 girls. 
If 3 students are randomly selected from these 7 students to participate in a math competition, 
and the selection must include both boys and girls, then the number of different ways to select the 
students is 30.
-/
theorem selection_ways (B G : ℕ) (students : ℕ) (selections : ℕ) (condition_boys_girls : B = 4 ∧ G = 3)
  (condition_students : students = B + G) (condition_selections : selections = 3) :
  (B = 4 ∧ G = 3 ∧ students = 7 ∧ selections = 3) → 
  ∃ (res : ℕ), res = 30 :=
by
  sorry

end selection_ways_l404_40451


namespace initial_cd_count_l404_40455

variable (X : ℕ)

theorem initial_cd_count (h1 : (2 / 3 : ℝ) * X + 8 = 22) : X = 21 :=
by
  sorry

end initial_cd_count_l404_40455


namespace eval_expr_l404_40422

theorem eval_expr : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 262400 := by
  sorry

end eval_expr_l404_40422


namespace sculptures_not_on_display_eq_1200_l404_40415

-- Define the number of pieces of art in the gallery
def total_pieces_art := 2700

-- Define the number of pieces on display (1/3 of total pieces)
def pieces_on_display := total_pieces_art / 3

-- Define the number of pieces not on display
def pieces_not_on_display := total_pieces_art - pieces_on_display

-- Define the number of sculptures on display (1/6 of pieces on display)
def sculptures_on_display := pieces_on_display / 6

-- Define the number of paintings not on display (1/3 of pieces not on display)
def paintings_not_on_display := pieces_not_on_display / 3

-- Prove the number of sculptures not on display
theorem sculptures_not_on_display_eq_1200 :
  total_pieces_art = 2700 →
  pieces_on_display = total_pieces_art / 3 →
  pieces_not_on_display = total_pieces_art - pieces_on_display →
  sculptures_on_display = pieces_on_display / 6 →
  paintings_not_on_display = pieces_not_on_display / 3 →
  pieces_not_on_display - paintings_not_on_display = 1200 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end sculptures_not_on_display_eq_1200_l404_40415


namespace value_of_m_sub_n_l404_40459

theorem value_of_m_sub_n (m n : ℤ) (h1 : |m| = 5) (h2 : n^2 = 36) (h3 : m * n < 0) : m - n = 11 ∨ m - n = -11 := 
by 
  sorry

end value_of_m_sub_n_l404_40459


namespace sufficient_condition_not_necessary_condition_l404_40406

/--
\(a > 1\) is a sufficient but not necessary condition for \(\frac{1}{a} < 1\).
-/
theorem sufficient_condition (a : ℝ) (h : a > 1) : 1 / a < 1 :=
by
  sorry

theorem not_necessary_condition (a : ℝ) (h : 1 / a < 1) : a > 1 ∨ a < 0 :=
by
  sorry

end sufficient_condition_not_necessary_condition_l404_40406


namespace at_least_one_basketball_selected_l404_40417

theorem at_least_one_basketball_selected (balls : Finset ℕ) (basketballs : Finset ℕ) (volleyballs : Finset ℕ) :
  basketballs.card = 6 → volleyballs.card = 2 → balls ⊆ (basketballs ∪ volleyballs) →
  balls.card = 3 → ∃ b ∈ balls, b ∈ basketballs :=
by
  intros h₁ h₂ h₃ h₄
  sorry

end at_least_one_basketball_selected_l404_40417


namespace total_family_members_l404_40438

variable (members_father_side : Nat) (percent_incr : Nat)
variable (members_mother_side := members_father_side + (members_father_side * percent_incr / 100))
variable (total_members := members_father_side + members_mother_side)

theorem total_family_members 
  (h1 : members_father_side = 10) 
  (h2 : percent_incr = 30) :
  total_members = 23 :=
by
  sorry

end total_family_members_l404_40438


namespace num_solutions_non_negative_reals_l404_40413

-- Define the system of equations as a function to express the cyclic nature
def system_of_equations (n : ℕ) (x : ℕ → ℝ) (k : ℕ) : Prop :=
  x (k + 1 % n) + (x (if k = 0 then n else k) ^ 2) = 4 * x (if k = 0 then n else k)

-- Define the main theorem stating the number of solutions
theorem num_solutions_non_negative_reals {n : ℕ} (hn : 0 < n) : 
  ∃ (s : Finset (ℕ → ℝ)), (∀ x ∈ s, ∀ k, 0 ≤ (x k) ∧ system_of_equations n x k) ∧ s.card = 2^n :=
sorry

end num_solutions_non_negative_reals_l404_40413


namespace dinner_cost_per_kid_l404_40480

theorem dinner_cost_per_kid
  (row_ears : ℕ)
  (seeds_bag : ℕ)
  (seeds_ear : ℕ)
  (pay_row : ℝ)
  (bags_used : ℕ)
  (dinner_fraction : ℝ)
  (h1 : row_ears = 70)
  (h2 : seeds_bag = 48)
  (h3 : seeds_ear = 2)
  (h4 : pay_row = 1.5)
  (h5 : bags_used = 140)
  (h6 : dinner_fraction = 0.5) :
  ∃ (dinner_cost : ℝ), dinner_cost = 36 :=
by
  sorry

end dinner_cost_per_kid_l404_40480


namespace find_b_squared_l404_40470

theorem find_b_squared :
  let ellipse_eq := ∀ x y : ℝ, x^2 / 25 + y^2 / b^2 = 1
  let hyperbola_eq := ∀ x y : ℝ, x^2 / 225 - y^2 / 144 = 1 / 36
  let coinciding_foci := 
    let c_ellipse := Real.sqrt (25 - b^2)
    let c_hyperbola := Real.sqrt ((225 / 36) + (144 / 36))
    c_ellipse = c_hyperbola
  ellipse_eq ∧ hyperbola_eq ∧ coinciding_foci → b^2 = 14.75
:= by sorry

end find_b_squared_l404_40470


namespace range_of_values_for_sqrt_l404_40414

theorem range_of_values_for_sqrt (x : ℝ) : (x + 3 ≥ 0) ↔ (x ≥ -3) :=
by
  sorry

end range_of_values_for_sqrt_l404_40414


namespace rectangle_area_eq_six_l404_40425

-- Define the areas of the small squares
def smallSquareArea : ℝ := 1

-- Define the number of small squares
def numberOfSmallSquares : ℤ := 2

-- Define the area of the larger square
def largeSquareArea : ℝ := (2 ^ 2)

-- Define the area of rectangle ABCD
def areaRectangleABCD : ℝ :=
  (numberOfSmallSquares * smallSquareArea) + largeSquareArea

-- The theorem we want to prove
theorem rectangle_area_eq_six :
  areaRectangleABCD = 6 := by sorry

end rectangle_area_eq_six_l404_40425


namespace ball_bounce_height_l404_40489

theorem ball_bounce_height (b : ℕ) (h₀: ℝ) (r: ℝ) (h_final: ℝ) :
  h₀ = 200 ∧ r = 3 / 4 ∧ h_final = 25 →
  200 * (3 / 4) ^ b < 25 ↔ b ≥ 25 := by
  sorry

end ball_bounce_height_l404_40489


namespace larger_number_is_400_l404_40448

def problem_statement : Prop :=
  ∃ (a b hcf lcm num1 num2 : ℕ),
  hcf = 25 ∧
  a = 14 ∧
  b = 16 ∧
  lcm = hcf * a * b ∧
  num1 = hcf * a ∧
  num2 = hcf * b ∧
  num1 < num2 ∧
  num2 = 400

theorem larger_number_is_400 : problem_statement :=
  sorry

end larger_number_is_400_l404_40448


namespace three_digit_numbers_divide_26_l404_40487

def divides (d n : ℕ) : Prop := ∃ k, n = d * k

theorem three_digit_numbers_divide_26 (a b c : ℕ) (h₀ : a ≠ 0) (h₁ : a < 10) (h₂ : b < 10) (h₃ : c < 10) :
  (divides 26 (a^2 + b^2 + c^2)) ↔ 
    ((a = 1 ∧ b = 0 ∧ c = 0) ∨
     (a = 1 ∧ b = 1 ∧ c = 0) ∨
     (a = 3 ∧ b = 2 ∧ c = 0) ∨
     (a = 5 ∧ b = 1 ∧ c = 0) ∨
     (a = 4 ∧ b = 3 ∧ c = 1)) :=
by 
  sorry

end three_digit_numbers_divide_26_l404_40487


namespace number_doubled_is_12_l404_40494

theorem number_doubled_is_12 (A B C D E : ℝ) (h1 : (A + B + C + D + E) / 5 = 6.8)
  (X : ℝ) (h2 : ((A + B + C + D + E - X) + 2 * X) / 5 = 9.2) : X = 12 :=
by
  sorry

end number_doubled_is_12_l404_40494


namespace angle_420_mod_360_eq_60_l404_40400

def angle_mod_equiv (a b : ℕ) : Prop := a % 360 = b

theorem angle_420_mod_360_eq_60 : angle_mod_equiv 420 60 := 
by
  sorry

end angle_420_mod_360_eq_60_l404_40400


namespace population_net_increase_l404_40462

-- Define the birth rate and death rate conditions
def birth_rate := 4 / 2 -- people per second
def death_rate := 2 / 2 -- people per second
def net_increase_per_sec := birth_rate - death_rate -- people per second

-- Define the duration of one day in seconds
def seconds_in_a_day := 24 * 3600 -- seconds

-- Define the problem to prove
theorem population_net_increase :
  net_increase_per_sec * seconds_in_a_day = 86400 :=
by
  sorry

end population_net_increase_l404_40462


namespace train_speed_l404_40432

theorem train_speed (length : ℝ) (time : ℝ) (speed : ℝ) 
    (h1 : length = 55) 
    (h2 : time = 5.5) 
    (h3 : speed = (length / time) * (3600 / 1000)) : 
    speed = 36 :=
sorry

end train_speed_l404_40432


namespace sqrt_of_second_number_l404_40441

-- Given condition: the arithmetic square root of a natural number n is x
variable (x : ℕ)
def first_number := x ^ 2
def second_number := first_number + 1

-- The theorem statement we want to prove
theorem sqrt_of_second_number (x : ℕ) : Real.sqrt (x^2 + 1) = Real.sqrt (first_number x + 1) :=
by
  sorry

end sqrt_of_second_number_l404_40441


namespace order_fractions_l404_40493

theorem order_fractions : (16/13 : ℚ) < 21/17 ∧ 21/17 < 20/15 :=
by {
  -- use cross-multiplication:
  -- 16*17 < 21*13 -> 272 < 273 -> true
  -- 16*15 < 20*13 -> 240 < 260 -> true
  -- 21*15 < 20*17 -> 315 < 340 -> true
  sorry
}

end order_fractions_l404_40493


namespace triangle_shape_area_l404_40484

theorem triangle_shape_area (a b : ℕ) (area_small area_middle area_large : ℕ) :
  a = 2 →
  b = 2 →
  area_small = (1 / 2) * a * b →
  area_middle = 2 * area_small →
  area_large = 2 * area_middle →
  area_small + area_middle + area_large = 14 :=
by
  intros
  sorry

end triangle_shape_area_l404_40484


namespace problem_statement_l404_40464

-- Define the constants and variables
variables (x y z a b c : ℝ)

-- Define the conditions given in the problem
def condition1 : Prop := x / a + y / b + z / c = 4
def condition2 : Prop := a / x + b / y + c / z = 1

-- State the theorem that proves the question equals the correct answer
theorem problem_statement (h1 : condition1 x y z a b c) (h2 : condition2 x y z a b c) :
    x^2 / a^2 + y^2 / b^2 + z^2 / c^2 = 12 :=
sorry

end problem_statement_l404_40464


namespace drum_oil_ratio_l404_40427

theorem drum_oil_ratio (C_X C_Y : ℝ) (h1 : (1 / 2) * C_X + (1 / 5) * C_Y = 0.45 * C_Y) : 
  C_Y / C_X = 2 :=
by
  -- Cannot provide the proof
  sorry

end drum_oil_ratio_l404_40427


namespace determine_x_l404_40418

theorem determine_x (x : ℚ) (h : ∀ y : ℚ, 10 * x * y - 15 * y + 3 * x - (9 / 2) = 0) : x = 3 / 2 :=
sorry

end determine_x_l404_40418


namespace find_weights_l404_40421

theorem find_weights (x y z : ℕ) (h1 : x + y + z = 11) (h2 : 3 * x + 7 * y + 14 * z = 108) :
  x = 1 ∧ y = 5 ∧ z = 5 :=
by
  sorry

end find_weights_l404_40421


namespace playground_length_l404_40426

theorem playground_length
  (L_g : ℝ) -- length of the garden
  (L_p : ℝ) -- length of the playground
  (width_garden : ℝ := 24) -- width of the garden
  (width_playground : ℝ := 12) -- width of the playground
  (perimeter_garden : ℝ := 64) -- perimeter of the garden
  (area_garden : ℝ := L_g * 24) -- area of the garden
  (area_playground : ℝ := L_p * 12) -- area of the playground
  (areas_equal : area_garden = area_playground) -- equal areas
  (perimeter_condition : 2 * (L_g + 24) = 64) -- perimeter condition
  : L_p = 16 := 
by
  sorry

end playground_length_l404_40426


namespace num_real_solutions_system_l404_40439

theorem num_real_solutions_system :
  ∃! (num_solutions : ℕ), 
  num_solutions = 5 ∧
  ∃ x y z w : ℝ, 
    (x = z + w + x * z) ∧ 
    (y = w + x + y * w) ∧ 
    (z = x + y + z * x) ∧ 
    (w = y + z + w * z) :=
sorry

end num_real_solutions_system_l404_40439


namespace poodle_barks_count_l404_40460

-- Define the conditions as hypothesis
variables (poodle_barks terrier_barks terrier_hushes : ℕ)

-- Define the conditions
def condition1 : Prop :=
  poodle_barks = 2 * terrier_barks

def condition2 : Prop :=
  terrier_hushes = terrier_barks / 2

def condition3 : Prop :=
  terrier_hushes = 6

-- The theorem we need to prove
theorem poodle_barks_count (poodle_barks terrier_barks terrier_hushes : ℕ)
  (h1 : condition1 poodle_barks terrier_barks)
  (h2 : condition2 terrier_barks terrier_hushes)
  (h3 : condition3 terrier_hushes) :
  poodle_barks = 24 :=
by
  -- Proof is not required as per instructions
  sorry

end poodle_barks_count_l404_40460


namespace Suresh_completes_job_in_15_hours_l404_40437

theorem Suresh_completes_job_in_15_hours :
  ∃ S : ℝ,
    (∀ (T_A Ashutosh_time Suresh_time : ℝ), Ashutosh_time = 15 ∧ Suresh_time = 9 
    → T_A = Ashutosh_time → 6 / T_A + Suresh_time / S = 1) ∧ S = 15 :=
by
  sorry

end Suresh_completes_job_in_15_hours_l404_40437


namespace trisha_total_distance_l404_40408

theorem trisha_total_distance :
  let distance1 := 0.11
  let distance2 := 0.11
  let distance3 := 0.67
  distance1 + distance2 + distance3 = 0.89 :=
by
  sorry

end trisha_total_distance_l404_40408


namespace euler_totient_divisibility_l404_40479

def euler_totient (n : ℕ) : ℕ := sorry

theorem euler_totient_divisibility (n : ℕ) (hn : 0 < n) : 2^(n * (n + 1)) ∣ 32 * euler_totient (2^(2^n) - 1) := 
sorry

end euler_totient_divisibility_l404_40479


namespace men_in_first_group_l404_40482

theorem men_in_first_group
  (M : ℕ) -- number of men in the first group
  (h1 : M * 8 * 24 = 12 * 8 * 16) : M = 8 :=
sorry

end men_in_first_group_l404_40482


namespace abs_eq_ax_plus_1_one_negative_root_no_positive_roots_l404_40404

theorem abs_eq_ax_plus_1_one_negative_root_no_positive_roots (a : ℝ) :
  (∃ x : ℝ, |x| = a * x + 1 ∧ x < 0) ∧ (∀ x : ℝ, |x| = a * x + 1 → x ≤ 0) → a > -1 :=
by
  sorry

end abs_eq_ax_plus_1_one_negative_root_no_positive_roots_l404_40404


namespace base_7_sum_of_product_l404_40401

-- Definitions of the numbers in base-10 for base-7 numbers
def base_7_to_base_10 (d1 d0 : ℕ) : ℕ := d1 * 7 + d0

def sum_digits_base_7 (n : ℕ) : ℕ := 
  let d2 := n / 343
  let r2 := n % 343
  let d1 := r2 / 49
  let r1 := r2 % 49
  let d0 := r1 / 7 + r1 % 7
  d2 + d1 + d0

def convert_10_to_7 (n : ℕ) : ℕ := 
  let d1 := n / 7
  let r1 := n % 7
  d1 * 10 + r1

theorem base_7_sum_of_product : 
  let n36  := base_7_to_base_10 3 6
  let n52  := base_7_to_base_10 5 2
  let nadd := base_7_to_base_10 2 0
  let prod := n36 * n52
  let suma := prod + nadd
  convert_10_to_7 (sum_digits_base_7 suma) = 23 :=
by
  sorry

end base_7_sum_of_product_l404_40401


namespace distance_between_locations_A_and_B_l404_40430

theorem distance_between_locations_A_and_B 
  (speed_A speed_B speed_C : ℝ)
  (distance_CD : ℝ)
  (distance_initial_A : ℝ)
  (distance_A_to_B : ℝ)
  (h1 : speed_A = 3 * speed_C)
  (h2 : speed_A = 1.5 * speed_B)
  (h3 : distance_CD = 12)
  (h4 : distance_initial_A = 50)
  (h5 : distance_A_to_B = 130)
  : distance_A_to_B = 130 :=
by
  sorry

end distance_between_locations_A_and_B_l404_40430


namespace elevator_max_weight_capacity_l404_40461

theorem elevator_max_weight_capacity 
  (num_adults : ℕ)
  (weight_adult : ℕ)
  (num_children : ℕ)
  (weight_child : ℕ)
  (max_next_person_weight : ℕ) 
  (H_adults : num_adults = 3)
  (H_weight_adult : weight_adult = 140)
  (H_children : num_children = 2)
  (H_weight_child : weight_child = 64)
  (H_max_next : max_next_person_weight = 52) : 
  num_adults * weight_adult + num_children * weight_child + max_next_person_weight = 600 := 
by
  sorry

end elevator_max_weight_capacity_l404_40461


namespace log_fraction_identity_l404_40481

theorem log_fraction_identity (a b : ℝ) (h2 : Real.log 2 = a) (h3 : Real.log 3 = b) :
  (Real.log 12 / Real.log 15) = (2 * a + b) / (1 - a + b) := 
  sorry

end log_fraction_identity_l404_40481


namespace line_through_intersection_parallel_to_y_axis_l404_40491

theorem line_through_intersection_parallel_to_y_axis:
  ∃ x, (∃ y, 3 * x + 2 * y - 5 = 0 ∧ x - 3 * y + 2 = 0) ∧
       (x = 1) :=
sorry

end line_through_intersection_parallel_to_y_axis_l404_40491


namespace original_proposition_converse_proposition_false_inverse_proposition_false_contrapositive_proposition_true_l404_40467

-- Definition of the quadrilateral being a rhombus
def is_rhombus (quad : Type) : Prop := 
-- A quadrilateral is a rhombus if and only if all its sides are equal in length
sorry

-- Definition of the diagonals of quadrilateral being perpendicular
def diagonals_are_perpendicular (quad : Type) : Prop := 
-- The diagonals of a quadrilateral are perpendicular
sorry

-- Original proposition: If a quadrilateral is a rhombus, then its diagonals are perpendicular to each other
theorem original_proposition (quad : Type) : is_rhombus quad → diagonals_are_perpendicular quad :=
sorry

-- Converse proposition: If the diagonals of a quadrilateral are perpendicular to each other, then it is a rhombus, which is False
theorem converse_proposition_false (quad : Type) : diagonals_are_perpendicular quad → ¬ is_rhombus quad :=
sorry

-- Inverse proposition: If a quadrilateral is not a rhombus, then its diagonals are not perpendicular, which is False
theorem inverse_proposition_false (quad : Type) : ¬ is_rhombus quad → ¬ diagonals_are_perpendicular quad :=
sorry

-- Contrapositive proposition: If the diagonals of a quadrilateral are not perpendicular, then it is not a rhombus, which is True
theorem contrapositive_proposition_true (quad : Type) : ¬ diagonals_are_perpendicular quad → ¬ is_rhombus quad :=
sorry

end original_proposition_converse_proposition_false_inverse_proposition_false_contrapositive_proposition_true_l404_40467


namespace cosine_sine_difference_identity_l404_40449

theorem cosine_sine_difference_identity :
  (Real.cos (75 * Real.pi / 180) * Real.cos (15 * Real.pi / 180)
  - Real.sin (255 * Real.pi / 180) * Real.sin (165 * Real.pi / 180)) = 1 / 2 := by
  -- Proof goes here
  sorry

end cosine_sine_difference_identity_l404_40449


namespace max_cube_side_length_max_rect_parallelepiped_dimensions_l404_40444

-- Part (a)
theorem max_cube_side_length (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ∃ s : ℝ, s = a * b * c / (a * b + b * c + a * c) :=
sorry

-- Part (b)
theorem max_rect_parallelepiped_dimensions (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ∃ x y z : ℝ, x = a / 3 ∧ y = b / 3 ∧ z = c / 3 :=
sorry

end max_cube_side_length_max_rect_parallelepiped_dimensions_l404_40444


namespace cube_root_of_27_l404_40419

theorem cube_root_of_27 : ∃ x : ℝ, x ^ 3 = 27 ↔ ∃ y : ℝ, y = 3 := by
  sorry

end cube_root_of_27_l404_40419


namespace energy_soda_packs_l404_40416

-- Definitions and conditions
variables (total_bottles : ℕ) (regular_soda : ℕ) (diet_soda : ℕ) (pack_size : ℕ)
variables (complete_packs : ℕ) (remaining_regular : ℕ) (remaining_diet : ℕ) (remaining_energy : ℕ)

-- Conditions given in the problem
axiom h_total_bottles : total_bottles = 200
axiom h_regular_soda : regular_soda = 55
axiom h_diet_soda : diet_soda = 40
axiom h_pack_size : pack_size = 3

-- Proving the correct answer
theorem energy_soda_packs :
  complete_packs = (total_bottles - (regular_soda + diet_soda)) / pack_size ∧
  remaining_regular = regular_soda ∧
  remaining_diet = diet_soda ∧
  remaining_energy = (total_bottles - (regular_soda + diet_soda)) % pack_size :=
by
  sorry

end energy_soda_packs_l404_40416


namespace abs_diff_expr_l404_40436

theorem abs_diff_expr :
  let a := -3 * (7 - 15)
  let b := (5 - 7)^2 + (-4)^2
  |a| - |b| = 4 :=
by
  let a := -3 * (7 - 15)
  let b := (5 - 7)^2 + (-4)^2
  sorry

end abs_diff_expr_l404_40436


namespace find_x_l404_40402

theorem find_x : 2^4 + 3 = 5^2 - 6 :=
by
  sorry

end find_x_l404_40402


namespace remainder_mod_5_is_0_l404_40410

theorem remainder_mod_5_is_0 :
  (88144 * 88145 + 88146 + 88147 + 88148 + 88149 + 88150) % 5 = 0 := by
  sorry

end remainder_mod_5_is_0_l404_40410


namespace transformation_correct_l404_40407

-- Define the original function
noncomputable def original_function (x : ℝ) : ℝ := Real.sin x

-- Define the transformation functions
noncomputable def shift_right_by_pi_over_10 (f : ℝ → ℝ) (x : ℝ) : ℝ := f (x - Real.pi / 10)
noncomputable def stretch_x_by_factor_of_2 (f : ℝ → ℝ) (x : ℝ) : ℝ := f (x / 2)

-- Define the transformed function
noncomputable def transformed_function : ℝ → ℝ :=
  stretch_x_by_factor_of_2 (shift_right_by_pi_over_10 original_function)

-- Define the expected resulting function
noncomputable def expected_function (x : ℝ) : ℝ := Real.sin (x / 2 - Real.pi / 10)

-- State the theorem
theorem transformation_correct :
  ∀ x : ℝ, transformed_function x = expected_function x :=
by
  sorry

end transformation_correct_l404_40407


namespace cheryl_distance_walked_l404_40468

theorem cheryl_distance_walked :
  let s1 := 2  -- speed during the first segment in miles per hour
  let t1 := 3  -- time during the first segment in hours
  let s2 := 4  -- speed during the second segment in miles per hour
  let t2 := 2  -- time during the second segment in hours
  let s3 := 1  -- speed during the third segment in miles per hour
  let t3 := 3  -- time during the third segment in hours
  let s4 := 3  -- speed during the fourth segment in miles per hour
  let t4 := 5  -- time during the fourth segment in hours
  let d1 := s1 * t1  -- distance for the first segment
  let d2 := s2 * t2  -- distance for the second segment
  let d3 := s3 * t3  -- distance for the third segment
  let d4 := s4 * t4  -- distance for the fourth segment
  d1 + d2 + d3 + d4 = 32 :=
by
  sorry

end cheryl_distance_walked_l404_40468


namespace sum_of_ages_of_henrys_brothers_l404_40431

theorem sum_of_ages_of_henrys_brothers (a b c : ℕ) : 
  a = 2 * b → 
  b = c ^ 2 →
  a ≠ b ∧ a ≠ c ∧ b ≠ c →
  a < 10 ∧ b < 10 ∧ c < 10 →
  a + b + c = 14 :=
by
  intro h₁ h₂ h₃ h₄
  sorry

end sum_of_ages_of_henrys_brothers_l404_40431


namespace horse_distance_traveled_l404_40477

theorem horse_distance_traveled :
  let r2 := 12
  let n2 := 120
  let D2 := n2 * 2 * Real.pi * r2
  D2 = 2880 * Real.pi :=
by
  sorry

end horse_distance_traveled_l404_40477


namespace range_of_theta_l404_40497

theorem range_of_theta (θ : ℝ) (hθ : 0 ≤ θ ∧ θ < 2 * Real.pi)
    (h_ineq : 3 * (Real.sin θ ^ 5 + Real.cos (2 * θ) ^ 5) > 5 * (Real.sin θ ^ 3 + Real.cos (2 * θ) ^ 3)) :
    θ ∈ Set.Ico (7 * Real.pi / 6) (11 * Real.pi / 6) :=
sorry

end range_of_theta_l404_40497


namespace solution_for_factorial_equation_l404_40424

theorem solution_for_factorial_equation:
  { (n, k) : ℕ × ℕ | 0 < n ∧ 0 < k ∧ n! + n = n^k } = {(2,2), (3,2), (5,3)} :=
by
  sorry

end solution_for_factorial_equation_l404_40424


namespace find_x_l404_40485

-- Definition of logarithm in Lean
noncomputable def log (b a: ℝ) : ℝ := Real.log a / Real.log b

-- Problem statement in Lean
theorem find_x (x : ℝ) (h : log 64 4 = 1 / 3) : log x 8 = 1 / 3 → x = 512 :=
by sorry

end find_x_l404_40485


namespace locus_centers_of_tangent_circles_l404_40466

theorem locus_centers_of_tangent_circles (a b : ℝ) :
  (x^2 + y^2 = 1) ∧ ((x - 1)^2 + (y -1)^2 = 81) →
  (a^2 + b^2 - (2 * a * b) / 63 - (66 * a) / 63 - (66 * b) / 63 + 17 = 0) :=
by
  sorry

end locus_centers_of_tangent_circles_l404_40466


namespace multiplication_problem_l404_40450

theorem multiplication_problem :
  250 * 24.98 * 2.498 * 1250 = 19484012.5 := by
  sorry

end multiplication_problem_l404_40450


namespace find_number_l404_40457

theorem find_number (x : ℝ) (h : 0.95 * x - 12 = 178) : x = 200 :=
sorry

end find_number_l404_40457


namespace min_value_l404_40488

-- Defining the conditions
variables {x y z : ℝ}

-- Problem statement translating the conditions
theorem min_value (h₁ : x > 0) (h₂ : y > 0) (h₃ : z > 0) (h₄ : x + y + z = 5) : 
  ∃ (minval : ℝ), minval = 36/5 ∧ ∀ w, w = (1/x + 4/y + 9/z) → w ≥ minval :=
by
  sorry

end min_value_l404_40488


namespace find_peaches_l404_40498

theorem find_peaches (A P : ℕ) (h1 : A + P = 15) (h2 : 1000 * A + 2000 * P = 22000) : P = 7 := sorry

end find_peaches_l404_40498


namespace number_of_elements_in_M_l404_40403

def positive_nats : Set ℕ := {n | n > 0}
def M : Set ℕ := {m | ∃ n ∈ positive_nats, m = 2 * n - 1 ∧ m < 60}

theorem number_of_elements_in_M : ∃ s : Finset ℕ, (∀ x, x ∈ s ↔ x ∈ M) ∧ s.card = 30 := 
by
  sorry

end number_of_elements_in_M_l404_40403


namespace evaluate_expression_l404_40423

theorem evaluate_expression :
  71 * Real.sqrt (3 + 2 * Real.sqrt 2) - Real.sqrt (3 - 2 * Real.sqrt 2) = 72 + 70 * Real.sqrt 2 :=
by
  sorry

end evaluate_expression_l404_40423


namespace solution_to_problem_l404_40405

theorem solution_to_problem (x y : ℕ) : 
  (x.gcd y + x.lcm y = x + y) ↔ 
  ∃ (d k : ℕ), (x = d ∧ y = d * k) ∨ (x = d * k ∧ y = d) :=
by sorry

end solution_to_problem_l404_40405


namespace find_x_l404_40447

theorem find_x (x : ℝ) (h : (3 * x - 7) / 4 = 14) : x = 21 :=
sorry

end find_x_l404_40447


namespace avg_mpg_sum_l404_40458

def first_car_gallons : ℕ := 25
def second_car_gallons : ℕ := 35
def total_miles : ℕ := 2275
def first_car_mpg : ℕ := 40

noncomputable def sum_of_avg_mpg_of_two_cars : ℝ := 76.43

theorem avg_mpg_sum :
  let first_car_miles := (first_car_gallons * first_car_mpg : ℕ)
  let second_car_miles := total_miles - first_car_miles
  let second_car_mpg := (second_car_miles : ℝ) / second_car_gallons
  let sum_avg_mpg := (first_car_mpg : ℝ) + second_car_mpg
  sum_avg_mpg = sum_of_avg_mpg_of_two_cars :=
by
  sorry

end avg_mpg_sum_l404_40458


namespace max_f_angle_A_of_triangle_l404_40456

noncomputable def f (x : ℝ) : ℝ := (Real.cos (2 * x - 4 * Real.pi / 3)) + 2 * (Real.cos x)^2

theorem max_f : ∃ x : ℝ, f x = 2 := sorry

theorem angle_A_of_triangle (A B C : ℝ) (h : A + B + C = Real.pi)
  (h2 : f (B + C) = 3 / 2) : A = Real.pi / 3 := sorry

end max_f_angle_A_of_triangle_l404_40456


namespace volume_of_cuboid_l404_40453

-- Define the edges of the cuboid
def edge1 : ℕ := 6
def edge2 : ℕ := 5
def edge3 : ℕ := 6

-- Define the volume formula for a cuboid
def volume (a b c : ℕ) : ℕ := a * b * c

-- State the theorem
theorem volume_of_cuboid : volume edge1 edge2 edge3 = 180 := by
  sorry

end volume_of_cuboid_l404_40453


namespace nathan_tomato_plants_l404_40483

theorem nathan_tomato_plants (T: ℕ) : 
  5 * 14 + T * 16 = 186 * 7 / 6 + 9 * 10 :=
  sorry

end nathan_tomato_plants_l404_40483


namespace total_area_of_union_of_six_triangles_l404_40409

theorem total_area_of_union_of_six_triangles :
  let s := 2 * Real.sqrt 2
  let area_one_triangle := (Real.sqrt 3 / 4) * s^2
  let total_area_without_overlaps := 6 * area_one_triangle
  let side_overlap := Real.sqrt 2
  let area_one_overlap := (Real.sqrt 3 / 4) * side_overlap ^ 2
  let total_overlap_area := 5 * area_one_overlap
  let net_area := total_area_without_overlaps - total_overlap_area
  net_area = 9.5 * Real.sqrt 3 := 
by
  sorry

end total_area_of_union_of_six_triangles_l404_40409


namespace symmetric_circle_equation_l404_40492

noncomputable def equation_of_symmetric_circle (C₁ : ℝ → ℝ → Prop) (l : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, C₁ x y ↔ x^2 + y^2 - 4 * x - 8 * y + 19 = 0

theorem symmetric_circle_equation :
  ∀ (C₁ : ℝ → ℝ → Prop) (l : ℝ → ℝ → Prop),
  equation_of_symmetric_circle C₁ l →
  (∀ x y, l x y ↔ x + 2 * y - 5 = 0) →
  ∃ C₂ : ℝ → ℝ → Prop, (∀ x y, C₂ x y ↔ x^2 + y^2 = 1) :=
by
  intros C₁ l hC₁ hₗ
  sorry

end symmetric_circle_equation_l404_40492


namespace r_at_5_l404_40486

def r (x : ℝ) : ℝ := (x - 1) * (x - 2) * (x - 3) * (x - 4) + x^2 - 1

theorem r_at_5 :
  r 5 = 48 := by
  sorry

end r_at_5_l404_40486


namespace min_absolute_difference_l404_40476

open Int

theorem min_absolute_difference (x y : ℤ) (hx : 0 < x) (hy : 0 < y) (h : x * y - 4 * x + 3 * y = 215) : |x - y| = 15 :=
sorry

end min_absolute_difference_l404_40476


namespace necessary_and_sufficient_condition_l404_40428

theorem necessary_and_sufficient_condition (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (a^2 + b^2 ≥ 2 * a * b) ↔ (a/b + b/a ≥ 2) :=
sorry

end necessary_and_sufficient_condition_l404_40428


namespace range_of_a_l404_40429

-- Function definition for op
def op (x y : ℝ) : ℝ := x * (2 - y)

-- Predicate that checks the inequality for all t
def inequality_holds_for_all_t (a : ℝ) : Prop :=
  ∀ t : ℝ, (op (t - a) (t + a)) < 1

-- Prove that the range of a is (0, 2)
theorem range_of_a : 
  ∀ a : ℝ, inequality_holds_for_all_t a ↔ 0 < a ∧ a < 2 := 
by
  sorry

end range_of_a_l404_40429


namespace cube_of_square_of_third_smallest_prime_l404_40499

-- Define the third smallest prime number
def third_smallest_prime : ℕ := 5

-- Theorem to prove the cube of the square of the third smallest prime number
theorem cube_of_square_of_third_smallest_prime :
  (third_smallest_prime^2)^3 = 15625 := by
  sorry

end cube_of_square_of_third_smallest_prime_l404_40499


namespace find_f1_l404_40435

def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f (x)

def functional_equation (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x + 2) = x * f (x)

theorem find_f1 (f : ℝ → ℝ) 
  (h1 : even_function f) 
  (h2 : functional_equation f) : 
  f 1 = 0 :=
sorry

end find_f1_l404_40435


namespace defective_percentage_is_0_05_l404_40490

-- Define the problem conditions as Lean definitions
def total_meters : ℕ := 4000
def defective_meters : ℕ := 2

-- Define the percentage calculation function
def percentage_defective (defective total : ℕ) : ℚ :=
  (defective : ℚ) / (total : ℚ) * 100

-- Rewrite the proof statement using these definitions
theorem defective_percentage_is_0_05 :
  percentage_defective defective_meters total_meters = 0.05 :=
by
  sorry

end defective_percentage_is_0_05_l404_40490


namespace positive_integer_solutions_condition_l404_40445

theorem positive_integer_solutions_condition (a : ℕ) (A B : ℝ) :
  (∃ (x y z : ℕ), x^2 + y^2 + z^2 = (13 * a)^2 ∧
  x^2 * (A * x^2 + B * y^2) + y^2 * (A * y^2 + B * z^2) + z^2 * (A * z^2 + B * x^2) = (1/4) * (2 * A + B) * (13 * a)^4)
  ↔ A = (1 / 2) * B := 
sorry

end positive_integer_solutions_condition_l404_40445


namespace ab_c_work_days_l404_40442

noncomputable def W_ab : ℝ := 1 / 15
noncomputable def W_c : ℝ := 1 / 30
noncomputable def W_abc : ℝ := W_ab + W_c

theorem ab_c_work_days :
  (1 / W_abc) = 10 :=
by
  sorry

end ab_c_work_days_l404_40442


namespace value_of_expr_l404_40446

theorem value_of_expr : (365^2 - 349^2) / 16 = 714 := by
  sorry

end value_of_expr_l404_40446


namespace ratio_of_puzzle_times_l404_40495

def total_time := 70
def warmup_time := 10
def remaining_puzzles := 60 / 2

theorem ratio_of_puzzle_times : (remaining_puzzles / warmup_time) = 3 := by
  -- Given Conditions
  have H1 : 70 = 10 + 2 * (60 / 2) := by sorry
  -- Simplification and Calculation
  have H2 : (remaining_puzzles = 30) := by sorry
  -- Ratio Calculation
  have ratio_calculation: (30 / 10) = 3 := by sorry
  exact ratio_calculation

end ratio_of_puzzle_times_l404_40495


namespace correct_equations_l404_40411

-- Defining the problem statement
theorem correct_equations (m n : ℕ) :
  (∀ (m n : ℕ), 40 * m + 10 = 43 * m + 1 ∧ 
   (n - 10) / 40 = (n - 1) / 43) :=
by
  sorry

end correct_equations_l404_40411


namespace trajectory_range_k_l404_40474

-- Condition Definitions
def A : ℝ × ℝ := (-1, 0)
def B : ℝ × ℝ := (1, 0)
def N (x : ℝ) : ℝ × ℝ := (x, 0)
def vector_MN (x y : ℝ) : ℝ × ℝ := (0, -y)
def vector_AN (x : ℝ) : ℝ × ℝ := (x + 1, 0)
def vector_BN (x : ℝ) : ℝ × ℝ := (x - 1, 0)
def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

-- Problem 1: Prove the trajectory equation
theorem trajectory (x y : ℝ) (h : (vector_MN x y).1^2 + (vector_MN x y).2^2 = dot_product (vector_AN x) (vector_BN x)) :
  x^2 - y^2 = 1 :=
sorry

-- Problem 2: Prove the range of k
theorem range_k (k : ℝ) :
  (∃ x y : ℝ, y = k * x - 1 ∧ x^2 - y^2 = 1) ↔ -Real.sqrt 2 ≤ k ∧ k ≤ Real.sqrt 2 :=
sorry

end trajectory_range_k_l404_40474


namespace circular_seat_coloring_l404_40472

def count_colorings (n : ℕ) : ℕ :=
  sorry

theorem circular_seat_coloring :
  count_colorings 6 = 66 :=
by
  sorry

end circular_seat_coloring_l404_40472


namespace percentage_of_a_l404_40469

theorem percentage_of_a (a : ℕ) (x : ℕ) (h1 : a = 190) (h2 : (x * a) / 100 = 95) : x = 50 := by
  sorry

end percentage_of_a_l404_40469


namespace remainder_2365947_div_8_l404_40465

theorem remainder_2365947_div_8 : (2365947 % 8) = 3 :=
by
  sorry

end remainder_2365947_div_8_l404_40465


namespace Barry_reach_l404_40420

noncomputable def Larry_full_height : ℝ := 5
noncomputable def Larry_shoulder_height : ℝ := Larry_full_height - 0.2 * Larry_full_height
noncomputable def combined_reach : ℝ := 9

theorem Barry_reach :
  combined_reach - Larry_shoulder_height = 5 := 
by
  -- Correct answer verification comparing combined reach minus Larry's shoulder height equals 5
  sorry

end Barry_reach_l404_40420


namespace contrapositive_example_l404_40443

theorem contrapositive_example (x : ℝ) :
  (¬ (x = 3 ∧ x = 4)) → (x^2 - 7 * x + 12 ≠ 0) →
  (x^2 - 7 * x + 12 = 0) → (x = 3 ∨ x = 4) :=
by
  intros h h1 h2
  sorry  -- proof is not required

end contrapositive_example_l404_40443


namespace repeating_decimal_fraction_l404_40433

noncomputable def x : ℚ := 75 / 99  -- 0.\overline{75}
noncomputable def y : ℚ := 223 / 99  -- 2.\overline{25}

theorem repeating_decimal_fraction : (x / y) = 2475 / 7329 :=
by
  -- Further proof details can be added here
  sorry

end repeating_decimal_fraction_l404_40433


namespace geometric_sequence_first_term_l404_40478

theorem geometric_sequence_first_term 
  (T : ℕ → ℝ) 
  (h1 : T 5 = 243) 
  (h2 : T 6 = 729) 
  (hr : ∃ r : ℝ, ∀ n : ℕ, T n = T 1 * r^(n - 1)) :
  T 1 = 3 :=
by
  sorry

end geometric_sequence_first_term_l404_40478


namespace angel_vowels_written_l404_40434

theorem angel_vowels_written (num_vowels : ℕ) (times_written : ℕ) (h1 : num_vowels = 5) (h2 : times_written = 4) : num_vowels * times_written = 20 := by
  sorry

end angel_vowels_written_l404_40434


namespace percentage_increase_soda_price_l404_40452

theorem percentage_increase_soda_price
  (C_new : ℝ) (S_new : ℝ) (C_increase : ℝ) (C_total_before : ℝ)
  (h1 : C_new = 20)
  (h2: S_new = 6)
  (h3: C_increase = 0.25)
  (h4: C_new * (1 - C_increase) + S_new * (1 + (S_new / (S_new * (1 + (S_new / (S_new * 0.5)))))) = C_total_before) : 
  (S_new - S_new * (1 - C_increase) * 100 / (S_new * (1 + 0.5)) * C_total_before) = 50 := 
by 
  -- This is where the proof would go.
  sorry

end percentage_increase_soda_price_l404_40452


namespace positive_difference_is_zero_l404_40473

-- Definitions based on conditions
def jo_sum (n : ℕ) : ℕ := (n * (n + 1)) / 2

def rounded_to_nearest_5 (x : ℕ) : ℕ :=
  if x % 5 = 0 then x
  else (x / 5) * 5 + (if x % 5 >= 3 then 5 else 0)

def alan_sum (n : ℕ) : ℕ :=
  (List.range (n + 1)).map rounded_to_nearest_5 |>.sum

-- Theorem based on question and correct answer
theorem positive_difference_is_zero :
  jo_sum 120 - alan_sum 120 = 0 := sorry

end positive_difference_is_zero_l404_40473


namespace value_of_some_number_l404_40463

theorem value_of_some_number (a : ℤ) (h : a = 105) :
  (a ^ 3 = 3 * (5 ^ 3) * (3 ^ 2) * (7 ^ 2)) :=
by {
  sorry
}

end value_of_some_number_l404_40463


namespace chapters_ratio_l404_40471

theorem chapters_ratio
  (c1 : ℕ) (c2 : ℕ) (total : ℕ) (x : ℕ)
  (h1 : c1 = 20)
  (h2 : c2 = 15)
  (h3 : total = 75)
  (h4 : x = (c1 + 2 * c2) / 2)
  (h5 : c1 + 2 * c2 + x = total) :
  (x : ℚ) / (c1 + 2 * c2 : ℚ) = 1 / 2 :=
by
  sorry

end chapters_ratio_l404_40471


namespace wall_area_160_l404_40496

noncomputable def wall_area (small_tile_area : ℝ) (fraction_small : ℝ) : ℝ :=
  small_tile_area / fraction_small

theorem wall_area_160 (small_tile_area : ℝ) (fraction_small : ℝ) (h1 : small_tile_area = 80) (h2 : fraction_small = 1 / 2) :
  wall_area small_tile_area fraction_small = 160 :=
by
  rw [wall_area, h1, h2]
  norm_num

end wall_area_160_l404_40496


namespace fermats_little_theorem_l404_40475

theorem fermats_little_theorem (n p : ℕ) [hp : Fact p.Prime] : p ∣ (n^p - n) :=
sorry

end fermats_little_theorem_l404_40475


namespace domain_of_f_l404_40412

def domain_f (x : ℝ) : Prop :=
  2 * x - 3 ≥ 0 ∧ x ≠ 3

def domain_set : Set ℝ :=
  { x | (3 / 2) ≤ x ∧ x < 3 ∨ 3 < x }

theorem domain_of_f :
  { x : ℝ | domain_f x } = domain_set := by
  sorry

end domain_of_f_l404_40412


namespace sum_of_edges_not_always_smaller_at_least_one_edge_shorter_l404_40440

-- Definitions for vertices of pyramids
variables (A B C D E : ℝ)

-- Assuming E is inside pyramid ABCD
variable (inside : E ∈ convex_hull ℝ {A, B, C, D})

-- Assertion 1
theorem sum_of_edges_not_always_smaller
  (h1 : A ≠ B) (h2 : B ≠ C) (h3 : C ≠ A) (h4 : A ≠ D) (h5 : B ≠ D) (h6 : C ≠ D) (h7 : D ≠ E):
  ¬ (abs A - E + abs B - E + abs C - E < abs A - D + abs B - D + abs C - D) :=
sorry

-- Assertion 2
theorem at_least_one_edge_shorter
  (h1 : A ≠ B) (h2 : B ≠ C) (h3 : C ≠ A)
  (h4 : A ≠ D) (h5 : B ≠ D) (h6 : C ≠ D)
  (h7 : D ≠ E):
  abs A - E < abs A - D ∨ abs B - E < abs B - D ∨ abs C - E < abs C - D :=
sorry

end sum_of_edges_not_always_smaller_at_least_one_edge_shorter_l404_40440
