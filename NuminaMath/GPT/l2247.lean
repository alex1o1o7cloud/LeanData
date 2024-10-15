import Mathlib

namespace NUMINAMATH_GPT_function_decreasing_iff_l2247_224750

theorem function_decreasing_iff (a : ℝ) :
  (0 < a ∧ a < 1) ∧ a ≤ 1/4 ↔ (0 < a ∧ a ≤ 1/4) :=
by
  sorry

end NUMINAMATH_GPT_function_decreasing_iff_l2247_224750


namespace NUMINAMATH_GPT_length_of_chord_l2247_224738

-- Define the parabola y^2 = 4x
def parabola (x y : ℝ) : Prop :=
  y^2 = 4 * x

-- Define the line y = x - 1 with slope 1 passing through the focus (1, 0)
def line (x y : ℝ) : Prop :=
  y = x - 1

-- Prove that the length of the chord |AB| is 8
theorem length_of_chord 
  (x1 y1 x2 y2 : ℝ) 
  (h1 : parabola x1 y1) 
  (h2 : parabola x2 y2) 
  (h3 : line x1 y1) 
  (h4 : line x2 y2) : 
  abs (x2 - x1) = 8 :=
sorry

end NUMINAMATH_GPT_length_of_chord_l2247_224738


namespace NUMINAMATH_GPT_volume_removed_percentage_l2247_224756

noncomputable def volume_of_box (length width height : ℝ) : ℝ := 
  length * width * height

noncomputable def volume_of_cube (side : ℝ) : ℝ := 
  side ^ 3

noncomputable def volume_removed (length width height side : ℝ) : ℝ :=
  8 * (volume_of_cube side)

noncomputable def percentage_removed (length width height side : ℝ) : ℝ :=
  (volume_removed length width height side) / (volume_of_box length width height) * 100

theorem volume_removed_percentage :
  percentage_removed 20 15 12 4 = 14.22 := 
by
  sorry

end NUMINAMATH_GPT_volume_removed_percentage_l2247_224756


namespace NUMINAMATH_GPT_fraction_power_zero_l2247_224723

variable (a b : ℤ)
variable (h_a : a ≠ 0) (h_b : b ≠ 0)

theorem fraction_power_zero : (a / b)^0 = 1 := by
  sorry

end NUMINAMATH_GPT_fraction_power_zero_l2247_224723


namespace NUMINAMATH_GPT_radius_of_cylinder_is_correct_l2247_224794

/-- 
  A right circular cylinder is inscribed in a right circular cone such that:
  - The diameter of the cylinder is equal to its height.
  - The cone has a diameter of 8.
  - The cone has an altitude of 10.
  - The axes of the cylinder and cone coincide.
  Prove that the radius of the cylinder is 20/9.
-/
theorem radius_of_cylinder_is_correct :
  ∀ (r : ℚ), 
    (2 * r = 8 - 2 * r ∧ 10 - 2 * r = (10 / 4) * r) → 
    r = 20 / 9 :=
by
  intro r
  intro h
  sorry

end NUMINAMATH_GPT_radius_of_cylinder_is_correct_l2247_224794


namespace NUMINAMATH_GPT_smallest_two_digit_number_l2247_224737

theorem smallest_two_digit_number :
  ∃ (n : ℕ), 10 ≤ n ∧ n < 100 ∧
            n % 12 = 0 ∧
            n % 5 = 4 ∧
            ∀ m : ℕ, 10 ≤ m ∧ m < 100 ∧ m % 12 = 0 ∧ m % 5 = 4 → n ≤ m :=
  by {
  -- proof shows the mathematical statement is true
  sorry
}

end NUMINAMATH_GPT_smallest_two_digit_number_l2247_224737


namespace NUMINAMATH_GPT_fraction_to_decimal_l2247_224701

theorem fraction_to_decimal :
  (7 / 125 : ℚ) = 0.056 :=
sorry

end NUMINAMATH_GPT_fraction_to_decimal_l2247_224701


namespace NUMINAMATH_GPT_Tom_runs_60_miles_in_a_week_l2247_224763

variable (days_per_week : ℕ) (hours_per_day : ℝ) (speed : ℝ)
variable (h_days_per_week : days_per_week = 5)
variable (h_hours_per_day : hours_per_day = 1.5)
variable (h_speed : speed = 8)

theorem Tom_runs_60_miles_in_a_week : (days_per_week * hours_per_day * speed) = 60 := by
  sorry

end NUMINAMATH_GPT_Tom_runs_60_miles_in_a_week_l2247_224763


namespace NUMINAMATH_GPT_proof_statement_l2247_224766

-- Define the initial dimensions and areas
def initial_length : ℕ := 7
def initial_width : ℕ := 5

-- Shortened dimensions by one side and the corresponding area condition
def shortened_new_width : ℕ := 3
def shortened_area : ℕ := 21

-- Define the task
def task_statement : Prop :=
  (initial_length - 2) * initial_width = shortened_area ∧
  (initial_width - 2) * initial_length = shortened_area →
  (initial_length - 2) * (initial_width - 2) = 25

theorem proof_statement : task_statement :=
by {
  sorry -- Proof goes here
}

end NUMINAMATH_GPT_proof_statement_l2247_224766


namespace NUMINAMATH_GPT_abc_is_cube_l2247_224734

theorem abc_is_cube (a b c : ℤ) (h : (a:ℚ) / (b:ℚ) + (b:ℚ) / (c:ℚ) + (c:ℚ) / (a:ℚ) = 3) : ∃ x : ℤ, abc = x^3 :=
by
  sorry

end NUMINAMATH_GPT_abc_is_cube_l2247_224734


namespace NUMINAMATH_GPT_sum_of_reciprocals_is_two_l2247_224718

variable (x y : ℝ)
variable (h1 : x + y = 50)
variable (h2 : x * y = 25)

theorem sum_of_reciprocals_is_two (hx : x ≠ 0) (hy : y ≠ 0) : 
  (1/x + 1/y) = 2 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_reciprocals_is_two_l2247_224718


namespace NUMINAMATH_GPT_area_percentage_change_l2247_224720

variable (a b : ℝ)

def initial_area : ℝ := a * b

def new_length (a : ℝ) : ℝ := a * 1.35

def new_width (b : ℝ) : ℝ := b * 0.86

def new_area (a b : ℝ) : ℝ := (new_length a) * (new_width b)

theorem area_percentage_change :
    ((new_area a b) / (initial_area a b)) = 1.161 :=
by
  sorry

end NUMINAMATH_GPT_area_percentage_change_l2247_224720


namespace NUMINAMATH_GPT_ones_digit_expression_l2247_224711

theorem ones_digit_expression :
  ((73 ^ 1253 * 44 ^ 987 + 47 ^ 123 / 39 ^ 654 * 86 ^ 1484 - 32 ^ 1987) % 10) = 2 := by
  sorry

end NUMINAMATH_GPT_ones_digit_expression_l2247_224711


namespace NUMINAMATH_GPT_stratified_sampling_third_year_students_l2247_224752

theorem stratified_sampling_third_year_students :
  let total_students := 900
  let first_year_students := 300
  let second_year_students := 200
  let third_year_students := 400
  let sample_size := 45
  let sampling_ratio := (sample_size : ℚ) / (total_students : ℚ)
  (third_year_students : ℚ) * sampling_ratio = 20 :=
by 
  let total_students := 900
  let first_year_students := 300
  let second_year_students := 200
  let third_year_students := 400
  let sample_size := 45
  let sampling_ratio := (sample_size : ℚ) / (total_students : ℚ)
  show (third_year_students : ℚ) * sampling_ratio = 20
  sorry

end NUMINAMATH_GPT_stratified_sampling_third_year_students_l2247_224752


namespace NUMINAMATH_GPT_distance_proof_l2247_224773

-- Define the speeds of Alice and Bob
def aliceSpeed : ℚ := 1 / 20 -- Alice's speed in miles per minute
def bobSpeed : ℚ := 3 / 40 -- Bob's speed in miles per minute

-- Define the time they walk/jog
def time : ℚ := 120 -- Time in minutes (2 hours)

-- Calculate the distances
def aliceDistance : ℚ := aliceSpeed * time -- Distance Alice walked
def bobDistance : ℚ := bobSpeed * time -- Distance Bob jogged

-- The total distance between Alice and Bob after 2 hours
def totalDistance : ℚ := aliceDistance + bobDistance

-- Prove that the total distance is 15 miles
theorem distance_proof : totalDistance = 15 := by
  sorry

end NUMINAMATH_GPT_distance_proof_l2247_224773


namespace NUMINAMATH_GPT_anna_score_correct_l2247_224757

-- Given conditions
def correct_answers : ℕ := 17
def incorrect_answers : ℕ := 6
def unanswered_questions : ℕ := 7
def point_per_correct : ℕ := 1
def point_per_incorrect : ℕ := 0
def deduction_per_unanswered : ℤ := -1 / 2

-- Proving the score
theorem anna_score_correct : 
  correct_answers * point_per_correct + incorrect_answers * point_per_incorrect + unanswered_questions * deduction_per_unanswered = 27 / 2 :=
by
  sorry

end NUMINAMATH_GPT_anna_score_correct_l2247_224757


namespace NUMINAMATH_GPT_blue_hat_cost_is_6_l2247_224731

-- Total number of hats is 85
def total_hats : ℕ := 85

-- Number of green hats
def green_hats : ℕ := 20

-- Number of blue hats
def blue_hats : ℕ := total_hats - green_hats

-- Cost of each green hat
def cost_per_green_hat : ℕ := 7

-- Total cost for all hats
def total_cost : ℕ := 530

-- Total cost of green hats
def total_cost_green_hats : ℕ := green_hats * cost_per_green_hat

-- Total cost of blue hats
def total_cost_blue_hats : ℕ := total_cost - total_cost_green_hats

-- Cost per blue hat
def cost_per_blue_hat : ℕ := total_cost_blue_hats / blue_hats 

-- Prove that the cost of each blue hat is $6
theorem blue_hat_cost_is_6 : cost_per_blue_hat = 6 :=
by
  sorry

end NUMINAMATH_GPT_blue_hat_cost_is_6_l2247_224731


namespace NUMINAMATH_GPT_find_number_of_women_l2247_224732

-- Define the work rate variables and the equations from conditions
variables (m w : ℝ) (x : ℝ)

-- Define the first condition
def condition1 : Prop := 3 * m + x * w = 6 * m + 2 * w

-- Define the second condition
def condition2 : Prop := 4 * m + 2 * w = (5 / 7) * (3 * m + x * w)

-- The theorem stating that, given the above conditions, x must be 23
theorem find_number_of_women (hmw : m = 7 * w) (h1 : condition1 m w x) (h2 : condition2 m w x) : x = 23 :=
sorry

end NUMINAMATH_GPT_find_number_of_women_l2247_224732


namespace NUMINAMATH_GPT_three_digit_decimal_bounds_l2247_224735

def is_rounded_half_up (x : ℝ) (y : ℝ) : Prop :=
  (y - 0.005 ≤ x) ∧ (x < y + 0.005)

theorem three_digit_decimal_bounds :
  ∃ (x : ℝ), (8.725 ≤ x) ∧ (x ≤ 8.734) ∧ is_rounded_half_up x 8.73 :=
by
  sorry

end NUMINAMATH_GPT_three_digit_decimal_bounds_l2247_224735


namespace NUMINAMATH_GPT_bacteria_growth_time_l2247_224760

theorem bacteria_growth_time (initial_bacteria : ℕ) (final_bacteria : ℕ) (doubling_time : ℕ) :
  initial_bacteria = 1000 →
  final_bacteria = 128000 →
  doubling_time = 3 →
  (∃ t : ℕ, final_bacteria = initial_bacteria * 2 ^ (t / doubling_time) ∧ t = 21) :=
by
  sorry

end NUMINAMATH_GPT_bacteria_growth_time_l2247_224760


namespace NUMINAMATH_GPT_leak_empty_time_l2247_224753

-- Define the given conditions
def tank_volume := 2160 -- Tank volume in litres
def inlet_rate := 6 * 60 -- Inlet rate in litres per hour
def combined_empty_time := 12 -- Time in hours to empty the tank with the inlet on

-- Define the derived conditions
def net_rate := tank_volume / combined_empty_time -- Net rate of emptying in litres per hour

-- Define the rate of leakage
def leak_rate := inlet_rate - net_rate -- Rate of leak in litres per hour

-- Prove the main statement
theorem leak_empty_time : (2160 / leak_rate) = 12 :=
by
  unfold leak_rate
  exact sorry

end NUMINAMATH_GPT_leak_empty_time_l2247_224753


namespace NUMINAMATH_GPT_number_of_minibusses_l2247_224768

def total_students := 156
def students_per_van := 10
def students_per_minibus := 24
def number_of_vans := 6

theorem number_of_minibusses : (total_students - number_of_vans * students_per_van) / students_per_minibus = 4 :=
by
  sorry

end NUMINAMATH_GPT_number_of_minibusses_l2247_224768


namespace NUMINAMATH_GPT_decagon_diagonals_intersect_probability_l2247_224759

theorem decagon_diagonals_intersect_probability :
  let n := 10  -- number of vertices in decagon
  let diagonals := n * (n - 3) / 2  -- number of diagonals in decagon
  let pairs_diagonals := (diagonals * (diagonals - 1)) / 2  -- ways to choose 2 diagonals from diagonals
  let ways_choose_4 := Nat.choose 10 4  -- ways to choose 4 vertices from 10
  let probability := (4 * ways_choose_4) / pairs_diagonals  -- four vertices chosen determine two intersecting diagonals forming a convex quadrilateral
  probability = (210 / 595) := by
  -- Definitions (diagonals, pairs_diagonals, ways_choose_4) are directly used as hypothesis

  sorry  -- skipping the proof

end NUMINAMATH_GPT_decagon_diagonals_intersect_probability_l2247_224759


namespace NUMINAMATH_GPT_original_group_size_l2247_224798

theorem original_group_size (M : ℕ) (R : ℕ) :
  (M * R * 40 = (M - 5) * R * 50) → M = 25 :=
by
  sorry

end NUMINAMATH_GPT_original_group_size_l2247_224798


namespace NUMINAMATH_GPT_like_terms_sum_l2247_224771

theorem like_terms_sum (m n : ℕ) (h1 : 2 * m = 2) (h2 : n = 3) : m + n = 4 :=
sorry

end NUMINAMATH_GPT_like_terms_sum_l2247_224771


namespace NUMINAMATH_GPT_set_of_values_l2247_224703

theorem set_of_values (a : ℝ) (h : 2 ∉ {x : ℝ | x - a < 0}) : a ≤ 2 := 
sorry

end NUMINAMATH_GPT_set_of_values_l2247_224703


namespace NUMINAMATH_GPT_average_branches_per_foot_correct_l2247_224708

def height_tree_1 : ℕ := 50
def branches_tree_1 : ℕ := 200
def height_tree_2 : ℕ := 40
def branches_tree_2 : ℕ := 180
def height_tree_3 : ℕ := 60
def branches_tree_3 : ℕ := 180
def height_tree_4 : ℕ := 34
def branches_tree_4 : ℕ := 153

def total_height := height_tree_1 + height_tree_2 + height_tree_3 + height_tree_4
def total_branches := branches_tree_1 + branches_tree_2 + branches_tree_3 + branches_tree_4
def average_branches_per_foot := total_branches / total_height

theorem average_branches_per_foot_correct : average_branches_per_foot = 713 / 184 := 
  by
    -- Proof omitted, directly state the result
    sorry

end NUMINAMATH_GPT_average_branches_per_foot_correct_l2247_224708


namespace NUMINAMATH_GPT_final_price_percentage_of_original_l2247_224709

theorem final_price_percentage_of_original (original_price sale_price final_price : ℝ)
  (h1 : sale_price = original_price * 0.5)
  (h2 : final_price = sale_price * 0.9) :
  final_price = original_price * 0.45 :=
by
  sorry

end NUMINAMATH_GPT_final_price_percentage_of_original_l2247_224709


namespace NUMINAMATH_GPT_bullet_speed_difference_l2247_224719

def speed_horse : ℕ := 20  -- feet per second
def speed_bullet : ℕ := 400  -- feet per second

def speed_forward : ℕ := speed_bullet + speed_horse
def speed_backward : ℕ := speed_bullet - speed_horse

theorem bullet_speed_difference : speed_forward - speed_backward = 40 :=
by
  sorry

end NUMINAMATH_GPT_bullet_speed_difference_l2247_224719


namespace NUMINAMATH_GPT_Linda_total_amount_at_21_years_l2247_224791

theorem Linda_total_amount_at_21_years (P : ℝ) (r : ℝ) (n : ℕ) (initial_principal : P = 1500) (annual_rate : r = 0.03) (years : n = 21):
    P * (1 + r)^n = 2709.17 :=
by
  sorry

end NUMINAMATH_GPT_Linda_total_amount_at_21_years_l2247_224791


namespace NUMINAMATH_GPT_avg_weight_of_children_is_138_l2247_224744

-- Define the average weight of boys and girls
def average_weight_of_boys := 150
def number_of_boys := 6
def average_weight_of_girls := 120
def number_of_girls := 4

-- Calculate total weights and average weight of all children
noncomputable def total_weight_of_boys := number_of_boys * average_weight_of_boys
noncomputable def total_weight_of_girls := number_of_girls * average_weight_of_girls
noncomputable def total_weight_of_children := total_weight_of_boys + total_weight_of_girls
noncomputable def number_of_children := number_of_boys + number_of_girls
noncomputable def average_weight_of_children := total_weight_of_children / number_of_children

-- Lean statement to prove the average weight of all children is 138 pounds
theorem avg_weight_of_children_is_138 : average_weight_of_children = 138 := by
    sorry

end NUMINAMATH_GPT_avg_weight_of_children_is_138_l2247_224744


namespace NUMINAMATH_GPT_numer_greater_than_denom_iff_l2247_224727

theorem numer_greater_than_denom_iff (x : ℝ) (h : -1 ≤ x ∧ x ≤ 3) : 
  (4 * x - 3 > 9 - 2 * x) ↔ (2 < x ∧ x ≤ 3) :=
sorry

end NUMINAMATH_GPT_numer_greater_than_denom_iff_l2247_224727


namespace NUMINAMATH_GPT_solve_equation_l2247_224713

theorem solve_equation (x : ℝ) (h₁ : x ≠ 2) (h₂ : x ≠ 0) :
  (3 / (x - 2) = 2 + x / (2 - x)) ↔ x = 7 :=
sorry

end NUMINAMATH_GPT_solve_equation_l2247_224713


namespace NUMINAMATH_GPT_intersection_point_of_lines_l2247_224755

theorem intersection_point_of_lines :
  ∃ x y : ℚ, 
    (y = -3 * x + 4) ∧ 
    (y = (1 / 3) * x + 1) ∧ 
    x = 9 / 10 ∧ 
    y = 13 / 10 :=
by sorry

end NUMINAMATH_GPT_intersection_point_of_lines_l2247_224755


namespace NUMINAMATH_GPT_capital_payment_l2247_224751

theorem capital_payment (m : ℕ) (hm : m ≥ 3) : 
  ∃ d : ℕ, d = (1000 * (3^m - 2^(m-1))) / (3^m - 2^m) 
  ∧ (∃ a : ℕ, a = 4000 ∧ a = ((3/2)^(m-1) * (3000 - 3 * d) + 2 * d)) := 
by
  sorry

end NUMINAMATH_GPT_capital_payment_l2247_224751


namespace NUMINAMATH_GPT_judson_contribution_l2247_224792

theorem judson_contribution (J K C : ℝ) (hK : K = 1.20 * J) (hC : C = K + 200) (h_total : J + K + C = 1900) : J = 500 :=
by
  -- This is where the proof would go, but we are skipping it as per the instructions.
  sorry

end NUMINAMATH_GPT_judson_contribution_l2247_224792


namespace NUMINAMATH_GPT_find_largest_number_l2247_224730

theorem find_largest_number
  (a b c d : ℕ)
  (h1 : a + b + c = 222)
  (h2 : a + b + d = 208)
  (h3 : a + c + d = 197)
  (h4 : b + c + d = 180) :
  max a (max b (max c d)) = 89 :=
by
  sorry

end NUMINAMATH_GPT_find_largest_number_l2247_224730


namespace NUMINAMATH_GPT_average_age_of_adults_l2247_224781

theorem average_age_of_adults 
  (total_members : ℕ)
  (avg_age_total : ℕ)
  (num_girls : ℕ)
  (num_boys : ℕ)
  (num_adults : ℕ)
  (avg_age_girls : ℕ)
  (avg_age_boys : ℕ)
  (total_sum_ages : ℕ := total_members * avg_age_total)
  (sum_ages_girls : ℕ := num_girls * avg_age_girls)
  (sum_ages_boys : ℕ := num_boys * avg_age_boys)
  (sum_ages_adults : ℕ := total_sum_ages - sum_ages_girls - sum_ages_boys)
  : (num_adults = 10) → (avg_age_total = 20) → (num_girls = 30) → (avg_age_girls = 18) → (num_boys = 20) → (avg_age_boys = 22) → (total_sum_ages = 1200) → (sum_ages_girls = 540) → (sum_ages_boys = 440) → (sum_ages_adults = 220) → (sum_ages_adults / num_adults = 22) :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9 h10
  sorry

end NUMINAMATH_GPT_average_age_of_adults_l2247_224781


namespace NUMINAMATH_GPT_total_skips_correct_l2247_224764

def S (n : ℕ) : ℕ := n^2 + n

def TotalSkips5 : ℕ :=
  S 1 + S 2 + S 3 + S 4 + S 5

def Skips6 : ℕ :=
  2 * S 6

theorem total_skips_correct : TotalSkips5 + Skips6 = 154 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_total_skips_correct_l2247_224764


namespace NUMINAMATH_GPT_items_from_B_l2247_224780

noncomputable def totalItems : ℕ := 1200
noncomputable def ratioA : ℕ := 3
noncomputable def ratioB : ℕ := 4
noncomputable def ratioC : ℕ := 5
noncomputable def totalRatio : ℕ := ratioA + ratioB + ratioC
noncomputable def sampledItems : ℕ := 60
noncomputable def numberB := sampledItems * ratioB / totalRatio

theorem items_from_B :
  numberB = 20 :=
by
  sorry

end NUMINAMATH_GPT_items_from_B_l2247_224780


namespace NUMINAMATH_GPT_shopkeeper_total_cards_l2247_224761

-- Definition of the number of cards in a complete deck
def cards_in_deck : Nat := 52

-- Definition of the number of complete decks the shopkeeper has
def number_of_decks : Nat := 3

-- Definition of the additional cards the shopkeeper has
def additional_cards : Nat := 4

-- The total number of cards the shopkeeper should have
def total_cards : Nat := number_of_decks * cards_in_deck + additional_cards

-- Theorem statement to prove the total number of cards is 160
theorem shopkeeper_total_cards : total_cards = 160 := by
  sorry

end NUMINAMATH_GPT_shopkeeper_total_cards_l2247_224761


namespace NUMINAMATH_GPT_find_x_squared_inv_x_squared_l2247_224733

theorem find_x_squared_inv_x_squared (x : ℝ) (h : x^3 + 1/x^3 = 110) : x^2 + 1/x^2 = 23 :=
sorry

end NUMINAMATH_GPT_find_x_squared_inv_x_squared_l2247_224733


namespace NUMINAMATH_GPT_theater_ticket_sales_l2247_224785

theorem theater_ticket_sales (R H : ℕ) (h1 : R = 25) (h2 : H = 3 * R + 18) : H = 93 :=
by
  sorry

end NUMINAMATH_GPT_theater_ticket_sales_l2247_224785


namespace NUMINAMATH_GPT_two_students_follow_all_celebrities_l2247_224782

theorem two_students_follow_all_celebrities :
  ∀ (students : Finset ℕ) (celebrities_followers : ℕ → Finset ℕ),
    (students.card = 120) →
    (∀ c : ℕ, c < 10 → (celebrities_followers c).card ≥ 85 ∧ (celebrities_followers c) ⊆ students) →
    ∃ (s1 s2 : ℕ), s1 ∈ students ∧ s2 ∈ students ∧ s1 ≠ s2 ∧
      (∀ c : ℕ, c < 10 → (s1 ∈ celebrities_followers c ∨ s2 ∈ celebrities_followers c)) :=
by
  intros students celebrities_followers h_students_card h_followers_cond
  sorry

end NUMINAMATH_GPT_two_students_follow_all_celebrities_l2247_224782


namespace NUMINAMATH_GPT_point_transformation_l2247_224783

theorem point_transformation : ∀ (P : ℝ×ℝ), P = (1, -2) → P = (-1, 2) :=
by
  sorry

end NUMINAMATH_GPT_point_transformation_l2247_224783


namespace NUMINAMATH_GPT_valid_seating_arrangements_l2247_224707

theorem valid_seating_arrangements :
  let total_arrangements := Nat.factorial 10
  let restricted_arrangements := Nat.factorial 7 * Nat.factorial 4
  total_arrangements - restricted_arrangements = 3507840 :=
by
  sorry

end NUMINAMATH_GPT_valid_seating_arrangements_l2247_224707


namespace NUMINAMATH_GPT_three_2x2_squares_exceed_100_l2247_224739

open BigOperators

noncomputable def sum_of_1_to_64 : ℕ :=
  (64 * (64 + 1)) / 2

theorem three_2x2_squares_exceed_100 :
  ∀ (s : Fin 16 → ℕ),
    (∑ i, s i = sum_of_1_to_64) →
    (∀ i j, i ≠ j → s i = s j ∨ s i > s j ∨ s i < s j) →
    (∃ i₁ i₂ i₃, i₁ ≠ i₂ ∧ i₂ ≠ i₃ ∧ i₁ ≠ i₃ ∧ s i₁ > 100 ∧ s i₂ > 100 ∧ s i₃ > 100) := sorry

end NUMINAMATH_GPT_three_2x2_squares_exceed_100_l2247_224739


namespace NUMINAMATH_GPT_solve_for_s_l2247_224775

theorem solve_for_s (s t : ℚ) (h1 : 15 * s + 7 * t = 210) (h2 : t = 3 * s) : s = 35 / 6 := 
by
  sorry

end NUMINAMATH_GPT_solve_for_s_l2247_224775


namespace NUMINAMATH_GPT_min_birthday_employees_wednesday_l2247_224726

theorem min_birthday_employees_wednesday :
  ∀ (employees : ℕ) (n : ℕ), 
  employees = 50 → 
  n ≥ 1 →
  ∃ (x : ℕ), 6 * x + (x + n) = employees ∧ x + n ≥ 8 :=
by
  sorry

end NUMINAMATH_GPT_min_birthday_employees_wednesday_l2247_224726


namespace NUMINAMATH_GPT_parabola_points_relation_l2247_224736

theorem parabola_points_relation {a b c y1 y2 y3 : ℝ} 
  (hA : y1 = a * (1 / 2)^2 + b * (1 / 2) + c)
  (hB : y2 = a * (0)^2 + b * (0) + c)
  (hC : y3 = a * (-1)^2 + b * (-1) + c)
  (h_cond : 0 < 2 * a ∧ 2 * a < b) : 
  y1 > y2 ∧ y2 > y3 :=
by 
  sorry

end NUMINAMATH_GPT_parabola_points_relation_l2247_224736


namespace NUMINAMATH_GPT_green_pill_cost_is_21_l2247_224765

-- Definitions based on conditions
def number_of_days : ℕ := 21
def total_cost : ℕ := 819
def daily_cost : ℕ := total_cost / number_of_days
def green_pill_cost (pink_pill_cost : ℕ) : ℕ := pink_pill_cost + 3

-- Given pink pill cost is x, then green pill cost is x + 3
-- We need to prove that for some x, the daily cost of the pills equals 39, and thus green pill cost is 21

theorem green_pill_cost_is_21 (pink_pill_cost : ℕ) (h : daily_cost = (green_pill_cost pink_pill_cost) + pink_pill_cost) :
    green_pill_cost pink_pill_cost = 21 :=
by
  sorry

end NUMINAMATH_GPT_green_pill_cost_is_21_l2247_224765


namespace NUMINAMATH_GPT_range_of_k_l2247_224728

theorem range_of_k
  (x y k : ℝ)
  (h1 : 3 * x + y = k + 1)
  (h2 : x + 3 * y = 3)
  (h3 : 0 < x + y)
  (h4 : x + y < 1) :
  -4 < k ∧ k < 0 :=
sorry

end NUMINAMATH_GPT_range_of_k_l2247_224728


namespace NUMINAMATH_GPT_wheel_speed_is_12_mph_l2247_224767

theorem wheel_speed_is_12_mph
  (r : ℝ) -- speed in miles per hour
  (C : ℝ := 15 / 5280) -- circumference in miles
  (H1 : ∃ t, r * t = C * 3600) -- initial condition that speed times time for one rotation equals 15/5280 miles in seconds
  (H2 : ∃ t, (r + 7) * (t - 1/21600) = C * 3600) -- condition that speed increases by 7 mph when time shortens by 1/6 second
  : r = 12 :=
sorry

end NUMINAMATH_GPT_wheel_speed_is_12_mph_l2247_224767


namespace NUMINAMATH_GPT_ratio_monkeys_camels_l2247_224769

-- Definitions corresponding to conditions
variables (zebras camels monkeys giraffes : ℕ)
variables (multiple : ℕ)

-- Conditions
def condition1 := zebras = 12
def condition2 := camels = zebras / 2
def condition3 := monkeys = camels * multiple
def condition4 := giraffes = 2
def condition5 := monkeys = giraffes + 22

-- Question: What is the ratio of monkeys to camels? Prove it is 4:1 given the conditions.
theorem ratio_monkeys_camels (zebras camels monkeys giraffes multiple : ℕ) 
  (h1 : condition1 zebras) 
  (h2 : condition2 zebras camels)
  (h3 : condition3 camels monkeys multiple)
  (h4 : condition4 giraffes)
  (h5 : condition5 monkeys giraffes) :
  multiple = 4 :=
sorry

end NUMINAMATH_GPT_ratio_monkeys_camels_l2247_224769


namespace NUMINAMATH_GPT_phase_shift_of_sine_function_l2247_224729

theorem phase_shift_of_sine_function :
  ∀ x : ℝ, y = 3 * Real.sin (3 * x + π / 4) → (∃ φ : ℝ, φ = -π / 12) :=
by sorry

end NUMINAMATH_GPT_phase_shift_of_sine_function_l2247_224729


namespace NUMINAMATH_GPT_cost_of_50_lavenders_l2247_224721

noncomputable def cost_of_bouquet (lavenders : ℕ) : ℚ :=
  (25 / 15) * lavenders

theorem cost_of_50_lavenders :
  cost_of_bouquet 50 = 250 / 3 :=
sorry

end NUMINAMATH_GPT_cost_of_50_lavenders_l2247_224721


namespace NUMINAMATH_GPT_largest_among_given_numbers_l2247_224712

theorem largest_among_given_numbers : 
    let a := 24680 + (1 / 1357)
    let b := 24680 - (1 / 1357)
    let c := 24680 * (1 / 1357)
    let d := 24680 / (1 / 1357)
    let e := 24680.1357
    d > a ∧ d > b ∧ d > c ∧ d > e :=
by
  sorry

end NUMINAMATH_GPT_largest_among_given_numbers_l2247_224712


namespace NUMINAMATH_GPT_girls_first_half_l2247_224779

theorem girls_first_half (total_students boys_first_half girls_first_half boys_second_half girls_second_half boys_whole_year : ℕ)
  (h1: total_students = 56)
  (h2: boys_first_half = 25)
  (h3: girls_first_half = 15)
  (h4: boys_second_half = 26)
  (h5: girls_second_half = 25)
  (h6: boys_whole_year = 23) : 
  ∃ girls_first_half_only : ℕ, girls_first_half_only = 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_girls_first_half_l2247_224779


namespace NUMINAMATH_GPT_total_airflow_correct_l2247_224745

def airflow_fan_A : ℕ := 10 * 10 * 60 * 7
def airflow_fan_B : ℕ := 15 * 20 * 60 * 5
def airflow_fan_C : ℕ := 25 * 30 * 60 * 5
def airflow_fan_D : ℕ := 20 * 15 * 60 * 2
def airflow_fan_E : ℕ := 30 * 60 * 60 * 6

def total_airflow : ℕ :=
  airflow_fan_A + airflow_fan_B + airflow_fan_C + airflow_fan_D + airflow_fan_E

theorem total_airflow_correct : total_airflow = 1041000 := by
  sorry

end NUMINAMATH_GPT_total_airflow_correct_l2247_224745


namespace NUMINAMATH_GPT_more_girls_than_boys_l2247_224716

def initial_girls : ℕ := 632
def initial_boys : ℕ := 410
def new_girls_joined : ℕ := 465
def total_girls : ℕ := initial_girls + new_girls_joined

theorem more_girls_than_boys :
  total_girls - initial_boys = 687 :=
by
  -- Proof goes here
  sorry


end NUMINAMATH_GPT_more_girls_than_boys_l2247_224716


namespace NUMINAMATH_GPT_find_c_plus_d_l2247_224778

theorem find_c_plus_d (a b c d : ℤ) (h1 : a + b = 14) (h2 : b + c = 9) (h3 : a + d = 8) : c + d = 3 := 
by
  sorry

end NUMINAMATH_GPT_find_c_plus_d_l2247_224778


namespace NUMINAMATH_GPT_steve_family_time_l2247_224777

theorem steve_family_time :
  let day_hours := 24
  let sleeping_fraction := 1 / 3
  let school_fraction := 1 / 6
  let assignments_fraction := 1 / 12
  let sleeping_hours := day_hours * sleeping_fraction
  let school_hours := day_hours * school_fraction
  let assignments_hours := day_hours * assignments_fraction
  let total_activity_hours := sleeping_hours + school_hours + assignments_hours
  day_hours - total_activity_hours = 10 :=
by
  let day_hours := 24
  let sleeping_fraction := 1 / 3
  let school_fraction := 1 / 6
  let assignments_fraction := 1 / 12
  let sleeping_hours := day_hours * sleeping_fraction
  let school_hours := day_hours * school_fraction
  let assignments_hours := day_hours *  assignments_fraction
  let total_activity_hours := sleeping_hours +
                              school_hours + 
                              assignments_hours
  show day_hours - total_activity_hours = 10
  sorry

end NUMINAMATH_GPT_steve_family_time_l2247_224777


namespace NUMINAMATH_GPT_total_cost_pencils_l2247_224749

theorem total_cost_pencils
  (boxes : ℕ)
  (cost_per_box : ℕ → ℕ → ℕ)
  (price_regular : ℕ)
  (price_bulk : ℕ)
  (box_size : ℕ)
  (bulk_threshold : ℕ)
  (total_pencils : ℕ) :
  total_pencils = 3150 →
  box_size = 150 →
  price_regular = 40 →
  price_bulk = 35 →
  bulk_threshold = 2000 →
  boxes = (total_pencils + box_size - 1) / box_size →
  (total_pencils > bulk_threshold → cost_per_box boxes price_bulk = boxes * price_bulk) →
  (total_pencils ≤ bulk_threshold → cost_per_box boxes price_regular = boxes * price_regular) →
  total_pencils > bulk_threshold →
  cost_per_box boxes price_bulk = 735 :=
by
  intro h_total_pencils
  intro h_box_size
  intro h_price_regular
  intro h_price_bulk
  intro h_bulk_threshold
  intro h_boxes
  intro h_cost_bulk
  intro h_cost_regular
  intro h_bulk_discount_passt
  -- sorry statement as we don't provide the actual proof here
  sorry

end NUMINAMATH_GPT_total_cost_pencils_l2247_224749


namespace NUMINAMATH_GPT_ray_reflection_and_distance_l2247_224788

-- Define the initial conditions
def pointA : ℝ × ℝ := (-3, 3)
def circleC_eq (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 4*y + 7 = 0

-- Definitions of the lines for incident and reflected rays
def incident_ray_line (x y : ℝ) : Prop := 4*x + 3*y + 3 = 0
def reflected_ray_line (x y : ℝ) : Prop := 3*x + 4*y - 3 = 0

-- Distance traveled by the ray
def distance_traveled (A T : ℝ × ℝ) := 7

theorem ray_reflection_and_distance :
  ∃ (x₁ y₁ : ℝ), incident_ray_line x₁ y₁ ∧ reflected_ray_line x₁ y₁ ∧ circleC_eq x₁ y₁ ∧ 
  (∀ (P : ℝ × ℝ), P = pointA → distance_traveled P (x₁, y₁) = 7) :=
sorry

end NUMINAMATH_GPT_ray_reflection_and_distance_l2247_224788


namespace NUMINAMATH_GPT_range_of_y_l2247_224702

theorem range_of_y (x : ℝ) : 
  - (Real.sqrt 3) / 3 ≤ (Real.sin x) / (2 - Real.cos x) ∧ (Real.sin x) / (2 - Real.cos x) ≤ (Real.sqrt 3) / 3 :=
sorry

end NUMINAMATH_GPT_range_of_y_l2247_224702


namespace NUMINAMATH_GPT_distance_from_origin_to_line_l2247_224704

def ellipse (x y : ℝ) : Prop := (x^2 / 2) + y^2 = 1

-- definition of the perpendicular property of chords
def perpendicular (O A B : ℝ × ℝ) : Prop := (A.1 * B.1 + A.2 * B.2 = 0)

theorem distance_from_origin_to_line
  (xA yA xB yB : ℝ)
  (hA : ellipse xA yA)
  (hB : ellipse xB yB)
  (h_perpendicular : perpendicular (0, 0) (xA, yA) (xB, yB))
  : ∃ d : ℝ, d = (Real.sqrt 6) / 3 :=
sorry

end NUMINAMATH_GPT_distance_from_origin_to_line_l2247_224704


namespace NUMINAMATH_GPT_total_students_l2247_224705

theorem total_students (T : ℝ) (h : 0.50 * T = 440) : T = 880 := 
by {
  sorry
}

end NUMINAMATH_GPT_total_students_l2247_224705


namespace NUMINAMATH_GPT_david_profit_l2247_224776

def weight : ℝ := 50
def cost : ℝ := 50
def price_per_kg : ℝ := 1.20
def total_earnings : ℝ := weight * price_per_kg
def profit : ℝ := total_earnings - cost

theorem david_profit : profit = 10 := by
  sorry

end NUMINAMATH_GPT_david_profit_l2247_224776


namespace NUMINAMATH_GPT_yuna_initial_pieces_l2247_224784

variable (Y : ℕ)

theorem yuna_initial_pieces
  (namjoon_initial : ℕ := 250)
  (given_pieces : ℕ := 60)
  (namjoon_after : namjoon_initial - given_pieces = Y + given_pieces - 20) :
  Y = 150 :=
by
  sorry

end NUMINAMATH_GPT_yuna_initial_pieces_l2247_224784


namespace NUMINAMATH_GPT_retail_price_l2247_224714

/-- A retailer bought a machine at a wholesale price of $99 and later sold it after a 10% discount of the retail price.
If the retailer made a profit equivalent to 20% of the wholesale price, then the retail price of the machine before the discount was $132. -/
theorem retail_price (wholesale_price : ℝ) (profit_percent discount_percent : ℝ) (P : ℝ) 
  (h₁ : wholesale_price = 99) 
  (h₂ : profit_percent = 0.20) 
  (h₃ : discount_percent = 0.10)
  (h₄ : (1 - discount_percent) * P = wholesale_price + profit_percent * wholesale_price) : 
  P = 132 := 
by
  sorry

end NUMINAMATH_GPT_retail_price_l2247_224714


namespace NUMINAMATH_GPT_flight_distance_l2247_224741

theorem flight_distance (D : ℝ) :
  let t_out := D / 300
  let t_return := D / 500
  t_out + t_return = 8 -> D = 1500 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_flight_distance_l2247_224741


namespace NUMINAMATH_GPT_solution_comparison_l2247_224722

variables (a a' b b' : ℝ)

theorem solution_comparison (ha : a ≠ 0) (ha' : a' ≠ 0) :
  (-(b / a) < -(b' / a')) ↔ (b' / a' < b / a) :=
by sorry

end NUMINAMATH_GPT_solution_comparison_l2247_224722


namespace NUMINAMATH_GPT_smallest_YZ_minus_XZ_l2247_224790

theorem smallest_YZ_minus_XZ 
  (XZ YZ XY : ℕ)
  (h_sum : XZ + YZ + XY = 3001)
  (h_order : XZ < YZ ∧ YZ ≤ XY)
  (h_triangle_ineq1 : XZ + YZ > XY)
  (h_triangle_ineq2 : XZ + XY > YZ)
  (h_triangle_ineq3 : YZ + XY > XZ) :
  ∃ XZ YZ XY : ℕ, YZ - XZ = 1 := sorry

end NUMINAMATH_GPT_smallest_YZ_minus_XZ_l2247_224790


namespace NUMINAMATH_GPT_determine_omega_phi_l2247_224795

noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (ω * x + φ)

theorem determine_omega_phi (ω φ : ℝ) (x : ℝ)
  (h₁ : 0 < ω) (h₂ : |φ| < Real.pi)
  (h₃ : f ω φ (5 * Real.pi / 8) = 2)
  (h₄ : f ω φ (11 * Real.pi / 8) = 0)
  (h₅ : (2 * Real.pi / ω) > 2 * Real.pi) :
  ω = 2 / 3 ∧ φ = Real.pi / 12 :=
sorry

end NUMINAMATH_GPT_determine_omega_phi_l2247_224795


namespace NUMINAMATH_GPT_simplify_and_evaluate_expression_l2247_224774

theorem simplify_and_evaluate_expression (x : ℤ) (h : x = 2 ∨ x ≠ -1 ∧ x ≠ 0 ∧ x ≠ 1 ∧ x ≠ -1) :
  ((1 / (x:ℚ) - 1 / (x + 1)) / ((x^2 - 1) / (x^2 + 2*x + 1))) = (1 / 2) :=
by
  -- Skipping the proof
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_expression_l2247_224774


namespace NUMINAMATH_GPT_difference_in_cents_l2247_224758

theorem difference_in_cents (pennies dimes : ℕ) (h : pennies + dimes = 5050) (hpennies : 1 ≤ pennies) (hdimes : 1 ≤ dimes) : 
  let total_value := pennies + 10 * dimes
  let max_value := 50500 - 9 * 1
  let min_value := 50500 - 9 * 5049
  max_value - min_value = 45432 := 
by 
  -- proof goes here
  sorry

end NUMINAMATH_GPT_difference_in_cents_l2247_224758


namespace NUMINAMATH_GPT_minimum_planks_required_l2247_224715

theorem minimum_planks_required (colors : Finset ℕ) (planks : List ℕ) :
  colors.card = 100 ∧
  ∀ i j, i ∈ colors → j ∈ colors → i ≠ j →
  ∃ k₁ k₂, k₁ < k₂ ∧ planks.get? k₁ = some i ∧ planks.get? k₂ = some j
  → planks.length = 199 := 
sorry

end NUMINAMATH_GPT_minimum_planks_required_l2247_224715


namespace NUMINAMATH_GPT_vectorBC_computation_l2247_224700

open Vector

def vectorAB : ℝ × ℝ := (2, 4)

def vectorAC : ℝ × ℝ := (1, 3)

theorem vectorBC_computation :
  (vectorAC.1 - vectorAB.1, vectorAC.2 - vectorAB.2) = (-1, -1) :=
sorry

end NUMINAMATH_GPT_vectorBC_computation_l2247_224700


namespace NUMINAMATH_GPT_pradeep_passing_percentage_l2247_224772

theorem pradeep_passing_percentage (score failed_by max_marks : ℕ) :
  score = 185 → failed_by = 25 → max_marks = 600 →
  ((score + failed_by) / max_marks : ℚ) * 100 = 35 :=
by
  intros h_score h_failed_by h_max_marks
  sorry

end NUMINAMATH_GPT_pradeep_passing_percentage_l2247_224772


namespace NUMINAMATH_GPT_determine_s_value_l2247_224743

def f (x : ℚ) : ℚ := abs (x - 1) - abs x

def u : ℚ := f (5 / 16)
def v : ℚ := f u
def s : ℚ := f v

theorem determine_s_value : s = 1 / 2 :=
by
  -- Proof needed here
  sorry

end NUMINAMATH_GPT_determine_s_value_l2247_224743


namespace NUMINAMATH_GPT_roots_quadratic_inequality_l2247_224710

theorem roots_quadratic_inequality (t x1 x2 : ℝ) (h_eqn : x1 ^ 2 - t * x1 + t = 0) 
  (h_eqn2 : x2 ^ 2 - t * x2 + t = 0) (h_real : x1 + x2 = t) (h_prod : x1 * x2 = t) :
  x1 ^ 2 + x2 ^ 2 ≥ 2 * (x1 + x2) := 
sorry

end NUMINAMATH_GPT_roots_quadratic_inequality_l2247_224710


namespace NUMINAMATH_GPT_find_possible_values_l2247_224717

theorem find_possible_values (x : ℝ) (h : x^2 + 9 * (x / (x - 3))^2 = 90) :
  ∃ y, (y = (x - 3)^2 * (x + 4) / (3 * x - 4)) ∧ (y = 36 / 11 ∨ y = 468 / 23) :=
by
  sorry

end NUMINAMATH_GPT_find_possible_values_l2247_224717


namespace NUMINAMATH_GPT_second_day_more_than_third_day_l2247_224724

-- Define the conditions
def total_people (d1 d2 d3 : ℕ) := d1 + d2 + d3 = 246 
def first_day := 79
def third_day := 120

-- Define the statement to prove
theorem second_day_more_than_third_day : 
  ∃ d2 : ℕ, total_people first_day d2 third_day ∧ (d2 - third_day) = 47 :=
by
  sorry

end NUMINAMATH_GPT_second_day_more_than_third_day_l2247_224724


namespace NUMINAMATH_GPT_total_get_well_cards_l2247_224793

-- Definitions for the number of cards received in each place
def cardsInHospital : ℕ := 403
def cardsAtHome : ℕ := 287

-- Theorem statement:
theorem total_get_well_cards : cardsInHospital + cardsAtHome = 690 := by
  sorry

end NUMINAMATH_GPT_total_get_well_cards_l2247_224793


namespace NUMINAMATH_GPT_imo_2007_p6_l2247_224740

theorem imo_2007_p6 (x y : ℕ) (hx : 0 < x) (hy : 0 < y) :
  ∃ k : ℕ, (x = 11 * k^2) ∧ (y = 11 * k) ↔
  ∃ k : ℕ, (∃ k₁ : ℤ, k₁ = (x^2 * y + x + y) / (x * y^2 + y + 11)) :=
sorry

end NUMINAMATH_GPT_imo_2007_p6_l2247_224740


namespace NUMINAMATH_GPT_subtract_fractions_l2247_224754

theorem subtract_fractions (p q : ℚ) (h₁ : 4 / p = 8) (h₂ : 4 / q = 18) : p - q = 5 / 18 := 
by 
  sorry

end NUMINAMATH_GPT_subtract_fractions_l2247_224754


namespace NUMINAMATH_GPT_no_int_solutions_5x2_minus_4y2_eq_2017_l2247_224746

theorem no_int_solutions_5x2_minus_4y2_eq_2017 :
  ¬ ∃ x y : ℤ, 5 * x^2 - 4 * y^2 = 2017 :=
by
  -- The detailed proof goes here
  sorry

end NUMINAMATH_GPT_no_int_solutions_5x2_minus_4y2_eq_2017_l2247_224746


namespace NUMINAMATH_GPT_gcd_n3_plus_16_n_plus_4_l2247_224748

/-- For a given positive integer n > 2^4, the greatest common divisor of n^3 + 16 and n + 4 is 1. -/
theorem gcd_n3_plus_16_n_plus_4 (n : ℕ) (h : n > 2^4) : Nat.gcd (n^3 + 16) (n + 4) = 1 := by
  sorry

end NUMINAMATH_GPT_gcd_n3_plus_16_n_plus_4_l2247_224748


namespace NUMINAMATH_GPT_geometric_sequence_alpha5_eq_three_l2247_224796

theorem geometric_sequence_alpha5_eq_three (α : ℕ → ℝ) 
  (h1 : ∀ n : ℕ, α (n + 1) = α n * r) 
  (h2 : α 4 * α 5 * α 6 = 27) : α 5 = 3 := 
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_alpha5_eq_three_l2247_224796


namespace NUMINAMATH_GPT_arctan_tan_expr_is_75_degrees_l2247_224725

noncomputable def arctan_tan_expr : ℝ := Real.arctan (Real.tan (75 * Real.pi / 180) - 2 * Real.tan (30 * Real.pi / 180))

theorem arctan_tan_expr_is_75_degrees : (arctan_tan_expr * 180 / Real.pi) = 75 := 
by
  sorry

end NUMINAMATH_GPT_arctan_tan_expr_is_75_degrees_l2247_224725


namespace NUMINAMATH_GPT_alice_bob_total_dollars_l2247_224797

-- Define Alice's amount in dollars
def alice_amount : ℚ := 5 / 8

-- Define Bob's amount in dollars
def bob_amount : ℚ := 3 / 5

-- Define the total amount in dollars
def total_amount : ℚ := alice_amount + bob_amount

theorem alice_bob_total_dollars : (alice_amount + bob_amount : ℚ) = 1.225 := by
    sorry

end NUMINAMATH_GPT_alice_bob_total_dollars_l2247_224797


namespace NUMINAMATH_GPT_shara_savings_l2247_224706

theorem shara_savings 
  (original_price : ℝ)
  (discount1 : ℝ := 0.08)
  (discount2 : ℝ := 0.05)
  (sales_tax : ℝ := 0.06)
  (final_price : ℝ := 184)
  (h : (original_price * (1 - discount1) * (1 - discount2) * (1 + sales_tax)) = final_price) :
  original_price - final_price = 25.78 :=
sorry

end NUMINAMATH_GPT_shara_savings_l2247_224706


namespace NUMINAMATH_GPT_total_coins_is_twenty_l2247_224787

def piles_of_quarters := 2
def piles_of_dimes := 3
def coins_per_pile := 4

theorem total_coins_is_twenty : piles_of_quarters * coins_per_pile + piles_of_dimes * coins_per_pile = 20 :=
by sorry

end NUMINAMATH_GPT_total_coins_is_twenty_l2247_224787


namespace NUMINAMATH_GPT_inequality_solution_set_range_of_a_l2247_224742

def f (x : ℝ) : ℝ := abs (3*x + 2)

theorem inequality_solution_set :
  { x : ℝ | f x < 4 - abs (x - 1) } = { x : ℝ | -5/4 < x ∧ x < 1/2 } :=
by 
  sorry

theorem range_of_a (a : ℝ) (m n : ℝ) (h1 : m + n = 1) (h2 : 0 < m) (h3 : 0 < n) 
  (h4 : ∀ x : ℝ, abs (x - a) - f x ≤ 1 / m + 1 / n) : 
  0 < a ∧ a ≤ 10/3 :=
by 
  sorry

end NUMINAMATH_GPT_inequality_solution_set_range_of_a_l2247_224742


namespace NUMINAMATH_GPT_xy_sum_is_one_l2247_224799

theorem xy_sum_is_one (x y : ℝ) (h : x^2 + y^2 + x * y = 12 * x - 8 * y + 2) : x + y = 1 :=
sorry

end NUMINAMATH_GPT_xy_sum_is_one_l2247_224799


namespace NUMINAMATH_GPT_sum_of_all_possible_values_of_g_11_l2247_224789

def f (x : ℝ) : ℝ := x^2 - 6 * x + 14

def g (x : ℝ) : ℝ := 3 * x + 4

theorem sum_of_all_possible_values_of_g_11 :
  (∀ x : ℝ, f x = 11 → g x = 13 ∨ g x = 7) →
  (13 + 7 = 20) := by
  intros h
  sorry

end NUMINAMATH_GPT_sum_of_all_possible_values_of_g_11_l2247_224789


namespace NUMINAMATH_GPT_cost_of_five_dozen_l2247_224747

noncomputable def price_per_dozen (total_cost : ℝ) (num_dozen : ℕ) : ℝ :=
  total_cost / num_dozen

noncomputable def total_cost (price_per_dozen : ℝ) (num_dozen : ℕ) : ℝ :=
  price_per_dozen * num_dozen

theorem cost_of_five_dozen (total_cost_threedozens : ℝ := 28.20) (num_threedozens : ℕ := 3) (num_fivedozens : ℕ := 5) :
  total_cost (price_per_dozen total_cost_threedozens num_threedozens) num_fivedozens = 47.00 :=
  by sorry

end NUMINAMATH_GPT_cost_of_five_dozen_l2247_224747


namespace NUMINAMATH_GPT_part1_part2_l2247_224762

-- Definition of the function f
def f (a x : ℝ) : ℝ := |a * x - 2| - |x + 2|

-- First Proof Statement: Inequality for a = 2
theorem part1 : ∀ x : ℝ, - (1 : ℝ) / 3 ≤ x ∧ x ≤ 5 → f 2 x ≤ 1 :=
by
  sorry

-- Second Proof Statement: Range for a such that -4 ≤ f(x) ≤ 4 for all x ∈ ℝ
theorem part2 : ∀ x : ℝ, -4 ≤ f a x ∧ f a x ≤ 4 ↔ a = 1 ∨ a = -1 :=
by
  sorry

end NUMINAMATH_GPT_part1_part2_l2247_224762


namespace NUMINAMATH_GPT_negation_of_p_l2247_224770

open Real

-- Define the statement to be negated
def p := ∀ x : ℝ, -π/2 < x ∧ x < π/2 → tan x > 0

-- Define the negation of the statement
def not_p := ∃ x_0 : ℝ, -π/2 < x_0 ∧ x_0 < π/2 ∧ tan x_0 ≤ 0

-- Theorem stating that the negation of p is not_p
theorem negation_of_p : ¬ p ↔ not_p :=
sorry

end NUMINAMATH_GPT_negation_of_p_l2247_224770


namespace NUMINAMATH_GPT_common_ratio_l2247_224786

variable {a : ℕ → ℝ} -- Define a as a sequence of real numbers

-- Define the conditions as hypotheses
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n, a (n + 1) = q * a n

variables (q : ℝ) (h1 : a 2 = 2) (h2 : a 5 = 1 / 4)

-- Define the theorem to prove the common ratio
theorem common_ratio (h_geom : is_geometric_sequence a q) : q = 1 / 2 :=
  sorry

end NUMINAMATH_GPT_common_ratio_l2247_224786
