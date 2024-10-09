import Mathlib

namespace plates_used_l2407_240713

theorem plates_used (P : ℕ) (h : 3 * 2 * P + 4 * 8 = 38) : P = 1 := by
  sorry

end plates_used_l2407_240713


namespace probability_of_triangle_l2407_240746

/-- There are 12 figures in total: 4 squares, 5 triangles, and 3 rectangles.
    Prove that the probability of choosing a triangle is 5/12. -/
theorem probability_of_triangle (total_figures : ℕ) (num_squares : ℕ) (num_triangles : ℕ) (num_rectangles : ℕ)
  (h1 : total_figures = 12)
  (h2 : num_squares = 4)
  (h3 : num_triangles = 5)
  (h4 : num_rectangles = 3) :
  num_triangles / total_figures = 5 / 12 :=
sorry

end probability_of_triangle_l2407_240746


namespace parrots_left_l2407_240795

theorem parrots_left 
  (c : Nat)   -- The initial number of crows
  (x : Nat)   -- The number of parrots and crows that flew away
  (h1 : 7 + c = 13)          -- Initial total number of birds
  (h2 : c - x = 1)           -- Number of crows left
  : 7 - x = 2 :=             -- Number of parrots left
by
  sorry

end parrots_left_l2407_240795


namespace gyeonghun_climbing_l2407_240794

variable (t_up t_down d_up d_down : ℝ)
variable (h1 : t_up + t_down = 4) 
variable (h2 : d_down = d_up + 2)
variable (h3 : t_up = d_up / 3)
variable (h4 : t_down = d_down / 4)

theorem gyeonghun_climbing (h1 : t_up + t_down = 4) (h2 : d_down = d_up + 2) (h3 : t_up = d_up / 3) (h4 : t_down = d_down / 4) :
  t_up = 2 :=
by
  sorry

end gyeonghun_climbing_l2407_240794


namespace percentage_women_no_french_speak_spanish_german_l2407_240754

variable (total_workforce : Nat)
variable (men_percentage women_percentage : ℕ)
variable (men_only_french men_only_spanish men_only_german : ℕ)
variable (men_both_french_spanish men_both_french_german men_both_spanish_german : ℕ)
variable (men_all_three_languages women_only_french women_only_spanish : ℕ)
variable (women_only_german women_both_french_spanish women_both_french_german : ℕ)
variable (women_both_spanish_german women_all_three_languages : ℕ)

-- Conditions
axiom h1 : men_percentage = 60
axiom h2 : women_percentage = 40
axiom h3 : women_only_french = 30
axiom h4 : women_only_spanish = 25
axiom h5 : women_only_german = 20
axiom h6 : women_both_french_spanish = 10
axiom h7 : women_both_french_german = 5
axiom h8 : women_both_spanish_german = 5
axiom h9 : women_all_three_languages = 5

theorem percentage_women_no_french_speak_spanish_german:
  women_only_spanish + women_only_german + women_both_spanish_german = 50 := by
  sorry

end percentage_women_no_french_speak_spanish_german_l2407_240754


namespace nth_equation_l2407_240735

theorem nth_equation (n : ℕ) : 
  1 - (1 / ((n + 1)^2)) = (n / (n + 1)) * ((n + 2) / (n + 1)) :=
by sorry

end nth_equation_l2407_240735


namespace smallest_b_l2407_240789

noncomputable def f (a b x : ℝ) : ℝ := a * Real.log x + b / x - 2 * a

theorem smallest_b (a : ℝ) (b : ℝ) (x : ℝ) : (1 < a ∧ a < 4) → (0 < x) → (f a b x > 0) → b ≥ 11 :=
by
  -- placeholder for the proof
  sorry

end smallest_b_l2407_240789


namespace cucumbers_count_l2407_240799

theorem cucumbers_count (c : ℕ) (n : ℕ) (additional : ℕ) (initial_cucumbers : ℕ) (total_cucumbers : ℕ) :
  c = 4 → n = 10 → additional = 2 → initial_cucumbers = n - c → total_cucumbers = initial_cucumbers + additional → total_cucumbers = 8 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2] at h4
  simp at h4
  rw [h4, h3] at h5
  simp at h5
  exact h5

end cucumbers_count_l2407_240799


namespace closing_price_l2407_240766

theorem closing_price (opening_price : ℝ) (percent_increase : ℝ) (closing_price : ℝ) 
  (h₀ : opening_price = 6) (h₁ : percent_increase = 0.3333) : closing_price = 8 :=
by
  sorry

end closing_price_l2407_240766


namespace lindsay_dolls_l2407_240728

theorem lindsay_dolls (B B_b B_k : ℕ) 
  (h1 : B_b = 4 * B)
  (h2 : B_k = 4 * B - 2)
  (h3 : B_b + B_k = B + 26) : B = 4 :=
by
  sorry

end lindsay_dolls_l2407_240728


namespace bird_cages_count_l2407_240774

-- Definitions based on the conditions provided
def num_parrots_per_cage : ℕ := 2
def num_parakeets_per_cage : ℕ := 7
def total_birds_per_cage : ℕ := num_parrots_per_cage + num_parakeets_per_cage
def total_birds_in_store : ℕ := 54
def num_bird_cages : ℕ := total_birds_in_store / total_birds_per_cage

-- The proof we need to derive
theorem bird_cages_count : num_bird_cages = 6 := by
  sorry

end bird_cages_count_l2407_240774


namespace mrs_hilt_found_nickels_l2407_240771

theorem mrs_hilt_found_nickels : 
  ∀ (total cents quarter cents dime cents nickel cents : ℕ), 
    total = 45 → 
    quarter = 25 → 
    dime = 10 → 
    nickel = 5 → 
    ((total - (quarter + dime)) / nickel) = 2 := 
by
  intros total quarter dime nickel h_total h_quarter h_dime h_nickel
  sorry

end mrs_hilt_found_nickels_l2407_240771


namespace circle_center_and_radius_l2407_240725

-- Define the given conditions
variable (a : ℝ) (h : a^2 = a + 2 ∧ a ≠ 0)

-- Define the equation
noncomputable def circle_equation (x y : ℝ) : ℝ := a^2 * x^2 + (a + 2) * y^2 + 4 * x + 8 * y + 5 * a

-- Lean definition to represent the problem
theorem circle_center_and_radius :
  (∃a : ℝ, a ≠ 0 ∧ a^2 = a + 2 ∧
    (∃x y : ℝ, circle_equation a x y = 0) ∧
    ((a = -1) → ((∃x y : ℝ, (x + 2)^2 + (y + 4)^2 = 25) ∧
                 (center_x = -2) ∧ (center_y = -4) ∧ (radius = 5)))) :=
by
  sorry

end circle_center_and_radius_l2407_240725


namespace totalUniqueStudents_l2407_240757

-- Define the club memberships and overlap
variable (mathClub scienceClub artClub overlap : ℕ)

-- Conditions based on the problem
def mathClubSize : Prop := mathClub = 15
def scienceClubSize : Prop := scienceClub = 10
def artClubSize : Prop := artClub = 12
def overlapSize : Prop := overlap = 5

-- Main statement to prove
theorem totalUniqueStudents : 
  mathClubSize mathClub → 
  scienceClubSize scienceClub →
  artClubSize artClub →
  overlapSize overlap →
  mathClub + scienceClub + artClub - overlap = 32 := by
  intros
  sorry

end totalUniqueStudents_l2407_240757


namespace student_correct_answers_l2407_240772

-- Definitions based on the conditions
def total_questions : ℕ := 100
def score (correct incorrect : ℕ) : ℕ := correct - 2 * incorrect
def studentScore : ℕ := 73

-- Main theorem to prove
theorem student_correct_answers (C I : ℕ) (h1 : C + I = total_questions) (h2 : score C I = studentScore) : C = 91 :=
by
  sorry

end student_correct_answers_l2407_240772


namespace plot_length_l2407_240793

variable (b length : ℝ)

theorem plot_length (h1 : length = b + 10)
  (fence_N_cost : ℝ := 26.50 * (b + 10))
  (fence_E_cost : ℝ := 32 * b)
  (fence_S_cost : ℝ := 22 * (b + 10))
  (fence_W_cost : ℝ := 30 * b)
  (total_cost : ℝ := fence_N_cost + fence_E_cost + fence_S_cost + fence_W_cost)
  (h2 : 1.05 * total_cost = 7500) :
  length = 70.25 := by
  sorry

end plot_length_l2407_240793


namespace part_I_part_II_l2407_240740

noncomputable def f (x a : ℝ) : ℝ := |x - a| - 2 * |x - 1|

-- Part I
theorem part_I (x : ℝ) : (f x 3) ≥ 1 ↔ (0 ≤ x ∧ x ≤ 4 / 3) :=
by sorry

-- Part II
theorem part_II (a : ℝ) : (∀ x, 1 ≤ x ∧ x ≤ 2 → f x a - |2 * x - 5| ≤ 0) ↔ (-1 ≤ a ∧ a ≤ 4) :=
by sorry

end part_I_part_II_l2407_240740


namespace math_problem_l2407_240747

theorem math_problem (c d : ℝ) (hc : c^2 - 6 * c + 15 = 27) (hd : d^2 - 6 * d + 15 = 27) (h_cd : c ≥ d) : 
  3 * c + 2 * d = 15 + Real.sqrt 21 :=
by
  sorry

end math_problem_l2407_240747


namespace cost_per_load_is_25_cents_l2407_240701

-- Define the given conditions
def loads_per_bottle : ℕ := 80
def usual_price_per_bottle : ℕ := 2500 -- in cents
def sale_price_per_bottle : ℕ := 2000 -- in cents
def bottles_bought : ℕ := 2

-- Defining the total cost and total loads
def total_cost : ℕ := bottles_bought * sale_price_per_bottle
def total_loads : ℕ := bottles_bought * loads_per_bottle

-- Define the cost per load in cents
def cost_per_load_in_cents : ℕ := (total_cost * 100) / total_loads

-- Formal proof statement
theorem cost_per_load_is_25_cents 
    (h1 : loads_per_bottle = 80)
    (h2 : usual_price_per_bottle = 2500)
    (h3 : sale_price_per_bottle = 2000)
    (h4 : bottles_bought = 2)
    (h5 : total_cost = bottles_bought * sale_price_per_bottle)
    (h6 : total_loads = bottles_bought * loads_per_bottle)
    (h7 : cost_per_load_in_cents = (total_cost * 100) / total_loads):
  cost_per_load_in_cents = 25 := by
  sorry

end cost_per_load_is_25_cents_l2407_240701


namespace prob_next_black_ball_l2407_240732

theorem prob_next_black_ball
  (total_balls : ℕ := 100) 
  (black_balls : Fin 101) 
  (next_black_ball_probability : ℚ := 2 / 3) :
  black_balls.val ≤ total_balls →
  ∃ p q : ℕ, Nat.gcd p q = 1 ∧ (p : ℚ) / q = next_black_ball_probability ∧ p + q = 5 :=
by
  intros h
  use 2, 3
  repeat { sorry }

end prob_next_black_ball_l2407_240732


namespace mother_nickels_eq_two_l2407_240756

def initial_nickels : ℕ := 7
def dad_nickels : ℕ := 9
def total_nickels : ℕ := 18

theorem mother_nickels_eq_two : (total_nickels = initial_nickels + dad_nickels + 2) :=
by
  sorry

end mother_nickels_eq_two_l2407_240756


namespace intersection_lines_l2407_240733

theorem intersection_lines (c d : ℝ) :
    (∃ x y, x = (1/3) * y + c ∧ y = (1/3) * x + d ∧ x = 3 ∧ y = -1) →
    c + d = 4 / 3 :=
by
  sorry

end intersection_lines_l2407_240733


namespace max_value_x_plus_2y_l2407_240723

theorem max_value_x_plus_2y (x y : ℝ) (h : x^2 + 4 * y^2 - 2 * x * y = 4) :
  x + 2 * y ≤ 4 :=
sorry

end max_value_x_plus_2y_l2407_240723


namespace steve_and_laura_meet_time_l2407_240703

structure PathsOnParallelLines where
  steve_speed : ℝ
  laura_speed : ℝ
  path_separation : ℝ
  art_diameter : ℝ
  initial_distance_hidden : ℝ

def meet_time (p : PathsOnParallelLines) : ℝ :=
  sorry -- To be proven

-- Define the specific case for Steve and Laura
def steve_and_laura_paths : PathsOnParallelLines :=
  { steve_speed := 3,
    laura_speed := 1,
    path_separation := 240,
    art_diameter := 80,
    initial_distance_hidden := 230 }

theorem steve_and_laura_meet_time :
  meet_time steve_and_laura_paths = 45 :=
  sorry

end steve_and_laura_meet_time_l2407_240703


namespace odd_factor_form_l2407_240780

theorem odd_factor_form (n : ℕ) (x y : ℕ) (h_n : n > 0) (h_gcd : Nat.gcd x y = 1) :
  ∀ p, p ∣ (x ^ (2 ^ n) + y ^ (2 ^ n)) ∧ Odd p → ∃ k > 0, p = 2^(n+1) * k + 1 := 
by
  sorry

end odd_factor_form_l2407_240780


namespace find_a_perpendicular_lines_l2407_240704

theorem find_a_perpendicular_lines (a : ℝ) :
  (∀ (x y : ℝ),
    a * x + 2 * y + 6 = 0 → 
    x + (a - 1) * y + a^2 - 1 = 0 → (a * 1 + 2 * (a - 1) = 0)) → 
  a = 2/3 :=
by
  intros h
  sorry

end find_a_perpendicular_lines_l2407_240704


namespace total_cost_of_tires_and_battery_l2407_240750

theorem total_cost_of_tires_and_battery :
  (4 * 42 + 56 = 224) := 
  by
    sorry

end total_cost_of_tires_and_battery_l2407_240750


namespace dozens_of_golf_balls_l2407_240717

theorem dozens_of_golf_balls (total_balls : ℕ) (dozen_size : ℕ) (h1 : total_balls = 156) (h2 : dozen_size = 12) : total_balls / dozen_size = 13 :=
by
  have h_total : total_balls = 156 := h1
  have h_size : dozen_size = 12 := h2
  sorry

end dozens_of_golf_balls_l2407_240717


namespace lcm_10_to_30_l2407_240718

def list_of_ints := [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]

def lcm_of_list (l : List Nat) : Nat :=
  l.foldr Nat.lcm 1

theorem lcm_10_to_30 : lcm_of_list list_of_ints = 232792560 :=
  sorry

end lcm_10_to_30_l2407_240718


namespace tickets_sold_l2407_240796

def advanced_purchase_tickets := ℕ
def door_purchase_tickets := ℕ

variable (A D : ℕ)

theorem tickets_sold :
  (A + D = 140) →
  (8 * A + 14 * D = 1720) →
  A = 40 :=
by
  intros h1 h2
  sorry

end tickets_sold_l2407_240796


namespace z_rate_per_rupee_of_x_l2407_240765

-- Given conditions as definitions in Lean 4
def x_share := 1 -- x gets Rs. 1 for this proof
def y_rate_per_rupee_of_x := 0.45
def y_share := 27
def total_amount := 105

-- The statement to prove
theorem z_rate_per_rupee_of_x :
  (105 - (1 * 60) - 27) / 60 = 0.30 :=
by
  sorry

end z_rate_per_rupee_of_x_l2407_240765


namespace product_xyz_l2407_240727

theorem product_xyz (x y z : ℝ) (h1 : x + 1 / y = 2) (h2 : y + 1 / z = 2) (h3 : x + 1 / z = 3) : x * y * z = 2 := 
by sorry

end product_xyz_l2407_240727


namespace arithmetic_sqrt_9_l2407_240758

def arithmetic_sqrt (x : ℕ) : ℕ :=
  if h : 0 ≤ x then Nat.sqrt x else 0

theorem arithmetic_sqrt_9 : arithmetic_sqrt 9 = 3 :=
by {
  sorry
}

end arithmetic_sqrt_9_l2407_240758


namespace mod_mult_congruence_l2407_240745

theorem mod_mult_congruence (n : ℤ) (h1 : 215 ≡ 65 [ZMOD 75])
  (h2 : 789 ≡ 39 [ZMOD 75]) (h3 : 215 * 789 ≡ n [ZMOD 75]) (hn : 0 ≤ n ∧ n < 75) :
  n = 60 :=
by
  sorry

end mod_mult_congruence_l2407_240745


namespace cost_of_natural_seedless_raisins_l2407_240724

theorem cost_of_natural_seedless_raisins
  (cost_golden: ℝ) (n_golden: ℕ) (n_natural: ℕ) (cost_mixture: ℝ) (cost_per_natural: ℝ) :
  cost_golden = 2.55 ∧ n_golden = 20 ∧ n_natural = 20 ∧ cost_mixture = 3
  → cost_per_natural = 3.45 :=
by
  sorry

end cost_of_natural_seedless_raisins_l2407_240724


namespace intersection_of_sets_l2407_240788

def setA := { x : ℝ | x / (x - 1) < 0 }
def setB := { x : ℝ | 0 < x ∧ x < 3 }
def setIntersect := { x : ℝ | 0 < x ∧ x < 1 }

theorem intersection_of_sets :
  ∀ x : ℝ, x ∈ setA ∧ x ∈ setB ↔ x ∈ setIntersect := 
by
  sorry

end intersection_of_sets_l2407_240788


namespace compute_product_l2407_240760

variable (x1 y1 x2 y2 x3 y3 : ℝ)

def condition1 (x y : ℝ) : Prop := x^3 - 3 * x * y^2 = 2010
def condition2 (x y : ℝ) : Prop := y^3 - 3 * x^2 * y = 2000

theorem compute_product (h1 : condition1 x1 y1) (h2 : condition2 x1 y1)
    (h3 : condition1 x2 y2) (h4 : condition2 x2 y2)
    (h5 : condition1 x3 y3) (h6 : condition2 x3 y3) :
    (1 - x1 / y1) * (1 - x2 / y2) * (1 - x3 / y3) = 1 / 100 := 
    sorry

end compute_product_l2407_240760


namespace tiffany_optimal_area_l2407_240773

def optimal_area (A : ℕ) : Prop :=
  ∃ l w : ℕ, l + w = 160 ∧ l ≥ 85 ∧ w ≥ 45 ∧ A = l * w

theorem tiffany_optimal_area : optimal_area 6375 :=
  sorry

end tiffany_optimal_area_l2407_240773


namespace speed_second_half_l2407_240700

theorem speed_second_half (total_time : ℝ) (first_half_speed : ℝ) (total_distance : ℝ) :
    total_time = 12 → first_half_speed = 35 → total_distance = 560 → 
    (280 / (12 - (280 / 35)) = 70) :=
by
  intros ht hf hd
  sorry

end speed_second_half_l2407_240700


namespace boxes_with_neither_l2407_240786

def total_boxes : ℕ := 15
def boxes_with_stickers : ℕ := 9
def boxes_with_stamps : ℕ := 5
def boxes_with_both : ℕ := 3

theorem boxes_with_neither
  (total_boxes : ℕ)
  (boxes_with_stickers : ℕ)
  (boxes_with_stamps : ℕ)
  (boxes_with_both : ℕ) :
  total_boxes - ((boxes_with_stickers + boxes_with_stamps) - boxes_with_both) = 4 :=
by
  sorry

end boxes_with_neither_l2407_240786


namespace lock_settings_are_5040_l2407_240716

def num_unique_settings_for_lock : ℕ := 10 * 9 * 8 * 7

theorem lock_settings_are_5040 : num_unique_settings_for_lock = 5040 :=
by
  sorry

end lock_settings_are_5040_l2407_240716


namespace esther_evening_speed_l2407_240736

/-- Esther's average speed in the evening was 30 miles per hour -/
theorem esther_evening_speed : 
  let morning_speed := 45   -- miles per hour
  let total_commuting_time := 1 -- hour
  let morning_distance := 18  -- miles
  let evening_distance := 18  -- miles (same route)
  let time_morning := morning_distance / morning_speed
  let time_evening := total_commuting_time - time_morning
  let evening_speed := evening_distance / time_evening
  evening_speed = 30 := 
by sorry

end esther_evening_speed_l2407_240736


namespace find_m_l2407_240770

theorem find_m (n : ℝ) : 21 * (m + n) + 21 = 21 * (-m + n) + 21 → m = 0 :=
by
  sorry

end find_m_l2407_240770


namespace additional_people_needed_l2407_240779

-- Define the initial number of people and time they take to mow the lawn 
def initial_people : ℕ := 8
def initial_time : ℕ := 3

-- Define total person-hours required to mow the lawn
def total_person_hours : ℕ := initial_people * initial_time

-- Define the time in which we want to find out how many people can mow the lawn
def desired_time : ℕ := 2

-- Define the number of people needed in desired_time to mow the lawn
def required_people : ℕ := total_person_hours / desired_time

-- Define the additional people required to mow the lawn in desired_time
def additional_people : ℕ := required_people - initial_people

-- Statement to be proved
theorem additional_people_needed : additional_people = 4 := by
  -- Proof to be filled in
  sorry

end additional_people_needed_l2407_240779


namespace sqrt_expression_l2407_240785

theorem sqrt_expression (y : ℝ) (hy : y < 0) : 
  Real.sqrt (y / (1 - ((y - 2) / y))) = -y / Real.sqrt 2 := 
sorry

end sqrt_expression_l2407_240785


namespace fewest_printers_l2407_240730

theorem fewest_printers (x y : ℕ) (h : 8 * x = 7 * y) : x + y = 15 :=
sorry

end fewest_printers_l2407_240730


namespace inequality_proof_l2407_240744

theorem inequality_proof (a b c : ℝ) (k : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : k ≥ 1) : 
  (a^(k + 1) / b^k + b^(k + 1) / c^k + c^(k + 1) / a^k) ≥ (a^k / b^(k - 1) + b^k / c^(k - 1) + c^k / a^(k - 1)) :=
by
  sorry

end inequality_proof_l2407_240744


namespace employee_pays_correct_amount_l2407_240737

def wholesale_cost : ℝ := 200
def markup_percentage : ℝ := 0.20
def discount_percentage : ℝ := 0.10

def retail_price (wholesale: ℝ) (markup_percentage: ℝ) : ℝ :=
  wholesale * (1 + markup_percentage)

def discount_amount (price: ℝ) (discount_percentage: ℝ) : ℝ :=
  price * discount_percentage

def final_price (retail: ℝ) (discount: ℝ) : ℝ :=
  retail - discount

theorem employee_pays_correct_amount : final_price (retail_price wholesale_cost markup_percentage) 
                                                     (discount_amount (retail_price wholesale_cost markup_percentage) discount_percentage) = 216 := 
by
  sorry

end employee_pays_correct_amount_l2407_240737


namespace right_triangle_smaller_angle_l2407_240715

theorem right_triangle_smaller_angle (x : ℝ) (h_right_triangle : 0 < x ∧ x < 90)
  (h_double_angle : ∃ y : ℝ, y = 2 * x)
  (h_angle_sum : x + 2 * x = 90) :
  x = 30 :=
  sorry

end right_triangle_smaller_angle_l2407_240715


namespace f_2015_value_l2407_240751

def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

noncomputable def f : ℝ → ℝ := sorry

axiom f_is_odd : odd_function f
axiom f_periodicity : ∀ x : ℝ, f (x + 2) = -f x
axiom f_definition_in_interval : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → f x = 3^x - 1

theorem f_2015_value : f 2015 = -2 :=
by
  sorry

end f_2015_value_l2407_240751


namespace quadratic_root_l2407_240719

theorem quadratic_root (k : ℝ) (h : (1 : ℝ)^2 + k * 1 - 3 = 0) : k = 2 := 
sorry

end quadratic_root_l2407_240719


namespace cistern_filling_time_with_leak_l2407_240790

theorem cistern_filling_time_with_leak (T : ℝ) (h1 : 1 / T - 1 / 4 = 1 / (T + 2)) : T = 4 :=
by
  sorry

end cistern_filling_time_with_leak_l2407_240790


namespace probability_two_students_same_school_l2407_240741

/-- Definition of the problem conditions -/
def total_students : ℕ := 3
def total_schools : ℕ := 4
def total_basic_events : ℕ := total_schools ^ total_students
def favorable_events : ℕ := 36

/-- Theorem stating the probability of exactly two students choosing the same school -/
theorem probability_two_students_same_school : 
  favorable_events / (total_schools ^ total_students) = 9 / 16 := 
  sorry

end probability_two_students_same_school_l2407_240741


namespace A_knit_time_l2407_240712

def rate_A (x : ℕ) : ℚ := 1 / x
def rate_B : ℚ := 1 / 6

def combined_rate_two_pairs_in_4_days (x : ℕ) : Prop :=
  rate_A x + rate_B = 1 / 2

theorem A_knit_time : ∃ x : ℕ, combined_rate_two_pairs_in_4_days x ∧ x = 3 :=
by
  existsi 3
  -- (Formal proof would go here)
  sorry

end A_knit_time_l2407_240712


namespace range_of_expression_l2407_240764

theorem range_of_expression (a : ℝ) : (∃ a : ℝ, a + 1 ≥ 0 ∧ a - 2 ≠ 0) → (a ≥ -1 ∧ a ≠ 2) := 
by sorry

end range_of_expression_l2407_240764


namespace two_digit_number_conditions_l2407_240759

-- Definitions for two-digit number and its conditions
def is_two_digit_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def tens_digit (n : ℕ) : ℕ := n / 10
def units_digit (n : ℕ) : ℕ := n % 10
def sum_of_digits (n : ℕ) : ℕ := tens_digit n + units_digit n

-- The proof problem statement in Lean 4
theorem two_digit_number_conditions (N : ℕ) (c d : ℕ) :
  is_two_digit_number N ∧ N = 10 * c + d ∧ N' = N + 7 ∧ 
  N = 6 * sum_of_digits (N + 7) →
  N = 24 ∨ N = 78 :=
by
  sorry

end two_digit_number_conditions_l2407_240759


namespace least_positive_n_for_reducible_fraction_l2407_240752

theorem least_positive_n_for_reducible_fraction :
  ∃ n : ℕ, 0 < n ∧ (∃ k : ℤ, k > 1 ∧ k ∣ (n - 17) ∧ k ∣ (6 * n + 7)) ∧ n = 126 :=
by
  sorry

end least_positive_n_for_reducible_fraction_l2407_240752


namespace damaged_books_l2407_240706

theorem damaged_books (O D : ℕ) (h1 : O = 6 * D - 8) (h2 : D + O = 69) : D = 11 :=
by
  sorry

end damaged_books_l2407_240706


namespace third_quadrant_condition_l2407_240755

-- Define the conditions for the third quadrant
def in_third_quadrant (p: ℝ × ℝ) : Prop := p.1 < 0 ∧ p.2 < 0

-- Translate the problem statement to a Lean theorem
theorem third_quadrant_condition (a b : ℝ) (h1 : a + b < 0) (h2 : a * b > 0) : in_third_quadrant (a, b) :=
sorry

end third_quadrant_condition_l2407_240755


namespace weekly_goal_cans_l2407_240721

theorem weekly_goal_cans : (20 +  (20 * 1.5) + (20 * 2) + (20 * 2.5) + (20 * 3)) = 200 := by
  sorry

end weekly_goal_cans_l2407_240721


namespace electrical_bill_undetermined_l2407_240739

theorem electrical_bill_undetermined
    (gas_bill : ℝ)
    (gas_paid_fraction : ℝ)
    (additional_gas_payment : ℝ)
    (water_bill : ℝ)
    (water_paid_fraction : ℝ)
    (internet_bill : ℝ)
    (internet_payments : ℝ)
    (payment_amounts: ℝ)
    (total_remaining : ℝ) :
    gas_bill = 40 →
    gas_paid_fraction = 3 / 4 →
    additional_gas_payment = 5 →
    water_bill = 40 →
    water_paid_fraction = 1 / 2 →
    internet_bill = 25 →
    internet_payments = 4 * 5 →
    total_remaining = 30 →
    (∃ electricity_bill : ℝ, true) -> 
    false := by
  intro gas_bill_eq gas_paid_fraction_eq additional_gas_payment_eq
  intro water_bill_eq water_paid_fraction_eq
  intro internet_bill_eq internet_payments_eq 
  intro total_remaining_eq 
  intro exists_electricity_bill 
  sorry -- Proof that the electricity bill cannot be determined

end electrical_bill_undetermined_l2407_240739


namespace inequality_max_k_l2407_240734

theorem inequality_max_k (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (a + b + c) * (3^4 * (a + b + c + d)^5 + 2^4 * (a + b + c + 2 * d)^5) ≥ 174960 * a * b * c * d^3 :=
sorry

end inequality_max_k_l2407_240734


namespace sin_cos_product_l2407_240777

theorem sin_cos_product (x : ℝ) (h : Real.sin x = 5 * Real.cos x) : Real.sin x * Real.cos x = 5 / 26 := by
  sorry

end sin_cos_product_l2407_240777


namespace valid_combinations_count_l2407_240761

theorem valid_combinations_count : 
  let wrapping_paper_count := 10
  let ribbon_count := 3
  let gift_card_count := 5
  let invalid_combinations := 1 -- red ribbon with birthday card
  let total_combinations := wrapping_paper_count * ribbon_count * gift_card_count
  total_combinations - invalid_combinations = 149 := 
by 
  sorry

end valid_combinations_count_l2407_240761


namespace problem1_problem2_l2407_240782

-- Definition for the first problem
def isPerfectSquare (m : ℕ) : Prop := ∃ k : ℕ, k * k = m

-- First Lean 4 statement for 2^n + 3 = x^2
theorem problem1 (n : ℕ) (h : isPerfectSquare (2^n + 3)) : n = 0 :=
sorry

-- Second Lean 4 statement for 2^n + 1 = x^2
theorem problem2 (n : ℕ) (h : isPerfectSquare (2^n + 1)) : n = 3 :=
sorry

end problem1_problem2_l2407_240782


namespace find_p_current_age_l2407_240702

theorem find_p_current_age (x p q : ℕ) (h1 : p - 3 = 4 * x) (h2 : q - 3 = 3 * x) (h3 : (p + 6) / (q + 6) = 7 / 6) : p = 15 := 
sorry

end find_p_current_age_l2407_240702


namespace sandwich_cost_is_five_l2407_240722

-- Define the cost of each sandwich
variables (x : ℝ)

-- Conditions
def jack_orders_sandwiches (cost_per_sandwich : ℝ) : Prop :=
  3 * cost_per_sandwich = 15

-- Proof problem statement (no proof provided)
theorem sandwich_cost_is_five (h : jack_orders_sandwiches x) : x = 5 :=
sorry

end sandwich_cost_is_five_l2407_240722


namespace x_gt_1_sufficient_but_not_necessary_for_abs_x_gt_1_l2407_240749

theorem x_gt_1_sufficient_but_not_necessary_for_abs_x_gt_1 (x : ℝ) : (x > 1 → |x| > 1) ∧ (¬(x > 1 ↔ |x| > 1)) :=
by
  sorry

end x_gt_1_sufficient_but_not_necessary_for_abs_x_gt_1_l2407_240749


namespace group_interval_eq_l2407_240748

noncomputable def group_interval (a b m h : ℝ) : ℝ := abs (a - b)

theorem group_interval_eq (a b m h : ℝ) 
  (h1 : h = m / abs (a - b)) :
  abs (a - b) = m / h := 
by 
  sorry

end group_interval_eq_l2407_240748


namespace coupons_per_coloring_book_l2407_240705

theorem coupons_per_coloring_book 
  (initial_books : ℝ) (books_sold : ℝ) (coupons_used : ℝ)
  (h1 : initial_books = 40) (h2 : books_sold = 20) (h3 : coupons_used = 80) : 
  (coupons_used / (initial_books - books_sold) = 4) :=
by 
  simp [*, sub_eq_add_neg]
  sorry

end coupons_per_coloring_book_l2407_240705


namespace leak_emptying_time_l2407_240708

-- Definitions based on given conditions
def tank_fill_rate_without_leak : ℚ := 1 / 3
def combined_fill_and_leak_rate : ℚ := 1 / 4

-- Leak emptying time to be proven
theorem leak_emptying_time (R : ℚ := tank_fill_rate_without_leak) (C : ℚ := combined_fill_and_leak_rate) :
  (1 : ℚ) / (R - C) = 12 := by
  sorry

end leak_emptying_time_l2407_240708


namespace running_problem_l2407_240797

variables (x y : ℝ)

theorem running_problem :
  (5 * x = 5 * y + 10) ∧ (4 * x = 4 * y + 2 * y) :=
by
  sorry

end running_problem_l2407_240797


namespace circle_intersection_zero_l2407_240791

theorem circle_intersection_zero :
  (∀ θ : ℝ, ∀ r1 : ℝ, r1 = 3 * Real.cos θ → ∀ r2 : ℝ, r2 = 6 * Real.sin (2 * θ) → False) :=
by 
  sorry

end circle_intersection_zero_l2407_240791


namespace division_remainder_l2407_240763

/-- The remainder when 3572 is divided by 49 is 44. -/
theorem division_remainder :
  3572 % 49 = 44 :=
by
  sorry

end division_remainder_l2407_240763


namespace original_number_l2407_240769

theorem original_number (x : ℕ) (h : x / 3 = 42) : x = 126 :=
sorry

end original_number_l2407_240769


namespace scale_length_discrepancy_l2407_240768

theorem scale_length_discrepancy
  (scale_length_feet : ℝ)
  (parts : ℕ)
  (part_length_inches : ℝ)
  (ft_to_inch : ℝ := 12)
  (total_length_inches : ℝ := parts * part_length_inches)
  (scale_length_inches : ℝ := scale_length_feet * ft_to_inch) :
  scale_length_feet = 7 → 
  parts = 4 → 
  part_length_inches = 24 →
  total_length_inches - scale_length_inches = 12 := by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end scale_length_discrepancy_l2407_240768


namespace toll_booth_ratio_l2407_240743

theorem toll_booth_ratio (total_cars : ℕ) (monday_cars tuesday_cars friday_cars saturday_cars sunday_cars : ℕ)
  (x : ℕ) (h1 : total_cars = 450) (h2 : monday_cars = 50) (h3 : tuesday_cars = 50) (h4 : friday_cars = 50)
  (h5 : saturday_cars = 50) (h6 : sunday_cars = 50) (h7 : monday_cars + tuesday_cars + x + x + friday_cars + saturday_cars + sunday_cars = total_cars) :
  x = 100 ∧ x / monday_cars = 2 :=
by
  sorry

end toll_booth_ratio_l2407_240743


namespace opposite_of_4_l2407_240783

theorem opposite_of_4 : ∃ x, 4 + x = 0 ∧ x = -4 :=
by sorry

end opposite_of_4_l2407_240783


namespace average_speed_l2407_240776

-- Define the problem conditions and provide the proof statement
theorem average_speed (D : ℝ) (hD0 : D > 0) : 
  let speed_1 := 80
  let speed_2 := 24
  let speed_3 := 60
  let time_1 := (D / 3) / speed_1
  let time_2 := (D / 3) / speed_2
  let time_3 := (D / 3) / speed_3
  let total_time := time_1 + time_2 + time_3
  let average_speed := D / total_time
  average_speed = 720 / 17 := 
by
  sorry

end average_speed_l2407_240776


namespace quadratic_inequality_solution_l2407_240731

theorem quadratic_inequality_solution (a : ℝ) :
  ((0 ≤ a ∧ a < 3) → ∀ x : ℝ, a * x^2 - 2 * a * x + 3 > 0) :=
  sorry

end quadratic_inequality_solution_l2407_240731


namespace added_water_is_18_l2407_240792

def capacity : ℕ := 40

def initial_full_percent : ℚ := 0.30

def final_full_fraction : ℚ := 3/4

def initial_water (capacity : ℕ) (initial_full_percent : ℚ) : ℚ :=
  initial_full_percent * capacity

def final_water (capacity : ℕ) (final_full_fraction : ℚ) : ℚ :=
  final_full_fraction * capacity

def water_added (initial_water : ℚ) (final_water : ℚ) : ℚ :=
  final_water - initial_water

theorem added_water_is_18 :
  water_added (initial_water capacity initial_full_percent) (final_water capacity final_full_fraction) = 18 := by
  sorry

end added_water_is_18_l2407_240792


namespace next_term_geometric_sequence_l2407_240784

theorem next_term_geometric_sequence (x : ℝ) (r : ℝ) (a₀ a₃ next_term : ℝ)
    (h1 : a₀ = 2)
    (h2 : r = 3 * x)
    (h3 : a₃ = 54 * x^3)
    (h4 : next_term = a₃ * r) :
    next_term = 162 * x^4 := by
  sorry

end next_term_geometric_sequence_l2407_240784


namespace bridget_bakery_profit_l2407_240738

theorem bridget_bakery_profit :
  let loaves := 36
  let cost_per_loaf := 1
  let morning_sale_price := 3
  let afternoon_sale_price := 1.5
  let late_afternoon_sale_price := 1
  
  let morning_loaves := (2/3 : ℝ) * loaves
  let morning_revenue := morning_loaves * morning_sale_price
  
  let remaining_after_morning := loaves - morning_loaves
  let afternoon_loaves := (1/2 : ℝ) * remaining_after_morning
  let afternoon_revenue := afternoon_loaves * afternoon_sale_price
  
  let late_afternoon_loaves := remaining_after_morning - afternoon_loaves
  let late_afternoon_revenue := late_afternoon_loaves * late_afternoon_sale_price
  
  let total_revenue := morning_revenue + afternoon_revenue + late_afternoon_revenue
  let total_cost := loaves * cost_per_loaf
  
  total_revenue - total_cost = 51 := by sorry

end bridget_bakery_profit_l2407_240738


namespace volume_of_cuboctahedron_l2407_240767

def points (i j : ℕ) (A : ℕ → ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x0, y0, z0) := A 0
  let (xi, yi, zi) := A i
  let (xj, yj, zj) := A j
  (xi - xj, yi - yj, zi - zj)

def is_cuboctahedron (points_set : Set (ℝ × ℝ × ℝ)) : Prop :=
  -- Insert specific conditions that define a cuboctahedron
  sorry

theorem volume_of_cuboctahedron : 
  let A := fun 
    | 0 => (0, 0, 0)
    | 1 => (1, 0, 0)
    | 2 => (0, 1, 0)
    | 3 => (0, 0, 1)
    | _ => (0, 0, 0)
  let P_ij := 
    {p | ∃ i j : ℕ, i ≠ j ∧ p = points i j A}
  ∃ v : ℝ, is_cuboctahedron P_ij ∧ v = 10 / 3 :=
sorry

end volume_of_cuboctahedron_l2407_240767


namespace square_area_side4_l2407_240709

theorem square_area_side4
  (s : ℕ)
  (A : ℕ)
  (P : ℕ)
  (h_s : s = 4)
  (h_A : A = s * s)
  (h_P : P = 4 * s)
  (h_eqn : (A + s) - P = 4) : A = 16 := sorry

end square_area_side4_l2407_240709


namespace faster_train_speed_l2407_240753

theorem faster_train_speed (dist_between_stations : ℕ) (extra_distance : ℕ) (slower_speed : ℕ) 
  (dist_between_stations_eq : dist_between_stations = 444)
  (extra_distance_eq : extra_distance = 60) 
  (slower_speed_eq : slower_speed = 16) :
  ∃ (faster_speed : ℕ), faster_speed = 21 := by
  sorry

end faster_train_speed_l2407_240753


namespace skyscraper_anniversary_l2407_240778

theorem skyscraper_anniversary 
  (years_since_built : ℕ)
  (target_years : ℕ)
  (years_before_200th : ℕ)
  (years_future : ℕ) 
  (h1 : years_since_built = 100) 
  (h2 : target_years = 200 - 5) 
  (h3 : years_future = target_years - years_since_built) : 
  years_future = 95 :=
by
  sorry

end skyscraper_anniversary_l2407_240778


namespace sqrt_D_irrational_l2407_240714

open Real

theorem sqrt_D_irrational (a : ℤ) (D : ℝ) (hD : D = a^2 + (a + 2)^2 + (a^2 + (a + 2))^2) : ¬ ∃ m : ℤ, D = m^2 :=
by
  sorry

end sqrt_D_irrational_l2407_240714


namespace certain_number_eq_40_l2407_240775

theorem certain_number_eq_40 (x : ℝ) 
    (h : (20 + x + 60) / 3 = (20 + 60 + 25) / 3 + 5) : x = 40 := 
by
  sorry

end certain_number_eq_40_l2407_240775


namespace shirts_before_buying_l2407_240711

-- Define the conditions
variable (new_shirts : ℕ)
variable (total_shirts : ℕ)

-- Define the statement where we need to prove the number of shirts Sarah had before buying the new ones
theorem shirts_before_buying (h₁ : new_shirts = 8) (h₂ : total_shirts = 17) : total_shirts - new_shirts = 9 :=
by
  -- Proof goes here
  sorry

end shirts_before_buying_l2407_240711


namespace john_bought_metres_l2407_240707

-- Define the conditions
def total_cost := 425.50
def cost_per_metre := 46.00

-- State the theorem
theorem john_bought_metres : total_cost / cost_per_metre = 9.25 :=
by
  sorry

end john_bought_metres_l2407_240707


namespace maximize_x2y5_l2407_240742

theorem maximize_x2y5 (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + y = 50) : 
  x = 100 / 7 ∧ y = 250 / 7 :=
sorry

end maximize_x2y5_l2407_240742


namespace center_of_hyperbola_l2407_240787

-- Define the given equation of the hyperbola
def hyperbola_eq (x y : ℝ) : Prop :=
  ((3 * y - 6)^2 / 8^2) - ((4 * x - 5)^2 / 3^2) = 1

-- Prove that the center of the hyperbola is (5 / 4, 2)
theorem center_of_hyperbola :
  (∃ h k : ℝ, h = 5 / 4 ∧ k = 2 ∧ ∀ x y : ℝ, hyperbola_eq x y ↔ ((y - k)^2 / (8 / 3)^2 - (x - h)^2 / (3 / 4)^2 = 1)) :=
sorry

end center_of_hyperbola_l2407_240787


namespace min_side_length_l2407_240710

def table_diagonal (w h : ℕ) : ℕ :=
  Nat.sqrt (w * w + h * h)

theorem min_side_length (w h : ℕ) (S : ℕ) (dw : w = 9) (dh : h = 12) (dS : S = 15) :
  S >= table_diagonal w h :=
by
  sorry

end min_side_length_l2407_240710


namespace find_marks_in_chemistry_l2407_240720

theorem find_marks_in_chemistry
  (marks_english : ℕ)
  (marks_math : ℕ)
  (marks_physics : ℕ)
  (marks_biology : ℕ)
  (average_marks : ℕ)
  (num_subjects : ℕ)
  (marks_english_eq : marks_english = 86)
  (marks_math_eq : marks_math = 85)
  (marks_physics_eq : marks_physics = 92)
  (marks_biology_eq : marks_biology = 95)
  (average_marks_eq : average_marks = 89)
  (num_subjects_eq : num_subjects = 5) : 
  ∃ marks_chemistry : ℕ, marks_chemistry = 87 :=
by
  sorry

end find_marks_in_chemistry_l2407_240720


namespace michael_birth_year_l2407_240729

theorem michael_birth_year (first_imo_year : ℕ) (annual_event : ∀ n : ℕ, n > 0 → (first_imo_year + n) ≥ first_imo_year) 
  (michael_age_at_10th_imo : ℕ) (imo_count : ℕ) 
  (H1 : first_imo_year = 1959) (H2 : imo_count = 10) (H3 : michael_age_at_10th_imo = 15) : 
  (first_imo_year + imo_count - 1 - michael_age_at_10th_imo = 1953) := 
by 
  sorry

end michael_birth_year_l2407_240729


namespace picture_size_l2407_240726

theorem picture_size (total_pics_A : ℕ) (size_A : ℕ) (total_pics_B : ℕ) (C : ℕ)
  (hA : total_pics_A * size_A = C) (hB : total_pics_B = 3000) : 
  (C / total_pics_B = 8) :=
by
  sorry

end picture_size_l2407_240726


namespace solve_for_x_l2407_240781

theorem solve_for_x :
  ∃ x : ℝ, 40 + (5 * x) / (180 / 3) = 41 ∧ x = 12 :=
by
  sorry

end solve_for_x_l2407_240781


namespace perpendicular_lines_l2407_240762

theorem perpendicular_lines :
  ∃ y x : ℝ, (4 * y - 3 * x = 15) ∧ (3 * y + 4 * x = 12) :=
by
  sorry

end perpendicular_lines_l2407_240762


namespace final_price_is_correct_l2407_240798

/-- 
  The original price of a suit is $200.
-/
def original_price : ℝ := 200

/-- 
  The price increased by 25%, therefore the increase is 25% of the original price.
-/
def increase : ℝ := 0.25 * original_price

/-- 
  The new price after the price increase.
-/
def increased_price : ℝ := original_price + increase

/-- 
  After the increase, a 25% off coupon is applied.
-/
def discount : ℝ := 0.25 * increased_price

/-- 
  The final price consumers pay for the suit.
-/
def final_price : ℝ := increased_price - discount

/-- 
  Prove that the consumers paid $187.50 for the suit.
-/
theorem final_price_is_correct : final_price = 187.50 :=
by sorry

end final_price_is_correct_l2407_240798
