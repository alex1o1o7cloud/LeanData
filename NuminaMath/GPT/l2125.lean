import Mathlib

namespace NUMINAMATH_GPT_minimum_monkeys_required_l2125_212565

theorem minimum_monkeys_required (total_weight : ℕ) (weapon_max_weight : ℕ) (monkey_max_capacity : ℕ) 
  (num_monkeys : ℕ) (total_weapons : ℕ) 
  (H1 : total_weight = 600) 
  (H2 : weapon_max_weight = 30) 
  (H3 : monkey_max_capacity = 50) 
  (H4 : total_weapons = 600 / 30) 
  (H5 : num_monkeys = 23) : 
  num_monkeys ≤ (total_weapons * weapon_max_weight) / monkey_max_capacity :=
sorry

end NUMINAMATH_GPT_minimum_monkeys_required_l2125_212565


namespace NUMINAMATH_GPT_sum_of_x_y_possible_values_l2125_212566

theorem sum_of_x_y_possible_values (x y : ℝ) (h : x^3 + 21 * x * y + y^3 = 343) : x + y = 7 ∨ x + y = -14 := 
sorry

end NUMINAMATH_GPT_sum_of_x_y_possible_values_l2125_212566


namespace NUMINAMATH_GPT_min_time_to_pass_l2125_212518

noncomputable def tunnel_length : ℝ := 2150
noncomputable def num_vehicles : ℝ := 55
noncomputable def vehicle_length : ℝ := 10
noncomputable def speed_limit : ℝ := 20
noncomputable def max_speed : ℝ := 40

noncomputable def distance_between_vehicles (x : ℝ) : ℝ :=
if 0 < x ∧ x ≤ 10 then 20 else
if 10 < x ∧ x ≤ 20 then (1/6) * x ^ 2 + (1/3) * x else
0

noncomputable def time_to_pass_through_tunnel (x : ℝ) : ℝ :=
if 0 < x ∧ x ≤ 10 then (2150 + 10 * 55 + 20 * (55 - 1)) / x else
if 10 < x ∧ x ≤ 20 then (2150 + 10 * 55 + ((1/6) * x^2 + (1/3) * x) * (55 - 1)) / x + 9 * x + 18 else
0

theorem min_time_to_pass : ∃ x : ℝ, (10 < x ∧ x ≤ 20) ∧ x = 17.3 ∧ time_to_pass_through_tunnel x = 329.4 :=
sorry

end NUMINAMATH_GPT_min_time_to_pass_l2125_212518


namespace NUMINAMATH_GPT_Joan_video_game_expense_l2125_212584

theorem Joan_video_game_expense : 
  let basketball_price := 5.20
  let racing_price := 4.23
  let action_price := 7.12
  let discount_rate := 0.10
  let sales_tax_rate := 0.06
  let discounted_basketball_price := basketball_price * (1 - discount_rate)
  let discounted_racing_price := racing_price * (1 - discount_rate)
  let discounted_action_price := action_price * (1 - discount_rate)
  let total_cost_before_tax := discounted_basketball_price + discounted_racing_price + discounted_action_price
  let sales_tax := total_cost_before_tax * sales_tax_rate
  let total_cost := total_cost_before_tax + sales_tax
  total_cost = 15.79 :=
by
  sorry

end NUMINAMATH_GPT_Joan_video_game_expense_l2125_212584


namespace NUMINAMATH_GPT_parabola_vertex_coordinates_l2125_212535

theorem parabola_vertex_coordinates :
  ∀ x : ℝ, -x^2 + 15 ≥ -x^2 + 15 :=
by
  sorry

end NUMINAMATH_GPT_parabola_vertex_coordinates_l2125_212535


namespace NUMINAMATH_GPT_parallelogram_sides_l2125_212582

theorem parallelogram_sides (x y : ℝ) 
  (h1 : 5 * x - 7 = 14) 
  (h2 : 3 * y + 4 = 8 * y - 3) : 
  x + y = 5.6 :=
sorry

end NUMINAMATH_GPT_parallelogram_sides_l2125_212582


namespace NUMINAMATH_GPT_reliefSuppliesCalculation_l2125_212548

noncomputable def totalReliefSupplies : ℝ := 644

theorem reliefSuppliesCalculation
    (A_capacity : ℝ)
    (B_capacity : ℝ)
    (A_capacity_per_day : A_capacity = 64.4)
    (capacity_ratio : A_capacity = 1.75 * B_capacity)
    (additional_transport : ∃ t : ℝ, A_capacity * t - B_capacity * t = 138 ∧ A_capacity * t = 322) :
  totalReliefSupplies = 644 := by
  sorry

end NUMINAMATH_GPT_reliefSuppliesCalculation_l2125_212548


namespace NUMINAMATH_GPT_lines_intersection_l2125_212593

theorem lines_intersection (n c : ℝ) : 
    (∀ x y : ℝ, y = n * x + 5 → y = 4 * x + c → (x, y) = (8, 9)) → 
    n + c = -22.5 := 
by
    intro h
    sorry

end NUMINAMATH_GPT_lines_intersection_l2125_212593


namespace NUMINAMATH_GPT_line_equation_l2125_212522

theorem line_equation (x y : ℝ) (h : (2, 3) ∈ {p : ℝ × ℝ | (∃ a, p.1 + p.2 = a) ∨ (∃ k, p.2 = k * p.1)}) :
  (3 * x - 2 * y = 0) ∨ (x + y - 5 = 0) :=
sorry

end NUMINAMATH_GPT_line_equation_l2125_212522


namespace NUMINAMATH_GPT_pyramid_top_value_l2125_212539

theorem pyramid_top_value 
  (p : ℕ) (q : ℕ) (z : ℕ) (m : ℕ) (n : ℕ) (left_mid : ℕ) (right_mid : ℕ) 
  (left_upper : ℕ) (right_upper : ℕ) (x_pre : ℕ) (x : ℕ) : 
  p = 20 → 
  q = 6 → 
  z = 44 → 
  m = p + 34 → 
  n = q + z → 
  left_mid = 17 + 29 → 
  right_mid = m + n → 
  left_upper = 36 + left_mid → 
  right_upper = right_mid + 42 → 
  x_pre = left_upper + 78 → 
  x = 2 * x_pre → 
  x = 320 :=
by
  intros
  sorry

end NUMINAMATH_GPT_pyramid_top_value_l2125_212539


namespace NUMINAMATH_GPT_minimum_f_l2125_212551

def f (x : ℝ) : ℝ := |3 - x| + |x - 7|

theorem minimum_f : ∀ x : ℝ, min (f x) = 4 := sorry

end NUMINAMATH_GPT_minimum_f_l2125_212551


namespace NUMINAMATH_GPT_crushing_load_value_l2125_212543

-- Define the given formula and values
def T : ℕ := 3
def H : ℕ := 9
def K : ℕ := 2

-- Given formula for L
def L (T H K : ℕ) : ℚ := 50 * T^5 / (K * H^3)

-- Prove that L = 8 + 1/3
theorem crushing_load_value :
  L T H K = 8 + 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_crushing_load_value_l2125_212543


namespace NUMINAMATH_GPT_quadratic_rational_solutions_product_l2125_212572

theorem quadratic_rational_solutions_product :
  ∃ (c₁ c₂ : ℕ), (7 * x^2 + 15 * x + c₁ = 0 ∧ 225 - 28 * c₁ = k^2 ∧ ∃ k : ℤ, k^2 = 225 - 28 * c₁) ∧
                 (7 * x^2 + 15 * x + c₂ = 0 ∧ 225 - 28 * c₂ = k^2 ∧ ∃ k : ℤ, k^2 = 225 - 28 * c₂) ∧
                 (c₁ = 1) ∧ (c₂ = 8) ∧ (c₁ * c₂ = 8) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_rational_solutions_product_l2125_212572


namespace NUMINAMATH_GPT_total_people_in_line_l2125_212546

theorem total_people_in_line (n : ℕ) (h : n = 5): n + 2 = 7 :=
by
  -- This is where the proof would normally go, but we omit it with "sorry"
  sorry

end NUMINAMATH_GPT_total_people_in_line_l2125_212546


namespace NUMINAMATH_GPT_apps_more_than_files_l2125_212509

theorem apps_more_than_files
  (initial_apps : ℕ)
  (initial_files : ℕ)
  (deleted_apps : ℕ)
  (deleted_files : ℕ)
  (remaining_apps : ℕ)
  (remaining_files : ℕ)
  (h1 : initial_apps - deleted_apps = remaining_apps)
  (h2 : initial_files - deleted_files = remaining_files)
  (h3 : initial_apps = 24)
  (h4 : initial_files = 9)
  (h5 : remaining_apps = 12)
  (h6 : remaining_files = 5) :
  remaining_apps - remaining_files = 7 :=
by {
  sorry
}

end NUMINAMATH_GPT_apps_more_than_files_l2125_212509


namespace NUMINAMATH_GPT_max_bicycle_distance_l2125_212536

-- Define the properties of the tires
def front_tire_duration : ℕ := 5000
def rear_tire_duration : ℕ := 3000

-- Define the maximum distance the bicycle can travel
def max_distance : ℕ := 3750

-- The main statement to be proven (proof is not required)
theorem max_bicycle_distance 
  (swap_usage : ∀ (d1 d2 : ℕ), d1 + d2 <= front_tire_duration + rear_tire_duration) : 
  ∃ (x : ℕ), x = max_distance := 
sorry

end NUMINAMATH_GPT_max_bicycle_distance_l2125_212536


namespace NUMINAMATH_GPT_solve_for_A_l2125_212504

theorem solve_for_A : 
  ∃ (A B : ℕ), (100 * A + 78) - (200 + 10 * B + 4) = 364 → A = 5 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_A_l2125_212504


namespace NUMINAMATH_GPT_part_I_part_II_l2125_212574

def sequence_sn (n : ℕ) : ℚ := (3 / 2 : ℚ) * n^2 + (1 / 2 : ℚ) * n

def sequence_a (n : ℕ) : ℕ := 3 * n - 1

def sequence_b (n : ℕ) : ℚ := (1 / 2 : ℚ)^n

def sequence_C (n : ℕ) : ℚ := sequence_a (sequence_a n) + sequence_b (sequence_a n)

def sum_of_first_n_terms (f : ℕ → ℚ) (n : ℕ) : ℚ :=
  (Finset.range n).sum f

theorem part_I (n : ℕ) : sequence_a n = 3 * n - 1 ∧ sequence_b n = (1 / 2)^n :=
by {
  sorry
}

theorem part_II (n : ℕ) : sum_of_first_n_terms sequence_C n =
  (n * (9 * n + 1) / 2) - (2 / 7) * (1 / 8)^n + (2 / 7) :=
by {
  sorry
}

end NUMINAMATH_GPT_part_I_part_II_l2125_212574


namespace NUMINAMATH_GPT_b_eq_6_l2125_212544

theorem b_eq_6 (a b : ℤ) (h₁ : |a| = 1) (h₂ : ∀ x : ℝ, a * x^2 - 2 * x - b + 5 = 0 → x < 0) : b = 6 := 
by
  sorry

end NUMINAMATH_GPT_b_eq_6_l2125_212544


namespace NUMINAMATH_GPT_probability_one_no_GP_l2125_212568

def num_pies : ℕ := 6
def growth_pies : ℕ := 2
def shrink_pies : ℕ := 4
def picked_pies : ℕ := 3
def total_outcomes : ℕ := Nat.choose num_pies picked_pies

def fav_outcomes : ℕ := Nat.choose shrink_pies 2 -- Choosing 2 out of the 4 SP

def probability_complementary : ℚ := fav_outcomes / total_outcomes
def probability : ℚ := 1 - probability_complementary

theorem probability_one_no_GP :
  probability = 0.4 := by
  sorry

end NUMINAMATH_GPT_probability_one_no_GP_l2125_212568


namespace NUMINAMATH_GPT_number_of_children_l2125_212560

theorem number_of_children (C B : ℕ) (h1 : B = 2 * C) (h2 : B = 4 * (C - 390)) : C = 780 :=
by
  sorry

end NUMINAMATH_GPT_number_of_children_l2125_212560


namespace NUMINAMATH_GPT_trains_cross_time_l2125_212564

noncomputable def timeToCross (length1 length2 speed1 speed2 : ℝ) : ℝ :=
  let speed1_mps := speed1 * (5 / 18)
  let speed2_mps := speed2 * (5 / 18)
  let relative_speed := speed1_mps + speed2_mps
  let total_length := length1 + length2
  total_length / relative_speed

theorem trains_cross_time
  (length1 length2 : ℝ)
  (speed1 speed2 : ℝ)
  (h_length1 : length1 = 250)
  (h_length2 : length2 = 250)
  (h_speed1 : speed1 = 90)
  (h_speed2 : speed2 = 110) :
  timeToCross length1 length2 speed1 speed2 = 9 := 
by sorry

end NUMINAMATH_GPT_trains_cross_time_l2125_212564


namespace NUMINAMATH_GPT_balloons_popped_on_ground_l2125_212554

def max_rate : Nat := 2
def max_time : Nat := 30
def zach_rate : Nat := 3
def zach_time : Nat := 40
def total_filled_balloons : Nat := 170

theorem balloons_popped_on_ground :
  (max_rate * max_time + zach_rate * zach_time) - total_filled_balloons = 10 :=
by
  sorry

end NUMINAMATH_GPT_balloons_popped_on_ground_l2125_212554


namespace NUMINAMATH_GPT_model_to_statue_ratio_inch_per_feet_model_inches_for_statue_feet_l2125_212526

theorem model_to_statue_ratio_inch_per_feet (statue_height_ft : ℝ) (model_height_in : ℝ) :
  statue_height_ft = 120 → model_height_in = 6 → (120 / 6 = 20)
:= by
  intros h1 h2
  sorry

theorem model_inches_for_statue_feet (model_per_inch_feet : ℝ) :
  model_per_inch_feet = 20 → (30 / 20 = 1.5)
:= by
  intros h
  sorry

end NUMINAMATH_GPT_model_to_statue_ratio_inch_per_feet_model_inches_for_statue_feet_l2125_212526


namespace NUMINAMATH_GPT_max_abcsum_l2125_212598

theorem max_abcsum (a b c : ℕ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) (h_eq : a * b^2 * c^3 = 1350) : 
  a + b + c ≤ 154 :=
sorry

end NUMINAMATH_GPT_max_abcsum_l2125_212598


namespace NUMINAMATH_GPT_max_a_condition_range_a_condition_l2125_212514

-- Definitions of the functions f and g
def f (x a : ℝ) : ℝ := |2 * x - a| + a
def g (x : ℝ) : ℝ := |2 * x - 1|

-- Problem (I)
theorem max_a_condition (a : ℝ) :
  (∀ x, g x ≤ 5 → f x a ≤ 6) → a ≤ 1 :=
sorry

-- Problem (II)
theorem range_a_condition (a : ℝ) :
  (∀ x, f x a + g x ≥ 3) → a ≥ 2 :=
sorry

end NUMINAMATH_GPT_max_a_condition_range_a_condition_l2125_212514


namespace NUMINAMATH_GPT_find_xy_l2125_212586

noncomputable def xy_value (x y : ℝ) := x * y

theorem find_xy :
  ∃ x y : ℝ, (x + y = 2) ∧ (x^2 * y^3 + y^2 * x^3 = 32) ∧ xy_value x y = -8 :=
by
  sorry

end NUMINAMATH_GPT_find_xy_l2125_212586


namespace NUMINAMATH_GPT_road_width_l2125_212591

theorem road_width
  (road_length : ℝ) 
  (truckload_area : ℝ) 
  (truckload_cost : ℝ) 
  (sales_tax : ℝ) 
  (total_cost : ℝ) :
  road_length = 2000 ∧
  truckload_area = 800 ∧
  truckload_cost = 75 ∧
  sales_tax = 0.20 ∧
  total_cost = 4500 →
  ∃ width : ℝ, width = 20 :=
by
  sorry

end NUMINAMATH_GPT_road_width_l2125_212591


namespace NUMINAMATH_GPT_merchant_marked_price_l2125_212581

-- Definitions
def list_price : ℝ := 100
def purchase_price (L : ℝ) : ℝ := 0.8 * L
def selling_price_with_discount (x : ℝ) : ℝ := 0.75 * x
def profit (purchase_price : ℝ) (selling_price : ℝ) : ℝ := selling_price - purchase_price
def desired_profit (selling_price : ℝ) : ℝ := 0.3 * selling_price

-- Statement to prove
theorem merchant_marked_price :
  ∃ (x : ℝ), 
    profit (purchase_price list_price) (selling_price_with_discount x) = desired_profit (selling_price_with_discount x) ∧
    x / list_price = 152.38 / 100 :=
sorry

end NUMINAMATH_GPT_merchant_marked_price_l2125_212581


namespace NUMINAMATH_GPT_distance_C_distance_BC_l2125_212520

variable (A B C D : ℕ)

theorem distance_C
  (hA : A = 350)
  (hAB : A + B = 600)
  (hABCD : A + B + C + D = 1500)
  (hD : D = 275)
  : C = 625 :=
by
  sorry

theorem distance_BC
  (A B C D : ℕ)
  (hA : A = 350)
  (hAB : A + B = 600)
  (hABCD : A + B + C + D = 1500)
  (hD : D = 275)
  : B + C = 875 :=
by
  sorry

end NUMINAMATH_GPT_distance_C_distance_BC_l2125_212520


namespace NUMINAMATH_GPT_proof_theorem_l2125_212530

noncomputable def proof_problem (a b c : ℝ) := 
  (2 * b = a + c) ∧ 
  (2 / b = 1 / a + 1 / c ∨ 2 / a = 1 / b + 1 / c ∨ 2 / c = 1 / a + 1 / b) ∧ 
  (a ≠ 0) ∧ (b ≠ 0) ∧ (c ≠ 0)

theorem proof_theorem (a b c : ℝ) (h : proof_problem a b c) :
  (a = b ∧ b = c) ∨ 
  (∃ (x : ℝ), x ≠ 0 ∧ a = -4 * x ∧ b = -x ∧ c = 2 * x) :=
by
  sorry

end NUMINAMATH_GPT_proof_theorem_l2125_212530


namespace NUMINAMATH_GPT_certain_number_minus_15_l2125_212579

theorem certain_number_minus_15 (n : ℕ) (h : n / 10 = 6) : n - 15 = 45 :=
sorry

end NUMINAMATH_GPT_certain_number_minus_15_l2125_212579


namespace NUMINAMATH_GPT_jose_investment_l2125_212578

theorem jose_investment (P T : ℝ) (X : ℝ) (months_tom months_jose : ℝ) (profit_total profit_jose profit_tom : ℝ) :
  T = 30000 →
  months_tom = 12 →
  months_jose = 10 →
  profit_total = 54000 →
  profit_jose = 30000 →
  profit_tom = profit_total - profit_jose →
  profit_tom / profit_jose = (T * months_tom) / (X * months_jose) →
  X = 45000 :=
by sorry

end NUMINAMATH_GPT_jose_investment_l2125_212578


namespace NUMINAMATH_GPT_total_pieces_of_junk_mail_l2125_212573

def houses : ℕ := 6
def pieces_per_house : ℕ := 4

theorem total_pieces_of_junk_mail : houses * pieces_per_house = 24 :=
by 
  sorry

end NUMINAMATH_GPT_total_pieces_of_junk_mail_l2125_212573


namespace NUMINAMATH_GPT_present_age_of_B_l2125_212577

theorem present_age_of_B 
  (a b : ℕ)
  (h1 : a + 10 = 2 * (b - 10))
  (h2 : a = b + 9) :
  b = 39 :=
by
  sorry

end NUMINAMATH_GPT_present_age_of_B_l2125_212577


namespace NUMINAMATH_GPT_min_voters_for_tall_24_l2125_212524

/-
There are 105 voters divided into 5 districts, each district divided into 7 sections, with each section having 3 voters.
A section is won by a majority vote. A district is won by a majority of sections. The contest is won by a majority of districts.
Tall won the contest. Prove that the minimum number of voters who could have voted for Tall is 24.
-/
noncomputable def min_voters_for_tall (total_voters districts sections voters_per_section : ℕ) (sections_needed_to_win_district districts_needed_to_win_contest : ℕ) : ℕ :=
  let voters_needed_per_section := voters_per_section / 2 + 1
  sections_needed_to_win_district * districts_needed_to_win_contest * voters_needed_per_section

theorem min_voters_for_tall_24 :
  min_voters_for_tall 105 5 7 3 4 3 = 24 :=
sorry

end NUMINAMATH_GPT_min_voters_for_tall_24_l2125_212524


namespace NUMINAMATH_GPT_distance_between_parallel_lines_l2125_212576

theorem distance_between_parallel_lines :
  let a := 4
  let b := -3
  let c1 := 2
  let c2 := -1
  let d := (abs (c1 - c2)) / (Real.sqrt (a^2 + b^2))
  d = 3 / 5 :=
by
  sorry

end NUMINAMATH_GPT_distance_between_parallel_lines_l2125_212576


namespace NUMINAMATH_GPT_pq_eqv_l2125_212545

theorem pq_eqv (p q : Prop) : 
  ((¬ p ∧ ¬ q) ∧ (p ∨ q)) ↔ ((p ∧ ¬ q) ∨ (¬ p ∧ q)) :=
by
  sorry

end NUMINAMATH_GPT_pq_eqv_l2125_212545


namespace NUMINAMATH_GPT_Borgnine_total_legs_l2125_212532

def numChimps := 12
def numLions := 8
def numLizards := 5
def numTarantulas := 125

def chimpLegsEach := 2
def lionLegsEach := 4
def lizardLegsEach := 4
def tarantulaLegsEach := 8

def legsSeen := numChimps * chimpLegsEach +
                numLions * lionLegsEach +
                numLizards * lizardLegsEach

def legsToSee := numTarantulas * tarantulaLegsEach

def totalLegs := legsSeen + legsToSee

theorem Borgnine_total_legs : totalLegs = 1076 := by
  sorry

end NUMINAMATH_GPT_Borgnine_total_legs_l2125_212532


namespace NUMINAMATH_GPT_graph_symmetry_about_line_l2125_212500

noncomputable def f (x : ℝ) : ℝ := 3 * Real.cos (2 * x - (Real.pi / 3))

theorem graph_symmetry_about_line (x : ℝ) : 
  ∀ x, f (2 * (Real.pi / 3) - x) = f x :=
by
  sorry

end NUMINAMATH_GPT_graph_symmetry_about_line_l2125_212500


namespace NUMINAMATH_GPT_corrected_mean_l2125_212594

theorem corrected_mean (incorrect_mean : ℕ) (num_observations : ℕ) (wrong_value actual_value : ℕ) : 
  (50 * 36 + (43 - 23)) / 50 = 36.4 :=
by
  sorry

end NUMINAMATH_GPT_corrected_mean_l2125_212594


namespace NUMINAMATH_GPT_min_max_expression_l2125_212510

theorem min_max_expression (x : ℝ) (h : 2 ≤ x ∧ x ≤ 7) :
  ∃ (a : ℝ) (b : ℝ), a = 11 / 3 ∧ b = 87 / 16 ∧ 
  (∀ y, 2 ≤ y ∧ y ≤ 7 → 11 / 3 ≤ (y^2 + 4*y + 10) / (2*y + 2)) ∧
  (∀ y, 2 ≤ y ∧ y ≤ 7 → (y^2 + 4*y + 10) / (2*y + 2) ≤ 87 / 16) :=
sorry

end NUMINAMATH_GPT_min_max_expression_l2125_212510


namespace NUMINAMATH_GPT_parabola_directrix_value_l2125_212547

noncomputable def parabola_p_value (p : ℝ) : Prop :=
(∀ y : ℝ, y^2 = 2 * p * (-2 - (-2)))

theorem parabola_directrix_value : parabola_p_value 4 :=
by
  -- proof steps here
  sorry

end NUMINAMATH_GPT_parabola_directrix_value_l2125_212547


namespace NUMINAMATH_GPT_find_F_l2125_212552

theorem find_F (F C : ℝ) (h1 : C = 35) (h2 : C = (7/12) * (F - 40)) : F = 100 :=
by
  sorry

end NUMINAMATH_GPT_find_F_l2125_212552


namespace NUMINAMATH_GPT_jacque_suitcase_weight_l2125_212527

noncomputable def suitcase_weight_return (original_weight : ℝ)
                                         (perfume_weight_oz : ℕ → ℝ)
                                         (chocolate_weight_lb : ℝ)
                                         (soap_weight_oz : ℕ → ℝ)
                                         (jam_weight_oz : ℕ → ℝ)
                                         (sculpture_weight_kg : ℝ)
                                         (shirt_weight_g : ℕ → ℝ)
                                         (oz_to_lb : ℝ)
                                         (kg_to_lb : ℝ)
                                         (g_to_kg : ℝ) : ℝ :=
  original_weight +
  (perfume_weight_oz 5 / oz_to_lb) +
  chocolate_weight_lb +
  (soap_weight_oz 2 / oz_to_lb) +
  (jam_weight_oz 2 / oz_to_lb) +
  (sculpture_weight_kg * kg_to_lb) +
  ((shirt_weight_g 3 / g_to_kg) * kg_to_lb)

theorem jacque_suitcase_weight :
  suitcase_weight_return 12 
                        (fun n => n * 1.2) 
                        4 
                        (fun n => n * 5) 
                        (fun n => n * 8)
                        3.5 
                        (fun n => n * 300) 
                        16 
                        2.20462 
                        1000 
  = 27.70 :=
sorry

end NUMINAMATH_GPT_jacque_suitcase_weight_l2125_212527


namespace NUMINAMATH_GPT_slower_pipe_fills_tank_in_200_minutes_l2125_212589

noncomputable def slower_pipe_filling_time (F S : ℝ) (h1 : F = 4 * S) (h2 : F + S = 1 / 40) : ℝ :=
  1 / S

theorem slower_pipe_fills_tank_in_200_minutes (F S : ℝ) (h1 : F = 4 * S) (h2 : F + S = 1 / 40) :
  slower_pipe_filling_time F S h1 h2 = 200 :=
sorry

end NUMINAMATH_GPT_slower_pipe_fills_tank_in_200_minutes_l2125_212589


namespace NUMINAMATH_GPT_smallest_abs_value_l2125_212507

theorem smallest_abs_value : 
    ∀ (a b c d : ℝ), 
    a = -1/2 → b = -2/3 → c = 4 → d = -5 → 
    abs a < abs b ∧ abs a < abs c ∧ abs a < abs d := 
by
  intros a b c d ha hb hc hd
  rw [ha, hb, hc, hd]
  simp
  -- Proof omitted for brevity
  sorry

end NUMINAMATH_GPT_smallest_abs_value_l2125_212507


namespace NUMINAMATH_GPT_largest_common_value_l2125_212597

/-- The largest value less than 300 that appears in both sequences 
    {7, 14, 21, 28, ...} and {5, 15, 25, 35, ...} -/
theorem largest_common_value (a : ℕ) (n m k : ℕ) :
  (a = 7 * (1 + n)) ∧ (a = 5 + 10 * m) ∧ (a < 300) ∧ (∀ k, (55 + 70 * k < 300) → (55 + 70 * k) ≤ a) 
  → a = 265 :=
by
  sorry

end NUMINAMATH_GPT_largest_common_value_l2125_212597


namespace NUMINAMATH_GPT_tax_percentage_l2125_212556

-- Definitions
def salary_before_taxes := 5000
def rent_expense_per_month := 1350
def total_late_rent_payments := 2 * rent_expense_per_month
def fraction_of_next_salary_after_taxes := (3 / 5 : ℚ)

-- Main statement to prove
theorem tax_percentage (T : ℚ) : 
  fraction_of_next_salary_after_taxes * (salary_before_taxes - (T / 100) * salary_before_taxes) = total_late_rent_payments → 
  T = 10 :=
by
  sorry

end NUMINAMATH_GPT_tax_percentage_l2125_212556


namespace NUMINAMATH_GPT_pressure_force_correct_l2125_212503

-- Define the conditions
noncomputable def base_length : ℝ := 4
noncomputable def vertex_depth : ℝ := 4
noncomputable def gamma : ℝ := 1000 -- density of water in kg/m^3
noncomputable def g : ℝ := 9.81 -- acceleration due to gravity in m/s^2

-- Define the calculation of the pressure force on the parabolic segment
noncomputable def pressure_force (base_length vertex_depth gamma g : ℝ) : ℝ :=
  19620 * (4 * ((2/3) * (4 : ℝ)^(3/2)) - ((2/5) * (4 : ℝ)^(5/2)))

-- State the theorem
theorem pressure_force_correct : pressure_force base_length vertex_depth gamma g = 167424 := 
by
  sorry

end NUMINAMATH_GPT_pressure_force_correct_l2125_212503


namespace NUMINAMATH_GPT_total_apples_purchased_l2125_212583

theorem total_apples_purchased (M : ℝ) (T : ℝ) (W : ℝ) 
    (hM : M = 15.5)
    (hT : T = 3.2 * M)
    (hW : W = 1.05 * T) :
    M + T + W = 117.18 := by
  sorry

end NUMINAMATH_GPT_total_apples_purchased_l2125_212583


namespace NUMINAMATH_GPT_sphere_radius_same_volume_l2125_212590

noncomputable def tent_radius : ℝ := 3
noncomputable def tent_height : ℝ := 9

theorem sphere_radius_same_volume : 
  (4 / 3) * Real.pi * ( (20.25)^(1/3) )^3 = (1 / 3) * Real.pi * tent_radius^2 * tent_height :=
by
  sorry

end NUMINAMATH_GPT_sphere_radius_same_volume_l2125_212590


namespace NUMINAMATH_GPT_range_of_a_l2125_212553

-- Defining the propositions P and Q 
def P (a : ℝ) : Prop := ∀ x : ℝ, a*x^2 + a*x + 1 > 0
def Q (a : ℝ) : Prop := ∃ x1 x2 : ℝ, x1^2 - x1 + a = 0 ∧ x2^2 - x2 + a = 0

-- Stating the theorem
theorem range_of_a (a : ℝ) :
  (P a ∧ ¬Q a) ∨ (¬P a ∧ Q a) ↔ a ∈ Set.Ioo (1/4 : ℝ) 4 ∪ Set.Iio 0 :=
sorry

end NUMINAMATH_GPT_range_of_a_l2125_212553


namespace NUMINAMATH_GPT_correct_sum_is_826_l2125_212505

theorem correct_sum_is_826 (ABC : ℕ)
  (h1 : 100 ≤ ABC ∧ ABC < 1000)  -- Ensuring ABC is a three-digit number
  (h2 : ∃ A B C : ℕ, ABC = 100 * A + 10 * B + C ∧ C = 6) -- Misread ones digit is 6
  (incorrect_sum : ℕ)
  (h3 : incorrect_sum = ABC + 57)  -- Sum obtained by Yoongi was 823
  (h4 : incorrect_sum = 823) : ABC + 57 + 3 = 826 :=  -- Correcting the sum considering the 6 to 9 error
by
  sorry

end NUMINAMATH_GPT_correct_sum_is_826_l2125_212505


namespace NUMINAMATH_GPT_evaluate_expression_l2125_212562

theorem evaluate_expression (x y : ℝ) (h1 : x = 3) (h2 : y = 0) : y * (y - 3 * x) = 0 :=
by sorry

end NUMINAMATH_GPT_evaluate_expression_l2125_212562


namespace NUMINAMATH_GPT_find_y_value_l2125_212557

theorem find_y_value (y : ℝ) (h : 12^2 * y^3 / 432 = 72) : y = 6 :=
by
  sorry

end NUMINAMATH_GPT_find_y_value_l2125_212557


namespace NUMINAMATH_GPT_solution_set_of_inequality_l2125_212550

theorem solution_set_of_inequality : 
  {x : ℝ | |x|^3 - 2 * x^2 - 4 * |x| + 3 < 0} = 
  { x : ℝ | -3 < x ∧ x < -1 } ∪ { x : ℝ | 1 < x ∧ x < 3 } := 
by
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l2125_212550


namespace NUMINAMATH_GPT_calculate_otimes_l2125_212587

def otimes (a b : ℚ) : ℚ := (a + b) / (a - b)

theorem calculate_otimes :
  otimes (otimes 8 6) 12 = -19 / 5 := by
  sorry

end NUMINAMATH_GPT_calculate_otimes_l2125_212587


namespace NUMINAMATH_GPT_range_of_m_l2125_212529

theorem range_of_m (x m : ℝ) :
  (∀ x, (x - 1) / 2 ≥ (x - 2) / 3 → 2 * x - m ≥ x → x ≥ m) ↔ m ≥ -1 := by
  sorry

end NUMINAMATH_GPT_range_of_m_l2125_212529


namespace NUMINAMATH_GPT_find_c_value_l2125_212569

theorem find_c_value :
  ∃ c : ℝ, (∀ x y : ℝ, (x + 10) ^ 2 + (y + 4) ^ 2 = 169 ∧ (x - 3) ^ 2 + (y - 9) ^ 2 = 65 → x + y = c) ∧ c = 3 :=
sorry

end NUMINAMATH_GPT_find_c_value_l2125_212569


namespace NUMINAMATH_GPT_quadratic_roots_ratio_l2125_212592

theorem quadratic_roots_ratio (r1 r2 p q n : ℝ) (h1 : p = r1 * r2) (h2 : q = -(r1 + r2)) (h3 : p ≠ 0) (h4 : q ≠ 0) (h5 : n ≠ 0) (h6 : r1 ≠ 0) (h7 : r2 ≠ 0) (h8 : x^2 + q * x + p = 0) (h9 : x^2 + p * x + n = 0) :
  n / q = -3 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_roots_ratio_l2125_212592


namespace NUMINAMATH_GPT_find_x_l2125_212511

-- Definitions for the vectors a and b
def a : ℝ × ℝ := (3, 4)
def b : ℝ × ℝ := (2, 1)

-- Definition for the condition of parallel vectors
def are_parallel (v1 v2 : ℝ × ℝ) : Prop :=
  v1.1 * v2.2 = v1.2 * v2.1

-- Mathematical statement to prove
theorem find_x (x : ℝ) 
  (h_parallel : are_parallel (a.1 + x * b.1, a.2 + x * b.2) (a.1 - b.1, a.2 - b.2)) : 
  x = -1 :=
sorry

end NUMINAMATH_GPT_find_x_l2125_212511


namespace NUMINAMATH_GPT_yvon_combination_l2125_212534

theorem yvon_combination :
  let num_notebooks := 4
  let num_pens := 5
  num_notebooks * num_pens = 20 :=
by
  sorry

end NUMINAMATH_GPT_yvon_combination_l2125_212534


namespace NUMINAMATH_GPT_Q_at_1_eq_neg_1_l2125_212596

def P (x : ℝ) : ℝ := 3 * x^3 - 5 * x^2 + 2 * x - 1

noncomputable def mean_coefficient : ℝ := (3 - 5 + 2 - 1) / 4

noncomputable def Q (x : ℝ) : ℝ := mean_coefficient * x^3 + mean_coefficient * x^2 + mean_coefficient * x + mean_coefficient

theorem Q_at_1_eq_neg_1 : Q 1 = -1 := by
  sorry

end NUMINAMATH_GPT_Q_at_1_eq_neg_1_l2125_212596


namespace NUMINAMATH_GPT_trinomial_ne_binomial_l2125_212533

theorem trinomial_ne_binomial (a b c A B : ℝ) (h : a ≠ 0) : 
  ¬ ∀ x : ℝ, ax^2 + bx + c = Ax + B :=
by
  sorry

end NUMINAMATH_GPT_trinomial_ne_binomial_l2125_212533


namespace NUMINAMATH_GPT_m_ducks_l2125_212516

variable (M C K : ℕ)

theorem m_ducks :
  (M = C + 4) ∧
  (M = 2 * C + K + 3) ∧
  (M + C + K = 90) →
  M = 89 := by
  sorry

end NUMINAMATH_GPT_m_ducks_l2125_212516


namespace NUMINAMATH_GPT_remainder_of_504_divided_by_100_is_4_l2125_212513

theorem remainder_of_504_divided_by_100_is_4 :
  (504 % 100) = 4 :=
by
  sorry

end NUMINAMATH_GPT_remainder_of_504_divided_by_100_is_4_l2125_212513


namespace NUMINAMATH_GPT_difference_of_two_numbers_l2125_212542

theorem difference_of_two_numbers :
  ∃ S : ℕ, S * 16 + 15 = 1600 ∧ 1600 - S = 1501 :=
by
  sorry

end NUMINAMATH_GPT_difference_of_two_numbers_l2125_212542


namespace NUMINAMATH_GPT_min_value_of_a_sq_plus_b_sq_over_a_minus_b_l2125_212585

theorem min_value_of_a_sq_plus_b_sq_over_a_minus_b {a b : ℝ} (h1 : a > b) (h2 : a * b = 1) : 
  ∃ x, x = 2 * Real.sqrt 2 ∧ ∀ y, y = (a^2 + b^2) / (a - b) → y ≥ x :=
by {
  sorry
}

end NUMINAMATH_GPT_min_value_of_a_sq_plus_b_sq_over_a_minus_b_l2125_212585


namespace NUMINAMATH_GPT_division_and_multiplication_l2125_212541

theorem division_and_multiplication (dividend divisor quotient factor product : ℕ) 
  (h₁ : dividend = 24) 
  (h₂ : divisor = 3) 
  (h₃ : quotient = dividend / divisor) 
  (h₄ : factor = 5) 
  (h₅ : product = quotient * factor) : 
  quotient = 8 ∧ product = 40 := 
by 
  sorry

end NUMINAMATH_GPT_division_and_multiplication_l2125_212541


namespace NUMINAMATH_GPT_estimate_m_value_l2125_212558

-- Definition of polynomial P(x) and its roots related to the problem
noncomputable def P (x : ℂ) (a b c : ℂ) : ℂ := x^3 + a * x^2 + b * x + c

-- Statement of the problem in Lean 4
theorem estimate_m_value :
  ∀ (a b c : ℕ),
  a ≤ 100 ∧ b ≤ 100 ∧ c ≤ 100 ∧
  (∃ z1 z2 z3 : ℂ, z1 ≠ z2 ∧ z1 ≠ z3 ∧ z2 ≠ z3 ∧ 
  P z1 a b c = 0 ∧ P z2 a b c = 0 ∧ P z3 a b c = 0) →
  ∃ m : ℕ, m = 8097 :=
sorry

end NUMINAMATH_GPT_estimate_m_value_l2125_212558


namespace NUMINAMATH_GPT_min_unsuccessful_placements_8x8_l2125_212559

-- Define the board, the placement, and the unsuccessful condition
def is_unsuccessful_placement (board : ℕ → ℕ → ℤ) (i j : ℕ) : Prop :=
  (i < 7 ∧ j < 7 ∧ (board i j + board (i+1) j + board i (j+1) + board (i+1) (j+1)) ≠ 0)

-- Main theorem: The minimum number of unsuccessful placements is 36 on an 8x8 board
theorem min_unsuccessful_placements_8x8 (board : ℕ → ℕ → ℤ) (H : ∀ i j, board i j = 1 ∨ board i j = -1) :
  ∃ (n : ℕ), n = 36 ∧ (∀ m : ℕ, (∀ i j, is_unsuccessful_placement board i j → m < 36 ) → m = n) :=
sorry

end NUMINAMATH_GPT_min_unsuccessful_placements_8x8_l2125_212559


namespace NUMINAMATH_GPT_largest_x_l2125_212512

-- Define the condition of the problem.
def equation_holds (x : ℝ) : Prop :=
  (5 * x - 20) / (4 * x - 5) ^ 2 + (5 * x - 20) / (4 * x - 5) = 20

-- State the theorem to prove the largest value of x is 9/5.
theorem largest_x : ∃ x : ℝ, equation_holds x ∧ ∀ y : ℝ, equation_holds y → y ≤ 9 / 5 :=
by
  sorry

end NUMINAMATH_GPT_largest_x_l2125_212512


namespace NUMINAMATH_GPT_volume_units_correct_l2125_212563

/-- Definition for the volume of a bottle of coconut juice in milliliters (200 milliliters). -/
def volume_of_coconut_juice := 200 

/-- Definition for the volume of an electric water heater in liters (50 liters). -/
def volume_of_electric_water_heater := 50 

/-- Prove that the volume of a bottle of coconut juice is measured in milliliters (200 milliliters)
    and the volume of an electric water heater is measured in liters (50 liters).
-/
theorem volume_units_correct :
  volume_of_coconut_juice = 200 ∧ volume_of_electric_water_heater = 50 :=
sorry

end NUMINAMATH_GPT_volume_units_correct_l2125_212563


namespace NUMINAMATH_GPT_find_a_b_and_compare_y_values_l2125_212515

-- Conditions
def quadratic (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x - 1
def linear (a : ℝ) (x : ℝ) : ℝ := a * x

-- Problem statement
theorem find_a_b_and_compare_y_values (a b y1 y2 y3 : ℝ) (h₀ : quadratic a b (-2) = 1) (h₁ : linear a (-2) = 1)
    (h2 : y1 = quadratic a b 2) (h3 : y2 = quadratic a b b) (h4 : y3 = quadratic a b (a - b)) :
  (a = -1/2) ∧ (b = -2) ∧ y1 < y3 ∧ y3 < y2 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_find_a_b_and_compare_y_values_l2125_212515


namespace NUMINAMATH_GPT_second_account_interest_rate_l2125_212525

theorem second_account_interest_rate
  (investment1 : ℝ)
  (rate1 : ℝ)
  (interest1 : ℝ)
  (investment2 : ℝ)
  (interest2 : ℝ)
  (h1 : 4000 = investment1)
  (h2 : 0.08 = rate1)
  (h3 : 320 = interest1)
  (h4 : 7200 - 4000 = investment2)
  (h5 : interest1 = interest2) :
  interest2 / investment2 = 0.1 :=
by
  sorry

end NUMINAMATH_GPT_second_account_interest_rate_l2125_212525


namespace NUMINAMATH_GPT_pilot_speed_outbound_l2125_212517

theorem pilot_speed_outbound (v : ℝ) (d : ℝ) (s_return : ℝ) (t_total : ℝ) 
    (return_time : ℝ := d / s_return) 
    (outbound_time : ℝ := t_total - return_time) 
    (speed_outbound : ℝ := d / outbound_time) :
  d = 1500 → s_return = 500 → t_total = 8 → speed_outbound = 300 :=
by
  intros hd hs ht
  sorry

end NUMINAMATH_GPT_pilot_speed_outbound_l2125_212517


namespace NUMINAMATH_GPT_expression_value_l2125_212540

theorem expression_value : (4 - 2) ^ 3 = 8 :=
by sorry

end NUMINAMATH_GPT_expression_value_l2125_212540


namespace NUMINAMATH_GPT_area_of_smaller_circle_l2125_212521

noncomputable def radius_of_smaller_circle (r : ℝ) : ℝ := r

noncomputable def radius_of_larger_circle (r : ℝ) : ℝ := 3 * r

noncomputable def length_PA := 5
noncomputable def length_AB := 5

theorem area_of_smaller_circle (r : ℝ) (h1 : radius_of_smaller_circle r = r)
  (h2 : radius_of_larger_circle r = 3 * r)
  (h3 : length_PA = 5) (h4 : length_AB = 5) :
  π * r^2 = (25 / 3) * π :=
  sorry

end NUMINAMATH_GPT_area_of_smaller_circle_l2125_212521


namespace NUMINAMATH_GPT_fewer_twos_result_100_l2125_212501

theorem fewer_twos_result_100 :
  (222 / 2) - (22 / 2) = 100 := by
  sorry

end NUMINAMATH_GPT_fewer_twos_result_100_l2125_212501


namespace NUMINAMATH_GPT_sequence_is_geometric_not_arithmetic_l2125_212502

noncomputable def a_n (n : ℕ) : ℕ :=
  if n = 1 then 1
  else 2^(n-1)

def S_n (n : ℕ) : ℕ :=
  2^n - 1

theorem sequence_is_geometric_not_arithmetic (n : ℕ) : 
  (∀ n ≥ 2, a_n n = S_n n - S_n (n - 1)) ∧
  (a_n 1 = 1) ∧
  (∃ r : ℕ, r > 1 ∧ ∀ n ≥ 1, a_n (n + 1) = r * a_n n) ∧
  ¬(∃ d : ℤ, ∀ n, (a_n (n + 1) : ℤ) = a_n n + d) :=
by
  sorry

end NUMINAMATH_GPT_sequence_is_geometric_not_arithmetic_l2125_212502


namespace NUMINAMATH_GPT_soldiers_line_l2125_212567

theorem soldiers_line (n x y z : ℕ) (h₁ : y = 6 * x) (h₂ : y = 7 * z)
                      (h₃ : n = x + y) (h₄ : n = 7 * x) (h₅ : n = 8 * z) : n = 98 :=
by 
  sorry

end NUMINAMATH_GPT_soldiers_line_l2125_212567


namespace NUMINAMATH_GPT_sum_of_number_and_preceding_l2125_212575

theorem sum_of_number_and_preceding (n : ℤ) (h : 6 * n - 2 = 100) : n + (n - 1) = 33 :=
by {
  sorry
}

end NUMINAMATH_GPT_sum_of_number_and_preceding_l2125_212575


namespace NUMINAMATH_GPT_percentage_problem_l2125_212531

theorem percentage_problem 
    (y : ℝ)
    (h₁ : 0.47 * 1442 = 677.74)
    (h₂ : (677.74 - (y / 100) * 1412) + 63 = 3) :
    y = 52.25 :=
by sorry

end NUMINAMATH_GPT_percentage_problem_l2125_212531


namespace NUMINAMATH_GPT_find_baking_soda_boxes_l2125_212595

-- Define the quantities and costs
def num_flour_boxes := 3
def cost_per_flour_box := 3
def num_egg_trays := 3
def cost_per_egg_tray := 10
def num_milk_liters := 7
def cost_per_milk_liter := 5
def baking_soda_cost_per_box := 3
def total_cost := 80

-- Define the total cost of flour, eggs, and milk
def total_flour_cost := num_flour_boxes * cost_per_flour_box
def total_egg_cost := num_egg_trays * cost_per_egg_tray
def total_milk_cost := num_milk_liters * cost_per_milk_liter

-- Define the total cost of non-baking soda items
def total_non_baking_soda_cost := total_flour_cost + total_egg_cost + total_milk_cost

-- Define the remaining cost for baking soda
def baking_soda_total_cost := total_cost - total_non_baking_soda_cost

-- Define the number of baking soda boxes
def num_baking_soda_boxes := baking_soda_total_cost / baking_soda_cost_per_box

theorem find_baking_soda_boxes : num_baking_soda_boxes = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_baking_soda_boxes_l2125_212595


namespace NUMINAMATH_GPT_find_initial_apples_l2125_212549

def initial_apples (a b c : ℕ) : Prop :=
  b + c = a

theorem find_initial_apples (a b initial_apples : ℕ) (h : b + initial_apples = a) : initial_apples = 8 :=
by
  sorry

end NUMINAMATH_GPT_find_initial_apples_l2125_212549


namespace NUMINAMATH_GPT_diagonal_length_of_rhombus_l2125_212508

-- Definitions for the conditions
def side_length_of_square : ℝ := 8
def area_of_square : ℝ := side_length_of_square ^ 2
def area_of_rhombus : ℝ := 64
def d2 : ℝ := 8
-- Question
theorem diagonal_length_of_rhombus (d1 : ℝ) : (d1 * d2) / 2 = area_of_rhombus ↔ d1 = 16 := by
  sorry

end NUMINAMATH_GPT_diagonal_length_of_rhombus_l2125_212508


namespace NUMINAMATH_GPT_pq_logic_l2125_212506

theorem pq_logic (p q : Prop) (h1 : p ∨ q) (h2 : ¬p) : ¬p ∧ q :=
by
  sorry

end NUMINAMATH_GPT_pq_logic_l2125_212506


namespace NUMINAMATH_GPT_nine_x_plus_twenty_seven_y_l2125_212555

theorem nine_x_plus_twenty_seven_y (x y : ℤ) (h : 17 * x + 51 * y = 102) : 9 * x + 27 * y = 54 := 
by sorry

end NUMINAMATH_GPT_nine_x_plus_twenty_seven_y_l2125_212555


namespace NUMINAMATH_GPT_elizabeth_time_l2125_212570

-- Defining the conditions
def tom_time_minutes : ℕ := 120
def time_ratio : ℕ := 4

-- Proving Elizabeth's time
theorem elizabeth_time : tom_time_minutes / time_ratio = 30 := 
by
  sorry

end NUMINAMATH_GPT_elizabeth_time_l2125_212570


namespace NUMINAMATH_GPT_valid_k_range_l2125_212599

noncomputable def fx (k : ℝ) (x : ℝ) : ℝ :=
  k * x^2 + k * x + k + 3

theorem valid_k_range:
  (∀ x : ℝ, -2 ≤ x ∧ x ≤ 3 → fx k x ≥ 0) ↔ (k ≥ -3 / 13) :=
by
  sorry

end NUMINAMATH_GPT_valid_k_range_l2125_212599


namespace NUMINAMATH_GPT_total_balls_in_box_l2125_212528

theorem total_balls_in_box :
  ∀ (W B R : ℕ), 
    W = 16 →
    B = W + 12 →
    R = 2 * B →
    W + B + R = 100 :=
by
  intros W B R hW hB hR
  sorry

end NUMINAMATH_GPT_total_balls_in_box_l2125_212528


namespace NUMINAMATH_GPT_find_n_l2125_212588

theorem find_n (n : ℕ) :
  Int.lcm n 16 = 52 ∧ Nat.gcd n 16 = 8 → n = 26 :=
by
  sorry

end NUMINAMATH_GPT_find_n_l2125_212588


namespace NUMINAMATH_GPT_coach_mike_change_l2125_212561

theorem coach_mike_change (cost amount_given change : ℕ) 
    (h_cost : cost = 58) (h_amount_given : amount_given = 75) : 
    change = amount_given - cost → change = 17 := by
    sorry

end NUMINAMATH_GPT_coach_mike_change_l2125_212561


namespace NUMINAMATH_GPT_quadratic_two_distinct_real_roots_l2125_212538

theorem quadratic_two_distinct_real_roots (k : ℝ) :
    (∃ x : ℝ, 2 * k * x^2 + (8 * k + 1) * x + 8 * k = 0 ∧ 2 * k ≠ 0) →
    k > -1/16 ∧ k ≠ 0 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_quadratic_two_distinct_real_roots_l2125_212538


namespace NUMINAMATH_GPT_inequality_solution_l2125_212519

theorem inequality_solution (x : ℝ) : x ∈ Set.Ioo (-7 : ℝ) (7 : ℝ) ↔ (x^2 - 49) / (x + 7) < 0 :=
by 
  sorry

end NUMINAMATH_GPT_inequality_solution_l2125_212519


namespace NUMINAMATH_GPT_jeanne_should_buy_more_tickets_l2125_212537

theorem jeanne_should_buy_more_tickets :
  let cost_ferris_wheel := 5
  let cost_roller_coaster := 4
  let cost_bumper_cars := 4
  let jeanne_current_tickets := 5
  let total_tickets_needed := cost_ferris_wheel + cost_roller_coaster + cost_bumper_cars
  let tickets_needed_to_buy := total_tickets_needed - jeanne_current_tickets
  tickets_needed_to_buy = 8 :=
by
  sorry

end NUMINAMATH_GPT_jeanne_should_buy_more_tickets_l2125_212537


namespace NUMINAMATH_GPT_max_min_difference_l2125_212571

variable (x y z : ℝ)

theorem max_min_difference :
  x + y + z = 3 →
  x^2 + y^2 + z^2 = 18 →
  (max z (-z)) - ((min z (-z))) = 6 :=
  by
    intros h1 h2
    sorry

end NUMINAMATH_GPT_max_min_difference_l2125_212571


namespace NUMINAMATH_GPT_sum_n_k_l2125_212523

theorem sum_n_k (n k : ℕ) (h₁ : (x+1)^n = 2 * x^k + 3 * x^(k+1) + 4 * x^(k+2)) (h₂ : 3 * k + 3 = 2 * n - 2 * k)
  (h₃ : 4 * k + 8 = 3 * n - 3 * k - 3) : n + k = 47 := 
sorry

end NUMINAMATH_GPT_sum_n_k_l2125_212523


namespace NUMINAMATH_GPT_rhombus_second_diagonal_l2125_212580

theorem rhombus_second_diagonal (perimeter : ℝ) (d1 : ℝ) (side : ℝ) (half_d2 : ℝ) (d2 : ℝ) :
  perimeter = 52 → d1 = 24 → side = 13 → (half_d2 = 5) → d2 = 2 * half_d2 → d2 = 10 :=
by
  sorry

end NUMINAMATH_GPT_rhombus_second_diagonal_l2125_212580
