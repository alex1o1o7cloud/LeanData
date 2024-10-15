import Mathlib

namespace NUMINAMATH_GPT_cos_30_deg_plus_2a_l217_21751

theorem cos_30_deg_plus_2a (a : ℝ) (h : Real.cos (Real.pi * (75 / 180) - a) = 1 / 3) : 
  Real.cos (Real.pi * (30 / 180) + 2 * a) = 7 / 9 := 
by 
  sorry

end NUMINAMATH_GPT_cos_30_deg_plus_2a_l217_21751


namespace NUMINAMATH_GPT_probability_of_neighboring_points_l217_21704

theorem probability_of_neighboring_points (n : ℕ) (h : n ≥ 3) : 
  (2 / (n - 1) : ℝ) = (n / (n * (n - 1) / 2) : ℝ) :=
by sorry

end NUMINAMATH_GPT_probability_of_neighboring_points_l217_21704


namespace NUMINAMATH_GPT_roots_reciprocal_l217_21775

theorem roots_reciprocal (x1 x2 : ℝ) (h1 : x1^2 - 3 * x1 - 1 = 0) (h2 : x2^2 - 3 * x2 - 1 = 0) 
                         (h_sum : x1 + x2 = 3) (h_prod : x1 * x2 = -1) :
  (1 / x1) + (1 / x2) = -3 :=
by
  sorry

end NUMINAMATH_GPT_roots_reciprocal_l217_21775


namespace NUMINAMATH_GPT_semicircle_radius_in_trapezoid_l217_21771

theorem semicircle_radius_in_trapezoid 
  (AB CD : ℝ) (AD BC : ℝ) (r : ℝ)
  (h1 : AB = 27) 
  (h2 : CD = 45) 
  (h3 : AD = 13) 
  (h4 : BC = 15) 
  (h5 : r = 13.5) :
  r = 13.5 :=
by
  sorry  -- Detailed proof steps will go here

end NUMINAMATH_GPT_semicircle_radius_in_trapezoid_l217_21771


namespace NUMINAMATH_GPT_solve_for_x_l217_21795

theorem solve_for_x {x : ℤ} (h : 3 * x + 7 = -2) : x = -3 := 
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l217_21795


namespace NUMINAMATH_GPT_fish_offspring_base10_l217_21710

def convert_base_7_to_10 (n : ℕ) : ℕ :=
  let d2 := n / 49
  let r2 := n % 49
  let d1 := r2 / 7
  let d0 := r2 % 7
  d2 * 49 + d1 * 7 + d0

theorem fish_offspring_base10 :
  convert_base_7_to_10 265 = 145 :=
by
  sorry

end NUMINAMATH_GPT_fish_offspring_base10_l217_21710


namespace NUMINAMATH_GPT_no_three_consecutive_geometric_l217_21755

open Nat

def a (n : ℕ) : ℤ := 3^n - 2^n

theorem no_three_consecutive_geometric :
  ∀ (k : ℕ), ¬ (∃ n m : ℕ, m = n + 1 ∧ k = m + 1 ∧ (a n) * (a k) = (a m)^2) :=
by
  sorry

end NUMINAMATH_GPT_no_three_consecutive_geometric_l217_21755


namespace NUMINAMATH_GPT_range_of_m_l217_21758

-- Define the ellipse and conditions
def ellipse (x y : ℝ) (m : ℝ) : Prop := (x^2 / m) + (y^2 / 2) = 1
def point_exists (M : ℝ × ℝ) (C : ℝ → ℝ → ℝ → Prop) : Prop := ∃ p : ℝ × ℝ, C p.1 p.2 (M.1 + M.2)

-- State the theorem
theorem range_of_m (m : ℝ) (h₁ : ellipse x y m) (h₂ : point_exists M ellipse) :
  (0 < m ∧ m <= 1/2) ∨ (8 <= m) := 
sorry

end NUMINAMATH_GPT_range_of_m_l217_21758


namespace NUMINAMATH_GPT_angle_value_l217_21740

theorem angle_value (y : ℝ) (h1 : 2 * y + 140 = 360) : y = 110 :=
by {
  -- Proof will be written here
  sorry
}

end NUMINAMATH_GPT_angle_value_l217_21740


namespace NUMINAMATH_GPT_least_positive_t_geometric_progression_l217_21716

open Real

theorem least_positive_t_geometric_progression (α : ℝ) (hα1 : 0 < α) (hα2 : α < π / 2) : 
  ∃ t : ℕ, ∀ t' : ℕ, (t' > 0) → 
  (|arcsin (sin (t' * α)) - 8 * α| = 0) → t = 8 :=
by
  sorry

end NUMINAMATH_GPT_least_positive_t_geometric_progression_l217_21716


namespace NUMINAMATH_GPT_total_area_of_folded_blankets_l217_21742

-- Define the initial conditions
def initial_area : ℕ := 8 * 8
def folds : ℕ := 4
def num_blankets : ℕ := 3

-- Define the hypothesis about folding
def folded_area (initial_area : ℕ) (folds : ℕ) : ℕ :=
  initial_area / (2 ^ folds)

-- The total area of all folded blankets
def total_folded_area (initial_area : ℕ) (folds : ℕ) (num_blankets : ℕ) : ℕ :=
  num_blankets * folded_area initial_area folds

-- The theorem we want to prove
theorem total_area_of_folded_blankets : total_folded_area initial_area folds num_blankets = 12 := by
  sorry

end NUMINAMATH_GPT_total_area_of_folded_blankets_l217_21742


namespace NUMINAMATH_GPT_last_digit_of_expression_l217_21746

def last_digit (n : ℕ) : ℕ := n % 10

theorem last_digit_of_expression (n : ℕ) : last_digit (n ^ 9999 - n ^ 5555) = 0 :=
by
  sorry

end NUMINAMATH_GPT_last_digit_of_expression_l217_21746


namespace NUMINAMATH_GPT_minimize_y_l217_21708

theorem minimize_y (a b : ℝ) : 
  ∃ x : ℝ, x = (3 * a + b) / 4 ∧ 
  ∀ y : ℝ, (3 * (y - a) ^ 2 + (y - b) ^ 2) ≥ (3 * ((3 * a + b) / 4 - a) ^ 2 + ((3 * a + b) / 4 - b) ^ 2) :=
sorry

end NUMINAMATH_GPT_minimize_y_l217_21708


namespace NUMINAMATH_GPT_hamburger_combinations_l217_21725

theorem hamburger_combinations : 
  let condiments := 10  -- Number of available condiments
  let patty_choices := 4 -- Number of meat patty options
  2^condiments * patty_choices = 4096 :=
by sorry

end NUMINAMATH_GPT_hamburger_combinations_l217_21725


namespace NUMINAMATH_GPT_teams_in_league_l217_21792

def number_of_teams (n : ℕ) := n * (n - 1) / 2

theorem teams_in_league : ∃ n : ℕ, number_of_teams n = 36 ∧ n = 9 := by
  sorry

end NUMINAMATH_GPT_teams_in_league_l217_21792


namespace NUMINAMATH_GPT_b_plus_d_over_a_l217_21707

theorem b_plus_d_over_a (a b c d e : ℝ) (h : a ≠ 0) 
  (root1 : a * (5:ℝ)^4 + b * (5:ℝ)^3 + c * (5:ℝ)^2 + d * (5:ℝ) + e = 0)
  (root2 : a * (-3:ℝ)^4 + b * (-3:ℝ)^3 + c * (-3:ℝ)^2 + d * (-3:ℝ) + e = 0)
  (root3 : a * (2:ℝ)^4 + b * (2:ℝ)^3 + c * (2:ℝ)^2 + d * (2:ℝ) + e = 0) :
  (b + d) / a = - (12496 / 3173) :=
sorry

end NUMINAMATH_GPT_b_plus_d_over_a_l217_21707


namespace NUMINAMATH_GPT_Jolene_raised_total_money_l217_21730

-- Definitions for the conditions
def babysits_earning_per_family : ℤ := 30
def number_of_families : ℤ := 4
def cars_earning_per_car : ℤ := 12
def number_of_cars : ℤ := 5

-- Calculation of total earnings
def babysitting_earnings : ℤ := babysits_earning_per_family * number_of_families
def car_washing_earnings : ℤ := cars_earning_per_car * number_of_cars
def total_earnings : ℤ := babysitting_earnings + car_washing_earnings

-- The proof statement
theorem Jolene_raised_total_money : total_earnings = 180 := by
  sorry

end NUMINAMATH_GPT_Jolene_raised_total_money_l217_21730


namespace NUMINAMATH_GPT_math_problem_l217_21785

theorem math_problem (x y : ℝ) (h1 : x^7 > y^6) (h2 : y^7 > x^6) : x + y > 2 :=
sorry

end NUMINAMATH_GPT_math_problem_l217_21785


namespace NUMINAMATH_GPT_central_angle_double_score_l217_21786

theorem central_angle_double_score 
  (prob: ℚ)
  (total_angle: ℚ)
  (num_regions: ℚ)
  (eq_regions: ℚ → Prop)
  (double_score_prob: prob = 1/8)
  (total_angle_eq: total_angle = 360)
  (num_regions_eq: num_regions = 6) 
  : ∃ x: ℚ, (prob = x / total_angle) → x = 45 :=
by
  sorry

end NUMINAMATH_GPT_central_angle_double_score_l217_21786


namespace NUMINAMATH_GPT_rita_book_pages_l217_21750

theorem rita_book_pages (x : ℕ) (h1 : ∃ n₁, n₁ = (1/6 : ℚ) * x + 10) 
                                  (h2 : ∃ n₂, n₂ = (1/5 : ℚ) * ((5/6 : ℚ) * x - 10) + 20)
                                  (h3 : ∃ n₃, n₃ = (1/4 : ℚ) * ((4/5 : ℚ) * ((5/6 : ℚ) * x - 10) - 20) + 25)
                                  (h4 : ((3/4 : ℚ) * ((2/3 : ℚ) * x - 28) - 25) = 50) :
    x = 192 := 
sorry

end NUMINAMATH_GPT_rita_book_pages_l217_21750


namespace NUMINAMATH_GPT_jenny_profit_l217_21741

-- Define the constants given in the problem
def cost_per_pan : ℝ := 10.00
def price_per_pan : ℝ := 25.00
def num_pans : ℝ := 20.0

-- Define the total revenue function
def total_revenue (num_pans : ℝ) (price_per_pan : ℝ) : ℝ := num_pans * price_per_pan

-- Define the total cost function
def total_cost (num_pans : ℝ) (cost_per_pan : ℝ) : ℝ := num_pans * cost_per_pan

-- Define the profit function as the total revenue minus the total cost
def total_profit (num_pans : ℝ) (price_per_pan : ℝ) (cost_per_pan : ℝ) : ℝ := 
  total_revenue num_pans price_per_pan - total_cost num_pans cost_per_pan

-- The statement to prove in Lean
theorem jenny_profit : total_profit num_pans price_per_pan cost_per_pan = 300.00 := 
by 
  sorry

end NUMINAMATH_GPT_jenny_profit_l217_21741


namespace NUMINAMATH_GPT_integer_to_the_fourth_l217_21733

theorem integer_to_the_fourth (a : ℤ) (h : a = 243) : 3^12 * 3^8 = a^4 :=
by {
  sorry
}

end NUMINAMATH_GPT_integer_to_the_fourth_l217_21733


namespace NUMINAMATH_GPT_units_digit_of_modifiedLucas_L20_eq_d_l217_21777

def modifiedLucas : ℕ → ℕ
| 0 => 3
| 1 => 2
| n + 2 => 2 * modifiedLucas (n + 1) + modifiedLucas n

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_of_modifiedLucas_L20_eq_d :
  ∃ d, units_digit (modifiedLucas (modifiedLucas 20)) = d :=
by
  sorry

end NUMINAMATH_GPT_units_digit_of_modifiedLucas_L20_eq_d_l217_21777


namespace NUMINAMATH_GPT_cube_edge_length_l217_21769

-- Definitions based on given conditions
def paper_cost_per_kg : ℝ := 60
def paper_area_coverage_per_kg : ℝ := 20
def total_expenditure : ℝ := 1800
def surface_area_of_cube (a : ℝ) : ℝ := 6 * a^2

-- The main proof problem
theorem cube_edge_length :
  ∃ a : ℝ, surface_area_of_cube a = paper_area_coverage_per_kg * (total_expenditure / paper_cost_per_kg) ∧ a = 10 :=
by
  sorry

end NUMINAMATH_GPT_cube_edge_length_l217_21769


namespace NUMINAMATH_GPT_randy_blocks_left_l217_21776

-- Formalize the conditions
def initial_blocks : ℕ := 78
def blocks_used_first_tower : ℕ := 19
def blocks_used_second_tower : ℕ := 25

-- Formalize the result for verification
def blocks_left : ℕ := initial_blocks - blocks_used_first_tower - blocks_used_second_tower

-- State the theorem to be proven
theorem randy_blocks_left :
  blocks_left = 34 :=
by
  -- Not providing the proof as per instructions
  sorry

end NUMINAMATH_GPT_randy_blocks_left_l217_21776


namespace NUMINAMATH_GPT_number_of_apples_and_erasers_l217_21759

def totalApplesAndErasers (a e : ℕ) : Prop :=
  a + e = 84

def applesPerFriend (a : ℕ) : ℕ :=
  a / 3

def erasersPerTeacher (e : ℕ) : ℕ :=
  e / 2

theorem number_of_apples_and_erasers (a e : ℕ) (h : totalApplesAndErasers a e) :
  applesPerFriend a = a / 3 ∧ erasersPerTeacher e = e / 2 :=
by
  sorry

end NUMINAMATH_GPT_number_of_apples_and_erasers_l217_21759


namespace NUMINAMATH_GPT_hexagons_cover_65_percent_l217_21779

noncomputable def hexagon_percent_coverage
    (a : ℝ)
    (square_area : ℝ := a^2) 
    (hexagon_area : ℝ := (3 * Real.sqrt 3 / 8 * a^2))
    (tile_pattern : ℝ := 3): Prop :=
    hexagon_area / square_area * tile_pattern = (65 / 100)

theorem hexagons_cover_65_percent (a : ℝ) : hexagon_percent_coverage a :=
by
    sorry

end NUMINAMATH_GPT_hexagons_cover_65_percent_l217_21779


namespace NUMINAMATH_GPT_house_spirits_elevator_l217_21720

-- Define the given conditions
def first_floor_domovoi := 1
def middle_floor_domovoi := 2
def last_floor_domovoi := 1
def total_floors := 7
def spirits_per_cycle := first_floor_domovoi + 5 * middle_floor_domovoi + last_floor_domovoi

-- Prove the statement
theorem house_spirits_elevator (n : ℕ) (floor : ℕ) (h1 : total_floors = 7) (h2 : spirits_per_cycle = 12) (h3 : n = 1000) :
  floor = 4 :=
by
  sorry

end NUMINAMATH_GPT_house_spirits_elevator_l217_21720


namespace NUMINAMATH_GPT_solve_equation_l217_21780

theorem solve_equation (x : ℝ) : 3 * x * (x + 3) = 2 * (x + 3) ↔ (x = -3 ∨ x = 2/3) :=
by
  sorry

end NUMINAMATH_GPT_solve_equation_l217_21780


namespace NUMINAMATH_GPT_percentage_less_than_y_l217_21738

variable (w x y z : ℝ)

-- Given conditions
variable (h1 : w = 0.60 * x)
variable (h2 : x = 0.60 * y)
variable (h3 : z = 1.50 * w)

theorem percentage_less_than_y : ( (y - z) / y) * 100 = 46 := by
  sorry

end NUMINAMATH_GPT_percentage_less_than_y_l217_21738


namespace NUMINAMATH_GPT_marcel_potatoes_eq_l217_21772

-- Define the given conditions
def marcel_corn := 10
def dale_corn := marcel_corn / 2
def dale_potatoes := 8
def total_vegetables := 27

-- Define the fact that they bought 27 vegetables in total
def total_corn := marcel_corn + dale_corn
def total_potatoes := total_vegetables - total_corn

-- State the theorem
theorem marcel_potatoes_eq :
  (total_potatoes - dale_potatoes) = 4 :=
by
  -- Lean proof would go here
  sorry

end NUMINAMATH_GPT_marcel_potatoes_eq_l217_21772


namespace NUMINAMATH_GPT_mary_number_l217_21706

-- Definitions for conditions
def has_factor_150 (m : ℕ) : Prop := 150 ∣ m
def is_multiple_of_45 (m : ℕ) : Prop := 45 ∣ m
def in_range (m : ℕ) : Prop := 1000 < m ∧ m < 3000

-- Theorem stating that Mary's number is one of {1350, 1800, 2250, 2700} given the conditions
theorem mary_number 
  (m : ℕ) 
  (h1 : has_factor_150 m)
  (h2 : is_multiple_of_45 m)
  (h3 : in_range m) :
  m = 1350 ∨ m = 1800 ∨ m = 2250 ∨ m = 2700 :=
sorry

end NUMINAMATH_GPT_mary_number_l217_21706


namespace NUMINAMATH_GPT_abe_age_is_22_l217_21701

-- Define the conditions of the problem
def abe_age_condition (A : ℕ) : Prop := A + (A - 7) = 37

-- State the theorem
theorem abe_age_is_22 : ∃ A : ℕ, abe_age_condition A ∧ A = 22 :=
by
  sorry

end NUMINAMATH_GPT_abe_age_is_22_l217_21701


namespace NUMINAMATH_GPT_floor_sum_value_l217_21763

theorem floor_sum_value (a b c d : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0) (h_pos_d : d > 0)
(h1 : a^2 + b^2 = 2016) (h2 : c^2 + d^2 = 2016) (h3 : a * c = 1024) (h4 : b * d = 1024) :
  ⌊a + b + c + d⌋ = 127 := sorry

end NUMINAMATH_GPT_floor_sum_value_l217_21763


namespace NUMINAMATH_GPT_area_of_scalene_right_triangle_l217_21735

noncomputable def area_of_triangle_DEF (DE EF : ℝ) (h1 : DE > 0) (h2 : EF > 0) (h3 : DE / EF = 3) (h4 : DE^2 + EF^2 = 16) : ℝ :=
1 / 2 * DE * EF

theorem area_of_scalene_right_triangle (DE EF : ℝ) 
  (h1 : DE > 0)
  (h2 : EF > 0)
  (h3 : DE / EF = 3)
  (h4 : DE^2 + EF^2 = 16) :
  area_of_triangle_DEF DE EF h1 h2 h3 h4 = 2.4 :=
sorry

end NUMINAMATH_GPT_area_of_scalene_right_triangle_l217_21735


namespace NUMINAMATH_GPT_basketball_game_l217_21729

variable (H E : ℕ)

theorem basketball_game (h_eq_sum : H + E = 50) (h_margin : H = E + 6) : E = 22 := by
  sorry

end NUMINAMATH_GPT_basketball_game_l217_21729


namespace NUMINAMATH_GPT_dad_gave_nickels_l217_21732

-- Definitions
def original_nickels : ℕ := 9
def total_nickels_after : ℕ := 12

-- Theorem to be proven
theorem dad_gave_nickels {original_nickels total_nickels_after : ℕ} : 
    total_nickels_after - original_nickels = 3 := 
by
  /- Sorry proof omitted -/
  sorry

end NUMINAMATH_GPT_dad_gave_nickels_l217_21732


namespace NUMINAMATH_GPT_which_is_system_lin_eq_l217_21793

def option_A : Prop := ∀ (x : ℝ), x - 1 = 2 * x
def option_B : Prop := ∀ (x y : ℝ), x - 1/y = 1
def option_C : Prop := ∀ (x z : ℝ), x + z = 3
def option_D : Prop := ∀ (x y z : ℝ), x - y + z = 1

theorem which_is_system_lin_eq (hA : option_A) (hB : option_B) (hC : option_C) (hD : option_D) :
    (∀ (x z : ℝ), x + z = 3) :=
by
  sorry

end NUMINAMATH_GPT_which_is_system_lin_eq_l217_21793


namespace NUMINAMATH_GPT_prob_exactly_one_hits_prob_at_least_one_hits_l217_21784

noncomputable def prob_A_hits : ℝ := 1 / 2
noncomputable def prob_B_hits : ℝ := 1 / 3
noncomputable def prob_A_misses : ℝ := 1 - prob_A_hits
noncomputable def prob_B_misses : ℝ := 1 - prob_B_hits

theorem prob_exactly_one_hits :
  (prob_A_hits * prob_B_misses) + (prob_A_misses * prob_B_hits) = 1 / 2 :=
by sorry

theorem prob_at_least_one_hits :
  1 - (prob_A_misses * prob_B_misses) = 2 / 3 :=
by sorry

end NUMINAMATH_GPT_prob_exactly_one_hits_prob_at_least_one_hits_l217_21784


namespace NUMINAMATH_GPT_descent_time_l217_21796

-- Definitions based on conditions
def time_to_top : ℝ := 4
def avg_speed_up : ℝ := 2.625
def avg_speed_total : ℝ := 3.5
def distance_to_top : ℝ := avg_speed_up * time_to_top -- 10.5 km
def total_distance : ℝ := 2 * distance_to_top       -- 21 km

-- Theorem statement: the time to descend (t_down) should be 2 hours
theorem descent_time (t_down : ℝ) : 
  avg_speed_total * (time_to_top + t_down) = total_distance →
  t_down = 2 := 
by 
  -- skip the proof
  sorry

end NUMINAMATH_GPT_descent_time_l217_21796


namespace NUMINAMATH_GPT_cosine_squared_is_half_l217_21794

def sides_of_triangle (p q r : ℝ) : Prop :=
  p > 0 ∧ q > 0 ∧ r > 0 ∧ p + q > r ∧ q + r > p ∧ r + p > q

noncomputable def cosine_squared (p q r : ℝ) : ℝ :=
  ((p^2 + q^2 - r^2) / (2 * p * q))^2

theorem cosine_squared_is_half (p q r : ℝ) (h : sides_of_triangle p q r) 
  (h_eq : p^4 + q^4 + r^4 = 2 * r^2 * (p^2 + q^2)) : cosine_squared p q r = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_cosine_squared_is_half_l217_21794


namespace NUMINAMATH_GPT_aluminum_atomic_weight_l217_21714

theorem aluminum_atomic_weight (Al_w : ℤ) 
  (compound_molecular_weight : ℤ) 
  (num_fluorine_atoms : ℕ) 
  (fluorine_atomic_weight : ℤ) 
  (h1 : compound_molecular_weight = 84) 
  (h2 : num_fluorine_atoms = 3) 
  (h3 : fluorine_atomic_weight = 19) :
  Al_w = 27 := 
by
  -- Proof goes here, but it is skipped.
  sorry

end NUMINAMATH_GPT_aluminum_atomic_weight_l217_21714


namespace NUMINAMATH_GPT_lana_total_winter_clothing_l217_21768

-- Define the number of boxes, scarves per box, and mittens per box as given in the conditions
def num_boxes : ℕ := 5
def scarves_per_box : ℕ := 7
def mittens_per_box : ℕ := 8

-- The total number of pieces of winter clothing is calculated as total scarves plus total mittens
def total_winter_clothing : ℕ := num_boxes * scarves_per_box + num_boxes * mittens_per_box

-- State the theorem that needs to be proven
theorem lana_total_winter_clothing : total_winter_clothing = 75 := by
  sorry

end NUMINAMATH_GPT_lana_total_winter_clothing_l217_21768


namespace NUMINAMATH_GPT_rectangle_area_l217_21774

theorem rectangle_area (w L : ℝ) (h1 : L = w^2) (h2 : L + w = 25) : 
  L * w = (Real.sqrt 101 - 1)^3 / 8 := 
sorry

end NUMINAMATH_GPT_rectangle_area_l217_21774


namespace NUMINAMATH_GPT_sum_of_shaded_cells_l217_21723

theorem sum_of_shaded_cells (a b c d e f : ℕ) 
  (h1: (a = 1 ∨ a = 2 ∨ a = 3) ∧ (b = 1 ∨ b = 2 ∨ b = 3) ∧ (c = 1 ∨ c = 2 ∨ c = 3) ∧ 
       (d = 1 ∨ d = 2 ∨ d = 3) ∧ (e = 1 ∨ e = 2 ∨ e = 3) ∧ (f = 1 ∨ f = 2 ∨ f = 3))
  (h2: (a ≠ b ∧ a ≠ c ∧ b ≠ c) ∧ 
       (d ≠ e ∧ d ≠ f ∧ e ≠ f) ∧ 
       (a ≠ d ∧ a ≠ f ∧ d ≠ f ∧ 
        b ≠ e ∧ b ≠ f ∧ c ≠ e ∧ c ≠ f))
  (h3: c = 3 ∧ d = 3 ∧ b = 2 ∧ e = 2)
  : b + e = 4 := 
sorry

end NUMINAMATH_GPT_sum_of_shaded_cells_l217_21723


namespace NUMINAMATH_GPT_mangoes_total_l217_21766

theorem mangoes_total (Dilan Ashley Alexis : ℕ) (h1 : Alexis = 4 * (Dilan + Ashley)) (h2 : Ashley = 2 * Dilan) (h3 : Alexis = 60) : Dilan + Ashley + Alexis = 75 :=
by
  sorry

end NUMINAMATH_GPT_mangoes_total_l217_21766


namespace NUMINAMATH_GPT_students_in_grades_v_vi_l217_21724

theorem students_in_grades_v_vi (n a b c p q : ℕ) (h1 : n = 100*a + 10*b + c)
  (h2 : a * b * c = p) (h3 : (p / 10) * (p % 10) = q) : n = 144 :=
sorry

end NUMINAMATH_GPT_students_in_grades_v_vi_l217_21724


namespace NUMINAMATH_GPT_point_on_graph_l217_21790

noncomputable def f (x : ℝ) : ℝ := abs (x^3 + 1) + abs (x^3 - 1)

theorem point_on_graph (a : ℝ) : ∃ (x y : ℝ), (x = a) ∧ (y = f (-a)) ∧ (y = f x) :=
by 
  sorry

end NUMINAMATH_GPT_point_on_graph_l217_21790


namespace NUMINAMATH_GPT_binom_inequality_l217_21744

-- Defining the conditions as non-computable functions
def is_nonneg_integer := ℕ

-- Defining the binomial coefficient function
noncomputable def binom (n k : ℕ) : ℕ := Nat.choose n k

-- The statement of the theorem
theorem binom_inequality (n k h : ℕ) (hn : n ≥ k + h) : binom n (k + h) ≥ binom (n - k) h :=
  sorry

end NUMINAMATH_GPT_binom_inequality_l217_21744


namespace NUMINAMATH_GPT_carpenter_additional_logs_needed_l217_21797

theorem carpenter_additional_logs_needed 
  (total_woodblocks_needed : ℕ) 
  (logs_available : ℕ) 
  (woodblocks_per_log : ℕ) 
  (additional_logs_needed : ℕ)
  (h1 : total_woodblocks_needed = 80)
  (h2 : logs_available = 8)
  (h3 : woodblocks_per_log = 5)
  (h4 : additional_logs_needed = 8) : 
  (total_woodblocks_needed - (logs_available * woodblocks_per_log)) / woodblocks_per_log = additional_logs_needed :=
by
  sorry

end NUMINAMATH_GPT_carpenter_additional_logs_needed_l217_21797


namespace NUMINAMATH_GPT_trigonometric_expression_eq_neg3_l217_21754

theorem trigonometric_expression_eq_neg3
  {α : ℝ} (h : Real.tan α = 1 / 2) :
  (1 + 2 * Real.sin (π - α) * Real.cos (-2 * π - α)) /
  ((Real.sin (-α))^2 - (Real.sin (5 * π / 2 - α))^2) = -3 :=
sorry

end NUMINAMATH_GPT_trigonometric_expression_eq_neg3_l217_21754


namespace NUMINAMATH_GPT_cans_collected_is_232_l217_21719

-- Definitions of the conditions
def total_students : ℕ := 30
def half_students : ℕ := total_students / 2
def cans_per_half_student : ℕ := 12
def remaining_students : ℕ := 13
def cans_per_remaining_student : ℕ := 4

-- Calculate total cans collected
def total_cans_collected : ℕ := (half_students * cans_per_half_student) + (remaining_students * cans_per_remaining_student)

-- The theorem to be proved
theorem cans_collected_is_232 : total_cans_collected = 232 := by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_cans_collected_is_232_l217_21719


namespace NUMINAMATH_GPT_find_smaller_number_l217_21767

theorem find_smaller_number (x : ℕ) (h : 3 * x + 4 * x = 420) : 3 * x = 180 :=
by
  sorry

end NUMINAMATH_GPT_find_smaller_number_l217_21767


namespace NUMINAMATH_GPT_statement_B_is_false_l217_21762

def heartsuit (x y : ℝ) : ℝ := x^2 - y^2

theorem statement_B_is_false (x y : ℝ) : 3 * (heartsuit x y) ≠ heartsuit (3 * x) y := by
  sorry

end NUMINAMATH_GPT_statement_B_is_false_l217_21762


namespace NUMINAMATH_GPT_suraj_average_increase_l217_21787

namespace SurajAverage

theorem suraj_average_increase (A : ℕ) (h : (16 * A + 112) / 17 = A + 6) : (A + 6) = 16 :=
  by
  sorry

end SurajAverage

end NUMINAMATH_GPT_suraj_average_increase_l217_21787


namespace NUMINAMATH_GPT_probability_of_continuous_stripe_pattern_l217_21702

def tetrahedron_stripes := 
  let faces := 4
  let configurations_per_face := 2
  2 ^ faces

def continuous_stripe_probability := 
  let total_configurations := tetrahedron_stripes
  1 / total_configurations * 4 -- Since final favorable outcomes calculation is already given and inferred to be 1/4.
  -- or any other logic that follows here based on problem description but this matches problem's derivation

theorem probability_of_continuous_stripe_pattern : continuous_stripe_probability = 1 / 4 := by
  sorry

end NUMINAMATH_GPT_probability_of_continuous_stripe_pattern_l217_21702


namespace NUMINAMATH_GPT_Joseph_has_122_socks_l217_21712

def JosephSocks : Nat := 
  let red_pairs := 9 / 2
  let white_pairs := red_pairs + 2
  let green_pairs := 2 * red_pairs
  let blue_pairs := 3 * green_pairs
  let black_pairs := blue_pairs - 5
  (red_pairs + white_pairs + green_pairs + blue_pairs + black_pairs) * 2

theorem Joseph_has_122_socks : JosephSocks = 122 := 
  by
  sorry

end NUMINAMATH_GPT_Joseph_has_122_socks_l217_21712


namespace NUMINAMATH_GPT_expression_evaluation_l217_21773

open Rat

theorem expression_evaluation :
  ∀ (a b c : ℚ),
  c = b - 4 →
  b = a + 4 →
  a = 3 →
  (a + 1 ≠ 0) →
  (b - 3 ≠ 0) →
  (c + 7 ≠ 0) →
  (a + 3) / (a + 1) * (b - 1) / (b - 3) * (c + 10) / (c + 7) = 117 / 40 :=
by
  intros a b c hc hb ha h1 h2 h3
  simp [hc, hb, ha]
  have h1 : 3 + 1 ≠ 0 := by norm_num
  have h2 : 7 - 3 ≠ 0 := by norm_num
  have h3 : 3 + 7 ≠ 0 := by norm_num
  -- Placeholder for the simplified expression computation
  sorry

end NUMINAMATH_GPT_expression_evaluation_l217_21773


namespace NUMINAMATH_GPT_p_q_false_of_not_or_l217_21749

variables (p q : Prop)

theorem p_q_false_of_not_or (h : ¬(p ∨ q)) : ¬p ∧ ¬q :=
by {
  sorry
}

end NUMINAMATH_GPT_p_q_false_of_not_or_l217_21749


namespace NUMINAMATH_GPT_base_conversion_l217_21726

theorem base_conversion (A B : ℕ) (hA : A < 8) (hB : B < 6) (h : 7 * A = 5 * B) : 8 * A + B = 47 :=
by
  sorry

end NUMINAMATH_GPT_base_conversion_l217_21726


namespace NUMINAMATH_GPT_max_f_value_l217_21709

open Real

noncomputable def problem (x1 x2 x3 : ℝ) (h1 : 0 ≤ x1 ∧ x1 ≤ 12) (h2 : 0 ≤ x2 ∧ x2 ≤ 12) (h3 : 0 ≤ x3 ∧ x3 ≤ 12) : Prop :=
  x1 * x2 * x3 = ((12 - x1) * (12 - x2) * (12 - x3))^2

theorem max_f_value (x1 x2 x3 : ℝ) (h1 : 0 ≤ x1 ∧ x1 ≤ 12) (h2 : 0 ≤ x2 ∧ x2 ≤ 12) (h3 : 0 ≤ x3 ∧ x3 ≤ 12) (h : problem x1 x2 x3 h1 h2 h3) : 
  x1 * x2 * x3 ≤ 729 :=
sorry

end NUMINAMATH_GPT_max_f_value_l217_21709


namespace NUMINAMATH_GPT_mr_wang_returns_to_start_elevator_electricity_consumption_l217_21737

-- Definition for the first part of the problem
def floor_movements : List Int := [6, -3, 10, -8, 12, -7, -10]

theorem mr_wang_returns_to_start : List.sum floor_movements = 0 := by
  -- Calculation here, we'll replace with sorry for now.
  sorry

-- Definitions for the second part of the problem
def height_per_floor : Int := 3
def electricity_per_meter : Float := 0.2

-- Calculation of electricity consumption (distance * electricity_per_meter per floor)
def total_distance_traveled : Int := 
  (floor_movements.map Int.natAbs).sum * height_per_floor

theorem elevator_electricity_consumption : 
  (Float.ofInt total_distance_traveled) * electricity_per_meter = 33.6 := by
  -- Calculation here, we'll replace with sorry for now.
  sorry

end NUMINAMATH_GPT_mr_wang_returns_to_start_elevator_electricity_consumption_l217_21737


namespace NUMINAMATH_GPT_cost_price_per_meter_l217_21781

def selling_price_for_85_meters : ℝ := 8925
def profit_per_meter : ℝ := 25
def number_of_meters : ℝ := 85

theorem cost_price_per_meter : (selling_price_for_85_meters - profit_per_meter * number_of_meters) / number_of_meters = 80 := by
  sorry

end NUMINAMATH_GPT_cost_price_per_meter_l217_21781


namespace NUMINAMATH_GPT_inequality_solution_l217_21798

theorem inequality_solution (x : ℝ) : 
  (x + 1) * (2 - x) < 0 ↔ x < -1 ∨ x > 2 := 
sorry

end NUMINAMATH_GPT_inequality_solution_l217_21798


namespace NUMINAMATH_GPT_solve_ordered_pair_l217_21745

theorem solve_ordered_pair (x y : ℝ) 
  (h1 : x + y = (7 - x) + (7 - y))
  (h2 : x^2 - y = (x - 2) + (y - 2)) :
  (x = -5 ∧ y = 12) ∨ (x = 2 ∧ y = 5) :=
  sorry

end NUMINAMATH_GPT_solve_ordered_pair_l217_21745


namespace NUMINAMATH_GPT_students_without_A_l217_21791

theorem students_without_A (total_students : ℕ) (students_english : ℕ) 
  (students_math : ℕ) (students_both : ℕ) (students_only_math : ℕ) :
  total_students = 30 → students_english = 6 → students_math = 15 → 
  students_both = 3 → students_only_math = 1 →
  (total_students - (students_math - students_only_math + 
                     students_english - students_both + 
                     students_both) = 12) :=
by sorry

end NUMINAMATH_GPT_students_without_A_l217_21791


namespace NUMINAMATH_GPT_average_visitors_per_day_l217_21715

theorem average_visitors_per_day (avg_visitors_Sunday : ℕ) (avg_visitors_other_days : ℕ) (total_days : ℕ) (starts_on_Sunday : Bool) :
  avg_visitors_Sunday = 500 → 
  avg_visitors_other_days = 140 → 
  total_days = 30 → 
  starts_on_Sunday = true → 
  (4 * avg_visitors_Sunday + 26 * avg_visitors_other_days) / total_days = 188 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_average_visitors_per_day_l217_21715


namespace NUMINAMATH_GPT_loss_percent_l217_21799

theorem loss_percent (CP SP : ℝ) (h_CP : CP = 600) (h_SP : SP = 550) :
  ((CP - SP) / CP) * 100 = 8.33 := by
  sorry

end NUMINAMATH_GPT_loss_percent_l217_21799


namespace NUMINAMATH_GPT_negation_of_at_most_one_odd_l217_21782

variable (a b c : ℕ)

def is_odd (n : ℕ) : Prop := n % 2 = 1

def at_most_one_odd (a b c : ℕ) : Prop :=
  (is_odd a ∧ ¬is_odd b ∧ ¬is_odd c) ∨
  (¬is_odd a ∧ is_odd b ∧ ¬is_odd c) ∨
  (¬is_odd a ∧ ¬is_odd b ∧ is_odd c) ∨
  (¬is_odd a ∧ ¬is_odd b ∧ ¬is_odd c)

theorem negation_of_at_most_one_odd :
  ¬ at_most_one_odd a b c ↔
  ∃ x y, (x = a ∨ x = b ∨ x = c) ∧ (y = a ∨ y = b ∨ y = c) ∧ x ≠ y ∧ is_odd x ∧ is_odd y :=
sorry

end NUMINAMATH_GPT_negation_of_at_most_one_odd_l217_21782


namespace NUMINAMATH_GPT_quadratic_function_distinct_zeros_l217_21731

theorem quadratic_function_distinct_zeros (a : ℝ) (h : a ≠ 0) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ a * x1^2 + 4 * x1 - 2 = 0 ∧ a * x2^2 + 4 * x2 - 2 = 0) ↔ (a ∈ Set.Ioo (-2) 0 ∪ Set.Ioi 0) := 
by
  sorry

end NUMINAMATH_GPT_quadratic_function_distinct_zeros_l217_21731


namespace NUMINAMATH_GPT_roja_speed_l217_21728

theorem roja_speed (R : ℕ) (h1 : 3 + R = 7) : R = 7 - 3 :=
by sorry

end NUMINAMATH_GPT_roja_speed_l217_21728


namespace NUMINAMATH_GPT_factorization_correct_l217_21727

theorem factorization_correct (x : ℝ) : 2 * x^2 - 4 * x + 2 = 2 * (x - 1)^2 :=
by sorry

end NUMINAMATH_GPT_factorization_correct_l217_21727


namespace NUMINAMATH_GPT_dice_five_prob_l217_21752

-- Define a standard six-sided die probability
def prob_five : ℚ := 1 / 6

-- Define the probability of all four dice showing five
def prob_all_five : ℚ := prob_five * prob_five * prob_five * prob_five

-- State the theorem
theorem dice_five_prob : prob_all_five = 1 / 1296 := by
  sorry

end NUMINAMATH_GPT_dice_five_prob_l217_21752


namespace NUMINAMATH_GPT_rationalize_sqrt_35_l217_21721

theorem rationalize_sqrt_35 : (35 / Real.sqrt 35) = Real.sqrt 35 :=
  sorry

end NUMINAMATH_GPT_rationalize_sqrt_35_l217_21721


namespace NUMINAMATH_GPT_cone_volume_half_sector_rolled_l217_21703

theorem cone_volume_half_sector_rolled {r slant_height h V : ℝ}
  (radius_given : r = 3)
  (height_calculated : h = 3 * Real.sqrt 3)
  (slant_height_given : slant_height = 6)
  (arc_length : 2 * Real.pi * r = 6 * Real.pi)
  (volume_formula : V = (1 / 3) * Real.pi * (r^2) * h) :
  V = 9 * Real.pi * Real.sqrt 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_cone_volume_half_sector_rolled_l217_21703


namespace NUMINAMATH_GPT_jamshid_takes_less_time_l217_21753

open Real

theorem jamshid_takes_less_time (J : ℝ) (hJ : J < 15) (h_work_rate : (1 / J) + (1 / 15) = 1 / 5) :
  (15 - J) / 15 * 100 = 50 :=
by
  sorry

end NUMINAMATH_GPT_jamshid_takes_less_time_l217_21753


namespace NUMINAMATH_GPT_isosceles_triangle_base_length_l217_21717

theorem isosceles_triangle_base_length
  (a b : ℕ)
  (ha : a = 8)
  (hp : 2 * a + b = 25)
  : b = 9 :=
by
  sorry

end NUMINAMATH_GPT_isosceles_triangle_base_length_l217_21717


namespace NUMINAMATH_GPT_change_in_profit_rate_l217_21757

theorem change_in_profit_rate (A B C : Type) (P : ℝ) (r1 r2 : ℝ) (income_increase : ℝ) (capital : ℝ) :
  (A_receives : ℝ) = (2 / 3) → 
  (B_C_divide : ℝ) = (1 - (2 / 3)) / 2 → 
  income_increase = 300 → 
  capital = 15000 →
  ((2 / 3) * capital * (r2 / 100) - (2 / 3) * capital * (r1 / 100)) = income_increase →
  (r2 - r1) = 3 :=
by
  intros
  sorry

end NUMINAMATH_GPT_change_in_profit_rate_l217_21757


namespace NUMINAMATH_GPT_number_of_boys_girls_l217_21718

-- Define the initial conditions.
def group_size : ℕ := 8
def total_ways : ℕ := 90

-- Define the actual proof problem.
theorem number_of_boys_girls 
  (n m : ℕ) 
  (h1 : n + m = group_size) 
  (h2 : Nat.choose n 2 * Nat.choose m 1 * Nat.factorial 3 = total_ways) 
  : n = 3 ∧ m = 5 :=
sorry

end NUMINAMATH_GPT_number_of_boys_girls_l217_21718


namespace NUMINAMATH_GPT_relationship_between_abc_l217_21711

open Real

-- Define the constants for the problem
noncomputable def a : ℝ := sqrt 2023 - sqrt 2022
noncomputable def b : ℝ := sqrt 2022 - sqrt 2021
noncomputable def c : ℝ := sqrt 2021 - sqrt 2020

-- State the theorem we want to prove
theorem relationship_between_abc : c > b ∧ b > a := 
sorry

end NUMINAMATH_GPT_relationship_between_abc_l217_21711


namespace NUMINAMATH_GPT_interval_of_n_l217_21748

theorem interval_of_n (n : ℕ) (h_pos : 0 < n) (h_lt_2000 : n < 2000) 
                      (h_div_99999999 : 99999999 % n = 0) (h_div_999999 : 999999 % (n + 6) = 0) : 
                      801 ≤ n ∧ n ≤ 1200 :=
by {
  sorry
}

end NUMINAMATH_GPT_interval_of_n_l217_21748


namespace NUMINAMATH_GPT_socorro_training_hours_l217_21761

theorem socorro_training_hours :
  let daily_multiplication_time := 10  -- in minutes
  let daily_division_time := 20        -- in minutes
  let training_days := 10              -- in days
  let minutes_per_hour := 60           -- minutes in an hour
  let daily_total_time := daily_multiplication_time + daily_division_time
  let total_training_time := daily_total_time * training_days
  total_training_time / minutes_per_hour = 5 :=
by sorry

end NUMINAMATH_GPT_socorro_training_hours_l217_21761


namespace NUMINAMATH_GPT_company_percentage_increase_l217_21764

/-- Company P had 426.09 employees in January and 490 employees in December.
    Prove that the percentage increase in employees from January to December is 15%. --/
theorem company_percentage_increase :
  ∀ (employees_jan employees_dec : ℝ),
  employees_jan = 426.09 → 
  employees_dec = 490 → 
  ((employees_dec - employees_jan) / employees_jan) * 100 = 15 :=
by
  intros employees_jan employees_dec h_jan h_dec
  sorry

end NUMINAMATH_GPT_company_percentage_increase_l217_21764


namespace NUMINAMATH_GPT_hyperbola_condition_l217_21765

theorem hyperbola_condition (k : ℝ) : 
  (0 < k ∧ k < 1) → ¬((k > 1 ∨ k < -2) ↔ (0 < k ∧ k < 1)) :=
by
  intro hk
  sorry

end NUMINAMATH_GPT_hyperbola_condition_l217_21765


namespace NUMINAMATH_GPT_find_x_coordinate_l217_21770

open Real

noncomputable def point_on_parabola (x y : ℝ) : Prop :=
  y^2 = 6 * x ∧ x > 0 

noncomputable def is_twice_distance (x : ℝ) : Prop :=
  let focus_x : ℝ := 3 / 2
  let d1 := x + focus_x
  let d2 := x
  d1 = 2 * d2

theorem find_x_coordinate (x y : ℝ) :
  point_on_parabola x y →
  is_twice_distance x →
  x = 3 / 2 :=
by
  intros
  sorry

end NUMINAMATH_GPT_find_x_coordinate_l217_21770


namespace NUMINAMATH_GPT_cube_sum_inequality_l217_21760

theorem cube_sum_inequality (a b : ℝ) (ha : a < 0) (hb : b < 0) : 
  a^3 + b^3 ≤ a * b^2 + a^2 * b :=
sorry

end NUMINAMATH_GPT_cube_sum_inequality_l217_21760


namespace NUMINAMATH_GPT_d_is_distance_function_l217_21778

noncomputable def d (x y : ℝ) : ℝ := |x - y| / (Real.sqrt (1 + x^2) * Real.sqrt (1 + y^2))

theorem d_is_distance_function : 
  (∀ x, d x x = 0) ∧ 
  (∀ x y, d x y = d y x) ∧ 
  (∀ x y z, d x y + d y z ≥ d x z) :=
by
  sorry

end NUMINAMATH_GPT_d_is_distance_function_l217_21778


namespace NUMINAMATH_GPT_domain_of_sqrt_quadratic_l217_21700

open Set

def domain_of_f : Set ℝ := {x : ℝ | 2*x - x^2 ≥ 0}

theorem domain_of_sqrt_quadratic :
  domain_of_f = Icc 0 2 :=
by
  sorry

end NUMINAMATH_GPT_domain_of_sqrt_quadratic_l217_21700


namespace NUMINAMATH_GPT_find_a5_l217_21788

-- Define the problem conditions within Lean
variable {a : ℕ → ℝ}
variable {q : ℝ}

-- Conditions of the problem
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) := ∀ n, a (n + 1) = a n * q
def positive_terms (a : ℕ → ℝ) := ∀ n, 0 < a n
def condition1 (a : ℕ → ℝ) := a 1 * a 3 = 4
def condition2 (a : ℕ → ℝ) := a 7 * a 9 = 25

-- Proposition to prove
theorem find_a5 :
  geometric_sequence a q →
  positive_terms a →
  condition1 a →
  condition2 a →
  a 5 = Real.sqrt 10 :=
by
  sorry

end NUMINAMATH_GPT_find_a5_l217_21788


namespace NUMINAMATH_GPT_range_of_m_l217_21713

open Real

theorem range_of_m (a b m : ℝ) (x : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 1 / a + 9 / b = 1) :
  a + b ≥ -x^2 + 4 * x + 18 - m ↔ m ≥ 6 :=
by sorry

end NUMINAMATH_GPT_range_of_m_l217_21713


namespace NUMINAMATH_GPT_initial_card_distribution_l217_21705

variables {A B C D : ℕ}

theorem initial_card_distribution 
  (total_cards : A + B + C + D = 32)
  (alfred_final : ∀ c, c = A → ((c / 2) + (c / 2)) + B + C + D = 8)
  (bruno_final : ∀ c, c = B → ((c / 2) + (c / 2)) + A + C + D = 8)
  (christof_final : ∀ c, c = C → ((c / 2) + (c / 2)) + A + B + D = 8)
  : A = 7 ∧ B = 7 ∧ C = 10 ∧ D = 8 :=
by sorry

end NUMINAMATH_GPT_initial_card_distribution_l217_21705


namespace NUMINAMATH_GPT_maximize_profit_l217_21756

noncomputable def profit (x : ℝ) : ℝ :=
  16 - 4/(x+1) - x

theorem maximize_profit (a : ℝ) (h : 0 ≤ a) :
  ∃ x : ℝ, 0 ≤ x ∧ x ≤ a ∧ profit x = max 13 (16 - 4/(a+1) - a) := by
  sorry

end NUMINAMATH_GPT_maximize_profit_l217_21756


namespace NUMINAMATH_GPT_evaluate_expression_l217_21734

theorem evaluate_expression : 500 * (500 ^ 500) * 500 = 500 ^ 502 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l217_21734


namespace NUMINAMATH_GPT_proof_m_range_l217_21783

variable {x m : ℝ}

def A (m : ℝ) : Set ℝ := {x | x^2 + x + m + 2 = 0}
def B : Set ℝ := {x | x > 0}

theorem proof_m_range (h : A m ∩ B = ∅) : m ≤ -2 := 
sorry

end NUMINAMATH_GPT_proof_m_range_l217_21783


namespace NUMINAMATH_GPT_correct_calculation_l217_21722

theorem correct_calculation (a b x y : ℝ) :
  (7 * a^2 * b - 7 * b * a^2 = 0) ∧ 
  (¬ (6 * a + 4 * b = 10 * a * b)) ∧ 
  (¬ (7 * x^2 * y - 3 * x^2 * y = 4 * x^4 * y^2)) ∧ 
  (¬ (8 * x^2 + 8 * x^2 = 16 * x^4)) :=
sorry

end NUMINAMATH_GPT_correct_calculation_l217_21722


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l217_21743

open Real

theorem sufficient_but_not_necessary_condition {x y : ℝ} :
  (x = y → |x| = |y|) ∧ (|x| = |y| → x = y) = false :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l217_21743


namespace NUMINAMATH_GPT_find_d_l217_21747

theorem find_d (a b c d : ℕ) (ha : 1 < a) (hb : 1 < b) (hc : 1 < c) (hd : 1 < d) 
  (h_eq : ∀ M : ℝ, M ≠ 1 → (M^(1/a)) * (M^(1/(a * b))) * (M^(1/(a * b * c))) * (M^(1/(a * b * c * d))) = M^(17/24)) : d = 8 :=
sorry

end NUMINAMATH_GPT_find_d_l217_21747


namespace NUMINAMATH_GPT_rectangle_area_l217_21739

theorem rectangle_area (w l : ℝ) (h_width : w = 4) (h_perimeter : 2 * l + 2 * w = 30) :
    l * w = 44 :=
by 
  sorry

end NUMINAMATH_GPT_rectangle_area_l217_21739


namespace NUMINAMATH_GPT_sum_of_ages_l217_21736

variable (S M : ℝ)  -- Variables for Sarah's and Matt's ages

-- Conditions
def sarah_older := S = M + 8
def future_age_relationship := S + 10 = 3 * (M - 5)

-- Theorem: The sum of their current ages is 41
theorem sum_of_ages (h1 : sarah_older S M) (h2 : future_age_relationship S M) : S + M = 41 := by
  sorry

end NUMINAMATH_GPT_sum_of_ages_l217_21736


namespace NUMINAMATH_GPT_count_total_coins_l217_21789

theorem count_total_coins (quarters nickels : Nat) (h₁ : quarters = 4) (h₂ : nickels = 8) : quarters + nickels = 12 :=
by sorry

end NUMINAMATH_GPT_count_total_coins_l217_21789
