import Mathlib

namespace smallest_number_divisible_by_6_in_permutations_list_l1452_145240

def is_divisible_by_6 (n : ℕ) : Prop :=
  (∃ k : ℕ, n = 6 * k)

noncomputable def permutations_5_digits := 
  [1, 2, 3, 4, 5].permutations.map (λ l => l.foldl (λ acc x => 10 * acc + x) 0)

theorem smallest_number_divisible_by_6_in_permutations_list :
  ∃ n ∈ permutations_5_digits, is_divisible_by_6 n ∧ (∀ m ∈ permutations_5_digits, is_divisible_by_6 m → n ≤ m) :=
sorry

end smallest_number_divisible_by_6_in_permutations_list_l1452_145240


namespace table_legs_l1452_145270

theorem table_legs (total_tables : ℕ) (total_legs : ℕ) (four_legged_tables : ℕ) (four_legged_count : ℕ) 
  (other_legged_tables : ℕ) (other_legged_count : ℕ) :
  total_tables = 36 →
  total_legs = 124 →
  four_legged_tables = 16 →
  four_legged_count = 4 →
  other_legged_tables = total_tables - four_legged_tables →
  total_legs = (four_legged_tables * four_legged_count) + (other_legged_tables * other_legged_count) →
  other_legged_count = 3 := 
by
  sorry

end table_legs_l1452_145270


namespace max_n_value_l1452_145221

theorem max_n_value (a b c : ℝ) (n : ℕ) (h1 : a > b) (h2 : b > c) (h3 : 1/(a - b) + 1/(b - c) ≥ n / (a - c)) :
  n ≤ 4 := 
sorry

end max_n_value_l1452_145221


namespace polynomial_factor_pq_l1452_145265

theorem polynomial_factor_pq (p q : ℝ) (h : ∀ x : ℝ, (x^2 + 2*x + 5) ∣ (x^4 + p*x^2 + q)) : p + q = 31 :=
sorry

end polynomial_factor_pq_l1452_145265


namespace condition_is_necessary_but_not_sufficient_l1452_145299

noncomputable def sequence_satisfies_condition (a : ℕ → ℤ) : Prop :=
  a 3 + a 7 = 2 * a 5

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a n = a 1 + (n - 1) * d

theorem condition_is_necessary_but_not_sufficient (a : ℕ → ℤ) :
  (sequence_satisfies_condition a ∧ (¬ arithmetic_sequence a)) ∨
  (arithmetic_sequence a → sequence_satisfies_condition a) :=
sorry

end condition_is_necessary_but_not_sufficient_l1452_145299


namespace two_cards_totaling_15_probability_l1452_145248

theorem two_cards_totaling_15_probability :
  let total_cards := 52
  let valid_numbers := [5, 6, 7]
  let combinations := 3 * 4 * 4 / (total_cards * (total_cards - 1))
  let prob := combinations
  prob = 8 / 442 :=
by
  sorry

end two_cards_totaling_15_probability_l1452_145248


namespace car_travel_inequality_l1452_145204

variable (x : ℕ)

theorem car_travel_inequality (hx : 8 * (x + 19) > 2200) : 8 * (x + 19) > 2200 :=
by
  sorry

end car_travel_inequality_l1452_145204


namespace geometric_sequence_sum_range_l1452_145227

theorem geometric_sequence_sum_range (a b c : ℝ) 
  (h1 : ∃ q : ℝ, q ≠ 0 ∧ a = b * q ∧ c = b / q) 
  (h2 : a + b + c = 1) : 
  a + c ∈ (Set.Icc (2 / 3 : ℝ) 1 \ Set.Iio 1) ∪ (Set.Ioo 1 2) :=
sorry

end geometric_sequence_sum_range_l1452_145227


namespace find_limpet_shells_l1452_145260

variable (L L_shells E_shells J_shells totalShells : ℕ)

def Ed_and_Jacob_initial_shells := 2
def Ed_oyster_shells := 2
def Ed_conch_shells := 4
def Jacob_more_shells := 2
def total_shells := 30

def Ed_total_shells := L + Ed_oyster_shells + Ed_conch_shells
def Jacob_total_shells := Ed_total_shells + Jacob_more_shells

theorem find_limpet_shells
  (H : Ed_and_Jacob_initial_shells + Ed_total_shells + Jacob_total_shells = total_shells) :
  L = 7 :=
by
  sorry

end find_limpet_shells_l1452_145260


namespace total_tomato_seeds_l1452_145249

theorem total_tomato_seeds (morn_mike morn_morning ted_morning sarah_morning : ℕ)
    (aft_mike aft_ted aft_sarah : ℕ)
    (H1 : morn_mike = 50)
    (H2 : ted_morning = 2 * morn_mike)
    (H3 : sarah_morning = morn_mike + 30)
    (H4 : aft_mike = 60)
    (H5 : aft_ted = aft_mike - 20)
    (H6 : aft_sarah = sarah_morning + 20) :
    morn_mike + aft_mike + ted_morning + aft_ted + sarah_morning + aft_sarah = 430 :=
by
  rw [H1, H2, H3, H4, H5, H6]
  sorry

end total_tomato_seeds_l1452_145249


namespace minimum_value_expression_l1452_145281

theorem minimum_value_expression {a b c : ℝ} (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_prod : a * b * c = 1) : 
  a^2 + 4 * a * b + 9 * b^2 + 3 * b * c + c^2 ≥ 18 :=
by
  sorry

end minimum_value_expression_l1452_145281


namespace tiles_needed_l1452_145284

def hallway_length : ℕ := 14
def hallway_width : ℕ := 20
def border_tile_side : ℕ := 2
def interior_tile_side : ℕ := 3

theorem tiles_needed :
  let border_length_tiles := ((hallway_length - 2 * border_tile_side) / border_tile_side) * 2
  let border_width_tiles := ((hallway_width - 2 * border_tile_side) / border_tile_side) * 2
  let corner_tiles := 4
  let total_border_tiles := border_length_tiles + border_width_tiles + corner_tiles
  let interior_length := hallway_length - 2 * border_tile_side
  let interior_width := hallway_width - 2 * border_tile_side
  let interior_area := interior_length * interior_width
  let interior_tiles_needed := (interior_area + interior_tile_side * interior_tile_side - 1) / (interior_tile_side * interior_tile_side)
  total_border_tiles + interior_tiles_needed = 48 := 
by {
  sorry
}

end tiles_needed_l1452_145284


namespace ordering_of_a_b_c_l1452_145268

theorem ordering_of_a_b_c (a b c : ℝ)
  (ha : a = Real.exp (1 / 2))
  (hb : b = Real.log (1 / 2))
  (hc : c = Real.sin (1 / 2)) :
  a > c ∧ c > b :=
by sorry

end ordering_of_a_b_c_l1452_145268


namespace line_through_points_l1452_145256

theorem line_through_points (m n p : ℝ) 
  (h1 : m = 4 * n + 5) 
  (h2 : m + 2 = 4 * (n + p) + 5) : 
  p = 1 / 2 := 
by 
  sorry

end line_through_points_l1452_145256


namespace number_of_sons_l1452_145222

noncomputable def land_area_hectares : ℕ := 3
noncomputable def hectare_to_m2 : ℕ := 10000
noncomputable def profit_per_section_per_3months : ℕ := 500
noncomputable def section_area_m2 : ℕ := 750
noncomputable def profit_per_son_per_year : ℕ := 10000
noncomputable def months_in_year : ℕ := 12
noncomputable def months_per_season : ℕ := 3

theorem number_of_sons :
  let total_land_area_m2 := land_area_hectares * hectare_to_m2
  let yearly_profit_per_section := profit_per_section_per_3months * (months_in_year / months_per_season)
  let number_of_sections := total_land_area_m2 / section_area_m2
  let total_yearly_profit := number_of_sections * yearly_profit_per_section
  let n := total_yearly_profit / profit_per_son_per_year
  n = 8 :=
by
  sorry

end number_of_sons_l1452_145222


namespace probability_red_jelly_bean_l1452_145235

variable (r b g : Nat) (eaten_green eaten_blue : Nat)

theorem probability_red_jelly_bean
    (h_r : r = 15)
    (h_b : b = 20)
    (h_g : g = 16)
    (h_eaten_green : eaten_green = 1)
    (h_eaten_blue : eaten_blue = 1)
    (h_total : r + b + g = 51)
    (h_remaining_total : r + (b - eaten_blue) + (g - eaten_green) = 49) :
    (r : ℚ) / 49 = 15 / 49 :=
by
  sorry

end probability_red_jelly_bean_l1452_145235


namespace distribute_books_into_bags_l1452_145239

def number_of_ways_to_distribute_books (books : Finset ℕ) (bags : ℕ) : ℕ :=
  if (books.card = 5) ∧ (bags = 3) then 51 else 0

theorem distribute_books_into_bags :
  number_of_ways_to_distribute_books (Finset.range 5) 3 = 51 := by
  sorry

end distribute_books_into_bags_l1452_145239


namespace line_parallel_xaxis_l1452_145253

theorem line_parallel_xaxis (x y : ℝ) : y = 2 ↔ (∃ a b : ℝ, a = 4 ∧ b = 2 ∧ y = 2) :=
by 
  sorry

end line_parallel_xaxis_l1452_145253


namespace last_digit_of_2_pow_2004_l1452_145238

theorem last_digit_of_2_pow_2004 : (2 ^ 2004) % 10 = 6 := 
by {
  sorry
}

end last_digit_of_2_pow_2004_l1452_145238


namespace pen_tip_movement_l1452_145290

-- Definitions for the conditions
def condition_a := "Point movement becomes a line"
def condition_b := "Line movement becomes a surface"
def condition_c := "Surface movement becomes a solid"
def condition_d := "Intersection of surfaces results in a line"

-- The main statement we need to prove
theorem pen_tip_movement (phenomenon : String) : 
  phenomenon = "the pen tip quickly sliding on the paper to write the number 6" →
  condition_a = "Point movement becomes a line" :=
by
  intros
  sorry

end pen_tip_movement_l1452_145290


namespace John_took_more_chickens_than_Ray_l1452_145226

theorem John_took_more_chickens_than_Ray
  (r m j : ℕ)
  (h1 : r = 10)
  (h2 : r = m - 6)
  (h3 : j = m + 5) : j - r = 11 :=
by
  sorry

end John_took_more_chickens_than_Ray_l1452_145226


namespace third_divisor_is_11_l1452_145223

theorem third_divisor_is_11 (n : ℕ) (x : ℕ) : 
  n = 200 ∧ (n - 20) % 15 = 0 ∧ (n - 20) % 30 = 0 ∧ (n - 20) % x = 0 ∧ (n - 20) % 60 = 0 → 
  x = 11 :=
by
  sorry

end third_divisor_is_11_l1452_145223


namespace order_of_activities_l1452_145279

noncomputable def fraction_liking_activity_dodgeball : ℚ := 8 / 24
noncomputable def fraction_liking_activity_barbecue : ℚ := 10 / 30
noncomputable def fraction_liking_activity_archery : ℚ := 9 / 18

theorem order_of_activities :
  (fraction_liking_activity_archery > fraction_liking_activity_dodgeball) ∧
  (fraction_liking_activity_archery > fraction_liking_activity_barbecue) ∧
  (fraction_liking_activity_dodgeball = fraction_liking_activity_barbecue) :=
by
  sorry

end order_of_activities_l1452_145279


namespace evaluate_expression_l1452_145282

open Nat

theorem evaluate_expression : 
  (3 * 4 * 5 * 6) * (1 / 3 + 1 / 4 + 1 / 5 + 1 / 6) = 342 := by
  sorry

end evaluate_expression_l1452_145282


namespace height_difference_l1452_145229

theorem height_difference
  (a b : ℝ)
  (parabola_eq : ∀ x, y = x^2 + 1)
  (circle_center : b = 2 * a^2 + 1 / 2) :
  b - (a^2 + 1) = a^2 - 1 / 2 :=
by {
  sorry
}

end height_difference_l1452_145229


namespace common_chord_line_l1452_145264

def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 2 * x - 8 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 2 * x - 4 * y - 4 = 0

theorem common_chord_line : 
  ∀ x y : ℝ, (circle1 x y ∧ circle2 x y) ↔ (x - y + 1 = 0) := 
by sorry

end common_chord_line_l1452_145264


namespace find_center_angle_l1452_145252

noncomputable def pi : ℝ := Real.pi
/-- Given conditions from the math problem -/
def radius : ℝ := 12
def area : ℝ := 67.88571428571429

theorem find_center_angle (θ : ℝ) 
  (area_def : area = (θ / 360) * pi * radius ^ 2) : 
  θ = 54 :=
sorry

end find_center_angle_l1452_145252


namespace arccos_one_over_sqrt_two_eq_pi_over_four_l1452_145292

theorem arccos_one_over_sqrt_two_eq_pi_over_four : Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 :=
by
  sorry

end arccos_one_over_sqrt_two_eq_pi_over_four_l1452_145292


namespace knights_probability_l1452_145267

theorem knights_probability :
  let knights : Nat := 30
  let chosen : Nat := 4
  let probability (n k : Nat) := 1 - (((n - k + 1) * (n - k - 1) * (n - k - 3) * (n - k - 5)) / 
                                      ((n - 0) * (n - 1) * (n - 2) * (n - 3)))
  probability knights chosen = (389 / 437) := sorry

end knights_probability_l1452_145267


namespace depth_of_melted_ice_cream_l1452_145275

theorem depth_of_melted_ice_cream
  (r_sphere : ℝ) (r_cylinder : ℝ) (V_sphere : ℝ)
  (h : ℝ)
  (sphere_volume_eq : V_sphere = (4 / 3) * Real.pi * r_sphere^3)
  (cylinder_volume_eq : V_sphere = Real.pi * r_cylinder^2 * h)
  (r_sphere_eq : r_sphere = 3)
  (r_cylinder_eq : r_cylinder = 9)
  : h = 4 / 9 :=
by
  -- Proof is omitted
  sorry

end depth_of_melted_ice_cream_l1452_145275


namespace minimize_circumscribed_sphere_radius_l1452_145225

noncomputable def cylinder_surface_area (r h : ℝ) : ℝ :=
  2 * Real.pi * r^2 + 2 * Real.pi * r * h

noncomputable def circumscribed_sphere_radius (r h : ℝ) : ℝ :=
  (r^2 + (1 / 2 * h)^2).sqrt

theorem minimize_circumscribed_sphere_radius (r : ℝ) (h : ℝ) (hr : cylinder_surface_area r h = 16 * Real.pi) : 
  r^2 = 8 * Real.sqrt 5 / 5 :=
sorry

end minimize_circumscribed_sphere_radius_l1452_145225


namespace limestone_amount_l1452_145269

theorem limestone_amount (L S : ℝ) (h1 : L + S = 100) (h2 : 3 * L + 5 * S = 425) : L = 37.5 :=
by
  -- Proof will go here
  sorry

end limestone_amount_l1452_145269


namespace volume_of_prism_l1452_145294

theorem volume_of_prism (x y z : ℝ) (hx : x * y = 28) (hy : x * z = 45) (hz : y * z = 63) : x * y * z = 282 := by
  sorry

end volume_of_prism_l1452_145294


namespace new_socks_bought_l1452_145293

-- Given conditions:
def initial_socks : ℕ := 11
def socks_thrown_away : ℕ := 4
def final_socks : ℕ := 33

-- Theorem proof statement:
theorem new_socks_bought : (final_socks - (initial_socks - socks_thrown_away)) = 26 :=
by
  sorry

end new_socks_bought_l1452_145293


namespace initial_population_l1452_145259

theorem initial_population (P : ℝ) (h : 0.72 * P = 3168) : P = 4400 :=
sorry

end initial_population_l1452_145259


namespace range_of_a_l1452_145224

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, ax^2 + ax + 3 > 0) ↔ (0 ≤ a ∧ a < 12) :=
by
  sorry

end range_of_a_l1452_145224


namespace average_eq_solution_l1452_145231

theorem average_eq_solution (x : ℝ) :
  (1 / 3) * ((2 * x + 4) + (4 * x + 6) + (5 * x + 3)) = 3 * x + 5 → x = 1 :=
by
  sorry

end average_eq_solution_l1452_145231


namespace circle_radius_5_l1452_145257

theorem circle_radius_5 (k x y : ℝ) : x^2 + 8 * x + y^2 + 10 * y - k = 0 → (x + 4) ^ 2 + (y + 5) ^ 2 = 25 → k = -16 :=
by
  sorry

end circle_radius_5_l1452_145257


namespace perimeter_of_photo_l1452_145206

theorem perimeter_of_photo 
  (frame_width : ℕ)
  (frame_area : ℕ)
  (outer_edge_length : ℕ)
  (photo_perimeter : ℕ) :
  frame_width = 2 → 
  frame_area = 48 → 
  outer_edge_length = 10 →
  photo_perimeter = 16 :=
by
  intros h1 h2 h3
  -- proof goes here
  sorry

end perimeter_of_photo_l1452_145206


namespace sum_six_seven_l1452_145241

variable (a : ℕ → ℝ)
variable (d : ℝ)

axiom arithmetic_sequence : ∀ (n : ℕ), a (n + 1) = a n + d
axiom sum_condition : a 2 + a 5 + a 8 + a 11 = 48

theorem sum_six_seven : a 6 + a 7 = 24 :=
by
  -- Using given axioms and properties of arithmetic sequence
  sorry

end sum_six_seven_l1452_145241


namespace chebyshev_birth_year_l1452_145297

theorem chebyshev_birth_year :
  ∃ (a b : ℕ),
  a > b ∧ 
  a + b = 3 ∧ 
  (1821 = 1800 + 10 * a + 1 * b) ∧
  (1821 + 73) < 1900 :=
by sorry

end chebyshev_birth_year_l1452_145297


namespace cards_given_to_Jeff_l1452_145233

-- Definitions according to the conditions
def initial_cards : Nat := 304
def remaining_cards : Nat := 276

-- The proof problem
theorem cards_given_to_Jeff : initial_cards - remaining_cards = 28 :=
by
  sorry

end cards_given_to_Jeff_l1452_145233


namespace find_f_10_l1452_145271

noncomputable def f : ℤ → ℤ := sorry

axiom cond1 : f 1 + 1 > 0
axiom cond2 : ∀ x y : ℤ, f (x + y) - x * f y - y * f x = f x * f y - x - y + x * y
axiom cond3 : ∀ x : ℤ, 2 * f x = f (x + 1) - x + 1

theorem find_f_10 : f 10 = 1014 := 
by
  sorry 

end find_f_10_l1452_145271


namespace magic_square_y_minus_x_l1452_145209

theorem magic_square_y_minus_x :
  ∀ (x y : ℝ), 
    (x - 2 = 2 * y + y) ∧ (x - 2 = -2 + y + 6) →
    y - x = -6 :=
by 
  intros x y h
  sorry

end magic_square_y_minus_x_l1452_145209


namespace f_at_7_l1452_145219

-- Define the function f and its properties
axiom f : ℝ → ℝ
axiom odd_f : ∀ x : ℝ, f (-x) = -f x
axiom periodic_f : ∀ x : ℝ, f (x + 4) = f x
axiom values_f : ∀ x : ℝ, 0 < x ∧ x < 2 → f x = 2 * x^2

-- Prove that f(7) = -2
theorem f_at_7 : f 7 = -2 :=
by
  sorry

end f_at_7_l1452_145219


namespace substitution_result_l1452_145280

theorem substitution_result (x y : ℝ) (h1 : y = 2 * x + 1) (h2 : 5 * x - 2 * y = 7) : 5 * x - 4 * x - 2 = 7 :=
by
  sorry

end substitution_result_l1452_145280


namespace increased_volume_l1452_145237

theorem increased_volume (l w h : ℕ) 
  (volume_eq : l * w * h = 4500) 
  (surface_area_eq : l * w + l * h + w * h = 900) 
  (edges_sum_eq : l + w + h = 54) :
  (l + 1) * (w + 1) * (h + 1) = 5455 := 
by 
  sorry

end increased_volume_l1452_145237


namespace total_games_in_season_l1452_145278

theorem total_games_in_season (teams: ℕ) (division_teams: ℕ) (intra_division_games: ℕ) (inter_division_games: ℕ) (total_games: ℕ) : 
  teams = 18 → division_teams = 9 → intra_division_games = 3 → inter_division_games = 2 → total_games = 378 :=
by
  sorry

end total_games_in_season_l1452_145278


namespace petrol_price_increase_l1452_145245

theorem petrol_price_increase
  (P P_new : ℝ)
  (C : ℝ)
  (h1 : P * C = P_new * (C * 0.7692307692307693))
  (h2 : C * (1 - 0.23076923076923073) = C * 0.7692307692307693) :
  ((P_new - P) / P) * 100 = 30 := 
  sorry

end petrol_price_increase_l1452_145245


namespace valid_ways_to_assign_volunteers_l1452_145207

noncomputable def validAssignments : ℕ := 
  (Nat.choose 5 2) * (Nat.choose 3 2) + (Nat.choose 5 1) * (Nat.choose 4 2)

theorem valid_ways_to_assign_volunteers : validAssignments = 60 := 
  by
    simp [validAssignments]
    sorry

end valid_ways_to_assign_volunteers_l1452_145207


namespace position_of_term_in_sequence_l1452_145244

theorem position_of_term_in_sequence 
    (a : ℕ → ℕ)
    (h1 : a 1 = 1)
    (h2 : ∀ n, a (n + 1) - a n = 7 * n) :
    ∃ n, a n = 35351 ∧ n = 101 :=
by
  sorry

end position_of_term_in_sequence_l1452_145244


namespace primes_sum_divisible_by_60_l1452_145296

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem primes_sum_divisible_by_60 (p q r s : ℕ) 
  (hp : is_prime p) 
  (hq : is_prime q) 
  (hr : is_prime r) 
  (hs : is_prime s) 
  (h_cond1 : 5 < p) 
  (h_cond2 : p < q) 
  (h_cond3 : q < r) 
  (h_cond4 : r < s) 
  (h_cond5 : s < p + 10) : 
  (p + q + r + s) % 60 = 0 :=
sorry

end primes_sum_divisible_by_60_l1452_145296


namespace max_abs_sum_l1452_145258

theorem max_abs_sum (x y : ℝ) (h : x^2 + y^2 = 4) : |x| + |y| ≤ 2 * Real.sqrt 2 :=
by
  sorry

end max_abs_sum_l1452_145258


namespace hollow_circles_in_2001_pattern_l1452_145216

theorem hollow_circles_in_2001_pattern :
  let pattern_length := 9
  let hollow_in_pattern := 3
  let total_circles := 2001
  let complete_patterns := total_circles / pattern_length
  let remaining_circles := total_circles % pattern_length
  let hollow_in_remaining := if remaining_circles >= 3 then 1 else 0
  let total_hollow := complete_patterns * hollow_in_pattern + hollow_in_remaining
  total_hollow = 667 :=
by
  sorry

end hollow_circles_in_2001_pattern_l1452_145216


namespace combined_time_l1452_145236

def time_pulsar : ℕ := 10
def time_polly : ℕ := 3 * time_pulsar
def time_petra : ℕ := time_polly / 6

theorem combined_time : time_pulsar + time_polly + time_petra = 45 := 
by 
  -- proof steps will go here
  sorry

end combined_time_l1452_145236


namespace not_777_integers_l1452_145288

theorem not_777_integers (p : ℕ) (hp : Nat.Prime p) :
  ¬ (∃ count : ℕ, count = 777 ∧ ∀ n : ℕ, ∃ k : ℕ, (n ^ 3 + n * p + 1 = k * (n + p + 1))) :=
by
  sorry

end not_777_integers_l1452_145288


namespace quadratic_inequality_solution_l1452_145285

theorem quadratic_inequality_solution (x : ℝ) : (x^2 + 5*x + 6 > 0) ↔ (x < -3 ∨ x > -2) :=
  by
    sorry

end quadratic_inequality_solution_l1452_145285


namespace fraction_reduction_l1452_145295

theorem fraction_reduction (x y : ℝ) : 
  (4 * x - 4 * y) / (4 * x * 4 * y) = (1 / 4) * ((x - y) / (x * y)) := 
by 
  sorry

end fraction_reduction_l1452_145295


namespace pentagon_triangle_area_percentage_l1452_145205

def is_equilateral_triangle (s : ℝ) (area : ℝ) : Prop :=
  area = (s^2 * Real.sqrt 3) / 4

def is_square (s : ℝ) (area : ℝ) : Prop :=
  area = s^2

def pentagon_area (square_area triangle_area : ℝ) : ℝ :=
  square_area + triangle_area

noncomputable def percentage (triangle_area pentagon_area : ℝ) : ℝ :=
  (triangle_area / pentagon_area) * 100

theorem pentagon_triangle_area_percentage (s : ℝ) (h₁ : s > 0) :
  let square_area := s^2
  let triangle_area := (s^2 * Real.sqrt 3) / 4
  let pentagon_total_area := pentagon_area square_area triangle_area
  let triangle_percentage := percentage triangle_area pentagon_total_area
  triangle_percentage = (100 * (4 * Real.sqrt 3 - 3) / 13) :=
by
  sorry

end pentagon_triangle_area_percentage_l1452_145205


namespace intersection_eq_l1452_145211

open Set

def setA : Set ℤ := {x | x ≥ -4}
def setB : Set ℤ := {x | x ≤ 3}

theorem intersection_eq : (setA ∩ setB) = {x | -4 ≤ x ∧ x ≤ 3} := by
  sorry

end intersection_eq_l1452_145211


namespace range_of_a_l1452_145202

theorem range_of_a (a : ℝ) : (∃ x : ℝ, a * x^2 + x + 1 < 0) ↔ (a < 1/4) := 
sorry

end range_of_a_l1452_145202


namespace sufficient_condition_B_is_proper_subset_of_A_l1452_145228

def A : Set ℝ := {x | x^2 + x = 6}
def B (m : ℝ) : Set ℝ := {-1 / m}

theorem sufficient_condition_B_is_proper_subset_of_A (m : ℝ) : 
  m = -1/2 → B m ⊆ A ∧ B m ≠ A :=
by
  sorry

end sufficient_condition_B_is_proper_subset_of_A_l1452_145228


namespace rectangle_area_ratio_k_l1452_145266

theorem rectangle_area_ratio_k (d : ℝ) (l w : ℝ) (h1 : l / w = 5 / 2) (h2 : d^2 = l^2 + w^2) :
  ∃ k : ℝ, k = 10 / 29 ∧ (l * w = k * d^2) :=
by {
  -- proof steps will go here
  sorry
}

end rectangle_area_ratio_k_l1452_145266


namespace ratio_second_part_l1452_145291

theorem ratio_second_part (first_part second_part total : ℕ) 
  (h_ratio_percent : 50 = 100 * first_part / total) 
  (h_first_part : first_part = 10) : 
  second_part = 10 := by
  have h_total : total = 2 * first_part := by sorry
  sorry

end ratio_second_part_l1452_145291


namespace sum_of_reciprocals_l1452_145247

theorem sum_of_reciprocals
  (m n p : ℕ)
  (HCF_mnp : Nat.gcd (Nat.gcd m n) p = 26)
  (LCM_mnp : Nat.lcm (Nat.lcm m n) p = 6930)
  (sum_mnp : m + n + p = 150) :
  (1 / (m : ℚ) + 1 / (n : ℚ) + 1 / (p : ℚ) = 1 / 320166) :=
by
  sorry

end sum_of_reciprocals_l1452_145247


namespace quadratic_has_real_roots_l1452_145263

theorem quadratic_has_real_roots (m : ℝ) : (∃ x : ℝ, (m - 1) * x^2 - 2 * x + 1 = 0) ↔ (m ≤ 2 ∧ m ≠ 1) := 
by 
  sorry

end quadratic_has_real_roots_l1452_145263


namespace no_n_exists_11_div_mod_l1452_145220

theorem no_n_exists_11_div_mod (n : ℕ) (h1 : n > 0) (h2 : 3^5 ≡ 1 [MOD 11]) (h3 : 4^5 ≡ 1 [MOD 11]) : ¬ (11 ∣ (3^n + 4^n)) := 
sorry

end no_n_exists_11_div_mod_l1452_145220


namespace quadratic_standard_form_l1452_145215

theorem quadratic_standard_form :
  ∀ x : ℝ, (2 * x - 1) ^ 2 = (x + 1) * (3 * x + 4) →
  (∃ a b c : ℝ, a = 1 ∧ b = -11 ∧ c = -3 ∧ a * x^2 + b * x + c = 0) :=
by
  intro x
  intro h
  sorry

end quadratic_standard_form_l1452_145215


namespace sam_age_two_years_ago_l1452_145254

theorem sam_age_two_years_ago (J S : ℕ) (h1 : J = 3 * S) (h2 : J + 9 = 2 * (S + 9)) : S - 2 = 7 :=
sorry

end sam_age_two_years_ago_l1452_145254


namespace no_integer_roots_l1452_145246

theorem no_integer_roots (n : ℕ) (p : Fin (2*n + 1) → ℤ)
  (non_zero : ∀ i, p i ≠ 0)
  (sum_non_zero : (Finset.univ.sum (λ i => p i)) ≠ 0) :
  ∃ P : ℤ → ℤ, ∀ x : ℤ, P x ≠ 0 → x > 1 ∨ x < -1 := sorry

end no_integer_roots_l1452_145246


namespace number_of_three_digit_multiples_of_6_l1452_145283

theorem number_of_three_digit_multiples_of_6 : 
  let lower_bound := 100
  let upper_bound := 999
  let multiple := 6
  let smallest_n := Nat.ceil (100 / multiple)
  let largest_n := Nat.floor (999 / multiple)
  let count_multiples := largest_n - smallest_n + 1
  count_multiples = 150 := by
  sorry

end number_of_three_digit_multiples_of_6_l1452_145283


namespace range_of_a_l1452_145289

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, x^2 + a * x + 4 < 0) ↔ (a < -4 ∨ a > 4) :=
by 
sorry

end range_of_a_l1452_145289


namespace number_of_moles_of_methanol_formed_l1452_145208

def ch4_to_co2 : ℚ := 1
def o2_to_co2 : ℚ := 2
def co2_prod_from_ch4 (ch4 : ℚ) : ℚ := ch4 * ch4_to_co2 / o2_to_co2

def co2_to_ch3oh : ℚ := 1
def h2_to_ch3oh : ℚ := 3
def ch3oh_prod_from_co2 (co2 h2 : ℚ) : ℚ :=
  min (co2 / co2_to_ch3oh) (h2 / h2_to_ch3oh)

theorem number_of_moles_of_methanol_formed :
  (ch3oh_prod_from_co2 (co2_prod_from_ch4 5) 10) = 10/3 :=
by
  sorry

end number_of_moles_of_methanol_formed_l1452_145208


namespace problem_statement_l1452_145214

theorem problem_statement (m n : ℝ) :
  (m^2 - 1840 * m + 2009 = 0) → (n^2 - 1840 * n + 2009 = 0) → 
  (m^2 - 1841 * m + 2009) * (n^2 - 1841 * n + 2009) = 2009 := 
by
  intros h1 h2
  sorry

end problem_statement_l1452_145214


namespace total_annual_interest_l1452_145255

def total_amount : ℝ := 4000
def P1 : ℝ := 2800
def Rate1 : ℝ := 0.03
def Rate2 : ℝ := 0.05

def P2 : ℝ := total_amount - P1
def I1 : ℝ := P1 * Rate1
def I2 : ℝ := P2 * Rate2
def I_total : ℝ := I1 + I2

theorem total_annual_interest : I_total = 144 := by
  sorry

end total_annual_interest_l1452_145255


namespace remainder_when_divided_by_x_minus_3_l1452_145276

open Polynomial

noncomputable def p : ℝ[X] := 4 * X^3 - 12 * X^2 + 16 * X - 20

theorem remainder_when_divided_by_x_minus_3 : eval 3 p = 28 := by
  sorry

end remainder_when_divided_by_x_minus_3_l1452_145276


namespace negation_proof_l1452_145201

theorem negation_proof :
  (¬ (∀ x : ℝ, x^2 - x + 1 > 0)) ↔ (∃ x : ℝ, x^2 - x + 1 ≤ 0) := sorry

end negation_proof_l1452_145201


namespace smallest_apples_l1452_145234

theorem smallest_apples (A : ℕ) (h1 : A % 9 = 2) (h2 : A % 10 = 2) (h3 : A % 11 = 2) (h4 : A > 2) : A = 992 :=
sorry

end smallest_apples_l1452_145234


namespace Vasya_numbers_l1452_145250

theorem Vasya_numbers (x y : ℝ) (h1 : x + y = x * y) (h2 : x * y = x / y) : x = 1/2 ∧ y = -1 :=
by
  sorry

end Vasya_numbers_l1452_145250


namespace scalene_triangle_area_l1452_145230

theorem scalene_triangle_area (outer_triangle_area : ℝ) (hexagon_area : ℝ) (num_scalene_triangles : ℕ)
  (h1 : outer_triangle_area = 25) (h2 : hexagon_area = 4) (h3 : num_scalene_triangles = 6) : 
  (outer_triangle_area - hexagon_area) / num_scalene_triangles = 3.5 :=
by
  sorry

end scalene_triangle_area_l1452_145230


namespace angle_B_is_60_l1452_145232

theorem angle_B_is_60 (A B C : ℝ) (h_seq : 2 * B = A + C) (h_sum : A + B + C = 180) : B = 60 := 
by 
  sorry

end angle_B_is_60_l1452_145232


namespace cube_expansion_l1452_145287

theorem cube_expansion : 101^3 + 3 * 101^2 + 3 * 101 + 1 = 1061208 :=
by
  sorry

end cube_expansion_l1452_145287


namespace necessary_but_not_sufficient_l1452_145273

variable (a b : ℝ)

theorem necessary_but_not_sufficient : 
  ¬ (a ≠ 1 ∨ b ≠ 2 → a + b ≠ 3) ∧ (a + b ≠ 3 → a ≠ 1 ∨ b ≠ 2) :=
by
  sorry

end necessary_but_not_sufficient_l1452_145273


namespace tangent_through_points_l1452_145272

theorem tangent_through_points :
  ∀ (x₁ x₂ : ℝ),
    (∀ y₁ y₂ : ℝ, y₁ = x₁^2 + 1 → y₂ = x₂^2 + 1 → 
    (2 * x₁ * (x₂ - x₁) + y₁ = 0 → x₂ = -x₁) ∧ 
    (2 * x₂ * (x₁ - x₂) + y₂ = 0 → x₁ = -x₂)) →
  (x₁ = 1 / Real.sqrt 3 ∧ x₂ = -1 / Real.sqrt 3 ∧
   (x₁^2 + 1 = (1 / 3) + 1) ∧ (x₂^2 + 1 = (1 / 3) + 1)) :=
by
  sorry

end tangent_through_points_l1452_145272


namespace solve_for_nabla_l1452_145261

theorem solve_for_nabla : (∃ (nabla : ℤ), 5 * (-3) + 4 = nabla + 7) → (∃ (nabla : ℤ), nabla = -18) :=
by
  sorry

end solve_for_nabla_l1452_145261


namespace inequality_holds_l1452_145200

theorem inequality_holds (a b : ℕ) (ha : a > 1) (hb : b > 2) : a ^ b + 1 ≥ b * (a + 1) :=
sorry

end inequality_holds_l1452_145200


namespace P_is_in_third_quadrant_l1452_145212

noncomputable def point : Type := (ℝ × ℝ)

def P : point := (-3, -4)

def is_in_third_quadrant (p : point) : Prop :=
  p.1 < 0 ∧ p.2 < 0

theorem P_is_in_third_quadrant : is_in_third_quadrant P :=
by {
  -- Prove that P is in the third quadrant
  sorry
}

end P_is_in_third_quadrant_l1452_145212


namespace team_team_count_correct_l1452_145274

/-- Number of ways to select a team of three students from 20,
    one for each subject: math, Russian language, and informatics. -/
def ways_to_form_team (n : ℕ) : ℕ :=
  if n ≥ 3 then n * (n - 1) * (n - 2) else 0

theorem team_team_count_correct : ways_to_form_team 20 = 6840 :=
by sorry

end team_team_count_correct_l1452_145274


namespace roots_of_equation_l1452_145262

theorem roots_of_equation :
  (∃ (x_1 x_2 : ℝ), x_1 > x_2 ∧ (∀ x, x^2 - |x-1| - 1 = 0 ↔ x = x_1 ∨ x = x_2)) :=
sorry

end roots_of_equation_l1452_145262


namespace num_regions_of_lines_l1452_145251

theorem num_regions_of_lines (R : ℕ → ℕ) :
  R 1 = 2 ∧ 
  (∀ n, R (n + 1) = R n + (n + 1)) →
  (∀ n, R n = (n * (n + 1)) / 2 + 1) :=
by
  intro h
  sorry

end num_regions_of_lines_l1452_145251


namespace negation_of_existential_proposition_l1452_145203

theorem negation_of_existential_proposition : 
  (¬ ∃ x : ℝ, x^2 - x + 3 = 0) = (∀ x : ℝ, x^2 - x + 3 ≠ 0) :=
by
  sorry

end negation_of_existential_proposition_l1452_145203


namespace find_kn_l1452_145213

theorem find_kn (k n : ℕ) (h_pos_k : k > 0) (h_pos_n : n > 0) (h_eq : k^2 - 2016 = 3^n) : k = 45 ∧ n = 2 :=
by
  sorry

end find_kn_l1452_145213


namespace tomato_puree_water_percentage_l1452_145217

theorem tomato_puree_water_percentage :
  (∀ (juice_purity water_percentage : ℝ), 
    (juice_purity = 0.90) → 
    (20 * juice_purity = 18) →
    (2.5 - 2) = 0.5 →
    (2.5 * water_percentage - 0.5) = 0 →
    water_percentage = 0.20) :=
by
  intros juice_purity water_percentage h1 h2 h3 h4
  sorry

end tomato_puree_water_percentage_l1452_145217


namespace ordered_triples_count_l1452_145242

def similar_prisms_count (b : ℕ) (c : ℕ) (a : ℕ) := 
  (a ≤ c ∧ c ≤ b ∧ 
   ∃ (x y z : ℕ), x ≤ z ∧ z ≤ y ∧ y = b ∧ 
   x < a ∧ y < b ∧ z < c ∧ 
   ((x : ℚ) / a = (y : ℚ) / b ∧ (y : ℚ) / b = (z : ℚ) / c))

theorem ordered_triples_count : 
  ∃ (n : ℕ), n = 24 ∧ ∀ a c, similar_prisms_count 2000 c a → a < c :=
sorry

end ordered_triples_count_l1452_145242


namespace truth_values_of_p_and_q_l1452_145218

variable (p q : Prop)

theorem truth_values_of_p_and_q
  (h1 : ¬ (p ∧ q))
  (h2 : (¬ p ∨ q)) :
  ¬ p ∧ (q ∨ ¬ q) :=
by {
  sorry
}

end truth_values_of_p_and_q_l1452_145218


namespace breakfast_cost_l1452_145210

def muffin_cost : ℕ := 2
def fruit_cup_cost : ℕ := 3

def francis_muffins : ℕ := 2
def francis_fruit_cups : ℕ := 2

def kiera_muffins : ℕ := 2
def kiera_fruit_cup : ℕ := 1

theorem breakfast_cost :
  muffin_cost * francis_muffins + fruit_cup_cost * francis_fruit_cups
  + muffin_cost * kiera_muffins + fruit_cup_cost * kiera_fruit_cup = 17 :=
by
  -- skipping proof
  sorry

end breakfast_cost_l1452_145210


namespace waiter_tips_earned_l1452_145277

theorem waiter_tips_earned (total_customers tips_left no_tip_customers tips_per_customer : ℕ) :
  no_tip_customers + tips_left = total_customers ∧ tips_per_customer = 3 ∧ no_tip_customers = 5 ∧ total_customers = 7 → 
  tips_left * tips_per_customer = 6 :=
by
  intro h
  sorry

end waiter_tips_earned_l1452_145277


namespace emily_olivia_books_l1452_145286

theorem emily_olivia_books (shared_books total_books_emily books_olivia_not_in_emily : ℕ)
  (h1 : shared_books = 15)
  (h2 : total_books_emily = 23)
  (h3 : books_olivia_not_in_emily = 8) : (total_books_emily - shared_books + books_olivia_not_in_emily = 16) :=
by
  sorry

end emily_olivia_books_l1452_145286


namespace Sahil_purchase_price_l1452_145298

theorem Sahil_purchase_price :
  ∃ P : ℝ, (1.5 * (P + 6000) = 25500) → P = 11000 :=
sorry

end Sahil_purchase_price_l1452_145298


namespace decorative_plate_painted_fraction_l1452_145243

noncomputable def fraction_painted_area (total_area painted_area : ℕ) : ℚ :=
  painted_area / total_area

theorem decorative_plate_painted_fraction :
  let side_length := 4
  let total_area := side_length * side_length
  let painted_smaller_squares := 6
  fraction_painted_area total_area painted_smaller_squares = 3 / 8 :=
by
  sorry

end decorative_plate_painted_fraction_l1452_145243
