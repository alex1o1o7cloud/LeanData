import Mathlib

namespace multiply_polynomials_l1446_144662

theorem multiply_polynomials (x : ℝ) : (x^4 + 50 * x^2 + 625) * (x^2 - 25) = x^6 - 15625 := by
  sorry

end multiply_polynomials_l1446_144662


namespace min_value_expr_l1446_144692

theorem min_value_expr (a b : ℝ) (h1 : 2 * a + b = a * b) (h2 : a > 0) (h3 : b > 0) : 
  ∃ a b, (a > 0 ∧ b > 0 ∧ 2 * a + b = a * b) ∧ (∀ x y, (x > 0 ∧ y > 0 ∧ 2 * x + y = x * y) → (1 / (x - 1) + 2 / (y - 2)) ≥ 2) ∧ ((1 / (a - 1) + 2 / (b - 2)) = 2) :=
by
  sorry

end min_value_expr_l1446_144692


namespace min_colors_needed_for_boxes_l1446_144612

noncomputable def min_colors_needed : Nat := 23

theorem min_colors_needed_for_boxes :
  ∀ (boxes : Fin 8 → Fin 6 → Nat), 
  (∀ i, ∀ j : Fin 6, boxes i j < min_colors_needed) → 
  (∀ i, (Function.Injective (boxes i))) → 
  (∀ c1 c2, c1 ≠ c2 → (∃! b, ∃ p1 p2, (p1 ≠ p2 ∧ boxes b p1 = c1 ∧ boxes b p2 = c2))) → 
  min_colors_needed = 23 := 
by sorry

end min_colors_needed_for_boxes_l1446_144612


namespace solve_fraction_eq_zero_l1446_144682

theorem solve_fraction_eq_zero (a : ℝ) (h : a ≠ -1) : (a^2 - 1) / (a + 1) = 0 ↔ a = 1 :=
by {
  sorry
}

end solve_fraction_eq_zero_l1446_144682


namespace simplify_expression_l1446_144685

open Real

theorem simplify_expression (x : ℝ) (hx : 0 < x) : Real.sqrt (Real.sqrt (x^3 * sqrt (x^5))) = x^(11/8) :=
sorry

end simplify_expression_l1446_144685


namespace percentage_of_rotten_oranges_l1446_144677

-- Define the conditions
def total_oranges : ℕ := 600
def total_bananas : ℕ := 400
def rotten_bananas_percentage : ℝ := 0.08
def good_fruits_percentage : ℝ := 0.878

-- Define the proof problem
theorem percentage_of_rotten_oranges :
  let total_fruits := total_oranges + total_bananas
  let number_of_rotten_bananas := rotten_bananas_percentage * total_bananas
  let number_of_good_fruits := good_fruits_percentage * total_fruits
  let number_of_rotten_fruits := total_fruits - number_of_good_fruits
  let number_of_rotten_oranges := number_of_rotten_fruits - number_of_rotten_bananas
  let percentage_of_rotten_oranges := (number_of_rotten_oranges / total_oranges) * 100
  percentage_of_rotten_oranges = 15 := 
by
  sorry

end percentage_of_rotten_oranges_l1446_144677


namespace anna_apple_ratio_l1446_144660

-- Definitions based on conditions
def tuesday_apples : ℕ := 4
def wednesday_apples : ℕ := 2 * tuesday_apples
def total_apples : ℕ := 14

-- Theorem statement
theorem anna_apple_ratio :
  ∃ thursday_apples : ℕ, 
  thursday_apples = total_apples - (tuesday_apples + wednesday_apples) ∧
  (thursday_apples : ℚ) / tuesday_apples = 1 / 2 :=
by
  sorry

end anna_apple_ratio_l1446_144660


namespace birds_left_in_tree_l1446_144657

-- Define the initial number of birds in the tree
def initialBirds : ℝ := 42.5

-- Define the number of birds that flew away
def birdsFlewAway : ℝ := 27.3

-- Theorem statement: Prove the number of birds left in the tree
theorem birds_left_in_tree : initialBirds - birdsFlewAway = 15.2 :=
by 
  sorry

end birds_left_in_tree_l1446_144657


namespace floor_plus_x_eq_205_l1446_144643

theorem floor_plus_x_eq_205 (x : ℝ) (h : ⌊x⌋ + x = 20.5) : x = 10.5 :=
sorry

end floor_plus_x_eq_205_l1446_144643


namespace customer_bought_29_eggs_l1446_144691

-- Defining the conditions
def baskets : List ℕ := [4, 6, 12, 13, 22, 29]
def total_eggs : ℕ := 86
def is_multiple_of_three (n : ℕ) : Prop := n % 3 = 0

-- Stating the problem
theorem customer_bought_29_eggs :
  ∃ eggs_in_basket,
    eggs_in_basket ∈ baskets ∧
    is_multiple_of_three (total_eggs - eggs_in_basket) ∧
    eggs_in_basket = 29 :=
by sorry

end customer_bought_29_eggs_l1446_144691


namespace range_of_a_l1446_144616

def p (a : ℝ) : Prop := ∀ x : ℝ, x^2 - 2 * a * x + 4 > 0

def q (a : ℝ) : Prop :=
  ∃ x y : ℝ, (x > 0 ∧ y > 0 ∨ x < 0 ∧ y < 0) ∧ y + (a - 1) * x + 2 * a - 1 = 0

def valid_a (a : ℝ) : Prop :=
  (p a ∨ q a) ∧ ¬(p a ∧ q a)

theorem range_of_a (a : ℝ) :
  valid_a a →
  (a ≤ -2 ∨ (1 ≤ a ∧ a < 2)) :=
sorry

end range_of_a_l1446_144616


namespace find_original_cost_price_l1446_144694

variable (C S : ℝ)

-- Conditions
def original_profit (C S : ℝ) : Prop := S = 1.25 * C
def new_profit_condition (C S : ℝ) : Prop := 1.04 * C = S - 12.60

-- Main Theorem
theorem find_original_cost_price (h1 : original_profit C S) (h2 : new_profit_condition C S) : C = 60 := 
sorry

end find_original_cost_price_l1446_144694


namespace fall_increase_l1446_144638

noncomputable def percentage_increase_in_fall (x : ℝ) : ℝ :=
  x

theorem fall_increase :
  ∃ (x : ℝ), (1 + percentage_increase_in_fall x / 100) * (1 - 19 / 100) = 1 + 11.71 / 100 :=
by
  sorry

end fall_increase_l1446_144638


namespace how_many_peaches_l1446_144678

-- Define the main problem statement and conditions.
theorem how_many_peaches (A P J_A J_P : ℕ) (h_person_apples: A = 16) (h_person_peaches: P = A + 1) (h_jake_apples: J_A = A + 8) (h_jake_peaches: J_P = P - 6) : P = 17 :=
by
  -- Since the proof is not required, we use sorry to skip it.
  sorry

end how_many_peaches_l1446_144678


namespace find_number_l1446_144647

theorem find_number (x : ℝ) (h : 0.20 * x = 0.20 * 650 + 190) : x = 1600 := by 
  sorry

end find_number_l1446_144647


namespace prime_product_div_by_four_l1446_144641

theorem prime_product_div_by_four 
  (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hpq1 : Nat.Prime (p * q + 1)) : 
  4 ∣ (2 * p + q) * (p + 2 * q) := 
sorry

end prime_product_div_by_four_l1446_144641


namespace coplanar_lines_condition_l1446_144676

theorem coplanar_lines_condition (h : ℝ) : 
  (∃ c : ℝ, 
    (2 : ℝ) = 3 * c ∧ 
    (-1 : ℝ) = c ∧ 
    (h : ℝ) = -2 * c) ↔ 
  (h = 2) :=
by
  sorry

end coplanar_lines_condition_l1446_144676


namespace main_theorem_l1446_144605

variable (x : ℝ)

-- Define proposition p
def p : Prop := ∃ x0 : ℝ, x0^2 < x0

-- Define proposition q
def q : Prop := ∀ x : ℝ, x^2 - x + 1 > 0

-- Main proof problem
theorem main_theorem : p ∧ q := 
by {
  sorry
}

end main_theorem_l1446_144605


namespace evaluate_expression_l1446_144603

theorem evaluate_expression (b : ℕ) (hb : b = 2) : (b^3 * b^4) - b^2 = 124 :=
by
  -- leave the proof empty with a placeholder
  sorry

end evaluate_expression_l1446_144603


namespace total_shaded_area_l1446_144649

theorem total_shaded_area (r R : ℝ) (h1 : π * R^2 = 100 * π) (h2 : r = R / 2) : 
    (1/4) * π * R^2 + (1/4) * π * r^2 = 31.25 * π :=
by
  sorry

end total_shaded_area_l1446_144649


namespace find_4_digit_number_l1446_144606

theorem find_4_digit_number (a b c d : ℕ) 
(h1 : 1000 * a + 100 * b + 10 * c + d = 1000 * d + 100 * c + 10 * b + a - 7182)
(h2 : 1 ≤ a) (h3 : a ≤ 9) (h4 : 0 ≤ b) (h5 : b ≤ 9) 
(h6 : 0 ≤ c) (h7 : c ≤ 9) (h8 : 1 ≤ d) (h9 : d ≤ 9) : 
1000 * a + 100 * b + 10 * c + d = 1909 :=
sorry

end find_4_digit_number_l1446_144606


namespace greatest_k_dividing_abcdef_l1446_144666

theorem greatest_k_dividing_abcdef {a b c d e f : ℤ}
  (h : a^2 + b^2 + c^2 + d^2 + e^2 = f^2) :
  ∃ k, (∀ a b c d e f, a^2 + b^2 + c^2 + d^2 + e^2 = f^2 → k ∣ (a * b * c * d * e * f)) ∧ k = 24 :=
sorry

end greatest_k_dividing_abcdef_l1446_144666


namespace abs_value_x_minus_2_plus_x_plus_3_ge_4_l1446_144627

theorem abs_value_x_minus_2_plus_x_plus_3_ge_4 :
  ∀ x : ℝ, (|x - 2| + |x + 3| ≥ 4) ↔ (x ≤ - (5 / 2)) := 
sorry

end abs_value_x_minus_2_plus_x_plus_3_ge_4_l1446_144627


namespace max_value_m_l1446_144659

theorem max_value_m {m : ℝ} (h : ∀ x : ℝ, -Real.pi / 4 ≤ x ∧ x ≤ Real.pi / 4 → m ≤ Real.tan x + 1) : m = 2 :=
sorry

end max_value_m_l1446_144659


namespace total_pages_read_l1446_144652

-- Define the reading rates
def ReneReadingRate : ℕ := 30  -- pages in 60 minutes
def LuluReadingRate : ℕ := 27  -- pages in 60 minutes
def CherryReadingRate : ℕ := 25  -- pages in 60 minutes

-- Total time in minutes
def totalTime : ℕ := 240  -- minutes

-- Define a function to calculate pages read in given time
def pagesRead (rate : ℕ) (time : ℕ) : ℕ :=
  rate * (time / 60)

-- Theorem to prove the total number of pages read
theorem total_pages_read :
  pagesRead ReneReadingRate totalTime +
  pagesRead LuluReadingRate totalTime +
  pagesRead CherryReadingRate totalTime = 328 :=
by
  -- Proof is not required, hence replaced with sorry
  sorry

end total_pages_read_l1446_144652


namespace cone_radius_from_melted_cylinder_l1446_144625

theorem cone_radius_from_melted_cylinder :
  ∀ (r_cylinder h_cylinder r_cone h_cone : ℝ),
  r_cylinder = 8 ∧ h_cylinder = 2 ∧ h_cone = 6 ∧
  (π * r_cylinder^2 * h_cylinder = (1 / 3) * π * r_cone^2 * h_cone) →
  r_cone = 8 :=
by
  sorry

end cone_radius_from_melted_cylinder_l1446_144625


namespace Amanda_second_day_tickets_l1446_144604

/-- Amanda's ticket sales problem set up -/
def Amanda_total_tickets := 80
def Amanda_first_day_tickets := 5 * 4
def Amanda_third_day_tickets := 28

theorem Amanda_second_day_tickets :
  ∃ (tickets_sold_second_day : ℕ), tickets_sold_second_day = 32 :=
by
  let first_day := Amanda_first_day_tickets
  let third_day := Amanda_third_day_tickets
  let needed_before_third := Amanda_total_tickets - third_day
  let second_day := needed_before_third - first_day
  use second_day
  sorry

end Amanda_second_day_tickets_l1446_144604


namespace circle_land_represents_30105_l1446_144684

-- Definitions based on the problem's conditions
def circleLandNumber (digits : List (ℕ × ℕ)) : ℕ :=
  digits.foldl (λ acc (d_circle : ℕ × ℕ) => acc + d_circle.fst * 10^d_circle.snd) 0

-- Example 207
def number_207 : List (ℕ × ℕ) := [(2, 2), (0, 0), (7, 0)]

-- Example 4520
def number_4520 : List (ℕ × ℕ) := [(4, 3), (5, 1), (2, 0), (0, 0)]

-- The diagram to analyze
def given_diagram : List (ℕ × ℕ) := [(3, 4), (1, 2), (5, 0)]

-- The statement proving the given diagram represents 30105 in Circle Land
theorem circle_land_represents_30105 : circleLandNumber given_diagram = 30105 :=
  sorry

end circle_land_represents_30105_l1446_144684


namespace positive_difference_of_R_coords_l1446_144654

theorem positive_difference_of_R_coords :
    ∀ (xR yR : ℝ),
    ∃ (k : ℝ),
    (∀ (A B C R S : ℝ × ℝ), 
    A = (-1, 6) ∧ B = (1, 2) ∧ C = (7, 2) ∧ 
    R = (k, -0.5 * k + 5.5) ∧ S = (k, 2) ∧
    (0.5 * |7 - k| * |0.5 * k - 3.5| = 8)) → 
    |xR - yR| = 1 :=
by
  sorry

end positive_difference_of_R_coords_l1446_144654


namespace second_odd_integer_l1446_144623

theorem second_odd_integer (n : ℤ) (h : (n - 2) + (n + 2) = 128) : n = 64 :=
by
  sorry

end second_odd_integer_l1446_144623


namespace remaining_surface_area_l1446_144687

def edge_length_original : ℝ := 9
def edge_length_small : ℝ := 2
def surface_area (a : ℝ) : ℝ := 6 * a^2

theorem remaining_surface_area :
  surface_area edge_length_original - 3 * (edge_length_small ^ 2) + 3 * (edge_length_small ^ 2) = 486 :=
by
  sorry

end remaining_surface_area_l1446_144687


namespace photos_last_weekend_45_l1446_144697

theorem photos_last_weekend_45 (photos_animals photos_flowers photos_scenery total_photos_this_weekend photos_last_weekend : ℕ)
  (h1 : photos_animals = 10)
  (h2 : photos_flowers = 3 * photos_animals)
  (h3 : photos_scenery = photos_flowers - 10)
  (h4 : total_photos_this_weekend = photos_animals + photos_flowers + photos_scenery)
  (h5 : photos_last_weekend = total_photos_this_weekend - 15) :
  photos_last_weekend = 45 :=
sorry

end photos_last_weekend_45_l1446_144697


namespace find_pairs_l1446_144693

theorem find_pairs :
  ∀ (x y : ℕ), 0 < x → 0 < y → 7 ^ x - 3 * 2 ^ y = 1 → (x, y) = (1, 1) ∨ (x, y) = (2, 4) :=
by
  intros x y hx hy h
  -- Proof would go here
  sorry

end find_pairs_l1446_144693


namespace domain_of_p_l1446_144631

def is_domain_of_p (x : ℝ) : Prop := x > 5

theorem domain_of_p :
  {x : ℝ | ∃ y : ℝ, y = 5*x + 2 ∧ ∃ z : ℝ, z = 2*x - 10 ∧
    z ≥ 0 ∧ z ≠ 0 ∧ p = 5*x + 2} = {x : ℝ | x > 5} :=
by
  sorry

end domain_of_p_l1446_144631


namespace find_length_of_bridge_l1446_144629

noncomputable def length_of_train : ℝ := 165
noncomputable def speed_of_train_kmph : ℝ := 54
noncomputable def time_to_cross_bridge_seconds : ℝ := 67.66125376636536

noncomputable def speed_of_train_mps : ℝ :=
  speed_of_train_kmph * (1000 / 3600)

noncomputable def total_distance_covered : ℝ :=
  speed_of_train_mps * time_to_cross_bridge_seconds

noncomputable def length_of_bridge : ℝ :=
  total_distance_covered - length_of_train

theorem find_length_of_bridge : length_of_bridge = 849.92 := by
  sorry

end find_length_of_bridge_l1446_144629


namespace painted_cubes_l1446_144667

/-- 
  Given a cube of side 9 painted red and cut into smaller cubes of side 3,
  prove the number of smaller cubes with paint on exactly 2 sides is 12.
-/
theorem painted_cubes (l : ℕ) (s : ℕ) (n : ℕ) (edges : ℕ) (faces : ℕ)
  (hcube_dimension : l = 9) (hsmaller_cubes_dimension : s = 3) 
  (hedges : edges = 12) (hfaces : faces * edges = 12) 
  (htotal_cubes : n = (l^3) / (s^3)) : 
  n * faces = 12 :=
sorry

end painted_cubes_l1446_144667


namespace midline_equation_l1446_144680

theorem midline_equation (a b : ℝ) (K1 K2 : ℝ)
  (h1 : K1^2 = (a^2) / 4 + b^2)
  (h2 : K2^2 = a^2 + (b^2) / 4) :
  16 * K2^2 - 4 * K1^2 = 15 * a^2 :=
by
  sorry

end midline_equation_l1446_144680


namespace find_z_when_w_15_l1446_144609

-- Define a direct variation relationship
def varies_directly (z w : ℕ) (k : ℕ) : Prop :=
  z = k * w

-- Using the given conditions and to prove the statement
theorem find_z_when_w_15 :
  ∃ k, (varies_directly 10 5 k) → (varies_directly 30 15 k) :=
by
  sorry

end find_z_when_w_15_l1446_144609


namespace sum_of_reciprocals_factors_12_l1446_144636

theorem sum_of_reciprocals_factors_12 : (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12) = 7/3 :=
by sorry

end sum_of_reciprocals_factors_12_l1446_144636


namespace face_opposite_one_is_three_l1446_144675

def faces : List ℕ := [1, 2, 3, 4, 5, 6]

theorem face_opposite_one_is_three (x : ℕ) (h1 : x ∈ faces) (h2 : x ≠ 1) : x = 3 :=
by
  sorry

end face_opposite_one_is_three_l1446_144675


namespace fill_tub_in_seconds_l1446_144630

theorem fill_tub_in_seconds 
  (faucet_rate : ℚ)
  (four_faucet_rate : ℚ := 4 * faucet_rate)
  (three_faucet_rate : ℚ := 3 * faucet_rate)
  (time_for_100_gallons_in_minutes : ℚ := 6)
  (time_for_100_gallons_in_seconds : ℚ := time_for_100_gallons_in_minutes * 60)
  (volume_100_gallons : ℚ := 100)
  (rate_per_three_faucets_in_gallons_per_second : ℚ := volume_100_gallons / time_for_100_gallons_in_seconds)
  (rate_per_faucet : ℚ := rate_per_three_faucets_in_gallons_per_second / 3)
  (rate_per_four_faucets : ℚ := 4 * rate_per_faucet)
  (volume_50_gallons : ℚ := 50)
  (expected_time_seconds : ℚ := volume_50_gallons / rate_per_four_faucets) :
  expected_time_seconds = 135 :=
sorry

end fill_tub_in_seconds_l1446_144630


namespace expression_evaluation_l1446_144624

theorem expression_evaluation :
  (0.86^3) - ((0.1^3) / (0.86^2)) + 0.086 + (0.1^2) = 0.730704 := 
by 
  sorry

end expression_evaluation_l1446_144624


namespace ladder_distance_from_wall_l1446_144642

noncomputable def dist_from_wall (ladder_length : ℝ) (angle_deg : ℝ) : ℝ :=
  ladder_length * Real.cos (angle_deg * Real.pi / 180)

theorem ladder_distance_from_wall :
  ∀ (ladder_length : ℝ) (angle_deg : ℝ), ladder_length = 19 → angle_deg = 60 → dist_from_wall ladder_length angle_deg = 9.5 :=
by
  intros ladder_length angle_deg h1 h2
  sorry

end ladder_distance_from_wall_l1446_144642


namespace shorten_to_sixth_power_l1446_144608

theorem shorten_to_sixth_power (x n m p q r : ℕ) (h1 : x > 1000000)
  (h2 : x / 10 = n^2)
  (h3 : n^2 / 10 = m^3)
  (h4 : m^3 / 10 = p^4)
  (h5 : p^4 / 10 = q^5) :
  q^5 / 10 = r^6 :=
sorry

end shorten_to_sixth_power_l1446_144608


namespace solution_set_inequality_l1446_144681

theorem solution_set_inequality (x : ℝ) : 
  (x + 5) * (3 - 2 * x) ≤ 6 ↔ (x ≤ -9/2 ∨ x ≥ 1) :=
by
  sorry  -- proof skipped as instructed

end solution_set_inequality_l1446_144681


namespace shaded_area_percentage_l1446_144648

-- Define the given conditions
def square_area := 6 * 6
def shaded_area_left := (1 / 2) * 2 * 6
def shaded_area_right := (1 / 2) * 4 * 6
def total_shaded_area := shaded_area_left + shaded_area_right

-- State the theorem
theorem shaded_area_percentage : (total_shaded_area / square_area) * 100 = 50 := by
  sorry

end shaded_area_percentage_l1446_144648


namespace problem_statement_l1446_144628

noncomputable def verify_ratio (x y c d : ℝ) (h1 : 4 * x - 2 * y = c) (h2 : 6 * y - 12 * x = d) (h3 : d ≠ 0) : Prop :=
  c / d = -1/3

theorem problem_statement (x y c d : ℝ) (h1 : 4 * x - 2 * y = c) (h2 : 6 * y - 12 * x = d) (h3 : d ≠ 0) : verify_ratio x y c d h1 h2 h3 :=
  sorry

end problem_statement_l1446_144628


namespace monotone_intervals_range_of_t_for_three_roots_l1446_144665

def f (t x : ℝ) : ℝ := x^3 - 2 * x^2 + x + t

def f_prime (x : ℝ) : ℝ := 3 * x^2 - 4 * x + 1

-- 1. Monotonic intervals
theorem monotone_intervals (t : ℝ) :
  (∀ x, f_prime x > 0 → x < 1/3 ∨ x > 1) ∧
  (∀ x, f_prime x < 0 → 1/3 < x ∧ x < 1) :=
sorry

-- 2. Range of t for three real roots
theorem range_of_t_for_three_roots (t : ℝ) :
  (∃ a b : ℝ, f t a = 0 ∧ f t b = 0 ∧ a ≠ b ∧
   a = 1/3 ∧ b = 1 ∧
   -4/27 + t > 0 ∧ t < 0) :=
sorry

end monotone_intervals_range_of_t_for_three_roots_l1446_144665


namespace brookdale_avg_temp_l1446_144622

def highs : List ℤ := [51, 64, 60, 59, 48, 55]
def lows : List ℤ := [42, 49, 47, 43, 41, 44]

def average_temperature : ℚ :=
  let total_sum := highs.sum + lows.sum
  let count := (highs.length + lows.length : ℚ)
  total_sum / count

theorem brookdale_avg_temp :
  average_temperature = 49.4 :=
by
  -- The proof goes here
  sorry

end brookdale_avg_temp_l1446_144622


namespace Daisy_lunch_vs_breakfast_l1446_144632

noncomputable def breakfast_cost : ℝ := 2.0 + 3.0 + 4.0 + 3.5
noncomputable def lunch_cost_before_service_charge : ℝ := 3.75 + 5.75 + 1.0
noncomputable def service_charge : ℝ := 0.10 * lunch_cost_before_service_charge
noncomputable def total_lunch_cost : ℝ := lunch_cost_before_service_charge + service_charge

theorem Daisy_lunch_vs_breakfast : total_lunch_cost - breakfast_cost = -0.95 := by
  sorry

end Daisy_lunch_vs_breakfast_l1446_144632


namespace common_terms_only_1_and_7_l1446_144658

def sequence_a (n : ℕ) : ℕ :=
  if n = 0 then 1
  else if n = 1 then 2
  else 4 * sequence_a (n - 1) - sequence_a (n - 2)

def sequence_b (n : ℕ) : ℕ :=
  if n = 0 then 1
  else if n = 1 then 7
  else 6 * sequence_b (n - 1) - sequence_b (n - 2)

theorem common_terms_only_1_and_7 :
  ∀ n m : ℕ, (sequence_a n = sequence_b m) → (sequence_a n = 1 ∨ sequence_a n = 7) :=
by {
  sorry
}

end common_terms_only_1_and_7_l1446_144658


namespace total_hours_uploaded_l1446_144611

def hours_June_1_to_10 : ℝ := 5 * 2 * 10
def hours_June_11_to_20 : ℝ := 10 * 1 * 10
def hours_June_21_to_25 : ℝ := 7 * 3 * 5
def hours_June_26_to_30 : ℝ := 15 * 0.5 * 5

def total_video_hours : ℝ :=
  hours_June_1_to_10 + hours_June_11_to_20 + hours_June_21_to_25 + hours_June_26_to_30

theorem total_hours_uploaded :
  total_video_hours = 342.5 :=
by
  sorry

end total_hours_uploaded_l1446_144611


namespace part1_part2_l1446_144690

-- Part (1) statement
theorem part1 {x : ℝ} : (|x - 1| + |x + 2| >= 5) ↔ (x <= -3 ∨ x >= 2) := 
sorry

-- Part (2) statement
theorem part2 (a : ℝ) : (∀ x : ℝ, (|a * x - 2| < 3 ↔ -5/3 < x ∧ x < 1/3)) → a = -3 :=
sorry

end part1_part2_l1446_144690


namespace inverse_proposition_l1446_144626

theorem inverse_proposition (a b c : ℝ) : (a > b → a + c > b + c) → (a + c > b + c → a > b) :=
sorry

end inverse_proposition_l1446_144626


namespace proof_l1446_144653

noncomputable def problem_statement (a b : ℝ) :=
  7 * (Real.sin a + Real.sin b) + 6 * (Real.cos a * Real.cos b - 1) = 0 →
  (Real.tan (a / 2) * Real.tan (b / 2) = 1 ∨ Real.tan (a / 2) * Real.tan (b / 2) = -1)

theorem proof : ∀ a b : ℝ, problem_statement a b := sorry

end proof_l1446_144653


namespace range_of_m_l1446_144672

noncomputable def quadratic_expr_never_equal (m : ℝ) : Prop :=
  ∀ (x : ℝ), 2 * x^2 + 4 * x + m ≠ 3 * x^2 - 2 * x + 6

theorem range_of_m (m : ℝ) : quadratic_expr_never_equal m ↔ m < -3 := 
by
  sorry

end range_of_m_l1446_144672


namespace dog_bones_initial_count_l1446_144614

theorem dog_bones_initial_count (buried : ℝ) (final : ℝ) : buried = 367.5 → final = -860 → (buried + (final + 367.5) + 860) = 367.5 :=
by
  intros h1 h2
  sorry

end dog_bones_initial_count_l1446_144614


namespace passing_marks_l1446_144615

variable (T P : ℝ)

theorem passing_marks :
  (0.35 * T = P - 40) →
  (0.60 * T = P + 25) →
  P = 131 :=
by
  intro h1 h2
  -- Proof steps should follow here.
  sorry

end passing_marks_l1446_144615


namespace min_digits_fraction_l1446_144689

def minDigitsToRightOfDecimal (n : ℕ) : ℕ :=
  -- This represents the minimum number of digits needed to express n / (2^15 * 5^7)
  -- as a decimal.
  -- The actual function body is hypothetical and not implemented here.
  15

theorem min_digits_fraction :
  minDigitsToRightOfDecimal 987654321 = 15 :=
by
  sorry

end min_digits_fraction_l1446_144689


namespace perpendicular_line_through_P_l1446_144635

open Real

-- Define the point (1, 0)
def P : ℝ × ℝ := (1, 0)

-- Define the initial line x - 2y - 2 = 0
def initial_line (x y : ℝ) : Prop := x - 2 * y = 2

-- Define the desired line 2x + y - 2 = 0
def desired_line (x y : ℝ) : Prop := 2 * x + y = 2

-- State that the desired line passes through the point (1, 0) and is perpendicular to the initial line
theorem perpendicular_line_through_P :
  (∃ m b, b ∈ Set.univ ∧ (∀ x y, desired_line x y → y = m * x + b)) ∧ ∀ x y, 
  initial_line x y → x ≠ 0 → desired_line y (-x / 2) :=
sorry

end perpendicular_line_through_P_l1446_144635


namespace max_n_for_positive_sum_l1446_144669

-- Define the arithmetic sequence \(a_n\)
def arithmetic_sequence (a d : ℤ) (n : ℕ) := a + n * d

-- Define the sum of the first n terms of the arithmetic sequence
def S_n (a d : ℤ) (n : ℕ) := n * (2 * a + (n-1) * d) / 2

theorem max_n_for_positive_sum 
  (a : ℤ) 
  (d : ℤ) 
  (h_max_sum : ∃ m : ℕ, S_n a d m = S_n a d (m+1))
  (h_ratio : (arithmetic_sequence a d 15) / (arithmetic_sequence a d 14) < -1) :
  27 = 27 :=
sorry

end max_n_for_positive_sum_l1446_144669


namespace cos_diff_alpha_beta_l1446_144668

theorem cos_diff_alpha_beta (α β : ℝ) (h1 : Real.sin α = 2 / 3) (h2 : Real.cos β = -3 / 4)
    (h3 : α ∈ Set.Ioo (π / 2) π) (h4 : β ∈ Set.Ioo π (3 * π / 2)) :
    Real.cos (α - β) = (3 * Real.sqrt 5 - 2 * Real.sqrt 7) / 12 := 
sorry

end cos_diff_alpha_beta_l1446_144668


namespace alexis_initial_budget_l1446_144633

-- Define all the given conditions
def cost_shirt : Int := 30
def cost_pants : Int := 46
def cost_coat : Int := 38
def cost_socks : Int := 11
def cost_belt : Int := 18
def cost_shoes : Int := 41
def amount_left : Int := 16

-- Define the total expenses
def total_expenses : Int := cost_shirt + cost_pants + cost_coat + cost_socks + cost_belt + cost_shoes

-- Define the initial budget
def initial_budget : Int := total_expenses + amount_left

-- The proof statement
theorem alexis_initial_budget : initial_budget = 200 := by
  sorry

end alexis_initial_budget_l1446_144633


namespace xiao_zhang_winning_probability_max_expected_value_l1446_144620

-- Definitions for the conditions
variables (a b c : ℕ)
variable (h_sum : a + b + c = 6)

-- Main theorem statement 1: Probability of Xiao Zhang winning
theorem xiao_zhang_winning_probability (h_sum : a + b + c = 6) :
  (3 * a + 2 * b + c) / 36 = a / 6 * 3 / 6 + b / 6 * 2 / 6 + c / 6 * 1 / 6 :=
sorry

-- Main theorem statement 2: Maximum expected value of Xiao Zhang's score
theorem max_expected_value (h_sum : a + b + c = 6) :
  (3 * a + 4 * b + 3 * c) / 36 = (1 / 2 + b / 36) →  (a = 0 ∧ b = 6 ∧ c = 0) :=
sorry

end xiao_zhang_winning_probability_max_expected_value_l1446_144620


namespace elise_spent_on_puzzle_l1446_144679

-- Definitions based on the problem conditions:
def initial_money : ℕ := 8
def saved_money : ℕ := 13
def spent_on_comic : ℕ := 2
def remaining_money : ℕ := 1

-- Prove that the amount spent on the puzzle is $18.
theorem elise_spent_on_puzzle : initial_money + saved_money - spent_on_comic - remaining_money = 18 := by
  sorry

end elise_spent_on_puzzle_l1446_144679


namespace find_sol_y_pct_l1446_144645

-- Define the conditions
def sol_x_vol : ℕ := 200            -- Volume of solution x in milliliters
def sol_y_vol : ℕ := 600            -- Volume of solution y in milliliters
def sol_x_pct : ℕ := 10             -- Percentage of alcohol in solution x
def final_sol_pct : ℕ := 25         -- Percentage of alcohol in the final solution
def final_sol_vol := sol_x_vol + sol_y_vol -- Total volume of the final solution

-- Define the problem statement
theorem find_sol_y_pct (sol_x_vol sol_y_vol final_sol_vol : ℕ) 
  (sol_x_pct final_sol_pct : ℕ) : 
  (600 * 10 + sol_y_vol * 30) / 800 = 25 :=
by
  sorry

end find_sol_y_pct_l1446_144645


namespace portion_of_larger_jar_full_l1446_144646

noncomputable def smaller_jar_capacity (S L : ℝ) : Prop :=
  (1 / 5) * S = (1 / 4) * L

noncomputable def larger_jar_capacity (L : ℝ) : ℝ :=
  (1 / 5) * (5 / 4) * L

theorem portion_of_larger_jar_full (S L : ℝ) 
  (h1 : smaller_jar_capacity S L) : 
  (1 / 4) * L + (1 / 4) * L = (1 / 2) * L := 
sorry

end portion_of_larger_jar_full_l1446_144646


namespace Papi_Calot_plants_l1446_144664

theorem Papi_Calot_plants :
  let initial_potatoes_plants := 10 * 25
  let initial_carrots_plants := 15 * 30
  let initial_onions_plants := 12 * 20
  let total_potato_plants := initial_potatoes_plants + 20
  let total_carrot_plants := initial_carrots_plants + 30
  let total_onion_plants := initial_onions_plants + 10
  total_potato_plants = 270 ∧
  total_carrot_plants = 480 ∧
  total_onion_plants = 250 := by
  sorry

end Papi_Calot_plants_l1446_144664


namespace simplify_expr1_simplify_expr2_simplify_expr3_l1446_144655

theorem simplify_expr1 (y : ℤ) (hy : y = 2) : -3 * y^2 - 6 * y + 2 * y^2 + 5 * y = -6 := 
by sorry

theorem simplify_expr2 (a : ℤ) (ha : a = -2) : 15 * a^2 * (-4 * a^2 + (6 * a - a^2) - 3 * a) = -1560 :=
by sorry

theorem simplify_expr3 (x y : ℤ) (h1 : x * y = 2) (h2 : x + y = 3) : (3 * x * y + 10 * y) + (5 * x - (2 * x * y + 2 * y - 3 * x)) = 26 :=
by sorry

end simplify_expr1_simplify_expr2_simplify_expr3_l1446_144655


namespace quadratic_eq_integer_roots_iff_l1446_144698

theorem quadratic_eq_integer_roots_iff (n : ℕ) (hn : n > 0) :
  (∃ x y : ℤ, x * y = n ∧ x + y = 4) ↔ (n = 3 ∨ n = 4) :=
by
  sorry

end quadratic_eq_integer_roots_iff_l1446_144698


namespace find_number_l1446_144651

theorem find_number (x : ℝ) (h : x^2 + 50 = (x - 10)^2) : x = 2.5 :=
sorry

end find_number_l1446_144651


namespace share_of_A_eq_70_l1446_144686

theorem share_of_A_eq_70 (A B C : ℝ) (h1 : A = (2/3) * B) (h2 : B = (1/4) * C) (h3 : A + B + C = 595) : A = 70 :=
sorry

end share_of_A_eq_70_l1446_144686


namespace peanut_butter_last_days_l1446_144688

-- Definitions for the problem conditions
def daily_consumption : ℕ := 2
def servings_per_jar : ℕ := 15
def num_jars : ℕ := 4

-- The statement to prove
theorem peanut_butter_last_days : 
  (num_jars * servings_per_jar) / daily_consumption = 30 :=
by
  sorry

end peanut_butter_last_days_l1446_144688


namespace sum_six_times_product_l1446_144621

variable (a b x : ℝ)

theorem sum_six_times_product (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a + b = 6 * x) (h4 : 1/a + 1/b = 6) :
  x = a * b := sorry

end sum_six_times_product_l1446_144621


namespace problem1_solution_problem2_solution_l1446_144600

theorem problem1_solution (x : ℝ) :
  (2 < |2 * x - 5| ∧ |2 * x - 5| ≤ 7) → ((-1 ≤ x ∧ x < 3 / 2) ∨ (7 / 2 < x ∧ x ≤ 6)) := by
  sorry

theorem problem2_solution (x : ℝ) :
  (1 / (x - 1) > x + 1) → (x < -Real.sqrt 2 ∨ (1 < x ∧ x < Real.sqrt 2)) := by
  sorry

end problem1_solution_problem2_solution_l1446_144600


namespace range_of_m_l1446_144656

def f (x : ℝ) : ℝ := x^3 - 3 * x - 1

theorem range_of_m (m : ℝ) (h : ∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ f x1 = m ∧ f x2 = m ∧ f x3 = m) : -3 < m ∧ m < 1 := 
sorry

end range_of_m_l1446_144656


namespace arithmetic_expr_eval_l1446_144613

/-- A proof that the arithmetic expression (80 + (5 * 12) / (180 / 3)) ^ 2 * (7 - (3^2)) evaluates to -13122. -/
theorem arithmetic_expr_eval : (80 + (5 * 12) / (180 / 3)) ^ 2 * (7 - (3^2)) = -13122 :=
by
  sorry

end arithmetic_expr_eval_l1446_144613


namespace sqrt_sum_odds_l1446_144673

theorem sqrt_sum_odds : 
  (Real.sqrt 1 + Real.sqrt (1+3) + Real.sqrt (1+3+5) + Real.sqrt (1+3+5+7) + Real.sqrt (1+3+5+7+9) + Real.sqrt (1+3+5+7+9+11)) = 21 := 
by
  sorry

end sqrt_sum_odds_l1446_144673


namespace find_ab_l1446_144602

variables {a b : ℝ}

theorem find_ab
  (h : ∀ x : ℝ, 0 ≤ x → 0 ≤ x^4 - x^3 + a * x + b ∧ x^4 - x^3 + a * x + b ≤ (x^2 - 1)^2) :
  a * b = -1 :=
sorry

end find_ab_l1446_144602


namespace evaluate_expression_l1446_144695

variable (b : ℝ)

theorem evaluate_expression : ( ( (b^(16/8))^(1/4) )^3 * ( (b^(16/4))^(1/8) )^3 ) = b^3 := by
  sorry

end evaluate_expression_l1446_144695


namespace toms_age_l1446_144610

theorem toms_age (T S : ℕ) (h1 : T = 2 * S - 1) (h2 : T + S = 14) : T = 9 :=
sorry

end toms_age_l1446_144610


namespace solve_for_y_l1446_144661

theorem solve_for_y (y : ℕ) (h : 5 * (2^y) = 320) : y = 6 := 
by 
  sorry

end solve_for_y_l1446_144661


namespace diff_of_squares_div_l1446_144607

-- Definitions from the conditions
def a : ℕ := 125
def b : ℕ := 105

-- The main statement to be proved
theorem diff_of_squares_div {a b : ℕ} (h1 : a = 125) (h2 : b = 105) : (a^2 - b^2) / 20 = 230 := by
  sorry

end diff_of_squares_div_l1446_144607


namespace point_P_in_first_quadrant_l1446_144674

def point_P := (3, 2)
def first_quadrant (p : ℕ × ℕ) : Prop := p.1 > 0 ∧ p.2 > 0

theorem point_P_in_first_quadrant : first_quadrant point_P :=
by
  sorry

end point_P_in_first_quadrant_l1446_144674


namespace cookie_sales_l1446_144634

theorem cookie_sales (n M A : ℕ) 
  (hM : M = n - 9)
  (hA : A = n - 2)
  (h_sum : M + A < n)
  (hM_positive : M ≥ 1)
  (hA_positive : A ≥ 1) : 
  n = 10 := 
sorry

end cookie_sales_l1446_144634


namespace circular_seating_count_l1446_144650

theorem circular_seating_count :
  let D := 5 -- Number of Democrats
  let R := 5 -- Number of Republicans
  let total_politicians := D + R -- Total number of politicians
  let linear_arrangements := Nat.factorial total_politicians -- Total linear arrangements
  let unique_circular_arrangements := linear_arrangements / total_politicians -- Adjusting for circular rotations
  unique_circular_arrangements = 362880 :=
by
  sorry

end circular_seating_count_l1446_144650


namespace no_five_consecutive_divisible_by_2005_l1446_144637

def seq (n : ℕ) : ℕ := 1 + 2^n + 3^n + 4^n + 5^n

theorem no_five_consecutive_divisible_by_2005 :
  ¬ (∃ m : ℕ, ∀ k : ℕ, k < 5 → (seq (m + k)) % 2005 = 0) :=
sorry

end no_five_consecutive_divisible_by_2005_l1446_144637


namespace fraction_equal_l1446_144699

theorem fraction_equal {a b x : ℝ} (h1 : x = a / b) (h2 : a ≠ b) (h3 : b ≠ 0) : 
  (a + b) / (a - b) = (x + 1) / (x - 1) := 
by
  sorry

end fraction_equal_l1446_144699


namespace square_of_binomial_conditions_l1446_144601

variable (x a b m : ℝ)

theorem square_of_binomial_conditions :
  ∃ u v : ℝ, (x + a) * (x - a) = u^2 - v^2 ∧
             ∃ e f : ℝ, (-x - b) * (x - b) = - (e^2 - f^2) ∧
             ∃ g h : ℝ, (b + m) * (m - b) = g^2 - h^2 ∧
             ¬ ∃ p q : ℝ, (a + b) * (-a - b) = p^2 - q^2 :=
by
  sorry

end square_of_binomial_conditions_l1446_144601


namespace reciprocal_of_repeating_decimal_l1446_144619

theorem reciprocal_of_repeating_decimal :
  let x := 0.36363636 -- simplified as .\overline{36}
  ∃ y : ℚ, x = 4 / 11 ∧ y = 1 / x ∧ y = 11 / 4 :=
by
  sorry

end reciprocal_of_repeating_decimal_l1446_144619


namespace closest_point_is_correct_l1446_144670

def line_eq (x : ℝ) : ℝ := -3 * x + 5

def closest_point_on_line_to_given_point : Prop :=
  ∃ (x y : ℝ), y = line_eq x ∧ (x, y) = (17 / 10, -1 / 10) ∧
  (∀ (x' y' : ℝ), y' = line_eq x' → (x' - -4)^2 + (y' - -2)^2 ≥ (x - -4)^2 + (y - -2)^2)
  
theorem closest_point_is_correct : closest_point_on_line_to_given_point :=
sorry

end closest_point_is_correct_l1446_144670


namespace sum_slopes_const_zero_l1446_144640

-- Define variables and constants
variable (p : ℝ) (h : 0 < p)

-- Define parabola and circle equations
def parabola_C1 (x y : ℝ) : Prop := y^2 = 2 * p * x
def circle_C2 (x y : ℝ) : Prop := x^2 + y^2 = p^2

-- Condition: The line segment length from circle cut by directrix
def segment_length_condition : Prop := ∃ d : ℝ, d^2 + 3 = p^2

-- The main theorem to prove
theorem sum_slopes_const_zero
  (A : ℝ × ℝ)
  (F : ℝ × ℝ := (p / 2, 0))
  (M N : ℝ × ℝ)
  (line_n_through_A : ∀ x : ℝ, x = 1 / p - 1 + 1 / p → (1 / p - 1 + x) = 0)
  (intersection_prop: parabola_C1 p M.1 M.2 ∧ parabola_C1 p N.1 N.2) 
  (slope_MF : ℝ := (M.2 / (p / 2 - M.1)) ) 
  (slope_NF : ℝ := (N.2 / (p / 2 - N.1))) :
  slope_MF + slope_NF = 0 := 
sorry

end sum_slopes_const_zero_l1446_144640


namespace range_of_m_l1446_144644

theorem range_of_m (x y m : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + y = 3) (h4 : ∀ x y, x > 0 → y > 0 → x + y = 3 → (4 / (x + 1) + 16 / y > m^2 - 3 * m + 11)) : 1 < m ∧ m < 2 :=
by
  sorry

end range_of_m_l1446_144644


namespace midpoint_of_diagonal_l1446_144696

-- Definition of the points
def point1 : ℝ × ℝ := (2, -3)
def point2 : ℝ × ℝ := (14, 9)

-- Statement about the midpoint of a diagonal in a rectangle
theorem midpoint_of_diagonal : 
  ∀ (x1 y1 x2 y2 : ℝ), (x1, y1) = point1 → (x2, y2) = point2 →
  let midpoint_x := (x1 + x2) / 2
  let midpoint_y := (y1 + y2) / 2
  (midpoint_x, midpoint_y) = (8, 3) :=
by
  intros
  sorry

end midpoint_of_diagonal_l1446_144696


namespace value_of_expression_l1446_144683

theorem value_of_expression (a b c d : ℕ) (h1 : a = 4 * b) (h2 : b = 3 * c) (h3 : c = 5 * d) :
  (a * c) / (b * d) = 20 := by
  sorry

end value_of_expression_l1446_144683


namespace non_congruent_rectangles_count_l1446_144618

theorem non_congruent_rectangles_count :
  let grid_width := 6
  let grid_height := 4
  let axis_aligned_rectangles := (grid_width.choose 2) * (grid_height.choose 2)
  let squares_1x1 := (grid_width - 1) * (grid_height - 1)
  let squares_2x2 := (grid_width - 2) * (grid_height - 2)
  let non_congruent_rectangles := axis_aligned_rectangles - (squares_1x1 + squares_2x2)
  non_congruent_rectangles = 67 := 
by {
  sorry
}

end non_congruent_rectangles_count_l1446_144618


namespace remove_least_candies_l1446_144663

theorem remove_least_candies (total_candies : ℕ) (friends : ℕ) (candies_remaining : ℕ) : total_candies = 34 ∧ friends = 5 ∧ candies_remaining = 4 → (total_candies % friends = candies_remaining) :=
by
  intros h
  sorry

end remove_least_candies_l1446_144663


namespace flight_duration_sum_l1446_144617

theorem flight_duration_sum 
  (departure_time : ℕ×ℕ) (arrival_time : ℕ×ℕ) (delay : ℕ)
  (h m : ℕ)
  (h0 : 0 < m ∧ m < 60)
  (h1 : departure_time = (9, 20))
  (h2 : arrival_time = (13, 45)) -- using 13 for 1 PM, 24-hour format
  (h3 : delay = 25)
  (h4 : ((arrival_time.1 * 60 + arrival_time.2) - (departure_time.1 * 60 + departure_time.2) + delay) = h * 60 + m) :
  h + m = 29 :=
by {
  -- Proof is skipped
  sorry
}

end flight_duration_sum_l1446_144617


namespace find_large_no_l1446_144639

theorem find_large_no (L S : ℤ) (h1 : L - S = 1365) (h2 : L = 6 * S + 15) : L = 1635 :=
by 
  sorry

end find_large_no_l1446_144639


namespace faye_homework_problems_l1446_144671

----- Definitions based on the conditions given -----

def total_math_problems : ℕ := 46
def total_science_problems : ℕ := 9
def problems_finished_at_school : ℕ := 40

----- Theorem statement -----

theorem faye_homework_problems : total_math_problems + total_science_problems - problems_finished_at_school = 15 := by
  -- Sorry is used here to skip the proof
  sorry

end faye_homework_problems_l1446_144671
