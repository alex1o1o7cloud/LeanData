import Mathlib

namespace lance_hourly_earnings_l1400_140079

theorem lance_hourly_earnings
  (hours_per_week : ℕ)
  (workdays_per_week : ℕ)
  (daily_earnings : ℕ)
  (total_weekly_earnings : ℕ)
  (hourly_wage : ℕ)
  (h1 : hours_per_week = 35)
  (h2 : workdays_per_week = 5)
  (h3 : daily_earnings = 63)
  (h4 : total_weekly_earnings = daily_earnings * workdays_per_week)
  (h5 : total_weekly_earnings = hourly_wage * hours_per_week)
  : hourly_wage = 9 :=
sorry

end lance_hourly_earnings_l1400_140079


namespace part1_part2_l1400_140057

-- Definitions and assumptions based on the problem
def f (x a : ℝ) : ℝ := abs (x - a)

-- Condition (1) with given function and inequality solution set
theorem part1 (a : ℝ) :
  (∀ x, f x a ≤ 3 ↔ -1 ≤ x ∧ x ≤ 5) → a = 2 :=
by
  sorry

-- Condition (2) with the range of m under the previously found value of a
theorem part2 (m : ℝ) :
  (∃ x, f x 2 + f (x + 5) 2 < m) → m > 5 :=
by
  sorry

end part1_part2_l1400_140057


namespace new_weight_is_77_l1400_140094

theorem new_weight_is_77 (weight_increase_per_person : ℝ) (number_of_persons : ℕ) (old_weight : ℝ) 
  (total_weight_increase : ℝ) (new_weight : ℝ) 
  (h1 : weight_increase_per_person = 1.5)
  (h2 : number_of_persons = 8)
  (h3 : old_weight = 65)
  (h4 : total_weight_increase = number_of_persons * weight_increase_per_person)
  (h5 : new_weight = old_weight + total_weight_increase) :
  new_weight = 77 :=
sorry

end new_weight_is_77_l1400_140094


namespace jerky_batch_size_l1400_140088

theorem jerky_batch_size
  (total_order_bags : ℕ)
  (initial_bags : ℕ)
  (days_to_fulfill : ℕ)
  (remaining_bags : ℕ := total_order_bags - initial_bags)
  (production_per_day : ℕ := remaining_bags / days_to_fulfill) :
  total_order_bags = 60 →
  initial_bags = 20 →
  days_to_fulfill = 4 →
  production_per_day = 10 :=
by
  intros
  sorry

end jerky_batch_size_l1400_140088


namespace graphs_intersect_at_three_points_l1400_140074

noncomputable def is_invertible (f : ℝ → ℝ) := ∃ (g : ℝ → ℝ), ∀ x : ℝ, f (g x) = x ∧ g (f x) = x

theorem graphs_intersect_at_three_points (f : ℝ → ℝ) (h_inv : is_invertible f) :
  ∃ (xs : Finset ℝ), (∀ x ∈ xs, f (x^2) = f (x^6)) ∧ xs.card = 3 :=
by 
  sorry

end graphs_intersect_at_three_points_l1400_140074


namespace investment_calculation_l1400_140035

theorem investment_calculation
    (R Trishul Vishal Alok Harshit : ℝ)
    (hTrishul : Trishul = 0.9 * R)
    (hVishal : Vishal = 0.99 * R)
    (hAlok : Alok = 1.035 * Trishul)
    (hHarshit : Harshit = 0.95 * Vishal)
    (hTotal : R + Trishul + Vishal + Alok + Harshit = 22000) :
  R = 22000 / 3.8655 ∧
  Trishul = 0.9 * R ∧
  Vishal = 0.99 * R ∧
  Alok = 1.035 * Trishul ∧
  Harshit = 0.95 * Vishal ∧
  R + Trishul + Vishal + Alok + Harshit = 22000 :=
sorry

end investment_calculation_l1400_140035


namespace find_roots_of_polynomial_l1400_140032

theorem find_roots_of_polynomial :
  ∀ x : ℝ, (3 * x ^ 4 - x ^ 3 - 8 * x ^ 2 - x + 3 = 0) →
    (x = 2 ∨ x = 1/3 ∨ x = -1) :=
by
  intros x h
  sorry

end find_roots_of_polynomial_l1400_140032


namespace hydrogen_to_oxygen_ratio_l1400_140038

theorem hydrogen_to_oxygen_ratio (total_mass_water mass_hydrogen mass_oxygen : ℝ) 
(h1 : total_mass_water = 117)
(h2 : mass_hydrogen = 13)
(h3 : mass_oxygen = total_mass_water - mass_hydrogen) :
(mass_hydrogen / mass_oxygen) = 1 / 8 := 
sorry

end hydrogen_to_oxygen_ratio_l1400_140038


namespace cakes_baker_made_initially_l1400_140030

theorem cakes_baker_made_initially (x : ℕ) (h1 : x - 75 + 76 = 111) : x = 110 :=
by
  sorry

end cakes_baker_made_initially_l1400_140030


namespace quadratic_rewrite_correct_a_b_c_l1400_140033

noncomputable def quadratic_rewrite (x : ℝ) : ℝ := -6*x^2 + 36*x + 216

theorem quadratic_rewrite_correct_a_b_c :
  ∃ a b c : ℝ, quadratic_rewrite x = a * (x + b)^2 + c ∧ a + b + c = 261 :=
by
  sorry

end quadratic_rewrite_correct_a_b_c_l1400_140033


namespace constant_term_expansion_l1400_140073

theorem constant_term_expansion (r : Nat) (h : 12 - 3 * r = 0) :
  (Nat.choose 6 r) * 2^r = 240 :=
sorry

end constant_term_expansion_l1400_140073


namespace each_student_gets_8_pieces_l1400_140010

-- Define the number of pieces of candy
def candy : Nat := 344

-- Define the number of students
def students : Nat := 43

-- Define the number of pieces each student gets, which we need to prove
def pieces_per_student : Nat := candy / students

-- The proof problem statement
theorem each_student_gets_8_pieces : pieces_per_student = 8 :=
by
  -- This proof content is omitted as per instructions
  sorry

end each_student_gets_8_pieces_l1400_140010


namespace lesser_fraction_l1400_140055

theorem lesser_fraction (x y : ℚ) (h1 : x + y = 10 / 11) (h2 : x * y = 1 / 8) : min x y = (80 - 2 * Real.sqrt 632) / 176 := 
by sorry

end lesser_fraction_l1400_140055


namespace original_cost_price_l1400_140060

theorem original_cost_price 
  (C SP SP_new C_new : ℝ)
  (h1 : SP = 1.05 * C)
  (h2 : C_new = 0.95 * C)
  (h3 : SP_new = SP - 8)
  (h4 : SP_new = 1.045 * C_new) :
  C = 1600 :=
by
  sorry

end original_cost_price_l1400_140060


namespace g_at_negative_two_l1400_140076

-- Function definition
def g (x : ℝ) : ℝ := 3*x^5 - 4*x^4 + 2*x^3 - 5*x^2 - x + 8

-- Theorem statement
theorem g_at_negative_two : g (-2) = -186 :=
by
  -- Proof will go here, but it is skipped with sorry
  sorry

end g_at_negative_two_l1400_140076


namespace rectangle_side_excess_l1400_140004

theorem rectangle_side_excess
  (L W : ℝ)  -- length and width of the rectangle
  (x : ℝ)   -- percentage in excess for the first side
  (h1 : 0.95 * (L * (1 + x / 100) * W) = 1.102 * (L * W)) :
  x = 16 :=
by
  sorry

end rectangle_side_excess_l1400_140004


namespace exponent_multiplication_correct_l1400_140084

theorem exponent_multiplication_correct (a : ℝ) : a^3 * a^4 = a^7 := by
  sorry

end exponent_multiplication_correct_l1400_140084


namespace F_shaped_to_cube_l1400_140068

-- Define the problem context in Lean 4
structure F_shaped_figure :=
  (squares : Finset (Fin 5) )

structure additional_squares :=
  (label : String )

def is_valid_configuration (f : F_shaped_figure) (s : additional_squares) : Prop :=
  -- This function should encapsulate the logic for checking the validity of a configuration
  sorry -- Implementation of validity check is omitted (replacing it with sorry)

-- The main theorem statement
theorem F_shaped_to_cube (f : F_shaped_figure) (squares: Finset additional_squares) : 
  ∃ valid_squares : Finset additional_squares, valid_squares.card = 3 ∧ 
    ∀ s ∈ valid_squares, is_valid_configuration f s := 
sorry

end F_shaped_to_cube_l1400_140068


namespace three_digit_integers_product_36_l1400_140029

theorem three_digit_integers_product_36 : 
  ∃ (num_digits : ℕ), num_digits = 21 ∧ 
    ∀ (a b c : ℕ), 
      (1 ≤ a ∧ a ≤ 9) ∧ 
      (1 ≤ b ∧ b ≤ 9) ∧ 
      (1 ≤ c ∧ c ≤ 9) ∧ 
      (a * b * c = 36) → 
      num_digits = 21 :=
sorry

end three_digit_integers_product_36_l1400_140029


namespace solution_set_of_inequality_l1400_140008

theorem solution_set_of_inequality (x : ℝ) : 
  (x + 3) * (1 - x) ≥ 0 ↔ -3 ≤ x ∧ x ≤ 1 := 
sorry

end solution_set_of_inequality_l1400_140008


namespace math_problem_l1400_140096

theorem math_problem (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ x, (x - 1) * (deriv f x) ≥ 0) : 
  f 0 + f 2 ≥ 2 * f 1 :=
sorry

end math_problem_l1400_140096


namespace find_k_l1400_140056

-- Define the vectors
def e1 : ℝ × ℝ := (1, 0)
def e2 : ℝ × ℝ := (0, 1)

def a : ℝ × ℝ := (e1.1 - 2 * e2.1, e1.2 - 2 * e2.2)
def b (k : ℝ) : ℝ × ℝ := (k * e1.1 + e2.1, k * e1.2 + e2.2)

-- Define the parallel condition
def parallel (u v : ℝ × ℝ) : Prop :=
  u.1 * v.2 = u.2 * v.1

-- Statement of the problem translated to a Lean theorem
theorem find_k (k : ℝ) : 
  parallel a (b k) -> k = -1 / 2 := by
  sorry

end find_k_l1400_140056


namespace max_diameter_min_diameter_l1400_140081

-- Definitions based on problem conditions
def base_diameter : ℝ := 30
def positive_tolerance : ℝ := 0.03
def negative_tolerance : ℝ := 0.04

-- The corresponding proof problem statements in Lean 4
theorem max_diameter : base_diameter + positive_tolerance = 30.03 := sorry
theorem min_diameter : base_diameter - negative_tolerance = 29.96 := sorry

end max_diameter_min_diameter_l1400_140081


namespace total_cost_l1400_140066

variable (a b : ℝ)

def tomato_cost (a : ℝ) := 30 * a
def cabbage_cost (b : ℝ) := 50 * b

theorem total_cost (a b : ℝ) : 
  tomato_cost a + cabbage_cost b = 30 * a + 50 * b := 
by 
  unfold tomato_cost cabbage_cost
  sorry

end total_cost_l1400_140066


namespace tan_a1_a13_eq_sqrt3_l1400_140093

-- Definition of required constants and properties of the geometric sequence
noncomputable def a (n : Nat) : ℝ := sorry -- Geometric sequence definition (abstract)

-- Given condition: a_3 * a_11 + 2 * a_7^2 = 4π
axiom geom_seq_cond : a 3 * a 11 + 2 * (a 7)^2 = 4 * Real.pi

-- Property of geometric sequence: a_3 * a_11 = a_7^2
axiom geom_seq_property : a 3 * a 11 = (a 7)^2

-- To prove: tan(a_1 * a_13) = √3
theorem tan_a1_a13_eq_sqrt3 : Real.tan (a 1 * a 13) = Real.sqrt 3 := by
  sorry

end tan_a1_a13_eq_sqrt3_l1400_140093


namespace length_of_square_cut_off_l1400_140042

theorem length_of_square_cut_off 
  (x : ℝ) 
  (h_eq : (48 - 2 * x) * (36 - 2 * x) * x = 5120) : 
  x = 8 := 
sorry

end length_of_square_cut_off_l1400_140042


namespace evaluate_expression_l1400_140025

theorem evaluate_expression (a b : ℕ) (h_a : a = 15) (h_b : b = 7) :
  (a + b) ^ 2 - (a ^ 2 + b ^ 2) = 210 :=
by 
  rw [h_a, h_b]
  sorry

end evaluate_expression_l1400_140025


namespace ratio_of_perimeters_l1400_140049

theorem ratio_of_perimeters (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a^2 / b^2 = 49 / 64) : a / b = 7 / 8 :=
by
  sorry

end ratio_of_perimeters_l1400_140049


namespace find_slope_of_line_l_l1400_140028

-- Define the vectors OA and OB
def OA : ℝ × ℝ := (4, 1)
def OB : ℝ × ℝ := (2, -3)

-- The slope k is such that the lengths of projections of OA and OB on line l are equal
theorem find_slope_of_line_l (k : ℝ) :
  (|4 + k| = |2 - 3 * k|) → (k = 3 ∨ k = -1/2) :=
by
  -- Intentionally leave the proof out
  sorry

end find_slope_of_line_l_l1400_140028


namespace collinear_points_k_value_l1400_140085

theorem collinear_points_k_value : 
  (∀ k : ℝ, ∃ (a : ℝ) (b : ℝ), ∀ (x : ℝ) (y : ℝ),
    ((x, y) = (1, -2) ∨ (x, y) = (3, 2) ∨ (x, y) = (6, k / 3)) → y = a * x + b) → k = 24 :=
by
sorry

end collinear_points_k_value_l1400_140085


namespace olivia_hair_length_l1400_140080

def emilys_hair_length (logan_hair : ℕ) : ℕ := logan_hair + 6
def kates_hair_length (emily_hair : ℕ) : ℕ := emily_hair / 2
def jacks_hair_length (kate_hair : ℕ) : ℕ := (7 * kate_hair) / 2
def olivias_hair_length (jack_hair : ℕ) : ℕ := (2 * jack_hair) / 3

theorem olivia_hair_length
  (logan_hair : ℕ)
  (h_logan : logan_hair = 20)
  (h_emily : emilys_hair_length logan_hair = logan_hair + 6)
  (h_emily_value : emilys_hair_length logan_hair = 26)
  (h_kate : kates_hair_length (emilys_hair_length logan_hair) = 13)
  (h_jack : jacks_hair_length (kates_hair_length (emilys_hair_length logan_hair)) = 45)
  (h_olivia : olivias_hair_length (jacks_hair_length (kates_hair_length (emilys_hair_length logan_hair))) = 30) :
  olivias_hair_length
    (jacks_hair_length
      (kates_hair_length (emilys_hair_length logan_hair))) = 30 := by
  sorry

end olivia_hair_length_l1400_140080


namespace extracurricular_hours_l1400_140071

theorem extracurricular_hours :
  let soccer_hours_per_day := 2
  let soccer_days := 3
  let band_hours_per_day := 1.5
  let band_days := 2
  let total_soccer_hours := soccer_hours_per_day * soccer_days
  let total_band_hours := band_hours_per_day * band_days
  total_soccer_hours + total_band_hours = 9 := 
by
  -- The proof steps go here.
  sorry

end extracurricular_hours_l1400_140071


namespace total_students_left_l1400_140013

-- Definitions for given conditions
def initialBoys := 14
def initialGirls := 10
def boysDropOut := 4
def girlsDropOut := 3

-- The proof problem statement
theorem total_students_left : 
  initialBoys - boysDropOut + (initialGirls - girlsDropOut) = 17 := 
by
  sorry

end total_students_left_l1400_140013


namespace cats_remaining_l1400_140082

theorem cats_remaining 
  (n_initial n_given_away : ℝ) 
  (h_initial : n_initial = 17.0) 
  (h_given_away : n_given_away = 14.0) : 
  (n_initial - n_given_away) = 3.0 :=
by
  rw [h_initial, h_given_away]
  norm_num

end cats_remaining_l1400_140082


namespace joe_time_to_school_l1400_140036

theorem joe_time_to_school
    (r_w : ℝ) -- Joe's walking speed
    (t_w : ℝ) -- Time to walk halfway
    (t_stop : ℝ) -- Time stopped at the store
    (r_running_factor : ℝ) -- Factor by which running speed is faster than walking speed
    (initial_walk_time_halfway : t_w = 10)
    (store_stop_time : t_stop = 3)
    (running_speed_factor : r_running_factor = 4) :
    t_w + t_stop + t_w / r_running_factor = 15.5 :=
by
    -- Implementation skipped, just verifying statement is correctly captured
    sorry

end joe_time_to_school_l1400_140036


namespace minimum_bailing_rate_l1400_140045

theorem minimum_bailing_rate (distance_to_shore : ℝ) (row_speed : ℝ) (leak_rate : ℝ) (max_water_intake : ℝ)
  (time_to_shore : ℝ := distance_to_shore / row_speed * 60) (total_water_intake : ℝ := time_to_shore * leak_rate) :
  distance_to_shore = 1.5 → row_speed = 3 → leak_rate = 10 → max_water_intake = 40 →
  ∃ (bail_rate : ℝ), bail_rate ≥ 9 :=
by
  sorry

end minimum_bailing_rate_l1400_140045


namespace volume_region_between_concentric_spheres_l1400_140063

open Real

theorem volume_region_between_concentric_spheres (r1 r2 : ℝ) (h_r1 : r1 = 4) (h_r2 : r2 = 8) :
  (4 / 3 * π * r2^3 - 4 / 3 * π * r1^3) = 1792 / 3 * π :=
by
  sorry

end volume_region_between_concentric_spheres_l1400_140063


namespace smaller_angle_measure_l1400_140001

theorem smaller_angle_measure (α β : ℝ) (h1 : α + β = 90) (h2 : α = 4 * β) : β = 18 :=
by
  sorry

end smaller_angle_measure_l1400_140001


namespace daria_multiple_pizzas_l1400_140072

variable (m : ℝ)
variable (don_pizzas : ℝ) (total_pizzas : ℝ)

axiom don_pizzas_def : don_pizzas = 80
axiom total_pizzas_def : total_pizzas = 280

theorem daria_multiple_pizzas (m : ℝ) (don_pizzas : ℝ) (total_pizzas : ℝ) 
    (h1 : don_pizzas = 80) (h2 : total_pizzas = 280) 
    (h3 : total_pizzas = don_pizzas + m * don_pizzas) : 
    m = 2.5 :=
by sorry

end daria_multiple_pizzas_l1400_140072


namespace blind_box_problem_l1400_140059

theorem blind_box_problem (x y : ℕ) :
  x + y = 135 ∧ 2 * x = 3 * y :=
sorry

end blind_box_problem_l1400_140059


namespace solve_for_x_l1400_140046

-- Define the conditions
def percentage15_of_25 : ℝ := 0.15 * 25
def percentage12 (x : ℝ) : ℝ := 0.12 * x
def condition (x : ℝ) : Prop := percentage15_of_25 + percentage12 x = 9.15

-- The target statement to prove
theorem solve_for_x : ∃ x : ℝ, condition x ∧ x = 45 :=
by 
  -- The proof is omitted
  sorry

end solve_for_x_l1400_140046


namespace solve_quadratic_equation_l1400_140051

theorem solve_quadratic_equation (x : ℝ) : 
  (2 * x^2 - 4 * x = 6 - 3 * x) ↔ 
  (x = -3/2 ∨ x = 2) :=
by
  sorry

end solve_quadratic_equation_l1400_140051


namespace least_upper_bound_neg_expression_l1400_140061

noncomputable def least_upper_bound : ℝ :=
  - (9 / 2)

theorem least_upper_bound_neg_expression (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 1) :
  ∃ M, M = least_upper_bound ∧
  ∀ x, (∀ a b, 0 < a → 0 < b → a + b = 1 → x ≤ - (1 / (2 * a)) - (2 / b)) ↔ x ≤ M :=
sorry

end least_upper_bound_neg_expression_l1400_140061


namespace pat_stickers_l1400_140040

theorem pat_stickers (stickers_given_away stickers_left : ℝ) 
(h_given_away : stickers_given_away = 22.0)
(h_left : stickers_left = 17.0) : 
(stickers_given_away + stickers_left = 39) :=
by
  sorry

end pat_stickers_l1400_140040


namespace cost_difference_l1400_140043

/-- The selling price and cost of pants -/
def selling_price : ℕ := 34
def store_cost : ℕ := 26

/-- The proof that the store paid 8 dollars less than the selling price -/
theorem cost_difference : selling_price - store_cost = 8 := by
  sorry

end cost_difference_l1400_140043


namespace find_a_mul_b_l1400_140039

theorem find_a_mul_b (x y z a b : ℝ)
  (h1 : a = x)
  (h2 : b = y)
  (h3 : x + x = y * x)
  (h4 : b = z)
  (h5 : x + x = z * z)
  (h6 : y = 3)
  : a * b = 4 := by
  sorry

end find_a_mul_b_l1400_140039


namespace more_red_flowers_than_white_l1400_140026

-- Definitions based on given conditions
def yellow_and_white := 13
def red_and_yellow := 17
def red_and_white := 14
def blue_and_yellow := 16

-- Definitions based on the requirements of the problem
def red_flowers := red_and_yellow + red_and_white
def white_flowers := yellow_and_white + red_and_white

-- Theorem to prove the number of more flowers containing red than white
theorem more_red_flowers_than_white : red_flowers - white_flowers = 4 := by
  sorry

end more_red_flowers_than_white_l1400_140026


namespace find_n_l1400_140014

variable {a_n : ℕ → ℤ}
variable (a2 : ℤ) (an : ℤ) (d : ℤ) (n : ℕ)

def arithmetic_sequence (a2 : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  a2 + (n - 2) * d

theorem find_n 
  (h1 : a2 = 12)
  (h2 : an = -20)
  (h3 : d = -2)
  : n = 18 := by
  sorry

end find_n_l1400_140014


namespace probability_top_red_second_black_l1400_140007

def num_red_cards : ℕ := 39
def num_black_cards : ℕ := 39
def total_cards : ℕ := 78

theorem probability_top_red_second_black :
  (num_red_cards * num_black_cards) / (total_cards * (total_cards - 1)) = 507 / 2002 := 
sorry

end probability_top_red_second_black_l1400_140007


namespace frank_money_l1400_140062

theorem frank_money (X : ℝ) (h1 : (3/4) * (4/5) * X = 360) : X = 600 :=
sorry

end frank_money_l1400_140062


namespace coefficient_of_pi_x_over_5_l1400_140022

-- Definition of the function where we find the coefficient
def coefficient_of_fraction (expr : ℝ) : ℝ := sorry

-- Statement with proof obligation
theorem coefficient_of_pi_x_over_5 :
  coefficient_of_fraction (π * x / 5) = π / 5 :=
sorry

end coefficient_of_pi_x_over_5_l1400_140022


namespace monthly_rent_l1400_140058

-- Definitions based on the given conditions
def length_ft : ℕ := 360
def width_ft : ℕ := 1210
def sq_feet_per_acre : ℕ := 43560
def cost_per_acre_per_month : ℕ := 60

-- Statement of the problem
theorem monthly_rent : (length_ft * width_ft / sq_feet_per_acre) * cost_per_acre_per_month = 600 := sorry

end monthly_rent_l1400_140058


namespace find_245th_digit_in_decimal_rep_of_13_div_17_l1400_140098

-- Definition of the repeating sequence for the fractional division
def repeating_sequence_13_div_17 : List Char := ['7', '6', '4', '7', '0', '5', '8', '8', '2', '3', '5', '2', '9', '4', '1', '1']

-- Period of the repeating sequence
def period : ℕ := 16

-- Function to find the n-th digit in a repeating sequence
def nth_digit_in_repeating_sequence (seq : List Char) (period : ℕ) (n : ℕ) : Char :=
  seq.get! ((n - 1) % period)

-- Hypothesis: The repeating sequence of 13/17 and its period
axiom repeating_sequence_period : repeating_sequence_13_div_17.length = period

-- The theorem to prove
theorem find_245th_digit_in_decimal_rep_of_13_div_17 : nth_digit_in_repeating_sequence repeating_sequence_13_div_17 period 245 = '7' := 
  by
  sorry

end find_245th_digit_in_decimal_rep_of_13_div_17_l1400_140098


namespace solve_fraction_l1400_140086

theorem solve_fraction (x : ℚ) : (x^2 + 3*x + 5) / (x + 6) = x + 7 ↔ x = -37 / 10 :=
by
  sorry

end solve_fraction_l1400_140086


namespace fraction_of_primes_is_prime_l1400_140027

theorem fraction_of_primes_is_prime
  (p q r : ℕ) 
  (hp : Nat.Prime p)
  (hq : Nat.Prime q)
  (hr : Nat.Prime r)
  (h : ∃ k : ℕ, p * q * r = k * (p + q + r)) :
  Nat.Prime (p * q * r / (p + q + r)) := 
sorry

end fraction_of_primes_is_prime_l1400_140027


namespace smallest_value_floor_l1400_140020

theorem smallest_value_floor (a b c : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : c > 0) :
  (⌊(2 * a + b) / c⌋ + ⌊(b + 2 * c) / a⌋ + ⌊(c + 2 * a) / b⌋) = 9 :=
sorry

end smallest_value_floor_l1400_140020


namespace pages_to_read_tomorrow_l1400_140021

-- Define the conditions
def total_pages : ℕ := 100
def pages_yesterday : ℕ := 35
def pages_today : ℕ := pages_yesterday - 5

-- Lean statement for the proof problem
theorem pages_to_read_tomorrow : (total_pages - (pages_yesterday + pages_today) = 35) :=
by
  let yesterday := pages_yesterday
  let today := pages_today
  let read_so_far := yesterday + today
  have read_so_far_eq : yesterday + today = 65 := by sorry
  have total_eq : total_pages - read_so_far = 35 := by sorry
  exact total_eq

end pages_to_read_tomorrow_l1400_140021


namespace probability_quarter_circle_is_pi_div_16_l1400_140099

open Real

noncomputable def probability_quarter_circle : ℝ :=
  let side_length := 2
  let total_area := side_length * side_length
  let quarter_circle_area := π / 4
  quarter_circle_area / total_area

theorem probability_quarter_circle_is_pi_div_16 :
  probability_quarter_circle = π / 16 :=
by
  sorry

end probability_quarter_circle_is_pi_div_16_l1400_140099


namespace percentage_of_students_owning_only_cats_is_10_percent_l1400_140031

def total_students : ℕ := 500
def cat_owners : ℕ := 75
def dog_owners : ℕ := 150
def both_cat_and_dog_owners : ℕ := 25
def only_cat_owners : ℕ := cat_owners - both_cat_and_dog_owners
def percent_owning_only_cats : ℚ := (only_cat_owners * 100) / total_students

theorem percentage_of_students_owning_only_cats_is_10_percent : percent_owning_only_cats = 10 := by
  sorry

end percentage_of_students_owning_only_cats_is_10_percent_l1400_140031


namespace find_f2_l1400_140050

noncomputable def f (x : ℝ) : ℝ := sorry -- Placeholder function definition

theorem find_f2 (h : ∀ x : ℝ, f x + 2 * f (1 - x) = x^3 + 1) : f 2 = -3 :=
by
  -- Lean proof goes here
  sorry

end find_f2_l1400_140050


namespace find_X_l1400_140016

-- Define the variables for income, tax, and the variable X
def income := 58000
def tax := 8000

-- Define the tax formula as per the problem
def tax_formula (X : ℝ) : ℝ :=
  0.11 * X + 0.20 * (income - X)

-- The theorem we want to prove
theorem find_X :
  ∃ X : ℝ, tax_formula X = tax ∧ X = 40000 :=
sorry

end find_X_l1400_140016


namespace friendly_triangle_angle_l1400_140091

theorem friendly_triangle_angle (α : ℝ) (β : ℝ) (γ : ℝ) (hα12β : α = 2 * β) (h_sum : α + β + γ = 180) :
    (α = 42 ∨ α = 84 ∨ α = 92) ∧ (42 = β ∨ 42 = γ) := 
sorry

end friendly_triangle_angle_l1400_140091


namespace list_size_is_2017_l1400_140092

def has_sum (L : List ℤ) (n : ℤ) : Prop :=
  List.sum L = n

def has_product (L : List ℤ) (n : ℤ) : Prop :=
  List.prod L = n

def includes (L : List ℤ) (n : ℤ) : Prop :=
  n ∈ L

theorem list_size_is_2017 
(L : List ℤ) :
  has_sum L 2018 ∧ 
  has_product L 2018 ∧ 
  includes L 2018 
  → L.length = 2017 :=
by 
  sorry

end list_size_is_2017_l1400_140092


namespace hyperbola_sum_l1400_140023

theorem hyperbola_sum (h k a b : ℝ) (c : ℝ)
  (h_eq : h = 3)
  (k_eq : k = -5)
  (a_eq : a = 5)
  (c_eq : c = 7)
  (c_squared_eq : c^2 = a^2 + b^2) :
  h + k + a + b = 3 + 2 * Real.sqrt 6 :=
by
  rw [h_eq, k_eq, a_eq, c_eq] at *
  sorry

end hyperbola_sum_l1400_140023


namespace new_estimated_y_value_l1400_140054

theorem new_estimated_y_value
  (initial_slope : ℝ) (initial_intercept : ℝ) (avg_x_initial : ℝ)
  (datapoints_removed_low_x : ℝ) (datapoints_removed_high_x : ℝ)
  (datapoints_removed_low_y : ℝ) (datapoints_removed_high_y : ℝ)
  (new_slope : ℝ) 
  (x_value : ℝ)
  (estimated_y_new : ℝ) :
  initial_slope = 1.5 →
  initial_intercept = 1 →
  avg_x_initial = 2 →
  datapoints_removed_low_x = 2.6 →
  datapoints_removed_high_x = 1.4 →
  datapoints_removed_low_y = 2.8 →
  datapoints_removed_high_y = 5.2 →
  new_slope = 1.4 →
  x_value = 6 →
  estimated_y_new = new_slope * x_value + (4 - new_slope * avg_x_initial) →
  estimated_y_new = 9.6 := by
  sorry

end new_estimated_y_value_l1400_140054


namespace integer_solution_pair_l1400_140009

theorem integer_solution_pair (x y : ℤ) (h : x^2 + x * y = y^2) : (x = 0 ∧ y = 0) :=
by
  sorry

end integer_solution_pair_l1400_140009


namespace additional_flour_minus_salt_l1400_140005

structure CakeRecipe where
  flour    : ℕ
  sugar    : ℕ
  salt     : ℕ

def MaryHasAdded (cups_flour : ℕ) (cups_sugar : ℕ) (cups_salt : ℕ) : Prop :=
  cups_flour = 2 ∧ cups_sugar = 0 ∧ cups_salt = 0

variable (r : CakeRecipe)

theorem additional_flour_minus_salt (H : MaryHasAdded 2 0 0) : 
  (r.flour - 2) - r.salt = 3 :=
sorry

end additional_flour_minus_salt_l1400_140005


namespace y_share_per_rupee_l1400_140053

theorem y_share_per_rupee (a p : ℝ) (h1 : a * p = 18)
                            (h2 : p + a * p + 0.30 * p = 70) :
    a = 0.45 :=
by 
  sorry

end y_share_per_rupee_l1400_140053


namespace pear_weight_l1400_140048

theorem pear_weight
  (w_apple : ℕ)
  (p_weight_relation : 12 * w_apple = 8 * P + 5400)
  (apple_weight : w_apple = 530) :
  P = 120 :=
by
  -- sorry, proof is omitted as per instructions
  sorry

end pear_weight_l1400_140048


namespace average_speed_is_70_l1400_140017

noncomputable def average_speed (d₁ d₂ d₃ d₄ t₁ t₂ t₃ t₄ : ℝ) : ℝ :=
  (d₁ + d₂ + d₃ + d₄) / (t₁ + t₂ + t₃ + t₄)

theorem average_speed_is_70 :
  let d₁ := 30
  let s₁ := 60
  let t₁ := d₁ / s₁
  let d₂ := 35
  let s₂ := 70
  let t₂ := d₂ / s₂
  let d₃ := 80
  let t₃ := 1
  let s₃ := d₃ / t₃
  let s₄ := 55
  let t₄ := 20/60.0
  let d₄ := s₄ * t₄
  average_speed d₁ d₂ d₃ d₄ t₁ t₂ t₃ t₄ = 70 :=
by
  sorry

end average_speed_is_70_l1400_140017


namespace longest_segment_is_CD_l1400_140034

-- Define points A, B, C, D
def A := (-3, 0)
def B := (0, 2)
def C := (3, 0)
def D := (0, -1)

-- Angles in triangle ABD
def angle_ABD := 35
def angle_BAD := 95
def angle_ADB := 50

-- Angles in triangle BCD
def angle_BCD := 55
def angle_BDC := 60
def angle_CBD := 65

-- Length comparison conclusion from triangle ABD
axiom compare_lengths_ABD : ∀ (AD AB BD : ℝ), AD < AB ∧ AB < BD

-- Length comparison conclusion from triangle BCD
axiom compare_lengths_BCD : ∀ (BC BD CD : ℝ), BC < BD ∧ BD < CD

-- Combine results
theorem longest_segment_is_CD : ∀ (AD AB BD BC CD : ℝ), AD < AB → AB < BD → BC < BD → BD < CD → CD ≥ AD ∧ CD ≥ AB ∧ CD ≥ BD ∧ CD ≥ BC :=
by
  intros AD AB BD BC CD h1 h2 h3 h4
  sorry

end longest_segment_is_CD_l1400_140034


namespace smallest_possible_time_for_travel_l1400_140044

theorem smallest_possible_time_for_travel :
  ∃ t : ℝ, (∀ D M P : ℝ, D = 6 → M = 6 → P = 6 → 
    ∀ motorcycle_speed distance : ℝ, motorcycle_speed = 90 → distance = 135 → 
    t < 3.9) :=
  sorry

end smallest_possible_time_for_travel_l1400_140044


namespace parallel_planes_of_skew_lines_l1400_140090

variables {Plane : Type*} {Line : Type*}
variables (α β : Plane)
variables (a b : Line)

-- Conditions
def is_parallel (p1 p2 : Plane) : Prop := sorry -- Parallel planes relation
def line_in_plane (l : Line) (p : Plane) : Prop := sorry -- Line in plane relation
def line_parallel_plane (l : Line) (p : Plane) : Prop := sorry -- Line parallel to plane relation
def is_skew_lines (l1 l2 : Line) : Prop := sorry -- Skew lines relation

-- Theorem to prove
theorem parallel_planes_of_skew_lines 
  (h1 : line_in_plane a α)
  (h2 : line_in_plane b β)
  (h3 : line_parallel_plane a β)
  (h4 : line_parallel_plane b α)
  (h5 : is_skew_lines a b) :
  is_parallel α β :=
sorry

end parallel_planes_of_skew_lines_l1400_140090


namespace total_lobster_pounds_l1400_140065

theorem total_lobster_pounds
  (combined_other_harbors : ℕ)
  (hooper_bay : ℕ)
  (H1 : combined_other_harbors = 160)
  (H2 : hooper_bay = 2 * combined_other_harbors) :
  combined_other_harbors + hooper_bay = 480 :=
by
  -- proof goes here
  sorry

end total_lobster_pounds_l1400_140065


namespace circle_equation_translation_l1400_140083

theorem circle_equation_translation (x y : ℝ) :
  x^2 + y^2 - 4 * x + 6 * y - 68 = 0 → (x - 2)^2 + (y + 3)^2 = 81 :=
by
  intro h
  sorry

end circle_equation_translation_l1400_140083


namespace porch_length_is_6_l1400_140064

-- Define the conditions for the house and porch areas
def house_length : ℝ := 20.5
def house_width : ℝ := 10
def porch_width : ℝ := 4.5
def total_shingle_area : ℝ := 232

-- Define the area calculations
def house_area : ℝ := house_length * house_width
def porch_area : ℝ := total_shingle_area - house_area

-- The theorem to prove
theorem porch_length_is_6 : porch_area / porch_width = 6 := by
  sorry

end porch_length_is_6_l1400_140064


namespace share_a_is_240_l1400_140047

def total_profit : ℕ := 630

def initial_investment_a : ℕ := 3000
def initial_investment_b : ℕ := 4000

def months_a1 : ℕ := 8
def months_a2 : ℕ := 4
def investment_a1 : ℕ := initial_investment_a * months_a1
def investment_a2 : ℕ := (initial_investment_a - 1000) * months_a2
def total_investment_a : ℕ := investment_a1 + investment_a2

def months_b1 : ℕ := 8
def months_b2 : ℕ := 4
def investment_b1 : ℕ := initial_investment_b * months_b1
def investment_b2 : ℕ := (initial_investment_b + 1000) * months_b2
def total_investment_b : ℕ := investment_b1 + investment_b2

def ratio_a : ℕ := 8
def ratio_b : ℕ := 13
def total_ratio : ℕ := ratio_a + ratio_b

noncomputable def share_a (total_profit : ℕ) (ratio_a ratio_total : ℕ) : ℕ :=
  (ratio_a * total_profit) / ratio_total

theorem share_a_is_240 :
  share_a total_profit ratio_a total_ratio = 240 :=
by
  sorry

end share_a_is_240_l1400_140047


namespace fruitseller_apples_l1400_140097

theorem fruitseller_apples (x : ℝ) (sold_percent remaining_apples : ℝ) 
  (h_sold : sold_percent = 0.80) 
  (h_remaining : remaining_apples = 500) 
  (h_equation : (1 - sold_percent) * x = remaining_apples) : 
  x = 2500 := 
by 
  sorry

end fruitseller_apples_l1400_140097


namespace base_equivalence_l1400_140006

theorem base_equivalence :
  let n_7 := 4 * 7 + 3  -- 43 in base 7 expressed in base 10.
  ∃ d : ℕ, (3 * d + 4 = n_7) → d = 9 :=
by
  let n_7 := 31
  sorry

end base_equivalence_l1400_140006


namespace statement_C_correct_l1400_140012

theorem statement_C_correct (a b : ℝ) (h1 : a < b) (h2 : a * b ≠ 0) : (1 / a) > (1 / b) :=
sorry

end statement_C_correct_l1400_140012


namespace Julie_can_print_complete_newspapers_l1400_140070

def sheets_in_box_A : ℕ := 4 * 200
def sheets_in_box_B : ℕ := 3 * 350
def total_sheets : ℕ := sheets_in_box_A + sheets_in_box_B

def front_section_sheets : ℕ := 10
def sports_section_sheets : ℕ := 7
def arts_section_sheets : ℕ := 5
def events_section_sheets : ℕ := 3

def sheets_per_newspaper : ℕ := front_section_sheets + sports_section_sheets + arts_section_sheets + events_section_sheets

theorem Julie_can_print_complete_newspapers : total_sheets / sheets_per_newspaper = 74 := by
  sorry

end Julie_can_print_complete_newspapers_l1400_140070


namespace survey_pop_and_coke_l1400_140052

theorem survey_pop_and_coke (total_people : ℕ) (angle_pop angle_coke : ℕ) 
  (h_total : total_people = 500) (h_angle_pop : angle_pop = 240) (h_angle_coke : angle_coke = 90) :
  ∃ (pop_people coke_people : ℕ), pop_people = 333 ∧ coke_people = 125 :=
by 
  sorry

end survey_pop_and_coke_l1400_140052


namespace fernanda_savings_calculation_l1400_140002

theorem fernanda_savings_calculation :
  ∀ (aryan_debt kyro_debt aryan_payment kyro_payment savings total_savings : ℝ),
    aryan_debt = 1200 ∧
    aryan_debt = 2 * kyro_debt ∧
    aryan_payment = (60 / 100) * aryan_debt ∧
    kyro_payment = (80 / 100) * kyro_debt ∧
    savings = 300 ∧
    total_savings = savings + aryan_payment + kyro_payment →
    total_savings = 1500 := by
    sorry

end fernanda_savings_calculation_l1400_140002


namespace original_lettuce_cost_l1400_140087

theorem original_lettuce_cost
  (original_cost: ℝ) (tomatoes_original: ℝ) (tomatoes_new: ℝ) (celery_original: ℝ) (celery_new: ℝ) (lettuce_new: ℝ)
  (delivery_tip: ℝ) (new_bill: ℝ)
  (H1: original_cost = 25)
  (H2: tomatoes_original = 0.99) (H3: tomatoes_new = 2.20)
  (H4: celery_original = 1.96) (H5: celery_new = 2.00)
  (H6: lettuce_new = 1.75)
  (H7: delivery_tip = 8.00)
  (H8: new_bill = 35) :
  ∃ (lettuce_original: ℝ), lettuce_original = 1.00 :=
by
  let tomatoes_diff := tomatoes_new - tomatoes_original
  let celery_diff := celery_new - celery_original
  let new_cost_without_lettuce := original_cost + tomatoes_diff + celery_diff
  let new_cost_excl_delivery := new_bill - delivery_tip
  have lettuce_diff := new_cost_excl_delivery - new_cost_without_lettuce
  let lettuce_original := lettuce_new - lettuce_diff
  exists lettuce_original
  sorry

end original_lettuce_cost_l1400_140087


namespace moles_of_NaHCO3_needed_l1400_140078

theorem moles_of_NaHCO3_needed 
  (HC2H3O2_moles: ℕ)
  (H2O_moles: ℕ)
  (NaHCO3_HC2H3O2_molar_ratio: ℕ)
  (reaction: NaHCO3_HC2H3O2_molar_ratio = 1 ∧ H2O_moles = 3) :
  ∃ NaHCO3_moles : ℕ, NaHCO3_moles = 3 :=
by
  sorry

end moles_of_NaHCO3_needed_l1400_140078


namespace rhombus_diagonal_length_l1400_140011

theorem rhombus_diagonal_length (d1 d2 : ℝ) (A : ℝ) (h1 : d1 = 25) (h2 : A = 250) (h3 : A = (d1 * d2) / 2) : d2 = 20 := 
by
  rw [h1, h2] at h3
  sorry

end rhombus_diagonal_length_l1400_140011


namespace election_votes_l1400_140019

theorem election_votes (T : ℝ) (Vf Va Vn : ℝ)
  (h1 : Va = 0.375 * T)
  (h2 : Vn = 0.125 * T)
  (h3 : Vf = Va + 78)
  (h4 : T = Vf + Va + Vn) :
  T = 624 :=
by
  sorry

end election_votes_l1400_140019


namespace average_age_increase_39_l1400_140077

variable (n : ℕ) (A : ℝ)
noncomputable def average_age_increase (r : ℝ) : Prop :=
  (r = 7) →
  (n + 1) * (A + r) = n * A + 39 →
  (n + 1) * (A - 1) = n * A + 15 →
  r = 7

theorem average_age_increase_39 : ∀ (n : ℕ) (A : ℝ), average_age_increase n A 7 :=
by
  intros n A
  unfold average_age_increase
  intros hr h1 h2
  exact hr

end average_age_increase_39_l1400_140077


namespace caterpillar_reaches_top_in_16_days_l1400_140024

-- Define the constants for the problem
def pole_height : ℕ := 20
def daytime_climb : ℕ := 5
def nighttime_slide : ℕ := 4

-- Define the final result we want to prove
theorem caterpillar_reaches_top_in_16_days :
  ∃ days : ℕ, days = 16 ∧ 
  ((20 - 5) / (daytime_climb - nighttime_slide) + 1) = 16 := by
  sorry

end caterpillar_reaches_top_in_16_days_l1400_140024


namespace find_window_cost_l1400_140069

-- Definitions (conditions)
def total_damages : ℕ := 1450
def cost_of_tire : ℕ := 250
def number_of_tires : ℕ := 3
def cost_of_tires := number_of_tires * cost_of_tire

-- The cost of the window that needs to be proven
def window_cost := total_damages - cost_of_tires

-- We state the theorem that the window costs $700 and provide a sorry as placeholder for its proof
theorem find_window_cost : window_cost = 700 :=
by sorry

end find_window_cost_l1400_140069


namespace a_alone_can_finish_job_l1400_140067

def work_in_one_day (A B : ℕ) : Prop := 1/A + 1/B = 1/40

theorem a_alone_can_finish_job (A B : ℕ)
  (work_rate : work_in_one_day A B) 
  (together_10_days : 10 * (1/A + 1/B) = 1/4) 
  (a_21_days : 21 * (1/A) = 3/4) : 
  A = 28 := 
sorry

end a_alone_can_finish_job_l1400_140067


namespace fraction_inequality_l1400_140075

theorem fraction_inequality (a b c : ℝ) : 
  (a / (a + 2 * b + c)) + (b / (a + b + 2 * c)) + (c / (2 * a + b + c)) ≥ 3 / 4 := 
by
  sorry

end fraction_inequality_l1400_140075


namespace combined_function_is_linear_l1400_140018

def original_parabola (x : ℝ) : ℝ := 3 * x^2 + 4 * x - 5

def reflected_parabola (x : ℝ) : ℝ := -original_parabola x

def translated_original_parabola (x : ℝ) : ℝ := 3 * (x - 4)^2 + 4 * (x - 4) - 5

def translated_reflected_parabola (x : ℝ) : ℝ := -3 * (x + 6)^2 - 4 * (x + 6) + 5

def combined_function (x : ℝ) : ℝ := translated_original_parabola x + translated_reflected_parabola x

theorem combined_function_is_linear : ∃ (a b : ℝ), ∀ x : ℝ, combined_function x = a * x + b := by
  sorry

end combined_function_is_linear_l1400_140018


namespace product_of_roots_l1400_140089

-- Let x₁ and x₂ be roots of the quadratic equation x^2 + x - 1 = 0
theorem product_of_roots (x₁ x₂ : ℝ) (h₁ : x₁^2 + x₁ - 1 = 0) (h₂ : x₂^2 + x₂ - 1 = 0) :
  x₁ * x₂ = -1 :=
sorry

end product_of_roots_l1400_140089


namespace liam_savings_per_month_l1400_140000

theorem liam_savings_per_month (trip_cost bill_cost left_after_bills : ℕ) 
                               (months_in_two_years : ℕ) (total_savings_per_month : ℕ) :
  trip_cost = 7000 →
  bill_cost = 3500 →
  left_after_bills = 8500 →
  months_in_two_years = 24 →
  total_savings_per_month = 19000 →
  total_savings_per_month / months_in_two_years = 79167 / 100 :=
by
  intros
  sorry

end liam_savings_per_month_l1400_140000


namespace expression_may_not_hold_l1400_140095

theorem expression_may_not_hold (a b c : ℝ) (h : a = b) (hc : c = 0) :
  a = b → ¬ (a / c = b / c) := 
by
  intro hab
  intro h_div
  sorry

end expression_may_not_hold_l1400_140095


namespace cakes_sold_l1400_140037

theorem cakes_sold (total_made : ℕ) (cakes_left : ℕ) (cakes_sold : ℕ) :
  total_made = 217 ∧ cakes_left = 72 → cakes_sold = 145 :=
by
  -- Assuming total_made is 217 and cakes_left is 72, we need to show cakes_sold = 145
  sorry

end cakes_sold_l1400_140037


namespace y_real_for_all_x_l1400_140003

theorem y_real_for_all_x (x : ℝ) : ∃ y : ℝ, 9 * y^2 + 3 * x * y + x - 3 = 0 :=
by
  sorry

end y_real_for_all_x_l1400_140003


namespace always_positive_iff_k_gt_half_l1400_140041

theorem always_positive_iff_k_gt_half (k : ℝ) :
  (∀ x : ℝ, k * x^2 + x + k > 0) ↔ k > 0.5 :=
sorry

end always_positive_iff_k_gt_half_l1400_140041


namespace incircle_intersections_equation_l1400_140015

-- Assume a triangle ABC with the given configuration
variables {A B C D E F M N : Type}

-- Incircle touches sides CA, AB at points E, F respectively
-- Lines BE and CF intersect the incircle again at points M and N respectively

theorem incircle_intersections_equation
  (triangle_ABC : Type)
  (incircle_I : Type)
  (touch_CA : Type)
  (touch_AB : Type)
  (intersect_BE : Type)
  (intersect_CF : Type)
  (E F : triangle_ABC → incircle_I)
  (M N : intersect_BE → intersect_CF)
  : 
  MN * EF = 3 * MF * NE :=
by 
  -- Sorry as the proof is omitted
  sorry

end incircle_intersections_equation_l1400_140015
