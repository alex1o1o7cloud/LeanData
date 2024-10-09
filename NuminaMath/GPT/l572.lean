import Mathlib

namespace initial_paint_l572_57292

variable (total_needed : ℕ) (paint_bought : ℕ) (still_needed : ℕ)

theorem initial_paint (h_total_needed : total_needed = 70)
                      (h_paint_bought : paint_bought = 23)
                      (h_still_needed : still_needed = 11) : 
                      ∃ x : ℕ, x = 36 :=
by
  sorry

end initial_paint_l572_57292


namespace martin_improved_lap_time_l572_57205

def initial_laps := 15
def initial_time := 45 -- in minutes
def final_laps := 18
def final_time := 42 -- in minutes

noncomputable def initial_lap_time := initial_time / initial_laps
noncomputable def final_lap_time := final_time / final_laps
noncomputable def improvement := initial_lap_time - final_lap_time

theorem martin_improved_lap_time : improvement = 2 / 3 := by 
  sorry

end martin_improved_lap_time_l572_57205


namespace total_pupils_correct_l572_57219

-- Definitions of the conditions
def number_of_girls : ℕ := 308
def number_of_boys : ℕ := 318

-- Definition of the number of pupils
def total_number_of_pupils : ℕ := number_of_girls + number_of_boys

-- The theorem to be proven
theorem total_pupils_correct : total_number_of_pupils = 626 := by
  -- The proof would go here
  sorry

end total_pupils_correct_l572_57219


namespace lemon_pie_degrees_l572_57256

-- Defining the constants
def total_students : ℕ := 45
def chocolate_pie : ℕ := 15
def apple_pie : ℕ := 10
def blueberry_pie : ℕ := 9

-- Defining the remaining students
def remaining_students := total_students - (chocolate_pie + apple_pie + blueberry_pie)

-- Half of the remaining students prefer cherry pie and half prefer lemon pie
def students_prefer_cherry : ℕ := remaining_students / 2
def students_prefer_lemon : ℕ := remaining_students / 2

-- Defining the degree measure function
def degrees (students : ℕ) := (students * 360) / total_students

-- Proof statement
theorem lemon_pie_degrees : degrees students_prefer_lemon = 48 := by
  sorry  -- proof omitted

end lemon_pie_degrees_l572_57256


namespace problem_l572_57241

noncomputable def f : ℝ → ℝ := sorry

theorem problem (x : ℝ) (h : ∀ x : ℝ, f (4 * x) = 4) : f (2 * x) = 4 :=
by
  sorry

end problem_l572_57241


namespace find_smallest_integer_y_l572_57260

theorem find_smallest_integer_y : ∃ y : ℤ, (8 / 12 : ℚ) < (y / 15) ∧ ∀ z : ℤ, z < y → ¬ ((8 / 12 : ℚ) < (z / 15)) :=
by
  sorry

end find_smallest_integer_y_l572_57260


namespace train_stoppage_time_l572_57215

-- Definitions of the conditions
def speed_excluding_stoppages : ℝ := 48 -- in kmph
def speed_including_stoppages : ℝ := 32 -- in kmph
def time_per_hour : ℝ := 60 -- 60 minutes in an hour

-- The problem statement
theorem train_stoppage_time :
  (speed_excluding_stoppages - speed_including_stoppages) * time_per_hour / speed_excluding_stoppages = 20 :=
by
  -- Initial statement
  sorry

end train_stoppage_time_l572_57215


namespace expression_value_l572_57268

/--
Prove that for a = 51 and b = 15, the expression (a + b)^2 - (a^2 + b^2) equals 1530.
-/
theorem expression_value (a b : ℕ) (h1 : a = 51) (h2 : b = 15) : (a + b)^2 - (a^2 + b^2) = 1530 := by
  rw [h1, h2]
  sorry

end expression_value_l572_57268


namespace problem1_l572_57218

theorem problem1 (a b : ℝ) (i : ℝ) (h : (a-2*i)*i = b-i) : a^2 + b^2 = 5 := by
  sorry

end problem1_l572_57218


namespace ratio_of_x_to_y_l572_57291

theorem ratio_of_x_to_y (x y : ℝ) (h : (12 * x - 5 * y) / (15 * x - 3 * y) = 4 / 7) : x / y = 23 / 24 :=
sorry

end ratio_of_x_to_y_l572_57291


namespace eggs_removed_l572_57236

theorem eggs_removed (initial remaining : ℕ) (h1 : initial = 27) (h2 : remaining = 20) : initial - remaining = 7 :=
by
  sorry

end eggs_removed_l572_57236


namespace shop_earnings_correct_l572_57263

theorem shop_earnings_correct :
  let cola_price := 3
  let juice_price := 1.5
  let water_price := 1
  let cola_sold := 15
  let juice_sold := 12
  let water_sold := 25
  cola_price * cola_sold + juice_price * juice_sold + water_price * water_sold = 88 := 
by 
  sorry

end shop_earnings_correct_l572_57263


namespace ellipse_foci_y_axis_l572_57261

theorem ellipse_foci_y_axis (k : ℝ) (h_eq : ∀ x y : ℝ, x^2 + k * y^2 = 2)
  (h_foci : ∀ x y : ℝ, x^2 ≤ 2 ∧ k * y^2 ≤ 2) :
  0 < k ∧ k < 1 :=
  sorry

end ellipse_foci_y_axis_l572_57261


namespace correct_operation_l572_57247

theorem correct_operation (a : ℕ) :
  (a^2 * a^3 = a^5) ∧ ¬((a^2)^3 = a^5) ∧ ¬(a^6 / a^2 = a^3) ∧ ¬(3 * a^2 - 2 * a = a^2) :=
by
  sorry

end correct_operation_l572_57247


namespace card_count_l572_57224

theorem card_count (x y : ℕ) (h1 : x + y + 2 = 10) (h2 : 3 * x + 4 * y + 10 = 39) : x = 3 :=
by {
  sorry
}

end card_count_l572_57224


namespace repeating_decimal_sum_l572_57226

theorem repeating_decimal_sum (x : ℚ) (hx : x = 0.417) :
  let num := 46
  let denom := 111
  let sum := num + denom
  sum = 157 :=
by
  sorry

end repeating_decimal_sum_l572_57226


namespace Keith_initial_picked_l572_57258

-- Definitions based on the given conditions
def Mike_picked := 12
def Keith_gave_away := 46
def remaining_pears := 13

-- Question: Prove that Keith initially picked 47 pears.
theorem Keith_initial_picked :
  ∃ K : ℕ, K = 47 ∧ (K - Keith_gave_away + Mike_picked = remaining_pears) :=
sorry

end Keith_initial_picked_l572_57258


namespace at_least_one_not_less_than_six_l572_57217

-- Definitions for the conditions.
variables {a b c : ℝ}
variables (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)

-- The proof statement.
theorem at_least_one_not_less_than_six :
  (a + 4 / b) < 6 ∧ (b + 9 / c) < 6 ∧ (c + 16 / a) < 6 → false :=
by
  sorry

end at_least_one_not_less_than_six_l572_57217


namespace fraction_sum_of_roots_l572_57237

theorem fraction_sum_of_roots (x1 x2 : ℝ) (h1 : 5 * x1^2 - 3 * x1 - 2 = 0) (h2 : 5 * x2^2 - 3 * x2 - 2 = 0) (hx : x1 ≠ x2) :
  (1 / x1 + 1 / x2 = -3 / 2) :=
by
  sorry

end fraction_sum_of_roots_l572_57237


namespace int_solution_count_l572_57262

def g (n : ℤ) : ℤ :=
  ⌈97 * n / 98⌉ - ⌊98 * n / 99⌋

theorem int_solution_count :
  (∃! n : ℤ, 1 + ⌊98 * n / 99⌋ = ⌈97 * n / 98⌉) :=
sorry

end int_solution_count_l572_57262


namespace cone_surface_area_ratio_l572_57200

theorem cone_surface_area_ratio (l : ℝ) (h_l_pos : 0 < l) :
  let θ := (120 * Real.pi) / 180 -- converting 120 degrees to radians
  let side_area := (1/2) * l^2 * θ
  let r := l / 3
  let base_area := Real.pi * r^2
  let surface_area := side_area + base_area
  side_area ≠ 0 → 
  surface_area / side_area = 4 / 3 := 
by
  -- Provide the proof here
  sorry

end cone_surface_area_ratio_l572_57200


namespace solve_inequality1_solve_inequality_system_l572_57244

-- Define the first condition inequality
def inequality1 (x : ℝ) : Prop := 
  (2 * x - 1) / 3 - (9 * x + 2) / 6 ≤ 1

-- Theorem for the first inequality proving x >= -2
theorem solve_inequality1 {x : ℝ} (h : inequality1 x) : x ≥ -2 := 
sorry

-- Define the first condition for the system of inequalities
def inequality2 (x : ℝ) : Prop := 
  x - 3 * (x - 2) ≥ 4

-- Define the second condition for the system of inequalities
def inequality3 (x : ℝ) : Prop := 
  (2 * x - 1) / 5 < (x + 1) / 2

-- Theorem for the system of inequalities proving -7 < x ≤ 1
theorem solve_inequality_system {x : ℝ} (h1 : inequality2 x) (h2 : inequality3 x) : -7 < x ∧ x ≤ 1 := 
sorry

end solve_inequality1_solve_inequality_system_l572_57244


namespace find_chocolate_boxes_l572_57285

section
variable (x : Nat)
variable (candy_per_box : Nat := 8)
variable (caramel_boxes : Nat := 3)
variable (total_candy : Nat := 80)

theorem find_chocolate_boxes :
  8 * x + candy_per_box * caramel_boxes = total_candy -> x = 7 :=
by
  sorry
end

end find_chocolate_boxes_l572_57285


namespace geometric_sequence_seventh_term_l572_57265

theorem geometric_sequence_seventh_term (a r: ℤ) (h1 : a = 3) (h2 : a * r ^ 5 = 729) : a * r ^ 6 = 2187 :=
by sorry

end geometric_sequence_seventh_term_l572_57265


namespace tan_triple_angle_l572_57238

theorem tan_triple_angle (θ : ℝ) (h : Real.tan θ = 3) : Real.tan (3 * θ) = 9 / 13 :=
by
  sorry

end tan_triple_angle_l572_57238


namespace intersection_of_function_and_inverse_l572_57264

theorem intersection_of_function_and_inverse (c k : ℤ) (f : ℤ → ℤ)
  (hf : ∀ x:ℤ, f x = 4 * x + c) 
  (hf_inv : ∀ y:ℤ, (∃ x:ℤ, f x = y) → (∃ x:ℤ, f y = x))
  (h_intersection : ∀ k:ℤ, f 2 = k ∧ f k = 2 ) 
  : k = 2 :=
sorry

end intersection_of_function_and_inverse_l572_57264


namespace inscribed_square_area_l572_57297

noncomputable def area_inscribed_square (AB CD : ℕ) (BCFE : ℕ) : Prop :=
  AB = 36 ∧ CD = 64 ∧ BCFE = (AB * CD)

theorem inscribed_square_area :
  ∀ (AB CD : ℕ),
  area_inscribed_square AB CD 2304 :=
by
  intros
  sorry

end inscribed_square_area_l572_57297


namespace shaded_fraction_is_half_l572_57231

-- Define the number of rows and columns in the grid
def num_rows : ℕ := 8
def num_columns : ℕ := 8

-- Define the number of shaded triangles based on the pattern explained
def shaded_rows : List ℕ := [1, 3, 5, 7]
def num_shaded_rows : ℕ := 4
def triangles_per_row : ℕ := num_columns
def num_shaded_triangles : ℕ := num_shaded_rows * triangles_per_row

-- Define the total number of triangles
def total_triangles : ℕ := num_rows * num_columns

-- Define the fraction of shaded triangles
def shaded_fraction : ℚ := num_shaded_triangles / total_triangles

-- Prove the shaded fraction is 1/2
theorem shaded_fraction_is_half : shaded_fraction = 1 / 2 :=
by
  -- Provide the calculations
  sorry

end shaded_fraction_is_half_l572_57231


namespace complex_root_cubic_l572_57252

theorem complex_root_cubic (a b q r : ℝ) (h_b_ne_zero : b ≠ 0)
  (h_root : (Polynomial.C a + Polynomial.C b * Polynomial.C I) * 
             (Polynomial.C a - Polynomial.C b * Polynomial.C I) * 
             (Polynomial.C (-2 * a)) 
             = Polynomial.X^3 + Polynomial.C q * Polynomial.X + Polynomial.C r) :
  q = b^2 - 3 * a^2 :=
sorry

end complex_root_cubic_l572_57252


namespace smallest_marbles_l572_57234

theorem smallest_marbles
  : ∃ n : ℕ, ((n % 8 = 5) ∧ (n % 7 = 2) ∧ (n = 37) ∧ (37 % 9 = 1)) :=
by
  sorry

end smallest_marbles_l572_57234


namespace even_decreasing_function_l572_57204

theorem even_decreasing_function (f : ℝ → ℝ) (x1 x2 : ℝ)
  (hf_even : ∀ x, f x = f (-x))
  (hf_decreasing : ∀ x y, x < y → x < 0 → y < 0 → f y < f x)
  (hx1_neg : x1 < 0)
  (hx1x2_pos : x1 + x2 > 0) :
  f x1 < f x2 :=
sorry

end even_decreasing_function_l572_57204


namespace male_students_plant_trees_l572_57208

theorem male_students_plant_trees (total_avg : ℕ) (female_trees : ℕ) (male_trees : ℕ) 
  (h1 : total_avg = 6) 
  (h2 : female_trees = 15)
  (h3 : 1 / (male_trees : ℝ) + 1 / (female_trees : ℝ) = 1 / (total_avg : ℝ)) : 
  male_trees = 10 := 
sorry

end male_students_plant_trees_l572_57208


namespace width_of_rectangular_plot_l572_57279

theorem width_of_rectangular_plot 
  (length : ℝ) 
  (poles : ℕ) 
  (distance_between_poles : ℝ) 
  (num_poles : ℕ) 
  (total_wire_length : ℝ) 
  (perimeter : ℝ) 
  (width : ℝ) :
  length = 90 ∧ 
  distance_between_poles = 5 ∧ 
  num_poles = 56 ∧ 
  total_wire_length = (num_poles - 1) * distance_between_poles ∧ 
  total_wire_length = 275 ∧ 
  perimeter = 2 * (length + width) 
  → width = 47.5 :=
by
  sorry

end width_of_rectangular_plot_l572_57279


namespace jiahao_estimate_larger_l572_57278

variable (x y : ℝ)
variable (hxy : x > y)
variable (hy0 : y > 0)

theorem jiahao_estimate_larger (x y : ℝ) (hxy : x > y) (hy0 : y > 0) :
  (x + 2) - (y - 1) > x - y :=
by
  sorry

end jiahao_estimate_larger_l572_57278


namespace B_wins_four_rounds_prob_is_0_09_C_wins_three_rounds_prob_is_0_162_l572_57249

namespace GoGame

-- Define the players: A, B, C
inductive Player
| A
| B
| C

open Player

-- Define the probabilities as given
def P_A_beats_B : ℝ := 0.4
def P_B_beats_C : ℝ := 0.5
def P_C_beats_A : ℝ := 0.6

-- Define the game rounds and logic
def probability_B_winning_four_rounds 
  (P_A_beats_B : ℝ) (P_B_beats_C : ℝ) (P_C_beats_A : ℝ) : ℝ :=
(1 - P_A_beats_B)^2 * P_B_beats_C^2

def probability_C_winning_three_rounds 
  (P_A_beats_B : ℝ) (P_B_beats_C : ℝ) (P_C_beats_A : ℝ) : ℝ :=
  P_A_beats_B * P_C_beats_A^2 * P_B_beats_C + 
  (1 - P_A_beats_B) * P_B_beats_C^2 * P_C_beats_A

-- Proof statements
theorem B_wins_four_rounds_prob_is_0_09 : 
  probability_B_winning_four_rounds P_A_beats_B P_B_beats_C P_C_beats_A = 0.09 :=
by
  sorry

theorem C_wins_three_rounds_prob_is_0_162 : 
  probability_C_winning_three_rounds P_A_beats_B P_B_beats_C P_C_beats_A = 0.162 :=
by
  sorry

end GoGame

end B_wins_four_rounds_prob_is_0_09_C_wins_three_rounds_prob_is_0_162_l572_57249


namespace positive_difference_of_perimeters_l572_57212

theorem positive_difference_of_perimeters :
  let length1 := 6
  let width1 := 1
  let length2 := 3
  let width2 := 2
  let perimeter1 := 2 * (length1 + width1)
  let perimeter2 := 2 * (length2 + width2)
  (perimeter1 - perimeter2) = 4 :=
by
  let length1 := 6
  let width1 := 1
  let length2 := 3
  let width2 := 2
  let perimeter1 := 2 * (length1 + width1)
  let perimeter2 := 2 * (length2 + width2)
  show (perimeter1 - perimeter2) = 4
  sorry

end positive_difference_of_perimeters_l572_57212


namespace compute_expression_l572_57290

theorem compute_expression : 7^3 - 5 * (6^2) + 2^4 = 179 :=
by
  sorry

end compute_expression_l572_57290


namespace find_y_six_l572_57201

theorem find_y_six (y : ℝ) (h : y > 0) (h_eq : (2 - y^3)^(1/3) + (2 + y^3)^(1/3) = 2) : 
    y^6 = 116 / 27 :=
by
  sorry

end find_y_six_l572_57201


namespace find_crayons_in_pack_l572_57289

variables (crayons_in_locker : ℕ) (crayons_given_by_bobby : ℕ) (crayons_given_to_mary : ℕ) (crayons_final_count : ℕ) (crayons_in_pack : ℕ)

-- Definitions from the conditions
def initial_crayons := 36
def bobby_gave := initial_crayons / 2
def mary_crayons := 25
def final_crayons := initial_crayons + bobby_gave - mary_crayons

-- The theorem to prove
theorem find_crayons_in_pack : initial_crayons = 36 ∧ bobby_gave = 18 ∧ mary_crayons = 25 ∧ final_crayons = 29 → crayons_in_pack = 29 :=
by
  sorry

end find_crayons_in_pack_l572_57289


namespace beginner_trigonometry_probability_l572_57253

def BC := ℝ
def AC := ℝ
def IC := ℝ
def BT := ℝ
def AT := ℝ
def IT := ℝ
def T := 5000

theorem beginner_trigonometry_probability :
  ∀ (BC AC IC BT AT IT : ℝ),
  (BC + AC + IC = 0.60 * T) →
  (BT + AT + IT = 0.40 * T) →
  (BC + BT = 0.45 * T) →
  (AC + AT = 0.35 * T) →
  (IC + IT = 0.20 * T) →
  (BC = 1.25 * BT) →
  (IC + AC = 1.20 * (IT + AT)) →
  (BT / T = 1/5) :=
by
  intros
  sorry

end beginner_trigonometry_probability_l572_57253


namespace second_discount_percentage_l572_57272

-- Defining the variables
variables (P S : ℝ) (d1 d2 : ℝ)

-- Given conditions
def original_price : P = 200 := by sorry
def sale_price_after_initial_discount : S = 171 := by sorry
def first_discount_rate : d1 = 0.10 := by sorry

-- Required to prove
theorem second_discount_percentage :
  ∃ d2, (d2 = 0.05) :=
sorry

end second_discount_percentage_l572_57272


namespace polygon_interior_angle_increase_l572_57214

theorem polygon_interior_angle_increase (n : ℕ) (h : 3 ≤ n) :
  ((n + 1 - 2) * 180 - (n - 2) * 180 = 180) :=
by sorry

end polygon_interior_angle_increase_l572_57214


namespace simplified_expression_correct_l572_57269

def simplify_expression (x : ℝ) : ℝ :=
  4 * (x ^ 2 - 5 * x) - 5 * (2 * x ^ 2 + 3 * x)

theorem simplified_expression_correct (x : ℝ) :
  simplify_expression x = -6 * x ^ 2 - 35 * x :=
by
  sorry

end simplified_expression_correct_l572_57269


namespace digit_sum_26_l572_57207

theorem digit_sum_26 
  (A B C D E : ℕ)
  (h1 : 1 ≤ A ∧ A ≤ 9)
  (h2 : 0 ≤ B ∧ B ≤ 9)
  (h3 : 0 ≤ C ∧ C ≤ 9)
  (h4 : 0 ≤ D ∧ D ≤ 9)
  (h5 : 0 ≤ E ∧ E ≤ 9)
  (h6 : 100000 + 10000 * A + 1000 * B + 100 * C + 10 * D + E * 3 = 100000 * A + 10000 * B + 1000 * C + 100 * D + 10 * E + 1):
  A + B + C + D + E = 26 
  := 
  by
    sorry

end digit_sum_26_l572_57207


namespace original_weight_of_marble_l572_57242

variable (W: ℝ) 

theorem original_weight_of_marble (h: 0.80 * 0.82 * 0.72 * W = 85.0176): W = 144 := 
by
  sorry

end original_weight_of_marble_l572_57242


namespace total_yield_l572_57299

noncomputable def johnson_hectare_yield_2months : ℕ := 80
noncomputable def neighbor_hectare_yield_multiplier : ℕ := 2
noncomputable def neighbor_hectares : ℕ := 2
noncomputable def months : ℕ := 6

theorem total_yield (jh2 : ℕ := johnson_hectare_yield_2months) 
                    (nhm : ℕ := neighbor_hectare_yield_multiplier) 
                    (nh : ℕ := neighbor_hectares) 
                    (m : ℕ := months): 
                    3 * jh2 + 3 * nh * jh2 * nhm = 1200 :=
by
  sorry

end total_yield_l572_57299


namespace boat_cannot_complete_round_trip_l572_57282

theorem boat_cannot_complete_round_trip
  (speed_still_water : ℝ)
  (speed_current : ℝ)
  (distance : ℝ)
  (total_time : ℝ)
  (speed_still_water_pos : speed_still_water > 0)
  (speed_current_nonneg : speed_current ≥ 0)
  (distance_pos : distance > 0)
  (total_time_pos : total_time > 0) :
  let speed_downstream := speed_still_water + speed_current
  let speed_upstream := speed_still_water - speed_current
  let time_downstream := distance / speed_downstream
  let time_upstream := distance / speed_upstream
  let total_trip_time := time_downstream + time_upstream
  total_trip_time > total_time :=
by {
  -- Proof goes here
  sorry
}

end boat_cannot_complete_round_trip_l572_57282


namespace f_positive_for_all_x_f_increasing_solution_set_inequality_l572_57259

namespace ProofProblem

-- Define the function f and its properties
def f : ℝ → ℝ := sorry

axiom f_zero_ne_zero : f 0 ≠ 0
axiom f_one_eq_two : f 1 = 2
axiom f_pos_when_pos : ∀ x : ℝ, x > 0 → f x > 1
axiom f_add_mul : ∀ a b : ℝ, f (a + b) = f a * f b

-- Problem 1: Prove that f(x) > 0 for all x ∈ ℝ
theorem f_positive_for_all_x : ∀ x : ℝ, f x > 0 := sorry

-- Problem 2: Prove that f(x) is increasing on ℝ
theorem f_increasing : ∀ x y : ℝ, x < y → f x < f y := sorry

-- Problem 3: Find the solution set of the inequality f(3-2x) > 4
theorem solution_set_inequality : { x : ℝ | f (3 - 2 * x) > 4 } = { x | x < 1 / 2 } := sorry

end ProofProblem

end f_positive_for_all_x_f_increasing_solution_set_inequality_l572_57259


namespace total_dress_designs_l572_57243

theorem total_dress_designs:
  let colors := 5
  let patterns := 4
  let sleeve_lengths := 2
  colors * patterns * sleeve_lengths = 40 := 
by
  sorry

end total_dress_designs_l572_57243


namespace intersection_A_B_l572_57228

-- Define the set A
def A : Set ℕ := {1, 2, 3, 4, 5}

-- Define the set B as the set of natural numbers greater than 2.5
def B : Set ℕ := {x : ℕ | 2 * x > 5}

-- Prove that the intersection of A and B is {3, 4, 5}
theorem intersection_A_B : A ∩ B = {3, 4, 5} :=
by sorry

end intersection_A_B_l572_57228


namespace functional_equation_solution_l572_57254

theorem functional_equation_solution :
  ∃ f : ℝ → ℝ,
  (f 1 = 1 ∧ (∀ x y : ℝ, f (x * y + f x) = x * f y + f x)) ∧ f (1/2) = 1/2 :=
by
  sorry

end functional_equation_solution_l572_57254


namespace certain_number_l572_57202

theorem certain_number (n : ℕ) : 
  (55 * 57) % n = 6 ∧ n = 1043 :=
by
  sorry

end certain_number_l572_57202


namespace count_perfect_square_factors_of_360_l572_57277

def is_prime_fact_even (n : ℕ) : Prop :=
  n % 2 = 0

def is_perfect_square (d : ℕ) : Prop :=
  ∃ a b c : ℕ, d = 2^(2*a) * 3^(2*b) * 5^(2*c)

def prime_factorization_360 : Prop :=
  ∀ d : ℕ, d ∣ 360 → is_perfect_square d

theorem count_perfect_square_factors_of_360 : ∃ count : ℕ, count = 4 :=
  sorry

end count_perfect_square_factors_of_360_l572_57277


namespace brandon_initial_skittles_l572_57271

theorem brandon_initial_skittles (initial_skittles : ℕ) (loss : ℕ) (final_skittles : ℕ) (h1 : final_skittles = 87) (h2 : loss = 9) (h3 : final_skittles = initial_skittles - loss) : initial_skittles = 96 :=
sorry

end brandon_initial_skittles_l572_57271


namespace divide_subtract_multiply_l572_57270

theorem divide_subtract_multiply :
  (-5) / ((1/4) - (1/3)) * 12 = 720 := by
  sorry

end divide_subtract_multiply_l572_57270


namespace volume_formula_l572_57209

noncomputable def volume_of_parallelepiped
  (a b : ℝ) (h : ℝ) (θ : ℝ) 
  (θ_eq : θ = Real.pi / 3) 
  (base_diagonal : ℝ)
  (base_diagonal_eq : base_diagonal = Real.sqrt (a ^ 2 + b ^ 2)) : ℝ :=
  a * b * h 

theorem volume_formula 
  (a b : ℝ) (h : ℝ) (θ : ℝ)
  (area_base : ℝ) 
  (area_of_base_eq : area_base = a * b) 
  (θ_eq : θ = Real.pi / 3) 
  (base_diagonal : ℝ) 
  (base_diagonal_eq : base_diagonal = Real.sqrt (a ^ 2 + b ^ 2))
  (height_eq : h = (base_diagonal / 2) * (Real.sqrt 3)): 
  volume_of_parallelepiped a b h θ θ_eq base_diagonal base_diagonal_eq 
  = (144 * Real.sqrt 3) / 5 :=
by {
  sorry
}

end volume_formula_l572_57209


namespace walnut_trees_planted_l572_57267

-- Define the initial number of walnut trees
def initial_walnut_trees : ℕ := 22

-- Define the total number of walnut trees after planting
def total_walnut_trees_after : ℕ := 55

-- The Lean statement to prove the number of walnut trees planted today
theorem walnut_trees_planted : (total_walnut_trees_after - initial_walnut_trees = 33) :=
by
  sorry

end walnut_trees_planted_l572_57267


namespace sum_of_underlined_numbers_non_negative_l572_57203

-- Definitions used in the problem
def is_positive (n : Int) : Prop := n > 0
def underlined (nums : List Int) : List Int := sorry -- Define underlining based on conditions

def sum_of_underlined_numbers (nums : List Int) : Int :=
  (underlined nums).sum

-- The proof problem statement
theorem sum_of_underlined_numbers_non_negative
  (nums : List Int)
  (h_len : nums.length = 100) :
  0 < sum_of_underlined_numbers nums := sorry

end sum_of_underlined_numbers_non_negative_l572_57203


namespace find_original_shirt_price_l572_57296

noncomputable def original_shirt_price (S pants_orig_price jacket_orig_price total_paid : ℝ) :=
  let discounted_shirt := S * 0.5625
  let discounted_pants := pants_orig_price * 0.70
  let discounted_jacket := jacket_orig_price * 0.64
  let total_before_loyalty := discounted_shirt + discounted_pants + discounted_jacket
  let total_after_loyalty := total_before_loyalty * 0.90
  let total_after_tax := total_after_loyalty * 1.15
  total_after_tax = total_paid

theorem find_original_shirt_price : 
  original_shirt_price S 50 75 150 → S = 110.07 :=
by
  intro h
  sorry

end find_original_shirt_price_l572_57296


namespace roads_with_five_possible_roads_with_four_not_possible_l572_57281

-- Problem (a)
theorem roads_with_five_possible :
  ∃ (cities : Fin 16 → Finset (Fin 16)),
  (∀ c, cities c = {d | d ≠ c ∧ d ∈ cities c}) ∧
  (∀ c, (cities c).card ≤ 5) ∧
  (∀ c d, d ≠ c → ∃ e, e ≠ c ∧ e ≠ d ∧ d ∈ cities c ∪ {e}) := by
  sorry

-- Problem (b)
theorem roads_with_four_not_possible :
  ¬ ∃ (cities : Fin 16 → Finset (Fin 16)),
  (∀ c, cities c = {d | d ≠ c ∧ d ∈ cities c}) ∧
  (∀ c, (cities c).card ≤ 4) ∧
  (∀ c d, d ≠ c → ∃ e, e ≠ c ∧ e ≠ d ∧ d ∈ cities c ∪ {e}) := by
  sorry

end roads_with_five_possible_roads_with_four_not_possible_l572_57281


namespace compute_expr_l572_57286

theorem compute_expr : 5^2 - 3 * 4 + 3^2 = 22 := by
  sorry

end compute_expr_l572_57286


namespace lacrosse_more_than_football_l572_57225

-- Define the constants and conditions
def total_bottles := 254
def football_players := 11
def bottles_per_football_player := 6
def soccer_bottles := 53
def rugby_bottles := 49

-- Calculate the number of bottles needed by each team
def football_bottles := football_players * bottles_per_football_player
def other_teams_bottles := football_bottles + soccer_bottles + rugby_bottles
def lacrosse_bottles := total_bottles - other_teams_bottles

-- The theorem to be proven
theorem lacrosse_more_than_football : lacrosse_bottles - football_bottles = 20 :=
by
  sorry

end lacrosse_more_than_football_l572_57225


namespace solve_abs_system_eq_l572_57294

theorem solve_abs_system_eq (x y : ℝ) :
  (|x + y| + |1 - x| = 6) ∧ (|x + y + 1| + |1 - y| = 4) ↔ x = -2 ∧ y = -1 :=
by sorry

end solve_abs_system_eq_l572_57294


namespace sales_tax_is_5_percent_l572_57245

theorem sales_tax_is_5_percent :
  let cost_tshirt := 8
  let cost_sweater := 18
  let cost_jacket := 80
  let discount := 0.10
  let num_tshirts := 6
  let num_sweaters := 4
  let num_jackets := 5
  let total_cost_with_tax := 504
  let total_cost_before_discount := (num_jackets * cost_jacket)
  let discount_amount := discount * total_cost_before_discount
  let discounted_cost_jackets := total_cost_before_discount - discount_amount
  let total_cost_before_tax := (num_tshirts * cost_tshirt) + (num_sweaters * cost_sweater) + discounted_cost_jackets
  let sales_tax := (total_cost_with_tax - total_cost_before_tax)
  let sales_tax_percentage := (sales_tax / total_cost_before_tax) * 100
  sales_tax_percentage = 5 := by
  sorry

end sales_tax_is_5_percent_l572_57245


namespace carla_sharpening_time_l572_57206

theorem carla_sharpening_time (x : ℕ) (h : x + 3 * x = 40) : x = 10 :=
by
  sorry

end carla_sharpening_time_l572_57206


namespace find_a8_a12_sum_l572_57240

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem find_a8_a12_sum
  (a : ℕ → ℝ) 
  (h_geom : geometric_sequence a) 
  (h1 : a 2 + a 6 = 3) 
  (h2 : a 6 + a 10 = 12) : 
  a 8 + a 12 = 24 :=
sorry

end find_a8_a12_sum_l572_57240


namespace percentage_increase_in_pay_rate_l572_57221

-- Given conditions
def regular_rate : ℝ := 10
def total_surveys : ℕ := 50
def cellphone_surveys : ℕ := 35
def total_earnings : ℝ := 605

-- We need to demonstrate that the percentage increase in the pay rate for surveys involving the use of her cellphone is 30%
theorem percentage_increase_in_pay_rate :
  let earnings_at_regular_rate := regular_rate * total_surveys
  let earnings_from_cellphone_surveys := total_earnings - earnings_at_regular_rate
  let rate_per_cellphone_survey := earnings_from_cellphone_surveys / cellphone_surveys
  let increase_in_rate := rate_per_cellphone_survey - regular_rate
  let percentage_increase := (increase_in_rate / regular_rate) * 100
  percentage_increase = 30 :=
by
  sorry

end percentage_increase_in_pay_rate_l572_57221


namespace batsman_average_17th_innings_l572_57266

theorem batsman_average_17th_innings:
  ∀ (A : ℝ), 
  (16 * A + 85 = 17 * (A + 3)) →
  (A + 3 = 37) :=
by
  intros A h
  sorry

end batsman_average_17th_innings_l572_57266


namespace log_sum_range_l572_57233

theorem log_sum_range (x y : ℝ) (hx_pos : x > 0) (hy_pos : y > 0) (hx_ne_one : x ≠ 1) (hy_ne_one : y ≠ 1) :
  (Real.log y / Real.log x + Real.log x / Real.log y) ∈ Set.union (Set.Iic (-2)) (Set.Ici 2) :=
sorry

end log_sum_range_l572_57233


namespace cacti_average_height_l572_57251

variables {Cactus1 Cactus2 Cactus3 Cactus4 Cactus5 Cactus6 : ℕ}
variables (condition1 : Cactus1 = 14)
variables (condition3 : Cactus3 = 7)
variables (condition6 : Cactus6 = 28)
variables (condition2 : Cactus2 = 14)
variables (condition4 : Cactus4 = 14)
variables (condition5 : Cactus5 = 14)

theorem cacti_average_height : 
  (Cactus1 + Cactus2 + Cactus3 + Cactus4 + Cactus5 + Cactus6 : ℕ) = 91 → 
  (91 : ℝ) / 6 = (15.2 : ℝ) :=
by
  sorry

end cacti_average_height_l572_57251


namespace quadratic_roots_eq_k_quadratic_inequality_k_range_l572_57229

theorem quadratic_roots_eq_k (k : ℝ) (h1 : k > 0)
  (h2 : ∀ x : ℝ, 2 < x → x < 3 → k*x^2 - 2*x + 6*k < 0)
  (h3: (2 + 3) = (2/k)) : k = 2/5 :=
by sorry

theorem quadratic_inequality_k_range (k : ℝ) 
  (h1 : k > 0)
  (h2 : ∀ x : ℝ, 2 < x → x < 3 → k*x^2 - 2*x + 6*k < 0) 
: 0 < k ∧ k <= 2/5 :=
by sorry

end quadratic_roots_eq_k_quadratic_inequality_k_range_l572_57229


namespace factor_x4_plus_81_l572_57283

theorem factor_x4_plus_81 (x : ℝ) : x^4 + 81 = (x^2 + 6 * x + 9) * (x^2 - 6 * x + 9) :=
by 
  -- The proof is omitted.
  sorry

end factor_x4_plus_81_l572_57283


namespace work_speed_ratio_l572_57280

open Real

theorem work_speed_ratio (A B : Type) 
  (A_work_speed B_work_speed : ℝ) 
  (combined_work_time : ℝ) 
  (B_work_time : ℝ)
  (h_combined : combined_work_time = 12)
  (h_B : B_work_time = 36)
  (combined_speed : A_work_speed + B_work_speed = 1 / combined_work_time)
  (B_speed : B_work_speed = 1 / B_work_time) :
  A_work_speed / B_work_speed = 2 :=
by sorry

end work_speed_ratio_l572_57280


namespace m_range_circle_l572_57213

noncomputable def circle_equation (m : ℝ) : Prop :=
  ∀ (x y : ℝ), x^2 + y^2 + 2 * x + 4 * y + m = 0

theorem m_range_circle (m : ℝ) : circle_equation m → m < 5 := by
  sorry

end m_range_circle_l572_57213


namespace train_length_l572_57230

/-- Given a train traveling at 72 km/hr passing a pole in 8 seconds,
     prove that the length of the train in meters is 160. -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) (speed_m_s : ℝ) (distance_m : ℝ) :
  speed_kmh = 72 → 
  time_s = 8 → 
  speed_m_s = (speed_kmh * 1000) / 3600 → 
  distance_m = speed_m_s * time_s → 
  distance_m = 160 :=
by
  sorry

end train_length_l572_57230


namespace sin_of_5pi_over_6_l572_57276

theorem sin_of_5pi_over_6 : Real.sin (5 * Real.pi / 6) = 1 / 2 := 
by
  sorry

end sin_of_5pi_over_6_l572_57276


namespace polygon_sides_l572_57246

theorem polygon_sides (x : ℝ) (hx : 0 < x) (h : x + 5 * x = 180) : 12 = 360 / x :=
by {
  -- Steps explaining: x should be the exterior angle then proof follows.
  sorry
}

end polygon_sides_l572_57246


namespace total_cookies_l572_57232

def mona_cookies : ℕ := 20
def jasmine_cookies : ℕ := mona_cookies - 5
def rachel_cookies : ℕ := jasmine_cookies + 10

theorem total_cookies : mona_cookies + jasmine_cookies + rachel_cookies = 60 := 
by
  have h1 : jasmine_cookies = 15 := by sorry
  have h2 : rachel_cookies = 25 := by sorry
  have h3 : mona_cookies = 20 := by sorry
  sorry

end total_cookies_l572_57232


namespace probability_heart_or_king_l572_57293

theorem probability_heart_or_king (cards hearts kings : ℕ) (prob_non_heart_king : ℚ) 
    (prob_two_non_heart_king : ℚ) : 
    cards = 52 → hearts = 13 → kings = 4 → 
    prob_non_heart_king = 36 / 52 → prob_two_non_heart_king = (36 / 52) ^ 2 → 
    1 - prob_two_non_heart_king = 88 / 169 :=
by
  intros h_cards h_hearts h_kings h_prob_non_heart_king h_prob_two_non_heart_king
  sorry

end probability_heart_or_king_l572_57293


namespace number_of_ways_to_take_one_ball_from_pockets_number_of_ways_to_take_one_ball_each_from_pockets_l572_57275

-- Let's define the conditions.
def balls_in_first_pocket : Nat := 2
def balls_in_second_pocket : Nat := 4
def balls_in_third_pocket : Nat := 5

-- Proof for the first question
theorem number_of_ways_to_take_one_ball_from_pockets : 
  balls_in_first_pocket + balls_in_second_pocket + balls_in_third_pocket = 11 := 
by
  sorry

-- Proof for the second question
theorem number_of_ways_to_take_one_ball_each_from_pockets : 
  balls_in_first_pocket * balls_in_second_pocket * balls_in_third_pocket = 40 := 
by
  sorry

end number_of_ways_to_take_one_ball_from_pockets_number_of_ways_to_take_one_ball_each_from_pockets_l572_57275


namespace anna_has_2_fewer_toys_than_amanda_l572_57250

-- Define the variables for the number of toys each person has
variables (A B : ℕ)

-- Define the conditions
def conditions (M : ℕ) : Prop :=
  M = 20 ∧ A = 3 * M ∧ A + M + B = 142

-- The theorem to prove
theorem anna_has_2_fewer_toys_than_amanda (M : ℕ) (h : conditions A B M) : B - A = 2 :=
sorry

end anna_has_2_fewer_toys_than_amanda_l572_57250


namespace compare_fractions_difference_l572_57287

theorem compare_fractions_difference :
  let a := (1 : ℝ) / 2
  let b := (1 : ℝ) / 3
  a - b = 1 / 6 :=
by
  sorry

end compare_fractions_difference_l572_57287


namespace parabola_vertex_and_point_l572_57288

theorem parabola_vertex_and_point (a b c : ℝ) : 
  (∀ x, y = a * x^2 + b * x + c) ∧ 
  ∃ x y, (y = a * (x - 4)^2 + 3) → 
  (a * 2^2 + b * 2 + c = 5) → 
  (a = 1/2 ∧ b = -4 ∧ c = 11) :=
by
  sorry

end parabola_vertex_and_point_l572_57288


namespace geo_seq_sum_l572_57239

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geo_seq_sum (a : ℕ → ℝ) (h : geometric_sequence a) (h1 : a 0 + a 1 = 30) (h4 : a 3 + a 4 = 120) :
  a 6 + a 7 = 480 :=
sorry

end geo_seq_sum_l572_57239


namespace employee_discount_percentage_l572_57248

theorem employee_discount_percentage (wholesale_cost retail_price employee_price discount_percentage : ℝ) 
  (h1 : wholesale_cost = 200)
  (h2 : retail_price = wholesale_cost * 1.2)
  (h3 : employee_price = 204)
  (h4 : discount_percentage = ((retail_price - employee_price) / retail_price) * 100) :
  discount_percentage = 15 :=
by
  sorry

end employee_discount_percentage_l572_57248


namespace Amanda_notebooks_l572_57274

theorem Amanda_notebooks (initial ordered lost final : ℕ) 
  (h_initial: initial = 65) 
  (h_ordered: ordered = 23) 
  (h_lost: lost = 14) : 
  final = 74 := 
by 
  -- calculation and proof will go here
  sorry 

end Amanda_notebooks_l572_57274


namespace cyclist_speed_l572_57255

theorem cyclist_speed
  (V : ℝ)
  (H1 : ∃ t_p : ℝ, V * t_p = 96 ∧ t_p = (96 / (V - 1)) - 2)
  (H2 : V > 1.25 * (V - 1)) :
  V = 16 :=
by
  sorry

end cyclist_speed_l572_57255


namespace max_abs_cubic_at_least_one_fourth_l572_57223

def cubic_polynomial (p q r x : ℝ) : ℝ := x^3 + p * x^2 + q * x + r

theorem max_abs_cubic_at_least_one_fourth (p q r : ℝ) :
  ∃ x ∈ Set.Icc (-1 : ℝ) 1, |cubic_polynomial p q r x| ≥ 1 / 4 :=
by
  sorry

end max_abs_cubic_at_least_one_fourth_l572_57223


namespace flower_bee_relationship_l572_57220

def numberOfBees (flowers : ℕ) (fewer_bees : ℕ) : ℕ :=
  flowers - fewer_bees

theorem flower_bee_relationship :
  numberOfBees 5 2 = 3 := by
  sorry

end flower_bee_relationship_l572_57220


namespace percentage_mr_william_land_l572_57298

theorem percentage_mr_william_land 
  (T W : ℝ) -- Total taxable land of the village and the total land of Mr. William
  (tax_collected_village : ℝ) -- Total tax collected from the village
  (tax_paid_william : ℝ) -- Tax paid by Mr. William
  (h1 : tax_collected_village = 3840) 
  (h2 : tax_paid_william = 480) 
  (h3 : (480 / 3840) = (25 / 100) * (W / T)) 
: (W / T) * 100 = 12.5 :=
by sorry

end percentage_mr_william_land_l572_57298


namespace solution_of_equation_l572_57210

theorem solution_of_equation (x : ℤ) : 7 * x - 5 = 6 * x → x = 5 := by
  intro h
  sorry

end solution_of_equation_l572_57210


namespace k_2_sufficient_but_not_necessary_l572_57257

def vector_a : ℝ × ℝ := (2, 1)
def vector_b (k : ℝ) : ℝ × ℝ := (1, k^2 - 1) - (2, 1)

def perpendicular (x y : ℝ × ℝ) : Prop := x.1 * y.1 + x.2 * y.2 = 0

theorem k_2_sufficient_but_not_necessary (k : ℝ) :
  k = 2 → perpendicular vector_a (vector_b k) ∧ ∃ k, not (k = 2) ∧ perpendicular vector_a (vector_b k) :=
by
  sorry

end k_2_sufficient_but_not_necessary_l572_57257


namespace total_students_in_class_l572_57227

-- Definitions based on the conditions
def volleyball_participants : Nat := 22
def basketball_participants : Nat := 26
def both_participants : Nat := 4

-- The theorem statement
theorem total_students_in_class : volleyball_participants + basketball_participants - both_participants = 44 :=
by
  -- Sorry to skip the proof
  sorry

end total_students_in_class_l572_57227


namespace cost_price_l572_57216

-- Given conditions
variable (x : ℝ)
def profit (x : ℝ) : ℝ := 54 - x
def loss (x : ℝ) : ℝ := x - 40

-- Claim
theorem cost_price (h : profit x = loss x) : x = 47 :=
by {
  -- This is where the proof would go
  sorry
}

end cost_price_l572_57216


namespace AC_plus_third_BA_l572_57222

def point := (ℝ × ℝ)

def A : point := (2, 4)
def B : point := (-1, -5)
def C : point := (3, -2)

noncomputable def vec (p₁ p₂ : point) : point :=
  (p₂.1 - p₁.1, p₂.2 - p₁.2)

noncomputable def scal_mult (scalar : ℝ) (v : point) : point :=
  (scalar * v.1, scalar * v.2)

noncomputable def vec_add (v₁ v₂ : point) : point :=
  (v₁.1 + v₂.1, v₁.2 + v₂.2)

theorem AC_plus_third_BA : 
  vec_add (vec A C) (scal_mult (1 / 3) (vec B A)) = (2, -3) :=
by
  sorry

end AC_plus_third_BA_l572_57222


namespace probability_ball_sports_l572_57273

theorem probability_ball_sports (clubs : Finset String)
  (ball_clubs : Finset String)
  (count_clubs : clubs.card = 5)
  (count_ball_clubs : ball_clubs.card = 3)
  (h1 : "basketball" ∈ clubs)
  (h2 : "soccer" ∈ clubs)
  (h3 : "volleyball" ∈ clubs)
  (h4 : "swimming" ∈ clubs)
  (h5 : "gymnastics" ∈ clubs)
  (h6 : "basketball" ∈ ball_clubs)
  (h7 : "soccer" ∈ ball_clubs)
  (h8 : "volleyball" ∈ ball_clubs) :
  (2 / ((5 : ℝ) * (4 : ℝ)) * ((3 : ℝ) * (2 : ℝ)) = (3 / 10)) :=
by
  sorry

end probability_ball_sports_l572_57273


namespace find_stadium_width_l572_57211

-- Conditions
def stadium_length : ℝ := 24
def stadium_height : ℝ := 16
def longest_pole : ℝ := 34

-- Width to be solved
def stadium_width : ℝ := 18

-- Theorem stating that given the conditions, the width must be 18
theorem find_stadium_width :
  stadium_length^2 + stadium_width^2 + stadium_height^2 = longest_pole^2 :=
by
  sorry

end find_stadium_width_l572_57211


namespace total_people_on_boats_l572_57295

-- Definitions based on the given conditions
def boats : Nat := 5
def people_per_boat : Nat := 3

-- Theorem statement to prove the total number of people on boats in the lake
theorem total_people_on_boats : boats * people_per_boat = 15 :=
by
  sorry

end total_people_on_boats_l572_57295


namespace solution_set_all_real_solution_set_empty_exists_at_least_one_solution_l572_57235

-- Definitions for the inequality ax^2 - 2ax + 2a - 3 < 0
def quadratic_expr (a x : ℝ) : ℝ := a * x^2 - 2 * a * x + 2 * a - 3

-- Requirement (1): The solution set is ℝ
theorem solution_set_all_real (a : ℝ) (h : a ≤ 0) : 
  ∀ x : ℝ, quadratic_expr a x < 0 :=
by sorry

-- Requirement (2): The solution set is ∅
theorem solution_set_empty (a : ℝ) (h : a ≥ 3) : 
  ¬∃ x : ℝ, quadratic_expr a x < 0 :=
by sorry

-- Requirement (3): There is at least one real solution
theorem exists_at_least_one_solution (a : ℝ) (h : a < 3) : 
  ∃ x : ℝ, quadratic_expr a x < 0 :=
by sorry

end solution_set_all_real_solution_set_empty_exists_at_least_one_solution_l572_57235


namespace quadrilateral_probability_l572_57284

def total_shapes : ℕ := 6
def quadrilateral_shapes : ℕ := 3

theorem quadrilateral_probability : (quadrilateral_shapes : ℚ) / (total_shapes : ℚ) = 1 / 2 :=
by
  sorry

end quadrilateral_probability_l572_57284
