import Mathlib

namespace max_b_no_lattice_point_l281_281622

theorem max_b_no_lattice_point (m : ℚ) (x : ℤ) (b : ℚ) :
  (y = mx + 3) → (0 < x ∧ x ≤ 50) → (2/5 < m ∧ m < b) → 
  ∀ (x : ℕ), y ≠ m * x + 3 →
  b = 11/51 :=
sorry

end max_b_no_lattice_point_l281_281622


namespace find_set_B_l281_281988

open Set

variable (U : Finset ℕ) (A B : Finset ℕ)
variable (hU : U = {1, 2, 3, 4, 5, 6, 7})
variable (h1 : (U \ (A ∪ B)) = {1, 3})
variable (h2 : A ∩ (U \ B) = {2, 5})

theorem find_set_B : B = {4, 6, 7} := by
  sorry

end find_set_B_l281_281988


namespace xy_inequality_l281_281078

theorem xy_inequality (x y θ : ℝ) 
    (h : (x * Real.cos θ + y * Real.sin θ)^2 + x * Real.sin θ - y * Real.cos θ = 1) : 
    x^2 + y^2 ≥ 3/4 :=
sorry

end xy_inequality_l281_281078


namespace arithmetic_sequence_general_formula_l281_281508

noncomputable def a_n (n : ℕ) : ℝ :=
sorry

theorem arithmetic_sequence_general_formula (h1 : (a_n 2 + a_n 6) / 2 = 5)
                                            (h2 : (a_n 3 + a_n 7) / 2 = 7) :
  a_n n = 2 * (n : ℝ) - 3 :=
sorry

end arithmetic_sequence_general_formula_l281_281508


namespace simplify_expression_l281_281245

theorem simplify_expression (x : ℕ) : (5 * x^4)^3 = 125 * x^(12) := by
  sorry

end simplify_expression_l281_281245


namespace eval_expr_l281_281009

theorem eval_expr : 4 * (-3) + 60 / (-15) = -16 := by
  sorry

end eval_expr_l281_281009


namespace molecular_weight_calculation_l281_281441

theorem molecular_weight_calculation
    (moles_total_mw : ℕ → ℝ)
    (hw : moles_total_mw 9 = 900) :
    moles_total_mw 1 = 100 :=
by
  sorry

end molecular_weight_calculation_l281_281441


namespace intersection_eq_l281_281841

def M : Set ℤ := {-1, 0, 1}
def N : Set ℤ := {0, 1, 2}

theorem intersection_eq : M ∩ N = {0, 1} := by
  sorry

end intersection_eq_l281_281841


namespace system_solutions_l281_281784

theorem system_solutions (a b : ℝ) :
  (∃ (x y : ℝ), x^2 - 2*x + y^2 = 0 ∧ a*x + y = a*b) ↔ 0 ≤ b ∧ b ≤ 2 := 
sorry

end system_solutions_l281_281784


namespace tamia_bell_pepper_pieces_l281_281253

def total_pieces (n k p : Nat) : Nat :=
  let slices := n * k
  let half_slices := slices / 2
  let smaller_pieces := half_slices * p
  let total := half_slices + smaller_pieces
  total

theorem tamia_bell_pepper_pieces :
  total_pieces 5 20 3 = 200 :=
by
  sorry

end tamia_bell_pepper_pieces_l281_281253


namespace chess_tournament_total_players_l281_281687

theorem chess_tournament_total_players :
  ∃ n : ℕ,
    n + 12 = 35 ∧
    ∀ p : ℕ,
      (∃ pts : ℕ,
        p = n + 12 ∧
        pts = (p * (p - 1)) / 2 ∧
        pts = n^2 - n + 132) ∧
      ( ∃ (gained_half_points : ℕ → Prop),
          (∀ k ≤ 12, gained_half_points k) ∧
          (∀ k > 12, ¬ gained_half_points k)) :=
by
  sorry

end chess_tournament_total_players_l281_281687


namespace chimpanzee_count_l281_281730

def total_chimpanzees (moving_chimps : ℕ) (staying_chimps : ℕ) : ℕ :=
  moving_chimps + staying_chimps

theorem chimpanzee_count : total_chimpanzees 18 27 = 45 :=
by
  sorry

end chimpanzee_count_l281_281730


namespace jello_cost_l281_281853

def cost_to_fill_tub_with_jello (water_volume_cubic_feet : ℕ) (gallons_per_cubic_foot : ℕ) 
    (pounds_per_gallon : ℕ) (tablespoons_per_pound : ℕ) (cost_per_tablespoon : ℕ) : ℕ :=
  water_volume_cubic_feet * gallons_per_cubic_foot * pounds_per_gallon * tablespoons_per_pound * cost_per_tablespoon

theorem jello_cost (water_volume_cubic_feet : ℕ) (gallons_per_cubic_foot : ℕ) 
    (pounds_per_gallon : ℕ) (tablespoons_per_pound : ℕ) (cost_per_tablespoon : ℕ) : 
    water_volume_cubic_feet = 6 ∧ gallons_per_cubic_foot = 7 ∧ pounds_per_gallon = 8 ∧ 
    tablespoons_per_pound = 1 ∧ cost_per_tablespoon = 1 →
    cost_to_fill_tub_with_jello water_volume_cubic_feet gallons_per_cubic_foot pounds_per_gallon tablespoons_per_pound cost_per_tablespoon = 270 :=
  by 
    sorry

end jello_cost_l281_281853


namespace largest_five_digit_integer_congruent_to_16_mod_25_l281_281740

theorem largest_five_digit_integer_congruent_to_16_mod_25 :
  ∃ x : ℤ, x % 25 = 16 ∧ x < 100000 ∧ ∀ y : ℤ, y % 25 = 16 → y < 100000 → y ≤ x :=
by
  sorry

end largest_five_digit_integer_congruent_to_16_mod_25_l281_281740


namespace identity_true_for_any_abc_l281_281945

theorem identity_true_for_any_abc : 
  ∀ (a b c : ℝ), (a + b) * (b + c) * (c + a) = (a + b + c) * (a * b + b * c + c * a) - a * b * c :=
by
  sorry

end identity_true_for_any_abc_l281_281945


namespace coefficient_x_is_five_l281_281665

theorem coefficient_x_is_five (x y a : ℤ) (h1 : a * x + y = 19) (h2 : x + 3 * y = 1) (h3 : 3 * x + 2 * y = 10) : a = 5 :=
by sorry

end coefficient_x_is_five_l281_281665


namespace opposite_sides_line_l281_281414

theorem opposite_sides_line (a : ℝ) : (0 + 0 - a) * (1 + 1 - a) < 0 → 0 < a ∧ a < 2 := by
  sorry

end opposite_sides_line_l281_281414


namespace factor_poly_l281_281654

theorem factor_poly : 
  ∃ (a b c d e f : ℤ), a < d ∧ (a*x^2 + b*x + c)*(d*x^2 + e*x + f) = (x^2 + 6*x + 9 - 64*x^4) ∧ 
  (a = -8 ∧ b = 1 ∧ c = 3 ∧ d = 8 ∧ e = 1 ∧ f = 3) := 
sorry

end factor_poly_l281_281654


namespace triangle_fraction_squared_l281_281760

theorem triangle_fraction_squared (a b c : ℝ) (h1 : b > a) 
  (h2 : a / b = (1 / 2) * (b / c)) (h3 : a + b + c = 12) 
  (h4 : c = Real.sqrt (a^2 + b^2)) : 
  (a / b)^2 = 1 / 2 := 
by 
  sorry

end triangle_fraction_squared_l281_281760


namespace xy_plus_one_is_perfect_square_l281_281344

theorem xy_plus_one_is_perfect_square (x y : ℕ) (h : 1 / (x : ℝ) + 1 / (y : ℝ) = 1 / (x + 2 : ℝ) + 1 / (y - 2 : ℝ)) :
  ∃ k : ℕ, xy + 1 = k^2 :=
by
  sorry

end xy_plus_one_is_perfect_square_l281_281344


namespace sum_of_repeating_decimals_l281_281769

noncomputable def x := (2 : ℚ) / (3 : ℚ)
noncomputable def y := (5 : ℚ) / (11 : ℚ)

theorem sum_of_repeating_decimals : x + y = (37 : ℚ) / (33 : ℚ) :=
by {
  sorry
}

end sum_of_repeating_decimals_l281_281769


namespace largest_natural_number_has_sum_of_digits_property_l281_281499

noncomputable def largest_nat_num_digital_sum : ℕ :=
  let a : ℕ := 1
  let b : ℕ := 0
  let d3 := a + b
  let d4 := 2 * a + 2 * b
  let d5 := 4 * a + 4 * b
  let d6 := 8 * a + 8 * b
  100000 * a + 10000 * b + 1000 * d3 + 100 * d4 + 10 * d5 + d6

theorem largest_natural_number_has_sum_of_digits_property :
  largest_nat_num_digital_sum = 101248 :=
by
  sorry

end largest_natural_number_has_sum_of_digits_property_l281_281499


namespace sum_of_ages_is_220_l281_281005

-- Definitions based on the conditions
def father_age (S : ℕ) := (7 * S) / 4
def sum_ages (F S : ℕ) := F + S

-- The proof statement
theorem sum_of_ages_is_220 (F S : ℕ) (h1 : 4 * F = 7 * S)
  (h2 : 3 * (F + 10) = 5 * (S + 10)) : sum_ages F S = 220 :=
by
  sorry

end sum_of_ages_is_220_l281_281005


namespace factorial_div_result_l281_281323

theorem factorial_div_result : Nat.factorial 13 / Nat.factorial 11 = 156 :=
sorry

end factorial_div_result_l281_281323


namespace find_range_of_x_l281_281145

noncomputable def f (x : ℝ) : ℝ :=
if x >= 0 then 2 ^ x else 2 ^ (-x)

theorem find_range_of_x (x : ℝ) : 
  f (1 - 2 * x) < f 3 ↔ (-1 < x ∧ x < 2) := 
sorry

end find_range_of_x_l281_281145


namespace penny_nickel_dime_heads_probability_l281_281573

def num_successful_outcomes : Nat :=
1 * 1 * 1 * 2

def total_possible_outcomes : Nat :=
2 ^ 4

def probability_event : ℚ :=
num_successful_outcomes / total_possible_outcomes

theorem penny_nickel_dime_heads_probability :
  probability_event = 1 / 8 := 
by
  sorry

end penny_nickel_dime_heads_probability_l281_281573


namespace tamia_bell_pepper_pieces_l281_281255

theorem tamia_bell_pepper_pieces :
  let bell_peppers := 5
  let slices_per_pepper := 20
  let initial_slices := bell_peppers * slices_per_pepper
  let half_slices_cut := initial_slices / 2
  let small_pieces := half_slices_cut * 3
  let total_pieces := (initial_slices - half_slices_cut) + small_pieces
  total_pieces = 200 :=
by
  sorry

end tamia_bell_pepper_pieces_l281_281255


namespace Brittany_age_after_vacation_l281_281317

variable (Rebecca_age Brittany_age vacation_years : ℕ)
variable (h1 : Rebecca_age = 25)
variable (h2 : Brittany_age = Rebecca_age + 3)
variable (h3 : vacation_years = 4)

theorem Brittany_age_after_vacation : 
    Brittany_age + vacation_years = 32 := 
by
  sorry

end Brittany_age_after_vacation_l281_281317


namespace productProbLessThan36_l281_281995

noncomputable def pacoSpinner : List ℤ := [1, 2, 3, 4, 5, 6]
noncomputable def manuSpinner : List ℤ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

def product_less_than_36 (p : ℤ) (m : ℤ) : Prop :=
  p * m < 36

def probability_product_less_than_36 : ℚ :=
  (∑ m in manuSpinner, ∑ p in pacoSpinner, if product_less_than_36 p m then 1 else 0) /
  (manuSpinner.length * pacoSpinner.length)

theorem productProbLessThan36 : probability_product_less_than_36 = 8 / 30 := sorry

end productProbLessThan36_l281_281995


namespace smallest_solution_proof_l281_281335

noncomputable def smallest_solution (x : ℝ) : Prop :=
  (1 / (x - 1) + 1 / (x - 5) = 4 / (x - 2)) ∧ 
  (∀ y : ℝ, 1 / (y - 1) + 1 / (y - 5) = 4 / (y - 2) → y ≥ x)

theorem smallest_solution_proof : smallest_solution ( (7 - Real.sqrt 33) / 2 ) :=
sorry

end smallest_solution_proof_l281_281335


namespace bridge_length_l281_281132

theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (time_to_cross_bridge : ℝ) 
  (train_speed_m_s : train_speed_kmh * (1000 / 3600) = 15) : 
  train_length = 110 → train_speed_kmh = 54 → time_to_cross_bridge = 16.13204276991174 → 
  ((train_speed_kmh * (1000 / 3600)) * time_to_cross_bridge - train_length = 131.9806415486761) :=
by
  intros h1 h2 h3
  sorry

end bridge_length_l281_281132


namespace combined_moment_l281_281833

-- Definitions based on given conditions
variables (P Q Z : ℝ) -- Positions of the points and center of mass
variables (p q : ℝ) -- Masses of the points
variables (Mom_s : ℝ → ℝ) -- Moment function relative to axis s

-- Given:
-- 1. Positions P and Q with masses p and q respectively
-- 2. Combined point Z with total mass p + q
-- 3. Moments relative to the axis s: Mom_s P and Mom_s Q
-- To Prove: Moment of the combined point Z relative to axis s
-- is the sum of the moments of P and Q relative to the same axis

theorem combined_moment (hZ : Z = (P * p + Q * q) / (p + q)) :
  Mom_s Z = Mom_s P + Mom_s Q :=
sorry

end combined_moment_l281_281833


namespace proof_problem_l281_281410

variable {a b : ℤ}

theorem proof_problem (h1 : ∃ k : ℤ, a = 4 * k) (h2 : ∃ l : ℤ, b = 8 * l) : 
  (∃ m : ℤ, b = 4 * m) ∧
  (∃ n : ℤ, a - b = 4 * n) ∧
  (∃ p : ℤ, a + b = 2 * p) := 
by
  sorry

end proof_problem_l281_281410


namespace volunteer_group_selection_l281_281099

theorem volunteer_group_selection :
  let M := 4  -- Number of male teachers
  let F := 5  -- Number of female teachers
  let G := 3  -- Total number of teachers in the group
  -- Calculate the number of ways to select 2 male teachers and 1 female teacher
  let ways1 := (Nat.choose M 2) * (Nat.choose F 1)
  -- Calculate the number of ways to select 1 male teacher and 2 female teachers
  let ways2 := (Nat.choose M 1) * (Nat.choose F 2)
  -- The total number of ways to form the group
  ways1 + ways2 = 70 := by sorry

end volunteer_group_selection_l281_281099


namespace max_y_value_of_3x_plus_4_div_x_corresponds_value_of_x_l281_281944

noncomputable def y (x : ℝ) : ℝ := 3 * x + 4 / x
def max_value (x : ℝ) := y x ≤ -4 * Real.sqrt 3

theorem max_y_value_of_3x_plus_4_div_x (h : x < 0) : max_value x :=
sorry

theorem corresponds_value_of_x (x : ℝ) (h : x = -2 * Real.sqrt 3 / 3) : y x = -4 * Real.sqrt 3 :=
sorry

end max_y_value_of_3x_plus_4_div_x_corresponds_value_of_x_l281_281944


namespace average_spring_headcount_average_fall_headcount_l281_281485

namespace AverageHeadcount

def springHeadcounts := [10900, 10500, 10700, 11300]
def fallHeadcounts := [11700, 11500, 11600, 11300]

def averageHeadcount (counts : List ℕ) : ℕ :=
  counts.sum / counts.length

theorem average_spring_headcount :
  averageHeadcount springHeadcounts = 10850 := by
  sorry

theorem average_fall_headcount :
  averageHeadcount fallHeadcounts = 11525 := by
  sorry

end AverageHeadcount

end average_spring_headcount_average_fall_headcount_l281_281485


namespace bird_family_problem_l281_281272

def initial_bird_families (f s i : Nat) : Prop :=
  i = f + s

theorem bird_family_problem : initial_bird_families 32 35 67 :=
by
  -- Proof would go here
  sorry

end bird_family_problem_l281_281272


namespace two_pow_p_plus_three_pow_p_not_nth_power_l281_281967

theorem two_pow_p_plus_three_pow_p_not_nth_power (p n : ℕ) (prime_p : Nat.Prime p) (one_lt_n : 1 < n) :
  ¬ ∃ k : ℕ, 2 ^ p + 3 ^ p = k ^ n :=
sorry

end two_pow_p_plus_three_pow_p_not_nth_power_l281_281967


namespace donation_total_correct_l281_281641

noncomputable def total_donation (t : ℝ) (y : ℝ) (x : ℝ) : ℝ :=
  t + t + x
  
theorem donation_total_correct (t : ℝ) (y : ℝ) (x : ℝ)
  (h1 : t = 570.00) (h2 : y = 140.00) (h3 : t = x + y) : total_donation t y x = 1570.00 :=
by
  sorry

end donation_total_correct_l281_281641


namespace min_value_of_A_sq_sub_B_sq_l281_281708

noncomputable def A (x y z : ℝ) : ℝ :=
  Real.sqrt (x + 4) + Real.sqrt (y + 7) + Real.sqrt (z + 13)

noncomputable def B (x y z : ℝ) : ℝ :=
  Real.sqrt (2 * x + 2) + Real.sqrt (2 * y + 2) + Real.sqrt (2 * z + 2)

theorem min_value_of_A_sq_sub_B_sq (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) : 
  A x y z ^ 2 - B x y z ^ 2 ≥ 36 :=
  sorry

end min_value_of_A_sq_sub_B_sq_l281_281708


namespace probability_sum_11_is_1_over_8_l281_281137

theorem probability_sum_11_is_1_over_8 : 
  let outcomes := ({2, 3, 4, 5, 6, 7, 8, 9} : Finset ℕ)
  let pairs := {p : ℕ × ℕ | p.1 ∈ outcomes ∧ p.2 ∈ outcomes ∧ p.1 + p.2 = 11}
  let total_outcomes := outcomes.card * outcomes.card
  let favorable_outcomes := pairs.card
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 8
:= by 
  sorry

end probability_sum_11_is_1_over_8_l281_281137


namespace how_many_buns_each_student_gets_l281_281356

theorem how_many_buns_each_student_gets 
  (packages : ℕ) 
  (buns_per_package : ℕ) 
  (classes : ℕ) 
  (students_per_class : ℕ)
  (h1 : buns_per_package = 8)
  (h2 : packages = 30)
  (h3 : classes = 4)
  (h4 : students_per_class = 30) :
  (packages * buns_per_package) / (classes * students_per_class) = 2 :=
by sorry

end how_many_buns_each_student_gets_l281_281356


namespace classroom_not_1_hectare_l281_281081

def hectare_in_sq_meters : ℕ := 10000
def classroom_area_approx : ℕ := 60

theorem classroom_not_1_hectare : ¬ (classroom_area_approx = hectare_in_sq_meters) :=
by 
  sorry

end classroom_not_1_hectare_l281_281081


namespace ratio_Laura_to_Ken_is_2_to_1_l281_281648

def Don_paint_tiles_per_minute : ℕ := 3

def Ken_paint_tiles_per_minute : ℕ := Don_paint_tiles_per_minute + 2

def multiple : ℕ := sorry -- Needs to be introduced, not directly from the solution steps

def Laura_paint_tiles_per_minute : ℕ := multiple * Ken_paint_tiles_per_minute

def Kim_paint_tiles_per_minute : ℕ := Laura_paint_tiles_per_minute - 3

def total_tiles_in_15_minutes : ℕ := 375

def total_tiles_per_minute : ℕ := total_tiles_in_15_minutes / 15

def total_tiles_equation : Prop :=
  Don_paint_tiles_per_minute + Ken_paint_tiles_per_minute + Laura_paint_tiles_per_minute + Kim_paint_tiles_per_minute = total_tiles_per_minute

theorem ratio_Laura_to_Ken_is_2_to_1 :
  (total_tiles_equation → Laura_paint_tiles_per_minute / Ken_paint_tiles_per_minute = 2) := sorry

end ratio_Laura_to_Ken_is_2_to_1_l281_281648


namespace find_L_l281_281676

theorem find_L (RI G SP T M N : ℝ) (h1 : RI + G + SP = 50) (h2 : RI + T + M = 63) (h3 : G + T + SP = 25) 
(h4 : SP + M = 13) (h5 : M + RI = 48) (h6 : N = 1) :
  ∃ L : ℝ, L * M * T + SP * RI * N * G = 2023 ∧ L = 341 / 40 := 
by
  sorry

end find_L_l281_281676


namespace factor_expression_l281_281332

theorem factor_expression (x : ℝ) : 35 * x ^ 13 + 245 * x ^ 26 = 35 * x ^ 13 * (1 + 7 * x ^ 13) :=
by {
  sorry
}

end factor_expression_l281_281332


namespace smallest_possible_value_l281_281798

theorem smallest_possible_value (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (hxy : x ≠ y) (h : (1 / (x : ℝ)) + (1 / (y : ℝ)) = 1 / 15) : x + y = 64 :=
sorry

end smallest_possible_value_l281_281798


namespace ski_boat_rental_cost_per_hour_l281_281699

-- Let the cost per hour to rent a ski boat be x dollars
variable (x : ℝ)

-- Conditions
def cost_sailboat : ℝ := 60
def duration : ℝ := 3 * 2 -- 3 hours a day for 2 days
def cost_ken : ℝ := cost_sailboat * 2 -- Ken's total cost
def additional_cost : ℝ := 120
def cost_aldrich : ℝ := cost_ken + additional_cost -- Aldrich's total cost

-- Statement to prove
theorem ski_boat_rental_cost_per_hour (h : (duration * x = cost_aldrich)) : x = 40 := by
  sorry

end ski_boat_rental_cost_per_hour_l281_281699


namespace projection_of_a_on_b_l281_281970

open Real -- Use real numbers for vector operations

variables (a b : ℝ) -- Define a and b to be real numbers

-- Define the conditions as assumptions in Lean 4
def vector_magnitude_a (a : ℝ) : Prop := abs a = 1
def vector_magnitude_b (b : ℝ) : Prop := abs b = 1
def vector_dot_product (a b : ℝ) : Prop := (a + b) * b = 3 / 2

-- Define the goal to prove, using the assumptions
theorem projection_of_a_on_b (ha : vector_magnitude_a a) (hb : vector_magnitude_b b) (h_ab : vector_dot_product a b) : (abs a) * (a / b) = 1 / 2 :=
by
  sorry

end projection_of_a_on_b_l281_281970


namespace outfit_choices_l281_281361

noncomputable def calculate_outfits : Nat :=
  let shirts := 6
  let pants := 6
  let hats := 6
  let total_outfits := shirts * pants * hats
  let matching_colors := 4 -- tan, black, blue, gray for matching
  total_outfits - matching_colors

theorem outfit_choices : calculate_outfits = 212 :=
by
  sorry

end outfit_choices_l281_281361


namespace white_longer_than_blue_l281_281030

noncomputable def whiteLineInches : ℝ := 7.666666666666667
noncomputable def blueLineInches : ℝ := 3.3333333333333335
noncomputable def inchToCm : ℝ := 2.54
noncomputable def cmToMm : ℝ := 10

theorem white_longer_than_blue :
  let whiteLineCm := whiteLineInches * inchToCm
  let blueLineCm := blueLineInches * inchToCm
  let differenceCm := whiteLineCm - blueLineCm
  let differenceMm := differenceCm * cmToMm
  differenceMm = 110.05555555555553 := by
  sorry

end white_longer_than_blue_l281_281030


namespace john_naps_70_days_l281_281702

def total_naps_in_days (naps_per_week nap_duration days_in_week total_days : ℕ) : ℕ :=
  let total_weeks := total_days / days_in_week
  let total_naps := total_weeks * naps_per_week
  total_naps * nap_duration

theorem john_naps_70_days
  (naps_per_week : ℕ)
  (nap_duration : ℕ)
  (days_in_week : ℕ)
  (total_days : ℕ)
  (h_naps_per_week : naps_per_week = 3)
  (h_nap_duration : nap_duration = 2)
  (h_days_in_week : days_in_week = 7)
  (h_total_days : total_days = 70) :
  total_naps_in_days naps_per_week nap_duration days_in_week total_days = 60 :=
by
  rw [h_naps_per_week, h_nap_duration, h_days_in_week, h_total_days]
  sorry

end john_naps_70_days_l281_281702


namespace old_man_coins_l281_281746

theorem old_man_coins (x y : ℕ) (h : x ≠ y) (h_condition : x^2 - y^2 = 81 * (x - y)) : x + y = 81 := 
sorry

end old_man_coins_l281_281746


namespace coins_remainder_l281_281013

theorem coins_remainder 
  (n : ℕ)
  (h₁ : n % 8 = 6)
  (h₂ : n % 7 = 2)
  (h₃ : n = 30) :
  n % 9 = 3 :=
sorry

end coins_remainder_l281_281013


namespace sin_half_alpha_l281_281177

theorem sin_half_alpha (α : ℝ) (hα : 0 < α ∧ α < π / 2) (hcosα : cos α = (1 + real.sqrt 5) / 4) :
  sin (α / 2) = (real.sqrt 5 - 1) / 4 :=
by
  sorry

end sin_half_alpha_l281_281177


namespace final_statement_l281_281159

variable (f : ℝ → ℝ)

-- Conditions
axiom even_function : ∀ x, f (x) = f (-x)
axiom periodic_minus_one : ∀ x, f (x + 1) = -f (x)
axiom increasing_on_neg_one_to_zero : ∀ x y, -1 ≤ x ∧ x < y ∧ y ≤ 0 → f (x) < f (y)

-- Statement
theorem final_statement :
  (∀ x, f (x + 2) = f (x)) ∧
  (¬ (∀ x, 0 ≤ x ∧ x ≤ 1 → f (x) < f (x + 1))) ∧
  (∀ x y, 1 ≤ x ∧ x < y ∧ y ≤ 2 → f (x) < f (y)) ∧
  (f (2) = f (0)) :=
by
  sorry

end final_statement_l281_281159


namespace determinant_equality_l281_281511

-- Given values p, q, r, s such that the determinant of the first matrix is 5
variables {p q r s : ℝ}

-- Define the determinant condition
def det_condition (p q r s : ℝ) : Prop := p * s - q * r = 5

-- State the theorem that we need to prove
theorem determinant_equality (h : det_condition p q r s) :
  p * (5*r + 2*s) - r * (5*p + 2*q) = 10 :=
sorry

end determinant_equality_l281_281511


namespace problem_solution_l281_281709

noncomputable def M (a b c : ℝ) : ℝ := (1 - 1/a) * (1 - 1/b) * (1 - 1/c)

theorem problem_solution (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (habc : a + b + c = 1) :
  M a b c ≤ -8 :=
sorry

end problem_solution_l281_281709


namespace original_number_is_600_l281_281745

theorem original_number_is_600 (x : Real) (h : x * 1.10 = 660) : x = 600 := by
  sorry

end original_number_is_600_l281_281745


namespace product_xyz_l281_281052

theorem product_xyz (x y z : ℝ) (h1 : x + 1 / y = 3) (h2 : y + 1 / z = 5) : 
  x * y * z = 1 / 9 := 
by
  sorry

end product_xyz_l281_281052


namespace sin_half_angle_l281_281163

theorem sin_half_angle (α : ℝ) (h0 : 0 < α ∧ α < π / 2) (h1 : cos α = (1 + real.sqrt 5) / 4) :
  sin (α / 2) = (-1 + real.sqrt 5) / 4 :=
by sorry

end sin_half_angle_l281_281163


namespace no_15_students_with_unique_colors_l281_281592

-- Conditions as definitions
def num_students : Nat := 30
def num_colors : Nat := 15

-- The main statement
theorem no_15_students_with_unique_colors
  (students : Fin num_students → (Fin num_colors × Fin num_colors)) :
  ¬ ∃ (subset : Fin 15 → Fin num_students),
    ∀ i j (hi : i ≠ j), (students (subset i)).1 ≠ (students (subset j)).1 ∧
                         (students (subset i)).2 ≠ (students (subset j)).2 :=
by sorry

end no_15_students_with_unique_colors_l281_281592


namespace probability_grunters_win_at_least_4_of_5_games_l281_281576

theorem probability_grunters_win_at_least_4_of_5_games :
  let p := 3/5 in
  let q := 2/5 in
  let n := 5 in
  let k := 4 in
  (nat.choose n k) * (p ^ k) * (q ^ (n - k)) + (p ^ n) = 1053 / 3125 := by
  -- We state the problem and leave the proof as an exercise
  sorry

end probability_grunters_win_at_least_4_of_5_games_l281_281576


namespace series_proof_l281_281556

theorem series_proof (a b : ℝ) (h : (∑' n : ℕ, (-1)^n * a / b^(n+1)) = 6) : 
  (∑' n : ℕ, (-1)^n * a / (a - b)^(n+1)) = 6 / 7 := 
sorry

end series_proof_l281_281556


namespace students_still_in_school_l281_281627

-- Declare the number of students initially in the school
def initial_students : Nat := 1000

-- Declare that half of the students were taken to the beach
def taken_to_beach (total_students : Nat) : Nat := total_students / 2

-- Declare that half of the remaining students were sent home
def sent_home (remaining_students : Nat) : Nat := remaining_students / 2

-- Declare the theorem to prove the final number of students still in school
theorem students_still_in_school : 
  let total_students := initial_students in
  let students_at_beach := taken_to_beach total_students in
  let students_remaining := total_students - students_at_beach in
  let students_sent_home := sent_home students_remaining in
  let students_left := students_remaining - students_sent_home in
  students_left = 250 := by
  sorry

end students_still_in_school_l281_281627


namespace find_a_l281_281861

noncomputable def f (a x : ℝ) : ℝ := a^x + Real.logb a (x + 1)

theorem find_a : 
  ( ∀ a : ℝ, 
    (∀ x : ℝ,  0 ≤ x ∧ x ≤ 1 → f a 0 + f a 1 = a) → a = 1/2 ) :=
sorry

end find_a_l281_281861


namespace algebraic_expression_value_l281_281336

theorem algebraic_expression_value (x : ℝ) (h : 2 * x^2 - x - 1 = 5) : 6 * x^2 - 3 * x - 9 = 9 := 
by 
  sorry

end algebraic_expression_value_l281_281336


namespace steve_speed_ratio_l281_281584

/-- Define the distance from Steve's house to work. -/
def distance_to_work := 30

/-- Define the total time spent on the road by Steve. -/
def total_time_on_road := 6

/-- Define Steve's speed on the way back from work. -/
def speed_back := 15

/-- Calculate the ratio of Steve's speed on the way back to his speed on the way to work. -/
theorem steve_speed_ratio (v : ℝ) (h_v_pos : v > 0) 
    (h1 : distance_to_work / v + distance_to_work / speed_back = total_time_on_road) :
    speed_back / v = 2 := 
by
  -- We will provide the proof here
  sorry

end steve_speed_ratio_l281_281584


namespace descending_order_of_weights_l281_281264

variables (S B K A : ℝ)

theorem descending_order_of_weights
  (h1 : S > B)
  (h2 : A + B > S + K)
  (h3 : K + A = S + B) :
  A > S ∧ S > B ∧ B > K :=
by
  -- Sketch of proof for illustration:
  -- From h3: K + A = S + B, we can rearrange to get A = S + B - K
  -- Substitute A in h2: S + B - K + B > S + K
  -- Simplify: S + 2B - K > S + K
  -- Therefore: 2B > 2K, hence B > K
  -- From h1 and derived B > K, we also have A > S
  -- Summarize the order: A > S > B > K
  sorry

end descending_order_of_weights_l281_281264


namespace smallest_x_plus_y_l281_281794

theorem smallest_x_plus_y (x y : ℕ) (h_xy_not_equal : x ≠ y) (h_condition : (1 : ℚ) / x + 1 / y = 1 / 15) :
  x + y = 64 :=
sorry

end smallest_x_plus_y_l281_281794


namespace remainder_when_x_plus_3uy_divided_by_y_eq_v_l281_281080

theorem remainder_when_x_plus_3uy_divided_by_y_eq_v
  (x y u v : ℕ) (h_pos_y : 0 < y) (h_division_algo : x = u * y + v) (h_remainder : 0 ≤ v ∧ v < y) :
  (x + 3 * u * y) % y = v :=
by
  sorry

end remainder_when_x_plus_3uy_divided_by_y_eq_v_l281_281080


namespace fraction_of_product_l281_281875

theorem fraction_of_product (c d: ℕ) 
  (h1: 5 * 64 + 4 * 8 + 3 = 355)
  (h2: 2 * (10 * c + d) = 355)
  (h3: c < 10)
  (h4: d < 10):
  (c * d : ℚ) / 12 = 5 / 4 :=
by
  sorry

end fraction_of_product_l281_281875


namespace tamia_bell_pepper_pieces_l281_281251

def total_pieces (n k p : Nat) : Nat :=
  let slices := n * k
  let half_slices := slices / 2
  let smaller_pieces := half_slices * p
  let total := half_slices + smaller_pieces
  total

theorem tamia_bell_pepper_pieces :
  total_pieces 5 20 3 = 200 :=
by
  sorry

end tamia_bell_pepper_pieces_l281_281251


namespace chicken_distribution_l281_281947

theorem chicken_distribution :
  (nat.choose 4 2) = 6 :=
by
  -- The proof is skipped
  sorry

end chicken_distribution_l281_281947


namespace pre_bought_tickets_l281_281726

theorem pre_bought_tickets (P : ℕ) 
  (h1 : ∃ P, 155 * P + 2900 = 6000) : P = 20 :=
by {
  -- Insert formalization of steps leading to P = 20
  sorry
}

end pre_bought_tickets_l281_281726


namespace number_of_roses_ian_kept_l281_281362

-- Definitions representing the conditions
def initial_roses : ℕ := 20
def roses_to_mother : ℕ := 6
def roses_to_grandmother : ℕ := 9
def roses_to_sister : ℕ := 4

-- The theorem statement we want to prove
theorem number_of_roses_ian_kept : (initial_roses - (roses_to_mother + roses_to_grandmother + roses_to_sister) = 1) :=
by
  sorry

end number_of_roses_ian_kept_l281_281362


namespace number_of_sheep_l281_281096

theorem number_of_sheep (legs animals : ℕ) (h1 : legs = 60) (h2 : animals = 20)
  (chickens sheep : ℕ) (hc : chickens + sheep = animals) (hl : 2 * chickens + 4 * sheep = legs) :
  sheep = 10 :=
sorry

end number_of_sheep_l281_281096


namespace probability_abs_ξ_lt_1_96_l281_281830

-- Random variable ξ following a standard normal distribution N(0,1)
noncomputable def ξ : ℝ → ℝ := sorry

-- Given condition
axiom H1 : ∀ x : ℝ, P(ξ ≤ x) = ∫ y in Iic x, (1 / (sqrt (2 * π))) * exp ((-y ^ 2) / 2) dy

-- Specific given probability
axiom H2 : P(ξ ≤ -1.96) = 0.025

-- The theorem to prove
theorem probability_abs_ξ_lt_1_96 : P(abs ξ < 1.96) = 0.950 :=
by
  sorry -- Proof omitted

end probability_abs_ξ_lt_1_96_l281_281830


namespace squirrels_acorns_l281_281754

theorem squirrels_acorns (total_acorns : ℕ) (num_squirrels : ℕ) (required_per_squirrel : ℕ) 
  (h1 : total_acorns = 575) (h2 : num_squirrels = 5) (h3 : required_per_squirrel = 130) :
  let acorns_per_squirrel := total_acorns / num_squirrels in
  required_per_squirrel - acorns_per_squirrel = 15 :=
by
  sorry

end squirrels_acorns_l281_281754


namespace battery_difference_l281_281420

def flashlights_batteries := 2
def toys_batteries := 15
def difference := 13

theorem battery_difference : toys_batteries - flashlights_batteries = difference :=
by
  sorry

end battery_difference_l281_281420


namespace isosceles_triangle_perimeter_eq_70_l281_281266

-- Define the conditions
def is_equilateral_triangle (a b c : ℕ) : Prop :=
  a = b ∧ b = c

def is_isosceles_triangle (a b c : ℕ) : Prop :=
  a = b ∨ a = c ∨ b = c

-- Given conditions
def equilateral_triangle_perimeter : ℕ := 60
def isosceles_triangle_base : ℕ := 30

-- Calculate the side of equilateral triangle
def equilateral_triangle_side : ℕ := equilateral_triangle_perimeter / 3

-- Lean 4 statement
theorem isosceles_triangle_perimeter_eq_70 :
  ∃ (a b c : ℕ), is_equilateral_triangle a b c ∧ 
  a + b + c = equilateral_triangle_perimeter →
  (is_isosceles_triangle a a isosceles_triangle_base) →
  a + a + isosceles_triangle_base = 70 :=
by
  sorry -- proof is omitted

end isosceles_triangle_perimeter_eq_70_l281_281266


namespace inverse_of_A_cubed_l281_281962

noncomputable def A_inv : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![3, 7],
    ![-2, -5]]

theorem inverse_of_A_cubed :
  (A_inv ^ 3)⁻¹ = ![![13, -15],
                     ![-14, -29]] :=
by
  sorry

end inverse_of_A_cubed_l281_281962


namespace Hoelder_l281_281997

variable (A B p q : ℝ)

theorem Hoelder (hA : 0 < A) (hB : 0 < B) (hp : 0 < p) (hq : 0 < q) (h : 1 / p + 1 / q = 1) : 
  A^(1/p) * B^(1/q) ≤ A / p + B / q := 
sorry

end Hoelder_l281_281997


namespace simplify_expression_l281_281643

variable (i : ℂ)

-- Define the conditions

def i_squared_eq_neg_one : Prop := i^2 = -1
def i_cubed_eq_neg_i : Prop := i^3 = i * i^2 ∧ i^3 = -i
def i_fourth_eq_one : Prop := i^4 = (i^2)^2 ∧ i^4 = 1
def i_fifth_eq_i : Prop := i^5 = i * i^4 ∧ i^5 = i

-- Define the proof problem

theorem simplify_expression (h1 : i_squared_eq_neg_one i) (h2 : i_cubed_eq_neg_i i) (h3 : i_fourth_eq_one i) (h4 : i_fifth_eq_i i) : 
  i + i^2 + i^3 + i^4 + i^5 = i := 
  by sorry

end simplify_expression_l281_281643


namespace goshawk_eurasian_reserve_l281_281685

theorem goshawk_eurasian_reserve (B : ℝ)
  (h1 : 0.30 * B + 0.28 * B + K * 0.28 * B = 0.65 * B)
  : K = 0.25 :=
by sorry

end goshawk_eurasian_reserve_l281_281685


namespace smallest_sum_of_inverses_l281_281814

theorem smallest_sum_of_inverses 
  (x y : ℕ) (hx : x ≠ y) (h1 : 0 < x) (h2 : 0 < y) (h_condition : (1 / x : ℚ) + 1 / y = 1 / 15) :
  x + y = 64 := 
sorry

end smallest_sum_of_inverses_l281_281814


namespace arrangement_plans_count_l281_281948

/-- 
Given 5 volunteers and the requirement to select 3 people to serve at 
the Swiss Pavilion, the Spanish Pavilion, and the Italian Pavilion such that
1. Each pavilion is assigned 1 person.
2. Individuals A and B cannot go to the Swiss Pavilion.
Prove that the total number of different arrangement plans is 36.
-/

theorem arrangement_plans_count (volunteers : Finset ℕ) (A B : ℕ) (Swiss Pavilion : ℕ) :
  volunteers.card = 5 →
  A ≠ Swiss →
  B ≠ Swiss →
  ∃ arrangements, arrangements.card = 36 :=
begin
  sorry
end

end arrangement_plans_count_l281_281948


namespace fg_value_l281_281365

def g (x : ℕ) : ℕ := 4 * x + 10
def f (x : ℕ) : ℕ := 6 * x - 12

theorem fg_value : f (g 10) = 288 := by
  sorry

end fg_value_l281_281365


namespace brittany_age_after_vacation_l281_281320

-- Definitions of the conditions
def rebecca_age : ℕ := 25
def age_difference : ℕ := 3
def vacation_years : ℕ := 4

-- Prove the main statement
theorem brittany_age_after_vacation : rebecca_age + age_difference + vacation_years = 32 := by
  sorry

end brittany_age_after_vacation_l281_281320


namespace business_total_profit_l281_281903

noncomputable def total_profit (spending_ratio income_ratio total_income : ℕ) : ℕ :=
  let total_parts := spending_ratio + income_ratio
  let one_part_value := total_income / income_ratio
  let spending := spending_ratio * one_part_value
  total_income - spending

theorem business_total_profit :
  total_profit 5 9 108000 = 48000 :=
by
  -- We omit the proof steps, as instructed.
  sorry

end business_total_profit_l281_281903


namespace triangle_area_r_l281_281634

theorem triangle_area_r (r : ℝ) (h₁ : 12 ≤ (r - 3) ^ (3 / 2)) (h₂ : (r - 3) ^ (3 / 2) ≤ 48) : 15 ≤ r ∧ r ≤ 19 := by
  sorry

end triangle_area_r_l281_281634


namespace count_integers_P_leq_0_l281_281771

def P(x : ℤ) : ℤ := 
  (x - 1^3) * (x - 2^3) * (x - 3^3) * (x - 4^3) * (x - 5^3) *
  (x - 6^3) * (x - 7^3) * (x - 8^3) * (x - 9^3) * (x - 10^3) *
  (x - 11^3) * (x - 12^3) * (x - 13^3) * (x - 14^3) * (x - 15^3) *
  (x - 16^3) * (x - 17^3) * (x - 18^3) * (x - 19^3) * (x - 20^3) *
  (x - 21^3) * (x - 22^3) * (x - 23^3) * (x - 24^3) * (x - 25^3) *
  (x - 26^3) * (x - 27^3) * (x - 28^3) * (x - 29^3) * (x - 30^3) *
  (x - 31^3) * (x - 32^3) * (x - 33^3) * (x - 34^3) * (x - 35^3) *
  (x - 36^3) * (x - 37^3) * (x - 38^3) * (x - 39^3) * (x - 40^3) *
  (x - 41^3) * (x - 42^3) * (x - 43^3) * (x - 44^3) * (x - 45^3) *
  (x - 46^3) * (x - 47^3) * (x - 48^3) * (x - 49^3) * (x - 50^3)

theorem count_integers_P_leq_0 : 
  ∃ n : ℕ, n = 15650 ∧ ∀ k : ℤ, (P k ≤ 0) → (n = 15650) :=
by sorry

end count_integers_P_leq_0_l281_281771


namespace distance_between_trees_l281_281010

theorem distance_between_trees (yard_length : ℕ) (number_of_trees : ℕ) (number_of_gaps : ℕ)
  (h1 : yard_length = 400) (h2 : number_of_trees = 26) (h3 : number_of_gaps = number_of_trees - 1) :
  yard_length / number_of_gaps = 16 := by
  sorry

end distance_between_trees_l281_281010


namespace Dan_age_is_28_l281_281313

theorem Dan_age_is_28 (B D : ℕ) (h1 : B = D - 3) (h2 : B + D = 53) : D = 28 :=
by
  sorry

end Dan_age_is_28_l281_281313


namespace sophie_clothes_expense_l281_281729

theorem sophie_clothes_expense :
  let initial_fund := 260
  let shirt_cost := 18.50
  let trousers_cost := 63
  let num_shirts := 2
  let num_remaining_clothes := 4
  let total_spent := num_shirts * shirt_cost + trousers_cost
  let remaining_amount := initial_fund - total_spent
  let individual_item_cost := remaining_amount / num_remaining_clothes
  individual_item_cost = 40 := 
by 
  sorry

end sophie_clothes_expense_l281_281729


namespace work_completion_l281_281293

theorem work_completion 
  (x_work_days : ℕ) 
  (y_work_days : ℕ) 
  (y_worked_days : ℕ) 
  (x_rate := 1 / (x_work_days : ℚ)) 
  (y_rate := 1 / (y_work_days : ℚ)) 
  (work_remaining := 1 - y_rate * y_worked_days) 
  (remaining_work_days := work_remaining / x_rate) : 
  x_work_days = 18 → 
  y_work_days = 15 → 
  y_worked_days = 5 → 
  remaining_work_days = 12 := 
by
  intros
  sorry

end work_completion_l281_281293


namespace cos_7_theta_l281_281530

variable (θ : Real)

namespace CosineProof

theorem cos_7_theta (h : Real.cos θ = 1 / 4) : Real.cos (7 * θ) = -5669 / 16384 := by
  sorry

end CosineProof

end cos_7_theta_l281_281530


namespace total_kids_played_tag_with_l281_281858

theorem total_kids_played_tag_with : 
  let kids_mon : Nat := 12
  let kids_tues : Nat := 7
  let kids_wed : Nat := 15
  let kids_thurs : Nat := 10
  let kids_fri : Nat := 18
  (kids_mon + kids_tues + kids_wed + kids_thurs + kids_fri) = 62 := by
  sorry

end total_kids_played_tag_with_l281_281858


namespace ellipse_tangent_line_l281_281212

theorem ellipse_tangent_line (m : ℝ) : 
  (∀ (x y : ℝ), (x ^ 2 / 4) + (y ^ 2 / m) = 1 → (y = mx + 2)) → m = 1 :=
by sorry

end ellipse_tangent_line_l281_281212


namespace find_value_l281_281961

theorem find_value (a b : ℝ) (h : (a + 1)^2 + |b - 2| = 0) : a^2006 + (a + b)^2007 = 2 := 
by
  sorry

end find_value_l281_281961


namespace find_a_l281_281231

noncomputable def A : Set ℝ := {x | x^2 - 8*x + 15 = 0}
noncomputable def B (a : ℝ) : Set ℝ := {x | x * a = 1}
axiom A_is_B (a : ℝ) : A ∩ B a = B a → (a = 0) ∨ (a = 1/3) ∨ (a = 1/5)

-- statement to prove
theorem find_a (a : ℝ) (h : A ∩ B a = B a) : (a = 0) ∨ (a = 1/3) ∨ (a = 1/5) :=
by 
  apply A_is_B
  assumption

end find_a_l281_281231


namespace minimum_value_of_f_l281_281056

noncomputable def f (x : ℝ) : ℝ := (x^2 - 1) * (x^2 - 4 * x + 3)

theorem minimum_value_of_f : ∃ m : ℝ, m = -16 ∧ ∀ x : ℝ, f x ≥ m :=
by
  use -16
  sorry

end minimum_value_of_f_l281_281056


namespace largest_prime_divisor_of_1202102_5_l281_281942

def base_5_to_decimal (n : String) : ℕ := 
  let digits := n.toList.map (λ c => c.toNat - '0'.toNat)
  digits.foldr (λ (digit acc : ℕ) => acc * 5 + digit) 0

def largest_prime_factor (n : ℕ) : ℕ := sorry -- Placeholder for the actual factorization logic.

theorem largest_prime_divisor_of_1202102_5 : 
  largest_prime_factor (base_5_to_decimal "1202102") = 307 := 
sorry

end largest_prime_divisor_of_1202102_5_l281_281942


namespace A_ge_B_l281_281789

def A (a b : ℝ) : ℝ := a^3 + 3 * a^2 * b^2 + 2 * b^2 + 3 * b
def B (a b : ℝ) : ℝ := a^3 - a^2 * b^2 + b^2 + 3 * b

theorem A_ge_B (a b : ℝ) : A a b ≥ B a b := by
  sorry

end A_ge_B_l281_281789


namespace number_division_l281_281464

theorem number_division (x : ℚ) (h : x / 6 = 1 / 10) : (x / (3 / 25)) = 5 :=
by {
  sorry
}

end number_division_l281_281464


namespace work_completion_times_l281_281866

variable {M P S : ℝ} -- Let M, P, and S be work rates for Matt, Peter, and Sarah.

theorem work_completion_times (h1 : M + P + S = 1 / 15)
                             (h2 : 10 * (P + S) = 7 / 15) :
                             (1 / M = 50) ∧ (1 / (P + S) = 150 / 7) :=
by
  -- Proof comes here
  -- Calculation skipped
  sorry

end work_completion_times_l281_281866


namespace find_angle_C_find_triangle_area_l281_281541

theorem find_angle_C (A B C : ℝ) (a b c : ℝ) 
  (h1 : B = Real.pi / 4) 
  (h2 : Real.cos A - Real.cos (2 * A) = 0) 
  (h3 : B + C + A = Real.pi) :
  C = Real.pi / 12 :=
by
  sorry

theorem find_triangle_area (A B C : ℝ) (a b c : ℝ)
  (h1 : B = Real.pi / 4) 
  (h2 : Real.cos A - Real.cos (2 * A) = 0) 
  (h3 : b^2 + c^2 = a - b * c + 2) 
  (h4 : B + C + A = Real.pi) 
  (h5 : a^2 = b^2 + c^2 + b * c) :
  (1/2) * a * b * Real.sin C = 1 - Real.sqrt 3 / 3 :=
by
  sorry

end find_angle_C_find_triangle_area_l281_281541


namespace bottom_left_square_side_length_l281_281887

theorem bottom_left_square_side_length (x y : ℕ) 
  (h1 : 1 + (x - 1) = 1) 
  (h2 : 2 * x - 1 = (x - 2) + (x - 3) + y) :
  y = 4 :=
sorry

end bottom_left_square_side_length_l281_281887


namespace max_b_c_plus_four_over_a_l281_281045

theorem max_b_c_plus_four_over_a (a b c : ℝ) (ha : a < 0)
  (h_quad : ∀ x : ℝ, -1 < x ∧ x < 2 → (a * x^2 + b * x + c) > 0) : 
  b - c + 4 / a ≤ -4 :=
sorry

end max_b_c_plus_four_over_a_l281_281045


namespace mn_minus_n_values_l281_281839

theorem mn_minus_n_values (m n : ℝ) (h1 : |m| = 4) (h2 : |n| = 2.5) (h3 : m * n < 0) :
  m * n - n = -7.5 ∨ m * n - n = -12.5 :=
sorry

end mn_minus_n_values_l281_281839


namespace brittany_age_after_vacation_l281_281322

-- Definitions of the conditions
def rebecca_age : ℕ := 25
def age_difference : ℕ := 3
def vacation_years : ℕ := 4

-- Prove the main statement
theorem brittany_age_after_vacation : rebecca_age + age_difference + vacation_years = 32 := by
  sorry

end brittany_age_after_vacation_l281_281322


namespace probability_at_least_one_two_l281_281109

def num_dice := 2
def sides_dice := 8
def total_outcomes := sides_dice ^ num_dice
def num_non_favorable_outcomes := (sides_dice - 1) ^ num_dice
def num_favorable_outcomes := total_outcomes - num_non_favorable_outcomes
def probability_favorable_outcomes := (15 : ℚ) / (64 : ℚ)

theorem probability_at_least_one_two :
  probability_favorable_outcomes = 15 / 64 :=
sorry

end probability_at_least_one_two_l281_281109


namespace equivalent_shaded_areas_l281_281312

/- 
  Definitions and parameters:
  - l_sq: the side length of the larger square.
  - s_sq: the side length of the smaller square.
-/
variables (l_sq s_sq : ℝ)
  
-- The area of the larger square
def area_larger_square : ℝ := l_sq * l_sq
  
-- The area of the smaller square
def area_smaller_square : ℝ := s_sq * s_sq
  
-- The shaded area in diagram i
def shaded_area_diagram_i : ℝ := area_larger_square l_sq - area_smaller_square s_sq

-- The polygonal areas in diagrams ii and iii
variables (polygon_area_ii polygon_area_iii : ℝ)

-- The theorem to prove the equivalence of the areas
theorem equivalent_shaded_areas :
  polygon_area_ii = shaded_area_diagram_i l_sq s_sq ∧ polygon_area_iii = shaded_area_diagram_i l_sq s_sq :=
sorry

end equivalent_shaded_areas_l281_281312


namespace triangle_with_incircle_radius_one_has_sides_5_4_3_l281_281034

variable {a b c : ℕ} (h1 : a ≥ b ∧ b ≥ c)
variable (h2 : ∃ (a b c : ℕ), (a + b + c) / 2 * 1 = (a + b + c) / 2 * ((a + b + c) / 2 - a) * ((a + b + c) / 2 - b) * ((a + b + c) / 2 - c))

theorem triangle_with_incircle_radius_one_has_sides_5_4_3 :
  a = 5 ∧ b = 4 ∧ c = 3 :=
by
    sorry

end triangle_with_incircle_radius_one_has_sides_5_4_3_l281_281034


namespace sum_of_integers_l281_281084

theorem sum_of_integers (x y : ℕ) (h1 : x > y) (h2 : x - y = 15) (h3 : x * y = 56) : x + y = Real.sqrt 449 :=
by
  sorry

end sum_of_integers_l281_281084


namespace find_a1_l281_281197

-- Define the sequence
def seq (a : ℕ → ℝ) := ∀ n : ℕ, 0 < n → a n = (1/2) * a (n + 1)

-- Given conditions
def a3_value (a : ℕ → ℝ) := a 3 = 12

-- Theorem statement
theorem find_a1 (a : ℕ → ℝ) (h_seq : seq a) (h_a3 : a3_value a) : a 1 = 3 :=
by
  sorry

end find_a1_l281_281197


namespace power_function_even_l281_281349

-- Define the function and its properties
def f (x : ℝ) (α : ℤ) : ℝ := x ^ (Int.toNat α)

-- State the theorem with given conditions
theorem power_function_even (α : ℤ) 
    (h : f 1 α ^ 2 + f (-1) α ^ 2 = 2 * (f 1 α + f (-1) α - 1)) : 
    ∀ x : ℝ, f x α = f (-x) α :=
by
  sorry

end power_function_even_l281_281349


namespace distance_at_1_5_l281_281395

def total_distance : ℝ := 174
def speed : ℝ := 60
def travel_time (x : ℝ) : ℝ := total_distance - speed * x

theorem distance_at_1_5 :
  travel_time 1.5 = 84 := by
  sorry

end distance_at_1_5_l281_281395


namespace jellybeans_to_buy_l281_281635

-- Define the conditions: a minimum of 150 jellybeans and a remainder of 15 when divided by 17.
def condition (n : ℕ) : Prop :=
  n ≥ 150 ∧ n % 17 = 15

-- Define the main statement to prove: if condition holds, then n is 151
theorem jellybeans_to_buy (n : ℕ) (h : condition n) : n = 151 :=
by
  -- Proof is skipped with sorry
  sorry

end jellybeans_to_buy_l281_281635


namespace midpoint_AM_l281_281338

noncomputable def midpoint (C B : Point) : Point := 
  Point.mk ((C.x + B.x) / 2) ((C.y + B.y) / 2)

theorem midpoint_AM (k1 k2 : Circle) (C A B M : Point) 
  (hC_outside_k1 : ¬(k1.contains C))
  (hTangent_CA : k1.isTangent (Line.mk C A))
  (hTangent_CB : k1.isTangent (Line.mk C B))
  (hTangent_k2 : k2.isTangentWith k1 {seg := Segment.mk A B, point := B})
  (hPasses_C : k2.contains C)
  (hIntersect_k1_k2 : k1.intersect k2 = {M}) :
  lies_on (Line.mk A M) (midpoint C B) :=
sorry

end midpoint_AM_l281_281338


namespace triplets_of_positive_integers_l281_281328

/-- We want to determine all positive integer triplets (a, b, c) such that
    ab - c, bc - a, and ca - b are all powers of 2.
    A power of 2 is an integer of the form 2^n, where n is a non-negative integer.-/
theorem triplets_of_positive_integers (a b c : ℕ) (h1: 0 < a) (h2: 0 < b) (h3: 0 < c) :
  ((∃ k1 : ℕ, ab - c = 2^k1) ∧ (∃ k2 : ℕ, bc - a = 2^k2) ∧ (∃ k3 : ℕ, ca - b = 2^k3))
  ↔ (a = 2 ∧ b = 2 ∧ c = 2) ∨ (a = 3 ∧ b = 2 ∧ c = 2) ∨ (a = 2 ∧ b = 6 ∧ c = 11) ∨ (a = 3 ∧ b = 5 ∧ c = 7) :=
sorry

end triplets_of_positive_integers_l281_281328


namespace smallest_possible_value_l281_281799

theorem smallest_possible_value (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (hxy : x ≠ y) (h : (1 / (x : ℝ)) + (1 / (y : ℝ)) = 1 / 15) : x + y = 64 :=
sorry

end smallest_possible_value_l281_281799


namespace cookie_sales_l281_281454

theorem cookie_sales (n M A : ℕ) 
  (hM : M = n - 9)
  (hA : A = n - 2)
  (h_sum : M + A < n)
  (hM_positive : M ≥ 1)
  (hA_positive : A ≥ 1) : 
  n = 10 := 
sorry

end cookie_sales_l281_281454


namespace trig_identity_l281_281790

theorem trig_identity (θ : ℝ) (h : Real.tan θ = 2) : 
  Real.sin θ ^ 2 + Real.sin θ * Real.cos θ - 2 * Real.cos θ ^ 2 = 4 / 5 :=
by 
  sorry

end trig_identity_l281_281790


namespace smallest_possible_value_l281_281800

theorem smallest_possible_value (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (hxy : x ≠ y) (h : (1 / (x : ℝ)) + (1 / (y : ℝ)) = 1 / 15) : x + y = 64 :=
sorry

end smallest_possible_value_l281_281800


namespace exists_product_sum_20000_l281_281278

theorem exists_product_sum_20000 :
  ∃ (k m : ℕ), 1 ≤ k ∧ k ≤ 999 ∧ 1 ≤ m ∧ m ≤ 999 ∧ k * (k + 1) + m * (m + 1) = 20000 :=
by 
  sorry

end exists_product_sum_20000_l281_281278


namespace smallest_nonfactor_product_of_factors_of_48_l281_281426

theorem smallest_nonfactor_product_of_factors_of_48 :
  ∃ a b : ℕ, a ≠ b ∧ a ∣ 48 ∧ b ∣ 48 ∧ ¬ (a * b ∣ 48) ∧ (∀ c d : ℕ, c ≠ d ∧ c ∣ 48 ∧ d ∣ 48 ∧ ¬ (c * d ∣ 48) → a * b ≤ c * d) ∧ a * b = 18 :=
sorry

end smallest_nonfactor_product_of_factors_of_48_l281_281426


namespace father_payment_l281_281449

variable (x y : ℤ)

theorem father_payment :
  5 * x - 3 * y = 24 :=
sorry

end father_payment_l281_281449


namespace symmetric_point_l281_281262

-- Definitions
def P : ℝ × ℝ := (5, -2)
def line (x y : ℝ) : Prop := x - y + 5 = 0

-- Statement 
theorem symmetric_point (a b : ℝ) 
  (symmetric_condition1 : ∀ x y, line x y → (b + 2)/(a - 5) * 1 = -1)
  (symmetric_condition2 : ∀ x y, line x y → (a + 5)/2 - (b - 2)/2 + 5 = 0) :
  (a, b) = (-7, 10) :=
sorry

end symmetric_point_l281_281262


namespace sugar_spilled_l281_281077

-- Define the initial amount of sugar and the amount left
def initial_sugar : ℝ := 9.8
def remaining_sugar : ℝ := 4.6

-- State the problem as a theorem
theorem sugar_spilled :
  initial_sugar - remaining_sugar = 5.2 := 
sorry

end sugar_spilled_l281_281077


namespace simplify_fraction_l281_281404

theorem simplify_fraction : (3 / 462 + 17 / 42) = 95 / 231 :=
by 
  sorry

end simplify_fraction_l281_281404


namespace smallest_x_plus_y_l281_281795

theorem smallest_x_plus_y (x y : ℕ) (h_xy_not_equal : x ≠ y) (h_condition : (1 : ℚ) / x + 1 / y = 1 / 15) :
  x + y = 64 :=
sorry

end smallest_x_plus_y_l281_281795


namespace three_digit_number_division_l281_281844

theorem three_digit_number_division :
  ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ (∃ m : ℕ, 10 ≤ m ∧ m < 100 ∧ n / m = 8 ∧ n % m = 6) → n = 342 :=
by
  sorry

end three_digit_number_division_l281_281844


namespace volume_increased_by_3_l281_281626

theorem volume_increased_by_3 {l w h : ℝ}
  (h1 : l * w * h = 5000)
  (h2 : l * w + w * h + l * h = 925)
  (h3 : l + w + h = 60) :
  (l + 3) * (w + 3) * (h + 3) = 8342 := 
by
  sorry

end volume_increased_by_3_l281_281626


namespace sin_theta_of_triangle_l281_281307

theorem sin_theta_of_triangle (area : ℝ) (side : ℝ) (median : ℝ) (θ : ℝ)
  (h_area : area = 30)
  (h_side : side = 10)
  (h_median : median = 9) :
  Real.sin θ = 2 / 3 := by
  sorry

end sin_theta_of_triangle_l281_281307


namespace sin_half_alpha_l281_281169

variable (α : ℝ)
variable (h₀ : 0 < α ∧ α < π/2)
variable (h₁ : cos α = (1 + sqrt 5) / 4)

theorem sin_half_alpha : sin (α / 2) = (sqrt 5 - 1) / 4 :=
by
  sorry

end sin_half_alpha_l281_281169


namespace li_to_zhang_l281_281386

theorem li_to_zhang :
  (∀ (meter chi : ℕ), 3 * meter = chi) →
  (∀ (zhang chi : ℕ), 10 * zhang = chi) →
  (∀ (kilometer li : ℕ), 2 * li = kilometer) →
  (1 * lin = 150 * zhang) :=
by
  intro h_meter h_zhang h_kilometer
  sorry

end li_to_zhang_l281_281386


namespace find_coordinates_of_P_l281_281242

theorem find_coordinates_of_P (P : ℝ × ℝ) (hx : abs P.2 = 5) (hy : abs P.1 = 3) (hq : P.1 < 0 ∧ P.2 > 0) : 
  P = (-3, 5) := 
  sorry

end find_coordinates_of_P_l281_281242


namespace problem_statement_l281_281736

theorem problem_statement : 6 * (3/2 + 2/3) = 13 :=
by
  sorry

end problem_statement_l281_281736


namespace birds_initially_l281_281098

-- Definitions of the conditions
def initial_birds (B : Nat) := B
def initial_storks := 4
def additional_storks := 6
def total := 13

-- The theorem we need to prove
theorem birds_initially (B : Nat) (h : initial_birds B + initial_storks + additional_storks = total) : initial_birds B = 3 :=
by
  -- The proof can go here
  sorry

end birds_initially_l281_281098


namespace arithmetic_sequence_value_l281_281043

variable {a : ℕ → ℝ}

theorem arithmetic_sequence_value (h : a 3 + a 5 + a 11 + a 13 = 80) : a 8 = 20 :=
sorry

end arithmetic_sequence_value_l281_281043


namespace sin_half_angle_l281_281172

theorem sin_half_angle (α : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : Real.cos α = (1 + Real.sqrt 5) / 4) : 
  Real.sin (α / 2) = (Real.sqrt 5 - 1) / 4 := 
by 
  sorry

end sin_half_angle_l281_281172


namespace sum_of_row_10_pascals_triangle_sum_of_squares_of_row_10_pascals_triangle_l281_281974

def row_10_pascals_triangle : List ℕ := [1, 10, 45, 120, 210, 252, 210, 120, 45, 10, 1]

theorem sum_of_row_10_pascals_triangle :
  (List.sum row_10_pascals_triangle) = 1024 := by
  sorry

theorem sum_of_squares_of_row_10_pascals_triangle :
  (List.sum (List.map (fun x => x * x) row_10_pascals_triangle)) = 183756 := by
  sorry

end sum_of_row_10_pascals_triangle_sum_of_squares_of_row_10_pascals_triangle_l281_281974


namespace house_number_digits_cost_l281_281835

/-
The constants represent:
- cost_1: the cost of 1 unit (1000 rubles)
- cost_12: the cost of 12 units (2000 rubles)
- cost_512: the cost of 512 units (3000 rubles)
- P: the cost per digit of a house number (1000 rubles)
- n: the number of digits in a house number
- The goal is to prove that the cost for 1, 12, and 512 units follows the pattern described
-/

theorem house_number_digits_cost :
  ∃ (P : ℕ),
    (P = 1000) ∧
    (∃ (cost_1 cost_12 cost_512 : ℕ),
      cost_1 = 1000 ∧
      cost_12 = 2000 ∧
      cost_512 = 3000 ∧
      (∃ n1 n2 n3 : ℕ,
        n1 = 1 ∧
        n2 = 2 ∧
        n3 = 3 ∧
        cost_1 = P * n1 ∧
        cost_12 = P * n2 ∧
        cost_512 = P * n3)) :=
by
  sorry

end house_number_digits_cost_l281_281835


namespace complement_intersection_l281_281199

open Set

-- Definitions of sets
def U : Set ℕ := {2, 3, 4, 5, 6}
def A : Set ℕ := {2, 5, 6}
def B : Set ℕ := {3, 5}

-- The theorem statement
theorem complement_intersection :
  (U \ B) ∩ A = {2, 6} := by
  sorry

end complement_intersection_l281_281199


namespace find_number_l281_281540

theorem find_number (x n : ℝ) (h1 : x > 0) (h2 : x / 50 + x / n = 0.06 * x) : n = 25 :=
by
  sorry

end find_number_l281_281540


namespace student_missed_number_l281_281451

theorem student_missed_number (student_sum : ℕ) (n : ℕ) (actual_sum : ℕ) : 
  student_sum = 575 → 
  actual_sum = n * (n + 1) / 2 → 
  n = 34 → 
  actual_sum - student_sum = 20 := 
by 
  sorry

end student_missed_number_l281_281451


namespace find_kids_l281_281484

theorem find_kids (A K : ℕ) (h1 : A + K = 12) (h2 : 3 * A = 15) : K = 7 :=
sorry

end find_kids_l281_281484


namespace num_exclusive_multiples_4_6_less_151_l281_281049

def numMultiplesExclusive (n : ℕ) (a b : ℕ) : ℕ :=
  let lcm_ab := Nat.lcm a b
  (n-1) / a - (n-1) / lcm_ab + (n-1) / b - (n-1) / lcm_ab

theorem num_exclusive_multiples_4_6_less_151 : 
  numMultiplesExclusive 151 4 6 = 38 := 
by 
  sorry

end num_exclusive_multiples_4_6_less_151_l281_281049


namespace equal_focal_distances_condition_l281_281885

theorem equal_focal_distances_condition (k : ℝ) : 
  (∀ x y : ℝ, (9*x^2 + 25*y^2 = 225) → 
              (∀ x y : ℝ, (x^2/(16-k) - y^2/k = 1) → 
              (2*sqrt(25 - 9) = 2*sqrt(16)))) ↔ 0 < k ∧ k < 16 :=
by
  sorry

end equal_focal_distances_condition_l281_281885


namespace part1_part2_l281_281198

def A (x : ℤ) := ∃ m n : ℤ, x = m^2 - n^2
def B (x : ℤ) := ∃ k : ℤ, x = 2 * k + 1

theorem part1 (h1: A 8) (h2: A 9) (h3: ¬ A 10) : 
  (A 8) ∧ (A 9) ∧ (¬ A 10) :=
by {
  sorry
}

theorem part2 (x : ℤ) (h : A x) : B x :=
by {
  sorry
}

end part1_part2_l281_281198


namespace sin_half_alpha_l281_281174

noncomputable def alpha : ℝ := sorry
def is_acute (alpha : ℝ) : Prop := 0 < alpha ∧ alpha < π / 2

axiom acos_alpha : cos alpha = (1 + sqrt 5) / 4
axiom acute_alpha : is_acute alpha

theorem sin_half_alpha : sin (alpha / 2) = (sqrt 5 - 1) / 4 :=
by
  sorry

end sin_half_alpha_l281_281174


namespace number_of_spiders_l281_281063

theorem number_of_spiders (total_legs : ℕ) (legs_per_spider : ℕ) (h1 : total_legs = 32) (h2 : legs_per_spider = 8) :
  total_legs / legs_per_spider = 4 := by
  sorry

end number_of_spiders_l281_281063


namespace hyperbola_s_eq_l281_281458

theorem hyperbola_s_eq (s : ℝ) 
  (hyp1 : ∃ b > 0, ∀ x y : ℝ, (x, y) = (5, -3) → x^2 / 9 - y^2 / b^2 = 1) 
  (hyp2 : ∃ b > 0, ∀ x y : ℝ, (x, y) = (3, 0) → x^2 / 9 - y^2 / b^2 = 1) 
  (hyp3 : ∃ b > 0, ∀ x y : ℝ, (x, y) = (s, -1) → x^2 / 9 - y^2 / b^2 = 1) :
  s^2 = 873 / 81 :=
sorry

end hyperbola_s_eq_l281_281458


namespace smallest_x_plus_y_l281_281797

theorem smallest_x_plus_y (x y : ℕ) (h_xy_not_equal : x ≠ y) (h_condition : (1 : ℚ) / x + 1 / y = 1 / 15) :
  x + y = 64 :=
sorry

end smallest_x_plus_y_l281_281797


namespace initial_ratio_of_milk_to_water_l281_281976

theorem initial_ratio_of_milk_to_water 
  (M W : ℕ) 
  (h1 : M + 10 + W = 30)
  (h2 : (M + 10) * 2 = W * 5)
  (h3 : M + W = 20) : 
  M = 11 ∧ W = 9 := 
by 
  sorry

end initial_ratio_of_milk_to_water_l281_281976


namespace time_after_3250_minutes_final_answer_is_A_l281_281117

open Nat

/-- 
  What time is it 3250 minutes after 3:00 AM on January 1, 2020?
-/
noncomputable def minutes_from (start : posix_time) (minutes : ℕ) : posix_time :=
  posix_time.add_seconds start (minutes * 60)

/-- 
  The start time is January 1, 2020 at 3:00 AM 
-/
def start_time : posix_time := 
  posix_time.mk 1577865600 -- Timestamp for January 1, 2020 00:00:00 UTC 
  + 3 * 60 * 60           -- Adding 3 hours in seconds

/-- 
  The expected time after 3250 minutes
-/
def expected_time : posix_time := 
  start_time + (3250 * 60) -- Converting 3250 minutes to seconds

/--
  Prove that 3250 minutes after 3:00 AM on January 1, 2020 is January 3, 2020 at 9:10 AM.
-/
theorem time_after_3250_minutes : minutes_from start_time 3250 = expected_time :=
by
  sorry

theorem final_answer_is_A : 
  minutes_from start_time 3250 == expected_time :=
by
  rw [time_after_3250_minutes]
  sorry

end time_after_3250_minutes_final_answer_is_A_l281_281117


namespace solve_equations_l281_281246

theorem solve_equations :
  (∃ x1 x2 : ℝ, x1 = 2 + Real.sqrt 5 ∧ x2 = 2 - Real.sqrt 5 ∧ (x1^2 - 4 * x1 - 1 = 0) ∧ (x2^2 - 4 * x2 - 1 = 0)) ∧
  (∃ y1 y2 : ℝ, y1 = -4 ∧ y2 = 1 ∧ ((y1 + 4)^2 = 5 * (y1 + 4)) ∧ ((y2 + 4)^2 = 5 * (y2 + 4))) :=
by
  sorry

end solve_equations_l281_281246


namespace roots_negative_reciprocal_condition_l281_281645

theorem roots_negative_reciprocal_condition (a b c : ℝ) (h : a ≠ 0) :
  (∃ r s : ℝ, r ≠ 0 ∧ s ≠ 0 ∧ r * s = -1 ∧ a * r^2 + b * r + c = 0 ∧ a * s^2 + b * s + c = 0) → c = -a :=
by
  sorry

end roots_negative_reciprocal_condition_l281_281645


namespace smallest_product_not_factor_of_48_l281_281437

theorem smallest_product_not_factor_of_48 (a b : ℕ) (h1 : a ≠ b) (h2 : a ∣ 48) (h3 : b ∣ 48) (h4 : ¬ (a * b ∣ 48)) : a * b = 32 :=
sorry

end smallest_product_not_factor_of_48_l281_281437


namespace minimum_flowers_to_guarantee_bouquets_l281_281218

-- Definitions based on conditions given
def types_of_flowers : ℕ := 6
def flowers_needed_for_bouquet : ℕ := 5
def required_bouquets : ℕ := 10

-- Problem statement in Lean 4
theorem minimum_flowers_to_guarantee_bouquets (types : ℕ) (needed: ℕ) (bouquets: ℕ) 
    (h_types: types = types_of_flowers) (h_needed: needed = flowers_needed_for_bouquet) 
    (h_bouquets: bouquets = required_bouquets) : 
    (minimum_number_of_flowers_to_guarantee_bouquets types needed bouquets) = 70 :=
by sorry


end minimum_flowers_to_guarantee_bouquets_l281_281218


namespace choir_blonde_black_ratio_l281_281378

theorem choir_blonde_black_ratio 
  (b x : ℕ) 
  (h1 : ∀ (b x : ℕ), b / ((5 / 3 : ℚ) * b) = (3 / 5 : ℚ)) 
  (h2 : ∀ (b x : ℕ), (b + x) / ((5 / 3 : ℚ) * b) = (3 / 2 : ℚ)) :
  x = (3 / 2 : ℚ) * b ∧ 
  ∃ k : ℚ, k = (5 / 3 : ℚ) * b :=
by {
  sorry
}

end choir_blonde_black_ratio_l281_281378


namespace find_n_l281_281214

theorem find_n (x n : ℝ) (h1 : ((x / n) * 5) + 10 - 12 = 48) (h2 : x = 40) : n = 4 :=
sorry

end find_n_l281_281214


namespace count_paths_word_l281_281360

def move_right_or_down_paths (n : ℕ) : ℕ := 2^n

theorem count_paths_word (n : ℕ) (w : String) (start : Char) (end_ : Char) :
    w = "строка" ∧ start = 'C' ∧ end_ = 'A' ∧ n = 5 →
    move_right_or_down_paths n = 32 :=
by
  intro h
  cases h
  sorry

end count_paths_word_l281_281360


namespace has_exactly_one_zero_point_l281_281346

noncomputable def f (x a b : ℝ) : ℝ := (x - 1) * Real.exp x - a * x^2 + b

theorem has_exactly_one_zero_point
  (a b : ℝ) 
  (h1 : (1/2 < a ∧ a ≤ Real.exp 2 / 2 ∧ b > 2 * a) ∨ (0 < a ∧ a < 1/2 ∧ b ≤ 2 * a)) :
  ∃! x : ℝ, f x a b = 0 := 
sorry

end has_exactly_one_zero_point_l281_281346


namespace number_of_bags_proof_l281_281738

def total_flight_time_hours : ℕ := 2
def minutes_per_hour : ℕ := 60
def total_minutes := total_flight_time_hours * minutes_per_hour

def peanuts_per_minute : ℕ := 1
def total_peanuts_eaten := total_minutes * peanuts_per_minute

def peanuts_per_bag : ℕ := 30
def number_of_bags : ℕ := total_peanuts_eaten / peanuts_per_bag

theorem number_of_bags_proof : number_of_bags = 4 := by
  -- proof goes here
  sorry

end number_of_bags_proof_l281_281738


namespace martha_meeting_distance_l281_281711

theorem martha_meeting_distance (t : ℝ) (d : ℝ)
  (h1 : 0 < t)
  (h2 : d = 45 * (t + 0.75))
  (h3 : d - 45 = 55 * (t - 1)) :
  d = 230.625 := 
  sorry

end martha_meeting_distance_l281_281711


namespace students_remaining_l281_281630

theorem students_remaining (n : ℕ) (h1 : n = 1000)
  (h_beach : n / 2 = 500)
  (h_home : (n - n / 2) / 2 = 250) :
  n - (n / 2 + (n - n / 2) / 2) = 250 :=
by
  sorry

end students_remaining_l281_281630


namespace lines_in_plane_l281_281696

  -- Define the necessary objects in Lean
  structure Line (α : Type) := (equation : α → α → Prop)

  def same_plane (l1 l2 : Line ℝ) : Prop := 
  -- Here you can define what it means for l1 and l2 to be in the same plane.
  sorry

  def intersect (l1 l2 : Line ℝ) : Prop := 
  -- Define what it means for two lines to intersect.
  sorry

  def parallel (l1 l2 : Line ℝ) : Prop := 
  -- Define what it means for two lines to be parallel.
  sorry

  theorem lines_in_plane (l1 l2 : Line ℝ) (h : same_plane l1 l2) : 
    (intersect l1 l2) ∨ (parallel l1 l2) := 
  by 
      sorry
  
end lines_in_plane_l281_281696


namespace books_total_l281_281981

theorem books_total (J T : ℕ) (hJ : J = 10) (hT : T = 38) : J + T = 48 :=
by {
  sorry
}

end books_total_l281_281981


namespace find_x_when_y_neg4_l281_281247

variable {x y : ℝ}
variable (k : ℝ)

-- Condition: x is inversely proportional to y
def inversely_proportional (x y : ℝ) (k : ℝ) : Prop :=
  x * y = k

theorem find_x_when_y_neg4 (h : inversely_proportional 5 10 50) :
  inversely_proportional x (-4) 50 → x = -25 / 2 :=
by sorry

end find_x_when_y_neg4_l281_281247


namespace smallest_sum_of_inverses_l281_281815

theorem smallest_sum_of_inverses 
  (x y : ℕ) (hx : x ≠ y) (h1 : 0 < x) (h2 : 0 < y) (h_condition : (1 / x : ℚ) + 1 / y = 1 / 15) :
  x + y = 64 := 
sorry

end smallest_sum_of_inverses_l281_281815


namespace razors_blades_equation_l281_281468

/-- Given the number of razors sold x,
each razor sold brings a profit of 30 yuan,
each blade sold incurs a loss of 0.5 yuan,
the number of blades sold is twice the number of razors sold,
and the total profit from these two products is 5800 yuan,
prove that the linear equation is -0.5 * 2 * x + 30 * x = 5800 -/
theorem razors_blades_equation (x : ℝ) :
  -0.5 * 2 * x + 30 * x = 5800 := 
sorry

end razors_blades_equation_l281_281468


namespace problem_statement_l281_281118

def contrapositive {P Q : Prop} (h : P → Q) : ¬Q → ¬P :=
by sorry

def sufficient_but_not_necessary (P Q : Prop) : (P → Q) ∧ ¬(Q → P) :=
by sorry

def proposition_C (p q : Prop) : ¬(p ∧ q) → (¬p ∨ ¬q) :=
by sorry

def negate_exists (P : ℝ → Prop) : (∃ x : ℝ, P x) → ¬(∀ x : ℝ, ¬P x) :=
by sorry

theorem problem_statement : 
¬ (∀ (P Q : Prop), ¬(P ∧ Q) → (¬P ∨ ¬Q)) :=
by sorry

end problem_statement_l281_281118


namespace arrangement_two_rows_arrangement_no_head_tail_arrangement_girls_together_arrangement_no_boys_next_l281_281418

theorem arrangement_two_rows :
  ∃ (ways : ℕ), ways = 5040 := by
  sorry

theorem arrangement_no_head_tail (A : ℕ):
  ∃ (ways : ℕ), ways = 3600 := by
  sorry

theorem arrangement_girls_together :
  ∃ (ways : ℕ), ways = 576 := by
  sorry

theorem arrangement_no_boys_next :
  ∃ (ways : ℕ), ways = 1440 := by
  sorry

end arrangement_two_rows_arrangement_no_head_tail_arrangement_girls_together_arrangement_no_boys_next_l281_281418


namespace back_seat_people_l281_281543

-- Define the problem conditions

def leftSideSeats : ℕ := 15
def seatDifference : ℕ := 3
def peoplePerSeat : ℕ := 3
def totalBusCapacity : ℕ := 88

-- Define the formula for calculating the people at the back seat
def peopleAtBackSeat := 
  totalBusCapacity - ((leftSideSeats * peoplePerSeat) + ((leftSideSeats - seatDifference) * peoplePerSeat))

-- The statement we need to prove
theorem back_seat_people : peopleAtBackSeat = 7 :=
by
  sorry

end back_seat_people_l281_281543


namespace rectangle_aspect_ratio_l281_281597

theorem rectangle_aspect_ratio (x y : ℝ) (h : x > 0 ∧ y > 0 ∧ x / y = 2 * y / x) : x / y = Real.sqrt 2 :=
by
  sorry

end rectangle_aspect_ratio_l281_281597


namespace bad_games_count_l281_281715

/-- 
  Oliver bought a total of 11 video games, and 6 of them worked.
  Prove that the number of bad games he bought is 5.
-/
theorem bad_games_count (total_games : ℕ) (working_games : ℕ) (h1 : total_games = 11) (h2 : working_games = 6) : total_games - working_games = 5 :=
by
  sorry

end bad_games_count_l281_281715


namespace find_sum_of_xyz_l281_281674

theorem find_sum_of_xyz : ∃ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ 
  (151 / 44 : ℚ) = 3 + 1 / (x + 1 / (y + 1 / z)) ∧ x + y + z = 11 :=
by 
  sorry

end find_sum_of_xyz_l281_281674


namespace inequality_problems_l281_281039

theorem inequality_problems
  (m n l : ℝ)
  (h1 : m > n)
  (h2 : n > l) :
  (m + 1/m > n + 1/n) ∧ (m + 1/n > n + 1/m) :=
by
  sorry

end inequality_problems_l281_281039


namespace circle_through_three_points_l281_281334

open Real

structure Point where
  x : ℝ
  y : ℝ

def circle_equation (D E F : ℝ) (P : Point) : Prop :=
  P.x^2 + P.y^2 + D * P.x + E * P.y + F = 0

theorem circle_through_three_points :
  ∃ (D E F : ℝ), 
    (circle_equation D E F ⟨1, 12⟩) ∧ 
    (circle_equation D E F ⟨7, 10⟩) ∧ 
    (circle_equation D E F ⟨-9, 2⟩) ∧
    (D = -2) ∧ (E = -4) ∧ (F = -95) :=
by
  sorry

end circle_through_three_points_l281_281334


namespace order_of_four_l281_281949

theorem order_of_four {m n p q : ℝ} (hmn : m < n) (hpq : p < q) (h1 : (p - m) * (p - n) < 0) (h2 : (q - m) * (q - n) < 0) : m < p ∧ p < q ∧ q < n :=
by
  sorry

end order_of_four_l281_281949


namespace impossible_sequence_l281_281647

theorem impossible_sequence (α : ℝ) (hα : 0 < α ∧ α < 1) (a : ℕ → ℝ) (ha : ∀ n, 0 < a n) :
  (∀ n, 1 + a (n + 1) ≤ a n + (α / n) * a n) → false :=
by
  sorry

end impossible_sequence_l281_281647


namespace sum_of_transformed_roots_l281_281054

theorem sum_of_transformed_roots (α β γ : ℂ) (h₁ : α^3 - α + 1 = 0) (h₂ : β^3 - β + 1 = 0) (h₃ : γ^3 - γ + 1 = 0) :
  (1 - α) / (1 + α) + (1 - β) / (1 + β) + (1 - γ) / (1 + γ) = 1 :=
by
  sorry

end sum_of_transformed_roots_l281_281054


namespace alex_new_salary_in_may_l281_281309

def initial_salary : ℝ := 50000
def february_increase (s : ℝ) : ℝ := s * 1.10
def april_bonus (s : ℝ) : ℝ := s + 2000
def may_pay_cut (s : ℝ) : ℝ := s * 0.95

theorem alex_new_salary_in_may : may_pay_cut (april_bonus (february_increase initial_salary)) = 54150 :=
by
  sorry

end alex_new_salary_in_may_l281_281309


namespace combined_rent_C_D_l281_281134

theorem combined_rent_C_D :
  let rent_per_month_area_z := 100
  let rent_per_month_area_w := 120
  let months_c := 3
  let months_d := 6
  let rent_c := months_c * rent_per_month_area_z
  let rent_d := months_d * rent_per_month_area_w
  let combined_rent := rent_c + rent_d
  combined_rent = 1020 :=
by
  let rent_per_month_area_z := 100
  let rent_per_month_area_w := 120
  let months_c := 3
  let months_d := 6
  let rent_c := months_c * rent_per_month_area_z
  let rent_d := months_d * rent_per_month_area_w
  let combined_rent := rent_c + rent_d
  show combined_rent = 1020
  sorry

end combined_rent_C_D_l281_281134


namespace mitya_age_l281_281991

theorem mitya_age {M S: ℕ} (h1 : M = S + 11) (h2 : S = 2 * (S - (M - S))) : M = 33 :=
by
  -- proof steps skipped
  sorry

end mitya_age_l281_281991


namespace true_proposition_l281_281161

open Real

-- Proposition p
def p : Prop := ∀ x > 0, log x + 4 * x ≥ 3

-- Proposition q
def q : Prop := ∃ x > 0, 8 * x + 1 / (2 * x) ≤ 4

theorem true_proposition : ¬ p ∧ q := by
  sorry

end true_proposition_l281_281161


namespace problem_statement_l281_281343

theorem problem_statement (a b c : ℤ) (h : c = b + 2) : 
  (a - (b + c)) - ((a + c) - b) = 0 :=
by
  sorry

end problem_statement_l281_281343


namespace factor_poly_l281_281655

theorem factor_poly : 
  ∃ (a b c d e f : ℤ), a < d ∧ (a*x^2 + b*x + c)*(d*x^2 + e*x + f) = (x^2 + 6*x + 9 - 64*x^4) ∧ 
  (a = -8 ∧ b = 1 ∧ c = 3 ∧ d = 8 ∧ e = 1 ∧ f = 3) := 
sorry

end factor_poly_l281_281655


namespace aquarium_visitors_not_ill_l281_281933

theorem aquarium_visitors_not_ill :
  let visitors_monday := 300
  let visitors_tuesday := 500
  let visitors_wednesday := 400
  let ill_monday := (15 / 100) * visitors_monday
  let ill_tuesday := (30 / 100) * visitors_tuesday
  let ill_wednesday := (20 / 100) * visitors_wednesday
  let not_ill_monday := visitors_monday - ill_monday
  let not_ill_tuesday := visitors_tuesday - ill_tuesday
  let not_ill_wednesday := visitors_wednesday - ill_wednesday
  let total_not_ill := not_ill_monday + not_ill_tuesday + not_ill_wednesday
  total_not_ill = 925 := 
by
  sorry

end aquarium_visitors_not_ill_l281_281933


namespace a_2n_is_square_l281_281706

def a_n (n : ℕ) : ℕ := 
  if n = 0 then 0 
  else if n = 1 then 1
  else if n = 2 then 1
  else if n = 3 then 2
  else a_n (n - 1) + a_n (n - 3) + a_n (n - 4)

theorem a_2n_is_square (n : ℕ) : ∃ k : ℕ, a_n (2 * n) = k * k := by
  sorry

end a_2n_is_square_l281_281706


namespace dot_product_of_a_and_b_l281_281350

-- Define the vectors
def a : ℝ × ℝ := (1, -3)
def b : ℝ × ℝ := (3, 7)

-- Define the dot product function
def dot_product (v₁ v₂ : ℝ × ℝ) : ℝ := v₁.1 * v₂.1 + v₁.2 * v₂.2

-- State the theorem
theorem dot_product_of_a_and_b : dot_product a b = -18 := by
  sorry

end dot_product_of_a_and_b_l281_281350


namespace arrangement_count_correct_l281_281139

def num_arrangements_exactly_two_females_next_to_each_other (males : ℕ) (females : ℕ) : ℕ :=
  if males = 4 ∧ females = 3 then 3600 else 0

theorem arrangement_count_correct :
  num_arrangements_exactly_two_females_next_to_each_other 4 3 = 3600 :=
by
  sorry

end arrangement_count_correct_l281_281139


namespace sin_half_alpha_l281_281165

noncomputable def given_cos_alpha (α : ℝ) : Prop :=
  α ∈ Ioo 0 (π / 2) ∧ cos α = (1 + real.sqrt 5) / 4

theorem sin_half_alpha (α : ℝ) (hα : given_cos_alpha α) :
  sin (α / 2) = (-1 + real.sqrt 5) / 4 :=
sorry

end sin_half_alpha_l281_281165


namespace average_age_of_others_when_youngest_was_born_l281_281011

noncomputable def average_age_when_youngest_was_born (total_people : ℕ) (average_age : ℕ) (youngest_age : ℕ) : ℚ :=
  let total_age := total_people * average_age
  let age_without_youngest := total_age - youngest_age
  age_without_youngest / (total_people - 1)

theorem average_age_of_others_when_youngest_was_born :
  average_age_when_youngest_was_born 7 30 7 = 33.833 :=
by
  sorry

end average_age_of_others_when_youngest_was_born_l281_281011


namespace probability_physics_majors_consecutive_l281_281877

open Nat

theorem probability_physics_majors_consecutive :
  let total_people := 10
  let num_math := 5
  let num_physics := 3
  let num_bio := 2
  let total_permutations := Nat.factorial total_people
  let favorable_outcomes := 10 * (Nat.factorial num_physics)
  let probability := favorable_outcomes / total_permutations
  in probability = (1 / 12) := by
  sorry

end probability_physics_majors_consecutive_l281_281877


namespace password_probability_l281_281642

def isNonNegativeSingleDigit (n : ℕ) : Prop := n ≤ 9

def isOddSingleDigit (n : ℕ) : Prop := isNonNegativeSingleDigit n ∧ n % 2 = 1

def isPositiveSingleDigit (n : ℕ) : Prop := 0 < n ∧ n ≤ 9

def isVowel (c : Char) : Prop := c = 'A' ∨ c = 'E' ∨ c = 'I' ∨ c = 'O' ∨ c = 'U'

-- Probability that an odd single-digit number followed by a vowel and a positive single-digit number
def prob_odd_vowel_positive_digits : ℚ :=
  let prob_first := 5 / 10 -- Probability of odd single-digit number
  let prob_vowel := 5 / 26 -- Probability of vowel
  let prob_last := 9 / 10 -- Probability of positive single-digit number
  prob_first * prob_vowel * prob_last

theorem password_probability :
  prob_odd_vowel_positive_digits = 9 / 104 :=
by
  sorry

end password_probability_l281_281642


namespace taco_truck_revenue_l281_281471

-- Conditions
def price_of_soft_taco : ℕ := 2
def price_of_hard_taco : ℕ := 5
def family_soft_tacos : ℕ := 3
def family_hard_tacos : ℕ := 4
def other_customers : ℕ := 10
def soft_tacos_per_other_customer : ℕ := 2

-- Calculation
def total_soft_tacos : ℕ := family_soft_tacos + other_customers * soft_tacos_per_other_customer
def revenue_from_soft_tacos : ℕ := total_soft_tacos * price_of_soft_taco
def revenue_from_hard_tacos : ℕ := family_hard_tacos * price_of_hard_taco
def total_revenue : ℕ := revenue_from_soft_tacos + revenue_from_hard_tacos

-- The proof problem
theorem taco_truck_revenue : total_revenue = 66 := 
by 
-- The proof should go here
sorry

end taco_truck_revenue_l281_281471


namespace problem_nine_chapters_l281_281846

theorem problem_nine_chapters (x y : ℝ) :
  (x + (1 / 2) * y = 50) →
  (y + (2 / 3) * x = 50) →
  (x + (1 / 2) * y = 50) ∧ (y + (2 / 3) * x = 50) :=
by
  intros h1 h2
  exact ⟨h1, h2⟩

end problem_nine_chapters_l281_281846


namespace gear_angular_speeds_ratio_l281_281667

noncomputable def gear_ratio (x y z w : ℕ) (ω_A ω_B ω_C ω_D : ℝ) :=
  x * ω_A = y * ω_B ∧ y * ω_B = z * ω_C ∧ z * ω_C = w * ω_D

theorem gear_angular_speeds_ratio (x y z w : ℕ) (ω_A ω_B ω_C ω_D : ℝ) 
  (h : gear_ratio x y z w ω_A ω_B ω_C ω_D) :
  ω_A / ω_B = y / x ∧ ω_B / ω_C = z / y ∧ ω_C / ω_D = w / z :=
by sorry

end gear_angular_speeds_ratio_l281_281667


namespace coordinate_plane_points_l281_281146

theorem coordinate_plane_points (x y : ℝ) :
    4 * x^2 * y^2 = 4 * x * y + 3 ↔ (x * y = 3 / 2 ∨ x * y = -1 / 2) :=
by 
  sorry

end coordinate_plane_points_l281_281146


namespace find_x_l281_281154

noncomputable section

open Real

theorem find_x (x : ℝ) (hx : 0 < x ∧ x < 180) : 
  tan (120 * π / 180 - x * π / 180) = (sin (120 * π / 180) - sin (x * π / 180)) / (cos (120 * π / 180) - cos (x * π / 180)) →
  x = 100 :=
by
  sorry

end find_x_l281_281154


namespace sin_half_alpha_l281_281167

noncomputable def given_cos_alpha (α : ℝ) : Prop :=
  α ∈ Ioo 0 (π / 2) ∧ cos α = (1 + real.sqrt 5) / 4

theorem sin_half_alpha (α : ℝ) (hα : given_cos_alpha α) :
  sin (α / 2) = (-1 + real.sqrt 5) / 4 :=
sorry

end sin_half_alpha_l281_281167


namespace unique_zero_point_mn_l281_281211

noncomputable def f (a : ℝ) (x : ℝ) := a * (x^2 + 2 / x) - Real.log x

theorem unique_zero_point_mn (a : ℝ) (m n x₀ : ℝ) (hmn : m + 1 = n) (a_pos : 0 < a) (f_zero : f a x₀ = 0) (x0_in_range : m < x₀ ∧ x₀ < n) : m + n = 5 := by
  sorry

end unique_zero_point_mn_l281_281211


namespace inequality_proof_l281_281998

theorem inequality_proof (x y z : ℝ) (h1 : x ≥ y) (h2 : y ≥ z) (h3 : z ≥ 0) : 
  (x*y^2/z + y*z^2/x + z*x^2/y) ≥ (x^2 + y^2 + z^2) := by
  sorry

end inequality_proof_l281_281998


namespace problem_1_problem_3_problem_4_l281_281069

-- Definition of the function f(x)
def f (x : ℝ) (b c : ℝ) : ℝ := (|x| * x) + (b * x) + c

-- Prove that when b > 0, f(x) is monotonically increasing on ℝ
theorem problem_1 (b c : ℝ) (h : b > 0) : 
  ∀ x y : ℝ, x < y → f x b c < f y b c :=
sorry

-- Prove that the graph of f(x) is symmetric about the point (0, c) when b = 0
theorem problem_3 (b c : ℝ) (h : b = 0) :
  ∀ x : ℝ, f x b c = f (-x) b c :=
sorry

-- Prove that when b < 0, f(x) = 0 can have three real roots
theorem problem_4 (b c : ℝ) (h : b < 0) :
  ∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧ f x₁ b c = 0 ∧ f x₂ b c = 0 ∧ f x₃ b c = 0 :=
sorry

end problem_1_problem_3_problem_4_l281_281069


namespace stock_comparison_l281_281920

-- Quantities of the first year depreciation or growth rates
def initial_investment : ℝ := 200.0
def dd_first_year_growth : ℝ := 1.10
def ee_first_year_decline : ℝ := 0.85
def ff_first_year_growth : ℝ := 1.05

-- Quantities of the second year depreciation or growth rates
def dd_second_year_growth : ℝ := 1.05
def ee_second_year_growth : ℝ := 1.15
def ff_second_year_decline : ℝ := 0.90

-- Mathematical expression to determine final values after first year
def dd_after_first_year := initial_investment * dd_first_year_growth
def ee_after_first_year := initial_investment * ee_first_year_decline
def ff_after_first_year := initial_investment * ff_first_year_growth

-- Mathematical expression to determine final values after second year
def dd_final := dd_after_first_year * dd_second_year_growth
def ee_final := ee_after_first_year * ee_second_year_growth
def ff_final := ff_after_first_year * ff_second_year_decline

-- Theorem representing the final comparison
theorem stock_comparison : ff_final < ee_final ∧ ee_final < dd_final :=
by {
  -- Here we would provide the proof, but as per instruction we'll place sorry
  sorry
}

end stock_comparison_l281_281920


namespace probability_at_least_one_two_l281_281107

def num_dice := 2
def sides_dice := 8
def total_outcomes := sides_dice ^ num_dice
def num_non_favorable_outcomes := (sides_dice - 1) ^ num_dice
def num_favorable_outcomes := total_outcomes - num_non_favorable_outcomes
def probability_favorable_outcomes := (15 : ℚ) / (64 : ℚ)

theorem probability_at_least_one_two :
  probability_favorable_outcomes = 15 / 64 :=
sorry

end probability_at_least_one_two_l281_281107


namespace option_D_min_value_is_2_l281_281310

noncomputable def funcD (x : ℝ) : ℝ :=
  (x^2 + 2) / Real.sqrt (x^2 + 1)

theorem option_D_min_value_is_2 :
  ∃ x : ℝ, funcD x = 2 :=
sorry

end option_D_min_value_is_2_l281_281310


namespace smallest_sum_of_xy_l281_281806

theorem smallest_sum_of_xy (x y : ℕ) (h1 : x ≠ y) (h2 : 0 < x) (h3 : 0 < y) 
  (h4 : (1:ℚ)/x + (1:ℚ)/y = (1:ℚ)/15) : x + y = 64 :=
sorry

end smallest_sum_of_xy_l281_281806


namespace calculate_total_cost_l281_281758

-- Define the cost for type A and type B fast foods as constants
def cost_of_type_A : ℕ := 30
def cost_of_type_B : ℕ := 20

-- Define the number of servings as variables
variables (a b : ℕ)

-- Define a function that calculates the total cost
def total_cost (a b : ℕ) : ℕ :=
  cost_of_type_A * a + cost_of_type_B * b

-- The theorem statement: Prove that the total cost is as calculated
theorem calculate_total_cost (a b : ℕ) :
  total_cost a b = 30 * a + 20 * b := by
  unfold total_cost
  simp

end calculate_total_cost_l281_281758


namespace value_of_x_l281_281899

def condition (x : ℝ) : Prop :=
  3 * x = (20 - x) + 20

theorem value_of_x : ∃ x : ℝ, condition x ∧ x = 10 := 
by
  sorry

end value_of_x_l281_281899


namespace expected_participants_2008_l281_281847

theorem expected_participants_2008 (initial_participants : ℕ) (annual_increase_rate : ℝ) :
  initial_participants = 1000 ∧ annual_increase_rate = 1.25 →
  (initial_participants * annual_increase_rate ^ 3) = 1953.125 :=
by
  sorry

end expected_participants_2008_l281_281847


namespace arrange_2015_integers_l281_281979

theorem arrange_2015_integers :
  ∃ (f : Fin 2015 → Fin 2015),
    (∀ i, (Nat.gcd ((f i).val + (f (i + 1)).val) 4 = 1 ∨ Nat.gcd ((f i).val + (f (i + 1)).val) 7 = 1)) ∧
    Function.Injective f ∧ 
    (∀ i, 1 ≤ (f i).val ∧ (f i).val ≤ 2015) :=
sorry

end arrange_2015_integers_l281_281979


namespace expected_second_ace_position_l281_281014

noncomputable def expected_position_of_second_ace (n : ℕ) : ℝ :=
((n + 1) : ℝ) / 2

theorem expected_second_ace_position (n : ℕ) (h : 2 < n) :
  expected_position_of_second_ace n = (n + 1) / 2 := by
sorry

end expected_second_ace_position_l281_281014


namespace min_velocity_increase_l281_281975

theorem min_velocity_increase (V_A V_B V_C : ℕ) (d_AB d_AC : ℕ) (h_VB_lt_VA : V_B < V_A) :
  d_AB = 50 ∧ d_AC = 300 ∧ V_B = 50 ∧ V_C = 70 ∧ V_A = 68 →
  (let ΔV := (370 / 5) - V_A in ΔV = 6) :=
by
  intros h_conditions,
  cases h_conditions with h_dAB h_remainder,
  cases h_remainder with h_dAC h_remainder2,
  cases h_remainder2 with h_VB h_remainder3,
  cases h_remainder3 with h_VC h_VA,
  let quotient := 370 / 5,
  let ΔV := quotient - 68,
  focus
    { rw [h_VB, h_VC, h_VA] at ⊢,
      sorry }

end min_velocity_increase_l281_281975


namespace gmat_test_statistics_l281_281207

theorem gmat_test_statistics 
    (p1 : ℝ) (p2 : ℝ) (p12 : ℝ) (neither : ℝ) (S : ℝ) 
    (h1 : p1 = 0.85)
    (h2 : p12 = 0.60) 
    (h3 : neither = 0.05) :
    0.25 + S = 0.95 → S = 0.70 :=
by
  sorry

end gmat_test_statistics_l281_281207


namespace train_crossing_time_l281_281917

/-- Time for a train of length 1500 meters traveling at 108 km/h to cross an electric pole is 50 seconds -/
theorem train_crossing_time (length : ℕ) (speed_kmph : ℕ) 
    (h₁ : length = 1500) (h₂ : speed_kmph = 108) : 
    (length / ((speed_kmph * 1000) / 3600) = 50) :=
by
  sorry

end train_crossing_time_l281_281917


namespace sunzi_oranges_l281_281008

theorem sunzi_oranges :
  ∃ (a : ℕ), ( 5 * a + 10 * 3 = 60 ) ∧ ( ∀ n, n = 0 → a = 6 ) :=
by
  sorry

end sunzi_oranges_l281_281008


namespace decreasing_function_implies_inequality_l281_281088

theorem decreasing_function_implies_inequality (k b : ℝ) (h : ∀ x : ℝ, (2 * k + 1) * x + b = (2 * k + 1) * x + b) :
  (∀ x1 x2 : ℝ, x1 < x2 → (2 * k + 1) * x1 + b > (2 * k + 1) * x2 + b) → k < -1/2 :=
by sorry

end decreasing_function_implies_inequality_l281_281088


namespace bags_le_40kg_l281_281308

theorem bags_le_40kg (capacity boxes crates sacks box_weight crate_weight sack_weight bag_weight: ℕ)
  (h_capacity: capacity = 13500)
  (h_boxes: boxes = 100)
  (h_crates: crates = 10)
  (h_sacks: sacks = 50)
  (h_box_weight: box_weight = 100)
  (h_crate_weight: crate_weight = 60)
  (h_sack_weight: sack_weight = 50)
  (h_bag_weight: bag_weight = 40) :
  10 = (capacity - (boxes * box_weight + crates * crate_weight + sacks * sack_weight)) / bag_weight := by 
  sorry

end bags_le_40kg_l281_281308


namespace total_winter_clothing_l281_281076

def num_scarves (boxes : ℕ) (scarves_per_box : ℕ) : ℕ := boxes * scarves_per_box
def num_mittens (boxes : ℕ) (mittens_per_box : ℕ) : ℕ := boxes * mittens_per_box
def num_hats (boxes : ℕ) (hats_per_box : ℕ) : ℕ := boxes * hats_per_box
def num_jackets (boxes : ℕ) (jackets_per_box : ℕ) : ℕ := boxes * jackets_per_box

theorem total_winter_clothing :
    num_scarves 4 8 + num_mittens 3 6 + num_hats 2 5 + num_jackets 1 3 = 63 :=
by
  -- The proof will use the given definitions and calculate the total
  sorry

end total_winter_clothing_l281_281076


namespace pears_more_than_apples_l281_281094

theorem pears_more_than_apples (red_apples green_apples pears : ℕ) (h1 : red_apples = 15) (h2 : green_apples = 8) (h3 : pears = 32) : (pears - (red_apples + green_apples) = 9) :=
by
  sorry

end pears_more_than_apples_l281_281094


namespace dave_more_than_jerry_games_l281_281226

variable (K D J : ℕ)  -- Declaring the variables for Ken, Dave, and Jerry respectively

-- Defining the conditions
def ken_more_games := K = D + 5
def dave_more_than_jerry := D > 7
def jerry_games := J = 7
def total_games := K + D + 7 = 32

-- Defining the proof problem
theorem dave_more_than_jerry_games (hK : ken_more_games K D) (hD : dave_more_than_jerry D) (hJ : jerry_games J) (hT : total_games K D) : D - 7 = 3 :=
by
  sorry

end dave_more_than_jerry_games_l281_281226


namespace problem_1_problem_2_l281_281392

noncomputable def f (x m : ℝ) := |x - 4 / m| + |x + m|

theorem problem_1 (m : ℝ) (hm : 0 < m) (x : ℝ) : f x m ≥ 4 := sorry

theorem problem_2 (m : ℝ) (hm : f 2 m > 5) : 
  m ∈ Set.Ioi ((1 + Real.sqrt 17) / 2) ∪ Set.Ioo 0 1 := sorry

end problem_1_problem_2_l281_281392


namespace mean_of_jane_scores_l281_281389

theorem mean_of_jane_scores :
  let scores := [96, 95, 90, 87, 91, 75]
  let n := 6
  let sum_scores := 96 + 95 + 90 + 87 + 91 + 75
  let mean := sum_scores / n
  mean = 89 := by
    sorry

end mean_of_jane_scores_l281_281389


namespace find_alpha_l281_281517

theorem find_alpha (α : ℝ) (h1 : α ∈ Set.Ioo (Real.pi / 2) (3 * Real.pi / 2))
  (h2 : ∃ k : ℝ, (Real.cos α, Real.sin α) = k • (-3, -3)) :
  α = 3 * Real.pi / 4 :=
by
  sorry

end find_alpha_l281_281517


namespace length_of_first_train_l281_281595

noncomputable def length_first_train
  (speed_train1_kmh : ℝ)
  (speed_train2_kmh : ℝ)
  (time_sec : ℝ)
  (length_train2_m : ℝ) : ℝ :=
  let relative_speed_mps := (speed_train1_kmh + speed_train2_kmh) * (1000 / 3600)
  let total_distance_m := relative_speed_mps * time_sec
  total_distance_m - length_train2_m

theorem length_of_first_train :
  length_first_train 80 65 7.82006405004841 165 = 150.106201 :=
  by
  -- Proof steps would go here.
  sorry

end length_of_first_train_l281_281595


namespace integer_fahrenheit_temps_count_l281_281585

theorem integer_fahrenheit_temps_count :
  let C (F : ℤ) : ℤ := Int.round ((5 / 9:ℚ) * (F - 32))
  let F' (F : ℤ) : ℤ := Int.round ((9 / 5:ℚ) * (C F)) + 32
  ∃ count : ℕ, count = 539 ∧ 
  (count = ∑ F in Finset.Icc (32 : ℤ) 1000, if F = F' F then 1 else 0) :=
by
  sorry

end integer_fahrenheit_temps_count_l281_281585


namespace present_value_of_machine_l281_281298

theorem present_value_of_machine {
  V0 : ℝ
} (h : 36100 = V0 * (0.95)^2) : V0 = 39978.95 :=
sorry

end present_value_of_machine_l281_281298


namespace mitya_age_l281_281993

-- Definitions of the ages
variables (M S : ℕ)

-- Conditions based on the problem statements
axiom condition1 : M = S + 11
axiom condition2 : S = 2 * (S - (M - S))

-- The theorem stating that Mitya is 33 years old
theorem mitya_age : M = 33 :=
by
  -- Outline the proof
  sorry

end mitya_age_l281_281993


namespace rectangle_sides_l281_281303

theorem rectangle_sides (a b : ℝ) (h1 : a < b) (h2 : a * b = 2 * a + 2 * b) : a < 4 ∧ b > 4 :=
by
  sorry

end rectangle_sides_l281_281303


namespace repayment_days_least_integer_l281_281072

theorem repayment_days_least_integer:
  ∀ (x : ℤ), (20 + 2 * x ≥ 60) → (x ≥ 20) :=
by
  intro x
  intro h
  sorry

end repayment_days_least_integer_l281_281072


namespace most_frequent_day_2014_l281_281561

/-- Given that March 9, 2014, is a Sunday, prove that Wednesday is the day that occurs most frequently in the year 2014. -/
theorem most_frequent_day_2014 
  (h: nat.mod (5 + 68) 7 = 0):
  (most_frequent_day 2014) = "Wednesday" :=
sorry

end most_frequent_day_2014_l281_281561


namespace six_vertex_graph_has_3_clique_or_independent_set_l281_281872

-- Define a 3-vertex clique
def is_clique (G : SimpleGraph V) (H : Set V) : Prop :=
  ∀ ⦃x y⦄, x ∈ H → y ∈ H → x ≠ y → G.Adj x y

-- Define a 3-vertex independent set (anticlique)
def is_independent_set (G : SimpleGraph V) (H : Set V) : Prop :=
  ∀ ⦃x y⦄, x ∈ H → y ∈ H → ¬G.Adj x y

-- Main theorem stating that any 6-vertex graph has a 3-vertex clique or a 3-vertex independent set
theorem six_vertex_graph_has_3_clique_or_independent_set 
  (V : Type) [Fintype V] [Nonempty V] (G : SimpleGraph V) (hV : Fintype.card V = 6) :
  ∃ (H : Set V), (H.card = 3 ∧ (is_clique G H ∨ is_independent_set G H)) :=
  sorry

end six_vertex_graph_has_3_clique_or_independent_set_l281_281872


namespace compound_interest_correct_amount_l281_281269

-- Define constants and conditions
def simple_interest (P R T : ℕ) : ℕ := (P * R * T) / 100

def compound_interest (P R T : ℕ) : ℕ := P * ((1 + R / 100) ^ T - 1)

-- Given values and conditions
def P₁ : ℕ := 1750
def R₁ : ℕ := 8
def T₁ : ℕ := 3
def R₂ : ℕ := 10
def T₂ : ℕ := 2

def SI : ℕ := simple_interest P₁ R₁ T₁
def CI : ℕ := 2 * SI

def P₂ : ℕ := 4000

-- The statement to be proven
theorem compound_interest_correct_amount : 
  compound_interest P₂ R₂ T₂ = CI := 
by 
  sorry

end compound_interest_correct_amount_l281_281269


namespace opposite_number_l281_281732

theorem opposite_number (x : ℤ) (h : -x = -2) : x = 2 :=
sorry

end opposite_number_l281_281732


namespace gasoline_tank_capacity_l281_281620

-- Given conditions
def initial_fraction_full := 5 / 6
def used_gallons := 15
def final_fraction_full := 2 / 3

-- Mathematical problem statement in Lean 4
theorem gasoline_tank_capacity (x : ℝ)
  (initial_full : initial_fraction_full * x = 5 / 6 * x)
  (final_full : initial_fraction_full * x - used_gallons = final_fraction_full * x) :
  x = 90 := by
  sorry

end gasoline_tank_capacity_l281_281620


namespace tamia_bell_pepper_pieces_l281_281254

theorem tamia_bell_pepper_pieces :
  let bell_peppers := 5
  let slices_per_pepper := 20
  let initial_slices := bell_peppers * slices_per_pepper
  let half_slices_cut := initial_slices / 2
  let small_pieces := half_slices_cut * 3
  let total_pieces := (initial_slices - half_slices_cut) + small_pieces
  total_pieces = 200 :=
by
  sorry

end tamia_bell_pepper_pieces_l281_281254


namespace quadratic_inequality_solution_range_l281_281375

theorem quadratic_inequality_solution_range (m : ℝ) :
  (∀ x : ℝ, x^2 + m * x + 1 > 0) ↔ -2 < m ∧ m < 2 :=
by
  sorry

end quadratic_inequality_solution_range_l281_281375


namespace arun_weight_average_l281_281684

theorem arun_weight_average (w : ℝ) 
  (h1 : 64 < w ∧ w < 72) 
  (h2 : 60 < w ∧ w < 70) 
  (h3 : w ≤ 67) : 
  (64 + 67) / 2 = 65.5 := 
  by sorry

end arun_weight_average_l281_281684


namespace surface_area_of_large_cube_is_486_cm_squared_l281_281867

noncomputable def surfaceAreaLargeCube : ℕ :=
  let small_box_count := 27
  let edge_small_box := 3
  let edge_large_cube := (small_box_count^(1/3)) * edge_small_box
  6 * edge_large_cube^2

theorem surface_area_of_large_cube_is_486_cm_squared :
  surfaceAreaLargeCube = 486 := 
sorry

end surface_area_of_large_cube_is_486_cm_squared_l281_281867


namespace chef_made_10_cakes_l281_281580

-- Definitions based on the conditions
def total_eggs : ℕ := 60
def eggs_in_fridge : ℕ := 10
def eggs_per_cake : ℕ := 5

-- Calculated values based on the definitions
def eggs_for_cakes : ℕ := total_eggs - eggs_in_fridge
def number_of_cakes : ℕ := eggs_for_cakes / eggs_per_cake

-- Theorem to prove
theorem chef_made_10_cakes : number_of_cakes = 10 := by
  sorry

end chef_made_10_cakes_l281_281580


namespace games_within_division_l281_281457

variables (N M : ℕ)
  (h1 : N > 2 * M)
  (h2 : M > 4)
  (h3 : 3 * N + 4 * M = 76)

theorem games_within_division :
  3 * N = 48 :=
sorry

end games_within_division_l281_281457


namespace area_of_circle_with_radius_2_is_4pi_l281_281515

theorem area_of_circle_with_radius_2_is_4pi :
  ∀ (π : ℝ), ∀ (r : ℝ), r = 2 → π > 0 → π * r^2 = 4 * π := 
by
  intros π r hr hπ
  sorry

end area_of_circle_with_radius_2_is_4pi_l281_281515


namespace polar_coordinates_of_point_l281_281928

noncomputable theory

-- Define the point in rectangular coordinates
def point_rect : ℝ × ℝ := (2, -2)

-- Define the conversion to polar coordinates function
def convertToPolar (x y : ℝ) : ℝ × ℝ :=
  let r := Real.sqrt (x^2 + y^2)
  let θ := if x > 0 then
             if y >= 0 then Real.arctan (y / x)
             else 2 * Real.pi - Real.arctan (Real.abs y / x)
           else if x < 0 then
             if y >= 0 then Real.pi - Real.arctan (y / (Real.abs x))
             else Real.pi + Real.arctan (Real.abs y / (Real.abs x))
           else if y > 0 then Real.pi / 2
           else 3 * Real.pi / 2
  (r, θ)

theorem polar_coordinates_of_point :
  convertToPolar (fst point_rect) (snd point_rect) = (2 * Real.sqrt 2, 7 * Real.pi / 4) :=
  by
    sorry

end polar_coordinates_of_point_l281_281928


namespace polynomial_sum_evaluation_l281_281863

noncomputable def q1 : Polynomial ℤ := Polynomial.X^3
noncomputable def q2 : Polynomial ℤ := Polynomial.X^2 + Polynomial.X + 1
noncomputable def q3 : Polynomial ℤ := Polynomial.X - 1
noncomputable def q4 : Polynomial ℤ := Polynomial.X^2 + 1

theorem polynomial_sum_evaluation :
  q1.eval 3 + q2.eval 3 + q3.eval 3 + q4.eval 3 = 52 :=
by
  sorry

end polynomial_sum_evaluation_l281_281863


namespace simplify_expression_l281_281331

theorem simplify_expression :
  2 + 3 / (4 + 5 / (6 + 7 / 8)) = 137 / 52 :=
by
  sorry

end simplify_expression_l281_281331


namespace tangent_division_l281_281411

theorem tangent_division (a b c d e : ℝ) (h0 : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d ∧ 0 ≤ e) :
  ∃ t1 t5 : ℝ, t1 = (a + b - c - d + e) / 2 ∧ t5 = (a - b - c + d + e) / 2 ∧ t1 + t5 = a :=
by
  sorry

end tangent_division_l281_281411


namespace linear_function_quadrants_l281_281969

theorem linear_function_quadrants (k b : ℝ) :
  (∀ x, (0 < x → 0 < k * x + b) ∧ (x < 0 → 0 < k * x + b) ∧ (x < 0 → k * x + b < 0)) →
  k > 0 ∧ b > 0 :=
by
  sorry

end linear_function_quadrants_l281_281969


namespace remainder_of_7_pow_145_mod_9_l281_281280

theorem remainder_of_7_pow_145_mod_9 : (7 ^ 145) % 9 = 7 := by
  sorry

end remainder_of_7_pow_145_mod_9_l281_281280


namespace average_wage_per_day_l281_281613

variable (numMaleWorkers : ℕ) (wageMale : ℕ) (numFemaleWorkers : ℕ) (wageFemale : ℕ) (numChildWorkers : ℕ) (wageChild : ℕ)

theorem average_wage_per_day :
  numMaleWorkers = 20 →
  wageMale = 35 →
  numFemaleWorkers = 15 →
  wageFemale = 20 →
  numChildWorkers = 5 →
  wageChild = 8 →
  (20 * 35 + 15 * 20 + 5 * 8) / (20 + 15 + 5) = 26 :=
by
  intros
  -- Proof would follow here
  sorry

end average_wage_per_day_l281_281613


namespace necessary_but_not_sufficient_l281_281587

theorem necessary_but_not_sufficient (x : ℝ) : (1 - x) * (1 + |x|) > 0 -> x < 2 :=
by
  sorry

end necessary_but_not_sufficient_l281_281587


namespace cases_in_1995_l281_281972

theorem cases_in_1995 (initial_cases cases_2010 : ℕ) (years_total : ℕ) (years_passed : ℕ) (cases_1995 : ℕ)
  (h1 : initial_cases = 700000) 
  (h2 : cases_2010 = 1000) 
  (h3 : years_total = 40) 
  (h4 : years_passed = 25)
  (h5 : cases_1995 = initial_cases - (years_passed * (initial_cases - cases_2010) / years_total)) : 
  cases_1995 = 263125 := 
sorry

end cases_in_1995_l281_281972


namespace max_k_consecutive_sum_2_times_3_pow_8_l281_281369

theorem max_k_consecutive_sum_2_times_3_pow_8 :
  ∃ k : ℕ, 0 < k ∧ 
           (∃ n : ℕ, 2 * 3^8 = (k * (2 * n + k + 1)) / 2) ∧
           (∀ k' : ℕ, (∃ n' : ℕ, 0 < k' ∧ 2 * 3^8 = (k' * (2 * n' + k' + 1)) / 2) → k' ≤ 81) :=
sorry

end max_k_consecutive_sum_2_times_3_pow_8_l281_281369


namespace base_k_conversion_l281_281514

theorem base_k_conversion (k : ℕ) (hk : 4 * k + 4 = 36) : 6 * 8 + 7 = 55 :=
by
  -- Proof skipped
  sorry

end base_k_conversion_l281_281514


namespace expand_polynomial_l281_281021

theorem expand_polynomial (x : ℂ) : 
  (1 + x^4) * (1 - x^5) * (1 + x^7) = 1 + x^4 - x^5 + x^7 + x^11 - x^9 - x^12 - x^16 := 
sorry

end expand_polynomial_l281_281021


namespace total_length_correct_l281_281017

def segment_lengths_Figure1 : List ℕ := [10, 3, 1, 1, 5, 7]

def removed_segments : List ℕ := [3, 1, 1, 5]

def remaining_segments_Figure2 : List ℕ := [10, (3 + 1 + 1), 7, 1]

def total_length_Figure2 : ℕ := remaining_segments_Figure2.sum

theorem total_length_correct :
  total_length_Figure2 = 23 :=
by
  sorry

end total_length_correct_l281_281017


namespace cost_of_each_ring_l281_281710

theorem cost_of_each_ring (R : ℝ) 
  (h1 : 4 * 12 + 8 * R = 80) : R = 4 :=
by 
  sorry

end cost_of_each_ring_l281_281710


namespace third_competitor_eats_l281_281778

-- Define the conditions based on the problem description
def first_competitor_hot_dogs : ℕ := 12
def second_competitor_hot_dogs := 2 * first_competitor_hot_dogs
def third_competitor_hot_dogs := second_competitor_hot_dogs - (second_competitor_hot_dogs / 4)

-- The theorem we need to prove
theorem third_competitor_eats :
  third_competitor_hot_dogs = 18 := by
  sorry

end third_competitor_eats_l281_281778


namespace line_eq_l281_281086

theorem line_eq (a b : ℝ) (h_a_pos : 0 < a) (h_b_pos : 0 < b) (h_eq : 1 / a + 9 / b = 1) (h_min_interp : a + b = 16) : 
  ∃ l : ℝ × ℝ → ℝ, ∀ x y : ℝ, l (x, y) = 3 * x + y - 12 :=
by
  sorry

end line_eq_l281_281086


namespace detergent_for_9_pounds_l281_281075

-- Define the given condition.
def detergent_per_pound : ℕ := 2

-- Define the total weight of clothes
def weight_of_clothes : ℕ := 9

-- Define the result of the detergent used.
def detergent_used (d : ℕ) (w : ℕ) : ℕ := d * w

-- Prove that the detergent used to wash 9 pounds of clothes is 18 ounces
theorem detergent_for_9_pounds :
  detergent_used detergent_per_pound weight_of_clothes = 18 := 
sorry

end detergent_for_9_pounds_l281_281075


namespace mass_percentage_Al_aluminum_carbonate_l281_281943

theorem mass_percentage_Al_aluminum_carbonate :
  let m_Al := 26.98  -- molar mass of Al in g/mol
  let m_C := 12.01  -- molar mass of C in g/mol
  let m_O := 16.00  -- molar mass of O in g/mol
  let molar_mass_CO3 := m_C + 3 * m_O  -- molar mass of CO3 in g/mol
  let molar_mass_Al2CO33 := 2 * m_Al + 3 * molar_mass_CO3  -- molar mass of Al2(CO3)3 in g/mol
  let mass_Al_in_Al2CO33 := 2 * m_Al  -- mass of Al in Al2(CO3)3 in g/mol
  (mass_Al_in_Al2CO33 / molar_mass_Al2CO33) * 100 = 23.05 :=
by
  -- Proof goes here
  sorry

end mass_percentage_Al_aluminum_carbonate_l281_281943


namespace original_class_strength_l281_281750

theorem original_class_strength (T N : ℕ) (h1 : T = 40 * N) (h2 : T + 12 * 32 = 36 * (N + 12)) : N = 12 :=
by
  sorry

end original_class_strength_l281_281750


namespace largest_angle_of_triangle_l281_281415

theorem largest_angle_of_triangle (x : ℝ) (h : x + 3 * x + 5 * x = 180) : 5 * x = 100 :=
sorry

end largest_angle_of_triangle_l281_281415


namespace smallest_product_not_factor_of_48_exists_l281_281434

theorem smallest_product_not_factor_of_48_exists : 
  ∃ (a b : ℕ), (a ≠ b) ∧ (a ∣ 48) ∧ (b ∣ 48) ∧ ¬ (a * b ∣ 48) ∧ 
  (∀ (c d : ℕ), (c ≠ d) ∧ (c ∣ 48) ∧ (d ∣ 48) ∧ ¬ (c * d ∣ 48) → a * b ≤ c * d) ∧ (a * b = 18) :=
sorry

end smallest_product_not_factor_of_48_exists_l281_281434


namespace probability_at_least_one_2_on_8_sided_dice_l281_281110

theorem probability_at_least_one_2_on_8_sided_dice :
  (∃ (d1 d2 : Fin 8), d1 = 1 ∨ d2 = 1) → (15 / 64) = (15 / 64) := by
  intro h
  sorry

end probability_at_least_one_2_on_8_sided_dice_l281_281110


namespace circle_area_from_points_l281_281566

theorem circle_area_from_points (C D : ℝ × ℝ) (hC : C = (2, 3)) (hD : D = (8, 9)) : 
  ∃ A : ℝ, A = 18 * Real.pi :=
by
  sorry

end circle_area_from_points_l281_281566


namespace length_of_bridge_l281_281003

theorem length_of_bridge (speed_kmh : ℝ) (time_min : ℝ) (length_m : ℝ) :
  speed_kmh = 5 → time_min = 15 → length_m = 1250 :=
by
  sorry

end length_of_bridge_l281_281003


namespace sin_half_alpha_l281_281166

noncomputable def given_cos_alpha (α : ℝ) : Prop :=
  α ∈ Ioo 0 (π / 2) ∧ cos α = (1 + real.sqrt 5) / 4

theorem sin_half_alpha (α : ℝ) (hα : given_cos_alpha α) :
  sin (α / 2) = (-1 + real.sqrt 5) / 4 :=
sorry

end sin_half_alpha_l281_281166


namespace simplify_expr1_simplify_and_evaluate_l281_281723

-- First problem: simplify and prove equality.
theorem simplify_expr1 (a : ℝ) :
  -2 * a^2 + 3 - (3 * a^2 - 6 * a + 1) + 3 = -5 * a^2 + 6 * a + 2 :=
by sorry

-- Second problem: simplify and evaluate under given conditions.
theorem simplify_and_evaluate (x y : ℝ) (h_x : x = -2) (h_y : y = -3) :
  (1 / 2) * x - 2 * (x - (1 / 3) * y^2) + (-(3 / 2) * x + (1 / 3) * y^2) = 15 :=
by sorry

end simplify_expr1_simplify_and_evaluate_l281_281723


namespace how_many_buns_each_student_gets_l281_281358

theorem how_many_buns_each_student_gets 
  (packages : ℕ) 
  (buns_per_package : ℕ) 
  (classes : ℕ) 
  (students_per_class : ℕ)
  (h1 : buns_per_package = 8)
  (h2 : packages = 30)
  (h3 : classes = 4)
  (h4 : students_per_class = 30) :
  (packages * buns_per_package) / (classes * students_per_class) = 2 :=
by sorry

end how_many_buns_each_student_gets_l281_281358


namespace quadratic_complete_square_l281_281267

open Real

theorem quadratic_complete_square (d e : ℝ) :
  (∀ x, x^2 - 24 * x + 50 = (x + d)^2 + e) → d + e = -106 :=
by
  intros h
  have h_eq := h 12
  sorry

end quadratic_complete_square_l281_281267


namespace determine_x_l281_281773

theorem determine_x (x : ℝ) (h : (1 / (Real.log x / Real.log 3) + 1 / (Real.log x / Real.log 5) + 1 / (Real.log x / Real.log 6) = 1)) : 
    x = 90 := 
by 
  sorry

end determine_x_l281_281773


namespace least_possible_value_of_quadratic_l281_281536

theorem least_possible_value_of_quadratic (p q : ℝ) (hq : ∀ x : ℝ, x^2 + p * x + q ≥ 0) : q = (p^2) / 4 :=
sorry

end least_possible_value_of_quadratic_l281_281536


namespace find_k_l281_281883

theorem find_k (k : ℝ)
  (h1 : k ≠ 0)
  (h2 : ∃ A B : ℝ × ℝ, A ≠ B ∧ (A.1 + A.2) = (B.1 + B.2) / 2 ∧ (A.1^2 + A.2^2 - 6 * A.1 - 4 * A.2 + 9 = 0) ∧ (B.1^2 + B.2^2 - 6 * B.1 - 4 * B.2 + 9 = 0)
     ∧ dist A B = 2 * Real.sqrt 3)
  (h3 : ∀ x y : ℝ, y = k * x + 3 → (x^2 + y^2 - 6 * x - 4 * y + 9) = 0)
  : k = 1 := sorry

end find_k_l281_281883


namespace equivalent_multipliers_l281_281206

variable (a b : ℝ)

theorem equivalent_multipliers (a b : ℝ) :
  let a_final := 0.93 * a
  let expr := a_final + 0.05 * b
  expr = 0.93 * a + 0.05 * b  :=
by
  -- Proof placeholder
  sorry

end equivalent_multipliers_l281_281206


namespace distance_formula_proof_l281_281203

open Real

noncomputable def distance_between_points_on_curve
  (a b c d m k : ℝ)
  (h1 : b = m * a^2 + k)
  (h2 : d = m * c^2 + k) :
  ℝ :=
  |c - a| * sqrt (1 + m^2 * (c + a)^2)

theorem distance_formula_proof
  (a b c d m k : ℝ)
  (h1 : b = m * a^2 + k)
  (h2 : d = m * c^2 + k) :
  distance_between_points_on_curve a b c d m k h1 h2 = |c - a| * sqrt (1 + m^2 * (c + a)^2) :=
by
  sorry

end distance_formula_proof_l281_281203


namespace committee_vote_change_l281_281843

-- Let x be the number of votes for the resolution initially.
-- Let y be the number of votes against the resolution initially.
-- The total number of voters is 500: x + y = 500.
-- The initial margin by which the resolution was defeated: y - x = m.
-- In the re-vote, the resolution passed with a margin three times the initial margin: x' - y' = 3m.
-- The number of votes for the re-vote was 13/12 of the votes against initially: x' = 13/12 * y.
-- The total number of voters remains 500 in the re-vote: x' + y' = 500.

theorem committee_vote_change (x y x' y' m : ℕ)
  (h1 : x + y = 500)
  (h2 : y - x = m)
  (h3 : x' - y' = 3 * m)
  (h4 : x' = 13 * y / 12)
  (h5 : x' + y' = 500) : x' - x = 40 := 
  by
  sorry

end committee_vote_change_l281_281843


namespace value_of_x_l281_281535

theorem value_of_x (x : ℚ) (h : (3 * x + 4) / 7 = 15) : x = 101 / 3 :=
by
  sorry

end value_of_x_l281_281535


namespace total_seeds_grace_can_plant_l281_281352

theorem total_seeds_grace_can_plant :
  let lettuce_seeds_per_row := 25
  let carrot_seeds_per_row := 20
  let radish_seeds_per_row := 30
  let large_bed_rows_limit := 5
  let medium_bed_rows_limit := 3
  let small_bed_rows_limit := 2
  let large_beds := 2
  let medium_beds := 2
  let small_bed := 1
  let large_bed_planting := 
    [(3, lettuce_seeds_per_row), (2, carrot_seeds_per_row)]  -- 3 rows of lettuce, 2 rows of carrots in large beds
  let medium_bed_planting := 
    [(1, lettuce_seeds_per_row), (1, carrot_seeds_per_row), (1, radish_seeds_per_row)] --in medium beds
  let small_bed_planting := 
    [(1, carrot_seeds_per_row), (1, radish_seeds_per_row)] --in small beds
  (3 * lettuce_seeds_per_row + 2 * carrot_seeds_per_row) * large_beds +
  (1 * lettuce_seeds_per_row + 1 * carrot_seeds_per_row + 1 * radish_seeds_per_row) * medium_beds +
  (1 * carrot_seeds_per_row + 1 * radish_seeds_per_row) * small_bed = 430 :=
by
  sorry

end total_seeds_grace_can_plant_l281_281352


namespace determine_d_l281_281148

def Q (x d : ℝ) : ℝ := x^3 - 3*x^2 + d*x - 8

theorem determine_d (d : ℝ) : (∃ d, Q (-2) d = 0) → d = -14 := by
  sorry

end determine_d_l281_281148


namespace problem_1_problem_2_problem_3_l281_281240

-- The sequence S_n and its given condition
def S (n : ℕ) (a : ℕ → ℕ) : ℕ := 2 * a n - 2 * n

-- Definitions for a_1, a_2, and a_3 based on S_n conditions
theorem problem_1 (S : ℕ → ℕ) (a : ℕ → ℕ) (h : ∀ n, S n = 2 * a n - 2 * n) :
  a 1 = 2 ∧ a 2 = 6 ∧ a 3 = 14 :=
sorry

-- Definition of sequence b_n and its property of being geometric
def b (n : ℕ) (a : ℕ → ℕ) : ℕ := a n + 2

theorem problem_2 (S : ℕ → ℕ) (a : ℕ → ℕ) (h : ∀ n, S n = 2 * a n - 2 * n) :
  ∀ n ≥ 1, b n a = 2 * b (n - 1) a :=
sorry

-- The sum of the first n terms of the sequence {na_n}, denoted by T_n
def T (n : ℕ) (a : ℕ → ℕ) : ℕ := (n + 1) * 2 ^ (n + 2) + 4 - n * (n + 1)

theorem problem_3 (S : ℕ → ℕ) (a : ℕ → ℕ) (h : ∀ n, S n = 2 * a n - 2 * n) :
  ∀ n, T n a = (n + 1) * 2 ^ (n + 2) + 4 - n * (n + 1) :=
sorry

end problem_1_problem_2_problem_3_l281_281240


namespace lead_amount_in_mixture_l281_281301

theorem lead_amount_in_mixture 
  (W : ℝ) 
  (h_copper : 0.60 * W = 12) 
  (h_mixture_composition : (0.15 * W = 0.15 * W) ∧ (0.25 * W = 0.25 * W) ∧ (0.60 * W = 0.60 * W)) :
  (0.25 * W = 5) :=
by
  sorry

end lead_amount_in_mixture_l281_281301


namespace solve_fractional_equation_l281_281417

theorem solve_fractional_equation (x : ℝ) (h₀ : 2 = 3 * (x + 1) / (4 - x)) : x = 1 :=
sorry

end solve_fractional_equation_l281_281417


namespace factor_polynomial_l281_281656

theorem factor_polynomial : 
  (∀ x : ℝ, (x^2 + 6 * x + 9 - 64 * x^4) = (-8 * x^2 + x + 3) * (8 * x^2 + x + 3)) :=
by
  intro x
  -- Left-hand side
  let lhs := x^2 + 6 * x + 9 - 64 * x^4
  -- Right-hand side after factorization
  let rhs := (-8 * x^2 + x + 3) * (8 * x^2 + x + 3)
  -- Prove the equality
  show lhs = rhs
  sorry

end factor_polynomial_l281_281656


namespace solve_for_diamond_l281_281204

-- Define what it means for a digit to represent a base-9 number and base-10 number
noncomputable def fromBase (b : ℕ) (digits : List ℕ) : ℕ :=
  digits.foldl (λ acc d => acc * b + d) 0

-- The theorem we want to prove
theorem solve_for_diamond (diamond : ℕ) (h_digit : diamond < 10) :
  fromBase 9 [diamond, 3] = fromBase 10 [diamond, 2] → diamond = 1 :=
by 
  sorry

end solve_for_diamond_l281_281204


namespace smallest_nonfactor_product_of_factors_of_48_l281_281428

theorem smallest_nonfactor_product_of_factors_of_48 :
  ∃ a b : ℕ, a ≠ b ∧ a ∣ 48 ∧ b ∣ 48 ∧ ¬ (a * b ∣ 48) ∧ (∀ c d : ℕ, c ≠ d ∧ c ∣ 48 ∧ d ∣ 48 ∧ ¬ (c * d ∣ 48) → a * b ≤ c * d) ∧ a * b = 18 :=
sorry

end smallest_nonfactor_product_of_factors_of_48_l281_281428


namespace resistance_between_opposite_vertices_of_cube_l281_281764

-- Define the parameters of the problem
def resistance_cube_edge : ℝ := 1

-- Define the function to calculate the equivalent resistance
noncomputable def equivalent_resistance_opposite_vertices (R : ℝ) : ℝ :=
  let R1 := R / 3
  let R2 := R / 6
  let R3 := R / 3
  R1 + R2 + R3

-- State the theorem to prove the resistance between two opposite vertices
theorem resistance_between_opposite_vertices_of_cube :
  equivalent_resistance_opposite_vertices resistance_cube_edge = 5 / 6 :=
by
  sorry

end resistance_between_opposite_vertices_of_cube_l281_281764


namespace sixty_percent_of_40_greater_than_four_fifths_of_25_l281_281836

theorem sixty_percent_of_40_greater_than_four_fifths_of_25 :
  (60 / 100 : ℝ) * 40 - (4 / 5 : ℝ) * 25 = 4 := by
  sorry

end sixty_percent_of_40_greater_than_four_fifths_of_25_l281_281836


namespace nonnegative_integer_pairs_solution_l281_281151

theorem nonnegative_integer_pairs_solution :
  ∀ (x y: ℕ), ((x * y + 2) ^ 2 = x^2 + y^2) ↔ (x = 0 ∧ y = 2) ∨ (x = 2 ∧ y = 0) :=
by
  sorry

end nonnegative_integer_pairs_solution_l281_281151


namespace circle_tangent_y_eq_2_center_on_y_axis_radius_1_l281_281880

theorem circle_tangent_y_eq_2_center_on_y_axis_radius_1 :
  ∃ (y0 : ℝ), (∀ x y : ℝ, (x - 0)^2 + (y - y0)^2 = 1 ↔ y = y0 + 1 ∨ y = y0 - 1) := by
  sorry

end circle_tangent_y_eq_2_center_on_y_axis_radius_1_l281_281880


namespace shadow_length_of_flagpole_l281_281907

theorem shadow_length_of_flagpole :
  ∀ (S : ℝ), (18 : ℝ) / S = (22 : ℝ) / 55 → S = 45 :=
by
  intro S h
  sorry

end shadow_length_of_flagpole_l281_281907


namespace smallest_nonfactor_product_of_factors_of_48_l281_281427

theorem smallest_nonfactor_product_of_factors_of_48 :
  ∃ a b : ℕ, a ≠ b ∧ a ∣ 48 ∧ b ∣ 48 ∧ ¬ (a * b ∣ 48) ∧ (∀ c d : ℕ, c ≠ d ∧ c ∣ 48 ∧ d ∣ 48 ∧ ¬ (c * d ∣ 48) → a * b ≤ c * d) ∧ a * b = 18 :=
sorry

end smallest_nonfactor_product_of_factors_of_48_l281_281427


namespace import_tax_calculation_l281_281286

def import_tax_rate : ℝ := 0.07
def excess_value_threshold : ℝ := 1000
def total_value_item : ℝ := 2610
def correct_import_tax : ℝ := 112.7

theorem import_tax_calculation :
  (total_value_item - excess_value_threshold) * import_tax_rate = correct_import_tax :=
by
  sorry

end import_tax_calculation_l281_281286


namespace pyramid_volume_l281_281625

noncomputable def volume_of_pyramid 
  (EFGH_rect : ℝ × ℝ) 
  (EF_len : EFGH_rect.1 = 15 * Real.sqrt 2) 
  (FG_len : EFGH_rect.2 = 14 * Real.sqrt 2)
  (isosceles_pyramid : Prop) : ℝ :=
  sorry

theorem pyramid_volume 
  (EFGH_rect : ℝ × ℝ) 
  (EF_len : EFGH_rect.1 = 15 * Real.sqrt 2) 
  (FG_len : EFGH_rect.2 = 14 * Real.sqrt 2) 
  (isosceles_pyramid : Prop) : 
  volume_of_pyramid EFGH_rect EF_len FG_len isosceles_pyramid = 735 := 
sorry

end pyramid_volume_l281_281625


namespace slope_interval_non_intersect_l281_281233

noncomputable def parabola (x : ℝ) : ℝ := x^2 + 5

def Q : ℝ × ℝ := (10, 10)

theorem slope_interval_non_intersect (r s : ℝ) (h : ∀ m : ℝ,
  ¬∃ x : ℝ, parabola x = m * (x - 10) + 10 ↔ r < m ∧ m < s) :
  r + s = 40 :=
sorry

end slope_interval_non_intersect_l281_281233


namespace calculate_g3_l281_281968

def g (x : ℚ) : ℚ := (2 * x - 3) / (5 * x + 2)

theorem calculate_g3 : g 3 = 3 / 17 :=
by {
    -- Here we add the proof steps if necessary, but for now we use sorry
    sorry
}

end calculate_g3_l281_281968


namespace locus_of_Q_eq_l281_281194

open Real

noncomputable def ellipse_eq (x y : ℝ) : Prop := (x^2 / 24) + (y^2 / 16) = 1
noncomputable def line_eq (x y : ℝ) : Prop := (x / 12) + (y / 8) = 1
noncomputable def point_on_OP_eq (x y : ℝ) (P : ℝ) (Q R : ℝ) : Prop := 
  R^2 = |x * Q|

axiom point_on_line (x y : ℝ) : line_eq x y

axiom loci_eq (x y : ℝ) : ellipse_eq x y ∧ ∀ P ∈ point_on_line x y, 
  let R := (x^2 / 24 + y^2 / 16) ^ 1/2,
      Q := |x * P / R^2| 
  in ( Q = (x, y))

theorem locus_of_Q_eq (x y : ℝ) : loci_eq x y → 
  ((x - 1)^2 / (5/2)) + ((y - 1)^2 / (5/3)) = 1 := 
by
  sorry

end locus_of_Q_eq_l281_281194


namespace arithmetic_seq_sum_l281_281977

theorem arithmetic_seq_sum (a : ℕ → ℤ) (d : ℤ) 
  (h1 : ∀ n, a (n + 1) = a n + d) 
  (h2 : a 3 + a 7 = 38) : 
  a 2 + a 4 + a 6 + a 8 = 76 :=
by 
  sorry

end arithmetic_seq_sum_l281_281977


namespace zoo_gorillas_sent_6_l281_281305

theorem zoo_gorillas_sent_6 (G : ℕ) : 
  let initial_animals := 68
  let after_sending_gorillas := initial_animals - G
  let after_adopting_hippopotamus := after_sending_gorillas + 1
  let after_taking_rhinos := after_adopting_hippopotamus + 3
  let after_birth_lion_cubs := after_taking_rhinos + 8
  let after_adding_meerkats := after_birth_lion_cubs + (2 * 8)
  let final_animals := 90
  after_adding_meerkats = final_animals → G = 6 := 
by
  intros
  let initial_animals := 68
  let after_sending_gorillas := initial_animals - G
  let after_adopting_hippopotamus := after_sending_gorillas + 1
  let after_taking_rhinos := after_adopting_hippopotamus + 3
  let after_birth_lion_cubs := after_taking_rhinos + 8
  let after_adding_meerkats := after_birth_lion_cubs + (2 * 8)
  let final_animals := 90
  sorry

end zoo_gorillas_sent_6_l281_281305


namespace daily_avg_for_entire_month_is_correct_l281_281542

-- conditions
def avg_first_25_days := 63
def days_first_25 := 25
def avg_last_5_days := 33
def days_last_5 := 5
def total_days := days_first_25 + days_last_5

-- question: What is the daily average for the entire month?
theorem daily_avg_for_entire_month_is_correct : 
  (avg_first_25_days * days_first_25 + avg_last_5_days * days_last_5) / total_days = 58 := by
  sorry

end daily_avg_for_entire_month_is_correct_l281_281542


namespace correct_system_l281_281125

def system_of_equations (x y : ℤ) : Prop :=
  (5 * x + 45 = y) ∧ (7 * x - 3 = y)

theorem correct_system : ∃ x y : ℤ, system_of_equations x y :=
sorry

end correct_system_l281_281125


namespace sin_half_alpha_l281_281188

theorem sin_half_alpha (α : ℝ) (hα : 0 < α ∧ α < π / 2) (hcos : Real.cos α = (1 + Real.sqrt 5) / 4) : 
  Real.sin (α / 2) = (Real.sqrt 5 - 1) / 4 := 
by 
  sorry

end sin_half_alpha_l281_281188


namespace third_competitor_eats_l281_281777

-- Define the conditions based on the problem description
def first_competitor_hot_dogs : ℕ := 12
def second_competitor_hot_dogs := 2 * first_competitor_hot_dogs
def third_competitor_hot_dogs := second_competitor_hot_dogs - (second_competitor_hot_dogs / 4)

-- The theorem we need to prove
theorem third_competitor_eats :
  third_competitor_hot_dogs = 18 := by
  sorry

end third_competitor_eats_l281_281777


namespace jasmine_max_cards_l281_281700

-- Define constants and conditions
def initial_card_price : ℝ := 0.95
def discount_card_price : ℝ := 0.85
def budget : ℝ := 9.00
def threshold : ℕ := 6

-- Define the condition for the total cost if more than 6 cards are bought
def total_cost (n : ℕ) : ℝ :=
  if n ≤ threshold then initial_card_price * n
  else initial_card_price * threshold + discount_card_price * (n - threshold)

-- Define the condition for the maximum number of cards Jasmine can buy 
def max_cards (n : ℕ) : Prop :=
  total_cost n ≤ budget ∧ ∀ m : ℕ, total_cost m ≤ budget → m ≤ n

-- Theore statement stating Jasmine can buy a maximum of 9 cards
theorem jasmine_max_cards : max_cards 9 :=
sorry

end jasmine_max_cards_l281_281700


namespace nathaniel_initial_tickets_l281_281713

theorem nathaniel_initial_tickets (a b c : ℕ) (h1 : a = 2) (h2 : b = 4) (h3 : c = 3) :
  a * b + c = 11 :=
by
  sorry

end nathaniel_initial_tickets_l281_281713


namespace expression_for_A_div_B_l281_281681

theorem expression_for_A_div_B (x A B : ℝ)
  (h1 : x^3 + 1/x^3 = A)
  (h2 : x - 1/x = B) :
  A / B = B^2 + 3 := 
sorry

end expression_for_A_div_B_l281_281681


namespace find_n_l281_281501

theorem find_n (x y : ℝ) (h1 : (7 * x + 2 * y) / (x - n * y) = 23) (h2 : x / (2 * y) = 3 / 2) :
  ∃ n : ℝ, n = 2 := by
  sorry

end find_n_l281_281501


namespace graph_eq_pair_of_straight_lines_l281_281412

theorem graph_eq_pair_of_straight_lines (x y : ℝ) :
  x^2 - 9*y^2 = 0 ↔ (x = 3*y ∨ x = -3*y) :=
by
  sorry

end graph_eq_pair_of_straight_lines_l281_281412


namespace sequence_a_n_correctness_l281_281522

theorem sequence_a_n_correctness (a : ℕ → ℚ) (h₀ : a 1 = 1)
  (h₁ : ∀ n : ℕ, n > 0 → 2 * a (n + 1) = 2 * a n + 1) : a 2 = 1.5 := by
  sorry

end sequence_a_n_correctness_l281_281522


namespace problem_solution_l281_281824

variable {A B : Set ℝ}

def definition_A : Set ℝ := {x | 0 < x ∧ x ≤ 2}
def definition_B : Set ℝ := {x | x ≤ 3}

theorem problem_solution : definition_A ∩ definition_B = definition_A :=
by
  sorry

end problem_solution_l281_281824


namespace A_inter_B_eq_A_l281_281820

def A := {x : ℝ | 0 < x ∧ x ≤ 2}
def B := {x : ℝ | x ≤ 3}

theorem A_inter_B_eq_A : A ∩ B = A := 
by 
  sorry 

end A_inter_B_eq_A_l281_281820


namespace third_competitor_hot_dogs_l281_281776

theorem third_competitor_hot_dogs (first_ate : ℕ) (second_multiplier : ℕ) (third_percent_less : ℕ) (second_ate third_ate : ℕ) 
  (H1 : first_ate = 12)
  (H2 : second_multiplier = 2)
  (H3 : third_percent_less = 25)
  (H4 : second_ate = first_ate * second_multiplier)
  (H5 : third_ate = second_ate - (third_percent_less * second_ate / 100)) : 
  third_ate = 18 := 
by 
  sorry

end third_competitor_hot_dogs_l281_281776


namespace tractors_moved_l281_281619

-- Define initial conditions
def total_area (tractors: ℕ) (days: ℕ) (hectares_per_day: ℕ) := tractors * days * hectares_per_day

theorem tractors_moved (original_tractors remaining_tractors: ℕ)
  (days_original: ℕ) (hectares_per_day_original: ℕ)
  (days_remaining: ℕ) (hectares_per_day_remaining: ℕ)
  (total_area_original: ℕ) 
  (h1: total_area original_tractors days_original hectares_per_day_original = total_area_original)
  (h2: total_area remaining_tractors days_remaining hectares_per_day_remaining = total_area_original) :
  original_tractors - remaining_tractors = 2 :=
by
  sorry

end tractors_moved_l281_281619


namespace James_gold_bars_l281_281550

theorem James_gold_bars (P : ℝ) (h_condition1 : 60 - P / 100 * 60 = 54) : P = 10 := 
  sorry

end James_gold_bars_l281_281550


namespace height_of_circular_segment_l281_281498

theorem height_of_circular_segment (d a : ℝ) (h : ℝ) :
  (h = (d - Real.sqrt (d^2 - a^2)) / 2) ↔ 
  ((a / 2)^2 + (d / 2 - h)^2 = (d / 2)^2) :=
sorry

end height_of_circular_segment_l281_281498


namespace sum_bases_l281_281689

theorem sum_bases (R1 R2 : ℕ) (F1 F2 : ℚ)
  (h1 : F1 = (4 * R1 + 5) / (R1 ^ 2 - 1))
  (h2 : F2 = (5 * R1 + 4) / (R1 ^ 2 - 1))
  (h3 : F1 = (3 * R2 + 8) / (R2 ^ 2 - 1))
  (h4 : F2 = (6 * R2 + 1) / (R2 ^ 2 - 1)) :
  R1 + R2 = 19 :=
sorry

end sum_bases_l281_281689


namespace outer_perimeter_fence_l281_281408

-- Definitions based on given conditions
def total_posts : Nat := 16
def post_width_feet : Real := 0.5 -- 6 inches converted to feet
def gap_length_feet : Real := 6 -- gap between posts in feet
def num_sides : Nat := 4 -- square field has 4 sides

-- Hypotheses that capture conditions and intermediate calculations
def num_corners : Nat := 4
def non_corner_posts : Nat := total_posts - num_corners
def non_corner_posts_per_side : Nat := non_corner_posts / num_sides
def posts_per_side : Nat := non_corner_posts_per_side + 2
def gaps_per_side : Nat := posts_per_side - 1
def length_gaps_per_side : Real := gaps_per_side * gap_length_feet
def total_post_width_per_side : Real := posts_per_side * post_width_feet
def length_one_side : Real := length_gaps_per_side + total_post_width_per_side
def perimeter : Real := num_sides * length_one_side

-- The theorem to prove
theorem outer_perimeter_fence : perimeter = 106 := by
  sorry

end outer_perimeter_fence_l281_281408


namespace value_of_expression_l281_281531

variable (m : ℝ)

theorem value_of_expression (h : 2 * m^2 + 3 * m - 1 = 0) : 
  4 * m^2 + 6 * m - 2019 = -2017 :=
by
  sorry

end value_of_expression_l281_281531


namespace unique_prime_satisfying_condition_l281_281032

theorem unique_prime_satisfying_condition :
  ∃! p : ℕ, Prime p ∧ (∀ q : ℕ, Prime q ∧ q < p → ∀ k r : ℕ, p = k * q + r ∧ 0 ≤ r ∧ r < q → ∀ a : ℕ, a > 1 → ¬ a^2 ∣ r) ∧ p = 13 :=
sorry

end unique_prime_satisfying_condition_l281_281032


namespace assign_roles_l281_281129

def maleRoles : ℕ := 3
def femaleRoles : ℕ := 3
def eitherGenderRoles : ℕ := 4
def menCount : ℕ := 7
def womenCount : ℕ := 8

theorem assign_roles : 
  (menCount.choose maleRoles) * 
  (womenCount.choose femaleRoles) * 
  ((menCount + womenCount - maleRoles - femaleRoles).choose eitherGenderRoles) = 213955200 := 
  sorry

end assign_roles_l281_281129


namespace initial_volume_of_mixture_l281_281380

-- Define the conditions of the problem as hypotheses
variable (milk_ratio water_ratio : ℕ) (W : ℕ) (initial_mixture : ℕ)
variable (h1 : milk_ratio = 2) (h2 : water_ratio = 1)
variable (h3 : W = 60)
variable (h4 : water_ratio + milk_ratio = 3) -- The sum of the ratios used in the equation

theorem initial_volume_of_mixture : initial_mixture = 60 :=
by
  sorry

end initial_volume_of_mixture_l281_281380


namespace opposite_sides_parallel_l281_281735

-- Definitions of the vertices and structure of the hexagon
structure ConvexHexagon (α : Type) [AddGroup α] [LinearOrderedAddCommMonoid α] :=
  (A B C D E F : α)
  (convex : true) -- convexity condition, mocked here as true for simplification
  (equal_sides : (A - B).abs = (B - C).abs ∧ (B - C).abs = (C - D).abs ∧ (C - D).abs = (D - E).abs ∧ (D - E).abs = (E - F).abs ∧ (E - F).abs = (F - A).abs)
  (sum_angles_ace : ∀ (angles : α), angles.sum = 360)
  (sum_angles_bdf : ∀ (angles : α), angles.sum = 360)

-- Theorem statement of our problem based on the given definitions and conditions
theorem opposite_sides_parallel {α : Type} [AddGroup α] [LinearOrderedAddCommMonoid α] 
  (H : ConvexHexagon α) : 
  parallel(H.A H.D) ∧ parallel(H.B H.E) ∧ parallel(H.C H.F) :=
sorry

end opposite_sides_parallel_l281_281735


namespace length_of_platform_l281_281004

theorem length_of_platform (length_of_train : ℕ) (speed_kmph : ℕ) (time_s : ℕ) (L : ℕ) :
  length_of_train = 160 → speed_kmph = 72 → time_s = 25 → (L = 340) :=
by
  sorry

end length_of_platform_l281_281004


namespace sin_half_alpha_l281_281175

noncomputable def alpha : ℝ := sorry
def is_acute (alpha : ℝ) : Prop := 0 < alpha ∧ alpha < π / 2

axiom acos_alpha : cos alpha = (1 + sqrt 5) / 4
axiom acute_alpha : is_acute alpha

theorem sin_half_alpha : sin (alpha / 2) = (sqrt 5 - 1) / 4 :=
by
  sorry

end sin_half_alpha_l281_281175


namespace initial_pollykawgs_computation_l281_281779

noncomputable def initial_pollykawgs_in_pond (daily_rate_matured : ℕ) (daily_rate_caught : ℕ)
  (total_days : ℕ) (catch_days : ℕ) : ℕ :=
let first_phase := (daily_rate_matured + daily_rate_caught) * catch_days
let second_phase := daily_rate_matured * (total_days - catch_days)
first_phase + second_phase

theorem initial_pollykawgs_computation :
  initial_pollykawgs_in_pond 50 10 44 20 = 2400 :=
by sorry

end initial_pollykawgs_computation_l281_281779


namespace value_of_function_at_2_l281_281020

theorem value_of_function_at_2 (q : ℝ → ℝ) : q 2 = 5 :=
by
  -- Condition: The point (2, 5) lies on the graph of q
  have point_on_graph : q 2 = 5 := sorry
  exact point_on_graph

end value_of_function_at_2_l281_281020


namespace devin_teaching_years_l281_281594

theorem devin_teaching_years (total_years : ℕ) (tom_years : ℕ) (devin_years : ℕ) 
  (half_tom_years : ℕ)
  (h1 : total_years = 70) 
  (h2 : tom_years = 50)
  (h3 : total_years = tom_years + devin_years) 
  (h4 : half_tom_years = tom_years / 2) : 
  half_tom_years - devin_years = 5 :=
by
  sorry

end devin_teaching_years_l281_281594


namespace students_per_group_l281_281593

theorem students_per_group (total_students not_picked groups : ℕ) 
    (h1 : total_students = 64) 
    (h2 : not_picked = 36) 
    (h3 : groups = 4) : (total_students - not_picked) / groups = 7 :=
by
  sorry

end students_per_group_l281_281593


namespace ratio_adult_child_l281_281263

theorem ratio_adult_child (total_fee adults_fee children_fee adults children : ℕ) 
  (h1 : adults ≥ 1) (h2 : children ≥ 1) 
  (h3 : adults_fee = 30) (h4 : children_fee = 15) 
  (h5 : total_fee = 2250) 
  (h6 : adults_fee * adults + children_fee * children = total_fee) :
  (2 : ℚ) = adults / children :=
sorry

end ratio_adult_child_l281_281263


namespace Anna_phone_chargers_l281_281922

-- Define the conditions and the goal in Lean
theorem Anna_phone_chargers (P L : ℕ) (h1 : L = 5 * P) (h2 : P + L = 24) : P = 4 :=
by
  sorry

end Anna_phone_chargers_l281_281922


namespace find_x_l281_281068

-- Definition of the binary operation
def binary_operation (a b c d : ℤ) : ℤ × ℤ :=
  (a - c, b + d)

-- Definition of our main theorem to be proved
theorem find_x (x y : ℤ) (h : binary_operation x y 2 3 = (4, 5)) : x = 6 :=
  by sorry

end find_x_l281_281068


namespace year_2013_is_not_lucky_l281_281774

-- Definitions based on conditions
def last_two_digits (year : ℕ) : ℕ := year % 100

def is_valid_date (month : ℕ) (day : ℕ) (year : ℕ) : Prop :=
  month * day = last_two_digits year

def is_lucky_year (year : ℕ) : Prop :=
  ∃ (month : ℕ) (day : ℕ), month <= 12 ∧ day <= 12 ∧ is_valid_date month day year

-- The main statement to prove
theorem year_2013_is_not_lucky : ¬ is_lucky_year 2013 :=
by {
  sorry
}

end year_2013_is_not_lucky_l281_281774


namespace number_of_cakes_l281_281578

theorem number_of_cakes (total_eggs eggs_in_fridge eggs_per_cake : ℕ) (h1 : total_eggs = 60) (h2 : eggs_in_fridge = 10) (h3 : eggs_per_cake = 5) :
  (total_eggs - eggs_in_fridge) / eggs_per_cake = 10 :=
by
  sorry

end number_of_cakes_l281_281578


namespace smallest_sum_of_xy_l281_281803

theorem smallest_sum_of_xy (x y : ℕ) (h1 : x ≠ y) (h2 : 0 < x) (h3 : 0 < y) 
  (h4 : (1:ℚ)/x + (1:ℚ)/y = (1:ℚ)/15) : x + y = 64 :=
sorry

end smallest_sum_of_xy_l281_281803


namespace pieces_of_gum_l281_281100

variable (initial_gum total_gum given_gum : ℕ)

theorem pieces_of_gum (h1 : given_gum = 16) (h2 : total_gum = 54) : initial_gum = 38 :=
by
  sorry

end pieces_of_gum_l281_281100


namespace AM_GM_inequality_l281_281237

theorem AM_GM_inequality (a b c d : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : d > 0) : 
  (a / b + b / c + c / d + d / a) ≥ 4 := 
sorry

end AM_GM_inequality_l281_281237


namespace water_volume_correct_l281_281297

noncomputable def volume_of_water : ℝ :=
  let r := 4
  let h := 9
  let d := 2
  48 * Real.pi - 36 * Real.sqrt 3

theorem water_volume_correct :
  volume_of_water = 48 * Real.pi - 36 * Real.sqrt 3 := 
by sorry

end water_volume_correct_l281_281297


namespace gray_part_area_l281_281276

theorem gray_part_area (area_rect1 area_rect2 area_black area_white gray_part_area : ℕ)
  (h_rect1 : area_rect1 = 80)
  (h_rect2 : area_rect2 = 108)
  (h_black : area_black = 37)
  (h_white : area_white = area_rect1 - area_black)
  (h_white_correct : area_white = 43)
  : gray_part_area = area_rect2 - area_white :=
by
  sorry

end gray_part_area_l281_281276


namespace train_length_is_135_l281_281452

noncomputable def length_of_train (speed_kmh : ℕ) (time_sec : ℕ) : ℕ :=
  let speed_ms := speed_kmh * 1000 / 3600
  speed_ms * time_sec

theorem train_length_is_135 :
  length_of_train 54 9 = 135 := 
by
  -- Conditions: 
  -- speed_kmh = 54
  -- time_sec = 9
  sorry

end train_length_is_135_l281_281452


namespace negation_of_exists_inequality_l281_281413

theorem negation_of_exists_inequality :
  ¬ (∃ x : ℝ, x * x + 4 * x + 5 ≤ 0) ↔ ∀ x : ℝ, x * x + 4 * x + 5 > 0 :=
by
  sorry

end negation_of_exists_inequality_l281_281413


namespace probability_at_least_one_two_l281_281103

theorem probability_at_least_one_two (dice_fair : ∀ i, 1 ≤ i ∧ i ≤ 8) (dice_count : 2):
  ∃ probability, probability = 15 / 64 := 
by
  sorry

end probability_at_least_one_two_l281_281103


namespace decreasing_interval_l281_281279

theorem decreasing_interval (f : ℝ → ℝ) (hf : ∀ x, f x = x^2 - 2 * x) :
  {x | deriv f x < 0} = {x | x < 1} :=
by
  sorry

end decreasing_interval_l281_281279


namespace roots_of_polynomial_l281_281327

theorem roots_of_polynomial : {x : ℝ | (x^2 - 5*x + 6)*(x - 1)*(x - 6) = 0} = {1, 2, 3, 6} :=
by
  -- proof goes here
  sorry

end roots_of_polynomial_l281_281327


namespace curve_is_line_l281_281583

noncomputable def curve_representation (x y : ℝ) : Prop :=
  (2 * x + 3 * y - 1) * (-1) = 0

theorem curve_is_line (x y : ℝ) (h : curve_representation x y) : 2 * x + 3 * y - 1 = 0 :=
by
  sorry

end curve_is_line_l281_281583


namespace centipede_shoes_and_socks_l281_281127

-- Define number of legs
def num_legs : ℕ := 10

-- Define the total number of items
def total_items : ℕ := 2 * num_legs

-- Define the total permutations without constraints
def total_permutations : ℕ := Nat.factorial total_items

-- Define the probability constraint for each leg
def single_leg_probability : ℚ := 1 / 2

-- Define the combined probability constraint for all legs
def all_legs_probability : ℚ := single_leg_probability ^ num_legs

-- Define the number of valid permutations (the answer to prove)
def valid_permutations : ℚ := total_permutations / all_legs_probability

theorem centipede_shoes_and_socks : valid_permutations = (Nat.factorial 20 : ℚ) / 2^10 :=
by
  -- The proof is omitted
  sorry

end centipede_shoes_and_socks_l281_281127


namespace correct_product_l281_281545

theorem correct_product (a b : ℕ)
  (h1 : 10 ≤ a ∧ a < 100)  -- a is a two-digit number
  (h2 : 0 < b)  -- b is a positive integer
  (h3 : (a % 10) * 10 + (a / 10) * b = 161)  -- Reversing the digits of a and multiplying by b yields 161
  : a * b = 224 := 
sorry

end correct_product_l281_281545


namespace solve_system_of_equations_l281_281725

theorem solve_system_of_equations :
  ∃ (x1 x2 x3 x4 x5 x6 x7 x8 : ℤ), 
    x1 + x2 + x3 = 6 ∧
    x2 + x3 + x4 = 9 ∧
    x3 + x4 + x5 = 3 ∧
    x4 + x5 + x6 = -3 ∧
    x5 + x6 + x7 = -9 ∧
    x6 + x7 + x8 = -6 ∧
    x7 + x8 + x1 = -2 ∧
    x8 + x1 + x2 = 2 ∧
    (x1, x2, x3, x4, x5, x6, x7, x8) = (1, 2, 3, 4, -4, -3, -2, -1) :=
by
  -- solution will be here
  sorry

end solve_system_of_equations_l281_281725


namespace percent_not_red_balls_l281_281480

theorem percent_not_red_balls (percent_cubes percent_red_balls : ℝ) 
  (h1 : percent_cubes = 0.3) (h2 : percent_red_balls = 0.25) : 
  (1 - percent_red_balls) * (1 - percent_cubes) = 0.525 :=
by
  sorry

end percent_not_red_balls_l281_281480


namespace final_position_relative_total_fuel_needed_l281_281596

noncomputable def navigation_records : List ℤ := [-7, 11, -6, 10, -5]

noncomputable def fuel_consumption_rate : ℝ := 0.5

theorem final_position_relative (records : List ℤ) : 
  (records.sum = 3) := by 
  sorry

theorem total_fuel_needed (records : List ℤ) (rate : ℝ) : 
  (rate * (records.map Int.natAbs).sum = 19.5) := by 
  sorry

#check final_position_relative navigation_records
#check total_fuel_needed navigation_records fuel_consumption_rate

end final_position_relative_total_fuel_needed_l281_281596


namespace simplify_fraction_l281_281406

theorem simplify_fraction : (3 : ℚ) / 462 + 17 / 42 = 95 / 231 :=
by sorry

end simplify_fraction_l281_281406


namespace right_triangle_median_l281_281041

noncomputable def median_to_hypotenuse_length (a b : ℝ) : ℝ :=
  let hypotenuse := Real.sqrt (a^2 + b^2)
  hypotenuse / 2

theorem right_triangle_median
  (a b : ℝ) (h_a : a = 3) (h_b : b = 4) :
  median_to_hypotenuse_length a b = 2.5 :=
by
  sorry

end right_triangle_median_l281_281041


namespace solve_for_b_l281_281489

noncomputable def P (x a b d c : ℝ) : ℝ := x^4 + a * x^3 + b * x^2 + d * x + c

theorem solve_for_b (a b d c : ℝ) (h1 : -a = d) (h2 : d = 1 + a + b + d + c) (h3 : c = 8) :
    b = -17 :=
by
  sorry

end solve_for_b_l281_281489


namespace distance_between_houses_l281_281636

theorem distance_between_houses
  (alice_speed : ℕ) (bob_speed : ℕ) (alice_distance : ℕ) 
  (alice_walk_time : ℕ) (bob_walk_time : ℕ)
  (alice_start : ℕ) (bob_start : ℕ)
  (bob_start_after_alice : bob_start = alice_start + 1)
  (alice_speed_eq : alice_speed = 5)
  (bob_speed_eq : bob_speed = 4)
  (alice_distance_eq : alice_distance = 25)
  (alice_walk_time_eq : alice_walk_time = alice_distance / alice_speed)
  (bob_walk_time_eq : bob_walk_time = alice_walk_time - 1)
  (bob_distance_eq : bob_walk_time = bob_walk_time * bob_speed)
  (total_distance : ℕ)
  (total_distance_eq : total_distance = alice_distance + bob_distance) :
  total_distance = 41 :=
by sorry

end distance_between_houses_l281_281636


namespace incorrect_operation_B_l281_281895

theorem incorrect_operation_B : (4 + 5)^2 ≠ 4^2 + 5^2 := 
  sorry

end incorrect_operation_B_l281_281895


namespace sin_half_alpha_l281_281184

theorem sin_half_alpha (α : ℝ) (hα1 : 0 < α ∧ α < π / 2) (hα2 : Real.cos α = (1 + Real.sqrt 5) / 4) : 
    Real.sin (α / 2) = (-1 + Real.sqrt 5) / 4 := 
by
  sorry

end sin_half_alpha_l281_281184


namespace find_a_l281_281539

theorem find_a (x y a : ℝ) (h1 : x + 2 * y = 2) (h2 : 2 * x + y = a) (h3 : x + y = 5) : a = 13 := by
  sorry

end find_a_l281_281539


namespace number_of_customers_who_tipped_is_3_l281_281860

-- Definitions of conditions
def charge_per_lawn : ℤ := 33
def lawns_mowed : ℤ := 16
def total_earnings : ℤ := 558
def tip_per_customer : ℤ := 10

-- Calculate intermediate values
def earnings_from_mowing : ℤ := lawns_mowed * charge_per_lawn
def earnings_from_tips : ℤ := total_earnings - earnings_from_mowing
def number_of_tips : ℤ := earnings_from_tips / tip_per_customer

-- Theorem stating our proof
theorem number_of_customers_who_tipped_is_3 : number_of_tips = 3 := by
  sorry

end number_of_customers_who_tipped_is_3_l281_281860


namespace salary_reduction_l281_281733

theorem salary_reduction (S : ℝ) (R : ℝ) :
  ((S - (R / 100 * S)) * 1.25 = S) → (R = 20) :=
by
  sorry

end salary_reduction_l281_281733


namespace smallest_nonfactor_product_of_factors_of_48_l281_281429

theorem smallest_nonfactor_product_of_factors_of_48 :
  ∃ a b : ℕ, a ≠ b ∧ a ∣ 48 ∧ b ∣ 48 ∧ ¬ (a * b ∣ 48) ∧ (∀ c d : ℕ, c ≠ d ∧ c ∣ 48 ∧ d ∣ 48 ∧ ¬ (c * d ∣ 48) → a * b ≤ c * d) ∧ a * b = 18 :=
sorry

end smallest_nonfactor_product_of_factors_of_48_l281_281429


namespace value_of_S6_l281_281391

theorem value_of_S6 (x : ℝ) (h : x + 1/x = 5) : x^6 + 1/x^6 = 12077 :=
by sorry

end value_of_S6_l281_281391


namespace number_of_cells_after_9_days_l281_281302

theorem number_of_cells_after_9_days : 
  let initial_cells := 4 
  let doubling_period := 3 
  let total_duration := 9 
  ∀ cells_after_9_days, cells_after_9_days = initial_cells * 2^(total_duration / doubling_period) 
  → cells_after_9_days = 32 :=
by
  sorry

end number_of_cells_after_9_days_l281_281302


namespace sin_half_angle_l281_281191

variable (α : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : cos α = (1 + Real.sqrt 5) / 4)

theorem sin_half_angle :
  sin (α / 2) = (Real.sqrt 5 - 1) / 4 :=
sorry

end sin_half_angle_l281_281191


namespace rectangular_field_length_l281_281304

theorem rectangular_field_length (w l : ℝ) (h1 : l = w + 10) (h2 : l^2 + w^2 = 22^2) : l = 22 := 
sorry

end rectangular_field_length_l281_281304


namespace exists_three_points_l281_281952

theorem exists_three_points (n : ℕ) (h : 3 ≤ n) (points : Fin n → EuclideanSpace ℝ (Fin 2))
  (distinct : ∀ i j : Fin n, i ≠ j → points i ≠ points j) :
  ∃ (A B C : Fin n),
    A ≠ B ∧ A ≠ C ∧ B ≠ C ∧ 
    1 ≤ dist (points A) (points B) / dist (points A) (points C) ∧ 
    dist (points A) (points B) / dist (points A) (points C) < (n + 1) / (n - 1) := 
sorry

end exists_three_points_l281_281952


namespace total_number_of_outcomes_two_white_one_black_outcomes_at_least_two_white_outcomes_probability_two_white_one_black_probability_at_least_two_white_l281_281215

namespace BallDrawing

-- Definitions based on the problem statement
def num_white_balls : ℕ := 4
def num_black_balls : ℕ := 5
def total_balls : ℕ := num_white_balls + num_black_balls
def num_drawn_balls : ℕ := 3

-- Problem statements (to be proven)
theorem total_number_of_outcomes : combin.choose total_balls num_drawn_balls = 84 := sorry
theorem two_white_one_black_outcomes : combin.choose num_white_balls 2 * combin.choose num_black_balls 1 = 30 := sorry
theorem at_least_two_white_outcomes : 
  combin.choose num_white_balls 2 * combin.choose num_black_balls 1 + combin.choose num_white_balls 3 * combin.choose num_black_balls 0 = 34 := sorry
theorem probability_two_white_one_black: 
  (combin.choose num_white_balls 2 * combin.choose num_black_balls 1) / (combin.choose total_balls num_drawn_balls : ℚ) = 5/14 := sorry
theorem probability_at_least_two_white: 
  ((combin.choose num_white_balls 2 * combin.choose num_black_balls 1 + combin.choose num_white_balls 3 * combin.choose num_black_balls 0) 
   / (combin.choose total_balls num_drawn_balls : ℚ) = 17/42) := sorry

end BallDrawing

end total_number_of_outcomes_two_white_one_black_outcomes_at_least_two_white_outcomes_probability_two_white_one_black_probability_at_least_two_white_l281_281215


namespace function_positivity_range_l281_281957

theorem function_positivity_range (m x : ℝ): 
  (∀ x, (2 * x^2 + (4 - m) * x + 4 - m > 0) ∨ (m * x > 0)) ↔ m < 4 :=
sorry

end function_positivity_range_l281_281957


namespace jack_leftover_money_l281_281064

theorem jack_leftover_money :
  let saved_money_base8 : ℕ := 3 * 8^3 + 7 * 8^2 + 7 * 8^1 + 7 * 8^0
  let ticket_cost_base10 : ℕ := 1200
  saved_money_base8 - ticket_cost_base10 = 847 :=
by
  let saved_money_base8 := 3 * 8^3 + 7 * 8^2 + 7 * 8^1 + 7 * 8^0
  let ticket_cost_base10 := 1200
  show saved_money_base8 - ticket_cost_base10 = 847
  sorry

end jack_leftover_money_l281_281064


namespace arithmetic_sequence_a8_value_l281_281387

theorem arithmetic_sequence_a8_value
  (a : ℕ → ℤ) 
  (h1 : a 1 + 3 * a 8 + a 15 = 120)
  (h2 : a 1 + a 15 = 2 * a 8) :
  a 8 = 24 := 
sorry

end arithmetic_sequence_a8_value_l281_281387


namespace parabola_relationship_l281_281717

theorem parabola_relationship (a : ℝ) (h : a < 0) :
  let y1 := 24 * a - 7
  let y2 := -7
  let y3 := 3 * a - 7
  y1 < y3 ∧ y3 < y2 :=
by
  let y1 := 24 * a - 7
  let y2 := -7
  let y3 := 3 * a - 7
  sorry

end parabola_relationship_l281_281717


namespace area_ratio_l281_281339

-- Definitions for the geometric entities
structure Point where
  x : ℚ
  y : ℚ

def A : Point := ⟨0, 0⟩
def B : Point := ⟨0, 4⟩
def C : Point := ⟨2, 4⟩
def D : Point := ⟨2, 0⟩
def E : Point := ⟨1, 2⟩  -- Midpoint of BD
def F : Point := ⟨6 / 5, 0⟩  -- Given DF = 2/5 DA

-- Function to calculate the area of a triangle given its vertices
def triangle_area (P Q R : Point) : ℚ :=
  (1 / 2) * abs ((Q.x - P.x) * (R.y - P.y) - (R.x - P.x) * (Q.y - P.y))

-- Function to calculate the sum of the area of two triangles
def quadrilateral_area (P Q R S : Point) : ℚ :=
  triangle_area P Q R + triangle_area P R S

-- Prove the ratio of the areas
theorem area_ratio : 
  triangle_area D F E / quadrilateral_area A B E F = 4 / 13 := 
by {
  sorry
}

end area_ratio_l281_281339


namespace Suzanna_rides_8_miles_in_40_minutes_l281_281575

theorem Suzanna_rides_8_miles_in_40_minutes :
  (∀ n : ℕ, Suzanna_distance_in_n_minutes = (n / 10) * 2) → Suzanna_distance_in_40_minutes = 8 :=
by
  sorry

-- Definitions for Suzanna's distance conditions
def Suzanna_distance_in_n_minutes (n : ℕ) : ℕ := (n / 10) * 2

noncomputable def Suzanna_distance_in_40_minutes := Suzanna_distance_in_n_minutes 40

#check Suzanna_rides_8_miles_in_40_minutes

end Suzanna_rides_8_miles_in_40_minutes_l281_281575


namespace cloth_sales_value_l281_281606

theorem cloth_sales_value (commission_rate : ℝ) (commission : ℝ) (total_sales : ℝ) 
  (h1: commission_rate = 2.5)
  (h2: commission = 18)
  (h3: total_sales = commission / (commission_rate / 100)):
  total_sales = 720 := by
  sorry

end cloth_sales_value_l281_281606


namespace find_a_l281_281232

noncomputable def A : Set ℝ := {x | x^2 - 8*x + 15 = 0}
noncomputable def B (a : ℝ) : Set ℝ := {x | x * a = 1}
axiom A_is_B (a : ℝ) : A ∩ B a = B a → (a = 0) ∨ (a = 1/3) ∨ (a = 1/5)

-- statement to prove
theorem find_a (a : ℝ) (h : A ∩ B a = B a) : (a = 0) ∨ (a = 1/3) ∨ (a = 1/5) :=
by 
  apply A_is_B
  assumption

end find_a_l281_281232


namespace relationship_between_y_l281_281510

theorem relationship_between_y
  (m y₁ y₂ y₃ : ℝ)
  (hA : y₁ = -(-1)^2 + 2 * -1 + m)
  (hB : y₂ = -(1)^2 + 2 * 1 + m)
  (hC : y₃ = -(2)^2 + 2 * 2 + m) :
  y₁ < y₃ ∧ y₃ < y₂ :=
sorry

end relationship_between_y_l281_281510


namespace find_e_l281_281453

variable (p j t e : ℝ)

def condition1 : Prop := j = 0.75 * p
def condition2 : Prop := j = 0.80 * t
def condition3 : Prop := t = p * (1 - e / 100)

theorem find_e (h1 : condition1 p j)
               (h2 : condition2 j t)
               (h3 : condition3 t e p) : e = 6.25 :=
by sorry

end find_e_l281_281453


namespace value_of_b_l281_281340

theorem value_of_b (b : ℝ) : 
  (∃ (x : ℝ), x^2 + b * x - 45 = 0 ∧ x = -4) →
  b = -29 / 4 :=
by
  -- Introduce the condition and rewrite it properly
  intro h
  obtain ⟨x, hx1, hx2⟩ := h
  -- Proceed with assumption that we have the condition and need to prove the statement
  sorry

end value_of_b_l281_281340


namespace smallest_nonfactor_product_of_factors_of_48_l281_281430

theorem smallest_nonfactor_product_of_factors_of_48 :
  ∃ a b : ℕ, a ≠ b ∧ a ∣ 48 ∧ b ∣ 48 ∧ ¬ (a * b ∣ 48) ∧ (∀ c d : ℕ, c ≠ d ∧ c ∣ 48 ∧ d ∣ 48 ∧ ¬ (c * d ∣ 48) → a * b ≤ c * d) ∧ a * b = 18 :=
sorry

end smallest_nonfactor_product_of_factors_of_48_l281_281430


namespace sum_of_numbers_l281_281090

theorem sum_of_numbers (x : ℝ) (h : x^2 + (2 * x)^2 + (4 * x)^2 = 4725) : 
  x + 2 * x + 4 * x = 105 := 
sorry

end sum_of_numbers_l281_281090


namespace chewbacca_pack_size_l281_281027

/-- Given Chewbacca has 20 pieces of cherry gum and 30 pieces of grape gum,
if losing one pack of cherry gum keeps the ratio of cherry to grape gum the same
as when finding 5 packs of grape gum, determine the number of pieces x in each 
complete pack of gum. We show that x = 14. -/
theorem chewbacca_pack_size :
  ∃ (x : ℕ), (20 - x) * (30 + 5 * x) = 20 * 30 ∧ ∀ (y : ℕ), (20 - y) * (30 + 5 * y) = 600 → y = 14 :=
by
  sorry

end chewbacca_pack_size_l281_281027


namespace contrapositive_is_false_l281_281261

-- Define the property of collinearity between two vectors
def collinear (a b : ℝ × ℝ) : Prop := 
  ∃ k : ℝ, a = k • b

-- Define the property of vectors having the same direction
def same_direction (a b : ℝ × ℝ) : Prop := 
  ∃ k : ℝ, k > 0 ∧ a = k • b

-- Original proposition in Lean statement
def original_proposition (a b : ℝ × ℝ) : Prop :=
  collinear a b → same_direction a b

-- Contrapositive of the original proposition
def contrapositive_proposition (a b : ℝ × ℝ) : Prop :=
  ¬ same_direction a b → ¬ collinear a b

-- The proof goal that the contrapositive is false
theorem contrapositive_is_false (a b : ℝ × ℝ) :
  (contrapositive_proposition a b) = false :=
sorry

end contrapositive_is_false_l281_281261


namespace greatest_x_value_l281_281938

theorem greatest_x_value :
  ∃ x, (∀ y, (y ≠ 6) → (y^2 - y - 30) / (y - 6) = 2 / (y + 4) → y ≤ x) ∧ (x^2 - x - 30) / (x - 6) = 2 / (x + 4) ∧ x ≠ 6 ∧ x = -3 :=
sorry

end greatest_x_value_l281_281938


namespace possible_values_of_ratio_l281_281494

theorem possible_values_of_ratio (a d : ℝ) (h : a ≠ 0) (h_eq : a^2 - 6 * a * d + 8 * d^2 = 0) : 
  ∃ x : ℝ, (x = 1/2 ∨ x = 1/4) ∧ x = d/a :=
by
  sorry

end possible_values_of_ratio_l281_281494


namespace John_nap_hours_l281_281701

def weeksInDays (d : ℕ) : ℕ := d / 7
def totalNaps (weeks : ℕ) (naps_per_week : ℕ) : ℕ := weeks * naps_per_week
def totalNapHours (naps : ℕ) (hours_per_nap : ℕ) : ℕ := naps * hours_per_nap

theorem John_nap_hours (d : ℕ) (naps_per_week : ℕ) (hours_per_nap : ℕ) (days_per_week : ℕ) : 
  d = 70 →
  naps_per_week = 3 →
  hours_per_nap = 2 →
  days_per_week = 7 →
  totalNapHours (totalNaps (weeksInDays d) naps_per_week) hours_per_nap = 60 :=
by
  intros h1 h2 h3 h4
  unfold weeksInDays
  unfold totalNaps
  unfold totalNapHours
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end John_nap_hours_l281_281701


namespace prob_at_least_one_2_in_two_8_sided_dice_l281_281105

/-- Probability of getting at least one 2 when rolling two 8-sided dice -/
theorem prob_at_least_one_2_in_two_8_sided_dice : 
  let total_outcomes := 64
  let favorable_outcomes := 15
  (favorable_outcomes : ℝ) / (total_outcomes : ℝ) = 15 / 64 :=
by
  sorry

end prob_at_least_one_2_in_two_8_sided_dice_l281_281105


namespace value_of_expression_l281_281633

theorem value_of_expression : (7^2 - 6^2)^4 = 28561 :=
by sorry

end value_of_expression_l281_281633


namespace num_5_letter_words_with_at_least_two_consonants_l281_281529

open Finset

def letters : Finset Char := {'A', 'B', 'C', 'D', 'E', 'F'}
def consonants : Finset Char := {'B', 'C', 'D', 'F'}
def vowels : Finset Char := {'A', 'E'}

def total_5_letter_words : ℕ := 6^5
def words_with_0_consonants : ℕ := 2^5
def words_with_1_consonant : ℕ := 5 * 4 * 2^4

theorem num_5_letter_words_with_at_least_two_consonants : 
  total_5_letter_words - (words_with_0_consonants + words_with_1_consonant) = 7424 := by
  sorry

end num_5_letter_words_with_at_least_two_consonants_l281_281529


namespace smallest_product_not_factor_of_48_l281_281440

theorem smallest_product_not_factor_of_48 (a b : ℕ) (h1 : a ≠ b) (h2 : a ∣ 48) (h3 : b ∣ 48) (h4 : ¬ (a * b ∣ 48)) : a * b = 32 :=
sorry

end smallest_product_not_factor_of_48_l281_281440


namespace sum_of_floor_sqrt_and_neg_floor_sqrt_l281_281284

theorem sum_of_floor_sqrt_and_neg_floor_sqrt (n : ℕ) (h : n = 1989 * 1990) :
  (∑ x in Finset.range (n + 1), (Int.floor (Real.sqrt x) + Int.floor (-(Real.sqrt x)))) = -3956121 :=
by
  -- placeholder proof
  sorry

end sum_of_floor_sqrt_and_neg_floor_sqrt_l281_281284


namespace money_left_l281_281552

-- Conditions
def initial_savings : ℤ := 6000
def spent_on_flight : ℤ := 1200
def spent_on_hotel : ℤ := 800
def spent_on_food : ℤ := 3000

-- Total spent
def total_spent : ℤ := spent_on_flight + spent_on_hotel + spent_on_food

-- Prove that the money left is $1,000
theorem money_left (h1 : initial_savings = 6000)
                   (h2 : spent_on_flight = 1200)
                   (h3 : spent_on_hotel = 800)
                   (h4 : spent_on_food = 3000) :
                   initial_savings - total_spent = 1000 :=
by
  -- Insert proof steps here
  sorry

end money_left_l281_281552


namespace max_value_at_2_l281_281192

noncomputable def f (x : ℝ) : ℝ := -x^3 + 12 * x

theorem max_value_at_2 : ∃ a : ℝ, (∀ x : ℝ, f x ≤ f a) ∧ a = 2 := 
by
  sorry

end max_value_at_2_l281_281192


namespace cosine_theorem_l281_281752

theorem cosine_theorem (a b c : ℝ) (A : ℝ) (hA : 0 < A) (hA_lt_pi : A < π) :
  a^2 = b^2 + c^2 - 2 * b * c * Real.cos A := sorry

end cosine_theorem_l281_281752


namespace simplify_expression_l281_281873

theorem simplify_expression (x : ℝ) : (3 * x + 20) + (50 * x + 25) = 53 * x + 45 :=
by
  sorry

end simplify_expression_l281_281873


namespace sin_half_angle_l281_281182

theorem sin_half_angle (α : ℝ) (h₁ : 0 < α ∧ α < π / 2) (h₂ : real.cos α = (1 + real.sqrt 5) / 4) :
  real.sin (α / 2) = (real.sqrt 5 - 1) / 4 :=
by
  sorry

end sin_half_angle_l281_281182


namespace hannah_total_spent_l281_281679

-- Definitions based on conditions
def sweatshirts_bought : ℕ := 3
def t_shirts_bought : ℕ := 2
def cost_per_sweatshirt : ℕ := 15
def cost_per_t_shirt : ℕ := 10

-- Definition of the theorem that needs to be proved
theorem hannah_total_spent : 
  (sweatshirts_bought * cost_per_sweatshirt + t_shirts_bought * cost_per_t_shirt) = 65 :=
by
  sorry

end hannah_total_spent_l281_281679


namespace problem_l281_281827

def setA : Set ℝ := {x : ℝ | 0 < x ∧ x ≤ 2}
def setB : Set ℝ := {x : ℝ | x ≤ 3}

theorem problem : setA ∩ setB = setA := sorry

end problem_l281_281827


namespace buns_per_student_correct_l281_281355

variables (packages_per_bun : Nat) (num_packages : Nat)
           (num_classes : Nat) (students_per_class : Nat)

def total_buns (packages_per_bun : Nat) (num_packages : Nat) : Nat :=
  packages_per_bun * num_packages

def total_students (num_classes : Nat) (students_per_class : Nat) : Nat :=
  num_classes * students_per_class

def buns_per_student (total_buns : Nat) (total_students : Nat) : Nat :=
  total_buns / total_students

theorem buns_per_student_correct :
  packages_per_bun = 8 →
  num_packages = 30 →
  num_classes = 4 →
  students_per_class = 30 →
  buns_per_student (total_buns packages_per_bun num_packages) 
                  (total_students num_classes students_per_class) = 2 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end buns_per_student_correct_l281_281355


namespace part_I_part_II_l281_281677

noncomputable def f (x : ℝ) : ℝ :=
  2 * (Real.cos x) ^ 2 - Real.sin (2 * x - (7 * Real.pi / 6))

theorem part_I :
  (∀ x, f x ≤ 2) ∧ (∃ x, f x = 2 ∧ ∃ k : ℤ, x = k * Real.pi + Real.pi / 6) :=
by
  sorry

theorem part_II (A a b c : ℝ) (h1 : f A = 3 / 2) (h2 : b + c = 2) :
  a >= 1 :=
by
  sorry

end part_I_part_II_l281_281677


namespace carpet_needed_l281_281015

def room_length : ℕ := 15
def room_width : ℕ := 10
def ft2_to_yd2 : ℕ := 9

theorem carpet_needed :
  (room_length * room_width / ft2_to_yd2).ceil = 17 :=
by
  sorry

end carpet_needed_l281_281015


namespace remainder_twice_sum_first_150_mod_10000_eq_2650_l281_281599

theorem remainder_twice_sum_first_150_mod_10000_eq_2650 :
  let n := 150
  let S := n * (n + 1) / 2  -- Sum of first 150 numbers
  let result := 2 * S
  result % 10000 = 2650 :=
by
  sorry -- proof not required

end remainder_twice_sum_first_150_mod_10000_eq_2650_l281_281599


namespace cartesian_eq_C1_cartesian_eq_C2_distance_AB_l281_281388

theorem cartesian_eq_C1 :
  ∀ α : ℝ, let x := 2 * sqrt 5 * cos α in
             let y := 2 * sin α in
             (x / (2 * sqrt 5))^2 + (y / 2)^2 = 1 :=
by sorry

theorem cartesian_eq_C2 :
  ∀ (ρ θ : ℝ), let x := ρ * cos θ in
                let y := ρ * sin θ in
                ρ^2 + 4 * ρ * cos θ - 2 * ρ * sin θ + 4 = 0 → 
                (x + 2)^2 + (y - 1)^2 = 1 :=
by sorry

theorem distance_AB :
  let C2_eq := (x + 2)^2 + (y - 1)^2 = 1 in
  let L_focus := (-4, 0) in
  ∀ t1 t2 : ℝ, t1 * t2 = 4 ∧ t1 + t2 = 3 * sqrt 2 →
  ∃ x1 y1 x2 y2, 
    ((x1 = -4 + sqrt 2 / 2 * t1 ∧ y1 = sqrt 2 / 2 * t1) ∧ 
     (x2 = -4 + sqrt 2 / 2 * t2 ∧ y2 = sqrt 2 / 2 * t2) ∧ 
    (x1, y1) ∈ C2_eq ∧ (x2, y2) ∈ C2_eq) → 
    |t1 - t2| = sqrt 2 :=
by sorry

end cartesian_eq_C1_cartesian_eq_C2_distance_AB_l281_281388


namespace inequality_proof_l281_281513

theorem inequality_proof (a b c : ℝ) 
    (ha : a > 1) (hb : b > 1) (hc : c > 1) :
    (a^2 / (b - 1)) + (b^2 / (c - 1)) + (c^2 / (a - 1)) ≥ 12 :=
by {
    sorry
}

end inequality_proof_l281_281513


namespace number_of_valid_triples_l281_281201

theorem number_of_valid_triples :
  ∃ (count : ℕ), count = 3 ∧
  ∀ (x y z : ℕ), 0 < x → 0 < y → 0 < z →
  Nat.lcm x y = 120 → Nat.lcm y z = 1000 → Nat.lcm x z = 480 →
  (∃ (u v w : ℕ), u = x ∧ v = y ∧ w = z ∧ count = 3) :=
by
  sorry

end number_of_valid_triples_l281_281201


namespace sum_of_four_terms_l281_281544

theorem sum_of_four_terms (a d : ℕ) (h1 : a + d > a) (h2 : a + 2 * d > a + d)
  (h3 : (a + 2 * d) * (a + 2 * d) = (a + d) * (a + 3 * d)) (h4 : (a + 3 * d) - a = 30) :
  a + (a + d) + (a + 2 * d) + (a + 3 * d) = 129 :=
sorry

end sum_of_four_terms_l281_281544


namespace stratified_sampling_junior_teachers_l281_281911

theorem stratified_sampling_junior_teachers 
    (total_teachers : ℕ) (senior_teachers : ℕ) 
    (intermediate_teachers : ℕ) (junior_teachers : ℕ) 
    (sample_size : ℕ) 
    (H1 : total_teachers = 200)
    (H2 : senior_teachers = 20)
    (H3 : intermediate_teachers = 100)
    (H4 : junior_teachers = 80) 
    (H5 : sample_size = 50)
    : (junior_teachers * sample_size / total_teachers = 20) := 
  by 
    sorry

end stratified_sampling_junior_teachers_l281_281911


namespace compute_expression_l281_281770

theorem compute_expression : 
  Real.sqrt 8 - (2017 - Real.pi)^0 - 4^(-1 : Int) + (-1/2)^2 = 2 * Real.sqrt 2 - 1 := 
by 
  sorry

end compute_expression_l281_281770


namespace cyclists_travel_same_distance_l281_281273

-- Define constants for speeds
def v1 := 12   -- speed of the first cyclist in km/h
def v2 := 16   -- speed of the second cyclist in km/h
def v3 := 24   -- speed of the third cyclist in km/h

-- Define the known total time
def total_time := 3  -- total time in hours

-- Hypothesis: Prove that the distance traveled by each cyclist is 16 km
theorem cyclists_travel_same_distance (d : ℚ) : 
  (v1 * (total_time * 3 / 13)) = d ∧
  (v2 * (total_time * 4 / 13)) = d ∧
  (v3 * (total_time * 6 / 13)) = d ∧
  d = 16 :=
by
  sorry

end cyclists_travel_same_distance_l281_281273


namespace tan_960_eq_sqrt_3_l281_281093

theorem tan_960_eq_sqrt_3 : Real.tan (960 * Real.pi / 180) = Real.sqrt 3 := by
  sorry

end tan_960_eq_sqrt_3_l281_281093


namespace value_of_expression_l281_281532

variable (m : ℝ)

theorem value_of_expression (h : 2 * m^2 + 3 * m - 1 = 0) : 
  4 * m^2 + 6 * m - 2019 = -2017 :=
by
  sorry

end value_of_expression_l281_281532


namespace taco_truck_revenue_l281_281472

-- Conditions
def price_of_soft_taco : ℕ := 2
def price_of_hard_taco : ℕ := 5
def family_soft_tacos : ℕ := 3
def family_hard_tacos : ℕ := 4
def other_customers : ℕ := 10
def soft_tacos_per_other_customer : ℕ := 2

-- Calculation
def total_soft_tacos : ℕ := family_soft_tacos + other_customers * soft_tacos_per_other_customer
def revenue_from_soft_tacos : ℕ := total_soft_tacos * price_of_soft_taco
def revenue_from_hard_tacos : ℕ := family_hard_tacos * price_of_hard_taco
def total_revenue : ℕ := revenue_from_soft_tacos + revenue_from_hard_tacos

-- The proof problem
theorem taco_truck_revenue : total_revenue = 66 := 
by 
-- The proof should go here
sorry

end taco_truck_revenue_l281_281472


namespace value_of_a_l281_281230

def A := { x : ℝ | x^2 - 8*x + 15 = 0 }
def B (a : ℝ) := { x : ℝ | x * a - 1 = 0 }

theorem value_of_a (a : ℝ) : (∀ x, x ∈ B a → x ∈ A) ↔ (a = 0 ∨ a = 1/3 ∨ a = 1/5) :=
by sorry

end value_of_a_l281_281230


namespace min_distance_origin_to_intersections_l281_281036

theorem min_distance_origin_to_intersections (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (hline : (1 : ℝ)/a + 4/b = 1) :
  |(0 : ℝ) - a| + |(0 : ℝ) - b| = 9 :=
sorry

end min_distance_origin_to_intersections_l281_281036


namespace find_n_value_l281_281728

theorem find_n_value :
  ∃ m n : ℝ, (4 * x^2 + 8 * x - 448 = 0 → (x + m)^2 = n) ∧ n = 113 :=
by
  sorry

end find_n_value_l281_281728


namespace brittany_age_when_returning_l281_281316

def rebecca_age : ℕ := 25
def age_difference : ℕ := 3
def vacation_duration : ℕ := 4

theorem brittany_age_when_returning : (rebecca_age + age_difference + vacation_duration) = 32 := by
  sorry

end brittany_age_when_returning_l281_281316


namespace solution_set_of_quadratic_inequality_l281_281416

theorem solution_set_of_quadratic_inequality (x : ℝ) : 
  x^2 + 3*x - 4 < 0 ↔ -4 < x ∧ x < 1 :=
sorry

end solution_set_of_quadratic_inequality_l281_281416


namespace binomial_alternating_sum_eq_neg_2_pow_50_l281_281663

open BigOperators

noncomputable def sum_of_binomials: ℤ :=
  (∑ k in (Finset.range 51).filter (λ k, Even k), (-1)^k * (Nat.choose 101 k))

theorem binomial_alternating_sum_eq_neg_2_pow_50 :
  sum_of_binomials = -2^50 :=
sorry

end binomial_alternating_sum_eq_neg_2_pow_50_l281_281663


namespace train_length_proof_l281_281473

def train_length_crosses_bridge (train_speed_kmh : ℕ) (bridge_length_m : ℕ) (crossing_time_s : ℕ) : ℕ :=
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  let distance := train_speed_ms * crossing_time_s
  distance - bridge_length_m

theorem train_length_proof : 
  train_length_crosses_bridge 72 150 20 = 250 :=
by
  let train_speed_kmh := 72
  let bridge_length_m := 150
  let crossing_time_s := 20
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  let distance := train_speed_ms * crossing_time_s
  have h : distance = 400 := by sorry
  have h_eq : distance - bridge_length_m = 250 := by sorry
  exact h_eq

end train_length_proof_l281_281473


namespace brittany_age_when_returning_l281_281314

def rebecca_age : ℕ := 25
def age_difference : ℕ := 3
def vacation_duration : ℕ := 4

theorem brittany_age_when_returning : (rebecca_age + age_difference + vacation_duration) = 32 := by
  sorry

end brittany_age_when_returning_l281_281314


namespace value_of_f_neg1_l281_281443

def f (x : ℤ) : ℤ := x^2 - 2 * x

theorem value_of_f_neg1 : f (-1) = 3 := by
  sorry

end value_of_f_neg1_l281_281443


namespace factorization_of_polynomial_l281_281650

theorem factorization_of_polynomial : 
  (x^2 + 6 * x + 9 - 64 * x^4) = (-8 * x^2 + x + 3) * (8 * x^2 + x + 3) :=
by sorry

end factorization_of_polynomial_l281_281650


namespace time_after_2004_hours_l281_281092

variable (h : ℕ) 

-- Current time is represented as an integer from 0 to 11 (9 o'clock).
def current_time : ℕ := 9

-- 12-hour clock cycles every 12 hours.
def cycle : ℕ := 12

-- Time after 2004 hours.
def hours_after : ℕ := 2004

-- Proof statement
theorem time_after_2004_hours (h : ℕ) :
  (current_time + hours_after) % cycle = current_time := 
sorry

end time_after_2004_hours_l281_281092


namespace smallest_x_plus_y_l281_281796

theorem smallest_x_plus_y (x y : ℕ) (h_xy_not_equal : x ≠ y) (h_condition : (1 : ℚ) / x + 1 / y = 1 / 15) :
  x + y = 64 :=
sorry

end smallest_x_plus_y_l281_281796


namespace lada_vs_elevator_l281_281555

def Lada_speed_ratio (V U : ℝ) (S : ℝ) : Prop :=
  (∃ t_wait t_wait' : ℝ,
  ((t_wait = 3*S/U - 3*S/V) ∧ (t_wait' = 7*S/(2*U) - 7*S/V)) ∧
   (t_wait' = 3 * t_wait)) →
  U = 11/4 * V

theorem lada_vs_elevator (V U : ℝ) (S : ℝ) : Lada_speed_ratio V U S :=
sorry

end lada_vs_elevator_l281_281555


namespace butterfat_mixture_l281_281120

theorem butterfat_mixture (x : ℝ) :
  (0.10 * x + 0.30 * 8 = 0.20 * (x + 8)) → x = 8 :=
by
  intro h
  sorry

end butterfat_mixture_l281_281120


namespace original_height_l281_281291

theorem original_height (total_travel : ℝ) (h : ℝ) (half: h/2 = (1/2 * h)): 
  (total_travel = h + 2 * (h / 2) + 2 * (h / 4)) → total_travel = 260 → h = 104 :=
by
  intro travel_eq
  intro travel_value
  sorry

end original_height_l281_281291


namespace sum_mean_median_mode_eq_38_over_9_l281_281601

-- Define the list of numbers
def numbers : List ℕ := [4, 2, 5, 4, 0, 4, 1, 0, 0]

-- Define functions to find the mean, median, and mode
def mean (l : List ℕ) : Rat :=
  let s := l.sum 
  let n := l.length
  s / n

def median (l : List ℕ) : ℕ :=
  match l.sort.batches with
  | [] => 0
  | mid :: _ => mid

def mode (l : List ℕ) : ℕ :=
  l.foldl (λ acc x => max (l.count x) (l.count acc)) 0

-- Define a function to sum the mean, median, and mode
def sum_mean_median_mode (l : List ℕ) : Rat :=
  mean l + (median l) + (mode l)

-- Statement to prove
theorem sum_mean_median_mode_eq_38_over_9 : 
  sum_mean_median_mode numbers = 38 / 9 :=
by
  -- This is where the proof would go, but for now we use sorry.
  sorry

end sum_mean_median_mode_eq_38_over_9_l281_281601


namespace distance_walked_north_l281_281300

-- Definition of the problem parameters
def distance_west : ℝ := 10
def total_distance : ℝ := 14.142135623730951

-- The theorem stating the result
theorem distance_walked_north (x : ℝ) (h : distance_west ^ 2 + x ^ 2 = total_distance ^ 2) : x = 10 :=
by sorry

end distance_walked_north_l281_281300


namespace parallel_lines_slope_l281_281200

theorem parallel_lines_slope {a : ℝ} (h : -a / 3 = -2 / 3) : a = 2 := 
by
  sorry

end parallel_lines_slope_l281_281200


namespace remainder_when_divided_by_24_l281_281002

theorem remainder_when_divided_by_24 (m k : ℤ) (h : m = 288 * k + 47) : m % 24 = 23 :=
by
  sorry

end remainder_when_divided_by_24_l281_281002


namespace how_many_correct_l281_281359

def calc1 := (2 * Real.sqrt 3) * (3 * Real.sqrt 3) = 6 * Real.sqrt 3
def calc2 := Real.sqrt 2 + Real.sqrt 3 = Real.sqrt 5
def calc3 := (5 * Real.sqrt 5) - (2 * Real.sqrt 2) = 3 * Real.sqrt 3
def calc4 := (Real.sqrt 2) / (Real.sqrt 3) = (Real.sqrt 6) / 3

theorem how_many_correct : (¬ calc1) ∧ (¬ calc2) ∧ (¬ calc3) ∧ calc4 → 1 = 1 :=
by { sorry }

end how_many_correct_l281_281359


namespace probability_at_least_one_two_l281_281101

theorem probability_at_least_one_two (dice_fair : ∀ i, 1 ≤ i ∧ i ≤ 8) (dice_count : 2):
  ∃ probability, probability = 15 / 64 := 
by
  sorry

end probability_at_least_one_two_l281_281101


namespace time_first_tap_to_fill_cistern_l281_281906

-- Defining the conditions
axiom second_tap_empty_time : ℝ
axiom combined_tap_fill_time : ℝ
axiom second_tap_rate : ℝ
axiom combined_tap_rate : ℝ

-- Specifying the given conditions
def problem_conditions :=
  second_tap_empty_time = 8 ∧
  combined_tap_fill_time = 8 ∧
  second_tap_rate = 1 / 8 ∧
  combined_tap_rate = 1 / 8

-- Defining the problem statement
theorem time_first_tap_to_fill_cistern :
  problem_conditions →
  (∃ T : ℝ, (1 / T - 1 / 8 = 1 / 8) ∧ T = 4) :=
by
  intro h
  sorry

end time_first_tap_to_fill_cistern_l281_281906


namespace emily_expenditure_l281_281481

-- Define the conditions
def price_per_flower : ℕ := 3
def roses_bought : ℕ := 2
def daisies_bought : ℕ := 2

-- Total flowers bought
def total_flowers (roses daisies : ℕ) : ℕ :=
  roses + daisies

-- Define the cost function
def cost (flowers price_per_flower : ℕ) : ℕ :=
  flowers * price_per_flower

-- Theorem to prove the total expenditure
theorem emily_expenditure : 
  cost (total_flowers roses_bought daisies_bought) price_per_flower = 12 :=
by
  sorry

end emily_expenditure_l281_281481


namespace complement_intersection_eq_l281_281989

open Set

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 3, 4}

theorem complement_intersection_eq :
  U \ (A ∩ B) = {1, 4, 5} := by
  sorry

end complement_intersection_eq_l281_281989


namespace polygon_area_is_400_l281_281330

-- Definition of the points and polygon
def Point := (ℝ × ℝ)
def Polygon := List Point

def points : List Point := [(0, 0), (20, 0), (20, 20), (0, 20), (10, 0), (20, 10), (10, 20), (0, 10)]

def polygon : Polygon := [(0,0), (10,0), (20,10), (20,20), (10,20), (0,10), (0,0)]

-- Function to calculate the area of the polygon
noncomputable def polygon_area (p : Polygon) : ℝ := 
  -- Assume we have the necessary function to calculate the area of a polygon given a list of vertices
  sorry

-- Theorem statement: The area of the given polygon is 400
theorem polygon_area_is_400 : polygon_area polygon = 400 := sorry

end polygon_area_is_400_l281_281330


namespace find_smaller_number_l281_281565

def one_number_is_11_more_than_3times_another (x y : ℕ) : Prop :=
  y = 3 * x + 11

def their_sum_is_55 (x y : ℕ) : Prop :=
  x + y = 55

theorem find_smaller_number (x y : ℕ) (h1 : one_number_is_11_more_than_3times_another x y) (h2 : their_sum_is_55 x y) :
  x = 11 :=
by
  -- The proof will be inserted here
  sorry

end find_smaller_number_l281_281565


namespace perpendicular_line_directional_vector_l281_281682

theorem perpendicular_line_directional_vector
  (l1 : ℝ → ℝ → Prop)
  (l2 : ℝ → ℝ → Prop)
  (perpendicular : ∀ x y, l1 x y ↔ l2 y (-x))
  (l2_eq : ∀ x y, l2 x y ↔ 2 * x + 5 * y = 1) :
  ∃ d1 d2, (d1, d2) = (5, -2) ∧ (d1 * 2 + d2 * 5 = 0) :=
by
  sorry

end perpendicular_line_directional_vector_l281_281682


namespace sin_half_angle_l281_281164

theorem sin_half_angle (α : ℝ) (h0 : 0 < α ∧ α < π / 2) (h1 : cos α = (1 + real.sqrt 5) / 4) :
  sin (α / 2) = (-1 + real.sqrt 5) / 4 :=
by sorry

end sin_half_angle_l281_281164


namespace find_x_range_l281_281670

noncomputable def p (x : ℝ) := x^2 + 2*x - 3 > 0
noncomputable def q (x : ℝ) := 1/(3 - x) > 1

theorem find_x_range (x : ℝ) : (¬q x ∧ p x) → (x ≥ 3 ∨ (1 < x ∧ x ≤ 2) ∨ x < -3) :=
by
  intro h
  sorry

end find_x_range_l281_281670


namespace probability_at_least_one_two_l281_281102

theorem probability_at_least_one_two (dice_fair : ∀ i, 1 ≤ i ∧ i ≤ 8) (dice_count : 2):
  ∃ probability, probability = 15 / 64 := 
by
  sorry

end probability_at_least_one_two_l281_281102


namespace find_abc_l281_281236

theorem find_abc :
  ∃ (a b c : ℤ), a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ 
  a + b + c = 30 ∧
  (1/a + 1/b + 1/c + 450/(a*b*c) = 1) ∧ 
  a*b*c = 1912 :=
sorry

end find_abc_l281_281236


namespace ramu_profit_percent_l281_281570

-- Definitions of the given conditions
def usd_to_inr (usd : ℤ) : ℤ := usd * 45 / 10
def eur_to_inr (eur : ℤ) : ℤ := eur * 567 / 100
def jpy_to_inr (jpy : ℤ) : ℤ := jpy * 1667 / 10000

def cost_of_car_in_inr := usd_to_inr 10000
def engine_repair_cost_in_inr := eur_to_inr 3000
def bodywork_repair_cost_in_inr := jpy_to_inr 150000
def total_cost_in_inr := cost_of_car_in_inr + engine_repair_cost_in_inr + bodywork_repair_cost_in_inr

def selling_price_in_inr : ℤ := 80000
def profit_or_loss_in_inr : ℤ := selling_price_in_inr - total_cost_in_inr

-- Profit percent calculation
def profit_percent (profit_or_loss total_cost : ℤ) : ℚ := (profit_or_loss : ℚ) / (total_cost : ℚ) * 100

-- The theorem stating the mathematically equivalent problem
theorem ramu_profit_percent :
  profit_percent profit_or_loss_in_inr total_cost_in_inr = -8.06 := by
  sorry

end ramu_profit_percent_l281_281570


namespace eggs_per_group_l281_281244

-- Conditions
def total_eggs : ℕ := 9
def total_groups : ℕ := 3

-- Theorem statement
theorem eggs_per_group : total_eggs / total_groups = 3 :=
sorry

end eggs_per_group_l281_281244


namespace solve_equation_solve_inequality_system_l281_281610

theorem solve_equation (x : ℝ) : x^2 - 2 * x - 4 = 0 ↔ x = 1 + Real.sqrt 5 ∨ x = 1 - Real.sqrt 5 :=
by
  sorry

theorem solve_inequality_system (x : ℝ) : (4 * (x - 1) < x + 2) ∧ ((x + 7) / 3 > x) ↔ x < 2 :=
by
  sorry

end solve_equation_solve_inequality_system_l281_281610


namespace volume_inequality_holds_l281_281119

def volume (x : ℕ) : ℤ :=
  (x^2 - 16) * (x^3 + 25)

theorem volume_inequality_holds :
  ∃ (n : ℕ), n = 1 ∧ ∃ x : ℕ, volume x < 1000 ∧ (x - 4) > 0 :=
by
  sorry

end volume_inequality_holds_l281_281119


namespace part1_part2_part3_l281_281958

noncomputable def quadratic_has_real_roots (k : ℝ) : Prop :=
  ∃ x1 x2 : ℝ, x1^2 - 2*k*x1 + k^2 + k + 1 = 0 ∧ x2^2 - 2*k*x2 + k^2 + k + 1 = 0

theorem part1 (k : ℝ) :
  quadratic_has_real_roots k → k ≤ -1 :=
sorry

theorem part2 (k : ℝ) (x1 x2 : ℝ) :
  quadratic_has_real_roots k ∧ x1^2 + x2^2 = 10 → k = -2 :=
sorry

theorem part3 (k : ℝ) (x1 x2 : ℝ) :
  quadratic_has_real_roots k ∧ (|x1| + |x2| = 2) → k = -1 :=
sorry

end part1_part2_part3_l281_281958


namespace min_time_to_cover_distance_l281_281900

variable (distance : ℝ := 3)
variable (vasya_speed_run : ℝ := 4)
variable (vasya_speed_skate : ℝ := 8)
variable (petya_speed_run : ℝ := 5)
variable (petya_speed_skate : ℝ := 10)

theorem min_time_to_cover_distance :
  ∃ (t : ℝ), t = 0.5 ∧
    ∃ (x : ℝ), 
    0 ≤ x ∧ x ≤ distance ∧ 
    (distance - x) / vasya_speed_run + x / vasya_speed_skate = t ∧
    x / petya_speed_run + (distance - x) / petya_speed_skate = t :=
by
  sorry

end min_time_to_cover_distance_l281_281900


namespace highlighter_difference_l281_281973

theorem highlighter_difference :
  ∀ (yellow pink blue : ℕ),
    yellow = 7 →
    pink = yellow + 7 →
    yellow + pink + blue = 40 →
    blue - pink = 5 :=
by
  intros yellow pink blue h_yellow h_pink h_total
  rw [h_yellow, h_pink] at h_total
  sorry

end highlighter_difference_l281_281973


namespace general_term_l281_281196

def S (n : ℕ) : ℤ := n^2 - 4*n

noncomputable def a (n : ℕ) : ℤ := 
  if n = 1 then S 1
  else S n - S (n - 1)

theorem general_term (n : ℕ) (hn : n ≥ 1) : a n = (2 * n - 5) := by
  sorry

end general_term_l281_281196


namespace total_job_applications_l281_281025

theorem total_job_applications (apps_in_state : ℕ) (apps_other_states : ℕ) 
  (h1 : apps_in_state = 200)
  (h2 : apps_other_states = 2 * apps_in_state) :
  apps_in_state + apps_other_states = 600 :=
by
  sorry

end total_job_applications_l281_281025


namespace calculate_volume_from_measurements_l281_281128

variables (r h : ℝ) (P : ℝ × ℝ)

noncomputable def volume_truncated_cylinder (area_base : ℝ) (height_segment : ℝ) : ℝ :=
  area_base * height_segment

theorem calculate_volume_from_measurements
    (radius : ℝ) (height : ℝ)
    (area_base : ℝ := π * radius^2)
    (P : ℝ × ℝ)  -- intersection point on the axis
    (height_segment : ℝ) : 
    volume_truncated_cylinder area_base height_segment = area_base * height_segment :=
by
  -- The proof would involve demonstrating the relationship mathematically
  sorry

end calculate_volume_from_measurements_l281_281128


namespace bananas_in_each_box_l281_281393

-- You might need to consider noncomputable if necessary here for Lean's real number support.
noncomputable def bananas_per_box (total_bananas : ℕ) (total_boxes : ℕ) : ℕ :=
  total_bananas / total_boxes

theorem bananas_in_each_box :
  bananas_per_box 40 8 = 5 := by
  sorry

end bananas_in_each_box_l281_281393


namespace number_of_integer_solutions_l281_281537

theorem number_of_integer_solutions
    (a : ℤ)
    (x : ℤ)
    (h1 : ∃ x : ℤ, (1 - a) / (x - 2) + 2 = 1 / (2 - x))
    (h2 : ∀ x : ℤ, 4 * x ≥ 3 * (x - 1) ∧ x + (2 * x - 1) / 2 < (a - 1) / 2) :
    (a = 4) :=
sorry

end number_of_integer_solutions_l281_281537


namespace find_chocolate_boxes_l281_281925

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

end find_chocolate_boxes_l281_281925


namespace students_still_in_school_l281_281628

-- Declare the number of students initially in the school
def initial_students : Nat := 1000

-- Declare that half of the students were taken to the beach
def taken_to_beach (total_students : Nat) : Nat := total_students / 2

-- Declare that half of the remaining students were sent home
def sent_home (remaining_students : Nat) : Nat := remaining_students / 2

-- Declare the theorem to prove the final number of students still in school
theorem students_still_in_school : 
  let total_students := initial_students in
  let students_at_beach := taken_to_beach total_students in
  let students_remaining := total_students - students_at_beach in
  let students_sent_home := sent_home students_remaining in
  let students_left := students_remaining - students_sent_home in
  students_left = 250 := by
  sorry

end students_still_in_school_l281_281628


namespace olivia_pieces_of_paper_l281_281716

theorem olivia_pieces_of_paper (initial_pieces : ℕ) (used_pieces : ℕ) (pieces_left : ℕ) 
  (h1 : initial_pieces = 81) (h2 : used_pieces = 56) : 
  pieces_left = 81 - 56 :=
by
  sorry

end olivia_pieces_of_paper_l281_281716


namespace total_decorations_l281_281156

-- Define the conditions
def decorations_per_box := 4 + 1 + 5
def total_boxes := 11 + 1

-- Statement of the problem: Prove that the total number of decorations handed out is 120
theorem total_decorations : total_boxes * decorations_per_box = 120 := by
  sorry

end total_decorations_l281_281156


namespace slope_of_line_l281_281326

open Function

def parabola (y : ℝ) : ℝ := y^2 - 4

def focus : ℝ × ℝ := (1, 0)

def line_eq (k : ℝ) (x y : ℝ) : Prop := y = k * (x - 1)

theorem slope_of_line
  (A B : ℝ × ℝ)
  (k : ℝ)
  (h_intersectA : parabola A.2 = 4 * A.1)
  (h_intersectB : parabola B.2 = 4 * B.1)
  (h_lineA : line_eq k A.1 A.2)
  (h_lineB : line_eq k B.1 B.2)
  (h_distance : |A.1 - 1| * |A.2| = 4 * (|B.1 - 1| * |B.2|)) :
  k = 4 / 3 ∨ k = -4 / 3 :=
by
  sorry

end slope_of_line_l281_281326


namespace eighty_percent_of_number_l281_281608

theorem eighty_percent_of_number (x : ℝ) (h : 0.20 * x = 60) : 0.80 * x = 240 := 
by sorry

end eighty_percent_of_number_l281_281608


namespace smallest_x_plus_y_l281_281793

theorem smallest_x_plus_y (x y : ℕ) (h_xy_not_equal : x ≠ y) (h_condition : (1 : ℚ) / x + 1 / y = 1 / 15) :
  x + y = 64 :=
sorry

end smallest_x_plus_y_l281_281793


namespace expectation_defective_items_variance_of_defective_items_l281_281956
-- Importing the necessary library from Mathlib

-- Define the conditions
def total_products : ℕ := 100
def defective_products : ℕ := 10
def selected_products : ℕ := 3

-- Define the expected number of defective items
def expected_defective_items : ℝ := 0.3

-- Define the variance of defective items
def variance_defective_items : ℝ := 0.2645

-- Lean statements to verify the conditions and results
theorem expectation_defective_items :
  let p := (defective_products: ℝ) / (total_products: ℝ)
  p * (selected_products: ℝ) = expected_defective_items := by sorry

theorem variance_of_defective_items :
  let p := (defective_products: ℝ) / (total_products: ℝ)
  let n := (selected_products: ℝ)
  n * p * (1 - p) * (total_products - n) / (total_products - 1) = variance_defective_items := by sorry

end expectation_defective_items_variance_of_defective_items_l281_281956


namespace max_value_of_y_l281_281341

open Real

theorem max_value_of_y (x : ℝ) (h1 : 0 < x) (h2 : x < sqrt 3) : x * sqrt (3 - x^2) ≤ 9 / 4 :=
sorry

end max_value_of_y_l281_281341


namespace hypotenuse_length_is_13_l281_281683

theorem hypotenuse_length_is_13 (a b c : ℝ) (ha : a = 5) (hb : b = 12)
  (hrt : a ^ 2 + b ^ 2 = c ^ 2) : c = 13 :=
by
  -- to complete the proof, fill in the details here
  sorry

end hypotenuse_length_is_13_l281_281683


namespace compute_trig_expression_l281_281325

theorem compute_trig_expression : 
  (1 - 1 / (Real.cos (37 * Real.pi / 180))) *
  (1 + 1 / (Real.sin (53 * Real.pi / 180))) *
  (1 - 1 / (Real.sin (37 * Real.pi / 180))) *
  (1 + 1 / (Real.cos (53 * Real.pi / 180))) = 1 :=
sorry

end compute_trig_expression_l281_281325


namespace average_side_lengths_of_squares_l281_281258

theorem average_side_lengths_of_squares:
  let a₁ := 25
  let a₂ := 36
  let a₃ := 64

  let s₁ := Real.sqrt a₁
  let s₂ := Real.sqrt a₂
  let s₃ := Real.sqrt a₃

  (s₁ + s₂ + s₃) / 3 = 19 / 3 :=
by 
  sorry

end average_side_lengths_of_squares_l281_281258


namespace fraction_sum_eq_neg_one_l281_281504

theorem fraction_sum_eq_neg_one (p q : ℝ) (hpq : (1 / p) + (1 / q) = (1 / (p + q))) :
  (p / q) + (q / p) = -1 :=
by
  sorry

end fraction_sum_eq_neg_one_l281_281504


namespace students_in_class_l281_281889

theorem students_in_class (S : ℕ) 
  (h1 : (1 / 4) * (9 / 10 : ℚ) * S = 9) : S = 40 :=
sorry

end students_in_class_l281_281889


namespace subtraction_problem_solution_l281_281602

theorem subtraction_problem_solution :
  ∃ x : ℝ, (8 - x) / (9 - x) = 4 / 5 :=
by
  use 4
  sorry

end subtraction_problem_solution_l281_281602


namespace Brittany_age_after_vacation_l281_281318

variable (Rebecca_age Brittany_age vacation_years : ℕ)
variable (h1 : Rebecca_age = 25)
variable (h2 : Brittany_age = Rebecca_age + 3)
variable (h3 : vacation_years = 4)

theorem Brittany_age_after_vacation : 
    Brittany_age + vacation_years = 32 := 
by
  sorry

end Brittany_age_after_vacation_l281_281318


namespace equivalence_condition_l281_281234

universe u

variables {U : Type u} (A B : Set U)

theorem equivalence_condition :
  (∃ (C : Set U), A ⊆ C ∧ B ⊆ Cᶜ) ↔ (A ∩ B = ∅) :=
sorry

end equivalence_condition_l281_281234


namespace binomial_alternating_sum_l281_281661

theorem binomial_alternating_sum :
  ∑ k in Finset.range (101 \\ 2 + 1), (-1)^k * (Nat.choose 101 (2 * k)) = -2^50 := by
sorry

end binomial_alternating_sum_l281_281661


namespace net_cut_square_l281_281718

-- Define the dimensions of the parallelepiped
structure Parallelepiped :=
  (length width height : ℕ)
  (length_eq : length = 2)
  (width_eq : width = 1)
  (height_eq : height = 1)

-- Define the net of the parallelepiped
structure NetConfig :=
  (total_squares : ℕ)
  (cut_squares : ℕ)
  (remaining_squares : ℕ)
  (cut_positions : Fin 5) -- Five possible cut positions

-- The remaining net has 9 squares after cutting one square
theorem net_cut_square (p : Parallelepiped) : 
  ∃ net : NetConfig, net.total_squares = 10 ∧ net.cut_squares = 1 ∧ net.remaining_squares = 9 ∧ net.cut_positions = 5 := 
sorry

end net_cut_square_l281_281718


namespace find_greatest_natural_number_l281_281497

-- Definitions for terms used in the conditions

def sum_of_squares (m : ℕ) : ℕ :=
  (m * (m + 1) * (2 * m + 1)) / 6

def is_perfect_square (a : ℕ) : Prop :=
  ∃ b : ℕ, b * b = a

-- Conditions defined in Lean terms
def condition1 (n : ℕ) : Prop := n ≤ 2010

def condition2 (n : ℕ) : Prop := 
  let sum1 := sum_of_squares n
  let sum2 := sum_of_squares (2 * n) - sum_of_squares n
  is_perfect_square (sum1 * sum2)

-- Main theorem statement
theorem find_greatest_natural_number : ∃ n, n ≤ 2010 ∧ condition2 n ∧ ∀ m, m ≤ 2010 ∧ condition2 m → m ≤ n := 
by 
  sorry

end find_greatest_natural_number_l281_281497


namespace sin_half_angle_l281_281190

variable (α : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : cos α = (1 + Real.sqrt 5) / 4)

theorem sin_half_angle :
  sin (α / 2) = (Real.sqrt 5 - 1) / 4 :=
sorry

end sin_half_angle_l281_281190


namespace smallest_sum_of_xy_l281_281804

theorem smallest_sum_of_xy (x y : ℕ) (h1 : x ≠ y) (h2 : 0 < x) (h3 : 0 < y) 
  (h4 : (1:ℚ)/x + (1:ℚ)/y = (1:ℚ)/15) : x + y = 64 :=
sorry

end smallest_sum_of_xy_l281_281804


namespace largest_n_divisibility_l281_281152

theorem largest_n_divisibility :
  ∃ n : ℕ, (n^3 + 100) % (n + 10) = 0 ∧
  (∀ m : ℕ, (m^3 + 100) % (m + 10) = 0 → m ≤ n) ∧ n = 890 :=
by
  sorry

end largest_n_divisibility_l281_281152


namespace husband_and_wife_age_l281_281462

theorem husband_and_wife_age (x y : ℕ) (h1 : 11 * x = 2 * (22 * y - 11 * x)) (h2 : 11 * x ≠ 0) (h3 : 11 * y ≠ 0) (h4 : 11 * (x + y) ≤ 99) : 
  x = 4 ∧ y = 3 :=
by
  sorry

end husband_and_wife_age_l281_281462


namespace diameter_of_larger_circle_l281_281257

theorem diameter_of_larger_circle (R r D : ℝ) 
  (h1 : R^2 - r^2 = 25) 
  (h2 : D = 2 * R) : 
  D = Real.sqrt (100 + 4 * r^2) := 
by 
  sorry

end diameter_of_larger_circle_l281_281257


namespace max_value_of_N_l281_281038

def I_k (k : Nat) : Nat :=
  10^(k + 1) + 32

def N (k : Nat) : Nat :=
  (Nat.factors (I_k k)).count 2

theorem max_value_of_N :
  ∃ k : Nat, N k = 6 ∧ (∀ m : Nat, N m ≤ 6) :=
by
  sorry

end max_value_of_N_l281_281038


namespace sin_half_alpha_l281_281185

theorem sin_half_alpha (α : ℝ) (hα1 : 0 < α ∧ α < π / 2) (hα2 : Real.cos α = (1 + Real.sqrt 5) / 4) : 
    Real.sin (α / 2) = (-1 + Real.sqrt 5) / 4 := 
by
  sorry

end sin_half_alpha_l281_281185


namespace unique_function_satisfying_conditions_l281_281033

theorem unique_function_satisfying_conditions :
  ∀ f : ℚ → ℚ, (f 1 = 2) → (∀ x y : ℚ, f (x * y) = f x * f y - f (x + y) + 1) → (∀ x : ℚ, f x = x + 1) :=
by
  intro f h1 hCond
  sorry

end unique_function_satisfying_conditions_l281_281033


namespace number_of_outcomes_exactly_two_evening_l281_281946

theorem number_of_outcomes_exactly_two_evening (chickens : Finset ℕ) (h_chickens : chickens.card = 4) 
    (day_places evening_places : ℕ) (h_day_places : day_places = 2) (h_evening_places : evening_places = 3) :
    ∃ n, n = (chickens.card.choose 2) ∧ n = 6 :=
by
  sorry

end number_of_outcomes_exactly_two_evening_l281_281946


namespace smallest_non_factor_product_l281_281424

open Nat

def is_factor (n d : ℕ) := d > 0 ∧ n % d = 0

theorem smallest_non_factor_product (a b : ℕ) (h1 : a ≠ b) (h2 : is_factor 48 a) (h3 : is_factor 48 b) (h4 : ¬ is_factor 48 (a * b)) : a * b = 18 :=
by
  sorry

end smallest_non_factor_product_l281_281424


namespace smallest_sum_of_inverses_l281_281813

theorem smallest_sum_of_inverses 
  (x y : ℕ) (hx : x ≠ y) (h1 : 0 < x) (h2 : 0 < y) (h_condition : (1 / x : ℚ) + 1 / y = 1 / 15) :
  x + y = 64 := 
sorry

end smallest_sum_of_inverses_l281_281813


namespace pictures_deleted_l281_281448

theorem pictures_deleted (zoo_pics museum_pics remaining_pics : ℕ) 
  (h1 : zoo_pics = 15) 
  (h2 : museum_pics = 18) 
  (h3 : remaining_pics = 2) : 
  zoo_pics + museum_pics - remaining_pics = 31 :=
by 
  sorry

end pictures_deleted_l281_281448


namespace ratio_expression_l281_281050

-- Given conditions: X : Y : Z = 3 : 2 : 6
def ratio (X Y Z : ℚ) : Prop := X / Y = 3 / 2 ∧ Y / Z = 2 / 6

-- The expression to be evaluated
def expr (X Y Z : ℚ) : ℚ := (4 * X + 3 * Y) / (5 * Z - 2 * X)

-- The proof problem itself
theorem ratio_expression (X Y Z : ℚ) (h : ratio X Y Z) : expr X Y Z = 3 / 4 := by
  sorry

end ratio_expression_l281_281050


namespace jogger_distance_l281_281621

theorem jogger_distance 
(speed_jogger : ℝ := 9)
(speed_train : ℝ := 45)
(train_length : ℕ := 120)
(time_to_pass : ℕ := 38)
(relative_speed_mps : ℝ := (speed_train - speed_jogger) * (1 / 3.6))
(distance_covered : ℝ := (relative_speed_mps * time_to_pass))
(d : ℝ := distance_covered - train_length) :
d = 260 := sorry

end jogger_distance_l281_281621


namespace integer_solution_l281_281446

theorem integer_solution (n : ℤ) (h1 : n + 15 > 16) (h2 : -3 * n^2 > -27) : n = 2 :=
by {
  sorry
}

end integer_solution_l281_281446


namespace quilt_shading_fraction_l281_281270

/-- 
Statement:
Given a quilt block made from nine unit squares, where two unit squares are divided diagonally into triangles, 
and one unit square is divided into four smaller equal squares with one of the smaller squares shaded, 
the fraction of the quilt that is shaded is \( \frac{5}{36} \).
-/
theorem quilt_shading_fraction : 
  let total_area := 9 
  let shaded_area := 1 / 4 + 1 / 2 + 1 / 2 
  shaded_area / total_area = 5 / 36 :=
by
  -- Definitions based on conditions
  let total_area := 9
  let shaded_area := 1 / 4 + 1 / 2 + 1 / 2
  -- The proof statement (fraction of shaded area)
  have h : shaded_area / total_area = 5 / 36 := sorry
  exact h

end quilt_shading_fraction_l281_281270


namespace susan_average_speed_l281_281574

noncomputable def average_speed_trip (d1 d2 : ℝ) (v1 v2 : ℝ) : ℝ :=
  let total_distance := d1 + d2
  let time1 := d1 / v1
  let time2 := d2 / v2
  let total_time := time1 + time2
  total_distance / total_time

theorem susan_average_speed :
  average_speed_trip 60 30 30 60 = 36 := 
by
  -- The proof can be filled in here
  sorry

end susan_average_speed_l281_281574


namespace monotonicity_f_has_unique_zero_point_f_has_unique_zero_point_l281_281347

noncomputable def f (a b x : ℝ) : ℝ := (x - 1) * Real.exp x - a * x^2 + b

theorem monotonicity (a b : ℝ) :
  (∀ x < 0, f a b x < 0) ∧ (∀ x > 0, f a b x > 0) → ∀ x ∈ Ioo (-∞ : ℝ) (0 : ℝ), f a b x < 0 :=
sorry

theorem f_has_unique_zero_point (a b : ℝ) (h1 : 1 / 2 < a ∧ a ≤ (Real.exp 2) / 2 ∧ b > 2 * a) :
  ∃! x : ℝ, f a b x = 0 :=
sorry

theorem f_has_unique_zero_point' (a b : ℝ) (h2 : 0 < a ∧ a < 1 / 2 ∧ b ≤ 2 * a) :
  ∃! x : ℝ, f a b x = 0 :=
sorry

end monotonicity_f_has_unique_zero_point_f_has_unique_zero_point_l281_281347


namespace change_making_ways_l281_281202

-- Define the conditions
def is_valid_combination (quarters nickels pennies : ℕ) : Prop :=
  quarters ≤ 2 ∧ 25 * quarters + 5 * nickels + pennies = 50

-- Define the main statement
theorem change_making_ways : 
  ∃(num_ways : ℕ), (∀(quarters nickels pennies : ℕ), is_valid_combination quarters nickels pennies → num_ways = 18) :=
sorry

end change_making_ways_l281_281202


namespace right_triangle_k_value_l281_281761

theorem right_triangle_k_value (x : ℝ) (k : ℝ) (s : ℝ) 
(h_triangle : 3*x + 4*x + 5*x = k * (1/2 * 3*x * 4*x)) 
(h_square : s = 10) (h_eq_apothems : 4*x = s/2) : 
k = 8 / 5 :=
by {
  sorry
}

end right_triangle_k_value_l281_281761


namespace minimum_m_plus_n_l281_281046

theorem minimum_m_plus_n
  (m n : ℝ)
  (h_m_pos : m > 0)
  (h_n_pos : n > 0)
  (h_ellipse : 1 / m + 4 / n = 1) :
  m + n = 9 :=
sorry

end minimum_m_plus_n_l281_281046


namespace number_of_ways_to_select_60_l281_281786

-- Define the conditions: five volunteers, and choose 2 people each day such that one serves both days
def select_ways (n : ℕ) (r : ℕ) : ℕ :=
  Nat.choose n r

theorem number_of_ways_to_select_60 : select_ways 5 1 * (select_ways 4 1 * select_ways 3 1) = 60 := by
  sorry

end number_of_ways_to_select_60_l281_281786


namespace yarn_total_length_l281_281089

/-- The green yarn is 156 cm long, the red yarn is 8 cm more than three times the green yarn,
    prove that the total length of the two pieces of yarn is 632 cm. --/
theorem yarn_total_length : 
  let green_yarn := 156
  let red_yarn := 3 * green_yarn + 8
  green_yarn + red_yarn = 632 :=
by
  let green_yarn := 156
  let red_yarn := 3 * green_yarn + 8
  sorry

end yarn_total_length_l281_281089


namespace cuboid_surface_area_increase_l281_281444

variables (L W H : ℝ)
def SA_original (L W H : ℝ) : ℝ := 2 * (L * W + L * H + W * H)

def SA_new (L W H : ℝ) : ℝ := 2 * ((1.50 * L) * (1.70 * W) + (1.50 * L) * (1.80 * H) + (1.70 * W) * (1.80 * H))

theorem cuboid_surface_area_increase :
  (SA_new L W H - SA_original L W H) / SA_original L W H * 100 = 315.5 :=
by
  sorry

end cuboid_surface_area_increase_l281_281444


namespace total_carrots_l281_281856

def Joan_carrots : ℕ := 29
def Jessica_carrots : ℕ := 11

theorem total_carrots : Joan_carrots + Jessica_carrots = 40 := by
  sorry

end total_carrots_l281_281856


namespace inverse_value_ratio_l281_281881

noncomputable def g (x : ℚ) : ℚ := (3 * x + 1) / (x - 4)

theorem inverse_value_ratio :
  (∃ (a b c d : ℚ), ∀ x, g ((a * x + b) / (c * x + d)) = x) → ∃ a c : ℚ, a / c = -4 :=
by
  sorry

end inverse_value_ratio_l281_281881


namespace students_still_in_school_l281_281632

theorem students_still_in_school
  (total_students : ℕ)
  (half_trip : total_students / 2 > 0)
  (half_remaining_sent_home : (total_students / 2) / 2 > 0)
  (total_students_val : total_students = 1000)
  :
  let students_still_in_school := total_students - (total_students / 2) - ((total_students - (total_students / 2)) / 2)
  students_still_in_school = 250 :=
by
  sorry

end students_still_in_school_l281_281632


namespace smallest_product_not_factor_of_48_exists_l281_281432

theorem smallest_product_not_factor_of_48_exists : 
  ∃ (a b : ℕ), (a ≠ b) ∧ (a ∣ 48) ∧ (b ∣ 48) ∧ ¬ (a * b ∣ 48) ∧ 
  (∀ (c d : ℕ), (c ≠ d) ∧ (c ∣ 48) ∧ (d ∣ 48) ∧ ¬ (c * d ∣ 48) → a * b ≤ c * d) ∧ (a * b = 18) :=
sorry

end smallest_product_not_factor_of_48_exists_l281_281432


namespace pressure_force_correct_l281_281646

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

end pressure_force_correct_l281_281646


namespace maria_profit_disks_l281_281071

theorem maria_profit_disks (cost_price_per_5 : ℝ) (sell_price_per_4 : ℝ) (desired_profit : ℝ) : 
  (cost_price_per_5 = 6) → (sell_price_per_4 = 8) → (desired_profit = 120) →
  (150 : ℝ) = desired_profit / ((sell_price_per_4 / 4) - (cost_price_per_5 / 5)) :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  done

end maria_profit_disks_l281_281071


namespace correct_proposition_l281_281558

variables {Point Line Plane : Type}
variables (m n : Line) (α β γ : Plane)

-- Conditions
axiom perpendicular (m : Line) (α : Plane) : Prop
axiom parallel (n : Line) (α : Plane) : Prop

-- Specific conditions given
axiom m_perp_α : perpendicular m α
axiom n_par_α : parallel n α

-- Statement to prove
theorem correct_proposition : perpendicular m n := sorry

end correct_proposition_l281_281558


namespace margaret_time_l281_281562

def factorial (n : Nat) : Nat :=
  match n with
  | 0 => 1
  | n+1 => (n+1) * factorial n

def total_permutations (n : Nat) : Nat :=
  factorial n

def total_time_in_minutes (total_permutations : Nat) (rate : Nat) : Nat :=
  total_permutations / rate

def time_in_hours_and_minutes (total_minutes : Nat) : Nat × Nat :=
  let hours := total_minutes / 60
  let minutes := total_minutes % 60
  (hours, minutes)

theorem margaret_time :
  let n := 8
  let r := 15
  let permutations := total_permutations n
  let total_minutes := total_time_in_minutes permutations r
  time_in_hours_and_minutes total_minutes = (44, 48) := by
  sorry

end margaret_time_l281_281562


namespace triangle_perimeter_l281_281265

theorem triangle_perimeter (r A : ℝ) (h_r : r = 2.5) (h_A : A = 50) : 
  ∃ p : ℝ, p = 40 :=
by
  sorry

end triangle_perimeter_l281_281265


namespace probability_odd_80_heads_l281_281018

noncomputable def coin_toss_probability_odd (n : ℕ) (p : ℝ) : ℝ :=
  (1 / 2) * (1 - (1 / 3^n))

theorem probability_odd_80_heads :
  coin_toss_probability_odd 80 (3 / 4) = (1 / 2) * (1 - 1 / 3^80) :=
by
  sorry

end probability_odd_80_heads_l281_281018


namespace students_still_in_school_l281_281631

theorem students_still_in_school
  (total_students : ℕ)
  (half_trip : total_students / 2 > 0)
  (half_remaining_sent_home : (total_students / 2) / 2 > 0)
  (total_students_val : total_students = 1000)
  :
  let students_still_in_school := total_students - (total_students / 2) - ((total_students - (total_students / 2)) / 2)
  students_still_in_school = 250 :=
by
  sorry

end students_still_in_school_l281_281631


namespace right_triangle_area_l281_281598

theorem right_triangle_area (a b : ℕ) (h₁ : a = 6) (h₂ : b = 8) :
  (1 / 2 : ℝ) * a * b = 24 := by
  sorry

end right_triangle_area_l281_281598


namespace probability_green_light_is_8_over_15_l281_281763

def total_cycle_duration (red yellow green : ℕ) : ℕ :=
  red + yellow + green

def probability_green_light (red yellow green : ℕ) : ℚ :=
  green / (total_cycle_duration red yellow green : ℚ)

theorem probability_green_light_is_8_over_15 :
  probability_green_light 30 5 40 = 8 / 15 := by
  sorry

end probability_green_light_is_8_over_15_l281_281763


namespace coefficients_sum_l281_281828

noncomputable def f (x : ℝ) : ℝ := (2 * x - 1) ^ 4

theorem coefficients_sum : 
  ∃ (a₀ a₁ a₂ a₃ a₄ : ℝ), 
  ((2 * x - 1) ^ 4 = a₀ + a₁ * x + a₂ * x ^ 2 + a₃ * x ^ 3 + a₄ * x ^ 4) 
  ∧ (a₁ + 2 * a₂ + 3 * a₃ + 4 * a₄ = 8) :=
sorry

end coefficients_sum_l281_281828


namespace max_dot_product_l281_281373

theorem max_dot_product : 
  let E := {p : ℝ × ℝ | p.1^2 / 4 + p.2^2 / 3 = 1},
  O : ℝ × ℝ := (0, 0),
  F : ℝ × ℝ := (-2, 0),
  p ∈ E →
  ∃ (max_val : ℝ), max_val = 6 ∧ 
    ∀ P ∈ E, (P.1 * (P.1 + 2)) + (P.2 * P.2) ≤ max_val :=
sorry

end max_dot_product_l281_281373


namespace smallest_product_not_factor_of_48_exists_l281_281431

theorem smallest_product_not_factor_of_48_exists : 
  ∃ (a b : ℕ), (a ≠ b) ∧ (a ∣ 48) ∧ (b ∣ 48) ∧ ¬ (a * b ∣ 48) ∧ 
  (∀ (c d : ℕ), (c ≠ d) ∧ (c ∣ 48) ∧ (d ∣ 48) ∧ ¬ (c * d ∣ 48) → a * b ≤ c * d) ∧ (a * b = 18) :=
sorry

end smallest_product_not_factor_of_48_exists_l281_281431


namespace points_lie_on_parabola_l281_281155

noncomputable def lies_on_parabola (t : ℝ) : Prop :=
  let x := Real.cos t ^ 2
  let y := Real.sin t * Real.cos t
  y ^ 2 = x * (1 - x)

-- Statement to prove
theorem points_lie_on_parabola : ∀ t : ℝ, lies_on_parabola t :=
by
  intro t
  sorry

end points_lie_on_parabola_l281_281155


namespace cost_to_fill_bathtub_with_jello_l281_281851

-- Define the conditions
def pounds_per_gallon : ℝ := 8
def gallons_per_cubic_foot : ℝ := 7.5
def cubic_feet_of_water : ℝ := 6
def tablespoons_per_pound : ℝ := 1.5
def cost_per_tablespoon : ℝ := 0.5

-- The theorem stating the cost to fill the bathtub with jello
theorem cost_to_fill_bathtub_with_jello : 
  let total_gallons := cubic_feet_of_water * gallons_per_cubic_foot in
  let total_pounds := total_gallons * pounds_per_gallon in
  let total_tablespoons := total_pounds * tablespoons_per_pound in
  let total_cost := total_tablespoons * cost_per_tablespoon in
  total_cost = 270 := 
by {
  -- Here's where we would provide the proof steps, but just add sorry to skip it
  sorry
}

end cost_to_fill_bathtub_with_jello_l281_281851


namespace soccer_team_starters_l281_281398

theorem soccer_team_starters (players : Fin 16) 
  (quadruplets : Fin 4)
  (choose7 : Fin 7):
  (choose (4 : ℕ) 2) * (choose (12 : ℕ) 5) = 4752 := by
  sorry

end soccer_team_starters_l281_281398


namespace pages_per_donut_l281_281066

def pages_written (total_pages : ℕ) (calories_per_donut : ℕ) (total_calories : ℕ) : ℕ :=
  let donuts := total_calories / calories_per_donut
  total_pages / donuts

theorem pages_per_donut (total_pages : ℕ) (calories_per_donut : ℕ) (total_calories : ℕ): 
  total_pages = 12 → calories_per_donut = 150 → total_calories = 900 → pages_written total_pages calories_per_donut total_calories = 2 := by
  intros
  sorry

end pages_per_donut_l281_281066


namespace distance_rowed_upstream_l281_281299

noncomputable def speed_of_boat_in_still_water := 18 -- from solution step; b = 18 km/h
def speed_of_stream := 3 -- given
def time := 4 -- given
def distance_downstream := 84 -- given

theorem distance_rowed_upstream 
  (b : ℕ) (s : ℕ) (t : ℕ) (d_down : ℕ) (d_up : ℕ)
  (h_stream : s = 3) 
  (h_time : t = 4)
  (h_distance_downstream : d_down = 84) 
  (h_speed_boat : b = 18) 
  (h_effective_downstream_speed : b + s = d_down / t) :
  d_up = 60 := by
  sorry

end distance_rowed_upstream_l281_281299


namespace feet_in_mile_l281_281479

theorem feet_in_mile (d t : ℝ) (speed_mph : ℝ) (speed_fps : ℝ) (miles_to_feet : ℝ) (hours_to_seconds : ℝ) :
  d = 200 → t = 4 → speed_mph = 34.09 → miles_to_feet = 5280 → hours_to_seconds = 3600 → 
  speed_fps = d / t → speed_fps = speed_mph * miles_to_feet / hours_to_seconds → 
  miles_to_feet = 5280 :=
by
  intros hd ht hspeed_mph hmiles_to_feet hhours_to_seconds hspeed_fps_eq hconversion
  -- You can add the proof steps here.
  sorry

end feet_in_mile_l281_281479


namespace point_in_second_quadrant_l281_281385

theorem point_in_second_quadrant (a : ℝ) :
  ∃ q : ℕ, q = 2 ∧ (-3 : ℝ) < 0 ∧ (a^2 + 1) > 0 := 
by sorry

end point_in_second_quadrant_l281_281385


namespace sector_longest_segment_squared_l281_281133

theorem sector_longest_segment_squared (d : ℝ) (n : ℕ) (m : ℝ) :
  d = 16 ∧ n = 4 →
  m = 8 * real.sqrt 2 →
  m^2 = 128 :=
by
  intro h1 h2
  sorry

end sector_longest_segment_squared_l281_281133


namespace other_liquid_cost_l281_281902

-- Definitions based on conditions
def total_fuel_gallons : ℕ := 12
def fuel_price_per_gallon : ℝ := 8
def oil_price_per_gallon : ℝ := 15
def fuel_cost : ℝ := total_fuel_gallons * fuel_price_per_gallon
def other_liquid_price_per_gallon (x : ℝ) : Prop :=
  (7 * x + 5 * oil_price_per_gallon = fuel_cost) ∨
  (7 * oil_price_per_gallon + 5 * x = fuel_cost)

-- Question: The cost of the other liquid per gallon
theorem other_liquid_cost :
  ∃ x, other_liquid_price_per_gallon x ∧ x = 3 :=
sorry

end other_liquid_cost_l281_281902


namespace complement_of_A_in_U_l281_281505

noncomputable def U : Set ℝ := {x | (x - 2) / x ≤ 1}

noncomputable def A : Set ℝ := {x | 2 - x ≤ 1}

theorem complement_of_A_in_U :
  (U \ A) = {x | 0 < x ∧ x < 1} :=
by
  sorry

end complement_of_A_in_U_l281_281505


namespace no_perfect_square_l281_281720

theorem no_perfect_square (n : ℕ) : ¬ ∃ k : ℕ, k^2 = 2 * 13^n + 5 * 7^n + 26 :=
sorry

end no_perfect_square_l281_281720


namespace correct_expression_l281_281603

theorem correct_expression (a b c : ℝ) : a - b + c = a - (b - c) :=
by
  sorry

end correct_expression_l281_281603


namespace sum_of_first_3n_terms_l281_281734

theorem sum_of_first_3n_terms (S_n S_2n S_3n : ℕ) (h1 : S_n = 48) (h2 : S_2n = 60) :
  S_3n = 63 :=
by
  sorry

end sum_of_first_3n_terms_l281_281734


namespace expected_value_X_l281_281894

noncomputable def fair_coin := MassFunction.replicate 2 0.5

def is_heads_tails (outcome : Finset ℕ) : Prop :=
  outcome.card = 2 ∧ 1 ∈ outcome ∧ 2 ∈ outcome

def X_binomial_distribution : ℕ → ℕ → MassFunction ℕ
| n p := MassFunction.bind (MassFunction.replicate n 0.5 ν) X

theorem expected_value_X :
  X_binomial_distribution 4 (is_heads_tails) = 2 := sorry

end expected_value_X_l281_281894


namespace eccentricity_of_hyperbola_l281_281195

noncomputable def hyperbola_eccentricity : Prop :=
  ∀ (a b : ℝ), a > 0 → b > 0 → (∀ (x y : ℝ), x - 3 * y + 1 = 0 → y = 3 * x)
  → (∀ (c : ℝ), c^2 = a^2 + b^2) → b = 3 * a → ∃ e : ℝ, e = Real.sqrt 10

-- Statement of the problem without proof (includes the conditions)
theorem eccentricity_of_hyperbola (a b : ℝ) (h : a > 0) (h2 : b > 0) 
  (h3 : ∀ (x y : ℝ), x - 3 * y + 1 = 0 → y = 3 * x)
  (h4 : ∀ (c : ℝ), c^2 = a^2 + b^2) : hyperbola_eccentricity := 
  sorry

end eccentricity_of_hyperbola_l281_281195


namespace polynomial_constant_l281_281130

theorem polynomial_constant
  (P : Polynomial ℤ)
  (h : ∀ Q F G : Polynomial ℤ, P.comp Q = F * G → F.degree = 0 ∨ G.degree = 0) :
  P.degree = 0 :=
by sorry

end polynomial_constant_l281_281130


namespace half_of_4_pow_2022_is_2_pow_4043_l281_281114

theorem half_of_4_pow_2022_is_2_pow_4043 :
  (4 ^ 2022) / 2 = 2 ^ 4043 :=
by sorry

end half_of_4_pow_2022_is_2_pow_4043_l281_281114


namespace sin_half_angle_l281_281173

theorem sin_half_angle (α : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : Real.cos α = (1 + Real.sqrt 5) / 4) : 
  Real.sin (α / 2) = (Real.sqrt 5 - 1) / 4 := 
by 
  sorry

end sin_half_angle_l281_281173


namespace michael_class_choosing_l281_281074

open Nat

theorem michael_class_choosing :
  (choose 6 3) * (choose 4 2) + (choose 6 4) * (choose 4 1) + (choose 6 5) = 186 := 
by
  sorry

end michael_class_choosing_l281_281074


namespace Chad_savings_l281_281486

theorem Chad_savings :
  let earnings_mowing := 600
  let earnings_birthday := 250
  let earnings_video_games := 150
  let earnings_odd_jobs := 150
  let total_earnings := earnings_mowing + earnings_birthday + earnings_video_games + earnings_odd_jobs
  let savings_rate := 0.40
  let savings := savings_rate * total_earnings
  savings = 460 :=
by
  -- Definitions
  let earnings_mowing : ℤ := 600
  let earnings_birthday : ℤ := 250
  let earnings_video_games : ℤ := 150
  let earnings_odd_jobs : ℤ := 150
  let total_earnings : ℤ := earnings_mowing + earnings_birthday + earnings_video_games + earnings_odd_jobs
  let savings_rate := (40:ℚ) / 100
  let savings : ℚ := savings_rate * total_earnings
  -- Proof (to be completed by sorry)
  exact sorry

end Chad_savings_l281_281486


namespace preimages_of_f_l281_281048

noncomputable def f (x : ℝ) : ℝ := -x^2 + 2 * x

theorem preimages_of_f (k : ℝ) : (∃ x₁ x₂ : ℝ, f x₁ = k ∧ f x₂ = k ∧ x₁ ≠ x₂) ↔ k < 1 := by
  sorry

end preimages_of_f_l281_281048


namespace squirrels_acorns_l281_281755

theorem squirrels_acorns (squirrels : ℕ) (total_collected : ℕ) (acorns_needed_per_squirrel : ℕ) (total_needed : ℕ) (acorns_still_needed : ℕ) : 
  squirrels = 5 → 
  total_collected = 575 → 
  acorns_needed_per_squirrel = 130 → 
  total_needed = squirrels * acorns_needed_per_squirrel →
  acorns_still_needed = total_needed - total_collected →
  acorns_still_needed / squirrels = 15 :=
by
  sorry

end squirrels_acorns_l281_281755


namespace sin_half_angle_l281_281189

variable (α : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : cos α = (1 + Real.sqrt 5) / 4)

theorem sin_half_angle :
  sin (α / 2) = (Real.sqrt 5 - 1) / 4 :=
sorry

end sin_half_angle_l281_281189


namespace distance_between_incenter_and_circumcenter_of_right_triangle_l281_281909

theorem distance_between_incenter_and_circumcenter_of_right_triangle (a b c : ℝ) (h1 : a = 6) (h2 : b = 8) (h3 : c = 10) (right_triangle : a^2 + b^2 = c^2) :
    ∃ (IO : ℝ), IO = Real.sqrt 5 :=
by
  rw [h1, h2, h3] at right_triangle
  have h_sum : 6^2 + 8^2 = 10^2 := by sorry
  exact ⟨Real.sqrt 5, by sorry⟩

end distance_between_incenter_and_circumcenter_of_right_triangle_l281_281909


namespace smallest_sum_of_xy_l281_281805

theorem smallest_sum_of_xy (x y : ℕ) (h1 : x ≠ y) (h2 : 0 < x) (h3 : 0 < y) 
  (h4 : (1:ℚ)/x + (1:ℚ)/y = (1:ℚ)/15) : x + y = 64 :=
sorry

end smallest_sum_of_xy_l281_281805


namespace probability_5_consecutive_heads_in_8_flips_l281_281617

noncomputable def probability_at_least_5_consecutive_heads (n : ℕ) : ℚ :=
  if n = 8 then 5 / 128 else 0  -- Using conditional given the specificity to n = 8

theorem probability_5_consecutive_heads_in_8_flips : 
  probability_at_least_5_consecutive_heads 8 = 5 / 128 := 
by
  -- Proof to be provided here
  sorry

end probability_5_consecutive_heads_in_8_flips_l281_281617


namespace winning_percentage_l281_281467

/-- A soccer team played 158 games and won 63.2 games. 
    Prove that the winning percentage of the team is 40%. --/
theorem winning_percentage (total_games : ℕ) (won_games : ℝ) (h1 : total_games = 158) (h2 : won_games = 63.2) :
  (won_games / total_games) * 100 = 40 :=
sorry

end winning_percentage_l281_281467


namespace a_cubed_value_l281_281876

theorem a_cubed_value (a b : ℝ) (k : ℝ) (h1 : a^3 * b^2 = k) (h2 : a = 5) (h3 : b = 2) : 
  ∃ (a : ℝ), (64 * a^3 = 500) → (a^3 = 125 / 16) :=
by
  sorry

end a_cubed_value_l281_281876


namespace equal_focal_distances_l281_281884

theorem equal_focal_distances (k : ℝ) (h₁ : k ≠ 0) (h₂ : 16 - k ≠ 0) 
  (h_hyperbola : ∀ x y, (x^2) / (16 - k) - (y^2) / k = 1)
  (h_ellipse : ∀ x y, 9 * x^2 + 25 * y^2 = 225) :
  0 < k ∧ k < 16 :=
sorry

end equal_focal_distances_l281_281884


namespace correct_value_l281_281612

-- Given condition
def incorrect_calculation (x : ℝ) : Prop := (x + 12) / 8 = 8

-- Theorem to prove the correct value
theorem correct_value (x : ℝ) (h : incorrect_calculation x) : (x - 12) * 9 = 360 :=
by
  sorry

end correct_value_l281_281612


namespace taco_truck_earnings_l281_281469

/-
Question: How many dollars did the taco truck make during the lunch rush?
Conditions:
1. Soft tacos are $2 each.
2. Hard shell tacos are $5 each.
3. The family buys 4 hard shell tacos and 3 soft tacos.
4. There are ten other customers.
5. Each of the ten other customers buys 2 soft tacos.
Answer: The taco truck made $66 during the lunch rush.
-/

theorem taco_truck_earnings :
  let soft_taco_price := 2
  let hard_taco_price := 5
  let family_hard_tacos := 4
  let family_soft_tacos := 3
  let other_customers := 10
  let other_customers_soft_tacos := 2
  (family_hard_tacos * hard_taco_price + family_soft_tacos * soft_taco_price +
   other_customers * other_customers_soft_tacos * soft_taco_price) = 66 := by
  sorry

end taco_truck_earnings_l281_281469


namespace vector_intersecting_line_parameter_l281_281929

theorem vector_intersecting_line_parameter :
  ∃ (a b s : ℝ), a = 3 * s + 5 ∧ b = 2 * s + 4 ∧
                   (∃ r, (a, b) = (3 * r, 2 * r)) ∧
                   (a, b) = (6, 14 / 3) :=
by
  sorry

end vector_intersecting_line_parameter_l281_281929


namespace solve_problem_l281_281829

def f (x : ℝ) : ℝ := x^2 - 4*x + 7
def g (x : ℝ) : ℝ := 2*x + 1

theorem solve_problem : f (g 3) - g (f 3) = 19 := by
  sorry

end solve_problem_l281_281829


namespace smallest_product_not_factor_of_48_l281_281438

theorem smallest_product_not_factor_of_48 (a b : ℕ) (h1 : a ≠ b) (h2 : a ∣ 48) (h3 : b ∣ 48) (h4 : ¬ (a * b ∣ 48)) : a * b = 32 :=
sorry

end smallest_product_not_factor_of_48_l281_281438


namespace original_cost_price_l281_281450

theorem original_cost_price ( C S : ℝ )
  (h1 : S = 1.05 * C)
  (h2 : S - 3 = 1.10 * 0.95 * C)
  : C = 600 :=
sorry

end original_cost_price_l281_281450


namespace handshake_count_l281_281061

theorem handshake_count (n : ℕ) (m : ℕ) (couples : ℕ) (people : ℕ) 
  (h1 : couples = 15) 
  (h2 : people = 2 * couples)
  (h3 : people = 30)
  (h4 : n = couples) 
  (h5 : m = people / 2)
  (h6 : ∀ i : ℕ, i < m → ∀ j : ℕ, j < m → i ≠ j → i * j + i ≠ n 
    ∧ j * i + j ≠ n) 
  : n * (n - 1) / 2 + (2 * n - 2) * n = 315 :=
by
  sorry

end handshake_count_l281_281061


namespace find_number_of_girls_l281_281289

theorem find_number_of_girls (B G : ℕ) 
  (h1 : B + G = 604) 
  (h2 : 12 * B + 11 * G = 47 * 604 / 4) : 
  G = 151 :=
by
  sorry

end find_number_of_girls_l281_281289


namespace evaluate_expression_l281_281927

def f (x : ℝ) : ℝ := 3 * x^2 + 5 * x - 7

theorem evaluate_expression : 3 * f 2 - 2 * f (-2) = 55 := by
  sorry

end evaluate_expression_l281_281927


namespace digit_7_count_correct_l281_281882

def base8ToBase10 (n : Nat) : Nat :=
  -- converting base 8 number 1000 to base 10
  1 * 8^3 + 0 * 8^2 + 0 * 8^1 + 0 * 8^0

def countDigit7 (n : Nat) : Nat :=
  -- counts the number of times the digit '7' appears in numbers from 1 to n
  let digits := (List.range (n + 1)).map fun x => x.digits 10
  digits.foldl (fun acc ds => acc + ds.count 7) 0

theorem digit_7_count_correct : countDigit7 512 = 123 := by
  sorry

end digit_7_count_correct_l281_281882


namespace x_cubed_plus_y_cubed_l281_281951

theorem x_cubed_plus_y_cubed (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 17) : x^3 + y^3 = 65 := 
by 
  sorry

end x_cubed_plus_y_cubed_l281_281951


namespace cos_pi_minus_alpha_correct_l281_281549

noncomputable def cos_pi_minus_alpha (α : ℝ) (P : ℝ × ℝ) : ℝ :=
  let x := P.1
  let y := P.2
  let h := Real.sqrt (x^2 + y^2)
  let cos_alpha := x / h
  let cos_pi_minus_alpha := -cos_alpha
  cos_pi_minus_alpha

theorem cos_pi_minus_alpha_correct :
  cos_pi_minus_alpha α (-1, 2) = Real.sqrt 5 / 5 :=
by
  sorry

end cos_pi_minus_alpha_correct_l281_281549


namespace find_fg_satisfy_l281_281333

noncomputable def f (x : ℝ) : ℝ := (Real.sin x + Real.cos x) / 2
noncomputable def g (x : ℝ) (c : ℝ) : ℝ := (Real.sin x - Real.cos x) / 2 + c

theorem find_fg_satisfy (c : ℝ) : ∀ x y : ℝ,
  Real.sin x + Real.cos y = f x + f y + g x c - g y c := 
by 
  intros;
  rw [f, g, g, f];
  sorry

end find_fg_satisfy_l281_281333


namespace real_root_exists_for_all_K_l281_281772

theorem real_root_exists_for_all_K (K : ℝ) : ∃ x : ℝ, x = K^2 * (x - 1) * (x - 2) * (x - 3) :=
sorry

end real_root_exists_for_all_K_l281_281772


namespace sufficient_but_not_necessary_condition_l281_281964

variables {a b : ℝ}

theorem sufficient_but_not_necessary_condition (h₁ : b < -4) : |a| + |b| > 4 :=
by {
    sorry
}

end sufficient_but_not_necessary_condition_l281_281964


namespace stock_price_is_108_l281_281690

noncomputable def dividend_income (FV : ℕ) (D : ℕ) : ℕ :=
  FV * D / 100

noncomputable def face_value_of_stock (I : ℕ) (D : ℕ) : ℕ :=
  I * 100 / D

noncomputable def price_of_stock (Inv : ℕ) (FV : ℕ) : ℕ :=
  Inv * 100 / FV

theorem stock_price_is_108 (I D Inv : ℕ) (hI : I = 450) (hD : D = 10) (hInv : Inv = 4860) :
  price_of_stock Inv (face_value_of_stock I D) = 108 :=
by
  -- Placeholder for proof
  sorry

end stock_price_is_108_l281_281690


namespace smallest_product_not_factor_of_48_exists_l281_281433

theorem smallest_product_not_factor_of_48_exists : 
  ∃ (a b : ℕ), (a ≠ b) ∧ (a ∣ 48) ∧ (b ∣ 48) ∧ ¬ (a * b ∣ 48) ∧ 
  (∀ (c d : ℕ), (c ≠ d) ∧ (c ∣ 48) ∧ (d ∣ 48) ∧ ¬ (c * d ∣ 48) → a * b ≤ c * d) ∧ (a * b = 18) :=
sorry

end smallest_product_not_factor_of_48_exists_l281_281433


namespace find_quartic_polynomial_l281_281500

noncomputable def p (x : ℝ) : ℝ := -(1 / 9) * x^4 + (40 / 9) * x^3 - 8 * x^2 + 10 * x + 2

theorem find_quartic_polynomial :
  p 1 = -3 ∧
  p 2 = -1 ∧
  p 3 = 1 ∧
  p 4 = -7 ∧
  p 0 = 2 :=
by
  sorry

end find_quartic_polynomial_l281_281500


namespace solve_quadratic_equation_l281_281874

theorem solve_quadratic_equation (x : ℝ) :
  (x^2 - 2 * x - 5 = 0) ↔ (x = 1 + Real.sqrt 6 ∨ x = 1 - Real.sqrt 6) := 
sorry

end solve_quadratic_equation_l281_281874


namespace arithmetic_seq_a8_l281_281691

def is_arith_seq (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_seq_a8
  (a : ℕ → ℤ)
  (h_arith : is_arith_seq a)
  (h_a2 : a 2 = 2)
  (h_a4 : a 4 = 6) :
  a 8 = 14 := sorry

end arithmetic_seq_a8_l281_281691


namespace profit_percent_is_correct_l281_281908

noncomputable def profit_percent : ℝ := 
  let marked_price_per_pen := 1 
  let pens_bought := 56 
  let effective_payment := 46 
  let discount := 0.01
  let cost_price_per_pen := effective_payment / pens_bought
  let selling_price_per_pen := marked_price_per_pen * (1 - discount)
  let total_selling_price := pens_bought * selling_price_per_pen
  let profit := total_selling_price - effective_payment
  (profit / effective_payment) * 100

theorem profit_percent_is_correct : abs (profit_percent - 20.52) < 0.01 :=
by
  sorry

end profit_percent_is_correct_l281_281908


namespace prime_exponent_50_factorial_5_l281_281223

def count_factors_in_factorial (n p : ℕ) : ℕ :=
  if p ≤ 1 then 0 else
    let rec count (k acc : ℕ) : ℕ :=
      if n < p^k then acc else count (k+1) (acc + n/(p^k))
    count 1 0

theorem prime_exponent_50_factorial_5 : count_factors_in_factorial 50 5 = 12 :=
  by
    sorry

end prime_exponent_50_factorial_5_l281_281223


namespace minimum_inhabitants_to_ask_to_be_certain_l281_281396

theorem minimum_inhabitants_to_ask_to_be_certain
  (knights civilians : ℕ) (total_inhabitants : ℕ) :
  knights = 50 → civilians = 15 → total_inhabitants = 65 →
  ∃ (n : ℕ), n = 31 ∧
    (∀ (asked_knights asked_civilians : ℕ),
     asked_knights + asked_civilians = n →
     asked_knights ≥ 16) :=
by
  intro h_knights h_civilians h_total_inhabitants
  use 31
  split
  { rfl }
  { intros asked_knights asked_civilians h_total_asked
    have h_asked_bound : asked_knights ≥ 16,
    { linarith [h_total_asked, le_of_add_le_add_left h_total_asked] },
    exact h_asked_bound }

end minimum_inhabitants_to_ask_to_be_certain_l281_281396


namespace fill_tank_time_l281_281912

-- Definitions based on provided conditions
def pipeA_time := 60 -- Pipe A fills the tank in 60 minutes
def pipeB_time := 40 -- Pipe B fills the tank in 40 minutes

-- Theorem statement
theorem fill_tank_time (T : ℕ) : 
  (T / 2) / pipeB_time + (T / 2) * (1 / pipeA_time + 1 / pipeB_time) = 1 → 
  T = 48 :=
by
  intro h
  sorry

end fill_tank_time_l281_281912


namespace find_length_JM_l281_281224

-- Definitions of the problem
variables {DE DF EF : ℝ} 
variable {DEF : Triangle ℝ} 
variable {DG EH FI : Line ℝ} 
variable {J M : Point ℝ}

def centroid (DEF : Triangle ℝ) : Point ℝ := centroid(DEF)

def foot_of_altitude (J : Point ℝ) (EF : Line ℝ) : Point ℝ := foot_of_altitude(J, EF)

theorem find_length_JM :
  DE = 14 → DF = 15 → EF = 21 → 
  let G := centroid DEF,
      H := foot_of_altitude J EF,
      DN : ℝ := altitude_length DEF D EF,
      area_DEF : ℝ := herons_formula 14 15 21 
  in 
  G = J ∧ H = M ∧ DN ≠ 0 ∧ 
  84 = area DEF ∧ DN = (2 * 84) / 21 →
  JM = 8 / 3 :=
sorry

end find_length_JM_l281_281224


namespace solve_B_share_l281_281474

def ratio_shares (A B C : ℚ) : Prop :=
  A = 1/2 ∧ B = 1/3 ∧ C = 1/4

def initial_capitals (total_capital : ℚ) (A_s B_s C_s : ℚ) : Prop :=
  A_s = 1/2 * total_capital ∧ B_s = 1/3 * total_capital ∧ C_s = 1/4 * total_capital

def total_capital_contribution (A_contrib B_contrib C_contrib : ℚ) : Prop :=
  A_contrib = 42 ∧ B_contrib = 48 ∧ C_contrib = 36

def B_share (B_contrib total_contrib profit : ℚ) : ℚ := 
  (B_contrib / total_contrib) * profit

theorem solve_B_share : 
  ∀ (A_s B_s C_s total_capital profit A_contrib B_contrib C_contrib total_contrib : ℚ),
  ratio_shares (1/2) (1/3) (1/4) →
  initial_capitals total_capital A_s B_s C_s →
  total_capital_contribution A_contrib B_contrib C_contrib →
  total_contrib = A_contrib + B_contrib + C_contrib →
  profit = 378 →
  B_s = (1/3) * total_capital →
  B_contrib = 48 →
  B_share B_contrib total_contrib profit = 108 := by 
    sorry

end solve_B_share_l281_281474


namespace johnnyMoneyLeft_l281_281703

noncomputable def johnnySavingsSeptember : ℝ := 30
noncomputable def johnnySavingsOctober : ℝ := 49
noncomputable def johnnySavingsNovember : ℝ := 46
noncomputable def johnnySavingsDecember : ℝ := 55

noncomputable def johnnySavingsJanuary : ℝ := johnnySavingsDecember * 1.15

noncomputable def totalSavings : ℝ := johnnySavingsSeptember + johnnySavingsOctober + johnnySavingsNovember + johnnySavingsDecember + johnnySavingsJanuary

noncomputable def videoGameCost : ℝ := 58
noncomputable def bookCost : ℝ := 25
noncomputable def birthdayPresentCost : ℝ := 40

noncomputable def totalSpent : ℝ := videoGameCost + bookCost + birthdayPresentCost

noncomputable def moneyLeft : ℝ := totalSavings - totalSpent

theorem johnnyMoneyLeft : moneyLeft = 120.25 := by
  sorry

end johnnyMoneyLeft_l281_281703


namespace salary_increase_l281_281589

theorem salary_increase (x : ℕ) (hB_C_sum : 2*x + 3*x = 6000) : 
  ((3 * x - 1 * x) / (1 * x) ) * 100 = 200 :=
by
  -- Placeholder for the proof
  sorry

end salary_increase_l281_281589


namespace quadratic_inequality_condition_l281_281337

theorem quadratic_inequality_condition
  (a b c : ℝ)
  (h1 : b^2 - 4 * a * c < 0)
  (h2 : ∀ x : ℝ, a * x^2 + b * x + c < 0) :
  False :=
sorry

end quadratic_inequality_condition_l281_281337


namespace Ian_kept_1_rose_l281_281363

theorem Ian_kept_1_rose : 
  ∀ (total_r: ℕ) (mother_r: ℕ) (grandmother_r: ℕ) (sister_r: ℕ), 
  total_r = 20 → 
  mother_r = 6 → 
  grandmother_r = 9 → 
  sister_r = 4 → 
  total_r - (mother_r + grandmother_r + sister_r) = 1 :=
by
  intros total_r mother_r grandmother_r sister_r h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  exact Nat.sub_eq_of_eq_add' (by rfl)

end Ian_kept_1_rose_l281_281363


namespace mitya_age_l281_281994

-- Definitions of the ages
variables (M S : ℕ)

-- Conditions based on the problem statements
axiom condition1 : M = S + 11
axiom condition2 : S = 2 * (S - (M - S))

-- The theorem stating that Mitya is 33 years old
theorem mitya_age : M = 33 :=
by
  -- Outline the proof
  sorry

end mitya_age_l281_281994


namespace parallel_line_distance_equation_l281_281496

theorem parallel_line_distance_equation :
  ∃ m : ℝ, (m = -20 ∨ m = 32) ∧
  ∀ x y : ℝ, (5 * x - 12 * y + 6 = 0) → 
            (5 * x - 12 * y + m = 0) :=
by
  sorry

end parallel_line_distance_equation_l281_281496


namespace smallest_sum_of_xy_l281_281807

theorem smallest_sum_of_xy (x y : ℕ) (h1 : x ≠ y) (h2 : 0 < x) (h3 : 0 < y) 
  (h4 : (1:ℚ)/x + (1:ℚ)/y = (1:ℚ)/15) : x + y = 64 :=
sorry

end smallest_sum_of_xy_l281_281807


namespace intersection_of_domains_l281_281519

def M (x : ℝ) : Prop := x < 1
def N (x : ℝ) : Prop := x > -1
def P (x : ℝ) : Prop := -1 < x ∧ x < 1

theorem intersection_of_domains : {x : ℝ | M x} ∩ {x : ℝ | N x} = {x : ℝ | P x} :=
by
  sorry

end intersection_of_domains_l281_281519


namespace estimate_fish_population_l281_281012

theorem estimate_fish_population :
  ∀ (initial_tagged: ℕ) (august_sample: ℕ) (tagged_in_august: ℕ) (leaving_rate: ℝ) (new_rate: ℝ),
  initial_tagged = 50 →
  august_sample = 80 →
  tagged_in_august = 4 →
  leaving_rate = 0.30 →
  new_rate = 0.45 →
  ∃ (april_population : ℕ),
  april_population = 550 :=
by
  intros initial_tagged august_sample tagged_in_august leaving_rate new_rate
  intros h_initial_tagged h_august_sample h_tagged_in_august h_leaving_rate h_new_rate
  existsi 550
  sorry

end estimate_fish_population_l281_281012


namespace polynomial_factors_l281_281680

theorem polynomial_factors (t q : ℤ) (h1 : 81 - 3 * t + q = 0) (h2 : -3 + t + q = 0) : |3 * t - 2 * q| = 99 :=
sorry

end polynomial_factors_l281_281680


namespace remainder_of_3056_mod_32_l281_281893

theorem remainder_of_3056_mod_32 : 3056 % 32 = 16 := by
  sorry

end remainder_of_3056_mod_32_l281_281893


namespace habitable_fraction_of_earth_l281_281372

theorem habitable_fraction_of_earth :
  (1 / 2) * (1 / 4) = 1 / 8 := by
  sorry

end habitable_fraction_of_earth_l281_281372


namespace probability_A2_equals_zero_matrix_l281_281238

noncomputable def probability_A2_zero (n : ℕ) (hn : n ≥ 2) : ℚ :=
  let numerator := (n - 1) * (n - 2)
  let denominator := n * (n - 1)
  numerator / denominator

theorem probability_A2_equals_zero_matrix (n : ℕ) (hn : n ≥ 2) :
  probability_A2_zero n hn = ((n - 1) * (n - 2) / (n * (n - 1))) := by
  sorry

end probability_A2_equals_zero_matrix_l281_281238


namespace sum_coordinates_l281_281996

variables (x y : ℝ)
def A_coord := (9, 3)
def M_coord := (3, 7)

def midpoint_condition_x : Prop := (x + 9) / 2 = 3
def midpoint_condition_y : Prop := (y + 3) / 2 = 7

theorem sum_coordinates (h1 : midpoint_condition_x x) (h2 : midpoint_condition_y y) : 
  x + y = 8 :=
by 
  sorry

end sum_coordinates_l281_281996


namespace clock_hands_angle_120_between_7_and_8_l281_281147

theorem clock_hands_angle_120_between_7_and_8 :
  ∃ (t₁ t₂ : ℕ), (t₁ = 5) ∧ (t₂ = 16) ∧ 
  (∃ (h₀ m₀ : ℕ → ℝ), 
    h₀ 7 = 210 ∧ 
    m₀ 7 = 0 ∧
    (∀ t : ℕ, h₀ (7 + t / 60) = 210 + t * (30 / 60)) ∧
    (∀ t : ℕ, m₀ (7 + t / 60) = t * (360 / 60)) ∧
    ((h₀ (7 + t₁ / 60) - m₀ (7 + t₁ / 60)) % 360 = 120) ∧ 
    ((h₀ (7 + t₂ / 60) - m₀ (7 + t₂ / 60)) % 360 = 120)) := by
  sorry

end clock_hands_angle_120_between_7_and_8_l281_281147


namespace false_proposition_l281_281567

-- Definitions based on conditions
def opposite_angles (α β : ℝ) : Prop := α = β
def perpendicular (l m : ℝ → ℝ) : Prop := ∀ x, l x * m x = -1
def parallel (l m : ℝ → ℝ) : Prop := ∃ c, ∀ x, l x = m x + c
def corresponding_angles (α β : ℝ) : Prop := α = β

-- Propositions from the problem
def proposition1 : Prop := ∀ α β, opposite_angles α β → α = β
def proposition2 : Prop := ∀ l m n, perpendicular l n → perpendicular m n → parallel l m
def proposition3 : Prop := ∀ α β, α = β → opposite_angles α β
def proposition4 : Prop := ∀ α β, corresponding_angles α β → α = β

-- Statement to prove proposition 3 is false under given conditions
theorem false_proposition : ¬ proposition3 := by
  -- By our analysis, if proposition 3 is false, then it means the given definition for proposition 3 holds under all circumstances.
  sorry

end false_proposition_l281_281567


namespace cost_to_fill_bathtub_with_jello_l281_281850

-- Define the conditions
def pounds_per_gallon : ℝ := 8
def gallons_per_cubic_foot : ℝ := 7.5
def cubic_feet_of_water : ℝ := 6
def tablespoons_per_pound : ℝ := 1.5
def cost_per_tablespoon : ℝ := 0.5

-- The theorem stating the cost to fill the bathtub with jello
theorem cost_to_fill_bathtub_with_jello : 
  let total_gallons := cubic_feet_of_water * gallons_per_cubic_foot in
  let total_pounds := total_gallons * pounds_per_gallon in
  let total_tablespoons := total_pounds * tablespoons_per_pound in
  let total_cost := total_tablespoons * cost_per_tablespoon in
  total_cost = 270 := 
by {
  -- Here's where we would provide the proof steps, but just add sorry to skip it
  sorry
}

end cost_to_fill_bathtub_with_jello_l281_281850


namespace inequality_AM_GM_l281_281228

theorem inequality_AM_GM
  (a b c : ℝ)
  (ha : 0 < a) 
  (hb : 0 < b) 
  (hc : 0 < c)
  (habc : a + b + c = 1) : 
  (a + 2 * a * b + 2 * a * c + b * c) ^ a * 
  (b + 2 * b * c + 2 * b * a + c * a) ^ b * 
  (c + 2 * c * a + 2 * c * b + a * b) ^ c ≤ 1 :=
by
  sorry

end inequality_AM_GM_l281_281228


namespace warehouse_problem_l281_281756

/-- 
Problem Statement:
A certain unit decides to invest 3200 yuan to build a warehouse (in the shape of a rectangular prism) with a constant height.
The back wall will be built reusing the old wall at no cost, the front will be made of iron grilles at a cost of 40 yuan per meter in length,
and the two side walls will be built with bricks at a cost of 45 yuan per meter in length.
The top will have a cost of 20 yuan per square meter.
Let the length of the iron grilles be x meters and the length of one brick wall be y meters.
Find:
1. Write down the relationship between x and y.
2. Determine the maximum allowable value of the warehouse area S. In order to maximize S without exceeding the budget, how long should the front iron grille be designed
-/

theorem warehouse_problem (x y : ℝ) :
    (40 * x + 90 * y + 20 * x * y = 3200 ∧ 0 < x ∧ x < 80) →
    (y = (320 - 4 * x) / (9 + 2 * x) ∧ x = 15 ∧ y = 20 / 3 ∧ x * y = 100) :=
by
  sorry

end warehouse_problem_l281_281756


namespace square_of_complex_l281_281144

def z : Complex := 5 - 2 * Complex.I

theorem square_of_complex : z^2 = 21 - 20 * Complex.I := by
  sorry

end square_of_complex_l281_281144


namespace speed_of_second_train_l281_281901

-- Define the given values
def length_train1 := 290.0 -- in meters
def speed_train1 := 120.0 -- in km/h
def length_train2 := 210.04 -- in meters
def crossing_time := 9.0 -- in seconds

-- Define the conversion factors and useful calculations
def meters_per_second_to_kmph (v : Float) : Float := v * 3.6
def total_distance := length_train1 + length_train2
def relative_speed_ms := total_distance / crossing_time
def relative_speed_kmph := meters_per_second_to_kmph relative_speed_ms

-- Define the proof statement
theorem speed_of_second_train : relative_speed_kmph - speed_train1 = 80.0 :=
by
  sorry

end speed_of_second_train_l281_281901


namespace algebraic_expression_value_l281_281205

theorem algebraic_expression_value (x y : ℝ) (h : x - 2 * y + 3 = 0) : 1 - 2 * x + 4 * y = 7 := 
by
  sorry

end algebraic_expression_value_l281_281205


namespace volume_of_cone_l281_281955

theorem volume_of_cone (l : ℝ) (A : ℝ) (r : ℝ) (h : ℝ) : 
  l = 10 → A = 60 * Real.pi → (r = 6) → (h = Real.sqrt (10^2 - 6^2)) → 
  (1 / 3 * Real.pi * r^2 * h) = 96 * Real.pi :=
by
  intros
  -- here the proof would be written
  sorry

end volume_of_cone_l281_281955


namespace at_least_one_not_less_than_2_l281_281831

theorem at_least_one_not_less_than_2 (x y z : ℝ) (hp : 0 < x ∧ 0 < y ∧ 0 < z) :
  let a := x + 1/y
  let b := y + 1/z
  let c := z + 1/x
  (a ≥ 2 ∨ b ≥ 2 ∨ c ≥ 2) := by
    sorry

end at_least_one_not_less_than_2_l281_281831


namespace sons_age_l281_281461

theorem sons_age (S M : ℕ) (h1 : M = 3 * S) (h2 : M + 12 = 2 * (S + 12)) : S = 12 :=
by 
  sorry

end sons_age_l281_281461


namespace largest_piece_length_l281_281918

theorem largest_piece_length (v : ℝ) (hv : v + (3/2) * v + (9/4) * v = 95) : 
  (9/4) * v = 45 :=
by sorry

end largest_piece_length_l281_281918


namespace ryan_lost_initially_l281_281871

-- Define the number of leaves initially collected
def initial_leaves : ℤ := 89

-- Define the number of leaves broken afterwards
def broken_leaves : ℤ := 43

-- Define the number of leaves left in the collection
def remaining_leaves : ℤ := 22

-- Define the lost leaves
def lost_leaves (L : ℤ) : Prop :=
  initial_leaves - L - broken_leaves = remaining_leaves

theorem ryan_lost_initially : ∃ L : ℤ, lost_leaves L ∧ L = 24 :=
by
  sorry

end ryan_lost_initially_l281_281871


namespace quadratic_distinct_real_roots_l281_281210

theorem quadratic_distinct_real_roots (c : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ x^2 - 2*x + c = 0 ∧ y^2 - 2*y + c = 0) ↔ c < 1 :=
by
  sorry

end quadratic_distinct_real_roots_l281_281210


namespace length_of_string_for_circle_l281_281083

theorem length_of_string_for_circle (A : ℝ) (pi_approx : ℝ) (extra_length : ℝ) (hA : A = 616) (hpi : pi_approx = 22 / 7) (hextra : extra_length = 5) :
  ∃ (length : ℝ), length = 93 :=
by {
  sorry
}

end length_of_string_for_circle_l281_281083


namespace mitya_age_l281_281992

theorem mitya_age {M S: ℕ} (h1 : M = S + 11) (h2 : S = 2 * (S - (M - S))) : M = 33 :=
by
  -- proof steps skipped
  sorry

end mitya_age_l281_281992


namespace rectangle_width_l281_281577

theorem rectangle_width (length_rect : ℝ) (width_rect : ℝ) (side_square : ℝ)
  (h1 : side_square * side_square = 5 * (length_rect * width_rect))
  (h2 : length_rect = 125)
  (h3 : 4 * side_square = 800) : width_rect = 64 :=
by 
  sorry

end rectangle_width_l281_281577


namespace mul_point_five_point_three_l281_281768

theorem mul_point_five_point_three : 0.5 * 0.3 = 0.15 := 
by  sorry

end mul_point_five_point_three_l281_281768


namespace minimum_distance_square_l281_281672

/-- Given the equation of a circle centered at (2,3) with radius 1, find the minimum value of 
the function z = x^2 + y^2 -/
theorem minimum_distance_square (x y : ℝ) 
  (h : (x - 2)^2 + (y - 3)^2 = 1) : ∃ (z : ℝ), z = x^2 + y^2 ∧ z = 14 - 2 * Real.sqrt 13 :=
sorry

end minimum_distance_square_l281_281672


namespace average_speed_third_hour_l281_281403

theorem average_speed_third_hour
  (total_distance : ℝ)
  (total_time : ℝ)
  (speed_first_hour : ℝ)
  (speed_second_hour : ℝ)
  (speed_third_hour : ℝ) :
  total_distance = 150 →
  total_time = 3 →
  speed_first_hour = 45 →
  speed_second_hour = 55 →
  (speed_first_hour + speed_second_hour + speed_third_hour) / total_time = 50 →
  speed_third_hour = 50 :=
sorry

end average_speed_third_hour_l281_281403


namespace problem_solution_l281_281923

-- Define the conditions: stops and distance
def stops : Finset ℕ := Finset.range 15   -- Stops numbered 0 to 14 (instead of 1 to 15 for simplicity)
def distance (i j : ℕ) : ℕ := 100 * (abs (i - j))

-- Define the probability calculation
def probability_feet_le_500 : ℚ :=
  let valid_count := (2 * (5 + 6 + 7 + 8 + 9) + 5 * 10)
  let total_count := 15 * 14
  (valid_count : ℚ) / total_count

-- Define the final result m + n
def result_m_plus_n : ℕ :=
  let frac := probability_feet_le_500
  frac.num.natAbs + frac.denom.natAbs

-- State the final proof problem
theorem problem_solution : result_m_plus_n = 11 := by
  sorry

end problem_solution_l281_281923


namespace efficiency_ratio_l281_281126

variable (A_eff B_eff : ℝ)

-- Condition 1: A and B together finish a piece of work in 36 days
def combined_efficiency := A_eff + B_eff = 1 / 36

-- Condition 2: B alone finishes the work in 108 days
def B_efficiency := B_eff = 1 / 108

-- Theorem: Prove that the ratio of A's efficiency to B's efficiency is 2:1
theorem efficiency_ratio (h1 : combined_efficiency A_eff B_eff) (h2 : B_efficiency B_eff) : (A_eff / B_eff) = 2 := by
  sorry

end efficiency_ratio_l281_281126


namespace carpet_needed_in_sq_yards_l281_281016

theorem carpet_needed_in_sq_yards :
  let length := 15
  let width := 10
  let area_sq_feet := length * width
  let conversion_factor := 9
  let area_sq_yards := area_sq_feet / conversion_factor
  area_sq_yards = 16.67 := by
  sorry

end carpet_needed_in_sq_yards_l281_281016


namespace incorrect_statement_l281_281447

def angles_on_x_axis := {α : ℝ | ∃ (k : ℤ), α = k * Real.pi}
def angles_on_y_axis := {α : ℝ | ∃ (k : ℤ), α = Real.pi / 2 + k * Real.pi}
def angles_on_axes := {α : ℝ | ∃ (k : ℤ), α = k * Real.pi / 2}
def angles_on_y_eq_neg_x := {α : ℝ | ∃ (k : ℤ), α = Real.pi / 4 + 2 * k * Real.pi}

theorem incorrect_statement : ¬ (angles_on_y_eq_neg_x = {α : ℝ | ∃ (k : ℤ), α = Real.pi / 4 + 2 * k * Real.pi}) :=
sorry

end incorrect_statement_l281_281447


namespace simplify_fraction_l281_281407

theorem simplify_fraction : (3 : ℚ) / 462 + 17 / 42 = 95 / 231 :=
by sorry

end simplify_fraction_l281_281407


namespace riding_is_four_times_walking_l281_281475

variable (D : ℝ) -- Total distance of the route
variable (v_r v_w : ℝ) -- Riding speed and walking speed
variable (t_r t_w : ℝ) -- Time spent riding and walking

-- Conditions given in the problem
axiom distance_riding : (2/3) * D = v_r * t_r
axiom distance_walking : (1/3) * D = v_w * t_w
axiom time_relation : t_w = 2 * t_r

-- Desired statement to prove
theorem riding_is_four_times_walking : v_r = 4 * v_w := by
  sorry

end riding_is_four_times_walking_l281_281475


namespace slips_drawn_l281_281491

theorem slips_drawn (P : ℚ) (P_value : P = 24⁻¹) :
  ∃ n : ℕ, (n ≤ 5 ∧ P = (Nat.choose 5 n) / (Nat.choose 10 n) ∧ n = 4) := by
{
  sorry
}

end slips_drawn_l281_281491


namespace factor_polynomial_l281_281657

theorem factor_polynomial : 
  (∀ x : ℝ, (x^2 + 6 * x + 9 - 64 * x^4) = (-8 * x^2 + x + 3) * (8 * x^2 + x + 3)) :=
by
  intro x
  -- Left-hand side
  let lhs := x^2 + 6 * x + 9 - 64 * x^4
  -- Right-hand side after factorization
  let rhs := (-8 * x^2 + x + 3) * (8 * x^2 + x + 3)
  -- Prove the equality
  show lhs = rhs
  sorry

end factor_polynomial_l281_281657


namespace cos_pi_minus_alpha_proof_l281_281548

-- Define the initial conditions: angle α and point P
def α : ℝ := arbitrary ℝ
def P : ℝ × ℝ := (-1, 2)

-- Noncomputable to define functions involving real number calculations
noncomputable def hypotenuse : ℝ := real.sqrt ((P.1)^2 + (P.2)^2)
noncomputable def cos_alpha : ℝ := P.1 / hypotenuse
noncomputable def cos_pi_minus_alpha : ℝ := -cos_alpha

-- The theorem to prove
theorem cos_pi_minus_alpha_proof :
  cos_pi_minus_alpha = real.sqrt 5 / 5 :=
by sorry

end cos_pi_minus_alpha_proof_l281_281548


namespace find_teacher_age_l281_281123

noncomputable def age_of_teacher (avg_age_students : ℕ) (num_students : ℕ) 
                                (avg_age_inclusive : ℕ) (num_people_inclusive : ℕ) : ℕ :=
  let total_age_students := num_students * avg_age_students
  let total_age_inclusive := num_people_inclusive * avg_age_inclusive
  total_age_inclusive - total_age_students

theorem find_teacher_age : age_of_teacher 15 10 16 11 = 26 := 
by 
  sorry

end find_teacher_age_l281_281123


namespace calc1_l281_281324

theorem calc1 : (2 - Real.sqrt 3) ^ 0 - Real.sqrt 12 + Real.tan (Real.pi / 3) = 1 - Real.sqrt 3 :=
by
  sorry

end calc1_l281_281324


namespace expenditure_of_negative_l281_281208

def income := 5000
def expenditure (x : Int) : Int := -x

theorem expenditure_of_negative (x : Int) : expenditure (-x) = x :=
by
  sorry

example : expenditure (-400) = 400 :=
by 
  exact expenditure_of_negative 400

end expenditure_of_negative_l281_281208


namespace circle_equation_from_parabola_l281_281624

theorem circle_equation_from_parabola :
  let F := (2, 0)
  let A := (2, 4)
  let B := (2, -4)
  let diameter := 8
  let center := F
  let radius_squared := diameter^2 / 4
  (x - center.1)^2 + y^2 = radius_squared :=
by
  sorry

end circle_equation_from_parabola_l281_281624


namespace expression_equals_24_l281_281739

-- Given values
def a := 7
def b := 4
def c := 1
def d := 7

-- Statement to prove
theorem expression_equals_24 : (a - b) * (c + d) = 24 := by
  sorry

end expression_equals_24_l281_281739


namespace normalize_equation1_normalize_equation2_l281_281714

-- Define the first equation
def equation1 (x y : ℝ) := 2 * x - 3 * y - 10 = 0

-- Define the normalized form of the first equation
def normalized_equation1 (x y : ℝ) := (2 / Real.sqrt 13) * x - (3 / Real.sqrt 13) * y - (10 / Real.sqrt 13) = 0

-- Prove that the normalized form of the first equation is correct
theorem normalize_equation1 (x y : ℝ) (h : equation1 x y) : normalized_equation1 x y := 
sorry

-- Define the second equation
def equation2 (x y : ℝ) := 3 * x + 4 * y = 0

-- Define the normalized form of the second equation
def normalized_equation2 (x y : ℝ) := (3 / 5) * x + (4 / 5) * y = 0

-- Prove that the normalized form of the second equation is correct
theorem normalize_equation2 (x y : ℝ) (h : equation2 x y) : normalized_equation2 x y := 
sorry

end normalize_equation1_normalize_equation2_l281_281714


namespace jason_initial_speed_correct_l281_281980

noncomputable def jason_initial_speed (d : ℝ) (t1 t2 : ℝ) (v2 : ℝ) : ℝ :=
  let t_total := t1 + t2
  let d2 := v2 * t2
  let d1 := d - d2
  let v1 := d1 / t1
  v1

theorem jason_initial_speed_correct :
  jason_initial_speed 120 0.5 1 90 = 60 := 
by 
  sorry

end jason_initial_speed_correct_l281_281980


namespace days_not_worked_correct_l281_281476

def total_days : ℕ := 20
def earnings_for_work (days_worked : ℕ) : ℤ := 80 * days_worked
def penalty_for_no_work (days_not_worked : ℕ) : ℤ := -40 * days_not_worked
def final_earnings (days_worked days_not_worked : ℕ) : ℤ := 
  (earnings_for_work days_worked) + (penalty_for_no_work days_not_worked)
def received_amount : ℤ := 880

theorem days_not_worked_correct {y x : ℕ} 
  (h1 : x + y = total_days) 
  (h2 : final_earnings x y = received_amount) :
  y = 6 :=
sorry

end days_not_worked_correct_l281_281476


namespace tamia_bell_pepper_pieces_l281_281256

theorem tamia_bell_pepper_pieces :
  let bell_peppers := 5
  let slices_per_pepper := 20
  let initial_slices := bell_peppers * slices_per_pepper
  let half_slices_cut := initial_slices / 2
  let small_pieces := half_slices_cut * 3
  let total_pieces := (initial_slices - half_slices_cut) + small_pieces
  total_pieces = 200 :=
by
  sorry

end tamia_bell_pepper_pieces_l281_281256


namespace value_of_a_l281_281229

def A := { x : ℝ | x^2 - 8*x + 15 = 0 }
def B (a : ℝ) := { x : ℝ | x * a - 1 = 0 }

theorem value_of_a (a : ℝ) : (∀ x, x ∈ B a → x ∈ A) ↔ (a = 0 ∨ a = 1/3 ∨ a = 1/5) :=
by sorry

end value_of_a_l281_281229


namespace lcm_100_40_is_200_l281_281569

theorem lcm_100_40_is_200 : Nat.lcm 100 40 = 200 := by
  sorry

end lcm_100_40_is_200_l281_281569


namespace find_triples_l281_281937

theorem find_triples (x p n : ℕ) (hp : Nat.Prime p) :
  2 * x * (x + 5) = p^n + 3 * (x - 1) →
  (x = 2 ∧ p = 5 ∧ n = 2) ∨ (x = 0 ∧ p = 3 ∧ n = 1) :=
by
  sorry

end find_triples_l281_281937


namespace smallest_value_expression_l281_281512

theorem smallest_value_expression (x y : ℝ) (hx : -5 ≤ x ∧ x ≤ -3) (hy : 3 ≤ y ∧ y ≤ 5) :
  ∃ m, m = y ∧ m = 3 :=
by
  sorry

end smallest_value_expression_l281_281512


namespace josanna_minimum_test_score_l281_281553

theorem josanna_minimum_test_score 
  (scores : List ℕ) (target_increase : ℕ) (new_score : ℕ)
  (h_scores : scores = [92, 78, 84, 76, 88]) 
  (h_target_increase : target_increase = 5):
  (List.sum scores + new_score) / (List.length scores + 1) ≥ (List.sum scores / List.length scores + target_increase) →
  new_score = 114 :=
by
  sorry

end josanna_minimum_test_score_l281_281553


namespace smallest_positive_leading_coefficient_l281_281442

variable {a b c : ℚ} -- Define variables a, b, c that are rational numbers
variable (P : ℤ → ℚ) -- Define the polynomial P as a function from integers to rationals

-- State that P(x) is in the form of ax^2 + bx + c
def is_quadratic_polynomial (P : ℤ → ℚ) (a b c : ℚ) :=
  ∀ x : ℤ, P x = a * x^2 + b * x + c

-- State that P(x) takes integer values for all integer x
def takes_integer_values (P : ℤ → ℚ) :=
  ∀ x : ℤ, ∃ k : ℤ, P x = k

-- The statement we want to prove
theorem smallest_positive_leading_coefficient (h1 : is_quadratic_polynomial P a b c)
                                              (h2 : takes_integer_values P) :
  ∃ a : ℚ, 0 < a ∧ ∀ b c : ℚ, is_quadratic_polynomial P a b c → takes_integer_values P → a = 1/2 :=
sorry

end smallest_positive_leading_coefficient_l281_281442


namespace probability_composite_first_50_l281_281292

open Nat

def is_composite (n : ℕ) : Prop :=
  ¬ Prime n ∧ n ≠ 1

lemma first_50_composites_count : (Finset.filter is_composite (Finset.range 51)).card = 34 := 
sorry

theorem probability_composite_first_50 : 
  ((Finset.filter is_composite (Finset.range 51)).card : ℚ) / 50 = 17 / 25 :=
by 
  rw first_50_composites_count
  norm_cast
  exact (by norm_num : (34 : ℚ) / 50 = 17 / 25)

end probability_composite_first_50_l281_281292


namespace triangle_side_difference_l281_281693

theorem triangle_side_difference (y : ℝ) (h : y > 6) :
  max (y + 6) (y + 3) - min (y + 6) (y + 3) = 3 :=
by
  sorry

end triangle_side_difference_l281_281693


namespace total_number_of_applications_l281_281024

def in_state_apps := 200
def out_state_apps := 2 * in_state_apps
def total_apps := in_state_apps + out_state_apps

theorem total_number_of_applications : total_apps = 600 := by
  sorry

end total_number_of_applications_l281_281024


namespace smallest_non_factor_product_l281_281423

open Nat

def is_factor (n d : ℕ) := d > 0 ∧ n % d = 0

theorem smallest_non_factor_product (a b : ℕ) (h1 : a ≠ b) (h2 : is_factor 48 a) (h3 : is_factor 48 b) (h4 : ¬ is_factor 48 (a * b)) : a * b = 18 :=
by
  sorry

end smallest_non_factor_product_l281_281423


namespace no_odd_integer_trinomial_has_root_1_over_2022_l281_281149

theorem no_odd_integer_trinomial_has_root_1_over_2022 :
  ¬ ∃ (a b c : ℤ), (a % 2 = 1 ∧ b % 2 = 1 ∧ c % 2 = 1 ∧ (a * (1 / 2022)^2 + b * (1 / 2022) + c = 0)) :=
by
  sorry

end no_odd_integer_trinomial_has_root_1_over_2022_l281_281149


namespace Tiffany_total_score_l281_281419

def points_per_treasure_type : Type := ℕ × ℕ × ℕ
def treasures_per_level : Type := ℕ × ℕ × ℕ

def points (bronze silver gold : ℕ) : ℕ :=
  bronze * 6 + silver * 15 + gold * 30

def treasures_level1 : treasures_per_level := (2, 3, 1)
def treasures_level2 : treasures_per_level := (3, 1, 2)
def treasures_level3 : treasures_per_level := (5, 2, 1)

def total_points (l1 l2 l3 : treasures_per_level) : ℕ :=
  let (b1, s1, g1) := l1
  let (b2, s2, g2) := l2
  let (b3, s3, g3) := l3
  points b1 s1 g1 + points b2 s2 g2 + points b3 s3 g3

theorem Tiffany_total_score :
  total_points treasures_level1 treasures_level2 treasures_level3 = 270 :=
by
  sorry

end Tiffany_total_score_l281_281419


namespace tub_drain_time_l281_281121

theorem tub_drain_time (t : ℝ) (p q : ℝ) (h1 : t = 4) (h2 : p = 5 / 7) (h3 : q = 2 / 7) :
  q * t / p = 1.6 := by
  sorry

end tub_drain_time_l281_281121


namespace sum_of_a_and_b_l281_281838

theorem sum_of_a_and_b (a b : ℝ) (h1 : abs a = 5) (h2 : b = -2) (h3 : a * b > 0) : a + b = -7 := by
  sorry

end sum_of_a_and_b_l281_281838


namespace hcf_of_two_numbers_l281_281748

noncomputable def find_hcf (x y : ℕ) (lcm_xy : ℕ) (prod_xy : ℕ) : ℕ :=
  prod_xy / lcm_xy

theorem hcf_of_two_numbers (x y : ℕ) (lcm_xy: ℕ) (prod_xy: ℕ) 
  (h_lcm: lcm x y = lcm_xy) (h_prod: x * y = prod_xy) :
  find_hcf x y lcm_xy prod_xy = 75 :=
by
  sorry

end hcf_of_two_numbers_l281_281748


namespace find_angle_C_find_a_and_b_l281_281383

-- Conditions from the problem
variables {A B C : ℝ} {a b c : ℝ}
variables {m n : ℝ × ℝ}
variables (h1 : m = (Real.sin A, Real.sin B - Real.sin C))
variables (h2 : n = (a - Real.sqrt 3 * b, b + c))
variables (h3 : m.1 * n.1 + m.2 * n.2 = 0)
variables (h4 : ∀ θ ∈ Set.Ioo 0 Real.pi, θ ≠ C → Real.cos θ = (a^2 + b^2 - c^2) / (2 * a * b))

-- Hypotheses for part (2)
variables (circumradius : ℝ) (area : ℝ)
variables (h5 : circumradius = 2)
variables (h6 : area = Real.sqrt 3)
variables (h7 : a > b)

-- Theorem statement for part (1)
theorem find_angle_C (h1 : m = (Real.sin A, Real.sin B - Real.sin C))
  (h2 : n = (a - Real.sqrt 3 * b, b + c))
  (h3 : m.1 * n.1 + m.2 * n.2 = 0)
  (h4 : ∀ C ∈ Set.Ioo 0 Real.pi, Real.cos C = (a^2 + b^2 - c^2) / (2 * a * b)) : 
  C = Real.pi / 6 := sorry

-- Theorem statement for part (2)
theorem find_a_and_b (circumradius : ℝ) (area : ℝ) (a b : ℝ)
  (h5 : circumradius = 2) (h6 : area = Real.sqrt 3) (h7 : a > b)
  (h8 : ∀ C ∈ Set.Ioo 0 Real.pi, Real.cos C = (a^2 + b^2 - c^2) / (2 * a * b))
  (h9 : Real.sin C ≠ 0): 
  a = 2 * Real.sqrt 3 ∧ b = 2 := sorry

end find_angle_C_find_a_and_b_l281_281383


namespace evaluate_expression_l281_281649

theorem evaluate_expression : 5000 * 5000^3000 = 5000^3001 := 
by sorry

end evaluate_expression_l281_281649


namespace prove_value_l281_281950

variable (m n : ℤ)

-- Conditions from the problem
def condition1 : Prop := m^2 + 2 * m * n = 384
def condition2 : Prop := 3 * m * n + 2 * n^2 = 560

-- Proposition to be proved
theorem prove_value (h1 : condition1 m n) (h2 : condition2 m n) : 2 * m^2 + 13 * m * n + 6 * n^2 - 444 = 2004 := by
  sorry

end prove_value_l281_281950


namespace total_disks_in_bag_l281_281029

/-- Given that the number of blue disks b, yellow disks y, and green disks g are in the ratio 3:7:8,
    and there are 30 more green disks than blue disks (g = b + 30),
    prove that the total number of disks is 108. -/
theorem total_disks_in_bag (b y g : ℕ) (h1 : 3 * y = 7 * b) (h2 : 8 * y = 7 * g) (h3 : g = b + 30) :
  b + y + g = 108 := by
  sorry

end total_disks_in_bag_l281_281029


namespace solve_a1_solve_a2_l281_281058

noncomputable def initial_volume := 1  -- in m^3
noncomputable def initial_pressure := 10^5  -- in Pa
noncomputable def initial_temperature := 300  -- in K

theorem solve_a1 (a1 : ℝ) : a1 = -10^5 :=
  sorry

theorem solve_a2 (a2 : ℝ) : a2 = -1.4 * 10^5 :=
  sorry

end solve_a1_solve_a2_l281_281058


namespace car_speed_l281_281483

theorem car_speed (uses_one_gallon_per_30_miles : ∀ d : ℝ, d = 30 → d / 30 ≥ 1)
    (full_tank : ℝ := 10)
    (travel_time : ℝ := 5)
    (fraction_of_tank_used : ℝ := 0.8333333333333334)
    (speed : ℝ := 50) :
  let amount_of_gasoline_used := fraction_of_tank_used * full_tank
  let distance_traveled := amount_of_gasoline_used * 30
  speed = distance_traveled / travel_time :=
by
  sorry

end car_speed_l281_281483


namespace arithmetic_sequence_50th_term_l281_281930

theorem arithmetic_sequence_50th_term :
  let a_1 := 3
  let d := 7
  let n := 50
  (a_1 + (n - 1) * d) = 346 :=
by
  let a_1 := 3
  let d := 7
  let n := 50
  show (a_1 + (n - 1) * d) = 346
  sorry

end arithmetic_sequence_50th_term_l281_281930


namespace max_k_consecutive_sum_l281_281367

theorem max_k_consecutive_sum :
  ∃ n : ℕ, n > 0 ∧ ∃ k : ℕ, k * (2 * n + k - 1) = 2^2 * 3^8 ∧ ∀ k' > k, ¬ ∃ n', n' > 0 ∧ k' * (2 * n' + k' - 1) = 2^2 * 3^8 := sorry

end max_k_consecutive_sum_l281_281367


namespace no_perfect_powers_in_sequence_l281_281698

noncomputable def nth_triplet (n : Nat) : Nat × Nat × Nat :=
  Nat.recOn n (2, 3, 5) (λ _ ⟨a, b, c⟩ => (a + c, a + b, b + c))

def is_perfect_power (x : Nat) : Prop :=
  ∃ (m : Nat) (k : Nat), k ≥ 2 ∧ m^k = x

theorem no_perfect_powers_in_sequence : ∀ (n : Nat), ∀ (a b c : Nat),
  nth_triplet n = (a, b, c) →
  ¬(is_perfect_power a ∨ is_perfect_power b ∨ is_perfect_power c) :=
by
  intros
  sorry

end no_perfect_powers_in_sequence_l281_281698


namespace total_length_segments_l281_281136

noncomputable def segment_length (rect_horizontal_1 rect_horizontal_2 rect_vertical left_segment : ℕ) :=
  let total_length := rect_horizontal_1 + rect_horizontal_2 + rect_vertical
  total_length - 8 + left_segment

theorem total_length_segments
  (rect_horizontal_1 rect_horizontal_2 rect_vertical left_segment total_left : ℕ)
  (h1 : rect_horizontal_1 = 10)
  (h2 : rect_horizontal_2 = 3)
  (h3 : rect_vertical = 12)
  (h4 : left_segment = 8)
  (h5 : total_left = 19)
  : segment_length rect_horizontal_1 rect_horizontal_2 rect_vertical left_segment = total_left :=
sorry

end total_length_segments_l281_281136


namespace circle_equation_l281_281675

def circle_center : (ℝ × ℝ) := (1, 2)
def radius : ℝ := 3

theorem circle_equation : 
  (∀ x y : ℝ, (x - circle_center.1) ^ 2 + (y - circle_center.2) ^ 2 = radius ^ 2 ↔ 
  (x - 1) ^ 2 + (y - 2) ^ 2 = 9) := 
by
  sorry

end circle_equation_l281_281675


namespace complete_the_square_d_l281_281285

theorem complete_the_square_d (x : ℝ) : (∃ c d : ℝ, x^2 + 6 * x - 4 = 0 → (x + c)^2 = d) ∧ d = 13 :=
by
  sorry

end complete_the_square_d_l281_281285


namespace player_1_winning_strategy_l281_281787

-- Define the properties and rules of the game
def valid_pair (a b : ℕ) : Prop := a > 0 ∧ b > 0 ∧ a + b = 2005

def move (current t a b : ℕ) : Prop := 
  current = t - a ∨ current = t - b

def first_player_wins (t a b : ℕ) : Prop :=
  ∀ k : ℕ, t > k * 2005 → ∃ m : ℕ, move (t - m) t a b

-- Main theorem statement
theorem player_1_winning_strategy : ∃ (t : ℕ) (a b : ℕ), valid_pair a b ∧ first_player_wins t a b :=
sorry

end player_1_winning_strategy_l281_281787


namespace factor_poly_l281_281653

theorem factor_poly : 
  ∃ (a b c d e f : ℤ), a < d ∧ (a*x^2 + b*x + c)*(d*x^2 + e*x + f) = (x^2 + 6*x + 9 - 64*x^4) ∧ 
  (a = -8 ∧ b = 1 ∧ c = 3 ∧ d = 8 ∧ e = 1 ∧ f = 3) := 
sorry

end factor_poly_l281_281653


namespace find_value_l281_281840

variable (x y a c : ℝ)

-- Conditions
def condition1 : Prop := x * y = 2 * c
def condition2 : Prop := (1 / x ^ 2) + (1 / y ^ 2) = 3 * a

-- Proof statement
theorem find_value : condition1 x y c ∧ condition2 x y a ↔ (x + y) ^ 2 = 12 * a * c ^ 2 + 4 * c := 
by 
  -- Placeholder for the actual proof
  sorry

end find_value_l281_281840


namespace find_exponent_l281_281671

theorem find_exponent (x : ℕ) (h : 2^x + 2^x + 2^x + 2^x + 2^x = 2048) : x = 9 :=
sorry

end find_exponent_l281_281671


namespace m_mul_m_add_1_not_power_of_integer_l281_281243

theorem m_mul_m_add_1_not_power_of_integer (m n k : ℕ) : m * (m + 1) ≠ n^k :=
by
  sorry

end m_mul_m_add_1_not_power_of_integer_l281_281243


namespace negation_of_proposition_l281_281731

theorem negation_of_proposition :
  (¬ ∃ x : ℝ, x^2 + 2*x > 0) ↔ (∀ x : ℝ, x^2 + 2*x ≤ 0) :=
sorry

end negation_of_proposition_l281_281731


namespace egg_processing_plant_l281_281220

-- Definitions based on the conditions
def original_ratio (E : ℕ) : ℕ × ℕ := (24 * E / 25, E / 25)
def new_ratio (E : ℕ) : ℕ × ℕ := (99 * E / 100, E / 100)

-- The mathematical proof problem
theorem egg_processing_plant (E : ℕ) (h : new_ratio E = (original_ratio E).fst + 12, (original_ratio E).snd) : E = 400 := 
  sorry

end egg_processing_plant_l281_281220


namespace prob_at_least_one_2_in_two_8_sided_dice_l281_281106

/-- Probability of getting at least one 2 when rolling two 8-sided dice -/
theorem prob_at_least_one_2_in_two_8_sided_dice : 
  let total_outcomes := 64
  let favorable_outcomes := 15
  (favorable_outcomes : ℝ) / (total_outcomes : ℝ) = 15 / 64 :=
by
  sorry

end prob_at_least_one_2_in_two_8_sided_dice_l281_281106


namespace remainder_2n_div_9_l281_281122

theorem remainder_2n_div_9 (n : ℤ) (h : n % 18 = 10) : (2 * n) % 9 = 2 := 
sorry

end remainder_2n_div_9_l281_281122


namespace jan_25_on_thursday_l281_281932

/-- 
  Given that December 25 is on Monday,
  prove that January 25 in the following year falls on Thursday.
-/
theorem jan_25_on_thursday (day_of_week : Fin 7) (h : day_of_week = 0) : 
  ((day_of_week + 31) % 7 + 25) % 7 = 4 := 
sorry

end jan_25_on_thursday_l281_281932


namespace problem1_l281_281294

theorem problem1 (x y z : ℝ) (h : x^2 + y^2 + z^2 = 1) : 
  |x - y| + |y - z| + |z - x| ≤ 2 * Real.sqrt 2 := 
sorry

end problem1_l281_281294


namespace sin_half_alpha_l281_281186

theorem sin_half_alpha (α : ℝ) (hα : 0 < α ∧ α < π / 2) (hcos : Real.cos α = (1 + Real.sqrt 5) / 4) : 
  Real.sin (α / 2) = (Real.sqrt 5 - 1) / 4 := 
by 
  sorry

end sin_half_alpha_l281_281186


namespace distance_between_lines_l281_281879

/-- Define the lines by their equations -/
def line1 (x y : ℝ) : Prop := 3 * x + 4 * y - 12 = 0
def line2 (x y : ℝ) : Prop := 6 * x + 8 * y + 6 = 0

/-- Define the simplified form of the second line -/
def simplified_line2 (x y : ℝ) : Prop := 3 * x + 4 * y + 3 = 0

/-- Prove the distance between the two lines is 3 -/
theorem distance_between_lines : 
  let A : ℝ := 3
  let B : ℝ := 4
  let C1 : ℝ := -12
  let C2 : ℝ := 3
  (|C2 - C1| / Real.sqrt (A^2 + B^2) = 3) :=
by
  sorry

end distance_between_lines_l281_281879


namespace max_m_sufficient_min_m_necessary_l281_281669

-- Define variables and conditions
variables (x m : ℝ) (p : Prop := abs x ≤ m) (q : Prop := -1 ≤ x ∧ x ≤ 4) 

-- Problem 1: Maximum value of m for sufficient condition
theorem max_m_sufficient : (∀ x, abs x ≤ m → (-1 ≤ x ∧ x ≤ 4)) → m = 4 := sorry

-- Problem 2: Minimum value of m for necessary condition
theorem min_m_necessary : (∀ x, (-1 ≤ x ∧ x ≤ 4) → abs x ≤ m) → m = 4 := sorry

end max_m_sufficient_min_m_necessary_l281_281669


namespace incorrect_statement_D_l281_281311

theorem incorrect_statement_D :
  ¬ (abs (-1) - abs 1 = 2) :=
by
  sorry

end incorrect_statement_D_l281_281311


namespace jello_cost_calculation_l281_281855

-- Conditions as definitions
def jello_per_pound : ℝ := 1.5
def tub_volume_cubic_feet : ℝ := 6
def cubic_foot_to_gallons : ℝ := 7.5
def gallon_weight_pounds : ℝ := 8
def cost_per_tablespoon_jello : ℝ := 0.5

-- Tub total water calculation
def tub_water_gallons (volume_cubic_feet : ℝ) (cubic_foot_to_gallons : ℝ) : ℝ :=
  volume_cubic_feet * cubic_foot_to_gallons

-- Water weight calculation
def water_weight_pounds (water_gallons : ℝ) (gallon_weight_pounds : ℝ) : ℝ :=
  water_gallons * gallon_weight_pounds

-- Jello mix required calculation
def jello_mix_tablespoons (water_pounds : ℝ) (jello_per_pound : ℝ) : ℝ :=
  water_pounds * jello_per_pound

-- Total cost calculation
def total_cost (jello_mix_tablespoons : ℝ) (cost_per_tablespoon_jello : ℝ) : ℝ :=
  jello_mix_tablespoons * cost_per_tablespoon_jello

-- Theorem statement
theorem jello_cost_calculation :
  total_cost (jello_mix_tablespoons (water_weight_pounds (tub_water_gallons tub_volume_cubic_feet cubic_foot_to_gallons) gallon_weight_pounds) jello_per_pound) cost_per_tablespoon_jello = 270 := 
by sorry

end jello_cost_calculation_l281_281855


namespace triangle_inequality_l281_281953

theorem triangle_inequality (a b c : ℝ) (habc : a + b > c ∧ a + c > b ∧ b + c > a) :
  a^2 * (b + c - a) + b^2 * (c + a - b) + c^2 * (a + b - c) ≤ 3 * a * b * c :=
by
  sorry

end triangle_inequality_l281_281953


namespace word_count_proof_l281_281526

def letters : List Char := ['A', 'B', 'C', 'D', 'E', 'F']
def consonants : List Char := ['B', 'C', 'D', 'F']
def vowels : List Char := ['A', 'E']

def count_unrestricted_words : ℕ := 6 ^ 5
def count_all_vowel_words : ℕ := 2 ^ 5
def count_one_consonant_words : ℕ := 5 * 4 * (2 ^ 4)
def count_fewer_than_two_consonant_words : ℕ := count_all_vowel_words + count_one_consonant_words

def count_words_with_at_least_two_consonants : ℕ :=
  count_unrestricted_words - count_fewer_than_two_consonant_words

theorem word_count_proof :
  count_words_with_at_least_two_consonants = 7424 := 
by
  -- Proof will be provided here. For now we skip it.
  sorry

end word_count_proof_l281_281526


namespace points_on_line_l281_281490

-- Define the points
def P1 : (ℝ × ℝ) := (8, 16)
def P2 : (ℝ × ℝ) := (2, 4)

-- Define the line equation as a predicate
def on_line (m b : ℝ) (p : ℝ × ℝ) : Prop := p.2 = m * p.1 + b

-- Define the given points to be checked
def P3 : (ℝ × ℝ) := (5, 10)
def P4 : (ℝ × ℝ) := (7, 14)
def P5 : (ℝ × ℝ) := (4, 7)
def P6 : (ℝ × ℝ) := (10, 20)
def P7 : (ℝ × ℝ) := (3, 6)

theorem points_on_line :
  let m := 2
  let b := 0
  on_line m b P3 ∧
  on_line m b P4 ∧
  ¬ on_line m b P5 ∧
  on_line m b P6 ∧
  on_line m b P7 :=
by
  sorry

end points_on_line_l281_281490


namespace quadratic_expression_l281_281044

theorem quadratic_expression (x y : ℝ) 
  (h1 : 3 * x + 2 * y = 6) 
  (h2 : 2 * x + 3 * y = 8) : 
  13 * x^2 + 22 * x * y + 13 * y^2 = 98.08 := 
by sorry

end quadratic_expression_l281_281044


namespace find_sides_of_isosceles_triangle_l281_281878

noncomputable def isosceles_triangle_sides (b a : ℝ) : Prop :=
  ∃ (AI IL₁ : ℝ), AI = 5 ∧ IL₁ = 3 ∧
  b = 10 ∧ a = 12 ∧
  a = (6 / 5) * b ∧
  (b^2 = 8^2 + (3/5 * b)^2)

-- Proof problem statement
theorem find_sides_of_isosceles_triangle :
  ∀ (b a : ℝ), isosceles_triangle_sides b a → b = 10 ∧ a = 12 :=
by
  intros b a h
  sorry

end find_sides_of_isosceles_triangle_l281_281878


namespace probability_at_least_one_two_l281_281108

def num_dice := 2
def sides_dice := 8
def total_outcomes := sides_dice ^ num_dice
def num_non_favorable_outcomes := (sides_dice - 1) ^ num_dice
def num_favorable_outcomes := total_outcomes - num_non_favorable_outcomes
def probability_favorable_outcomes := (15 : ℚ) / (64 : ℚ)

theorem probability_at_least_one_two :
  probability_favorable_outcomes = 15 / 64 :=
sorry

end probability_at_least_one_two_l281_281108


namespace simplify_and_evaluate_expr_l281_281079

noncomputable def a : ℝ := Real.sqrt 2 - 2

noncomputable def expr (a : ℝ) : ℝ := (a - 2) / (a - 1) / (a + 1 - 3 / (a - 1))

theorem simplify_and_evaluate_expr :
  expr (Real.sqrt 2 - 2) = Real.sqrt 2 / 2 :=
by sorry

end simplify_and_evaluate_expr_l281_281079


namespace sin_half_alpha_l281_281168

variable (α : ℝ)
variable (h₀ : 0 < α ∧ α < π/2)
variable (h₁ : cos α = (1 + sqrt 5) / 4)

theorem sin_half_alpha : sin (α / 2) = (sqrt 5 - 1) / 4 :=
by
  sorry

end sin_half_alpha_l281_281168


namespace rectangle_area_l281_281131

theorem rectangle_area (ABCD : Type*) (small_square : ℕ) (shaded_squares : ℕ) (side_length : ℕ) 
  (shaded_area : ℕ) (width : ℕ) (height : ℕ)
  (H1 : shaded_squares = 3) 
  (H2 : side_length = 2)
  (H3 : shaded_area = side_length * side_length)
  (H4 : width = 6)
  (H5 : height = 4)
  : (width * height) = 24 :=
by
  sorry

end rectangle_area_l281_281131


namespace number_of_cakes_l281_281579

theorem number_of_cakes (total_eggs eggs_in_fridge eggs_per_cake : ℕ) (h1 : total_eggs = 60) (h2 : eggs_in_fridge = 10) (h3 : eggs_per_cake = 5) :
  (total_eggs - eggs_in_fridge) / eggs_per_cake = 10 :=
by
  sorry

end number_of_cakes_l281_281579


namespace elvis_writing_time_per_song_l281_281935

-- Define the conditions based on the problem statement
def total_studio_time_minutes := 300   -- 5 hours converted to minutes
def songs := 10
def recording_time_per_song := 12
def total_editing_time := 30

-- Define the total recording time
def total_recording_time := songs * recording_time_per_song

-- Define the total time available for writing songs
def total_writing_time := total_studio_time_minutes - total_recording_time - total_editing_time

-- Define the time to write each song
def time_per_song_writing := total_writing_time / songs

-- State the proof goal
theorem elvis_writing_time_per_song : time_per_song_writing = 15 := by
  sorry

end elvis_writing_time_per_song_l281_281935


namespace part1_part2_l281_281390

-- Define the conditions p and q
def p (a x : ℝ) : Prop := (x - a) * (x - 3 * a) < 0
def q (x : ℝ) : Prop := (x - 2) * (x - 4) < 0 ∧ (x - 3) * (x - 5) > 0

-- Problem Part 1: Prove that if a = 1 and p ∧ q is true, then 2 < x < 3
theorem part1 (x : ℝ) : p 1 x ∧ q x → 2 < x ∧ x < 3 :=
by
  intro h
  sorry

-- Problem Part 2: Prove that if p is a necessary but not sufficient condition for q, then 1 ≤ a ≤ 2
theorem part2 (a : ℝ) : (∀ x, q x → p a x) ∧ (∃ x, p a x ∧ ¬q x) → 1 ≤ a ∧ a ≤ 2 :=
by
  intro h
  sorry

end part1_part2_l281_281390


namespace baker_final_stock_l281_281640

-- Given conditions as Lean definitions
def initial_cakes : Nat := 173
def additional_cakes : Nat := 103
def damaged_percentage : Nat := 25
def sold_first_day : Nat := 86
def sold_next_day_percentage : Nat := 10

-- Calculate new cakes Baker adds to the stock after accounting for damaged cakes
def new_undamaged_cakes : Nat := (additional_cakes * (100 - damaged_percentage)) / 100

-- Calculate stock after adding new cakes
def stock_after_new_cakes : Nat := initial_cakes + new_undamaged_cakes

-- Calculate stock after first day's sales
def stock_after_first_sale : Nat := stock_after_new_cakes - sold_first_day

-- Calculate cakes sold on the second day
def sold_next_day : Nat := (stock_after_first_sale * sold_next_day_percentage) / 100

-- Final stock calculations
def final_stock : Nat := stock_after_first_sale - sold_next_day

-- Prove that Baker has 148 cakes left
theorem baker_final_stock : final_stock = 148 := by
  sorry

end baker_final_stock_l281_281640


namespace geometric_sequence_sum_l281_281692

theorem geometric_sequence_sum
  (a : ℕ → ℝ)
  (r : ℝ)
  (h_geom : ∀ n, a (n + 1) = a n * r)
  (h1 : a 1 + a 3 = 20)
  (h2 : a 2 + a 4 = 40)
  :
  a 3 + a 5 = 80 :=
sorry

end geometric_sequence_sum_l281_281692


namespace factorization_of_polynomial_l281_281651

theorem factorization_of_polynomial : 
  (x^2 + 6 * x + 9 - 64 * x^4) = (-8 * x^2 + x + 3) * (8 * x^2 + x + 3) :=
by sorry

end factorization_of_polynomial_l281_281651


namespace scientific_notation_l281_281546

theorem scientific_notation (n : ℝ) (h : n = 40.9 * 10^9) : n = 4.09 * 10^10 :=
by sorry

end scientific_notation_l281_281546


namespace smallest_non_factor_product_l281_281421

open Nat

def is_factor (n d : ℕ) := d > 0 ∧ n % d = 0

theorem smallest_non_factor_product (a b : ℕ) (h1 : a ≠ b) (h2 : is_factor 48 a) (h3 : is_factor 48 b) (h4 : ¬ is_factor 48 (a * b)) : a * b = 18 :=
by
  sorry

end smallest_non_factor_product_l281_281421


namespace problem_part1_problem_part2_l281_281070

noncomputable def f (x a : ℝ) := |x - a| + x

theorem problem_part1 (a : ℝ) (h_a : a = 1) :
  {x : ℝ | f x a ≥ x + 2} = {x : ℝ | x ≥ 3} ∪ {x : ℝ | x ≤ -1} :=
by 
  simp [h_a, f]
  sorry

theorem problem_part2 (a : ℝ) (h_solution : {x : ℝ | f x a ≤ 3 * x} = {x : ℝ | x ≥ 2}) :
  a = 6 :=
by
  simp [f] at h_solution
  sorry

end problem_part1_problem_part2_l281_281070


namespace find_divisor_l281_281749

-- Definitions
def dividend := 199
def quotient := 11
def remainder := 1

-- Statement of the theorem
theorem find_divisor : ∃ x : ℕ, dividend = (x * quotient) + remainder ∧ x = 18 := by
  sorry

end find_divisor_l281_281749


namespace find_a2_plus_b2_l281_281235

theorem find_a2_plus_b2 (a b : ℝ) (h1 : a * b = 16) (h2 : a + b = 10) : a^2 + b^2 = 68 :=
sorry

end find_a2_plus_b2_l281_281235


namespace egg_processing_l281_281219

theorem egg_processing (E : ℕ) 
  (h1 : (24 / 25) * E + 12 = (99 / 100) * E) : 
  E = 400 :=
sorry

end egg_processing_l281_281219


namespace minimum_area_isosceles_trapezoid_l281_281150

theorem minimum_area_isosceles_trapezoid (r x a d : ℝ) (h_circumscribed : a + d = 2 * x) (h_minimal : x ≥ 2 * r) :
  4 * r^2 ≤ (a + d) * r :=
by sorry

end minimum_area_isosceles_trapezoid_l281_281150


namespace smallest_x_plus_y_l281_281811

theorem smallest_x_plus_y (x y : ℕ) (hx_pos : 0 < x) (hy_pos : 0 < y) (hxy_neq : x ≠ y) (h_eq : 1 / (x:ℝ) + 1 / (y:ℝ) = 1 / 15) : x + y = 64 :=
sorry

end smallest_x_plus_y_l281_281811


namespace sum_of_fourth_powers_l281_281588

theorem sum_of_fourth_powers (n : ℤ) 
  (h : n * (n + 1) * (n + 2) = 12 * (n + (n + 1) + (n + 2))) : 
  (n^4 + (n + 1)^4 + (n + 2)^4) = 7793 := 
by 
  sorry

end sum_of_fourth_powers_l281_281588


namespace trajectory_no_intersection_distance_AB_l281_281040

variable (M : Type) [MetricSpace M]

-- Point M on the plane
variable (M : ℝ × ℝ)

-- Given conditions
def condition1 (M : ℝ × ℝ) : Prop := 
  (Real.sqrt ((M.1 - 8)^2 + M.2^2) = 2 * Real.sqrt ((M.1 - 2)^2 + M.2^2))

-- 1. Proving the trajectory C of M
theorem trajectory (M : ℝ × ℝ) (h : condition1 M) : M.1^2 + M.2^2 = 16 :=
by
  sorry

-- 2. Range of values for k such that y = kx - 5 does not intersect trajectory C
theorem no_intersection (k : ℝ) : 
  (∀ (x y : ℝ), x^2 + y^2 = 16 → y ≠ k * x - 5) ↔ (-3 / 4 < k ∧ k < 3 / 4) :=
by
  sorry

-- 3. Distance between intersection points A and B of given circles
def intersection_condition (x y : ℝ) : Prop :=
  (x^2 + y^2 = 16) ∧ (x^2 + y^2 - 8 * x - 8 * y + 16 = 0)

theorem distance_AB (A B : ℝ × ℝ) (hA : intersection_condition A.1 A.2) (hB : intersection_condition B.1 B.2) : 
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 4 * Real.sqrt 2 :=
by
  sorry

end trajectory_no_intersection_distance_AB_l281_281040


namespace sequence_sum_S6_l281_281507

theorem sequence_sum_S6 (a_n : ℕ → ℕ) (S_n : ℕ → ℕ) (h : ∀ n, S_n n = 2 * a_n n - 3) :
  S_n 6 = 189 :=
by
  sorry

end sequence_sum_S6_l281_281507


namespace taco_truck_earnings_l281_281470

/-
Question: How many dollars did the taco truck make during the lunch rush?
Conditions:
1. Soft tacos are $2 each.
2. Hard shell tacos are $5 each.
3. The family buys 4 hard shell tacos and 3 soft tacos.
4. There are ten other customers.
5. Each of the ten other customers buys 2 soft tacos.
Answer: The taco truck made $66 during the lunch rush.
-/

theorem taco_truck_earnings :
  let soft_taco_price := 2
  let hard_taco_price := 5
  let family_hard_tacos := 4
  let family_soft_tacos := 3
  let other_customers := 10
  let other_customers_soft_tacos := 2
  (family_hard_tacos * hard_taco_price + family_soft_tacos * soft_taco_price +
   other_customers * other_customers_soft_tacos * soft_taco_price) = 66 := by
  sorry

end taco_truck_earnings_l281_281470


namespace sin_half_alpha_l281_281183

theorem sin_half_alpha (α : ℝ) (hα1 : 0 < α ∧ α < π / 2) (hα2 : Real.cos α = (1 + Real.sqrt 5) / 4) : 
    Real.sin (α / 2) = (-1 + Real.sqrt 5) / 4 := 
by
  sorry

end sin_half_alpha_l281_281183


namespace smallest_non_factor_product_l281_281422

open Nat

def is_factor (n d : ℕ) := d > 0 ∧ n % d = 0

theorem smallest_non_factor_product (a b : ℕ) (h1 : a ≠ b) (h2 : is_factor 48 a) (h3 : is_factor 48 b) (h4 : ¬ is_factor 48 (a * b)) : a * b = 18 :=
by
  sorry

end smallest_non_factor_product_l281_281422


namespace smallest_multiple_of_2019_of_form_abcabcabc_l281_281503

def is_digit (n : ℕ) : Prop := n < 10

theorem smallest_multiple_of_2019_of_form_abcabcabc
    (a b c : ℕ)
    (h_a : is_digit a)
    (h_b : is_digit b)
    (h_c : is_digit c)
    (k : ℕ)
    (form : Nat)
    (rep: ℕ) : 
  (form = (a * 100 + b * 10 + c) * rep) →
  (∃ n : ℕ, form = 2019 * n) →
  form >= 673673673 :=
sorry

end smallest_multiple_of_2019_of_form_abcabcabc_l281_281503


namespace area_difference_of_tablets_l281_281006

theorem area_difference_of_tablets 
  (d1 d2 : ℝ) (s1 s2 : ℝ)
  (h1 : d1 = 6) (h2 : d2 = 5) 
  (hs1 : d1^2 = 2 * s1^2) (hs2 : d2^2 = 2 * s2^2) 
  (A1 : ℝ) (A2 : ℝ) (hA1 : A1 = s1^2) (hA2 : A2 = s2^2)
  : A1 - A2 = 5.5 := 
sorry

end area_difference_of_tablets_l281_281006


namespace minimum_value_of_GP_l281_281221

theorem minimum_value_of_GP (a : ℕ → ℝ) (h : ∀ n, 0 < a n) (h_prod : a 2 * a 10 = 9) :
  a 5 + a 7 = 6 :=
by
  -- proof steps will be filled in here
  sorry

end minimum_value_of_GP_l281_281221


namespace smallest_x_plus_y_l281_281810

theorem smallest_x_plus_y (x y : ℕ) (hx_pos : 0 < x) (hy_pos : 0 < y) (hxy_neq : x ≠ y) (h_eq : 1 / (x:ℝ) + 1 / (y:ℝ) = 1 / 15) : x + y = 64 :=
sorry

end smallest_x_plus_y_l281_281810


namespace ticTacToeConfigCorrect_l281_281140

def ticTacToeConfigCount (board : Fin 3 → Fin 3 → Option Char) : Nat := 
  sorry -- this function will count the configurations according to the game rules

theorem ticTacToeConfigCorrect (board : Fin 3 → Fin 3 → Option Char) :
  ticTacToeConfigCount board = 438 := 
  sorry

end ticTacToeConfigCorrect_l281_281140


namespace intersecting_lines_l281_281538

theorem intersecting_lines (m n : ℝ) : 
  (∀ x y : ℝ, y = x / 2 + n → y = mx - 1 → (x = 1 ∧ y = -2)) → 
  m = -1 ∧ n = -5 / 2 :=
by
  sorry

end intersecting_lines_l281_281538


namespace smallest_x_plus_y_l281_281808

theorem smallest_x_plus_y (x y : ℕ) (hx_pos : 0 < x) (hy_pos : 0 < y) (hxy_neq : x ≠ y) (h_eq : 1 / (x:ℝ) + 1 / (y:ℝ) = 1 / 15) : x + y = 64 :=
sorry

end smallest_x_plus_y_l281_281808


namespace find_n_l281_281666

theorem find_n : ∃ n : ℕ, 5^3 - 7 = 6^2 + n ∧ n = 82 := by
  use 82
  sorry

end find_n_l281_281666


namespace tan_alpha_gt_tan_beta_l281_281051

open Real

theorem tan_alpha_gt_tan_beta (α β : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : 0 < β ∧ β < π / 2) (h3 : sin α > sin β) : tan α > tan β :=
sorry

end tan_alpha_gt_tan_beta_l281_281051


namespace min_value_of_a_plus_2b_l281_281342

theorem min_value_of_a_plus_2b (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 1/b = 1) : a + 2*b = 3 + 2*Real.sqrt 2 := 
sorry

end min_value_of_a_plus_2b_l281_281342


namespace exchange_5_dollars_to_francs_l281_281916

-- Define the exchange rates
def dollar_to_lire (d : ℕ) : ℕ := d * 5000
def lire_to_francs (l : ℕ) : ℕ := (l / 1000) * 3

-- Define the main theorem
theorem exchange_5_dollars_to_francs : lire_to_francs (dollar_to_lire 5) = 75 :=
by
  sorry

end exchange_5_dollars_to_francs_l281_281916


namespace question1_question2_l281_281157

-- Condition: p
def p (x : ℝ) : Prop := 4 * x^2 + 12 * x - 7 ≤ 0

-- Condition: q
def q (a : ℝ) (x : ℝ) : Prop := a - 3 ≤ x ∧ x ≤ a + 3

-- Question 1 statement: Given p is true and q is false when a = 0, find range of x
theorem question1 (x : ℝ) (h : p x ∧ ¬q 0 x) : -7/2 ≤ x ∧ x < -3 :=
sorry

-- Question 2 statement: If p is a sufficient condition for q, find range of a
theorem question2 (a : ℝ) (h : ∀ x, p x → q a x) : -5/2 ≤ a ∧ a ≤ 1/2 :=
sorry

end question1_question2_l281_281157


namespace series_sum_l281_281022

open BigOperators

theorem series_sum :
  (∑ n in Finset.range 99, (1 : ℝ) / ((n + 1) * (n + 2))) = 99 / 100 :=
by
  sorry

end series_sum_l281_281022


namespace problem_f8_f2018_l281_281445

theorem problem_f8_f2018 (f : ℕ → ℝ) (h₀ : ∀ n, f (n + 3) = (f n - 1) / (f n + 1)) 
  (h₁ : f 1 ≠ 0) (h₂ : f 1 ≠ 1) (h₃ : f 1 ≠ -1) : 
  f 8 * f 2018 = -1 :=
sorry

end problem_f8_f2018_l281_281445


namespace initial_welders_count_l281_281727

theorem initial_welders_count
  (W : ℕ)
  (complete_in_5_days : W * 5 = 1)
  (leave_after_1_day : 12 ≤ W) 
  (remaining_complete_in_6_days : (W - 12) * 6 = 1) : 
  W = 72 :=
by
  -- proof steps here
  sorry

end initial_welders_count_l281_281727


namespace toy_sword_cost_l281_281227

theorem toy_sword_cost (L S : ℕ) (play_dough_cost total_cost : ℕ) :
    L = 250 →
    play_dough_cost = 35 →
    total_cost = 1940 →
    3 * L + 7 * S + 10 * play_dough_cost = total_cost →
    S = 120 :=
by
  intros hL h_play_dough_cost h_total_cost h_eq
  sorry

end toy_sword_cost_l281_281227


namespace intersection_M_N_l281_281864

def M : Set ℤ := {0}
def N : Set ℤ := {x | -1 < x ∧ x < 1}

theorem intersection_M_N : M ∩ N = {0} :=
by
  sorry

end intersection_M_N_l281_281864


namespace constant_c_for_local_maximum_l281_281055

theorem constant_c_for_local_maximum (c : ℝ) (f : ℝ → ℝ) (h1 : ∀ x, f x = x * (x - c) ^ 2) (h2 : ∃ δ > 0, ∀ x, |x - 2| < δ → f x ≤ f 2) : c = 6 :=
sorry

end constant_c_for_local_maximum_l281_281055


namespace g_positive_l281_281345

noncomputable def f (x : ℝ) : ℝ :=
  if x ≠ 0 then 1 / 2 + 1 / (2^x - 1) else 0

noncomputable def g (x : ℝ) : ℝ :=
  x^3 * f x

theorem g_positive (x : ℝ) (hx : x ≠ 0) : g x > 0 :=
  sorry -- Proof to be filled in

end g_positive_l281_281345


namespace tamia_bell_pepper_pieces_l281_281252

def total_pieces (n k p : Nat) : Nat :=
  let slices := n * k
  let half_slices := slices / 2
  let smaller_pieces := half_slices * p
  let total := half_slices + smaller_pieces
  total

theorem tamia_bell_pepper_pieces :
  total_pieces 5 20 3 = 200 :=
by
  sorry

end tamia_bell_pepper_pieces_l281_281252


namespace root_expression_value_l281_281533

theorem root_expression_value (m : ℝ) (h : 2 * m^2 + 3 * m - 1 = 0) : 4 * m^2 + 6 * m - 2019 = -2017 :=
by
  sorry

end root_expression_value_l281_281533


namespace find_x_l281_281783

noncomputable def series_sum (x : ℝ) : ℝ :=
∑' n : ℕ, (1 + 6 * n) * x^n

theorem find_x (x : ℝ) (h : series_sum x = 100) (hx : |x| < 1) : x = 3 / 5 := 
sorry

end find_x_l281_281783


namespace chord_length_of_intersection_l281_281477

def ellipse (x y : ℝ) := x^2 + 4 * y^2 = 16
def line (x y : ℝ) := y = (1/2) * x + 1

theorem chord_length_of_intersection :
  ∃ A B : ℝ × ℝ, ellipse A.fst A.snd ∧ ellipse B.fst B.snd ∧ line A.fst A.snd ∧ line B.fst B.snd ∧
  dist A B = Real.sqrt 35 :=
sorry

end chord_length_of_intersection_l281_281477


namespace max_minus_min_all_three_languages_l281_281910

def student_population := 1500

def english_students (e : ℕ) : Prop := 1050 ≤ e ∧ e ≤ 1125
def spanish_students (s : ℕ) : Prop := 750 ≤ s ∧ s ≤ 900
def german_students (g : ℕ) : Prop := 300 ≤ g ∧ g ≤ 450

theorem max_minus_min_all_three_languages (e s g e_s e_g s_g e_s_g : ℕ) 
    (he : english_students e)
    (hs : spanish_students s)
    (hg : german_students g)
    (pie : e + s + g - e_s - e_g - s_g + e_s_g = student_population) 
    : (M - m = 450) :=
sorry

end max_minus_min_all_three_languages_l281_281910


namespace sin_half_alpha_l281_281178

theorem sin_half_alpha (α : ℝ) (hα : 0 < α ∧ α < π / 2) (hcosα : cos α = (1 + real.sqrt 5) / 4) :
  sin (α / 2) = (real.sqrt 5 - 1) / 4 :=
by
  sorry

end sin_half_alpha_l281_281178


namespace simplify_fraction_l281_281405

theorem simplify_fraction : (3 / 462 + 17 / 42) = 95 / 231 :=
by 
  sorry

end simplify_fraction_l281_281405


namespace area_outside_circle_of_equilateral_triangle_l281_281268

noncomputable def equilateral_triangle_area_outside_circle {a : ℝ} (h : a > 0) : ℝ :=
  let S1 := a^2 * Real.sqrt 3 / 4
  let S2 := Real.pi * (a / 3)^2
  let S3 := (Real.pi * (a / 3)^2 / 6) - (a^2 * Real.sqrt 3 / 36)
  S1 - S2 + 3 * S3

theorem area_outside_circle_of_equilateral_triangle
  (a : ℝ) (h : a > 0) :
  equilateral_triangle_area_outside_circle h = a^2 * (3 * Real.sqrt 3 - Real.pi) / 18 :=
sorry

end area_outside_circle_of_equilateral_triangle_l281_281268


namespace inequality_division_by_positive_l281_281965

theorem inequality_division_by_positive (x y : ℝ) (h : x > y) : (x / 5 > y / 5) :=
by
  sorry

end inequality_division_by_positive_l281_281965


namespace good_number_sum_l281_281463

def is_good (n : ℕ) : Prop := ∃ k : ℕ, n = k * (k + 1)

theorem good_number_sum (a : ℕ) (h1 : a > 6) (h2 : is_good a) :
  ∃ x y : ℕ, is_good x ∧ is_good y ∧ a * (a + 1) = x * (x + 1) + 3 * y * (y + 1) :=
sorry

end good_number_sum_l281_281463


namespace problem_solution_l281_281823

variable {A B : Set ℝ}

def definition_A : Set ℝ := {x | 0 < x ∧ x ≤ 2}
def definition_B : Set ℝ := {x | x ≤ 3}

theorem problem_solution : definition_A ∩ definition_B = definition_A :=
by
  sorry

end problem_solution_l281_281823


namespace matrix_determinant_zero_l281_281142

noncomputable def matrix_example : Matrix (Fin 3) (Fin 3) ℝ := 
  ![
    ![Real.sin 1, Real.sin 2, Real.sin 3],
    ![Real.sin 4, Real.sin 5, Real.sin 6],
    ![Real.sin 7, Real.sin 8, Real.sin 9]
  ]

theorem matrix_determinant_zero : matrix_example.det = 0 := 
by 
  sorry

end matrix_determinant_zero_l281_281142


namespace maximum_shapes_in_grid_l281_281891

-- Define the grid size and shape properties
def grid_width : Nat := 8
def grid_height : Nat := 14
def shape_area : Nat := 3
def shape_grid_points : Nat := 8

-- Define the total grid points in the rectangular grid
def total_grid_points : Nat := (grid_width + 1) * (grid_height + 1)

-- Define the question and the condition that needs to be proved
theorem maximum_shapes_in_grid : (total_grid_points / shape_grid_points) = 16 := by
  sorry

end maximum_shapes_in_grid_l281_281891


namespace factor_polynomial_l281_281658

theorem factor_polynomial : 
  (∀ x : ℝ, (x^2 + 6 * x + 9 - 64 * x^4) = (-8 * x^2 + x + 3) * (8 * x^2 + x + 3)) :=
by
  intro x
  -- Left-hand side
  let lhs := x^2 + 6 * x + 9 - 64 * x^4
  -- Right-hand side after factorization
  let rhs := (-8 * x^2 + x + 3) * (8 * x^2 + x + 3)
  -- Prove the equality
  show lhs = rhs
  sorry

end factor_polynomial_l281_281658


namespace complex_magnitude_of_3_minus_4i_l281_281492

open Complex

theorem complex_magnitude_of_3_minus_4i : Complex.abs ⟨3, -4⟩ = 5 := sorry

end complex_magnitude_of_3_minus_4i_l281_281492


namespace students_remaining_l281_281629

theorem students_remaining (n : ℕ) (h1 : n = 1000)
  (h_beach : n / 2 = 500)
  (h_home : (n - n / 2) / 2 = 250) :
  n - (n / 2 + (n - n / 2) / 2) = 250 :=
by
  sorry

end students_remaining_l281_281629


namespace total_weight_of_nuts_l281_281905

theorem total_weight_of_nuts (weight_almonds weight_pecans : ℝ) (h1 : weight_almonds = 0.14) (h2 : weight_pecans = 0.38) : weight_almonds + weight_pecans = 0.52 :=
by
  sorry

end total_weight_of_nuts_l281_281905


namespace Brittany_age_after_vacation_l281_281319

variable (Rebecca_age Brittany_age vacation_years : ℕ)
variable (h1 : Rebecca_age = 25)
variable (h2 : Brittany_age = Rebecca_age + 3)
variable (h3 : vacation_years = 4)

theorem Brittany_age_after_vacation : 
    Brittany_age + vacation_years = 32 := 
by
  sorry

end Brittany_age_after_vacation_l281_281319


namespace fill_tank_time_l281_281913

-- Definitions based on provided conditions
def pipeA_time := 60 -- Pipe A fills the tank in 60 minutes
def pipeB_time := 40 -- Pipe B fills the tank in 40 minutes

-- Theorem statement
theorem fill_tank_time (T : ℕ) : 
  (T / 2) / pipeB_time + (T / 2) * (1 / pipeA_time + 1 / pipeB_time) = 1 → 
  T = 48 :=
by
  intro h
  sorry

end fill_tank_time_l281_281913


namespace area_ratio_l281_281743

variable (A_shape A_triangle : ℝ)

-- Condition: The area ratio given.
axiom ratio_condition : A_shape / A_triangle = 2

-- Theorem statement
theorem area_ratio (A_shape A_triangle : ℝ) (h : A_shape / A_triangle = 2) : A_shape / A_triangle = 2 :=
by
  exact h

end area_ratio_l281_281743


namespace find_line_equation_l281_281586

theorem find_line_equation :
  ∃ (m : ℝ), ∃ (b : ℝ), (∀ x y : ℝ,
  (x + 3 * y - 2 = 0 → y = -1/3 * x + 2/3) ∧
  (x = 3 → y = 0) →
  y = m * x + b) ∧
  (m = 3 ∧ b = -9) :=
  sorry

end find_line_equation_l281_281586


namespace probability_at_least_one_2_on_8_sided_dice_l281_281112

theorem probability_at_least_one_2_on_8_sided_dice :
  (∃ (d1 d2 : Fin 8), d1 = 1 ∨ d2 = 1) → (15 / 64) = (15 / 64) := by
  intro h
  sorry

end probability_at_least_one_2_on_8_sided_dice_l281_281112


namespace smallest_x_plus_y_l281_281809

theorem smallest_x_plus_y (x y : ℕ) (hx_pos : 0 < x) (hy_pos : 0 < y) (hxy_neq : x ≠ y) (h_eq : 1 / (x:ℝ) + 1 / (y:ℝ) = 1 / 15) : x + y = 64 :=
sorry

end smallest_x_plus_y_l281_281809


namespace least_number_to_add_l281_281742

theorem least_number_to_add (m n : ℕ) (h₁ : m = 1052) (h₂ : n = 23) : 
  ∃ k : ℕ, (m + k) % n = 0 ∧ k = 6 :=
by
  sorry

end least_number_to_add_l281_281742


namespace eccentricity_range_of_ellipse_l281_281209

theorem eccentricity_range_of_ellipse
  (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b)
  (e : ℝ) (he1 : e > 0) (he2 : e < 1)
  (h_directrix : 2 * (a / e) ≤ 3 * (2 * a)) :
  (1 / 3) ≤ e ∧ e < 1 := 
sorry

end eccentricity_range_of_ellipse_l281_281209


namespace average_monthly_increase_is_20_percent_l281_281886

-- Define the given conditions in Lean
def V_Jan : ℝ := 2 
def V_Mar : ℝ := 2.88 

-- Percentage increase each month over the previous month is the same
def consistent_growth_rate (x : ℝ) : Prop := 
  V_Jan * (1 + x)^2 = V_Mar

-- We need to prove that the monthly growth rate x is 0.2 (or 20%)
theorem average_monthly_increase_is_20_percent : 
  ∃ x : ℝ, consistent_growth_rate x ∧ x = 0.2 :=
by
  sorry

end average_monthly_increase_is_20_percent_l281_281886


namespace max_unsuccessful_attempts_l281_281623

theorem max_unsuccessful_attempts (n_rings letters_per_ring : ℕ) (h_rings : n_rings = 3) (h_letters : letters_per_ring = 6) : 
  (letters_per_ring ^ n_rings) - 1 = 215 := 
by 
  -- conditions
  rw [h_rings, h_letters]
  -- necessary imports and proof generation
  sorry

end max_unsuccessful_attempts_l281_281623


namespace find_x_l281_281611

theorem find_x (x : ℝ) (h : 0.45 * x = (1 / 3) * x + 110) : x = 942.857 :=
by
  sorry

end find_x_l281_281611


namespace ellipse_equation_l281_281193

-- Definitions based on the problem conditions
def hyperbola_foci (x y : ℝ) : Prop := 2 * x^2 - 2 * y^2 = 1
def passes_through_point (p : ℝ × ℝ) (x y : ℝ) : Prop := p = (1, -3 / 2)

-- The statement to be proved
theorem ellipse_equation (c : ℝ) (a b : ℝ) :
    hyperbola_foci (-1) 0 ∧ hyperbola_foci 1 0 ∧
    passes_through_point (1, -3 / 2) 1 (-3 / 2) ∧
    (a = 2) ∧ (b = Real.sqrt 3) ∧ (c = 1)
    → ∀ x y : ℝ, x^2 / 4 + y^2 / 3 = 1 :=
by
  sorry

end ellipse_equation_l281_281193


namespace devin_biked_more_l281_281062

def cyra_distance := 77
def cyra_time := 7
def cyra_speed := cyra_distance / cyra_time
def devin_speed := cyra_speed + 3
def marathon_time := 7
def devin_distance := devin_speed * marathon_time
def distance_difference := devin_distance - cyra_distance

theorem devin_biked_more : distance_difference = 21 := 
  by
    sorry

end devin_biked_more_l281_281062


namespace trapezoid_ABCD_BCE_area_l281_281845

noncomputable def triangle_area (a b c : ℝ) (angle_abc : ℝ) : ℝ :=
  1 / 2 * a * b * Real.sin angle_abc

noncomputable def area_of_triangle_BCE (AB DC AD : ℝ) (angle_DAB : ℝ) (area_triangle_DCB : ℝ) : ℝ :=
  let ratio := AB / DC
  (ratio / (1 + ratio)) * area_triangle_DCB

theorem trapezoid_ABCD_BCE_area :
  ∀ (AB DC AD : ℝ) (angle_DAB : ℝ) (area_triangle_DCB : ℝ),
    AB = 30 →
    DC = 24 →
    AD = 3 →
    angle_DAB = Real.pi / 3 →
    area_triangle_DCB = 18 * Real.sqrt 3 →
    area_of_triangle_BCE AB DC AD angle_DAB area_triangle_DCB = 10 * Real.sqrt 3 := 
by
  intros
  sorry

end trapezoid_ABCD_BCE_area_l281_281845


namespace range_of_a_l281_281376

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x^2 - 2*x + a ≥ 0) ↔ (1 ≤ a) :=
by sorry

end range_of_a_l281_281376


namespace smallest_positive_debt_resolvable_l281_281113

/-- The value of a sheep in dollars -/
def sheep_value : ℕ := 250

/-- The value of a lamb in dollars -/
def lamb_value : ℕ := 150

/-- Given a debt D that can be expressed in the form of 250s + 150l for integers s and l,
prove that the smallest positive amount of D is 50 dollars -/
theorem smallest_positive_debt_resolvable : 
  ∃ (s l : ℤ), sheep_value * s + lamb_value * l = 50 :=
sorry

end smallest_positive_debt_resolvable_l281_281113


namespace max_is_twice_emily_probability_l281_281780

noncomputable def probability_event_max_gt_twice_emily : ℝ :=
  let total_area := 1000 * 3000
  let triangle_area := 1/2 * 1000 * 1000
  let rectangle_area := 1000 * (3000 - 2000)
  let favorable_area := triangle_area + rectangle_area
  favorable_area / total_area

theorem max_is_twice_emily_probability :
  probability_event_max_gt_twice_emily = 1 / 2 :=
by
  sorry

end max_is_twice_emily_probability_l281_281780


namespace shortest_side_length_l281_281954

theorem shortest_side_length (perimeter : ℝ) (shortest : ℝ) (side1 side2 side3 : ℝ) 
  (h1 : side1 + side2 + side3 = perimeter)
  (h2 : side1 = 2 * shortest)
  (h3 : side2 = 2 * shortest) :
  shortest = 3 := by
  sorry

end shortest_side_length_l281_281954


namespace adult_dog_cost_is_100_l281_281848

-- Define the costs for cats, puppies, and dogs.
def cat_cost : ℕ := 50
def puppy_cost : ℕ := 150

-- Define the number of each type of animal.
def number_of_cats : ℕ := 2
def number_of_adult_dogs : ℕ := 3
def number_of_puppies : ℕ := 2

-- The total cost
def total_cost : ℕ := 700

-- Define what needs to be proven: the cost of getting each adult dog ready for adoption.
theorem adult_dog_cost_is_100 (D : ℕ) (h : number_of_cats * cat_cost + number_of_adult_dogs * D + number_of_puppies * puppy_cost = total_cost) : D = 100 :=
by 
  sorry

end adult_dog_cost_is_100_l281_281848


namespace find_integer_x_l281_281478

theorem find_integer_x (x : ℤ) :
  1 < x ∧ x < 9 ∧
  2 < x ∧ x < 15 ∧
  -1 < x ∧ x < 7 ∧
  0 < x ∧ x < 4 ∧
  x + 1 < 5 → 
  x = 3 :=
by
  sorry

end find_integer_x_l281_281478


namespace smallest_product_not_factor_of_48_l281_281439

theorem smallest_product_not_factor_of_48 (a b : ℕ) (h1 : a ≠ b) (h2 : a ∣ 48) (h3 : b ∣ 48) (h4 : ¬ (a * b ∣ 48)) : a * b = 32 :=
sorry

end smallest_product_not_factor_of_48_l281_281439


namespace suffering_correctness_l281_281638

noncomputable def expected_total_suffering (n m : ℕ) : ℕ :=
  if n = 8 ∧ m = 256 then (2^135 - 2^128 + 1) / (2^119 * 129) else 0

theorem suffering_correctness :
  expected_total_suffering 8 256 = (2^135 - 2^128 + 1) / (2^119 * 129) :=
sorry

end suffering_correctness_l281_281638


namespace ending_number_condition_l281_281590

theorem ending_number_condition (h : ∃ k : ℕ, k < 21 ∧ 100 < 19 * k) : ∃ n, 21.05263157894737 * 19 = n → n = 399 :=
by
  sorry  -- this is where the proof would go

end ending_number_condition_l281_281590


namespace mn_eq_neg_infty_to_0_l281_281704

-- Definitions based on the conditions
def M : Set ℝ := {y | y ≤ 2}
def N : Set ℝ := {y | 0 ≤ y ∧ y ≤ 3}

-- Set difference definition
def set_diff (A B : Set ℝ) : Set ℝ := {y | y ∈ A ∧ y ∉ B}

-- The proof statement we need to prove
theorem mn_eq_neg_infty_to_0 : set_diff M N = {y | y < 0} :=
  sorry  -- Proof will go here

end mn_eq_neg_infty_to_0_l281_281704


namespace area_change_l281_281607

variable (L B : ℝ)

def initial_area (L B : ℝ) : ℝ := L * B

def new_length (L : ℝ) : ℝ := 1.20 * L

def new_breadth (B : ℝ) : ℝ := 0.95 * B

def new_area (L B : ℝ) : ℝ := (new_length L) * (new_breadth B)

theorem area_change (L B : ℝ) : new_area L B = 1.14 * (initial_area L B) := by
  -- Proof goes here
  sorry

end area_change_l281_281607


namespace digit_relationship_l281_281382

theorem digit_relationship (d1 d2 : ℕ) (h1 : d1 * 10 + d2 = 16) (h2 : d1 + d2 = 7) : d2 = 6 * d1 :=
by
  sorry

end digit_relationship_l281_281382


namespace bananas_in_each_box_l281_281394

-- You might need to consider noncomputable if necessary here for Lean's real number support.
noncomputable def bananas_per_box (total_bananas : ℕ) (total_boxes : ℕ) : ℕ :=
  total_bananas / total_boxes

theorem bananas_in_each_box :
  bananas_per_box 40 8 = 5 := by
  sorry

end bananas_in_each_box_l281_281394


namespace smallest_pos_integer_for_frac_reducible_l281_281660

theorem smallest_pos_integer_for_frac_reducible :
  ∃ n : ℕ, n > 0 ∧ ∃ d > 1, d ∣ (n - 17) ∧ d ∣ (6 * n + 8) ∧ n = 127 :=
by
  sorry

end smallest_pos_integer_for_frac_reducible_l281_281660


namespace third_competitor_hot_dogs_l281_281775

theorem third_competitor_hot_dogs (first_ate : ℕ) (second_multiplier : ℕ) (third_percent_less : ℕ) (second_ate third_ate : ℕ) 
  (H1 : first_ate = 12)
  (H2 : second_multiplier = 2)
  (H3 : third_percent_less = 25)
  (H4 : second_ate = first_ate * second_multiplier)
  (H5 : third_ate = second_ate - (third_percent_less * second_ate / 100)) : 
  third_ate = 18 := 
by 
  sorry

end third_competitor_hot_dogs_l281_281775


namespace ashok_total_subjects_l281_281482

variable (n : ℕ) (T : ℕ)

theorem ashok_total_subjects (h_ave_all : 75 * n = T + 80)
                       (h_ave_first : T = 74 * (n - 1)) :
  n = 6 := sorry

end ashok_total_subjects_l281_281482


namespace number_of_integer_segments_l281_281400

theorem number_of_integer_segments (DE EF : ℝ) (H1 : DE = 24) (H2 : EF = 25) : 
  ∃ n : ℕ, n = 2 :=
by
  sorry

end number_of_integer_segments_l281_281400


namespace total_number_of_students_l281_281277

/-- The total number of high school students in the school given sampling constraints. -/
theorem total_number_of_students (F1 F2 F3 : ℕ) (sample_size : ℕ) (consistency_ratio : ℕ) :
  F2 = 300 ∧ sample_size = 45 ∧ (F1 / F3) = 2 ∧ 
  (20 + 10 + (sample_size - 30)) = sample_size → F1 + F2 + F3 = 900 :=
by
  sorry

end total_number_of_students_l281_281277


namespace geometric_seq_a4_l281_281547

variable {a : ℕ → ℝ}

theorem geometric_seq_a4 (h : ∀ n, a (n + 2) / a n = a 2 / a 0)
  (root_condition1 : a 2 * a 6 = 64)
  (root_condition2 : a 2 + a 6 = 34) :
  a 4 = 8 :=
by
  sorry

end geometric_seq_a4_l281_281547


namespace sum_first_23_natural_numbers_l281_281283

theorem sum_first_23_natural_numbers :
  (23 * (23 + 1)) / 2 = 276 := 
by
  sorry

end sum_first_23_natural_numbers_l281_281283


namespace problem_1_problem_2_l281_281724

theorem problem_1 (x y : ℝ) (h1 : x - y = 3) (h2 : 3*x - 8*y = 14) : x = 2 ∧ y = -1 :=
sorry

theorem problem_2 (x y : ℝ) (h1 : 3*x + 4*y = 16) (h2 : 5*x - 6*y = 33) : x = 6 ∧ y = -1/2 :=
sorry

end problem_1_problem_2_l281_281724


namespace correct_factorization_l281_281288

theorem correct_factorization :
  (∀ (x y : ℝ), x^2 + y^2 ≠ (x + y)^2) ∧
  (∀ (x y : ℝ), x^2 + 2*x*y + y^2 ≠ (x - y)^2) ∧
  (∀ (x : ℝ), x^2 + x ≠ x * (x - 1)) ∧
  (∀ (x y : ℝ), x^2 - y^2 = (x + y) * (x - y)) :=
by 
  sorry

end correct_factorization_l281_281288


namespace n_gon_partition_l281_281031

-- Define a function to determine if an n-gon can be partitioned as required
noncomputable def canBePartitioned (n : ℕ) (h : n ≥ 3) : Prop :=
  n ≠ 4 ∧ n ≥ 3

theorem n_gon_partition (n : ℕ) (h : n ≥ 3) : canBePartitioned n h ↔ (n = 3 ∨ n ≥ 5) :=
by sorry

end n_gon_partition_l281_281031


namespace a7_arithmetic_sequence_l281_281978

variable (a : ℕ → ℝ)

def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

-- Given conditions
def a1 : ℝ := 2
def a4 : ℝ := 5

theorem a7_arithmetic_sequence : ∃ d : ℝ, is_arithmetic_sequence a d ∧ a 1 = a1 ∧ a 4 = a4 → a 7 = 8 :=
by
  sorry

end a7_arithmetic_sequence_l281_281978


namespace tangent_line_to_curve_at_Mpi_l281_281087

noncomputable def tangent_line_eq_at_point (x : ℝ) (y : ℝ) : Prop :=
  y = (Real.sin x) / x

theorem tangent_line_to_curve_at_Mpi :
  (∀ x y, tangent_line_eq_at_point x y →
    (∃ (m : ℝ), m = -1 / π) →
    (∀ x1 y1 (hx : x1 = π) (hy : y1 = 0), x + π * y - π = 0)) :=
by
  sorry

end tangent_line_to_curve_at_Mpi_l281_281087


namespace min_flowers_for_bouquets_l281_281216

open Classical

noncomputable def minimum_flowers (types : ℕ) (flowers_for_bouquet : ℕ) (bouquets : ℕ) : ℕ := 
  sorry

theorem min_flowers_for_bouquets : minimum_flowers 6 5 10 = 70 := 
  sorry

end min_flowers_for_bouquets_l281_281216


namespace wrapping_paper_per_present_l281_281402

theorem wrapping_paper_per_present :
  let sum_paper := 1 / 2
  let num_presents := 5
  (sum_paper / num_presents) = 1 / 10 := by
  sorry

end wrapping_paper_per_present_l281_281402


namespace calculate_minutes_worked_today_l281_281921

-- Define the conditions
def production_rate := 6 -- shirts per minute
def total_shirts_today := 72 

-- The statement to prove
theorem calculate_minutes_worked_today :
  total_shirts_today / production_rate = 12 := 
by
  sorry

end calculate_minutes_worked_today_l281_281921


namespace equal_students_initially_l281_281615

theorem equal_students_initially (B G : ℕ) (h1 : B = G) (h2 : B = 2 * (G - 8)) : B + G = 32 :=
by
  sorry

end equal_students_initially_l281_281615


namespace probability_at_least_5_consecutive_heads_fair_8_flips_l281_281618

theorem probability_at_least_5_consecutive_heads_fair_8_flips :
  (number_of_outcomes_with_at_least_5_consecutive_heads_in_n_flips 8 (λ _, true)) / (2^8) = 39 / 256 := sorry

def number_of_outcomes_with_at_least_5_consecutive_heads_in_n_flips (n : ℕ) (coin : ℕ → Prop) : ℕ := 
  -- This should be the function that calculates the number of favorable outcomes
  -- replacing "coin" with conditions for heads and tails but for simplicity,
  -- we are stating it as an undefined function here.
  sorry

#eval probability_at_least_5_consecutive_heads_fair_8_flips

end probability_at_least_5_consecutive_heads_fair_8_flips_l281_281618


namespace find_d_l281_281366

-- Define the polynomial g(x)
def g (d : ℚ) (x : ℚ) : ℚ := d * x^4 + 11 * x^3 + 5 * d * x^2 - 28 * x + 72

-- The main proof statement
theorem find_d (hd : g d 4 = 0) : d = -83 / 42 := by
  sorry -- proof not needed as per prompt

end find_d_l281_281366


namespace smallest_sum_of_inverses_l281_281816

theorem smallest_sum_of_inverses 
  (x y : ℕ) (hx : x ≠ y) (h1 : 0 < x) (h2 : 0 < y) (h_condition : (1 / x : ℚ) + 1 / y = 1 / 15) :
  x + y = 64 := 
sorry

end smallest_sum_of_inverses_l281_281816


namespace find_common_ratio_and_difference_l281_281919

theorem find_common_ratio_and_difference (q d : ℤ) 
  (h1 : q^3 = 1 + 7 * d) 
  (h2 : 1 + q + q^2 + q^3 = 1 + 7 * d + 21) : 
  (q = 4 ∧ d = 9) ∨ (q = -5 ∧ d = -18) :=
by
  sorry

end find_common_ratio_and_difference_l281_281919


namespace matrix_determinant_sin_zero_l281_281143

theorem matrix_determinant_sin_zero : 
  let A : Matrix (Fin 3) (Fin 3) ℝ := ![
    [Real.sin 1, Real.sin 2, Real.sin 3],
    [Real.sin 4, Real.sin 5, Real.sin 6],
    [Real.sin 7, Real.sin 8, Real.sin 9]
  ] in Matrix.det A = 0 :=
by
  let A : Matrix (Fin 3) (Fin 3) ℝ := ![
    [Real.sin 1, Real.sin 2, Real.sin 3],
    [Real.sin 4, Real.sin 5, Real.sin 6],
    [Real.sin 7, Real.sin 8, Real.sin 9]
  ]
  sorry

end matrix_determinant_sin_zero_l281_281143


namespace brittany_age_after_vacation_l281_281321

-- Definitions of the conditions
def rebecca_age : ℕ := 25
def age_difference : ℕ := 3
def vacation_years : ℕ := 4

-- Prove the main statement
theorem brittany_age_after_vacation : rebecca_age + age_difference + vacation_years = 32 := by
  sorry

end brittany_age_after_vacation_l281_281321


namespace sin_angle_GAC_correct_l281_281751

noncomputable def sin_angle_GAC (AB AD AE : ℝ) := 
  let AC := Real.sqrt (AB^2 + AD^2)
  let AG := Real.sqrt (AB^2 + AD^2 + AE^2)
  (AC / AG)

theorem sin_angle_GAC_correct : sin_angle_GAC 2 3 4 = Real.sqrt 377 / 29 := by
  sorry

end sin_angle_GAC_correct_l281_281751


namespace largest_alpha_l281_281509

theorem largest_alpha (a b : ℕ) (h1 : a < b) (h2 : b < 2 * a) (N : ℕ) :
  ∃ (α : ℝ), α = 1 / (2 * a^2 - 2 * a * b + b^2) ∧
  (∃ marked_cells : ℕ, marked_cells ≥ α * (N:ℝ)^2) :=
by
  sorry

end largest_alpha_l281_281509


namespace sin_neg_045_unique_solution_l281_281834

theorem sin_neg_045_unique_solution (x : ℝ) (hx : 0 ≤ x ∧ x < 180) (h: ℝ) :
  (h = Real.sin x → h = -0.45) → 
  ∃! x, 0 ≤ x ∧ x < 180 ∧ Real.sin x = -0.45 :=
by sorry

end sin_neg_045_unique_solution_l281_281834


namespace total_cost_l281_281759

theorem total_cost (a b : ℕ) : 30 * a + 20 * b = 30 * a + 20 * b :=
by
  sorry

end total_cost_l281_281759


namespace grid_path_theorem_l281_281926

open Nat

variables (m n : ℕ)
variables (A B C : ℕ)

def conditions (m n : ℕ) : Prop := m ≥ 4 ∧ n ≥ 4

noncomputable def grid_path_problem (m n A B C : ℕ) : Prop :=
  conditions m n ∧
  ((m - 1) * (n - 1) = A + (B + C)) ∧
  A = B - C + m + n - 1

theorem grid_path_theorem (m n A B C : ℕ) (h : grid_path_problem m n A B C) : 
  A = B - C + m + n - 1 :=
  sorry

end grid_path_theorem_l281_281926


namespace sin_half_alpha_l281_281179

theorem sin_half_alpha (α : ℝ) (hα : 0 < α ∧ α < π / 2) (hcosα : cos α = (1 + real.sqrt 5) / 4) :
  sin (α / 2) = (real.sqrt 5 - 1) / 4 :=
by
  sorry

end sin_half_alpha_l281_281179


namespace digits_property_l281_281888

theorem digits_property (n : ℕ) (h : 100 ≤ n ∧ n < 1000) :
  (∃ (f : ℕ → Prop), ∀ d ∈ [n / 100, (n / 10) % 10, n % 10], f d ∧ (¬ d = 0 ∧ ¬ Nat.Prime d)) ↔ 
  (∀ d ∈ [n / 100, (n / 10) % 10, n % 10], d ∈ [1, 4, 6, 8, 9]) :=
sorry

end digits_property_l281_281888


namespace smallest_integer_mod_conditions_l281_281000

theorem smallest_integer_mod_conditions : 
  ∃ x : ℕ, 
  (x % 4 = 3) ∧ (x % 3 = 2) ∧ (∀ y : ℕ, (y % 4 = 3) ∧ (y % 3 = 2) → x ≤ y) ∧ x = 11 :=
by
  sorry

end smallest_integer_mod_conditions_l281_281000


namespace sequence_term_2010_l281_281697

theorem sequence_term_2010 :
  ∀ (a : ℕ → ℤ), a 1 = 1 → a 2 = 2 → 
    (∀ n : ℕ, n ≥ 3 → a n = a (n - 1) - a (n - 2)) → 
    a 2010 = -1 :=
by
  sorry

end sequence_term_2010_l281_281697


namespace proof_correct_chemical_information_l281_281260

def chemical_formula_starch : String := "(C_{6}H_{10}O_{5})_{n}"
def structural_formula_glycine : String := "H_{2}N-CH_{2}-COOH"
def element_in_glass_ceramics_cement : String := "Si"
def elements_cause_red_tide : List String := ["N", "P"]

theorem proof_correct_chemical_information :
  chemical_formula_starch = "(C_{6}H_{10}O_{5})_{n}" ∧
  structural_formula_glycine = "H_{2}N-CH_{2}-COOH" ∧
  element_in_glass_ceramics_cement = "Si" ∧
  elements_cause_red_tide = ["N", "P"] :=
by
  sorry

end proof_correct_chemical_information_l281_281260


namespace present_age_of_son_l281_281460

variable (S M : ℝ)

-- Conditions
def condition1 : Prop := M = S + 35
def condition2 : Prop := M + 5 = 3 * (S + 5)

-- Proof Problem
theorem present_age_of_son
  (h1 : condition1 S M)
  (h2 : condition2 S M) :
  S = 12.5 :=
sorry

end present_age_of_son_l281_281460


namespace sin_half_angle_l281_281181

theorem sin_half_angle (α : ℝ) (h₁ : 0 < α ∧ α < π / 2) (h₂ : real.cos α = (1 + real.sqrt 5) / 4) :
  real.sin (α / 2) = (real.sqrt 5 - 1) / 4 :=
by
  sorry

end sin_half_angle_l281_281181


namespace required_speed_is_85_l281_281459

-- Definitions based on conditions
def speed1 := 60
def time1 := 3
def total_time := 5
def average_speed := 70

-- Derived conditions
def distance1 := speed1 * time1
def total_distance := average_speed * total_time
def remaining_distance := total_distance - distance1
def remaining_time := total_time - time1
def required_speed := remaining_distance / remaining_time

-- Theorem statement
theorem required_speed_is_85 : required_speed = 85 := by
    sorry

end required_speed_is_85_l281_281459


namespace find_k_l281_281213

open Real

noncomputable def chord_intersection (k : ℝ) : Prop :=
  let R : ℝ := 3
  let d := abs (k + 1) / sqrt (1 + k^2)
  d^2 + (12 * sqrt 5 / 10)^2 = R^2

theorem find_k (k : ℝ) (h : k > 1) (h_intersect : chord_intersection k) : k = 2 := by
  sorry

end find_k_l281_281213


namespace cosine_seventh_power_expansion_l281_281644

theorem cosine_seventh_power_expansion :
  let b1 := (35 : ℝ) / 64
  let b2 := (0 : ℝ)
  let b3 := (21 : ℝ) / 64
  let b4 := (0 : ℝ)
  let b5 := (7 : ℝ) / 64
  let b6 := (0 : ℝ)
  let b7 := (1 : ℝ) / 64
  b1^2 + b2^2 + b3^2 + b4^2 + b5^2 + b6^2 + b7^2 = 1687 / 4096 := by
  sorry

end cosine_seventh_power_expansion_l281_281644


namespace max_k_consecutive_sum_2_times_3_pow_8_l281_281370

theorem max_k_consecutive_sum_2_times_3_pow_8 :
  ∃ k : ℕ, 0 < k ∧ 
           (∃ n : ℕ, 2 * 3^8 = (k * (2 * n + k + 1)) / 2) ∧
           (∀ k' : ℕ, (∃ n' : ℕ, 0 < k' ∧ 2 * 3^8 = (k' * (2 * n' + k' + 1)) / 2) → k' ≤ 81) :=
sorry

end max_k_consecutive_sum_2_times_3_pow_8_l281_281370


namespace A_inter_B_eq_A_l281_281819

def A := {x : ℝ | 0 < x ∧ x ≤ 2}
def B := {x : ℝ | x ≤ 3}

theorem A_inter_B_eq_A : A ∩ B = A := 
by 
  sorry 

end A_inter_B_eq_A_l281_281819


namespace smallest_x_plus_y_l281_281812

theorem smallest_x_plus_y (x y : ℕ) (hx_pos : 0 < x) (hy_pos : 0 < y) (hxy_neq : x ≠ y) (h_eq : 1 / (x:ℝ) + 1 / (y:ℝ) = 1 / 15) : x + y = 64 :=
sorry

end smallest_x_plus_y_l281_281812


namespace find_angle_D_l281_281381

theorem find_angle_D (A B C D : ℝ)
  (h1 : A + B = 180)
  (h2 : C = 2 * D)
  (h3 : A + B + C + D = 360) : D = 60 :=
sorry

end find_angle_D_l281_281381


namespace linear_function_no_fourth_quadrant_l281_281502

theorem linear_function_no_fourth_quadrant (k : ℝ) (hk : k > 2) : 
  ∀ x (hx : x > 0), (k-2) * x + k ≥ 0 :=
by
  sorry

end linear_function_no_fourth_quadrant_l281_281502


namespace max_k_consecutive_sum_l281_281368

theorem max_k_consecutive_sum :
  ∃ n : ℕ, n > 0 ∧ ∃ k : ℕ, k * (2 * n + k - 1) = 2^2 * 3^8 ∧ ∀ k' > k, ¬ ∃ n', n' > 0 ∧ k' * (2 * n' + k' - 1) = 2^2 * 3^8 := sorry

end max_k_consecutive_sum_l281_281368


namespace buses_required_is_12_l281_281762

-- Define the conditions given in the problem
def students : ℕ := 535
def bus_capacity : ℕ := 45

-- Define the minimum number of buses required
def buses_needed (students : ℕ) (bus_capacity : ℕ) : ℕ :=
  (students + bus_capacity - 1) / bus_capacity

-- The theorem stating the number of buses required is 12
theorem buses_required_is_12 :
  buses_needed students bus_capacity = 12 :=
sorry

end buses_required_is_12_l281_281762


namespace guests_did_not_come_l281_281767

theorem guests_did_not_come 
  (total_cookies : ℕ) 
  (prepared_guests : ℕ) 
  (cookies_per_guest : ℕ) 
  (total_cookies_eq : total_cookies = 18) 
  (prepared_guests_eq : prepared_guests = 10)
  (cookies_per_guest_eq : cookies_per_guest = 18) 
  (total_cookies_computation : total_cookies = cookies_per_guest) :
  prepared_guests - total_cookies / cookies_per_guest = 9 :=
by
  sorry

end guests_did_not_come_l281_281767


namespace probability_divisibility_9_correct_l281_281987

-- Define the set S
def S : Set ℕ := { n | ∃ a b: ℕ, 0 ≤ a ∧ a < 40 ∧ 0 ≤ b ∧ b < 40 ∧ a ≠ b ∧ n = 2^a + 2^b }

-- Define the criteria for divisibility by 9
def divisible_by_9 (n : ℕ) : Prop := 9 ∣ n

-- Define the total size of set S
def size_S : ℕ := 780  -- as calculated from combination

-- Count valid pairs (a, b) such that 2^a + 2^b is divisible by 9
def valid_pairs : ℕ := 133  -- as calculated from summation

-- Define the probability
def probability_divisible_by_9 : ℕ := valid_pairs / size_S

-- The proof statement
theorem probability_divisibility_9_correct:
  (valid_pairs : ℚ) / (size_S : ℚ) = 133 / 780 := sorry

end probability_divisibility_9_correct_l281_281987


namespace rattlesnakes_count_l281_281095

theorem rattlesnakes_count (total_snakes : ℕ) (boa_constrictors pythons rattlesnakes : ℕ)
  (h1 : total_snakes = 200)
  (h2 : boa_constrictors = 40)
  (h3 : pythons = 3 * boa_constrictors)
  (h4 : total_snakes = boa_constrictors + pythons + rattlesnakes) :
  rattlesnakes = 40 :=
by
  sorry

end rattlesnakes_count_l281_281095


namespace shoe_price_l281_281999

theorem shoe_price :
  ∀ (P : ℝ),
    (6 * P + 18 * 2 = 27 * 2) → P = 3 :=
by
  intro P H
  sorry

end shoe_price_l281_281999


namespace sin_half_angle_l281_281180

theorem sin_half_angle (α : ℝ) (h₁ : 0 < α ∧ α < π / 2) (h₂ : real.cos α = (1 + real.sqrt 5) / 4) :
  real.sin (α / 2) = (real.sqrt 5 - 1) / 4 :=
by
  sorry

end sin_half_angle_l281_281180


namespace buns_per_student_correct_l281_281353

variables (packages_per_bun : Nat) (num_packages : Nat)
           (num_classes : Nat) (students_per_class : Nat)

def total_buns (packages_per_bun : Nat) (num_packages : Nat) : Nat :=
  packages_per_bun * num_packages

def total_students (num_classes : Nat) (students_per_class : Nat) : Nat :=
  num_classes * students_per_class

def buns_per_student (total_buns : Nat) (total_students : Nat) : Nat :=
  total_buns / total_students

theorem buns_per_student_correct :
  packages_per_bun = 8 →
  num_packages = 30 →
  num_classes = 4 →
  students_per_class = 30 →
  buns_per_student (total_buns packages_per_bun num_packages) 
                  (total_students num_classes students_per_class) = 2 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end buns_per_student_correct_l281_281353


namespace smallest_b_in_arithmetic_series_l281_281705

theorem smallest_b_in_arithmetic_series (a b c d : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_arith_series : a = b - d ∧ c = b + d) (h_product : a * b * c = 125) : b ≥ 5 :=
sorry

end smallest_b_in_arithmetic_series_l281_281705


namespace arithmetic_mean_three_fractions_l281_281329

theorem arithmetic_mean_three_fractions :
  let a := (5 : ℚ) / 8
  let b := (7 : ℚ) / 8
  let c := (3 : ℚ) / 4
  (a + b) / 2 = c :=
by
  sorry

end arithmetic_mean_three_fractions_l281_281329


namespace outlinedSquareDigit_l281_281274

-- We define the conditions for three-digit powers of 2 and 3
def isThreeDigitPowerOf (base : ℕ) (n : ℕ) : Prop :=
  let power := base ^ n
  power >= 100 ∧ power < 1000

-- Define the sets of three-digit powers of 2 and 3
def threeDigitPowersOf2 : List ℕ := [128, 256, 512]
def threeDigitPowersOf3 : List ℕ := [243, 729]

-- Define the condition that the digit in the outlined square should be common as a last digit in any power of 2 and 3 that's three-digit long
def commonLastDigitOfPowers (a b : List ℕ) : Option ℕ :=
  let aLastDigits := a.map (λ x => x % 10)
  let bLastDigits := b.map (λ x => x % 10)
  (aLastDigits.inter bLastDigits).head?

theorem outlinedSquareDigit : (commonLastDigitOfPowers threeDigitPowersOf2 threeDigitPowersOf3) = some 3 :=
by
  sorry

end outlinedSquareDigit_l281_281274


namespace larry_wins_probability_eq_l281_281554

-- Define the conditions
def larry_probability_knocks_off : ℚ := 1 / 3
def julius_probability_knocks_off : ℚ := 1 / 4
def larry_throws_first : Prop := True
def independent_events : Prop := True

-- Define the proof that Larry wins the game with probability 2/3
theorem larry_wins_probability_eq :
  larry_throws_first ∧ independent_events →
  larry_probability_knocks_off = 1/3 ∧ julius_probability_knocks_off = 1/4 →
  ∃ p : ℚ, p = 2 / 3 :=
by
  sorry

end larry_wins_probability_eq_l281_281554


namespace circle_area_in_square_centimeters_l281_281116

theorem circle_area_in_square_centimeters (d_meters : ℤ) (h : d_meters = 8) :
  ∃ (A : ℤ), A = 160000 * Real.pi ∧ 
  A = π * (d_meters / 2) ^ 2 * 10000 :=
by
  sorry

end circle_area_in_square_centimeters_l281_281116


namespace ring_rotation_count_l281_281609

-- Define the constants and parameters from the conditions
variables (R ω μ g : ℝ) -- radius, angular velocity, coefficient of friction, and gravity constant
-- Additional constraints on these variables
variable (m : ℝ) -- mass of the ring

theorem ring_rotation_count :
  ∃ n : ℝ, n = (ω^2 * R * (1 + μ^2)) / (4 * π * g * μ * (1 + μ)) :=
sorry

end ring_rotation_count_l281_281609


namespace sequence_periodic_l281_281521

noncomputable def sequence (n : ℕ) : ℝ :=
  Nat.recOn n 2 (λ n a_n, 1 - 1 / a_n)

theorem sequence_periodic : (sequence 2018) = 1 / 2 := by
  sorry

end sequence_periodic_l281_281521


namespace doughnuts_in_each_box_l281_281124

theorem doughnuts_in_each_box (total_doughnuts : ℕ) (boxes : ℕ) (h1 : total_doughnuts = 48) (h2 : boxes = 4) : total_doughnuts / boxes = 12 :=
by
  sorry

end doughnuts_in_each_box_l281_281124


namespace analytical_expression_smallest_positive_period_min_value_max_value_l281_281160

noncomputable def P (x : ℝ) : ℝ × ℝ :=
  (Real.cos (2 * x) + 1, 1)

noncomputable def Q (x : ℝ) : ℝ × ℝ :=
  (1, Real.sqrt 3 * Real.sin (2 * x) + 1)

noncomputable def f (x : ℝ) : ℝ :=
  (P x).1 * (Q x).1 + (P x).2 * (Q x).2

theorem analytical_expression (x : ℝ) : 
  f x = 2 * Real.sin (2 * x + Real.pi / 6) + 2 :=
sorry

theorem smallest_positive_period : 
  ∀ x : ℝ, f (x + Real.pi) = f x :=
sorry

theorem min_value : 
  ∃ x : ℝ, f x = 0 :=
sorry

theorem max_value : 
  ∃ y : ℝ, f y = 4 :=
sorry

end analytical_expression_smallest_positive_period_min_value_max_value_l281_281160


namespace Hari_contribution_l281_281399

theorem Hari_contribution (P T_P T_H : ℕ) (r1 r2 : ℕ) (H : ℕ) :
  P = 3500 → 
  T_P = 12 → 
  T_H = 7 → 
  r1 = 2 → 
  r2 = 3 →
  (P * T_P) * r2 = (H * T_H) * r1 →
  H = 9000 :=
by
  sorry

end Hari_contribution_l281_281399


namespace smallest_x_l281_281741

open Int

def f (x : ℤ) : ℤ :=
  abs (8 * x * x - 50 * x + 21)

theorem smallest_x (x : ℤ) (h1 : Prime (f x)) : x = 1 :=
sorry

end smallest_x_l281_281741


namespace trevor_eggs_left_l281_281849

def gertrude_eggs : Nat := 4
def blanche_eggs : Nat := 3
def nancy_eggs : Nat := 2
def martha_eggs : Nat := 2
def dropped_eggs : Nat := 2

theorem trevor_eggs_left : 
  (gertrude_eggs + blanche_eggs + nancy_eggs + martha_eggs - dropped_eggs) = 9 := 
  by sorry

end trevor_eggs_left_l281_281849


namespace triangle_sides_consecutive_and_angle_relationship_l281_281668

theorem triangle_sides_consecutive_and_angle_relationship (a b c : ℕ) 
  (h1 : a < b) (h2 : b < c) (h3 : b = a + 1) (h4 : c = b + 1) 
  (angle_A angle_B angle_C : ℝ) 
  (h_angle_sum : angle_A + angle_B + angle_C = π) 
  (h_angle_relation : angle_B = 2 * angle_A) : 
  (a, b, c) = (4, 5, 6) :=
sorry

end triangle_sides_consecutive_and_angle_relationship_l281_281668


namespace sin_half_angle_l281_281162

theorem sin_half_angle (α : ℝ) (h0 : 0 < α ∧ α < π / 2) (h1 : cos α = (1 + real.sqrt 5) / 4) :
  sin (α / 2) = (-1 + real.sqrt 5) / 4 :=
by sorry

end sin_half_angle_l281_281162


namespace solution_set_for_absolute_value_inequality_l281_281600

theorem solution_set_for_absolute_value_inequality :
  {x : ℝ | |2 * x - 1| ≤ 3} = {x : ℝ | -1 ≤ x ∧ x ≤ 2} :=
by 
  sorry

end solution_set_for_absolute_value_inequality_l281_281600


namespace max_M_correct_l281_281719

variable (A : ℝ) (x y : ℝ)

axiom A_pos : A > 0

noncomputable def max_M : ℝ :=
if A ≤ 4 then 2 + A / 2 else 2 * Real.sqrt A

theorem max_M_correct : 
  (∀ x y : ℝ, 0 < x → 0 < y → 1/x + 1/y + A/(x + y) ≥ max_M A / Real.sqrt (x * y)) ∧ 
  (A ≤ 4 → max_M A = 2 + A / 2) ∧ 
  (A > 4 → max_M A = 2 * Real.sqrt A) :=
sorry

end max_M_correct_l281_281719


namespace middle_number_of_five_consecutive_numbers_l281_281271

theorem middle_number_of_five_consecutive_numbers (n : ℕ) 
  (h : (n - 2) + (n - 1) + n + (n + 1) + (n + 2) = 60) : n = 12 :=
by
  sorry

end middle_number_of_five_consecutive_numbers_l281_281271


namespace solve_equation_l281_281409

theorem solve_equation (x : ℝ) : x * (2 * x - 1) = 4 * x - 2 ↔ x = 2 ∨ x = 1 / 2 := 
by {
  sorry -- placeholder for the proof
}

end solve_equation_l281_281409


namespace word_count_proof_l281_281524

def letters : List Char := ['A', 'B', 'C', 'D', 'E', 'F']
def consonants : List Char := ['B', 'C', 'D', 'F']
def vowels : List Char := ['A', 'E']

def count_unrestricted_words : ℕ := 6 ^ 5
def count_all_vowel_words : ℕ := 2 ^ 5
def count_one_consonant_words : ℕ := 5 * 4 * (2 ^ 4)
def count_fewer_than_two_consonant_words : ℕ := count_all_vowel_words + count_one_consonant_words

def count_words_with_at_least_two_consonants : ℕ :=
  count_unrestricted_words - count_fewer_than_two_consonant_words

theorem word_count_proof :
  count_words_with_at_least_two_consonants = 7424 := 
by
  -- Proof will be provided here. For now we skip it.
  sorry

end word_count_proof_l281_281524


namespace burger_meal_cost_l281_281869

-- Define the conditions
variables (B S : ℝ)
axiom cost_of_soda : S = (1 / 3) * B
axiom total_cost : B + S + 2 * (B + S) = 24

-- Prove that the cost of the burger meal is $6
theorem burger_meal_cost : B = 6 :=
by {
  -- We'll use both the axioms provided to show B equals 6
  sorry
}

end burger_meal_cost_l281_281869


namespace quadrilateral_area_is_48_l281_281695

structure Quadrilateral :=
  (PQ QR RS SP : ℝ)
  (angle_QRS angle_SPQ : ℝ)

def quadrilateral_example : Quadrilateral :=
{ PQ := 11, QR := 7, RS := 9, SP := 3, angle_QRS := 90, angle_SPQ := 90 }

noncomputable def area_of_quadrilateral (Q : Quadrilateral) : ℝ :=
  (1/2 * Q.PQ * Q.SP) + (1/2 * Q.QR * Q.RS)

theorem quadrilateral_area_is_48 (Q : Quadrilateral) (h1 : Q.PQ = 11) (h2 : Q.QR = 7) (h3 : Q.RS = 9) (h4 : Q.SP = 3) (h5 : Q.angle_QRS = 90) (h6 : Q.angle_SPQ = 90) :
  area_of_quadrilateral Q = 48 :=
by
  -- Here would be the proof
  sorry

end quadrilateral_area_is_48_l281_281695


namespace product_of_sums_of_squares_l281_281568

-- Given conditions as definitions
def sum_of_squares (a b : ℤ) : ℤ := a^2 + b^2

-- Prove that the product of two sums of squares is also a sum of squares
theorem product_of_sums_of_squares (a b n k : ℤ) (K P : ℤ) (hK : K = sum_of_squares a b) (hP : P = sum_of_squares n k) :
    K * P = (a * n + b * k)^2 + (a * k - b * n)^2 := 
by
  sorry

end product_of_sums_of_squares_l281_281568


namespace problem_part1_problem_part2_l281_281792

noncomputable def quadratic_roots_conditions (x1 x2 m : ℝ) : Prop :=
  (x1 = 1) ∧ (x1 + x2 = 6) ∧ (x1 * x2 = 2 * m - 1)

noncomputable def existence_of_m (x1 x2 : ℝ) (m : ℝ) : Prop :=
  (x1 = 1) ∧ (x1 + x2 = 6) ∧ (x1 * x2 = 2 * m - 1) ∧ ((x1 - 1) * (x2 - 1) = 6 / (m - 5))

theorem problem_part1 : 
  ∃ x2 m, quadratic_roots_conditions 1 x2 m :=
sorry

theorem problem_part2 :
  ∃ m, ∃ x2, existence_of_m 1 x2 m ∧ m ≤ 5 :=
sorry

end problem_part1_problem_part2_l281_281792


namespace complement_union_and_complement_intersect_l281_281241

-- Definitions of sets according to the problem conditions
def U : Set ℝ := Set.univ
def A : Set ℝ := { x | 3 ≤ x ∧ x < 7 }
def B : Set ℝ := { x | 2 < x ∧ x < 10 }

-- The correct answers derived in the solution
def complement_union_A_B : Set ℝ := { x | x ≤ 2 ∨ 10 ≤ x }
def complement_A_intersect_B : Set ℝ := { x | (2 < x ∧ x < 3) ∨ (7 ≤ x ∧ x < 10) }

-- Statement of the mathematically equivalent proof problem
theorem complement_union_and_complement_intersect:
  (Set.compl (A ∪ B) = complement_union_A_B) ∧ 
  ((Set.compl A) ∩ B = complement_A_intersect_B) :=
  by 
    sorry

end complement_union_and_complement_intersect_l281_281241


namespace smallest_sum_of_inverses_l281_281817

theorem smallest_sum_of_inverses 
  (x y : ℕ) (hx : x ≠ y) (h1 : 0 < x) (h2 : 0 < y) (h_condition : (1 / x : ℚ) + 1 / y = 1 / 15) :
  x + y = 64 := 
sorry

end smallest_sum_of_inverses_l281_281817


namespace min_value_of_x2_add_y2_l281_281842

theorem min_value_of_x2_add_y2 (x y : ℝ) (h : (x + 5)^2 + (y - 12)^2 = 14^2) : x^2 + y^2 ≥ 1 :=
sorry

end min_value_of_x2_add_y2_l281_281842


namespace combined_area_win_bonus_l281_281757

theorem combined_area_win_bonus (r : ℝ) (P_win P_bonus : ℝ) : 
  r = 8 → P_win = 1 / 4 → P_bonus = 1 / 8 → 
  (P_win * (Real.pi * r^2) + P_bonus * (Real.pi * r^2) = 24 * Real.pi) :=
by
  intro h_r h_Pwin h_Pbonus
  rw [h_r, h_Pwin, h_Pbonus]
  -- Calculation is skipped as per the instructions
  sorry

end combined_area_win_bonus_l281_281757


namespace sqrt_sum_difference_product_l281_281091

open Real

theorem sqrt_sum_difference_product :
  (sqrt 3 + sqrt 2) * (sqrt 3 - sqrt 2) = 1 := by
  sorry

end sqrt_sum_difference_product_l281_281091


namespace alpha_quadrant_l281_281295

variable {α : ℝ}

theorem alpha_quadrant
  (sin_alpha_neg : Real.sin α < 0)
  (tan_alpha_pos : Real.tan α > 0) :
  ∃ k : ℤ, k = 1 ∧ π < α - 2 * π * k ∧ α - 2 * π * k < 3 * π :=
by
  sorry

end alpha_quadrant_l281_281295


namespace probability_0_2_l281_281686

noncomputable def measurement_result : Type := ℝ

def normal_distribution (μ σ : ℝ) (x : ℝ) : ℝ :=
  1 / (σ * (2 * Real.pi).sqrt) * Real.exp (- ((x - μ) ^ 2) / (2 * σ ^ 2))

variable (σ : ℝ)
variable (ξ : ℝ → ℝ)

axiom hξ : ξ = normal_distribution 1 σ

axiom h0_1 : 
  ∫ x in 0..1, ξ x = 0.4

theorem probability_0_2 : 
  ∫ x in 0..2, ξ x = 0.8 := 
sorry

end probability_0_2_l281_281686


namespace problem_l281_281826

def setA : Set ℝ := {x : ℝ | 0 < x ∧ x ≤ 2}
def setB : Set ℝ := {x : ℝ | x ≤ 3}

theorem problem : setA ∩ setB = setA := sorry

end problem_l281_281826


namespace tamia_total_slices_and_pieces_l281_281249

-- Define the conditions
def num_bell_peppers : ℕ := 5
def slices_per_pepper : ℕ := 20
def num_large_slices : ℕ := num_bell_peppers * slices_per_pepper
def num_half_slices : ℕ := num_large_slices / 2
def small_pieces_per_slice : ℕ := 3
def num_small_pieces : ℕ := num_half_slices * small_pieces_per_slice
def num_uncut_slices : ℕ := num_half_slices

-- Define the total number of pieces and slices
def total_pieces_and_slices : ℕ := num_uncut_slices + num_small_pieces

-- State the theorem and provide a placeholder for the proof
theorem tamia_total_slices_and_pieces : total_pieces_and_slices = 200 :=
by
  sorry

end tamia_total_slices_and_pieces_l281_281249


namespace monotonic_increase_range_of_alpha_l281_281520

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ :=
  (1 / 2) * Real.sin (ω * x) - (Real.sqrt 3 / 2) * Real.cos (ω * x)

theorem monotonic_increase_range_of_alpha
  (ω : ℝ) (hω : ω > 0)
  (zeros_form_ap : ∀ k : ℤ, ∃ x₀ : ℝ, f ω x₀ = 0 ∧ ∀ n : ℤ, f ω (x₀ + n * (π / 2)) = 0) :
  ∃ α : ℝ, 0 < α ∧ α < 5 * π / 12 ∧ ∀ x y : ℝ, 0 ≤ x ∧ x ≤ y ∧ y ≤ α → f ω x ≤ f ω y :=
sorry

end monotonic_increase_range_of_alpha_l281_281520


namespace sum_mobile_phone_keypad_l281_281971

/-- The numbers on a standard mobile phone keypad are 0 through 9. -/
def mobile_phone_keypad : List ℕ := [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

/-- The sum of all the numbers on a standard mobile phone keypad is 45. -/
theorem sum_mobile_phone_keypad : mobile_phone_keypad.sum = 45 := by
  sorry

end sum_mobile_phone_keypad_l281_281971


namespace spellbook_cost_in_gold_l281_281832

-- Define the constants
def num_spellbooks : ℕ := 5
def cost_potion_kit_in_silver : ℕ := 20
def num_potion_kits : ℕ := 3
def cost_owl_in_gold : ℕ := 28
def conversion_rate : ℕ := 9
def total_payment_in_silver : ℕ := 537

-- Define the problem to prove the cost of each spellbook in gold given the conditions
theorem spellbook_cost_in_gold : (total_payment_in_silver 
  - (cost_potion_kit_in_silver * num_potion_kits + cost_owl_in_gold * conversion_rate)) / num_spellbooks / conversion_rate = 5 := 
  by
  sorry

end spellbook_cost_in_gold_l281_281832


namespace parrots_left_l281_281721

theorem parrots_left 
  (c : Nat)   -- The initial number of crows
  (x : Nat)   -- The number of parrots and crows that flew away
  (h1 : 7 + c = 13)          -- Initial total number of birds
  (h2 : c - x = 1)           -- Number of crows left
  : 7 - x = 2 :=             -- Number of parrots left
by
  sorry

end parrots_left_l281_281721


namespace economy_class_seats_l281_281466

-- Definitions based on the conditions
def first_class_people : ℕ := 3
def business_class_people : ℕ := 22
def economy_class_fullness (E : ℕ) : ℕ := E / 2

-- Problem statement: Proving E == 50 given the conditions
theorem economy_class_seats :
  ∃ E : ℕ,  economy_class_fullness E = first_class_people + business_class_people → E = 50 :=
by
  sorry

end economy_class_seats_l281_281466


namespace minimum_inhabitants_to_ask_l281_281397

def knights_count : ℕ := 50
def civilians_count : ℕ := 15

theorem minimum_inhabitants_to_ask (knights civilians : ℕ) (h_knights : knights = knights_count) (h_civilians : civilians = civilians_count) :
  ∃ n, (∀ asked : ℕ, (asked ≥ n) → asked - civilians > civilians) ∧ n = 31 :=
by
  sorry

end minimum_inhabitants_to_ask_l281_281397


namespace num_5_letter_words_with_at_least_two_consonants_l281_281528

open Finset

def letters : Finset Char := {'A', 'B', 'C', 'D', 'E', 'F'}
def consonants : Finset Char := {'B', 'C', 'D', 'F'}
def vowels : Finset Char := {'A', 'E'}

def total_5_letter_words : ℕ := 6^5
def words_with_0_consonants : ℕ := 2^5
def words_with_1_consonant : ℕ := 5 * 4 * 2^4

theorem num_5_letter_words_with_at_least_two_consonants : 
  total_5_letter_words - (words_with_0_consonants + words_with_1_consonant) = 7424 := by
  sorry

end num_5_letter_words_with_at_least_two_consonants_l281_281528


namespace average_speed_l281_281551

theorem average_speed (d1 d2 d3 v1 v2 v3 total_distance total_time avg_speed : ℝ)
    (h1 : d1 = 40) (h2 : d2 = 20) (h3 : d3 = 10) 
    (h4 : v1 = 8) (h5 : v2 = 40) (h6 : v3 = 20) 
    (h7 : total_distance = d1 + d2 + d3)
    (h8 : total_time = d1 / v1 + d2 / v2 + d3 / v3) 
    (h9 : avg_speed = total_distance / total_time) : avg_speed = 11.67 :=
by 
  sorry

end average_speed_l281_281551


namespace smallest_possible_value_l281_281801

theorem smallest_possible_value (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (hxy : x ≠ y) (h : (1 / (x : ℝ)) + (1 / (y : ℝ)) = 1 / 15) : x + y = 64 :=
sorry

end smallest_possible_value_l281_281801


namespace smallest_positive_integer_conditioned_l281_281001

theorem smallest_positive_integer_conditioned :
  ∃ a : ℕ, a > 0 ∧ (a % 4 = 3) ∧ (a % 3 = 2) ∧ ∀ b : ℕ, b > 0 ∧ (b % 4 = 3) ∧ (b % 3 = 2) → a ≤ b :=
begin
  sorry
end

end smallest_positive_integer_conditioned_l281_281001


namespace sin_half_alpha_l281_281170

variable (α : ℝ)
variable (h₀ : 0 < α ∧ α < π/2)
variable (h₁ : cos α = (1 + sqrt 5) / 4)

theorem sin_half_alpha : sin (α / 2) = (sqrt 5 - 1) / 4 :=
by
  sorry

end sin_half_alpha_l281_281170


namespace sin_half_angle_l281_281171

theorem sin_half_angle (α : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : Real.cos α = (1 + Real.sqrt 5) / 4) : 
  Real.sin (α / 2) = (Real.sqrt 5 - 1) / 4 := 
by 
  sorry

end sin_half_angle_l281_281171


namespace sin_half_alpha_l281_281176

noncomputable def alpha : ℝ := sorry
def is_acute (alpha : ℝ) : Prop := 0 < alpha ∧ alpha < π / 2

axiom acos_alpha : cos alpha = (1 + sqrt 5) / 4
axiom acute_alpha : is_acute alpha

theorem sin_half_alpha : sin (alpha / 2) = (sqrt 5 - 1) / 4 :=
by
  sorry

end sin_half_alpha_l281_281176


namespace allison_rolls_greater_probability_l281_281637

theorem allison_rolls_greater_probability :
  let allison_roll : ℕ := 6
  let charlie_prob_less_6 := 5 / 6
  let mia_prob_rolls_3 := 4 / 6
  let combined_prob := charlie_prob_less_6 * (mia_prob_rolls_3)
  combined_prob = 5 / 9 := by
  sorry

end allison_rolls_greater_probability_l281_281637


namespace root_expression_value_l281_281534

theorem root_expression_value (m : ℝ) (h : 2 * m^2 + 3 * m - 1 = 0) : 4 * m^2 + 6 * m - 2019 = -2017 :=
by
  sorry

end root_expression_value_l281_281534


namespace min_value_proof_l281_281560

noncomputable def min_value_of_expression (a b c d e f g h : ℝ) : ℝ :=
  (a * e)^2 + (b * f)^2 + (c * g)^2 + (d * h)^2

theorem min_value_proof (a b c d e f g h : ℝ)
  (h1 : a * b * c * d = 8)
  (h2 : e * f * g * h = 16) :
  ∃ (x : ℝ), x = 32 ∧ min_value_of_expression a b c d e f g h = x :=
by
  use 32
  sorry

end min_value_proof_l281_281560


namespace problem_correct_choice_l281_281818

-- Definitions of the propositions
def p : Prop := ∃ n : ℕ, 3 = 2 * n + 1
def q : Prop := ∃ n : ℕ, 5 = 2 * n

-- The problem statement
theorem problem_correct_choice : p ∨ q :=
sorry

end problem_correct_choice_l281_281818


namespace snail_climbs_well_l281_281306

theorem snail_climbs_well (h : ℕ) (c : ℕ) (s : ℕ) (d : ℕ) (h_eq : h = 12) (c_eq : c = 3) (s_eq : s = 2) : d = 10 :=
by
  sorry

end snail_climbs_well_l281_281306


namespace correct_factorization_l281_281287

theorem correct_factorization :
  (∀ (x y : ℝ), x^2 + y^2 ≠ (x + y)^2) ∧
  (∀ (x y : ℝ), x^2 + 2*x*y + y^2 ≠ (x - y)^2) ∧
  (∀ (x : ℝ), x^2 + x ≠ x * (x - 1)) ∧
  (∀ (x y : ℝ), x^2 - y^2 = (x + y) * (x - y)) :=
by 
  sorry

end correct_factorization_l281_281287


namespace integral_sin_from_0_to_pi_div_2_l281_281782

theorem integral_sin_from_0_to_pi_div_2 :
  ∫ x in (0 : ℝ)..(Real.pi / 2), Real.sin x = 1 := by
  sorry

end integral_sin_from_0_to_pi_div_2_l281_281782


namespace a_6_value_l281_281523

noncomputable def a_n (n : ℕ) : ℚ :=
  if h : n > 0 then (3 * n - 2) / (2 ^ (n - 1))
  else 0

theorem a_6_value : a_n 6 = 1 / 2 :=
by
  -- placeholder for the proof
  sorry

end a_6_value_l281_281523


namespace gcd_of_12547_23791_l281_281035

theorem gcd_of_12547_23791 : Nat.gcd 12547 23791 = 1 :=
by
  sorry

end gcd_of_12547_23791_l281_281035


namespace find_side_c_l281_281384

noncomputable theory

variables {A B C : ℝ}
variables {a b c : ℝ}

-- Define the conditions of the problem
def is_right_angled_triangle (A B C : ℝ) :=
  A + B + C = 180 ∧ C = 90

def side_opposite_angle (angle : ℝ) (side : ℝ) (A B C : ℝ) : Prop :=
  (angle = A ∧ side = a) ∨ (angle = B ∧ side = b) ∨ (angle = C ∧ side = c)

def given_conditions (A B C a b : ℝ) : Prop := 
  A = 30 ∧ a = 1 ∧ b = sqrt 3 ∧ is_right_angled_triangle A B C

-- The main theorem to prove
theorem find_side_c (A B C a b c : ℝ) (h : given_conditions A B C a b) : c = 2 :=
sorry

end find_side_c_l281_281384


namespace probability_computation_l281_281896

-- Definitions of individual success probabilities
def probability_Xavier_solving_problem : ℚ := 1 / 4
def probability_Yvonne_solving_problem : ℚ := 2 / 3
def probability_William_solving_problem : ℚ := 7 / 10
def probability_Zelda_solving_problem : ℚ := 5 / 8
def probability_Zelda_notsolving_problem : ℚ := 1 - probability_Zelda_solving_problem

-- The target probability that only Xavier, Yvonne, and William, but not Zelda, will solve the problem
def target_probability : ℚ := (1 / 4) * (2 / 3) * (7 / 10) * (3 / 8)

-- The simplified form of the computed probability
def simplified_target_probability : ℚ := 7 / 160

-- Lean 4 statement to prove the equality of the computed and the target probabilities
theorem probability_computation :
  target_probability = simplified_target_probability := by
  sorry

end probability_computation_l281_281896


namespace sin_half_alpha_l281_281187

theorem sin_half_alpha (α : ℝ) (hα : 0 < α ∧ α < π / 2) (hcos : Real.cos α = (1 + Real.sqrt 5) / 4) : 
  Real.sin (α / 2) = (Real.sqrt 5 - 1) / 4 := 
by 
  sorry

end sin_half_alpha_l281_281187


namespace double_transmission_yellow_twice_double_transmission_less_single_l281_281379

variables {α : ℝ} (hα : 0 < α ∧ α < 1)

-- Statement B
theorem double_transmission_yellow_twice (hα : 0 < α ∧ α < 1) :
  probability_displays_yellow_twice = α^2 :=
sorry

-- Statement D
theorem double_transmission_less_single (hα : 0 < α ∧ α < 1) :
  (1 - α)^2 < (1 - α) :=
sorry

end double_transmission_yellow_twice_double_transmission_less_single_l281_281379


namespace option_d_necessary_sufficient_l281_281138

theorem option_d_necessary_sufficient (a : ℝ) : (a ≠ 0) ↔ (∃! x : ℝ, a * x = 1) := 
sorry

end option_d_necessary_sufficient_l281_281138


namespace bouncy_balls_per_package_l281_281990

variable (x : ℝ)

def maggie_bought_packs : ℝ := 8.0 * x
def maggie_gave_away_packs : ℝ := 4.0 * x
def maggie_bought_again_packs : ℝ := 4.0 * x
def total_kept_bouncy_balls : ℝ := 80

theorem bouncy_balls_per_package :
  (maggie_bought_packs x = total_kept_bouncy_balls) → 
  x = 10 :=
by
  intro h
  sorry

end bouncy_balls_per_package_l281_281990


namespace jello_cost_calculation_l281_281854

-- Conditions as definitions
def jello_per_pound : ℝ := 1.5
def tub_volume_cubic_feet : ℝ := 6
def cubic_foot_to_gallons : ℝ := 7.5
def gallon_weight_pounds : ℝ := 8
def cost_per_tablespoon_jello : ℝ := 0.5

-- Tub total water calculation
def tub_water_gallons (volume_cubic_feet : ℝ) (cubic_foot_to_gallons : ℝ) : ℝ :=
  volume_cubic_feet * cubic_foot_to_gallons

-- Water weight calculation
def water_weight_pounds (water_gallons : ℝ) (gallon_weight_pounds : ℝ) : ℝ :=
  water_gallons * gallon_weight_pounds

-- Jello mix required calculation
def jello_mix_tablespoons (water_pounds : ℝ) (jello_per_pound : ℝ) : ℝ :=
  water_pounds * jello_per_pound

-- Total cost calculation
def total_cost (jello_mix_tablespoons : ℝ) (cost_per_tablespoon_jello : ℝ) : ℝ :=
  jello_mix_tablespoons * cost_per_tablespoon_jello

-- Theorem statement
theorem jello_cost_calculation :
  total_cost (jello_mix_tablespoons (water_weight_pounds (tub_water_gallons tub_volume_cubic_feet cubic_foot_to_gallons) gallon_weight_pounds) jello_per_pound) cost_per_tablespoon_jello = 270 := 
by sorry

end jello_cost_calculation_l281_281854


namespace cone_base_radius_l281_281225

theorem cone_base_radius (r_paper : ℝ) (n_parts : ℕ) (r_cone_base : ℝ) 
  (h_radius_paper : r_paper = 16)
  (h_n_parts : n_parts = 4)
  (h_cone_part : r_cone_base = r_paper / n_parts) : r_cone_base = 4 := by
  sorry

end cone_base_radius_l281_281225


namespace prob_at_least_one_2_in_two_8_sided_dice_l281_281104

/-- Probability of getting at least one 2 when rolling two 8-sided dice -/
theorem prob_at_least_one_2_in_two_8_sided_dice : 
  let total_outcomes := 64
  let favorable_outcomes := 15
  (favorable_outcomes : ℝ) / (total_outcomes : ℝ) = 15 / 64 :=
by
  sorry

end prob_at_least_one_2_in_two_8_sided_dice_l281_281104


namespace positive_solution_l281_281047

variable {x y z : ℝ}

theorem positive_solution (h1 : x * y = 8 - 2 * x - 3 * y)
    (h2 : y * z = 8 - 4 * y - 2 * z)
    (h3 : x * z = 40 - 5 * x - 3 * z) :
    x = 10 := by
  sorry

end positive_solution_l281_281047


namespace no_solution_system_of_equations_l281_281028

theorem no_solution_system_of_equations :
  ¬ ∃ (x y : ℝ), (2 * x - 3 * y = 6) ∧ (4 * x - 6 * y = 8) ∧ (5 * x - 5 * y = 15) :=
by {
  sorry
}

end no_solution_system_of_equations_l281_281028


namespace smallest_product_not_factor_of_48_exists_l281_281435

theorem smallest_product_not_factor_of_48_exists : 
  ∃ (a b : ℕ), (a ≠ b) ∧ (a ∣ 48) ∧ (b ∣ 48) ∧ ¬ (a * b ∣ 48) ∧ 
  (∀ (c d : ℕ), (c ≠ d) ∧ (c ∣ 48) ∧ (d ∣ 48) ∧ ¬ (c * d ∣ 48) → a * b ≤ c * d) ∧ (a * b = 18) :=
sorry

end smallest_product_not_factor_of_48_exists_l281_281435


namespace total_dots_not_visible_proof_l281_281037

def total_dots_on_one_die : ℕ := 21

def total_dots_on_five_dice : ℕ := 5 * total_dots_on_one_die

def visible_numbers : List ℕ := [1, 2, 3, 1, 4, 5, 6, 2]

def sum_visible_numbers : ℕ := visible_numbers.sum

def total_dots_not_visible (total : ℕ) (visible_sum : ℕ) : ℕ :=
  total - visible_sum

theorem total_dots_not_visible_proof :
  total_dots_not_visible total_dots_on_five_dice sum_visible_numbers = 81 :=
by
  sorry

end total_dots_not_visible_proof_l281_281037


namespace how_many_buns_each_student_gets_l281_281357

theorem how_many_buns_each_student_gets 
  (packages : ℕ) 
  (buns_per_package : ℕ) 
  (classes : ℕ) 
  (students_per_class : ℕ)
  (h1 : buns_per_package = 8)
  (h2 : packages = 30)
  (h3 : classes = 4)
  (h4 : students_per_class = 30) :
  (packages * buns_per_package) / (classes * students_per_class) = 2 :=
by sorry

end how_many_buns_each_student_gets_l281_281357


namespace correct_inequality_l281_281963

variable (a b : ℝ)

theorem correct_inequality (h : a > b) : a - 3 > b - 3 :=
by
  sorry

end correct_inequality_l281_281963


namespace monotonicity_a_le_0_has_one_zero_point_condition_1_has_one_zero_point_condition_2_l281_281348

-- Define the function f
def f (x a b : ℝ) := (x - 1) * exp x - a * x^2 + b

-- Define the monotonicity part
theorem monotonicity_a_le_0 (a b : ℝ) (h : a ≤ 0) : 
  (∀ x, deriv (λ x, f x a b) x = x * (exp x - 2 * a)) ∧ 
  (∀ x < 0, deriv (λ x, f x a b) x < 0) ∧ 
  (∀ x > 0, deriv (λ x, f x a b) x > 0) :=
sorry

-- Define the conditions to check exactly one zero point for Condition ①
theorem has_one_zero_point_condition_1 (a b : ℝ) 
(h1 : 1/2 < a) (h2 : a ≤ exp 2 / 2) (h3 : b > 2 * a) : 
  ∃! x, f x a b = 0 :=
sorry

-- Define the conditions to check exactly one zero point for Condition ②
theorem has_one_zero_point_condition_2 (a b : ℝ) 
(h1 : 0 < a) (h2 : a < 1 / 2) (h3 : b ≤ 2 * a) : 
  ∃! x, f x a b = 0 :=
sorry

end monotonicity_a_le_0_has_one_zero_point_condition_1_has_one_zero_point_condition_2_l281_281348


namespace ufo_convention_males_l281_281766

-- Define the total number of attendees
constant total_attendees : Nat := 120

-- Define the conditions in the problem
constant num_female : Nat
constant num_male : Nat

axiom total_condition : num_female + num_male = total_attendees
axiom more_males_condition : num_male = num_female + 4

-- State the problem to prove the number of male attendees
theorem ufo_convention_males : num_male = 62 :=
by
  sorry

end ufo_convention_males_l281_281766


namespace total_area_of_paintings_l281_281897

-- Definitions based on the conditions
def painting1_area := 3 * (5 * 5) -- 3 paintings of 5 feet by 5 feet
def painting2_area := 10 * 8 -- 1 painting of 10 feet by 8 feet
def painting3_area := 5 * 9 -- 1 painting of 5 feet by 9 feet

-- The proof statement we aim to prove
theorem total_area_of_paintings : painting1_area + painting2_area + painting3_area = 200 :=
by
  sorry

end total_area_of_paintings_l281_281897


namespace coffee_containers_used_l281_281572

theorem coffee_containers_used :
  let Suki_coffee := 6.5 * 22
  let Jimmy_coffee := 4.5 * 18
  let combined_coffee := Suki_coffee + Jimmy_coffee
  let containers := combined_coffee / 8
  containers = 28 := 
by
  sorry

end coffee_containers_used_l281_281572


namespace num_5_letter_words_with_at_least_two_consonants_l281_281527

open Finset

def letters : Finset Char := {'A', 'B', 'C', 'D', 'E', 'F'}
def consonants : Finset Char := {'B', 'C', 'D', 'F'}
def vowels : Finset Char := {'A', 'E'}

def total_5_letter_words : ℕ := 6^5
def words_with_0_consonants : ℕ := 2^5
def words_with_1_consonant : ℕ := 5 * 4 * 2^4

theorem num_5_letter_words_with_at_least_two_consonants : 
  total_5_letter_words - (words_with_0_consonants + words_with_1_consonant) = 7424 := by
  sorry

end num_5_letter_words_with_at_least_two_consonants_l281_281527


namespace smallest_x_condition_l281_281281

theorem smallest_x_condition (x : ℕ) : (∃ x > 0, (3 * x + 28)^2 % 53 = 0) -> x = 26 := 
by
  sorry

end smallest_x_condition_l281_281281


namespace min_letters_required_l281_281591

theorem min_letters_required (n : ℕ) (hn : n = 26) : 
  ∃ k, (∀ (collectors : Fin n) (leader : Fin n), k = 2 * (n - 1)) := 
sorry

end min_letters_required_l281_281591


namespace greatest_x_value_l281_281939

theorem greatest_x_value :
  ∃ x, (∀ y, (y ≠ 6) → (y^2 - y - 30) / (y - 6) = 2 / (y + 4) → y ≤ x) ∧ (x^2 - x - 30) / (x - 6) = 2 / (x + 4) ∧ x ≠ 6 ∧ x = -3 :=
sorry

end greatest_x_value_l281_281939


namespace problem_l281_281825

def setA : Set ℝ := {x : ℝ | 0 < x ∧ x ≤ 2}
def setB : Set ℝ := {x : ℝ | x ≤ 3}

theorem problem : setA ∩ setB = setA := sorry

end problem_l281_281825


namespace ratio_B_A_l281_281986

theorem ratio_B_A (A B : ℤ) (h : ∀ (x : ℝ), x ≠ -6 → x ≠ 0 → x ≠ 5 → 
  (A / (x + 6) + B / (x^2 - 5*x) = (x^3 - 3*x^2 + 12) / (x^3 + x^2 - 30*x))) :
  (B : ℚ) / A = 2.2 := by
  sorry

end ratio_B_A_l281_281986


namespace units_digit_of_n_squared_plus_2_n_is_7_l281_281862

def n : ℕ := 2023 ^ 2 + 2 ^ 2023

theorem units_digit_of_n_squared_plus_2_n_is_7 : (n ^ 2 + 2 ^ n) % 10 = 7 := 
by
  sorry

end units_digit_of_n_squared_plus_2_n_is_7_l281_281862


namespace jello_cost_l281_281852

def cost_to_fill_tub_with_jello (water_volume_cubic_feet : ℕ) (gallons_per_cubic_foot : ℕ) 
    (pounds_per_gallon : ℕ) (tablespoons_per_pound : ℕ) (cost_per_tablespoon : ℕ) : ℕ :=
  water_volume_cubic_feet * gallons_per_cubic_foot * pounds_per_gallon * tablespoons_per_pound * cost_per_tablespoon

theorem jello_cost (water_volume_cubic_feet : ℕ) (gallons_per_cubic_foot : ℕ) 
    (pounds_per_gallon : ℕ) (tablespoons_per_pound : ℕ) (cost_per_tablespoon : ℕ) : 
    water_volume_cubic_feet = 6 ∧ gallons_per_cubic_foot = 7 ∧ pounds_per_gallon = 8 ∧ 
    tablespoons_per_pound = 1 ∧ cost_per_tablespoon = 1 →
    cost_to_fill_tub_with_jello water_volume_cubic_feet gallons_per_cubic_foot pounds_per_gallon tablespoons_per_pound cost_per_tablespoon = 270 :=
  by 
    sorry

end jello_cost_l281_281852


namespace zoe_correct_percentage_l281_281487

noncomputable def t : ℝ := sorry  -- total number of problems
noncomputable def chloe_alone_correct : ℝ := 0.70 * (1/3 * t)  -- Chloe's correct answers alone
noncomputable def chloe_total_correct : ℝ := 0.85 * t  -- Chloe's overall correct answers
noncomputable def together_correct : ℝ := chloe_total_correct - chloe_alone_correct  -- Problems solved correctly together
noncomputable def zoe_alone_correct : ℝ := 0.85 * (1/3 * t)  -- Zoe's correct answers alone
noncomputable def zoe_total_correct : ℝ := zoe_alone_correct + together_correct  -- Zoe's total correct answers
noncomputable def zoe_percentage_correct : ℝ := (zoe_total_correct / t) * 100  -- Convert to percentage

theorem zoe_correct_percentage : zoe_percentage_correct = 90 := 
by
  sorry

end zoe_correct_percentage_l281_281487


namespace alternating_binomial_sum_l281_281662

theorem alternating_binomial_sum :
  \(\sum_{k=0}^{50} (-1)^k \binom{101}{2k} = -2^{50}\) := 
  sorry

end alternating_binomial_sum_l281_281662


namespace inequality_holds_l281_281788

variable (a b c d : ℝ)
variable (ha : a > 0)
variable (hb : b > 0)
variable (hc : c > 0)
variable (hd : d > 0)
variable (h : 1 / (a^3 + 1) + 1 / (b^3 + 1) + 1 / (c^3 + 1) + 1 / (d^3 + 1) = 2)

theorem inequality_holds (ha : a > 0)
  (hb : b > 0) 
  (hc : c > 0) 
  (hd : d > 0) 
  (h : 1 / (a^3 + 1) + 1 / (b^3 + 1) + 1 / (c^3 + 1) + 1 / (d^3 + 1) = 2) :
  (1 - a) / (a^2 - a + 1) + (1 - b) / (b^2 - b + 1) + (1 - c) / (c^2 - c + 1) + (1 - d) / (d^2 - d + 1) ≥ 0 := by
  sorry

end inequality_holds_l281_281788


namespace bananas_per_box_l281_281857

def total_bananas : ℕ := 40
def number_of_boxes : ℕ := 10

theorem bananas_per_box : total_bananas / number_of_boxes = 4 := by
  sorry

end bananas_per_box_l281_281857


namespace A_inter_B_eq_A_l281_281821

def A := {x : ℝ | 0 < x ∧ x ≤ 2}
def B := {x : ℝ | x ≤ 3}

theorem A_inter_B_eq_A : A ∩ B = A := 
by 
  sorry 

end A_inter_B_eq_A_l281_281821


namespace laura_house_distance_l281_281859

-- Definitions based on conditions
def x : Real := 10  -- Distance from Laura's house to her school in miles

def distance_to_school_per_day := 2 * x
def school_days_per_week := 5
def distance_to_school_per_week := school_days_per_week * distance_to_school_per_day

def distance_to_supermarket := x + 10
def supermarket_trips_per_week := 2
def distance_to_supermarket_per_trip := 2 * distance_to_supermarket
def distance_to_supermarket_per_week := supermarket_trips_per_week * distance_to_supermarket_per_trip

def total_distance_per_week := 220

-- The proof statement
theorem laura_house_distance :
  distance_to_school_per_week + distance_to_supermarket_per_week = total_distance_per_week ∧ x = 10 := by
  sorry

end laura_house_distance_l281_281859


namespace kolya_made_mistake_l281_281983

theorem kolya_made_mistake (ab cd effe : ℕ)
  (h_eq : ab * cd = effe)
  (h_eff_div_11 : effe % 11 = 0)
  (h_ab_cd_not_div_11 : ab % 11 ≠ 0 ∧ cd % 11 ≠ 0) :
  false :=
by
  -- Note: This is where the proof would go, but we are illustrating the statement only.
  sorry

end kolya_made_mistake_l281_281983


namespace actual_distance_between_city_centers_l281_281085

-- Define the conditions
def map_distance_cm : ℝ := 45
def scale_cm_to_km : ℝ := 10

-- Define the proof statement
theorem actual_distance_between_city_centers
  (md : ℝ := map_distance_cm)
  (scale : ℝ := scale_cm_to_km) :
  md * scale = 450 :=
by
  sorry

end actual_distance_between_city_centers_l281_281085


namespace paul_collected_total_cans_l281_281868

theorem paul_collected_total_cans :
  let saturday_bags := 10
  let sunday_bags := 5
  let saturday_cans_per_bag := 12
  let sunday_cans_per_bag := 15
  let saturday_total_cans := saturday_bags * saturday_cans_per_bag
  let sunday_total_cans := sunday_bags * sunday_cans_per_bag
  let total_cans := saturday_total_cans + sunday_total_cans
  total_cans = 195 := 
by
  sorry

end paul_collected_total_cans_l281_281868


namespace area_of_square_l281_281564

-- Conditions: Points A (5, -2) and B (5, 3) are adjacent corners of a square.
def A : ℝ × ℝ := (5, -2)
def B : ℝ × ℝ := (5, 3)

-- The statement to prove that the area of the square formed by these points is 25.
theorem area_of_square : (∃ s : ℝ, s = Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)) → s^2 = 25 :=
sorry

end area_of_square_l281_281564


namespace find_x_l281_281516

noncomputable def vector_a : ℝ × ℝ := (1, 2)
noncomputable def vector_b : ℝ × ℝ := (1, 5)
noncomputable def vector_c (x : ℝ) : ℝ × ℝ := (x, 1)

def collinear (u v : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, u = (k * v.1, k * v.2)

theorem find_x :
  ∃ x : ℝ, collinear (2 • vector_a - vector_b) (vector_c x) ∧ x = -1 := by
  sorry

end find_x_l281_281516


namespace sum_of_coefficients_zero_l281_281984

open Real

theorem sum_of_coefficients_zero (a b c p1 p2 q1 q2 : ℝ)
  (h1 : ∃ p1 p2 : ℝ, p1 ≠ p2 ∧ a * p1^2 + b * p1 + c = 0 ∧ a * p2^2 + b * p2 + c = 0)
  (h2 : ∃ q1 q2 : ℝ, q1 ≠ q2 ∧ c * q1^2 + b * q1 + a = 0 ∧ c * q2^2 + b * q2 + a = 0)
  (h3 : q1 = p1 + (p2 - p1) / 2 ∧ p2 = p1 + (p2 - p1) ∧ q2 = p1 + 3 * (p2 - p1) / 2) :
  a + c = 0 := sorry

end sum_of_coefficients_zero_l281_281984


namespace total_students_in_class_l281_281688

def students_play_football : Nat := 26
def students_play_tennis : Nat := 20
def students_play_both : Nat := 17
def students_play_neither : Nat := 7

theorem total_students_in_class :
  (students_play_football + students_play_tennis - students_play_both + students_play_neither) = 36 :=
by
  sorry

end total_students_in_class_l281_281688


namespace smallest_product_not_factor_of_48_l281_281436

theorem smallest_product_not_factor_of_48 (a b : ℕ) (h1 : a ≠ b) (h2 : a ∣ 48) (h3 : b ∣ 48) (h4 : ¬ (a * b ∣ 48)) : a * b = 32 :=
sorry

end smallest_product_not_factor_of_48_l281_281436


namespace smartphone_cost_l281_281865

theorem smartphone_cost :
  let current_savings : ℕ := 40
  let weekly_saving : ℕ := 15
  let num_months : ℕ := 2
  let weeks_in_month : ℕ := 4 
  let total_weeks := num_months * weeks_in_month
  let total_savings := weekly_saving * total_weeks
  let total_money := current_savings + total_savings
  total_money = 160 := by
  sorry

end smartphone_cost_l281_281865


namespace sara_total_score_l281_281222

-- Definitions based on the conditions
def correct_points (correct_answers : Nat) : Int := correct_answers * 2
def incorrect_points (incorrect_answers : Nat) : Int := incorrect_answers * (-1)
def unanswered_points (unanswered_questions : Nat) : Int := unanswered_questions * 0

def total_score (correct_answers incorrect_answers unanswered_questions : Nat) : Int :=
  correct_points correct_answers + incorrect_points incorrect_answers + unanswered_points unanswered_questions

-- The main theorem stating the problem requirement
theorem sara_total_score :
  total_score 18 10 2 = 26 :=
by
  sorry

end sara_total_score_l281_281222


namespace simplify_exponential_expression_l281_281898

theorem simplify_exponential_expression :
  (3 * (-5)^2)^(3/4) = (75)^(3/4) := 
  sorry

end simplify_exponential_expression_l281_281898


namespace solve_for_x_l281_281518

theorem solve_for_x (x y : ℝ) (h1 : 3 * x - 2 * y = 7) (h2 : x + 3 * y = 6) : x = 3 := 
by 
  sorry

end solve_for_x_l281_281518


namespace total_sandwiches_prepared_l281_281753

def num_people := 219.0
def sandwiches_per_person := 3.0

theorem total_sandwiches_prepared : num_people * sandwiches_per_person = 657.0 :=
by
  sorry

end total_sandwiches_prepared_l281_281753


namespace number_of_biscuits_per_day_l281_281737

theorem number_of_biscuits_per_day 
  (price_cupcake : ℝ) (price_cookie : ℝ) (price_biscuit : ℝ)
  (cupcakes_per_day : ℕ) (cookies_per_day : ℕ) (total_earnings_five_days : ℝ) :
  price_cupcake = 1.5 → 
  price_cookie = 2 → 
  price_biscuit = 1 → 
  cupcakes_per_day = 20 → 
  cookies_per_day = 10 → 
  total_earnings_five_days = 350 →
  (total_earnings_five_days - 
   (5 * (cupcakes_per_day * price_cupcake + cookies_per_day * price_cookie))) / (5 * price_biscuit) = 20 :=
by
  intros price_cupcake_eq price_cookie_eq price_biscuit_eq cupcakes_per_day_eq cookies_per_day_eq total_earnings_five_days_eq
  sorry

end number_of_biscuits_per_day_l281_281737


namespace sum_of_xyz_l281_281053

theorem sum_of_xyz (x y z : ℝ) (h : (x - 3)^2 + (y - 4)^2 + (z - 5)^2 = 0) : x + y + z = 12 :=
sorry

end sum_of_xyz_l281_281053


namespace min_rice_pounds_l281_281067

variable {o r : ℝ}

theorem min_rice_pounds (h1 : o ≥ 8 + r / 3) (h2 : o ≤ 2 * r) : r ≥ 5 :=
sorry

end min_rice_pounds_l281_281067


namespace max_val_4ab_sqrt3_12bc_l281_281559

theorem max_val_4ab_sqrt3_12bc {a b c : ℝ} (h₀ : 0 ≤ a) (h₁ : 0 ≤ b) (h₂ : 0 ≤ c) (h₃ : a^2 + b^2 + c^2 = 3) :
  4 * a * b * Real.sqrt 3 + 12 * b * c ≤ Real.sqrt 39 :=
sorry

end max_val_4ab_sqrt3_12bc_l281_281559


namespace oldest_child_age_l281_281259

theorem oldest_child_age 
  (x : ℕ)
  (h1 : (6 + 8 + 10 + x) / 4 = 9)
  (h2 : 6 + 8 + 10 = 24) :
  x = 12 := 
by 
  sorry

end oldest_child_age_l281_281259


namespace average_speed_l281_281290

-- Define the average speed v
variable {v : ℝ}

-- Conditions
def day1_distance : ℝ := 160  -- 160 miles on the first day
def day2_distance : ℝ := 280  -- 280 miles on the second day
def time_difference : ℝ := 3  -- 3 hours difference

-- Theorem to prove the average speed
theorem average_speed (h1 : day1_distance / v + time_difference = day2_distance / v) : v = 40 := 
by 
  sorry  -- Proof is omitted

end average_speed_l281_281290


namespace part1_part2_l281_281493

theorem part1 (n : Nat) (hn : 0 < n) : 
  (∃ k, -5^4 + 5^5 + 5^n = k^2) -> n = 5 :=
by
  sorry

theorem part2 (n : Nat) (hn : 0 < n) : 
  (∃ m, 2^4 + 2^7 + 2^n = m^2) -> n = 8 :=
by
  sorry

end part1_part2_l281_281493


namespace word_count_proof_l281_281525

def letters : List Char := ['A', 'B', 'C', 'D', 'E', 'F']
def consonants : List Char := ['B', 'C', 'D', 'F']
def vowels : List Char := ['A', 'E']

def count_unrestricted_words : ℕ := 6 ^ 5
def count_all_vowel_words : ℕ := 2 ^ 5
def count_one_consonant_words : ℕ := 5 * 4 * (2 ^ 4)
def count_fewer_than_two_consonant_words : ℕ := count_all_vowel_words + count_one_consonant_words

def count_words_with_at_least_two_consonants : ℕ :=
  count_unrestricted_words - count_fewer_than_two_consonant_words

theorem word_count_proof :
  count_words_with_at_least_two_consonants = 7424 := 
by
  -- Proof will be provided here. For now we skip it.
  sorry

end word_count_proof_l281_281525


namespace sum_of_solutions_l281_281153

theorem sum_of_solutions (x : ℝ) (h : x ≠ 1 ∧ x ≠ -1) :
  (∃ x₁ x₂ : ℝ, (3 * x₁ + 2) * (x₁ - 4) = 0 ∧ (3 * x₂ + 2) * (x₂ - 4) = 0 ∧
  x₁ ≠ 1 ∧ x₁ ≠ -1 ∧ x₂ ≠ 1 ∧ x₂ ≠ -1 ∧ x₁ + x₂ = 10 / 3) :=
sorry

end sum_of_solutions_l281_281153


namespace max_AC_not_RS_l281_281060

theorem max_AC_not_RS (TotalCars NoACCars MinRS MaxACnotRS : ℕ)
  (h1 : TotalCars = 100)
  (h2 : NoACCars = 49)
  (h3 : MinRS >= 51)
  (h4 : (TotalCars - NoACCars) - MinRS = MaxACnotRS)
  : MaxACnotRS = 0 :=
by
  sorry

end max_AC_not_RS_l281_281060


namespace calculate_rectangle_length_l281_281073

theorem calculate_rectangle_length (side_of_square : ℝ) (width_of_rectangle : ℝ)
  (length_of_wire : ℝ) (perimeter_of_rectangle : ℝ) :
  side_of_square = 20 → 
  width_of_rectangle = 14 → 
  length_of_wire = 4 * side_of_square →
  perimeter_of_rectangle = length_of_wire →
  2 * (width_of_rectangle + length_of_rectangle) = perimeter_of_rectangle →
  length_of_rectangle = 26 :=
by
  intros
  sorry

end calculate_rectangle_length_l281_281073


namespace find_R_l281_281371

def is_prime (n : ℤ) : Prop := n > 1 ∧ ∀ m : ℤ, m > 0 → m < n → ¬ (m ∣ n)

theorem find_R :
  ∃ R : ℤ, R > 0 ∧ (∃ Q : ℤ, is_prime (R^3 + 4 * R^2 + (Q - 93) * R + 14 * Q + 10)) ∧ R = 5 :=
  sorry

end find_R_l281_281371


namespace polynomial_coeff_sum_l281_281959

theorem polynomial_coeff_sum :
  let p := ((Polynomial.C 1 + Polynomial.X)^3 * (Polynomial.C 2 + Polynomial.X)^2)
  let a0 := p.coeff 0
  let a2 := p.coeff 2
  let a4 := p.coeff 4
  a4 + a2 + a0 = 36 := by 
  sorry

end polynomial_coeff_sum_l281_281959


namespace hexagon_triangle_count_l281_281465

-- Definitions based on problem conditions
def numPoints : ℕ := 7
def totalTriangles := Nat.choose numPoints 3
def collinearCases : ℕ := 3

-- Proof problem
theorem hexagon_triangle_count : totalTriangles - collinearCases = 32 :=
by
  -- Calculation is expected here
  sorry

end hexagon_triangle_count_l281_281465


namespace P_subset_Q_l281_281488

def P : Set ℝ := {m | -1 < m ∧ m < 0}

def Q : Set ℝ := {m | ∀ x : ℝ, m * x^2 + 4 * m * x - 4 < 0}

theorem P_subset_Q : P ⊂ Q :=
by
  sorry

end P_subset_Q_l281_281488


namespace derivative_at_1_l281_281791

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x

theorem derivative_at_1 : (deriv f 1) = 2 * Real.exp 1 := by
  sorry

end derivative_at_1_l281_281791


namespace find_y_l281_281664

theorem find_y (x y : ℝ) 
  (h1 : 2 * x^2 + 6 * x + 4 * y + 2 = 0)
  (h2 : 3 * x + y + 4 = 0) :
  y^2 + 17 * y - 11 = 0 :=
by 
  sorry

end find_y_l281_281664


namespace total_number_of_applications_l281_281023

def in_state_apps := 200
def out_state_apps := 2 * in_state_apps
def total_apps := in_state_apps + out_state_apps

theorem total_number_of_applications : total_apps = 600 := by
  sorry

end total_number_of_applications_l281_281023


namespace sequence_non_positive_l281_281042

theorem sequence_non_positive (n : ℕ) (a : ℕ → ℝ) 
  (h₀ : a 0 = 0) 
  (hₙ : a n = 0)
  (h : ∀ k, 1 ≤ k ∧ k < n → a (k - 1) - 2 * a k + a (k + 1) ≥ 0) : 
  ∀ k, k ≤ n → a k ≤ 0 :=
by
  sorry

end sequence_non_positive_l281_281042


namespace ellipse_distance_pf2_l281_281158

noncomputable def ellipse_focal_length := 2 * Real.sqrt 2
noncomputable def ellipse_equation (a : ℝ) (a_gt_one : a > 1)
  (P : ℝ × ℝ) : Prop :=
  let x := P.1
  let y := P.2
  (x^2 / a) + y^2 = 1

theorem ellipse_distance_pf2
  (a : ℝ) (a_gt_one : a > 1)
  (focus_distance : 2 * Real.sqrt (a - 1) = 2 * Real.sqrt 2)
  (F1 F2 P : ℝ × ℝ)
  (on_ellipse : ellipse_equation a a_gt_one P)
  (PF1_eq_two : dist P F1 = 2)
  (a_eq : a = 3) :
  dist P F2 = 2 * Real.sqrt 3 - 2 := 
sorry

end ellipse_distance_pf2_l281_281158


namespace clock_angle_9_30_l281_281019

theorem clock_angle_9_30 : 
  let hour_hand_pos := 9.5 
  let minute_hand_pos := 6 
  let degrees_per_division := 30 
  let divisions_apart := hour_hand_pos - minute_hand_pos
  let angle := divisions_apart * degrees_per_division
  angle = 105 :=
by
  sorry

end clock_angle_9_30_l281_281019


namespace people_in_rooms_l281_281890

theorem people_in_rooms (x y : ℕ) (h1 : x + y = 76) (h2 : x - 30 = y - 40) : x = 33 ∧ y = 43 := by
  sorry

end people_in_rooms_l281_281890


namespace time_to_fill_tank_l281_281914

-- Definitions of conditions
def fill_by_pipe (time rate : ℚ) : ℚ := time * rate

def pipe_A_rate : ℚ := 1 / 60
def pipe_B_rate : ℚ := 1 / 40
def combined_rate : ℚ := pipe_A_rate + pipe_B_rate

-- Question to be proved
theorem time_to_fill_tank : 
    ∃ T : ℚ, (fill_by_pipe (T / 2) pipe_B_rate + fill_by_pipe (T / 2) combined_rate = 1) ∧ T = 30 :=
by 
    use 30
    sorry

end time_to_fill_tank_l281_281914


namespace correct_operation_l281_281604

theorem correct_operation (x y : ℝ) : (-x - y) ^ 2 = x ^ 2 + 2 * x * y + y ^ 2 :=
sorry

end correct_operation_l281_281604


namespace bellas_goal_product_l281_281377

theorem bellas_goal_product (g1 g2 g3 g4 g5 g6 : ℕ) (g7 g8 : ℕ) 
  (h1 : g1 = 5) 
  (h2 : g2 = 3) 
  (h3 : g3 = 2) 
  (h4 : g4 = 4)
  (h5 : g5 = 1) 
  (h6 : g6 = 6)
  (h7 : g7 < 10)
  (h8 : (g1 + g2 + g3 + g4 + g5 + g6 + g7) % 7 = 0) 
  (h9 : g8 < 10)
  (h10 : (g1 + g2 + g3 + g4 + g5 + g6 + g7 + g8) % 8 = 0) :
  g7 * g8 = 28 :=
by 
  sorry

end bellas_goal_product_l281_281377


namespace lunch_break_duration_l281_281892

/-- Define the total recess time as a sum of two 15-minute breaks and one 20-minute break. -/
def total_recess_time : ℕ := 15 + 15 + 20

/-- Define the total time spent outside of class. -/
def total_outside_class_time : ℕ := 80

/-- Prove that the lunch break is 30 minutes long. -/
theorem lunch_break_duration : total_outside_class_time - total_recess_time = 30 :=
by
  sorry

end lunch_break_duration_l281_281892


namespace find_other_number_product_find_third_number_sum_l281_281141

-- First Question
theorem find_other_number_product (x : ℚ) (h : x * (1/7 : ℚ) = -2) : x = -14 :=
sorry

-- Second Question
theorem find_third_number_sum (y : ℚ) (h : (1 : ℚ) + (-4) + y = -5) : y = -2 :=
sorry

end find_other_number_product_find_third_number_sum_l281_281141


namespace parking_garage_floors_l281_281065

theorem parking_garage_floors 
  (total_time : ℕ)
  (time_per_floor : ℕ)
  (gate_time : ℕ)
  (every_n_floors : ℕ) 
  (F : ℕ) 
  (h1 : total_time = 1440)
  (h2 : time_per_floor = 80)
  (h3 : gate_time = 120)
  (h4 : every_n_floors = 3)
  :
  F = 13 :=
by
  have total_id_time : ℕ := gate_time * ((F - 1) / every_n_floors)
  have total_drive_time : ℕ := time_per_floor * (F - 1)
  have total_time_calc : ℕ := total_drive_time + total_id_time
  have h5 := total_time_calc = total_time
  -- Now we simplify the algebraic equation given the problem conditions
  sorry

end parking_garage_floors_l281_281065


namespace plane_ticket_price_l281_281275

theorem plane_ticket_price :
  ∀ (P : ℕ),
  (20 * 155) + 2900 = 30 * P →
  P = 200 := 
by
  sorry

end plane_ticket_price_l281_281275


namespace age_discrepancy_l281_281870

theorem age_discrepancy (R G M F A : ℕ)
  (hR : R = 12)
  (hG : G = 7 * R)
  (hM : M = G / 2)
  (hF : F = M + 5)
  (hA : A = G - 8)
  (hDiff : A - F = 10) :
  false :=
by
  -- proofs and calculations leading to contradiction go here
  sorry

end age_discrepancy_l281_281870


namespace smallest_sector_angle_l281_281563

-- Definitions and conditions identified in step a.

def a1 (d : ℕ) : ℕ := (48 - 14 * d) / 2

-- Proof statement
theorem smallest_sector_angle : ∀ d : ℕ, d ≥ 0 → d ≤ 3 → 15 * (a1 d + (a1 d + 14 * d)) = 720 → (a1 d = 3) :=
by
  sorry

end smallest_sector_angle_l281_281563


namespace vertex_of_quadratic_function_l281_281582

theorem vertex_of_quadratic_function :
  ∀ x: ℝ, (2 - (x + 1)^2) = 2 - (x + 1)^2 → (∃ h k : ℝ, (h, k) = (-1, 2) ∧ ∀ x: ℝ, (2 - (x + 1)^2) = k - (x - h)^2) :=
by
  sorry

end vertex_of_quadratic_function_l281_281582


namespace tamia_total_slices_and_pieces_l281_281250

-- Define the conditions
def num_bell_peppers : ℕ := 5
def slices_per_pepper : ℕ := 20
def num_large_slices : ℕ := num_bell_peppers * slices_per_pepper
def num_half_slices : ℕ := num_large_slices / 2
def small_pieces_per_slice : ℕ := 3
def num_small_pieces : ℕ := num_half_slices * small_pieces_per_slice
def num_uncut_slices : ℕ := num_half_slices

-- Define the total number of pieces and slices
def total_pieces_and_slices : ℕ := num_uncut_slices + num_small_pieces

-- State the theorem and provide a placeholder for the proof
theorem tamia_total_slices_and_pieces : total_pieces_and_slices = 200 :=
by
  sorry

end tamia_total_slices_and_pieces_l281_281250


namespace meaningful_expression_range_l281_281374

theorem meaningful_expression_range (x : ℝ) : (∃ y, y = 1 / (x - 4)) ↔ x ≠ 4 := 
by
  sorry

end meaningful_expression_range_l281_281374


namespace time_to_fill_tank_l281_281915

-- Definitions of conditions
def fill_by_pipe (time rate : ℚ) : ℚ := time * rate

def pipe_A_rate : ℚ := 1 / 60
def pipe_B_rate : ℚ := 1 / 40
def combined_rate : ℚ := pipe_A_rate + pipe_B_rate

-- Question to be proved
theorem time_to_fill_tank : 
    ∃ T : ℚ, (fill_by_pipe (T / 2) pipe_B_rate + fill_by_pipe (T / 2) combined_rate = 1) ∧ T = 30 :=
by 
    use 30
    sorry

end time_to_fill_tank_l281_281915


namespace area_diff_of_rectangle_l281_281455

theorem area_diff_of_rectangle (a : ℝ) : 
  let length_increased := 1.40 * a
  let breadth_increased := 1.30 * a
  let original_area := a * a
  let new_area := length_increased * breadth_increased
  (new_area - original_area) = 0.82 * (a * a) :=
by 
sorry

end area_diff_of_rectangle_l281_281455


namespace triangle_is_isosceles_l281_281678

theorem triangle_is_isosceles (α β γ δ ε : ℝ)
  (h1 : α + β + γ = 180) 
  (h2 : α + β = δ) 
  (h3 : β + γ = ε) : 
  α = γ ∨ β = γ ∨ α = β := 
sorry

end triangle_is_isosceles_l281_281678


namespace avg_height_and_variance_correct_l281_281057

noncomputable def avg_height_and_variance
  (total_students : ℕ)
  (boys : ℕ)
  (girls : ℕ)
  (boys_avg_height : ℕ)
  (boys_variance : ℕ)
  (girls_avg_height : ℕ)
  (girls_variance : ℕ) : (ℕ × ℕ) := 
  let total_students := 300
  let boys := 180
  let girls := 120
  let boys_avg_height := 170
  let boys_variance := 14
  let girls_avg_height := 160
  let girls_variance := 24
  let avg_height := (boys * boys_avg_height + girls * girls_avg_height) / total_students 
  let variance := (boys * (boys_variance + (boys_avg_height - avg_height) ^ 2) 
                    + girls * (girls_variance + (girls_avg_height - avg_height) ^ 2)) / total_students
  (avg_height, variance)

theorem avg_height_and_variance_correct:
   avg_height_and_variance 300 180 120 170 14 160 24 = (166, 42) := 
  by {
    sorry
  }

end avg_height_and_variance_correct_l281_281057


namespace chef_earns_less_than_manager_l281_281747

noncomputable def manager_wage : ℚ := 8.50
noncomputable def dishwasher_wage : ℚ := manager_wage / 2
noncomputable def chef_wage : ℚ := dishwasher_wage + 0.22 * dishwasher_wage

theorem chef_earns_less_than_manager :
  manager_wage - chef_wage = 3.315 := by
  sorry

end chef_earns_less_than_manager_l281_281747


namespace nearest_integer_to_a_plus_b_l281_281364

theorem nearest_integer_to_a_plus_b
  (a b : ℝ)
  (h1 : |a| + b = 5)
  (h2 : |a| * b + a^3 = -8) :
  abs (a + b - 3) ≤ 0.5 :=
sorry

end nearest_integer_to_a_plus_b_l281_281364


namespace number_of_ways_to_select_starting_lineup_l281_281296

noncomputable def choose (n k : ℕ) : ℕ := 
if h : k ≤ n then Nat.choose n k else 0

theorem number_of_ways_to_select_starting_lineup (n k : ℕ) (h : n = 12) (h1 : k = 5) : 
  12 * choose 11 4 = 3960 := 
by sorry

end number_of_ways_to_select_starting_lineup_l281_281296


namespace probability_at_least_5_consecutive_heads_l281_281616

theorem probability_at_least_5_consecutive_heads (flips : Fin 256) :
  let successful_outcomes := 13
  in let total_outcomes := 256
  in (successful_outcomes.to_rat / total_outcomes.to_rat) = (13 : ℚ) / 256 :=
sorry

end probability_at_least_5_consecutive_heads_l281_281616


namespace perimeter_of_region_proof_l281_281082

noncomputable def perimeter_of_region (total_area : ℕ) (num_squares : ℕ) (arrangement : String) : ℕ :=
  if total_area = 512 ∧ num_squares = 8 ∧ arrangement = "vertical rectangle" then 160 else 0

theorem perimeter_of_region_proof :
  perimeter_of_region 512 8 "vertical rectangle" = 160 :=
by
  sorry

end perimeter_of_region_proof_l281_281082


namespace fruit_problem_l281_281456

def number_of_pears (A : ℤ) : ℤ := (3 * A) / 5
def number_of_apples (B : ℤ) : ℤ := (3 * B) / 7

theorem fruit_problem
  (A B : ℤ)
  (h1 : A + B = 82)
  (h2 : abs (A - B) < 10)
  (x : ℤ := (2 * A) / 5)
  (y : ℤ := (4 * B) / 7) :
  number_of_pears A = 24 ∧ number_of_apples B = 18 :=
by
  sorry

end fruit_problem_l281_281456


namespace correct_email_sequence_l281_281694

theorem correct_email_sequence :
  let a := "Open the mailbox"
  let b := "Enter the recipient's address"
  let c := "Enter the subject"
  let d := "Enter the content of the email"
  let e := "Click 'Compose'"
  let f := "Click 'Send'"
  (a, e, b, c, d, f) = ("Open the mailbox", "Click 'Compose'", "Enter the recipient's address", "Enter the subject", "Enter the content of the email", "Click 'Send'") := 
sorry

end correct_email_sequence_l281_281694


namespace total_cost_is_correct_l281_281712

-- Define the costs as constants
def marbles_cost : ℝ := 9.05
def football_cost : ℝ := 4.95
def baseball_cost : ℝ := 6.52

-- Assert that the total cost is correct
theorem total_cost_is_correct : marbles_cost + football_cost + baseball_cost = 20.52 :=
by sorry

end total_cost_is_correct_l281_281712


namespace money_distribution_l281_281135

theorem money_distribution :
  ∀ (A B C : ℕ), 
  A + B + C = 900 → 
  B + C = 750 → 
  C = 250 → 
  A + C = 400 := 
by
  intros A B C h1 h2 h3
  sorry

end money_distribution_l281_281135


namespace max_balls_in_cube_l281_281115

theorem max_balls_in_cube 
  (radius : ℝ) (side_length : ℝ) 
  (ball_volume : ℝ := (4 / 3) * Real.pi * (radius^3)) 
  (cube_volume : ℝ := side_length^3) 
  (max_balls : ℝ := cube_volume / ball_volume) :
  radius = 3 ∧ side_length = 8 → Int.floor max_balls = 4 := 
by
  intro h
  rw [h.left, h.right]
  -- further proof would use numerical evaluation
  sorry

end max_balls_in_cube_l281_281115


namespace problem_solution_l281_281822

variable {A B : Set ℝ}

def definition_A : Set ℝ := {x | 0 < x ∧ x ≤ 2}
def definition_B : Set ℝ := {x | x ≤ 3}

theorem problem_solution : definition_A ∩ definition_B = definition_A :=
by
  sorry

end problem_solution_l281_281822


namespace weekly_crab_meat_cost_l281_281982

-- Declare conditions as definitions
def dishes_per_day : ℕ := 40
def pounds_per_dish : ℝ := 1.5
def cost_per_pound : ℝ := 8
def closed_days_per_week : ℕ := 3
def days_per_week : ℕ := 7

-- Define the Lean statement to prove the weekly cost
theorem weekly_crab_meat_cost :
  let days_open_per_week := days_per_week - closed_days_per_week
  let pounds_per_day := dishes_per_day * pounds_per_dish
  let daily_cost := pounds_per_day * cost_per_pound
  let weekly_cost := daily_cost * (days_open_per_week : ℝ)
  weekly_cost = 1920 :=
by
  sorry

end weekly_crab_meat_cost_l281_281982


namespace calories_per_cookie_l281_281934

theorem calories_per_cookie (C : ℝ) (h1 : ∀ cracker, cracker = 15)
    (h2 : ∀ cookie, cookie = C)
    (h3 : 7 * C + 10 * 15 = 500) :
    C = 50 :=
  by
    sorry

end calories_per_cookie_l281_281934


namespace simplify_and_evaluate_l281_281722

variable (a : ℕ)

theorem simplify_and_evaluate :
  (a^2 - 4) / a^2 / (1 - 2 / a) = 7 / 5 :=
by
  -- Assign the condition
  let a := 5
  sorry -- skip the proof

end simplify_and_evaluate_l281_281722


namespace solve_quadratic_eq_l281_281571

theorem solve_quadratic_eq (x : ℝ) : x^2 + 2 * x - 1 = 0 ↔ (x = -1 + Real.sqrt 2 ∨ x = -1 - Real.sqrt 2) :=
by
  sorry

end solve_quadratic_eq_l281_281571


namespace chef_made_10_cakes_l281_281581

-- Definitions based on the conditions
def total_eggs : ℕ := 60
def eggs_in_fridge : ℕ := 10
def eggs_per_cake : ℕ := 5

-- Calculated values based on the definitions
def eggs_for_cakes : ℕ := total_eggs - eggs_in_fridge
def number_of_cakes : ℕ := eggs_for_cakes / eggs_per_cake

-- Theorem to prove
theorem chef_made_10_cakes : number_of_cakes = 10 := by
  sorry

end chef_made_10_cakes_l281_281581


namespace find_x_for_sin_cos_l281_281495

theorem find_x_for_sin_cos (x : ℝ) (h1 : 0 ≤ x) (h2 : x < 2 * Real.pi) (h3 : Real.sin x + Real.cos x = Real.sqrt 2) : x = Real.pi / 4 :=
sorry

end find_x_for_sin_cos_l281_281495


namespace student_correct_answers_l281_281605

variable (C I : ℕ) -- Define C and I as natural numbers
variable (score totalQuestions : ℕ) -- Define score and totalQuestions as natural numbers

-- Define the conditions
def grading_system (C I score : ℕ) : Prop := C - 2 * I = score
def total_questions (C I totalQuestions : ℕ) : Prop := C + I = totalQuestions

-- The theorem statement to prove
theorem student_correct_answers :
  (grading_system C I 76) ∧ (total_questions C I 100) → C = 92 := by
  sorry -- Proof to be filled in

end student_correct_answers_l281_281605


namespace greatest_x_solution_l281_281941

theorem greatest_x_solution (x : ℝ) (h₁ : (x^2 - x - 30) / (x - 6) = 2 / (x + 4)) : x ≤ -3 :=
sorry

end greatest_x_solution_l281_281941


namespace probability_at_least_one_2_on_8_sided_dice_l281_281111

theorem probability_at_least_one_2_on_8_sided_dice :
  (∃ (d1 d2 : Fin 8), d1 = 1 ∨ d2 = 1) → (15 / 64) = (15 / 64) := by
  intro h
  sorry

end probability_at_least_one_2_on_8_sided_dice_l281_281111


namespace range_of_m_l281_281985

noncomputable def abs_sum (x : ℝ) : ℝ := |x - 5| + |x - 3|

theorem range_of_m (m : ℝ) : (∃ x : ℝ, abs_sum x < m) ↔ m > 2 := 
by 
  sorry

end range_of_m_l281_281985


namespace comparing_exponents_l281_281557

theorem comparing_exponents {a b : ℝ} (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : Real.exp a + 2 * a = Real.exp b + 3 * b) : a > b :=
sorry

end comparing_exponents_l281_281557


namespace avg_daily_production_l281_281007

theorem avg_daily_production (x y : ℕ) (h1 : x + y = 350) (h2 : 2 * x - y = 250) : x = 200 ∧ y = 150 := 
by
  sorry

end avg_daily_production_l281_281007


namespace gopi_turbans_annual_salary_l281_281351

variable (T : ℕ) (annual_salary_turbans : ℕ)
variable (annual_salary_money : ℕ := 90)
variable (months_worked : ℕ := 9)
variable (total_months_in_year : ℕ := 12)
variable (received_money : ℕ := 55)
variable (turban_price : ℕ := 50)
variable (received_turbans : ℕ := 1)
variable (servant_share_fraction : ℚ := 3 / 4)

theorem gopi_turbans_annual_salary 
    (annual_salary_turbans : ℕ)
    (H : (servant_share_fraction * (annual_salary_money + turban_price * annual_salary_turbans) = received_money + turban_price * received_turbans))
    : annual_salary_turbans = 1 :=
sorry

end gopi_turbans_annual_salary_l281_281351


namespace factorization_of_polynomial_l281_281652

theorem factorization_of_polynomial : 
  (x^2 + 6 * x + 9 - 64 * x^4) = (-8 * x^2 + x + 3) * (8 * x^2 + x + 3) :=
by sorry

end factorization_of_polynomial_l281_281652


namespace cards_probability_comparison_l281_281059

noncomputable def probability_case_a : ℚ :=
  (Nat.choose 13 10) * (Nat.choose 39 3) / Nat.choose 52 13

noncomputable def probability_case_b : ℚ :=
  4 ^ 13 / Nat.choose 52 13

theorem cards_probability_comparison :
  probability_case_b > probability_case_a :=
  sorry

end cards_probability_comparison_l281_281059


namespace book_pages_l281_281401

-- Define the number of pages Sally reads on weekdays and weekends
def pages_on_weekdays : ℕ := 10
def pages_on_weekends : ℕ := 20

-- Define the number of weekdays and weekends in 2 weeks
def weekdays_in_two_weeks : ℕ := 5 * 2
def weekends_in_two_weeks : ℕ := 2 * 2

-- Total number of pages read in 2 weeks
def total_pages_read (pages_on_weekdays : ℕ) (pages_on_weekends : ℕ) (weekdays_in_two_weeks : ℕ) (weekends_in_two_weeks : ℕ) : ℕ :=
  (pages_on_weekdays * weekdays_in_two_weeks) + (pages_on_weekends * weekends_in_two_weeks)

-- Prove the number of pages in the book
theorem book_pages : total_pages_read 10 20 10 4 = 180 := by
  sorry

end book_pages_l281_281401


namespace total_job_applications_l281_281026

theorem total_job_applications (apps_in_state : ℕ) (apps_other_states : ℕ) 
  (h1 : apps_in_state = 200)
  (h2 : apps_other_states = 2 * apps_in_state) :
  apps_in_state + apps_other_states = 600 :=
by
  sorry

end total_job_applications_l281_281026


namespace ac_bd_sum_l281_281837

theorem ac_bd_sum (a b c d : ℝ) (h1 : a + b + c = 6) (h2 : a + b + d = -3) (h3 : a + c + d = 0) (h4 : b + c + d = -9) : 
  a * c + b * d = 23 := 
sorry

end ac_bd_sum_l281_281837


namespace original_population_l281_281614

theorem original_population (p: ℝ) :
  (p + 1500) * 0.85 = p - 45 -> p = 8800 :=
by
  sorry

end original_population_l281_281614


namespace smallest_non_factor_product_l281_281425

open Nat

def is_factor (n d : ℕ) := d > 0 ∧ n % d = 0

theorem smallest_non_factor_product (a b : ℕ) (h1 : a ≠ b) (h2 : is_factor 48 a) (h3 : is_factor 48 b) (h4 : ¬ is_factor 48 (a * b)) : a * b = 18 :=
by
  sorry

end smallest_non_factor_product_l281_281425


namespace smallest_real_number_l281_281639

theorem smallest_real_number (A B C D : ℝ) 
  (hA : A = |(-2 : ℝ)|) 
  (hB : B = -1) 
  (hC : C = 0) 
  (hD : D = -1 / 2) : 
  min A (min B (min C D)) = B := 
by
  sorry

end smallest_real_number_l281_281639


namespace correct_equation_l281_281904

def initial_investment : ℝ := 2500
def expected_investment : ℝ := 6600
def growth_rate (x : ℝ) : ℝ := x

theorem correct_equation (x : ℝ) : 
  initial_investment * (1 + growth_rate x) + initial_investment * (1 + growth_rate x)^2 = expected_investment :=
by
  sorry

end correct_equation_l281_281904


namespace smallest_possible_value_l281_281802

theorem smallest_possible_value (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (hxy : x ≠ y) (h : (1 / (x : ℝ)) + (1 / (y : ℝ)) = 1 / 15) : x + y = 64 :=
sorry

end smallest_possible_value_l281_281802


namespace brittany_age_when_returning_l281_281315

def rebecca_age : ℕ := 25
def age_difference : ℕ := 3
def vacation_duration : ℕ := 4

theorem brittany_age_when_returning : (rebecca_age + age_difference + vacation_duration) = 32 := by
  sorry

end brittany_age_when_returning_l281_281315


namespace select_one_person_for_both_days_l281_281785

noncomputable def combination (n r : ℕ) := n.choose r

def volunteers := 5
def serve_both_days := combination volunteers 1
def remaining_for_saturday := volunteers - 1
def serve_saturday := combination remaining_for_saturday 1
def remaining_for_sunday := remaining_for_saturday - 1
def serve_sunday := combination remaining_for_sunday 1
def total_ways := serve_both_days * serve_saturday * serve_sunday

theorem select_one_person_for_both_days :
  total_ways = 60 := 
by
  -- We skip the proof details for now
  sorry

end select_one_person_for_both_days_l281_281785


namespace alice_bob_sum_proof_l281_281936

noncomputable def alice_bob_sum_is_22 : Prop :=
  ∃ A B : ℕ, (1 ≤ A ∧ A ≤ 50) ∧ (1 ≤ B ∧ B ≤ 50) ∧ (B % 3 = 0) ∧ (∃ k : ℕ, 2 * B + A = k^2) ∧ (A + B = 22)

theorem alice_bob_sum_proof : alice_bob_sum_is_22 :=
sorry

end alice_bob_sum_proof_l281_281936


namespace find_A_l281_281097

theorem find_A : ∃ A : ℕ, 691 - (600 + A * 10 + 7) = 4 ∧ A = 8 := by
  sorry

end find_A_l281_281097


namespace find_positive_integer_x_l281_281707

def positive_integer (x : ℕ) : Prop :=
  x > 0

def n (x : ℕ) : ℕ :=
  x^2 + 3 * x + 20

def d (x : ℕ) : ℕ :=
  3 * x + 4

def division_property (x : ℕ) : Prop :=
  ∃ q r : ℕ, q = x ∧ r = 8 ∧ n x = q * d x + r

theorem find_positive_integer_x :
  ∃ x : ℕ, positive_integer x ∧ n x = x * d x + 8 :=
sorry

end find_positive_integer_x_l281_281707


namespace bella_age_is_five_l281_281924

-- Definitions from the problem:
def is_age_relation (bella_age brother_age : ℕ) : Prop :=
  brother_age = bella_age + 9 ∧ bella_age + brother_age = 19

-- The main proof statement:
theorem bella_age_is_five (bella_age brother_age : ℕ) (h : is_age_relation bella_age brother_age) :
  bella_age = 5 :=
by {
  -- Placeholder for proof steps
  sorry
}

end bella_age_is_five_l281_281924


namespace tamia_total_slices_and_pieces_l281_281248

-- Define the conditions
def num_bell_peppers : ℕ := 5
def slices_per_pepper : ℕ := 20
def num_large_slices : ℕ := num_bell_peppers * slices_per_pepper
def num_half_slices : ℕ := num_large_slices / 2
def small_pieces_per_slice : ℕ := 3
def num_small_pieces : ℕ := num_half_slices * small_pieces_per_slice
def num_uncut_slices : ℕ := num_half_slices

-- Define the total number of pieces and slices
def total_pieces_and_slices : ℕ := num_uncut_slices + num_small_pieces

-- State the theorem and provide a placeholder for the proof
theorem tamia_total_slices_and_pieces : total_pieces_and_slices = 200 :=
by
  sorry

end tamia_total_slices_and_pieces_l281_281248


namespace area_difference_quarter_circles_l281_281765

theorem area_difference_quarter_circles :
  let r1 := 28
  let r2 := 14
  let pi := (22 / 7)
  let quarter_area_big := (1 / 4) * pi * r1^2
  let quarter_area_small := (1 / 4) * pi * r2^2
  let rectangle_area := r1 * r2
  (quarter_area_big - (quarter_area_small + rectangle_area)) = 70 := by
  -- Placeholder for the proof
  sorry

end area_difference_quarter_circles_l281_281765


namespace identical_remainders_l281_281239

theorem identical_remainders (a : Fin 11 → Fin 11) (h_perm : ∀ n, ∃ m, a m = n) :
  ∃ (i j : Fin 11), i ≠ j ∧ (i * a i) % 11 = (j * a j) % 11 :=
by 
  sorry

end identical_remainders_l281_281239


namespace smallest_prime_divisor_of_sum_of_powers_l281_281282

theorem smallest_prime_divisor_of_sum_of_powers :
  let a := 5
  let b := 7
  let n := 23
  let m := 17
  Nat.minFac (a^n + b^m) = 2 := by
  sorry

end smallest_prime_divisor_of_sum_of_powers_l281_281282


namespace people_with_fewer_than_seven_cards_l281_281966

theorem people_with_fewer_than_seven_cards (total_cards : ℕ) (num_people : ℕ) (cards_per_person : ℕ) (extra_cards : ℕ)
  (h1 : total_cards = 52) (h2 : num_people = 8) (h3 : total_cards = num_people * cards_per_person + extra_cards) (h4 : extra_cards < num_people) :
  ∃ fewer_than_seven : ℕ, num_people - extra_cards = fewer_than_seven :=
by
  have remainder := (52 % 8)
  have cards_per_person := (52 / 8)
  have number_fewer_than_seven := num_people - remainder
  existsi number_fewer_than_seven
  sorry

end people_with_fewer_than_seven_cards_l281_281966


namespace find_k_solve_quadratic_l281_281673

-- Define the conditions
variables (x1 x2 k : ℝ)

-- Given conditions
def quadratic_roots : Prop :=
  x1 + x2 = 6 ∧ x1 * x2 = k

def condition_A (x1 x2 : ℝ) : Prop :=
  x1^2 * x2^2 - x1 - x2 = 115

-- Prove that k = -11 given the conditions
theorem find_k (h1: quadratic_roots x1 x2 k) (h2 : condition_A x1 x2) : k = -11 :=
  sorry

-- Prove the roots of the quadratic equation when k = -11
theorem solve_quadratic (h1 : quadratic_roots x1 x2 (-11)) : 
  x1 = 3 + 2 * Real.sqrt 5 ∧ x2 = 3 - 2 * Real.sqrt 5 ∨ 
  x1 = 3 - 2 * Real.sqrt 5 ∧ x2 = 3 + 2 * Real.sqrt 5 :=
  sorry

end find_k_solve_quadratic_l281_281673


namespace greatest_x_solution_l281_281940

theorem greatest_x_solution (x : ℝ) (h₁ : (x^2 - x - 30) / (x - 6) = 2 / (x + 4)) : x ≤ -3 :=
sorry

end greatest_x_solution_l281_281940


namespace leak_empties_in_24_hours_l281_281744

noncomputable def tap_rate := 1 / 6
noncomputable def combined_rate := 1 / 8
noncomputable def leak_rate := tap_rate - combined_rate
noncomputable def time_to_empty := 1 / leak_rate

theorem leak_empties_in_24_hours :
  time_to_empty = 24 := by
  sorry

end leak_empties_in_24_hours_l281_281744


namespace buns_per_student_correct_l281_281354

variables (packages_per_bun : Nat) (num_packages : Nat)
           (num_classes : Nat) (students_per_class : Nat)

def total_buns (packages_per_bun : Nat) (num_packages : Nat) : Nat :=
  packages_per_bun * num_packages

def total_students (num_classes : Nat) (students_per_class : Nat) : Nat :=
  num_classes * students_per_class

def buns_per_student (total_buns : Nat) (total_students : Nat) : Nat :=
  total_buns / total_students

theorem buns_per_student_correct :
  packages_per_bun = 8 →
  num_packages = 30 →
  num_classes = 4 →
  students_per_class = 30 →
  buns_per_student (total_buns packages_per_bun num_packages) 
                  (total_students num_classes students_per_class) = 2 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end buns_per_student_correct_l281_281354


namespace minimum_flowers_to_guarantee_bouquets_l281_281217

theorem minimum_flowers_to_guarantee_bouquets :
  (∀ (num_types : ℕ) (flowers_per_bouquet : ℕ) (num_bouquets : ℕ),
   num_types = 6 → flowers_per_bouquet = 5 → num_bouquets = 10 →
   ∃ min_flowers : ℕ, min_flowers = 70 ∧
   ∀ (picked_flowers : ℕ → ℕ), 
     (∀ t : ℕ, t < num_types → picked_flowers t ≥ 0 ∧ 
                (t < num_types - 1 → picked_flowers t ≤ flowers_per_bouquet * (num_bouquets - 1) + 4)) → 
     ∑ t in finset.range num_types, picked_flowers t = min_flowers → 
     ∑ t in finset.range num_types, (picked_flowers t / flowers_per_bouquet) ≥ num_bouquets) := 
by {
  intro num_types flowers_per_bouquet num_bouquets,
  intro h1 h2 h3,
  use 70,
  split,
  {
    exact rfl,
  },
  {
    intros picked_flowers h_picked,
    sorry,
  }
}

end minimum_flowers_to_guarantee_bouquets_l281_281217


namespace angle_bisector_slope_l281_281931

theorem angle_bisector_slope :
  let m₁ := 2
  let m₂ := 5
  let k := (7 - 2 * Real.sqrt 5) / 11
  True :=
by admit

end angle_bisector_slope_l281_281931


namespace sin_transformation_identity_l281_281506

theorem sin_transformation_identity 
  (θ : ℝ) 
  (h : Real.cos (π / 12 - θ) = 1 / 3) : 
  Real.sin (2 * θ + π / 3) = -7 / 9 := 
by 
  sorry

end sin_transformation_identity_l281_281506


namespace count_multiples_of_14_between_100_and_400_l281_281960

theorem count_multiples_of_14_between_100_and_400 : 
  ∃ n : ℕ, n = 21 ∧ (∀ k : ℕ, (100 ≤ k ∧ k ≤ 400 ∧ 14 ∣ k) ↔ (∃ i : ℕ, k = 14 * i ∧ 8 ≤ i ∧ i ≤ 28)) :=
sorry

end count_multiples_of_14_between_100_and_400_l281_281960


namespace eval_expression_at_neg3_l281_281781

def evaluate_expression (x : ℤ) : ℚ :=
  (5 + x * (5 + x) - 4 ^ 2 : ℤ) / (x - 4 + x ^ 3 : ℤ)

theorem eval_expression_at_neg3 :
  evaluate_expression (-3) = -17 / 20 := by
  sorry

end eval_expression_at_neg3_l281_281781


namespace find_number_l281_281659

theorem find_number (x : ℝ) : 2.75 + 0.003 + x = 2.911 -> x = 0.158 := 
by
  intros h
  sorry

end find_number_l281_281659
