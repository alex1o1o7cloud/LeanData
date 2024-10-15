import Mathlib

namespace NUMINAMATH_GPT_base_length_of_parallelogram_l955_95533

theorem base_length_of_parallelogram (A h : ℝ) (hA : A = 44) (hh : h = 11) :
  ∃ b : ℝ, b = 4 ∧ A = b * h :=
by
  sorry

end NUMINAMATH_GPT_base_length_of_parallelogram_l955_95533


namespace NUMINAMATH_GPT_no_integer_solutions_for_2891_l955_95580

theorem no_integer_solutions_for_2891 (x y : ℤ) : ¬ (x^3 - 3 * x * y^2 + y^3 = 2891) :=
sorry

end NUMINAMATH_GPT_no_integer_solutions_for_2891_l955_95580


namespace NUMINAMATH_GPT_ray_has_4_nickels_left_l955_95507

theorem ray_has_4_nickels_left (initial_cents : ℕ) (given_to_peter : ℕ)
    (given_to_randi : ℕ) (value_of_nickel : ℕ) (remaining_cents : ℕ) 
    (remaining_nickels : ℕ) :
    initial_cents = 95 →
    given_to_peter = 25 →
    given_to_randi = 2 * given_to_peter →
    value_of_nickel = 5 →
    remaining_cents = initial_cents - given_to_peter - given_to_randi →
    remaining_nickels = remaining_cents / value_of_nickel →
    remaining_nickels = 4 :=
by
  intros
  sorry

end NUMINAMATH_GPT_ray_has_4_nickels_left_l955_95507


namespace NUMINAMATH_GPT_loop_execution_count_l955_95577

theorem loop_execution_count : 
  ∀ (a b : ℤ), a = 2 → b = 20 → (b - a + 1) = 19 :=
by
  intros a b ha hb
  rw [ha, hb]
  -- Here, we explicitly compute (20 - 2 + 1) = 19
  exact rfl

end NUMINAMATH_GPT_loop_execution_count_l955_95577


namespace NUMINAMATH_GPT_perimeter_triangle_l955_95548

-- Definitions and conditions
def side1 : ℕ := 2
def side2 : ℕ := 5
def is_odd (n : ℕ) : Prop := n % 2 = 1
def valid_third_side (x : ℕ) : Prop := 3 < x ∧ x < 7 ∧ is_odd x

-- Theorem statement
theorem perimeter_triangle : ∃ (x : ℕ), valid_third_side x ∧ (side1 + side2 + x = 12) :=
by 
  sorry

end NUMINAMATH_GPT_perimeter_triangle_l955_95548


namespace NUMINAMATH_GPT_rebecca_less_than_toby_l955_95583

-- Define the conditions
variable (x : ℕ) -- Thomas worked x hours
variable (tobyHours : ℕ := 2 * x - 10) -- Toby worked 10 hours less than twice what Thomas worked
variable (rebeccaHours : ℕ := 56) -- Rebecca worked 56 hours

-- Define the total hours worked in one week
axiom total_hours_worked : x + tobyHours + rebeccaHours = 157

-- The proof goal
theorem rebecca_less_than_toby : tobyHours - rebeccaHours = 8 := 
by
  -- (proof steps would go here)
  sorry

end NUMINAMATH_GPT_rebecca_less_than_toby_l955_95583


namespace NUMINAMATH_GPT_binary_division_remainder_correct_l955_95518

-- Define the last two digits of the binary number
def b_1 : ℕ := 1
def b_0 : ℕ := 1

-- Define the function to calculate the remainder when dividing by 4
def binary_remainder (b1 b0 : ℕ) : ℕ := 2 * b1 + b0

-- Expected remainder in binary form
def remainder_in_binary : ℕ := 0b11  -- '11' in binary is 3 in decimal

-- The theorem to prove
theorem binary_division_remainder_correct :
  binary_remainder b_1 b_0 = remainder_in_binary :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_binary_division_remainder_correct_l955_95518


namespace NUMINAMATH_GPT_minimum_cost_l955_95517

noncomputable def f (x : ℝ) : ℝ := (1000 / (x + 5)) + 5 * x + (1 / 2) * (x^2 + 25)

theorem minimum_cost :
  (2 ≤ x ∧ x ≤ 8) →
  (f 5 = 150 ∧ (∀ y, 2 ≤ y ∧ y ≤ 8 → f y ≥ f 5)) :=
by
  intro h
  have f_exp : f x = (1000 / (x+5)) + 5*x + (1/2)*(x^2 + 25) := rfl
  sorry

end NUMINAMATH_GPT_minimum_cost_l955_95517


namespace NUMINAMATH_GPT_total_vehicles_in_lanes_l955_95511

theorem total_vehicles_in_lanes :
  ∀ (lanes : ℕ) (trucks_per_lane cars_total trucks_total : ℕ),
  lanes = 4 →
  trucks_per_lane = 60 →
  trucks_total = trucks_per_lane * lanes →
  cars_total = 2 * trucks_total →
  (trucks_total + cars_total) = 2160 :=
by intros lanes trucks_per_lane cars_total trucks_total hlanes htrucks_per_lane htrucks_total hcars_total
   -- sorry added to skip the proof
   sorry

end NUMINAMATH_GPT_total_vehicles_in_lanes_l955_95511


namespace NUMINAMATH_GPT_pass_in_both_subjects_l955_95569

variable (F_H F_E F_HE : ℝ)

theorem pass_in_both_subjects (h1 : F_H = 20) (h2 : F_E = 70) (h3 : F_HE = 10) :
  100 - ((F_H + F_E) - F_HE) = 20 :=
by
  sorry

end NUMINAMATH_GPT_pass_in_both_subjects_l955_95569


namespace NUMINAMATH_GPT_common_difference_l955_95546

variable (a : ℕ → ℝ)
variable (d : ℝ)

-- Conditions
axiom h1 : a 3 + a 7 = 10
axiom h2 : a 8 = 8

-- Statement to prove
theorem common_difference (h : ∀ n, a (n + 1) = a n + d) : d = 1 :=
  sorry

end NUMINAMATH_GPT_common_difference_l955_95546


namespace NUMINAMATH_GPT_quadratic_real_roots_l955_95586

theorem quadratic_real_roots (k : ℝ) : 
  (∀ x : ℝ, (2 * x^2 + 4 * x + k - 1 = 0) → ∃ x : ℝ, 2 * x^2 + 4 * x + k - 1 = 0) → 
  k ≤ 3 :=
by
  intro h
  have h_discriminant : 16 - 8 * k >= 0 := sorry
  linarith

end NUMINAMATH_GPT_quadratic_real_roots_l955_95586


namespace NUMINAMATH_GPT_penny_frogs_count_l955_95555

theorem penny_frogs_count :
  let tree_frogs := 55
  let poison_frogs := 10
  let wood_frogs := 13
  tree_frogs + poison_frogs + wood_frogs = 78 :=
by
  let tree_frogs := 55
  let poison_frogs := 10
  let wood_frogs := 13
  show tree_frogs + poison_frogs + wood_frogs = 78
  sorry

end NUMINAMATH_GPT_penny_frogs_count_l955_95555


namespace NUMINAMATH_GPT_sum_first_75_terms_arith_seq_l955_95514

theorem sum_first_75_terms_arith_seq (a_1 d : ℕ) (n : ℕ) (h_a1 : a_1 = 3) (h_d : d = 4) (h_n : n = 75) : 
  (n * (2 * a_1 + (n - 1) * d)) / 2 = 11325 := 
by
  subst h_a1
  subst h_d
  subst h_n
  sorry

end NUMINAMATH_GPT_sum_first_75_terms_arith_seq_l955_95514


namespace NUMINAMATH_GPT_table_runner_combined_area_l955_95573

theorem table_runner_combined_area
    (table_area : ℝ) (cover_percentage : ℝ) (area_two_layers : ℝ) (area_three_layers : ℝ) (A : ℝ) :
    table_area = 175 →
    cover_percentage = 0.8 →
    area_two_layers = 24 →
    area_three_layers = 28 →
    A = (cover_percentage * table_area - area_two_layers - area_three_layers) + area_two_layers + 2 * area_three_layers →
    A = 168 :=
by
  intros h_table_area h_cover_percentage h_area_two_layers h_area_three_layers h_A
  sorry

end NUMINAMATH_GPT_table_runner_combined_area_l955_95573


namespace NUMINAMATH_GPT_minimum_value_of_y_at_l955_95560

noncomputable def y (x : ℝ) : ℝ := x * 2^x

theorem minimum_value_of_y_at :
  ∃ x : ℝ, (∀ x' : ℝ, y x ≤ y x') ∧ x = -1 / Real.log 2 :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_of_y_at_l955_95560


namespace NUMINAMATH_GPT_total_trees_planted_l955_95537

theorem total_trees_planted :
  let fourth_graders := 30
  let fifth_graders := 2 * fourth_graders
  let sixth_graders := 3 * fifth_graders - 30
  fourth_graders + fifth_graders + sixth_graders = 240 :=
by
  sorry

end NUMINAMATH_GPT_total_trees_planted_l955_95537


namespace NUMINAMATH_GPT_evaluate_expression_l955_95592

theorem evaluate_expression (a : ℚ) (h : a = 4 / 3) : 
  (6 * a ^ 2 - 15 * a + 5) * (3 * a - 4) = 0 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l955_95592


namespace NUMINAMATH_GPT_value_of_x_l955_95570

theorem value_of_x (g : ℝ → ℝ) (h : ∀ x, g (5 * x + 2) = 3 * x - 4) : g (-13) = -13 :=
by {
  sorry
}

end NUMINAMATH_GPT_value_of_x_l955_95570


namespace NUMINAMATH_GPT_altitude_change_correct_l955_95524

noncomputable def altitude_change (T_ground T_high : ℝ) (deltaT_per_km : ℝ) : ℝ :=
  (T_high - T_ground) / deltaT_per_km

theorem altitude_change_correct :
  altitude_change 18 (-48) (-6) = 11 :=
by 
  sorry

end NUMINAMATH_GPT_altitude_change_correct_l955_95524


namespace NUMINAMATH_GPT_johns_family_total_members_l955_95591

theorem johns_family_total_members (n_f : ℕ) (h_f : n_f = 10) (n_m : ℕ) (h_m : n_m = (13 * n_f) / 10) :
  n_f + n_m = 23 := by
  rw [h_f, h_m]
  norm_num
  sorry

end NUMINAMATH_GPT_johns_family_total_members_l955_95591


namespace NUMINAMATH_GPT_solution_2016_121_solution_2016_144_l955_95519

-- Definitions according to the given conditions
def delta_fn (f : ℕ → ℕ → ℕ) :=
  (∀ a b : ℕ, f (a + b) b = f a b + 1) ∧ (∀ a b : ℕ, f a b * f b a = 0)

-- Proof objectives
theorem solution_2016_121 (f : ℕ → ℕ → ℕ) (h : delta_fn f) : f 2016 121 = 16 :=
sorry

theorem solution_2016_144 (f : ℕ → ℕ → ℕ) (h : delta_fn f) : f 2016 144 = 13 :=
sorry

end NUMINAMATH_GPT_solution_2016_121_solution_2016_144_l955_95519


namespace NUMINAMATH_GPT_isosceles_triangle_area_l955_95549

open Real

noncomputable def area_of_isosceles_triangle (b : ℝ) (h : ℝ) : ℝ :=
  (1/2) * b * h

theorem isosceles_triangle_area :
  ∃ (b : ℝ) (l : ℝ), h = 8 ∧ (2 * l + b = 32) ∧ (area_of_isosceles_triangle b h = 48) :=
by
  sorry

end NUMINAMATH_GPT_isosceles_triangle_area_l955_95549


namespace NUMINAMATH_GPT_smallest_integer_solution_l955_95558

theorem smallest_integer_solution :
  ∃ y : ℤ, (5 / 8 < (y - 3) / 19) ∧ ∀ z : ℤ, (5 / 8 < (z - 3) / 19) → y ≤ z :=
sorry

end NUMINAMATH_GPT_smallest_integer_solution_l955_95558


namespace NUMINAMATH_GPT_find_vector_at_t5_l955_95510

def vector_on_line (t : ℝ) : ℝ × ℝ := 
  let a := (0, 11) -- From solving the system of equations
  let d := (2, -4) -- From solving the system of equations
  (a.1 + t * d.1, a.2 + t * d.2)

theorem find_vector_at_t5 : vector_on_line 5 = (10, -9) := 
by 
  sorry

end NUMINAMATH_GPT_find_vector_at_t5_l955_95510


namespace NUMINAMATH_GPT_henry_total_payment_l955_95509

-- Define the conditions
def painting_payment : ℕ := 5
def selling_extra_payment : ℕ := 8
def total_payment_per_bike : ℕ := painting_payment + selling_extra_payment  -- 13

-- Define the quantity of bikes
def bikes_count : ℕ := 8

-- Calculate the total payment for painting and selling 8 bikes
def total_payment : ℕ := bikes_count * total_payment_per_bike  -- 144

-- The statement to prove
theorem henry_total_payment : total_payment = 144 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_henry_total_payment_l955_95509


namespace NUMINAMATH_GPT_proper_fraction_cubed_numerator_triples_denominator_add_three_l955_95541

theorem proper_fraction_cubed_numerator_triples_denominator_add_three
  (a b : ℕ)
  (h1 : a < b)
  (h2 : (a^3 : ℚ) / (b + 3) = 3 * (a : ℚ) / b) : 
  a = 2 ∧ b = 9 :=
by
  sorry

end NUMINAMATH_GPT_proper_fraction_cubed_numerator_triples_denominator_add_three_l955_95541


namespace NUMINAMATH_GPT_fraction_of_males_on_time_l955_95584

theorem fraction_of_males_on_time (A : ℕ) :
  (2 / 9 : ℚ) * A = (2 / 9 : ℚ) * A → 
  (2 / 3 : ℚ) * A = (2 / 3 : ℚ) * A → 
  (5 / 6 : ℚ) * ((1 / 3 : ℚ) * A) = (5 / 6 : ℚ) * ((1 / 3 : ℚ) * A) → 
  ((7 / 9 : ℚ) * A - (5 / 18 : ℚ) * A) / ((2 / 3 : ℚ) * A) = (1 / 2 : ℚ) :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_fraction_of_males_on_time_l955_95584


namespace NUMINAMATH_GPT_number_of_tacos_you_ordered_l955_95504

variable {E : ℝ} -- E represents the cost of one enchilada in dollars

-- Conditions
axiom h1 : ∃ t : ℕ, 0.9 * (t : ℝ) + 3 * E = 7.80
axiom h2 : 0.9 * 3 + 5 * E = 12.70

theorem number_of_tacos_you_ordered (E : ℝ) : ∃ t : ℕ, t = 2 := by
  sorry

end NUMINAMATH_GPT_number_of_tacos_you_ordered_l955_95504


namespace NUMINAMATH_GPT_solve_correct_problems_l955_95566

theorem solve_correct_problems (x : ℕ) (h1 : 3 * x + x = 120) : x = 30 :=
by
  sorry

end NUMINAMATH_GPT_solve_correct_problems_l955_95566


namespace NUMINAMATH_GPT_actual_distance_between_towns_l955_95544

-- Definitions based on conditions
def scale_inch_to_miles : ℚ := 8
def map_distance_inches : ℚ := 27 / 8

-- Proof statement
theorem actual_distance_between_towns : scale_inch_to_miles * map_distance_inches / (1 / 4) = 108 := by
  sorry

end NUMINAMATH_GPT_actual_distance_between_towns_l955_95544


namespace NUMINAMATH_GPT_intersection_of_prime_and_even_is_two_l955_95568

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_even (n : ℕ) : Prop :=
  ∃ k : ℤ, n = 2 * k

theorem intersection_of_prime_and_even_is_two :
  {n : ℕ | is_prime n} ∩ {n : ℕ | is_even n} = {2} :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_prime_and_even_is_two_l955_95568


namespace NUMINAMATH_GPT_last_three_digits_of_8_pow_108_l955_95502

theorem last_three_digits_of_8_pow_108 :
  (8^108 % 1000) = 38 := 
sorry

end NUMINAMATH_GPT_last_three_digits_of_8_pow_108_l955_95502


namespace NUMINAMATH_GPT_fifth_rectangle_is_square_l955_95557

-- Define the conditions
variables (s : ℝ) (a b : ℝ)
variables (R1 R2 R3 R4 : Set (ℝ × ℝ))
variables (R5 : Set (ℝ × ℝ))

-- Assume the areas of the corner rectangles are equal
def equal_area (R : Set (ℝ × ℝ)) (k : ℝ) : Prop :=
  ∃ (a b : ℝ), R = {p | p.1 < a ∧ p.2 < b} ∧ a * b = k

-- State the conditions
axiom h1 : equal_area R1 a
axiom h2 : equal_area R2 a
axiom h3 : equal_area R3 a
axiom h4 : equal_area R4 a

axiom h5 : ∀ (p : ℝ × ℝ), p ∈ R5 → p.1 ≠ 0 → p.2 ≠ 0

-- Prove that the fifth rectangle is a square
theorem fifth_rectangle_is_square : ∃ c : ℝ, ∀ r1 r2, r1 ∈ R5 → r2 ∈ R5 → r1.1 - r2.1 = c ∧ r1.2 - r2.2 = c :=
by sorry

end NUMINAMATH_GPT_fifth_rectangle_is_square_l955_95557


namespace NUMINAMATH_GPT_win_sector_area_l955_95593

theorem win_sector_area (r : ℝ) (P : ℝ) (h0 : r = 8) (h1 : P = 3 / 8) :
    let area_total := Real.pi * r ^ 2
    let area_win := P * area_total
    area_win = 24 * Real.pi :=
by 
  sorry

end NUMINAMATH_GPT_win_sector_area_l955_95593


namespace NUMINAMATH_GPT_tom_walking_distance_l955_95562

noncomputable def walking_rate_miles_per_minute : ℝ := 1 / 18
def walking_time_minutes : ℝ := 15
def expected_distance_miles : ℝ := 0.8

theorem tom_walking_distance :
  walking_rate_miles_per_minute * walking_time_minutes = expected_distance_miles :=
by
  -- Calculation steps and conversion to decimal are skipped
  sorry

end NUMINAMATH_GPT_tom_walking_distance_l955_95562


namespace NUMINAMATH_GPT_usual_time_cover_journey_l955_95561

theorem usual_time_cover_journey (S T : ℝ) (H : S / T = (5/6 * S) / (T + 8)) : T = 48 :=
by
  sorry

end NUMINAMATH_GPT_usual_time_cover_journey_l955_95561


namespace NUMINAMATH_GPT_least_prime_factor_of_expr_l955_95588

theorem least_prime_factor_of_expr : ∀ n : ℕ, n = 11^5 - 11^2 → (∃ p : ℕ, Nat.Prime p ∧ p ≤ 2 ∧ p ∣ n) :=
by
  intros n h
  -- here will be proof steps, currently skipped
  sorry

end NUMINAMATH_GPT_least_prime_factor_of_expr_l955_95588


namespace NUMINAMATH_GPT_symmetric_axis_of_parabola_l955_95516

theorem symmetric_axis_of_parabola :
  (∃ x : ℝ, x = 6 ∧ (∀ y : ℝ, y = 1/2 * x^2 - 6 * x + 21)) :=
sorry

end NUMINAMATH_GPT_symmetric_axis_of_parabola_l955_95516


namespace NUMINAMATH_GPT_part_a_part_b_l955_95552

-- Part (a)
theorem part_a (x y z : ℤ) : (x^2 + y^2 + z^2 = 2 * x * y * z) → (x = 0 ∧ y = 0 ∧ z = 0) :=
by
  sorry

-- Part (b)
theorem part_b : ∃ (x y z v : ℤ), (x^2 + y^2 + z^2 + v^2 = 2 * x * y * z * v) → (x = 0 ∧ y = 0 ∧ z = 0 ∧ v = 0) :=
by
  sorry

end NUMINAMATH_GPT_part_a_part_b_l955_95552


namespace NUMINAMATH_GPT_arithmetic_sequence_150th_term_l955_95589

open Nat

-- Define the nth term of an arithmetic sequence
def nth_term_arithmetic (a1 d n : ℕ) : ℕ :=
  a1 + (n - 1) * d

-- Theorem to prove
theorem arithmetic_sequence_150th_term (a1 d n : ℕ) (h1 : a1 = 3) (h2 : d = 7) (h3 : n = 150) :
  nth_term_arithmetic a1 d n = 1046 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_150th_term_l955_95589


namespace NUMINAMATH_GPT_hawkeye_fewer_mainecoons_than_gordon_l955_95532

-- Definitions based on conditions
def JamiePersians : ℕ := 4
def JamieMaineCoons : ℕ := 2
def GordonPersians : ℕ := JamiePersians / 2
def GordonMaineCoons : ℕ := JamieMaineCoons + 1
def TotalCats : ℕ := 13
def JamieTotalCats : ℕ := JamiePersians + JamieMaineCoons
def GordonTotalCats : ℕ := GordonPersians + GordonMaineCoons
def JamieAndGordonTotalCats : ℕ := JamieTotalCats + GordonTotalCats
def HawkeyeTotalCats : ℕ := TotalCats - JamieAndGordonTotalCats
def HawkeyePersians : ℕ := 0
def HawkeyeMaineCoons : ℕ := HawkeyeTotalCats - HawkeyePersians

-- Theorem statement to prove: Hawkeye owns 1 fewer Maine Coon than Gordon
theorem hawkeye_fewer_mainecoons_than_gordon : HawkeyeMaineCoons + 1 = GordonMaineCoons :=
by
  sorry

end NUMINAMATH_GPT_hawkeye_fewer_mainecoons_than_gordon_l955_95532


namespace NUMINAMATH_GPT_product_of_three_numbers_l955_95545

theorem product_of_three_numbers (a b c : ℝ) 
  (h1 : a + b + c = 30) 
  (h2 : a = 5 * (b + c)) 
  (h3 : b = 9 * c) : 
  a * b * c = 56.25 := 
by 
  sorry

end NUMINAMATH_GPT_product_of_three_numbers_l955_95545


namespace NUMINAMATH_GPT_sector_area_l955_95564

theorem sector_area (theta l : ℝ) (h_theta : theta = 2) (h_l : l = 2) :
    let r := l / theta
    let S := 1 / 2 * l * r
    S = 1 := by
  sorry

end NUMINAMATH_GPT_sector_area_l955_95564


namespace NUMINAMATH_GPT_octal_to_decimal_l955_95521

theorem octal_to_decimal (d0 d1 : ℕ) (n8 : ℕ) (n10 : ℕ) 
  (h1 : d0 = 3) (h2 : d1 = 5) (h3 : n8 = 53) (h4 : n10 = 43) : 
  (d1 * 8^1 + d0 * 8^0 = n10) :=
by
  sorry

end NUMINAMATH_GPT_octal_to_decimal_l955_95521


namespace NUMINAMATH_GPT_delta_delta_delta_45_l955_95527

def delta (P : ℚ) : ℚ := (2 / 3) * P + 2

theorem delta_delta_delta_45 :
  delta (delta (delta 45)) = 158 / 9 :=
by sorry

end NUMINAMATH_GPT_delta_delta_delta_45_l955_95527


namespace NUMINAMATH_GPT_walking_west_negation_l955_95598

theorem walking_west_negation (distance_east distance_west : Int) (h_east : distance_east = 6) (h_west : distance_west = -10) : 
    (10 : Int) = - distance_west := by
  sorry

end NUMINAMATH_GPT_walking_west_negation_l955_95598


namespace NUMINAMATH_GPT_smallest_odd_prime_factor_2021_8_plus_1_l955_95529

noncomputable def least_odd_prime_factor (n : ℕ) : ℕ :=
  if 2021^8 + 1 = 0 then 2021^8 + 1 else sorry 

theorem smallest_odd_prime_factor_2021_8_plus_1 :
  least_odd_prime_factor (2021^8 + 1) = 97 :=
  by
    sorry

end NUMINAMATH_GPT_smallest_odd_prime_factor_2021_8_plus_1_l955_95529


namespace NUMINAMATH_GPT_piravena_flight_cost_l955_95515

noncomputable def cost_of_flight (distance_km : ℕ) (booking_fee : ℕ) (rate_per_km : ℕ) : ℕ :=
  booking_fee + (distance_km * rate_per_km / 100)

def check_cost_of_flight : Prop :=
  let distance_bc := 1000
  let booking_fee := 100
  let rate_per_km := 10
  cost_of_flight distance_bc booking_fee rate_per_km = 200

theorem piravena_flight_cost : check_cost_of_flight := 
by {
  sorry
}

end NUMINAMATH_GPT_piravena_flight_cost_l955_95515


namespace NUMINAMATH_GPT_base8_subtraction_correct_l955_95522

theorem base8_subtraction_correct :
  ∀ (a b : ℕ) (h1 : a = 7534) (h2 : b = 3267),
      (a - b) % 8 = 4243 % 8 := by
  sorry

end NUMINAMATH_GPT_base8_subtraction_correct_l955_95522


namespace NUMINAMATH_GPT_number_of_ants_l955_95553

def spiders := 8
def spider_legs := 8
def ants := 12
def ant_legs := 6
def total_legs := 136

theorem number_of_ants :
  spiders * spider_legs + ants * ant_legs = total_legs → ants = 12 :=
by
  sorry

end NUMINAMATH_GPT_number_of_ants_l955_95553


namespace NUMINAMATH_GPT_GCD_180_252_315_l955_95538

theorem GCD_180_252_315 : Nat.gcd 180 (Nat.gcd 252 315) = 9 := by
  sorry

end NUMINAMATH_GPT_GCD_180_252_315_l955_95538


namespace NUMINAMATH_GPT_calculation_2015_l955_95554

theorem calculation_2015 :
  2015 ^ 2 - 2016 * 2014 = 1 :=
by
  sorry

end NUMINAMATH_GPT_calculation_2015_l955_95554


namespace NUMINAMATH_GPT_max_power_sum_l955_95550

open Nat

theorem max_power_sum (a b : ℕ) (h_a_pos : a > 0) (h_b_gt_one : b > 1) (h_max : a ^ b < 500 ∧ 
  ∀ (a' b' : ℕ), a' > 0 → b' > 1 → a' ^ b' < 500 → a' ^ b' ≤ a ^ b ) : a + b = 24 :=
sorry

end NUMINAMATH_GPT_max_power_sum_l955_95550


namespace NUMINAMATH_GPT_trig_inequality_l955_95523

theorem trig_inequality : Real.tan 1 > Real.sin 1 ∧ Real.sin 1 > Real.cos 1 := by
  sorry

end NUMINAMATH_GPT_trig_inequality_l955_95523


namespace NUMINAMATH_GPT_min_value_of_z_l955_95543

theorem min_value_of_z (x y : ℝ) (h : y^2 = 4 * x) : 
  ∃ (z : ℝ), z = 3 ∧ ∀ (x' : ℝ) (hx' : x' ≥ 0), ∃ (y' : ℝ), y'^2 = 4 * x' → z ≤ (1/2) * y'^2 + x'^2 + 3 :=
by sorry

end NUMINAMATH_GPT_min_value_of_z_l955_95543


namespace NUMINAMATH_GPT_student_marks_l955_95551

theorem student_marks (T P F M : ℕ)
  (hT : T = 600)
  (hP : P = 33)
  (hF : F = 73)
  (hM : M = (P * T / 100) - F) : M = 125 := 
by 
  sorry

end NUMINAMATH_GPT_student_marks_l955_95551


namespace NUMINAMATH_GPT_nina_expected_tomato_harvest_l955_95556

noncomputable def expected_tomato_harvest 
  (garden_length : ℝ) (garden_width : ℝ) 
  (plants_per_sq_ft : ℝ) (tomatoes_per_plant : ℝ) : ℝ :=
  garden_length * garden_width * plants_per_sq_ft * tomatoes_per_plant

theorem nina_expected_tomato_harvest : 
  expected_tomato_harvest 10 20 5 10 = 10000 :=
by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_nina_expected_tomato_harvest_l955_95556


namespace NUMINAMATH_GPT_math_problem_l955_95595

theorem math_problem : 12 - (- 18) + (- 7) - 15 = 8 :=
by
  sorry

end NUMINAMATH_GPT_math_problem_l955_95595


namespace NUMINAMATH_GPT_parabola_directrix_eq_l955_95575

def parabola_directrix (p : ℝ) : ℝ := -p

theorem parabola_directrix_eq (x y p : ℝ) (h : y ^ 2 = 8 * x) (hp : 2 * p = 8) : 
  parabola_directrix p = -2 :=
by
  sorry

end NUMINAMATH_GPT_parabola_directrix_eq_l955_95575


namespace NUMINAMATH_GPT_alyssa_puppies_l955_95567

-- Definitions from the problem conditions
def initial_puppies (P x : ℕ) : ℕ := P + x

-- Lean 4 Statement of the problem
theorem alyssa_puppies (P x : ℕ) (given_aw: 7 = 7) (remaining: 5 = 5) :
  initial_puppies P x = 12 :=
sorry

end NUMINAMATH_GPT_alyssa_puppies_l955_95567


namespace NUMINAMATH_GPT_book_cost_l955_95535

-- Definitions from conditions
def priceA : ℝ := 340
def priceB : ℝ := 350
def gain_percent_more : ℝ := 0.05

-- proof problem
theorem book_cost (C : ℝ) (G : ℝ) :
  (priceA - C = G) →
  (priceB - C = (1 + gain_percent_more) * G) →
  C = 140 :=
by
  intros
  sorry

end NUMINAMATH_GPT_book_cost_l955_95535


namespace NUMINAMATH_GPT_sam_found_pennies_l955_95530

-- Define the function that computes the number of pennies Sam found given the initial and current amounts of pennies
def find_pennies (initial_pennies current_pennies : Nat) : Nat :=
  current_pennies - initial_pennies

-- Define the main proof problem
theorem sam_found_pennies : find_pennies 98 191 = 93 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_GPT_sam_found_pennies_l955_95530


namespace NUMINAMATH_GPT_sum_series_eq_3_over_4_l955_95508

theorem sum_series_eq_3_over_4 :
  (∑' k: ℕ, (k + 1) / (3:ℚ)^(k+1)) = 3 / 4 := sorry

end NUMINAMATH_GPT_sum_series_eq_3_over_4_l955_95508


namespace NUMINAMATH_GPT_investment_accumulation_l955_95525

variable (P : ℝ) -- Initial investment amount
variable (r1 r2 r3 : ℝ) -- Interest rates for the first 3 years
variable (r4 : ℝ) -- Interest rate for the fourth year
variable (r5 : ℝ) -- Interest rate for the fifth year

-- Conditions
def conditions : Prop :=
  r1 = 0.07 ∧ 
  r2 = 0.08 ∧
  r3 = 0.10 ∧
  r4 = r3 + r3 * 0.12 ∧
  r5 = r4 - r4 * 0.08

-- The accumulated amount after 5 years
def accumulated_amount : ℝ :=
  P * (1 + r1) * (1 + r2) * (1 + r3) * (1 + r4) * (1 + r5)

-- Proof problem
theorem investment_accumulation (P : ℝ) :
  conditions r1 r2 r3 r4 r5 → 
  accumulated_amount P r1 r2 r3 r4 r5 = 1.8141 * P := by
  sorry

end NUMINAMATH_GPT_investment_accumulation_l955_95525


namespace NUMINAMATH_GPT_octal_to_base12_conversion_l955_95559

-- Define the computation functions required
def octalToDecimal (n : ℕ) : ℕ :=
  let d0 := n % 10
  let d1 := (n / 10) % 10
  let d2 := (n / 100) % 10
  d2 * 64 + d1 * 8 + d0

def decimalToBase12 (n : ℕ) : List ℕ :=
  let d0 := n % 12
  let n1 := n / 12
  let d1 := n1 % 12
  let n2 := n1 / 12
  let d2 := n2 % 12
  [d2, d1, d0]

-- The main theorem that combines both conversions
theorem octal_to_base12_conversion :
  decimalToBase12 (octalToDecimal 563) = [2, 6, 11] :=
sorry

end NUMINAMATH_GPT_octal_to_base12_conversion_l955_95559


namespace NUMINAMATH_GPT_largest_x_value_l955_95563

theorem largest_x_value (x y z : ℝ) (h1 : x + y + z = 6) (h2 : x * y + x * z + y * z = 9) : x ≤ 4 := 
sorry

end NUMINAMATH_GPT_largest_x_value_l955_95563


namespace NUMINAMATH_GPT_total_seats_at_round_table_l955_95506

-- Define namespace and conditions
namespace KingArthur

variable (n : ℕ) -- Total number of seats

-- Conditions
def king_position : ℕ := 10
def lancelot_position : ℕ := 29
def opposite (a b : ℕ) (n : ℕ) : Prop := (a + (n / 2)) % n = b

-- Proof statement
theorem total_seats_at_round_table : opposite 10 29 n ∧ 29 < n → n = 38 :=
by
  sorry

end KingArthur

end NUMINAMATH_GPT_total_seats_at_round_table_l955_95506


namespace NUMINAMATH_GPT_workshop_processing_equation_l955_95542

noncomputable def process_equation (x : ℝ) : Prop :=
  (4000 / x - 4200 / (1.5 * x) = 3)

theorem workshop_processing_equation (x : ℝ) (hx : x > 0) :
  process_equation x :=
by
  sorry

end NUMINAMATH_GPT_workshop_processing_equation_l955_95542


namespace NUMINAMATH_GPT_each_piglet_ate_9_straws_l955_95512

theorem each_piglet_ate_9_straws (t : ℕ) (h_t : t = 300)
                                 (p : ℕ) (h_p : p = 20)
                                 (f : ℕ) (h_f : f = (3 * t / 5)) :
  f / p = 9 :=
by
  sorry

end NUMINAMATH_GPT_each_piglet_ate_9_straws_l955_95512


namespace NUMINAMATH_GPT_sin_double_angle_identity_l955_95579

theorem sin_double_angle_identity (α : ℝ) (h : Real.cos α = 1 / 4) : 
  Real.sin (π / 2 - 2 * α) = -7 / 8 :=
by 
  sorry

end NUMINAMATH_GPT_sin_double_angle_identity_l955_95579


namespace NUMINAMATH_GPT_problem_statement_l955_95596

def a : ℤ := 2020
def b : ℤ := 2022

theorem problem_statement : b^3 - a * b^2 - a^2 * b + a^3 = 16168 := by
  sorry

end NUMINAMATH_GPT_problem_statement_l955_95596


namespace NUMINAMATH_GPT_circle_formed_by_PO_equals_3_l955_95503

variable (P : ℝ × ℝ) (O : ℝ × ℝ) (r : ℝ)
variable (h_O_fixed : True)
variable (h_PO_constant : dist P O = 3)

theorem circle_formed_by_PO_equals_3 : 
  {P | ∃ (x y : ℝ), dist (x, y) O = 3} = {P | (dist P O = r) ∧ (r = 3)} :=
by
  sorry

end NUMINAMATH_GPT_circle_formed_by_PO_equals_3_l955_95503


namespace NUMINAMATH_GPT_amelia_drove_distance_on_Monday_l955_95500

theorem amelia_drove_distance_on_Monday 
  (total_distance : ℕ) (tuesday_distance : ℕ) (remaining_distance : ℕ)
  (total_distance_eq : total_distance = 8205) 
  (tuesday_distance_eq : tuesday_distance = 582) 
  (remaining_distance_eq : remaining_distance = 6716) :
  ∃ x : ℕ, x + tuesday_distance + remaining_distance = total_distance ∧ x = 907 :=
by
  sorry

end NUMINAMATH_GPT_amelia_drove_distance_on_Monday_l955_95500


namespace NUMINAMATH_GPT_min_value_of_reciprocal_sum_l955_95540

variables {m n : ℝ}
variables (h1 : m > 0)
variables (h2 : n > 0)
variables (h3 : m + n = 1)

theorem min_value_of_reciprocal_sum : 
  (1 / m + 1 / n) = 4 :=
by
  sorry

end NUMINAMATH_GPT_min_value_of_reciprocal_sum_l955_95540


namespace NUMINAMATH_GPT_colten_chickens_l955_95590

variable (Colten Skylar Quentin : ℕ)

def chicken_problem_conditions :=
  (Skylar = 3 * Colten - 4) ∧
  (Quentin = 6 * Skylar + 17) ∧
  (Colten + Skylar + Quentin = 383)

theorem colten_chickens (h : chicken_problem_conditions Colten Skylar Quentin) : Colten = 37 :=
sorry

end NUMINAMATH_GPT_colten_chickens_l955_95590


namespace NUMINAMATH_GPT_bananas_unit_measurement_l955_95531

-- Definition of given conditions
def units_per_day : ℕ := 13
def total_bananas : ℕ := 9828
def total_weeks : ℕ := 9
def days_per_week : ℕ := 7
def total_days : ℕ := total_weeks * days_per_week
def bananas_per_day : ℕ := total_bananas / total_days
def bananas_per_unit : ℕ := bananas_per_day / units_per_day

-- Main theorem statement
theorem bananas_unit_measurement :
  bananas_per_unit = 12 := sorry

end NUMINAMATH_GPT_bananas_unit_measurement_l955_95531


namespace NUMINAMATH_GPT_image_preimage_f_l955_95547

-- Defining the function f
def f (p : ℝ × ℝ) : ℝ × ℝ := (p.1 + p.2, p.1 - p.2)

-- Given conditions
def A : Set (ℝ × ℝ) := {p | True}
def B : Set (ℝ × ℝ) := {p | True}

-- Proof statement
theorem image_preimage_f :
  f (1, 3) = (4, -2) ∧ ∃ x y : ℝ, f (x, y) = (1, 3) ∧ (x, y) = (2, -1) :=
by
  sorry

end NUMINAMATH_GPT_image_preimage_f_l955_95547


namespace NUMINAMATH_GPT_least_common_multiple_first_ten_integers_l955_95572

theorem least_common_multiple_first_ten_integers : 
  Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7) 8) 9) 10 = 2520 :=
sorry

end NUMINAMATH_GPT_least_common_multiple_first_ten_integers_l955_95572


namespace NUMINAMATH_GPT_words_per_page_l955_95587

theorem words_per_page (p : ℕ) (h1 : 150 * p ≡ 210 [MOD 221]) (h2 : p ≤ 90) : p = 90 :=
sorry

end NUMINAMATH_GPT_words_per_page_l955_95587


namespace NUMINAMATH_GPT_average_of_consecutive_numbers_l955_95534

-- Define the 7 consecutive numbers and their properties
variables (a : ℝ) (b : ℝ) (c : ℝ) (d : ℝ) (e : ℝ) (f : ℝ) (g : ℝ)

-- Conditions given in the problem
def consecutive_numbers (a b c d e f g : ℝ) : Prop :=
  b = a + 1 ∧ c = a + 2 ∧ d = a + 3 ∧ e = a + 4 ∧ f = a + 5 ∧ g = a + 6

def percent_relationship (a g : ℝ) : Prop :=
  g = 1.5 * a

-- The proof problem
theorem average_of_consecutive_numbers (a b c d e f g : ℝ)
  (h1 : consecutive_numbers a b c d e f g)
  (h2 : percent_relationship a g) :
  (a + b + c + d + e + f + g) / 7 = 15 :=
by {
  sorry -- Proof goes here
}

-- To ensure it passes the type checker but without providing the actual proof, we use sorry.

end NUMINAMATH_GPT_average_of_consecutive_numbers_l955_95534


namespace NUMINAMATH_GPT_fraction_equivalence_l955_95571

-- Given fractions
def frac1 : ℚ := 3 / 7
def frac2 : ℚ := 4 / 5
def frac3 : ℚ := 5 / 12
def frac4 : ℚ := 2 / 9

-- Expectation
def result : ℚ := 1548 / 805

-- Theorem to prove the equality
theorem fraction_equivalence : ((frac1 + frac2) / (frac3 + frac4)) = result := by
  sorry

end NUMINAMATH_GPT_fraction_equivalence_l955_95571


namespace NUMINAMATH_GPT_equation_line_through_intersections_l955_95505

theorem equation_line_through_intersections (A1 B1 A2 B2 : ℝ)
  (h1 : 2 * A1 + 3 * B1 = 1)
  (h2 : 2 * A2 + 3 * B2 = 1) :
  ∃ (a b c : ℝ), a = 2 ∧ b = 3 ∧ c = -1 ∧ (a * x + b * y + c = 0) := 
sorry

end NUMINAMATH_GPT_equation_line_through_intersections_l955_95505


namespace NUMINAMATH_GPT_find_value_of_ratio_l955_95597

theorem find_value_of_ratio (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x / y + y / x = 4) :
  (x + 2 * y) / (x - 2 * y) = Real.sqrt 33 / 3 := 
  sorry

end NUMINAMATH_GPT_find_value_of_ratio_l955_95597


namespace NUMINAMATH_GPT_value_of_composed_operations_l955_95576

def op1 (x : ℝ) : ℝ := 9 - x
def op2 (x : ℝ) : ℝ := x - 9

theorem value_of_composed_operations : op2 (op1 15) = -15 :=
by
  sorry

end NUMINAMATH_GPT_value_of_composed_operations_l955_95576


namespace NUMINAMATH_GPT_econ_not_feasible_l955_95599

theorem econ_not_feasible (x y p q: ℕ) (h_xy : 26 * x + 29 * y = 687) (h_pq : 27 * p + 31 * q = 687) : p + q ≥ x + y := by
  sorry

end NUMINAMATH_GPT_econ_not_feasible_l955_95599


namespace NUMINAMATH_GPT_symmetric_points_subtraction_l955_95585

theorem symmetric_points_subtraction (a b : ℝ) (h1 : -2 = -a) (h2 : b = -3) : a - b = 5 :=
by {
  sorry
}

end NUMINAMATH_GPT_symmetric_points_subtraction_l955_95585


namespace NUMINAMATH_GPT_quadratic_negative_roots_pq_value_l955_95574

theorem quadratic_negative_roots_pq_value (r : ℝ) :
  (∃ p q : ℝ, p = -87 ∧ q = -23 ∧ x^2 - (r + 7)*x + r + 87 = 0 ∧ p < r ∧ r < q)
  → ((-87)^2 + (-23)^2 = 8098) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_negative_roots_pq_value_l955_95574


namespace NUMINAMATH_GPT_find_f_of_3_l955_95565

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_f_of_3 :
  (∀ y : ℝ, y > 0 → f ((4 * y + 1) / (y + 1)) = 1 / y) → f 3 = 1 / 2 :=
by
  intros h
  have : f 3 = 1 / 2 := sorry
  exact this

end NUMINAMATH_GPT_find_f_of_3_l955_95565


namespace NUMINAMATH_GPT_max_is_twice_emily_probability_l955_95536

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

end NUMINAMATH_GPT_max_is_twice_emily_probability_l955_95536


namespace NUMINAMATH_GPT_grasshopper_position_after_100_jumps_l955_95578

theorem grasshopper_position_after_100_jumps :
  let start_pos := 1
  let jumps (n : ℕ) := n
  let total_positions := 6
  let total_distance := (100 * (100 + 1)) / 2
  (start_pos + (total_distance % total_positions)) % total_positions = 5 :=
by
  sorry

end NUMINAMATH_GPT_grasshopper_position_after_100_jumps_l955_95578


namespace NUMINAMATH_GPT_find_x_l955_95513

noncomputable def f (x : ℝ) : ℝ := x^2 * (x - 1)

theorem find_x (x : ℝ) (h : deriv f x = x) : x = 0 ∨ x = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l955_95513


namespace NUMINAMATH_GPT_steve_pie_difference_l955_95581

-- Definitions of conditions
def apple_pie_days : Nat := 3
def cherry_pie_days : Nat := 2
def pies_per_day : Nat := 12

-- Theorem statement
theorem steve_pie_difference : 
  (apple_pie_days * pies_per_day) - (cherry_pie_days * pies_per_day) = 12 := 
by
  sorry

end NUMINAMATH_GPT_steve_pie_difference_l955_95581


namespace NUMINAMATH_GPT_find_a_l955_95501

theorem find_a (a : ℝ) (h₁ : a ≠ 0) (h₂ : ∀ x y : ℝ, x^2 + a*y^2 + a^2 = 0) (h₃ : 4 = 4) :
  a = (1 - Real.sqrt 17) / 2 := sorry

end NUMINAMATH_GPT_find_a_l955_95501


namespace NUMINAMATH_GPT_triangle_middle_side_at_least_sqrt_two_l955_95526

theorem triangle_middle_side_at_least_sqrt_two
    (a b c : ℝ)
    (h1 : a ≥ b) (h2 : b ≥ c)
    (h3 : ∃ α : ℝ, 0 < α ∧ α < π ∧ 1 = 1/2 * b * c * Real.sin α) :
  b ≥ Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_triangle_middle_side_at_least_sqrt_two_l955_95526


namespace NUMINAMATH_GPT_price_correct_l955_95528

noncomputable def price_per_glass_on_second_day 
  (O : ℝ) 
  (price_first_day : ℝ) 
  (revenue_equal : 2 * O * price_first_day = 3 * O * P) 
  : ℝ := 0.40

theorem price_correct 
  (O : ℝ) 
  (price_first_day : ℝ) 
  (revenue_equal : 2 * O * price_first_day = 3 * O * 0.40) 
  : price_per_glass_on_second_day O price_first_day revenue_equal = 0.40 := 
by 
  sorry

end NUMINAMATH_GPT_price_correct_l955_95528


namespace NUMINAMATH_GPT_chess_tournament_games_l955_95594

def num_games (n : Nat) : Nat := n * (n - 1) * 2

theorem chess_tournament_games : num_games 7 = 84 :=
by
  sorry

end NUMINAMATH_GPT_chess_tournament_games_l955_95594


namespace NUMINAMATH_GPT_james_total_cost_l955_95520

def milk_cost : ℝ := 4.50
def milk_tax_rate : ℝ := 0.20
def banana_cost : ℝ := 3.00
def banana_tax_rate : ℝ := 0.15
def baguette_cost : ℝ := 2.50
def baguette_tax_rate : ℝ := 0.0
def cereal_cost : ℝ := 6.00
def cereal_discount_rate : ℝ := 0.20
def cereal_tax_rate : ℝ := 0.12
def eggs_cost : ℝ := 3.50
def eggs_coupon : ℝ := 1.00
def eggs_tax_rate : ℝ := 0.18

theorem james_total_cost :
  let milk_total := milk_cost * (1 + milk_tax_rate)
  let banana_total := banana_cost * (1 + banana_tax_rate)
  let baguette_total := baguette_cost * (1 + baguette_tax_rate)
  let cereal_discounted := cereal_cost * (1 - cereal_discount_rate)
  let cereal_total := cereal_discounted * (1 + cereal_tax_rate)
  let eggs_discounted := eggs_cost - eggs_coupon
  let eggs_total := eggs_discounted * (1 + eggs_tax_rate)
  milk_total + banana_total + baguette_total + cereal_total + eggs_total = 19.68 := 
by
  sorry

end NUMINAMATH_GPT_james_total_cost_l955_95520


namespace NUMINAMATH_GPT_people_lost_l955_95539

-- Define the given conditions
def ratio_won_to_lost : ℕ × ℕ := (4, 1)
def people_won : ℕ := 28

-- Define the proof problem
theorem people_lost (L : ℕ) (h_ratio : ratio_won_to_lost = (4, 1)) (h_won : people_won = 28) : L = 7 :=
by
  -- Skip the proof
  sorry

end NUMINAMATH_GPT_people_lost_l955_95539


namespace NUMINAMATH_GPT_pizza_slices_have_both_cheese_and_bacon_l955_95582

theorem pizza_slices_have_both_cheese_and_bacon:
  ∀ (total_slices cheese_slices bacon_slices n : ℕ),
  total_slices = 15 →
  cheese_slices = 8 →
  bacon_slices = 13 →
  (total_slices = cheese_slices + bacon_slices - n) →
  n = 6 :=
by {
  -- proof skipped
  sorry
}

end NUMINAMATH_GPT_pizza_slices_have_both_cheese_and_bacon_l955_95582
