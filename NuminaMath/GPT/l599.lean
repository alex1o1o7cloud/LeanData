import Mathlib

namespace NUMINAMATH_GPT_intersection_nonempty_implies_range_l599_59948

namespace ProofProblem

def M (x y : ℝ) : Prop := x + y + 1 ≥ Real.sqrt (2 * (x^2 + y^2))
def N (a x y : ℝ) : Prop := |x - a| + |y - 1| ≤ 1

theorem intersection_nonempty_implies_range (a : ℝ) :
  (∃ x y : ℝ, M x y ∧ N a x y) → (1 - Real.sqrt 6 ≤ a ∧ a ≤ 3 + Real.sqrt 10) :=
by
  sorry

end ProofProblem

end NUMINAMATH_GPT_intersection_nonempty_implies_range_l599_59948


namespace NUMINAMATH_GPT_geometric_sequence_first_term_l599_59919

noncomputable def first_term_of_geometric_sequence (a r : ℝ) : ℝ :=
  a

theorem geometric_sequence_first_term 
  (a r : ℝ)
  (h1 : a * r^3 = 720)   -- The fourth term is 6!
  (h2 : a * r^6 = 5040)  -- The seventh term is 7!
  : first_term_of_geometric_sequence a r = 720 / 7 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_first_term_l599_59919


namespace NUMINAMATH_GPT_not_always_true_inequality_l599_59956

variable {x y z : ℝ} {k : ℤ}

theorem not_always_true_inequality :
  x > 0 → y > 0 → x > y → z ≠ 0 → k ≠ 0 → ¬ ( ∀ z, (x / (z^k) > y / (z^k)) ) :=
by
  intro hx hy hxy hz hk
  sorry

end NUMINAMATH_GPT_not_always_true_inequality_l599_59956


namespace NUMINAMATH_GPT_balloon_highest_elevation_l599_59971

theorem balloon_highest_elevation 
  (lift_rate : ℕ)
  (descend_rate : ℕ)
  (pull_time1 : ℕ)
  (release_time : ℕ)
  (pull_time2 : ℕ) :
  lift_rate = 50 →
  descend_rate = 10 →
  pull_time1 = 15 →
  release_time = 10 →
  pull_time2 = 15 →
  (lift_rate * pull_time1 - descend_rate * release_time + lift_rate * pull_time2) = 1400 :=
by
  sorry

end NUMINAMATH_GPT_balloon_highest_elevation_l599_59971


namespace NUMINAMATH_GPT_time_difference_l599_59965

-- Definitions
def time_chinese : ℕ := 5
def time_english : ℕ := 7

-- Statement to prove
theorem time_difference : time_english - time_chinese = 2 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_time_difference_l599_59965


namespace NUMINAMATH_GPT_remaining_work_hours_l599_59949

theorem remaining_work_hours (initial_hours_per_week initial_weeks total_earnings first_weeks first_week_hours : ℝ) 
  (hourly_wage remaining_weeks remaining_earnings total_hours_required : ℝ) : 
  15 = initial_hours_per_week →
  15 = initial_weeks →
  4500 = total_earnings →
  3 = first_weeks →
  5 = first_week_hours →
  hourly_wage = total_earnings / (initial_hours_per_week * initial_weeks) →
  remaining_earnings = total_earnings - (first_week_hours * hourly_wage * first_weeks) →
  remaining_weeks = initial_weeks - first_weeks →
  total_hours_required = remaining_earnings / (hourly_wage * remaining_weeks) →
  total_hours_required = 17.5 :=
by
  intros
  sorry

end NUMINAMATH_GPT_remaining_work_hours_l599_59949


namespace NUMINAMATH_GPT_a_2008_lt_5_l599_59950

theorem a_2008_lt_5 :
  ∃ a b : ℕ → ℝ, 
    a 1 = 1 ∧ 
    b 1 = 2 ∧ 
    (∀ n, a (n + 1) = (1 + a n + a n * b n) / (b n)) ∧ 
    (∀ n, b (n + 1) = (1 + b n + a n * b n) / (a n)) ∧ 
    a 2008 < 5 := 
sorry

end NUMINAMATH_GPT_a_2008_lt_5_l599_59950


namespace NUMINAMATH_GPT_geometric_to_arithmetic_sequence_l599_59967

theorem geometric_to_arithmetic_sequence {a : ℕ → ℝ} (q : ℝ) 
    (h_gt0 : 0 < q) (h_pos : ∀ n, 0 < a n)
    (h_geom_seq : ∀ n, a (n + 1) = a n * q)
    (h_arith_seq : 2 * (1 / 2 * a 3) = a 1 + 2 * a 2) :
    a 10 / a 8 = 3 + 2 * Real.sqrt 2 := 
by
  sorry

end NUMINAMATH_GPT_geometric_to_arithmetic_sequence_l599_59967


namespace NUMINAMATH_GPT_arithmetic_geometric_seq_proof_l599_59997

theorem arithmetic_geometric_seq_proof
  (a1 a2 b1 b2 b3 : ℝ)
  (h1 : a1 - a2 = -1)
  (h2 : 1 * (b2 * b2) = 4)
  (h3 : b2 > 0) :
  (a1 - a2) / b2 = -1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_geometric_seq_proof_l599_59997


namespace NUMINAMATH_GPT_problem_statement_l599_59933

open Real Polynomial

theorem problem_statement (a1 a2 a3 d1 d2 d3 : ℝ) 
  (h : ∀ x : ℝ, x^8 - x^7 + x^6 - x^5 + x^4 - x^3 + x^2 - x + 1 =
                 (x^2 + a1 * x + d1) * (x^2 + a2 * x + d2) * (x^2 + a3 * x + d3) * (x^2 - 1)) :
  a1 * d1 + a2 * d2 + a3 * d3 = -1 := 
sorry

end NUMINAMATH_GPT_problem_statement_l599_59933


namespace NUMINAMATH_GPT_solve_equation_l599_59936

theorem solve_equation (x : ℝ) : 
  (1 / (x^2 + 13*x - 16) + 1 / (x^2 + 4*x - 16) + 1 / (x^2 - 15*x - 16) = 0) ↔ 
    (x = 1 ∨ x = -16 ∨ x = 4 ∨ x = -4) :=
by
  sorry

end NUMINAMATH_GPT_solve_equation_l599_59936


namespace NUMINAMATH_GPT_reach_any_position_l599_59988

/-- We define a configuration of marbles in terms of a finite list of natural numbers, which corresponds to the number of marbles in each hole. A configuration transitions to another by moving marbles from one hole to subsequent holes in a circular manner. -/
def configuration (n : ℕ) := List ℕ 

/-- Define the operation of distributing marbles from one hole to subsequent holes. -/
def redistribute (l : configuration n) (i : ℕ) : configuration n :=
  sorry -- The exact redistribution function would need to be implemented based on the conditions.

theorem reach_any_position (n : ℕ) (m : ℕ) (init_config final_config : configuration n)
  (h_num_marbles : init_config.sum = m)
  (h_final_marbles : final_config.sum = m) :
  ∃ steps, final_config = (steps : List ℕ).foldl redistribute init_config :=
sorry

end NUMINAMATH_GPT_reach_any_position_l599_59988


namespace NUMINAMATH_GPT_production_analysis_l599_59915

def daily_change (day: ℕ) : ℤ :=
  match day with
  | 0 => 40    -- Monday
  | 1 => -30   -- Tuesday
  | 2 => 90    -- Wednesday
  | 3 => -50   -- Thursday
  | 4 => -20   -- Friday
  | 5 => -10   -- Saturday
  | 6 => 20    -- Sunday
  | _ => 0     -- Invalid day, just in case

def planned_daily_production : ℤ := 500

def actual_production (day: ℕ) : ℤ :=
  planned_daily_production + (List.sum (List.map daily_change (List.range (day + 1))))

def total_production : ℤ :=
  List.sum (List.map actual_production (List.range 7))

theorem production_analysis :
  ∃ largest_increase_day smallest_increase_day : ℕ,
    largest_increase_day = 2 ∧  -- Wednesday
    smallest_increase_day = 1 ∧  -- Tuesday
    total_production = 3790 ∧
    total_production > 7 * planned_daily_production := by
  sorry

end NUMINAMATH_GPT_production_analysis_l599_59915


namespace NUMINAMATH_GPT_math_problem_l599_59982

theorem math_problem :
  ( (1 / 3 * 9) ^ 2 * (1 / 27 * 81) ^ 2 * (1 / 243 * 729) ^ 2) = 729 := by
  sorry

end NUMINAMATH_GPT_math_problem_l599_59982


namespace NUMINAMATH_GPT_largest_possible_a_l599_59905

theorem largest_possible_a (a b c d : ℕ) (ha : a < 2 * b) (hb : b < 3 * c) (hc : c < 4 * d) (hd : d < 100) : 
  a ≤ 2367 :=
sorry

end NUMINAMATH_GPT_largest_possible_a_l599_59905


namespace NUMINAMATH_GPT_div_eq_frac_l599_59959

theorem div_eq_frac : 250 / (5 + 12 * 3^2) = 250 / 113 :=
by
  sorry

end NUMINAMATH_GPT_div_eq_frac_l599_59959


namespace NUMINAMATH_GPT_parallel_lines_value_of_a_l599_59977

theorem parallel_lines_value_of_a (a : ℝ) : 
  (∀ x y : ℝ, ax + (a+2)*y + 2 = 0 → x + a*y + 1 = 0 → ∀ m n : ℝ, ax + (a + 2)*n + 2 = 0 → x + a*n + 1 = 0) →
  a = -1 := 
sorry

end NUMINAMATH_GPT_parallel_lines_value_of_a_l599_59977


namespace NUMINAMATH_GPT_range_of_a_l599_59929

-- Definitions related to the conditions in the problem
def polynomial (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := 3 * x ^ 5 - 4 * a * x ^ 3 + 2 * b ^ 2 * x ^ 2 + 1

def v_2 (x : ℝ) (a : ℝ) : ℝ := (3 * x + 0) * x - 4 * a

def v_3 (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := (((3 * x + 0) * x - 4 * a) * x + 2 * b ^ 2)

-- The main statement to prove
theorem range_of_a (x a b : ℝ) (h1 : x = 2) (h2 : ∀ b : ℝ, (v_2 x a) < (v_3 x a b)) : a < 3 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l599_59929


namespace NUMINAMATH_GPT_polar_equation_parabola_l599_59909

/-- Given a polar equation 4 * ρ * (sin(θ / 2))^2 = 5, prove that it represents a parabola in Cartesian coordinates. -/
theorem polar_equation_parabola (ρ θ : ℝ) (h : 4 * ρ * (Real.sin (θ / 2))^ 2 = 5) : 
  ∃ (a : ℝ), a ≠ 0 ∧ (∃ b c : ℝ, ∀ x y : ℝ, (y^2 = a * (x + b)) ∨ (x = c ∨ y = 0)) := 
sorry

end NUMINAMATH_GPT_polar_equation_parabola_l599_59909


namespace NUMINAMATH_GPT_x_squared_plus_y_squared_geq_five_l599_59968

theorem x_squared_plus_y_squared_geq_five (x y : ℝ) (h : abs (x - 2 * y) = 5) : x^2 + y^2 ≥ 5 := 
sorry

end NUMINAMATH_GPT_x_squared_plus_y_squared_geq_five_l599_59968


namespace NUMINAMATH_GPT_certain_number_unique_l599_59930

theorem certain_number_unique (x : ℝ) (hx1 : 213 * x = 3408) (hx2 : 21.3 * x = 340.8) : x = 16 :=
by
  sorry

end NUMINAMATH_GPT_certain_number_unique_l599_59930


namespace NUMINAMATH_GPT_rotate_circle_sectors_l599_59907

theorem rotate_circle_sectors (n : ℕ) (h : n > 0) :
  (∀ i, i < n → ∃ θ : ℝ, θ < (π / (n^2 - n + 1))) →
  ∃ θ : ℝ, 0 < θ ∧ θ < 2 * π ∧
  (∀ i : ℕ, i < n → (θ * i) % (2 * π) > (π / (n^2 - n + 1))) :=
sorry

end NUMINAMATH_GPT_rotate_circle_sectors_l599_59907


namespace NUMINAMATH_GPT_sum_of_center_coords_l599_59943

theorem sum_of_center_coords (x y : ℝ) :
  (∃ k : ℝ, (x + 2)^2 + (y + 3)^2 = k ∧ (x^2 + y^2 = -4 * x - 6 * y + 5)) -> x + y = -5 :=
by
sorry

end NUMINAMATH_GPT_sum_of_center_coords_l599_59943


namespace NUMINAMATH_GPT_license_plate_count_correct_l599_59983

-- Define the number of choices for digits and letters
def num_digit_choices : ℕ := 10^3
def num_letter_block_choices : ℕ := 26^3
def num_position_choices : ℕ := 4

-- Compute the total number of distinct license plates
def total_license_plates : ℕ := num_position_choices * num_digit_choices * num_letter_block_choices

-- The proof statement
theorem license_plate_count_correct : total_license_plates = 70304000 := by
  -- This proof is left as an exercise
  sorry

end NUMINAMATH_GPT_license_plate_count_correct_l599_59983


namespace NUMINAMATH_GPT_find_years_l599_59998

def sum_interest_years (P R : ℝ) (T : ℝ) : Prop :=
  (P * (R + 5) / 100 * T = P * R / 100 * T + 300) ∧ P = 600

theorem find_years {R : ℝ} {T : ℝ} (h1 : sum_interest_years 600 R T) : T = 10 :=
by
  -- proof omitted
  sorry

end NUMINAMATH_GPT_find_years_l599_59998


namespace NUMINAMATH_GPT_number_of_cars_sold_next_four_days_cars_sold_each_day_next_four_days_l599_59941

def cars_sold_each_day_first_three_days : ℕ := 5
def days_first_period : ℕ := 3
def quota : ℕ := 50
def cars_remaining_after_next_four_days : ℕ := 23
def days_next_period : ℕ := 4

theorem number_of_cars_sold_next_four_days :
  (quota - cars_sold_each_day_first_three_days * days_first_period) - cars_remaining_after_next_four_days = 12 :=
by
  sorry

theorem cars_sold_each_day_next_four_days :
  (quota - cars_sold_each_day_first_three_days * days_first_period - cars_remaining_after_next_four_days) / days_next_period = 3 :=
by
  sorry

end NUMINAMATH_GPT_number_of_cars_sold_next_four_days_cars_sold_each_day_next_four_days_l599_59941


namespace NUMINAMATH_GPT_find_square_sum_of_xy_l599_59966

theorem find_square_sum_of_xy (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (h1 : x * y + x + y = 83) (h2 : x^2 * y + x * y^2 = 1056) : x^2 + y^2 = 458 :=
sorry

end NUMINAMATH_GPT_find_square_sum_of_xy_l599_59966


namespace NUMINAMATH_GPT_PQ_parallel_to_AB_3_times_l599_59989

-- Definitions for the problem
structure Rectangle :=
  (A B C D : Type)
  (AB AD : ℝ)
  (P Q : ℝ → ℝ)
  (P_speed Q_speed : ℝ)
  (time : ℝ)

noncomputable def rectangle_properties (R : Rectangle) : Prop :=
  R.AB = 4 ∧
  R.AD = 12 ∧
  ∀ t, 0 ≤ t → t ≤ 12 → R.P t = t ∧  -- P moves from A to D at 1 cm/s
  R.Q_speed = 3 ∧                     -- Q moves at 3 cm/s
  ∀ t, R.Q t = R.Q_speed * t ∧             -- Q moves from C to B and back
  ∃ s1 s2 s3, R.P s1 = 4 ∧ R.P s2 = 8 ∧ R.P s3 = 12 ∧
  (R.Q s1 = 3 ∨ R.Q s1 = 1) ∧
  (R.Q s2 = 6 ∨ R.Q s2 = 2) ∧
  (R.Q s3 = 9 ∨ R.Q s3 = 0)

theorem PQ_parallel_to_AB_3_times : 
  ∀ (R : Rectangle), rectangle_properties R → 
  ∃ (times : ℕ), times = 3 :=
by
  sorry

end NUMINAMATH_GPT_PQ_parallel_to_AB_3_times_l599_59989


namespace NUMINAMATH_GPT_boat_navigation_under_arch_l599_59926

theorem boat_navigation_under_arch (h_arch : ℝ) (w_arch: ℝ) (boat_width: ℝ) (boat_height: ℝ) (boat_above_water: ℝ) :
  (h_arch = 5) → 
  (w_arch = 8) → 
  (boat_width = 4) → 
  (boat_height = 2) → 
  (boat_above_water = 0.75) →
  (h_arch - 2 = 3) :=
by
  intros h_arch_eq w_arch_eq boat_w_eq boat_h_eq boat_above_water_eq
  sorry

end NUMINAMATH_GPT_boat_navigation_under_arch_l599_59926


namespace NUMINAMATH_GPT_binary_add_sub_l599_59945

theorem binary_add_sub : 
  (1101 + 111 - 101 + 1001 - 11 : ℕ) = (10101 : ℕ) := by
  sorry

end NUMINAMATH_GPT_binary_add_sub_l599_59945


namespace NUMINAMATH_GPT_union_set_equiv_l599_59912

namespace ProofProblem

-- Define the sets A and B
def A : Set ℝ := { x | x - 1 > 0 }
def B : Set ℝ := { x | x^2 - x - 2 > 0 }

-- Define the union of A and B
def unionAB : Set ℝ := A ∪ B

-- State the proof problem
theorem union_set_equiv : unionAB = (Set.Iio (-1)) ∪ (Set.Ioi 1) := by
  sorry

end ProofProblem

end NUMINAMATH_GPT_union_set_equiv_l599_59912


namespace NUMINAMATH_GPT_precision_tens_place_l599_59955

-- Given
def given_number : ℝ := 4.028 * (10 ^ 5)

-- Prove that the precision of the given_number is to the tens place.
theorem precision_tens_place : true := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_precision_tens_place_l599_59955


namespace NUMINAMATH_GPT_find_f_value_l599_59944

noncomputable def f (a b c : ℝ) (x : ℝ) : ℝ :=
  a * x^5 - b * x^3 + c * x - 3

theorem find_f_value (a b c : ℝ) (h : f a b c (-3) = 7) : f a b c 3 = -13 :=
by
  sorry

end NUMINAMATH_GPT_find_f_value_l599_59944


namespace NUMINAMATH_GPT_horizontal_asymptote_of_rational_function_l599_59979

theorem horizontal_asymptote_of_rational_function :
  (∃ y, y = (10 * x ^ 4 + 3 * x ^ 3 + 7 * x ^ 2 + 6 * x + 4) / (2 * x ^ 4 + 5 * x ^ 3 + 4 * x ^ 2 + 2 * x + 1) → y = 5) := sorry

end NUMINAMATH_GPT_horizontal_asymptote_of_rational_function_l599_59979


namespace NUMINAMATH_GPT_angle_I_measure_l599_59910

theorem angle_I_measure {x y : ℝ} 
  (h1 : x = y - 50) 
  (h2 : 3 * x + 2 * y = 540)
  : y = 138 := 
by 
  sorry

end NUMINAMATH_GPT_angle_I_measure_l599_59910


namespace NUMINAMATH_GPT_wholesale_price_of_milk_l599_59974

theorem wholesale_price_of_milk (W : ℝ) 
  (h1 : ∀ p : ℝ, p = 1.25 * W) 
  (h2 : ∀ q : ℝ, q = 0.95 * (1.25 * W)) 
  (h3 : q = 4.75) :
  W = 4 :=
by
  sorry

end NUMINAMATH_GPT_wholesale_price_of_milk_l599_59974


namespace NUMINAMATH_GPT_jerry_initial_action_figures_l599_59942

theorem jerry_initial_action_figures 
(A : ℕ) 
(h1 : ∀ A, A + 7 = 9 + 3)
: A = 5 :=
by
  sorry

end NUMINAMATH_GPT_jerry_initial_action_figures_l599_59942


namespace NUMINAMATH_GPT_fraction_add_eq_l599_59993

theorem fraction_add_eq (x y : ℝ) (hx : y / x = 3 / 7) : (x + y) / x = 10 / 7 :=
by
  sorry

end NUMINAMATH_GPT_fraction_add_eq_l599_59993


namespace NUMINAMATH_GPT_houses_with_dogs_l599_59995

theorem houses_with_dogs (C B Total : ℕ) (hC : C = 30) (hB : B = 10) (hTotal : Total = 60) :
  ∃ D, D = 40 :=
by
  -- The overall proof would go here
  sorry

end NUMINAMATH_GPT_houses_with_dogs_l599_59995


namespace NUMINAMATH_GPT_probability_of_sum_being_6_l599_59972

noncomputable def prob_sum_6 : ℚ :=
  let total_outcomes := 6 * 6
  let favorable_outcomes := 5
  favorable_outcomes / total_outcomes

theorem probability_of_sum_being_6 :
  prob_sum_6 = 5 / 36 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_sum_being_6_l599_59972


namespace NUMINAMATH_GPT_cookie_division_l599_59935

theorem cookie_division (C : ℝ) (blue_fraction : ℝ := 1/4) (green_fraction_of_remaining : ℝ := 5/9)
  (remaining_fraction : ℝ := 3/4) (green_fraction : ℝ := 5/12) :
  blue_fraction + green_fraction = 2/3 := by
  sorry

end NUMINAMATH_GPT_cookie_division_l599_59935


namespace NUMINAMATH_GPT_sum_geometric_sequence_first_10_terms_l599_59958

theorem sum_geometric_sequence_first_10_terms :
  let a₁ : ℚ := 12
  let r : ℚ := 1 / 3
  let S₁₀ : ℚ := 12 * (1 - (1 / 3)^10) / (1 - 1 / 3)
  S₁₀ = 1062864 / 59049 := by
  sorry

end NUMINAMATH_GPT_sum_geometric_sequence_first_10_terms_l599_59958


namespace NUMINAMATH_GPT_number_line_distance_l599_59920

theorem number_line_distance (x : ℝ) : (abs (-3 - x) = 2) ↔ (x = -5 ∨ x = -1) :=
by
  sorry

end NUMINAMATH_GPT_number_line_distance_l599_59920


namespace NUMINAMATH_GPT_smallest_whole_number_l599_59985

theorem smallest_whole_number (m : ℕ) :
  m % 2 = 1 ∧
  m % 3 = 1 ∧
  m % 4 = 1 ∧
  m % 5 = 1 ∧
  m % 6 = 1 ∧
  m % 8 = 1 ∧
  m % 11 = 0 → 
  m = 1801 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_smallest_whole_number_l599_59985


namespace NUMINAMATH_GPT_problem_integer_pairs_l599_59927

theorem problem_integer_pairs (a b q r : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : a^2 + b^2 = q * (a + b) + r) (h4 : q^2 + r = 1977) :
    (a, b) = (50, 7) ∨ (a, b) = (50, 37) ∨ (a, b) = (7, 50) ∨ (a, b) = (37, 50) :=
sorry

end NUMINAMATH_GPT_problem_integer_pairs_l599_59927


namespace NUMINAMATH_GPT_greatest_drop_in_june_l599_59990

def monthly_changes := [("January", 1.50), ("February", -2.25), ("March", 0.75), ("April", -3.00), ("May", 1.00), ("June", -4.00)]

theorem greatest_drop_in_june : ∀ months : List (String × Float), (months = monthly_changes) → 
  (∃ month : String, 
    month = "June" ∧ 
    ∀ m p, m ≠ "June" → (m, p) ∈ months → p ≥ -4.00) :=
by
  sorry

end NUMINAMATH_GPT_greatest_drop_in_june_l599_59990


namespace NUMINAMATH_GPT_true_if_a_gt_1_and_b_gt_1_then_ab_gt_1_l599_59991

theorem true_if_a_gt_1_and_b_gt_1_then_ab_gt_1 (a b : ℝ) (ha : a > 1) (hb : b > 1) : ab > 1 :=
sorry

end NUMINAMATH_GPT_true_if_a_gt_1_and_b_gt_1_then_ab_gt_1_l599_59991


namespace NUMINAMATH_GPT_sequence_a_n_a_99_value_l599_59908

theorem sequence_a_n_a_99_value :
  ∃ (a : ℕ → ℝ), a 1 = 3 ∧ (∀ n, 2 * (a (n + 1)) - 2 * (a n) = 1) ∧ a 99 = 52 :=
by {
  sorry
}

end NUMINAMATH_GPT_sequence_a_n_a_99_value_l599_59908


namespace NUMINAMATH_GPT_exponent_product_to_sixth_power_l599_59970

theorem exponent_product_to_sixth_power :
  ∃ n : ℤ, 3^(12) * 3^(18) = n^6 ∧ n = 243 :=
by
  use 243
  sorry

end NUMINAMATH_GPT_exponent_product_to_sixth_power_l599_59970


namespace NUMINAMATH_GPT_files_remaining_correct_l599_59939

-- Definitions for the original number of files
def music_files_original : ℕ := 4
def video_files_original : ℕ := 21
def document_files_original : ℕ := 12
def photo_files_original : ℕ := 30
def app_files_original : ℕ := 7

-- Definitions for the number of deleted files
def video_files_deleted : ℕ := 15
def document_files_deleted : ℕ := 10
def photo_files_deleted : ℕ := 18
def app_files_deleted : ℕ := 3

-- Definitions for the remaining number of files
def music_files_remaining : ℕ := music_files_original
def video_files_remaining : ℕ := video_files_original - video_files_deleted
def document_files_remaining : ℕ := document_files_original - document_files_deleted
def photo_files_remaining : ℕ := photo_files_original - photo_files_deleted
def app_files_remaining : ℕ := app_files_original - app_files_deleted

-- The proof problem statement
theorem files_remaining_correct : 
  music_files_remaining + video_files_remaining + document_files_remaining + photo_files_remaining + app_files_remaining = 28 :=
by
  rw [music_files_remaining, video_files_remaining, document_files_remaining, photo_files_remaining, app_files_remaining]
  exact rfl


end NUMINAMATH_GPT_files_remaining_correct_l599_59939


namespace NUMINAMATH_GPT_only_zero_and_one_square_equal_themselves_l599_59992

theorem only_zero_and_one_square_equal_themselves (x: ℝ) : (x^2 = x) ↔ (x = 0 ∨ x = 1) :=
by sorry

end NUMINAMATH_GPT_only_zero_and_one_square_equal_themselves_l599_59992


namespace NUMINAMATH_GPT_inequality_holds_iff_m_range_l599_59925

theorem inequality_holds_iff_m_range (m : ℝ) : (∀ x : ℝ, m * x^2 - 2 * m * x - 3 < 0) ↔ (-3 < m ∧ m ≤ 0) :=
by
  sorry

end NUMINAMATH_GPT_inequality_holds_iff_m_range_l599_59925


namespace NUMINAMATH_GPT_correlation_coefficient_value_relation_between_gender_and_electric_car_expectation_X_value_l599_59914

-- Definition 1: Variance and regression coefficients and correlation coefficient calculation
noncomputable def correlation_coefficient : ℝ := 4.7 * (Real.sqrt (2 / 50))

-- Theorem 1: Correlation coefficient computation
theorem correlation_coefficient_value :
  correlation_coefficient = 0.94 :=
sorry

-- Definition 2: Chi-square calculation for independence test
noncomputable def chi_square : ℝ :=
  (100 * ((30 * 35 - 20 * 15)^2 : ℝ)) / (50 * 50 * 45 * 55)

-- Theorem 2: Chi-square test result
theorem relation_between_gender_and_electric_car :
  chi_square > 6.635 :=
sorry

-- Definition 3: Probability distribution and expectation calculation
def probability_distribution : Finset ℚ :=
{(21/55), (28/55), (6/55)}

noncomputable def expectation_X : ℚ :=
(0 * (21/55) + 1 * (28/55) + 2 * (6/55))

-- Theorem 3: Expectation of X calculation
theorem expectation_X_value :
  expectation_X = 8/11 :=
sorry

end NUMINAMATH_GPT_correlation_coefficient_value_relation_between_gender_and_electric_car_expectation_X_value_l599_59914


namespace NUMINAMATH_GPT_eval_expression_l599_59947

theorem eval_expression : (4^2 - 2^3) = 8 := by
  sorry

end NUMINAMATH_GPT_eval_expression_l599_59947


namespace NUMINAMATH_GPT_gcd_1113_1897_l599_59917

theorem gcd_1113_1897 : Int.gcd 1113 1897 = 7 := by
  sorry

end NUMINAMATH_GPT_gcd_1113_1897_l599_59917


namespace NUMINAMATH_GPT_negation_of_universal_statement_l599_59913

theorem negation_of_universal_statement :
  (¬ ∀ x : ℝ, x^4 - x^3 + x^2 + 5 ≤ 0) ↔ (∃ x : ℝ, x^4 - x^3 + x^2 + 5 > 0) :=
by sorry

end NUMINAMATH_GPT_negation_of_universal_statement_l599_59913


namespace NUMINAMATH_GPT_workers_days_not_worked_l599_59904

theorem workers_days_not_worked (W N : ℕ) (h1 : W + N = 30) (h2 : 100 * W - 25 * N = 0) : N = 24 :=
sorry

end NUMINAMATH_GPT_workers_days_not_worked_l599_59904


namespace NUMINAMATH_GPT_number_of_valid_pairs_l599_59960

theorem number_of_valid_pairs (m n : ℕ) (h1 : n > m) (h2 : 3 * (m - 4) * (n - 4) = m * n) : 
  (m, n) = (7, 18) ∨ (m, n) = (8, 12) ∨ (m, n) = (9, 10) ∨ (m-6) * (n-6) = 12 := sorry

end NUMINAMATH_GPT_number_of_valid_pairs_l599_59960


namespace NUMINAMATH_GPT_reciprocal_of_neg2019_l599_59938

theorem reciprocal_of_neg2019 : (1 / -2019) = - (1 / 2019) := 
by
  sorry

end NUMINAMATH_GPT_reciprocal_of_neg2019_l599_59938


namespace NUMINAMATH_GPT_horizontal_asymptote_of_rational_function_l599_59924

theorem horizontal_asymptote_of_rational_function :
  ∀ (x : ℝ), (y = (7 * x^2 - 5) / (4 * x^2 + 6 * x + 3)) → (∃ b : ℝ, b = 7 / 4) :=
by
  intro x y
  sorry

end NUMINAMATH_GPT_horizontal_asymptote_of_rational_function_l599_59924


namespace NUMINAMATH_GPT_correct_option_B_l599_59953

theorem correct_option_B (a b : ℝ) : (-a^2 * b^3)^2 = a^4 * b^6 := 
  sorry

end NUMINAMATH_GPT_correct_option_B_l599_59953


namespace NUMINAMATH_GPT_parents_without_fulltime_jobs_l599_59975

theorem parents_without_fulltime_jobs (total : ℕ) (mothers fathers full_time_mothers full_time_fathers : ℕ) 
(h1 : mothers = 2 * fathers / 3)
(h2 : full_time_mothers = 9 * mothers / 10)
(h3 : full_time_fathers = 3 * fathers / 4)
(h4 : mothers + fathers = total) :
(100 * (total - (full_time_mothers + full_time_fathers))) / total = 19 :=
by
  sorry

end NUMINAMATH_GPT_parents_without_fulltime_jobs_l599_59975


namespace NUMINAMATH_GPT_evaluate_expression_at_zero_l599_59994

theorem evaluate_expression_at_zero :
  ∀ x : ℝ, (x ≠ -1) ∧ (x ≠ 3) →
  ( (3 * x^2 - 2 * x + 1) / ((x + 1) * (x - 3)) - (5 + 2 * x) / ((x + 1) * (x - 3)) ) = 2 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_at_zero_l599_59994


namespace NUMINAMATH_GPT_Jonas_needs_to_buy_35_pairs_of_socks_l599_59984

theorem Jonas_needs_to_buy_35_pairs_of_socks
  (socks : ℕ)
  (shoes : ℕ)
  (pants : ℕ)
  (tshirts : ℕ)
  (double_items : ℕ)
  (needed_items : ℕ)
  (pairs_of_socks_needed : ℕ) :
  socks = 20 →
  shoes = 5 →
  pants = 10 →
  tshirts = 10 →
  double_items = 2 * (2 * socks + 2 * shoes + pants + tshirts) →
  needed_items = double_items - (2 * socks + 2 * shoes + pants + tshirts) →
  pairs_of_socks_needed = needed_items / 2 →
  pairs_of_socks_needed = 35 :=
by sorry

end NUMINAMATH_GPT_Jonas_needs_to_buy_35_pairs_of_socks_l599_59984


namespace NUMINAMATH_GPT_determine_1000g_weight_l599_59902

-- Define the weights
def weights : List ℕ := [1000, 1001, 1002, 1004, 1007]

-- Define the weight sets
def Group1 : List ℕ := [weights.get! 0, weights.get! 1]
def Group2 : List ℕ := [weights.get! 2, weights.get! 3]
def Group3 : List ℕ := [weights.get! 4]

-- Definition to choose the lighter group or determine equality
def lighterGroup (g1 g2 : List ℕ) : List ℕ :=
  if g1.sum = g2.sum then Group3 else if g1.sum < g2.sum then g1 else g2

-- Determine the 1000 g weight functionally
def identify1000gWeightUsing3Weighings : ℕ :=
  let firstWeighing := lighterGroup Group1 Group2
  if firstWeighing = Group3 then Group3.get! 0 else
  let remainingWeights := firstWeighing
  if remainingWeights.get! 0 = remainingWeights.get! 1 then Group3.get! 0
  else if remainingWeights.get! 0 < remainingWeights.get! 1 then remainingWeights.get! 0 else remainingWeights.get! 1

theorem determine_1000g_weight : identify1000gWeightUsing3Weighings = 1000 :=
sorry

end NUMINAMATH_GPT_determine_1000g_weight_l599_59902


namespace NUMINAMATH_GPT_tetrahedron_sum_l599_59922

theorem tetrahedron_sum :
  let edges := 6
  let corners := 4
  let faces := 4
  edges + corners + faces = 14 :=
by
  sorry

end NUMINAMATH_GPT_tetrahedron_sum_l599_59922


namespace NUMINAMATH_GPT_smallest_number_of_students_l599_59962

-- Define the conditions as given in the problem
def eight_to_six_ratio : ℕ × ℕ := (5, 3) -- ratio of 8th-graders to 6th-graders
def eight_to_nine_ratio : ℕ × ℕ := (7, 4) -- ratio of 8th-graders to 9th-graders

theorem smallest_number_of_students (a b c : ℕ)
  (h1 : a = 5 * b) (h2 : b = 3 * c) (h3 : a = 7 * c) : a + b + c = 76 := 
sorry

end NUMINAMATH_GPT_smallest_number_of_students_l599_59962


namespace NUMINAMATH_GPT_min_value_expression_l599_59987

theorem min_value_expression (x : ℝ) : 
  ∃ y : ℝ, (y = (x+2)*(x+3)*(x+4)*(x+5) + 3033) ∧ y ≥ 3032 ∧ 
  (∀ z : ℝ, (z = (x+2)*(x+3)*(x+4)*(x+5) + 3033) → z ≥ 3032) := 
sorry

end NUMINAMATH_GPT_min_value_expression_l599_59987


namespace NUMINAMATH_GPT_hockey_players_count_l599_59951

theorem hockey_players_count (cricket_players : ℕ) (football_players : ℕ) (softball_players : ℕ) (total_players : ℕ) 
(h_cricket : cricket_players = 16) 
(h_football : football_players = 18) 
(h_softball : softball_players = 13) 
(h_total : total_players = 59) : 
  total_players - (cricket_players + football_players + softball_players) = 12 := 
by sorry

end NUMINAMATH_GPT_hockey_players_count_l599_59951


namespace NUMINAMATH_GPT_cubic_common_roots_l599_59957

theorem cubic_common_roots:
  ∃ (c d : ℝ), 
  (∀ r s : ℝ,
    r ≠ s ∧ 
    (r ∈ {x : ℝ | x^3 + c * x^2 + 16 * x + 9 = 0}) ∧
    (s ∈ {x : ℝ | x^3 + c * x^2 + 16 * x + 9 = 0}) ∧ 
    (r ∈ {x : ℝ | x^3 + d * x^2 + 20 * x + 12 = 0}) ∧
    (s ∈ {x : ℝ | x^3 + d * x^2 + 20 * x + 12 = 0})) → 
  c = 8 ∧ d = 9 := 
by
  sorry

end NUMINAMATH_GPT_cubic_common_roots_l599_59957


namespace NUMINAMATH_GPT_freddy_travel_time_l599_59918

theorem freddy_travel_time (dist_A_B : ℝ) (time_Eddy : ℝ) (dist_A_C : ℝ) (speed_ratio : ℝ) (travel_time_Freddy : ℝ) :
  dist_A_B = 540 ∧ time_Eddy = 3 ∧ dist_A_C = 300 ∧ speed_ratio = 2.4 →
  travel_time_Freddy = dist_A_C / (dist_A_B / time_Eddy / speed_ratio) :=
  sorry

end NUMINAMATH_GPT_freddy_travel_time_l599_59918


namespace NUMINAMATH_GPT_max_single_player_salary_l599_59952

theorem max_single_player_salary (n : ℕ) (m : ℕ) (T : ℕ) (n_pos : n = 18) (m_pos : m = 20000) (T_pos : T = 800000) :
  ∃ x : ℕ, (∀ y : ℕ, y ≤ x → y ≤ 460000) ∧ (17 * m + x ≤ T) :=
by
  sorry

end NUMINAMATH_GPT_max_single_player_salary_l599_59952


namespace NUMINAMATH_GPT_alpha_minus_beta_l599_59937

-- Providing the conditions
variable (α β : ℝ)
variable (hα1 : 0 < α ∧ α < Real.pi / 2)
variable (hβ1 : 0 < β ∧ β < Real.pi / 2)
variable (hα2 : Real.tan α = 4 / 3)
variable (hβ2 : Real.tan β = 1 / 7)

-- The goal is to show that α - β = π / 4 given the conditions
theorem alpha_minus_beta :
  α - β = Real.pi / 4 := by
  sorry

end NUMINAMATH_GPT_alpha_minus_beta_l599_59937


namespace NUMINAMATH_GPT_find_numbers_l599_59986

theorem find_numbers (a b c : ℕ) (h₁ : 10 ≤ b ∧ b < 100) (h₂ : 10 ≤ c ∧ c < 100)
    (h₃ : 10^4 * a + 100 * b + c = (a + b + c)^3) : (a = 9 ∧ b = 11 ∧ c = 25) :=
by
  sorry

end NUMINAMATH_GPT_find_numbers_l599_59986


namespace NUMINAMATH_GPT_trivia_team_members_l599_59964

theorem trivia_team_members (n p s x y : ℕ) (h1 : n = 12) (h2 : p = 64) (h3 : s = 8) (h4 : x = p / s) (h5 : y = n - x) : y = 4 :=
by
  sorry

end NUMINAMATH_GPT_trivia_team_members_l599_59964


namespace NUMINAMATH_GPT_interior_angle_of_regular_hexagon_l599_59923

theorem interior_angle_of_regular_hexagon : 
  ∀ (n : ℕ), n = 6 → (∃ sumInteriorAngles : ℕ, sumInteriorAngles = (n - 2) * 180) →
  ∀ (interiorAngle : ℕ), (∃ sumInteriorAngles : ℕ, sumInteriorAngles = 720) → 
  interiorAngle = sumInteriorAngles / 6 →
  interiorAngle = 120 :=
by
  sorry

end NUMINAMATH_GPT_interior_angle_of_regular_hexagon_l599_59923


namespace NUMINAMATH_GPT_gcd_of_78_and_36_l599_59934

theorem gcd_of_78_and_36 : Int.gcd 78 36 = 6 := by
  sorry

end NUMINAMATH_GPT_gcd_of_78_and_36_l599_59934


namespace NUMINAMATH_GPT_are_naptime_l599_59996

def flight_duration := 11 * 60 + 20  -- in minutes

def time_spent_reading := 2 * 60      -- in minutes
def time_spent_watching_movies := 4 * 60  -- in minutes
def time_spent_eating_dinner := 30    -- in minutes
def time_spent_listening_to_radio := 40   -- in minutes
def time_spent_playing_games := 1 * 60 + 10   -- in minutes

def total_time_spent_on_activities := 
  time_spent_reading + 
  time_spent_watching_movies + 
  time_spent_eating_dinner + 
  time_spent_listening_to_radio + 
  time_spent_playing_games

def remaining_time := (flight_duration - total_time_spent_on_activities) / 60  -- in hours

theorem are_naptime : remaining_time = 3 := by
  sorry

end NUMINAMATH_GPT_are_naptime_l599_59996


namespace NUMINAMATH_GPT_can_construct_prism_with_fewer_than_20_shapes_l599_59932

/-
  We have 5 congruent unit cubes glued together to form complex shapes.
  4 of these cubes form a 4-unit high prism, and the fifth is attached to one of the inner cubes with a full face.
  Prove that we can construct a solid rectangular prism using fewer than 20 of these shapes.
-/

theorem can_construct_prism_with_fewer_than_20_shapes :
  ∃ (n : ℕ), n < 20 ∧ (∃ (length width height : ℕ), length * width * height = 5 * n) :=
sorry

end NUMINAMATH_GPT_can_construct_prism_with_fewer_than_20_shapes_l599_59932


namespace NUMINAMATH_GPT_remaining_digits_product_l599_59969

theorem remaining_digits_product (a b c : ℕ)
  (h1 : (a + b) % 10 = c % 10)
  (h2 : (b + c) % 10 = a % 10)
  (h3 : (c + a) % 10 = b % 10) :
  ((a * b * c) % 1000 = 0 ∨
   (a * b * c) % 1000 = 250 ∨
   (a * b * c) % 1000 = 500 ∨
   (a * b * c) % 1000 = 750) :=
sorry

end NUMINAMATH_GPT_remaining_digits_product_l599_59969


namespace NUMINAMATH_GPT_vector_perpendicular_iff_l599_59981

theorem vector_perpendicular_iff (k : ℝ) :
  let a := (Real.sqrt 3, 1)
  let b := (0, 1)
  let c := (k, Real.sqrt 3)
  let ab := (Real.sqrt 3, 3)  -- a + 2b
  a.1 * c.1 + ab.2 * c.2 = 0 → k = -3 :=
by
  let a := (Real.sqrt 3, 1)
  let b := (0, 1)
  let c := (k, Real.sqrt 3)
  let ab := (Real.sqrt 3, 3)  -- a + 2b
  intro h
  sorry

end NUMINAMATH_GPT_vector_perpendicular_iff_l599_59981


namespace NUMINAMATH_GPT_find_numbers_l599_59976

theorem find_numbers (x y : ℕ) (hx : 10 ≤ x ∧ x < 100) (hy : 10 ≤ y ∧ y < 100)
                     (hxy_mul : 2000 ≤ x * y ∧ x * y < 3000) (hxy_add : 100 ≤ x + y ∧ x + y < 1000)
                     (h_digit_relation : x * y = 2000 + x + y) : 
                     (x = 24 ∧ y = 88) ∨ (x = 88 ∧ y = 24) ∨ (x = 30 ∧ y = 70) ∨ (x = 70 ∧ y = 30) :=
by
  -- The proof will go here
  sorry

end NUMINAMATH_GPT_find_numbers_l599_59976


namespace NUMINAMATH_GPT_D_working_alone_completion_time_l599_59928

variable (A_rate D_rate : ℝ)
variable (A_job_hours D_job_hours : ℝ)

-- Conditions
def A_can_complete_in_15_hours : Prop := (A_job_hours = 15)
def A_and_D_together_complete_in_10_hours : Prop := (1/A_rate + 1/D_rate = 10)

-- Proof statement
theorem D_working_alone_completion_time
  (hA : A_job_hours = 15)
  (hAD : 1/A_rate + 1/D_rate = 10) :
  D_job_hours = 30 := sorry

end NUMINAMATH_GPT_D_working_alone_completion_time_l599_59928


namespace NUMINAMATH_GPT_number_is_composite_l599_59906

theorem number_is_composite : ∃ k l : ℕ, k * l = 53 * 83 * 109 + 40 * 66 * 96 ∧ k > 1 ∧ l > 1 :=
by
  have h1 : 53 + 96 = 149 := by norm_num
  have h2 : 83 + 66 = 149 := by norm_num
  have h3 : 109 + 40 = 149 := by norm_num
  sorry

end NUMINAMATH_GPT_number_is_composite_l599_59906


namespace NUMINAMATH_GPT_part1_part2_l599_59940

def P (x : ℝ) : Prop := |x - 1| > 2
def S (x : ℝ) (a : ℝ) : Prop := x^2 - (a + 1) * x + a > 0

theorem part1 (a : ℝ) (h : a = 2) : ∀ x, S x a ↔ x < 1 ∨ x > 2 :=
by
  sorry

theorem part2 (a : ℝ) (h : a ≠ 1) : ∀ x, (P x → S x a) → (-1 ≤ a ∧ a < 1) ∨ (1 < a ∧ a ≤ 3) :=
by
  sorry

end NUMINAMATH_GPT_part1_part2_l599_59940


namespace NUMINAMATH_GPT_vectors_orthogonal_x_value_l599_59980

theorem vectors_orthogonal_x_value :
  (∀ x : ℝ, (3 * x + 4 * (-7) = 0) → (x = 28 / 3)) := 
by 
  sorry

end NUMINAMATH_GPT_vectors_orthogonal_x_value_l599_59980


namespace NUMINAMATH_GPT_center_of_circle_l599_59954

theorem center_of_circle (x1 y1 x2 y2 : ℝ) (h1 : (x1, y1) = (3, 8)) (h2 : (x2, y2) = (11, -4)) :
  ((x1 + x2) / 2, (y1 + y2) / 2) = (7, 2) := by
  sorry

end NUMINAMATH_GPT_center_of_circle_l599_59954


namespace NUMINAMATH_GPT_sin_cos_product_neg_l599_59916

theorem sin_cos_product_neg (α : ℝ) (h : Real.tan α < 0) : Real.sin α * Real.cos α < 0 :=
sorry

end NUMINAMATH_GPT_sin_cos_product_neg_l599_59916


namespace NUMINAMATH_GPT_find_b_value_l599_59921

theorem find_b_value (a b : ℤ) (h₁ : a + 2 * b = 32) (h₂ : |a| > 2) (h₃ : a = 4) : b = 14 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_find_b_value_l599_59921


namespace NUMINAMATH_GPT_point_in_fourth_quadrant_l599_59999

variable (a : ℝ)

theorem point_in_fourth_quadrant (h : a < -1) : 
    let x := a^2 - 2*a - 1
    let y := (a + 1) / abs (a + 1)
    (x > 0) ∧ (y < 0) := 
by
  let x := a^2 - 2*a - 1
  let y := (a + 1) / abs (a + 1)
  sorry

end NUMINAMATH_GPT_point_in_fourth_quadrant_l599_59999


namespace NUMINAMATH_GPT_hal_battery_change_25th_time_l599_59973

theorem hal_battery_change_25th_time (months_in_year : ℕ) 
    (battery_interval : ℕ) 
    (first_change_month : ℕ) 
    (change_count : ℕ) : 
    (battery_interval * (change_count-1)) % months_in_year + first_change_month % months_in_year = first_change_month % months_in_year :=
by
    have h1 : months_in_year = 12 := by sorry
    have h2 : battery_interval = 5 := by sorry
    have h3 : first_change_month = 5 := by sorry -- May is represented by 5 (0 = January, 1 = February, ..., 4 = April, 5 = May, ...)
    have h4 : change_count = 25 := by sorry
    sorry

end NUMINAMATH_GPT_hal_battery_change_25th_time_l599_59973


namespace NUMINAMATH_GPT_Thelma_cuts_each_tomato_into_8_slices_l599_59946

-- Conditions given in the problem
def slices_per_meal := 20
def family_size := 8
def tomatoes_needed := 20

-- The quantity we want to prove
def slices_per_tomato := 8

-- Statement to be proven: Thelma cuts each green tomato into the correct number of slices
theorem Thelma_cuts_each_tomato_into_8_slices :
  (slices_per_meal * family_size) = (tomatoes_needed * slices_per_tomato) :=
by 
  sorry

end NUMINAMATH_GPT_Thelma_cuts_each_tomato_into_8_slices_l599_59946


namespace NUMINAMATH_GPT_discount_profit_percentage_l599_59901

theorem discount_profit_percentage (CP : ℝ) (P_no_discount : ℝ) (D : ℝ) (profit_with_discount : ℝ) (SP_no_discount : ℝ) (SP_discount : ℝ) :
  P_no_discount = 50 ∧ D = 4 ∧ SP_no_discount = CP + 0.5 * CP ∧ SP_discount = SP_no_discount - (D / 100) * SP_no_discount ∧ profit_with_discount = SP_discount - CP →
  (profit_with_discount / CP) * 100 = 44 :=
by sorry

end NUMINAMATH_GPT_discount_profit_percentage_l599_59901


namespace NUMINAMATH_GPT_Humphrey_birds_l599_59911

-- Definitions for the given conditions:
def Marcus_birds : ℕ := 7
def Darrel_birds : ℕ := 9
def average_birds : ℕ := 9
def number_of_people : ℕ := 3

-- Proof statement
theorem Humphrey_birds : ∀ x : ℕ, (average_birds * number_of_people = Marcus_birds + Darrel_birds + x) → x = 11 :=
by
  intro x h
  sorry

end NUMINAMATH_GPT_Humphrey_birds_l599_59911


namespace NUMINAMATH_GPT_mark_candy_bars_consumption_l599_59961

theorem mark_candy_bars_consumption 
  (recommended_intake : ℕ := 150)
  (soft_drink_calories : ℕ := 2500)
  (soft_drink_added_sugar_percent : ℕ := 5)
  (candy_bar_added_sugar_calories : ℕ := 25)
  (exceeded_percentage : ℕ := 100)
  (actual_intake := recommended_intake + (recommended_intake * exceeded_percentage / 100))
  (soft_drink_added_sugar := soft_drink_calories * soft_drink_added_sugar_percent / 100)
  (candy_bars_added_sugar := actual_intake - soft_drink_added_sugar)
  (number_of_bars := candy_bars_added_sugar / candy_bar_added_sugar_calories) : 
  number_of_bars = 7 := 
by
  sorry

end NUMINAMATH_GPT_mark_candy_bars_consumption_l599_59961


namespace NUMINAMATH_GPT_monotonic_intervals_range_of_c_l599_59978

noncomputable def f (x : ℝ) (b c : ℝ) : ℝ := c * Real.log x + (1 / 2) * x ^ 2 + b * x

lemma extreme_point_condition {b c : ℝ} (h1 : c ≠ 0) (h2 : f 1 b c = 0) : b + c + 1 = 0 :=
sorry

theorem monotonic_intervals (b c : ℝ) (h1 : c ≠ 0) (h2 : f 1 b c = 0) (h3 : c > 1) :
  (∀ x, 0 < x ∧ x < 1 → f 1 b c < f x b c) ∧ 
  (∀ x, 1 < x ∧ x < c → f 1 b c > f x b c) ∧ 
  (∀ x, x > c → f 1 b c < f x b c) :=
sorry

theorem range_of_c (b c : ℝ) (h1 : c ≠ 0) (h2 : f 1 b c = 0) (h3 : (f 1 b c < 0)) :
  -1 / 2 < c ∧ c < 0 :=
sorry

end NUMINAMATH_GPT_monotonic_intervals_range_of_c_l599_59978


namespace NUMINAMATH_GPT_intersection_A_B_l599_59900

-- Define the sets A and B based on given conditions
def A : Set ℝ := { x | x^2 ≤ 1 }
def B : Set ℝ := { x | (x - 2) / x ≤ 0 }

-- State the proof problem
theorem intersection_A_B : A ∩ B = { x | 0 < x ∧ x ≤ 1 } :=
by
  sorry

end NUMINAMATH_GPT_intersection_A_B_l599_59900


namespace NUMINAMATH_GPT_find_pairs_l599_59931

def isDivisible (m n : ℕ) : Prop := ∃ k : ℕ, m = k * n
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n
def satisfiesConditions (a b : ℕ) : Prop :=
  (isDivisible (a^2 + 6 * a + 8) b ∧
  (a^2 + a * b - 6 * b^2 - 15 * b - 9 = 0) ∧
  ¬ (a + 2 * b + 2) % 4 = 0 ∧
  isPrime (a + 6 * b + 2)) ∨
  (isDivisible (a^2 + 6 * a + 8) b ∧
  (a^2 + a * b - 6 * b^2 - 15 * b - 9 = 0) ∧
  ¬ (a + 2 * b + 2) % 4 = 0 ∧
  ¬ isPrime (a + 6 * b + 2))

theorem find_pairs (a b : ℕ) :
  (a = 5 ∧ b = 1) ∨ 
  (a = 17 ∧ b = 7) → 
  satisfiesConditions a b :=
by
  -- Proof to be completed
  sorry

end NUMINAMATH_GPT_find_pairs_l599_59931


namespace NUMINAMATH_GPT_sqrt_of_0_01_l599_59903

theorem sqrt_of_0_01 : Real.sqrt 0.01 = 0.1 :=
by
  sorry

end NUMINAMATH_GPT_sqrt_of_0_01_l599_59903


namespace NUMINAMATH_GPT_linear_function_solution_l599_59963

theorem linear_function_solution (f : ℝ → ℝ) (h1 : ∀ x, f (f x) = 16 * x - 15) :
  (∀ x, f x = 4 * x - 3) ∨ (∀ x, f x = -4 * x + 5) :=
sorry

end NUMINAMATH_GPT_linear_function_solution_l599_59963
