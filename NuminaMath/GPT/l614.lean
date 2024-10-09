import Mathlib

namespace calculate_g_l614_61426

def g (a b c : ℚ) : ℚ := (2 * c + a) / (b - c)

theorem calculate_g : g 3 6 (-1) = 1 / 7 :=
by
    -- Proof is not included
    sorry

end calculate_g_l614_61426


namespace divisible_by_five_l614_61425

theorem divisible_by_five (x y z : ℤ) (hxy : x ≠ y) (hyz : y ≠ z) (hzx : z ≠ x) :
  ∃ k : ℤ, (x-y)^5 + (y-z)^5 + (z-x)^5 = 5 * k * (y-z) * (z-x) * (x-y) :=
  sorry

end divisible_by_five_l614_61425


namespace find_m_l614_61411

-- Define the conditions
def parabola_eq (m : ℝ) (x y : ℝ) : Prop := x^2 = m * y
def vertex_to_directrix_dist (d : ℝ) : Prop := d = 1 / 2

-- State the theorem
theorem find_m (m : ℝ) (x y d : ℝ) 
  (h1 : parabola_eq m x y) 
  (h2 : vertex_to_directrix_dist d) :
  m = 2 :=
by
  sorry

end find_m_l614_61411


namespace ratio_red_to_black_l614_61422

theorem ratio_red_to_black (a b x : ℕ) (h1 : x + b = 3 * a) (h2 : x = 2 * b - 3 * a) :
  a / b = 1 / 2 := by
  sorry

end ratio_red_to_black_l614_61422


namespace trains_time_distance_l614_61468

-- Define the speeds of the two trains
def speed1 : ℕ := 11
def speed2 : ℕ := 31

-- Define the distance between the two trains after time t
def distance_between_trains (t : ℕ) : ℕ :=
  speed2 * t - speed1 * t

-- Define the condition that this distance is 160 miles
def condition (t : ℕ) : Prop :=
  distance_between_trains t = 160

-- State the theorem to prove
theorem trains_time_distance : ∃ t : ℕ, condition t ∧ t = 8 :=
by
  use 8
  unfold condition
  unfold distance_between_trains
  -- Verifying the calculated distance
  sorry

end trains_time_distance_l614_61468


namespace probability_a_2b_3c_gt_5_l614_61494

def isInUnitCube (a b c : ℝ) : Prop :=
  0 ≤ a ∧ a ≤ 1 ∧ 0 ≤ b ∧ b ≤ 1 ∧ 0 ≤ c ∧ c ≤ 1

theorem probability_a_2b_3c_gt_5 (a b c : ℝ) :
  isInUnitCube a b c → ¬(a + 2 * b + 3 * c > 5) :=
by
  intro h
  -- The proof goes here, currently using sorry as placeholder
  sorry

end probability_a_2b_3c_gt_5_l614_61494


namespace five_eight_sided_dice_not_all_same_l614_61407

noncomputable def probability_not_all_same : ℚ :=
  let total_outcomes := 8^5
  let same_number_outcomes := 8
  1 - (same_number_outcomes / total_outcomes)

theorem five_eight_sided_dice_not_all_same :
  probability_not_all_same = 4095 / 4096 :=
by
  sorry

end five_eight_sided_dice_not_all_same_l614_61407


namespace find_solutions_l614_61486

theorem find_solutions (a m n : ℕ) (h : a > 0) (h₁ : m > 0) (h₂ : n > 0) :
  (a^m + 1) ∣ (a + 1)^n → 
  ((a = 1 ∧ True) ∨ (True ∧ m = 1) ∨ (a = 2 ∧ m = 3 ∧ n ≥ 2)) :=
by sorry

end find_solutions_l614_61486


namespace brian_tape_needed_l614_61440

-- Define lengths and number of each type of box
def long_side_15_30 := 32
def short_side_15_30 := 17
def num_15_30 := 5

def side_40_40 := 42
def num_40_40 := 2

def long_side_20_50 := 52
def short_side_20_50 := 22
def num_20_50 := 3

-- Calculate the total tape required
def total_tape : Nat :=
  (num_15_30 * (long_side_15_30 + 2 * short_side_15_30)) +
  (num_40_40 * (3 * side_40_40)) +
  (num_20_50 * (long_side_20_50 + 2 * short_side_20_50))

-- Proof statement
theorem brian_tape_needed : total_tape = 870 := by
  sorry

end brian_tape_needed_l614_61440


namespace car_speed_first_hour_l614_61477

theorem car_speed_first_hour (x : ℝ) (h : (79 = (x + 60) / 2)) : x = 98 :=
by {
  sorry
}

end car_speed_first_hour_l614_61477


namespace find_a6_l614_61496

noncomputable def a (n : ℕ) : ℝ := sorry

axiom geom_seq_inc :
  ∀ n : ℕ, a n < a (n + 1)

axiom root_eqn_a2_a4 :
  ∃ a2 a4 : ℝ, (a 2 = a2) ∧ (a 4 = a4) ∧ (a2^2 - 6 * a2 + 5 = 0) ∧ (a4^2 - 6 * a4 + 5 = 0)

theorem find_a6 : a 6 = 25 := 
sorry

end find_a6_l614_61496


namespace triangle_inequality_l614_61434

theorem triangle_inequality (ABC: Triangle) (M : Point) (a b c : ℝ)
  (h1 : a = BC) (h2 : b = CA) (h3 : c = AB) :
  (1 / a^2) + (1 / b^2) + (1 / c^2) ≥ 3 / (MA^2 + MB^2 + MC^2) := 
sorry

end triangle_inequality_l614_61434


namespace find_n_l614_61454

theorem find_n (n : ℕ) : (256 : ℝ) ^ (1 / 4 : ℝ) = 4 ^ n → 256 = (4 ^ 4 : ℝ) → n = 1 :=
by
  intros h₁ h₂
  sorry

end find_n_l614_61454


namespace polynomial_has_roots_l614_61460

theorem polynomial_has_roots :
  ∃ x : ℝ, x ∈ [-4, -3, -1, 2] ∧ (x^4 + 6 * x^3 + 7 * x^2 - 14 * x - 12 = 0) :=
by
  sorry

end polynomial_has_roots_l614_61460


namespace jade_handled_84_transactions_l614_61428

def Mabel_transactions : ℕ := 90

def Anthony_transactions (mabel : ℕ) : ℕ := mabel + mabel / 10

def Cal_transactions (anthony : ℕ) : ℕ := (2 * anthony) / 3

def Jade_transactions (cal : ℕ) : ℕ := cal + 18

theorem jade_handled_84_transactions :
  Jade_transactions (Cal_transactions (Anthony_transactions Mabel_transactions)) = 84 := 
sorry

end jade_handled_84_transactions_l614_61428


namespace remainder_of_prime_powers_l614_61474

theorem remainder_of_prime_powers (p q : ℕ) (hp : Prime p) (hq : Prime q) (hpq : p ≠ q) :
  (p^(q-1) + q^(p-1)) % (p * q) = 1 := 
sorry

end remainder_of_prime_powers_l614_61474


namespace find_part_length_in_inches_find_part_length_in_feet_and_inches_l614_61412

def feetToInches (feet : ℕ) : ℕ := feet * 12

def totalLengthInInches (feet : ℕ) (inches : ℕ) : ℕ := feetToInches feet + inches

def partLengthInInches (totalLength : ℕ) (parts : ℕ) : ℕ := totalLength / parts

def inchesToFeetAndInches (inches : ℕ) : Nat × Nat := (inches / 12, inches % 12)

theorem find_part_length_in_inches (feet : ℕ) (inches : ℕ) (parts : ℕ)
    (h1 : feet = 10) (h2 : inches = 5) (h3 : parts = 5) :
    partLengthInInches (totalLengthInInches feet inches) parts = 25 := by
  sorry

theorem find_part_length_in_feet_and_inches (feet : ℕ) (inches : ℕ) (parts : ℕ)
    (h1 : feet = 10) (h2 : inches = 5) (h3 : parts = 5) :
    inchesToFeetAndInches (partLengthInInches (totalLengthInInches feet inches) parts) = (2, 1) := by
  sorry

end find_part_length_in_inches_find_part_length_in_feet_and_inches_l614_61412


namespace sum_squares_inequality_l614_61484

theorem sum_squares_inequality (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) 
(h_sum : x + y + z = 3) : 
  1 / (x^2 + y + z) + 1 / (x + y^2 + z) + 1 / (x + y + z^2) ≤ 1 := 
by 
  sorry

end sum_squares_inequality_l614_61484


namespace sum_of_ages_l614_61401

variables (Matthew Rebecca Freddy: ℕ)
variables (H1: Matthew = Rebecca + 2)
variables (H2: Matthew = Freddy - 4)
variables (H3: Freddy = 15)

theorem sum_of_ages
  (H1: Matthew = Rebecca + 2)
  (H2: Matthew = Freddy - 4)
  (H3: Freddy = 15):
  Matthew + Rebecca + Freddy = 35 :=
  sorry

end sum_of_ages_l614_61401


namespace compound_cost_correct_l614_61438

noncomputable def compound_cost_per_pound (limestone_cost shale_mix_cost : ℝ) (total_weight limestone_weight : ℝ) : ℝ :=
  let shale_mix_weight := total_weight - limestone_weight
  let total_cost := (limestone_weight * limestone_cost) + (shale_mix_weight * shale_mix_cost)
  total_cost / total_weight

theorem compound_cost_correct :
  compound_cost_per_pound 3 5 100 37.5 = 4.25 := by
  sorry

end compound_cost_correct_l614_61438


namespace polynomial_divisibility_l614_61403

def poly1 (x : ℝ) (k : ℝ) : ℝ := 3*x^3 - 9*x^2 + k*x - 12

theorem polynomial_divisibility (k : ℝ) :
  (∀ (x : ℝ), poly1 x k = (x - 3) * (3*x^2 + 4)) → (poly1 3 k = 0) := sorry

end polynomial_divisibility_l614_61403


namespace number_of_meters_sold_l614_61483

-- Define the given conditions
def price_per_meter : ℕ := 436 -- in kopecks
def total_revenue_end : ℕ := 728 -- in kopecks
def max_total_revenue : ℕ := 50000 -- in kopecks

-- State the problem formally in Lean 4
theorem number_of_meters_sold (x : ℕ) :
  price_per_meter * x ≡ total_revenue_end [MOD 1000] ∧
  price_per_meter * x ≤ max_total_revenue →
  x = 98 :=
sorry

end number_of_meters_sold_l614_61483


namespace like_terms_mn_l614_61437

theorem like_terms_mn (m n : ℤ) 
  (H1 : m - 2 = 3) 
  (H2 : n + 2 = 1) : 
  m * n = -5 := 
by
  sorry

end like_terms_mn_l614_61437


namespace part1_part2_part3_l614_61480

-- Definitions for the given functions
def y1 (x : ℝ) : ℝ := -x + 1
def y2 (x : ℝ) : ℝ := -3 * x + 2

-- Part (1)
theorem part1 (a : ℝ) : (∃ x : ℝ, y1 x = a + y2 x ∧ x > 0) ↔ (a > -1) := sorry

-- Part (2)
theorem part2 (x y : ℝ) (h1 : y = y1 x) (h2 : y = y2 x) : 12*x^2 + 12*x*y + 3*y^2 = 27/4 := sorry

-- Part (3)
theorem part3 (A B : ℝ) (x : ℝ) (h : (4 - 2 * x) / ((3 * x - 2) * (x - 1)) = A / y1 x + B / y2 x) : (A / B + B / A) = -17 / 4 := sorry

end part1_part2_part3_l614_61480


namespace certain_number_is_51_l614_61423

theorem certain_number_is_51 (G C : ℤ) 
  (h1 : G = 33) 
  (h2 : 3 * G = 2 * C - 3) : 
  C = 51 := 
by
  sorry

end certain_number_is_51_l614_61423


namespace cost_price_percentage_l614_61453

theorem cost_price_percentage (SP CP : ℝ) (hp : SP - CP = (1/3) * CP) : CP = 0.75 * SP :=
by
  sorry

end cost_price_percentage_l614_61453


namespace first_candidate_percentage_l614_61450

noncomputable
def passing_marks_approx : ℝ := 240

noncomputable
def total_marks (P : ℝ) : ℝ := (P + 30) / 0.45

noncomputable
def percentage_marks (T P : ℝ) : ℝ := ((P - 60) / T) * 100

theorem first_candidate_percentage :
  let P := passing_marks_approx
  let T := total_marks P
  percentage_marks T P = 30 :=
by
  sorry

end first_candidate_percentage_l614_61450


namespace maximum_value_F_l614_61409

noncomputable def f (x : Real) : Real := Real.sin x + Real.cos x

noncomputable def f' (x : Real) : Real := Real.cos x - Real.sin x

noncomputable def F (x : Real) : Real := f x * f' x + (f x) ^ 2

theorem maximum_value_F : ∃ x : Real, F x = 1 + Real.sqrt 2 :=
by
  -- The proof steps are to be added here.
  sorry

end maximum_value_F_l614_61409


namespace calories_remaining_for_dinner_l614_61482

theorem calories_remaining_for_dinner :
  let daily_limit := 2200
  let breakfast := 353
  let lunch := 885
  let snack := 130
  let total_consumed := breakfast + lunch + snack
  let remaining_for_dinner := daily_limit - total_consumed
  remaining_for_dinner = 832 := by
  sorry

end calories_remaining_for_dinner_l614_61482


namespace exists_231_four_digit_integers_l614_61451

theorem exists_231_four_digit_integers (n : ℕ) : 
  (∃ A B C D : ℕ, 
     A ≠ 0 ∧ 
     1 ≤ A ∧ A ≤ 9 ∧ 
     0 ≤ B ∧ B ≤ 9 ∧ 
     0 ≤ C ∧ C ≤ 9 ∧ 
     0 ≤ D ∧ D ≤ 9 ∧ 
     999 * (A - D) + 90 * (B - C) = n^3) ↔ n = 231 :=
by sorry

end exists_231_four_digit_integers_l614_61451


namespace log10_cubic_solution_l614_61413

noncomputable def log10 (x: ℝ) : ℝ := Real.log x / Real.log 10

open Real

theorem log10_cubic_solution 
  (x : ℝ) 
  (hx1 : x < 1) 
  (hx2 : (log10 x)^3 - log10 (x^4) = 640) : 
  (log10 x)^4 - log10 (x^4) = 645 := 
by 
  sorry

end log10_cubic_solution_l614_61413


namespace pentagon_area_greater_than_square_third_l614_61467

theorem pentagon_area_greater_than_square_third (a b : ℝ) :
  a^2 + (a * b) / 4 + (Real.sqrt 3 / 4) * b^2 > ((a + b)^2) / 3 :=
by
  sorry

end pentagon_area_greater_than_square_third_l614_61467


namespace geo_seq_product_l614_61427

theorem geo_seq_product (a : ℕ → ℝ) (r : ℝ) (h_pos : ∀ n, 0 < a n) 
  (h_geom : ∀ n, a (n + 1) = a n * r) (h_a1a9 : a 1 * a 9 = 16) :
  a 2 * a 5 * a 8 = 64 :=
sorry

end geo_seq_product_l614_61427


namespace solution_unique_s_l614_61469

theorem solution_unique_s (s : ℝ) (hs : ⌊s⌋ + s = 22.7) : s = 11.7 :=
sorry

end solution_unique_s_l614_61469


namespace wipes_per_pack_l614_61431

theorem wipes_per_pack (days : ℕ) (wipes_per_day : ℕ) (packs : ℕ) (total_wipes : ℕ) (n : ℕ)
    (h1 : days = 360)
    (h2 : wipes_per_day = 2)
    (h3 : packs = 6)
    (h4 : total_wipes = wipes_per_day * days)
    (h5 : total_wipes = n * packs) : 
    n = 120 := 
by 
  sorry

end wipes_per_pack_l614_61431


namespace cubes_with_even_red_faces_count_l614_61463

def block_dimensions : ℕ × ℕ × ℕ := (6, 4, 2)
def is_painted_red : Prop := true
def total_cubes : ℕ := 48
def cubes_with_even_red_faces : ℕ := 24

theorem cubes_with_even_red_faces_count :
  ∀ (dimensions : ℕ × ℕ × ℕ) (painted_red : Prop) (cubes_count : ℕ), 
  dimensions = block_dimensions → painted_red = is_painted_red → cubes_count = total_cubes → 
  (cubes_with_even_red_faces = 24) :=
by intros dimensions painted_red cubes_count h1 h2 h3; exact sorry

end cubes_with_even_red_faces_count_l614_61463


namespace rational_abs_eq_l614_61475

theorem rational_abs_eq (a : ℚ) (h : |-3 - a| = 3 + |a|) : 0 ≤ a := 
by
  sorry

end rational_abs_eq_l614_61475


namespace original_inhabitants_l614_61442

theorem original_inhabitants (X : ℝ) 
  (h1 : 10 ≤ X) 
  (h2 : 0.9 * X * 0.75 + 0.225 * X * 0.15 = 5265) : 
  X = 7425 := 
sorry

end original_inhabitants_l614_61442


namespace number_of_revolutions_wheel_half_mile_l614_61462

theorem number_of_revolutions_wheel_half_mile :
  let diameter := 10 * (1 : ℝ)
  let radius := diameter / 2
  let circumference := 2 * Real.pi * radius
  let half_mile_in_feet := 2640
  (half_mile_in_feet / circumference) = 264 / Real.pi :=
by
  let diameter := 10 * (1 : ℝ)
  let radius := diameter / 2
  let circumference := 2 * Real.pi * radius
  let half_mile_in_feet := 2640
  have h : (half_mile_in_feet / circumference) = 264 / Real.pi := by
    sorry
  exact h

end number_of_revolutions_wheel_half_mile_l614_61462


namespace solve_abs_inequality_l614_61478

theorem solve_abs_inequality (x : ℝ) :
  (|x-2| ≥ |x|) → x ≤ 1 :=
by
  sorry

end solve_abs_inequality_l614_61478


namespace afternoon_pear_sales_l614_61449

theorem afternoon_pear_sales (morning_sales afternoon_sales total_sales : ℕ)
  (h1 : afternoon_sales = 2 * morning_sales)
  (h2 : total_sales = morning_sales + afternoon_sales)
  (h3 : total_sales = 420) : 
  afternoon_sales = 280 :=
by {
  -- placeholders for the proof
  sorry 
}

end afternoon_pear_sales_l614_61449


namespace smallest_c_value_l614_61456

theorem smallest_c_value (c d : ℝ) (h_nonneg_c : 0 ≤ c) (h_nonneg_d : 0 ≤ d)
  (h_eq_cos : ∀ x : ℤ, Real.cos (c * x - d) = Real.cos (35 * x)) :
  c = 35 := by
  sorry

end smallest_c_value_l614_61456


namespace unique_B_for_A47B_divisible_by_7_l614_61489

-- Define the conditions
def A : ℕ := 4

-- Define the main proof problem statement
theorem unique_B_for_A47B_divisible_by_7 : 
  ∃! B : ℕ, B ≤ 9 ∧ (100 * A + 70 + B) % 7 = 0 :=
        sorry

end unique_B_for_A47B_divisible_by_7_l614_61489


namespace betty_bracelets_l614_61424

theorem betty_bracelets : (140 / 14) = 10 := 
by
  norm_num

end betty_bracelets_l614_61424


namespace problem_statement_l614_61464

variables (x y : ℚ)

theorem problem_statement 
  (h1 : x + y = 8 / 15) 
  (h2 : x - y = 1 / 105) : 
  x^2 - y^2 = 8 / 1575 :=
sorry

end problem_statement_l614_61464


namespace proof_problem_l614_61439

theorem proof_problem (x y z : ℝ) (h₁ : x ≠ y) 
  (h₂ : (x^2 - y*z) / (x * (1 - y*z)) = (y^2 - x*z) / (y * (1 - x*z))) :
  x + y + z = 1/x + 1/y + 1/z :=
sorry

end proof_problem_l614_61439


namespace problem1_problem2_l614_61476

-- Definitions
def total_questions := 5
def multiple_choice := 3
def true_false := 2
def total_outcomes := total_questions * (total_questions - 1)

-- (1) Probability of A drawing a true/false question and B drawing a multiple-choice question
def favorable_outcomes_1 := true_false * multiple_choice

-- (2) Probability of at least one of A or B drawing a multiple-choice question
def unfavorable_outcomes_2 := true_false * (true_false - 1)

-- Statements to be proved
theorem problem1 : favorable_outcomes_1 / total_outcomes = 3 / 10 := by sorry

theorem problem2 : 1 - (unfavorable_outcomes_2 / total_outcomes) = 9 / 10 := by sorry

end problem1_problem2_l614_61476


namespace find_f_one_seventh_l614_61400

-- Define the function f
variable (f : ℝ → ℝ)

-- Given conditions
variable (monotonic_f : MonotonicOn f (Set.Ioi 0))
variable (h : ∀ x ∈ Set.Ioi (0 : ℝ), f (f x - 1 / x) = 2)

-- Define the domain
variable (x : ℝ)
variable (hx : x ∈ Set.Ioi (0 : ℝ))

-- The theorem to prove
theorem find_f_one_seventh : f (1 / 7) = 8 := by
  -- proof starts here
  sorry

end find_f_one_seventh_l614_61400


namespace puppies_given_l614_61470

-- Definitions of the initial and left numbers of puppies
def initial_puppies : ℕ := 7
def left_puppies : ℕ := 2

-- Theorem stating that the number of puppies given to friends is the difference
theorem puppies_given : initial_puppies - left_puppies = 5 := by
  sorry -- Proof not required, so we use sorry

end puppies_given_l614_61470


namespace alice_commute_distance_l614_61455

noncomputable def office_distance_commute (commute_time_regular commute_time_holiday : ℝ) (speed_increase : ℝ) : ℝ := 
  let v := commute_time_regular * ((commute_time_regular + speed_increase) / commute_time_holiday - speed_increase)
  commute_time_regular * v

theorem alice_commute_distance : 
  office_distance_commute 0.5 0.3 12 = 9 := 
sorry

end alice_commute_distance_l614_61455


namespace solution_set_f_less_x_plus_1_l614_61495

open Real

noncomputable def f (x : ℝ) : ℝ := sorry

axiom f_continuous : Continuous f
axiom f_at_1 : f 1 = 2
axiom f_derivative : ∀ x, deriv f x < 1

theorem solution_set_f_less_x_plus_1 : 
  ∀ x : ℝ, (f x < x + 1) ↔ (x > 1) :=
by
  sorry

end solution_set_f_less_x_plus_1_l614_61495


namespace initial_amount_l614_61448

theorem initial_amount (spent_sweets friends_each left initial : ℝ) 
  (h1 : spent_sweets = 3.25) (h2 : friends_each = 2.20) (h3 : left = 2.45) :
  initial = spent_sweets + (friends_each * 2) + left :=
by
  sorry

end initial_amount_l614_61448


namespace product_of_roots_l614_61457

theorem product_of_roots :
  ∀ α β : ℝ, (Polynomial.roots (Polynomial.C 2 * Polynomial.X^2 + Polynomial.C 5 * Polynomial.X + Polynomial.C (-12))).prod = -6 :=
by
  sorry

end product_of_roots_l614_61457


namespace meal_service_count_l614_61471

/-- Define the number of people -/
def people_count : ℕ := 10

/-- Define the number of people that order pasta -/
def pasta_count : ℕ := 5

/-- Define the number of people that order salad -/
def salad_count : ℕ := 5

/-- Combination function to choose 2 people from 10 -/
def choose_2_from_10 : ℕ := Nat.choose 10 2

/-- Number of derangements of 8 people where exactly 2 people receive their correct meals -/
def derangement_8 : ℕ := 21

/-- Number of ways to correctly serve the meals where exactly 2 people receive the correct meal -/
theorem meal_service_count :
  choose_2_from_10 * derangement_8 = 945 :=
  by sorry

end meal_service_count_l614_61471


namespace rectangular_plot_breadth_l614_61441

theorem rectangular_plot_breadth:
  ∀ (b l : ℝ), (l = b + 10) → (24 * b = l * b) → b = 14 :=
by
  intros b l hl hs
  sorry

end rectangular_plot_breadth_l614_61441


namespace combined_selling_price_correct_l614_61481

noncomputable def cost_A : ℝ := 500
noncomputable def cost_B : ℝ := 800
noncomputable def profit_A_perc : ℝ := 0.10
noncomputable def profit_B_perc : ℝ := 0.15
noncomputable def tax_perc : ℝ := 0.05
noncomputable def packaging_fee : ℝ := 50

-- Calculating selling prices before tax and fees
noncomputable def selling_price_A_before_tax_fees : ℝ := cost_A * (1 + profit_A_perc)
noncomputable def selling_price_B_before_tax_fees : ℝ := cost_B * (1 + profit_B_perc)

-- Calculating taxes
noncomputable def tax_A : ℝ := selling_price_A_before_tax_fees * tax_perc
noncomputable def tax_B : ℝ := selling_price_B_before_tax_fees * tax_perc

-- Adding tax to selling prices
noncomputable def selling_price_A_incl_tax : ℝ := selling_price_A_before_tax_fees + tax_A
noncomputable def selling_price_B_incl_tax : ℝ := selling_price_B_before_tax_fees + tax_B

-- Adding packaging and shipping fees
noncomputable def final_selling_price_A : ℝ := selling_price_A_incl_tax + packaging_fee
noncomputable def final_selling_price_B : ℝ := selling_price_B_incl_tax + packaging_fee

-- Combined selling price
noncomputable def combined_selling_price : ℝ := final_selling_price_A + final_selling_price_B

theorem combined_selling_price_correct : 
  combined_selling_price = 1643.5 := by
  sorry

end combined_selling_price_correct_l614_61481


namespace scientific_notation_of_3900000000_l614_61492

theorem scientific_notation_of_3900000000 : 3900000000 = 3.9 * 10^9 :=
by 
  sorry

end scientific_notation_of_3900000000_l614_61492


namespace plants_needed_correct_l614_61490

def total_plants_needed (ferns palms succulents total_desired : ℕ) : ℕ :=
 total_desired - (ferns + palms + succulents)

theorem plants_needed_correct : total_plants_needed 3 5 7 24 = 9 := by
  sorry

end plants_needed_correct_l614_61490


namespace angle_RPS_is_1_degree_l614_61497

-- Definitions of the given angles
def angle_QRS : ℝ := 150
def angle_PQS : ℝ := 60
def angle_PSQ : ℝ := 49
def angle_QPR : ℝ := 70

-- Definition for the calculated angle QPS
def angle_QPS : ℝ := 180 - angle_PQS - angle_PSQ

-- Definition for the target angle RPS
def angle_RPS : ℝ := angle_QPS - angle_QPR

-- The theorem we aim to prove
theorem angle_RPS_is_1_degree : angle_RPS = 1 := by
  sorry

end angle_RPS_is_1_degree_l614_61497


namespace find_cos_minus_sin_l614_61461

variable (θ : ℝ)
variable (h1 : θ ∈ Set.Ioo (3 * Real.pi / 4) Real.pi)
variable (h2 : Real.sin θ * Real.cos θ = -Real.sqrt 3 / 2)

theorem find_cos_minus_sin : Real.cos θ - Real.sin θ = -Real.sqrt (1 + Real.sqrt 3) := by
  sorry

end find_cos_minus_sin_l614_61461


namespace b_joined_after_a_l614_61405

def months_b_joined (a_investment : ℕ) (b_investment : ℕ) (profit_ratio : ℕ × ℕ) (total_months : ℕ) : ℕ :=
  let a_months := total_months
  let b_months := total_months - (b_investment / (3500 * profit_ratio.snd / profit_ratio.fst / b_investment))
  total_months - b_months

theorem b_joined_after_a (a_investment b_investment total_months : ℕ) (profit_ratio : ℕ × ℕ) (h_a_investment : a_investment = 3500)
   (h_b_investment : b_investment = 21000) (h_profit_ratio : profit_ratio = (2, 3)) : months_b_joined a_investment b_investment profit_ratio total_months = 9 := by
  sorry

end b_joined_after_a_l614_61405


namespace hiking_packing_weight_l614_61485

theorem hiking_packing_weight
  (miles_per_hour : ℝ)
  (hours_per_day : ℝ)
  (days : ℝ)
  (supply_per_mile : ℝ)
  (resupply_fraction : ℝ)
  (expected_first_pack_weight : ℝ)
  (hiking_hours : ℝ := hours_per_day * days)
  (total_miles : ℝ := miles_per_hour * hiking_hours)
  (total_weight_needed : ℝ := supply_per_mile * total_miles)
  (resupply_weight : ℝ := resupply_fraction * total_weight_needed)
  (first_pack_weight : ℝ := total_weight_needed - resupply_weight) :
  first_pack_weight = 37.5 :=
by
  -- The proof goes here, but is omitted since the proof is not required.
  sorry

end hiking_packing_weight_l614_61485


namespace smallest_n_l614_61498

theorem smallest_n (n : ℕ) (h1 : 100 ≤ n ∧ n < 1000)
  (h2 : n % 9 = 2)
  (h3 : n % 6 = 4) : n = 146 :=
sorry

end smallest_n_l614_61498


namespace total_time_is_correct_l614_61465

-- Definitions based on conditions
def timeouts_for_running : ℕ := 5
def timeouts_for_throwing_food : ℕ := 5 * timeouts_for_running - 1
def timeouts_for_swearing : ℕ := timeouts_for_throwing_food / 3

-- Definition for total time-outs
def total_timeouts : ℕ := timeouts_for_running + timeouts_for_throwing_food + timeouts_for_swearing
-- Each time-out is 5 minutes
def timeout_duration : ℕ := 5

-- Total time in minutes
def total_time_in_minutes : ℕ := total_timeouts * timeout_duration

-- The proof statement
theorem total_time_is_correct : total_time_in_minutes = 185 := by
  sorry

end total_time_is_correct_l614_61465


namespace pounds_per_pie_l614_61459

-- Define the conditions
def total_weight : ℕ := 120
def applesauce_weight := total_weight / 2
def pies_weight := total_weight - applesauce_weight
def number_of_pies := 15

-- Define the required proof for pounds per pie
theorem pounds_per_pie :
  pies_weight / number_of_pies = 4 := by
  sorry

end pounds_per_pie_l614_61459


namespace bus_passengers_remaining_l614_61466

theorem bus_passengers_remaining (initial_passengers : ℕ := 22) 
                                 (boarding_alighting1 : (ℤ × ℤ) := (4, -8)) 
                                 (boarding_alighting2 : (ℤ × ℤ) := (6, -5)) : 
                                 (initial_passengers : ℤ) + 
                                 (boarding_alighting1.fst + boarding_alighting1.snd) + 
                                 (boarding_alighting2.fst + boarding_alighting2.snd) = 19 :=
by
  sorry

end bus_passengers_remaining_l614_61466


namespace difference_of_two_numbers_l614_61406

theorem difference_of_two_numbers (a b : ℕ) 
(h1 : a + b = 17402) 
(h2 : ∃ k : ℕ, b = 10 * k) 
(h3 : ∃ k : ℕ, a + 9 * k = b) : 
10 * a - a = 14238 :=
by sorry

end difference_of_two_numbers_l614_61406


namespace william_tickets_l614_61435

theorem william_tickets (initial_tickets final_tickets : ℕ) (h1 : initial_tickets = 15) (h2 : final_tickets = 18) : 
  final_tickets - initial_tickets = 3 := 
by
  sorry

end william_tickets_l614_61435


namespace perimeter_of_triangle_XYZ_l614_61499

/-- 
  Given the inscribed circle of triangle XYZ is tangent to XY at P,
  its radius is 15, XP = 30, and PY = 36, then the perimeter of 
  triangle XYZ is 83.4.
-/
theorem perimeter_of_triangle_XYZ :
  ∀ (XYZ : Type) (P : XYZ) (radius : ℝ) (XP PY perimeter : ℝ),
    radius = 15 → 
    XP = 30 → 
    PY = 36 →
    perimeter = 83.4 :=
by 
  intros XYZ P radius XP PY perimeter h_radius h_XP h_PY
  sorry

end perimeter_of_triangle_XYZ_l614_61499


namespace two_students_solve_all_problems_l614_61430

theorem two_students_solve_all_problems
    (students : Fin 15 → Fin 6 → Prop)
    (h : ∀ (p : Fin 6), (∃ (s1 s2 s3 s4 s5 s6 s7 s8 : Fin 15), 
          students s1 p ∧ students s2 p ∧ students s3 p ∧ students s4 p ∧ 
          students s5 p ∧ students s6 p ∧ students s7 p ∧ students s8 p)) :
    ∃ (s1 s2 : Fin 15), ∀ (p : Fin 6), students s1 p ∨ students s2 p := 
by
    sorry

end two_students_solve_all_problems_l614_61430


namespace average_rate_of_change_nonzero_l614_61446

-- Define the conditions related to the average rate of change.
variables {x0 : ℝ} {Δx : ℝ}

-- Define the statement to prove that in the definition of the average rate of change, Δx ≠ 0.
theorem average_rate_of_change_nonzero (h : Δx ≠ 0) : True :=
sorry  -- The proof is omitted as per instruction.

end average_rate_of_change_nonzero_l614_61446


namespace average_age_l614_61443

theorem average_age (Devin_age Eden_age mom_age : ℕ)
  (h1 : Devin_age = 12)
  (h2 : Eden_age = 2 * Devin_age)
  (h3 : mom_age = 2 * Eden_age) :
  (Devin_age + Eden_age + mom_age) / 3 = 28 := by
  sorry

end average_age_l614_61443


namespace calculate_expression_l614_61408

theorem calculate_expression :
  -15 - 21 + 8 = -28 :=
by
  sorry

end calculate_expression_l614_61408


namespace range_of_m_l614_61414

variable {x m : ℝ}

def quadratic (x m : ℝ) : ℝ := x^2 + (m - 1) * x + (m^2 - 3 * m + 1)

def absolute_quadratic (x m : ℝ) : ℝ := abs (quadratic x m)

theorem range_of_m (h : ∀ x ∈ Set.Icc (-1 : ℝ) 0, absolute_quadratic x m ≥ absolute_quadratic (x - 1) m) :
  m = 1 ∨ m ≥ 3 :=
sorry

end range_of_m_l614_61414


namespace inverse_sum_l614_61417

noncomputable def g (x : ℝ) : ℝ :=
if x < 15 then 2 * x + 4 else 3 * x - 1

theorem inverse_sum :
  g⁻¹ (10) + g⁻¹ (50) = 20 :=
sorry

end inverse_sum_l614_61417


namespace solve_system_of_equations_l614_61420

theorem solve_system_of_equations : 
  ∀ x y : ℝ, 
    (2 * x^2 - 3 * x * y + y^2 = 3) ∧ 
    (x^2 + 2 * x * y - 2 * y^2 = 6) 
    ↔ (x = 2 ∧ y = 1) ∨ (x = -2 ∧ y = -1) :=
by
  sorry

end solve_system_of_equations_l614_61420


namespace bedroom_curtain_width_l614_61419

theorem bedroom_curtain_width
  (initial_fabric_area : ℕ)
  (living_room_curtain_area : ℕ)
  (fabric_left : ℕ)
  (bedroom_curtain_height : ℕ)
  (bedroom_curtain_area : ℕ)
  (bedroom_curtain_width : ℕ) :
  initial_fabric_area = 16 * 12 →
  living_room_curtain_area = 4 * 6 →
  fabric_left = 160 →
  bedroom_curtain_height = 4 →
  bedroom_curtain_area = 168 - 160 →
  bedroom_curtain_area = bedroom_curtain_width * bedroom_curtain_height →
  bedroom_curtain_width = 2 :=
by
  intros h1 h2 h3 h4 h5 h6
  -- Skipping the proof
  sorry

end bedroom_curtain_width_l614_61419


namespace tiffany_max_points_l614_61410

section
  variables
  (initial_money : ℕ := 3)
  (cost_per_game : ℕ := 1)
  (rings_per_game : ℕ := 5)
  (points_red_bucket : ℕ := 2)
  (points_green_bucket : ℕ := 3)
  (points_miss : ℕ := 0)
  (games_played : ℕ := 2)
  (red_buckets : ℕ := 4)
  (green_buckets : ℕ := 5)
  (additional_games : ℕ := initial_money - games_played)
  (points_per_game_from_green_buckets : ℕ := rings_per_game * points_green_bucket)
  (total_points : ℕ := (red_buckets * points_red_bucket) + (green_buckets * points_green_bucket) + (additional_games * points_per_game_from_green_buckets))

  theorem tiffany_max_points : total_points = 38 := 
  sorry
end

end tiffany_max_points_l614_61410


namespace arithmetic_sequence_general_term_l614_61404

theorem arithmetic_sequence_general_term (S : ℕ → ℤ) (a : ℕ → ℤ) :
  (∀ n, S n = 3 * n^2 + 2 * n) →
  a 1 = S 1 ∧ (∀ n ≥ 2, a n = S n - S (n - 1)) →
  ∀ n, a n = 6 * n - 1 :=
by
  sorry

end arithmetic_sequence_general_term_l614_61404


namespace kerosene_cost_l614_61445

theorem kerosene_cost (R E K : ℕ) (h1 : E = R) (h2 : K = 6 * E) (h3 : R = 24) : 2 * K = 288 :=
by
  sorry

end kerosene_cost_l614_61445


namespace lap_length_l614_61416

theorem lap_length (I P : ℝ) (K : ℝ) 
  (h1 : 2 * I - 2 * P = 3 * K) 
  (h2 : 3 * I + 10 - 3 * P = 7 * K) : 
  K = 4 :=
by 
  -- Proof goes here
  sorry

end lap_length_l614_61416


namespace evaluate_x_from_geometric_series_l614_61447

theorem evaluate_x_from_geometric_series (x : ℝ) (h : ∑' n : ℕ, x ^ n = 4) : x = 3 / 4 :=
sorry

end evaluate_x_from_geometric_series_l614_61447


namespace find_angle_A_l614_61493

theorem find_angle_A (A B C : ℝ) (a b c : ℝ) 
  (h : 1 + (Real.tan A / Real.tan B) = 2 * c / b) : 
  A = Real.pi / 3 :=
sorry

end find_angle_A_l614_61493


namespace find_value_of_a_l614_61402

theorem find_value_of_a (a : ℝ) (h : (3 + a + 10) / 3 = 5) : a = 2 := 
by {
  sorry
}

end find_value_of_a_l614_61402


namespace starting_number_range_l614_61433

theorem starting_number_range (n : ℕ) (h₁: ∀ m : ℕ, (m > n) → (m ≤ 50) → (m = 55) → True) : n = 54 :=
sorry

end starting_number_range_l614_61433


namespace incorrect_weight_conclusion_l614_61488

theorem incorrect_weight_conclusion (x y : ℝ) (h1 : y = 0.85 * x - 85.71) :
  ¬ (x = 160 → y = 50.29) :=
sorry

end incorrect_weight_conclusion_l614_61488


namespace total_messages_l614_61444

theorem total_messages (x : ℕ) (h : x * (x - 1) = 420) : x * (x - 1) = 420 :=
by
  sorry

end total_messages_l614_61444


namespace total_campers_went_rowing_l614_61479

theorem total_campers_went_rowing (morning_campers afternoon_campers : ℕ) (h_morning : morning_campers = 35) (h_afternoon : afternoon_campers = 27) : morning_campers + afternoon_campers = 62 := by
  -- handle the proof
  sorry

end total_campers_went_rowing_l614_61479


namespace cost_of_soap_for_year_l614_61432

theorem cost_of_soap_for_year
  (months_per_bar cost_per_bar : ℕ)
  (months_in_year : ℕ)
  (h1 : months_per_bar = 2)
  (h2 : cost_per_bar = 8)
  (h3 : months_in_year = 12) :
  (months_in_year / months_per_bar) * cost_per_bar = 48 := by
  sorry

end cost_of_soap_for_year_l614_61432


namespace selling_price_l614_61418

theorem selling_price (cost_price profit_percentage selling_price : ℝ) (h1 : cost_price = 86.95652173913044)
  (h2 : profit_percentage = 0.15) : 
  selling_price = 100 :=
by
  sorry

end selling_price_l614_61418


namespace price_of_each_apple_l614_61421

-- Define the constants and conditions
def price_banana : ℝ := 0.60
def total_fruits : ℕ := 9
def total_cost : ℝ := 5.60

-- Declare the variables for number of apples and price of apples
variables (A : ℝ) (x y : ℕ)

-- Define the conditions in Lean
axiom h1 : x + y = total_fruits
axiom h2 : A * x + price_banana * y = total_cost

-- Prove that the price of each apple is $0.80
theorem price_of_each_apple : A = 0.80 :=
by sorry

end price_of_each_apple_l614_61421


namespace pens_sales_consistency_books_left_indeterminate_l614_61473

-- The initial conditions
def initial_pens : ℕ := 42
def initial_books : ℕ := 143
def pens_left : ℕ := 19
def pens_sold : ℕ := 23

-- Prove the consistency of the number of pens sold
theorem pens_sales_consistency : initial_pens - pens_left = pens_sold := by
  sorry

-- Assert that the number of books left is indeterminate based on provided conditions
theorem books_left_indeterminate : ∃ b_left : ℕ, b_left ≤ initial_books ∧
    ∀ n_books_sold : ℕ, n_books_sold > 0 → b_left = initial_books - n_books_sold := by
  sorry

end pens_sales_consistency_books_left_indeterminate_l614_61473


namespace inequality_proof_l614_61491

variable (a b : ℝ)

theorem inequality_proof (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : a + b = 1) :
  (1 / a) + (1 / b) + (1 / (a * b)) ≥ 8 :=
  by
    sorry

end inequality_proof_l614_61491


namespace sum_of_cubes_l614_61415

-- Definitions based on the conditions
variables (a b : ℝ)
variables (h1 : a + b = 2) (h2 : a * b = -3)

-- The Lean statement to prove the sum of their cubes is 26
theorem sum_of_cubes (a b : ℝ) (h1 : a + b = 2) (h2 : a * b = -3) : a^3 + b^3 = 26 :=
by
  sorry

end sum_of_cubes_l614_61415


namespace age_problem_l614_61429

theorem age_problem (c b a : ℕ) (h1 : b = 2 * c) (h2 : a = b + 2) (h3 : a + b + c = 47) : b = 18 :=
by
  sorry

end age_problem_l614_61429


namespace hyperbola_eqn_l614_61458

-- Definitions of given conditions
def a := 4
def b := 3
def c := 5

def hyperbola (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

-- Hypotheses derived from conditions
axiom asymptotes : b / a = 3 / 4
axiom right_focus : a^2 + b^2 = c^2

-- Main theorem statement
theorem hyperbola_eqn : (forall x y, hyperbola x y ↔ x^2 / 16 - y^2 / 9 = 1) :=
by
  intros
  sorry

end hyperbola_eqn_l614_61458


namespace find_x_on_line_segment_l614_61487

theorem find_x_on_line_segment (x : ℚ) : 
    (∃ m : ℚ, m = (9 - (-1))/(1 - (-2)) ∧ (2 - 9 = m * (x - 1))) → x = -11/10 :=
by 
  sorry

end find_x_on_line_segment_l614_61487


namespace simplify_expression_l614_61436

theorem simplify_expression : 4 * (14 / 5) * (20 / -42) = -4 / 15 := 
by sorry

end simplify_expression_l614_61436


namespace perfect_squares_solutions_l614_61472

noncomputable def isPerfectSquare (n : ℤ) : Prop :=
  ∃ m : ℤ, m * m = n

theorem perfect_squares_solutions :
  ∀ (a b : ℕ),
    0 < a → 0 < b →
    (isPerfectSquare (↑a * ↑a - 4 * ↑b)) →
    (isPerfectSquare (↑b * ↑b - 4 * ↑a)) →
      (a = 4 ∧ b = 4) ∨
      (a = 5 ∧ b = 6) ∨
      (a = 6 ∧ b = 5) :=
by
  -- Proof omitted
  sorry

end perfect_squares_solutions_l614_61472


namespace odd_function_f_neg_x_l614_61452

noncomputable def f (x : ℝ) : ℝ :=
if x >= 0 then x^2 - 2 * x else -(x^2 + 2 * x)

theorem odd_function_f_neg_x (x : ℝ) (hx : x < 0) :
  f x = -x^2 - 2 * x :=
by
  sorry

end odd_function_f_neg_x_l614_61452
