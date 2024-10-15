import Mathlib

namespace NUMINAMATH_GPT_number_of_ways_to_seat_Kolya_and_Olya_next_to_each_other_l626_62611

def number_of_seatings (n : ℕ) : ℕ := Nat.factorial n

theorem number_of_ways_to_seat_Kolya_and_Olya_next_to_each_other :
  let k := 2      -- Kolya and Olya as a unit
  let remaining := 3 -- The remaining people
  let pairs := 4 -- Pairs of seats that Kolya and Olya can take
  let arrangements_kolya_olya := pairs * 2 -- Each pair can have Kolya and Olya in 2 arrangements
  let arrangements_remaining := number_of_seatings remaining 
  arrangements_kolya_olya * arrangements_remaining = 48 := by
{
  -- This would be the location for the proof implementation
  sorry
}

end NUMINAMATH_GPT_number_of_ways_to_seat_Kolya_and_Olya_next_to_each_other_l626_62611


namespace NUMINAMATH_GPT_john_website_days_l626_62608

theorem john_website_days
  (monthly_visits : ℕ)
  (cents_per_visit : ℝ)
  (dollars_per_day : ℝ)
  (monthly_visits_eq : monthly_visits = 30000)
  (cents_per_visit_eq : cents_per_visit = 0.01)
  (dollars_per_day_eq : dollars_per_day = 10) :
  (monthly_visits / (dollars_per_day / cents_per_visit)) = 30 :=
by
  sorry

end NUMINAMATH_GPT_john_website_days_l626_62608


namespace NUMINAMATH_GPT_isosceles_triangle_l626_62678

theorem isosceles_triangle
  (a b c : ℝ)
  (α β γ : ℝ)
  (h1 : a + b = Real.tan (γ / 2) * (a * Real.tan α + b * Real.tan β)) :
  α = β ∨ α = γ ∨ β = γ :=
sorry

end NUMINAMATH_GPT_isosceles_triangle_l626_62678


namespace NUMINAMATH_GPT_intersecting_rectangles_shaded_area_l626_62625

theorem intersecting_rectangles_shaded_area 
  (a_w : ℕ) (a_l : ℕ) (b_w : ℕ) (b_l : ℕ) (c_w : ℕ) (c_l : ℕ)
  (overlap_ab_w : ℕ) (overlap_ab_h : ℕ)
  (overlap_ac_w : ℕ) (overlap_ac_h : ℕ)
  (overlap_bc_w : ℕ) (overlap_bc_h : ℕ)
  (triple_overlap_w : ℕ) (triple_overlap_h : ℕ) :
  a_w = 4 → a_l = 12 →
  b_w = 5 → b_l = 10 →
  c_w = 3 → c_l = 6 →
  overlap_ab_w = 4 → overlap_ab_h = 5 →
  overlap_ac_w = 3 → overlap_ac_h = 4 →
  overlap_bc_w = 3 → overlap_bc_h = 3 →
  triple_overlap_w = 3 → triple_overlap_h = 3 →
  ((a_w * a_l) + (b_w * b_l) + (c_w * c_l)) - 
  ((overlap_ab_w * overlap_ab_h) + (overlap_ac_w * overlap_ac_h) + (overlap_bc_w * overlap_bc_h)) + 
  (triple_overlap_w * triple_overlap_h) = 84 :=
by 
  sorry

end NUMINAMATH_GPT_intersecting_rectangles_shaded_area_l626_62625


namespace NUMINAMATH_GPT_bucket_full_weight_l626_62667

theorem bucket_full_weight (p q : ℝ) (x y : ℝ) 
  (h1 : x + (1 / 3) * y = p) 
  (h2 : x + (3 / 4) * y = q) : 
  x + y = (8 * q - 3 * p) / 5 := 
  by
    sorry

end NUMINAMATH_GPT_bucket_full_weight_l626_62667


namespace NUMINAMATH_GPT_average_mileage_city_l626_62663

variable (total_distance : ℝ) (gallons : ℝ) (highway_mpg : ℝ) (city_mpg : ℝ)

-- The given conditions
def conditions : Prop := (total_distance = 280.6) ∧ (gallons = 23) ∧ (highway_mpg = 12.2)

-- The theorem to prove
theorem average_mileage_city (h : conditions total_distance gallons highway_mpg) :
  total_distance / gallons = 12.2 :=
sorry

end NUMINAMATH_GPT_average_mileage_city_l626_62663


namespace NUMINAMATH_GPT_value_of_a_plus_b_l626_62605

variable {F : Type} [Field F]

theorem value_of_a_plus_b (a b : F) (h1 : ∀ x, x ≠ 0 → a + b / x = 2 ↔ x = -2)
                                      (h2 : ∀ x, x ≠ 0 → a + b / x = 6 ↔ x = -6) :
  a + b = 20 :=
sorry

end NUMINAMATH_GPT_value_of_a_plus_b_l626_62605


namespace NUMINAMATH_GPT_right_triangle_property_l626_62660

theorem right_triangle_property
  (a b c x : ℝ)
  (h1 : c^2 = a^2 + b^2)
  (h2 : 1/2 * a * b = 1/2 * c * x)
  : 1/x^2 = 1/a^2 + 1/b^2 :=
sorry

end NUMINAMATH_GPT_right_triangle_property_l626_62660


namespace NUMINAMATH_GPT_annual_rent_per_square_foot_l626_62633

-- Given conditions
def dimensions_length : ℕ := 10
def dimensions_width : ℕ := 10
def monthly_rent : ℕ := 1300

-- Derived conditions
def area : ℕ := dimensions_length * dimensions_width
def annual_rent : ℕ := monthly_rent * 12

-- The problem statement as a theorem in Lean 4
theorem annual_rent_per_square_foot :
  annual_rent / area = 156 := by
  sorry

end NUMINAMATH_GPT_annual_rent_per_square_foot_l626_62633


namespace NUMINAMATH_GPT_minimum_cubes_required_l626_62640

def cube_snaps_visible (n : Nat) : Prop := 
  ∀ (cubes : Fin n → Fin 6 → Bool),
    (∀ i, (cubes i 0 ∧ cubes i 1) ∨ ¬(cubes i 0 ∨ cubes i 1)) → 
    ∃ i j, (i ≠ j) ∧ 
            (cubes i 0 ↔ ¬ cubes j 0) ∧ 
            (cubes i 1 ↔ ¬ cubes j 1)

theorem minimum_cubes_required : 
  ∃ n, cube_snaps_visible n ∧ n = 4 := 
  by sorry

end NUMINAMATH_GPT_minimum_cubes_required_l626_62640


namespace NUMINAMATH_GPT_sqrt_trig_identity_l626_62635

theorem sqrt_trig_identity
  (α : ℝ)
  (P : ℝ × ℝ)
  (hP: P = (Real.sin 2, Real.cos 2))
  (h_terminal: ∃ (θ : ℝ), P = (Real.cos θ, Real.sin θ)) :
  Real.sqrt (2 * (1 - Real.sin α)) = 2 * Real.sin 1 := 
sorry

end NUMINAMATH_GPT_sqrt_trig_identity_l626_62635


namespace NUMINAMATH_GPT_batsman_average_after_17th_inning_l626_62622

theorem batsman_average_after_17th_inning
  (A : ℕ)
  (h1 : (16 * A + 88) / 17 = A + 3) :
  37 + 3 = 40 :=
by sorry

end NUMINAMATH_GPT_batsman_average_after_17th_inning_l626_62622


namespace NUMINAMATH_GPT_power_mod_eq_five_l626_62688

theorem power_mod_eq_five
  (m : ℕ)
  (h₀ : 0 ≤ m)
  (h₁ : m < 8)
  (h₂ : 13^5 % 8 = m) : m = 5 :=
by 
  sorry

end NUMINAMATH_GPT_power_mod_eq_five_l626_62688


namespace NUMINAMATH_GPT_volume_of_polyhedron_l626_62699

theorem volume_of_polyhedron (s : ℝ) : 
  let base_area := (3 * Real.sqrt 3 / 2) * s^2
  let height := s
  let volume := (1 / 3) * base_area * height
  volume = (Real.sqrt 3 / 2) * s^3 :=
by
  let base_area := (3 * Real.sqrt 3 / 2) * s^2
  let height := s
  let volume := (1 / 3) * base_area * height
  show volume = (Real.sqrt 3 / 2) * s^3
  sorry

end NUMINAMATH_GPT_volume_of_polyhedron_l626_62699


namespace NUMINAMATH_GPT_number_of_appointments_l626_62657

-- Define the conditions
variables {hours_in_workday : ℕ} {appointments_duration : ℕ} {permit_rate : ℕ} {total_permits : ℕ}
variables (H1 : hours_in_workday = 8) (H2 : appointments_duration = 3) (H3 : permit_rate = 50) (H4: total_permits = 100)

-- Define the question as a theorem with the correct answer
theorem number_of_appointments : 
  (hours_in_workday - (total_permits / permit_rate)) / appointments_duration = 2 :=
by
  -- Proof is not required
  sorry

end NUMINAMATH_GPT_number_of_appointments_l626_62657


namespace NUMINAMATH_GPT_triangle_at_most_one_obtuse_l626_62670

-- Define the notion of a triangle and obtuse angle
def isTriangle (A B C: ℝ) : Prop := (A + B > C) ∧ (A + C > B) ∧ (B + C > A)
def isObtuseAngle (theta: ℝ) : Prop := 90 < theta ∧ theta < 180

-- A theorem to prove that a triangle cannot have more than one obtuse angle 
theorem triangle_at_most_one_obtuse (A B C: ℝ) (angleA angleB angleC : ℝ) 
    (h1 : isTriangle A B C)
    (h2 : isObtuseAngle angleA)
    (h3 : isObtuseAngle angleB)
    (h4 : angleA + angleB + angleC = 180):
    false :=
by
  sorry

end NUMINAMATH_GPT_triangle_at_most_one_obtuse_l626_62670


namespace NUMINAMATH_GPT_vertex_angle_of_isosceles_with_angle_30_l626_62643

def isosceles_triangle (a b c : ℝ) : Prop :=
  (a = b ∨ b = c ∨ c = a) ∧ a + b + c = 180

theorem vertex_angle_of_isosceles_with_angle_30 (a b c : ℝ) 
  (ha : isosceles_triangle a b c) 
  (h1 : a = 30 ∨ b = 30 ∨ c = 30) :
  (a = 30 ∨ b = 30 ∨ c = 30) ∨ (a = 120 ∨ b = 120 ∨ c = 120) := 
sorry

end NUMINAMATH_GPT_vertex_angle_of_isosceles_with_angle_30_l626_62643


namespace NUMINAMATH_GPT_lowest_price_correct_l626_62648

noncomputable def lowest_price (cost_per_component shipping_cost_per_unit fixed_costs number_of_components : ℕ) : ℕ :=
(cost_per_component + shipping_cost_per_unit) * number_of_components + fixed_costs

theorem lowest_price_correct :
  lowest_price 80 5 16500 150 / 150 = 195 :=
by
  sorry

end NUMINAMATH_GPT_lowest_price_correct_l626_62648


namespace NUMINAMATH_GPT_min_score_to_achieve_average_l626_62616

theorem min_score_to_achieve_average (a b c : ℕ) (h₁ : a = 76) (h₂ : b = 94) (h₃ : c = 87) :
  ∃ d e : ℕ, d + e = 148 ∧ d ≤ 100 ∧ e ≤ 100 ∧ min d e = 48 :=
by sorry

end NUMINAMATH_GPT_min_score_to_achieve_average_l626_62616


namespace NUMINAMATH_GPT_parallel_lines_m_values_l626_62639

theorem parallel_lines_m_values (m : ℝ) :
  (∀ x y : ℝ, (3 + m) * x + 4 * y = 5 → 2 * x + (5 + m) * y = 8) →
  (m = -1 ∨ m = -7) :=
by
  sorry

end NUMINAMATH_GPT_parallel_lines_m_values_l626_62639


namespace NUMINAMATH_GPT_train_cross_time_l626_62628

noncomputable def speed_conversion (speed_kmh : ℝ) : ℝ :=
  speed_kmh * (1000 / 3600)

noncomputable def time_to_cross_pole (length_m speed_kmh : ℝ) : ℝ :=
  length_m / speed_conversion speed_kmh

theorem train_cross_time (length_m : ℝ) (speed_kmh : ℝ) :
  length_m = 225 → speed_kmh = 250 → time_to_cross_pole length_m speed_kmh = 3.24 := by
  intros hlen hspeed
  simp [time_to_cross_pole, speed_conversion, hlen, hspeed]
  sorry

end NUMINAMATH_GPT_train_cross_time_l626_62628


namespace NUMINAMATH_GPT_express_in_scientific_notation_l626_62692

theorem express_in_scientific_notation :
  (10.58 * 10^9) = 1.058 * 10^10 :=
by
  sorry

end NUMINAMATH_GPT_express_in_scientific_notation_l626_62692


namespace NUMINAMATH_GPT_complete_square_solution_l626_62642

theorem complete_square_solution (x : ℝ) :
  x^2 - 8 * x + 6 = 0 → (x - 4)^2 = 10 :=
by
  intro h
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_complete_square_solution_l626_62642


namespace NUMINAMATH_GPT_incentive_given_to_john_l626_62668

-- Conditions (definitions)
def commission_held : ℕ := 25000
def advance_fees : ℕ := 8280
def amount_given_to_john : ℕ := 18500

-- Problem statement
theorem incentive_given_to_john : (amount_given_to_john - (commission_held - advance_fees)) = 1780 := 
by
  sorry

end NUMINAMATH_GPT_incentive_given_to_john_l626_62668


namespace NUMINAMATH_GPT_multiply_expression_l626_62607

variable {x : ℝ}

theorem multiply_expression :
  (x^4 + 10*x^2 + 25) * (x^2 - 25) = x^4 + 10*x^2 :=
by
  sorry

end NUMINAMATH_GPT_multiply_expression_l626_62607


namespace NUMINAMATH_GPT_mr_yadav_expenses_l626_62683

theorem mr_yadav_expenses (S : ℝ) 
  (h1 : S > 0) 
  (h2 : 0.6 * S > 0) 
  (h3 : (12 * 0.2 * S) = 48456) : 
  0.2 * S = 4038 :=
by
  sorry

end NUMINAMATH_GPT_mr_yadav_expenses_l626_62683


namespace NUMINAMATH_GPT_candy_bars_given_to_sister_first_time_l626_62627

theorem candy_bars_given_to_sister_first_time (x : ℕ) :
  (7 - x) + 30 - 4 * x = 22 → x = 3 :=
by
  sorry

end NUMINAMATH_GPT_candy_bars_given_to_sister_first_time_l626_62627


namespace NUMINAMATH_GPT_equivalence_of_statements_l626_62604

theorem equivalence_of_statements (S X Y : Prop) : 
  (S → (¬ X ∧ ¬ Y)) ↔ ((X ∨ Y) → ¬ S) :=
by sorry

end NUMINAMATH_GPT_equivalence_of_statements_l626_62604


namespace NUMINAMATH_GPT_fiona_correct_answers_l626_62664

-- 5 marks for each correct answer in Questions 1-15
def marks_questions_1_to_15 (correct1 : ℕ) : ℕ := 5 * correct1

-- 6 marks for each correct answer in Questions 16-25
def marks_questions_16_to_25 (correct2 : ℕ) : ℕ := 6 * correct2

-- 1 mark penalty for incorrect answers in Questions 16-20
def penalty_questions_16_to_20 (incorrect1 : ℕ) : ℕ := incorrect1

-- 2 mark penalty for incorrect answers in Questions 21-25
def penalty_questions_21_to_25 (incorrect2 : ℕ) : ℕ := 2 * incorrect2

-- Total marks given correct and incorrect answers
def total_marks (correct1 correct2 incorrect1 incorrect2 : ℕ) : ℕ :=
  marks_questions_1_to_15 correct1 +
  marks_questions_16_to_25 correct2 -
  penalty_questions_16_to_20 incorrect1 -
  penalty_questions_21_to_25 incorrect2

-- Fiona's total score
def fionas_total_score : ℕ := 80

-- The proof problem: Fiona answered 16 questions correctly
theorem fiona_correct_answers (correct1 correct2 incorrect1 incorrect2 : ℕ) :
  total_marks correct1 correct2 incorrect1 incorrect2 = fionas_total_score → 
  (correct1 + correct2 = 16) := sorry

end NUMINAMATH_GPT_fiona_correct_answers_l626_62664


namespace NUMINAMATH_GPT_part_a_part_b_l626_62638

-- Part (a)
theorem part_a (m n : ℕ) (hm : 0 < m) (hn : 0 < n) (h : m > n) : 
  (1 + 1 / (m:ℝ))^m > (1 + 1 / (n:ℝ))^n :=
by sorry

-- Part (b)
theorem part_b (m n : ℕ) (hm : 0 < m) (hn : 1 < n) (h : m > n) : 
  (1 + 1 / (m:ℝ))^(m + 1) < (1 + 1 / (n:ℝ))^(n + 1) :=
by sorry

end NUMINAMATH_GPT_part_a_part_b_l626_62638


namespace NUMINAMATH_GPT_number_of_dials_l626_62666

theorem number_of_dials (k : ℕ) (aligned_sums : ℕ → ℕ) :
  (∀ i j : ℕ, i ≠ j → aligned_sums i % 12 = aligned_sums j % 12) ↔ k = 12 :=
by
  sorry

end NUMINAMATH_GPT_number_of_dials_l626_62666


namespace NUMINAMATH_GPT_arithmetic_seq_S13_l626_62601

noncomputable def arithmetic_sequence_sum (a d n : ℕ) : ℕ := n * (2 * a + (n - 1) * d) / 2

theorem arithmetic_seq_S13 (a_1 d : ℕ) (h : a_1 + 6 * d = 10) :
  arithmetic_sequence_sum a_1 d 13 = 130 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_seq_S13_l626_62601


namespace NUMINAMATH_GPT_option_C_is_quadratic_l626_62680

-- Define the conditions
def option_A (x : ℝ) : Prop := 2 * x = 3
def option_B (a b c x : ℝ) : Prop := a * x^2 + b * x + c = 0
def option_C (x : ℝ) : Prop := (4 * x - 3) * (3 * x + 1) = 0
def option_D (x : ℝ) : Prop := (x + 3) * (x - 2) = (x - 2) * (x + 1)

-- Define what it means to be a quadratic equation
def is_quadratic (f : ℝ → Prop) : Prop :=
  ∃ a b c : ℝ, (∀ x, f x = (a * x^2 + b * x + c = 0)) ∧ a ≠ 0

-- The main theorem statement
theorem option_C_is_quadratic : is_quadratic option_C :=
sorry

end NUMINAMATH_GPT_option_C_is_quadratic_l626_62680


namespace NUMINAMATH_GPT_evaluate_fraction_l626_62693

theorem evaluate_fraction : (25 * 5 + 5^2) / (5^2 - 15) = 15 := 
by
  sorry

end NUMINAMATH_GPT_evaluate_fraction_l626_62693


namespace NUMINAMATH_GPT_circuit_boards_fail_inspection_l626_62651

theorem circuit_boards_fail_inspection (P F : ℝ) (h1 : P + F = 3200)
    (h2 : (1 / 8) * P + F = 456) : F = 64 :=
by
  sorry

end NUMINAMATH_GPT_circuit_boards_fail_inspection_l626_62651


namespace NUMINAMATH_GPT_combined_tax_rate_33_33_l626_62630

-- Define the necessary conditions
def mork_tax_rate : ℝ := 0.40
def mindy_tax_rate : ℝ := 0.30
def mindy_income_ratio : ℝ := 2.0

-- Main theorem statement
theorem combined_tax_rate_33_33 :
  ∀ (X : ℝ), ((mork_tax_rate * X + mindy_income_ratio * mindy_tax_rate * X) / (X + mindy_income_ratio * X) * 100) = 100 / 3 :=
by
  intro X
  sorry

end NUMINAMATH_GPT_combined_tax_rate_33_33_l626_62630


namespace NUMINAMATH_GPT_leastCookies_l626_62653

theorem leastCookies (b : ℕ) :
  (b % 6 = 5) ∧ (b % 8 = 3) ∧ (b % 9 = 7) →
  b = 179 :=
by
  sorry

end NUMINAMATH_GPT_leastCookies_l626_62653


namespace NUMINAMATH_GPT_decimal_to_fraction_l626_62618

theorem decimal_to_fraction (x : ℝ) (hx : x = 2.35) : x = 47 / 20 := by
  sorry

end NUMINAMATH_GPT_decimal_to_fraction_l626_62618


namespace NUMINAMATH_GPT_beth_final_students_l626_62609

-- Define the initial conditions
def initial_students : ℕ := 150
def students_joined : ℕ := 30
def students_left : ℕ := 15

-- Define the number of students after the first additional year
def after_first_year : ℕ := initial_students + students_joined

-- Define the final number of students after students leaving
def final_students : ℕ := after_first_year - students_left

-- Theorem to prove the number of students in the final year
theorem beth_final_students : 
  final_students = 165 :=
by
  sorry

end NUMINAMATH_GPT_beth_final_students_l626_62609


namespace NUMINAMATH_GPT_julie_initial_savings_l626_62675

theorem julie_initial_savings (S r : ℝ) 
  (h1 : (S / 2) * r * 2 = 120) 
  (h2 : (S / 2) * ((1 + r)^2 - 1) = 124) : 
  S = 1800 := 
sorry

end NUMINAMATH_GPT_julie_initial_savings_l626_62675


namespace NUMINAMATH_GPT_total_cost_football_games_l626_62632

-- Define the initial conditions
def games_this_year := 14
def games_last_year := 29
def price_this_year := 45
def price_lowest := 40
def price_highest := 65
def one_third_games_last_year := games_last_year / 3
def one_fourth_games_last_year := games_last_year / 4

-- Define the assertions derived from the conditions
def games_lowest_price := 9  -- rounded down from games_last_year / 3
def games_highest_price := 7  -- rounded down from games_last_year / 4
def remaining_games := games_last_year - (games_lowest_price + games_highest_price)

-- Define the costs calculation
def cost_this_year := games_this_year * price_this_year
def cost_lowest_price_games := games_lowest_price * price_lowest
def cost_highest_price_games := games_highest_price * price_highest
def total_cost := cost_this_year + cost_lowest_price_games + cost_highest_price_games

-- The theorem statement
theorem total_cost_football_games (h1 : games_lowest_price = 9) (h2 : games_highest_price = 7) 
  (h3 : cost_this_year = 630) (h4 : cost_lowest_price_games = 360) (h5 : cost_highest_price_games = 455) :
  total_cost = 1445 :=
by
  -- Since this is just the statement, we can simply put 'sorry' here.
  sorry

end NUMINAMATH_GPT_total_cost_football_games_l626_62632


namespace NUMINAMATH_GPT_largest_real_solution_l626_62671

theorem largest_real_solution (x : ℝ) (h : (⌊x⌋ / x = 7 / 8)) : x ≤ 48 / 7 := by
  sorry

end NUMINAMATH_GPT_largest_real_solution_l626_62671


namespace NUMINAMATH_GPT_product_identity_l626_62650

theorem product_identity : 
  (7^3 - 1) / (7^3 + 1) * 
  (8^3 - 1) / (8^3 + 1) * 
  (9^3 - 1) / (9^3 + 1) * 
  (10^3 - 1) / (10^3 + 1) * 
  (11^3 - 1) / (11^3 + 1) = 
  133 / 946 := 
by
  sorry

end NUMINAMATH_GPT_product_identity_l626_62650


namespace NUMINAMATH_GPT_sqrt_equality_l626_62690

theorem sqrt_equality (n : ℤ) (h : Real.sqrt (8 + n) = 9) : n = 73 :=
by
  sorry

end NUMINAMATH_GPT_sqrt_equality_l626_62690


namespace NUMINAMATH_GPT_seating_arrangements_l626_62682

def count_arrangements (n k : ℕ) : ℕ :=
  (n.factorial) / (n - k).factorial

theorem seating_arrangements : count_arrangements 6 5 * 3 = 360 :=
  sorry

end NUMINAMATH_GPT_seating_arrangements_l626_62682


namespace NUMINAMATH_GPT_lines_parallel_if_perpendicular_to_plane_l626_62685

axiom line : Type
axiom plane : Type

-- Definitions of perpendicular and parallel
axiom perp : line → plane → Prop
axiom parallel : line → line → Prop

variables (a b : line) (α : plane)

theorem lines_parallel_if_perpendicular_to_plane (h1 : perp a α) (h2 : perp b α) : parallel a b :=
sorry

end NUMINAMATH_GPT_lines_parallel_if_perpendicular_to_plane_l626_62685


namespace NUMINAMATH_GPT_exists_radius_for_marked_points_l626_62610

theorem exists_radius_for_marked_points :
  ∃ R : ℝ, (∀ θ : ℝ, (0 ≤ θ ∧ θ < 2 * π) →
    (∃ n : ℕ, (θ ≤ (n * 2 * π * R) % (2 * π * R) + 1 / R ∧ (n * 2 * π * R) % (2 * π * R) < θ + 1))) :=
sorry

end NUMINAMATH_GPT_exists_radius_for_marked_points_l626_62610


namespace NUMINAMATH_GPT_number_of_selected_in_interval_l626_62612

-- Definitions and conditions based on the problem statement
def total_employees : ℕ := 840
def sample_size : ℕ := 42
def systematic_sampling_interval : ℕ := total_employees / sample_size
def interval_start : ℕ := 481
def interval_end : ℕ := 720

-- Main theorem statement that we need to prove
theorem number_of_selected_in_interval :
  let selected_in_interval : ℕ := (interval_end - interval_start + 1) / systematic_sampling_interval
  selected_in_interval = 12 := by
  sorry

end NUMINAMATH_GPT_number_of_selected_in_interval_l626_62612


namespace NUMINAMATH_GPT_negate_proposition_l626_62669

theorem negate_proposition :
  (¬ ∃ (x₀ : ℝ), x₀^2 + 2 * x₀ + 3 ≤ 0) ↔ (∀ (x : ℝ), x^2 + 2 * x + 3 > 0) :=
by
  sorry

end NUMINAMATH_GPT_negate_proposition_l626_62669


namespace NUMINAMATH_GPT_find_inradius_l626_62615

-- Define variables and constants
variables (P A : ℝ)
variables (s r : ℝ)

-- Given conditions as definitions
def perimeter_triangle : Prop := P = 36
def area_triangle : Prop := A = 45

-- Semi-perimeter definition
def semi_perimeter : Prop := s = P / 2

-- Inradius and area relationship
def inradius_area_relation : Prop := A = r * s

-- Theorem statement
theorem find_inradius (hP : perimeter_triangle P) (hA : area_triangle A) (hs : semi_perimeter P s) (har : inradius_area_relation A r s) :
  r = 2.5 :=
by
  sorry

end NUMINAMATH_GPT_find_inradius_l626_62615


namespace NUMINAMATH_GPT_find_divisor_l626_62652

theorem find_divisor (dividend quotient remainder : ℕ) (h₁ : dividend = 176) (h₂ : quotient = 9) (h₃ : remainder = 5) : 
  ∃ divisor, dividend = (divisor * quotient) + remainder ∧ divisor = 19 := by
sorry

end NUMINAMATH_GPT_find_divisor_l626_62652


namespace NUMINAMATH_GPT_number_of_male_rabbits_l626_62654

-- Definitions based on the conditions
def white_rabbits : ℕ := 12
def black_rabbits : ℕ := 9
def female_rabbits : ℕ := 8

-- The question and proof goal
theorem number_of_male_rabbits : 
  (white_rabbits + black_rabbits - female_rabbits) = 13 :=
by
  sorry

end NUMINAMATH_GPT_number_of_male_rabbits_l626_62654


namespace NUMINAMATH_GPT_Megan_finish_all_problems_in_8_hours_l626_62637

theorem Megan_finish_all_problems_in_8_hours :
  ∀ (math_problems spelling_problems problems_per_hour : ℕ),
    math_problems = 36 →
    spelling_problems = 28 →
    problems_per_hour = 8 →
    (math_problems + spelling_problems) / problems_per_hour = 8 :=
by
  intros
  sorry

end NUMINAMATH_GPT_Megan_finish_all_problems_in_8_hours_l626_62637


namespace NUMINAMATH_GPT_problem_xyz_l626_62629

theorem problem_xyz (x y : ℝ) (h1 : (x + y)^2 = 16) (h2 : x * y = -8) :
  x^2 + y^2 = 32 :=
by
  sorry

end NUMINAMATH_GPT_problem_xyz_l626_62629


namespace NUMINAMATH_GPT_inequality_proof_l626_62659

noncomputable def given_condition_1 (a b c u : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ (∃ x, (a * x^2 - b * x + c = 0)) ∧
  a * u^2 - b * u + c ≤ 0

noncomputable def given_condition_2 (A B C v : ℝ) : Prop :=
  A > 0 ∧ B > 0 ∧ C > 0 ∧ (∃ x, (A * x^2 - B * x + C = 0)) ∧
  A * v^2 - B * v + C ≤ 0

theorem inequality_proof (a b c A B C u v : ℝ) (h1 : given_condition_1 a b c u) (h2 : given_condition_2 A B C v) :
  (a * u + A * v) * (c / u + C / v) ≤ (b + B) ^ 2 / 4 :=
by
    sorry

end NUMINAMATH_GPT_inequality_proof_l626_62659


namespace NUMINAMATH_GPT_find_value_of_r_l626_62689

theorem find_value_of_r (a r : ℝ) (h1 : a / (1 - r) = 20) (h2 : a * r / (1 - r^2) = 8) : r = 2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_find_value_of_r_l626_62689


namespace NUMINAMATH_GPT_carols_total_peanuts_l626_62691

-- Define the initial number of peanuts Carol has
def initial_peanuts : ℕ := 2

-- Define the number of peanuts given by Carol's father
def peanuts_given : ℕ := 5

-- Define the total number of peanuts Carol has
def total_peanuts : ℕ := initial_peanuts + peanuts_given

-- The statement we need to prove
theorem carols_total_peanuts : total_peanuts = 7 := by
  sorry

end NUMINAMATH_GPT_carols_total_peanuts_l626_62691


namespace NUMINAMATH_GPT_negation_of_p_is_neg_p_l626_62661

def p (f : ℝ → ℝ) : Prop :=
  ∀ x1 x2 : ℝ, (f x2 - f x1) * (x2 - x1) ≥ 0

def neg_p (f : ℝ → ℝ) : Prop :=
  ∃ x1 x2 : ℝ, (f x2 - f x1) * (x2 - x1) < 0

theorem negation_of_p_is_neg_p (f : ℝ → ℝ) : ¬ p f ↔ neg_p f :=
by
  sorry -- Proof of this theorem

end NUMINAMATH_GPT_negation_of_p_is_neg_p_l626_62661


namespace NUMINAMATH_GPT_intersection_A_B_l626_62623

-- Define sets A and B
def A : Set ℤ := {-2, 0, 2}
def B : Set ℤ := {x | ∃ y ∈ A, |y| = x}

-- Prove that the intersection of A and B is {0, 2}
theorem intersection_A_B :
  A ∩ B = {0, 2} :=
by
  sorry

end NUMINAMATH_GPT_intersection_A_B_l626_62623


namespace NUMINAMATH_GPT_f_relationship_l626_62672

noncomputable def f (x : ℝ) : ℝ := sorry -- definition of f needs to be filled in later

-- Conditions given in the problem
variable (h_diff : Differentiable ℝ f)
variable (h_gt : ∀ x: ℝ, deriv f x > f x)
variable (a : ℝ) (h_pos : a > 0)

theorem f_relationship (f : ℝ → ℝ) (h_diff : Differentiable ℝ f) 
  (h_gt : ∀ x: ℝ, deriv f x > f x) (a : ℝ) (h_pos : a > 0) :
  f a > Real.exp a * f 0 :=
sorry

end NUMINAMATH_GPT_f_relationship_l626_62672


namespace NUMINAMATH_GPT_solve_inequality_l626_62649

theorem solve_inequality (x : ℝ) (h : 0 < x ∧ x < 2) : abs (2 * x - 1) < abs x + 1 :=
by
  sorry

end NUMINAMATH_GPT_solve_inequality_l626_62649


namespace NUMINAMATH_GPT_original_people_in_room_l626_62696

theorem original_people_in_room (x : ℝ) (h1 : x / 3 * 2 / 2 = 18) : x = 54 :=
sorry

end NUMINAMATH_GPT_original_people_in_room_l626_62696


namespace NUMINAMATH_GPT_proof_problem_l626_62617

theorem proof_problem (x y : ℝ) (h1 : 3 * x ^ 2 - 5 * x + 4 * y + 6 = 0) 
                      (h2 : 3 * x - 2 * y + 1 = 0) : 
                      4 * y ^ 2 - 2 * y + 24 = 0 := 
by 
  sorry

end NUMINAMATH_GPT_proof_problem_l626_62617


namespace NUMINAMATH_GPT_tangent_line_at_one_extreme_points_and_inequality_l626_62646

noncomputable def f (x a : ℝ) := x^2 - 2*x + a * Real.log x

-- Question 1: Tangent Line
theorem tangent_line_at_one (x a : ℝ) (h_a : a = 2) (hx_pos : x > 0) :
    2*x - Real.log x - (2*x - Real.log 1 - 1) = 0 := by
  sorry

-- Question 2: Extreme Points and Inequality
theorem extreme_points_and_inequality (a x1 x2 : ℝ) (h1 : 2*x1^2 - 2*x1 + a = 0)
    (h2 : 2*x2^2 - 2*x2 + a = 0) (hx12 : x1 < x2) (hx1_pos : x1 > 0) (hx2_pos : x2 > 0) :
    0 < a ∧ a < 1/2 ∧ (f x1 a) / x2 > -3/2 - Real.log 2 := by
  sorry

end NUMINAMATH_GPT_tangent_line_at_one_extreme_points_and_inequality_l626_62646


namespace NUMINAMATH_GPT_alice_chicken_weight_l626_62613

theorem alice_chicken_weight (total_cost_needed : ℝ)
  (amount_to_spend_more : ℝ)
  (cost_lettuce : ℝ)
  (cost_tomatoes : ℝ)
  (sweet_potato_quantity : ℝ)
  (cost_per_sweet_potato : ℝ)
  (broccoli_quantity : ℝ)
  (cost_per_broccoli : ℝ)
  (brussel_sprouts_weight : ℝ)
  (cost_per_brussel_sprouts : ℝ)
  (cost_per_pound_chicken : ℝ)
  (total_cost_excluding_chicken : ℝ) :
  total_cost_needed = 35 ∧
  amount_to_spend_more = 11 ∧
  cost_lettuce = 3 ∧
  cost_tomatoes = 2.5 ∧
  sweet_potato_quantity = 4 ∧
  cost_per_sweet_potato = 0.75 ∧
  broccoli_quantity = 2 ∧
  cost_per_broccoli = 2 ∧
  brussel_sprouts_weight = 1 ∧
  cost_per_brussel_sprouts = 2.5 ∧
  total_cost_excluding_chicken = (cost_lettuce + cost_tomatoes + sweet_potato_quantity * cost_per_sweet_potato + broccoli_quantity * cost_per_broccoli + brussel_sprouts_weight * cost_per_brussel_sprouts) →
  (total_cost_needed - amount_to_spend_more - total_cost_excluding_chicken) / cost_per_pound_chicken = 1.5 :=
by
  intros
  sorry

end NUMINAMATH_GPT_alice_chicken_weight_l626_62613


namespace NUMINAMATH_GPT_system_of_two_linear_equations_l626_62644

theorem system_of_two_linear_equations :
  ((∃ x y z, x + z = 5 ∧ x - 2 * y = 6) → False) ∧
  ((∃ x y, x * y = 5 ∧ x - 4 * y = 2) → False) ∧
  ((∃ x y, x + y = 5 ∧ 3 * x - 4 * y = 12) → True) ∧
  ((∃ x y, x^2 + y = 2 ∧ x - y = 9) → False) :=
by {
  sorry
}

end NUMINAMATH_GPT_system_of_two_linear_equations_l626_62644


namespace NUMINAMATH_GPT_cost_price_l626_62677

variables (SP DS CP : ℝ)
variables (discount_rate profit_rate : ℝ)
variables (H1 : SP = 24000)
variables (H2 : discount_rate = 0.10)
variables (H3 : profit_rate = 0.08)
variables (H4 : DS = SP - (discount_rate * SP))
variables (H5 : DS = CP + (profit_rate * CP))

theorem cost_price (H1 : SP = 24000) (H2 : discount_rate = 0.10) 
  (H3 : profit_rate = 0.08) (H4 : DS = SP - (discount_rate * SP)) 
  (H5 : DS = CP + (profit_rate * CP)) : 
  CP = 20000 := 
sorry

end NUMINAMATH_GPT_cost_price_l626_62677


namespace NUMINAMATH_GPT_part1_part2a_part2b_part2c_l626_62600

def f (x a : ℝ) := |2 * x - 1| + |x - a|

theorem part1 (x : ℝ) (h : 0 ≤ x ∧ x ≤ 2) : f x 3 ≤ 4 := sorry

theorem part2a (a x : ℝ) (h0 : a < 1 / 2) (h1 : a ≤ x ∧ x ≤ 1 / 2) : f x a = |x - 1 + a| := sorry

theorem part2b (a x : ℝ) (h0 : a = 1 / 2) (h1 : x = 1 / 2) : f x a = |x - 1 + a| := sorry

theorem part2c (a x : ℝ) (h0 : a > 1 / 2) (h1 : 1 / 2 ≤ x ∧ x ≤ a) : f x a = |x - 1 + a| := sorry

end NUMINAMATH_GPT_part1_part2a_part2b_part2c_l626_62600


namespace NUMINAMATH_GPT_pets_remaining_l626_62655

-- Definitions based on conditions
def initial_puppies : ℕ := 7
def initial_kittens : ℕ := 6
def sold_puppies : ℕ := 2
def sold_kittens : ℕ := 3

-- Theorem statement
theorem pets_remaining : initial_puppies + initial_kittens - (sold_puppies + sold_kittens) = 8 :=
by
  sorry

end NUMINAMATH_GPT_pets_remaining_l626_62655


namespace NUMINAMATH_GPT_john_text_messages_l626_62602

/-- John decides to get a new phone number and it ends up being a recycled number. 
    He used to get some text messages a day. 
    Now he is getting 55 text messages a day, 
    and he is getting 245 text messages per week that are not intended for him. 
    How many text messages a day did he used to get?
-/
theorem john_text_messages (m : ℕ) (h1 : 55 = m + 35) (h2 : 245 = 7 * 35) : m = 20 := 
by 
  sorry

end NUMINAMATH_GPT_john_text_messages_l626_62602


namespace NUMINAMATH_GPT_real_solutions_l626_62624

theorem real_solutions:
  ∀ x: ℝ, 
    (x ≠ 2) ∧ (x ≠ 4) ∧ 
    ((x - 1) * (x - 2) * (x - 3) * (x - 4) * (x - 3) * (x - 2) * (x - 1)) / 
    ((x - 2) * (x - 4) * (x - 2)) = 1 
    → (x = 2 + Real.sqrt 2) ∨ (x = 2 - Real.sqrt 2) :=
by
  sorry

end NUMINAMATH_GPT_real_solutions_l626_62624


namespace NUMINAMATH_GPT_n_not_2_7_l626_62626

open Set

variable (M N : Set ℕ)

-- Define the given set M
def M_def : Prop := M = {1, 4, 7}

-- Define the condition M ∪ N = M
def union_condition : Prop := M ∪ N = M

-- The main statement to be proved
theorem n_not_2_7 (M_def : M = {1, 4, 7}) (union_condition : M ∪ N = M) : N ≠ {2, 7} :=
  sorry

end NUMINAMATH_GPT_n_not_2_7_l626_62626


namespace NUMINAMATH_GPT_total_distance_l626_62676

theorem total_distance (D : ℝ) 
  (h₁ : 60 * (D / 2 / 60) = D / 2) 
  (h₂ : 40 * ((D / 2) / 4 / 40) = D / 8) 
  (h₃ : 50 * (105 / 50) = 105)
  (h₄ : D = D / 2 + D / 8 + 105) : 
  D = 280 :=
by sorry

end NUMINAMATH_GPT_total_distance_l626_62676


namespace NUMINAMATH_GPT_final_price_of_coat_is_correct_l626_62645

-- Define the conditions as constants
def original_price : ℝ := 120
def discount_rate : ℝ := 0.30
def tax_rate : ℝ := 0.15

-- Define the discounted amount calculation
def discount_amount : ℝ := original_price * discount_rate

-- Define the sale price after the discount
def sale_price : ℝ := original_price - discount_amount

-- Define the tax amount calculation on the sale price
def tax_amount : ℝ := sale_price * tax_rate

-- Define the total selling price
def total_selling_price : ℝ := sale_price + tax_amount

-- The theorem that needs to be proven
theorem final_price_of_coat_is_correct : total_selling_price = 96.6 :=
by
  sorry

end NUMINAMATH_GPT_final_price_of_coat_is_correct_l626_62645


namespace NUMINAMATH_GPT_right_triangle_sides_l626_62697

-- Definitions based on the conditions
def is_right_triangle (a b c : ℕ) : Prop := a^2 + b^2 = c^2
def perimeter (a b c : ℕ) : ℕ := a + b + c
def inscribed_circle_radius (a b c : ℕ) : ℕ := (a + b - c) / 2

-- The theorem statement
theorem right_triangle_sides (a b c : ℕ) 
  (h_perimeter : perimeter a b c = 40)
  (h_radius : inscribed_circle_radius a b c = 3)
  (h_right : is_right_triangle a b c) :
  (a = 8 ∧ b = 15 ∧ c = 17) ∨ (a = 15 ∧ b = 8 ∧ c = 17) :=
by sorry

end NUMINAMATH_GPT_right_triangle_sides_l626_62697


namespace NUMINAMATH_GPT_product_of_intersection_coords_l626_62681

open Real

-- Define the two circles
def circle1 (x y: ℝ) : Prop := x^2 - 2*x + y^2 - 10*y + 21 = 0
def circle2 (x y: ℝ) : Prop := x^2 - 8*x + y^2 - 10*y + 52 = 0

-- Prove that the product of the coordinates of intersection points equals 189
theorem product_of_intersection_coords :
  (∃ (x1 y1 x2 y2 : ℝ), circle1 x1 y1 ∧ circle2 x1 y1 ∧ circle1 x2 y2 ∧ circle2 x2 y2 ∧ x1 * y1 * x2 * y2 = 189) :=
by
  sorry

end NUMINAMATH_GPT_product_of_intersection_coords_l626_62681


namespace NUMINAMATH_GPT_pq_true_l626_62694

-- Proposition p: a^2 + b^2 < 0 is false
def p_false (a b : ℝ) : Prop := ¬ (a^2 + b^2 < 0)

-- Proposition q: (a-2)^2 + |b-3| ≥ 0 is true
def q_true (a b : ℝ) : Prop := (a - 2)^2 + |b - 3| ≥ 0

-- Theorem stating that "p ∨ q" is true
theorem pq_true (a b : ℝ) (h1 : p_false a b) (h2 : q_true a b) : (a^2 + b^2 < 0 ∨ (a - 2)^2 + |b - 3| ≥ 0) :=
by {
  sorry
}

end NUMINAMATH_GPT_pq_true_l626_62694


namespace NUMINAMATH_GPT_find_triangles_l626_62674

/-- In a triangle, if the side lengths a, b, c (a ≤ b ≤ c) are integers, form a geometric progression (i.e., b² = ac),
    and at least one of a or c is equal to 100, then the possible values for the triple (a, b, c) are:
    (49, 70, 100), (64, 80, 100), (81, 90, 100), 
    (100, 100, 100), (100, 110, 121), (100, 120, 144),
    (100, 130, 169), (100, 140, 196), (100, 150, 225), (100, 160, 256). 
-/
theorem find_triangles (a b c : ℕ) (h1 : a ≤ b ∧ b ≤ c) 
(h2 : b * b = a * c)
(h3 : a = 100 ∨ c = 100) : 
  (a = 49 ∧ b = 70 ∧ c = 100) ∨ 
  (a = 64 ∧ b = 80 ∧ c = 100) ∨ 
  (a = 81 ∧ b = 90 ∧ c = 100) ∨ 
  (a = 100 ∧ b = 100 ∧ c = 100) ∨ 
  (a = 100 ∧ b = 110 ∧ c = 121) ∨ 
  (a = 100 ∧ b = 120 ∧ c = 144) ∨ 
  (a = 100 ∧ b = 130 ∧ c = 169) ∨ 
  (a = 100 ∧ b = 140 ∧ c = 196) ∨ 
  (a = 100 ∧ b = 150 ∧ c = 225) ∨ 
  (a = 100 ∧ b = 160 ∧ c = 256) := sorry

end NUMINAMATH_GPT_find_triangles_l626_62674


namespace NUMINAMATH_GPT_flu_infection_equation_l626_62662

theorem flu_infection_equation
  (x : ℝ) :
  (1 + x)^2 = 25 :=
sorry

end NUMINAMATH_GPT_flu_infection_equation_l626_62662


namespace NUMINAMATH_GPT_count_distinct_even_numbers_l626_62606

theorem count_distinct_even_numbers : 
  ∃ c, c = 37 ∧ ∀ d1 d2 d3, d1 ≠ d2 → d2 ≠ d3 → d1 ≠ d3 → (d1 ∈ ({0, 1, 2, 3, 4, 5} : Finset ℕ)) → (d2 ∈ ({0, 1, 2, 3, 4, 5} : Finset ℕ)) → (d3 ∈ ({0, 1, 2, 3, 4, 5} : Finset ℕ)) → (∃ n : ℕ, n / 10 ^ 2 = d1 ∧ (n / 10) % 10 = d2 ∧ n % 10 = d3 ∧ n % 2 = 0) :=
sorry

end NUMINAMATH_GPT_count_distinct_even_numbers_l626_62606


namespace NUMINAMATH_GPT_min_le_mult_l626_62614

theorem min_le_mult {x y z m : ℝ} (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z)
    (hm : m = min (min (min 1 (x^9)) (y^9)) (z^7)) : m ≤ x * y^2 * z^3 :=
by
  sorry

end NUMINAMATH_GPT_min_le_mult_l626_62614


namespace NUMINAMATH_GPT_equation_of_line_AB_l626_62631

def is_midpoint (P A B : ℝ × ℝ) : Prop :=
  P = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

def on_circle (C : ℝ × ℝ) (r : ℝ) (P : ℝ × ℝ) : Prop :=
  (P.1 - C.1) ^ 2 + P.2 ^ 2 = r ^ 2

theorem equation_of_line_AB : 
  ∃ A B : ℝ × ℝ, 
    is_midpoint (2, -1) A B ∧ 
    on_circle (1, 0) 5 A ∧ 
    on_circle (1, 0) 5 B ∧ 
    ∀ x y : ℝ, (x - y - 3 = 0) ∧ 
    ∃ t : ℝ, ∃ u : ℝ, (t - u - 3 = 0) := 
sorry

end NUMINAMATH_GPT_equation_of_line_AB_l626_62631


namespace NUMINAMATH_GPT_expression_value_l626_62634

theorem expression_value :
  3 + Real.sqrt 3 + 1 / (3 + Real.sqrt 3) + 1 / (Real.sqrt 3 - 3) = 3 + 2 * Real.sqrt 3 / 3 :=
by
  sorry

end NUMINAMATH_GPT_expression_value_l626_62634


namespace NUMINAMATH_GPT_find_a_l626_62695

theorem find_a (a x y : ℤ) (h_x : x = 1) (h_y : y = -3) (h_eq : a * x - y = 1) : a = -2 := by
  -- Proof skipped
  sorry

end NUMINAMATH_GPT_find_a_l626_62695


namespace NUMINAMATH_GPT_mike_pumpkins_l626_62603

def pumpkins : ℕ :=
  let sandy_pumpkins := 51
  let total_pumpkins := 74
  total_pumpkins - sandy_pumpkins

theorem mike_pumpkins : pumpkins = 23 :=
by
  sorry

end NUMINAMATH_GPT_mike_pumpkins_l626_62603


namespace NUMINAMATH_GPT_average_score_l626_62673

theorem average_score (s1 s2 s3 : ℕ) (n : ℕ) (h1 : s1 = 115) (h2 : s2 = 118) (h3 : s3 = 115) (h4 : n = 3) :
    (s1 + s2 + s3) / n = 116 :=
by
    sorry

end NUMINAMATH_GPT_average_score_l626_62673


namespace NUMINAMATH_GPT_ratio_of_length_to_width_l626_62665

variable (P W L : ℕ)
variable (ratio : ℕ × ℕ)

theorem ratio_of_length_to_width (h1 : P = 336) (h2 : W = 70) (h3 : 2 * L + 2 * W = P) : ratio = (7, 5) :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_length_to_width_l626_62665


namespace NUMINAMATH_GPT_initial_amount_l626_62647

theorem initial_amount (A : ℝ) (h : (9 / 8) * (9 / 8) * A = 40500) : 
  A = 32000 :=
sorry

end NUMINAMATH_GPT_initial_amount_l626_62647


namespace NUMINAMATH_GPT_sum_A_B_l626_62656

noncomputable def num_four_digit_odd_numbers_divisible_by_3 : ℕ := 1500
noncomputable def num_four_digit_multiples_of_7 : ℕ := 1286

theorem sum_A_B (A B : ℕ) :
  A = num_four_digit_odd_numbers_divisible_by_3 →
  B = num_four_digit_multiples_of_7 →
  A + B = 2786 :=
by
  intros hA hB
  rw [hA, hB]
  exact rfl

end NUMINAMATH_GPT_sum_A_B_l626_62656


namespace NUMINAMATH_GPT_original_strip_length_l626_62641

theorem original_strip_length (x : ℝ) 
  (h1 : 3 + x + 3 + x + 3 + x + 3 + x + 3 = 27) : 
  4 * 9 + 4 * 3 = 57 := 
  sorry

end NUMINAMATH_GPT_original_strip_length_l626_62641


namespace NUMINAMATH_GPT_kelseys_sister_age_in_2021_l626_62619

-- Definitions based on given conditions
def kelsey_birth_year : ℕ := 1999 - 25
def sister_birth_year : ℕ := kelsey_birth_year - 3

-- Prove that Kelsey's older sister is 50 years old in 2021
theorem kelseys_sister_age_in_2021 : (2021 - sister_birth_year) = 50 :=
by
  -- Add proof here
  sorry

end NUMINAMATH_GPT_kelseys_sister_age_in_2021_l626_62619


namespace NUMINAMATH_GPT_production_bottles_l626_62658

-- Definitions from the problem conditions
def machines_production_rate (machines : ℕ) (rate : ℕ) : ℕ := rate / machines
def total_production (machines rate minutes : ℕ) : ℕ := machines * rate * minutes

-- Theorem to prove the solution
theorem production_bottles :
  machines_production_rate 6 300 = 50 →
  total_production 10 50 4 = 2000 :=
by
  intro h
  have : 10 * 50 * 4 = 2000 := by norm_num
  exact this

end NUMINAMATH_GPT_production_bottles_l626_62658


namespace NUMINAMATH_GPT_probability_point_inside_circle_l626_62684

theorem probability_point_inside_circle :
  (∃ (m n : ℕ), 1 ≤ m ∧ m ≤ 6 ∧ 1 ≤ n ∧ n ≤ 6) →
  (∃ (P : ℚ), P = 2/9) :=
by
  sorry

end NUMINAMATH_GPT_probability_point_inside_circle_l626_62684


namespace NUMINAMATH_GPT_ounces_of_wax_for_car_l626_62679

noncomputable def ounces_wax_for_SUV : ℕ := 4
noncomputable def initial_wax_amount : ℕ := 11
noncomputable def wax_spilled : ℕ := 2
noncomputable def wax_left_after_detailing : ℕ := 2
noncomputable def total_wax_used : ℕ := initial_wax_amount - wax_spilled - wax_left_after_detailing

theorem ounces_of_wax_for_car :
  (initial_wax_amount - wax_spilled - wax_left_after_detailing) - ounces_wax_for_SUV = 3 :=
by
  sorry

end NUMINAMATH_GPT_ounces_of_wax_for_car_l626_62679


namespace NUMINAMATH_GPT_average_speed_joey_round_trip_l626_62621

noncomputable def average_speed_round_trip
  (d : ℝ) (t₁ : ℝ) (r : ℝ) (s₂ : ℝ) : ℝ :=
  2 * d / (t₁ + d / s₂)

-- Lean statement for the proof problem
theorem average_speed_joey_round_trip :
  average_speed_round_trip 6 1 6 12 = 8 := sorry

end NUMINAMATH_GPT_average_speed_joey_round_trip_l626_62621


namespace NUMINAMATH_GPT_factor_theorem_l626_62698

-- Define the polynomial function f(x)
def f (k : ℚ) (x : ℚ) : ℚ := k * x^3 + 27 * x^2 - k * x + 55

-- State the theorem to find the value of k such that x+5 is a factor of f(x)
theorem factor_theorem (k : ℚ) : f k (-5) = 0 ↔ k = 73 / 12 :=
by sorry

end NUMINAMATH_GPT_factor_theorem_l626_62698


namespace NUMINAMATH_GPT_unique_positive_real_solution_l626_62687

theorem unique_positive_real_solution :
  ∃! x : ℝ, 0 < x ∧ (x^8 + 5 * x^7 + 10 * x^6 + 2023 * x^5 - 2021 * x^4 = 0) := sorry

end NUMINAMATH_GPT_unique_positive_real_solution_l626_62687


namespace NUMINAMATH_GPT_correct_number_of_outfits_l626_62686

def num_shirts : ℕ := 7
def num_pants : ℕ := 7
def num_hats : ℕ := 7
def num_colors : ℕ := 7

def total_outfits : ℕ := num_shirts * num_pants * num_hats
def invalid_outfits : ℕ := num_colors
def valid_outfits : ℕ := total_outfits - invalid_outfits

theorem correct_number_of_outfits : valid_outfits = 336 :=
by {
  -- sorry can be removed when providing the proof.
  sorry
}

end NUMINAMATH_GPT_correct_number_of_outfits_l626_62686


namespace NUMINAMATH_GPT_yang_hui_problem_solution_l626_62620

theorem yang_hui_problem_solution (x : ℕ) (h : x * (x - 1) = 650) : x * (x - 1) = 650 :=
by
  exact h

end NUMINAMATH_GPT_yang_hui_problem_solution_l626_62620


namespace NUMINAMATH_GPT_friend_pays_correct_percentage_l626_62636

theorem friend_pays_correct_percentage (adoption_fee : ℝ) (james_payment : ℝ) (friend_payment : ℝ) 
  (h1 : adoption_fee = 200) 
  (h2 : james_payment = 150)
  (h3 : friend_payment = adoption_fee - james_payment) : 
  (friend_payment / adoption_fee) * 100 = 25 :=
by
  sorry

end NUMINAMATH_GPT_friend_pays_correct_percentage_l626_62636
