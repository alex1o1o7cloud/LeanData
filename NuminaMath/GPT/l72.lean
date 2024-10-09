import Mathlib

namespace find_x_l72_7298

theorem find_x : ∃ (x : ℚ), (3 * x - 5) / 7 = 15 ∧ x = 110 / 3 := by
  sorry

end find_x_l72_7298


namespace integer_solution_zero_l72_7248

theorem integer_solution_zero (x y z : ℤ) (h : x^2 + y^2 + z^2 = 2 * x * y * z) : x = 0 ∧ y = 0 ∧ z = 0 := 
sorry

end integer_solution_zero_l72_7248


namespace second_hose_correct_l72_7238

/-- Define the problem parameters -/
def first_hose_rate : ℕ := 50
def initial_hours : ℕ := 3
def additional_hours : ℕ := 2
def total_capacity : ℕ := 390

/-- Define the total hours the first hose was used -/
def total_hours (initial_hours additional_hours : ℕ) : ℕ := initial_hours + additional_hours

/-- Define the amount of water sprayed by the first hose -/
def first_hose_total (first_hose_rate initial_hours additional_hours : ℕ) : ℕ :=
  first_hose_rate * (initial_hours + additional_hours)

/-- Define the remaining water needed to fill the pool -/
def remaining_water (total_capacity first_hose_total : ℕ) : ℕ :=
  total_capacity - first_hose_total

/-- Define the additional water sprayed by the first hose during the last 2 hours -/
def additional_first_hose (first_hose_rate additional_hours : ℕ) : ℕ :=
  first_hose_rate * additional_hours

/-- Define the water sprayed by the second hose -/
def second_hose_total (remaining_water additional_first_hose : ℕ) : ℕ :=
  remaining_water - additional_first_hose

/-- Define the rate of the second hose (output) -/
def second_hose_rate (second_hose_total additional_hours : ℕ) : ℕ :=
  second_hose_total / additional_hours

/-- Define the theorem we want to prove -/
theorem second_hose_correct :
  second_hose_rate
    (second_hose_total
        (remaining_water total_capacity (first_hose_total first_hose_rate initial_hours additional_hours))
        (additional_first_hose first_hose_rate additional_hours))
    additional_hours = 20 := by
  sorry

end second_hose_correct_l72_7238


namespace div_gt_sum_div_sq_l72_7200

theorem div_gt_sum_div_sq (n d d' : ℕ) (h₁ : d' > d) (h₂ : d ∣ n) (h₃ : d' ∣ n) : 
  d' > d + d * d / n :=
by 
  sorry

end div_gt_sum_div_sq_l72_7200


namespace solve_stream_speed_l72_7231

noncomputable def boat_travel (v : ℝ) : Prop :=
  let downstream_speed := 12 + v
  let upstream_speed := 12 - v
  let downstream_time := 60 / downstream_speed
  let upstream_time := 60 / upstream_speed
  upstream_time - downstream_time = 2

theorem solve_stream_speed : ∃ v : ℝ, boat_travel v ∧ v = 2.31 :=
by {
  sorry
}

end solve_stream_speed_l72_7231


namespace boat_travels_125_km_downstream_l72_7214

/-- The speed of the boat in still water is 20 km/hr -/
def boat_speed_still_water : ℝ := 20

/-- The speed of the stream is 5 km/hr -/
def stream_speed : ℝ := 5

/-- The total time taken downstream is 5 hours -/
def total_time_downstream : ℝ := 5

/-- The effective speed of the boat downstream -/
def effective_speed_downstream : ℝ := boat_speed_still_water + stream_speed

/-- The distance the boat travels downstream -/
def distance_downstream : ℝ := effective_speed_downstream * total_time_downstream

/-- The boat travels 125 km downstream -/
theorem boat_travels_125_km_downstream :
  distance_downstream = 125 := 
sorry

end boat_travels_125_km_downstream_l72_7214


namespace arcsin_sqrt_three_over_two_l72_7276

theorem arcsin_sqrt_three_over_two : Real.arcsin (Real.sqrt 3 / 2) = Real.pi / 3 :=
by
  -- The proof is omitted
  sorry

end arcsin_sqrt_three_over_two_l72_7276


namespace workers_problem_l72_7210

theorem workers_problem (W : ℕ) (A : ℕ) :
  (W * 45 = A) ∧ ((W + 10) * 35 = A) → W = 35 :=
by
  sorry

end workers_problem_l72_7210


namespace games_within_division_l72_7279

/-- 
Given a baseball league with two four-team divisions,
where each team plays N games against other teams in its division,
and M games against teams in the other division.
Given that N > 2M and M > 6, and each team plays a total of 92 games in a season,
prove that each team plays 60 games within its own division.
-/
theorem games_within_division (N M : ℕ) (hN : N > 2 * M) (hM : M > 6) (h_total : 3 * N + 4 * M = 92) :
  3 * N = 60 :=
by
  -- The proof is omitted.
  sorry

end games_within_division_l72_7279


namespace average_score_of_class_l72_7218

theorem average_score_of_class : 
  ∀ (total_students assigned_students make_up_students : ℕ)
    (assigned_avg_score make_up_avg_score : ℚ),
    total_students = 100 →
    assigned_students = 70 →
    make_up_students = total_students - assigned_students →
    assigned_avg_score = 60 →
    make_up_avg_score = 80 →
    (assigned_students * assigned_avg_score + make_up_students * make_up_avg_score) / total_students = 66 :=
by
  intro total_students assigned_students make_up_students assigned_avg_score make_up_avg_score
  intros h_total_students h_assigned_students h_make_up_students h_assigned_avg_score h_make_up_avg_score
  sorry

end average_score_of_class_l72_7218


namespace distance_after_time_l72_7293

noncomputable def Adam_speed := 12 -- speed in mph
noncomputable def Simon_speed := 6 -- speed in mph
noncomputable def time_when_100_miles_apart := 100 / 15 -- hours

theorem distance_after_time (x : ℝ) : 
  (Adam_speed * x)^2 + (Simon_speed * x)^2 = 100^2 ->
  x = time_when_100_miles_apart := 
by
  sorry

end distance_after_time_l72_7293


namespace sin_cos_value_l72_7209

-- Given function definition
noncomputable def f (α : ℝ) (x : ℝ) : ℝ := x^2 + (Real.sin α - 2 * Real.cos α) * x + 1

-- Definitions and proof problem statement
theorem sin_cos_value (α : ℝ) : 
  (∀ x : ℝ, f α x = f α (-x)) → (Real.sin α * Real.cos α = 2 / 5) :=
by
  intro h_even
  sorry

end sin_cos_value_l72_7209


namespace lightbulb_stops_on_friday_l72_7202

theorem lightbulb_stops_on_friday
  (total_hours : ℕ) (daily_usage : ℕ) (start_day : ℕ) (stops_day : ℕ)
  (h_total_hours : total_hours = 24999)
  (h_daily_usage : daily_usage = 2)
  (h_start_day : start_day = 1) : 
  stops_day = 5 := by
  sorry

end lightbulb_stops_on_friday_l72_7202


namespace equation_of_line_l72_7211

variable {a b k T : ℝ}

theorem equation_of_line (h_b_ne_zero : b ≠ 0)
  (h_line_passing_through : ∃ (line : ℝ → ℝ), line (-a) = b)
  (h_triangle_area : ∃ (h : ℝ), T = 1 / 2 * ka * (h - b))
  (h_base_length : ∃ (base : ℝ), base = ka) :
  ∃ (x y : ℝ), 2 * T * x - k * a^2 * y + k * a^2 * b + 2 * a * T = 0 :=
sorry

end equation_of_line_l72_7211


namespace find_vector_from_origin_to_line_l72_7284

theorem find_vector_from_origin_to_line :
  ∃ t : ℝ, (3 * t + 1, 2 * t + 3) = (16, 32 / 3) ∧
  ∃ k : ℝ, (16, 32 / 3) = (3 * k, 2 * k) :=
sorry

end find_vector_from_origin_to_line_l72_7284


namespace angle_remains_unchanged_l72_7229

-- Definition of magnification condition (though it does not affect angle in mathematics, we state it as given)
def magnifying_glass (magnification : ℝ) (initial_angle : ℝ) : ℝ := 
  initial_angle  -- Magnification does not change the angle in this context.

-- Given condition
def initial_angle : ℝ := 30

-- Theorem we want to prove
theorem angle_remains_unchanged (magnification : ℝ) (h_magnify : magnification = 100) :
  magnifying_glass magnification initial_angle = initial_angle :=
by
  sorry

end angle_remains_unchanged_l72_7229


namespace wages_problem_l72_7208

variable {S W_y W_x : ℝ}
variable {D_x : ℝ}

theorem wages_problem
  (h1 : S = 45 * W_y)
  (h2 : S = 20 * (W_x + W_y))
  (h3 : S = D_x * W_x) :
  D_x = 36 :=
sorry

end wages_problem_l72_7208


namespace exists_f_prime_eq_inverses_l72_7240

theorem exists_f_prime_eq_inverses (f : ℝ → ℝ) (a b : ℝ)
  (h1 : 0 < a)
  (h2 : a < b)
  (h3 : ContinuousOn f (Set.Icc a b))
  (h4 : DifferentiableOn ℝ f (Set.Ioo a b)) :
  ∃ c ∈ Set.Ioo a b, (deriv f c) = (1 / (a - c)) + (1 / (b - c)) + (1 / (a + b)) :=
by
  sorry

end exists_f_prime_eq_inverses_l72_7240


namespace log_three_nine_cubed_l72_7274

theorem log_three_nine_cubed : ∀ (log : ℕ → ℕ → ℕ), log 3 9 = 2 → log 3 3 = 1 → (∀ (a b n : ℕ), log a (b ^ n) = n * log a b) → log 3 (9 ^ 3) = 6 :=
by
  intros log h1 h2 h3
  sorry

end log_three_nine_cubed_l72_7274


namespace total_tape_length_is_230_l72_7292

def tape_length (n : ℕ) (len_piece : ℕ) (overlap : ℕ) : ℕ :=
  len_piece + (n - 1) * (len_piece - overlap)

theorem total_tape_length_is_230 :
  tape_length 15 20 5 = 230 := 
    sorry

end total_tape_length_is_230_l72_7292


namespace proposition_p_neither_sufficient_nor_necessary_l72_7277

-- Define propositions p and q
def p (m : ℝ) : Prop := m = -1
def q (m : ℝ) : Prop := ∀ x y : ℝ, (x - 1 = 0) ∧ (x + m^2 * y = 0) → ∀ x' y' : ℝ, x' = x ∧ y' = y → (x - 1) * (x + m^2 * y) = 0

-- Main theorem statement
theorem proposition_p_neither_sufficient_nor_necessary (m : ℝ) : ¬ (p m → q m) ∧ ¬ (q m → p m) :=
by
  sorry

end proposition_p_neither_sufficient_nor_necessary_l72_7277


namespace initial_candies_is_720_l72_7241

-- Definitions according to the conditions
def candies_remaining_after_day_n (initial_candies : ℕ) (n : ℕ) : ℕ :=
  match n with
  | 1 => initial_candies / 2
  | 2 => (initial_candies / 2) / 3
  | 3 => (initial_candies / 2) / 3 / 4
  | 4 => (initial_candies / 2) / 3 / 4 / 5
  | 5 => (initial_candies / 2) / 3 / 4 / 5 / 6
  | _ => 0 -- For days beyond the fifth, this is nonsensical

-- Proof statement
theorem initial_candies_is_720 : ∀ (initial_candies : ℕ), candies_remaining_after_day_n initial_candies 5 = 1 → initial_candies = 720 :=
by
  intros initial_candies h
  sorry

end initial_candies_is_720_l72_7241


namespace tan_sum_eq_one_l72_7224

theorem tan_sum_eq_one (a b : ℝ) (h1 : Real.tan a = 1 / 2) (h2 : Real.tan b = 1 / 3) :
    Real.tan (a + b) = 1 := 
by
  sorry

end tan_sum_eq_one_l72_7224


namespace solution_largest_a_exists_polynomial_l72_7289

def largest_a_exists_polynomial : Prop :=
  ∃ (P : ℝ → ℝ) (a b c d e : ℝ),
    (∀ x, P x = a * x^4 + b * x^3 + c * x^2 + d * x + e) ∧
    (∀ x, -1 ≤ x ∧ x ≤ 1 → 0 ≤ P x ∧ P x ≤ 1) ∧
    a = 4

theorem solution_largest_a_exists_polynomial : largest_a_exists_polynomial :=
  sorry

end solution_largest_a_exists_polynomial_l72_7289


namespace number_of_valid_ns_l72_7263

theorem number_of_valid_ns :
  ∃ (S : Finset ℕ), S.card = 13 ∧ ∀ n ∈ S, n ≤ 1000 ∧ Nat.floor (995 / n) + Nat.floor (996 / n) + Nat.floor (997 / n) % 4 ≠ 0 :=
by
  sorry

end number_of_valid_ns_l72_7263


namespace percentage_increase_l72_7247

theorem percentage_increase (original_interval : ℕ) (new_interval : ℕ) 
  (h1 : original_interval = 30) (h2 : new_interval = 45) :
  ((new_interval - original_interval) / original_interval) * 100 = 50 := 
by 
  -- Provide the proof here
  sorry

end percentage_increase_l72_7247


namespace cone_volume_l72_7294

theorem cone_volume (V_f : ℝ) (A1 A2 : ℝ) (V : ℝ)
  (h1 : V_f = 78)
  (h2 : A1 = 9 * A2) :
  V = 81 :=
sorry

end cone_volume_l72_7294


namespace abs_m_plus_one_l72_7273

theorem abs_m_plus_one (m : ℝ) (h : |m| = m + 1) : (4 * m - 1) ^ 4 = 81 := by
  sorry

end abs_m_plus_one_l72_7273


namespace percentage_of_teachers_with_neither_issue_l72_7244

theorem percentage_of_teachers_with_neither_issue 
  (total_teachers : ℕ)
  (teachers_with_bp : ℕ)
  (teachers_with_stress : ℕ)
  (teachers_with_both : ℕ)
  (h1 : total_teachers = 150)
  (h2 : teachers_with_bp = 90)
  (h3 : teachers_with_stress = 60)
  (h4 : teachers_with_both = 30) :
  let neither_issue_teachers := total_teachers - (teachers_with_bp + teachers_with_stress - teachers_with_both)
  let percentage := (neither_issue_teachers * 100) / total_teachers
  percentage = 20 :=
by
  -- skipping the proof
  sorry

end percentage_of_teachers_with_neither_issue_l72_7244


namespace cross_ratio_eq_one_implies_equal_points_l72_7285

-- Definitions corresponding to the points and hypothesis.
variable {A B C D : ℝ}
variable (h_line : collinear ℝ A B C D) (h_cross_ratio : cross_ratio A B C D = 1)

-- The theorem statement based on the given problem and solution.
theorem cross_ratio_eq_one_implies_equal_points :
  A = B ∨ C = D :=
sorry

end cross_ratio_eq_one_implies_equal_points_l72_7285


namespace solve_for_y_l72_7251

theorem solve_for_y (x y : ℝ) (h1 : x = 2) (h2 : x^(3*y) = 8) : y = 1 :=
by {
  -- Apply the conditions and derive the proof
  sorry
}

end solve_for_y_l72_7251


namespace max_distinct_tangent_counts_l72_7203

-- Define the types and conditions for our circles and tangents
structure Circle where
  radius : ℝ

def circle1 : Circle := { radius := 3 }
def circle2 : Circle := { radius := 4 }

-- Define the statement to be proved
theorem max_distinct_tangent_counts :
  ∃ (k : ℕ), k = 5 :=
sorry

end max_distinct_tangent_counts_l72_7203


namespace max_total_profit_max_avg_annual_profit_l72_7222

noncomputable def total_profit (x : ℕ) : ℝ := - (x : ℝ)^2 + 18 * x - 36
noncomputable def avg_annual_profit (x : ℕ) : ℝ := (total_profit x) / x

theorem max_total_profit : ∃ x : ℕ, total_profit x = 45 ∧ x = 9 :=
  by sorry

theorem max_avg_annual_profit : ∃ x : ℕ, avg_annual_profit x = 6 ∧ x = 6 :=
  by sorry

end max_total_profit_max_avg_annual_profit_l72_7222


namespace tan_arith_seq_l72_7252

theorem tan_arith_seq (x y z : ℝ)
  (h₁ : y = x + π / 3)
  (h₂ : z = x + 2 * π / 3) :
  (Real.tan x * Real.tan y) + (Real.tan y * Real.tan z) + (Real.tan z * Real.tan x) = -3 :=
sorry

end tan_arith_seq_l72_7252


namespace trig_identity_proof_l72_7260

theorem trig_identity_proof : 
  (Real.cos (70 * Real.pi / 180) * Real.sin (80 * Real.pi / 180) + 
   Real.cos (20 * Real.pi / 180) * Real.sin (10 * Real.pi / 180) = 1 / 2) :=
by
  sorry

end trig_identity_proof_l72_7260


namespace mean_score_74_l72_7261

theorem mean_score_74 
  (M SD : ℝ)
  (h1 : 58 = M - 2 * SD)
  (h2 : 98 = M + 3 * SD) : 
  M = 74 :=
by
  sorry

end mean_score_74_l72_7261


namespace female_kittens_count_l72_7237

theorem female_kittens_count (initial_cats total_cats male_kittens female_kittens : ℕ)
  (h1 : initial_cats = 2)
  (h2 : total_cats = 7)
  (h3 : male_kittens = 2)
  (h4 : female_kittens = total_cats - initial_cats - male_kittens) :
  female_kittens = 3 :=
by
  sorry

end female_kittens_count_l72_7237


namespace twenty_five_percent_of_x_l72_7239

-- Define the number x and the conditions
variable (x : ℝ)
variable (h : x - (3/4) * x = 100)

-- The theorem statement
theorem twenty_five_percent_of_x : (1/4) * x = 100 :=
by 
  -- Assume x satisfies the given condition
  sorry

end twenty_five_percent_of_x_l72_7239


namespace star_eq_122_l72_7221

noncomputable def solveForStar (star : ℕ) : Prop :=
  45 - (28 - (37 - (15 - star))) = 56

theorem star_eq_122 : solveForStar 122 :=
by
  -- proof
  sorry

end star_eq_122_l72_7221


namespace cos_double_angle_l72_7266

theorem cos_double_angle (θ : ℝ) (h : ∑' n : ℕ, (Real.cos θ)^(2*n) = 8) :
  Real.cos (2 * θ) = 3 / 4 :=
sorry

end cos_double_angle_l72_7266


namespace largest_number_after_removal_l72_7246

theorem largest_number_after_removal :
  ∀ (s : Nat), s = 1234567891011121314151617181920 -- representing the start of the sequence
  → true
  := by
    sorry

end largest_number_after_removal_l72_7246


namespace max_S_n_l72_7280

noncomputable def S (n : ℕ) : ℝ := sorry  -- Definition of the sum of the first n terms

theorem max_S_n (S : ℕ → ℝ) (h16 : S 16 > 0) (h17 : S 17 < 0) : ∃ n, S n = S 8 :=
sorry

end max_S_n_l72_7280


namespace black_female_pigeons_more_than_males_l72_7223

theorem black_female_pigeons_more_than_males:
  let total_pigeons := 70
  let black_pigeons := total_pigeons / 2
  let black_male_percentage := 20 / 100
  let black_male_pigeons := black_pigeons * black_male_percentage
  let black_female_pigeons := black_pigeons - black_male_pigeons
  black_female_pigeons - black_male_pigeons = 21 := by
{
  let total_pigeons := 70
  let black_pigeons := total_pigeons / 2
  let black_male_percentage := 20 / 100
  let black_male_pigeons := black_pigeons * black_male_percentage
  let black_female_pigeons := black_pigeons - black_male_pigeons
  show black_female_pigeons - black_male_pigeons = 21
  sorry
}

end black_female_pigeons_more_than_males_l72_7223


namespace inequality_proof_l72_7255

theorem inequality_proof (x y : ℝ) (h1 : x > y) (h2 : y > 2 / (x - y)) : x^2 > y^2 + 4 :=
by
  sorry

end inequality_proof_l72_7255


namespace find_y_l72_7236

-- Define the problem conditions
variable (x y : ℕ)
variable (hx : x > 0)
variable (hy : y > 0)
variable (rem_eq : x % y = 3)
variable (div_eq : (x : ℝ) / y = 96.12)

-- The theorem to prove
theorem find_y : y = 25 :=
sorry

end find_y_l72_7236


namespace find_ratio_l72_7275

variables {a b c d : ℝ}
variables (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d)
variables (h5 : (5 * a + b) / (5 * c + d) = (6 * a + b) / (6 * c + d))
variables (h6 : (7 * a + b) / (7 * c + d) = 9)

theorem find_ratio (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d)
    (h5 : (5 * a + b) / (5 * c + d) = (6 * a + b) / (6 * c + d))
    (h6 : (7 * a + b) / (7 * c + d) = 9) :
    (9 * a + b) / (9 * c + d) = 9 := 
by {
    sorry
}

end find_ratio_l72_7275


namespace parabola_intersects_xaxis_at_least_one_l72_7257

theorem parabola_intersects_xaxis_at_least_one {a b c : ℝ} (h : a ≠ b ∧ b ≠ c ∧ c ≠ a) :
  (∃ x1 x2, x1 ≠ x2 ∧ (a * x1^2 + 2 * b * x1 + c = 0) ∧ (a * x2^2 + 2 * b * x2 + c = 0)) ∨
  (∃ x1 x2, x1 ≠ x2 ∧ (b * x1^2 + 2 * c * x1 + a = 0) ∧ (b * x2^2 + 2 * c * x2 + a = 0)) ∨
  (∃ x1 x2, x1 ≠ x2 ∧ (c * x1^2 + 2 * a * x1 + b = 0) ∧ (c * x2^2 + 2 * a * x2 + b = 0)) :=
by
  sorry

end parabola_intersects_xaxis_at_least_one_l72_7257


namespace factorize_l72_7287

theorem factorize (a : ℝ) : 5*a^3 - 125*a = 5*a*(a + 5)*(a - 5) :=
sorry

end factorize_l72_7287


namespace shaded_area_calc_l72_7281

theorem shaded_area_calc (r1_area r2_area overlap_area circle_area : ℝ)
  (h_r1_area : r1_area = 36)
  (h_r2_area : r2_area = 28)
  (h_overlap_area : overlap_area = 21)
  (h_circle_area : circle_area = Real.pi) : 
  (r1_area + r2_area - overlap_area - circle_area) = 64 - Real.pi :=
by
  sorry

end shaded_area_calc_l72_7281


namespace algebraic_expression_value_l72_7286

theorem algebraic_expression_value
  (x y : ℚ)
  (h : |2 * x - 3 * y + 1| + (x + 3 * y + 5)^2 = 0) :
  (-2 * x * y)^2 * (-y^2) * 6 * x * y^2 = 192 :=
  sorry

end algebraic_expression_value_l72_7286


namespace largest_angle_measure_l72_7228

theorem largest_angle_measure (v : ℝ) (h : v > 3/2) :
  ∃ θ, θ = Real.arccos ((4 * v - 4) / (2 * Real.sqrt ((2 * v - 3) * (4 * v - 4)))) ∧
       θ = π - θ ∧
       θ = Real.arccos ((2 * v - 3) / (2 * Real.sqrt ((2 * v + 3) * (4 * v - 4)))) := 
sorry

end largest_angle_measure_l72_7228


namespace remainder_of_2_pow_33_mod_9_l72_7265

theorem remainder_of_2_pow_33_mod_9 : 2^33 % 9 = 8 := by
  sorry

end remainder_of_2_pow_33_mod_9_l72_7265


namespace tangent_line_exponential_passing_through_origin_l72_7253

theorem tangent_line_exponential_passing_through_origin :
  ∃ (p : ℝ × ℝ) (m : ℝ), 
  (p = (1, Real.exp 1)) ∧ (m = Real.exp 1) ∧ 
  (∀ x : ℝ, x ≠ 1 → ¬ (∃ k : ℝ, k = (Real.exp x - 0) / (x - 0) ∧ k = Real.exp x)) :=
by 
  sorry

end tangent_line_exponential_passing_through_origin_l72_7253


namespace unique_solution_triple_l72_7288

def satisfies_system (x y z : ℝ) :=
  x^3 = 3 * x - 12 * y + 50 ∧
  y^3 = 12 * y + 3 * z - 2 ∧
  z^3 = 27 * z + 27 * x

theorem unique_solution_triple (x y z : ℝ) :
  satisfies_system x y z ↔ (x = 2 ∧ y = 4 ∧ z = 6) :=
by sorry

end unique_solution_triple_l72_7288


namespace same_solution_m_iff_m_eq_2_l72_7219

theorem same_solution_m_iff_m_eq_2 (m y : ℝ) (h1 : my - 2 = 4) (h2 : y - 2 = 1) : m = 2 :=
by {
  sorry
}

end same_solution_m_iff_m_eq_2_l72_7219


namespace option_b_correct_l72_7262

theorem option_b_correct (a : ℝ) : (-a)^3 / (-a)^2 = -a :=
by sorry

end option_b_correct_l72_7262


namespace geometric_sequence_sum_l72_7270

theorem geometric_sequence_sum (a : ℕ → ℝ) (h_geom : ∀ n, a (n + 1) ^ 2 = a n * a (n + 2))
  (h_pos : ∀ n, 0 < a n) (h_given : a 1 * a 5 + 2 * a 3 * a 6 + a 1 * a 11 = 16) :
  a 3 + a 6 = 4 := 
sorry

end geometric_sequence_sum_l72_7270


namespace discriminant_is_four_l72_7226

-- Define the quadratic equation components
def quadratic_a (a : ℝ) := 1
def quadratic_b (a : ℝ) := 2 * a
def quadratic_c (a : ℝ) := a^2 - 1

-- Define the discriminant of the quadratic equation
def discriminant (a : ℝ) := quadratic_b a ^ 2 - 4 * quadratic_a a * quadratic_c a

-- Statement to prove: The discriminant is 4
theorem discriminant_is_four (a : ℝ) : discriminant a = 4 :=
by {
  sorry
}

end discriminant_is_four_l72_7226


namespace proof_l72_7269

noncomputable def question (a b c : ℂ) : ℂ := (a^3 + b^3 + c^3) / (a * b * c)

theorem proof (a b c : ℂ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0)
  (h4 : a + b + c = 15)
  (h5 : (a - b)^2 + (a - c)^2 + (b - c)^2 = 2 * a * b * c) :
  question a b c = 18 :=
by
  sorry

end proof_l72_7269


namespace average_age_of_5_l72_7299

theorem average_age_of_5 (h1 : 19 * 15 = 285) (h2 : 9 * 16 = 144) (h3 : 15 = 71) :
    (285 - 144 - 71) / 5 = 14 :=
sorry

end average_age_of_5_l72_7299


namespace mayor_cup_num_teams_l72_7271

theorem mayor_cup_num_teams (x : ℕ) (h : x * (x - 1) / 2 = 21) : 
    ∃ x, x * (x - 1) / 2 = 21 := 
by
  sorry

end mayor_cup_num_teams_l72_7271


namespace geometric_sequence_proof_l72_7272

-- Define a geometric sequence with first term 1 and common ratio q with |q| ≠ 1
noncomputable def geometric_sequence (q : ℝ) (n : ℕ) : ℝ :=
  if h : |q| ≠ 1 then (1 : ℝ) * q ^ (n - 1) else 0

-- m should be 11 given the conditions
theorem geometric_sequence_proof (q : ℝ) (m : ℕ) (h : |q| ≠ 1) 
  (hm : geometric_sequence q m = geometric_sequence q 1 * geometric_sequence q 2 * geometric_sequence q 3 * geometric_sequence q 4 * geometric_sequence q 5 ) : 
  m = 11 :=
by
  sorry

end geometric_sequence_proof_l72_7272


namespace problem_power_function_l72_7278

-- Defining the conditions
variable {f : ℝ → ℝ}
variable (a : ℝ)
variable (h₁ : ∀ x, f x = x^a)
variable (h₂ : f 2 = Real.sqrt 2)

-- Stating what we need to prove
theorem problem_power_function : f 4 = 2 :=
by sorry

end problem_power_function_l72_7278


namespace determine_value_of_m_l72_7205

theorem determine_value_of_m (m : ℤ) :
  2^2002 - 2^2000 - 2^1999 + 2^1998 = m * 2^1998 ↔ m = 11 := 
sorry

end determine_value_of_m_l72_7205


namespace son_l72_7206

noncomputable def my_age_in_years : ℕ := 84
noncomputable def total_age_in_years : ℕ := 140
noncomputable def months_in_a_year : ℕ := 12
noncomputable def weeks_in_a_year : ℕ := 52

theorem son's_age_in_weeks (G_d S_m G_m S_y : ℕ) (G_y : ℚ) :
  G_d = S_m →
  G_m = my_age_in_years * months_in_a_year →
  G_y = (G_m : ℚ) / months_in_a_year →
  G_y + S_y + my_age_in_years = total_age_in_years →
  S_y * weeks_in_a_year = 2548 :=
by
  intros h1 h2 h3 h4
  sorry

end son_l72_7206


namespace least_number_to_subtract_l72_7215

theorem least_number_to_subtract (n : ℕ) (p : ℕ) (hdiv : p = 47) (hn : n = 929) 
: ∃ k, n - 44 = k * p := by
  sorry

end least_number_to_subtract_l72_7215


namespace room_length_l72_7258

theorem room_length (w : ℝ) (cost_rate : ℝ) (total_cost : ℝ) (h : w = 4) (h1 : cost_rate = 800) (h2 : total_cost = 17600) : 
  let L := total_cost / (w * cost_rate)
  L = 5.5 :=
by
  sorry

end room_length_l72_7258


namespace conscript_from_western_village_l72_7259

/--
Given:
- The population of the northern village is 8758
- The population of the western village is 7236
- The population of the southern village is 8356
- The total number of conscripts needed is 378

Prove that the number of people to be conscripted from the western village is 112.
-/
theorem conscript_from_western_village (hnorth : ℕ) (hwest : ℕ) (hsouth : ℕ) (hconscripts : ℕ)
    (htotal : hnorth + hwest + hsouth = 24350) :
    let prop := (hwest / (hnorth + hwest + hsouth)) * hconscripts
    hnorth = 8758 → hwest = 7236 → hsouth = 8356 → hconscripts = 378 → prop = 112 :=
by
  intros
  simp_all
  sorry

end conscript_from_western_village_l72_7259


namespace jerry_remaining_debt_l72_7234

variable (two_months_ago_payment last_month_payment total_debt : ℕ)

def remaining_debt : ℕ := total_debt - (two_months_ago_payment + last_month_payment)

theorem jerry_remaining_debt :
  two_months_ago_payment = 12 →
  last_month_payment = 12 + 3 →
  total_debt = 50 →
  remaining_debt two_months_ago_payment last_month_payment total_debt = 23 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end jerry_remaining_debt_l72_7234


namespace remaining_days_temperature_l72_7216

theorem remaining_days_temperature :
  let avg_temp := 60
  let total_days := 7
  let temp_day1 := 40
  let temp_day2 := 40
  let temp_day3 := 40
  let temp_day4 := 80
  let temp_day5 := 80
  let total_temp := avg_temp * total_days
  let temp_first_five_days := temp_day1 + temp_day2 + temp_day3 + temp_day4 + temp_day5
  total_temp - temp_first_five_days = 140 :=
by
  -- proof is omitted
  sorry

end remaining_days_temperature_l72_7216


namespace C_should_pay_correct_amount_l72_7283

def A_oxen_months : ℕ := 10 * 7
def B_oxen_months : ℕ := 12 * 5
def C_oxen_months : ℕ := 15 * 3
def D_oxen_months : ℕ := 20 * 6

def total_rent : ℚ := 225

def C_share_of_rent : ℚ :=
  total_rent * (C_oxen_months : ℚ) / (A_oxen_months + B_oxen_months + C_oxen_months + D_oxen_months)

theorem C_should_pay_correct_amount : C_share_of_rent = 225 * (45 : ℚ) / 295 := by
  sorry

end C_should_pay_correct_amount_l72_7283


namespace solve_inequality_l72_7295

theorem solve_inequality (x : ℝ) :
  -2 * x^2 - x + 6 ≥ 0 ↔ -2 ≤ x ∧ x ≤ 3 / 2 :=
sorry

end solve_inequality_l72_7295


namespace neha_mother_age_l72_7250

variable (N M : ℕ)

theorem neha_mother_age (h1 : M - 12 = 4 * (N - 12)) (h2 : M + 12 = 2 * (N + 12)) : M = 60 := by
  sorry

end neha_mother_age_l72_7250


namespace Mrs_Amaro_roses_l72_7243

theorem Mrs_Amaro_roses :
  ∀ (total_roses red_roses yellow_roses pink_roses remaining_roses white_and_purple white_roses purple_roses : ℕ),
    total_roses = 500 →
    5 * total_roses % 8 = 0 →
    red_roses = total_roses * 5 / 8 →
    yellow_roses = (total_roses - red_roses) * 1 / 8 →
    pink_roses = (total_roses - red_roses) * 2 / 8 →
    remaining_roses = total_roses - red_roses - yellow_roses - pink_roses →
    remaining_roses % 2 = 0 →
    white_roses = remaining_roses / 2 →
    purple_roses = remaining_roses / 2 →
    red_roses + white_roses + purple_roses = 430 :=
by
  intros total_roses red_roses yellow_roses pink_roses remaining_roses white_and_purple white_roses purple_roses
  intro total_roses_eq
  intro red_roses_divisible
  intro red_roses_def
  intro yellow_roses_def
  intro pink_roses_def
  intro remaining_roses_def
  intro remaining_roses_even
  intro white_roses_def
  intro purple_roses_def
  sorry

end Mrs_Amaro_roses_l72_7243


namespace arithmetic_sqrt_of_sqrt_16_l72_7207

theorem arithmetic_sqrt_of_sqrt_16 : Real.sqrt (Real.sqrt 16) = 2 := 
by
  sorry

end arithmetic_sqrt_of_sqrt_16_l72_7207


namespace origami_papers_per_cousin_l72_7227

theorem origami_papers_per_cousin (total_papers : ℕ) (num_cousins : ℕ) (same_papers_each : ℕ) 
  (h1 : total_papers = 48) 
  (h2 : num_cousins = 6) 
  (h3 : same_papers_each = total_papers / num_cousins) : 
  same_papers_each = 8 := 
by 
  sorry

end origami_papers_per_cousin_l72_7227


namespace n_minus_k_minus_l_square_number_l72_7212

variable (n k l x : ℕ)

theorem n_minus_k_minus_l_square_number (h1 : x^2 < n)
                                        (h2 : n < (x + 1)^2)
                                        (h3 : n - k = x^2)
                                        (h4 : n + l = (x + 1)^2) :
  ∃ m : ℕ, n - k - l = m ^ 2 :=
by
  sorry

end n_minus_k_minus_l_square_number_l72_7212


namespace radio_selling_price_l72_7225

noncomputable def sellingPrice (costPrice : ℝ) (lossPercentage : ℝ) : ℝ :=
  costPrice - (lossPercentage / 100 * costPrice)

theorem radio_selling_price :
  sellingPrice 490 5 = 465.5 :=
by
  sorry

end radio_selling_price_l72_7225


namespace solve_swim_problem_l72_7249

/-- A man swims downstream 36 km and upstream some distance taking 3 hours each time. 
The speed of the man in still water is 9 km/h. -/
def swim_problem : Prop :=
  ∃ (v : ℝ) (d : ℝ),
    (9 + v) * 3 = 36 ∧ -- effective downstream speed and distance condition
    (9 - v) * 3 = d ∧ -- effective upstream speed and distance relation
    d = 18            -- required distance upstream is 18 km

theorem solve_swim_problem : swim_problem :=
  sorry

end solve_swim_problem_l72_7249


namespace quadratic_eq_of_sum_and_product_l72_7204

theorem quadratic_eq_of_sum_and_product (a b c : ℝ) (h_sum : -b / a = 4) (h_product : c / a = 3) :
    ∀ (x : ℝ), a * x^2 + b * x + c = a * x^2 - 4 * a * x + 3 * a :=
by
  sorry

end quadratic_eq_of_sum_and_product_l72_7204


namespace last_bead_color_is_blue_l72_7235

def bead_color_cycle := ["Red", "Orange", "Yellow", "Yellow", "Green", "Blue", "Purple"]

def bead_color (n : Nat) : String :=
  bead_color_cycle.get! (n % bead_color_cycle.length)

theorem last_bead_color_is_blue :
  bead_color 82 = "Blue" := 
by
  sorry

end last_bead_color_is_blue_l72_7235


namespace intersection_points_of_circle_and_vertical_line_l72_7254

theorem intersection_points_of_circle_and_vertical_line :
  (∃ y1 y2 : ℝ, y1 ≠ y2 ∧ (3, y1) ∈ { p : ℝ × ℝ | p.1^2 + p.2^2 = 16 } ∧ 
                    (3, y2) ∈ { p : ℝ × ℝ | p.1^2 + p.2^2 = 16 } ∧ 
                    (3, y1) ≠ (3, y2)) := 
by
  sorry

end intersection_points_of_circle_and_vertical_line_l72_7254


namespace range_of_sum_l72_7296

theorem range_of_sum (a b : ℝ) (h1 : -2 < a) (h2 : a < -1) (h3 : -1 < b) (h4 : b < 0) : 
  -3 < a + b ∧ a + b < -1 :=
by
  sorry

end range_of_sum_l72_7296


namespace not_possible_to_tile_l72_7201

theorem not_possible_to_tile 
    (m n : ℕ) (a b : ℕ)
    (h_m : m = 2018)
    (h_n : n = 2020)
    (h_a : a = 5)
    (h_b : b = 8) :
    ¬ ∃ k : ℕ, k * (a * b) = m * n := by
sorry

end not_possible_to_tile_l72_7201


namespace greatest_prime_divisor_digits_sum_l72_7245

theorem greatest_prime_divisor_digits_sum (h : 8191 = 2^13 - 1) : (1 + 2 + 7) = 10 :=
by
  sorry

end greatest_prime_divisor_digits_sum_l72_7245


namespace find_values_l72_7242

noncomputable def value_of_a (a : ℚ) : Prop :=
  4 + a = 2

noncomputable def value_of_b (b : ℚ) : Prop :=
  b^2 - 2 * b = 24 ∧ 4 * b^2 - 2 * b = 72

theorem find_values (a b : ℚ) (h1 : value_of_a a) (h2 : value_of_b b) :
  a = -2 ∧ b = -4 :=
by
  sorry

end find_values_l72_7242


namespace find_d_over_a_l72_7267

variable (a b c d : ℚ)

-- Conditions
def condition1 : Prop := a / b = 8
def condition2 : Prop := c / b = 4
def condition3 : Prop := c / d = 2 / 3

-- Theorem statement
theorem find_d_over_a (h1 : condition1 a b) (h2 : condition2 c b) (h3 : condition3 c d) : d / a = 3 / 4 :=
by
  -- Proof is omitted
  sorry

end find_d_over_a_l72_7267


namespace number_of_members_l72_7220

theorem number_of_members
  (headband_cost : ℕ := 3)
  (jersey_cost : ℕ := 10)
  (total_cost : ℕ := 2700)
  (cost_per_member : ℕ := 26) :
  total_cost / cost_per_member = 103 := by
  sorry

end number_of_members_l72_7220


namespace find_c_value_l72_7256

-- Given condition: x^2 + 300x + c = (x + a)^2
-- Problem statement: Prove that c = 22500 for the given conditions
theorem find_c_value (x a c : ℝ) : (x^2 + 300 * x + c = (x + 150)^2) → (c = 22500) :=
by
  intro h
  sorry

end find_c_value_l72_7256


namespace area_enclosed_by_trajectory_of_P_l72_7217

-- Definitions of points
structure Point where
  x : ℝ
  y : ℝ

-- Definition of fixed points A and B
def A : Point := { x := -3, y := 0 }
def B : Point := { x := 3, y := 0 }

-- Condition for the ratio of distances
def ratio_condition (P : Point) : Prop :=
  ((P.x + 3)^2 + P.y^2) / ((P.x - 3)^2 + P.y^2) = 1 / 4

-- Definition of a circle based on the derived condition in the solution
def circle_eq (P : Point) : Prop :=
  (P.x + 5)^2 + P.y^2 = 16

-- Theorem stating the area enclosed by the trajectory of point P is 16π
theorem area_enclosed_by_trajectory_of_P : 
  (∀ P : Point, ratio_condition P → circle_eq P) →
  ∃ A : ℝ, A = 16 * Real.pi :=
by
  sorry

end area_enclosed_by_trajectory_of_P_l72_7217


namespace possible_values_of_expression_l72_7268

theorem possible_values_of_expression (p q r s : ℝ) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (hs : s ≠ 0) :
  ∃ (vals : Finset ℤ), vals = {6, 2, 0, -2, -6} ∧
  (∃ val ∈ vals, val = (if p > 0 then 1 else -1) + 
                         (if q > 0 then 1 else -1) + 
                         (if r > 0 then 1 else -1) + 
                         (if s > 0 then 1 else -1) + 
                         (if (p * q * r) > 0 then 1 else -1) + 
                         (if (p * r * s) > 0 then 1 else -1)) :=
by
  sorry

end possible_values_of_expression_l72_7268


namespace no_real_solution_l72_7290

theorem no_real_solution (P : ℝ → ℝ) (h_cont : Continuous P) (h_no_fixed_point : ∀ x : ℝ, P x ≠ x) : ∀ x : ℝ, P (P x) ≠ x :=
by
  sorry

end no_real_solution_l72_7290


namespace parabola_complementary_slope_l72_7232

theorem parabola_complementary_slope
  (p x0 y0 x1 y1 x2 y2 : ℝ)
  (hp : p > 0)
  (hy0 : y0 > 0)
  (hP : y0^2 = 2 * p * x0)
  (hA : y1^2 = 2 * p * x1)
  (hB : y2^2 = 2 * p * x2)
  (h_slopes : (y1 - y0) / (x1 - x0) = - (2 * p / (y2 + y0))) :
  (y1 + y2) / y0 = -2 :=
by
  sorry

end parabola_complementary_slope_l72_7232


namespace collinear_points_count_l72_7230

-- Definitions for the problem conditions
def vertices_count := 8
def midpoints_count := 12
def face_centers_count := 6
def cube_center_count := 1
def total_points_count := vertices_count + midpoints_count + face_centers_count + cube_center_count

-- Lean statement to express the proof problem
theorem collinear_points_count :
  (total_points_count = 27) →
  (vertices_count = 8) →
  (midpoints_count = 12) →
  (face_centers_count = 6) →
  (cube_center_count = 1) →
  ∃ n, n = 49 :=
by
  intros
  existsi 49
  sorry

end collinear_points_count_l72_7230


namespace find_number_l72_7213

theorem find_number :
  ∃ n : ℕ, n * (1 / 7)^2 = 7^3 :=
by
  sorry

end find_number_l72_7213


namespace find_value_of_expression_l72_7291

variables (a b c : ℝ)

theorem find_value_of_expression
  (h1 : a ^ 4 * b ^ 3 * c ^ 5 = 18)
  (h2 : a ^ 3 * b ^ 5 * c ^ 4 = 8) :
  a ^ 5 * b * c ^ 6 = 81 / 2 :=
sorry

end find_value_of_expression_l72_7291


namespace count_distinct_digits_l72_7264

theorem count_distinct_digits (n : ℕ) (h1 : ∃ (n : ℕ), n^3 = 125) : 
  n = 5 :=
by
  sorry

end count_distinct_digits_l72_7264


namespace rectangle_to_square_y_l72_7282

theorem rectangle_to_square_y (y : ℝ) (a b : ℝ) (s : ℝ) (h1 : a = 7) (h2 : b = 21)
  (h3 : s^2 = a * b) (h4 : y = s / 2) : y = 7 * Real.sqrt 3 / 2 :=
by
  -- proof skipped
  sorry

end rectangle_to_square_y_l72_7282


namespace mingi_initial_tomatoes_l72_7233

theorem mingi_initial_tomatoes (n m r : ℕ) (h1 : n = 15) (h2 : m = 20) (h3 : r = 6) : n * m + r = 306 := by
  sorry

end mingi_initial_tomatoes_l72_7233


namespace total_rooms_in_hotel_l72_7297

def first_wing_floors : ℕ := 9
def first_wing_halls_per_floor : ℕ := 6
def first_wing_rooms_per_hall : ℕ := 32

def second_wing_floors : ℕ := 7
def second_wing_halls_per_floor : ℕ := 9
def second_wing_rooms_per_hall : ℕ := 40

def third_wing_floors : ℕ := 12
def third_wing_halls_per_floor : ℕ := 4
def third_wing_rooms_per_hall : ℕ := 50

def first_wing_total_rooms : ℕ := 
  first_wing_floors * first_wing_halls_per_floor * first_wing_rooms_per_hall

def second_wing_total_rooms : ℕ := 
  second_wing_floors * second_wing_halls_per_floor * second_wing_rooms_per_hall

def third_wing_total_rooms : ℕ := 
  third_wing_floors * third_wing_halls_per_floor * third_wing_rooms_per_hall

theorem total_rooms_in_hotel : 
  first_wing_total_rooms + second_wing_total_rooms + third_wing_total_rooms = 6648 := 
by 
  sorry

end total_rooms_in_hotel_l72_7297
