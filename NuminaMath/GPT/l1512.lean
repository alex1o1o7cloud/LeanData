import Mathlib

namespace find_n_from_binomial_terms_l1512_151263

theorem find_n_from_binomial_terms (x a : ℕ) (n : ℕ) 
  (h1 : n.choose 1 * x^(n-1) * a = 56) 
  (h2 : n.choose 2 * x^(n-2) * a^2 = 168) 
  (h3 : n.choose 3 * x^(n-3) * a^3 = 336) : 
  n = 5 :=
by
  sorry

end find_n_from_binomial_terms_l1512_151263


namespace smallest_possible_value_of_other_number_l1512_151230

theorem smallest_possible_value_of_other_number (x n : ℕ) (h_pos : x > 0) 
  (h_gcd : Nat.gcd 72 n = x + 6) (h_lcm : Nat.lcm 72 n = x * (x + 6)) : n = 12 := by
  sorry

end smallest_possible_value_of_other_number_l1512_151230


namespace exists_integer_n_l1512_151212

theorem exists_integer_n (k : ℕ) (hk : 0 < k) : 
  ∃ n : ℤ, (n + 1981^k)^(1/2 : ℝ) + (n : ℝ)^(1/2 : ℝ) = (1982^(1/2 : ℝ) + 1) ^ k :=
sorry

end exists_integer_n_l1512_151212


namespace children_count_l1512_151261

theorem children_count 
  (A B C : Finset ℕ)
  (hA : A.card = 7)
  (hB : B.card = 6)
  (hC : C.card = 5)
  (hA_inter_B : (A ∩ B).card = 4)
  (hA_inter_C : (A ∩ C).card = 3)
  (hB_inter_C : (B ∩ C).card = 2)
  (hA_inter_B_inter_C : (A ∩ B ∩ C).card = 1) :
  (A ∪ B ∪ C).card = 10 := 
by
  sorry

end children_count_l1512_151261


namespace sarahs_loan_amount_l1512_151205

theorem sarahs_loan_amount 
  (down_payment : ℕ := 10000)
  (monthly_payment : ℕ := 600)
  (repayment_years : ℕ := 5)
  (interest_rate : ℚ := 0) : down_payment + (monthly_payment * (12 * repayment_years)) = 46000 :=
by
  sorry

end sarahs_loan_amount_l1512_151205


namespace jerry_clock_reading_l1512_151217

noncomputable def clock_reading_after_pills (pills : ℕ) (start_time : ℕ) (interval : ℕ) : ℕ :=
(start_time + (pills - 1) * interval) % 12

theorem jerry_clock_reading :
  clock_reading_after_pills 150 12 5 = 1 :=
by
  sorry

end jerry_clock_reading_l1512_151217


namespace football_combinations_l1512_151233

theorem football_combinations : 
  ∃ (W D L : ℕ), W + D + L = 15 ∧ 3 * W + D = 33 ∧ 
  (9 ≤ W ∧ W ≤ 11) ∧
  (W = 9 → D = 6 ∧ L = 0) ∧
  (W = 10 → D = 3 ∧ L = 2) ∧
  (W = 11 → D = 0 ∧ L = 4) :=
sorry

end football_combinations_l1512_151233


namespace exists_min_a_l1512_151229

open Real

theorem exists_min_a (x y z : ℝ) : 
  (∃ x y z : ℝ, (sqrt (x - 1) + sqrt (y - 1) + sqrt (z - 1) = (11/2 - 1)) ∧ 
  (sqrt (x + 1) + sqrt (y + 1) + sqrt (z + 1) = (11/2 + 1))) :=
sorry

end exists_min_a_l1512_151229


namespace students_standing_count_l1512_151225

def students_seated : ℕ := 300
def teachers_seated : ℕ := 30
def total_attendees : ℕ := 355

theorem students_standing_count : total_attendees - (students_seated + teachers_seated) = 25 :=
by
  sorry

end students_standing_count_l1512_151225


namespace parabola_through_point_l1512_151250

theorem parabola_through_point (x y : ℝ) (hx : x = 2) (hy : y = 4) : 
  (∃ a : ℝ, y^2 = a * x ∧ a = 8) ∨ (∃ b : ℝ, x^2 = b * y ∧ b = 1) :=
sorry

end parabola_through_point_l1512_151250


namespace solve_for_x_y_l1512_151266

theorem solve_for_x_y (x y : ℝ) (h1 : x^2 + x * y + y = 14) (h2 : y^2 + x * y + x = 28) : 
  x + y = -7 ∨ x + y = 6 :=
by 
  -- We'll write sorry here to indicate the proof is to be completed
  sorry

end solve_for_x_y_l1512_151266


namespace students_taking_both_courses_l1512_151268

theorem students_taking_both_courses (n_total n_F n_G n_neither number_both : ℕ)
  (h_total : n_total = 79)
  (h_F : n_F = 41)
  (h_G : n_G = 22)
  (h_neither : n_neither = 25)
  (h_any_language : n_total - n_neither = 54)
  (h_sum_languages : n_F + n_G = 63)
  (h_both : n_F + n_G - (n_total - n_neither) = number_both) :
  number_both = 9 :=
by {
  sorry
}

end students_taking_both_courses_l1512_151268


namespace tank_fill_time_l1512_151223

theorem tank_fill_time (A_rate B_rate C_rate : ℝ) (hA : A_rate = 1/30) (hB : B_rate = 1/20) (hC : C_rate = -1/40) : 
  1 / (A_rate + B_rate + C_rate) = 120 / 7 :=
by
  -- proof goes here
  sorry

end tank_fill_time_l1512_151223


namespace graph_fixed_point_l1512_151249

theorem graph_fixed_point (f : ℝ → ℝ) (h : f 1 = 1) : f 1 = 1 :=
by
  sorry

end graph_fixed_point_l1512_151249


namespace ratio_of_shaded_area_l1512_151278

theorem ratio_of_shaded_area 
  (AC : ℝ) (CB : ℝ) 
  (AB : ℝ := AC + CB) 
  (radius_AC : ℝ := AC / 2) 
  (radius_CB : ℝ := CB / 2)
  (radius_AB : ℝ := AB / 2) 
  (shaded_area : ℝ := (radius_AB ^ 2 * Real.pi / 2) - (radius_AC ^ 2 * Real.pi / 2) - (radius_CB ^ 2 * Real.pi / 2))
  (CD : ℝ := Real.sqrt (AC^2 - radius_CB^2))
  (circle_area : ℝ := CD^2 * Real.pi) :
  (shaded_area / circle_area = 21 / 187) := 
by 
  sorry

end ratio_of_shaded_area_l1512_151278


namespace find_x_l1512_151272

noncomputable def vector_a : ℝ × ℝ := (1, 2)
noncomputable def vector_b : ℝ × ℝ := (1, 5)
noncomputable def vector_c (x : ℝ) : ℝ × ℝ := (x, 1)

def collinear (u v : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, u = (k * v.1, k * v.2)

theorem find_x :
  ∃ x : ℝ, collinear (2 • vector_a - vector_b) (vector_c x) ∧ x = -1 := by
  sorry

end find_x_l1512_151272


namespace jason_age_at_end_of_2004_l1512_151257

noncomputable def jason_age_in_1997 (y : ℚ) (g : ℚ) : Prop :=
  y = g / 3 

noncomputable def birth_years_sum (y : ℚ) (g : ℚ) : Prop :=
  (1997 - y) + (1997 - g) = 3852

theorem jason_age_at_end_of_2004
  (y g : ℚ)
  (h1 : jason_age_in_1997 y g)
  (h2 : birth_years_sum y g) :
  y + 7 = 42.5 :=
by
  sorry

end jason_age_at_end_of_2004_l1512_151257


namespace vampires_after_two_nights_l1512_151276

def initial_population : ℕ := 300
def initial_vampires : ℕ := 3
def conversion_rate : ℕ := 7

theorem vampires_after_two_nights :
  let first_night := initial_vampires * conversion_rate
  let total_first_night := initial_vampires + first_night
  let second_night := total_first_night * conversion_rate
  let total_second_night := total_first_night + second_night
  total_second_night = 192 :=
by
  let first_night := initial_vampires * conversion_rate
  let total_first_night := initial_vampires + first_night
  let second_night := total_first_night * conversion_rate
  let total_second_night := total_first_night + second_night
  have h1 : first_night = 21 := rfl
  have h2 : total_first_night = 24 := rfl
  have h3 : second_night = 168 := rfl
  have h4 : total_second_night = 192 := rfl
  exact rfl

end vampires_after_two_nights_l1512_151276


namespace parabola_maximum_value_l1512_151224

noncomputable def maximum_parabola (a b c : ℝ) (h := -b / (2*a)) (k := a * h^2 + b * h + c) : Prop :=
  ∀ (x : ℝ), a ≠ 0 → b = 12 → c = 4 → a = -3 → k = 16

theorem parabola_maximum_value : maximum_parabola (-3) 12 4 :=
by
  sorry

end parabola_maximum_value_l1512_151224


namespace find_number_l1512_151227

theorem find_number (x : ℚ) (h : 0.15 * 0.30 * 0.50 * x = 108) : x = 4800 :=
by
  sorry

end find_number_l1512_151227


namespace derivative_of_function_l1512_151284

theorem derivative_of_function
  (y : ℝ → ℝ)
  (h : ∀ x, y x = (1/2) * (Real.exp x + Real.exp (-x))) :
  ∀ x, deriv y x = (1/2) * (Real.exp x - Real.exp (-x)) :=
by
  sorry

end derivative_of_function_l1512_151284


namespace intersection_points_on_hyperbola_l1512_151270

theorem intersection_points_on_hyperbola (p x y : ℝ) :
  (2*p*x - 3*y - 4*p = 0) ∧ (4*x - 3*p*y - 6 = 0) → 
  (∃ a b : ℝ, (x^2) / (a^2) - (y^2) / (b^2) = 1) :=
by
  intros h
  sorry

end intersection_points_on_hyperbola_l1512_151270


namespace angle_measure_l1512_151285

theorem angle_measure (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 :=
by
  sorry

end angle_measure_l1512_151285


namespace find_p_l1512_151221

variables {m n p : ℚ}

theorem find_p (h1 : m = 3 * n + 5) (h2 : (m + 2) = 3 * (n + p) + 5) : p = 2 / 3 :=
by
  -- Proof steps go here
  sorry

end find_p_l1512_151221


namespace value_of_y_l1512_151258

theorem value_of_y (y : ℝ) (a : ℝ) (b : ℝ) (c : ℝ) (d : ℝ) :
  a = 10^3 → b = 10^4 → 
  a^y * 10^(3 * y) = (b^4) → 
  y = 8 / 3 :=
by 
  intro ha hb hc
  rw [ha, hb] at hc
  sorry

end value_of_y_l1512_151258


namespace sin_405_eq_sqrt2_div2_l1512_151242

theorem sin_405_eq_sqrt2_div2 : Real.sin (405 * Real.pi / 180) = Real.sqrt 2 / 2 := 
by
  sorry

end sin_405_eq_sqrt2_div2_l1512_151242


namespace volunteers_allocation_scheme_count_l1512_151244

theorem volunteers_allocation_scheme_count :
  let volunteers := 6
  let groups_of_two := 2
  let groups_of_one := 2
  let pavilions := 4
  let calculate_combinations (n k : ℕ) := Nat.choose n k
  calculate_combinations volunteers 2 * calculate_combinations (volunteers - 2) 2 * 
  calculate_combinations pavilions 2 * Nat.factorial pavilions = 1080 := by
sorry

end volunteers_allocation_scheme_count_l1512_151244


namespace integer_a_can_be_written_in_form_l1512_151297

theorem integer_a_can_be_written_in_form 
  (a x y : ℤ) 
  (h : 3 * a = x^2 + 2 * y^2) : 
  ∃ u v : ℤ, a = u^2 + 2 * v^2 :=
sorry

end integer_a_can_be_written_in_form_l1512_151297


namespace find_z_l1512_151289

open Complex

theorem find_z (z : ℂ) (h : (1 + 2 * z) / (1 - z) = Complex.I) : 
  z = -1 / 5 + 3 / 5 * Complex.I := 
sorry

end find_z_l1512_151289


namespace A_investment_l1512_151209

-- Conditions as definitions
def B_investment := 72000
def C_investment := 81000
def C_profit := 36000
def Total_profit := 80000

-- Statement to prove
theorem A_investment : 
  ∃ (x : ℕ), x = 27000 ∧
  (C_profit / Total_profit = (9 : ℕ) / 20) ∧
  (C_investment / (x + B_investment + C_investment) = (9 : ℕ) / 20) :=
by sorry

end A_investment_l1512_151209


namespace cars_15th_time_l1512_151237

noncomputable def minutes_since_8am (hour : ℕ) (minute : ℕ) : ℕ :=
  hour * 60 + minute

theorem cars_15th_time :
  let initial_time := minutes_since_8am 8 0
  let interval := 5
  let obstacles_time := 3 * 10
  let minutes_passed := (15 - 1) * interval + obstacles_time
  let total_time := initial_time + minutes_passed
  let expected_time := minutes_since_8am 9 40
  total_time = expected_time :=
by
  let initial_time := minutes_since_8am 8 0
  let interval := 5
  let obstacles_time := 3 * 10
  let minutes_passed := (15 - 1) * interval + obstacles_time
  let total_time := initial_time + minutes_passed
  let expected_time := minutes_since_8am 9 40
  show total_time = expected_time
  sorry

end cars_15th_time_l1512_151237


namespace inequality_2_pow_n_plus_2_gt_n_squared_l1512_151298

theorem inequality_2_pow_n_plus_2_gt_n_squared (n : ℕ) (hn : n > 0) : 2^n + 2 > n^2 := sorry

end inequality_2_pow_n_plus_2_gt_n_squared_l1512_151298


namespace train_ride_length_l1512_151271

theorem train_ride_length :
  let reading_time := 2
  let eating_time := 1
  let watching_time := 3
  let napping_time := 3
  reading_time + eating_time + watching_time + napping_time = 9 := 
by
  sorry

end train_ride_length_l1512_151271


namespace opposite_of_half_l1512_151207

theorem opposite_of_half : -(- (1/2)) = (1/2) := 
by 
  sorry

end opposite_of_half_l1512_151207


namespace max_value_at_x0_l1512_151239

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / (1 + x) - Real.log x

theorem max_value_at_x0 {x0 : ℝ} (h : ∃ x0, ∀ x, f x ≤ f x0) : 
  f x0 = x0 :=
sorry

end max_value_at_x0_l1512_151239


namespace range_of_2a_plus_3b_l1512_151279

theorem range_of_2a_plus_3b (a b : ℝ)
  (h1 : -1 ≤ a + b ∧ a + b ≤ 1)
  (h2 : -1 ≤ a - b ∧ a - b ≤ 1) :
  -3 ≤ 2 * a + 3 * b ∧ 2 * a + 3 * b ≤ 3 :=
by
  sorry

end range_of_2a_plus_3b_l1512_151279


namespace divisibility_by_5_l1512_151274

theorem divisibility_by_5 (x y : ℤ) (h : 5 ∣ (x + 9 * y)) : 5 ∣ (8 * x + 7 * y) :=
sorry

end divisibility_by_5_l1512_151274


namespace positive_rationals_in_S_l1512_151200

variable (S : Set ℚ)

-- Conditions
axiom closed_under_addition (a b : ℚ) (ha : a ∈ S) (hb : b ∈ S) : a + b ∈ S
axiom closed_under_multiplication (a b : ℚ) (ha : a ∈ S) (hb : b ∈ S) : a * b ∈ S
axiom zero_rule : ∀ r : ℚ, r ∈ S ∨ -r ∈ S ∨ r = 0

-- Prove that S is the set of positive rational numbers
theorem positive_rationals_in_S : S = {r : ℚ | 0 < r} :=
by
  sorry

end positive_rationals_in_S_l1512_151200


namespace total_books_per_year_l1512_151283

variable (c s : ℕ)

theorem total_books_per_year (hc : 0 < c) (hs : 0 < s) :
  6 * 12 * (c * s) = 72 * c * s := by
  sorry

end total_books_per_year_l1512_151283


namespace solve_for_x_l1512_151218

theorem solve_for_x (x : ℝ) (h : 144 / 0.144 = x / 0.0144) : x = 14.4 :=
by
  sorry

end solve_for_x_l1512_151218


namespace gcd_1260_924_l1512_151256

theorem gcd_1260_924 : Nat.gcd 1260 924 = 84 :=
by
  sorry

end gcd_1260_924_l1512_151256


namespace circle_passes_first_and_second_quadrants_l1512_151280

theorem circle_passes_first_and_second_quadrants :
  ∀ (x y : ℝ), (x - 1)^2 + (y - 3)^2 = 4 → ((x ≥ 0 ∧ y ≥ 0) ∨ (x ≤ 0 ∧ y ≥ 0)) :=
by
  sorry

end circle_passes_first_and_second_quadrants_l1512_151280


namespace lifting_to_bodyweight_ratio_l1512_151231

variable (t : ℕ) (w : ℕ) (p : ℕ) (delta_w : ℕ)

def lifting_total_after_increase (t : ℕ) (p : ℕ) : ℕ :=
  t + (t * p / 100)

def bodyweight_after_increase (w : ℕ) (delta_w : ℕ) : ℕ :=
  w + delta_w

theorem lifting_to_bodyweight_ratio (h_t : t = 2200) (h_w : w = 245) (h_p : p = 15) (h_delta_w : delta_w = 8) :
  lifting_total_after_increase t p / bodyweight_after_increase w delta_w = 10 :=
  by
    -- Use the given conditions
    rw [h_t, h_w, h_p, h_delta_w]
    -- Calculation steps are omitted, directly providing the final assertion
    sorry

end lifting_to_bodyweight_ratio_l1512_151231


namespace ab_range_l1512_151253

theorem ab_range (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + b + (1 / a) + (1 / b) = 5) :
  1 ≤ a + b ∧ a + b ≤ 4 :=
by
  sorry

end ab_range_l1512_151253


namespace percentage_increase_of_base_l1512_151241

theorem percentage_increase_of_base
  (h b : ℝ) -- Original height and base
  (h_new : ℝ) -- New height
  (b_new : ℝ) -- New base
  (A_original A_new : ℝ) -- Original and new areas
  (p : ℝ) -- Percentage increase in the base
  (h_new_def : h_new = 0.60 * h)
  (b_new_def : b_new = b * (1 + p / 100))
  (A_original_def : A_original = 0.5 * b * h)
  (A_new_def : A_new = 0.5 * b_new * h_new)
  (area_decrease : A_new = 0.84 * A_original) :
  p = 40 := by
  sorry

end percentage_increase_of_base_l1512_151241


namespace fraction_of_remaining_paint_used_l1512_151213

theorem fraction_of_remaining_paint_used (total_paint : ℕ) (first_week_fraction : ℚ) (total_used : ℕ) :
  total_paint = 360 ∧ first_week_fraction = 1/6 ∧ total_used = 120 →
  (total_used - first_week_fraction * total_paint) / (total_paint - first_week_fraction * total_paint) = 1/5 :=
  by
    sorry

end fraction_of_remaining_paint_used_l1512_151213


namespace no_such_convex_polyhedron_exists_l1512_151219

-- Definitions of convex polyhedron and the properties related to its faces and vertices.
structure ConvexPolyhedron where
  V : ℕ  -- Number of vertices
  E : ℕ  -- Number of edges
  F : ℕ  -- Number of faces
  -- Additional properties and constraints can be added if necessary

-- Definition that captures the condition where each face has more than 5 sides.
def each_face_has_more_than_five_sides (P : ConvexPolyhedron) : Prop :=
  ∀ f, f > 5 -- Simplified assumption

-- Definition that captures the condition where more than five edges meet at each vertex.
def more_than_five_edges_meet_each_vertex (P : ConvexPolyhedron) : Prop :=
  ∀ v, v > 5 -- Simplified assumption

-- The statement to be proven
theorem no_such_convex_polyhedron_exists :
  ¬ ∃ (P : ConvexPolyhedron), (each_face_has_more_than_five_sides P) ∨ (more_than_five_edges_meet_each_vertex P) := by
  -- Proof of this theorem is omitted with "sorry"
  sorry

end no_such_convex_polyhedron_exists_l1512_151219


namespace smooth_transition_l1512_151245

theorem smooth_transition (R : ℝ) (x₀ y₀ : ℝ) :
  ∃ m : ℝ, ∀ (x y : ℝ), (x - x₀)^2 + (y - y₀)^2 = R^2 → y - y₀ = m * (x - x₀) :=
sorry

end smooth_transition_l1512_151245


namespace sum_of_money_invested_l1512_151243

noncomputable def principal_sum_of_money (R : ℝ) (T : ℝ) (CI_minus_SI : ℝ) : ℝ :=
  let SI := (625 * R * T / 100)
  let CI := 625 * ((1 + R / 100)^(T : ℝ) - 1)
  if (CI - SI = CI_minus_SI)
  then 625
  else 0

theorem sum_of_money_invested : 
  (principal_sum_of_money 4 2 1) = 625 :=
by
  unfold principal_sum_of_money
  sorry

end sum_of_money_invested_l1512_151243


namespace world_expo_visitors_l1512_151214

noncomputable def per_person_cost (x : ℕ) : ℕ :=
  if x <= 30 then 120 else max (120 - 2 * (x - 30)) 90

theorem world_expo_visitors (x : ℕ) (h_cost : x * per_person_cost x = 4000) : x = 40 :=
by
  sorry

end world_expo_visitors_l1512_151214


namespace part1_part2_l1512_151204

noncomputable def f (a x : ℝ) : ℝ := (a * 2^x + a - 2) / (2^x + 1)

-- Statement 1: If f(x) is an odd function, then a = 1.
theorem part1 (a : ℝ) : (∀ x : ℝ, f a (-x) = - f a x) → a = 1 :=
sorry

-- Statement 2: If f(x) is defined on [-4, +∞), and for all x in the domain, 
-- f(cos(x) + b + 1/4) ≥ f(sin^2(x) - b - 3), then b ∈ [-1,1].
theorem part2 (a : ℝ) (b : ℝ) :
  (∀ x : ℝ, f a (Real.cos x + b + 1/4) ≥ f a (Real.sin x ^ 2 - b - 3)) ∧
  (∀ x : ℝ, -4 ≤ x) ∧ -4 ≤ a ∧ a = 1 → -1 ≤ b ∧ b ≤ 1 :=
sorry

end part1_part2_l1512_151204


namespace inequality_proof_l1512_151295

theorem inequality_proof (a b : ℝ) (h₀ : b > a) (h₁ : ab > 0) : 
  (1 / a > 1 / b) ∧ (a + b < 2 * b) :=
by
  sorry

end inequality_proof_l1512_151295


namespace greatest_four_digit_p_l1512_151267

-- Define conditions
def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000
def reverse_digits (n : ℕ) : ℕ := 
  let d1 := n / 1000 % 10
  let d2 := n / 100 % 10
  let d3 := n / 10 % 10
  let d4 := n % 10
  d4 * 1000 + d3 * 100 + d2 * 10 + d1
def is_divisible_by (a b : ℕ) : Prop := b ∣ a

-- Proof problem
theorem greatest_four_digit_p (p : ℕ) (q : ℕ) 
    (hp1 : is_four_digit p)
    (hp2 : q = reverse_digits p)
    (hp3 : is_four_digit q)
    (hp4 : is_divisible_by p 63)
    (hp5 : is_divisible_by q 63)
    (hp6 : is_divisible_by p 19) :
  p = 5985 :=
sorry

end greatest_four_digit_p_l1512_151267


namespace albrecht_correct_substitution_l1512_151211

theorem albrecht_correct_substitution (a b : ℕ) (h : (a + 2 * b - 3)^2 = a^2 + 4 * b^2 - 9) :
  (a = 2 ∧ b = 15) ∨ (a = 3 ∧ b = 6) ∨ (a = 6 ∧ b = 3) ∨ (a = 15 ∧ b = 2) :=
by
  -- The proof will be filled in here
  sorry

end albrecht_correct_substitution_l1512_151211


namespace original_price_of_books_l1512_151246

theorem original_price_of_books (purchase_cost : ℝ) (original_price : ℝ) :
  (purchase_cost = 162) →
  (original_price ≤ 100) ∨ 
  (100 < original_price ∧ original_price ≤ 200 ∧ purchase_cost = original_price * 0.9) ∨ 
  (original_price > 200 ∧ purchase_cost = original_price * 0.8) →
  (original_price = 180 ∨ original_price = 202.5) :=
by
  sorry

end original_price_of_books_l1512_151246


namespace problem_min_value_problem_inequality_range_l1512_151226

theorem problem_min_value (a b : ℝ) (h : a + b = 1) (ha : a > 0) (hb : b > 0) :
  (1 / a + 4 / b) ≥ 9 :=
sorry

theorem problem_inequality_range (a b : ℝ) (h : a + b = 1) (ha : a > 0) (hb : b > 0) (x : ℝ) :
  (1 / a + 4 / b) ≥ |2 * x - 1| - |x + 1| ↔ -7 ≤ x ∧ x ≤ 11 :=
sorry

end problem_min_value_problem_inequality_range_l1512_151226


namespace abs_diff_of_two_numbers_l1512_151296

variable {x y : ℝ}

theorem abs_diff_of_two_numbers (h1 : x + y = 40) (h2 : x * y = 396) : abs (x - y) = 4 := by
  sorry

end abs_diff_of_two_numbers_l1512_151296


namespace arc_length_calculation_l1512_151210

theorem arc_length_calculation (C θ : ℝ) (hC : C = 72) (hθ : θ = 45) :
  (θ / 360) * C = 9 :=
by
  sorry

end arc_length_calculation_l1512_151210


namespace quadratic_equation_real_roots_l1512_151234

theorem quadratic_equation_real_roots (k : ℝ) : 
  (∃ x : ℝ, k * x^2 - 6 * x + 9 = 0) ↔ (k ≤ 1 ∧ k ≠ 0) :=
by
  sorry

end quadratic_equation_real_roots_l1512_151234


namespace ab_cd_eq_zero_l1512_151201

theorem ab_cd_eq_zero  
  (a b c d : ℝ) 
  (h1 : a^2 + b^2 = 1)
  (h2 : c^2 + d^2 = 1)
  (h3 : ad - bc = -1) :
  ab + cd = 0 :=
by
  sorry

end ab_cd_eq_zero_l1512_151201


namespace customers_left_l1512_151299

-- Given conditions:
def initial_customers : ℕ := 21
def remaining_customers : ℕ := 12

-- Prove that the number of customers who left is 9
theorem customers_left : initial_customers - remaining_customers = 9 := by
  sorry

end customers_left_l1512_151299


namespace alice_savings_third_month_l1512_151291

theorem alice_savings_third_month :
  ∀ (saved_first : ℕ) (increase_per_month : ℕ),
  saved_first = 10 →
  increase_per_month = 30 →
  let saved_second := saved_first + increase_per_month
  let saved_third := saved_second + increase_per_month
  saved_third = 70 :=
by intros saved_first increase_per_month h1 h2;
   let saved_second := saved_first + increase_per_month;
   let saved_third := saved_second + increase_per_month;
   sorry

end alice_savings_third_month_l1512_151291


namespace percent_nonunion_women_l1512_151292

variable (E : ℝ) -- Total number of employees

-- Definitions derived from the problem conditions
def menPercent : ℝ := 0.46
def unionPercent : ℝ := 0.60
def nonUnionPercent : ℝ := 1 - unionPercent
def nonUnionWomenPercent : ℝ := 0.90

theorem percent_nonunion_women :
  nonUnionWomenPercent = 0.90 :=
by
  sorry

end percent_nonunion_women_l1512_151292


namespace work_together_days_l1512_151282

theorem work_together_days (A_rate B_rate x total_work B_days_worked : ℚ)
  (hA : A_rate = 1/4)
  (hB : B_rate = 1/8)
  (hCombined : (A_rate + B_rate) * x + B_rate * B_days_worked = total_work)
  (hTotalWork : total_work = 1)
  (hBDays : B_days_worked = 2) : x = 2 :=
by
  sorry

end work_together_days_l1512_151282


namespace intersecting_lines_a_plus_b_l1512_151220

theorem intersecting_lines_a_plus_b :
  ∃ (a b : ℝ), (∀ x y : ℝ, (x = 1 / 3 * y + a) ∧ (y = 1 / 3 * x + b) → (x = 3 ∧ y = 4)) ∧ a + b = 14 / 3 :=
sorry

end intersecting_lines_a_plus_b_l1512_151220


namespace complex_imaginary_part_l1512_151232

theorem complex_imaginary_part : 
  Complex.im ((1 : ℂ) / (-2 + Complex.I) + (1 : ℂ) / (1 - 2 * Complex.I)) = 1/5 := 
  sorry

end complex_imaginary_part_l1512_151232


namespace prove_f_2_eq_3_l1512_151248

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
if x ≤ 1 then 3 * a ^ x else Real.log (2 * x + 4) / Real.log a

theorem prove_f_2_eq_3 (a : ℝ) (h1 : f 1 a = 6) : f 2 a = 3 :=
by
  -- Define the conditions
  have h1 : 3 * a = 6 := by simp [f] at h1; assumption
  -- Two subcases: x <= 1 and x > 1
  have : a = 2 := by linarith
  simp [f, this]
  sorry

end prove_f_2_eq_3_l1512_151248


namespace quadrilateral_is_trapezoid_or_parallelogram_l1512_151222

noncomputable def quadrilateral_property (s1 s2 s3 s4 : ℝ) : Prop :=
  (s1 + s2) * (s3 + s4) = (s1 + s4) * (s2 + s3)

theorem quadrilateral_is_trapezoid_or_parallelogram
  (s1 s2 s3 s4 : ℝ) (h : quadrilateral_property s1 s2 s3 s4) :
  (s1 = s3) ∨ (s2 = s4) ∨ -- Trapezoid conditions
  ∃ (p : ℝ), (p * s1 = s3 * (s1 + s4)) := -- Add necessary conditions to represent a parallelogram
sorry

end quadrilateral_is_trapezoid_or_parallelogram_l1512_151222


namespace quadratic_point_value_l1512_151247

theorem quadratic_point_value 
  (a b c : ℝ) 
  (h_min : ∀ x : ℝ, a * x^2 + b * x + c ≥ a * (-1)^2 + b * (-1) + c) 
  (h_at_min : a * (-1)^2 + b * (-1) + c = -3)
  (h_point : a * (1)^2 + b * (1) + c = 7) : 
  a * (3)^2 + b * (3) + c = 37 :=
sorry

end quadratic_point_value_l1512_151247


namespace probability_no_obtuse_triangle_correct_l1512_151294

noncomputable def probability_no_obtuse_triangle (circle_points : List Point) (center : Point) : ℚ :=
  if circle_points.length = 4 then 3 / 64 else 0

theorem probability_no_obtuse_triangle_correct (circle_points : List Point) (center : Point) 
  (h : circle_points.length = 4) :
  probability_no_obtuse_triangle circle_points center = 3 / 64 :=
by
  sorry

end probability_no_obtuse_triangle_correct_l1512_151294


namespace cookies_per_student_l1512_151293

theorem cookies_per_student (students : ℕ) (percent : ℝ) (oatmeal_cookies : ℕ) 
                            (h_students : students = 40)
                            (h_percent : percent = 10 / 100)
                            (h_oatmeal : oatmeal_cookies = 8) :
                            (oatmeal_cookies / percent / students) = 2 := by
  sorry

end cookies_per_student_l1512_151293


namespace probability_A_not_losing_is_80_percent_l1512_151273

def probability_A_winning : ℝ := 0.30
def probability_draw : ℝ := 0.50
def probability_A_not_losing : ℝ := probability_A_winning + probability_draw

theorem probability_A_not_losing_is_80_percent : probability_A_not_losing = 0.80 :=
by 
  sorry

end probability_A_not_losing_is_80_percent_l1512_151273


namespace miles_left_l1512_151228

theorem miles_left (d_total d_covered d_left : ℕ) 
  (h₁ : d_total = 78) 
  (h₂ : d_covered = 32) 
  (h₃ : d_left = d_total - d_covered):
  d_left = 46 := 
by {
  sorry 
}

end miles_left_l1512_151228


namespace min_possible_value_of_box_l1512_151251

theorem min_possible_value_of_box
  (c d : ℤ)
  (distinct : c ≠ d)
  (h_cd : c * d = 29) :
  ∃ (box : ℤ), c^2 + d^2 = box ∧ box = 842 :=
by
  sorry

end min_possible_value_of_box_l1512_151251


namespace estate_value_l1512_151235

theorem estate_value (E : ℝ) (x : ℝ) (hx : 5 * x = 0.6 * E) (charity_share : ℝ)
  (hcharity : charity_share = 800) (hwife : 3 * x * 4 = 12 * x)
  (htotal : E = 17 * x + charity_share) : E = 1923 :=
by
  sorry

end estate_value_l1512_151235


namespace sqrt_expression_simplification_l1512_151290

theorem sqrt_expression_simplification :
  (Real.sqrt 72 / Real.sqrt 3 - Real.sqrt (1 / 2) * Real.sqrt 12 - |2 - Real.sqrt 6|) = 2 :=
by
  sorry

end sqrt_expression_simplification_l1512_151290


namespace value_of_abs_h_l1512_151275

theorem value_of_abs_h (h : ℝ) : 
  (∃ r s : ℝ, (r + s = -4 * h) ∧ (r * s = -5) ∧ (r^2 + s^2 = 13)) → 
  |h| = (Real.sqrt 3) / 4 :=
by
  sorry

end value_of_abs_h_l1512_151275


namespace solve_system_of_equations_l1512_151208

theorem solve_system_of_equations :
  ∃ (x y: ℝ), (x - y - 1 = 0) ∧ (4 * (x - y) - y = 0) ∧ (x = 5) ∧ (y = 4) :=
by
  sorry

end solve_system_of_equations_l1512_151208


namespace find_erased_number_l1512_151254

/-- Define the variables used in the conditions -/
def n : ℕ := 69
def erased_number_mean : ℚ := 35 + 7 / 17
def sequence_sum : ℕ := n * (n + 1) / 2

/-- State the condition for the erased number -/
noncomputable def erased_number (x : ℕ) : Prop :=
  (sequence_sum - x) / (n - 1) = erased_number_mean

/-- The main theorem stating that the erased number is 7 -/
theorem find_erased_number : ∃ x : ℕ, erased_number x ∧ x = 7 :=
by
  use 7
  unfold erased_number sequence_sum
  -- Sum of first 69 natural numbers is 69 * (69 + 1) / 2
  -- Hence,
  -- (69 * 70 / 2 - 7) / 68 = 35 + 7 / 17
  -- which simplifies to true under these conditions
  -- Detailed proof skipped here as per instructions
  sorry

end find_erased_number_l1512_151254


namespace power_multiplication_result_l1512_151203

theorem power_multiplication_result :
  ( (8 / 9)^3 * (1 / 3)^3 * (2 / 5)^3 = (4096 / 2460375) ) :=
by
  sorry

end power_multiplication_result_l1512_151203


namespace cube_root_59319_cube_root_103823_l1512_151264

theorem cube_root_59319 : ∃ x : ℕ, x ^ 3 = 59319 ∧ x = 39 :=
by
  -- Sorry used to skip the proof, which is not required 
  sorry

theorem cube_root_103823 : ∃ x : ℕ, x ^ 3 = 103823 ∧ x = 47 :=
by
  -- Sorry used to skip the proof, which is not required 
  sorry

end cube_root_59319_cube_root_103823_l1512_151264


namespace cousins_initial_money_l1512_151236

theorem cousins_initial_money (x : ℕ) :
  let Carmela_initial := 7
  let num_cousins := 4
  let gift_each := 1
  Carmela_initial - num_cousins * gift_each = x + gift_each →
  x = 2 :=
by
  intro h
  sorry

end cousins_initial_money_l1512_151236


namespace max_product_two_integers_sum_300_l1512_151216

theorem max_product_two_integers_sum_300 : 
  ∃ x : ℤ, 0 ≤ x ∧ x ≤ 300 ∧ (x * (300 - x) = 22500) := 
by
  sorry

end max_product_two_integers_sum_300_l1512_151216


namespace find_naturals_divisibility_l1512_151252

theorem find_naturals_divisibility :
  {n : ℕ | (2^n + n) ∣ (8^n + n)} = {1, 2, 4, 6} :=
by sorry

end find_naturals_divisibility_l1512_151252


namespace pages_in_book_l1512_151286

theorem pages_in_book
  (x : ℝ)
  (h1 : x - (x / 6 + 10) = (5 * x) / 6 - 10)
  (h2 : (5 * x) / 6 - 10 - ((1 / 5) * ((5 * x) / 6 - 10) + 20) = (2 * x) / 3 - 28)
  (h3 : (2 * x) / 3 - 28 - ((1 / 4) * ((2 * x) / 3 - 28) + 25) = x / 2 - 46)
  (h4 : x / 2 - 46 = 72) :
  x = 236 := 
sorry

end pages_in_book_l1512_151286


namespace find_minimum_x_and_values_l1512_151277

theorem find_minimum_x_and_values (x y z w : ℝ) (h1 : y = x - 2003)
  (h2 : z = 2 * y - 2003)
  (h3 : w = 3 * z - 2003)
  (h4 : 0 ≤ x)
  (h5 : 0 ≤ y)
  (h6 : 0 ≤ z)
  (h7 : 0 ≤ w) :
  x ≥ 10015 / 3 ∧ 
  (x = 10015 / 3 → y = 4006 / 3 ∧ z = 2003 / 3 ∧ w = 0) := by
  sorry

end find_minimum_x_and_values_l1512_151277


namespace problem_correct_calculation_l1512_151287

theorem problem_correct_calculation (a b : ℕ) : 
  (4 * a - 2 * a ≠ 2) ∧ 
  (a^8 / a^4 ≠ a^2) ∧ 
  (a^2 * a^3 = a^5) ∧ 
  ((b^2)^3 ≠ b^5) :=
by {
  sorry
}

end problem_correct_calculation_l1512_151287


namespace vessel_reaches_boat_in_shortest_time_l1512_151215

-- Define the given conditions as hypotheses
variable (dist_AC : ℝ) (angle_C : ℝ) (speed_CB : ℝ) (angle_B : ℝ) (speed_A : ℝ)

-- Assign values to variables based on the problem statement
def vessel_distress_boat_condition : Prop :=
  dist_AC = 10 ∧ angle_C = 45 ∧ speed_CB = 9 ∧ angle_B = 105 ∧ speed_A = 21

-- Define the time (in minutes) for the vessel to reach the fishing boat
noncomputable def shortest_time_to_reach_boat : ℝ :=
  25

-- The theorem that we need to prove given the conditions
theorem vessel_reaches_boat_in_shortest_time :
  vessel_distress_boat_condition dist_AC angle_C speed_CB angle_B speed_A → 
  shortest_time_to_reach_boat = 25 := by
    intros
    sorry

end vessel_reaches_boat_in_shortest_time_l1512_151215


namespace parallel_lines_a_eq_neg2_l1512_151288

theorem parallel_lines_a_eq_neg2 (a : ℝ) :
  (∀ x y : ℝ, (ax + y - 1 - a = 0) ↔ (x - (1/2) * y = 0)) → a = -2 :=
by sorry

end parallel_lines_a_eq_neg2_l1512_151288


namespace lambda_property_l1512_151238
open Int

noncomputable def lambda : ℝ := 1 + Real.sqrt 2

theorem lambda_property (n : ℕ) (hn : n > 0) :
  2 * ⌊lambda * n⌋ = 1 - n + ⌊lambda * ⌊lambda * n⌋⌋ :=
sorry

end lambda_property_l1512_151238


namespace find_t_value_l1512_151240

theorem find_t_value (x y z t : ℕ) (hx : x = 1) (hy : y = 2) (hz : z = 3) (hpos_x : 0 < x) (hpos_y : 0 < y) (hpos_z : 0 < z) :
  x + y + z + t = 10 → t = 4 :=
by
  -- Proof goes here
  sorry

end find_t_value_l1512_151240


namespace sector_area_is_8pi_over_3_l1512_151260

noncomputable def sector_area {r θ1 θ2 : ℝ} 
  (hθ1 : θ1 = π / 3)
  (hθ2 : θ2 = 2 * π / 3)
  (hr : r = 4) : ℝ := 
    1 / 2 * (θ2 - θ1) * r ^ 2

theorem sector_area_is_8pi_over_3 (θ1 θ2 : ℝ) 
  (hθ1 : θ1 = π / 3)
  (hθ2 : θ2 = 2 * π / 3)
  (r : ℝ) (hr : r = 4) : 
  sector_area hθ1 hθ2 hr = 8 * π / 3 :=
by
  sorry

end sector_area_is_8pi_over_3_l1512_151260


namespace blender_customers_l1512_151255

variable (p_t p_b : ℕ) (c_t c_b : ℕ) (k : ℕ)

-- Define the conditions
def condition_toaster_popularity : p_t = 20 := sorry
def condition_toaster_cost : c_t = 300 := sorry
def condition_blender_cost : c_b = 450 := sorry
def condition_inverse_proportionality : p_t * c_t = k := sorry

-- Proof goal: number of customers who would buy the blender
theorem blender_customers : p_b = 13 :=
by
  have h1 : p_t * c_t = 6000 := by sorry -- Using the given conditions
  have h2 : p_b * c_b = 6000 := by sorry -- Assumption for the same constant k
  have h3 : c_b = 450 := sorry
  have h4 : p_b = 6000 / 450 := by sorry
  have h5 : p_b = 13 := by sorry
  exact h5

end blender_customers_l1512_151255


namespace scallops_cost_calculation_l1512_151262

def scallops_per_pound : ℕ := 8
def cost_per_pound : ℝ := 24.00
def scallops_per_person : ℕ := 2
def number_of_people : ℕ := 8

def total_cost : ℝ := 
  let total_scallops := number_of_people * scallops_per_person
  let total_pounds := total_scallops / scallops_per_pound
  total_pounds * cost_per_pound

theorem scallops_cost_calculation :
  total_cost = 48.00 :=
by sorry

end scallops_cost_calculation_l1512_151262


namespace intervals_of_monotonicity_range_of_values_l1512_151269

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  x - a * Real.log x

noncomputable def g (a : ℝ) (x : ℝ) : ℝ :=
  -(1 + a) / x

noncomputable def h (a : ℝ) (x : ℝ) : ℝ :=
  f a x - g a x

theorem intervals_of_monotonicity (a : ℝ) (h_pos : 0 < a) :
  (∀ x > 0, x < 1 + a → h a x < h a (1 + a)) ∧
  (∀ x > 1 + a, h a x > h a (1 + a)) :=
sorry

theorem range_of_values (x0 : ℝ) (h_x0 : 1 ≤ x0 ∧ x0 ≤ Real.exp 1) (h_fx_gx : f a x0 < g a x0) :
  a > (Real.exp 1)^2 + 1 / (Real.exp 1 - 1) ∨ a < -2 :=
sorry

end intervals_of_monotonicity_range_of_values_l1512_151269


namespace range_g_l1512_151206

noncomputable def g (x : Real) : Real := (Real.sin x)^6 + (Real.cos x)^4

theorem range_g :
  ∃ (a : Real), 
    (∀ x : Real, g x ≥ a ∧ g x ≤ 1) ∧
    (∀ y : Real, y < a → ¬∃ x : Real, g x = y) :=
sorry

end range_g_l1512_151206


namespace cos_pi_minus_half_alpha_l1512_151265

-- Conditions given in the problem
variable (α : ℝ)
variable (hα1 : 0 < α ∧ α < π / 2)
variable (hα2 : Real.sin α = 3 / 5)

-- The proof problem statement
theorem cos_pi_minus_half_alpha (hα1 : 0 < α ∧ α < π / 2) (hα2 : Real.sin α = 3 / 5) : 
  Real.cos (π - α / 2) = -3 * Real.sqrt 10 / 10 := 
sorry

end cos_pi_minus_half_alpha_l1512_151265


namespace non_integer_sum_exists_l1512_151281

theorem non_integer_sum_exists (k l : ℕ) (hk : 0 < k) (hl : 0 < l) :
  ∃ M : ℕ, ∀ n : ℕ, n > M → ¬ ∃ t : ℤ, (k + 1/2)^n + (l + 1/2)^n = t := 
sorry

end non_integer_sum_exists_l1512_151281


namespace part_I_solution_set_part_II_min_value_l1512_151259

-- Define the function f
def f (x a : ℝ) := 2*|x + 1| - |x - a|

-- Part I: Prove the solution set of f(x) ≥ 0 when a = 2
theorem part_I_solution_set (x : ℝ) :
  f x 2 ≥ 0 ↔ x ≤ -4 ∨ x ≥ 0 :=
sorry

-- Define the function g
def g (x a : ℝ) := f x a + 3*|x - a|

-- Part II: Prove the minimum value of m + n given t = 4 when a = 1
theorem part_II_min_value (m n : ℝ) (hm : m > 0) (hn : n > 0) :
  (∀ x, g x 1 ≥ 4) → (2/m + 1/(2*n) = 4) → m + n = 9/8 :=
sorry

end part_I_solution_set_part_II_min_value_l1512_151259


namespace find_y_l1512_151202

open Real

theorem find_y : ∃ y : ℝ, (sqrt ((3 - (-5))^2 + (y - 4)^2) = 12) ∧ (y > 0) ∧ (y = 4 + 4 * sqrt 5) :=
by
  use 4 + 4 * sqrt 5
  -- The proof steps would go here.
  sorry

end find_y_l1512_151202
