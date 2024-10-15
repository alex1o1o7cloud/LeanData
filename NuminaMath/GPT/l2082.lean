import Mathlib

namespace NUMINAMATH_GPT_intersection_points_calculation_l2082_208211

-- Define the quadratic function and related functions
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c
def u (a b c x : ℝ) : ℝ := - f a b c (-x)
def v (a b c x : ℝ) : ℝ := f a b c (x + 1)

-- Define the number of intersection points
def m : ℝ := 1
def n : ℝ := 0

-- The proof goal
theorem intersection_points_calculation (a b c : ℝ) : 7 * m + 3 * n = 7 :=
by sorry

end NUMINAMATH_GPT_intersection_points_calculation_l2082_208211


namespace NUMINAMATH_GPT_digit_A_in_comb_60_15_correct_l2082_208256

-- Define the combination function
def comb (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- The main theorem we want to prove
theorem digit_A_in_comb_60_15_correct : 
  ∃ (A : ℕ), (660 * 10^9 + A * 10^8 + B * 10^7 + 5 * 10^6 + A * 10^4 + 640 * 10^1 + A) = comb 60 15 ∧ A = 6 :=
by
  sorry

end NUMINAMATH_GPT_digit_A_in_comb_60_15_correct_l2082_208256


namespace NUMINAMATH_GPT_minimum_value_expr_l2082_208230

theorem minimum_value_expr (x : ℝ) (h : x > 2) :
  ∃ y, y = (x^2 - 6 * x + 8) / (2 * x - 4) ∧ y = -1/2 := sorry

end NUMINAMATH_GPT_minimum_value_expr_l2082_208230


namespace NUMINAMATH_GPT_right_triangle_area_l2082_208224

theorem right_triangle_area (a b c : ℝ) (h₁ : a = 24) (h₂ : c = 26) (h₃ : a^2 + b^2 = c^2) : 
  (1 / 2) * a * b = 120 :=
by
  sorry

end NUMINAMATH_GPT_right_triangle_area_l2082_208224


namespace NUMINAMATH_GPT_sum_m_n_zero_l2082_208294

theorem sum_m_n_zero
  (m n p : ℝ)
  (h1 : mn + p^2 + 4 = 0)
  (h2 : m - n = 4) :
  m + n = 0 :=
sorry

end NUMINAMATH_GPT_sum_m_n_zero_l2082_208294


namespace NUMINAMATH_GPT_batsman_average_after_12_innings_l2082_208229

theorem batsman_average_after_12_innings
  (score_12th: ℕ) (increase_avg: ℕ) (initial_innings: ℕ) (final_innings: ℕ) 
  (initial_avg: ℕ) (final_avg: ℕ) :
  score_12th = 48 ∧ increase_avg = 2 ∧ initial_innings = 11 ∧ final_innings = 12 ∧
  final_avg = initial_avg + increase_avg ∧
  12 * final_avg = initial_innings * initial_avg + score_12th →
  final_avg = 26 :=
by 
  sorry

end NUMINAMATH_GPT_batsman_average_after_12_innings_l2082_208229


namespace NUMINAMATH_GPT_ln_abs_x_minus_a_even_iff_a_zero_l2082_208250

theorem ln_abs_x_minus_a_even_iff_a_zero (a : ℝ) : 
  (∀ x : ℝ, Real.log (|x - a|) = Real.log (|(-x) - a|)) ↔ a = 0 :=
sorry

end NUMINAMATH_GPT_ln_abs_x_minus_a_even_iff_a_zero_l2082_208250


namespace NUMINAMATH_GPT_percent_freshmen_psychology_majors_l2082_208249

-- Define the total number of students in our context
def total_students : ℕ := 100

-- Define what 80% of total students being freshmen means
def freshmen (total : ℕ) : ℕ := 8 * total / 10

-- Define what 60% of freshmen being in the school of liberal arts means
def freshmen_in_liberal_arts (total : ℕ) : ℕ := 6 * freshmen total / 10

-- Define what 50% of freshmen in the school of liberal arts being psychology majors means
def freshmen_psychology_majors (total : ℕ) : ℕ := 5 * freshmen_in_liberal_arts total / 10

theorem percent_freshmen_psychology_majors :
  (freshmen_psychology_majors total_students : ℝ) / total_students * 100 = 24 :=
by
  sorry

end NUMINAMATH_GPT_percent_freshmen_psychology_majors_l2082_208249


namespace NUMINAMATH_GPT_remainder_8_times_10_pow_18_plus_1_pow_18_div_9_l2082_208281

theorem remainder_8_times_10_pow_18_plus_1_pow_18_div_9 :
  (8 * 10^18 + 1^18) % 9 = 0 := 
by 
  sorry

end NUMINAMATH_GPT_remainder_8_times_10_pow_18_plus_1_pow_18_div_9_l2082_208281


namespace NUMINAMATH_GPT_second_number_value_l2082_208225

theorem second_number_value
  (a b : ℝ)
  (h1 : a * (a - 6) = 7)
  (h2 : b * (b - 6) = 7)
  (h3 : a ≠ b)
  (h4 : a + b = 6) :
  b = 7 := by
sorry

end NUMINAMATH_GPT_second_number_value_l2082_208225


namespace NUMINAMATH_GPT_sum_remainder_l2082_208205

theorem sum_remainder (m : ℤ) : ((9 - m) + (m + 5)) % 8 = 6 :=
by
  sorry

end NUMINAMATH_GPT_sum_remainder_l2082_208205


namespace NUMINAMATH_GPT_candy_necklaces_left_l2082_208219

theorem candy_necklaces_left (total_packs : ℕ) (candy_per_pack : ℕ) 
  (opened_packs : ℕ) (candy_necklaces : ℕ)
  (h1 : total_packs = 9) 
  (h2 : candy_per_pack = 8) 
  (h3 : opened_packs = 4)
  (h4 : candy_necklaces = total_packs * candy_per_pack) :
  (total_packs - opened_packs) * candy_per_pack = 40 :=
by
  sorry

end NUMINAMATH_GPT_candy_necklaces_left_l2082_208219


namespace NUMINAMATH_GPT_equation_solution_l2082_208226

theorem equation_solution (x : ℝ) (h : (2 / (x + 3) + 3 * x / (x + 3) - 5 / (x + 3) = 2)) : x = 9 :=
by
  sorry

end NUMINAMATH_GPT_equation_solution_l2082_208226


namespace NUMINAMATH_GPT_opposite_face_of_x_l2082_208227

theorem opposite_face_of_x 
    (A D F B E x : Prop) 
    (h1 : x → (A ∧ D ∧ F))
    (h2 : x → B)
    (h3 : E → D ∧ ¬x) : B := 
sorry

end NUMINAMATH_GPT_opposite_face_of_x_l2082_208227


namespace NUMINAMATH_GPT_max_value_of_linear_function_l2082_208221

theorem max_value_of_linear_function :
  ∀ (x : ℝ), -3 ≤ x ∧ x ≤ 3 → y = 5 / 3 * x + 2 → ∃ (y_max : ℝ), y_max = 7 ∧ ∀ (x' : ℝ), -3 ≤ x' ∧ x' ≤ 3 → 5 / 3 * x' + 2 ≤ y_max :=
by
  intro x interval_x function_y
  sorry

end NUMINAMATH_GPT_max_value_of_linear_function_l2082_208221


namespace NUMINAMATH_GPT_intersection_M_N_l2082_208276

def M : Set ℝ := { y | ∃ x, y = 2^x ∧ x > 0 }
def N : Set ℝ := { y | ∃ z, y = Real.log z ∧ z ∈ M }

theorem intersection_M_N : M ∩ N = { y | y > 1 } := sorry

end NUMINAMATH_GPT_intersection_M_N_l2082_208276


namespace NUMINAMATH_GPT_circle_radius_l2082_208293

theorem circle_radius (r : ℝ) (x y : ℝ) (h₁ : x = π * r ^ 2) (h₂ : y = 2 * π * r - 6) (h₃ : x + y = 94 * π) : 
  r = 10 :=
sorry

end NUMINAMATH_GPT_circle_radius_l2082_208293


namespace NUMINAMATH_GPT_graph_shift_proof_l2082_208252

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x)
noncomputable def g (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 4)
noncomputable def h (x : ℝ) : ℝ := g (x + Real.pi / 8)

theorem graph_shift_proof : ∀ x, h x = f x := by
  sorry

end NUMINAMATH_GPT_graph_shift_proof_l2082_208252


namespace NUMINAMATH_GPT_polynomial_value_at_minus_2_l2082_208275

-- Define the polynomial f(x)
def f (x : ℤ) := x^6 - 5 * x^5 + 6 * x^4 + x^2 + 3 * x + 2

-- Define the evaluation point
def x_val : ℤ := -2

-- State the theorem we want to prove
theorem polynomial_value_at_minus_2 : f x_val = 320 := 
by sorry

end NUMINAMATH_GPT_polynomial_value_at_minus_2_l2082_208275


namespace NUMINAMATH_GPT_charlene_gave_18_necklaces_l2082_208290

theorem charlene_gave_18_necklaces
  (initial_necklaces : ℕ) (sold_necklaces : ℕ) (left_necklaces : ℕ)
  (h1 : initial_necklaces = 60)
  (h2 : sold_necklaces = 16)
  (h3 : left_necklaces = 26) :
  initial_necklaces - sold_necklaces - left_necklaces = 18 :=
by
  sorry

end NUMINAMATH_GPT_charlene_gave_18_necklaces_l2082_208290


namespace NUMINAMATH_GPT_jogger_ahead_distance_l2082_208213

/-- The jogger is running at a constant speed of 9 km/hr, the train at a speed of 45 km/hr,
    it is 210 meters long and passes the jogger in 41 seconds.
    Prove the jogger is 200 meters ahead of the train. -/
theorem jogger_ahead_distance 
  (v_j : ℝ) (v_t : ℝ) (L : ℝ) (t : ℝ) (d : ℝ) 
  (hv_j : v_j = 9) (hv_t : v_t = 45) (hL : L = 210) (ht : t = 41) :
  d = 200 :=
by {
  -- The conditions and the final proof step, 
  -- actual mathematical proofs steps are not necessary according to the problem statement.
  sorry
}

end NUMINAMATH_GPT_jogger_ahead_distance_l2082_208213


namespace NUMINAMATH_GPT_sin_double_angle_ratio_l2082_208210

theorem sin_double_angle_ratio (α : ℝ) (h : Real.sin α = 3 * Real.cos α) : 
  Real.sin (2 * α) / (Real.cos α)^2 = 6 :=
by 
  sorry

end NUMINAMATH_GPT_sin_double_angle_ratio_l2082_208210


namespace NUMINAMATH_GPT_FB_length_correct_l2082_208246

-- Define a structure for the problem context
structure Triangle (A B C D E F : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E] [MetricSpace F] where
  AB : ℝ
  CD : ℝ
  AE : ℝ
  altitude_CD : C -> (A -> B -> Prop)  -- CD is an altitude to AB
  altitude_AE : E -> (B -> C -> Prop)  -- AE is an altitude to BC
  angle_bisector_AF : F -> (B -> C -> Prop)  -- AF is the angle bisector of ∠BAC intersecting BC at F
  intersect_AF_BC_at_F : (F -> B -> Prop)  -- AF intersects BC at F

noncomputable def length_of_FB (A B C D E F : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E] [MetricSpace F] 
  (t : Triangle A B C D E F) : ℝ := 
  2  -- From given conditions and conclusion

-- The main theorem to prove
theorem FB_length_correct (A B C D E F : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E] [MetricSpace F] 
  (t : Triangle A B C D E F) : 
  t.AB = 8 ∧ t.CD = 3 ∧ t.AE = 4 → length_of_FB A B C D E F t = 2 :=
by
  intro h
  obtain ⟨AB_eq, CD_eq, AE_eq⟩ := h
  sorry

end NUMINAMATH_GPT_FB_length_correct_l2082_208246


namespace NUMINAMATH_GPT_eqn_has_real_root_in_interval_l2082_208245

noncomputable def f (x : ℝ) : ℝ := Real.log (x + 1) + x - 3

theorem eqn_has_real_root_in_interval (k : ℤ) :
  (∃ (x : ℝ), x > k ∧ x < (k + 1) ∧ f x = 0) → k = 2 :=
by
  sorry

end NUMINAMATH_GPT_eqn_has_real_root_in_interval_l2082_208245


namespace NUMINAMATH_GPT_weeds_in_rice_l2082_208269

-- Define the conditions
def total_weight_of_rice := 1536
def sample_size := 224
def weeds_in_sample := 28

-- State the main proof
theorem weeds_in_rice (total_rice : ℕ) (sample_size : ℕ) (weeds_sample : ℕ) 
  (H1 : total_rice = total_weight_of_rice) (H2 : sample_size = sample_size) (H3 : weeds_sample = weeds_in_sample) :
  total_rice * weeds_sample / sample_size = 192 := 
by
  -- Evidence of calculations and external assumptions, translated initial assumptions into mathematical format
  sorry

end NUMINAMATH_GPT_weeds_in_rice_l2082_208269


namespace NUMINAMATH_GPT_triangle_two_solutions_range_of_a_l2082_208274

noncomputable def range_of_a (a b : ℝ) (A : ℝ) : Prop :=
b * Real.sin A < a ∧ a < b

theorem triangle_two_solutions_range_of_a (a : ℝ) (A : ℝ := Real.pi / 6) (b : ℝ := 2) :
  range_of_a a b A ↔ 1 < a ∧ a < 2 := by
sorry

end NUMINAMATH_GPT_triangle_two_solutions_range_of_a_l2082_208274


namespace NUMINAMATH_GPT_son_age_is_9_l2082_208235

-- Definitions for the conditions in the problem
def son_age (S F : ℕ) : Prop := S = (1 / 4 : ℝ) * F - 1
def father_age (S F : ℕ) : Prop := F = 5 * S - 5

-- Main statement of the equivalent problem
theorem son_age_is_9 : ∃ S F : ℕ, son_age S F ∧ father_age S F ∧ S = 9 :=
by
  -- We will leave the proof as an exercise
  sorry

end NUMINAMATH_GPT_son_age_is_9_l2082_208235


namespace NUMINAMATH_GPT_isabel_pictures_l2082_208287

theorem isabel_pictures
  (phone_pics : ℕ)
  (camera_pics : ℕ)
  (total_albums : ℕ)
  (h_phone_pics : phone_pics = 2)
  (h_camera_pics : camera_pics = 4)
  (h_total_albums : total_albums = 3) :
  (phone_pics + camera_pics) / total_albums = 2 :=
by
  sorry

end NUMINAMATH_GPT_isabel_pictures_l2082_208287


namespace NUMINAMATH_GPT_total_skips_l2082_208204

-- Definitions based on conditions
def fifth_throw := 8
def fourth_throw := fifth_throw - 1
def third_throw := fourth_throw + 3
def second_throw := third_throw / 2
def first_throw := second_throw - 2

-- Statement of the proof problem
theorem total_skips : first_throw + second_throw + third_throw + fourth_throw + fifth_throw = 33 := by
  sorry

end NUMINAMATH_GPT_total_skips_l2082_208204


namespace NUMINAMATH_GPT_probability_of_choosing_A_l2082_208273

def P (n : ℕ) : ℝ :=
  if n = 0 then 1 else 0.5 + 0.5 * (-0.2)^(n-1)

theorem probability_of_choosing_A (n : ℕ) :
  P n = if n = 0 then 1 else 0.5 + 0.5 * (-0.2)^(n-1) := 
by {
  sorry
}

end NUMINAMATH_GPT_probability_of_choosing_A_l2082_208273


namespace NUMINAMATH_GPT_solution_set_of_inequality_min_value_of_expression_l2082_208201

def f (x : ℝ) : ℝ := |x + 1| - |2 * x - 2|

-- (I) Prove that the solution set of the inequality f(x) ≥ x - 1 is [0, 2]
theorem solution_set_of_inequality 
  (x : ℝ) : f x ≥ x - 1 ↔ 0 ≤ x ∧ x ≤ 2 := 
sorry

-- (II) Given the maximum value m of f(x) is 2 and a + b + c = 2, prove the minimum value of b^2/a + c^2/b + a^2/c is 2
theorem min_value_of_expression
  (a b c : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c) (h₄ : a + b + c = 2) :
  b^2 / a + c^2 / b + a^2 / c ≥ 2 :=
sorry

end NUMINAMATH_GPT_solution_set_of_inequality_min_value_of_expression_l2082_208201


namespace NUMINAMATH_GPT_sin_alpha_cos_alpha_l2082_208206

theorem sin_alpha_cos_alpha (α : ℝ) (h : Real.sin (3 * Real.pi - α) = -2 * Real.sin (Real.pi / 2 + α)) : 
  Real.sin α * Real.cos α = -2 / 5 :=
by
  sorry

end NUMINAMATH_GPT_sin_alpha_cos_alpha_l2082_208206


namespace NUMINAMATH_GPT_find_number_l2082_208295

-- Define the number x and state the condition 55 + x = 88
def x := 33

-- State the theorem to be proven: if 55 + x = 88, then x = 33
theorem find_number (h : 55 + x = 88) : x = 33 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l2082_208295


namespace NUMINAMATH_GPT_inequality_equivalence_l2082_208214

theorem inequality_equivalence (x : ℝ) : 
  (x + 2) / (x - 1) ≥ 0 ↔ (x + 2) * (x - 1) ≥ 0 :=
sorry

end NUMINAMATH_GPT_inequality_equivalence_l2082_208214


namespace NUMINAMATH_GPT_infinitely_many_digitally_divisible_integers_l2082_208253

theorem infinitely_many_digitally_divisible_integers :
  ∀ n : ℕ, ∃ k : ℕ, k = (10 ^ (3 ^ n) - 1) / 9 ∧ (3 ^ n ∣ k) :=
by
  sorry

end NUMINAMATH_GPT_infinitely_many_digitally_divisible_integers_l2082_208253


namespace NUMINAMATH_GPT_solve_fractional_eq_l2082_208233

theorem solve_fractional_eq (x : ℝ) (hx1 : x ≠ 1 / 3) (hx2 : x ≠ -3) :
  (3 * x + 2) / (3 * x * x + 8 * x - 3) = (3 * x) / (3 * x - 1) ↔ 
  (x = -1 + (Real.sqrt 15) / 3) ∨ (x = -1 - (Real.sqrt 15) / 3) := 
by 
  sorry

end NUMINAMATH_GPT_solve_fractional_eq_l2082_208233


namespace NUMINAMATH_GPT_first_set_broken_percent_l2082_208223

-- Defining some constants
def firstSetTotal : ℕ := 50
def secondSetTotal : ℕ := 60
def secondSetBrokenPercent : ℕ := 20
def totalBrokenMarbles : ℕ := 17

-- Define the function that calculates broken marbles from percentage
def brokenMarbles (percent marbles : ℕ) : ℕ := (percent * marbles) / 100

-- Theorem statement
theorem first_set_broken_percent :
  ∃ (x : ℕ), brokenMarbles x firstSetTotal + brokenMarbles secondSetBrokenPercent secondSetTotal = totalBrokenMarbles ∧ x = 10 :=
by
  sorry

end NUMINAMATH_GPT_first_set_broken_percent_l2082_208223


namespace NUMINAMATH_GPT_sale_in_fifth_month_condition_l2082_208270

theorem sale_in_fifth_month_condition 
  (sale1 sale2 sale3 sale4 sale6 : ℕ)
  (avg_sale : ℕ)
  (n_months : ℕ)
  (total_sales : ℕ)
  (first_four_sales_and_sixth : ℕ) :
  sale1 = 6435 → 
  sale2 = 6927 → 
  sale3 = 6855 → 
  sale4 = 7230 → 
  sale6 = 6791 → 
  avg_sale = 6800 → 
  n_months = 6 → 
  total_sales = avg_sale * n_months → 
  first_four_sales_and_sixth = sale1 + sale2 + sale3 + sale4 + sale6 → 
  ∃ sale5, sale5 = total_sales - first_four_sales_and_sixth ∧ sale5 = 6562 := 
by 
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9
  sorry

end NUMINAMATH_GPT_sale_in_fifth_month_condition_l2082_208270


namespace NUMINAMATH_GPT_probability_of_D_l2082_208259

theorem probability_of_D (P_A P_B P_C P_D : ℚ) (hA : P_A = 1/4) (hB : P_B = 1/3) (hC : P_C = 1/6) 
  (hSum : P_A + P_B + P_C + P_D = 1) : P_D = 1/4 := 
by
  sorry

end NUMINAMATH_GPT_probability_of_D_l2082_208259


namespace NUMINAMATH_GPT_kimberly_total_skittles_l2082_208241

def initial_skittles : ℝ := 7.5
def skittles_eaten : ℝ := 2.25
def skittles_given : ℝ := 1.5
def promotion_skittles : ℝ := 3.75
def oranges_bought : ℝ := 18
def exchange_oranges : ℝ := 6
def exchange_skittles : ℝ := 10.5

theorem kimberly_total_skittles :
  initial_skittles - skittles_eaten - skittles_given + promotion_skittles + exchange_skittles = 18 := by
  sorry

end NUMINAMATH_GPT_kimberly_total_skittles_l2082_208241


namespace NUMINAMATH_GPT_inequality_example_l2082_208284

theorem inequality_example (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a / Real.sqrt (a ^ 2 + 8 * b * c)) + (b / Real.sqrt (b ^ 2 + 8 * c * a)) + (c / Real.sqrt (c ^ 2 + 8 * a * b)) ≥ 1 :=
by
  sorry

end NUMINAMATH_GPT_inequality_example_l2082_208284


namespace NUMINAMATH_GPT_min_value_abs_function_l2082_208279

theorem min_value_abs_function : ∀ x : ℝ, 4 ≤ x ∧ x ≤ 6 → (|x - 4| + |x - 6| = 2) :=
by
  sorry


end NUMINAMATH_GPT_min_value_abs_function_l2082_208279


namespace NUMINAMATH_GPT_inscribed_circle_radius_l2082_208217

theorem inscribed_circle_radius :
  ∀ (a b c : ℝ), a = 3 → b = 6 → c = 18 → (∃ (r : ℝ), (1 / r) = (1 / a) + (1 / b) + (1 / c) + 4 * Real.sqrt ((1 / (a * b)) + (1 / (a * c)) + (1 / (b * c))) ∧ r = 9 / (5 + 6 * Real.sqrt 3)) :=
by
  intros a b c h₁ h₂ h₃
  sorry

end NUMINAMATH_GPT_inscribed_circle_radius_l2082_208217


namespace NUMINAMATH_GPT_evaluate_expression_l2082_208268

theorem evaluate_expression : (20^40) / (40^20) = 10^20 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l2082_208268


namespace NUMINAMATH_GPT_number_of_people_who_bought_1_balloon_l2082_208296

-- Define the variables and the main theorem statement
variables (x1 x2 x3 x4 : ℕ)

theorem number_of_people_who_bought_1_balloon : 
  (x1 + x2 + x3 + x4 = 101) → 
  (x1 + 2 * x2 + 3 * x3 + 4 * x4 = 212) →
  (x4 = x2 + 13) → 
  x1 = 52 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_number_of_people_who_bought_1_balloon_l2082_208296


namespace NUMINAMATH_GPT_original_recipe_pasta_l2082_208244

noncomputable def pasta_per_person (total_pasta : ℕ) (total_people : ℕ) : ℚ :=
  total_pasta / total_people

noncomputable def original_pasta (pasta_per_person : ℚ) (people_served : ℕ) : ℚ :=
  pasta_per_person * people_served

theorem original_recipe_pasta (total_pasta : ℕ) (total_people : ℕ) (people_served : ℕ) (required_pasta : ℚ) :
  total_pasta = 10 → total_people = 35 → people_served = 7 → required_pasta = 2 →
  pasta_per_person total_pasta total_people * people_served = required_pasta :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end NUMINAMATH_GPT_original_recipe_pasta_l2082_208244


namespace NUMINAMATH_GPT_recycling_money_l2082_208202

theorem recycling_money (cans_per_unit : ℕ) (payment_per_unit_cans : ℝ) 
  (newspapers_per_unit : ℕ) (payment_per_unit_newspapers : ℝ) 
  (total_cans : ℕ) (total_newspapers : ℕ) : 
  cans_per_unit = 12 → payment_per_unit_cans = 0.50 → 
  newspapers_per_unit = 5 → payment_per_unit_newspapers = 1.50 → 
  total_cans = 144 → total_newspapers = 20 → 
  (total_cans / cans_per_unit) * payment_per_unit_cans + 
  (total_newspapers / newspapers_per_unit) * payment_per_unit_newspapers = 12 := 
by
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4, h5, h6]
  norm_num
  done

end NUMINAMATH_GPT_recycling_money_l2082_208202


namespace NUMINAMATH_GPT_minimum_value_inequality_l2082_208218

def minimum_value_inequality_problem : Prop :=
∀ (a b : ℝ), (0 < a) → (0 < b) → (a + 3 * b = 1) → (1 / a + 1 / (3 * b)) = 4

theorem minimum_value_inequality : minimum_value_inequality_problem :=
sorry

end NUMINAMATH_GPT_minimum_value_inequality_l2082_208218


namespace NUMINAMATH_GPT_not_sum_of_squares_l2082_208271

def P (x y : ℝ) : ℝ := 4 + x^2 * y^4 + x^4 * y^2 - 3 * x^2 * y^2

theorem not_sum_of_squares (P : ℝ → ℝ → ℝ) : 
  (¬ ∃ g₁ g₂ : ℝ → ℝ → ℝ, ∀ x y : ℝ, P x y = g₁ x y * g₁ x y + g₂ x y * g₂ x y) :=
  by
  {
    -- By contradiction proof as outlined in the example problem
    sorry
  }

end NUMINAMATH_GPT_not_sum_of_squares_l2082_208271


namespace NUMINAMATH_GPT_part1_3_neg5_is_pair_part1_neg2_4_is_not_pair_part2_find_n_part3_find_k_l2082_208267

def is_equation_number_pair (a b : ℝ) : Prop :=
  ∀ x : ℝ, (x = 1 / (a + b) ↔ a / x + 1 = b)

theorem part1_3_neg5_is_pair : is_equation_number_pair 3 (-5) :=
sorry

theorem part1_neg2_4_is_not_pair : ¬ is_equation_number_pair (-2) 4 :=
sorry

theorem part2_find_n (n : ℝ) : is_equation_number_pair n (3 - n) ↔ n = 1 / 2 :=
sorry

theorem part3_find_k (m k : ℝ) (hm : m ≠ -1) (hm0 : m ≠ 0) (hk1 : k ≠ 1) :
  is_equation_number_pair (m - k) k → k = (m^2 + 1) / (m + 1) :=
sorry

end NUMINAMATH_GPT_part1_3_neg5_is_pair_part1_neg2_4_is_not_pair_part2_find_n_part3_find_k_l2082_208267


namespace NUMINAMATH_GPT_equivalent_proof_problem_l2082_208272

variables {a b c d e : ℚ}

theorem equivalent_proof_problem
  (h1 : 3 * a + 4 * b + 6 * c + 8 * d + 10 * e = 55)
  (h2 : 4 * (d + c + e) = b)
  (h3 : 4 * b + 2 * c = a)
  (h4 : c - 2 = d)
  (h5 : d + 1 = e) : 
  a * b * c * d * e = -1912397372 / 78364164096 := 
sorry

end NUMINAMATH_GPT_equivalent_proof_problem_l2082_208272


namespace NUMINAMATH_GPT_bug_probability_nine_moves_l2082_208228

noncomputable def bug_cube_probability (moves : ℕ) : ℚ := sorry

/-- 
The probability that after exactly 9 moves, a bug starting at one vertex of a cube 
and moving randomly along the edges will have visited every vertex exactly once and 
revisited one vertex once more. 
-/
theorem bug_probability_nine_moves : bug_cube_probability 9 = 16 / 6561 := by
  sorry

end NUMINAMATH_GPT_bug_probability_nine_moves_l2082_208228


namespace NUMINAMATH_GPT_problem1_problem2_l2082_208261

theorem problem1 (α : ℝ) (hα : Real.tan α = 2) :
    (Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = 3 := 
sorry

theorem problem2 (α : ℝ) (hα : Real.tan α = 2) :
    (Real.sin (↑(π/2) + α) * Real.cos (↑(5*π/2) - α) * Real.tan (↑(-π) + α)) / 
    (Real.tan (↑(7*π) - α) * Real.sin (↑π + α)) = Real.cos α := 
sorry

end NUMINAMATH_GPT_problem1_problem2_l2082_208261


namespace NUMINAMATH_GPT_a8_equals_two_or_minus_two_l2082_208280

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ n m : ℕ, ∃ r : ℝ, a (n + m) = a n * a m / a 0

theorem a8_equals_two_or_minus_two (a : ℕ → ℝ) 
    (h_geom : geometric_sequence a)
    (h_roots : ∃ x y : ℝ, x^2 - 8 * x + 4 = 0 ∧ y^2 - 8 * y + 4 = 0 ∧ a 6 = x ∧ a 10 = y) :
  a 8 = 2 ∨ a 8 = -2 :=
by
  sorry

end NUMINAMATH_GPT_a8_equals_two_or_minus_two_l2082_208280


namespace NUMINAMATH_GPT_parallel_vectors_m_value_l2082_208207

theorem parallel_vectors_m_value :
  ∀ (m : ℝ), (∀ k : ℝ, (1 : ℝ) = k * m ∧ (-2) = k * (-1)) -> m = (1 / 2) :=
by
  intros m h
  sorry

end NUMINAMATH_GPT_parallel_vectors_m_value_l2082_208207


namespace NUMINAMATH_GPT_triangles_formed_l2082_208288

-- Define the combinatorial function for binomial coefficients.
def binom (n k : ℕ) : ℕ :=
  if h : k ≤ n then Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k)) else 0

-- Given conditions
def points_on_first_line := 6
def points_on_second_line := 8

-- Number of triangles calculation
def total_triangles :=
  binom points_on_first_line 2 * binom points_on_second_line 1 +
  binom points_on_first_line 1 * binom points_on_second_line 2

-- The final theorem to prove
theorem triangles_formed : total_triangles = 288 :=
by
  sorry

end NUMINAMATH_GPT_triangles_formed_l2082_208288


namespace NUMINAMATH_GPT_pears_value_l2082_208264

-- Condition: 3/4 of 12 apples is equivalent to 6 pears
def apples_to_pears (a p : ℕ) : Prop := (3 / 4) * a = 6 * p

-- Target: 1/3 of 9 apples is equivalent to 2 pears
def target_equiv : Prop := (1 / 3) * 9 = 2

theorem pears_value (a p : ℕ) (h : apples_to_pears 12 6) : target_equiv := by
  sorry

end NUMINAMATH_GPT_pears_value_l2082_208264


namespace NUMINAMATH_GPT_union_sets_S_T_l2082_208215

open Set Int

def S : Set Int := { s : Int | ∃ n : Int, s = 2 * n + 1 }
def T : Set Int := { t : Int | ∃ n : Int, t = 4 * n + 1 }

theorem union_sets_S_T : S ∪ T = S := 
by sorry

end NUMINAMATH_GPT_union_sets_S_T_l2082_208215


namespace NUMINAMATH_GPT_compare_expressions_l2082_208299

-- Define the theorem statement
theorem compare_expressions (x : ℝ) : (x - 2) * (x + 3) > x^2 + x - 7 := by
  sorry -- The proof is omitted.

end NUMINAMATH_GPT_compare_expressions_l2082_208299


namespace NUMINAMATH_GPT_sequence_correct_l2082_208291

def seq_formula (n : ℕ) : ℚ := 3/2 + (-1)^n * 11/2

theorem sequence_correct (n : ℕ) :
  (n % 2 = 0 ∧ seq_formula n = 7) ∨ (n % 2 = 1 ∧ seq_formula n = -4) :=
by
  sorry

end NUMINAMATH_GPT_sequence_correct_l2082_208291


namespace NUMINAMATH_GPT_min_cost_per_student_is_80_l2082_208222

def num_students : ℕ := 48
def swims_per_student : ℕ := 8
def cost_per_card : ℕ := 240
def cost_per_bus : ℕ := 40

def total_swims : ℕ := num_students * swims_per_student

def min_cost_per_student : ℕ :=
  let n := 8
  let c := total_swims / n
  let total_cost := cost_per_card * n + cost_per_bus * c
  total_cost / num_students

theorem min_cost_per_student_is_80 :
  min_cost_per_student = 80 :=
sorry

end NUMINAMATH_GPT_min_cost_per_student_is_80_l2082_208222


namespace NUMINAMATH_GPT_determine_a_l2082_208237

theorem determine_a (a : ℝ) :
  (∃ (x y : ℝ), (|y - 10| + |x + 3| - 2) * (x^2 + y^2 - 6) = 0 ∧ (x + 3)^2 + (y - 5)^2 = a) →
  (a = 49 ∨ a = 40 - 4 * Real.sqrt 51) :=
by sorry

end NUMINAMATH_GPT_determine_a_l2082_208237


namespace NUMINAMATH_GPT_Tonya_spent_on_brushes_l2082_208265

section
variable (total_spent : ℝ)
variable (cost_canvases : ℝ)
variable (cost_paints : ℝ)
variable (cost_easel : ℝ)
variable (cost_brushes : ℝ)

def Tonya_total_spent : Prop := total_spent = 90.0
def Cost_of_canvases : Prop := cost_canvases = 40.0
def Cost_of_paints : Prop := cost_paints = cost_canvases / 2
def Cost_of_easel : Prop := cost_easel = 15.0
def Cost_of_brushes : Prop := cost_brushes = total_spent - (cost_canvases + cost_paints + cost_easel)

theorem Tonya_spent_on_brushes : Tonya_total_spent total_spent →
  Cost_of_canvases cost_canvases →
  Cost_of_paints cost_paints cost_canvases →
  Cost_of_easel cost_easel →
  Cost_of_brushes cost_brushes total_spent cost_canvases cost_paints cost_easel →
  cost_brushes = 15.0 := by
  intro h_total_spent h_cost_canvases h_cost_paints h_cost_easel h_cost_brushes
  rw [Tonya_total_spent, Cost_of_canvases, Cost_of_paints, Cost_of_easel, Cost_of_brushes] at *
  sorry
end

end NUMINAMATH_GPT_Tonya_spent_on_brushes_l2082_208265


namespace NUMINAMATH_GPT_number_is_37_5_l2082_208262

theorem number_is_37_5 (y : ℝ) (h : 0.4 * y = 15) : y = 37.5 :=
sorry

end NUMINAMATH_GPT_number_is_37_5_l2082_208262


namespace NUMINAMATH_GPT_roman_numeral_calculation_l2082_208257

def I : ℕ := 1
def V : ℕ := 5
def X : ℕ := 10
def L : ℕ := 50
def C : ℕ := 100
def D : ℕ := 500
def M : ℕ := 1000

theorem roman_numeral_calculation : 2 * M + 5 * L + 7 * X + 9 * I = 2329 := by
  sorry

end NUMINAMATH_GPT_roman_numeral_calculation_l2082_208257


namespace NUMINAMATH_GPT_building_floors_l2082_208243

-- Define the properties of the staircases
def staircaseA_steps : Nat := 104
def staircaseB_steps : Nat := 117
def staircaseC_steps : Nat := 156

-- The problem asks us to show the number of floors, which is the gcd of the steps of all staircases 
theorem building_floors :
  Nat.gcd (Nat.gcd staircaseA_steps staircaseB_steps) staircaseC_steps = 13 :=
by
  sorry

end NUMINAMATH_GPT_building_floors_l2082_208243


namespace NUMINAMATH_GPT_upper_side_length_l2082_208242

variable (L U h : ℝ)

-- Given conditions
def condition1 : Prop := U = L - 6
def condition2 : Prop := 72 = (1 / 2) * (L + U) * 8
def condition3 : Prop := h = 8

-- The length of the upper side of the trapezoid
theorem upper_side_length (h : h = 8) (c1 : U = L - 6) (c2 : 72 = (1 / 2) * (L + U) * 8) : U = 6 := 
by
  sorry

end NUMINAMATH_GPT_upper_side_length_l2082_208242


namespace NUMINAMATH_GPT_average_score_for_girls_at_both_schools_combined_l2082_208292

/-
  The following conditions are given:
  - Average score for boys at Lincoln HS = 75
  - Average score for boys at Monroe HS = 85
  - Average score for boys at both schools combined = 82
  - Average score for girls at Lincoln HS = 78
  - Average score for girls at Monroe HS = 92
  - Average score for boys and girls combined at Lincoln HS = 76
  - Average score for boys and girls combined at Monroe HS = 88

  The goal is to prove that the average score for the girls at both schools combined is 89.
-/
theorem average_score_for_girls_at_both_schools_combined 
  (L l M m : ℕ)
  (h1 : (75 * L + 78 * l) / (L + l) = 76)
  (h2 : (85 * M + 92 * m) / (M + m) = 88)
  (h3 : (75 * L + 85 * M) / (L + M) = 82)
  : (78 * l + 92 * m) / (l + m) = 89 := 
sorry

end NUMINAMATH_GPT_average_score_for_girls_at_both_schools_combined_l2082_208292


namespace NUMINAMATH_GPT_problem_l2082_208240

theorem problem :
  ∀ (x y a b : ℝ), 
  |x + y| + |x - y| = 2 → 
  a > 0 → 
  b > 0 → 
  ∀ z : ℝ, 
  z = 4 * a * x + b * y → 
  (∀ (x y : ℝ), |x + y| + |x - y| = 2 → 4 * a * x + b * y ≤ 1) →
  (1 = 4 * a * 1 + b * 1) →
  (1 = 4 * a * (-1) + b * 1) →
  (1 = 4 * a * (-1) + b * (-1)) →
  (1 = 4 * a * 1 + b * (-1)) →
  ∀ a b : ℝ, a > 0 → b > 0 → (1 = 4 * a + b) →
  (a = 1 / 6 ∧ b = 1 / 3) → 
  (1 / a + 1 / b = 9) :=
by
  sorry

end NUMINAMATH_GPT_problem_l2082_208240


namespace NUMINAMATH_GPT_total_weight_peppers_l2082_208239

def weight_green_peppers : ℝ := 0.3333333333333333
def weight_red_peppers : ℝ := 0.3333333333333333

theorem total_weight_peppers : weight_green_peppers + weight_red_peppers = 0.6666666666666666 := 
by sorry

end NUMINAMATH_GPT_total_weight_peppers_l2082_208239


namespace NUMINAMATH_GPT_alice_probability_multiple_of_4_l2082_208236

noncomputable def probability_one_multiple_of_4 (choices : ℕ) : ℚ :=
  let p_not_multiple_of_4 : ℚ := 45 / 60
  let p_all_not_multiple_of_4 : ℚ := p_not_multiple_of_4 ^ choices
  1 - p_all_not_multiple_of_4

theorem alice_probability_multiple_of_4 :
  probability_one_multiple_of_4 3 = 37 / 64 :=
by
  sorry

end NUMINAMATH_GPT_alice_probability_multiple_of_4_l2082_208236


namespace NUMINAMATH_GPT_tan_add_pi_div_four_sine_cosine_ratio_l2082_208248

-- Definition of the tangent function and trigonometric identities
variable {α : ℝ}

-- Given condition: tan(α) = 2
axiom tan_alpha_eq_2 : Real.tan α = 2

-- Problem 1: Prove that tan(α + π/4) = -3
theorem tan_add_pi_div_four : Real.tan ( α + Real.pi / 4 ) = -3 :=
by
  sorry

-- Problem 2: Prove that (6 * sin(α) + cos(α)) / (3 * sin(α) - cos(α)) = 13 / 5
theorem sine_cosine_ratio : 
  ( 6 * Real.sin α + Real.cos α ) / ( 3 * Real.sin α - Real.cos α ) = 13 / 5 :=
by
  sorry

end NUMINAMATH_GPT_tan_add_pi_div_four_sine_cosine_ratio_l2082_208248


namespace NUMINAMATH_GPT_watch_cost_price_l2082_208203

theorem watch_cost_price (cost_price : ℝ)
  (h1 : SP_loss = 0.90 * cost_price)
  (h2 : SP_gain = 1.08 * cost_price)
  (h3 : SP_gain - SP_loss = 540) :
  cost_price = 3000 := 
sorry

end NUMINAMATH_GPT_watch_cost_price_l2082_208203


namespace NUMINAMATH_GPT_solve_logarithmic_system_l2082_208297

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem solve_logarithmic_system :
  ∃ x y : ℝ, log_base 2 x + log_base 4 y = 4 ∧ log_base 4 x + log_base 2 y = 5 ∧ x = 4 ∧ y = 16 :=
by
  sorry

end NUMINAMATH_GPT_solve_logarithmic_system_l2082_208297


namespace NUMINAMATH_GPT_most_stable_athlete_l2082_208277

theorem most_stable_athlete (s2_A s2_B s2_C s2_D : ℝ) 
  (hA : s2_A = 0.5) 
  (hB : s2_B = 0.5) 
  (hC : s2_C = 0.6) 
  (hD : s2_D = 0.4) :
  s2_D < s2_A ∧ s2_D < s2_B ∧ s2_D < s2_C :=
by
  sorry

end NUMINAMATH_GPT_most_stable_athlete_l2082_208277


namespace NUMINAMATH_GPT_vector_sum_l2082_208282

def v1 : ℤ × ℤ := (5, -3)
def v2 : ℤ × ℤ := (-2, 4)
def scalar : ℤ := 3

theorem vector_sum : 
  (v1.1 + scalar * v2.1, v1.2 + scalar * v2.2) = (-1, 9) := 
by 
  sorry

end NUMINAMATH_GPT_vector_sum_l2082_208282


namespace NUMINAMATH_GPT_minimum_value_of_k_l2082_208298

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * x + b
noncomputable def g (a c : ℝ) (x : ℝ) : ℝ := a * x + c
noncomputable def h (a b c : ℝ) (x : ℝ) : ℝ := (f a b x)^2 + 8 * (g a c x)
noncomputable def k (a b c : ℝ) (x : ℝ) : ℝ := (g a c x)^2 + 8 * (f a b x)

theorem minimum_value_of_k:
  ∃ a b c : ℝ, a ≠ 0 ∧ (∀ x : ℝ, h a b c x ≥ -29) → (∃ x : ℝ, k a b c x = -3) := sorry

end NUMINAMATH_GPT_minimum_value_of_k_l2082_208298


namespace NUMINAMATH_GPT_arithmetic_sequence_25th_term_l2082_208234

theorem arithmetic_sequence_25th_term (a1 a2 : ℕ) (d : ℕ) (n : ℕ) (h1 : a1 = 2) (h2 : a2 = 5) (h3 : d = a2 - a1) (h4 : n = 25) :
  a1 + (n - 1) * d = 74 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_25th_term_l2082_208234


namespace NUMINAMATH_GPT_PTA_money_left_l2082_208232

theorem PTA_money_left (initial_savings : ℝ) (spent_on_supplies : ℝ) (spent_on_food : ℝ) :
  initial_savings = 400 →
  spent_on_supplies = initial_savings / 4 →
  spent_on_food = (initial_savings - spent_on_supplies) / 2 →
  (initial_savings - spent_on_supplies - spent_on_food) = 150 :=
by
  intro initial_savings_eq
  intro spent_on_supplies_eq
  intro spent_on_food_eq
  sorry

end NUMINAMATH_GPT_PTA_money_left_l2082_208232


namespace NUMINAMATH_GPT_divide_inequality_by_negative_l2082_208289

theorem divide_inequality_by_negative {x : ℝ} (h : -6 * x > 2) : x < -1 / 3 :=
by sorry

end NUMINAMATH_GPT_divide_inequality_by_negative_l2082_208289


namespace NUMINAMATH_GPT_scientific_notation_of_858_million_l2082_208266

theorem scientific_notation_of_858_million :
  858000000 = 8.58 * 10 ^ 8 :=
sorry

end NUMINAMATH_GPT_scientific_notation_of_858_million_l2082_208266


namespace NUMINAMATH_GPT_eval_floor_neg_sqrt_l2082_208238

theorem eval_floor_neg_sqrt : (Int.floor (-Real.sqrt (64 / 9)) = -3) := sorry

end NUMINAMATH_GPT_eval_floor_neg_sqrt_l2082_208238


namespace NUMINAMATH_GPT_cube_root_squared_l2082_208209

noncomputable def solve_for_x (x : ℝ) : Prop :=
  (x^(1/3))^2 = 81 → x = 729

theorem cube_root_squared (x : ℝ) :
  solve_for_x x :=
by
  sorry

end NUMINAMATH_GPT_cube_root_squared_l2082_208209


namespace NUMINAMATH_GPT_find_M_l2082_208255

theorem find_M (a b c M : ℝ) (h1 : a + b + c = 120) (h2 : a - 9 = M) (h3 : b + 9 = M) (h4 : 9 * c = M) : 
  M = 1080 / 19 :=
by sorry

end NUMINAMATH_GPT_find_M_l2082_208255


namespace NUMINAMATH_GPT_total_journey_time_l2082_208216

theorem total_journey_time
  (river_speed : ℝ)
  (boat_speed_still_water : ℝ)
  (distance_upstream : ℝ)
  (total_journey_time : ℝ) :
  river_speed = 2 → 
  boat_speed_still_water = 6 → 
  distance_upstream = 48 → 
  total_journey_time = (distance_upstream / (boat_speed_still_water - river_speed) + distance_upstream / (boat_speed_still_water + river_speed)) → 
  total_journey_time = 18 := 
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_total_journey_time_l2082_208216


namespace NUMINAMATH_GPT_price_of_adult_ticket_l2082_208200

theorem price_of_adult_ticket (total_payment : ℕ) (child_price : ℕ) (difference : ℕ) (children : ℕ) (adults : ℕ) (A : ℕ)
  (h1 : total_payment = 720) 
  (h2 : child_price = 8) 
  (h3 : difference = 25) 
  (h4 : children = 15)
  (h5 : adults = children + difference)
  (h6 : total_payment = children * child_price + adults * A) :
  A = 15 :=
by
  sorry

end NUMINAMATH_GPT_price_of_adult_ticket_l2082_208200


namespace NUMINAMATH_GPT_hens_count_l2082_208260

theorem hens_count (H C : ℕ) (h1 : H + C = 48) (h2 : 2 * H + 4 * C = 136) : H = 28 := by
  sorry

end NUMINAMATH_GPT_hens_count_l2082_208260


namespace NUMINAMATH_GPT_m_range_l2082_208263

noncomputable def G (x : ℝ) (m : ℝ) : ℝ := (8 * x^2 + 22 * x + 5 * m) / 8

theorem m_range (m : ℝ) : 2.5 ≤ m ∧ m ≤ 3.5 ↔ m = 121 / 40 := by
  sorry

end NUMINAMATH_GPT_m_range_l2082_208263


namespace NUMINAMATH_GPT_chocolate_discount_l2082_208212

theorem chocolate_discount :
    let original_cost : ℝ := 2
    let final_price : ℝ := 1.43
    let discount := original_cost - final_price
    discount = 0.57 := by
  sorry

end NUMINAMATH_GPT_chocolate_discount_l2082_208212


namespace NUMINAMATH_GPT_three_pow_124_mod_7_l2082_208258

theorem three_pow_124_mod_7 : (3^124) % 7 = 4 := by
  sorry

end NUMINAMATH_GPT_three_pow_124_mod_7_l2082_208258


namespace NUMINAMATH_GPT_joe_first_lift_weight_l2082_208278

variable (x y : ℕ)

def conditions : Prop :=
  (x + y = 1800) ∧ (2 * x = y + 300)

theorem joe_first_lift_weight (h : conditions x y) : x = 700 := by
  sorry

end NUMINAMATH_GPT_joe_first_lift_weight_l2082_208278


namespace NUMINAMATH_GPT_condition_p_neither_sufficient_nor_necessary_l2082_208208

theorem condition_p_neither_sufficient_nor_necessary
  (x : ℝ) :
  (1/x ≤ 1 → x^2 - 2 * x ≥ 0) = false ∧ 
  (x^2 - 2 * x ≥ 0 → 1/x ≤ 1) = false := 
by 
  sorry

end NUMINAMATH_GPT_condition_p_neither_sufficient_nor_necessary_l2082_208208


namespace NUMINAMATH_GPT_fg_of_minus_three_l2082_208247

-- Definitions of the functions f and g
def f (x : ℤ) : ℤ := 2 * x - 1
def g (x : ℤ) : ℤ := x * x + 4

-- The theorem to prove
theorem fg_of_minus_three : f (g (-3)) = 25 := by
  sorry

end NUMINAMATH_GPT_fg_of_minus_three_l2082_208247


namespace NUMINAMATH_GPT_percent_parrots_among_non_pelicans_l2082_208286

theorem percent_parrots_among_non_pelicans 
  (parrots_percent pelicans_percent owls_percent sparrows_percent : ℝ) 
  (H1 : parrots_percent = 40) 
  (H2 : pelicans_percent = 20) 
  (H3 : owls_percent = 15) 
  (H4 : sparrows_percent = 100 - parrots_percent - pelicans_percent - owls_percent)
  (H5 : pelicans_percent / 100 < 1) :
  parrots_percent / (100 - pelicans_percent) * 100 = 50 :=
by sorry

end NUMINAMATH_GPT_percent_parrots_among_non_pelicans_l2082_208286


namespace NUMINAMATH_GPT_find_cd_l2082_208220

def g (c d x : ℝ) : ℝ := c * x^3 - 4 * x^2 + d * x - 7

theorem find_cd :
  let c := -1 / 3
  let d := 28 / 3
  g c d 2 = -7 ∧ g c d (-1) = -20 :=
by sorry

end NUMINAMATH_GPT_find_cd_l2082_208220


namespace NUMINAMATH_GPT_inequality_of_distinct_positives_l2082_208251

variable {a b c : ℝ}

theorem inequality_of_distinct_positives (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
(habc : a ≠ b) (hbc : b ≠ c) (hca : c ≠ a) : 
  2 * (a^3 + b^3 + c^3) > a^2 * (b + c) + b^2 * (a + c) + c^2 * (a + b) :=
by
  sorry

end NUMINAMATH_GPT_inequality_of_distinct_positives_l2082_208251


namespace NUMINAMATH_GPT_complex_multiplication_l2082_208285

def i := Complex.I

theorem complex_multiplication (i := Complex.I) : (-1 + i) * (2 - i) = -1 + 3 * i := 
by 
    -- The actual proof steps would go here.
    sorry

end NUMINAMATH_GPT_complex_multiplication_l2082_208285


namespace NUMINAMATH_GPT_total_students_l2082_208283

theorem total_students (N : ℕ)
    (h1 : (15 * 75) + (10 * 90) = N * 81) :
    N = 25 :=
by
  sorry

end NUMINAMATH_GPT_total_students_l2082_208283


namespace NUMINAMATH_GPT_annual_sales_profit_relationship_and_maximum_l2082_208231

def cost_per_unit : ℝ := 6
def selling_price (x : ℝ) := x > 6
def sales_volume (u : ℝ) := u * 10000
def proportional_condition (x u : ℝ) := (585 / 8) - u = 2 * (x - 21 / 4) ^ 2
def sales_volume_condition : Prop := proportional_condition 10 28

theorem annual_sales_profit_relationship_and_maximum (x u y : ℝ) 
    (hx : selling_price x) 
    (hu : proportional_condition x u) 
    (hs : sales_volume_condition) :
    (y = (-2 * x^3 + 33 * x^2 - 108 * x - 108)) ∧ 
    (x = 9 → y = 135) := 
sorry

end NUMINAMATH_GPT_annual_sales_profit_relationship_and_maximum_l2082_208231


namespace NUMINAMATH_GPT_multiples_7_not_14_less_350_l2082_208254

theorem multiples_7_not_14_less_350 : 
  ∃ n : ℕ, n = 25 ∧ (∀ k : ℕ, k < 350 → (k % 7 = 0 ∧ k % 14 ≠ 0 → k ∈ {7 * m | m : ℕ}) ∨ (k % 14 = 0 → k ∉ {7 * m | m : ℕ})) := 
sorry

end NUMINAMATH_GPT_multiples_7_not_14_less_350_l2082_208254
