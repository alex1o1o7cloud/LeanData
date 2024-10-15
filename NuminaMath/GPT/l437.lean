import Mathlib

namespace NUMINAMATH_GPT_trig_identity_l437_43799

variable {α : ℝ}

theorem trig_identity (h : Real.sin α = 2 * Real.cos α) :
  (Real.sin α - Real.cos α) / (Real.sin α + Real.cos α) = 1 / 3 := by
  sorry

end NUMINAMATH_GPT_trig_identity_l437_43799


namespace NUMINAMATH_GPT_solve_quadratic_equation_l437_43795

theorem solve_quadratic_equation (x : ℝ) : 
  2 * x^2 - 4 * x = 6 - 3 * x ↔ (x = -3/2 ∨ x = 2) :=
by
  sorry

end NUMINAMATH_GPT_solve_quadratic_equation_l437_43795


namespace NUMINAMATH_GPT_inequality_to_prove_l437_43708

variable (x y z : ℝ)

axiom h1 : 0 ≤ x
axiom h2 : 0 ≤ y
axiom h3 : 0 ≤ z
axiom h4 : y * z + z * x + x * y = 1

theorem inequality_to_prove : x * (1 - y)^2 * (1 - z^2) + y * (1 - z^2) * (1 - x^2) + z * (1 - x^2) * (1 - y^2) ≤ (4 / 9) * Real.sqrt 3 :=
by 
  -- The proof is omitted.
  sorry

end NUMINAMATH_GPT_inequality_to_prove_l437_43708


namespace NUMINAMATH_GPT_problem_statement_l437_43787

theorem problem_statement :
  ¬ (3^2 = 6) ∧ 
  ¬ ((-1 / 4) / (-4) = 1) ∧
  ¬ ((-8)^2 = -16) ∧
  (-5 - (-2) = -3) := 
by 
  sorry

end NUMINAMATH_GPT_problem_statement_l437_43787


namespace NUMINAMATH_GPT_distance_between_stripes_l437_43749

theorem distance_between_stripes
  (h1 : ∀ (curbs_are_parallel : Prop), curbs_are_parallel → true)
  (h2 : ∀ (distance_between_curbs : ℝ), distance_between_curbs = 60 → true)
  (h3 : ∀ (length_of_curb : ℝ), length_of_curb = 20 → true)
  (h4 : ∀ (stripe_length : ℝ), stripe_length = 75 → true) :
  ∃ (d : ℝ), d = 16 :=
by
  sorry

end NUMINAMATH_GPT_distance_between_stripes_l437_43749


namespace NUMINAMATH_GPT_find_x2_y2_l437_43763

theorem find_x2_y2 (x y : ℕ) (h1 : x > 0) (h2 : y > 0) (h3 : xy + x + y = 35) (h4 : xy * (x + y) = 360) : x^2 + y^2 = 185 := by
  sorry

end NUMINAMATH_GPT_find_x2_y2_l437_43763


namespace NUMINAMATH_GPT_compare_numbers_l437_43716

theorem compare_numbers : 222^2 < 22^22 ∧ 22^22 < 2^222 :=
by {
  sorry
}

end NUMINAMATH_GPT_compare_numbers_l437_43716


namespace NUMINAMATH_GPT_algebra_or_drafting_not_both_l437_43702

theorem algebra_or_drafting_not_both {A D : Finset ℕ} (h1 : (A ∩ D).card = 10) (h2 : A.card = 24) (h3 : D.card - (A ∩ D).card = 11) : (A ∪ D).card - (A ∩ D).card = 25 := by
  sorry

end NUMINAMATH_GPT_algebra_or_drafting_not_both_l437_43702


namespace NUMINAMATH_GPT_walter_age_in_2001_l437_43740

/-- In 1996, Walter was one-third as old as his grandmother, 
and the sum of the years in which they were born is 3864.
Prove that Walter will be 37 years old at the end of 2001. -/
theorem walter_age_in_2001 (y : ℕ) (H1 : ∃ g, g = 3 * y)
  (H2 : 1996 - y + (1996 - (3 * y)) = 3864) : y + 5 = 37 :=
by sorry

end NUMINAMATH_GPT_walter_age_in_2001_l437_43740


namespace NUMINAMATH_GPT_range_of_a_l437_43744

noncomputable def f (a b x : ℝ) : ℝ := Real.log x - (1/2) * a * x^2 - b * x

theorem range_of_a (a b x : ℝ) (h1 : ∀ x > 0, (1/x) - a * x - b ≠ 0) (h2 : ∀ x > 0, x = 1 → (1/x) - a * x - b = 0) : 
  (1 - a) = b ∧ a > -1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l437_43744


namespace NUMINAMATH_GPT_temperature_difference_l437_43786

theorem temperature_difference (initial_temp rise fall : ℤ) (h1 : initial_temp = 25)
    (h2 : rise = 3) (h3 : fall = 15) : initial_temp + rise - fall = 13 := by
  rw [h1, h2, h3]
  norm_num

end NUMINAMATH_GPT_temperature_difference_l437_43786


namespace NUMINAMATH_GPT_hexagon_same_length_probability_l437_43713

noncomputable def hexagon_probability_same_length : ℚ :=
  let sides := 6
  let diagonals := 9
  let total_segments := sides + diagonals
  let probability_side_first := (sides : ℚ) / total_segments
  let probability_diagonal_first := (diagonals : ℚ) / total_segments
  let probability_second_side := (sides - 1 : ℚ) / (total_segments - 1)
  let probability_second_diagonal_same_length := 2 / (total_segments - 1)
  probability_side_first * probability_second_side + 
  probability_diagonal_first * probability_second_diagonal_same_length

theorem hexagon_same_length_probability : hexagon_probability_same_length = 11 / 35 := 
  sorry

end NUMINAMATH_GPT_hexagon_same_length_probability_l437_43713


namespace NUMINAMATH_GPT_determine_counterfeit_coin_l437_43771

theorem determine_counterfeit_coin (wt_1 wt_2 wt_3 wt_5 : ℕ) (coin : ℕ) :
  (wt_1 = 1) ∧ (wt_2 = 2) ∧ (wt_3 = 3) ∧ (wt_5 = 5) ∧
  (coin = wt_1 ∨ coin = wt_2 ∨ coin = wt_3 ∨ coin = wt_5) ∧
  (coin ≠ 1 ∨ coin ≠ 2 ∨ coin ≠ 3 ∨ coin ≠ 5) → 
  ∃ (counterfeit : ℕ), (counterfeit = 1 ∨ counterfeit = 2 ∨ counterfeit = 3 ∨ counterfeit = 5) ∧ 
  (counterfeit ≠ 1 ∧ counterfeit ≠ 2 ∧ counterfeit ≠ 3 ∧ counterfeit ≠ 5) :=
by
  sorry

end NUMINAMATH_GPT_determine_counterfeit_coin_l437_43771


namespace NUMINAMATH_GPT_team_leaders_lcm_l437_43760

/-- Amanda, Brian, Carla, and Derek are team leaders rotating every
    5, 8, 10, and 12 weeks respectively. Given that this week they all are leading
    projects together, prove that they will all lead projects together again in 120 weeks. -/
theorem team_leaders_lcm :
  Nat.lcm (Nat.lcm 5 8) (Nat.lcm 10 12) = 120 := 
  by
  sorry

end NUMINAMATH_GPT_team_leaders_lcm_l437_43760


namespace NUMINAMATH_GPT_june_biking_time_l437_43797

theorem june_biking_time :
  ∀ (d_jj d_jb : ℕ) (t_jj : ℕ), (d_jj = 2) → (t_jj = 8) → (d_jb = 6) →
  (t_jb : ℕ) → t_jb = (d_jb * t_jj) / d_jj → t_jb = 24 :=
by
  intros d_jj d_jb t_jj h_djj h_tjj h_djb t_jb h_eq
  rw [h_djj, h_tjj, h_djb] at h_eq
  simp at h_eq
  exact h_eq

end NUMINAMATH_GPT_june_biking_time_l437_43797


namespace NUMINAMATH_GPT_gcd_ab_l437_43704

def a := 59^7 + 1
def b := 59^7 + 59^3 + 1

theorem gcd_ab : Nat.gcd a b = 1 := by
  sorry

end NUMINAMATH_GPT_gcd_ab_l437_43704


namespace NUMINAMATH_GPT_greatest_int_less_neg_22_3_l437_43796

theorem greatest_int_less_neg_22_3 : ∃ n : ℤ, n = -8 ∧ n < -22 / 3 ∧ ∀ m : ℤ, m < -22 / 3 → m ≤ n :=
by
  sorry

end NUMINAMATH_GPT_greatest_int_less_neg_22_3_l437_43796


namespace NUMINAMATH_GPT_candidates_count_l437_43798

theorem candidates_count (n : ℕ) (h : n * (n - 1) = 42) : n = 7 :=
by sorry

end NUMINAMATH_GPT_candidates_count_l437_43798


namespace NUMINAMATH_GPT_consecutive_green_balls_l437_43737

theorem consecutive_green_balls : ∃ (fill_ways : ℕ), fill_ways = 21 ∧ 
  (∃ (boxes : Fin 6 → Bool), 
    (∀ i, boxes i = true → 
      (∀ j, boxes j = true → (i ≤ j ∨ j ≤ i)) ∧ 
      ∃ k, boxes k = true)) :=
by
  sorry

end NUMINAMATH_GPT_consecutive_green_balls_l437_43737


namespace NUMINAMATH_GPT_distinct_arrangements_BOOKKEEPER_l437_43729

theorem distinct_arrangements_BOOKKEEPER :
  let n := 9
  let nO := 2
  let nK := 2
  let nE := 3
  ∃ arrangements : ℕ,
  arrangements = Nat.factorial n / (Nat.factorial nO * Nat.factorial nK * Nat.factorial nE) ∧
  arrangements = 15120 :=
by { sorry }

end NUMINAMATH_GPT_distinct_arrangements_BOOKKEEPER_l437_43729


namespace NUMINAMATH_GPT_largest_triangle_angle_l437_43751

theorem largest_triangle_angle (y : ℝ) (h1 : 45 + 60 + y = 180) : y = 75 :=
by { sorry }

end NUMINAMATH_GPT_largest_triangle_angle_l437_43751


namespace NUMINAMATH_GPT_find_width_of_room_l437_43790

theorem find_width_of_room (length room_cost cost_per_sqm total_cost width W : ℕ) 
  (h1 : length = 13)
  (h2 : cost_per_sqm = 12)
  (h3 : total_cost = 1872)
  (h4 : room_cost = length * W * cost_per_sqm)
  (h5 : total_cost = room_cost) : 
  W = 12 := 
by sorry

end NUMINAMATH_GPT_find_width_of_room_l437_43790


namespace NUMINAMATH_GPT_days_per_week_equals_two_l437_43727

-- Definitions based on conditions
def hourly_rate : ℕ := 10
def hours_per_delivery : ℕ := 3
def total_weeks : ℕ := 6
def total_earnings : ℕ := 360

-- Proof statement: determine the number of days per week Jamie delivers flyers is 2
theorem days_per_week_equals_two (d : ℕ) :
  10 * (total_weeks * d * hours_per_delivery) = total_earnings → d = 2 := by
  sorry

end NUMINAMATH_GPT_days_per_week_equals_two_l437_43727


namespace NUMINAMATH_GPT_opposite_of_B_is_I_l437_43720

inductive Face
| A | B | C | D | E | F | G | H | I

open Face

def opposite_face (f : Face) : Face :=
  match f with
  | A => G
  | B => I
  | C => H
  | D => F
  | E => E
  | F => F
  | G => A
  | H => C
  | I => B

theorem opposite_of_B_is_I : opposite_face B = I :=
  by
    sorry

end NUMINAMATH_GPT_opposite_of_B_is_I_l437_43720


namespace NUMINAMATH_GPT_sum_sq_roots_cubic_l437_43723

noncomputable def sum_sq_roots (r s t : ℝ) : ℝ :=
  r^2 + s^2 + t^2

theorem sum_sq_roots_cubic :
  ∀ r s t, (2 * r^3 + 3 * r^2 - 5 * r + 1 = 0) →
           (2 * s^3 + 3 * s^2 - 5 * s + 1 = 0) →
           (2 * t^3 + 3 * t^2 - 5 * t + 1 = 0) →
           (r + s + t = -3 / 2) →
           (r * s + r * t + s * t = 5 / 2) →
           sum_sq_roots r s t = -11 / 4 :=
by 
  intros r s t h₁ h₂ h₃ sum_roots prod_roots
  sorry

end NUMINAMATH_GPT_sum_sq_roots_cubic_l437_43723


namespace NUMINAMATH_GPT_find_some_number_l437_43756

theorem find_some_number : 
  ∃ x : ℝ, 
  (6 + 9 * 8 / x - 25 = 5) ↔ (x = 3) :=
by 
  sorry

end NUMINAMATH_GPT_find_some_number_l437_43756


namespace NUMINAMATH_GPT_percentage_subtracted_l437_43746

theorem percentage_subtracted (a : ℝ) (p : ℝ) (h : (1 - p / 100) * a = 0.97 * a) : p = 3 :=
by
  sorry

end NUMINAMATH_GPT_percentage_subtracted_l437_43746


namespace NUMINAMATH_GPT_sum_possible_values_of_p_l437_43794

theorem sum_possible_values_of_p (p q : ℤ) (h1 : p + q = 2010)
  (h2 : ∃ (α β : ℕ), (10 * α * β = q) ∧ (10 * (α + β) = -p)) :
  p = -3100 :=
by
  sorry

end NUMINAMATH_GPT_sum_possible_values_of_p_l437_43794


namespace NUMINAMATH_GPT_tank_filling_time_l437_43770

theorem tank_filling_time (p q r s : ℝ) (leakage : ℝ) :
  (p = 1 / 6) →
  (q = 1 / 12) →
  (r = 1 / 24) →
  (s = 1 / 18) →
  (leakage = -1 / 48) →
  (1 / (p + q + r + s + leakage) = 48 / 15.67) :=
by
  intros hp hq hr hs hleak
  rw [hp, hq, hr, hs, hleak]
  norm_num
  sorry

end NUMINAMATH_GPT_tank_filling_time_l437_43770


namespace NUMINAMATH_GPT_janice_trash_fraction_l437_43712

noncomputable def janice_fraction : ℚ :=
  let homework := 30
  let cleaning := homework / 2
  let walking_dog := homework + 5
  let total_tasks := homework + cleaning + walking_dog
  let total_time := 120
  let time_left := 35
  let time_spent := total_time - time_left
  let trash_time := time_spent - total_tasks
  trash_time / homework

theorem janice_trash_fraction : janice_fraction = 1 / 6 :=
by
  sorry

end NUMINAMATH_GPT_janice_trash_fraction_l437_43712


namespace NUMINAMATH_GPT_angles_in_interval_l437_43781

-- Define the main statement we need to prove
theorem angles_in_interval (theta : ℝ) (h1 : 0 ≤ theta) (h2 : theta ≤ 2 * Real.pi) :
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → x^2 * Real.cos theta - x * (1 - x) + (1-x)^2 * Real.sin theta < 0) →
  (Real.pi / 2 < theta ∧ theta < 3 * Real.pi / 2) :=
by
  sorry

end NUMINAMATH_GPT_angles_in_interval_l437_43781


namespace NUMINAMATH_GPT_domain_of_sqrt_2cosx_plus_1_l437_43779

noncomputable def domain_sqrt_2cosx_plus_1 (x : ℝ) : Prop :=
  ∃ (k : ℤ), (2 * k * Real.pi - 2 * Real.pi / 3) ≤ x ∧ x ≤ (2 * k * Real.pi + 2 * Real.pi / 3)

theorem domain_of_sqrt_2cosx_plus_1 :
  (∀ (x: ℝ), 0 ≤ 2 * Real.cos x + 1 ↔ domain_sqrt_2cosx_plus_1 x) :=
by
  sorry

end NUMINAMATH_GPT_domain_of_sqrt_2cosx_plus_1_l437_43779


namespace NUMINAMATH_GPT_solve_inequality_prove_inequality_l437_43739

open Real

-- Problem 1: Solve the inequality
theorem solve_inequality (x : ℝ) : (x - 1) / (2 * x + 1) ≤ 0 ↔ (-1 / 2) < x ∧ x ≤ 1 :=
sorry

-- Problem 2: Prove the inequality given positive a, b, and c
theorem prove_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
    (a + b + c) * (1 / a + 1 / (b + c)) ≥ 4 :=
sorry

end NUMINAMATH_GPT_solve_inequality_prove_inequality_l437_43739


namespace NUMINAMATH_GPT_min_value_x_plus_2y_l437_43747

theorem min_value_x_plus_2y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 / x + 1 / y = 4) : x + 2 * y = 2 :=
sorry

end NUMINAMATH_GPT_min_value_x_plus_2y_l437_43747


namespace NUMINAMATH_GPT_no_prime_numbers_divisible_by_91_l437_43710

-- Define the concept of a prime number.
def is_prime (n : ℕ) : Prop :=
  1 < n ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

-- Define the factors of 91.
def factors_of_91 (n : ℕ) : Prop :=
  n = 7 ∨ n = 13

-- State the problem formally: there are no prime numbers divisible by 91.
theorem no_prime_numbers_divisible_by_91 :
  ∀ p : ℕ, is_prime p → ¬ (91 ∣ p) :=
by
  intros p prime_p div91
  sorry

end NUMINAMATH_GPT_no_prime_numbers_divisible_by_91_l437_43710


namespace NUMINAMATH_GPT_frustum_volume_and_lateral_surface_area_l437_43755

theorem frustum_volume_and_lateral_surface_area (h : ℝ) 
    (A1 A2 : ℝ) (r R : ℝ) (V S_lateral : ℝ) : 
    A1 = 4 * Real.pi → 
    A2 = 25 * Real.pi → 
    h = 4 → 
    r = 2 → 
    R = 5 → 
    V = (1 / 3) * (A1 + A2 + Real.sqrt (A1 * A2)) * h → 
    S_lateral = Real.pi * r * Real.sqrt (h ^ 2 + (R - r) ^ 2) + Real.pi * R * Real.sqrt (h ^ 2 + (R - r) ^ 2) → 
    V = 42 * Real.pi ∧ S_lateral = 35 * Real.pi := by
  sorry

end NUMINAMATH_GPT_frustum_volume_and_lateral_surface_area_l437_43755


namespace NUMINAMATH_GPT_spadesuit_evaluation_l437_43759

-- Define the operation
def spadesuit (a b : ℝ) : ℝ := (a + b) * (a - b)

-- The theorem to prove
theorem spadesuit_evaluation : spadesuit 4 (spadesuit 5 (-2)) = -425 :=
by
  sorry

end NUMINAMATH_GPT_spadesuit_evaluation_l437_43759


namespace NUMINAMATH_GPT_count_sets_B_l437_43721

open Set

def A : Set ℕ := {1, 2}

theorem count_sets_B (B : Set ℕ) (h1 : A ∪ B = {1, 2, 3}) : 
  (∃ Bs : Finset (Set ℕ), ∀ b ∈ Bs, A ∪ b = {1, 2, 3} ∧ Bs.card = 4) := sorry

end NUMINAMATH_GPT_count_sets_B_l437_43721


namespace NUMINAMATH_GPT_side_length_of_base_l437_43784

-- Given conditions
def lateral_face_area := 90 -- Area of one lateral face in square meters
def slant_height := 20 -- Slant height in meters

-- The theorem statement
theorem side_length_of_base 
  (s : ℝ)
  (h : ℝ := slant_height)
  (a : ℝ := lateral_face_area)
  (h_area : 2 * a = s * h) :
  s = 9 := 
sorry

end NUMINAMATH_GPT_side_length_of_base_l437_43784


namespace NUMINAMATH_GPT_cub_eqn_root_sum_l437_43777

noncomputable def cos_x := Real.cos (Real.pi / 5)

theorem cub_eqn_root_sum (a b : ℕ) (h1 : a > 0) (h2 : b > 0)
(h3 : a * cos_x ^ 3 - b * cos_x - 1 = 0) : a + b = 12 :=
sorry

end NUMINAMATH_GPT_cub_eqn_root_sum_l437_43777


namespace NUMINAMATH_GPT_music_marks_l437_43745

variable (M : ℕ) -- Variable to represent marks in music

/-- Conditions -/
def science_marks : ℕ := 70
def social_studies_marks : ℕ := 85
def total_marks : ℕ := 275
def physics_marks : ℕ := M / 2

theorem music_marks :
  science_marks + M + social_studies_marks + physics_marks M = total_marks → M = 80 :=
by
  sorry

end NUMINAMATH_GPT_music_marks_l437_43745


namespace NUMINAMATH_GPT_min_value_expression_l437_43783

theorem min_value_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  ( (x + y) / z + (x + z) / y + (y + z) / x + (x + y + z) / (x + y) ) ≥ 7 :=
sorry

end NUMINAMATH_GPT_min_value_expression_l437_43783


namespace NUMINAMATH_GPT_train_length_proof_l437_43732

noncomputable def length_of_first_train (speed1 speed2 : ℝ) (time : ℝ) (length2 : ℝ) : ℝ :=
  let relative_speed := (speed1 + speed2) * (5 / 18) -- convert to m/s
  let total_distance := relative_speed * time
  total_distance - length2

theorem train_length_proof (speed1 speed2 : ℝ) (time : ℝ) (length2 : ℝ) :
  speed1 = 120 →
  speed2 = 80 →
  time = 9 →
  length2 = 270.04 →
  length_of_first_train speed1 speed2 time length2 = 230 :=
by
  intros h1 h2 h3 h4
  -- Use the defined function and simplify
  rw [h1, h2, h3, h4]
  simp [length_of_first_train]
  sorry

end NUMINAMATH_GPT_train_length_proof_l437_43732


namespace NUMINAMATH_GPT_func_translation_right_symm_yaxis_l437_43733

def f (x : ℝ) : ℝ := sorry

theorem func_translation_right_symm_yaxis (f : ℝ → ℝ) 
  (h1 : ∀ x, f (x - 1) = e ^ (-x)) :
  ∀ x, f x = e ^ (-x - 1) := sorry

end NUMINAMATH_GPT_func_translation_right_symm_yaxis_l437_43733


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_for_reciprocal_l437_43754

theorem sufficient_but_not_necessary_for_reciprocal (x : ℝ) : (x > 1 → 1/x < 1) ∧ (¬ (1/x < 1 → x > 1)) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_for_reciprocal_l437_43754


namespace NUMINAMATH_GPT_not_divisible_by_121_l437_43791

theorem not_divisible_by_121 (n : ℤ) : ¬ (121 ∣ (n^2 + 2 * n + 2014)) :=
sorry

end NUMINAMATH_GPT_not_divisible_by_121_l437_43791


namespace NUMINAMATH_GPT_f_of_5_eq_1_l437_43743

noncomputable def f : ℝ → ℝ := sorry

theorem f_of_5_eq_1
    (h1 : ∀ x : ℝ, f (-x) = -f x)
    (h2 : ∀ x : ℝ, f (-x) + f (x + 3) = 0)
    (h3 : f (-1) = 1) :
    f 5 = 1 :=
sorry

end NUMINAMATH_GPT_f_of_5_eq_1_l437_43743


namespace NUMINAMATH_GPT_simplify_expression_l437_43785

theorem simplify_expression : 
  let x := 2
  let y := -1 / 2
  (2 * x^2 + (-x^2 - 2 * x * y + 2 * y^2) - 3 * (x^2 - x * y + 2 * y^2)) = -10 := by
  sorry

end NUMINAMATH_GPT_simplify_expression_l437_43785


namespace NUMINAMATH_GPT_b_seq_formula_c_seq_arithmetic_c_seq_formula_a_seq_formula_sum_S_5_l437_43775

-- Define the sequences a_n, b_n, and c_n along with their properties

-- Definitions
def a_seq (n : ℕ) : ℕ := sorry            -- Define a_n

def S_seq (n : ℕ) : ℕ := sorry            -- Define S_n

def b_seq (n : ℕ) : ℕ := a_seq (n+1) - 2 * a_seq n

def c_seq (n : ℕ) : ℕ := a_seq n / 2^n

-- Conditions
axiom S_n_condition (n : ℕ) : S_seq (n+1) = 4 * a_seq n + 2
axiom a_1_condition : a_seq 1 = 1

-- Goals
theorem b_seq_formula (n : ℕ) : b_seq n = 3 * 2^(n-1) := sorry

theorem c_seq_arithmetic (n : ℕ) : c_seq (n+1) - c_seq n = 3 / 4 := sorry

theorem c_seq_formula (n : ℕ) : c_seq n = (3 * n - 1) / 4 := sorry

theorem a_seq_formula (n : ℕ) : a_seq n = (3 * n - 1) * 2^(n-2) := sorry

theorem sum_S_5 : S_seq 5 = 178 := sorry

end NUMINAMATH_GPT_b_seq_formula_c_seq_arithmetic_c_seq_formula_a_seq_formula_sum_S_5_l437_43775


namespace NUMINAMATH_GPT_b_value_l437_43774

theorem b_value (x y b : ℝ) (h1 : x / (2 * y) = 3 / 2) (h2 : (7 * x + b * y) / (x - 2 * y) = 25) : b = 4 := 
by
  sorry

end NUMINAMATH_GPT_b_value_l437_43774


namespace NUMINAMATH_GPT_four_digit_number_divisible_by_9_l437_43772

theorem four_digit_number_divisible_by_9
    (a b c d e f g h i j : ℕ)
    (h₀ : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧ a ≠ i ∧ a ≠ j ∧
               b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧ b ≠ i ∧ b ≠ j ∧
               c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧ c ≠ i ∧ c ≠ j ∧
               d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧ d ≠ i ∧ d ≠ j ∧
               e ≠ f ∧ e ≠ g ∧ e ≠ h ∧ e ≠ i ∧ e ≠ j ∧
               f ≠ g ∧ f ≠ h ∧ f ≠ i ∧ f ≠ j ∧
               g ≠ h ∧ g ≠ i ∧ g ≠ j ∧
               h ≠ i ∧ h ≠ j ∧
               i ≠ j )
    (h₁ : a + b + c + d + e + f + g + h + i + j = 45)
    (h₂ : 100 * a + 10 * b + c + 100 * d + 10 * e + f = 1000 * g + 100 * h + 10 * i + j) :
  ((1000 * g + 100 * h + 10 * i + j) % 9 = 0) := sorry

end NUMINAMATH_GPT_four_digit_number_divisible_by_9_l437_43772


namespace NUMINAMATH_GPT_parallelogram_area_l437_43773

theorem parallelogram_area
  (a b : ℕ)
  (h1 : a + b = 15)
  (h2 : 2 * a = 3 * b) :
  2 * a = 18 :=
by
  -- Proof is omitted; the statement shows what needs to be proven
  sorry

end NUMINAMATH_GPT_parallelogram_area_l437_43773


namespace NUMINAMATH_GPT_B_work_days_l437_43788

theorem B_work_days (x : ℝ) :
  (1 / 3 + 1 / x = 1 / 2) → x = 6 := by
  sorry

end NUMINAMATH_GPT_B_work_days_l437_43788


namespace NUMINAMATH_GPT_range_of_m_l437_43768

theorem range_of_m (m : ℝ) :
  (∃ (x : ℤ), (x > 2 * m ∧ x ≥ m - 3) ∧ x = 1) ↔ 0 ≤ m ∧ m < 0.5 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l437_43768


namespace NUMINAMATH_GPT_cosine_of_negative_135_l437_43789

theorem cosine_of_negative_135 : Real.cos (-(135 * Real.pi / 180)) = -Real.sqrt 2 / 2 :=
by
  sorry

end NUMINAMATH_GPT_cosine_of_negative_135_l437_43789


namespace NUMINAMATH_GPT_swans_count_l437_43706

def numberOfSwans : Nat := 12

theorem swans_count (y : Nat) (x : Nat) (h1 : y = 5) (h2 : ∃ n m : Nat, x = 2 * n + 2 ∧ x = 3 * m - 3) : x = numberOfSwans := 
  by 
    sorry

end NUMINAMATH_GPT_swans_count_l437_43706


namespace NUMINAMATH_GPT_evaluate_polynomial_at_4_l437_43767

noncomputable def polynomial_horner (x : ℤ) : ℤ :=
  (((((3 * x + 6) * x - 20) * x - 8) * x + 15) * x + 9)

theorem evaluate_polynomial_at_4 :
  polynomial_horner 4 = 3269 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_polynomial_at_4_l437_43767


namespace NUMINAMATH_GPT_interest_difference_l437_43703

noncomputable def principal := 63100
noncomputable def rate := 10 / 100
noncomputable def time := 2

noncomputable def simple_interest := principal * rate * time
noncomputable def compound_interest := principal * (1 + rate)^time - principal

theorem interest_difference :
  (compound_interest - simple_interest) = 671 := by
  sorry

end NUMINAMATH_GPT_interest_difference_l437_43703


namespace NUMINAMATH_GPT_base_number_min_sum_l437_43769

theorem base_number_min_sum (a b : ℕ) (h₁ : 5 * a + 2 = 2 * b + 5) : a + b = 9 :=
by {
  -- this proof is skipped with sorry
  sorry
}

end NUMINAMATH_GPT_base_number_min_sum_l437_43769


namespace NUMINAMATH_GPT_waiter_earnings_l437_43750

theorem waiter_earnings (total_customers tipping_customers no_tip_customers tips_each : ℕ) (h1 : total_customers = 7) (h2 : no_tip_customers = 4) (h3 : tips_each = 9) (h4 : tipping_customers = total_customers - no_tip_customers) :
  tipping_customers * tips_each = 27 :=
by sorry

end NUMINAMATH_GPT_waiter_earnings_l437_43750


namespace NUMINAMATH_GPT_max_possible_value_l437_43776

theorem max_possible_value (a b : ℝ) (h : ∀ n : ℕ, 1 ≤ n → n ≤ 2008 → a + b = a^n + b^n) :
  ∃ a b, ∀ n : ℕ, 1 ≤ n → n ≤ 2008 → a + b = a^n + b^n → ∃ s : ℝ, (s = 0 ∨ s = 1 ∨ s = 2) →
  max (1 / a^(2009) + 1 / b^(2009)) = 2 :=
sorry

end NUMINAMATH_GPT_max_possible_value_l437_43776


namespace NUMINAMATH_GPT_find_x_l437_43761

-- Define the custom operation on m and n
def operation (m n : ℤ) : ℤ := 2 * m - 3 * n

-- Lean statement of the problem
theorem find_x (x : ℤ) (h : operation x 7 = operation 7 x) : x = 7 :=
by {
  -- Placeholder for the proof
  sorry
}

end NUMINAMATH_GPT_find_x_l437_43761


namespace NUMINAMATH_GPT_proof_problem_l437_43793

noncomputable def real_numbers (a x y : ℝ) (h₁ : 0 < a ∧ a < 1) (h₂ : a^x < a^y) : Prop :=
  x^3 > y^3

-- The theorem statement
theorem proof_problem (a x y : ℝ) (h₁ : 0 < a) (h₂ : a < 1) (h₃ : a^x < a^y) : x^3 > y^3 :=
by
  sorry

end NUMINAMATH_GPT_proof_problem_l437_43793


namespace NUMINAMATH_GPT_max_b_minus_a_l437_43730

theorem max_b_minus_a (a b : ℝ) (h_a: a < 0) (h_ineq: ∀ x : ℝ, (3 * x^2 + a) * (2 * x + b) ≥ 0) : 
b - a = 1 / 3 := 
sorry

end NUMINAMATH_GPT_max_b_minus_a_l437_43730


namespace NUMINAMATH_GPT_minimum_value_of_expression_l437_43714

theorem minimum_value_of_expression (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a + b + c = 6) :
  (9 / a + 16 / b + 25 / c) = 24 :=
sorry

end NUMINAMATH_GPT_minimum_value_of_expression_l437_43714


namespace NUMINAMATH_GPT_packs_of_red_bouncy_balls_l437_43717

/-- Given the following conditions:
1. Kate bought 6 packs of yellow bouncy balls.
2. Each pack contained 18 bouncy balls.
3. Kate bought 18 more red bouncy balls than yellow bouncy balls.
Prove that the number of packs of red bouncy balls Kate bought is 7. -/
theorem packs_of_red_bouncy_balls (packs_yellow : ℕ) (balls_per_pack : ℕ) (extra_red_balls : ℕ)
  (h1 : packs_yellow = 6)
  (h2 : balls_per_pack = 18)
  (h3 : extra_red_balls = 18)
  : (packs_yellow * balls_per_pack + extra_red_balls) / balls_per_pack = 7 :=
by
  sorry

end NUMINAMATH_GPT_packs_of_red_bouncy_balls_l437_43717


namespace NUMINAMATH_GPT_rainwater_cows_l437_43780

theorem rainwater_cows (chickens goats cows : ℕ) 
  (h1 : chickens = 18) 
  (h2 : goats = 2 * chickens) 
  (h3 : goats = 4 * cows) : 
  cows = 9 := 
sorry

end NUMINAMATH_GPT_rainwater_cows_l437_43780


namespace NUMINAMATH_GPT_wrongly_noted_mark_l437_43707

theorem wrongly_noted_mark (n : ℕ) (avg_wrong avg_correct correct_mark : ℝ) (x : ℝ)
  (h1 : n = 30)
  (h2 : avg_wrong = 60)
  (h3 : avg_correct = 57.5)
  (h4 : correct_mark = 15)
  (h5 : n * avg_wrong - n * avg_correct = x - correct_mark)
  : x = 90 :=
sorry

end NUMINAMATH_GPT_wrongly_noted_mark_l437_43707


namespace NUMINAMATH_GPT_eggs_sold_l437_43728

/-- Define the notion of trays and eggs in this context -/
def trays_of_eggs : ℤ := 30

/-- Define the initial collection of trays by Haman -/
def initial_trays : ℤ := 10

/-- Define the number of trays dropped by Haman -/
def dropped_trays : ℤ := 2

/-- Define the additional trays that Haman's father told him to collect -/
def additional_trays : ℤ := 7

/-- Define the total eggs sold -/
def total_eggs_sold : ℤ :=
  (initial_trays - dropped_trays) * trays_of_eggs + additional_trays * trays_of_eggs

-- Theorem to prove the total eggs sold
theorem eggs_sold : total_eggs_sold = 450 :=
by 
  -- Insert proof here
  sorry

end NUMINAMATH_GPT_eggs_sold_l437_43728


namespace NUMINAMATH_GPT_perimeter_of_square_l437_43742

variable (s : ℝ) (side_length : ℝ)
def is_square_side_length_5 (s : ℝ) : Prop := s = 5
theorem perimeter_of_square (h: is_square_side_length_5 s) : 4 * s = 20 := sorry

end NUMINAMATH_GPT_perimeter_of_square_l437_43742


namespace NUMINAMATH_GPT_infinitely_many_divisible_by_100_l437_43705

open Nat

theorem infinitely_many_divisible_by_100 : ∀ p : ℕ, ∃ n : ℕ, n = 100 * p + 6 ∧ 100 ∣ (2^n + n^2) := by
  sorry

end NUMINAMATH_GPT_infinitely_many_divisible_by_100_l437_43705


namespace NUMINAMATH_GPT_solve_for_x_l437_43778

theorem solve_for_x (x : ℝ) (h : x^4 = (-3)^4) : x = 3 ∨ x = -3 :=
sorry

end NUMINAMATH_GPT_solve_for_x_l437_43778


namespace NUMINAMATH_GPT_remainder_of_12111_div_3_l437_43726

theorem remainder_of_12111_div_3 : 12111 % 3 = 0 := by
  sorry

end NUMINAMATH_GPT_remainder_of_12111_div_3_l437_43726


namespace NUMINAMATH_GPT_min_value_of_square_sum_l437_43715

theorem min_value_of_square_sum (x y : ℝ) (h : (x-1)^2 + y^2 = 16) : ∃ (a : ℝ), a = x^2 + y^2 ∧ a = 9 :=
by 
  sorry

end NUMINAMATH_GPT_min_value_of_square_sum_l437_43715


namespace NUMINAMATH_GPT_find_B_inter_complement_U_A_l437_43725

-- Define Universal set U
def U : Set ℤ := {-1, 0, 1, 2, 3, 4}

-- Define Set A
def A : Set ℤ := {2, 3}

-- Define complement of A relative to U
def complement_U_A : Set ℤ := U \ A

-- Define set B
def B : Set ℤ := {1, 4}

-- The goal to prove
theorem find_B_inter_complement_U_A : B ∩ complement_U_A = {1, 4} :=
by 
  have h1 : A = {2, 3} := rfl
  have h2 : U = {-1, 0, 1, 2, 3, 4} := rfl
  have h3 : B = {1, 4} := rfl
  sorry

end NUMINAMATH_GPT_find_B_inter_complement_U_A_l437_43725


namespace NUMINAMATH_GPT_value_of_k_l437_43782

theorem value_of_k (k x : ℕ) (h1 : 2^x - 2^(x - 2) = k * 2^10) (h2 : x = 12) : k = 3 := by
  sorry

end NUMINAMATH_GPT_value_of_k_l437_43782


namespace NUMINAMATH_GPT_find_x_l437_43734

theorem find_x (x y : ℕ) (h1 : x / y = 8 / 3) (h2 : y = 27) : x = 72 := by
  sorry

end NUMINAMATH_GPT_find_x_l437_43734


namespace NUMINAMATH_GPT_liquidX_percentage_l437_43722

variable (wA wB : ℝ) (pA pB : ℝ) (mA mB : ℝ)

-- Conditions
def weightA : ℝ := 200
def weightB : ℝ := 700
def percentA : ℝ := 0.8
def percentB : ℝ := 1.8

-- The question and answer.
theorem liquidX_percentage :
  (percentA / 100 * weightA + percentB / 100 * weightB) / (weightA + weightB) * 100 = 1.58 := by
  sorry

end NUMINAMATH_GPT_liquidX_percentage_l437_43722


namespace NUMINAMATH_GPT_walking_rate_on_escalator_l437_43736

theorem walking_rate_on_escalator 
  (escalator_speed person_time : ℝ) 
  (escalator_length : ℝ) 
  (h1 : escalator_speed = 12) 
  (h2 : person_time = 15) 
  (h3 : escalator_length = 210) 
  : (∃ v : ℝ, escalator_length = (v + escalator_speed) * person_time ∧ v = 2) :=
by
  use 2
  rw [h1, h2, h3]
  sorry

end NUMINAMATH_GPT_walking_rate_on_escalator_l437_43736


namespace NUMINAMATH_GPT_correct_system_of_equations_l437_43762

theorem correct_system_of_equations :
  ∃ (x y : ℝ), (4 * x + y = 5 * y + x) ∧ (5 * x + 6 * y = 16) := sorry

end NUMINAMATH_GPT_correct_system_of_equations_l437_43762


namespace NUMINAMATH_GPT_tv_power_consumption_l437_43731

-- Let's define the problem conditions
def hours_per_day : ℕ := 4
def days_per_week : ℕ := 7
def weekly_cost : ℝ := 49              -- in cents
def cost_per_kwh : ℝ := 14             -- in cents

-- Define the theorem to prove the TV power consumption is 125 watts per hour
theorem tv_power_consumption : (weekly_cost / cost_per_kwh) / (hours_per_day * days_per_week) * 1000 = 125 :=
by
  sorry

end NUMINAMATH_GPT_tv_power_consumption_l437_43731


namespace NUMINAMATH_GPT_ram_ravi_selected_probability_l437_43719

noncomputable def probability_both_selected : ℝ := 
  let probability_ram_80 := (1 : ℝ) / 7
  let probability_ravi_80 := (1 : ℝ) / 5
  let probability_both_80 := probability_ram_80 * probability_ravi_80
  let num_applicants := 200
  let num_spots := 4
  let probability_single_selection := (num_spots : ℝ) / (num_applicants : ℝ)
  let probability_both_selected_given_80 := probability_single_selection * probability_single_selection
  probability_both_80 * probability_both_selected_given_80

theorem ram_ravi_selected_probability :
  probability_both_selected = 1 / 87500 := 
by
  sorry

end NUMINAMATH_GPT_ram_ravi_selected_probability_l437_43719


namespace NUMINAMATH_GPT_vanya_first_place_l437_43758

theorem vanya_first_place {n : ℕ} {E A : Finset ℕ} (e_v : ℕ) (a_v : ℕ)
  (he_v : e_v = n)
  (h_distinct_places : E.card = (E ∪ A).card)
  (h_all_worse : ∀ e_i ∈ E, e_i ≠ e_v → ∃ a_i ∈ A, a_i > e_i)
  : a_v = 1 := 
sorry

end NUMINAMATH_GPT_vanya_first_place_l437_43758


namespace NUMINAMATH_GPT_modulo_4_equiv_2_l437_43711

open Nat

noncomputable def f (n : ℕ) [Fintype (ZMod n)] : ZMod n → ZMod n := sorry

theorem modulo_4_equiv_2 (n : ℕ) [hn : Fact (n > 0)] 
  (f : ZMod n → ZMod n)
  (h1 : ∀ x, f x ≠ x)
  (h2 : ∀ x, f (f x) = x)
  (h3 : ∀ x, f (f (f (x + 1) + 1) + 1) = x) : 
  n % 4 = 2 := 
sorry

end NUMINAMATH_GPT_modulo_4_equiv_2_l437_43711


namespace NUMINAMATH_GPT_intersection_A_B_l437_43735

open Set

variable (x : ℝ)

def A : Set ℝ := {x | x^2 - 3 * x - 4 < 0}
def B : Set ℝ := {-4, 1, 3, 5}

theorem intersection_A_B :
  A ∩ B = { 1, 3 } :=
sorry

end NUMINAMATH_GPT_intersection_A_B_l437_43735


namespace NUMINAMATH_GPT_tip_percentage_l437_43765

/--
A family paid $30 for food, the sales tax rate is 9.5%, and the total amount paid was $35.75. Prove that the tip percentage is 9.67%.
-/
theorem tip_percentage (food_cost : ℝ) (sales_tax_rate : ℝ) (total_paid : ℝ)
  (h1 : food_cost = 30)
  (h2 : sales_tax_rate = 0.095)
  (h3 : total_paid = 35.75) :
  ((total_paid - (food_cost * (1 + sales_tax_rate))) / food_cost) * 100 = 9.67 :=
by
  sorry

end NUMINAMATH_GPT_tip_percentage_l437_43765


namespace NUMINAMATH_GPT_abs_mult_example_l437_43718

theorem abs_mult_example : (|(-3)| * 2) = 6 := by
  have h1 : |(-3)| = 3 := by
    exact abs_of_neg (show -3 < 0 by norm_num)
  rw [h1]
  exact mul_eq_mul_left_iff.mpr (Or.inl rfl)

end NUMINAMATH_GPT_abs_mult_example_l437_43718


namespace NUMINAMATH_GPT_books_sold_online_l437_43757

theorem books_sold_online (X : ℤ) 
  (h1: 743 = 502 + (37 + X) + (74 + X + 34) - 160) : 
  X = 128 := 
by sorry

end NUMINAMATH_GPT_books_sold_online_l437_43757


namespace NUMINAMATH_GPT_triangle_interior_angles_not_greater_than_60_l437_43738

theorem triangle_interior_angles_not_greater_than_60 (α β γ : ℝ) (h_sum : α + β + γ = 180) 
  (h_pos : α > 0 ∧ β > 0 ∧ γ > 0) :
  α ≤ 60 ∨ β ≤ 60 ∨ γ ≤ 60 :=
by
  sorry

end NUMINAMATH_GPT_triangle_interior_angles_not_greater_than_60_l437_43738


namespace NUMINAMATH_GPT_carlton_outfits_l437_43701

theorem carlton_outfits (button_up_shirts sweater_vests : ℕ) 
  (h1 : sweater_vests = 2 * button_up_shirts)
  (h2 : button_up_shirts = 3) :
  sweater_vests * button_up_shirts = 18 :=
by
  sorry

end NUMINAMATH_GPT_carlton_outfits_l437_43701


namespace NUMINAMATH_GPT_solution_of_abs_square_eq_zero_l437_43792

-- Define the given conditions as hypotheses
variables {x y : ℝ}
theorem solution_of_abs_square_eq_zero (h : |x + 2| + (y - 1)^2 = 0) : x = -2 ∧ y = 1 :=
sorry

end NUMINAMATH_GPT_solution_of_abs_square_eq_zero_l437_43792


namespace NUMINAMATH_GPT_adoption_days_l437_43766

theorem adoption_days (P0 P_in P_adopt_rate : Nat) (P_total : Nat) (hP0 : P0 = 3) (hP_in : P_in = 3) (hP_adopt_rate : P_adopt_rate = 3) (hP_total : P_total = P0 + P_in) :
  P_total / P_adopt_rate = 2 := 
by
  sorry

end NUMINAMATH_GPT_adoption_days_l437_43766


namespace NUMINAMATH_GPT_compare_P_Q_l437_43709

noncomputable def P : ℝ := Real.sqrt 7 - 1
noncomputable def Q : ℝ := Real.sqrt 11 - Real.sqrt 5

theorem compare_P_Q : P > Q :=
sorry

end NUMINAMATH_GPT_compare_P_Q_l437_43709


namespace NUMINAMATH_GPT_find_y_plus_inv_y_l437_43724

theorem find_y_plus_inv_y (y : ℝ) (h : y^3 + 1 / y^3 = 110) : y + 1 / y = 5 :=
sorry

end NUMINAMATH_GPT_find_y_plus_inv_y_l437_43724


namespace NUMINAMATH_GPT_winner_more_than_third_l437_43741

theorem winner_more_than_third (W S T F : ℕ) (h1 : F = 199) 
(h2 : W = F + 105) (h3 : W = S + 53) (h4 : W + S + T + F = 979) : 
W - T = 79 :=
by
  -- Here, the proof steps would go, but they are not required as per instructions.
  sorry

end NUMINAMATH_GPT_winner_more_than_third_l437_43741


namespace NUMINAMATH_GPT_fixed_point_exists_l437_43700

theorem fixed_point_exists (m : ℝ) :
  ∀ (x y : ℝ), (2 * m + 1) * x + (m + 1) * y - 7 * m - 4 = 0 → x = 3 ∧ y = 1 :=
by
  sorry

end NUMINAMATH_GPT_fixed_point_exists_l437_43700


namespace NUMINAMATH_GPT_range_of_a_l437_43752

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  a * Real.log x + (1/2) * x^2

theorem range_of_a (a : ℝ) (h : 0 < a) :
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → 0 < x₁ → 0 < x₂ → (f a x₁ - f a x₂) / (x₁ - x₂) ≥ 2) ↔ (1 ≤ a) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l437_43752


namespace NUMINAMATH_GPT_simplify_fraction_l437_43748

theorem simplify_fraction : 
  (1 / (1 / (Real.sqrt 2 + 1) + 2 / (Real.sqrt 3 - 1))) = Real.sqrt 3 - Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_simplify_fraction_l437_43748


namespace NUMINAMATH_GPT_solve_for_star_l437_43764

theorem solve_for_star : ∀ (star : ℝ), (45 - (28 - (37 - (15 - star))) = 54) → star = 15 := by
  intros star h
  sorry

end NUMINAMATH_GPT_solve_for_star_l437_43764


namespace NUMINAMATH_GPT_infinite_primes_of_the_year_2022_l437_43753

theorem infinite_primes_of_the_year_2022 :
  ∃ᶠ p in Filter.atTop, ∃ n : ℕ, p % 2 = 1 ∧ p ^ 2022 ∣ n ^ 2022 + 2022 :=
sorry

end NUMINAMATH_GPT_infinite_primes_of_the_year_2022_l437_43753
