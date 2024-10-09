import Mathlib

namespace geometric_sum_3030_l483_48330

theorem geometric_sum_3030 {a r : ℝ}
  (h1 : a * (1 - r ^ 1010) / (1 - r) = 300)
  (h2 : a * (1 - r ^ 2020) / (1 - r) = 540) :
  a * (1 - r ^ 3030) / (1 - r) = 732 :=
sorry

end geometric_sum_3030_l483_48330


namespace percent_value_quarters_l483_48395

noncomputable def value_in_cents (dimes quarters nickels : ℕ) : ℕ := 
  (dimes * 10) + (quarters * 25) + (nickels * 5)

noncomputable def percent_in_quarters (quarters total_value : ℕ) : ℚ := 
  (quarters * 25 : ℚ) / total_value * 100

theorem percent_value_quarters 
  (h_dimes : ℕ := 80) 
  (h_quarters : ℕ := 30) 
  (h_nickels : ℕ := 40) 
  (h_total_value := value_in_cents h_dimes h_quarters h_nickels) : 
  percent_in_quarters h_quarters h_total_value = 42.86 :=
by sorry

end percent_value_quarters_l483_48395


namespace students_in_line_l483_48365

theorem students_in_line (between : ℕ) (Yoojung Eunji : ℕ) (h1 : Yoojung = 1) (h2 : Eunji = 1) : 
  between + Yoojung + Eunji = 16 :=
  sorry

end students_in_line_l483_48365


namespace true_proposition_among_ABCD_l483_48333

theorem true_proposition_among_ABCD : 
  (∀ x : ℝ, x^2 < x + 1) = false ∧
  (∀ x : ℝ, x^2 ≥ x + 1) = false ∧
  (∃ x : ℝ, ∀ y : ℝ, x * y^2 ≠ y^2) = true ∧
  (∀ x : ℝ, ∃ y : ℝ, x > y^2) = false :=
by 
  sorry

end true_proposition_among_ABCD_l483_48333


namespace smallest_multiple_of_6_and_9_l483_48362

theorem smallest_multiple_of_6_and_9 : ∃ n : ℕ, n > 0 ∧ (n % 6 = 0) ∧ (n % 9 = 0) ∧ ∀ m : ℕ, m > 0 ∧ (m % 6 = 0) ∧ (m % 9 = 0) → n ≤ m :=
  by
    sorry

end smallest_multiple_of_6_and_9_l483_48362


namespace quadratic_inequality_l483_48355

noncomputable def exists_real_roots (a : ℝ) : Prop :=
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1^2 + (a - 1) * x1 + (2 * a - 5) = 0 ∧ x2^2 + (a - 1) * x2 + (2 * a - 5) = 0

noncomputable def valid_values (a : ℝ) : Prop :=
  a > 5 / 2 ∧ a < 10

theorem quadratic_inequality (a : ℝ) 
  (h1 : exists_real_roots a) 
  (h2 : ∀ x1 x2 : ℝ, x1 ≠ x2 ∧ x1^2 + (a - 1) * x1 + (2 * a - 5) = 0 ∧ x2^2 + (a - 1) * x2 + (2 * a - 5) = 0 
  → (1 / x1 + 1 / x2 < -3 / 5)) :
  valid_values a :=
sorry

end quadratic_inequality_l483_48355


namespace trig_identity_l483_48386

theorem trig_identity : (Real.cos (15 * Real.pi / 180))^2 - (Real.sin (15 * Real.pi / 180))^2 = Real.sqrt 3 / 2 :=
by
  sorry

end trig_identity_l483_48386


namespace find_value_added_l483_48358

theorem find_value_added :
  ∀ (n x : ℤ), (2 * n + x = 8 * n - 4) → (n = 4) → (x = 20) :=
by
  intros n x h1 h2
  sorry

end find_value_added_l483_48358


namespace postit_notes_area_l483_48304

theorem postit_notes_area (length width adhesive_len : ℝ) (num_notes : ℕ)
  (h_length : length = 9.4) (h_width : width = 3.7) (h_adh_len : adhesive_len = 0.6) (h_num_notes : num_notes = 15) :
  (length + (length - adhesive_len) * (num_notes - 1)) * width = 490.62 :=
by
  rw [h_length, h_width, h_adh_len, h_num_notes]
  sorry

end postit_notes_area_l483_48304


namespace estimated_probability_l483_48334

noncomputable def needle_intersection_probability : ℝ := 0.4

structure NeedleExperimentData :=
(distance_between_lines : ℝ)
(length_of_needle : ℝ)
(num_trials_intersections : List (ℕ × ℕ))
(intersection_frequencies : List ℝ)

def experiment_data : NeedleExperimentData :=
{ distance_between_lines := 5,
  length_of_needle := 3,
  num_trials_intersections := [(50, 23), (100, 48), (200, 83), (500, 207), (1000, 404), (2000, 802)],
  intersection_frequencies := [0.460, 0.480, 0.415, 0.414, 0.404, 0.401] }

theorem estimated_probability (data : NeedleExperimentData) :
  ∀ P : ℝ, (∀ n m, (n, m) ∈ data.num_trials_intersections → abs (m / n - P) < 0.1) → P = needle_intersection_probability :=
by
  intro P hP
  sorry

end estimated_probability_l483_48334


namespace find_side_length_l483_48380

def hollow_cube_formula (n : ℕ) : ℕ :=
  6 * n^2 - (n^2 + 4 * (n - 2))

theorem find_side_length :
  ∃ n : ℕ, hollow_cube_formula n = 98 ∧ n = 9 :=
by
  sorry

end find_side_length_l483_48380


namespace expression_evaluation_l483_48384

theorem expression_evaluation :
  (1007 * (((7/4 : ℚ) / (3/4) + (3 / (9/4)) + (1/3)) /
    ((1 + 2 + 3 + 4 + 5) * 5 - 22)) / 19) = (4 : ℚ) :=
by
  sorry

end expression_evaluation_l483_48384


namespace solve_for_x_l483_48328

namespace RationalOps

-- Define the custom operation ※ on rational numbers
def star (a b : ℚ) : ℚ := a + b

-- Define the equation involving the custom operation
def equation (x : ℚ) : Prop := star 4 (star x 3) = 1

-- State the theorem to prove the solution
theorem solve_for_x : ∃ x : ℚ, equation x ∧ x = -6 := by
  sorry

end solve_for_x_l483_48328


namespace library_wall_length_l483_48329

theorem library_wall_length 
  (D B : ℕ) 
  (h1: D = B) 
  (desk_length bookshelf_length leftover_space : ℝ) 
  (h2: desk_length = 2) 
  (h3: bookshelf_length = 1.5) 
  (h4: leftover_space = 1) : 
  3.5 * D + leftover_space = 8 :=
by { sorry }

end library_wall_length_l483_48329


namespace boundary_length_of_pattern_l483_48326

theorem boundary_length_of_pattern (area : ℝ) (num_points : ℕ) 
(points_per_side : ℕ) : 
area = 144 → num_points = 4 → points_per_side = 4 →
∃ length : ℝ, length = 92.5 :=
by
  intros
  sorry

end boundary_length_of_pattern_l483_48326


namespace box_volume_increase_l483_48309

theorem box_volume_increase (l w h : ℝ) 
  (h1 : l * w * h = 5000)
  (h2 : l * w + w * h + h * l = 900)
  (h3 : l + w + h = 60) : 
  (l + 2) * (w + 2) * (h + 2) = 7048 := 
by 
  sorry

end box_volume_increase_l483_48309


namespace sequence_periodic_l483_48300

def last_digit (n : ℕ) : ℕ := n % 10

noncomputable def a_n (n : ℕ) : ℕ := last_digit (n^(n^n))

theorem sequence_periodic :
  ∃ period : ℕ, period = 20 ∧ ∀ n m : ℕ, n ≡ m [MOD period] → a_n n = a_n m :=
sorry

end sequence_periodic_l483_48300


namespace tan_addition_identity_l483_48351

theorem tan_addition_identity 
  (tan_30 : Real := Real.tan (Real.pi / 6))
  (tan_15 : Real := 2 - Real.sqrt 3) : 
  tan_15 + tan_30 + tan_15 * tan_30 = 1 := 
by
  have h1 : tan_30 = Real.sqrt 3 / 3 := sorry
  have h2 : tan_15 = 2 - Real.sqrt 3 := sorry
  sorry

end tan_addition_identity_l483_48351


namespace first_three_workers_dig_time_l483_48331

variables 
  (a b c d : ℝ) -- work rates of the four workers
  (hours : ℝ) -- time to dig the trench

def work_together (a b c d hours : ℝ) := (a + b + c + d) * hours = 1

def scenario1 (a b c d : ℝ) := (2 * a + (1/2) * b + c + d) * 6 = 1

def scenario2 (a b c d : ℝ) := (a/2 + 2 * b + c + d) * 4 = 1

theorem first_three_workers_dig_time
  (h1 : work_together a b c d 6)
  (h2 : scenario1 a b c d)
  (h3 : scenario2 a b c d) :
  hours = 6 := 
sorry

end first_three_workers_dig_time_l483_48331


namespace highlighter_total_l483_48372

theorem highlighter_total 
  (pink_highlighters : ℕ)
  (yellow_highlighters : ℕ)
  (blue_highlighters : ℕ)
  (h_pink : pink_highlighters = 4)
  (h_yellow : yellow_highlighters = 2)
  (h_blue : blue_highlighters = 5) :
  pink_highlighters + yellow_highlighters + blue_highlighters = 11 :=
by
  sorry

end highlighter_total_l483_48372


namespace find_coefficient_y_l483_48350

theorem find_coefficient_y (a b c : ℕ) (h1 : 100 * a + 10 * b + c - 7 * (a + b + c) = 100) (h2 : a + b + c ≠ 0) :
  100 * c + 10 * b + a = 43 * (a + b + c) :=
by
  sorry

end find_coefficient_y_l483_48350


namespace probability_of_winning_reward_l483_48316

-- Definitions representing the problem conditions
def red_envelopes : ℕ := 4
def card_types : ℕ := 3

-- Theorem statement: Prove the probability of winning the reward is 4/9
theorem probability_of_winning_reward : 
  (∃ (n m : ℕ), n = card_types^red_envelopes ∧ m = (Nat.choose red_envelopes 2) * (Nat.factorial 3)) → 
  (m / n = 4/9) :=
by
  sorry  -- Proof to be filled in

end probability_of_winning_reward_l483_48316


namespace smallest_number_exceeding_triangle_perimeter_l483_48392

theorem smallest_number_exceeding_triangle_perimeter (a b : ℕ) (a_eq_7 : a = 7) (b_eq_21 : b = 21) :
  ∃ P : ℕ, (∀ c : ℝ, 14 < c ∧ c < 28 → a + b + c < P) ∧ P = 56 := by
  sorry

end smallest_number_exceeding_triangle_perimeter_l483_48392


namespace jackson_saving_l483_48369

theorem jackson_saving (total_amount : ℝ) (months : ℕ) (paychecks_per_month : ℕ) (savings_per_paycheck : ℝ) :
  total_amount = 3000 → months = 15 → paychecks_per_month = 2 →
  savings_per_paycheck = total_amount / months / paychecks_per_month :=
by sorry

end jackson_saving_l483_48369


namespace henry_twice_jill_l483_48303

-- Conditions
def Henry := 29
def Jill := 19
def sum_ages : Nat := Henry + Jill

-- Prove the statement
theorem henry_twice_jill (Y : Nat) (H J : Nat) (h_sum : H + J = 48) (h_H : H = 29) (h_J : J = 19) :
  H - Y = 2 * (J - Y) ↔ Y = 9 :=
by {
  -- Here, we would provide the proof, but we'll skip that with sorry.
  sorry
}

end henry_twice_jill_l483_48303


namespace chairs_to_exclude_l483_48308

theorem chairs_to_exclude (chairs : ℕ) (h : chairs = 765) : 
  ∃ n, n^2 ≤ chairs ∧ chairs - n^2 = 36 := 
by 
  sorry

end chairs_to_exclude_l483_48308


namespace possible_rectangular_arrays_l483_48399

theorem possible_rectangular_arrays (n : ℕ) (h : n = 48) :
  ∃ (m k : ℕ), m * k = n ∧ 2 ≤ m ∧ 2 ≤ k :=
sorry

end possible_rectangular_arrays_l483_48399


namespace geometric_sum_common_ratios_l483_48354

theorem geometric_sum_common_ratios (k p r : ℝ) 
  (hp : p ≠ r) (h_seq : p ≠ 1 ∧ r ≠ 1 ∧ p ≠ 0 ∧ r ≠ 0) 
  (h : k * p^4 - k * r^4 = 4 * (k * p^2 - k * r^2)) : 
  p + r = 3 :=
by
  -- Details omitted as requested
  sorry

end geometric_sum_common_ratios_l483_48354


namespace quadratic_variation_y_l483_48383

theorem quadratic_variation_y (k : ℝ) (x y : ℝ) (h1 : y = k * x^2) (h2 : (25 : ℝ) = k * (5 : ℝ)^2) :
  y = 25 :=
by
sorry

end quadratic_variation_y_l483_48383


namespace annie_has_12_brownies_left_l483_48381

noncomputable def initial_brownies := 100
noncomputable def portion_for_admin := (3 / 5 : ℚ) * initial_brownies
noncomputable def leftover_after_admin := initial_brownies - portion_for_admin
noncomputable def portion_for_carl := (1 / 4 : ℚ) * leftover_after_admin
noncomputable def leftover_after_carl := leftover_after_admin - portion_for_carl
noncomputable def portion_for_simon := 3
noncomputable def leftover_after_simon := leftover_after_carl - portion_for_simon
noncomputable def portion_for_friends := (2 / 3 : ℚ) * leftover_after_simon
noncomputable def each_friend_get := portion_for_friends / 5
noncomputable def total_given_to_friends := each_friend_get * 5
noncomputable def final_brownies := leftover_after_simon - total_given_to_friends

theorem annie_has_12_brownies_left : final_brownies = 12 := by
  sorry

end annie_has_12_brownies_left_l483_48381


namespace range_of_x_l483_48310

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then Real.log x / Real.log 2
  else Real.log (-x) / Real.log (1 / 2)

theorem range_of_x (x : ℝ) : f x > f (-x) ↔ (x > 1) ∨ (-1 < x ∧ x < 0) :=
by
  sorry

end range_of_x_l483_48310


namespace total_toys_l483_48323

theorem total_toys (K A L : ℕ) (h1 : A = K + 30) (h2 : L = 2 * K) (h3 : K + A = 160) : 
    K + A + L = 290 :=
by
  sorry

end total_toys_l483_48323


namespace sqrt_abc_sum_l483_48301

variable (a b c : ℝ)

theorem sqrt_abc_sum (h1 : b + c = 17) (h2 : c + a = 20) (h3 : a + b = 23) :
  Real.sqrt (a * b * c * (a + b + c)) = 10 * Real.sqrt 273 := by
  sorry

end sqrt_abc_sum_l483_48301


namespace expand_binomial_l483_48352

theorem expand_binomial (x : ℝ) : (x + 3) * (x + 8) = x^2 + 11 * x + 24 :=
by sorry

end expand_binomial_l483_48352


namespace john_pays_more_than_jane_l483_48394

noncomputable def original_price : ℝ := 34.00
noncomputable def discount : ℝ := 0.10
noncomputable def tip_percent : ℝ := 0.15

noncomputable def discounted_price : ℝ := original_price - (discount * original_price)
noncomputable def john_tip : ℝ := tip_percent * original_price
noncomputable def john_total : ℝ := discounted_price + john_tip
noncomputable def jane_tip : ℝ := tip_percent * discounted_price
noncomputable def jane_total : ℝ := discounted_price + jane_tip

theorem john_pays_more_than_jane : john_total - jane_total = 0.51 := by
  sorry

end john_pays_more_than_jane_l483_48394


namespace coin_flip_probability_l483_48382

/--
Suppose we flip five coins simultaneously: a penny, a nickel, a dime, a quarter, and a half-dollar.
What is the probability that the penny and dime both come up heads, and the half-dollar comes up tails?
-/

theorem coin_flip_probability :
  let outcomes := 2^5
  let success := 1 * 1 * 1 * 2 * 2
  success / outcomes = (1 : ℚ) / 8 :=
by
  /- Proof goes here -/
  sorry

end coin_flip_probability_l483_48382


namespace magic_square_d_e_sum_l483_48335

theorem magic_square_d_e_sum 
  (S : ℕ)
  (a b c d e : ℕ)
  (h1 : S = 45 + d)
  (h2 : S = 51 + e) :
  d + e = 57 :=
by
  sorry

end magic_square_d_e_sum_l483_48335


namespace parabola_vertex_y_coordinate_l483_48366

theorem parabola_vertex_y_coordinate (x y : ℝ) :
  y = 5 * x^2 + 20 * x + 45 ∧ (∃ h k, y = 5 * (x + h)^2 + k ∧ k = 25) :=
by
  sorry

end parabola_vertex_y_coordinate_l483_48366


namespace min_value_of_f_l483_48389

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x + Real.sin (2 * x)

theorem min_value_of_f : 
  ∃ x : ℝ, ∀ y : ℝ, f x ≤ f y :=
sorry

end min_value_of_f_l483_48389


namespace convert_rectangular_to_spherical_l483_48321

theorem convert_rectangular_to_spherical :
  ∀ (x y z : ℝ) (ρ θ φ : ℝ),
    (x, y, z) = (2, -2 * Real.sqrt 2, 2) →
    ρ = Real.sqrt (x^2 + y^2 + z^2) →
    z = ρ * Real.cos φ →
    x = ρ * Real.sin φ * Real.cos θ →
    y = ρ * Real.sin φ * Real.sin θ →
    0 < ρ ∧ 0 ≤ θ ∧ θ < 2 * Real.pi ∧ 0 ≤ φ ∧ φ ≤ Real.pi →
    (ρ, θ, φ) = (4, 2 * Real.pi - Real.arcsin (Real.sqrt 6 / 3), Real.pi / 3) :=
by
  intros x y z ρ θ φ H Hρ Hφ Hθ1 Hθ2 Hconditions
  sorry

end convert_rectangular_to_spherical_l483_48321


namespace three_digit_solutions_l483_48314

def three_digit_number (n a x y z : ℕ) : Prop :=
  n = 100 * x + 10 * y + z ∧
  1 ≤ x ∧ x < 10 ∧ 
  0 ≤ y ∧ y < 10 ∧ 
  0 ≤ z ∧ z < 10 ∧ 
  n + (x + y + z) = 111 * a

theorem three_digit_solutions (n : ℕ) (a x y z : ℕ) :
  three_digit_number n a x y z ↔ 
  n = 105 ∨ n = 324 ∨ n = 429 ∨ n = 543 ∨ 
  n = 648 ∨ n = 762 ∨ n = 867 ∨ n = 981 :=
sorry

end three_digit_solutions_l483_48314


namespace smallest_c_l483_48349

variable {f : ℝ → ℝ}

def satisfies_conditions (f : ℝ → ℝ) : Prop :=
  (f 0 = 0) ∧ (f 1 = 1) ∧ (∀ x, 0 ≤ x ∧ x ≤ 1 → 0 ≤ f x) ∧ (∀ x1 x2, 0 ≤ x1 ∧ 0 ≤ x2 ∧ x1 + x2 ≤ 1 → f (x1 + x2) ≥ f x1 + f x2)

theorem smallest_c (f : ℝ → ℝ) (h : satisfies_conditions f) : (∀ x, 0 ≤ x ∧ x ≤ 1 → f x ≤ 2 * x) ∧ (∀ c, c < 2 → ∃ x, 0 < x ∧ x ≤ 1 ∧ ¬ (f x ≤ c * x)) :=
by
  sorry

end smallest_c_l483_48349


namespace slope_of_parallel_line_l483_48371

theorem slope_of_parallel_line (a b c : ℝ) (x y : ℝ) (h : 3 * x + 6 * y = -12):
  (∀ m : ℝ, (∀ (x y : ℝ), (3 * x + 6 * y = -12) → y = m * x + (-(12 / 6) / 6)) → m = -1/2) :=
sorry

end slope_of_parallel_line_l483_48371


namespace cricketer_initial_average_l483_48373

def initial_bowling_average
  (runs_for_last_5_wickets : ℝ)
  (decreased_average : ℝ)
  (final_wickets : ℝ)
  (initial_wickets : ℝ)
  (initial_average : ℝ) : Prop :=
  (initial_average * initial_wickets + runs_for_last_5_wickets) / final_wickets =
    initial_average - decreased_average

theorem cricketer_initial_average :
  initial_bowling_average 26 0.4 85 80 12 :=
by
  unfold initial_bowling_average
  sorry

end cricketer_initial_average_l483_48373


namespace polar_to_cartesian_parabola_l483_48319

theorem polar_to_cartesian_parabola (r θ : ℝ) (h : r = 1 / (1 - Real.sin θ)) :
  ∃ x y : ℝ, x^2 = 2 * y + 1 :=
by
  sorry

end polar_to_cartesian_parabola_l483_48319


namespace comm_add_comm_mul_distrib_l483_48347

variable {α : Type*} [AddCommMonoid α] [Mul α] [Distrib α]

theorem comm_add (a b : α) : a + b = b + a :=
by sorry

theorem comm_mul (a b : α) : a * b = b * a :=
by sorry

theorem distrib (a b c : α) : (a + b) * c = a * c + b * c :=
by sorry

end comm_add_comm_mul_distrib_l483_48347


namespace price_of_davids_toy_l483_48364

theorem price_of_davids_toy :
  ∀ (n : ℕ) (avg_before : ℕ) (avg_after : ℕ) (total_toys_after : ℕ), 
    n = 5 →
    avg_before = 10 →
    avg_after = 11 →
    total_toys_after = 6 →
  (total_toys_after * avg_after - n * avg_before = 16) :=
by
  intros n avg_before avg_after total_toys_after h_n h_avg_before h_avg_after h_total_toys_after
  sorry

end price_of_davids_toy_l483_48364


namespace area_of_triangle_with_given_medians_l483_48302

noncomputable def area_of_triangle (m1 m2 m3 : ℝ) : ℝ :=
sorry

theorem area_of_triangle_with_given_medians :
    area_of_triangle 3 4 5 = 8 :=
sorry

end area_of_triangle_with_given_medians_l483_48302


namespace cosine_of_angle_between_tangents_l483_48307

-- Definitions based on the conditions given in a)
def circle_eq (x y : ℝ) : Prop := x^2 - 2 * x + y^2 - 2 * y + 1 = 0
def P : ℝ × ℝ := (3, 2)

-- The main theorem to be proved
theorem cosine_of_angle_between_tangents (x y : ℝ)
  (hx : circle_eq x y) : 
  cos_angle_between_tangents := 
  sorry

end cosine_of_angle_between_tangents_l483_48307


namespace range_of_a_l483_48377

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, x ≥ 1 → (x^2 + a * x + 9) ≥ 0) : a ≥ -6 := 
sorry

end range_of_a_l483_48377


namespace inequality_for_real_numbers_l483_48361

theorem inequality_for_real_numbers (x y z : ℝ) : 
  - (3 / 2) * (x^2 + y^2 + 2 * z^2) ≤ 3 * x * y + y * z + z * x ∧ 
  3 * x * y + y * z + z * x ≤ (3 + Real.sqrt 13) / 4 * (x^2 + y^2 + 2 * z^2) :=
by
  sorry

end inequality_for_real_numbers_l483_48361


namespace inequality_holds_l483_48338

variable (a t1 t2 t3 t4 : ℝ)

theorem inequality_holds
  (a_pos : 0 < a)
  (h_a_le : a ≤ 7/9)
  (t1_pos : 0 < t1)
  (t2_pos : 0 < t2)
  (t3_pos : 0 < t3)
  (t4_pos : 0 < t4)
  (h_prod : t1 * t2 * t3 * t4 = a^4) :
  (1 / Real.sqrt (1 + t1) + 1 / Real.sqrt (1 + t2) + 1 / Real.sqrt (1 + t3) + 1 / Real.sqrt (1 + t4)) ≤ (4 / Real.sqrt (1 + a)) :=
by
  sorry 

end inequality_holds_l483_48338


namespace chess_club_boys_l483_48318

theorem chess_club_boys (G B : ℕ) 
  (h1 : G + B = 30)
  (h2 : (2 / 3) * G + (3 / 4) * B = 18) : B = 24 :=
by
  sorry

end chess_club_boys_l483_48318


namespace sum_of_positive_integers_n_l483_48336

theorem sum_of_positive_integers_n
  (n : ℕ) (h1: n > 0)
  (h2 : Nat.lcm n 100 = Nat.gcd n 100 + 300) :
  n = 350 :=
sorry

end sum_of_positive_integers_n_l483_48336


namespace find_sum_xyz_l483_48397

-- Define the problem
def system_of_equations (x y z : ℝ) : Prop :=
  x^2 + x * y + y^2 = 27 ∧
  y^2 + y * z + z^2 = 9 ∧
  z^2 + z * x + x^2 = 36

-- The main theorem to be proved
theorem find_sum_xyz (x y z : ℝ) (h : system_of_equations x y z) : 
  x * y + y * z + z * x = 18 :=
sorry

end find_sum_xyz_l483_48397


namespace fraction_given_to_son_l483_48340

theorem fraction_given_to_son : 
  ∀ (blue_apples yellow_apples total_apples remaining_apples given_apples : ℕ),
    blue_apples = 5 →
    yellow_apples = 2 * blue_apples →
    total_apples = blue_apples + yellow_apples →
    remaining_apples = 12 →
    given_apples = total_apples - remaining_apples →
    (given_apples : ℚ) / total_apples = 1 / 5 :=
by
  intros
  sorry

end fraction_given_to_son_l483_48340


namespace unique_solution_eqn_l483_48342

theorem unique_solution_eqn (a : ℝ) :
  (∃! x : ℝ, 3^(x^2 + 6 * a * x + 9 * a^2) = a * x^2 + 6 * a^2 * x + 9 * a^3 + a^2 - 4 * a + 4) ↔ (a = 1) :=
by
  sorry

end unique_solution_eqn_l483_48342


namespace pies_sold_l483_48393

-- Define the conditions in Lean
def num_cakes : ℕ := 453
def price_per_cake : ℕ := 12
def total_earnings : ℕ := 6318
def price_per_pie : ℕ := 7

-- Define the problem
theorem pies_sold (P : ℕ) (h1 : num_cakes * price_per_cake + P * price_per_pie = total_earnings) : P = 126 := 
by 
  sorry

end pies_sold_l483_48393


namespace correct_operation_result_l483_48368

variable (x : ℕ)

theorem correct_operation_result 
  (h : x / 15 = 6) : 15 * x = 1350 :=
sorry

end correct_operation_result_l483_48368


namespace johns_beef_order_l483_48370

theorem johns_beef_order (B : ℕ)
  (h1 : 8 * B + 6 * (2 * B) = 14000) :
  B = 1000 :=
by
  sorry

end johns_beef_order_l483_48370


namespace right_angled_triangle_l483_48360

theorem right_angled_triangle (a b c : ℕ) (h₀ : a = 7) (h₁ : b = 9) (h₂ : c = 13) :
  a^2 + b^2 ≠ c^2 :=
by
  sorry

end right_angled_triangle_l483_48360


namespace eggs_left_on_shelf_l483_48324

-- Define the conditions as variables in the Lean statement
variables (x y z : ℝ)

-- Define the final theorem statement
theorem eggs_left_on_shelf (hx : 0 ≤ x) (hy : 0 ≤ y) (hy1 : y ≤ 1) (hz : 0 ≤ z) :
  x * (1 - y) - z = (x - y * x) - z :=
by
  sorry

end eggs_left_on_shelf_l483_48324


namespace quadratic_roots_l483_48385

-- Definitions based on the conditions provided.
def condition1 (x y : ℝ) : Prop := x^2 - 6 * x + 9 = -(abs (y - 1))

-- The main theorem we want to prove.
theorem quadratic_roots (x y : ℝ) (h : condition1 x y) : (a : ℝ) → (a - 3) * (a - 1) = a^2 - 4 * a + 3 :=
  by sorry

end quadratic_roots_l483_48385


namespace vanessa_savings_weeks_l483_48357

-- Define the conditions as constants
def dress_cost : ℕ := 120
def initial_savings : ℕ := 25
def weekly_allowance : ℕ := 30
def weekly_arcade_spending : ℕ := 15
def weekly_snack_spending : ℕ := 5

-- The theorem statement based on the problem
theorem vanessa_savings_weeks : 
  ∃ (n : ℕ), (n * (weekly_allowance - weekly_arcade_spending - weekly_snack_spending) + initial_savings) ≥ dress_cost ∧ 
             (n - 1) * (weekly_allowance - weekly_arcade_spending - weekly_snack_spending) + initial_savings < dress_cost := by
  sorry

end vanessa_savings_weeks_l483_48357


namespace max_marks_l483_48375

theorem max_marks (M : ℝ) :
  (0.33 * M = 125 + 73) → M = 600 := by
  intro h
  sorry

end max_marks_l483_48375


namespace points_on_line_relationship_l483_48353

theorem points_on_line_relationship :
  let m := 2 * Real.sqrt 2 + 1
  let n := 4
  m < n :=
by
  sorry

end points_on_line_relationship_l483_48353


namespace minimum_value_proof_l483_48363

noncomputable def min_value (x y : ℝ) : ℝ :=
  (x^2 / (x + 2)) + (y^2 / (y + 1))

theorem minimum_value_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y = 1) :
  min_value x y = 1 / 4 :=
  sorry

end minimum_value_proof_l483_48363


namespace fraction_zero_implies_x_is_two_l483_48345

theorem fraction_zero_implies_x_is_two {x : ℝ} (hfrac : (2 - |x|) / (x + 2) = 0) (hdenom : x ≠ -2) : x = 2 :=
by
  sorry

end fraction_zero_implies_x_is_two_l483_48345


namespace sequence_first_number_l483_48343

theorem sequence_first_number (a: ℕ → ℕ) (h1: a 7 = 14) (h2: a 8 = 19) (h3: a 9 = 33) :
  (∀ n, n ≥ 2 → a (n+1) = a n + a (n-1)) → a 1 = 30 :=
by
  sorry

end sequence_first_number_l483_48343


namespace maximum_pairwise_sum_is_maximal_l483_48346

noncomputable def maximum_pairwise_sum (set_sums : List ℝ) (x y z w : ℝ) : Prop :=
  ∃ (a b c d e : ℝ), set_sums = [400, 500, 600, 700, 800, 900, x, y, z, w] ∧  
  ((2 / 5) * (400 + 500 + 600 + 700 + 800 + 900 + x + y + z + w)) = 
    (a + b + c + d + e) ∧ 
  5 * (a + b + c + d + e) - (400 + 500 + 600 + 700 + 800 + 900) = 1966.67

theorem maximum_pairwise_sum_is_maximal :
  maximum_pairwise_sum [400, 500, 600, 700, 800, 900] 1966.67 (1966.67 / 4) 
(1966.67 / 3) (1966.67 / 2) :=
sorry

end maximum_pairwise_sum_is_maximal_l483_48346


namespace tan_half_angle_product_l483_48378

theorem tan_half_angle_product (a b : ℝ) 
  (h : 7 * (Real.cos a + Real.cos b) + 6 * (Real.cos a * Real.cos b + 1) = 0) :
  ∃ x : ℝ, x = Real.tan (a / 2) * Real.tan (b / 2) ∧ (x = Real.sqrt (26 / 7) ∨ x = -Real.sqrt (26 / 7)) :=
by
  sorry

end tan_half_angle_product_l483_48378


namespace exterior_angle_regular_octagon_l483_48325

-- Definition and proof statement
theorem exterior_angle_regular_octagon :
  let n := 8 -- The number of sides of the polygon (octagon)
  let interior_angle_sum := 180 * (n - 2) 
  let interior_angle := interior_angle_sum / n
  let exterior_angle := 180 - interior_angle
  exterior_angle = 45 :=
by
  let n := 8
  let interior_angle_sum := 180 * (n - 2) 
  let interior_angle := interior_angle_sum / n
  let exterior_angle := 180 - interior_angle
  sorry

end exterior_angle_regular_octagon_l483_48325


namespace simplify_sqrt_expression_l483_48398

theorem simplify_sqrt_expression :
  ( (Real.sqrt 112 + Real.sqrt 567) / Real.sqrt 175) = 13 / 5 := by
  -- conditions for simplification
  have h1 : Real.sqrt 112 = 4 * Real.sqrt 7 := sorry
  have h2 : Real.sqrt 567 = 9 * Real.sqrt 7 := sorry
  have h3 : Real.sqrt 175 = 5 * Real.sqrt 7 := sorry
  
  -- Use the conditions to simplify the expression
  rw [h1, h2, h3]
  -- Further simplification to achieve the result 13 / 5
  sorry

end simplify_sqrt_expression_l483_48398


namespace blocks_needed_for_wall_l483_48327

theorem blocks_needed_for_wall (length height : ℕ) (block_heights block_lengths : List ℕ)
  (staggered : Bool) (even_ends : Bool)
  (h_length : length = 120)
  (h_height : height = 8)
  (h_block_heights : block_heights = [1])
  (h_block_lengths : block_lengths = [1, 2, 3])
  (h_staggered : staggered = true)
  (h_even_ends : even_ends = true) :
  ∃ (n : ℕ), n = 404 := 
sorry

end blocks_needed_for_wall_l483_48327


namespace problem_1_problem_2_l483_48396

-- Definitions of sets A and B
def A : Set ℝ := { x : ℝ | x^2 - 2 * x - 3 ≤ 0 }
def B (m : ℝ) : Set ℝ := { x : ℝ | m - 1 ≤ x ∧ x ≤ m + 1 }

-- Problem 1: Prove that if A ∩ B = [1, 3], then m = 2
theorem problem_1 (m : ℝ) (h : (A ∩ B m) = {x : ℝ | 1 ≤ x ∧ x ≤ 3}) : m = 2 :=
sorry

-- Problem 2: Prove that if A ⊆ complement ℝ B m, then m > 4 or m < -2
theorem problem_2 (m : ℝ) (h : A ⊆ { x : ℝ | x < m - 1 ∨ x > m + 1 }) : m > 4 ∨ m < -2 :=
sorry

end problem_1_problem_2_l483_48396


namespace right_triangle_leg_length_l483_48387

theorem right_triangle_leg_length (a b c : ℕ) (h₁ : a = 8) (h₂ : c = 17) (h₃ : a^2 + b^2 = c^2) : b = 15 := 
by
  sorry

end right_triangle_leg_length_l483_48387


namespace temperature_difference_correct_l483_48374

def refrigerator_temp : ℝ := 3
def freezer_temp : ℝ := -10
def temperature_difference : ℝ := refrigerator_temp - freezer_temp

theorem temperature_difference_correct : temperature_difference = 13 := 
by
  sorry

end temperature_difference_correct_l483_48374


namespace find_e1_l483_48348

-- Definitions related to the problem statement
variable (P F1 F2 : Type)
variable (cos_angle : ℝ)
variable (e1 e2 : ℝ)

-- Conditions
def cosine_angle_condition := cos_angle = 3 / 5
def eccentricity_relation := e2 = 2 * e1

-- Theorem that needs to be proved
theorem find_e1 (h_cos : cosine_angle_condition cos_angle)
                (h_ecc_rel : eccentricity_relation e1 e2) :
  e1 = Real.sqrt 10 / 5 :=
by
  sorry

end find_e1_l483_48348


namespace average_sales_l483_48312

theorem average_sales (jan feb mar apr : ℝ) (h_jan : jan = 100) (h_feb : feb = 60) (h_mar : mar = 40) (h_apr : apr = 120) : 
  (jan + feb + mar + apr) / 4 = 80 :=
by {
  sorry
}

end average_sales_l483_48312


namespace area_PQR_is_4_5_l483_48359

noncomputable def point := (ℝ × ℝ)

def P : point := (2, 1)
def Q : point := (1, 4)
def R_line (x: ℝ) : point := (x, 6 - x)

def area_triangle (A B C : point) : ℝ :=
  0.5 * abs ((A.1 * B.2 + B.1 * C.2 + C.1 * A.2) - (A.2 * B.1 + B.2 * C.1 + C.2 * A.1))

theorem area_PQR_is_4_5 (x : ℝ) (h : R_line x ∈ {p : point | p.1 + p.2 = 6}) : 
  area_triangle P Q (R_line x) = 4.5 :=
    sorry

end area_PQR_is_4_5_l483_48359


namespace order_of_abc_l483_48317

section
variables {a b c : ℝ}

def a_def : a = (1/2) * Real.log 2 := by sorry
def b_def : b = (1/4) * Real.log 16 := by sorry
def c_def : c = (1/6) * Real.log 27 := by sorry

theorem order_of_abc : a < c ∧ c < b :=
by
  have ha : a = (1/2) * Real.log 2 := by sorry
  have hb : b = (1/2) * Real.log 4 := by sorry
  have hc : c = (1/2) * Real.log 3 := by sorry
  sorry
end

end order_of_abc_l483_48317


namespace rainfall_november_is_180_l483_48337

-- Defining the conditions
def daily_rainfall_first_15_days := 4 -- inches per day
def days_in_first_period := 15
def total_days_in_november := 30
def multiplier_for_second_period := 2

-- Calculation based on the problem's conditions
def total_rainfall_november := 
  (daily_rainfall_first_15_days * days_in_first_period) + 
  (multiplier_for_second_period * daily_rainfall_first_15_days * (total_days_in_november - days_in_first_period))

-- Prove that the total rainfall in November is 180 inches
theorem rainfall_november_is_180 : total_rainfall_november = 180 :=
by
  -- Proof steps (to be filled in)
  sorry

end rainfall_november_is_180_l483_48337


namespace number_of_x_values_l483_48390

theorem number_of_x_values : 
  (∃ x_values : Finset ℕ, (∀ x ∈ x_values, 10 ≤ x ∧ x < 25) ∧ x_values.card = 15) :=
by
  sorry

end number_of_x_values_l483_48390


namespace sticker_price_l483_48341

theorem sticker_price (y : ℝ) (h1 : ∀ (p : ℝ), p = 0.8 * y - 60 → p ≤ y)
  (h2 : ∀ (q : ℝ), q = 0.7 * y → q ≤ y)
  (h3 : (0.8 * y - 60) + 20 = 0.7 * y) :
  y = 400 :=
by
  sorry

end sticker_price_l483_48341


namespace remaining_trees_correct_l483_48305

def initial_oak_trees := 57
def initial_maple_trees := 43

def full_cut_oak := 13
def full_cut_maple := 8

def partial_cut_oak := 2.5
def partial_cut_maple := 1.5

def remaining_oak_trees := initial_oak_trees - full_cut_oak
def remaining_maple_trees := initial_maple_trees - full_cut_maple

def total_remaining_trees := remaining_oak_trees + remaining_maple_trees

theorem remaining_trees_correct : remaining_oak_trees = 44 ∧ remaining_maple_trees = 35 ∧ total_remaining_trees = 79 :=
by
  sorry

end remaining_trees_correct_l483_48305


namespace freddy_call_duration_l483_48388

theorem freddy_call_duration (total_cost : ℕ) (local_cost_per_minute : ℕ) (international_cost_per_minute : ℕ) (local_duration : ℕ)
  (total_cost_eq : total_cost = 1000) -- cost in cents
  (local_cost_eq : local_cost_per_minute = 5)
  (international_cost_eq : international_cost_per_minute = 25)
  (local_duration_eq : local_duration = 45) :
  (total_cost - local_duration * local_cost_per_minute) / international_cost_per_minute = 31 :=
by
  sorry

end freddy_call_duration_l483_48388


namespace part1_part2_l483_48311

def P : Set ℝ := {x | 4 / (x + 2) ≥ 1}
def S (m : ℝ) : Set ℝ := {x | 1 - m ≤ x ∧ x ≤ 1 + m}

theorem part1 (h : m = 2) : P ∪ S m = {x | -2 < x ∧ x ≤ 3} :=
  by sorry

theorem part2 (h : ∀ x, x ∈ S m → x ∈ P) : 0 ≤ m ∧ m ≤ 1 :=
  by sorry

end part1_part2_l483_48311


namespace expression_simplification_l483_48391

theorem expression_simplification (x y : ℝ) (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : 2 * x + y / 2 ≠ 0) :
  (2 * x + y / 2)⁻¹ * ((2 * x)⁻¹ + (y / 2)⁻¹) = (x * y)⁻¹ := 
sorry

end expression_simplification_l483_48391


namespace sqrt_domain_l483_48356

theorem sqrt_domain (x : ℝ) : 1 - x ≥ 0 → x ≤ 1 := by
  sorry

end sqrt_domain_l483_48356


namespace boys_cannot_score_twice_as_girls_l483_48320

theorem boys_cannot_score_twice_as_girls :
  ∀ (participants : Finset ℕ) (boys girls : ℕ) (points : ℕ → ℝ),
    participants.card = 6 →
    boys = 2 →
    girls = 4 →
    (∀ p, p ∈ participants → points p = 1 ∨ points p = 0.5 ∨ points p = 0) →
    (∀ (p q : ℕ), p ∈ participants → q ∈ participants → p ≠ q → points p + points q = 1) →
    ¬ (∃ (boys_points girls_points : ℝ), 
      (∀ b ∈ (Finset.range 2), boys_points = points b) ∧
      (∀ g ∈ (Finset.range 4), girls_points = points g) ∧
      boys_points = 2 * girls_points) :=
by
  sorry

end boys_cannot_score_twice_as_girls_l483_48320


namespace urn_problem_l483_48376

theorem urn_problem : 
  (5 / 12 * 20 / (20 + M) + 7 / 12 * M / (20 + M) = 0.62) → M = 111 :=
by
  intro h
  sorry

end urn_problem_l483_48376


namespace abs_neg_five_l483_48339

theorem abs_neg_five : abs (-5) = 5 := 
by 
  sorry

end abs_neg_five_l483_48339


namespace find_middle_number_l483_48332

theorem find_middle_number (x y z : ℤ) (h1 : x + y = 22) (h2 : x + z = 29) (h3 : y + z = 37) (h4 : x < y) (h5 : y < z) : y = 15 :=
by
  sorry

end find_middle_number_l483_48332


namespace odd_pos_4_digit_ints_div_5_no_digit_5_l483_48315

open Nat

def is_valid_digit (d : Nat) : Prop :=
  d ≠ 5

def valid_odd_4_digit_ints_count : Nat :=
  let a := 8  -- First digit possibilities: {1, 2, 3, 4, 6, 7, 8, 9}
  let bc := 9  -- Second and third digit possibilities: {0, 1, 2, 3, 4, 6, 7, 8, 9}
  let d := 4  -- Fourth digit possibilities: {1, 3, 7, 9}
  a * bc * bc * d

theorem odd_pos_4_digit_ints_div_5_no_digit_5 : valid_odd_4_digit_ints_count = 2592 := by
  sorry

end odd_pos_4_digit_ints_div_5_no_digit_5_l483_48315


namespace geometric_sequence_product_l483_48344

variable {a b c : ℝ}

theorem geometric_sequence_product (h : ∃ r : ℝ, r ≠ 0 ∧ -4 = c * r ∧ c = b * r ∧ b = a * r ∧ a = -1 * r) (hb : b < 0) : a * b * c = -8 :=
by
  sorry

end geometric_sequence_product_l483_48344


namespace find_f_2_l483_48313

variable {f : ℕ → ℤ}

-- Assume the condition given in the problem
axiom h : ∀ x : ℕ, f (x + 1) = x^2 - 1

-- Prove that f(2) = 0
theorem find_f_2 : f 2 = 0 := 
sorry

end find_f_2_l483_48313


namespace part_a_value_range_part_b_value_product_l483_48367

-- Define the polynomial 
def P (x y : ℤ) : ℤ := 2 * x^2 - 6 * x * y + 5 * y^2

-- Part (a)
theorem part_a_value_range :
  ∀ (x y : ℤ), (1 ≤ P x y) ∧ (P x y ≤ 100) → ∃ (a b : ℤ), 1 ≤ P a b ∧ P a b ≤ 100 := sorry

-- Part (b)
theorem part_b_value_product :
  ∀ (a b c d : ℤ),
    P a b = r → P c d = s → ∀ (r s : ℤ), (∃ (x y : ℤ), P x y = r) ∧ (∃ (z w : ℤ), P z w = s) → 
    ∃ (u v : ℤ), P u v = r * s := sorry

end part_a_value_range_part_b_value_product_l483_48367


namespace circle_radius_l483_48322

theorem circle_radius (r : ℝ) (x y : ℝ) :
  x = π * r^2 ∧ y = 2 * π * r ∧ x + y = 100 * π → r = 10 := 
  by
  sorry

end circle_radius_l483_48322


namespace coffee_price_increase_l483_48306

theorem coffee_price_increase (price_first_quarter price_fourth_quarter : ℕ) 
  (h_first : price_first_quarter = 40) (h_fourth : price_fourth_quarter = 60) : 
  ((price_fourth_quarter - price_first_quarter) * 100) / price_first_quarter = 50 := 
by
  -- proof would proceed here
  sorry

end coffee_price_increase_l483_48306


namespace value_of_expression_l483_48379

theorem value_of_expression : 10^2 + 10 + 1 = 111 :=
by
  sorry

end value_of_expression_l483_48379
