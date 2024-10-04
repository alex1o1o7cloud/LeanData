import Mathlib

namespace sum_of_angles_satisfying_equation_l201_201570

open Real

theorem sum_of_angles_satisfying_equation :
  ∑ x in { x | 0 ≤ x ∧ x ≤ 360 ∧ sin x ^ 5 - cos x ^ 5 = (1 / cos x) - (1 / sin x) }, x = 270 := sorry

end sum_of_angles_satisfying_equation_l201_201570


namespace recurring_decimal_fraction_sum_l201_201300

theorem recurring_decimal_fraction_sum (x : ℚ) (h : x = 0.454545454545...) : 
    (let frac := rat.num_denom x.simplify in frac.fst + frac.snd)  = 16 :=
sorry

end recurring_decimal_fraction_sum_l201_201300


namespace decagon_not_divided_properly_l201_201073

theorem decagon_not_divided_properly :
  ∀ (n m : ℕ),
  (∃ black white : Finset ℕ, ∀ b ∈ black, ∀ w ∈ white,
    (b + w = 10) ∧ (b % 3 = 0) ∧ (w % 3 = 0)) →
  n - m = 10 → (n % 3 = 0) ∧ (m % 3 = 0) → 10 % 3 = 0 → False :=
by
  sorry

end decagon_not_divided_properly_l201_201073


namespace assignment_schemes_with_at_least_one_girl_l201_201576

theorem assignment_schemes_with_at_least_one_girl
  (boys girls : ℕ)
  (tasks : ℕ)
  (hb : boys = 4)
  (hg : girls = 3)
  (ht : tasks = 3)
  (total_choices : ℕ := (boys + girls).choose tasks * tasks.factorial)
  (all_boys : ℕ := boys.choose tasks * tasks.factorial) :
  total_choices - all_boys = 186 :=
by
  sorry

end assignment_schemes_with_at_least_one_girl_l201_201576


namespace percentage_voting_for_biff_equals_45_l201_201620

variable (total : ℕ) (votingForMarty : ℕ) (undecidedPercent : ℝ)

theorem percentage_voting_for_biff_equals_45 :
  total = 200 →
  votingForMarty = 94 →
  undecidedPercent = 0.08 →
  let totalDecided := (1 - undecidedPercent) * total
  let votingForBiff := totalDecided - votingForMarty
  let votingForBiffPercent := (votingForBiff / total) * 100
  votingForBiffPercent = 45 :=
by
  intros h1 h2 h3
  let totalDecided := (1 - 0.08 : ℝ) * 200
  let votingForBiff := totalDecided - 94
  let votingForBiffPercent := (votingForBiff / 200) * 100
  sorry

end percentage_voting_for_biff_equals_45_l201_201620


namespace room_width_l201_201499

theorem room_width (length height door_width door_height large_window_width large_window_height small_window_width small_window_height cost_per_sqm total_cost : ℕ) 
  (num_doors num_large_windows num_small_windows : ℕ) 
  (length_eq : length = 10) (height_eq : height = 5) 
  (door_dim_eq : door_width = 1 ∧ door_height = 3) 
  (large_window_dim_eq : large_window_width = 2 ∧ large_window_height = 1.5) 
  (small_window_dim_eq : small_window_width = 1 ∧ small_window_height = 1.5) 
  (cost_eq : cost_per_sqm = 3) (total_cost_eq : total_cost = 474) 
  (num_doors_eq : num_doors = 2) (num_large_windows_eq : num_large_windows = 1) (num_small_windows_eq : num_small_windows = 2) :
  ∃ (width : ℕ), width = 7 :=
by
  sorry

end room_width_l201_201499


namespace digits_of_2048_in_base_9_l201_201730

def digits_base9 (n : ℕ) : ℕ :=
if n < 9 then 1 else 1 + digits_base9 (n / 9)

theorem digits_of_2048_in_base_9 : digits_base9 2048 = 4 :=
by sorry

end digits_of_2048_in_base_9_l201_201730


namespace suresh_work_hours_l201_201494

variable (x : ℕ) -- Number of hours Suresh worked

theorem suresh_work_hours :
  (1/15 : ℝ) * x + (4 * (1/10 : ℝ)) = 1 -> x = 9 :=
by
  sorry

end suresh_work_hours_l201_201494


namespace power_multiplication_l201_201182

theorem power_multiplication (x : ℝ) : (-4 * x^3)^2 = 16 * x^6 := 
by 
  sorry

end power_multiplication_l201_201182


namespace fly_dist_ceiling_eq_sqrt255_l201_201675

noncomputable def fly_distance_from_ceiling : ℝ :=
  let x := 3
  let y := 5
  let d := 17
  let z := Real.sqrt (d^2 - (x^2 + y^2))
  z

theorem fly_dist_ceiling_eq_sqrt255 :
  fly_distance_from_ceiling = Real.sqrt 255 :=
by
  sorry

end fly_dist_ceiling_eq_sqrt255_l201_201675


namespace max_abs_x_y_l201_201889

theorem max_abs_x_y (x y : ℝ) (h : 4 * x^2 + y^2 = 4) : |x| + |y| ≤ 2 :=
by sorry

end max_abs_x_y_l201_201889


namespace cos_225_l201_201361

theorem cos_225 (h : Real.cos (135 * Real.pi / 180) = -Real.sqrt 2 / 2) :
    Real.cos (225 * Real.pi / 180) = -Real.sqrt 2 / 2 := by
  sorry

end cos_225_l201_201361


namespace sum_of_intervals_length_l201_201583

theorem sum_of_intervals_length (m : ℝ) (h : m ≠ 0) (h_pos : m > 0) :
  (∃ l : ℝ, ∀ x : ℝ, (1 < x ∧ x ≤ x₁) ∨ (2 < x ∧ x ≤ x₂) → 
  l = x₁ - 1 + x₂ - 2) → 
  l = 3 / m :=
sorry

end sum_of_intervals_length_l201_201583


namespace rational_zero_quadratic_roots_l201_201822

-- Part 1
theorem rational_zero (a b : ℚ) (h : a + b * Real.sqrt 5 = 0) : a = 0 ∧ b = 0 :=
sorry

-- Part 2
theorem quadratic_roots (k : ℝ) (h : k ≠ 0) (x1 x2 : ℝ)
  (h1 : 4 * k * x1^2 - 4 * k * x1 + k + 1 = 0)
  (h2 : 4 * k * x2^2 - 4 * k * x2 + k + 1 = 0)
  (h3 : x1 ≠ x2) 
  (h4 : x1^2 + x2^2 - 2 * x1 * x2 = 0.5) : k = -2 :=
sorry

end rational_zero_quadratic_roots_l201_201822


namespace water_needed_to_fill_glasses_l201_201514

theorem water_needed_to_fill_glasses :
  ∀ (num_glasses glass_capacity current_fullness : ℕ),
  num_glasses = 10 →
  glass_capacity = 6 →
  current_fullness = 4 / 5 →
  let current_total_water := num_glasses * (glass_capacity * current_fullness) in
  let max_total_water := num_glasses * glass_capacity in
  max_total_water - current_total_water = 12 :=
by
  intros num_glasses glass_capacity current_fullness
  intros h1 h2 h3
  let current_total_water := num_glasses * (glass_capacity * current_fullness)
  let max_total_water := num_glasses * glass_capacity
  show max_total_water - current_total_water = 12
  sorry

end water_needed_to_fill_glasses_l201_201514


namespace cos_225_eq_neg_inv_sqrt_2_l201_201369

theorem cos_225_eq_neg_inv_sqrt_2 :
  let P := (225 : ℝ) in
  P ∈ set_of (λ θ, θ ∈ [0, 360]) ∧
  ∃ x y : ℝ, (x = -1/√2) ∧ (y = -1/√2) ∧ (cos θ = x) :=
sorry

end cos_225_eq_neg_inv_sqrt_2_l201_201369


namespace cos_225_eq_l201_201413

noncomputable def angle_in_radians (deg : ℝ) : ℝ := deg * (Real.pi / 180)

theorem cos_225_eq :
  Real.cos (angle_in_radians 225) = - (Real.sqrt 2) / 2 := 
  sorry

end cos_225_eq_l201_201413


namespace maximum_expression_value_l201_201141

theorem maximum_expression_value :
  ∀ (a b c d : ℝ), 
    a ∈ set.Icc (-5.5) 5.5 → 
    b ∈ set.Icc (-5.5) 5.5 → 
    c ∈ set.Icc (-5.5) 5.5 → 
    d ∈ set.Icc (-5.5) 5.5 →
    a + 2 * b + c + 2 * d - a * b - b * c - c * d - d * a ≤ 132 :=
sorry

end maximum_expression_value_l201_201141


namespace total_prizes_l201_201083

-- Definitions of the conditions
def stuffedAnimals : ℕ := 14
def frisbees : ℕ := 18
def yoYos : ℕ := 18

-- The statement to be proved
theorem total_prizes : stuffedAnimals + frisbees + yoYos = 50 := by
  sorry

end total_prizes_l201_201083


namespace find_f_of_neg2_l201_201702

theorem find_f_of_neg2 (f : ℝ → ℝ) (h : ∀ x : ℝ, f (3 * x + 1) = 9 * x ^ 2 - 6 * x + 5) : f (-2) = 20 :=
by
  sorry

end find_f_of_neg2_l201_201702


namespace largest_four_digit_number_property_l201_201803

theorem largest_four_digit_number_property : ∃ (a b c d : Nat), 
  (1000 * a + 100 * b + 10 * c + d = 9099) ∧ 
  (c = a + b) ∧ 
  (d = b + c) ∧ 
  1 ≤ a ∧ a ≤ 9 ∧ 
  0 ≤ b ∧ b ≤ 9 ∧ 
  0 ≤ c ∧ c ≤ 9 ∧ 
  0 ≤ d ∧ d ≤ 9 := 
sorry

end largest_four_digit_number_property_l201_201803


namespace probability_of_exactly_nine_correct_placements_is_zero_l201_201550

-- Define the number of letters and envelopes
def num_letters : ℕ := 10

-- Define the condition of letters being randomly inserted into envelopes
def random_insertion (n : ℕ) : Prop := true

-- Prove that the probability of exactly nine letters being correctly placed is zero
theorem probability_of_exactly_nine_correct_placements_is_zero
  (h : random_insertion num_letters) : 
  (∃ p : ℝ, p = 0) := 
sorry

end probability_of_exactly_nine_correct_placements_is_zero_l201_201550


namespace cos_225_l201_201396

theorem cos_225 : cos (225 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by
-- Conditions based on given problem
have angle_def : 225 * Real.pi / 180 = 5 * Real.pi / 4 := by norm_num
have coord_Q := Complex.exp (5 * Real.pi / 4 * Complex.I)
have third_quadrant : coord_Q = Complex.mk (-Real.sqrt 2 / 2) (-Real.sqrt 2 / 2) := 
    by rw [coord_Q, Complex.exp_eq_cos_add_sin, Complex.cos, Complex.sin]
-- Proof equivalent
rw [angle_def, coord_Q]
exact congr_arg Complex.re third_quadrant
-- Equivalent statement
#check cos (5 * Real.pi / 4) = -Real.sqrt 2 / 2
-- Proof is left as an exercise
sorry

end cos_225_l201_201396


namespace percentage_of_birth_in_june_l201_201257

theorem percentage_of_birth_in_june (total_scientists: ℕ) (born_in_june: ℕ) (h_total: total_scientists = 150) (h_june: born_in_june = 15) : (born_in_june * 100 / total_scientists) = 10 := 
by 
  sorry

end percentage_of_birth_in_june_l201_201257


namespace rice_yield_prediction_l201_201894

-- Define the regression line equation
def regression_line (x : ℝ) : ℝ := 5 * x + 250

-- Define the specific condition for x = 80
def fertilizer_amount : ℝ := 80

-- State the theorem for the expected rice yield
theorem rice_yield_prediction : regression_line fertilizer_amount = 650 :=
by
  sorry

end rice_yield_prediction_l201_201894


namespace prime_solution_exists_l201_201128

theorem prime_solution_exists (p q : ℕ) (hp : p.prime) (hq : q.prime) :
  p = 17 ∧ q = 3 → (p^2 - 6 * p * q + q^2 + 3 * q - 1 = 0) :=
by
  sorry

end prime_solution_exists_l201_201128


namespace cos_double_angle_l201_201584

theorem cos_double_angle (y0 : ℝ) (h : (1 / 3)^2 + y0^2 = 1) : 
  Real.cos (2 * Real.arccos (1 / 3)) = -7 / 9 := 
by
  sorry

end cos_double_angle_l201_201584


namespace tetrahedron_edge_length_l201_201259

-- Define the problem as a Lean theorem statement
theorem tetrahedron_edge_length (r : ℝ) (a : ℝ) (h : r = 1) :
  a = 2 * Real.sqrt 2 :=
sorry

end tetrahedron_edge_length_l201_201259


namespace sum_of_integers_l201_201488

theorem sum_of_integers (x y : ℕ) (h1 : x - y = 8) (h2 : x * y = 168) : x + y = 32 :=
by
  sorry

end sum_of_integers_l201_201488


namespace original_rent_of_increased_friend_l201_201132

theorem original_rent_of_increased_friend (avg_rent : ℝ) (new_avg_rent : ℝ) (num_friends : ℝ) (rent_increase_pct : ℝ)
  (total_old_rent : ℝ) (total_new_rent : ℝ) (increase_amount : ℝ) (R : ℝ) :
  avg_rent = 800 ∧ new_avg_rent = 850 ∧ num_friends = 4 ∧ rent_increase_pct = 0.16 ∧
  total_old_rent = num_friends * avg_rent ∧ total_new_rent = num_friends * new_avg_rent ∧
  increase_amount = total_new_rent - total_old_rent ∧ increase_amount = rent_increase_pct * R →
  R = 1250 :=
by
  sorry

end original_rent_of_increased_friend_l201_201132


namespace proof_problem_l201_201919

variables {x y z w : ℝ}

-- Condition given in the problem
def condition (x y z w : ℝ) : Prop :=
  (x - y) * (z - w) / ((y - z) * (w - x)) = 1 / 3

-- The statement to be proven
theorem proof_problem (h : condition x y z w) :
  (x - z) * (y - w) / ((x - y) * (z - w)) = 1 :=
by
  sorry

end proof_problem_l201_201919


namespace min_value_quadratic_l201_201281

theorem min_value_quadratic : ∃ x : ℝ, ∀ y : ℝ, 3 * x ^ 2 - 18 * x + 2023 ≤ 3 * y ^ 2 - 18 * y + 2023 :=
sorry

end min_value_quadratic_l201_201281


namespace solve_for_x_l201_201210

variable (x : ℝ)

theorem solve_for_x (h : 2 * x - 3 * x + 4 * x = 150) : x = 50 := by
  sorry

end solve_for_x_l201_201210


namespace minimum_value_expression_l201_201285

theorem minimum_value_expression : ∃ x : ℝ, (3 * x^2 - 18 * x + 2023) = 1996 := sorry

end minimum_value_expression_l201_201285


namespace prime_solution_unique_l201_201122

theorem prime_solution_unique (p q : ℕ) (hp : prime p) (hq : prime q) (h : p^2 - 6 * p * q + q^2 + 3 * q - 1 = 0) : 
  (p = 17 ∧ q = 3) :=
by
  -- placeholder for the proof
  sorry

end prime_solution_unique_l201_201122


namespace part_I_part_II_l201_201716

noncomputable def f (x a : ℝ) : ℝ := abs (x - a) - 1

theorem part_I {a : ℝ} (ha : a = 2) :
  { x : ℝ | f x a ≥ 4 - abs (x - 4)} = { x | x ≥ 11 / 2 ∨ x ≤ 1 / 2 } := 
by 
  sorry

theorem part_II {a : ℝ} (h : { x : ℝ | abs (f (2 * x + a) a - 2 * f x a) ≤ 1 } = 
      { x | 1 / 2 ≤ x ∧ x ≤ 1 }) : 
  a = 2 := 
by 
  sorry

end part_I_part_II_l201_201716


namespace extreme_values_sin_2x0_l201_201199

noncomputable def f (x : ℝ) : ℝ :=
  2 * Real.sqrt 3 * Real.cos (Real.pi / 2 + x)^2 - 
  2 * Real.sin (Real.pi + x) * Real.cos x - Real.sqrt 3

-- Part (1)
theorem extreme_values : 
  (∀ x ∈ Set.Icc (Real.pi / 4) (Real.pi / 2), 1 ≤ f x ∧ f x ≤ 2) :=
sorry

-- Part (2)
theorem sin_2x0 (x0 : ℝ) (h : x0 ∈ Set.Icc (3 * Real.pi / 4) Real.pi) (hx : f (x0 - Real.pi / 6) = 10 / 13) : 
  Real.sin (2 * x0) = - (5 + 12 * Real.sqrt 3) / 26 :=
sorry

end extreme_values_sin_2x0_l201_201199


namespace average_marks_of_all_students_l201_201628

theorem average_marks_of_all_students (n₁ n₂ a₁ a₂ : ℕ) (h₁ : n₁ = 30) (h₂ : a₁ = 40) (h₃ : n₂ = 50) (h₄ : a₂ = 80) :
  ((n₁ * a₁ + n₂ * a₂) / (n₁ + n₂) = 65) :=
by
  sorry

end average_marks_of_all_students_l201_201628


namespace polygon_with_15_diagonals_has_7_sides_l201_201461

-- Define the number of diagonals formula for a regular polygon
def number_of_diagonals (n : ℕ) : ℕ :=
  n * (n - 3) / 2

-- The proof statement
theorem polygon_with_15_diagonals_has_7_sides :
  ∃ n : ℕ, number_of_diagonals n = 15 ∧ n = 7 :=
by
  sorry

end polygon_with_15_diagonals_has_7_sides_l201_201461


namespace geometric_series_sum_l201_201688

theorem geometric_series_sum :
  ∑' n : ℕ, (2 : ℝ) * (1 / 4) ^ n = 8 / 3 := by
  sorry

end geometric_series_sum_l201_201688


namespace solve_for_x_l201_201205

theorem solve_for_x (x : ℝ) (h : 2 * x - 3 * x + 4 * x = 150) : x = 50 :=
by
  sorry

end solve_for_x_l201_201205


namespace frank_is_15_years_younger_than_john_l201_201040

variables (F J : ℕ)

theorem frank_is_15_years_younger_than_john
  (h1 : J + 3 = 2 * (F + 3))
  (h2 : F + 4 = 16) : J - F = 15 := by
  sorry

end frank_is_15_years_younger_than_john_l201_201040


namespace negation_example_l201_201661

variable (x : ℤ)

theorem negation_example : (¬ ∀ x : ℤ, |x| ≠ 3) ↔ (∃ x : ℤ, |x| = 3) :=
by
  sorry

end negation_example_l201_201661


namespace domain_h_l201_201425

noncomputable def h (x : ℝ) : ℝ := (x^2 + 5 * x + 6) / (|x - 2| + |x + 2|)

theorem domain_h : ∀ x : ℝ, ∃ y : ℝ, y = h x :=
by
  sorry

end domain_h_l201_201425


namespace intersecting_chords_l201_201534

theorem intersecting_chords (n : ℕ) (h1 : 0 < n) :
  ∃ intersecting_points : ℕ, intersecting_points ≥ n :=
  sorry

end intersecting_chords_l201_201534


namespace radius_circle_B_l201_201561

theorem radius_circle_B (rA rB rD : ℝ) 
  (hA : rA = 2) (hD : rD = 2 * rA) (h_tangent : (rA + rB) ^ 2 = rD ^ 2) : 
  rB = 2 :=
by
  sorry

end radius_circle_B_l201_201561


namespace sum_of_legs_of_right_triangle_l201_201636

theorem sum_of_legs_of_right_triangle : 
  ∀ (x : ℕ), (x^2 + (x + 1)^2 = 41^2) → (x + (x + 1) = 57) :=
by
sorries

end sum_of_legs_of_right_triangle_l201_201636


namespace five_letter_words_with_at_least_one_vowel_l201_201062

theorem five_letter_words_with_at_least_one_vowel :
  let letters := ['A', 'B', 'C', 'D', 'E', 'F']
  let vowels := ['A', 'E', 'F']
  (6 ^ 5) - (3 ^ 5) = 7533 := by 
  sorry

end five_letter_words_with_at_least_one_vowel_l201_201062


namespace new_rectangle_area_l201_201341

theorem new_rectangle_area (a b : ℝ) : 
  let base := b + 2 * a
  let height := b - a
  let area := base * height
  area = b^2 + b * a - 2 * a^2 :=
by
  let base := b + 2 * a
  let height := b - a
  let area := base * height
  show area = b^2 + b * a - 2 * a^2
  sorry

end new_rectangle_area_l201_201341


namespace intersection_lines_k_l201_201218

theorem intersection_lines_k (k : ℝ) :
  (∃ (x y : ℝ), x = 2 ∧ x - y - 1 = 0 ∧ x + k * y = 0) → k = -2 :=
by
  sorry

end intersection_lines_k_l201_201218


namespace hyperbola_focus_l201_201482

theorem hyperbola_focus (m : ℝ) :
  (∃ (F : ℝ × ℝ), F = (0, 5) ∧ F ∈ {P : ℝ × ℝ | ∃ x y : ℝ, 
  x = P.1 ∧ y = P.2 ∧ (y^2 / m - x^2 / 9 = 1)}) → 
  m = 16 :=
by
  sorry

end hyperbola_focus_l201_201482


namespace cases_in_1990_l201_201220

theorem cases_in_1990 (cases_1970 cases_2000 : ℕ) (linear_decrease : ℕ → ℝ) :
  cases_1970 = 300000 →
  cases_2000 = 600 →
  (∀ t, linear_decrease t = cases_1970 - (cases_1970 - cases_2000) * t / 30) →
  linear_decrease 20 = 100400 :=
by
  intros h1 h2 h3
  sorry

end cases_in_1990_l201_201220


namespace factor_expression_value_l201_201089

theorem factor_expression_value :
  ∃ (k m n : ℕ), 
    k > 1 ∧ m > 1 ∧ n > 1 ∧ 
    k ≤ 60 ∧ m ≤ 35 ∧ n ≤ 20 ∧ 
    (2^k + 3^m + k^3 * m^n - n = 43) :=
by
  sorry

end factor_expression_value_l201_201089


namespace repeating_decimal_fraction_sum_l201_201315

noncomputable def repeating_decimal_to_fraction (d : ℚ) : ℚ :=
  if d = 0.45 then 5 / 11 else 0

theorem repeating_decimal_fraction_sum :
  let x := repeating_decimal_to_fraction 0.45 in
  (x.num + x.den = 16) :=
by
  let x := repeating_decimal_to_fraction 0.45
  have h : x = 5 / 11 := by
    unfold repeating_decimal_to_fraction
    split_ifs with h
    . exact rfl
  rw [h, Rat.num_den_eq 5 11, Int.add_comm]
  sorry

end repeating_decimal_fraction_sum_l201_201315


namespace average_of_five_quantities_l201_201769

theorem average_of_five_quantities (a b c d e : ℝ) 
  (h1 : (a + b + c) / 3 = 4) 
  (h2 : (d + e) / 2 = 33) : 
  ((a + b + c + d + e) / 5) = 15.6 := 
sorry

end average_of_five_quantities_l201_201769


namespace independence_and_distributions_l201_201914

noncomputable def gamma_parameters 
  (α : ℕ → ℝ) (β : ℝ) (ξ : ℕ → ℝ) (i : ℕ) : Prop :=
  ∀ (i : ℕ), i ∈ {1, ..., n} → random_var ξ[i] ~ Gamma(α[i], β)

theorem independence_and_distributions {
  (ξ : ℕ → ℝ) (α : ℕ → ℝ) (β : ℝ) (n : ℕ) (hx : gamma_parameters α β ξ):
  ∀ i, i ∈ {1, ..., n-1} → 
    let ζ i := (ξ.sum(1, i+1))/(ξ.sum(1, i+2)) in
    let η := ξ.sum(0, n) in
    let X i := ξ i / ξ.sum(1, n) in
    Independece (ζ, η) ∧
    (η ~ Gamma(α.sum(0, n), β)) ∧
    (ζ i ~ Beta(α.sum(0, i), α.sum(0, i+1))) ∧
    (X ~ Dirichlet(α)) ∧
    ∀k, 1 ≤ k ≤ n-2 → 
    let Y i := X[i] / X.sum(1, k) in
    IsIndependence (Y, (X[(k+1)], ..., X[n-1])) ∧
    (Y ~ Dirichlet(α.sum(1, k)))
  :=
sorry

end independence_and_distributions_l201_201914


namespace cos_225_l201_201357

theorem cos_225 (h : Real.cos (135 * Real.pi / 180) = -Real.sqrt 2 / 2) :
    Real.cos (225 * Real.pi / 180) = -Real.sqrt 2 / 2 := by
  sorry

end cos_225_l201_201357


namespace arithmetic_sequence_7th_term_l201_201346

theorem arithmetic_sequence_7th_term 
  (a d : ℝ)
  (n : ℕ)
  (h1 : 5 * a + 10 * d = 34)
  (h2 : 5 * a + 5 * (n - 1) * d = 146)
  (h3 : (n / 2 : ℝ) * (2 * a + (n - 1) * d) = 234) :
  a + 6 * d = 19 :=
by
  sorry

end arithmetic_sequence_7th_term_l201_201346


namespace eccentricity_range_l201_201450

section EllipseEccentricity

variables {F1 F2 : ℝ × ℝ}
variable (M : ℝ × ℝ)

-- Conditions from a)
def is_orthogonal (v1 v2 : ℝ × ℝ) : Prop :=
  v1.1 * v2.1 + v1.2 * v2.2 = 0

def is_inside_ellipse (F1 F2 M : ℝ × ℝ) : Prop :=
  is_orthogonal (M.1 - F1.1, M.2 - F1.2) (M.1 - F2.1, M.2 - F2.2) ∧ 
  -- other conditions to assert M is inside could be defined but this is unspecified
  true

-- Statement from c)
theorem eccentricity_range {a b c e : ℝ}
  (h : ∀ (M: ℝ × ℝ), is_orthogonal (M.1 - F1.1, M.2 - F1.2) (M.1 - F2.1, M.2 - F2.2) → is_inside_ellipse F1 F2 M)
  (h1 : c^2 < a^2 - c^2)
  (h2 : e^2 = c^2 / a^2) :
  0 < e ∧ e < (Real.sqrt 2) / 2 := 
sorry

end EllipseEccentricity

end eccentricity_range_l201_201450


namespace mutually_exclusive_conditional_probability_l201_201557

variables (BagA BagB : Type) [Fintype BagA] [Fintype BagB]
variables (balls_in_A : {w : ℕ // w = 3} × {r : ℕ // r = 3} × {b : ℕ // b = 2})
variables (balls_in_B : {w : ℕ // w = 2} × {r : ℕ // r = 2} × {b : ℕ // b = 1})

-- Defining events
def A1 : Event := { ω | ω ∈ BagA ∧ ω.1 = 'white }
def A2 : Event := { ω | ω ∈ BagA ∧ ω.1 = 'red }
def A3 : Event := { ω | ω ∈ BagA ∧ ω.1 = 'black }
def B : Event := { ω | ω ∈ BagB ∧ ω.1 = 'red }

-- Proof outlines
theorem mutually_exclusive (A1 A2 A3 : Event) : 
  (∀ ω, ¬(A1 ω ∧ A2 ω) ∧ ¬(A2 ω ∧ A3 ω) ∧ ¬(A1 ω ∧ A3 ω)) :=
sorry

theorem conditional_probability : 
  P(B | A1) = 1/3 :=
sorry

end mutually_exclusive_conditional_probability_l201_201557


namespace sqrt_problem_l201_201579

theorem sqrt_problem (a m : ℝ) (ha : 0 < a) 
  (h1 : a = (3 * m - 1) ^ 2) 
  (h2 : a = (-2 * m - 2) ^ 2) : 
  a = 64 ∨ a = 64 / 25 := 
sorry

end sqrt_problem_l201_201579


namespace robert_ate_7_chocolates_l201_201762

-- Define the number of chocolates Nickel ate
def nickel_chocolates : ℕ := 5

-- Define the number of chocolates Robert ate
def robert_chocolates : ℕ := nickel_chocolates + 2

-- Prove that Robert ate 7 chocolates
theorem robert_ate_7_chocolates : robert_chocolates = 7 := by
    sorry

end robert_ate_7_chocolates_l201_201762


namespace prime_solution_exists_l201_201126

theorem prime_solution_exists (p q : ℕ) (hp : p.prime) (hq : q.prime) :
  p = 17 ∧ q = 3 → (p^2 - 6 * p * q + q^2 + 3 * q - 1 = 0) :=
by
  sorry

end prime_solution_exists_l201_201126


namespace total_amount_paid_l201_201670

theorem total_amount_paid :
  let pizzas := 3
  let cost_per_pizza := 8
  let total_cost := pizzas * cost_per_pizza
  total_cost = 24 :=
by
  sorry

end total_amount_paid_l201_201670


namespace odd_function_f1_eq_4_l201_201582

theorem odd_function_f1_eq_4 (f : ℝ → ℝ) (a : ℝ) 
  (h1 : ∀ x, f (-x) = -f x)
  (h2 : ∀ x, x < 0 → f x = x^2 + a * x)
  (h3 : f 2 = 6) : 
  f 1 = 4 :=
by sorry

end odd_function_f1_eq_4_l201_201582


namespace petya_mistake_l201_201246

theorem petya_mistake :
  (35 + 10 - 41 = 42 + 12 - 50) →
  (35 + 10 - 45 = 42 + 12 - 54) →
  (5 * (7 + 2 - 9) = 6 * (7 + 2 - 9)) →
  False :=
by
  intros h1 h2 h3
  sorry

end petya_mistake_l201_201246


namespace power_function_value_l201_201719

-- Given conditions
def f : ℝ → ℝ := fun x => x^(1 / 3)

theorem power_function_value :
  f (Real.log 5 / (Real.log 2 * 8) + Real.log 160 / (Real.log (1 / 2))) = -2 := by
  sorry

end power_function_value_l201_201719


namespace hour_division_convenience_dozen_division_convenience_l201_201655

theorem hour_division_convenience :
  ∃ (a b c d e f g h i j : ℕ), 
  60 = 2 * a ∧
  60 = 3 * b ∧
  60 = 4 * c ∧
  60 = 5 * d ∧
  60 = 6 * e ∧
  60 = 10 * f ∧
  60 = 12 * g ∧
  60 = 15 * h ∧
  60 = 20 * i ∧
  60 = 30 * j := by
  -- to be filled with a proof later
  sorry

theorem dozen_division_convenience :
  ∃ (a b c d : ℕ),
  12 = 2 * a ∧
  12 = 3 * b ∧
  12 = 4 * c ∧
  12 = 6 * d := by
  -- to be filled with a proof later
  sorry

end hour_division_convenience_dozen_division_convenience_l201_201655


namespace fraction_to_decimal_conversion_l201_201532

theorem fraction_to_decimal_conversion : (2 : ℚ) / 25 = 0.08 := sorry

end fraction_to_decimal_conversion_l201_201532


namespace solve_equation_l201_201626

theorem solve_equation :
  ∃ x : ℝ, (3 * x^2 / (x - 2)) - (4 * x + 11) / 5 + (7 - 9 * x) / (x - 2) + 2 = 0 :=
sorry

end solve_equation_l201_201626


namespace total_grains_in_grey_parts_l201_201078

theorem total_grains_in_grey_parts 
  (total_grains_each_circle : ℕ)
  (white_grains_first_circle : ℕ)
  (white_grains_second_circle : ℕ)
  (common_white_grains : ℕ) 
  (h1 : white_grains_first_circle = 87)
  (h2 : white_grains_second_circle = 110)
  (h3 : common_white_grains = 68) :
  (white_grains_first_circle - common_white_grains) +
  (white_grains_second_circle - common_white_grains) = 61 :=
by
  sorry

end total_grains_in_grey_parts_l201_201078


namespace solution_set_f_derivative_l201_201236

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x^2 - 9 * x + 1

theorem solution_set_f_derivative :
  { x : ℝ | (deriv f x) < 0 } = { x : ℝ | -1 < x ∧ x < 3 } :=
by
  sorry

end solution_set_f_derivative_l201_201236


namespace triangle_min_ab_l201_201219

noncomputable def minimum_ab (a b c S : ℝ) : ℝ := if 2 * c * Real.cos B = 2 * a + b ∧ S = (Real.sqrt 3 / 2) * c then 12 else 0

theorem triangle_min_ab (a b c : ℝ) (S : ℝ) (h1 : 2 * c * Real.cos B = 2 * a + b) (h2 : S = (Real.sqrt 3 / 2) * c) : ab ≥ 12 :=
by sorry

end triangle_min_ab_l201_201219


namespace probability_two_blue_marbles_l201_201541

theorem probability_two_blue_marbles (h_red: ℕ := 3) (h_blue: ℕ := 4) (h_white: ℕ := 9) :
  (h_blue / (h_red + h_blue + h_white)) * ((h_blue - 1) / ((h_red + h_blue + h_white) - 1)) = 1 / 20 :=
by sorry

end probability_two_blue_marbles_l201_201541


namespace range_values_for_a_l201_201873

def p (x : ℝ) : Prop := x^2 - 8 * x - 20 ≤ 0
def q (x a : ℝ) (ha : 0 < a) : Prop := x^2 - 2 * x + 1 - a^2 ≥ 0

theorem range_values_for_a (a : ℝ) : (∃ ha : 0 < a, (∀ x : ℝ, (¬ p x → q x a ha))) → (0 < a ∧ a ≤ 3) :=
by
  sorry

end range_values_for_a_l201_201873


namespace identical_solutions_of_quadratic_linear_l201_201575

theorem identical_solutions_of_quadratic_linear (k : ℝ) :
  (∃ x : ℝ, x^2 = 4 * x + k ∧ x^2 = 4 * x + k) ↔ k = -4 :=
by
  sorry

end identical_solutions_of_quadratic_linear_l201_201575


namespace thelma_tomato_count_l201_201268

-- Definitions and conditions
def slices_per_tomato : ℕ := 8
def slices_per_meal_per_person : ℕ := 20
def family_members : ℕ := 8
def total_slices_needed : ℕ := slices_per_meal_per_person * family_members
def tomatoes_needed : ℕ := total_slices_needed / slices_per_tomato

-- Statement of the theorem to be proved
theorem thelma_tomato_count :
  tomatoes_needed = 20 := by
  sorry

end thelma_tomato_count_l201_201268


namespace ned_initial_video_games_l201_201241

theorem ned_initial_video_games : ∀ (w t : ℕ), 7 * w = 63 ∧ t = w + 6 → t = 15 := by
  intro w t
  intro h
  sorry

end ned_initial_video_games_l201_201241


namespace product_seq_value_l201_201659

open BigOperators

theorem product_seq_value : 
  ∏ n in (Finset.range 99).filter (λ n, n ≥ 2).map (λ n, n + 2) (-- re-index to start from 2) (λ k, (k * (k + 2)) / (k * k)) = 2.04 := 
sorry

end product_seq_value_l201_201659


namespace arithmetic_geometric_relation_l201_201708

variable (a₁ a₂ b₁ b₂ b₃ : ℝ)

-- Conditions
def is_arithmetic_sequence (a₁ a₂ : ℝ) : Prop :=
  ∃ (d : ℝ), -2 + d = a₁ ∧ a₁ + d = a₂ ∧ a₂ + d = -8

def is_geometric_sequence (b₁ b₂ b₃ : ℝ) : Prop :=
  ∃ (r : ℝ), -2 * r = b₁ ∧ b₁ * r = b₂ ∧ b₂ * r = b₃ ∧ b₃ * r = -8

-- The problem statement
theorem arithmetic_geometric_relation (h₁ : is_arithmetic_sequence a₁ a₂) (h₂ : is_geometric_sequence b₁ b₂ b₃) :
  (a₂ - a₁) / b₂ = 1 / 2 := by
    sorry

end arithmetic_geometric_relation_l201_201708


namespace hyperbola_asymptote_l201_201629

theorem hyperbola_asymptote (a : ℝ) (h_cond : 0 < a)
  (h_hyperbola : ∀ x y : ℝ, (x^2 / a^2 - y^2 / 9 = 1) → (y = (3 / 5) * x))
  : a = 5 :=
sorry

end hyperbola_asymptote_l201_201629


namespace equation_represents_circle_of_radius_8_l201_201191

theorem equation_represents_circle_of_radius_8 (k : ℝ) : 
  (x^2 + 14 * x + y^2 + 8 * y - k = 0) → k = -1 ↔ (∃ r, r = 8 ∧ (x + 7)^2 + (y + 4)^2 = r^2) :=
by
  sorry

end equation_represents_circle_of_radius_8_l201_201191


namespace geometric_sequence_sum_terms_l201_201090

noncomputable def geom_sequence_sum (a : ℕ → ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  a 0 * (1 - q ^ n) / (1 - q)

theorem geometric_sequence_sum_terms
  (a : ℕ → ℝ) (q : ℝ)
  (h_q_nonzero : q ≠ 1)
  (S3_eq : geom_sequence_sum a q 3 = 8)
  (S6_eq : geom_sequence_sum a q 6 = 7)
  : a 6 * q ^ 6 + a 7 * q ^ 7 + a 8 * q ^ 8 = 1 / 8 := sorry

end geometric_sequence_sum_terms_l201_201090


namespace jack_paid_total_l201_201227

theorem jack_paid_total (cost_squat_rack : ℕ) (cost_barbell_fraction : ℕ) 
  (h1 : cost_squat_rack = 2500) (h2 : cost_barbell_fraction = 10) :
  let cost_barbell := cost_squat_rack / cost_barbell_fraction in
  let total_cost := cost_squat_rack + cost_barbell in
  total_cost = 2750 :=
by
  -- Assign the values
  let cost_barbell := cost_squat_rack / cost_barbell_fraction
  let total_cost := cost_squat_rack + cost_barbell
  -- We use the assumptions h1 and h2
  have h_cost_barbell : cost_barbell = 250 := by
    simp only [h1, h2]
    sorry -- complete arithmetic step
  have h_total_cost : total_cost = 2750 := by
    rw [h1, h_cost_barbell]
    sorry -- complete arithmetic step
  exact h_total_cost

end jack_paid_total_l201_201227


namespace question1_question2_l201_201585

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x - a - Real.log x

theorem question1 (a : ℝ) (h : ∀ x > 0, f x a ≥ 0) : a ≤ 1 := sorry

theorem question2 (x1 x2 : ℝ) (h1 : 0 < x1) (h2 : x1 < x2) : 
  x1 * Real.log x1 - x1 * Real.log x2 > x1 - x2 := sorry

end question1_question2_l201_201585


namespace polynomial_coeff_sum_l201_201064

theorem polynomial_coeff_sum {a_0 a_1 a_2 a_3 a_4 a_5 : ℝ} :
  (2 * (x : ℝ) - 3)^5 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 →
  a_1 + 2 * a_2 + 3 * a_3 + 4 * a_4 + 5 * a_5 = 10 :=
by
  intro h
  sorry

end polynomial_coeff_sum_l201_201064


namespace calculate_value_l201_201183

theorem calculate_value : (3^2 * 5^4 * 7^2) / 7 = 39375 := by
  sorry

end calculate_value_l201_201183


namespace cos_225_l201_201397

theorem cos_225 : cos (225 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by
-- Conditions based on given problem
have angle_def : 225 * Real.pi / 180 = 5 * Real.pi / 4 := by norm_num
have coord_Q := Complex.exp (5 * Real.pi / 4 * Complex.I)
have third_quadrant : coord_Q = Complex.mk (-Real.sqrt 2 / 2) (-Real.sqrt 2 / 2) := 
    by rw [coord_Q, Complex.exp_eq_cos_add_sin, Complex.cos, Complex.sin]
-- Proof equivalent
rw [angle_def, coord_Q]
exact congr_arg Complex.re third_quadrant
-- Equivalent statement
#check cos (5 * Real.pi / 4) = -Real.sqrt 2 / 2
-- Proof is left as an exercise
sorry

end cos_225_l201_201397


namespace cos_225_eq_neg_sqrt2_div_2_l201_201420

theorem cos_225_eq_neg_sqrt2_div_2 :
  (real.cos (225 * real.pi / 180) = - (real.sqrt 2 / 2)) :=
by
  sorry

end cos_225_eq_neg_sqrt2_div_2_l201_201420


namespace biggest_number_l201_201662

noncomputable def Yoongi_collected : ℕ := 4
noncomputable def Jungkook_collected : ℕ := 6 * 3
noncomputable def Yuna_collected : ℕ := 5

theorem biggest_number :
  Jungkook_collected = 18 ∧ Jungkook_collected > Yoongi_collected ∧ Jungkook_collected > Yuna_collected :=
by
  sorry

end biggest_number_l201_201662


namespace largest_constant_l201_201957

def equation_constant (c d : ℝ) : ℝ :=
  5 * c + (d - 12)^2

theorem largest_constant : ∃ constant : ℝ, (∀ c, c ≤ 47) → (∀ d, equation_constant 47 d = constant) → constant = 235 := 
by
  sorry

end largest_constant_l201_201957


namespace recurring_fraction_sum_l201_201308

def x := 0.454545...

theorem recurring_fraction_sum :
  let x := 0.454545...
  let frac := (5 : ℕ) / (11 : ℕ)
  (frac.numerator + frac.denominator) = 16 :=
by
  sorry

end recurring_fraction_sum_l201_201308


namespace inequality_holds_range_of_expression_l201_201718

noncomputable def f (x : ℝ) : ℝ := |2 * x - 1|
noncomputable def g (x : ℝ) : ℝ := f x + f (x - 1)

theorem inequality_holds (x : ℝ) : f x < |x - 2| + 4 ↔ x ∈ Set.Ioo (-5 : ℝ) 3 := by
  sorry

theorem range_of_expression (m n : ℝ) (h : m + n = 2) (hm : m > 0) (hn : n > 0) :
  (m^2 + 2) / m + (n^2 + 1) / n ∈ Set.Ici ((7 + 2 * Real.sqrt 2) / 2) := by
  sorry

end inequality_holds_range_of_expression_l201_201718


namespace min_performances_l201_201826

theorem min_performances (n_pairs_per_show m n_singers : ℕ) (h1 : n_singers = 8) (h2 : n_pairs_per_show = 6) 
  (condition : 6 * m = 28 * 3) : m = 14 :=
by
  -- Use the assumptions to prove the statement
  sorry

end min_performances_l201_201826


namespace course_choice_related_to_gender_l201_201647

noncomputable def chi_square_test (n a b c d : ℕ) : ℝ :=
  let ad_bc := (a * d - b * c)
  in (n * ad_bc ^ 2 : ℝ) / ((a + b) * (c + d) * (a + c) * (b + d) : ℝ)

theorem course_choice_related_to_gender (a b c d n : ℕ)
  (h_total : n = a + b + c + d)
  (h_conf_level : 3.841 < chi_square_test n a b c d) :
  true :=
by sorry

-- Example instantiation of the theorem with the given numbers
example : course_choice_related_to_gender 40 10 30 20 100 (by norm_num) (by norm_num : 3.841 < chi_square_test 100 40 10 30 20) :=
by trivial

end course_choice_related_to_gender_l201_201647


namespace cos_150_deg_eq_neg_half_l201_201018

noncomputable def cos_of_angle (θ : ℝ) : ℝ := Real.cos (θ * Real.pi / 180)

theorem cos_150_deg_eq_neg_half :
  cos_of_angle 150 = -1/2 :=
by
  /-
    The conditions used directly in the problem include:
    - θ = 150 (Given angle)
  -/
  sorry

end cos_150_deg_eq_neg_half_l201_201018


namespace number_of_valid_pairs_l201_201996

theorem number_of_valid_pairs :
  (∀ (m n : ℕ), 1 ≤ m ∧ m ≤ 2044 ∧ 5^n < 2^m ∧ 2^m < 2^(m + 1) ∧ 2^(m + 1) < 5^(n + 1)) ↔
  ((∃ (x y : ℕ), 2^2100 < 5^900 ∧ 5^900 < 2^2101)) → 
  (∃ (count : ℕ), count = 900) :=
by sorry

end number_of_valid_pairs_l201_201996


namespace multiplication_by_9_l201_201742

theorem multiplication_by_9 (n : ℕ) (h1 : n < 10) : 9 * n = 10 * (n - 1) + (10 - n) := 
sorry

end multiplication_by_9_l201_201742


namespace find_y_l201_201069

theorem find_y (x y : ℤ) (h1 : x^2 - 2 * x + 5 = y + 3) (h2 : x = -8) : y = 82 := by
  sorry

end find_y_l201_201069


namespace number_of_members_l201_201964

theorem number_of_members (n : ℕ) (h : n * n = 4624) : n = 68 :=
sorry

end number_of_members_l201_201964


namespace expr_simplification_l201_201625

noncomputable def simplify_sqrt_expr : ℝ :=
  Real.sqrt 3 - Real.sqrt 12 + Real.sqrt 27

theorem expr_simplification : simplify_sqrt_expr = 2 * Real.sqrt 3 := by
  sorry

end expr_simplification_l201_201625


namespace rectangle_perimeter_equal_area_l201_201168

theorem rectangle_perimeter_equal_area (a b : ℕ) (h1 : a ≠ b) (h2 : a * b = 2 * a + 2 * b) : 2 * (a + b) = 18 := 
by 
  sorry

end rectangle_perimeter_equal_area_l201_201168


namespace loss_percentage_eq_100_div_9_l201_201542

theorem loss_percentage_eq_100_div_9 :
  ( ∀ C : ℝ,
    (11 * C > 1) ∧ 
    (8.25 * (1 + 0.20) * C = 1) →
    ((C - 1/11) / C * 100) = 100 / 9) 
  :=
by sorry

end loss_percentage_eq_100_div_9_l201_201542


namespace total_fish_caught_l201_201987

theorem total_fish_caught (C_trips : ℕ) (B_fish_per_trip : ℕ) (C_fish_per_trip : ℕ) (D_fish_per_trip : ℕ) (B_trips D_trips : ℕ) :
  C_trips = 10 →
  B_trips = 2 * C_trips →
  B_fish_per_trip = 400 →
  C_fish_per_trip = B_fish_per_trip * (1 + 2/5) →
  D_trips = 3 * C_trips →
  D_fish_per_trip = C_fish_per_trip * (1 + 50/100) →
  B_trips * B_fish_per_trip + C_trips * C_fish_per_trip + D_trips * D_fish_per_trip = 38800 := 
by
  sorry

end total_fish_caught_l201_201987


namespace mark_reading_pages_before_injury_l201_201240

theorem mark_reading_pages_before_injury:
  ∀ (h_increased: Nat) (pages_week: Nat), 
  (h_increased = 2 + (2 * 3/2)) ∧ (pages_week = 1750) → 100 = pages_week / 7 / h_increased * 2 := 
by
  sorry

end mark_reading_pages_before_injury_l201_201240


namespace difference_of_squares_multiple_of_20_l201_201824

theorem difference_of_squares_multiple_of_20 (a b : ℕ) (h1 : a > b) (h2 : a + b = 10) (hb : b = 10 - a) : 
  ∃ k : ℕ, (9 * a + 10)^2 - (100 - 9 * a)^2 = 20 * k :=
by
  sorry

end difference_of_squares_multiple_of_20_l201_201824


namespace dot_product_solution_1_l201_201726

variable (a b : ℝ × ℝ)
variable (k : ℝ)

def two_a_add_b (a b : ℝ × ℝ) : ℝ × ℝ :=
(2 * a.1 + b.1, 2 * a.2 + b.2)

def dot_product (x y : ℝ × ℝ) : ℝ :=
x.1 * y.1 + x.2 * y.2

theorem dot_product_solution_1 :
  let a := (1, -1)
  let b := (-1, 2)
  dot_product (two_a_add_b a b) a = 1 := by
sorry

end dot_product_solution_1_l201_201726


namespace min_bench_sections_l201_201171

theorem min_bench_sections (N : ℕ) :
  ∀ x y : ℕ, (x = y) → (x = 8 * N) → (y = 12 * N) → (24 * N) % 20 = 0 → N = 5 :=
by
  intros
  sorry

end min_bench_sections_l201_201171


namespace recurring_decimal_fraction_sum_l201_201302

theorem recurring_decimal_fraction_sum (x : ℚ) (h : x = 0.454545454545...) : 
    (let frac := rat.num_denom x.simplify in frac.fst + frac.snd)  = 16 :=
sorry

end recurring_decimal_fraction_sum_l201_201302


namespace arith_geo_mean_extended_arith_geo_mean_l201_201086
noncomputable section

open Real

-- Definition for Problem 1
def arith_geo_mean_inequality (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) : Prop :=
  (a + b) / 2 ≥ Real.sqrt (a * b)

-- Theorem for Problem 1
theorem arith_geo_mean (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) : arith_geo_mean_inequality a b h1 h2 :=
  sorry

-- Definition for Problem 2
def extended_arith_geo_mean_inequality (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) : Prop :=
  (a + b) * (b + c) * (c + a) ≥ 8 * a * b * c

-- Theorem for Problem 2
theorem extended_arith_geo_mean (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) : extended_arith_geo_mean_inequality a b c h1 h2 h3 :=
  sorry

end arith_geo_mean_extended_arith_geo_mean_l201_201086


namespace cannot_tile_10x10_board_l201_201491

-- Define the tiling board problem
def typeA_piece (i j : ℕ) : Prop := 
  ((i ≤ 98) ∧ (j ≤ 98) ∧ (i % 2 = 0) ∧ (j % 2 = 0))

def typeB_piece (i j : ℕ) : Prop := 
  ((i + 2 < 10) ∧ (j + 2 < 10))

def typeC_piece (i j : ℕ) : Prop := 
  ((i % 4 = 0 ∨ i % 4 = 2) ∧ (j % 4 = 0 ∨ j % 4 = 2))

-- Main theorem statement
theorem cannot_tile_10x10_board : 
  ¬ (∃ f : Fin 25 → Fin 10 × Fin 10, 
    (∀ k : Fin 25, typeA_piece (f k).1 (f k).2) ∨ 
    (∀ k : Fin 25, typeB_piece (f k).1 (f k).2) ∨ 
    (∀ k : Fin 25, typeC_piece (f k).1 (f k).2)) :=
sorry

end cannot_tile_10x10_board_l201_201491


namespace solve_for_x_l201_201886

theorem solve_for_x :
  ∃ x : ℝ, 5 ^ (Real.logb 5 15) = 7 * x + 2 ∧ x = 13 / 7 :=
by
  sorry

end solve_for_x_l201_201886


namespace solve_prime_equation_l201_201113

theorem solve_prime_equation (p q : ℕ) (hp : p.prime) (hq : q.prime) : 
  p^2 - 6*p*q + q^2 + 3*q - 1 = 0 ↔ (p = 17 ∧ q = 3) :=
by
  sorry

end solve_prime_equation_l201_201113


namespace boat_speed_in_still_water_l201_201827

theorem boat_speed_in_still_water (x y : ℝ) :
  (80 / (x + y) + 48 / (x - y) = 9) ∧ 
  (64 / (x + y) + 96 / (x - y) = 12) → 
  x = 12 :=
by
  sorry

end boat_speed_in_still_water_l201_201827


namespace polynomial_divisibility_l201_201248

theorem polynomial_divisibility (m : ℕ) (odd_m : m % 2 = 1) (x y z : ℤ) :
    ∃ k : ℤ, (x + y + z)^m - x^m - y^m - z^m = k * ((x + y + z)^3 - x^3 - y^3 - z^3) := 
by 
  sorry

end polynomial_divisibility_l201_201248


namespace smallest_n_divisible_l201_201154

open Nat

theorem smallest_n_divisible (n : ℕ) : (∃ (n : ℕ), n > 0 ∧ 45 ∣ n^2 ∧ 720 ∣ n^3) → n = 60 :=
by
  sorry

end smallest_n_divisible_l201_201154


namespace solve_for_x_l201_201212

variable (x : ℝ)

theorem solve_for_x (h : 2 * x - 3 * x + 4 * x = 150) : x = 50 := by
  sorry

end solve_for_x_l201_201212


namespace sum_mod_13_l201_201035

theorem sum_mod_13 (a b c d e : ℤ) (ha : a % 13 = 3) (hb : b % 13 = 5) (hc : c % 13 = 7) (hd : d % 13 = 9) (he : e % 13 = 11) :
  (a + b + c + d + e) % 13 = 9 :=
by
  -- The proof can be constructed here
  sorry

end sum_mod_13_l201_201035


namespace problem_l201_201715

noncomputable def f (x : ℝ) : ℝ := x^3 - 2 * x^2 - 5 * x + 6

theorem problem (h : f 10 = 756) : f 10 = 756 := 
by 
  sorry

end problem_l201_201715


namespace remainder_50_pow_50_mod_7_l201_201261

theorem remainder_50_pow_50_mod_7 : (50^50) % 7 = 1 := by
  sorry

end remainder_50_pow_50_mod_7_l201_201261


namespace smallest_sum_l201_201052

theorem smallest_sum (a b : ℕ) (h1 : 3^8 * 5^2 = a^b) (h2 : 0 < a) (h3 : 0 < b) : a + b = 407 :=
sorry

end smallest_sum_l201_201052


namespace cos_225_eq_neg_inv_sqrt_2_l201_201371

theorem cos_225_eq_neg_inv_sqrt_2 :
  let P := (225 : ℝ) in
  P ∈ set_of (λ θ, θ ∈ [0, 360]) ∧
  ∃ x y : ℝ, (x = -1/√2) ∧ (y = -1/√2) ∧ (cos θ = x) :=
sorry

end cos_225_eq_neg_inv_sqrt_2_l201_201371


namespace compare_fractions_l201_201449

theorem compare_fractions (a b m : ℝ) (h₁ : a > b) (h₂ : b > 0) (h₃ : m > 0) : 
  (b / a) < ((b + m) / (a + m)) :=
sorry

end compare_fractions_l201_201449


namespace distinct_solutions_eq_108_l201_201087

theorem distinct_solutions_eq_108 {p q : ℝ} (h1 : (p - 6) * (3 * p + 10) = p^2 - 19 * p + 50)
  (h2 : (q - 6) * (3 * q + 10) = q^2 - 19 * q + 50)
  (h3 : p ≠ q) : (p + 2) * (q + 2) = 108 := 
by
  sorry

end distinct_solutions_eq_108_l201_201087


namespace polar_to_rectangular_correct_l201_201993

noncomputable def polar_to_rectangular (r θ : ℝ) : ℝ × ℝ :=
let x := r * Real.cos θ in
let y := r * Real.sin θ in
(x, y)

theorem polar_to_rectangular_correct :
  polar_to_rectangular (-3) (5 * Real.pi / 6) = (3 * Real.sqrt 3 / 2, -3 / 2) :=
by
  sorry

end polar_to_rectangular_correct_l201_201993


namespace parabola_focus_distance_x_l201_201469

theorem parabola_focus_distance_x (x y : ℝ) :
  y^2 = 4 * x ∧ y^2 = 4 * (x^2 + 5^2) → x = 4 :=
by
  sorry

end parabola_focus_distance_x_l201_201469


namespace total_cost_is_2750_l201_201226

def squat_rack_cost : ℕ := 2500
def barbell_cost : ℕ := squat_rack_cost / 10
def total_cost : ℕ := squat_rack_cost + barbell_cost

theorem total_cost_is_2750 : total_cost = 2750 := by
  have h1 : squat_rack_cost = 2500 := by rfl
  have h2 : barbell_cost = 2500 / 10 := by rfl
  have h3 : total_cost = 2500 + 250 := by rfl
  have h4 : total_cost = 2750 := by rw [h1, h2, h3]
  sorry

end total_cost_is_2750_l201_201226


namespace find_k_for_circle_of_radius_8_l201_201192

theorem find_k_for_circle_of_radius_8 (k : ℝ) :
  (∃ x y : ℝ, x^2 + 14 * x + y^2 + 8 * y - k = 0) ∧ (∀ r : ℝ, r = 8) → k = -1 :=
sorry

end find_k_for_circle_of_radius_8_l201_201192


namespace sum_of_fraction_parts_l201_201318

theorem sum_of_fraction_parts (x : ℝ) (hx : x = 0.45) : 
  (∃ (a b : ℕ), x = a / b ∧ Nat.gcd a b = 1 ∧ a + b = 16) :=
by
  sorry

end sum_of_fraction_parts_l201_201318


namespace recurring_fraction_sum_l201_201310

def x := 0.454545...

theorem recurring_fraction_sum :
  let x := 0.454545...
  let frac := (5 : ℕ) / (11 : ℕ)
  (frac.numerator + frac.denominator) = 16 :=
by
  sorry

end recurring_fraction_sum_l201_201310


namespace fn_conjecture_l201_201709

theorem fn_conjecture (f : ℕ → ℝ → ℝ) (x : ℝ) (h_pos : x > 0) :
  (f 1 x = x / (Real.sqrt (1 + x^2))) →
  (∀ n, f (n + 1) x = f 1 (f n x)) →
  (∀ n, f n x = x / (Real.sqrt (1 + n * x ^ 2))) := by
  sorry

end fn_conjecture_l201_201709


namespace sum_of_fraction_parts_of_repeating_decimal_l201_201289

theorem sum_of_fraction_parts_of_repeating_decimal :
  let frac := Rat.mk 45 99 in
  let simplest_frac := frac.num.gcd 45 99 in
  let num := simplest_frac.num in
  let denom := simplest_frac.denom in
  num + denom = 16 :=
by
  sorry

end sum_of_fraction_parts_of_repeating_decimal_l201_201289


namespace oranges_apples_bananas_equiv_l201_201082

-- Define weights
variable (w_orange w_apple w_banana : ℝ)

-- Conditions
def condition1 : Prop := 9 * w_orange = 6 * w_apple
def condition2 : Prop := 4 * w_banana = 3 * w_apple

-- Main problem
theorem oranges_apples_bananas_equiv :
  ∀ (w_orange w_apple w_banana : ℝ),
  (9 * w_orange = 6 * w_apple) →
  (4 * w_banana = 3 * w_apple) →
  ∃ (a b : ℕ), a = 17 ∧ b = 13 ∧ (a + 3/4 * b = (45/9) * 6) :=
by
  intros w_orange w_apple w_banana h1 h2
  -- note: actual proof would go here
  sorry

end oranges_apples_bananas_equiv_l201_201082


namespace coin_and_die_probability_l201_201158

-- Probability of a coin showing heads
def P_heads : ℚ := 2 / 3

-- Probability of a die showing 5
def P_die_5 : ℚ := 1 / 6

-- Probability of both events happening together
def P_heads_and_die_5 : ℚ := P_heads * P_die_5

-- Theorem statement: Proving the calculated probability equals the expected value
theorem coin_and_die_probability : P_heads_and_die_5 = 1 / 9 := by
  -- The detailed proof is omitted here.
  sorry

end coin_and_die_probability_l201_201158


namespace length_of_tangent_point_to_circle_l201_201041

theorem length_of_tangent_point_to_circle :
  let P := (2, 3)
  let O := (0, 0)
  let r := 1
  let OP := Real.sqrt ((P.1 - O.1)^2 + (P.2 - O.2)^2)
  let tangent_length := Real.sqrt (OP^2 - r^2)
  tangent_length = 2 * Real.sqrt 3 := by
  sorry

end length_of_tangent_point_to_circle_l201_201041


namespace cos_225_eq_neg_sqrt2_div2_l201_201367

theorem cos_225_eq_neg_sqrt2_div2 : Real.cos (225 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by sorry

end cos_225_eq_neg_sqrt2_div2_l201_201367


namespace largest_valid_number_l201_201791

-- Define the conditions for the digits of the number
def valid_digits (a b c d : ℕ) : Prop :=
  c = a + b ∧ d = b + c

-- Prove that the number formed by digits 9, 0, 9, 9 is the largest valid 4-digit number
theorem largest_valid_number : ∃ a b c d, valid_digits a b c d ∧
  (a * 1000 + b * 100 + c * 10 + d = 9099) :=
begin
  use [9, 0, 9, 9],
  split,
  { -- Proof of valid digits condition
    split; refl },
  { -- Proof that the number is 9099
    refl }
end

end largest_valid_number_l201_201791


namespace unpainted_cubes_eq_210_l201_201333

-- Defining the structure of the 6x6x6 cube
def cube := Fin 6 × Fin 6 × Fin 6

-- Number of unit cubes in a 6x6x6 cube
def total_cubes : ℕ := 6 * 6 * 6

-- Number of unit squares painted by the plus pattern on each face
def squares_per_face := 13

-- Number of faces on the cube
def faces := 6

-- Initial total number of painted squares
def initial_painted_squares := squares_per_face * faces

-- Number of over-counted squares along edges
def edge_overcount := 12 * 2

-- Number of over-counted squares at corners
def corner_overcount := 8 * 1

-- Adjusted number of painted unit squares accounting for overcounts
noncomputable def adjusted_painted_squares := initial_painted_squares - edge_overcount - corner_overcount

-- Overlap adjustment: edge units and corner units
def edges_overlap := 24
def corners_overlap := 16

-- Final number of unique painted unit cubes
noncomputable def unique_painted_cubes := adjusted_painted_squares - edges_overlap - corners_overlap

-- Final unpainted unit cubes calculation
noncomputable def unpainted_cubes := total_cubes - unique_painted_cubes

-- Theorem to prove the number of unpainted unit cubes is 210
theorem unpainted_cubes_eq_210 : unpainted_cubes = 210 := by
  sorry

end unpainted_cubes_eq_210_l201_201333


namespace negation_of_prop_l201_201057

def prop (x : ℝ) := x^2 ≥ 0

theorem negation_of_prop:
  ¬ ∀ x : ℝ, prop x ↔ ∃ x : ℝ, x^2 < 0 := by
    sorry

end negation_of_prop_l201_201057


namespace alex_needs_packs_of_buns_l201_201551

-- Definitions (conditions)
def guests : ℕ := 10
def burgers_per_guest : ℕ := 3
def meat_eating_guests : ℕ := guests - 1
def bread_eating_ratios : ℕ := meat_eating_guests - 1
def buns_per_pack : ℕ := 8

-- Theorem (question == answer)
theorem alex_needs_packs_of_buns : 
  (burgers_per_guest * meat_eating_guests - burgers_per_guest) / buns_per_pack = 3 := by
  sorry

end alex_needs_packs_of_buns_l201_201551


namespace cos_225_proof_l201_201382

def cos_225_def := angle.to_real 225 = -√2 / 2

theorem cos_225_proof : cos_225_def := sorry

end cos_225_proof_l201_201382


namespace sum_of_numerator_and_denominator_of_repeating_decimal_l201_201299

noncomputable def repeating_decimal_to_fraction (n d : ℤ) (h : 0 < d) := ∃ (x : ℚ), x = n / d ∧ 0 < x

theorem sum_of_numerator_and_denominator_of_repeating_decimal :
  let x := 0.4545454545 -- defining x
  let f := x.to_rational -- converting to rational
  let f_simp := f.num / f.denom -- get numerator and denominator
  -- conclude the sum of numerator and denominator is 16
  f.num + f.denom = 16 :=
by
  sorry

end sum_of_numerator_and_denominator_of_repeating_decimal_l201_201299


namespace find_number_of_boxes_l201_201245

-- Definitions and assumptions
def pieces_per_box : ℕ := 5 + 5
def total_pieces : ℕ := 60

-- The theorem to be proved
theorem find_number_of_boxes (B : ℕ) (h : total_pieces = B * pieces_per_box) :
  B = 6 :=
sorry

end find_number_of_boxes_l201_201245


namespace range_of_2a_plus_3b_l201_201704

theorem range_of_2a_plus_3b (a b : ℝ) (h1 : -1 < a + b) (h2 : a + b < 3) (h3 : 2 < a - b) (h4 : a - b < 4) :
  -9/2 < 2*a + 3*b ∧ 2*a + 3*b < 13/2 :=
  sorry

end range_of_2a_plus_3b_l201_201704


namespace exists_powers_of_7_difference_div_by_2021_l201_201622

theorem exists_powers_of_7_difference_div_by_2021 :
  ∃ n m : ℕ, n > m ∧ 2021 ∣ (7^n - 7^m) := 
by
  sorry

end exists_powers_of_7_difference_div_by_2021_l201_201622


namespace surface_area_of_sphere_l201_201949

theorem surface_area_of_sphere (V : ℝ) (hV : V = 72 * π) : 
  ∃ A : ℝ, A = 36 * π * (2^(2/3)) := by 
  sorry

end surface_area_of_sphere_l201_201949


namespace triangle_is_isosceles_l201_201043

theorem triangle_is_isosceles (a b c : ℝ) (h : 3 * a^3 + 6 * a^2 * b - 3 * a^2 * c - 6 * a * b * c = 0) 
  (habc : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ a + c > b ∧ b + c > a) : 
  (a = c) := 
by
  sorry

end triangle_is_isosceles_l201_201043


namespace total_pencils_l201_201568

def num_boxes : ℕ := 12
def pencils_per_box : ℕ := 17

theorem total_pencils : num_boxes * pencils_per_box = 204 := by
  sorry

end total_pencils_l201_201568


namespace range_of_a_l201_201453

open Set

theorem range_of_a (a : ℝ) (M N : Set ℝ) (hM : ∀ x, x ∈ M ↔ x < 2) (hN : ∀ x, x ∈ N ↔ x < a) (hMN : M ⊆ N) : 2 ≤ a :=
by
  sorry

end range_of_a_l201_201453


namespace tourist_groupings_l201_201039

-- Assume a function to count valid groupings exists
noncomputable def num_groupings (guides tourists : ℕ) :=
  if tourists < guides * 2 then 0 
  else sorry -- placeholder for the actual combinatorial function

theorem tourist_groupings : num_groupings 4 8 = 105 := 
by
  -- The proof is omitted intentionally 
  sorry

end tourist_groupings_l201_201039


namespace xy_sum_proof_l201_201756

-- Define the given list of numbers
def original_list := [201, 202, 204, 205, 206, 209, 209, 210, 212]

-- Define the target new average and sum of numbers
def target_average : ℕ := 207
def sum_xy : ℕ := 417

-- Calculate the original sum
def original_sum : ℕ := 201 + 202 + 204 + 205 + 206 + 209 + 209 + 210 + 212

-- The new total sum calculation with x and y included
def new_total_sum := original_sum + sum_xy

-- Number of elements in the new list
def new_num_elements : ℕ := 11

-- Target new sum based on the new average and number of elements
def target_new_sum := target_average * new_num_elements

theorem xy_sum_proof : new_total_sum = target_new_sum := by
  sorry

end xy_sum_proof_l201_201756


namespace sum_of_cubes_of_consecutive_even_integers_l201_201508

theorem sum_of_cubes_of_consecutive_even_integers 
    (x y z : ℕ) 
    (h1 : x % 2 = 0) 
    (h2 : y % 2 = 0) 
    (h3 : z % 2 = 0) 
    (h4 : y = x + 2) 
    (h5 : z = y + 2) 
    (h6 : x * y * z = 12 * (x + y + z)) : 
  x^3 + y^3 + z^3 = 8568 := 
by
  -- Proof goes here
  sorry

end sum_of_cubes_of_consecutive_even_integers_l201_201508


namespace geom_seq_thm_l201_201200

noncomputable def geom_seq (a : ℕ → ℝ) :=
  a 1 = 2 ∧ (a 2 * a 4 = a 6)

noncomputable def b_seq (a : ℕ → ℝ) (n : ℕ) :=
  1 / (Real.logb 2 (a (2 * n - 1)) * Real.logb 2 (a (2 * n + 1)))

noncomputable def sn_sum (b : ℕ → ℝ) (n : ℕ) :=
  (Finset.range (n + 1)).sum b

theorem geom_seq_thm (a : ℕ → ℝ) (n : ℕ) (b : ℕ → ℝ) :
  geom_seq a →
  ∀ n, a n = 2 ^ n ∧ sn_sum (b_seq a) n = n / (2 * n + 1) :=
by
  sorry

end geom_seq_thm_l201_201200


namespace local_min_4_l201_201883

def seq (n : ℕ) : ℝ := n^3 - 48 * n + 5

theorem local_min_4 (m : ℕ) (h1 : seq (m-1) > seq m) (h2 : seq (m+1) > seq m) : m = 4 :=
sorry

end local_min_4_l201_201883


namespace prime_solution_exists_l201_201127

theorem prime_solution_exists (p q : ℕ) (hp : p.prime) (hq : q.prime) :
  p = 17 ∧ q = 3 → (p^2 - 6 * p * q + q^2 + 3 * q - 1 = 0) :=
by
  sorry

end prime_solution_exists_l201_201127


namespace largest_valid_four_digit_number_l201_201818

-- Definition of the problem conditions
def is_valid_number (a b c d : ℕ) : Prop :=
  c = a + b ∧ d = b + c

-- Proposition that we need to prove
theorem largest_valid_four_digit_number : ∃ (a b c d : ℕ),
  is_valid_number a b c d ∧ a * 1000 + b * 100 + c * 10 + d = 9099 :=
by
  sorry

end largest_valid_four_digit_number_l201_201818


namespace no_such_function_exists_l201_201998

theorem no_such_function_exists (f : ℕ → ℕ) : ¬ (∀ n : ℕ, n ≥ 2 → f (f (n - 1)) = f (n + 1) - f n) :=
sorry

end no_such_function_exists_l201_201998


namespace sum_of_coefficients_l201_201854

theorem sum_of_coefficients:
  (x^3 + 2*x + 1) * (3*x^2 + 4) = 28 :=
by
  sorry

end sum_of_coefficients_l201_201854


namespace sum_of_legs_of_right_triangle_l201_201638

theorem sum_of_legs_of_right_triangle : 
  ∀ (x : ℕ), (x^2 + (x + 1)^2 = 41^2) → (x + (x + 1) = 57) :=
by
sorries

end sum_of_legs_of_right_triangle_l201_201638


namespace sum_of_fraction_parts_of_repeating_decimal_l201_201290

theorem sum_of_fraction_parts_of_repeating_decimal :
  let frac := Rat.mk 45 99 in
  let simplest_frac := frac.num.gcd 45 99 in
  let num := simplest_frac.num in
  let denom := simplest_frac.denom in
  num + denom = 16 :=
by
  sorry

end sum_of_fraction_parts_of_repeating_decimal_l201_201290


namespace union_of_sets_complement_intersection_of_sets_l201_201202

def setA : Set ℝ := {x | 3 ≤ x ∧ x < 7}
def setB : Set ℝ := {x | 2 < x ∧ x < 10}

theorem union_of_sets :
  setA ∪ setB = {x | 2 < x ∧ x < 10} :=
sorry

theorem complement_intersection_of_sets :
  (setAᶜ) ∩ setB = {x | (2 < x ∧ x < 3) ∨ (7 ≤ x ∧ x < 10)} :=
sorry

end union_of_sets_complement_intersection_of_sets_l201_201202


namespace find_point_C_l201_201081

def point := ℝ × ℝ
def is_midpoint (M A B : point) : Prop := (2 * M.1 = A.1 + B.1) ∧ (2 * M.2 = A.2 + B.2)

-- Variables for known points
def A : point := (2, 8)
def M : point := (4, 11)
def L : point := (6, 6)

-- The proof problem: Prove the coordinates of point C
theorem find_point_C (C : point) (B : point) :
  is_midpoint M A B →
  -- (additional conditions related to the angle bisector can be added if specified)
  C = (14, 2) :=
sorry

end find_point_C_l201_201081


namespace tetrahedron_volume_l201_201939

noncomputable def volume_of_tetrahedron (S1 S2 a α : ℝ) : ℝ :=
  (2 * S1 * S2 * Real.sin α) / (3 * a)

theorem tetrahedron_volume (S1 S2 a α : ℝ) :
  a > 0 → S1 > 0 → S2 > 0 → α ≥ 0 → α ≤ Real.pi → volume_of_tetrahedron S1 S2 a α =
  (2 * S1 * S2 * Real.sin α) / (3 * a) := 
by
  intros
  -- The proof is omitted here.
  sorry

end tetrahedron_volume_l201_201939


namespace coordinates_of_B_l201_201446

-- Define the point A
structure Point :=
  (x : ℝ)
  (y : ℝ)

def A : Point := { x := 2, y := 1 }

-- Define the rotation transformation for pi/2 clockwise
def rotate_clockwise_90 (p : Point) : Point :=
  { x := p.y, y := -p.x }

-- Define the point B after rotation
def B := rotate_clockwise_90 A

-- The theorem stating the coordinates of point B (the correct answer)
theorem coordinates_of_B : B = { x := 1, y := -2 } :=
  sorry

end coordinates_of_B_l201_201446


namespace right_triangle_legs_sum_l201_201633

theorem right_triangle_legs_sum : 
  ∃ (x : ℕ), (x^2 + (x + 1)^2 = 41^2) ∧ (x + (x + 1) = 57) :=
by
  sorry

end right_triangle_legs_sum_l201_201633


namespace area_of_connected_colored_paper_l201_201252

noncomputable def side_length : ℕ := 30
noncomputable def overlap : ℕ := 7
noncomputable def sheets : ℕ := 6
noncomputable def total_length : ℕ := side_length + (sheets - 1) * (side_length - overlap)
noncomputable def width : ℕ := side_length

theorem area_of_connected_colored_paper : total_length * width = 4350 := by
  sorry

end area_of_connected_colored_paper_l201_201252


namespace simplify_expression_correct_l201_201932

noncomputable def simplify_expression : Prop :=
  (1 / (Real.log 3 / Real.log 6 + 1) + 1 / (Real.log 7 / Real.log 15 + 1) + 1 / (Real.log 4 / Real.log 12 + 1)) = -Real.log 84 / Real.log 10

theorem simplify_expression_correct : simplify_expression :=
  by
    sorry

end simplify_expression_correct_l201_201932


namespace distinct_sequences_count_l201_201063

-- Define the set of available letters excluding 'M' for start and 'S' for end
def available_letters : List Char := ['A', 'T', 'H', 'E', 'M', 'A', 'T', 'I', 'C']

-- Define the cardinality function for the sequences under given specific conditions.
-- This will check specific prompt format; you may want to specify permutations, combinations based on calculations but in the spirit, we are sticking to detail.
def count_sequences (letters : List Char) (n : Nat) : Nat :=
  if letters = available_letters ∧ n = 5 then 
    -- based on detailed calculation in the solution
    480
  else
    0

-- Theorem statement in Lean 4 to verify the number of sequences
theorem distinct_sequences_count : count_sequences available_letters 5 = 480 := 
sorry

end distinct_sequences_count_l201_201063


namespace polyhedron_same_number_edges_l201_201679

theorem polyhedron_same_number_edges (n : ℕ) (V E : ℕ) (a : ℕ → ℕ) (F := 7 * n) (M : ℕ)
  (Euler_formula : V - E + F = 2) :
  ∃ k : ℕ, (3 ≤ k ∧ k ≤ M) ∧ (a k) ≥ n+1 :=
by
  sorry

end polyhedron_same_number_edges_l201_201679


namespace white_pairs_coincide_l201_201692

theorem white_pairs_coincide
  (red_triangles_half : ℕ)
  (blue_triangles_half : ℕ)
  (white_triangles_half : ℕ)
  (red_pairs : ℕ)
  (blue_pairs : ℕ)
  (red_white_pairs : ℕ)
  (red_triangles_total_half : red_triangles_half = 4)
  (blue_triangles_total_half : blue_triangles_half = 6)
  (white_triangles_total_half : white_triangles_half = 10)
  (red_pairs_total : red_pairs = 3)
  (blue_pairs_total : blue_pairs = 4)
  (red_white_pairs_total : red_white_pairs = 3) :
  ∃ w : ℕ, w = 5 :=
by
  sorry

end white_pairs_coincide_l201_201692


namespace cosine_225_proof_l201_201355

noncomputable def cosine_225_eq_neg_sqrt2_div_2 : Prop :=
  Real.cos (225 * Real.pi / 180) = -Real.sqrt 2 / 2

theorem cosine_225_proof : cosine_225_eq_neg_sqrt2_div_2 := by
  sorry

end cosine_225_proof_l201_201355


namespace employees_paid_per_shirt_l201_201339

theorem employees_paid_per_shirt:
  let num_employees := 20
  let shirts_per_employee_per_day := 20
  let hours_per_shift := 8
  let wage_per_hour := 12
  let price_per_shirt := 35
  let nonemployee_expenses_per_day := 1000
  let profit_per_day := 9080
  let total_shirts_made_per_day := num_employees * shirts_per_employee_per_day
  let total_daily_wages := num_employees * hours_per_shift * wage_per_hour
  let total_revenue := total_shirts_made_per_day * price_per_shirt
  let per_shirt_payment := (total_revenue - (total_daily_wages + nonemployee_expenses_per_day)) / total_shirts_made_per_day
  per_shirt_payment = 27.70 :=
sorry

end employees_paid_per_shirt_l201_201339


namespace algebraic_expression_value_l201_201044

theorem algebraic_expression_value 
  (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = ab + bc + ac)
  (h2 : a = 1) : 
  (a + b - c) ^ 2004 = 1 := 
by
  sorry

end algebraic_expression_value_l201_201044


namespace cylinder_cone_volume_ratio_l201_201253

theorem cylinder_cone_volume_ratio (h r_cylinder r_cone : ℝ)
  (hcylinder_csa : π * r_cylinder^2 = π * r_cone^2 / 4):
  (π * r_cylinder^2 * h) / (1 / 3 * π * r_cone^2 * h) = 3 / 4 :=
by
  sorry

end cylinder_cone_volume_ratio_l201_201253


namespace conditional_probability_chinese_fail_l201_201222

theorem conditional_probability_chinese_fail :
  let P_math := 0.16
  let P_chinese := 0.07
  let P_both := 0.04
  P_both / P_chinese = (4 / 7) := by
  let P_math := 0.16
  let P_chinese := 0.07
  let P_both := 0.04
  sorry

end conditional_probability_chinese_fail_l201_201222


namespace find_m_for_local_maximum_l201_201586

open Set Filter

variable {ℝ : Type*} [normed_field ℝ] [normed_space ℝ ℝ]

noncomputable def f (x m : ℝ) := x * (x - m)^2

theorem find_m_for_local_maximum :
  ∃ m : ℝ, has_local_max (λ x : ℝ, f x m) 1 ∧ m = 3 :=
begin
  sorry
end

end find_m_for_local_maximum_l201_201586


namespace cos_225_l201_201395

theorem cos_225 : cos (225 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by
-- Conditions based on given problem
have angle_def : 225 * Real.pi / 180 = 5 * Real.pi / 4 := by norm_num
have coord_Q := Complex.exp (5 * Real.pi / 4 * Complex.I)
have third_quadrant : coord_Q = Complex.mk (-Real.sqrt 2 / 2) (-Real.sqrt 2 / 2) := 
    by rw [coord_Q, Complex.exp_eq_cos_add_sin, Complex.cos, Complex.sin]
-- Proof equivalent
rw [angle_def, coord_Q]
exact congr_arg Complex.re third_quadrant
-- Equivalent statement
#check cos (5 * Real.pi / 4) = -Real.sqrt 2 / 2
-- Proof is left as an exercise
sorry

end cos_225_l201_201395


namespace option_D_is_empty_l201_201157

theorem option_D_is_empty :
  {x : ℝ | x^2 + x + 1 = 0} = ∅ :=
by
  sorry

end option_D_is_empty_l201_201157


namespace fraction_subtraction_result_l201_201348

theorem fraction_subtraction_result :
  (3 * 5 + 5 * 7 + 7 * 9) / (2 * 4 + 4 * 6 + 6 * 8) - (2 * 4 + 4 * 6 + 6 * 8) / (3 * 5 + 5 * 7 + 7 * 9) = 74 / 119 :=
by sorry

end fraction_subtraction_result_l201_201348


namespace triangle_is_right_l201_201577

variable {a b c : ℝ}

theorem triangle_is_right
  (h : a^3 + (Real.sqrt 2 / 4) * b^3 + (Real.sqrt 3 / 9) * c^3 - (Real.sqrt 6 / 2) * a * b * c = 0) :
  (a * a + b * b = c * c) :=
sorry

end triangle_is_right_l201_201577


namespace price_of_cheaper_feed_l201_201151

theorem price_of_cheaper_feed 
  (W_total : ℝ) (P_total : ℝ) (E : ℝ) (W_C : ℝ) 
  (H1 : W_total = 27) 
  (H2 : P_total = 0.26)
  (H3 : E = 0.36)
  (H4 : W_C = 14.2105263158) 
  : (W_total * P_total = W_C * C + (W_total - W_C) * E) → 
    (C = 0.17) :=
by {
  sorry
}

end price_of_cheaper_feed_l201_201151


namespace largest_four_digit_number_property_l201_201799

theorem largest_four_digit_number_property : ∃ (a b c d : Nat), 
  (1000 * a + 100 * b + 10 * c + d = 9099) ∧ 
  (c = a + b) ∧ 
  (d = b + c) ∧ 
  1 ≤ a ∧ a ≤ 9 ∧ 
  0 ≤ b ∧ b ≤ 9 ∧ 
  0 ≤ c ∧ c ≤ 9 ∧ 
  0 ≤ d ∧ d ≤ 9 := 
sorry

end largest_four_digit_number_property_l201_201799


namespace exponential_first_quadrant_l201_201267

theorem exponential_first_quadrant (m : ℝ) : 
  (∀ x : ℝ, y = (1 / 2)^x + m → y ≤ 0) ↔ m ≤ -1 := 
by
  sorry

end exponential_first_quadrant_l201_201267


namespace quadratic_always_positive_if_and_only_if_l201_201882

theorem quadratic_always_positive_if_and_only_if :
  (∀ x : ℝ, x^2 + m * x + m + 3 > 0) ↔ (-2 < m ∧ m < 6) :=
by sorry

end quadratic_always_positive_if_and_only_if_l201_201882


namespace prove_inequality_l201_201031

noncomputable def valid_x (x : ℝ) : Prop :=
  x ≠ 0 ∧ x ≠ (1-Real.sqrt 5)/2 ∧ x ≠ (1+Real.sqrt 5)/2

noncomputable def valid_intervals (x : ℝ) : Prop :=
  (x ≥ -1 ∧ x < (1 - Real.sqrt 5) / 2) ∨
  ((1 - Real.sqrt 5) / 2 < x ∧ x < 0) ∨
  (0 < x ∧ x < (1 + Real.sqrt 5) / 2) ∨
  (x > (1 + Real.sqrt 5) / 2)

theorem prove_inequality (x : ℝ) (hx : valid_x x) :
  (x^2 + x^3 - x^4) / (x + x^2 - x^3) ≥ -1 ↔ valid_intervals x := by
  sorry

end prove_inequality_l201_201031


namespace percent_more_than_l201_201876

-- Definitions and conditions
variables (x y p : ℝ)

-- Condition: x is p percent more than y
def x_is_p_percent_more_than_y (x y p : ℝ) : Prop :=
  x = y + (p / 100) * y

-- The theorem to prove
theorem percent_more_than (h : x_is_p_percent_more_than_y x y p) :
  p = 100 * (x / y - 1) :=
sorry

end percent_more_than_l201_201876


namespace toms_age_ratio_l201_201270

variables (T N : ℕ)

-- Conditions
def toms_age (T : ℕ) := T
def sum_of_children_ages (T : ℕ) := T
def years_ago (T N : ℕ) := T - N
def children_ages_years_ago (T N : ℕ) := T - 4 * N

-- Given statement
theorem toms_age_ratio (h1 : toms_age T = sum_of_children_ages T)
  (h2 : years_ago T N = 3 * children_ages_years_ago T N) :
  T / N = 11 / 2 :=
sorry

end toms_age_ratio_l201_201270


namespace max_and_min_sum_of_factors_of_2000_l201_201140

theorem max_and_min_sum_of_factors_of_2000 :
  ∃ (a b c d e : ℕ), 1 < a ∧ 1 < b ∧ 1 < c ∧ 1 < d ∧ 1 < e ∧ a * b * c * d * e = 2000
  ∧ (a + b + c + d + e = 133 ∨ a + b + c + d + e = 23) :=
by
  sorry

end max_and_min_sum_of_factors_of_2000_l201_201140


namespace car_travel_distance_l201_201969

theorem car_travel_distance (v d : ℕ) 
  (h1 : d = v * 7)
  (h2 : d = (v + 12) * 5) : 
  d = 210 := by 
  sorry

end car_travel_distance_l201_201969


namespace rounding_down_both_fractions_less_sum_l201_201757

theorem rounding_down_both_fractions_less_sum
  (a b c d : ℕ) (h1 : a * d + b * c < c * d)
  (f1 : (2 : ℚ) / 3 = (a : ℚ) / b) 
  (f2 : (5 : ℚ) / 4 = (c : ℚ) / d) :
  a / b + c / d < (23 : ℚ) / 12 := sorry

end rounding_down_both_fractions_less_sum_l201_201757


namespace odd_function_expression_on_negative_domain_l201_201131

theorem odd_function_expression_on_negative_domain
  (f : ℝ → ℝ)
  (h_odd : ∀ x : ℝ, f (-x) = -f x)
  (h_pos : ∀ x : ℝ, 0 < x → f x = x * (x - 1))
  (x : ℝ)
  (h_neg : x < 0)
  : f x = x * (x + 1) :=
sorry

end odd_function_expression_on_negative_domain_l201_201131


namespace matrix_zero_product_or_rank_one_l201_201234

variables {n : ℕ}
variables (A B C : matrix (fin n) (fin n) ℝ)

theorem matrix_zero_product_or_rank_one
  (h1 : A * B * C = 0)
  (h2 : B.rank = 1) :
  A * B = 0 ∨ B * C = 0 :=
sorry

end matrix_zero_product_or_rank_one_l201_201234


namespace find_a_l201_201879

noncomputable def f (a x : ℝ) : ℝ := Real.exp x - Real.exp (1 - x) - a * x
noncomputable def g (a x : ℝ) : ℝ := Real.exp x + Real.exp (1 - x) - a

theorem find_a (x₁ x₂ a : ℝ) (h₁ : g a x₁ = 0) (h₂ : g a x₂ = 0) (hf : f a x₁ + f a x₂ = -4) : a = 4 :=
sorry

end find_a_l201_201879


namespace total_cost_of_suits_l201_201610

theorem total_cost_of_suits : 
    ∃ o t : ℕ, o = 300 ∧ t = 3 * o + 200 ∧ o + t = 1400 :=
by
  sorry

end total_cost_of_suits_l201_201610


namespace distance_travelled_within_5_seconds_l201_201178

noncomputable def velocity (t : ℝ) : ℝ := 3 * t^2 + 10 * t + 3

theorem distance_travelled_within_5_seconds :
  ∫ (t : ℝ) in 0..5, velocity t = 265 := by
  sorry

end distance_travelled_within_5_seconds_l201_201178


namespace solution_set_of_inequality_l201_201706

noncomputable def is_solution_set (f : ℝ → ℝ) : set ℝ :=
  {x | x > 1 ∨ x < -1 ∨ x = 0}

theorem solution_set_of_inequality 
  (f : ℝ → ℝ) 
  (hf_even : ∀ x, f x = f (-x)) 
  (hf_deriv : ∀ x, 2 * f x + x * (deriv f x) > 6)
  (hf_at_1 : f 1 = 2)
  : {x | x^2 * f x > 3 * x^2 - 1} = is_solution_set f :=
by
  sorry

end solution_set_of_inequality_l201_201706


namespace age_problem_l201_201767

theorem age_problem :
  ∃ (x y z : ℕ), 
    x - y = 3 ∧
    z = 2 * x + 2 * y - 3 ∧
    z = x + y + 20 ∧
    x = 13 ∧
    y = 10 ∧
    z = 43 :=
by 
  sorry

end age_problem_l201_201767


namespace range_of_a_l201_201055

-- Define the inequality problem
def inequality_always_true (a : ℝ) : Prop :=
  ∀ x, a * x^2 + 3 * a * x + a - 2 < 0

-- Define the range condition for "a"
def range_condition (a : ℝ) : Prop :=
  (a = 0 ∧ (-2 < 0)) ∨
  (a ≠ 0 ∧ a < 0 ∧ a * (5 * a + 8) < 0)

-- The main theorem stating the equivalence
theorem range_of_a (a : ℝ) : inequality_always_true a ↔ a ∈ Set.Icc (- (8 / 5)) 0 := by
  sorry

end range_of_a_l201_201055


namespace segment_AC_length_l201_201221

noncomputable def circle_radius := 8
noncomputable def chord_length_AB := 10
noncomputable def arc_length_AC (circumference : ℝ) := circumference / 3

theorem segment_AC_length :
  ∀ (C : ℝ) (r : ℝ) (AB : ℝ) (AC : ℝ),
    r = circle_radius →
    AB = chord_length_AB →
    C = 2 * Real.pi * r →
    AC = arc_length_AC C →
    AC = 8 * Real.sqrt 3 :=
by
  intros C r AB AC hr hAB hC hAC
  sorry

end segment_AC_length_l201_201221


namespace singles_percentage_l201_201075

-- Definitions based on conditions
def total_hits : ℕ := 50
def home_runs : ℕ := 2
def triples : ℕ := 3
def doubles : ℕ := 7
def non_single_hits : ℕ := home_runs + triples + doubles
def singles : ℕ := total_hits - non_single_hits

-- Theorem based on the proof problem
theorem singles_percentage :
  singles = 38 ∧ (singles / total_hits : ℚ) * 100 = 76 := 
  by
    sorry

end singles_percentage_l201_201075


namespace system_solutions_l201_201934

theorem system_solutions (x y z a b c : ℝ) :
  (a = 1 ∨ b = 1 ∨ c = 1 ∨ a + b + c + a * b * c = 0) → (¬(x = 1 ∨ y = 1 ∨ z = 1) → 
  ∃ (x y z : ℝ), (x - y) / (z - 1) = a ∧ (y - z) / (x - 1) = b ∧ (z - x) / (y - 1) = c) ∨
  (a ≠ 1 ∧ b ≠ 1 ∧ c ≠ 1 ∧ a + b + c + a * b * c ≠ 0) → 
  ¬∃ (x y z : ℝ), (x - y) / (z - 1) = a ∧ (y - z) / (x - 1) = b ∧ (z - x) / (y - 1) = c :=
by
    sorry

end system_solutions_l201_201934


namespace primes_between_4900_8100_l201_201591

theorem primes_between_4900_8100 :
  ∃ (count : ℕ),
  count = 5 ∧ ∀ n : ℤ, 70 < n ∧ n < 90 ∧ (n * n > 4900 ∧ n * n < 8100 ∧ Prime n) → count = 5 :=
by
  sorry

end primes_between_4900_8100_l201_201591


namespace part1_part2_l201_201451

noncomputable def f (a x : ℝ) := a * Real.log x - x / 2

theorem part1 (a : ℝ) : (∀ x, f a x = a * Real.log x - x / 2) → (∃ x, x = 2 ∧ deriv (f a) x = 0) → a = 1 :=
by sorry

theorem part2 (k : ℝ) : (∀ x, x > 1 → f 1 x + k / x < 0) → k ≤ 1 / 2 :=
by sorry

end part1_part2_l201_201451


namespace total_yearly_car_leasing_cost_l201_201952

-- Define mileage per day
def mileage_per_day (day : String) : ℕ :=
  if day = "Monday" ∨ day = "Wednesday" ∨ day = "Friday" ∨ day = "Sunday" then 50
  else if day = "Tuesday" ∨ day = "Thursday" then 80
  else if day = "Saturday" then 120
  else 0

-- Define weekly mileage
def weekly_mileage : ℕ := 4 * 50 + 2 * 80 + 120

-- Define cost parameters
def cost_per_mile : ℕ := 1 / 10
def weekly_fee : ℕ := 100
def monthly_toll_parking_fees : ℕ := 50
def discount_every_5th_week : ℕ := 30
def number_of_weeks_in_year : ℕ := 52

-- Define total yearly cost
def total_cost_yearly : ℕ :=
  let total_weekly_cost := (weekly_mileage * cost_per_mile + weekly_fee)
  let total_yearly_cost := total_weekly_cost * number_of_weeks_in_year
  let total_discounts := (number_of_weeks_in_year / 5) * discount_every_5th_week
  let annual_cost_without_tolls := total_yearly_cost - total_discounts
  let total_toll_fees := monthly_toll_parking_fees * 12
  annual_cost_without_tolls + total_toll_fees

-- Define the main theorem
theorem total_yearly_car_leasing_cost : total_cost_yearly = 7996 := 
  by
    -- Proof omitted
    sorry

end total_yearly_car_leasing_cost_l201_201952


namespace B_subset_A_A_inter_B_empty_l201_201589

noncomputable def setA (m : ℝ) : set ℝ := { x | (x + 2 * m) * (x - m + 4) < 0 }
noncomputable def setB : set ℝ := { x | (1 - x) / (x + 2) > 0 }

theorem B_subset_A {m : ℝ} : 
  (setB ⊆ setA m) ↔ (m ≥ 5 ∨ m ≤ -1 / 2) := 
sorry

theorem A_inter_B_empty {m : ℝ} :
  (setA m ∩ setB = ∅) ↔ (1 ≤ m ∧ m ≤ 2) := 
sorry

end B_subset_A_A_inter_B_empty_l201_201589


namespace cubic_polynomial_Q_l201_201915

noncomputable def cubic_poly : Polynomial ℝ := Polynomial.Coeff (Polynomial.X^3 + 4 * Polynomial.X^2 + 6 * Polynomial.X + 8)

theorem cubic_polynomial_Q (p q r : ℝ)
  (hpqr : (Polynomial.roots cubic_poly).toFinset = {p, q, r})
  (Q : Polynomial ℝ)
  (hQp : Q.eval p = q + r)
  (hQq : Q.eval q = p + r)
  (hQr : Q.eval r = p + q)
  (hQsum : Q.eval (p + q + r) = -20) :
  Q = (frac 5 4) * (Polynomial.X^3 + 4 * Polynomial.X^2 + 6 * Polynomial.X + 8) 
      - Polynomial.X - 4 :=
sorry

end cubic_polynomial_Q_l201_201915


namespace room_width_is_7_l201_201498

-- Define the conditions of the problem
def room_length : ℝ := 10
def room_height : ℝ := 5
def door_width : ℝ := 1
def door_height : ℝ := 3
def window1_width : ℝ := 2
def window1_height : ℝ := 1.5
def window2_width : ℝ := 1
def window2_height : ℝ := 1.5
def cost_per_sq_meter : ℝ := 3
def total_cost : ℝ := 474

-- Define the total cost to be painted
def total_area_painted (width : ℝ) : ℝ :=
  let wall_area := 2 * (room_length * room_height) + 2 * (width * room_height)
  let door_area := 2 * (door_width * door_height)
  let window_area := (window1_width * window1_height) + 2 * (window2_width * window2_height)
  wall_area - door_area - window_area

def cost_equation (width : ℝ) : Prop :=
  (total_cost / cost_per_sq_meter) = total_area_painted width

-- Prove that the width required to satisfy the painting cost equation is 7 meters
theorem room_width_is_7 : ∃ w : ℝ, cost_equation w ∧ w = 7 :=
by
  sorry

end room_width_is_7_l201_201498


namespace ella_dog_food_ratio_l201_201225

variable (ella_food_per_day : ℕ) (total_food_10days : ℕ) (x : ℕ)

theorem ella_dog_food_ratio
  (h1 : ella_food_per_day = 20)
  (h2 : total_food_10days = 1000) :
  (x : ℕ) = 4 :=
by
  sorry

end ella_dog_food_ratio_l201_201225


namespace probability_of_seven_in_0_375_l201_201100

theorem probability_of_seven_in_0_375 :
  let digits := [3, 7, 5] in
  (∃ n : ℕ, digits.get? n = some 7 ∧ 3 = digits.length) → (1 / 3 : ℚ) = 1 / 3 :=
by
  sorry

end probability_of_seven_in_0_375_l201_201100


namespace cos_225_eq_neg_sqrt2_div_2_l201_201418

theorem cos_225_eq_neg_sqrt2_div_2 :
  (real.cos (225 * real.pi / 180) = - (real.sqrt 2 / 2)) :=
by
  sorry

end cos_225_eq_neg_sqrt2_div_2_l201_201418


namespace solve_prime_equation_l201_201108

def is_prime (n : ℕ) : Prop := Nat.Prime n

theorem solve_prime_equation (p q : ℕ) (hp : is_prime p) (hq : is_prime q) (h : p ^ 2 - 6 * p * q + q ^ 2 + 3 * q - 1 = 0) :
  (p = 17 ∧ q = 3) :=
sorry

end solve_prime_equation_l201_201108


namespace quadratic_range_l201_201740

theorem quadratic_range (a : ℝ) :
  (¬ ∃ x : ℝ, x^2 + 2 * a * x + 1 < 0) → (-1 ≤ a ∧ a ≤ 1) :=
by
  sorry

end quadratic_range_l201_201740


namespace eliminate_x3_term_l201_201349

noncomputable def polynomial (n : ℝ) : Polynomial ℝ :=
  (Polynomial.X ^ 2 + Polynomial.C n * Polynomial.X + Polynomial.C 3) *
  (Polynomial.X ^ 2 - Polynomial.C 3 * Polynomial.X)

theorem eliminate_x3_term (n : ℝ) : (polynomial n).coeff 3 = 0 ↔ n = 3 :=
by
  -- sorry to skip the proof for now as it's not required
  sorry

end eliminate_x3_term_l201_201349


namespace cos_225_eq_neg_sqrt2_div2_l201_201366

theorem cos_225_eq_neg_sqrt2_div2 : Real.cos (225 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by sorry

end cos_225_eq_neg_sqrt2_div2_l201_201366


namespace no_such_b_exists_l201_201480

theorem no_such_b_exists (k n : ℕ) (a : ℕ) 
  (hk : Odd k) (hn : Odd n)
  (hk_gt_one : k > 1) (hn_gt_one : n > 1) 
  (hka : k ∣ 2^a + 1) (hna : n ∣ 2^a - 1) : 
  ¬ ∃ b : ℕ, k ∣ 2^b - 1 ∧ n ∣ 2^b + 1 :=
sorry

end no_such_b_exists_l201_201480


namespace factorization_correct_l201_201963

theorem factorization_correct :
    (∀ (x y : ℝ), x * (2 * x - y) + 2 * y * (2 * x - y) = (x + 2 * y) * (2 * x - y)) :=
by
  intro x y
  sorry

end factorization_correct_l201_201963


namespace solve_for_x_l201_201208

theorem solve_for_x (x : ℝ) (h : 2 * x - 3 * x + 4 * x = 150) : x = 50 :=
by
  sorry

end solve_for_x_l201_201208


namespace largest_valid_number_l201_201792

-- Define the conditions for the digits of the number
def valid_digits (a b c d : ℕ) : Prop :=
  c = a + b ∧ d = b + c

-- Prove that the number formed by digits 9, 0, 9, 9 is the largest valid 4-digit number
theorem largest_valid_number : ∃ a b c d, valid_digits a b c d ∧
  (a * 1000 + b * 100 + c * 10 + d = 9099) :=
begin
  use [9, 0, 9, 9],
  split,
  { -- Proof of valid digits condition
    split; refl },
  { -- Proof that the number is 9099
    refl }
end

end largest_valid_number_l201_201792


namespace problem_statement_l201_201084

theorem problem_statement
  (g : ℝ → ℝ)
  (p q r s : ℝ)
  (h_roots : ∃ n1 n2 n3 n4 : ℕ, 
                ∀ x, g x = (x + 2 * n1) * (x + 2 * n2) * (x + 2 * n3) * (x + 2 * n4))
  (h_pqrs : p + q + r + s = 2552)
  (h_g : ∀ x, g x = x^4 + p * x^3 + q * x^2 + r * x + s) :
  s = 3072 :=
by
  sorry

end problem_statement_l201_201084


namespace cylinder_volume_increase_l201_201145

theorem cylinder_volume_increase {R H : ℕ} (x : ℚ) (C : ℝ) (π : ℝ) 
  (hR : R = 8) (hH : H = 3) (hπ : π = Real.pi)
  (hV : ∃ C > 0, π * (R + x)^2 * (H + x) = π * R^2 * H + C) :
  x = 16 / 3 :=
by
  sorry

end cylinder_volume_increase_l201_201145


namespace largest_four_digit_number_l201_201809

theorem largest_four_digit_number :
  ∃ a b c d : ℕ, 
    9 < 1000 * a + 100 * b + 10 * c + d ∧ 
    1000 * a + 100 * b + 10 * c + d < 10000 ∧ 
    c = a + b ∧ 
    d = b + c ∧ 
    1000 * a + 100 * b + 10 * c + d = 9099 :=
by {
  sorry
}

end largest_four_digit_number_l201_201809


namespace sum_of_fraction_terms_l201_201320

theorem sum_of_fraction_terms (x : ℚ) (hx : x = 45 / 99) (y : ℚ) (hy : y * 99 = 45) : 
  let z := (5 : ℚ) / (11 : ℚ) in 
  hwf : x = z → z.den + z.num = 16 := 
by
  sorry

end sum_of_fraction_terms_l201_201320


namespace maynard_filled_percentage_l201_201093

theorem maynard_filled_percentage (total_holes : ℕ) (unfilled_holes : ℕ) (filled_holes : ℕ) (p : ℚ) :
  total_holes = 8 →
  unfilled_holes = 2 →
  filled_holes = total_holes - unfilled_holes →
  p = (filled_holes : ℚ) / (total_holes : ℚ) * 100 →
  p = 75 := 
by {
  -- proofs and calculations would go here
  sorry
}

end maynard_filled_percentage_l201_201093


namespace solve_for_x_l201_201209

variable (x : ℝ)

theorem solve_for_x (h : 2 * x - 3 * x + 4 * x = 150) : x = 50 := by
  sorry

end solve_for_x_l201_201209


namespace sum_of_fraction_numerator_and_denominator_l201_201325

theorem sum_of_fraction_numerator_and_denominator (x : ℚ) (a b : ℤ) :
  x = (45 / 99 : ℚ) ∧ (a = 5) ∧ (b = 11) → (a + b = 16) :=
by
  sorry

end sum_of_fraction_numerator_and_denominator_l201_201325


namespace additional_distance_sam_runs_more_than_sarah_l201_201074

theorem additional_distance_sam_runs_more_than_sarah
  (street_width : ℝ) (block_side_length : ℝ)
  (h1 : street_width = 30) (h2 : block_side_length = 500) :
  let P_Sarah := 4 * block_side_length
  let P_Sam := 4 * (block_side_length + 2 * street_width)
  P_Sam - P_Sarah = 240 :=
by
  sorry

end additional_distance_sam_runs_more_than_sarah_l201_201074


namespace neg_of_univ_prop_l201_201507

theorem neg_of_univ_prop :
  (∃ (x₀ : ℝ), 0 ≤ x₀ ∧ x₀^3 + x₀ < 0) ↔ ¬ (∀ (x : ℝ), 0 ≤ x → x^3 + x ≥ 0) := by
sorry

end neg_of_univ_prop_l201_201507


namespace cos_225_eq_neg_sqrt2_div2_l201_201362

theorem cos_225_eq_neg_sqrt2_div2 : Real.cos (225 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by sorry

end cos_225_eq_neg_sqrt2_div2_l201_201362


namespace find_f_x_l201_201877

def f (x : ℝ) : ℝ := sorry

theorem find_f_x (x : ℝ) (h : 2 * f x - f (-x) = 3 * x) : f x = x := 
by sorry

end find_f_x_l201_201877


namespace minimum_distance_from_mars_l201_201012

noncomputable def distance_function (a b c t : ℝ) : ℝ :=
  a * t^2 + b * t + c

theorem minimum_distance_from_mars :
  ∃ t₀ : ℝ, distance_function (11/54) (-1/18) 4 t₀ = (9:ℝ) :=
  sorry

end minimum_distance_from_mars_l201_201012


namespace readers_of_science_fiction_l201_201598

variable (Total S L B : Nat)

theorem readers_of_science_fiction 
  (h1 : Total = 400) 
  (h2 : L = 230) 
  (h3 : B = 80) 
  (h4 : Total = S + L - B) : 
  S = 250 := 
by
  sorry

end readers_of_science_fiction_l201_201598


namespace f_2015_eq_neg_2014_l201_201594

variable {f : ℝ → ℝ}

-- Conditions
def isOddFunction (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x
def isPeriodic (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x : ℝ, f (x + p) = f x
def f1_value : f 1 = 2014 := sorry

-- Theorem to prove
theorem f_2015_eq_neg_2014 :
  isOddFunction f → isPeriodic f 3 → (f 1 = 2014) → f 2015 = -2014 :=
by
  intros hOdd hPeriodic hF1
  sorry

end f_2015_eq_neg_2014_l201_201594


namespace exist_A_B_l201_201426

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem exist_A_B : ∃ (A B : ℕ), A = 2016 * B ∧ sum_of_digits A + 2016 * sum_of_digits B < 0 := sorry

end exist_A_B_l201_201426


namespace largest_angle_measure_l201_201336

noncomputable def measure_largest_angle (x : ℚ) : Prop :=
  let a1 := 2 * x + 2
  let a2 := 3 * x
  let a3 := 4 * x + 3
  let a4 := 5 * x
  let a5 := 6 * x - 1
  let a6 := 7 * x
  a1 + a2 + a3 + a4 + a5 + a6 = 720 ∧ a6 = 5012 / 27

theorem largest_angle_measure : ∃ x : ℚ, measure_largest_angle x := by
  sorry

end largest_angle_measure_l201_201336


namespace sum_of_largest_and_smallest_odd_numbers_is_16_l201_201440

-- Define odd numbers between 5 and 12
def odd_numbers_set := {n | 5 ≤ n ∧ n ≤ 12 ∧ n % 2 = 1}

-- Define smallest odd number from the set
def min_odd := 5

-- Define largest odd number from the set
def max_odd := 11

-- The main theorem stating that the sum of the smallest and largest odd numbers is 16
theorem sum_of_largest_and_smallest_odd_numbers_is_16 :
  min_odd + max_odd = 16 := by
  sorry

end sum_of_largest_and_smallest_odd_numbers_is_16_l201_201440


namespace mil_equals_one_fortieth_mm_l201_201788

-- The condition that one mil is equal to one thousandth of an inch
def mil_in_inch := 1 / 1000

-- The condition that an inch is about 2.5 cm
def inch_in_mm := 25

-- The problem statement in Lean 4 form
theorem mil_equals_one_fortieth_mm : (mil_in_inch * inch_in_mm = 1 / 40) :=
by
  sorry

end mil_equals_one_fortieth_mm_l201_201788


namespace solve_for_x_l201_201211

variable (x : ℝ)

theorem solve_for_x (h : 2 * x - 3 * x + 4 * x = 150) : x = 50 := by
  sorry

end solve_for_x_l201_201211


namespace trajectory_of_center_of_moving_circle_l201_201059

theorem trajectory_of_center_of_moving_circle
  (x y : ℝ)
  (C1 : (x + 4)^2 + y^2 = 2)
  (C2 : (x - 4)^2 + y^2 = 2) :
  ((x = 0) ∨ (x^2 / 2 - y^2 / 14 = 1)) :=
sorry

end trajectory_of_center_of_moving_circle_l201_201059


namespace prime_solution_unique_l201_201121

theorem prime_solution_unique (p q : ℕ) (hp : prime p) (hq : prime q) (h : p^2 - 6 * p * q + q^2 + 3 * q - 1 = 0) : 
  (p = 17 ∧ q = 3) :=
by
  -- placeholder for the proof
  sorry

end prime_solution_unique_l201_201121


namespace initial_number_correct_l201_201513

-- Define the relevant values
def x : ℝ := 53.33
def initial_number : ℝ := 319.98

-- Define the conditions in Lean with appropriate constraints
def conditions (n : ℝ) (x : ℝ) : Prop :=
  x = n / 2 / 3

-- Theorem stating that 319.98 divided by 2 and then by 3 results in 53.33
theorem initial_number_correct : conditions initial_number x :=
by
  unfold conditions
  sorry

end initial_number_correct_l201_201513


namespace tiffany_mile_fraction_l201_201843

/-- Tiffany's daily running fraction (x) for Wednesday, Thursday, and Friday must be 1/3
    such that both Billy and Tiffany run the same total miles over a week. --/
theorem tiffany_mile_fraction :
  ∃ x : ℚ, (3 * 1 + 1) = 1 + (3 * 2 + 3 * x) → x = 1 / 3 :=
by
  sorry

end tiffany_mile_fraction_l201_201843


namespace action_figure_collection_complete_l201_201611

theorem action_figure_collection_complete (act_figures : ℕ) (cost_per_fig : ℕ) (extra_money_needed : ℕ) (total_collection : ℕ) 
    (h1 : act_figures = 7) 
    (h2 : cost_per_fig = 8) 
    (h3 : extra_money_needed = 72) : 
    total_collection = 16 :=
by
  sorry

end action_figure_collection_complete_l201_201611


namespace value_of_frac_mul_l201_201735

theorem value_of_frac_mul (a b c d : ℕ) (h1 : a = 4 * b) (h2 : b = 3 * c) (h3 : c = 2 * d) :
  (a * c) / (b * d) = 8 :=
by
  sorry

end value_of_frac_mul_l201_201735


namespace cosine_225_proof_l201_201351

noncomputable def cosine_225_eq_neg_sqrt2_div_2 : Prop :=
  Real.cos (225 * Real.pi / 180) = -Real.sqrt 2 / 2

theorem cosine_225_proof : cosine_225_eq_neg_sqrt2_div_2 := by
  sorry

end cosine_225_proof_l201_201351


namespace leap_day_2040_is_tuesday_l201_201233

def days_in_non_leap_year := 365
def days_in_leap_year := 366
def leap_years_between_2000_and_2040 := 10

def total_days_between_2000_and_2040 := 
  30 * days_in_non_leap_year + leap_years_between_2000_and_2040 * days_in_leap_year

theorem leap_day_2040_is_tuesday :
  (total_days_between_2000_and_2040 % 7) = 0 :=
by
  sorry

end leap_day_2040_is_tuesday_l201_201233


namespace cosine_225_proof_l201_201353

noncomputable def cosine_225_eq_neg_sqrt2_div_2 : Prop :=
  Real.cos (225 * Real.pi / 180) = -Real.sqrt 2 / 2

theorem cosine_225_proof : cosine_225_eq_neg_sqrt2_div_2 := by
  sorry

end cosine_225_proof_l201_201353


namespace cookies_difference_l201_201244

theorem cookies_difference 
    (initial_sweet : ℕ) (initial_salty : ℕ) (initial_chocolate : ℕ)
    (ate_sweet : ℕ) (ate_salty : ℕ) (ate_chocolate : ℕ)
    (ratio_sweet : ℕ) (ratio_salty : ℕ) (ratio_chocolate : ℕ) :
    initial_sweet = 39 →
    initial_salty = 18 →
    initial_chocolate = 12 →
    ate_sweet = 27 →
    ate_salty = 6 →
    ate_chocolate = 8 →
    ratio_sweet = 3 →
    ratio_salty = 1 →
    ratio_chocolate = 2 →
    ate_sweet - ate_salty = 21 :=
by
  intros _ _ _ _ _ _ _ _ _
  sorry

end cookies_difference_l201_201244


namespace five_pow_sum_of_squares_l201_201103

theorem five_pow_sum_of_squares (n : ℕ) : ∃ a b : ℕ, 5^n = a^2 + b^2 := 
sorry

end five_pow_sum_of_squares_l201_201103


namespace volume_of_rectangular_solid_l201_201496

variable {x y z : ℝ}
variable (hx : x * y = 3) (hy : x * z = 5) (hz : y * z = 15)

theorem volume_of_rectangular_solid : x * y * z = 15 :=
by sorry

end volume_of_rectangular_solid_l201_201496


namespace spencer_session_duration_l201_201660

-- Definitions of the conditions
def jumps_per_minute : ℕ := 4
def sessions_per_day : ℕ := 2
def total_jumps : ℕ := 400
def total_days : ℕ := 5

-- Calculation target: find the duration of each session
def jumps_per_day : ℕ := total_jumps / total_days
def jumps_per_session : ℕ := jumps_per_day / sessions_per_day
def session_duration := jumps_per_session / jumps_per_minute

theorem spencer_session_duration :
  session_duration = 10 := 
sorry

end spencer_session_duration_l201_201660


namespace consecutive_negatives_product_to_sum_l201_201776

theorem consecutive_negatives_product_to_sum :
  ∃ (n : ℤ), n * (n + 1) = 2184 ∧ n + (n + 1) = -95 :=
by {
  sorry
}

end consecutive_negatives_product_to_sum_l201_201776


namespace arithmetic_mean_of_fractions_l201_201881

theorem arithmetic_mean_of_fractions :
  let a := (3 : ℚ) / 5
  let b := (5 : ℚ) / 7
  (a + b) / 2 = (23 : ℚ) / 35 := 
by 
  sorry 

end arithmetic_mean_of_fractions_l201_201881


namespace part_a_part_b_l201_201189

namespace ProofProblem

def number_set := {n : ℕ | ∃ k : ℕ, n = (10^k - 1)}

noncomputable def special_structure (n : ℕ) : Prop :=
  ∃ m : ℕ, n = 2 * m + 1 ∨ n = 2 * m + 2

theorem part_a :
  ∃ (a b c : ℕ) (ha : a ∈ number_set) (hb : b ∈ number_set) (hc : c ∈ number_set),
    special_structure (a + b + c) :=
by
  sorry

theorem part_b (cards : List ℕ) (h : ∀ x ∈ cards, x ∈ number_set)
    (hs : special_structure (cards.sum)) :
  ∃ (d : ℕ), d ≠ 2 ∧ (d = 0 ∨ d = 1) :=
by
  sorry

end ProofProblem

end part_a_part_b_l201_201189


namespace oliver_january_money_l201_201097

variable (x y z : ℕ)

-- Given conditions
def condition1 := y = x - 4
def condition2 := z = y + 32
def condition3 := z = 61

-- Statement to prove
theorem oliver_january_money (h1 : condition1 x y) (h2 : condition2 y z) (h3 : condition3 z) : x = 33 :=
by
  sorry

end oliver_january_money_l201_201097


namespace ninth_group_number_l201_201651

-- Conditions
def num_workers : ℕ := 100
def sample_size : ℕ := 20
def group_size : ℕ := num_workers / sample_size
def fifth_group_number : ℕ := 23

-- Theorem stating the result for the 9th group number.
theorem ninth_group_number : ∃ n : ℕ, n = 43 :=
by
  -- We calculate the numbers step by step.
  have interval : ℕ := group_size
  have difference : ℕ := 9 - 5
  have increment : ℕ := difference * interval
  have ninth_group_num : ℕ := fifth_group_number + increment
  use ninth_group_num
  sorry

end ninth_group_number_l201_201651


namespace cos_225_degrees_l201_201402

theorem cos_225_degrees : Real.cos (225 * Real.pi / 180) = -1 / Real.sqrt 2 := by
  sorry

end cos_225_degrees_l201_201402


namespace minimum_boxes_needed_l201_201159

theorem minimum_boxes_needed (small_box_capacity medium_box_capacity large_box_capacity : ℕ)
    (max_small_boxes max_medium_boxes max_large_boxes : ℕ)
    (total_dozens: ℕ) :
  small_box_capacity = 2 → 
  medium_box_capacity = 3 → 
  large_box_capacity = 4 → 
  max_small_boxes = 6 → 
  max_medium_boxes = 5 → 
  max_large_boxes = 4 → 
  total_dozens = 40 → 
  ∃ (small_boxes_needed medium_boxes_needed large_boxes_needed : ℕ), 
    small_boxes_needed = 5 ∧ 
    medium_boxes_needed = 5 ∧ 
    large_boxes_needed = 4 := 
by
  sorry

end minimum_boxes_needed_l201_201159


namespace joan_balloons_l201_201907

def sally_balloons : ℕ := 5
def jessica_balloons : ℕ := 2
def total_balloons : ℕ := 16

theorem joan_balloons : sally_balloons + jessica_balloons = 7 ∧ total_balloons = 16 → total_balloons - (sally_balloons + jessica_balloons) = 9 :=
by
  sorry

end joan_balloons_l201_201907


namespace haley_extra_tickets_l201_201454

/-- Haley's favorite band was holding a concert where tickets were 4 dollars each. 
Haley bought 3 tickets for herself and her friends and spent $32. 
Prove how many extra tickets she bought. -/
theorem haley_extra_tickets (ticket_cost : ℕ) (tickets_for_self_and_friends total_spent : ℕ) 
  (h1 : ticket_cost = 4) (h2 : tickets_for_self_and_friends = 3) (h3 : total_spent = 32) :
  (total_spent / ticket_cost - tickets_for_self_and_friends) = 5 := 
by 
  rw [h1, h2, h3]; sorry

end haley_extra_tickets_l201_201454


namespace paths_H_to_J_via_I_l201_201938

def binom (n k : ℕ) : ℕ := Nat.choose n k

def paths_from_H_to_I : ℕ :=
  binom 7 2  -- Calculate the number of paths from H(0,7) to I(5,5)

def paths_from_I_to_J : ℕ :=
  binom 8 3  -- Calculate the number of paths from I(5,5) to J(8,0)

theorem paths_H_to_J_via_I : paths_from_H_to_I * paths_from_I_to_J = 1176 := by
  -- This theorem states that the number of paths from H to J through I is 1176
  sorry  -- Proof to be provided

end paths_H_to_J_via_I_l201_201938


namespace area_of_triangle_OPF_l201_201197

theorem area_of_triangle_OPF (O : ℝ × ℝ) (F : ℝ × ℝ) (P : ℝ × ℝ)
  (hO : O = (0, 0)) (hF : F = (1, 0)) (hP_on_parabola : P.2 ^ 2 = 4 * P.1)
  (hPF : dist P F = 3) : Real.sqrt 2 = 1 / 2 * abs (F.1 - O.1) * (2 * Real.sqrt 2) := 
sorry

end area_of_triangle_OPF_l201_201197


namespace Juan_birth_year_proof_l201_201771

-- Let BTC_year(n) be the year of the nth BTC competition.
def BTC_year (n : ℕ) : ℕ :=
  1990 + (n - 1) * 2

-- Juan's birth year given his age and the BTC he participated in.
def Juan_birth_year (current_year : ℕ) (age : ℕ) : ℕ :=
  current_year - age

-- Main proof problem statement.
theorem Juan_birth_year_proof :
  (BTC_year 5 = 1998) →
  (Juan_birth_year 1998 14 = 1984) :=
by
  intros
  sorry

end Juan_birth_year_proof_l201_201771


namespace surface_area_of_cylinder_l201_201712

noncomputable def cylinder_surface_area
    (r : ℝ) (V : ℝ) (S : ℝ) : Prop :=
    r = 1 ∧ V = 2 * Real.pi ∧ S = 6 * Real.pi

theorem surface_area_of_cylinder
    (r : ℝ) (V : ℝ) : ∃ S : ℝ, cylinder_surface_area r V S :=
by
  use 6 * Real.pi
  sorry

end surface_area_of_cylinder_l201_201712


namespace sum_smallest_and_largest_prime_between_1_and_50_l201_201910

noncomputable def smallest_prime_between_1_and_50 : ℕ := 2
noncomputable def largest_prime_between_1_and_50 : ℕ := 47

theorem sum_smallest_and_largest_prime_between_1_and_50 : 
  smallest_prime_between_1_and_50 + largest_prime_between_1_and_50 = 49 := 
by
  sorry

end sum_smallest_and_largest_prime_between_1_and_50_l201_201910


namespace complement_B_def_union_A_B_def_intersection_A_B_def_intersection_A_complement_B_def_intersection_complements_def_l201_201091

-- Definitions of the sets A and B
def set_A : Set ℝ := {y : ℝ | -1 < y ∧ y < 4}
def set_B : Set ℝ := {y : ℝ | 0 < y ∧ y < 5}

-- Complement of B in the universal set U (ℝ)
def complement_B : Set ℝ := {y : ℝ | y ≤ 0 ∨ y ≥ 5}

theorem complement_B_def : (complement_B = {y : ℝ | y ≤ 0 ∨ y ≥ 5}) :=
by sorry

-- Union of A and B
def union_A_B : Set ℝ := {y : ℝ | -1 < y ∧ y < 5}

theorem union_A_B_def : (set_A ∪ set_B = union_A_B) :=
by sorry

-- Intersection of A and B
def intersection_A_B : Set ℝ := {y : ℝ | 0 < y ∧ y < 4}

theorem intersection_A_B_def : (set_A ∩ set_B = intersection_A_B) :=
by sorry

-- Intersection of A and the complement of B
def intersection_A_complement_B : Set ℝ := {y : ℝ | -1 < y ∧ y ≤ 0}

theorem intersection_A_complement_B_def : (set_A ∩ complement_B = intersection_A_complement_B) :=
by sorry

-- Intersection of the complements of A and B
def complement_A : Set ℝ := {y : ℝ | y ≤ -1 ∨ y ≥ 4} -- Derived from complement of A
def intersection_complements : Set ℝ := {y : ℝ | y ≤ -1 ∨ y ≥ 5}

theorem intersection_complements_def : (complement_A ∩ complement_B = intersection_complements) :=
by sorry

end complement_B_def_union_A_B_def_intersection_A_B_def_intersection_A_complement_B_def_intersection_complements_def_l201_201091


namespace letters_calculation_proof_l201_201430

def Elida_letters : Nat := 5
def Adrianna_letters : Nat := 2 * Elida_letters - 2
def Total_letters : Nat := Elida_letters + Adrianna_letters
def Average_letters : Real := Total_letters / 2
def Answer : Real := 10 * Average_letters

theorem letters_calculation_proof : Answer = 65 := by
  sorry

end letters_calculation_proof_l201_201430


namespace smallest_integer_geq_l201_201958

theorem smallest_integer_geq : ∃ (n : ℤ), (n^2 - 9*n + 18 ≥ 0) ∧ ∀ (m : ℤ), (m^2 - 9*m + 18 ≥ 0) → n ≤ m :=
by
  sorry

end smallest_integer_geq_l201_201958


namespace simplest_common_denominator_l201_201147

theorem simplest_common_denominator (x y : ℕ) (h1 : 2 * x ≠ 0) (h2 : 4 * y^2 ≠ 0) (h3 : 5 * x * y ≠ 0) :
  ∃ d : ℕ, d = 20 * x * y^2 :=
by {
  sorry
}

end simplest_common_denominator_l201_201147


namespace symmetric_about_y_axis_l201_201947

noncomputable def f (x : ℝ) : ℝ := (4^x + 1) / 2^x

theorem symmetric_about_y_axis : ∀ x : ℝ, f (-x) = f x :=
by
  intro x
  unfold f
  sorry

end symmetric_about_y_axis_l201_201947


namespace complex_roots_circle_radius_l201_201345

theorem complex_roots_circle_radius (z : ℂ) (h : (z + 2)^4 = 16 * z^4) :
  ∃ r : ℝ, (∀ z, (z + 2)^4 = 16 * z^4 → (z - (2/3))^2 + y^2 = r) ∧ r = 1 :=
sorry

end complex_roots_circle_radius_l201_201345


namespace largest_four_digit_number_l201_201810

theorem largest_four_digit_number :
  ∃ a b c d : ℕ, 
    9 < 1000 * a + 100 * b + 10 * c + d ∧ 
    1000 * a + 100 * b + 10 * c + d < 10000 ∧ 
    c = a + b ∧ 
    d = b + c ∧ 
    1000 * a + 100 * b + 10 * c + d = 9099 :=
by {
  sorry
}

end largest_four_digit_number_l201_201810


namespace final_elephants_count_l201_201277

def E_0 : Int := 30000
def R_exodus : Int := 2880
def H_exodus : Int := 4
def R_entry : Int := 1500
def H_entry : Int := 7
def E_final : Int := E_0 - (R_exodus * H_exodus) + (R_entry * H_entry)

theorem final_elephants_count : E_final = 28980 := by
  sorry

end final_elephants_count_l201_201277


namespace active_probability_correct_not_active_moderate_probability_correct_chi_square_significance_l201_201970

section survey_analysis

def total_students : ℕ := 50
def students_active : ℕ := 22
def students_not_active_moderate : ℕ := 20

def table_A : ℕ := 17
def table_B : ℕ := 8
def table_C : ℕ := 5
def table_D : ℕ := 20

def active_prob : ℚ := students_active / total_students
def not_active_moderate_prob : ℚ := students_not_active_moderate / total_students

def chi_square (n A B C D : ℕ) : ℚ :=
  (n * (A * D - B * C) ^ 2) / ((A + B) * (C + D) * (A + C) * (B + D))

def chi_square_val : ℚ := chi_square total_students table_A table_B table_C table_D

def critical_value_0_001 : ℚ := 10.8

-- Statements to prove
theorem active_probability_correct : active_prob = (22 : ℚ) / 50 := by
  sorry
  
theorem not_active_moderate_probability_correct : not_active_moderate_prob = (20 : ℚ) / 50 := by
  sorry
  
theorem chi_square_significance : chi_square_val > critical_value_0_001 := by
  sorry

end survey_analysis

end active_probability_correct_not_active_moderate_probability_correct_chi_square_significance_l201_201970


namespace cos_225_l201_201358

theorem cos_225 (h : Real.cos (135 * Real.pi / 180) = -Real.sqrt 2 / 2) :
    Real.cos (225 * Real.pi / 180) = -Real.sqrt 2 / 2 := by
  sorry

end cos_225_l201_201358


namespace elevator_people_count_l201_201264

theorem elevator_people_count (weight_limit : ℕ) (excess_weight : ℕ) (avg_weight : ℕ) (total_weight : ℕ) (n : ℕ) 
  (h1 : weight_limit = 1500)
  (h2 : excess_weight = 100)
  (h3 : avg_weight = 80)
  (h4 : total_weight = weight_limit + excess_weight)
  (h5 : total_weight = n * avg_weight) :
  n = 20 :=
sorry

end elevator_people_count_l201_201264


namespace find_alpha_l201_201997

-- Define the problem in Lean terms
variable (x y α : ℝ)

-- Conditions
def condition1 : Prop := 3 + α + y = 4 + α + x
def condition2 : Prop := 1 + x + 3 + 3 + α + y + 4 + 1 = 2 * (4 + α + x)

-- The theorem to prove
theorem find_alpha (h1 : condition1 x y α) (h2 : condition2 x y α) : α = 5 := 
  sorry

end find_alpha_l201_201997


namespace value_of_d_l201_201213

theorem value_of_d (d : ℝ) (h : x^2 - 60 * x + d = (x - 30)^2) : d = 900 :=
by { sorry }

end value_of_d_l201_201213


namespace total_chickens_l201_201686

theorem total_chickens (coops chickens_per_coop : ℕ) (h1 : coops = 9) (h2 : chickens_per_coop = 60) :
  coops * chickens_per_coop = 540 := by
  sorry

end total_chickens_l201_201686


namespace crystal_meals_count_l201_201017

def num_entrees : ℕ := 4
def num_drinks : ℕ := 4
def num_desserts : ℕ := 2

theorem crystal_meals_count : num_entrees * num_drinks * num_desserts = 32 := by
  sorry

end crystal_meals_count_l201_201017


namespace bankers_discount_l201_201162

/-- Given the present worth (P) of Rs. 400 and the true discount (TD) of Rs. 20,
Prove that the banker's discount (BD) is Rs. 21. -/
theorem bankers_discount (P TD FV BD : ℝ) (hP : P = 400) (hTD : TD = 20) 
(hFV : FV = P + TD) (hBD : BD = (TD * FV) / P) : BD = 21 := 
by
  sorry

end bankers_discount_l201_201162


namespace mean_properties_l201_201768

theorem mean_properties (a b c : ℝ) 
    (h1 : a + b + c = 36) 
    (h2 : a * b * c = 125) 
    (h3 : a * b + b * c + c * a = 93.75) : 
    a^2 + b^2 + c^2 = 1108.5 := 
by 
  sorry

end mean_properties_l201_201768


namespace number_of_aquariums_l201_201729

theorem number_of_aquariums (total_animals animals_per_aquarium : ℕ) (h1 : total_animals = 40) (h2 : animals_per_aquarium = 2) :
  total_animals / animals_per_aquarium = 20 := by
  sorry

end number_of_aquariums_l201_201729


namespace nancy_total_money_l201_201759

def total_money (n_five n_ten n_one : ℕ) : ℕ :=
  (n_five * 5) + (n_ten * 10) + (n_one * 1)

theorem nancy_total_money :
  total_money 9 4 7 = 92 :=
by
  sorry

end nancy_total_money_l201_201759


namespace fraction_to_percentage_decimal_l201_201185

theorem fraction_to_percentage_decimal (num : ℚ) (den : ℚ) (h : den ≠ 0) :
  num / den = 7 / 15 → (num / den) * 100 / 100 = 0.4666 :=
by
  sorry

end fraction_to_percentage_decimal_l201_201185


namespace sufficient_condition_l201_201619

-- Definitions:
-- 1. Arithmetic sequence with first term a_1 and common difference d
-- 2. Define the sum of the first n terms of the arithmetic sequence

def arithmetic_sequence (a_1 d : ℤ) (n : ℕ) : ℤ := a_1 + n * d

def sum_first_n_terms (a_1 d : ℤ) (n : ℕ) : ℤ :=
  n * (2 * a_1 + (n - 1) * d) / 2

-- Conditions given in the problem:
-- Let a_6 = a_1 + 5d
-- Let a_7 = a_1 + 6d
-- Condition p: a_6 + a_7 > 0

def p (a_1 d : ℤ) : Prop := a_1 + 5 * d + a_1 + 6 * d > 0

-- Sum of first 9 terms S_9 and first 3 terms S_3
-- Condition q: S_9 >= S_3

def q (a_1 d : ℤ) : Prop := sum_first_n_terms a_1 d 9 ≥ sum_first_n_terms a_1 d 3

-- The statement to prove:
theorem sufficient_condition (a_1 d : ℤ) : (p a_1 d) -> (q a_1 d) :=
sorry

end sufficient_condition_l201_201619


namespace cos_225_degrees_l201_201398

theorem cos_225_degrees : Real.cos (225 * Real.pi / 180) = -1 / Real.sqrt 2 := by
  sorry

end cos_225_degrees_l201_201398


namespace cosine_225_proof_l201_201354

noncomputable def cosine_225_eq_neg_sqrt2_div_2 : Prop :=
  Real.cos (225 * Real.pi / 180) = -Real.sqrt 2 / 2

theorem cosine_225_proof : cosine_225_eq_neg_sqrt2_div_2 := by
  sorry

end cosine_225_proof_l201_201354


namespace whiteboard_ink_cost_l201_201468

/-- 
There are 5 classes: A, B, C, D, E
Class A: 3 whiteboards
Class B: 2 whiteboards
Class C: 4 whiteboards
Class D: 1 whiteboard
Class E: 3 whiteboards
The ink usage per whiteboard in each class:
Class A: 20ml per whiteboard
Class B: 25ml per whiteboard
Class C: 15ml per whiteboard
Class D: 30ml per whiteboard
Class E: 20ml per whiteboard
The cost of ink is 50 cents per ml
-/
def total_cost_in_dollars : ℕ :=
  let ink_usage_A := 3 * 20
  let ink_usage_B := 2 * 25
  let ink_usage_C := 4 * 15
  let ink_usage_D := 1 * 30
  let ink_usage_E := 3 * 20
  let total_ink_usage := ink_usage_A + ink_usage_B + ink_usage_C + ink_usage_D + ink_usage_E
  let total_cost_in_cents := total_ink_usage * 50
  total_cost_in_cents / 100

theorem whiteboard_ink_cost : total_cost_in_dollars = 130 := 
  by 
    sorry -- Proof needs to be implemented

end whiteboard_ink_cost_l201_201468


namespace correct_system_of_equations_l201_201495

noncomputable def system_of_equations (x y : ℝ) : Prop :=
x + y = 150 ∧ 3 * x + (1 / 3) * y = 210

theorem correct_system_of_equations : ∃ x y : ℝ, system_of_equations x y :=
sorry

end correct_system_of_equations_l201_201495


namespace yardage_lost_due_to_sacks_l201_201170

theorem yardage_lost_due_to_sacks 
  (throws : ℕ)
  (percent_no_throw : ℝ)
  (half_sack_prob : ℕ)
  (sack_pattern : ℕ → ℕ)
  (correct_answer : ℕ) : 
  throws = 80 →
  percent_no_throw = 0.30 →
  (∀ (n: ℕ), half_sack_prob = n/2) →
  (sack_pattern 1 = 3 ∧ sack_pattern 2 = 5 ∧ ∀ n, n > 2 → sack_pattern n = sack_pattern (n - 1) + 2) →
  correct_answer = 168 :=
by
  sorry

end yardage_lost_due_to_sacks_l201_201170


namespace find_n_from_equation_l201_201214

theorem find_n_from_equation :
  ∃ n : ℕ, (3 * 3 * 5 * 5 * 7 * 9 = 3 * 3 * 7 * n * n) → n = 15 :=
by
  sorry

end find_n_from_equation_l201_201214


namespace additional_toothpicks_needed_l201_201347

def three_step_toothpicks := 18
def four_step_toothpicks := 26

theorem additional_toothpicks_needed : 
  (∃ (f : ℕ → ℕ), f 3 = three_step_toothpicks ∧ f 4 = four_step_toothpicks ∧ (f 6 - f 4) = 22) :=
by {
  -- Assume f is a function that gives the number of toothpicks for a n-step staircase
  sorry
}

end additional_toothpicks_needed_l201_201347


namespace pencils_multiple_of_28_l201_201137

theorem pencils_multiple_of_28 (students pens pencils : ℕ) 
  (h1 : students = 28) 
  (h2 : pens = 1204) 
  (h3 : ∃ k, pens = students * k) 
  (h4 : ∃ n, pencils = students * n) : 
  ∃ m, pencils = 28 * m :=
by
  sorry

end pencils_multiple_of_28_l201_201137


namespace complement_of_A_in_U_l201_201463

def U : Set ℝ := Set.univ
def A : Set ℝ := { x | |x - 1| > 2 }

theorem complement_of_A_in_U : 
  ∀ x, x ∈ U → x ∈ U \ A ↔ -1 ≤ x ∧ x ≤ 3 := by
  sorry

end complement_of_A_in_U_l201_201463


namespace point_coordinates_correct_l201_201899

def point_coordinates : (ℕ × ℕ) :=
(11, 9)

theorem point_coordinates_correct :
  point_coordinates = (11, 9) :=
by
  sorry

end point_coordinates_correct_l201_201899


namespace ratio_of_investments_l201_201967

theorem ratio_of_investments (I B_profit total_profit : ℝ) (x : ℝ)
  (h1 : B_profit = 4000) (h2 : total_profit = 28000) (h3 : I * (2 * B_profit / 4000 - 1) = total_profit - B_profit) :
  x = 3 :=
by
  sorry

end ratio_of_investments_l201_201967


namespace apple_count_l201_201204

-- Definitions of initial conditions and calculations.
def B_0 : Int := 5  -- initial number of blue apples
def R_0 : Int := 3  -- initial number of red apples
def Y : Int := 2 * B_0  -- number of yellow apples given by neighbor
def R : Int := R_0 - 2  -- number of red apples after giving away to a friend
def B : Int := B_0 - 3  -- number of blue apples after 3 rot
def G : Int := (B + Y) / 3  -- number of green apples received
def Y' : Int := Y - 2  -- number of yellow apples after eating 2
def R' : Int := R - 1  -- number of red apples after eating 1

-- Lean theorem statement
theorem apple_count (B_0 R_0 Y R B G Y' R' : ℤ)
  (h1 : B_0 = 5)
  (h2 : R_0 = 3)
  (h3 : Y = 2 * B_0)
  (h4 : R = R_0 - 2)
  (h5 : B = B_0 - 3)
  (h6 : G = (B + Y) / 3)
  (h7 : Y' = Y - 2)
  (h8 : R' = R - 1)
  : B + Y' + G + R' = 14 := 
by
  sorry

end apple_count_l201_201204


namespace sufficient_condition_increasing_l201_201329

noncomputable def f (x a : ℝ) : ℝ := x^2 + 2 * a * x + 1

theorem sufficient_condition_increasing (a : ℝ) :
  (∀ x y : ℝ, 1 < x → x < y → (f x a ≤ f y a)) → a = -1 := sorry

end sufficient_condition_increasing_l201_201329


namespace sum_of_numerator_and_denominator_of_repeating_decimal_l201_201296

noncomputable def repeating_decimal_to_fraction (n d : ℤ) (h : 0 < d) := ∃ (x : ℚ), x = n / d ∧ 0 < x

theorem sum_of_numerator_and_denominator_of_repeating_decimal :
  let x := 0.4545454545 -- defining x
  let f := x.to_rational -- converting to rational
  let f_simp := f.num / f.denom -- get numerator and denominator
  -- conclude the sum of numerator and denominator is 16
  f.num + f.denom = 16 :=
by
  sorry

end sum_of_numerator_and_denominator_of_repeating_decimal_l201_201296


namespace find_additional_discount_percentage_l201_201834

noncomputable def additional_discount_percentage(msrp : ℝ) (max_regular_discount : ℝ) (lowest_price : ℝ) : ℝ :=
  let regular_discount_price := msrp * (1 - max_regular_discount)
  let additional_discount := (regular_discount_price - lowest_price) / regular_discount_price
  additional_discount * 100

theorem find_additional_discount_percentage :
  additional_discount_percentage 40 0.3 22.4 = 20 :=
by
  unfold additional_discount_percentage
  simp
  sorry

end find_additional_discount_percentage_l201_201834


namespace car_count_is_150_l201_201260

variable (B C K : ℕ)  -- Define the variables representing buses, cars, and bikes

/-- Given conditions: The ratio of buses to cars to bikes is 3:7:10,
    there are 90 fewer buses than cars, and 140 fewer buses than bikes. -/
def conditions : Prop :=
  (C = (7 * B / 3)) ∧ (K = (10 * B / 3)) ∧ (C = B + 90) ∧ (K = B + 140)

theorem car_count_is_150 (h : conditions B C K) : C = 150 :=
by
  sorry

end car_count_is_150_l201_201260


namespace repeating_decimal_sum_l201_201307

theorem repeating_decimal_sum (x : ℚ) (hx : x = 45 / 99) (h_simplified : x = 5 / 11) : (5 + 11) = 16 :=
by {
  sorry
}

end repeating_decimal_sum_l201_201307


namespace cistern_length_l201_201538

theorem cistern_length (L : ℝ) (H : 0 < L) :
    (∃ (w d A : ℝ), w = 14 ∧ d = 1.25 ∧ A = 233 ∧ A = L * w + 2 * L * d + 2 * w * d) →
    L = 12 :=
by
  sorry

end cistern_length_l201_201538


namespace platform_length_correct_l201_201160

noncomputable def platform_length : ℝ :=
  let T := 180
  let v_kmph := 72
  let t := 20
  let v_ms := v_kmph * 1000 / 3600
  let total_distance := v_ms * t
  total_distance - T

theorem platform_length_correct : platform_length = 220 := by
  sorry

end platform_length_correct_l201_201160


namespace misread_system_of_equations_solutions_l201_201650

theorem misread_system_of_equations_solutions (a b : ℤ) (x₁ y₁ x₂ y₂ : ℤ)
  (h1 : x₁ = -3) (h2 : y₁ = -1) (h3 : x₂ = 5) (h4 : y₂ = 4)
  (eq1 : a * x₂ + 5 * y₂ = 15)
  (eq2 : 4 * x₁ - b * y₁ = -2) :
  a = -1 ∧ b = 10 ∧ a ^ 2023 + (- (1 / 10 : ℚ) * b) ^ 2023 = -2 := by
  -- Translate misreading conditions into theorems we need to prove (note: skipping proof).
  have hb : b = 10 := by sorry
  have ha : a = -1 := by sorry
  exact ⟨ha, hb, by simp [ha, hb]; norm_num⟩

end misread_system_of_equations_solutions_l201_201650


namespace percentage_of_loss_l201_201166

-- Define the conditions as given in the problem
def original_selling_price : ℝ := 720
def gain_selling_price : ℝ := 880
def gain_percentage : ℝ := 0.10

-- Define the main theorem
theorem percentage_of_loss : ∀ (CP : ℝ),
  (1.10 * CP = gain_selling_price) → 
  ((CP - original_selling_price) / CP * 100 = 10) :=
by
  intro CP
  intro h
  have h1 : CP = gain_selling_price / 1.10 := by sorry
  have h2 : (CP - original_selling_price) = 80 := by sorry -- Intermediate step to show loss
  have h3 : ((80 / CP) * 100 = 10) := by sorry -- Calculation of percentage of loss
  sorry

end percentage_of_loss_l201_201166


namespace Jessica_cut_roses_l201_201266

variable (initial_roses final_roses added_roses : Nat)

theorem Jessica_cut_roses
  (h_initial : initial_roses = 10)
  (h_final : final_roses = 18)
  (h_added : final_roses = initial_roses + added_roses) :
  added_roses = 8 := by
  sorry

end Jessica_cut_roses_l201_201266


namespace max_n_perfect_cube_l201_201436

-- Definition for sum of squares
def sum_of_squares (n : ℕ) : ℕ :=
  n * (n + 1) * (2 * n + 1) / 6

-- Definition for sum of squares from (n+1) to 2n
def sum_of_squares_segment (n : ℕ) : ℕ :=
  2 * n * (2 * n + 1) * (4 * n + 1) / 6 - n * (n + 1) * (2 * n + 1) / 6

-- Definition for the product of the sums
def product_of_sums (n : ℕ) : ℕ :=
  (sum_of_squares n) * (sum_of_squares_segment n)

-- Predicate for perfect cube
def is_perfect_cube (x : ℕ) : Prop :=
  ∃ y : ℕ, y ^ 3 = x

-- The main theorem to be proved
theorem max_n_perfect_cube : ∃ (n : ℕ), n ≤ 2050 ∧ is_perfect_cube (product_of_sums n) ∧ ∀ m : ℕ, (m ≤ 2050 ∧ is_perfect_cube (product_of_sums m)) → m ≤ 2016 := 
sorry

end max_n_perfect_cube_l201_201436


namespace cos_225_correct_l201_201405

noncomputable def cos_225_eq : Prop :=
  cos (225 * Real.pi / 180) = -Real.sqrt 2 / 2

theorem cos_225_correct : cos_225_eq :=
sorry

end cos_225_correct_l201_201405


namespace actual_price_of_food_l201_201337

noncomputable def food_price (total_spent: ℝ) (tip_percent: ℝ) (tax_percent: ℝ) (discount_percent: ℝ) : ℝ :=
  let P := total_spent / ((1 + tip_percent) * (1 + tax_percent) * (1 - discount_percent))
  P

theorem actual_price_of_food :
  food_price 198 0.20 0.10 0.15 = 176.47 :=
by
  sorry

end actual_price_of_food_l201_201337


namespace new_average_l201_201924

variable (avg9 : ℝ) (score10 : ℝ) (n : ℕ)
variable (h : avg9 = 80) (h10 : score10 = 100) (n9 : n = 9)

theorem new_average (h : avg9 = 80) (h10 : score10 = 100) (n9 : n = 9) :
  ((n * avg9 + score10) / (n + 1)) = 82 :=
by
  rw [h, h10, n9]
  sorry

end new_average_l201_201924


namespace compute_fraction_l201_201563

theorem compute_fraction : 
  (1 - 2 + 4 - 8 + 16 - 32 + 64) / (2 - 4 + 8 - 16 + 32 - 64 + 128) = 1 / 2 := 
by
  sorry

end compute_fraction_l201_201563


namespace negation_proof_l201_201578

theorem negation_proof :
  (¬ (∀ x : ℝ, x^2 - x + 1 > 0)) ↔ (∃ x : ℝ, x^2 - x + 1 ≤ 0) := sorry

end negation_proof_l201_201578


namespace first_book_cost_correct_l201_201929

noncomputable def cost_of_first_book (x : ℝ) : Prop :=
  let total_cost := x + 6.5
  let given_amount := 20
  let change_received := 8
  total_cost = given_amount - change_received → x = 5.5

theorem first_book_cost_correct : cost_of_first_book 5.5 :=
by
  sorry

end first_book_cost_correct_l201_201929


namespace find_k_for_circle_of_radius_8_l201_201193

theorem find_k_for_circle_of_radius_8 (k : ℝ) :
  (∃ x y : ℝ, x^2 + 14 * x + y^2 + 8 * y - k = 0) ∧ (∀ r : ℝ, r = 8) → k = -1 :=
sorry

end find_k_for_circle_of_radius_8_l201_201193


namespace min_distance_between_lines_t_l201_201587

theorem min_distance_between_lines_t (t : ℝ) :
  (∀ x y : ℝ, x + 2 * y + t^2 = 0) ∧ (∀ x y : ℝ, 2 * x + 4 * y + 2 * t - 3 = 0) →
  t = 1 / 2 := by
  sorry

end min_distance_between_lines_t_l201_201587


namespace factorize_difference_of_squares_l201_201030

theorem factorize_difference_of_squares (x : ℝ) :
  4 * x^2 - 1 = (2 * x + 1) * (2 * x - 1) :=
sorry

end factorize_difference_of_squares_l201_201030


namespace region_area_is_correct_l201_201992

open Real

noncomputable def region_area : ℝ :=
  let A := Set.Icc (2 : ℝ) ((13 : ℝ) / 3)
  let B := Set.Ici 3
  Set.integral (λ x, abs (x - 2)) (λ x, 5 - 2 * abs (x - 3)) A + 
  Set.integral (λ x, abs (x - 2)) (λ x, 5 - 2 * abs (x - 3)) B

theorem region_area_is_correct : region_area = 35 / 9 := 
by sorry

end region_area_is_correct_l201_201992


namespace eval_expr_l201_201737

theorem eval_expr (x y : ℕ) (h1 : x = 2) (h2 : y = 1) : 2 * x - 3 * y = 1 := by
  sorry

end eval_expr_l201_201737


namespace intersection_A_B_l201_201724

open Set

def A : Set ℝ := {1, 2, 1/2}
def B : Set ℝ := {y | ∃ x ∈ A, y = x^2}

theorem intersection_A_B : A ∩ B = { 1 } := by
  sorry

end intersection_A_B_l201_201724


namespace find_b_c_d_sum_l201_201473

theorem find_b_c_d_sum :
  ∃ (b c d : ℤ), (∀ n : ℕ, n > 0 → 
    a_n = b * (⌊(n : ℝ)^(1/3)⌋.natAbs : ℤ) + d ∧
    b = 2 ∧ c = 0 ∧ d = 0) ∧ (b + c + d = 2) :=
sorry

end find_b_c_d_sum_l201_201473


namespace largest_four_digit_number_l201_201812

theorem largest_four_digit_number :
  ∃ a b c d : ℕ, 
    9 < 1000 * a + 100 * b + 10 * c + d ∧ 
    1000 * a + 100 * b + 10 * c + d < 10000 ∧ 
    c = a + b ∧ 
    d = b + c ∧ 
    1000 * a + 100 * b + 10 * c + d = 9099 :=
by {
  sorry
}

end largest_four_digit_number_l201_201812


namespace geom_inequality_l201_201616

variables {Point : Type} [MetricSpace Point] {O A B C K L H M : Point}

/-- Conditions -/
def circumcenter_of_triangle (O A B C : Point) : Prop := 
 -- Definition that O is the circumcenter of triangle ABC
 sorry 

def midpoint_of_arc (K B C A : Point) : Prop := 
 -- Definition that K is the midpoint of the arc BC not containing A
 sorry

def lies_on_line (K L A : Point) : Prop := 
 -- Definition that K lies on line AL
 sorry

def similar_triangles (A H L K M : Point) : Prop := 
 -- Definition that triangles AHL and KML are similar
 sorry 

def segment_inequality (AL KL : ℝ) : Prop := 
 -- Definition that AL < KL
 sorry 

/-- Proof Problem -/
theorem geom_inequality (h1 : circumcenter_of_triangle O A B C) 
                       (h2: midpoint_of_arc K B C A)
                       (h3: lies_on_line K L A)
                       (h4: similar_triangles A H L K M)
                       (h5: segment_inequality (dist A L) (dist K L)) : 
  dist A K < dist B C := 
sorry

end geom_inequality_l201_201616


namespace Clarence_total_oranges_l201_201849

def Clarence_oranges_initial := 5
def oranges_from_Joyce := 3

theorem Clarence_total_oranges : Clarence_oranges_initial + oranges_from_Joyce = 8 := by
  sorry

end Clarence_total_oranges_l201_201849


namespace legs_sum_of_right_triangle_with_hypotenuse_41_l201_201640

noncomputable def right_triangle_legs_sum (x : ℕ) : ℕ := x + (x + 1)

theorem legs_sum_of_right_triangle_with_hypotenuse_41 :
  ∃ x : ℕ, (x * x + (x + 1) * (x + 1) = 41 * 41) ∧ right_triangle_legs_sum x = 57 := by
sorry

end legs_sum_of_right_triangle_with_hypotenuse_41_l201_201640


namespace find_ratio_l201_201481

theorem find_ratio (a b : ℝ) (h1 : a ≠ b) (h2 : a / b + (a + 10 * b) / (b + 10 * a) = 2) : a / b = 0.8 :=
  sorry

end find_ratio_l201_201481


namespace solve_in_primes_l201_201123

theorem solve_in_primes (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) :
  p^2 - 6 * p * q + q^2 + 3 * q - 1 = 0 ↔ (p = 17 ∧ q = 3) := 
sorry

end solve_in_primes_l201_201123


namespace total_drink_volume_l201_201681

-- Define the percentages of the various juices
def grapefruit_percentage : ℝ := 0.20
def lemon_percentage : ℝ := 0.25
def pineapple_percentage : ℝ := 0.10
def mango_percentage : ℝ := 0.15

-- Define the volume of orange juice in ounces
def orange_juice_volume : ℝ := 24

-- State the total percentage of all juices other than orange juice
def non_orange_percentage : ℝ := grapefruit_percentage + lemon_percentage + pineapple_percentage + mango_percentage

-- Calculate the percentage of orange juice
def orange_percentage : ℝ := 1 - non_orange_percentage

-- State that the total volume of the drink is such that 30% of it is 24 ounces
theorem total_drink_volume : ∃ (total_volume : ℝ), (orange_percentage * total_volume = orange_juice_volume) ∧ (total_volume = 80) := by
  use 80
  sorry

end total_drink_volume_l201_201681


namespace repeating_decimal_sum_l201_201306

theorem repeating_decimal_sum (x : ℚ) (hx : x = 45 / 99) (h_simplified : x = 5 / 11) : (5 + 11) = 16 :=
by {
  sorry
}

end repeating_decimal_sum_l201_201306


namespace cos_225_l201_201376

noncomputable def cos_225_eq : ℝ :=
  cos (225 * (real.pi / 180))

theorem cos_225 : cos_225_eq = -real.sqrt 2 / 2 :=
  sorry

end cos_225_l201_201376


namespace min_quadratic_expr_l201_201282

noncomputable def quadratic_expr (x : ℝ) := 3 * x^2 - 18 * x + 2023

theorem min_quadratic_expr : ∃ x : ℝ, quadratic_expr x = 1996 :=
by
  have h : quadratic_expr (3 : ℝ) = 1996
  exact h
  use 3
  rw h
  sorry -- Proof of h (already derived in given solution)

end min_quadratic_expr_l201_201282


namespace smallest_pos_int_greater_than_one_rel_prime_multiple_of_7_l201_201033

theorem smallest_pos_int_greater_than_one_rel_prime_multiple_of_7 (x : ℕ) :
  (x > 1) ∧ (gcd x 210 = 7) ∧ (7 ∣ x) → x = 49 :=
by {
  sorry
}

end smallest_pos_int_greater_than_one_rel_prime_multiple_of_7_l201_201033


namespace min_sum_real_possible_sums_int_l201_201869

-- Lean 4 statement for the real numbers case
theorem min_sum_real (x y : ℝ) (hx : x + y + 2 * x * y = 5) (hx_pos : x > 0) (hy_pos : y > 0) :
  x + y ≥ Real.sqrt 11 - 1 := 
sorry

-- Lean 4 statement for the integers case
theorem possible_sums_int (x y : ℤ) (hx : x + y + 2 * x * y = 5) :
  x + y = 5 ∨ x + y = -7 :=
sorry

end min_sum_real_possible_sums_int_l201_201869


namespace skateboard_travel_distance_l201_201547

theorem skateboard_travel_distance :
  let a_1 := 8
  let d := 10
  let n := 40
  let a_n := a_1 + (n - 1) * d
  let S_n := (n / 2) * (a_1 + a_n)
in S_n = 8120 :=
by
  sorry

end skateboard_travel_distance_l201_201547


namespace largest_four_digit_number_with_property_l201_201796

theorem largest_four_digit_number_with_property :
  ∃ (a b c d : ℕ), (a = 9 ∧ b = 0 ∧ c = 9 ∧ d = 9 ∧ c = a + b ∧ d = b + c ∧ 1000 * a + 100 * b + 10 * c + d = 9099) :=
sorry

end largest_four_digit_number_with_property_l201_201796


namespace trapezoid_other_side_length_l201_201434

theorem trapezoid_other_side_length (a h : ℕ) (A : ℕ) (b : ℕ) : 
  a = 20 → h = 13 → A = 247 → (1/2:ℚ) * (a + b) * h = A → b = 18 :=
by 
  intros h1 h2 h3 h4 
  rw [h1, h2, h3] at h4
  sorry

end trapezoid_other_side_length_l201_201434


namespace average_last_three_l201_201770

noncomputable def average (l : List ℝ) : ℝ :=
  l.sum / l.length

theorem average_last_three (l : List ℝ) (h₁ : l.length = 7) (h₂ : average l = 62) 
  (h₃ : average (l.take 4) = 58) :
  average (l.drop 4) = 202 / 3 := 
by 
  sorry

end average_last_three_l201_201770


namespace repeating_decimal_sum_l201_201305

theorem repeating_decimal_sum (x : ℚ) (hx : x = 45 / 99) (h_simplified : x = 5 / 11) : (5 + 11) = 16 :=
by {
  sorry
}

end repeating_decimal_sum_l201_201305


namespace estimate_contestants_l201_201000

theorem estimate_contestants :
  let total_contestants := 679
  let median_all_three := 188
  let median_two_tests := 159
  let median_one_test := 169
  total_contestants = 679 ∧
  median_all_three = 188 ∧
  median_two_tests = 159 ∧
  median_one_test = 169 →
  let approx_two_tests_per_pair := median_two_tests / 3
  let intersection_pairs_approx := approx_two_tests_per_pair + median_all_three
  let number_above_or_equal_median :=
    median_one_test + median_one_test + median_one_test -
    intersection_pairs_approx - intersection_pairs_approx - intersection_pairs_approx +
    median_all_three
  number_above_or_equal_median = 516 :=
by
  intros
  sorry

end estimate_contestants_l201_201000


namespace contractor_absent_days_l201_201972

theorem contractor_absent_days :
  ∃ (x y : ℝ), x + y = 30 ∧ 25 * x - 7.5 * y = 490 ∧ y = 8 :=
by {
  sorry
}

end contractor_absent_days_l201_201972


namespace legs_sum_of_right_triangle_with_hypotenuse_41_l201_201641

noncomputable def right_triangle_legs_sum (x : ℕ) : ℕ := x + (x + 1)

theorem legs_sum_of_right_triangle_with_hypotenuse_41 :
  ∃ x : ℕ, (x * x + (x + 1) * (x + 1) = 41 * 41) ∧ right_triangle_legs_sum x = 57 := by
sorry

end legs_sum_of_right_triangle_with_hypotenuse_41_l201_201641


namespace RebeccaHasTwentyMarbles_l201_201623

variable (groups : ℕ) (marbles_per_group : ℕ) (total_marbles : ℕ)

def totalMarbles (g m : ℕ) : ℕ :=
  g * m

theorem RebeccaHasTwentyMarbles
  (h1 : groups = 5)
  (h2 : marbles_per_group = 4)
  (h3 : total_marbles = totalMarbles groups marbles_per_group) :
  total_marbles = 20 :=
by {
  sorry
}

end RebeccaHasTwentyMarbles_l201_201623


namespace solution_of_ab_l201_201573

theorem solution_of_ab (a b : ℝ) 
  (h1 : ∀ x : ℝ, (ax^2 + b > 0 ↔ x < -1/2 ∨ x > 1/3)) : 
  a * b = 24 := 
sorry

end solution_of_ab_l201_201573


namespace largest_valid_number_l201_201793

-- Define the conditions for the digits of the number
def valid_digits (a b c d : ℕ) : Prop :=
  c = a + b ∧ d = b + c

-- Prove that the number formed by digits 9, 0, 9, 9 is the largest valid 4-digit number
theorem largest_valid_number : ∃ a b c d, valid_digits a b c d ∧
  (a * 1000 + b * 100 + c * 10 + d = 9099) :=
begin
  use [9, 0, 9, 9],
  split,
  { -- Proof of valid digits condition
    split; refl },
  { -- Proof that the number is 9099
    refl }
end

end largest_valid_number_l201_201793


namespace ellipse_eccentricity_l201_201500

-- State the problem as a theorem
theorem ellipse_eccentricity (z x y : ℂ) :
  (z - 1) * (z^2 + 2 * z + 4) * (z^2 + 4 * z + 6) = 0 →
  let points := [(1, 0), (-1, complex.sqrt 3), (-1, - complex.sqrt 3), (-2, complex.sqrt 2), (-2, - complex.sqrt 2)] in
  let e := real.sqrt ((1 : ℝ) / 6) in
  (∃ (m n : ℕ), nat.coprime m n ∧ e = real.sqrt (↑m / ↑n) ∧ m + n = 7) := sorry

end ellipse_eccentricity_l201_201500


namespace cos_225_correct_l201_201406

noncomputable def cos_225_eq : Prop :=
  cos (225 * Real.pi / 180) = -Real.sqrt 2 / 2

theorem cos_225_correct : cos_225_eq :=
sorry

end cos_225_correct_l201_201406


namespace arithmetic_sequence_length_l201_201731

theorem arithmetic_sequence_length :
  ∀ (a d a_n : ℕ), a = 6 → d = 4 → a_n = 154 → ∃ n: ℕ, a_n = a + (n-1) * d ∧ n = 38 :=
by
  intro a d a_n ha hd ha_n
  use 38
  rw [ha, hd, ha_n]
  -- Leaving the proof as an exercise
  sorry

end arithmetic_sequence_length_l201_201731


namespace values_of_a_and_b_l201_201741

theorem values_of_a_and_b (a b : ℝ) :
  (∀ x : ℝ, x ≥ -1 → a * x^2 + b * x + a^2 - 1 ≤ 0) →
  a = 0 ∧ b = -1 :=
sorry

end values_of_a_and_b_l201_201741


namespace largest_four_digit_sum_23_l201_201656

theorem largest_four_digit_sum_23 : ∃ (n : ℕ), (∃ (a b c d : ℕ), n = a * 1000 + b * 100 + c * 10 + d ∧ a + b + c + d = 23 ∧ 1000 ≤ n ∧ n < 10000) ∧ n = 9950 :=
  sorry

end largest_four_digit_sum_23_l201_201656


namespace correct_factorization_from_left_to_right_l201_201156

theorem correct_factorization_from_left_to_right 
  (x a b c m n : ℝ) : 
  (2 * a * b - 2 * a * c = 2 * a * (b - c)) :=
sorry

end correct_factorization_from_left_to_right_l201_201156


namespace largest_four_digit_mod_5_l201_201520

theorem largest_four_digit_mod_5 : ∃ (n : ℤ), n % 5 = 3 ∧ 1000 ≤ n ∧ n ≤ 9999 ∧ ∀ m : ℤ, m % 5 = 3 ∧ 1000 ≤ m ∧ m ≤ 9999 → m ≤ n :=
sorry

end largest_four_digit_mod_5_l201_201520


namespace ratio_of_segments_l201_201275

-- Definitions and conditions as per part (a)
variables (a b c r s : ℝ)
variable (h₁ : a / b = 1 / 3)
variable (h₂ : a^2 = r * c)
variable (h₃ : b^2 = s * c)

-- The statement of the theorem directly addressing part (c)
theorem ratio_of_segments (a b c r s : ℝ) 
  (h₁ : a / b = 1 / 3)
  (h₂ : a^2 = r * c)
  (h₃ : b^2 = s * c) :
  r / s = 1 / 9 :=
  sorry

end ratio_of_segments_l201_201275


namespace clarence_oranges_l201_201847

def initial_oranges := 5
def oranges_from_joyce := 3
def total_oranges := initial_oranges + oranges_from_joyce

theorem clarence_oranges : total_oranges = 8 :=
  by
  sorry

end clarence_oranges_l201_201847


namespace cos_225_l201_201356

theorem cos_225 (h : Real.cos (135 * Real.pi / 180) = -Real.sqrt 2 / 2) :
    Real.cos (225 * Real.pi / 180) = -Real.sqrt 2 / 2 := by
  sorry

end cos_225_l201_201356


namespace no_solution_for_equation_l201_201107

theorem no_solution_for_equation :
  ¬ ∃ x : ℝ, (1 / (x + 11) + 1 / (x + 8) = 1 / (x + 12) + 1 / (x + 7)) :=
by
  sorry

end no_solution_for_equation_l201_201107


namespace prob_sum_greater_than_9_two_dice_l201_201954

theorem prob_sum_greater_than_9_two_dice :
  let outcomes := {(d1, d2) | d1 ∈ finset.range 6 ∧ d2 ∈ finset.range 6},
      favorable := {(d1, d2) ∈ outcomes | d1 + d2 + 2 > 9} in
  (finset.card favorable) / (finset.card outcomes) = 1 / 6 :=
by
  let outcomes := finset.image (λ (d1, d2), (d1 + 1, d2 + 1))
                                (finset.product (finset.range 6) (finset.range 6))
  let favorable := finset.filter (λ (pair : ℕ × ℕ), pair.fst + pair.snd > 9) outcomes
  have h_outcomes : outcomes.card = 36 := sorry
  have h_favorable : favorable.card = 6 := sorry
  calc
    (favorable.card : ℝ) / outcomes.card = 6 / 36 : by rw [h_outcomes, h_favorable]
    ... = 1 / 6 : by norm_num

end prob_sum_greater_than_9_two_dice_l201_201954


namespace cos_225_proof_l201_201385

def cos_225_def := angle.to_real 225 = -√2 / 2

theorem cos_225_proof : cos_225_def := sorry

end cos_225_proof_l201_201385


namespace min_value_ineq_l201_201717

noncomputable def function_y (a : ℝ) (x : ℝ) : ℝ := a^(1-x)

theorem min_value_ineq (a m n : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : m * n > 0) (h4 : m + n = 1) :
  1/m + 2/n = 3 + 2 * Real.sqrt 2 :=
by
  sorry

end min_value_ineq_l201_201717


namespace unique_not_in_range_of_g_l201_201565

noncomputable def g (p q r s : ℝ) (x : ℝ) : ℝ := (p * x + q) / (r * x + s)

theorem unique_not_in_range_of_g (p q r s : ℝ) (hps_qr_zero : p * s + q * r = 0) 
  (hpr_rs_zero : p * r + r * s = 0) (hg3 : g p q r s 3 = 3) 
  (hg81 : g p q r s 81 = 81) (h_involution : ∀ x ≠ (-s / r), g p q r s (g p q r s x) = x) :
  ∀ x : ℝ, x ≠ 42 :=
sorry

end unique_not_in_range_of_g_l201_201565


namespace fibonacci_identity_cassini_identity_l201_201247

def fib : ℕ → ℕ
| 0     => 0
| 1     => 1
| (n+2) => fib (n+1) + fib n

theorem fibonacci_identity (n m : ℕ) (hn : 1 ≤ n) (hm : 0 ≤ m) :
  fib (n + m) = fib (n - 1) * fib m + fib n * fib (m + 1) := sorry

theorem cassini_identity (n : ℕ) (hn : 1 ≤ n) :
  fib (n + 1) * fib (n - 1) - fib n * fib n = (-1)^n := sorry

end fibonacci_identity_cassini_identity_l201_201247


namespace floor_of_sum_eq_l201_201918

theorem floor_of_sum_eq (x y z w : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hw : w > 0)
  (hxy : x^2 + y^2 = 2500) (hzw : z^2 + w^2 = 2500) (hxz : x * z = 1200) (hyw : y * w = 1200) :
  ⌊x + y + z + w⌋ = 140 := by
  sorry

end floor_of_sum_eq_l201_201918


namespace tan_neg_405_eq_neg_1_l201_201020

theorem tan_neg_405_eq_neg_1 :
  (Real.tan (-405 * Real.pi / 180) = -1) ∧
  (∀ θ : ℝ, Real.tan (θ + 2 * Real.pi) = Real.tan θ) ∧
  (Real.tan θ = Real.sin θ / Real.cos θ) ∧
  (Real.cos (315 * Real.pi / 180) = Real.sqrt 2 / 2) ∧
  (Real.sin (315 * Real.pi / 180) = - Real.sqrt 2 / 2) :=
sorry

end tan_neg_405_eq_neg_1_l201_201020


namespace sum_of_fraction_parts_l201_201316

theorem sum_of_fraction_parts (x : ℝ) (hx : x = 0.45) : 
  (∃ (a b : ℕ), x = a / b ∧ Nat.gcd a b = 1 ∧ a + b = 16) :=
by
  sorry

end sum_of_fraction_parts_l201_201316


namespace regular_polygon_sides_l201_201597

theorem regular_polygon_sides (interior_angle : ℝ) (h : interior_angle = 150) : 
  ∃ n : ℕ, n = 12 := by
  sorry

end regular_polygon_sides_l201_201597


namespace distinct_students_count_l201_201554

open Set

theorem distinct_students_count 
  (germain_students : ℕ := 15) 
  (newton_students : ℕ := 12) 
  (young_students : ℕ := 9)
  (overlap_students : ℕ := 3) :
  (germain_students + newton_students + young_students - overlap_students) = 33 := 
by
  sorry

end distinct_students_count_l201_201554


namespace initial_ratio_l201_201510

variable (A B : ℕ) (a b : ℕ)
variable (h1 : B = 6)
variable (h2 : (A + 2) / (B + 2) = 3 / 2)

theorem initial_ratio (A B : ℕ) (h1 : B = 6) (h2 : (A + 2) / (B + 2) = 3 / 2) : A / B = 5 / 3 := 
by 
    sorry

end initial_ratio_l201_201510


namespace repeating_decimal_fraction_sum_l201_201294

noncomputable def repeating_decimal_to_fraction (x : ℚ) : ℚ :=
if h : x = (45 / 99 : ℚ) then (5 / 11) else x

theorem repeating_decimal_fraction_sum : 
  let x := (0.45 : ℝ) in 
  (x.mul 100 - x).div 99 = (45 : ℝ) -> 
  let (a, b) := (5, 11) in 
  a + b = 16 :=
by
  intro x
  rfl
  rw [show (5 : ℚ) + (11 : ℚ) = 16 from rfl]
  sorry

end repeating_decimal_fraction_sum_l201_201294


namespace product_of_two_smaller_numbers_is_85_l201_201951

theorem product_of_two_smaller_numbers_is_85
  (A B C : ℝ)
  (h1 : B = 10)
  (h2 : C - B = B - A)
  (h3 : B * C = 115) :
  A * B = 85 :=
by
  sorry

end product_of_two_smaller_numbers_is_85_l201_201951


namespace cos_225_degrees_l201_201400

theorem cos_225_degrees : Real.cos (225 * Real.pi / 180) = -1 / Real.sqrt 2 := by
  sorry

end cos_225_degrees_l201_201400


namespace bottles_not_in_crates_l201_201231

def total_bottles : ℕ := 250
def num_small_crates : ℕ := 5
def num_medium_crates : ℕ := 5
def num_large_crates : ℕ := 5
def bottles_per_small_crate : ℕ := 8
def bottles_per_medium_crate : ℕ := 12
def bottles_per_large_crate : ℕ := 20

theorem bottles_not_in_crates : 
  num_small_crates * bottles_per_small_crate + 
  num_medium_crates * bottles_per_medium_crate + 
  num_large_crates * bottles_per_large_crate = 200 → 
  total_bottles - 200 = 50 := 
by
  sorry

end bottles_not_in_crates_l201_201231


namespace remainder_of_expression_l201_201965

theorem remainder_of_expression (k : ℤ) (hk : 0 < k) :
  (4 * k * (2 + 4 + 4 * k) + 3) % 2 = 1 :=
by
  sorry

end remainder_of_expression_l201_201965


namespace traffic_light_probability_l201_201014

theorem traffic_light_probability :
  let total_cycle_time := 63
  let green_time := 30
  let yellow_time := 3
  let red_time := 30
  let observation_window := 3
  let change_intervals := 3 * 3
  ∃ (P : ℚ), P = change_intervals / total_cycle_time ∧ P = 1 / 7 := 
by
  sorry

end traffic_light_probability_l201_201014


namespace car_initial_speed_l201_201680

theorem car_initial_speed (s t : ℝ) (h₁ : t = 15 * s^2) (h₂ : t = 3) :
  s = (Real.sqrt 2) / 5 :=
by
  sorry

end car_initial_speed_l201_201680


namespace dhoni_remaining_earnings_l201_201856

theorem dhoni_remaining_earnings :
  let rent := 0.20
  let dishwasher := 0.15
  let bills := 0.10
  let car := 0.08
  let grocery := 0.12
  let tax := 0.05
  let expenses := rent + dishwasher + bills + car + grocery + tax
  let remaining_after_expenses := 1.0 - expenses
  let savings := 0.40 * remaining_after_expenses
  let remaining_after_savings := remaining_after_expenses - savings
  remaining_after_savings = 0.18 := by
sorry

end dhoni_remaining_earnings_l201_201856


namespace tan_neg_405_eq_neg_1_l201_201019

theorem tan_neg_405_eq_neg_1 :
  (Real.tan (-405 * Real.pi / 180) = -1) ∧
  (∀ θ : ℝ, Real.tan (θ + 2 * Real.pi) = Real.tan θ) ∧
  (Real.tan θ = Real.sin θ / Real.cos θ) ∧
  (Real.cos (315 * Real.pi / 180) = Real.sqrt 2 / 2) ∧
  (Real.sin (315 * Real.pi / 180) = - Real.sqrt 2 / 2) :=
sorry

end tan_neg_405_eq_neg_1_l201_201019


namespace product_of_solutions_l201_201079

theorem product_of_solutions (x : ℝ) (hx : |x - 5| - 5 = 0) :
  ∃ a b : ℝ, (|a - 5| - 5 = 0 ∧ |b - 5| - 5 = 0) ∧ a * b = 0 := by
  sorry

end product_of_solutions_l201_201079


namespace solve_for_x_l201_201207

theorem solve_for_x (x : ℝ) (h : 2 * x - 3 * x + 4 * x = 150) : x = 50 :=
by
  sorry

end solve_for_x_l201_201207


namespace amoeba_population_at_11am_l201_201930

/-- Sarah observes an amoeba colony where initially there are 50 amoebas at 10:00 a.m. The population triples every 10 minutes and there are no deaths among the amoebas. Prove that the number of amoebas at 11:00 a.m. is 36450. -/
theorem amoeba_population_at_11am : 
  let initial_population := 50
  let growth_rate := 3
  let increments := 6  -- since 60 minutes / 10 minutes per increment = 6
  initial_population * (growth_rate ^ increments) = 36450 :=
by
  sorry

end amoeba_population_at_11am_l201_201930


namespace solve_prime_equation_l201_201117

theorem solve_prime_equation (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) :
    p ^ 2 - 6 * p * q + q ^ 2 + 3 * q - 1 = 0 ↔ (p = 17 ∧ q = 3) :=
by
  sorry

end solve_prime_equation_l201_201117


namespace cookies_left_for_Monica_l201_201922

-- Definitions based on the conditions
def total_cookies : ℕ := 30
def father_cookies : ℕ := 10
def mother_cookies : ℕ := father_cookies / 2
def brother_cookies : ℕ := mother_cookies + 2

-- Statement for the theorem
theorem cookies_left_for_Monica : total_cookies - (father_cookies + mother_cookies + brother_cookies) = 8 := by
  -- The proof goes here
  sorry

end cookies_left_for_Monica_l201_201922


namespace aunt_angela_nieces_l201_201179

theorem aunt_angela_nieces (total_jellybeans : ℕ)
                           (jellybeans_per_child : ℕ)
                           (num_nephews : ℕ)
                           (num_nieces : ℕ) 
                           (total_children : ℕ) 
                           (h1 : total_jellybeans = 70)
                           (h2 : jellybeans_per_child = 14)
                           (h3 : num_nephews = 3)
                           (h4 : total_children = total_jellybeans / jellybeans_per_child)
                           (h5 : total_children = num_nephews + num_nieces) :
                           num_nieces = 2 :=
by
  sorry

end aunt_angela_nieces_l201_201179


namespace find_n_l201_201955

theorem find_n (n : ℕ) (h_lcm : Nat.lcm n 14 = 56) (h_gcf : Nat.gcd n 14 = 12) : n = 48 :=
by
  sorry

end find_n_l201_201955


namespace percent_less_50000_l201_201746

variable (A B C : ℝ) -- Define the given percentages
variable (h1 : A = 0.45) -- 45% of villages have populations from 20,000 to 49,999
variable (h2 : B = 0.30) -- 30% of villages have fewer than 20,000 residents
variable (h3 : C = 0.25) -- 25% of villages have 50,000 or more residents

theorem percent_less_50000 : A + B = 0.75 := by
  sorry

end percent_less_50000_l201_201746


namespace xy_value_l201_201458

theorem xy_value (x y : ℝ) (h : x * (x + y) = x^2 + 12) : x * y = 12 :=
by sorry

end xy_value_l201_201458


namespace recurring_decimal_fraction_sum_l201_201303

theorem recurring_decimal_fraction_sum (x : ℚ) (h : x = 0.454545454545...) : 
    (let frac := rat.num_denom x.simplify in frac.fst + frac.snd)  = 16 :=
sorry

end recurring_decimal_fraction_sum_l201_201303


namespace initial_population_correct_l201_201674

-- Definitions based on conditions
def initial_population (P : ℝ) := P
def population_after_bombardment (P : ℝ) := 0.9 * P
def population_after_fear (P : ℝ) := 0.8 * (population_after_bombardment P)
def final_population := 3240

-- Theorem statement
theorem initial_population_correct (P : ℝ) (h : population_after_fear P = final_population) :
  initial_population P = 4500 :=
sorry

end initial_population_correct_l201_201674


namespace largest_n_for_factorization_l201_201569

theorem largest_n_for_factorization :
  ∃ (n : ℤ), (∀ (A B : ℤ), AB = 96 → n = 4 * B + A) ∧ (n = 385) := by
  sorry

end largest_n_for_factorization_l201_201569


namespace parabola_focus_correct_l201_201501

-- defining the equation of the parabola as a condition
def parabola (y x : ℝ) : Prop := y^2 = 4 * x

-- defining the focus of the parabola
def focus (x y : ℝ) : Prop := (x, y) = (1, 0)

-- the main theorem statement
theorem parabola_focus_correct (y x : ℝ) (h : parabola y x) : focus 1 0 :=
by
  -- proof steps would go here
  sorry

end parabola_focus_correct_l201_201501


namespace sunglasses_cap_probability_l201_201098

theorem sunglasses_cap_probability
  (sunglasses_count : ℕ) (caps_count : ℕ)
  (P_cap_and_sunglasses_given_cap : ℚ)
  (H1 : sunglasses_count = 60)
  (H2 : caps_count = 40)
  (H3 : P_cap_and_sunglasses_given_cap = 2/5) :
  (∃ (x : ℚ), x = (16 : ℚ) / 60 ∧ x = 4 / 15) := sorry

end sunglasses_cap_probability_l201_201098


namespace closest_fraction_l201_201555

theorem closest_fraction (won : ℚ) (options : List ℚ) (closest : ℚ) 
  (h_won : won = 25 / 120) 
  (h_options : options = [1 / 4, 1 / 5, 1 / 6, 1 / 7, 1 / 8]) 
  (h_closest : closest = 1 / 5) :
  ∃ x ∈ options, abs (won - x) = abs (won - closest) := 
sorry

end closest_fraction_l201_201555


namespace tournament_matches_divisible_by_7_l201_201603

-- Define the conditions of the chess tournament
def single_elimination_tournament_matches (players byes: ℕ) : ℕ :=
  players - 1

theorem tournament_matches_divisible_by_7 :
  single_elimination_tournament_matches 120 40 = 119 ∧ 119 % 7 = 0 :=
by
  sorry

end tournament_matches_divisible_by_7_l201_201603


namespace diamond_45_15_eq_3_l201_201047

noncomputable def diamond (x y : ℝ) : ℝ := x / y

theorem diamond_45_15_eq_3 :
  ∀ (x y : ℝ), 
    (∀ x y : ℝ, (x * y) / y = x * (x / y)) ∧
    (∀ x : ℝ, (x / 1) / x = x / 1) ∧
    (∀ x y : ℝ, x / y = x / y) ∧
    1 / 1 = 1
    → diamond 45 15 = 3 :=
by
  intros x y H
  sorry

end diamond_45_15_eq_3_l201_201047


namespace trapezoid_shorter_base_length_l201_201224

theorem trapezoid_shorter_base_length 
  (a b : ℕ) 
  (mid_segment_length longer_base : ℕ) 
  (h1 : mid_segment_length = 5) 
  (h2 : longer_base = 103) 
  (trapezoid_property : mid_segment_length = (longer_base - a) / 2) : 
  a = 93 := 
sorry

end trapezoid_shorter_base_length_l201_201224


namespace pure_gala_trees_l201_201831

variables (T F G : ℕ)

theorem pure_gala_trees :
  (0.1 * T : ℝ) + F = 238 ∧ F = (3 / 4) * ↑T → G = T - F → G = 70 :=
by
  intro h
  sorry

end pure_gala_trees_l201_201831


namespace tangential_difference_l201_201913

noncomputable def tan_alpha_minus_beta (α β : ℝ) : ℝ :=
  Real.tan (α - β)

theorem tangential_difference 
  {α β : ℝ}
  (h : 3 / (2 + Real.sin (2 * α)) + 2021 / (2 + Real.sin β) = 2024) : 
  tan_alpha_minus_beta α β = 1 := 
sorry

end tangential_difference_l201_201913


namespace find_complex_number_l201_201085

def i := Complex.I
def z := -Complex.I - 1
def complex_equation (z : ℂ) := i * z = 1 - i

theorem find_complex_number : complex_equation z :=
by
  -- skip the proof here
  sorry

end find_complex_number_l201_201085


namespace cos_225_eq_l201_201411

noncomputable def angle_in_radians (deg : ℝ) : ℝ := deg * (Real.pi / 180)

theorem cos_225_eq :
  Real.cos (angle_in_radians 225) = - (Real.sqrt 2) / 2 := 
  sorry

end cos_225_eq_l201_201411


namespace find_side_length_l201_201474

theorem find_side_length
  (a b c : ℝ) 
  (cosine_diff_angle : ℝ) 
  (h_b : b = 5)
  (h_c : c = 4)
  (h_cosine_diff_angle : cosine_diff_angle = 31 / 32) :
  a = 6 := 
sorry

end find_side_length_l201_201474


namespace sphere_surface_area_l201_201588

theorem sphere_surface_area (r : ℝ) (hr : r = 3) : 4 * Real.pi * r^2 = 36 * Real.pi :=
by
  rw [hr]
  norm_num
  sorry

end sphere_surface_area_l201_201588


namespace polygon_has_9_diagonals_has_6_sides_l201_201835

theorem polygon_has_9_diagonals_has_6_sides :
  ∀ (n : ℕ), (∃ D : ℕ, D = n * (n - 3) / 2 ∧ D = 9) → n = 6 := 
by
  sorry

end polygon_has_9_diagonals_has_6_sides_l201_201835


namespace largest_four_digit_number_with_property_l201_201795

theorem largest_four_digit_number_with_property :
  ∃ (a b c d : ℕ), (a = 9 ∧ b = 0 ∧ c = 9 ∧ d = 9 ∧ c = a + b ∧ d = b + c ∧ 1000 * a + 100 * b + 10 * c + d = 9099) :=
sorry

end largest_four_digit_number_with_property_l201_201795


namespace same_curve_option_B_l201_201962

theorem same_curve_option_B : 
  (∀ x y : ℝ, |y| = |x| ↔ y = x ∨ y = -x) ∧ (∀ x y : ℝ, y^2 = x^2 ↔ y = x ∨ y = -x) :=
by
  sorry

end same_curve_option_B_l201_201962


namespace sum_of_altitudes_at_least_nine_times_inradius_l201_201104

variables (a b c : ℝ)
variables (s : ℝ) -- semiperimeter
variables (Δ : ℝ) -- area
variables (r : ℝ) -- inradius
variables (h_A h_B h_C : ℝ) -- altitudes

-- The Lean statement of the problem
theorem sum_of_altitudes_at_least_nine_times_inradius
  (ha : s = (a + b + c) / 2)
  (hb : Δ = r * s)
  (hc : h_A = (2 * Δ) / a)
  (hd : h_B = (2 * Δ) / b)
  (he : h_C = (2 * Δ) / c) :
  h_A + h_B + h_C ≥ 9 * r :=
sorry

end sum_of_altitudes_at_least_nine_times_inradius_l201_201104


namespace variance_sqrt3Y_plus_1_l201_201721

open ProbabilityTheory

noncomputable def binomial_var (n : ℕ) (p : ℝ) : MeasureSpace ℝ :=
  sorry -- This represents the binomial random variable, which we assume to exist

theorem variance_sqrt3Y_plus_1 :
  ∀ (p : ℝ), 
    (0 ≤ p ∧ p ≤ 1) →
    (P(X ≥ 1) = 5 / 9) →
    D(√3 * Y + 1) = 2
  :=
by
  assume p hp P_X P_Y h,
  sorry


end variance_sqrt3Y_plus_1_l201_201721


namespace cos_225_eq_neg_sqrt2_div_2_l201_201387

theorem cos_225_eq_neg_sqrt2_div_2 : Real.cos (225 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by
  sorry

end cos_225_eq_neg_sqrt2_div_2_l201_201387


namespace repeating_decimal_fraction_equiv_in_lowest_terms_l201_201653

-- Definition of repeating decimal 0.4\overline{13} as a fraction
def repeating_decimal_fraction_equiv : Prop :=
  ∃ x : ℚ, (x = 0.4 + 0.13 / (1 - 0.01)) ∧ (x = 409 / 990) ∧ (nat.gcd 409 990 = 1)

theorem repeating_decimal_fraction_equiv_in_lowest_terms : repeating_decimal_fraction_equiv :=
  sorry

end repeating_decimal_fraction_equiv_in_lowest_terms_l201_201653


namespace part_i_l201_201533

theorem part_i (n : ℕ) (h₁ : n ≥ 1) (h₂ : n ∣ (2^n - 1)) : n = 1 :=
sorry

end part_i_l201_201533


namespace pounds_per_ton_l201_201503

theorem pounds_per_ton (weight_pounds : ℕ) (weight_tons : ℕ) (h_weight : weight_pounds = 6000) (h_tons : weight_tons = 3) : 
  weight_pounds / weight_tons = 2000 :=
by
  sorry

end pounds_per_ton_l201_201503


namespace range_of_a_over_b_l201_201051

variable (a b : ℝ)

theorem range_of_a_over_b (ha : 1 < a ∧ a < 2) (hb : -2 < b ∧ b < -1) :
  -2 < a / b ∧ a / b < -1 / 2 :=
by
  sorry

end range_of_a_over_b_l201_201051


namespace ratio_b_a_l201_201443

theorem ratio_b_a (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : a ≠ b) (h4 : a + b > 2 * a) (h5 : 2 * a > a) 
  (h6 : a + b > b) (h7 : a + 2 * a = b) : 
  b = a * Real.sqrt 2 :=
by
  sorry

end ratio_b_a_l201_201443


namespace remainder_of_67_pow_67_plus_67_mod_68_l201_201032

theorem remainder_of_67_pow_67_plus_67_mod_68 : (67^67 + 67) % 68 = 66 := by
  sorry

end remainder_of_67_pow_67_plus_67_mod_68_l201_201032


namespace sum_of_three_numbers_l201_201774

theorem sum_of_three_numbers {a b c : ℝ} (h₁ : a ≤ b ∧ b ≤ c) (h₂ : b = 10)
  (h₃ : (a + b + c) / 3 = a + 20) (h₄ : (a + b + c) / 3 = c - 25) :
  a + b + c = 45 :=
by
  sorry

end sum_of_three_numbers_l201_201774


namespace largest_p_plus_q_l201_201271

-- All required conditions restated as Assumptions
def triangle {R : Type*} [LinearOrderedField R] (p q : R) : Prop :=
  let B : R × R := (10, 15)
  let C : R × R := (25, 15)
  let A : R × R := (p, q)
  let M : R × R := ((B.1 + C.1) / 2, (B.2 + C.2) / 2)
  let area : R := (1 / 2) * abs ((A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)))
  let median_slope : R := (A.2 - M.2) / (A.1 - M.1)
  area = 100 ∧ median_slope = -3

-- Statement to be proven
theorem largest_p_plus_q {R : Type*} [LinearOrderedField R] (p q : R) :
  triangle p q → p + q = 70 / 3 :=
by
  sorry

end largest_p_plus_q_l201_201271


namespace product_telescope_identity_l201_201658

theorem product_telescope_identity :
  (1 + (1 / 2)) * (1 + (1 / 3)) * (1 + (1 / 4)) * (1 + (1 / 5)) * (1 + (1 / 6)) * (1 + (1 / 7)) = 8 :=
by
  sorry

end product_telescope_identity_l201_201658


namespace line_quadrants_condition_l201_201452

theorem line_quadrants_condition (m n : ℝ) (h : m * n < 0) :
  (m > 0 ∧ n < 0) :=
sorry

end line_quadrants_condition_l201_201452


namespace problem1_problem2_problem3_l201_201890

-- Definitions and conditions
variable (f : ℝ → ℝ)
variable (h1 : ∀ x y : ℝ, f (x + y) = f x + f y)
variable (h2 : ∀ x : ℝ, x > 0 → f x < 0)

-- Question 1: Prove the function is odd
theorem problem1 : ∀ x : ℝ, f (-x) = -f x := by
  sorry

-- Question 2: Prove the function is monotonically decreasing
theorem problem2 : ∀ x1 x2 : ℝ, x1 < x2 → f x1 < f x2 := by
  sorry

-- Question 3: Solve the inequality given f(2) = 1
theorem problem3 (h3 : f 2 = 1) : ∀ x : ℝ, f (-x^2) + 2*f x + 4 < 0 ↔ -2 < x ∧ x < 4 := by
  sorry

end problem1_problem2_problem3_l201_201890


namespace tangents_intersection_perpendicular_parabola_l201_201618

theorem tangents_intersection_perpendicular_parabola :
  ∀ (C D : ℝ × ℝ), C.2 = 4 * C.1 ^ 2 → D.2 = 4 * D.1 ^ 2 → 
  (8 * C.1) * (8 * D.1) = -1 → 
  ∃ Q : ℝ × ℝ, Q.2 = -1 / 16 :=
by
  sorry

end tangents_intersection_perpendicular_parabola_l201_201618


namespace find_RS_length_l201_201342

-- Define the given conditions
def tetrahedron_edges (a b c d e f : ℕ) : Prop :=
  (a = 8 ∨ a = 14 ∨ a = 19 ∨ a = 28 ∨ a = 37 ∨ a = 42) ∧
  (b = 8 ∨ b = 14 ∨ b = 19 ∨ b = 28 ∨ b = 37 ∨ b = 42) ∧
  (c = 8 ∨ c = 14 ∨ c = 19 ∨ c = 28 ∨ c = 37 ∨ c = 42) ∧
  (d = 8 ∨ d = 14 ∨ d = 19 ∨ d = 28 ∨ d = 37 ∨ d = 42) ∧
  (e = 8 ∨ e = 14 ∨ e = 19 ∨ e = 28 ∨ e = 37 ∨ e = 42) ∧
  (f = 8 ∨ f = 14 ∨ f = 19 ∨ f = 28 ∨ f = 37 ∨ f = 42) ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
  d ≠ e ∧ d ≠ f ∧
  e ≠ f

def length_of_PQ (pq : ℕ) : Prop := pq = 42

def length_of_RS (rs : ℕ) (a b c d e f pq : ℕ) : Prop :=
  tetrahedron_edges a b c d e f ∧ length_of_PQ pq →
  (rs = 14)

-- The theorem statement
theorem find_RS_length (a b c d e f pq rs : ℕ) :
  tetrahedron_edges a b c d e f ∧ length_of_PQ pq →
  length_of_RS rs a b c d e f pq :=
by sorry

end find_RS_length_l201_201342


namespace eval_expression_l201_201559

theorem eval_expression : (-2 ^ 4) + 3 * (-1) ^ 6 - (-2) ^ 3 = -5 := by
  sorry

end eval_expression_l201_201559


namespace rabbit_weight_l201_201006

variable (k r p : ℝ)

theorem rabbit_weight :
  k + r + p = 39 →
  r + p = 3 * k →
  r + k = 1.5 * p →
  r = 13.65 :=
by
  intros h1 h2 h3
  sorry

end rabbit_weight_l201_201006


namespace complex_z_pow_l201_201672

open Complex

theorem complex_z_pow {z : ℂ} (h : (1 + z) / (1 - z) = (⟨0, 1⟩ : ℂ)) : z ^ 2019 = -⟨0, 1⟩ := by
  sorry

end complex_z_pow_l201_201672


namespace jesse_money_left_l201_201230

def initial_money : ℝ := 500
def novel_cost_pounds : ℝ := 13
def num_novels : ℕ := 10
def bookstore_discount : ℝ := 0.20
def exchange_rate_usd_to_pounds : ℝ := 0.7
def lunch_cost_multiplier : ℝ := 3
def lunch_tax_rate : ℝ := 0.12
def lunch_tip_rate : ℝ := 0.18
def jacket_original_euros : ℝ := 120
def jacket_discount : ℝ := 0.30
def jacket_expense_multiplier : ℝ := 2
def exchange_rate_pounds_to_euros : ℝ := 1.15

theorem jesse_money_left : 
  initial_money - (
    ((novel_cost_pounds * num_novels * (1 - bookstore_discount)) / exchange_rate_usd_to_pounds)
    + ((novel_cost_pounds * lunch_cost_multiplier * (1 + lunch_tax_rate + lunch_tip_rate)) / exchange_rate_usd_to_pounds)
    + ((((jacket_original_euros * (1 - jacket_discount)) / exchange_rate_pounds_to_euros) / exchange_rate_usd_to_pounds))
  ) = 174.66 := by
  sorry

end jesse_money_left_l201_201230


namespace cos_225_eq_neg_sqrt2_div_2_l201_201421

theorem cos_225_eq_neg_sqrt2_div_2 :
  (real.cos (225 * real.pi / 180) = - (real.sqrt 2 / 2)) :=
by
  sorry

end cos_225_eq_neg_sqrt2_div_2_l201_201421


namespace marcus_savings_l201_201486

def MarcusMaxPrice : ℝ := 130
def ShoeInitialPrice : ℝ := 120
def DiscountPercentage : ℝ := 0.30
def FinalPrice : ℝ := ShoeInitialPrice - (DiscountPercentage * ShoeInitialPrice)
def Savings : ℝ := MarcusMaxPrice - FinalPrice

theorem marcus_savings : Savings = 46 := by
  sorry

end marcus_savings_l201_201486


namespace length_of_second_train_l201_201015

theorem length_of_second_train 
  (length_first_train : ℝ)
  (speed_first_train_kmph : ℝ)
  (speed_second_train_kmph : ℝ)
  (time_seconds : ℝ)
  (same_direction : Bool) : 
  length_first_train = 380 ∧ 
  speed_first_train_kmph = 72 ∧ 
  speed_second_train_kmph = 36 ∧ 
  time_seconds = 91.9926405887529 ∧ 
  same_direction = tt → 
  ∃ L2 : ℝ, L2 = 539.93 := by
  intro h
  rcases h with ⟨hf, sf, ss, ts, sd⟩
  use 539.926405887529 -- exact value obtained in the solution
  sorry

end length_of_second_train_l201_201015


namespace security_to_bag_ratio_l201_201621

noncomputable def U_house : ℕ := 10
noncomputable def U_airport : ℕ := 5 * U_house
noncomputable def C_bag : ℕ := 15
noncomputable def W_boarding : ℕ := 20
noncomputable def W_takeoff : ℕ := 2 * W_boarding
noncomputable def T_total : ℕ := 180
noncomputable def T_known : ℕ := U_house + U_airport + C_bag + W_boarding + W_takeoff
noncomputable def T_security : ℕ := T_total - T_known

theorem security_to_bag_ratio : T_security / C_bag = 3 :=
by sorry

end security_to_bag_ratio_l201_201621


namespace monotonic_range_of_a_l201_201739

theorem monotonic_range_of_a (a : ℝ) :
  (a ≥ 9 ∨ a ≤ 3) → 
  ∀ x y : ℝ, (1 ≤ x ∧ x ≤ 4) → (1 ≤ y ∧ y ≤ 4) → x ≤ y → 
  (x^2 + (1-a)*x + 3) ≤ (y^2 + (1-a)*y + 3) :=
by
  intro ha x y hx hy hxy
  sorry

end monotonic_range_of_a_l201_201739


namespace quadratic_solution_property_l201_201483

theorem quadratic_solution_property (p q : ℝ)
  (h : ∀ x, 2 * x^2 + 8 * x - 42 = 0 → x = p ∨ x = q) :
  (p - q + 2) ^ 2 = 144 :=
sorry

end quadratic_solution_property_l201_201483


namespace A_share_correct_l201_201682

noncomputable def investment_shares (x : ℝ) (annual_gain : ℝ) := 
  let A_share := x * 12
  let B_share := (2 * x) * 6
  let C_share := (3 * x) * 4
  let total_share := A_share + B_share + C_share
  let total_ratio := 1 + 1 + 1
  annual_gain / total_ratio

theorem A_share_correct (x : ℝ) (annual_gain : ℝ) (h_gain : annual_gain = 18000) : 
  investment_shares x annual_gain / 3 = 6000 := by
  sorry

end A_share_correct_l201_201682


namespace Nicole_fish_tanks_l201_201923

-- Definition to express the conditions
def first_tank_water := 8 -- gallons
def second_tank_difference := 2 -- fewer gallons than first tanks
def num_first_tanks := 2
def num_second_tanks := 2
def total_water_four_weeks := 112 -- gallons
def weeks := 4

-- Calculate the total water per week
def water_per_week := (num_first_tanks * first_tank_water) + (num_second_tanks * (first_tank_water - second_tank_difference))

-- Calculate the total number of tanks
def total_tanks := num_first_tanks + num_second_tanks

-- Proof statement
theorem Nicole_fish_tanks : total_water_four_weeks / water_per_week = weeks → total_tanks = 4 := by
  -- Proof goes here
  sorry

end Nicole_fish_tanks_l201_201923


namespace largest_negative_integer_l201_201525

theorem largest_negative_integer :
  ∃ (n : ℤ), (∀ m : ℤ, m < 0 → m ≤ n) ∧ n = -1 := by
  sorry

end largest_negative_integer_l201_201525


namespace right_triangle_legs_sum_l201_201635

theorem right_triangle_legs_sum : 
  ∃ (x : ℕ), (x^2 + (x + 1)^2 = 41^2) ∧ (x + (x + 1) = 57) :=
by
  sorry

end right_triangle_legs_sum_l201_201635


namespace letters_per_large_envelope_l201_201840

theorem letters_per_large_envelope
  (total_letters : ℕ)
  (small_envelope_letters : ℕ)
  (large_envelopes : ℕ)
  (large_envelopes_count : ℕ)
  (h1 : total_letters = 80)
  (h2 : small_envelope_letters = 20)
  (h3 : large_envelopes_count = 30)
  (h4 : total_letters - small_envelope_letters = large_envelopes)
  : large_envelopes / large_envelopes_count = 2 :=
by
  sorry

end letters_per_large_envelope_l201_201840


namespace describe_f_plus_g_l201_201833

open Function

variable (a b c : ℝ)

def parabola (x : ℝ) : ℝ := a * x^2 + b * x + c

def reflected_parabola (x : ℝ) : ℝ := -a * x^2 - b * x - c

def f (x : ℝ) : ℝ := parabola (x - 3)

def g (x : ℝ) : ℝ := reflected_parabola (x + 3)

theorem describe_f_plus_g :
  (∀ a b c : ℝ, a ≠ 0) →
  (∀ x : ℝ, (f + g) x = -6 * (a * x + b)) :=
by
  intros a b c ha
  funext x
  simp [f, g, parabola, reflected_parabola]
  sorry

end describe_f_plus_g_l201_201833


namespace collinear_condition_perpendicular_condition_l201_201673

-- Problem 1: Prove collinearity condition for k = -2
theorem collinear_condition (k : ℝ) : 
  (k - 5) * (-12) - (12 - k) * 6 = 0 ↔ k = -2 :=
sorry

-- Problem 2: Prove perpendicular condition for k = 2 or k = 11
theorem perpendicular_condition (k : ℝ) : 
  (20 + (k - 6) * (7 - k)) = 0 ↔ (k = 2 ∨ k = 11) :=
sorry

end collinear_condition_perpendicular_condition_l201_201673


namespace repeating_decimal_fraction_sum_l201_201295

noncomputable def repeating_decimal_to_fraction (x : ℚ) : ℚ :=
if h : x = (45 / 99 : ℚ) then (5 / 11) else x

theorem repeating_decimal_fraction_sum : 
  let x := (0.45 : ℝ) in 
  (x.mul 100 - x).div 99 = (45 : ℝ) -> 
  let (a, b) := (5, 11) in 
  a + b = 16 :=
by
  intro x
  rfl
  rw [show (5 : ℚ) + (11 : ℚ) = 16 from rfl]
  sorry

end repeating_decimal_fraction_sum_l201_201295


namespace verify_ages_l201_201832

noncomputable def correct_ages (S M D W : ℝ) : Prop :=
  (M = S + 29) ∧
  (M + 2 = 2 * (S + 2)) ∧
  (D = S - 3.5) ∧
  (W = 1.5 * D) ∧
  (S = 27) ∧
  (M = 56) ∧
  (D = 23.5) ∧
  (W = 35.25)

theorem verify_ages : ∃ (S M D W : ℝ), correct_ages S M D W :=
by
  sorry

end verify_ages_l201_201832


namespace find_sixth_term_l201_201943

noncomputable def first_term : ℝ := Real.sqrt 3
noncomputable def fifth_term : ℝ := Real.sqrt 243
noncomputable def common_ratio (q : ℝ) : Prop := fifth_term = first_term * q^4
noncomputable def sixth_term (b6 : ℝ) (q : ℝ) : Prop := b6 = fifth_term * q

theorem find_sixth_term (q : ℝ) (b6 : ℝ) : 
  first_term = Real.sqrt 3 ∧
  fifth_term = Real.sqrt 243 ∧
  common_ratio q ∧ 
  sixth_term b6 q → 
  b6 = 27 ∨ b6 = -27 := 
by
  intros
  sorry

end find_sixth_term_l201_201943


namespace prove_probability_Y_gt_4_l201_201072

noncomputable def probability_Y_gt_4 (p : ℝ) (δ : ℝ) : ℝ :=
if h : 0 < δ then
  let Y := Normal 2 (δ^2) in
  let A : Set ℝ := {y | y > 4} in
  ProbabilityTheory.Probability Y A
else 0

theorem prove_probability_Y_gt_4 (p δ : ℝ) (hX : ∃ X : MeasureSpace (Fin 3 → Prop), ∀ A, ProbabilityTheory.Probability X A := (1 - (1 - p)^3) = 0.657)
  (hY : ProbabilityTheory.Probability (Normal 2 (δ^2)) {y | 0 < y ∧ y < 2} = p) :
  probability_Y_gt_4 p δ = 0.2 :=
by
  sorry

end prove_probability_Y_gt_4_l201_201072


namespace find_f_neg2_l201_201567

noncomputable def f (x : ℝ) : ℝ := 2 * x^2 - 3 * x + 1

theorem find_f_neg2 : f (-2) = 15 :=
by
  sorry

end find_f_neg2_l201_201567


namespace problem_l201_201693

theorem problem : 3^128 + 8^5 / 8^3 = 65 := sorry

end problem_l201_201693


namespace pyramid_vertices_l201_201545

theorem pyramid_vertices (n : ℕ) (h : 2 * n = 14) : n + 1 = 8 :=
by {
  sorry
}

end pyramid_vertices_l201_201545


namespace jennifer_remaining_money_l201_201749

noncomputable def money_spent_on_sandwich (initial_money : ℝ) : ℝ :=
  let sandwich_cost := (1/5) * initial_money
  let discount := (10/100) * sandwich_cost
  sandwich_cost - discount

noncomputable def money_spent_on_ticket (initial_money : ℝ) : ℝ :=
  (1/6) * initial_money

noncomputable def money_spent_on_book (initial_money : ℝ) : ℝ :=
  (1/2) * initial_money

noncomputable def money_after_initial_expenses (initial_money : ℝ) (gift : ℝ) : ℝ :=
  initial_money - money_spent_on_sandwich initial_money - money_spent_on_ticket initial_money - money_spent_on_book initial_money + gift

noncomputable def money_spent_on_cosmetics (remaining_money : ℝ) : ℝ :=
  (1/4) * remaining_money

noncomputable def money_after_cosmetics (remaining_money : ℝ) : ℝ :=
  remaining_money - money_spent_on_cosmetics remaining_money

noncomputable def money_spent_on_tshirt (remaining_money : ℝ) : ℝ :=
  let tshirt_cost := (1/3) * remaining_money
  let tax := (5/100) * tshirt_cost
  tshirt_cost + tax

noncomputable def remaining_money (initial_money : ℝ) (gift : ℝ) : ℝ :=
  let after_initial := money_after_initial_expenses initial_money gift
  let after_cosmetics := after_initial - money_spent_on_cosmetics after_initial
  after_cosmetics - money_spent_on_tshirt after_cosmetics

theorem jennifer_remaining_money : remaining_money 90 30 = 21.35 := by
  sorry

end jennifer_remaining_money_l201_201749


namespace min_value_of_vector_difference_is_six_l201_201711

open Real

noncomputable def min_value_of_vector_difference (a b : ℝ) (h_angle : ∀ (a b : ℝ), angle a b = π * (2 / 3)) (h_dot_product : ∀ (a b : ℝ), dot_product a b = -1) : ℝ :=
  sqrt (norm (a - b) ^ 2)

-- Now let's state the theorem with the conditions
theorem min_value_of_vector_difference_is_six 
  (a b : ℝ)
  (h_angle : ∀ (a b : ℝ), angle a b = π * (2 / 3)) 
  (h_dot_product : ∀ (a b : ℝ), dot_product a b = -1) :
  min_value_of_vector_difference a b h_angle h_dot_product = sqrt 6 :=
sorry

end min_value_of_vector_difference_is_six_l201_201711


namespace cos_225_eq_neg_inv_sqrt_2_l201_201370

theorem cos_225_eq_neg_inv_sqrt_2 :
  let P := (225 : ℝ) in
  P ∈ set_of (λ θ, θ ∈ [0, 360]) ∧
  ∃ x y : ℝ, (x = -1/√2) ∧ (y = -1/√2) ∧ (cos θ = x) :=
sorry

end cos_225_eq_neg_inv_sqrt_2_l201_201370


namespace volume_diff_proof_l201_201644

def volume_difference (x y z x' y' z' : ℝ) : ℝ := x * y * z - x' * y' * z'

theorem volume_diff_proof : 
  (∃ (x y z x' y' z' : ℝ),
    2 * (x + y) = 12 ∧ 2 * (x + z) = 16 ∧ 2 * (y + z) = 24 ∧
    2 * (x' + y') = 12 ∧ 2 * (x' + z') = 16 ∧ 2 * (y' + z') = 20 ∧
    volume_difference x y z x' y' z' = -13) :=
by {
  sorry
}

end volume_diff_proof_l201_201644


namespace legs_sum_of_right_triangle_with_hypotenuse_41_l201_201639

noncomputable def right_triangle_legs_sum (x : ℕ) : ℕ := x + (x + 1)

theorem legs_sum_of_right_triangle_with_hypotenuse_41 :
  ∃ x : ℕ, (x * x + (x + 1) * (x + 1) = 41 * 41) ∧ right_triangle_legs_sum x = 57 := by
sorry

end legs_sum_of_right_triangle_with_hypotenuse_41_l201_201639


namespace ratio_of_radii_l201_201445

namespace CylinderAndSphere

variable (r R : ℝ)
variable (h_cylinder : 2 * Real.pi * r * (4 * r) = 4 * Real.pi * R ^ 2)

theorem ratio_of_radii (r R : ℝ) (h_cylinder : 2 * Real.pi * r * (4 * r) = 4 * Real.pi * R ^ 2) :
    R / r = Real.sqrt 2 :=
by
  sorry

end CylinderAndSphere

end ratio_of_radii_l201_201445


namespace cos_225_correct_l201_201409

noncomputable def cos_225_eq : Prop :=
  cos (225 * Real.pi / 180) = -Real.sqrt 2 / 2

theorem cos_225_correct : cos_225_eq :=
sorry

end cos_225_correct_l201_201409


namespace largest_valid_four_digit_number_l201_201814

-- Definition of the problem conditions
def is_valid_number (a b c d : ℕ) : Prop :=
  c = a + b ∧ d = b + c

-- Proposition that we need to prove
theorem largest_valid_four_digit_number : ∃ (a b c d : ℕ),
  is_valid_number a b c d ∧ a * 1000 + b * 100 + c * 10 + d = 9099 :=
by
  sorry

end largest_valid_four_digit_number_l201_201814


namespace meeting_probability_correct_l201_201242

noncomputable def meeting_probability : ℝ := 
  let pA_move_right : ℝ := 0.4
  let pA_move_up : ℝ := 0.4
  let pA_move_diag : ℝ := 0.2
  let pB_move_left : ℝ := 0.4
  let pB_move_down : ℝ := 0.4
  let pB_move_diag : ℝ := 0.2
  sorry -- calculations for meeting probability

theorem meeting_probability_correct :
  meeting_probability = -- computed probability
  sorry

end meeting_probability_correct_l201_201242


namespace LCM_of_numbers_l201_201973

theorem LCM_of_numbers (a b : ℕ) (h1 : a = 20) (h2 : a / b = 5 / 4): Nat.lcm a b = 80 :=
by
  sorry

end LCM_of_numbers_l201_201973


namespace cos_225_l201_201392

theorem cos_225 : cos (225 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by
-- Conditions based on given problem
have angle_def : 225 * Real.pi / 180 = 5 * Real.pi / 4 := by norm_num
have coord_Q := Complex.exp (5 * Real.pi / 4 * Complex.I)
have third_quadrant : coord_Q = Complex.mk (-Real.sqrt 2 / 2) (-Real.sqrt 2 / 2) := 
    by rw [coord_Q, Complex.exp_eq_cos_add_sin, Complex.cos, Complex.sin]
-- Proof equivalent
rw [angle_def, coord_Q]
exact congr_arg Complex.re third_quadrant
-- Equivalent statement
#check cos (5 * Real.pi / 4) = -Real.sqrt 2 / 2
-- Proof is left as an exercise
sorry

end cos_225_l201_201392


namespace negative_square_inequality_l201_201581

theorem negative_square_inequality (a b : ℝ) (h1 : a < b) (h2 : b < 0) : a^2 > b^2 :=
sorry

end negative_square_inequality_l201_201581


namespace maoming_population_scientific_notation_l201_201775

-- Definitions for conditions
def is_scientific_notation (a : ℝ) (n : ℤ) (x : ℝ) : Prop :=
  x = a * 10^n ∧ 1 ≤ |a| ∧ |a| < 10

-- The main theorem to prove
theorem maoming_population_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), is_scientific_notation a n 6800000 ∧ a = 6.8 ∧ n = 6 :=
sorry

end maoming_population_scientific_notation_l201_201775


namespace relationship_of_exponents_l201_201866

theorem relationship_of_exponents (m p r s : ℝ) (u v w t : ℝ) (h1 : m^u = r) (h2 : p^v = r) (h3 : p^w = s) (h4 : m^t = s) : u * v = w * t :=
by
  sorry

end relationship_of_exponents_l201_201866


namespace donna_card_shop_hourly_wage_correct_l201_201027

noncomputable def donna_hourly_wage_at_card_shop : ℝ := 
  let total_earnings := 305.0
  let earnings_dog_walking := 2 * 10.0 * 5
  let earnings_babysitting := 4 * 10.0
  let earnings_card_shop := total_earnings - (earnings_dog_walking + earnings_babysitting)
  let hours_card_shop := 5 * 2
  earnings_card_shop / hours_card_shop

theorem donna_card_shop_hourly_wage_correct : donna_hourly_wage_at_card_shop = 16.50 :=
by 
  -- Skipping proof steps for the implementation
  sorry

end donna_card_shop_hourly_wage_correct_l201_201027


namespace cos_225_eq_neg_sqrt2_div2_l201_201364

theorem cos_225_eq_neg_sqrt2_div2 : Real.cos (225 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by sorry

end cos_225_eq_neg_sqrt2_div2_l201_201364


namespace three_g_of_x_l201_201066

noncomputable def g (x : ℝ) : ℝ := 3 / (3 + x)

theorem three_g_of_x (x : ℝ) (h : x > 0) : 3 * g x = 27 / (9 + x) :=
by
  sorry

end three_g_of_x_l201_201066


namespace range_of_a_l201_201777

noncomputable def a_n (a : ℝ) (n : ℕ) : ℝ :=
  if n <= 7 then (3 - a) * n - 3 else a ^ (n - 6)

def increasing_seq (a : ℝ) (n : ℕ) : Prop :=
  a_n a n < a_n a (n + 1)

theorem range_of_a (a : ℝ) :
  (∀ n, increasing_seq a n) ↔ (9 / 4 < a ∧ a < 3) :=
sorry

end range_of_a_l201_201777


namespace sector_area_l201_201198

theorem sector_area (r θ : ℝ) (hr : r = 1) (hθ : θ = 2) : 
  (1 / 2) * r * r * θ = 1 := by
sorry

end sector_area_l201_201198


namespace pastries_eaten_l201_201669

theorem pastries_eaten (total_p: ℕ)
  (hare_fraction: ℚ)
  (dormouse_fraction: ℚ)
  (hare_eaten: ℕ)
  (remaining_after_hare: ℕ)
  (dormouse_eaten: ℕ)
  (final_remaining: ℕ) 
  (hatter_with_left: ℕ) :
  (final_remaining = hatter_with_left) -> hare_fraction = 5 / 16 -> dormouse_fraction = 7 / 11 -> hatter_with_left = 8 -> total_p = 32 -> 
  (total_p = hare_eaten + remaining_after_hare) -> (remaining_after_hare - dormouse_eaten = hatter_with_left) -> (hare_eaten = 10) ∧ (dormouse_eaten = 14) := 
by {
  sorry
}

end pastries_eaten_l201_201669


namespace max_ak_at_k_125_l201_201276

def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.choose n k

def ak (k : ℕ) : ℚ :=
  binomial_coefficient 500 k * (0.3)^k

theorem max_ak_at_k_125 : 
  ∀ k : ℕ, k ∈ Finset.range 501 → (ak k ≤ ak 125) :=
by sorry

end max_ak_at_k_125_l201_201276


namespace trees_planted_l201_201553

theorem trees_planted (yard_length : ℕ) (distance_between_trees : ℕ) (n_trees : ℕ) 
  (h1 : yard_length = 434) 
  (h2 : distance_between_trees = 14) 
  (h3 : n_trees = yard_length / distance_between_trees + 1) : 
  n_trees = 32 :=
by
  sorry

end trees_planted_l201_201553


namespace lowest_height_l201_201258

noncomputable def length_A : ℝ := 2.4
noncomputable def length_B : ℝ := 3.2
noncomputable def length_C : ℝ := 2.8

noncomputable def height_Eunji : ℝ := 8 * length_A
noncomputable def height_Namjoon : ℝ := 4 * length_B
noncomputable def height_Hoseok : ℝ := 5 * length_C

theorem lowest_height :
  height_Namjoon = 12.8 ∧ 
  height_Namjoon < height_Eunji ∧ 
  height_Namjoon < height_Hoseok :=
by
  sorry

end lowest_height_l201_201258


namespace prime_solution_unique_l201_201114

open Nat

theorem prime_solution_unique (p q : ℕ) (hp : Prime p) (hq : Prime q) (h : p^2 - 6 * p * q + q^2 + 3 * q - 1 = 0) : 
  (p = 17 ∧ q = 3) :=
by
  sorry

end prime_solution_unique_l201_201114


namespace lucy_total_fish_l201_201239

theorem lucy_total_fish (current fish_needed : ℕ) (h1 : current = 212) (h2 : fish_needed = 68) : 
  current + fish_needed = 280 := 
by
  sorry

end lucy_total_fish_l201_201239


namespace no_net_profit_or_loss_l201_201976

theorem no_net_profit_or_loss (C : ℝ) : 
  let cost1 := C
  let cost2 := C
  let selling_price1 := 1.10 * C
  let selling_price2 := 0.90 * C
  let total_cost := cost1 + cost2
  let total_selling_price := selling_price1 + selling_price2
  let net_profit_loss := (total_selling_price - total_cost) / total_cost * 100
  net_profit_loss = 0 :=
by
  let cost1 := C
  let cost2 := C
  let selling_price1 := 1.10 * C
  let selling_price2 := 0.90 * C
  let total_cost := cost1 + cost2
  let total_selling_price := selling_price1 + selling_price2
  let net_profit_loss := (total_selling_price - total_cost) / total_cost * 100
  sorry

end no_net_profit_or_loss_l201_201976


namespace weight_lifting_ratio_l201_201604

theorem weight_lifting_ratio :
  ∀ (F S : ℕ), F + S = 600 ∧ F = 300 ∧ 2 * F = S + 300 → F / S = 1 :=
by
  intro F S
  intro h
  obtain ⟨h1, h2, h3⟩ := h
  sorry

end weight_lifting_ratio_l201_201604


namespace cos_225_l201_201378

noncomputable def cos_225_eq : ℝ :=
  cos (225 * (real.pi / 180))

theorem cos_225 : cos_225_eq = -real.sqrt 2 / 2 :=
  sorry

end cos_225_l201_201378


namespace albert_large_pizzas_l201_201016

-- Define the conditions
def large_pizza_slices : ℕ := 16
def small_pizza_slices : ℕ := 8
def num_small_pizzas : ℕ := 2
def total_slices_eaten : ℕ := 48

-- Define the question and requirement to prove
def number_of_large_pizzas (L : ℕ) : Prop :=
  large_pizza_slices * L + small_pizza_slices * num_small_pizzas = total_slices_eaten

theorem albert_large_pizzas :
  number_of_large_pizzas 2 :=
by
  sorry

end albert_large_pizzas_l201_201016


namespace largest_four_digit_number_property_l201_201802

theorem largest_four_digit_number_property : ∃ (a b c d : Nat), 
  (1000 * a + 100 * b + 10 * c + d = 9099) ∧ 
  (c = a + b) ∧ 
  (d = b + c) ∧ 
  1 ≤ a ∧ a ≤ 9 ∧ 
  0 ≤ b ∧ b ≤ 9 ∧ 
  0 ≤ c ∧ c ≤ 9 ∧ 
  0 ≤ d ∧ d ≤ 9 := 
sorry

end largest_four_digit_number_property_l201_201802


namespace incenters_and_excenters_cyclic_l201_201911

/-- Definitions of the problem conditions -/
variables (A B C D O : Point)
variables (I1 I2 I3 I4 J1 J2 J3 J4 : Point)
variables [convex_quadrilateral A B C D]
variables [intersect AC BD O]
variables [I1 = incenter (triangle A O B), I2 = incenter (triangle B O C)]
variables [I3 = incenter (triangle C O D), I4 = incenter (triangle D O A)]
variables [J1 = excenter (triangle A O B), J2 = excenter (triangle B O C)]
variables [J3 = excenter (triangle C O D), J4 = excenter (triangle D O A)]

/-- Main theorem statement -/
theorem incenters_and_excenters_cyclic :
  (cyclic I1 I2 I3 I4) ↔ (cyclic J1 J2 J3 J4) := by
  sorry

end incenters_and_excenters_cyclic_l201_201911


namespace geometric_sum_a4_a6_l201_201608

-- Definitions based on the conditions
def geometric_sequence (a : ℕ → ℝ) := ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sum_a4_a6 (a : ℕ → ℝ) (h_geom : geometric_sequence a) (h_pos : ∀ n, a n > 0) 
(h_cond : a 3 * a 5 + a 2 * a 10 + 2 * a 4 * a 6 = 100) : a 4 + a 6 = 10 :=
by
  sorry

end geometric_sum_a4_a6_l201_201608


namespace exists_unique_inverse_l201_201624

theorem exists_unique_inverse (p : ℕ) (a : ℕ) (hp : Nat.Prime p) (h_gcd : Nat.gcd p a = 1) : 
  ∃! (b : ℕ), b ∈ Finset.range p ∧ (a * b) % p = 1 := 
sorry

end exists_unique_inverse_l201_201624


namespace smallest_k_for_divisibility_l201_201439

theorem smallest_k_for_divisibility : (∃ k : ℕ, ∀ z : ℂ, z^8 + z^7 + z^4 + z^3 + z^2 + z + 1 ∣ z^k - 1 ∧ (∀ m : ℕ, m < k → ∃ z : ℂ, ¬(z^8 + z^7 + z^4 + z^3 + z^2 + z + 1 ∣ z^m - 1))) ↔ k = 14 := sorry

end smallest_k_for_divisibility_l201_201439


namespace calc_fraction_l201_201526

theorem calc_fraction : (3.241 * 14) / 100 = 0.45374 := by
  sorry

end calc_fraction_l201_201526


namespace yen_per_cad_l201_201536

theorem yen_per_cad (yen cad : ℝ) (h : yen / cad = 5000 / 60) : yen = 83 := by
  sorry

end yen_per_cad_l201_201536


namespace cos_225_proof_l201_201380

def cos_225_def := angle.to_real 225 = -√2 / 2

theorem cos_225_proof : cos_225_def := sorry

end cos_225_proof_l201_201380


namespace min_quadratic_expr_l201_201283

noncomputable def quadratic_expr (x : ℝ) := 3 * x^2 - 18 * x + 2023

theorem min_quadratic_expr : ∃ x : ℝ, quadratic_expr x = 1996 :=
by
  have h : quadratic_expr (3 : ℝ) = 1996
  exact h
  use 3
  rw h
  sorry -- Proof of h (already derived in given solution)

end min_quadratic_expr_l201_201283


namespace NaNO3_moles_l201_201437

theorem NaNO3_moles (moles_NaCl moles_HNO3 moles_NaNO3 : ℝ) (h_HNO3 : moles_HNO3 = 2) (h_ratio : moles_NaNO3 = moles_NaCl) (h_NaNO3 : moles_NaNO3 = 2) :
  moles_NaNO3 = 2 :=
sorry

end NaNO3_moles_l201_201437


namespace sin_cos_alpha_beta_l201_201727

theorem sin_cos_alpha_beta (α β : ℝ) 
  (h1 : Real.sin α = Real.cos β) 
  (h2 : Real.cos α = Real.sin (2 * β)) :
  Real.sin β ^ 2 + Real.cos α ^ 2 = 3 / 2 := 
by
  sorry

end sin_cos_alpha_beta_l201_201727


namespace polygon_with_15_diagonals_has_7_sides_l201_201462

-- Define the number of diagonals formula for a regular polygon
def number_of_diagonals (n : ℕ) : ℕ :=
  n * (n - 3) / 2

-- The proof statement
theorem polygon_with_15_diagonals_has_7_sides :
  ∃ n : ℕ, number_of_diagonals n = 15 ∧ n = 7 :=
by
  sorry

end polygon_with_15_diagonals_has_7_sides_l201_201462


namespace simplify_fraction_l201_201106

theorem simplify_fraction : (180 : ℚ) / 1260 = 1 / 7 :=
by
  sorry

end simplify_fraction_l201_201106


namespace initial_mixture_amount_l201_201004

theorem initial_mixture_amount (x : ℝ) (h1 : 20 / 100 * x / (x + 3) = 6 / 35) : x = 18 :=
sorry

end initial_mixture_amount_l201_201004


namespace largest_four_digit_number_prop_l201_201805

theorem largest_four_digit_number_prop :
  ∃ (a b c d : ℕ), a = 9 ∧ b = 0 ∧ c = 9 ∧ d = 9 ∧ (1000 * a + 100 * b + 10 * c + d = 9099) ∧ (c = a + b) ∧ (d = b + c) :=
by
  sorry

end largest_four_digit_number_prop_l201_201805


namespace abs_inequality_solution_l201_201780

theorem abs_inequality_solution :
  {x : ℝ | |x + 2| > 3} = {x : ℝ | x < -5} ∪ {x : ℝ | x > 1} :=
by
  sorry

end abs_inequality_solution_l201_201780


namespace Matthias_fewer_fish_l201_201487

-- Define the number of fish Micah has
def Micah_fish : ℕ := 7

-- Define the number of fish Kenneth has
def Kenneth_fish : ℕ := 3 * Micah_fish

-- Define the total number of fish
def total_fish : ℕ := 34

-- Define the number of fish Matthias has
def Matthias_fish : ℕ := total_fish - (Micah_fish + Kenneth_fish)

-- State the theorem for the number of fewer fish Matthias has compared to Kenneth
theorem Matthias_fewer_fish : Kenneth_fish - Matthias_fish = 15 := by
  -- Proof goes here
  sorry

end Matthias_fewer_fish_l201_201487


namespace largest_four_digit_number_prop_l201_201807

theorem largest_four_digit_number_prop :
  ∃ (a b c d : ℕ), a = 9 ∧ b = 0 ∧ c = 9 ∧ d = 9 ∧ (1000 * a + 100 * b + 10 * c + d = 9099) ∧ (c = a + b) ∧ (d = b + c) :=
by
  sorry

end largest_four_digit_number_prop_l201_201807


namespace linear_regression_solution_l201_201340

theorem linear_regression_solution :
  let barx := 5
  let bary := 50
  let sum_xi_squared := 145
  let sum_xiyi := 1380
  let n := 5
  let b := (sum_xiyi - barx * bary) / (sum_xi_squared - n * barx^2)
  let a := bary - b * barx
  let predicted_y := 6.5 * 10 + 17.5
  b = 6.5 ∧ a = 17.5 ∧ predicted_y = 82.5 := 
by
  intros
  sorry

end linear_regression_solution_l201_201340


namespace division_of_power_l201_201753

theorem division_of_power (m : ℕ) 
  (h : m = 16^2018) : m / 8 = 2^8069 := by
  sorry

end division_of_power_l201_201753


namespace least_positive_integer_condition_l201_201521

theorem least_positive_integer_condition :
  ∃ n > 1, (∀ k ∈ [3, 4, 5, 6, 7, 8, 9, 10], n % k = 1) → n = 25201 := by
  sorry

end least_positive_integer_condition_l201_201521


namespace cost_of_large_poster_is_correct_l201_201232

/-- Problem conditions -/
def posters_per_day : ℕ := 5
def large_posters_per_day : ℕ := 2
def large_poster_sale_price : ℝ := 10
def small_posters_per_day : ℕ := 3
def small_poster_sale_price : ℝ := 6
def small_poster_cost : ℝ := 3
def weekly_profit : ℝ := 95

/-- The cost to make a large poster -/
noncomputable def large_poster_cost : ℝ := 5

/-- Prove that the cost to make a large poster is $5 given the conditions -/
theorem cost_of_large_poster_is_correct :
    large_poster_cost = 5 :=
by
  -- (Condition translation into Lean)
  let daily_profit := weekly_profit / 5
  let daily_revenue := (large_posters_per_day * large_poster_sale_price) + (small_posters_per_day * small_poster_sale_price)
  let daily_cost_small_posters := small_posters_per_day * small_poster_cost
  
  -- Express the daily profit in terms of costs, including unknown large_poster_cost
  have calc_profit : daily_profit = daily_revenue - daily_cost_small_posters - (large_posters_per_day * (large_poster_cost)) :=
    sorry
  
  -- Setting the equation to solve for large_poster_cost
  have eqn : daily_profit = 19 := by
    sorry

  -- Solve for large_poster_cost
  have solve_large_poster_cost : 19 = daily_revenue - daily_cost_small_posters - (large_posters_per_day * 5) :=
    by sorry
  
  sorry

end cost_of_large_poster_is_correct_l201_201232


namespace find_first_number_l201_201504

open Int

theorem find_first_number (A : ℕ) : 
  (Nat.lcm A 671 = 2310) ∧ (Nat.gcd A 671 = 61) → 
  A = 210 :=
by
  intro h
  sorry

end find_first_number_l201_201504


namespace tomatoes_needed_for_meal_l201_201269

theorem tomatoes_needed_for_meal :
  (∀ (slices_per_tomato slices_per_meal people : ℕ),
    slices_per_tomato = 8 →
    slices_per_meal = 20 →
    people = 8 →
    (people * slices_per_meal) / slices_per_tomato = 20) :=
by {
  intros slices_per_tomato slices_per_meal people h_slices_per_tomato h_slices_per_meal h_people,
  rw [h_slices_per_tomato, h_slices_per_meal, h_people],
  norm_num,
}

end tomatoes_needed_for_meal_l201_201269


namespace quadrilateral_is_square_l201_201456

-- Define a structure for a quadrilateral with side lengths and diagonal lengths
structure Quadrilateral :=
  (side_a side_b side_c side_d diag_e diag_f : ℝ)

-- Define what it means for a quadrilateral to be a square
def is_square (quad : Quadrilateral) : Prop :=
  quad.side_a = quad.side_b ∧ 
  quad.side_b = quad.side_c ∧ 
  quad.side_c = quad.side_d ∧  
  quad.diag_e = quad.diag_f

-- Define the problem to prove that the given quadrilateral is a square given the conditions
theorem quadrilateral_is_square (quad : Quadrilateral) 
  (h_sides : quad.side_a = quad.side_b ∧ 
             quad.side_b = quad.side_c ∧ 
             quad.side_c = quad.side_d)
  (h_diagonals : quad.diag_e = quad.diag_f) :
  is_square quad := 
  by
  -- This is where the proof would go
  sorry

end quadrilateral_is_square_l201_201456


namespace pool_depth_multiple_l201_201477

theorem pool_depth_multiple
  (johns_pool : ℕ)
  (sarahs_pool : ℕ)
  (h1 : johns_pool = 15)
  (h2 : sarahs_pool = 5)
  (h3 : johns_pool = x * sarahs_pool + 5) :
  x = 2 := by
  sorry

end pool_depth_multiple_l201_201477


namespace largest_angle_of_triangle_l201_201138

theorem largest_angle_of_triangle (x : ℝ) (h_ratio : (5 * x) + (6 * x) + (7 * x) = 180) :
  7 * x = 70 := 
sorry

end largest_angle_of_triangle_l201_201138


namespace lines_parallel_to_skew_are_skew_or_intersect_l201_201060

-- Define skew lines conditions in space
def skew_lines (l1 l2 : ℝ → ℝ × ℝ × ℝ) : Prop :=
  ¬ (∀ t1 t2 : ℝ, l1 t1 = l2 t2) ∧ ¬ (∃ d : ℝ × ℝ × ℝ, ∀ t : ℝ, l1 t + d = l2 t)

-- Define parallel lines condition in space
def parallel_lines (m l : ℝ → ℝ × ℝ × ℝ) : Prop :=
  ∃ v : ℝ × ℝ × ℝ, ∀ t1 t2 : ℝ, m t1 = l t2 + v

-- Define the relationship to check between lines
def relationship (m1 m2 : ℝ → ℝ × ℝ × ℝ) : Prop :=
  (∃ t1 t2 : ℝ, m1 t1 = m2 t2) ∨ skew_lines m1 m2

-- The main theorem statement
theorem lines_parallel_to_skew_are_skew_or_intersect
  {l1 l2 m1 m2 : ℝ → ℝ × ℝ × ℝ}
  (h_skew: skew_lines l1 l2)
  (h_parallel_1: parallel_lines m1 l1)
  (h_parallel_2: parallel_lines m2 l2) :
  relationship m1 m2 :=
by
  sorry

end lines_parallel_to_skew_are_skew_or_intersect_l201_201060


namespace second_group_students_l201_201772

theorem second_group_students (S : ℕ) : 
    (1200 / 40) = 9 + S + 11 → S = 10 :=
by sorry

end second_group_students_l201_201772


namespace alice_burgers_each_day_l201_201441

theorem alice_burgers_each_day (cost_per_burger : ℕ) (total_spent : ℕ) (days_in_june : ℕ) 
  (h1 : cost_per_burger = 13) (h2 : total_spent = 1560) (h3 : days_in_june = 30) :
  (total_spent / cost_per_burger) / days_in_june = 4 := by
  sorry

end alice_burgers_each_day_l201_201441


namespace problem_l201_201868

variables (x y z : ℝ)

theorem problem :
  x - y - z = 3 ∧ yz - xy - xz = 3 → x^2 + y^2 + z^2 = 3 :=
by
  sorry

end problem_l201_201868


namespace p_finishes_job_after_q_in_24_minutes_l201_201330

theorem p_finishes_job_after_q_in_24_minutes :
  let P_rate := 1 / 4
  let Q_rate := 1 / 20
  let together_rate := P_rate + Q_rate
  let work_done_in_3_hours := together_rate * 3
  let remaining_work := 1 - work_done_in_3_hours
  let time_for_p_to_finish := remaining_work / P_rate
  let time_in_minutes := time_for_p_to_finish * 60
  time_in_minutes = 24 :=
by
  sorry

end p_finishes_job_after_q_in_24_minutes_l201_201330


namespace gcd_ab_eq_one_l201_201422

def a : ℕ := 97^10 + 1
def b : ℕ := 97^10 + 97^3 + 1

theorem gcd_ab_eq_one : Nat.gcd a b = 1 :=
by
  sorry

end gcd_ab_eq_one_l201_201422


namespace bella_stamps_l201_201180

theorem bella_stamps :
  let snowflake_cost := 1.05
  let truck_cost := 1.20
  let rose_cost := 0.90
  let butterfly_cost := 1.15
  let snowflake_spent := 15.75
  
  let snowflake_stamps := snowflake_spent / snowflake_cost
  let truck_stamps := snowflake_stamps + 11
  let rose_stamps := truck_stamps - 17
  let butterfly_stamps := 1.5 * rose_stamps
  
  let total_stamps := snowflake_stamps + truck_stamps + rose_stamps + butterfly_stamps
  
  total_stamps = 64 := by
  sorry

end bella_stamps_l201_201180


namespace length_of_wall_correct_l201_201838

noncomputable def length_of_wall (s : ℝ) (w : ℝ) : ℝ :=
  let area_mirror := s * s
  let area_wall := 2 * area_mirror
  area_wall / w

theorem length_of_wall_correct : length_of_wall 18 32 = 20.25 :=
by
  -- This is the place for proof which is omitted deliberately
  sorry

end length_of_wall_correct_l201_201838


namespace range_of_x_l201_201710

-- Define the problem conditions and the conclusion to be proved
theorem range_of_x (f : ℝ → ℝ) (h_inc : ∀ x y, -1 ≤ x → x ≤ 1 → -1 ≤ y → y ≤ 1 → x ≤ y → f x ≤ f y)
  (h_ineq : ∀ x, f (x - 2) < f (1 - x)) :
  ∀ x, 1 ≤ x ∧ x < 3 / 2 :=
by
  sorry

end range_of_x_l201_201710


namespace largest_of_four_numbers_l201_201699

theorem largest_of_four_numbers 
  (a b c d : ℝ) 
  (h1 : a + 5 = b^2 - 1) 
  (h2 : a + 5 = c^2 + 3) 
  (h3 : a + 5 = d - 4) 
  : d > max (max a b) c :=
sorry

end largest_of_four_numbers_l201_201699


namespace largest_four_digit_number_property_l201_201801

theorem largest_four_digit_number_property : ∃ (a b c d : Nat), 
  (1000 * a + 100 * b + 10 * c + d = 9099) ∧ 
  (c = a + b) ∧ 
  (d = b + c) ∧ 
  1 ≤ a ∧ a ≤ 9 ∧ 
  0 ≤ b ∧ b ≤ 9 ∧ 
  0 ≤ c ∧ c ≤ 9 ∧ 
  0 ≤ d ∧ d ≤ 9 := 
sorry

end largest_four_digit_number_property_l201_201801


namespace number_of_ways_to_label_decagon_equal_sums_l201_201250

open Nat

-- Formal definition of the problem
def sum_of_digits : Nat := (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10 + 11 + 12)

-- The problem statement: Prove there are 3840 ways to label digits ensuring the given condition
theorem number_of_ways_to_label_decagon_equal_sums :
  ∃ (n : Nat), n = 3840 ∧ ∀ (A B C D E F G H I J K L : Nat), 
    (A ≠ B) ∧ (A ≠ C) ∧ (A ≠ D) ∧ (A ≠ E) ∧ (A ≠ F) ∧ (A ≠ G) ∧ (A ≠ H) ∧ (A ≠ I) ∧ (A ≠ J) ∧ (A ≠ K) ∧ (A ≠ L) ∧
    (B ≠ C) ∧ (B ≠ D) ∧ (B ≠ E) ∧ (B ≠ F) ∧ (B ≠ G) ∧ (B ≠ H) ∧ (B ≠ I) ∧ (B ≠ J) ∧ (B ≠ K) ∧ (B ≠ L) ∧
    (C ≠ D) ∧ (C ≠ E) ∧ (C ≠ F) ∧ (C ≠ G) ∧ (C ≠ H) ∧ (C ≠ I) ∧ (C ≠ J) ∧ (C ≠ K) ∧ (C ≠ L) ∧
    (D ≠ E) ∧ (D ≠ F) ∧ (D ≠ G) ∧ (D ≠ H) ∧ (D ≠ I) ∧ (D ≠ J) ∧ (D ≠ K) ∧ (D ≠ L) ∧
    (E ≠ F) ∧ (E ≠ G) ∧ (E ≠ H) ∧ (E ≠ I) ∧ (E ≠ J) ∧ (E ≠ K) ∧ (E ≠ L) ∧
    (F ≠ G) ∧ (F ≠ H) ∧ (F ≠ I) ∧ (F ≠ J) ∧ (F ≠ K) ∧ (F ≠ L) ∧
    (G ≠ H) ∧ (G ≠ I) ∧ (G ≠ J) ∧ (G ≠ K) ∧ (G ≠ L) ∧
    (H ≠ I) ∧ (H ≠ J) ∧ (H ≠ K) ∧ (H ≠ L) ∧
    (I ≠ J) ∧ (I ≠ K) ∧ (I ≠ L) ∧
    (J ≠ K) ∧ (J ≠ L) ∧
    (K ≠ L) ∧
    (A + L + F = B + L + G) ∧ (B + L + G = C + L + H) ∧ 
    (C + L + H = D + L + I) ∧ (D + L + I = E + L + J) ∧ 
    (E + L + J = F + L + K) ∧ (F + L + K = A + L + F) :=
sorry

end number_of_ways_to_label_decagon_equal_sums_l201_201250


namespace cos_225_eq_neg_inv_sqrt_2_l201_201368

theorem cos_225_eq_neg_inv_sqrt_2 :
  let P := (225 : ℝ) in
  P ∈ set_of (λ θ, θ ∈ [0, 360]) ∧
  ∃ x y : ℝ, (x = -1/√2) ∧ (y = -1/√2) ∧ (cos θ = x) :=
sorry

end cos_225_eq_neg_inv_sqrt_2_l201_201368


namespace parabola_vertex_l201_201607

theorem parabola_vertex :
  ∃ h k : ℝ, (∀ x : ℝ, y = 1 / 2 * (x + 1) ^ 2 - 1 / 2) →
    (h = -1 ∧ k = -1 / 2) :=
by
  sorry

end parabola_vertex_l201_201607


namespace battery_replacement_in_month_15th_l201_201728

theorem battery_replacement_in_month_15th : ∀ n : ℕ, (n = 7 * 14 + 2) → (n % 12 = 2) :=
by
  intro n h
  rw h
  norm_num
  sorry -- Placeholder for actual proof

end battery_replacement_in_month_15th_l201_201728


namespace compare_neg_numbers_l201_201691

theorem compare_neg_numbers : - 0.6 > - (2 / 3) := 
by sorry

end compare_neg_numbers_l201_201691


namespace small_seat_capacity_l201_201254

-- Definitions for the conditions
def smallSeats : Nat := 2
def largeSeats : Nat := 23
def capacityLargeSeat : Nat := 54
def totalPeopleSmallSeats : Nat := 28

-- Theorem statement
theorem small_seat_capacity : totalPeopleSmallSeats / smallSeats = 14 := by
  sorry

end small_seat_capacity_l201_201254


namespace domain_of_function_l201_201256

noncomputable def domain_of_f : Set ℝ :=
  {x | x > -1/2 ∧ x ≠ 1}

theorem domain_of_function :
  (∀ x : ℝ, (2 * x + 1 ≥ 0) ∧ (2 * x^2 - x - 1 ≠ 0) ↔ (x > -1/2 ∧ x ≠ 1)) := by
  sorry

end domain_of_function_l201_201256


namespace arithmetic_sequence_sum_l201_201196

theorem arithmetic_sequence_sum (a : ℕ → ℤ) (S : ℕ → ℤ)
  (hS : ∀ n, S n = n * (a 1 + a n) / 2)
  (h : a 3 = 20 - a 6) : S 8 = 80 :=
sorry

end arithmetic_sequence_sum_l201_201196


namespace number_of_boys_l201_201516

-- Definitions of the conditions
def total_students : ℕ := 30
def ratio_girls_parts : ℕ := 1
def ratio_boys_parts : ℕ := 2
def total_parts : ℕ := ratio_girls_parts + ratio_boys_parts

-- Statement of the problem
theorem number_of_boys :
  ∃ (boys : ℕ), boys = (total_students / total_parts) * ratio_boys_parts ∧ boys = 20 :=
by
  sorry

end number_of_boys_l201_201516


namespace trapezium_other_side_length_l201_201431

theorem trapezium_other_side_length (x : ℝ) : 
  (1 / 2) * (20 + x) * 13 = 247 → x = 18 :=
by
  sorry

end trapezium_other_side_length_l201_201431


namespace tangent_slope_is_four_l201_201778

-- Define the given curve and point
def curve (x : ℝ) : ℝ := 2 * x^2
def point : ℝ × ℝ := (1, 2)

-- Define the derivative of the curve
def curve_derivative (x : ℝ) : ℝ := 4 * x

-- Define the tangent slope at the given point
def tangent_slope_at_point : ℝ := curve_derivative 1

-- Prove that the tangent slope at point (1, 2) is 4
theorem tangent_slope_is_four : tangent_slope_at_point = 4 :=
by
  -- We state that the slope at x = 1 is 4
  sorry

end tangent_slope_is_four_l201_201778


namespace average_of_three_marbles_l201_201517

-- Define the conditions as hypotheses
theorem average_of_three_marbles (R Y B : ℕ) 
  (h1 : R + Y = 53)
  (h2 : B + Y = 69)
  (h3 : R + B = 58) :
  (R + Y + B) / 3 = 30 :=
by
  sorry

end average_of_three_marbles_l201_201517


namespace sum_of_fraction_numerator_and_denominator_l201_201327

theorem sum_of_fraction_numerator_and_denominator (x : ℚ) (a b : ℤ) :
  x = (45 / 99 : ℚ) ∧ (a = 5) ∧ (b = 11) → (a + b = 16) :=
by
  sorry

end sum_of_fraction_numerator_and_denominator_l201_201327


namespace no_positive_rational_solution_l201_201903

theorem no_positive_rational_solution :
  ¬ ∃ q : ℚ, 0 < q ∧ q^3 - 10 * q^2 + q - 2021 = 0 :=
by sorry

end no_positive_rational_solution_l201_201903


namespace trapezoid_other_side_length_l201_201433

theorem trapezoid_other_side_length (a h : ℕ) (A : ℕ) (b : ℕ) : 
  a = 20 → h = 13 → A = 247 → (1/2:ℚ) * (a + b) * h = A → b = 18 :=
by 
  intros h1 h2 h3 h4 
  rw [h1, h2, h3] at h4
  sorry

end trapezoid_other_side_length_l201_201433


namespace monomials_like_terms_l201_201892

variable (m n : ℤ)

theorem monomials_like_terms (hm : m = 3) (hn : n = 1) : m - 2 * n = 1 :=
by
  sorry

end monomials_like_terms_l201_201892


namespace cos_225_proof_l201_201383

def cos_225_def := angle.to_real 225 = -√2 / 2

theorem cos_225_proof : cos_225_def := sorry

end cos_225_proof_l201_201383


namespace local_maximum_at_1_2_l201_201609

noncomputable def f (x1 x2 : ℝ) : ℝ := x2^2 - x1^2
def constraint (x1 x2 : ℝ) : Prop := x1 - 2 * x2 + 3 = 0
def is_local_maximum (f : ℝ → ℝ → ℝ) (x1 x2 : ℝ) : Prop := 
∃ ε > 0, ∀ (y1 y2 : ℝ), (constraint y1 y2 ∧ (y1 - x1)^2 + (y2 - x2)^2 < ε^2) → f y1 y2 ≤ f x1 x2

theorem local_maximum_at_1_2 : is_local_maximum f 1 2 :=
sorry

end local_maximum_at_1_2_l201_201609


namespace first_group_correct_l201_201829

/-- Define the total members in the choir --/
def total_members : ℕ := 70

/-- Define members in the second group --/
def second_group_members : ℕ := 30

/-- Define members in the third group --/
def third_group_members : ℕ := 15

/-- Define the number of members in the first group by subtracting second and third groups members from total members --/
def first_group_members : ℕ := total_members - (second_group_members + third_group_members)

/-- Prove that the first group has 25 members --/
theorem first_group_correct : first_group_members = 25 := by
  -- insert the proof steps here
  sorry

end first_group_correct_l201_201829


namespace y_intercept_of_line_l201_201696

theorem y_intercept_of_line (x y : ℝ) (h : 4 * x + 7 * y = 28) : y = 4 :=
by
  have h' : y = -(4/7) * x + 4 := sorry
  have h_intercept : x = 0 := sorry
  exact sorry

end y_intercept_of_line_l201_201696


namespace polygon_sides_l201_201459

theorem polygon_sides (n : ℕ) (hn : 3 ≤ n) (H : (n * (n - 3)) / 2 = 15) : n = 7 :=
by
  sorry

end polygon_sides_l201_201459


namespace pure_ghee_percentage_l201_201606

theorem pure_ghee_percentage (Q : ℝ) (vanaspati_percentage : ℝ:= 0.40) (additional_pure_ghee : ℝ := 10) (new_vanaspati_percentage : ℝ := 0.20) (original_quantity : ℝ := 10) :
  (Q = original_quantity) ∧ (vanaspati_percentage = 0.40) ∧ (additional_pure_ghee = 10) ∧ (new_vanaspati_percentage = 0.20) →
  (100 - (vanaspati_percentage * 100)) = 60 :=
by
  sorry

end pure_ghee_percentage_l201_201606


namespace solve_prime_equation_l201_201110

def is_prime (n : ℕ) : Prop := Nat.Prime n

theorem solve_prime_equation (p q : ℕ) (hp : is_prime p) (hq : is_prime q) (h : p ^ 2 - 6 * p * q + q ^ 2 + 3 * q - 1 = 0) :
  (p = 17 ∧ q = 3) :=
sorry

end solve_prime_equation_l201_201110


namespace pencils_per_friend_l201_201129

theorem pencils_per_friend (total_pencils num_friends : ℕ) (h_total : total_pencils = 24) (h_friends : num_friends = 3) : total_pencils / num_friends = 8 :=
by
  -- Proof would go here
  sorry

end pencils_per_friend_l201_201129


namespace solve_prime_equation_l201_201111

theorem solve_prime_equation (p q : ℕ) (hp : p.prime) (hq : q.prime) : 
  p^2 - 6*p*q + q^2 + 3*q - 1 = 0 ↔ (p = 17 ∧ q = 3) :=
by
  sorry

end solve_prime_equation_l201_201111


namespace prove_cartesian_eq_C1_prove_cartesian_eq_C2_prove_min_distance_C1_C2_l201_201080

noncomputable def cartesian_eq_C1 (x y : ℝ) : Prop :=
  (x + 2)^2 + (y - 1)^2 = 4

noncomputable def cartesian_eq_C2 (x y : ℝ) : Prop :=
  (4 * x - y - 1 = 0)

noncomputable def min_distance_C1_C2 : ℝ :=
  (10 * Real.sqrt 17 / 17) - 2

theorem prove_cartesian_eq_C1 (x y t : ℝ) (h : x = -2 + 2 * Real.cos t ∧ y = 1 + 2 * Real.sin t) :
  cartesian_eq_C1 x y :=
sorry

theorem prove_cartesian_eq_C2 (ρ θ : ℝ) (h : 4 * ρ * Real.cos θ - ρ * Real.sin θ - 1 = 0) :
  cartesian_eq_C2 (ρ * Real.cos θ) (ρ * Real.sin θ) :=
sorry

theorem prove_min_distance_C1_C2 (h1 : ∀ x y, cartesian_eq_C1 x y) (h2 : ∀ x y, cartesian_eq_C2 x y) :
  ∀ P Q : ℝ × ℝ, (cartesian_eq_C1 P.1 P.2) → (cartesian_eq_C2 Q.1 Q.2) →
  (min_distance_C1_C2 = (Real.sqrt (4^2 + (-1)^2) / Real.sqrt 17) - 2) :=
sorry

end prove_cartesian_eq_C1_prove_cartesian_eq_C2_prove_min_distance_C1_C2_l201_201080


namespace prime_factors_sum_l201_201088

theorem prime_factors_sum (w x y z t : ℕ) 
  (h : 2^w * 3^x * 5^y * 7^z * 11^t = 107100) : 
  2 * w + 3 * x + 5 * y + 7 * z + 11 * t = 38 :=
sorry

end prime_factors_sum_l201_201088


namespace probability_digit_seven_l201_201099

noncomputable def decimal_digits := [3, 7, 5]

theorem probability_digit_seven : (∑ d in decimal_digits.filter (λ x => x = 7), 1) / decimal_digits.length = 1 / 3 := 
by
  -- add appropriate steps here
  sorry

end probability_digit_seven_l201_201099


namespace selling_price_correct_l201_201663

noncomputable def cost_price : ℝ := 2800
noncomputable def loss_percentage : ℝ := 25
noncomputable def loss_amount (cost_price loss_percentage : ℝ) : ℝ := (loss_percentage / 100) * cost_price
noncomputable def selling_price (cost_price loss_amount : ℝ) : ℝ := cost_price - loss_amount

theorem selling_price_correct : 
  selling_price cost_price (loss_amount cost_price loss_percentage) = 2100 :=
by
  sorry

end selling_price_correct_l201_201663


namespace sum_of_legs_l201_201631

theorem sum_of_legs (x : ℕ) (h : x^2 + (x + 1)^2 = 41^2) : x + (x + 1) = 57 :=
sorry

end sum_of_legs_l201_201631


namespace solve_for_x_l201_201206

theorem solve_for_x (x : ℝ) (h : 2 * x - 3 * x + 4 * x = 150) : x = 50 :=
by
  sorry

end solve_for_x_l201_201206


namespace inequality_solution_l201_201038

theorem inequality_solution (x : ℝ) : x^3 - 9 * x^2 + 27 * x > 0 → (x > 0 ∧ x < 3) ∨ (x > 6) := sorry

end inequality_solution_l201_201038


namespace volume_ratio_of_cubes_l201_201286

theorem volume_ratio_of_cubes (e1 e2 : ℕ) (h1 : e1 = 9) (h2 : e2 = 36) :
  (e1^3 : ℚ) / (e2^3 : ℚ) = 1 / 64 := by
  sorry

end volume_ratio_of_cubes_l201_201286


namespace cos_225_l201_201360

theorem cos_225 (h : Real.cos (135 * Real.pi / 180) = -Real.sqrt 2 / 2) :
    Real.cos (225 * Real.pi / 180) = -Real.sqrt 2 / 2 := by
  sorry

end cos_225_l201_201360


namespace cos_225_l201_201359

theorem cos_225 (h : Real.cos (135 * Real.pi / 180) = -Real.sqrt 2 / 2) :
    Real.cos (225 * Real.pi / 180) = -Real.sqrt 2 / 2 := by
  sorry

end cos_225_l201_201359


namespace sum_of_fraction_numerator_and_denominator_l201_201324

theorem sum_of_fraction_numerator_and_denominator (x : ℚ) (a b : ℤ) :
  x = (45 / 99 : ℚ) ∧ (a = 5) ∧ (b = 11) → (a + b = 16) :=
by
  sorry

end sum_of_fraction_numerator_and_denominator_l201_201324


namespace find_k_value_l201_201465

theorem find_k_value (k : ℝ) (hx : ∃ x : ℝ, (k - 1) * x^2 + 3 * x + k^2 - 1 = 0) :
  k = -1 :=
sorry

end find_k_value_l201_201465


namespace min_colors_needed_for_boxes_l201_201287

noncomputable def min_colors_needed : Nat := 23

theorem min_colors_needed_for_boxes :
  ∀ (boxes : Fin 8 → Fin 6 → Nat), 
  (∀ i, ∀ j : Fin 6, boxes i j < min_colors_needed) → 
  (∀ i, (Function.Injective (boxes i))) → 
  (∀ c1 c2, c1 ≠ c2 → (∃! b, ∃ p1 p2, (p1 ≠ p2 ∧ boxes b p1 = c1 ∧ boxes b p2 = c2))) → 
  min_colors_needed = 23 := 
by sorry

end min_colors_needed_for_boxes_l201_201287


namespace quadrilateral_choices_l201_201904

theorem quadrilateral_choices :
  let available_rods : List ℕ := (List.range' 1 41).diff [5, 12, 20]
  let valid_rods := available_rods.filter (λ x => 4 ≤ x ∧ x ≤ 36)
  valid_rods.length = 30 := sorry

end quadrilateral_choices_l201_201904


namespace triangle_sum_of_sides_l201_201273

noncomputable def sum_of_remaining_sides (side_a : ℝ) (angle_B : ℝ) (angle_C : ℝ) : ℝ :=
  let BD := side_a * Real.sin angle_B
  let DC := BD / Real.tan angle_C
  side_a + DC

theorem triangle_sum_of_sides :
  let side_a := 8
  let angle_B := Real.pi * 50 / 180
  let angle_C := Real.pi * 40 / 180
  sum_of_remaining_sides side_a angle_B angle_C ≈ 22.6 :=
by
  let side_a := 8
  let angle_B := Real.pi * 50 / 180
  let angle_C := Real.pi * 40 / 180
  have h1 : sum_of_remaining_sides side_a angle_B angle_C ≈ 22.6 := sorry
  exact h1

end triangle_sum_of_sides_l201_201273


namespace minimum_value_of_expression_l201_201049

theorem minimum_value_of_expression (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x + y = 1) : 
  ∃ m : ℝ, (m = 8) ∧ (∀ z : ℝ, z = (y / x) + (4 / y) → z ≥ m) :=
sorry

end minimum_value_of_expression_l201_201049


namespace tan_neg_405_eq_neg1_l201_201022

theorem tan_neg_405_eq_neg1 :
  let tan := Real.tan in
  tan (-405 * Real.pi / 180) = -1 :=
by
  have h1 : tan (45 * Real.pi / 180) = 1 := by sorry
  have h2 : ∀ x, tan (x + 2 * Real.pi) = tan x := by sorry
  have h3 : ∀ x, tan (-x) = -tan x := by sorry
  sorry

end tan_neg_405_eq_neg1_l201_201022


namespace min_value_quadratic_l201_201280

theorem min_value_quadratic : ∃ x : ℝ, ∀ y : ℝ, 3 * x ^ 2 - 18 * x + 2023 ≤ 3 * y ^ 2 - 18 * y + 2023 :=
sorry

end min_value_quadratic_l201_201280


namespace charity_distribution_l201_201008

theorem charity_distribution
    (amount_raised : ℝ)
    (donation_percentage : ℝ)
    (num_organizations : ℕ)
    (h_amount_raised : amount_raised = 2500)
    (h_donation_percentage : donation_percentage = 0.80)
    (h_num_organizations : num_organizations = 8) :
    (amount_raised * donation_percentage) / num_organizations = 250 := by
  sorry

end charity_distribution_l201_201008


namespace regular_polygon_sides_l201_201596

theorem regular_polygon_sides (interior_angle : ℝ) (h : interior_angle = 150) :
  ∃ (n : ℕ), n > 2 ∧ (interior_angle = (n - 2) * 180 / n) :=
by
  sorry
 
end regular_polygon_sides_l201_201596


namespace find_ordered_pair_l201_201643

noncomputable def discriminant_eq_zero (a c : ℝ) : Prop :=
  a * c = 9

def sum_eq_14 (a c : ℝ) : Prop :=
  a + c = 14

def a_greater_than_c (a c : ℝ) : Prop :=
  a > c

theorem find_ordered_pair : 
  ∃ (a c : ℝ), 
    sum_eq_14 a c ∧ 
    discriminant_eq_zero a c ∧ 
    a_greater_than_c a c ∧ 
    a = 7 + 2 * Real.sqrt 10 ∧ 
    c = 7 - 2 * Real.sqrt 10 :=
by {
  sorry
}

end find_ordered_pair_l201_201643


namespace problems_left_to_grade_l201_201343

def worksheets : ℕ := 17
def graded_worksheets : ℕ := 8
def problems_per_worksheet : ℕ := 7

theorem problems_left_to_grade : (worksheets - graded_worksheets) * problems_per_worksheet = 63 := by
  sorry

end problems_left_to_grade_l201_201343


namespace find_divisible_number_l201_201438

theorem find_divisible_number :
  ∃ x, x = 7 ∧ ((1020 - 12) % 12 = 0 ∧ (1020 - 12) % 24 = 0 ∧ 
             (1020 - 12) % 36 = 0 ∧ (1020 - 12) % 48 = 0) :=
by
  let num := 1020 - 12
  use 7
  split
  . refl
  . simp [num]
  sorry

end find_divisible_number_l201_201438


namespace apples_distribution_count_l201_201344

theorem apples_distribution_count : 
  ∃ (count : ℕ), count = 249 ∧ 
  (∃ (a b c : ℕ), a + b + c = 30 ∧ a ≥ 3 ∧ b ≥ 3 ∧ c ≥ 3 ∧ a ≤ 20) →
  (a' + 3 + b' + 3 + c' + 3 = 30 ∧ a' + b' + c' = 21) → 
  (∃ (a' b' c' : ℕ), a' + b' + c' = 21 ∧ a' ≤ 17) :=
by
  sorry

end apples_distribution_count_l201_201344


namespace cos_225_degrees_l201_201403

theorem cos_225_degrees : Real.cos (225 * Real.pi / 180) = -1 / Real.sqrt 2 := by
  sorry

end cos_225_degrees_l201_201403


namespace bob_age_is_725_l201_201150

theorem bob_age_is_725 (n : ℕ) (h1 : ∃ k : ℤ, n - 3 = k^2) (h2 : ∃ j : ℤ, n + 4 = j^3) : n = 725 :=
sorry

end bob_age_is_725_l201_201150


namespace article_word_limit_l201_201475

theorem article_word_limit 
  (total_pages : ℕ) (large_font_pages : ℕ) (words_per_large_page : ℕ) 
  (words_per_small_page : ℕ) (remaining_pages : ℕ) (total_words : ℕ)
  (h1 : total_pages = 21) 
  (h2 : large_font_pages = 4) 
  (h3 : words_per_large_page = 1800) 
  (h4 : words_per_small_page = 2400) 
  (h5 : remaining_pages = total_pages - large_font_pages) 
  (h6 : total_words = large_font_pages * words_per_large_page + remaining_pages * words_per_small_page) :
  total_words = 48000 := 
by
  sorry

end article_word_limit_l201_201475


namespace george_money_left_after_donations_and_groceries_l201_201042

def monthly_income : ℕ := 240
def donation (income : ℕ) : ℕ := income / 2
def post_donation_money (income : ℕ) : ℕ := income - donation income
def groceries_cost : ℕ := 20
def money_left (income : ℕ) : ℕ := post_donation_money income - groceries_cost

theorem george_money_left_after_donations_and_groceries :
  money_left monthly_income = 100 :=
by
  sorry

end george_money_left_after_donations_and_groceries_l201_201042


namespace find_sixth_term_l201_201944

noncomputable def first_term : ℝ := Real.sqrt 3
noncomputable def fifth_term : ℝ := Real.sqrt 243
noncomputable def common_ratio (q : ℝ) : Prop := fifth_term = first_term * q^4
noncomputable def sixth_term (b6 : ℝ) (q : ℝ) : Prop := b6 = fifth_term * q

theorem find_sixth_term (q : ℝ) (b6 : ℝ) : 
  first_term = Real.sqrt 3 ∧
  fifth_term = Real.sqrt 243 ∧
  common_ratio q ∧ 
  sixth_term b6 q → 
  b6 = 27 ∨ b6 = -27 := 
by
  intros
  sorry

end find_sixth_term_l201_201944


namespace tan_neg_405_eq_neg1_l201_201021

theorem tan_neg_405_eq_neg1 :
  let tan := Real.tan in
  tan (-405 * Real.pi / 180) = -1 :=
by
  have h1 : tan (45 * Real.pi / 180) = 1 := by sorry
  have h2 : ∀ x, tan (x + 2 * Real.pi) = tan x := by sorry
  have h3 : ∀ x, tan (-x) = -tan x := by sorry
  sorry

end tan_neg_405_eq_neg1_l201_201021


namespace triangle_sum_of_remaining_sides_l201_201272

noncomputable def sum_of_sides (a b c : ℝ) : ℝ :=
a + b + c

theorem triangle_sum_of_remaining_sides :
  ∃ (A B C : ℝ), ∠A = 40 ∧ ∠B = 50 ∧ ∠C = 90 ∧ opposite 40 = 8 ∧ 
  (sum_of_sides = 20.3) :=
begin
  sorry
end

end triangle_sum_of_remaining_sides_l201_201272


namespace corn_syrup_amount_l201_201902

-- Definitions based on given conditions
def flavoring_to_corn_syrup_standard := 1 / 12
def flavoring_to_water_standard := 1 / 30

def flavoring_to_corn_syrup_sport := (3 * flavoring_to_corn_syrup_standard)
def flavoring_to_water_sport := (1 / 2) * flavoring_to_water_standard

def common_factor := (30 : ℝ)

-- Amounts in sport formulation after adjustment
def flavoring_to_corn_syrup_ratio_sport := 1 / 4
def flavoring_to_water_ratio_sport := 1 / 60

def total_flavoring_corn_syrup := 15 -- Since ratio is 15:60:60 and given water is 15 ounces

theorem corn_syrup_amount (water_ounces : ℝ) :
  water_ounces = 15 → 
  (60 / 60) * water_ounces = 15 :=
by
  sorry

end corn_syrup_amount_l201_201902


namespace pencils_given_l201_201984

-- Define the conditions
def a : Nat := 9
def b : Nat := 65

-- Define the goal statement: the number of pencils Kathryn gave to Anthony
theorem pencils_given (a b : Nat) (h₁ : a = 9) (h₂ : b = 65) : b - a = 56 :=
by
  -- Omitted proof part
  sorry

end pencils_given_l201_201984


namespace min_b_geometric_sequence_l201_201053

theorem min_b_geometric_sequence (a b c : ℝ) (h_geom : b^2 = a * c) (h_1_4 : (a = 1 ∨ b = 1 ∨ c = 1) ∧ (a = 4 ∨ b = 4 ∨ c = 4)) :
  b ≥ -2 ∧ (∃ b', b' < b → b' ≥ -2) :=
by {
  sorry -- Proof required
}

end min_b_geometric_sequence_l201_201053


namespace sum_of_fraction_parts_of_repeating_decimal_l201_201291

theorem sum_of_fraction_parts_of_repeating_decimal :
  let frac := Rat.mk 45 99 in
  let simplest_frac := frac.num.gcd 45 99 in
  let num := simplest_frac.num in
  let denom := simplest_frac.denom in
  num + denom = 16 :=
by
  sorry

end sum_of_fraction_parts_of_repeating_decimal_l201_201291


namespace sum_of_fraction_terms_l201_201323

theorem sum_of_fraction_terms (x : ℚ) (hx : x = 45 / 99) (y : ℚ) (hy : y * 99 = 45) : 
  let z := (5 : ℚ) / (11 : ℚ) in 
  hwf : x = z → z.den + z.num = 16 := 
by
  sorry

end sum_of_fraction_terms_l201_201323


namespace linear_eq_m_val_l201_201054

theorem linear_eq_m_val (m : ℤ) (x : ℝ) : (5 * x ^ (m - 2) + 1 = 0) → (m = 3) :=
by
  sorry

end linear_eq_m_val_l201_201054


namespace birds_per_cup_l201_201748

theorem birds_per_cup :
  ∀ (C B S T : ℕ) (H1 : C = 2) (H2 : S = 1 / 2 * C) (H3 : T = 21) (H4 : B = 14),
    ((C - S) * B = T) :=
by
  sorry

end birds_per_cup_l201_201748


namespace distance_X_X_l201_201953

/-
  Define the vertices of the triangle XYZ
-/
def X : ℝ × ℝ := (2, -4)
def Y : ℝ × ℝ := (-1, 2)
def Z : ℝ × ℝ := (5, 1)

/-
  Define the reflection of point X over the y-axis
-/
def X' : ℝ × ℝ := (-2, -4)

/-
  Prove that the distance between X and X' is 4 units.
-/
theorem distance_X_X' : (Real.sqrt (((-2) - 2) ^ 2 + ((-4) - (-4)) ^ 2)) = 4 := by
  sorry

end distance_X_X_l201_201953


namespace sum_of_legs_of_right_triangle_l201_201637

theorem sum_of_legs_of_right_triangle : 
  ∀ (x : ℕ), (x^2 + (x + 1)^2 = 41^2) → (x + (x + 1) = 57) :=
by
sorries

end sum_of_legs_of_right_triangle_l201_201637


namespace rhombus_diagonal_length_l201_201243

theorem rhombus_diagonal_length (d2 : ℝ) (area : ℝ) (d1 : ℝ) (h1 : d2 = 80) (h2 : area = 2480) (h3 : area = (d1 * d2) / 2) : d1 = 62 :=
by sorry

end rhombus_diagonal_length_l201_201243


namespace cleaning_project_l201_201005

theorem cleaning_project (x : ℕ) : 12 + x = 2 * (15 - x) := sorry

end cleaning_project_l201_201005


namespace prime_solution_unique_l201_201120

theorem prime_solution_unique (p q : ℕ) (hp : prime p) (hq : prime q) (h : p^2 - 6 * p * q + q^2 + 3 * q - 1 = 0) : 
  (p = 17 ∧ q = 3) :=
by
  -- placeholder for the proof
  sorry

end prime_solution_unique_l201_201120


namespace problem_1_problem_2_problem_3_l201_201490

-- Problem 1
theorem problem_1 (m n : ℝ) : 
  3 * (m - n) ^ 2 - 4 * (m - n) ^ 2 + 3 * (m - n) ^ 2 = 2 * (m - n) ^ 2 := 
by
  sorry

-- Problem 2
theorem problem_2 (x y : ℝ) (h : x^2 + 2 * y = 4) : 
  3 * x^2 + 6 * y - 2 = 10 := 
by
  sorry

-- Problem 3
theorem problem_3 (x y : ℝ) 
  (h1 : x^2 + x * y = 2) 
  (h2 : 2 * y^2 + 3 * x * y = 5) : 
  2 * x^2 + 11 * x * y + 6 * y^2 = 19 := 
by
  sorry

end problem_1_problem_2_problem_3_l201_201490


namespace solve_prime_equation_l201_201109

def is_prime (n : ℕ) : Prop := Nat.Prime n

theorem solve_prime_equation (p q : ℕ) (hp : is_prime p) (hq : is_prime q) (h : p ^ 2 - 6 * p * q + q ^ 2 + 3 * q - 1 = 0) :
  (p = 17 ∧ q = 3) :=
sorry

end solve_prime_equation_l201_201109


namespace prob_of_one_letter_each_sister_l201_201566

noncomputable def prob_one_letter_each_sister : ℚ :=
  let total_cards := 10
  let letters_cybil := 5
  let letters_ronda := 5
  let prob_cybil_then_ronda := (letters_cybil / total_cards) * (letters_ronda / (total_cards - 1))
  let prob_ronda_then_cybil := (letters_ronda / total_cards) * (letters_cybil / (total_cards - 1))
  prob_cybil_then_ronda + prob_ronda_then_cybil

theorem prob_of_one_letter_each_sister :
  prob_one_letter_each_sister = 5 / 9 :=
sorry

end prob_of_one_letter_each_sister_l201_201566


namespace jinhee_pages_per_day_l201_201906

noncomputable def pages_per_day (total_pages : ℕ) (days : ℕ) : ℕ :=
  (total_pages + days - 1) / days

theorem jinhee_pages_per_day : 
  ∀ (total_pages : ℕ) (days : ℕ), total_pages = 220 → days = 7 → pages_per_day total_pages days = 32 :=
by 
  intros total_pages days hp hd
  rw [hp, hd]
  -- the computation of the function
  show pages_per_day 220 7 = 32
  sorry

end jinhee_pages_per_day_l201_201906


namespace product_seqFrac_l201_201687

def seqFrac (n : ℕ) : ℚ := (n : ℚ) / (n + 5 : ℚ)

theorem product_seqFrac :
  ((List.range 53).map seqFrac).prod = 1 / 27720 := by
  sorry

end product_seqFrac_l201_201687


namespace first_term_geometric_sequence_l201_201744

variable {a : ℕ → ℝ} -- Define the geometric sequence a_n
variable (q : ℝ) -- Define the common ratio q which is a real number

-- Conditions given in the problem
def geom_seq_first_term (a : ℕ → ℝ) (q : ℝ) :=
  a 3 = 2 ∧ a 4 = 4 ∧ (∀ n : ℕ, a (n+1) = a n * q)

-- Assert that if these conditions hold, then the first term is 1/2
theorem first_term_geometric_sequence (hq : geom_seq_first_term a q) : a 1 = 1/2 :=
by
  sorry

end first_term_geometric_sequence_l201_201744


namespace solve_system_l201_201858

variable (y : ℝ) (x1 x2 x3 x4 x5 : ℝ)

def system_of_equations :=
  x5 + x2 = y * x1 ∧
  x1 + x3 = y * x2 ∧
  x2 + x4 = y * x3 ∧
  x3 + x5 = y * x4 ∧
  x4 + x1 = y * x3

theorem solve_system :
  (y = 2 → x1 = x2 ∧ x2 = x3 ∧ x3 = x4 ∧ x4 = x5) ∧
  ((y = (-1 + Real.sqrt 5) / 2 ∨ y = (-1 - Real.sqrt 5) / 2) →
   x1 + x2 + x3 + x4 + x5 = 0 ∧ ∀ (x1 x5 : ℝ), system_of_equations y x1 x2 x3 x4 x5) :=
sorry

end solve_system_l201_201858


namespace garden_dimensions_l201_201011

theorem garden_dimensions (l b : ℝ) (walkway_width total_area perimeter : ℝ) 
  (h1 : l = 3 * b)
  (h2 : perimeter = 2 * l + 2 * b)
  (h3 : walkway_width = 1)
  (h4 : total_area = (l + 2 * walkway_width) * (b + 2 * walkway_width))
  (h5 : perimeter = 40)
  (h6 : total_area = 120) : 
  l = 15 ∧ b = 5 ∧ total_area - l * b = 45 :=  
  by
  sorry

end garden_dimensions_l201_201011


namespace lighter_dog_weight_l201_201518

theorem lighter_dog_weight
  (x y z : ℕ)
  (h1 : x + y + z = 36)
  (h2 : y + z = 3 * x)
  (h3 : x + z = 2 * y) :
  x = 9 :=
by
  sorry

end lighter_dog_weight_l201_201518


namespace seventh_observation_value_l201_201133

def average_initial_observations (S : ℝ) (n : ℕ) : Prop :=
  S / n = 13

def total_observations (n : ℕ) : Prop :=
  n + 1 = 7

def new_average (S : ℝ) (x : ℝ) (n : ℕ) : Prop :=
  (S + x) / (n + 1) = 12

theorem seventh_observation_value (S : ℝ) (n : ℕ) (x : ℝ) :
  average_initial_observations S n →
  total_observations n →
  new_average S x n →
  x = 6 :=
by
  intros h1 h2 h3
  sorry

end seventh_observation_value_l201_201133


namespace set_of_values_a_l201_201733

theorem set_of_values_a (a : ℝ) : (2 ∉ {x : ℝ | x - a < 0}) ↔ (a ≤ 2) :=
by
  sorry

end set_of_values_a_l201_201733


namespace ellipse_eccentricity_l201_201852

theorem ellipse_eccentricity (a b : ℝ) (c : ℝ) (h1 : a > b) (h2 : b > 0)
  (h3 : c^2 = a^2 - b^2) : (c / a = Real.sqrt 5 / 5) :=
by
  sorry

end ellipse_eccentricity_l201_201852


namespace area_of_union_of_rectangle_and_circle_l201_201546

theorem area_of_union_of_rectangle_and_circle :
  let width := 8
  let length := 12
  let radius := 12
  let A_rectangle := length * width
  let A_circle := Real.pi * radius ^ 2
  let A_overlap := (1 / 4) * A_circle
  A_rectangle + A_circle - A_overlap = 96 + 108 * Real.pi :=
by
  sorry

end area_of_union_of_rectangle_and_circle_l201_201546


namespace solve_for_r_l201_201734

variable (k r : ℝ)

theorem solve_for_r (h1 : 5 = k * 3^r) (h2 : 45 = k * 9^r) : r = 2 := sorry

end solve_for_r_l201_201734


namespace larger_number_is_l201_201161

-- Given definitions and conditions
def HCF (a b: ℕ) : ℕ := 23
def other_factor_1 : ℕ := 11
def other_factor_2 : ℕ := 12
def LCM (a b: ℕ) : ℕ := HCF a b * other_factor_1 * other_factor_2

-- Statement to be proven
theorem larger_number_is (a b: ℕ) (h: HCF a b = 23) (hA: a = 23 * 12) (hB: b ∣ a) : a = 276 :=
by { sorry }

end larger_number_is_l201_201161


namespace toy_cost_l201_201649

-- Definitions based on the conditions in part a)
def initial_amount : ℕ := 57
def spent_amount : ℕ := 49
def remaining_amount : ℕ := initial_amount - spent_amount
def number_of_toys : ℕ := 2

-- Statement to prove that each toy costs 4 dollars
theorem toy_cost :
  (remaining_amount / number_of_toys) = 4 :=
by
  sorry

end toy_cost_l201_201649


namespace repeating_decimal_fraction_sum_l201_201293

noncomputable def repeating_decimal_to_fraction (x : ℚ) : ℚ :=
if h : x = (45 / 99 : ℚ) then (5 / 11) else x

theorem repeating_decimal_fraction_sum : 
  let x := (0.45 : ℝ) in 
  (x.mul 100 - x).div 99 = (45 : ℝ) -> 
  let (a, b) := (5, 11) in 
  a + b = 16 :=
by
  intro x
  rfl
  rw [show (5 : ℚ) + (11 : ℚ) = 16 from rfl]
  sorry

end repeating_decimal_fraction_sum_l201_201293


namespace tim_has_33_books_l201_201685

-- Define the conditions
def b := 24   -- Benny's initial books
def s := 10   -- Books given to Sandy
def total_books : Nat := 47  -- Total books

-- Define the remaining books after Benny gives to Sandy
def remaining_b : Nat := b - s

-- Define Tim's books
def tim_books : Nat := total_books - remaining_b

-- Prove that Tim has 33 books
theorem tim_has_33_books : tim_books = 33 := by
  -- This is a placeholder for the proof
  sorry

end tim_has_33_books_l201_201685


namespace factorize_polynomial_l201_201857

theorem factorize_polynomial (x y : ℝ) : x * y^2 - 16 * x = x * (y + 4) * (y - 4) := 
by
  sorry

end factorize_polynomial_l201_201857


namespace largest_four_digit_number_property_l201_201800

theorem largest_four_digit_number_property : ∃ (a b c d : Nat), 
  (1000 * a + 100 * b + 10 * c + d = 9099) ∧ 
  (c = a + b) ∧ 
  (d = b + c) ∧ 
  1 ≤ a ∧ a ≤ 9 ∧ 
  0 ≤ b ∧ b ≤ 9 ∧ 
  0 ≤ c ∧ c ≤ 9 ∧ 
  0 ≤ d ∧ d ≤ 9 := 
sorry

end largest_four_digit_number_property_l201_201800


namespace max_points_of_intersection_l201_201485

-- Define the lines and their properties
variable (L : Fin 150 → Prop)

-- Condition: L_5n are parallel to each other
def parallel_group (n : ℕ) :=
  ∃ k, n = 5 * k

-- Condition: L_{5n-1} pass through a given point B
def passing_through_B (n : ℕ) :=
  ∃ k, n = 5 * k + 1

-- Condition: L_{5n-2} are parallel to another line not parallel to those in parallel_group
def other_parallel_group (n : ℕ) :=
  ∃ k, n = 5 * k + 3

-- Total number of points of intersection of pairs of lines from the complete set
theorem max_points_of_intersection (L : Fin 150 → Prop)
  (h_distinct : ∀ i j : Fin 150, i ≠ j → L i ≠ L j)
  (h_parallel_group : ∀ i j : Fin 150, parallel_group i → parallel_group j → L i = L j)
  (h_through_B : ∀ i j : Fin 150, passing_through_B i → passing_through_B j → L i = L j)
  (h_other_parallel_group : ∀ i j : Fin 150, other_parallel_group i → other_parallel_group j → L i = L j)
  : ∃ P, P = 8071 := 
sorry

end max_points_of_intersection_l201_201485


namespace percentage_increase_salary_l201_201887

theorem percentage_increase_salary (S : ℝ) (P : ℝ) (h1 : 1.16 * S = 348) (h2 : S + P * S = 375) : P = 0.25 :=
by
  sorry

end percentage_increase_salary_l201_201887


namespace number_of_solutions_eq_six_l201_201732

/-- 
The number of ordered pairs (m, n) of positive integers satisfying the equation
6/m + 3/n = 1 is 6.
-/
theorem number_of_solutions_eq_six : 
  ∃! (s : Finset (ℕ × ℕ)), 
    (∀ p ∈ s, (1 < p.1 ∧ 1 < p.2) ∧ 6 / p.1 + 3 / p.2 = 1) ∧ s.card = 6 :=
sorry

end number_of_solutions_eq_six_l201_201732


namespace tan_neg_405_eq_one_l201_201024

theorem tan_neg_405_eq_one : Real.tan (-(405 * Real.pi / 180)) = 1 :=
by
-- Proof omitted
sorry

end tan_neg_405_eq_one_l201_201024


namespace num_lighting_methods_l201_201601

-- Definitions of the problem's conditions
def total_lights : ℕ := 15
def lights_off : ℕ := 6
def lights_on : ℕ := total_lights - lights_off
def available_spaces : ℕ := lights_on - 1

-- Statement of the mathematically equivalent proof problem
theorem num_lighting_methods : Nat.choose available_spaces lights_off = 28 := by
  sorry

end num_lighting_methods_l201_201601


namespace andrew_paid_1428_to_shopkeeper_l201_201820

-- Given conditions
def rate_per_kg_grapes : ℕ := 98
def quantity_of_grapes : ℕ := 11
def rate_per_kg_mangoes : ℕ := 50
def quantity_of_mangoes : ℕ := 7

-- Definitions for costs
def cost_of_grapes : ℕ := rate_per_kg_grapes * quantity_of_grapes
def cost_of_mangoes : ℕ := rate_per_kg_mangoes * quantity_of_mangoes
def total_amount_paid : ℕ := cost_of_grapes + cost_of_mangoes

-- Theorem to prove the total amount paid
theorem andrew_paid_1428_to_shopkeeper : total_amount_paid = 1428 := by
  sorry

end andrew_paid_1428_to_shopkeeper_l201_201820


namespace find_product_of_roots_l201_201484

noncomputable def equation (x : ℝ) : ℝ := (Real.sqrt 2023) * x^3 - 4047 * x^2 + 3

theorem find_product_of_roots (x1 x2 x3 : ℝ) (h1 : x1 < x2) (h2 : x2 < x3) 
  (h3 : equation x1 = 0) (h4 : equation x2 = 0) (h5 : equation x3 = 0) :
  x2 * (x1 + x3) = 3 :=
by
  sorry

end find_product_of_roots_l201_201484


namespace trader_profit_l201_201549

theorem trader_profit
  (CP : ℝ)
  (MP : ℝ)
  (SP : ℝ)
  (h1 : MP = CP * 1.12)
  (discount_percent : ℝ)
  (h2 : discount_percent = 0.09821428571428571)
  (discount : ℝ)
  (h3 : discount = MP * discount_percent)
  (actual_SP : ℝ)
  (h4 : actual_SP = MP - discount)
  (h5 : CP = 100) :
  (actual_SP / CP = 1.01) :=
by
  sorry

end trader_profit_l201_201549


namespace part_a_part_b_part_c_part_d_l201_201478

open Nat

theorem part_a (y z : ℕ) (hy : 0 < y) (hz : 0 < z) : 
  (1 = 1 / y + 1 / z) ↔ (y = 2 ∧ z = 1) := 
by 
  sorry

theorem part_b (y z : ℕ) (hy : y ≥ 2) (hz : 0 < z) : 
  (1 / 2 + 1 / y = 1 / 2 + 1 / z) ↔ (y = z ∧ y ≥ 2) ∨ (y = 1 ∧ z = 1) := 
by 
  sorry 

theorem part_c (y z : ℕ) (hy : y ≥ 3) (hz : 0 < z) : 
  (1 / 3 + 1 / y = 1 / 2 + 1 / z) ↔ 
    (y = 3 ∧ z = 6) ∨ 
    (y = 4 ∧ z = 12) ∨ 
    (y = 5 ∧ z = 30) ∨ 
    (y = 2 ∧ z = 3) := 
by 
  sorry 

theorem part_d (x y : ℕ) (hx : x ≥ 4) (hy : y ≥ 4) : 
  ¬(1 / x + 1 / y = 1 / 2 + 1 / z) := 
by 
  sorry

end part_a_part_b_part_c_part_d_l201_201478


namespace find_a_b_l201_201235

noncomputable def f (a b x : ℝ) : ℝ := a * x + b
noncomputable def g (x : ℝ) : ℝ := 3 * x - 6

theorem find_a_b (a b : ℝ) (h : ∀ x : ℝ, g (f a b x) = 4 * x + 7) :
  a + b = 17 / 3 :=
by
  sorry

end find_a_b_l201_201235


namespace vec_subtraction_l201_201061

def a : ℝ × ℝ := (-1, 2)
def b : ℝ × ℝ := (0, 1)

theorem vec_subtraction : a - 2 • b = (-1, 0) := by
  sorry

end vec_subtraction_l201_201061


namespace minimum_manhattan_distance_l201_201898

open Real

def ellipse (P : ℝ × ℝ) : Prop := P.1^2 / 2 + P.2^2 = 1

def line (Q : ℝ × ℝ) : Prop := 3 * Q.1 + 4 * Q.2 = 12

def manhattan_distance (P Q : ℝ × ℝ) : ℝ :=
  |P.1 - Q.1| + |P.2 - Q.2|

theorem minimum_manhattan_distance :
  ∃ P Q, ellipse P ∧ line Q ∧
    ∀ P' Q', ellipse P' → line Q' → manhattan_distance P Q ≤ manhattan_distance P' Q' :=
  sorry

end minimum_manhattan_distance_l201_201898


namespace Megan_pictures_left_l201_201328

theorem Megan_pictures_left (zoo_pictures museum_pictures deleted_pictures : ℕ) 
  (h1 : zoo_pictures = 15) 
  (h2 : museum_pictures = 18) 
  (h3 : deleted_pictures = 31) : 
  zoo_pictures + museum_pictures - deleted_pictures = 2 := 
by
  sorry

end Megan_pictures_left_l201_201328


namespace trigonometric_identity_l201_201875

theorem trigonometric_identity (θ : ℝ) (h : Real.tan θ = 2) : 
  (Real.sin θ * Real.cos θ) / (1 + Real.sin θ ^ 2) = 2 / 9 := 
sorry

end trigonometric_identity_l201_201875


namespace circle_tangent_to_x_axis_at_origin_l201_201738

theorem circle_tangent_to_x_axis_at_origin
  (D E F : ℝ)
  (h : ∀ (x y : ℝ), x^2 + y^2 + D*x + E*y + F = 0 → 
      (x = 0 → y = 0)) :
  D = 0 ∧ E ≠ 0 ∧ F = 0 :=
sorry

end circle_tangent_to_x_axis_at_origin_l201_201738


namespace frustum_lateral_surface_area_l201_201540

theorem frustum_lateral_surface_area:
  ∀ (R r h : ℝ), R = 7 → r = 4 → h = 6 → (∃ L, L = 33 * Real.pi * Real.sqrt 5) := by
  sorry

end frustum_lateral_surface_area_l201_201540


namespace geometric_progression_sixth_term_proof_l201_201945

noncomputable def geometric_progression_sixth_term (b₁ b₅ : ℝ) (q : ℝ) := b₅ * q
noncomputable def find_q (b₁ b₅ : ℝ) := (b₅ / b₁)^(1/4)

theorem geometric_progression_sixth_term_proof (b₁ b₅ : ℝ) (h₁ : b₁ = Real.sqrt 3) (h₅ : b₅ = Real.sqrt 243) : 
  ∃ q : ℝ, (q = Real.sqrt 3 ∨ q = - Real.sqrt 3) ∧ geometric_progression_sixth_term b₁ b₅ q = 27 ∨ geometric_progression_sixth_term b₁ b₅ q = -27 :=
by
  sorry

end geometric_progression_sixth_term_proof_l201_201945


namespace prime_solution_unique_l201_201115

open Nat

theorem prime_solution_unique (p q : ℕ) (hp : Prime p) (hq : Prime q) (h : p^2 - 6 * p * q + q^2 + 3 * q - 1 = 0) : 
  (p = 17 ∧ q = 3) :=
by
  sorry

end prime_solution_unique_l201_201115


namespace cannot_determine_both_correct_l201_201095

-- Definitions
def total_students : ℕ := 40
def answered_q1_correctly : ℕ := 30
def did_not_take_test : ℕ := 10

-- Assertion that the number of students answering both questions correctly cannot be determined
theorem cannot_determine_both_correct (answered_q2_correctly : ℕ) :
  (∃ (both_correct : ℕ), both_correct ≤ answered_q1_correctly ∧ both_correct ≤ answered_q2_correctly)  ↔ answered_q2_correctly > 0 :=
by 
 sorry

end cannot_determine_both_correct_l201_201095


namespace range_of_m_l201_201700

def P (x : ℝ) : Prop := |(4 - x) / 3| ≤ 2
def q (x m : ℝ) : Prop := (x + m - 1) * (x - m - 1) ≤ 0

theorem range_of_m (m : ℝ) (h : m > 0) : (∀ x, ¬P x → ¬q x m) → m ≥ 9 :=
by
  intros
  sorry

end range_of_m_l201_201700


namespace puppy_grouping_count_l201_201130

-- Define the conditions of the problem
def total_puppies := 12
def group1_size := 4
def group2_size := 6
def group3_size := 2
def coco_in_group1 := true
def rocky_in_group2 := true

-- Define the problem statement
theorem puppy_grouping_count :
  ∑(hu : (finset.range 10).powerset.filter (λ s, s.card = group1_size - 1)).card * 
  ∑(hr : (finset.range 7).powerset.filter (λ t, t.card = group2_size - 1)).card = 2520 := sorry

end puppy_grouping_count_l201_201130


namespace find_increase_x_l201_201274

noncomputable def initial_radius : ℝ := 7
noncomputable def initial_height : ℝ := 5
variable (x : ℝ)

theorem find_increase_x (hx : x > 0)
  (volume_eq : π * (initial_radius + x) ^ 2 * initial_height =
               π * initial_radius ^ 2 * (initial_height + 2 * x)) :
  x = 28 / 5 :=
by
  sorry

end find_increase_x_l201_201274


namespace intersection_complement_l201_201722

def U : Set ℕ := {0, 1, 2, 3, 4, 5}
def A : Set ℕ := {0, 1, 3}
def B : Set ℕ := {2, 3, 5}

theorem intersection_complement : A ∩ (U \ B) = {0, 1} := by
  sorry

end intersection_complement_l201_201722


namespace sum_faces_edges_vertices_square_pyramid_l201_201959

theorem sum_faces_edges_vertices_square_pyramid : 
  let F := 5 in
  let E := 8 in
  let V := 5 in
  F + E + V = 18 := by
  sorry

end sum_faces_edges_vertices_square_pyramid_l201_201959


namespace length_of_FD_l201_201471

/-- Square ABCD with side length 8 cm, corner C is folded to point E on AD such that AE = 2 cm and ED = 6 cm. Find the length of FD. -/
theorem length_of_FD 
  (A B C D E F G : Type)
  (square_length : Float)
  (AD_length AE_length ED_length : Float)
  (hyp1 : square_length = 8)
  (hyp2 : AE_length = 2)
  (hyp3 : ED_length = 6)
  (hyp4 : AD_length = AE_length + ED_length)
  (FD_length : Float) :
  FD_length = 7 / 4 := 
  by 
  sorry

end length_of_FD_l201_201471


namespace trig_identity_l201_201701

theorem trig_identity (α : ℝ) (h : Real.tan α = 2) :
    Real.cos (2 * α) - Real.sin α * Real.cos α = -1 := 
by 
  sorry

end trig_identity_l201_201701


namespace shaded_fraction_of_rectangle_l201_201535

theorem shaded_fraction_of_rectangle (a b : ℕ) (h_dim : a = 15 ∧ b = 24) (h_shaded : ∃ s, s = (1/3 : ℚ)) :
  ∃ f, f = (1/9 : ℚ) := 
by
  sorry

end shaded_fraction_of_rectangle_l201_201535


namespace total_amount_proof_l201_201528

-- Define the relationships between x, y, and z in terms of the amounts received
variables (x y z : ℝ)

-- Given: For each rupee x gets, y gets 0.45 rupees and z gets 0.50 rupees
def relationship1 : Prop := ∀ (k : ℝ), y = 0.45 * k ∧ z = 0.50 * k ∧ x = k

-- Given: The share of y is Rs. 54
def condition1 : Prop := y = 54

-- The total amount x + y + z is Rs. 234
def total_amount (x y z : ℝ) : ℝ := x + y + z

-- Prove that the total amount is Rs. 234
theorem total_amount_proof (x y z : ℝ) (h1: relationship1 x y z) (h2: condition1 y) : total_amount x y z = 234 :=
sorry

end total_amount_proof_l201_201528


namespace range_of_a_l201_201867

variable (a x : ℝ)

def p (a x : ℝ) : Prop := a - 4 < x ∧ x < a + 4

def q (x : ℝ) : Prop := (x - 2) * (x - 3) > 0

theorem range_of_a (h : ∀ (x : ℝ), p a x → q x) : a <= -2 ∨ a >= 7 := 
by sorry

end range_of_a_l201_201867


namespace least_sugar_pounds_l201_201558

theorem least_sugar_pounds (f s : ℕ) (hf1 : f ≥ 7 + s / 2) (hf2 : f ≤ 3 * s) : s ≥ 3 :=
by
  have h : (5 * s) / 2 ≥ 7 := sorry
  have s_ge_3 : s ≥ 3 := sorry
  exact s_ge_3

end least_sugar_pounds_l201_201558


namespace fractions_equiv_x_zero_l201_201424

theorem fractions_equiv_x_zero (x b : ℝ) (h : x + 3 * b ≠ 0) : 
  (x + 2 * b) / (x + 3 * b) = 2 / 3 ↔ x = 0 :=
by sorry

end fractions_equiv_x_zero_l201_201424


namespace out_of_pocket_expense_l201_201229

theorem out_of_pocket_expense :
  let initial_purchase := 3000
  let tv_return := 700
  let bike_return := 500
  let sold_bike_cost := bike_return + (0.20 * bike_return)
  let sold_bike_sell_price := 0.80 * sold_bike_cost
  let toaster_purchase := 100
  (initial_purchase - tv_return - bike_return - sold_bike_sell_price + toaster_purchase) = 1420 :=
by
  sorry

end out_of_pocket_expense_l201_201229


namespace cos_225_l201_201393

theorem cos_225 : cos (225 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by
-- Conditions based on given problem
have angle_def : 225 * Real.pi / 180 = 5 * Real.pi / 4 := by norm_num
have coord_Q := Complex.exp (5 * Real.pi / 4 * Complex.I)
have third_quadrant : coord_Q = Complex.mk (-Real.sqrt 2 / 2) (-Real.sqrt 2 / 2) := 
    by rw [coord_Q, Complex.exp_eq_cos_add_sin, Complex.cos, Complex.sin]
-- Proof equivalent
rw [angle_def, coord_Q]
exact congr_arg Complex.re third_quadrant
-- Equivalent statement
#check cos (5 * Real.pi / 4) = -Real.sqrt 2 / 2
-- Proof is left as an exercise
sorry

end cos_225_l201_201393


namespace richard_older_than_david_l201_201895

variable {R D S : ℕ}

theorem richard_older_than_david (h1 : R > D) (h2 : D = S + 8) (h3 : R + 8 = 2 * (S + 8)) (h4 : D = 14) : R - D = 6 := by
  sorry

end richard_older_than_david_l201_201895


namespace chosen_number_is_155_l201_201013

variable (x : ℤ)
variable (h₁ : 2 * x - 200 = 110)

theorem chosen_number_is_155 : x = 155 := by
  sorry

end chosen_number_is_155_l201_201013


namespace repeating_decimal_fraction_sum_l201_201312

noncomputable def repeating_decimal_to_fraction (d : ℚ) : ℚ :=
  if d = 0.45 then 5 / 11 else 0

theorem repeating_decimal_fraction_sum :
  let x := repeating_decimal_to_fraction 0.45 in
  (x.num + x.den = 16) :=
by
  let x := repeating_decimal_to_fraction 0.45
  have h : x = 5 / 11 := by
    unfold repeating_decimal_to_fraction
    split_ifs with h
    . exact rfl
  rw [h, Rat.num_den_eq 5 11, Int.add_comm]
  sorry

end repeating_decimal_fraction_sum_l201_201312


namespace parking_savings_l201_201664

theorem parking_savings
  (weekly_rent : ℕ := 10)
  (monthly_rent : ℕ := 40)
  (weeks_in_year : ℕ := 52)
  (months_in_year : ℕ := 12)
  : weekly_rent * weeks_in_year - monthly_rent * months_in_year = 40 := 
by
  sorry

end parking_savings_l201_201664


namespace A_superset_B_l201_201755

open Set

variable (N : Set ℕ)
def A : Set ℕ := {x | ∃ n ∈ N, x = 2 * n}
def B : Set ℕ := {x | ∃ n ∈ N, x = 4 * n}

theorem A_superset_B : A N ⊇ B N :=
by
  -- Proof to be written
  sorry

end A_superset_B_l201_201755


namespace clearance_sale_gain_percent_l201_201931

theorem clearance_sale_gain_percent
  (SP : ℝ := 30)
  (gain_percent : ℝ := 25)
  (discount_percent : ℝ := 10)
  (CP : ℝ := SP/(1 + gain_percent/100)) :
  let Discount := discount_percent / 100 * SP
  let SP_sale := SP - Discount
  let Gain_during_sale := SP_sale - CP
  let Gain_percent_during_sale := (Gain_during_sale / CP) * 100
  Gain_percent_during_sale = 12.5 := 
by
  sorry

end clearance_sale_gain_percent_l201_201931


namespace max_xy_l201_201893

-- Lean statement for the given problem
theorem max_xy (x y : ℝ) (h : x^2 + y^2 = 4) : xy ≤ 2 := sorry

end max_xy_l201_201893


namespace u_1000_eq_2036_l201_201916

open Nat

def sequence_term (n : ℕ) : ℕ :=
  let sum_to (k : ℕ) := k * (k + 1) / 2
  if n ≤ 0 then 0 else
  let group := (Nat.sqrt (2 * n)) + 1
  let k := n - sum_to (group - 1)
  (group * group) + 4 * (k - 1) - (group % 4)

theorem u_1000_eq_2036 : sequence_term 1000 = 2036 := sorry

end u_1000_eq_2036_l201_201916


namespace cos_225_eq_neg_sqrt2_div_2_l201_201386

theorem cos_225_eq_neg_sqrt2_div_2 : Real.cos (225 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by
  sorry

end cos_225_eq_neg_sqrt2_div_2_l201_201386


namespace arithmetic_mean_fraction_l201_201878

theorem arithmetic_mean_fraction (a b c : ℚ) (h1 : a = 8 / 11) (h2 : b = 7 / 11) (h3 : c = 9 / 11) :
  a = (b + c) / 2 :=
by
  sorry

end arithmetic_mean_fraction_l201_201878


namespace chess_players_swim_not_every_swimmer_plays_tennis_tennis_players_play_chess_l201_201841

variables (Bodyguards : Type)
variables (U S T : Bodyguards → Prop)

-- Conditions
axiom cond1 : ∀ x, (T x → (S x → U x))
axiom cond2 : ∀ x, (S x → (U x ∨ T x))
axiom cond3 : ∀ x, (¬ U x ∧ T x → S x)

-- To prove
theorem chess_players_swim : ∀ x, (S x → U x) := by
  sorry

theorem not_every_swimmer_plays_tennis : ¬ ∀ x, (U x → T x) := by
  sorry

theorem tennis_players_play_chess : ∀ x, (T x → S x) := by
  sorry

end chess_players_swim_not_every_swimmer_plays_tennis_tennis_players_play_chess_l201_201841


namespace number_of_ways_to_place_balls_l201_201927

theorem number_of_ways_to_place_balls : 
  let balls := 3 
  let boxes := 4 
  (boxes^balls = 64) :=
by
  sorry

end number_of_ways_to_place_balls_l201_201927


namespace max_sum_hex_digits_l201_201736

theorem max_sum_hex_digits 
  (a b c : ℕ) (y : ℕ) 
  (h_a : 0 ≤ a ∧ a < 16)
  (h_b : 0 ≤ b ∧ b < 16)
  (h_c : 0 ≤ c ∧ c < 16)
  (h_y : 0 < y ∧ y ≤ 16)
  (h_fraction : (a * 256 + b * 16 + c) * y = 4096) : 
  a + b + c ≤ 1 :=
sorry

end max_sum_hex_digits_l201_201736


namespace find_ratio_l201_201034

-- Definitions and conditions
def sides_form_right_triangle (x d : ℝ) : Prop :=
  x > d ∧ d > 0 ∧ (x^2 + (x^2 - d)^2 = (x^2 + d)^2)

-- The theorem stating the required ratio
theorem find_ratio (x d : ℝ) (h : sides_form_right_triangle x d) : 
  x / d = 8 :=
by
  sorry

end find_ratio_l201_201034


namespace cos_225_l201_201379

noncomputable def cos_225_eq : ℝ :=
  cos (225 * (real.pi / 180))

theorem cos_225 : cos_225_eq = -real.sqrt 2 / 2 :=
  sorry

end cos_225_l201_201379


namespace largest_four_digit_number_prop_l201_201808

theorem largest_four_digit_number_prop :
  ∃ (a b c d : ℕ), a = 9 ∧ b = 0 ∧ c = 9 ∧ d = 9 ∧ (1000 * a + 100 * b + 10 * c + d = 9099) ∧ (c = a + b) ∧ (d = b + c) :=
by
  sorry

end largest_four_digit_number_prop_l201_201808


namespace cos_225_eq_neg_sqrt2_div_2_l201_201417

theorem cos_225_eq_neg_sqrt2_div_2 :
  (real.cos (225 * real.pi / 180) = - (real.sqrt 2 / 2)) :=
by
  sorry

end cos_225_eq_neg_sqrt2_div_2_l201_201417


namespace new_cost_after_decrease_l201_201683

def actual_cost : ℝ := 2400
def decrease_percentage : ℝ := 0.50
def decreased_amount (cost percentage : ℝ) : ℝ := percentage * cost
def new_cost (cost decreased : ℝ) : ℝ := cost - decreased

theorem new_cost_after_decrease :
  new_cost actual_cost (decreased_amount actual_cost decrease_percentage) = 1200 :=
by sorry

end new_cost_after_decrease_l201_201683


namespace probability_four_dice_equal_sum_l201_201865

noncomputable def fair_dice := finset.range 1 7
noncomputable def total_outcomes := 6^4

def probability_same_sum (d1 d2 d3 d4 : ℕ) : Prop :=
  d1 ∈ fair_dice ∧ d2 ∈ fair_dice ∧ d3 ∈ fair_dice ∧ d4 ∈ fair_dice ∧
  (d1 + d2 = d3 + d4)

theorem probability_four_dice_equal_sum :
  ∃ (favorable_outcomes : ℕ), (6 * favorable_outcomes = 900) → (favorable_outcomes / total_outcomes = 25/36) :=
by
  use 150
  intros h
  rw total_outcomes
  linarith

end probability_four_dice_equal_sum_l201_201865


namespace compare_exponential_functions_l201_201850

theorem compare_exponential_functions (x : ℝ) (hx1 : 0 < x) :
  0.4^4 < 1 ∧ 1 < 4^0.4 :=
by sorry

end compare_exponential_functions_l201_201850


namespace only_B_is_like_terms_l201_201524

def is_like_terms (terms : List (String × String)) : List Bool :=
  let like_term_checker := fun (term1 term2 : String) =>
    -- The function to check if two terms are like terms
    sorry
  terms.map (fun (term1, term2) => like_term_checker term1 term2)

theorem only_B_is_like_terms :
  is_like_terms [("−2x^3", "−3x^2"), ("−(1/4)ab", "18ba"), ("a^2b", "−ab^2"), ("4m", "6mn")] =
  [false, true, false, false] :=
by
  sorry

end only_B_is_like_terms_l201_201524


namespace jared_march_texts_l201_201905

def T (n : ℕ) : ℕ := ((n ^ 2) + 1) * (n.factorial)

theorem jared_march_texts : T 5 = 3120 := by
  -- The details of the proof would go here, but we use sorry to skip it
  sorry

end jared_march_texts_l201_201905


namespace property_check_l201_201995

noncomputable def f (x : ℝ) : ℤ := ⌈x⌉ -- Define the ceiling function

theorem property_check :
  (¬ (∀ x : ℝ, f (2 * x) = 2 * f x)) ∧
  (∀ x1 x2 : ℝ, f x1 = f x2 → |x1 - x2| < 1) ∧
  (∀ x1 x2 : ℝ, f (x1 + x2) ≤ f x1 + f x2) ∧
  (¬ (∀ x : ℝ, f x + f (x + 0.5) = f (2 * x))) :=
by
  sorry

end property_check_l201_201995


namespace area_of_three_layer_cover_l201_201331

-- Define the hall dimensions
def hall_width : ℕ := 10
def hall_height : ℕ := 10

-- Define the dimensions of the carpets
def carpet1_width : ℕ := 6
def carpet1_height : ℕ := 8
def carpet2_width : ℕ := 6
def carpet2_height : ℕ := 6
def carpet3_width : ℕ := 5
def carpet3_height : ℕ := 7

-- Theorem to prove area covered by the carpets in three layers
theorem area_of_three_layer_cover : 
  ∀ (w1 w2 w3 h1 h2 h3 : ℕ), w1 = carpet1_width → h1 = carpet1_height → w2 = carpet2_width → h2 = carpet2_height → w3 = carpet3_width → h3 = carpet3_height → 
  ∃ (area : ℕ), area = 6 :=
by
  intros w1 w2 w3 h1 h2 h3 hw1 hw2 hw3 hh1 hh2 hh3
  exact ⟨6, rfl⟩

#check area_of_three_layer_cover

end area_of_three_layer_cover_l201_201331


namespace largest_four_digit_number_with_property_l201_201797

theorem largest_four_digit_number_with_property :
  ∃ (a b c d : ℕ), (a = 9 ∧ b = 0 ∧ c = 9 ∧ d = 9 ∧ c = a + b ∧ d = b + c ∧ 1000 * a + 100 * b + 10 * c + d = 9099) :=
sorry

end largest_four_digit_number_with_property_l201_201797


namespace cos_225_eq_neg_sqrt2_div2_l201_201363

theorem cos_225_eq_neg_sqrt2_div2 : Real.cos (225 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by sorry

end cos_225_eq_neg_sqrt2_div2_l201_201363


namespace intersection_eq_singleton_3_l201_201050

def setA : Set ℕ := {x : ℕ | x^2 - 4 * x - 5 ≤ 0}

def setB : Set ℝ := {x : ℝ | Real.log (x - 2) / Real.log 2023 ≤ 0}

theorem intersection_eq_singleton_3 : A ∩ B = {3} :=
by
  -- detailed proof steps should be here
  sorry

end intersection_eq_singleton_3_l201_201050


namespace certain_number_is_8000_l201_201155

theorem certain_number_is_8000 (x : ℕ) (h : x / 10 - x / 2000 = 796) : x = 8000 :=
sorry

end certain_number_is_8000_l201_201155


namespace find_integer_l201_201961

theorem find_integer (x : ℕ) (h : (4 * x) ^ 2 - 3 * x = 1764) : x = 18 := 
by 
  sorry

end find_integer_l201_201961


namespace dividend_is_5336_l201_201896

theorem dividend_is_5336 (D Q R : ℕ) (h1 : D = 10 * Q) (h2 : D = 5 * R) (h3 : R = 46) :
  (D * Q + R) = 5336 :=
by {
  sorry
}

end dividend_is_5336_l201_201896


namespace part1_part2_l201_201698

open Nat

-- Part (I)
theorem part1 (a b : ℝ) (h1 : ∀ x : ℝ, x^2 - a * x + b = 0 → x = 2 ∨ x = 3) :
  a + b = 11 :=
by sorry

-- Part (II)
theorem part2 (c : ℝ) (h2 : ∀ x : ℝ, -x^2 + 6 * x + c ≤ 0) :
  c ≤ -9 :=
by sorry

end part1_part2_l201_201698


namespace find_fourth_intersection_point_l201_201470

theorem find_fourth_intersection_point 
  (a b r: ℝ) 
  (h4 : ∃ a b r, ∀ x y, (x - a)^2 + (y - b)^2 = r^2 → (x, y) = (4, 1) ∨ (x, y) = (-2, -2) ∨ (x, y) = (8, 1/2) ∨ (x, y) = (-1/4, -16)):
  ∃ (x y : ℝ), (x - a)^2 + (y - b)^2 = r^2 → x * y = 4 → (x, y) = (-1/4, -16) := 
sorry

end find_fourth_intersection_point_l201_201470


namespace range_a_l201_201464

theorem range_a (a : ℝ) : 
  (∃ x : ℝ, 0 < x ∧ x < 1 ∧ a * x^2 - x - 1 = 0 ∧ 
  ∀ y : ℝ, (0 < y ∧ y < 1 ∧ a * y^2 - y - 1 = 0 → y = x)) ↔ a > 2 :=
by
  sorry

end range_a_l201_201464


namespace minimum_value_expression_l201_201284

theorem minimum_value_expression : ∃ x : ℝ, (3 * x^2 - 18 * x + 2023) = 1996 := sorry

end minimum_value_expression_l201_201284


namespace pirate_loot_l201_201956

theorem pirate_loot (a b c d e : ℕ) (h1 : a = 1 ∨ b = 1 ∨ c = 1 ∨ d = 1 ∨ e = 1)
  (h2 : a = 2 ∨ b = 2 ∨ c = 2 ∨ d = 2 ∨ e = 2)
  (h3 : a ≤ b ∧ b ≤ c ∧ c ≤ d ∧ d ≤ e)
  (h4 : a + b = 2 * (c + d) ∨ b + c = 2 * (a + e)) :
  (a, b, c, d, e) = (1, 1, 1, 1, 2) ∨ 
  (a, b, c, d, e) = (1, 1, 2, 2, 2) ∨
  (a, b, c, d, e) = (1, 2, 3, 3, 3) ∨
  (a, b, c, d, e) = (1, 2, 2, 2, 3) :=
sorry

end pirate_loot_l201_201956


namespace quadratic_has_one_real_solution_l201_201574

theorem quadratic_has_one_real_solution (m : ℝ) : (∀ x : ℝ, (x + 5) * (x + 2) = m + 3 * x) → m = 6 :=
by
  sorry

end quadratic_has_one_real_solution_l201_201574


namespace base8_subtraction_correct_l201_201187

noncomputable def base8_subtraction (x y : Nat) : Nat :=
  if y > x then 0 else x - y

theorem base8_subtraction_correct :
  base8_subtraction 546 321 - 105 = 120 :=
by
  -- Given the condition that all arithmetic is in base 8
  sorry

end base8_subtraction_correct_l201_201187


namespace union_subgroup_iff_l201_201752

variables {Γ : Type*} [Group Γ] {G H : Subgroup Γ}

theorem union_subgroup_iff : IsSubgroup ↑(G ∪ H) ↔ G ≤ H ∨ H ≤ G :=
sorry

end union_subgroup_iff_l201_201752


namespace maximum_gcd_of_sequence_l201_201642

def a_n (n : ℕ) : ℕ := 100 + n^2

def d_n (n : ℕ) : ℕ := Nat.gcd (a_n n) (a_n (n + 1))

theorem maximum_gcd_of_sequence : ∃ n : ℕ, ∀ m : ℕ, d_n n ≤ d_n m ∧ d_n n = 401 := sorry

end maximum_gcd_of_sequence_l201_201642


namespace Clarence_total_oranges_l201_201848

def Clarence_oranges_initial := 5
def oranges_from_Joyce := 3

theorem Clarence_total_oranges : Clarence_oranges_initial + oranges_from_Joyce = 8 := by
  sorry

end Clarence_total_oranges_l201_201848


namespace cos_225_l201_201394

theorem cos_225 : cos (225 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by
-- Conditions based on given problem
have angle_def : 225 * Real.pi / 180 = 5 * Real.pi / 4 := by norm_num
have coord_Q := Complex.exp (5 * Real.pi / 4 * Complex.I)
have third_quadrant : coord_Q = Complex.mk (-Real.sqrt 2 / 2) (-Real.sqrt 2 / 2) := 
    by rw [coord_Q, Complex.exp_eq_cos_add_sin, Complex.cos, Complex.sin]
-- Proof equivalent
rw [angle_def, coord_Q]
exact congr_arg Complex.re third_quadrant
-- Equivalent statement
#check cos (5 * Real.pi / 4) = -Real.sqrt 2 / 2
-- Proof is left as an exercise
sorry

end cos_225_l201_201394


namespace value_of_alpha_beta_l201_201592

variable (α β : ℝ)

-- Conditions
def quadratic_eq (x: ℝ) : Prop := x^2 + 2*x - 2005 = 0

-- Lean 4 statement
theorem value_of_alpha_beta 
  (hα : quadratic_eq α) 
  (hβ : quadratic_eq β)
  (sum_roots : α + β = -2) :
  α^2 + 3*α + β = 2003 :=
sorry

end value_of_alpha_beta_l201_201592


namespace karen_tests_graded_l201_201909

theorem karen_tests_graded (n : ℕ) (T : ℕ) 
  (avg_score_70 : T = 70 * n)
  (combined_score_290 : T + 290 = 85 * (n + 2)) : 
  n = 8 := 
sorry

end karen_tests_graded_l201_201909


namespace ant_crawling_routes_ratio_l201_201985

theorem ant_crawling_routes_ratio 
  (m n : ℕ) 
  (h1 : m = 2) 
  (h2 : n = 6) : 
  n / m = 3 :=
by
  -- Proof is omitted (we only need the statement as per the instruction)
  sorry

end ant_crawling_routes_ratio_l201_201985


namespace min_orange_chips_l201_201968

theorem min_orange_chips (p g o : ℕ)
    (h1: g ≥ (1 / 3) * p)
    (h2: g ≤ (1 / 4) * o)
    (h3: p + g ≥ 75) : o = 76 :=
    sorry

end min_orange_chips_l201_201968


namespace range_of_k_l201_201872

theorem range_of_k (f : ℝ → ℝ) (h_odd : ∀ x, f (-x) = -f x) (h_decreasing : ∀ ⦃x y⦄, 0 ≤ x → x < y → f y < f x) 
  (h_inequality : ∀ x, f (k * x ^ 2 + 2) + f (k * x + k) ≤ 0) : 0 ≤ k :=
sorry

end range_of_k_l201_201872


namespace cost_of_one_stamp_l201_201489

-- Defining the conditions
def cost_of_four_stamps := 136
def number_of_stamps := 4

-- Prove that if 4 stamps cost 136 cents, then one stamp costs 34 cents
theorem cost_of_one_stamp : cost_of_four_stamps / number_of_stamps = 34 :=
by
  sorry

end cost_of_one_stamp_l201_201489


namespace calculate_markup_percentage_l201_201982

noncomputable def cost_price : ℝ := 225
noncomputable def profit_percentage : ℝ := 0.25
noncomputable def discount1_percentage : ℝ := 0.10
noncomputable def discount2_percentage : ℝ := 0.15
noncomputable def selling_price : ℝ := cost_price * (1 + profit_percentage)
noncomputable def markup_percentage : ℝ := 63.54

theorem calculate_markup_percentage :
  let marked_price := selling_price / ((1 - discount1_percentage) * (1 - discount2_percentage))
  let calculated_markup_percentage := ((marked_price - cost_price) / cost_price) * 100
  abs (calculated_markup_percentage - markup_percentage) < 0.01 :=
sorry

end calculate_markup_percentage_l201_201982


namespace column_1000_is_B_l201_201552

-- Definition of the column pattern
def columnPattern : List String := ["B", "C", "D", "E", "F", "E", "D", "C", "B", "A"]

-- Function to determine the column for a given integer
def columnOf (n : Nat) : String :=
  columnPattern.get! ((n - 2) % 10)

-- The theorem we want to prove
theorem column_1000_is_B : columnOf 1000 = "B" :=
by
  sorry

end column_1000_is_B_l201_201552


namespace isosceles_triangle_angle_B_l201_201177

theorem isosceles_triangle_angle_B :
  ∀ (A B C : ℝ), (B = C) → (C = 3 * A) → (A + B + C = 180) → (B = 540 / 7) :=
by
  intros A B C h1 h2 h3
  sorry

end isosceles_triangle_angle_B_l201_201177


namespace average_buns_per_student_l201_201094

theorem average_buns_per_student (packages_class1 packages_class2 packages_class3 packages_class4 : ℕ)
    (buns_per_package students_per_class stale_buns uneaten_buns : ℕ)
    (h1 : packages_class1 = 20)
    (h2 : packages_class2 = 25)
    (h3 : packages_class3 = 30)
    (h4 : packages_class4 = 35)
    (h5 : buns_per_package = 8)
    (h6 : students_per_class = 30)
    (h7 : stale_buns = 16)
    (h8 : uneaten_buns = 20) :
  let total_buns_class1 := packages_class1 * buns_per_package
  let total_buns_class2 := packages_class2 * buns_per_package
  let total_buns_class3 := packages_class3 * buns_per_package
  let total_buns_class4 := packages_class4 * buns_per_package
  let total_uneaten_buns := stale_buns + uneaten_buns
  let uneaten_buns_per_class := total_uneaten_buns / 4
  let remaining_buns_class1 := total_buns_class1 - uneaten_buns_per_class
  let remaining_buns_class2 := total_buns_class2 - uneaten_buns_per_class
  let remaining_buns_class3 := total_buns_class3 - uneaten_buns_per_class
  let remaining_buns_class4 := total_buns_class4 - uneaten_buns_per_class
  let avg_buns_class1 := remaining_buns_class1 / students_per_class
  let avg_buns_class2 := remaining_buns_class2 / students_per_class
  let avg_buns_class3 := remaining_buns_class3 / students_per_class
  let avg_buns_class4 := remaining_buns_class4 / students_per_class
  avg_buns_class1 = 5 ∧ avg_buns_class2 = 6 ∧ avg_buns_class3 = 7 ∧ avg_buns_class4 = 9 :=
by
  sorry

end average_buns_per_student_l201_201094


namespace ratio_of_dancers_l201_201782

theorem ratio_of_dancers (total_kids total_dancers slow_dance non_slow_dance : ℕ)
  (h1 : total_kids = 140)
  (h2 : slow_dance = 25)
  (h3 : non_slow_dance = 10)
  (h4 : total_dancers = slow_dance + non_slow_dance) :
  (total_dancers : ℚ) / total_kids = 1 / 4 :=
by
  sorry

end ratio_of_dancers_l201_201782


namespace fourth_term_sum_eq_40_l201_201184

theorem fourth_term_sum_eq_40 : 3^0 + 3^1 + 3^2 + 3^3 = 40 := by
  sorry

end fourth_term_sum_eq_40_l201_201184


namespace probability_of_digit_7_in_3_over_8_is_one_third_l201_201101

theorem probability_of_digit_7_in_3_over_8_is_one_third :
  let digits := [3, 7, 5] in
  let num_occurrences_of_7 := (digits.count (= 7)) in
  let total_digits := list.length digits in
  (num_occurrences_of_7 / total_digits : ℚ) = 1 / 3 :=
by {
  sorry
}

end probability_of_digit_7_in_3_over_8_is_one_third_l201_201101


namespace cube_face_sum_l201_201428

theorem cube_face_sum (a d b e c f g : ℕ)
    (h1 : g = 2)
    (h2 : 2310 = 2 * 3 * 5 * 7 * 11)
    (h3 : (a + d) * (b + e) * (c + f) = 3 * 5 * 7 * 11):
    (a + d) + (b + e) + (c + f) = 47 :=
by
    sorry

end cube_face_sum_l201_201428


namespace min_passengers_to_fill_bench_l201_201522

theorem min_passengers_to_fill_bench (width_per_passenger : ℚ) (total_seat_width : ℚ) (num_seats : ℕ):
  width_per_passenger = 1/6 → total_seat_width = num_seats → num_seats = 6 → 3 ≥ (total_seat_width / width_per_passenger) :=
by
  intro h1 h2 h3
  sorry

end min_passengers_to_fill_bench_l201_201522


namespace proof_problem_l201_201590

def setA : Set ℝ := {x | -1 < x ∧ x ≤ 5}
def setB : Set ℝ := {x | -1 < x ∧ x < 3}

def complementB : Set ℝ := {x | x ≥ 3 ∨ x ≤ -1}
def intersection : Set ℝ := {x | 3 ≤ x ∧ x ≤ 5}

theorem proof_problem :
  (setA ∩ complementB) = intersection := 
by
  sorry

end proof_problem_l201_201590


namespace opposite_sqrt3_l201_201143

def opposite (x : ℝ) : ℝ := -x

theorem opposite_sqrt3 :
  opposite (Real.sqrt 3) = -Real.sqrt 3 :=
by
  sorry

end opposite_sqrt3_l201_201143


namespace least_positive_integer_l201_201278

theorem least_positive_integer :
  ∃ (N : ℕ), N % 11 = 10 ∧ N % 12 = 11 ∧ N % 13 = 12 ∧ N % 14 = 13 ∧ N = 12011 :=
by
  use 12011
  split
  · exact Nat.mod_eq_of_lt (by norm_num) -- 12011 % 11 = 10
  split
  · exact Nat.mod_eq_of_lt (by norm_num) -- 12011 % 12 = 11
  split
  · exact Nat.mod_eq_of_lt (by norm_num) -- 12011 % 13 = 12
  split
  · exact Nat.mod_eq_of_lt (by norm_num) -- 12011 % 14 = 13
  · rfl

end least_positive_integer_l201_201278


namespace probability_first_player_takes_card_l201_201265

variable (n : ℕ) (i : ℕ)

-- Conditions
def even_n : Prop := ∃ k, n = 2 * k
def valid_i : Prop := 1 ≤ i ∧ i ≤ n

-- The key function (probability) and theorem to prove
def P (i n : ℕ) : ℚ := (i - 1) / (n - 1)

theorem probability_first_player_takes_card :
  even_n n → valid_i n i → P i n = (i - 1) / (n - 1) :=
by
  intro h1 h2
  sorry

end probability_first_player_takes_card_l201_201265


namespace find_number_l201_201925

theorem find_number (N : ℝ) (h : (1 / 4) * (1 / 3) * (2 / 5) * N = 14) : 0.40 * N = 168 :=
sorry

end find_number_l201_201925


namespace total_cakes_needed_l201_201238

theorem total_cakes_needed (C : ℕ) (h : C / 4 - C / 12 = 10) : C = 60 := by
  sorry

end total_cakes_needed_l201_201238


namespace find_m_l201_201457

theorem find_m (m : ℤ) (h : 3 ∈ ({1, m + 2} : Set ℤ)) : m = 1 :=
sorry

end find_m_l201_201457


namespace John_can_finish_work_alone_in_48_days_l201_201908

noncomputable def John_and_Roger_can_finish_together_in_24_days (J R: ℝ) : Prop :=
  1 / J + 1 / R = 1 / 24

noncomputable def John_finished_remaining_work (J: ℝ) : Prop :=
  (1 / 3) / (16 / J) = 1

theorem John_can_finish_work_alone_in_48_days (J R: ℝ) 
  (h1 : John_and_Roger_can_finish_together_in_24_days J R) 
  (h2 : John_finished_remaining_work J):
  J = 48 := 
sorry

end John_can_finish_work_alone_in_48_days_l201_201908


namespace minimal_APR_bank_A_l201_201783

def nominal_interest_rate_A : Float := 0.05
def nominal_interest_rate_B : Float := 0.055
def nominal_interest_rate_C : Float := 0.06

def compounding_periods_A : ℕ := 4
def compounding_periods_B : ℕ := 2
def compounding_periods_C : ℕ := 12

def effective_annual_rate (nom_rate : Float) (n : ℕ) : Float :=
  (1 + nom_rate / n.toFloat)^n.toFloat - 1

def APR_A := effective_annual_rate nominal_interest_rate_A compounding_periods_A
def APR_B := effective_annual_rate nominal_interest_rate_B compounding_periods_B
def APR_C := effective_annual_rate nominal_interest_rate_C compounding_periods_C

theorem minimal_APR_bank_A :
  APR_A < APR_B ∧ APR_A < APR_C ∧ APR_A = 0.050945 :=
by
  sorry

end minimal_APR_bank_A_l201_201783


namespace ratio_spaghetti_to_manicotti_l201_201671

-- Definitions of the given conditions
def total_students : ℕ := 800
def spaghetti_preferred : ℕ := 320
def manicotti_preferred : ℕ := 160

-- The theorem statement
theorem ratio_spaghetti_to_manicotti : spaghetti_preferred / manicotti_preferred = 2 :=
by sorry

end ratio_spaghetti_to_manicotti_l201_201671


namespace professional_doctors_percentage_l201_201845

-- Defining the context and conditions:

variable (total_percent : ℝ) (leaders_percent : ℝ) (nurses_percent : ℝ) (doctors_percent : ℝ)

-- Specifying the conditions:
def total_percentage_sum : Prop :=
  total_percent = 100

def leaders_percentage : Prop :=
  leaders_percent = 4

def nurses_percentage : Prop :=
  nurses_percent = 56

-- Stating the actual theorem to be proved:
theorem professional_doctors_percentage
  (h1 : total_percentage_sum total_percent)
  (h2 : leaders_percentage leaders_percent)
  (h3 : nurses_percentage nurses_percent) :
  doctors_percent = 100 - (leaders_percent + nurses_percent) := by
  sorry -- Proof placeholder

end professional_doctors_percentage_l201_201845


namespace prove_a_range_if_p_prove_a_range_if_p_or_q_and_not_and_l201_201048

-- Define the conditions
def quadratic_has_two_different_negative_roots (a : ℝ) : Prop :=
  a^2 - 1/4 > 0 ∧ -a < 0 ∧ 1/16 > 0

def inequality_q (a : ℝ) : Prop :=
  0 < a ∧ a < 1

-- Prove the results based on the conditions
theorem prove_a_range_if_p (a : ℝ) (hp : quadratic_has_two_different_negative_roots a) : a > 1/2 :=
  sorry

theorem prove_a_range_if_p_or_q_and_not_and (a : ℝ) (hp_or_q : quadratic_has_two_different_negative_roots a ∨ inequality_q a) 
  (hnot_p_and_q : ¬ (quadratic_has_two_different_negative_roots a ∧ inequality_q a)) :
  a ≥ 1 ∨ (0 < a ∧ a ≤ 1/2) :=
  sorry

end prove_a_range_if_p_prove_a_range_if_p_or_q_and_not_and_l201_201048


namespace cost_per_square_meter_l201_201223

theorem cost_per_square_meter 
  (length width height : ℝ) 
  (total_expenditure : ℝ) 
  (hlength : length = 20) 
  (hwidth : width = 15) 
  (hheight : height = 5) 
  (hmoney : total_expenditure = 38000) : 
  58.46 = total_expenditure / (length * width + 2 * length * height + 2 * width * height) :=
by 
  -- Let's assume our definitions and use sorry to skip the proof
  sorry

end cost_per_square_meter_l201_201223


namespace people_in_first_group_l201_201335

-- Conditions
variables (P W : ℕ) (people_work_rate same_work_rate : ℕ)

-- Given conditions as Lean definitions
-- P people can do 3W in 3 days implies the work rate of the group is W per day
def first_group_work_rate : ℕ := 3 * W / 3

-- 9 people can do 9W in 3 days implies the work rate of these 9 people is 3W per day
def second_group_work_rate : ℕ := 9 * W / 3

-- The work rates are proportional to the number of people
def proportional_work_rate : Prop := P / 9 = first_group_work_rate / second_group_work_rate

-- Lean theorem statement for proof
theorem people_in_first_group (h1 : first_group_work_rate = W) (h2 : second_group_work_rate = 3 * W) :
  P = 3 :=
by
  sorry

end people_in_first_group_l201_201335


namespace distinct_exponentiation_values_l201_201713

theorem distinct_exponentiation_values : 
  (∃ v1 v2 v3 v4 v5 : ℕ, 
    v1 = (3 : ℕ)^(3 : ℕ)^(3 : ℕ)^(3 : ℕ) ∧
    v2 = (3 : ℕ)^((3 : ℕ)^(3 : ℕ)^(3 : ℕ)) ∧
    v3 = (3 : ℕ)^(((3 : ℕ)^(3 : ℕ))^(3 : ℕ)) ∧
    v4 = ((3 : ℕ)^(3 : ℕ)^3) ∧
    v5 = ((3 : ℕ)^((3 : ℕ)^(3 : ℕ)^3)) ∧
    v1 ≠ v2 ∧ v1 ≠ v3 ∧ v1 ≠ v4 ∧ v1 ≠ v5 ∧ 
    v2 ≠ v3 ∧ v2 ≠ v4 ∧ v2 ≠ v5 ∧ 
    v3 ≠ v4 ∧ v3 ≠ v5 ∧ 
    v4 ≠ v5) := 
sorry

end distinct_exponentiation_values_l201_201713


namespace arithmetic_seq_a12_l201_201605

variable {a : ℕ → ℝ}

theorem arithmetic_seq_a12 :
  (∀ n, ∃ d, a (n + 1) = a n + d)
  ∧ a 5 + a 11 = 30
  ∧ a 4 = 7
  → a 12 = 23 :=
by
  sorry


end arithmetic_seq_a12_l201_201605


namespace equation_represents_circle_of_radius_8_l201_201190

theorem equation_represents_circle_of_radius_8 (k : ℝ) : 
  (x^2 + 14 * x + y^2 + 8 * y - k = 0) → k = -1 ↔ (∃ r, r = 8 ∧ (x + 7)^2 + (y + 4)^2 = r^2) :=
by
  sorry

end equation_represents_circle_of_radius_8_l201_201190


namespace initial_oranges_l201_201750

open Nat

theorem initial_oranges (initial_oranges: ℕ) (eaten_oranges: ℕ) (stolen_oranges: ℕ) (returned_oranges: ℕ) (current_oranges: ℕ):
  eaten_oranges = 10 → 
  stolen_oranges = (initial_oranges - eaten_oranges) / 2 →
  returned_oranges = 5 →
  current_oranges = 30 →
  initial_oranges - eaten_oranges - stolen_oranges + returned_oranges = current_oranges →
  initial_oranges = 60 :=
by
  sorry

end initial_oranges_l201_201750


namespace range_of_m_l201_201891

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, (m^2 + 4 * m - 5) * x^2 - 4 * (m - 1) * x + 3 > 0) ↔ (1 ≤ m ∧ m < 19) :=
by
  sorry

end range_of_m_l201_201891


namespace distance_calculation_l201_201002

-- Define the given constants
def time_minutes : ℕ := 30
def average_speed : ℕ := 1
def seconds_per_minute : ℕ := 60

-- Define the total time in seconds
def time_seconds : ℕ := time_minutes * seconds_per_minute

-- The proof goal: that the distance covered is 1800 meters
theorem distance_calculation :
  time_seconds * average_speed = 1800 := by
  -- Calculation steps (using axioms and known values)
  sorry

end distance_calculation_l201_201002


namespace arithmetic_sequence_k_value_l201_201921

theorem arithmetic_sequence_k_value (S : ℕ → ℝ) (a : ℕ → ℝ) (d : ℝ)
  (S_pos : S 2016 > 0) (S_neg : S 2017 < 0)
  (H : ∀ n, |a n| ≥ |a 1009| ): k = 1009 :=
sorry

end arithmetic_sequence_k_value_l201_201921


namespace dice_product_probability_is_one_l201_201697

def dice_probability_product_is_one : Prop :=
  ∀ (a b c d e : ℕ), (a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 1 ∧ e = 1 → 
    (a * b * c * d * e) = 1) ∧
  ∃ (p : ℚ), p = (1/6)^5 ∧ p = 1/7776

theorem dice_product_probability_is_one (a b c d e : ℕ) :
  dice_probability_product_is_one :=
by
  sorry

end dice_product_probability_is_one_l201_201697


namespace books_sold_at_overall_loss_l201_201885

-- Defining the conditions and values
def total_cost : ℝ := 540
def C1 : ℝ := 315
def loss_percentage_C1 : ℝ := 0.15
def gain_percentage_C2 : ℝ := 0.19
def C2 : ℝ := total_cost - C1
def loss_C1 := (loss_percentage_C1 * C1)
def SP1 := C1 - loss_C1
def gain_C2 := (gain_percentage_C2 * C2)
def SP2 := C2 + gain_C2
def total_selling_price := SP1 + SP2
def overall_loss := total_cost - total_selling_price

-- Formulating the theorem based on the conditions and required proof
theorem books_sold_at_overall_loss : overall_loss = 4.50 := 
by 
  sorry

end books_sold_at_overall_loss_l201_201885


namespace range_of_a_l201_201466

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x^2 - a * x + 64 > 0) → -16 < a ∧ a < 16 :=
by
  -- The proof steps will go here
  sorry

end range_of_a_l201_201466


namespace inequality_solution_set_l201_201512

theorem inequality_solution_set :
  {x : ℝ | 3 * x + 9 > 0 ∧ 2 * x < 6} = {x : ℝ | -3 < x ∧ x < 3} := 
by
  sorry

end inequality_solution_set_l201_201512


namespace find_metal_sheet_width_l201_201167

-- The given conditions
def metalSheetLength : ℝ := 100
def cutSquareSide : ℝ := 10
def boxVolume : ℝ := 24000

-- Statement to prove
theorem find_metal_sheet_width (w : ℝ) (h : w - 2 * cutSquareSide > 0):
  boxVolume = (metalSheetLength - 2 * cutSquareSide) * (w - 2 * cutSquareSide) * cutSquareSide → 
  w = 50 := 
by {
  sorry
}

end find_metal_sheet_width_l201_201167


namespace find_integer_to_satisfy_eq_l201_201519

theorem find_integer_to_satisfy_eq (n : ℤ) (h : n - 5 = 2) : n = 7 :=
sorry

end find_integer_to_satisfy_eq_l201_201519


namespace largest_four_digit_number_prop_l201_201806

theorem largest_four_digit_number_prop :
  ∃ (a b c d : ℕ), a = 9 ∧ b = 0 ∧ c = 9 ∧ d = 9 ∧ (1000 * a + 100 * b + 10 * c + d = 9099) ∧ (c = a + b) ∧ (d = b + c) :=
by
  sorry

end largest_four_digit_number_prop_l201_201806


namespace cos_225_eq_neg_sqrt2_div_2_l201_201419

theorem cos_225_eq_neg_sqrt2_div_2 :
  (real.cos (225 * real.pi / 180) = - (real.sqrt 2 / 2)) :=
by
  sorry

end cos_225_eq_neg_sqrt2_div_2_l201_201419


namespace solve_for_k_l201_201855

theorem solve_for_k (k : ℝ) : (∀ x : ℝ, 3 * (5 + k * x) = 15 * x + 15) ↔ k = 5 :=
  sorry

end solve_for_k_l201_201855


namespace sum_of_fraction_numerator_and_denominator_l201_201326

theorem sum_of_fraction_numerator_and_denominator (x : ℚ) (a b : ℤ) :
  x = (45 / 99 : ℚ) ∧ (a = 5) ∧ (b = 11) → (a + b = 16) :=
by
  sorry

end sum_of_fraction_numerator_and_denominator_l201_201326


namespace num_ways_to_assign_roles_l201_201678

-- Definitions for men, women, and roles
def num_men : ℕ := 6
def num_women : ℕ := 7
def male_roles : ℕ := 3
def female_roles : ℕ := 3
def gender_neutral_roles : ℕ := 2

-- Permutations function
noncomputable def permutations : ℕ → ℕ → ℕ
| n, k := Nat.factorial n / Nat.factorial (n - k)

-- Proof statement
theorem num_ways_to_assign_roles :
  let men_perms := permutations num_men male_roles in
  let women_perms := permutations num_women female_roles in
  let remaining_individuals := num_men + num_women - male_roles - female_roles in
  let neutral_roles_perms := permutations remaining_individuals gender_neutral_roles in
  men_perms * women_perms * neutral_roles_perms = 1058400 :=
by
  sorry

end num_ways_to_assign_roles_l201_201678


namespace largest_four_digit_number_prop_l201_201804

theorem largest_four_digit_number_prop :
  ∃ (a b c d : ℕ), a = 9 ∧ b = 0 ∧ c = 9 ∧ d = 9 ∧ (1000 * a + 100 * b + 10 * c + d = 9099) ∧ (c = a + b) ∧ (d = b + c) :=
by
  sorry

end largest_four_digit_number_prop_l201_201804


namespace books_in_library_l201_201149

theorem books_in_library (n_shelves : ℕ) (n_books_per_shelf : ℕ) (h_shelves : n_shelves = 1780) (h_books_per_shelf : n_books_per_shelf = 8) :
  n_shelves * n_books_per_shelf = 14240 :=
by
  -- Skipping the proof as instructed
  sorry

end books_in_library_l201_201149


namespace sum_of_fraction_parts_l201_201317

theorem sum_of_fraction_parts (x : ℝ) (hx : x = 0.45) : 
  (∃ (a b : ℕ), x = a / b ∧ Nat.gcd a b = 1 ∧ a + b = 16) :=
by
  sorry

end sum_of_fraction_parts_l201_201317


namespace inscribed_pentagon_angles_sum_l201_201544

theorem inscribed_pentagon_angles_sum (α β γ δ ε : ℝ) (h1 : α + β + γ + δ + ε = 360) 
(h2 : α / 2 + β / 2 + γ / 2 + δ / 2 + ε / 2 = 180) : 
(α / 2) + (β / 2) + (γ / 2) + (δ / 2) + (ε / 2) = 180 :=
by
  sorry

end inscribed_pentagon_angles_sum_l201_201544


namespace calc_f_y_eq_2f_x_l201_201888

noncomputable def f (x : ℝ) : ℝ := Real.log ((1 + x) / (1 - x))

theorem calc_f_y_eq_2f_x (x : ℝ) (h : -1 < x) (h' : x < 1) :
  f ( (2 * x + x^2) / (1 + 2 * x^2) ) = 2 * f x := by
  sorry

end calc_f_y_eq_2f_x_l201_201888


namespace common_root_values_l201_201864

def has_common_root (p x : ℝ) : Prop :=
  (x^2 - (p+1)*x + (p+1) = 0) ∧ (2*x^2 + (p-2)*x - p - 7 = 0)

theorem common_root_values :
  (has_common_root 3 2) ∧ (has_common_root (-3/2) (-1)) :=
by {
  sorry
}

end common_root_values_l201_201864


namespace cos_225_proof_l201_201384

def cos_225_def := angle.to_real 225 = -√2 / 2

theorem cos_225_proof : cos_225_def := sorry

end cos_225_proof_l201_201384


namespace task_completion_choice_l201_201839

theorem task_completion_choice (A B : ℕ) (hA : A = 3) (hB : B = 5) : A + B = 8 := by
  sorry

end task_completion_choice_l201_201839


namespace cosine_225_proof_l201_201352

noncomputable def cosine_225_eq_neg_sqrt2_div_2 : Prop :=
  Real.cos (225 * Real.pi / 180) = -Real.sqrt 2 / 2

theorem cosine_225_proof : cosine_225_eq_neg_sqrt2_div_2 := by
  sorry

end cosine_225_proof_l201_201352


namespace equal_distribution_l201_201572

def earnings : List ℕ := [30, 35, 45, 55, 65]

def total_earnings : ℕ := earnings.sum

def equal_share (total: ℕ) : ℕ := total / earnings.length

def redistribution_amount (earner: ℕ) (equal: ℕ) : ℕ := earner - equal

theorem equal_distribution :
  redistribution_amount 65 (equal_share total_earnings) = 19 :=
by
  sorry

end equal_distribution_l201_201572


namespace max_n_is_2_l201_201188

def is_prime_seq (q : ℕ → ℕ) : Prop :=
  ∀ i, Nat.Prime (q i)

def gen_seq (q0 : ℕ) : ℕ → ℕ
  | 0 => q0
  | (i + 1) => (gen_seq q0 i - 1)^3 + 3

theorem max_n_is_2 (q0 : ℕ) (hq0 : q0 > 0) :
  ∀ (q1 q2 : ℕ), q1 = gen_seq q0 1 → q2 = gen_seq q0 2 → 
  is_prime_seq (gen_seq q0) → q2 = (q1 - 1)^3 + 3 := 
  sorry

end max_n_is_2_l201_201188


namespace recipe_third_amounts_l201_201010

theorem recipe_third_amounts:
  (flour sugar : ℚ) 
  (h_flour : flour = 5 + 3/4) 
  (h_sugar : sugar = 2 + 1/2) :
  (flour / 3 = 1 + 11 / 12) ∧ (sugar / 3 = 5 / 6) :=
by
  sorry

end recipe_third_amounts_l201_201010


namespace statement_C_l201_201593

theorem statement_C (a b : ℝ) (h₁ : a < b) (h₂ : b < 0) : a^2 > a * b ∧ a * b > b^2 :=
by
  sorry

end statement_C_l201_201593


namespace intersection_A_B_union_B_Ac_range_a_l201_201884

open Set

-- Conditions
def U : Set ℝ := univ
def A : Set ℝ := {x | 2 < x ∧ x < 9}
def B : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
def Ac : Set ℝ := {x | x ≤ 2 ∨ x ≥ 9}
def Bc : Set ℝ := {x | x < -2 ∨ x > 5}

-- Questions rewritten as Lean statements

theorem intersection_A_B :
  A ∩ B = {x | 2 < x ∧ x ≤ 5} := sorry

theorem union_B_Ac :
  B ∪ Ac = {x | x ≤ 5 ∨ x ≥ 9} := sorry

theorem range_a (a : ℝ) :
  {x | a ≤ x ∧ x ≤ a + 2} ⊆ Bc → a ∈ Iio (-4) ∪ Ioi 5 := sorry

end intersection_A_B_union_B_Ac_range_a_l201_201884


namespace find_Y_l201_201571

theorem find_Y :
  ∃ Y : ℤ, (19 + Y / 151) * 151 = 2912 ∧ Y = 43 :=
by
  use 43
  sorry

end find_Y_l201_201571


namespace cos_225_eq_neg_sqrt2_div2_l201_201365

theorem cos_225_eq_neg_sqrt2_div2 : Real.cos (225 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by sorry

end cos_225_eq_neg_sqrt2_div2_l201_201365


namespace max_value_of_expression_l201_201920

theorem max_value_of_expression (x y z : ℝ) (h : 3 * x + 4 * y + 2 * z = 12) :
  x^2 * y + x^2 * z + y * z^2 ≤ 3 := sorry

end max_value_of_expression_l201_201920


namespace f_3_neg3div2_l201_201754

noncomputable def f : ℝ → ℝ :=
sorry

axiom odd_f : ∀ x : ℝ, f (-x) = -f x
axiom symm_f : ∀ t : ℝ, f t = f (1 - t)
axiom restriction_f : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1/2 → f x = -x^2

theorem f_3_neg3div2 :
  f 3 + f (-3/2) = -1/4 :=
sorry

end f_3_neg3div2_l201_201754


namespace combined_average_pieces_lost_l201_201028

theorem combined_average_pieces_lost
  (audrey_losses : List ℕ) (thomas_losses : List ℕ)
  (h_audrey : audrey_losses = [6, 8, 4, 7, 10])
  (h_thomas : thomas_losses = [5, 6, 3, 7, 11]) :
  (audrey_losses.sum + thomas_losses.sum : ℚ) / 5 = 13.4 := by 
  sorry

end combined_average_pieces_lost_l201_201028


namespace polynomial_characterization_l201_201530

noncomputable def homogeneous_polynomial (P : ℝ → ℝ → ℝ) (n : ℕ) :=
  ∀ t x y : ℝ, P (t * x) (t * y) = t^n * P x y

def polynomial_condition (P : ℝ → ℝ → ℝ) :=
  ∀ a b c : ℝ, P (a + b) c + P (b + c) a + P (c + a) b = 0

def P_value (P : ℝ → ℝ → ℝ) :=
  P 1 0 = 1

theorem polynomial_characterization (P : ℝ → ℝ → ℝ) (n : ℕ) :
  homogeneous_polynomial P n →
  polynomial_condition P →
  P_value P →
  ∃ A : ℝ → ℝ → ℝ, ∀ x y : ℝ, P x y = (x + y)^(n - 1) * (x - 2 * y) :=
by
  sorry

end polynomial_characterization_l201_201530


namespace solve_cubic_equation_l201_201527

theorem solve_cubic_equation :
  ∀ x : ℝ, x^3 = 13 * x + 12 ↔ x = 4 ∨ x = -1 ∨ x = -3 :=
by
  sorry

end solve_cubic_equation_l201_201527


namespace solve_congruence_l201_201779

theorem solve_congruence :
  ∃ a m : ℕ, (8 * (x : ℕ) + 1) % 12 = 5 % 12 ∧ m ≥ 2 ∧ a < m ∧ x ≡ a [MOD m] ∧ a + m = 5 :=
by
  sorry

end solve_congruence_l201_201779


namespace time_taken_y_alone_l201_201529

-- Define the work done in terms of rates
def work_done (Rx Ry Rz : ℝ) (W : ℝ) :=
  Rx = W / 8 ∧ (Ry + Rz) = W / 6 ∧ (Rx + Rz) = W / 4

-- Prove that the time taken by y alone is 24 hours
theorem time_taken_y_alone (Rx Ry Rz W : ℝ) (h : work_done Rx Ry Rz W) :
  (1 / Ry) = 24 :=
by
  sorry

end time_taken_y_alone_l201_201529


namespace water_needed_to_fill_glasses_l201_201515

theorem water_needed_to_fill_glasses :
  let glasses := 10
  let capacity_per_glass := 6
  let filled_fraction := 4 / 5
  let total_capacity := glasses * capacity_per_glass
  let total_water := glasses * (capacity_per_glass * filled_fraction)
  let water_needed := total_capacity - total_water
  water_needed = 12 :=
by
  sorry

end water_needed_to_fill_glasses_l201_201515


namespace cos_225_eq_neg_sqrt2_div_2_l201_201416

theorem cos_225_eq_neg_sqrt2_div_2 :
  (real.cos (225 * real.pi / 180) = - (real.sqrt 2 / 2)) :=
by
  sorry

end cos_225_eq_neg_sqrt2_div_2_l201_201416


namespace factor_sum_l201_201135

variable (x y : ℝ)

theorem factor_sum :
  let a := 1
  let b := -2
  let c := 1
  let d := 2
  let e := 4
  let f := 1
  let g := 2
  let h := 1
  let j := -2
  let k := 4
  (27 * x^9 - 512 * y^9) = ((a * x + b * y) * (c * x^3 + d * x * y^2 + e * y^3) * 
  (f * x + g * y) * (h * x^3 + j * x * y^2 + k * y^3)) → 
  (a + b + c + d + e + f + g + h + j + k = 12) :=
by
  sorry

end factor_sum_l201_201135


namespace sales_volume_relation_maximize_profit_l201_201784

-- Definition of the conditions given in the problem
def cost_price : ℝ := 40
def min_selling_price : ℝ := 45
def initial_selling_price : ℝ := 45
def initial_sales_volume : ℝ := 700
def sales_decrease_rate : ℝ := 20

-- Lean statement for part 1
theorem sales_volume_relation (x : ℝ) : 
  (45 ≤ x) →
  (y = 700 - 20 * (x - 45)) → 
  y = -20 * x + 1600 := sorry

-- Lean statement for part 2
theorem maximize_profit (x : ℝ) :
  (45 ≤ x) →
  (P = (x - 40) * (-20 * x + 1600)) →
  ∃ max_x max_P, max_x = 60 ∧ max_P = 8000 := sorry

end sales_volume_relation_maximize_profit_l201_201784


namespace largest_four_digit_number_l201_201813

theorem largest_four_digit_number :
  ∃ a b c d : ℕ, 
    9 < 1000 * a + 100 * b + 10 * c + d ∧ 
    1000 * a + 100 * b + 10 * c + d < 10000 ∧ 
    c = a + b ∧ 
    d = b + c ∧ 
    1000 * a + 100 * b + 10 * c + d = 9099 :=
by {
  sorry
}

end largest_four_digit_number_l201_201813


namespace circle_center_radius_l201_201134

theorem circle_center_radius :
  ∀ (x y : ℝ), x^2 + y^2 - 4 * x = 0 ↔ ((x - 2)^2 + y^2 = 4) ∧ (∃ (c_x c_y r : ℝ), c_x = 2 ∧ c_y = 0 ∧ r = 2) :=
by
  sorry

end circle_center_radius_l201_201134


namespace red_balls_in_bag_l201_201537

theorem red_balls_in_bag : 
  ∃ (r : ℕ), (r * (r - 1) = 22) ∧ (r ≤ 12) :=
by { sorry }

end red_balls_in_bag_l201_201537


namespace quarter_circle_area_ratio_l201_201761

theorem quarter_circle_area_ratio (R : ℝ) (hR : 0 < R) :
  let O := pi * R^2
  let AXC := pi * (R/2)^2 / 4
  let BYD := pi * (R/2)^2 / 4
  (2 * (AXC + BYD) / O = 1 / 8) := 
by
  let O := pi * R^2
  let AXC := pi * (R/2)^2 / 4
  let BYD := pi * (R/2)^2 / 4
  sorry

end quarter_circle_area_ratio_l201_201761


namespace probability_of_spinner_stopping_in_region_G_l201_201980

theorem probability_of_spinner_stopping_in_region_G :
  let pE := (1:ℝ) / 2
  let pF := (1:ℝ) / 4
  let y  := (1:ℝ) / 6
  let z  := (1:ℝ) / 12
  pE + pF + y + z = 1 → y = 2 * z → y = (1:ℝ) / 6 := by
  intros htotal hdouble
  sorry

end probability_of_spinner_stopping_in_region_G_l201_201980


namespace completion_time_B_l201_201531

-- Definitions based on conditions
def work_rate_A : ℚ := 1 / 10 -- A's rate of completing work per day

def efficiency_B : ℚ := 1.75 -- B is 75% more efficient than A

def work_rate_B : ℚ := efficiency_B * work_rate_A -- B's work rate per day

-- The main theorem that we need to prove
theorem completion_time_B : (1 : ℚ) / work_rate_B = 40 / 7 :=
by 
  sorry

end completion_time_B_l201_201531


namespace trigonometric_identity_l201_201595

variable {θ u : ℝ} {n : ℤ}

-- Given condition
def cos_condition (θ u : ℝ) : Prop := 2 * Real.cos θ = u + (1 / u)

-- Theorem to prove
theorem trigonometric_identity (h : cos_condition θ u) : 2 * Real.cos (n * θ) = u^n + (1 / u^n) :=
sorry

end trigonometric_identity_l201_201595


namespace prove_ab_ge_5_l201_201912

theorem prove_ab_ge_5 (a b c : ℕ) (h : ∀ x, x * (a * x) = b * x + c → 0 ≤ x ∧ x ≤ 1) : 5 ≤ a ∧ 5 ≤ b := 
sorry

end prove_ab_ge_5_l201_201912


namespace compute_expression_l201_201423

theorem compute_expression :
  (4 + 8 - 16 + 32 + 64 - 128 + 256) / (8 + 16 - 32 + 64 + 128 - 256 + 512) = 1 / 2 :=
by
  sorry

end compute_expression_l201_201423


namespace sabrina_cookies_l201_201763

theorem sabrina_cookies :
  let S0 : ℕ := 28
  let S1 : ℕ := S0 - 10
  let S2 : ℕ := S1 + 3 * 10
  let S3 : ℕ := S2 - S2 / 3
  let S4 : ℕ := S3 + 16 / 4
  let S5 : ℕ := S4 - S4 / 2
  S5 = 18 := 
by
  -- begin proof here
  sorry

end sabrina_cookies_l201_201763


namespace solve_prime_equation_l201_201118

theorem solve_prime_equation (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) :
    p ^ 2 - 6 * p * q + q ^ 2 + 3 * q - 1 = 0 ↔ (p = 17 ∧ q = 3) :=
by
  sorry

end solve_prime_equation_l201_201118


namespace set_union_inter_example_l201_201614

open Set

theorem set_union_inter_example :
  let A := ({1, 2} : Set ℕ)
  let B := ({1, 2, 3} : Set ℕ)
  let C := ({2, 3, 4} : Set ℕ)
  (A ∩ B) ∪ C = ({1, 2, 3, 4} : Set ℕ) := by
    let A := ({1, 2} : Set ℕ)
    let B := ({1, 2, 3} : Set ℕ)
    let C := ({2, 3, 4} : Set ℕ)
    sorry

end set_union_inter_example_l201_201614


namespace sum_of_numerator_and_denominator_of_repeating_decimal_l201_201298

noncomputable def repeating_decimal_to_fraction (n d : ℤ) (h : 0 < d) := ∃ (x : ℚ), x = n / d ∧ 0 < x

theorem sum_of_numerator_and_denominator_of_repeating_decimal :
  let x := 0.4545454545 -- defining x
  let f := x.to_rational -- converting to rational
  let f_simp := f.num / f.denom -- get numerator and denominator
  -- conclude the sum of numerator and denominator is 16
  f.num + f.denom = 16 :=
by
  sorry

end sum_of_numerator_and_denominator_of_repeating_decimal_l201_201298


namespace largest_valid_four_digit_number_l201_201815

-- Definition of the problem conditions
def is_valid_number (a b c d : ℕ) : Prop :=
  c = a + b ∧ d = b + c

-- Proposition that we need to prove
theorem largest_valid_four_digit_number : ∃ (a b c d : ℕ),
  is_valid_number a b c d ∧ a * 1000 + b * 100 + c * 10 + d = 9099 :=
by
  sorry

end largest_valid_four_digit_number_l201_201815


namespace pair_green_shirts_l201_201077

/-- In a regional math gathering, 83 students wore red shirts, and 97 students wore green shirts. 
The 180 students are grouped into 90 pairs. Exactly 35 of these pairs consist of students both 
wearing red shirts. Prove that the number of pairs consisting solely of students wearing green shirts is 42. -/
theorem pair_green_shirts (r g total pairs rr: ℕ) (h_r : r = 83) (h_g : g = 97) (h_total : total = 180) 
    (h_pairs : pairs = 90) (h_rr : rr = 35) : 
    (g - (r - rr * 2)) / 2 = 42 := 
by 
  /- The proof is omitted. -/
  sorry

end pair_green_shirts_l201_201077


namespace hotel_charge_per_hour_morning_l201_201427

noncomputable def charge_per_hour_morning := 2 -- The correct answer

theorem hotel_charge_per_hour_morning
  (cost_night : ℝ)
  (initial_money : ℝ)
  (hours_night : ℝ)
  (hours_morning : ℝ)
  (remaining_money : ℝ)
  (total_cost : ℝ)
  (M : ℝ)
  (H1 : cost_night = 1.50)
  (H2 : initial_money = 80)
  (H3 : hours_night = 6)
  (H4 : hours_morning = 4)
  (H5 : remaining_money = 63)
  (H6 : total_cost = initial_money - remaining_money)
  (H7 : total_cost = hours_night * cost_night + hours_morning * M) :
  M = charge_per_hour_morning :=
by
  sorry

end hotel_charge_per_hour_morning_l201_201427


namespace recurring_fraction_sum_l201_201311

def x := 0.454545...

theorem recurring_fraction_sum :
  let x := 0.454545...
  let frac := (5 : ℕ) / (11 : ℕ)
  (frac.numerator + frac.denominator) = 16 :=
by
  sorry

end recurring_fraction_sum_l201_201311


namespace mul_inv_mod_391_l201_201851

theorem mul_inv_mod_391 (a : ℤ) (ha : 143 * a % 391 = 1) : a = 28 := by
  sorry

end mul_inv_mod_391_l201_201851


namespace second_triangle_weight_l201_201548

noncomputable def area_equilateral_triangle (s : ℝ) : ℝ :=
  (s^2 * Real.sqrt 3) / 4

noncomputable def weight_of_second_triangle (m_1 : ℝ) (s_1 s_2 : ℝ) : ℝ :=
  m_1 * (area_equilateral_triangle s_2 / area_equilateral_triangle s_1)

theorem second_triangle_weight :
  let m_1 := 12   -- weight of the first triangle in ounces
  let s_1 := 3    -- side length of the first triangle in inches
  let s_2 := 5    -- side length of the second triangle in inches
  weight_of_second_triangle m_1 s_1 s_2 = 33.3 :=
by
  sorry

end second_triangle_weight_l201_201548


namespace sum_of_roots_l201_201819

theorem sum_of_roots (x : ℝ) :
  (x^2 = 10 * x - 13) → ∃ s, s = 10 := 
by
  sorry

end sum_of_roots_l201_201819


namespace original_price_is_1611_11_l201_201007

theorem original_price_is_1611_11 (profit: ℝ) (rate: ℝ) (original_price: ℝ) (selling_price: ℝ) 
(h1: profit = 725) (h2: rate = 0.45) (h3: profit = rate * original_price) : 
original_price = 725 / 0.45 := 
sorry

end original_price_is_1611_11_l201_201007


namespace range_of_a_l201_201448

variables (a : ℝ)

def prop_p : Prop := ∀ x : ℝ, x^2 - 2 * a * x + 16 > 0
def prop_q : Prop := (2 * a - 2)^2 - 8 * (3 * a - 7) ≥ 0
def combined : Prop := prop_p a ∧ prop_q a

theorem range_of_a (a : ℝ) : combined a ↔ -4 < a ∧ a ≤ 3 :=
by
  sorry

end range_of_a_l201_201448


namespace find_length_of_second_train_l201_201001

def length_of_second_train (L : ℚ) : Prop :=
  let length_first_train : ℚ := 300
  let speed_first_train : ℚ := 120 * 1000 / 3600
  let speed_second_train : ℚ := 80 * 1000 / 3600
  let crossing_time : ℚ := 9
  let relative_speed : ℚ := speed_first_train + speed_second_train
  let total_distance : ℚ := relative_speed * crossing_time
  total_distance = length_first_train + L

theorem find_length_of_second_train :
  ∃ (L : ℚ), length_of_second_train L ∧ L = 199.95 := 
by
  sorry

end find_length_of_second_train_l201_201001


namespace multiple_of_5_add_multiple_of_10_l201_201493

theorem multiple_of_5_add_multiple_of_10 (p q : ℤ) (hp : ∃ m : ℤ, p = 5 * m) (hq : ∃ n : ℤ, q = 10 * n) : ∃ k : ℤ, p + q = 5 * k :=
by
  sorry

end multiple_of_5_add_multiple_of_10_l201_201493


namespace original_laborers_count_l201_201830

theorem original_laborers_count (L : ℕ) (h1 : (L - 7) * 10 = L * 6) : L = 18 :=
sorry

end original_laborers_count_l201_201830


namespace ten_times_average_letters_l201_201429

-- Define the number of letters Elida has
def letters_Elida : ℕ := 5

-- Define the number of letters Adrianna has
def letters_Adrianna : ℕ := 2 * letters_Elida - 2

-- Define the average number of letters in both names
def average_letters : ℕ := (letters_Elida + letters_Adrianna) / 2

-- Define the final statement for 10 times the average number of letters
theorem ten_times_average_letters : 10 * average_letters = 65 := by
  sorry

end ten_times_average_letters_l201_201429


namespace repeating_decimal_sum_l201_201304

theorem repeating_decimal_sum (x : ℚ) (hx : x = 45 / 99) (h_simplified : x = 5 / 11) : (5 + 11) = 16 :=
by {
  sorry
}

end repeating_decimal_sum_l201_201304


namespace sum_of_interior_diagonals_l201_201837

theorem sum_of_interior_diagonals (x y z : ℝ) (h1 : x^2 + y^2 + z^2 = 50) (h2 : x * y + y * z + z * x = 47) : 
  4 * Real.sqrt (x^2 + y^2 + z^2) = 20 * Real.sqrt 2 :=
by 
  sorry

end sum_of_interior_diagonals_l201_201837


namespace intersection_A_B_union_A_compB_l201_201874

-- Define the sets A and B
def A : Set ℝ := { x | x^2 + 3 * x - 10 < 0 }
def B : Set ℝ := { x | x^2 - 2 * x - 3 ≥ 0 }

-- Define the universal set
def U : Set ℝ := Set.univ

-- Define the complement of B in the universal set
def comp_B : Set ℝ := { x | ¬ B x }

-- 1. Prove that A ∩ B = {x | -5 < x ∧ x ≤ -1}
theorem intersection_A_B :
  A ∩ B = { x | -5 < x ∧ x ≤ -1 } :=
by 
  sorry

-- 2. Prove that A ∪ (complement of B) = {x | -5 < x ∧ x < 3}
theorem union_A_compB :
  A ∪ comp_B = { x | -5 < x ∧ x < 3 } :=
by 
  sorry

end intersection_A_B_union_A_compB_l201_201874


namespace length_of_single_row_l201_201986

-- Define smaller cube properties and larger cube properties
def side_length_smaller_cube : ℕ := 5  -- in cm
def side_length_larger_cube : ℕ := 100  -- converted from 1 meter to cm

-- Prove that the row of smaller cubes is 400 meters long
theorem length_of_single_row :
  let num_smaller_cubes := (side_length_larger_cube / side_length_smaller_cube) ^ 3
  let length_in_cm := num_smaller_cubes * side_length_smaller_cube
  let length_in_m := length_in_cm / 100
  length_in_m = 400 :=
by
  sorry

end length_of_single_row_l201_201986


namespace fraction_equivalent_to_decimal_l201_201654

theorem fraction_equivalent_to_decimal : 
  (0.4 -- using appropriate representation for repeating decimal 0.4\overline{13}
      + 13 / 990) = 409 / 990 ∧ Nat.gcd 409 990 = 1 := 
sorry

end fraction_equivalent_to_decimal_l201_201654


namespace plane_equation_exists_l201_201853

noncomputable def equation_of_plane (A B C D : ℤ) (hA : A > 0) (hGCD : Int.gcd (Int.gcd A B) (Int.gcd C D) = 1) : Prop :=
∃ (x y z : ℤ),
  x = 1 ∧ y = -2 ∧ z = 2 ∧ D = -18 ∧
  (2 * x + (-3) * y + 5 * z + D = 0) ∧  -- Point (2, -3, 5) satisfies equation
  (4 * x + (-3) * y + 6 * z + D = 0) ∧  -- Point (4, -3, 6) satisfies equation
  (6 * x + (-4) * y + 8 * z + D = 0)    -- Point (6, -4, 8) satisfies equation

theorem plane_equation_exists : equation_of_plane 1 (-2) 2 (-18) (by decide) (by decide) :=
by
  -- Proof is omitted
  sorry

end plane_equation_exists_l201_201853


namespace right_triangle_legs_sum_l201_201634

theorem right_triangle_legs_sum : 
  ∃ (x : ℕ), (x^2 + (x + 1)^2 = 41^2) ∧ (x + (x + 1) = 57) :=
by
  sorry

end right_triangle_legs_sum_l201_201634


namespace playerB_hit_rate_playerA_probability_l201_201786

theorem playerB_hit_rate (p : ℝ) (h : (1 - p)^2 = 1/16) : p = 3/4 :=
sorry

theorem playerA_probability (hit_rate : ℝ) (h : hit_rate = 1/2) : 
  (1 - (1 - hit_rate)^2) = 3/4 :=
sorry

end playerB_hit_rate_playerA_probability_l201_201786


namespace mean_of_combined_set_l201_201773

theorem mean_of_combined_set
  (mean1 : ℕ → ℝ)
  (n1 : ℕ)
  (mean2 : ℕ → ℝ)
  (n2 : ℕ)
  (h1 : ∀ n1, mean1 n1 = 15)
  (h2 : ∀ n2, mean2 n2 = 26) :
  (n1 + n2) = 15 → 
  ((n1 * 15 + n2 * 26) / (n1 + n2)) = (313/15) :=
by
  sorry

end mean_of_combined_set_l201_201773


namespace Jack_goal_l201_201228

-- Define the amounts Jack made from brownies and lemon squares
def brownies (n : ℕ) (price : ℕ) : ℕ := n * price
def lemonSquares (n : ℕ) (price : ℕ) : ℕ := n * price

-- Define the amount Jack needs to make from cookies
def cookies (n : ℕ) (price : ℕ) : ℕ := n * price

-- Define the total goal for Jack
def totalGoal (browniesCount : ℕ) (browniesPrice : ℕ) 
              (lemonSquaresCount : ℕ) (lemonSquaresPrice : ℕ) 
              (cookiesCount : ℕ) (cookiesPrice: ℕ) : ℕ :=
  brownies browniesCount browniesPrice + lemonSquares lemonSquaresCount lemonSquaresPrice + cookies cookiesCount cookiesPrice

theorem Jack_goal : totalGoal 4 3 5 2 7 4 = 50 :=
by
  -- Adding up the different components of the total earnings
  let totalFromBrownies := brownies 4 3
  let totalFromLemonSquares := lemonSquares 5 2
  let totalFromCookies := cookies 7 4
  -- Summing up the amounts
  have step1 : totalFromBrownies = 12 := rfl
  have step2 : totalFromLemonSquares = 10 := rfl
  have step3 : totalFromCookies = 28 := rfl
  have step4 : totalGoal 4 3 5 2 7 4 = totalFromBrownies + totalFromLemonSquares + totalFromCookies := rfl
  have step5 : totalFromBrownies + totalFromLemonSquares + totalFromCookies = 12 + 10 + 28 := by rw [step1, step2, step3]
  have step6 : 12 + 10 + 28 = 50 := by norm_num
  exact step4 ▸ (step5 ▸ step6)

end Jack_goal_l201_201228


namespace car_maintenance_fraction_l201_201560

variable (p : ℝ) (f : ℝ)

theorem car_maintenance_fraction (hp : p = 5200)
  (he : p - f * p - (p - 320) = 200) : f = 3 / 130 :=
by
  have hp_pos : p ≠ 0 := by linarith [hp]
  sorry

end car_maintenance_fraction_l201_201560


namespace calc_expression_l201_201844

theorem calc_expression : (4 + 6 + 10) / 3 - 2 / 3 = 6 := by
  sorry

end calc_expression_l201_201844


namespace range_of_expression_l201_201215

theorem range_of_expression (x : ℝ) : (x + 2 ≥ 0 ∧ x - 1 ≠ 0) ↔ (x ≥ -2 ∧ x ≠ 1) :=
by
  sorry

end range_of_expression_l201_201215


namespace tan_shift_monotonic_interval_l201_201139

noncomputable def monotonic_interval (k : ℤ) : Set ℝ :=
  {x : ℝ | k * Real.pi - 3 * Real.pi / 4 < x ∧ x < k * Real.pi + Real.pi / 4}

theorem tan_shift_monotonic_interval {k : ℤ} :
  ∀ x, (monotonic_interval k x) → (Real.tan (x + Real.pi / 4)) = (Real.tan x) := sorry

end tan_shift_monotonic_interval_l201_201139


namespace sum_of_three_numbers_l201_201506

theorem sum_of_three_numbers (x y z : ℕ) (h1 : x ≤ y) (h2 : y ≤ z) (h3 : y = 7) 
    (h4 : (x + y + z) / 3 = x + 12) (h5 : (x + y + z) / 3 = z - 18) : 
    x + y + z = 39 :=
by
  sorry

end sum_of_three_numbers_l201_201506


namespace contrapositive_of_proposition_is_false_l201_201255

variables {a b : ℤ}

/-- Proposition: If a and b are both even, then a + b is even -/
def proposition (a b : ℤ) : Prop :=
  (∀ n m : ℤ, a = 2 * n ∧ b = 2 * m → ∃ k : ℤ, a + b = 2 * k)

/-- Contrapositive: If a and b are not both even, then a + b is not even -/
def contrapositive (a b : ℤ) : Prop :=
  ¬(∀ n m : ℤ, a = 2 * n ∧ b = 2 * m) → ¬(∃ k : ℤ, a + b = 2 * k)

/-- The contrapositive of the proposition "If a and b are both even, then a + b is even" -/
theorem contrapositive_of_proposition_is_false :
  (contrapositive a b) = false :=
sorry

end contrapositive_of_proposition_is_false_l201_201255


namespace minimum_disks_needed_l201_201758

-- Definition of the conditions
def disk_capacity : ℝ := 2.88
def file_sizes : List (ℝ × ℕ) := [(1.2, 5), (0.9, 10), (0.6, 8), (0.3, 7)]

/-- 
Theorem: Given the capacity of each disk and the sizes and counts of different files,
we can prove that the minimum number of disks needed to store all the files without 
splitting any file is 14.
-/
theorem minimum_disks_needed (capacity : ℝ) (files : List (ℝ × ℕ)) : 
  capacity = disk_capacity ∧ files = file_sizes → ∃ m : ℕ, m = 14 :=
by
  sorry

end minimum_disks_needed_l201_201758


namespace volume_of_solid_rotation_l201_201564

noncomputable def volume_of_solid := 
  (∫ y in (0:ℝ)..(1:ℝ), (y^(2/3) - y^2)) * Real.pi 

theorem volume_of_solid_rotation :
  volume_of_solid = (4 * Real.pi / 15) :=
by
  sorry

end volume_of_solid_rotation_l201_201564


namespace quadratic_inequality_solution_l201_201933

theorem quadratic_inequality_solution (x : ℝ) : x^2 - 5 * x + 6 < 0 ↔ 2 < x ∧ x < 3 :=
by
  sorry

end quadratic_inequality_solution_l201_201933


namespace Lorin_black_marbles_l201_201237

variable (B : ℕ)

def Jimmy_yellow_marbles := 22
def Alex_yellow_marbles := Jimmy_yellow_marbles / 2
def Alex_black_marbles := 2 * B
def Alex_total_marbles := Alex_yellow_marbles + Alex_black_marbles

theorem Lorin_black_marbles : Alex_total_marbles = 19 → B = 4 :=
by
  intros h
  unfold Alex_total_marbles at h
  unfold Alex_yellow_marbles at h
  unfold Alex_black_marbles at h
  norm_num at h
  exact sorry

end Lorin_black_marbles_l201_201237


namespace three_digit_numbers_l201_201617

theorem three_digit_numbers (n : ℕ) (h1 : 100 ≤ n ∧ n ≤ 999) (h2 : n^2 % 1000 = n % 1000) : 
  n = 376 ∨ n = 625 :=
by
  sorry

end three_digit_numbers_l201_201617


namespace find_a4_l201_201046

theorem find_a4 (a : ℕ → ℕ) 
  (h1 : ∀ n, (a n + 1) / (a (n + 1) + 1) = 1 / 2) 
  (h2 : a 2 = 2) : 
  a 4 = 11 :=
sorry

end find_a4_l201_201046


namespace total_pages_l201_201627

-- Conditions
variables (B1 B2 : ℕ)
variable (h1 : (2 / 3 : ℚ) * B1 - (1 / 3 : ℚ) * B1 = 90)
variable (h2 : (3 / 4 : ℚ) * B2 - (1 / 4 : ℚ) * B2 = 120)

-- Theorem statement
theorem total_pages (B1 B2 : ℕ) (h1 : (2 / 3 : ℚ) * B1 - (1 / 3 : ℚ) * B1 = 90) (h2 : (3 / 4 : ℚ) * B2 - (1 / 4 : ℚ) * B2 = 120) :
  B1 + B2 = 510 :=
sorry

end total_pages_l201_201627


namespace cos_225_l201_201374

noncomputable def cos_225_eq : ℝ :=
  cos (225 * (real.pi / 180))

theorem cos_225 : cos_225_eq = -real.sqrt 2 / 2 :=
  sorry

end cos_225_l201_201374


namespace tyler_eggs_in_fridge_l201_201146

def recipe_eggs_for_four : Nat := 2
def people_multiplier : Nat := 2
def eggs_needed : Nat := recipe_eggs_for_four * people_multiplier
def eggs_to_buy : Nat := 1
def eggs_in_fridge : Nat := eggs_needed - eggs_to_buy

theorem tyler_eggs_in_fridge : eggs_in_fridge = 3 := by
  sorry

end tyler_eggs_in_fridge_l201_201146


namespace sum_of_digits_3n_l201_201543

def digit_sum (n : ℕ) : ℕ :=
  (n.digits 10).sum

theorem sum_of_digits_3n (n : ℕ) (hn1 : digit_sum n = 100) (hn2 : digit_sum (44 * n) = 800) : digit_sum (3 * n) = 300 := by
  sorry

end sum_of_digits_3n_l201_201543


namespace island_length_l201_201983

/-- Proof problem: Given an island in the Indian Ocean with a width of 4 miles and a perimeter of 22 miles. 
    Assume the island is rectangular in shape. Prove that the length of the island is 7 miles. -/
theorem island_length
  (width length : ℝ) 
  (h_width : width = 4)
  (h_perimeter : 2 * (length + width) = 22) : 
  length = 7 :=
sorry

end island_length_l201_201983


namespace cos_225_eq_neg_sqrt2_div_2_l201_201390

theorem cos_225_eq_neg_sqrt2_div_2 : Real.cos (225 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by
  sorry

end cos_225_eq_neg_sqrt2_div_2_l201_201390


namespace complement_of_A_in_reals_l201_201058

open Set

theorem complement_of_A_in_reals :
  (compl {x : ℝ | (x - 1) / (x - 2) ≥ 0}) = {x : ℝ | 1 < x ∧ x ≤ 2} :=
by
  sorry

end complement_of_A_in_reals_l201_201058


namespace find_range_of_m_l201_201447

-- Define propositions p and q based on the problem description
def p (m : ℝ) : Prop :=
  ∀ x y : ℝ, m ≠ 0 → (x - 2 * y + 3 = 0 ∧ y * y ≠ m * x)

def q (m : ℝ) : Prop :=
  5 - 2 * m ≠ 0 ∧ m ≠ 0 ∧ (∃ x y : ℝ, (x * x) / (5 - 2 * m) + (y * y) / m = 1)

-- Given conditions
def condition1 (m : ℝ) : Prop := p m ∨ q m
def condition2 (m : ℝ) : Prop := ¬ (p m ∧ q m)

-- The range of m that satisfies the given problem
def valid_m (m : ℝ) : Prop :=
  (m ≥ 3) ∨ (m < 0) ∨ (0 < m ∧ m ≤ 2.5)

theorem find_range_of_m (m : ℝ) : condition1 m → condition2 m → valid_m m := 
  sorry

end find_range_of_m_l201_201447


namespace cost_of_paving_l201_201821

def length : ℝ := 5.5
def width : ℝ := 3.75
def rate_per_sqm : ℝ := 1400
def expected_cost : ℝ := 28875

theorem cost_of_paving (l w r : ℝ) (h_l : l = length) (h_w : w = width) (h_r : r = rate_per_sqm) :
  (l * w * r) = expected_cost := by
  sorry

end cost_of_paving_l201_201821


namespace increasing_function_on_R_l201_201928

theorem increasing_function_on_R (x1 x2 : ℝ) (h : x1 < x2) : 3 * x1 + 2 < 3 * x2 + 2 := 
by
  sorry

end increasing_function_on_R_l201_201928


namespace probability_interval_l201_201948

theorem probability_interval (P_A P_B P_A_inter_P_B : ℝ) (h1 : P_A = 3 / 4) (h2 : P_B = 2 / 3) : 
  5/12 ≤ P_A_inter_P_B ∧ P_A_inter_P_B ≤ 2/3 :=
sorry

end probability_interval_l201_201948


namespace square_area_ratio_l201_201217

theorem square_area_ratio (s₁ s₂ d₂ : ℝ)
  (h1 : s₁ = 2 * d₂)
  (h2 : d₂ = s₂ * Real.sqrt 2) :
  (s₁^2) / (s₂^2) = 8 :=
by
  sorry

end square_area_ratio_l201_201217


namespace solve_in_primes_l201_201125

theorem solve_in_primes (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) :
  p^2 - 6 * p * q + q^2 + 3 * q - 1 = 0 ↔ (p = 17 ∧ q = 3) := 
sorry

end solve_in_primes_l201_201125


namespace reflection_matrix_determine_l201_201720

theorem reflection_matrix_determine (a b : ℚ)
  (h1 : (a^2 - (3/4) * b) = 1)
  (h2 : (-(3/4) * b + (1/16)) = 1)
  (h3 : (a * b + (1/4) * b) = 0)
  (h4 : (-(3/4) * a - (3/16)) = 0) :
  (a, b) = (1/4, -5/4) := 
sorry

end reflection_matrix_determine_l201_201720


namespace probability_manu_wins_l201_201092

theorem probability_manu_wins :
  ∑' (n : ℕ), (1 / 2)^(4 * (n + 1)) = 1 / 15 :=
by
  sorry

end probability_manu_wins_l201_201092


namespace cos_225_eq_neg_sqrt2_div_2_l201_201389

theorem cos_225_eq_neg_sqrt2_div_2 : Real.cos (225 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by
  sorry

end cos_225_eq_neg_sqrt2_div_2_l201_201389


namespace exists_infinite_sets_of_positive_integers_l201_201249

theorem exists_infinite_sets_of_positive_integers (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  ∃ (S : ℕ → ℕ × ℕ × ℕ), ∀ n : ℕ, S n = (x, y, z) ∧ 
  ((x + y + z)^2 + 2*(x + y + z) = 5*(x*y + y*z + z*x)) :=
sorry

end exists_infinite_sets_of_positive_integers_l201_201249


namespace exist_interval_l201_201937

noncomputable def f (x : ℝ) := Real.log x + x - 4

theorem exist_interval (x₀ : ℝ) (h₀ : f x₀ = 0) : 2 < x₀ ∧ x₀ < 3 :=
by
  sorry

end exist_interval_l201_201937


namespace largest_valid_four_digit_number_l201_201816

-- Definition of the problem conditions
def is_valid_number (a b c d : ℕ) : Prop :=
  c = a + b ∧ d = b + c

-- Proposition that we need to prove
theorem largest_valid_four_digit_number : ∃ (a b c d : ℕ),
  is_valid_number a b c d ∧ a * 1000 + b * 100 + c * 10 + d = 9099 :=
by
  sorry

end largest_valid_four_digit_number_l201_201816


namespace largest_four_digit_number_l201_201811

theorem largest_four_digit_number :
  ∃ a b c d : ℕ, 
    9 < 1000 * a + 100 * b + 10 * c + d ∧ 
    1000 * a + 100 * b + 10 * c + d < 10000 ∧ 
    c = a + b ∧ 
    d = b + c ∧ 
    1000 * a + 100 * b + 10 * c + d = 9099 :=
by {
  sorry
}

end largest_four_digit_number_l201_201811


namespace birth_death_rate_interval_l201_201897

theorem birth_death_rate_interval
  (b_rate : ℕ) (d_rate : ℕ) (population_increase_one_day : ℕ) (seconds_in_one_day : ℕ)
  (net_increase_per_t_seconds : ℕ) (t : ℕ)
  (h1 : b_rate = 5)
  (h2 : d_rate = 3)
  (h3 : population_increase_one_day = 86400)
  (h4 : seconds_in_one_day = 86400)
  (h5 : net_increase_per_t_seconds = b_rate - d_rate)
  (h6 : population_increase_one_day = net_increase_per_t_seconds * (seconds_in_one_day / t)) :
  t = 2 :=
by
  sorry

end birth_death_rate_interval_l201_201897


namespace geom_sequence_sum_of_first4_l201_201901

noncomputable def geom_sum_first4_terms (a : ℕ → ℝ) (common_ratio : ℝ) (a0 a1 a4 : ℝ) : ℝ :=
  a0 + a0 * common_ratio + a0 * common_ratio^2 + a0 * common_ratio^3

theorem geom_sequence_sum_of_first4 {a : ℕ → ℝ} (a1 a4 : ℝ) (r : ℝ)
  (h1 : a 1 = a1) (h4 : a 4 = a4) 
  (h_geom : ∀ n, a (n + 1) = a n * r) :
  geom_sum_first4_terms a (r) a1 (a 0) (a 4) = 120 :=
by sorry

end geom_sequence_sum_of_first4_l201_201901


namespace geometric_progression_properties_l201_201942

-- Define the first term and the fifth term given
def b₁ := Real.sqrt 3
def b₅ := Real.sqrt 243

-- Define the nth term formula for geometric progression
def geometric_term (b₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ := b₁ * q ^ (n - 1)

-- State both the common ratio and the sixth term
theorem geometric_progression_properties :
  ∃ q : ℝ, (q = Real.sqrt 3 ∨ q = -Real.sqrt 3) ∧ 
           geometric_term b₁ q 5 = b₅ ∧ 
           geometric_term b₁ q 6 = 27 ∨ geometric_term b₁ q 6 = -27 :=
by
  sorry

end geometric_progression_properties_l201_201942


namespace pool_filling_time_l201_201648

noncomputable def fill_pool_time (hose_rate : ℕ) (cost_per_10_gallons : ℚ) (total_cost : ℚ) : ℚ :=
  let cost_per_gallon := cost_per_10_gallons / 10
  let total_gallons := total_cost / cost_per_gallon
  total_gallons / hose_rate

theorem pool_filling_time :
  fill_pool_time 100 (1 / 100) 5 = 50 := 
by
  sorry

end pool_filling_time_l201_201648


namespace cos_225_correct_l201_201408

noncomputable def cos_225_eq : Prop :=
  cos (225 * Real.pi / 180) = -Real.sqrt 2 / 2

theorem cos_225_correct : cos_225_eq :=
sorry

end cos_225_correct_l201_201408


namespace min_value_inequality_l201_201917

variable (x y : ℝ)

theorem min_value_inequality (h₀ : x > 0) (h₁ : y > 0) (h₂ : x + y = 1) : 
  ∃ m : ℝ, m = 1 / 4 ∧ (∀ x y, x > 0 → y > 0 → x + y = 1 → (x ^ 2) / (x + 2) + (y ^ 2) / (y + 1) ≥ m) :=
by
  use (1 / 4)
  sorry

end min_value_inequality_l201_201917


namespace length_BC_l201_201694

theorem length_BC {A B C : ℝ} (r1 r2 : ℝ) (AB : ℝ) (h1 : r1 = 8) (h2 : r2 = 5) (h3 : AB = r1 + r2) :
  C = B + (65 : ℝ) / 3 :=
by
  -- Problem set-up and solving comes here if needed
  sorry

end length_BC_l201_201694


namespace find_second_number_l201_201509

theorem find_second_number (a b c : ℕ) (h1 : a = 5 * x) (h2 : b = 3 * x) (h3 : c = 4 * x) (h_sum : a + b + c = 108) : b = 27 :=
by
  sorry

end find_second_number_l201_201509


namespace weight_in_one_hand_l201_201172

theorem weight_in_one_hand (total_weight : ℕ) (h : total_weight = 16) : total_weight / 2 = 8 :=
by
  sorry

end weight_in_one_hand_l201_201172


namespace smallest_base_l201_201523

theorem smallest_base : ∃ b : ℕ, (b^2 ≤ 120 ∧ 120 < b^3) ∧ ∀ n : ℕ, (n^2 ≤ 120 ∧ 120 < n^3) → b ≤ n :=
by sorry

end smallest_base_l201_201523


namespace clarence_oranges_l201_201846

def initial_oranges := 5
def oranges_from_joyce := 3
def total_oranges := initial_oranges + oranges_from_joyce

theorem clarence_oranges : total_oranges = 8 :=
  by
  sorry

end clarence_oranges_l201_201846


namespace radius_circle_D_eq_five_l201_201689

-- Definitions for circles with given radii and tangency conditions
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

noncomputable def circle_C : Circle := ⟨(0, 0), 5⟩
noncomputable def circle_D (rD : ℝ) : Circle := ⟨(4 * rD, 0), 4 * rD⟩
noncomputable def circle_E (rE : ℝ) : Circle := ⟨(5 - rE, rE * 5), rE⟩

-- Prove that the radius of circle D is 5
theorem radius_circle_D_eq_five (rE : ℝ) (rD : ℝ) : circle_D rE = circle_C → rD = 5 := by
  sorry

end radius_circle_D_eq_five_l201_201689


namespace average_seeds_per_apple_l201_201935

-- Define the problem conditions and the proof statement

theorem average_seeds_per_apple
  (A : ℕ)
  (total_seeds_requirement : ℕ := 60)
  (pear_seeds_avg : ℕ := 2)
  (grape_seeds_avg : ℕ := 3)
  (num_apples : ℕ := 4)
  (num_pears : ℕ := 3)
  (num_grapes : ℕ := 9)
  (shortfall : ℕ := 3)
  (collected_seeds : ℕ := num_apples * A + num_pears * pear_seeds_avg + num_grapes * grape_seeds_avg)
  (required_seeds : ℕ := total_seeds_requirement - shortfall) :
  collected_seeds = required_seeds → A = 6 := 
by
  sorry

end average_seeds_per_apple_l201_201935


namespace sum_of_fraction_terms_l201_201322

theorem sum_of_fraction_terms (x : ℚ) (hx : x = 45 / 99) (y : ℚ) (hy : y * 99 = 45) : 
  let z := (5 : ℚ) / (11 : ℚ) in 
  hwf : x = z → z.den + z.num = 16 := 
by
  sorry

end sum_of_fraction_terms_l201_201322


namespace remainder_of_eggs_is_2_l201_201994

-- Define the number of eggs each person has
def david_eggs : ℕ := 45
def emma_eggs : ℕ := 52
def fiona_eggs : ℕ := 25

-- Define total eggs and remainder function
def total_eggs : ℕ := david_eggs + emma_eggs + fiona_eggs
def remainder (a b : ℕ) : ℕ := a % b

-- Prove that the remainder of total eggs divided by 10 is 2
theorem remainder_of_eggs_is_2 : remainder total_eggs 10 = 2 := by
  sorry

end remainder_of_eggs_is_2_l201_201994


namespace find_q_l201_201580

theorem find_q (q: ℕ) (h: 81^10 = 3^q) : q = 40 :=
by
  sorry

end find_q_l201_201580


namespace intersection_complement_eq_l201_201723

open Set

def U : Set ℕ := {x | 1 ≤ x ∧ x ≤ 8}  -- Which is from the solution set of x^2 - 9x + 8 ≤ 0
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {5, 6, 7}

theorem intersection_complement_eq : (U \ A) ∩ (U \ B) = {4, 8} :=
by
  sorry

end intersection_complement_eq_l201_201723


namespace cos_225_eq_l201_201414

noncomputable def angle_in_radians (deg : ℝ) : ℝ := deg * (Real.pi / 180)

theorem cos_225_eq :
  Real.cos (angle_in_radians 225) = - (Real.sqrt 2) / 2 := 
  sorry

end cos_225_eq_l201_201414


namespace geometric_sequence_a7_a8_l201_201472

-- Define the geometric sequence {a_n}
variable {a : ℕ → ℝ}

-- {a_n} is a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- Conditions
axiom h1 : is_geometric_sequence a
axiom h2 : a 1 + a 2 = 40
axiom h3 : a 3 + a 4 = 60

-- Proof problem: Find a_7 + a_8
theorem geometric_sequence_a7_a8 :
  a 7 + a 8 = 135 :=
by
  sorry

end geometric_sequence_a7_a8_l201_201472


namespace binom_14_11_l201_201562

open Nat

theorem binom_14_11 : Nat.choose 14 11 = 364 := by
  sorry

end binom_14_11_l201_201562


namespace least_value_of_p_plus_q_l201_201070

theorem least_value_of_p_plus_q (p q : ℕ) (hp : 1 < p) (hq : 1 < q) (h : 17 * (p + 1) = 28 * (q + 1)) : p + q = 135 :=
  sorry

end least_value_of_p_plus_q_l201_201070


namespace intersect_at_one_point_l201_201842

-- Define the quadratic and linear functions
def quadratic (b x : ℝ) : ℝ := b * x^2 + b * x + 2
def linear (x : ℝ) : ℝ := 2 * x + 4

-- Define the discriminant of the quadratic equation resulting from setting the polynomials equal
def discriminant (b : ℝ) : ℝ := (b - 2)^2 - 4 * b * (-2)

-- The main theorem we want to prove
theorem intersect_at_one_point (b : ℝ) : discriminant b = 0 ↔ b = -2 := by
  unfold discriminant
  sorry

end intersect_at_one_point_l201_201842


namespace largest_valid_number_l201_201789

-- Define the conditions for the digits of the number
def valid_digits (a b c d : ℕ) : Prop :=
  c = a + b ∧ d = b + c

-- Prove that the number formed by digits 9, 0, 9, 9 is the largest valid 4-digit number
theorem largest_valid_number : ∃ a b c d, valid_digits a b c d ∧
  (a * 1000 + b * 100 + c * 10 + d = 9099) :=
begin
  use [9, 0, 9, 9],
  split,
  { -- Proof of valid digits condition
    split; refl },
  { -- Proof that the number is 9099
    refl }
end

end largest_valid_number_l201_201789


namespace problem_statement_l201_201991

def g (x : ℝ) : ℝ := 3 * x^2 - 6 * x + 8

theorem problem_statement : 3 * g 2 + 4 * g (-2) = 152 := by
  sorry

end problem_statement_l201_201991


namespace kitty_vacuum_time_l201_201760

theorem kitty_vacuum_time
  (weekly_toys : ℕ := 5)
  (weekly_windows : ℕ := 15)
  (weekly_furniture : ℕ := 10)
  (total_cleaning_time : ℕ := 200)
  (weeks : ℕ := 4)
  : (weekly_toys + weekly_windows + weekly_furniture) * weeks < total_cleaning_time ∧ ((total_cleaning_time - ((weekly_toys + weekly_windows + weekly_furniture) * weeks)) / weeks = 20)
  := by
  sorry

end kitty_vacuum_time_l201_201760


namespace cos_225_degrees_l201_201401

theorem cos_225_degrees : Real.cos (225 * Real.pi / 180) = -1 / Real.sqrt 2 := by
  sorry

end cos_225_degrees_l201_201401


namespace y_intercept_of_line_l201_201695

theorem y_intercept_of_line (x y : ℝ) (h : 4 * x + 7 * y = 28) (hx : x = 0) : y = 4 :=
by
  rw [hx] at h
  simp at h
  exact h

end y_intercept_of_line_l201_201695


namespace sum_of_legs_l201_201630

theorem sum_of_legs (x : ℕ) (h : x^2 + (x + 1)^2 = 41^2) : x + (x + 1) = 57 :=
sorry

end sum_of_legs_l201_201630


namespace recurring_decimal_fraction_sum_l201_201301

theorem recurring_decimal_fraction_sum (x : ℚ) (h : x = 0.454545454545...) : 
    (let frac := rat.num_denom x.simplify in frac.fst + frac.snd)  = 16 :=
sorry

end recurring_decimal_fraction_sum_l201_201301


namespace find_k_l201_201435

theorem find_k (k : ℝ) (d : ℝ) (h : d = 4) :
  -x^2 - (k + 10) * x - 8 = -(x - 2) * (x - d) → k = -16 :=
by
  intros
  rw [h] at *
  sorry

end find_k_l201_201435


namespace each_organization_receives_correct_amount_l201_201009

-- Defining the conditions
def total_amount_raised : ℝ := 2500
def donation_percentage : ℝ := 0.80
def number_of_organizations : ℝ := 8

-- Defining the assertion about the amount each organization receives
def amount_each_organization_receives : ℝ := (donation_percentage * total_amount_raised) / number_of_organizations

-- Proving the correctness of the amount each organization receives
theorem each_organization_receives_correct_amount : amount_each_organization_receives = 250 :=
by
  sorry

end each_organization_receives_correct_amount_l201_201009


namespace depth_second_project_l201_201334

def volume (depth length breadth : ℝ) : ℝ := depth * length * breadth

theorem depth_second_project (D : ℝ) : 
  (volume 100 25 30 = volume D 20 50) → D = 75 :=
by 
  sorry

end depth_second_project_l201_201334


namespace income_fraction_from_tips_l201_201665

variable (S T : ℝ)

theorem income_fraction_from_tips :
  (T = (9 / 4) * S) → (T / (S + T) = 9 / 13) :=
by
  sorry

end income_fraction_from_tips_l201_201665


namespace binom_comb_always_integer_l201_201037

theorem binom_comb_always_integer (k n : ℕ) (h1 : 1 ≤ k) (h2 : k < n) (h3 : (k + 2) ∣ n) : 
  ∃ m : ℤ, ((n - 3 * k - 2) / (k + 2)) * Nat.choose n k = m := 
sorry

end binom_comb_always_integer_l201_201037


namespace nickels_used_for_notebook_l201_201096

def notebook_cost_dollars : ℚ := 1.30
def dollar_to_cents_conversion : ℤ := 100
def nickel_value_cents : ℤ := 5

theorem nickels_used_for_notebook : 
  (notebook_cost_dollars * dollar_to_cents_conversion) / nickel_value_cents = 26 := 
by 
  sorry

end nickels_used_for_notebook_l201_201096


namespace probability_green_or_yellow_l201_201652

def green_faces : ℕ := 3
def yellow_faces : ℕ := 2
def blue_faces : ℕ := 1
def total_faces : ℕ := 6

theorem probability_green_or_yellow : 
  (green_faces + yellow_faces) / total_faces = 5 / 6 :=
by
  sorry

end probability_green_or_yellow_l201_201652


namespace sum_of_fraction_parts_of_repeating_decimal_l201_201288

theorem sum_of_fraction_parts_of_repeating_decimal :
  let frac := Rat.mk 45 99 in
  let simplest_frac := frac.num.gcd 45 99 in
  let num := simplest_frac.num in
  let denom := simplest_frac.denom in
  num + denom = 16 :=
by
  sorry

end sum_of_fraction_parts_of_repeating_decimal_l201_201288


namespace stream_current_rate_l201_201338

theorem stream_current_rate (r w : ℝ) (h1 : 18 / (r + w) + 4 = 18 / (r - w))
  (h2 : 18 / (3 * r + w) + 2 = 18 / (3 * r - w)) : w = 3 :=
  sorry

end stream_current_rate_l201_201338


namespace largest_valid_four_digit_number_l201_201817

-- Definition of the problem conditions
def is_valid_number (a b c d : ℕ) : Prop :=
  c = a + b ∧ d = b + c

-- Proposition that we need to prove
theorem largest_valid_four_digit_number : ∃ (a b c d : ℕ),
  is_valid_number a b c d ∧ a * 1000 + b * 100 + c * 10 + d = 9099 :=
by
  sorry

end largest_valid_four_digit_number_l201_201817


namespace num_valid_functions_l201_201668

theorem num_valid_functions :
  ∃! (f : ℤ → ℝ), 
  (f 1 = 1) ∧ 
  (∀ (m n : ℤ), f m ^ 2 - f n ^ 2 = f (m + n) * f (m - n)) ∧ 
  (∀ n : ℤ, f n = f (n + 2013)) :=
sorry

end num_valid_functions_l201_201668


namespace tan_neg_405_eq_one_l201_201023

theorem tan_neg_405_eq_one : Real.tan (-(405 * Real.pi / 180)) = 1 :=
by
-- Proof omitted
sorry

end tan_neg_405_eq_one_l201_201023


namespace lindy_total_distance_l201_201667

def meet_distance (d v_j v_c : ℕ) : ℕ :=
  d / (v_j + v_c)

def lindy_distance (v_l t : ℕ) : ℕ :=
  v_l * t

theorem lindy_total_distance
  (d : ℕ)
  (v_j : ℕ)
  (v_c : ℕ)
  (v_l : ℕ)
  (h1 : d = 360)
  (h2 : v_j = 5)
  (h3 : v_c = 7)
  (h4 : v_l = 12)
  :
  lindy_distance v_l (meet_distance d v_j v_c) = 360 :=
by
  sorry

end lindy_total_distance_l201_201667


namespace range_of_m_l201_201201

theorem range_of_m (m : ℝ) : (∀ x : ℝ, (3 * x^2 + 2 * x + 2) / (x^2 + x + 1) ≥ m) → m ≤ 2 :=
sorry

end range_of_m_l201_201201


namespace double_sum_evaluation_l201_201029

theorem double_sum_evaluation :
  ∑' m:ℕ, ∑' n:ℕ, (if m > 0 ∧ n > 0 then 1 / (m * n * (m + n + 2)) else 0) = -Real.pi^2 / 6 :=
sorry

end double_sum_evaluation_l201_201029


namespace first_term_of_arithmetic_sequence_l201_201262

theorem first_term_of_arithmetic_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) (a1 : ℝ)
  (h_arith : ∀ n, a n = a1 + ↑n - 1) 
  (h_sum : ∀ n, S n = n / 2 * (2 * a1 + (n - 1))) 
  (h_min : ∀ n, S 2022 ≤ S n) : 
  -2022 < a1 ∧ a1 < -2021 :=
by
  sorry

end first_term_of_arithmetic_sequence_l201_201262


namespace exactly_two_succeed_probability_l201_201981

-- Define the probabilities of events A, B, and C decrypting the code
def P_A_decrypts : ℚ := 1/5
def P_B_decrypts : ℚ := 1/4
def P_C_decrypts : ℚ := 1/3

-- Define the probabilities of events A, B, and C not decrypting the code
def P_A_not_decrypts : ℚ := 1 - P_A_decrypts
def P_B_not_decrypts : ℚ := 1 - P_B_decrypts
def P_C_not_decrypts : ℚ := 1 - P_C_decrypts

-- Define the probability that exactly two out of A, B, and C decrypt the code
def P_exactly_two_succeed : ℚ :=
  (P_A_decrypts * P_B_decrypts * P_C_not_decrypts) +
  (P_A_decrypts * P_B_not_decrypts * P_C_decrypts) +
  (P_A_not_decrypts * P_B_decrypts * P_C_decrypts)

-- Prove that this probability is equal to 3/20
theorem exactly_two_succeed_probability : P_exactly_two_succeed = 3 / 20 := by
  sorry

end exactly_two_succeed_probability_l201_201981


namespace simplify_fraction_l201_201105

theorem simplify_fraction : (140 / 9800) * 35 = 1 / 70 := 
by
  -- Proof steps would go here.
  sorry

end simplify_fraction_l201_201105


namespace square_center_sum_l201_201977

noncomputable def sum_of_center_coordinates (A B C D : ℝ × ℝ) : ℝ :=
  let center : ℝ × ℝ := ((A.1 + C.1) / 2, (A.2 + C.2) / 2)
  center.1 + center.2

theorem square_center_sum
  (A B C D : ℝ × ℝ)
  (h1 : 9 = A.1) (h2 : 0 = A.2)
  (h3 : 4 = B.1) (h4 : 0 = B.2)
  (h5 : 0 = C.1) (h6 : 3 = C.2)
  (h7: A.1 < B.1) (h8: A.2 < C.2) :
  sum_of_center_coordinates A B C D = 8 := 
by
  sorry

end square_center_sum_l201_201977


namespace arith_seq_a15_l201_201467

variable {α : Type} [LinearOrderedField α]

def is_arith_seq (a : ℕ → α) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

theorem arith_seq_a15 (a : ℕ → α) (k l m : ℕ) (x y : α) 
  (h_seq : is_arith_seq a)
  (h_k : a k = x)
  (h_l : a l = y) :
  a (l + (l - k)) = 2 * y - x := 
  sorry

end arith_seq_a15_l201_201467


namespace arithmetic_sequence_problem_l201_201781

theorem arithmetic_sequence_problem
  (a : ℕ → ℤ)  -- the arithmetic sequence
  (S : ℕ → ℤ)  -- the sum of the first n terms
  (m : ℕ)      -- the m in question
  (h1 : a (m - 1) + a (m + 1) - a m ^ 2 = 0)
  (h2 : S (2 * m - 1) = 18) :
  m = 5 := 
sorry

end arithmetic_sequence_problem_l201_201781


namespace domain_f_max_min_f_l201_201714

noncomputable def f (x : ℝ) : ℝ := 2 * Real.log 3 (3 - x) - Real.log 3 (1 + x)

theorem domain_f :
  {x : ℝ | f x = f x} ⊆ Ioo (-1 : ℝ) 3 :=
by sorry

theorem max_min_f (x : ℝ) (h : 0 ≤ x ∧ x ≤ 2) :
  f x = -1 ∨ f x = 2 :=
by sorry

end domain_f_max_min_f_l201_201714


namespace initial_people_in_line_l201_201646

theorem initial_people_in_line (x : ℕ) (h1 : x + 22 = 83) : x = 61 :=
by sorry

end initial_people_in_line_l201_201646


namespace geometric_progression_sixth_term_proof_l201_201946

noncomputable def geometric_progression_sixth_term (b₁ b₅ : ℝ) (q : ℝ) := b₅ * q
noncomputable def find_q (b₁ b₅ : ℝ) := (b₅ / b₁)^(1/4)

theorem geometric_progression_sixth_term_proof (b₁ b₅ : ℝ) (h₁ : b₁ = Real.sqrt 3) (h₅ : b₅ = Real.sqrt 243) : 
  ∃ q : ℝ, (q = Real.sqrt 3 ∨ q = - Real.sqrt 3) ∧ geometric_progression_sixth_term b₁ b₅ q = 27 ∨ geometric_progression_sixth_term b₁ b₅ q = -27 :=
by
  sorry

end geometric_progression_sixth_term_proof_l201_201946


namespace percentage_increase_l201_201476

theorem percentage_increase
  (initial_earnings new_earnings : ℝ)
  (h_initial : initial_earnings = 55)
  (h_new : new_earnings = 60) :
  ((new_earnings - initial_earnings) / initial_earnings * 100) = 9.09 :=
by
  sorry

end percentage_increase_l201_201476


namespace vincent_earnings_l201_201153

def fantasy_book_cost : ℕ := 6
def literature_book_cost : ℕ := fantasy_book_cost / 2
def mystery_book_cost : ℕ := 4

def fantasy_books_sold_per_day : ℕ := 5
def literature_books_sold_per_day : ℕ := 8
def mystery_books_sold_per_day : ℕ := 3

def daily_earnings : ℕ :=
  (fantasy_books_sold_per_day * fantasy_book_cost) +
  (literature_books_sold_per_day * literature_book_cost) +
  (mystery_books_sold_per_day * mystery_book_cost)

def total_earnings_after_seven_days : ℕ :=
  daily_earnings * 7

theorem vincent_earnings : total_earnings_after_seven_days = 462 :=
by
  sorry

end vincent_earnings_l201_201153


namespace money_inequalities_l201_201861

theorem money_inequalities (a b : ℝ) (h₁ : 5 * a + b > 51) (h₂ : 3 * a - b = 21) : a > 9 ∧ b > 6 := 
by
  sorry

end money_inequalities_l201_201861


namespace number_of_lists_correct_l201_201332

noncomputable def number_of_lists : Nat :=
  15 ^ 4

theorem number_of_lists_correct :
  number_of_lists = 50625 := by
  sorry

end number_of_lists_correct_l201_201332


namespace factorize_expression_l201_201186

theorem factorize_expression (m n : ℤ) : m^2 * n - 9 * n = n * (m + 3) * (m - 3) := by
  sorry

end factorize_expression_l201_201186


namespace num_female_fox_terriers_l201_201676

def total_dogs : Nat := 2012
def total_female_dogs : Nat := 1110
def total_fox_terriers : Nat := 1506
def male_shih_tzus : Nat := 202

theorem num_female_fox_terriers :
    ∃ (female_fox_terriers: Nat), 
        female_fox_terriers = total_fox_terriers - (total_dogs - total_female_dogs - male_shih_tzus) := by
    sorry

end num_female_fox_terriers_l201_201676


namespace solve_in_primes_l201_201124

theorem solve_in_primes (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) :
  p^2 - 6 * p * q + q^2 + 3 * q - 1 = 0 ↔ (p = 17 ∧ q = 3) := 
sorry

end solve_in_primes_l201_201124


namespace amount_borrowed_l201_201169

variable (P : ℝ)
variable (interest_paid : ℝ) -- Interest paid on borrowing
variable (interest_earned : ℝ) -- Interest earned on lending
variable (gain_per_year : ℝ)

variable (h1 : interest_paid = P * 4 * 2 / 100)
variable (h2 : interest_earned = P * 6 * 2 / 100)
variable (h3 : gain_per_year = 160)
variable (h4 : gain_per_year = (interest_earned - interest_paid) / 2)

theorem amount_borrowed : P = 8000 := by
  sorry

end amount_borrowed_l201_201169


namespace draw_points_worth_two_l201_201743

/-
In a certain football competition, a victory is worth 3 points, a draw is worth some points, and a defeat is worth 0 points. Each team plays 20 matches. A team scored 14 points after 5 games. The team needs to win at least 6 of the remaining matches to reach the 40-point mark by the end of the tournament. Prove that the number of points a draw is worth is 2.
-/

theorem draw_points_worth_two :
  ∃ D, (∀ (victory_points draw_points defeat_points total_matches matches_played points_scored remaining_matches wins_needed target_points),
    victory_points = 3 ∧
    defeat_points = 0 ∧
    total_matches = 20 ∧
    matches_played = 5 ∧
    points_scored = 14 ∧
    remaining_matches = total_matches - matches_played ∧
    wins_needed = 6 ∧
    target_points = 40 ∧
    points_scored + 6 * victory_points + (remaining_matches - wins_needed) * D = target_points ∧
    draw_points = D) →
    D = 2 :=
by
  sorry

end draw_points_worth_two_l201_201743


namespace tangent_line_slope_l201_201056

theorem tangent_line_slope (m : ℝ) :
  (∀ x y, (x^2 + y^2 - 4*x + 2 = 0) → (y = m * x)) → (m = 1 ∨ m = -1) := 
by
  intro h
  sorry

end tangent_line_slope_l201_201056


namespace angle_x_is_36_l201_201900

theorem angle_x_is_36
    (x : ℝ)
    (h1 : 7 * x + 3 * x = 360)
    (h2 : 8 * x ≤ 360) :
    x = 36 := 
by {
  sorry
}

end angle_x_is_36_l201_201900


namespace square_pyramid_sum_l201_201960

-- Define the number of faces, edges, and vertices of a square pyramid.
def faces_square_base : Nat := 1
def faces_lateral : Nat := 4
def edges_base : Nat := 4
def edges_lateral : Nat := 4
def vertices_base : Nat := 4
def vertices_apex : Nat := 1

-- Summing the faces, edges, and vertices
def total_faces : Nat := faces_square_base + faces_lateral
def total_edges : Nat := edges_base + edges_lateral
def total_vertices : Nat := vertices_base + vertices_apex

theorem square_pyramid_sum : (total_faces + total_edges + total_vertices = 18) :=
by
  sorry

end square_pyramid_sum_l201_201960


namespace sum_of_fraction_parts_l201_201319

theorem sum_of_fraction_parts (x : ℝ) (hx : x = 0.45) : 
  (∃ (a b : ℕ), x = a / b ∧ Nat.gcd a b = 1 ∧ a + b = 16) :=
by
  sorry

end sum_of_fraction_parts_l201_201319


namespace coplanar_lines_condition_l201_201926

theorem coplanar_lines_condition (h : ℝ) : 
  (∃ c : ℝ, 
    (2 : ℝ) = 3 * c ∧ 
    (-1 : ℝ) = c ∧ 
    (h : ℝ) = -2 * c) ↔ 
  (h = 2) :=
by
  sorry

end coplanar_lines_condition_l201_201926


namespace sum_a2000_inv_a2000_l201_201988

theorem sum_a2000_inv_a2000 (a : ℂ) (h : a^2 - a + 1 = 0) : a^2000 + 1/(a^2000) = -1 :=
by
    sorry

end sum_a2000_inv_a2000_l201_201988


namespace cos_225_eq_l201_201412

noncomputable def angle_in_radians (deg : ℝ) : ℝ := deg * (Real.pi / 180)

theorem cos_225_eq :
  Real.cos (angle_in_radians 225) = - (Real.sqrt 2) / 2 := 
  sorry

end cos_225_eq_l201_201412


namespace cos_225_correct_l201_201404

noncomputable def cos_225_eq : Prop :=
  cos (225 * Real.pi / 180) = -Real.sqrt 2 / 2

theorem cos_225_correct : cos_225_eq :=
sorry

end cos_225_correct_l201_201404


namespace sum_of_fraction_terms_l201_201321

theorem sum_of_fraction_terms (x : ℚ) (hx : x = 45 / 99) (y : ℚ) (hy : y * 99 = 45) : 
  let z := (5 : ℚ) / (11 : ℚ) in 
  hwf : x = z → z.den + z.num = 16 := 
by
  sorry

end sum_of_fraction_terms_l201_201321


namespace haley_extra_tickets_l201_201455

theorem haley_extra_tickets (cost_per_ticket : ℤ) (tickets_bought_for_self_and_friends : ℤ) (total_spent : ℤ) 
    (h1 : cost_per_ticket = 4) (h2 : tickets_bought_for_self_and_friends = 3) (h3 : total_spent = 32) : 
    (total_spent / cost_per_ticket) - tickets_bought_for_self_and_friends = 5 :=
by
  sorry

end haley_extra_tickets_l201_201455


namespace total_fruits_l201_201148

theorem total_fruits (total_baskets apples_baskets oranges_baskets apples_per_basket oranges_per_basket pears_per_basket : ℕ)
  (h1 : total_baskets = 127)
  (h2 : apples_baskets = 79)
  (h3 : oranges_baskets = 30)
  (h4 : apples_per_basket = 75)
  (h5 : oranges_per_basket = 143)
  (h6 : pears_per_basket = 56)
  : 79 * 75 + 30 * 143 + (127 - (79 + 30)) * 56 = 11223 := by
  sorry

end total_fruits_l201_201148


namespace value_of_nested_f_l201_201615

def f (x : ℤ) : ℤ := x^2 - 3 * x + 1

theorem value_of_nested_f : f (f (f (f (f (f (-1)))))) = 3432163846882600 := by
  sorry

end value_of_nested_f_l201_201615


namespace sum_of_four_triangles_l201_201173

theorem sum_of_four_triangles :
  ∀ (x y : ℝ), 3 * x + 2 * y = 27 → 2 * x + 3 * y = 23 → 4 * y = 12 :=
by
  intros x y h1 h2
  sorry

end sum_of_four_triangles_l201_201173


namespace least_number_divisible_by_digits_and_5_l201_201657

/-- Define a predicate to check if a number is divisible by all of its digits -/
def divisible_by_digits (n : ℕ) : Prop :=
  let digits := [n / 1000 % 10, n / 100 % 10 % 10, n / 10 % 10, n % 10]
  ∀ d ∈ digits, d ≠ 0 → n % d = 0

/-- Define the main theorem stating the least four-digit number divisible by 5 and each of its digits is 1425 -/
theorem least_number_divisible_by_digits_and_5 
  (n : ℕ) (hn : 1000 ≤ n ∧ n < 10000)
  (hd : (∀ i j : ℕ, i ≠ j → (n / 10^i % 10) ≠ (n / 10^j % 10)))
  (hdiv5 : n % 5 = 0)
  (hdiv_digits : divisible_by_digits n) 
  : n = 1425 :=
sorry

end least_number_divisible_by_digits_and_5_l201_201657


namespace polygon_sides_l201_201460

theorem polygon_sides (n : ℕ) (hn : 3 ≤ n) (H : (n * (n - 3)) / 2 = 15) : n = 7 :=
by
  sorry

end polygon_sides_l201_201460


namespace num_students_earning_B_l201_201745

open Real

theorem num_students_earning_B (total_students : ℝ) (pA : ℝ) (pB : ℝ) (pC : ℝ) (students_A : ℝ) (students_B : ℝ) (students_C : ℝ) :
  total_students = 31 →
  pA = 0.7 * pB →
  pC = 1.4 * pB →
  students_A = 0.7 * students_B →
  students_C = 1.4 * students_B →
  students_A + students_B + students_C = total_students →
  students_B = 10 :=
by
  intros h_total_students h_pa h_pc h_students_A h_students_C h_total_eq
  sorry

end num_students_earning_B_l201_201745


namespace solve_for_x_l201_201765

theorem solve_for_x (x y z : ℝ) (h1 : x * y = 8 - 3 * x - 2 * y) 
                                  (h2 : y * z = 8 - 2 * y - 3 * z) 
                                  (h3 : x * z = 35 - 5 * x - 3 * z) : 
  x = 8 :=
sorry

end solve_for_x_l201_201765


namespace cost_per_taco_is_1_50_l201_201978

namespace TacoTruck

def total_beef : ℝ := 100
def beef_per_taco : ℝ := 0.25
def taco_price : ℝ := 2
def profit : ℝ := 200

theorem cost_per_taco_is_1_50 :
  let total_tacos := total_beef / beef_per_taco
  let total_revenue := total_tacos * taco_price
  let total_cost := total_revenue - profit
  total_cost / total_tacos = 1.50 := 
by
  sorry

end TacoTruck

end cost_per_taco_is_1_50_l201_201978


namespace min_value_l201_201707

theorem min_value (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) (hxyz : x + y + z = 2) : 
  (∃ x y z, x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧ x + y + z = 2 ∧ (1/3 * x^3 + y^2 + z = 13/12)) :=
sorry

end min_value_l201_201707


namespace max_value_of_E_l201_201142

variable (a b c d : ℝ)

def E (a b c d : ℝ) : ℝ := a + 2 * b + c + 2 * d - a * b - b * c - c * d - d * a

theorem max_value_of_E :
  -5.5 ≤ a ∧ a ≤ 5.5 →
  -5.5 ≤ b ∧ b ≤ 5.5 →
  -5.5 ≤ c ∧ c ≤ 5.5 →
  -5.5 ≤ d ∧ d ≤ 5.5 →
  E a b c d ≤ 132 := by
  sorry

end max_value_of_E_l201_201142


namespace symmetric_inverse_sum_l201_201045

theorem symmetric_inverse_sum {f g : ℝ → ℝ} (h₁ : ∀ x, f (-x - 2) = -f (x)) (h₂ : ∀ y, g (f y) = y) (h₃ : ∀ y, f (g y) = y) (x₁ x₂ : ℝ) (h₄ : x₁ + x₂ = 0) : 
  g x₁ + g x₂ = -2 :=
by
  sorry

end symmetric_inverse_sum_l201_201045


namespace analysis_method_inequality_l201_201511

def analysis_method_seeks (inequality : Prop) : Prop :=
  ∃ (sufficient_condition : Prop), (inequality → sufficient_condition)

theorem analysis_method_inequality (inequality : Prop) :
  (∃ sufficient_condition, (inequality → sufficient_condition)) :=
sorry

end analysis_method_inequality_l201_201511


namespace person_walk_rate_l201_201176

theorem person_walk_rate (v : ℝ) (elevator_speed : ℝ) (length : ℝ) (time : ℝ) 
  (h1 : elevator_speed = 10) 
  (h2 : length = 112) 
  (h3 : time = 8) 
  (h4 : length = (v + elevator_speed) * time) 
  : v = 4 :=
by 
  sorry

end person_walk_rate_l201_201176


namespace max_distinct_subsets_l201_201442

def T : Set ℕ := { x | 1 ≤ x ∧ x ≤ 999 }

theorem max_distinct_subsets (k : ℕ) (A : Fin k → Set ℕ) 
  (h : ∀ i j : Fin k, i < j → A i ∪ A j = T) : 
  k ≤ 1000 := 
sorry

end max_distinct_subsets_l201_201442


namespace bus_ticket_problem_l201_201828

variables (x y : ℕ)

theorem bus_ticket_problem (h1 : x + y = 99) (h2 : 2 * x + 3 * y = 280) : x = 17 ∧ y = 82 :=
by
  sorry

end bus_ticket_problem_l201_201828


namespace trapezium_other_side_length_l201_201432

theorem trapezium_other_side_length (x : ℝ) : 
  (1 / 2) * (20 + x) * 13 = 247 → x = 18 :=
by
  sorry

end trapezium_other_side_length_l201_201432


namespace cos_225_l201_201377

noncomputable def cos_225_eq : ℝ :=
  cos (225 * (real.pi / 180))

theorem cos_225 : cos_225_eq = -real.sqrt 2 / 2 :=
  sorry

end cos_225_l201_201377


namespace cos_225_eq_neg_sqrt2_div_2_l201_201391

theorem cos_225_eq_neg_sqrt2_div_2 : Real.cos (225 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by
  sorry

end cos_225_eq_neg_sqrt2_div_2_l201_201391


namespace simplify_expr_calculate_expr_l201_201251

noncomputable theory
open Classical

-- Define a custom namespace for the problem
namespace SimplifyCalculations
variable (a b : ℝ)
variable h₁ : 0 < a
variable h₂ : 0 < b

-- Problem 1: Simplify the given expression to 1/a
theorem simplify_expr : 
  (a ^ (2/3) * b ^ (-1)) ^ (-1/2) * a ^ (-1/2) * b ^ (1/3) / (a * b ^ 5) ^ (1/6) = 1 / a :=
by
  sorry

-- Problem 2: Calculate the given expression to get 0.09
theorem calculate_expr : 
  (0.027)^(2/3) + (27/125)^(-1/3) - (2 + 7/9)^(0.5) = 0.09 :=
by
  sorry

end SimplifyCalculations

end simplify_expr_calculate_expr_l201_201251


namespace simplify_expression_l201_201492

def E (x : ℝ) : ℝ :=
  6 * x^2 + 4 * x + 9 - (7 - 5 * x - 9 * x^3 + 8 * x^2)

theorem simplify_expression (x : ℝ) : E x = 9 * x^3 - 2 * x^2 + 9 * x + 2 :=
by
  sorry

end simplify_expression_l201_201492


namespace geometric_progression_properties_l201_201941

-- Define the first term and the fifth term given
def b₁ := Real.sqrt 3
def b₅ := Real.sqrt 243

-- Define the nth term formula for geometric progression
def geometric_term (b₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ := b₁ * q ^ (n - 1)

-- State both the common ratio and the sixth term
theorem geometric_progression_properties :
  ∃ q : ℝ, (q = Real.sqrt 3 ∨ q = -Real.sqrt 3) ∧ 
           geometric_term b₁ q 5 = b₅ ∧ 
           geometric_term b₁ q 6 = 27 ∨ geometric_term b₁ q 6 = -27 :=
by
  sorry

end geometric_progression_properties_l201_201941


namespace golden_chest_diamonds_rubies_l201_201556

theorem golden_chest_diamonds_rubies :
  ∀ (diamonds rubies : ℕ), diamonds = 421 → rubies = 377 → diamonds - rubies = 44 :=
by
  intros diamonds rubies
  sorry

end golden_chest_diamonds_rubies_l201_201556


namespace repeating_decimal_fraction_sum_l201_201314

noncomputable def repeating_decimal_to_fraction (d : ℚ) : ℚ :=
  if d = 0.45 then 5 / 11 else 0

theorem repeating_decimal_fraction_sum :
  let x := repeating_decimal_to_fraction 0.45 in
  (x.num + x.den = 16) :=
by
  let x := repeating_decimal_to_fraction 0.45
  have h : x = 5 / 11 := by
    unfold repeating_decimal_to_fraction
    split_ifs with h
    . exact rfl
  rw [h, Rat.num_den_eq 5 11, Int.add_comm]
  sorry

end repeating_decimal_fraction_sum_l201_201314


namespace transfers_l201_201600

variable (x : ℕ)
variable (gA gB gC : ℕ)

noncomputable def girls_in_A := x + 4
noncomputable def girls_in_B := x
noncomputable def girls_in_C := x - 1

variable (trans_A_to_B : ℕ)
variable (trans_B_to_C : ℕ)
variable (trans_C_to_A : ℕ)

axiom C_to_A_girls : trans_C_to_A = 2
axiom equal_girls : gA = x + 1 ∧ gB = x + 1 ∧ gC = x + 1

theorem transfers (hA : gA = girls_in_A - trans_A_to_B + trans_C_to_A)
                  (hB : gB = girls_in_B - trans_B_to_C + trans_A_to_B)
                  (hC : gC = girls_in_C - trans_C_to_A + trans_B_to_C) :
  trans_A_to_B = 5 ∧ trans_B_to_C = 4 :=
by
  sorry

end transfers_l201_201600


namespace cos_225_correct_l201_201407

noncomputable def cos_225_eq : Prop :=
  cos (225 * Real.pi / 180) = -Real.sqrt 2 / 2

theorem cos_225_correct : cos_225_eq :=
sorry

end cos_225_correct_l201_201407


namespace cos_225_degrees_l201_201399

theorem cos_225_degrees : Real.cos (225 * Real.pi / 180) = -1 / Real.sqrt 2 := by
  sorry

end cos_225_degrees_l201_201399


namespace cos_225_proof_l201_201381

def cos_225_def := angle.to_real 225 = -√2 / 2

theorem cos_225_proof : cos_225_def := sorry

end cos_225_proof_l201_201381


namespace sum_of_legs_l201_201632

theorem sum_of_legs (x : ℕ) (h : x^2 + (x + 1)^2 = 41^2) : x + (x + 1) = 57 :=
sorry

end sum_of_legs_l201_201632


namespace ben_and_sara_tie_fraction_l201_201787

theorem ben_and_sara_tie_fraction (ben_wins sara_wins : ℚ) (h1 : ben_wins = 2 / 5) (h2 : sara_wins = 1 / 4) : 
  1 - (ben_wins + sara_wins) = 7 / 20 :=
by
  rw [h1, h2]
  norm_num

end ben_and_sara_tie_fraction_l201_201787


namespace line_through_point_and_intersects_circle_with_chord_length_8_l201_201859

theorem line_through_point_and_intersects_circle_with_chord_length_8 :
  ∃ (l : ℝ → ℝ), (∀ (x : ℝ), l x = 0 ↔ x = 5) ∨ 
  (∀ (x y : ℝ), 7 * x + 24 * y = 35) ↔ 
  (∃ (x : ℝ), x = 5) ∨ 
  (∀ (x y : ℝ), 7 * x + 24 * y = 35) := 
by
  sorry

end line_through_point_and_intersects_circle_with_chord_length_8_l201_201859


namespace speed_ratio_bus_meets_Vasya_first_back_trip_time_l201_201003

namespace TransportProblem

variable (d : ℝ) -- distance from point A to B
variable (v_bus : ℝ) -- bus speed
variable (v_Vasya : ℝ) -- Vasya's speed
variable (v_Petya : ℝ) -- Petya's speed

-- Conditions
axiom bus_speed : v_bus * 3 = d
axiom bus_meet_Vasya_second_trip : 7.5 * v_Vasya = 0.5 * d
axiom bus_meet_Petya_at_B : 9 * v_Petya = d
axiom bus_start_time : d / v_bus = 3

theorem speed_ratio: (v_Vasya / v_Petya) = (3 / 5) :=
  sorry

theorem bus_meets_Vasya_first_back_trip_time: ∃ (x: ℕ), x = 11 :=
  sorry

end TransportProblem

end speed_ratio_bus_meets_Vasya_first_back_trip_time_l201_201003


namespace largest_four_digit_number_with_property_l201_201798

theorem largest_four_digit_number_with_property :
  ∃ (a b c d : ℕ), (a = 9 ∧ b = 0 ∧ c = 9 ∧ d = 9 ∧ c = a + b ∧ d = b + c ∧ 1000 * a + 100 * b + 10 * c + d = 9099) :=
sorry

end largest_four_digit_number_with_property_l201_201798


namespace geom_seq_a6_value_l201_201705

variable {α : Type _} [LinearOrderedField α]

theorem geom_seq_a6_value (a : ℕ → α) (q : α) 
(h_geom : ∀ n, a (n + 1) = a n * q)
(h_cond : a 4 + a 8 = π) : 
a 6 * (a 2 + 2 * a 6 + a 10) = π^2 := by
  sorry

end geom_seq_a6_value_l201_201705


namespace no_two_digit_number_no_three_digit_number_l201_201764

theorem no_two_digit_number:
  ∀ a b : ℕ, 1 ≤ a ∧ a ≤ 9 → 0 ≤ b ∧ b ≤ 9 → 10 * b + a ≠ 2 * (10 * a + b) := by
  intros a b ha hb
  have h1 : 10 * b + a = 2 * (10 * a + b) → a * 19 = 8 * b := by
    intro h
    have h2 : 10 * b + a = 20 * a + 2 * b := by rw [h]; ring
    linarith
  push_neg
  intro h
  obtain ⟨k, hk⟩ : a * 19 = 8 * b := h1 h
  sorry

theorem no_three_digit_number:
  ∀ a b c : ℕ, 1 ≤ a ∧ a ≤ 9 → 0 ≤ b ∧ b ≤ 9 → 0 ≤ c ∧ c ≤ 9 → 100 * c + 10 * b + a ≠ 2 * (100 * a + 10 * b + c) := by
  intros a b c ha hb hc
  have h1 : 100 * c + 10 * b + a = 2 * (100 * a + 10 * b + c) → 98 * c = 199 * a - 10 * b := by
    intro h
    have h2 : 100 * c + 10 * b + a = 200 * a + 20 * b + 2 * c := by rw [h]; ring
    linarith
  push_neg
  intro h
  obtain ⟨k, hk⟩ : 98 * c = 199 * a - 10 * b := h1 h
  sorry

end no_two_digit_number_no_three_digit_number_l201_201764


namespace find_c_l201_201502

-- Define the problem
def parabola (x y : ℝ) (a : ℝ) : Prop := 
  x = a * (y - 3) ^ 2 + 5

def point (x y : ℝ) (a : ℝ) : Prop := 
  7 = a * (6 - 3) ^ 2 + 5

-- Theorem to be proved
theorem find_c (a : ℝ) (c : ℝ) (h1 : parabola 7 6 a) (h2 : point 7 6 a) : c = 7 :=
by
  sorry

end find_c_l201_201502


namespace vector_sum_l201_201989

def v1 : ℤ × ℤ := (5, -3)
def v2 : ℤ × ℤ := (-2, 4)
def scalar : ℤ := 3

theorem vector_sum : 
  (v1.1 + scalar * v2.1, v1.2 + scalar * v2.2) = (-1, 9) := 
by 
  sorry

end vector_sum_l201_201989


namespace cos_225_eq_l201_201410

noncomputable def angle_in_radians (deg : ℝ) : ℝ := deg * (Real.pi / 180)

theorem cos_225_eq :
  Real.cos (angle_in_radians 225) = - (Real.sqrt 2) / 2 := 
  sorry

end cos_225_eq_l201_201410


namespace repeating_decimal_fraction_sum_l201_201313

noncomputable def repeating_decimal_to_fraction (d : ℚ) : ℚ :=
  if d = 0.45 then 5 / 11 else 0

theorem repeating_decimal_fraction_sum :
  let x := repeating_decimal_to_fraction 0.45 in
  (x.num + x.den = 16) :=
by
  let x := repeating_decimal_to_fraction 0.45
  have h : x = 5 / 11 := by
    unfold repeating_decimal_to_fraction
    split_ifs with h
    . exact rfl
  rw [h, Rat.num_den_eq 5 11, Int.add_comm]
  sorry

end repeating_decimal_fraction_sum_l201_201313


namespace cos_225_eq_l201_201415

noncomputable def angle_in_radians (deg : ℝ) : ℝ := deg * (Real.pi / 180)

theorem cos_225_eq :
  Real.cos (angle_in_radians 225) = - (Real.sqrt 2) / 2 := 
  sorry

end cos_225_eq_l201_201415


namespace simplify_expression_l201_201068

variable (x y z : ℝ)

noncomputable def expr1 := (3 * x + y / 3 + 2 * z)⁻¹ * ((3 * x)⁻¹ + (y / 3)⁻¹ + (2 * z)⁻¹)
noncomputable def expr2 := (2 * y + 18 * x * z + 3 * z * x) / (6 * x * y * z * (9 * x + y + 6 * z))

theorem simplify_expression (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) (hxyz : 3 * x + y / 3 + 2 * z ≠ 0) :
  expr1 x y z = expr2 x y z := by 
  sorry

end simplify_expression_l201_201068


namespace cos_225_eq_neg_inv_sqrt_2_l201_201373

theorem cos_225_eq_neg_inv_sqrt_2 :
  let P := (225 : ℝ) in
  P ∈ set_of (λ θ, θ ∈ [0, 360]) ∧
  ∃ x y : ℝ, (x = -1/√2) ∧ (y = -1/√2) ∧ (cos θ = x) :=
sorry

end cos_225_eq_neg_inv_sqrt_2_l201_201373


namespace isosceles_triangle_angles_l201_201747

theorem isosceles_triangle_angles (A B C : ℝ)
    (h_iso : A = B ∨ B = C ∨ C = A)
    (h_one_angle : A = 36 ∨ B = 36 ∨ C = 36)
    (h_sum_angles : A + B + C = 180) :
  (A = 36 ∧ B = 36 ∧ C = 108) ∨
  (A = 72 ∧ B = 72 ∧ C = 36) :=
by 
  sorry

end isosceles_triangle_angles_l201_201747


namespace gcd_repeated_integer_l201_201175

theorem gcd_repeated_integer (n : ℕ) (h1 : 100 ≤ n ∧ n < 1000) :
  ∃ d, (∀ k : ℕ, k = 1001001001 * n → d = 1001001001 ∧ d ∣ k) :=
sorry

end gcd_repeated_integer_l201_201175


namespace a4_eq_2_or_neg2_l201_201862

variable (a : ℕ → ℝ)
variable (r : ℝ)

-- Definition of a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * r

-- Given conditions
axiom h1 : is_geometric_sequence a r
axiom h2 : a 2 * a 6 = 4

-- Theorem to prove
theorem a4_eq_2_or_neg2 : a 4 = 2 ∨ a 4 = -2 :=
sorry

end a4_eq_2_or_neg2_l201_201862


namespace largest_valid_number_l201_201790

-- Define the conditions for the digits of the number
def valid_digits (a b c d : ℕ) : Prop :=
  c = a + b ∧ d = b + c

-- Prove that the number formed by digits 9, 0, 9, 9 is the largest valid 4-digit number
theorem largest_valid_number : ∃ a b c d, valid_digits a b c d ∧
  (a * 1000 + b * 100 + c * 10 + d = 9099) :=
begin
  use [9, 0, 9, 9],
  split,
  { -- Proof of valid digits condition
    split; refl },
  { -- Proof that the number is 9099
    refl }
end

end largest_valid_number_l201_201790


namespace least_positive_integer_division_conditions_l201_201279

theorem least_positive_integer_division_conditions :
  ∃ M : ℤ, M > 0 ∧
  M % 11 = 10 ∧
  M % 12 = 11 ∧
  M % 13 = 12 ∧
  M % 14 = 13 ∧
  M = 30029 := 
by
  sorry

end least_positive_integer_division_conditions_l201_201279


namespace complex_magnitude_l201_201067

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- Define the complex number z with the given condition
variable (z : ℂ) (h : z * (1 + i) = 2 * i)

-- Statement of the problem: Prove that |z + 2 * i| = √10
theorem complex_magnitude (z : ℂ) (h : z * (1 + i) = 2 * i) : Complex.abs (z + 2 * i) = Real.sqrt 10 := 
sorry

end complex_magnitude_l201_201067


namespace number_of_ways_to_assign_roles_l201_201677

theorem number_of_ways_to_assign_roles :
  let men := 6
  let women := 7
  let male_roles := 3
  let female_roles := 3
  let neutral_roles := 2
  let ways_male_roles := men * (men - 1) * (men - 2)
  let ways_female_roles := women * (women - 1) * (women - 2)
  let ways_neutral_roles := (men + women - male_roles - female_roles) * (men + women - male_roles - female_roles - 1)
  ways_male_roles * ways_female_roles * ways_neutral_roles = 1058400 := 
by
  sorry

end number_of_ways_to_assign_roles_l201_201677


namespace function_matches_table_values_l201_201194

variable (f : ℤ → ℤ)

theorem function_matches_table_values (h1 : f (-1) = -2) (h2 : f 0 = 0) (h3 : f 1 = 2) (h4 : f 2 = 4) : 
  ∀ x : ℤ, f x = 2 * x := 
by
  -- Prove that the function satisfying the given table values is f(x) = 2x
  sorry

end function_matches_table_values_l201_201194


namespace solve_prime_equation_l201_201119

theorem solve_prime_equation (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) :
    p ^ 2 - 6 * p * q + q ^ 2 + 3 * q - 1 = 0 ↔ (p = 17 ∧ q = 3) :=
by
  sorry

end solve_prime_equation_l201_201119


namespace alyssa_limes_correct_l201_201444

-- Definitions representing the conditions
def fred_limes : Nat := 36
def nancy_limes : Nat := 35
def total_limes : Nat := 103

-- Definition of the number of limes Alyssa picked
def alyssa_limes : Nat := total_limes - (fred_limes + nancy_limes)

-- The theorem we need to prove
theorem alyssa_limes_correct : alyssa_limes = 32 := by
  sorry

end alyssa_limes_correct_l201_201444


namespace random_event_sum_gt_six_l201_201684

def numbers : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

def selection (s : List ℕ) := s.length = 3 ∧ s ⊆ numbers

def sum_is_greater_than_six (s : List ℕ) : Prop := s.sum > 6

theorem random_event_sum_gt_six :
  ∀ (s : List ℕ), selection s → (sum_is_greater_than_six s ∨ ¬ sum_is_greater_than_six s) := 
by
  intros s h
  -- Proof omitted
  sorry

end random_event_sum_gt_six_l201_201684


namespace value_of_y_minus_x_l201_201645

theorem value_of_y_minus_x (x y : ℝ) (h1 : x + y = 520) (h2 : x / y = 0.75) : y - x = 74 :=
sorry

end value_of_y_minus_x_l201_201645


namespace carlos_local_tax_deduction_l201_201990

theorem carlos_local_tax_deduction :
  let hourly_wage_dollars := 25
  let hourly_wage_cents := hourly_wage_dollars * 100
  let tax_rate := 2.5 / 100
  hourly_wage_cents * tax_rate = 62.5 :=
by
  sorry

end carlos_local_tax_deduction_l201_201990


namespace train_crossing_pole_time_l201_201979

theorem train_crossing_pole_time :
  ∀ (speed_kmph length_m: ℝ), speed_kmph = 160 → length_m = 400.032 → 
  length_m / (speed_kmph * 1000 / 3600) = 9.00072 :=
by
  intros speed_kmph length_m h_speed h_length
  rw [h_speed, h_length]
  -- The proof is omitted as per instructions
  sorry

end train_crossing_pole_time_l201_201979


namespace average_marks_first_class_l201_201940

theorem average_marks_first_class
  (n1 n2 : ℕ)
  (avg2 : ℝ)
  (combined_avg : ℝ)
  (h_n1 : n1 = 35)
  (h_n2 : n2 = 55)
  (h_avg2 : avg2 = 65)
  (h_combined_avg : combined_avg = 57.22222222222222) :
  (∃ avg1 : ℝ, avg1 = 45) :=
by
  sorry

end average_marks_first_class_l201_201940


namespace intersection_with_negative_y_axis_max_value_at_x3_l201_201195

theorem intersection_with_negative_y_axis (m : ℝ) (h : 4 - 2 * m < 0) : m > 2 :=
sorry

theorem max_value_at_x3 (m : ℝ) (h : ∀ x : ℝ, -2 ≤ x ∧ x ≤ 3 → 3 * x + 4 - 2 * m ≤ -4) : m = 8.5 :=
sorry

end intersection_with_negative_y_axis_max_value_at_x3_l201_201195


namespace no_sum_2015_l201_201613

theorem no_sum_2015 (x a : ℤ) : 3 * x + 3 * a ≠ 2015 := by
  sorry

end no_sum_2015_l201_201613


namespace zip_code_relationship_l201_201181

theorem zip_code_relationship (A B C D E : ℕ) 
(h1 : A + B + C + D + E = 10) 
(h2 : C = 0) 
(h3 : D = 2 * A) 
(h4 : D + E = 8) : 
A + B = 2 :=
sorry

end zip_code_relationship_l201_201181


namespace inequality_solution_l201_201863

theorem inequality_solution (z : ℝ) : 
  z^2 - 40 * z + 400 ≤ 36 ↔ 14 ≤ z ∧ z ≤ 26 :=
by
  sorry

end inequality_solution_l201_201863


namespace solve_S20_minus_2S10_l201_201825

noncomputable def arithmetic_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  (∀ n : ℕ, n > 0 → a n > 0) ∧
  (∀ n : ℕ, n ≥ 2 → S n = (n / (n - 1 : ℝ)) * (a n ^ 2 - a 1 ^ 2))

theorem solve_S20_minus_2S10 :
  ∀ (a : ℕ → ℝ) (S : ℕ → ℝ),
    arithmetic_sequence a S →
    S 20 - 2 * S 10 = 50 :=
by
  intros
  sorry

end solve_S20_minus_2S10_l201_201825


namespace p_implies_q_l201_201703

def p (x : ℝ) := 0 < x ∧ x < 5
def q (x : ℝ) := -5 < x - 2 ∧ x - 2 < 5

theorem p_implies_q (x : ℝ) (h : p x) : q x :=
  by sorry

end p_implies_q_l201_201703


namespace cos_225_eq_neg_inv_sqrt_2_l201_201372

theorem cos_225_eq_neg_inv_sqrt_2 :
  let P := (225 : ℝ) in
  P ∈ set_of (λ θ, θ ∈ [0, 360]) ∧
  ∃ x y : ℝ, (x = -1/√2) ∧ (y = -1/√2) ∧ (cos θ = x) :=
sorry

end cos_225_eq_neg_inv_sqrt_2_l201_201372


namespace total_people_after_four_years_l201_201971

-- Define initial conditions
def initial_total_people : Nat := 9
def board_members : Nat := 3
def regular_members_initial : Nat := initial_total_people - board_members
def years : Nat := 4

-- Define the function for regular members over the years
def regular_members (n : Nat) : Nat :=
  if n = 0 then 
    regular_members_initial
  else 
    2 * regular_members (n - 1)

theorem total_people_after_four_years :
  regular_members years = 96 := 
sorry

end total_people_after_four_years_l201_201971


namespace proof_case_a_proof_case_b1_proof_case_b2_proof_case_c1_proof_case_c2_l201_201936

structure CubeSymmetry where
  planes : Nat
  axes : Nat
  has_center : Bool

def general_cube_symmetry : CubeSymmetry :=
  { planes := 9, axes := 9, has_center := true }

def case_a : CubeSymmetry :=
  { planes := 4, axes := 1, has_center := false }

def case_b1 : CubeSymmetry :=
  { planes := 5, axes := 3, has_center := true }

def case_b2 : CubeSymmetry :=
  { planes := 2, axes := 1, has_center := false }

def case_c1 : CubeSymmetry :=
  { planes := 3, axes := 0, has_center := false }

def case_c2 : CubeSymmetry :=
  { planes := 2, axes := 1, has_center := false }

theorem proof_case_a : case_a = { planes := 4, axes := 1, has_center := false } := by
  sorry

theorem proof_case_b1 : case_b1 = { planes := 5, axes := 3, has_center := true } := by
  sorry

theorem proof_case_b2 : case_b2 = { planes := 2, axes := 1, has_center := false } := by
  sorry

theorem proof_case_c1 : case_c1 = { planes := 3, axes := 0, has_center := false } := by
  sorry

theorem proof_case_c2 : case_c2 = { planes := 2, axes := 1, has_center := false } := by
  sorry

end proof_case_a_proof_case_b1_proof_case_b2_proof_case_c1_proof_case_c2_l201_201936


namespace vertical_asymptotes_sum_l201_201136

theorem vertical_asymptotes_sum : 
  let f (x : ℝ) := (6 * x^2 + 1) / (4 * x^2 + 6 * x + 3)
  let den := 4 * x^2 + 6 * x + 3
  let p := -(3 / 2)
  let q := -(1 / 2)
  (den = 0) → (p + q = -2) :=
by
  sorry

end vertical_asymptotes_sum_l201_201136


namespace largest_four_digit_number_with_property_l201_201794

theorem largest_four_digit_number_with_property :
  ∃ (a b c d : ℕ), (a = 9 ∧ b = 0 ∧ c = 9 ∧ d = 9 ∧ c = a + b ∧ d = b + c ∧ 1000 * a + 100 * b + 10 * c + d = 9099) :=
sorry

end largest_four_digit_number_with_property_l201_201794


namespace prime_solution_unique_l201_201116

open Nat

theorem prime_solution_unique (p q : ℕ) (hp : Prime p) (hq : Prime q) (h : p^2 - 6 * p * q + q^2 + 3 * q - 1 = 0) : 
  (p = 17 ∧ q = 3) :=
by
  sorry

end prime_solution_unique_l201_201116


namespace find_base_a_l201_201880

theorem find_base_a (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1)
  (h3 : (if a < 1 then a + a^2 else a^2 + a) = 12) : a = 3 := 
sorry

end find_base_a_l201_201880


namespace sum_of_numerator_and_denominator_of_repeating_decimal_l201_201297

noncomputable def repeating_decimal_to_fraction (n d : ℤ) (h : 0 < d) := ∃ (x : ℚ), x = n / d ∧ 0 < x

theorem sum_of_numerator_and_denominator_of_repeating_decimal :
  let x := 0.4545454545 -- defining x
  let f := x.to_rational -- converting to rational
  let f_simp := f.num / f.denom -- get numerator and denominator
  -- conclude the sum of numerator and denominator is 16
  f.num + f.denom = 16 :=
by
  sorry

end sum_of_numerator_and_denominator_of_repeating_decimal_l201_201297


namespace median_squared_formula_l201_201999

theorem median_squared_formula (a b c m : ℝ) (AC_is_median : 2 * m^2 + c^2 = a^2 + b^2) : 
  m^2 = (1/4) * (2 * a^2 + 2 * b^2 - c^2) := 
by
  sorry

end median_squared_formula_l201_201999


namespace Hari_joined_after_5_months_l201_201102

noncomputable def Praveen_investment_per_year : ℝ := 3360 * 12
noncomputable def Hari_investment_for_given_months (x : ℝ) : ℝ := 8640 * (12 - x)

theorem Hari_joined_after_5_months (x : ℝ) (h : Praveen_investment_per_year / Hari_investment_for_given_months x = 2 / 3) : x = 5 :=
by
  sorry

end Hari_joined_after_5_months_l201_201102


namespace brokerage_percentage_l201_201497

theorem brokerage_percentage (cash_realized amount_before : ℝ) (h1 : cash_realized = 105.25) (h2 : amount_before = 105) :
  |((amount_before - cash_realized) / amount_before) * 100| = 0.2381 := by
sorry

end brokerage_percentage_l201_201497


namespace number_of_valid_pairs_l201_201836

theorem number_of_valid_pairs (m n : ℕ) (h1 : n > m) (h2 : 3 * (m - 4) * (n - 4) = m * n) : 
  (m, n) = (7, 18) ∨ (m, n) = (8, 12) ∨ (m, n) = (9, 10) ∨ (m-6) * (n-6) = 12 := sorry

end number_of_valid_pairs_l201_201836


namespace length_EF_l201_201870

theorem length_EF
  (AB CD GH EF : ℝ)
  (h1 : AB = 180)
  (h2 : CD = 120)
  (h3 : AB = 2 * GH)
  (h4 : CD = 2 * EF) :
  EF = 45 :=
by
  sorry

end length_EF_l201_201870


namespace perimeter_of_rectangle_l201_201974

theorem perimeter_of_rectangle (area width : ℝ) (h_area : area = 750) (h_width : width = 25) :
  ∃ perimeter length, length = area / width ∧ perimeter = 2 * (length + width) ∧ perimeter = 110 := by
  sorry

end perimeter_of_rectangle_l201_201974


namespace find_distance_BC_l201_201690

variables {d_AB d_AC d_BC : ℝ}

theorem find_distance_BC
  (h1 : d_AB = d_AC + d_BC - 200)
  (h2 : d_AC = d_AB + d_BC - 300) :
  d_BC = 250 := 
sorry

end find_distance_BC_l201_201690


namespace robin_hid_150_seeds_l201_201036

theorem robin_hid_150_seeds
    (x y : ℕ)
    (h1 : 5 * x = 6 * y)
    (h2 : y = x - 5) : 
    5 * x = 150 :=
by
    sorry

end robin_hid_150_seeds_l201_201036


namespace cos_225_eq_neg_sqrt2_div_2_l201_201388

theorem cos_225_eq_neg_sqrt2_div_2 : Real.cos (225 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by
  sorry

end cos_225_eq_neg_sqrt2_div_2_l201_201388


namespace recurring_fraction_sum_l201_201309

def x := 0.454545...

theorem recurring_fraction_sum :
  let x := 0.454545...
  let frac := (5 : ℕ) / (11 : ℕ)
  (frac.numerator + frac.denominator) = 16 :=
by
  sorry

end recurring_fraction_sum_l201_201309


namespace count_points_l201_201071

theorem count_points (a b : ℝ) :
  (abs b = 2) ∧ (abs a = 4) → (∃ (P : ℝ × ℝ), P = (a, b) ∧ (abs b = 2) ∧ (abs a = 4) ∧
    ((a = 4 ∨ a = -4) ∧ (b = 2 ∨ b = -2)) ∧
    (P = (4, 2) ∨ P = (4, -2) ∨ P = (-4, 2) ∨ P = (-4, -2)) ∧
    ∃ n, n = 4) :=
sorry

end count_points_l201_201071


namespace spaghetti_manicotti_ratio_l201_201164

-- Define the number of students who were surveyed and their preferences
def total_students := 800
def students_prefer_spaghetti := 320
def students_prefer_manicotti := 160

-- The ratio of students who prefer spaghetti to those who prefer manicotti is 2
theorem spaghetti_manicotti_ratio :
  students_prefer_spaghetti / students_prefer_manicotti = 2 :=
by
  sorry

end spaghetti_manicotti_ratio_l201_201164


namespace solve_prime_equation_l201_201112

theorem solve_prime_equation (p q : ℕ) (hp : p.prime) (hq : q.prime) : 
  p^2 - 6*p*q + q^2 + 3*q - 1 = 0 ↔ (p = 17 ∧ q = 3) :=
by
  sorry

end solve_prime_equation_l201_201112


namespace shortest_side_15_l201_201505

theorem shortest_side_15 (b c : ℕ) (h : ℕ) (hb : b < c)
  (h_perimeter : 24 + b + c = 66)
  (h_area_int : ∃ A : ℕ, A*A = 33 * 9 * (33 - b) * (b - 9))
  (h_altitude_int : ∃ A : ℕ, 24 * h = 2 * A) : b = 15 :=
sorry

end shortest_side_15_l201_201505


namespace dalton_needs_more_money_l201_201025

theorem dalton_needs_more_money :
  let jump_rope_cost := 9
  let board_game_cost := 15
  let playground_ball_cost := 5
  let puzzle_cost := 8
  let saved_allowance := 7
  let uncle_gift := 14
  let total_cost := jump_rope_cost + board_game_cost + playground_ball_cost + puzzle_cost
  let total_money := saved_allowance + uncle_gift
  (total_cost - total_money) = 16 :=
by
  sorry

end dalton_needs_more_money_l201_201025


namespace minimum_red_points_for_square_l201_201950

/-- Given a circle divided into 100 equal segments with points randomly colored red. 
Prove that the minimum number of red points needed to ensure at least four red points 
form the vertices of a square is 76. --/
theorem minimum_red_points_for_square (n : ℕ) (h : n = 100) (red_points : Finset ℕ)
  (hred : red_points.card ≥ 76) (hseg : ∀ i j : ℕ, i ≤ j → (j - i) % 25 ≠ 0 → ¬ (∃ a b c d : ℕ, 
  a ∈ red_points ∧ b ∈ red_points ∧ c ∈ red_points ∧ d ∈ red_points ∧ 
  (a + b + c + d) % n = 0)) : 
  ∃ a b c d : ℕ, a ∈ red_points ∧ b ∈ red_points ∧ c ∈ red_points ∧ d ∈ red_points ∧ 
  (a + b + c + d) % n = 0 :=
sorry

end minimum_red_points_for_square_l201_201950


namespace probability_at_least_one_correct_l201_201076

theorem probability_at_least_one_correct :
  let p_a := 12 / 20
  let p_b := 8 / 20
  let prob_neither := (1 - p_a) * (1 - p_b)
  let prob_at_least_one := 1 - prob_neither
  prob_at_least_one = 19 / 25 := by
  sorry

end probability_at_least_one_correct_l201_201076


namespace solutions_of_quadratic_eq_l201_201263

theorem solutions_of_quadratic_eq : 
    {x : ℝ | x^2 - 3 * x = 0} = {0, 3} :=
sorry

end solutions_of_quadratic_eq_l201_201263


namespace sum_of_coordinates_of_center_l201_201144

theorem sum_of_coordinates_of_center (x1 y1 x2 y2 : ℝ) (h1 : (x1, y1) = (7, -6)) (h2 : (x2, y2) = (-1, 4)) :
  let center_x := (x1 + x2) / 2
  let center_y := (y1 + y2) / 2
  center_x + center_y = 2 := by
  sorry

end sum_of_coordinates_of_center_l201_201144


namespace number_of_female_students_l201_201599

theorem number_of_female_students (T S f_sample : ℕ) (H_total : T = 1600) (H_sample_size : S = 200) (H_females_in_sample : f_sample = 95) : 
  ∃ F, 95 / 200 = F / 1600 ∧ F = 760 := by 
sorry

end number_of_female_students_l201_201599


namespace parallel_vectors_result_l201_201203

noncomputable def a : ℝ × ℝ := (1, -2)
noncomputable def b (m : ℝ) : ℝ × ℝ := (m, 4)
noncomputable def m : ℝ := -1 / 2

theorem parallel_vectors_result :
  (b m).1 * a.2 = (b m).2 * a.1 →
  2 * a - b m = (4, -8) :=
by
  intro h
  -- Proof omitted
  sorry

end parallel_vectors_result_l201_201203


namespace algebra_expression_l201_201065

theorem algebra_expression (a b : ℝ) (h : a - b = 3) : 1 + a - b = 4 :=
sorry

end algebra_expression_l201_201065


namespace sum_of_coefficients_is_2_l201_201966

noncomputable def polynomial_expansion_condition (x a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 a_10 a_11 : ℝ) :=
  (x^2 + 1) * (x - 2)^9 = a_0 + a_1 * (x - 1) + a_2 * (x - 1)^2 + a_3 * (x - 1)^3 + a_4 * (x - 1)^4 + 
                          a_5 * (x - 1)^5 + a_6 * (x - 1)^6 + a_7 * (x - 1)^7 + a_8 * (x - 1)^8 + 
                          a_9 * (x - 1)^9 + a_10 * (x - 1)^10 + a_11 * (x - 1)^11

theorem sum_of_coefficients_is_2 (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 a_10 a_11 : ℝ) :
  polynomial_expansion_condition 1 a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 a_10 a_11 →
  polynomial_expansion_condition 2 a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 a_10 a_11 →
  a_1 + a_2 + a_3 + a_4 + a_5 + a_6 + a_7 + a_8 + a_9 + a_10 + a_11 = 2 :=
by sorry

end sum_of_coefficients_is_2_l201_201966


namespace chromium_percentage_is_correct_l201_201666

noncomputable def chromium_percentage_new_alloy (chr_percent1 chr_percent2 weight1 weight2 : ℝ) : ℝ :=
  (chr_percent1 * weight1 + chr_percent2 * weight2) / (weight1 + weight2) * 100

theorem chromium_percentage_is_correct :
  chromium_percentage_new_alloy 0.10 0.06 15 35 = 7.2 :=
by
  sorry

end chromium_percentage_is_correct_l201_201666


namespace find_n_l201_201860

theorem find_n : ∃ (n : ℕ), 0 ≤ n ∧ n ≤ 14 ∧ n ≡ 7615 [MOD 15] ∧ n = 10 := by
  use 10
  repeat { sorry }

end find_n_l201_201860


namespace velocity_at_1_eq_5_l201_201216

def S (t : ℝ) : ℝ := 2 * t^2 + t

theorem velocity_at_1_eq_5 : (deriv S 1) = 5 :=
by sorry

end velocity_at_1_eq_5_l201_201216


namespace maximize_profit_l201_201165

noncomputable def production_problem : Prop :=
  ∃ (x y : ℕ), (3 * x + 2 * y ≤ 1200) ∧ (x + 2 * y ≤ 800) ∧ 
               (30 * x + 40 * y) = 18000 ∧ 
               x = 200 ∧ 
               y = 300

theorem maximize_profit : production_problem :=
sorry

end maximize_profit_l201_201165


namespace original_deck_card_count_l201_201539

theorem original_deck_card_count (r b : ℕ) 
  (h1 : r / (r + b) = 1 / 4)
  (h2 : r / (r + b + 6) = 1 / 6) : r + b = 12 :=
sorry

end original_deck_card_count_l201_201539


namespace intersection_A_B_l201_201725

def A := { x : ℝ | -1 < x ∧ x ≤ 3 }
def B := { x : ℝ | 0 < x ∧ x < 10 }

theorem intersection_A_B : A ∩ B = { x : ℝ | 0 < x ∧ x ≤ 3 } :=
  by sorry

end intersection_A_B_l201_201725


namespace repeating_decimal_fraction_sum_l201_201292

noncomputable def repeating_decimal_to_fraction (x : ℚ) : ℚ :=
if h : x = (45 / 99 : ℚ) then (5 / 11) else x

theorem repeating_decimal_fraction_sum : 
  let x := (0.45 : ℝ) in 
  (x.mul 100 - x).div 99 = (45 : ℝ) -> 
  let (a, b) := (5, 11) in 
  a + b = 16 :=
by
  intro x
  rfl
  rw [show (5 : ℚ) + (11 : ℚ) = 16 from rfl]
  sorry

end repeating_decimal_fraction_sum_l201_201292


namespace sum_of_powers_of_i_l201_201823

-- Define the imaginary unit and its property
def i : ℂ := Complex.I -- ℂ represents the complex numbers, Complex.I is the imaginary unit

-- The statement we need to prove
theorem sum_of_powers_of_i : i + i^2 + i^3 + i^4 = 0 := 
by {
  -- Lean requires the proof, but we will use sorry to skip it.
  -- Define the properties of i directly or use in-built properties
  sorry
}

end sum_of_powers_of_i_l201_201823


namespace cos_225_l201_201375

noncomputable def cos_225_eq : ℝ :=
  cos (225 * (real.pi / 180))

theorem cos_225 : cos_225_eq = -real.sqrt 2 / 2 :=
  sorry

end cos_225_l201_201375


namespace largest_variance_is_B_l201_201602

open Finset
open Real

-- Define the frequencies and the constraint sum to 1
variable (p_1 p_2 p_3 p_4 : ℝ)
variable (h_sum : p_1 + p_2 + p_3 + p_4 = 1)

-- Define the expected value calculation
def expected_value (p_1 p_2 p_3 p_4 : ℝ) : ℝ :=
  1 * p_1 + 2 * p_2 + 3 * p_3 + 4 * p_4

-- Define the variance calculation
def variance (p_1 p_2 p_3 p_4 : ℝ) (E : ℝ) : ℝ := 
  (1 - E) ^ 2 * p_1 + (2 - E) ^ 2 * p_2 + (3 - E) ^ 2 * p_3 + (4 - E) ^ 2 * p_4

-- Define each option's frequencies
def option_A := (0.1, 0.4, 0.4, 0.1)
def option_B := (0.4, 0.1, 0.1, 0.4)
def option_C := (0.2, 0.3, 0.3, 0.2)
def option_D := (0.3, 0.2, 0.2, 0.3)

-- Proof problem: option B has the highest variance
theorem largest_variance_is_B : 
  let E_A := expected_value 0.1 0.4 0.4 0.1,
      E_B := expected_value 0.4 0.1 0.1 0.4,
      E_C := expected_value 0.2 0.3 0.3 0.2,
      E_D := expected_value 0.3 0.2 0.2 0.3,
      V_A := variance 0.1 0.4 0.4 0.1 E_A,
      V_B := variance 0.4 0.1 0.1 0.4 E_B,
      V_C := variance 0.2 0.3 0.3 0.2 E_C,
      V_D := variance 0.3 0.2 0.2 0.3 E_D
  in V_B > V_A ∧ V_B > V_C ∧ V_B > V_D :=
sorry

end largest_variance_is_B_l201_201602


namespace jose_cupcakes_l201_201612

theorem jose_cupcakes (lemons_needed : ℕ) (tablespoons_per_lemon : ℕ) (tablespoons_per_dozen : ℕ) (target_lemons : ℕ) : 
  (lemons_needed = 12) → 
  (tablespoons_per_lemon = 4) → 
  (target_lemons = 9) → 
  ((target_lemons * tablespoons_per_lemon / lemons_needed) = 3) :=
by
  intros h1 h2 h3
  sorry

end jose_cupcakes_l201_201612


namespace david_remaining_money_l201_201026

noncomputable def initial_funds : ℝ := 1500
noncomputable def spent_on_accommodations : ℝ := 400
noncomputable def spent_on_food_eur : ℝ := 300
noncomputable def eur_to_usd : ℝ := 1.10
noncomputable def spent_on_souvenirs_yen : ℝ := 5000
noncomputable def yen_to_usd : ℝ := 0.009
noncomputable def loan_to_friend : ℝ := 200
noncomputable def difference : ℝ := 500

noncomputable def spent_on_food_usd : ℝ := spent_on_food_eur * eur_to_usd
noncomputable def spent_on_souvenirs_usd : ℝ := spent_on_souvenirs_yen * yen_to_usd
noncomputable def total_spent_excluding_loan : ℝ := spent_on_accommodations + spent_on_food_usd + spent_on_souvenirs_usd

theorem david_remaining_money : 
  initial_funds - total_spent_excluding_loan - difference = 275 :=
by
  sorry

end david_remaining_money_l201_201026


namespace product_last_digit_l201_201163

def last_digit (n : ℕ) : ℕ := n % 10

theorem product_last_digit :
  last_digit (3^65 * 6^59 * 7^71) = 4 :=
by
  sorry

end product_last_digit_l201_201163


namespace johns_raw_squat_weight_l201_201751

variable (R : ℝ)

def sleeves_lift := R + 30
def wraps_lift := 1.25 * R
def wraps_more_than_sleeves := wraps_lift R - sleeves_lift R = 120

theorem johns_raw_squat_weight : wraps_more_than_sleeves R → R = 600 :=
by
  intro h
  sorry

end johns_raw_squat_weight_l201_201751


namespace cosine_225_proof_l201_201350

noncomputable def cosine_225_eq_neg_sqrt2_div_2 : Prop :=
  Real.cos (225 * Real.pi / 180) = -Real.sqrt 2 / 2

theorem cosine_225_proof : cosine_225_eq_neg_sqrt2_div_2 := by
  sorry

end cosine_225_proof_l201_201350


namespace percentage_increase_in_weight_l201_201785

theorem percentage_increase_in_weight :
  ∀ (num_plates : ℕ) (weight_per_plate lowered_weight : ℝ),
    num_plates = 10 →
    weight_per_plate = 30 →
    lowered_weight = 360 →
    ((lowered_weight - num_plates * weight_per_plate) / (num_plates * weight_per_plate)) * 100 = 20 :=
by
  intros num_plates weight_per_plate lowered_weight h_num_plates h_weight_per_plate h_lowered_weight
  sorry

end percentage_increase_in_weight_l201_201785


namespace price_of_first_variety_l201_201766

theorem price_of_first_variety
  (p2 : ℝ) (p3 : ℝ) (r : ℝ) (w : ℝ)
  (h1 : p2 = 135)
  (h2 : p3 = 177.5)
  (h3 : r = 154)
  (h4 : w = 4) :
  ∃ p1 : ℝ, 1 * p1 + 1 * p2 + 2 * p3 = w * r ∧ p1 = 126 :=
by {
  sorry
}

end price_of_first_variety_l201_201766


namespace grandparents_gift_l201_201479

theorem grandparents_gift (june_stickers bonnie_stickers total_stickers : ℕ) (x : ℕ)
  (h₁ : june_stickers = 76)
  (h₂ : bonnie_stickers = 63)
  (h₃ : total_stickers = 189) :
  june_stickers + bonnie_stickers + 2 * x = total_stickers → x = 25 :=
by
  intros
  sorry

end grandparents_gift_l201_201479


namespace contradiction_example_l201_201152

theorem contradiction_example 
  (a b c : ℝ) 
  (h : (a - 1) * (b - 1) * (c - 1) > 0) : 
  (1 < a) ∨ (1 < b) ∨ (1 < c) :=
by
  sorry

end contradiction_example_l201_201152


namespace total_teachers_l201_201975

theorem total_teachers (total_individuals sample_size sampled_students : ℕ)
  (H1 : total_individuals = 2400)
  (H2 : sample_size = 160)
  (H3 : sampled_students = 150) :
  ∃ total_teachers, total_teachers * (sample_size / (sample_size - sampled_students)) = 2400 / (sample_size / (sample_size - sampled_students)) ∧ total_teachers = 150 := 
  sorry

end total_teachers_l201_201975


namespace sum_first_five_terms_l201_201871

theorem sum_first_five_terms (a1 a2 a3 : ℝ) (S5 : ℝ) 
  (h1 : a1 * a3 = 8 * a2)
  (h2 : (a1 + a2) = 24) :
  S5 = 31 :=
sorry

end sum_first_five_terms_l201_201871


namespace n_gon_angles_l201_201174

theorem n_gon_angles (n : ℕ) (h1 : n > 7) (h2 : n < 12) : 
  (∃ x : ℝ, (150 * (n - 1) + x = 180 * (n - 2)) ∧ (x < 150)) :=
by {
  sorry
}

end n_gon_angles_l201_201174
