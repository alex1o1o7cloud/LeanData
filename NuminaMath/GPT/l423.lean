import Mathlib

namespace angle_hyperbola_l423_42374

theorem angle_hyperbola (a b : ℝ) (e : ℝ) 
  (h1 : a > b) (h2 : b > 0) 
  (hyperbola_eq : ∀ (x y : ℝ), ((x^2)/(a^2) - (y^2)/(b^2) = 1)) 
  (eccentricity_eq : e = 2 + Real.sqrt 6 - Real.sqrt 3 - Real.sqrt 2) :
  ∃ α : ℝ, α = 15 :=
by
  sorry

end angle_hyperbola_l423_42374


namespace second_sample_correct_l423_42317

def total_samples : ℕ := 7341
def first_sample : ℕ := 4221
def second_sample : ℕ := total_samples - first_sample

theorem second_sample_correct : second_sample = 3120 :=
by
  sorry

end second_sample_correct_l423_42317


namespace michael_truck_meetings_l423_42389

theorem michael_truck_meetings :
  let michael_speed := 6
  let truck_speed := 12
  let pail_distance := 200
  let truck_stop_time := 20
  let initial_distance := pail_distance
  ∃ (meetings : ℕ), 
  (michael_speed, truck_speed, pail_distance, truck_stop_time, initial_distance, meetings) = 
  (6, 12, 200, 20, 200, 10) :=
sorry

end michael_truck_meetings_l423_42389


namespace solve_xy_l423_42357

theorem solve_xy (x y : ℕ) :
  (x^2 + (x + y)^2 = (x + 9)^2) ↔ (x = 0 ∧ y = 9) ∨ (x = 8 ∧ y = 7) ∨ (x = 20 ∧ y = 1) :=
by
  sorry

end solve_xy_l423_42357


namespace ratio_ba_in_range_l423_42300

theorem ratio_ba_in_range (a b : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0) 
  (h1 : a + 2 * b = 7) (h2 : a^2 + b^2 ≤ 25) : 
  (3 / 4 : ℝ) ≤ b / a ∧ b / a ≤ 4 / 3 :=
by {
  sorry
}

end ratio_ba_in_range_l423_42300


namespace dogs_in_pet_shop_l423_42358

variable (D C B x : ℕ)

theorem dogs_in_pet_shop 
  (h1 : D = 7 * x) 
  (h2 : B = 8 * x)
  (h3 : D + B = 330) : 
  D = 154 :=
by
  sorry

end dogs_in_pet_shop_l423_42358


namespace number_of_distinct_triangles_l423_42332

-- Definition of the grid
def grid_points : List (ℕ × ℕ) := 
  [(0,0), (1,0), (2,0), (3,0), (0,1), (1,1), (2,1), (3,1)]

-- Definition involving combination logic
def binomial (n k : ℕ) : ℕ := n.choose k

-- Count all possible combinations of 3 points
def total_combinations : ℕ := binomial 8 3

-- Count the degenerate cases (collinear points) in the grid
def degenerate_cases : ℕ := 2 * binomial 4 3

-- The required value of distinct triangles
def distinct_triangles : ℕ := total_combinations - degenerate_cases

theorem number_of_distinct_triangles :
  distinct_triangles = 48 :=
by
  sorry

end number_of_distinct_triangles_l423_42332


namespace value_of_a_minus_b_l423_42385

theorem value_of_a_minus_b (a b : ℤ) (h1 : |a| = 4) (h2 : |b| = 2) (h3 : |a + b| = - (a + b)) :
  a - b = -2 ∨ a - b = -6 := sorry

end value_of_a_minus_b_l423_42385


namespace grid_midpoint_exists_l423_42352

theorem grid_midpoint_exists (points : Fin 5 → ℤ × ℤ) :
  ∃ i j : Fin 5, i ≠ j ∧ (points i).fst % 2 = (points j).fst % 2 ∧ (points i).snd % 2 = (points j).snd % 2 :=
by 
  sorry

end grid_midpoint_exists_l423_42352


namespace max_oranges_to_teachers_l423_42323

theorem max_oranges_to_teachers {n r : ℕ} (h1 : n % 8 = r) (h2 : r < 8) : r = 7 :=
sorry

end max_oranges_to_teachers_l423_42323


namespace problem_composite_for_n_geq_9_l423_42379

theorem problem_composite_for_n_geq_9 (n : ℤ) (h : n ≥ 9) : ∃ k m : ℤ, (2 ≤ k ∧ 2 ≤ m ∧ n + 7 = k * m) :=
by
  sorry

end problem_composite_for_n_geq_9_l423_42379


namespace neg_p_true_l423_42330

theorem neg_p_true :
  (∃ x : ℝ, x^2 ≤ 0) :=
sorry

end neg_p_true_l423_42330


namespace total_students_l423_42310

theorem total_students (T : ℝ) 
  (h1 : 0.28 * T = 280) : 
  T = 1000 :=
by {
  sorry
}

end total_students_l423_42310


namespace stratified_sampling_young_employees_l423_42303

variable (total_employees elderly_employees middle_aged_employees young_employees sample_size : ℕ)

-- Conditions
axiom total_employees_eq : total_employees = 750
axiom elderly_employees_eq : elderly_employees = 150
axiom middle_aged_employees_eq : middle_aged_employees = 250
axiom young_employees_eq : young_employees = 350
axiom sample_size_eq : sample_size = 15

-- The proof problem
theorem stratified_sampling_young_employees :
  young_employees / total_employees * sample_size = 7 :=
by
  sorry

end stratified_sampling_young_employees_l423_42303


namespace pyarelal_loss_l423_42351

theorem pyarelal_loss (P : ℝ) (total_loss : ℝ) (ashok_ratio pyarelal_ratio : ℝ)
  (h1 : ashok_ratio = 1/9) (h2 : pyarelal_ratio = 1)
  (h3 : total_loss = 2000) : (pyarelal_ratio / (ashok_ratio + pyarelal_ratio)) * total_loss = 1800 :=
by
  sorry

end pyarelal_loss_l423_42351


namespace positive_root_of_equation_l423_42331

theorem positive_root_of_equation :
  ∃ a b : ℤ, (a + b * Real.sqrt 3)^3 - 5 * (a + b * Real.sqrt 3)^2 + 2 * (a + b * Real.sqrt 3) - Real.sqrt 3 = 0 ∧
    a + b * Real.sqrt 3 > 0 ∧
    (a + b * Real.sqrt 3) = 3 + Real.sqrt 3 := 
by
  sorry

end positive_root_of_equation_l423_42331


namespace initial_people_in_gym_l423_42373

variables (W A S : ℕ)

theorem initial_people_in_gym (h1 : (W - 3 + 2 - 3 + 4 - 2 + 1 = W + 1))
                              (h2 : (A + 2 - 1 + 3 - 3 + 1 = A + 2))
                              (h3 : (S + 1 - 2 + 1 + 3 - 2 + 2 = S + 3))
                              (final_total : (W + 1) + (A + 2) + (S + 3) + 2 = 30) :
  W + A + S = 22 :=
by 
  sorry

end initial_people_in_gym_l423_42373


namespace ratio_a_b_l423_42346

theorem ratio_a_b (a b c : ℝ) (h1 : a * (-1) ^ 2 + b * (-1) + c = 1) (h2 : a * 3 ^ 2 + b * 3 + c = 1) : 
  a / b = -2 :=
by 
  sorry

end ratio_a_b_l423_42346


namespace diagonals_of_polygon_l423_42397

theorem diagonals_of_polygon (f : ℕ → ℕ) (k : ℕ) (h_k : k ≥ 3) : f (k + 1) = f k + (k - 1) :=
sorry

end diagonals_of_polygon_l423_42397


namespace union_M_N_is_real_l423_42312

def M : Set ℝ := {x | x^2 + x > 0}
def N : Set ℝ := {x | |x| > 2}

theorem union_M_N_is_real : M ∪ N = Set.univ := by
  sorry

end union_M_N_is_real_l423_42312


namespace gcd_of_polynomial_and_multiple_l423_42311

-- Definitions based on given conditions
def multiple_of (a b : ℕ) : Prop := ∃ k, a = k * b

-- The main statement of the problem
theorem gcd_of_polynomial_and_multiple (y : ℕ) (h : multiple_of y 56790) :
  Nat.gcd ((3 * y + 2) * (5 * y + 3) * (11 * y + 7) * (y + 17)) y = 714 :=
sorry

end gcd_of_polynomial_and_multiple_l423_42311


namespace total_population_l423_42349

theorem total_population (b g t : ℕ) (h₁ : b = 6 * g) (h₂ : g = 5 * t) :
  b + g + t = 36 * t :=
by
  sorry

end total_population_l423_42349


namespace find_height_of_pyramid_l423_42313

noncomputable def volume (B h : ℝ) : ℝ := (1/3) * B * h
noncomputable def area_of_isosceles_right_triangle (leg : ℝ) : ℝ := (1/2) * leg * leg

theorem find_height_of_pyramid (leg : ℝ) (h : ℝ) (V : ℝ) (B : ℝ)
  (Hleg : leg = 3)
  (Hvol : V = 6)
  (Hbase : B = area_of_isosceles_right_triangle leg)
  (Hvol_eq : V = volume B h) :
  h = 4 :=
by
  sorry

end find_height_of_pyramid_l423_42313


namespace find_digit_A_l423_42327

def sum_of_digits_divisible_by_3 (A : ℕ) : Prop :=
  (2 + A + 3) % 3 = 0

theorem find_digit_A (A : ℕ) (hA : sum_of_digits_divisible_by_3 A) : A = 1 ∨ A = 4 :=
  sorry

end find_digit_A_l423_42327


namespace suraj_next_innings_runs_l423_42325

variable (A R : ℕ)

def suraj_average_eq (A : ℕ) : Prop :=
  A + 8 = 128

def total_runs_eq (A R : ℕ) : Prop :=
  9 * A + R = 10 * 128

theorem suraj_next_innings_runs :
  ∃ A : ℕ, suraj_average_eq A ∧ ∃ R : ℕ, total_runs_eq A R ∧ R = 200 := 
by
  sorry

end suraj_next_innings_runs_l423_42325


namespace cow_problem_l423_42324

noncomputable def problem_statement : Prop :=
  ∃ (F M : ℕ), F + M = 300 ∧
               (∃ S H : ℕ, S = 1/2 * F ∧ H = 1/2 * M ∧ S = H + 50) ∧
               F = 2 * M

theorem cow_problem : problem_statement :=
sorry

end cow_problem_l423_42324


namespace chord_line_equation_l423_42378

/-- 
  Given the parabola y^2 = 4x and a chord AB 
  that exactly bisects at point P(1,1), prove 
  that the equation of the line on which chord AB lies is 2x - y - 1 = 0.
-/
theorem chord_line_equation (x y : ℝ) 
  (hx : y^2 = 4 * x)
  (bisect : ∃ A B : ℝ × ℝ, 
             (A.1^2 = 4 * A.2) ∧ (B.1^2 = 4 * B.2) ∧
             (A.1 + B.1 = 2 * 1) ∧ (A.2 + B.2 = 2 * 1)) :
  2 * x - y - 1 = 0 := sorry

end chord_line_equation_l423_42378


namespace solve_system_l423_42353

theorem solve_system : ∃ s t : ℝ, (11 * s + 7 * t = 240) ∧ (s = 1 / 2 * t + 3) ∧ (t = 414 / 25) :=
by
  sorry

end solve_system_l423_42353


namespace length_of_plot_correct_l423_42307

noncomputable def length_of_plot (b : ℕ) : ℕ := b + 30

theorem length_of_plot_correct (b : ℕ) (cost_per_meter total_cost : ℝ) 
    (h1 : length_of_plot b = b + 30)
    (h2 : cost_per_meter = 26.50)
    (h3 : total_cost = 5300)
    (h4 : 2 * (b + (b + 30)) * cost_per_meter = total_cost) :
    length_of_plot 35 = 65 :=
by
  sorry

end length_of_plot_correct_l423_42307


namespace distance_covered_at_40_kmph_l423_42396

theorem distance_covered_at_40_kmph (x : ℝ) : 
  (x / 40 + (250 - x) / 60 = 5.4) → (x = 148) :=
by
  intro h
  sorry

end distance_covered_at_40_kmph_l423_42396


namespace check_double_root_statements_l423_42354

-- Condition Definitions
def is_double_root_equation (a b c : ℝ) : Prop :=
  ∃ r : ℝ, a * r^2 + b * r + c = 0 ∧ a * (2 * r)^2 + b * (2 * r) + c = 0

-- Statement ①
def statement_1 : Prop := ¬is_double_root_equation 1 2 (-8)

-- Statement ②
def statement_2 : Prop := is_double_root_equation 1 (-3) 2

-- Statement ③
def statement_3 (m n : ℝ) : Prop := 
  (∃ r : ℝ, (r - 2) * (m * r + n) = 0 ∧ (m * (2 * r) + n = 0) ∧ r = 2) → 4 * m^2 + 5 * m * n + n^2 = 0

-- Statement ④
def statement_4 (p q : ℝ) : Prop := 
  (p * q = 2 → is_double_root_equation p 3 q)

-- Main proof problem statement
theorem check_double_root_statements (m n p q : ℝ) : 
  statement_1 ∧ statement_2 ∧ statement_3 m n ∧ statement_4 p q :=
by
  sorry

end check_double_root_statements_l423_42354


namespace final_coordinates_l423_42394

-- Definitions for the given conditions
def initial_point : ℝ × ℝ := (-2, 6)

def reflect_x_axis (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

def translate_up (p : ℝ × ℝ) (d : ℝ) : ℝ × ℝ := (p.1, p.2 + d)

-- The final proof statement
theorem final_coordinates :
  let S_reflected := reflect_x_axis initial_point
  let S_translated := translate_up S_reflected 10
  S_translated = (-2, 4) :=
by
  sorry

end final_coordinates_l423_42394


namespace sum_of_reciprocal_transformed_roots_l423_42343

-- Define the polynomial f
def f (x : ℝ) : ℝ := 15 * x^3 - 35 * x^2 + 20 * x - 2

-- Define the condition that the roots are distinct real numbers between 0 and 1
def is_root (f : ℝ → ℝ) (x : ℝ) : Prop := f x = 0
def roots_between_0_and_1 (a b c : ℝ) : Prop := 
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
  0 < a ∧ a < 1 ∧ 
  0 < b ∧ b < 1 ∧ 
  0 < c ∧ c < 1 ∧
  is_root f a ∧ is_root f b ∧ is_root f c

-- The theorem representing the proof problem
theorem sum_of_reciprocal_transformed_roots (a b c : ℝ) 
  (h : roots_between_0_and_1 a b c) :
  (1/(1-a)) + (1/(1-b)) + (1/(1-c)) = 2/3 :=
by
  sorry

end sum_of_reciprocal_transformed_roots_l423_42343


namespace find_compounding_frequency_l423_42345

-- Lean statement defining the problem conditions and the correct answer

theorem find_compounding_frequency (P A : ℝ) (r t : ℝ) (hP : P = 12000) (hA : A = 13230) 
(hri : r = 0.10) (ht : t = 1) 
: ∃ (n : ℕ), A = P * (1 + r / n) ^ (n * t) ∧ n = 2 := 
by
  -- Definitions from the conditions
  have hP := hP
  have hA := hA
  have hr := hri
  have ht := ht
  
  -- Substitute known values
  use 2
  -- Show that the statement holds with n = 2
  sorry

end find_compounding_frequency_l423_42345


namespace find_number_of_cats_l423_42308

theorem find_number_of_cats (dogs ferrets cats total_shoes shoes_per_animal : ℕ) 
  (h_dogs : dogs = 3)
  (h_ferrets : ferrets = 1)
  (h_total_shoes : total_shoes = 24)
  (h_shoes_per_animal : shoes_per_animal = 4) :
  cats = (total_shoes - (dogs + ferrets) * shoes_per_animal) / shoes_per_animal := by
  sorry

end find_number_of_cats_l423_42308


namespace percentage_is_36_point_4_l423_42328

def part : ℝ := 318.65
def whole : ℝ := 875.3

theorem percentage_is_36_point_4 : (part / whole) * 100 = 36.4 := 
by sorry

end percentage_is_36_point_4_l423_42328


namespace probability_not_snowing_l423_42340

theorem probability_not_snowing (P_snowing : ℚ) (h : P_snowing = 2/7) :
  (1 - P_snowing) = 5/7 :=
sorry

end probability_not_snowing_l423_42340


namespace neg_prop_l423_42372

-- Definition of the proposition to be negated
def prop (x : ℝ) : Prop := x^2 + 2 * x + 5 = 0

-- Negation of the proposition
theorem neg_prop : ¬ (∃ x : ℝ, prop x) ↔ ∀ x : ℝ, ¬ prop x :=
by
  sorry

end neg_prop_l423_42372


namespace f_satisfies_equation_l423_42381

noncomputable def f (x : ℝ) : ℝ := (20 / 3) * x * (Real.sqrt (1 - x^2))

theorem f_satisfies_equation (f : ℝ → ℝ) :
  (∀ x ∈ Set.Icc (-Real.pi / 4) (Real.pi / 4), 2 * f (Real.sin x * -1) + 3 * f (Real.sin x) = 4 * Real.sin x * Real.cos x) →
  (∀ x ∈ Set.Icc (-Real.sqrt 2 / 2) (Real.sqrt 2 / 2), f x = (20 / 3) * x * (Real.sqrt (1 - x^2))) :=
by
  intro h
  sorry

end f_satisfies_equation_l423_42381


namespace monomial_properties_l423_42342

theorem monomial_properties (a b : ℕ) (h : a = 2 ∧ b = 1) : 
  (2 * a ^ 2 * b = 2 * (a ^ 2) * b) ∧ (2 = 2) ∧ ((2 + 1) = 3) :=
by
  sorry

end monomial_properties_l423_42342


namespace not_solvable_det_three_times_l423_42335

theorem not_solvable_det_three_times (a b c d : ℝ) (h : a * d - b * c = 5) :
  ¬∃ (x : ℝ), (3 * a + 1) * (3 * d + 1) - (3 * b + 1) * (3 * c + 1) = x :=
by {
  -- This is where the proof would go, but the problem states that it's not solvable with the given information.
  sorry
}

end not_solvable_det_three_times_l423_42335


namespace smallest_positive_integer_exists_l423_42388

theorem smallest_positive_integer_exists
    (x : ℕ) :
    (x % 7 = 2) ∧
    (x % 4 = 3) ∧
    (x % 6 = 1) →
    x = 135 :=
by
    sorry

end smallest_positive_integer_exists_l423_42388


namespace num_integer_solutions_l423_42341

def circle_center := (3, 3)
def circle_radius := 10

theorem num_integer_solutions :
  (∃ f : ℕ, f = 15) :=
sorry

end num_integer_solutions_l423_42341


namespace sandy_total_sums_attempted_l423_42306

theorem sandy_total_sums_attempted (C I : ℕ) 
  (marks_per_correct_sum : ℕ := 3) 
  (marks_lost_per_incorrect_sum : ℕ := 2) 
  (total_marks : ℕ := 45) 
  (correct_sums : ℕ := 21) 
  (H : 3 * correct_sums - 2 * I = total_marks) 
  : C + I = 30 := 
by 
  sorry

end sandy_total_sums_attempted_l423_42306


namespace sum_of_reciprocals_l423_42305

theorem sum_of_reciprocals (x y : ℝ) (h1 : x + y = 14) (h2 : x * y = 45) : 
  1/x + 1/y = 14/45 := 
sorry

end sum_of_reciprocals_l423_42305


namespace find_m_l423_42334

theorem find_m (x p q m : ℝ) 
    (h1 : 4 * p^2 + 9 * q^2 = 2) 
    (h2 : (1/2) * x + 3 * p * q = 1) 
    (h3 : ∀ x, x^2 + 2 * m * x - 3 * m + 1 ≥ 1) :
    m = -3 ∨ m = 1 :=
sorry

end find_m_l423_42334


namespace polynomial_has_real_root_l423_42367

theorem polynomial_has_real_root (b : ℝ) : ∃ x : ℝ, (x^4 + b * x^3 + 2 * x^2 + b * x - 2 = 0) := sorry

end polynomial_has_real_root_l423_42367


namespace bus_ride_cost_l423_42322

variable (cost_bus cost_train : ℝ)

-- Condition 1: cost_train = cost_bus + 2.35
#check (cost_train = cost_bus + 2.35)

-- Condition 2: cost_bus + cost_train = 9.85
#check (cost_bus + cost_train = 9.85)

theorem bus_ride_cost :
  (∃ (cost_bus cost_train : ℝ),
    cost_train = cost_bus + 2.35 ∧
    cost_bus + cost_train = 9.85) →
  cost_bus = 3.75 :=
sorry

end bus_ride_cost_l423_42322


namespace kaleb_balance_l423_42369

theorem kaleb_balance (springEarnings : ℕ) (summerEarnings : ℕ) (suppliesCost : ℕ) (totalBalance : ℕ)
  (h1 : springEarnings = 4)
  (h2 : summerEarnings = 50)
  (h3 : suppliesCost = 4)
  (h4 : totalBalance = (springEarnings + summerEarnings) - suppliesCost) : totalBalance = 50 := by
  sorry

end kaleb_balance_l423_42369


namespace range_of_m_l423_42387

variables (f : ℝ → ℝ) (m : ℝ)

-- Assume f is a decreasing function
def is_decreasing (f : ℝ → ℝ) : Prop := ∀ x y : ℝ, x < y → f x > f y

-- Assume f is an odd function
def is_odd (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x

-- Theorem stating the main condition and the implication
theorem range_of_m (h_decreasing : is_decreasing f) (h_odd : is_odd f) (h_condition : f (m - 1) + f (2 * m - 1) > 0) : m > 2 / 3 :=
sorry

end range_of_m_l423_42387


namespace number_solution_l423_42316

variable (a : ℝ) (x : ℝ)

theorem number_solution :
  (a^(-x) + 25^(-2*x) + 5^(-4*x) = 11) ∧ (x = 0.25) → a = 625 / 7890481 :=
by 
  sorry

end number_solution_l423_42316


namespace car_rental_cost_l423_42301

def daily_rental_rate : ℝ := 29
def per_mile_charge : ℝ := 0.08
def rental_duration : ℕ := 1
def distance_driven : ℝ := 214.0

theorem car_rental_cost : 
  (daily_rental_rate * rental_duration + per_mile_charge * distance_driven) = 46.12 := 
by 
  sorry

end car_rental_cost_l423_42301


namespace symmetry_probability_l423_42364

-- Define the setting of the problem
def grid_points : ℕ := 121
def grid_size : ℕ := 11
def center_point : (ℕ × ℕ) := (6, 6)
def total_points : ℕ := grid_points - 1
def symmetric_lines : ℕ := 4
def points_per_line : ℕ := 10
def total_symmetric_points : ℕ := symmetric_lines * points_per_line
def probability : ℚ := total_symmetric_points / total_points

-- Theorem statement
theorem symmetry_probability 
  (hp: grid_points = 121) 
  (hs: grid_size = 11) 
  (hc: center_point = (6, 6))
  (htp: total_points = 120)
  (hsl: symmetric_lines = 4)
  (hpl: points_per_line = 10)
  (htsp: total_symmetric_points = 40)
  (hp: probability = 1 / 3) : 
  probability = 1 / 3 :=
by 
  sorry

end symmetry_probability_l423_42364


namespace largest_a1_l423_42314

theorem largest_a1
  (a : ℕ+ → ℝ)
  (h_pos : ∀ n, 0 < a n)
  (h_eq : ∀ n, (2 * a (n + 1) - a n) * (a (n + 1) * a n - 1) = 0)
  (h_initial : a 1 = a 10) :
  ∃ (max_a1 : ℝ), max_a1 = 16 ∧ ∀ x, x = a 1 → x ≤ 16 :=
by
  sorry

end largest_a1_l423_42314


namespace initially_calculated_average_height_l423_42344

/-- Suppose the average height of 20 students was initially calculated incorrectly. Later, it was found that one student's height 
was incorrectly recorded as 151 cm instead of 136 cm. Given the actual average height of the students is 174.25 cm, prove that the 
initially calculated average height was 173.5 cm. -/
theorem initially_calculated_average_height
  (initial_avg actual_avg : ℝ)
  (num_students : ℕ)
  (incorrect_height correct_height : ℝ)
  (h_avg : actual_avg = 174.25)
  (h_students : num_students = 20)
  (h_incorrect : incorrect_height = 151)
  (h_correct : correct_height = 136)
  (h_total_actual : num_students * actual_avg = num_students * initial_avg + incorrect_height - correct_height) :
  initial_avg = 173.5 :=
by
  sorry

end initially_calculated_average_height_l423_42344


namespace midpoint_of_hyperbola_l423_42399

theorem midpoint_of_hyperbola :
  ∃ (A B : ℝ × ℝ),
    (A.1 ^ 2 - (A.2 ^ 2) / 9 = 1) ∧
    (B.1 ^ 2 - (B.2 ^ 2) / 9 = 1) ∧
    (∃ (M : ℝ × ℝ), (M = (-1, -4)) ∧ 
      (A.1 + B.1) / 2 = M.1 ∧ (A.2 + B.2) / 2 = M.2) ∧
    ¬(∃ (A B : ℝ × ℝ), (A.1 ^ 2 - (A.2 ^ 2) / 9 = 1) ∧ 
      (B.1 ^ 2 - (B.2 ^ 2) / 9 = 1) ∧ 
      ((A.1 + B.1) / 2 = 1 ∧ (A.2 + B.2) / 2 = 1) ∨ 
      ((A.1 + B.1) / 2 = -1 ∧ (A.2 + B.2) / 2 = 2) ∨ 
      ((A.1 + B.1) / 2 = 1 ∧ (A.2 + B.2) / 2 = 3)) :=
sorry

end midpoint_of_hyperbola_l423_42399


namespace masking_tape_needed_l423_42362

def wall1_width : ℝ := 4
def wall1_count : ℕ := 2
def wall2_width : ℝ := 6
def wall2_count : ℕ := 2
def door_width : ℝ := 2
def door_count : ℕ := 1
def window_width : ℝ := 1.5
def window_count : ℕ := 2

def total_width_of_walls : ℝ := (wall1_count * wall1_width) + (wall2_count * wall2_width)
def total_width_of_door_and_windows : ℝ := (door_count * door_width) + (window_count * window_width)

theorem masking_tape_needed : total_width_of_walls - total_width_of_door_and_windows = 15 := by
  sorry

end masking_tape_needed_l423_42362


namespace smallest_period_sum_l423_42370

noncomputable def smallest_positive_period (f : ℝ → ℝ) (g : ℝ → ℝ): ℝ → ℝ :=
λ x => f x + g x

theorem smallest_period_sum
  (f g : ℝ → ℝ)
  (m n : ℕ)
  (hf : ∀ x, f (x + m) = f x)
  (hg : ∀ x, g (x + n) = g x)
  (hm : m > 1)
  (hn : n > 1)
  (hgcd : Nat.gcd m n = 1)
  : ∃ T, T > 0 ∧ (∀ x, smallest_positive_period f g (x + T) = smallest_positive_period f g x) ∧ T = m * n := by
  sorry

end smallest_period_sum_l423_42370


namespace evaluate_expression_l423_42356

theorem evaluate_expression (x y z : ℤ) (h1 : x = -2) (h2 : y = -4) (h3 : z = 3) :
  (5 * (x - y)^2 - x * z^2) / (z - y) = 38 / 7 := by
  sorry

end evaluate_expression_l423_42356


namespace absolute_value_and_power_sum_l423_42368

theorem absolute_value_and_power_sum :
  |(-4 : ℤ)| + (3 - Real.pi)^0 = 5 := by
  sorry

end absolute_value_and_power_sum_l423_42368


namespace final_retail_price_l423_42329

theorem final_retail_price (wholesale_price markup_percentage discount_percentage desired_profit_percentage : ℝ)
  (h_wholesale : wholesale_price = 90)
  (h_markup : markup_percentage = 1)
  (h_discount : discount_percentage = 0.2)
  (h_desired_profit : desired_profit_percentage = 0.6) :
  let initial_retail_price := wholesale_price + (wholesale_price * markup_percentage)
  let discount_amount := initial_retail_price * discount_percentage
  let final_retail_price := initial_retail_price - discount_amount
  final_retail_price = 144 ∧ final_retail_price = wholesale_price + (wholesale_price * desired_profit_percentage) := by
 sorry

end final_retail_price_l423_42329


namespace chemistry_more_than_physics_l423_42338

noncomputable def M : ℕ := sorry
noncomputable def P : ℕ := sorry
noncomputable def C : ℕ := sorry
noncomputable def x : ℕ := sorry

theorem chemistry_more_than_physics :
  M + P = 20 ∧ C = P + x ∧ (M + C) / 2 = 20 → x = 20 :=
by
  sorry

end chemistry_more_than_physics_l423_42338


namespace find_a_l423_42321

theorem find_a
  (f : ℝ → ℝ)
  (h₁ : ∀ x, f x = 3 * Real.sin (2 * x - Real.pi / 3))
  (a : ℝ)
  (h₂ : 0 < a)
  (h₃ : a < Real.pi / 2)
  (h₄ : ∀ x, f (x + a) = f (-x + a)) :
  a = 5 * Real.pi / 12 :=
sorry

end find_a_l423_42321


namespace intersect_complement_eq_l423_42392

def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7}
def A : Set ℕ := {3, 4, 5}
def B : Set ℕ := {1, 3, 6}
def comp_B : Set ℕ := U \ B

theorem intersect_complement_eq :
  A ∩ comp_B = {4, 5} := by
  sorry

end intersect_complement_eq_l423_42392


namespace two_digit_number_is_24_l423_42393

-- Definitions from the problem conditions
def is_two_digit_number (n : ℕ) := n ≥ 10 ∧ n < 100

def tens_digit (n : ℕ) := n / 10

def ones_digit (n : ℕ) := n % 10

def condition_2 (n : ℕ) := tens_digit n = ones_digit n - 2

def condition_3 (n : ℕ) := 3 * tens_digit n * ones_digit n = n

-- The proof problem statement
theorem two_digit_number_is_24 (n : ℕ) (h1 : is_two_digit_number n)
  (h2 : condition_2 n) (h3 : condition_3 n) : n = 24 := by
  sorry

end two_digit_number_is_24_l423_42393


namespace prob_at_least_one_hit_correct_prob_exactly_two_hits_correct_l423_42326

-- Define the probabilities that A, B, and C hit the target
def prob_A := 0.7
def prob_B := 0.6
def prob_C := 0.5

-- Define the probabilities that A, B, and C miss the target
def miss_A := 1 - prob_A
def miss_B := 1 - prob_B
def miss_C := 1 - prob_C

-- Probability that no one hits the target
def prob_no_hits := miss_A * miss_B * miss_C

-- Probability that at least one person hits the target
def prob_at_least_one_hit := 1 - prob_no_hits

-- Probabilities for the cases where exactly two people hit the target:
def prob_A_B_hits := prob_A * prob_B * miss_C
def prob_A_C_hits := prob_A * miss_B * prob_C
def prob_B_C_hits := miss_A * prob_B * prob_C

-- Probability that exactly two people hit the target
def prob_exactly_two_hits := prob_A_B_hits + prob_A_C_hits + prob_B_C_hits

-- Theorem statement to prove the probabilities match given conditions
theorem prob_at_least_one_hit_correct : prob_at_least_one_hit = 0.94 := by
  sorry

theorem prob_exactly_two_hits_correct : prob_exactly_two_hits = 0.44 := by
  sorry

end prob_at_least_one_hit_correct_prob_exactly_two_hits_correct_l423_42326


namespace sphere_radius_equals_4_l423_42339

noncomputable def radius_of_sphere
  (sun_parallel : true)
  (meter_stick_height : ℝ)
  (meter_stick_shadow : ℝ)
  (sphere_shadow_distance : ℝ) : ℝ :=
if h : meter_stick_height / meter_stick_shadow = sphere_shadow_distance / 16 then
  4
else
  sorry

theorem sphere_radius_equals_4 
  (sun_parallel : true = true)
  (meter_stick_height : ℝ := 1)
  (meter_stick_shadow : ℝ := 4)
  (sphere_shadow_distance : ℝ := 16) : 
  radius_of_sphere sun_parallel meter_stick_height meter_stick_shadow sphere_shadow_distance = 4 :=
by
  simp [radius_of_sphere]
  sorry

end sphere_radius_equals_4_l423_42339


namespace lunch_special_cost_l423_42365

theorem lunch_special_cost (total_bill : ℕ) (num_people : ℕ) (cost_per_lunch_special : ℕ)
  (h1 : total_bill = 24) 
  (h2 : num_people = 3) 
  (h3 : cost_per_lunch_special = total_bill / num_people) : 
  cost_per_lunch_special = 8 := 
by
  sorry

end lunch_special_cost_l423_42365


namespace M_intersect_N_l423_42384

def M : Set ℝ := {x | 1 + x > 0}
def N : Set ℝ := {x | x < 1}

theorem M_intersect_N : M ∩ N = {x | -1 < x ∧ x < 1} := 
by
  sorry

end M_intersect_N_l423_42384


namespace zoo_gorillas_sent_6_l423_42390

theorem zoo_gorillas_sent_6 (G : ℕ) : 
  let initial_animals := 68
  let after_sending_gorillas := initial_animals - G
  let after_adopting_hippopotamus := after_sending_gorillas + 1
  let after_taking_rhinos := after_adopting_hippopotamus + 3
  let after_birth_lion_cubs := after_taking_rhinos + 8
  let after_adding_meerkats := after_birth_lion_cubs + (2 * 8)
  let final_animals := 90
  after_adding_meerkats = final_animals → G = 6 := 
by
  intros
  let initial_animals := 68
  let after_sending_gorillas := initial_animals - G
  let after_adopting_hippopotamus := after_sending_gorillas + 1
  let after_taking_rhinos := after_adopting_hippopotamus + 3
  let after_birth_lion_cubs := after_taking_rhinos + 8
  let after_adding_meerkats := after_birth_lion_cubs + (2 * 8)
  let final_animals := 90
  sorry

end zoo_gorillas_sent_6_l423_42390


namespace greatest_value_x_l423_42333

theorem greatest_value_x (x: ℤ) : 
  (∃ k: ℤ, (x^2 - 5 * x + 14) = k * (x - 4)) → x ≤ 14 :=
sorry

end greatest_value_x_l423_42333


namespace Billy_Reads_3_Books_l423_42347

theorem Billy_Reads_3_Books 
    (weekend_days : ℕ) 
    (hours_per_day : ℕ) 
    (reading_percentage : ℕ) 
    (pages_per_hour : ℕ) 
    (pages_per_book : ℕ) : 
    (weekend_days = 2) ∧ 
    (hours_per_day = 8) ∧ 
    (reading_percentage = 25) ∧ 
    (pages_per_hour = 60) ∧ 
    (pages_per_book = 80) → 
    ((weekend_days * hours_per_day * reading_percentage / 100 * pages_per_hour) / pages_per_book = 3) :=
by
  intros
  sorry

end Billy_Reads_3_Books_l423_42347


namespace conditional_probability_age_30_40_female_l423_42380

noncomputable def total_people : ℕ := 350
noncomputable def total_females : ℕ := 180
noncomputable def females_30_40 : ℕ := 50

theorem conditional_probability_age_30_40_female :
  (females_30_40 : ℚ) / total_females = 5 / 18 :=
by
  sorry

end conditional_probability_age_30_40_female_l423_42380


namespace q_investment_l423_42360

theorem q_investment (p_investment : ℕ) (ratio_pq : ℕ × ℕ) (profit_ratio : ℕ × ℕ) (hp : p_investment = 12000) (hpr : ratio_pq = (3, 5)) : 
  (∃ q_investment, q_investment = 20000) :=
  sorry

end q_investment_l423_42360


namespace does_not_pass_first_quadrant_l423_42391

def linear_function (x : ℝ) : ℝ := -3 * x - 2

def in_first_quadrant (x y : ℝ) : Prop := x > 0 ∧ y > 0
def in_second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0
def in_third_quadrant (x y : ℝ) : Prop := x < 0 ∧ y < 0
def in_fourth_quadrant (x y : ℝ) : Prop := x > 0 ∧ y < 0

theorem does_not_pass_first_quadrant : ∀ (x : ℝ), ¬ in_first_quadrant x (linear_function x) := 
sorry

end does_not_pass_first_quadrant_l423_42391


namespace polynomial_pattern_1_polynomial_pattern_2_polynomial_calculation_polynomial_factorization_l423_42302

theorem polynomial_pattern_1 (a b : ℝ) : (a + b) * (a ^ 2 - a * b + b ^ 2) = a ^ 3 + b ^ 3 :=
sorry

theorem polynomial_pattern_2 (a b : ℝ) : (a - b) * (a ^ 2 + a * b + b ^ 2) = a ^ 3 - b ^ 3 :=
sorry

theorem polynomial_calculation (a b : ℝ) : (a + 2 * b) * (a ^ 2 - 2 * a * b + 4 * b ^ 2) = a ^ 3 + 8 * b ^ 3 :=
sorry

theorem polynomial_factorization (a : ℝ) : a ^ 3 - 8 = (a - 2) * (a ^ 2 + 2 * a + 4) :=
sorry

end polynomial_pattern_1_polynomial_pattern_2_polynomial_calculation_polynomial_factorization_l423_42302


namespace line_intercepts_l423_42398

theorem line_intercepts (x y : ℝ) (P : ℝ × ℝ) (h1 : P = (1, 4)) (h2 : ∃ k : ℝ, (x + y = k ∨ 4 * x - y = 0) ∧ 
  ∃ intercepts_p : ℝ × ℝ, intercepts_p = (k / 2, k / 2)) :
  ∃ k : ℝ, (x + y - k = 0 ∧ k = 5) ∨ (4 * x - y = 0) :=
sorry

end line_intercepts_l423_42398


namespace find_a_l423_42355

variable (a : ℕ) (N : ℕ)
variable (h1 : Nat.gcd (2 * a + 1) (2 * a + 2) = 1) 
variable (h2 : Nat.gcd (2 * a + 1) (2 * a + 3) = 1)
variable (h3 : Nat.gcd (2 * a + 2) (2 * a + 3) = 2)
variable (hN : N = Nat.lcm (2 * a + 1) (Nat.lcm (2 * a + 2) (2 * a + 3)))
variable (hDiv : (2 * a + 4) ∣ N)

theorem find_a (h_pos : a > 0) : a = 1 :=
by
  -- Lean proof code will go here
  sorry

end find_a_l423_42355


namespace parallel_lines_a_value_l423_42318

theorem parallel_lines_a_value 
    (a : ℝ) 
    (l₁ : ∀ x y : ℝ, 2 * x + y - 1 = 0) 
    (l₂ : ∀ x y : ℝ, (a - 1) * x + 3 * y - 2 = 0) 
    (h_parallel : ∀ x y : ℝ, 2 / (a - 1) = 1 / 3) : 
    a = 7 := 
    sorry

end parallel_lines_a_value_l423_42318


namespace B_work_days_l423_42319

theorem B_work_days (a b : ℝ) (h1 : a + b = 1/4) (h2 : a = 1/14) : 1 / b = 5.6 :=
by
  sorry

end B_work_days_l423_42319


namespace find_last_number_2_l423_42336

theorem find_last_number_2 (A B C D : ℤ) 
  (h1 : A + B + C = 18)
  (h2 : B + C + D = 9)
  (h3 : A + D = 13) : 
  D = 2 := 
sorry

end find_last_number_2_l423_42336


namespace bus_probability_l423_42337

/-- A bus arrives randomly between 3:00 and 4:00, waits for 15 minutes, and then leaves. 
Sarah also arrives randomly between 3:00 and 4:00. Prove the probability that the bus 
will be there when Sarah arrives is 4275/7200. -/
theorem bus_probability : (4275 : ℚ) / 7200 = (4275 / 7200) :=
by 
  sorry

end bus_probability_l423_42337


namespace janice_items_l423_42383

theorem janice_items : 
  ∃ a b c : ℕ, 
    a + b + c = 60 ∧ 
    15 * a + 400 * b + 500 * c = 6000 ∧ 
    a = 50 := 
by 
  sorry

end janice_items_l423_42383


namespace equivalent_proof_l423_42376

theorem equivalent_proof :
  let a := 4
  let b := Real.sqrt 17 - a
  b^2020 * (a + Real.sqrt 17)^2021 = Real.sqrt 17 + 4 :=
by
  let a := 4
  let b := Real.sqrt 17 - a
  sorry

end equivalent_proof_l423_42376


namespace range_of_m_l423_42361

theorem range_of_m (m : ℝ) (x : ℝ) 
  (h1 : ∀ x : ℝ, -1 < x ∧ x < 4 → x > 2 * m^2 - 3)
  (h2 : ¬ (∀ x : ℝ, x > 2 * m^2 - 3 → -1 < x ∧ x < 4))
  :
  -1 ≤ m ∧ m ≤ 1 :=
sorry

end range_of_m_l423_42361


namespace smallest_z_in_arithmetic_and_geometric_progression_l423_42375

theorem smallest_z_in_arithmetic_and_geometric_progression :
  ∃ x y z : ℤ, x < y ∧ y < z ∧ (2 * y = x + z) ∧ (z^2 = x * y) ∧ z = -2 :=
by
  sorry

end smallest_z_in_arithmetic_and_geometric_progression_l423_42375


namespace smallest_n_l423_42315

theorem smallest_n (n : ℕ) (h1 : ∃ k : ℕ, 3 * n = k ^ 2) (h2 : ∃ m : ℕ, 5 * n = m ^ 5) : n = 151875 := sorry

end smallest_n_l423_42315


namespace microorganism_half_filled_time_l423_42348

theorem microorganism_half_filled_time :
  (∀ x, 2^x = 2^9 ↔ x = 9) :=
by
  sorry

end microorganism_half_filled_time_l423_42348


namespace quadratic_interlaced_roots_l423_42304

theorem quadratic_interlaced_roots
  (p1 p2 q1 q2 : ℝ)
  (h : (q1 - q2)^2 + (p1 - p2) * (p1 * q2 - p2 * q1) < 0) :
  ∃ (r1 r2 s1 s2 : ℝ),
    (r1^2 + p1 * r1 + q1 = 0) ∧
    (r2^2 + p1 * r2 + q1 = 0) ∧
    (s1^2 + p2 * s1 + q2 = 0) ∧
    (s2^2 + p2 * s2 + q2 = 0) ∧
    (r1 < s1 ∧ s1 < r2 ∨ s1 < r1 ∧ r1 < s2) :=
sorry

end quadratic_interlaced_roots_l423_42304


namespace equations_not_equivalent_l423_42377

theorem equations_not_equivalent :
  (∀ x, (2 * (x - 10) / (x^2 - 13 * x + 30) = 1 ↔ x = 5)) ∧ 
  (∃ x, x ≠ 5 ∧ (x^2 - 15 * x + 50 = 0)) :=
sorry

end equations_not_equivalent_l423_42377


namespace multiple_of_four_l423_42359

open BigOperators

theorem multiple_of_four (n : ℕ) (x y z : Fin n → ℤ)
  (hx : ∀ i, x i = 1 ∨ x i = -1)
  (hy : ∀ i, y i = 1 ∨ y i = -1)
  (hz : ∀ i, z i = 1 ∨ z i = -1)
  (hxy : ∑ i, x i * y i = 0)
  (hxz : ∑ i, x i * z i = 0)
  (hyz : ∑ i, y i * z i = 0) :
  (n % 4 = 0) :=
sorry

end multiple_of_four_l423_42359


namespace carrots_chloe_l423_42382

theorem carrots_chloe (c_i c_t c_p : ℕ) (H1 : c_i = 48) (H2 : c_t = 45) (H3 : c_p = 42) : 
  c_i - c_t + c_p = 45 := by
  sorry

end carrots_chloe_l423_42382


namespace percentage_of_a_is_4b_l423_42371

variable (a b : ℝ)

theorem percentage_of_a_is_4b (h : a = 1.2 * b) : 4 * b = (10 / 3) * a := 
by 
    sorry

end percentage_of_a_is_4b_l423_42371


namespace xy_yx_eq_zy_yz_eq_xz_zx_l423_42366

theorem xy_yx_eq_zy_yz_eq_xz_zx 
  (x y z : ℝ) 
  (h : x * (y + z - x) / x = y * (z + x - y) / y ∧ y * (z + x - y) / y = z * (x + y - z) / z): 
  x ^ y * y ^ x = z ^ y * y ^ z ∧ z ^ y * y ^ z = x ^ z * z ^ x :=
by
  sorry

end xy_yx_eq_zy_yz_eq_xz_zx_l423_42366


namespace question1_question2_l423_42350

-- Define the sets A and B as given in the problem
def A (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2 * m - 1}
def B : Set ℝ := {x | x < -2 ∨ x > 5}

-- Lean statement for (1)
theorem question1 (m : ℝ) : 
  (A m ⊆ B) ↔ (m < 2 ∨ m > 4) :=
by
  sorry

-- Lean statement for (2)
theorem question2 (m : ℝ) : 
  (A m ∩ B = ∅) ↔ (m ≤ 3) :=
by
  sorry

end question1_question2_l423_42350


namespace value_of_f_12_l423_42386

theorem value_of_f_12 (f : ℕ → ℤ) 
  (h1 : f 2 = 5)
  (h2 : f 3 = 7)
  (h3 : ∀ m n : ℕ, 0 < m → 0 < n → f m + f n = f (m * n)) :
  f 12 = 17 :=
by
  sorry

end value_of_f_12_l423_42386


namespace ivan_ivanovich_increase_l423_42363

variable (p v s i : ℝ)
variable (k : ℝ)

-- Conditions
def initial_shares_sum := p + v + s + i = 1
def petya_doubles := 2 * p + v + s + i = 1.3
def vanya_doubles := p + 2 * v + s + i = 1.4
def sergey_triples := p + v + 3 * s + i = 1.2

-- Target statement to be proved
theorem ivan_ivanovich_increase (hp : p = 0.3) (hv : v = 0.4) (hs : s = 0.1)
  (hi : i = 0.2) (k : ℝ) : k * i > 0.75 → k > 3.75 :=
sorry

end ivan_ivanovich_increase_l423_42363


namespace right_triangle_area_l423_42395

theorem right_triangle_area (a b : ℝ) (ha : a = 3) (hb : b = 5) : 
  (1 / 2) * a * b = 7.5 := 
by
  rw [ha, hb]
  sorry

end right_triangle_area_l423_42395


namespace shortest_distance_from_parabola_to_line_l423_42309

open Real

noncomputable def parabola_point (M : ℝ × ℝ) : Prop :=
  M.snd^2 = 6 * M.fst

noncomputable def distance_to_line (M : ℝ × ℝ) (a b c : ℝ) : ℝ :=
  abs (a * M.fst + b * M.snd + c) / sqrt (a^2 + b^2)

theorem shortest_distance_from_parabola_to_line (M : ℝ × ℝ) (h : parabola_point M) :
  distance_to_line M 3 (-4) 12 = 3 :=
by
  sorry

end shortest_distance_from_parabola_to_line_l423_42309


namespace forty_percent_of_n_l423_42320

theorem forty_percent_of_n (N : ℝ) (h : (1/4) * (1/3) * (2/5) * N = 16) : 0.40 * N = 384 :=
by
  sorry

end forty_percent_of_n_l423_42320
