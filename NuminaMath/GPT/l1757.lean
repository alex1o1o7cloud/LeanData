import Mathlib

namespace simplify_and_evaluate_expression_l1757_175737

theorem simplify_and_evaluate_expression (x : ℤ) (h : x = -2) : 
  2 * x * (x - 3) - (x - 2) * (x + 1) = 16 :=
by
  sorry

end simplify_and_evaluate_expression_l1757_175737


namespace sea_horses_count_l1757_175781

theorem sea_horses_count (S P : ℕ) (h1 : 11 * S = 5 * P) (h2 : P = S + 85) : S = 70 :=
by
  sorry

end sea_horses_count_l1757_175781


namespace nine_chapters_compensation_difference_l1757_175765

noncomputable def pig_consumes (x : ℝ) := x
noncomputable def sheep_consumes (x : ℝ) := 2 * x
noncomputable def horse_consumes (x : ℝ) := 4 * x
noncomputable def cow_consumes (x : ℝ) := 8 * x

theorem nine_chapters_compensation_difference :
  ∃ (x : ℝ), 
    cow_consumes x + horse_consumes x + sheep_consumes x + pig_consumes x = 9 ∧
    (horse_consumes x - pig_consumes x) = 9 / 5 :=
by
  sorry

end nine_chapters_compensation_difference_l1757_175765


namespace parabola_equation_l1757_175762

/--
Given a point P (4, -2) on a parabola, prove that the equation of the parabola is either:
1) y^2 = x or
2) x^2 = -8y.
-/
theorem parabola_equation (p : ℝ) (x y : ℝ) (h1 : (4 : ℝ) = 4) (h2 : (-2 : ℝ) = -2) :
  (∃ p : ℝ, y^2 = 2 * p * x ∧ 4 = 4 ∧ y = -2) ∨ (∃ p : ℝ, x^2 = 2 * p * y ∧ 4 = 4 ∧ x = 4) :=
sorry

end parabola_equation_l1757_175762


namespace quadratic_roots_distinct_l1757_175720

theorem quadratic_roots_distinct (m : ℝ) : 
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1^2 + 2*x1 + m = 0 ∧ x2^2 + 2*x2 + m = 0) →
  m < 1 := 
by
  sorry

end quadratic_roots_distinct_l1757_175720


namespace number_of_tables_l1757_175741

theorem number_of_tables (x : ℕ) (h : 2 * (x - 1) + 3 = 65) : x = 32 :=
sorry

end number_of_tables_l1757_175741


namespace exists_infinite_sets_of_positive_integers_l1757_175795

theorem exists_infinite_sets_of_positive_integers (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  ∃ (S : ℕ → ℕ × ℕ × ℕ), ∀ n : ℕ, S n = (x, y, z) ∧ 
  ((x + y + z)^2 + 2*(x + y + z) = 5*(x*y + y*z + z*x)) :=
sorry

end exists_infinite_sets_of_positive_integers_l1757_175795


namespace total_miles_walked_by_group_in_6_days_l1757_175783

-- Conditions translated to Lean definitions
def miles_per_day_group := 3
def additional_miles_per_day := 2
def days_in_week := 6
def total_ladies := 5

-- Question translated to a Lean theorem statement
theorem total_miles_walked_by_group_in_6_days : 
  ∀ (miles_per_day_group additional_miles_per_day days_in_week total_ladies : ℕ),
  (miles_per_day_group * total_ladies * days_in_week) + 
  ((miles_per_day_group * (total_ladies - 1) * days_in_week) + (additional_miles_per_day * days_in_week)) = 120 := 
by
  intros
  sorry

end total_miles_walked_by_group_in_6_days_l1757_175783


namespace smallest_of_5_consecutive_natural_numbers_sum_100_l1757_175764

theorem smallest_of_5_consecutive_natural_numbers_sum_100
  (n : ℕ)
  (h : n + (n + 1) + (n + 2) + (n + 3) + (n + 4) = 100) :
  n = 18 := sorry

end smallest_of_5_consecutive_natural_numbers_sum_100_l1757_175764


namespace minimum_value_function_l1757_175705

theorem minimum_value_function :
  ∀ x : ℝ, x ≥ 0 → (∃ y : ℝ, y = (3 * x^2 + 9 * x + 20) / (7 * (2 + x)) ∧
    (∀ z : ℝ, z ≥ 0 → (3 * z^2 + 9 * z + 20) / (7 * (2 + z)) ≥ y)) ∧
    (∃ x0 : ℝ, x0 = 0 ∧ y = (3 * x0^2 + 9 * x0 + 20) / (7 * (2 + x0)) ∧ y = 10 / 7) :=
by
  sorry

end minimum_value_function_l1757_175705


namespace tiled_board_remainder_l1757_175770

def num_ways_to_tile_9x1 : Nat := -- hypothetical function to calculate the number of ways
  sorry

def N : Nat :=
  num_ways_to_tile_9x1 -- placeholder for N, should be computed using correct formula

theorem tiled_board_remainder : N % 1000 = 561 :=
  sorry

end tiled_board_remainder_l1757_175770


namespace ackermann_3_2_l1757_175727

-- Define the Ackermann function
def ackermann : ℕ → ℕ → ℕ
| 0, n => n + 1
| (m + 1), 0 => ackermann m 1
| (m + 1), (n + 1) => ackermann m (ackermann (m + 1) n)

-- Prove that A(3, 2) = 29
theorem ackermann_3_2 : ackermann 3 2 = 29 := by
  sorry

end ackermann_3_2_l1757_175727


namespace extrema_of_f_l1757_175759

noncomputable def f (x : ℝ) : ℝ := -x^3 + 3 * x + 1

theorem extrema_of_f :
  (∃ x : ℝ, f x = -1 ∧ ∀ y : ℝ, f y ≥ f x) ∧
  (∃ x : ℝ, f x = 3 ∧ ∀ y : ℝ, f y ≤ f x) :=
by
  sorry

end extrema_of_f_l1757_175759


namespace spiders_hired_l1757_175702

theorem spiders_hired (total_workers beavers : ℕ) (h_total : total_workers = 862) (h_beavers : beavers = 318) : (total_workers - beavers) = 544 := by
  sorry

end spiders_hired_l1757_175702


namespace angle_measure_l1757_175755

-- Define the angle in degrees
def angle (x : ℝ) : Prop :=
  180 - x = 3 * (90 - x)

-- Desired proof statement
theorem angle_measure :
  ∀ (x : ℝ), angle x → x = 45 := by
  intros x h
  sorry

end angle_measure_l1757_175755


namespace distance_between_homes_l1757_175756

-- Define the conditions as Lean functions and values
def walking_speed_maxwell : ℝ := 3
def running_speed_brad : ℝ := 5
def distance_traveled_maxwell : ℝ := 15

-- State the theorem
theorem distance_between_homes : 
  ∃ D : ℝ, 
    (15 = walking_speed_maxwell * (distance_traveled_maxwell / walking_speed_maxwell)) ∧ 
    (D - 15 = running_speed_brad * (distance_traveled_maxwell / walking_speed_maxwell)) ∧ 
    D = 40 :=
by 
  sorry

end distance_between_homes_l1757_175756


namespace no_integer_solutions_l1757_175704

theorem no_integer_solutions :
  ¬ ∃ (x y z : ℤ),
    x^6 + x^3 + x^3 * y + y = 147^157 ∧
    x^3 + x^3 * y + y^2 + y + z^9 = 157^147 :=
by
  sorry

end no_integer_solutions_l1757_175704


namespace difference_between_roots_l1757_175788

noncomputable def a : ℝ := 1
noncomputable def b : ℝ := -7
noncomputable def c : ℝ := 11

noncomputable def quadratic_roots (a b c : ℝ) : (ℝ × ℝ) :=
  let discriminant := b ^ 2 - 4 * a * c
  ((-b + Real.sqrt discriminant) / (2 * a), (-b - Real.sqrt discriminant) / (2 * a))

-- Extract the roots from the equation
noncomputable def r1_r2 := quadratic_roots a b c

noncomputable def r1 : ℝ := r1_r2.1
noncomputable def r2 : ℝ := r1_r2.2

-- Theorem statement: the difference between the roots is sqrt(5)
theorem difference_between_roots :
  |r1 - r2| = Real.sqrt 5 :=
  sorry

end difference_between_roots_l1757_175788


namespace johns_salary_before_raise_l1757_175713

variable (x : ℝ)

theorem johns_salary_before_raise (h : x + 0.3333 * x = 80) : x = 60 :=
by
  sorry

end johns_salary_before_raise_l1757_175713


namespace cos_4_arccos_l1757_175716

theorem cos_4_arccos (y : ℝ) (hy1 : y = Real.arccos (2/5)) (hy2 : Real.cos y = 2/5) : 
  Real.cos (4 * y) = -47 / 625 := 
by 
  sorry

end cos_4_arccos_l1757_175716


namespace t_shirt_jersey_price_difference_l1757_175773

theorem t_shirt_jersey_price_difference :
  ∀ (T J : ℝ), (0.9 * T = 192) → (0.9 * J = 34) → (T - J = 175.55) :=
by
  intros T J hT hJ
  sorry

end t_shirt_jersey_price_difference_l1757_175773


namespace find_principal_l1757_175736

theorem find_principal (x y : ℝ) : 
  (2 * x * y / 100 = 400) → 
  (2 * x * y + x * y^2 / 100 = 41000) → 
  x = 4000 := 
by
  sorry

end find_principal_l1757_175736


namespace geometric_sequence_product_of_terms_l1757_175790

theorem geometric_sequence_product_of_terms 
  (a : ℕ → ℝ)
  (r : ℝ)
  (h_geom : ∀ n, a (n + 1) = a n * r)
  (h_a2 : a 2 = 2)
  (h_a6 : a 6 = 8) : 
  a 3 * a 4 * a 5 = 64 := 
by
  sorry

end geometric_sequence_product_of_terms_l1757_175790


namespace necessary_and_sufficient_condition_l1757_175793

open Real

theorem necessary_and_sufficient_condition 
  {x y : ℝ} (p : x > y) (q : x - y + sin (x - y) > 0) : 
  (x > y) ↔ (x - y + sin (x - y) > 0) :=
sorry

end necessary_and_sufficient_condition_l1757_175793


namespace trajectory_is_plane_l1757_175777

/--
Given that the vertical coordinate of a moving point P is always 2, 
prove that the trajectory of the moving point P forms a plane in a 
three-dimensional Cartesian coordinate system.
-/
theorem trajectory_is_plane (P : ℝ × ℝ × ℝ) (hP : ∀ t : ℝ, ∃ x y, P = (x, y, 2)) :
  ∃ a b c d, a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0 ∧ (∀ x y, ∃ z, (a * x + b * y + c * z + d = 0) ∧ z = 2) :=
by
  -- This proof should show that there exist constants a, b, c, and d such that 
  -- the given equation represents a plane and the z-coordinate is always 2.
  sorry

end trajectory_is_plane_l1757_175777


namespace square_of_positive_difference_l1757_175729

theorem square_of_positive_difference {y : ℝ}
  (h : (45 + y) / 2 = 50) :
  (|y - 45|)^2 = 100 :=
by
  sorry

end square_of_positive_difference_l1757_175729


namespace equation_of_circle_l1757_175760

theorem equation_of_circle :
  ∃ (a : ℝ), a < 0 ∧ (∀ (x y : ℝ), (x + 2 * y = 0) → (x + 5)^2 + y^2 = 5) :=
by
  sorry

end equation_of_circle_l1757_175760


namespace collinear_points_value_l1757_175728

/-- 
If the points (2, a, b), (a, 3, b), and (a, b, 4) are collinear, 
then the value of a + b is 7.
-/
theorem collinear_points_value (a b : ℝ) (h_collinear : ∃ l : ℝ → ℝ × ℝ × ℝ, 
  l 0 = (2, a, b) ∧ l 1 = (a, 3, b) ∧ l 2 = (a, b, 4) ∧ 
  ∀ t s : ℝ, l t = l s → t = s) :
  a + b = 7 :=
sorry

end collinear_points_value_l1757_175728


namespace chocolate_bars_in_large_box_l1757_175750

def num_small_boxes : ℕ := 17
def chocolate_bars_per_small_box : ℕ := 26
def total_chocolate_bars : ℕ := 17 * 26

theorem chocolate_bars_in_large_box :
  total_chocolate_bars = 442 :=
by
  sorry

end chocolate_bars_in_large_box_l1757_175750


namespace bill_difference_proof_l1757_175761

variable (a b c : ℝ)

def alice_condition := (25/100) * a = 5
def bob_condition := (20/100) * b = 6
def carol_condition := (10/100) * c = 7

theorem bill_difference_proof (ha : alice_condition a) (hb : bob_condition b) (hc : carol_condition c) :
  max a (max b c) - min a (min b c) = 50 :=
by sorry

end bill_difference_proof_l1757_175761


namespace proof_problem_l1757_175717

-- Conditions: p and q are solutions to the quadratic equation 3x^2 - 5x - 8 = 0
def is_solution (p q : ℝ) : Prop := (3 * p^2 - 5 * p - 8 = 0) ∧ (3 * q^2 - 5 * q - 8 = 0)

-- Question: Compute the value of (3 * p^2 - 3 * q^2) / (p - q) given the conditions
theorem proof_problem (p q : ℝ) (h : is_solution p q) :
  (3 * p^2 - 3 * q^2) * (p - q)⁻¹ = 5 := sorry

end proof_problem_l1757_175717


namespace hillary_descending_rate_correct_l1757_175742

-- Define the conditions in Lean
def base_to_summit := 5000 -- height from base camp to the summit
def departure_time := 6 -- departure time in hours after midnight (6:00)
def summit_time_hillary := 5 -- time taken by Hillary to reach 1000 ft short of the summit
def passing_time := 12 -- time when Hillary and Eddy pass each other (12:00)
def climb_rate_hillary := 800 -- Hillary's climbing rate in ft/hr
def climb_rate_eddy := 500 -- Eddy's climbing rate in ft/hr
def stop_short := 1000 -- distance short of the summit Hillary stops at

-- Define the correct answer based on the conditions
def descending_rate_hillary := 1000 -- Hillary's descending rate in ft/hr

-- Create the theorem to prove Hillary's descending rate
theorem hillary_descending_rate_correct (base_to_summit departure_time summit_time_hillary passing_time climb_rate_hillary climb_rate_eddy stop_short descending_rate_hillary : ℕ) :
  (descending_rate_hillary = 1000) :=
sorry

end hillary_descending_rate_correct_l1757_175742


namespace halfway_between_l1757_175776

theorem halfway_between (a b : ℚ) (h₁ : a = 1/8) (h₂ : b = 1/3) : (a + b) / 2 = 11 / 48 := 
by
  sorry

end halfway_between_l1757_175776


namespace det_B_squared_minus_3B_l1757_175738

open Matrix
open Real

variable {α : Type*} [Fintype α] {n : ℕ}
variable [DecidableEq α]

noncomputable def B : Matrix (Fin 2) (Fin 2) ℝ := ![
  ![2, 4],
  ![1, 3]
]

theorem det_B_squared_minus_3B : det (B * B - 3 • B) = -8 := sorry

end det_B_squared_minus_3B_l1757_175738


namespace exists_positive_integers_abcd_l1757_175708

theorem exists_positive_integers_abcd (m : ℤ) : ∃ (a b c d : ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ (a * b - c * d = m) := by
  sorry

end exists_positive_integers_abcd_l1757_175708


namespace square_roots_of_16_l1757_175724

theorem square_roots_of_16 :
  {y : ℤ | y^2 = 16} = {4, -4} :=
by
  sorry

end square_roots_of_16_l1757_175724


namespace n_m_odd_implies_sum_odd_l1757_175740

theorem n_m_odd_implies_sum_odd {n m : ℤ} (h : Odd (n^2 + m^2)) : Odd (n + m) :=
by
  sorry

end n_m_odd_implies_sum_odd_l1757_175740


namespace raghu_investment_approx_l1757_175700

-- Define the investments
def investments (R : ℝ) : Prop :=
  let Trishul := 0.9 * R
  let Vishal := 0.99 * R
  let Deepak := 1.188 * R
  R + Trishul + Vishal + Deepak = 8578

-- State the theorem to prove that Raghu invested approximately Rs. 2103.96
theorem raghu_investment_approx : 
  ∃ R : ℝ, investments R ∧ abs (R - 2103.96) < 1 :=
by
  sorry

end raghu_investment_approx_l1757_175700


namespace average_sequence_x_l1757_175725

theorem average_sequence_x (x : ℚ) (h : (5050 + x) / 101 = 50 * x) : x = 5050 / 5049 :=
by
  sorry

end average_sequence_x_l1757_175725


namespace inequality_proof_l1757_175758

variable {x y z : ℝ}

theorem inequality_proof 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (hxyz : x + y + z = 1) : 
  (1 / x^2 + x) * (1 / y^2 + y) * (1 / z^2 + z) ≥ (28 / 3)^3 :=
sorry

end inequality_proof_l1757_175758


namespace calculate_div_expression_l1757_175722

variable (x y : ℝ)

theorem calculate_div_expression : (6 * x^3 * y^2) / (-3 * x * y) = -2 * x^2 * y := by
  sorry

end calculate_div_expression_l1757_175722


namespace problem_a_problem_b_l1757_175707

section ProblemA

variable (x : ℝ)

theorem problem_a :
  x ≠ 0 ∧ x ≠ -3/8 ∧ x ≠ 3/7 →
  2 + 5 / (4 * x) - 15 / (4 * x * (8 * x + 3)) = 2 * (7 * x + 1) / (7 * x - 3) →
  x = 9 := by
  sorry

end ProblemA

section ProblemB

variable (x : ℝ)

theorem problem_b :
  x ≠ 0 →
  2 / x + 1 / x^2 - (7 + 10 * x) / (x^2 * (x^2 + 7)) = 2 / (x + 3 / (x + 4 / x)) →
  x = 4 := by
  sorry

end ProblemB

end problem_a_problem_b_l1757_175707


namespace part1_part2_l1757_175744

variables {A B C a b c : ℝ}

-- Condition: sides opposite to angles A, B, and C are a, b, and c respectively and 4b * sin A = sqrt 7 * a
def condition1 : 4 * b * Real.sin A = Real.sqrt 7 * a := sorry

-- Prove that sin B = sqrt 7 / 4
theorem part1 (h : 4 * b * Real.sin A = Real.sqrt 7 * a) :
  Real.sin B = Real.sqrt 7 / 4 := sorry

-- Condition: a, b, and c form an arithmetic sequence with a common difference greater than 0
def condition2 : 2 * b = a + c := sorry

-- Prove that cos A - cos C = sqrt 7 / 2
theorem part2 (h1 : 4 * b * Real.sin A = Real.sqrt 7 * a) (h2 : 2 * b = a + c) :
  Real.cos A - Real.cos C = Real.sqrt 7 / 2 := sorry

end part1_part2_l1757_175744


namespace symmetric_polynomial_evaluation_l1757_175796

theorem symmetric_polynomial_evaluation :
  ∃ (a b : ℝ), (∀ x : ℝ, (x^2 + 3 * x) * (x^2 + a * x + b) = ((2 - x)^2 + 3 * (2 - x)) * ((2 - x)^2 + a * (2 - x) + b)) ∧
  ((3^2 + 3 * 3) * (3^2 + (-6) * 3 + 8) = -18) :=
sorry

end symmetric_polynomial_evaluation_l1757_175796


namespace river_flow_speed_l1757_175710

theorem river_flow_speed (v : ℝ) :
  (6 - v ≠ 0) ∧ (6 + v ≠ 0) ∧ ((48 / (6 - v)) + (48 / (6 + v)) = 18) → v = 2 := 
by
  sorry

end river_flow_speed_l1757_175710


namespace largest_prime_factor_3136_l1757_175778

theorem largest_prime_factor_3136 : ∀ (n : ℕ), n = 3136 → ∃ p : ℕ, Prime p ∧ (p ∣ n) ∧ ∀ q : ℕ, (Prime q ∧ q ∣ n) → p ≥ q :=
by {
  sorry
}

end largest_prime_factor_3136_l1757_175778


namespace paint_two_faces_red_l1757_175714

theorem paint_two_faces_red (f : Fin 8 → ℕ) (H : ∀ i, 1 ≤ f i ∧ f i ≤ 8) : 
  (∃ pair_count : ℕ, pair_count = 9 ∧
    ∀ i j, i < j → f i + f j ≤ 7 → true) :=
sorry

end paint_two_faces_red_l1757_175714


namespace office_speed_l1757_175757

variable (d v : ℝ)

theorem office_speed (h1 : v > 0) (h2 : ∀ t : ℕ, t = 30) (h3 : (2 * d) / (d / v + d / 30) = 24) : v = 20 := 
sorry

end office_speed_l1757_175757


namespace worth_of_stuff_l1757_175799

theorem worth_of_stuff (x : ℝ)
  (h1 : 1.05 * x - 8 = 34) :
  x = 40 :=
by
  sorry

end worth_of_stuff_l1757_175799


namespace find_f2_l1757_175766

def f (x : ℝ) (a b : ℝ) : ℝ := x^5 + a * x^3 + b * x - 8

theorem find_f2 (a b : ℝ) (h : f (-2) a b = 10) : f 2 a b = -26 := by
  sorry

end find_f2_l1757_175766


namespace possible_values_of_n_l1757_175732

theorem possible_values_of_n :
  let a := 1500
  let max_r2 := 562499
  let total := max_r2
  let perfect_squares := (750 : Nat)
  total - perfect_squares = 561749 := by
    sorry

end possible_values_of_n_l1757_175732


namespace chord_bisected_by_point_l1757_175785

theorem chord_bisected_by_point (x1 y1 x2 y2 : ℝ) :
  (x1^2 / 36 + y1^2 / 9 = 1) ∧ (x2^2 / 36 + y2^2 / 9 = 1) ∧ 
  (x1 + x2 = 4) ∧ (y1 + y2 = 4) → (x + 4 * y - 10 = 0) :=
sorry

end chord_bisected_by_point_l1757_175785


namespace doctor_lindsay_daily_income_l1757_175748

def patients_per_hour_adult : ℕ := 4
def patients_per_hour_child : ℕ := 3
def cost_per_adult : ℕ := 50
def cost_per_child : ℕ := 25
def work_hours_per_day : ℕ := 8

theorem doctor_lindsay_daily_income : 
  (patients_per_hour_adult * cost_per_adult + patients_per_hour_child * cost_per_child) * work_hours_per_day = 2200 := 
by
  sorry

end doctor_lindsay_daily_income_l1757_175748


namespace symmetric_points_x_axis_l1757_175706

theorem symmetric_points_x_axis (m n : ℤ) (h1 : m + 1 = 1) (h2 : 3 = -(n - 2)) : m - n = 1 :=
by
  sorry

end symmetric_points_x_axis_l1757_175706


namespace hitting_first_third_fifth_probability_hitting_exactly_three_out_of_five_probability_l1757_175745

noncomputable def probability_hitting_first_third_fifth (P : ℚ) : ℚ :=
  P * (1 - P) * P * (1 - P) * P

noncomputable def binomial_coefficient (n k : ℕ) : ℕ := 
  Nat.choose n k

noncomputable def probability_hitting_exactly_three_out_of_five (P : ℚ) : ℚ :=
  binomial_coefficient 5 3 * P^3 * (1 - P)^2

theorem hitting_first_third_fifth_probability :
  probability_hitting_first_third_fifth (3/5) = 108/3125 := by
  sorry

theorem hitting_exactly_three_out_of_five_probability :
  probability_hitting_exactly_three_out_of_five (3/5) = 216/625 := by
  sorry

end hitting_first_third_fifth_probability_hitting_exactly_three_out_of_five_probability_l1757_175745


namespace simplify_and_evaluate_expr_l1757_175752

noncomputable def a : ℝ := Real.sqrt 2 - 2

noncomputable def expr (a : ℝ) : ℝ := (a - 2) / (a - 1) / (a + 1 - 3 / (a - 1))

theorem simplify_and_evaluate_expr :
  expr (Real.sqrt 2 - 2) = Real.sqrt 2 / 2 :=
by sorry

end simplify_and_evaluate_expr_l1757_175752


namespace discount_equation_l1757_175739

theorem discount_equation (x : ℝ) : 280 * (1 - x) ^ 2 = 177 := 
by 
  sorry

end discount_equation_l1757_175739


namespace average_speed_l1757_175771

theorem average_speed (v : ℝ) (v_pos : 0 < v) (v_pos_10 : 0 < v + 10):
  420 / v - 420 / (v + 10) = 2 → v = 42 :=
by
  sorry

end average_speed_l1757_175771


namespace last_three_digits_W_555_2_l1757_175784

noncomputable def W : ℕ → ℕ → ℕ
| n, 0 => n ^ n
| n, (k + 1) => W (W n k) k

theorem last_three_digits_W_555_2 : (W 555 2) % 1000 = 375 := 
by
  sorry

end last_three_digits_W_555_2_l1757_175784


namespace g_2002_equals_1_l1757_175712

theorem g_2002_equals_1 (f : ℝ → ℝ)
  (hf1 : f 1 = 1)
  (hf2 : ∀ x : ℝ, f (x + 5) ≥ f x + 5)
  (hf3 : ∀ x : ℝ, f (x + 1) ≤ f x + 1)
  (g : ℝ → ℝ := fun x => f x + 1 - x)
  : g 2002 = 1 :=
by
  sorry

end g_2002_equals_1_l1757_175712


namespace second_smallest_packs_hot_dogs_l1757_175711

theorem second_smallest_packs_hot_dogs 
    (n : ℕ) 
    (k : ℤ) 
    (h1 : 10 * n ≡ 4 [MOD 8]) 
    (h2 : n = 4 * k + 2) : 
    n = 6 :=
by sorry

end second_smallest_packs_hot_dogs_l1757_175711


namespace fraction_sum_l1757_175767

variable {a b : ℝ}

theorem fraction_sum (ha : a ≠ 0) (hb : b ≠ 0) (hab : a ≠ b) (h1 : a^2 + a - 2007 = 0) (h2 : b^2 + b - 2007 = 0) :
  (1/a + 1/b) = 1/2007 :=
by
  sorry

end fraction_sum_l1757_175767


namespace inequality_solution_l1757_175721

theorem inequality_solution (x : ℝ) :
  (x > -4 ∧ x < -5 / 3) ↔ 
  (2 * x + 3) / (3 * x + 5) > (4 * x + 1) / (x + 4) := 
sorry

end inequality_solution_l1757_175721


namespace shahrazad_stories_not_power_of_two_l1757_175786

theorem shahrazad_stories_not_power_of_two :
  ∀ (a b c : ℕ) (k : ℕ),
  a + b + c = 1001 → 27 * a + 14 * b + c = 2^k → False :=
by {
  sorry
}

end shahrazad_stories_not_power_of_two_l1757_175786


namespace determine_f_function_l1757_175731

variable (f : ℝ → ℝ)

theorem determine_f_function (x : ℝ) (h : f (1 - x) = 1 + x) : f x = 2 - x := 
sorry

end determine_f_function_l1757_175731


namespace complex_exp1990_sum_theorem_l1757_175789

noncomputable def complex_exp1990_sum (x y : Complex) (h : x ≠ 0 ∧ y ≠ 0 ∧ x^2 + x*y + y^2 = 0) : Prop :=
  (x / (x + y))^1990 + (y / (x + y))^1990 = -1

theorem complex_exp1990_sum_theorem (x y : Complex) (h : x ≠ 0 ∧ y ≠ 0 ∧ x^2 + x*y + y^2 = 0) : complex_exp1990_sum x y h :=
  sorry

end complex_exp1990_sum_theorem_l1757_175789


namespace speed_of_sound_correct_l1757_175775

-- Define the given conditions
def heard_second_blast_after : ℕ := 30 * 60 + 24 -- 30 minutes and 24 seconds in seconds
def time_sound_travelled : ℕ := 24 -- The sound traveled for 24 seconds
def distance_travelled : ℕ := 7920 -- Distance in meters

-- Define the expected answer for the speed of sound 
def expected_speed_of_sound : ℕ := 330 -- Speed in meters per second

-- The proposition that states the speed of sound given the conditions
theorem speed_of_sound_correct : (distance_travelled / time_sound_travelled) = expected_speed_of_sound := 
by {
  -- use division to compute the speed of sound
  sorry
}

end speed_of_sound_correct_l1757_175775


namespace maximum_value_of_f_l1757_175730

noncomputable def f : ℝ → ℝ :=
  fun x => -x^2 * (x^2 + 4*x + 4)

theorem maximum_value_of_f :
  ∀ x : ℝ, x ≠ 0 → x ≠ -2 → x ≠ 1 → x ≠ -3 → f x ≤ 0 ∧ f 0 = 0 :=
by
  sorry

end maximum_value_of_f_l1757_175730


namespace binom_13_10_eq_286_l1757_175726

noncomputable def binomial (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binom_13_10_eq_286 : binomial 13 10 = 286 := by
  sorry

end binom_13_10_eq_286_l1757_175726


namespace adam_room_shelves_l1757_175780

def action_figures_per_shelf : ℕ := 15
def total_action_figures : ℕ := 120
def total_shelves (total_figures shelves_capacity : ℕ) : ℕ := total_figures / shelves_capacity

theorem adam_room_shelves :
  total_shelves total_action_figures action_figures_per_shelf = 8 :=
by
  sorry

end adam_room_shelves_l1757_175780


namespace fraction_addition_l1757_175794

theorem fraction_addition :
  (3/8 : ℚ) / (4/9 : ℚ) + 1/6 = 97/96 := by
  sorry

end fraction_addition_l1757_175794


namespace inequality_abc_l1757_175733

theorem inequality_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
    1 / (a * b * c) + 1 ≥ 3 * (1 / (a^2 + b^2 + c^2) + 1 / (a + b + c)) :=
by
  sorry

end inequality_abc_l1757_175733


namespace original_price_of_trouser_l1757_175792

theorem original_price_of_trouser (P : ℝ) (sale_price : ℝ) (percent_decrease : ℝ) 
  (h1 : sale_price = 40) (h2 : percent_decrease = 0.60) 
  (h3 : sale_price = P * (1 - percent_decrease)) : P = 100 :=
by
  sorry

end original_price_of_trouser_l1757_175792


namespace alice_cookie_fills_l1757_175791

theorem alice_cookie_fills :
  (∀ (a b : ℚ), a = 3 + (3/4) ∧ b = 1/3 → (a / b) = 12) :=
sorry

end alice_cookie_fills_l1757_175791


namespace investment_difference_l1757_175798

noncomputable def A_Maria : ℝ := 60000 * (1 + 0.045)^3
noncomputable def A_David : ℝ := 60000 * (1 + 0.0175)^6
noncomputable def investment_diff : ℝ := A_Maria - A_David

theorem investment_difference : abs (investment_diff - 1803.30) < 1 :=
by
  have hM : A_Maria = 60000 * (1 + 0.045)^3 := by rfl
  have hD : A_David = 60000 * (1 + 0.0175)^6 := by rfl
  have hDiff : investment_diff = A_Maria - A_David := by rfl
  -- Proof would go here; using the provided approximations
  sorry

end investment_difference_l1757_175798


namespace count_divisible_digits_l1757_175709

def is_divisible (a b : ℕ) : Prop := b ≠ 0 ∧ a % b = 0

theorem count_divisible_digits :
  ∃! (s : Finset ℕ), s = {n | n ∈ Finset.range 10 ∧ n ≠ 0 ∧ is_divisible (25 * n) n} ∧ (Finset.card s = 3) := 
by
  sorry

end count_divisible_digits_l1757_175709


namespace convert_to_standard_spherical_coordinates_l1757_175797

theorem convert_to_standard_spherical_coordinates :
  let ρ := 4
  let θ := (3 * Real.pi) / 4
  let φ := (9 * Real.pi) / 5
  let adjusted_φ := 2 * Real.pi - φ
  let adjusted_θ := θ + Real.pi
  (ρ, adjusted_θ, adjusted_φ) = (4, (7 * Real.pi) / 4, Real.pi / 5) :=
by
  let ρ := 4
  let θ := (3 * Real.pi) / 4
  let φ := (9 * Real.pi) / 5
  let adjusted_φ := 2 * Real.pi - φ
  let adjusted_θ := θ + Real.pi
  sorry

end convert_to_standard_spherical_coordinates_l1757_175797


namespace problem_statement_l1757_175749

def g (x : ℝ) : ℝ := 3 * x^2 - 6 * x + 8

theorem problem_statement : 3 * g 2 + 4 * g (-2) = 152 := by
  sorry

end problem_statement_l1757_175749


namespace tapB_fill_in_20_l1757_175769

-- Conditions definitions
def tapA_rate (A: ℝ) : Prop := A = 3 -- Tap A fills 3 liters per minute
def total_volume (V: ℝ) : Prop := V = 36 -- Total bucket volume is 36 liters
def together_fill_time (t: ℝ) : Prop := t = 10 -- Both taps fill the bucket in 10 minutes

-- Tap B's rate can be derived from these conditions
def tapB_rate (B: ℝ) (A: ℝ) (V: ℝ) (t: ℝ) : Prop := V - (A * t) = B * t

-- The final question we need to prove
theorem tapB_fill_in_20 (B: ℝ) (A: ℝ) (V: ℝ) (t: ℝ) : 
  tapA_rate A → total_volume V → together_fill_time t → tapB_rate B A V t → B * 20 = 12 := by
  sorry

end tapB_fill_in_20_l1757_175769


namespace circle_line_intersection_zero_l1757_175719

theorem circle_line_intersection_zero (x_0 y_0 r : ℝ) (hP : x_0^2 + y_0^2 < r^2) :
  ∀ (x y : ℝ), (x^2 + y^2 = r^2) → (x_0 * x + y_0 * y = r^2) → false :=
by
  sorry

end circle_line_intersection_zero_l1757_175719


namespace mean_sharpening_instances_l1757_175779

def pencil_sharpening_instances : List ℕ :=
  [13, 8, 13, 21, 7, 23, 15, 19, 12, 9, 28, 6, 17, 29, 31, 10, 4, 20, 16, 12, 2, 18, 27, 22, 5, 14, 31, 29, 8, 25]

def mean (l : List ℕ) : ℚ :=
  (l.sum : ℚ) / l.length

theorem mean_sharpening_instances :
  mean pencil_sharpening_instances = 18.1 := by
  sorry

end mean_sharpening_instances_l1757_175779


namespace parabola_directrix_y_neg1_l1757_175718

-- We define the problem given the conditions.
def parabola_directrix (p : ℝ) : Prop :=
  ∀ x y : ℝ, x^2 = 4 * y → y = -p

-- Now we state what needs to be proved.
theorem parabola_directrix_y_neg1 : parabola_directrix 1 :=
by
  sorry

end parabola_directrix_y_neg1_l1757_175718


namespace min_value_of_z_l1757_175782

-- Define the conditions and objective function
def constraints (x y : ℝ) : Prop :=
  (y ≥ x + 2) ∧ 
  (x + y ≤ 6) ∧ 
  (x ≥ 1)

def z (x y : ℝ) : ℝ :=
  2 * |x - 2| + |y|

-- The formal theorem stating the minimum value of z under the given constraints
theorem min_value_of_z : ∃ x y : ℝ, constraints x y ∧ z x y = 4 :=
sorry

end min_value_of_z_l1757_175782


namespace five_nat_numbers_product_1000_l1757_175703

theorem five_nat_numbers_product_1000 :
  ∃ (a b c d e : ℕ), 
    a * b * c * d * e = 1000 ∧ 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ 
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ 
    c ≠ d ∧ c ≠ e ∧ 
    d ≠ e := 
by
  sorry

end five_nat_numbers_product_1000_l1757_175703


namespace marble_cut_percentage_first_week_l1757_175735

theorem marble_cut_percentage_first_week :
  ∀ (W1 W2 : ℝ), 
  W1 = W2 / 0.70 → 
  W2 = 124.95 / 0.85 → 
  (300 - W1) / 300 * 100 = 30 :=
by
  intros W1 W2 h1 h2
  sorry

end marble_cut_percentage_first_week_l1757_175735


namespace bob_selling_price_per_muffin_l1757_175751

variable (dozen_muffins_per_day : ℕ := 12)
variable (cost_per_muffin : ℝ := 0.75)
variable (weekly_profit : ℝ := 63)
variable (days_per_week : ℕ := 7)

theorem bob_selling_price_per_muffin : 
  let daily_cost := dozen_muffins_per_day * cost_per_muffin
  let weekly_cost := daily_cost * days_per_week
  let weekly_revenue := weekly_profit + weekly_cost
  let muffins_per_week := dozen_muffins_per_day * days_per_week
  let selling_price_per_muffin := weekly_revenue / muffins_per_week
  selling_price_per_muffin = 1.50 := 
by
  sorry

end bob_selling_price_per_muffin_l1757_175751


namespace problem_statement_l1757_175734

noncomputable def a (k : ℕ) : ℝ := 2^k / (3^(2^k) + 1)
noncomputable def A : ℝ := (Finset.range 10).sum (λ k => a k)
noncomputable def B : ℝ := (Finset.range 10).prod (λ k => a k)

theorem problem_statement : A / B = (3^(2^10) - 1) / 2^47 - 1 / 2^36 := 
by
  sorry

end problem_statement_l1757_175734


namespace initial_markup_percentage_l1757_175715

theorem initial_markup_percentage (C : ℝ) (M : ℝ) 
  (h1 : ∀ S_1 : ℝ, S_1 = C * (1 + M))
  (h2 : ∀ S_2 : ℝ, S_2 = C * (1 + M) * 1.25)
  (h3 : ∀ S_3 : ℝ, S_3 = C * (1 + M) * 1.25 * 0.94)
  (h4 : ∀ S_3 : ℝ, S_3 = C * 1.41) : 
  M = 0.2 :=
by
  sorry

end initial_markup_percentage_l1757_175715


namespace average_length_of_remaining_strings_l1757_175774

theorem average_length_of_remaining_strings :
  ∀ (n_cat : ℕ) 
    (avg_len_total avg_len_one_fourth avg_len_one_third : ℝ)
    (total_length total_length_one_fourth total_length_one_third remaining_length : ℝ),
    n_cat = 12 →
    avg_len_total = 90 →
    avg_len_one_fourth = 75 →
    avg_len_one_third = 65 →
    total_length = n_cat * avg_len_total →
    total_length_one_fourth = (n_cat / 4) * avg_len_one_fourth →
    total_length_one_third = (n_cat / 3) * avg_len_one_third →
    remaining_length = total_length - (total_length_one_fourth + total_length_one_third) →
    remaining_length / (n_cat - (n_cat / 4 + n_cat / 3)) = 119 :=
by sorry

end average_length_of_remaining_strings_l1757_175774


namespace helga_tried_on_66_pairs_of_shoes_l1757_175743

variables 
  (n1 n2 n3 n4 n5 n6 : ℕ)
  (h1 : n1 = 7)
  (h2 : n2 = n1 + 2)
  (h3 : n3 = 0)
  (h4 : n4 = 2 * (n1 + n2 + n3))
  (h5 : n5 = n2 - 3)
  (h6 : n6 = n1 + 5)
  (total : ℕ := n1 + n2 + n3 + n4 + n5 + n6)

theorem helga_tried_on_66_pairs_of_shoes : total = 66 :=
by sorry

end helga_tried_on_66_pairs_of_shoes_l1757_175743


namespace sum_of_squares_of_roots_l1757_175701

theorem sum_of_squares_of_roots (s1 s2 : ℝ) (h1 : s1 * s2 = 4) (h2 : s1 + s2 = 16) : s1^2 + s2^2 = 248 :=
by
  sorry

end sum_of_squares_of_roots_l1757_175701


namespace nylon_cord_length_l1757_175768

theorem nylon_cord_length {L : ℝ} (hL : L = 30) : ∃ (w : ℝ), w = 5 := 
by sorry

end nylon_cord_length_l1757_175768


namespace proper_divisors_condition_l1757_175747

theorem proper_divisors_condition (N : ℕ) :
  ∀ x : ℕ, (x ∣ N ∧ x ≠ 1 ∧ x ≠ N) → 
  (∀ L : ℕ, (L ∣ N ∧ L ≠ 1 ∧ L ≠ N) → (L = x^3 + 3 ∨ L = x^3 - 3)) → 
  (N = 10 ∨ N = 22) :=
by
  sorry

end proper_divisors_condition_l1757_175747


namespace probability_heads_9_of_12_is_correct_l1757_175754

-- Defining the probability of getting exactly 9 heads in 12 flips of a fair coin
noncomputable def probability_heads_9_of_12 := (55 : ℚ) / 1024

-- Theorem statement
theorem probability_heads_9_of_12_is_correct :
  probability_heads_9_of_12 = (55 : ℚ) / 1024 := 
  by sorry

end probability_heads_9_of_12_is_correct_l1757_175754


namespace total_sales_first_three_days_total_earnings_seven_days_l1757_175753

def planned_daily_sales : Int := 100

def deviation : List Int := [4, -3, -5, 14, -8, 21, -6]

def selling_price_per_pound : Int := 8
def freight_cost_per_pound : Int := 3

-- Part (1): Proof statement for the total amount sold in the first three days
theorem total_sales_first_three_days :
  let monday_sales := planned_daily_sales + deviation.head!
  let tuesday_sales := planned_daily_sales + (deviation.drop 1).head!
  let wednesday_sales := planned_daily_sales + (deviation.drop 2).head!
  monday_sales + tuesday_sales + wednesday_sales = 296 := by
  sorry

-- Part (2): Proof statement for Xiaoming's total earnings for the seven days
theorem total_earnings_seven_days :
  let total_sales := (List.sum (deviation.map (λ x => planned_daily_sales + x)))
  total_sales * (selling_price_per_pound - freight_cost_per_pound) = 3585 := by
  sorry

end total_sales_first_three_days_total_earnings_seven_days_l1757_175753


namespace total_detergent_used_l1757_175763

-- Define the parameters of the problem
def total_pounds_of_clothes : ℝ := 9
def pounds_of_cotton : ℝ := 4
def pounds_of_woolen : ℝ := 5
def detergent_per_pound_cotton : ℝ := 2
def detergent_per_pound_woolen : ℝ := 1.5

-- Main theorem statement
theorem total_detergent_used : 
  (pounds_of_cotton * detergent_per_pound_cotton) + (pounds_of_woolen * detergent_per_pound_woolen) = 15.5 :=
by
  sorry

end total_detergent_used_l1757_175763


namespace votes_switched_l1757_175746

theorem votes_switched (x : ℕ) (total_votes : ℕ) (half_votes : ℕ) 
  (votes_first_round : ℕ) (votes_second_round_winner : ℕ) (votes_second_round_loser : ℕ)
  (cond1 : total_votes = 48000)
  (cond2 : half_votes = total_votes / 2)
  (cond3 : votes_first_round = half_votes)
  (cond4 : votes_second_round_winner = half_votes + x)
  (cond5 : votes_second_round_loser = half_votes - x)
  (cond6 : votes_second_round_winner = 5 * votes_second_round_loser) :
  x = 16000 := by
  -- Proof will go here
  sorry

end votes_switched_l1757_175746


namespace percentage_increase_is_20_l1757_175723

def number_of_students_this_year : ℕ := 960
def number_of_students_last_year : ℕ := 800

theorem percentage_increase_is_20 :
  ((number_of_students_this_year - number_of_students_last_year : ℕ) / number_of_students_last_year * 100) = 20 := 
by
  sorry

end percentage_increase_is_20_l1757_175723


namespace max_matches_l1757_175772

theorem max_matches (x y z m : ℕ) (h1 : x + y + z = 19) (h2 : x * y + y * z + x * z = m) : m ≤ 120 :=
sorry

end max_matches_l1757_175772


namespace population_reaches_210_l1757_175787

noncomputable def population_function (x : ℕ) : ℝ :=
  200 * (1 + 0.01)^x

theorem population_reaches_210 :
  ∃ x : ℕ, population_function x >= 210 :=
by
  existsi 5
  apply le_of_lt
  sorry

end population_reaches_210_l1757_175787
