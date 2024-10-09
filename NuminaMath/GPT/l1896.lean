import Mathlib

namespace arithmetic_progression_even_terms_l1896_189620

theorem arithmetic_progression_even_terms (a d n : ℕ) (h_even : n % 2 = 0)
  (h_last_first_diff : (n - 1) * d = 16)
  (h_sum_odd : n * (a + (n - 2) * d / 2) = 81)
  (h_sum_even : n * (a + d + (n - 2) * d / 2) = 75) :
  n = 8 :=
by sorry

end arithmetic_progression_even_terms_l1896_189620


namespace total_cubes_l1896_189626

noncomputable def original_cubes : ℕ := 2
noncomputable def additional_cubes : ℕ := 7

theorem total_cubes : original_cubes + additional_cubes = 9 := by
  sorry

end total_cubes_l1896_189626


namespace range_of_k_real_roots_l1896_189640

variable (k : ℝ)
def quadratic_has_real_roots : Prop :=
  let a := k - 1
  let b := 2
  let c := 1
  let Δ := b^2 - 4 * a * c
  Δ ≥ 0 ∧ a ≠ 0

theorem range_of_k_real_roots :
  quadratic_has_real_roots k ↔ (k ≤ 2 ∧ k ≠ 1) := by
  sorry

end range_of_k_real_roots_l1896_189640


namespace find_sum_of_m_and_k_l1896_189605

theorem find_sum_of_m_and_k
  (d m k : ℤ)
  (h : (9 * d^2 - 5 * d + m) * (4 * d^2 + k * d - 6) = 36 * d^4 + 11 * d^3 - 59 * d^2 + 10 * d + 12) :
  m + k = -7 :=
by sorry

end find_sum_of_m_and_k_l1896_189605


namespace range_of_a_l1896_189611

variables (m a x y : ℝ)

def p (m a : ℝ) : Prop := m^2 + 12 * a^2 < 7 * a * m ∧ a > 0

def ellipse (m x y : ℝ) : Prop := (x^2)/(m-1) + (y^2)/(2-m) = 1

def q (m : ℝ) (x y : ℝ) : Prop := ellipse m x y ∧ 1 < m ∧ m < 3/2

theorem range_of_a :
  (∃ m, p m a → (∀ x y, q m x y)) → (1/3 ≤ a ∧ a ≤ 3/8) :=
sorry

end range_of_a_l1896_189611


namespace evaluate_tan_fraction_l1896_189633

theorem evaluate_tan_fraction:
  (1 - Real.tan (15 * Real.pi / 180)) / (1 + Real.tan (15 * Real.pi / 180)) = Real.sqrt 3 / 3 :=
by
  sorry

end evaluate_tan_fraction_l1896_189633


namespace max_digit_sum_watch_l1896_189673

def digit_sum (n : Nat) : Nat :=
  (n / 10) + (n % 10)

theorem max_digit_sum_watch :
  ∃ (h m : Nat), (1 <= h ∧ h <= 12) ∧ (0 <= m ∧ m <= 59) 
  ∧ (digit_sum h + digit_sum m = 23) :=
by 
  sorry

end max_digit_sum_watch_l1896_189673


namespace intersect_at_one_point_l1896_189666

-- Definitions of points and circles
variable (Point : Type)
variable (Circle : Type)
variable (A : Point)
variable (C1 C2 C3 C4 : Circle)

-- Definition of intersection points
variable (B12 B13 B14 B23 B24 B34 : Point)

-- Note: Assumptions around the geometry structure axioms need to be defined
-- Assuming we have a function that checks if three points are collinear:
variable (are_collinear : Point → Point → Point → Prop)
-- Assuming we have a function that checks if a point is part of a circle:
variable (on_circle : Point → Circle → Prop)

-- Axioms related to the conditions
axiom collinear_B12_B34_B (hC1 : on_circle B12 C1) (hC2 : on_circle B12 C2) (hC3 : on_circle B34 C3) (hC4 : on_circle B34 C4) : 
  ∃ P : Point, are_collinear B12 P B34 

axiom collinear_B13_B24_B (hC1 : on_circle B13 C1) (hC2 : on_circle B13 C3) (hC3 : on_circle B24 C2) (hC4 : on_circle B24 C4) : 
  ∃ P : Point, are_collinear B13 P B24 

axiom collinear_B14_B23_B (hC1 : on_circle B14 C1) (hC2 : on_circle B14 C4) (hC3 : on_circle B23 C2) (hC4 : on_circle B23 C3) : 
  ∃ P : Point, are_collinear B14 P B23 

-- The theorem to be proved
theorem intersect_at_one_point :
  ∃ P : Point, 
    are_collinear B12 P B34 ∧ are_collinear B13 P B24 ∧ are_collinear B14 P B23 := 
sorry

end intersect_at_one_point_l1896_189666


namespace fencing_rate_correct_l1896_189671

noncomputable def rate_per_meter (d : ℝ) (cost : ℝ) : ℝ :=
  cost / (Real.pi * d)

theorem fencing_rate_correct : rate_per_meter 26 122.52211349000194 = 1.5 := by
  sorry

end fencing_rate_correct_l1896_189671


namespace bie_l1896_189684

noncomputable def surface_area_of_sphere (PA AB AC : ℝ) (hPA_AB : PA = AB) (hPA : PA = 2) (hAC : AC = 4) (r : ℝ) : ℝ :=
  let PC := Real.sqrt (PA ^ 2 + AC ^ 2)
  let radius := PC / 2
  4 * Real.pi * radius ^ 2

theorem bie'zhi_tetrahedron_surface_area
  (PA AB AC : ℝ)
  (hPA_AB : PA = AB)
  (hPA : PA = 2)
  (hAC : AC = 4)
  (PC : ℝ := Real.sqrt (PA ^ 2 + AC ^ 2))
  (r : ℝ := PC / 2)
  (surface_area : ℝ := 4 * Real.pi * r ^ 2)
  :
  surface_area = 20 * Real.pi := 
sorry

end bie_l1896_189684


namespace shape_is_cone_l1896_189618

-- Define spherical coordinates
structure SphericalCoordinates where
  ρ : ℝ
  θ : ℝ
  φ : ℝ

-- Define the positive constant c
def c : ℝ := sorry

-- Assume c is positive
axiom c_positive : c > 0

-- Define the shape equation in spherical coordinates
def shape_equation (p : SphericalCoordinates) : Prop :=
  p.ρ = c * Real.sin p.φ

-- The theorem statement
theorem shape_is_cone (p : SphericalCoordinates) : shape_equation p → 
  ∃ z : ℝ, (z = p.ρ * Real.cos p.φ) ∧ (p.ρ ^ 2 = (c * Real.sin p.φ) ^ 2 + z ^ 2) :=
sorry

end shape_is_cone_l1896_189618


namespace batsman_avg_l1896_189631

variable (A : ℕ) -- The batting average in 46 innings

-- Given conditions
variables (highest lowest : ℕ)
variables (diff : ℕ) (avg_excl : ℕ) (num_excl : ℕ)

namespace cricket

-- Define the given values
def highest_score := 225
def difference := 150
def avg_excluding := 58
def num_excluding := 44

-- Calculate the lowest score
def lowest_score := highest_score - difference

-- Calculate the total runs in 44 innings excluding highest and lowest scores
def total_run_excluded := avg_excluding * num_excluding

-- Calculate the total runs in 46 innings
def total_runs := total_run_excluded + highest_score + lowest_score

-- Define the equation relating the average to everything else
def batting_avg_eq : Prop :=
  total_runs = 46 * A

-- Prove that the batting average A is 62 given the conditions
theorem batsman_avg :
  A = 62 :=
  by
    sorry

end cricket

end batsman_avg_l1896_189631


namespace sequence_inequality_l1896_189606

theorem sequence_inequality (a : ℕ → ℕ) (strictly_increasing : ∀ n, a n < a (n + 1))
  (sum_condition : ∀ m : ℕ, ∃ i j : ℕ, m = a i + a j) :
  ∀ n, a n ≤ n^2 :=
by sorry

end sequence_inequality_l1896_189606


namespace george_correct_answer_l1896_189623

variable (y : ℝ)

theorem george_correct_answer (h : y / 7 = 30) : 70 + y = 280 :=
sorry

end george_correct_answer_l1896_189623


namespace volume_of_cube_l1896_189628

theorem volume_of_cube (SA : ℝ) (H : SA = 600) : (10^3 : ℝ) = 1000 :=
by
  sorry

end volume_of_cube_l1896_189628


namespace algebraic_identity_l1896_189661

theorem algebraic_identity (a b : ℝ) (h1 : a + b = -2) (h2 : a - b = 4) :
    a^2 - b^2 = -8 := by
  sorry

end algebraic_identity_l1896_189661


namespace vector_parallel_solution_l1896_189622

theorem vector_parallel_solution (x : ℝ) : 
  let a := (2, 3)
  let b := (x, -9)
  (a.snd = 3) → (a.fst = 2) → (b.snd = -9) → (a.fst * b.snd = a.snd * (b.fst)) → x = -6 := 
by
  intros 
  sorry

end vector_parallel_solution_l1896_189622


namespace bottom_level_legos_l1896_189656

theorem bottom_level_legos
  (x : ℕ)
  (h : x^2 + (x - 1)^2 + (x - 2)^2 = 110) :
  x = 7 :=
by {
  sorry
}

end bottom_level_legos_l1896_189656


namespace loss_of_50_denoted_as_minus_50_l1896_189607

def is_profit (x : Int) : Prop :=
  x > 0

def is_loss (x : Int) : Prop :=
  x < 0

theorem loss_of_50_denoted_as_minus_50 : is_loss (-50) :=
  by
    -- proof steps would go here
    sorry

end loss_of_50_denoted_as_minus_50_l1896_189607


namespace ellie_runs_8_miles_in_24_minutes_l1896_189650

theorem ellie_runs_8_miles_in_24_minutes (time_max : ℝ) (distance_max : ℝ) 
  (time_ellie_fraction : ℝ) (distance_ellie : ℝ) (distance_ellie_final : ℝ)
  (h1 : distance_max = 6) 
  (h2 : time_max = 36) 
  (h3 : time_ellie_fraction = 1/3) 
  (h4 : distance_ellie = 4) 
  (h5 : distance_ellie_final = 8) :
  ((time_ellie_fraction * time_max) / distance_ellie) * distance_ellie_final = 24 :=
by
  sorry

end ellie_runs_8_miles_in_24_minutes_l1896_189650


namespace average_computer_time_per_person_is_95_l1896_189678

def people : ℕ := 8
def computers : ℕ := 5
def work_time : ℕ := 152 -- total working day minutes

def total_computer_time : ℕ := work_time * computers
def average_time_per_person : ℕ := total_computer_time / people

theorem average_computer_time_per_person_is_95 :
  average_time_per_person = 95 := 
by
  sorry

end average_computer_time_per_person_is_95_l1896_189678


namespace helicopter_rental_cost_l1896_189688

theorem helicopter_rental_cost
  (hours_per_day : ℕ)
  (total_days : ℕ)
  (total_cost : ℕ)
  (H1 : hours_per_day = 2)
  (H2 : total_days = 3)
  (H3 : total_cost = 450) :
  total_cost / (hours_per_day * total_days) = 75 :=
by
  sorry

end helicopter_rental_cost_l1896_189688


namespace fifth_term_sequence_l1896_189658

theorem fifth_term_sequence : 
  (4 + 8 + 16 + 32 + 64) = 124 := 
by 
  sorry

end fifth_term_sequence_l1896_189658


namespace inclination_of_line_l1896_189697

theorem inclination_of_line (α : ℝ) (h1 : ∃ l : ℝ, ∀ x y : ℝ, x + y + 1 = 0 → y = -x - 1) : α = 135 :=
by
  sorry

end inclination_of_line_l1896_189697


namespace solve_for_x_l1896_189682

variable {a b c x : ℝ}
variable (h_pos : a > 0 ∧ b > 0 ∧ c > 0)
variable (h_eq : (x - a - b) / c + (x - b - c) / a + (x - c - a) / b = 3)

theorem solve_for_x (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_eq : (x - a - b) / c + (x - b - c) / a + (x - c - a) / b = 3) : 
  x = a + b + c :=
sorry

end solve_for_x_l1896_189682


namespace porter_l1896_189609

def previous_sale_amount : ℕ := 9000

def recent_sale_price (previous_sale_amount : ℕ) : ℕ :=
  5 * previous_sale_amount - 1000

theorem porter's_recent_sale : recent_sale_price previous_sale_amount = 44000 :=
by
  sorry

end porter_l1896_189609


namespace square_side_length_l1896_189675

theorem square_side_length (d s : ℝ) (h_diag : d = 2) (h_rel : d = s * Real.sqrt 2) : s = Real.sqrt 2 :=
sorry

end square_side_length_l1896_189675


namespace min_x_value_l1896_189612

noncomputable def circle_eq (x y : ℝ) : Prop :=
  x^2 + y^2 = 18 * x + 50 * y + 56

theorem min_x_value : 
  ∃ (x : ℝ), ∃ (y : ℝ), circle_eq x y ∧ x = 9 - Real.sqrt 762 :=
by
  sorry

end min_x_value_l1896_189612


namespace shaded_area_percentage_l1896_189637

theorem shaded_area_percentage (side : ℕ) (total_shaded_area : ℕ) (expected_percentage : ℕ)
  (h1 : side = 5)
  (h2 : total_shaded_area = 15)
  (h3 : expected_percentage = 60) :
  ((total_shaded_area : ℚ) / (side * side) * 100) = expected_percentage :=
by
  sorry

end shaded_area_percentage_l1896_189637


namespace ratio_m_q_l1896_189692

theorem ratio_m_q (m n p q : ℚ) (h1 : m / n = 25) (h2 : p / n = 5) (h3 : p / q = 1 / 15) : 
  m / q = 1 / 3 :=
by 
  sorry

end ratio_m_q_l1896_189692


namespace max_profit_l1896_189604

noncomputable def fixed_cost := 20000
noncomputable def variable_cost (x : ℝ) : ℝ :=
  if x < 8 then (1/3) * x^2 + 2 * x else 7 * x + 100 / x - 37
noncomputable def sales_price_per_unit : ℝ := 6
noncomputable def profit (x : ℝ) : ℝ :=
  let revenue := sales_price_per_unit * x
  let cost := fixed_cost / 10000 + variable_cost x
  revenue - cost

theorem max_profit : ∃ x : ℝ, (0 < x) ∧ (15 = profit 10) :=
by {
  sorry
}

end max_profit_l1896_189604


namespace emma_additional_miles_l1896_189602

theorem emma_additional_miles :
  ∀ (initial_distance : ℝ) (initial_speed : ℝ) (additional_speed : ℝ) (desired_avg_speed : ℝ) (total_distance : ℝ) (additional_distance : ℝ),
    initial_distance = 20 →
    initial_speed = 40 →
    additional_speed = 70 →
    desired_avg_speed = 60 →
    total_distance = initial_distance + additional_distance →
    (total_distance / ((initial_distance / initial_speed) + (additional_distance / additional_speed))) = desired_avg_speed →
    additional_distance = 70 :=
by
  intros initial_distance initial_speed additional_speed desired_avg_speed total_distance additional_distance
  intros h1 h2 h3 h4 h5 h6
  sorry

end emma_additional_miles_l1896_189602


namespace verify_BG_BF_verify_FG_EG_find_x_l1896_189616

noncomputable def verify_angles (CBG GBE EBF BCF FCE : ℝ) :=
  CBG = 20 ∧ GBE = 40 ∧ EBF = 20 ∧ BCF = 50 ∧ FCE = 30

theorem verify_BG_BF (CBG GBE EBF BCF FCE : ℝ) :
  verify_angles CBG GBE EBF BCF FCE → BG = BF :=
by
  sorry

theorem verify_FG_EG (CBG GBE EBF BCF FCE : ℝ) :
  verify_angles CBG GBE EBF BCF FCE → FG = EG :=
by
  sorry

theorem find_x (CBG GBE EBF BCF FCE : ℝ) :
  verify_angles CBG GBE EBF BCF FCE → x = 30 :=
by
  sorry

end verify_BG_BF_verify_FG_EG_find_x_l1896_189616


namespace lucas_total_pages_l1896_189655

-- Define the variables and conditions
def lucas_read_pages : Nat :=
  let pages_first_four_days := 4 * 20
  let pages_break_day := 0
  let pages_next_four_days := 4 * 30
  let pages_last_day := 15
  pages_first_four_days + pages_break_day + pages_next_four_days + pages_last_day

-- State the theorem
theorem lucas_total_pages :
  lucas_read_pages = 215 :=
sorry

end lucas_total_pages_l1896_189655


namespace sauna_max_couples_l1896_189680

def max_couples (n : ℕ) : ℕ :=
  n - 1

theorem sauna_max_couples (n : ℕ) (rooms unlimited_capacity : Prop) (no_female_male_cohabsimult : Prop)
                          (males_shared_room_constraint females_shared_room_constraint : Prop)
                          (males_known_iff_wives_known : Prop) : max_couples n = n - 1 := 
  sorry

end sauna_max_couples_l1896_189680


namespace ant_rest_position_l1896_189654

noncomputable def percent_way_B_to_C (s : ℕ) : ℕ :=
  let perimeter := 3 * s
  let distance_traveled := (42 * perimeter) / 100
  let distance_AB := s
  let remaining_distance := distance_traveled - distance_AB
  (remaining_distance * 100) / s

theorem ant_rest_position :
  ∀ (s : ℕ), percent_way_B_to_C s = 26 :=
by
  intros
  unfold percent_way_B_to_C
  sorry

end ant_rest_position_l1896_189654


namespace apple_capacity_l1896_189698

/-- Question: What is the largest possible number of apples that can be held by the 6 boxes and 4 extra trays?
 Conditions:
 - Paul has 6 boxes.
 - Each box contains 12 trays.
 - Paul has 4 extra trays.
 - Each tray can hold 8 apples.
 Answer: 608 apples
-/
theorem apple_capacity :
  let boxes := 6
  let trays_per_box := 12
  let extra_trays := 4
  let apples_per_tray := 8
  let total_trays := (boxes * trays_per_box) + extra_trays
  let total_apples_capacity := total_trays * apples_per_tray
  total_apples_capacity = 608 := 
by
  sorry

end apple_capacity_l1896_189698


namespace solve_for_a_plus_b_l1896_189608

theorem solve_for_a_plus_b (a b : ℝ) :
  (∀ x : ℝ, (-1 < x ∧ x < 1 / 3) → ax^2 + bx + 1 > 0) →
  a * (-3) + b = -5 :=
by
  intro h
  -- Here we can use the proofs provided in the solution steps.
  sorry

end solve_for_a_plus_b_l1896_189608


namespace dice_surface_dots_l1896_189643

def total_dots_on_die := 1 + 2 + 3 + 4 + 5 + 6

def total_dots_on_seven_dice := 7 * total_dots_on_die

def hidden_dots_on_central_die := total_dots_on_die

def visible_dots_on_surface := total_dots_on_seven_dice - hidden_dots_on_central_die

theorem dice_surface_dots : visible_dots_on_surface = 105 := by
  sorry

end dice_surface_dots_l1896_189643


namespace sum_of_three_terms_divisible_by_3_l1896_189676

theorem sum_of_three_terms_divisible_by_3 (a : Fin 5 → ℤ) :
  ∃ (i j k : Fin 5), i ≠ j ∧ i ≠ k ∧ j ≠ k ∧ (a i + a j + a k) % 3 = 0 :=
by
  sorry

end sum_of_three_terms_divisible_by_3_l1896_189676


namespace prime_factor_of_difference_l1896_189646

theorem prime_factor_of_difference (A B C : ℕ) (hA : A ≠ 0) (hABC_digits : A ≠ B ∧ A ≠ C ∧ B ≠ C) 
  (hA_range : 0 ≤ A ∧ A ≤ 9) (hB_range : 0 ≤ B ∧ B ≤ 9) (hC_range : 0 ≤ C ∧ C ≤ 9) :
  11 ∣ (100 * A + 10 * B + C) - (100 * C + 10 * B + A) :=
by
  sorry

end prime_factor_of_difference_l1896_189646


namespace alcohol_mix_problem_l1896_189645

theorem alcohol_mix_problem
  (x_volume : ℕ) (y_volume : ℕ)
  (x_percentage : ℝ) (y_percentage : ℝ)
  (target_percentage : ℝ)
  (x_volume_eq : x_volume = 200)
  (x_percentage_eq : x_percentage = 0.10)
  (y_percentage_eq : y_percentage = 0.30)
  (target_percentage_eq : target_percentage = 0.14)
  (y_solution : ℝ)
  (h : y_volume = 50) :
  (20 + 0.3 * y_solution) / (200 + y_solution) = target_percentage := by sorry

end alcohol_mix_problem_l1896_189645


namespace arithmetic_sequence_propositions_l1896_189681

theorem arithmetic_sequence_propositions (a_n : ℕ → ℤ) (S : ℕ → ℤ)
  (h_S_def : ∀ n, S n = n * (a_n 1 + (a_n (n - 1))) / 2)
  (h_cond : S 6 > S 7 ∧ S 7 > S 5) :
  (∃ d, d < 0 ∧ S 11 > 0) :=
by
  sorry

end arithmetic_sequence_propositions_l1896_189681


namespace rate_percent_l1896_189674

theorem rate_percent (SI P T: ℝ) (h₁: SI = 250) (h₂: P = 1500) (h₃: T = 5) : 
  ∃ R : ℝ, R = (SI * 100) / (P * T) := 
by
  use (250 * 100) / (1500 * 5)
  sorry

end rate_percent_l1896_189674


namespace find_c_l1896_189632

theorem find_c (c : ℝ) : (∃ a : ℝ, (x : ℝ) → (x^2 + 80*x + c = (x + a)^2)) → (c = 1600) := by
  sorry

end find_c_l1896_189632


namespace tangent_line_of_cubic_at_l1896_189641

theorem tangent_line_of_cubic_at (x y : ℝ) (h : y = x^3) (hx : x = 1) (hy : y = 1) : 
  3 * x - y - 2 = 0 :=
sorry

end tangent_line_of_cubic_at_l1896_189641


namespace equivalence_sufficient_necessary_l1896_189627

-- Definitions for conditions
variables (A B : Prop)

-- Statement to prove
theorem equivalence_sufficient_necessary :
  (A → B) ↔ (¬B → ¬A) :=
by sorry

end equivalence_sufficient_necessary_l1896_189627


namespace no_real_roots_quadratic_l1896_189642

theorem no_real_roots_quadratic (a b c : ℝ) (h : a = 1 ∧ b = -4 ∧ c = 8) :
    (a ≠ 0) → (∀ x : ℝ, a * x^2 + b * x + c ≠ 0) :=
by
  sorry

end no_real_roots_quadratic_l1896_189642


namespace jenna_eel_length_l1896_189601

theorem jenna_eel_length (J B L : ℝ)
  (h1 : J = (2 / 5) * B)
  (h2 : J = (3 / 7) * L)
  (h3 : J + B + L = 124) : 
  J = 21 := 
sorry

end jenna_eel_length_l1896_189601


namespace positive_solution_iff_abs_a_b_lt_one_l1896_189659

theorem positive_solution_iff_abs_a_b_lt_one
  (a b : ℝ)
  (x1 x2 x3 x4 : ℝ)
  (h1 : x1 - x2 = a)
  (h2 : x3 - x4 = b)
  (h3 : x1 + x2 + x3 + x4 = 1)
  (h4 : x1 > 0)
  (h5 : x2 > 0)
  (h6 : x3 > 0)
  (h7 : x4 > 0) :
  |a| + |b| < 1 :=
sorry

end positive_solution_iff_abs_a_b_lt_one_l1896_189659


namespace tan_shift_monotonic_interval_l1896_189614

noncomputable def monotonic_interval (k : ℤ) : Set ℝ :=
  {x : ℝ | k * Real.pi - 3 * Real.pi / 4 < x ∧ x < k * Real.pi + Real.pi / 4}

theorem tan_shift_monotonic_interval {k : ℤ} :
  ∀ x, (monotonic_interval k x) → (Real.tan (x + Real.pi / 4)) = (Real.tan x) := sorry

end tan_shift_monotonic_interval_l1896_189614


namespace books_left_after_giveaways_l1896_189653

def initial_books : ℝ := 48.0
def first_giveaway : ℝ := 34.0
def second_giveaway : ℝ := 3.0

theorem books_left_after_giveaways : 
  initial_books - first_giveaway - second_giveaway = 11.0 :=
by
  sorry

end books_left_after_giveaways_l1896_189653


namespace total_tiles_needed_l1896_189690

-- Define the dimensions of the dining room
def dining_room_length : ℕ := 15
def dining_room_width : ℕ := 20

-- Define the width of the border
def border_width : ℕ := 2

-- Areas for one-foot by one-foot border tiles
def one_foot_tile_border_tiles : ℕ :=
  2 * (dining_room_width + (dining_room_width - 2 * border_width)) + 
  2 * ((dining_room_length - 2) + (dining_room_length - 2 * border_width))

-- Dimensions of the inner area
def inner_length : ℕ := dining_room_length - 2 * border_width
def inner_width : ℕ := dining_room_width - 2 * border_width

-- Area for two-foot by two-foot tiles
def inner_area : ℕ := inner_length * inner_width
def two_foot_tile_inner_tiles : ℕ := inner_area / 4

-- Total number of tiles
def total_tiles : ℕ := one_foot_tile_border_tiles + two_foot_tile_inner_tiles

-- Prove that the total number of tiles needed is 168
theorem total_tiles_needed : total_tiles = 168 := sorry

end total_tiles_needed_l1896_189690


namespace quadratic_roots_properties_quadratic_roots_max_min_l1896_189625

theorem quadratic_roots_properties (k : ℝ) (h : 2 ≤ k ∧ k ≤ 8)
  (x1 x2 : ℝ) (h_roots : x1 + x2 = 2 * (k - 1) ∧ x1 * x2 = 2 * k^2 - 12 * k + 17) :
  (x1^2 + x2^2) = 16 * k - 30 :=
sorry

theorem quadratic_roots_max_min :
  (∀ k ∈ { k : ℝ | 2 ≤ k ∧ k ≤ 8 }, 
    ∃ (x1 x2 : ℝ), 
      (x1 + x2 = 2 * (k - 1) ∧ x1 * x2 = 2 * k^2 - 12 * k + 17) 
      ∧ (x1^2 + x2^2) = (if k = 8 then 98 else if k = 2 then 2 else 16 * k - 30)) :=
sorry

end quadratic_roots_properties_quadratic_roots_max_min_l1896_189625


namespace eugene_initial_pencils_l1896_189662

theorem eugene_initial_pencils (P : ℕ) (h1 : P + 6 = 57) : P = 51 :=
by
  sorry

end eugene_initial_pencils_l1896_189662


namespace RectangleAreaDiagonalk_l1896_189657

theorem RectangleAreaDiagonalk {length width : ℝ} {d : ℝ}
  (h_ratio : length / width = 5 / 2)
  (h_perimeter : 2 * (length + width) = 42)
  (h_diagonal : d = Real.sqrt (length^2 + width^2))
  : (∃ k, k = 10 / 29 ∧ ∀ A, A = k * d^2) :=
by {
  sorry
}

end RectangleAreaDiagonalk_l1896_189657


namespace arithmetic_mean_difference_l1896_189615

theorem arithmetic_mean_difference (p q r : ℝ) 
  (h1 : (p + q) / 2 = 10) 
  (h2 : (q + r) / 2 = 26) : 
  r - p = 32 := by
  sorry

end arithmetic_mean_difference_l1896_189615


namespace min_c_plus_3d_l1896_189679

theorem min_c_plus_3d (c d : ℝ) (hc : 0 < c) (hd : 0 < d) 
    (h1 : c^2 ≥ 12 * d) (h2 : 9 * d^2 ≥ 4 * c) : 
  c + 3 * d ≥ 8 :=
  sorry

end min_c_plus_3d_l1896_189679


namespace magic_square_y_value_l1896_189696

/-- In a magic square, where the sum of three entries in any row, column, or diagonal is the same value.
    Given the entries as shown below, prove that \(y = -38\).
    The entries are: 
    - \( y \) at position (1,1)
    - 23 at position (1,2)
    - 101 at position (1,3)
    - 4 at position (2,1)
    The remaining positions are denoted as \( a, b, c, d, e \).
-/
theorem magic_square_y_value :
    ∃ y a b c d e: ℤ,
        y + 4 + c = y + 23 + 101 ∧ -- Condition from first column and first row
        23 + a + d = 101 + b + 4 ∧ -- Condition from middle column and diagonal
        c + d + e = 101 + b + e ∧ -- Condition from bottom row and rightmost column
        y + 23 + 101 = 4 + a + b → -- Condition from top row
        y = -38 := 
by
    sorry

end magic_square_y_value_l1896_189696


namespace jerry_gets_logs_l1896_189629

def logs_per_pine_tree : ℕ := 80
def logs_per_maple_tree : ℕ := 60
def logs_per_walnut_tree : ℕ := 100
def logs_per_oak_tree : ℕ := 90
def logs_per_birch_tree : ℕ := 55

def pine_trees_cut : ℕ := 8
def maple_trees_cut : ℕ := 3
def walnut_trees_cut : ℕ := 4
def oak_trees_cut : ℕ := 7
def birch_trees_cut : ℕ := 5

def total_logs : ℕ :=
  pine_trees_cut * logs_per_pine_tree +
  maple_trees_cut * logs_per_maple_tree +
  walnut_trees_cut * logs_per_walnut_tree +
  oak_trees_cut * logs_per_oak_tree +
  birch_trees_cut * logs_per_birch_tree

theorem jerry_gets_logs : total_logs = 2125 :=
by
  sorry

end jerry_gets_logs_l1896_189629


namespace centroid_of_triangle_l1896_189634

theorem centroid_of_triangle (x1 y1 x2 y2 x3 y3 : ℝ) :
  let x_centroid := (x1 + x2 + x3) / 3
  let y_centroid := (y1 + y2 + y3) / 3
  (x_centroid, y_centroid) = (1/3 * (x1 + x2 + x3), 1/3 * (y1 + y2 + y3)) :=
by
  sorry

end centroid_of_triangle_l1896_189634


namespace lukas_avg_points_per_game_l1896_189685

theorem lukas_avg_points_per_game (total_points games_played : ℕ) (h_total_points : total_points = 60) (h_games_played : games_played = 5) :
  (total_points / games_played = 12) :=
by
  sorry

end lukas_avg_points_per_game_l1896_189685


namespace buttons_pattern_total_buttons_sum_l1896_189687

-- Define the sequence of the number of buttons in each box
def buttons_in_box (n : ℕ) : ℕ := 3^(n-1)

-- Define the sum of buttons up to the n-th box
def total_buttons (n : ℕ) : ℕ := (3^n - 1) / 2

-- Theorem statements to prove
theorem buttons_pattern (n : ℕ) : buttons_in_box n = 3^(n-1) := by
  sorry

theorem total_buttons_sum (n : ℕ) : total_buttons n = (3^n - 1) / 2 := by
  sorry

end buttons_pattern_total_buttons_sum_l1896_189687


namespace inequality_solution_l1896_189617

theorem inequality_solution (x : ℝ) :
  (2 * x^2 - 4 * x - 70 > 0) ∧ (x ≠ -2) ∧ (x ≠ 0) ↔ (x < -5 ∨ x > 7) :=
by
  sorry

end inequality_solution_l1896_189617


namespace williams_land_percentage_l1896_189660

variable (total_tax : ℕ) (williams_tax : ℕ)

theorem williams_land_percentage (h1 : total_tax = 3840) (h2 : williams_tax = 480) : 
  (williams_tax:ℚ) / (total_tax:ℚ) * 100 = 12.5 := 
  sorry

end williams_land_percentage_l1896_189660


namespace lia_quadrilateral_rod_count_l1896_189652

theorem lia_quadrilateral_rod_count :
  let rods := {n : ℕ | 1 ≤ n ∧ n ≤ 40}
  let selected_rods := {5, 10, 20}
  let remaining_rods := rods \ selected_rods
  rod_count = 26 ∧ (∃ d ∈ remaining_rods, 
    (5 + 10 + 20) > d ∧ (10 + 20 + d) > 5 ∧ (5 + 20 + d) > 10 ∧ (5 + 10 + d) > 20)
:=
sorry

end lia_quadrilateral_rod_count_l1896_189652


namespace sum_lengths_DE_EF_equals_9_l1896_189644

variable (AB BC FA : ℝ)
variable (area_ABCDEF : ℝ)
variable (DE EF : ℝ)

theorem sum_lengths_DE_EF_equals_9 (h1 : area_ABCDEF = 52) (h2 : AB = 8) (h3 : BC = 9) (h4 : FA = 5)
  (h5 : AB * BC - area_ABCDEF = DE * EF) (h6 : BC - FA = DE) : DE + EF = 9 := 
by 
  sorry

end sum_lengths_DE_EF_equals_9_l1896_189644


namespace ellipse_area_l1896_189649

theorem ellipse_area :
  ∃ a b : ℝ, 
    (∀ x y : ℝ, (x^2 - 2 * x + 9 * y^2 + 18 * y + 16 = 0) → 
    (a = 2 ∧ b = (2 / 3) ∧ (π * a * b = 4 * π / 3))) :=
sorry

end ellipse_area_l1896_189649


namespace robert_turns_30_after_2_years_l1896_189665

variable (P R : ℕ) -- P for Patrick's age, R for Robert's age
variable (h1 : P = 14) -- Patrick is 14 years old now
variable (h2 : P * 2 = R) -- Patrick is half the age of Robert

theorem robert_turns_30_after_2_years : R + 2 = 30 :=
by
  -- Here should be the proof, but for now we skip it with sorry
  sorry

end robert_turns_30_after_2_years_l1896_189665


namespace find_x_l1896_189648

-- Introducing the main theorem
theorem find_x (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) (x : ℝ) (h_x : 0 < x) : 
  let r := (4 * a) ^ (4 * b)
  let y := x ^ 2
  r = a ^ b * y → 
  x = 16 ^ b * a ^ (1.5 * b) :=
by
  sorry

end find_x_l1896_189648


namespace complex_square_simplification_l1896_189686

theorem complex_square_simplification (i : ℂ) (h : i^2 = -1) : (4 - 3 * i)^2 = 7 - 24 * i :=
by {
  sorry
}

end complex_square_simplification_l1896_189686


namespace andrei_kolya_ages_l1896_189636

def sum_of_digits (n : ℕ) : ℕ :=
  (n % 10) + ((n / 10) % 10) + ((n / 100) % 10) + (n / 1000)

theorem andrei_kolya_ages :
  ∃ (y1 y2 : ℕ), (sum_of_digits y1 = 2021 - y1) ∧ (sum_of_digits y2 = 2021 - y2) ∧ (y1 ≠ y2) ∧ ((2022 - y1 = 8 ∧ 2022 - y2 = 26) ∨ (2022 - y1 = 26 ∧ 2022 - y2 = 8)) :=
by
  sorry

end andrei_kolya_ages_l1896_189636


namespace exists_x0_f_leq_one_tenth_l1896_189664

noncomputable def f (x a : ℝ) : ℝ := x^2 + (Real.log (3*x))^2 - 2*a*x - 6*a*(Real.log (3*x)) + 10*a^2

theorem exists_x0_f_leq_one_tenth (a : ℝ) : (∃ x₀, f x₀ a ≤ 1/10) ↔ a = 1/30 := by
  sorry

end exists_x0_f_leq_one_tenth_l1896_189664


namespace linear_eq_m_value_l1896_189699

theorem linear_eq_m_value (x m : ℝ) (h : 2 * x + m = 5) (hx : x = 1) : m = 3 :=
by
  -- Here we would carry out the proof steps
  sorry

end linear_eq_m_value_l1896_189699


namespace time_to_sweep_one_room_l1896_189689

theorem time_to_sweep_one_room (x : ℕ) :
  (10 * x) = (2 * 9 + 6 * 2) → x = 3 := by
  sorry

end time_to_sweep_one_room_l1896_189689


namespace masha_problem_l1896_189693

noncomputable def sum_arithmetic_series (a l n : ℕ) : ℕ :=
  (n * (a + l)) / 2

theorem masha_problem : 
  let a_even := 372
  let l_even := 506
  let n_even := 67
  let a_odd := 373
  let l_odd := 505
  let n_odd := 68
  let S_even := sum_arithmetic_series a_even l_even n_even
  let S_odd := sum_arithmetic_series a_odd l_odd n_odd
  S_odd - S_even = 439 := 
by sorry

end masha_problem_l1896_189693


namespace major_axis_endpoints_of_ellipse_l1896_189668

theorem major_axis_endpoints_of_ellipse :
  ∀ x y, 6 * x^2 + y^2 = 6 ↔ (x = 0 ∧ (y = -Real.sqrt 6 ∨ y = Real.sqrt 6)) :=
by
  -- Proof
  sorry

end major_axis_endpoints_of_ellipse_l1896_189668


namespace derivative_of_f_l1896_189619

variable (x : ℝ)
def f (x : ℝ) := (5 * x - 4) ^ 3

theorem derivative_of_f :
  (deriv f x) = 15 * (5 * x - 4) ^ 2 :=
sorry

end derivative_of_f_l1896_189619


namespace pig_problem_l1896_189621

theorem pig_problem (x y : ℕ) (h₁ : y - 100 = 100 * x) (h₂ : y = 90 * x) : x = 10 ∧ y = 900 := 
by
  sorry

end pig_problem_l1896_189621


namespace time_needed_to_gather_remaining_flowers_l1896_189613

-- conditions
def classmates : ℕ := 30
def time_per_flower : ℕ := 10
def gathering_time : ℕ := 2 * 60
def lost_flowers : ℕ := 3

-- question and proof goal
theorem time_needed_to_gather_remaining_flowers : 
  let flowers_needed := classmates - ((gathering_time / time_per_flower) - lost_flowers)
  flowers_needed * time_per_flower = 210 :=
by
  sorry

end time_needed_to_gather_remaining_flowers_l1896_189613


namespace least_number_to_add_1054_23_l1896_189651

def least_number_to_add (n k : ℕ) : ℕ :=
  let remainder := n % k
  if remainder = 0 then 0 else k - remainder

theorem least_number_to_add_1054_23 : least_number_to_add 1054 23 = 4 :=
by
  -- This is a placeholder for the actual proof
  sorry

end least_number_to_add_1054_23_l1896_189651


namespace num_intersections_circle_line_eq_two_l1896_189691

theorem num_intersections_circle_line_eq_two :
  ∃ (points : Finset (ℝ × ℝ)), {p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 = 25 ∧ p.1 = 3} = points ∧ points.card = 2 :=
by
  sorry

end num_intersections_circle_line_eq_two_l1896_189691


namespace new_team_average_weight_is_113_l1896_189677

-- Defining the given constants and conditions
def original_players := 7
def original_average_weight := 121 
def weight_new_player1 := 110 
def weight_new_player2 := 60 

-- Definition to calculate the new average weight
def new_average_weight : ℕ :=
  let original_total_weight := original_players * original_average_weight
  let new_total_weight := original_total_weight + weight_new_player1 + weight_new_player2
  let new_total_players := original_players + 2
  new_total_weight / new_total_players

-- Statement to prove
theorem new_team_average_weight_is_113 : new_average_weight = 113 :=
sorry

end new_team_average_weight_is_113_l1896_189677


namespace central_angle_relation_l1896_189639

theorem central_angle_relation
  (R L : ℝ)
  (α : ℝ)
  (r l β : ℝ)
  (h1 : r = 0.5 * R)
  (h2 : l = 1.5 * L)
  (h3 : L = R * α)
  (h4 : l = r * β) : 
  β = 3 * α :=
by
  sorry

end central_angle_relation_l1896_189639


namespace valid_transformation_b_l1896_189610

theorem valid_transformation_b (a b : ℚ) : ((-a - b) / (a + b) = -1) := sorry

end valid_transformation_b_l1896_189610


namespace tan_315_eq_neg1_l1896_189635

theorem tan_315_eq_neg1 : Real.tan (315 * Real.pi / 180) = -1 := 
by
  sorry

end tan_315_eq_neg1_l1896_189635


namespace exists_term_not_of_form_l1896_189683

theorem exists_term_not_of_form (a d : ℕ) (h_seq : ∀ i j : ℕ, (i < 40 ∧ j < 40 ∧ i ≠ j) → a + i * d ≠ a + j * d)
  (pos_a : a > 0) (pos_d : d > 0) 
  : ∃ h : ℕ, h < 40 ∧ ¬ ∃ k l : ℕ, a + h * d = 2^k + 3^l :=
by {
  sorry
}

end exists_term_not_of_form_l1896_189683


namespace player_c_wins_l1896_189647

theorem player_c_wins :
  ∀ (A_wins A_losses B_wins B_losses C_losses C_wins : ℕ),
  A_wins = 4 →
  A_losses = 2 →
  B_wins = 3 →
  B_losses = 3 →
  C_losses = 3 →
  A_wins + B_wins + C_wins = A_losses + B_losses + C_losses →
  C_wins = 2 :=
by
  intros A_wins A_losses B_wins B_losses C_losses C_wins
  sorry

end player_c_wins_l1896_189647


namespace total_material_ordered_l1896_189695

theorem total_material_ordered :
  12.468 + 4.6278 + 7.9101 + 8.3103 + 5.6327 = 38.9499 :=
by
  sorry

end total_material_ordered_l1896_189695


namespace max_happy_times_l1896_189663

theorem max_happy_times (weights : Fin 2021 → ℝ) (unique_mass : Function.Injective weights) : 
  ∃ max_happy : Nat, max_happy = 673 :=
by
  sorry

end max_happy_times_l1896_189663


namespace exists_h_not_divisible_l1896_189624

theorem exists_h_not_divisible : ∃ (h : ℝ), ∀ (n : ℕ), ¬ (⌊h * 1969^n⌋ % ⌊h * 1969^(n-1)⌋ = 0) :=
by
  sorry

end exists_h_not_divisible_l1896_189624


namespace find_v_3_l1896_189694

def u (x : ℤ) : ℤ := 4 * x - 9

def v (z : ℤ) : ℤ := z^2 + 4 * z - 1

theorem find_v_3 : v 3 = 20 := by
  sorry

end find_v_3_l1896_189694


namespace toy_cost_price_l1896_189630

theorem toy_cost_price (C : ℕ) (h : 18 * C + 3 * C = 25200) : C = 1200 := by
  -- The proof is not required
  sorry

end toy_cost_price_l1896_189630


namespace simplify_expression_l1896_189603

theorem simplify_expression (x : ℕ) (h : x = 100) :
  (x + 1) * (x - 1) + x * (2 - x) + (x - 1) ^ 2 = 10000 := by
  sorry

end simplify_expression_l1896_189603


namespace frog_ends_on_horizontal_side_l1896_189638

-- Definitions for the problem conditions
def frog_jump_probability (x y : ℤ) : ℚ := sorry

-- Main theorem statement based on the identified question and correct answer
theorem frog_ends_on_horizontal_side :
  frog_jump_probability 2 3 = 13 / 14 :=
sorry

end frog_ends_on_horizontal_side_l1896_189638


namespace not_both_267_and_269_non_standard_l1896_189670

def G : ℤ → ℤ := sorry

def exists_x_ne_c (G : ℤ → ℤ) : Prop :=
  ∀ c : ℤ, ∃ x : ℤ, G x ≠ c

def non_standard (G : ℤ → ℤ) (a : ℤ) : Prop :=
  ∀ x : ℤ, G x = G (a - x)

theorem not_both_267_and_269_non_standard (G : ℤ → ℤ)
  (h1 : exists_x_ne_c G) :
  ¬ (non_standard G 267 ∧ non_standard G 269) :=
sorry

end not_both_267_and_269_non_standard_l1896_189670


namespace band_row_lengths_l1896_189672

theorem band_row_lengths (x y : ℕ) :
  (x * y = 90) → (5 ≤ x ∧ x ≤ 20) → (Even y) → False :=
by sorry

end band_row_lengths_l1896_189672


namespace tank_third_dimension_l1896_189667

theorem tank_third_dimension (x : ℕ) (h1 : 4 * 5 = 20) (h2 : 2 * (4 * x) + 2 * (5 * x) = 18 * x) (h3 : (40 + 18 * x) * 20 = 1520) :
  x = 2 :=
by
  sorry

end tank_third_dimension_l1896_189667


namespace parallel_lines_no_intersection_l1896_189600

theorem parallel_lines_no_intersection (k : ℝ) :
  (∀ t s : ℝ, 
    ∃ (a b : ℝ), (a, b) = (1, -3) + t • (2, 5) ∧ (a, b) = (-4, 2) + s • (3, k)) → 
  k = 15 / 2 :=
by
  sorry

end parallel_lines_no_intersection_l1896_189600


namespace extreme_value_f_g_gt_one_l1896_189669

noncomputable def f (x : ℝ) : ℝ := Real.exp x - x - 1
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := Real.exp x * (a * x + x * Real.cos x + 1)

theorem extreme_value_f : f 0 = 0 :=
by
  sorry

theorem g_gt_one (a : ℝ) (h : a > -1) (x : ℝ) (hx : 0 < x) (hx1 : x < 1) : g x a > 1 :=
by
  sorry

end extreme_value_f_g_gt_one_l1896_189669
