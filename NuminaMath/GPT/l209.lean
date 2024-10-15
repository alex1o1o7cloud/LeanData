import Mathlib

namespace NUMINAMATH_GPT_inscribed_sphere_radius_l209_20923

theorem inscribed_sphere_radius (a α : ℝ) (hα : 0 < α ∧ α < 2 * Real.pi) :
  ∃ (ρ : ℝ), ρ = a * (1 - Real.cos α) / (2 * Real.sqrt (1 + Real.cos α) * (1 + Real.sqrt (- Real.cos α))) :=
  sorry

end NUMINAMATH_GPT_inscribed_sphere_radius_l209_20923


namespace NUMINAMATH_GPT_determine_number_of_solutions_l209_20984

noncomputable def num_solutions_eq : Prop :=
  let f (x : ℝ) := (3 * x ^ 2 - 15 * x) / (x ^ 2 - 7 * x + 10)
  let g (x : ℝ) := x - 4
  ∃ S : Finset ℝ, 
    (∀ x ∈ S, (x ≠ 2 ∧ x ≠ 5) ∧ f x = g x) ∧
    S.card = 2

theorem determine_number_of_solutions : num_solutions_eq :=
  by
  sorry

end NUMINAMATH_GPT_determine_number_of_solutions_l209_20984


namespace NUMINAMATH_GPT_determine_f_value_l209_20952

noncomputable def f (t : ℝ) : ℝ := t^2 + 2

theorem determine_f_value : f 3 = 11 := by
  sorry

end NUMINAMATH_GPT_determine_f_value_l209_20952


namespace NUMINAMATH_GPT_proof_expression_C_equals_negative_one_l209_20913

def A : ℤ := abs (-1)
def B : ℤ := -(-1)
def C : ℤ := -(1^2)
def D : ℤ := (-1)^2

theorem proof_expression_C_equals_negative_one : C = -1 :=
by 
  sorry

end NUMINAMATH_GPT_proof_expression_C_equals_negative_one_l209_20913


namespace NUMINAMATH_GPT_num_subsets_with_even_is_24_l209_20945

def A : Set ℕ := {1, 2, 3, 4, 5}
def odd_subsets_count : ℕ := 2^3

theorem num_subsets_with_even_is_24 : 
  let total_subsets := 2^5
  total_subsets - odd_subsets_count = 24 := by
  sorry

end NUMINAMATH_GPT_num_subsets_with_even_is_24_l209_20945


namespace NUMINAMATH_GPT_initial_jelly_beans_l209_20985

theorem initial_jelly_beans (total_children : ℕ) (percentage : ℕ) (jelly_per_child : ℕ) (remaining_jelly : ℕ) :
  (percentage = 80) → (total_children = 40) → (jelly_per_child = 2) → (remaining_jelly = 36) →
  (total_children * percentage / 100 * jelly_per_child + remaining_jelly = 100) :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_initial_jelly_beans_l209_20985


namespace NUMINAMATH_GPT_find_a4_b4_l209_20903

theorem find_a4_b4 :
  ∃ (a₁ a₂ a₃ a₄ b₁ b₂ b₃ b₄ : ℝ),
    a₁ * b₁ + a₂ * b₃ = 1 ∧
    a₁ * b₂ + a₂ * b₄ = 0 ∧
    a₃ * b₁ + a₄ * b₃ = 0 ∧
    a₃ * b₂ + a₄ * b₄ = 1 ∧
    a₂ * b₃ = 7 ∧
    a₄ * b₄ = -6 :=
by
  sorry

end NUMINAMATH_GPT_find_a4_b4_l209_20903


namespace NUMINAMATH_GPT_max_value_F_l209_20955

noncomputable def f (x : ℝ) : ℝ := 1 - 2 * x^2
noncomputable def g (x : ℝ) : ℝ := x^2 - 2 * x

noncomputable def F (x : ℝ) : ℝ :=
if f x ≥ g x then f x else g x

theorem max_value_F : ∃ x : ℝ, ∀ y : ℝ, F y ≤ F x ∧ F x = 7 / 9 := 
sorry

end NUMINAMATH_GPT_max_value_F_l209_20955


namespace NUMINAMATH_GPT_problem1_problem2_l209_20981

theorem problem1 (n : ℕ) : 2 ≤ (1 + 1 / n) ^ n ∧ (1 + 1 / n) ^ n < 3 :=
sorry

theorem problem2 (n : ℕ) : (n / 3) ^ n < n! :=
sorry

end NUMINAMATH_GPT_problem1_problem2_l209_20981


namespace NUMINAMATH_GPT_hyperbola_eccentricity_l209_20904

theorem hyperbola_eccentricity (a b c e : ℝ)
  (h1 : a > 0)
  (h2 : b > 0)
  (h3 : a > b)
  (h4 : c = 3 * b) 
  (h5 : c * c = a * a + b * b)
  (h6 : e = c / a) :
  e = 3 * Real.sqrt 2 / 4 :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_eccentricity_l209_20904


namespace NUMINAMATH_GPT_calculate_expression_l209_20999

theorem calculate_expression :
  2⁻¹ + (3 - Real.pi)^0 + abs (2 * Real.sqrt 3 - Real.sqrt 2) + 2 * Real.cos (Real.pi / 4) - Real.sqrt 12 = 3 / 2 :=
sorry

end NUMINAMATH_GPT_calculate_expression_l209_20999


namespace NUMINAMATH_GPT_jane_mean_score_l209_20989

-- Define Jane's scores as a list
def jane_scores : List ℕ := [95, 88, 94, 86, 92, 91]

-- Define the total number of quizzes
def total_quizzes : ℕ := 6

-- Define the sum of Jane's scores
def sum_scores : ℕ := 95 + 88 + 94 + 86 + 92 + 91

-- Define the mean score calculation
def mean_score : ℕ := sum_scores / total_quizzes

-- The theorem to state Jane's mean score
theorem jane_mean_score : mean_score = 91 := by
  -- This theorem statement correctly reflects the mathematical problem provided.
  sorry

end NUMINAMATH_GPT_jane_mean_score_l209_20989


namespace NUMINAMATH_GPT_solutions_to_cube_eq_27_l209_20991

theorem solutions_to_cube_eq_27 (z : ℂ) : 
  (z^3 = 27) ↔ (z = 3 ∨ z = (Complex.mk (-3 / 2) (3 * Real.sqrt 3 / 2)) ∨ z = (Complex.mk (-3 / 2) (-3 * Real.sqrt 3 / 2))) :=
by sorry

end NUMINAMATH_GPT_solutions_to_cube_eq_27_l209_20991


namespace NUMINAMATH_GPT_cube_sum_divisible_by_six_l209_20907

theorem cube_sum_divisible_by_six
  (a b c : ℤ)
  (h1 : 6 ∣ (a^2 + b^2 + c^2))
  (h2 : 3 ∣ (a * b + b * c + c * a))
  : 6 ∣ (a^3 + b^3 + c^3) := 
sorry

end NUMINAMATH_GPT_cube_sum_divisible_by_six_l209_20907


namespace NUMINAMATH_GPT_time_elephants_l209_20900

def total_time := 130
def time_seals := 13
def time_penguins := 8 * time_seals

theorem time_elephants : total_time - (time_seals + time_penguins) = 13 :=
by
  sorry

end NUMINAMATH_GPT_time_elephants_l209_20900


namespace NUMINAMATH_GPT_range_of_m_l209_20957

theorem range_of_m (m x y : ℝ) 
  (h1 : x + y = -1) 
  (h2 : 5 * x + 2 * y = 6 * m + 7) 
  (h3 : 2 * x - y < 19) : 
  m < 3 / 2 := 
sorry

end NUMINAMATH_GPT_range_of_m_l209_20957


namespace NUMINAMATH_GPT_total_score_is_248_l209_20925

def geography_score : ℕ := 50
def math_score : ℕ := 70
def english_score : ℕ := 66

def history_score : ℕ := (geography_score + math_score + english_score) / 3

theorem total_score_is_248 : geography_score + math_score + english_score + history_score = 248 := by
  -- proofs go here
  sorry

end NUMINAMATH_GPT_total_score_is_248_l209_20925


namespace NUMINAMATH_GPT_C_increases_with_n_l209_20937

variables (n e R r : ℝ)
variables (h_pos_e : e > 0) (h_pos_R : R > 0)
variables (h_pos_r : r > 0) (h_R_nr : R > n * r)
noncomputable def C : ℝ := (e * n) / (R - n * r)

theorem C_increases_with_n (h_pos_e : e > 0) (h_pos_R : R > 0)
(h_pos_r : r > 0) (h_R_nr : R > n * r) (hn1 hn2 : ℝ)
(h_inequality : hn1 < hn2) : 
((e*hn1) / (R - hn1*r)) < ((e*hn2) / (R - hn2*r)) :=
by sorry

end NUMINAMATH_GPT_C_increases_with_n_l209_20937


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_l209_20963

theorem necessary_but_not_sufficient (x : ℝ) : (x > 1 → x > 2) = (false) ∧ (x > 2 → x > 1) = (true) := by
  sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_l209_20963


namespace NUMINAMATH_GPT_sum_abcd_eq_16_l209_20936

variable (a b c d : ℝ)

def cond1 : Prop := a^2 + b^2 + c^2 + d^2 = 250
def cond2 : Prop := a * b + b * c + c * a + a * d + b * d + c * d = 3

theorem sum_abcd_eq_16 (h1 : cond1 a b c d) (h2 : cond2 a b c d) : a + b + c + d = 16 := 
by 
  sorry

end NUMINAMATH_GPT_sum_abcd_eq_16_l209_20936


namespace NUMINAMATH_GPT_find_missing_number_l209_20997

theorem find_missing_number (x : ℤ) (h : (4 + 3) + (8 - x - 1) = 11) : x = 3 :=
sorry

end NUMINAMATH_GPT_find_missing_number_l209_20997


namespace NUMINAMATH_GPT_days_to_complete_work_l209_20969

-- Let's define the conditions as Lean definitions based on the problem.

variables (P D : ℕ)
noncomputable def original_work := P * D
noncomputable def half_work_by_double_people := 2 * P * 3

-- Here is our theorem statement
theorem days_to_complete_work : original_work P D = 2 * half_work_by_double_people P :=
by sorry

end NUMINAMATH_GPT_days_to_complete_work_l209_20969


namespace NUMINAMATH_GPT_total_sand_arrived_l209_20926

theorem total_sand_arrived :
  let truck1_carry := 4.1
  let truck1_loss := 2.4
  let truck2_carry := 5.7
  let truck2_loss := 3.6
  let truck3_carry := 8.2
  let truck3_loss := 1.9
  (truck1_carry - truck1_loss) + 
  (truck2_carry - truck2_loss) + 
  (truck3_carry - truck3_loss) = 10.1 :=
by
  sorry

end NUMINAMATH_GPT_total_sand_arrived_l209_20926


namespace NUMINAMATH_GPT_simplified_value_l209_20954

noncomputable def simplify_expression : ℝ :=
  1 / (Real.log (3) / Real.log (20) + 1) + 
  1 / (Real.log (4) / Real.log (15) + 1) + 
  1 / (Real.log (7) / Real.log (12) + 1)

theorem simplified_value : simplify_expression = 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_simplified_value_l209_20954


namespace NUMINAMATH_GPT_alex_min_additional_coins_l209_20947

theorem alex_min_additional_coins (n m k : ℕ) (h_n : n = 15) (h_m : m = 120) :
  k = 0 ↔ m = (n * (n + 1)) / 2 :=
by
  sorry

end NUMINAMATH_GPT_alex_min_additional_coins_l209_20947


namespace NUMINAMATH_GPT_N_square_solutions_l209_20960

theorem N_square_solutions :
  ∀ N : ℕ, (N > 0 → ∃ k : ℕ, 2^N - 2 * N = k^2) → (N = 1 ∨ N = 2) :=
by
  sorry

end NUMINAMATH_GPT_N_square_solutions_l209_20960


namespace NUMINAMATH_GPT_point_in_third_quadrant_cos_sin_l209_20928

theorem point_in_third_quadrant_cos_sin (P : ℝ × ℝ) (hP : P = (Real.cos (2009 * Real.pi / 180), Real.sin (2009 * Real.pi / 180))) :
  P.1 < 0 ∧ P.2 < 0 :=
by
  sorry

end NUMINAMATH_GPT_point_in_third_quadrant_cos_sin_l209_20928


namespace NUMINAMATH_GPT_fixed_point_l209_20942

theorem fixed_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  2 + a^(1-1) = 3 :=
by
  sorry

end NUMINAMATH_GPT_fixed_point_l209_20942


namespace NUMINAMATH_GPT_pet_store_total_birds_l209_20912

def total_birds_in_pet_store (bird_cages parrots_per_cage parakeets_per_cage : ℕ) : ℕ :=
  bird_cages * (parrots_per_cage + parakeets_per_cage)

theorem pet_store_total_birds :
  total_birds_in_pet_store 4 8 2 = 40 :=
by
  sorry

end NUMINAMATH_GPT_pet_store_total_birds_l209_20912


namespace NUMINAMATH_GPT_tiffany_lives_after_game_l209_20975

/-- Tiffany's initial number of lives -/
def initial_lives : ℕ := 43

/-- Lives Tiffany loses in the hard part of the game -/
def lost_lives : ℕ := 14

/-- Lives Tiffany gains in the next level -/
def gained_lives : ℕ := 27

/-- Calculate the total lives Tiffany has after losing and gaining lives -/
def total_lives : ℕ := (initial_lives - lost_lives) + gained_lives

-- Prove that the total number of lives Tiffany has is 56
theorem tiffany_lives_after_game : total_lives = 56 := by
  -- This is where the proof would go
  sorry

end NUMINAMATH_GPT_tiffany_lives_after_game_l209_20975


namespace NUMINAMATH_GPT_polynomial_identity_sum_l209_20933

theorem polynomial_identity_sum (A B C D : ℤ) (h : (x + 3) * (4 * x^2 - 2 * x + 7) = A * x^3 + B * x^2 + C * x + D) : 
  A + B + C + D = 36 := 
by 
  sorry

end NUMINAMATH_GPT_polynomial_identity_sum_l209_20933


namespace NUMINAMATH_GPT_sum_of_x_coords_f_eq_3_l209_20962

section
-- Define the piecewise linear function, splits into five segments
def f1 (x : ℝ) : ℝ := 2 * x + 6
def f2 (x : ℝ) : ℝ := -2 * x + 6
def f3 (x : ℝ) : ℝ := 2 * x + 2
def f4 (x : ℝ) : ℝ := -x + 2
def f5 (x : ℝ) : ℝ := 2 * x - 4

-- The sum of x-coordinates where f(x) = 3
noncomputable def x_coords_3_sum : ℝ := -1.5 + 0.5 + 3.5

-- Goal statement
theorem sum_of_x_coords_f_eq_3 : -1.5 + 0.5 + 3.5 = 2.5 := by
  sorry
end

end NUMINAMATH_GPT_sum_of_x_coords_f_eq_3_l209_20962


namespace NUMINAMATH_GPT_vertex_of_parabola_is_max_and_correct_l209_20902

theorem vertex_of_parabola_is_max_and_correct (x y : ℝ) (h : y = -3 * x^2 + 6 * x + 1) :
  (x, y) = (1, 4) ∧ ∃ ε > 0, ∀ z : ℝ, abs (z - x) < ε → y ≥ -3 * z^2 + 6 * z + 1 :=
by
  sorry

end NUMINAMATH_GPT_vertex_of_parabola_is_max_and_correct_l209_20902


namespace NUMINAMATH_GPT_philip_school_trip_days_l209_20986

-- Define the distances for the trips
def school_trip_one_way_miles : ℝ := 2.5
def market_trip_one_way_miles : ℝ := 2

-- Define the number of times he makes the trips in a day and in a week
def school_round_trips_per_day : ℕ := 2
def market_round_trips_per_week : ℕ := 1

-- Define the total mileage in a week
def weekly_mileage : ℕ := 44

-- Define the equation based on the given conditions
def weekly_school_trip_distance (d : ℕ) : ℝ :=
  (school_trip_one_way_miles * 2 * school_round_trips_per_day) * d

def weekly_market_trip_distance : ℝ :=
  (market_trip_one_way_miles * 2) * market_round_trips_per_week

-- Define the main theorem to be proved
theorem philip_school_trip_days :
  ∃ d : ℕ, weekly_school_trip_distance d + weekly_market_trip_distance = weekly_mileage ∧ d = 4 :=
by
  sorry

end NUMINAMATH_GPT_philip_school_trip_days_l209_20986


namespace NUMINAMATH_GPT_interest_rate_l209_20948

theorem interest_rate (SI P T R : ℝ) (h1 : SI = 100) (h2 : P = 500) (h3 : T = 4) (h4 : SI = (P * R * T) / 100) :
  R = 5 :=
by
  sorry

end NUMINAMATH_GPT_interest_rate_l209_20948


namespace NUMINAMATH_GPT_exponent_logarithm_simplifies_l209_20968

theorem exponent_logarithm_simplifies :
  (1/2 : ℝ) ^ (Real.log 3 / Real.log 2 - 1) = 2 / 3 :=
by sorry

end NUMINAMATH_GPT_exponent_logarithm_simplifies_l209_20968


namespace NUMINAMATH_GPT_plane_equation_l209_20980

theorem plane_equation (x y z : ℝ)
  (h₁ : ∃ t : ℝ, x = 2 * t + 1 ∧ y = -3 * t ∧ z = 3 - t)
  (h₂ : ∃ (t₁ t₂ : ℝ), 4 * t₁ + 5 * t₂ - 3 = 0 ∧ 2 * t₁ + t₂ + 2 * t₂ = 0) : 
  2*x - y + 7*z - 23 = 0 :=
sorry

end NUMINAMATH_GPT_plane_equation_l209_20980


namespace NUMINAMATH_GPT_factor_x10_minus_1296_l209_20994

theorem factor_x10_minus_1296 (x : ℝ) : (x^10 - 1296) = (x^5 + 36) * (x^5 - 36) :=
  by
  sorry

end NUMINAMATH_GPT_factor_x10_minus_1296_l209_20994


namespace NUMINAMATH_GPT_range_of_x_l209_20971

theorem range_of_x (x : ℝ) (h : 4 * x - 12 ≥ 0) : x ≥ 3 := 
sorry

end NUMINAMATH_GPT_range_of_x_l209_20971


namespace NUMINAMATH_GPT_middle_number_is_11_l209_20922

theorem middle_number_is_11 (a b c : ℕ) (h1 : a + b = 18) (h2 : a + c = 22) (h3 : b + c = 26) (h4 : c - a = 10) :
  b = 11 :=
by
  sorry

end NUMINAMATH_GPT_middle_number_is_11_l209_20922


namespace NUMINAMATH_GPT_charlie_older_than_bobby_by_three_l209_20909

variable (J C B x : ℕ)

def jenny_older_charlie_by_five (J C : ℕ) := J = C + 5
def charlie_age_when_jenny_twice_bobby_age (C x : ℕ) := C + x = 11
def jenny_twice_bobby (J B x : ℕ) := J + x = 2 * (B + x)

theorem charlie_older_than_bobby_by_three
  (h1 : jenny_older_charlie_by_five J C)
  (h2 : charlie_age_when_jenny_twice_bobby_age C x)
  (h3 : jenny_twice_bobby J B x) :
  (C = B + 3) :=
by
  sorry

end NUMINAMATH_GPT_charlie_older_than_bobby_by_three_l209_20909


namespace NUMINAMATH_GPT_molecular_weight_correct_l209_20920

-- Define atomic weights
def atomic_weight_aluminium : Float := 26.98
def atomic_weight_oxygen : Float := 16.00
def atomic_weight_hydrogen : Float := 1.01
def atomic_weight_silicon : Float := 28.09
def atomic_weight_nitrogen : Float := 14.01

-- Define the number of each atom in the compound
def num_aluminium : Nat := 2
def num_oxygen : Nat := 6
def num_hydrogen : Nat := 3
def num_silicon : Nat := 2
def num_nitrogen : Nat := 4

-- Calculate the expected molecular weight
def expected_molecular_weight : Float :=
  (2 * atomic_weight_aluminium) + 
  (6 * atomic_weight_oxygen) + 
  (3 * atomic_weight_hydrogen) + 
  (2 * atomic_weight_silicon) + 
  (4 * atomic_weight_nitrogen)

-- Prove that the expected molecular weight is 265.21 amu
theorem molecular_weight_correct : expected_molecular_weight = 265.21 :=
by
  sorry

end NUMINAMATH_GPT_molecular_weight_correct_l209_20920


namespace NUMINAMATH_GPT_dad_steps_l209_20931

theorem dad_steps (total_steps_Masha_Yasha : ℕ) (h1 : ∀ d_steps m_steps, d_steps = 3 * m_steps) 
  (h2 : ∀ m_steps y_steps, m_steps = 3 * (y_steps / 5)) 
  (h3 : total_steps_Masha_Yasha = 400) : 
  ∃ d_steps : ℕ, d_steps = 90 :=
by
  sorry

end NUMINAMATH_GPT_dad_steps_l209_20931


namespace NUMINAMATH_GPT_washes_per_bottle_l209_20961

def bottle_cost : ℝ := 4.0
def total_weeks : ℕ := 20
def total_cost : ℝ := 20.0

theorem washes_per_bottle : (total_weeks / (total_cost / bottle_cost)) = 4 := by
  sorry

end NUMINAMATH_GPT_washes_per_bottle_l209_20961


namespace NUMINAMATH_GPT_n_n_plus_1_divisible_by_2_l209_20958

theorem n_n_plus_1_divisible_by_2 (n : ℤ) (h1 : 1 ≤ n) (h2 : n ≤ 99) : (n * (n + 1)) % 2 = 0 := 
sorry

end NUMINAMATH_GPT_n_n_plus_1_divisible_by_2_l209_20958


namespace NUMINAMATH_GPT_factor_expression_l209_20946

theorem factor_expression (y : ℝ) : 3 * y^2 - 12 = 3 * (y + 2) * (y - 2) := 
by
  sorry

end NUMINAMATH_GPT_factor_expression_l209_20946


namespace NUMINAMATH_GPT_find_t_l209_20993

theorem find_t (t : ℝ) (h : (1 / (t+3) + 3 * t / (t+3) - 4 / (t+3)) = 5) : t = -9 :=
by
  sorry

end NUMINAMATH_GPT_find_t_l209_20993


namespace NUMINAMATH_GPT_total_profit_l209_20949

-- Definitions based on the conditions
def tom_investment : ℝ := 30000
def tom_duration : ℝ := 12
def jose_investment : ℝ := 45000
def jose_duration : ℝ := 10
def jose_share_profit : ℝ := 25000

-- Theorem statement
theorem total_profit (tom_investment tom_duration jose_investment jose_duration jose_share_profit : ℝ) :
  (jose_share_profit / (jose_investment * jose_duration / (tom_investment * tom_duration + jose_investment * jose_duration)) = 5 / 9) →
  ∃ P : ℝ, P = 45000 :=
by
  sorry

end NUMINAMATH_GPT_total_profit_l209_20949


namespace NUMINAMATH_GPT_find_f2_l209_20979

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := x^5 + a * x^3 + b * x - 8

theorem find_f2 (a b : ℝ) (h : f (-2) a b = 10) : f 2 a b = -26 := 
by
  sorry

end NUMINAMATH_GPT_find_f2_l209_20979


namespace NUMINAMATH_GPT_max_value_l209_20932

noncomputable def satisfies_equation (x y : ℝ) : Prop :=
  x + 4 * y - x * y = 0

theorem max_value (x y : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_eq : satisfies_equation x y) :
  ∃ m, m = (4 / (x + y)) ∧ m ≤ (4 / 9) :=
by
  sorry

end NUMINAMATH_GPT_max_value_l209_20932


namespace NUMINAMATH_GPT_part_one_part_two_l209_20995

noncomputable def f (x a: ℝ) : ℝ := abs (x - 1) + abs (x + a)
noncomputable def g (a : ℝ) : ℝ := a^2 - a - 2

theorem part_one (x : ℝ) : f x 3 > g 3 + 2 ↔ x < -4 ∨ x > 2 := by
  sorry

theorem part_two (a : ℝ) :
  (∀ x : ℝ, -a ≤ x ∧ x ≤ 1 → f x a ≤ g a) ↔ a ≥ 3 := by
  sorry

end NUMINAMATH_GPT_part_one_part_two_l209_20995


namespace NUMINAMATH_GPT_sequence_first_equals_last_four_l209_20965

theorem sequence_first_equals_last_four (n : ℕ) (S : ℕ → ℕ) (h_length : ∀ i < n, S i = 0 ∨ S i = 1)
  (h_condition : ∀ (i j : ℕ), 1 ≤ i ∧ i < j ∧ j ≤ n - 4 → 
    (S i = S j ∧ S (i + 1) = S (j + 1) ∧ S (i + 2) = S (j + 2) ∧ S (i + 3) = S (j + 3) ∧ S (i + 4) = S (j + 4)) → false) :
  S 1 = S (n - 3) ∧ S 2 = S (n - 2) ∧ S 3 = S (n - 1) ∧ S 4 = S n :=
sorry

end NUMINAMATH_GPT_sequence_first_equals_last_four_l209_20965


namespace NUMINAMATH_GPT_favorite_number_l209_20951

theorem favorite_number (S₁ S₂ S₃ : ℕ) (total_sum : ℕ) (adjacent_sum : ℕ) 
  (h₁ : S₁ = 8) (h₂ : S₂ = 14) (h₃ : S₃ = 12) 
  (h_total_sum : total_sum = 17) 
  (h_adjacent_sum : adjacent_sum = 12) : 
  ∃ x : ℕ, x = 5 := 
by 
  sorry

end NUMINAMATH_GPT_favorite_number_l209_20951


namespace NUMINAMATH_GPT_edward_cards_l209_20983

noncomputable def num_cards_each_binder : ℝ := (7496.5 + 27.7) / 23
noncomputable def num_cards_fewer_binder : ℝ := num_cards_each_binder - 27.7

theorem edward_cards : 
  (⌊num_cards_each_binder + 0.5⌋ = 327) ∧ (⌊num_cards_fewer_binder + 0.5⌋ = 299) :=
by
  sorry

end NUMINAMATH_GPT_edward_cards_l209_20983


namespace NUMINAMATH_GPT_fraction_value_l209_20901

theorem fraction_value (x y : ℕ) (hx : x = 3) (hy : y = 4) :
  (1 / (y : ℚ) / (1 / (x : ℚ))) = 3 / 4 :=
by
  rw [hx, hy]
  norm_num

end NUMINAMATH_GPT_fraction_value_l209_20901


namespace NUMINAMATH_GPT_arithmetic_sequence_general_term_l209_20917

noncomputable def an (a_1 d : ℤ) (n : ℕ) : ℤ := a_1 + (n - 1) * d
def bn (a_n : ℤ) : ℚ := (1 / 2)^a_n

theorem arithmetic_sequence_general_term
  (a_n : ℕ → ℤ)
  (b_1 b_2 b_3 : ℚ)
  (a_1 d : ℤ)
  (h_seq : ∀ n, a_n n = a_1 + (n - 1) * d)
  (h_b1 : b_1 = (1 / 2)^(a_n 1))
  (h_b2 : b_2 = (1 / 2)^(a_n 2))
  (h_b3 : b_3 = (1 / 2)^(a_n 3))
  (h_sum : b_1 + b_2 + b_3 = 21 / 8)
  (h_prod : b_1 * b_2 * b_3 = 1 / 8)
  : (∀ n, a_n n = 2 * n - 3) ∨ (∀ n, a_n n = 5 - 2 * n) :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_general_term_l209_20917


namespace NUMINAMATH_GPT_no_real_roots_eq_xsq_abs_x_plus_1_eq_0_l209_20914

theorem no_real_roots_eq_xsq_abs_x_plus_1_eq_0 :
  ¬ ∃ x : ℝ, x^2 + abs x + 1 = 0 :=
by
  sorry

end NUMINAMATH_GPT_no_real_roots_eq_xsq_abs_x_plus_1_eq_0_l209_20914


namespace NUMINAMATH_GPT_max_cookies_Andy_could_have_eaten_l209_20966

theorem max_cookies_Andy_could_have_eaten (cookies : ℕ) (Andy Alexa : ℕ) 
  (h1 : cookies = 24) 
  (h2 : Alexa = k * Andy) 
  (h3 : k > 0) 
  (h4 : Andy + Alexa = cookies) 
  : Andy ≤ 12 := 
sorry

end NUMINAMATH_GPT_max_cookies_Andy_could_have_eaten_l209_20966


namespace NUMINAMATH_GPT_empty_subset_singleton_l209_20970

theorem empty_subset_singleton : (∅ ⊆ ({0} : Set ℕ)) = true :=
by sorry

end NUMINAMATH_GPT_empty_subset_singleton_l209_20970


namespace NUMINAMATH_GPT_cells_after_one_week_l209_20929

theorem cells_after_one_week : (3 ^ 7) = 2187 :=
by sorry

end NUMINAMATH_GPT_cells_after_one_week_l209_20929


namespace NUMINAMATH_GPT_simplify_and_evaluate_expression_l209_20972

-- Define the condition
def condition (x y : ℝ) := (x - 2) ^ 2 + |y + 1| = 0

-- Define the expression
def expression (x y : ℝ) := 3 * x ^ 2 * y - (2 * x ^ 2 * y - 3 * (2 * x * y - x ^ 2 * y) + 5 * x * y)

-- State the theorem
theorem simplify_and_evaluate_expression (x y : ℝ) (h : condition x y) : expression x y = 6 :=
by
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_expression_l209_20972


namespace NUMINAMATH_GPT_balanced_number_example_l209_20988

/--
A number is balanced if it is a three-digit number, all digits are different,
and it equals the sum of all possible two-digit numbers composed from its different digits.
-/
def isBalanced (n : ℕ) : Prop :=
  (n / 100 ≠ (n / 10) % 10) ∧ (n / 100 ≠ n % 10) ∧ ((n / 10) % 10 ≠ n % 10) ∧
  (n = (10 * (n / 100) + (n / 10) % 10) + (10 * (n / 100) + n % 10) +
    (10 * ((n / 10) % 10) + n / 100) + (10 * ((n / 10) % 10) + n % 10) +
    (10 * (n % 10) + n / 100) + (10 * (n % 10) + ((n / 10) % 10)))

theorem balanced_number_example : isBalanced 132 :=
  sorry

end NUMINAMATH_GPT_balanced_number_example_l209_20988


namespace NUMINAMATH_GPT_discriminant_of_quadratic_l209_20935

theorem discriminant_of_quadratic :
  let a := (5 : ℚ)
  let b := (5 + 1/5 : ℚ)
  let c := (1/5 : ℚ)
  let Δ := b^2 - 4 * a * c
  Δ = 576 / 25 :=
by
  sorry

end NUMINAMATH_GPT_discriminant_of_quadratic_l209_20935


namespace NUMINAMATH_GPT_find_four_real_numbers_l209_20910

theorem find_four_real_numbers
  (x1 x2 x3 x4 : ℝ)
  (h1 : x1 + x2 * x3 * x4 = 2)
  (h2 : x2 + x1 * x3 * x4 = 2)
  (h3 : x3 + x1 * x2 * x4 = 2)
  (h4 : x4 + x1 * x2 * x3 = 2) :
  (x1 = 1 ∧ x2 = 1 ∧ x3 = 1 ∧ x4 = 1) ∨
  (x1 = -1 ∧ x2 = -1 ∧ x3 = -1 ∧ x4 = 3) :=
sorry

end NUMINAMATH_GPT_find_four_real_numbers_l209_20910


namespace NUMINAMATH_GPT_number_of_valid_pairs_l209_20982

theorem number_of_valid_pairs :
  (∃ (count : ℕ), count = 280 ∧
    (∃ (m n : ℕ),
      1 ≤ m ∧ m ≤ 2899 ∧
      5^n < 2^m ∧ 2^m < 2^(m+3) ∧ 2^(m+3) < 5^(n+1))) :=
sorry

end NUMINAMATH_GPT_number_of_valid_pairs_l209_20982


namespace NUMINAMATH_GPT_no_monochromatic_ap_11_l209_20921

open Function

theorem no_monochromatic_ap_11 :
  ∃ (coloring : ℕ → Fin 4), (∀ a r : ℕ, r > 0 → a + 10 * r ≤ 2014 → ∃ i j : ℕ, (i ≠ j) ∧ (a + i * r < 1 ∨ a + j * r > 2014 ∨ coloring (a + i * r) ≠ coloring (a + j * r))) :=
sorry

end NUMINAMATH_GPT_no_monochromatic_ap_11_l209_20921


namespace NUMINAMATH_GPT_five_digit_number_unique_nonzero_l209_20978

theorem five_digit_number_unique_nonzero (a b c d e : ℕ) (h1 : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e) (h2 : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ e ≠ 0) (h3 : (100 * a + 10 * b + c) * 7 = 100 * c + 10 * d + e) : a = 1 ∧ b = 2 ∧ c = 9 ∧ d = 4 ∧ e = 6 :=
by
  sorry

end NUMINAMATH_GPT_five_digit_number_unique_nonzero_l209_20978


namespace NUMINAMATH_GPT_tourist_group_people_count_l209_20976

def large_room_people := 3
def small_room_people := 2
def small_rooms_rented := 1
def people_in_small_room := small_rooms_rented * small_room_people

theorem tourist_group_people_count : 
  ∀ x : ℕ, x ≥ 1 ∧ (x + small_rooms_rented) = (people_in_small_room + x * large_room_people) → 
  (people_in_small_room + x * large_room_people) = 5 := 
  by
  sorry

end NUMINAMATH_GPT_tourist_group_people_count_l209_20976


namespace NUMINAMATH_GPT_total_coins_l209_20996

def piles_of_quarters : Nat := 5
def piles_of_dimes : Nat := 5
def coins_per_pile : Nat := 3

theorem total_coins :
  (piles_of_quarters * coins_per_pile) + (piles_of_dimes * coins_per_pile) = 30 := by
  sorry

end NUMINAMATH_GPT_total_coins_l209_20996


namespace NUMINAMATH_GPT_smallest_number_divisible_l209_20919

/-- The smallest number which, when diminished by 20, is divisible by 15, 30, 45, and 60 --/
theorem smallest_number_divisible (n : ℕ) (h : ∀ k : ℕ, n - 20 = k * Int.lcm 15 (Int.lcm 30 (Int.lcm 45 60))) : n = 200 :=
sorry

end NUMINAMATH_GPT_smallest_number_divisible_l209_20919


namespace NUMINAMATH_GPT_volleyball_club_girls_l209_20990

theorem volleyball_club_girls (B G : ℕ) (h1 : B + G = 32) (h2 : (1 / 3 : ℝ) * G + ↑B = 20) : G = 18 := 
by
  sorry

end NUMINAMATH_GPT_volleyball_club_girls_l209_20990


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_l209_20930

theorem necessary_but_not_sufficient (a b : ℝ) : 
  (b < -1 → |a| + |b| > 1) ∧ (∃ a b : ℝ, |a| + |b| > 1 ∧ b >= -1) :=
by
  sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_l209_20930


namespace NUMINAMATH_GPT_find_sum_l209_20916

variables (x y : ℝ)

def condition1 : Prop := x^3 - 3 * x^2 + 5 * x = 1
def condition2 : Prop := y^3 - 3 * y^2 + 5 * y = 5

theorem find_sum : condition1 x → condition2 y → x + y = 2 := 
by 
  sorry -- The proof goes here

end NUMINAMATH_GPT_find_sum_l209_20916


namespace NUMINAMATH_GPT_find_m_l209_20974

-- Definitions of the given vectors a, b, and c
def vec_a (m : ℝ) : ℝ × ℝ := (1, m)
def vec_b : ℝ × ℝ := (2, 5)
def vec_c (m : ℝ) : ℝ × ℝ := (m, 3)

-- Definition of vector addition and subtraction
def vec_add (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 + v.1, u.2 + v.2)
def vec_sub (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 - v.1, u.2 - v.2)

-- Parallel vectors condition: the ratio of their components must be equal
def parallel (u v : ℝ × ℝ) : Prop :=
  u.1 * v.2 = u.2 * v.1

-- The main theorem stating the desired result
theorem find_m (m : ℝ) :
  parallel (vec_add (vec_a m) (vec_c m)) (vec_sub (vec_a m) vec_b) ↔ 
  m = (3 + Real.sqrt 17) / 2 ∨ m = (3 - Real.sqrt 17) / 2 :=
by
  sorry

end NUMINAMATH_GPT_find_m_l209_20974


namespace NUMINAMATH_GPT_hotel_fee_original_flat_fee_l209_20943

theorem hotel_fee_original_flat_fee
  (f n : ℝ)
  (H1 : 0.85 * (f + 3 * n) = 210)
  (H2 : f + 6 * n = 400) :
  f = 94.12 :=
by
  -- Sorry is used to indicate that the proof is not provided
  sorry

end NUMINAMATH_GPT_hotel_fee_original_flat_fee_l209_20943


namespace NUMINAMATH_GPT_find_ratio_of_constants_l209_20977

theorem find_ratio_of_constants (x y c d : ℚ) (hx : x ≠ 0) (hy : y ≠ 0) (hd : d ≠ 0) 
  (h₁ : 8 * x - 6 * y = c) (h₂ : 12 * y - 18 * x = d) : c / d = -4 / 9 := 
sorry

end NUMINAMATH_GPT_find_ratio_of_constants_l209_20977


namespace NUMINAMATH_GPT_solve_equation_l209_20973

theorem solve_equation :
  ∀ y : ℤ, 4 * (y - 1) = 1 - 3 * (y - 3) → y = 2 :=
by
  intros y h
  sorry

end NUMINAMATH_GPT_solve_equation_l209_20973


namespace NUMINAMATH_GPT_cost_of_4_bags_of_ice_l209_20939

theorem cost_of_4_bags_of_ice (
  cost_per_2_bags : ℝ := 1.46
) 
  (h : cost_per_2_bags / 2 = 0.73)
  :
  4 * (cost_per_2_bags / 2) = 2.92 :=
by 
  sorry

end NUMINAMATH_GPT_cost_of_4_bags_of_ice_l209_20939


namespace NUMINAMATH_GPT_smallest_common_multiple_l209_20967

theorem smallest_common_multiple (n : ℕ) (h8 : n % 8 = 0) (h15 : n % 15 = 0) : n = 120 :=
sorry

end NUMINAMATH_GPT_smallest_common_multiple_l209_20967


namespace NUMINAMATH_GPT_gina_college_expenses_l209_20987

theorem gina_college_expenses
  (credits : ℕ)
  (cost_per_credit : ℕ)
  (num_textbooks : ℕ)
  (cost_per_textbook : ℕ)
  (facilities_fee : ℕ)
  (H_credits : credits = 14)
  (H_cost_per_credit : cost_per_credit = 450)
  (H_num_textbooks : num_textbooks = 5)
  (H_cost_per_textbook : cost_per_textbook = 120)
  (H_facilities_fee : facilities_fee = 200)
  : (credits * cost_per_credit) + (num_textbooks * cost_per_textbook) + facilities_fee = 7100 := by
  sorry

end NUMINAMATH_GPT_gina_college_expenses_l209_20987


namespace NUMINAMATH_GPT_shirt_cost_l209_20956

def george_initial_money : ℕ := 100
def total_spent_on_clothes (initial_money remaining_money : ℕ) : ℕ := initial_money - remaining_money
def socks_cost : ℕ := 11
def remaining_money_after_purchase : ℕ := 65

theorem shirt_cost
  (initial_money : ℕ)
  (remaining_money : ℕ)
  (total_spent : ℕ)
  (socks_cost : ℕ)
  (remaining_money_after_purchase : ℕ) :
  initial_money = 100 →
  remaining_money = 65 →
  total_spent = initial_money - remaining_money →
  total_spent = 35 →
  socks_cost = 11 →
  remaining_money_after_purchase = remaining_money →
  (total_spent - socks_cost = 24) :=
by
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h2, h4] at *
  exact sorry

end NUMINAMATH_GPT_shirt_cost_l209_20956


namespace NUMINAMATH_GPT_hexagon_angle_sum_l209_20959

theorem hexagon_angle_sum 
  (mA mB mC x y : ℝ)
  (hA : mA = 34)
  (hB : mB = 80)
  (hC : mC = 30)
  (hx' : x = 36 - y) : x + y = 36 :=
by
  sorry

end NUMINAMATH_GPT_hexagon_angle_sum_l209_20959


namespace NUMINAMATH_GPT_deer_families_initial_count_l209_20938

theorem deer_families_initial_count (stayed moved_out : ℕ) (h_stayed : stayed = 45) (h_moved_out : moved_out = 34) :
  stayed + moved_out = 79 :=
by
  sorry

end NUMINAMATH_GPT_deer_families_initial_count_l209_20938


namespace NUMINAMATH_GPT_original_pencils_l209_20992

-- Define the conditions
def pencils_added : ℕ := 30
def total_pencils_now : ℕ := 71

-- Define the theorem to prove the original number of pencils
theorem original_pencils (original_pencils : ℕ) :
  total_pencils_now = original_pencils + pencils_added → original_pencils = 41 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_original_pencils_l209_20992


namespace NUMINAMATH_GPT_average_runs_l209_20918

theorem average_runs (games : ℕ) (runs1 matches1 runs2 matches2 runs3 matches3 : ℕ)
  (h1 : runs1 = 1) 
  (h2 : matches1 = 1) 
  (h3 : runs2 = 4) 
  (h4 : matches2 = 2)
  (h5 : runs3 = 5) 
  (h6 : matches3 = 3) 
  (h_games : games = matches1 + matches2 + matches3) :
  (runs1 * matches1 + runs2 * matches2 + runs3 * matches3) / games = 4 :=
by
  sorry

end NUMINAMATH_GPT_average_runs_l209_20918


namespace NUMINAMATH_GPT_find_angle_l209_20934

-- Given definitions:
def complement (α : ℝ) : ℝ := 90 - α
def supplement (α : ℝ) : ℝ := 180 - α

-- Condition:
def condition (α : ℝ) : Prop :=
  supplement α = 3 * complement α + 10

-- Statement to prove:
theorem find_angle (α : ℝ) (h : condition α) : α = 50 :=
sorry

end NUMINAMATH_GPT_find_angle_l209_20934


namespace NUMINAMATH_GPT_total_turnover_in_first_quarter_l209_20941

theorem total_turnover_in_first_quarter (x : ℝ) : 
  200 + 200 * (1 + x) + 200 * (1 + x) ^ 2 = 1000 :=
sorry

end NUMINAMATH_GPT_total_turnover_in_first_quarter_l209_20941


namespace NUMINAMATH_GPT_regular_polygons_enclosing_hexagon_l209_20924

theorem regular_polygons_enclosing_hexagon (m n : ℕ) 
  (hm : m = 6)
  (h_exterior_angle_central : 180 - ((m - 2) * 180 / m) = 60)
  (h_exterior_angle_enclosing : 2 * 60 = 120): 
  n = 3 := sorry

end NUMINAMATH_GPT_regular_polygons_enclosing_hexagon_l209_20924


namespace NUMINAMATH_GPT_sheep_count_l209_20911

theorem sheep_count (S H : ℕ) (h1 : S / H = 2 / 7) (h2 : H * 230 = 12880) : S = 16 :=
by 
  -- Lean proof goes here
  sorry

end NUMINAMATH_GPT_sheep_count_l209_20911


namespace NUMINAMATH_GPT_calculation_l209_20908

theorem calculation (a b c d e : ℤ)
  (h1 : a = (-4)^6)
  (h2 : b = 4^4)
  (h3 : c = 2^5)
  (h4 : d = 7^2)
  (h5 : e = (a / b) + c - d) :
  e = -1 := by
  sorry

end NUMINAMATH_GPT_calculation_l209_20908


namespace NUMINAMATH_GPT_cube_side_length_l209_20905

theorem cube_side_length (n : ℕ) (h1 : 6 * n^2 / (6 * n^3) = 1 / 3) : n = 3 :=
by
  sorry

end NUMINAMATH_GPT_cube_side_length_l209_20905


namespace NUMINAMATH_GPT_value_of_x_for_real_y_l209_20915

theorem value_of_x_for_real_y (x y : ℝ) (h : 4 * y^2 + 2 * x * y + |x| + 8 = 0) :
  (x ≤ -10) ∨ (x ≥ 10) :=
sorry

end NUMINAMATH_GPT_value_of_x_for_real_y_l209_20915


namespace NUMINAMATH_GPT_series_product_solution_l209_20953

theorem series_product_solution (y : ℚ) :
  ( (∑' n, (1 / 2) * (1 / 3) ^ n) * (∑' n, (1 / 3) * (-1 / 3) ^ n) ) = ∑' n, (1 / y) ^ (n + 1) → y = 19 / 3 :=
by
  sorry

end NUMINAMATH_GPT_series_product_solution_l209_20953


namespace NUMINAMATH_GPT_triangle_side_length_b_l209_20964

theorem triangle_side_length_b (a b c : ℝ) (A B C : ℝ)
  (hB : B = 30) 
  (h_area : 1/2 * a * c * Real.sin (B * Real.pi/180) = 3/2) 
  (h_sine : Real.sin (A * Real.pi/180) + Real.sin (C * Real.pi/180) = 2 * Real.sin (B * Real.pi/180)) :
  b = Real.sqrt 3 + 1 :=
by
  sorry

end NUMINAMATH_GPT_triangle_side_length_b_l209_20964


namespace NUMINAMATH_GPT_longer_subsegment_of_YZ_l209_20998

/-- In triangle XYZ with sides in the ratio 3:4:5, and side YZ being 12 cm.
    The angle bisector XW divides side YZ into segments YW and ZW.
    Prove that the length of ZW is 48/7 cm. --/
theorem longer_subsegment_of_YZ (YZ : ℝ) (hYZ : YZ = 12)
    (XY XZ : ℝ) (hRatio : XY / XZ = 3 / 4) : 
    ∃ ZW : ℝ, ZW = 48 / 7 :=
by
  -- We would provide proof here
  sorry

end NUMINAMATH_GPT_longer_subsegment_of_YZ_l209_20998


namespace NUMINAMATH_GPT_ticket_cost_per_ride_l209_20950

theorem ticket_cost_per_ride (total_tickets : ℕ) (spent_tickets : ℕ) (rides : ℕ) (remaining_tickets : ℕ) (cost_per_ride : ℕ) 
  (h1 : total_tickets = 79) 
  (h2 : spent_tickets = 23) 
  (h3 : rides = 8) 
  (h4 : remaining_tickets = total_tickets - spent_tickets) 
  (h5 : remaining_tickets / rides = cost_per_ride) 
  : cost_per_ride = 7 := 
sorry

end NUMINAMATH_GPT_ticket_cost_per_ride_l209_20950


namespace NUMINAMATH_GPT_o_l209_20927

theorem o'hara_triple_example (a b x : ℕ) (h₁ : a = 49) (h₂ : b = 16) (h₃ : x = (Int.sqrt a).toNat + (Int.sqrt b).toNat) : x = 11 := 
by
  sorry

end NUMINAMATH_GPT_o_l209_20927


namespace NUMINAMATH_GPT_tile_count_difference_l209_20906

theorem tile_count_difference :
  let red_initial := 15
  let yellow_initial := 10
  let yellow_added := 18
  let yellow_total := yellow_initial + yellow_added
  let red_total := red_initial
  yellow_total - red_total = 13 :=
by
  sorry

end NUMINAMATH_GPT_tile_count_difference_l209_20906


namespace NUMINAMATH_GPT_handshakes_max_number_of_men_l209_20940

theorem handshakes_max_number_of_men (n : ℕ) (h: n * (n - 1) / 2 = 190) : n = 20 :=
sorry

end NUMINAMATH_GPT_handshakes_max_number_of_men_l209_20940


namespace NUMINAMATH_GPT_rewrite_sum_l209_20944

theorem rewrite_sum (S_b S : ℕ → ℕ) (n S_1 : ℕ) (a b c : ℕ) :
  b = 4 → (a + b + c) / 3 = 6 →
  S_b n = b * n + (a + b + c) / 3 * (S n - n * S_1) →
  S_b n = 4 * n + 6 * (S n - n * S_1) := by
sorry

end NUMINAMATH_GPT_rewrite_sum_l209_20944
