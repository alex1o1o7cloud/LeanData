import Mathlib

namespace equation_of_line_l679_67903

theorem equation_of_line :
  ∃ m : ℝ, ∀ x y : ℝ, (y = m * x - m ∧ (m = 2 ∧ x = 1 ∧ y = 0)) ∧ 
  ∀ x : ℝ, ¬(4 * x^2 - (m * x - m)^2 - 8 * x = 12) → m = 2 → y = 2 * x - 2 :=
by sorry

end equation_of_line_l679_67903


namespace expected_faces_rolled_six_times_l679_67933

-- Define a random variable indicating appearance of a particular face
noncomputable def ζi (n : ℕ): ℝ := if n > 0 then 1 - (5 / 6) ^ 6 else 0

-- Define the expected number of distinct faces
noncomputable def expected_distinct_faces : ℝ := 6 * ζi 1

theorem expected_faces_rolled_six_times :
  expected_distinct_faces = (6 ^ 6 - 5 ^ 6) / 6 ^ 5 :=
by
  -- Here we would provide the proof
  sorry

end expected_faces_rolled_six_times_l679_67933


namespace mixed_doubles_teams_l679_67987

theorem mixed_doubles_teams (males females : ℕ) (hm : males = 6) (hf : females = 7) : (males * females) = 42 :=
by
  sorry

end mixed_doubles_teams_l679_67987


namespace even_product_implies_sum_of_squares_odd_product_implies_no_sum_of_squares_l679_67980

theorem even_product_implies_sum_of_squares (a b : ℕ) (h : ∃ (a b : ℕ), a * b % 2 = 0 → ∃ (c d : ℕ), a^2 + b^2 + c^2 = d^2) : 
  ∃ (c d : ℕ), a^2 + b^2 + c^2 = d^2 :=
sorry

theorem odd_product_implies_no_sum_of_squares (a b : ℕ) (h : ∃ (a b : ℕ), a * b % 2 ≠ 0 → ¬∃ (c d : ℕ), a^2 + b^2 + c^2 = d^2) : 
  ¬∃ (c d : ℕ), a^2 + b^2 + c^2 = d^2 :=
sorry

end even_product_implies_sum_of_squares_odd_product_implies_no_sum_of_squares_l679_67980


namespace points_coplanar_if_and_only_if_b_neg1_l679_67900

/-- Points (0, 0, 0), (1, b, 0), (0, 1, b), (b, 0, 1) are coplanar if and only if b = -1. --/
theorem points_coplanar_if_and_only_if_b_neg1 (a b : ℝ) :
  (∃ u v w : ℝ, (u, v, w) = (0, 0, 0) ∨ (u, v, w) = (1, b, 0) ∨ (u, v, w) = (0, 1, b) ∨ (u, v, w) = (b, 0, 1)) →
  (b = -1) :=
sorry

end points_coplanar_if_and_only_if_b_neg1_l679_67900


namespace ratio_of_fractions_l679_67996

-- Given conditions
variables {x y : ℚ}
variables (h1 : 5 * x = 3 * y) (h2 : x * y ≠ 0)

-- Assertion to be proved
theorem ratio_of_fractions (h1 : 5 * x = 3 * y) (h2 : x * y ≠ 0) :
  (1 / 5 * x) / (1 / 6 * y) = 18 / 25 :=
sorry

end ratio_of_fractions_l679_67996


namespace find_t_correct_l679_67947

theorem find_t_correct : 
  ∃ t : ℝ, (∀ x : ℝ, (3 * x^2 - 4 * x + 5) * (5 * x^2 + t * x + 15) = 15 * x^4 - 47 * x^3 + 115 * x^2 - 110 * x + 75) ∧ t = -10 :=
sorry

end find_t_correct_l679_67947


namespace least_positive_integer_division_conditions_l679_67912

theorem least_positive_integer_division_conditions :
  ∃ M : ℤ, M > 0 ∧
  M % 11 = 10 ∧
  M % 12 = 11 ∧
  M % 13 = 12 ∧
  M % 14 = 13 ∧
  M = 30029 := 
by
  sorry

end least_positive_integer_division_conditions_l679_67912


namespace tallest_giraffe_height_l679_67983

theorem tallest_giraffe_height :
  ∃ (height : ℕ), height = 96 ∧ (height = 68 + 28) := by
  sorry

end tallest_giraffe_height_l679_67983


namespace problem_b_problem_d_l679_67954

variable (x y t : ℝ)

def condition_curve (t : ℝ) : Prop :=
  ∃ C : ℝ × ℝ → Prop, ∀ x y : ℝ, C (x, y) ↔ (x^2 / (5 - t) + y^2 / (t - 1) = 1)

theorem problem_b (h1 : t < 1) : condition_curve t → ∃ (C : ℝ × ℝ → Prop), (∀ x y, C (x, y) ↔ x^2 / (5 - t) + y^2 / (t - 1) = 1) → ¬(5 - t) < 0 ∧ (t - 1) < 0 := 
sorry

theorem problem_d (h1 : 3 < t) (h2 : t < 5) (h3 : condition_curve t) : ∃ (C : ℝ × ℝ → Prop), (∀ x y, C (x, y) ↔ x^2 / (5 - t) + y^2 / (t - 1) = 1) → 0 < (t - 1) ∧ (t - 1) > (5 - t) := 
sorry

end problem_b_problem_d_l679_67954


namespace market_survey_l679_67935

theorem market_survey (X Y Z : ℕ) (h1 : X / Y = 3)
  (h2 : X / Z = 2 / 3) (h3 : X = 60) : X + Y + Z = 170 :=
by
  sorry

end market_survey_l679_67935


namespace arc_length_of_sector_l679_67923

theorem arc_length_of_sector (r : ℝ) (θ : ℝ) (h_r : r = 2) (h_θ : θ = π / 3) :
  l = r * θ := by
  sorry

end arc_length_of_sector_l679_67923


namespace find_center_of_circle_l679_67941

noncomputable def center_of_circle (θ ρ : ℝ) : Prop :=
  ρ = (1 : ℝ) ∧ θ = (-Real.pi / (3 : ℝ))

theorem find_center_of_circle (θ ρ : ℝ) (h : ρ = Real.cos θ - Real.sqrt 3 * Real.sin θ) :
  center_of_circle θ ρ := by
  sorry

end find_center_of_circle_l679_67941


namespace union_sets_l679_67989

-- Given sets A and B
def A : Set ℕ := {1, 2, 3, 4}
def B : Set ℕ := {2, 4, 5}

theorem union_sets : A ∪ B = {1, 2, 3, 4, 5} := by
  sorry

end union_sets_l679_67989


namespace missing_fraction_is_correct_l679_67971

def sum_of_fractions (x : ℚ) : Prop :=
  (1/3 : ℚ) + (1/2) + (-5/6) + (1/5) + (1/4) + (-9/20) + x = (45/100 : ℚ)

theorem missing_fraction_is_correct : sum_of_fractions (27/60 : ℚ) :=
  by sorry

end missing_fraction_is_correct_l679_67971


namespace evaporation_period_l679_67963

theorem evaporation_period
  (initial_amount : ℚ)
  (evaporation_rate : ℚ)
  (percentage_evaporated : ℚ)
  (actual_days : ℚ)
  (h_initial : initial_amount = 10)
  (h_evap_rate : evaporation_rate = 0.007)
  (h_percentage : percentage_evaporated = 3.5000000000000004)
  (h_days : actual_days = (percentage_evaporated / 100) * initial_amount / evaporation_rate):
  actual_days = 50 := by
  sorry

end evaporation_period_l679_67963


namespace isosceles_triangle_largest_angle_l679_67956

theorem isosceles_triangle_largest_angle (A B C : ℝ) (h_triangle : A + B + C = 180) 
  (h_isosceles : A = B) (h_given_angle : A = 40) : C = 100 :=
by
  sorry

end isosceles_triangle_largest_angle_l679_67956


namespace find_m_and_n_l679_67940

theorem find_m_and_n (x y m n : ℝ) 
  (h1 : 5 * x - 2 * y = 3) 
  (h2 : m * x + 5 * y = 4) 
  (h3 : x - 4 * y = -3) 
  (h4 : 5 * x + n * y = 1) :
  m = -1 ∧ n = -4 :=
by
  sorry

end find_m_and_n_l679_67940


namespace polynomials_equal_l679_67977

noncomputable def P : ℝ → ℝ := sorry -- assume P is a nonconstant polynomial
noncomputable def Q : ℝ → ℝ := sorry -- assume Q is a nonconstant polynomial

axiom floor_eq_for_all_y (y : ℝ) : ⌊P y⌋ = ⌊Q y⌋

theorem polynomials_equal (x : ℝ) : P x = Q x :=
by
  sorry

end polynomials_equal_l679_67977


namespace integer_solutions_to_equation_l679_67909

theorem integer_solutions_to_equation :
  ∀ (a b c : ℤ), a^2 + b^2 + c^2 = a^2 * b^2 → a = 0 ∧ b = 0 ∧ c = 0 :=
by
  sorry

end integer_solutions_to_equation_l679_67909


namespace AC_eq_200_l679_67948

theorem AC_eq_200 (A B C : ℕ) (h1 : A + B + C = 500) (h2 : B + C = 330) (h3 : C = 30) : A + C = 200 := by
  sorry

end AC_eq_200_l679_67948


namespace b_divisible_by_8_l679_67975

theorem b_divisible_by_8 (b : ℕ) (h_even: ∃ k : ℕ, b = 2 * k) (h_square: ∃ n : ℕ, n > 1 ∧ ∃ m : ℕ, (b ^ n - 1) / (b - 1) = m ^ 2) : b % 8 = 0 := 
by
  sorry

end b_divisible_by_8_l679_67975


namespace at_least_one_not_lt_one_l679_67908

theorem at_least_one_not_lt_one (a b c : ℝ) (h : a + b + c = 3) : ¬ (a < 1 ∧ b < 1 ∧ c < 1) :=
by
  sorry

end at_least_one_not_lt_one_l679_67908


namespace quadratic_roots_sum_square_l679_67930

theorem quadratic_roots_sum_square (u v : ℝ) 
  (h1 : u^2 - 5*u + 3 = 0) (h2 : v^2 - 5*v + 3 = 0) 
  (h3 : u ≠ v) : u^2 + v^2 + u*v = 22 := 
by
  sorry

end quadratic_roots_sum_square_l679_67930


namespace value_of_f_at_6_l679_67973

-- The condition that f is an odd function
def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f (x)

-- The condition that f(x + 2) = -f(x)
def periodic_sign_flip (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (x + 2) = -f (x)

-- The theorem statement
theorem value_of_f_at_6 (f : ℝ → ℝ) (h1 : is_odd_function f) (h2 : periodic_sign_flip f) : f 6 = 0 :=
sorry

end value_of_f_at_6_l679_67973


namespace quotient_equivalence_l679_67920

variable (N H J : ℝ)

theorem quotient_equivalence
  (h1 : N / H = 1.2)
  (h2 : H / J = 5 / 6) :
  N / J = 1 := by
  sorry

end quotient_equivalence_l679_67920


namespace bodhi_yacht_animals_l679_67993

def total_animals (cows foxes zebras sheep : ℕ) : ℕ :=
  cows + foxes + zebras + sheep

theorem bodhi_yacht_animals :
  ∀ (cows foxes sheep : ℕ), foxes = 15 → cows = 20 → sheep = 20 → total_animals cows foxes (3 * foxes) sheep = 100 :=
by
  intros cows foxes sheep h1 h2 h3
  rw [h1, h2, h3]
  show total_animals 20 15 (3 * 15) 20 = 100
  sorry

end bodhi_yacht_animals_l679_67993


namespace man_speed_in_still_water_l679_67955

theorem man_speed_in_still_water (V_m V_s : ℝ) 
  (h1 : V_m + V_s = 8)
  (h2 : V_m - V_s = 6) : 
  V_m = 7 := 
by
  sorry

end man_speed_in_still_water_l679_67955


namespace price_reduction_proof_l679_67911

theorem price_reduction_proof (x : ℝ) : 256 * (1 - x) ^ 2 = 196 :=
sorry

end price_reduction_proof_l679_67911


namespace completing_square_transformation_l679_67936

theorem completing_square_transformation : ∀ x : ℝ, x^2 - 4 * x - 7 = 0 → (x - 2)^2 = 11 :=
by
  intros x h
  sorry

end completing_square_transformation_l679_67936


namespace age_ratio_l679_67917

-- Conditions
def DeepakPresentAge := 27
def RahulAgeAfterSixYears := 42
def YearsToReach42 := 6

-- The theorem to prove the ratio of their ages
theorem age_ratio (R D : ℕ) (hR : R + YearsToReach42 = RahulAgeAfterSixYears) (hD : D = DeepakPresentAge) : R / D = 4 / 3 := by
  sorry

end age_ratio_l679_67917


namespace sin_double_angle_l679_67927

theorem sin_double_angle (α : ℝ) (h1 : Real.tan α = 2) (h2 : 0 < α ∧ α < Real.pi / 2) : 
  Real.sin (2 * α) = 4 / 5 :=
sorry

end sin_double_angle_l679_67927


namespace triangle_area_is_32_5_l679_67928

-- Define points A, B, and C
def A : ℝ × ℝ := (-3, 4)
def B : ℝ × ℝ := (1, 7)
def C : ℝ × ℝ := (4, -1)

-- Calculate the area directly using the determinant method for the area of a triangle given by coordinates
def area_triangle (A B C : ℝ × ℝ) : ℝ :=
  0.5 * abs (
    A.1 * (B.2 - C.2) +
    B.1 * (C.2 - A.2) +
    C.1 * (A.2 - B.2)
  )

-- Define the statement to be proved
theorem triangle_area_is_32_5 : area_triangle A B C = 32.5 := 
  by
  -- proof to be filled in
  sorry

end triangle_area_is_32_5_l679_67928


namespace not_necessarily_a_squared_lt_b_squared_l679_67970
-- Import the necessary library

-- Define the variables and the condition
variables {a b : ℝ}
axiom h : a < b

-- The theorem statement that needs to be proved/disproved
theorem not_necessarily_a_squared_lt_b_squared (a b : ℝ) (h : a < b) : ¬ (a^2 < b^2) :=
sorry

end not_necessarily_a_squared_lt_b_squared_l679_67970


namespace first_month_sale_l679_67905

def sale_second_month : ℕ := 5744
def sale_third_month : ℕ := 5864
def sale_fourth_month : ℕ := 6122
def sale_fifth_month : ℕ := 6588
def sale_sixth_month : ℕ := 4916
def average_sale_six_months : ℕ := 5750

def expected_total_sales : ℕ := 6 * average_sale_six_months
def known_sales : ℕ := sale_second_month + sale_third_month + sale_fourth_month + sale_fifth_month

theorem first_month_sale :
  (expected_total_sales - (known_sales + sale_sixth_month)) = 5266 :=
by
  sorry

end first_month_sale_l679_67905


namespace arithmetic_sequence_length_l679_67966

theorem arithmetic_sequence_length :
  ∃ n : ℕ, ∀ (a_1 d a_n : ℤ), a_1 = -3 ∧ d = 4 ∧ a_n = 45 → n = 13 :=
by
  sorry

end arithmetic_sequence_length_l679_67966


namespace average_people_per_hour_l679_67901

-- Define the conditions
def people_moving : ℕ := 3000
def days : ℕ := 5
def hours_per_day : ℕ := 24
def total_hours : ℕ := days * hours_per_day

-- State the problem
theorem average_people_per_hour :
  people_moving / total_hours = 25 :=
by
  -- Proof goes here
  sorry

end average_people_per_hour_l679_67901


namespace maximum_value_ab_l679_67906

noncomputable def g (x : ℝ) : ℝ := 2 ^ x

theorem maximum_value_ab (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : g a * g b = 2) :
  ab ≤ (1 / 4) := sorry

end maximum_value_ab_l679_67906


namespace find_k_l679_67965

variables {r k : ℝ}
variables {O A B C D : EuclideanSpace ℝ (Fin 3)}

-- Points A, B, C, and D lie on a sphere centered at O with radius r
variables (hA : dist O A = r) (hB : dist O B = r) (hC : dist O C = r) (hD : dist O D = r)
-- The given vector equation
variables (h_eq : 4 • (A - O) - 3 • (B - O) + 6 • (C - O) + k • (D - O) = (0 : EuclideanSpace ℝ (Fin 3)))

theorem find_k (hA : dist O A = r) (hB : dist O B = r) (hC : dist O C = r) (hD : dist O D = r)
(h_eq : 4 • (A - O) - 3 • (B - O) + 6 • (C - O) + k • (D - O) = (0 : EuclideanSpace ℝ (Fin 3))) : 
k = -7 :=
sorry

end find_k_l679_67965


namespace sum_center_radius_eq_neg2_l679_67992

theorem sum_center_radius_eq_neg2 (c d s : ℝ) (h_eq : ∀ x y : ℝ, x^2 + 14 * x + y^2 - 8 * y = -64 ↔ (x + c)^2 + (y + d)^2 = s^2) :
  c + d + s = -2 :=
sorry

end sum_center_radius_eq_neg2_l679_67992


namespace max_xyz_l679_67937

theorem max_xyz (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) 
(h4 : (x * y) + 3 * z = (x + 3 * z) * (y + 3 * z)) 
: ∀ x y z, ∃ (a : ℝ), a = (x * y * z) ∧ a ≤ (1/81) :=
sorry

end max_xyz_l679_67937


namespace james_collects_15_gallons_per_inch_l679_67958

def rain_gallons_per_inch (G : ℝ) : Prop :=
  let monday_rain := 4
  let tuesday_rain := 3
  let price_per_gallon := 1.2
  let total_money := 126
  let total_rain := monday_rain + tuesday_rain
  (total_rain * G = total_money / price_per_gallon)

theorem james_collects_15_gallons_per_inch : rain_gallons_per_inch 15 :=
by
  -- This is the theorem statement; the proof is not required.
  sorry

end james_collects_15_gallons_per_inch_l679_67958


namespace arithmetic_progression_numbers_l679_67994

theorem arithmetic_progression_numbers :
  ∃ (a d : ℚ), (3 * (2 * a - d) = 2 * (a + d)) ∧ ((a - d) * (a + d) = (a - 2)^2) ∧
  ((a = 5 ∧ d = 4 ∧ ∃ b c : ℚ, b = (a - d) ∧ c = (a + d) ∧ b = 1 ∧ c = 9) 
   ∨ (a = 5 / 4 ∧ d = 1 ∧ ∃ b c : ℚ, b = (a - d) ∧ c = (a + d) ∧ b = 1 / 4 ∧ c = 9 / 4)) :=
by
  sorry

end arithmetic_progression_numbers_l679_67994


namespace tournament_games_l679_67926

theorem tournament_games (n : ℕ) (k : ℕ) (h_n : n = 30) (h_k : k = 5) : 
  (n * (n - 1) / 2) * k = 2175 := by
  sorry

end tournament_games_l679_67926


namespace area_ratio_equilateral_triangle_extension_l679_67976

variable (s : ℝ)

theorem area_ratio_equilateral_triangle_extension :
  (let A := (0, 0)
   let B := (s, 0)
   let C := (s / 2, s * (Real.sqrt 3 / 2))
   let A' := (0, -4 * s * (Real.sqrt 3 / 2))
   let B' := (3 * s, 0)
   let C' := (s / 2, s * (Real.sqrt 3 / 2) + 3 * s * (Real.sqrt 3 / 2))
   let area_ABC := (Real.sqrt 3 / 4) * s^2
   let area_A'B'C' := (Real.sqrt 3 / 4) * 60 * s^2
   area_A'B'C' / area_ABC = 60) :=
sorry

end area_ratio_equilateral_triangle_extension_l679_67976


namespace probability_of_one_red_ball_is_one_third_l679_67997

-- Define the number of red and black balls
def red_balls : Nat := 2
def black_balls : Nat := 4
def total_balls : Nat := red_balls + black_balls

-- Define the probability calculation
def probability_red_ball : ℚ := red_balls / (red_balls + black_balls)

-- State the theorem
theorem probability_of_one_red_ball_is_one_third :
  probability_red_ball = 1 / 3 :=
by
  sorry

end probability_of_one_red_ball_is_one_third_l679_67997


namespace sufficient_not_necessary_condition_l679_67951

noncomputable def sufficient_but_not_necessary (x y : ℝ) : Prop :=
  (x > 1 ∧ y > 1) → (x + y > 2) ∧ (x + y > 2 → ¬(x > 1 ∧ y > 1))

theorem sufficient_not_necessary_condition (x y : ℝ) :
  sufficient_but_not_necessary x y :=
sorry

end sufficient_not_necessary_condition_l679_67951


namespace quadratic_trinomial_has_two_roots_l679_67982

theorem quadratic_trinomial_has_two_roots
  (a b c : ℝ) (h : b^2 - 4 * a * c > 0) : (2 * (a + b))^2 - 4 * 3 * a * (b + c) > 0 := by
  sorry

end quadratic_trinomial_has_two_roots_l679_67982


namespace single_point_graph_d_l679_67925

theorem single_point_graph_d (d : ℝ) : 
  (∀ x y : ℝ, 3 * x^2 + y^2 + 6 * x - 12 * y + d = 0 ↔ x = -1 ∧ y = 6) → d = 39 :=
by 
  sorry

end single_point_graph_d_l679_67925


namespace volume_of_truncated_cone_l679_67907

noncomputable def surface_area_top : ℝ := 3 * Real.pi
noncomputable def surface_area_bottom : ℝ := 12 * Real.pi
noncomputable def slant_height : ℝ := 2
noncomputable def volume_cone : ℝ := 7 * Real.pi

theorem volume_of_truncated_cone :
  ∃ V : ℝ, V = volume_cone :=
sorry

end volume_of_truncated_cone_l679_67907


namespace solution_set_of_equation_l679_67910

noncomputable def log_base (b x : ℝ) := Real.log x / Real.log b

theorem solution_set_of_equation (x : ℝ) (h : x > 0): (x^(log_base 10 x) = x^3 / 100) ↔ (x = 10 ∨ x = 100) := 
by sorry

end solution_set_of_equation_l679_67910


namespace lesser_solution_of_quadratic_l679_67972

theorem lesser_solution_of_quadratic :
  (∃ x y: ℝ, x ≠ y ∧ x^2 + 10*x - 24 = 0 ∧ y^2 + 10*y - 24 = 0 ∧ min x y = -12) :=
by {
  sorry
}

end lesser_solution_of_quadratic_l679_67972


namespace hotel_charge_per_hour_morning_l679_67946

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

end hotel_charge_per_hour_morning_l679_67946


namespace Kim_morning_routine_time_l679_67999

theorem Kim_morning_routine_time :
  let senior_employees := 3
  let junior_employees := 3
  let interns := 3

  let senior_overtime := 2
  let junior_overtime := 3
  let intern_overtime := 1
  let senior_not_overtime := senior_employees - senior_overtime
  let junior_not_overtime := junior_employees - junior_overtime
  let intern_not_overtime := interns - intern_overtime

  let coffee_time := 5
  let email_time := 10
  let supplies_time := 8
  let meetings_time := 6
  let reports_time := 5

  let status_update_time := 3 * senior_employees + 2 * junior_employees + 1 * interns
  let payroll_update_time := 
    4 * senior_overtime + 2 * senior_not_overtime +
    3 * junior_overtime + 1 * junior_not_overtime +
    2 * intern_overtime + 0.5 * intern_not_overtime
  let daily_tasks_time :=
    4 * senior_employees + 3 * junior_employees + 2 * interns

  let total_time := coffee_time + status_update_time + payroll_update_time + daily_tasks_time + email_time + supplies_time + meetings_time + reports_time
  total_time = 101 := by
  sorry

end Kim_morning_routine_time_l679_67999


namespace odd_exponent_divisibility_l679_67932

theorem odd_exponent_divisibility (x y : ℤ) (k : ℕ) (h : (x^(2*k-1) + y^(2*k-1)) % (x + y) = 0) : 
  (x^(2*k+1) + y^(2*k+1)) % (x + y) = 0 :=
sorry

end odd_exponent_divisibility_l679_67932


namespace smallest_positive_debt_resolvable_l679_67918

theorem smallest_positive_debt_resolvable :
  ∃ p g : ℤ, 280 * p + 200 * g = 40 ∧
  ∀ k : ℤ, k > 0 → (∃ p g : ℤ, 280 * p + 200 * g = k) → 40 ≤ k :=
by
  sorry

end smallest_positive_debt_resolvable_l679_67918


namespace problem_I5_1_l679_67950

theorem problem_I5_1 (a : ℝ) (h : a^2 - 8^2 = 12^2 + 9^2) : a = 17 := 
sorry

end problem_I5_1_l679_67950


namespace interval_of_monotonic_increase_parallel_vectors_tan_x_perpendicular_vectors_smallest_positive_x_l679_67921

noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.sin x, Real.cos x)
noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.cos x, Real.cos x)
noncomputable def f (x : ℝ) : ℝ := 2 * (a x).1 * (b x).1 + 2 * (a x).2 * (b x).2 - 1

theorem interval_of_monotonic_increase (x : ℝ) :
  ∃ k : ℤ, k * Real.pi - Real.pi / 3 ≤ x ∧ x ≤ k * Real.pi + Real.pi / 6 := sorry

theorem parallel_vectors_tan_x (x : ℝ) (h₁ : Real.sin x * Real.cos x - Real.sqrt 3 * Real.cos x * Real.cos x = 0) (h₂ : Real.cos x ≠ 0) :
  Real.tan x = Real.sqrt 3 := sorry

theorem perpendicular_vectors_smallest_positive_x (x : ℝ) (h₁ : Real.sqrt 3 * Real.sin x * Real.cos x + Real.cos x * Real.cos x = 0) (h₂ : Real.cos x ≠ 0) :
 x = 5 * Real.pi / 6 := sorry

end interval_of_monotonic_increase_parallel_vectors_tan_x_perpendicular_vectors_smallest_positive_x_l679_67921


namespace marble_ratio_l679_67957

-- Definitions and assumptions from the conditions
def my_marbles : ℕ := 16
def total_marbles : ℕ := 63
def transfer_amount : ℕ := 2

-- After transferring marbles to my brother
def my_marbles_after_transfer := my_marbles - transfer_amount
def brother_marbles (B : ℕ) := B + transfer_amount

-- Friend's marbles
def friend_marbles (F : ℕ) := F = 3 * my_marbles_after_transfer

-- Prove the ratio of marbles after transfer
theorem marble_ratio (B F : ℕ) (hf : F = 3 * my_marbles_after_transfer) (h_total : my_marbles + B + F = total_marbles)
  (h_multiple : ∃ M : ℕ, my_marbles_after_transfer = M * brother_marbles B) :
  (my_marbles_after_transfer : ℚ) / (brother_marbles B : ℚ) = 2 / 1 :=
by
  sorry

end marble_ratio_l679_67957


namespace spam_ratio_l679_67991

theorem spam_ratio (total_emails important_emails promotional_fraction promotional_emails spam_emails : ℕ) 
  (h1 : total_emails = 400) 
  (h2 : important_emails = 180) 
  (h3 : promotional_fraction = 2/5) 
  (h4 : total_emails - important_emails = spam_emails + promotional_emails) 
  (h5 : promotional_emails = promotional_fraction * (total_emails - important_emails)) 
  : spam_emails / total_emails = 33 / 100 := 
by {
  sorry
}

end spam_ratio_l679_67991


namespace clock_angle_at_3_20_is_160_l679_67979

noncomputable def clock_angle_3_20 : ℚ :=
  let hour_hand_at_3 : ℚ := 90
  let minute_hand_per_minute : ℚ := 6
  let hour_hand_per_minute : ℚ := 1 / 2
  let time_passed : ℚ := 20
  let angle_change_per_minute : ℚ := minute_hand_per_minute - hour_hand_per_minute
  let total_angle_change : ℚ := time_passed * angle_change_per_minute
  let final_angle : ℚ := hour_hand_at_3 + total_angle_change
  let smaller_angle : ℚ := if final_angle > 180 then 360 - final_angle else final_angle
  smaller_angle

theorem clock_angle_at_3_20_is_160 : clock_angle_3_20 = 160 :=
by
  sorry

end clock_angle_at_3_20_is_160_l679_67979


namespace unique_identity_function_l679_67961

theorem unique_identity_function (f : ℝ → ℝ) (H : ∀ x y z : ℝ, (x^3 + f y * x + f z = 0) → (f x ^ 3 + y * f x + z = 0)) :
  f = id :=
by sorry

end unique_identity_function_l679_67961


namespace monochromatic_triangle_probability_correct_l679_67967

noncomputable def monochromatic_triangle_probability (p : ℝ) : ℝ :=
  1 - (3 * (p^2) * (1 - p) + 3 * ((1 - p)^2) * p)^20

theorem monochromatic_triangle_probability_correct :
  monochromatic_triangle_probability (1/2) = 1 - (3/4)^20 :=
by
  sorry

end monochromatic_triangle_probability_correct_l679_67967


namespace not_possible_values_l679_67978

theorem not_possible_values (t h d : ℕ) (ht : 3 * t - 6 * h = 2001) (hd : t - h = d) (hh : 6 * h > 0) :
  ∃ n, n = 667 ∧ ∀ d : ℕ, d ≤ 667 → ¬ (t = h + d ∧ 3 * (h + d) - 6 * h = 2001) :=
by
  sorry

end not_possible_values_l679_67978


namespace range_of_m_l679_67914

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, (x - m > 0) → (2*x + 1 > 3) → (x > 1)) → (m ≤ 1) :=
by
  intros h
  sorry

end range_of_m_l679_67914


namespace ratio_b_to_c_l679_67924

theorem ratio_b_to_c (x a b c : ℤ) 
    (h1 : x = 100 * a + 10 * b + c)
    (h2 : a > 0)
    (h3 : 999 - x = 241) : (b : ℚ) / c = 5 / 8 :=
by
  sorry

end ratio_b_to_c_l679_67924


namespace spent_on_books_l679_67944

theorem spent_on_books (allowance games_fraction snacks_fraction toys_fraction : ℝ)
  (h_allowance : allowance = 50)
  (h_games : games_fraction = 1/4)
  (h_snacks : snacks_fraction = 1/5)
  (h_toys : toys_fraction = 2/5) :
  allowance - (allowance * games_fraction + allowance * snacks_fraction + allowance * toys_fraction) = 7.5 :=
by
  sorry

end spent_on_books_l679_67944


namespace rate_of_stream_l679_67952

-- Definitions from problem conditions
def rowing_speed_still_water : ℕ := 24

-- Assume v is the rate of the stream
variable (v : ℕ)

-- Time taken to row up is three times the time taken to row down
def rowing_time_condition : Prop :=
  1 / (rowing_speed_still_water - v) = 3 * (1 / (rowing_speed_still_water + v))

-- The rate of the stream (v) should be 12 kmph
theorem rate_of_stream (h : rowing_time_condition v) : v = 12 :=
  sorry

end rate_of_stream_l679_67952


namespace projection_is_orthocenter_l679_67998

-- Define a structure for a point in 3D space.
structure Point (α : Type) :=
(x : α)
(y : α)
(z : α)

-- Define mutually perpendicular edges condition.
def mutually_perpendicular {α : Type} [Field α] (A B C D : Point α) :=
(A.x - D.x) * (B.x - D.x) + (A.y - D.y) * (B.y - D.y) + (A.z - D.z) * (B.z - D.z) = 0 ∧
(A.x - D.x) * (C.x - D.x) + (A.y - D.y) * (C.y - D.y) + (A.z - D.z) * (C.z - D.z) = 0 ∧
(B.x - D.x) * (C.x - D.x) + (B.y - D.y) * (C.y - D.y) + (B.z - D.z) * (C.z - D.z) = 0

-- The main theorem statement.
theorem projection_is_orthocenter {α : Type} [Field α]
    (A B C D : Point α) (h : mutually_perpendicular A B C D) :
    ∃ O : Point α, -- there exists a point O (the orthocenter)
    (O.x * (B.y - A.y) + O.y * (A.y - B.y) + O.z * (A.y - B.y)) = 0 ∧
    (O.x * (C.y - B.y) + O.y * (B.y - C.y) + O.z * (B.y - C.y)) = 0 ∧
    (O.x * (A.y - C.y) + O.y * (C.y - A.y) + O.z * (C.y - A.y)) = 0 := 
sorry

end projection_is_orthocenter_l679_67998


namespace solution_set_of_3x2_minus_7x_gt_6_l679_67913

theorem solution_set_of_3x2_minus_7x_gt_6 (x : ℝ) :
  3 * x^2 - 7 * x > 6 ↔ (x < -2 / 3 ∨ x > 3) := 
by
  sorry

end solution_set_of_3x2_minus_7x_gt_6_l679_67913


namespace john_spending_l679_67981

variable (initial_cost : ℕ) (sale_price : ℕ) (new_card_cost : ℕ)

theorem john_spending (h1 : initial_cost = 1200) (h2 : sale_price = 300) (h3 : new_card_cost = 500) :
  initial_cost - sale_price + new_card_cost = 1400 := 
by
  sorry

end john_spending_l679_67981


namespace set_intersection_complement_l679_67960

variable (U : Set ℝ := Set.univ)
variable (M : Set ℝ := {x | ∃ y, y = Real.log (x^2 - 1)})
variable (N : Set ℝ := {x | 0 < x ∧ x < 2})

theorem set_intersection_complement :
  N ∩ (U \ M) = {x | 0 < x ∧ x ≤ 1} :=
  sorry

end set_intersection_complement_l679_67960


namespace geom_seq_sixth_term_l679_67916

theorem geom_seq_sixth_term (a : ℝ) (r : ℝ) (h1: a * r^3 = 512) (h2: a * r^8 = 8) : 
  a * r^5 = 128 := 
by 
  sorry

end geom_seq_sixth_term_l679_67916


namespace trig_comparison_l679_67988

theorem trig_comparison 
  (a : ℝ) (b : ℝ) (c : ℝ) :
  a = Real.sin (3 * Real.pi / 5) → 
  b = Real.cos (2 * Real.pi / 5) → 
  c = Real.tan (2 * Real.pi / 5) → 
  b < a ∧ a < c :=
by
  intro ha hb hc
  sorry

end trig_comparison_l679_67988


namespace stretched_curve_l679_67939

noncomputable def transformed_curve (x : ℝ) : ℝ :=
  2 * Real.sin (x / 3 + Real.pi / 3)

theorem stretched_curve (y x : ℝ) :
  y = 2 * Real.sin (x + Real.pi / 3) → y = transformed_curve x := by
  intro h
  sorry

end stretched_curve_l679_67939


namespace chess_games_total_l679_67995

-- Conditions
def crowns_per_win : ℕ := 8
def uncle_wins : ℕ := 4
def draws : ℕ := 5
def father_net_gain : ℤ := 24

-- Let total_games be the total number of games played
def total_games : ℕ := sorry

-- Proof that under the given conditions, total_games equals 16
theorem chess_games_total :
  total_games = uncle_wins + (father_net_gain + uncle_wins * crowns_per_win) / crowns_per_win + draws := by
  sorry

end chess_games_total_l679_67995


namespace necessary_condition_to_contain_circle_in_parabola_l679_67984

def M (x y : ℝ) : Prop := y ≥ x^2
def N (x y a : ℝ) : Prop := x^2 + (y - a)^2 ≤ 1

theorem necessary_condition_to_contain_circle_in_parabola (a : ℝ) : 
  (∀ x y, N x y a → M x y) ↔ a ≥ 5 / 4 := 
sorry

end necessary_condition_to_contain_circle_in_parabola_l679_67984


namespace order_of_fractions_l679_67990

theorem order_of_fractions (a b c d : ℝ) (hpos_a : a > 0) (hpos_b : b > 0) (hpos_c : c > 0) (hpos_d : d > 0)
(hab : a > b) : (b / a) < (b + c) / (a + c) ∧ (b + c) / (a + c) < (a + d) / (b + d) ∧ (a + d) / (b + d) < (a / b) :=
by
  sorry

end order_of_fractions_l679_67990


namespace range_of_a_l679_67949

noncomputable def f (x : ℝ) : ℝ := (x + 1) * Real.log (x + 1)

theorem range_of_a (a : ℝ) :
  (∀ x ≥ 0, f x ≥ a * x) ↔ (a ≤ 1) :=
by
  sorry

end range_of_a_l679_67949


namespace sum_of_six_primes_even_l679_67985

/-- If A, B, and C are positive integers such that A, B, C, A-B, A+B, and A+B+C are all prime numbers, 
    and B is specifically the prime number 2,
    then the sum of these six primes is even. -/
theorem sum_of_six_primes_even (A B C : ℕ) (hA : Prime A) (hB : Prime B) (hC : Prime C) 
    (h1 : Prime (A - B)) (h2 : Prime (A + B)) (h3 : Prime (A + B + C)) (hB_eq_two : B = 2) : 
    Even (A + B + C + (A - B) + (A + B) + (A + B + C)) :=
by
  sorry

end sum_of_six_primes_even_l679_67985


namespace cost_of_burger_l679_67904

theorem cost_of_burger :
  ∃ (b s f : ℕ), 
    4 * b + 3 * s + f = 540 ∧
    3 * b + 2 * s + 2 * f = 580 ∧
    b = 100 :=
by {
  sorry
}

end cost_of_burger_l679_67904


namespace remainder_when_divided_by_9_l679_67942

open Nat

theorem remainder_when_divided_by_9 (A B : ℕ) (h : A = B * 9 + 13) : A % 9 = 4 :=
by
  sorry

end remainder_when_divided_by_9_l679_67942


namespace cooking_people_count_l679_67959

variables (P Y W : ℕ)

def people_practicing_yoga := 25
def people_studying_weaving := 8
def people_studying_only_cooking := 2
def people_studying_cooking_and_yoga := 7
def people_studying_cooking_and_weaving := 3
def people_studying_all_curriculums := 3

theorem cooking_people_count :
  P = people_studying_only_cooking + (people_studying_cooking_and_yoga - people_studying_all_curriculums)
    + (people_studying_cooking_and_weaving - people_studying_all_curriculums) + people_studying_all_curriculums →
  P = 9 :=
by
  intro h
  unfold people_studying_only_cooking people_studying_cooking_and_yoga people_studying_cooking_and_weaving people_studying_all_curriculums at h
  sorry

end cooking_people_count_l679_67959


namespace arianna_sleeping_hours_l679_67934

def hours_in_day : ℕ := 24
def hours_at_work : ℕ := 6
def hours_on_chores : ℕ := 5
def hours_sleeping : ℕ := hours_in_day - (hours_at_work + hours_on_chores)

theorem arianna_sleeping_hours : hours_sleeping = 13 := by
  sorry

end arianna_sleeping_hours_l679_67934


namespace parametric_to_standard_l679_67953

theorem parametric_to_standard (t : ℝ) : 
  (x = (2 + 3 * t) / (1 + t)) ∧ (y = (1 - 2 * t) / (1 + t)) → (3 * x + y - 7 = 0) ∧ (x ≠ 3) := 
by 
  sorry

end parametric_to_standard_l679_67953


namespace product_of_roots_cubic_l679_67922

theorem product_of_roots_cubic :
  ∀ (x : ℝ), (x^3 - 15 * x^2 + 75 * x - 50 = 0) →
    (∃ a b c : ℝ, x = a * b * c ∧ x = 50) :=
by
  sorry

end product_of_roots_cubic_l679_67922


namespace find_x_l679_67962

theorem find_x (x : ℝ) (h : 0.60 / x = 6 / 2) : x = 0.2 :=
by {
  sorry
}

end find_x_l679_67962


namespace solution_l679_67964

def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

def is_even (g : ℝ → ℝ) : Prop :=
  ∀ y : ℝ, g (-y) = g y

def problem (f g : ℝ → ℝ) : Prop :=
  (∀ x y : ℝ, f (x + y) + f (x - y) = 2 * f x * g y) ∧
  (f 0 = 0) ∧
  (∃ x : ℝ, f x ≠ 0)

theorem solution (f g : ℝ → ℝ) (h : problem f g) : is_odd f ∧ is_even g :=
sorry

end solution_l679_67964


namespace triangle_right_angled_and_common_difference_equals_inscribed_circle_radius_l679_67969

noncomputable def a : ℝ := sorry
noncomputable def d : ℝ := a / 4
noncomputable def half_perimeter : ℝ := (a - d + a + (a + d)) / 2
noncomputable def r : ℝ := ((a - d) + a + (a + d)) / 2

theorem triangle_right_angled_and_common_difference_equals_inscribed_circle_radius :
  (half_perimeter > a + d) →
  ((a - d) + a + (a + d) = 2 * half_perimeter) →
  (a - d)^2 + a^2 = (a + d)^2 →
  d = r :=
by
  intros h1 h2 h3
  sorry

end triangle_right_angled_and_common_difference_equals_inscribed_circle_radius_l679_67969


namespace ratio_of_areas_of_squares_l679_67943

theorem ratio_of_areas_of_squares (a_side b_side : ℕ) (h_a : a_side = 36) (h_b : b_side = 42) : 
  (a_side ^ 2 : ℚ) / (b_side ^ 2 : ℚ) = 36 / 49 :=
by
  sorry

end ratio_of_areas_of_squares_l679_67943


namespace total_books_after_loss_l679_67974

-- Define variables for the problem
def sandy_books : ℕ := 10
def tim_books : ℕ := 33
def benny_lost_books : ℕ := 24

-- Prove the final number of books together
theorem total_books_after_loss : (sandy_books + tim_books - benny_lost_books) = 19 := by
  sorry

end total_books_after_loss_l679_67974


namespace find_a_value_l679_67945

theorem find_a_value 
  (a : ℝ) 
  (P : ℝ × ℝ) 
  (circle_eq : ∀ x y : ℝ, x^2 + y^2 - 2 * a * x + 2 * y - 1 = 0)
  (M N : ℝ × ℝ)
  (tangent_condition : (N.snd - M.snd) / (N.fst - M.fst) + (M.fst + N.fst - 2) / (M.snd + N.snd) = 0) : 
  a = 3 ∨ a = -2 := 
sorry

end find_a_value_l679_67945


namespace price_reduction_is_50_rubles_l679_67968

theorem price_reduction_is_50_rubles :
  let P_Feb : ℕ := 300
  let P_Mar : ℕ := 250
  P_Feb - P_Mar = 50 :=
by
  let P_Feb : ℕ := 300
  let P_Mar : ℕ := 250
  sorry

end price_reduction_is_50_rubles_l679_67968


namespace count_non_empty_subsets_of_odd_numbers_greater_than_one_l679_67915

-- Condition definitions
def given_set : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}
def odd_numbers_greater_than_one (s : Finset ℕ) : Finset ℕ := 
  s.filter (λ x => x % 2 = 1 ∧ x > 1)

-- The problem statement
theorem count_non_empty_subsets_of_odd_numbers_greater_than_one : 
  (odd_numbers_greater_than_one given_set).powerset.card - 1 = 15 := 
by 
  sorry

end count_non_empty_subsets_of_odd_numbers_greater_than_one_l679_67915


namespace hyperbola_eccentricity_l679_67929

variables (a b e : ℝ) (F1 F2 P : ℝ × ℝ)

-- The hyperbola assumption
def hyperbola : Prop := ∃ (x y : ℝ), (x, y) = P ∧ x^2 / a^2 - y^2 / b^2 = 1
-- a > 0 and b > 0
def positive_a_b : Prop := a > 0 ∧ b > 0
-- Distance between foci
def distance_foci : Prop := dist F1 F2 = 12
-- Distance PF2
def distance_p_f2 : Prop := dist P F2 = 5
-- To be proven, eccentricity of the hyperbola
def eccentricity : Prop := e = 3 / 2

theorem hyperbola_eccentricity : hyperbola a b P ∧ positive_a_b a b ∧ distance_foci F1 F2 ∧ distance_p_f2 P F2 → eccentricity e :=
by
  sorry

end hyperbola_eccentricity_l679_67929


namespace discount_threshold_l679_67902

-- Definitions based on given conditions
def photocopy_cost : ℝ := 0.02
def discount_percentage : ℝ := 0.25
def copies_needed_each : ℕ := 80
def total_savings : ℝ := 0.40 * 2 -- total savings for both Steve and Dennison

-- Minimum number of photocopies required to get the discount
def min_copies_for_discount : ℕ := 160

-- Lean statement to prove the minimum number of photocopies required for the discount
theorem discount_threshold :
  ∀ (x : ℕ),
  photocopy_cost * (x : ℝ) - (photocopy_cost * (1 - discount_percentage) * (x : ℝ)) * 2 = total_savings → 
  min_copies_for_discount = 160 :=
by sorry

end discount_threshold_l679_67902


namespace length_of_faster_train_is_correct_l679_67986

def speed_faster_train := 54 -- kmph
def speed_slower_train := 36 -- kmph
def crossing_time := 27 -- seconds

def kmph_to_mps (s : ℕ) : ℕ :=
  s * 1000 / 3600

def relative_speed_faster_train := kmph_to_mps (speed_faster_train - speed_slower_train)

def length_faster_train := relative_speed_faster_train * crossing_time

theorem length_of_faster_train_is_correct : length_faster_train = 135 := 
  by
  sorry

end length_of_faster_train_is_correct_l679_67986


namespace least_n_for_distance_l679_67938

-- Definitions ensuring our points and distances
def A_0 : (ℝ × ℝ) := (0, 0)

-- Assume we have distance function and equilateral triangles on given coordinates
def is_on_x_axis (p : ℕ → ℝ × ℝ) : Prop := ∀ n, (p n).snd = 0
def is_on_parabola (q : ℕ → ℝ × ℝ) : Prop := ∀ n, (q n).snd = (q n).fst^2
def is_equilateral (p : ℕ → ℝ × ℝ) (q : ℕ → ℝ × ℝ) (n : ℕ) : Prop :=
  let d1 := dist (p (n-1)) (q n)
  let d2 := dist (q n) (p n)
  let d3 := dist (p (n-1)) (p n)
  d1 = d2 ∧ d2 = d3

-- Define the main property we want to prove
def main_property (n : ℕ) (A : ℕ → ℝ × ℝ) (B : ℕ → ℝ × ℝ) : Prop :=
  A 0 = A_0 ∧ is_on_x_axis A ∧ is_on_parabola B ∧
  (∀ k, is_equilateral A B (k+1)) ∧
  dist A_0 (A n) ≥ 200

-- Final theorem statement
theorem least_n_for_distance (A : ℕ → ℝ × ℝ) (B : ℕ → ℝ × ℝ) :
  (∃ n, main_property n A B ∧ (∀ m, main_property m A B → n ≤ m)) ↔ n = 24 := by
  sorry

end least_n_for_distance_l679_67938


namespace ratio_problem_l679_67931

theorem ratio_problem (m n p q : ℚ) 
  (h1 : m / n = 12) 
  (h2 : p / n = 4) 
  (h3 : p / q = 1 / 8) :
  m / q = 3 / 8 :=
by
  sorry

end ratio_problem_l679_67931


namespace sum_of_solutions_l679_67919

theorem sum_of_solutions (a b : ℤ) (h₁ : a = -1) (h₂ : b = -4) (h₃ : ∀ x : ℝ, (16 - 4 * x - x^2 = 0 ↔ -x^2 - 4 * x + 16 = 0)) : 
  (-b / a) = 4 := 
by 
  rw [h₁, h₂]
  norm_num
  sorry

end sum_of_solutions_l679_67919
