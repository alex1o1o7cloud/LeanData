import Mathlib

namespace NUMINAMATH_GPT_frustum_lateral_surface_area_l1628_162839

theorem frustum_lateral_surface_area (r1 r2 h : ℝ) (r1_eq : r1 = 10) (r2_eq : r2 = 4) (h_eq : h = 6) :
  let s := Real.sqrt (h^2 + (r1 - r2)^2)
  let A := Real.pi * (r1 + r2) * s
  A = 84 * Real.pi * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_frustum_lateral_surface_area_l1628_162839


namespace NUMINAMATH_GPT_power_sum_prime_eq_l1628_162893

theorem power_sum_prime_eq (p a n : ℕ) (hp : p.Prime) (h_eq : 2^p + 3^p = a^n) : n = 1 :=
by sorry

end NUMINAMATH_GPT_power_sum_prime_eq_l1628_162893


namespace NUMINAMATH_GPT_product_of_first_five_terms_l1628_162804

variable {a : ℕ → ℝ}

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ m n p q : ℕ, m > 0 ∧ n > 0 ∧ p > 0 ∧ q > 0 ∧ m + n = p + q → a m * a n = a p * a q

theorem product_of_first_five_terms 
  (h : geometric_sequence a) 
  (h3 : a 3 = 2) : 
  a 1 * a 2 * a 3 * a 4 * a 5 = 32 :=
sorry

end NUMINAMATH_GPT_product_of_first_five_terms_l1628_162804


namespace NUMINAMATH_GPT_cyclic_quadrilateral_angles_l1628_162861

theorem cyclic_quadrilateral_angles (A B C D : ℝ) (h_cyclic : A + C = 180) (h_diag_bisect : (A = 2 * (B / 5 + B / 5)) ∧ (C = 2 * (D / 5 + D / 5))) (h_ratio : B / D = 2 / 3):
  A = 80 ∨ A = 100 ∨ A = 1080 / 11 ∨ A = 900 / 11 :=
  sorry

end NUMINAMATH_GPT_cyclic_quadrilateral_angles_l1628_162861


namespace NUMINAMATH_GPT_range_of_m_l1628_162892

theorem range_of_m
  (h : ∀ x : ℝ, (m / (2 * x - 4) = (1 - x) / (2 - x) - 2) → x > 0) :
  m < 6 ∧ m ≠ 2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l1628_162892


namespace NUMINAMATH_GPT_minimum_distance_l1628_162835

-- Define conditions and problem

def lies_on_line (P : ℝ × ℝ) : Prop :=
  P.1 + P.2 - 4 = 0

theorem minimum_distance (P : ℝ × ℝ) (h : lies_on_line P) : P.1^2 + P.2^2 ≥ 8 :=
sorry

end NUMINAMATH_GPT_minimum_distance_l1628_162835


namespace NUMINAMATH_GPT_wings_count_total_l1628_162840

def number_of_wings (num_planes : Nat) (wings_per_plane : Nat) : Nat :=
  num_planes * wings_per_plane

theorem wings_count_total :
  number_of_wings 45 2 = 90 :=
  by
    sorry

end NUMINAMATH_GPT_wings_count_total_l1628_162840


namespace NUMINAMATH_GPT_cat_food_insufficient_for_six_days_l1628_162846

theorem cat_food_insufficient_for_six_days
  (B S : ℕ)
  (h1 : B > S)
  (h2 : B < 2 * S)
  (h3 : B + 2 * S = 2 * ((B + 2 * S) / 2)) :
  4 * B + 4 * S < 3 * (B + 2 * S) :=
by sorry

end NUMINAMATH_GPT_cat_food_insufficient_for_six_days_l1628_162846


namespace NUMINAMATH_GPT_ophelia_age_l1628_162845

/-- 
If Lennon is currently 8 years old, 
and in two years Ophelia will be four times as old as Lennon,
then Ophelia is currently 38 years old 
-/
theorem ophelia_age 
  (lennon_age : ℕ) 
  (ophelia_age_in_two_years : ℕ) 
  (h1 : lennon_age = 8)
  (h2 : ophelia_age_in_two_years = 4 * (lennon_age + 2)) : 
  ophelia_age_in_two_years - 2 = 38 :=
by
  sorry

end NUMINAMATH_GPT_ophelia_age_l1628_162845


namespace NUMINAMATH_GPT_distance_from_dormitory_to_city_l1628_162888

theorem distance_from_dormitory_to_city (D : ℝ) 
  (h1 : D = (1/2) * D + (1/4) * D + 6) : D = 24 := 
  sorry

end NUMINAMATH_GPT_distance_from_dormitory_to_city_l1628_162888


namespace NUMINAMATH_GPT_complex_multiplication_l1628_162854

theorem complex_multiplication : ∀ (i : ℂ), i^2 = -1 → i * (2 + 3 * i) = (-3 : ℂ) + 2 * i :=
by
  intros i hi
  sorry

end NUMINAMATH_GPT_complex_multiplication_l1628_162854


namespace NUMINAMATH_GPT_contrapositive_proposition_l1628_162832

theorem contrapositive_proposition (x : ℝ) : (x ≤ -1 ∨ x ≥ 1) → (x^2 ≥ 1) :=
by
  sorry

end NUMINAMATH_GPT_contrapositive_proposition_l1628_162832


namespace NUMINAMATH_GPT_asymptotes_of_hyperbola_l1628_162856

theorem asymptotes_of_hyperbola (a b x y : ℝ) (h : a = 5 ∧ b = 2) :
  (x^2 / 25 - y^2 / 4 = 1) → (y = (2 / 5) * x ∨ y = -(2 / 5) * x) :=
by
  sorry

end NUMINAMATH_GPT_asymptotes_of_hyperbola_l1628_162856


namespace NUMINAMATH_GPT_john_speed_when_runs_alone_l1628_162807

theorem john_speed_when_runs_alone (x : ℝ) : 
  (6 * (1/2) + x * (1/2) = 5) → x = 4 :=
by
  intro h
  linarith

end NUMINAMATH_GPT_john_speed_when_runs_alone_l1628_162807


namespace NUMINAMATH_GPT_covered_digits_l1628_162847

def four_digit_int (n : ℕ) : Prop := n ≥ 1000 ∧ n < 10000

theorem covered_digits (a b c : ℕ) (n1 n2 n3 : ℕ) :
  four_digit_int n1 → four_digit_int n2 → four_digit_int n3 →
  n1 + n2 + n3 = 10126 →
  (n1 % 10 = 3 ∧ n2 % 10 = 7 ∧ n3 % 10 = 6) →
  (n1 / 10 % 10 = 4 ∧ n2 / 10 % 10 = a ∧ n3 / 10 % 10 = 2) →
  (n1 / 100 % 10 = 2 ∧ n2 / 100 % 10 = 1 ∧ n3 / 100 % 10 = c) →
  (n1 / 1000 = 1 ∧ n2 / 1000 = 2 ∧ n3 / 1000 = b) →
  (a = 5 ∧ b = 6 ∧ c = 7) := 
sorry

end NUMINAMATH_GPT_covered_digits_l1628_162847


namespace NUMINAMATH_GPT_converse_of_posImpPosSquare_l1628_162828

-- Let's define the condition proposition first
def posImpPosSquare (x : ℝ) : Prop := x > 0 → x^2 > 0

-- Now, we state the converse we need to prove
theorem converse_of_posImpPosSquare (x : ℝ) (h : posImpPosSquare x) : x^2 > 0 → x > 0 := sorry

end NUMINAMATH_GPT_converse_of_posImpPosSquare_l1628_162828


namespace NUMINAMATH_GPT_exists_plane_intersecting_in_parallel_lines_l1628_162830

variables {Point Line Plane : Type}
variables (a : Line) (S₁ S₂ : Plane)

-- Definitions and assumptions
def intersects_in (a : Line) (P : Plane) : Prop := sorry
def parallel_lines (l₁ l₂ : Line) : Prop := sorry

-- Proof problem statement
theorem exists_plane_intersecting_in_parallel_lines :
  ∃ P : Plane, intersects_in a P ∧
    (∀ l₁ l₂ : Line, (intersects_in l₁ S₁ ∧ intersects_in l₂ S₂ ∧ l₁ = l₂)
                     → parallel_lines l₁ l₂) :=
sorry

end NUMINAMATH_GPT_exists_plane_intersecting_in_parallel_lines_l1628_162830


namespace NUMINAMATH_GPT_coprime_solution_l1628_162882

theorem coprime_solution (a b : ℕ) (h_coprime : Nat.gcd a b = 1) (h_eq : 5 * a + 7 * b = 29 * (6 * a + 5 * b)) : a = 3 ∧ b = 2 :=
sorry

end NUMINAMATH_GPT_coprime_solution_l1628_162882


namespace NUMINAMATH_GPT_temp_fri_l1628_162808

-- Define the temperatures on Monday, Tuesday, Wednesday, Thursday, and Friday
variables (M T W Th F : ℝ)

-- Define the conditions as given in the problem
axiom avg_mon_thurs : (M + T + W + Th) / 4 = 48
axiom avg_tues_fri : (T + W + Th + F) / 4 = 46
axiom temp_mon : M = 39

-- The theorem to prove that the temperature on Friday is 31 degrees
theorem temp_fri : F = 31 :=
by
  -- placeholder for proof
  sorry

end NUMINAMATH_GPT_temp_fri_l1628_162808


namespace NUMINAMATH_GPT_max_side_of_triangle_exists_max_side_of_elevent_l1628_162837

noncomputable def is_valid_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem max_side_of_triangle (a b c : ℕ) (h₁ : a ≠ b) (h₂ : b ≠ c) (h₃ : a ≠ c)
  (h₄ : a + b + c = 24) (h_triangle : is_valid_triangle a b c) :
  max a (max b c) ≤ 11 :=
sorry

theorem exists_max_side_of_elevent (h₄ : ∃ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ a ≠ c 
  ∧ a + b + c = 24 ∧ is_valid_triangle a b c) :
  ∃ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ a ≠ c 
  ∧ a + b + c = 24 ∧ is_valid_triangle a b c 
  ∧ max a (max b c) = 11 :=
sorry

end NUMINAMATH_GPT_max_side_of_triangle_exists_max_side_of_elevent_l1628_162837


namespace NUMINAMATH_GPT_cuboid_area_correct_l1628_162819

def cuboid_surface_area (length breadth height : ℕ) :=
  2 * (length * height) + 2 * (breadth * height) + 2 * (length * breadth)

theorem cuboid_area_correct : cuboid_surface_area 4 6 5 = 148 := by
  sorry

end NUMINAMATH_GPT_cuboid_area_correct_l1628_162819


namespace NUMINAMATH_GPT_smallest_solution_of_quadratic_l1628_162884

theorem smallest_solution_of_quadratic :
  ∃ x : ℝ, 6 * x^2 - 29 * x + 35 = 0 ∧ x = 7 / 3 :=
sorry

end NUMINAMATH_GPT_smallest_solution_of_quadratic_l1628_162884


namespace NUMINAMATH_GPT_newspapers_on_sunday_l1628_162873

theorem newspapers_on_sunday (papers_weekend : ℕ) (diff_papers : ℕ) 
  (h1 : papers_weekend = 110) 
  (h2 : diff_papers = 20) 
  (h3 : ∃ (S Su : ℕ), Su = S + diff_papers ∧ S + Su = papers_weekend) :
  ∃ Su, Su = 65 :=
by
  sorry

end NUMINAMATH_GPT_newspapers_on_sunday_l1628_162873


namespace NUMINAMATH_GPT_function_decreases_l1628_162886

def op (m n : ℝ) : ℝ := - (m * n) + n

def f (x : ℝ) : ℝ := op x 2

theorem function_decreases (x1 x2 : ℝ) (h : x1 < x2) : f x1 > f x2 :=
by sorry

end NUMINAMATH_GPT_function_decreases_l1628_162886


namespace NUMINAMATH_GPT_fraction_of_income_from_tips_l1628_162806

theorem fraction_of_income_from_tips (S T : ℚ) (h : T = (11/4) * S) : (T / (S + T)) = (11/15) :=
by sorry

end NUMINAMATH_GPT_fraction_of_income_from_tips_l1628_162806


namespace NUMINAMATH_GPT_points_2_units_away_l1628_162896

theorem points_2_units_away : (∃ x : ℝ, (x = -3 ∨ x = 1) ∧ (abs (x - (-1)) = 2)) :=
by
  sorry

end NUMINAMATH_GPT_points_2_units_away_l1628_162896


namespace NUMINAMATH_GPT_laran_weekly_profit_l1628_162881

-- Definitions based on the problem conditions
def daily_posters_sold : ℕ := 5
def large_posters_sold_daily : ℕ := 2
def small_posters_sold_daily : ℕ := daily_posters_sold - large_posters_sold_daily

def price_large_poster : ℕ := 10
def cost_large_poster : ℕ := 5
def profit_large_poster : ℕ := price_large_poster - cost_large_poster

def price_small_poster : ℕ := 6
def cost_small_poster : ℕ := 3
def profit_small_poster : ℕ := price_small_poster - cost_small_poster

def daily_profit_large_posters : ℕ := large_posters_sold_daily * profit_large_poster
def daily_profit_small_posters : ℕ := small_posters_sold_daily * profit_small_poster
def total_daily_profit : ℕ := daily_profit_large_posters + daily_profit_small_posters

def school_days_week : ℕ := 5
def weekly_profit : ℕ := total_daily_profit * school_days_week

-- Statement to prove
theorem laran_weekly_profit : weekly_profit = 95 := sorry

end NUMINAMATH_GPT_laran_weekly_profit_l1628_162881


namespace NUMINAMATH_GPT_max_profit_l1628_162859

theorem max_profit (m : ℝ) :
  (m - 8) * (900 - 15 * m) = -15 * (m - 34) ^ 2 + 10140 :=
by
  sorry

end NUMINAMATH_GPT_max_profit_l1628_162859


namespace NUMINAMATH_GPT_fraction_problem_l1628_162834

theorem fraction_problem (a : ℕ) (h1 : (a:ℚ)/(a + 27) = 865/1000) : a = 173 := 
by
  sorry

end NUMINAMATH_GPT_fraction_problem_l1628_162834


namespace NUMINAMATH_GPT_payback_duration_l1628_162875

-- Define constants for the problem conditions
def C : ℝ := 25000
def R : ℝ := 4000
def E : ℝ := 1500

-- Formal statement to be proven
theorem payback_duration : C / (R - E) = 10 := 
by
  sorry

end NUMINAMATH_GPT_payback_duration_l1628_162875


namespace NUMINAMATH_GPT_sum_of_digits_of_N_l1628_162872

theorem sum_of_digits_of_N (N : ℕ) (h : N * (N + 1) / 2 = 2016) : (6 + 3 = 9) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_digits_of_N_l1628_162872


namespace NUMINAMATH_GPT_arithmetic_sequence_a_m_n_zero_l1628_162871

theorem arithmetic_sequence_a_m_n_zero
  (a : ℕ → ℕ)
  (m n : ℕ) 
  (hm : m > 0) (hn : n > 0)
  (h_ma_m : a m = n)
  (h_na_n : a n = m) : 
  a (m + n) = 0 :=
by 
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_a_m_n_zero_l1628_162871


namespace NUMINAMATH_GPT_train_length_l1628_162831

theorem train_length (speed_kmph : ℝ) (time_sec : ℝ) (length : ℝ) : 
  speed_kmph = 60 → time_sec = 12 → 
  length = speed_kmph * (1000 / 3600) * time_sec → 
  length = 200.04 :=
by
  intros h_speed h_time h_length
  sorry

end NUMINAMATH_GPT_train_length_l1628_162831


namespace NUMINAMATH_GPT_area_of_efgh_l1628_162812

def small_rectangle_shorter_side : ℝ := 7
def small_rectangle_longer_side : ℝ := 3 * small_rectangle_shorter_side
def larger_rectangle_width : ℝ := small_rectangle_longer_side
def larger_rectangle_length : ℝ := small_rectangle_longer_side + small_rectangle_shorter_side

theorem area_of_efgh :
  larger_rectangle_length * larger_rectangle_width = 588 := by
  sorry

end NUMINAMATH_GPT_area_of_efgh_l1628_162812


namespace NUMINAMATH_GPT_possible_values_of_x_l1628_162809

theorem possible_values_of_x (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (h1 : x + 1 / y = 10) (h2 : y + 1 / x = 5 / 12) : x = 4 ∨ x = 6 :=
by
  sorry

end NUMINAMATH_GPT_possible_values_of_x_l1628_162809


namespace NUMINAMATH_GPT_probability_heads_all_three_tosses_l1628_162816

theorem probability_heads_all_three_tosses :
  (1 / 2) * (1 / 2) * (1 / 2) = 1 / 8 := 
sorry

end NUMINAMATH_GPT_probability_heads_all_three_tosses_l1628_162816


namespace NUMINAMATH_GPT_weight_of_new_person_l1628_162810

theorem weight_of_new_person 
  (avg_increase : Real)
  (num_persons : Nat)
  (old_weight : Real)
  (new_avg_increase : avg_increase = 2.2)
  (number_of_persons : num_persons = 15)
  (weight_of_old_person : old_weight = 75)
  : (new_weight : Real) = old_weight + avg_increase * num_persons := 
  by sorry

end NUMINAMATH_GPT_weight_of_new_person_l1628_162810


namespace NUMINAMATH_GPT_solve_equation_l1628_162880

theorem solve_equation (x : ℝ) : x * (x - 1) = 0 ↔ x = 0 ∨ x = 1 := 
by
  sorry  -- Placeholder for the proof

end NUMINAMATH_GPT_solve_equation_l1628_162880


namespace NUMINAMATH_GPT_sin_120_eq_sqrt3_div_2_l1628_162813

theorem sin_120_eq_sqrt3_div_2 : Real.sin (120 * Real.pi / 180) = Real.sqrt 3 / 2 :=
sorry

end NUMINAMATH_GPT_sin_120_eq_sqrt3_div_2_l1628_162813


namespace NUMINAMATH_GPT_euston_carriages_l1628_162857

-- Definitions of the conditions
def E (N : ℕ) : ℕ := N + 20
def No : ℕ := 100
def FS : ℕ := No + 20
def total_carriages (E N : ℕ) : ℕ := E + N + No + FS

theorem euston_carriages (N : ℕ) (h : total_carriages (E N) N = 460) : E N = 130 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_euston_carriages_l1628_162857


namespace NUMINAMATH_GPT_truck_travel_distance_l1628_162805

theorem truck_travel_distance (miles_per_5gallons miles distance gallons rate : ℕ)
  (h1 : miles_per_5gallons = 150) 
  (h2 : gallons = 5) 
  (h3 : rate = miles_per_5gallons / gallons) 
  (h4 : gallons = 7) 
  (h5 : distance = rate * gallons) : 
  distance = 210 := 
by sorry

end NUMINAMATH_GPT_truck_travel_distance_l1628_162805


namespace NUMINAMATH_GPT_quadratic_inequality_l1628_162874

theorem quadratic_inequality : ∀ x : ℝ, -7 * x ^ 2 + 4 * x - 6 < 0 :=
by
  intro x
  have delta : 4 ^ 2 - 4 * (-7) * (-6) = -152 := by norm_num
  have neg_discriminant : -152 < 0 := by norm_num
  have coef : -7 < 0 := by norm_num
  sorry

end NUMINAMATH_GPT_quadratic_inequality_l1628_162874


namespace NUMINAMATH_GPT_probability_of_drawing_red_ball_l1628_162803

/-- Define the colors of the balls in the bag -/
def yellow_balls : ℕ := 2
def red_balls : ℕ := 3
def white_balls : ℕ := 5

/-- Define the total number of balls in the bag -/
def total_balls : ℕ := yellow_balls + red_balls + white_balls

/-- Define the probability of drawing exactly one red ball -/
def probability_of_red_ball : ℚ := red_balls / total_balls

/-- The main theorem to prove the given problem -/
theorem probability_of_drawing_red_ball :
  probability_of_red_ball = 3 / 10 :=
by
  -- Calculation steps would go here, but are omitted
  sorry

end NUMINAMATH_GPT_probability_of_drawing_red_ball_l1628_162803


namespace NUMINAMATH_GPT_goldfish_graph_finite_set_of_points_l1628_162849

-- Define the cost function for goldfish including the setup fee
def cost (n : ℕ) : ℝ := 20 * n + 5

-- Define the condition
def n_values := {n : ℕ | 1 ≤ n ∧ n ≤ 12}

-- The Lean statement to prove the nature of the graph
theorem goldfish_graph_finite_set_of_points :
  ∀ n ∈ n_values, ∃ k : ℝ, (k = cost n) :=
by
  sorry

end NUMINAMATH_GPT_goldfish_graph_finite_set_of_points_l1628_162849


namespace NUMINAMATH_GPT_simplify_fraction_rationalize_denominator_l1628_162850

noncomputable def sqrt (x : ℝ) : ℝ := Real.sqrt x
noncomputable def fraction := 5 / (sqrt 125 + 3 * sqrt 45 + 4 * sqrt 20 + sqrt 75)

theorem simplify_fraction_rationalize_denominator :
  fraction = sqrt 5 / 27 :=
by
  sorry

end NUMINAMATH_GPT_simplify_fraction_rationalize_denominator_l1628_162850


namespace NUMINAMATH_GPT_value_of_fg_neg_one_l1628_162877

def f (x : ℝ) : ℝ := x - 2

def g (x : ℝ) : ℝ := x^2 + 4 * x + 3

theorem value_of_fg_neg_one : f (g (-1)) = -2 :=
by
  sorry

end NUMINAMATH_GPT_value_of_fg_neg_one_l1628_162877


namespace NUMINAMATH_GPT_phone_plan_cost_equal_at_2500_l1628_162814

-- We define the costs C1 and C2 as described in the problem conditions.
def C1 (x : ℕ) : ℝ :=
  if x <= 500 then 50 else 50 + 0.35 * (x - 500)

def C2 (x : ℕ) : ℝ :=
  if x <= 1000 then 75 else 75 + 0.45 * (x - 1000)

-- We need to prove that the costs are equal when x = 2500.
theorem phone_plan_cost_equal_at_2500 : C1 2500 = C2 2500 := by
  sorry

end NUMINAMATH_GPT_phone_plan_cost_equal_at_2500_l1628_162814


namespace NUMINAMATH_GPT_razorback_tshirt_money_l1628_162833

noncomputable def money_made_from_texas_tech_game (tshirt_price : ℕ) (total_sold : ℕ) (arkansas_sold : ℕ) : ℕ :=
  tshirt_price * (total_sold - arkansas_sold)

theorem razorback_tshirt_money :
  money_made_from_texas_tech_game 78 186 172 = 1092 := by
  sorry

end NUMINAMATH_GPT_razorback_tshirt_money_l1628_162833


namespace NUMINAMATH_GPT_sum_of_d_and_e_l1628_162885

theorem sum_of_d_and_e (d e : ℤ) : 
  (∃ d e : ℤ, ∀ x : ℝ, x^2 - 24 * x + 50 = (x + d)^2 + e) → d + e = -106 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_d_and_e_l1628_162885


namespace NUMINAMATH_GPT_problem_statement_l1628_162862

theorem problem_statement (x y : ℝ) (log2_3 log5_3 : ℝ)
  (h1 : log2_3 > 1)
  (h2 : 0 < log5_3)
  (h3 : log5_3 < 1)
  (h4 : log2_3^x - log5_3^x ≥ log2_3^(-y) - log5_3^(-y)) :
  x + y ≥ 0 := 
sorry

end NUMINAMATH_GPT_problem_statement_l1628_162862


namespace NUMINAMATH_GPT_doubling_period_l1628_162899

theorem doubling_period (initial_capacity: ℝ) (final_capacity: ℝ) (years: ℝ) (initial_year: ℝ) (final_year: ℝ) (doubling_period: ℝ) :
  initial_capacity = 0.4 → final_capacity = 4100 → years = (final_year - initial_year) →
  initial_year = 2000 → final_year = 2050 →
  2 ^ (years / doubling_period) * initial_capacity = final_capacity :=
by
  intros h_initial h_final h_years h_i_year h_f_year
  sorry

end NUMINAMATH_GPT_doubling_period_l1628_162899


namespace NUMINAMATH_GPT_part1_solution_set_part2_range_of_a_l1628_162898

-- Definitions of f and g as provided in the problem.
def f (x : ℝ) : ℝ := |x + 1| + |x - 2|
def g (x a : ℝ) : ℝ := |x + 1| - |x - a| + a

-- Problem 1: Prove the solution set for f(x) ≤ 5 is [-2, 3]
theorem part1_solution_set : { x : ℝ | f x ≤ 5 } = { x : ℝ | -2 ≤ x ∧ x ≤ 3 } :=
  sorry

-- Problem 2: Prove the range of a when f(x) ≥ g(x) always holds is (-∞, 1]
theorem part2_range_of_a (a : ℝ) (h : ∀ x : ℝ, f x ≥ g x a) : a ≤ 1 :=
  sorry

end NUMINAMATH_GPT_part1_solution_set_part2_range_of_a_l1628_162898


namespace NUMINAMATH_GPT_evaluate_expression_l1628_162897

theorem evaluate_expression :
  (|(-1 : ℝ)|^2023 + (Real.sqrt 3)^2 - 2 * Real.sin (Real.pi / 6) + (1 / 2)⁻¹ = 5) :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1628_162897


namespace NUMINAMATH_GPT_Jasmine_shoe_size_l1628_162825

theorem Jasmine_shoe_size (J A : ℕ) (h1 : A = 2 * J) (h2 : J + A = 21) : J = 7 :=
by 
  sorry

end NUMINAMATH_GPT_Jasmine_shoe_size_l1628_162825


namespace NUMINAMATH_GPT_Elmo_books_count_l1628_162887

-- Define the number of books each person has
def Stu_books : ℕ := 4
def Laura_books : ℕ := 2 * Stu_books
def Elmo_books : ℕ := 3 * Laura_books

-- The theorem we need to prove
theorem Elmo_books_count : Elmo_books = 24 := by
  -- this part is skipped since no proof is required
  sorry

end NUMINAMATH_GPT_Elmo_books_count_l1628_162887


namespace NUMINAMATH_GPT_problem_l1628_162848

def f (a b c x : ℝ) : ℝ := a * x^5 + b * x^3 + c * x + 8

theorem problem 
  (a b c : ℝ) 
  (h : f a b c (-2) = 10) 
  : f a b c 2 = 6 :=
by
  sorry

end NUMINAMATH_GPT_problem_l1628_162848


namespace NUMINAMATH_GPT_not_perfect_square_l1628_162858

theorem not_perfect_square (h1 : ∃ x : ℝ, x^2 = 1 ^ 2018) 
                           (h2 : ¬ ∃ x : ℝ, x^2 = 2 ^ 2019)
                           (h3 : ∃ x : ℝ, x^2 = 3 ^ 2020)
                           (h4 : ∃ x : ℝ, x^2 = 4 ^ 2021)
                           (h5 : ∃ x : ℝ, x^2 = 6 ^ 2022) : 
  2 ^ 2019 ≠ x^2 := 
sorry

end NUMINAMATH_GPT_not_perfect_square_l1628_162858


namespace NUMINAMATH_GPT_not_equal_factorial_l1628_162842

noncomputable def permutations (n k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)

theorem not_equal_factorial (n : ℕ) :
  permutations (n + 1) n ≠ (by apply Nat.factorial n) := by
  sorry

end NUMINAMATH_GPT_not_equal_factorial_l1628_162842


namespace NUMINAMATH_GPT_lcm_of_23_46_827_l1628_162894

theorem lcm_of_23_46_827 : Nat.lcm (Nat.lcm 23 46) 827 = 38042 :=
by
  sorry

end NUMINAMATH_GPT_lcm_of_23_46_827_l1628_162894


namespace NUMINAMATH_GPT_expression_value_l1628_162869

theorem expression_value (a b : ℤ) (ha : a = -4) (hb : b = 3) : 
  -2 * a - b ^ 3 + 2 * a * b = -43 := by
  rw [ha, hb]
  sorry

end NUMINAMATH_GPT_expression_value_l1628_162869


namespace NUMINAMATH_GPT_cord_lengths_l1628_162843

noncomputable def cordLengthFirstDog (distance : ℝ) : ℝ :=
  distance / 2

noncomputable def cordLengthSecondDog (distance : ℝ) : ℝ :=
  distance / 2

noncomputable def cordLengthThirdDog (radius : ℝ) : ℝ :=
  radius

theorem cord_lengths (d1 d2 r : ℝ) (h1 : d1 = 30) (h2 : d2 = 40) (h3 : r = 20) :
  cordLengthFirstDog d1 = 15 ∧ cordLengthSecondDog d2 = 20 ∧ cordLengthThirdDog r = 20 := by
  sorry

end NUMINAMATH_GPT_cord_lengths_l1628_162843


namespace NUMINAMATH_GPT_linear_system_solution_l1628_162855

theorem linear_system_solution (x y : ℝ) (h1 : x = 2) (h2 : y = -3) : x + y = -1 :=
by
  sorry

end NUMINAMATH_GPT_linear_system_solution_l1628_162855


namespace NUMINAMATH_GPT_prob_exactly_two_passes_prob_at_least_one_fails_l1628_162863

-- Define the probabilities for students A, B, and C passing their tests.
def prob_A : ℚ := 4/5
def prob_B : ℚ := 3/5
def prob_C : ℚ := 7/10

-- Define the probabilities for students A, B, and C failing their tests.
def prob_not_A : ℚ := 1 - prob_A
def prob_not_B : ℚ := 1 - prob_B
def prob_not_C : ℚ := 1 - prob_C

-- (1) Prove that the probability of exactly two students passing is 113/250.
theorem prob_exactly_two_passes : 
  prob_A * prob_B * prob_not_C + prob_A * prob_not_B * prob_C + prob_not_A * prob_B * prob_C = 113/250 := 
sorry

-- (2) Prove that the probability that at least one student fails is 83/125.
theorem prob_at_least_one_fails : 
  1 - (prob_A * prob_B * prob_C) = 83/125 := 
sorry

end NUMINAMATH_GPT_prob_exactly_two_passes_prob_at_least_one_fails_l1628_162863


namespace NUMINAMATH_GPT_pet_store_dogs_l1628_162889

-- Define the given conditions as Lean definitions
def initial_dogs : ℕ := 2
def sunday_dogs : ℕ := 5
def monday_dogs : ℕ := 3

-- Define the total dogs calculation to use in the theorem
def total_dogs : ℕ := initial_dogs + sunday_dogs + monday_dogs

-- State the theorem
theorem pet_store_dogs : total_dogs = 10 := 
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_pet_store_dogs_l1628_162889


namespace NUMINAMATH_GPT_initial_number_of_men_l1628_162841

theorem initial_number_of_men (W : ℝ) (M : ℝ) (h1 : (M * 15) = W / 2) (h2 : ((M - 2) * 25) = W / 2) : M = 5 :=
sorry

end NUMINAMATH_GPT_initial_number_of_men_l1628_162841


namespace NUMINAMATH_GPT_charlie_coins_l1628_162822

variables (a c : ℕ)

axiom condition1 : c + 2 = 5 * (a - 2)
axiom condition2 : c - 2 = 4 * (a + 2)

theorem charlie_coins : c = 98 :=
by {
    sorry
}

end NUMINAMATH_GPT_charlie_coins_l1628_162822


namespace NUMINAMATH_GPT_parabola_intersects_x_axis_l1628_162824

-- Define the quadratic equation
def quadratic (x m : ℝ) : ℝ := x^2 + 2 * x + m - 1

-- Define the discriminant of the quadratic equation
def discriminant (m : ℝ) : ℝ := 4 - 4 * (m - 1)

-- Lean statement to prove the range of m
theorem parabola_intersects_x_axis (m : ℝ) : (∃ x : ℝ, quadratic x m = 0) ↔ m ≤ 2 := by
  sorry

end NUMINAMATH_GPT_parabola_intersects_x_axis_l1628_162824


namespace NUMINAMATH_GPT_jovana_shells_l1628_162864

theorem jovana_shells :
  let jovana_initial := 5
  let first_friend := 15
  let second_friend := 17
  jovana_initial + first_friend + second_friend = 37 := by
  sorry

end NUMINAMATH_GPT_jovana_shells_l1628_162864


namespace NUMINAMATH_GPT_inverse_of_parallel_lines_l1628_162826

theorem inverse_of_parallel_lines 
  (P Q : Prop) 
  (parallel_impl_alt_angles : P → Q) :
  (Q → P) := 
by
  sorry

end NUMINAMATH_GPT_inverse_of_parallel_lines_l1628_162826


namespace NUMINAMATH_GPT_least_number_of_cans_l1628_162829

theorem least_number_of_cans 
  (Maaza_volume : ℕ) (Pepsi_volume : ℕ) (Sprite_volume : ℕ) 
  (h1 : Maaza_volume = 80) (h2 : Pepsi_volume = 144) (h3 : Sprite_volume = 368) :
  (Maaza_volume / Nat.gcd (Nat.gcd Maaza_volume Pepsi_volume) Sprite_volume) +
  (Pepsi_volume / Nat.gcd (Nat.gcd Maaza_volume Pepsi_volume) Sprite_volume) +
  (Sprite_volume / Nat.gcd (Nat.gcd Maaza_volume Pepsi_volume) Sprite_volume) = 37 := by
  sorry

end NUMINAMATH_GPT_least_number_of_cans_l1628_162829


namespace NUMINAMATH_GPT_average_length_of_strings_l1628_162879

theorem average_length_of_strings (l1 l2 l3 : ℝ) (hl1 : l1 = 2) (hl2 : l2 = 5) (hl3 : l3 = 3) : 
  (l1 + l2 + l3) / 3 = 10 / 3 :=
by
  rw [hl1, hl2, hl3]
  change (2 + 5 + 3) / 3 = 10 / 3
  sorry

end NUMINAMATH_GPT_average_length_of_strings_l1628_162879


namespace NUMINAMATH_GPT_transform_center_l1628_162867

def point := (ℝ × ℝ)

def reflect_x_axis (p : point) : point :=
  (p.1, -p.2)

def translate_right (p : point) (d : ℝ) : point :=
  (p.1 + d, p.2)

theorem transform_center (C : point) (hx : C = (3, -4)) :
  translate_right (reflect_x_axis C) 3 = (6, 4) :=
by
  sorry

end NUMINAMATH_GPT_transform_center_l1628_162867


namespace NUMINAMATH_GPT_monotonicity_of_even_function_l1628_162823

-- Define the function and its properties
def f (m : ℝ) (x : ℝ) : ℝ := (m-1)*x^2 + 2*m*x + 3

-- A function is even if f(x) = f(-x) for all x
def is_even (g : ℝ → ℝ) : Prop := ∀ x : ℝ, g x = g (-x)

-- The main theorem statement
theorem monotonicity_of_even_function :
  ∀ (m : ℝ), is_even (f m) → (f 0 = 3) ∧ (∀ x : ℝ, f 0 x = - x^2 + 3) →
  (∀ a b, -3 < a ∧ a < b ∧ b < 1 → f 0 a < f 0 b → f 0 b > f 0 a) :=
by
  intro m
  intro h
  intro H
  sorry

end NUMINAMATH_GPT_monotonicity_of_even_function_l1628_162823


namespace NUMINAMATH_GPT_number_of_human_family_members_l1628_162844

-- Definitions for the problem
def num_birds := 4
def num_dogs := 3
def num_cats := 18
def bird_feet := 2
def dog_feet := 4
def cat_feet := 4
def human_feet := 2
def human_heads := 1

def animal_feet := (num_birds * bird_feet) + (num_dogs * dog_feet) + (num_cats * cat_feet)
def animal_heads := num_birds + num_dogs + num_cats

def total_feet (H : Nat) := animal_feet + (H * human_feet)
def total_heads (H : Nat) := animal_heads + (H * human_heads)

-- The problem statement translated to Lean
theorem number_of_human_family_members (H : Nat) : (total_feet H) = (total_heads H) + 74 → H = 7 :=
by
  sorry

end NUMINAMATH_GPT_number_of_human_family_members_l1628_162844


namespace NUMINAMATH_GPT_absolute_value_of_neg_eight_l1628_162836

/-- Absolute value of a number is the distance from 0 on the number line. -/
def absolute_value (x : ℤ) : ℤ :=
  if x >= 0 then x else -x

theorem absolute_value_of_neg_eight : absolute_value (-8) = 8 := by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_absolute_value_of_neg_eight_l1628_162836


namespace NUMINAMATH_GPT_power_sum_l1628_162876

theorem power_sum :
  (-3)^3 + (-3)^2 + (-3) + 3 + 3^2 + 3^3 = 18 :=
by
  sorry

end NUMINAMATH_GPT_power_sum_l1628_162876


namespace NUMINAMATH_GPT_unique_root_condition_l1628_162890

theorem unique_root_condition (a : ℝ) : 
  (∀ x : ℝ, x^3 + a*x^2 - 4*a*x + a^2 - 4 = 0 → ∃! x₀ : ℝ, x = x₀) ↔ a < 1 :=
by sorry

end NUMINAMATH_GPT_unique_root_condition_l1628_162890


namespace NUMINAMATH_GPT_equation_of_BC_area_of_triangle_l1628_162891

section triangle_geometry

variables (x y : ℝ)

/-- Given equations of the altitudes and vertex A, the equation of side BC is 2x + 3y + 7 = 0 -/
theorem equation_of_BC (h1 : 2 * x - 3 * y + 1 = 0) (h2 : x + y = 0) (A : ℝ × ℝ) (hA : A = (1, 2)) :
  ∃ (a b c : ℝ), (a * x + b * y + c = 0) ∧ (a, b, c) = (2, 3, 7) := 
sorry

/-- Given equations of the altitudes and vertex A, the area of triangle ABC is 45/2 -/
theorem area_of_triangle (h1 : 2 * x - 3 * y + 1 = 0) (h2 : x + y = 0) (A : ℝ × ℝ) (hA : A = (1, 2)) :
  ∃ (area : ℝ), (area = (45 / 2)) := 
sorry

end triangle_geometry

end NUMINAMATH_GPT_equation_of_BC_area_of_triangle_l1628_162891


namespace NUMINAMATH_GPT_num_candidates_l1628_162817

theorem num_candidates (n : ℕ) (h : n * (n - 1) = 30) : n = 6 :=
sorry

end NUMINAMATH_GPT_num_candidates_l1628_162817


namespace NUMINAMATH_GPT_rectangle_perimeter_bounds_l1628_162895

/-- Given 12 rectangular cardboard pieces, each measuring 4 cm in length and 3 cm in width,
  if these pieces are assembled to form a larger rectangle (possibly including squares),
  without overlapping or leaving gaps, then the minimum possible perimeter of the resulting 
  rectangle is 48 cm and the maximum possible perimeter is 102 cm. -/
theorem rectangle_perimeter_bounds (n : ℕ) (l w : ℝ) (total_area : ℝ) :
  n = 12 ∧ l = 4 ∧ w = 3 ∧ total_area = n * l * w →
  ∃ (min_perimeter max_perimeter : ℝ),
    min_perimeter = 48 ∧ max_perimeter = 102 :=
by
  intros
  sorry

end NUMINAMATH_GPT_rectangle_perimeter_bounds_l1628_162895


namespace NUMINAMATH_GPT_range_of_m_l1628_162878

noncomputable def set_A := { x : ℝ | x^2 + x - 6 = 0 }
noncomputable def set_B (m : ℝ) := { x : ℝ | m * x + 1 = 0 }

theorem range_of_m (m : ℝ) : set_A ∪ set_B m = set_A → m = 0 ∨ m = -1 / 2 ∨ m = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l1628_162878


namespace NUMINAMATH_GPT_print_time_correct_l1628_162870

-- Define the conditions
def pages_per_minute : ℕ := 23
def total_pages : ℕ := 345

-- Define the expected result
def expected_minutes : ℕ := 15

-- Prove the equivalence
theorem print_time_correct :
  total_pages / pages_per_minute = expected_minutes :=
by 
  -- Proof will be provided here
  sorry

end NUMINAMATH_GPT_print_time_correct_l1628_162870


namespace NUMINAMATH_GPT_max_value_real_roots_l1628_162802

theorem max_value_real_roots (k x1 x2 : ℝ) :
  (∀ k, k^2 + 3 * k + 5 ≥ 0) →
  (x1 + x2 = k - 2) →
  (x1 * x2 = k^2 + 3 * k + 5) →
  (x1^2 + x2^2 ≤ 18) :=
by
  intro h1 h2 h3
  sorry

end NUMINAMATH_GPT_max_value_real_roots_l1628_162802


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_condition_for_geometric_sequence_l1628_162818

theorem necessary_but_not_sufficient_condition_for_geometric_sequence
  (a b c : ℝ) :
  (∃ (r : ℝ), a = r * b ∧ b = r * c) → (b^2 = a * c) ∧ ¬((b^2 = a * c) → (∃ (r : ℝ), a = r * b ∧ b = r * c)) := 
by
  sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_condition_for_geometric_sequence_l1628_162818


namespace NUMINAMATH_GPT_Tom_needs_11_25_hours_per_week_l1628_162811

theorem Tom_needs_11_25_hours_per_week
  (summer_weeks: ℕ) (summer_weeks_val: summer_weeks = 8)
  (summer_hours_per_week: ℕ) (summer_hours_per_week_val: summer_hours_per_week = 45)
  (summer_earnings: ℝ) (summer_earnings_val: summer_earnings = 3600)
  (rest_weeks: ℕ) (rest_weeks_val: rest_weeks = 40)
  (rest_earnings_goal: ℝ) (rest_earnings_goal_val: rest_earnings_goal = 4500) :
  (rest_earnings_goal / (summer_earnings / (summer_hours_per_week * summer_weeks))) / rest_weeks = 11.25 :=
by
  simp [summer_earnings_val, rest_earnings_goal_val, summer_hours_per_week_val, summer_weeks_val]
  sorry

end NUMINAMATH_GPT_Tom_needs_11_25_hours_per_week_l1628_162811


namespace NUMINAMATH_GPT_num_dogs_correct_l1628_162868

-- Definitions based on conditions
def total_animals : ℕ := 17
def number_of_cats : ℕ := 8

-- Definition based on required proof
def number_of_dogs : ℕ := total_animals - number_of_cats

-- Proof statement
theorem num_dogs_correct : number_of_dogs = 9 :=
by
  sorry

end NUMINAMATH_GPT_num_dogs_correct_l1628_162868


namespace NUMINAMATH_GPT_catFinishesOnMondayNextWeek_l1628_162838

def morningConsumptionDaily (day : String) : ℚ := if day = "Wednesday" then 1 / 3 else 1 / 4
def eveningConsumptionDaily : ℚ := 1 / 6

def totalDailyConsumption (day : String) : ℚ :=
  morningConsumptionDaily day + eveningConsumptionDaily

-- List of days in order
def week : List String := ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

-- Total food available initially
def totalInitialFood : ℚ := 8

-- Function to calculate total food consumed until a given day
def foodConsumedUntil (day : String) : ℚ :=
  week.takeWhile (· != day) |>.foldl (λ acc d => acc + totalDailyConsumption d) 0

-- Function to determine the day when 8 cans are completely consumed
def finishingDay : String :=
  match week.find? (λ day => foodConsumedUntil day + totalDailyConsumption day = totalInitialFood) with
  | some day => day
  | none => "Monday"  -- If no exact match is found in the first week, it is Monday of the next week

theorem catFinishesOnMondayNextWeek :
  finishingDay = "Monday" := by
  sorry

end NUMINAMATH_GPT_catFinishesOnMondayNextWeek_l1628_162838


namespace NUMINAMATH_GPT_tan_eq_tan_of_period_for_405_l1628_162800

theorem tan_eq_tan_of_period_for_405 (m : ℤ) (h : -180 < m ∧ m < 180) :
  (Real.tan (m * (Real.pi / 180))) = (Real.tan (405 * (Real.pi / 180))) ↔ m = 45 ∨ m = -135 :=
by sorry

end NUMINAMATH_GPT_tan_eq_tan_of_period_for_405_l1628_162800


namespace NUMINAMATH_GPT_sum_series_eq_eight_l1628_162866

noncomputable def sum_series : ℝ := ∑' n : ℕ, (3 * (n + 1) + 2) / 2^(n + 1)

theorem sum_series_eq_eight : sum_series = 8 := 
 by
  sorry

end NUMINAMATH_GPT_sum_series_eq_eight_l1628_162866


namespace NUMINAMATH_GPT_problem1_problem2_problem3_l1628_162827

-- Define the functions f and g
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c
def g (a b : ℝ) (x : ℝ) : ℝ := a * x + b

-- Problem statements in Lean
theorem problem1 (a b c : ℝ) (h : ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → |f a b c x| ≤ 1) : |c| ≤ 1 :=
sorry

theorem problem2 (a b c : ℝ) (h₁ : ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → |f a b c x| ≤ 1) : 
  ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → |g a b x| ≤ 2 :=
sorry

theorem problem3 (a b c : ℝ) (ha : a > 0) (hx : ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → g a b x ≤ 2) (hf : ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → |f a b c x| ≤ 1) :
  ∃ x, -1 ≤ x ∧ x ≤ 1 ∧ g a b x = 2 :=
sorry

end NUMINAMATH_GPT_problem1_problem2_problem3_l1628_162827


namespace NUMINAMATH_GPT_minimum_cuts_to_unit_cubes_l1628_162860

def cubes := List (ℕ × ℕ × ℕ)

def cube_cut (c : cubes) (n : ℕ) (dim : ℕ) : cubes :=
  sorry -- Function body not required for the statement

theorem minimum_cuts_to_unit_cubes (c : cubes) (s : ℕ) (dim : ℕ) :
  c = [(4,4,4)] ∧ s = 64 ∧ dim = 3 →
  ∃ (n : ℕ), n = 9 ∧
    (∀ cuts : cubes, cube_cut cuts n dim = [(1,1,1)]) :=
sorry

end NUMINAMATH_GPT_minimum_cuts_to_unit_cubes_l1628_162860


namespace NUMINAMATH_GPT_number_of_paths_to_spell_MATH_l1628_162852

-- Define the problem setting and conditions
def number_of_paths_M_to_H (adj: ℕ) (steps: ℕ): ℕ :=
  adj^(steps-1)

-- State the problem in Lean 4
theorem number_of_paths_to_spell_MATH : number_of_paths_M_to_H 8 4 = 512 := 
by 
  unfold number_of_paths_M_to_H 
  -- The needed steps are included:
  -- We calculate: 8^(4-1) = 8^3 which should be 512.
  sorry

end NUMINAMATH_GPT_number_of_paths_to_spell_MATH_l1628_162852


namespace NUMINAMATH_GPT_non_empty_prime_subsets_count_l1628_162853

-- Definition of the set S
def S : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

-- Definition of primes in S
def prime_subset_S : Set ℕ := {x ∈ S | Nat.Prime x}

-- The statement to prove
theorem non_empty_prime_subsets_count : 
  ∃ n, n = 15 ∧ ∀ T ⊆ prime_subset_S, T ≠ ∅ → ∃ m, n = 2^m - 1 := 
by
  sorry

end NUMINAMATH_GPT_non_empty_prime_subsets_count_l1628_162853


namespace NUMINAMATH_GPT_john_needs_one_plank_l1628_162851

theorem john_needs_one_plank (total_nails : ℕ) (nails_per_plank : ℕ) (extra_nails : ℕ) (P : ℕ)
    (h1 : total_nails = 11)
    (h2 : nails_per_plank = 3)
    (h3 : extra_nails = 8)
    (h4 : total_nails = nails_per_plank * P + extra_nails) :
    P = 1 :=
by
    sorry

end NUMINAMATH_GPT_john_needs_one_plank_l1628_162851


namespace NUMINAMATH_GPT_geometric_sequence_general_term_l1628_162865

theorem geometric_sequence_general_term (n : ℕ) (a : ℕ → ℕ) (a1 : ℕ) (q : ℕ) 
  (h1 : a1 = 4) (h2 : q = 3) (h3 : ∀ n, a n = a1 * (q ^ (n - 1))) :
  a n = 4 * 3^(n - 1) := by
  sorry

end NUMINAMATH_GPT_geometric_sequence_general_term_l1628_162865


namespace NUMINAMATH_GPT_angle_perpendicular_sides_l1628_162883

theorem angle_perpendicular_sides (α β : ℝ) (hα : α = 80) 
  (h_perp : ∀ {x y}, ((x = α → y = 180 - x) ∨ (y = 180 - α → x = y))) : 
  β = 80 ∨ β = 100 :=
by
  sorry

end NUMINAMATH_GPT_angle_perpendicular_sides_l1628_162883


namespace NUMINAMATH_GPT_positive_difference_for_6_points_l1628_162820

def combinations (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

def positiveDifferenceTrianglesAndQuadrilaterals (n : ℕ) : ℕ :=
  combinations n 3 - combinations n 4

theorem positive_difference_for_6_points : positiveDifferenceTrianglesAndQuadrilaterals 6 = 5 :=
by
  sorry

end NUMINAMATH_GPT_positive_difference_for_6_points_l1628_162820


namespace NUMINAMATH_GPT_trains_meet_after_time_l1628_162815

/-- Given the lengths of two trains, the initial distance between them, and their speeds,
prove that they will meet after approximately 2.576 seconds. --/
theorem trains_meet_after_time 
  (length_train1 : ℝ) (length_train2 : ℝ) (initial_distance : ℝ)
  (speed_train1_kmph : ℝ) (speed_train2_mps : ℝ) :
  length_train1 = 87.5 →
  length_train2 = 94.3 →
  initial_distance = 273.2 →
  speed_train1_kmph = 65 →
  speed_train2_mps = 88 →
  abs ((initial_distance / ((speed_train1_kmph * 1000 / 3600) + speed_train2_mps)) - 2.576) < 0.001 := by
  sorry

end NUMINAMATH_GPT_trains_meet_after_time_l1628_162815


namespace NUMINAMATH_GPT_min_people_in_group_l1628_162801

theorem min_people_in_group (B G : ℕ) (h : B / (B + G : ℝ) > 0.94) : B + G ≥ 17 :=
sorry

end NUMINAMATH_GPT_min_people_in_group_l1628_162801


namespace NUMINAMATH_GPT_generate_13121_not_generate_12131_l1628_162821

theorem generate_13121 : ∃ n m : ℕ, 13121 + 1 = 2^n * 3^m := by
  sorry

theorem not_generate_12131 : ¬∃ n m : ℕ, 12131 + 1 = 2^n * 3^m := by
  sorry

end NUMINAMATH_GPT_generate_13121_not_generate_12131_l1628_162821
