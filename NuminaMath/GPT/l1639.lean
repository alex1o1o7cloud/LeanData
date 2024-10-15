import Mathlib

namespace NUMINAMATH_GPT_induction_first_step_l1639_163951

theorem induction_first_step (n : ℕ) (h₁ : n > 1) : 
  1 + 1/2 + 1/3 < 2 := 
sorry

end NUMINAMATH_GPT_induction_first_step_l1639_163951


namespace NUMINAMATH_GPT_find_a2_l1639_163919

variables {α : Type*} [LinearOrderedField α]

def geometric_sequence (a : ℕ → α) : Prop :=
  ∀ n m : ℕ, ∃ r : α, a (n + m) = (a n) * (a m) * r

theorem find_a2 (a : ℕ → α) (h_geom : geometric_sequence a) (h1 : a 3 * a 6 = 9) (h2 : a 2 * a 4 * a 5 = 27) :
  a 2 = 3 :=
sorry

end NUMINAMATH_GPT_find_a2_l1639_163919


namespace NUMINAMATH_GPT_original_number_people_l1639_163995

theorem original_number_people (n : ℕ) (h1 : n / 3 * 2 / 2 = 18) : n = 54 :=
sorry

end NUMINAMATH_GPT_original_number_people_l1639_163995


namespace NUMINAMATH_GPT_parabola_points_relationship_l1639_163986

theorem parabola_points_relationship (c y1 y2 y3 : ℝ)
  (h1 : y1 = -0^2 + 2 * 0 + c)
  (h2 : y2 = -1^2 + 2 * 1 + c)
  (h3 : y3 = -3^2 + 2 * 3 + c) :
  y2 > y1 ∧ y1 > y3 := by
  sorry

end NUMINAMATH_GPT_parabola_points_relationship_l1639_163986


namespace NUMINAMATH_GPT_arithmetic_mean_pq_is_10_l1639_163984

variables {p q r : ℝ}

theorem arithmetic_mean_pq_is_10 
  (h1 : (p + q) / 2 = 10)
  (h2 : (q + r) / 2 = 27)
  (h3 : r - p = 34) 
  : (p + q) / 2 = 10 :=
by 
  exact h1

end NUMINAMATH_GPT_arithmetic_mean_pq_is_10_l1639_163984


namespace NUMINAMATH_GPT_total_students_correct_l1639_163991

def students_in_school : ℕ :=
  let students_per_class := 23
  let classes_per_grade := 12
  let grades_per_school := 3
  students_per_class * classes_per_grade * grades_per_school

theorem total_students_correct :
  students_in_school = 828 :=
by
  sorry

end NUMINAMATH_GPT_total_students_correct_l1639_163991


namespace NUMINAMATH_GPT_max_y_value_l1639_163922

-- Definitions according to the problem conditions
def is_negative_integer (z : ℤ) : Prop := z < 0

-- The theorem to be proven
theorem max_y_value (x y : ℤ) (hx : is_negative_integer x) (hy : is_negative_integer y) 
  (h_eq : y = 10 * x / (10 - x)) : y = -5 :=
sorry

end NUMINAMATH_GPT_max_y_value_l1639_163922


namespace NUMINAMATH_GPT_tangent_curve_l1639_163918

theorem tangent_curve (a : ℝ) : 
  (∃ x : ℝ, 3 * x - 2 = x^3 - 2 * a ∧ 3 * x^2 = 3) →
  a = 0 ∨ a = 2 := 
sorry

end NUMINAMATH_GPT_tangent_curve_l1639_163918


namespace NUMINAMATH_GPT_min_value_P_l1639_163934

-- Define the polynomial P
def P (x y : ℝ) : ℝ := x^2 + y^2 - 6*x + 8*y + 7

-- Theorem statement: The minimum value of P(x, y) is -18
theorem min_value_P : ∃ (x y : ℝ), P x y = -18 := by
  sorry

end NUMINAMATH_GPT_min_value_P_l1639_163934


namespace NUMINAMATH_GPT_Paul_lost_161_crayons_l1639_163983

def total_crayons : Nat := 589
def crayons_given : Nat := 571
def extra_crayons_given : Nat := 410

theorem Paul_lost_161_crayons : ∃ L : Nat, crayons_given = L + extra_crayons_given ∧ L = 161 := by
  sorry

end NUMINAMATH_GPT_Paul_lost_161_crayons_l1639_163983


namespace NUMINAMATH_GPT_find_a_l1639_163958

noncomputable def f (x a : ℝ) : ℝ := (2 * x + a) ^ 2

theorem find_a (a : ℝ) (h1 : f 2 a = 20) : a = 1 :=
sorry

end NUMINAMATH_GPT_find_a_l1639_163958


namespace NUMINAMATH_GPT_probability_heads_exactly_9_of_12_l1639_163955

noncomputable def bin_coeff (n k : ℕ) : ℕ := Nat.choose n k

noncomputable def probability_heads_9_of_12_flips : ℚ :=
  (bin_coeff 12 9 : ℚ) / (2 ^ 12)

theorem probability_heads_exactly_9_of_12 :
  probability_heads_9_of_12_flips = 220 / 4096 :=
by
  sorry

end NUMINAMATH_GPT_probability_heads_exactly_9_of_12_l1639_163955


namespace NUMINAMATH_GPT_original_bullets_per_person_l1639_163905

theorem original_bullets_per_person (x : ℕ) (h : 5 * (x - 4) = x) : x = 5 :=
by
  sorry

end NUMINAMATH_GPT_original_bullets_per_person_l1639_163905


namespace NUMINAMATH_GPT_relation_between_x_and_y_l1639_163981

-- Definitions based on the conditions
variables (r x y : ℝ)

-- Power of a Point Theorem and provided conditions
variables (AE_eq_3EC : AE = 3 * EC)
variables (x_def : x = AE)
variables (y_def : y = r)

-- Main statement to be proved
theorem relation_between_x_and_y (r x y : ℝ) (AE_eq_3EC : AE = 3 * EC) (x_def : x = AE) (y_def : y = r) :
  y^2 = x^3 / (2 * r - x) :=
sorry

end NUMINAMATH_GPT_relation_between_x_and_y_l1639_163981


namespace NUMINAMATH_GPT_billy_watches_videos_l1639_163961

-- Conditions definitions
def num_suggestions_per_list : Nat := 15
def num_iterations : Nat := 5
def pick_index_on_final_list : Nat := 5

-- Main theorem statement
theorem billy_watches_videos : 
  num_suggestions_per_list * num_iterations + (pick_index_on_final_list - 1) = 79 :=
by
  sorry

end NUMINAMATH_GPT_billy_watches_videos_l1639_163961


namespace NUMINAMATH_GPT_initial_games_l1639_163968

-- Conditions
def games_given_away : ℕ := 7
def games_left : ℕ := 91

-- Theorem Statement
theorem initial_games (initial_games : ℕ) : 
  initial_games = games_left + games_given_away :=
by
  sorry

end NUMINAMATH_GPT_initial_games_l1639_163968


namespace NUMINAMATH_GPT_find_length_AB_l1639_163930

noncomputable def length_of_AB (DE DF : ℝ) (AC : ℝ) : ℝ :=
  (AC * DE) / DF

theorem find_length_AB (DE DF AC : ℝ) (pro1 : DE = 9) (pro2 : DF = 17) (pro3 : AC = 10) :
    length_of_AB DE DF AC = 90 / 17 :=
  by
    rw [pro1, pro2, pro3]
    unfold length_of_AB
    norm_num

end NUMINAMATH_GPT_find_length_AB_l1639_163930


namespace NUMINAMATH_GPT_min_distance_origin_to_line_l1639_163976

theorem min_distance_origin_to_line (a b : ℝ) (h : a + 2 * b = Real.sqrt 5) : 
  Real.sqrt (a^2 + b^2) ≥ 1 :=
sorry

end NUMINAMATH_GPT_min_distance_origin_to_line_l1639_163976


namespace NUMINAMATH_GPT_jordan_time_to_run_7_miles_l1639_163965

def time_taken (distance time_per_unit : ℝ) : ℝ :=
  distance * time_per_unit

theorem jordan_time_to_run_7_miles :
  ∀ (t_S d_S d_J : ℝ), t_S = 36 → d_S = 6 → d_J = 4 → time_taken 7 ((t_S / 2) / d_J) = 31.5 :=
by
  intros t_S d_S d_J h_t_S h_d_S h_d_J
  -- skipping the proof
  sorry

end NUMINAMATH_GPT_jordan_time_to_run_7_miles_l1639_163965


namespace NUMINAMATH_GPT_initial_time_to_cover_distance_l1639_163929

theorem initial_time_to_cover_distance (s t : ℝ) (h1 : 540 = s * t) (h2 : 540 = 60 * (3/4) * t) : t = 12 :=
sorry

end NUMINAMATH_GPT_initial_time_to_cover_distance_l1639_163929


namespace NUMINAMATH_GPT_intersection_S_T_eq_l1639_163964

def S : Set ℝ := { x | (x - 2) * (x - 3) ≥ 0 }
def T : Set ℝ := { x | x > 0 }

theorem intersection_S_T_eq : (S ∩ T) = { x | (0 < x ∧ x ≤ 2) ∨ (x ≥ 3) } :=
by
  sorry

end NUMINAMATH_GPT_intersection_S_T_eq_l1639_163964


namespace NUMINAMATH_GPT_isosceles_triangle_angle_sum_l1639_163969

theorem isosceles_triangle_angle_sum (y : ℕ) (a : ℕ) (b : ℕ) 
  (h_isosceles : a = b ∨ a = y ∨ b = y)
  (h_sum : a + b + y = 180) :
  a = 80 → b = 80 → y = 50 ∨ y = 20 ∨ y = 80 → y + y + y = 150 :=
by
  sorry

end NUMINAMATH_GPT_isosceles_triangle_angle_sum_l1639_163969


namespace NUMINAMATH_GPT_correct_q_solution_l1639_163949

noncomputable def solve_q (n m q : ℕ) : Prop :=
  (7 / 8 : ℚ) = (n / 96 : ℚ) ∧
  (7 / 8 : ℚ) = ((m + n) / 112 : ℚ) ∧
  (7 / 8 : ℚ) = ((q - m) / 144 : ℚ) ∧
  n = 84 ∧
  m = 14 →
  q = 140

theorem correct_q_solution : ∃ (q : ℕ), solve_q 84 14 q :=
by sorry

end NUMINAMATH_GPT_correct_q_solution_l1639_163949


namespace NUMINAMATH_GPT_mobile_price_two_years_ago_l1639_163998

-- Definitions and conditions
def price_now : ℝ := 1000
def decrease_rate : ℝ := 0.2
def years_ago : ℝ := 2

-- Main statement
theorem mobile_price_two_years_ago :
  ∃ (a : ℝ), a * (1 - decrease_rate)^years_ago = price_now :=
sorry

end NUMINAMATH_GPT_mobile_price_two_years_ago_l1639_163998


namespace NUMINAMATH_GPT_parallel_lines_implies_m_neg1_l1639_163928

theorem parallel_lines_implies_m_neg1 (m : ℝ) :
  (∀ (x y : ℝ), x + m * y + 6 = 0) ∧
  (∀ (x y : ℝ), (m - 2) * x + 3 * y + 2 * m = 0) ∧
  ∀ (l₁ l₂ : ℝ), l₁ = -(1 / m) ∧ l₂ = -((m - 2) / 3) ∧ l₁ = l₂ → m = -1 :=
by
  sorry

end NUMINAMATH_GPT_parallel_lines_implies_m_neg1_l1639_163928


namespace NUMINAMATH_GPT_total_goals_in_5_matches_l1639_163950

theorem total_goals_in_5_matches 
  (x : ℝ) 
  (h1 : 4 * x + 3 = 5 * (x + 0.2)) 
  : 4 * x + 3 = 11 :=
by
  -- The proof is omitted here
  sorry

end NUMINAMATH_GPT_total_goals_in_5_matches_l1639_163950


namespace NUMINAMATH_GPT_range_of_a_minus_b_l1639_163932

theorem range_of_a_minus_b (a b : ℝ) (h1 : 1 < a ∧ a < 4) (h2 : -2 < b ∧ b < 4) : -3 < a - b ∧ a - b < 6 :=
sorry

end NUMINAMATH_GPT_range_of_a_minus_b_l1639_163932


namespace NUMINAMATH_GPT_sin_60_eq_sqrt3_div_2_l1639_163987

theorem sin_60_eq_sqrt3_div_2 : Real.sin (60 * Real.pi / 180) = Real.sqrt 3 / 2 := by
  -- proof skipped
  sorry

end NUMINAMATH_GPT_sin_60_eq_sqrt3_div_2_l1639_163987


namespace NUMINAMATH_GPT_discount_percentage_l1639_163967

theorem discount_percentage
  (MP CP SP : ℝ)
  (h1 : CP = 0.64 * MP)
  (h2 : 37.5 = (SP - CP) / CP * 100) :
  (MP - SP) / MP * 100 = 12 := by
  sorry

end NUMINAMATH_GPT_discount_percentage_l1639_163967


namespace NUMINAMATH_GPT_solution_set_of_inequality_l1639_163917

theorem solution_set_of_inequality :
  { x : ℝ | (x + 3) * (6 - x) ≥ 0 } = { x : ℝ | -3 ≤ x ∧ x ≤ 6 } :=
sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l1639_163917


namespace NUMINAMATH_GPT_one_third_recipe_ingredients_l1639_163971

noncomputable def cups_of_flour (f : ℚ) := (f : ℚ)
noncomputable def cups_of_sugar (s : ℚ) := (s : ℚ)
def original_recipe_flour := (27 / 4 : ℚ)  -- mixed number 6 3/4 converted to improper fraction
def original_recipe_sugar := (5 / 2 : ℚ)  -- mixed number 2 1/2 converted to improper fraction

theorem one_third_recipe_ingredients :
  cups_of_flour (original_recipe_flour / 3) = (9 / 4) ∧
  cups_of_sugar (original_recipe_sugar / 3) = (5 / 6) :=
by
  sorry

end NUMINAMATH_GPT_one_third_recipe_ingredients_l1639_163971


namespace NUMINAMATH_GPT_trailingZeros_310_fact_l1639_163953

-- Define the function to compute trailing zeros in factorials
def trailingZeros (n : ℕ) : ℕ := 
  if n = 0 then 0 else n / 5 + trailingZeros (n / 5)

-- Define the specific case for 310!
theorem trailingZeros_310_fact : trailingZeros 310 = 76 := 
by 
  sorry

end NUMINAMATH_GPT_trailingZeros_310_fact_l1639_163953


namespace NUMINAMATH_GPT_negation_of_universal_proposition_l1639_163944

theorem negation_of_universal_proposition :
  (¬ (∀ x : ℝ, x^2 ≥ 0)) ↔ ∃ x : ℝ, x^2 < 0 :=
  sorry

end NUMINAMATH_GPT_negation_of_universal_proposition_l1639_163944


namespace NUMINAMATH_GPT_simplify_expression_l1639_163948

variable (a : ℝ)

theorem simplify_expression : 3 * a^2 - a * (2 * a - 1) = a^2 + a :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1639_163948


namespace NUMINAMATH_GPT_ball_hits_ground_at_two_seconds_l1639_163935

theorem ball_hits_ground_at_two_seconds :
  (∃ t : ℝ, (-6.1) * t^2 + 2.8 * t + 7 = 0 ∧ t = 2) :=
sorry

end NUMINAMATH_GPT_ball_hits_ground_at_two_seconds_l1639_163935


namespace NUMINAMATH_GPT_find_price_max_profit_l1639_163989

/-
Part 1: Prove the price per unit of type A and B
-/

def price_per_unit (x y : ℕ) : Prop :=
  (2 * x + 3 * y = 690) ∧ (x + 4 * y = 720)

theorem find_price :
  ∃ x y : ℕ, price_per_unit x y ∧ x = 120 ∧ y = 150 :=
by
  sorry

/-
Part 2: Prove the maximum profit with constraints
-/

def conditions (m : ℕ) : Prop :=
  m ≤ 3 * (40 - m) ∧ 120 * m + 150 * (40 - m) ≤ 5400

def profit (m : ℕ) : ℕ :=
  (160 - 120) * m + (200 - 150) * (40 - m)

theorem max_profit :
  ∃ m : ℕ, 20 ≤ m ∧ m ≤ 30 ∧ conditions m ∧ profit m = profit 20 :=
by
  sorry

end NUMINAMATH_GPT_find_price_max_profit_l1639_163989


namespace NUMINAMATH_GPT_total_cost_of_dishes_l1639_163909

theorem total_cost_of_dishes
  (e t b : ℝ)
  (h1 : 4 * e + 5 * t + 2 * b = 8.20)
  (h2 : 6 * e + 3 * t + 4 * b = 9.40) :
  5 * e + 6 * t + 3 * b = 12.20 := 
sorry

end NUMINAMATH_GPT_total_cost_of_dishes_l1639_163909


namespace NUMINAMATH_GPT_Nancy_weighs_90_pounds_l1639_163941

theorem Nancy_weighs_90_pounds (W : ℝ) (h1 : 0.60 * W = 54) : W = 90 :=
by
  sorry

end NUMINAMATH_GPT_Nancy_weighs_90_pounds_l1639_163941


namespace NUMINAMATH_GPT_slope_of_line_is_pm1_l1639_163956

noncomputable def polarCurve (θ : ℝ) : ℝ := 2 * Real.cos θ - 4 * Real.sin θ

noncomputable def lineParametric (t α : ℝ) : ℝ × ℝ := (1 + t * Real.cos α, -1 + t * Real.sin α)

theorem slope_of_line_is_pm1
  (t α : ℝ)
  (hAB : ∃ A B : ℝ × ℝ, lineParametric t α = A ∧ (∃ t1 t2 : ℝ, A = lineParametric t1 α ∧ B = lineParametric t2 α ∧ dist A B = 3 * Real.sqrt 2))
  (hC : ∃ θ : ℝ, polarCurve θ = dist (1, -1) (polarCurve θ * Real.cos θ, polarCurve θ * Real.sin θ)) :
  ∃ k : ℝ, k = 1 ∨ k = -1 :=
sorry

end NUMINAMATH_GPT_slope_of_line_is_pm1_l1639_163956


namespace NUMINAMATH_GPT_remainder_1234567_div_by_137_l1639_163980

theorem remainder_1234567_div_by_137 :
  (1234567 % 137) = 102 :=
by {
  sorry
}

end NUMINAMATH_GPT_remainder_1234567_div_by_137_l1639_163980


namespace NUMINAMATH_GPT_max_dogs_and_fish_l1639_163960

theorem max_dogs_and_fish (d c b p f : ℕ) (h_ratio : d / 7 = c / 7 ∧ d / 7 = b / 8 ∧ d / 7 = p / 3 ∧ d / 7 = f / 5)
  (h_dogs_bunnies : d + b = 330)
  (h_twice_fish : f ≥ 2 * c) :
  d = 154 ∧ f = 308 :=
by
  -- This is where the proof would go
  sorry

end NUMINAMATH_GPT_max_dogs_and_fish_l1639_163960


namespace NUMINAMATH_GPT_required_force_l1639_163912

theorem required_force (m : ℝ) (g : ℝ) (T : ℝ) (F : ℝ) 
    (h1 : m = 3)
    (h2 : g = 10)
    (h3 : T = m * g)
    (h4 : F = 4 * T) : F = 120 := by
  sorry

end NUMINAMATH_GPT_required_force_l1639_163912


namespace NUMINAMATH_GPT_train_speed_l1639_163962

theorem train_speed
  (length_of_train : ℝ)
  (time_to_cross_pole : ℝ)
  (h1 : length_of_train = 3000)
  (h2 : time_to_cross_pole = 120) :
  length_of_train / time_to_cross_pole = 25 :=
by {
  sorry
}

end NUMINAMATH_GPT_train_speed_l1639_163962


namespace NUMINAMATH_GPT_cos_double_plus_cos_l1639_163902

theorem cos_double_plus_cos (α : ℝ) (h : Real.sin (Real.pi / 2 + α) = 1 / 3) :
  Real.cos (2 * α) + Real.cos α = -4 / 9 :=
by
  sorry

end NUMINAMATH_GPT_cos_double_plus_cos_l1639_163902


namespace NUMINAMATH_GPT_students_not_enrolled_in_bio_l1639_163973

theorem students_not_enrolled_in_bio (total_students : ℕ) (p : ℕ) (p_half : p = (total_students / 2)) (total_students_eq : total_students = 880) : 
  total_students - p = 440 :=
by sorry

end NUMINAMATH_GPT_students_not_enrolled_in_bio_l1639_163973


namespace NUMINAMATH_GPT_sum_of_powers_divisible_by_30_l1639_163910

theorem sum_of_powers_divisible_by_30 {a b c : ℤ} (h : (a + b + c) % 30 = 0) : (a^5 + b^5 + c^5) % 30 = 0 := by
  sorry

end NUMINAMATH_GPT_sum_of_powers_divisible_by_30_l1639_163910


namespace NUMINAMATH_GPT_value_of_star_l1639_163957

theorem value_of_star : 
  ∀ (star : ℤ), 45 - (28 - (37 - (15 - star))) = 59 → star = -154 :=
by
  intro star
  intro h
  -- Proof to be provided
  sorry

end NUMINAMATH_GPT_value_of_star_l1639_163957


namespace NUMINAMATH_GPT_george_boxes_of_eggs_l1639_163937

theorem george_boxes_of_eggs (boxes_eggs : Nat) (h1 : ∀ (eggs_per_box : Nat), eggs_per_box = 3 → boxes_eggs * eggs_per_box = 15) :
  boxes_eggs = 5 :=
by
  sorry

end NUMINAMATH_GPT_george_boxes_of_eggs_l1639_163937


namespace NUMINAMATH_GPT_circle_radius_tangent_lines_l1639_163952

noncomputable def circle_radius (k : ℝ) (r : ℝ) : Prop :=
  k > 8 ∧ r = k / Real.sqrt 2 ∧ r = |k - 8|

theorem circle_radius_tangent_lines :
  ∃ k r : ℝ, k > 8 ∧ r = (k / Real.sqrt 2) ∧ r = |k - 8| ∧ r = 8 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_circle_radius_tangent_lines_l1639_163952


namespace NUMINAMATH_GPT_sum_values_l1639_163963

noncomputable def abs_eq_4 (x : ℝ) : Prop := |x| = 4
noncomputable def abs_eq_5 (x : ℝ) : Prop := |x| = 5

theorem sum_values (a b : ℝ) (h₁ : abs_eq_4 a) (h₂ : abs_eq_5 b) :
  a + b = 9 ∨ a + b = -1 ∨ a + b = 1 ∨ a + b = -9 := 
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_sum_values_l1639_163963


namespace NUMINAMATH_GPT_sum_first_ten_terms_arithmetic_sequence_l1639_163927

theorem sum_first_ten_terms_arithmetic_sequence (a d : ℝ) (S10 : ℝ) 
  (h1 : 0 < d) 
  (h2 : (a - d) + a + (a + d) = -6) 
  (h3 : (a - d) * a * (a + d) = 10) 
  (h4 : S10 = 5 * (2 * (a - d) + 9 * d)) :
  S10 = -20 + 35 * Real.sqrt 6.5 :=
by sorry

end NUMINAMATH_GPT_sum_first_ten_terms_arithmetic_sequence_l1639_163927


namespace NUMINAMATH_GPT_binomial_probability_l1639_163992

-- Define the binomial coefficient function
def binomial_coeff (n k : ℕ) : ℕ := Nat.choose n k

-- Define the binomial probability mass function
def binomial_pmf (n k : ℕ) (p : ℚ) : ℚ :=
  (binomial_coeff n k) * (p^k) * ((1 - p)^(n - k))

-- Define the conditions of the problem
def n := 5
def k := 2
def p : ℚ := 1/3

-- State the theorem
theorem binomial_probability :
  binomial_pmf n k p = binomial_coeff 5 2 * (1/3)^2 * (2/3)^3 := by
  sorry

end NUMINAMATH_GPT_binomial_probability_l1639_163992


namespace NUMINAMATH_GPT_katya_minimum_problems_l1639_163947

-- Defining the conditions
def katya_probability_solve : ℚ := 4 / 5
def pen_probability_solve : ℚ := 1 / 2
def total_problems : ℕ := 20
def minimum_correct_for_good_grade : ℚ := 13

-- The Lean statement to prove
theorem katya_minimum_problems (x : ℕ) :
  x * katya_probability_solve + (total_problems - x) * pen_probability_solve ≥ minimum_correct_for_good_grade → x ≥ 10 :=
sorry

end NUMINAMATH_GPT_katya_minimum_problems_l1639_163947


namespace NUMINAMATH_GPT_smallest_number_of_students_l1639_163946

theorem smallest_number_of_students 
  (A6 A7 A8 : Nat)
  (h1 : A8 * 3 = A6 * 5)
  (h2 : A8 * 5 = A7 * 8) :
  A6 + A7 + A8 = 89 :=
sorry

end NUMINAMATH_GPT_smallest_number_of_students_l1639_163946


namespace NUMINAMATH_GPT_diagonal_pairs_forming_60_degrees_l1639_163977

theorem diagonal_pairs_forming_60_degrees :
  let total_diagonals := 12
  let total_pairs := total_diagonals.choose 2
  let non_forming_pairs_per_set := 6
  let sets_of_parallel_planes := 3
  total_pairs - non_forming_pairs_per_set * sets_of_parallel_planes = 48 :=
by 
  let total_diagonals := 12
  let total_pairs := total_diagonals.choose 2
  let non_forming_pairs_per_set := 6
  let sets_of_parallel_planes := 3
  have calculation : total_pairs - non_forming_pairs_per_set * sets_of_parallel_planes = 48 := sorry
  exact calculation

end NUMINAMATH_GPT_diagonal_pairs_forming_60_degrees_l1639_163977


namespace NUMINAMATH_GPT_new_supervisor_salary_l1639_163943

theorem new_supervisor_salary
  (W S1 S2 : ℝ)
  (avg_old : (W + S1) / 9 = 430)
  (S1_val : S1 = 870)
  (avg_new : (W + S2) / 9 = 410) :
  S2 = 690 :=
by
  sorry

end NUMINAMATH_GPT_new_supervisor_salary_l1639_163943


namespace NUMINAMATH_GPT_first_step_is_remove_parentheses_l1639_163939

variable (x : ℝ)

def equation : Prop := 2 * x + 3 * (2 * x - 1) = 16 - (x + 1)

theorem first_step_is_remove_parentheses (x : ℝ) (eq : equation x) : 
  ∃ step : String, step = "remove the parentheses" := 
  sorry

end NUMINAMATH_GPT_first_step_is_remove_parentheses_l1639_163939


namespace NUMINAMATH_GPT_parameter_values_l1639_163942

def system_equation_1 (x y : ℝ) : Prop :=
  (|y + 9| + |x + 2| - 2) * (x^2 + y^2 - 3) = 0

def system_equation_2 (x y a : ℝ) : Prop :=
  (x + 2)^2 + (y + 4)^2 = a

theorem parameter_values (a : ℝ) :
  (∃ x y : ℝ, system_equation_1 x y ∧ system_equation_2 x y a ∧ 
    -- counting the number of solutions to the system of equations that total exactly three,
    -- meaning the system has exactly three solutions
    -- Placeholder for counting solutions
    sorry) ↔ (a = 9 ∨ a = 23 + 4 * Real.sqrt 15) := 
sorry

end NUMINAMATH_GPT_parameter_values_l1639_163942


namespace NUMINAMATH_GPT_problem_statement_l1639_163920

noncomputable def a := 2 * Real.sqrt 2
noncomputable def b := 2
def ellipse_eq (x y : ℝ) := (x^2) / 8 + (y^2) / 4 = 1
def line_eq (x y m : ℝ) := y = x + m
def circle_eq (x y : ℝ) := x^2 + y^2 = 1

theorem problem_statement (x1 y1 x2 y2 x0 y0 m : ℝ) (h1 : ellipse_eq x1 y1) (h2 : ellipse_eq x2 y2) 
  (hm : line_eq x0 y0 m) (h0 : (x1 + x2) / 2 = -2 * m / 3) (h0' : (y1 + y2) / 2 = m / 3) : 
  (ellipse_eq x y ∧ line_eq x y m ∧ circle_eq x0 y0) → m = (3 * Real.sqrt 5) / 5 ∨ m = -(3 * Real.sqrt 5) / 5 := 
by {
  sorry
}

end NUMINAMATH_GPT_problem_statement_l1639_163920


namespace NUMINAMATH_GPT_find_value_of_m_l1639_163906

def ellipse_condition (x y : ℝ) (m : ℝ) : Prop :=
  x^2 + m * y^2 = 1

theorem find_value_of_m (m : ℝ) 
  (h1 : ∀ (x y : ℝ), ellipse_condition x y m)
  (h2 : ∀ a b : ℝ, (a^2 = 1/m ∧ b^2 = 1) ∧ (a = 2 * b)) : 
  m = 1/4 :=
by
  sorry

end NUMINAMATH_GPT_find_value_of_m_l1639_163906


namespace NUMINAMATH_GPT_maria_younger_than_ann_l1639_163994

variable (M A : ℕ)

def maria_current_age : Prop := M = 7

def age_relation_four_years_ago : Prop := M - 4 = (1 / 2) * (A - 4)

theorem maria_younger_than_ann :
  maria_current_age M → age_relation_four_years_ago M A → A - M = 3 :=
by
  sorry

end NUMINAMATH_GPT_maria_younger_than_ann_l1639_163994


namespace NUMINAMATH_GPT_total_time_taken_l1639_163993

theorem total_time_taken (speed_boat : ℕ) (speed_stream : ℕ) (distance : ℕ) 
    (h1 : speed_boat = 12) (h2 : speed_stream = 4) (h3 : distance = 480) : 
    ((distance / (speed_boat + speed_stream)) + (distance / (speed_boat - speed_stream)) = 90) :=
by
  -- Sorry is used to skip the proof
  sorry

end NUMINAMATH_GPT_total_time_taken_l1639_163993


namespace NUMINAMATH_GPT_angle_parallel_lines_l1639_163915

variables {Line : Type} (a b c : Line) (theta : ℝ)
variable (angle_between : Line → Line → ℝ)

def is_parallel (a b : Line) : Prop := sorry

theorem angle_parallel_lines (h_parallel : is_parallel a b) (h_angle : angle_between a c = theta) : angle_between b c = theta := 
sorry

end NUMINAMATH_GPT_angle_parallel_lines_l1639_163915


namespace NUMINAMATH_GPT_roots_of_polynomial_l1639_163997

theorem roots_of_polynomial :
  {x | x * (2 * x - 5) ^ 2 * (x + 3) * (7 - x) = 0} = {0, 2.5, -3, 7} :=
by {
  sorry
}

end NUMINAMATH_GPT_roots_of_polynomial_l1639_163997


namespace NUMINAMATH_GPT_max_pieces_with_single_cut_min_cuts_to_intersect_all_pieces_l1639_163966

theorem max_pieces_with_single_cut (n : ℕ) (h : n = 4) :
  (∃ m : ℕ, m = 23) :=
sorry

theorem min_cuts_to_intersect_all_pieces (n : ℕ) (h : n = 4) :
  (∃ k : ℕ, k = 3) :=
sorry

noncomputable def pieces_of_cake : ℕ := 23

noncomputable def cuts_required : ℕ := 3

end NUMINAMATH_GPT_max_pieces_with_single_cut_min_cuts_to_intersect_all_pieces_l1639_163966


namespace NUMINAMATH_GPT_convert_8pi_over_5_to_degrees_l1639_163945

noncomputable def radian_to_degree (rad : ℝ) : ℝ := rad * (180 / Real.pi)

theorem convert_8pi_over_5_to_degrees : radian_to_degree (8 * Real.pi / 5) = 288 := by
  sorry

end NUMINAMATH_GPT_convert_8pi_over_5_to_degrees_l1639_163945


namespace NUMINAMATH_GPT_correct_propositions_l1639_163916

-- Definitions based on conditions
def diameter_perpendicular_bisects_chord (d : ℝ) (c : ℝ) : Prop :=
  ∃ (r : ℝ), d = 2 * r ∧ c = r

def triangle_vertices_determine_circle (a b c : ℝ) : Prop :=
  ∃ (O : ℝ), O = (a + b + c) / 3

def cyclic_quadrilateral_diagonals_supplementary (a b c d : ℕ) : Prop :=
  a + b + c + d = 360 -- incorrect statement

def tangent_perpendicular_to_radius (r t : ℝ) : Prop :=
  r * t = 1 -- assuming point of tangency

-- Theorem based on the problem conditions
theorem correct_propositions :
  diameter_perpendicular_bisects_chord 2 1 ∧
  triangle_vertices_determine_circle 1 2 3 ∧
  ¬ cyclic_quadrilateral_diagonals_supplementary 90 90 90 90 ∧
  tangent_perpendicular_to_radius 1 1 :=
by
  sorry

end NUMINAMATH_GPT_correct_propositions_l1639_163916


namespace NUMINAMATH_GPT_sixth_term_of_geometric_seq_l1639_163999

-- conditions
def is_geometric_sequence (seq : ℕ → ℕ) := 
  ∃ r : ℕ, ∀ n : ℕ, seq (n + 1) = seq n * r

def first_term (seq : ℕ → ℕ) := seq 1 = 3
def fifth_term (seq : ℕ → ℕ) := seq 5 = 243

-- question to be proved
theorem sixth_term_of_geometric_seq (seq : ℕ → ℕ) 
  (h_geom : is_geometric_sequence seq) 
  (h_first : first_term seq) 
  (h_fifth : fifth_term seq) : 
  seq 6 = 729 :=
sorry

end NUMINAMATH_GPT_sixth_term_of_geometric_seq_l1639_163999


namespace NUMINAMATH_GPT_remaining_plants_after_bugs_l1639_163904

theorem remaining_plants_after_bugs (initial_plants first_day_eaten second_day_fraction third_day_eaten remaining_plants : ℕ) : 
  initial_plants = 30 →
  first_day_eaten = 20 →
  second_day_fraction = 2 →
  third_day_eaten = 1 →
  remaining_plants = initial_plants - first_day_eaten - (initial_plants - first_day_eaten) / second_day_fraction - third_day_eaten →
  remaining_plants = 4 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_remaining_plants_after_bugs_l1639_163904


namespace NUMINAMATH_GPT_selling_price_is_correct_l1639_163988

-- Definitions based on conditions
def cost_price : ℝ := 280
def profit_percentage : ℝ := 0.3
def profit_amount : ℝ := cost_price * profit_percentage

-- Selling price definition
def selling_price : ℝ := cost_price + profit_amount

-- Theorem statement
theorem selling_price_is_correct : selling_price = 364 := by
  sorry

end NUMINAMATH_GPT_selling_price_is_correct_l1639_163988


namespace NUMINAMATH_GPT_chalkboard_area_l1639_163925

theorem chalkboard_area (width : ℝ) (h_w : width = 3) (h_l : 2 * width = length) : width * length = 18 := 
by 
  sorry

end NUMINAMATH_GPT_chalkboard_area_l1639_163925


namespace NUMINAMATH_GPT_number_of_integer_values_l1639_163911

theorem number_of_integer_values (x : ℕ) (h : ⌊ Real.sqrt x ⌋ = 8) : ∃ n : ℕ, n = 17 :=
by
  sorry

end NUMINAMATH_GPT_number_of_integer_values_l1639_163911


namespace NUMINAMATH_GPT_ab_le_1_e2_l1639_163903

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := Real.log x - a * x - b

theorem ab_le_1_e2 {a b : ℝ} (h : 0 < a) (hx : ∃ x : ℝ, 0 < x ∧ f x a b ≥ 0) : a * b ≤ 1 / Real.exp 2 :=
sorry

end NUMINAMATH_GPT_ab_le_1_e2_l1639_163903


namespace NUMINAMATH_GPT_basketball_minutes_played_l1639_163924

-- Definitions of the conditions in Lean
def football_minutes : ℕ := 60
def total_hours : ℕ := 2
def total_minutes : ℕ := total_hours * 60

-- The statement we need to prove (that basketball_minutes = 60)
theorem basketball_minutes_played : 
  (120 - football_minutes = 60) := by
  sorry

end NUMINAMATH_GPT_basketball_minutes_played_l1639_163924


namespace NUMINAMATH_GPT_f_at_seven_l1639_163926

variable {𝓡 : Type*} [CommRing 𝓡] [OrderedAddCommGroup 𝓡] [Module ℝ 𝓡]

-- Assuming f is a function from ℝ to ℝ with the given properties
variable (f : ℝ → ℝ)

-- Condition 1: f is an odd function.
def odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = - f x

-- Condition 2: f(x + 2) = -f(x) for all x.
def periodic_negation (f : ℝ → ℝ) : Prop := ∀ x, f (x + 2) = - f x 

-- Condition 3: f(x) = 2x^2 when x ∈ (0, 2)
def interval_definition (f : ℝ → ℝ) : Prop := ∀ x, 0 < x ∧ x < 2 → f x = 2 * x^2

theorem f_at_seven
  (h_odd : odd_function f)
  (h_periodic : periodic_negation f)
  (h_interval : interval_definition f) :
  f 7 = -2 :=
by
  sorry

end NUMINAMATH_GPT_f_at_seven_l1639_163926


namespace NUMINAMATH_GPT_find_x_y_l1639_163979

theorem find_x_y (A B C : ℝ) (x y : ℝ) (hA : A = 120) (hB : B = 100) (hC : C = 150)
  (hx : A = B + (x / 100) * B) (hy : A = C - (y / 100) * C) : x = 20 ∧ y = 20 :=
by
  sorry

end NUMINAMATH_GPT_find_x_y_l1639_163979


namespace NUMINAMATH_GPT_molecular_weight_correct_l1639_163978

namespace MolecularWeight

-- Define the atomic weights
def atomic_weight_N : ℝ := 14.01
def atomic_weight_H : ℝ := 1.01
def atomic_weight_Cl : ℝ := 35.45

-- Define the number of each atom in the compound
def n_N : ℝ := 1
def n_H : ℝ := 4
def n_Cl : ℝ := 1

-- Calculate the molecular weight of the compound
def molecular_weight : ℝ := (n_N * atomic_weight_N) + (n_H * atomic_weight_H) + (n_Cl * atomic_weight_Cl)

theorem molecular_weight_correct : molecular_weight = 53.50 := by
  -- Proof is omitted
  sorry

end MolecularWeight

end NUMINAMATH_GPT_molecular_weight_correct_l1639_163978


namespace NUMINAMATH_GPT_calculation_correct_l1639_163921

theorem calculation_correct : (3.456 - 1.234) * 0.5 = 1.111 :=
by
  sorry

end NUMINAMATH_GPT_calculation_correct_l1639_163921


namespace NUMINAMATH_GPT_max_value_l1639_163901

theorem max_value (x y : ℝ) (hx : x > 0) (hy : y > 0) : 
  ∃ m ≤ 4, ∀ (z w : ℝ), z > 0 → w > 0 → (x + y = z + w) → (z^3 + w^3 ≥ x^3 + y^3 → 
  (z + w)^3 / (z^3 + w^3) ≤ m) :=
sorry

end NUMINAMATH_GPT_max_value_l1639_163901


namespace NUMINAMATH_GPT_blue_markers_count_l1639_163959

-- Definitions based on the problem's conditions
def total_markers : ℕ := 3343
def red_markers : ℕ := 2315

-- Statement to prove
theorem blue_markers_count :
  total_markers - red_markers = 1028 := by
  sorry

end NUMINAMATH_GPT_blue_markers_count_l1639_163959


namespace NUMINAMATH_GPT_plane_speed_in_still_air_l1639_163975

theorem plane_speed_in_still_air (p w : ℝ) (h1 : (p + w) * 3 = 900) (h2 : (p - w) * 4 = 900) : p = 262.5 :=
by
  sorry

end NUMINAMATH_GPT_plane_speed_in_still_air_l1639_163975


namespace NUMINAMATH_GPT_problem1_solution_l1639_163933

theorem problem1_solution (p : ℕ) (hp : Nat.Prime p) (a b c : ℕ) (ha : 0 < a ∧ a ≤ p) (hb : 0 < b ∧ b ≤ p) (hc : 0 < c ∧ c ≤ p)
  (f : ℕ → ℕ) (hf : ∀ x : ℕ, 0 < x → p ∣ f x) :
  (∀ x, f x = a * x^2 + b * x + c) →
  (p = 2 → a + b + c = 4) ∧ (2 < p → p % 2 = 1 → a + b + c = 3 * p) :=
by
  sorry

end NUMINAMATH_GPT_problem1_solution_l1639_163933


namespace NUMINAMATH_GPT_dorothy_profit_l1639_163936

def cost_to_buy_ingredients : ℕ := 53
def number_of_doughnuts : ℕ := 25
def selling_price_per_doughnut : ℕ := 3

def revenue : ℕ := number_of_doughnuts * selling_price_per_doughnut
def profit : ℕ := revenue - cost_to_buy_ingredients

theorem dorothy_profit : profit = 22 :=
by
  -- calculation steps
  sorry

end NUMINAMATH_GPT_dorothy_profit_l1639_163936


namespace NUMINAMATH_GPT_arithmetic_geometric_sequence_l1639_163940

theorem arithmetic_geometric_sequence (a : ℕ → ℝ) (d : ℝ) (h₀ : d ≠ 0)
    (h₁ : a 3 = a 1 + 2 * d) (h₂ : a 9 = a 1 + 8 * d)
    (h₃ : (a 1 + 2 * d)^2 = a 1 * (a 1 + 8 * d)) :
    (a 1 + a 3 + a 9) / (a 2 + a 4 + a 10) = 13 / 16 := 
sorry

end NUMINAMATH_GPT_arithmetic_geometric_sequence_l1639_163940


namespace NUMINAMATH_GPT_range_of_a_l1639_163990

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x^2 + (a - 1) * x + a^2 > 0) ↔ (a < -1 ∨ a > (1 : ℝ) / 3) := 
sorry

end NUMINAMATH_GPT_range_of_a_l1639_163990


namespace NUMINAMATH_GPT_simplify_2M_minus_N_value_at_neg_1_M_gt_N_l1639_163954

-- Definitions of M and N
def M (x : ℝ) : ℝ := 4 * x^2 - 2 * x - 1
def N (x : ℝ) : ℝ := 3 * x^2 - 2 * x - 5

-- The simplified expression for 2M - N
theorem simplify_2M_minus_N {x : ℝ} : 2 * M x - N x = 5 * x^2 - 2 * x + 3 :=
by sorry

-- Value of the simplified expression when x = -1
theorem value_at_neg_1 : (5 * (-1)^2 - 2 * (-1) + 3) = 10 :=
by sorry

-- Relationship between M and N
theorem M_gt_N {x : ℝ} : M x > N x :=
by
  have h : M x - N x = x^2 + 4 := by sorry
  -- x^2 >= 0 for all x, so x^2 + 4 > 0 => M > N
  have nonneg : x^2 >= 0 := by sorry
  have add_pos : x^2 + 4 > 0 := by sorry
  sorry

end NUMINAMATH_GPT_simplify_2M_minus_N_value_at_neg_1_M_gt_N_l1639_163954


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l1639_163900

theorem intersection_of_A_and_B :
  let A := {x : ℝ | x > 0}
  let B := {x : ℝ | x^2 - 2*x - 3 < 0}
  (A ∩ B) = {x : ℝ | 0 < x ∧ x < 3} := by
  sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l1639_163900


namespace NUMINAMATH_GPT_total_remaining_staff_l1639_163938

-- Definitions of initial counts and doctors and nurses quitting.
def initial_doctors : ℕ := 11
def initial_nurses : ℕ := 18
def doctors_quitting : ℕ := 5
def nurses_quitting : ℕ := 2

-- Definition of remaining doctors and nurses.
def remaining_doctors : ℕ := initial_doctors - doctors_quitting
def remaining_nurses : ℕ := initial_nurses - nurses_quitting

-- Theorem stating the total number of doctors and nurses remaining.
theorem total_remaining_staff : remaining_doctors + remaining_nurses = 22 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_total_remaining_staff_l1639_163938


namespace NUMINAMATH_GPT_walnuts_count_l1639_163996

def nuts_problem (p a c w : ℕ) : Prop :=
  p + a + c + w = 150 ∧
  a = p / 2 ∧
  c = 4 * a ∧
  w = 3 * c

theorem walnuts_count (p a c w : ℕ) (h : nuts_problem p a c w) : w = 96 :=
by sorry

end NUMINAMATH_GPT_walnuts_count_l1639_163996


namespace NUMINAMATH_GPT_range_of_a_min_value_a_plus_4_over_a_sq_l1639_163972

noncomputable def f (x : ℝ) : ℝ :=
  |x - 10| + |x - 20|

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, f x < 10 * a + 10) ↔ 0 < a :=
sorry

theorem min_value_a_plus_4_over_a_sq (a : ℝ) (h : 0 < a) :
  ∃ y : ℝ, a + 4 / a ^ 2 = y ∧ y = 3 :=
sorry

end NUMINAMATH_GPT_range_of_a_min_value_a_plus_4_over_a_sq_l1639_163972


namespace NUMINAMATH_GPT_order_of_operations_example_l1639_163914

theorem order_of_operations_example :
  3^2 * 4 + 5 * (6 + 3) - 15 / 3 = 76 := by
  sorry

end NUMINAMATH_GPT_order_of_operations_example_l1639_163914


namespace NUMINAMATH_GPT_exp_mono_increasing_l1639_163923

theorem exp_mono_increasing (x y : ℝ) (h : x ≤ y) : (2:ℝ)^x ≤ (2:ℝ)^y :=
sorry

end NUMINAMATH_GPT_exp_mono_increasing_l1639_163923


namespace NUMINAMATH_GPT_polar_eq_is_circle_l1639_163907

-- Define the polar equation as a condition
def polar_eq (ρ : ℝ) := ρ = 5

-- Define the origin
def origin : ℝ × ℝ := (0, 0)

-- Prove that the curve represented by the polar equation is a circle
theorem polar_eq_is_circle (P : ℝ × ℝ) : (∃ ρ θ, P = (ρ * Real.cos θ, ρ * Real.sin θ) ∧ polar_eq ρ) ↔ dist P origin = 5 := 
by 
  sorry

end NUMINAMATH_GPT_polar_eq_is_circle_l1639_163907


namespace NUMINAMATH_GPT_kenny_played_basketball_last_week_l1639_163970

def time_practicing_trumpet : ℕ := 40
def time_running : ℕ := time_practicing_trumpet / 2
def time_playing_basketball : ℕ := time_running / 2
def answer : ℕ := 10

theorem kenny_played_basketball_last_week :
  time_playing_basketball = answer :=
by
  -- sorry to skip the proof
  sorry

end NUMINAMATH_GPT_kenny_played_basketball_last_week_l1639_163970


namespace NUMINAMATH_GPT_parallel_lines_a_value_l1639_163931

theorem parallel_lines_a_value (a : ℝ) : 
  (∀ x y : ℝ, 3 * x + 2 * a * y - 5 = 0 ↔ (3 * a - 1) * x - a * y - 2 = 0) →
  (a = 0 ∨ a = -1 / 6) :=
by
  sorry

end NUMINAMATH_GPT_parallel_lines_a_value_l1639_163931


namespace NUMINAMATH_GPT_initial_cupcakes_l1639_163913

variable (x : ℕ) -- Define x as the number of cupcakes Robin initially made

-- Define the conditions provided in the problem
def cupcakes_sold := 22
def cupcakes_made := 39
def final_cupcakes := 59

-- Formalize the problem statement: Prove that given the conditions, the initial cupcakes equals 42
theorem initial_cupcakes:
  x - cupcakes_sold + cupcakes_made = final_cupcakes → x = 42 := 
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_initial_cupcakes_l1639_163913


namespace NUMINAMATH_GPT_length_of_AC_l1639_163982

-- Definitions from the problem
variable (AB BC CD DA : ℝ)
variable (angle_ADC : ℝ)
variable (AC : ℝ)

-- Conditions from the problem
def conditions : Prop :=
  AB = 10 ∧ BC = 10 ∧ CD = 17 ∧ DA = 17 ∧ angle_ADC = 120

-- The mathematically equivalent proof statement
theorem length_of_AC (h : conditions AB BC CD DA angle_ADC) : AC = Real.sqrt 867 := sorry

end NUMINAMATH_GPT_length_of_AC_l1639_163982


namespace NUMINAMATH_GPT_no_rational_numbers_satisfy_l1639_163985

theorem no_rational_numbers_satisfy :
  ¬ ∃ (x y z : ℚ), x ≠ y ∧ y ≠ z ∧ z ≠ x ∧
    (1 / (x - y)^2 + 1 / (y - z)^2 + 1 / (z - x)^2 = 2014) :=
by
  sorry

end NUMINAMATH_GPT_no_rational_numbers_satisfy_l1639_163985


namespace NUMINAMATH_GPT_math_problem_l1639_163974

theorem math_problem (A B C : ℕ) (h_pos : A > 0 ∧ B > 0 ∧ C > 0) (h_gcd : Nat.gcd (Nat.gcd A B) C = 1) (h_eq : A * Real.log 5 / Real.log 200 + B * Real.log 2 / Real.log 200 = C) : A + B + C = 6 :=
sorry

end NUMINAMATH_GPT_math_problem_l1639_163974


namespace NUMINAMATH_GPT_anna_bought_five_chocolate_bars_l1639_163908

noncomputable section

def initial_amount : ℝ := 10
def price_chewing_gum : ℝ := 1
def price_candy_cane : ℝ := 0.5
def remaining_amount : ℝ := 1

def chewing_gum_cost : ℝ := 3 * price_chewing_gum
def candy_cane_cost : ℝ := 2 * price_candy_cane

def total_spent : ℝ := initial_amount - remaining_amount
def known_items_cost : ℝ := chewing_gum_cost + candy_cane_cost
def chocolate_bars_cost : ℝ := total_spent - known_items_cost
def price_chocolate_bar : ℝ := 1

def chocolate_bars_bought : ℝ := chocolate_bars_cost / price_chocolate_bar

theorem anna_bought_five_chocolate_bars : chocolate_bars_bought = 5 := 
by
  sorry

end NUMINAMATH_GPT_anna_bought_five_chocolate_bars_l1639_163908
