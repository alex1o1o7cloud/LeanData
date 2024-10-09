import Mathlib

namespace frac_m_over_q_l2209_220954

variable (m n p q : ℚ)

theorem frac_m_over_q (h1 : m / n = 10) (h2 : p / n = 2) (h3 : p / q = 1 / 5) : m / q = 1 :=
by
  sorry

end frac_m_over_q_l2209_220954


namespace problem1_problem2_l2209_220951

noncomputable def f (a x : ℝ) : ℝ :=
  if x < a then 2 * a - (x + 4 / x)
  else x - 4 / x

theorem problem1 (h : ∀ x : ℝ, f 1 x = 3 → x = 4) : ∃ x : ℝ, f 1 x = 3 ∧ x = 4 :=
sorry

theorem problem2 (h : ∀ x1 x2 x3 : ℝ, 
  (x1 < x2 ∧ x2 < x3 ∧ x2 - x1 = x3 - x2) →
  f a x1 = 3 ∧ f a x2 = 3 ∧ f a x3 = 3 ∧ a ≤ -1 → 
  a = -11 / 6) : ∃ a : ℝ, a ≤ -1 ∧ (∃ x1 x2 x3 : ℝ, 
  (x1 < x2 ∧ x2 < x3 ∧ x2 - x1 = x3 - x2) ∧ 
  f a x1 = 3 ∧ f a x2 = 3 ∧ f a x3 = 3 ∧ a = -11 / 6) :=
sorry

end problem1_problem2_l2209_220951


namespace number_of_continents_collected_l2209_220912

-- Definitions of the given conditions
def books_per_continent : ℕ := 122
def total_books : ℕ := 488

-- The mathematical statement to be proved
theorem number_of_continents_collected :
  total_books / books_per_continent = 4 :=
by
  -- Placeholder for the proof
  sorry

end number_of_continents_collected_l2209_220912


namespace determine_value_of_e_l2209_220902

theorem determine_value_of_e {a b c d e : ℝ} (h1 : a < b) (h2 : b < c) (h3 : c < d) (h4 : d < e) 
    (h5 : a + b = 32) (h6 : a + c = 36) (h7 : b + c = 37 ∨ a + d = 37) 
    (h8 : c + e = 48) (h9 : d + e = 51) : e = 27.5 :=
sorry

end determine_value_of_e_l2209_220902


namespace lowest_score_is_C_l2209_220917

variable (Score : Type) [LinearOrder Score]
variable (A B C : Score)

-- Translate conditions into Lean
variable (h1 : B ≠ max A (max B C) → A = min A (min B C))
variable (h2 : C ≠ min A (min B C) → A = max A (max B C))

-- Define the proof goal
theorem lowest_score_is_C : min A (min B C) =C :=
by
  sorry

end lowest_score_is_C_l2209_220917


namespace laundry_lcm_l2209_220922

theorem laundry_lcm :
  Nat.lcm (Nat.lcm 6 9) (Nat.lcm 12 15) = 180 :=
by
  sorry

end laundry_lcm_l2209_220922


namespace cyclic_ABCD_l2209_220939

variable {Point : Type}
variable {Angle LineCircle : Type → Type}
variable {cyclicQuadrilateral : List (Point) → Prop}
variable {convexQuadrilateral : List (Point) → Prop}
variable {lineSegment : Point → Point → LineCircle Point}
variable {onSegment : Point → LineCircle Point → Prop}
variable {angle : Point → Point → Point → Angle Point}

theorem cyclic_ABCD (A B C D P Q E : Point)
  (h1 : convexQuadrilateral [A, B, C, D])
  (h2 : cyclicQuadrilateral [P, Q, D, A])
  (h3 : cyclicQuadrilateral [Q, P, B, C])
  (h4 : onSegment E (lineSegment P Q))
  (h5 : angle P A E = angle Q D E)
  (h6 : angle P B E = angle Q C E) :
  cyclicQuadrilateral [A, B, C, D] :=
  sorry

end cyclic_ABCD_l2209_220939


namespace order_of_abc_l2209_220959

noncomputable def a : ℚ := 1 / 2
noncomputable def b : ℝ := Real.sqrt 7 - Real.sqrt 5
noncomputable def c : ℝ := Real.sqrt 6 - 2

theorem order_of_abc : a > c ∧ c > b := by
  sorry

end order_of_abc_l2209_220959


namespace greatest_four_digit_number_l2209_220975

theorem greatest_four_digit_number (x : ℕ) :
  x ≡ 1 [MOD 7] ∧ x ≡ 5 [MOD 8] ∧ 1000 ≤ x ∧ x < 10000 → x = 9997 :=
by
  sorry

end greatest_four_digit_number_l2209_220975


namespace orthogonal_vectors_l2209_220990

open Real

variables (a b : ℝ) (a_nonzero : a ≠ 0) (b_nonzero : b ≠ 0) (h : (a + b)^2 = (a - b)^2)

theorem orthogonal_vectors (a b : ℝ) (a_nonzero : a ≠ 0) (b_nonzero : b ≠ 0) 
  (h : (a + b)^2 = (a - b)^2) : a * b = 0 :=
by 
  sorry

end orthogonal_vectors_l2209_220990


namespace mr_william_farm_tax_l2209_220992

noncomputable def total_tax_collected : ℝ := 3840
noncomputable def mr_william_percentage : ℝ := 16.666666666666668 / 100  -- Convert percentage to decimal

theorem mr_william_farm_tax : (total_tax_collected * mr_william_percentage) = 640 := by
  sorry

end mr_william_farm_tax_l2209_220992


namespace integral_solution_unique_l2209_220973

theorem integral_solution_unique (a b c : ℤ) : a^2 + b^2 + c^2 = a^2 * b^2 → a = 0 ∧ b = 0 ∧ c = 0 :=
by
  sorry

end integral_solution_unique_l2209_220973


namespace algae_cell_count_at_day_nine_l2209_220966

noncomputable def initial_cells : ℕ := 5
noncomputable def division_frequency_days : ℕ := 3
noncomputable def total_days : ℕ := 9

def number_of_cycles (total_days division_frequency_days : ℕ) : ℕ :=
  total_days / division_frequency_days

noncomputable def common_ratio : ℕ := 2

noncomputable def number_of_cells_after_n_days (initial_cells common_ratio number_of_cycles : ℕ) : ℕ :=
  initial_cells * common_ratio ^ (number_of_cycles - 1)

theorem algae_cell_count_at_day_nine : number_of_cells_after_n_days initial_cells common_ratio (number_of_cycles total_days division_frequency_days) = 20 :=
by
  sorry

end algae_cell_count_at_day_nine_l2209_220966


namespace prove_d_minus_r_eq_1_l2209_220955

theorem prove_d_minus_r_eq_1 
  (d r : ℕ) 
  (h_d1 : d > 1)
  (h1 : 1122 % d = r)
  (h2 : 1540 % d = r)
  (h3 : 2455 % d = r) :
  d - r = 1 :=
by sorry

end prove_d_minus_r_eq_1_l2209_220955


namespace downstream_speed_l2209_220932

variable (V_u V_s V_d : ℝ)

theorem downstream_speed (h1 : V_u = 22) (h2 : V_s = 32) (h3 : V_s = (V_u + V_d) / 2) : V_d = 42 :=
sorry

end downstream_speed_l2209_220932


namespace no_lunch_students_l2209_220905

variable (total_students : ℕ) (cafeteria_eaters : ℕ) (lunch_bringers : ℕ)

theorem no_lunch_students : 
  total_students = 60 →
  cafeteria_eaters = 10 →
  lunch_bringers = 3 * cafeteria_eaters →
  total_students - (cafeteria_eaters + lunch_bringers) = 20 :=
by
  sorry

end no_lunch_students_l2209_220905


namespace glen_pop_l2209_220988

/-- In the village of Glen, the total population can be formulated as 21h + 6c
given the relationships between people, horses, sheep, cows, and ducks.
We need to prove that 96 cannot be expressed in the form 21h + 6c for
non-negative integers h and c. -/
theorem glen_pop (h c : ℕ) : 21 * h + 6 * c ≠ 96 :=
by
sorry

end glen_pop_l2209_220988


namespace general_formula_arithmetic_sum_of_geometric_terms_l2209_220916

noncomputable def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  a 2 = 2 ∧ a 5 = 8

noncomputable def geometric_sequence (b : ℕ → ℝ) (a : ℕ → ℤ) : Prop :=
  b 1 = 1 ∧ b 2 + b 3 = a 4

noncomputable def sum_of_terms (T : ℕ → ℝ) (b : ℕ → ℝ) : Prop :=
  ∀ n, T n = (2:ℝ)^n - 1

theorem general_formula_arithmetic (a : ℕ → ℤ) (h : arithmetic_sequence a) :
  ∀ n, a n = 2 * n - 2 :=
sorry

theorem sum_of_geometric_terms (a : ℕ → ℤ) (b : ℕ → ℝ) (T : ℕ → ℝ)
  (h : arithmetic_sequence a) (h2 : geometric_sequence b a) :
  sum_of_terms T b :=
sorry

end general_formula_arithmetic_sum_of_geometric_terms_l2209_220916


namespace incorrect_inequality_l2209_220979

theorem incorrect_inequality (a b : ℝ) (h : a < b) : ¬ (-4 * a < -4 * b) :=
by sorry

end incorrect_inequality_l2209_220979


namespace find_valid_pairs_l2209_220981

theorem find_valid_pairs :
  ∃ (a b c : ℕ), 
    (a = 33 ∧ b = 22 ∧ c = 1111) ∨
    (a = 66 ∧ b = 88 ∧ c = 4444) ∨
    (a = 88 ∧ b = 33 ∧ c = 7777) ∧
    (11 ≤ a ∧ a ≤ 99) ∧ (11 ≤ b ∧ b ≤ 99) ∧ (1111 ≤ c ∧ c ≤ 9999) ∧
    (a % 11 = 0) ∧ (b % 11 = 0) ∧ (c % 1111 = 0) ∧
    (a * a + b = c) := sorry

end find_valid_pairs_l2209_220981


namespace radius_of_sphere_l2209_220993

theorem radius_of_sphere (R : ℝ) (shots_count : ℕ) (shot_radius : ℝ) :
  shots_count = 125 →
  shot_radius = 1 →
  (shots_count : ℝ) * (4 / 3 * Real.pi * shot_radius^3) = 4 / 3 * Real.pi * R^3 →
  R = 5 :=
by
  intros h1 h2 h3
  sorry

end radius_of_sphere_l2209_220993


namespace gcd_seven_factorial_ten_fact_div_5_fact_l2209_220980

def factorial (n : ℕ) : ℕ := Nat.factorial n

-- Define 7!
def seven_factorial := factorial 7

-- Define 10! / 5!
def ten_fact_div_5_fact := factorial 10 / factorial 5

-- Prove that the GCD of 7! and (10! / 5!) is 2520
theorem gcd_seven_factorial_ten_fact_div_5_fact :
  Nat.gcd seven_factorial ten_fact_div_5_fact = 2520 := by
sorry

end gcd_seven_factorial_ten_fact_div_5_fact_l2209_220980


namespace johns_weekly_allowance_l2209_220994

theorem johns_weekly_allowance (A : ℝ) 
  (arcade_spent : A * (3/5) = 3 * (A/5)) 
  (remainder_after_arcade : (2/5) * A = A - 3 * (A/5))
  (toy_store_spent : (1/3) * (2/5) * A = 2 * (A/15)) 
  (remainder_after_toy_store : (2/5) * A - (2/15) * A = 4 * (A/15))
  (last_spent : (4/15) * A = 0.4) :
  A = 1.5 :=
sorry

end johns_weekly_allowance_l2209_220994


namespace range_of_x_l2209_220950

theorem range_of_x (x : ℤ) : x^2 < 3 * x → x = 1 ∨ x = 2 :=
by
  sorry

end range_of_x_l2209_220950


namespace minimum_quadratic_expression_l2209_220985

theorem minimum_quadratic_expression : ∃ (x : ℝ), (∀ y : ℝ, y^2 - 6*y + 5 ≥ -4) ∧ (x^2 - 6*x + 5 = -4) :=
by
  sorry

end minimum_quadratic_expression_l2209_220985


namespace gcd_2023_1991_l2209_220974

theorem gcd_2023_1991 : Nat.gcd 2023 1991 = 1 :=
by
  sorry

end gcd_2023_1991_l2209_220974


namespace range_of_k_l2209_220970

noncomputable def f (x : ℝ) : ℝ := (x - 3) * Real.exp x

theorem range_of_k :
  (∀ x : ℝ, 2 < x → f x > k) →
  k ≤ -Real.exp 2 :=
by
  sorry

end range_of_k_l2209_220970


namespace James_weight_after_gain_l2209_220982

theorem James_weight_after_gain 
    (initial_weight : ℕ)
    (muscle_gain_perc : ℕ)
    (fat_gain_fraction : ℚ)
    (weight_after_gain : ℕ) :
    initial_weight = 120 →
    muscle_gain_perc = 20 →
    fat_gain_fraction = 1/4 →
    weight_after_gain = 150 :=
by
  intros
  sorry

end James_weight_after_gain_l2209_220982


namespace bus_stop_time_l2209_220996

theorem bus_stop_time (v_no_stop v_with_stop : ℝ) (t_per_hour_minutes : ℝ) (h1 : v_no_stop = 48) (h2 : v_with_stop = 24) : t_per_hour_minutes = 30 := 
sorry

end bus_stop_time_l2209_220996


namespace value_added_to_half_is_five_l2209_220956

theorem value_added_to_half_is_five (n V : ℕ) (h₁ : n = 16) (h₂ : (1 / 2 : ℝ) * n + V = 13) : V = 5 := 
by 
  sorry

end value_added_to_half_is_five_l2209_220956


namespace joan_seashells_correct_l2209_220946

/-- Joan originally found 70 seashells -/
def joan_original_seashells : ℕ := 70

/-- Sam gave Joan 27 seashells -/
def seashells_given_by_sam : ℕ := 27

/-- The total number of seashells Joan has now -/
def joan_total_seashells : ℕ := joan_original_seashells + seashells_given_by_sam

theorem joan_seashells_correct : joan_total_seashells = 97 :=
by
  unfold joan_total_seashells
  unfold joan_original_seashells seashells_given_by_sam
  sorry

end joan_seashells_correct_l2209_220946


namespace carlson_max_candies_l2209_220987

theorem carlson_max_candies : 
  (∀ (erase_two_and_sum : ℕ → ℕ → ℕ) 
    (eat_candies : ℕ → ℕ → ℕ), 
  ∃ (maximum_candies : ℕ), 
  (erase_two_and_sum 1 1 = 2) ∧
  (eat_candies 1 1 = 1) ∧ 
  (maximum_candies = 496)) :=
by
  sorry

end carlson_max_candies_l2209_220987


namespace intersection_domains_l2209_220919

def domain_f := {x : ℝ | x < 1}
def domain_g := {x : ℝ | x ≠ 0}

theorem intersection_domains :
  {x : ℝ | x < 1} ∩ {x : ℝ | x ≠ 0} = {x : ℝ | x < 1 ∧ x ≠ 0} :=
by 
  sorry

end intersection_domains_l2209_220919


namespace num_packages_l2209_220908

theorem num_packages (total_shirts : ℕ) (shirts_per_package : ℕ) (h1 : total_shirts = 51) (h2 : shirts_per_package = 3) : total_shirts / shirts_per_package = 17 := by
  sorry

end num_packages_l2209_220908


namespace ratio_of_cost_to_marked_price_l2209_220964

variable (p : ℝ)

theorem ratio_of_cost_to_marked_price :
  let selling_price := (3/4) * p
  let cost_price := (5/8) * selling_price
  cost_price / p = 15 / 32 :=
by
  let selling_price := (3 / 4) * p
  let cost_price := (5 / 8) * selling_price
  sorry

end ratio_of_cost_to_marked_price_l2209_220964


namespace line_equation_l2209_220907

theorem line_equation (l : ℝ → ℝ → Prop) (a b : ℝ) 
  (h1 : ∀ x y, l x y ↔ y = - (b / a) * x + b) 
  (h2 : l 2 1) 
  (h3 : a + b = 0) : 
  l x y ↔ y = x - 1 ∨ y = x / 2 := 
by
  sorry

end line_equation_l2209_220907


namespace min_value_expression_l2209_220904

theorem min_value_expression (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_abc : a * b * c = 1/2) :
  a^2 + 4 * a * b + 12 * b^2 + 8 * b * c + 3 * c^2 ≥ 18 :=
sorry

end min_value_expression_l2209_220904


namespace machine_production_time_difference_undetermined_l2209_220931

theorem machine_production_time_difference_undetermined :
  ∀ (machineP_machineQ_440_hours_diff : ℝ)
    (machineQ_production_rate : ℝ)
    (machineA_production_rate : ℝ),
    machineA_production_rate = 4.000000000000005 →
    machineQ_production_rate = machineA_production_rate * 1.1 →
    machineP_machineQ_440_hours_diff > 0 →
    machineQ_production_rate * machineP_machineQ_440_hours_diff = 440 →
    ∃ machineP_production_rate, 
    ¬(∃ hours_diff : ℝ, hours_diff = 440 / machineP_production_rate - 440 / machineQ_production_rate) := sorry

end machine_production_time_difference_undetermined_l2209_220931


namespace minimize_t_l2209_220913

variable (Q : ℝ) (Q_1 Q_2 Q_3 Q_4 Q_5 Q_6 Q_7 Q_8 Q_9 : ℝ)

-- Definition of the sum of undirected lengths
def t (Q : ℝ) := 
  abs (Q - Q_1) + abs (Q - Q_2) + abs (Q - Q_3) + 
  abs (Q - Q_4) + abs (Q - Q_5) + abs (Q - Q_6) + 
  abs (Q - Q_7) + abs (Q - Q_8) + abs (Q - Q_9)

-- Statement that t is minimized when Q = Q_5
theorem minimize_t : ∀ Q : ℝ, t Q ≥ t Q_5 := 
sorry

end minimize_t_l2209_220913


namespace best_fitting_model_l2209_220910

theorem best_fitting_model (R2_1 R2_2 R2_3 R2_4 : ℝ)
  (h1 : R2_1 = 0.98)
  (h2 : R2_2 = 0.80)
  (h3 : R2_3 = 0.50)
  (h4 : R2_4 = 0.25) :
  R2_1 = 0.98 ∧ R2_1 > R2_2 ∧ R2_1 > R2_3 ∧ R2_1 > R2_4 :=
by { sorry }

end best_fitting_model_l2209_220910


namespace exponent_property_l2209_220924

theorem exponent_property (a : ℝ) : a^7 = a^3 * a^4 :=
by
  -- The proof statement follows from the properties of exponents:
  -- a^m * a^n = a^(m + n)
  -- Therefore, a^3 * a^4 = a^(3 + 4) = a^7.
  sorry

end exponent_property_l2209_220924


namespace probability_not_exceeding_40_l2209_220934

variable (P : ℝ → Prop)

def less_than_30_grams : Prop := P 0.3
def between_30_and_40_grams : Prop := P 0.5

theorem probability_not_exceeding_40 (h1 : less_than_30_grams P) (h2 : between_30_and_40_grams P) : P 0.8 :=
by
  sorry

end probability_not_exceeding_40_l2209_220934


namespace inequality_has_solutions_iff_a_ge_4_l2209_220962

theorem inequality_has_solutions_iff_a_ge_4 (a x : ℝ) :
  (∃ x : ℝ, |x + 1| + |x - 3| ≤ a) ↔ a ≥ 4 :=
sorry

end inequality_has_solutions_iff_a_ge_4_l2209_220962


namespace calculation_l2209_220972

theorem calculation : 2005^2 - 2003 * 2007 = 4 :=
by
  have h1 : 2003 = 2005 - 2 := by rfl
  have h2 : 2007 = 2005 + 2 := by rfl
  sorry

end calculation_l2209_220972


namespace AM_GM_inequality_example_l2209_220925

theorem AM_GM_inequality_example (a b c d : ℝ)
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0)
  (h_prod : a * b * c * d = 1) :
  a^3 + b^3 + c^3 + d^3 ≥ max (a + b + c + d) (1 / a + 1 / b + 1 / c + 1 / d) :=
by
  sorry

end AM_GM_inequality_example_l2209_220925


namespace cousin_cards_probability_l2209_220914

variable {Isabella_cards : ℕ}
variable {Evan_cards : ℕ}
variable {total_cards : ℕ}

theorem cousin_cards_probability 
  (h1 : Isabella_cards = 8)
  (h2 : Evan_cards = 2)
  (h3 : total_cards = 10) :
  (8 / 10 * 2 / 9) + (2 / 10 * 8 / 9) = 16 / 45 :=
by
  sorry

end cousin_cards_probability_l2209_220914


namespace system_has_two_distinct_solutions_for_valid_a_l2209_220952

noncomputable def log_eq (x y a : ℝ) : Prop := 
  Real.log (a * x + 4 * a) / Real.log (abs (x + 3)) = 
  2 * Real.log (x + y) / Real.log (abs (x + 3))

noncomputable def original_system (x y a : ℝ) : Prop :=
  log_eq x y a ∧ (x + 1 + Real.sqrt (x^2 + 2 * x + y - 4) = 0)

noncomputable def valid_range (a : ℝ) : Prop := 
  (4 < a ∧ a < 4.5) ∨ (4.5 < a ∧ a ≤ 16 / 3)

theorem system_has_two_distinct_solutions_for_valid_a (a : ℝ) :
  valid_range a → 
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ original_system x₁ 5 a ∧ original_system x₂ 5 a ∧ (-5 < x₁ ∧ x₁ ≤ -1) ∧ (-5 < x₂ ∧ x₂ ≤ -1) := 
sorry

end system_has_two_distinct_solutions_for_valid_a_l2209_220952


namespace shopkeeper_profit_percentage_l2209_220983

theorem shopkeeper_profit_percentage
  (C : ℝ) -- The cost price of one article
  (cost_price_50 : ℝ := 50 * C) -- The cost price of 50 articles
  (cost_price_70 : ℝ := 70 * C) -- The cost price of 70 articles
  (selling_price_50 : ℝ := 70 * C) -- Selling price of 50 articles is the cost price of 70 articles
  :
  ∃ (P : ℝ), P = 40 :=
by
  sorry

end shopkeeper_profit_percentage_l2209_220983


namespace total_students_in_class_l2209_220930

def period_length : ℕ := 40
def periods_per_student : ℕ := 4
def time_per_student : ℕ := 5

theorem total_students_in_class :
  ((period_length / time_per_student) * periods_per_student) = 32 :=
by
  sorry

end total_students_in_class_l2209_220930


namespace existence_of_f_and_g_l2209_220940

noncomputable def Set_n (n : ℕ) : Set ℕ := { x | x ≥ 1 ∧ x ≤ n }

theorem existence_of_f_and_g (n : ℕ) (f g : ℕ → ℕ) :
  (∀ x ∈ Set_n n, (f (g x) = x ∨ g (f x) = x) ∧ ¬(f (g x) = x ∧ g (f x) = x)) ↔ Even n := sorry

end existence_of_f_and_g_l2209_220940


namespace max_value_f_l2209_220920

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * (4 : ℝ) * x + 2

theorem max_value_f :
  ∃ x : ℝ, -f x = -18 ∧ (∀ y : ℝ, f y ≤ f x) :=
by
  sorry

end max_value_f_l2209_220920


namespace age_difference_l2209_220949

theorem age_difference (C D m : ℕ) 
  (h1 : C = D + m)
  (h2 : C - 1 = 3 * (D - 1)) 
  (h3 : C * D = 72) : 
  m = 9 :=
sorry

end age_difference_l2209_220949


namespace identify_clothes_l2209_220969

open Function

-- Definitions
def Alina : Prop := ∃ (tshirt short : String), tshirt = "Red" ∧ short = "Red"
def Bogdan : Prop := ∃ (tshirt short : String), tshirt = "Red" ∧ short = "Blue"
def Vika : Prop := ∃ (tshirt short : String), tshirt = "Blue" ∧ short = "Blue"
def Grisha : Prop := ∃ (tshirt short : String), tshirt = "Red" ∧ short = "Blue"

-- Problem statement
theorem identify_clothes :
  Alina ∧ Bogdan ∧ Vika ∧ Grisha :=
by
  sorry -- Proof will be developed here

end identify_clothes_l2209_220969


namespace vertical_angles_congruent_l2209_220995

namespace VerticalAngles

-- Define what it means for two angles to be vertical
def Vertical (α β : Real) : Prop :=
  -- Definition of vertical angles goes here
  sorry

-- Hypothesis: If two angles are vertical angles
theorem vertical_angles_congruent (α β : Real) (h : Vertical α β) : α = β :=
sorry

end VerticalAngles

end vertical_angles_congruent_l2209_220995


namespace log_travel_time_24_l2209_220963

noncomputable def time_for_log_to_travel (D u v : ℝ) (h1 : D / (u + v) = 4) (h2 : D / (u - v) = 6) : ℝ :=
  D / v

theorem log_travel_time_24 (D u v : ℝ) (h1 : D / (u + v) = 4) (h2 : D / (u - v) = 6) :
  time_for_log_to_travel D u v h1 h2 = 24 :=
sorry

end log_travel_time_24_l2209_220963


namespace find_wrongly_written_height_l2209_220911

def wrongly_written_height
  (n : ℕ)
  (avg_height_incorrect : ℝ)
  (actual_height : ℝ)
  (avg_height_correct : ℝ) : ℝ :=
  let total_height_incorrect := n * avg_height_incorrect
  let total_height_correct := n * avg_height_correct
  let height_difference := total_height_incorrect - total_height_correct
  actual_height + height_difference

theorem find_wrongly_written_height :
  wrongly_written_height 35 182 106 180 = 176 :=
by
  sorry

end find_wrongly_written_height_l2209_220911


namespace focus_of_parabola_l2209_220967

theorem focus_of_parabola :
  (∃ (x y : ℝ), y = 4 * x ^ 2 - 8 * x - 12 ∧ x = 1 ∧ y = -15.9375) :=
by
  sorry

end focus_of_parabola_l2209_220967


namespace central_angle_of_sector_l2209_220921

variable (r θ : ℝ)
variable (r_pos : 0 < r) (θ_pos : 0 < θ)

def perimeter_eq : Prop := 2 * r + r * θ = 5
def area_eq : Prop := (1 / 2) * r^2 * θ = 1

theorem central_angle_of_sector :
  perimeter_eq r θ ∧ area_eq r θ → θ = 1 / 2 :=
sorry

end central_angle_of_sector_l2209_220921


namespace min_value_of_3x_2y_l2209_220999

noncomputable def min_value (x y: ℝ) : ℝ := 3 * x + 2 * y

theorem min_value_of_3x_2y (x y : ℝ) (h₁ : 0 < x) (h₂ : 0 < y) (h₃ : x + y - x * y = 0) :
  min_value x y = 5 + 2 * Real.sqrt 6 :=
sorry

end min_value_of_3x_2y_l2209_220999


namespace work_days_for_c_l2209_220968

theorem work_days_for_c (A B C : ℝ)
  (h1 : A + B = 1 / 15)
  (h2 : A + B + C = 1 / 11) :
  1 / C = 41.25 :=
by
  sorry

end work_days_for_c_l2209_220968


namespace companyA_sold_bottles_l2209_220984

-- Let CompanyA and CompanyB be the prices per bottle for the respective companies
def CompanyA_price : ℝ := 4
def CompanyB_price : ℝ := 3.5

-- Company B sold 350 bottles
def CompanyB_bottles : ℕ := 350

-- Total revenue of Company B
def CompanyB_revenue : ℝ := CompanyB_price * CompanyB_bottles

-- Additional condition that the revenue difference is $25
def revenue_difference : ℝ := 25

-- Define the total revenue equations for both scenarios
def revenue_scenario1 (x : ℕ) : Prop :=
  CompanyA_price * x = CompanyB_revenue + revenue_difference

def revenue_scenario2 (x : ℕ) : Prop :=
  CompanyA_price * x + revenue_difference = CompanyB_revenue

-- The problem translates to finding x such that either of these conditions hold
theorem companyA_sold_bottles : ∃ x : ℕ, revenue_scenario2 x ∧ x = 300 :=
by
  sorry

end companyA_sold_bottles_l2209_220984


namespace integer_part_mod_8_l2209_220945

theorem integer_part_mod_8 (n : ℕ) (h : n ≥ 2009) :
  ∃ x : ℝ, x = (3 + Real.sqrt 8)^(2 * n) ∧ Int.floor (x) % 8 = 1 := 
sorry

end integer_part_mod_8_l2209_220945


namespace dark_chocolate_bars_sold_l2209_220960

theorem dark_chocolate_bars_sold (W D : ℕ) (h₁ : 4 * D = 3 * W) (h₂ : W = 20) : D = 15 :=
by
  sorry

end dark_chocolate_bars_sold_l2209_220960


namespace cube_surface_area_l2209_220901

theorem cube_surface_area (a : ℝ) (h : a = 1) :
    6 * a^2 = 6 := by
  sorry

end cube_surface_area_l2209_220901


namespace max_1x2_rectangles_in_3x3_grid_l2209_220927

theorem max_1x2_rectangles_in_3x3_grid : 
  ∀ unit_squares rectangles_1x2 : ℕ, unit_squares + rectangles_1x2 = 9 → 
  (∃ max_rectangles : ℕ, max_rectangles = rectangles_1x2 ∧ max_rectangles = 5) :=
by
  sorry

end max_1x2_rectangles_in_3x3_grid_l2209_220927


namespace commutative_op_l2209_220906

variable {S : Type} (op : S → S → S)

-- Conditions
axiom cond1 : ∀ (a b : S), op a (op a b) = b
axiom cond2 : ∀ (a b : S), op (op a b) b = a

-- Proof problem statement
theorem commutative_op : ∀ (a b : S), op a b = op b a :=
by
  intros a b
  sorry

end commutative_op_l2209_220906


namespace interior_triangle_area_l2209_220936

theorem interior_triangle_area (s1 s2 s3 : ℝ) (hs1 : s1 = 15) (hs2 : s2 = 6) (hs3 : s3 = 15) 
  (a1 a2 a3 : ℝ) (ha1 : a1 = 225) (ha2 : a2 = 36) (ha3 : a3 = 225) 
  (h1 : s1 * s1 = a1) (h2 : s2 * s2 = a2) (h3 : s3 * s3 = a3) :
  (1/2) * s1 * s2 = 45 :=
by
  sorry

end interior_triangle_area_l2209_220936


namespace exists_n_for_A_of_non_perfect_square_l2209_220918

theorem exists_n_for_A_of_non_perfect_square (A : ℕ) (h : ∀ k : ℕ, k^2 ≠ A) :
  ∃ n : ℕ, A = ⌊ n + Real.sqrt n + 1/2 ⌋ :=
sorry

end exists_n_for_A_of_non_perfect_square_l2209_220918


namespace decimal_expansion_of_13_over_625_l2209_220998

theorem decimal_expansion_of_13_over_625 : (13 : ℚ) / 625 = 0.0208 :=
by sorry

end decimal_expansion_of_13_over_625_l2209_220998


namespace sum_edge_lengths_rectangular_solid_vol_surface_area_geom_prog_l2209_220943

theorem sum_edge_lengths_rectangular_solid_vol_surface_area_geom_prog
  (a r : ℝ)
  (volume_cond : a^3 * r^3 = 288)
  (surface_area_cond : 2 * (a^2 * r^4 + a^2 * r^2 + a^2 * r) = 288)
  (geom_prog : True) :
  4 * (a * r^2 + a * r + a) = 92 := 
sorry

end sum_edge_lengths_rectangular_solid_vol_surface_area_geom_prog_l2209_220943


namespace marie_tasks_finish_time_l2209_220997

noncomputable def total_time (times : List ℕ) : ℕ :=
  times.foldr (· + ·) 0

theorem marie_tasks_finish_time :
  let task_times := [30, 40, 50, 60]
  let start_time := 8 * 60 -- Start time in minutes (8:00 AM)
  let end_time := start_time + total_time task_times
  end_time = 11 * 60 := -- 11:00 AM in minutes
by
  -- Add a placeholder for the proof
  sorry

end marie_tasks_finish_time_l2209_220997


namespace fraction_problem_l2209_220941

theorem fraction_problem (m n p q : ℚ) 
  (h1 : m / n = 20) 
  (h2 : p / n = 5) 
  (h3 : p / q = 1 / 15) : 
  m / q = 4 / 15 :=
sorry

end fraction_problem_l2209_220941


namespace positive_m_for_one_solution_l2209_220991

theorem positive_m_for_one_solution :
  ∀ (m : ℝ), (∃ x : ℝ, 9 * x^2 + m * x + 36 = 0) ∧ 
  (∀ x y : ℝ, 9 * x^2 + m * x + 36 = 0 → 9 * y^2 + m * y + 36 = 0 → x = y) → m = 36 := 
by {
  sorry
}

end positive_m_for_one_solution_l2209_220991


namespace initial_pencils_l2209_220935

theorem initial_pencils (P : ℕ) (h1 : 84 = P - (P - 15) / 4 + 16 - 12 + 23) : P = 71 :=
by
  sorry

end initial_pencils_l2209_220935


namespace b_integer_iff_a_special_form_l2209_220957

theorem b_integer_iff_a_special_form (a : ℝ) (b : ℝ) 
  (h1 : a > 0) 
  (h2 : b = (a + Real.sqrt (a ^ 2 + 1)) ^ (1 / 3) + (a - Real.sqrt (a ^ 2 + 1)) ^ (1 / 3)) : 
  (∃ (n : ℕ), a = 1 / 2 * (n * (n^2 + 3))) ↔ (∃ (n : ℕ), b = n) :=
sorry

end b_integer_iff_a_special_form_l2209_220957


namespace grocer_initial_stock_l2209_220938

noncomputable def initial_coffee_stock (x : ℝ) : Prop :=
  let initial_decaf := 0.20 * x
  let additional_coffee := 100
  let additional_decaf := 0.50 * additional_coffee
  let total_coffee := x + additional_coffee
  let total_decaf := initial_decaf + additional_decaf
  0.26 * total_coffee = total_decaf

theorem grocer_initial_stock :
  ∃ x : ℝ, initial_coffee_stock x ∧ x = 400 :=
by
  sorry

end grocer_initial_stock_l2209_220938


namespace last_two_digits_10_93_10_31_plus_3_eq_08_l2209_220909

def last_two_digits_fraction_floor (n m d : ℕ) : ℕ :=
  let x := 10^n
  let y := 10^m + d
  (x / y) % 100

theorem last_two_digits_10_93_10_31_plus_3_eq_08 :
  last_two_digits_fraction_floor 93 31 3 = 08 :=
by
  sorry

end last_two_digits_10_93_10_31_plus_3_eq_08_l2209_220909


namespace student_percentage_first_subject_l2209_220926

theorem student_percentage_first_subject
  (P : ℝ)
  (h1 : (P + 60 + 70) / 3 = 60) : P = 50 :=
  sorry

end student_percentage_first_subject_l2209_220926


namespace length_of_real_axis_of_hyperbola_l2209_220986

theorem length_of_real_axis_of_hyperbola :
  ∀ (x y : ℝ), 2 * x^2 - y^2 = 8 -> ∃ a : ℝ, 2 * a = 4 :=
by
intro x y h
sorry

end length_of_real_axis_of_hyperbola_l2209_220986


namespace find_missing_number_l2209_220976

theorem find_missing_number (x : ℕ) (h : x * 240 = 173 * 240) : x = 173 :=
sorry

end find_missing_number_l2209_220976


namespace larger_screen_diagonal_length_l2209_220937

theorem larger_screen_diagonal_length :
  (∃ d : ℝ, (∀ a : ℝ, a = 16 → d^2 = 2 * (a^2 + 34)) ∧ d = Real.sqrt 580) :=
by
  sorry

end larger_screen_diagonal_length_l2209_220937


namespace jane_paid_cashier_l2209_220948

-- Define the conditions in Lean
def skirts_bought : ℕ := 2
def price_per_skirt : ℕ := 13
def blouses_bought : ℕ := 3
def price_per_blouse : ℕ := 6
def change_received : ℤ := 56

-- Calculate the total cost in Lean
def cost_of_skirts : ℕ := skirts_bought * price_per_skirt
def cost_of_blouses : ℕ := blouses_bought * price_per_blouse
def total_cost : ℕ := cost_of_skirts + cost_of_blouses
def amount_paid : ℤ := total_cost + change_received

-- Lean statement to prove the question
theorem jane_paid_cashier :
  amount_paid = 100 :=
by
  sorry

end jane_paid_cashier_l2209_220948


namespace totalSleepIsThirtyHours_l2209_220915

-- Define constants and conditions
def recommendedSleep : ℝ := 8
def sleepOnTwoDays : ℝ := 3
def percentageSleepOnOtherDays : ℝ := 0.6
def daysInWeek : ℕ := 7
def daysWithThreeHoursSleep : ℕ := 2
def remainingDays : ℕ := daysInWeek - daysWithThreeHoursSleep

-- Define total sleep calculation
theorem totalSleepIsThirtyHours :
  let sleepOnFirstTwoDays := (daysWithThreeHoursSleep : ℝ) * sleepOnTwoDays
  let sleepOnRemainingDays := (remainingDays : ℝ) * (recommendedSleep * percentageSleepOnOtherDays)
  sleepOnFirstTwoDays + sleepOnRemainingDays = 30 := 
by
  sorry

end totalSleepIsThirtyHours_l2209_220915


namespace remainder_of_2n_div_7_l2209_220903

theorem remainder_of_2n_div_7 (n : ℤ) (k : ℤ) (h : n = 7 * k + 2) : (2 * n) % 7 = 4 :=
by
  sorry

end remainder_of_2n_div_7_l2209_220903


namespace proof_problem_exists_R1_R2_l2209_220900

def problem (R1 R2 : ℕ) : Prop :=
  let F1_R1 := (4 * R1 + 5) / (R1^2 - 1)
  let F2_R1 := (5 * R1 + 4) / (R1^2 - 1)
  let F1_R2 := (3 * R2 + 2) / (R2^2 - 1)
  let F2_R2 := (2 * R2 + 3) / (R2^2 - 1)
  F1_R1 = F1_R2 ∧ F2_R1 = F2_R2 ∧ R1 + R2 = 14

theorem proof_problem_exists_R1_R2 : ∃ (R1 R2 : ℕ), problem R1 R2 :=
sorry

end proof_problem_exists_R1_R2_l2209_220900


namespace minimum_button_presses_to_exit_l2209_220953

def arms_after (r y : ℕ) : ℕ := 3 + r - 2 * y
def doors_after (y g : ℕ) : ℕ := 3 + y - 2 * g

theorem minimum_button_presses_to_exit :
  ∃ r y g : ℕ, arms_after r y = 0 ∧ doors_after y g = 0 ∧ r + y + g = 9 :=
sorry

end minimum_button_presses_to_exit_l2209_220953


namespace rabbits_distribution_l2209_220961

def num_ways_to_distribute : ℕ :=
  20 + 390 + 150

theorem rabbits_distribution :
  num_ways_to_distribute = 560 := by
  sorry

end rabbits_distribution_l2209_220961


namespace complement_intersection_M_N_l2209_220944

def M : Set ℝ := {x | x < 3}
def N : Set ℝ := {x | x > -1}
def U : Set ℝ := Set.univ

theorem complement_intersection_M_N :
  U \ (M ∩ N) = {x | x ≤ -1} ∪ {x | x ≥ 3} :=
by
  sorry

end complement_intersection_M_N_l2209_220944


namespace find_x_for_g_l2209_220923

noncomputable def g (x : ℝ) : ℝ := (↑((x + 5) / 3) : ℝ)^(1/3 : ℝ)

theorem find_x_for_g :
  ∃ x : ℝ, g (3 * x) = 3 * g x ↔ x = -65 / 12 :=
by
  sorry

end find_x_for_g_l2209_220923


namespace product_zero_when_a_is_three_l2209_220929

theorem product_zero_when_a_is_three (a : ℤ) (h : a = 3) :
  (a - 9) * (a - 8) * (a - 7) * (a - 6) * (a - 5) * (a - 4) * (a - 3) * (a - 2) * (a - 1) * a = 0 :=
by
  cases h
  sorry

end product_zero_when_a_is_three_l2209_220929


namespace Toms_swimming_speed_is_2_l2209_220965

theorem Toms_swimming_speed_is_2
  (S : ℝ)
  (h1 : 2 * S + 4 * S = 12) :
  S = 2 :=
by
  sorry

end Toms_swimming_speed_is_2_l2209_220965


namespace divisibility_by_1897_l2209_220947

theorem divisibility_by_1897 (n : ℕ) : 1897 ∣ (2903 ^ n - 803 ^ n - 464 ^ n + 261 ^ n) :=
sorry

end divisibility_by_1897_l2209_220947


namespace cubic_roots_identity_l2209_220933

theorem cubic_roots_identity (p q r : ℝ) 
  (h1 : p + q + r = 0) 
  (h2 : p * q + q * r + r * p = -3) 
  (h3 : p * q * r = -2) : 
  p * (q - r) ^ 2 + q * (r - p) ^ 2 + r * (p - q) ^ 2 = 0 := 
by
  sorry

end cubic_roots_identity_l2209_220933


namespace solve_inequality_l2209_220971

noncomputable def within_interval (x : ℝ) : Prop :=
  x > -3 ∧ x < 5

theorem solve_inequality (x : ℝ) :
  (x^3 - 125) / (x + 3) < 0 ↔ within_interval x :=
sorry

end solve_inequality_l2209_220971


namespace least_number_subtracted_l2209_220977

-- Define the original number and the divisor
def original_number : ℕ := 427398
def divisor : ℕ := 14

-- Define the least number to be subtracted
def remainder := original_number % divisor
def least_number := remainder

-- The statement to be proven
theorem least_number_subtracted : least_number = 6 :=
by
  sorry

end least_number_subtracted_l2209_220977


namespace intervals_of_decrease_l2209_220978

open Real

noncomputable def func (x : ℝ) : ℝ :=
  cos (2 * x) + 2 * sin x

theorem intervals_of_decrease :
  {x | deriv func x < 0 ∧ 0 < x ∧ x < 2 * π} =
  {x | (π / 6 < x ∧ x < π / 2) ∨ (5 * π / 6 < x ∧ x < 3 * π / 2)} :=
by
  sorry

end intervals_of_decrease_l2209_220978


namespace salad_cost_is_correct_l2209_220958

-- Definitions of costs according to the given conditions
def muffin_cost : ℝ := 2
def coffee_cost : ℝ := 4
def soup_cost : ℝ := 3
def lemonade_cost : ℝ := 0.75

def breakfast_cost : ℝ := muffin_cost + coffee_cost
def lunch_cost : ℝ := breakfast_cost + 3

def salad_cost : ℝ := lunch_cost - (soup_cost + lemonade_cost)

-- Statement to prove
theorem salad_cost_is_correct : salad_cost = 5.25 :=
by
  sorry

end salad_cost_is_correct_l2209_220958


namespace original_square_area_l2209_220989

theorem original_square_area (s : ℕ) (h1 : s + 5 = s + 5) (h2 : (s + 5)^2 = s^2 + 225) : s^2 = 400 :=
by
  sorry

end original_square_area_l2209_220989


namespace charley_pencils_lost_l2209_220928

theorem charley_pencils_lost :
  ∃ x : ℕ, (30 - x - (1/3 : ℝ) * (30 - x) = 16) ∧ x = 6 :=
by
  -- Since x must be an integer and the equations naturally produce whole numbers,
  -- we work within the context of natural numbers, then cast to real as needed.
  use 6
  -- Express the main condition in terms of x
  have h: (30 - 6 - (1/3 : ℝ) * (30 - 6) = 16) := by sorry
  exact ⟨h, rfl⟩

end charley_pencils_lost_l2209_220928


namespace ball_box_distribution_l2209_220942

theorem ball_box_distribution :
  ∃ (distinct_ways : ℕ), distinct_ways = 7 :=
by
  let num_balls := 5
  let num_boxes := 4
  sorry

end ball_box_distribution_l2209_220942
