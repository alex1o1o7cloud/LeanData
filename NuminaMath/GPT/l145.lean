import Mathlib

namespace solve_for_x_l145_14522

theorem solve_for_x (x : ℕ) : 100^3 = 10^x → x = 6 := by
  sorry

end solve_for_x_l145_14522


namespace ellipse_standard_eq_l145_14552

theorem ellipse_standard_eq
  (e : ℝ) (a b : ℝ) (h1 : e = 1 / 2) (h2 : 2 * a = 4) (h3 : b^2 = a^2 - (a * e)^2)
  : (∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1) ↔
    ( ∀ x y : ℝ, x^2 / 4 + y^2 / 3 = 1 ) :=
by
  sorry

end ellipse_standard_eq_l145_14552


namespace square_pyramid_intersection_area_l145_14561

theorem square_pyramid_intersection_area (a b c d e : ℝ) (h_midpoints : a = 2 ∧ b = 4 ∧ c = 4 ∧ d = 4 ∧ e = 4) : 
  ∃ p : ℝ, (p = 80) :=
by
  sorry

end square_pyramid_intersection_area_l145_14561


namespace minimalYellowFraction_l145_14576

-- Definitions
def totalSurfaceArea (sideLength : ℕ) : ℕ := 6 * (sideLength * sideLength)

def minimalYellowExposedArea : ℕ := 15

theorem minimalYellowFraction (sideLength : ℕ) (totalYellow : ℕ) (totalBlue : ℕ) 
    (totalCubes : ℕ) (yellowExposed : ℕ) :
    sideLength = 4 → totalYellow = 16 → totalBlue = 48 →
    totalCubes = 64 → yellowExposed = minimalYellowExposedArea →
    (yellowExposed / (totalSurfaceArea sideLength) : ℚ) = 5 / 32 :=
by
  sorry

end minimalYellowFraction_l145_14576


namespace ed_lighter_than_al_l145_14599

theorem ed_lighter_than_al :
  let Al := Ben + 25
  let Ben := Carl - 16
  let Ed := 146
  let Carl := 175
  Al - Ed = 38 :=
by
  sorry

end ed_lighter_than_al_l145_14599


namespace seq_inequality_l145_14555

noncomputable def sequence_of_nonneg_reals (a : ℕ → ℝ) : Prop :=
  ∀ m n : ℕ, a (n + m) ≤ a n + a m

theorem seq_inequality
  (a : ℕ → ℝ)
  (h : sequence_of_nonneg_reals a)
  (h_nonneg : ∀ n, 0 ≤ a n) :
  ∀ n m : ℕ, m > 0 → n ≥ m → a n ≤ m * a 1 + ((n / m) - 1) * a m := 
by
  sorry

end seq_inequality_l145_14555


namespace jack_bought_apples_l145_14523

theorem jack_bought_apples :
  ∃ n : ℕ, 
    (∃ k : ℕ, k = 10 ∧ ∃ m : ℕ, m = 5 * 9 ∧ n = k + m) ∧ n = 55 :=
by
  sorry

end jack_bought_apples_l145_14523


namespace admission_fee_for_adults_l145_14572

theorem admission_fee_for_adults (C : ℝ) (N M N_c N_a : ℕ) (A : ℝ) 
  (h1 : C = 1.50) 
  (h2 : N = 2200) 
  (h3 : M = 5050) 
  (h4 : N_c = 700) 
  (h5 : N_a = 1500) :
  A = 2.67 := 
by
  sorry

end admission_fee_for_adults_l145_14572


namespace problem_statement_l145_14548

def U : Set ℝ := Set.univ
def A : Set ℝ := {x | (x + 2) * (x - 1) > 0}
def B : Set ℝ := {x | -3 ≤ x ∧ x < 0}
def C_U (B : Set ℝ) : Set ℝ := {x | x ∉ B}

theorem problem_statement : A ∪ C_U B = {x | x < -2 ∨ x ≥ 0} :=
by
  sorry

end problem_statement_l145_14548


namespace corrected_multiplication_result_l145_14547

theorem corrected_multiplication_result :
  ∃ n : ℕ, 987 * n = 559989 ∧ 987 * n ≠ 559981 ∧ 559981 % 100 = 98 :=
by
  sorry

end corrected_multiplication_result_l145_14547


namespace inequality_exponentiation_l145_14582

theorem inequality_exponentiation (a b c : ℝ) (ha : 0 < a) (hab : a < b) (hb : b < 1) (hc : c > 1) : 
  a * b^c > b * a^c := 
sorry

end inequality_exponentiation_l145_14582


namespace triangle_angle_sum_l145_14516

theorem triangle_angle_sum (α β γ : ℝ) (h : α + β + γ = 180) (h1 : α > 60) (h2 : β > 60) (h3 : γ > 60) : false :=
sorry

end triangle_angle_sum_l145_14516


namespace course_selection_l145_14562

noncomputable def number_of_ways (nA nB : ℕ) : ℕ :=
  (Nat.choose nA 2) * (Nat.choose nB 1) + (Nat.choose nA 1) * (Nat.choose nB 2)

theorem course_selection :
  (number_of_ways 3 4) = 30 :=
by
  sorry

end course_selection_l145_14562


namespace cost_price_per_meter_l145_14534

-- Define the given conditions
def selling_price : ℕ := 8925
def meters : ℕ := 85
def profit_per_meter : ℕ := 35

-- Define the statement to be proved
theorem cost_price_per_meter :
  (selling_price - profit_per_meter * meters) / meters = 70 := 
by
  sorry

end cost_price_per_meter_l145_14534


namespace dave_more_than_derek_l145_14501

def derek_initial : ℕ := 40
def derek_spent_on_self1 : ℕ := 14
def derek_spent_on_dad : ℕ := 11
def derek_spent_on_self2 : ℕ := 5

def dave_initial : ℕ := 50
def dave_spent_on_mom : ℕ := 7

def derek_remaining : ℕ := derek_initial - (derek_spent_on_self1 + derek_spent_on_dad + derek_spent_on_self2)
def dave_remaining : ℕ := dave_initial - dave_spent_on_mom

theorem dave_more_than_derek : dave_remaining - derek_remaining = 33 :=
by
  -- The proof goes here
  sorry

end dave_more_than_derek_l145_14501


namespace mans_rate_in_still_water_l145_14520

/-- The man's rowing speed in still water given his rowing speeds with and against the stream. -/
theorem mans_rate_in_still_water (v_with_stream v_against_stream : ℝ) (h1 : v_with_stream = 6) (h2 : v_against_stream = 2) : (v_with_stream + v_against_stream) / 2 = 4 := by
  sorry

end mans_rate_in_still_water_l145_14520


namespace totalPears_l145_14584

-- Define the number of pears picked by Sara and Sally
def saraPears : ℕ := 45
def sallyPears : ℕ := 11

-- Statement to prove
theorem totalPears : saraPears + sallyPears = 56 :=
by
  sorry

end totalPears_l145_14584


namespace combined_girls_avg_l145_14506

noncomputable def centralHS_boys_avg := 68
noncomputable def deltaHS_boys_avg := 78
noncomputable def combined_boys_avg := 74
noncomputable def centralHS_girls_avg := 72
noncomputable def deltaHS_girls_avg := 85
noncomputable def centralHS_combined_avg := 70
noncomputable def deltaHS_combined_avg := 80

theorem combined_girls_avg (C c D d : ℝ) 
  (h1 : (68 * C + 72 * c) / (C + c) = 70)
  (h2 : (78 * D + 85 * d) / (D + d) = 80)
  (h3 : (68 * C + 78 * D) / (C + D) = 74) :
  (3/7 * 72 + 4/7 * 85) = 79 := 
by 
  sorry

end combined_girls_avg_l145_14506


namespace train_speed_is_72_kmh_l145_14586

noncomputable def train_length : ℝ := 110
noncomputable def bridge_length : ℝ := 175
noncomputable def crossing_time : ℝ := 14.248860091192705

theorem train_speed_is_72_kmh :
  (train_length + bridge_length) / crossing_time * 3.6 = 72 := by
  sorry

end train_speed_is_72_kmh_l145_14586


namespace eccentricity_of_given_ellipse_l145_14593

noncomputable def ellipse_eccentricity (φ : Real) : Real :=
  let x := 3 * Real.cos φ
  let y := 5 * Real.sin φ
  let a := 5
  let b := 3
  let c := Real.sqrt (a * a - b * b)
  c / a

theorem eccentricity_of_given_ellipse (φ : Real) :
  ellipse_eccentricity φ = 4 / 5 :=
sorry

end eccentricity_of_given_ellipse_l145_14593


namespace quadratic_has_unique_solution_l145_14538

theorem quadratic_has_unique_solution (k : ℝ) :
  (∀ x : ℝ, (x + 6) * (x + 3) = k + 3 * x) → k = 9 :=
by
  intro h
  sorry

end quadratic_has_unique_solution_l145_14538


namespace ratio_of_girls_participated_to_total_l145_14530

noncomputable def ratio_participating_girls {a : ℕ} (h1 : a > 0)
    (equal_boys_girls : ∀ (b g : ℕ), b = a ∧ g = a)
    (girls_participated : ℕ := (3 * a) / 4)
    (boys_participated : ℕ := (2 * a) / 3) :
    ℚ :=
    girls_participated / (girls_participated + boys_participated)

theorem ratio_of_girls_participated_to_total {a : ℕ} (h1 : a > 0)
    (equal_boys_girls : ∀ (b g : ℕ), b = a ∧ g = a)
    (girls_participated : ℕ := (3 * a) / 4)
    (boys_participated : ℕ := (2 * a) / 3) :
    ratio_participating_girls h1 equal_boys_girls girls_participated boys_participated = 9 / 17 :=
by
    sorry

end ratio_of_girls_participated_to_total_l145_14530


namespace distribute_6_balls_in_3_boxes_l145_14571

def number_of_ways_to_distribute_balls (balls boxes : Nat) : Nat :=
  boxes ^ balls

theorem distribute_6_balls_in_3_boxes : number_of_ways_to_distribute_balls 6 3 = 729 := by
  sorry

end distribute_6_balls_in_3_boxes_l145_14571


namespace sequence_solution_l145_14580

theorem sequence_solution (a : ℕ → ℝ) 
  (h1 : a 1 = 1)
  (h2 : ∀ (m n : ℕ), 0 < m → 0 < n → |a n - a m| ≤ (2 * m * n) / (m ^ 2 + n ^ 2)) :
  ∀ (n : ℕ), a n = 1 :=
by
  sorry

end sequence_solution_l145_14580


namespace arithmetic_sequence_properties_geometric_sequence_properties_l145_14591

-- Define the arithmetic sequence {a_n}
def a (n : ℕ) : ℕ :=
  2 * n - 1

-- Define the sum of the first n terms of {a_n}
def S (n : ℕ) : ℕ :=
  n ^ 2

-- Prove the nth term and the sum of the first n terms of {a_n}
theorem arithmetic_sequence_properties (n : ℕ) :
  a n = 2 * n - 1 ∧ S n = n ^ 2 :=
by sorry

-- Define the geometric sequence {b_n}
def b (n : ℕ) : ℕ :=
  2 ^ (2 * n - 1)

-- Define the sum of the first n terms of {b_n}
def T (n : ℕ) : ℕ :=
  (2 ^ n * (4 ^ n - 1)) / 3

-- Prove the nth term and the sum of the first n terms of {b_n}
theorem geometric_sequence_properties (n : ℕ) (a4 S4 : ℕ) (q : ℕ)
  (h_a4 : a4 = a 4)
  (h_S4 : S4 = S 4)
  (h_q : q ^ 2 - (a4 + 1) * q + S4 = 0) :
  b n = 2 ^ (2 * n - 1) ∧ T n = (2 ^ n * (4 ^ n - 1)) / 3 :=
by sorry

end arithmetic_sequence_properties_geometric_sequence_properties_l145_14591


namespace range_of_m_l145_14503

theorem range_of_m (a b m : ℝ) (h₀ : a > 0) (h₁ : b > 1) (h₂ : a + b = 2) (h₃ : ∀ m, (4/a + 1/(b-1)) > m^2 + 8*m) : -9 < m ∧ m < 1 :=
by
  sorry

end range_of_m_l145_14503


namespace swimming_club_cars_l145_14504

theorem swimming_club_cars (c : ℕ) :
  let vans := 3
  let people_per_car := 5
  let people_per_van := 3
  let max_people_per_car := 6
  let max_people_per_van := 8
  let extra_people := 17
  let total_people := 5 * c + (people_per_van * vans)
  let max_capacity := max_people_per_car * c + (max_people_per_van * vans)
  (total_people + extra_people = max_capacity) → c = 2 := by
  sorry

end swimming_club_cars_l145_14504


namespace average_growth_rate_income_prediction_l145_14574

-- Define the given conditions
def income2018 : ℝ := 20000
def income2020 : ℝ := 24200
def growth_rate : ℝ := 0.1
def predicted_income2021 : ℝ := 26620

-- Lean 4 statement for the first part of the problem
theorem average_growth_rate :
  (income2020 = income2018 * (1 + growth_rate)^2) →
  growth_rate = 0.1 :=
by
  intros h
  sorry

-- Lean 4 statement for the second part of the problem
theorem income_prediction :
  (income2020 = income2018 * (1 + growth_rate)^2) →
  (growth_rate = 0.1) →
  (income2018 * (1 + growth_rate)^3 = predicted_income2021) :=
by
  intros h1 h2
  sorry

end average_growth_rate_income_prediction_l145_14574


namespace geometric_sequence_problem_l145_14525

variables {a : ℕ → ℝ} {q : ℝ}

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a 1 * q ^ n

theorem geometric_sequence_problem (h1 : a 1 + a 1 * q ^ 2 = 10) (h2 : a 1 * q + a 1 * q ^ 3 = 5) (h3 : geometric_sequence a q) :
  a 8 = 1 / 16 := sorry

end geometric_sequence_problem_l145_14525


namespace m_value_if_Q_subset_P_l145_14526

noncomputable def P : Set ℝ := {x | x^2 = 1}
def Q (m : ℝ) : Set ℝ := {x | m * x = 1}
def m_values (m : ℝ) : Prop := Q m ⊆ P → m = 0 ∨ m = 1 ∨ m = -1

theorem m_value_if_Q_subset_P (m : ℝ) : m_values m :=
sorry

end m_value_if_Q_subset_P_l145_14526


namespace g_evaluation_l145_14529

def g (a b : ℚ) : ℚ :=
  if a + b ≤ 4 then (2 * a * b - a + 3) / (3 * a)
  else (a * b - b - 1) / (-3 * b)

theorem g_evaluation : g 2 1 + g 2 4 = 7 / 12 := 
by {
  sorry
}

end g_evaluation_l145_14529


namespace circle_radius_eq_two_l145_14527

theorem circle_radius_eq_two (x y : ℝ) : (x^2 + y^2 + 1 = 2 * x + 4 * y) → (∃ c : ℝ × ℝ, ∃ r : ℝ, ((x - c.1)^2 + (y - c.2)^2 = r^2) ∧ r = 2) := by
  sorry

end circle_radius_eq_two_l145_14527


namespace find_y_for_orthogonal_vectors_l145_14560

theorem find_y_for_orthogonal_vectors : 
  (∀ y, ((3:ℝ) * y + (-4:ℝ) * 9 = 0) → y = 12) :=
by
  sorry

end find_y_for_orthogonal_vectors_l145_14560


namespace total_kayaks_built_l145_14575

/-- Geometric sequence sum definition -/
def geom_sum (a r : ℕ) (n : ℕ) : ℕ :=
  if r = 1 then n * a
  else a * (r ^ n - 1) / (r - 1)

/-- Problem statement: Prove that the total number of kayaks built by the end of June is 726 -/
theorem total_kayaks_built : geom_sum 6 3 5 = 726 :=
  sorry

end total_kayaks_built_l145_14575


namespace multiple_of_x_l145_14553

theorem multiple_of_x (k x y : ℤ) (hk : k * x + y = 34) (hx : 2 * x - y = 20) (hy : y^2 = 4) : k = 4 :=
sorry

end multiple_of_x_l145_14553


namespace min_liars_in_presidium_l145_14546

-- Define the conditions of the problem
def liars_and_truthlovers (grid : ℕ → ℕ → Prop) : Prop :=
  ∃ n : ℕ, n = 32 ∧ 
  (∀ i j, i < 4 ∧ j < 8 → 
    (∃ ni nj, (ni = i + 1 ∨ ni = i - 1 ∨ ni = i ∨ nj = j + 1 ∨ nj = j - 1 ∨ nj = j) ∧
      (ni < 4 ∧ nj < 8) → (grid i j ↔ ¬ grid ni nj)))

-- Define proof problem
theorem min_liars_in_presidium (grid : ℕ → ℕ → Prop) :
  liars_and_truthlovers grid → (∃ l, l = 8) := by
  sorry

end min_liars_in_presidium_l145_14546


namespace second_storm_duration_l145_14565

theorem second_storm_duration (x y : ℕ) 
  (h1 : x + y = 45) 
  (h2 : 30 * x + 15 * y = 975) : 
  y = 25 :=
by
  sorry

end second_storm_duration_l145_14565


namespace quadratic_root_square_condition_l145_14579

theorem quadratic_root_square_condition (p q r : ℝ) 
  (h1 : ∃ α β : ℝ, α + β = -q / p ∧ α * β = r / p ∧ β = α^2) : p - 4 * q ≥ 0 :=
sorry

end quadratic_root_square_condition_l145_14579


namespace classroom_problem_l145_14510

noncomputable def classroom_problem_statement : Prop :=
  ∀ (B G : ℕ) (b g : ℝ),
    b > 0 →
    g > 0 →
    B > 0 →
    G > 0 →
    ¬ ((B * g + G * b) / (B + G) = b + g ∧ b > 0 ∧ g > 0)

theorem classroom_problem : classroom_problem_statement :=
  by
    intros B G b g hb_gt0 hg_gt0 hB_gt0 hG_gt0
    sorry

end classroom_problem_l145_14510


namespace minimum_m_plus_n_l145_14590

theorem minimum_m_plus_n (m n : ℕ) (h1 : 98 * m = n ^ 3) (h2 : 0 < m) (h3 : 0 < n) : m + n = 42 :=
sorry

end minimum_m_plus_n_l145_14590


namespace rope_segments_divided_l145_14528

theorem rope_segments_divided (folds1 folds2 : ℕ) (cut : ℕ) (h_folds1 : folds1 = 3) (h_folds2 : folds2 = 2) (h_cut : cut = 1) :
  (folds1 * folds2 + cut = 7) :=
by {
  -- Proof steps would go here
  sorry
}

end rope_segments_divided_l145_14528


namespace find_g_25_l145_14583

noncomputable def g (x : ℝ) : ℝ := sorry

axiom h₁ : ∀ (x y : ℝ), x > 0 → y > 0 → g (x / y) = (y / x) * g x
axiom h₂ : g 50 = 4

theorem find_g_25 : g 25 = 4 / 25 :=
by {
  sorry
}

end find_g_25_l145_14583


namespace revenue_decrease_1_percent_l145_14589

variable (T C : ℝ)  -- Assumption: T and C are real numbers representing the original tax and consumption

noncomputable def original_revenue : ℝ := T * C
noncomputable def new_tax_rate : ℝ := T * 0.90
noncomputable def new_consumption : ℝ := C * 1.10
noncomputable def new_revenue : ℝ := new_tax_rate T * new_consumption C

theorem revenue_decrease_1_percent :
  new_revenue T C = 0.99 * original_revenue T C := by
  sorry

end revenue_decrease_1_percent_l145_14589


namespace work_completion_days_l145_14544

variable (Paul_days Rose_days Sam_days : ℕ)

def Paul_rate := 1 / 80
def Rose_rate := 1 / 120
def Sam_rate := 1 / 150

def combined_rate := Paul_rate + Rose_rate + Sam_rate

noncomputable def days_to_complete_work := 1 / combined_rate

theorem work_completion_days :
  Paul_days = 80 →
  Rose_days = 120 →
  Sam_days = 150 →
  days_to_complete_work = 37 := 
by
  intros
  simp only [Paul_rate, Rose_rate, Sam_rate, combined_rate, days_to_complete_work]
  sorry

end work_completion_days_l145_14544


namespace sum_of_common_ratios_l145_14536

theorem sum_of_common_ratios (k p r a2 a3 b2 b3 : ℝ)
  (h1 : a3 = k * p^2) (h2 : a2 = k * p) 
  (h3 : b3 = k * r^2) (h4 : b2 = k * r)
  (h5 : p ≠ r)
  (h6 : 3 * a3 - 4 * b3 = 5 * (3 * a2 - 4 * b2)) :
  p + r = 5 :=
by {
  sorry
}

end sum_of_common_ratios_l145_14536


namespace intersection_M_N_l145_14531

-- Define the sets M and N
def M : Set ℝ := {x | 0 < x ∧ x < 2}
def N : Set ℝ := {x | x ≥ 1} ∪ {x | x ≤ -3}

-- Prove the intersection of M and N is [1, 2)
theorem intersection_M_N : (M ∩ N) = {x | 1 ≤ x ∧ x < 2} := by
  sorry

end intersection_M_N_l145_14531


namespace most_followers_is_sarah_l145_14533

def initial_followers_susy : ℕ := 100
def initial_followers_sarah : ℕ := 50

def susy_week1_new : ℕ := 40
def susy_week2_new := susy_week1_new / 2
def susy_week3_new := susy_week2_new / 2
def susy_total_new := susy_week1_new + susy_week2_new + susy_week3_new
def susy_final_followers := initial_followers_susy + susy_total_new

def sarah_week1_new : ℕ := 90
def sarah_week2_new := sarah_week1_new / 3
def sarah_week3_new := sarah_week2_new / 3
def sarah_total_new := sarah_week1_new + sarah_week2_new + sarah_week3_new
def sarah_final_followers := initial_followers_sarah + sarah_total_new

theorem most_followers_is_sarah : 
    sarah_final_followers ≥ susy_final_followers := by
  sorry

end most_followers_is_sarah_l145_14533


namespace smallest_n_value_l145_14550

-- Define the given expression
def exp := (2^5) * (6^2) * (7^3) * (13^4)

-- Define the conditions
def condition_5_2 (n : ℕ) := ∃ k, n * exp = k * 5^2
def condition_3_3 (n : ℕ) := ∃ k, n * exp = k * 3^3
def condition_11_2 (n : ℕ) := ∃ k, n * exp = k * 11^2

-- Define the smallest possible value of n
def smallest_n (n : ℕ) : Prop :=
  condition_5_2 n ∧ condition_3_3 n ∧ condition_11_2 n ∧ ∀ m, (condition_5_2 m ∧ condition_3_3 m ∧ condition_11_2 m) → m ≥ n

-- The theorem statement
theorem smallest_n_value : smallest_n 9075 :=
  by
    sorry

end smallest_n_value_l145_14550


namespace g_of_five_eq_one_l145_14532

variable (g : ℝ → ℝ)

theorem g_of_five_eq_one (h1 : ∀ x y : ℝ, g (x - y) = g x * g y)
    (h2 : ∀ x : ℝ, g x ≠ 0) : g 5 = 1 :=
sorry

end g_of_five_eq_one_l145_14532


namespace find_k_for_circle_l145_14511

theorem find_k_for_circle (k : ℝ) : (∃ x y : ℝ, (x^2 + 8*x + y^2 + 4*y - k = 0) ∧ (x + 4)^2 + (y + 2)^2 = 25) → k = 5 := 
by 
  sorry

end find_k_for_circle_l145_14511


namespace jonas_shoes_l145_14585

theorem jonas_shoes (socks pairs_of_pants t_shirts shoes : ℕ) (new_socks : ℕ) (h1 : socks = 20) (h2 : pairs_of_pants = 10) (h3 : t_shirts = 10) (h4 : new_socks = 35 ∧ (socks + new_socks = 35)) :
  shoes = 35 :=
by
  sorry

end jonas_shoes_l145_14585


namespace number_of_indeterminate_conditions_l145_14570

noncomputable def angle_sum (A B C : ℝ) : Prop := A + B + C = 180
noncomputable def condition1 (A B C : ℝ) : Prop := A + B = C
noncomputable def condition2 (A B C : ℝ) : Prop := A = C / 6 ∧ B = 2 * (C / 6)
noncomputable def condition3 (A B : ℝ) : Prop := A = 90 - B
noncomputable def condition4 (A B C : ℝ) : Prop := A = B ∧ B = C
noncomputable def condition5 (A B C : ℝ) : Prop := 2 * A = C ∧ 2 * B = C
noncomputable def is_right_triangle (C : ℝ) : Prop := C = 90

theorem number_of_indeterminate_conditions (A B C : ℝ) :
  (angle_sum A B C) →
  (condition1 A B C → is_right_triangle C) →
  (condition2 A B C → is_right_triangle C) →
  (condition3 A B → is_right_triangle C) →
  (condition4 A B C → ¬ is_right_triangle C) →
  (condition5 A B C → is_right_triangle C) →
  ∃ n, n = 1 :=
sorry

end number_of_indeterminate_conditions_l145_14570


namespace find_n_cosine_l145_14521

theorem find_n_cosine (n : ℤ) (h1 : 100 ≤ n ∧ n ≤ 300) (h2 : Real.cos (n : ℝ) = Real.cos 140) : n = 220 :=
by
  sorry

end find_n_cosine_l145_14521


namespace workers_contribution_eq_l145_14512

variable (W C : ℕ)

theorem workers_contribution_eq :
  W * C = 300000 → W * (C + 50) = 320000 → W = 400 :=
by
  intros h1 h2
  -- Proof goes here
  sorry

end workers_contribution_eq_l145_14512


namespace solution_strategy_l145_14509

-- Defining the total counts for the groups
def total_elderly : ℕ := 28
def total_middle_aged : ℕ := 54
def total_young : ℕ := 81

-- The sample size we need
def sample_size : ℕ := 36

-- Proposing the strategy
def appropriate_sampling_method : Prop := 
  (total_elderly - 1) % sample_size.gcd (total_middle_aged.gcd total_young) = 0

theorem solution_strategy :
  appropriate_sampling_method :=
by {
  sorry
}

end solution_strategy_l145_14509


namespace x_investment_amount_l145_14564

variable (X : ℝ)
variable (investment_y : ℝ := 15000)
variable (total_profit : ℝ := 1600)
variable (x_share : ℝ := 400)

theorem x_investment_amount :
  (total_profit - x_share) / investment_y = x_share / X → X = 5000 :=
by
  intro ratio
  have h1: 1200 / 15000 = 400 / 5000 :=
    by sorry
  have h2: X = 5000 :=
    by sorry
  exact h2

end x_investment_amount_l145_14564


namespace range_of_a_l145_14513

open Real

-- The quadratic expression
def quadratic (a x : ℝ) : ℝ := a*x^2 + 2*x + a

-- The condition of the problem
def quadratic_nonnegative_for_all (a : ℝ) := ∀ x : ℝ, quadratic a x ≥ 0

-- The theorem to be proven
theorem range_of_a (a : ℝ) (h : quadratic_nonnegative_for_all a) : a ≥ 1 :=
sorry

end range_of_a_l145_14513


namespace lincoln_county_houses_l145_14567

theorem lincoln_county_houses (original_houses : ℕ) (built_houses : ℕ) (total_houses : ℕ) 
(h1 : original_houses = 20817) 
(h2 : built_houses = 97741) 
(h3 : total_houses = original_houses + built_houses) : 
total_houses = 118558 :=
by
  -- proof omitted
  sorry

end lincoln_county_houses_l145_14567


namespace David_pushups_l145_14558

-- Definitions and setup conditions
def Zachary_pushups : ℕ := 7
def additional_pushups : ℕ := 30

-- Theorem statement to be proved
theorem David_pushups 
  (zachary_pushups : ℕ) 
  (additional_pushups : ℕ) 
  (Zachary_pushups_val : zachary_pushups = Zachary_pushups) 
  (additional_pushups_val : additional_pushups = additional_pushups) :
  zachary_pushups + additional_pushups = 37 :=
sorry

end David_pushups_l145_14558


namespace find_RS_length_l145_14543

-- Define the conditions and the problem in Lean

theorem find_RS_length
  (radius : ℝ)
  (P Q R S T : ℝ)
  (center_to_T : ℝ)
  (PT : ℝ)
  (PQ : ℝ)
  (RT TS : ℝ)
  (h_radius : radius = 7)
  (h_center_to_T : center_to_T = 3)
  (h_PT : PT = 8)
  (h_bisect_PQ : PQ = 2 * PT)
  (h_intersecting_chords : PT * (PQ / 2) = RT * TS)
  (h_perfect_square : ∃ k : ℝ, k^2 = RT * TS) :
  RS = 16 :=
by
  sorry

end find_RS_length_l145_14543


namespace evaluate_polynomial_l145_14573

theorem evaluate_polynomial : 103^4 - 4 * 103^3 + 6 * 103^2 - 4 * 103 + 1 = 108243216 :=
by
  sorry

end evaluate_polynomial_l145_14573


namespace probability_of_white_crows_remain_same_l145_14581

theorem probability_of_white_crows_remain_same (a b c d : ℕ) (h1 : a + b = 50) (h2 : c + d = 50) 
  (ha1 : a > 0) (h3 : b ≥ a) (h4 : d ≥ c - 1) :
  ((b - a) * (d - c) + a + b) / (50 * 51) > (bc + ad) / (50 * 51)
:= by
  -- We need to show that the probability of the number of white crows on the birch remaining the same 
  -- is greater than the probability of it changing.
  sorry

end probability_of_white_crows_remain_same_l145_14581


namespace dodecagon_diagonals_l145_14549

def num_diagonals (n : ℕ) : ℕ :=
  n * (n - 3) / 2

theorem dodecagon_diagonals : num_diagonals 12 = 54 :=
by
  -- by sorry means we skip the actual proof
  sorry

end dodecagon_diagonals_l145_14549


namespace tan_shifted_value_l145_14508

theorem tan_shifted_value (x : ℝ) (h : Real.tan x = 3) : 
  Real.tan (x + Real.pi / 3) = -((6 + 5 * Real.sqrt 3) / 13) :=
by
  sorry

end tan_shifted_value_l145_14508


namespace probability_positive_ball_drawn_is_half_l145_14537

-- Definition of the problem elements
def balls : List Int := [-1, 0, 2, 3]

-- Definition for the event of drawing a positive number
def is_positive (x : Int) : Bool := x > 0

-- The proof statement
theorem probability_positive_ball_drawn_is_half : 
  (List.filter is_positive balls).length / balls.length = 1 / 2 :=
by
  sorry

end probability_positive_ball_drawn_is_half_l145_14537


namespace sum_of_digits_of_m_l145_14557

theorem sum_of_digits_of_m (k m : ℕ) : 
  1 ≤ k ∧ k ≤ 3 ∧ 10000 ≤ 11131 * k + 1203 ∧ 11131 * k + 1203 < 100000 ∧ 
  11131 * k + 1203 = m * m ∧ 3 * k < 10 → 
  (m.digits 10).sum = 15 :=
by 
  sorry

end sum_of_digits_of_m_l145_14557


namespace fraction_of_a_eq_1_fifth_of_b_l145_14541

theorem fraction_of_a_eq_1_fifth_of_b (a b : ℝ) (x : ℝ) 
  (h1 : a + b = 100) 
  (h2 : (1/5) * b = 12)
  (h3 : b = 60) : x = 3/10 := by
  sorry

end fraction_of_a_eq_1_fifth_of_b_l145_14541


namespace train_pass_bridge_in_50_seconds_l145_14578

def length_of_train : ℕ := 360
def length_of_bridge : ℕ := 140
def speed_of_train_kmh : ℕ := 36
def total_distance : ℕ := length_of_train + length_of_bridge
def speed_of_train_ms : ℚ := (speed_of_train_kmh * 1000 : ℚ) / 3600 -- we use ℚ to avoid integer division issues
def time_to_pass_bridge : ℚ := total_distance / speed_of_train_ms

theorem train_pass_bridge_in_50_seconds :
  time_to_pass_bridge = 50 := by
  sorry

end train_pass_bridge_in_50_seconds_l145_14578


namespace installation_cost_l145_14598

theorem installation_cost (P I : ℝ) (h₁ : 0.80 * P = 12500)
  (h₂ : 18400 = 1.15 * (12500 + 125 + I)) :
  I = 3375 :=
by
  sorry

end installation_cost_l145_14598


namespace tan_alpha_result_l145_14517

theorem tan_alpha_result (α : ℝ) (h : Real.tan (α - Real.pi / 4) = 1 / 6) : Real.tan α = 7 / 5 :=
by
  sorry

end tan_alpha_result_l145_14517


namespace valid_pic4_valid_pic5_l145_14594

-- Define the type for grid coordinates
structure Coord where
  x : ℕ
  y : ℕ

-- Define the function to check if two coordinates are adjacent by side
def adjacent (a b : Coord) : Prop :=
  (a.x = b.x ∧ (a.y = b.y + 1 ∨ a.y = b.y - 1)) ∨
  (a.y = b.y ∧ (a.x = b.x + 1 ∨ a.x = b.x - 1))

-- Define the coordinates for the pictures №4 and №5
def pic4_coords : List (ℕ × Coord) :=
  [(1, ⟨0, 0⟩), (2, ⟨1, 0⟩), (4, ⟨2, 0⟩), (3, ⟨0, 1⟩),
   (5, ⟨1, 1⟩), (6, ⟨2, 1⟩), (7, ⟨2, 2⟩), (8, ⟨1, 3⟩)]

def pic5_coords : List (ℕ × Coord) :=
  [(1, ⟨0, 0⟩), (2, ⟨0, 1⟩), (3, ⟨0, 2⟩), (4, ⟨0, 3⟩), (5, ⟨1, 3⟩)]

-- Define the validity condition for a picture
def valid_picture (coords : List (ℕ × Coord)) : Prop :=
  ∀ (n : ℕ) (c1 c2 : Coord), (n, c1) ∈ coords → (n + 1, c2) ∈ coords → adjacent c1 c2

-- The theorem to prove that pictures №4 and №5 are valid configurations
theorem valid_pic4 : valid_picture pic4_coords := sorry

theorem valid_pic5 : valid_picture pic5_coords := sorry

end valid_pic4_valid_pic5_l145_14594


namespace chandler_needs_to_sell_more_rolls_l145_14556

/-- Chandler's wrapping paper selling condition. -/
def chandler_needs_to_sell : ℕ := 12

def sold_to_grandmother : ℕ := 3
def sold_to_uncle : ℕ := 4
def sold_to_neighbor : ℕ := 3

def total_sold : ℕ := sold_to_grandmother + sold_to_uncle + sold_to_neighbor

theorem chandler_needs_to_sell_more_rolls : chandler_needs_to_sell - total_sold = 2 :=
by
  sorry

end chandler_needs_to_sell_more_rolls_l145_14556


namespace find_f_a5_a6_l145_14566

-- Define the function properties and initial conditions
variables {f : ℝ → ℝ} {a : ℕ → ℝ} {S : ℕ → ℝ}

-- Conditions for the function f
axiom f_odd : ∀ x, f (-x) = -f x
axiom f_period : ∀ x, f (3/2 - x) = f x
axiom f_minus_2 : f (-2) = -3

-- Initial sequence condition and recursive relation
axiom a_1 : a 1 = -1
axiom S_def : ∀ n, S n = 2 * a n + n
axiom seq_recursive : ∀ n ≥ 2, S (n - 1) = 2 * a (n - 1) + (n - 1)

-- Theorem to prove
theorem find_f_a5_a6 : f (a 5) + f (a 6) = 3 := by
  sorry

end find_f_a5_a6_l145_14566


namespace person_last_name_length_l145_14507

theorem person_last_name_length (samantha_lastname: ℕ) (bobbie_lastname: ℕ) (person_lastname: ℕ) 
  (h1: samantha_lastname + 3 = bobbie_lastname)
  (h2: bobbie_lastname - 2 = 2 * person_lastname)
  (h3: samantha_lastname = 7) :
  person_lastname = 4 :=
by 
  sorry

end person_last_name_length_l145_14507


namespace second_section_area_l145_14588

theorem second_section_area 
  (sod_area_per_square : ℕ := 4)
  (total_squares : ℕ := 1500)
  (first_section_length : ℕ := 30)
  (first_section_width : ℕ := 40)
  (total_area_needed : ℕ := total_squares * sod_area_per_square)
  (first_section_area : ℕ := first_section_length * first_section_width) :
  total_area_needed = first_section_area + 4800 := 
by 
  sorry

end second_section_area_l145_14588


namespace factor_x4_minus_64_l145_14587

theorem factor_x4_minus_64 :
  ∀ (x : ℝ), (x^4 - 64) = (x^2 - 8) * (x^2 + 8) :=
by
  intro x
  sorry

end factor_x4_minus_64_l145_14587


namespace sum_xyz_eq_10_l145_14540

theorem sum_xyz_eq_10 (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x + 2 * x * y + 3 * x * y * z = 115) : 
  x + y + z = 10 :=
sorry

end sum_xyz_eq_10_l145_14540


namespace ellipse_eccentricity_l145_14559

theorem ellipse_eccentricity (a : ℝ) :
  (∀ x y : ℝ, (x^2) / (a^2) + (y^2) / 16 = 1) ∧ (∃ e : ℝ, e = 3 / 4) ∧ (∀ c : ℝ, c = 3 / 4)
   → a = 7 :=
by
  sorry

end ellipse_eccentricity_l145_14559


namespace roger_has_more_candies_l145_14539

def candies_sandra_bag1 : ℕ := 6
def candies_sandra_bag2 : ℕ := 6
def candies_roger_bag1 : ℕ := 11
def candies_roger_bag2 : ℕ := 3

def total_candies_sandra := candies_sandra_bag1 + candies_sandra_bag2
def total_candies_roger := candies_roger_bag1 + candies_roger_bag2

theorem roger_has_more_candies : (total_candies_roger - total_candies_sandra) = 2 := by
  sorry

end roger_has_more_candies_l145_14539


namespace find_k_l145_14551

def f (x : ℝ) := x^2 - 7 * x

theorem find_k : ∃ a h k : ℝ, f x = a * (x - h)^2 + k ∧ k = -49 / 4 := 
sorry

end find_k_l145_14551


namespace determine_k_l145_14502

theorem determine_k (S : ℕ → ℝ) (k : ℝ)
  (hSn : ∀ n, S n = k + 2 * (1 / 3)^n)
  (a1 : ℝ := S 1)
  (a2 : ℝ := S 2 - S 1)
  (a3 : ℝ := S 3 - S 2)
  (geom_property : a2^2 = a1 * a3) :
  k = -2 := 
by
  sorry

end determine_k_l145_14502


namespace natural_number_square_l145_14518

theorem natural_number_square (n : ℕ) : 
  (∃ x : ℕ, n^4 + 4 * n^3 + 5 * n^2 + 6 * n = x^2) ↔ n = 1 := 
by 
  sorry

end natural_number_square_l145_14518


namespace kaleb_books_l145_14545

-- Define the initial number of books
def initial_books : ℕ := 34

-- Define the number of books sold
def books_sold : ℕ := 17

-- Define the number of books bought
def books_bought : ℕ := 7

-- Prove that the final number of books is 24
theorem kaleb_books (h : initial_books - books_sold + books_bought = 24) : initial_books - books_sold + books_bought = 24 :=
by
  exact h

end kaleb_books_l145_14545


namespace problem1_problem2_l145_14514

-- Definitions of M and N
def setM : Set ℝ := {x | -1 ≤ x ∧ x ≤ 5}
def setN (k : ℝ) : Set ℝ := {x | x - k ≤ 0}

-- Problem 1: Prove that if M ∩ N has only one element, then k = -1
theorem problem1 (h : ∀ x, x ∈ setM ∩ setN k → x = -1) : k = -1 := by 
  sorry

-- Problem 2: Given k = 2, prove the sets M ∩ N and M ∪ N
theorem problem2 (hk : k = 2) : (setM ∩ setN k = {x | -1 ≤ x ∧ x ≤ 2}) ∧ (setM ∪ setN k = {x | x ≤ 5}) := by
  sorry

end problem1_problem2_l145_14514


namespace apps_addition_vs_deletion_l145_14505

-- Defining the initial conditions
def initial_apps : ℕ := 21
def added_apps : ℕ := 89
def remaining_apps : ℕ := 24

-- The proof problem statement
theorem apps_addition_vs_deletion :
  added_apps - (initial_apps + added_apps - remaining_apps) = 3 :=
by
  sorry

end apps_addition_vs_deletion_l145_14505


namespace rent_percentage_l145_14554

variable (E : ℝ)

def rent_last_year (E : ℝ) : ℝ := 0.20 * E 
def earnings_this_year (E : ℝ) : ℝ := 1.15 * E
def rent_this_year (E : ℝ) : ℝ := 0.25 * (earnings_this_year E)

-- Prove that the rent this year is 143.75% of the rent last year
theorem rent_percentage : (rent_this_year E) = 1.4375 * (rent_last_year E) :=
by
  sorry

end rent_percentage_l145_14554


namespace domain_of_f_l145_14569

noncomputable def f (x : ℝ) : ℝ :=
  Real.tan (Real.arcsin (x^2))

theorem domain_of_f :
  ∀ x : ℝ, x ∈ Set.Icc (-1 : ℝ) 1 ↔ x ∈ {x : ℝ | f x = f x} :=
by
  sorry

end domain_of_f_l145_14569


namespace find_T_l145_14500

theorem find_T (T : ℝ) (h : (3/4) * (1/8) * T = (1/2) * (1/6) * 72) : T = 64 :=
by {
  -- proof goes here
  sorry
}

end find_T_l145_14500


namespace problem1_problem2_l145_14592

-- Problem 1
theorem problem1 (x : ℝ) : x * (x - 1) - 3 * (x - 1) = 0 → (x = 1) ∨ (x = 3) :=
by sorry

-- Problem 2
theorem problem2 (x : ℝ) : x^2 + 2*x - 1 = 0 → (x = -1 + Real.sqrt 2) ∨ (x = -1 - Real.sqrt 2) :=
by sorry

end problem1_problem2_l145_14592


namespace unique_function_l145_14524

def satisfies_inequality (f : ℝ → ℝ) (k : ℤ) : Prop :=
  ∀ x y z : ℝ, f (x * y) + f (x + z) + k * f x * f (y * z) ≥ k^2

theorem unique_function (k : ℤ) (h : k > 0) :
  ∃! f : ℝ → ℝ, satisfies_inequality f k :=
by
  sorry

end unique_function_l145_14524


namespace lara_flowers_l145_14535

theorem lara_flowers (M : ℕ) : 52 - M - (M + 6) - 16 = 0 → M = 15 :=
by
  sorry

end lara_flowers_l145_14535


namespace area_of_octagon_l145_14577

-- Define the basic geometric elements and properties
variables {A B C D E F G H : Type}
variables (isRectangle : BDEF A B C D E F G H)
variables (AB BC : ℝ) (hAB : AB = 2) (hBC : BC = 2)
variables (isRightIsosceles : ABC A B C D E F G H)

-- Assumptions and known facts
def BDEF_is_rectangle : Prop := isRectangle
def AB_eq_2 : AB = 2 := hAB
def BC_eq_2 : BC = 2 := hBC
def ABC_is_right_isosceles : Prop := isRightIsosceles

-- Statement of the problem to be proved
theorem area_of_octagon : (exists (area : ℝ), area = 8 * Real.sqrt 2) :=
by {
  -- The proof details will go here, which we skip for now
  sorry
}

end area_of_octagon_l145_14577


namespace price_of_basic_computer_l145_14568

-- Conditions
variables (C P : ℝ)
axiom cond1 : C + P = 2500
axiom cond2 : 3 * P = C + 500

-- Prove that the price of the basic computer is $1750
theorem price_of_basic_computer : C = 1750 :=
by 
  sorry

end price_of_basic_computer_l145_14568


namespace band_fundraising_goal_exceed_l145_14597

theorem band_fundraising_goal_exceed
    (goal : ℕ)
    (basic_wash_cost deluxe_wash_cost premium_wash_cost cookie_cost : ℕ)
    (basic_wash_families deluxe_wash_families premium_wash_families sold_cookies : ℕ)
    (total_earnings : ℤ) :
    
    goal = 150 →
    basic_wash_cost = 5 →
    deluxe_wash_cost = 8 →
    premium_wash_cost = 12 →
    cookie_cost = 2 →
    basic_wash_families = 10 →
    deluxe_wash_families = 6 →
    premium_wash_families = 2 →
    sold_cookies = 30 →
    total_earnings = 
        (basic_wash_cost * basic_wash_families +
         deluxe_wash_cost * deluxe_wash_families +
         premium_wash_cost * premium_wash_families +
         cookie_cost * sold_cookies : ℤ) →
    (goal : ℤ) - total_earnings = -32 :=
by
  intros h_goal h_basic h_deluxe h_premium h_cookie h_basic_fam h_deluxe_fam h_premium_fam h_sold_cookies h_total_earnings
  sorry

end band_fundraising_goal_exceed_l145_14597


namespace probability_of_rolling_8_l145_14515

theorem probability_of_rolling_8 :
  let num_favorable := 5
  let num_total := 36
  let probability := (5 : ℚ) / 36
  probability =
    (num_favorable : ℚ) / num_total :=
by
  sorry

end probability_of_rolling_8_l145_14515


namespace total_students_l145_14596

theorem total_students (absent_percent : ℝ) (present_students : ℕ) (total_students : ℝ) :
  absent_percent = 0.14 → present_students = 43 → total_students * (1 - absent_percent) = present_students → total_students = 50 := 
by
  intros
  sorry

end total_students_l145_14596


namespace first_quarter_spending_l145_14542

variables (spent_february_start spent_march_end spent_april_end : ℝ)

-- Given conditions
def begin_february_spent : Prop := spent_february_start = 0.5
def end_march_spent : Prop := spent_march_end = 1.5
def end_april_spent : Prop := spent_april_end = 2.0

-- Proof statement
theorem first_quarter_spending (h1 : begin_february_spent spent_february_start) 
                               (h2 : end_march_spent spent_march_end) 
                               (h3 : end_april_spent spent_april_end) : 
                                spent_march_end - spent_february_start = 1.5 :=
by sorry

end first_quarter_spending_l145_14542


namespace sum_integers_30_to_50_subtract_15_l145_14563

-- Definitions and proof problem based on conditions
def sumIntSeries (a b : ℕ) : ℕ :=
  (b - a + 1) * (a + b) / 2

theorem sum_integers_30_to_50_subtract_15 : sumIntSeries 30 50 - 15 = 825 := by
  -- We are stating that the sum of the integers from 30 to 50 minus 15 is equal to 825
  sorry


end sum_integers_30_to_50_subtract_15_l145_14563


namespace find_q_l145_14595

noncomputable def expr (a b c : ℝ) := a * (b - c)^2 + b * (c - a)^2 + c * (a - b)^2

noncomputable def lhs (a b c : ℝ) := (a - b) * (b - c) * (c - a)

theorem find_q (a b c : ℝ) : expr a b c = lhs a b c * 1 := by
  sorry

end find_q_l145_14595


namespace ben_minimum_test_score_l145_14519

theorem ben_minimum_test_score 
  (scores : List ℕ) 
  (current_avg : ℕ) 
  (desired_increase : ℕ) 
  (lowest_score : ℕ) 
  (required_score : ℕ) 
  (h_scores : scores = [95, 85, 75, 65, 90]) 
  (h_current_avg : current_avg = 82) 
  (h_desired_increase : desired_increase = 5) 
  (h_lowest_score : lowest_score = 65) 
  (h_required_score : required_score = 112) :
  (current_avg + desired_increase) = 87 ∧ 
  (6 * (current_avg + desired_increase)) = 522 ∧ 
  required_score = (522 - (95 + 85 + 75 + 65 + 90)) ∧ 
  (522 - (95 + 85 + 75 + 65 + 90)) > lowest_score :=
by 
  sorry

end ben_minimum_test_score_l145_14519
