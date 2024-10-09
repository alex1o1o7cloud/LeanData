import Mathlib

namespace odd_function_expression_l1508_150866

theorem odd_function_expression (f : ℝ → ℝ) 
  (h_odd : ∀ x, f (-x) = -f x)
  (h_pos : ∀ x, 0 < x → f x = x^2 + |x| - 1) : 
  ∀ x, x < 0 → f x = -x^2 + x + 1 :=
by
  sorry

end odd_function_expression_l1508_150866


namespace bart_firewood_burning_period_l1508_150810

-- We'll state the conditions as definitions.
def pieces_per_tree := 75
def trees_cut_down := 8
def logs_burned_per_day := 5

-- The theorem to prove the period Bart burns the logs.
theorem bart_firewood_burning_period :
  (trees_cut_down * pieces_per_tree) / logs_burned_per_day = 120 :=
by
  sorry

end bart_firewood_burning_period_l1508_150810


namespace calculate_expression_l1508_150814

theorem calculate_expression :
  (-1: ℤ) ^ 53 + 2 ^ (4 ^ 4 + 3 ^ 3 - 5 ^ 2) = -1 + 2 ^ 258 := 
by
  sorry

end calculate_expression_l1508_150814


namespace divisors_count_of_108n5_l1508_150892

theorem divisors_count_of_108n5 {n : ℕ} (hn_pos : 0 < n) (h_divisors_150n3 : (150 * n^3).divisors.card = 150) : 
(108 * n^5).divisors.card = 432 :=
sorry

end divisors_count_of_108n5_l1508_150892


namespace hyperbola_foci_distance_l1508_150877

theorem hyperbola_foci_distance :
  (∃ (h : ℝ → ℝ) (c : ℝ), (∀ x, h x = 2 * x + 3 ∨ h x = 1 - 2 * x)
    ∧ (h 4 = 5)
    ∧ 2 * Real.sqrt (20.25 + 4.444) = 2 * Real.sqrt 24.694) := 
  sorry

end hyperbola_foci_distance_l1508_150877


namespace sum_of_roots_of_quadratic_l1508_150872

variables {b x₁ x₂ : ℝ}

theorem sum_of_roots_of_quadratic (h : x₁^2 - 2 * x₁ + b = 0) (h' : x₂^2 - 2 * x₂ + b = 0) :
    x₁ + x₂ = 2 :=
sorry

end sum_of_roots_of_quadratic_l1508_150872


namespace solution_set_of_f_l1508_150819

theorem solution_set_of_f (f : ℝ → ℝ) (h1 : ∀ x, 2 < deriv f x) (h2 : f (-1) = 2) :
  ∀ x, x > -1 → f x > 2 * x + 4 := by
  sorry

end solution_set_of_f_l1508_150819


namespace x_minus_y_eq_eight_l1508_150833

theorem x_minus_y_eq_eight (x y : ℝ) (hx : 3 = 0.15 * x) (hy : 3 = 0.25 * y) : x - y = 8 :=
by
  sorry

end x_minus_y_eq_eight_l1508_150833


namespace sara_marbles_l1508_150836

theorem sara_marbles : 10 - 7 = 3 :=
by
  sorry

end sara_marbles_l1508_150836


namespace area_of_region_l1508_150897

theorem area_of_region (r : ℝ) (theta_deg : ℝ) (a b c : ℤ) : 
  r = 8 → 
  theta_deg = 45 → 
  (r^2 * theta_deg * Real.pi / 360) - (1/2 * r^2 * Real.sin (theta_deg * Real.pi / 180)) = (a * Real.sqrt b + c * Real.pi) →
  a + b + c = -22 :=
by 
  intros hr htheta Harea 
  sorry

end area_of_region_l1508_150897


namespace negation_universal_to_existential_l1508_150840

-- Setup the necessary conditions and types
variable (a : ℝ) (ha : 0 < a ∧ a < 1)

-- Negate the universal quantifier
theorem negation_universal_to_existential :
  (¬ ∀ x < 0, a^x > 1) ↔ ∃ x_0 < 0, a^(x_0) ≤ 1 :=
by sorry

end negation_universal_to_existential_l1508_150840


namespace calc_value_l1508_150874

theorem calc_value (n : ℕ) (h : 1 ≤ n) : 
  (5^(n+1) + 6^(n+2))^2 - (5^(n+1) - 6^(n+2))^2 = 144 * 30^(n+1) := 
sorry

end calc_value_l1508_150874


namespace sum_r_j_eq_3_l1508_150834

variable (p r j : ℝ)

theorem sum_r_j_eq_3
  (h : (6 * p^2 - 4 * p + r) * (2 * p^2 + j * p - 7) = 12 * p^4 - 34 * p^3 - 19 * p^2 + 28 * p - 21) :
  r + j = 3 := by
  sorry

end sum_r_j_eq_3_l1508_150834


namespace prime_factors_difference_l1508_150838

theorem prime_factors_difference (h : 184437 = 3 * 7 * 8783) : 8783 - 7 = 8776 :=
by sorry

end prime_factors_difference_l1508_150838


namespace range_of_a_l1508_150899

theorem range_of_a (a : ℝ) : (∃ x : ℝ, |x + a| + |x - 1| + a > 2009) ↔ a < 1004 := 
sorry

end range_of_a_l1508_150899


namespace sequence_count_21_l1508_150873

-- Define the conditions and the problem
def valid_sequence (n : ℕ) : ℕ :=
  if n = 21 then 114 else sorry

theorem sequence_count_21 : valid_sequence 21 = 114 :=
  by sorry

end sequence_count_21_l1508_150873


namespace minimum_t_is_2_l1508_150842

noncomputable def minimum_t_value (t : ℝ) : Prop :=
  let A := (-t, 0)
  let B := (t, 0)
  let C := (Real.sqrt 3, Real.sqrt 6)
  let r := 1
  ∃ P : ℝ × ℝ, 
    (P.1 - (Real.sqrt 3))^2 + (P.2 - (Real.sqrt 6))^2 = r^2 ∧ 
    (P.1 - A.1) * (P.1 - B.1) + (P.2 - A.2) * (P.2 - B.2) = 0

theorem minimum_t_is_2 : (∃ t : ℝ, t > 0 ∧ minimum_t_value t) → ∃ t : ℝ, t = 2 :=
sorry

end minimum_t_is_2_l1508_150842


namespace length_of_train_is_250_02_l1508_150860

noncomputable def train_speed_km_per_hr : ℝ := 100
noncomputable def time_to_cross_pole_sec : ℝ := 9

-- Convert speed from km/hr to m/s
noncomputable def speed_m_per_s : ℝ := train_speed_km_per_hr * (1000 / 3600)

-- Calculating the length of the train
noncomputable def length_of_train : ℝ := speed_m_per_s * time_to_cross_pole_sec

theorem length_of_train_is_250_02 :
  length_of_train = 250.02 := by
  -- Proof is omitted (replace 'sorry' with the actual proof)
  sorry

end length_of_train_is_250_02_l1508_150860


namespace candidates_appeared_l1508_150816

-- Define the conditions:
variables (A_selected B_selected : ℕ) (x : ℝ)

-- 12% candidates got selected in State A
def State_A_selected := 0.12 * x

-- 18% candidates got selected in State B
def State_B_selected := 0.18 * x

-- 250 more candidates got selected in State B than in State A
def selection_difference := State_B_selected = State_A_selected + 250

-- The statement to prove:
theorem candidates_appeared (h : selection_difference) : x = 4167 :=
by
  sorry

end candidates_appeared_l1508_150816


namespace find_range_of_m_l1508_150843

variable (m : ℝ)

-- Definition of p: There exists x in ℝ such that mx^2 - mx + 1 < 0
def p : Prop := ∃ x : ℝ, m * x ^ 2 - m * x + 1 < 0

-- Definition of q: The curve of the equation (x^2)/(m-1) + (y^2)/(3-m) = 1 is a hyperbola
def q : Prop := (m - 1) * (3 - m) < 0

-- Given conditions
def proposition_and : Prop := ¬ (p m ∧ q m)
def proposition_or : Prop := p m ∨ q m

-- Final theorem statement
theorem find_range_of_m : proposition_and m ∧ proposition_or m → (0 < m ∧ m ≤ 1) ∨ (3 ≤ m ∧ m < 4) :=
sorry

end find_range_of_m_l1508_150843


namespace find_ordered_pair_l1508_150898

theorem find_ordered_pair (s l : ℝ) :
  (∀ (x y : ℝ), (∃ t : ℝ, (x, y) = (-8 + t * l, s - 7 * t)) ↔ y = 2 * x - 3) →
  (s = -19 ∧ l = -7 / 2) :=
by
  intro h
  have : (∀ (x y : ℝ), (∃ t : ℝ, (x, y) = (-8 + t * l, s - 7 * t)) ↔ y = 2 * x - 3) := h
  sorry

end find_ordered_pair_l1508_150898


namespace sum_of_roots_l1508_150804

theorem sum_of_roots (x : ℝ) (h : (x - 6)^2 = 16) : (∃ a b : ℝ, a + b = 12 ∧ (x = a ∨ x = b)) :=
by
  sorry

end sum_of_roots_l1508_150804


namespace area_enclosed_by_3x2_l1508_150839

theorem area_enclosed_by_3x2 (a b : ℝ) (h₀ : a = 0) (h₁ : b = 1) :
  ∫ (x : ℝ) in a..b, 3 * x^2 = 1 :=
by 
  rw [h₀, h₁]
  sorry

end area_enclosed_by_3x2_l1508_150839


namespace convert_denominators_to_integers_l1508_150870

def original_equation (x : ℝ) : Prop :=
  (x + 1) / 0.4 - (0.2 * x - 1) / 0.7 = 1

def transformed_equation (x : ℝ) : Prop :=
  (10 * x + 10) / 4 - (2 * x - 10) / 7 = 1

theorem convert_denominators_to_integers (x : ℝ) 
  (h : original_equation x) : transformed_equation x :=
sorry

end convert_denominators_to_integers_l1508_150870


namespace simplify_fraction_l1508_150801

theorem simplify_fraction (m : ℝ) (h : m ≠ 1) : (m / (m - 1) + 1 / (1 - m) = 1) :=
by {
  sorry
}

end simplify_fraction_l1508_150801


namespace ratio_of_q_to_p_l1508_150846

theorem ratio_of_q_to_p (p q : ℝ) (h₀ : 0 < p) (h₁ : 0 < q) 
  (h₂ : Real.log p / Real.log 9 = Real.log q / Real.log 12) 
  (h₃ : Real.log q / Real.log 12 = Real.log (p + q) / Real.log 16) : 
  q / p = (1 + Real.sqrt 5) / 2 := 
by 
  sorry

end ratio_of_q_to_p_l1508_150846


namespace inequality_proof_l1508_150868

theorem inequality_proof (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
    (h4 : (3 / (a * b * c)) ≥ (a + b + c)) : 
    (1 / a + 1 / b + 1 / c) ≥ (a + b + c) :=
  sorry

end inequality_proof_l1508_150868


namespace at_least_one_travels_l1508_150864

-- Define the probabilities of A and B traveling
def P_A := 1 / 3
def P_B := 1 / 4

-- Define the probability that person A does not travel
def P_not_A := 1 - P_A

-- Define the probability that person B does not travel
def P_not_B := 1 - P_B

-- Define the probability that neither person A nor person B travels
def P_neither := P_not_A * P_not_B

-- Define the probability that at least one of them travels
def P_at_least_one := 1 - P_neither

theorem at_least_one_travels : P_at_least_one = 1 / 2 := by
  sorry

end at_least_one_travels_l1508_150864


namespace find_height_l1508_150800

-- Defining the known conditions
def length : ℝ := 3
def width : ℝ := 5
def cost_per_sqft : ℝ := 20
def total_cost : ℝ := 1240

-- Defining the unknown dimension as a variable
variable (height : ℝ)

-- Surface area formula for a rectangular tank
def surface_area (l w h : ℝ) : ℝ := 2 * l * w + 2 * l * h + 2 * w * h

-- Given statement to prove that the height is 2 feet.
theorem find_height : surface_area length width height = total_cost / cost_per_sqft → height = 2 := by
  sorry

end find_height_l1508_150800


namespace largest_multiple_of_8_less_than_100_l1508_150823

theorem largest_multiple_of_8_less_than_100 :
  ∃ n : ℕ, n < 100 ∧ (∃ k : ℕ, n = 8 * k) ∧ ∀ m : ℕ, m < 100 ∧ (∃ k : ℕ, m = 8 * k) → m ≤ 96 := 
by
  sorry

end largest_multiple_of_8_less_than_100_l1508_150823


namespace find_solution_l1508_150871

theorem find_solution (x y : ℕ) (h1 : y ∣ (x^2 + 1)) (h2 : x^2 ∣ (y^3 + 1)) : (x = 1 ∧ y = 1) :=
sorry

end find_solution_l1508_150871


namespace number_of_points_determined_l1508_150830

def A : Set ℕ := {5}
def B : Set ℕ := {1, 2}
def C : Set ℕ := {1, 3, 4}

theorem number_of_points_determined : (∃ n : ℕ, n = 33) :=
by
  -- sorry to skip the proof
  sorry

end number_of_points_determined_l1508_150830


namespace bobby_paid_for_shoes_l1508_150827

theorem bobby_paid_for_shoes :
  let mold_cost := 250
  let hourly_labor_rate := 75
  let hours_worked := 8
  let discount_rate := 0.80
  let materials_cost := 150
  let tax_rate := 0.10

  let labor_cost := hourly_labor_rate * hours_worked
  let discounted_labor_cost := discount_rate * labor_cost
  let total_cost_before_tax := mold_cost + discounted_labor_cost + materials_cost
  let tax := total_cost_before_tax * tax_rate
  let total_cost_with_tax := total_cost_before_tax + tax

  total_cost_with_tax = 968 :=
by
  sorry

end bobby_paid_for_shoes_l1508_150827


namespace find_some_value_l1508_150845

-- Define the main variables and assumptions
variable (m n some_value : ℝ)

-- State the assumptions based on the conditions
axiom h1 : m = n / 2 - 2 / 5
axiom h2 : m + some_value = (n + 4) / 2 - 2 / 5

-- State the theorem we are trying to prove
theorem find_some_value : some_value = 2 :=
by
  -- Proof goes here, for now we just put sorry
  sorry

end find_some_value_l1508_150845


namespace lake_view_population_l1508_150807

-- Define the populations of the cities
def population_of_Seattle : ℕ := 20000 -- Derived from the solution
def population_of_Boise : ℕ := (3 / 5) * population_of_Seattle
def population_of_Lake_View : ℕ := population_of_Seattle + 4000
def total_population : ℕ := population_of_Seattle + population_of_Boise + population_of_Lake_View

-- Statement to prove
theorem lake_view_population :
  total_population = 56000 →
  population_of_Lake_View = 24000 :=
sorry

end lake_view_population_l1508_150807


namespace probability_enemy_plane_hit_l1508_150818

noncomputable def P_A : ℝ := 0.6
noncomputable def P_B : ℝ := 0.4

theorem probability_enemy_plane_hit : 1 - ((1 - P_A) * (1 - P_B)) = 0.76 :=
by
  sorry

end probability_enemy_plane_hit_l1508_150818


namespace compute_area_ratio_l1508_150865

noncomputable def area_ratio (K : ℝ) : ℝ :=
  let small_triangle_area := 2 * K
  let large_triangle_area := 8 * K
  small_triangle_area / large_triangle_area

theorem compute_area_ratio (K : ℝ) : area_ratio K = 1 / 4 :=
by
  unfold area_ratio
  sorry

end compute_area_ratio_l1508_150865


namespace number_of_cats_l1508_150832

def number_of_dogs : ℕ := 43
def number_of_fish : ℕ := 72
def total_pets : ℕ := 149

theorem number_of_cats : total_pets - (number_of_dogs + number_of_fish) = 34 := 
by
  sorry

end number_of_cats_l1508_150832


namespace adam_final_score_l1508_150844

theorem adam_final_score : 
  let science_correct := 5
  let science_points := 10
  let history_correct := 3
  let history_points := 5
  let history_multiplier := 2
  let sports_correct := 1
  let sports_points := 15
  let literature_correct := 1
  let literature_points := 7
  let literature_penalty := 3
  
  let science_total := science_correct * science_points
  let history_total := (history_correct * history_points) * history_multiplier
  let sports_total := sports_correct * sports_points
  let literature_total := (literature_correct * literature_points) - literature_penalty
  
  let final_score := science_total + history_total + sports_total + literature_total
  final_score = 99 := by 
    sorry

end adam_final_score_l1508_150844


namespace jeff_stars_l1508_150887

noncomputable def eric_stars : ℕ := 4
noncomputable def chad_initial_stars : ℕ := 2 * eric_stars
noncomputable def chad_stars_after_sale : ℕ := chad_initial_stars - 2
noncomputable def total_stars : ℕ := 16
noncomputable def stars_eric_and_chad : ℕ := eric_stars + chad_stars_after_sale

theorem jeff_stars :
  total_stars - stars_eric_and_chad = 6 := 
by 
  sorry

end jeff_stars_l1508_150887


namespace count_congruent_to_3_mod_7_lt_500_l1508_150893

theorem count_congruent_to_3_mod_7_lt_500 : 
  ∃ n, n = 71 ∧ ∀ x, 0 < x ∧ x < 500 ∧ x % 7 = 3 ↔ ∃ k, 0 ≤ k ∧ k ≤ 70 ∧ x = 3 + 7 * k :=
sorry

end count_congruent_to_3_mod_7_lt_500_l1508_150893


namespace smallest_int_solution_l1508_150837

theorem smallest_int_solution : ∃ y : ℤ, y = 6 ∧ ∀ z : ℤ, z > 5 → y ≤ z := sorry

end smallest_int_solution_l1508_150837


namespace valid_votes_election_l1508_150886

-- Definition of the problem
variables (V : ℝ) -- the total number of valid votes
variables (hvoting_percentage : V > 0 ∧ V ≤ 1) -- constraints for voting percentage in general
variables (h_winning_votes : 0.70 * V) -- 70% of the votes
variables (h_losing_votes : 0.30 * V) -- 30% of the votes

-- Given condition: the winning candidate won by a majority of 184 votes
variables (majority : ℝ) (h_majority : 0.70 * V - 0.30 * V = 184)

/-- The total number of valid votes in the election. -/
theorem valid_votes_election : V = 460 :=
by
  sorry

end valid_votes_election_l1508_150886


namespace quadratic_intersects_x_axis_if_and_only_if_k_le_four_l1508_150850

-- Define the quadratic function
def quadratic_function (k x : ℝ) : ℝ :=
  (k - 3) * x^2 + 2 * x + 1

-- Theorem stating the relationship between the function intersecting the x-axis and k ≤ 4
theorem quadratic_intersects_x_axis_if_and_only_if_k_le_four
  (k : ℝ) :
  (∃ x : ℝ, quadratic_function k x = 0) ↔ k ≤ 4 :=
sorry

end quadratic_intersects_x_axis_if_and_only_if_k_le_four_l1508_150850


namespace calvin_winning_strategy_l1508_150803

theorem calvin_winning_strategy :
  ∃ (n : ℤ), ∃ (p : ℤ), ∃ (q : ℤ),
  (∀ k : ℕ, k > 0 → p = 0 ∧ (q = 2014 + k ∨ q = 2014 - k) → ∃ x : ℤ, (x^2 + p * x + q = 0)) :=
sorry

end calvin_winning_strategy_l1508_150803


namespace choose_roles_from_8_l1508_150883

-- Define the number of people
def num_people : ℕ := 8
-- Define the function to count the number of ways to choose different persons for the roles
def choose_roles (n : ℕ) : ℕ := n * (n - 1) * (n - 2)

theorem choose_roles_from_8 : choose_roles num_people = 336 := by
  -- sorry acts as a placeholder for the proof
  sorry

end choose_roles_from_8_l1508_150883


namespace george_older_than_christopher_l1508_150813

theorem george_older_than_christopher
  (G C F : ℕ)
  (h1 : C = 18)
  (h2 : F = C - 2)
  (h3 : G + C + F = 60) :
  G - C = 8 := by
  sorry

end george_older_than_christopher_l1508_150813


namespace household_A_bill_bill_formula_household_B_usage_household_C_usage_l1508_150863

-- Definition of the tiered water price system
def water_bill (x : ℕ) : ℕ :=
if x <= 22 then 3 * x
else if x <= 30 then 3 * 22 + 5 * (x - 22)
else 3 * 22 + 5 * 8 + 7 * (x - 30)

-- Prove that if a household uses 25m^3 of water, the water bill is 81 yuan.
theorem household_A_bill : water_bill 25 = 81 := by 
  sorry

-- Prove that the formula for the water bill when x > 30 is y = 7x - 104.
theorem bill_formula (x : ℕ) (hx : x > 30) : water_bill x = 7 * x - 104 := by 
  sorry

-- Prove that if a household paid 120 yuan for water, their usage was 32m^3.
theorem household_B_usage : ∃ x : ℕ, water_bill x = 120 ∧ x = 32 := by 
  sorry

-- Prove that if household C uses a total of 50m^3 over May and June with a total bill of 174 yuan, their usage was 18m^3 in May and 32m^3 in June.
theorem household_C_usage (a b : ℕ) (ha : a + b = 50) (hb : a < b) (total_bill : water_bill a + water_bill b = 174) :
  a = 18 ∧ b = 32 := by
  sorry

end household_A_bill_bill_formula_household_B_usage_household_C_usage_l1508_150863


namespace lisa_total_spoons_l1508_150861

def children_count : ℕ := 6
def spoons_per_child : ℕ := 4
def decorative_spoons : ℕ := 4
def large_spoons : ℕ := 20
def dessert_spoons : ℕ := 10
def soup_spoons : ℕ := 15
def tea_spoons : ℕ := 25

def baby_spoons_total : ℕ := children_count * spoons_per_child
def cutlery_set_total : ℕ := large_spoons + dessert_spoons + soup_spoons + tea_spoons

def total_spoons : ℕ := cutlery_set_total + baby_spoons_total + decorative_spoons

theorem lisa_total_spoons : total_spoons = 98 :=
by
  sorry

end lisa_total_spoons_l1508_150861


namespace hyperbola_hkabc_sum_l1508_150820

theorem hyperbola_hkabc_sum :
  ∃ h k a b : ℝ, h = 3 ∧ k = -1 ∧ a = 2 ∧ b = Real.sqrt 46 ∧ h + k + a + b = 4 + Real.sqrt 46 :=
by
  use 3
  use -1
  use 2
  use Real.sqrt 46
  simp
  sorry

end hyperbola_hkabc_sum_l1508_150820


namespace find_a_plus_b_l1508_150809

theorem find_a_plus_b :
  ∃ (a b : ℝ), (∀ x : ℝ, (3 * (a * x + b) - 6) = 4 * x + 5) ∧ a + b = 5 :=
by 
  sorry

end find_a_plus_b_l1508_150809


namespace solution_to_problem_l1508_150858

def number_exists (n : ℝ) : Prop :=
  n / 0.25 = 400

theorem solution_to_problem : ∃ n : ℝ, number_exists n ∧ n = 100 := by
  sorry

end solution_to_problem_l1508_150858


namespace find_original_fraction_l1508_150856

theorem find_original_fraction (x y : ℚ) (h : (1.15 * x) / (0.92 * y) = 15 / 16) :
  x / y = 69 / 92 :=
sorry

end find_original_fraction_l1508_150856


namespace probability_of_rolling_prime_is_half_l1508_150821

def is_prime (n : ℕ) : Prop :=
  n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7

def total_outcomes : ℕ := 8

def successful_outcomes : ℕ := 4 -- prime numbers between 1 and 8 are 2, 3, 5, and 7

def probability_of_rolling_prime : ℚ :=
  successful_outcomes / total_outcomes

theorem probability_of_rolling_prime_is_half : probability_of_rolling_prime = 1 / 2 :=
  sorry

end probability_of_rolling_prime_is_half_l1508_150821


namespace cosine_value_parallel_vectors_l1508_150841

theorem cosine_value_parallel_vectors (α : ℝ) (h1 : ∃ (a : ℝ × ℝ) (b : ℝ × ℝ), a = (Real.cos (Real.pi / 3 + α), 1) ∧ b = (1, 4) ∧ a.1 * b.2 - a.2 * b.1 = 0) : 
  Real.cos (Real.pi / 3 - 2 * α) = 7 / 8 := by
  sorry

end cosine_value_parallel_vectors_l1508_150841


namespace percentage_markup_l1508_150806

open Real

theorem percentage_markup (SP CP : ℝ) (hSP : SP = 5600) (hCP : CP = 4480) : 
  ((SP - CP) / CP) * 100 = 25 :=
by
  sorry

end percentage_markup_l1508_150806


namespace inequality_proof_l1508_150854

variables (a b c : ℝ)

theorem inequality_proof (h : a > b) : a * c^2 ≥ b * c^2 :=
by sorry

end inequality_proof_l1508_150854


namespace sqrt_sum_eq_l1508_150859

theorem sqrt_sum_eq : 
  (Real.sqrt (16 - 12 * Real.sqrt 3)) + (Real.sqrt (16 + 12 * Real.sqrt 3)) = 4 * Real.sqrt 6 :=
by
  sorry

end sqrt_sum_eq_l1508_150859


namespace dried_fruit_percentage_l1508_150891

-- Define the percentages for Sue, Jane, and Tom's trail mixes.
structure TrailMix :=
  (nuts : ℝ)
  (dried_fruit : ℝ)

def sue : TrailMix := { nuts := 0.30, dried_fruit := 0.70 }
def jane : TrailMix := { nuts := 0.60, dried_fruit := 0.00 }  -- Note: No dried fruit
def tom : TrailMix := { nuts := 0.40, dried_fruit := 0.50 }

-- Condition: Combined mix contains 45% nuts.
def combined_nuts (sue_nuts jane_nuts tom_nuts : ℝ) : Prop :=
  0.33 * sue_nuts + 0.33 * jane_nuts + 0.33 * tom_nuts = 0.45

-- Condition: Each contributes equally to the total mixture.
def equal_contribution (sue_cont jane_cont tom_cont : ℝ) : Prop :=
  sue_cont = jane_cont ∧ jane_cont = tom_cont

-- Theorem to be proven: Combined mixture contains 40% dried fruit.
theorem dried_fruit_percentage :
  combined_nuts sue.nuts jane.nuts tom.nuts →
  equal_contribution (1 / 3) (1 / 3) (1 / 3) →
  0.33 * sue.dried_fruit + 0.33 * tom.dried_fruit = 0.40 :=
by sorry

end dried_fruit_percentage_l1508_150891


namespace exists_root_in_interval_l1508_150848

noncomputable def f (x : ℝ) : ℝ := 1 / (x - 1) - Real.log (x - 1) / Real.log 2

theorem exists_root_in_interval :
  ∃ c : ℝ, 2 < c ∧ c < 3 ∧ f c = 0 :=
by
  -- Proof goes here
  sorry

end exists_root_in_interval_l1508_150848


namespace sweet_potatoes_not_yet_sold_l1508_150881

def total_harvested := 80
def sold_to_adams := 20
def sold_to_lenon := 15
def not_yet_sold : ℕ := total_harvested - (sold_to_adams + sold_to_lenon)

theorem sweet_potatoes_not_yet_sold :
  not_yet_sold = 45 :=
by
  unfold not_yet_sold
  unfold total_harvested sold_to_adams sold_to_lenon
  sorry

end sweet_potatoes_not_yet_sold_l1508_150881


namespace product_of_two_numbers_l1508_150829

variable {x y : ℝ}

theorem product_of_two_numbers (h1 : x + y = 25) (h2 : x - y = 7) : x * y = 144 := by
  sorry

end product_of_two_numbers_l1508_150829


namespace opposite_of_neg_2023_l1508_150805

def opposite (n : ℤ) : ℤ :=
  -n

theorem opposite_of_neg_2023 : opposite (-2023) = 2023 :=
by
  sorry

end opposite_of_neg_2023_l1508_150805


namespace complete_the_square_l1508_150847

theorem complete_the_square (x : ℝ) :
  x^2 + 6 * x - 4 = 0 → (x + 3)^2 = 13 :=
by
  sorry

end complete_the_square_l1508_150847


namespace point_A_is_minus_five_l1508_150885

theorem point_A_is_minus_five 
  (A B C : ℝ)
  (h1 : A + 4 = B)
  (h2 : B - 2 = C)
  (h3 : C = -3) : 
  A = -5 := 
by 
  sorry

end point_A_is_minus_five_l1508_150885


namespace bookmarks_sold_l1508_150825

-- Definitions pertaining to the problem
def total_books_sold : ℕ := 72
def books_ratio : ℕ := 9
def bookmarks_ratio : ℕ := 2

-- Statement of the theorem
theorem bookmarks_sold :
  (total_books_sold / books_ratio) * bookmarks_ratio = 16 :=
by
  sorry

end bookmarks_sold_l1508_150825


namespace common_root_divisibility_l1508_150884

variables (a b c : ℤ)

theorem common_root_divisibility 
  (h1 : c ≠ b) 
  (h2 : ∃ x : ℝ, a * x^2 + b * x + c = 0 ∧ (c - b) * x^2 + (c - a) * x + (a + b) = 0) 
  : 3 ∣ (a + b + 2 * c) :=
sorry

end common_root_divisibility_l1508_150884


namespace number_of_triplets_with_sum_6n_l1508_150835

theorem number_of_triplets_with_sum_6n (n : ℕ) : 
  ∃ (count : ℕ), count = 3 * n^2 ∧ 
  (∀ (x y z : ℕ), x ≤ y → y ≤ z → x + y + z = 6 * n → count = 1) :=
sorry

end number_of_triplets_with_sum_6n_l1508_150835


namespace problem_statement_l1508_150875

-- The conditions of the problem
variables (x : Real)

-- Define the conditions as hypotheses
def condition1 : Prop := (Real.sin (3 * x) * Real.sin (4 * x)) = (Real.cos (3 * x) * Real.cos (4 * x))
def condition2 : Prop := Real.sin (7 * x) = 0

-- The theorem we need to prove
theorem problem_statement (h1 : condition1 x) (h2 : condition2 x) : x = Real.pi / 7 :=
by sorry

end problem_statement_l1508_150875


namespace max_cart_length_l1508_150876

-- Definitions for the hallway and cart dimensions
def hallway_width : ℝ := 1.5
def cart_width : ℝ := 1

-- The proposition stating the maximum length of the cart that can smoothly navigate the hallway
theorem max_cart_length : ∃ L : ℝ, L = 3 * Real.sqrt 2 ∧
  (∀ (a b : ℝ), a > 0 ∧ b > 0 → (3 / a) + (3 / b) = 2 → Real.sqrt (a^2 + b^2) = L) :=
  sorry

end max_cart_length_l1508_150876


namespace incorrect_option_D_l1508_150824

-- Definitions based on the given conditions:
def contrapositive_correct : Prop :=
  ∀ x : ℝ, (x ≠ 1 → x^2 - 3 * x + 2 ≠ 0) ↔ (x^2 - 3 * x + 2 = 0 → x = 1)

def sufficient_but_not_necessary : Prop :=
  ∀ x : ℝ, (x > 2 → x^2 - 3 * x + 2 > 0) ∧ (x^2 - 3 * x + 2 > 0 → x > 2 ∨ x < 1)

def negation_correct (p : Prop) (neg_p : Prop) : Prop :=
  p ↔ ∀ x : ℝ, x^2 + x + 1 ≠ 0 ∧ neg_p ↔ ∃ x_0 : ℝ, x_0^2 + x_0 + 1 = 0

theorem incorrect_option_D (p q : Prop) (h : p ∨ q) :
  ¬ (p ∧ q) :=
sorry  -- Proof is to be done later

end incorrect_option_D_l1508_150824


namespace find_b_l1508_150852

theorem find_b (x y b : ℝ) (h1 : (7 * x + b * y) / (x - 2 * y) = 13) (h2 : x / (2 * y) = 5 / 2) : b = 4 :=
  sorry

end find_b_l1508_150852


namespace average_speed_30_l1508_150815

theorem average_speed_30 (v : ℝ) (h₁ : 0 < v) (h₂ : 210 / v - 1 = 210 / (v + 5)) : v = 30 :=
sorry

end average_speed_30_l1508_150815


namespace find_m_l1508_150878

theorem find_m (m : ℝ) :
  let a : ℝ × ℝ := (2, m)
  let b : ℝ × ℝ := (1, -1)
  (b.1 * (a.1 + 2 * b.1) + b.2 * (a.2 + 2 * b.2) = 0) → 
  m = 6 := by 
  sorry

end find_m_l1508_150878


namespace find_n_l1508_150851

def num_of_trailing_zeros (n : ℕ) : ℕ :=
  if n = 0 then 0 else (n / 5) + num_of_trailing_zeros (n / 5)

theorem find_n (n : ℕ) (k : ℕ) (h1 : n > 3) (h2 : k = num_of_trailing_zeros n) (h3 : 2*k + 1 = num_of_trailing_zeros (2*n)) (h4 : k > 0) : n = 6 :=
by
  sorry

end find_n_l1508_150851


namespace bones_received_on_sunday_l1508_150812

-- Definitions based on the conditions
def initial_bones : ℕ := 50
def bones_eaten : ℕ := initial_bones / 2
def bones_left_after_saturday : ℕ := initial_bones - bones_eaten
def total_bones_after_sunday : ℕ := 35

-- The theorem to prove how many bones received on Sunday
theorem bones_received_on_sunday : 
  (total_bones_after_sunday - bones_left_after_saturday = 10) :=
by
  -- proof will be filled in here
  sorry

end bones_received_on_sunday_l1508_150812


namespace y_work_days_eq_10_l1508_150882

noncomputable def work_days_y (W d : ℝ) : Prop :=
  let work_rate_x := W / 30
  let work_rate_y := W / 15
  let days_x_remaining := 10.000000000000002
  let work_done_by_y := d * work_rate_y
  let work_done_by_x := days_x_remaining * work_rate_x
  work_done_by_y + work_done_by_x = W

/-- The number of days y worked before leaving the job is 10 -/
theorem y_work_days_eq_10 (W : ℝ) : work_days_y W 10 :=
by
  sorry

end y_work_days_eq_10_l1508_150882


namespace find_multiple_of_benjy_peaches_l1508_150890

theorem find_multiple_of_benjy_peaches
(martine_peaches gabrielle_peaches : ℕ)
(benjy_peaches : ℕ)
(m : ℕ)
(h1 : martine_peaches = 16)
(h2 : gabrielle_peaches = 15)
(h3 : benjy_peaches = gabrielle_peaches / 3)
(h4 : martine_peaches = m * benjy_peaches + 6) :
m = 2 := by
sorry

end find_multiple_of_benjy_peaches_l1508_150890


namespace fred_bought_books_l1508_150822

theorem fred_bought_books (initial_money : ℕ) (remaining_money : ℕ) (book_cost : ℕ)
  (h1 : initial_money = 236)
  (h2 : remaining_money = 14)
  (h3 : book_cost = 37) :
  (initial_money - remaining_money) / book_cost = 6 :=
by {
  sorry
}

end fred_bought_books_l1508_150822


namespace cube_volume_from_surface_area_l1508_150855

theorem cube_volume_from_surface_area (SA : ℕ) (h : SA = 600) :
  ∃ V : ℕ, V = 1000 := by
  sorry

end cube_volume_from_surface_area_l1508_150855


namespace determine_z_l1508_150889

theorem determine_z (z : ℕ) (h1: z.factors.count = 18) (h2: 16 ∣ z) (h3: 18 ∣ z) : z = 288 := 
  by 
  sorry

end determine_z_l1508_150889


namespace y_coordinate_equidistant_l1508_150895

theorem y_coordinate_equidistant :
  ∃ y : ℝ, (∀ P : ℝ × ℝ, P = (0, y) → dist (3, 0) P = dist (2, 5) P) ∧ y = 2 := 
by
  sorry

end y_coordinate_equidistant_l1508_150895


namespace max_additional_plates_l1508_150853

def initial_plates_count : ℕ := 5 * 3 * 4 * 2
def new_second_set_size : ℕ := 5  -- second set after adding two letters
def new_fourth_set_size : ℕ := 3 -- fourth set after adding one letter
def new_plates_count : ℕ := 5 * new_second_set_size * 4 * new_fourth_set_size

theorem max_additional_plates :
  new_plates_count - initial_plates_count = 180 := by
  sorry

end max_additional_plates_l1508_150853


namespace fifth_equation_pattern_l1508_150896

theorem fifth_equation_pattern :
  (1 = 1) →
  (2 + 3 + 4 = 9) →
  (3 + 4 + 5 + 6 + 7 = 25) →
  (4 + 5 + 6 + 7 + 8 + 9 + 10 = 49) →
  (5 + 6 + 7 + 8 + 9 + 10 + 11 + 12 + 13 = 81) :=
by 
  intros h1 h2 h3 h4
  sorry

end fifth_equation_pattern_l1508_150896


namespace diagonals_in_octadecagon_l1508_150817

def num_sides : ℕ := 18

def num_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

theorem diagonals_in_octadecagon : num_diagonals num_sides = 135 := by 
  sorry

end diagonals_in_octadecagon_l1508_150817


namespace player_avg_increase_l1508_150894

theorem player_avg_increase
  (matches_played : ℕ)
  (initial_avg : ℕ)
  (next_match_runs : ℕ)
  (total_runs : ℕ)
  (new_total_runs : ℕ)
  (new_avg : ℕ)
  (desired_avg_increase : ℕ) :
  matches_played = 10 ∧ initial_avg = 32 ∧ next_match_runs = 76 ∧ total_runs = 320 ∧ 
  new_total_runs = 396 ∧ new_avg = 32 + desired_avg_increase ∧ 
  11 * new_avg = new_total_runs → desired_avg_increase = 4 := 
by
  sorry

end player_avg_increase_l1508_150894


namespace percentage_increase_in_expenses_l1508_150862

-- Define the variables and conditions
def monthly_salary : ℝ := 7272.727272727273
def original_savings_percentage : ℝ := 0.10
def new_savings : ℝ := 400
def original_savings : ℝ := original_savings_percentage * monthly_salary
def savings_difference : ℝ := original_savings - new_savings
def original_expenses : ℝ := (1 - original_savings_percentage) * monthly_salary

-- Formalize the question as a theorem
theorem percentage_increase_in_expenses (P : ℝ) :
  P = (savings_difference / original_expenses) * 100 ↔ P = 5 := 
sorry

end percentage_increase_in_expenses_l1508_150862


namespace exponent_multiplication_l1508_150857

variable (x : ℤ)

theorem exponent_multiplication :
  (-x^2) * x^3 = -x^5 :=
sorry

end exponent_multiplication_l1508_150857


namespace dad_strawberries_now_weight_l1508_150826

-- Definitions based on the conditions given
def total_weight : ℕ := 36
def weight_lost_by_dad : ℕ := 8
def weight_of_marco_strawberries : ℕ := 12

-- Theorem to prove the question as an equality
theorem dad_strawberries_now_weight :
  total_weight - weight_lost_by_dad - weight_of_marco_strawberries = 16 := by
  sorry

end dad_strawberries_now_weight_l1508_150826


namespace tetrahedron_volume_distance_relation_l1508_150808

theorem tetrahedron_volume_distance_relation
  (V : ℝ) (S1 S2 S3 S4 : ℝ) (H1 H2 H3 H4 : ℝ)
  (k : ℝ)
  (hS : (S1 / 1) = k) (hS2 : (S2 / 2) = k) (hS3 : (S3 / 3) = k) (hS4 : (S4 / 4) = k) :
  H1 + 2 * H2 + 3 * H3 + 4 * H4 = 3 * V / k :=
sorry

end tetrahedron_volume_distance_relation_l1508_150808


namespace absolute_difference_m_n_l1508_150867

theorem absolute_difference_m_n (m n : ℝ) (h1 : m * n = 6) (h2 : m + n = 7) : |m - n| = 5 := 
by 
  sorry

end absolute_difference_m_n_l1508_150867


namespace hyperbola_center_l1508_150888

theorem hyperbola_center (x y : ℝ) :
  9 * x ^ 2 - 54 * x - 36 * y ^ 2 + 360 * y - 900 = 0 →
  (x, y) = (3, 5) :=
sorry

end hyperbola_center_l1508_150888


namespace probability_A2_equals_zero_matrix_l1508_150869

noncomputable def probability_A2_zero (n : ℕ) (hn : n ≥ 2) : ℚ :=
  let numerator := (n - 1) * (n - 2)
  let denominator := n * (n - 1)
  numerator / denominator

theorem probability_A2_equals_zero_matrix (n : ℕ) (hn : n ≥ 2) :
  probability_A2_zero n hn = ((n - 1) * (n - 2) / (n * (n - 1))) := by
  sorry

end probability_A2_equals_zero_matrix_l1508_150869


namespace decompose_375_l1508_150802

theorem decompose_375 : 375 = 3 * 100 + 7 * 10 + 5 * 1 :=
by
  sorry

end decompose_375_l1508_150802


namespace find_k_values_l1508_150879

noncomputable def problem (a b c d k : ℂ) : Prop :=
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧
  (a * k^3 + b * k^2 + c * k + d = 0) ∧
  (b * k^3 + c * k^2 + d * k + a = 0)

theorem find_k_values (a b c d k : ℂ) (h : problem a b c d k) : 
  k = 1 ∨ k = -1 ∨ k = Complex.I ∨ k = -Complex.I :=
sorry

end find_k_values_l1508_150879


namespace geometry_problem_l1508_150811

theorem geometry_problem
  (A_square : ℝ)
  (A_rectangle : ℝ)
  (A_triangle : ℝ)
  (side_length : ℝ)
  (rectangle_width : ℝ)
  (rectangle_length : ℝ)
  (triangle_base : ℝ)
  (triangle_height : ℝ)
  (square_area_eq : A_square = side_length ^ 2)
  (rectangle_area_eq : A_rectangle = rectangle_width * rectangle_length)
  (triangle_area_eq : A_triangle = (triangle_base * triangle_height) / 2)
  (side_length_eq : side_length = 4)
  (rectangle_width_eq : rectangle_width = 4)
  (triangle_base_eq : triangle_base = 8)
  (areas_equal : A_square = A_rectangle ∧ A_square = A_triangle) :
  rectangle_length = 4 ∧ triangle_height = 4 :=
by
  sorry

end geometry_problem_l1508_150811


namespace number_of_distinct_lines_l1508_150828

theorem number_of_distinct_lines (S : Finset ℕ) (h : S = {1, 2, 3, 4, 5}) :
  (S.card.choose 2) - 2 = 18 :=
by
  -- Conditions
  have hS : S = {1, 2, 3, 4, 5} := h
  -- Conclusion
  sorry

end number_of_distinct_lines_l1508_150828


namespace sum_first_49_nat_nums_l1508_150880

theorem sum_first_49_nat_nums : (Finset.range 50).sum (fun x => x) = 1225 := 
by
  sorry

end sum_first_49_nat_nums_l1508_150880


namespace union_of_A_and_B_l1508_150831

/-- Let the universal set U = ℝ, and let the sets A = {x | x^2 - x - 2 = 0}
and B = {y | ∃ x, x ∈ A ∧ y = x + 3}. We want to prove that A ∪ B = {-1, 2, 5}.
-/
theorem union_of_A_and_B (U : Set ℝ) (A B : Set ℝ) (A_def : ∀ x, x ∈ A ↔ x^2 - x - 2 = 0)
  (B_def : ∀ y, y ∈ B ↔ ∃ x, x ∈ A ∧ y = x + 3) :
  A ∪ B = {-1, 2, 5} :=
sorry

end union_of_A_and_B_l1508_150831


namespace greatest_value_of_b_l1508_150849

theorem greatest_value_of_b : ∃ b, (∀ a, (-a^2 + 7 * a - 10 ≥ 0) → (a ≤ b)) ∧ b = 5 :=
by
  sorry

end greatest_value_of_b_l1508_150849
