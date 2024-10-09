import Mathlib

namespace series_sum_l1230_123034

noncomputable def sum_series : Real :=
  ∑' n: ℕ, (4 * (n + 1) + 2) / (3 : ℝ)^(n + 1)

theorem series_sum : sum_series = 3 := by
  sorry

end series_sum_l1230_123034


namespace number_of_classes_l1230_123068

theorem number_of_classes (x : ℕ) (h : x * (x - 1) / 2 = 28) : x = 8 := by
  sorry

end number_of_classes_l1230_123068


namespace pradeep_marks_l1230_123090

-- Conditions as definitions
def passing_percentage : ℝ := 0.35
def max_marks : ℕ := 600
def fail_difference : ℕ := 25

def passing_marks (total_marks : ℕ) (percentage : ℝ) : ℝ :=
  percentage * total_marks

def obtained_marks (passing_marks : ℝ) (difference : ℕ) : ℝ :=
  passing_marks - difference

-- Theorem statement
theorem pradeep_marks : obtained_marks (passing_marks max_marks passing_percentage) fail_difference = 185 := by
  sorry

end pradeep_marks_l1230_123090


namespace rectangular_reconfiguration_l1230_123074

theorem rectangular_reconfiguration (k : ℕ) (n : ℕ) (h₁ : k - 5 > 0) (h₂ : k ≥ 6) (h₃ : k ≤ 9) :
  (k * (k - 5) = n^2) → (n = 6) :=
by {
  sorry  -- proof is omitted
}

end rectangular_reconfiguration_l1230_123074


namespace second_train_catches_first_l1230_123098

-- Define the starting times and speeds
def t1_start_time := 14 -- 2:00 pm in 24-hour format
def t1_speed := 70 -- km/h
def t2_start_time := 15 -- 3:00 pm in 24-hour format
def t2_speed := 80 -- km/h

-- Define the time at which the second train catches the first train
def catch_time := 22 -- 10:00 pm in 24-hour format

theorem second_train_catches_first :
  ∃ t : ℕ, t = catch_time ∧
    t1_speed * ((t - t1_start_time) + 1) = t2_speed * (t - t2_start_time) := by
  sorry

end second_train_catches_first_l1230_123098


namespace subtracted_value_from_numbers_l1230_123085

theorem subtracted_value_from_numbers (A B C D E X : ℝ) 
  (h1 : (A + B + C + D + E) / 5 = 5)
  (h2 : ((A - X) + (B - X) + (C - X) + (D - X) + E) / 5 = 3.4) :
  X = 2 :=
by
  sorry

end subtracted_value_from_numbers_l1230_123085


namespace alice_total_spending_l1230_123029

theorem alice_total_spending :
  let book_price_gbp := 15
  let souvenir_price_eur := 20
  let gbp_to_usd_rate := 1.25
  let eur_to_usd_rate := 1.10
  let book_price_usd := book_price_gbp * gbp_to_usd_rate
  let souvenir_price_usd := souvenir_price_eur * eur_to_usd_rate
  let total_usd := book_price_usd + souvenir_price_usd
  total_usd = 40.75 :=
by
  sorry

end alice_total_spending_l1230_123029


namespace charlotte_avg_speed_l1230_123018

def distance : ℕ := 60  -- distance in miles
def time : ℕ := 6       -- time in hours

theorem charlotte_avg_speed : (distance / time) = 10 := by
  sorry

end charlotte_avg_speed_l1230_123018


namespace prob_rain_both_days_correct_l1230_123093

-- Definitions according to the conditions
def prob_rain_Saturday : ℝ := 0.4
def prob_rain_Sunday : ℝ := 0.3
def cond_prob_rain_Sunday_given_Saturday : ℝ := 0.5

-- Target probability to prove
def prob_rain_both_days : ℝ := prob_rain_Saturday * cond_prob_rain_Sunday_given_Saturday

-- Theorem statement
theorem prob_rain_both_days_correct : prob_rain_both_days = 0.2 :=
by
  sorry

end prob_rain_both_days_correct_l1230_123093


namespace Lisa_favorite_number_l1230_123086

theorem Lisa_favorite_number (a b : ℕ) (h : 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9) :
  (10 * a + b)^2 = (a + b)^3 → 10 * a + b = 27 := by
  intro h_eq
  sorry

end Lisa_favorite_number_l1230_123086


namespace pet_store_problem_l1230_123044

noncomputable def num_ways_to_buy_pets (puppies kittens hamsters birds : ℕ) (people : ℕ) : ℕ :=
  (puppies * kittens * hamsters * birds) * (people.factorial)

theorem pet_store_problem :
  num_ways_to_buy_pets 12 10 5 3 4 = 43200 :=
by
  sorry

end pet_store_problem_l1230_123044


namespace milk_left_l1230_123027

theorem milk_left (initial_milk : ℝ) (milk_james : ℝ) (milk_maria : ℝ) :
  initial_milk = 5 → milk_james = 15 / 4 → milk_maria = 3 / 4 → 
  initial_milk - (milk_james + milk_maria) = 1 / 2 :=
by
  intros h_initial h_james h_maria
  rw [h_initial, h_james, h_maria]
  -- The calculation would be performed here.
  sorry

end milk_left_l1230_123027


namespace initial_percentage_female_workers_l1230_123017

theorem initial_percentage_female_workers
(E : ℕ) (F : ℝ) 
(h1 : E + 30 = 360) 
(h2 : (F / 100) * E = (55 / 100) * (E + 30)) :
F = 60 :=
by
  -- proof omitted
  sorry

end initial_percentage_female_workers_l1230_123017


namespace valid_parameterizations_l1230_123078

noncomputable def line_equation (x y : ℝ) : Prop := y = (5/3) * x + 1

def parametrize_A (t : ℝ) : Prop :=
  ∃ (x y : ℝ), (x, y) = (3 + t * 3, 6 + t * 5) ∧ line_equation x y

def parametrize_D (t : ℝ) : Prop :=
  ∃ (x y : ℝ), (x, y) = (-1 + t * 3, -2/3 + t * 5) ∧ line_equation x y

theorem valid_parameterizations : parametrize_A t ∧ parametrize_D t :=
by
  -- Proof steps are skipped
  sorry

end valid_parameterizations_l1230_123078


namespace vector_BC_l1230_123054

/-- Given points A (0,1), B (3,2) and vector AC (-4,-3), prove that BC = (-7, -4) -/
theorem vector_BC
  (A B : ℝ × ℝ)
  (AC : ℝ × ℝ)
  (hA : A = (0, 1))
  (hB : B = (3, 2))
  (hAC : AC = (-4, -3)) :
  (AC - (B - A)) = (-7, -4) :=
by
  sorry

end vector_BC_l1230_123054


namespace correct_conclusions_l1230_123037

def pos_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n > 0

def sum_of_n_terms (S a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n+1) * S (n+1) = 9

def second_term_less_than_3 (a S : ℕ → ℝ) : Prop :=
  a 1 < 3

def is_decreasing_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) < a n

def exists_term_less_than_1_over_100 (a : ℕ → ℝ) : Prop :=
  ∃ n : ℕ, a n < 1/100

theorem correct_conclusions (a S : ℕ → ℝ) :
  pos_sequence a → sum_of_n_terms S a →
  second_term_less_than_3 a S ∧ (¬(∀ q : ℝ, ∃ r : ℝ, ∀ n : ℕ, a n = r * q ^ n)) ∧ is_decreasing_sequence a ∧ exists_term_less_than_1_over_100 a :=
sorry

end correct_conclusions_l1230_123037


namespace largest_constant_inequality_l1230_123005

theorem largest_constant_inequality :
  ∃ C : ℝ, (∀ x y z : ℝ, x^2 + y^2 + z^2 + 1 ≥ C * (x + y + z)) ∧ C = Real.sqrt (4 / 3) :=
by {
  sorry
}

end largest_constant_inequality_l1230_123005


namespace cos_sum_to_product_l1230_123056

theorem cos_sum_to_product (x : ℝ) : 
  (∃ a b c d : ℕ, a * Real.cos (b * x) * Real.cos (c * x) * Real.cos (d * x) =
  Real.cos (2 * x) + Real.cos (6 * x) + Real.cos (10 * x) + Real.cos (14 * x) 
  ∧ a + b + c + d = 18) :=
sorry

end cos_sum_to_product_l1230_123056


namespace total_collisions_100_balls_l1230_123063

def num_of_collisions (n: ℕ) : ℕ :=
  n * (n - 1) / 2

theorem total_collisions_100_balls :
  num_of_collisions 100 = 4950 :=
by
  sorry

end total_collisions_100_balls_l1230_123063


namespace geometric_sequence_problem_l1230_123030

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 1 then x * (Real.log x)
  else (Real.log x) / x

theorem geometric_sequence_problem
  (a : ℕ → ℝ) 
  (r : ℝ)
  (h1 : ∃ r > 0, ∀ n, a (n + 1) = r * a n)
  (h2 : a 3 * a 4 * a 5 = 1)
  (h3 : f (a 1) + f (a 2) + f (a 3) + f (a 4) + f (a 5) + f (a 6) = 2 * a 1) :
  a 1 = Real.exp 2 :=
sorry

end geometric_sequence_problem_l1230_123030


namespace percentage_decrease_to_gain_30_percent_profit_l1230_123028

theorem percentage_decrease_to_gain_30_percent_profit
  (C : ℝ) (P : ℝ) (S : ℝ) (S_new : ℝ) 
  (C_eq : C = 60)
  (S_eq : S = 1.25 * C)
  (S_new_eq1 : S_new = S - 12.60)
  (S_new_eq2 : S_new = 1.30 * (C - P * C)) : 
  P = 0.20 := by
  sorry

end percentage_decrease_to_gain_30_percent_profit_l1230_123028


namespace cost_of_15_brown_socks_is_3_dollars_l1230_123015

def price_of_brown_sock (price_white_socks : ℚ) (price_white_more_than_brown : ℚ) : ℚ :=
  (price_white_socks - price_white_more_than_brown) / 2

def cost_of_15_brown_socks (price_brown_sock : ℚ) : ℚ :=
  15 * price_brown_sock

theorem cost_of_15_brown_socks_is_3_dollars
  (price_white_socks : ℚ) (price_white_more_than_brown : ℚ) 
  (h1 : price_white_socks = 0.45) (h2 : price_white_more_than_brown = 0.25) :
  cost_of_15_brown_socks (price_of_brown_sock price_white_socks price_white_more_than_brown) = 3 := 
by
  sorry

end cost_of_15_brown_socks_is_3_dollars_l1230_123015


namespace circumcircle_radius_l1230_123091

theorem circumcircle_radius (a b c : ℝ) (h₁ : a = 5) (h₂ : b = 12) (h₃ : c = 13) :
  let s₁ := a^2 + b^2
  let s₂ := c^2
  s₁ = s₂ → 
  (c / 2) = 6.5 :=
by
  sorry

end circumcircle_radius_l1230_123091


namespace fractions_ordered_l1230_123014

theorem fractions_ordered :
  (2 / 5 : ℚ) < (3 / 5) ∧ (3 / 5) < (4 / 6) ∧ (4 / 6) < (4 / 5) ∧ (4 / 5) < (6 / 5) ∧ (6 / 5) < (4 / 3) :=
by
  sorry

end fractions_ordered_l1230_123014


namespace find_y_given_conditions_l1230_123073

theorem find_y_given_conditions (x y : ℝ) (h₁ : 3 * x^2 = y - 6) (h₂ : x = 4) : y = 54 :=
  sorry

end find_y_given_conditions_l1230_123073


namespace minuend_is_not_integer_l1230_123052

theorem minuend_is_not_integer (M S D : ℚ) (h1 : M + S + D = 555) (h2 : M - S = D) : ¬ ∃ n : ℤ, M = n := 
by
  sorry

end minuend_is_not_integer_l1230_123052


namespace swimming_speed_l1230_123041

theorem swimming_speed (v : ℝ) (water_speed : ℝ) (swim_time : ℝ) (distance : ℝ) :
  water_speed = 8 →
  swim_time = 8 →
  distance = 16 →
  distance = (v - water_speed) * swim_time →
  v = 10 := 
by
  intros h1 h2 h3 h4
  sorry

end swimming_speed_l1230_123041


namespace find_theta_l1230_123020

theorem find_theta (θ : ℝ) :
  (0 : ℝ) ≤ θ ∧ θ ≤ 2 * Real.pi →
  (∀ x, (0 : ℝ) ≤ x ∧ x ≤ 2 →
    x^2 * Real.cos θ - 2 * x * (1 - x) + (2 - x)^2 * Real.sin θ > 0) →
  (Real.pi / 12 < θ ∧ θ < 5 * Real.pi / 12) :=
by
  intros hθ hx
  sorry

end find_theta_l1230_123020


namespace determine_gizmos_l1230_123081

theorem determine_gizmos (g d : ℝ)
  (h1 : 80 * (g * 160 + d * 240) = 80)
  (h2 : 100 * (3 * g * 900 + 3 * d * 600) = 100)
  (h3 : 70 * (5 * g * n + 5 * d * 1050) = 70 * 5 * (g + d) ) :
  n = 70 := sorry

end determine_gizmos_l1230_123081


namespace unique_zero_in_interval_l1230_123001

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (x + 1) + a * x ^ 2

theorem unique_zero_in_interval
  (a : ℝ) (ha : a > 0)
  (x₀ : ℝ) (hx₀ : f a x₀ = 0)
  (h_interval : -1 < x₀ ∧ x₀ < 0) :
  Real.exp (-2) < x₀ + 1 ∧ x₀ + 1 < Real.exp (-1) :=
sorry

end unique_zero_in_interval_l1230_123001


namespace find_B_divisible_by_6_l1230_123064

theorem find_B_divisible_by_6 (B : ℕ) : (5170 + B) % 6 = 0 ↔ (B = 2 ∨ B = 8) :=
by
  -- Conditions extracted from the problem are directly used here:
  sorry -- Proof would be here

end find_B_divisible_by_6_l1230_123064


namespace probability_sum_18_two_12_sided_dice_l1230_123047

theorem probability_sum_18_two_12_sided_dice :
  let total_outcomes := 12 * 12
  let successful_outcomes := 7
  successful_outcomes / total_outcomes = 7 / 144 := by
sorry

end probability_sum_18_two_12_sided_dice_l1230_123047


namespace sum_of_possible_values_N_l1230_123057

variable (a b c N : ℕ)

theorem sum_of_possible_values_N :
  (N = a * b * c) ∧ (N = 8 * (a + b + c)) ∧ (c = 2 * a + b) → N = 136 := 
by
  sorry

end sum_of_possible_values_N_l1230_123057


namespace average_speed_of_entire_trip_l1230_123049

/-- Conditions -/
def distance_local : ℝ := 40  -- miles
def speed_local : ℝ := 20  -- mph
def distance_highway : ℝ := 180  -- miles
def speed_highway : ℝ := 60  -- mph

/-- Average speed proof statement -/
theorem average_speed_of_entire_trip :
  let total_distance := distance_local + distance_highway
  let total_time := distance_local / speed_local + distance_highway / speed_highway
  total_distance / total_time = 44 :=
by
  sorry

end average_speed_of_entire_trip_l1230_123049


namespace b_investment_months_after_a_l1230_123000

-- Definitions based on the conditions
def a_investment : ℕ := 100
def b_investment : ℕ := 200
def total_yearly_investment_period : ℕ := 12
def total_profit : ℕ := 100
def a_share_of_profit : ℕ := 50
def x (x_val : ℕ) : Prop := x_val = 6

-- Main theorem to prove
theorem b_investment_months_after_a (x_val : ℕ) 
  (h1 : a_investment = 100)
  (h2 : b_investment = 200)
  (h3 : total_yearly_investment_period = 12)
  (h4 : total_profit = 100)
  (h5 : a_share_of_profit = 50) :
  (100 * total_yearly_investment_period) = 200 * (total_yearly_investment_period - x_val) → 
  x x_val := 
by
  sorry

end b_investment_months_after_a_l1230_123000


namespace solution_l1230_123097

-- Define the conditions based on the given problem
variables {A B C D : Type}
variables {AB BC CD DA : ℝ} (h1 : AB = 65) (h2 : BC = 105) (h3 : CD = 125) (h4 : DA = 95)
variables (cy_in_circle : CyclicQuadrilateral A B C D)
variables (circ_inscribed : TangentialQuadrilateral A B C D)

-- Function that computes the absolute difference between segments x and y on side of length CD
noncomputable def find_absolute_difference (x y : ℝ) (h5 : x + y = 125) : ℝ := |x - y|

-- The proof statement
theorem solution :
  ∃ (x y : ℝ), x + y = 125 ∧
  (find_absolute_difference x y (by sorry) = 14) := sorry

end solution_l1230_123097


namespace scott_awards_l1230_123007

theorem scott_awards (S : ℕ) 
  (h1 : ∃ J, J = 3 * S)
  (h2 : ∃ B, B = 2 * (3 * S) ∧ B = 24) : S = 4 := 
by 
  sorry

end scott_awards_l1230_123007


namespace trigonometric_comparison_l1230_123035

noncomputable def a : ℝ := 2 * Real.sin (13 * Real.pi / 180) * Real.cos (13 * Real.pi / 180)
noncomputable def b : ℝ := 2 * Real.tan (76 * Real.pi / 180) / (1 + Real.tan (76 * Real.pi / 180)^2)
noncomputable def c : ℝ := Real.sqrt ((1 - Real.cos (50 * Real.pi / 180)) / 2)

theorem trigonometric_comparison : b > a ∧ a > c := by
  sorry

end trigonometric_comparison_l1230_123035


namespace negation_of_proposition_l1230_123083

theorem negation_of_proposition (a b : ℝ) : 
  (¬ (∀ (a b : ℝ), (ab > 0 → a > 0)) ↔ ∀ (a b : ℝ), (ab ≤ 0 → a ≤ 0)) := 
sorry

end negation_of_proposition_l1230_123083


namespace danny_wrappers_more_than_soda_cans_l1230_123046

theorem danny_wrappers_more_than_soda_cans :
  (67 - 22 = 45) := sorry

end danny_wrappers_more_than_soda_cans_l1230_123046


namespace negation_exists_x_squared_leq_abs_x_l1230_123067

theorem negation_exists_x_squared_leq_abs_x :
  (¬ ∃ x : ℝ, x ∈ Set.Ioo (-1 : ℝ) (0 : ℝ) ∧ x^2 ≤ |x|) ↔ (∀ x : ℝ, x ∈ Set.Ioo (-1 : ℝ) (0 : ℝ) → x^2 > |x|) :=
by
  sorry

end negation_exists_x_squared_leq_abs_x_l1230_123067


namespace no_lattice_points_on_hyperbola_l1230_123013

theorem no_lattice_points_on_hyperbola : ∀ x y : ℤ, x^2 - y^2 ≠ 2022 :=
by
  intro x y
  -- proof omitted
  sorry

end no_lattice_points_on_hyperbola_l1230_123013


namespace broccoli_area_l1230_123089

/--
A farmer grows broccoli in a square-shaped farm. This year, he produced 2601 broccoli,
which is 101 more than last year. The shape of the area used for growing the broccoli 
has remained square in both years. Assuming each broccoli takes up an equal amount of 
area, prove that each broccoli takes up 1 square unit of area.
-/
theorem broccoli_area (x y : ℕ) 
  (h1 : y^2 = x^2 + 101) 
  (h2 : y^2 = 2601) : 
  1 = 1 := 
sorry

end broccoli_area_l1230_123089


namespace reflect_origin_l1230_123010

theorem reflect_origin (x y : ℝ) (h₁ : x = 4) (h₂ : y = -3) : 
  (-x, -y) = (-4, 3) :=
by {
  sorry
}

end reflect_origin_l1230_123010


namespace total_distance_from_A_through_B_to_C_l1230_123072

noncomputable def distance_A_B_map : ℝ := 120
noncomputable def distance_B_C_map : ℝ := 70
noncomputable def map_scale : ℝ := 10 -- km per cm

noncomputable def distance_A_B := distance_A_B_map * map_scale -- Distance from City A to City B in km
noncomputable def distance_B_C := distance_B_C_map * map_scale -- Distance from City B to City C in km
noncomputable def total_distance := distance_A_B + distance_B_C -- Total distance in km

theorem total_distance_from_A_through_B_to_C :
  total_distance = 1900 := by
  sorry

end total_distance_from_A_through_B_to_C_l1230_123072


namespace Andrew_is_19_l1230_123051

-- Define individuals and their relationships
def Andrew_age (Bella_age : ℕ) : ℕ := Bella_age - 5
def Bella_age (Carlos_age : ℕ) : ℕ := Carlos_age + 4
def Carlos_age : ℕ := 20

-- Formulate the problem statement
theorem Andrew_is_19 : Andrew_age (Bella_age Carlos_age) = 19 :=
by
  sorry

end Andrew_is_19_l1230_123051


namespace trap_speed_independent_of_location_l1230_123024

theorem trap_speed_independent_of_location 
  (h b a : ℝ) (v_mouse : ℝ) 
  (path_length : ℝ := Real.sqrt (a^2 + (3*h)^2)) 
  (T : ℝ := path_length / v_mouse) 
  (step_height : ℝ := h) 
  (v_trap : ℝ := step_height / T) 
  (h_val : h = 3) 
  (b_val : b = 1) 
  (a_val : a = 8) 
  (v_mouse_val : v_mouse = 17) : 
  v_trap = 8 := by
  sorry

end trap_speed_independent_of_location_l1230_123024


namespace square_of_product_of_third_sides_l1230_123039

-- Given data for triangles P1 and P2
variables {a b c d : ℝ}

-- Areas of triangles P1 and P2
def area_P1_pos (a b : ℝ) : Prop := a * b / 2 = 3
def area_P2_pos (a d : ℝ) : Prop := a * d / 2 = 6

-- Condition that b = d / 2
def side_ratio (b d : ℝ) : Prop := b = d / 2

-- Pythagorean theorem applied to both triangles
def pythagorean_P1 (a b c : ℝ) : Prop := a^2 + b^2 = c^2
def pythagorean_P2 (a d c : ℝ) : Prop := a^2 + d^2 = c^2

-- The goal is to prove (cd)^2 = 120
theorem square_of_product_of_third_sides (a b c d : ℝ)
  (h_area_P1: area_P1_pos a b) 
  (h_area_P2: area_P2_pos a d) 
  (h_side_ratio: side_ratio b d) 
  (h_pythagorean_P1: pythagorean_P1 a b c) 
  (h_pythagorean_P2: pythagorean_P2 a d c) :
  (c * d)^2 = 120 := 
sorry

end square_of_product_of_third_sides_l1230_123039


namespace rectangle_perimeter_l1230_123025

theorem rectangle_perimeter
  (L W : ℕ)
  (h1 : L * W = 360)
  (h2 : (L + 10) * (W - 6) = 360) :
  2 * L + 2 * W = 76 := 
sorry

end rectangle_perimeter_l1230_123025


namespace charlie_and_dana_proof_l1230_123071

noncomputable def charlie_and_dana_ways 
    (cookies : ℕ) (smoothies : ℕ) (total_items : ℕ) 
    (distinct_charlie : ℕ) 
    (repeatable_dana : ℕ) : ℕ :=
    if cookies = 8 ∧ smoothies = 5 ∧ total_items = 5 ∧ distinct_charlie = 0 
       ∧ repeatable_dana = 0 then 27330 else 0

theorem charlie_and_dana_proof :
  charlie_and_dana_ways 8 5 5 0 0 = 27330 := 
  sorry

end charlie_and_dana_proof_l1230_123071


namespace flower_stones_per_bracelet_l1230_123066

theorem flower_stones_per_bracelet (total_stones : ℝ) (bracelets : ℝ)  (H_total: total_stones = 88.0) (H_bracelets: bracelets = 8.0) :
  (total_stones / bracelets = 11.0) :=
by
  rw [H_total, H_bracelets]
  norm_num

end flower_stones_per_bracelet_l1230_123066


namespace frustum_shortest_distance_l1230_123069

open Real

noncomputable def shortest_distance (R1 R2 : ℝ) (AB : ℝ) (string_from_midpoint : Bool) : ℝ :=
  if R1 = 5 ∧ R2 = 10 ∧ AB = 20 ∧ string_from_midpoint = true then 4 else 0

theorem frustum_shortest_distance : 
  shortest_distance 5 10 20 true = 4 :=
by sorry

end frustum_shortest_distance_l1230_123069


namespace units_digit_product_l1230_123076

theorem units_digit_product : (3^5 * 2^3) % 10 = 4 := 
sorry

end units_digit_product_l1230_123076


namespace find_positive_integers_satisfying_inequality_l1230_123082

theorem find_positive_integers_satisfying_inequality :
  (∃ n : ℕ, (n - 1) * (n - 3) * (n - 5) * (n - 7) * (n - 9) * (n - 11) * (n - 13) * (n - 15) *
    (n - 17) * (n - 19) * (n - 21) * (n - 23) * (n - 25) * (n - 27) * (n - 29) * (n - 31) *
    (n - 33) * (n - 35) * (n - 37) * (n - 39) * (n - 41) * (n - 43) * (n - 45) * (n - 47) *
    (n - 49) * (n - 51) * (n - 53) * (n - 55) * (n - 57) * (n - 59) * (n - 61) * (n - 63) *
    (n - 65) * (n - 67) * (n - 69) * (n - 71) * (n - 73) * (n - 75) * (n - 77) * (n - 79) *
    (n - 81) * (n - 83) * (n - 85) * (n - 87) * (n - 89) * (n - 91) * (n - 93) * (n - 95) *
    (n - 97) * (n - 99) < 0 ∧ 1 ≤ n ∧ n ≤ 99) 
  → ∃ f : ℕ → ℕ, (∀ i, f i = 2 + 4 * i) ∧ (∀ i, 1 ≤ f i ∧ f i ≤ 24) :=
by
  sorry

end find_positive_integers_satisfying_inequality_l1230_123082


namespace distance_between_parallel_lines_l1230_123077

theorem distance_between_parallel_lines (r d : ℝ) :
  let c₁ := 36
  let c₂ := 36
  let c₃ := 40
  let expr1 := (324 : ℝ) + (1 / 4) * d^2
  let expr2 := (400 : ℝ) + d^2
  let radius_eq1 := r^2 = expr1
  let radius_eq2 := r^2 = expr2
  radius_eq1 ∧ radius_eq2 → d = Real.sqrt (304 / 3) :=
by
  sorry

end distance_between_parallel_lines_l1230_123077


namespace cone_csa_l1230_123095

theorem cone_csa (r l : ℝ) (h_r : r = 8) (h_l : l = 18) : 
  (Real.pi * r * l) = 144 * Real.pi :=
by 
  rw [h_r, h_l]
  norm_num
  sorry

end cone_csa_l1230_123095


namespace pictures_vertically_l1230_123096

def total_pictures := 30
def haphazard_pictures := 5
def horizontal_pictures := total_pictures / 2

theorem pictures_vertically : total_pictures - (horizontal_pictures + haphazard_pictures) = 10 := by
  sorry

end pictures_vertically_l1230_123096


namespace income_remaining_percentage_l1230_123094

theorem income_remaining_percentage :
  let initial_income := 100
  let food_percentage := 42
  let education_percentage := 18
  let transportation_percentage := 12
  let house_rent_percentage := 55
  let total_spent := food_percentage + education_percentage + transportation_percentage
  let remaining_after_expenses := initial_income - total_spent
  let house_rent_amount := (house_rent_percentage * remaining_after_expenses) / 100
  let final_remaining_income := remaining_after_expenses - house_rent_amount
  final_remaining_income = 12.6 :=
by
  sorry

end income_remaining_percentage_l1230_123094


namespace number_of_6mb_pictures_l1230_123065

theorem number_of_6mb_pictures
    (n : ℕ)             -- initial number of pictures
    (size_old : ℕ)      -- size of old pictures in megabytes
    (size_new : ℕ)      -- size of new pictures in megabytes
    (total_capacity : ℕ)  -- total capacity of the memory card in megabytes
    (h1 : n = 3000)      -- given memory card can hold 3000 pictures
    (h2 : size_old = 8)  -- each old picture is 8 megabytes
    (h3 : size_new = 6)  -- each new picture is 6 megabytes
    (h4 : total_capacity = n * size_old)  -- total capacity calculated from old pictures
    : total_capacity / size_new = 4000 :=  -- the number of new pictures that can be held
by
  sorry

end number_of_6mb_pictures_l1230_123065


namespace correct_total_distance_l1230_123099

theorem correct_total_distance (km_to_m : 3.5 * 1000 = 3500) (add_m : 3500 + 200 = 3700) : 
  3.5 * 1000 + 200 = 3700 :=
by
  -- The proof would be filled here.
  sorry

end correct_total_distance_l1230_123099


namespace max_marks_for_test_l1230_123006

theorem max_marks_for_test (M : ℝ) (h1: (0.30 * M) = 180) : M = 600 :=
by
  sorry

end max_marks_for_test_l1230_123006


namespace next_sales_amount_l1230_123019

theorem next_sales_amount
  (royalties1: ℝ)
  (sales1: ℝ)
  (royalties2: ℝ)
  (percentage_decrease: ℝ)
  (X: ℝ)
  (h1: royalties1 = 4)
  (h2: sales1 = 20)
  (h3: royalties2 = 9)
  (h4: percentage_decrease = 58.333333333333336 / 100)
  (h5: royalties2 / X = royalties1 / sales1 - ((royalties1 / sales1) * percentage_decrease)): 
  X = 108 := 
  by 
    -- Proof omitted
    sorry

end next_sales_amount_l1230_123019


namespace train_cross_time_l1230_123061

noncomputable def speed_kmh := 72
noncomputable def speed_mps : ℝ := speed_kmh * (1000 / 3600)
noncomputable def length_train := 180
noncomputable def length_bridge := 270
noncomputable def total_distance := length_train + length_bridge
noncomputable def time_to_cross := total_distance / speed_mps

theorem train_cross_time :
  time_to_cross = 22.5 := 
sorry

end train_cross_time_l1230_123061


namespace remainder_2021_2025_mod_17_l1230_123045

theorem remainder_2021_2025_mod_17 : 
  (2021 * 2022 * 2023 * 2024 * 2025) % 17 = 0 :=
by 
  -- Proof omitted for brevity
  sorry

end remainder_2021_2025_mod_17_l1230_123045


namespace mean_equivalence_l1230_123009

theorem mean_equivalence {x : ℚ} :
  (8 + 15 + 21) / 3 = (18 + x) / 2 → x = 34 / 3 :=
by
  sorry

end mean_equivalence_l1230_123009


namespace base8_addition_l1230_123031

theorem base8_addition : (234 : ℕ) + (157 : ℕ) = (4 * 8^2 + 1 * 8^1 + 3 * 8^0 : ℕ) :=
by sorry

end base8_addition_l1230_123031


namespace perimeter_of_rectangle_l1230_123004

theorem perimeter_of_rectangle (area width : ℝ) (h_area : area = 750) (h_width : width = 25) :
  ∃ perimeter length, length = area / width ∧ perimeter = 2 * (length + width) ∧ perimeter = 110 := by
  sorry

end perimeter_of_rectangle_l1230_123004


namespace find_ratio_l1230_123012

noncomputable def decagon_area : ℝ := 12
noncomputable def area_below_PQ : ℝ := 6
noncomputable def unit_square_area : ℝ := 1
noncomputable def triangle_base : ℝ := 6
noncomputable def area_above_PQ : ℝ := 6
noncomputable def XQ : ℝ := 4
noncomputable def QY : ℝ := 2

theorem find_ratio {XQ QY : ℝ} (h1 : decagon_area = 12) (h2 : area_below_PQ = 6)
                   (h3 : unit_square_area = 1) (h4 : triangle_base = 6)
                   (h5 : area_above_PQ = 6) (h6 : XQ + QY = 6) :
  XQ / QY = 2 := by { sorry }

end find_ratio_l1230_123012


namespace symmetric_line_equation_l1230_123080

theorem symmetric_line_equation (x y : ℝ) : 
  3 * x - 4 * y + 5 = 0 → (3 * x + 4 * y - 5 = 0) :=
by
sorry

end symmetric_line_equation_l1230_123080


namespace max_value_of_a_plus_b_l1230_123032

theorem max_value_of_a_plus_b (a b : ℕ) (h1 : 7 * a + 19 * b = 213) (h2 : a > 0) (h3 : b > 0) : a + b = 27 :=
sorry

end max_value_of_a_plus_b_l1230_123032


namespace unique_solution_tan_eq_sin_cos_l1230_123033

theorem unique_solution_tan_eq_sin_cos :
  ∃! x, 0 ≤ x ∧ x ≤ Real.arccos 0.1 ∧ Real.tan x = Real.sin (Real.cos x) :=
sorry

end unique_solution_tan_eq_sin_cos_l1230_123033


namespace raft_travel_time_l1230_123060

noncomputable def downstream_speed (x y : ℝ) : ℝ := x + y
noncomputable def upstream_speed (x y : ℝ) : ℝ := x - y

theorem raft_travel_time {x y : ℝ} 
  (h1 : 7 * upstream_speed x y = 5 * downstream_speed x y) : (35 : ℝ) = (downstream_speed x y) * 7 / 4 := by sorry

end raft_travel_time_l1230_123060


namespace round_trip_time_l1230_123062

theorem round_trip_time 
  (d1 d2 d3 : ℝ) 
  (s1 s2 s3 t : ℝ) 
  (h1 : d1 = 18) 
  (h2 : d2 = 18) 
  (h3 : d3 = 36) 
  (h4 : s1 = 12) 
  (h5 : s2 = 10) 
  (h6 : s3 = 9) 
  (h7 : t = (d1 / s1) + (d2 / s2) + (d3 / s3)) :
  t = 7.3 :=
by
  sorry

end round_trip_time_l1230_123062


namespace equal_share_is_168_l1230_123055

namespace StrawberryProblem

def brother_baskets : ℕ := 3
def strawberries_per_basket : ℕ := 15
def brother_strawberries : ℕ := brother_baskets * strawberries_per_basket

def kimberly_multiplier : ℕ := 8
def kimberly_strawberries : ℕ := kimberly_multiplier * brother_strawberries

def parents_difference : ℕ := 93
def parents_strawberries : ℕ := kimberly_strawberries - parents_difference

def total_strawberries : ℕ := kimberly_strawberries + brother_strawberries + parents_strawberries
def total_people : ℕ := 4

def equal_share : ℕ := total_strawberries / total_people

theorem equal_share_is_168 :
  equal_share = 168 := by
  -- We state that for the given problem conditions,
  -- the total number of strawberries divided equally among the family members results in 168 strawberries per person.
  sorry

end StrawberryProblem

end equal_share_is_168_l1230_123055


namespace solution_set_of_inequality_l1230_123016

theorem solution_set_of_inequality (x : ℝ) (n : ℕ) (h1 : n ≤ x ∧ x < n + 1 ∧ 0 < n) :
  4 * (⌊x⌋ : ℝ)^2 - 36 * (⌊x⌋ : ℝ) + 45 < 0 ↔ ∃ k : ℕ, (2 ≤ k ∧ k < 8 ∧ ⌊x⌋ = k) :=
by sorry

end solution_set_of_inequality_l1230_123016


namespace fans_received_all_offers_l1230_123042

theorem fans_received_all_offers :
  let hotdog_freq := 90
  let soda_freq := 45
  let popcorn_freq := 60
  let stadium_capacity := 4500
  let lcm_freq := Nat.lcm (Nat.lcm hotdog_freq soda_freq) popcorn_freq
  (stadium_capacity / lcm_freq) = 25 :=
by
  sorry

end fans_received_all_offers_l1230_123042


namespace translate_parabola_l1230_123003

theorem translate_parabola (x : ℝ) :
  (x^2 + 3) = (x - 5)^2 + 3 :=
sorry

end translate_parabola_l1230_123003


namespace real_root_exists_l1230_123059

theorem real_root_exists (a b c : ℝ) :
  (∃ x : ℝ, x^2 + (a - b) * x + (b - c) = 0) ∨ 
  (∃ x : ℝ, x^2 + (b - c) * x + (c - a) = 0) ∨ 
  (∃ x : ℝ, x^2 + (c - a) * x + (a - b) = 0) :=
by {
  sorry
}

end real_root_exists_l1230_123059


namespace sum_of_three_pairwise_relatively_prime_integers_l1230_123022

theorem sum_of_three_pairwise_relatively_prime_integers
  (a b c : ℕ)
  (h1 : a > 1)
  (h2 : b > 1)
  (h3 : c > 1)
  (h4 : a * b * c = 13824)
  (h5 : Nat.gcd a b = 1)
  (h6 : Nat.gcd b c = 1)
  (h7 : Nat.gcd a c = 1) :
  a + b + c = 144 :=
by
  sorry

end sum_of_three_pairwise_relatively_prime_integers_l1230_123022


namespace mul_exponents_l1230_123092

theorem mul_exponents (m : ℝ) : 2 * m^3 * 3 * m^4 = 6 * m^7 :=
by sorry

end mul_exponents_l1230_123092


namespace fraction_of_satisfactory_is_15_over_23_l1230_123002

def num_students_with_grade_A : ℕ := 6
def num_students_with_grade_B : ℕ := 5
def num_students_with_grade_C : ℕ := 4
def num_students_with_grade_D : ℕ := 2
def num_students_with_grade_F : ℕ := 6

def num_satisfactory_students : ℕ := 
  num_students_with_grade_A + num_students_with_grade_B + num_students_with_grade_C

def total_students : ℕ := 
  num_satisfactory_students + num_students_with_grade_D + num_students_with_grade_F

def fraction_satisfactory : ℚ := 
  (num_satisfactory_students : ℚ) / (total_students : ℚ)

theorem fraction_of_satisfactory_is_15_over_23 : 
  fraction_satisfactory = 15/23 :=
by
  -- proof omitted
  sorry

end fraction_of_satisfactory_is_15_over_23_l1230_123002


namespace mass_of_added_water_with_temp_conditions_l1230_123023

theorem mass_of_added_water_with_temp_conditions
  (m_l : ℝ) (t_pi t_B t : ℝ) (c_B c_l lambda : ℝ) :
  m_l = 0.05 →
  t_pi = -10 →
  t_B = 10 →
  t = 0 →
  c_B = 4200 →
  c_l = 2100 →
  lambda = 3.3 * 10^5 →
  (0.0028 ≤ (2.1 * m_l * 10 + lambda * m_l) / (42 * 10) 
  ∧ (2.1 * m_l * 10) / (42 * 10) ≤ 0.418) :=
by
  sorry

end mass_of_added_water_with_temp_conditions_l1230_123023


namespace paul_has_5point86_left_l1230_123084

noncomputable def paulLeftMoney : ℝ := 15 - (2 + (3 - 0.1*3) + 2*2 + 0.05 * (2 + (3 - 0.1*3) + 2*2))

theorem paul_has_5point86_left :
  paulLeftMoney = 5.86 :=
by
  sorry

end paul_has_5point86_left_l1230_123084


namespace closest_ratio_l1230_123053

theorem closest_ratio
  (a_0 : ℝ)
  (h_pos : a_0 > 0)
  (a_10 : ℝ)
  (h_eq : a_10 = a_0 * (1 + 0.05) ^ 10) :
  abs ((a_10 / a_0) - 1.6) ≤ abs ((a_10 / a_0) - 1.5) ∧
  abs ((a_10 / a_0) - 1.6) ≤ abs ((a_10 / a_0) - 1.7) ∧
  abs ((a_10 / a_0) - 1.6) ≤ abs ((a_10 / a_0) - 1.8) := 
sorry

end closest_ratio_l1230_123053


namespace compound_interest_rate_l1230_123040

theorem compound_interest_rate
  (P : ℝ)  -- Principal amount
  (r : ℝ)  -- Annual interest rate in decimal
  (A2 A3 : ℝ)  -- Amounts after 2 and 3 years
  (h2 : A2 = P * (1 + r)^2)
  (h3 : A3 = P * (1 + r)^3) :
  A2 = 17640 → A3 = 22932 → r = 0.3 := by
  sorry

end compound_interest_rate_l1230_123040


namespace family_reunion_handshakes_l1230_123075

theorem family_reunion_handshakes (married_couples : ℕ) (participants : ℕ) (allowed_handshakes : ℕ) (total_handshakes : ℕ) :
  married_couples = 8 →
  participants = married_couples * 2 →
  allowed_handshakes = participants - 1 - 1 - 6 →
  total_handshakes = (participants * allowed_handshakes) / 2 →
  total_handshakes = 64 :=
by
  intros h1 h2 h3 h4
  sorry

end family_reunion_handshakes_l1230_123075


namespace f_of_f_five_l1230_123087

noncomputable def f : ℝ → ℝ := sorry

axiom f_periodicity (x : ℝ) : f (x + 2) = 1 / f x
axiom f_initial_value : f 1 = -5

theorem f_of_f_five : f (f 5) = -1 / 5 :=
by sorry

end f_of_f_five_l1230_123087


namespace parallelogram_base_length_l1230_123070

theorem parallelogram_base_length (A h : ℕ) (hA : A = 32) (hh : h = 8) : (A / h) = 4 := by
  sorry

end parallelogram_base_length_l1230_123070


namespace height_percentage_difference_l1230_123050

theorem height_percentage_difference (A B : ℝ) (h : B = A * (4/3)) : 
  (A * (1/3) / B) * 100 = 25 := by
  sorry

end height_percentage_difference_l1230_123050


namespace find_m_l1230_123043

def g (n : ℤ) : ℤ :=
  if n % 2 = 1 then n + 5
  else if n % 3 = 0 then n / 3
  else n / 2

theorem find_m (m : ℤ) (h_odd : m % 2 = 1) (h_g : g (g (g m)) = 16) : m = 59 ∨ m = 91 :=
by sorry

end find_m_l1230_123043


namespace sum_non_prime_between_50_and_60_eq_383_l1230_123026

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def non_primes_between_50_and_60 : List ℕ :=
  [51, 52, 54, 55, 56, 57, 58]

def sum_non_primes_between_50_and_60 : ℕ :=
  non_primes_between_50_and_60.sum

theorem sum_non_prime_between_50_and_60_eq_383 :
  sum_non_primes_between_50_and_60 = 383 :=
by
  sorry

end sum_non_prime_between_50_and_60_eq_383_l1230_123026


namespace g_of_3_eq_seven_over_two_l1230_123058

theorem g_of_3_eq_seven_over_two :
  ∀ f g : ℝ → ℝ,
  (∀ x, f x = (2 * x + 3) / (x - 1)) →
  (∀ x, g x = (x + 4) / (x - 1)) →
  g 3 = 7 / 2 :=
by
  sorry

end g_of_3_eq_seven_over_two_l1230_123058


namespace annual_concert_tickets_l1230_123079

theorem annual_concert_tickets (S NS : ℕ) (h1 : S + NS = 150) (h2 : 5 * S + 8 * NS = 930) : NS = 60 :=
by
  sorry

end annual_concert_tickets_l1230_123079


namespace largest_prime_factor_among_numbers_l1230_123038

-- Definitions of the numbers with their prime factors
def num1 := 39
def num2 := 51
def num3 := 77
def num4 := 91
def num5 := 121

def prime_factors (n : ℕ) : List ℕ := sorry  -- Placeholder for the prime factors function

-- Prime factors for the given numbers
def factors_num1 := prime_factors num1
def factors_num2 := prime_factors num2
def factors_num3 := prime_factors num3
def factors_num4 := prime_factors num4
def factors_num5 := prime_factors num5

-- Extract the largest prime factor from a list of factors
def largest_prime_factor (factors : List ℕ) : ℕ := sorry  -- Placeholder for the largest_prime_factor function

-- Largest prime factors for each number
def largest_prime_factor_num1 := largest_prime_factor factors_num1
def largest_prime_factor_num2 := largest_prime_factor factors_num2
def largest_prime_factor_num3 := largest_prime_factor factors_num3
def largest_prime_factor_num4 := largest_prime_factor factors_num4
def largest_prime_factor_num5 := largest_prime_factor factors_num5

theorem largest_prime_factor_among_numbers :
  largest_prime_factor_num2 = 17 ∧
  largest_prime_factor_num1 = 13 ∧
  largest_prime_factor_num3 = 11 ∧
  largest_prime_factor_num4 = 13 ∧
  largest_prime_factor_num5 = 11 ∧
  (largest_prime_factor_num2 > largest_prime_factor_num1) ∧
  (largest_prime_factor_num2 > largest_prime_factor_num3) ∧
  (largest_prime_factor_num2 > largest_prime_factor_num4) ∧
  (largest_prime_factor_num2 > largest_prime_factor_num5)
:= by
  -- skeleton proof, details to be filled in
  sorry

end largest_prime_factor_among_numbers_l1230_123038


namespace total_pieces_10_rows_l1230_123036

-- Define the conditions for the rods
def rod_seq (n : ℕ) : ℕ := 3 * n

-- Define the sum of the arithmetic sequence for rods
def sum_rods (n : ℕ) : ℕ := 3 * (n * (n + 1)) / 2

-- Define the conditions for the connectors
def connector_seq (n : ℕ) : ℕ := n + 1

-- Define the sum of the arithmetic sequence for connectors
def sum_connectors (n : ℕ) : ℕ := (n * (n + 1)) / 2

-- Define the total pieces calculation
def total_pieces (n : ℕ) : ℕ := sum_rods n + sum_connectors (n + 1)

-- The target statement
theorem total_pieces_10_rows : total_pieces 10 = 231 :=
by
  sorry

end total_pieces_10_rows_l1230_123036


namespace point_B_in_first_quadrant_l1230_123048

def is_first_quadrant (p : ℝ × ℝ) : Prop := p.1 > 0 ∧ p.2 > 0

theorem point_B_in_first_quadrant : is_first_quadrant (1, 2) :=
by
  sorry

end point_B_in_first_quadrant_l1230_123048


namespace triangle_perimeter_l1230_123088

theorem triangle_perimeter (x : ℕ) (a b c : ℕ) 
  (h1 : a = 3 * x) (h2 : b = 4 * x) (h3 : c = 5 * x)  
  (h4 : c - a = 6) : a + b + c = 36 := 
by
  sorry

end triangle_perimeter_l1230_123088


namespace percentage_drop_l1230_123011

theorem percentage_drop (P N P' N' : ℝ) (h1 : N' = 1.60 * N) (h2 : P' * N' = 1.2800000000000003 * (P * N)) :
  P' = 0.80 * P :=
by
  sorry

end percentage_drop_l1230_123011


namespace digits_sum_unique_l1230_123008

variable (A B C D E F G H : ℕ)

theorem digits_sum_unique :
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ F ∧ A ≠ G ∧ A ≠ H ∧
  B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ F ∧ B ≠ G ∧ B ≠ H ∧
  C ≠ D ∧ C ≠ E ∧ C ≠ F ∧ C ≠ G ∧ C ≠ H ∧
  D ≠ E ∧ D ≠ F ∧ D ≠ G ∧ D ≠ H ∧
  E ≠ F ∧ E ≠ G ∧ E ≠ H ∧
  F ≠ G ∧ F ≠ H ∧
  G ≠ H ∧
  0 ≤ A ∧ A ≤ 9 ∧ 0 ≤ B ∧ B ≤ 9 ∧ 0 ≤ C ∧ C ≤ 9 ∧ 0 ≤ D ∧ D ≤ 9 ∧
  0 ≤ E ∧ E ≤ 9 ∧ 0 ≤ F ∧ F ≤ 9 ∧ 0 ≤ G ∧ G ≤ 9 ∧ 0 ≤ H ∧ H ≤ 9 ∧
  (A * 1000 + B * 100 + C * 10 + D) + (E * 1000 + F * 100 + G * 10 + H) = 10652 ∧
  A = 9 ∧ B = 5 ∧ C = 6 ∧ D = 7 ∧
  E = 1 ∧ F = 0 ∧ G = 8 ∧ H = 5 :=
sorry

end digits_sum_unique_l1230_123008


namespace sale_in_third_month_l1230_123021

theorem sale_in_third_month 
  (sale1 sale2 sale4 sale5 sale6 : ℕ) 
  (average_sales : ℕ)
  (h1 : sale1 = 5420)
  (h2 : sale2 = 5660)
  (h4 : sale4 = 6350)
  (h5 : sale5 = 6500)
  (h6 : sale6 = 6470)
  (h_avg : average_sales = 6100) : 
  ∃ sale3, sale1 + sale2 + sale3 + sale4 + sale5 + sale6 = average_sales * 6 ∧ sale3 = 6200 :=
by
  sorry

end sale_in_third_month_l1230_123021
