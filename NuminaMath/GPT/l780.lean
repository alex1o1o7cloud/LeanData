import Mathlib

namespace total_fruit_count_l780_78092

-- Define the conditions as variables and equations
def apples := 4 -- based on the final deduction from the solution
def pears := 6 -- calculated from the condition of bananas
def bananas := 9 -- given in the problem

-- State the conditions
axiom h1 : pears = apples + 2
axiom h2 : bananas = pears + 3
axiom h3 : bananas = 9

-- State the proof objective
theorem total_fruit_count : apples + pears + bananas = 19 :=
by
  sorry

end total_fruit_count_l780_78092


namespace math_problem_l780_78029

variables (x y z w p q : ℕ)
variables (x_pos : 0 < x) (y_pos : 0 < y) (z_pos : 0 < z) (w_pos : 0 < w)

theorem math_problem
  (h1 : x^3 = y^2)
  (h2 : z^4 = w^3)
  (h3 : z - x = 22)
  (hx : x = p^2)
  (hy : y = p^3)
  (hz : z = q^3)
  (hw : w = q^4) : w - y = q^4 - p^3 :=
sorry

end math_problem_l780_78029


namespace arithmetic_progression_condition_l780_78017

theorem arithmetic_progression_condition
  (a b c : ℝ) : ∃ (A B : ℤ), A ≠ 0 ∧ B ≠ 0 ∧ (b - a) * B = (c - b) * A := 
by {
  sorry
}

end arithmetic_progression_condition_l780_78017


namespace common_ratio_of_geometric_seq_l780_78087

-- Define the arithmetic sequence
def arithmetic_seq (a d : ℤ) (n : ℕ) : ℤ := a + n * d

-- Define the geometric sequence property
def geometric_seq_property (a2 a3 a6 : ℤ) : Prop :=
  a3 * a3 = a2 * a6

-- State the main theorem
theorem common_ratio_of_geometric_seq (a d : ℤ) (h : ¬d = 0) :
  geometric_seq_property (arithmetic_seq a d 2) (arithmetic_seq a d 3) (arithmetic_seq a d 6) →
  ∃ q : ℤ, q = 3 ∨ q = 1 :=
by
  sorry

end common_ratio_of_geometric_seq_l780_78087


namespace P_sufficient_but_not_necessary_for_Q_l780_78033

def P (x : ℝ) : Prop := abs (2 * x - 3) < 1
def Q (x : ℝ) : Prop := x * (x - 3) < 0

theorem P_sufficient_but_not_necessary_for_Q : 
  (∀ x : ℝ, P x → Q x) ∧ (∃ x : ℝ, Q x ∧ ¬ P x) := 
by
  sorry

end P_sufficient_but_not_necessary_for_Q_l780_78033


namespace sum_nine_terms_of_arithmetic_sequence_l780_78015

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def sum_of_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, S n = (n * (a 0 + a (n - 1))) / 2

theorem sum_nine_terms_of_arithmetic_sequence
  (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h1 : arithmetic_sequence a)
  (h2 : sum_of_first_n_terms a S)
  (h3 : a 5 = 7) :
  S 9 = 63 := by
  sorry

end sum_nine_terms_of_arithmetic_sequence_l780_78015


namespace pow_simplification_l780_78090

theorem pow_simplification :
  9^6 * 3^3 / 27^4 = 27 :=
by
  sorry

end pow_simplification_l780_78090


namespace find_divisor_l780_78084

theorem find_divisor :
  ∃ d : ℕ, (4499 + 1) % d = 0 ∧ d = 2 :=
by
  sorry

end find_divisor_l780_78084


namespace part1_part2_l780_78085

-- Definition of sets A and B
def A (a : ℝ) : Set ℝ := {x | a-1 < x ∧ x < a+1}
def B : Set ℝ := {x | x^2 - 4*x + 3 ≥ 0}

-- Theorem (Ⅰ)
theorem part1 (a : ℝ) : (A a ∩ B = ∅ ∧ A a ∪ B = Set.univ) → a = 2 :=
by
  sorry

-- Theorem (Ⅱ)
theorem part2 (a : ℝ) : (A a ⊆ B ∧ A a ≠ ∅) → (a ≤ 0 ∨ a ≥ 4) :=
by
  sorry

end part1_part2_l780_78085


namespace smallest_b_of_factored_quadratic_l780_78072

theorem smallest_b_of_factored_quadratic (r s : ℕ) (h1 : r * s = 1620) : (r + s) = 84 :=
sorry

end smallest_b_of_factored_quadratic_l780_78072


namespace find_common_chord_l780_78064

-- Define the two circles
def C1 (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 8*y - 8 = 0
def C2 (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 4*y - 2 = 0

-- The common chord is the line we need to prove
def CommonChord (x y : ℝ) : Prop := x + 2*y - 1 = 0

-- The theorem stating that the common chord is the line x + 2*y - 1 = 0
theorem find_common_chord (x y : ℝ) (p : C1 x y ∧ C2 x y) : CommonChord x y :=
sorry

end find_common_chord_l780_78064


namespace sum_of_cubes_is_81720_l780_78095

-- Let n be the smallest of these consecutive even integers.
def smallest_even : Int := 28

-- Assumptions given the conditions
def sum_of_squares (n : Int) : Int := n^2 + (n + 2)^2 + (n + 4)^2

-- The condition provided is that sum of the squares is 2930
lemma sum_of_squares_is_2930 : sum_of_squares smallest_even = 2930 := by
  sorry

-- To prove that the sum of the cubes of these three integers is 81720
def sum_of_cubes (n : Int) : Int := n^3 + (n + 2)^3 + (n + 4)^3

theorem sum_of_cubes_is_81720 : sum_of_cubes smallest_even = 81720 := by
  sorry

end sum_of_cubes_is_81720_l780_78095


namespace restaurant_production_in_june_l780_78049

def cheese_pizzas_per_day (hot_dogs_per_day : ℕ) : ℕ :=
  hot_dogs_per_day + 40

def pepperoni_pizzas_per_day (cheese_pizzas_per_day : ℕ) : ℕ :=
  2 * cheese_pizzas_per_day

def hot_dogs_per_day := 60
def beef_hot_dogs_per_day := 30
def chicken_hot_dogs_per_day := 30
def days_in_june := 30

theorem restaurant_production_in_june :
  (cheese_pizzas_per_day hot_dogs_per_day * days_in_june = 3000) ∧
  (pepperoni_pizzas_per_day (cheese_pizzas_per_day hot_dogs_per_day) * days_in_june = 6000) ∧
  (beef_hot_dogs_per_day * days_in_june = 900) ∧
  (chicken_hot_dogs_per_day * days_in_june = 900) :=
by
  sorry

end restaurant_production_in_june_l780_78049


namespace smallest_positive_integer_l780_78059

theorem smallest_positive_integer (m n : ℤ) : ∃ k : ℕ, k > 0 ∧ (∃ m n : ℤ, k = 5013 * m + 111111 * n) ∧ k = 3 :=
by {
  sorry 
}

end smallest_positive_integer_l780_78059


namespace log_function_domain_l780_78000

noncomputable def domain_of_log_function (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) : Set ℝ :=
  { x : ℝ | x < a }

theorem log_function_domain (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  ∀ x, x ∈ domain_of_log_function a h1 h2 ↔ x < a :=
by
  sorry

end log_function_domain_l780_78000


namespace percentage_of_hindu_boys_l780_78041

-- Define the total number of boys in the school
def total_boys := 700

-- Define the percentage of Muslim boys
def muslim_percentage := 44 / 100

-- Define the percentage of Sikh boys
def sikh_percentage := 10 / 100

-- Define the number of boys from other communities
def other_communities_boys := 126

-- State the main theorem to prove the percentage of Hindu boys
theorem percentage_of_hindu_boys (h1 : total_boys = 700)
                                 (h2 : muslim_percentage = 44 / 100)
                                 (h3 : sikh_percentage = 10 / 100)
                                 (h4 : other_communities_boys = 126) : 
                                 ((total_boys - (total_boys * muslim_percentage + total_boys * sikh_percentage + other_communities_boys)) / total_boys) * 100 = 28 :=
by {
  sorry
}

end percentage_of_hindu_boys_l780_78041


namespace expected_lifetime_flashlight_l780_78040

noncomputable def E (X : ℝ) : ℝ := sorry -- Define E as the expectation operator

variables (ξ η : ℝ) -- Define ξ and η as random variables representing lifetimes of blue and red bulbs
variable (h_exi : E ξ = 2) -- Given condition E ξ = 2

theorem expected_lifetime_flashlight (h_min : ∀ x y : ℝ, min x y ≤ x) :
  E (min ξ η) ≤ 2 :=
by
  sorry

end expected_lifetime_flashlight_l780_78040


namespace vector_parallel_has_value_x_l780_78013

-- Define the vectors a and b
def a : ℝ × ℝ := (3, 2)
def b (x : ℝ) : ℝ × ℝ := (x, 4)

-- Define the parallel condition
def parallel (a b : ℝ × ℝ) : Prop := a.1 * b.2 = a.2 * b.1

-- The theorem statement
theorem vector_parallel_has_value_x :
  ∀ x : ℝ, parallel a (b x) → x = 6 :=
by
  intros x h
  sorry

end vector_parallel_has_value_x_l780_78013


namespace minimize_cost_l780_78080

noncomputable def cost_function (x : ℝ) : ℝ :=
  (1 / 2) * (x + 5)^2 + 1000 / (x + 5)

theorem minimize_cost :
  (∀ x, 2 ≤ x ∧ x ≤ 8 → cost_function x ≥ 150) ∧ cost_function 5 = 150 :=
by
  sorry

end minimize_cost_l780_78080


namespace grasshopper_total_distance_l780_78038

theorem grasshopper_total_distance :
  let initial := 2
  let first_jump := -3
  let second_jump := 8
  let final_jump := -1
  abs (first_jump - initial) + abs (second_jump - first_jump) + abs (final_jump - second_jump) = 25 :=
by
  sorry

end grasshopper_total_distance_l780_78038


namespace remainder_when_divided_by_r_minus_2_l780_78005

-- Define polynomial p(r)
def p (r : ℝ) : ℝ := r ^ 11 - 3

-- The theorem stating the problem
theorem remainder_when_divided_by_r_minus_2 : p 2 = 2045 := by
  sorry

end remainder_when_divided_by_r_minus_2_l780_78005


namespace total_cost_in_dollars_l780_78096

def pencil_price := 20 -- price of one pencil in cents
def tolu_pencils := 3 -- pencils Tolu wants
def robert_pencils := 5 -- pencils Robert wants
def melissa_pencils := 2 -- pencils Melissa wants

theorem total_cost_in_dollars :
  (pencil_price * (tolu_pencils + robert_pencils + melissa_pencils)) / 100 = 2 := 
by
  sorry

end total_cost_in_dollars_l780_78096


namespace number_of_boys_l780_78068

-- Definitions based on conditions
def students_in_class : ℕ := 30
def cups_brought_total : ℕ := 90
def cups_per_boy : ℕ := 5

-- Definition of boys and girls, with a constraint from the conditions
variable (B : ℕ)
def girls_in_class (B : ℕ) : ℕ := 2 * B

-- Properties from the conditions
axiom h1 : B + girls_in_class B = students_in_class
axiom h2 : B * cups_per_boy = cups_brought_total - (students_in_class - B) * 0 -- Assume no girl brought any cup

-- We state the question as a theorem to be proved
theorem number_of_boys (B : ℕ) : B = 10 :=
by
  sorry

end number_of_boys_l780_78068


namespace variance_proof_l780_78034

noncomputable def calculate_mean (scores : List ℝ) : ℝ :=
  (scores.sum / scores.length)

noncomputable def calculate_variance (scores : List ℝ) : ℝ :=
  let mean := calculate_mean scores
  (scores.map (λ x => (x - mean)^2)).sum / scores.length

def scores_A : List ℝ := [8, 6, 9, 5, 10, 7, 4, 7, 9, 5]
def scores_B : List ℝ := [7, 6, 5, 8, 6, 9, 6, 8, 8, 7]

noncomputable def variance_A : ℝ := calculate_variance scores_A
noncomputable def variance_B : ℝ := calculate_variance scores_B

theorem variance_proof :
  variance_A = 3.6 ∧ variance_B = 1.4 ∧ variance_B < variance_A :=
by
  -- proof steps - use sorry to skip the proof
  sorry

end variance_proof_l780_78034


namespace sum_series_a_eq_one_sum_series_b_eq_half_sum_series_c_eq_third_l780_78065

noncomputable def sum_series_a : ℝ :=
∑' n, (1 / (n * (n + 1)))

noncomputable def sum_series_b : ℝ :=
∑' n, (1 / ((n + 1) * (n + 2)))

noncomputable def sum_series_c : ℝ :=
∑' n, (1 / ((n + 2) * (n + 3)))

theorem sum_series_a_eq_one : sum_series_a = 1 := sorry

theorem sum_series_b_eq_half : sum_series_b = 1 / 2 := sorry

theorem sum_series_c_eq_third : sum_series_c = 1 / 3 := sorry

end sum_series_a_eq_one_sum_series_b_eq_half_sum_series_c_eq_third_l780_78065


namespace min_x_squared_plus_y_squared_l780_78077

theorem min_x_squared_plus_y_squared (x y : ℝ) (h : (x + 5) * (y - 5) = 0) : x^2 + y^2 ≥ 50 := by
  sorry

end min_x_squared_plus_y_squared_l780_78077


namespace at_least_12_boxes_l780_78076

theorem at_least_12_boxes (extra_boxes : Nat) : 
  let total_boxes := 12 + extra_boxes
  extra_boxes ≥ 0 → total_boxes ≥ 12 :=
by
  intros
  sorry

end at_least_12_boxes_l780_78076


namespace potato_gun_distance_l780_78098

noncomputable def length_of_football_field_in_yards : ℕ := 200
noncomputable def conversion_factor_yards_to_feet : ℕ := 3
noncomputable def length_of_football_field_in_feet : ℕ := length_of_football_field_in_yards * conversion_factor_yards_to_feet

noncomputable def dog_running_speed : ℕ := 400
noncomputable def time_for_dog_to_fetch_potato : ℕ := 9
noncomputable def total_distance_dog_runs : ℕ := dog_running_speed * time_for_dog_to_fetch_potato

noncomputable def actual_distance_to_potato : ℕ := total_distance_dog_runs / 2

noncomputable def distance_in_football_fields : ℕ := actual_distance_to_potato / length_of_football_field_in_feet

theorem potato_gun_distance :
  distance_in_football_fields = 3 :=
by
  sorry

end potato_gun_distance_l780_78098


namespace find_abc_value_l780_78074

noncomputable def abc_value_condition (a b c : ℝ) : Prop := 
  a + b + c = 4 ∧
  b * c + c * a + a * b = 5 ∧
  a^3 + b^3 + c^3 = 10

theorem find_abc_value (a b c : ℝ) (h : abc_value_condition a b c) : a * b * c = 2 := 
sorry

end find_abc_value_l780_78074


namespace least_real_number_K_l780_78025

theorem least_real_number_K (x y z K : ℝ) (h_cond1 : -2 ≤ x ∧ x ≤ 2) (h_cond2 : -2 ≤ y ∧ y ≤ 2) (h_cond3 : -2 ≤ z ∧ z ≤ 2) (h_eq : x^2 + y^2 + z^2 + x * y * z = 4) :
  (∀ x y z : ℝ, -2 ≤ x ∧ x ≤ 2 ∧ -2 ≤ y ∧ y ≤ 2 ∧ -2 ≤ z ∧ z ≤ 2 ∧ x^2 + y^2 + z^2 + x * y * z = 4 → z * (x * z + y * z + y) / (x * y + y^2 + z^2 + 1) ≤ K) → K = 4 / 3 :=
by
  sorry

end least_real_number_K_l780_78025


namespace find_N_l780_78009

theorem find_N : ∃ (N : ℤ), N > 0 ∧ (36^2 * 60^2 = 30^2 * N^2) ∧ (N = 72) :=
by
  sorry

end find_N_l780_78009


namespace impossible_four_teams_tie_possible_three_teams_tie_l780_78047

-- Definitions for the conditions
def num_teams : ℕ := 4
def num_matches : ℕ := (num_teams * (num_teams - 1)) / 2
def total_possible_outcomes : ℕ := 2^num_matches
def winning_rate : ℚ := 1 / 2

-- Problem 1: It is impossible for exactly four teams to tie for first place.
theorem impossible_four_teams_tie :
  ¬ ∃ (score : ℕ), (∀ (team : ℕ) (h : team < num_teams), team = score ∧
                     (num_teams * score = num_matches / 2 ∧
                      num_teams * score + num_matches / 2 = num_matches)) := sorry

-- Problem 2: It is possible for exactly three teams to tie for first place.
theorem possible_three_teams_tie :
  ∃ (score : ℕ), (∃ (teamA teamB teamC teamD : ℕ),
  (teamA < num_teams ∧ teamB < num_teams ∧ teamC < num_teams ∧ teamD <num_teams ∧ teamA ≠ teamB ∧ teamA ≠ teamC ∧ teamA ≠ teamD ∧ 
  teamB ≠ teamC ∧ teamB ≠ teamD ∧ teamC ≠ teamD)) ∧
  (teamA = score ∧ teamB = score ∧ teamC = score ∧ teamD = 0) := sorry

end impossible_four_teams_tie_possible_three_teams_tie_l780_78047


namespace number_of_cut_red_orchids_l780_78024

variable (initial_red_orchids added_red_orchids final_red_orchids : ℕ)

-- Conditions
def initial_red_orchids_in_vase (initial_red_orchids : ℕ) : Prop :=
  initial_red_orchids = 9

def final_red_orchids_in_vase (final_red_orchids : ℕ) : Prop :=
  final_red_orchids = 15

-- Proof statement
theorem number_of_cut_red_orchids (initial_red_orchids added_red_orchids final_red_orchids : ℕ)
  (h1 : initial_red_orchids_in_vase initial_red_orchids) 
  (h2 : final_red_orchids_in_vase final_red_orchids) :
  final_red_orchids = initial_red_orchids + added_red_orchids → added_red_orchids = 6 := by
  simp [initial_red_orchids_in_vase, final_red_orchids_in_vase] at *
  sorry

end number_of_cut_red_orchids_l780_78024


namespace find_f_2547_l780_78044

theorem find_f_2547 (f : ℚ → ℚ)
  (h1 : ∀ x y : ℚ, f (x + y) = f x + f y + 2547)
  (h2 : f 2004 = 2547) :
  f 2547 = 2547 :=
sorry

end find_f_2547_l780_78044


namespace aunt_masha_butter_usage_l780_78057

theorem aunt_masha_butter_usage
  (x y : ℝ)
  (h1 : x + 10 * y = 600)
  (h2 : x = 5 * y) :
  (2 * x + 2 * y = 480) := 
by
  sorry

end aunt_masha_butter_usage_l780_78057


namespace jorges_total_yield_l780_78037

def total_yield (good_acres clay_acres : ℕ) (good_yield clay_yield : ℕ) : ℕ :=
  good_acres * good_yield + clay_acres * clay_yield / 2

theorem jorges_total_yield :
  let acres := 60
  let good_yield_per_acre := 400
  let clay_yield_per_acre := good_yield_per_acre / 2
  let good_acres := 2 * acres / 3
  let clay_acres := acres / 3
  total_yield good_acres clay_acres good_yield_per_acre clay_yield_per_acre = 20000 :=
by
  sorry

end jorges_total_yield_l780_78037


namespace rectangles_single_row_7_rectangles_grid_7_4_l780_78043

def rectangles_in_single_row (n : ℕ) : ℕ :=
  (n * (n + 1)) / 2

theorem rectangles_single_row_7 :
  rectangles_in_single_row 7 = 28 :=
by
  -- Add the proof here
  sorry

def rectangles_in_grid (rows cols : ℕ) : ℕ :=
  ((cols + 1) * cols / 2) * ((rows + 1) * rows / 2)

theorem rectangles_grid_7_4 :
  rectangles_in_grid 4 7 = 280 :=
by
  -- Add the proof here
  sorry

end rectangles_single_row_7_rectangles_grid_7_4_l780_78043


namespace trail_mix_total_weight_l780_78081

noncomputable def peanuts : ℝ := 0.16666666666666666
noncomputable def chocolate_chips : ℝ := 0.16666666666666666
noncomputable def raisins : ℝ := 0.08333333333333333

theorem trail_mix_total_weight :
  peanuts + chocolate_chips + raisins = 0.41666666666666663 :=
by
  unfold peanuts chocolate_chips raisins
  sorry

end trail_mix_total_weight_l780_78081


namespace oliver_learning_vowels_l780_78091

theorem oliver_learning_vowels : 
  let learn := 5
  let rest_days (n : Nat) := n
  let total_days :=
    (learn + rest_days 1) + -- For 'A'
    (learn + rest_days 2) + -- For 'E'
    (learn + rest_days 3) + -- For 'I'
    (learn + rest_days 4) + -- For 'O'
    (rest_days 5 + learn)  -- For 'U' and 'Y'
  total_days = 40 :=
by
  sorry

end oliver_learning_vowels_l780_78091


namespace muffins_in_each_pack_l780_78075

-- Define the conditions as constants
def total_amount_needed : ℕ := 120
def price_per_muffin : ℕ := 2
def number_of_cases : ℕ := 5
def packs_per_case : ℕ := 3

-- Define the theorem to prove
theorem muffins_in_each_pack :
  (total_amount_needed / price_per_muffin) / (number_of_cases * packs_per_case) = 4 :=
by
  sorry

end muffins_in_each_pack_l780_78075


namespace son_age_18_l780_78012

theorem son_age_18 (F S : ℤ) (h1 : F = S + 20) (h2 : F + 2 = 2 * (S + 2)) : S = 18 :=
by
  sorry

end son_age_18_l780_78012


namespace import_tax_calculation_l780_78094

def import_tax_rate : ℝ := 0.07
def excess_value_threshold : ℝ := 1000
def total_value_item : ℝ := 2610
def correct_import_tax : ℝ := 112.7

theorem import_tax_calculation :
  (total_value_item - excess_value_threshold) * import_tax_rate = correct_import_tax :=
by
  sorry

end import_tax_calculation_l780_78094


namespace original_number_is_45_l780_78060

theorem original_number_is_45 (x : ℕ) (h : x - 30 = x / 3) : x = 45 :=
by {
  sorry
}

end original_number_is_45_l780_78060


namespace total_items_children_carry_l780_78014

theorem total_items_children_carry 
  (pieces_per_pizza : ℕ) (number_of_fourthgraders : ℕ) (pizza_per_fourthgrader : ℕ) 
  (pepperoni_per_pizza : ℕ) (mushrooms_per_pizza : ℕ) (olives_per_pizza : ℕ) 
  (total_pizzas : ℕ) (total_pieces_of_pizza : ℕ) (total_pepperoni : ℕ) (total_mushrooms : ℕ) 
  (total_olives : ℕ) (total_toppings : ℕ) (total_items : ℕ) : 
  pieces_per_pizza = 6 →
  number_of_fourthgraders = 10 →
  pizza_per_fourthgrader = 20 →
  pepperoni_per_pizza = 5 →
  mushrooms_per_pizza = 3 →
  olives_per_pizza = 8 →
  total_pizzas = number_of_fourthgraders * pizza_per_fourthgrader →
  total_pieces_of_pizza = total_pizzas * pieces_per_pizza →
  total_pepperoni = total_pizzas * pepperoni_per_pizza →
  total_mushrooms = total_pizzas * mushrooms_per_pizza →
  total_olives = total_pizzas * olives_per_pizza →
  total_toppings = total_pepperoni + total_mushrooms + total_olives →
  total_items = total_pieces_of_pizza + total_toppings →
  total_items = 4400 :=
by
  sorry

end total_items_children_carry_l780_78014


namespace q_is_false_l780_78061

-- Given conditions
variables (p q : Prop)
axiom h1 : ¬ (p ∧ q)
axiom h2 : ¬ ¬ p

-- Proof that q is false
theorem q_is_false : q = False :=
by
  sorry

end q_is_false_l780_78061


namespace max_value_expr_l780_78006

variable (x y z : ℝ)

theorem max_value_expr (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) :
  (∃ a, ∀ x y z, (a = (x*y + y*z) / (x^2 + y^2 + z^2)) ∧ a ≤ (Real.sqrt 2) / 2) ∧
  (∃ x' y' z', (x' > 0) ∧ (y' > 0) ∧ (z' > 0) ∧ ((x'*y' + y'*z') / (x'^2 + y'^2 + z'^2) = (Real.sqrt 2) / 2)) :=
by
  sorry

end max_value_expr_l780_78006


namespace original_number_l780_78028

theorem original_number (x : ℤ) (h : x / 2 = 9) : x = 18 := by
  sorry

end original_number_l780_78028


namespace f_neg_9_over_2_f_in_7_8_l780_78052

noncomputable def f (x : ℝ) : ℝ :=
if 0 ≤ x ∧ x < 1 then x / (x + 1) else sorry

theorem f_neg_9_over_2 (h : ∀ x : ℝ, f (x + 2) = f x) (h_odd : ∀ x : ℝ, f (-x) = -f x) : 
  f (-9 / 2) = -1 / 3 :=
by
  have h_period : f (-9 / 2) = f (-1 / 2) := by
    sorry  -- Using periodicity property
  have h_odd1 : f (-1 / 2) = -f (1 / 2) := by
    sorry  -- Using odd function property
  have h_def : f (1 / 2) = 1 / 3 := by
    sorry  -- Using the definition of f(x) for x in [0, 1)
  rw [h_period, h_odd1, h_def]
  norm_num

theorem f_in_7_8 (h : ∀ x : ℝ, f (x + 2) = f x) (h_odd : ∀ x : ℝ, f (-x) = -f x) :
  ∀ x : ℝ, (7 < x ∧ x ≤ 8) → f x = - (x - 8) / (x - 9) :=
by
  intro x hx
  have h_period : f x = f (x - 8) := by
    sorry  -- Using periodicity property
  sorry  -- Apply the negative intervals and substitution to achieve the final form

end f_neg_9_over_2_f_in_7_8_l780_78052


namespace total_amount_l780_78071

variable (x y z : ℝ)

def condition1 : Prop := y = 0.45 * x
def condition2 : Prop := z = 0.30 * x
def condition3 : Prop := y = 36

theorem total_amount (h1 : condition1 x y)
                     (h2 : condition2 x z)
                     (h3 : condition3 y) :
  x + y + z = 140 :=
by
  sorry

end total_amount_l780_78071


namespace exists_disk_of_radius_one_containing_1009_points_l780_78039

theorem exists_disk_of_radius_one_containing_1009_points
  (points : Fin 2017 → ℝ × ℝ)
  (h : ∀ (a b c : Fin 2017), (dist (points a) (points b) < 1) ∨ (dist (points b) (points c) < 1) ∨ (dist (points c) (points a) < 1)) :
  ∃ (center : ℝ × ℝ), ∃ (sub_points : Finset (Fin 2017)), sub_points.card ≥ 1009 ∧ ∀ p ∈ sub_points, dist (center) (points p) ≤ 1 :=
sorry

end exists_disk_of_radius_one_containing_1009_points_l780_78039


namespace initial_cost_of_article_l780_78097

variable (P : ℝ)

theorem initial_cost_of_article (h1 : 0.70 * P = 2100) (h2 : 0.50 * (0.70 * P) = 1050) : P = 3000 :=
by
  sorry

end initial_cost_of_article_l780_78097


namespace not_divides_l780_78004

theorem not_divides (d a n : ℕ) (h1 : 3 ≤ d) (h2 : d ≤ 2^(n+1)) : ¬ d ∣ a^(2^n) + 1 := 
sorry

end not_divides_l780_78004


namespace prime_square_minus_one_divisible_by_twelve_l780_78093

theorem prime_square_minus_one_divisible_by_twelve (p : ℕ) (hp : Nat.Prime p) (hp_gt : p > 3) : 12 ∣ (p^2 - 1) :=
by
  sorry

end prime_square_minus_one_divisible_by_twelve_l780_78093


namespace original_number_divisible_l780_78011

theorem original_number_divisible (n : ℕ) (h : (n - 8) % 20 = 0) : n = 28 := 
by
  sorry

end original_number_divisible_l780_78011


namespace recommendation_plans_count_l780_78045

theorem recommendation_plans_count :
  let total_students := 7
  let sports_talents := 2
  let artistic_talents := 2
  let other_talents := 3
  let recommend_count := 4
  let condition_sports := recommend_count >= 1
  let condition_artistic := recommend_count >= 1
  (condition_sports ∧ condition_artistic) → 
  ∃ (n : ℕ), n = 25 := sorry

end recommendation_plans_count_l780_78045


namespace rugged_terrain_distance_ratio_l780_78063

theorem rugged_terrain_distance_ratio (D k : ℝ) 
  (hD : D > 0) 
  (hk : k > 0) 
  (v_M v_P : ℝ) 
  (hm : v_M = 2 * k) 
  (hp : v_P = 3 * k)
  (v_Mr v_Pr : ℝ) 
  (hmr : v_Mr = k) 
  (hpr : v_Pr = 3 * k / 2) :
  ∀ (x y a b : ℝ), (x + y = D / 2) → (a + b = D / 2) → (y + b = 2 * D / 3) →
  (x / (2 * k) + y / k = a / (3 * k) + 2 * b / (3 * k)) → 
  (y / b = 1 / 3) := 
sorry

end rugged_terrain_distance_ratio_l780_78063


namespace obtuse_triangle_area_side_l780_78027

theorem obtuse_triangle_area_side (a b : ℝ) (C : ℝ) 
  (h1 : a = 8) 
  (h2 : C = 150 * (π / 180)) -- converting degrees to radians
  (h3 : 1 / 2 * a * b * Real.sin C = 24) : 
  b = 12 :=
by sorry

end obtuse_triangle_area_side_l780_78027


namespace impossible_transformation_l780_78032

variable (G : Type) [Group G]

/-- Initial word represented by 2003 'a's followed by 'b' --/
def initial_word := "aaa...ab"

/-- Transformed word represented by 'b' followed by 2003 'a's --/
def transformed_word := "baaa...a"

/-- Hypothetical group relations derived from transformations --/
axiom aba_to_b (a b : G) : (a * b * a = b)
axiom bba_to_a (a b : G) : (b * b * a = a)

/-- Impossible transformation proof --/
theorem impossible_transformation (a b : G) : 
  (initial_word = transformed_word) → False := by
  sorry

end impossible_transformation_l780_78032


namespace joan_spent_on_toys_l780_78062

theorem joan_spent_on_toys :
  let toy_cars := 14.88
  let toy_trucks := 5.86
  toy_cars + toy_trucks = 20.74 :=
by
  let toy_cars := 14.88
  let toy_trucks := 5.86
  sorry

end joan_spent_on_toys_l780_78062


namespace find_prime_triplet_l780_78056

open Nat

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, 2 ≤ m → m ≤ n / 2 → (m ∣ n) → False

theorem find_prime_triplet :
  ∃ p q r : ℕ, is_prime p ∧ is_prime q ∧ is_prime r ∧ 
  (p + q = r) ∧ 
  (∃ k : ℕ, (r - p) * (q - p) - 27 * p = k * k) ∧ 
  (p = 2 ∧ q = 29 ∧ r = 31) := by
  sorry

end find_prime_triplet_l780_78056


namespace sum_of_roots_l780_78010

noncomputable def equation (x : ℝ) := 2 * (x^2 + 1 / x^2) - 3 * (x + 1 / x) = 1

theorem sum_of_roots (r s : ℝ) (hr : equation r) (hs : equation s) (hne : r ≠ s) :
  r + s = -5 / 2 :=
sorry

end sum_of_roots_l780_78010


namespace property_tax_difference_correct_l780_78003

-- Define the tax rates for different ranges
def tax_rate (value : ℕ) : ℝ :=
  if value ≤ 10000 then 0.05
  else if value ≤ 20000 then 0.075
  else if value ≤ 30000 then 0.10
  else 0.125

-- Define the progressive tax calculation for a given assessed value
def calculate_tax (value : ℕ) : ℝ :=
  if value ≤ 10000 then value * 0.05
  else if value ≤ 20000 then 10000 * 0.05 + (value - 10000) * 0.075
  else if value <= 30000 then 10000 * 0.05 + 10000 * 0.075 + (value - 20000) * 0.10
  else 10000 * 0.05 + 10000 * 0.075 + 10000 * 0.10 + (value - 30000) * 0.125

-- Define the initial and new assessed values
def initial_value : ℕ := 20000
def new_value : ℕ := 28000

-- Define the difference in tax calculation
def tax_difference : ℝ := calculate_tax new_value - calculate_tax initial_value

theorem property_tax_difference_correct : tax_difference = 550 := by
  sorry

end property_tax_difference_correct_l780_78003


namespace modules_count_l780_78002

theorem modules_count (x y: ℤ) (hx: 10 * x + 35 * y = 450) (hy: x + y = 11) : y = 10 :=
by
  sorry

end modules_count_l780_78002


namespace intersection_A_B_l780_78048

def A : Set ℝ := {y | ∃ x : ℝ, y = x^2 + 1 / (x^2 + 1) }
def B : Set ℝ := {x | 3 * x - 2 < 7}

theorem intersection_A_B : A ∩ B = Set.Ico 1 3 := 
by
  sorry

end intersection_A_B_l780_78048


namespace kelly_grade_correct_l780_78099

variable (Jenny Jason Bob Kelly : ℕ)

def jenny_grade : ℕ := 95
def jason_grade := jenny_grade - 25
def bob_grade := jason_grade / 2
def kelly_grade := bob_grade + (bob_grade / 5)  -- 20% of Bob's grade is (Bob's grade * 0.20), which is the same as (Bob's grade / 5)

theorem kelly_grade_correct : kelly_grade = 42 :=
by
  sorry

end kelly_grade_correct_l780_78099


namespace sequences_properties_l780_78078

-- Definitions based on the problem conditions
def geom_sequence (a : ℕ → ℕ) := ∃ q : ℕ, a 1 = 2 ∧ a 3 = 18 ∧ ∀ n, a (n + 1) = a n * q
def arith_sequence (b : ℕ → ℕ) := b 1 = 2 ∧ ∃ d : ℕ, ∀ n, b (n + 1) = b n + d
def condition (a : ℕ → ℕ) (b : ℕ → ℕ) := a 1 + a 2 + a 3 > 20 ∧ a 1 + a 2 + a 3 = b 1 + b 2 + b 3 + b 4

-- Proof statement: proving the general term of the geometric sequence and the sum of the arithmetic sequence
theorem sequences_properties (a : ℕ → ℕ) (b : ℕ → ℕ) :
  geom_sequence a → arith_sequence b → condition a b →
  (∀ n, a n = 2 * 3^(n - 1)) ∧ (∀ n, S_n = 3 / 2 * n^2 + 1 / 2 * n) :=
by
  sorry

end sequences_properties_l780_78078


namespace negation_proposition_l780_78031

open Real

theorem negation_proposition (h : ∀ x : ℝ, x^2 - 2*x - 1 > 0) :
  ¬ (∀ x : ℝ, x^2 - 2*x - 1 > 0) = ∃ x_0 : ℝ, x_0^2 - 2*x_0 - 1 ≤ 0 :=
by 
  sorry

end negation_proposition_l780_78031


namespace tangent_line_value_l780_78073

theorem tangent_line_value {k : ℝ} 
  (h1 : ∃ x y : ℝ, x^2 + y^2 - 6*y + 8 = 0) 
  (h2 : ∃ P Q : ℝ, x^2 + y^2 - 6*y + 8 = 0 ∧ Q = k * P)
  (h3 : P * k < 0 ∧ P < 0 ∧ Q > 0) : 
  k = -2 * Real.sqrt 2 :=
sorry

end tangent_line_value_l780_78073


namespace factorize_P_l780_78026

noncomputable def P (y : ℝ) : ℝ :=
  (16 * y ^ 7 - 36 * y ^ 5 + 8 * y) - (4 * y ^ 7 - 12 * y ^ 5 - 8 * y)

theorem factorize_P (y : ℝ) : P y = 8 * y * (3 * y ^ 6 - 6 * y ^ 4 + 4) :=
  sorry

end factorize_P_l780_78026


namespace number_drawn_from_3rd_group_l780_78007

theorem number_drawn_from_3rd_group {n k : ℕ} (pop_size : ℕ) (sample_size : ℕ) 
  (drawn_from_group : ℕ → ℕ) (group_id : ℕ) (num_in_13th_group : ℕ) : 
  pop_size = 160 → 
  sample_size = 20 → 
  (∀ i, 1 ≤ i ∧ i ≤ sample_size → ∃ j, group_id = i ∧ 
    (j = (i - 1) * (pop_size / sample_size) + drawn_from_group 1)) → 
  num_in_13th_group = 101 → 
  drawn_from_group 3 = 21 := 
by
  intros hp hs hg h13
  sorry

end number_drawn_from_3rd_group_l780_78007


namespace steven_more_peaches_than_apples_l780_78008

-- Definitions
def apples_steven := 11
def peaches_steven := 18

-- Theorem statement
theorem steven_more_peaches_than_apples : (peaches_steven - apples_steven) = 7 := by 
  sorry

end steven_more_peaches_than_apples_l780_78008


namespace circles_internally_tangent_l780_78053

theorem circles_internally_tangent :
  ∀ (x y : ℝ),
  (x^2 + y^2 - 6 * x + 4 * y + 12 = 0) ∧ (x^2 + y^2 - 14 * x - 2 * y + 14 = 0) →
  ∃ (C1 C2 : ℝ × ℝ) (r1 r2 : ℝ),
  C1 = (3, -2) ∧ r1 = 1 ∧
  C2 = (7, 1) ∧ r2 = 6 ∧
  dist C1 C2 = r2 - r1 :=
by
  sorry

end circles_internally_tangent_l780_78053


namespace custom_mul_4_3_l780_78042

-- Define the binary operation a*b = a^2 - ab + b^2
def custom_mul (a b : ℕ) : ℕ := a^2 - a*b + b^2

-- State the theorem to prove that 4 * 3 = 13
theorem custom_mul_4_3 : custom_mul 4 3 = 13 := by
  sorry -- Proof will be filled in here

end custom_mul_4_3_l780_78042


namespace cost_increase_per_scrap_rate_l780_78030

theorem cost_increase_per_scrap_rate (x : ℝ) :
  ∀ x Δx, y = 56 + 8 * x → Δx = 1 → y + Δy = 56 + 8 * (x + Δx) → Δy = 8 :=
by
  sorry

end cost_increase_per_scrap_rate_l780_78030


namespace monomial_combined_l780_78035

theorem monomial_combined (n m : ℕ) (h₁ : 2 = n) (h₂ : m = 4) : n^m = 16 := by
  sorry

end monomial_combined_l780_78035


namespace penthouse_floors_l780_78019

theorem penthouse_floors (R P : ℕ) (h1 : R + P = 23) (h2 : 12 * R + 2 * P = 256) : P = 2 :=
by
  sorry

end penthouse_floors_l780_78019


namespace problem_statement_l780_78082

-- Definitions
def div_remainder (a b : ℕ) : ℕ × ℕ :=
  (a / b, a % b)

-- Conditions and question as Lean structures
def condition := ∀ (a b k : ℕ), k ≠ 0 → div_remainder (a * k) (b * k) = (a / b, (a % b) * k)
def question := div_remainder 4900 600 = div_remainder 49 6

-- Theorem stating the problem's conclusion
theorem problem_statement (cond : condition) : ¬question :=
by
  sorry

end problem_statement_l780_78082


namespace maximum_ab_l780_78022

theorem maximum_ab (a b c : ℝ) (h1 : a + b + c = 4) (h2 : 3 * a + 2 * b - c = 0) : 
  ab <= 1/3 := 
by 
  sorry

end maximum_ab_l780_78022


namespace total_cans_given_away_l780_78083

noncomputable def total_cans_initial : ℕ := 2000
noncomputable def cans_taken_first_day : ℕ := 500
noncomputable def restocked_first_day : ℕ := 1500
noncomputable def people_second_day : ℕ := 1000
noncomputable def cans_per_person_second_day : ℕ := 2
noncomputable def restocked_second_day : ℕ := 3000

theorem total_cans_given_away :
  (cans_taken_first_day + (people_second_day * cans_per_person_second_day) = 2500) :=
by
  sorry

end total_cans_given_away_l780_78083


namespace perfect_even_multiples_of_3_under_3000_l780_78070

theorem perfect_even_multiples_of_3_under_3000 :
  ∃ n : ℕ, n = 9 ∧ ∀ (k : ℕ), (36 * k^2 < 3000) → (36 * k^2) % 2 = 0 ∧ (36 * k^2) % 3 = 0 ∧ ∃ m : ℕ, m^2 = 36 * k^2 :=
by
  sorry

end perfect_even_multiples_of_3_under_3000_l780_78070


namespace number_of_pencils_l780_78020

variable (P L : ℕ)

-- Conditions
def condition1 : Prop := P / L = 5 / 6
def condition2 : Prop := L = P + 5

-- Statement to prove
theorem number_of_pencils (h1 : condition1 P L) (h2 : condition2 P L) : L = 30 :=
  sorry

end number_of_pencils_l780_78020


namespace prod_div_sum_le_square_l780_78050

theorem prod_div_sum_le_square (m n : ℕ) (h : (m * n) ∣ (m + n)) : m + n ≤ n^2 := sorry

end prod_div_sum_le_square_l780_78050


namespace similar_triangles_l780_78079

theorem similar_triangles (y : ℝ) 
  (h₁ : 12 / y = 9 / 6) : y = 8 :=
by {
  -- solution here
  -- currently, we just provide the theorem statement as requested
  sorry
}

end similar_triangles_l780_78079


namespace minimum_value_expr_l780_78069

variable (a b : ℝ)

theorem minimum_value_expr (h1 : 0 < a) (h2 : 0 < b) : 
  (a + 1 / b) * (a + 1 / b - 1009) + (b + 1 / a) * (b + 1 / a - 1009) ≥ -509004.5 :=
sorry

end minimum_value_expr_l780_78069


namespace production_today_is_correct_l780_78001

theorem production_today_is_correct (n : ℕ) (P : ℕ) (T : ℕ) (average_daily_production : ℕ) (new_average_daily_production : ℕ) (h1 : n = 3) (h2 : average_daily_production = 70) (h3 : new_average_daily_production = 75) (h4 : P = n * average_daily_production) (h5 : P + T = (n + 1) * new_average_daily_production) : T = 90 :=
by
  sorry

end production_today_is_correct_l780_78001


namespace least_possible_number_of_coins_in_jar_l780_78023

theorem least_possible_number_of_coins_in_jar (n : ℕ) : 
  (n % 7 = 3) → (n % 4 = 1) → (n % 6 = 5) → n = 17 :=
by
  sorry

end least_possible_number_of_coins_in_jar_l780_78023


namespace bakery_gives_away_30_doughnuts_at_end_of_day_l780_78067

def boxes_per_day (total_doughnuts doughnuts_per_box : ℕ) : ℕ :=
  total_doughnuts / doughnuts_per_box

def leftover_boxes (total_boxes sold_boxes : ℕ) : ℕ :=
  total_boxes - sold_boxes

def doughnuts_given_away (leftover_boxes doughnuts_per_box : ℕ) : ℕ :=
  leftover_boxes * doughnuts_per_box

theorem bakery_gives_away_30_doughnuts_at_end_of_day 
  (total_doughnuts doughnuts_per_box sold_boxes : ℕ) 
  (H1 : total_doughnuts = 300) (H2 : doughnuts_per_box = 10) (H3 : sold_boxes = 27) : 
  doughnuts_given_away (leftover_boxes (boxes_per_day total_doughnuts doughnuts_per_box) sold_boxes) doughnuts_per_box = 30 :=
by
  sorry

end bakery_gives_away_30_doughnuts_at_end_of_day_l780_78067


namespace cats_more_than_spinsters_l780_78036

def ratio (a b : ℕ) := ∃ k : ℕ, a = b * k

theorem cats_more_than_spinsters (S C : ℕ) (h1 : ratio 2 9) (h2 : S = 12) (h3 : 2 * C = 108) :
  C - S = 42 := by 
  sorry

end cats_more_than_spinsters_l780_78036


namespace prism_width_calculation_l780_78088

theorem prism_width_calculation 
  (l h d : ℝ) 
  (h_l : l = 4) 
  (h_h : h = 10) 
  (h_d : d = 14) :
  ∃ w : ℝ, w = 4 * Real.sqrt 5 ∧ (l^2 + w^2 + h^2 = d^2) := 
by
  use 4 * Real.sqrt 5
  sorry

end prism_width_calculation_l780_78088


namespace calculate_fourth_quarter_shots_l780_78089

-- Definitions based on conditions
def first_quarters_shots : ℕ := 20
def first_quarters_successful_shots : ℕ := 12
def third_quarter_shots : ℕ := 10
def overall_accuracy : ℚ := 46 / 100
def total_shots (n : ℕ) : ℕ := first_quarters_shots + third_quarter_shots + n
def total_successful_shots (n : ℕ) : ℚ := first_quarters_successful_shots + 3 + (4 / 10 * n)


-- Main theorem to prove
theorem calculate_fourth_quarter_shots (n : ℕ) (h : (total_successful_shots n) / (total_shots n) = overall_accuracy) : 
  n = 20 :=
by {
  sorry
}

end calculate_fourth_quarter_shots_l780_78089


namespace scientific_notation_3080000_l780_78051

theorem scientific_notation_3080000 : (∃ (a : ℝ) (b : ℤ), 1 ≤ a ∧ a < 10 ∧ (3080000 : ℝ) = a * 10^b) ∧ (3080000 : ℝ) = 3.08 * 10^6 :=
by
  sorry

end scientific_notation_3080000_l780_78051


namespace four_circles_max_parts_l780_78086

theorem four_circles_max_parts (n : ℕ) (h1 : ∀ n, n = 1 ∨ n = 2 ∨ n = 3 → ∃ k, k = 2^n) :
    n = 4 → ∃ k, k = 14 :=
by
  sorry

end four_circles_max_parts_l780_78086


namespace find_y_coordinate_of_Q_l780_78054

noncomputable def y_coordinate_of_Q 
  (P R T S : ℝ × ℝ) (Q : ℝ × ℝ) (areaPentagon areaSquare : ℝ) : Prop :=
  P = (0, 0) ∧ 
  R = (0, 5) ∧ 
  T = (6, 0) ∧ 
  S = (6, 5) ∧ 
  Q.fst = 3 ∧ 
  areaSquare = 25 ∧ 
  areaPentagon = 50 ∧ 
  (1 / 2) * 6 * (Q.snd - 5) + areaSquare = areaPentagon

theorem find_y_coordinate_of_Q : 
  ∃ y_Q : ℝ, y_coordinate_of_Q (0, 0) (0, 5) (6, 0) (6, 5) (3, y_Q) 50 25 ∧ y_Q = 40 / 3 :=
sorry

end find_y_coordinate_of_Q_l780_78054


namespace coeff_x2_expansion_l780_78021

theorem coeff_x2_expansion (n r : ℕ) (a b : ℤ) :
  n = 5 → a = 1 → b = 2 → r = 2 →
  (Nat.choose n r) * (a^(n - r)) * (b^r) = 40 :=
by
  intros Hn Ha Hb Hr
  rw [Hn, Ha, Hb, Hr]
  simp
  sorry

end coeff_x2_expansion_l780_78021


namespace initial_weight_of_fish_l780_78046

theorem initial_weight_of_fish (B F : ℝ) 
  (h1 : B + F = 54) 
  (h2 : B + F / 2 = 29) : 
  F = 50 := 
sorry

end initial_weight_of_fish_l780_78046


namespace y_coordinate_of_second_point_l780_78066

variable {m n k : ℝ}

theorem y_coordinate_of_second_point (h1 : m = 2 * n + 5) (h2 : k = 0.5) : (n + k) = n + 0.5 := 
by
  sorry

end y_coordinate_of_second_point_l780_78066


namespace find_largest_number_l780_78055

theorem find_largest_number (a b c : ℝ) (h1 : a + b + c = 100) (h2 : c - b = 10) (h3 : b - a = 5) : c = 41.67 := 
sorry

end find_largest_number_l780_78055


namespace packets_of_gum_is_eight_l780_78016

-- Given conditions
def pieces_left : ℕ := 2
def pieces_chewed : ℕ := 54
def pieces_per_packet : ℕ := 7

-- Given he chews all the gum except for pieces_left pieces, and chews pieces_chewed pieces at once
def total_pieces_of_gum (pieces_chewed pieces_left : ℕ) : ℕ :=
  pieces_chewed + pieces_left

-- Calculate the number of packets
def number_of_packets (total_pieces pieces_per_packet : ℕ) : ℕ :=
  total_pieces / pieces_per_packet

-- The final theorem asserting the number of packets is 8
theorem packets_of_gum_is_eight : number_of_packets (total_pieces_of_gum pieces_chewed pieces_left) pieces_per_packet = 8 :=
  sorry

end packets_of_gum_is_eight_l780_78016


namespace boulder_splash_width_l780_78018

theorem boulder_splash_width :
  (6 * (1/4) + 3 * (1 / 2) + 2 * b = 7) -> b = 2 := by
  sorry

end boulder_splash_width_l780_78018


namespace mixed_number_eval_l780_78058

theorem mixed_number_eval :
  -|-(18/5 : ℚ)| - (- (12 /5 : ℚ)) + (4/5 : ℚ) = - (2 / 5 : ℚ) :=
by
  sorry

end mixed_number_eval_l780_78058
