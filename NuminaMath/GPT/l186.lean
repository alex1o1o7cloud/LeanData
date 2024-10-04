import Mathlib

namespace find_a_l186_186534

theorem find_a (a : ℝ) (x : ℝ) (h : x^2 + a * x + 4 = (x + 2)^2) : a = 4 :=
sorry

end find_a_l186_186534


namespace minimum_value_of_f_on_interval_l186_186062

noncomputable def f (x a : ℝ) : ℝ := -x^3 + 3 * x^2 + 9 * x + a

theorem minimum_value_of_f_on_interval (a : ℝ) (h : ∀ x ∈ Set.Icc (-2 : ℝ) 2, f x a ≤ 20) :
  a = -2 → ∃ min_val, min_val = -7 :=
by
  sorry

end minimum_value_of_f_on_interval_l186_186062


namespace at_least_one_zero_l186_186969

theorem at_least_one_zero (a b : ℤ) : (¬ (a ≠ 0) ∨ ¬ (b ≠ 0)) ↔ (a = 0 ∨ b = 0) :=
by
  sorry

end at_least_one_zero_l186_186969


namespace sue_receives_correct_answer_l186_186641

theorem sue_receives_correct_answer (x : ℕ) (y : ℕ) (z : ℕ) (h1 : y = 3 * (x + 2)) (h2 : z = 3 * (y - 2)) (hx : x = 6) : z = 66 :=
by
  sorry

end sue_receives_correct_answer_l186_186641


namespace seq1_general_formula_seq2_general_formula_l186_186482

-- Sequence (1): Initial condition and recurrence relation
def seq1 (a : ℕ → ℕ) : Prop :=
  a 1 = 0 ∧ ∀ n, a (n + 1) = a n + (2 * n - 1)

-- Proving the general formula for sequence (1)
theorem seq1_general_formula (a : ℕ → ℕ) (n : ℕ) (h : seq1 a) :
  a n = (n - 1) ^ 2 :=
sorry

-- Sequence (2): Initial condition and recurrence relation
def seq2 (a : ℕ → ℕ) : Prop :=
  a 1 = 3 ∧ ∀ n, a (n + 1) = 3 * a n

-- Proving the general formula for sequence (2)
theorem seq2_general_formula (a : ℕ → ℕ) (n : ℕ) (h : seq2 a) :
  a n = 3 ^ n :=
sorry

end seq1_general_formula_seq2_general_formula_l186_186482


namespace top_cell_pos_cases_l186_186079

-- Define the rule for the cell sign propagation
def cell_sign (a b : ℤ) : ℤ := 
  if a = b then 1 else -1

-- The pyramid height
def pyramid_height : ℕ := 5

-- Define the final condition for the top cell in the pyramid to be "+"
def top_cell_sign (a b c d e : ℤ) : ℤ :=
  a * b * c * d * e

-- Define the proof statement
theorem top_cell_pos_cases :
  (∃ a b c d e : ℤ,
    (a = 1 ∨ a = -1) ∧
    (b = 1 ∨ b = -1) ∧
    (c = 1 ∨ c = -1) ∧
    (d = 1 ∨ d = -1) ∧
    (e = 1 ∨ e = -1) ∧
    top_cell_sign a b c d e = 1) ∧
  (∃ n, n = 11) :=
by
  sorry

end top_cell_pos_cases_l186_186079


namespace rectangle_height_l186_186580

variable (h : ℕ) -- Define h as a natural number for the height

-- Given conditions
def width : ℕ := 32
def area_divided_by_diagonal : ℕ := 576

-- Math proof problem
theorem rectangle_height :
  (1 / 2 * (width * h) = area_divided_by_diagonal) → h = 36 := 
by
  sorry

end rectangle_height_l186_186580


namespace exists_graph_no_triangle_chromatic_gt_n_l186_186395

variable (n : ℕ)

theorem exists_graph_no_triangle_chromatic_gt_n (n : ℕ) : ∃ G : SimpleGraph ℕ, ¬∃ (t : Finset (G.vertex)), t.card = 3 ∧ G.Subgraph t ∧ G.IsComplete t ∧ G.IsChromatic n :=
by 
  sorry

end exists_graph_no_triangle_chromatic_gt_n_l186_186395


namespace distance_between_points_l186_186879

noncomputable def distance_AB (r : ℝ) (O1O2 : ℝ) (α : ℝ) : ℝ :=
  if (r > 1/2) then Real.sin α else 0

theorem distance_between_points
  (r : ℝ)
  (O1O2 : ℝ)
  (α : ℝ)
  (h1 : r > 1/2)
  (h2 : O1O2 = 1) :
  distance_AB r O1O2 α = Real.sin α :=
by
  -- Proof placeholder
  sorry

end distance_between_points_l186_186879


namespace factorial_sum_mod_26_l186_186499

theorem factorial_sum_mod_26 :
  (∑ n in Finset.range 26, n.factorial) % 26 = 0 :=
by
  sorry

end factorial_sum_mod_26_l186_186499


namespace girls_more_than_boys_l186_186742

variables (B G : ℕ)
def ratio_condition : Prop := 3 * G = 4 * B
def total_students_condition : Prop := B + G = 49

theorem girls_more_than_boys
  (h1 : ratio_condition B G)
  (h2 : total_students_condition B G) :
  G = B + 7 :=
sorry

end girls_more_than_boys_l186_186742


namespace sum_of_three_consecutive_odds_l186_186606

theorem sum_of_three_consecutive_odds (a : ℤ) (h : a % 2 = 1) (ha_mod : (a + 4) % 2 = 1) (h_sum : a + (a + 4) = 150) : a + (a + 2) + (a + 4) = 225 :=
sorry

end sum_of_three_consecutive_odds_l186_186606


namespace ImpossibleNonConformists_l186_186043

open Int

def BadPairCondition (f : ℤ → ℤ) (n : ℤ) :=
  ∃ (pairs : Finset (ℤ × ℤ)), 
    pairs.card ≤ ⌊0.001 * (n.natAbs^2 : ℝ)⌋₊ ∧ 
    ∀ (x y : ℤ), (x, y) ∈ pairs → max (abs x) (abs y) ≤ n ∧ f (x + y) ≠ f x + f y

def NonConformistCondition (f : ℤ → ℤ) (n : ℤ) :=
  ∃ (conformists : Finset ℤ), 
    conformists.card > n ∧ 
    ∀ (a : ℤ), abs a ≤ n → (f a ≠ a * f 1 → a ∈ conformists)

theorem ImpossibleNonConformists (f : ℤ → ℤ) :
  (∀ (n : ℤ), n ≥ 0 → BadPairCondition f n) → 
  ¬ ∃ (n : ℤ), n ≥ 0 ∧ NonConformistCondition f n :=
  by 
    intros h_cond h_ex
    sorry

end ImpossibleNonConformists_l186_186043


namespace sum_of_odd_powers_divisible_by_six_l186_186516

theorem sum_of_odd_powers_divisible_by_six (a1 a2 a3 a4 : ℤ)
    (h : a1^3 + a2^3 + a3^3 + a4^3 = 0) :
    ∀ k : ℕ, k % 2 = 1 → 6 ∣ (a1^k + a2^k + a3^k + a4^k) :=
by
  intros k hk
  sorry

end sum_of_odd_powers_divisible_by_six_l186_186516


namespace abs_gt_not_implies_gt_l186_186461

noncomputable def abs_gt_implies_gt (a b : ℝ) : Prop :=
  |a| > |b| → a > b

theorem abs_gt_not_implies_gt (a b : ℝ) :
  ¬ abs_gt_implies_gt a b :=
sorry

end abs_gt_not_implies_gt_l186_186461


namespace quadratic_inequality_solution_l186_186221

theorem quadratic_inequality_solution (a : ℝ) :
  (∀ x : ℝ, ax^2 + 2*x + a > 0 ↔ x ≠ -1/a) → a = 1 :=
by
  sorry

end quadratic_inequality_solution_l186_186221


namespace complement_union_eq_complement_l186_186907

open Set

section ComplementUnion

variable (k : ℤ)

def SetA : Set ℤ := {x | ∃ k : ℤ, x = 3 * k + 1}
def SetB : Set ℤ := {x | ∃ k : ℤ, x = 3 * k + 2}
def UniversalSet : Set ℤ := univ
def ComplementUnion : Set ℤ := {x | ∃ k : ℤ, x = 3 * k}

theorem complement_union_eq_complement :
  UniversalSet \ (SetA ∪ SetB) = ComplementUnion :=
by
  sorry

end ComplementUnion

end complement_union_eq_complement_l186_186907


namespace shape_of_intersections_is_rectangle_l186_186371

open Real

theorem shape_of_intersections_is_rectangle :
  (∀ (x y : ℝ), (xy = 18 ∧ x^2 + y^2 = 45) →
    set_of_points_formed_is_rectangle [{(6, 3), (-6, -3), (3, 6), (-3, -6)}]) :=
by
  sorry -- Proof to be filled in later

end shape_of_intersections_is_rectangle_l186_186371


namespace find_n_l186_186740

theorem find_n (n : ℤ) : 
  50 < n ∧ n < 120 ∧ (n % 8 = 0) ∧ (n % 7 = 3) ∧ (n % 9 = 3) → n = 192 :=
by
  sorry

end find_n_l186_186740


namespace faster_speed_l186_186309

theorem faster_speed (x : ℝ) (h1 : 40 = 8 * 5) (h2 : 60 = x * 5) : x = 12 :=
sorry

end faster_speed_l186_186309


namespace third_wins_against_seventh_l186_186075

-- Define the participants and their distinct points 
variables (p : ℕ → ℕ) (h_distinct : ∀ i j, i ≠ j → p i ≠ p j)
-- descending order condition
variables (h_order : ∀ i j, i < j → p i > p j)
-- second place points equals sum of last four places
variables (h_second : p 2 = p 5 + p 6 + p 7 + p 8)

-- Theorem stating the third place player won against the seventh place player
theorem third_wins_against_seventh :
  p 3 > p 7 :=
sorry

end third_wins_against_seventh_l186_186075


namespace positive_value_of_A_l186_186721

theorem positive_value_of_A (A : ℝ) (h : A^2 + 3^2 = 130) : A = 11 :=
sorry

end positive_value_of_A_l186_186721


namespace mary_added_peanuts_l186_186135

theorem mary_added_peanuts (initial final added : Nat) 
  (h1 : initial = 4)
  (h2 : final = 16)
  (h3 : final = initial + added) : 
  added = 12 := 
by {
  sorry
}

end mary_added_peanuts_l186_186135


namespace largest_divisor_of_5_consecutive_integers_l186_186755

theorem largest_divisor_of_5_consecutive_integers :
  ∃ d : ℤ, (∀ n : ℤ, d ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4))) ∧ d = 120 :=
sorry

end largest_divisor_of_5_consecutive_integers_l186_186755


namespace max_quadratic_function_l186_186065

theorem max_quadratic_function :
  ∃ M, (∀ x : ℝ, 0 ≤ x ∧ x ≤ 3 → (x^2 - 2*x - 1 ≤ M)) ∧
       (∀ y : ℝ, y = (x : ℝ) ^ 2 - 2 * x - 1 → x = 3 → y = M) :=
by
  use 2
  sorry

end max_quadratic_function_l186_186065


namespace inverse_g_167_is_2_l186_186220

def g (x : ℝ) := 5 * x^5 + 7

theorem inverse_g_167_is_2 : g⁻¹' {167} = {2} := by
  sorry

end inverse_g_167_is_2_l186_186220


namespace correct_option_l186_186978

-- Conditions
def option_A (a : ℝ) : Prop := a^3 + a^3 = a^6
def option_B (a : ℝ) : Prop := (a^3)^2 = a^9
def option_C (a : ℝ) : Prop := a^6 / a^3 = a^2
def option_D (a b : ℝ) : Prop := (a * b)^2 = a^2 * b^2

-- Proof Problem Statement
theorem correct_option (a b : ℝ) : option_D a b ↔ ¬option_A a ∧ ¬option_B a ∧ ¬option_C a :=
by
  sorry

end correct_option_l186_186978


namespace problem_condition_l186_186389

noncomputable def f (x b : ℝ) := Real.exp x * (x - b)
noncomputable def f_prime (x b : ℝ) := Real.exp x * (x - b + 1)
noncomputable def g (x : ℝ) := (x^2 + 2*x) / (x + 1)

theorem problem_condition (b : ℝ) :
  (∃ x ∈ Set.Icc (1 / 2 : ℝ) 2, f x b + x * f_prime x b > 0) → b < 8 / 3 :=
by
  sorry

end problem_condition_l186_186389


namespace hours_per_day_l186_186165

variable (m w : ℝ)
variable (h : ℕ)

-- Assume the equivalence of work done by women and men
axiom work_equiv : 3 * w = 2 * m

-- Total work done by men
def work_men := 15 * m * 21 * h
-- Total work done by women
def work_women := 21 * w * 36 * 5

-- The total work done by men and women is equal
theorem hours_per_day (h : ℕ) (w m : ℝ) (work_equiv : 3 * w = 2 * m) :
  15 * m * 21 * h = 21 * w * 36 * 5 → h = 8 :=
by
  intro H
  sorry

end hours_per_day_l186_186165


namespace john_flights_of_stairs_l186_186899

theorem john_flights_of_stairs (x : ℕ) : 
    let flight_height := 10
    let rope_height := flight_height / 2
    let ladder_height := rope_height + 10
    let total_height := 70
    10 * x + rope_height + ladder_height = total_height → x = 5 :=
by
    intro h
    sorry

end john_flights_of_stairs_l186_186899


namespace average_billboards_per_hour_l186_186946

def first_hour_billboards : ℕ := 17
def second_hour_billboards : ℕ := 20
def third_hour_billboards : ℕ := 23

theorem average_billboards_per_hour : 
  (first_hour_billboards + second_hour_billboards + third_hour_billboards) / 3 = 20 := 
by
  sorry

end average_billboards_per_hour_l186_186946


namespace total_vehicles_l186_186099

theorem total_vehicles (morn_minivans afternoon_minivans evening_minivans night_minivans : Nat)
                       (morn_sedans afternoon_sedans evening_sedans night_sedans : Nat)
                       (morn_SUVs afternoon_SUVs evening_SUVs night_SUVs : Nat)
                       (morn_trucks afternoon_trucks evening_trucks night_trucks : Nat)
                       (morn_motorcycles afternoon_motorcycles evening_motorcycles night_motorcycles : Nat) :
                       morn_minivans = 20 → afternoon_minivans = 22 → evening_minivans = 15 → night_minivans = 10 →
                       morn_sedans = 17 → afternoon_sedans = 13 → evening_sedans = 19 → night_sedans = 12 →
                       morn_SUVs = 12 → afternoon_SUVs = 15 → evening_SUVs = 18 → night_SUVs = 20 →
                       morn_trucks = 8 → afternoon_trucks = 10 → evening_trucks = 14 → night_trucks = 20 →
                       morn_motorcycles = 5 → afternoon_motorcycles = 7 → evening_motorcycles = 10 → night_motorcycles = 15 →
                       morn_minivans + afternoon_minivans + evening_minivans + night_minivans +
                       morn_sedans + afternoon_sedans + evening_sedans + night_sedans +
                       morn_SUVs + afternoon_SUVs + evening_SUVs + night_SUVs +
                       morn_trucks + afternoon_trucks + evening_trucks + night_trucks +
                       morn_motorcycles + afternoon_motorcycles + evening_motorcycles + night_motorcycles = 282 :=
by
  intros
  sorry

end total_vehicles_l186_186099


namespace total_combinations_l186_186526

-- Definitions to capture conditions
def earth_like_units := 3
def mars_like_units := 1
def total_units := 18
def total_earth_planets := 8
def total_mars_planets := 8

-- Proving the total number of combinations
theorem total_combinations : 
  (finset.card (finset.filter (λ x : finset (fin 16), finset.sum (finset.image x (λ y, if y < 8 then earth_like_units else mars_like_units)) = total_units) (finset.powerset_univ (fin 16)))) = 5124 :=
by
  -- We assume that there are 16 habitable planets in total, numerically indexed.
  sorry

end total_combinations_l186_186526


namespace part_a_l186_186208

theorem part_a (c : ℤ) : (∃ x : ℤ, x + (x / 2) = c) ↔ (c % 3 ≠ 2) :=
sorry

end part_a_l186_186208


namespace max_value_of_f_l186_186218

noncomputable def f (x : ℝ) : ℝ := 3 * x - x ^ 3

theorem max_value_of_f (a b : ℝ) (ha : ∀ x, f x ≤ b) (hfa : f a = b) : a - b = -1 :=
by
  sorry

end max_value_of_f_l186_186218


namespace largest_divisor_of_5_consecutive_integers_l186_186797

theorem largest_divisor_of_5_consecutive_integers :
  ∀ n : ℤ, ∃ k : ℤ, (product_of_5_consecutive_integers n = k) ∧ (60 ∣ k) 
:= sorry

end largest_divisor_of_5_consecutive_integers_l186_186797


namespace problem_solution_l186_186680

noncomputable def inequality_holds (a b : ℝ) (n : ℕ) : Prop :=
  (a + b)^n - a^n - b^n ≥ 2^(2 * n) - 2^(n - 1)

theorem problem_solution (a b : ℝ) (n : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : (1 / a + 1 / b = 1)) (h4 : 0 < n):
  inequality_holds a b n :=
by
  sorry

end problem_solution_l186_186680


namespace arithmetic_mean_difference_l186_186827

theorem arithmetic_mean_difference (p q r : ℝ) 
  (h1 : (p + q) / 2 = 10) 
  (h2 : (q + r) / 2 = 26) : 
  r - p = 32 := by
  sorry

end arithmetic_mean_difference_l186_186827


namespace triangle_circle_area_relation_l186_186170

theorem triangle_circle_area_relation (A B C : ℝ) (h : 15^2 + 20^2 = 25^2) (A_area_eq : A + B + 150 = C) :
  A + B + 150 = C :=
by
  -- The proof has been omitted.
  sorry

end triangle_circle_area_relation_l186_186170


namespace largest_divisor_of_product_of_five_consecutive_integers_l186_186766

theorem largest_divisor_of_product_of_five_consecutive_integers
  (a b c d e : ℤ) 
  (h: a + 1 = b ∧ b + 1 = c ∧ c + 1 = d ∧ d + 1 = e):
  ∃ (n : ℤ), n = 60 ∧ n ∣ (a * b * c * d * e) :=
sorry

end largest_divisor_of_product_of_five_consecutive_integers_l186_186766


namespace calc_expression_l186_186844

theorem calc_expression :
  (3 * Real.sqrt 48 - 2 * Real.sqrt 12) / Real.sqrt 3 = 8 :=
sorry

end calc_expression_l186_186844


namespace expanded_polynomial_correct_l186_186365

noncomputable def polynomial_product (x : ℚ) : ℚ :=
  (2 * x^3 - 3 * x + 1) * (x^2 + 4 * x + 3)

theorem expanded_polynomial_correct (x : ℚ) : 
  polynomial_product x = 2 * x^5 + 8 * x^4 + 3 * x^3 - 11 * x^2 - 5 * x + 3 := 
by
  sorry

end expanded_polynomial_correct_l186_186365


namespace find_tangent_point_l186_186964

noncomputable def exp_neg (x : ℝ) : ℝ := Real.exp (-x)

theorem find_tangent_point :
  ∃ P : ℝ × ℝ, P = (-Real.log 2, 2) ∧ P.snd = exp_neg P.fst ∧ deriv exp_neg P.fst = -2 :=
by
  sorry

end find_tangent_point_l186_186964


namespace inequalities_region_quadrants_l186_186504

theorem inequalities_region_quadrants:
  (∀ x y : ℝ, y > -2 * x + 3 → y > x / 2 + 1 → (x > 0 ∧ y > 0) ∨ (x < 0 ∧ y > 0)) :=
sorry

end inequalities_region_quadrants_l186_186504


namespace set_intersection_l186_186253

-- defining universal set U
def U : Set ℕ := {1, 3, 5, 7, 9}

-- defining set A
def A : Set ℕ := {1, 5, 9}

-- defining set B
def B : Set ℕ := {3, 7, 9}

-- complement of A in U
def complU (s : Set ℕ) := {x ∈ U | x ∉ s}

-- defining the intersection of complement of A with B
def intersection := complU A ∩ B

-- statement to be proved
theorem set_intersection : intersection = {3, 7} :=
by
  sorry

end set_intersection_l186_186253


namespace passengers_taken_at_second_station_l186_186840

noncomputable def initial_passengers : ℕ := 270
noncomputable def passengers_dropped_first_station := initial_passengers / 3
noncomputable def passengers_after_first_station := initial_passengers - passengers_dropped_first_station + 280
noncomputable def passengers_dropped_second_station := passengers_after_first_station / 2
noncomputable def passengers_after_second_station (x : ℕ) := passengers_after_first_station - passengers_dropped_second_station + x
noncomputable def passengers_at_third_station := 242

theorem passengers_taken_at_second_station : ∃ x : ℕ,
  passengers_after_second_station x = passengers_at_third_station ∧ x = 12 :=
by
  sorry

end passengers_taken_at_second_station_l186_186840


namespace cost_of_article_l186_186692

variable {C G : ℝ}

theorem cost_of_article (h : 350 = C * (1 + (G + 5) / 100)) (h' : 340 = C * (1 + G / 100)) : C = 200 := by
  sorry

end cost_of_article_l186_186692


namespace identify_tricksters_in_30_or_less_questions_l186_186320

-- Define the problem parameters
def inhabitants : Type := Fin 65

def is_knight (inhabitant : inhabitants) : Prop := sorry
def is_trickster (inhabitant : inhabitants) : Prop := sorry

-- Define the properties
axiom knight_truthful : ∀ (x : inhabitants), is_knight x → (forall y : inhabitants, True ↔ (is_knight y = x is_knight y))
axiom trickster_mixed : ∀ (x : inhabitants), is_trickster x → ((∀ y : inhabitants, True) ∨ (∃ y : inhabitants, y ∉ (is_knight y)))

-- Problem statement
theorem identify_tricksters_in_30_or_less_questions
  (inhabitants : Type)
  (n_tricksters : ℕ := 2) -- 2 tricksters
  (total_inhabitants : ℕ := 65) -- 65 total inhabitants
  (questions_limit : ℕ := 30) -- limit of 30 questions
  (knights : inhabitants → Prop)
  (tricksters : inhabitants → Prop) :
    ∃ (solution_exists : ∀ (is_trickster : inhabitants → Prop), ∃ k : inhabitants, (knights k) ∧ (is_trickster k)) 
    (possible_to_find_tricksters : ∀ (is_knight : inhabitants → Prop) (is_trickster : inhabitants → Prop), 
    ∃ (questions_used ≤ questions_limit), ∀ (xs : set inhabitants), questions_used ≤ 30 ∧ 
    (∃ trickster1 trickster2 : inhabitants, (tricksters trickster1 ∧ tricksters trickster2 ∧ trickster1 ≠ trickster2))) :=
sorry

end identify_tricksters_in_30_or_less_questions_l186_186320


namespace least_n_divisible_by_some_not_all_l186_186817

theorem least_n_divisible_by_some_not_all (n : ℕ) (h : 1 ≤ n):
  (∃ k : ℕ, 1 ≤ k ∧ k ≤ 10 ∧ k ∣ (n^2 - n)) ∧ ¬ (∀ k : ℕ, 1 ≤ k ∧ k ≤ 10 → k ∣ (n^2 - n)) ↔ n = 3 :=
by
  sorry

end least_n_divisible_by_some_not_all_l186_186817


namespace solve_for_x_l186_186235

theorem solve_for_x (x : ℝ) (h₁ : (x + 2) ≠ 0) (h₂ : (|x| - 2) / (x + 2) = 0) : x = 2 := by
  sorry

end solve_for_x_l186_186235


namespace periodicity_f_l186_186066

noncomputable def vectorA (x : ℝ) : ℝ × ℝ := (2 * Real.sin x, Real.cos x)
noncomputable def vectorB (x : ℝ) : ℝ × ℝ := (Real.cos x, 2 * Real.sqrt 3 * Real.cos x)
noncomputable def f (x : ℝ) : ℝ :=
  let a := vectorA x
  let b := vectorB x
  a.1 * b.1 + a.2 * b.2

theorem periodicity_f :
  (∀ x, f (x + Real.pi) = f x) ∧
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), (f x = 2 + Real.sqrt 3 ∨ f x = 0)) :=
by
  sorry

end periodicity_f_l186_186066


namespace circumcircle_equation_l186_186201

theorem circumcircle_equation :
  ∃ (a b r : ℝ), 
    (∀ {x y : ℝ}, (x, y) = (2, 2) → (x - a)^2 + (y - b)^2 = r^2) ∧
    (∀ {x y : ℝ}, (x, y) = (5, 3) → (x - a)^2 + (y - b)^2 = r^2) ∧
    (∀ {x y : ℝ}, (x, y) = (3, -1) → (x - a)^2 + (y - b)^2 = r^2) ∧
    ((x - 4)^2 + (y - 1)^2 = 5) :=
sorry

end circumcircle_equation_l186_186201


namespace largest_divisor_of_product_of_five_consecutive_integers_l186_186757

theorem largest_divisor_of_product_of_five_consecutive_integers :
  ∃ n, (∀ k : ℤ, n ∣ (k * (k + 1) * (k + 2) * (k + 3) * (k + 4))) ∧ n = 60 :=
by
  sorry

end largest_divisor_of_product_of_five_consecutive_integers_l186_186757


namespace rounded_product_less_than_original_l186_186268

theorem rounded_product_less_than_original
  (x y z : ℝ) (hx_pos : 0 < x) (hy_pos : 0 < y) (hz_pos : 0 < z) (hxy : x > 2 * y) :
  (x + z) * (y - z) < x * y :=
by
  sorry

end rounded_product_less_than_original_l186_186268


namespace combinedAverageAge_l186_186452

-- Definitions
def numFifthGraders : ℕ := 50
def avgAgeFifthGraders : ℕ := 10
def numParents : ℕ := 75
def avgAgeParents : ℕ := 40

-- Calculation of total ages
def totalAgeFifthGraders := numFifthGraders * avgAgeFifthGraders
def totalAgeParents := numParents * avgAgeParents
def combinedTotalAge := totalAgeFifthGraders + totalAgeParents

-- Calculation of total number of individuals
def totalIndividuals := numFifthGraders + numParents

-- The claim to prove
theorem combinedAverageAge : 
  combinedTotalAge / totalIndividuals = 28 := by
  -- Skipping the proof details.
  sorry

end combinedAverageAge_l186_186452


namespace point_Q_coordinates_l186_186441

noncomputable def coordinates_of_point_Q (start : ℝ × ℝ) (arc_length : ℝ) : ℝ × ℝ :=
  let θ := arc_length in
  (Real.cos θ, Real.sin θ)

theorem point_Q_coordinates :
  coordinates_of_point_Q (1, 0) (4 * Real.pi / 3) = (-1 / 2, -Real.sqrt 3 / 2) :=
by
  sorry

end point_Q_coordinates_l186_186441


namespace solution_strategy_l186_186169

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

end solution_strategy_l186_186169


namespace prob_of_caps_given_sunglasses_l186_186283

theorem prob_of_caps_given_sunglasses (n_sunglasses n_caps n_both : ℕ) (P_sunglasses_given_caps : ℚ) 
  (h_nsunglasses : n_sunglasses = 80) (h_ncaps : n_caps = 45)
  (h_Psunglasses_given_caps : P_sunglasses_given_caps = 3/8)
  (h_nboth : n_both = P_sunglasses_given_caps * n_sunglasses) :
  (n_both / n_caps) = 2/3 := 
by
  sorry

end prob_of_caps_given_sunglasses_l186_186283


namespace length_of_AB_l186_186252

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

theorem length_of_AB :
  let O := (0, 0)
  let A := (54^(1/3), 0)
  let B := (0, 54^(1/3))
  distance A B = 54^(1/3) * Real.sqrt 2 :=
by
  sorry

end length_of_AB_l186_186252


namespace number_of_friends_l186_186579

theorem number_of_friends (total_bill : ℝ) (discount_rate : ℝ) (paid_amount : ℝ) (n : ℝ) 
  (h_total_bill : total_bill = 400) 
  (h_discount_rate : discount_rate = 0.05)
  (h_paid_amount : paid_amount = 63.59) 
  (h_total_paid : n * paid_amount = total_bill * (1 - discount_rate)) : n = 6 := 
by
  -- proof goes here
  sorry

end number_of_friends_l186_186579


namespace volume_of_larger_cube_l186_186638

theorem volume_of_larger_cube (s : ℝ) (V : ℝ) :
  (∀ (n : ℕ), n = 125 →
    ∀ (v_sm : ℝ), v_sm = 1 →
    V = n * v_sm →
    V = s^3 →
    s = 5 →
    ∀ (sa_large : ℝ), sa_large = 6 * s^2 →
    sa_large = 150 →
    ∀ (sa_sm_total : ℝ), sa_sm_total = n * (6 * v_sm^(2/3)) →
    sa_sm_total = 750 →
    sa_sm_total - sa_large = 600 →
    V = 125) :=
by
  intros n n125 v_sm v1 Vdef Vcube sc5 sa_large sa_large_def sa_large150 sa_sm_total sa_sm_total_def sa_sm_total750 diff600
  simp at *
  sorry

end volume_of_larger_cube_l186_186638


namespace infinite_primes_dividing_f_l186_186560

noncomputable def f : ℕ → ℕ := sorry

def is_non_constant (f : ℕ → ℕ) : Prop :=
  ∃ a b : ℕ, a ≠ b ∧ f a ≠ f b

def is_multiple_of (f : ℕ → ℕ) (a b : ℕ) : Prop :=
  ∃ k : ℕ, f b - f a = k * (b - a)

theorem infinite_primes_dividing_f :
  (is_non_constant f) →
  (∀ a b : ℕ, is_multiple_of f a b) →
  ∃∞ p : ℕ, Nat.Prime p ∧ ∃ c : ℕ, p ∣ f c := 
sorry

end infinite_primes_dividing_f_l186_186560


namespace new_member_younger_by_160_l186_186951

theorem new_member_younger_by_160 
  (A : ℕ)  -- average age 8 years ago and today
  (O N : ℕ)  -- age of the old member and the new member respectively
  (h1 : 20 * A = 20 * A + O - N)  -- condition derived from the problem
  (h2 : 20 * 8 = 160)  -- age increase over 8 years for 20 members
  (h3 : O - N = 160) : O - N = 160 :=
by
  sorry

end new_member_younger_by_160_l186_186951


namespace largest_divisor_of_5_consecutive_integers_l186_186752

theorem largest_divisor_of_5_consecutive_integers :
  ∃ d : ℤ, (∀ n : ℤ, d ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4))) ∧ d = 120 :=
sorry

end largest_divisor_of_5_consecutive_integers_l186_186752


namespace least_three_digit_number_with_product_12_is_126_l186_186149

-- Define the condition for a three-digit number
def is_three_digit_number (n : ℕ) : Prop := n ≥ 100 ∧ n < 1000

-- Define the condition for product of digits being 12
def product_of_digits_is_12 (n : ℕ) : Prop := 
  let d100 := n / 100
  let d10 := (n % 100) / 10
  let d1 := n % 10
  d100 * d10 * d1 = 12

-- Define the property we want to prove, combining the above two conditions
def least_three_digit_number_with_product_12 : ℕ := 
  if h : ∃ n, is_three_digit_number n ∧ product_of_digits_is_12 n 
  then (Nat.find h)
  else 0  -- a default value if no such number exists, although it does in this case

-- Now the final theorem statement: proving least_three_digit_number_with_product_12 = 126
theorem least_three_digit_number_with_product_12_is_126 : 
  least_three_digit_number_with_product_12 = 126 :=
sorry

end least_three_digit_number_with_product_12_is_126_l186_186149


namespace total_cost_calculation_l186_186427

-- Definitions
def coffee_price : ℕ := 4
def cake_price : ℕ := 7
def ice_cream_price : ℕ := 3

def mell_coffee_qty : ℕ := 2
def mell_cake_qty : ℕ := 1
def friends_coffee_qty : ℕ := 2
def friends_cake_qty : ℕ := 1
def friends_ice_cream_qty : ℕ := 1

def total_coffee_qty : ℕ := 3 * mell_coffee_qty
def total_cake_qty : ℕ := 3 * mell_cake_qty
def total_ice_cream_qty : ℕ := 2 * friends_ice_cream_qty

def total_cost : ℕ := total_coffee_qty * coffee_price + total_cake_qty * cake_price + total_ice_cream_qty * ice_cream_price

-- Theorem Statement
theorem total_cost_calculation : total_cost = 51 := by
  sorry

end total_cost_calculation_l186_186427


namespace modulo_17_residue_l186_186596

theorem modulo_17_residue : (392 + 6 * 51 + 8 * 221 + 3^2 * 23) % 17 = 11 :=
by 
  sorry

end modulo_17_residue_l186_186596


namespace cost_price_l186_186690

theorem cost_price (C : ℝ) : 
  (0.05 * C = 350 - 340) → C = 200 :=
by
  assume h1 : 0.05 * C = 10
  sorry

end cost_price_l186_186690


namespace how_much_together_l186_186095

def madeline_money : ℕ := 48
def brother_money : ℕ := madeline_money / 2

theorem how_much_together : madeline_money + brother_money = 72 := by
  sorry

end how_much_together_l186_186095


namespace hash_op_calculation_l186_186058

-- Define the new operation
def hash_op (a b : ℚ) : ℚ :=
  a^2 + a * b - 5

-- Prove that (-3) # 6 = -14
theorem hash_op_calculation : hash_op (-3) 6 = -14 := by
  sorry

end hash_op_calculation_l186_186058


namespace complement_of_union_l186_186918

def A : Set ℤ := {x | ∃ k : ℤ, x = 3 * k + 1}
def B : Set ℤ := {x | ∃ k : ℤ, x = 3 * k + 2}
def U : Set ℤ := Set.univ

theorem complement_of_union (x : ℤ) : (x ∉ A ∪ B) ↔ ∃ k : ℤ, x = 3 * k := 
by {
  sorry
}

end complement_of_union_l186_186918


namespace largest_divisor_of_consecutive_product_l186_186807

theorem largest_divisor_of_consecutive_product (n : ℤ) (h : ∀ k : ℤ, (n = 5 * k)) :
  ∃ d : ℤ, d = 60 ∧ ∀ a : ℤ, divides d (a * (a + 1) * (a + 2) * (a + 3) * (a + 4)) :=
sorry

end largest_divisor_of_consecutive_product_l186_186807


namespace common_rational_root_neg_not_integer_l186_186656

theorem common_rational_root_neg_not_integer : 
  ∃ (p : ℚ), (p < 0) ∧ (¬ ∃ (z : ℤ), p = z) ∧ 
  (50 * p^4 + a * p^3 + b * p^2 + c * p + 20 = 0) ∧ 
  (20 * p^5 + d * p^4 + e * p^3 + f * p^2 + g * p + 50 = 0) := 
sorry

end common_rational_root_neg_not_integer_l186_186656


namespace sin_690_deg_l186_186360

noncomputable def sin_690_eq_neg_one_half : Prop :=
  sin (690 * real.pi / 180) = -(1 / 2)

theorem sin_690_deg : sin_690_eq_neg_one_half :=
  by sorry

end sin_690_deg_l186_186360


namespace trapezoidal_park_no_solution_l186_186269

theorem trapezoidal_park_no_solution :
  (∃ b1 b2 : ℕ, 2 * 1800 = 40 * (b1 + b2) ∧ (∃ m : ℕ, b1 = 5 * (2 * m + 1)) ∧ (∃ n : ℕ, b2 = 2 * n)) → false :=
by
  sorry

end trapezoidal_park_no_solution_l186_186269


namespace barbara_candies_left_l186_186842

def initial_candies: ℝ := 18.5
def candies_used_to_make_dessert: ℝ := 4.2
def candies_received_from_friend: ℝ := 6.8
def candies_eaten: ℝ := 2.7

theorem barbara_candies_left : 
  initial_candies - candies_used_to_make_dessert + candies_received_from_friend - candies_eaten = 18.4 := 
by
  sorry

end barbara_candies_left_l186_186842


namespace toy_cost_price_and_profit_l186_186157

-- Define the cost price of type A toy
def cost_A (x : ℝ) : ℝ := x

-- Define the cost price of type B toy
def cost_B (x : ℝ) : ℝ := 1.5 * x

-- Spending conditions
def spending_A (x : ℝ) (num_A : ℝ) : Prop := num_A = 1200 / x
def spending_B (x : ℝ) (num_B : ℝ) : Prop := num_B = 1500 / (1.5 * x)

-- Quantity difference condition
def quantity_difference (num_A num_B : ℝ) : Prop := num_A - num_B = 20

-- Selling prices
def selling_price_A : ℝ := 12
def selling_price_B : ℝ := 20

-- Total toys purchased condition
def total_toys (num_A num_B : ℝ) : Prop := num_A + num_B = 75

-- Profit condition
def profit_condition (num_A num_B cost_A cost_B : ℝ) : Prop :=
  (selling_price_A - cost_A) * num_A + (selling_price_B - cost_B) * num_B ≥ 300

theorem toy_cost_price_and_profit :
  ∃ (x : ℝ), 
  cost_A x = 10 ∧
  cost_B x = 15 ∧
  ∀ (num_A num_B : ℝ),
  spending_A x num_A →
  spending_B x num_B →
  quantity_difference num_A num_B →
  total_toys num_A num_B →
  profit_condition num_A num_B (cost_A x) (cost_B x) →
  num_A ≤ 25 :=
by
  sorry

end toy_cost_price_and_profit_l186_186157


namespace largest_divisor_of_5_consecutive_integers_l186_186754

theorem largest_divisor_of_5_consecutive_integers :
  ∃ d : ℤ, (∀ n : ℤ, d ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4))) ∧ d = 120 :=
sorry

end largest_divisor_of_5_consecutive_integers_l186_186754


namespace john_will_lose_weight_in_80_days_l186_186410

-- Assumptions based on the problem conditions
def calories_eaten : ℕ := 1800
def calories_burned : ℕ := 2300
def calories_to_lose_one_pound : ℕ := 4000
def pounds_to_lose : ℕ := 10

-- Definition of the net calories burned per day
def net_calories_burned_per_day : ℕ := calories_burned - calories_eaten

-- Definition of total calories to lose the target weight
def total_calories_to_lose_target_weight (pounds_to_lose : ℕ) : ℕ :=
  calories_to_lose_one_pound * pounds_to_lose

-- Definition of days to lose the target weight
def days_to_lose_weight (target_calories : ℕ) (daily_net_calories : ℕ) : ℕ :=
  target_calories / daily_net_calories

-- Prove that John will lose 10 pounds in 80 days
theorem john_will_lose_weight_in_80_days :
  days_to_lose_weight (total_calories_to_lose_target_weight pounds_to_lose) net_calories_burned_per_day = 80 := by
  sorry

end john_will_lose_weight_in_80_days_l186_186410


namespace cargo_transport_possible_l186_186834

theorem cargo_transport_possible 
  (total_cargo_weight : ℝ) 
  (weight_limit_per_box : ℝ) 
  (number_of_trucks : ℕ) 
  (max_load_per_truck : ℝ)
  (h1 : total_cargo_weight = 13.5)
  (h2 : weight_limit_per_box = 0.35)
  (h3 : number_of_trucks = 11)
  (h4 : max_load_per_truck = 1.5) :
  ∃ (n : ℕ), n ≤ number_of_trucks ∧ (total_cargo_weight / max_load_per_truck) ≤ n :=
by
  sorry

end cargo_transport_possible_l186_186834


namespace city_growth_rate_order_l186_186010

theorem city_growth_rate_order 
  (Dover Eden Fairview : Type) 
  (highest lowest : Type)
  (h1 : Dover = highest → ¬(Eden = highest) ∧ (Fairview = lowest))
  (h2 : ¬(Dover = highest) ∧ Eden = highest ∧ Fairview = lowest → Eden = highest ∧ Dover = lowest ∧ Fairview = highest)
  (h3 : ¬(Fairview = lowest) → ¬(Eden = highest) ∧ ¬(Dover = highest)) : 
  Eden = highest ∧ Dover = lowest ∧ Fairview = highest ∧ Eden ≠ lowest :=
by
  sorry

end city_growth_rate_order_l186_186010


namespace cost_price_of_article_l186_186039

theorem cost_price_of_article (SP : ℝ) (profit_percentage : ℝ) (profit_fraction : ℝ) (CP : ℝ) : 
  SP = 120 → profit_percentage = 25 → profit_fraction = profit_percentage / 100 → 
  SP = CP + profit_fraction * CP → CP = 96 :=
by intros hSP hprofit_percentage hprofit_fraction heq
   sorry

end cost_price_of_article_l186_186039


namespace shirts_left_l186_186719

-- Define the given conditions
def initial_shirts : ℕ := 4 * 12
def fraction_given : ℚ := 1 / 3

-- Define the proof goal
theorem shirts_left (initial_shirts : ℕ) (fraction_given : ℚ) : ℕ :=
let shirts_given := initial_shirts * fraction_given in
initial_shirts - (shirts_given : ℕ) = 32 :=
begin
  -- placeholder for the proof
  sorry
end

end shirts_left_l186_186719


namespace maximize_revenue_l186_186630

noncomputable def revenue (p : ℝ) : ℝ :=
p * (145 - 7 * p)

theorem maximize_revenue : ∃ p : ℕ, p ≤ 30 ∧ p = 10 ∧ ∀ q ≤ 30, revenue (q : ℝ) ≤ revenue 10 :=
by
  sorry

end maximize_revenue_l186_186630


namespace factorization_problem_1_factorization_problem_2_l186_186850

-- Problem 1: Factorize 2(m-n)^2 - m(n-m) and show it equals (n-m)(2n - 3m)
theorem factorization_problem_1 (m n : ℝ) :
  2 * (m - n)^2 - m * (n - m) = (n - m) * (2 * n - 3 * m) :=
by
  sorry

-- Problem 2: Factorize -4xy^2 + 4x^2y + y^3 and show it equals y(2x - y)^2
theorem factorization_problem_2 (x y : ℝ) :
  -4 * x * y^2 + 4 * x^2 * y + y^3 = y * (2 * x - y)^2 :=
by
  sorry

end factorization_problem_1_factorization_problem_2_l186_186850


namespace sum_of_three_consecutive_odds_l186_186608

theorem sum_of_three_consecutive_odds (a : ℤ) (h : a % 2 = 1) (ha_mod : (a + 4) % 2 = 1) (h_sum : a + (a + 4) = 150) : a + (a + 2) + (a + 4) = 225 :=
sorry

end sum_of_three_consecutive_odds_l186_186608


namespace binary_product_correct_l186_186855

-- Definitions based on the conditions
def bin1 : ℕ := 0b110011
def bin2 : ℕ := 0b1101
def product : ℕ := 0b10011000101

-- The Lean 4 statement for the proof problem
theorem binary_product_correct : bin1 * bin2 = product := by
  sorry

end binary_product_correct_l186_186855


namespace neg_prop_l186_186571

theorem neg_prop : ∃ (a : ℝ), ∀ (x : ℝ), (a * x^2 - 3 * x + 2 = 0) → x ≤ 0 :=
sorry

end neg_prop_l186_186571


namespace find_pairs_l186_186366

theorem find_pairs (a b : ℕ) (ha : 0 < a) (hb : 0 < b) : 
  (3 * a + 1 ∣ 4 * b - 1) ∧ (2 * b + 1 ∣ 3 * a - 1) ↔ (a = 2 ∧ b = 2) := 
by 
  sorry

end find_pairs_l186_186366


namespace min_bailing_rate_l186_186266

noncomputable def slowest_bailing_rate (distance : ℝ) (rowing_speed : ℝ) (leak_rate : ℝ) (max_capacity : ℝ) : ℝ :=
  let time_to_shore := distance / rowing_speed
  let time_to_shore_in_minutes := time_to_shore * 60
  let total_water_intake := leak_rate * time_to_shore_in_minutes
  let excess_water := total_water_intake - max_capacity
  excess_water / time_to_shore_in_minutes

theorem min_bailing_rate : slowest_bailing_rate 3 3 14 40 = 13.3 :=
by
  sorry

end min_bailing_rate_l186_186266


namespace tangent_value_range_l186_186129

theorem tangent_value_range : 
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ (π / 4) → 0 ≤ (Real.tan x) ∧ (Real.tan x) ≤ 1) :=
by
  sorry

end tangent_value_range_l186_186129


namespace total_miles_driven_l186_186505

-- Conditions
def miles_darius : ℕ := 679
def miles_julia : ℕ := 998

-- Proof statement
theorem total_miles_driven : miles_darius + miles_julia = 1677 := 
by
  -- placeholder for the proof steps
  sorry

end total_miles_driven_l186_186505


namespace trajectory_of_midpoint_l186_186728

theorem trajectory_of_midpoint {x y : ℝ} :
  (∃ Mx My : ℝ, (Mx + 3)^2 + My^2 = 4 ∧ (2 * x - 3 = Mx) ∧ (2 * y = My)) →
  x^2 + y^2 = 1 :=
by
  intro h
  sorry

end trajectory_of_midpoint_l186_186728


namespace false_statement_B_l186_186622

theorem false_statement_B : ¬ ∀ α β : ℝ, (α < 90) ∧ (β < 90) → (α + β > 90) :=
by
  sorry

end false_statement_B_l186_186622


namespace rhomboid_toothpicks_l186_186490

/-- 
Given:
- The rhomboid consists of two sections, each similar to half of a large equilateral triangle split along its height.
- The longest diagonal of the rhomboid contains 987 small equilateral triangles.
- The effective fact that each small equilateral triangle contributes on average 1.5 toothpicks due to shared sides.

Prove:
- The number of toothpicks required to construct the rhomboid is 1463598.
-/

-- Defining the number of small triangles along the base of the rhomboid
def base_triangles : ℕ := 987

-- Calculating the number of triangles in one section of the rhomboid
def triangles_in_section : ℕ := (base_triangles * (base_triangles + 1)) / 2

-- Calculating the total number of triangles in the rhomboid
def total_triangles : ℕ := 2 * triangles_in_section

-- Given the effective sides per triangle contributing to toothpicks is on average 1.5
def avg_sides_per_triangle : ℚ := 1.5

-- Calculating the total number of toothpicks required
def total_toothpicks : ℚ := avg_sides_per_triangle * total_triangles

theorem rhomboid_toothpicks (h : base_triangles = 987) : total_toothpicks = 1463598 := by
  sorry

end rhomboid_toothpicks_l186_186490


namespace least_number_with_digit_product_12_l186_186151

theorem least_number_with_digit_product_12 :
  ∃ n : ℕ, (n >= 100 ∧ n < 1000) ∧ 
           (∃ a b c : ℕ, a * b * c = 12 ∧ n = 100 * a + 10 * b + c ∧ a < b < c) ∧
           (∀ m : ℕ, (m >= 100 ∧ m < 1000) → 
                     (∃ x y z : ℕ, x * y * z = 12 ∧ m = 100 * x + 10 * y + z) → 
                     n ≤ m) :=
begin
  sorry
end

end least_number_with_digit_product_12_l186_186151


namespace tangent_line_curve_l186_186457

theorem tangent_line_curve (a b k : ℝ) (A : ℝ × ℝ) (hA : A = (1, 2))
  (tangent_condition : ∀ x, (k * x + 1 = x ^ 3 + a * x + b) → (k = 3 * x ^ 2 + a)) : b - a = 5 := by
  have hA := congr_arg (λ p : ℝ × ℝ, p.2) hA
  rw hA at tangent_condition
  sorry

end tangent_line_curve_l186_186457


namespace perpendicularity_condition_l186_186861

theorem perpendicularity_condition 
  (A B C D E F k b : ℝ) 
  (h1 : b ≠ 0)
  (line : ∀ (x : ℝ), y = k * x + b)
  (curve : ∀ (x y : ℝ), A * x^2 + 2 * B * x * y + C * y^2 + 2 * D * x + 2 * E * y + F = 0):
  A * b^2 - 2 * D * k * b + F * k^2 + C * b^2 + 2 * E * b + F = 0 :=
sorry

end perpendicularity_condition_l186_186861


namespace forester_total_trees_planted_l186_186635

theorem forester_total_trees_planted (initial_trees monday_trees tuesday_trees wednesday_trees total_trees : ℕ)
    (h1 : initial_trees = 30)
    (h2 : total_trees = 300)
    (h3 : monday_trees = 2 * initial_trees)
    (h4 : tuesday_trees = monday_trees / 3)
    (h5 : wednesday_trees = 2 * tuesday_trees) : 
    (monday_trees + tuesday_trees + wednesday_trees = 120) := by
  sorry

end forester_total_trees_planted_l186_186635


namespace Laura_bought_one_kg_of_potatoes_l186_186903

theorem Laura_bought_one_kg_of_potatoes :
  let price_salad : ℝ := 3
  let price_beef_per_kg : ℝ := 2 * price_salad
  let price_potato_per_kg : ℝ := price_salad * (1 / 3)
  let price_juice_per_liter : ℝ := 1.5
  let total_cost : ℝ := 22
  let num_salads : ℝ := 2
  let num_beef_kg : ℝ := 2
  let num_juice_liters : ℝ := 2
  let cost_salads := num_salads * price_salad
  let cost_beef := num_beef_kg * price_beef_per_kg
  let cost_juice := num_juice_liters * price_juice_per_liter
  (total_cost - (cost_salads + cost_beef + cost_juice)) / price_potato_per_kg = 1 :=
sorry

end Laura_bought_one_kg_of_potatoes_l186_186903


namespace max_value_of_f_on_interval_l186_186854

noncomputable def f (x : ℝ) : ℝ := 2^x + x * Real.log (1/4)

theorem max_value_of_f_on_interval :
  ∃ x ∈ Set.Icc (-2:ℝ) 2, f x = (1/4:ℝ) + 4 * Real.log 2 := 
sorry

end max_value_of_f_on_interval_l186_186854


namespace common_divisors_of_150_and_m_l186_186393

theorem common_divisors_of_150_and_m (m : ℕ) (h : ∃ (divs : Finset ℕ), divs = {1, q, q^2} ∧ 150 ∣ m ∧ (∀ d ∈ divs, d ∣ 150) ∧ (∀ d ∈ divs, d ∣ m)) : 
  ∃ q : ℕ, (nat.prime q) ∧ (q = 5) ∧ (150 ∣ m) ∧ (∀ d ∈ {1, q, q^2 : ℕ}, d ∣ 150) ∧ (∀ d ∈ {1, q, q^2 : ℕ}, d ∣ m) ∧ q^2 = 25 :=
by
  -- Proof construction skipped
  sorry

end common_divisors_of_150_and_m_l186_186393


namespace contrapositive_statement_l186_186274

theorem contrapositive_statement :
  (∀ n : ℕ, (n % 2 = 0 ∨ n % 5 = 0) → n % 10 = 0) →
  (∀ n : ℕ, n % 10 ≠ 0 → ¬(n % 2 = 0 ∧ n % 5 = 0)) :=
by
  sorry

end contrapositive_statement_l186_186274


namespace identity_proof_l186_186262

theorem identity_proof (a b c x y z : ℝ) : 
  (a * x + b * y + c * z) ^ 2 + (b * x + c * y + a * z) ^ 2 + (c * x + a * y + b * z) ^ 2 = 
  (c * x + b * y + a * z) ^ 2 + (b * x + a * y + c * z) ^ 2 + (a * x + c * y + b * z) ^ 2 := 
by
  sorry

end identity_proof_l186_186262


namespace two_by_three_grid_count_l186_186042

noncomputable def valid2x3Grids : Nat :=
  let valid_grids : Nat := 9
  valid_grids

theorem two_by_three_grid_count : valid2x3Grids = 9 := by
  -- Skipping the proof steps, but stating the theorem.
  sorry

end two_by_three_grid_count_l186_186042


namespace probability_of_black_black_red_l186_186996

-- Definitions:
def deck_size := 52
def black_cards := 26
def red_cards := 26

-- The probability calculation specific to the problem.
def probability_first_two_black_third_red : ℚ :=
  (black_cards * (black_cards - 1) / (deck_size * (deck_size - 1))) *
  (red_cards / (deck_size - 2))

-- The main statement of the theorem.
theorem probability_of_black_black_red :
  probability_first_two_black_third_red = 13 / 102 :=
by
  -- skip the proof using sorry.
  sorry

end probability_of_black_black_red_l186_186996


namespace Amy_finish_time_l186_186479

-- Definitions and assumptions based on conditions
def Patrick_time : ℕ := 60
def Manu_time : ℕ := Patrick_time + 12
def Amy_time : ℕ := Manu_time / 2

-- Theorem statement to be proved
theorem Amy_finish_time : Amy_time = 36 :=
by
  sorry

end Amy_finish_time_l186_186479


namespace prove_product_reduced_difference_l186_186128

-- We are given two numbers x and y such that:
variable (x y : ℚ)
-- 1. The sum of the numbers is 6
axiom sum_eq_six : x + y = 6
-- 2. The quotient of the larger number by the smaller number is 6
axiom quotient_eq_six : x / y = 6

-- We need to prove that the product of these two numbers reduced by their difference is 6/49
theorem prove_product_reduced_difference (x y : ℚ) 
  (sum_eq_six : x + y = 6) (quotient_eq_six : x / y = 6) : 
  (x * y) - (x - y) = 6 / 49 := 
by
  sorry

end prove_product_reduced_difference_l186_186128


namespace negation_of_p_implication_q_l186_186863

noncomputable def negation_of_conditions : Prop :=
∀ (a : ℝ), (a > 0 → a^2 > a) ∧ (¬(a > 0) ↔ ¬(a^2 > a)) → ¬(a ≤ 0 → a^2 ≤ a)

theorem negation_of_p_implication_q :
  negation_of_conditions :=
by {
  sorry
}

end negation_of_p_implication_q_l186_186863


namespace total_cost_l186_186424

-- Given conditions
def pen_cost : ℕ := 4
def briefcase_cost : ℕ := 5 * pen_cost

-- Theorem stating the total cost Marcel paid for both items
theorem total_cost (pen_cost briefcase_cost : ℕ) (h_pen: pen_cost = 4) (h_briefcase: briefcase_cost = 5 * pen_cost) :
  pen_cost + briefcase_cost = 24 := by
  sorry

end total_cost_l186_186424


namespace allan_correct_answers_l186_186567

theorem allan_correct_answers (x y : ℕ) (h1 : x + y = 120) (h2 : x - (0.25 : ℝ) * y = 100) : x = 104 :=
by
  sorry

end allan_correct_answers_l186_186567


namespace louise_needs_eight_boxes_l186_186564

-- Define the given conditions
def red_pencils : ℕ := 20
def blue_pencils : ℕ := 2 * red_pencils
def yellow_pencils : ℕ := 40
def green_pencils : ℕ := red_pencils + blue_pencils
def pencils_per_box : ℕ := 20

-- Define the functions to calculate the required number of boxes for each color
def boxes_needed (pencils : ℕ) : ℕ := (pencils + pencils_per_box - 1) / pencils_per_box

-- Calculate the total number of boxes needed by summing the boxes for each color
def total_boxes_needed : ℕ := boxes_needed red_pencils + boxes_needed blue_pencils + boxes_needed yellow_pencils + boxes_needed green_pencils

-- The proof problem statement
theorem louise_needs_eight_boxes : total_boxes_needed = 8 :=
by
  sorry

end louise_needs_eight_boxes_l186_186564


namespace Sophie_donuts_problem_l186_186450

noncomputable def total_cost_before_discount (cost_per_box : ℝ) (num_boxes : ℕ) : ℝ :=
  cost_per_box * num_boxes

noncomputable def discount_amount (total_cost : ℝ) (discount_rate : ℝ) : ℝ :=
  total_cost * discount_rate

noncomputable def total_cost_after_discount (total_cost : ℝ) (discount : ℝ) : ℝ :=
  total_cost - discount

noncomputable def total_donuts (donuts_per_box : ℕ) (num_boxes : ℕ) : ℕ :=
  donuts_per_box * num_boxes

noncomputable def donuts_left (total_donuts : ℕ) (donuts_given_away : ℕ) : ℕ :=
  total_donuts - donuts_given_away

theorem Sophie_donuts_problem
  (budget : ℝ)
  (cost_per_box : ℝ)
  (discount_rate : ℝ)
  (num_boxes : ℕ)
  (donuts_per_box : ℕ)
  (donuts_given_to_mom : ℕ)
  (donuts_given_to_sister : ℕ)
  (half_dozen : ℕ) :
  budget = 50 →
  cost_per_box = 12 →
  discount_rate = 0.10 →
  num_boxes = 4 →
  donuts_per_box = 12 →
  donuts_given_to_mom = 12 →
  donuts_given_to_sister = 6 →
  half_dozen = 6 →
  total_cost_after_discount (total_cost_before_discount cost_per_box num_boxes) (discount_amount (total_cost_before_discount cost_per_box num_boxes) discount_rate) = 43.2 ∧
  donuts_left (total_donuts donuts_per_box num_boxes) (donuts_given_to_mom + donuts_given_to_sister) = 30 :=
by
  sorry

end Sophie_donuts_problem_l186_186450


namespace five_digit_odd_and_multiples_of_5_sum_l186_186415

theorem five_digit_odd_and_multiples_of_5_sum :
  let A := 9 * 10^3 * 5
  let B := 9 * 10^3 * 1
  A + B = 45000 := by
sorry

end five_digit_odd_and_multiples_of_5_sum_l186_186415


namespace exists_polynomial_triangle_property_l186_186985

noncomputable def f (x y z : ℝ) : ℝ :=
  (x + y + z) * (-x + y + z) * (x - y + z) * (x + y - z)

theorem exists_polynomial_triangle_property :
  ∀ (x y z : ℝ), (f x y z > 0 ↔ (|x| + |y| > |z| ∧ |y| + |z| > |x| ∧ |z| + |x| > |y|)) :=
sorry

end exists_polynomial_triangle_property_l186_186985


namespace sin_690_eq_negative_one_half_l186_186357

theorem sin_690_eq_negative_one_half : Real.sin (690 * Real.pi / 180) = -1 / 2 := by
  sorry

end sin_690_eq_negative_one_half_l186_186357


namespace Mike_siblings_l186_186003

-- Define the types for EyeColor, HairColor and Sport
inductive EyeColor
| Blue
| Green

inductive HairColor
| Black
| Blonde

inductive Sport
| Soccer
| Basketball

-- Define the Child structure
structure Child where
  name : String
  eyeColor : EyeColor
  hairColor : HairColor
  favoriteSport : Sport

-- Define all the children based on the given conditions
def Lily : Child := { name := "Lily", eyeColor := EyeColor.Green, hairColor := HairColor.Black, favoriteSport := Sport.Soccer }
def Mike : Child := { name := "Mike", eyeColor := EyeColor.Blue, hairColor := HairColor.Blonde, favoriteSport := Sport.Basketball }
def Oliver : Child := { name := "Oliver", eyeColor := EyeColor.Blue, hairColor := HairColor.Black, favoriteSport := Sport.Soccer }
def Emma : Child := { name := "Emma", eyeColor := EyeColor.Green, hairColor := HairColor.Blonde, favoriteSport := Sport.Basketball }
def Jacob : Child := { name := "Jacob", eyeColor := EyeColor.Blue, hairColor := HairColor.Blonde, favoriteSport := Sport.Soccer }
def Sophia : Child := { name := "Sophia", eyeColor := EyeColor.Green, hairColor := HairColor.Blonde, favoriteSport := Sport.Soccer }

-- Siblings relation
def areSiblings (c1 c2 c3 : Child) : Prop :=
  (c1.eyeColor = c2.eyeColor ∨ c1.hairColor = c2.hairColor ∨ c1.favoriteSport = c2.favoriteSport) ∧
  (c1.eyeColor = c3.eyeColor ∨ c1.hairColor = c3.hairColor ∨ c1.favoriteSport = c3.favoriteSport) ∧
  (c2.eyeColor = c3.eyeColor ∨ c2.hairColor = c3.hairColor ∨ c2.favoriteSport = c3.favoriteSport)

-- The proof statement
theorem Mike_siblings : areSiblings Mike Emma Jacob := by
  -- Proof must be provided here
  sorry

end Mike_siblings_l186_186003


namespace Walter_age_in_2010_l186_186336

-- Define Walter's age in 2005 as y
def Walter_age_2005 (y : ℕ) : Prop :=
  (2005 - y) + (2005 - 3 * y) = 3858

-- Define Walter's age in 2010
theorem Walter_age_in_2010 (y : ℕ) (hy : Walter_age_2005 y) : y + 5 = 43 :=
by
  sorry

end Walter_age_in_2010_l186_186336


namespace find_starting_number_of_range_l186_186966

theorem find_starting_number_of_range : 
  ∃ (n : ℤ), 
    (∀ k, (0 ≤ k ∧ k < 7) → (n + k * 3 ≤ 31 ∧ n + k * 3 % 3 = 0)) ∧ 
    n + 6 * 3 = 30 - 6 * 3 :=
by
  sorry

end find_starting_number_of_range_l186_186966


namespace temp_neg_represents_below_zero_l186_186891

-- Definitions based on the conditions in a)
def above_zero (x: ℤ) : Prop := x > 0
def below_zero (x: ℤ) : Prop := x < 0

-- Proof problem derived from c)
theorem temp_neg_represents_below_zero (t1 t2: ℤ) 
  (h1: above_zero t1) (h2: t1 = 10) 
  (h3: below_zero t2) (h4: t2 = -3) : 
  -t2 = 3 :=
by
  sorry

end temp_neg_represents_below_zero_l186_186891


namespace largest_divisor_of_5_consecutive_integers_l186_186794

theorem largest_divisor_of_5_consecutive_integers :
  ∀ n : ℤ, ∃ k : ℤ, (product_of_5_consecutive_integers n = k) ∧ (60 ∣ k) 
:= sorry

end largest_divisor_of_5_consecutive_integers_l186_186794


namespace rate_per_sq_meter_l186_186456

theorem rate_per_sq_meter
  (L : ℝ) (W : ℝ) (total_cost : ℝ)
  (hL : L = 6) (hW : W = 4.75) (h_total_cost : total_cost = 25650) :
  total_cost / (L * W) = 900 :=
by
  sorry

end rate_per_sq_meter_l186_186456


namespace sum_of_three_consecutive_odds_l186_186609

theorem sum_of_three_consecutive_odds (a : ℤ) (h : a % 2 = 1) (ha_mod : (a + 4) % 2 = 1) (h_sum : a + (a + 4) = 150) : a + (a + 2) + (a + 4) = 225 :=
sorry

end sum_of_three_consecutive_odds_l186_186609


namespace christmas_day_december_25_l186_186506

-- Define the conditions
def is_thursday (d: ℕ) : Prop := d % 7 = 4
def thanksgiving := 26
def december_christmas := 25

-- Define the problem as a proof problem
theorem christmas_day_december_25 :
  is_thursday (thanksgiving) → thanksgiving = 26 →
  december_christmas = 25 → 
  30 - 26 + 25 = 28 → 
  is_thursday (30 - 26 + 25) :=
by
  intro h_thursday h_thanksgiving h_christmas h_days
  -- skipped proof
  sorry

end christmas_day_december_25_l186_186506


namespace amy_race_time_l186_186477

theorem amy_race_time (patrick_time : ℕ) (manu_time : ℕ) (amy_time : ℕ)
  (h1 : patrick_time = 60)
  (h2 : manu_time = patrick_time + 12)
  (h3 : amy_time = manu_time / 2) : 
  amy_time = 36 := 
sorry

end amy_race_time_l186_186477


namespace simplify_and_evaluate_l186_186109

-- Define the variables
variables (x y : ℝ)

-- Define the expression
def expression := 2 * x * y + (3 * x * y - 2 * y^2) - 2 * (x * y - y^2)

-- Introduce the conditions
theorem simplify_and_evaluate : 
  (x = -1) → (y = 2) → expression x y = -6 := 
by 
  intro hx hy 
  sorry

end simplify_and_evaluate_l186_186109


namespace find_correct_t_l186_186383

theorem find_correct_t (t : ℝ) :
  (∃! x1 x2 x3 : ℝ, x1^2 - 4*|x1| + 3 = t ∧
                     x2^2 - 4*|x2| + 3 = t ∧
                     x3^2 - 4*|x3| + 3 = t) → t = 3 :=
by
  sorry

end find_correct_t_l186_186383


namespace max_theater_members_l186_186282

theorem max_theater_members (N : ℕ) :
  (∃ (k : ℕ), (N = k^2 + 3)) ∧ (∃ (n : ℕ), (N = n * (n + 9))) → N ≤ 360 :=
by
  sorry

end max_theater_members_l186_186282


namespace greatest_possible_avg_speed_l186_186496

theorem greatest_possible_avg_speed (initial_odometer : ℕ) (max_speed : ℕ) (time_hours : ℕ) (max_distance : ℕ) (target_palindrome : ℕ) :
  initial_odometer = 12321 →
  max_speed = 80 →
  time_hours = 4 →
  (target_palindrome = 12421 ∨ target_palindrome = 12521 ∨ target_palindrome = 12621 ∨ target_palindrome = 12721 ∨ target_palindrome = 12821 ∨ target_palindrome = 12921 ∨ target_palindrome = 13031) →
  target_palindrome - initial_odometer ≤ max_distance →
  max_distance = 300 →
  target_palindrome = 12621 →
  time_hours = 4 →
  target_palindrome - initial_odometer = 300 →
  (target_palindrome - initial_odometer) / time_hours = 75 :=
by 
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9
  sorry

end greatest_possible_avg_speed_l186_186496


namespace archery_competition_l186_186700

theorem archery_competition (points : Finset ℕ) (product : ℕ) : 
  points = {11, 7, 5, 2} ∧ product = 38500 → 
  ∃ n : ℕ, n = 7 := 
by
  intros h
  sorry

end archery_competition_l186_186700


namespace ratio_h_w_l186_186110

-- Definitions from conditions
variables (h w : ℝ)
variables (XY YZ : ℝ)
variables (h_pos : 0 < h) (w_pos : 0 < w) -- heights and widths are positive
variables (XY_pos : 0 < XY) (YZ_pos : 0 < YZ) -- segment lengths are positive

-- Given that in the right-angled triangle ∆XYZ, YZ = 2 * XY
axiom YZ_eq_2XY : YZ = 2 * XY

-- Prove that h / w = 3 / 8
theorem ratio_h_w (H : XY / YZ = 4 * h / (3 * w)) : h / w = 3 / 8 :=
by {
  -- Use the axioms and given conditions here to prove H == ratio
  sorry
}

end ratio_h_w_l186_186110


namespace solve_for_A_l186_186153

theorem solve_for_A : 
  ∃ (A B : ℕ), (100 * A + 78) - (200 + 10 * B + 4) = 364 → A = 5 :=
by
  sorry

end solve_for_A_l186_186153


namespace sum_of_products_of_roots_l186_186092

theorem sum_of_products_of_roots :
  ∀ (p q r : ℝ), (4 * p^3 - 6 * p^2 + 17 * p - 10 = 0) ∧ 
                 (4 * q^3 - 6 * q^2 + 17 * q - 10 = 0) ∧ 
                 (4 * r^3 - 6 * r^2 + 17 * r - 10 = 0) →
                 (p * q + q * r + r * p = 17 / 4) :=
by
  sorry

end sum_of_products_of_roots_l186_186092


namespace sahil_selling_price_l186_186106

-- Definitions based on the conditions
def purchase_price : ℕ := 10000
def repair_costs : ℕ := 5000
def transportation_charges : ℕ := 1000
def profit_percentage : ℕ := 50

def total_cost : ℕ := purchase_price + repair_costs + transportation_charges
def profit : ℕ := (profit_percentage * total_cost) / 100
def selling_price : ℕ := total_cost + profit

-- The theorem we need to prove
theorem sahil_selling_price : selling_price = 24000 :=
by
  sorry

end sahil_selling_price_l186_186106


namespace product_of_five_consecutive_integers_divisible_by_240_l186_186803

theorem product_of_five_consecutive_integers_divisible_by_240 (n : ℤ) : ∃ k : ℤ, (∏ i in range 5, n + i) = 240 * k := 
sorry

end product_of_five_consecutive_integers_divisible_by_240_l186_186803


namespace prob_sum_24_four_dice_l186_186375

-- The probability of each die landing on six
def prob_die_six : ℚ := 1 / 6

-- The probability of all four dice showing six
theorem prob_sum_24_four_dice : 
  prob_die_six ^ 4 = 1 / 1296 :=
by
  -- Equivalent Lean statement asserting the probability problem
  sorry

end prob_sum_24_four_dice_l186_186375


namespace rectangle_other_side_length_l186_186101

/-- Theorem: Consider a rectangle with one side of length 10 cm. Another rectangle of dimensions 
10 cm x 1 cm fits diagonally inside this rectangle. We need to prove that the length 
of the other side of the larger rectangle is 2.96 cm. -/
theorem rectangle_other_side_length :
  ∃ (x : ℝ), (x ≠ 0) ∧ (0 < x) ∧ (10 * 10 - x * x = 1 * 1) ∧ x = 2.96 :=
sorry

end rectangle_other_side_length_l186_186101


namespace total_marks_by_category_l186_186546

theorem total_marks_by_category 
  (num_candidates_A : ℕ) (num_candidates_B : ℕ) (num_candidates_C : ℕ)
  (avg_marks_A : ℕ) (avg_marks_B : ℕ) (avg_marks_C : ℕ) 
  (hA : num_candidates_A = 30) (hB : num_candidates_B = 25) (hC : num_candidates_C = 25)
  (h_avg_A : avg_marks_A = 35) (h_avg_B : avg_marks_B = 42) (h_avg_C : avg_marks_C = 46) :
  (num_candidates_A * avg_marks_A = 1050) ∧
  (num_candidates_B * avg_marks_B = 1050) ∧
  (num_candidates_C * avg_marks_C = 1150) := 
by
  sorry

end total_marks_by_category_l186_186546


namespace contingency_fund_l186_186071

theorem contingency_fund:
  let d := 240
  let cp := d * (1.0 / 3)
  let lc := d * (1.0 / 2)
  let r := d - cp - lc
  let lp := r * (1.0 / 4)
  let cf := r - lp
  cf = 30 := 
by
  sorry

end contingency_fund_l186_186071


namespace salt_solution_concentration_l186_186748

theorem salt_solution_concentration (m x : ℝ) (h1 : m > 30) (h2 : (m * m / 100) = ((m - 20) / 100) * (m + 2 * x)) :
  x = 10 * m / (m + 20) :=
sorry

end salt_solution_concentration_l186_186748


namespace find_2n_plus_m_l186_186741

theorem find_2n_plus_m (n m : ℤ) (h1 : 3 * n - m < 5) (h2 : n + m > 26) (h3 : 3 * m - 2 * n < 46) : 
  2 * n + m = 36 := 
sorry

end find_2n_plus_m_l186_186741


namespace triangle_problem_statement_l186_186930

variable (A B C D E : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E]
variables (AD DB AC CB AE EB BC : ℝ)
variable (ACE ECB DCB ACD BCE : ℝ)

theorem triangle_problem_statement 
  (h : (AD / DB) * (AE / EB) = (AC / CB) ^ 2) :
  (AE / EB = AC * Real.sin ACE / (BC * Real.sin ECB)) ∧ (ACD = BCE) := 
by
  sorry

end triangle_problem_statement_l186_186930


namespace find_e_l186_186250

-- Conditions
def f (x : ℝ) (b : ℝ) := 5 * x + b
def g (x : ℝ) (b : ℝ) := b * x + 4
def f_comp_g (x : ℝ) (b : ℝ) (e : ℝ) := 15 * x + e

-- Statement to prove
theorem find_e (b e : ℝ) (x : ℝ): 
  (f (g x b) b = f_comp_g x b e) → 
  (5 * b = 15) → 
  (20 + b = e) → 
  e = 23 :=
by 
  intros h1 h2 h3
  sorry

end find_e_l186_186250


namespace Annika_hike_time_l186_186332

-- Define the conditions
def hike_rate : ℝ := 12 -- in minutes per kilometer
def initial_distance_east : ℝ := 2.75 -- in kilometers
def total_distance_east : ℝ := 3.041666666666667 -- in kilometers
def total_time_needed : ℝ := 40 -- in minutes

-- The theorem to prove
theorem Annika_hike_time : 
  (initial_distance_east + (total_distance_east - initial_distance_east)) * hike_rate + total_distance_east * hike_rate = total_time_needed := 
by
  sorry

end Annika_hike_time_l186_186332


namespace numValidRoutesJackToJill_l186_186710

noncomputable def numPaths (n m : ℕ) : ℕ :=
  Nat.choose (n + m) n

theorem numValidRoutesJackToJill : 
  let totalRoutes := numPaths 5 3
  let pathsViaDanger := numPaths 2 2 * numPaths 3 1
  totalRoutes - pathsViaDanger = 32 :=
by
  let totalRoutes := numPaths 5 3
  let pathsViaDanger := numPaths 2 2 * numPaths 3 1
  show totalRoutes - pathsViaDanger = 32
  sorry

end numValidRoutesJackToJill_l186_186710


namespace complement_of_union_l186_186919

def A : Set ℤ := {x | ∃ k : ℤ, x = 3 * k + 1}
def B : Set ℤ := {x | ∃ k : ℤ, x = 3 * k + 2}
def U : Set ℤ := Set.univ

theorem complement_of_union (x : ℤ) : (x ∉ A ∪ B) ↔ ∃ k : ℤ, x = 3 * k := 
by {
  sorry
}

end complement_of_union_l186_186919


namespace xy_sum_value_l186_186672

theorem xy_sum_value (x y : ℝ) (h1 : x + Real.cos y = 1010) (h2 : x + 1010 * Real.sin y = 1009) (h3 : (Real.pi / 4) ≤ y ∧ y ≤ (Real.pi / 2)) :
  x + y = 1010 + (Real.pi / 2) := 
by
  sorry

end xy_sum_value_l186_186672


namespace fraction_comparison_l186_186653

theorem fraction_comparison : 
  (15 / 11 : ℝ) > (17 / 13 : ℝ) ∧ (17 / 13 : ℝ) > (19 / 15 : ℝ) :=
by
  sorry

end fraction_comparison_l186_186653


namespace largest_divisor_of_consecutive_five_l186_186787

theorem largest_divisor_of_consecutive_five (n : ℤ) : 
  (∃ d, ∀ i ∈ {n, n + 1, n + 2, n + 3, n + 4}, d ∣ (∏ x in {n, n + 1, n + 2, n + 3, n + 4}, x)) → 
  d = 60 
:= sorry

end largest_divisor_of_consecutive_five_l186_186787


namespace largest_divisor_of_five_consecutive_integers_l186_186773

theorem largest_divisor_of_five_consecutive_integers:
  ∀ (n : ℤ), 
    ∃ (a b c : ℤ), 
      (n = 5 * a ∨ n = 5 * a + 1 ∨ n = 5 * a + 2 ∨ n = 5 * a + 3 ∨ n = 5 * a + 4) ∧ 
      (n = 3 * b ∨ n = 3 * b + 1 ∨ n = 3 * b + 2) ∧ 
      (n = 4 * c ∨ n = 4 * c + 1 ∨ n = 4 * c + 2 ∨ n = 4 * c + 3)
      ⊢ 60 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  -- Sorry placeholder for the proof (not needed in the task)
  sorry

end largest_divisor_of_five_consecutive_integers_l186_186773


namespace revenue_95_percent_l186_186231

-- Definitions based on the conditions
variables (C : ℝ) (n : ℝ)
def revenue_full : ℝ := 1.20 * C
def tickets_sold_percentage : ℝ := 0.95

-- Statement of the theorem based on the problem translation
theorem revenue_95_percent (C : ℝ) :
  (tickets_sold_percentage * revenue_full C) = 1.14 * C :=
by
  sorry -- Proof to be provided

end revenue_95_percent_l186_186231


namespace range_of_a_l186_186074

theorem range_of_a (a : ℝ) : 
(∀ x : ℝ, |x - 1| + |x - 3| > a ^ 2 - 2 * a - 1) ↔ -1 < a ∧ a < 3 := 
sorry

end range_of_a_l186_186074


namespace volume_of_rectangular_solid_l186_186743

theorem volume_of_rectangular_solid (x y z : ℝ) 
  (h1 : x * y = 18) 
  (h2 : y * z = 15) 
  (h3 : z * x = 10) : 
  x * y * z = 30 * Real.sqrt 3 := 
sorry

end volume_of_rectangular_solid_l186_186743


namespace john_average_speed_l186_186552

/--
John drove continuously from 8:15 a.m. until 2:05 p.m. of the same day 
and covered a distance of 210 miles. Prove that his average speed in 
miles per hour was 36 mph.
-/
theorem john_average_speed :
  (210 : ℝ) / (((2 - 8) * 60 + 5 - 15) / 60) = 36 := by
  sorry

end john_average_speed_l186_186552


namespace sum_three_consecutive_odd_integers_l186_186612

theorem sum_three_consecutive_odd_integers (n : ℤ) 
  (h : n + (n + 4) = 150) : n + (n + 2) + (n + 4) = 225 :=
by
  sorry

end sum_three_consecutive_odd_integers_l186_186612


namespace jay_change_l186_186895

def cost_book : ℝ := 25
def cost_pen : ℝ := 4
def cost_ruler : ℝ := 1
def payment : ℝ := 50

theorem jay_change : (payment - (cost_book + cost_pen + cost_ruler) = 20) := sorry

end jay_change_l186_186895


namespace remainder_of_266_div_33_and_8_is_2_l186_186202

theorem remainder_of_266_div_33_and_8_is_2 :
  (266 % 33 = 2) ∧ (266 % 8 = 2) := by
  sorry

end remainder_of_266_div_33_and_8_is_2_l186_186202


namespace find_g_neg1_l186_186521

-- Define the function f and its property of being an odd function
def odd_function (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f (-x) = -f x

-- Define the function g in terms of f
def g (f : ℝ → ℝ) (x : ℝ) : ℝ := f x + 2

-- Given conditions
variables {f : ℝ → ℝ}
variable  (h_odd : odd_function f)
variable  (h_g1 : g f 1 = 1)

-- The statement we want to prove
theorem find_g_neg1 : g f (-1) = 3 :=
sorry

end find_g_neg1_l186_186521


namespace third_side_length_l186_186868

def is_odd (n : ℕ) := n % 2 = 1

theorem third_side_length (x : ℕ) (h1 : 2 + 5 > x) (h2 : x + 2 > 5) (h3 : is_odd x) : x = 5 :=
by
  sorry

end third_side_length_l186_186868


namespace ratio_of_areas_l186_186582

noncomputable def length_field : ℝ := 16
noncomputable def width_field : ℝ := length_field / 2
noncomputable def area_field : ℝ := length_field * width_field
noncomputable def side_pond : ℝ := 4
noncomputable def area_pond : ℝ := side_pond * side_pond
noncomputable def ratio_area_pond_to_field : ℝ := area_pond / area_field

theorem ratio_of_areas :
  ratio_area_pond_to_field = 1 / 8 :=
  by
  sorry

end ratio_of_areas_l186_186582


namespace zero_points_in_intervals_l186_186386

noncomputable def f (x : ℝ) : ℝ :=
  (1 / 3) * x - Real.log x

theorem zero_points_in_intervals :
  (∀ x : ℝ, x ∈ Set.Ioo (1 / Real.exp 1) 1 → f x ≠ 0) ∧
  (∃ x : ℝ, x ∈ Set.Ioo 1 (Real.exp 1) ∧ f x = 0) :=
by
  sorry

end zero_points_in_intervals_l186_186386


namespace sufficient_but_not_necessary_l186_186380

theorem sufficient_but_not_necessary {a b : ℝ} (h1 : a > 1) (h2 : b > 2) :
  (a + b > 3 ∧ a * b > 2) ∧ ¬((a + b > 3 ∧ a * b > 2) → (a > 1 ∧ b > 2)) :=
by
  sorry

end sufficient_but_not_necessary_l186_186380


namespace jay_change_l186_186898

theorem jay_change (book_price pen_price ruler_price payment : ℕ) (h1 : book_price = 25) (h2 : pen_price = 4) (h3 : ruler_price = 1) (h4 : payment = 50) : 
(book_price + pen_price + ruler_price ≤ payment) → (payment - (book_price + pen_price + ruler_price) = 20) :=
by
  intro h
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end jay_change_l186_186898


namespace total_fruits_in_four_baskets_l186_186174

theorem total_fruits_in_four_baskets :
  let apples_basket1 := 9
  let oranges_basket1 := 15
  let bananas_basket1 := 14
  let apples_basket4 := apples_basket1 - 2
  let oranges_basket4 := oranges_basket1 - 2
  let bananas_basket4 := bananas_basket1 - 2
  (apples_basket1 + oranges_basket1 + bananas_basket1) * 3 + 
  (apples_basket4 + oranges_basket4 + bananas_basket4) = 70 := 
by
  intros
  let apples_basket1 := 9
  let oranges_basket1 := 15
  let bananas_basket1 := 14
  let apples_basket4 := apples_basket1 - 2
  let oranges_basket4 := oranges_basket1 - 2
  let bananas_basket4 := bananas_basket1 - 2
  
  -- Calculate the number of fruits in the first three baskets
  let total_fruits_first_three := apples_basket1 + oranges_basket1 + bananas_basket1

  -- Calculate the number of fruits in the fourth basket
  let total_fruits_fourth := apples_basket4 + oranges_basket4 + bananas_basket4

  -- Calculate the total number of fruits
  let total_fruits_all := total_fruits_first_three * 3 + total_fruits_fourth

  have h : total_fruits_all = 70 := by
    calc
      total_fruits_all = (apples_basket1 + oranges_basket1 + bananas_basket1) * 3 + (apples_basket4 + oranges_basket4 + bananas_basket4) : rfl
      ... = (9 + 15 + 14) * 3 + (9 - 2 + (15 - 2) + (14 - 2)) : rfl
      ... = 38 * 3 + 32 : rfl
      ... = 114 + 32 : rfl
      ... = 70 : rfl

  exact h
	
sorry

end total_fruits_in_four_baskets_l186_186174


namespace triangle_right_angled_solve_system_quadratic_roots_real_l186_186227

-- Problem 1
theorem triangle_right_angled (a b c : ℝ) (h : a^2 + b^2 + c^2 - 6 * a - 8 * b - 10 * c + 50 = 0) :
  (a = 3) ∧ (b = 4) ∧ (c = 5) ∧ (a^2 + b^2 = c^2) :=
sorry

-- Problem 2
theorem solve_system (x y : ℝ) (h1 : 3 * x + 4 * y = 30) (h2 : 5 * x + 3 * y = 28) :
  (x = 2) ∧ (y = 6) :=
sorry

-- Problem 3
theorem quadratic_roots_real (m : ℝ) :
  (∃ x : ℝ, ∃ y : ℝ, 3 * x^2 + 4 * x + m = 0 ∧ 3 * y^2 + 4 * y + m = 0) ↔ (m ≤ 4 / 3) :=
sorry

end triangle_right_angled_solve_system_quadratic_roots_real_l186_186227


namespace largest_divisor_of_consecutive_product_l186_186808

theorem largest_divisor_of_consecutive_product (n : ℤ) (h : ∀ k : ℤ, (n = 5 * k)) :
  ∃ d : ℤ, d = 60 ∧ ∀ a : ℤ, divides d (a * (a + 1) * (a + 2) * (a + 3) * (a + 4)) :=
sorry

end largest_divisor_of_consecutive_product_l186_186808


namespace bread_weight_eq_anton_weight_l186_186033

-- Definitions of variables
variables (A B F X : ℝ)

-- Given conditions
axiom cond1 : X + F = A + B
axiom cond2 : B + X = A + F

-- Theorem to prove
theorem bread_weight_eq_anton_weight : X = A :=
by
  sorry

end bread_weight_eq_anton_weight_l186_186033


namespace sherman_weekend_driving_time_l186_186949

def total_driving_time_per_week : ℕ := 9
def commute_time_per_day : ℕ := 1
def work_days_per_week : ℕ := 5
def weekend_days : ℕ := 2

theorem sherman_weekend_driving_time :
  (total_driving_time_per_week - commute_time_per_day * work_days_per_week) / weekend_days = 2 :=
sorry

end sherman_weekend_driving_time_l186_186949


namespace ellipse_center_x_coordinate_l186_186645

theorem ellipse_center_x_coordinate (C : ℝ × ℝ)
  (h1 : C.1 = 3)
  (h2 : 4 ≤ C.2 ∧ C.2 ≤ 12)
  (hx : ∃ F1 F2 : ℝ × ℝ, F1 = (3, 4) ∧ F2 = (3, 12)
    ∧ (F1.1 = F2.1 ∧ F1.2 < F2.2)
    ∧ C = ((F1.1 + F2.1)/2, (F1.2 + F2.2)/2))
  (tangent : ∀ P : ℝ × ℝ, (P.1 - 0) * (P.2 - 0) = 0)
  (ellipse : ∃ a b : ℝ, a > 0 ∧ b > 0
    ∧ ∀ P : ℝ × ℝ,
      (P.1 - C.1)^2/a^2 + (P.2 - C.2)^2/b^2 = 1) :
   C.1 = 3 := sorry

end ellipse_center_x_coordinate_l186_186645


namespace largest_divisor_of_product_of_5_consecutive_integers_l186_186784

/-- What is the largest integer that must divide the product of any 5 consecutive integers? -/
theorem largest_divisor_of_product_of_5_consecutive_integers :
  ∀ n : ℤ, ∃ d : ℤ, d = 24 ∧ d ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  sorry

end largest_divisor_of_product_of_5_consecutive_integers_l186_186784


namespace total_cost_proof_l186_186428

-- Define the prices of items
def price_coffee : ℕ := 4
def price_cake : ℕ := 7
def price_ice_cream : ℕ := 3

-- Define the number of items ordered by Mell and her friends
def mell_coffee : ℕ := 2
def mell_cake : ℕ := 1
def friend_coffee : ℕ := 2
def friend_cake : ℕ := 1
def friend_ice_cream : ℕ := 1
def number_of_friends : ℕ := 2

-- Calculate total cost for Mell
def total_mell : ℕ := (mell_coffee * price_coffee) + (mell_cake * price_cake)

-- Calculate total cost per friend
def total_friend : ℕ := (friend_coffee * price_coffee) + (friend_cake * price_cake) + (friend_ice_cream * price_ice_cream)

-- Calculate total cost for all friends
def total_friends : ℕ := number_of_friends * total_friend

-- Calculate total cost for Mell and her friends
def total_cost : ℕ := total_mell + total_friends

-- The theorem to prove
theorem total_cost_proof : total_cost = 51 := by
  sorry

end total_cost_proof_l186_186428


namespace sin_690_degree_l186_186347

theorem sin_690_degree : sin (690 : ℝ) * (Real.pi / 180) = -(1 / 2) := by
  sorry

end sin_690_degree_l186_186347


namespace convert_spherical_to_rectangular_l186_186845

noncomputable def spherical_to_rectangular (ρ θ φ : ℝ) : ℝ × ℝ × ℝ :=
  (ρ * Real.sin φ * Real.cos θ, ρ * Real.sin φ * Real.sin θ, ρ * Real.cos φ)

theorem convert_spherical_to_rectangular :
  spherical_to_rectangular 15 (3 * Real.pi / 4) (Real.pi / 2) = 
    (-15 * Real.sqrt 2 / 2, 15 * Real.sqrt 2 / 2, 0) :=
by 
  sorry

end convert_spherical_to_rectangular_l186_186845


namespace total_workers_l186_186828

-- Definitions for the conditions in the problem
def avg_salary_all : ℝ := 8000
def num_technicians : ℕ := 7
def avg_salary_technicians : ℝ := 18000
def avg_salary_non_technicians : ℝ := 6000

-- Main theorem stating the total number of workers
theorem total_workers (W : ℕ) :
  (7 * avg_salary_technicians + (W - 7) * avg_salary_non_technicians = W * avg_salary_all) → W = 42 :=
by
  sorry

end total_workers_l186_186828


namespace probability_of_rolling_2_4_or_6_l186_186621

theorem probability_of_rolling_2_4_or_6 (die : Type) [Fintype die] (p : die → Prop) 
  (h_fair : ∀ x : die, 1/(Fintype.card die) = 1/6)
  (h_sides : Fintype.card die = 6) : 
  let favorable : die → Prop := λ x, x = 2 ∨ x = 4 ∨ x = 6 in 
  let P : ℚ := (Fintype.card {x // favorable x}).toRat / (Fintype.card die).toRat
  in P = 1/2 := 
by
  sorry

end probability_of_rolling_2_4_or_6_l186_186621


namespace problem_1_problem_2_0_lt_a_lt_1_problem_2_a_gt_1_l186_186523

noncomputable def f (a x : ℝ) := a^(3 * x + 1)
noncomputable def g (a x : ℝ) := (1 / a)^(5 * x - 2)

variables {a x : ℝ}

theorem problem_1 (h : 0 < a ∧ a < 1) : f a x < 1 ↔ x > -1/3 :=
sorry

theorem problem_2_0_lt_a_lt_1 (h : 0 < a ∧ a < 1) : f a x ≥ g a x ↔ x ≤ 1 / 8 :=
sorry

theorem problem_2_a_gt_1 (h : a > 1) : f a x ≥ g a x ↔ x ≥ 1 / 8 :=
sorry

end problem_1_problem_2_0_lt_a_lt_1_problem_2_a_gt_1_l186_186523


namespace cube_volume_l186_186259

theorem cube_volume (a : ℝ) (h : (a - 1) * (a - 1) * (a + 1) = a^3 - 7) : a^3 = 8 :=
  sorry

end cube_volume_l186_186259


namespace Tony_age_at_end_of_period_l186_186286

-- Definitions based on the conditions in a):
def hours_per_day := 2
def days_worked := 60
def total_earnings := 1140
def earnings_per_hour (age : ℕ) := age

-- The main property we need to prove: Tony's age at the end of the period is 12 years old
theorem Tony_age_at_end_of_period : ∃ age : ℕ, (2 * age * days_worked = total_earnings) ∧ age = 12 :=
by
  sorry

end Tony_age_at_end_of_period_l186_186286


namespace least_three_digit_product_12_l186_186152

theorem least_three_digit_product_12 : 
  ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ (∀ m : ℕ, 100 ≤ m ∧ m < 1000 → (∃ (d1 d2 d3 : ℕ), m = d1 * 100 + d2 * 10 + d3 ∧ d1 * d2 * d3 = 12) → n ≤ m) ∧ (∃ (d1 d2 d3 : ℕ), n = d1 * 100 + d2 * 10 + d3 ∧ d1 * d2 * d3 = 12) :=
by {
  use 134,
  split, linarith,
  split, linarith,
  split,
  { intros m hm hm_digits,
    obtain ⟨d1, d2, d3, h1, h2⟩ := hm_digits,
    cases d1; cases d2; cases d3;
    linarith, },
  { use [1, 3, 4],
    split, refl,
    norm_num }
}

example := least_three_digit_product_12

end least_three_digit_product_12_l186_186152


namespace complement_union_l186_186912

def A := { x : ℤ | ∃ k : ℤ, x = 3 * k + 1 }
def B := { x : ℤ | ∃ k : ℤ, x = 3 * k + 2 }
def U := ℤ

theorem complement_union :
  { x : ℤ | x ∉ (A ∪ B) } = { x : ℤ | ∃ k : ℤ, x = 3 * k } := 
by 
  sorry

end complement_union_l186_186912


namespace moles_of_MgSO4_formed_l186_186204

def moles_of_Mg := 3
def moles_of_H2SO4 := 3

theorem moles_of_MgSO4_formed
  (Mg : ℕ)
  (H2SO4 : ℕ)
  (react : ℕ → ℕ → ℕ × ℕ)
  (initial_Mg : Mg = moles_of_Mg)
  (initial_H2SO4 : H2SO4 = moles_of_H2SO4)
  (balanced_eq : react Mg H2SO4 = (Mg, H2SO4)) :
  (react Mg H2SO4).1 = 3 :=
by
  sorry

end moles_of_MgSO4_formed_l186_186204


namespace inequality_example_l186_186210

theorem inequality_example (a b c : ℝ) : a^2 + 4 * b^2 + 9 * c^2 ≥ 2 * a * b + 3 * a * c + 6 * b * c :=
by
  sorry

end inequality_example_l186_186210


namespace concentric_circles_circumference_difference_l186_186979

theorem concentric_circles_circumference_difference :
  ∀ (radius_diff inner_diameter : ℝ),
  radius_diff = 15 →
  inner_diameter = 50 →
  ((π * (inner_diameter + 2 * radius_diff)) - (π * inner_diameter)) = 30 * π :=
by
  sorry

end concentric_circles_circumference_difference_l186_186979


namespace abs_diff_of_two_numbers_l186_186465

variable {x y : ℝ}

theorem abs_diff_of_two_numbers (h1 : x + y = 40) (h2 : x * y = 396) : abs (x - y) = 4 := by
  sorry

end abs_diff_of_two_numbers_l186_186465


namespace correct_system_of_equations_l186_186547

theorem correct_system_of_equations :
  ∃ (x y : ℝ), (4 * x + y = 5 * y + x) ∧ (5 * x + 6 * y = 16) := sorry

end correct_system_of_equations_l186_186547


namespace find_cost_price_l186_186688

-- Definitions based on conditions
def cost_price (C : ℝ) : Prop := 0.05 * C = 10

-- The theorem stating the problem to be proven
theorem find_cost_price (C : ℝ) (h : cost_price C) : C = 200 :=
by
  sorry

end find_cost_price_l186_186688


namespace triangle_ctg_inequality_l186_186093

noncomputable def ctg (x : Real) := Real.cos x / Real.sin x

theorem triangle_ctg_inequality (α β γ : Real) (h : α + β + γ = Real.pi) :
  ctg α ^ 2 + ctg β ^ 2 + ctg γ ^ 2 ≥ 1 :=
sorry

end triangle_ctg_inequality_l186_186093


namespace min_a_b_sum_l186_186056

theorem min_a_b_sum (a b : ℕ) (x : ℕ → ℕ)
  (h0 : x 1 = a)
  (h1 : x 2 = b)
  (h2 : ∀ n, x (n+2) = x n + x (n+1))
  (h3 : ∃ n, x n = 1000) : a + b = 10 :=
sorry

end min_a_b_sum_l186_186056


namespace find_tricksters_in_16_questions_l186_186326

-- Definitions
def Inhabitant := {i : Nat // i < 65}
def isKnight (i : Inhabitant) : Prop := sorry  -- placeholder for actual definition
def isTrickster (i : Inhabitant) : Prop := sorry -- placeholder for actual definition

-- Conditions
def condition1 : ∀ i : Inhabitant, isKnight i ∨ isTrickster i := sorry
def condition2 : ∃ t1 t2 : Inhabitant, t1 ≠ t2 ∧ isTrickster t1 ∧ isTrickster t2 :=
  sorry
def condition3 : ∀ t1 t2 : Inhabitant, t1 ≠ t2 → ¬(isTrickster t1 ∧ isTrickster t2) → isKnight t1 ∧ isKnight t2 := 
  sorry 
def question (i : Inhabitant) (group : List Inhabitant) : Prop :=
  ∀ j ∈ group, isKnight j

-- Theorem statement
theorem find_tricksters_in_16_questions : ∃ (knight : Inhabitant) (knaves : (Inhabitant × Inhabitant)), 
  isKnight knight ∧ isTrickster knaves.fst ∧ isTrickster knaves.snd ∧ knaves.fst ≠ knaves.snd ∧
  (∀ questionsAsked ≤ 30, sorry) :=
  sorry

end find_tricksters_in_16_questions_l186_186326


namespace ticket_price_l186_186178

theorem ticket_price (P : ℝ) (h_capacity : 50 * P - 24 * P = 208) :
  P = 8 :=
sorry

end ticket_price_l186_186178


namespace problem_l186_186535

theorem problem {x y n : ℝ} 
  (h1 : 2 * x + y = 4) 
  (h2 : (x + y) / 3 = 1) 
  (h3 : x + 2 * y = n) : n = 5 := 
sorry

end problem_l186_186535


namespace garden_enlargement_l186_186491

-- Define the problem conditions
def rect_length : ℝ := 40
def rect_width : ℝ := 20
def rect_area : ℝ := rect_length * rect_width
def rect_perimeter : ℝ := 2 * (rect_length + rect_width)
def square_side : ℝ := rect_perimeter / 4
def square_area : ℝ := square_side * square_side

-- State the theorem to be proved
theorem garden_enlargement : square_area - rect_area = 100 := by
  sorry

end garden_enlargement_l186_186491


namespace complement_of_union_l186_186916

def A : Set ℤ := {x | ∃ k : ℤ, x = 3 * k + 1}
def B : Set ℤ := {x | ∃ k : ℤ, x = 3 * k + 2}
def U : Set ℤ := Set.univ

theorem complement_of_union (x : ℤ) : (x ∉ A ∪ B) ↔ ∃ k : ℤ, x = 3 * k := 
by {
  sorry
}

end complement_of_union_l186_186916


namespace frac_nonneg_iff_pos_l186_186459

theorem frac_nonneg_iff_pos (x : ℝ) : (2 / x ≥ 0) ↔ (x > 0) :=
by sorry

end frac_nonneg_iff_pos_l186_186459


namespace scalene_triangle_geometric_progression_common_ratio_l186_186584

theorem scalene_triangle_geometric_progression_common_ratio
  (b q : ℝ) (hb : b > 0) (h1 : 1 + q > q^2) (h2 : q + q^2 > 1) (h3 : 1 + q^2 > q) :
  0.618 < q ∧ q < 1.618 :=
begin
  sorry
end

end scalene_triangle_geometric_progression_common_ratio_l186_186584


namespace bracelet_arrangements_l186_186342

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem bracelet_arrangements : 
  (factorial 8) / (8 * 2) = 2520 := by
    sorry

end bracelet_arrangements_l186_186342


namespace max_value_of_quadratic_l186_186398

-- Define the quadratic function
def quadratic (x : ℝ) : ℝ := -2 * x^2 + 8

-- State the problem formally in Lean
theorem max_value_of_quadratic : ∀ x : ℝ, quadratic x ≤ quadratic 0 :=
by
  -- Skipping the proof
  sorry

end max_value_of_quadratic_l186_186398


namespace number_of_t_in_T_such_that_f_t_mod_8_eq_0_l186_186251

def f (x : ℤ) : ℤ := x^3 + 2 * x^2 + 3 * x + 4

def T := { n : ℤ | 0 ≤ n ∧ n ≤ 50 }

theorem number_of_t_in_T_such_that_f_t_mod_8_eq_0 : 
  (∃ t ∈ T, f t % 8 = 0) = false := sorry

end number_of_t_in_T_such_that_f_t_mod_8_eq_0_l186_186251


namespace arithmetic_sequence_a10_l186_186890

theorem arithmetic_sequence_a10 
  (S : ℕ → ℕ)
  (a : ℕ → ℕ)
  (h_seq : ∀ n, S n = n * (2 * a 1 + (n - 1) * (a 2 - a 1)) / 2)
  (h_S4 : S 4 = 10)
  (h_S9 : S 9 = 45) :
  a 10 = 10 :=
sorry

end arithmetic_sequence_a10_l186_186890


namespace last_week_profit_min_selling_price_red_beauty_l186_186624

theorem last_week_profit (x kgs_of_red_beauty x_green : ℕ) 
  (purchase_cost_red_beauty_per_kg selling_cost_red_beauty_per_kg 
  purchase_cost_xiangshan_green_per_kg selling_cost_xiangshan_green_per_kg
  total_weight total_cost all_fruits_profit : ℕ) :
  purchase_cost_red_beauty_per_kg = 20 ->
  selling_cost_red_beauty_per_kg = 35 ->
  purchase_cost_xiangshan_green_per_kg = 5 ->
  selling_cost_xiangshan_green_per_kg = 10 ->
  total_weight = 300 ->
  total_cost = 3000 ->
  x * purchase_cost_red_beauty_per_kg + (total_weight - x) * purchase_cost_xiangshan_green_per_kg = total_cost ->
  all_fruits_profit = x * (selling_cost_red_beauty_per_kg - purchase_cost_red_beauty_per_kg) +
  (total_weight - x) * (selling_cost_xiangshan_green_per_kg - purchase_cost_xiangshan_green_per_kg) -> 
  all_fruits_profit = 2500 := sorry

theorem min_selling_price_red_beauty (last_week_profit : ℕ) (x kgs_of_red_beauty x_green damaged_ratio : ℝ) 
  (purchase_cost_red_beauty_per_kg profit_last_week selling_cost_xiangshan_per_kg 
  total_weight total_cost : ℝ) :
  purchase_cost_red_beauty_per_kg = 20 ->
  profit_last_week = 2500 ->
  damaged_ratio = 0.1 ->
  x = 100 ->
  (profit_last_week = 
    x * (35 - purchase_cost_red_beauty_per_kg) + (total_weight - x) * (10 - 5)) ->
  90 * (purchase_cost_red_beauty_per_kg + (last_week_profit - 15 * (total_weight - x) / 90)) ≥ 1500 ->
  profit_last_week / (90 * (90 * (purchase_cost_red_beauty_per_kg + (2500 - 15 * (300 - x) / 90)))) >=
  (36.7 - 20 / purchase_cost_red_beauty_per_kg) :=
  sorry

end last_week_profit_min_selling_price_red_beauty_l186_186624


namespace value_of_m_l186_186382

theorem value_of_m 
  (m : ℝ)
  (h1 : |m - 1| = 1)
  (h2 : m - 2 ≠ 0) : 
  m = 0 :=
  sorry

end value_of_m_l186_186382


namespace problem_statement_l186_186278

noncomputable def count_valid_numbers : Nat :=
  let digits := [1, 2, 3, 4, 5]
  let repeated_digit_choices := 5
  let positions_for_repeated_digits := Nat.choose 5 2
  let cases_for_tens_and_hundreds :=
    2 * 3 + 2 + 1
  let two_remaining_digits_permutations := 2
  repeated_digit_choices * positions_for_repeated_digits * cases_for_tens_and_hundreds * two_remaining_digits_permutations

theorem problem_statement : count_valid_numbers = 800 := by
  sorry

end problem_statement_l186_186278


namespace arithmetic_sequence_9th_term_l186_186242

variables {a_n : ℕ → ℤ}

def arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_9th_term
  (a : ℕ → ℤ) (d : ℤ)
  (h1 : a 3 = 6)
  (h2 : a 6 = 3)
  (h_seq : arithmetic_sequence a d) :
  a 9 = 0 :=
sorry

end arithmetic_sequence_9th_term_l186_186242


namespace problem_1_problem_2_l186_186678

def f (x : ℝ) : ℝ := |x + 1| - |x - 2|

theorem problem_1 : {x : ℝ | f x ≥ 1} = {x : ℝ | x ≥ 1} :=
  sorry

theorem problem_2 (m : ℝ) : (∃ x : ℝ, f x ≥ x^2 - x + m) → m ≤ 5/4 :=
  sorry

end problem_1_problem_2_l186_186678


namespace park_area_calculation_l186_186023

noncomputable def width_of_park := Real.sqrt (9000000 / 65)
noncomputable def length_of_park := 8 * width_of_park

def actual_area_of_park (w l : ℝ) : ℝ := w * l

theorem park_area_calculation :
  let w := width_of_park
  let l := length_of_park
  actual_area_of_park w l = 1107746.48 :=
by
  -- Calculations from solution are provided here directly as conditions and definitions
  sorry

end park_area_calculation_l186_186023


namespace arithmetic_sequence_sum_l186_186597

-- Definitions based on conditions from step a
def first_term : ℕ := 1
def last_term : ℕ := 36
def num_terms : ℕ := 8

-- The problem statement in Lean 4
theorem arithmetic_sequence_sum :
  (num_terms / 2) * (first_term + last_term) = 148 := by
  sorry

end arithmetic_sequence_sum_l186_186597


namespace mary_jenny_red_marble_ratio_l186_186940

def mary_red_marble := 30  -- Given that Mary collects the same as Jenny.
def jenny_red_marble := 30 -- Given
def jenny_blue_marble := 25 -- Given
def anie_red_marble := mary_red_marble + 20 -- Anie's red marbles count
def anie_blue_marble := 2 * jenny_blue_marble -- Anie's blue marbles count
def mary_blue_marble := anie_blue_marble / 2 -- Mary's blue marbles count

theorem mary_jenny_red_marble_ratio : 
  mary_red_marble / jenny_red_marble = 1 :=
by
  sorry

end mary_jenny_red_marble_ratio_l186_186940


namespace speed_of_stream_l186_186294

theorem speed_of_stream (v_d v_u : ℝ) (h_d : v_d = 13) (h_u : v_u = 8) :
  (v_d - v_u) / 2 = 2.5 :=
by
  -- Insert proof steps here
  sorry

end speed_of_stream_l186_186294


namespace num_solutions_l186_186394

-- Define the conditions for the complex number z
def is_solution (z : ℂ) : Prop :=
  (complex.abs z < 30) ∧ (complex.exp z = (z - 1) / (z + 1))

-- State the theorem we want to prove
theorem num_solutions : (finset.univ.filter is_solution).card = 10 :=
sorry

end num_solutions_l186_186394


namespace hydrogen_atoms_in_compound_l186_186634

-- Define atoms and their weights
def C_weight : ℕ := 12
def H_weight : ℕ := 1
def O_weight : ℕ := 16

-- Number of each atom in the compound and total molecular weight
def num_C : ℕ := 4
def num_O : ℕ := 1
def total_weight : ℕ := 65

-- Total mass of carbon and oxygen in the compound
def mass_C_O : ℕ := (num_C * C_weight) + (num_O * O_weight)

-- Mass and number of hydrogen atoms in the compound
def mass_H : ℕ := total_weight - mass_C_O
def num_H : ℕ := mass_H / H_weight

theorem hydrogen_atoms_in_compound : num_H = 1 := by
  sorry

end hydrogen_atoms_in_compound_l186_186634


namespace average_difference_l186_186115

theorem average_difference :
  let avg1 := (10 + 30 + 50) / 3
  let avg2 := (20 + 40 + 6) / 3
  avg1 - avg2 = 8 := by
  sorry

end average_difference_l186_186115


namespace number_of_juniors_l186_186887

variables (J S x : ℕ)

theorem number_of_juniors (h1 : (2 / 5 : ℚ) * J = x)
                          (h2 : (1 / 4 : ℚ) * S = x)
                          (h3 : J + S = 30) :
  J = 11 :=
sorry

end number_of_juniors_l186_186887


namespace tan_ratio_l186_186419

open Real

variables (p q : ℝ)

-- Conditions
def cond1 := (sin p / cos q + sin q / cos p = 2)
def cond2 := (cos p / sin q + cos q / sin p = 3)

-- Proof statement
theorem tan_ratio (hpq : cond1 p q) (hq : cond2 p q) :
  (tan p / tan q + tan q / tan p = 8 / 5) :=
sorry

end tan_ratio_l186_186419


namespace largest_number_l186_186820

def A : ℚ := 97 / 100
def B : ℚ := 979 / 1000
def C : ℚ := 9709 / 10000
def D : ℚ := 907 / 1000
def E : ℚ := 9089 / 10000

theorem largest_number : B > A ∧ B > C ∧ B > D ∧ B > E := by
  sorry

end largest_number_l186_186820


namespace jay_change_l186_186896

def cost_book : ℝ := 25
def cost_pen : ℝ := 4
def cost_ruler : ℝ := 1
def payment : ℝ := 50

theorem jay_change : (payment - (cost_book + cost_pen + cost_ruler) = 20) := sorry

end jay_change_l186_186896


namespace min_pencils_to_ensure_18_l186_186453

theorem min_pencils_to_ensure_18 :
  ∀ (total red green yellow blue brown black : ℕ),
  total = 120 → red = 35 → green = 23 → yellow = 14 → blue = 26 → brown = 11 → black = 11 →
  ∃ (n : ℕ), n = 88 ∧
  (∀ (picked_pencils : ℕ → ℕ), (
    (picked_pencils 0 + picked_pencils 1 + picked_pencils 2 + picked_pencils 3 + picked_pencils 4 + picked_pencils 5 = n) →
    (picked_pencils 0 ≤ red) → (picked_pencils 1 ≤ green) → (picked_pencils 2 ≤ yellow) →
    (picked_pencils 3 ≤ blue) → (picked_pencils 4 ≤ brown) → (picked_pencils 5 ≤ black) →
    (picked_pencils 0 ≥ 18 ∨ picked_pencils 1 ≥ 18 ∨ picked_pencils 2 ≥ 18 ∨ picked_pencils 3 ≥ 18 ∨ picked_pencils 4 ≥ 18 ∨ picked_pencils 5 ≥ 18)
  )) := 
sorry

end min_pencils_to_ensure_18_l186_186453


namespace roberta_has_11_3_left_l186_186104

noncomputable def roberta_leftover_money (initial: ℝ) (shoes: ℝ) (bag: ℝ) (lunch: ℝ) (dress: ℝ) (accessory: ℝ) : ℝ :=
  initial - (shoes + bag + lunch + dress + accessory)

theorem roberta_has_11_3_left :
  roberta_leftover_money 158 45 28 (28 / 4) (62 - 0.15 * 62) (2 * (28 / 4)) = 11.3 :=
by
  sorry

end roberta_has_11_3_left_l186_186104


namespace find_product_x_plus_1_x_minus_1_l186_186518

theorem find_product_x_plus_1_x_minus_1 (x : ℕ) (h : 2^x + 2^x + 2^x + 2^x = 128) : (x + 1) * (x - 1) = 24 := sorry

end find_product_x_plus_1_x_minus_1_l186_186518


namespace correct_operation_l186_186974

theorem correct_operation (x y a b : ℝ) :
  (-2 * x) * (3 * y) = -6 * x * y :=
by
  sorry

end correct_operation_l186_186974


namespace min_sum_matrix_l186_186247

noncomputable def sum_matrix_elements {n : ℕ} (A : matrix (fin n) (fin n) ℕ) : ℕ :=
  ∑ i j, A i j

theorem min_sum_matrix (n : ℕ) (A : matrix (fin n) (fin n) ℕ) 
  (h : ∀ i j, A i j = 0 → (∑ k, A i k) + (∑ k, A k j) ≥ n) :
  sum_matrix_elements A ≥ n * n / 2 :=
by
  sorry

end min_sum_matrix_l186_186247


namespace poly_roots_arith_progression_l186_186473

theorem poly_roots_arith_progression (a b c : ℝ) :
  (∃ x₁ x₂ x₃ : ℝ, -- There exist roots x₁, x₂, x₃
    (x₁ + x₃ = 2 * x₂) ∧ -- Roots form an arithmetic progression
    (x₁ * x₂ * x₃ = -c) ∧ -- Roots satisfy polynomial's product condition
    (x₁ + x₂ + x₃ = -a) ∧ -- Roots satisfy polynomial's sum condition
    ((x₁ * x₂) + (x₂ * x₃) + (x₃ * x₁) = b)) -- Roots satisfy polynomial's sum of products condition
  → (2 * a^3 / 27 - a * b / 3 + c = 0) := 
sorry -- proof is not required

end poly_roots_arith_progression_l186_186473


namespace alex_lost_fish_l186_186035

theorem alex_lost_fish (jacob_initial : ℕ) (alex_catch_ratio : ℕ) (jacob_additional : ℕ) (alex_initial : ℕ) (alex_final : ℕ) : 
  (jacob_initial = 8) → 
  (alex_catch_ratio = 7) → 
  (jacob_additional = 26) →
  (alex_initial = alex_catch_ratio * jacob_initial) →
  (alex_final = (jacob_initial + jacob_additional) - 1) → 
  alex_initial - alex_final = 23 :=
by
  intros
  sorry

end alex_lost_fish_l186_186035


namespace find_tricksters_l186_186324

def inhab_group : Type := { n : ℕ // n < 65 }
def is_knight (i : inhab_group) : Prop := ∀ g : inhab_group, true -- Placeholder for the actual property

theorem find_tricksters (inhabitants : inhab_group → Prop)
  (is_knight : inhab_group → Prop)
  (knight_always_tells_truth : ∀ i : inhab_group, is_knight i → inhabitants i = true)
  (tricksters_2_and_rest_knights : ∃ t1 t2 : inhab_group, t1 ≠ t2 ∧ ¬is_knight t1 ∧ ¬is_knight t2 ∧
    (∀ i : inhab_group, i ≠ t1 → i ≠ t2 → is_knight i)) :
  ∃ find_them : inhab_group → inhab_group → Prop, (∀ q_count : ℕ, q_count ≤ 16) → 
  ∃ t1 t2 : inhab_group, find_them t1 t2 :=
by 
  -- The proof goes here
  sorry

end find_tricksters_l186_186324


namespace cow_manure_growth_percentage_l186_186712

variable (control_height bone_meal_growth_percentage cow_manure_height : ℝ)
variable (bone_meal_height : ℝ := bone_meal_growth_percentage * control_height)
variable (percentage_growth : ℝ := (cow_manure_height / bone_meal_height) * 100)

theorem cow_manure_growth_percentage 
  (h₁ : control_height = 36)
  (h₂ : bone_meal_growth_percentage = 1.25)
  (h₃ : cow_manure_height = 90) :
  percentage_growth = 200 :=
by {
  sorry
}

end cow_manure_growth_percentage_l186_186712


namespace isosceles_triangle_area_l186_186331

noncomputable def area_of_isosceles_triangle (b s : ℝ) (h1 : 10 * 10 + b * b = s * s) (h2 : s + b = 20) : ℝ :=
  1 / 2 * (2 * b) * 10

theorem isosceles_triangle_area (b s : ℝ) (h1 : 10 * 10 + b * b = s * s) (h2 : s + b = 20)
  (h3 : 2 * s + 2 * b = 40) : area_of_isosceles_triangle b s h1 h2 = 75 :=
sorry

end isosceles_triangle_area_l186_186331


namespace surface_area_spherical_segment_l186_186261

-- Definitions based on given conditions
variables {R h : ℝ}

-- The theorem to be proven
theorem surface_area_spherical_segment (h_pos : 0 < h) (R_pos : 0 < R)
  (planes_not_intersect_sphere : h < 2 * R) :
  S = 2 * π * R * h := by
  sorry

end surface_area_spherical_segment_l186_186261


namespace distance_bob_walked_when_met_l186_186260

theorem distance_bob_walked_when_met (distance_XY walk_rate_Yolanda walk_rate_Bob : ℕ)
  (start_time_Yolanda start_time_Bob : ℕ) (y_distance b_distance : ℕ) (t : ℕ)
  (h1 : distance_XY = 65)
  (h2 : walk_rate_Yolanda = 5)
  (h3 : walk_rate_Bob = 7)
  (h4 : start_time_Yolanda = 0)
  (h5 : start_time_Bob = 1)
  (h6 : y_distance = walk_rate_Yolanda * (t + start_time_Bob))
  (h7 : b_distance = walk_rate_Bob * t)
  (h8 : y_distance + b_distance = distance_XY) : 
  b_distance = 35 := 
sorry

end distance_bob_walked_when_met_l186_186260


namespace square_side_length_l186_186972

theorem square_side_length (A : ℝ) (h : A = 169) : ∃ s : ℝ, s^2 = A ∧ s = 13 := by
  sorry

end square_side_length_l186_186972


namespace complement_union_A_B_l186_186924

def A := {x : ℤ | ∃ k : ℤ, x = 3 * k + 1}
def B := {x : ℤ | ∃ k : ℤ, x = 3 * k + 2}
def U := {x : ℤ | True}

theorem complement_union_A_B:
  (U \ (A ∪ B) = {x : ℤ | ∃ k : ℤ, x = 3 * k}) :=
by
  sorry

end complement_union_A_B_l186_186924


namespace max_download_speed_l186_186750

def download_speed (size_GB : ℕ) (time_hours : ℕ) : ℚ :=
  let size_MB := size_GB * 1024
  let time_seconds := time_hours * 60 * 60
  size_MB / time_seconds

theorem max_download_speed (h₁ : size_GB = 360) (h₂ : time_hours = 2) :
  download_speed size_GB time_hours = 51.2 :=
by
  sorry

end max_download_speed_l186_186750


namespace sin_690_eq_neg_half_l186_186352

theorem sin_690_eq_neg_half :
  let rad := Real.pi / 180 in -- Convert degrees to radians
  Real.sin (690 * rad) = -1 / 2 :=
by
  sorry

end sin_690_eq_neg_half_l186_186352


namespace total_money_spent_l186_186749

def candy_bar_cost : ℕ := 14
def cookie_box_cost : ℕ := 39
def total_spent : ℕ := candy_bar_cost + cookie_box_cost

theorem total_money_spent : total_spent = 53 := by
  sorry

end total_money_spent_l186_186749


namespace groupB_avg_weight_eq_141_l186_186581

def initial_group_weight (avg_weight : ℝ) : ℝ := 50 * avg_weight
def groupA_weight_gain : ℝ := 20 * 15
def groupB_weight_gain (x : ℝ) : ℝ := 20 * x

def total_weight (avg_weight : ℝ) (x : ℝ) : ℝ :=
  initial_group_weight avg_weight + groupA_weight_gain + groupB_weight_gain x

def total_avg_weight : ℝ := 46
def num_friends : ℝ := 90

def original_avg_weight : ℝ := total_avg_weight - 12
def final_total_weight : ℝ := num_friends * total_avg_weight

theorem groupB_avg_weight_eq_141 : 
  ∀ (avg_weight : ℝ) (x : ℝ),
    avg_weight = original_avg_weight →
    initial_group_weight avg_weight + groupA_weight_gain + groupB_weight_gain x = final_total_weight →
    avg_weight + x = 141 :=
by 
  intros avg_weight x h₁ h₂
  sorry

end groupB_avg_weight_eq_141_l186_186581


namespace proportion_solve_x_l186_186982

theorem proportion_solve_x :
  (0.75 / x = 5 / 7) → x = 1.05 :=
by
  sorry

end proportion_solve_x_l186_186982


namespace lines_perpendicular_and_intersect_l186_186880

variable {a b : ℝ}

theorem lines_perpendicular_and_intersect 
  (h_ab_nonzero : a * b ≠ 0)
  (h_orthogonal : a + b = 0) : 
  ∃ p, p ≠ 0 ∧ 
    (∀ x y, x = -y * b^2 → y = 0 → p = (x, y)) ∧ 
    (∀ x y, y = x / a^2 → x = 0 → p = (x, y)) ∧ 
    (∀ x y, x = -y * b^2 ∧ y = x / a^2 → x = 0 ∧ y = 0) := 
sorry

end lines_perpendicular_and_intersect_l186_186880


namespace sin_690_eq_neg_half_l186_186350

theorem sin_690_eq_neg_half :
  let rad := Real.pi / 180 in -- Convert degrees to radians
  Real.sin (690 * rad) = -1 / 2 :=
by
  sorry

end sin_690_eq_neg_half_l186_186350


namespace fraction_unshaded_area_l186_186888

theorem fraction_unshaded_area (s : ℝ) :
  let P := (s / 2, 0)
  let Q := (s, s / 2)
  let top_left := (0, s)
  let area_triangle : ℝ := 1 / 2 * (s / 2) * (s / 2)
  let area_square : ℝ := s * s
  let unshaded_area : ℝ := area_square - area_triangle
  let fraction_unshaded : ℝ := unshaded_area / area_square
  fraction_unshaded = 7 / 8 := 
by 
  sorry

end fraction_unshaded_area_l186_186888


namespace solve_for_y_l186_186529

theorem solve_for_y (x y : ℤ) (h1 : x - y = 16) (h2 : x + y = 10) : y = -3 :=
sorry

end solve_for_y_l186_186529


namespace average_age_proof_l186_186967

noncomputable def average_age_when_youngest_born (total_people : ℕ) (current_avg_age : ℚ) (youngest_age : ℚ) : ℚ :=
  (total_people * current_avg_age - youngest_age * (total_people - 1)) / total_people

theorem average_age_proof :
  average_age_when_youngest_born 7 30 3 ≈ 27.43 :=
by
  sorry

end average_age_proof_l186_186967


namespace sum_of_two_squares_l186_186199

theorem sum_of_two_squares (n : ℕ) (h : ∀ m, m = n → n = 2 ∨ (n = 2 * 10 + m) → n % 8 = m) :
  (∃ a b : ℕ, n = a^2 + b^2) ↔ n = 2 := by
  sorry

end sum_of_two_squares_l186_186199


namespace susan_correct_question_percentage_l186_186542

theorem susan_correct_question_percentage (y : ℕ) : 
  (75 * (2 * y - 1) / y) = 
  ((6 * y - 3) / (8 * y) * 100)  :=
sorry

end susan_correct_question_percentage_l186_186542


namespace carlos_and_dana_rest_days_l186_186041

structure Schedule where
  days_of_cycle : ℕ
  work_days : ℕ
  rest_days : ℕ

def carlos : Schedule := ⟨7, 5, 2⟩
def dana : Schedule := ⟨13, 9, 4⟩

def days_both_rest (days_count : ℕ) (sched1 sched2 : Schedule) : ℕ :=
  let lcm_cycle := Nat.lcm sched1.days_of_cycle sched2.days_of_cycle
  let coincidences_in_cycle := 2  -- As derived from the solution
  let full_cycles := days_count / lcm_cycle
  coincidences_in_cycle * full_cycles

theorem carlos_and_dana_rest_days :
  days_both_rest 1500 carlos dana = 32 := by
  sorry

end carlos_and_dana_rest_days_l186_186041


namespace tricksters_identification_l186_186317

variable (Inhabitant : Type)
variable [inhab : Fintype Inhabitant]
variable [decEqInhab : DecidableEq Inhabitant]
variable (knight : Inhabitant → Prop)
variable (trickster : Inhabitant → Prop)

variables (n : ℕ) (q : ℕ)
variable (is_truthful : Inhabitant → Prop)

constant inhabitants_count : 65
constant tricksters_count : 2

-- Define the property that a knight always tells the truth.
axiom knight_truth (x : Inhabitant) : knight x → is_truthful x

-- Define the property that a trickster can tell the truth or lie.
axiom trickster_behavior (x : Inhabitant) : trickster x → (is_truthful x ∨ ¬ is_truthful x)

 -- Define the type of the question which can be asked to an inhabitant.
inductive Question (Inhabitant : Type) : Type
| is_knight : Inhabitant → Question

-- Define the type of the answer to the question.
inductive Answer (Inhabitant : Type) : Type
| yes : Answer
| no :  Answer

-- Define a function that simulates asking a question to an inhabitant.
constant ask : Inhabitant → Question Inhabitant → Answer Inhabitant

noncomputable def find_tricksters (inhabitants : fin inhabitants_count → Inhabitant) : (fin tricksters_count → Inhabitant) :=
sorry

theorem tricksters_identification : 
  ∃ (f : (fin inhabitants_count → Inhabitant) → (fin tricksters_count → Inhabitant)), 
    ∀ inhabitants : fin inhabitants_count → Inhabitant, 
      (∀ (q_list : list (Inhabitant × Question Inhabitant)),
        q_list.length ≤ 30 → 
        let a_list := q_list.map (λ pq, ask (pq.fst) (pq.snd)) in 
        true) ∧ 
      (∃ t1 t2, trickster (f inhabitants) t1 ∧ trickster (f inhabitants) t2) :=
sorry

end tricksters_identification_l186_186317


namespace rectangle_area_proof_l186_186082

noncomputable def rectangle_area (AB AC : ℕ) : ℕ := 
  let BC := Int.sqrt (AC^2 - AB^2)
  AB * BC

theorem rectangle_area_proof (AB AC : ℕ) (h1 : AB = 15) (h2 : AC = 17) :
  rectangle_area AB AC = 120 := by
  rw [rectangle_area, h1, h2]
  norm_num
  sorry

end rectangle_area_proof_l186_186082


namespace expression_value_l186_186187

-- The problem statement definition
def expression := 2 + 3 * 4 - 5 / 5 + 7

-- Theorem statement asserting the final result
theorem expression_value : expression = 20 := 
by sorry

end expression_value_l186_186187


namespace slices_leftover_l186_186339

def total_initial_slices : ℕ := 12 * 2
def bob_slices : ℕ := 12 / 2
def tom_slices : ℕ := 12 / 3
def sally_slices : ℕ := 12 / 6
def jerry_slices : ℕ := 12 / 4
def total_slices_eaten : ℕ := bob_slices + tom_slices + sally_slices + jerry_slices

theorem slices_leftover : total_initial_slices - total_slices_eaten = 9 := by
  sorry

end slices_leftover_l186_186339


namespace product_of_five_consecutive_integers_divisible_by_240_l186_186802

theorem product_of_five_consecutive_integers_divisible_by_240 (n : ℤ) : ∃ k : ℤ, (∏ i in range 5, n + i) = 240 * k := 
sorry

end product_of_five_consecutive_integers_divisible_by_240_l186_186802


namespace sufficient_but_not_necessary_l186_186670

noncomputable def condition_to_bool (a b : ℝ) : Bool :=
a > b ∧ b > 0

theorem sufficient_but_not_necessary (a b : ℝ) (h : condition_to_bool a b) :
  (a > b ∧ b > 0) → (a^2 > b^2) ∧ (∃ a' b' : ℝ, a'^2 > b'^2 ∧ ¬ (a' > b' ∧ b' > 0)) :=
by
  sorry

end sufficient_but_not_necessary_l186_186670


namespace expression_may_not_hold_l186_186070

theorem expression_may_not_hold (a b c : ℝ) (h : a = b) (hc : c = 0) :
  a = b → ¬ (a / c = b / c) := 
by
  intro hab
  intro h_div
  sorry

end expression_may_not_hold_l186_186070


namespace amount_of_first_alloy_used_is_15_l186_186705

-- Definitions of percentages and weights
def chromium_percentage_first_alloy : ℝ := 0.12
def chromium_percentage_second_alloy : ℝ := 0.08
def weight_second_alloy : ℝ := 40
def chromium_percentage_new_alloy : ℝ := 0.0909090909090909
def total_weight_new_alloy (x : ℝ) : ℝ := x + weight_second_alloy
def chromium_content_first_alloy (x : ℝ) : ℝ := chromium_percentage_first_alloy * x
def chromium_content_second_alloy : ℝ := chromium_percentage_second_alloy * weight_second_alloy
def total_chromium_content (x : ℝ) : ℝ := chromium_content_first_alloy x + chromium_content_second_alloy

-- The proof problem
theorem amount_of_first_alloy_used_is_15 :
  ∃ x : ℝ, total_chromium_content x = chromium_percentage_new_alloy * total_weight_new_alloy x ∧ x = 15 :=
by
  sorry

end amount_of_first_alloy_used_is_15_l186_186705


namespace solve_system_of_equations_l186_186281

theorem solve_system_of_equations :
  {p : ℝ × ℝ | 
    (p.1^2 + p.2 + 1) * (p.2^2 + p.1 + 1) = 4 ∧
    (p.1^2 + p.2)^2 + (p.2^2 + p.1)^2 = 2} =
  {(0, 1), (1, 0), 
   ( (-1 + Real.sqrt 5) / 2, (-1 + Real.sqrt 5) / 2),
   ( (-1 - Real.sqrt 5) / 2, (-1 - Real.sqrt 5) / 2) } :=
by
  sorry

end solve_system_of_equations_l186_186281


namespace value_of_a_plus_b_l186_186935

noncomputable def f (x : ℝ) := abs (Real.log (x + 1))

theorem value_of_a_plus_b (a b : ℝ) (h1 : a < b) 
  (h2 : f a = f (- (b + 1) / (b + 2))) 
  (h3 : f (10 * a + 6 * b + 21) = 4 * Real.log 2) : 
  a + b = -11 / 15 := 
by 
  sorry

end value_of_a_plus_b_l186_186935


namespace DanAgeIs12_l186_186846

def DanPresentAge (x : ℕ) : Prop :=
  (x + 18 = 5 * (x - 6))

theorem DanAgeIs12 : ∃ x : ℕ, DanPresentAge x ∧ x = 12 :=
by
  use 12
  unfold DanPresentAge
  sorry

end DanAgeIs12_l186_186846


namespace differential_equation_solution_l186_186892

theorem differential_equation_solution (x y : ℝ) (C : ℝ) :
  (∀ dx dy, 2 * x * y * dx + x^2 * dy = 0) → x^2 * y = C :=
sorry

end differential_equation_solution_l186_186892


namespace transform_equation_l186_186695

open Real

theorem transform_equation (m : ℝ) (x : ℝ) (h1 : x^2 + 4 * x = m) (h2 : (x + 2)^2 = 5) : m = 1 := by
  sorry

end transform_equation_l186_186695


namespace brick_wall_problem_l186_186701

theorem brick_wall_problem : 
  ∀ (B1 B2 B3 B4 B5 : ℕ) (d : ℕ),
  B1 = 38 →
  B1 + B2 + B3 + B4 + B5 = 200 →
  B2 = B1 - d →
  B3 = B1 - 2 * d →
  B4 = B1 - 3 * d →
  B5 = B1 - 4 * d →
  d = 1 :=
by
  intros B1 B2 B3 B4 B5 d h1 h2 h3 h4 h5 h6
  rw [h1] at h2
  sorry

end brick_wall_problem_l186_186701


namespace function_properties_l186_186219

-- Define the function f
def f (x p q : ℝ) : ℝ := x^3 + p * x^2 + 9 * q * x + p + q + 3

-- Stating the main theorem
theorem function_properties (p q : ℝ) :
  ( ∀ x : ℝ, f (-x) p q = -f x p q ) →
  (p = 0 ∧ q = -3 ∧ ∀ x : ℝ, f x 0 (-3) = x^3 - 27 * x ∧
   ( ∀ x ∈ Set.Icc (-1 : ℝ) 4, f x 0 (-3) ≤ 26 ) ∧
   ( ∀ x ∈ Set.Icc (-1 : ℝ) 4, f x 0 (-3) ≥ -54 )) := 
sorry

end function_properties_l186_186219


namespace triangle_count_l186_186502

/-- Define points coordinate constraints and calculate the number of possible triangles. -/
theorem triangle_count (h : ∀ x y : ℕ, 31 * x + y = 2017) : 
  ∑ p in finset.Icc 0 65, ∑ q in finset.Icc 0 65, (p ≠ q) ∧ (p - q) % 2 = 0 = 1056 :=
begin
  sorry
end

end triangle_count_l186_186502


namespace trig_identity_tangent_l186_186397

variable {θ : ℝ}

theorem trig_identity_tangent (h : Real.tan θ = 2) : 
  (Real.sin θ * (Real.cos θ * Real.cos θ - Real.sin θ * Real.sin θ)) / (Real.cos θ - Real.sin θ) = 6 / 5 := 
sorry

end trig_identity_tangent_l186_186397


namespace largest_divisor_of_product_of_five_consecutive_integers_l186_186759

theorem largest_divisor_of_product_of_five_consecutive_integers :
  ∃ n, (∀ k : ℤ, n ∣ (k * (k + 1) * (k + 2) * (k + 3) * (k + 4))) ∧ n = 60 :=
by
  sorry

end largest_divisor_of_product_of_five_consecutive_integers_l186_186759


namespace find_q_l186_186559

noncomputable def q (x : ℝ) : ℝ := -2 * x^4 + 10 * x^3 - 2 * x^2 + 7 * x + 3

theorem find_q :
  ∀ x : ℝ,
  q x + (2 * x^4 - 5 * x^2 + 8 * x + 3) = (10 * x^3 - 7 * x^2 + 15 * x + 6) :=
by
  intro x
  unfold q
  sorry

end find_q_l186_186559


namespace sin_690_degree_l186_186348

theorem sin_690_degree : sin (690 : ℝ) * (Real.pi / 180) = -(1 / 2) := by
  sorry

end sin_690_degree_l186_186348


namespace complement_union_l186_186914

def A := { x : ℤ | ∃ k : ℤ, x = 3 * k + 1 }
def B := { x : ℤ | ∃ k : ℤ, x = 3 * k + 2 }
def U := ℤ

theorem complement_union :
  { x : ℤ | x ∉ (A ∪ B) } = { x : ℤ | ∃ k : ℤ, x = 3 * k } := 
by 
  sorry

end complement_union_l186_186914


namespace student_opinion_change_l186_186037

theorem student_opinion_change (init_enjoy : ℕ) (init_not_enjoy : ℕ)
                               (final_enjoy : ℕ) (final_not_enjoy : ℕ) :
  init_enjoy = 40 ∧ init_not_enjoy = 60 ∧ final_enjoy = 75 ∧ final_not_enjoy = 25 →
  ∃ y_min y_max : ℕ, 
    y_min = 35 ∧ y_max = 75 ∧ (y_max - y_min = 40) :=
by
  sorry

end student_opinion_change_l186_186037


namespace ravi_money_l186_186574

theorem ravi_money (n q d : ℕ) (h1 : q = n + 2) (h2 : d = q + 4) (h3 : n = 6) :
  (n * 5 + q * 25 + d * 10) = 350 := by
  sorry

end ravi_money_l186_186574


namespace larger_number_is_1641_l186_186118

theorem larger_number_is_1641 (L S : ℕ) (h1 : L - S = 1370) (h2 : L = 6 * S + 15) : L = 1641 :=
by
  sorry

end larger_number_is_1641_l186_186118


namespace greatest_perimeter_among_four_pieces_l186_186245

/--
Given an isosceles triangle with a base of 12 inches and a height of 15 inches,
the greatest perimeter among the four pieces of equal area obtained by cutting
the triangle into four smaller triangles is approximately 33.43 inches.
-/
theorem greatest_perimeter_among_four_pieces :
  let base : ℝ := 12
  let height : ℝ := 15
  ∃ (P : ℝ), P = (3 + Real.sqrt (225 + 4) + Real.sqrt (225 + 9)) ∧ abs (P - 33.43) < 0.01 := sorry

end greatest_perimeter_among_four_pieces_l186_186245


namespace candy_problem_l186_186053

variable (total_pieces_eaten : ℕ) (pieces_from_sister : ℕ) (pieces_from_neighbors : ℕ)

theorem candy_problem
  (h1 : total_pieces_eaten = 18)
  (h2 : pieces_from_sister = 13)
  (h3 : total_pieces_eaten = pieces_from_sister + pieces_from_neighbors) :
  pieces_from_neighbors = 5 := by
  -- Add proof here
  sorry

end candy_problem_l186_186053


namespace geometric_seq_properties_l186_186125

-- Declare the conditions and assertions as definitions and the statement to be proved
variable (b : ℕ → ℚ) (r : ℚ)

-- Conditions given in the problem
def condition_b2 : b 2 = 24.5 := sorry
def condition_b5 : b 5 = 196 := sorry

-- Definitions derived from the problem's context
def geo_seq (a : ℚ) (r : ℚ) (n : ℕ) := a * (r ^ (n - 1))
def b2_eq : b 2 = geo_seq (b 1) r 2 := sorry
def b5_eq : b 5 = geo_seq (b 1) r 5 := sorry

-- Resulting sequence properties from the problem
def term_b3 (a : ℚ) : Prop := geo_seq a r 3 = 49
def sum_S4 (a : ℚ) : Prop := ∑ i in (range 4).map (λ i, geo_seq a r (i+1)) = 183.75

-- Main Theorem in Lean statement form
theorem geometric_seq_properties (b1 : ℚ) (r : ℚ) (h1 : geo_seq b1 r 2 = 24.5) (h2 : geo_seq b1 r 5 = 196) : 
  term_b3 b1 ∧ sum_S4 b1 := 
by
  sorry

end geometric_seq_properties_l186_186125


namespace probability_of_cosine_range_l186_186993

noncomputable def probability_cos_in_intervals : ℝ :=
  let total_length := 2 * Real.pi in
  let interval_length := (5 * Real.pi / 6 - Real.pi / 6) + (Real.pi / 6 - -5 * Real.pi / 6) in
  interval_length / total_length

theorem probability_of_cosine_range :
  probability_cos_in_intervals = 2 / 3 :=
by
  sorry

end probability_of_cosine_range_l186_186993


namespace sin_double_theta_eq_three_fourths_l186_186536

theorem sin_double_theta_eq_three_fourths (θ : ℝ) (hθ : 0 < θ ∧ θ < π / 2)
  (h1 : Real.sin (π * Real.cos θ) = Real.cos (π * Real.sin θ)) :
  Real.sin (2 * θ) = 3 / 4 :=
  sorry

end sin_double_theta_eq_three_fourths_l186_186536


namespace time_for_pipe_a_to_fill_l186_186312

noncomputable def pipe_filling_time (a_rate b_rate c_rate : ℝ) (fill_time_together : ℝ) : ℝ := 
  (1 / a_rate)

theorem time_for_pipe_a_to_fill (a_rate b_rate c_rate : ℝ) (fill_time_together : ℝ) 
  (h1 : b_rate = 2 * a_rate) 
  (h2 : c_rate = 2 * b_rate) 
  (h3 : (a_rate + b_rate + c_rate) * fill_time_together = 1) : 
  pipe_filling_time a_rate b_rate c_rate fill_time_together = 42 :=
sorry

end time_for_pipe_a_to_fill_l186_186312


namespace probability_graph_connected_l186_186131

theorem probability_graph_connected :
  let E := (finset.card (finset.univ : finset (fin 20)).choose 2)
  let removed_edges := 35
  let V := 20
  (finset.card (finset.univ : finset (fin E - removed_edges))).choose 16 * V \< (finset.card (finset.univ : finset (fin E))).choose removed_edges / (finset.card (finset.univ : finset (fin (E - removed_edges))).choose 16) = 1 -
  (20 * ((choose 171 16 : ℝ) / choose 190 35)) :=
by
  sorry

end probability_graph_connected_l186_186131


namespace charlie_and_dana_proof_l186_186839

noncomputable def charlie_and_dana_ways 
    (cookies : ℕ) (smoothies : ℕ) (total_items : ℕ) 
    (distinct_charlie : ℕ) 
    (repeatable_dana : ℕ) : ℕ :=
    if cookies = 8 ∧ smoothies = 5 ∧ total_items = 5 ∧ distinct_charlie = 0 
       ∧ repeatable_dana = 0 then 27330 else 0

theorem charlie_and_dana_proof :
  charlie_and_dana_ways 8 5 5 0 0 = 27330 := 
  sorry

end charlie_and_dana_proof_l186_186839


namespace sum_of_three_consecutive_odd_integers_l186_186603

theorem sum_of_three_consecutive_odd_integers (x : ℤ) 
  (h1 : x % 2 = 1)                          -- x is odd
  (h2 : (x + 4) % 2 = 1)                    -- x+4 is odd
  (h3 : x + (x + 4) = 150)                  -- sum of first and third integers is 150
  : x + (x + 2) + (x + 4) = 225 :=          -- sum of the three integers is 225
by
  sorry

end sum_of_three_consecutive_odd_integers_l186_186603


namespace trigonometric_identity_l186_186228

theorem trigonometric_identity (θ : ℝ) (hθ1 : θ ∈ Set.Ico 0 Real.pi) (hθ2 : Real.cos θ * (Real.sin θ + Real.cos θ) = 1) :
  θ = 0 ∨ θ = Real.pi / 4 :=
sorry

end trigonometric_identity_l186_186228


namespace parabola_focus_coordinates_l186_186289

theorem parabola_focus_coordinates (x y p : ℝ) (h : y^2 = 8 * x) : 
  p = 2 → (p, 0) = (2, 0) := 
by 
  sorry

end parabola_focus_coordinates_l186_186289


namespace find_triple_l186_186200

theorem find_triple (A B C : ℕ) (h1 : A^2 + B - C = 100) (h2 : A + B^2 - C = 124) : 
  (A, B, C) = (12, 13, 57) := 
  sorry

end find_triple_l186_186200


namespace percentage_x_equals_twenty_percent_of_487_50_is_65_l186_186486

theorem percentage_x_equals_twenty_percent_of_487_50_is_65
    (x : ℝ)
    (hx : x = 150)
    (y : ℝ)
    (hy : y = 487.50) :
    (∃ (P : ℝ), P * x = 0.20 * y ∧ P * 100 = 65) :=
by
  sorry

end percentage_x_equals_twenty_percent_of_487_50_is_65_l186_186486


namespace triangle_angle_A_l186_186408

theorem triangle_angle_A (A B C : ℝ) (h1 : C = 3 * B) (h2 : B = 30) (h3 : A + B + C = 180) : A = 60 := by
  sorry

end triangle_angle_A_l186_186408


namespace joe_initial_tests_count_l186_186087

theorem joe_initial_tests_count (n S : ℕ) (h1 : S = 45 * n) (h2 : S - 30 = 50 * (n - 1)) : n = 4 := by
  sorry

end joe_initial_tests_count_l186_186087


namespace sin_690_eq_negative_one_half_l186_186358

theorem sin_690_eq_negative_one_half : Real.sin (690 * Real.pi / 180) = -1 / 2 := by
  sorry

end sin_690_eq_negative_one_half_l186_186358


namespace evaluate_sum_of_squares_l186_186059

theorem evaluate_sum_of_squares 
  (x y : ℝ) 
  (h1 : 3 * x + 2 * y = 20) 
  (h2 : 4 * x + y = 25) : (x + y)^2 = 49 :=
  sorry

end evaluate_sum_of_squares_l186_186059


namespace a_alone_can_finish_in_60_days_l186_186980

variables (A B C : ℚ)

noncomputable def a_b_work_rate := A + B = 1/40
noncomputable def a_c_work_rate := A + 1/30 = 1/20

theorem a_alone_can_finish_in_60_days (A B C : ℚ) 
  (h₁ : a_b_work_rate A B) 
  (h₂ : a_c_work_rate A) : 
  A = 1/60 := 
sorry

end a_alone_can_finish_in_60_days_l186_186980


namespace suitable_for_comprehensive_survey_l186_186155

-- Define the four survey options as a custom data type
inductive SurveyOption
  | A : SurveyOption -- Survey on the water quality of the Beijiang River
  | B : SurveyOption -- Survey on the quality of rice dumplings in the market during the Dragon Boat Festival
  | C : SurveyOption -- Survey on the vision of 50 students in a class
  | D : SurveyOption -- Survey by energy-saving lamp manufacturers on the service life of a batch of energy-saving lamps

-- Define feasibility for a comprehensive survey
def isComprehensiveSurvey (option : SurveyOption) : Prop :=
  match option with
  | SurveyOption.A => False
  | SurveyOption.B => False
  | SurveyOption.C => True
  | SurveyOption.D => False

-- The statement to be proven
theorem suitable_for_comprehensive_survey : ∃! o : SurveyOption, isComprehensiveSurvey o := by
  sorry

end suitable_for_comprehensive_survey_l186_186155


namespace part1_part2_l186_186387

noncomputable def f (x : ℝ) : ℝ := 2 * Real.log x - x^2
noncomputable def g (x a : ℝ) : ℝ := x * Real.exp x - (a - 1) * x^2 - x - 2 * Real.log x

theorem part1 : ∃ xₘ : ℝ, (∀ x > 0, f x ≤ f xₘ) ∧ f xₘ = -1 :=
by sorry

theorem part2 (a : ℝ) : (∀ x > 0, f x + g x a ≥ 0) ↔ a ≤ 1 :=
by sorry

end part1_part2_l186_186387


namespace average_billboards_per_hour_l186_186944

-- Define the number of billboards seen in each hour
def billboards_first_hour := 17
def billboards_second_hour := 20
def billboards_third_hour := 23

-- Define the number of hours
def total_hours := 3

-- Prove that the average number of billboards per hour is 20
theorem average_billboards_per_hour : 
  (billboards_first_hour + billboards_second_hour + billboards_third_hour) / total_hours = 20 :=
by
  sorry

end average_billboards_per_hour_l186_186944


namespace sin_690_deg_l186_186361

noncomputable def sin_690_eq_neg_one_half : Prop :=
  sin (690 * real.pi / 180) = -(1 / 2)

theorem sin_690_deg : sin_690_eq_neg_one_half :=
  by sorry

end sin_690_deg_l186_186361


namespace initial_cookies_count_l186_186180

def cookies_left : ℕ := 9
def cookies_eaten : ℕ := 9

theorem initial_cookies_count : cookies_left + cookies_eaten = 18 :=
by sorry

end initial_cookies_count_l186_186180


namespace money_lent_to_B_l186_186295

theorem money_lent_to_B (total_money : ℕ) (interest_A_rate : ℚ) (interest_B_rate : ℚ) (interest_difference : ℚ) (years : ℕ) 
  (x y : ℚ) 
  (h1 : total_money = 10000)
  (h2 : interest_A_rate = 0.15)
  (h3 : interest_B_rate = 0.18)
  (h4 : interest_difference = 360)
  (h5 : years = 2)
  (h6 : y = total_money - x)
  (h7 : ((x * interest_A_rate * years) = ((y * interest_B_rate * years) + interest_difference))) : 
  y = 4000 := 
sorry

end money_lent_to_B_l186_186295


namespace acetone_C_mass_percentage_l186_186015

noncomputable def mass_percentage_C_in_acetone : ℝ :=
  let atomic_mass_C := 12.01
  let atomic_mass_H := 1.01
  let atomic_mass_O := 16.00
  let molar_mass_acetone := (3 * atomic_mass_C) + (6 * atomic_mass_H) + (1 * atomic_mass_O)
  let total_mass_C := 3 * atomic_mass_C
  (total_mass_C / molar_mass_acetone) * 100

theorem acetone_C_mass_percentage :
  abs (mass_percentage_C_in_acetone - 62.01) < 0.01 := by
  sorry

end acetone_C_mass_percentage_l186_186015


namespace johns_daily_calorie_intake_l186_186713

variable (breakfast lunch dinner shake : ℕ)
variable (num_shakes meals_per_day : ℕ)
variable (lunch_inc : ℕ)
variable (dinner_mult : ℕ)

-- Define the conditions from the problem
def john_calories_per_day 
  (breakfast := 500)
  (lunch := breakfast + lunch_inc)
  (dinner := lunch * dinner_mult)
  (shake := 300)
  (num_shakes := 3)
  (lunch_inc := breakfast / 4)
  (dinner_mult := 2)
  : ℕ :=
  breakfast + lunch + dinner + (shake * num_shakes)

theorem johns_daily_calorie_intake : john_calories_per_day = 3275 := by
  sorry

end johns_daily_calorie_intake_l186_186713


namespace triangle_pentagon_side_ratio_l186_186330

theorem triangle_pentagon_side_ratio :
  let triangle_perimeter := 60
  let pentagon_perimeter := 60
  let triangle_side := triangle_perimeter / 3
  let pentagon_side := pentagon_perimeter / 5
  (triangle_side : ℕ) / (pentagon_side : ℕ) = 5 / 3 :=
by
  sorry

end triangle_pentagon_side_ratio_l186_186330


namespace problem_remainder_3_l186_186156

theorem problem_remainder_3 :
  88 % 5 = 3 :=
by
  sorry

end problem_remainder_3_l186_186156


namespace complement_of_union_l186_186915

def A : Set ℤ := {x | ∃ k : ℤ, x = 3 * k + 1}
def B : Set ℤ := {x | ∃ k : ℤ, x = 3 * k + 2}
def U : Set ℤ := Set.univ

theorem complement_of_union (x : ℤ) : (x ∉ A ∪ B) ↔ ∃ k : ℤ, x = 3 * k := 
by {
  sorry
}

end complement_of_union_l186_186915


namespace num_divisible_by_10_in_range_correct_l186_186405

noncomputable def num_divisible_by_10_in_range : ℕ :=
  let a1 := 100
  let d := 10
  let an := 500
  (an - a1) / d + 1

theorem num_divisible_by_10_in_range_correct :
  num_divisible_by_10_in_range = 41 := by
  sorry

end num_divisible_by_10_in_range_correct_l186_186405


namespace length_width_difference_l186_186744

theorem length_width_difference (L W : ℝ) 
  (h1 : W = 1/2 * L) 
  (h2 : L * W = 578) : L - W = 17 :=
sorry

end length_width_difference_l186_186744


namespace neg_p_l186_186064

-- Proposition p : For any x in ℝ, cos x ≤ 1
def p : Prop := ∀ (x : ℝ), Real.cos x ≤ 1

-- Negation of p: There exists an x₀ in ℝ such that cos x₀ > 1
theorem neg_p : ¬p ↔ (∃ (x₀ : ℝ), Real.cos x₀ > 1) := sorry

end neg_p_l186_186064


namespace sunil_total_amount_back_l186_186954

theorem sunil_total_amount_back 
  (CI : ℝ) (P : ℝ) (r : ℝ) (t : ℕ) (total_amount : ℝ) 
  (h1 : CI = 2828.80) 
  (h2 : r = 8) 
  (h3 : t = 2) 
  (h4 : CI = P * ((1 + r / 100) ^ t - 1)) : 
  total_amount = P + CI → 
  total_amount = 19828.80 :=
by
  sorry

end sunil_total_amount_back_l186_186954


namespace math_problem_l186_186040

theorem math_problem : 8 / 4 - 3 - 10 + 3 * 7 = 10 := by
  sorry

end math_problem_l186_186040


namespace abs_eq_condition_l186_186735

theorem abs_eq_condition (a b : ℝ) : |a - b| = |a - 1| + |b - 1| ↔ (a - 1) * (b - 1) ≤ 0 :=
sorry

end abs_eq_condition_l186_186735


namespace largest_divisor_of_five_consecutive_integers_l186_186770

theorem largest_divisor_of_five_consecutive_integers:
  ∀ (n : ℤ), 
    ∃ (a b c : ℤ), 
      (n = 5 * a ∨ n = 5 * a + 1 ∨ n = 5 * a + 2 ∨ n = 5 * a + 3 ∨ n = 5 * a + 4) ∧ 
      (n = 3 * b ∨ n = 3 * b + 1 ∨ n = 3 * b + 2) ∧ 
      (n = 4 * c ∨ n = 4 * c + 1 ∨ n = 4 * c + 2 ∨ n = 4 * c + 3)
      ⊢ 60 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  -- Sorry placeholder for the proof (not needed in the task)
  sorry

end largest_divisor_of_five_consecutive_integers_l186_186770


namespace range_of_a_l186_186577

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | 1 < x ∧ x ≤ 2}
def B (a : ℝ) : Set ℝ := {x : ℝ | x < a}

-- Lean statement asserting the requirement
theorem range_of_a (a : ℝ) (h : A ⊆ B a ∧ A ≠ B a) : 2 < a := by
  sorry

end range_of_a_l186_186577


namespace part1_part2_part3_l186_186420

-- Define the complex number z
def z (m : ℝ) : ℂ :=
  ⟨m^2 - 3*m + 2, m^2 - 1⟩  -- Note: This forms a complex number with real and imaginary parts

-- (1) Proof for z = 0 if and only if m = 1
theorem part1 (m : ℝ) : z m = 0 ↔ m = 1 :=
by sorry

-- (2) Proof for z being a pure imaginary number if and only if m = 2
theorem part2 (m : ℝ) : (z m).re = 0 ∧ (z m).im ≠ 0 ↔ m = 2 :=
by sorry

-- (3) Proof for the point corresponding to z being in the second quadrant if and only if 1 < m < 2
theorem part3 (m : ℝ) : (z m).re < 0 ∧ (z m).im > 0 ↔ 1 < m ∧ m < 2 :=
by sorry

end part1_part2_part3_l186_186420


namespace total_cost_calculation_l186_186425

-- Definitions
def coffee_price : ℕ := 4
def cake_price : ℕ := 7
def ice_cream_price : ℕ := 3

def mell_coffee_qty : ℕ := 2
def mell_cake_qty : ℕ := 1
def friends_coffee_qty : ℕ := 2
def friends_cake_qty : ℕ := 1
def friends_ice_cream_qty : ℕ := 1

def total_coffee_qty : ℕ := 3 * mell_coffee_qty
def total_cake_qty : ℕ := 3 * mell_cake_qty
def total_ice_cream_qty : ℕ := 2 * friends_ice_cream_qty

def total_cost : ℕ := total_coffee_qty * coffee_price + total_cake_qty * cake_price + total_ice_cream_qty * ice_cream_price

-- Theorem Statement
theorem total_cost_calculation : total_cost = 51 := by
  sorry

end total_cost_calculation_l186_186425


namespace complement_union_l186_186911

def A := { x : ℤ | ∃ k : ℤ, x = 3 * k + 1 }
def B := { x : ℤ | ∃ k : ℤ, x = 3 * k + 2 }
def U := ℤ

theorem complement_union :
  { x : ℤ | x ∉ (A ∪ B) } = { x : ℤ | ∃ k : ℤ, x = 3 * k } := 
by 
  sorry

end complement_union_l186_186911


namespace largest_divisor_of_5_consecutive_integers_l186_186793

theorem largest_divisor_of_5_consecutive_integers :
  ∀ n : ℤ, ∃ k : ℤ, (product_of_5_consecutive_integers n = k) ∧ (60 ∣ k) 
:= sorry

end largest_divisor_of_5_consecutive_integers_l186_186793


namespace binary_to_decimal_l186_186191

/-- The binary number 1011 (base 2) equals 11 (base 10). -/
theorem binary_to_decimal : (1 * 2^3 + 0 * 2^2 + 1 * 2^1 + 1 * 2^0) = 11 := by
  sorry

end binary_to_decimal_l186_186191


namespace kim_shirts_left_l186_186718

theorem kim_shirts_left (initial_dozens : ℕ) (fraction_given : ℚ) (num_pairs : ℕ)
  (h1 : initial_dozens = 4) 
  (h2 : fraction_given = 1 / 3)
  (h3 : num_pairs = initial_dozens * 12)
  (h4 : num_pairs * fraction_given  = (16 : ℕ)):
  48 - ((num_pairs * fraction_given).toNat) = 32 :=
by 
  sorry

end kim_shirts_left_l186_186718


namespace distance_left_to_drive_l186_186434

theorem distance_left_to_drive (total_distance : ℕ) (distance_driven : ℕ) 
  (h1 : total_distance = 78) (h2 : distance_driven = 32) : 
  total_distance - distance_driven = 46 := by
  sorry

end distance_left_to_drive_l186_186434


namespace tourists_left_l186_186990

noncomputable def tourists_remaining {initial remaining poisoned recovered : ℕ} 
  (h1 : initial = 30)
  (h2 : remaining = initial - 2)
  (h3 : poisoned = remaining / 2)
  (h4 : recovered = poisoned / 7)
  (h5 : remaining % 2 = 0) -- ensuring even division for / 2
  (h6 : poisoned % 7 = 0) -- ensuring even division for / 7
  : ℕ :=
  remaining - poisoned + recovered

theorem tourists_left 
  (initial remaining poisoned recovered : ℕ) 
  (h1 : initial = 30)
  (h2 : remaining = initial - 2)
  (h3 : poisoned = remaining / 2)
  (h4 : recovered = poisoned / 7)
  (h5 : remaining % 2 = 0) -- ensuring even division for / 2
  (h6 : poisoned % 7 = 0) -- ensuring even division for / 7
  : tourists_remaining h1 h2 h3 h4 h5 h6 = 16 :=
  by
  sorry

end tourists_left_l186_186990


namespace largest_divisor_of_consecutive_product_l186_186806

theorem largest_divisor_of_consecutive_product (n : ℤ) (h : ∀ k : ℤ, (n = 5 * k)) :
  ∃ d : ℤ, d = 60 ∧ ∀ a : ℤ, divides d (a * (a + 1) * (a + 2) * (a + 3) * (a + 4)) :=
sorry

end largest_divisor_of_consecutive_product_l186_186806


namespace spherical_circle_radius_l186_186586

theorem spherical_circle_radius:
  (∀ (θ : Real), ∃ (r : Real), r = 1 * Real.sin (Real.pi / 6)) → ∀ (θ : Real), r = 1 / 2 := by
  sorry

end spherical_circle_radius_l186_186586


namespace sum_of_squares_divisible_by_three_l186_186698

theorem sum_of_squares_divisible_by_three {a b : ℤ} 
  (h : 3 ∣ (a^2 + b^2)) : (3 ∣ a ∧ 3 ∣ b) :=
by 
  sorry

end sum_of_squares_divisible_by_three_l186_186698


namespace geometric_sequence_sum_l186_186422

-- We state the main problem in Lean as a theorem.
theorem geometric_sequence_sum (S : ℕ → ℕ) (S_4_eq : S 4 = 8) (S_8_eq : S 8 = 24) : S 12 = 88 :=
  sorry

end geometric_sequence_sum_l186_186422


namespace eggs_for_dinner_l186_186224

-- Definitions of the conditions
def eggs_for_breakfast := 2
def eggs_for_lunch := 3
def total_eggs := 6

-- The quantity of eggs for dinner needs to be proved
theorem eggs_for_dinner :
  ∃ x : ℕ, x + eggs_for_breakfast + eggs_for_lunch = total_eggs ∧ x = 1 :=
by
  sorry

end eggs_for_dinner_l186_186224


namespace find_b50_l186_186667

noncomputable def T (n : ℕ) : ℝ := if n = 1 then 2 else 2 / (6 * n - 5)

noncomputable def b (n : ℕ) : ℝ :=
  if n = 1 then 2 else T n - T (n - 1)

theorem find_b50 : b 50 = -6 / 42677.5 := by sorry

end find_b50_l186_186667


namespace children_tickets_sold_l186_186823

-- Given conditions
variables (A C : ℕ) -- A represents the number of adult tickets, C the number of children tickets.
variables (total_money total_tickets price_adult price_children : ℕ)
variables (total_money_eq : total_money = 104)
variables (total_tickets_eq : total_tickets = 21)
variables (price_adult_eq : price_adult = 6)
variables (price_children_eq : price_children = 4)
variables (money_eq : price_adult * A + price_children * C = total_money)
variables (tickets_eq : A + C = total_tickets)

-- Problem statement: prove that C = 11
theorem children_tickets_sold : C = 11 :=
by
  -- Necessary Lean code to handle proof here (omitting proof details as instructed)
  sorry

end children_tickets_sold_l186_186823


namespace solve_quadratic_eqn_l186_186588

theorem solve_quadratic_eqn :
  ∀ x : ℝ, (x - 2) * (x + 3) = 0 ↔ (x = 2 ∨ x = -3) :=
by
  intros
  simp
  sorry

end solve_quadratic_eqn_l186_186588


namespace product_of_five_consecutive_integers_divisible_by_240_l186_186801

theorem product_of_five_consecutive_integers_divisible_by_240 (n : ℤ) : ∃ k : ℤ, (∏ i in range 5, n + i) = 240 * k := 
sorry

end product_of_five_consecutive_integers_divisible_by_240_l186_186801


namespace arithmetic_sequence_sum_l186_186417

variable {a_n : ℕ → ℕ} -- the arithmetic sequence

-- Define condition
def condition (a : ℕ → ℕ) : Prop :=
  a 1 + a 5 + a 9 = 18

-- The sum of the first n terms of arithmetic sequence is S_n
def S (n : ℕ) (a : ℕ → ℕ) : ℕ :=
  (n * (a 1 + a n)) / 2

-- The goal is to prove that S 9 = 54
theorem arithmetic_sequence_sum (h : condition a_n) : S 9 a_n = 54 :=
sorry

end arithmetic_sequence_sum_l186_186417


namespace value_of_a0_plus_a8_l186_186858

/-- Theorem stating the value of a0 + a8 from the given polynomial equation -/
theorem value_of_a0_plus_a8 (a_0 a_8 : ℤ) :
  (∀ x : ℤ, (1 + x) ^ 10 = a_0 + a_1 * (1 - x) + a_2 * (1 - x) ^ 2 + 
              a_3 * (1 - x) ^ 3 + a_4 * (1 - x) ^ 4 + a_5 * (1 - x) ^ 5 +
              a_6 * (1 - x) ^ 6 + a_7 * (1 - x) ^ 7 + a_8 * (1 - x) ^ 8 + 
              a_9 * (1 - x) ^ 9 + a_10 * (1 - x) ^ 10) →
  a_0 + a_8 = 1204 :=
by
  sorry

end value_of_a0_plus_a8_l186_186858


namespace triangle_angle_sum_l186_186545

theorem triangle_angle_sum (A B C : Type) (angle_ABC angle_BAC angle_ACB : ℝ)
  (h₁ : angle_ABC = 110)
  (h₂ : angle_BAC = 45)
  (triangle_sum : angle_ABC + angle_BAC + angle_ACB = 180) :
  angle_ACB = 25 :=
by
  sorry

end triangle_angle_sum_l186_186545


namespace tourists_left_l186_186987

theorem tourists_left (initial_tourists eaten_by_anacondas poisoned_fraction recover_fraction : ℕ) 
(h_initial : initial_tourists = 30) 
(h_eaten : eaten_by_anacondas = 2)
(h_poisoned_fraction : poisoned_fraction = 2)
(h_recover_fraction : recover_fraction = 7) :
  initial_tourists - eaten_by_anacondas - (initial_tourists - eaten_by_anacondas) / poisoned_fraction + (initial_tourists - eaten_by_anacondas) / poisoned_fraction / recover_fraction = 16 :=
by
  sorry

end tourists_left_l186_186987


namespace completion_time_l186_186994

theorem completion_time (total_work : ℕ) (initial_num_men : ℕ) (initial_efficiency : ℝ)
  (new_num_men : ℕ) (new_efficiency : ℝ) :
  total_work = 12 ∧ initial_num_men = 4 ∧ initial_efficiency = 1.5 ∧
  new_num_men = 6 ∧ new_efficiency = 2.0 →
  total_work / (new_num_men * new_efficiency) = 1 :=
by
  sorry

end completion_time_l186_186994


namespace wholesale_cost_l186_186995

variable (W R P : ℝ)

-- Conditions
def retail_price := R = 1.20 * W
def employee_discount := P = 0.95 * R
def employee_payment := P = 228

-- Theorem statement
theorem wholesale_cost (H1 : retail_price R W) (H2 : employee_discount P R) (H3 : employee_payment P) : W = 200 :=
by
  sorry

end wholesale_cost_l186_186995


namespace count_right_triangles_l186_186068

theorem count_right_triangles: 
  ∃ n : ℕ, n = 9 ∧ ∃ (a b : ℕ), a^2 + b^2 = (b+2)^2 ∧ b < 100 ∧ a > 0 ∧ b > 0 := by
  sorry

end count_right_triangles_l186_186068


namespace complement_union_A_B_l186_186920

def A := {x : ℤ | ∃ k : ℤ, x = 3 * k + 1}
def B := {x : ℤ | ∃ k : ℤ, x = 3 * k + 2}
def U := {x : ℤ | True}

theorem complement_union_A_B:
  (U \ (A ∪ B) = {x : ℤ | ∃ k : ℤ, x = 3 * k}) :=
by
  sorry

end complement_union_A_B_l186_186920


namespace intersection_points_of_circle_and_line_l186_186046

theorem intersection_points_of_circle_and_line :
  (∃ y, (4, y) ∈ {p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 = 25}) → 
  ∃ s : Finset (ℝ × ℝ), s.card = 2 ∧ ∀ p ∈ s, (p.1 = 4 ∧ (p.1 ^ 2 + p.2 ^ 2 = 25)) :=
by
  sorry

end intersection_points_of_circle_and_line_l186_186046


namespace inv_113_mod_114_l186_186198

theorem inv_113_mod_114 :
  (113 * 113) % 114 = 1 % 114 :=
by
  sorry

end inv_113_mod_114_l186_186198


namespace weight_of_b_l186_186271

variable {a b c : ℝ}

theorem weight_of_b (h1 : (a + b + c) / 3 = 45)
                    (h2 : (a + b) / 2 = 40)
                    (h3 : (b + c) / 2 = 43) :
                    b = 31 := by
  sorry

end weight_of_b_l186_186271


namespace min_possible_A_div_C_l186_186531

theorem min_possible_A_div_C (x : ℝ) (A C : ℝ) (h1 : x^2 + (1/x)^2 = A) (h2 : x + 1/x = C) (h3 : 0 < A) (h4 : 0 < C) :
  ∃ (C : ℝ), C = Real.sqrt 2 ∧ (∀ B, (x^2 + (1/x)^2 = B) → (x + 1/x = C) → (B / C = 0 → B = 0)) :=
by
  sorry

end min_possible_A_div_C_l186_186531


namespace complement_union_A_B_l186_186929

def is_element_of_set_A (x : ℤ) : Prop := ∃ k : ℤ, x = 3 * k + 1
def is_element_of_set_B (x : ℤ) : Prop := ∃ k : ℤ, x = 3 * k + 2
def is_element_of_complement_U (x : ℤ) : Prop := ∃ k : ℤ, x = 3 * k

theorem complement_union_A_B :
  {x : ℤ | ¬ (is_element_of_set_A x ∨ is_element_of_set_B x)} = {x : ℤ | is_element_of_complement_U x} :=
by
  sorry

end complement_union_A_B_l186_186929


namespace john_days_to_lose_weight_l186_186412

noncomputable def john_calories_intake : ℕ := 1800
noncomputable def john_calories_burned : ℕ := 2300
noncomputable def calories_to_lose_1_pound : ℕ := 4000
noncomputable def pounds_to_lose : ℕ := 10

theorem john_days_to_lose_weight :
  (john_calories_burned - john_calories_intake) * (pounds_to_lose * calories_to_lose_1_pound / (john_calories_burned - john_calories_intake)) = 80 :=
by
  sorry

end john_days_to_lose_weight_l186_186412


namespace total_books_l186_186140

-- Definitions based on the conditions
def TimBooks : ℕ := 44
def SamBooks : ℕ := 52
def AlexBooks : ℕ := 65

-- Theorem to be proven
theorem total_books : TimBooks + SamBooks + AlexBooks = 161 := by
  sorry

end total_books_l186_186140


namespace divisor_is_18_l186_186970

def dividend : ℕ := 165
def quotient : ℕ := 9
def remainder : ℕ := 3

theorem divisor_is_18 (divisor : ℕ) : dividend = quotient * divisor + remainder → divisor = 18 :=
by sorry

end divisor_is_18_l186_186970


namespace largest_divisor_of_product_of_five_consecutive_integers_l186_186758

theorem largest_divisor_of_product_of_five_consecutive_integers :
  ∃ n, (∀ k : ℤ, n ∣ (k * (k + 1) * (k + 2) * (k + 3) * (k + 4))) ∧ n = 60 :=
by
  sorry

end largest_divisor_of_product_of_five_consecutive_integers_l186_186758


namespace find_tricksters_within_30_questions_l186_186322

/-- 
Given 65 inhabitants in a village where:
- Two inhabitants are tricksters and the rest are knights.
- Knights always tell the truth.
- Tricksters can either tell the truth or lie.
- One can show any inhabitant a list of some group of inhabitants (which can consist of one person)
  and ask if all of them are knights.

Prove that it is possible to find both tricksters with no more than 30 questions.
-/
theorem find_tricksters_within_30_questions :
  ∃ (ask_knights : (inhabitants : fin 65) → list (fin 65) → Prop),
  (∀ (i j : fin 65), i ≠ j → ask_knights i [j] = true → (inhabitants[j] = knight) ∨ (inhabitants[j] = trickster))
  ∧ ∀ (inhabitants : fin 65),
  (∃ S : finset (fin 65), S.card = 2 ∧ 
  (∀ i, i ∈ S → inhabitants[i] = trickster) ∧ 
  by asking no more than 30 questions,
  you can identify both tricksters.

end find_tricksters_within_30_questions_l186_186322


namespace number_of_people_in_group_l186_186293

theorem number_of_people_in_group (P : ℕ) : 
  (∃ (P : ℕ), 0 < P ∧ (364 / P - 1 = 364 / (P + 2))) → P = 26 :=
by
  sorry

end number_of_people_in_group_l186_186293


namespace property1_property2_l186_186213

/-- Given sequence a_n defined as a_n = 3(n^2 + n) + 7 -/
def a (n : ℕ) : ℕ := 3 * (n^2 + n) + 7

/-- Property 1: Out of any five consecutive terms in the sequence, only one term is divisible by 5. -/
theorem property1 (n : ℕ) : (∃ k : ℕ, a (5 * k + 2) % 5 = 0) ∧ (∀ k : ℕ, ∀ r : ℕ, r ≠ 2 → a (5 * k + r) % 5 ≠ 0) :=
by
  sorry

/-- Property 2: None of the terms in this sequence is a cube of an integer. -/
theorem property2 (n : ℕ) : ¬(∃ t : ℕ, a n = t^3) :=
by
  sorry

end property1_property2_l186_186213


namespace find_b_minus_a_l186_186022

theorem find_b_minus_a (a b : ℤ) (h1 : a * b = 2 * (a + b) + 11) (h2 : b = 7) : b - a = 2 :=
by sorry

end find_b_minus_a_l186_186022


namespace minimum_digits_for_divisibility_l186_186595

theorem minimum_digits_for_divisibility :
  ∃ n : ℕ, (10 * 2013 + n) % 2520 = 0 ∧ n < 1000 :=
sorry

end minimum_digits_for_divisibility_l186_186595


namespace find_minimum_n_l186_186464

noncomputable def a_seq (n : ℕ) : ℕ := 3 ^ (n - 1)

noncomputable def S_n (n : ℕ) : ℕ := 1 / 2 * (3 ^ n - 1)

theorem find_minimum_n (S_n : ℕ → ℕ) (n : ℕ) :
  (3^n - 1) / 2 > 1000 → n = 7 := 
sorry

end find_minimum_n_l186_186464


namespace ratio_aerobics_to_weight_training_l186_186901

def time_spent_exercising : ℕ := 250
def time_spent_aerobics : ℕ := 150
def time_spent_weight_training : ℕ := 100

theorem ratio_aerobics_to_weight_training :
    (time_spent_aerobics / gcd time_spent_aerobics time_spent_weight_training) = 3 ∧
    (time_spent_weight_training / gcd time_spent_aerobics time_spent_weight_training) = 2 :=
by
    sorry

end ratio_aerobics_to_weight_training_l186_186901


namespace problem_statement_l186_186138

theorem problem_statement (x y : ℝ) (h1 : 3 = 0.15 * x) (h2 : 3 = 0.25 * y) : x - y = 8 :=
by
  sorry

end problem_statement_l186_186138


namespace at_least_one_div_by_5_l186_186263

-- Define natural numbers and divisibility by 5
def is_div_by_5 (n : ℕ) : Prop := ∃ k : ℕ, n = 5 * k

-- Proposition: If a, b are natural numbers and ab is divisible by 5, then at least one of a or b must be divisible by 5.
theorem at_least_one_div_by_5 (a b : ℕ) (h_ab : is_div_by_5 (a * b)) : is_div_by_5 a ∨ is_div_by_5 b :=
  by
    sorry

end at_least_one_div_by_5_l186_186263


namespace percentage_correct_l186_186258

theorem percentage_correct (x : ℕ) (h : x > 0) : 
  (4 * x / (6 * x) * 100 = 200 / 3) :=
by
  sorry

end percentage_correct_l186_186258


namespace expansion_coefficient_a2_l186_186514

theorem expansion_coefficient_a2 (z x : ℂ) 
  (h : z = 1 + I) : 
  ∃ a_0 a_1 a_2 a_3 a_4 : ℂ,
    (z + x)^4 = a_4 * x^4 + a_3 * x^3 + a_2 * x^2 + a_1 * x + a_0
    ∧ a_2 = 12 * I :=
by
  sorry

end expansion_coefficient_a2_l186_186514


namespace investment_return_formula_l186_186462

noncomputable def investment_return (x : ℕ) (x_pos : x > 0) : ℝ :=
  if x = 1 then 0.5
  else 2 ^ (x - 2)

theorem investment_return_formula (x : ℕ) (x_pos : x > 0) : investment_return x x_pos = 2 ^ (x - 2) := 
by
  sorry

end investment_return_formula_l186_186462


namespace slices_leftover_l186_186338

def total_initial_slices : ℕ := 12 * 2
def bob_slices : ℕ := 12 / 2
def tom_slices : ℕ := 12 / 3
def sally_slices : ℕ := 12 / 6
def jerry_slices : ℕ := 12 / 4
def total_slices_eaten : ℕ := bob_slices + tom_slices + sally_slices + jerry_slices

theorem slices_leftover : total_initial_slices - total_slices_eaten = 9 := by
  sorry

end slices_leftover_l186_186338


namespace parabola_midpoint_length_squared_l186_186948

theorem parabola_midpoint_length_squared :
  ∀ (A B : ℝ × ℝ), 
  (∃ (x y : ℝ), A = (x, 3*x^2 + 4*x + 2) ∧ B = (-x, -(3*x^2 + 4*x + 2)) ∧ ((A.1 + B.1) / 2 = 0) ∧ ((A.2 + B.2) / 2 = 0)) →
  dist A B^2 = 8 :=
by
  sorry

end parabola_midpoint_length_squared_l186_186948


namespace find_the_number_l186_186629

theorem find_the_number (x : ℤ) (h : 2 + x = 6) : x = 4 :=
sorry

end find_the_number_l186_186629


namespace sum_of_three_consecutive_odd_integers_l186_186602

theorem sum_of_three_consecutive_odd_integers (x : ℤ) 
  (h1 : x % 2 = 1)                          -- x is odd
  (h2 : (x + 4) % 2 = 1)                    -- x+4 is odd
  (h3 : x + (x + 4) = 150)                  -- sum of first and third integers is 150
  : x + (x + 2) + (x + 4) = 225 :=          -- sum of the three integers is 225
by
  sorry

end sum_of_three_consecutive_odd_integers_l186_186602


namespace apples_count_l186_186302

variable (A : ℕ)

axiom h1 : 134 = 80 + 54
axiom h2 : A + 98 = 134

theorem apples_count : A = 36 :=
by
  sorry

end apples_count_l186_186302


namespace solution_is_correct_l186_186193

def valid_triple (a b c : ℕ) : Prop :=
  (Nat.gcd a 20 = b) ∧ (Nat.gcd b 15 = c) ∧ (Nat.gcd a c = 5)

def is_solution_set (triples : Set (ℕ × ℕ × ℕ)) : Prop :=
  ∀ a b c, (a, b, c) ∈ triples ↔ 
    (valid_triple a b c) ∧ 
    ((∃ k, a = 20 * k ∧ b = 20 ∧ c = 5) ∨
    (∃ k, a = 20 * k - 10 ∧ b = 10 ∧ c = 5) ∨
    (∃ k, a = 10 * k - 5 ∧ b = 5 ∧ c = 5))

theorem solution_is_correct : ∃ S, is_solution_set S :=
sorry

end solution_is_correct_l186_186193


namespace part_I_extreme_value_part_II_range_of_a_l186_186061

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * (x - 1)^2 + Real.log x + 1

theorem part_I_extreme_value (a : ℝ) (h1 : a = -1/4) :
  (∀ x > 0, f a x ≤ f a 2) ∧ f a 2 = 3/4 + Real.log 2 :=
sorry

theorem part_II_range_of_a (a : ℝ) :
  (∀ x ≥ 1, f a x ≤ x) ↔ a ≤ 0 :=
sorry

end part_I_extreme_value_part_II_range_of_a_l186_186061


namespace largest_divisor_of_consecutive_product_l186_186810

theorem largest_divisor_of_consecutive_product (n : ℤ) (h : ∀ k : ℤ, (n = 5 * k)) :
  ∃ d : ℤ, d = 60 ∧ ∀ a : ℤ, divides d (a * (a + 1) * (a + 2) * (a + 3) * (a + 4)) :=
sorry

end largest_divisor_of_consecutive_product_l186_186810


namespace triangle_construction_condition_l186_186344

variable (varrho_a varrho_b m_c : ℝ)

theorem triangle_construction_condition :
  (∃ (triangle : Type) (ABC : triangle)
    (r_a : triangle → ℝ)
    (r_b : triangle → ℝ)
    (h_from_C : triangle → ℝ),
      r_a ABC = varrho_a ∧
      r_b ABC = varrho_b ∧
      h_from_C ABC = m_c)
  ↔ 
  (1 / m_c = 1 / 2 * (1 / varrho_a + 1 / varrho_b)) :=
sorry

end triangle_construction_condition_l186_186344


namespace cost_price_l186_186691

theorem cost_price (C : ℝ) : 
  (0.05 * C = 350 - 340) → C = 200 :=
by
  assume h1 : 0.05 * C = 10
  sorry

end cost_price_l186_186691


namespace rectangle_area_l186_186080

theorem rectangle_area (AB AC : ℝ) (AB_eq : AB = 15) (AC_eq : AC = 17) : 
  ∃ (BC : ℝ), (BC^2 = AC^2 - AB^2) ∧ (AB * BC = 120) := 
by
  -- Assuming necessary geometry axioms, such as the definition of a rectangle and Pythagorean theorem.
  sorry

end rectangle_area_l186_186080


namespace value_of_otimes_difference_l186_186421

def otimes (a b : ℚ) : ℚ := (a^3) / (b^2)

theorem value_of_otimes_difference :
  otimes (otimes 2 3) 4 - otimes 2 (otimes 3 4) = - 1184 / 243 := 
by
  sorry

end value_of_otimes_difference_l186_186421


namespace find_tricksters_l186_186329

structure Inhabitant :=
  (isKnight : Prop)

constants (inhabitants : Fin 65 → Inhabitant)
          (tricksters : Fin 2 → Fin 65)

axiom two_tricksters_unique :
  ∃! (t₁ t₂ : Fin 65), t₁ ≠ t₂ ∧ t₁ ∈ (tricksters : Set (Fin 65)) ∧ t₂ ∈ (tricksters : Set (Fin 65))

axiom valid_question :
  ∀ (a : Fin 65) (group : Set (Fin 65)), (inhabitants a).isKnight → 
  (∀ i ∈ group, (inhabitants i).isKnight) ↔ 
  (knight a).isKnight

theorem find_tricksters :
  ∃ (q : ℕ), q ≤ 16 ∧
  ∃ (t₁ t₂ : Fin 65), t₁ ≠ t₂ ∧ t₁ ∈ (tricksters : Set (Fin 65)) ∧ t₂ ∈ (tricksters : Set (Fin 65)) :=
sorry

end find_tricksters_l186_186329


namespace probability_graph_connected_after_removing_edges_l186_186133

theorem probability_graph_connected_after_removing_edges:
  let n := 20
  let edges_removed := 35
  let total_edges := (n * (n - 1)) / 2
  let remaining_edges := total_edges - edges_removed
  let binom := λ a b : ℕ, nat.choose a b
  1 - (20 * (binom 171 16) / (binom 190 35)) = 1 - (20 * (binom remaining_edges (remaining_edges - edges_removed)) / (binom total_edges edges_removed)) := sorry

end probability_graph_connected_after_removing_edges_l186_186133


namespace min_value_a_l186_186661

theorem min_value_a (a : ℝ) :
  (∀ x : ℝ, |x + a| - |x + 1| ≤ 2 * a) → a ≥ 1 / 3 :=
by
  sorry

end min_value_a_l186_186661


namespace cricket_bat_profit_percentage_l186_186484

-- Definitions for the problem conditions
def selling_price : ℝ := 850
def profit : ℝ := 255
def cost_price : ℝ := selling_price - profit
def expected_profit_percentage : ℝ := 42.86

-- The theorem to be proven
theorem cricket_bat_profit_percentage : 
  (profit / cost_price) * 100 = expected_profit_percentage :=
by 
  sorry

end cricket_bat_profit_percentage_l186_186484


namespace min_x_y_sum_l186_186865

theorem min_x_y_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/(x+1) + 1/y = 1/2) : x + y ≥ 7 := 
by 
  sorry

end min_x_y_sum_l186_186865


namespace sum_of_three_consecutive_odd_integers_l186_186605

theorem sum_of_three_consecutive_odd_integers (x : ℤ) 
  (h1 : x % 2 = 1)                          -- x is odd
  (h2 : (x + 4) % 2 = 1)                    -- x+4 is odd
  (h3 : x + (x + 4) = 150)                  -- sum of first and third integers is 150
  : x + (x + 2) + (x + 4) = 225 :=          -- sum of the three integers is 225
by
  sorry

end sum_of_three_consecutive_odd_integers_l186_186605


namespace find_z_given_x4_l186_186962

theorem find_z_given_x4 (k : ℝ) (z : ℝ) (x : ℝ) :
  (7 * 4 = k / 2^3) → (7 * z = k / x^3) → (x = 4) → (z = 0.5) :=
by
  intro h1 h2 h3
  sorry

end find_z_given_x4_l186_186962


namespace sum_of_three_consecutive_odd_integers_l186_186617

theorem sum_of_three_consecutive_odd_integers (a : ℤ) (h₁ : a + (a + 4) = 150) :
  (a + (a + 2) + (a + 4) = 225) :=
by
  sorry

end sum_of_three_consecutive_odd_integers_l186_186617


namespace total_balloons_l186_186644

theorem total_balloons (allan_balloons : ℕ) (jake_balloons : ℕ)
  (h_allan : allan_balloons = 2)
  (h_jake : jake_balloons = 1) :
  allan_balloons + jake_balloons = 3 :=
by 
  -- Provide proof here
  sorry

end total_balloons_l186_186644


namespace inequality_solution_l186_186852

theorem inequality_solution
  : {x : ℝ | (x^2 / (x + 2)^2) ≥ 0} = {x : ℝ | x ≠ -2} :=
by
  sorry

end inequality_solution_l186_186852


namespace problem_statement_l186_186934

noncomputable def roots (a b : ℝ) (coef1 coef2 : ℝ) :=
  ∃ x : ℝ, (x = a ∨ x = b) ∧ x^2 + coef1 * x + coef2 = 0

theorem problem_statement
  (a b c d : ℝ)
  (h1 : a + b = -57)
  (h2 : a * b = 1)
  (h3 : c + d = 57)
  (h4 : c * d = 1) :
  (a + c) * (b + c) * (a - d) * (b - d) = 0 := 
by
  sorry

end problem_statement_l186_186934


namespace new_player_weight_l186_186469

theorem new_player_weight 
  (original_players : ℕ)
  (original_avg_weight : ℝ)
  (new_players : ℕ)
  (new_avg_weight : ℝ)
  (new_total_weight : ℝ) :
  original_players = 20 →
  original_avg_weight = 180 →
  new_players = 21 →
  new_avg_weight = 181.42857142857142 →
  new_total_weight = 3810 →
  (new_total_weight - original_players * original_avg_weight) = 210 :=
by
  intros
  sorry

end new_player_weight_l186_186469


namespace find_n_l186_186141

theorem find_n (n : ℕ) (h : 20 * n = Nat.factorial (n - 1)) : n = 6 :=
by {
  sorry
}

end find_n_l186_186141


namespace mean_of_remaining_four_numbers_l186_186270

theorem mean_of_remaining_four_numbers (a b c d : ℝ) :
  (a + b + c + d + 105) / 5 = 92 → (a + b + c + d) / 4 = 88.75 :=
by
  intro h
  sorry

end mean_of_remaining_four_numbers_l186_186270


namespace gdp_scientific_notation_l186_186400

theorem gdp_scientific_notation (trillion : ℕ) (five_year_growth : ℝ) (gdp : ℝ) :
  trillion = 10^12 ∧ 1 ≤ gdp / 10^14 ∧ gdp / 10^14 < 10 ∧ gdp = 121 * 10^12 → gdp = 1.21 * 10^14
:= by
  sorry

end gdp_scientific_notation_l186_186400


namespace lottery_win_amount_l186_186902

theorem lottery_win_amount (total_tax : ℝ) (federal_tax_rate : ℝ) (local_tax_rate : ℝ) (tax_paid : ℝ) :
  total_tax = tax_paid →
  federal_tax_rate = 0.25 →
  local_tax_rate = 0.15 →
  tax_paid = 18000 →
  ∃ x : ℝ, x = 49655 :=
by
  intros h1 h2 h3 h4
  use (tax_paid / (federal_tax_rate + local_tax_rate * (1 - federal_tax_rate))), by
    norm_num at h1 h2 h3 h4
    sorry

end lottery_win_amount_l186_186902


namespace ball_max_height_l186_186167

theorem ball_max_height : 
  (∃ t : ℝ, 
    ∀ u : ℝ, -16 * u ^ 2 + 80 * u + 35 ≤ -16 * t ^ 2 + 80 * t + 35 ∧ 
    -16 * t ^ 2 + 80 * t + 35 = 135) :=
sorry

end ball_max_height_l186_186167


namespace jesus_squares_l186_186440

theorem jesus_squares (J : ℕ) (linden_squares : ℕ) (pedro_squares : ℕ)
  (h1 : linden_squares = 75)
  (h2 : pedro_squares = 200)
  (h3 : pedro_squares = J + linden_squares + 65) : 
  J = 60 := 
by
  sorry

end jesus_squares_l186_186440


namespace novice_experienced_parts_l186_186488

variables (x y : ℕ)

theorem novice_experienced_parts :
  (y - x = 30) ∧ (x + 2 * y = 180) :=
sorry

end novice_experienced_parts_l186_186488


namespace percent_is_50_l186_186137

variable (cats hogs percent : ℕ)
variable (hogs_eq_3cats : hogs = 3 * cats)
variable (hogs_eq_75 : hogs = 75)

theorem percent_is_50
  (cats_minus_5_percent_eq_10 : (cats - 5) * percent = 1000)
  (cats_eq_25 : cats = 25) :
  percent = 50 := by
  sorry

end percent_is_50_l186_186137


namespace arc_length_of_sector_l186_186211

theorem arc_length_of_sector (n r : ℝ) (h_angle : n = 60) (h_radius : r = 3) : 
  (n * Real.pi * r / 180) = Real.pi :=
by 
  sorry

end arc_length_of_sector_l186_186211


namespace probability_two_even_balls_l186_186628

theorem probability_two_even_balls
  (total_balls : ℕ)
  (even_balls : ℕ)
  (h_total : total_balls = 16)
  (h_even : even_balls = 8)
  (first_draw : ℕ → ℚ)
  (second_draw : ℕ → ℚ)
  (h_first : first_draw even_balls = even_balls / total_balls)
  (h_second : second_draw (even_balls - 1) = (even_balls - 1) / (total_balls - 1)) :
  (first_draw even_balls) * (second_draw (even_balls - 1)) = 7 / 30 := 
sorry

end probability_two_even_balls_l186_186628


namespace find_least_x_divisible_by_17_l186_186658

theorem find_least_x_divisible_by_17 (x k : ℕ) (h : x + 2 = 17 * k) : x = 15 :=
sorry

end find_least_x_divisible_by_17_l186_186658


namespace Problem1_l186_186856

theorem Problem1 (x y : ℝ) (h : x^2 + y^2 = 1) : x^6 + 3*x^2*y^2 + y^6 = 1 := 
by
  sorry

end Problem1_l186_186856


namespace distance_CG_l186_186957

theorem distance_CG (a b c : ℝ) (h : c ^ 2 = a ^ 2 + b ^ 2) :
  ∃ (CG : ℝ), CG = (sqrt((a^4 - a^2*b^2 + b^4) / c^2)) :=
by
  sorry

end distance_CG_l186_186957


namespace light_glow_duration_l186_186007

-- Define the conditions
def total_time_seconds : ℕ := 4969
def glow_times : ℚ := 292.29411764705884

-- Prove the equivalent statement
theorem light_glow_duration :
  (total_time_seconds / glow_times) = 17 := by
  sorry

end light_glow_duration_l186_186007


namespace find_abc_value_l186_186729

noncomputable def given_conditions (a b c : ℝ) : Prop :=
  (a * b / (a + b) = 2) ∧ (b * c / (b + c) = 5) ∧ (c * a / (c + a) = 9)

theorem find_abc_value (a b c : ℝ) (h : given_conditions a b c) :
  a * b * c / (a * b + b * c + c * a) = 90 / 73 :=
sorry

end find_abc_value_l186_186729


namespace uncovered_area_is_52_l186_186027

-- Define the dimensions of the rectangles
def smaller_rectangle_length : ℕ := 4
def smaller_rectangle_width : ℕ := 2
def larger_rectangle_length : ℕ := 10
def larger_rectangle_width : ℕ := 6

-- Define the areas of both rectangles
def area_larger_rectangle : ℕ := larger_rectangle_length * larger_rectangle_width
def area_smaller_rectangle : ℕ := smaller_rectangle_length * smaller_rectangle_width

-- Define the area of the uncovered region
def area_uncovered_region : ℕ := area_larger_rectangle - area_smaller_rectangle

-- State the theorem
theorem uncovered_area_is_52 : area_uncovered_region = 52 := by sorry

end uncovered_area_is_52_l186_186027


namespace lieutenant_age_l186_186177

theorem lieutenant_age (n x : ℕ)
  (h1 : ∃ n, n.rows = n ∧ n.soldiers_per_row_initial = n + 5)
  (h2 : total_soldiers : n * (n + 5)) 
  (h3 : total_soldiers_second_alignment : x * (n + 9)) : x = 24 :=
by
  sorry

end lieutenant_age_l186_186177


namespace soda_quantity_difference_l186_186637

noncomputable def bottles_of_diet_soda := 19
noncomputable def bottles_of_regular_soda := 60
noncomputable def bottles_of_cherry_soda := 35
noncomputable def bottles_of_orange_soda := 45

theorem soda_quantity_difference : 
  (max bottles_of_regular_soda (max bottles_of_diet_soda 
    (max bottles_of_cherry_soda bottles_of_orange_soda)) 
  - min bottles_of_regular_soda (min bottles_of_diet_soda 
    (min bottles_of_cherry_soda bottles_of_orange_soda))) = 41 := 
by
  sorry

end soda_quantity_difference_l186_186637


namespace find_tricksters_l186_186318

theorem find_tricksters (inhabitants : Fin 65 → Prop) (is_knight : Fin 65 → Prop)
    (total_inhabitants : ∀ n, inhabitants n)
    (knights : ∀ n, is_knight n → inhabitants n)
    (tricksters_count : (∑ n, if ¬ is_knight n then 1 else 0) = 2)
    (knights_count : (∑ n, if is_knight n then 1 else 0) = 63)
    (knight_truth : ∀ n, is_knight n → ∀ l : list (Fin 65), (∀ m ∈ l, is_knight m) ↔ true)
    (ask_question : ∀ n, inhabitants n → ∀ l : list (Fin 65), bool) :
  ∃ (find_tricksters_function : (Fin 65 → Prop) → (Fin 65 → bool) → (list (Fin 65))) ,
    (length (find_tricksters_function inhabitants ask_question) ≤ 2) →
    (length (find_tricksters_function inhabitants ask_question) = 2) ∧
    ∀ t ∈ (find_tricksters_function inhabitants ask_question), ¬ is_knight t :=
by sorry

end find_tricksters_l186_186318


namespace perpendicular_lines_m_l186_186537

theorem perpendicular_lines_m (m : ℝ) : 
    (∀ x y : ℝ, x - 2 * y + 5 = 0 → 
                2 * x + m * y - 6 = 0 → 
                (1 / 2) * (-2 / m) = -1) → 
    m = 1 :=
by
  intros
  -- proof goes here
  sorry

end perpendicular_lines_m_l186_186537


namespace pieces_present_l186_186113

-- Define the pieces and their counts in a standard chess set
def total_pieces := 32
def missing_pieces := 12
def missing_kings := 1
def missing_queens := 2
def missing_knights := 3
def missing_pawns := 6

-- The theorem statement that we need to prove
theorem pieces_present : 
  (total_pieces - (missing_kings + missing_queens + missing_knights + missing_pawns)) = 20 :=
by
  sorry

end pieces_present_l186_186113


namespace total_seeds_eaten_l186_186335

theorem total_seeds_eaten :
  ∃ (first second third : ℕ), 
  first = 78 ∧ 
  second = 53 ∧ 
  third = second + 30 ∧ 
  first + second + third = 214 :=
by
  use 78, 53, 83
  split; norm_num
  split; norm_num
  split; norm_num
  sorry

end total_seeds_eaten_l186_186335


namespace coordinates_of_point_in_fourth_quadrant_l186_186381

-- Define the conditions as separate hypotheses
def point_in_fourth_quadrant (x y : ℝ) : Prop := (x > 0) ∧ (y < 0)

-- State the main theorem
theorem coordinates_of_point_in_fourth_quadrant
  (x y : ℝ) (h1 : point_in_fourth_quadrant x y) (h2 : |x| = 3) (h3 : |y| = 5) :
  (x = 3) ∧ (y = -5) :=
by
  sorry

end coordinates_of_point_in_fourth_quadrant_l186_186381


namespace range_of_a_l186_186958

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, |x + 3| + |x - 1| ≥ a^2 - 3 * a) ↔ -1 ≤ a ∧ a ≤ 4 :=
by
  sorry

end range_of_a_l186_186958


namespace tetrahedron_side_length_l186_186028

theorem tetrahedron_side_length (s : ℝ) (area : ℝ) (d : ℝ) :
  area = 16 → s^2 = area → d = s * Real.sqrt 2 → 4 * Real.sqrt 2 = d :=
by
  intros _ h1 h2
  sorry

end tetrahedron_side_length_l186_186028


namespace intersection_point_unique_l186_186959

theorem intersection_point_unique (k : ℝ) :
  (∃ y : ℝ, k = -2 * y^2 - 3 * y + 5) ∧ (∀ y₁ y₂ : ℝ, y₁ ≠ y₂ → -2 * y₁^2 - 3 * y₁ + 5 ≠ k ∨ -2 * y₂^2 - 3 * y₂ + 5 ≠ k)
  ↔ k = 49 / 8 := 
by sorry

end intersection_point_unique_l186_186959


namespace exist_m_n_l186_186090

theorem exist_m_n (p : ℕ) [hp : Fact (Nat.Prime p)] (h : 5 < p) :
  ∃ m n : ℕ, (m + n < p ∧ p ∣ (2^m * 3^n - 1)) := sorry

end exist_m_n_l186_186090


namespace Sn_divisible_by_7_l186_186442

open Real

-- Definitions for x1, x2, x3
def x1 := (2 * sin (π / 7))^2
def x2 := (2 * sin (2 * π / 7))^2
def x3 := (2 * sin (3 * π / 7))^2

-- Defining Sn
noncomputable def S (n : ℕ) : ℝ := x1^n + x2^n + x3^n

-- Main theorem statement
theorem Sn_divisible_by_7 (n : ℕ) : 7 ^ (n / 3) ∣ S n :=
  sorry

end Sn_divisible_by_7_l186_186442


namespace sum_of_three_consecutive_odd_integers_l186_186616

theorem sum_of_three_consecutive_odd_integers (a : ℤ) (h₁ : a + (a + 4) = 150) :
  (a + (a + 2) + (a + 4) = 225) :=
by
  sorry

end sum_of_three_consecutive_odd_integers_l186_186616


namespace determinant_problem_l186_186528

variables {p q r s : ℝ}

theorem determinant_problem
  (h : p * s - q * r = 5) :
  p * (4 * r + 2 * s) - (4 * p + 2 * q) * r = 10 := 
sorry

end determinant_problem_l186_186528


namespace final_value_of_A_l186_186519

theorem final_value_of_A (A : ℝ) (h1: A = 15) (h2: A = -A + 5) : A = -10 :=
sorry

end final_value_of_A_l186_186519


namespace ticket_cost_is_correct_l186_186018

-- Conditions
def total_amount_raised : ℕ := 620
def number_of_tickets_sold : ℕ := 155

-- Definition of cost per ticket
def cost_per_ticket : ℕ := total_amount_raised / number_of_tickets_sold

-- The theorem to be proven
theorem ticket_cost_is_correct : cost_per_ticket = 4 :=
by
  sorry

end ticket_cost_is_correct_l186_186018


namespace avg_age_of_new_persons_l186_186952

-- We define the given conditions
def initial_persons : ℕ := 12
def initial_avg_age : ℝ := 16
def new_persons : ℕ := 12
def new_avg_age : ℝ := 15.5

-- Define the total initial age
def total_initial_age : ℝ := initial_persons * initial_avg_age

-- Define the total number of persons after new persons join
def total_persons_after_join : ℕ := initial_persons + new_persons

-- Define the total age after new persons join
def total_age_after_join : ℝ := total_persons_after_join * new_avg_age

-- We wish to prove that the average age of the new persons who joined is 15
theorem avg_age_of_new_persons : 
  (total_initial_age + new_persons * 15) = total_age_after_join :=
sorry

end avg_age_of_new_persons_l186_186952


namespace area_of_EFGH_l186_186376

def short_side_length : ℕ := 4
def long_side_length : ℕ := short_side_length * 2
def number_of_rectangles : ℕ := 4
def larger_rectangle_length : ℕ := short_side_length
def larger_rectangle_width : ℕ := number_of_rectangles * long_side_length

theorem area_of_EFGH :
  (larger_rectangle_length * larger_rectangle_width) = 128 := 
  by
    sorry

end area_of_EFGH_l186_186376


namespace boxes_needed_l186_186563

-- Define the given conditions

def red_pencils : ℕ := 20
def blue_pencils : ℕ := 2 * red_pencils
def yellow_pencils : ℕ := 40
def green_pencils : ℕ := red_pencils + blue_pencils
def total_pencils : ℕ := red_pencils + blue_pencils + green_pencils + yellow_pencils
def pencils_per_box : ℕ := 20

-- Lean theorem statement to prove the number of boxes needed is 8

theorem boxes_needed : total_pencils / pencils_per_box = 8 :=
by
  -- This is where the proof would go
  sorry

end boxes_needed_l186_186563


namespace base_k_representation_l186_186055

theorem base_k_representation (k : ℕ) (hk : k > 0) (hk_exp : 7 / 51 = (2 * k + 3 : ℚ) / (k ^ 2 - 1 : ℚ)) : k = 16 :=
by {
  sorry
}

end base_k_representation_l186_186055


namespace factorize_expression_l186_186851

theorem factorize_expression (x y : ℝ) : x^2 + x * y + x = x * (x + y + 1) := 
by
  sorry

end factorize_expression_l186_186851


namespace find_tricksters_l186_186319

theorem find_tricksters (inhabitants : Fin 65 → Prop) (is_knight : Fin 65 → Prop)
    (total_inhabitants : ∀ n, inhabitants n)
    (knights : ∀ n, is_knight n → inhabitants n)
    (tricksters_count : (∑ n, if ¬ is_knight n then 1 else 0) = 2)
    (knights_count : (∑ n, if is_knight n then 1 else 0) = 63)
    (knight_truth : ∀ n, is_knight n → ∀ l : list (Fin 65), (∀ m ∈ l, is_knight m) ↔ true)
    (ask_question : ∀ n, inhabitants n → ∀ l : list (Fin 65), bool) :
  ∃ (find_tricksters_function : (Fin 65 → Prop) → (Fin 65 → bool) → (list (Fin 65))) ,
    (length (find_tricksters_function inhabitants ask_question) ≤ 2) →
    (length (find_tricksters_function inhabitants ask_question) = 2) ∧
    ∀ t ∈ (find_tricksters_function inhabitants ask_question), ¬ is_knight t :=
by sorry

end find_tricksters_l186_186319


namespace unit_prices_min_chess_sets_l186_186591

-- Define the conditions and prove the unit prices.
theorem unit_prices (x y : ℝ) 
  (h1 : 6 * x + 5 * y = 190)
  (h2 : 8 * x + 10 * y = 320) : 
  x = 15 ∧ y = 20 :=
by
  sorry

-- Define the conditions for the budget and prove the minimum number of chess sets.
theorem min_chess_sets (x y : ℝ) (m : ℕ)
  (hx : x = 15)
  (hy : y = 20)
  (number_sets : m + (100 - m) = 100)
  (budget : 15 * ↑m + 20 * ↑(100 - m) ≤ 1800) :
  m ≥ 40 :=
by
  sorry

end unit_prices_min_chess_sets_l186_186591


namespace Amy_finish_time_l186_186480

-- Definitions and assumptions based on conditions
def Patrick_time : ℕ := 60
def Manu_time : ℕ := Patrick_time + 12
def Amy_time : ℕ := Manu_time / 2

-- Theorem statement to be proved
theorem Amy_finish_time : Amy_time = 36 :=
by
  sorry

end Amy_finish_time_l186_186480


namespace normal_vector_to_line_l186_186307

theorem normal_vector_to_line : 
  ∀ (x y : ℝ), x - 3 * y + 6 = 0 → (1, -3) = (1, -3) :=
by
  intros x y h_line
  sorry

end normal_vector_to_line_l186_186307


namespace slices_leftover_is_9_l186_186340

-- Conditions and definitions
def total_pizzas : ℕ := 2
def slices_per_pizza : ℕ := 12
def bob_ate : ℕ := slices_per_pizza / 2
def tom_ate : ℕ := slices_per_pizza / 3
def sally_ate : ℕ := slices_per_pizza / 6
def jerry_ate : ℕ := slices_per_pizza / 4

-- Calculate total slices eaten and left over
def total_slices_eaten : ℕ := bob_ate + tom_ate + sally_ate + jerry_ate
def total_slices_available : ℕ := total_pizzas * slices_per_pizza
def slices_leftover : ℕ := total_slices_available - total_slices_eaten

-- Theorem to prove the number of slices left over
theorem slices_leftover_is_9 : slices_leftover = 9 := by
  -- Proof: omitted, add relevant steps here
  sorry

end slices_leftover_is_9_l186_186340


namespace largest_divisor_of_five_consecutive_integers_l186_186772

theorem largest_divisor_of_five_consecutive_integers:
  ∀ (n : ℤ), 
    ∃ (a b c : ℤ), 
      (n = 5 * a ∨ n = 5 * a + 1 ∨ n = 5 * a + 2 ∨ n = 5 * a + 3 ∨ n = 5 * a + 4) ∧ 
      (n = 3 * b ∨ n = 3 * b + 1 ∨ n = 3 * b + 2) ∧ 
      (n = 4 * c ∨ n = 4 * c + 1 ∨ n = 4 * c + 2 ∨ n = 4 * c + 3)
      ⊢ 60 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  -- Sorry placeholder for the proof (not needed in the task)
  sorry

end largest_divisor_of_five_consecutive_integers_l186_186772


namespace combined_cost_of_one_item_l186_186444

-- Definitions representing the given conditions
def initial_amount : ℝ := 50
def final_amount : ℝ := 14
def mangoes_purchased : ℕ := 6
def apple_juice_purchased : ℕ := 6

-- Hypothesis: The cost of mangoes and apple juice are the same
variables (M A : ℝ)

-- Total amount spent
def amount_spent : ℝ := initial_amount - final_amount

-- Combined number of items
def total_items : ℕ := mangoes_purchased + apple_juice_purchased

-- Lean statement to prove the combined cost of one mango and one carton of apple juice is $3
theorem combined_cost_of_one_item (h : mangoes_purchased * M + apple_juice_purchased * A = amount_spent) :
    (amount_spent / total_items) = (3 : ℝ) :=
by
  sorry

end combined_cost_of_one_item_l186_186444


namespace pencils_inequalities_l186_186136

theorem pencils_inequalities (x y : ℕ) :
  (3 * x < 48 ∧ 48 < 4 * x) ∧ (4 * y < 48 ∧ 48 < 5 * y) :=
sorry

end pencils_inequalities_l186_186136


namespace hall_area_l186_186130

variable {L W : ℝ}

theorem hall_area (h1 : W = 1 / 2 * L) (h2 : L - W = 17) : L * W = 578 := by
  sorry

end hall_area_l186_186130


namespace ratio_b_to_c_l186_186824

-- Define the ages of a, b, and c as A, B, and C respectively
variables (A B C : ℕ)

-- Given conditions
def condition1 := A = B + 2
def condition2 := B = 10
def condition3 := A + B + C = 27

-- The question: Prove the ratio of b's age to c's age is 2:1
theorem ratio_b_to_c : condition1 ∧ condition2 ∧ condition3 → B / C = 2 := 
by
  sorry

end ratio_b_to_c_l186_186824


namespace largest_divisor_of_5_consecutive_integers_l186_186753

theorem largest_divisor_of_5_consecutive_integers :
  ∃ d : ℤ, (∀ n : ℤ, d ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4))) ∧ d = 120 :=
sorry

end largest_divisor_of_5_consecutive_integers_l186_186753


namespace hexagon_perimeter_eq_4_sqrt_3_over_3_l186_186733

noncomputable def hexagon_perimeter (s : ℝ) : ℝ :=
  6 * s

theorem hexagon_perimeter_eq_4_sqrt_3_over_3 :
  ∀ (s : ℝ), (∃ s, (3 * Real.sqrt 3 / 2) * s^2 = s) → hexagon_perimeter s = 4 * Real.sqrt 3 / 3 :=
by
  simp
  sorry

end hexagon_perimeter_eq_4_sqrt_3_over_3_l186_186733


namespace triangle_perimeter_ABC_l186_186120

noncomputable def perimeter_triangle (AP PB r : ℕ) (hAP : AP = 23) (hPB : PB = 27) (hr : r = 21) : ℕ :=
  2 * (50 + 245 / 2)

theorem triangle_perimeter_ABC (AP PB r : ℕ) 
  (hAP : AP = 23) 
  (hPB : PB = 27) 
  (hr : r = 21) : 
  perimeter_triangle AP PB r hAP hPB hr = 345 :=
by
  sorry

end triangle_perimeter_ABC_l186_186120


namespace complement_union_A_B_l186_186925

def is_element_of_set_A (x : ℤ) : Prop := ∃ k : ℤ, x = 3 * k + 1
def is_element_of_set_B (x : ℤ) : Prop := ∃ k : ℤ, x = 3 * k + 2
def is_element_of_complement_U (x : ℤ) : Prop := ∃ k : ℤ, x = 3 * k

theorem complement_union_A_B :
  {x : ℤ | ¬ (is_element_of_set_A x ∨ is_element_of_set_B x)} = {x : ℤ | is_element_of_complement_U x} :=
by
  sorry

end complement_union_A_B_l186_186925


namespace sin_690_eq_neg_half_l186_186355

theorem sin_690_eq_neg_half :
  Real.sin (690 * Real.pi / 180) = -1 / 2 :=
by {
  sorry
}

end sin_690_eq_neg_half_l186_186355


namespace cube_faces_sum_eq_neg_3_l186_186835

theorem cube_faces_sum_eq_neg_3 
    (a b c d e f : ℤ)
    (h1 : a = -3)
    (h2 : b = a + 1)
    (h3 : c = b + 1)
    (h4 : d = c + 1)
    (h5 : e = d + 1)
    (h6 : f = e + 1)
    (h7 : a + f = b + e)
    (h8 : b + e = c + d) :
  a + b + c + d + e + f = -3 := sorry

end cube_faces_sum_eq_neg_3_l186_186835


namespace monotonic_when_a_is_neg1_find_extreme_points_l186_186390

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (1/3) * x ^ 3 - (1/2) * (a^2 + a + 2) * x ^ 2 + a^2 * (a + 2) * x

theorem monotonic_when_a_is_neg1 :
  ∀ x : ℝ, f x (-1) ≤ f x (-1) :=
sorry

theorem find_extreme_points (a : ℝ) :
  if h : a = -1 ∨ a = 2 then
    True  -- The function is monotonically increasing, no extreme points
  else if h : a < -1 ∨ a > 2 then
    ∃ x_max x_min : ℝ, x_max = a + 2 ∧ x_min = a^2 ∧ (f x_max a ≥ f x a ∧ f x_min a ≤ f x a) 
  else
    ∃ x_max x_min : ℝ, x_max = a^2 ∧ x_min = a + 2 ∧ (f x_max a ≥ f x a ∧ f x_min a ≤ f x a) :=
sorry

end monotonic_when_a_is_neg1_find_extreme_points_l186_186390


namespace cylinder_to_sphere_volume_ratio_l186_186738

theorem cylinder_to_sphere_volume_ratio:
  ∀ (a r : ℝ), (a^2 = π * r^2) → (a^3)/( (4/3) * π * r^3) = 3/2 :=
by
  intros a r h
  sorry

end cylinder_to_sphere_volume_ratio_l186_186738


namespace angle_B_measure_l186_186094

open Real EuclideanGeometry Classical

noncomputable def measure_angle_B (A C : ℝ) : ℝ := 180 - (180 - A - C)

theorem angle_B_measure
  (l m : ℝ → ℝ → Prop) -- parallel lines l and m (can be interpreted as propositions for simplicity)
  (h_parallel : ∀ x y, l x y → m x y → x = y) -- Lines l and m are parallel
  (A C : ℝ)
  (hA : A = 120)
  (hC : C = 70) :
  measure_angle_B A C = 130 := 
by
  sorry

end angle_B_measure_l186_186094


namespace students_with_average_age_of_16_l186_186451

theorem students_with_average_age_of_16
  (N : ℕ) (A : ℕ) (N14 : ℕ) (A15 : ℕ) (N16 : ℕ)
  (h1 : N = 15) (h2 : A = 15) (h3 : N14 = 5) (h4 : A15 = 11) :
  N16 = 9 :=
sorry

end students_with_average_age_of_16_l186_186451


namespace right_triangle_even_or_odd_l186_186399

theorem right_triangle_even_or_odd (a b c : ℕ) (ha : Even a ∨ Odd a) (hb : Even b ∨ Odd b) (h : a^2 + b^2 = c^2) : 
  Even c ∨ (Even a ∧ Odd b) ∨ (Odd a ∧ Even b) :=
by
  sorry

end right_triangle_even_or_odd_l186_186399


namespace total_cost_proof_l186_186429

-- Define the prices of items
def price_coffee : ℕ := 4
def price_cake : ℕ := 7
def price_ice_cream : ℕ := 3

-- Define the number of items ordered by Mell and her friends
def mell_coffee : ℕ := 2
def mell_cake : ℕ := 1
def friend_coffee : ℕ := 2
def friend_cake : ℕ := 1
def friend_ice_cream : ℕ := 1
def number_of_friends : ℕ := 2

-- Calculate total cost for Mell
def total_mell : ℕ := (mell_coffee * price_coffee) + (mell_cake * price_cake)

-- Calculate total cost per friend
def total_friend : ℕ := (friend_coffee * price_coffee) + (friend_cake * price_cake) + (friend_ice_cream * price_ice_cream)

-- Calculate total cost for all friends
def total_friends : ℕ := number_of_friends * total_friend

-- Calculate total cost for Mell and her friends
def total_cost : ℕ := total_mell + total_friends

-- The theorem to prove
theorem total_cost_proof : total_cost = 51 := by
  sorry

end total_cost_proof_l186_186429


namespace num_envelopes_requiring_charge_l186_186117

structure Envelope where
  length : ℕ
  height : ℕ

def requiresExtraCharge (env : Envelope) : Bool :=
  let ratio := env.length / env.height
  ratio < 3/2 ∨ ratio > 3

def envelopes : List Envelope :=
  [{ length := 7, height := 5 },  -- E
   { length := 10, height := 2 }, -- F
   { length := 8, height := 8 },  -- G
   { length := 12, height := 3 }] -- H

def countExtraChargedEnvelopes : ℕ :=
  envelopes.filter requiresExtraCharge |>.length

theorem num_envelopes_requiring_charge : countExtraChargedEnvelopes = 4 := by
  sorry

end num_envelopes_requiring_charge_l186_186117


namespace skittles_total_l186_186714

-- Define the conditions
def skittles_per_friend : ℝ := 40.0
def number_of_friends : ℝ := 5.0

-- Define the target statement using the conditions
theorem skittles_total : (skittles_per_friend * number_of_friends = 200.0) :=
by 
  -- Using sorry to placeholder the proof
  sorry

end skittles_total_l186_186714


namespace probability_even_sum_of_three_players_l186_186593

theorem probability_even_sum_of_three_players :
  let total := (Nat.choose 12 4) * (Nat.choose 8 4) * (Nat.choose 4 4),
      case1 := (Nat.choose 6 2)^3 * (Nat.choose 4 2)^2 * (Nat.choose 2 2),
      case2 := 2 * (Nat.choose 6 4)^2
  in let even_sum_prob := case1 + case2,
         p := 19,
         q := 77
  in Nat.gcd p q = 1 → (even_sum_prob / total) = (p / q) ∧ (p + q) = 96 :=
by {
  sorry
}

end probability_even_sum_of_three_players_l186_186593


namespace product_of_five_consecutive_integers_divisible_by_240_l186_186799

theorem product_of_five_consecutive_integers_divisible_by_240 (n : ℤ) : ∃ k : ℤ, (∏ i in range 5, n + i) = 240 * k := 
sorry

end product_of_five_consecutive_integers_divisible_by_240_l186_186799


namespace average_cd_l186_186961

theorem average_cd (c d: ℝ) (h: (4 + 6 + 8 + c + d) / 5 = 18) : (c + d) / 2 = 36 :=
by sorry

end average_cd_l186_186961


namespace number_of_boxes_l186_186745

variable (boxes : ℕ) -- number of boxes
variable (mangoes_per_box : ℕ) -- mangoes per box
variable (total_mangoes : ℕ) -- total mangoes

def dozen : ℕ := 12

-- Condition: each box contains 10 dozen mangoes
def condition1 : mangoes_per_box = 10 * dozen := by 
  sorry

-- Condition: total mangoes in all boxes together is 4320
def condition2 : total_mangoes = 4320 := by
  sorry

-- Proof problem: prove that the number of boxes is 36
theorem number_of_boxes (h1 : mangoes_per_box = 10 * dozen) 
                        (h2 : total_mangoes = 4320) :
  boxes = 4320 / (10 * dozen) :=
  by
  sorry

end number_of_boxes_l186_186745


namespace find_ratio_of_arithmetic_sequences_l186_186866

variable {a_n b_n : ℕ → ℕ}
variable {A_n B_n : ℕ → ℝ}

def arithmetic_sums (a_n b_n : ℕ → ℕ) (A_n B_n : ℕ → ℝ) : Prop :=
  ∀ n, A_n n = (n * (2 * a_n 1 + (n - 1) * (a_n 8 - a_n 7))) / 2 ∧
         B_n n = (n * (2 * b_n 1 + (n - 1) * (b_n 8 - b_n 7))) / 2

theorem find_ratio_of_arithmetic_sequences 
    (h : ∀ n, A_n n / B_n n = (5 * n - 3) / (n + 9)) :
    ∃ r : ℝ, r = 3 := by
  sorry

end find_ratio_of_arithmetic_sequences_l186_186866


namespace amy_race_time_l186_186478

theorem amy_race_time (patrick_time : ℕ) (manu_time : ℕ) (amy_time : ℕ)
  (h1 : patrick_time = 60)
  (h2 : manu_time = patrick_time + 12)
  (h3 : amy_time = manu_time / 2) : 
  amy_time = 36 := 
sorry

end amy_race_time_l186_186478


namespace quadratic_decreasing_l186_186674

theorem quadratic_decreasing (a : ℝ) (h : ∀ x1 x2 : ℝ, x1 ≤ x2 → x2 ≤ 4 → (x1^2 + 4*a*x1 - 2) ≥ (x2^2 + 4*a*x2 - 2)) : a ≤ -2 := 
by
  sorry

end quadratic_decreasing_l186_186674


namespace square_area_l186_186512

theorem square_area (perimeter : ℝ) (h_perimeter : perimeter = 40) : 
  ∃ (area : ℝ), area = 100 := by
  sorry

end square_area_l186_186512


namespace pure_imaginary_real_part_zero_l186_186072

-- Define the condition that the complex number a + i is a pure imaginary number.
def isPureImaginary (z : ℂ) : Prop :=
  ∃ b : ℝ, z = Complex.I * b

-- Define the complex number a + i.
def z (a : ℝ) : ℂ := a + Complex.I

-- The theorem states that if z is pure imaginary, then a = 0.
theorem pure_imaginary_real_part_zero (a : ℝ) (h : isPureImaginary (z a)) : a = 0 :=
by
  sorry

end pure_imaginary_real_part_zero_l186_186072


namespace find_m_l186_186209

-- Definitions for the given vectors
def a : ℝ × ℝ := (3, 4)
def b (m : ℝ) : ℝ × ℝ := (-1, 2 * m)
def c (m : ℝ) : ℝ × ℝ := (m, -4)

-- Definition of vector addition
def vector_add (u v : ℝ × ℝ) : ℝ × ℝ :=
  (u.1 + v.1, u.2 + v.2)

-- Definition of dot product
def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

-- Condition that c is perpendicular to a + b
def perpendicular_condition (m : ℝ) : Prop :=
  dot_product (c m) (vector_add a (b m)) = 0

-- Proof statement
theorem find_m : ∃ m : ℝ, perpendicular_condition m ∧ m = -8 / 3 :=
sorry

end find_m_l186_186209


namespace linear_system_solution_l186_186660

theorem linear_system_solution :
  ∃ (x y z : ℝ), (x ≠ 0) ∧ (y ≠ 0) ∧ (z ≠ 0) ∧
  (x + (85/3) * y + 4 * z = 0) ∧ 
  (4 * x + (85/3) * y + z = 0) ∧ 
  (3 * x + 5 * y - 2 * z = 0) ∧ 
  (x * z) / (y ^ 2) = 25 := 
sorry

end linear_system_solution_l186_186660


namespace correct_propositions_l186_186870

namespace ProofProblem

-- Define Curve C
def curve_C (x y t : ℝ) : Prop :=
  (x^2 / (4 - t)) + (y^2 / (t - 1)) = 1

-- Proposition ①
def proposition_1 (t : ℝ) : Prop :=
  ¬(1 < t ∧ t < 4 ∧ t ≠ 5 / 2)

-- Proposition ②
def proposition_2 (t : ℝ) : Prop :=
  t > 4 ∨ t < 1

-- Proposition ③
def proposition_3 (t : ℝ) : Prop :=
  t ≠ 5 / 2

-- Proposition ④
def proposition_4 (t : ℝ) : Prop :=
  1 < t ∧ t < (5 / 2)

-- The theorem we need to prove
theorem correct_propositions (t : ℝ) :
  (proposition_1 t = false) ∧
  (proposition_2 t = true) ∧
  (proposition_3 t = false) ∧
  (proposition_4 t = true) :=
by
  sorry

end ProofProblem

end correct_propositions_l186_186870


namespace find_tricksters_within_30_questions_l186_186323

/-- 
Given 65 inhabitants in a village where:
- Two inhabitants are tricksters and the rest are knights.
- Knights always tell the truth.
- Tricksters can either tell the truth or lie.
- One can show any inhabitant a list of some group of inhabitants (which can consist of one person)
  and ask if all of them are knights.

Prove that it is possible to find both tricksters with no more than 30 questions.
-/
theorem find_tricksters_within_30_questions :
  ∃ (ask_knights : (inhabitants : fin 65) → list (fin 65) → Prop),
  (∀ (i j : fin 65), i ≠ j → ask_knights i [j] = true → (inhabitants[j] = knight) ∨ (inhabitants[j] = trickster))
  ∧ ∀ (inhabitants : fin 65),
  (∃ S : finset (fin 65), S.card = 2 ∧ 
  (∀ i, i ∈ S → inhabitants[i] = trickster) ∧ 
  by asking no more than 30 questions,
  you can identify both tricksters.

end find_tricksters_within_30_questions_l186_186323


namespace sum_of_three_consecutive_odd_integers_l186_186614

theorem sum_of_three_consecutive_odd_integers (a : ℤ) (h₁ : a + (a + 4) = 150) :
  (a + (a + 2) + (a + 4) = 225) :=
by
  sorry

end sum_of_three_consecutive_odd_integers_l186_186614


namespace election_votes_l186_186078

theorem election_votes (V : ℕ) (h1 : ∃ Vb, Vb = 2509 ∧ (0.8 * V : ℝ) = (Vb + 0.15 * (V : ℝ)) + Vb) : V = 7720 :=
sorry

end election_votes_l186_186078


namespace smallest_k_l186_186722

theorem smallest_k (a b c : ℤ) (k : ℤ) (h1 : a < b) (h2 : b < c) 
  (h3 : 2 * b = a + c) (h4 : (k * c) ^ 2 = a * b) (h5 : k > 1) : 
  c > 0 → k = 2 := 
sorry

end smallest_k_l186_186722


namespace find_f1_and_f1_l186_186675

theorem find_f1_and_f1' (f : ℝ → ℝ) (f' : ℝ → ℝ) (h_deriv : ∀ x, deriv f x = f' x)
  (h_eq : ∀ x, f x = 2 * x * f' 1 + Real.log x) : f 1 + f' 1 = -3 :=
by sorry

end find_f1_and_f1_l186_186675


namespace polynomial_bound_swap_l186_186668

variable (a b c : ℝ)

theorem polynomial_bound_swap (h : ∀ x : ℝ, |x| ≤ 1 → |a * x^2 + b * x + c| ≤ 1) :
  ∀ (x : ℝ), |x| ≤ 1 → |c * x^2 + b * x + a| ≤ 2 := by
  sorry

end polynomial_bound_swap_l186_186668


namespace least_x_y_z_value_l186_186238

theorem least_x_y_z_value :
  ∃ (x y z : ℕ), (x > 0) ∧ (y > 0) ∧ (z > 0) ∧ (3 * x = 4 * y) ∧ (4 * y = 7 * z) ∧ (3 * x = 7 * z) ∧ (x - y + z = 19) :=
by
  sorry

end least_x_y_z_value_l186_186238


namespace sin_690_degree_l186_186349

theorem sin_690_degree : sin (690 : ℝ) * (Real.pi / 180) = -(1 / 2) := by
  sorry

end sin_690_degree_l186_186349


namespace product_of_five_consecutive_divisible_by_30_l186_186779

theorem product_of_five_consecutive_divisible_by_30 :
  ∀ n : ℤ, 30 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  sorry

end product_of_five_consecutive_divisible_by_30_l186_186779


namespace correct_option_l186_186292

-- Definitions based on the problem's conditions
def option_A (x : ℝ) : Prop := x^2 * x^4 = x^8
def option_B (x : ℝ) : Prop := (x^2)^3 = x^5
def option_C (x : ℝ) : Prop := x^2 + x^2 = 2 * x^2
def option_D (x : ℝ) : Prop := (3 * x)^2 = 3 * x^2

-- Theorem stating that out of the given options, option C is correct
theorem correct_option (x : ℝ) : option_C x :=
by {
  sorry
}

end correct_option_l186_186292


namespace largest_divisor_of_consecutive_five_l186_186789

theorem largest_divisor_of_consecutive_five (n : ℤ) : 
  (∃ d, ∀ i ∈ {n, n + 1, n + 2, n + 3, n + 4}, d ∣ (∏ x in {n, n + 1, n + 2, n + 3, n + 4}, x)) → 
  d = 60 
:= sorry

end largest_divisor_of_consecutive_five_l186_186789


namespace largest_divisor_of_consecutive_five_l186_186788

theorem largest_divisor_of_consecutive_five (n : ℤ) : 
  (∃ d, ∀ i ∈ {n, n + 1, n + 2, n + 3, n + 4}, d ∣ (∏ x in {n, n + 1, n + 2, n + 3, n + 4}, x)) → 
  d = 60 
:= sorry

end largest_divisor_of_consecutive_five_l186_186788


namespace find_m_plus_n_l186_186301

noncomputable def overlapping_points (A B: ℝ × ℝ) (C D: ℝ × ℝ) : Prop :=
  let k_AB := (B.2 - A.2) / (B.1 - A.1)
  let M_AB := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  let axis_slope := - 1 / k_AB
  let k_CD := (D.2 - C.2) / (D.1 - C.1)
  let M_CD := ((C.1 + D.1) / 2, (C.2 + D.2) / 2)
  k_CD = axis_slope ∧ (M_CD.2 - M_AB.2) = axis_slope * (M_CD.1 - M_AB.1)

theorem find_m_plus_n : 
  ∃ (m n: ℝ), overlapping_points (0, 2) (4, 0) (7, 3) (m, n) ∧ m + n = 34 / 5 :=
sorry

end find_m_plus_n_l186_186301


namespace sum_of_consecutive_odd_integers_l186_186600

-- Define the conditions
def consecutive_odd_integers (n : ℤ) : List ℤ :=
  [n, n + 2, n + 4]

def sum_first_and_third_eq_150 (n : ℤ) : Prop :=
  n + (n + 4) = 150

-- Proof to show that the sum of these integers is 225
theorem sum_of_consecutive_odd_integers (n : ℤ) (h : sum_first_and_third_eq_150 n) :
  consecutive_odd_integers n).sum = 225 :=
  sorry

end sum_of_consecutive_odd_integers_l186_186600


namespace solve_for_y_l186_186578

-- Given condition
def equation (y : ℚ) := (8 * y^2 + 90 * y + 5) / (3 * y^2 + 4 * y + 49) = 4 * y + 1

-- Prove the resulting polynomial equation
theorem solve_for_y (y : ℚ) (h : equation y) : 12 * y^3 + 11 * y^2 + 110 * y + 44 = 0 :=
sorry

end solve_for_y_l186_186578


namespace jogger_speed_is_9_l186_186489

noncomputable def train_speed := 45 -- in kmph
noncomputable def gap := 240 -- in meters
noncomputable def train_length := 120 -- in meters
noncomputable def time_to_pass := 36 -- in seconds

noncomputable def jogger_speed : ℝ :=
  let relative_speed_mps := (gap + train_length) / time_to_pass in
  (relative_speed_mps * 18) / 5

theorem jogger_speed_is_9 : jogger_speed = 9 := by
  sorry

end jogger_speed_is_9_l186_186489


namespace ratio_larger_to_smaller_l186_186955

noncomputable def ratio_of_numbers (a b : ℝ) : ℝ :=
  a / b

theorem ratio_larger_to_smaller (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a > b) 
  (h4 : a - b = 7 * ((a + b) / 2)) : ratio_of_numbers a b = 9 / 5 := 
  sorry

end ratio_larger_to_smaller_l186_186955


namespace determine_p_l186_186044

-- Define the quadratic equation
def quadratic_eq (p x : ℝ) : ℝ := 3 * x^2 - 5 * (p - 1) * x + (p^2 + 2)

-- Define the conditions for the roots x1 and x2
def conditions (p x1 x2 : ℝ) : Prop :=
  quadratic_eq p x1 = 0 ∧
  quadratic_eq p x2 = 0 ∧
  x1 + 4 * x2 = 14

-- Define the theorem to prove the correct values of p
theorem determine_p (p : ℝ) (x1 x2 : ℝ) :
  conditions p x1 x2 → p = 742 / 127 ∨ p = 4 :=
by
  sorry

end determine_p_l186_186044


namespace largest_integer_dividing_consecutive_product_l186_186814

theorem largest_integer_dividing_consecutive_product :
  ∀ (n : ℤ), (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) % 120 = 0 :=
by
  sorry

end largest_integer_dividing_consecutive_product_l186_186814


namespace taozi_is_faster_than_xiaoxiao_l186_186731

theorem taozi_is_faster_than_xiaoxiao : 
  let taozi_speed := 210
  let xiaoxiao_distance := 500
  let xiaoxiao_time := 3
  let xiaoxiao_speed := xiaoxiao_distance / xiaoxiao_time
  taozi_speed > xiaoxiao_speed
:= by
  let taozi_speed := 210
  let xiaoxiao_distance := 500
  let xiaoxiao_time := 3
  let xiaoxiao_speed := xiaoxiao_distance / xiaoxiao_time
  sorry

end taozi_is_faster_than_xiaoxiao_l186_186731


namespace storage_methods_l186_186000

-- Definitions for the vertices and edges of the pyramid
structure Pyramid :=
  (P A B C D : Type)
  
-- Edges of the pyramid represented by pairs of vertices
def edges (P A B C D : Type) := [(P, A), (P, B), (P, C), (P, D), (A, B), (A, C), (A, D), (B, C), (B, D), (C, D)]

-- Safe storage condition: No edges sharing a common vertex in the same warehouse
def safe (edge1 edge2 : (Type × Type)) : Prop :=
  edge1.1 ≠ edge2.1 ∧ edge1.1 ≠ edge2.2 ∧ edge1.2 ≠ edge2.1 ∧ edge1.2 ≠ edge2.2

-- The number of different methods to store the chemical products safely
def number_of_safe_storage_methods : Nat :=
  -- We should replace this part by actual calculation or combinatorial methods relevant to the problem
  48

theorem storage_methods (P A B C D : Type) : number_of_safe_storage_methods = 48 :=
  sorry

end storage_methods_l186_186000


namespace midpoint_of_intersection_l186_186876

noncomputable def parametric_line (t : ℝ) : ℝ × ℝ :=
  (1 + 2 * t, 2 * t)

noncomputable def polar_curve (θ : ℝ) : ℝ :=
  2 / Real.sqrt (1 + 3 * Real.sin θ ^ 2)

theorem midpoint_of_intersection :
  ∃ A B : ℝ × ℝ,
    (∃ t₁ t₂ : ℝ, 
      A = parametric_line t₁ ∧ 
      B = parametric_line t₂ ∧ 
      (A.1 ^ 2 / 4 + A.2 ^ 2 = 1) ∧ 
      (B.1 ^ 2 / 4 + B.2 ^ 2 = 1)) ∧
    ((A.1 + B.1) / 2, (A.2 + B.2) / 2) = (4 / 5, -1 / 5) :=
sorry

end midpoint_of_intersection_l186_186876


namespace triangle_angle_bisector_YE_l186_186243

noncomputable def triangle_segs_YE : ℝ := (36 : ℝ) / 7

theorem triangle_angle_bisector_YE
  (XYZ: Type)
  (XY XZ YZ YE EZ: ℝ)
  (YZ_length : YZ = 12)
  (side_ratios : XY / XZ = 3 / 4 ∧ XY / YZ  = 3 / 5 ∧ XZ / YZ = 4 / 5)
  (angle_bisector : YE / EZ = XY / XZ)
  (seg_sum : YE + EZ = YZ) :
  YE = (36 : ℝ) / 7 :=
by sorry

end triangle_angle_bisector_YE_l186_186243


namespace area_of_park_l186_186587

noncomputable def length (x : ℕ) : ℕ := 3 * x
noncomputable def width (x : ℕ) : ℕ := 2 * x
noncomputable def area (x : ℕ) : ℕ := length x * width x
noncomputable def cost_per_meter : ℕ := 80
noncomputable def total_cost : ℕ := 200
noncomputable def perimeter (x : ℕ) : ℕ := 2 * (length x + width x)

theorem area_of_park : ∃ x : ℕ, area x = 3750 ∧ total_cost = (perimeter x) * cost_per_meter / 100 := by
  sorry

end area_of_park_l186_186587


namespace dodecagon_area_constraint_l186_186006

theorem dodecagon_area_constraint 
    (a : ℕ) -- side length of the square
    (N : ℕ) -- a large number with 2017 digits, breaking it down as 2 * (10^2017 - 1) / 9
    (hN : N = (2 * (10^2017 - 1)) / 9) 
    (H : ∃ n : ℕ, (n * n) = 3 * a^2 / 2) :
    False :=
by
    sorry

end dodecagon_area_constraint_l186_186006


namespace largest_integer_dividing_consecutive_product_l186_186811

theorem largest_integer_dividing_consecutive_product :
  ∀ (n : ℤ), (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) % 120 = 0 :=
by
  sorry

end largest_integer_dividing_consecutive_product_l186_186811


namespace ratio_of_milk_water_in_larger_vessel_l186_186984

-- Definitions of conditions
def volume1 (V : ℝ) : ℝ := 3 * V
def volume2 (V : ℝ) : ℝ := 5 * V

def ratio_milk_water_1 : ℝ × ℝ := (1, 2)
def ratio_milk_water_2 : ℝ × ℝ := (3, 2)

-- Define the problem statement
theorem ratio_of_milk_water_in_larger_vessel (V : ℝ) (hV : V > 0) :
  (volume1 V / (ratio_milk_water_1.1 + ratio_milk_water_1.2)) = V ∧ 
  2 * (volume1 V / (ratio_milk_water_1.1 + ratio_milk_water_1.2)) = 2 * V ∧ 
  3 * (volume2 V / (ratio_milk_water_2.1 + ratio_milk_water_2.2)) = 3 * V ∧ 
  2 * (volume2 V / (ratio_milk_water_2.1 + ratio_milk_water_2.2)) = 2 * V →
  (4 * V) / (4 * V) = 1 :=
sorry

end ratio_of_milk_water_in_larger_vessel_l186_186984


namespace leo_amount_after_settling_debts_l186_186904

theorem leo_amount_after_settling_debts (total_amount : ℝ) (ryan_share : ℝ) (ryan_owes_leo : ℝ) (leo_owes_ryan : ℝ) 
  (h1 : total_amount = 48) 
  (h2 : ryan_share = (2 / 3) * total_amount) 
  (h3 : ryan_owes_leo = 10) 
  (h4 : leo_owes_ryan = 7) : 
  (total_amount - ryan_share) + (ryan_owes_leo - leo_owes_ryan) = 19 :=
by
  sorry

end leo_amount_after_settling_debts_l186_186904


namespace range_of_x_l186_186239

noncomputable def a (x : ℝ) : ℝ := x
def b : ℝ := 2
def B : ℝ := 60

-- State the problem: Prove the range of x given the conditions
theorem range_of_x (x : ℝ) (A : ℝ) (C : ℝ) (h1 : a x = b / (Real.sin (B * Real.pi / 180)) * (Real.sin (A * Real.pi / 180)))
  (h2 : A + C = 180 - 60) (two_solutions : (60 < A ∧ A < 120)) :
  2 < x ∧ x < 4 * Real.sqrt 3 / 3 :=
sorry

end range_of_x_l186_186239


namespace last_three_digits_of_3_pow_5000_l186_186145

theorem last_three_digits_of_3_pow_5000 : (3 ^ 5000) % 1000 = 1 := 
by
  -- skip the proof
  sorry

end last_three_digits_of_3_pow_5000_l186_186145


namespace sin_690_eq_negative_one_half_l186_186356

theorem sin_690_eq_negative_one_half : Real.sin (690 * Real.pi / 180) = -1 / 2 := by
  sorry

end sin_690_eq_negative_one_half_l186_186356


namespace largest_divisor_of_product_of_five_consecutive_integers_l186_186762

theorem largest_divisor_of_product_of_five_consecutive_integers :
  ∃ n, (∀ k : ℤ, n ∣ (k * (k + 1) * (k + 2) * (k + 3) * (k + 4))) ∧ n = 60 :=
by
  sorry

end largest_divisor_of_product_of_five_consecutive_integers_l186_186762


namespace quadratic_equation_unique_solution_l186_186009

theorem quadratic_equation_unique_solution 
  (a c : ℝ) (h1 : ∃ x : ℝ, a * x^2 + 8 * x + c = 0)
  (h2 : a + c = 10)
  (h3 : a < c) :
  (a, c) = (2, 8) := 
sorry

end quadratic_equation_unique_solution_l186_186009


namespace janet_total_earnings_l186_186550

def hourly_wage_exterminator := 70
def hourly_work_exterminator := 20
def sculpture_price_per_pound := 20
def sculpture_1_weight := 5
def sculpture_2_weight := 7

theorem janet_total_earnings :
  (hourly_wage_exterminator * hourly_work_exterminator) +
  (sculpture_price_per_pound * sculpture_1_weight) +
  (sculpture_price_per_pound * sculpture_2_weight) = 1640 := by
  sorry

end janet_total_earnings_l186_186550


namespace sufficient_but_not_necessary_l186_186936

theorem sufficient_but_not_necessary (a : ℝ) (h : a > 1) : (1 / a < 1) := 
by
  sorry

end sufficient_but_not_necessary_l186_186936


namespace henry_walks_distance_l186_186683

noncomputable def gym_distance : ℝ := 3

noncomputable def walk_factor : ℝ := 2 / 3

noncomputable def c_limit_position : ℝ := 1.5

noncomputable def d_limit_position : ℝ := 2.5

theorem henry_walks_distance :
  abs (c_limit_position - d_limit_position) = 1 := by
  sorry

end henry_walks_distance_l186_186683


namespace roll_seven_dice_at_least_one_pair_no_three_l186_186833

noncomputable def roll_seven_dice_probability : ℚ :=
  let total_outcomes := (6^7 : ℚ)
  let one_pair_case := (6 * 21 * 120 : ℚ)
  let two_pairs_case := (15 * 21 * 10 * 24 : ℚ)
  let successful_outcomes := one_pair_case + two_pairs_case
  successful_outcomes / total_outcomes

theorem roll_seven_dice_at_least_one_pair_no_three :
  roll_seven_dice_probability = 315 / 972 :=
by
  unfold roll_seven_dice_probability
  -- detailed steps to show the proof would go here
  sorry

end roll_seven_dice_at_least_one_pair_no_three_l186_186833


namespace fraction_expression_l186_186016

theorem fraction_expression :
  (1 / 4 - 1 / 6) / (1 / 3 + 1 / 2) = 1 / 10 :=
by
  sorry

end fraction_expression_l186_186016


namespace complement_of_union_l186_186917

def A : Set ℤ := {x | ∃ k : ℤ, x = 3 * k + 1}
def B : Set ℤ := {x | ∃ k : ℤ, x = 3 * k + 2}
def U : Set ℤ := Set.univ

theorem complement_of_union (x : ℤ) : (x ∉ A ∪ B) ↔ ∃ k : ℤ, x = 3 * k := 
by {
  sorry
}

end complement_of_union_l186_186917


namespace increasing_function_l186_186671

noncomputable def f (a x : ℝ) : ℝ := a * x + Real.sin x

theorem increasing_function (a : ℝ) :
  (∀ x y, x ≤ y → f a x ≤ f a y) ↔ (a ∈ Set.Ici (1 : ℝ)) := by
  sorry

end increasing_function_l186_186671


namespace find_b_of_perpendicular_bisector_l186_186956

theorem find_b_of_perpendicular_bisector :
  (∃ b : ℝ, (∀ x y : ℝ, x + y = b → x + y = 4 + 6)) → b = 10 :=
by
  sorry

end find_b_of_perpendicular_bisector_l186_186956


namespace initial_temperature_l186_186185

theorem initial_temperature (T_initial : ℝ) 
  (heating_rate : ℝ) (cooling_rate : ℝ) (total_time : ℝ) 
  (T_heat : ℝ) (T_cool : ℝ) (T_target : ℝ) (T_final : ℝ) 
  (h1 : heating_rate = 5) (h2 : cooling_rate = 7)
  (h3 : T_target = 240) (h4 : T_final = 170) 
  (h5 : total_time = 46)
  (h6 : T_cool = (T_target - T_final) / cooling_rate)
  (h7: total_time = T_heat + T_cool)
  (h8 : T_heat = (T_target - T_initial) / heating_rate) :
  T_initial = 60 :=
by
  -- Proof yet to be filled in
  sorry

end initial_temperature_l186_186185


namespace positive_integer_expression_l186_186045

-- Define the existence conditions for a given positive integer n
theorem positive_integer_expression (n : ℕ) (h : 0 < n) : ∃ a b c : ℤ, (n = a^2 + b^2 + c^2 + c) := 
sorry

end positive_integer_expression_l186_186045


namespace ava_planted_more_trees_l186_186337

theorem ava_planted_more_trees (L : ℕ) (h1 : 9 + L = 15) : 9 - L = 3 := 
by
  sorry

end ava_planted_more_trees_l186_186337


namespace vectors_parallel_same_direction_l186_186057

variables {a b : EuclideanSpace ℝ (Fin 2)}

-- Define non-zero vectors
axiom a_nonzero : a ≠ 0
axiom b_nonzero : b ≠ 0

-- Given condition
axiom norm_add_eq_norm_add : ∥a + b∥ = ∥a∥ + ∥b∥

theorem vectors_parallel_same_direction :
  (∃ (c : ℝ), 0 < c ∧ b = c • a) :=
sorry

end vectors_parallel_same_direction_l186_186057


namespace determine_ab_l186_186937

noncomputable def f (a b : ℕ) (x : ℝ) : ℝ := x^2 + 2 * a * x + b * 2^x

theorem determine_ab (a b : ℕ) (h : ∀ x : ℝ, f a b x = 0 ↔ f a b (f a b x) = 0) :
  (a, b) = (0, 0) ∨ (a, b) = (1, 0) :=
by
  sorry

end determine_ab_l186_186937


namespace calculation_l186_186497

theorem calculation :
  5 * 399 + 4 * 399 + 3 * 399 + 397 = 5185 :=
  by
    sorry

end calculation_l186_186497


namespace age_of_B_l186_186983

variable (A B C : ℕ)

theorem age_of_B (h1 : A + B + C = 84) (h2 : A + C = 58) : B = 26 := by
  sorry

end age_of_B_l186_186983


namespace area_of_shaded_region_l186_186287

def radius_of_first_circle : ℝ := 4
def radius_of_second_circle : ℝ := 5
def radius_of_third_circle : ℝ := 2
def radius_of_fourth_circle : ℝ := 9

theorem area_of_shaded_region :
  π * (radius_of_fourth_circle ^ 2) - π * (radius_of_first_circle ^ 2) - π * (radius_of_second_circle ^ 2) - π * (radius_of_third_circle ^ 2) = 36 * π :=
by {
  sorry
}

end area_of_shaded_region_l186_186287


namespace units_digit_base8_l186_186539

theorem units_digit_base8 (a b : ℕ) (h_a : a = 505) (h_b : b = 71) : 
  ((a * b) % 8) = 7 := 
by
  sorry

end units_digit_base8_l186_186539


namespace complement_union_A_B_l186_186921

def A := {x : ℤ | ∃ k : ℤ, x = 3 * k + 1}
def B := {x : ℤ | ∃ k : ℤ, x = 3 * k + 2}
def U := {x : ℤ | True}

theorem complement_union_A_B:
  (U \ (A ∪ B) = {x : ℤ | ∃ k : ℤ, x = 3 * k}) :=
by
  sorry

end complement_union_A_B_l186_186921


namespace symmetrical_character_is_C_l186_186704

-- Definitions of the characters and the concept of symmetry
def is_symmetrical (char: Char): Prop := 
  match char with
  | '中' => True
  | _ => False

-- The options given in the problem
def optionA := '爱'
def optionB := '我'
def optionC := '中'
def optionD := '国'

-- The problem statement: Prove that among the given options, the symmetrical character is 中.
theorem symmetrical_character_is_C : (is_symmetrical optionA = False) ∧ (is_symmetrical optionB = False) ∧ (is_symmetrical optionC = True) ∧ (is_symmetrical optionD = False) :=
by
  sorry

end symmetrical_character_is_C_l186_186704


namespace sum_of_consecutive_odd_integers_l186_186599

-- Define the conditions
def consecutive_odd_integers (n : ℤ) : List ℤ :=
  [n, n + 2, n + 4]

def sum_first_and_third_eq_150 (n : ℤ) : Prop :=
  n + (n + 4) = 150

-- Proof to show that the sum of these integers is 225
theorem sum_of_consecutive_odd_integers (n : ℤ) (h : sum_first_and_third_eq_150 n) :
  consecutive_odd_integers n).sum = 225 :=
  sorry

end sum_of_consecutive_odd_integers_l186_186599


namespace graph_connected_probability_l186_186134

open Finset

noncomputable def probability_connected : ℝ :=
  1 - (20 * ((finset.card (finset.range 171).powerset_len 16).toReal /
             (finset.card (finset.range 190).powerset_len 35).toReal))

theorem graph_connected_probability :
  ∀ (V : ℕ), (V = 20) → 
  let E := V * (V - 1) / 2 in
  let remaining_edges := E - 35 in
  probability_connected = 1 - (20 * ((finset.card (finset.range 171).powerset_len 16).toReal /
                                     (finset.card (finset.range 190).powerset_len 35).toReal)) :=
begin
  intros,
  -- Definitions of the complete graph and remaining edges after removing 35 edges
  sorry
end

end graph_connected_probability_l186_186134


namespace sum_of_three_consecutive_odds_l186_186607

theorem sum_of_three_consecutive_odds (a : ℤ) (h : a % 2 = 1) (ha_mod : (a + 4) % 2 = 1) (h_sum : a + (a + 4) = 150) : a + (a + 2) + (a + 4) = 225 :=
sorry

end sum_of_three_consecutive_odds_l186_186607


namespace triangle_is_isosceles_l186_186541

theorem triangle_is_isosceles (A B C a b c : ℝ) (h1 : c = 2 * a * Real.cos B) : 
  A = B → a = b := 
sorry

end triangle_is_isosceles_l186_186541


namespace part_i_part_ii_l186_186001

variable {b c : ℤ}

theorem part_i (hb : b ≠ 0) (hc : c ≠ 0) 
    (cond1 : ∃ m n : ℤ, m ≠ n ∧ (m + n = -b) ∧ (m * n = c))
    (cond2 : ∃ r s : ℤ, r ≠ s ∧ (r + s = -b) ∧ (r * s = -c)) :
    ∃ (p q : ℤ), p > 0 ∧ q > 0 ∧ p ≠ q ∧ 2 * b ^ 2 = p ^ 2 + q ^ 2 :=
sorry

theorem part_ii (hb : b ≠ 0) (hc : c ≠ 0) 
    (cond1 : ∃ m n : ℤ, m ≠ n ∧ (m + n = -b) ∧ (m * n = c))
    (cond2 : ∃ r s : ℤ, r ≠ s ∧ (r + s = -b) ∧ (r * s = -c)) :
    ∃ (r s : ℤ), r > 0 ∧ s > 0 ∧ r ≠ s ∧ b ^ 2 = r ^ 2 + s ^ 2 :=
sorry

end part_i_part_ii_l186_186001


namespace eval_expr_l186_186188

theorem eval_expr : 2 + 3 * 4 - 5 / 5 + 7 = 20 := by
  sorry

end eval_expr_l186_186188


namespace a_plus_b_in_D_l186_186215

def setA : Set ℤ := {x | ∃ k : ℤ, x = 4 * k}
def setB : Set ℤ := {x | ∃ m : ℤ, x = 4 * m + 1}
def setC : Set ℤ := {x | ∃ n : ℤ, x = 4 * n + 2}
def setD : Set ℤ := {x | ∃ t : ℤ, x = 4 * t + 3}

theorem a_plus_b_in_D (a b : ℤ) (ha : a ∈ setB) (hb : b ∈ setC) : a + b ∈ setD := by
  sorry

end a_plus_b_in_D_l186_186215


namespace area_ratio_trapezoid_triangle_l186_186085

-- Define the geometric elements and given conditions.
variable (AB CD EAB ABCD : ℝ)
variable (trapezoid_ABCD : AB = 10)
variable (trapezoid_ABCD_CD : CD = 25)
variable (ratio_areas_EDC_EAB : (CD / AB)^2 = 25 / 4)
variable (trapezoid_relation : (ABCD + EAB) / EAB = 25 / 4)

-- The goal is to prove the ratio of the areas of triangle EAB to trapezoid ABCD.
theorem area_ratio_trapezoid_triangle :
  (EAB / ABCD) = 4 / 21 :=
by
  sorry

end area_ratio_trapezoid_triangle_l186_186085


namespace effective_annual_rate_of_interest_l186_186275

theorem effective_annual_rate_of_interest 
  (i : ℝ) (n : ℕ) (h_i : i = 0.10) (h_n : n = 2) : 
  (1 + i / n)^n - 1 = 0.1025 :=
by
  sorry

end effective_annual_rate_of_interest_l186_186275


namespace principal_amount_borrowed_l186_186020

theorem principal_amount_borrowed (SI R T : ℝ) (h_SI : SI = 2000) (h_R : R = 4) (h_T : T = 10) : 
    ∃ P, SI = (P * R * T) / 100 ∧ P = 5000 :=
by
    sorry

end principal_amount_borrowed_l186_186020


namespace fruit_ratio_l186_186555

variable (A P B : ℕ)
variable (n : ℕ)

theorem fruit_ratio (h1 : A = 4) (h2 : P = n * A) (h3 : A + P + B = 21) (h4 : B = 5) : P / A = 3 := by
  sorry

end fruit_ratio_l186_186555


namespace non_planar_characterization_l186_186747

-- Definitions:
structure Graph where
  V : ℕ
  E : ℕ
  F : ℕ

def is_planar (G : Graph) : Prop :=
  G.V - G.E + G.F = 2

def edge_inequality (G : Graph) : Prop :=
  G.E ≤ 3 * G.V - 6

def has_subgraph_K5_or_K33 (G : Graph) : Prop := sorry -- Placeholder for the complex subgraph check

-- Theorem statement:
theorem non_planar_characterization (G : Graph) (hV : G.V ≥ 3) :
  ¬ is_planar G ↔ ¬ edge_inequality G ∨ has_subgraph_K5_or_K33 G := sorry

end non_planar_characterization_l186_186747


namespace sum_of_consecutive_odd_integers_l186_186601

-- Define the conditions
def consecutive_odd_integers (n : ℤ) : List ℤ :=
  [n, n + 2, n + 4]

def sum_first_and_third_eq_150 (n : ℤ) : Prop :=
  n + (n + 4) = 150

-- Proof to show that the sum of these integers is 225
theorem sum_of_consecutive_odd_integers (n : ℤ) (h : sum_first_and_third_eq_150 n) :
  consecutive_odd_integers n).sum = 225 :=
  sorry

end sum_of_consecutive_odd_integers_l186_186601


namespace model_x_completion_time_l186_186300

theorem model_x_completion_time (T_x : ℝ) : 
  (24 : ℕ) * (1 / T_x + 1 / 36) = 1 → T_x = 72 := 
by 
  sorry

end model_x_completion_time_l186_186300


namespace largest_divisor_of_product_of_five_consecutive_integers_l186_186761

theorem largest_divisor_of_product_of_five_consecutive_integers :
  ∃ n, (∀ k : ℤ, n ∣ (k * (k + 1) * (k + 2) * (k + 3) * (k + 4))) ∧ n = 60 :=
by
  sorry

end largest_divisor_of_product_of_five_consecutive_integers_l186_186761


namespace sequence_is_aperiodic_l186_186838

noncomputable def sequence_a (a : ℕ → ℕ) : Prop :=
∀ k n : ℕ, k < 2^n → a k ≠ a (k + 2^n)

theorem sequence_is_aperiodic (a : ℕ → ℕ) (h_a : sequence_a a) : ¬(∃ p : ℕ, ∀ n k : ℕ, a k = a (k + n * p)) :=
sorry

end sequence_is_aperiodic_l186_186838


namespace intersection_points_product_l186_186290

theorem intersection_points_product (x y : ℝ) :
  (x^2 - 2 * x + y^2 - 6 * y + 9 = 0) ∧ (x^2 - 8 * x + y^2 - 6 * y + 28 = 0) → x * y = 6 :=
by
  sorry

end intersection_points_product_l186_186290


namespace kim_shirts_left_l186_186717

-- Define the total number of shirts initially
def initial_shirts : ℕ := 4 * 12

-- Define the number of shirts given to the sister as 1/3 of the total
def shirts_given_to_sister : ℕ := initial_shirts / 3

-- Define the number of shirts left after giving some to the sister
def shirts_left : ℕ := initial_shirts - shirts_given_to_sister

-- The theorem we need to prove: Kim has 32 shirts left
theorem kim_shirts_left : shirts_left = 32 := by
  -- Proof is omitted
  sorry

end kim_shirts_left_l186_186717


namespace units_digit_of_n_l186_186363

-- Definitions
def units_digit (x : ℕ) : ℕ := x % 10

-- Conditions
variables (m n : ℕ)
axiom condition1 : m * n = 23^5
axiom condition2 : units_digit m = 4

-- Theorem statement
theorem units_digit_of_n : units_digit n = 8 :=
sorry

end units_digit_of_n_l186_186363


namespace tourists_left_l186_186988

theorem tourists_left (initial_tourists eaten_by_anacondas poisoned_fraction recover_fraction : ℕ) 
(h_initial : initial_tourists = 30) 
(h_eaten : eaten_by_anacondas = 2)
(h_poisoned_fraction : poisoned_fraction = 2)
(h_recover_fraction : recover_fraction = 7) :
  initial_tourists - eaten_by_anacondas - (initial_tourists - eaten_by_anacondas) / poisoned_fraction + (initial_tourists - eaten_by_anacondas) / poisoned_fraction / recover_fraction = 16 :=
by
  sorry

end tourists_left_l186_186988


namespace remainder_when_1200th_number_divided_by_200_l186_186720

theorem remainder_when_1200th_number_divided_by_200 
  (S : ℕ → ℕ)
  (h1 : ∀ n : ℕ, S n = nth_element_with_7_ones_in_binary n) :
  S 1199 % 200 = 80 :=
sorry

end remainder_when_1200th_number_divided_by_200_l186_186720


namespace price_of_child_ticket_l186_186313

theorem price_of_child_ticket (total_seats : ℕ) (adult_ticket_price : ℕ) (total_revenue : ℕ)
  (child_tickets_sold : ℕ) (child_ticket_price : ℕ) :
  total_seats = 80 →
  adult_ticket_price = 12 →
  total_revenue = 519 →
  child_tickets_sold = 63 →
  (17 * adult_ticket_price) + (child_tickets_sold * child_ticket_price) = total_revenue →
  child_ticket_price = 5 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end price_of_child_ticket_l186_186313


namespace total_amount_paid_l186_186181

def jacket_price : ℝ := 150
def sale_discount : ℝ := 0.25
def coupon_discount : ℝ := 10
def sales_tax : ℝ := 0.10

theorem total_amount_paid : 
  (jacket_price * (1 - sale_discount) - coupon_discount) * (1 + sales_tax) = 112.75 := 
by
  sorry

end total_amount_paid_l186_186181


namespace sum_of_triangle_angles_l186_186590

theorem sum_of_triangle_angles 
  (smallest largest middle : ℝ) 
  (h1 : smallest = 20) 
  (h2 : middle = 3 * smallest) 
  (h3 : largest = 5 * smallest) 
  (h4 : smallest + middle + largest = 180) :
  smallest + middle + largest = 180 :=
by sorry

end sum_of_triangle_angles_l186_186590


namespace greatest_integer_not_exceeding_x2_div_50_l186_186437

noncomputable def trapezoid_condition (b h : ℝ) :=
  let longer_base := b + 50 in
  let midline := (b + longer_base) / 2 in
  midline = b + 25 ∧
  (2 * b + 25) = 3 * (b + 37.5) ∧
  b = 37.5

noncomputable def segment_condition (x : ℝ) (b : ℝ := 37.5) :=
  (2 * (b + x) = 125) ∧ (x^2 - 37.5 * x + 2812.5 = 0)

theorem greatest_integer_not_exceeding_x2_div_50 (x : ℝ) (h : ℝ) :
  trapezoid_condition 37.5 h → segment_condition x 37.5 → 
  (⌊x^2 / 50⌋ = 112) :=
by
  intros h1 h2
  sorry

end greatest_integer_not_exceeding_x2_div_50_l186_186437


namespace largest_divisor_of_product_of_5_consecutive_integers_l186_186786

/-- What is the largest integer that must divide the product of any 5 consecutive integers? -/
theorem largest_divisor_of_product_of_5_consecutive_integers :
  ∀ n : ℤ, ∃ d : ℤ, d = 24 ∧ d ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  sorry

end largest_divisor_of_product_of_5_consecutive_integers_l186_186786


namespace plants_same_height_after_54_years_l186_186726

noncomputable def h1 (t : ℝ) : ℝ := 44 + (3 / 2) * t
noncomputable def h2 (t : ℝ) : ℝ := 80 + (5 / 6) * t

theorem plants_same_height_after_54_years :
  ∃ t : ℝ, h1 t = h2 t :=
by
  use 54
  sorry

end plants_same_height_after_54_years_l186_186726


namespace value_of_x_minus_y_l186_186882

theorem value_of_x_minus_y (x y a : ℝ) (h₁ : x + y > 0) (h₂ : a < 0) (h₃ : a * y > 0) : x - y > 0 :=
sorry

end value_of_x_minus_y_l186_186882


namespace math_score_prob_l186_186404

noncomputable def normal_dist_prob (μ σ: ℝ) (a b: ℝ) : ℝ :=
  ∫ x in a..b, PDF (Normal μ σ) x dx

theorem math_score_prob 
  (μ σ : ℝ) (hμ : μ = 90) (ha : normal_dist_prob μ σ 70 90 = 0.4) :
  normal_dist_prob μ σ (∅) 110 = 0.9 :=
by 
  sorry

end math_score_prob_l186_186404


namespace cafeteria_total_cost_l186_186431

-- Definitions based on conditions
def cost_per_coffee := 4
def cost_per_cake := 7
def cost_per_ice_cream := 3
def mell_coffee := 2 
def mell_cake := 1 
def friends_coffee := 2 
def friends_cake := 1 
def friends_ice_cream := 1 
def num_friends := 2
def total_coffee := mell_coffee + num_friends * friends_coffee
def total_cake := mell_cake + num_friends * friends_cake
def total_ice_cream := num_friends * friends_ice_cream

-- Total cost
def total_cost := total_coffee * cost_per_coffee + total_cake * cost_per_cake + total_ice_cream * cost_per_ice_cream

-- Theorem statement
theorem cafeteria_total_cost : total_cost = 51 := by
  sorry

end cafeteria_total_cost_l186_186431


namespace inequality_proof_l186_186217

theorem inequality_proof (a b : ℝ) (h1 : a + b < 0) (h2 : b > 0) : a^2 > -a * b ∧ -a * b > b^2 := 
by
  sorry

end inequality_proof_l186_186217


namespace min_possible_A_div_C_l186_186530

theorem min_possible_A_div_C (x : ℝ) (A C : ℝ) (h1 : x^2 + (1/x)^2 = A) (h2 : x + 1/x = C) (h3 : 0 < A) (h4 : 0 < C) :
  ∃ (C : ℝ), C = Real.sqrt 2 ∧ (∀ B, (x^2 + (1/x)^2 = B) → (x + 1/x = C) → (B / C = 0 → B = 0)) :=
by
  sorry

end min_possible_A_div_C_l186_186530


namespace area_of_smallest_square_l186_186147

theorem area_of_smallest_square (radius : ℝ) (h : radius = 6) : 
    ∃ s : ℝ, s = 2 * radius ∧ s^2 = 144 :=
by
  sorry

end area_of_smallest_square_l186_186147


namespace survivor_probability_l186_186585

noncomputable def choose (n k : ℕ) : ℕ := Nat.choose n k

theorem survivor_probability :
  let total_people := 20
  let tribe_size := 10
  let droppers := 3
  let total_ways := choose total_people droppers
  let tribe_ways := choose tribe_size droppers
  let same_tribe_ways := 2 * tribe_ways
  let probability := same_tribe_ways / total_ways
  probability = 20 / 95 :=
by
  let total_people := 20
  let tribe_size := 10
  let droppers := 3
  let total_ways := choose total_people droppers
  let tribe_ways := choose tribe_size droppers
  let same_tribe_ways := 2 * tribe_ways
  let probability := same_tribe_ways / total_ways
  have : probability = 20 / 95 := sorry
  exact this

end survivor_probability_l186_186585


namespace total_fruits_proof_l186_186175

-- Definitions of the quantities involved in the problem.
def apples_basket1_to_3 := 9
def oranges_basket1_to_3 := 15
def bananas_basket1_to_3 := 14
def apples_basket4 := apples_basket1_to_3 - 2
def oranges_basket4 := oranges_basket1_to_3 - 2
def bananas_basket4 := bananas_basket1_to_3 - 2

-- Total fruits in first three baskets
def total_fruits_baskets1_to_3 := 3 * (apples_basket1_to_3 + oranges_basket1_to_3 + bananas_basket1_to_3)

-- Total fruits in fourth basket
def total_fruits_basket4 := apples_basket4 + oranges_basket4 + bananas_basket4

-- Total fruits in all four baskets
def total_fruits_all_baskets := total_fruits_baskets1_to_3 + total_fruits_basket4

-- Theorem statement
theorem total_fruits_proof : total_fruits_all_baskets = 146 :=
by
  -- Placeholder for proof
  sorry

end total_fruits_proof_l186_186175


namespace averages_correct_l186_186715

variables (marksEnglish totalEnglish marksMath totalMath marksPhysics totalPhysics 
           marksChemistry totalChemistry marksBiology totalBiology 
           marksHistory totalHistory marksGeography totalGeography : ℕ)

variables (avgEnglish avgMath avgPhysics avgChemistry avgBiology avgHistory avgGeography : ℚ)

def Kamal_average_english : Prop :=
  marksEnglish = 76 ∧ totalEnglish = 120 ∧ avgEnglish = (marksEnglish / totalEnglish) * 100

def Kamal_average_math : Prop :=
  marksMath = 65 ∧ totalMath = 150 ∧ avgMath = (marksMath / totalMath) * 100

def Kamal_average_physics : Prop :=
  marksPhysics = 82 ∧ totalPhysics = 100 ∧ avgPhysics = (marksPhysics / totalPhysics) * 100

def Kamal_average_chemistry : Prop :=
  marksChemistry = 67 ∧ totalChemistry = 80 ∧ avgChemistry = (marksChemistry / totalChemistry) * 100

def Kamal_average_biology : Prop :=
  marksBiology = 85 ∧ totalBiology = 100 ∧ avgBiology = (marksBiology / totalBiology) * 100

def Kamal_average_history : Prop :=
  marksHistory = 92 ∧ totalHistory = 150 ∧ avgHistory = (marksHistory / totalHistory) * 100

def Kamal_average_geography : Prop :=
  marksGeography = 58 ∧ totalGeography = 75 ∧ avgGeography = (marksGeography / totalGeography) * 100

theorem averages_correct :
  ∀ (marksEnglish totalEnglish marksMath totalMath marksPhysics totalPhysics 
      marksChemistry totalChemistry marksBiology totalBiology 
      marksHistory totalHistory marksGeography totalGeography : ℕ),
  ∀ (avgEnglish avgMath avgPhysics avgChemistry avgBiology avgHistory avgGeography : ℚ),
  Kamal_average_english marksEnglish totalEnglish avgEnglish →
  Kamal_average_math marksMath totalMath avgMath →
  Kamal_average_physics marksPhysics totalPhysics avgPhysics →
  Kamal_average_chemistry marksChemistry totalChemistry avgChemistry →
  Kamal_average_biology marksBiology totalBiology avgBiology →
  Kamal_average_history marksHistory totalHistory avgHistory →
  Kamal_average_geography marksGeography totalGeography avgGeography →
  avgEnglish = 63.33 ∧ avgMath = 43.33 ∧ avgPhysics = 82 ∧
  avgChemistry = 83.75 ∧ avgBiology = 85 ∧ avgHistory = 61.33 ∧ avgGeography = 77.33 :=
by
  sorry

end averages_correct_l186_186715


namespace cubic_expression_l186_186881

theorem cubic_expression (x : ℝ) (hx : x + 1/x = -7) : x^3 + 1/x^3 = -322 :=
by sorry

end cubic_expression_l186_186881


namespace problem1_l186_186164

theorem problem1 (x y : ℝ) (h : x + y > 2) : x > 1 ∨ y > 1 :=
sorry

end problem1_l186_186164


namespace largest_divisor_of_product_of_5_consecutive_integers_l186_186783

/-- What is the largest integer that must divide the product of any 5 consecutive integers? -/
theorem largest_divisor_of_product_of_5_consecutive_integers :
  ∀ n : ℤ, ∃ d : ℤ, d = 24 ∧ d ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  sorry

end largest_divisor_of_product_of_5_consecutive_integers_l186_186783


namespace statement_A_statement_E_statement_B_false_statement_C_false_statement_D_false_l186_186822

theorem statement_A : ∃ n : ℤ, 20 = 4 * n := by 
  sorry

theorem statement_E : ∃ n : ℤ, 180 = 9 * n := by 
  sorry

theorem statement_B_false : ¬ (19 ∣ 57) := by 
  sorry

theorem statement_C_false : 30 ∣ 90 := by 
  sorry

theorem statement_D_false : 17 ∣ 51 := by 
  sorry

end statement_A_statement_E_statement_B_false_statement_C_false_statement_D_false_l186_186822


namespace sum_of_variables_l186_186884

theorem sum_of_variables (a b c d : ℝ) (h₁ : a * c + a * d + b * c + b * d = 68) (h₂ : c + d = 4) : a + b + c + d = 21 :=
sorry

end sum_of_variables_l186_186884


namespace find_tricksters_l186_186325

def inhab_group : Type := { n : ℕ // n < 65 }
def is_knight (i : inhab_group) : Prop := ∀ g : inhab_group, true -- Placeholder for the actual property

theorem find_tricksters (inhabitants : inhab_group → Prop)
  (is_knight : inhab_group → Prop)
  (knight_always_tells_truth : ∀ i : inhab_group, is_knight i → inhabitants i = true)
  (tricksters_2_and_rest_knights : ∃ t1 t2 : inhab_group, t1 ≠ t2 ∧ ¬is_knight t1 ∧ ¬is_knight t2 ∧
    (∀ i : inhab_group, i ≠ t1 → i ≠ t2 → is_knight i)) :
  ∃ find_them : inhab_group → inhab_group → Prop, (∀ q_count : ℕ, q_count ≤ 16) → 
  ∃ t1 t2 : inhab_group, find_them t1 t2 :=
by 
  -- The proof goes here
  sorry

end find_tricksters_l186_186325


namespace average_speed_of_entire_trip_l186_186297

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

end average_speed_of_entire_trip_l186_186297


namespace smallest_square_area_l186_186148

theorem smallest_square_area : ∀ (r : ℝ), r = 6 → ∃ (a : ℝ), a = 12^2 := by 
  intro r hr
  use (12:ℝ)^2
  sorry

end smallest_square_area_l186_186148


namespace total_length_figure_2_l186_186841

-- Define the conditions for Figure 1
def left_side_figure_1 := 10
def right_side_figure_1 := 7
def top_side_figure_1 := 3
def bottom_side_figure_1_seg1 := 2
def bottom_side_figure_1_seg2 := 1

-- Define the conditions for Figure 2 after removal
def left_side_figure_2 := left_side_figure_1
def right_side_figure_2 := right_side_figure_1
def top_side_figure_2 := 0
def bottom_side_figure_2 := top_side_figure_1 + bottom_side_figure_1_seg1 + bottom_side_figure_1_seg2

-- The Lean statement proving the total length in Figure 2
theorem total_length_figure_2 : 
  left_side_figure_2 + right_side_figure_2 + top_side_figure_2 + bottom_side_figure_2 = 23 := by
  sorry

end total_length_figure_2_l186_186841


namespace B_is_werewolf_l186_186406

def is_werewolf (x : Type) : Prop := sorry
def is_knight (x : Type) : Prop := sorry
def is_liar (x : Type) : Prop := sorry

variables (A B : Type)

-- Conditions
axiom one_is_werewolf : is_werewolf A ∨ is_werewolf B
axiom only_one_werewolf : ¬ (is_werewolf A ∧ is_werewolf B)
axiom A_statement : is_werewolf A → is_knight A
axiom B_statement : is_werewolf B → is_liar B

theorem B_is_werewolf : is_werewolf B := 
by
  sorry

end B_is_werewolf_l186_186406


namespace complement_union_l186_186913

def A := { x : ℤ | ∃ k : ℤ, x = 3 * k + 1 }
def B := { x : ℤ | ∃ k : ℤ, x = 3 * k + 2 }
def U := ℤ

theorem complement_union :
  { x : ℤ | x ∉ (A ∪ B) } = { x : ℤ | ∃ k : ℤ, x = 3 * k } := 
by 
  sorry

end complement_union_l186_186913


namespace speed_of_man_in_still_water_l186_186019

def upstream_speed := 34 -- in kmph
def downstream_speed := 48 -- in kmph

def speed_in_still_water := (upstream_speed + downstream_speed) / 2

theorem speed_of_man_in_still_water :
  speed_in_still_water = 41 := by
  sorry

end speed_of_man_in_still_water_l186_186019


namespace original_denominator_l186_186030

theorem original_denominator (d : ℤ) (h1 : 5 = d + 3) : d = 12 := 
by 
  sorry

end original_denominator_l186_186030


namespace gangster_avoid_police_l186_186548

variable (a v : ℝ)
variable (house_side_length streets_distance neighbouring_distance police_interval : ℝ)
variable (police_speed gangster_speed_to_avoid_police : ℝ)

-- Given conditions
axiom house_properties : house_side_length = a ∧ neighbouring_distance = 2 * a
axiom streets_properties : streets_distance = 3 * a
axiom police_properties : police_interval = 9 * a ∧ police_speed = v

-- Correct answer in terms of Lean
theorem gangster_avoid_police :
  gangster_speed_to_avoid_police = 2 * v ∨ gangster_speed_to_avoid_police = v / 2 :=
by
  sorry

end gangster_avoid_police_l186_186548


namespace complex_number_solution_l186_186724

theorem complex_number_solution (z : ℂ) (i : ℂ) (hi : i * i = -1) (h : i * z = 1) : z = -i :=
by
  -- Mathematical proof will be here
  sorry

end complex_number_solution_l186_186724


namespace find_tricksters_in_16_questions_l186_186327

-- Definitions
def Inhabitant := {i : Nat // i < 65}
def isKnight (i : Inhabitant) : Prop := sorry  -- placeholder for actual definition
def isTrickster (i : Inhabitant) : Prop := sorry -- placeholder for actual definition

-- Conditions
def condition1 : ∀ i : Inhabitant, isKnight i ∨ isTrickster i := sorry
def condition2 : ∃ t1 t2 : Inhabitant, t1 ≠ t2 ∧ isTrickster t1 ∧ isTrickster t2 :=
  sorry
def condition3 : ∀ t1 t2 : Inhabitant, t1 ≠ t2 → ¬(isTrickster t1 ∧ isTrickster t2) → isKnight t1 ∧ isKnight t2 := 
  sorry 
def question (i : Inhabitant) (group : List Inhabitant) : Prop :=
  ∀ j ∈ group, isKnight j

-- Theorem statement
theorem find_tricksters_in_16_questions : ∃ (knight : Inhabitant) (knaves : (Inhabitant × Inhabitant)), 
  isKnight knight ∧ isTrickster knaves.fst ∧ isTrickster knaves.snd ∧ knaves.fst ≠ knaves.snd ∧
  (∀ questionsAsked ≤ 30, sorry) :=
  sorry

end find_tricksters_in_16_questions_l186_186327


namespace product_of_five_consecutive_divisible_by_30_l186_186776

theorem product_of_five_consecutive_divisible_by_30 :
  ∀ n : ℤ, 30 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  sorry

end product_of_five_consecutive_divisible_by_30_l186_186776


namespace unique_solution_l186_186848

noncomputable def valid_solutions (a : ℝ) : Prop :=
  ∃ x : ℝ, x^2 + (2 - a) * x + 1 = 0 ∧ -1 < x ∧ x ≤ 3 ∧ x ≠ 0 ∧ x ≠ 1 ∧ x ≠ 2

theorem unique_solution (a : ℝ) :
  (valid_solutions a) ↔ (a = 4.5 ∨ (a < 0) ∨ (a > 16 / 3)) := 
sorry

end unique_solution_l186_186848


namespace cars_in_garage_l186_186285

/-
Conditions:
1. Total wheels in the garage: 22
2. Riding lawnmower wheels: 4
3. Timmy's bicycle wheels: 2
4. Each of Timmy's parents' bicycles: 2 wheels, and there are 2 bicycles.
5. Joey's tricycle wheels: 3
6. Timmy's dad's unicycle wheels: 1

Question: How many cars are inside the garage?

Correct Answer: The number of cars is 2.
-/
theorem cars_in_garage (total_wheels : ℕ) (lawnmower_wheels : ℕ)
  (timmy_bicycle_wheels : ℕ) (parents_bicycles_wheels : ℕ)
  (joey_tricycle_wheels : ℕ) (dad_unicycle_wheels : ℕ) 
  (cars_wheels : ℕ) (cars : ℕ) :
  total_wheels = 22 →
  lawnmower_wheels = 4 →
  timmy_bicycle_wheels = 2 →
  parents_bicycles_wheels = 2 * 2 →
  joey_tricycle_wheels = 3 →
  dad_unicycle_wheels = 1 →
  cars_wheels = total_wheels - (lawnmower_wheels + timmy_bicycle_wheels + parents_bicycles_wheels + joey_tricycle_wheels + dad_unicycle_wheels) →
  cars = cars_wheels / 4 →
  cars = 2 := by
  sorry

end cars_in_garage_l186_186285


namespace probability_of_rolling_2_4_6_l186_186620

open Set Classical

noncomputable def fair_six_sided_die_outcomes : Finset ℕ := {1, 2, 3, 4, 5, 6}

def successful_outcomes : Finset ℕ := {2, 4, 6}

theorem probability_of_rolling_2_4_6 : 
  (successful_outcomes.card : ℚ) / (fair_six_sided_die_outcomes.card : ℚ) = 1 / 2 :=
by
  sorry

end probability_of_rolling_2_4_6_l186_186620


namespace trihedral_sum_of_angles_le_sum_of_plane_angles_trihedral_sum_of_angles_ge_half_sum_of_plane_angles_l186_186826

-- Part a
theorem trihedral_sum_of_angles_le_sum_of_plane_angles
  (α β γ : ℝ) (ASB BSC CSA : ℝ)
  (h1 : α ≤ ASB)
  (h2 : β ≤ BSC)
  (h3 : γ ≤ CSA) :
  α + β + γ ≤ ASB + BSC + CSA :=
sorry

-- Part b
theorem trihedral_sum_of_angles_ge_half_sum_of_plane_angles
  (α_S β_S γ_S : ℝ) (ASB BSC CSA : ℝ) 
  (h_acute : ASB < (π / 2) ∧ BSC < (π / 2) ∧ CSA < (π / 2))
  (h1 : α_S ≥ (1/2) * ASB)
  (h2 : β_S ≥ (1/2) * BSC)
  (h3 : γ_S ≥ (1/2) * CSA) :
  α_S + β_S + γ_S ≥ (1/2) * (ASB + BSC + CSA) :=
sorry

end trihedral_sum_of_angles_le_sum_of_plane_angles_trihedral_sum_of_angles_ge_half_sum_of_plane_angles_l186_186826


namespace bowling_tournament_orders_l186_186240

theorem bowling_tournament_orders :
  let choices := 2
  let number_of_games := 5
  let total_orders := choices ^ number_of_games
  total_orders = 32 :=
by
  let choices := 2
  let number_of_games := 5
  let total_orders := choices ^ number_of_games
  show total_orders = 32
  sorry

end bowling_tournament_orders_l186_186240


namespace min_fraction_l186_186532

theorem min_fraction (x A C : ℝ) (hx : x > 0) (hA : A = x^2 + 1/x^2) (hC : C = x + 1/x) :
  ∃ m, m = 2 * Real.sqrt 2 ∧ ∀ B, B > 0 → x^2 + 1/x^2 = B → x + 1/x = C → B / C ≥ m :=
by
  sorry

end min_fraction_l186_186532


namespace largest_divisor_of_5_consecutive_integers_l186_186751

theorem largest_divisor_of_5_consecutive_integers :
  ∃ d : ℤ, (∀ n : ℤ, d ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4))) ∧ d = 120 :=
sorry

end largest_divisor_of_5_consecutive_integers_l186_186751


namespace oranges_in_bin_l186_186997

theorem oranges_in_bin (initial : ℕ) (thrown_away : ℕ) (added : ℕ) (result : ℕ)
    (h_initial : initial = 40)
    (h_thrown_away : thrown_away = 25)
    (h_added : added = 21)
    (h_result : result = 36) : initial - thrown_away + added = result :=
by
  -- skipped proof steps
  exact sorry

end oranges_in_bin_l186_186997


namespace initial_speed_is_correct_l186_186837

def initial_speed (v : ℝ) : Prop :=
  let D_total : ℝ := 70 * 5
  let D_2 : ℝ := 85 * 2
  let D_1 := v * 3
  D_total = D_1 + D_2

theorem initial_speed_is_correct :
  ∃ v : ℝ, initial_speed v ∧ v = 60 :=
by
  sorry

end initial_speed_is_correct_l186_186837


namespace infinite_non_congruent_right_triangles_l186_186279

noncomputable def right_triangle_equal_perimeter_area : Prop :=
  ∃ (a b c : ℝ), (a > 0) ∧ (b > 0) ∧ (c > 0) ∧ (a^2 + b^2 = c^2) ∧ 
  (a + b + c = (1/2) * a * b)

theorem infinite_non_congruent_right_triangles :
  ∃ (k : ℕ), right_triangle_equal_perimeter_area :=
sorry

end infinite_non_congruent_right_triangles_l186_186279


namespace constant_term_of_first_equation_l186_186871

theorem constant_term_of_first_equation
  (y z : ℤ)
  (h1 : 2 * 20 - y - z = 40)
  (h2 : 3 * 20 + y - z = 20)
  (hx : 20 = 20) :
  4 * 20 + y + z = 80 := 
sorry

end constant_term_of_first_equation_l186_186871


namespace tan_neg_five_pi_div_four_l186_186509

theorem tan_neg_five_pi_div_four : Real.tan (- (5 * Real.pi / 4)) = -1 := 
sorry

end tan_neg_five_pi_div_four_l186_186509


namespace kim_shoes_l186_186557

variable (n : ℕ)

theorem kim_shoes : 
  (∀ n, 2 * n = 6 → (1 : ℚ) / (2 * n - 1) = (1 : ℚ) / 5 → n = 3) := 
sorry

end kim_shoes_l186_186557


namespace intersection_A_B_l186_186561

-- Definitions for sets A and B based on the problem conditions
def A : Set ℝ := { x | x^2 - 2 * x - 3 < 0 }
def B : Set ℝ := { x | ∃ y : ℝ, y = Real.log (2 - x) }

-- Proof problem statement
theorem intersection_A_B : (A ∩ B) = { x : ℝ | -1 < x ∧ x < 2 } :=
by
  sorry

end intersection_A_B_l186_186561


namespace necessary_but_not_sufficient_l186_186214

-- Definitions from the conditions
def p (a b : ℤ) : Prop := True  -- Since their integrality is given
def q (a b : ℤ) : Prop := ∃ (x : ℤ), (x^2 + a * x + b = 0)

theorem necessary_but_not_sufficient (a b : ℤ) : 
  (¬ (p a b → q a b)) ∧ (q a b → p a b) :=
by
  sorry

end necessary_but_not_sufficient_l186_186214


namespace age_difference_l186_186446

variable (S R : ℝ)

theorem age_difference (h1 : S = 38.5) (h2 : S / R = 11 / 9) : S - R = 7 :=
by
  sorry

end age_difference_l186_186446


namespace largest_integer_dividing_consecutive_product_l186_186812

theorem largest_integer_dividing_consecutive_product :
  ∀ (n : ℤ), (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) % 120 = 0 :=
by
  sorry

end largest_integer_dividing_consecutive_product_l186_186812


namespace chocolate_syrup_per_glass_l186_186345

-- Definitions from the conditions
def each_glass_volume : ℝ := 8
def milk_per_glass : ℝ := 6.5
def total_milk : ℝ := 130
def total_chocolate_syrup : ℝ := 60
def total_chocolate_milk : ℝ := 160

-- Proposition and statement to prove
theorem chocolate_syrup_per_glass : 
  (total_chocolate_milk / each_glass_volume) * milk_per_glass = total_milk → 
  (each_glass_volume - milk_per_glass = 1.5) := 
by 
  sorry

end chocolate_syrup_per_glass_l186_186345


namespace cafeteria_total_cost_l186_186433

-- Definitions based on conditions
def cost_per_coffee := 4
def cost_per_cake := 7
def cost_per_ice_cream := 3
def mell_coffee := 2 
def mell_cake := 1 
def friends_coffee := 2 
def friends_cake := 1 
def friends_ice_cream := 1 
def num_friends := 2
def total_coffee := mell_coffee + num_friends * friends_coffee
def total_cake := mell_cake + num_friends * friends_cake
def total_ice_cream := num_friends * friends_ice_cream

-- Total cost
def total_cost := total_coffee * cost_per_coffee + total_cake * cost_per_cake + total_ice_cream * cost_per_ice_cream

-- Theorem statement
theorem cafeteria_total_cost : total_cost = 51 := by
  sorry

end cafeteria_total_cost_l186_186433


namespace pawns_on_black_squares_even_l186_186507

theorem pawns_on_black_squares_even (A : Fin 8 → Fin 8) :
  ∃ n : ℕ, ∀ i, (i + A i).val % 2 = 1 → n % 2 = 0 :=
sorry

end pawns_on_black_squares_even_l186_186507


namespace min_fraction_l186_186533

theorem min_fraction (x A C : ℝ) (hx : x > 0) (hA : A = x^2 + 1/x^2) (hC : C = x + 1/x) :
  ∃ m, m = 2 * Real.sqrt 2 ∧ ∀ B, B > 0 → x^2 + 1/x^2 = B → x + 1/x = C → B / C ≥ m :=
by
  sorry

end min_fraction_l186_186533


namespace haley_small_gardens_l186_186525

theorem haley_small_gardens (total_seeds seeds_in_big_garden seeds_per_small_garden : ℕ) (h1 : total_seeds = 56) (h2 : seeds_in_big_garden = 35) (h3 : seeds_per_small_garden = 3) :
  (total_seeds - seeds_in_big_garden) / seeds_per_small_garden = 7 :=
by
  sorry

end haley_small_gardens_l186_186525


namespace derivative_of_f_l186_186522

-- Define the function
def f (x : ℝ) : ℝ := x^2 - x

-- State the theorem to prove
theorem derivative_of_f : ∀ x : ℝ,  (deriv f x = 2 * x - 1) :=
by sorry

end derivative_of_f_l186_186522


namespace area_of_region_l186_186950

noncomputable def bounded_area := 
  let equation (x y : ℝ) := x^2 + y^2 = 4*|y-x| + 2*|y+x|
  ∀ x y : ℝ, equation x y → (∃ m n : ℤ, bounded_area = m + n * real.pi ∧ m + n = 40)

theorem area_of_region {x y : ℝ} :
  let equation := x^2 + y^2 = 4*|y-x| + 2*|y+x|
  (∀ (h : x^2 + y^2 = 4*|y-x| + 2*|y+x|), ∃ m n : ℤ, (bounded_area x y) = m + n * real.pi ∧ m + n = 40) := 
by
  sorry

end area_of_region_l186_186950


namespace probability_both_perfect_square_and_multiple_of_4_l186_186049

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n
def is_multiple_of_4 (n : ℕ) : Prop := n % 4 = 0

def fifty_cards := {n : ℕ | n > 0 ∧ n ≤ 50}
def both_conditions (n : ℕ) : Prop := is_perfect_square n ∧ is_multiple_of_4 n

theorem probability_both_perfect_square_and_multiple_of_4 : 
  (∑ x in fifty_cards, if both_conditions x then 1 else 0) / 50 = 3 / 50 :=
sorry

end probability_both_perfect_square_and_multiple_of_4_l186_186049


namespace DF_length_l186_186706

-- Definitions for the given problem.
variable (AB DC EB DE : ℝ)
variable (parallelogram_ABCD : Prop)
variable (DE_altitude_AB : Prop)
variable (DF_altitude_BC : Prop)

-- Conditions
axiom AB_eq_DC : AB = DC
axiom EB_eq_5 : EB = 5
axiom DE_eq_8 : DE = 8

-- The main theorem to prove
theorem DF_length (hAB : AB = 15) (hDC : DC = 15) (hEB : EB = 5) (hDE : DE = 8)
  (hPar : parallelogram_ABCD)
  (hAltAB : DE_altitude_AB)
  (hAltBC : DF_altitude_BC) :
  ∃ DF : ℝ, DF = 8 := 
sorry

end DF_length_l186_186706


namespace number_of_books_Ryan_l186_186105

structure LibraryProblem :=
  (Total_pages_Ryan : ℕ)
  (Total_days : ℕ)
  (Pages_per_book_brother : ℕ)
  (Extra_pages_Ryan : ℕ)

def calculate_books_received (p : LibraryProblem) : ℕ :=
  let Total_pages_brother := p.Pages_per_book_brother * p.Total_days
  let Ryan_daily_average := (Total_pages_brother / p.Total_days) + p.Extra_pages_Ryan
  p.Total_pages_Ryan / Ryan_daily_average

theorem number_of_books_Ryan (p : LibraryProblem) (h1 : p.Total_pages_Ryan = 2100)
  (h2 : p.Total_days = 7) (h3 : p.Pages_per_book_brother = 200) (h4 : p.Extra_pages_Ryan = 100) :
  calculate_books_received p = 7 := by
  sorry

end number_of_books_Ryan_l186_186105


namespace exists_infinitely_many_n_l186_186931

def sum_of_digits (m : ℕ) : ℕ := 
  m.digits 10 |>.sum

theorem exists_infinitely_many_n (S : ℕ → ℕ) (h_sum_of_digits : ∀ m, S m = sum_of_digits m) :
  ∀ N : ℕ, ∃ n ≥ N, S (3 ^ n) ≥ S (3 ^ (n + 1)) :=
by { sorry }

end exists_infinitely_many_n_l186_186931


namespace largest_divisor_of_5_consecutive_integers_l186_186795

theorem largest_divisor_of_5_consecutive_integers :
  ∀ n : ℤ, ∃ k : ℤ, (product_of_5_consecutive_integers n = k) ∧ (60 ∣ k) 
:= sorry

end largest_divisor_of_5_consecutive_integers_l186_186795


namespace division_of_pow_of_16_by_8_eq_2_pow_4041_l186_186558

theorem division_of_pow_of_16_by_8_eq_2_pow_4041 :
  (16^1011) / 8 = 2^4041 :=
by
  -- Assume m = 16^1011
  let m := 16^1011
  -- Then expressing m in base 2
  have h_m_base2 : m = 2^4044 := by sorry
  -- Dividing m by 8
  have h_division : m / 8 = 2^4041 := by sorry
  -- Conclusion
  exact h_division

end division_of_pow_of_16_by_8_eq_2_pow_4041_l186_186558


namespace desired_interest_rate_l186_186991

def nominalValue : ℝ := 20
def dividendRate : ℝ := 0.09
def marketValue : ℝ := 15

theorem desired_interest_rate : (dividendRate * nominalValue / marketValue) * 100 = 12 := by
  sorry

end desired_interest_rate_l186_186991


namespace xiao_wang_scores_problem_l186_186483

-- Defining the problem conditions and solution as a proof problem
theorem xiao_wang_scores_problem (x y : ℕ) (h1 : (x * y + 98) / (x + 1) = y + 1) 
                                 (h2 : (x * y + 98 + 70) / (x + 2) = y - 1) :
  (x + 2 = 10) ∧ (y - 1 = 88) :=
by 
  sorry

end xiao_wang_scores_problem_l186_186483


namespace john_will_lose_weight_in_80_days_l186_186411

-- Assumptions based on the problem conditions
def calories_eaten : ℕ := 1800
def calories_burned : ℕ := 2300
def calories_to_lose_one_pound : ℕ := 4000
def pounds_to_lose : ℕ := 10

-- Definition of the net calories burned per day
def net_calories_burned_per_day : ℕ := calories_burned - calories_eaten

-- Definition of total calories to lose the target weight
def total_calories_to_lose_target_weight (pounds_to_lose : ℕ) : ℕ :=
  calories_to_lose_one_pound * pounds_to_lose

-- Definition of days to lose the target weight
def days_to_lose_weight (target_calories : ℕ) (daily_net_calories : ℕ) : ℕ :=
  target_calories / daily_net_calories

-- Prove that John will lose 10 pounds in 80 days
theorem john_will_lose_weight_in_80_days :
  days_to_lose_weight (total_calories_to_lose_target_weight pounds_to_lose) net_calories_burned_per_day = 80 := by
  sorry

end john_will_lose_weight_in_80_days_l186_186411


namespace arithmetic_sequence_a2a3_l186_186077

noncomputable def arithmetic_sequence_sum (a : Nat → ℝ) (d : ℝ) : Prop :=
  ∀ n : Nat, a (n + 1) = a n + d

theorem arithmetic_sequence_a2a3 
  (a : Nat → ℝ) (d : ℝ) 
  (arith_seq : arithmetic_sequence_sum a d)
  (H : a 1 + a 2 + a 3 + a 4 = 30) : 
  a 2 + a 3 = 15 :=
by 
sorry

end arithmetic_sequence_a2a3_l186_186077


namespace sum_three_consecutive_odd_integers_l186_186613

theorem sum_three_consecutive_odd_integers (n : ℤ) 
  (h : n + (n + 4) = 150) : n + (n + 2) + (n + 4) = 225 :=
by
  sorry

end sum_three_consecutive_odd_integers_l186_186613


namespace largest_divisor_of_product_of_5_consecutive_integers_l186_186781

/-- What is the largest integer that must divide the product of any 5 consecutive integers? -/
theorem largest_divisor_of_product_of_5_consecutive_integers :
  ∀ n : ℤ, ∃ d : ℤ, d = 24 ∧ d ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  sorry

end largest_divisor_of_product_of_5_consecutive_integers_l186_186781


namespace complement_union_l186_186910

def A := { x : ℤ | ∃ k : ℤ, x = 3 * k + 1 }
def B := { x : ℤ | ∃ k : ℤ, x = 3 * k + 2 }
def U := ℤ

theorem complement_union :
  { x : ℤ | x ∉ (A ∪ B) } = { x : ℤ | ∃ k : ℤ, x = 3 * k } := 
by 
  sorry

end complement_union_l186_186910


namespace solve_quadratic_inequality_l186_186265

theorem solve_quadratic_inequality (x : ℝ) :
  (-3 * x^2 + 8 * x + 5 > 0) ↔ (x < -1 / 3) :=
by
  sorry

end solve_quadratic_inequality_l186_186265


namespace graph_connected_probability_l186_186132

-- Given a complete graph with 20 vertices
def complete_graph_vertices : ℕ := 20

-- Total number of edges in the complete graph
def complete_graph_edges (n : ℕ) : ℕ := (n * (n - 1)) / 2

-- Given that 35 edges are removed
def removed_edges : ℕ := 35

-- Calculating probabilities used in the final answer
noncomputable def binomial (n k : ℕ) : ℚ := nat.choose n k

-- The probability that the graph remains connected
noncomputable def probability_connected (n k : ℕ) : ℚ :=
  1 - (20 * binomial ((complete_graph_edges n) - removed_edges + 1) (k - 1)) / binomial (complete_graph_edges n) k

-- The proof problem
theorem graph_connected_probability :
  probability_connected complete_graph_vertices removed_edges = 1 - (20 * binomial 171 16) / binomial 190 35 :=
sorry

end graph_connected_probability_l186_186132


namespace extra_pieces_of_gum_l186_186264

theorem extra_pieces_of_gum (total_packages : ℕ) (pieces_per_package : ℕ) (total_pieces : ℕ) : ℕ :=
  if total_packages = 43 ∧ pieces_per_package = 23 ∧ total_pieces = 997 then
    997 - (43 * 23)
  else
    0  -- This is a dummy value for other cases, as they do not satisfy our conditions.

#print extra_pieces_of_gum

end extra_pieces_of_gum_l186_186264


namespace tan_neg_5pi_over_4_l186_186511

theorem tan_neg_5pi_over_4 : Real.tan (-5 * Real.pi / 4) = -1 :=
by 
  sorry

end tan_neg_5pi_over_4_l186_186511


namespace minimal_face_sum_of_larger_cube_l186_186992

-- Definitions
def num_small_cubes : ℕ := 27
def num_faces_per_cube : ℕ := 6

-- The goal: Prove the minimal sum of the integers shown on the faces of the larger cube
theorem minimal_face_sum_of_larger_cube (min_sum : ℤ) 
    (H : min_sum = 90) :
    min_sum = 90 :=
by {
  sorry
}

end minimal_face_sum_of_larger_cube_l186_186992


namespace total_divisors_7350_l186_186472

def primeFactorization (n : ℕ) : List (ℕ × ℕ) :=
  if n = 7350 then [(2, 1), (3, 1), (5, 2), (7, 2)] else []

def totalDivisors (factors : List (ℕ × ℕ)) : ℕ :=
  factors.foldl (λ acc (p : ℕ × ℕ) => acc * (p.snd + 1)) 1

theorem total_divisors_7350 : totalDivisors (primeFactorization 7350) = 36 :=
by
  sorry

end total_divisors_7350_l186_186472


namespace total_pictures_uploaded_is_65_l186_186225

-- Given conditions
def first_album_pics : ℕ := 17
def album_pics : ℕ := 8
def number_of_albums : ℕ := 6

-- The theorem to be proved
theorem total_pictures_uploaded_is_65 : first_album_pics + number_of_albums * album_pics = 65 :=
by
  sorry

end total_pictures_uploaded_is_65_l186_186225


namespace ball_hits_ground_time_l186_186739

theorem ball_hits_ground_time :
  ∀ t : ℝ, y = -20 * t^2 + 30 * t + 60 → y = 0 → t = (3 + Real.sqrt 57) / 4 := by
  sorry

end ball_hits_ground_time_l186_186739


namespace range_of_a_l186_186538

theorem range_of_a (a : ℝ) :
  (¬ ∃ x0 : ℝ, ∀ x : ℝ, x + a * x0 + 1 < 0) → (a ≥ -2 ∧ a ≤ 2) :=
by
  sorry

end range_of_a_l186_186538


namespace triangle_count_correct_l186_186503

def is_valid_triangle (x1 y1 x2 y2 : ℕ) : Prop :=
  31 * x1 + y1 = 2017 ∧ 31 * x2 + y2 = 2017 ∧
  x1 ≠ x2 ∧ (31 * x1 + y1) ≠ (31 * x2 + y2) ∧
  (x1 - x2) % 2 = 0 ∧
  x1 ≤ 65 ∧ x2 ≤ 65

def count_valid_triangles : ℕ :=
  let even_x := { x | ∀ (n : ℕ), n ≤ 65 ∧ n % 2 = 0 }
  let odd_x := { x | ∀ (n : ℕ), n ≤ 65 ∧ n % 2 = 1 }
  2 * (finset.card even_x.choose 2 + finset.card odd_x.choose 2)

theorem triangle_count_correct : count_valid_triangles = 1056 := 
sorry

end triangle_count_correct_l186_186503


namespace original_number_is_repeating_decimal_l186_186485

theorem original_number_is_repeating_decimal :
  ∃ N : ℚ, (N * 10 ^ 28) % 10^30 = 15 ∧ N * 5 = 0.7894736842105263 ∧ 
  (N = 3 / 19) :=
sorry

end original_number_is_repeating_decimal_l186_186485


namespace eval_expr_l186_186048

theorem eval_expr :
  - (18 / 3 * 8 - 48 + 4 * 6) = -24 := by
  sorry

end eval_expr_l186_186048


namespace pythagorean_triplet_l186_186017

theorem pythagorean_triplet (k : ℕ) :
  let a := k
  let b := 2 * k - 2
  let c := 2 * k - 1
  (a * b) ^ 2 + c ^ 2 = (2 * k ^ 2 - 2 * k + 1) ^ 2 :=
by
  sorry

end pythagorean_triplet_l186_186017


namespace car_speed_in_kmph_l186_186829

def speed_mps : ℝ := 10  -- The speed of the car in meters per second
def conversion_factor : ℝ := 3.6  -- The conversion factor from m/s to km/h

theorem car_speed_in_kmph : speed_mps * conversion_factor = 36 := 
by
  sorry

end car_speed_in_kmph_l186_186829


namespace median_production_l186_186454

def production_data : List ℕ := [5, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10]

def median (l : List ℕ) : ℕ :=
  if l.length % 2 = 1 then
    l.nthLe (l.length / 2) sorry
  else
    let m := l.length / 2
    (l.nthLe (m - 1) sorry + l.nthLe m sorry) / 2

theorem median_production :
  median (production_data) = 8 :=
by
  sorry

end median_production_l186_186454


namespace sum_of_fractions_l186_186649

theorem sum_of_fractions : 
  (1 / 10) + (2 / 10) + (3 / 10) + (4 / 10) + (10 / 10) + (11 / 10) + (15 / 10) + (20 / 10) + (25 / 10) + (50 / 10) = 14.1 :=
by sorry

end sum_of_fractions_l186_186649


namespace sin_double_angle_l186_186054

theorem sin_double_angle (α : ℝ) (h : Real.cos (π / 4 - α) = 3 / 5) : Real.sin (2 * α) = -7 / 25 :=
by
  sorry

end sin_double_angle_l186_186054


namespace calvin_gym_duration_l186_186343

theorem calvin_gym_duration (initial_weight loss_per_month final_weight : ℕ) (h1 : initial_weight = 250)
    (h2 : loss_per_month = 8) (h3 : final_weight = 154) : 
    (initial_weight - final_weight) / loss_per_month = 12 :=
by 
  sorry

end calvin_gym_duration_l186_186343


namespace michael_total_time_l186_186255

def time_for_200_meters (distance speed : ℕ) : ℚ :=
  distance / speed

def total_time_per_lap : ℚ :=
  (time_for_200_meters 200 6) + (time_for_200_meters 200 3)

def total_time_8_laps : ℚ :=
  8 * total_time_per_lap

theorem michael_total_time : total_time_8_laps = 800 :=
by
  -- The proof would go here
  sorry

end michael_total_time_l186_186255


namespace trigonometry_problem_l186_186859

noncomputable def f (a b x : ℝ) : ℝ := a * (Real.sin x) ^ 3 + b * Real.tan x + 1

theorem trigonometry_problem (a b : ℝ) (h : f a b 2 = 3) : f a b (2 * Real.pi - 2) = -1 :=
by
  unfold f at *
  sorry

end trigonometry_problem_l186_186859


namespace tom_savings_l186_186968

theorem tom_savings :
  let insurance_cost_per_month := 20
  let total_months := 24
  let procedure_cost := 5000
  let insurance_coverage := 0.80
  let total_insurance_cost := total_months * insurance_cost_per_month
  let insurance_cover_amount := procedure_cost * insurance_coverage
  let out_of_pocket_cost := procedure_cost - insurance_cover_amount
  let savings := procedure_cost - total_insurance_cost - out_of_pocket_cost
  savings = 3520 :=
by
  sorry

end tom_savings_l186_186968


namespace sin_690_eq_neg_half_l186_186351

theorem sin_690_eq_neg_half :
  let rad := Real.pi / 180 in -- Convert degrees to radians
  Real.sin (690 * rad) = -1 / 2 :=
by
  sorry

end sin_690_eq_neg_half_l186_186351


namespace largest_divisor_of_5_consecutive_integers_l186_186756

theorem largest_divisor_of_5_consecutive_integers :
  ∃ d : ℤ, (∀ n : ℤ, d ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4))) ∧ d = 120 :=
sorry

end largest_divisor_of_5_consecutive_integers_l186_186756


namespace classroom_gpa_l186_186161

theorem classroom_gpa (n : ℕ) (h1 : 1 ≤ n) : 
  (1/3 : ℝ) * 30 + (2/3 : ℝ) * 33 = 32 :=
by sorry

end classroom_gpa_l186_186161


namespace product_of_five_consecutive_divisible_by_30_l186_186780

theorem product_of_five_consecutive_divisible_by_30 :
  ∀ n : ℤ, 30 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  sorry

end product_of_five_consecutive_divisible_by_30_l186_186780


namespace distance_between_centers_l186_186878

variable (P R r : ℝ)
variable (h_tangent : P = R - r)
variable (h_radius1 : R = 6)
variable (h_radius2 : r = 3)

theorem distance_between_centers : P = 3 := by
  sorry

end distance_between_centers_l186_186878


namespace find_a_l186_186938

noncomputable def A : Set ℝ := {x | x^2 - 3 * x + 2 = 0}
noncomputable def B (a : ℝ) : Set ℝ := {x | x^2 - a * x + a - 1 = 0}

theorem find_a (a : ℝ) (h : A ∪ B a = A) : a = 2 ∨ a = 3 := by
  sorry

end find_a_l186_186938


namespace vertex_of_parabola_l186_186737

theorem vertex_of_parabola :
  ∃ h k : ℝ, (∀ x : ℝ, 3 * (x + 4)^2 - 9 = 3 * (x - h)^2 + k) ∧ (h, k) = (-4, -9) :=
by
  sorry

end vertex_of_parabola_l186_186737


namespace new_individuals_weight_l186_186953

variables (W : ℝ) (A B C : ℝ)

-- Conditions
def original_twelve_people_weight : ℝ := W
def weight_leaving_1 : ℝ := 64
def weight_leaving_2 : ℝ := 75
def weight_leaving_3 : ℝ := 81
def average_increase : ℝ := 3.6
def total_weight_increase : ℝ := 12 * average_increase
def weight_leaving_sum : ℝ := weight_leaving_1 + weight_leaving_2 + weight_leaving_3

-- Equation derived from the problem conditions
def new_individuals_weight_sum : ℝ := weight_leaving_sum + total_weight_increase

-- Theorem to prove
theorem new_individuals_weight :
  A + B + C = 263.2 :=
by
  sorry

end new_individuals_weight_l186_186953


namespace reciprocals_and_opposites_l186_186069

theorem reciprocals_and_opposites (a b c d : ℝ) (h_ab : a * b = 1) (h_cd : c + d = 0) : 
  (c + d)^2 - a * b = -1 := by
  sorry

end reciprocals_and_opposites_l186_186069


namespace focus_of_parabola_l186_186736

-- Define the given parabola equation
def parabola_eq (x y : ℝ) : Prop := y = (1/4) * x^2

-- Define the conditions about the focus and the parabola direction
def focus_on_y_axis : Prop := True -- Given condition
def opens_upwards : Prop := True -- Given condition

theorem focus_of_parabola (x y : ℝ) 
  (h1 : parabola_eq x y) 
  (h2 : focus_on_y_axis) 
  (h3 : opens_upwards) : 
  (x = 0 ∧ y = 1) :=
by
  sorry

end focus_of_parabola_l186_186736


namespace ekon_uma_diff_l186_186012

-- Definitions based on conditions
def total_videos := 411
def kelsey_videos := 160
def ekon_kelsey_diff := 43

-- Definitions derived from conditions
def ekon_videos := kelsey_videos - ekon_kelsey_diff
def uma_videos (E : ℕ) := total_videos - kelsey_videos - E

-- The Lean problem statement
theorem ekon_uma_diff : 
  uma_videos ekon_videos - ekon_videos = 17 := 
by 
  sorry

end ekon_uma_diff_l186_186012


namespace area_of_square_l186_186182

theorem area_of_square (d : ℝ) (hd : d = 14 * Real.sqrt 2) : 
  ∃ (A : ℝ), A = 196 := by
  sorry

end area_of_square_l186_186182


namespace largest_divisor_of_5_consecutive_integers_l186_186798

theorem largest_divisor_of_5_consecutive_integers :
  ∀ n : ℤ, ∃ k : ℤ, (product_of_5_consecutive_integers n = k) ∧ (60 ∣ k) 
:= sorry

end largest_divisor_of_5_consecutive_integers_l186_186798


namespace right_triangle_inradius_l186_186362

theorem right_triangle_inradius (a b c : ℕ) (h : a = 6) (h2 : b = 8) (h3 : c = 10) :
  ((a^2 + b^2 = c^2) ∧ (1/2 * ↑a * ↑b = 24) ∧ ((a + b + c) / 2 = 12) ∧ (24 = 12 * 2)) :=
by 
  sorry

end right_triangle_inradius_l186_186362


namespace total_weight_correct_l186_186468

-- Define the weights given in the problem
def dog_weight_kg := 2 -- weight in kilograms
def dog_weight_g := 600 -- additional grams
def cat_weight_g := 3700 -- weight in grams

-- Convert dog's weight to grams
def dog_weight_total_g : ℕ := dog_weight_kg * 1000 + dog_weight_g

-- Define the total weight of the animals (dog + cat)
def total_weight_animals_g : ℕ := dog_weight_total_g + cat_weight_g

-- Theorem stating that the total weight of the animals is 6300 grams
theorem total_weight_correct : total_weight_animals_g = 6300 := by
  sorry

end total_weight_correct_l186_186468


namespace length_of_faster_train_l186_186288

/-- Define the speeds of the trains in kmph -/
def speed_faster_train := 180 -- in kmph
def speed_slower_train := 90  -- in kmph

/-- Convert speeds to m/s -/
def kmph_to_mps (speed : ℕ) : ℕ := speed * 5 / 18

/-- Define the relative speed in m/s -/
def relative_speed := kmph_to_mps speed_faster_train - kmph_to_mps speed_slower_train

/-- Define the time it takes for the faster train to cross the man in seconds -/
def crossing_time := 15 -- in seconds

/-- Define the length of the train calculation in meters -/
noncomputable def length_faster_train := relative_speed * crossing_time

theorem length_of_faster_train :
  length_faster_train = 375 :=
by
  sorry

end length_of_faster_train_l186_186288


namespace principal_amount_correct_l186_186333

noncomputable def initial_amount (A : ℝ) (R : ℝ) (T : ℝ) : ℝ :=
  (A * 100) / (R * T + 100)

theorem principal_amount_correct : initial_amount 950 9.230769230769232 5 = 650 := by
  sorry

end principal_amount_correct_l186_186333


namespace water_level_balance_l186_186570

noncomputable def exponential_decay (a n t : ℝ) : ℝ := a * Real.exp (n * t)

theorem water_level_balance
  (a : ℝ)
  (n : ℝ)
  (m : ℝ)
  (h5 : exponential_decay a n 5 = a / 2)
  (h8 : exponential_decay a n m = a / 8) :
  m = 10 := by
  sorry

end water_level_balance_l186_186570


namespace average_billboards_per_hour_l186_186947

def first_hour_billboards : ℕ := 17
def second_hour_billboards : ℕ := 20
def third_hour_billboards : ℕ := 23

theorem average_billboards_per_hour : 
  (first_hour_billboards + second_hour_billboards + third_hour_billboards) / 3 = 20 := 
by
  sorry

end average_billboards_per_hour_l186_186947


namespace exists_initial_value_l186_186640

theorem exists_initial_value (x : ℤ) : ∃ y : ℤ, x + 49 = y^2 :=
sorry

end exists_initial_value_l186_186640


namespace distance_A_focus_l186_186467

-- Definitions from the problem conditions
def parabola_eq (x y : ℝ) : Prop := x^2 = 4 * y
def point_A (x : ℝ) : Prop := parabola_eq x 4
def focus_y_coord : ℝ := 1 -- Derived from the standard form of the parabola x^2 = 4py where p=1

-- State the theorem in Lean 4
theorem distance_A_focus (x : ℝ) (hA : point_A x) : |4 - focus_y_coord| = 3 :=
by
  -- Proof would go here
  sorry

end distance_A_focus_l186_186467


namespace meals_calculation_l186_186036

def combined_meals (k a : ℕ) : ℕ :=
  k + a

theorem meals_calculation :
  ∀ (k a : ℕ), k = 8 → (2 * a = k) → combined_meals k a = 12 :=
  by
    intros k a h1 h2
    rw [h1] at h2
    have ha : a = 4 := by linarith
    rw [h1, ha]
    unfold combined_meals
    sorry

end meals_calculation_l186_186036


namespace correct_operation_l186_186976

variables {x y : ℝ}

theorem correct_operation : -2 * x * 3 * y = -6 * x * y :=
by
  sorry

end correct_operation_l186_186976


namespace probability_A_correct_l186_186086

-- Definitions of probabilities
variable (P_A P_B : Prop)
variable (P_AB : Prop := P_A ∧ P_B)
variable (prob_AB : ℝ := 2 / 3)
variable (prob_B_given_A : ℝ := 8 / 9)

-- Lean statement of the mathematical problem
theorem probability_A_correct :
  (P_AB → P_A ∧ P_B) →
  (prob_AB = (2 / 3)) →
  (prob_B_given_A = (2 / 3) / prob_A) →
  (∃ prob_A : ℝ, prob_A = 3 / 4) :=
by
  sorry

end probability_A_correct_l186_186086


namespace min_ab_eq_11_l186_186942

theorem min_ab_eq_11 (a b : ℕ) (h : 23 * a - 13 * b = 1) : a + b = 11 :=
sorry

end min_ab_eq_11_l186_186942


namespace number_of_footballs_is_3_l186_186112

-- Define the variables and conditions directly from the problem

-- Let F be the cost of one football and S be the cost of one soccer ball
variable (F S : ℝ)

-- Condition 1: Some footballs and 1 soccer ball cost 155 dollars
variable (number_of_footballs : ℝ)
variable (H1 : F * number_of_footballs + S = 155)

-- Condition 2: 2 footballs and 3 soccer balls cost 220 dollars
variable (H2 : 2 * F + 3 * S = 220)

-- Condition 3: The cost of one soccer ball is 50 dollars
variable (H3 : S = 50)

-- Theorem: Prove that the number of footballs in the first set is 3
theorem number_of_footballs_is_3 (H1 H2 H3 : Prop) :
  number_of_footballs = 3 := by
  sorry

end number_of_footballs_is_3_l186_186112


namespace tangent_lines_to_curve_l186_186364

-- Define the curve
def curve (x : ℝ) : ℝ := x^3

-- Define the general form of a tangent line
def tangent_line (x : ℝ) (y : ℝ) (m : ℝ) (x0 : ℝ) (y0 : ℝ) : Prop :=
  y - y0 = m * (x - x0)

-- Define the conditions
def condition1 : Prop :=
  tangent_line 1 1 3 1 1

def condition2 : Prop :=
  tangent_line 1 1 (3/4) (-1/2) ((-1/2)^3)

-- Define the equations of the tangent lines
def line1 : Prop :=
  ∀ x y : ℝ, 3 * x - y - 2 = 0

def line2 : Prop :=
  ∀ x y : ℝ, 3 * x - 4 * y + 1 = 0

-- The final theorem statement
theorem tangent_lines_to_curve :
  (condition1 → line1) ∧ (condition2 → line2) :=
  by
    sorry -- Placeholder for proof

end tangent_lines_to_curve_l186_186364


namespace acute_triangle_probability_l186_186568

noncomputable def probability_acute_triangle : ℝ := sorry

theorem acute_triangle_probability :
  probability_acute_triangle = 1 / 4 := sorry

end acute_triangle_probability_l186_186568


namespace louise_needs_eight_boxes_l186_186565

-- Define the given conditions
def red_pencils : ℕ := 20
def blue_pencils : ℕ := 2 * red_pencils
def yellow_pencils : ℕ := 40
def green_pencils : ℕ := red_pencils + blue_pencils
def pencils_per_box : ℕ := 20

-- Define the functions to calculate the required number of boxes for each color
def boxes_needed (pencils : ℕ) : ℕ := (pencils + pencils_per_box - 1) / pencils_per_box

-- Calculate the total number of boxes needed by summing the boxes for each color
def total_boxes_needed : ℕ := boxes_needed red_pencils + boxes_needed blue_pencils + boxes_needed yellow_pencils + boxes_needed green_pencils

-- The proof problem statement
theorem louise_needs_eight_boxes : total_boxes_needed = 8 :=
by
  sorry

end louise_needs_eight_boxes_l186_186565


namespace units_digit_b_l186_186513

theorem units_digit_b (a b : ℕ) (h1 : a % 10 = 9) (h2 : a * b = 34^8) : b % 10 = 4 :=
by
  sorry

end units_digit_b_l186_186513


namespace tricksters_identification_l186_186316

variable (Inhabitant : Type)
variable [inhab : Fintype Inhabitant]
variable [decEqInhab : DecidableEq Inhabitant]
variable (knight : Inhabitant → Prop)
variable (trickster : Inhabitant → Prop)

variables (n : ℕ) (q : ℕ)
variable (is_truthful : Inhabitant → Prop)

constant inhabitants_count : 65
constant tricksters_count : 2

-- Define the property that a knight always tells the truth.
axiom knight_truth (x : Inhabitant) : knight x → is_truthful x

-- Define the property that a trickster can tell the truth or lie.
axiom trickster_behavior (x : Inhabitant) : trickster x → (is_truthful x ∨ ¬ is_truthful x)

 -- Define the type of the question which can be asked to an inhabitant.
inductive Question (Inhabitant : Type) : Type
| is_knight : Inhabitant → Question

-- Define the type of the answer to the question.
inductive Answer (Inhabitant : Type) : Type
| yes : Answer
| no :  Answer

-- Define a function that simulates asking a question to an inhabitant.
constant ask : Inhabitant → Question Inhabitant → Answer Inhabitant

noncomputable def find_tricksters (inhabitants : fin inhabitants_count → Inhabitant) : (fin tricksters_count → Inhabitant) :=
sorry

theorem tricksters_identification : 
  ∃ (f : (fin inhabitants_count → Inhabitant) → (fin tricksters_count → Inhabitant)), 
    ∀ inhabitants : fin inhabitants_count → Inhabitant, 
      (∀ (q_list : list (Inhabitant × Question Inhabitant)),
        q_list.length ≤ 30 → 
        let a_list := q_list.map (λ pq, ask (pq.fst) (pq.snd)) in 
        true) ∧ 
      (∃ t1 t2, trickster (f inhabitants) t1 ∧ trickster (f inhabitants) t2) :=
sorry

end tricksters_identification_l186_186316


namespace calculate_product_N1_N2_l186_186932

theorem calculate_product_N1_N2 : 
  (∃ (N1 N2 : ℝ), 
    (∀ x : ℝ, x ≠ 2 ∧ x ≠ 3 → 
      (60 * x - 46) / (x^2 - 5 * x + 6) = N1 / (x - 2) + N2 / (x - 3)) ∧
      N1 * N2 = -1036) :=
  sorry

end calculate_product_N1_N2_l186_186932


namespace waiter_customers_l186_186831

variable (initial_customers left_customers new_customers : ℕ)

theorem waiter_customers 
  (h1 : initial_customers = 33)
  (h2 : left_customers = 31)
  (h3 : new_customers = 26) :
  (initial_customers - left_customers + new_customers = 28) := 
by
  sorry

end waiter_customers_l186_186831


namespace euler_disproven_conjecture_solution_l186_186438

theorem euler_disproven_conjecture_solution : 
  ∃ (n : ℕ), n^5 = 133^5 + 110^5 + 84^5 + 27^5 ∧ n = 144 :=
by
  use 144
  have h : 144^5 = 133^5 + 110^5 + 84^5 + 27^5 := sorry
  exact ⟨h, rfl⟩

end euler_disproven_conjecture_solution_l186_186438


namespace circle_center_coordinates_l186_186272

theorem circle_center_coordinates (h k r : ℝ) :
  (∀ x y : ℝ, (x - 2)^2 + (y + 3)^2 = 1 → (x - h)^2 + (y - k)^2 = r^2) →
  (h, k) = (2, -3) :=
by
  intro H
  sorry

end circle_center_coordinates_l186_186272


namespace evaluate_expression_l186_186047

theorem evaluate_expression : 1 + 1 / (1 + 1 / (1 + 1 / (1 + 2))) = 11 / 7 :=
by 
  sorry

end evaluate_expression_l186_186047


namespace at_most_half_map_finite_subsets_l186_186091

open Function

variable {Z : Type} [Int Z] -- Z be the integers.

variable (f : Fin 10 → Z → Z) -- f_1, f_2, ..., f_{10} : Z -> Z

-- Ensure f_i are bijections
variable (hf : ∀ (i : Fin 10), Bijective (f i))

-- For any n ∈ Z, there exists a composition of the f_i's (with possible repetitions) that maps 0 to n
variable (hcomp : ∀ n : Z, ∃ k : List (Fin 10), (k.foldr (∘) id f) 0 = n)

-- Defining S
def S : Set (Z → Z) := 
  {g | ∃ k : Fin 10 → Bool, g = (λ x => (List.range 10).foldr (λ i acc => if k i then f i acc else acc) x) }

-- To show: At most half the functions in S map a finite (non-empty) subset of Z onto itself.
theorem at_most_half_map_finite_subsets (A : Finset Z) (hA : A.Nonempty) :
  (S.filter (λ g => ∃ B : Finset Z, B ⊆ A ∧ B.Nonempty ∧ ∀ x ∈ B, g x ∈ B)).card ≤ S.card / 2 := 
sorry

end at_most_half_map_finite_subsets_l186_186091


namespace smallest_positive_period_f_max_value_f_interval_min_value_f_interval_l186_186677

noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos x * (Real.sin x - Real.cos x) + 1

theorem smallest_positive_period_f : ∃ k > 0, ∀ x, f (x + k) = f x := 
sorry

theorem max_value_f_interval : ∃ x ∈ Set.Icc (Real.pi / 8) (3 * Real.pi / 4), f x = Real.sqrt 2 :=
sorry

theorem min_value_f_interval : ∃ x ∈ Set.Icc (Real.pi / 8) (3 * Real.pi / 4), f x = -1 :=
sorry

end smallest_positive_period_f_max_value_f_interval_min_value_f_interval_l186_186677


namespace tan_alpha_eq_one_third_l186_186222

-- Define the vectors a and b
def a : ℝ × ℝ := (1, 3)
def b (α : ℝ) : ℝ × ℝ := (Real.sin α, Real.cos α)

-- Definition of parallel vectors
def parallel (u v : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, u = (k * v.1, k * v.2)

-- Proof statement: Given the conditions, prove tan(α) = 1/3
theorem tan_alpha_eq_one_third (α : ℝ) (h₁ : parallel a (b α)) : Real.tan α = 1 / 3 := 
  sorry

end tan_alpha_eq_one_third_l186_186222


namespace solve_for_x_l186_186449

theorem solve_for_x (x : ℝ) (h : 2^x + 10 = 3 * 2^x - 14) : x = Real.log 12 / Real.log 2 :=
by
  sorry

end solve_for_x_l186_186449


namespace sum_2004_impossible_sum_2005_possible_l186_186284

-- Condition Definitions
def is_valid_square (s : ℕ × ℕ × ℕ × ℕ) : Prop :=
  s = (1, 2, 3, 4) ∨ s = (1, 2, 4, 3) ∨ s = (1, 3, 2, 4) ∨ s = (1, 3, 4, 2) ∨ 
  s = (1, 4, 2, 3) ∨ s = (1, 4, 3, 2) ∨ s = (2, 1, 3, 4) ∨ s = (2, 1, 4, 3) ∨ 
  s = (2, 3, 1, 4) ∨ s = (2, 3, 4, 1) ∨ s = (2, 4, 1, 3) ∨ s = (2, 4, 3, 1) ∨ 
  s = (3, 1, 2, 4) ∨ s = (3, 1, 4, 2) ∨ s = (3, 2, 1, 4) ∨ s = (3, 2, 4, 1) ∨ 
  s = (3, 4, 1, 2) ∨ s = (3, 4, 2, 1) ∨ s = (4, 1, 2, 3) ∨ s = (4, 1, 3, 2) ∨ 
  s = (4, 2, 1, 3) ∨ s = (4, 2, 3, 1) ∨ s = (4, 3, 1, 2) ∨ s = (4, 3, 2, 1)

-- Proof Problems
theorem sum_2004_impossible (n : ℕ) (corners : ℕ → ℕ × ℕ × ℕ × ℕ) (h : ∀ i, is_valid_square (corners i)) :
  4 * 2004 ≠ n * 10 := 
sorry

theorem sum_2005_possible (h : ∃ n, ∃ corners : ℕ → ℕ × ℕ × ℕ × ℕ, (∀ i, is_valid_square (corners i)) ∧ 4 * 2005 = n * 10 + 2005) :
  true := 
sorry

end sum_2004_impossible_sum_2005_possible_l186_186284


namespace product_of_five_consecutive_integers_divisible_by_240_l186_186804

theorem product_of_five_consecutive_integers_divisible_by_240 (n : ℤ) : ∃ k : ℤ, (∏ i in range 5, n + i) = 240 * k := 
sorry

end product_of_five_consecutive_integers_divisible_by_240_l186_186804


namespace largest_5_digit_congruent_l186_186594

theorem largest_5_digit_congruent (n : ℕ) (h1 : 29 * n + 17 < 100000) : 29 * 3447 + 17 = 99982 :=
by
  -- Proof goes here
  sorry

end largest_5_digit_congruent_l186_186594


namespace sequence_polynomial_degree_l186_186894

theorem sequence_polynomial_degree
  (k : ℕ)
  (v : ℕ → ℤ)
  (u : ℕ → ℤ)
  (h_diff_poly : ∃ p : Polynomial ℤ, ∀ n, v n = Polynomial.eval (n : ℤ) p)
  (h_diff_seq : ∀ n, v n = (u (n + 1) - u n)) :
  ∃ q : Polynomial ℤ, ∀ n, u n = Polynomial.eval (n : ℤ) q := 
sorry

end sequence_polynomial_degree_l186_186894


namespace num_combinations_L_shape_l186_186310

theorem num_combinations_L_shape (n : ℕ) (k : ℕ) (grid_size : ℕ) (L_shape_blocks : ℕ) 
  (h1 : n = 6) (h2 : k = 4) (h3 : grid_size = 36) (h4 : L_shape_blocks = 4) : 
  ∃ (total_combinations : ℕ), total_combinations = 1800 := by
  sorry

end num_combinations_L_shape_l186_186310


namespace printer_x_time_l186_186103

-- Define the basic parameters given in the problem
def job_time_printer_y := 12
def job_time_printer_z := 8
def ratio := 10 / 3

-- Work rates of the printers
def work_rate_y := 1 / job_time_printer_y
def work_rate_z := 1 / job_time_printer_z

-- Combined work rate and total time for printers Y and Z
def combined_work_rate_y_z := work_rate_y + work_rate_z
def time_printers_y_z := 1 / combined_work_rate_y_z

-- Given ratio relation
def time_printer_x := ratio * time_printers_y_z

-- Mathematical statement to prove: time it takes for printer X to do the job alone
theorem printer_x_time : time_printer_x = 16 := by
  sorry

end printer_x_time_l186_186103


namespace tourists_left_l186_186989

noncomputable def tourists_remaining {initial remaining poisoned recovered : ℕ} 
  (h1 : initial = 30)
  (h2 : remaining = initial - 2)
  (h3 : poisoned = remaining / 2)
  (h4 : recovered = poisoned / 7)
  (h5 : remaining % 2 = 0) -- ensuring even division for / 2
  (h6 : poisoned % 7 = 0) -- ensuring even division for / 7
  : ℕ :=
  remaining - poisoned + recovered

theorem tourists_left 
  (initial remaining poisoned recovered : ℕ) 
  (h1 : initial = 30)
  (h2 : remaining = initial - 2)
  (h3 : poisoned = remaining / 2)
  (h4 : recovered = poisoned / 7)
  (h5 : remaining % 2 = 0) -- ensuring even division for / 2
  (h6 : poisoned % 7 = 0) -- ensuring even division for / 7
  : tourists_remaining h1 h2 h3 h4 h5 h6 = 16 :=
  by
  sorry

end tourists_left_l186_186989


namespace Chandler_saves_enough_l186_186652

theorem Chandler_saves_enough (total_cost gift_money weekly_earnings : ℕ)
  (h_cost : total_cost = 550)
  (h_gift : gift_money = 130)
  (h_weekly : weekly_earnings = 18) : ∃ x : ℕ, (130 + 18 * x) >= 550 ∧ x = 24 := 
by
  sorry

end Chandler_saves_enough_l186_186652


namespace minimal_degree_of_g_l186_186847

noncomputable def g_degree_minimal (f g h : Polynomial ℝ) (deg_f : ℕ) (deg_h : ℕ) (h_eq : (5 : ℝ) • f + (7 : ℝ) • g = h) : Prop :=
  Polynomial.degree f = deg_f ∧ Polynomial.degree h = deg_h → Polynomial.degree g = 12

theorem minimal_degree_of_g (f g h : Polynomial ℝ) (h_eq : (5 : ℝ) • f + (7 : ℝ) • g = h)
    (deg_f : Polynomial.degree f = 5) (deg_h : Polynomial.degree h = 12) :
    Polynomial.degree g = 12 := by
  sorry

end minimal_degree_of_g_l186_186847


namespace angle_bisector_ratio_l186_186540

theorem angle_bisector_ratio (A B C Q : Type) (AC CB AQ QB : ℝ) (k : ℝ) 
  (hAC : AC = 4 * k) (hCB : CB = 5 * k) (angle_bisector_theorem : AQ / QB = AC / CB) :
  AQ / QB = 4 / 5 := 
by sorry

end angle_bisector_ratio_l186_186540


namespace garden_area_increase_l186_186492

-- Definitions corresponding to the conditions
def length := 40
def width := 20
def original_perimeter := 2 * (length + width)

-- Definition of the correct answer calculation
def original_area := length * width
def side_length := original_perimeter / 4
def new_area := side_length * side_length
def area_increase := new_area - original_area

-- The statement to be proven
theorem garden_area_increase : area_increase = 100 :=
by sorry

end garden_area_increase_l186_186492


namespace part1_part2_range_of_a_l186_186388

noncomputable def f1 (x : ℝ) : ℝ := Real.sin x - Real.log (x + 1)

theorem part1 (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 1) : f1 x ≥ 0 := sorry

noncomputable def f2 (x a : ℝ) : ℝ := Real.sin x - a * Real.log (x + 1)

theorem part2 {a : ℝ} (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ Real.pi) : f2 x a ≤ 2 * Real.exp x - 2 := sorry

theorem range_of_a : {a : ℝ | ∀ x : ℝ, 0 ≤ x → x ≤ Real.pi → f2 x a ≤ 2 * Real.exp x - 2} = {a : ℝ | a ≥ -1} := sorry

end part1_part2_range_of_a_l186_186388


namespace total_time_for_phd_l186_186089

def acclimation_period : ℕ := 1 -- in years
def basics_learning_phase : ℕ := 2 -- in years
def research_factor : ℝ := 1.75 -- 75% more time on research
def research_time_without_sabbaticals_and_conferences : ℝ := basics_learning_phase * research_factor
def first_sabbatical : ℝ := 0.5 -- in years (6 months)
def second_sabbatical : ℝ := 0.25 -- in years (3 months)
def first_conference : ℝ := 0.3333 -- in years (4 months)
def second_conference : ℝ := 0.4166 -- in years (5 months)
def additional_research_time : ℝ := first_sabbatical + second_sabbatical + first_conference + second_conference
def total_research_phase_time : ℝ := research_time_without_sabbaticals_and_conferences + additional_research_time
def dissertation_factor : ℝ := 0.5 -- half as long as acclimation period
def time_spent_writing_without_conference : ℝ := dissertation_factor * acclimation_period
def dissertation_conference : ℝ := 0.25 -- in years (3 months)
def total_dissertation_writing_time : ℝ := time_spent_writing_without_conference + dissertation_conference

theorem total_time_for_phd : 
  (acclimation_period + basics_learning_phase + total_research_phase_time + total_dissertation_writing_time) = 8.75 :=
by
  sorry

end total_time_for_phd_l186_186089


namespace evaluate_expression_l186_186197

theorem evaluate_expression : (20^40) / (40^20) = 10^20 := by
  sorry

end evaluate_expression_l186_186197


namespace max_robot_weight_l186_186439

-- Definitions of the given conditions
def standard_robot_weight : ℕ := 100
def battery_weight : ℕ := 20
def min_payload : ℕ := 10
def max_payload : ℕ := 25
def min_robot_weight_extra : ℕ := 5
def min_robot_weight : ℕ := standard_robot_weight + min_robot_weight_extra

-- Definition for total minimum weight of the robot
def min_total_weight : ℕ := min_robot_weight + battery_weight + min_payload

-- Proposition for the maximum weight condition
theorem max_robot_weight :
  2 * min_total_weight = 270 :=
by
  -- Insert proof here
  sorry

end max_robot_weight_l186_186439


namespace stacked_lego_volume_l186_186373

theorem stacked_lego_volume 
  (lego_volume : ℝ)
  (rows columns layers : ℕ)
  (h1 : lego_volume = 1)
  (h2 : rows = 7)
  (h3 : columns = 5)
  (h4 : layers = 3) :
  rows * columns * layers * lego_volume = 105 :=
by
  sorry

end stacked_lego_volume_l186_186373


namespace contest_end_time_l186_186986

theorem contest_end_time (start_time : Time := ⟨15, 0⟩ ) (duration_minutes : ℕ := 450) (break_minutes : ℕ := 15)
  : Time :=
by
  sorry

end contest_end_time_l186_186986


namespace movie_revenue_multiple_correct_l186_186179

-- Definitions from the conditions
def opening_weekend_revenue : ℝ := 120 * 10^6
def company_share_fraction : ℝ := 0.60
def profit : ℝ := 192 * 10^6
def production_cost : ℝ := 60 * 10^6

-- The statement to prove
theorem movie_revenue_multiple_correct : 
  ∃ M : ℝ, (company_share_fraction * (opening_weekend_revenue * M) - production_cost = profit) ∧ M = 3.5 :=
by
  sorry

end movie_revenue_multiple_correct_l186_186179


namespace solution_set_of_inequality_minimum_value_2a_plus_b_l186_186873

noncomputable def f (x : ℝ) : ℝ := x + 1 + |3 - x|

theorem solution_set_of_inequality :
  {x : ℝ | x ≥ -1 ∧ f x ≤ 6} = {x : ℝ | -1 ≤ x ∧ x ≤ 4} :=
by
  sorry

theorem minimum_value_2a_plus_b (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 8 * a * b = a + 2 * b) :
  2 * a + b = 9 / 8 :=
by
  sorry

end solution_set_of_inequality_minimum_value_2a_plus_b_l186_186873


namespace a_n_value_l186_186520

theorem a_n_value (a : ℕ → ℕ) (S : ℕ → ℕ) (h1 : a 1 = 3) (h2 : ∀ n, S (n + 1) = 2 * S n) (h3 : S 1 = a 1)
  (h4 : ∀ n, S n = 3 * 2^(n - 1)) : a 4 = 12 :=
sorry

end a_n_value_l186_186520


namespace sqrt_31_minus_2_in_range_l186_186196

-- Defining the conditions based on the problem statements
def five_squared : ℤ := 5 * 5
def six_squared : ℤ := 6 * 6
def thirty_one : ℤ := 31

theorem sqrt_31_minus_2_in_range : 
  (5 * 5 < thirty_one) ∧ (thirty_one < 6 * 6) →
  3 < (Real.sqrt thirty_one) - 2 ∧ (Real.sqrt thirty_one) - 2 < 4 :=
by
  sorry

end sqrt_31_minus_2_in_range_l186_186196


namespace solve_for_x_l186_186234

theorem solve_for_x (x : ℝ) (h₁ : (x + 2) ≠ 0) (h₂ : (|x| - 2) / (x + 2) = 0) : x = 2 := by
  sorry

end solve_for_x_l186_186234


namespace coffee_cost_l186_186481

def ounces_per_donut : ℕ := 2
def ounces_per_pot : ℕ := 12
def cost_per_pot : ℕ := 3
def dozens_of_donuts : ℕ := 3
def donuts_per_dozen : ℕ := 12

theorem coffee_cost :
  let total_donuts := dozens_of_donuts * donuts_per_dozen
  let total_ounces := ounces_per_donut * total_donuts
  let total_pots := total_ounces / ounces_per_pot
  let total_cost := total_pots * cost_per_pot
  total_cost = 18 := by
  sorry

end coffee_cost_l186_186481


namespace jacket_spending_l186_186711

def total_spent : ℝ := 14.28
def spent_on_shorts : ℝ := 9.54
def spent_on_jacket : ℝ := 4.74

theorem jacket_spending :
  spent_on_jacket = total_spent - spent_on_shorts :=
by sorry

end jacket_spending_l186_186711


namespace tan_neg_5pi_over_4_l186_186510

theorem tan_neg_5pi_over_4 : Real.tan (-5 * Real.pi / 4) = -1 :=
by 
  sorry

end tan_neg_5pi_over_4_l186_186510


namespace slices_leftover_is_9_l186_186341

-- Conditions and definitions
def total_pizzas : ℕ := 2
def slices_per_pizza : ℕ := 12
def bob_ate : ℕ := slices_per_pizza / 2
def tom_ate : ℕ := slices_per_pizza / 3
def sally_ate : ℕ := slices_per_pizza / 6
def jerry_ate : ℕ := slices_per_pizza / 4

-- Calculate total slices eaten and left over
def total_slices_eaten : ℕ := bob_ate + tom_ate + sally_ate + jerry_ate
def total_slices_available : ℕ := total_pizzas * slices_per_pizza
def slices_leftover : ℕ := total_slices_available - total_slices_eaten

-- Theorem to prove the number of slices left over
theorem slices_leftover_is_9 : slices_leftover = 9 := by
  -- Proof: omitted, add relevant steps here
  sorry

end slices_leftover_is_9_l186_186341


namespace professors_seat_choice_count_l186_186657

theorem professors_seat_choice_count : 
    let chairs := 11 -- number of chairs
    let students := 7 -- number of students
    let professors := 4 -- number of professors
    ∀ (P: Fin professors -> Fin chairs), 
    (∀ (p : Fin professors), 1 ≤ P p ∧ P p ≤ 9) -- Each professor is between seats 2-10
    ∧ (P 0 < P 1) ∧ (P 1 < P 2) ∧ (P 2 < P 3) -- Professors must be placed with at least one seat gap
    ∧ (P 0 ≠ 1 ∧ P 3 ≠ 11) -- First and last seats are excluded
    → ∃ (ways : ℕ), ways = 840 := sorry

end professors_seat_choice_count_l186_186657


namespace find_f_2_l186_186276

noncomputable def f : ℝ → ℝ := sorry

theorem find_f_2 (h1 : ∀ x1 x2 : ℝ, f (x1 * x2) = f x1 + f x2) (h2 : f 8 = 3) : f 2 = 1 :=
by
  sorry

end find_f_2_l186_186276


namespace find_n_l186_186370

theorem find_n (n : ℕ) (h1 : 0 ≤ n) (h2 : n ≤ 14) : n ≡ 14567 [MOD 15] → n = 2 := 
by
  sorry

end find_n_l186_186370


namespace new_average_l186_186625

theorem new_average (n : ℕ) (a : ℕ) (multiplier : ℕ) (average : ℕ) :
  (n = 35) →
  (a = 25) →
  (multiplier = 5) →
  (average = 125) →
  ((n * a * multiplier) / n = average) :=
by
  intros hn ha hm havg
  rw [hn, ha, hm]
  norm_num
  sorry

end new_average_l186_186625


namespace find_nat_nums_satisfying_eq_l186_186818

theorem find_nat_nums_satisfying_eq (m n : ℕ) (h_m : m = 3) (h_n : n = 3) : 2 ^ n + 1 = m ^ 2 :=
by
  rw [h_m, h_n]
  sorry

end find_nat_nums_satisfying_eq_l186_186818


namespace percent_less_than_m_plus_d_l186_186298

-- Define the given conditions
variables (m d : ℝ) (distribution : ℝ → ℝ)

-- Assume the distribution is symmetric about the mean m
axiom symmetric_distribution :
  ∀ x, distribution (m + x) = distribution (m - x)

-- 84 percent of the distribution lies within one standard deviation d of the mean
axiom within_one_sd :
  ∫ x in -d..d, distribution (m + x) = 0.84

-- The goal is to prove that 42 percent of the distribution is less than m + d
theorem percent_less_than_m_plus_d : 
  ( ∫ x in -d..0, distribution (m + x) ) = 0.42 :=
by 
  sorry

end percent_less_than_m_plus_d_l186_186298


namespace vincent_books_l186_186014

theorem vincent_books (x : ℕ) (h1 : 10 + 3 + x = 13 + x)
                      (h2 : 16 * (13 + x) = 224) : x = 1 :=
by sorry

end vincent_books_l186_186014


namespace correct_operation_l186_186821

variables (a b : ℝ)

theorem correct_operation : (3 * a + b) * (3 * a - b) = 9 * a^2 - b^2 :=
by sorry

end correct_operation_l186_186821


namespace find_a_l186_186060

-- Given Conditions
def is_hyperbola (a : ℝ) : Prop := ∀ x y : ℝ, (x^2 / a) - (y^2 / 2) = 1
def is_asymptote (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = 2 * x

-- Question
theorem find_a (a : ℝ) (f : ℝ → ℝ) (hyp : is_hyperbola a) (asym : is_asymptote f) : a = 1 / 2 :=
sorry

end find_a_l186_186060


namespace necessary_and_sufficient_condition_l186_186407

-- Definitions for sides opposite angles A, B, and C in a triangle.
variables {A B C : Real} {a b c : Real}

-- Condition p: sides a, b related to angles A, B via cosine
def condition_p (a b : Real) (A B : Real) : Prop := a / Real.cos A = b / Real.cos B

-- Condition q: sides a and b are equal
def condition_q (a b : Real) : Prop := a = b

theorem necessary_and_sufficient_condition (h1 : condition_p a b A B) : condition_q a b ↔ condition_p a b A B :=
by
  sorry

end necessary_and_sufficient_condition_l186_186407


namespace largest_divisor_of_product_of_five_consecutive_integers_l186_186760

theorem largest_divisor_of_product_of_five_consecutive_integers :
  ∃ n, (∀ k : ℤ, n ∣ (k * (k + 1) * (k + 2) * (k + 3) * (k + 4))) ∧ n = 60 :=
by
  sorry

end largest_divisor_of_product_of_five_consecutive_integers_l186_186760


namespace find_ratio_l186_186416

theorem find_ratio (P Q : ℤ)
  (h : ∀ x : ℝ, x ≠ -6 ∧ x ≠ 0 ∧ x ≠ 5 → 
  P / (x + 6) + Q / (x * (x - 5)) = (x^2 - x + 15) / (x^3 + x^2 - 30 * x)) :
  Q / P = 5 / 6 := sorry

end find_ratio_l186_186416


namespace canadian_ratio_correct_l186_186727

-- The total number of scientists
def total_scientists : ℕ := 70

-- Half of the scientists are from Europe
def european_scientists : ℕ := total_scientists / 2

-- The number of scientists from the USA
def usa_scientists : ℕ := 21

-- The number of Canadian scientists
def canadian_scientists : ℕ := total_scientists - european_scientists - usa_scientists

-- The ratio of the number of Canadian scientists to the total number of scientists
def canadian_ratio : ℚ := canadian_scientists / total_scientists

-- Prove that the ratio is 1:5
theorem canadian_ratio_correct : canadian_ratio = 1 / 5 :=
by
  sorry

end canadian_ratio_correct_l186_186727


namespace wednesday_more_than_tuesday_l186_186097

noncomputable def monday_minutes : ℕ := 450

noncomputable def tuesday_minutes : ℕ := monday_minutes / 2

noncomputable def wednesday_minutes : ℕ := 300

theorem wednesday_more_than_tuesday : wednesday_minutes - tuesday_minutes = 75 :=
by
  sorry

end wednesday_more_than_tuesday_l186_186097


namespace boxes_needed_l186_186562

-- Define the given conditions

def red_pencils : ℕ := 20
def blue_pencils : ℕ := 2 * red_pencils
def yellow_pencils : ℕ := 40
def green_pencils : ℕ := red_pencils + blue_pencils
def total_pencils : ℕ := red_pencils + blue_pencils + green_pencils + yellow_pencils
def pencils_per_box : ℕ := 20

-- Lean theorem statement to prove the number of boxes needed is 8

theorem boxes_needed : total_pencils / pencils_per_box = 8 :=
by
  -- This is where the proof would go
  sorry

end boxes_needed_l186_186562


namespace black_to_white_area_ratio_l186_186494

noncomputable def radius1 : ℝ := 2
noncomputable def radius2 : ℝ := 4
noncomputable def radius3 : ℝ := 6
noncomputable def radius4 : ℝ := 8
noncomputable def radius5 : ℝ := 10

noncomputable def area (r : ℝ) : ℝ := Real.pi * r^2

noncomputable def black_area : ℝ :=
  area radius1 + (area radius3 - area radius2) + (area radius5 - area radius4)

noncomputable def white_area : ℝ :=
  (area radius2 - area radius1) + (area radius4 - area radius3)

theorem black_to_white_area_ratio :
  black_area / white_area = 3 / 2 := by
  sorry

end black_to_white_area_ratio_l186_186494


namespace product_of_five_consecutive_divisible_by_30_l186_186777

theorem product_of_five_consecutive_divisible_by_30 :
  ∀ n : ℤ, 30 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  sorry

end product_of_five_consecutive_divisible_by_30_l186_186777


namespace find_initial_strawberries_l186_186102

-- Define the number of strawberries after picking 35 more to be 63
def strawberries_after_picking := 63

-- Define the number of strawberries picked
def strawberries_picked := 35

-- Define the initial number of strawberries
def initial_strawberries := 28

-- State the theorem
theorem find_initial_strawberries (x : ℕ) (h : x + strawberries_picked = strawberries_after_picking) : x = initial_strawberries :=
by
  -- Proof omitted
  sorry

end find_initial_strawberries_l186_186102


namespace scalene_triangle_geometric_progression_l186_186583

theorem scalene_triangle_geometric_progression :
  ∀ (q : ℝ), q ≠ 0 → 
  (∀ b : ℝ, b > 0 → b + q * b > q^2 * b ∧ q * b + q^2 * b > b ∧ b + q^2 * b > q * b) → 
  ¬((0.5 < q ∧ q < 1.7) ∨ q = 2.0) → false :=
by
  intros q hq_ne_zero hq hq_interval
  sorry

end scalene_triangle_geometric_progression_l186_186583


namespace candy_last_days_l186_186374

def pieces_from_neighbors : ℝ := 11.0
def pieces_from_sister : ℝ := 5.0
def pieces_per_day : ℝ := 8.0
def total_pieces : ℝ := pieces_from_neighbors + pieces_from_sister

theorem candy_last_days : total_pieces / pieces_per_day = 2 := by
    sorry

end candy_last_days_l186_186374


namespace dice_square_factor_probability_l186_186883

theorem dice_square_factor_probability :
  let dice_faces := {1, 2, 3, 4, 5, 6}
  in let no_square_factor_probability := ((4:ℚ) / 6) ^ 6
  in let square_factor_probability := 1 - no_square_factor_probability
  in square_factor_probability = 665 / 729 :=
by
  let dice_faces := {1, 2, 3, 4, 5, 6}
  let no_square_factor_probability := ((4:ℚ) / 6) ^ 6
  let square_factor_probability := 1 - no_square_factor_probability
  have : square_factor_probability = 665 / 729 := sorry
  exact this

end dice_square_factor_probability_l186_186883


namespace person_before_you_taller_than_you_l186_186696

-- Define the persons involved in the problem.
variable (Person : Type)
variable (Taller : Person → Person → Prop)
variable (P Q You : Person)

-- The conditions given in the problem.
axiom standing_queue : Taller P Q
axiom queue_structure : You = Q

-- The question we need to prove, which is the correct answer to the problem.
theorem person_before_you_taller_than_you : Taller P You :=
by
  sorry

end person_before_you_taller_than_you_l186_186696


namespace correct_operation_l186_186975

theorem correct_operation (x y a b : ℝ) :
  (-2 * x) * (3 * y) = -6 * x * y :=
by
  sorry

end correct_operation_l186_186975


namespace caleb_grandfather_age_l186_186500

theorem caleb_grandfather_age :
  let yellow_candles := 27
  let red_candles := 14
  let blue_candles := 38
  yellow_candles + red_candles + blue_candles = 79 :=
by
  sorry

end caleb_grandfather_age_l186_186500


namespace total_markup_l186_186124

theorem total_markup (p : ℝ) (o : ℝ) (n : ℝ) (m : ℝ) : 
  p = 48 → o = 0.35 → n = 18 → m = o * p + n → m = 34.8 :=
by
  intro hp ho hn hm
  sorry

end total_markup_l186_186124


namespace largest_divisor_of_consecutive_five_l186_186790

theorem largest_divisor_of_consecutive_five (n : ℤ) : 
  (∃ d, ∀ i ∈ {n, n + 1, n + 2, n + 3, n + 4}, d ∣ (∏ x in {n, n + 1, n + 2, n + 3, n + 4}, x)) → 
  d = 60 
:= sorry

end largest_divisor_of_consecutive_five_l186_186790


namespace sum_fractions_l186_186655

theorem sum_fractions:
  (Finset.range 16).sum (λ k => (k + 1) / 7) = 136 / 7 := by
  sorry

end sum_fractions_l186_186655


namespace midpoint_chord_hyperbola_l186_186703

theorem midpoint_chord_hyperbola (a b : ℝ) : 
  (∀ (x y : ℝ), (x^2 / a^2 + y^2 / b^2 = 1) → (∃ (mx my : ℝ), (mx / a^2 + my / b^2 = 0) ∧ (mx = x / 2) ∧ (my = y / 2))) →
  ∀ (x y : ℝ), (x^2 / a^2 - y^2 / b^2 = 1) →
  ∃ (mx my : ℝ), (mx / a^2 - my / b^2 = 0) ∧ (mx = x / 2) ∧ (my = y / 2) := 
sorry

end midpoint_chord_hyperbola_l186_186703


namespace find_segment_XY_length_l186_186139

theorem find_segment_XY_length (A B C D X Y : Type) 
  [DecidableEq A] [DecidableEq B] [DecidableEq C] [DecidableEq D] [DecidableEq X] [DecidableEq Y]
  (line_l : Type) (BX : ℝ) (DY : ℝ) (AB : ℝ) (BC : ℝ) (l : line_l)
  (hBX : BX = 4) (hDY : DY = 10) (hBC : BC = 2 * AB) :
  XY = 13 :=
  sorry

end find_segment_XY_length_l186_186139


namespace exists_polynomial_p_l186_186573

theorem exists_polynomial_p (x : ℝ) (h : x ∈ Set.Icc (1 / 10 : ℝ) (9 / 10 : ℝ)) :
  ∃ (P : ℝ → ℝ), (∀ (k : ℤ), P k = P k) ∧ (∀ (x : ℝ), x ∈ Set.Icc (1 / 10 : ℝ) (9 / 10 : ℝ) → 
  abs (P x - 1 / 2) < 1 / 1000) :=
by
  sorry

end exists_polynomial_p_l186_186573


namespace max_ratio_of_two_digit_numbers_with_mean_55_l186_186418

theorem max_ratio_of_two_digit_numbers_with_mean_55 (x y : ℕ) (h1 : 10 ≤ x) (h2 : x ≤ 99) (h3 : 10 ≤ y) (h4 : y ≤ 99) (h5 : (x + y) / 2 = 55) : x / y ≤ 9 :=
sorry

end max_ratio_of_two_digit_numbers_with_mean_55_l186_186418


namespace sin_690_eq_neg_half_l186_186353

theorem sin_690_eq_neg_half :
  Real.sin (690 * Real.pi / 180) = -1 / 2 :=
by {
  sorry
}

end sin_690_eq_neg_half_l186_186353


namespace speed_of_river_l186_186639

theorem speed_of_river :
  ∃ v : ℝ, 
    (∀ d : ℝ, (2 * d = 9.856) → 
              (d = 4.928) ∧ 
              (1 = (d / (10 - v) + d / (10 + v)))) 
    → v = 1.2 :=
sorry

end speed_of_river_l186_186639


namespace journey_ratio_proof_l186_186643

def journey_ratio (x y : ℝ) : Prop :=
  (x + y = 448) ∧ (x / 21 + y / 24 = 20) → (x / y = 1)

theorem journey_ratio_proof : ∃ x y : ℝ, journey_ratio x y :=
by
  sorry

end journey_ratio_proof_l186_186643


namespace student_sums_attempted_l186_186311

theorem student_sums_attempted (sums_right sums_wrong : ℕ) (h1 : sums_wrong = 2 * sums_right) (h2 : sums_right = 16) :
  sums_right + sums_wrong = 48 :=
by
  sorry

end student_sums_attempted_l186_186311


namespace teacher_age_l186_186021

theorem teacher_age (avg_age_students : ℕ) (num_students : ℕ) (avg_age_total : ℕ) (num_total : ℕ) (h1 : avg_age_students = 21) (h2 : num_students = 20) (h3 : avg_age_total = 22) (h4 : num_total = 21) :
  let total_age_students := avg_age_students * num_students
  let total_age_class := avg_age_total * num_total
  let teacher_age := total_age_class - total_age_students
  teacher_age = 42 :=
by
  sorry

end teacher_age_l186_186021


namespace largest_divisor_of_five_consecutive_integers_l186_186771

theorem largest_divisor_of_five_consecutive_integers:
  ∀ (n : ℤ), 
    ∃ (a b c : ℤ), 
      (n = 5 * a ∨ n = 5 * a + 1 ∨ n = 5 * a + 2 ∨ n = 5 * a + 3 ∨ n = 5 * a + 4) ∧ 
      (n = 3 * b ∨ n = 3 * b + 1 ∨ n = 3 * b + 2) ∧ 
      (n = 4 * c ∨ n = 4 * c + 1 ∨ n = 4 * c + 2 ∨ n = 4 * c + 3)
      ⊢ 60 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  -- Sorry placeholder for the proof (not needed in the task)
  sorry

end largest_divisor_of_five_consecutive_integers_l186_186771


namespace distinct_positive_integers_count_l186_186226

-- Define the digits' ranges
def digit (n : ℤ) : Prop := 0 ≤ n ∧ n ≤ 9
def nonzero_digit (n : ℤ) : Prop := 1 ≤ n ∧ n ≤ 9

-- Define the 4-digit numbers ABCD and DCBA
def ABCD (A B C D : ℤ) := 1000 * A + 100 * B + 10 * C + D
def DCBA (A B C D : ℤ) := 1000 * D + 100 * C + 10 * B + A

-- Define the difference
def difference (A B C D : ℤ) := ABCD A B C D - DCBA A B C D

-- The theorem to be proven
theorem distinct_positive_integers_count :
  ∃ n : ℤ, n = 161 ∧
  ∀ A B C D : ℤ,
  nonzero_digit A → nonzero_digit D → digit B → digit C → 
  0 < difference A B C D → (∃! x : ℤ, x = difference A B C D) :=
sorry

end distinct_positive_integers_count_l186_186226


namespace find_x2_l186_186471

theorem find_x2 (x1 x2 x3 : ℝ) (h1 : x1 + x2 = 14) (h2 : x1 + x3 = 17) (h3 : x2 + x3 = 33) : x2 = 15 :=
by
  sorry

end find_x2_l186_186471


namespace initial_population_of_first_village_l186_186999

theorem initial_population_of_first_village (P : ℕ) :
  (P - 1200 * 18) = (42000 + 800 * 18) → P = 78000 :=
by
  sorry

end initial_population_of_first_village_l186_186999


namespace find_smallest_k_l186_186051

variable (k : ℕ)

theorem find_smallest_k :
  (∀ a : ℝ, 0 ≤ a ∧ a ≤ 1 → (∀ n : ℕ, n > 0 → a^k * (1-a)^n < 1 / (n+1)^3)) ↔ k = 4 :=
sorry

end find_smallest_k_l186_186051


namespace equilateral_triangle_roots_l186_186723

noncomputable def omega : ℂ := complex.exp (2 * complex.I * real.pi / 3)

theorem equilateral_triangle_roots
  (z1 z2 p q : ℂ)
  (h_roots : z1^2 + p * z1 + q = 0)
  (h_eq_triangle : ∃ w : ℂ, w ≠ 0 ∧ z2 = w * z1 ∧ w^3 = 1 ∧ w ≠ 1) :
  (p * p / q = 1) :=
sorry

end equilateral_triangle_roots_l186_186723


namespace sports_minutes_in_newscast_l186_186455

-- Definitions based on the conditions
def total_newscast_minutes : ℕ := 30
def national_news_minutes : ℕ := 12
def international_news_minutes : ℕ := 5
def weather_forecasts_minutes : ℕ := 2
def advertising_minutes : ℕ := 6

-- The problem statement
theorem sports_minutes_in_newscast (t : ℕ) (n : ℕ) (i : ℕ) (w : ℕ) (a : ℕ) :
  t = 30 → n = 12 → i = 5 → w = 2 → a = 6 → t - n - i - w - a = 5 := 
by sorry

end sports_minutes_in_newscast_l186_186455


namespace largest_divisor_of_consecutive_five_l186_186792

theorem largest_divisor_of_consecutive_five (n : ℤ) : 
  (∃ d, ∀ i ∈ {n, n + 1, n + 2, n + 3, n + 4}, d ∣ (∏ x in {n, n + 1, n + 2, n + 3, n + 4}, x)) → 
  d = 60 
:= sorry

end largest_divisor_of_consecutive_five_l186_186792


namespace find_tricksters_l186_186328

structure Inhabitant :=
  (isKnight : Prop)

constants (inhabitants : Fin 65 → Inhabitant)
          (tricksters : Fin 2 → Fin 65)

axiom two_tricksters_unique :
  ∃! (t₁ t₂ : Fin 65), t₁ ≠ t₂ ∧ t₁ ∈ (tricksters : Set (Fin 65)) ∧ t₂ ∈ (tricksters : Set (Fin 65))

axiom valid_question :
  ∀ (a : Fin 65) (group : Set (Fin 65)), (inhabitants a).isKnight → 
  (∀ i ∈ group, (inhabitants i).isKnight) ↔ 
  (knight a).isKnight

theorem find_tricksters :
  ∃ (q : ℕ), q ≤ 16 ∧
  ∃ (t₁ t₂ : Fin 65), t₁ ≠ t₂ ∧ t₁ ∈ (tricksters : Set (Fin 65)) ∧ t₂ ∈ (tricksters : Set (Fin 65)) :=
sorry

end find_tricksters_l186_186328


namespace min_wins_required_l186_186162

theorem min_wins_required 
  (total_matches initial_matches remaining_matches : ℕ)
  (points_for_win points_for_draw points_for_defeat current_points target_points : ℕ)
  (matches_played_points : ℕ)
  (h_total : total_matches = 20)
  (h_initial : initial_matches = 5)
  (h_remaining : remaining_matches = total_matches - initial_matches)
  (h_win_points : points_for_win = 3)
  (h_draw_points : points_for_draw = 1)
  (h_defeat_points : points_for_defeat = 0)
  (h_current_points : current_points = 8)
  (h_target_points : target_points = 40)
  (h_matches_played_points : matches_played_points = current_points)
  :
  (∃ min_wins : ℕ, min_wins * points_for_win + (remaining_matches - min_wins) * points_for_defeat >= target_points - matches_played_points ∧ min_wins ≤ remaining_matches) ∧
  (∀ other_wins : ℕ, other_wins < min_wins → (other_wins * points_for_win + (remaining_matches - other_wins) * points_for_defeat < target_points - matches_played_points)) :=
sorry

end min_wins_required_l186_186162


namespace negated_proposition_false_l186_186963

theorem negated_proposition_false : ¬ ∀ x : ℝ, 2^x + x^2 > 1 :=
by 
sorry

end negated_proposition_false_l186_186963


namespace find_b_and_sinA_find_sin_2A_plus_pi_over_4_l186_186699

variable (A B C : ℝ)
variable (a b c : ℝ)
variable (sinB : ℝ)

-- Conditions
def triangle_conditions :=
  (a > b) ∧
  (a = 5) ∧
  (c = 6) ∧
  (sinB = 3 / 5)

-- Question 1: Prove b = sqrt 13 and sin A = (3 * sqrt 13) / 13
theorem find_b_and_sinA (h : triangle_conditions a b c sinB) :
  b = Real.sqrt 13 ∧
  ∃ sinA : ℝ, sinA = (3 * Real.sqrt 13) / 13 :=
  sorry

-- Question 2: Prove sin (2A + π/4) = 7 * sqrt 2 / 26
theorem find_sin_2A_plus_pi_over_4 (h : triangle_conditions a b c sinB)
  (hb : b = Real.sqrt 13)
  (sinA : ℝ)
  (h_sinA : sinA = (3 * Real.sqrt 13) / 13) :
  ∃ sin2Aπ4 : ℝ, sin2Aπ4 = (7 * Real.sqrt 2) / 26 :=
  sorry

end find_b_and_sinA_find_sin_2A_plus_pi_over_4_l186_186699


namespace chord_length_3pi_4_chord_bisected_by_P0_l186_186549

open Real

-- Define conditions and the problem.
def Circle := {p : ℝ × ℝ // p.1^2 + p.2^2 = 8}
def P0 : ℝ × ℝ := (-1, 2)

-- Proving the first part (1)
theorem chord_length_3pi_4 (α : ℝ) (hα : α = 3 * π / 4) (A B : ℝ × ℝ)
  (h1 : (A.1^2 + A.2^2 = 8) ∧ (B.1^2 + B.2^2 = 8))
  (h2 : (A.1 + B.1) / 2 = -1) (h3 : (A.2 + B.2) / 2 = 2) :
  dist A B = sqrt 30 := sorry

-- Proving the second part (2)
theorem chord_bisected_by_P0 (A B : ℝ × ℝ)
  (h1 : (A.1^2 + A.2^2 = 8) ∧ (B.1^2 + B.2^2 = 8))
  (h2 : (A.1 + B.1) / 2 = -1) (h3 : (A.2 + B.2) / 2 = 2) :
  ∃ k : ℝ, (B.2 - A.2) = k * (B.1 - A.1) ∧ k = 1 / 2 ∧
  (k * (x - (-1))) = y - 2 := sorry

end chord_length_3pi_4_chord_bisected_by_P0_l186_186549


namespace asymptotes_of_hyperbola_l186_186119

theorem asymptotes_of_hyperbola 
  (x y : ℝ)
  (h : x^2 / 4 - y^2 / 36 = 1) : 
  (y = 3 * x) ∨ (y = -3 * x) :=
sorry

end asymptotes_of_hyperbola_l186_186119


namespace solve_for_y_l186_186391

theorem solve_for_y (x y : ℝ) (h : 4 * x - y = 3) : y = 4 * x - 3 :=
by sorry

end solve_for_y_l186_186391


namespace cara_bread_dinner_amount_240_l186_186206

def conditions (B L D : ℕ) : Prop :=
  8 * L = D ∧ 6 * B = D ∧ B + L + D = 310

theorem cara_bread_dinner_amount_240 :
  ∃ (B L D : ℕ), conditions B L D ∧ D = 240 :=
by
  sorry

end cara_bread_dinner_amount_240_l186_186206


namespace Alec_goal_ratio_l186_186314

theorem Alec_goal_ratio (total_students half_votes thinking_votes more_needed fifth_votes : ℕ)
  (h_class : total_students = 60)
  (h_half : half_votes = total_students / 2)
  (remaining_students : ℕ := total_students - half_votes)
  (h_thinking : thinking_votes = 5)
  (h_fifth : fifth_votes = (remaining_students - thinking_votes) / 5)
  (h_current_votes : half_votes + thinking_votes + fifth_votes = 40)
  (h_needed : more_needed = 5)
  :
  (half_votes + thinking_votes + fifth_votes + more_needed) / total_students = 3 / 4 :=
by sorry

end Alec_goal_ratio_l186_186314


namespace john_days_to_lose_weight_l186_186413

noncomputable def john_calories_intake : ℕ := 1800
noncomputable def john_calories_burned : ℕ := 2300
noncomputable def calories_to_lose_1_pound : ℕ := 4000
noncomputable def pounds_to_lose : ℕ := 10

theorem john_days_to_lose_weight :
  (john_calories_burned - john_calories_intake) * (pounds_to_lose * calories_to_lose_1_pound / (john_calories_burned - john_calories_intake)) = 80 :=
by
  sorry

end john_days_to_lose_weight_l186_186413


namespace arrangement_count_l186_186402

def natSet := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

def evens := {2, 4, 6, 8, 10}

def odds := {1, 3, 5, 7, 9}

def is_odd_sum (col : Finset ℕ) : Prop :=
  col.sum id % 2 = 1

def valid_arrangement (table : Fin 2 → Fin 5 → ℕ) : Prop :=
  (∀ j, is_odd_sum (finset.univ.image (λ i, table i j))) ∧
  ∀ i j, table i j ∈ natSet

theorem arrangement_count :
  ∃ (n : ℕ), n = 2^5 * (5!)^2 ∧
  ∃ (tables : Finset (Fin 2 → Fin 5 → ℕ)), 
  tables.card = n ∧
  ∀ table ∈ tables, valid_arrangement table :=
sorry

end arrangement_count_l186_186402


namespace solve_inequality_l186_186111

theorem solve_inequality :
  { x : ℝ | x ≠ 1 ∧ x ≠ 3 ∧ x ≠ 5 ∧ x ≠ 7 ∧ 
    (2 / (x - 1) - 3 / (x - 3) + 5 / (x - 5) - 2 / (x - 7) < 1 / 15) } = 
  { x : ℝ | (x < -8) ∨ (-7 < x ∧ x < -1) ∨ (1 < x ∧ x < 3) ∨ (5 < x ∧ x < 7) ∨ (x > 8) } := sorry

end solve_inequality_l186_186111


namespace expr_value_l186_186830

theorem expr_value : 2 ^ (1 + 2 + 3) - (2 ^ 1 + 2 ^ 2 + 2 ^ 3) = 50 :=
by
  sorry

end expr_value_l186_186830


namespace max_consecutive_sum_2020_l186_186647

theorem max_consecutive_sum_2020 (k n : ℕ) (h₁ : 4040 % k = 0) (h₂ : k ≤ 100) 
    (h₃ : 2 * 2020 = k * (2 * n + k - 1)) : k ≤ 40 :=
sorry

end max_consecutive_sum_2020_l186_186647


namespace sum_three_consecutive_odd_integers_l186_186610

theorem sum_three_consecutive_odd_integers (n : ℤ) 
  (h : n + (n + 4) = 150) : n + (n + 2) + (n + 4) = 225 :=
by
  sorry

end sum_three_consecutive_odd_integers_l186_186610


namespace install_time_per_window_l186_186306

/-- A new building needed 14 windows. The builder had already installed 8 windows.
    It will take the builder 48 hours to install the rest of the windows. -/
theorem install_time_per_window (total_windows installed_windows remaining_install_time : ℕ)
  (h_total : total_windows = 14)
  (h_installed : installed_windows = 8)
  (h_remaining_time : remaining_install_time = 48) :
  (remaining_install_time / (total_windows - installed_windows)) = 8 :=
by
  -- Insert usual proof steps here
  sorry

end install_time_per_window_l186_186306


namespace total_eggs_found_l186_186232

def eggs_from_club_house : ℕ := 40
def eggs_from_park : ℕ := 25
def eggs_from_town_hall : ℕ := 15

theorem total_eggs_found : eggs_from_club_house + eggs_from_park + eggs_from_town_hall = 80 := by
  -- Proof of this theorem
  sorry

end total_eggs_found_l186_186232


namespace minimize_distance_school_l186_186011

-- Define the coordinates for the towns X, Y, and Z
def X_coord : ℕ × ℕ := (0, 0)
def Y_coord : ℕ × ℕ := (200, 0)
def Z_coord : ℕ × ℕ := (0, 300)

-- Define the population of the towns
def X_population : ℕ := 100
def Y_population : ℕ := 200
def Z_population : ℕ := 300

theorem minimize_distance_school : ∃ (x y : ℕ), x + y = 300 := by
  -- This should follow from the problem setup and conditions.
  sorry

end minimize_distance_school_l186_186011


namespace smallest_x_satisfying_abs_eq_l186_186194

theorem smallest_x_satisfying_abs_eq (x : ℝ) 
  (h : |2 * x^2 + 3 * x - 1| = 33) : 
  x = (-3 - Real.sqrt 281) / 4 := 
sorry

end smallest_x_satisfying_abs_eq_l186_186194


namespace employee_selection_l186_186299

theorem employee_selection
  (total_employees : ℕ)
  (under_35 : ℕ)
  (between_35_and_49 : ℕ)
  (over_50 : ℕ)
  (selected_employees : ℕ) :
  total_employees = 500 →
  under_35 = 125 →
  between_35_and_49 = 280 →
  over_50 = 95 →
  selected_employees = 100 →
  (under_35 * selected_employees / total_employees = 25) ∧
  (between_35_and_49 * selected_employees / total_employees = 56) ∧
  (over_50 * selected_employees / total_employees = 19) := by
  intros h1 h2 h3 h4 h5
  sorry

end employee_selection_l186_186299


namespace correct_ranking_l186_186900

-- Definitions for the colleagues
structure Colleague :=
  (name : String)
  (seniority : ℕ)

-- Colleagues: Julia, Kevin, Lana
def Julia := Colleague.mk "Julia" 1
def Kevin := Colleague.mk "Kevin" 0
def Lana := Colleague.mk "Lana" 2

-- Statements definitions
def Statement_I (c1 c2 c3 : Colleague) := c2.seniority < c1.seniority ∧ c1.seniority < c3.seniority 
def Statement_II (c1 c2 c3 : Colleague) := c1.seniority > c3.seniority
def Statement_III (c1 c2 c3 : Colleague) := c1.seniority ≠ c1.seniority

-- Exactly one of the statements is true
def Exactly_One_True (s1 s2 s3 : Prop) := (s1 ∨ s2 ∨ s3) ∧ ¬(s1 ∧ s2 ∨ s1 ∧ s3 ∨ s2 ∧ s3) ∧ ¬(s1 ∧ s2 ∧ s3)

-- The theorem to be proved
theorem correct_ranking :
  Exactly_One_True (Statement_I Kevin Lana Julia) (Statement_II Kevin Lana Julia) (Statement_III Kevin Lana Julia) →
  (Kevin.seniority < Lana.seniority ∧ Lana.seniority < Julia.seniority) := 
  by  sorry

end correct_ranking_l186_186900


namespace mustard_at_first_table_l186_186038

theorem mustard_at_first_table (M : ℝ) :
  (M + 0.25 + 0.38 = 0.88) → M = 0.25 :=
by
  intro h
  sorry

end mustard_at_first_table_l186_186038


namespace problem_1_problem_2_l186_186385

def f (x a : ℝ) := |x + a| + |x + 3|
def g (x : ℝ) := |x - 1| + 2

theorem problem_1 : ∀ x : ℝ, |g x| < 3 ↔ 0 < x ∧ x < 2 := 
by
  sorry

theorem problem_2 : (∀ x1 : ℝ, ∃ x2 : ℝ, f x1 a = g x2) ↔ a ≥ 5 ∨ a ≤ 1 := 
by
  sorry

end problem_1_problem_2_l186_186385


namespace dice_probability_l186_186843

theorem dice_probability :
  let P_even := (1 / 2 : ℝ) in
  let P_odd := (1 / 2 : ℝ) in
  let num_ways := Nat.choose 6 3 in
  let total_prob := num_ways * (P_even ^ 3) * (P_odd ^ 3) in
  total_prob = (5 / 16 : ℝ) :=
by
  have P_even := (1 / 2 : ℝ)
  have P_odd := (1 / 2 : ℝ)
  have num_ways := Nat.choose 6 3
  have total_prob := num_ways * (P_even ^ 3) * (P_odd ^ 3)
  sorry

end dice_probability_l186_186843


namespace transformed_line_equation_l186_186679

theorem transformed_line_equation {A B C x₀ y₀ : ℝ} 
    (h₀ : ¬(A = 0 ∧ B = 0)) 
    (h₁ : A * x₀ + B * y₀ + C = 0) : 
    ∀ {x y : ℝ}, A * x + B * y + C = 0 ↔ A * (x - x₀) + B * (y - y₀) = 0 :=
by
    sorry

end transformed_line_equation_l186_186679


namespace repave_today_l186_186024

theorem repave_today (total_repaved : ℕ) (repaved_before_today : ℕ) (repaved_today : ℕ) :
  total_repaved = 4938 → repaved_before_today = 4133 → repaved_today = total_repaved - repaved_before_today → repaved_today = 805 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3


end repave_today_l186_186024


namespace max_correct_answers_l186_186401

theorem max_correct_answers (a b c : ℕ) (h1 : a + b + c = 80) (h2 : 5 * a - 2 * c = 150) : a ≤ 44 :=
by
  sorry

end max_correct_answers_l186_186401


namespace complement_union_eq_complement_l186_186909

open Set

section ComplementUnion

variable (k : ℤ)

def SetA : Set ℤ := {x | ∃ k : ℤ, x = 3 * k + 1}
def SetB : Set ℤ := {x | ∃ k : ℤ, x = 3 * k + 2}
def UniversalSet : Set ℤ := univ
def ComplementUnion : Set ℤ := {x | ∃ k : ℤ, x = 3 * k}

theorem complement_union_eq_complement :
  UniversalSet \ (SetA ∪ SetB) = ComplementUnion :=
by
  sorry

end ComplementUnion

end complement_union_eq_complement_l186_186909


namespace main_problem_proof_l186_186466

def main_problem : Prop :=
  (1 : ℤ)^10 + (-1 : ℤ)^8 + (-1 : ℤ)^7 + (1 : ℤ)^5 = 2

theorem main_problem_proof : main_problem :=
by {
  sorry
}

end main_problem_proof_l186_186466


namespace value_of_x_plus_y_l186_186223

theorem value_of_x_plus_y (x y : ℝ) (h : |x - 1| + (y - 2)^2 = 0) : x + y = 3 := by
  sorry

end value_of_x_plus_y_l186_186223


namespace emily_small_gardens_l186_186163

theorem emily_small_gardens 
  (total_seeds : Nat)
  (big_garden_seeds : Nat)
  (small_garden_seeds : Nat)
  (remaining_seeds : total_seeds = big_garden_seeds + (small_garden_seeds * 3)) :
  3 = (total_seeds - big_garden_seeds) / small_garden_seeds :=
by
  have h1 : total_seeds = 42 := by sorry
  have h2 : big_garden_seeds = 36 := by sorry
  have h3 : small_garden_seeds = 2 := by sorry
  have h4 : 6 = total_seeds - big_garden_seeds := by sorry
  have h5 : 3 = 6 / small_garden_seeds := by sorry
  sorry

end emily_small_gardens_l186_186163


namespace permutation_formula_l186_186662

noncomputable def permutation (n k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)

theorem permutation_formula (n k : ℕ) (h : 1 ≤ k ∧ k ≤ n) : permutation n k = Nat.factorial n / Nat.factorial (n - k) :=
by
  unfold permutation
  sorry

end permutation_formula_l186_186662


namespace find_distance_to_place_l186_186127

noncomputable def distance_to_place (speed_boat : ℝ) (speed_stream : ℝ) (total_time : ℝ) : ℝ :=
  let downstream_speed := speed_boat + speed_stream
  let upstream_speed := speed_boat - speed_stream
  let distance := (total_time * (downstream_speed * upstream_speed)) / (downstream_speed + upstream_speed)
  distance

theorem find_distance_to_place :
  distance_to_place 16 2 937.1428571428571 = 7392.92 :=
by
  sorry

end find_distance_to_place_l186_186127


namespace complement_union_A_B_l186_186927

def is_element_of_set_A (x : ℤ) : Prop := ∃ k : ℤ, x = 3 * k + 1
def is_element_of_set_B (x : ℤ) : Prop := ∃ k : ℤ, x = 3 * k + 2
def is_element_of_complement_U (x : ℤ) : Prop := ∃ k : ℤ, x = 3 * k

theorem complement_union_A_B :
  {x : ℤ | ¬ (is_element_of_set_A x ∨ is_element_of_set_B x)} = {x : ℤ | is_element_of_complement_U x} :=
by
  sorry

end complement_union_A_B_l186_186927


namespace volume_frustum_fraction_l186_186495

-- Define the base edge and initial altitude of the pyramid.
def base_edge := 32 -- in inches
def altitude_original := 1 -- in feet

-- Define the fractional part representing the altitude of the smaller pyramid.
def altitude_fraction := 1/4

-- Define the volume of the original pyramid being V.
noncomputable def volume_original : ℝ := (1/3) * (base_edge ^ 2) * altitude_original

-- Define the volume of the smaller pyramid being removed.
noncomputable def volume_smaller : ℝ := (1/3) * ((altitude_fraction * base_edge) ^ 2) * (altitude_fraction * altitude_original)

-- We now state the proof
theorem volume_frustum_fraction : 
  (volume_original - volume_smaller) / volume_original = 63/64 :=
by
  sorry

end volume_frustum_fraction_l186_186495


namespace miniature_model_to_actual_statue_scale_l186_186184

theorem miniature_model_to_actual_statue_scale (height_actual : ℝ) (height_model : ℝ) : 
  height_actual = 90 → height_model = 6 → 
  (height_actual / height_model = 15) := 
by
  intros h_actual h_model
  rw [h_actual, h_model]
  sorry

end miniature_model_to_actual_statue_scale_l186_186184


namespace derivative_of_f_l186_186498

noncomputable def f (x : ℝ) : ℝ := (Real.sin (1 / x)) ^ 3

theorem derivative_of_f (x : ℝ) (hx : x ≠ 0) : 
  deriv f x = - (3 / x ^ 2) * (Real.sin (1 / x)) ^ 2 * Real.cos (1 / x) :=
by
  sorry 

end derivative_of_f_l186_186498


namespace point_on_parabola_distance_l186_186666

theorem point_on_parabola_distance (a b : ℝ) (h1 : a^2 = 20 * b) (h2 : |b + 5| = 25) : |a * b| = 400 :=
sorry

end point_on_parabola_distance_l186_186666


namespace problem1_problem2_problem3_problem4_problem5_problem6_l186_186654

theorem problem1 : 78 * 4 + 488 = 800 := by sorry
theorem problem2 : 1903 - 475 * 4 = 3 := by sorry
theorem problem3 : 350 * (12 + 342 / 9) = 17500 := by sorry
theorem problem4 : 480 / (125 - 117) = 60 := by sorry
theorem problem5 : (3600 - 18 * 200) / 253 = 0 := by sorry
theorem problem6 : (243 - 162) / 27 * 380 = 1140 := by sorry

end problem1_problem2_problem3_problem4_problem5_problem6_l186_186654


namespace sum_three_consecutive_odd_integers_l186_186611

theorem sum_three_consecutive_odd_integers (n : ℤ) 
  (h : n + (n + 4) = 150) : n + (n + 2) + (n + 4) = 225 :=
by
  sorry

end sum_three_consecutive_odd_integers_l186_186611


namespace prism_section_area_l186_186116

noncomputable def area_of_section (AC : ℝ) (C_distance : ℝ) : ℝ :=
  if AC = 4 ∧ C_distance = 12 / 5 then 5 * Real.sqrt 3 / 2 else 0

-- Definitions and hypotheses for the problem
variables (AC : ℝ) (C_distance : ℝ)
variables (angle_B : ℝ) (angle_C : ℝ)

-- The problem statement and proof structure (statement only, proof not required)
theorem prism_section_area :
  angle_B = 90 ∧ angle_C = 30 ∧ AC = 4 ∧ C_distance = 12 / 5 →
  area_of_section AC C_distance = 5 * Real.sqrt 3 / 2 :=
by
  sorry

end prism_section_area_l186_186116


namespace chromatic_number_of_grid_3x5_l186_186707

-- Define a 3x5 grid graph where each vertex represents a square and edges represent adjacency by vertex or side
def grid_3x5 : SimpleGraph (Fin 3 × Fin 5) :=
  { adj := λ x y, (x.1 = y.1 ∧ (x.2 = y.2 + 1 ∨ x.2 = y.2 - 1)) ∨
                 (x.2 = y.2 ∧ (x.1 = y.1 + 1 ∨ x.1 = y.1 - 1)) ∨
                 ((x.1 = y.1 + 1 ∨ x.1 = y.1 - 1) ∧ (x.2 = y.2 + 1 ∨ x.2 = y.2 - 1)),
    sym := by finish,
    loopless := by finish }

-- A proof problem to determine the chromatic number of grid_3x5 is 4
theorem chromatic_number_of_grid_3x5 : chromaticNumber grid_3x5 = 4 :=
sorry

end chromatic_number_of_grid_3x5_l186_186707


namespace books_borrowed_in_a_week_l186_186246

theorem books_borrowed_in_a_week 
  (daily_avg : ℕ)
  (friday_increase_pct : ℕ)
  (days_open : ℕ)
  (friday_books : ℕ)
  (total_books_week : ℕ)
  (h1 : daily_avg = 40)
  (h2 : friday_increase_pct = 40)
  (h3 : days_open = 5)
  (h4 : friday_books = daily_avg + (daily_avg * friday_increase_pct / 100))
  (h5 : total_books_week = (days_open - 1) * daily_avg + friday_books) :
  total_books_week = 216 :=
by {
  sorry
}

end books_borrowed_in_a_week_l186_186246


namespace find_R_l186_186867

theorem find_R (a b : ℝ) (Q R : ℝ) (hQ : Q = 4)
  (h1 : 1/a + 1/b = Q/(a + b))
  (h2 : a/b + b/a = R) : R = 2 :=
by
  sorry

end find_R_l186_186867


namespace number_of_cities_sampled_from_group_B_l186_186592

variable (N_total : ℕ) (N_A : ℕ) (N_B : ℕ) (N_C : ℕ) (S : ℕ)

theorem number_of_cities_sampled_from_group_B :
    N_total = 48 → 
    N_A = 10 → 
    N_B = 18 → 
    N_C = 20 → 
    S = 16 → 
    (N_B * S) / N_total = 6 :=
by
  sorry

end number_of_cities_sampled_from_group_B_l186_186592


namespace last_two_digits_of_sum_l186_186050

noncomputable def sum_last_two_digits : ℕ :=
  let seq := (List.range' 1 2017) in
  let blocks := seq.enumerate.map (λ ⟨i, n⟩, if (i / 50) % 2 = 0 then n^2 else -n^2) in
  let overall_sum := blocks.sum in
  overall_sum % 100

theorem last_two_digits_of_sum : sum_last_two_digits = 85 := 
  sorry

end last_two_digits_of_sum_l186_186050


namespace norma_initial_cards_l186_186257

theorem norma_initial_cards (x : ℝ) 
  (H1 : x + 70 = 158) : 
  x = 88 :=
by
  sorry

end norma_initial_cards_l186_186257


namespace jay_change_l186_186897

theorem jay_change (book_price pen_price ruler_price payment : ℕ) (h1 : book_price = 25) (h2 : pen_price = 4) (h3 : ruler_price = 1) (h4 : payment = 50) : 
(book_price + pen_price + ruler_price ≤ payment) → (payment - (book_price + pen_price + ruler_price) = 20) :=
by
  intro h
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end jay_change_l186_186897


namespace percentage_of_number_l186_186229

theorem percentage_of_number (N P : ℝ) (h1 : 0.60 * N = 240) (h2 : (P / 100) * N = 160) : P = 40 :=
by
  sorry

end percentage_of_number_l186_186229


namespace triangle_area_is_correct_l186_186378

noncomputable def triangle_area (a b c B : ℝ) : ℝ := 
  0.5 * a * c * Real.sin B

theorem triangle_area_is_correct :
  let a := Real.sqrt 2
  let c := Real.sqrt 2
  let b := Real.sqrt 6
  let B := 2 * Real.pi / 3 -- 120 degrees in radians
  triangle_area a b c B = Real.sqrt 3 / 2 :=
by
  -- The proof is omitted as per the instructions.
  sorry

end triangle_area_is_correct_l186_186378


namespace sqrt_domain_condition_l186_186396

theorem sqrt_domain_condition (x : ℝ) : (2 * x - 6 ≥ 0) ↔ (x ≥ 3) :=
by
  sorry

end sqrt_domain_condition_l186_186396


namespace cream_ratio_l186_186088

noncomputable def John_creme_amount : ℚ := 3
noncomputable def Janet_initial_amount : ℚ := 8
noncomputable def Janet_creme_added : ℚ := 3
noncomputable def Janet_total_mixture : ℚ := Janet_initial_amount + Janet_creme_added
noncomputable def Janet_creme_ratio : ℚ := Janet_creme_added / Janet_total_mixture
noncomputable def Janet_drank_amount : ℚ := 3
noncomputable def Janet_drank_creme : ℚ := Janet_drank_amount * Janet_creme_ratio
noncomputable def Janet_creme_remaining : ℚ := Janet_creme_added - Janet_drank_creme

theorem cream_ratio :
  (John_creme_amount / Janet_creme_remaining) = (11 / 5) :=
by
  sorry

end cream_ratio_l186_186088


namespace measure_of_angle_A_values_of_b_and_c_l186_186886

variable (a b c : ℝ) (A : ℝ)

-- Declare the conditions as hypotheses
def condition1 (a b c : ℝ) := a^2 - c^2 = b^2 - b * c
def condition2 (a : ℝ) := a = 2
def condition3 (b c : ℝ) := b + c = 4

-- Proof that A = 60 degrees when the conditions are satisfied
theorem measure_of_angle_A (h : condition1 a b c) : A = 60 := by
  sorry

-- Proof that b and c are 2 when given conditions are satisfied
theorem values_of_b_and_c (h1 : condition1 2 b c) (h2 : condition3 b c) : b = 2 ∧ c = 2 := by
  sorry

end measure_of_angle_A_values_of_b_and_c_l186_186886


namespace largest_divisor_of_five_consecutive_integers_l186_186774

theorem largest_divisor_of_five_consecutive_integers:
  ∀ (n : ℤ), 
    ∃ (a b c : ℤ), 
      (n = 5 * a ∨ n = 5 * a + 1 ∨ n = 5 * a + 2 ∨ n = 5 * a + 3 ∨ n = 5 * a + 4) ∧ 
      (n = 3 * b ∨ n = 3 * b + 1 ∨ n = 3 * b + 2) ∧ 
      (n = 4 * c ∨ n = 4 * c + 1 ∨ n = 4 * c + 2 ∨ n = 4 * c + 3)
      ⊢ 60 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  -- Sorry placeholder for the proof (not needed in the task)
  sorry

end largest_divisor_of_five_consecutive_integers_l186_186774


namespace coffee_blend_price_l186_186100

theorem coffee_blend_price (x : ℝ) : 
  (9 * 8 + x * 12) / 20 = 8.4 → x = 8 :=
by
  intro h
  sorry

end coffee_blend_price_l186_186100


namespace derivative_evaluation_at_pi_over_3_l186_186872

noncomputable def f (x : ℝ) : ℝ :=
  Real.sin (2 * x) + Real.tan x

theorem derivative_evaluation_at_pi_over_3 :
  deriv f (Real.pi / 3) = 3 :=
sorry

end derivative_evaluation_at_pi_over_3_l186_186872


namespace car_kilometers_per_gallon_l186_186631

theorem car_kilometers_per_gallon :
  ∀ (distance gallon_used : ℝ), distance = 120 → gallon_used = 6 →
  distance / gallon_used = 20 :=
by
  intros distance gallon_used h_distance h_gallon_used
  sorry

end car_kilometers_per_gallon_l186_186631


namespace necklace_length_l186_186305

-- Given conditions as definitions in Lean
def num_pieces : ℕ := 16
def piece_length : ℝ := 10.4
def overlap_length : ℝ := 3.5
def effective_length : ℝ := piece_length - overlap_length
def total_length : ℝ := effective_length * num_pieces

-- The theorem to prove
theorem necklace_length :
  total_length = 110.4 :=
by
  -- Proof omitted
  sorry

end necklace_length_l186_186305


namespace smallest_diff_l186_186142

noncomputable def triangleSides : ℕ → ℕ → ℕ → Prop := λ AB BC AC =>
  AB < BC ∧ BC ≤ AC ∧ AB + BC + AC = 2007

theorem smallest_diff (AB BC AC : ℕ) (h : triangleSides AB BC AC) : BC - AB = 1 :=
  sorry

end smallest_diff_l186_186142


namespace area_of_rectangle_l186_186081

variable (AB AC : ℝ) -- Define the variables for the given sides of the rectangle
variable (h1 : AB = 15) (h2 : AC = 17) -- Define the given conditions

theorem area_of_rectangle (BC : ℝ) (h3 : AB^2 + BC^2 = AC^2) : 
  let AD := BC in
  AB * AD = 120 :=
by
  sorry

end area_of_rectangle_l186_186081


namespace greatest_whole_number_satisfying_inequalities_l186_186369

theorem greatest_whole_number_satisfying_inequalities :
  ∃ x : ℕ, 3 * (x : ℤ) - 5 < 1 - x ∧ 2 * (x : ℤ) + 4 ≤ 8 ∧ ∀ y : ℕ, y > x → ¬ (3 * (y : ℤ) - 5 < 1 - y ∧ 2 * (y : ℤ) + 4 ≤ 8) :=
sorry

end greatest_whole_number_satisfying_inequalities_l186_186369


namespace cafeteria_total_cost_l186_186432

-- Definitions based on conditions
def cost_per_coffee := 4
def cost_per_cake := 7
def cost_per_ice_cream := 3
def mell_coffee := 2 
def mell_cake := 1 
def friends_coffee := 2 
def friends_cake := 1 
def friends_ice_cream := 1 
def num_friends := 2
def total_coffee := mell_coffee + num_friends * friends_coffee
def total_cake := mell_cake + num_friends * friends_cake
def total_ice_cream := num_friends * friends_ice_cream

-- Total cost
def total_cost := total_coffee * cost_per_coffee + total_cake * cost_per_cake + total_ice_cream * cost_per_ice_cream

-- Theorem statement
theorem cafeteria_total_cost : total_cost = 51 := by
  sorry

end cafeteria_total_cost_l186_186432


namespace largest_divisor_of_five_consecutive_integers_l186_186769

theorem largest_divisor_of_five_consecutive_integers:
  ∀ (n : ℤ), 
    ∃ (a b c : ℤ), 
      (n = 5 * a ∨ n = 5 * a + 1 ∨ n = 5 * a + 2 ∨ n = 5 * a + 3 ∨ n = 5 * a + 4) ∧ 
      (n = 3 * b ∨ n = 3 * b + 1 ∨ n = 3 * b + 2) ∧ 
      (n = 4 * c ∨ n = 4 * c + 1 ∨ n = 4 * c + 2 ∨ n = 4 * c + 3)
      ⊢ 60 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  -- Sorry placeholder for the proof (not needed in the task)
  sorry

end largest_divisor_of_five_consecutive_integers_l186_186769


namespace other_x_intercept_l186_186207

theorem other_x_intercept (a b c : ℝ) 
  (h_eq : ∀ x, a * x^2 + b * x + c = y) 
  (h_vertex: (5, 10) = ((-b / (2 * a)), (4 * a * 10 / (4 * a)))) 
  (h_intercept : ∃ x, a * x * 0 + b * 0 + c = 0) : ∃ x, x = 10 :=
by
  sorry

end other_x_intercept_l186_186207


namespace sufficient_not_necessary_condition_x_plus_a_div_x_geq_2_l186_186515

open Real

theorem sufficient_not_necessary_condition_x_plus_a_div_x_geq_2 (x a : ℝ)
  (h₁ : x > 0) :
  (∀ x > 0, x + a / x ≥ 2) → (a = 1) :=
sorry

end sufficient_not_necessary_condition_x_plus_a_div_x_geq_2_l186_186515


namespace least_three_digit_product_12_l186_186150

-- Problem statement: Find the least three-digit number whose digits multiply to 12
theorem least_three_digit_product_12 : ∃ (n : ℕ), n ≥ 100 ∧ n < 1000 ∧ (∃ a b c : ℕ, n = 100 * a + 10 * b + c ∧ a * b * c = 12 ∧ n = 126) :=
by
  sorry

end least_three_digit_product_12_l186_186150


namespace largest_divisor_of_consecutive_product_l186_186809

theorem largest_divisor_of_consecutive_product (n : ℤ) (h : ∀ k : ℤ, (n = 5 * k)) :
  ∃ d : ℤ, d = 60 ∧ ∀ a : ℤ, divides d (a * (a + 1) * (a + 2) * (a + 3) * (a + 4)) :=
sorry

end largest_divisor_of_consecutive_product_l186_186809


namespace max_value_m_l186_186885

theorem max_value_m (a b : ℝ) (ha : a > 0) (hb : b > 0) (m : ℝ)
  (h : (2 / a) + (1 / b) ≥ m / (2 * a + b)) : m ≤ 9 :=
sorry

end max_value_m_l186_186885


namespace reduced_price_is_correct_l186_186493

-- Definitions for the conditions in the problem
def original_price_per_dozen (P : ℝ) : Prop :=
∀ (X : ℝ), X * P = 40.00001

def reduced_price_per_dozen (P R : ℝ) : Prop :=
R = 0.60 * P

def bananas_purchased_additional (P R : ℝ) : Prop :=
∀ (X Y : ℝ), (Y = X + (64 / 12)) → (X * P = Y * R) 

-- Assertion of the proof problem
theorem reduced_price_is_correct : 
  ∃ (R : ℝ), 
  (∀ P, original_price_per_dozen P ∧ reduced_price_per_dozen P R ∧ bananas_purchased_additional P R) → 
  R = 3.00000075 := 
by sorry

end reduced_price_is_correct_l186_186493


namespace largest_integer_dividing_consecutive_product_l186_186815

theorem largest_integer_dividing_consecutive_product :
  ∀ (n : ℤ), (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) % 120 = 0 :=
by
  sorry

end largest_integer_dividing_consecutive_product_l186_186815


namespace union_eq_C_l186_186832

def A: Set ℝ := { x | x > 2 }
def B: Set ℝ := { x | x < 0 }
def C: Set ℝ := { x | x * (x - 2) > 0 }

theorem union_eq_C : (A ∪ B) = C :=
by
  sorry

end union_eq_C_l186_186832


namespace find_a8_a12_l186_186216

noncomputable def geometric_sequence_value_8_12 (a : ℕ → ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  if n = 0 then a 0 else a 0 * q^n

theorem find_a8_a12 (a : ℕ → ℝ) (q : ℝ) (terms_geometric : ∀ n, a n = a 0 * q^n)
  (h2_6 : a 2 + a 6 = 3) (h6_10 : a 6 + a 10 = 12) :
  a 8 + a 12 = 24 :=
by
  sorry

end find_a8_a12_l186_186216


namespace total_cost_proof_l186_186430

-- Define the prices of items
def price_coffee : ℕ := 4
def price_cake : ℕ := 7
def price_ice_cream : ℕ := 3

-- Define the number of items ordered by Mell and her friends
def mell_coffee : ℕ := 2
def mell_cake : ℕ := 1
def friend_coffee : ℕ := 2
def friend_cake : ℕ := 1
def friend_ice_cream : ℕ := 1
def number_of_friends : ℕ := 2

-- Calculate total cost for Mell
def total_mell : ℕ := (mell_coffee * price_coffee) + (mell_cake * price_cake)

-- Calculate total cost per friend
def total_friend : ℕ := (friend_coffee * price_coffee) + (friend_cake * price_cake) + (friend_ice_cream * price_ice_cream)

-- Calculate total cost for all friends
def total_friends : ℕ := number_of_friends * total_friend

-- Calculate total cost for Mell and her friends
def total_cost : ℕ := total_mell + total_friends

-- The theorem to prove
theorem total_cost_proof : total_cost = 51 := by
  sorry

end total_cost_proof_l186_186430


namespace division_of_decimals_l186_186684

theorem division_of_decimals :
  (0.1 / 0.001 = 100) ∧ (1 / 0.01 = 100) := by
  sorry

end division_of_decimals_l186_186684


namespace graph_symmetry_about_line_l186_186063

noncomputable def f (x : ℝ) : ℝ := 3 * Real.cos (2 * x - (Real.pi / 3))

theorem graph_symmetry_about_line (x : ℝ) : 
  ∀ x, f (2 * (Real.pi / 3) - x) = f x :=
by
  sorry

end graph_symmetry_about_line_l186_186063


namespace largest_divisor_of_5_consecutive_integers_l186_186796

theorem largest_divisor_of_5_consecutive_integers :
  ∀ n : ℤ, ∃ k : ℤ, (product_of_5_consecutive_integers n = k) ∧ (60 ∣ k) 
:= sorry

end largest_divisor_of_5_consecutive_integers_l186_186796


namespace angle_I_measure_l186_186543

theorem angle_I_measure {x y : ℝ} 
  (h1 : x = y - 50) 
  (h2 : 3 * x + 2 * y = 540)
  : y = 138 := 
by 
  sorry

end angle_I_measure_l186_186543


namespace James_trout_pounds_l186_186244

def pounds_trout (T : ℝ) : Prop :=
  let salmon := 1.5 * T
  let tuna := 2 * T
  T + salmon + tuna = 1100

theorem James_trout_pounds :
  ∃ T : ℝ, pounds_trout T ∧ T = 244 :=
sorry

end James_trout_pounds_l186_186244


namespace marble_leftovers_l186_186154

theorem marble_leftovers :
  ∃ r p : ℕ, (r % 8 = 5) ∧ (p % 8 = 7) ∧ ((r + p) % 10 = 0) → ((r + p) % 8 = 4) :=
by { sorry }

end marble_leftovers_l186_186154


namespace parabola_increasing_implies_a_lt_zero_l186_186233

theorem parabola_increasing_implies_a_lt_zero (a : ℝ) :
  (∀ x : ℝ, x < 0 → a * (2 * x) > 0) → a < 0 :=
by
  sorry

end parabola_increasing_implies_a_lt_zero_l186_186233


namespace largest_divisor_of_product_of_five_consecutive_integers_l186_186767

theorem largest_divisor_of_product_of_five_consecutive_integers
  (a b c d e : ℤ) 
  (h: a + 1 = b ∧ b + 1 = c ∧ c + 1 = d ∧ d + 1 = e):
  ∃ (n : ℤ), n = 60 ∧ n ∣ (a * b * c * d * e) :=
sorry

end largest_divisor_of_product_of_five_consecutive_integers_l186_186767


namespace ms_warren_walking_speed_correct_l186_186435

noncomputable def walking_speed_proof : Prop :=
  let running_speed := 6 -- mph
  let running_time := 20 / 60 -- hours
  let total_distance := 3 -- miles
  let distance_ran := running_speed * running_time
  let distance_walked := total_distance - distance_ran
  let walking_time := 30 / 60 -- hours
  let walking_speed := distance_walked / walking_time
  walking_speed = 2

theorem ms_warren_walking_speed_correct (walking_speed_proof : Prop) : walking_speed_proof :=
by sorry

end ms_warren_walking_speed_correct_l186_186435


namespace largest_divisor_of_consecutive_five_l186_186791

theorem largest_divisor_of_consecutive_five (n : ℤ) : 
  (∃ d, ∀ i ∈ {n, n + 1, n + 2, n + 3, n + 4}, d ∣ (∏ x in {n, n + 1, n + 2, n + 3, n + 4}, x)) → 
  d = 60 
:= sorry

end largest_divisor_of_consecutive_five_l186_186791


namespace parabola_intersection_value_l186_186869

theorem parabola_intersection_value (a : ℝ) (h : a^2 - a - 1 = 0) : a^2 - a + 2014 = 2015 :=
by
  sorry

end parabola_intersection_value_l186_186869


namespace expression_value_l186_186186

-- The problem statement definition
def expression := 2 + 3 * 4 - 5 / 5 + 7

-- Theorem statement asserting the final result
theorem expression_value : expression = 20 := 
by sorry

end expression_value_l186_186186


namespace record_loss_of_10_l186_186230

-- Definition of profit and loss recording
def record (x : Int) : Int :=
  if x ≥ 0 then x else -x

-- Condition: A profit of $20 should be recorded as +$20
axiom profit_recording : ∀ (p : Int), p ≥ 0 → record p = p

-- Condition: A loss should be recorded as a negative amount
axiom loss_recording : ∀ (l : Int), l < 0 → record l = l

-- Question: How should a loss of $10 be recorded?
-- Prove that if a small store lost $10, it should be recorded as -$10
theorem record_loss_of_10 : record (-10) = -10 :=
by sorry

end record_loss_of_10_l186_186230


namespace elliptical_oil_tank_depth_l186_186646

noncomputable def solve_oil_depth
  (length_tank : ℝ) 
  (major_axis : ℝ) 
  (minor_axis : ℝ)
  (oil_surface_area : ℝ)
  (rectangular_surface_area : ℝ) : ℝ :=
let h := (2.0 + 2.4) / 2.0 in -- Depth h of oil, to be computed based on geometric relations
h

theorem elliptical_oil_tank_depth
  (length_tank : ℝ)
  (major_axis : ℝ)
  (minor_axis : ℝ)
  (oil_surface_area : ℝ) : 
  (length_tank = 10) →
  (major_axis = 8) →
  (minor_axis = 6) →
  (oil_surface_area = 48) →
  solve_oil_depth length_tank major_axis minor_axis oil_surface_area 48 = 1.2 ∨ solve_oil_depth length_tank major_axis minor_axis oil_surface_area 48 = 4.8 :=
by 
  intros h_main h_major h_minor h_area
  -- Here we skip the proof, as it involves calculus and specific solving steps
  sorry

end elliptical_oil_tank_depth_l186_186646


namespace total_fruits_picked_l186_186107

theorem total_fruits_picked :
  let sara_pears := 6
  let tim_pears := 5
  let lily_apples := 4
  let max_oranges := 3
  sara_pears + tim_pears + lily_apples + max_oranges = 18 :=
by
  -- skip the proof
  sorry

end total_fruits_picked_l186_186107


namespace complement_union_A_B_l186_186928

def is_element_of_set_A (x : ℤ) : Prop := ∃ k : ℤ, x = 3 * k + 1
def is_element_of_set_B (x : ℤ) : Prop := ∃ k : ℤ, x = 3 * k + 2
def is_element_of_complement_U (x : ℤ) : Prop := ∃ k : ℤ, x = 3 * k

theorem complement_union_A_B :
  {x : ℤ | ¬ (is_element_of_set_A x ∨ is_element_of_set_B x)} = {x : ℤ | is_element_of_complement_U x} :=
by
  sorry

end complement_union_A_B_l186_186928


namespace constant_sum_powers_l186_186367

theorem constant_sum_powers (n : ℕ) (x y z : ℝ) (h_sum : x + y + z = 0) (h_prod : x * y * z = 1) :
  (∀ x y z : ℝ, x + y + z = 0 → x * y * z = 1 → x^n + y^n + z^n = x^n + y^n + z^n ↔ (n = 1 ∨ n = 3)) :=
by
  sorry

end constant_sum_powers_l186_186367


namespace gcd_g_y_l186_186673

def g (y : ℕ) : ℕ := (3*y + 4) * (8*y + 3) * (14*y + 9) * (y + 17)

theorem gcd_g_y (y : ℕ) (h : y % 42522 = 0) : Nat.gcd (g y) y = 102 := by
  sorry

end gcd_g_y_l186_186673


namespace ten_digit_number_contains_repeated_digit_l186_186121

open Nat

theorem ten_digit_number_contains_repeated_digit
  (n : ℕ)
  (h1 : 10^9 ≤ n^2 + 1)
  (h2 : n^2 + 1 < 10^10) :
  ∃ d1 d2 : ℕ, d1 ≠ d2 ∧ (d1 ∈ (digits 10 (n^2 + 1))) ∧ (d2 ∈ (digits 10 (n^2 + 1))) :=
sorry

end ten_digit_number_contains_repeated_digit_l186_186121


namespace cost_difference_of_dolls_proof_l186_186192

-- Define constants
def cost_large_doll : ℝ := 7
def total_spent : ℝ := 350
def additional_dolls : ℝ := 20

-- Define the function for the cost of small dolls
def cost_small_doll (S : ℝ) : Prop :=
  total_spent / S = total_spent / cost_large_doll + additional_dolls

-- The statement given the conditions and solving for the difference in cost
theorem cost_difference_of_dolls_proof : 
  ∃ S, cost_small_doll S ∧ (cost_large_doll - S = 2) :=
by
  sorry

end cost_difference_of_dolls_proof_l186_186192


namespace central_student_coins_l186_186195

theorem central_student_coins (n_students: ℕ) (total_coins : ℕ)
  (equidistant_same : Prop)
  (coin_exchange : Prop):
  (n_students = 16) →
  (total_coins = 3360) →
  (equidistant_same) →
  (coin_exchange) →
  ∃ coins_in_center: ℕ, coins_in_center = 280 :=
by
  intros
  sorry

end central_student_coins_l186_186195


namespace sin_inequalities_l186_186527

theorem sin_inequalities (x : ℝ) (h1 : 0 < x) (h2 : x < π / 4) :
  sin (sin x) < sin x ∧ sin x < sin (tan x) := by
sorry

end sin_inequalities_l186_186527


namespace polynomial_value_at_neg3_l186_186123

def polynomial (a b c x : ℝ) : ℝ := a * x^5 - b * x^3 + c * x - 7

theorem polynomial_value_at_neg3 (a b c : ℝ) (h : polynomial a b c 3 = 65) :
  polynomial a b c (-3) = -79 := 
sorry

end polynomial_value_at_neg3_l186_186123


namespace original_price_proof_l186_186308

noncomputable def original_price (profit selling_price : ℝ) : ℝ :=
  (profit / 0.20)

theorem original_price_proof (P : ℝ) : 
  original_price 600 (P + 600) = 3000 :=
by
  sorry

end original_price_proof_l186_186308


namespace sum_of_ages_l186_186965

-- Definitions based on given conditions
def J : ℕ := 19
def age_difference (B J : ℕ) : Prop := B - J = 32

-- Theorem stating the problem
theorem sum_of_ages (B : ℕ) (H : age_difference B J) : B + J = 70 :=
sorry

end sum_of_ages_l186_186965


namespace total_orchestra_l186_186108

def percussion_section : ℕ := 4
def brass_section : ℕ := 13
def strings_section : ℕ := 18
def woodwinds_section : ℕ := 10
def keyboards_and_harp_section : ℕ := 3
def maestro : ℕ := 1

theorem total_orchestra (p b s w k m : ℕ) 
  (h_p : p = percussion_section)
  (h_b : b = brass_section)
  (h_s : s = strings_section)
  (h_w : w = woodwinds_section)
  (h_k : k = keyboards_and_harp_section)
  (h_m : m = maestro) :
  p + b + s + w + k + m = 49 := by 
  rw [h_p, h_b, h_s, h_w, h_k, h_m]
  unfold percussion_section brass_section strings_section woodwinds_section keyboards_and_harp_section maestro
  norm_num

end total_orchestra_l186_186108


namespace math_competition_l186_186076

theorem math_competition (a b c d e f g : ℕ) (h1 : a + b + c + d + e + f + g = 25)
    (h2 : b = 2 * c + f) (h3 : a = d + e + g + 1) (h4 : a = b + c) :
    b = 6 :=
by
  -- The proof is omitted as the problem requests the statement only.
  sorry

end math_competition_l186_186076


namespace largest_integer_dividing_consecutive_product_l186_186816

theorem largest_integer_dividing_consecutive_product :
  ∀ (n : ℤ), (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) % 120 = 0 :=
by
  sorry

end largest_integer_dividing_consecutive_product_l186_186816


namespace total_earnings_proof_l186_186159

noncomputable def total_earnings (x y : ℝ) : ℝ :=
  let earnings_a := (18 * x * y) / 100
  let earnings_b := (20 * x * y) / 100
  let earnings_c := (20 * x * y) / 100
  earnings_a + earnings_b + earnings_c

theorem total_earnings_proof (x y : ℝ) (h : 2 * x * y = 15000) :
  total_earnings x y = 4350 := by
  sorry

end total_earnings_proof_l186_186159


namespace martian_angle_conversion_l186_186096

-- Defines the full circle measurements
def full_circle_clerts : ℕ := 600
def full_circle_degrees : ℕ := 360
def angle_degrees : ℕ := 60

-- The main statement to prove
theorem martian_angle_conversion : 
    (full_circle_clerts * angle_degrees) / full_circle_degrees = 100 :=
by
  sorry  

end martian_angle_conversion_l186_186096


namespace lieutenant_age_l186_186176

variables (n x : ℕ) 

-- Condition 1: Number of soldiers is the same in both formations
def total_soldiers_initial (n : ℕ) : ℕ := n * (n + 5)
def total_soldiers_new (n x : ℕ) : ℕ := x * (n + 9)

-- Condition 2: The number of soldiers is the same 
-- and Condition 3: Equations relating n and x
theorem lieutenant_age (n x : ℕ) (h1: total_soldiers_initial n = total_soldiers_new n x) (h2 : x = 24) : 
  x = 24 :=
by {
  sorry
}

end lieutenant_age_l186_186176


namespace original_expression_equals_l186_186447

noncomputable def evaluate_expression (a : ℝ) : ℝ :=
  ( (a / (a + 2) + 1 / (a^2 - 4)) / ( (a - 1) / (a + 2) + 1 / (a - 2) ))

theorem original_expression_equals (a : ℝ) (h : a = 2 + Real.sqrt 2) :
  evaluate_expression a = (Real.sqrt 2 + 1) :=
sorry

end original_expression_equals_l186_186447


namespace mass_of_empty_glass_l186_186172

theorem mass_of_empty_glass (mass_full : ℕ) (mass_half : ℕ) (G : ℕ) :
  mass_full = 1000 →
  mass_half = 700 →
  G = mass_full - (mass_full - mass_half) * 2 →
  G = 400 :=
by
  intros h_full h_half h_G_eq
  sorry

end mass_of_empty_glass_l186_186172


namespace negation_of_universal_prop_l186_186122

theorem negation_of_universal_prop :
  (¬ ∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) ↔ ∃ x : ℝ, x^3 - x^2 + 1 > 0 := 
by 
  sorry

end negation_of_universal_prop_l186_186122


namespace solution_l186_186084

def money_problem (x y : ℝ) : Prop :=
  (x + y / 2 = 50) ∧ (y + 2 * x / 3 = 50)

theorem solution :
  ∃ x y : ℝ, money_problem x y ∧ x = 37.5 ∧ y = 25 :=
by
  use 37.5, 25
  sorry

end solution_l186_186084


namespace sum_even_and_odd_numbers_up_to_50_l186_186372

def sum_even_numbers (n : ℕ) : ℕ :=
  (2 + 50) * n / 2

def sum_odd_numbers (n : ℕ) : ℕ :=
  (1 + 49) * n / 2

theorem sum_even_and_odd_numbers_up_to_50 : 
  sum_even_numbers 25 + sum_odd_numbers 25 = 1275 :=
by
  sorry

end sum_even_and_odd_numbers_up_to_50_l186_186372


namespace complement_union_eq_complement_l186_186908

open Set

section ComplementUnion

variable (k : ℤ)

def SetA : Set ℤ := {x | ∃ k : ℤ, x = 3 * k + 1}
def SetB : Set ℤ := {x | ∃ k : ℤ, x = 3 * k + 2}
def UniversalSet : Set ℤ := univ
def ComplementUnion : Set ℤ := {x | ∃ k : ℤ, x = 3 * k}

theorem complement_union_eq_complement :
  UniversalSet \ (SetA ∪ SetB) = ComplementUnion :=
by
  sorry

end ComplementUnion

end complement_union_eq_complement_l186_186908


namespace triangle_inequality_values_l186_186998

theorem triangle_inequality_values (x : ℕ) :
  x ≥ 2 ∧ x < 10 ↔ (x = 2 ∨ x = 3 ∨ x = 4 ∨ x = 5 ∨ x = 6 ∨ x = 7 ∨ x = 8 ∨ x = 9) :=
by sorry

end triangle_inequality_values_l186_186998


namespace gcd_455_299_eq_13_l186_186004

theorem gcd_455_299_eq_13 : Nat.gcd 455 299 = 13 := by
  sorry

end gcd_455_299_eq_13_l186_186004


namespace arithmetic_seq_common_diff_l186_186676

theorem arithmetic_seq_common_diff
  (a₃ a₇ S₁₀ : ℤ)
  (h₁ : a₃ + a₇ = 16)
  (h₂ : S₁₀ = 85)
  (a₃_eq : ∃ a₁ d : ℤ, a₃ = a₁ + 2 * d)
  (a₇_eq : ∃ a₁ d : ℤ, a₇ = a₁ + 6 * d)
  (S₁₀_eq : ∃ a₁ d : ℤ, S₁₀ = 10 * a₁ + 45 * d) :
  ∃ d : ℤ, d = 1 :=
by
  sorry

end arithmetic_seq_common_diff_l186_186676


namespace tangent_line_to_parabola_l186_186002

theorem tangent_line_to_parabola :
  (∀ (x y : ℝ), y = x^2 → x = -1 → y = 1 → 2 * x + y + 1 = 0) :=
by
  intro x y parabola eq_x eq_y
  sorry

end tangent_line_to_parabola_l186_186002


namespace total_spent_two_years_l186_186409

def home_game_price : ℕ := 60
def away_game_price : ℕ := 75
def home_playoff_price : ℕ := 120
def away_playoff_price : ℕ := 100

def this_year_home_games : ℕ := 2
def this_year_away_games : ℕ := 2
def this_year_home_playoff_games : ℕ := 1
def this_year_away_playoff_games : ℕ := 0

def last_year_home_games : ℕ := 6
def last_year_away_games : ℕ := 3
def last_year_home_playoff_games : ℕ := 1
def last_year_away_playoff_games : ℕ := 1

def calculate_total_cost : ℕ :=
  let this_year_cost := this_year_home_games * home_game_price + this_year_away_games * away_game_price + this_year_home_playoff_games * home_playoff_price + this_year_away_playoff_games * away_playoff_price
  let last_year_cost := last_year_home_games * home_game_price + last_year_away_games * away_game_price + last_year_home_playoff_games * home_playoff_price + last_year_away_playoff_games * away_playoff_price
  this_year_cost + last_year_cost

theorem total_spent_two_years : calculate_total_cost = 1195 :=
by
  sorry

end total_spent_two_years_l186_186409


namespace total_oranges_proof_l186_186436

def jeremyMonday : ℕ := 100
def jeremyTuesdayPlusBrother : ℕ := 3 * jeremyMonday
def jeremyWednesdayPlusBrotherPlusCousin : ℕ := 2 * jeremyTuesdayPlusBrother
def jeremyThursday : ℕ := (70 * jeremyMonday) / 100
def cousinWednesday : ℕ := jeremyTuesdayPlusBrother - (20 * jeremyTuesdayPlusBrother) / 100
def cousinThursday : ℕ := cousinWednesday + (30 * cousinWednesday) / 100

def total_oranges : ℕ :=
  jeremyMonday + jeremyTuesdayPlusBrother + jeremyWednesdayPlusBrotherPlusCousin + (jeremyThursday + (jeremyWednesdayPlusBrotherPlusCousin - cousinWednesday) + cousinThursday)

theorem total_oranges_proof : total_oranges = 1642 :=
by
  sorry

end total_oranges_proof_l186_186436


namespace cut_out_square_possible_l186_186377

/-- 
Formalization of cutting out eight \(2 \times 1\) rectangles from an \(8 \times 8\) 
checkered board, and checking if it is always possible to cut out a \(2 \times 2\) square
from the remaining part of the board.
-/
theorem cut_out_square_possible :
  ∀ (board : ℕ) (rectangles : ℕ), (board = 64) ∧ (rectangles = 8) → (4 ∣ board) →
  ∃ (remaining_squares : ℕ), (remaining_squares = 48) ∧ 
  (∃ (square_size : ℕ), (square_size = 4) ∧ (remaining_squares ≥ square_size)) :=
by {
  sorry
}

end cut_out_square_possible_l186_186377


namespace total_cost_l186_186423

-- Given conditions
def pen_cost : ℕ := 4
def briefcase_cost : ℕ := 5 * pen_cost

-- Theorem stating the total cost Marcel paid for both items
theorem total_cost (pen_cost briefcase_cost : ℕ) (h_pen: pen_cost = 4) (h_briefcase: briefcase_cost = 5 * pen_cost) :
  pen_cost + briefcase_cost = 24 := by
  sorry

end total_cost_l186_186423


namespace arithmetic_sequence_sum_l186_186379

theorem arithmetic_sequence_sum (a : ℕ → ℤ) (S : ℕ → ℤ) (b : ℕ → ℚ) (T : ℕ → ℚ) :
  (a 3 = 7) ∧ (a 5 + a 7 = 26) →
  (∀ n, a n = 2 * n + 1) ∧
  (∀ n, S n = n^2 + 2 * n) ∧
  (∀ n, b n = 1 / ((a n)^2 - 1)) →
  (∀ n, T n = n / (4 * (n + 1))) := sorry

end arithmetic_sequence_sum_l186_186379


namespace largest_divisor_of_product_of_five_consecutive_integers_l186_186763

theorem largest_divisor_of_product_of_five_consecutive_integers
  (a b c d e : ℤ) 
  (h: a + 1 = b ∧ b + 1 = c ∧ c + 1 = d ∧ d + 1 = e):
  ∃ (n : ℤ), n = 60 ∧ n ∣ (a * b * c * d * e) :=
sorry

end largest_divisor_of_product_of_five_consecutive_integers_l186_186763


namespace polygon_sides_eight_l186_186026

theorem polygon_sides_eight {n : ℕ} (h : (n - 2) * 180 = 3 * 360) : n = 8 :=
by
  sorry

end polygon_sides_eight_l186_186026


namespace largest_angle_bounds_triangle_angles_l186_186626

theorem largest_angle_bounds (A B C : ℝ) (angle_A angle_B angle_C : ℝ)
  (h_triangle : angle_A + angle_B + angle_C = 180)
  (h_tangent : angle_B + 2 * angle_C = 90) :
  90 ≤ angle_A ∧ angle_A < 135 :=
sorry

theorem triangle_angles (A B C : ℝ) (angle_A angle_B angle_C : ℝ)
  (h_triangle : angle_A + angle_B + angle_C = 180)
  (h_tangent_B : angle_B + 2 * angle_C = 90)
  (h_tangent_C : angle_C + 2 * angle_B = 90) :
  angle_A = 120 ∧ angle_B = 30 ∧ angle_C = 30 :=
sorry

end largest_angle_bounds_triangle_angles_l186_186626


namespace packing_heights_difference_l186_186013

-- Definitions based on conditions
def diameter := 8   -- Each pipe has a diameter of 8 cm
def num_pipes := 160 -- Each crate contains 160 pipes

-- Heights of the crates based on the given packing methods
def height_crate_A := 128 -- Calculated height for Crate A

noncomputable def height_crate_B := 8 + 60 * Real.sqrt 3 -- Calculated height for Crate B

-- Positive difference in the total heights of the two packings
noncomputable def delta_height := height_crate_A - height_crate_B

-- The goal to prove
theorem packing_heights_difference :
  delta_height = 120 - 60 * Real.sqrt 3 :=
sorry

end packing_heights_difference_l186_186013


namespace N_subset_proper_M_l186_186681

open Set Int

def set_M : Set ℝ := {x | ∃ k : ℤ, x = (k + 2) / 4}
def set_N : Set ℝ := {x | ∃ k : ℤ, x = (2 * k + 1) / 4}

theorem N_subset_proper_M : set_N ⊂ set_M := by
  sorry

end N_subset_proper_M_l186_186681


namespace three_flips_probability_l186_186971

open Probability

theorem three_flips_probability :
  ∀ (prob_heads : ℙ → bool) (prob_tails : ℙ → bool),
  (∀ (p : ℙ), prob_heads p = true → Prob p (λ x, x = true) = 1 / 2) →
  (∀ (p : ℙ), prob_tails p = false → Prob p (λ x, x = false) = 1 / 2) →
  let events := [prob_heads, prob_tails, prob_heads] in
  (∀ (p : ℙ), Prob p (λ x, events.all (λ f, f p x)) = 1 / 8) :=
by sorry

end three_flips_probability_l186_186971


namespace printer_cost_l186_186303

theorem printer_cost (total_cost : ℕ) (num_keyboards : ℕ) (keyboard_cost : ℕ) (num_printers : ℕ) (printer_cost : ℕ) :
  total_cost = 2050 → num_keyboards = 15 → keyboard_cost = 20 → num_printers = 25 →
  (total_cost - num_keyboards * keyboard_cost) / num_printers = 70 := 
by
  intros h_tc h_nk h_kc h_np
  rw [h_tc, h_nk, h_kc, h_np]
  norm_num
  sorry

end printer_cost_l186_186303


namespace ThreeStudentsGotA_l186_186205

-- Definitions of students receiving A grades
variable (Edward Fiona George Hannah Ian : Prop)

-- Conditions given in the problem
axiom H1 : Edward → Fiona
axiom H2 : Fiona → George
axiom H3 : George → Hannah
axiom H4 : Hannah → Ian
axiom H5 : (Edward → False) ∧ (Fiona → False)

-- Theorem stating the final result
theorem ThreeStudentsGotA : (George ∧ Hannah ∧ Ian) ∧ 
                            (¬Edward ∧ ¬Fiona) ∧ 
                            (Edward ∨ Fiona ∨ George ∨ Hannah ∨ Ian) :=
by
  sorry

end ThreeStudentsGotA_l186_186205


namespace smallest_z_value_l186_186403

theorem smallest_z_value :
  ∃ (w x y z : ℕ), w < x ∧ x < y ∧ y < z ∧
  w + 1 = x ∧ x + 1 = y ∧ y + 1 = z ∧
  w^3 + x^3 + y^3 = z^3 ∧ z = 6 := by
  sorry

end smallest_z_value_l186_186403


namespace sin_690_deg_l186_186359

noncomputable def sin_690_eq_neg_one_half : Prop :=
  sin (690 * real.pi / 180) = -(1 / 2)

theorem sin_690_deg : sin_690_eq_neg_one_half :=
  by sorry

end sin_690_deg_l186_186359


namespace cost_of_article_l186_186693

variable {C G : ℝ}

theorem cost_of_article (h : 350 = C * (1 + (G + 5) / 100)) (h' : 340 = C * (1 + G / 100)) : C = 200 := by
  sorry

end cost_of_article_l186_186693


namespace determine_function_l186_186414

open Nat

theorem determine_function (f : ℕ × ℕ → ℕ)
  (h₁ : ∀ (a b c : ℕ), f(gcd a b, c) = gcd a (f(c, b)))
  (h₂ : ∀ a : ℕ, f(a, a) ≥ a) :
  ∀ (a b : ℕ), f(a, b) = gcd a b := 
by
  sorry

end determine_function_l186_186414


namespace complement_union_A_B_l186_186923

def A := {x : ℤ | ∃ k : ℤ, x = 3 * k + 1}
def B := {x : ℤ | ∃ k : ℤ, x = 3 * k + 2}
def U := {x : ℤ | True}

theorem complement_union_A_B:
  (U \ (A ∪ B) = {x : ℤ | ∃ k : ℤ, x = 3 * k}) :=
by
  sorry

end complement_union_A_B_l186_186923


namespace problems_per_page_is_eight_l186_186576

noncomputable def totalProblems := 60
noncomputable def finishedProblems := 20
noncomputable def totalPages := 5
noncomputable def problemsLeft := totalProblems - finishedProblems
noncomputable def problemsPerPage := problemsLeft / totalPages

theorem problems_per_page_is_eight :
  problemsPerPage = 8 :=
by
  sorry

end problems_per_page_is_eight_l186_186576


namespace intersection_A_B_l186_186877

open Set

def A : Set ℤ := {x : ℤ | ∃ y : ℝ, y = Real.sqrt (1 - (x : ℝ)^2)}
def B : Set ℤ := {y : ℤ | ∃ x : ℤ, x ∈ A ∧ y = 2 * x - 1}

theorem intersection_A_B : A ∩ B = {-1, 1} := 
by {
  sorry
}

end intersection_A_B_l186_186877


namespace total_problems_l186_186256

-- We define the conditions as provided.
variables (p t : ℕ) -- p and t are positive whole numbers
variables (p_gt_10 : 10 < p) -- p is more than 10

theorem total_problems (p t : ℕ) (p_gt_10 : 10 < p) (h : p * t = (2 * p - 4) * (t - 2)):
  p * t = 60 :=
by
  sorry

end total_problems_l186_186256


namespace marcus_sees_7_l186_186254

variable (marcus humphrey darrel : ℕ)
variable (humphrey_sees : humphrey = 11)
variable (darrel_sees : darrel = 9)
variable (average_is_9 : (marcus + humphrey + darrel) / 3 = 9)

theorem marcus_sees_7 : marcus = 7 :=
by
  -- Needs proof
  sorry

end marcus_sees_7_l186_186254


namespace maria_waist_size_in_cm_l186_186939

noncomputable def waist_size_in_cm (waist_size_inches : ℕ) (extra_inch : ℕ) (inches_per_foot : ℕ) (cm_per_foot : ℕ) : ℚ :=
  let total_inches := waist_size_inches + extra_inch
  let total_feet := (total_inches : ℚ) / inches_per_foot
  total_feet * cm_per_foot

theorem maria_waist_size_in_cm :
  waist_size_in_cm 28 1 12 31 = 74.9 :=
by
  sorry

end maria_waist_size_in_cm_l186_186939


namespace point_reflection_xOy_l186_186544

structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def reflection_over_xOy (P : Point3D) : Point3D := 
  {x := P.x, y := P.y, z := -P.z}

theorem point_reflection_xOy :
  reflection_over_xOy {x := 1, y := 2, z := 3} = {x := 1, y := 2, z := -3} := by
  sorry

end point_reflection_xOy_l186_186544


namespace total_cost_calculation_l186_186426

-- Definitions
def coffee_price : ℕ := 4
def cake_price : ℕ := 7
def ice_cream_price : ℕ := 3

def mell_coffee_qty : ℕ := 2
def mell_cake_qty : ℕ := 1
def friends_coffee_qty : ℕ := 2
def friends_cake_qty : ℕ := 1
def friends_ice_cream_qty : ℕ := 1

def total_coffee_qty : ℕ := 3 * mell_coffee_qty
def total_cake_qty : ℕ := 3 * mell_cake_qty
def total_ice_cream_qty : ℕ := 2 * friends_ice_cream_qty

def total_cost : ℕ := total_coffee_qty * coffee_price + total_cake_qty * cake_price + total_ice_cream_qty * ice_cream_price

-- Theorem Statement
theorem total_cost_calculation : total_cost = 51 := by
  sorry

end total_cost_calculation_l186_186426


namespace calc_expr_l186_186651

theorem calc_expr :
  (2 * Real.sqrt 2 - 1) ^ 2 + (1 + Real.sqrt 3) * (1 - Real.sqrt 3) = 7 - 4 * Real.sqrt 2 :=
by
  sorry

end calc_expr_l186_186651


namespace adam_has_23_tattoos_l186_186183

-- Conditions as definitions
def tattoos_on_each_of_jason_arms := 2
def number_of_jason_arms := 2
def tattoos_on_each_of_jason_legs := 3
def number_of_jason_legs := 2

def jason_total_tattoos : Nat :=
  tattoos_on_each_of_jason_arms * number_of_jason_arms + tattoos_on_each_of_jason_legs * number_of_jason_legs

def adam_tattoos (jason_tattoos : Nat) : Nat :=
  2 * jason_tattoos + 3

-- The main theorem to be proved
theorem adam_has_23_tattoos : adam_tattoos jason_total_tattoos = 23 := by
  sorry

end adam_has_23_tattoos_l186_186183


namespace negation_of_universal_proposition_l186_186960

theorem negation_of_universal_proposition :
  (¬ (∀ x : ℝ, x^2 ≥ 0)) ↔ (∃ x_0 : ℝ, x_0^2 < 0) := sorry

end negation_of_universal_proposition_l186_186960


namespace am_gm_inequality_l186_186517

theorem am_gm_inequality (x y z : ℝ) (n : ℕ) (h_pos : x > 0 ∧ y > 0 ∧ z > 0) (h_sum : x + y + z = 1) (h_pos_n : n > 0):
  x^n + y^n + z^n ≥ 1 / 3^(n-1) :=
by
  sorry

end am_gm_inequality_l186_186517


namespace tan_neg_five_pi_div_four_l186_186508

theorem tan_neg_five_pi_div_four : Real.tan (- (5 * Real.pi / 4)) = -1 := 
sorry

end tan_neg_five_pi_div_four_l186_186508


namespace tan_alpha_minus_2beta_l186_186664

variables (α β : ℝ)

-- Given conditions
def tan_alpha_minus_beta : ℝ := 2 / 5
def tan_beta : ℝ := 1 / 2

-- The statement to prove
theorem tan_alpha_minus_2beta (h1 : tan (α - β) = tan_alpha_minus_beta) (h2 : tan β = tan_beta) :
  tan (α - 2 * β) = -1 / 12 :=
sorry

end tan_alpha_minus_2beta_l186_186664


namespace probability_of_rolling_2_4_or_6_l186_186619

theorem probability_of_rolling_2_4_or_6 :
  let outcomes := ({1, 2, 3, 4, 5, 6} : Finset ℕ)
      favorable_outcomes := ({2, 4, 6} : Finset ℕ)
  in 
  (favorable_outcomes.card : ℚ) / (outcomes.card : ℚ) = 1 / 2 := by
  sorry

end probability_of_rolling_2_4_or_6_l186_186619


namespace symmetrical_point_with_respect_to_x_axis_l186_186708

-- Define the point P with coordinates (-2, -1)
structure Point :=
(x : ℝ)
(y : ℝ)

-- Define the given point
def P : Point := { x := -2, y := -1 }

-- Define the symmetry with respect to the x-axis
def symmetry_x_axis (p : Point) : Point :=
{ x := p.x, y := -p.y }

-- Verify the symmetrical point
theorem symmetrical_point_with_respect_to_x_axis :
  symmetry_x_axis P = { x := -2, y := 1 } :=
by
  -- Skip the proof
  sorry

end symmetrical_point_with_respect_to_x_axis_l186_186708


namespace Kindergarten_Students_l186_186277

theorem Kindergarten_Students (X : ℕ) (h1 : 40 * X + 40 * 10 + 40 * 11 = 1200) : X = 9 :=
by
  sorry

end Kindergarten_Students_l186_186277


namespace selling_price_to_equal_percentage_profit_and_loss_l186_186460

-- Definition of the variables and conditions
def cost_price : ℝ := 1500
def sp_profit_25 : ℝ := 1875
def sp_loss : ℝ := 1280

theorem selling_price_to_equal_percentage_profit_and_loss :
  ∃ SP : ℝ, SP = 1720.05 ∧
  (sp_profit_25 = cost_price * 1.25) ∧
  (sp_loss < cost_price) ∧
  (14.67 = ((SP - cost_price) / cost_price) * 100) ∧
  (14.67 = ((cost_price - sp_loss) / cost_price) * 100) :=
by
  sorry

end selling_price_to_equal_percentage_profit_and_loss_l186_186460


namespace range_of_f_l186_186860

noncomputable def f (x : ℝ) : ℝ := sorry

axiom f_pos : ∀ x : ℝ, 0 < x → 0 < f(x)
axiom f_ineq : ∀ x : ℝ, 0 < x → f(x) < deriv (deriv f) x ∧ deriv (deriv f) x < 2 * f(x)

theorem range_of_f : (1 / Real.exp 2) < f 1 / f 2 ∧ f 1 / f 2 < (1 / Real.exp 1) :=
by sorry

end range_of_f_l186_186860


namespace sum_of_roots_of_cubic_eq_l186_186659

-- Define the cubic equation
def cubic_eq (x : ℝ) : ℝ := 3 * x^3 - 9 * x^2 - 72 * x + 6

-- Define the statement to prove
theorem sum_of_roots_of_cubic_eq : 
  ∀ (r p q : ℝ), (cubic_eq r = 0) ∧ (cubic_eq p = 0) ∧ (cubic_eq q = 0) → 
  (r + p + q) = 3 :=
sorry

end sum_of_roots_of_cubic_eq_l186_186659


namespace floor_area_l186_186315

theorem floor_area (length_feet : ℝ) (width_feet : ℝ) (feet_to_meters : ℝ) 
  (h_length : length_feet = 15) (h_width : width_feet = 10) (h_conversion : feet_to_meters = 0.3048) :
  let length_meters := length_feet * feet_to_meters
  let width_meters := width_feet * feet_to_meters
  let area_meters := length_meters * width_meters
  area_meters = 13.93 := 
by
  sorry

end floor_area_l186_186315


namespace complement_union_A_B_l186_186926

def is_element_of_set_A (x : ℤ) : Prop := ∃ k : ℤ, x = 3 * k + 1
def is_element_of_set_B (x : ℤ) : Prop := ∃ k : ℤ, x = 3 * k + 2
def is_element_of_complement_U (x : ℤ) : Prop := ∃ k : ℤ, x = 3 * k

theorem complement_union_A_B :
  {x : ℤ | ¬ (is_element_of_set_A x ∨ is_element_of_set_B x)} = {x : ℤ | is_element_of_complement_U x} :=
by
  sorry

end complement_union_A_B_l186_186926


namespace abs_b_lt_abs_a_lt_2abs_b_l186_186730

variable {a b : ℝ}

theorem abs_b_lt_abs_a_lt_2abs_b (h : (6 * a + 9 * b) / (a + b) < (4 * a - b) / (a - b)) :
  |b| < |a| ∧ |a| < 2 * |b| :=
sorry

end abs_b_lt_abs_a_lt_2abs_b_l186_186730


namespace kara_total_water_intake_l186_186553

/--
Kara has to drink 4 ounces of water every time she takes her medication.
Her medication instructions are to take one tablet three times a day.
She followed the instructions for one week.
In the second week, she forgot twice on one day.
How many ounces of water did she drink with her medication over those two weeks?
--/
theorem kara_total_water_intake : 
  ∀ (water_per_medication : ℕ) (medication_per_day : ℕ) (days_per_week : ℕ) 
  (forgotten_days : ℕ) (missed_medications : ℕ),
  water_per_medication = 4 →
  medication_per_day = 3 →
  days_per_week = 7 →
  forgotten_days = 1 →
  missed_medications = 2 →
  ((medication_per_day * days_per_week * water_per_medication) +
  ((medication_per_day * days_per_week - missed_medications) * water_per_medication)) = 160 :=
by
  intros water_per_medication medication_per_day days_per_week forgotten_days missed_medications
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h5]
  sorry

end kara_total_water_intake_l186_186553


namespace scientific_notation_correct_l186_186273

-- Define the conditions
def circumference_circular_orbit_chinese_space_station := 43000000

-- Define the target scientific notation representation
def scientific_notation_representation := 4.3 * 10 ^ 7

-- State the problem: Prove that the given number in scientific notation is equal to the circumference
theorem scientific_notation_correct : 
  circumference_circular_orbit_chinese_space_station = scientific_notation_representation :=
by
  sorry

end scientific_notation_correct_l186_186273


namespace solve_for_a_l186_186686

theorem solve_for_a (a : ℚ) (h : a + a / 3 = 8 / 3) : a = 2 :=
sorry

end solve_for_a_l186_186686


namespace clock_hand_speed_ratio_l186_186943

theorem clock_hand_speed_ratio :
  (360 / 720 : ℝ) / (360 / 60 : ℝ) = (2 / 24 : ℝ) := by
    sorry

end clock_hand_speed_ratio_l186_186943


namespace Suzanna_bike_distance_l186_186114

theorem Suzanna_bike_distance (ride_rate distance_time total_time : ℕ)
  (constant_rate : ride_rate = 3) (time_interval : distance_time = 10)
  (total_riding_time : total_time = 40) :
  (total_time / distance_time) * ride_rate = 12 :=
by
  -- Assuming the conditions:
  -- ride_rate = 3
  -- distance_time = 10
  -- total_time = 40
  sorry

end Suzanna_bike_distance_l186_186114


namespace age_ratio_l186_186825

theorem age_ratio (A B C : ℕ) (h1 : A = B + 2) (h2 : A + B + C = 27) (h3 : B = 10) : B / C = 2 :=
by
  sorry

end age_ratio_l186_186825


namespace lateral_surface_area_eq_total_surface_area_eq_l186_186029

def r := 3
def h := 10

theorem lateral_surface_area_eq : 2 * Real.pi * r * h = 60 * Real.pi := by
  sorry

theorem total_surface_area_eq : 2 * Real.pi * r * h + 2 * Real.pi * r^2 = 78 * Real.pi := by
  sorry

end lateral_surface_area_eq_total_surface_area_eq_l186_186029


namespace number_of_adults_l186_186487

theorem number_of_adults (A C S : ℕ) (h1 : C = A - 35) (h2 : S = 2 * C) (h3 : A + C + S = 127) : A = 58 :=
by
  sorry

end number_of_adults_l186_186487


namespace sum_of_consecutive_odd_integers_l186_186598

-- Define the conditions
def consecutive_odd_integers (n : ℤ) : List ℤ :=
  [n, n + 2, n + 4]

def sum_first_and_third_eq_150 (n : ℤ) : Prop :=
  n + (n + 4) = 150

-- Proof to show that the sum of these integers is 225
theorem sum_of_consecutive_odd_integers (n : ℤ) (h : sum_first_and_third_eq_150 n) :
  consecutive_odd_integers n).sum = 225 :=
  sorry

end sum_of_consecutive_odd_integers_l186_186598


namespace find_constants_exist_l186_186368

theorem find_constants_exist :
  ∃ A B C, (∀ x, 4 * x / ((x - 5) * (x - 3)^2) = A / (x - 5) + B / (x - 3) + C / (x - 3)^2)
  ∧ (A = 5) ∧ (B = -5) ∧ (C = -6) := 
sorry

end find_constants_exist_l186_186368


namespace activity_popularity_order_l186_186589

theorem activity_popularity_order
  (dodgeball : ℚ := 13 / 40)
  (picnic : ℚ := 9 / 30)
  (swimming : ℚ := 7 / 20)
  (crafts : ℚ := 3 / 15) :
  (swimming > dodgeball ∧ dodgeball > picnic ∧ picnic > crafts) :=
by 
  sorry

end activity_popularity_order_l186_186589


namespace problem_condition_neither_sufficient_nor_necessary_l186_186687

theorem problem_condition_neither_sufficient_nor_necessary 
  (m n : ℕ) (hm : m > 0) (hn : n > 0) (a b : ℝ) :
  (a > b → a^(m + n) + b^(m + n) > a^n * b^m + a^m * b^n) ∧
  (a^(m + n) + b^(m + n) > a^n * b^m + a^m * b^n → a > b) = false :=
by sorry

end problem_condition_neither_sufficient_nor_necessary_l186_186687


namespace largest_divisor_of_product_of_5_consecutive_integers_l186_186785

/-- What is the largest integer that must divide the product of any 5 consecutive integers? -/
theorem largest_divisor_of_product_of_5_consecutive_integers :
  ∀ n : ℤ, ∃ d : ℤ, d = 24 ∧ d ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  sorry

end largest_divisor_of_product_of_5_consecutive_integers_l186_186785


namespace least_N_no_square_l186_186203

theorem least_N_no_square (N : ℕ) : 
  (∀ k, (1000 * N) ≤ k ∧ k ≤ (1000 * N + 999) → 
  ∃ m, ¬ (k = m^2)) ↔ N = 282 :=
by
  sorry

end least_N_no_square_l186_186203


namespace largest_divisor_of_product_of_five_consecutive_integers_l186_186768

theorem largest_divisor_of_product_of_five_consecutive_integers
  (a b c d e : ℤ) 
  (h: a + 1 = b ∧ b + 1 = c ∧ c + 1 = d ∧ d + 1 = e):
  ∃ (n : ℤ), n = 60 ∧ n ∣ (a * b * c * d * e) :=
sorry

end largest_divisor_of_product_of_five_consecutive_integers_l186_186768


namespace average_billboards_per_hour_l186_186945

-- Define the number of billboards seen in each hour
def billboards_first_hour := 17
def billboards_second_hour := 20
def billboards_third_hour := 23

-- Define the number of hours
def total_hours := 3

-- Prove that the average number of billboards per hour is 20
theorem average_billboards_per_hour : 
  (billboards_first_hour + billboards_second_hour + billboards_third_hour) / total_hours = 20 :=
by
  sorry

end average_billboards_per_hour_l186_186945


namespace students_in_either_but_not_both_l186_186849

-- Definitions and conditions
def both : ℕ := 18
def geom : ℕ := 35
def only_stats : ℕ := 16

-- Correct answer to prove
def total_not_both : ℕ := geom - both + only_stats

theorem students_in_either_but_not_both : total_not_both = 33 := by
  sorry

end students_in_either_but_not_both_l186_186849


namespace concyclicity_equivalence_l186_186248

variables {A B C D H N O' : Point} {a b c R : Real}
variables (triangle_ABC : Triangle A B C) 
variables (orthocenter_H : Orthocenter H triangle_ABC)
variables (circumcenter_O_prime : Circumcenter O' (triangle.triangle_of_orthocircle_triangle orthocenter_H triangle_ABC))
variables (midpoint_N : Midpoint N (segment (line_segment A O')))
variables (reflection_D : Reflection D N (line (segment B C)))
variables (circumradius_R : CircumcircleRadius R triangle_ABC)

theorem concyclicity_equivalence :
  (Cyclic_Quad A B D C) ↔ (b^2 + c^2 - a^2 = 3 * R^2) := 
sorry

end concyclicity_equivalence_l186_186248


namespace complement_union_eq_complement_l186_186906

open Set

section ComplementUnion

variable (k : ℤ)

def SetA : Set ℤ := {x | ∃ k : ℤ, x = 3 * k + 1}
def SetB : Set ℤ := {x | ∃ k : ℤ, x = 3 * k + 2}
def UniversalSet : Set ℤ := univ
def ComplementUnion : Set ℤ := {x | ∃ k : ℤ, x = 3 * k}

theorem complement_union_eq_complement :
  UniversalSet \ (SetA ∪ SetB) = ComplementUnion :=
by
  sorry

end ComplementUnion

end complement_union_eq_complement_l186_186906


namespace arithmetic_identity_l186_186650

theorem arithmetic_identity : 15 * 30 + 45 * 15 - 15 * 10 = 975 :=
by
  sorry

end arithmetic_identity_l186_186650


namespace series_sum_eq_l186_186190

theorem series_sum_eq :
  (1^25 + 2^24 + 3^23 + 4^22 + 5^21 + 6^20 + 7^19 + 8^18 + 9^17 + 10^16 + 
  11^15 + 12^14 + 13^13 + 14^12 + 15^11 + 16^10 + 17^9 + 18^8 + 19^7 + 20^6 + 
  21^5 + 22^4 + 23^3 + 24^2 + 25^1) = 66071772829247409 := 
by
  sorry

end series_sum_eq_l186_186190


namespace determine_abc_l186_186168

-- Definitions
def parabola_equation (a b c : ℝ) (y : ℝ) : ℝ := a * y^2 + b * y + c

def vertex_condition (a b c : ℝ) : Prop :=
  ∀ y, parabola_equation a b c y = a * (y + 6)^2 + 3

def point_condition (a b c : ℝ) : Prop :=
  parabola_equation a b c (-6) = 3 ∧ parabola_equation a b c (-4) = 2

-- Proposition to prove
theorem determine_abc : 
  ∃ a b c : ℝ, vertex_condition a b c ∧ point_condition a b c
  ∧ (a + b + c = -25/4) :=
sorry

end determine_abc_l186_186168


namespace find_ice_cream_cost_l186_186032

def chapatis_cost (num: ℕ) (price: ℝ) : ℝ := num * price
def rice_cost (num: ℕ) (price: ℝ) : ℝ := num * price
def mixed_vegetable_cost (num: ℕ) (price: ℝ) : ℝ := num * price
def soup_cost (num: ℕ) (price: ℝ) : ℝ := num * price
def dessert_cost (num: ℕ) (price: ℝ) : ℝ := num * price
def soft_drink_cost (num: ℕ) (price: ℝ) (discount: ℝ) : ℝ := num * price * (1 - discount)
def total_cost (chap: ℝ) (rice: ℝ) (veg: ℝ) (soup: ℝ) (dessert: ℝ) (drink: ℝ) : ℝ := chap + rice + veg + soup + dessert + drink
def total_cost_with_tax (base_cost: ℝ) (tax_rate: ℝ) : ℝ := base_cost * (1 + tax_rate)

theorem find_ice_cream_cost :
  let chapatis := chapatis_cost 16 6
  let rice := rice_cost 5 45
  let veg := mixed_vegetable_cost 7 70
  let soup := soup_cost 4 30
  let dessert := dessert_cost 3 85
  let drinks := soft_drink_cost 2 50 0.1
  let base_cost := total_cost chapatis rice veg soup dessert drinks
  let final_cost := total_cost_with_tax base_cost 0.18
  final_cost + 6 * 108.89 = 2159 := 
  by sorry

end find_ice_cream_cost_l186_186032


namespace total_sweaters_l186_186941

-- Define the conditions
def washes_per_load : ℕ := 9
def total_shirts : ℕ := 19
def total_loads : ℕ := 3

-- Define the total_sweaters theorem to prove Nancy had to wash 9 sweaters
theorem total_sweaters {n : ℕ} (h1 : washes_per_load = 9) (h2 : total_shirts = 19) (h3 : total_loads = 3) : n = 9 :=
by
  sorry

end total_sweaters_l186_186941


namespace sequence_polynomial_exists_l186_186524

noncomputable def sequence_exists (k : ℕ) : Prop :=
∃ u : ℕ → ℝ,
  (∀ n : ℕ, u (n + 1) - u n = (n : ℝ) ^ k) ∧
  (∃ p : Polynomial ℝ, (∀ n : ℕ, u n = Polynomial.eval (n : ℝ) p) ∧ p.degree = k + 1 ∧ p.leadingCoeff = 1 / (k + 1))

theorem sequence_polynomial_exists (k : ℕ) : sequence_exists k :=
sorry

end sequence_polynomial_exists_l186_186524


namespace largest_divisor_of_product_of_5_consecutive_integers_l186_186782

/-- What is the largest integer that must divide the product of any 5 consecutive integers? -/
theorem largest_divisor_of_product_of_5_consecutive_integers :
  ∀ n : ℤ, ∃ d : ℤ, d = 24 ∧ d ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  sorry

end largest_divisor_of_product_of_5_consecutive_integers_l186_186782


namespace sum_series_equals_half_l186_186501

theorem sum_series_equals_half :
  ∑' n, 1 / (n * (n+1) * (n+2)) = 1 / 2 :=
sorry

end sum_series_equals_half_l186_186501


namespace product_of_five_consecutive_divisible_by_30_l186_186778

theorem product_of_five_consecutive_divisible_by_30 :
  ∀ n : ℤ, 30 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  sorry

end product_of_five_consecutive_divisible_by_30_l186_186778


namespace percentage_reduction_l186_186143

-- Define the problem within given conditions
def original_length := 30 -- original length in seconds
def new_length := 21 -- new length in seconds

-- State the theorem that needs to be proved
theorem percentage_reduction (original_length new_length : ℕ) : 
  original_length = 30 → 
  new_length = 21 → 
  ((original_length - new_length) / original_length: ℚ) * 100 = 30 :=
by 
  sorry

end percentage_reduction_l186_186143


namespace real_part_zero_implies_x3_l186_186694

theorem real_part_zero_implies_x3 (x : ℝ) : 
  (x^2 - 2*x - 3 = 0) ∧ (x + 1 ≠ 0) → x = 3 :=
by
  sorry

end real_part_zero_implies_x3_l186_186694


namespace sum_single_digits_l186_186709

theorem sum_single_digits (P Q R : ℕ) (hP : P ≠ Q) (hQ : Q ≠ R) (hR : R ≠ P)
  (h1 : R + R = 10)
  (h_sum : ∃ (P Q R : ℕ), P * 100 + 70 + R + 390 + R = R * 100 + Q * 10) :
  P + Q + R = 13 := 
sorry

end sum_single_digits_l186_186709


namespace village_population_l186_186627

theorem village_population (P : ℝ) (h1 : 0.08 * P = 4554) : P = 6325 :=
by
  sorry

end village_population_l186_186627


namespace identify_tricksters_in_30_or_less_questions_l186_186321

-- Define the problem parameters
def inhabitants : Type := Fin 65

def is_knight (inhabitant : inhabitants) : Prop := sorry
def is_trickster (inhabitant : inhabitants) : Prop := sorry

-- Define the properties
axiom knight_truthful : ∀ (x : inhabitants), is_knight x → (forall y : inhabitants, True ↔ (is_knight y = x is_knight y))
axiom trickster_mixed : ∀ (x : inhabitants), is_trickster x → ((∀ y : inhabitants, True) ∨ (∃ y : inhabitants, y ∉ (is_knight y)))

-- Problem statement
theorem identify_tricksters_in_30_or_less_questions
  (inhabitants : Type)
  (n_tricksters : ℕ := 2) -- 2 tricksters
  (total_inhabitants : ℕ := 65) -- 65 total inhabitants
  (questions_limit : ℕ := 30) -- limit of 30 questions
  (knights : inhabitants → Prop)
  (tricksters : inhabitants → Prop) :
    ∃ (solution_exists : ∀ (is_trickster : inhabitants → Prop), ∃ k : inhabitants, (knights k) ∧ (is_trickster k)) 
    (possible_to_find_tricksters : ∀ (is_knight : inhabitants → Prop) (is_trickster : inhabitants → Prop), 
    ∃ (questions_used ≤ questions_limit), ∀ (xs : set inhabitants), questions_used ≤ 30 ∧ 
    (∃ trickster1 trickster2 : inhabitants, (tricksters trickster1 ∧ tricksters trickster2 ∧ trickster1 ≠ trickster2))) :=
sorry

end identify_tricksters_in_30_or_less_questions_l186_186321


namespace sum_S11_l186_186933

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}
variable {a1 d : ℝ}

axiom arithmetic_sequence (n : ℕ) : a n = a1 + (n - 1) * d
axiom sum_of_first_n_terms (n : ℕ) : S n = n / 2 * (a 1 + a n)
axiom condition : a 3 + 4 = a 2 + a 7

theorem sum_S11 : S 11 = 44 := by
  sorry

end sum_S11_l186_186933


namespace ratio_of_arithmetic_sequences_l186_186682

-- Definitions for the conditions
variables {a_n b_n : ℕ → ℝ}
variables {S_n T_n : ℕ → ℝ}
variables (d_a d_b : ℝ)

-- Arithmetic sequences conditions
def is_arithmetic_sequence (u_n : ℕ → ℝ) (t : ℝ) (d : ℝ) : Prop :=
  ∀ (n : ℕ), u_n n = t + n * d

-- Sum of first n terms conditions
def sum_of_first_n_terms (u_n : ℕ → ℝ) (Sn : ℕ → ℝ) : Prop :=
  ∀ (n : ℕ), Sn n = n * (u_n 1 + u_n (n-1)) / 2

-- Main theorem statement
theorem ratio_of_arithmetic_sequences (h1 : is_arithmetic_sequence a_n (a_n 0) d_a)
                                     (h2 : is_arithmetic_sequence b_n (b_n 0) d_b)
                                     (h3 : sum_of_first_n_terms a_n S_n)
                                     (h4 : sum_of_first_n_terms b_n T_n)
                                     (h5 : ∀ n, (S_n n) / (T_n n) = (2 * n) / (3 * n + 1)) :
                                     ∀ n, (a_n n) / (b_n n) = (2 * n - 1) / (3 * n - 1) := sorry

end ratio_of_arithmetic_sequences_l186_186682


namespace min_value_reciprocal_sum_l186_186864

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hmean : (a + b) / 2 = 1 / 2) : 
  ∃ c, c = (1 / a + 1 / b) ∧ c ≥ 4 := 
sorry

end min_value_reciprocal_sum_l186_186864


namespace Sara_spent_on_each_movie_ticket_l186_186098

def Sara_spent_on_each_movie_ticket_correct : Prop :=
  let T := 36.78
  let R := 1.59
  let B := 13.95
  (T - R - B) / 2 = 10.62

theorem Sara_spent_on_each_movie_ticket : 
  Sara_spent_on_each_movie_ticket_correct :=
by
  sorry

end Sara_spent_on_each_movie_ticket_l186_186098


namespace sequence_1234_to_500_not_divisible_by_9_l186_186893

-- Definition for the sum of the digits of concatenated sequence
def sum_of_digits (n : ℕ) : ℕ :=
  -- This is a placeholder for the actual function calculating the sum of digits
  -- of all numbers from 1 to n concatenated together.
  sorry 

def is_divisible_by_9 (n : ℕ) : Prop :=
  n % 9 = 0

theorem sequence_1234_to_500_not_divisible_by_9 : ¬ is_divisible_by_9 (sum_of_digits 500) :=
by
  -- Placeholder indicating the solution facts and methods should go here.
  sorry

end sequence_1234_to_500_not_divisible_by_9_l186_186893


namespace slope_of_line_inclination_angle_l186_186280

theorem slope_of_line_inclination_angle 
  (k : ℝ) (θ : ℝ)
  (hθ1 : 30 * (π / 180) < θ)
  (hθ2 : θ < 90 * (π / 180)) :
  k = Real.tan θ → k > Real.tan (30 * (π / 180)) :=
by
  intro h
  sorry

end slope_of_line_inclination_angle_l186_186280


namespace complement_union_eq_complement_l186_186905

open Set

section ComplementUnion

variable (k : ℤ)

def SetA : Set ℤ := {x | ∃ k : ℤ, x = 3 * k + 1}
def SetB : Set ℤ := {x | ∃ k : ℤ, x = 3 * k + 2}
def UniversalSet : Set ℤ := univ
def ComplementUnion : Set ℤ := {x | ∃ k : ℤ, x = 3 * k}

theorem complement_union_eq_complement :
  UniversalSet \ (SetA ∪ SetB) = ComplementUnion :=
by
  sorry

end ComplementUnion

end complement_union_eq_complement_l186_186905


namespace profit_per_tire_l186_186633

theorem profit_per_tire
  (fixed_cost : ℝ)
  (variable_cost_per_tire : ℝ)
  (selling_price_per_tire : ℝ)
  (batch_size : ℕ)
  (total_cost : ℝ)
  (total_revenue : ℝ)
  (total_profit : ℝ)
  (profit_per_tire : ℝ)
  (h1 : fixed_cost = 22500)
  (h2 : variable_cost_per_tire = 8)
  (h3 : selling_price_per_tire = 20)
  (h4 : batch_size = 15000)
  (h5 : total_cost = fixed_cost + variable_cost_per_tire * batch_size)
  (h6 : total_revenue = selling_price_per_tire * batch_size)
  (h7 : total_profit = total_revenue - total_cost)
  (h8 : profit_per_tire = total_profit / batch_size) :
  profit_per_tire = 10.50 :=
sorry

end profit_per_tire_l186_186633


namespace parabola_symmetry_product_l186_186889

theorem parabola_symmetry_product (a p m : ℝ) 
  (hpr1 : a ≠ 0) 
  (hpr2 : p > 0) 
  (hpr3 : ∀ (x₀ y₀ : ℝ), y₀^2 = 2*p*x₀ → (a*(y₀ - m)^2 - 3*(y₀ - m) + 3 = x₀ + m)) :
  a * p * m = -3 := 
sorry

end parabola_symmetry_product_l186_186889


namespace projection_y_is_closed_of_closed_and_bounded_projection_x_l186_186575

open Set Filter

variable (S : Set (ℝ × ℝ))

def projection_x (S : Set (ℝ × ℝ)) : Set ℝ :=
  { x | ∃ y, (x, y) ∈ S }

def projection_y (S : Set (ℝ × ℝ)) : Set ℝ :=
  { y | ∃ x, (x, y) ∈ S }

theorem projection_y_is_closed_of_closed_and_bounded_projection_x
  (hS : IsClosed S)
  (hX : Bounded (projection_x S)) :
  IsClosed (projection_y S) :=
sorry

end projection_y_is_closed_of_closed_and_bounded_projection_x_l186_186575


namespace product_of_five_consecutive_divisible_by_30_l186_186775

theorem product_of_five_consecutive_divisible_by_30 :
  ∀ n : ℤ, 30 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  sorry

end product_of_five_consecutive_divisible_by_30_l186_186775


namespace wilson_hamburgers_l186_186623

def hamburger_cost (H : ℕ) := 5 * H
def cola_cost := 6
def discount := 4
def total_cost (H : ℕ) := hamburger_cost H + cola_cost - discount

theorem wilson_hamburgers (H : ℕ) (h : total_cost H = 12) : H = 2 :=
sorry

end wilson_hamburgers_l186_186623


namespace min_value_of_f_l186_186685

noncomputable def f (x : ℝ) : ℝ := (1 / x) + (2 * x / (1 - x))

theorem min_value_of_f (x : ℝ) (h1 : 0 < x) (h2 : x < 1) : 
  (∀ y, 0 < y ∧ y < 1 → f y ≥ 1 + 2 * Real.sqrt 2) := 
sorry

end min_value_of_f_l186_186685


namespace total_fruits_in_baskets_l186_186173

theorem total_fruits_in_baskets : 
  let apples := [9, 9, 9, 7]
    oranges := [15, 15, 15, 13]
    bananas := [14, 14, 14, 12]
    fruits := apples.zipWith (· + ·) oranges |>.zipWith (· + ·) bananas in
  (fruits.foldl (· + ·) 0) = 146 :=
by
  sorry

end total_fruits_in_baskets_l186_186173


namespace cone_base_diameter_l186_186171

theorem cone_base_diameter (l r : ℝ) 
  (h1 : (1/2) * π * l^2 + π * r^2 = 3 * π) 
  (h2 : π * l = 2 * π * r) : 2 * r = 2 :=
by
  sorry

end cone_base_diameter_l186_186171


namespace fraction_study_japanese_l186_186981

theorem fraction_study_japanese (J S : ℕ) (hS : S = 2 * J)
  (hS_japanese : rat.of_int S * (1 / 8) = rat.of_int S * 1 / 8)
  (hJ_japanese : rat.of_int J * (3 / 4) = rat.of_int J * 3 / 4) : 
  (rat.of_int (((1 / 8) * S) + ((3 / 4) * J))) / (rat.of_int (S + J)) = 1 / 3 := by
  sorry

end fraction_study_japanese_l186_186981


namespace expression_eval_l186_186648

theorem expression_eval :
  5 * 399 + 4 * 399 + 3 * 399 + 397 = 5185 :=
by
  sorry

end expression_eval_l186_186648


namespace original_number_divisible_l186_186618

theorem original_number_divisible (n : ℕ) (h : (n - 8) % 20 = 0) : n = 28 := 
by
  sorry

end original_number_divisible_l186_186618


namespace age_6_not_child_l186_186566

-- Definition and assumptions based on the conditions
def billboard_number : ℕ := 5353
def mr_smith_age : ℕ := 53
def children_ages : List ℕ := [1, 2, 3, 4, 5, 7, 8, 9, 10, 11] -- Excluding age 6

-- The theorem to prove that the age 6 is not one of Mr. Smith's children's ages.
theorem age_6_not_child :
  (billboard_number ≡ 53 * 101 [MOD 10^4]) ∧
  (∀ age ∈ children_ages, billboard_number % age = 0) ∧
  oldest_child_age = 11 → ¬(6 ∈ children_ages) :=
sorry

end age_6_not_child_l186_186566


namespace inequality_proof_l186_186296

variables {n : ℕ} {a : Fin n → ℝ}

-- Conditions:
axiom distinct_positive (h : ∀ i j : Fin n, i ≠ j → a i ≠ a j) (h' : ∀ k : Fin n, a k > 0)
axiom n_ge_two (h_n : 2 ≤ n)
axiom sum_inverse_2n (h_sum : ∑ k in Finset.univ, (a k) ^ (-2 * n) = 1)

-- Problem statement:
theorem inequality_proof :
  ∑ k in Finset.univ, (a k) ^ (2 * n)
  - n^2 * ∑ i in Finset.univ, ∑ j in Finset.filter (λ j, i < j) Finset.univ,
            ((a i / a j) - (a j / a i))^2 > n^2 :=
sorry

end inequality_proof_l186_186296


namespace expected_gold_coins_l186_186569

theorem expected_gold_coins :
  let E : ℕ → ℝ := λ n, if n = 0 then 1 else (1 / 2) * 0 + (1 / 2) * (1 + E n) in
  E 0 = 1 :=
by
  sorry

end expected_gold_coins_l186_186569


namespace integral_value_l186_186734

theorem integral_value (a : ℝ) (h : a = 2) : ∫ x in a..2*Real.exp 1, 1/x = 1 := by
  sorry

end integral_value_l186_186734


namespace rectangle_area_l186_186083

structure Rectangle (A B C D : Type) :=
(ab : ℝ)
(ac : ℝ)
(right_angle : ∃ (B B' : Type), B ≠ B' ∧ ac = ab + (ab ^ 2 + (B - B') ^ 2)^0.5)
(ab_value : ab = 15)
(ac_value : ac = 17)

noncomputable def area_ABCD : ℝ :=
have bc := ((ac ^ 2) - (ab ^ 2))^0.5,
ab * bc

theorem rectangle_area {A B C D : Type} (r : Rectangle A B C D) : r.ab * ((r.ac ^ 2 - r.ab ^ 2)^0.5) = 120 :=
by
  calc
    r.ab * ((r.ac ^ 2 - r.ab ^ 2)^0.5)
        = 15 * ((17 ^ 2 - 15 ^ 2)^0.5) : by { simp only [r.ab_value, r.ac_value] }
    ... = 15 * (64^0.5) : by { norm_num }
    ... = 15 * 8 : by { norm_num }
    ... = 120 : by { norm_num }

end rectangle_area_l186_186083


namespace complement_union_A_B_l186_186922

def A := {x : ℤ | ∃ k : ℤ, x = 3 * k + 1}
def B := {x : ℤ | ∃ k : ℤ, x = 3 * k + 2}
def U := {x : ℤ | True}

theorem complement_union_A_B:
  (U \ (A ∪ B) = {x : ℤ | ∃ k : ℤ, x = 3 * k}) :=
by
  sorry

end complement_union_A_B_l186_186922


namespace part1_part2_l186_186669

-- Definitions for Part (1)
def A : Set ℝ := { x | -2 ≤ x ∧ x ≤ 4 }
def B (m : ℝ) : Set ℝ := { x | -1 ≤ x ∧ x ≤ 3 }

-- Part (1) Statement
theorem part1 (m : ℝ) (hm : m = 2) : A ∩ ((compl B m)) = {x | (-2 ≤ x ∧ x < -1) ∨ (3 < x ∧ x ≤ 4)} := 
by
  sorry

-- Definitions for Part (2)
def B_interval (m : ℝ) : Set ℝ := { x | (1 - m) ≤ x ∧ x ≤ (1 + m) }

-- Part (2) Statement
theorem part2 (m : ℝ) (h : ∀ x, (x ∈ A → x ∈ B_interval m)) : 0 < m ∧ m < 3 := 
by
  sorry

end part1_part2_l186_186669


namespace lice_checks_l186_186463

theorem lice_checks (t_first t_second t_third t_total t_per_check n_first n_second n_third n_total n_per_check n_kg : ℕ) 
 (h1 : t_first = 19 * t_per_check)
 (h2 : t_second = 20 * t_per_check)
 (h3 : t_third = 25 * t_per_check)
 (h4 : t_total = 3 * 60)
 (h5 : t_per_check = 2)
 (h6 : n_first = t_first / t_per_check)
 (h7 : n_second = t_second / t_per_check)
 (h8 : n_third = t_third / t_per_check)
 (h9 : n_total = (t_total - (t_first + t_second + t_third)) / t_per_check) :
 n_total = 26 :=
sorry

end lice_checks_l186_186463


namespace cubic_has_one_real_root_l186_186008

theorem cubic_has_one_real_root :
  (∃ x : ℝ, x^3 - 6*x^2 + 9*x - 10 = 0) ∧ ∀ x y : ℝ, (x^3 - 6*x^2 + 9*x - 10 = 0) ∧ (y^3 - 6*y^2 + 9*y - 10 = 0) → x = y :=
by
  sorry

end cubic_has_one_real_root_l186_186008


namespace expression_value_l186_186475

theorem expression_value : (4 - 2) ^ 3 = 8 :=
by sorry

end expression_value_l186_186475


namespace intersection_set_l186_186392

-- Definition of the sets A and B
def setA : Set ℝ := { x | -2 < x ∧ x < 2 }
def setB : Set ℝ := { x | x < 0.5 }

-- The main theorem: Finding the intersection A ∩ B
theorem intersection_set : { x : ℝ | -2 < x ∧ x < 0.5 } = setA ∩ setB := by
  sorry

end intersection_set_l186_186392


namespace harmon_high_school_proof_l186_186034

noncomputable def harmon_high_school : Prop :=
  ∃ (total_players players_physics players_both players_chemistry : ℕ),
    total_players = 18 ∧
    players_physics = 10 ∧
    players_both = 3 ∧
    players_chemistry = (total_players - players_physics + players_both)

theorem harmon_high_school_proof : harmon_high_school :=
  sorry

end harmon_high_school_proof_l186_186034


namespace geometric_sequence_third_term_and_sum_l186_186126

noncomputable def geometric_sequence (b1 r : ℝ) (n : ℕ) : ℝ :=
  b1 * r^(n - 1)

theorem geometric_sequence_third_term_and_sum (b2 b5 : ℝ) (h1 : b2 = 24.5) (h2 : b5 = 196) :
  (∃ b1 r : ℝ, r ≠ 0 ∧ geometric_sequence b1 r 2 = b2 ∧ geometric_sequence b1 r 5 = b5 ∧
  geometric_sequence b1 r 3 = 49 ∧
  b1 * (r^4 - 1) / (r - 1) = 183.75) :=
by sorry

end geometric_sequence_third_term_and_sum_l186_186126


namespace inequality_always_true_l186_186005

theorem inequality_always_true (a : ℝ) : (∀ x : ℝ, |x - 1| - |x + 2| ≤ a) ↔ 3 ≤ a :=
by
  sorry

end inequality_always_true_l186_186005


namespace fraction_zero_implies_value_l186_186236

theorem fraction_zero_implies_value (x : ℝ) (h : (|x| - 2) / (x + 2) = 0) (h_non_zero : x + 2 ≠ 0) : x = 2 :=
sorry

end fraction_zero_implies_value_l186_186236


namespace problem1_part1_problem1_part2_l186_186665

theorem problem1_part1 (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) : 
  (a + b + c) * (a^2 + b^2 + c^2) ≤ 3 * (a^3 + b^3 + c^3) := 
sorry

theorem problem1_part2 (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) : 
  (a / (b + c) + b / (c + a) + c / (a + b)) ≥ 3 / 2 := 
sorry

end problem1_part1_problem1_part2_l186_186665


namespace unique_maximizing_line_l186_186702

-- Define the points A, B, and C in a Euclidean plane
variables {A B C : EuclideanGeometry.Point}

-- Define the property of a line maximizing the product of distances to A and B
def maximizing_line (A B C : EuclideanGeometry.Point) (L : EuclideanGeometry.Line) : Prop :=
  ∀ L', EuclideanGeometry.distance (L'.project A) (L.project A) * EuclideanGeometry.distance (L'.project B) (L.project B) ≤
       EuclideanGeometry.distance (L.project A) (L.project A) * EuclideanGeometry.distance (L.project B) (L.project B)

-- State the theorem regarding the uniqueness of such a line
theorem unique_maximizing_line (A B C : EuclideanGeometry.Point) :
  ∃! L : EuclideanGeometry.Line, maximizing_line A B C L := sorry

end unique_maximizing_line_l186_186702


namespace return_trip_time_l186_186025

variable {d p w : ℝ} -- Distance, plane's speed in calm air, wind speed

theorem return_trip_time (h1 : d = 75 * (p - w)) 
                         (h2 : d / (p + w) = d / p - 10) :
                         (d / (p + w) = 15 ∨ d / (p + w) = 50) :=
sorry

end return_trip_time_l186_186025


namespace sum_of_first_40_terms_l186_186212

variable {a : ℕ → ℤ}

-- Initial conditions
axiom a2 : a 2 = 2
axiom recurrence_relation : ∀ n : ℕ, a (n + 2) + (-1)^(n - 1) * a n = 1

-- Sum of first n terms
noncomputable def S (n : ℕ) : ℤ := ∑ i in Finset.range (n + 1), a i

-- Proof that S 40 = 240
theorem sum_of_first_40_terms : S 40 = 240 :=
by sorry

end sum_of_first_40_terms_l186_186212


namespace initial_amount_correct_l186_186551

-- Definitions
def spent_on_fruits : ℝ := 15.00
def left_to_spend : ℝ := 85.00
def initial_amount_given (spent: ℝ) (left: ℝ) : ℝ := spent + left

-- Theorem stating the problem
theorem initial_amount_correct :
  initial_amount_given spent_on_fruits left_to_spend = 100.00 :=
by
  sorry

end initial_amount_correct_l186_186551


namespace car_distance_l186_186160

variable (v_x v_y : ℝ) (Δt_x : ℝ) (d_x : ℝ)

theorem car_distance (h_vx : v_x = 35) (h_vy : v_y = 50) (h_Δt : Δt_x = 1.2)
  (h_dx : d_x = v_x * Δt_x):
  d_x + v_x * (d_x / (v_y - v_x)) = 98 := 
by sorry

end car_distance_l186_186160


namespace largest_divisor_of_product_of_five_consecutive_integers_l186_186764

theorem largest_divisor_of_product_of_five_consecutive_integers
  (a b c d e : ℤ) 
  (h: a + 1 = b ∧ b + 1 = c ∧ c + 1 = d ∧ d + 1 = e):
  ∃ (n : ℤ), n = 60 ∧ n ∣ (a * b * c * d * e) :=
sorry

end largest_divisor_of_product_of_five_consecutive_integers_l186_186764


namespace no_preimage_range_l186_186875

open Set

def f (x : ℝ) : ℝ := x^2 + 2 * x + 3

theorem no_preimage_range :
  { k : ℝ | ∀ x : ℝ, f x ≠ k } = Iio 2 := by
  sorry

end no_preimage_range_l186_186875


namespace polygon_sides_eq_seven_l186_186697

-- Given conditions:
def sum_interior_angles (n : ℕ) : ℝ := (n - 2) * 180
def sum_exterior_angles : ℝ := 360
def difference_in_angles (n : ℕ) : ℝ := sum_interior_angles n - sum_exterior_angles

-- Proof statement:
theorem polygon_sides_eq_seven (n : ℕ) (h : difference_in_angles n = 540) : n = 7 := sorry

end polygon_sides_eq_seven_l186_186697


namespace Moscow_Olympiad_1958_problem_l186_186572

theorem Moscow_Olympiad_1958_problem :
  ∀ n : ℤ, 1155 ^ 1958 + 34 ^ 1958 ≠ n ^ 2 := 
by 
  sorry

end Moscow_Olympiad_1958_problem_l186_186572


namespace sum_of_three_consecutive_odd_integers_l186_186615

theorem sum_of_three_consecutive_odd_integers (a : ℤ) (h₁ : a + (a + 4) = 150) :
  (a + (a + 2) + (a + 4) = 225) :=
by
  sorry

end sum_of_three_consecutive_odd_integers_l186_186615


namespace percentage_profit_double_price_l186_186158

theorem percentage_profit_double_price (C S1 S2 : ℝ) (h1 : S1 = 1.5 * C) (h2 : S2 = 2 * S1) : 
  ((S2 - C) / C) * 100 = 200 := by
  sorry

end percentage_profit_double_price_l186_186158


namespace find_cost_price_l186_186689

-- Definitions based on conditions
def cost_price (C : ℝ) : Prop := 0.05 * C = 10

-- The theorem stating the problem to be proven
theorem find_cost_price (C : ℝ) (h : cost_price C) : C = 200 :=
by
  sorry

end find_cost_price_l186_186689


namespace mod_multiplication_result_l186_186267

theorem mod_multiplication_result :
  ∃ n : ℕ, 507 * 873 ≡ n [MOD 77] ∧ 0 ≤ n ∧ n < 77 ∧ n = 15 := by
  sorry

end mod_multiplication_result_l186_186267


namespace total_flour_correct_l186_186725

-- Define the quantities specified in the conditions
def cups_of_flour_already_added : ℕ := 2
def cups_of_flour_to_add : ℕ := 7

-- Define the total cups of flour required by the recipe as a sum of the quantities
def cups_of_flour_required : ℕ := cups_of_flour_already_added + cups_of_flour_to_add

-- Prove that the total cups of flour required is 9
theorem total_flour_correct : cups_of_flour_required = 9 := by
  -- use auto proof placeholder
  rfl

end total_flour_correct_l186_186725


namespace profit_function_is_correct_marginal_profit_function_is_correct_profit_function_max_value_marginal_profit_function_max_value_profit_and_marginal_profit_max_not_equal_l186_186241

noncomputable def R (x : ℕ) : ℝ := 3000 * x - 20 * x^2
noncomputable def C (x : ℕ) : ℝ := 500 * x + 4000
noncomputable def p (x : ℕ) : ℝ := R x - C x
noncomputable def Mp (x : ℕ) : ℝ := p (x + 1) - p x

theorem profit_function_is_correct : ∀ x, p x = -20 * x^2 + 2500 * x - 4000 := 
by 
  intro x
  sorry

theorem marginal_profit_function_is_correct : ∀ x, 0 < x ∧ x ≤ 100 → Mp x = -40 * x + 2480 := 
by 
  intro x
  sorry

theorem profit_function_max_value : ∃ x, (x = 62 ∨ x = 63) ∧ p x = 74120 :=
by 
  sorry

theorem marginal_profit_function_max_value : ∃ x, x = 1 ∧ Mp x = 2440 :=
by 
  sorry

theorem profit_and_marginal_profit_max_not_equal : ¬ (∃ x y, (x = 62 ∨ x = 63) ∧ y = 1 ∧ p x = Mp y) :=
by 
  sorry

end profit_function_is_correct_marginal_profit_function_is_correct_profit_function_max_value_marginal_profit_function_max_value_profit_and_marginal_profit_max_not_equal_l186_186241


namespace eval_expr_l186_186189

theorem eval_expr : 2 + 3 * 4 - 5 / 5 + 7 = 20 := by
  sorry

end eval_expr_l186_186189


namespace tan_alpha_minus_2beta_l186_186663

theorem tan_alpha_minus_2beta (α β : Real) 
  (h1 : Real.tan (α - β) = 2 / 5)
  (h2 : Real.tan β = 1 / 2) :
  Real.tan (α - 2 * β) = -1 / 12 := 
by 
  sorry

end tan_alpha_minus_2beta_l186_186663


namespace math_proof_problem_l186_186836

-- Define the function and its properties
variable (f : ℝ → ℝ)
axiom even_function : ∀ x : ℝ, f x = f (-x)
axiom periodicity : ∀ x : ℝ, f (x + 1) = -f x
axiom increasing_on_interval : ∀ x y : ℝ, (-1 ≤ x ∧ x < y ∧ y ≤ 0) → f x < f y

-- Theorem statement expressing the questions and answers
theorem math_proof_problem :
  (∀ x : ℝ, f (x + 2) = f x) ∧
  (∀ x : ℝ, f (1 - x) = f (1 + x)) ∧
  (f 2 = f 0) :=
by
  sorry

end math_proof_problem_l186_186836


namespace math_problem_l186_186384

noncomputable def f (x : ℝ) : ℝ := if h : x > 0 then x * Real.log x else 0

theorem math_problem :
  (∀ x, x > 0 → f x = x * Real.log x) ∧
  (∀ x, x ≤ 0 → f x = 0) ∧ 
  (∃ x > 0, Real.log x + 1 = 2 → x = Real.exp 1) ∧
  (∀ x, x > 1 / Real.exp 1 → Real.log x + 1 > 0) ∧
  (∀ ε, 0 < ε → ∃ x, x = 1 / Real.exp 1 ∧ f x = -1 / Real.exp 1) :=
by sorry

end math_problem_l186_186384


namespace original_price_of_computer_l186_186073

theorem original_price_of_computer (P : ℝ) (h1 : 1.30 * P = 364) (h2 : 2 * P = 560) : P = 280 :=
by 
  -- The proof is skipped as per instruction
  sorry

end original_price_of_computer_l186_186073


namespace enlarged_decal_height_l186_186067

theorem enlarged_decal_height (original_width original_height new_width : ℕ)
  (original_width_eq : original_width = 3)
  (original_height_eq : original_height = 2)
  (new_width_eq : new_width = 15)
  (proportions_consistent : ∀ h : ℕ, new_width * original_height = original_width * h) :
  ∃ new_height, new_height = 10 :=
by sorry

end enlarged_decal_height_l186_186067


namespace tangent_line_condition_l186_186458

theorem tangent_line_condition (a b k : ℝ) (h1 : (1 : ℝ) + a + b = 2) (h2 : 3 + a = k) (h3 : k = 1) :
    b - a = 5 := 
by 
    sorry

end tangent_line_condition_l186_186458


namespace M_gt_N_l186_186249

-- Define the variables and conditions
variables (a : ℝ)
def M : ℝ := 5 * a^2 - a + 1
def N : ℝ := 4 * a^2 + a - 1

-- Statement to prove
theorem M_gt_N : M a > N a := by
  -- Placeholder for the actual proof
  sorry

end M_gt_N_l186_186249


namespace max_area_of_triangle_l186_186862

-- Define the problem conditions and the maximum area S
theorem max_area_of_triangle
  (A B C : ℝ)
  (a b c S : ℝ)
  (h1 : 4 * S = a^2 - (b - c)^2)
  (h2 : b + c = 8) :
  S ≤ 8 :=
sorry

end max_area_of_triangle_l186_186862


namespace inequality_solution_real_roots_range_l186_186874

noncomputable def f (x : ℝ) : ℝ :=
|2 * x - 4| - |x - 3|

theorem inequality_solution :
  ∀ x, f x ≤ 2 → x ∈ Set.Icc (-1 : ℝ) 3 :=
sorry

theorem real_roots_range (k : ℝ) :
  (∃ x, f x = 0) → k ∈ Set.Icc (-1 : ℝ) 3 :=
sorry

end inequality_solution_real_roots_range_l186_186874


namespace frog_ends_on_vertical_side_l186_186636

-- Definitions for frog jump problem
def square : set (ℝ × ℝ) := 
  {p | 0 ≤ p.1 ∧ p.1 ≤ 6 ∧ 0 ≤ p.2 ∧ p.2 ≤ 6}

def boundary : set (ℝ × ℝ) :=
  {p | (p.1 = 0 ∨ p.1 = 6 ∨ p.2 = 0 ∨ p.2 = 6)}

def vertical_boundary : set (ℝ × ℝ) :=
  {p | (p.1 = 0 ∨ p.1 = 6) ∧ (0 ≤ p.2 ∧ p.2 ≤ 6)}

noncomputable def P (x y : ℝ) : ℝ := 0 -- placeholder probability function

axiom P_boundary_vertical {x y : ℝ} :
  ((x, y) ∈ boundary) → (x = 0 ∨ x = 6) → P x y = 1

axiom P_boundary_horizontal {x y : ℝ} :
  ((x, y) ∈ boundary) → (y = 0 ∨ y = 6) → P x y = 0

axiom P_recursive (x y : ℝ) :
  (x, y) ∈ square \ boundary →
  P x y = 1/4 * (P (x-1) y + P (x+1) y + P x (y-1) + P x (y+1))

theorem frog_ends_on_vertical_side :
  P 2 3 = 3/5 :=
sorry

end frog_ends_on_vertical_side_l186_186636


namespace smallest_relatively_prime_to_180_is_7_l186_186052

theorem smallest_relatively_prime_to_180_is_7 :
  ∃ y : ℕ, y > 1 ∧ Nat.gcd y 180 = 1 ∧ ∀ z : ℕ, z > 1 ∧ Nat.gcd z 180 = 1 → y ≤ z :=
by
  sorry

end smallest_relatively_prime_to_180_is_7_l186_186052


namespace abs_diff_squares_104_98_l186_186474

theorem abs_diff_squares_104_98 : abs ((104 : ℤ)^2 - (98 : ℤ)^2) = 1212 := by
  sorry

end abs_diff_squares_104_98_l186_186474


namespace not_perfect_square_l186_186819

theorem not_perfect_square : ¬ ∃ x : ℝ, x^2 = 7^2025 := by
  sorry

end not_perfect_square_l186_186819


namespace divisor_of_51234_plus_3_l186_186476

theorem divisor_of_51234_plus_3 : ∃ d : ℕ, d > 1 ∧ (51234 + 3) % d = 0 ∧ d = 3 :=
by {
  sorry
}

end divisor_of_51234_plus_3_l186_186476


namespace aaron_total_amount_owed_l186_186031

def total_cost (monthly_payment : ℤ) (months : ℤ) : ℤ :=
  monthly_payment * months

def interest_fee (amount : ℤ) (rate : ℤ) : ℤ :=
  amount * rate / 100

def total_amount_owed (monthly_payment : ℤ) (months : ℤ) (rate : ℤ) : ℤ :=
  let amount := total_cost monthly_payment months
  let fee := interest_fee amount rate
  amount + fee

theorem aaron_total_amount_owed :
  total_amount_owed 100 12 10 = 1320 :=
by
  sorry

end aaron_total_amount_owed_l186_186031


namespace total_seeds_eaten_correct_l186_186334

-- Define the number of seeds each player ate
def seeds_first_player : ℕ := 78
def seeds_second_player : ℕ := 53
def seeds_third_player (seeds_second_player : ℕ) : ℕ := seeds_second_player + 30

-- Define the total seeds eaten
def total_seeds_eaten (seeds_first_player seeds_second_player seeds_third_player : ℕ) : ℕ :=
  seeds_first_player + seeds_second_player + seeds_third_player

-- Statement of the theorem
theorem total_seeds_eaten_correct : total_seeds_eaten seeds_first_player seeds_second_player (seeds_third_player seeds_second_player) = 214 :=
by
  sorry

end total_seeds_eaten_correct_l186_186334


namespace largest_divisor_of_consecutive_product_l186_186805

theorem largest_divisor_of_consecutive_product (n : ℤ) (h : ∀ k : ℤ, (n = 5 * k)) :
  ∃ d : ℤ, d = 60 ∧ ∀ a : ℤ, divides d (a * (a + 1) * (a + 2) * (a + 3) * (a + 4)) :=
sorry

end largest_divisor_of_consecutive_product_l186_186805


namespace simplify_fraction_l186_186973

theorem simplify_fraction :
  (3^100 + 3^98) / (3^100 - 3^98) = 5 / 4 := 
by sorry

end simplify_fraction_l186_186973


namespace explorers_crossing_time_l186_186166

/-- Define constants and conditions --/
def num_explorers : ℕ := 60
def boat_capacity : ℕ := 6
def crossing_time : ℕ := 3
def round_trip_crossings : ℕ := 2
def total_trips := 1 + (num_explorers - boat_capacity - 1) / (boat_capacity - 1) + 1

theorem explorers_crossing_time :
  total_trips * crossing_time * round_trip_crossings / 2 + crossing_time = 69 :=
by sorry

end explorers_crossing_time_l186_186166


namespace simplify_expression_l186_186448

theorem simplify_expression (a b c : ℝ) (ha : a = 7.4) (hb : b = 5 / 37) :
  1.6 * ((1 / a + 1 / b - 2 * c / (a * b)) * (a + b + 2 * c)) / 
  ((1 / a^2 + 1 / b^2 + 2 / (a * b) - 4 * c^2 / (a^2 * b^2))) = 1.6 :=
by 
  rw [ha, hb] 
  sorry

end simplify_expression_l186_186448


namespace olivia_hair_length_l186_186554

def emilys_hair_length (logan_hair : ℕ) : ℕ := logan_hair + 6
def kates_hair_length (emily_hair : ℕ) : ℕ := emily_hair / 2
def jacks_hair_length (kate_hair : ℕ) : ℕ := (7 * kate_hair) / 2
def olivias_hair_length (jack_hair : ℕ) : ℕ := (2 * jack_hair) / 3

theorem olivia_hair_length
  (logan_hair : ℕ)
  (h_logan : logan_hair = 20)
  (h_emily : emilys_hair_length logan_hair = logan_hair + 6)
  (h_emily_value : emilys_hair_length logan_hair = 26)
  (h_kate : kates_hair_length (emilys_hair_length logan_hair) = 13)
  (h_jack : jacks_hair_length (kates_hair_length (emilys_hair_length logan_hair)) = 45)
  (h_olivia : olivias_hair_length (jacks_hair_length (kates_hair_length (emilys_hair_length logan_hair))) = 30) :
  olivias_hair_length
    (jacks_hair_length
      (kates_hair_length (emilys_hair_length logan_hair))) = 30 := by
  sorry

end olivia_hair_length_l186_186554


namespace fraction_zero_implies_value_l186_186237

theorem fraction_zero_implies_value (x : ℝ) (h : (|x| - 2) / (x + 2) = 0) (h_non_zero : x + 2 ≠ 0) : x = 2 :=
sorry

end fraction_zero_implies_value_l186_186237


namespace part_one_retail_wholesale_l186_186632

theorem part_one_retail_wholesale (x : ℕ) (wholesale : ℕ) : 
  70 * x + 40 * wholesale = 4600 ∧ x + wholesale = 100 → x = 20 ∧ wholesale = 80 :=
by
  sorry

end part_one_retail_wholesale_l186_186632


namespace kennedy_is_larger_l186_186556

-- Definitions based on given problem conditions
def KennedyHouse : ℕ := 10000
def BenedictHouse : ℕ := 2350
def FourTimesBenedictHouse : ℕ := 4 * BenedictHouse

-- Goal defined as a theorem to be proved
theorem kennedy_is_larger : KennedyHouse - FourTimesBenedictHouse = 600 :=
by 
  -- these are the conditions translated into Lean format
  let K := KennedyHouse
  let B := BenedictHouse
  let FourB := 4 * B
  let Goal := K - FourB
  -- prove the goal
  sorry

end kennedy_is_larger_l186_186556


namespace victor_initial_books_l186_186144

theorem victor_initial_books (x : ℕ) : (x + 3 = 12) → (x = 9) :=
by
  sorry

end victor_initial_books_l186_186144


namespace subtraction_of_fractions_l186_186146

theorem subtraction_of_fractions :
  let S_1 := 3 + 5 + 7
  let S_2 := 2 + 4 + 6
  let S_3 := 2 + 4 + 6
  let S_4 := 3 + 5 + 7
  (S_1 / S_2 - S_3 / S_4) = 9 / 20 :=
by
  let S_1 := 3 + 5 + 7
  let S_2 := 2 + 4 + 6
  let S_3 := 2 + 4 + 6
  let S_4 := 3 + 5 + 7
  sorry

end subtraction_of_fractions_l186_186146


namespace triangle_angle_construction_l186_186443

-- Step d): Lean 4 Statement
theorem triangle_angle_construction (a b c : ℝ) (α β : ℝ) (γ : ℝ) (h1 : γ = 120)
  (h2 : a < c) (h3 : c < a + b) (h4 : b < c)  (h5 : c < a + b) :
    (∃ α' β' γ', α' = 60 ∧ β' = α ∧ γ' = 60 + β) ∧ 
    (∃ α'' β'' γ'', α'' = 60 ∧ β'' = β ∧ γ'' = 60 + α) :=
  sorry

end triangle_angle_construction_l186_186443


namespace houses_with_animals_l186_186642

theorem houses_with_animals (n A B C x y : ℕ) (h1 : n = 2017) (h2 : A = 1820) (h3 : B = 1651) (h4 : C = 1182) 
    (hx : x = 1182) (hy : y = 619) : x - y = 563 := 
by {
  sorry
}

end houses_with_animals_l186_186642


namespace sin_sum_identity_l186_186857

theorem sin_sum_identity 
  (α : ℝ) 
  (h : Real.sin (2 * Real.pi / 3 - α) + Real.sin α = 4 * Real.sqrt 3 / 5) : 
  Real.sin (α + 7 * Real.pi / 6) = -4 / 5 := 
by 
  sorry

end sin_sum_identity_l186_186857


namespace number_of_candidates_l186_186746

theorem number_of_candidates
  (P : ℕ) (A_c A_p A_f : ℕ)
  (h_p : P = 100)
  (h_ac : A_c = 35)
  (h_ap : A_p = 39)
  (h_af : A_f = 15) :
  ∃ T : ℕ, T = 120 := 
by
  sorry

end number_of_candidates_l186_186746


namespace katie_has_more_games_l186_186716

   -- Conditions
   def katie_games : Nat := 81
   def friends_games : Nat := 59

   -- Problem statement
   theorem katie_has_more_games : (katie_games - friends_games) = 22 :=
   by
     -- Proof to be provided
     sorry
   
end katie_has_more_games_l186_186716


namespace arithmetic_sequence_sum_ratio_l186_186470

theorem arithmetic_sequence_sum_ratio
  (S : ℕ → ℝ) (T : ℕ → ℝ) (a b : ℕ → ℝ)
  (k : ℝ)
  (h1 : ∀ n, S n = 3 * k * n^2)
  (h2 : ∀ n, T n = k * n * (2 * n + 1))
  (h3 : ∀ n, a n = S n - S (n - 1))
  (h4 : ∀ n, b n = T n - T (n - 1))
  (h5 : ∀ n, S n / T n = (3 * n) / (2 * n + 1)) :
  (a 1 + a 2 + a 14 + a 19) / (b 1 + b 3 + b 17 + b 19) = 17 / 13 :=
sorry

end arithmetic_sequence_sum_ratio_l186_186470


namespace younger_person_age_l186_186732

theorem younger_person_age (e y : ℕ) 
  (h1: e = y + 20)
  (h2: e - 10 = 5 * (y - 10)) : 
  y = 15 := 
by
  sorry

end younger_person_age_l186_186732


namespace correct_operation_l186_186977

variables {x y : ℝ}

theorem correct_operation : -2 * x * 3 * y = -6 * x * y :=
by
  sorry

end correct_operation_l186_186977


namespace math_proof_l186_186291

noncomputable def math_problem (x : ℝ) : ℝ :=
  (3 / (2 * x) * (1 / 2) * (2 / 5) * 5020) - ((2 ^ 3) * (1 / (3 * x + 2)) * 250) + Real.sqrt (900 / x)

theorem math_proof :
  math_problem 4 = 60.393 :=
by
  sorry

end math_proof_l186_186291


namespace sin_690_eq_neg_half_l186_186354

theorem sin_690_eq_neg_half :
  Real.sin (690 * Real.pi / 180) = -1 / 2 :=
by {
  sorry
}

end sin_690_eq_neg_half_l186_186354


namespace no_positive_integer_triples_l186_186853

theorem no_positive_integer_triples (x y n : ℕ) (hx : 0 < x) (hy : 0 < y) (hn : 0 < n) : ¬ (x^2 + y^2 + 41 = 2^n) :=
  sorry

end no_positive_integer_triples_l186_186853


namespace number_of_team_members_l186_186445

-- Let's define the conditions.
def packs : ℕ := 3
def pouches_per_pack : ℕ := 6
def total_pouches : ℕ := packs * pouches_per_pack
def coaches : ℕ := 3
def helpers : ℕ := 2
def total_people (members : ℕ) : ℕ := members + coaches + helpers

-- Prove the number of members on the baseball team.
theorem number_of_team_members (members : ℕ) (h : total_people members = total_pouches) : members = 13 :=
by
  sorry

end number_of_team_members_l186_186445


namespace product_of_five_consecutive_integers_divisible_by_240_l186_186800

theorem product_of_five_consecutive_integers_divisible_by_240 (n : ℤ) : ∃ k : ℤ, (∏ i in range 5, n + i) = 240 * k := 
sorry

end product_of_five_consecutive_integers_divisible_by_240_l186_186800


namespace largest_divisor_of_product_of_five_consecutive_integers_l186_186765

theorem largest_divisor_of_product_of_five_consecutive_integers
  (a b c d e : ℤ) 
  (h: a + 1 = b ∧ b + 1 = c ∧ c + 1 = d ∧ d + 1 = e):
  ∃ (n : ℤ), n = 60 ∧ n ∣ (a * b * c * d * e) :=
sorry

end largest_divisor_of_product_of_five_consecutive_integers_l186_186765


namespace largest_integer_dividing_consecutive_product_l186_186813

theorem largest_integer_dividing_consecutive_product :
  ∀ (n : ℤ), (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) % 120 = 0 :=
by
  sorry

end largest_integer_dividing_consecutive_product_l186_186813


namespace printer_cost_l186_186304

theorem printer_cost (num_keyboards : ℕ) (num_printers : ℕ) (total_cost : ℕ) (keyboard_cost : ℕ) (printer_cost : ℕ) :
  num_keyboards = 15 →
  num_printers = 25 →
  total_cost = 2050 →
  keyboard_cost = 20 →
  (total_cost - (num_keyboards * keyboard_cost)) / num_printers = printer_cost →
  printer_cost = 70 :=
by
  sorry

end printer_cost_l186_186304


namespace largest_inscribed_triangle_area_l186_186346

theorem largest_inscribed_triangle_area
  (D : Type) 
  (radius : ℝ) 
  (r_eq : radius = 8) 
  (triangle_area : ℝ)
  (max_area : triangle_area = 64) :
  ∃ (base height : ℝ), (base = 2 * radius) ∧ (height = radius) ∧ (triangle_area = (1 / 2) * base * height) := 
by
  sorry

end largest_inscribed_triangle_area_l186_186346


namespace sum_of_three_consecutive_odd_integers_l186_186604

theorem sum_of_three_consecutive_odd_integers (x : ℤ) 
  (h1 : x % 2 = 1)                          -- x is odd
  (h2 : (x + 4) % 2 = 1)                    -- x+4 is odd
  (h3 : x + (x + 4) = 150)                  -- sum of first and third integers is 150
  : x + (x + 2) + (x + 4) = 225 :=          -- sum of the three integers is 225
by
  sorry

end sum_of_three_consecutive_odd_integers_l186_186604
