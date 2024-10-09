import Mathlib

namespace minimize_sum_of_squares_l1865_186558

theorem minimize_sum_of_squares (x1 x2 x3 : ℝ) (hpos1 : 0 < x1) (hpos2 : 0 < x2) (hpos3 : 0 < x3)
  (h_eq : x1 + 3 * x2 + 5 * x3 = 100) : x1^2 + x2^2 + x3^2 = 2000 / 7 := 
sorry

end minimize_sum_of_squares_l1865_186558


namespace area_of_parallelogram_l1865_186587

variable (b : ℕ)
variable (h : ℕ)
variable (A : ℕ)

-- Condition: The height is twice the base.
def height_twice_base := h = 2 * b

-- Condition: The base is 9.
def base_is_9 := b = 9

-- Condition: The area of the parallelogram is base times height.
def area_formula := A = b * h

-- Question: Prove that the area of the parallelogram is 162.
theorem area_of_parallelogram 
  (h_twice : height_twice_base h b) 
  (b_val : base_is_9 b) 
  (area_form : area_formula A b h): A = 162 := 
sorry

end area_of_parallelogram_l1865_186587


namespace dot_product_eq_neg20_l1865_186508

def a : ℝ × ℝ := (3, -1)
def b : ℝ × ℝ := (-5, 5)

def dot_product (x y : ℝ × ℝ) : ℝ :=
x.1 * y.1 + x.2 * y.2

theorem dot_product_eq_neg20 :
  dot_product a b = -20 :=
by
  sorry

end dot_product_eq_neg20_l1865_186508


namespace cost_price_per_meter_l1865_186585

-- Definitions based on the conditions given in the problem
def meters_of_cloth : ℕ := 45
def selling_price : ℕ := 4500
def profit_per_meter : ℕ := 12

-- Statement to prove
theorem cost_price_per_meter :
  (selling_price - (profit_per_meter * meters_of_cloth)) / meters_of_cloth = 88 :=
by
  sorry

end cost_price_per_meter_l1865_186585


namespace number_of_ordered_pairs_lcm_232848_l1865_186516

theorem number_of_ordered_pairs_lcm_232848 :
  let count_pairs :=
    let pairs_1 := 9
    let pairs_2 := 7
    let pairs_3 := 5
    let pairs_4 := 3
    pairs_1 * pairs_2 * pairs_3 * pairs_4
  count_pairs = 945 :=
by
  sorry

end number_of_ordered_pairs_lcm_232848_l1865_186516


namespace small_square_perimeter_l1865_186571

-- Condition Definitions
def perimeter_difference := 17
def side_length_of_square (x : ℝ) := 2 * x = perimeter_difference

-- Theorem Statement
theorem small_square_perimeter (x : ℝ) (h : side_length_of_square x) : 4 * x = 34 :=
by
  sorry

end small_square_perimeter_l1865_186571


namespace problem_l1865_186536

noncomputable def consecutive_integers_sum (x : ℕ) : ℕ :=
  x + (x + 1)

theorem problem (x : ℕ) (hx : x * (x + 1) = 812) : consecutive_integers_sum x = 57 := by
  sorry

end problem_l1865_186536


namespace solution_of_system_of_inequalities_l1865_186513

theorem solution_of_system_of_inequalities (p : ℝ) :
  19 * p < 10 ∧ p > 0.5 ↔ (1 / 2) < p ∧ p < (10 / 19) :=
by
  sorry

end solution_of_system_of_inequalities_l1865_186513


namespace geometric_sequence_fifth_term_l1865_186566

theorem geometric_sequence_fifth_term 
    (a₁ : ℕ) (a₄ : ℕ) (r : ℕ) (a₅ : ℕ)
    (h₁ : a₁ = 3) (h₂ : a₄ = 240) 
    (h₃ : a₄ = a₁ * r^3) 
    (h₄ : a₅ = a₁ * r^4) : 
    a₅ = 768 :=
by
  sorry

end geometric_sequence_fifth_term_l1865_186566


namespace ticket_distribution_l1865_186555

noncomputable def num_dist_methods (n : ℕ) (A : ℕ) (B : ℕ) (C : ℕ) (D : ℕ) : ℕ := sorry

theorem ticket_distribution :
  num_dist_methods 18 5 6 7 10 = 140 := sorry

end ticket_distribution_l1865_186555


namespace final_price_correct_l1865_186595

noncomputable def original_price : ℝ := 49.99
noncomputable def first_discount : ℝ := 0.10
noncomputable def second_discount : ℝ := 0.20

theorem final_price_correct :
  let price_after_first_discount := original_price * (1 - first_discount)
  let price_after_second_discount := price_after_first_discount * (1 - second_discount)
  price_after_second_discount = 36.00 := by
    -- The proof would go here
    sorry

end final_price_correct_l1865_186595


namespace probability_one_card_each_l1865_186583

-- Define the total number of cards
def total_cards := 12

-- Define the number of cards from Adrian
def adrian_cards := 7

-- Define the number of cards from Bella
def bella_cards := 5

-- Calculate the probability of one card from each cousin when selecting two cards without replacement
theorem probability_one_card_each :
  (adrian_cards / total_cards) * (bella_cards / (total_cards - 1)) +
  (bella_cards / total_cards) * (adrian_cards / (total_cards - 1)) =
  35 / 66 := sorry

end probability_one_card_each_l1865_186583


namespace range_exp3_eq_l1865_186557

noncomputable def exp3 (x : ℝ) : ℝ := 3^x

theorem range_exp3_eq (x : ℝ) : Set.range (exp3) = Set.Ioi 0 :=
sorry

end range_exp3_eq_l1865_186557


namespace lenny_has_39_left_l1865_186500

/-- Define the initial amount Lenny has -/
def initial_amount : ℕ := 84

/-- Define the amount Lenny spent on video games -/
def spent_on_video_games : ℕ := 24

/-- Define the amount Lenny spent at the grocery store -/
def spent_on_groceries : ℕ := 21

/-- Define the total amount Lenny spent -/
def total_spent : ℕ := spent_on_video_games + spent_on_groceries

/-- Calculate the amount Lenny has left -/
def amount_left (initial amount_spent : ℕ) : ℕ :=
  initial - amount_spent

/-- The statement of our mathematical equivalent proof problem
  Prove that Lenny has $39 left given the initial amount,
  and the amounts spent on video games and groceries.
-/
theorem lenny_has_39_left :
  amount_left initial_amount total_spent = 39 :=
by
  -- Leave the proof as 'sorry' for now
  sorry

end lenny_has_39_left_l1865_186500


namespace sequence_periodic_l1865_186529

noncomputable def exists_N (a : ℕ → ℝ) : Prop :=
∀ n : ℕ, n ≥ 1 → a (n+2) = abs (a (n+1)) - a n

theorem sequence_periodic (a : ℕ → ℝ) (h : exists_N a) :
  ∃ N : ℕ, ∀ n : ℕ, n ≥ N → a (n+9) = a n :=
sorry

end sequence_periodic_l1865_186529


namespace simplify_expression_l1865_186574

variable (a b : ℝ)

theorem simplify_expression : a + (3 * a - 3 * b) - (a - 2 * b) = 3 * a - b := 
by 
  sorry

end simplify_expression_l1865_186574


namespace arcsin_add_arccos_eq_pi_div_two_l1865_186578

open Real

theorem arcsin_add_arccos_eq_pi_div_two (x : ℝ) (h : -1 ≤ x ∧ x ≤ 1) : 
  arcsin x + arccos x = (π / 2) :=
sorry

end arcsin_add_arccos_eq_pi_div_two_l1865_186578


namespace like_terms_monomials_l1865_186539

theorem like_terms_monomials (a b : ℕ) (x y : ℝ) (c : ℝ) (H1 : x^(a+1) * y^3 = c * y^b * x^2) : a = 1 ∧ b = 3 :=
by
  -- Proof will be provided here
  sorry

end like_terms_monomials_l1865_186539


namespace consecutive_product_not_mth_power_l1865_186567

theorem consecutive_product_not_mth_power (n m k : ℕ) :
  ¬ ∃ k, (n - 1) * n * (n + 1) = k^m := 
sorry

end consecutive_product_not_mth_power_l1865_186567


namespace student_correct_answers_l1865_186526

theorem student_correct_answers (C I : ℕ) 
  (h1 : C + I = 100) 
  (h2 : C - 2 * I = 61) : 
  C = 87 :=
by
  sorry

end student_correct_answers_l1865_186526


namespace sufficient_not_necessary_p_q_l1865_186518

theorem sufficient_not_necessary_p_q {m : ℝ} 
  (hp : ∀ x, (x^2 - 8*x - 20 ≤ 0) → (-2 ≤ x ∧ x ≤ 10))
  (hq : ∀ x, ((x - 1 - m) * (x - 1 + m) ≤ 0) → (1 - m ≤ x ∧ x ≤ 1 + m))
  (m_pos : 0 < m)  :
  (∀ x, (x - 1 - m) * (x - 1 + m) ≤ 0 → x^2 - 8*x - 20 ≤ 0) ∧ ¬ (∀ x, x^2 - 8*x - 20 ≤ 0 → (x - 1 - m) * (x - 1 + m) ≤ 0) →
  m ≤ 3 :=
sorry

end sufficient_not_necessary_p_q_l1865_186518


namespace scaling_transformation_l1865_186562

theorem scaling_transformation (a b : ℝ) :
  (∀ x y : ℝ, (y = 1 - x → y' = b * (1 - x))
    → (y' = b - b * x)) 
  ∧
  (∀ x' y' : ℝ, (y = (2 / 3) * x' + 2)
    → (y' = (2 / 3) * (a * x) + 2))
  → a = 3 ∧ b = 2 := by
  sorry

end scaling_transformation_l1865_186562


namespace num_coloring_l1865_186561

-- Define the set of numbers to be colored
def numbers_to_color : List ℕ := [2, 3, 4, 5, 6, 7, 8, 9]

-- Define the set of colors
inductive Color
| red
| green
| blue

-- Define proper divisors for the numbers in the list
def proper_divisors (n : ℕ) : List ℕ :=
  match n with
  | 2 => []
  | 3 => []
  | 4 => [2]
  | 5 => []
  | 6 => [2, 3]
  | 7 => []
  | 8 => [2, 4]
  | 9 => [3]
  | _ => []

-- The proof statement
theorem num_coloring (h : ∀ n ∈ numbers_to_color, ∀ d ∈ proper_divisors n, n ≠ d) :
  ∃ f : ℕ → Color, ∀ n ∈ numbers_to_color, ∀ d ∈ proper_divisors n, f n ≠ f d :=
  sorry

end num_coloring_l1865_186561


namespace find_second_number_l1865_186591

theorem find_second_number (x : ℕ) :
  22030 = (555 + x) * 2 * (x - 555) + 30 → 
  x = 564 :=
by
  intro h
  sorry

end find_second_number_l1865_186591


namespace square_section_dimensions_l1865_186599

theorem square_section_dimensions (x length : ℕ) :
  (250 ≤ x^2 + x * length ∧ x^2 + x * length ≤ 300) ∧ (25 ≤ length ∧ length ≤ 30) →
  (x = 7 ∨ x = 8) :=
  by
    sorry

end square_section_dimensions_l1865_186599


namespace average_donation_proof_l1865_186593

noncomputable def average_donation (total_people : ℝ) (donated_200 : ℝ) (donated_100 : ℝ) (donated_50 : ℝ) : ℝ :=
  let proportion_200 := donated_200 / total_people
  let proportion_100 := donated_100 / total_people
  let proportion_50 := donated_50 / total_people
  let total_donation := (200 * proportion_200) + (100 * proportion_100) + (50 * proportion_50)
  total_donation

theorem average_donation_proof 
  (total_people : ℝ)
  (donated_200 donated_100 donated_50 : ℝ)
  (h1 : proportion_200 = 1 / 10)
  (h2 : proportion_100 = 3 / 4)
  (h3 : proportion_50 = 1 - proportion_200 - proportion_100) :
  average_donation total_people donated_200 donated_100 donated_50 = 102.5 :=
  by 
    sorry

end average_donation_proof_l1865_186593


namespace women_in_luxury_suites_count_l1865_186564

noncomputable def passengers : ℕ := 300
noncomputable def percentage_women : ℝ := 70 / 100
noncomputable def percentage_luxury : ℝ := 15 / 100

noncomputable def women_on_ship : ℝ := passengers * percentage_women
noncomputable def women_in_luxury_suites : ℝ := women_on_ship * percentage_luxury

theorem women_in_luxury_suites_count : 
  round women_in_luxury_suites = 32 :=
by sorry

end women_in_luxury_suites_count_l1865_186564


namespace gmat_test_statistics_l1865_186577

theorem gmat_test_statistics 
    (p1 : ℝ) (p2 : ℝ) (p12 : ℝ) (neither : ℝ) (S : ℝ) 
    (h1 : p1 = 0.85)
    (h2 : p12 = 0.60) 
    (h3 : neither = 0.05) :
    0.25 + S = 0.95 → S = 0.70 :=
by
  sorry

end gmat_test_statistics_l1865_186577


namespace transformed_roots_l1865_186548

theorem transformed_roots (b c : ℝ) (h₁ : (Polynomial.C 1 * Polynomial.X^2 + Polynomial.C b * Polynomial.X + Polynomial.C c).roots = {2, -3}) :
  (Polynomial.C 1 * (Polynomial.X - Polynomial.C 4)^2 + Polynomial.C b * (Polynomial.X - Polynomial.C 4) + Polynomial.C c).roots = {1, 6} :=
by
  sorry

end transformed_roots_l1865_186548


namespace abs_value_solutions_l1865_186527

theorem abs_value_solutions (y : ℝ) :
  |4 * y - 5| = 39 ↔ (y = 11 ∨ y = -8.5) :=
by
  sorry

end abs_value_solutions_l1865_186527


namespace jordan_rectangle_width_l1865_186547

theorem jordan_rectangle_width :
  ∀ (areaC areaJ : ℕ) (lengthC widthC lengthJ widthJ : ℕ), 
    (areaC = lengthC * widthC) →
    (areaJ = lengthJ * widthJ) →
    (areaC = areaJ) →
    (lengthC = 5) →
    (widthC = 24) →
    (lengthJ = 3) →
    widthJ = 40 :=
by
  intros areaC areaJ lengthC widthC lengthJ widthJ
  intro hAreaC
  intro hAreaJ
  intro hEqualArea
  intro hLengthC
  intro hWidthC
  intro hLengthJ
  sorry

end jordan_rectangle_width_l1865_186547


namespace vicki_donated_fraction_l1865_186550

/-- Given Jeff had 300 pencils and donated 30% of them, and Vicki had twice as many pencils as Jeff originally 
    had, and there are 360 pencils remaining altogether after both donations,
    prove that Vicki donated 3/4 of her pencils. -/
theorem vicki_donated_fraction : 
  let jeff_pencils := 300
  let jeff_donated := jeff_pencils * 0.30
  let jeff_remaining := jeff_pencils - jeff_donated
  let vicki_pencils := 2 * jeff_pencils
  let total_remaining := 360
  let vicki_remaining := total_remaining - jeff_remaining
  let vicki_donated := vicki_pencils - vicki_remaining
  vicki_donated / vicki_pencils = 3 / 4 :=
by
  -- Proof needs to be inserted here
  sorry

end vicki_donated_fraction_l1865_186550


namespace min_value_of_expression_l1865_186501

noncomputable def smallest_value (a b c : ℕ) : ℤ :=
  3 * a - 2 * a * b + a * c

theorem min_value_of_expression : ∃ (a b c : ℕ), 0 < a ∧ a < 7 ∧ 0 < b ∧ b ≤ 3 ∧ 0 < c ∧ c ≤ 4 ∧ smallest_value a b c = -12 := by
  sorry

end min_value_of_expression_l1865_186501


namespace range_of_a_l1865_186576

def prop_p (a : ℝ) : Prop := ∀ x : ℝ, a * x^2 + a * x + 1 > 0
def prop_q (a : ℝ) : Prop := ∃ x : ℝ, x^2 - x + a = 0

theorem range_of_a (a : ℝ) : (prop_p a ∨ prop_q a) ∧ ¬(prop_p a ∧ prop_q a) ↔ (a < 0 ∨ (1 / 4 < a ∧ a < 4)) :=
by
  sorry

end range_of_a_l1865_186576


namespace greatest_natural_number_exists_l1865_186512

noncomputable def sum_of_squares (n : ℕ) : ℕ :=
    n * (n + 1) * (2 * n + 1) / 6

noncomputable def squared_sum_from_to (a b : ℕ) : ℕ :=
    sum_of_squares b - sum_of_squares (a - 1)

noncomputable def is_perfect_square (n : ℕ) : Prop :=
    ∃ k, k * k = n

theorem greatest_natural_number_exists :
    ∃ n : ℕ, n = 1921 ∧ n ≤ 2008 ∧ 
    is_perfect_square ((sum_of_squares n) * (squared_sum_from_to (n + 1) (2 * n))) :=
by
  sorry

end greatest_natural_number_exists_l1865_186512


namespace value_of_a_minus_c_l1865_186544

theorem value_of_a_minus_c
  (a b c d : ℝ) 
  (h1 : (a + d + b + d) / 2 = 80)
  (h2 : (b + d + c + d) / 2 = 180)
  (h3 : d = 2 * (a - b)) :
  a - c = -200 := sorry

end value_of_a_minus_c_l1865_186544


namespace right_building_shorter_l1865_186546

-- Define the conditions as hypotheses
def middle_building_height : ℕ := 100
def left_building_height : ℕ := (80 * middle_building_height) / 100
def combined_height_left_middle : ℕ := left_building_height + middle_building_height
def total_height : ℕ := 340
def right_building_height : ℕ := total_height - combined_height_left_middle

-- Define the statement we need to prove
theorem right_building_shorter :
  combined_height_left_middle - right_building_height = 20 :=
by sorry

end right_building_shorter_l1865_186546


namespace find_a_l1865_186586

noncomputable def f (a x : ℝ) : ℝ := a * x^3 + 2

theorem find_a (a : ℝ) (h : (3 * a * (-1 : ℝ)^2) = 3) : a = 1 :=
by
  sorry

end find_a_l1865_186586


namespace chelsea_cupcakes_time_l1865_186511

theorem chelsea_cupcakes_time
  (batches : ℕ)
  (bake_time_per_batch : ℕ)
  (ice_time_per_batch : ℕ)
  (total_time : ℕ)
  (h1 : batches = 4)
  (h2 : bake_time_per_batch = 20)
  (h3 : ice_time_per_batch = 30)
  (h4 : total_time = (bake_time_per_batch + ice_time_per_batch) * batches) :
  total_time = 200 :=
  by
  -- The proof statement here
  -- The proof would go here, but we skip it for now
  sorry

end chelsea_cupcakes_time_l1865_186511


namespace maddie_watched_138_on_monday_l1865_186524

-- Define the constants and variables from the problem statement
def total_episodes : ℕ := 8
def minutes_per_episode : ℕ := 44
def watched_thursday : ℕ := 21
def watched_friday_episodes : ℕ := 2
def watched_weekend : ℕ := 105

-- Calculate the total minutes watched from all episodes
def total_minutes : ℕ := total_episodes * minutes_per_episode

-- Calculate the minutes watched on Friday
def watched_friday : ℕ := watched_friday_episodes * minutes_per_episode

-- Calculate the total minutes watched on weekdays excluding Monday
def watched_other_days : ℕ := watched_thursday + watched_friday + watched_weekend

-- Statement to prove that Maddie watched 138 minutes on Monday
def minutes_watched_on_monday : ℕ := total_minutes - watched_other_days

-- The final statement for proof in Lean 4
theorem maddie_watched_138_on_monday : minutes_watched_on_monday = 138 := by
  -- This theorem should be proved using the above definitions and calculations, proof skipped with sorry
  sorry

end maddie_watched_138_on_monday_l1865_186524


namespace geom_seq_product_l1865_186592

-- Given conditions
variables (a : ℕ → ℝ)
variable (r : ℝ)
axiom geom_seq (n : ℕ) : a (n + 1) = a n * r
axiom a1_eq_1 : a 1 = 1
axiom a10_eq_3 : a 10 = 3

-- Proof goal
theorem geom_seq_product : a 2 * a 3 * a 4 * a 5 * a 6 * a 7 * a 8 * a 9 = 81 :=  
sorry

end geom_seq_product_l1865_186592


namespace abs_neg_implies_nonpositive_l1865_186563

theorem abs_neg_implies_nonpositive (a : ℝ) : |a| = -a → a ≤ 0 :=
by
  sorry

end abs_neg_implies_nonpositive_l1865_186563


namespace square_side_length_is_10_l1865_186568

-- Define the side lengths of the original squares
def side_length1 : ℝ := 8
def side_length2 : ℝ := 6

-- Define the areas of the original squares
def area1 : ℝ := side_length1^2
def area2 : ℝ := side_length2^2

-- Define the total area of the combined squares
def total_area : ℝ := area1 + area2

-- Define the side length of the new square
def side_length_new_square : ℝ := 10

-- Theorem statement to prove that the side length of the new square is 10 cm
theorem square_side_length_is_10 : side_length_new_square^2 = total_area := by
  sorry

end square_side_length_is_10_l1865_186568


namespace fitness_center_cost_effectiveness_l1865_186532

noncomputable def f (x : ℝ) : ℝ := 5 * x

noncomputable def g (x : ℝ) : ℝ :=
  if 15 ≤ x ∧ x ≤ 30 then 90 
  else 2 * x + 30

def cost_comparison (x : ℝ) (h1 : 15 ≤ x) (h2 : x ≤ 40) : Prop :=
  (15 ≤ x ∧ x < 18 → f x < g x) ∧
  (x = 18 → f x = g x) ∧
  (18 < x ∧ x ≤ 40 → f x > g x)

theorem fitness_center_cost_effectiveness (x : ℝ) (h1 : 15 ≤ x) (h2 : x ≤ 40) : cost_comparison x h1 h2 :=
by
  sorry

end fitness_center_cost_effectiveness_l1865_186532


namespace equal_roots_B_value_l1865_186517

theorem equal_roots_B_value (B : ℝ) :
  (∀ k : ℝ, (2 * k * x^2 + B * x + 2 = 0) → (k = 1 → (B^2 - 4 * (2 * 1) * 2 = 0))) → B = 4 ∨ B = -4 :=
by
  sorry

end equal_roots_B_value_l1865_186517


namespace g_neg_six_eq_neg_twenty_l1865_186505

theorem g_neg_six_eq_neg_twenty (g : ℤ → ℤ)
    (h1 : g 1 - 1 > 0)
    (h2 : ∀ x y : ℤ, g x * g y + x + y + x * y = g (x + y) + x * g y + y * g x)
    (h3 : ∀ x : ℤ, 3 * g (x + 1) = g x + 2 * x + 3) :
  g (-6) = -20 := 
sorry

end g_neg_six_eq_neg_twenty_l1865_186505


namespace sequence_bk_bl_sum_l1865_186589

theorem sequence_bk_bl_sum (b : ℕ → ℕ) (m : ℕ) 
  (h_pairwise_distinct : ∀ i j, i ≠ j → b i ≠ b j)
  (h_b0 : b 0 = 0)
  (h_b_lt_2n : ∀ n, 0 < n → b n < 2 * n) :
  ∃ k ℓ : ℕ, b k + b ℓ = m := 
  sorry

end sequence_bk_bl_sum_l1865_186589


namespace intersection_of_A_and_B_l1865_186537

noncomputable def A : Set ℝ := {-2, -1, 0, 1}
noncomputable def B : Set ℝ := {x | x^2 - 1 ≤ 0}

theorem intersection_of_A_and_B : A ∩ B = {-1, 0, 1} := 
by
  sorry

end intersection_of_A_and_B_l1865_186537


namespace no_solution_k_eq_7_l1865_186594

-- Define the condition that x should not be equal to 4 and 8
def condition (x : ℝ) : Prop := x ≠ 4 ∧ x ≠ 8

-- Define the equation
def equation (x k : ℝ) : Prop := (x - 3) / (x - 4) = (x - k) / (x - 8)

-- Prove that for the equation to have no solution, k must be 7
theorem no_solution_k_eq_7 : (∀ x, condition x → ¬ equation x 7) ↔ (∃ k, k = 7) :=
by
  sorry

end no_solution_k_eq_7_l1865_186594


namespace solve_equation_l1865_186533

theorem solve_equation (x : ℚ) :
  (1 / (x + 2) + 3 * x / (x + 2) + 4 / (x + 2) = 1) → x = -3 / 2 :=
by
  sorry

end solve_equation_l1865_186533


namespace evan_runs_200_more_feet_l1865_186572

def street_width : ℕ := 25
def block_side : ℕ := 500

def emily_path : ℕ := 4 * block_side
def evan_path : ℕ := 4 * (block_side + 2 * street_width)

theorem evan_runs_200_more_feet : evan_path - emily_path = 200 := by
  sorry

end evan_runs_200_more_feet_l1865_186572


namespace point_in_second_quadrant_iff_l1865_186596

theorem point_in_second_quadrant_iff (a : ℝ) : (a - 2 < 0) ↔ (a < 2) :=
by
  sorry

end point_in_second_quadrant_iff_l1865_186596


namespace price_on_friday_is_correct_l1865_186549

-- Define initial price on Tuesday
def price_on_tuesday : ℝ := 50

-- Define the percentage increase on Wednesday (20%)
def percentage_increase : ℝ := 0.20

-- Define the percentage discount on Friday (15%)
def percentage_discount : ℝ := 0.15

-- Define the price on Wednesday after the increase
def price_on_wednesday : ℝ := price_on_tuesday * (1 + percentage_increase)

-- Define the price on Friday after the discount
def price_on_friday : ℝ := price_on_wednesday * (1 - percentage_discount)

-- Theorem statement to prove that the price on Friday is 51 dollars
theorem price_on_friday_is_correct : price_on_friday = 51 :=
by
  sorry

end price_on_friday_is_correct_l1865_186549


namespace least_integer_greater_than_sqrt_500_l1865_186545

theorem least_integer_greater_than_sqrt_500 : ∃ n : ℤ, n = 23 ∧ ∀ m : ℤ, (m ≤ 23 → m^2 ≤ 500) → (m < 23 ∧ m > 0 → (m + 1)^2 > 500) :=
by
  sorry

end least_integer_greater_than_sqrt_500_l1865_186545


namespace handshake_count_l1865_186534

theorem handshake_count (couples : ℕ) (people : ℕ) (total_handshakes : ℕ) :
  couples = 6 →
  people = 2 * couples →
  total_handshakes = (people * (people - 1)) / 2 - couples →
  total_handshakes = 60 :=
by
  intros h_couples h_people h_handshakes
  sorry

end handshake_count_l1865_186534


namespace pqrs_predicate_l1865_186598

noncomputable def P (a b c : ℝ) := a + b - c
noncomputable def Q (a b c : ℝ) := b + c - a
noncomputable def R (a b c : ℝ) := c + a - b

theorem pqrs_predicate (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (P a b c) * (Q a b c) * (R a b c) > 0 ↔ (P a b c > 0 ∧ Q a b c > 0 ∧ R a b c > 0) :=
sorry

end pqrs_predicate_l1865_186598


namespace min_rectangles_to_cover_minimum_number_of_rectangles_required_l1865_186531

-- Definitions based on the conditions
def corners_type1 : Nat := 12
def corners_type2 : Nat := 12

theorem min_rectangles_to_cover (type1_corners type2_corners : Nat) (h1 : type1_corners = corners_type1) (h2 : type2_corners = corners_type2) : Nat :=
12

theorem minimum_number_of_rectangles_required (type1_corners type2_corners : Nat) (h1 : type1_corners = corners_type1) (h2 : type2_corners = corners_type2) :
  min_rectangles_to_cover type1_corners type2_corners h1 h2 = 12 := by
  sorry

end min_rectangles_to_cover_minimum_number_of_rectangles_required_l1865_186531


namespace distance_points_lt_2_over_3_r_l1865_186543

theorem distance_points_lt_2_over_3_r (r : ℝ) (h_pos_r : 0 < r) (points : Fin 17 → ℝ × ℝ)
  (h_points_in_circle : ∀ i, (points i).1 ^ 2 + (points i).2 ^ 2 < r ^ 2) :
  ∃ i j : Fin 17, i ≠ j ∧ (dist (points i) (points j) < 2 * r / 3) :=
by
  sorry

end distance_points_lt_2_over_3_r_l1865_186543


namespace smallest_three_digit_candy_number_l1865_186575

theorem smallest_three_digit_candy_number (n : ℕ) (hn1 : 100 ≤ n) (hn2 : n ≤ 999)
    (h1 : (n + 6) % 9 = 0) (h2 : (n - 9) % 6 = 0) : n = 111 := by
  sorry

end smallest_three_digit_candy_number_l1865_186575


namespace arithmetic_identity_l1865_186506

theorem arithmetic_identity : 72 * 989 - 12 * 989 = 59340 := by
  sorry

end arithmetic_identity_l1865_186506


namespace perfect_square_trinomial_k_l1865_186597

theorem perfect_square_trinomial_k (k : ℤ) : 
  (∀ x : ℝ, x^2 - k*x + 64 = (x + 8)^2 ∨ x^2 - k*x + 64 = (x - 8)^2) → 
  (k = 16 ∨ k = -16) :=
by
  sorry

end perfect_square_trinomial_k_l1865_186597


namespace isosceles_triangle_perimeter_l1865_186541

variable (a b : ℕ) 

theorem isosceles_triangle_perimeter (h1 : a = 3) (h2 : b = 6) : 
  ∃ P, (a = 3 ∧ b = 6 ∧ P = 15 ∨ b = 3 ∧ a = 6 ∧ P = 15) := by
  use 15
  sorry

end isosceles_triangle_perimeter_l1865_186541


namespace rectangle_side_length_relation_l1865_186556

variable (x y : ℝ)

-- Condition: The area of the rectangle is 10
def is_rectangle_area_10 (x y : ℝ) : Prop := x * y = 10

-- Theorem: Given the area condition, express y in terms of x
theorem rectangle_side_length_relation (h : is_rectangle_area_10 x y) : y = 10 / x :=
sorry

end rectangle_side_length_relation_l1865_186556


namespace dollar_neg3_4_eq_neg27_l1865_186553

-- Define the operation $$
def dollar (a b : ℤ) : ℤ := a * (b + 1) + a * b

-- Theorem stating the value of (-3) $$ 4
theorem dollar_neg3_4_eq_neg27 : dollar (-3) 4 = -27 := 
by
  sorry

end dollar_neg3_4_eq_neg27_l1865_186553


namespace desired_interest_rate_l1865_186523

theorem desired_interest_rate 
  (F : ℝ) -- Face value of each share
  (D : ℝ) -- Dividend rate
  (M : ℝ) -- Market value of each share
  (annual_dividend : ℝ := (D / 100) * F) -- Annual dividend per share
  (desired_interest_rate : ℝ := (annual_dividend / M) * 100) -- Desired interest rate
  (F_eq : F = 44) -- Given Face value
  (D_eq : D = 9) -- Given Dividend rate
  (M_eq : M = 33) -- Given Market value
  : desired_interest_rate = 12 := 
by
  sorry

end desired_interest_rate_l1865_186523


namespace box_made_by_Bellini_or_son_l1865_186590

-- Definitions of the conditions
variable (B : Prop) -- Bellini made the box
variable (S : Prop) -- Bellini's son made the box
variable (inscription_true : Prop) -- The inscription "I made this box" is truthful

-- The problem statement in Lean: Prove that B or S given the inscription is true
theorem box_made_by_Bellini_or_son (B S inscription_true : Prop) (h1 : inscription_true → (B ∨ S)) : B ∨ S :=
by
  sorry

end box_made_by_Bellini_or_son_l1865_186590


namespace part_a_part_b_l1865_186509

theorem part_a (α : ℝ) (h_irr : Irrational α) (a b : ℝ) (h_lt : a < b) :
  ∃ (m n : ℤ), a < m * α - n ∧ m * α - n < b :=
sorry

theorem part_b (α : ℝ) (h_irr : Irrational α) (a b : ℝ) (h_lt : a < b) :
  ∃ (m n : ℕ), a < m * α - n ∧ m * α - n < b :=
sorry

end part_a_part_b_l1865_186509


namespace no_integer_solution_for_conditions_l1865_186559

theorem no_integer_solution_for_conditions :
  ¬∃ (x : ℤ), 
    (18 + x = 2 * (5 + x)) ∧
    (18 + x = 3 * (2 + x)) ∧
    ((18 + x) + (5 + x) + (2 + x) = 50) :=
by
  sorry

end no_integer_solution_for_conditions_l1865_186559


namespace jerry_reaches_five_probability_l1865_186502

noncomputable def probability_move_reaches_five_at_some_point : ℚ :=
  let num_heads_needed := 7
  let num_tails_needed := 3
  let total_tosses := 10
  let num_ways_to_choose_heads := Nat.choose total_tosses num_heads_needed
  let total_possible_outcomes : ℚ := 2^total_tosses
  let prob_reach_4 := num_ways_to_choose_heads / total_possible_outcomes
  let prob_reach_5_at_some_point := 2 * prob_reach_4
  prob_reach_5_at_some_point

theorem jerry_reaches_five_probability :
  probability_move_reaches_five_at_some_point = 15 / 64 := by
  sorry

end jerry_reaches_five_probability_l1865_186502


namespace restaurant_bill_l1865_186510

theorem restaurant_bill 
  (salisbury_steak : ℝ := 16.00)
  (chicken_fried_steak : ℝ := 18.00)
  (mozzarella_sticks : ℝ := 8.00)
  (caesar_salad : ℝ := 6.00)
  (bowl_chili : ℝ := 7.00)
  (chocolate_lava_cake : ℝ := 7.50)
  (cheesecake : ℝ := 6.50)
  (iced_tea : ℝ := 3.00)
  (soda : ℝ := 3.50)
  (half_off_meal : ℝ := 0.5)
  (dessert_discount : ℝ := 0.1)
  (tip_percent : ℝ := 0.2)
  (sales_tax : ℝ := 0.085) :
  let total : ℝ :=
    (salisbury_steak * half_off_meal) +
    (chicken_fried_steak * half_off_meal) +
    mozzarella_sticks +
    caesar_salad +
    bowl_chili +
    (chocolate_lava_cake * (1 - dessert_discount)) +
    (cheesecake * (1 - dessert_discount)) +
    iced_tea +
    soda
  let total_with_tax : ℝ := total * (1 + sales_tax)
  let final_total : ℝ := total_with_tax * (1 + tip_percent)
  final_total = 73.04 :=
by
  sorry

end restaurant_bill_l1865_186510


namespace bricks_in_wall_l1865_186520

theorem bricks_in_wall (h : ℕ) 
  (brenda_rate : ℕ := h / 8)
  (brandon_rate : ℕ := h / 12)
  (combined_rate : ℕ := (5 * h) / 24)
  (decreased_combined_rate : ℕ := combined_rate - 15)
  (work_time : ℕ := 6) :
  work_time * decreased_combined_rate = h → h = 360 := by
  intros h_eq
  sorry

end bricks_in_wall_l1865_186520


namespace ticket_costs_l1865_186581

-- Define the conditions
def cost_per_ticket : ℕ := 44
def number_of_tickets : ℕ := 7

-- Define the total cost calculation
def total_cost : ℕ := cost_per_ticket * number_of_tickets

-- Prove that given the conditions, the total cost is 308
theorem ticket_costs :
  total_cost = 308 :=
by
  -- Proof steps here
  sorry

end ticket_costs_l1865_186581


namespace remainder_when_xyz_divided_by_9_is_0_l1865_186538

theorem remainder_when_xyz_divided_by_9_is_0
  (x y z : ℕ)
  (hx : x < 9)
  (hy : y < 9)
  (hz : z < 9)
  (h1 : x + 3 * y + 2 * z ≡ 0 [MOD 9])
  (h2 : 3 * x + 2 * y + z ≡ 5 [MOD 9])
  (h3 : 2 * x + y + 3 * z ≡ 5 [MOD 9]) :
  (x * y * z) % 9 = 0 := by
  sorry

end remainder_when_xyz_divided_by_9_is_0_l1865_186538


namespace real_root_quadratic_l1865_186521

theorem real_root_quadratic (b : ℝ) : 
  (∃ x : ℝ, x^2 + b * x + 9 = 0) ↔ b ≤ -6 ∨ b ≥ 6 := 
sorry

end real_root_quadratic_l1865_186521


namespace mean_score_74_9_l1865_186580

/-- 
In a class of 100 students, the score distribution is as follows:
- 10 students scored 100%
- 15 students scored 90%
- 20 students scored 80%
- 30 students scored 70%
- 20 students scored 60%
- 4 students scored 50%
- 1 student scored 40%

Prove that the mean percentage score of the class is 74.9.
-/
theorem mean_score_74_9 : 
  let scores := [100, 90, 80, 70, 60, 50, 40]
  let counts := [10, 15, 20, 30, 20, 4, 1]
  let total_students := 100
  let total_score := 1000 + 1350 + 1600 + 2100 + 1200 + 200 + 40
  (total_score / total_students : ℝ) = 74.9 :=
by {
  -- The detailed proof steps are omitted with sorry.
  sorry
}

end mean_score_74_9_l1865_186580


namespace continuity_at_x0_l1865_186565

def f (x : ℝ) : ℝ := -5 * x ^ 2 - 8
def x0 : ℝ := 2

theorem continuity_at_x0 :
  ∀ (ε : ℝ), ε > 0 → ∃ (δ : ℝ), δ > 0 ∧ (∀ (x : ℝ), abs (x - x0) < δ → abs (f x - f x0) < ε) :=
by
  sorry

end continuity_at_x0_l1865_186565


namespace last_two_digits_7_pow_2011_l1865_186554

noncomputable def pow_mod_last_two_digits (n : ℕ) : ℕ :=
  (7^n) % 100

theorem last_two_digits_7_pow_2011 : pow_mod_last_two_digits 2011 = 43 :=
by
  sorry

end last_two_digits_7_pow_2011_l1865_186554


namespace choose_correct_graph_l1865_186573

noncomputable def appropriate_graph : String :=
  let bar_graph := "Bar graph"
  let pie_chart := "Pie chart"
  let line_graph := "Line graph"
  let freq_dist_graph := "Frequency distribution graph"
  
  if (bar_graph = "Bar graph") ∧ (pie_chart = "Pie chart") ∧ (line_graph = "Line graph") ∧ (freq_dist_graph = "Frequency distribution graph") then
    "Line graph"
  else
    sorry

theorem choose_correct_graph :
  appropriate_graph = "Line graph" :=
by
  sorry

end choose_correct_graph_l1865_186573


namespace interest_payment_frequency_l1865_186504

theorem interest_payment_frequency (i : ℝ) (EAR : ℝ) (n : ℕ)
  (h1 : i = 0.10) (h2 : EAR = 0.1025) :
  (1 + i / n)^n = 1 + EAR → n = 2 :=
by
  intros
  sorry

end interest_payment_frequency_l1865_186504


namespace second_class_students_count_l1865_186552

theorem second_class_students_count 
    (x : ℕ)
    (h1 : 12 * 40 = 480)
    (h2 : ∀ x, x * 60 = 60 * x)
    (h3 : (12 + x) * 54 = 480 + 60 * x) : 
    x = 28 :=
by
  sorry

end second_class_students_count_l1865_186552


namespace original_savings_l1865_186514

-- Given conditions:
def total_savings (s : ℝ) : Prop :=
  1 / 4 * s = 230

-- Theorem statement: 
theorem original_savings (s : ℝ) (h : total_savings s) : s = 920 :=
sorry

end original_savings_l1865_186514


namespace part1_part2_l1865_186542

noncomputable def f (x : ℝ) : ℝ := |2 * x - 1| + |5 - x|

theorem part1 : ∃ m, m = 9 / 2 ∧ ∀ x, f x ≥ m :=
sorry

theorem part2 (a b : ℝ) (h1 : a ≥ 0) (h2 : b ≥ 0) (h3 : a + b = 3) : 
  (1 / (a + 1) + 1 / (b + 2)) ≥ 2 / 3 :=
sorry

end part1_part2_l1865_186542


namespace simplify_and_rationalize_l1865_186579

theorem simplify_and_rationalize :
  ( (Real.sqrt 5 / Real.sqrt 6) * (Real.sqrt 10 / Real.sqrt 15) *
    (Real.sqrt 12 / Real.sqrt 20) = 1 / 3 ) :=
by
  sorry

end simplify_and_rationalize_l1865_186579


namespace ellipse_equation_l1865_186530

noncomputable def point := (ℝ × ℝ)

theorem ellipse_equation (a b : ℝ) (P Q : point) (h1 : a > b) (h2: b > 0) (e : ℝ) (h3 : e = 1/2)
  (h4 : P = (2, 3)) (h5 : Q = (2, -3))
  (h6 : (P.1^2)/(a^2) + (P.2^2)/(b^2) = 1) (h7 : (Q.1^2)/(a^2) + (Q.2^2)/(b^2) = 1) :
  (∀ x y: ℝ, (x^2/16 + y^2/12 = 1) ↔ (x^2/a^2 + y^2/b^2 = 1)) :=
sorry

end ellipse_equation_l1865_186530


namespace two_divides_a_squared_minus_a_three_divides_a_cubed_minus_a_l1865_186519

theorem two_divides_a_squared_minus_a (a : ℤ) : ∃ k₁ : ℤ, a^2 - a = 2 * k₁ :=
sorry

theorem three_divides_a_cubed_minus_a (a : ℤ) : ∃ k₂ : ℤ, a^3 - a = 3 * k₂ :=
sorry

end two_divides_a_squared_minus_a_three_divides_a_cubed_minus_a_l1865_186519


namespace solve_for_x_l1865_186540

theorem solve_for_x (x : ℝ) (h : 2 * x - 3 * x + 4 * x = 150) : x = 50 :=
by
  sorry

end solve_for_x_l1865_186540


namespace part1_assoc_eq_part2_k_range_part3_m_range_l1865_186588

-- Part 1
theorem part1_assoc_eq (x : ℝ) :
  (2 * (x + 1) - x = -3 ∧ (-4 < x ∧ x ≤ 4)) ∨ 
  ((x+1)/3 - 1 = x ∧ (-4 < x ∧ x ≤ 4)) ∨ 
  (2 * x - 7 = 0 ∧ (-4 < x ∧ x ≤ 4)) :=
sorry

-- Part 2
theorem part2_k_range (k : ℝ) :
  (∀ (x : ℝ), (x = (k + 6) / 2) → -5 < x ∧ x ≤ -3) ↔ (-16 < k) ∧ (k ≤ -12) :=
sorry 

-- Part 3
theorem part3_m_range (m : ℝ) :
  (∀ (x : ℝ), (x = 6 * m - 5) → (0 < x) ∧ (x ≤ 3 * m + 1) ∧ (1 ≤ x) ∧ (x ≤ 3)) ↔ (5/6 < m) ∧ (m < 1) :=
sorry

end part1_assoc_eq_part2_k_range_part3_m_range_l1865_186588


namespace percentage_increase_after_lawnmower_l1865_186535

-- Definitions from conditions
def initial_daily_yards := 8
def weekly_yards_after_lawnmower := 84
def days_in_week := 7

-- Problem statement
theorem percentage_increase_after_lawnmower : 
  ((weekly_yards_after_lawnmower / days_in_week - initial_daily_yards) / initial_daily_yards) * 100 = 50 := 
by 
  sorry

end percentage_increase_after_lawnmower_l1865_186535


namespace fraction_sum_eq_decimal_l1865_186528

theorem fraction_sum_eq_decimal : (2 / 5) + (2 / 50) + (2 / 500) = 0.444 := by
  sorry

end fraction_sum_eq_decimal_l1865_186528


namespace min_value_of_expression_l1865_186525

/-- Given the area of △ ABC is 2, and the sides opposite to angles A, B, C are a, b, c respectively,
    prove that the minimum value of a^2 + 2b^2 + 3c^2 is 8 * sqrt(11). -/
theorem min_value_of_expression
  (a b c : ℝ)
  (h₁ : 1/2 * b * c * Real.sin A = 2) :
  a^2 + 2 * b^2 + 3 * c^2 ≥ 8 * Real.sqrt 11 :=
sorry

end min_value_of_expression_l1865_186525


namespace alices_number_l1865_186515

theorem alices_number :
  ∃ (m : ℕ), (180 ∣ m) ∧ (45 ∣ m) ∧ (1000 ≤ m) ∧ (m ≤ 3000) ∧
    (m = 1260 ∨ m = 1440 ∨ m = 1620 ∨ m = 1800 ∨ m = 1980 ∨
     m = 2160 ∨ m = 2340 ∨ m = 2520 ∨ m = 2700 ∨ m = 2880) :=
by
  sorry

end alices_number_l1865_186515


namespace triangle_perimeter_l1865_186503

noncomputable def smallest_perimeter (a b c : ℕ) : ℕ :=
  a + b + c

theorem triangle_perimeter (a b c : ℕ) (A B C : ℝ) (h1 : A = 2 * B) 
  (h2 : C > π / 2) (h3 : a^2 = b * (b + c)) (h4 : ∃ m n : ℕ, b = m^2 ∧ b + c = n^2 ∧ a = m * n) :
  smallest_perimeter 28 16 33 = 77 :=
by sorry

end triangle_perimeter_l1865_186503


namespace tangent_length_to_circle_l1865_186551

-- Definitions capturing the conditions
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 4 * x - 2 * y + 1 = 0
def line_l (x y a : ℝ) : Prop := x + a * y - 1 = 0
def point_A (a : ℝ) : ℝ × ℝ := (-4, a)

-- Main theorem statement proving the question against the answer
theorem tangent_length_to_circle (a : ℝ) (x y : ℝ) (hC : circle_C x y) (hl : line_l 2 1 a) :
  (a = -1) -> (point_A a = (-4, -1)) -> ∃ b : ℝ, b = 6 := 
sorry

end tangent_length_to_circle_l1865_186551


namespace gcd_of_polynomial_l1865_186584

def multiple_of (a b : ℕ) : Prop := ∃ k : ℕ, a = k * b

theorem gcd_of_polynomial (b : ℕ) (h : multiple_of b 456) :
  Nat.gcd (4 * b^3 + b^2 + 6 * b + 152) b = 152 := sorry

end gcd_of_polynomial_l1865_186584


namespace range_of_2x_minus_y_l1865_186560

theorem range_of_2x_minus_y (x y : ℝ) (hx : 0 < x ∧ x < 4) (hy : 0 < y ∧ y < 6) : -6 < 2 * x - y ∧ 2 * x - y < 8 := 
sorry

end range_of_2x_minus_y_l1865_186560


namespace range_of_m_l1865_186570

theorem range_of_m (m : ℝ) (h : ∀ x : ℝ, (m + 1) * x^2 ≥ 0) : m > -1 :=
by
  sorry

end range_of_m_l1865_186570


namespace average_calculation_l1865_186522

def average (a b c : ℚ) : ℚ := (a + b + c) / 3
def pairAverage (a b : ℚ) : ℚ := (a + b) / 2

theorem average_calculation :
  average (average (pairAverage 2 2) 3 1) (pairAverage 1 2) 1 = 3 / 2 := sorry

end average_calculation_l1865_186522


namespace work_completion_in_6_days_l1865_186507

-- Definitions for the work rates of a, b, and c.
def work_rate_a_b : ℚ := 1 / 8
def work_rate_a : ℚ := 1 / 16
def work_rate_c : ℚ := 1 / 24

-- The theorem to prove that a, b, and c together can complete the work in 6 days.
theorem work_completion_in_6_days : 
  (1 / (work_rate_a_b - work_rate_a)) + work_rate_c = 1 / 6 :=
by
  sorry

end work_completion_in_6_days_l1865_186507


namespace perimeter_of_similar_triangle_l1865_186569

def is_isosceles (a b c : ℕ) : Prop :=
  (a = b) ∨ (b = c) ∨ (a = c)

theorem perimeter_of_similar_triangle (a b c : ℕ) (d e f : ℕ) 
  (h1 : is_isosceles a b c) (h2 : min a b = 15) (h3 : min (min a b) c = 15)
  (h4 : d = 75) (h5 : (d / 15) = e / b) (h6 : f = e) :
  d + e + f = 375 :=
by
  sorry

end perimeter_of_similar_triangle_l1865_186569


namespace sarah_rye_flour_l1865_186582

-- Definitions
variables (b c p t r : ℕ)

-- Conditions
def condition1 : Prop := b = 10
def condition2 : Prop := c = 3
def condition3 : Prop := p = 2
def condition4 : Prop := t = 20

-- Proposition to prove
theorem sarah_rye_flour : condition1 b → condition2 c → condition3 p → condition4 t → r = t - (b + c + p) → r = 5 :=
by
  intros h1 h2 h3 h4 hr
  rw [h1, h2, h3, h4] at hr
  exact hr

end sarah_rye_flour_l1865_186582
