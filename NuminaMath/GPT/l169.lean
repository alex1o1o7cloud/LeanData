import Mathlib

namespace number_of_ways_to_choose_4_captains_from_15_l169_169545

def choose_captains (n r : ℕ) : ℕ :=
  Nat.choose n r

theorem number_of_ways_to_choose_4_captains_from_15 :
  choose_captains 15 4 = 1365 := by
  sorry

end number_of_ways_to_choose_4_captains_from_15_l169_169545


namespace shopkeeper_total_cards_l169_169153

-- Definition of the number of cards in a complete deck
def cards_in_deck : Nat := 52

-- Definition of the number of complete decks the shopkeeper has
def number_of_decks : Nat := 3

-- Definition of the additional cards the shopkeeper has
def additional_cards : Nat := 4

-- The total number of cards the shopkeeper should have
def total_cards : Nat := number_of_decks * cards_in_deck + additional_cards

-- Theorem statement to prove the total number of cards is 160
theorem shopkeeper_total_cards : total_cards = 160 := by
  sorry

end shopkeeper_total_cards_l169_169153


namespace calculate_value_expression_l169_169163

theorem calculate_value_expression :
  3000 * (3000 ^ 3000 + 3000 ^ 2999) = 3001 * 3000 ^ 3000 := 
by
  sorry

end calculate_value_expression_l169_169163


namespace minimum_photos_taken_l169_169046

theorem minimum_photos_taken (A B C : Type) [finite A] [finite B] [finite C] 
  (tourists : fin 42) 
  (photo : tourists → fin 3 → Prop)
  (h1 : ∀ t : tourists, ∀ m : fin 3, photo t m ∨ ¬photo t m) 
  (h2 : ∀ t1 t2 : tourists, ∃ m : fin 3, photo t1 m ∨ photo t2 m) :
  ∃(n : ℕ), n = 123 ∧ ∀ (photo_count : nat), photo_count = ∑ t in fin_range 42, ∑ m in fin_range 3, if photo t m then 1 else 0 → photo_count ≥ n :=
sorry

end minimum_photos_taken_l169_169046


namespace solve_for_k_l169_169676

theorem solve_for_k : 
  ∃ (k : ℕ), k > 0 ∧ k * k = 2012 * 2012 + 2010 * 2011 * 2013 * 2014 ∧ k = 4048142 :=
sorry

end solve_for_k_l169_169676


namespace pinning_7_nails_l169_169450

theorem pinning_7_nails {n : ℕ} (circles : Fin n → Set (ℝ × ℝ)) :
  (∀ i j : Fin n, i ≠ j → ∃ p : ℝ × ℝ, p ∈ circles i ∧ p ∈ circles j) →
  ∃ s : Finset (ℝ × ℝ), s.card ≤ 7 ∧ ∀ i : Fin n, ∃ p : ℝ × ℝ, p ∈ s ∧ p ∈ circles i :=
by sorry

end pinning_7_nails_l169_169450


namespace total_cost_proof_l169_169603

def sandwich_cost : ℝ := 2.49
def soda_cost : ℝ := 1.87
def num_sandwiches : ℕ := 2
def num_sodas : ℕ := 4
def total_cost : ℝ := 12.46

theorem total_cost_proof : (num_sandwiches * sandwich_cost + num_sodas * soda_cost) = total_cost :=
by
  sorry

end total_cost_proof_l169_169603


namespace domain_of_sqrt_quadratic_l169_169713

open Set

def domain_of_f : Set ℝ := {x : ℝ | 2*x - x^2 ≥ 0}

theorem domain_of_sqrt_quadratic :
  domain_of_f = Icc 0 2 :=
by
  sorry

end domain_of_sqrt_quadratic_l169_169713


namespace shoveling_problem_l169_169128

variable (S : ℝ) -- Wayne's son's shoveling rate (driveways per hour)
variable (W : ℝ) -- Wayne's shoveling rate (driveways per hour)
variable (T : ℝ) -- Time it takes for Wayne's son to shovel the driveway alone (hours)

theorem shoveling_problem 
  (h1 : W = 6 * S)
  (h2 : (S + W) * 3 = 1) : T = 21 := 
by
  sorry

end shoveling_problem_l169_169128


namespace arithmetic_seq_geom_eq_div_l169_169193

noncomputable def a (n : ℕ) (a1 d : ℝ) : ℝ := a1 + n * d

theorem arithmetic_seq_geom_eq_div (a1 d : ℝ) (h1 : d ≠ 0) (h2 : a1 ≠ 0) 
    (h_geom : (a 3 a1 d) ^ 2 = (a 1 a1 d) * (a 7 a1 d)) :
    (a 2 a1 d + a 5 a1 d + a 8 a1 d) / (a 3 a1 d + a 4 a1 d) = 2 := 
by
  sorry

end arithmetic_seq_geom_eq_div_l169_169193


namespace volume_of_triangular_prism_l169_169706

theorem volume_of_triangular_prism (S_side_face : ℝ) (distance : ℝ) :
  ∃ (Volume_prism : ℝ), Volume_prism = 1/2 * (S_side_face * distance) :=
by sorry

end volume_of_triangular_prism_l169_169706


namespace choose_4_from_15_is_1365_l169_169547

theorem choose_4_from_15_is_1365 : nat.choose 15 4 = 1365 :=
by
  sorry

end choose_4_from_15_is_1365_l169_169547


namespace right_triangle_sides_l169_169996

theorem right_triangle_sides (m n : ℝ) (x : ℝ) (a b c : ℝ)
  (h1 : 2 * x < m + n) 
  (h2 : a = Real.sqrt (2 * m * n) - m)
  (h3 : b = Real.sqrt (2 * m * n) - n)
  (h4 : c = m + n - Real.sqrt (2 * m * n))
  (h5 : a^2 + b^2 = c^2)
  (h6 : 4 * x^2 = (m - 2 * x)^2 + (n - 2 * x)^2) :
  a = Real.sqrt (2 * m * n) - m ∧ b = Real.sqrt (2 * m * n) - n ∧ c = m + n - Real.sqrt (2 * m * n) :=
by
  sorry

end right_triangle_sides_l169_169996


namespace total_people_wearing_hats_l169_169710

variable (total_adults : ℕ) (total_children : ℕ)
variable (half_adults : ℕ) (women : ℕ) (men : ℕ)
variable (women_with_hats : ℕ) (men_with_hats : ℕ)
variable (children_with_hats : ℕ)
variable (total_with_hats : ℕ)

-- Given conditions
def conditions : Prop :=
  total_adults = 1800 ∧
  total_children = 200 ∧
  half_adults = total_adults / 2 ∧
  women = half_adults ∧
  men = half_adults ∧
  women_with_hats = (25 * women) / 100 ∧
  men_with_hats = (12 * men) / 100 ∧
  children_with_hats = (10 * total_children) / 100 ∧
  total_with_hats = women_with_hats + men_with_hats + children_with_hats

-- Proof goal
theorem total_people_wearing_hats : conditions total_adults total_children half_adults women men women_with_hats men_with_hats children_with_hats total_with_hats → total_with_hats = 353 :=
by
  intros h
  sorry

end total_people_wearing_hats_l169_169710


namespace probability_not_e_after_n_spins_l169_169933

theorem probability_not_e_after_n_spins
    (S : Type)
    (e b c d : S)
    (p_e : ℝ)
    (p_b : ℝ)
    (p_c : ℝ)
    (p_d : ℝ) :
    (p_e = 0.25) →
    (p_b = 0.25) →
    (p_c = 0.25) →
    (p_d = 0.25) →
    (1 - p_e)^2 = 0.5625 :=
by
  sorry

end probability_not_e_after_n_spins_l169_169933


namespace intersection_of_lines_l169_169169

theorem intersection_of_lines : 
  (∃ x y : ℚ, y = -3 * x + 1 ∧ y = 5 * x + 4) ↔ 
  (∃ x y : ℚ, x = -3 / 8 ∧ y = 17 / 8) :=
by
  sorry

end intersection_of_lines_l169_169169


namespace cos_double_angle_of_tan_half_l169_169780

theorem cos_double_angle_of_tan_half (α : ℝ) (h : Real.tan α = 1 / 2) :
  Real.cos (2 * α) = 3 / 5 :=
sorry

end cos_double_angle_of_tan_half_l169_169780


namespace xiao_ming_min_correct_answers_l169_169115

theorem xiao_ming_min_correct_answers (x : ℕ) : (10 * x - 5 * (20 - x) > 100) → (x ≥ 14) := by
  sorry

end xiao_ming_min_correct_answers_l169_169115


namespace find_b_l169_169042

theorem find_b (a b : ℝ) (h₁ : ∀ x y, y = 0.75 * x + 1 → (4, b) = (x, y))
                (h₂ : k = 0.75) : b = 4 :=
by sorry

end find_b_l169_169042


namespace find_a_l169_169778

theorem find_a 
  (x y a : ℝ)
  (h₁ : x - 3 ≤ 0)
  (h₂ : y - a ≤ 0)
  (h₃ : x + y ≥ 0)
  (h₄ : ∃ (x y : ℝ), 2*x + y = 10): a = 4 :=
sorry

end find_a_l169_169778


namespace jam_fraction_left_l169_169062

theorem jam_fraction_left:
  let jam_total := 1 in
  let lunch_fraction := 1 / 3 in
  let after_lunch := jam_total - lunch_fraction in
  let dinner_fraction := 1 / 7 * after_lunch in
  let after_dinner := after_lunch - dinner_fraction in
  after_dinner = 4 / 7 := 
by {
  sorry
}

end jam_fraction_left_l169_169062


namespace hyperbola_focus_l169_169649

theorem hyperbola_focus :
  ∃ (x y : ℝ), 2 * x^2 - y^2 - 8 * x + 4 * y - 4 = 0 ∧ (x, y) = (2 + 2 * Real.sqrt 3, 2) :=
by
  -- The proof would go here
  sorry

end hyperbola_focus_l169_169649


namespace smallest_sum_xy_min_45_l169_169381

theorem smallest_sum_xy_min_45 (x y : ℕ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x ≠ y) (h4 : 1 / (x : ℝ) + 1 / (y : ℝ) = 1 / 10) :
  x + y = 45 :=
by
  sorry

end smallest_sum_xy_min_45_l169_169381


namespace total_games_l169_169275

-- Definitions and conditions
noncomputable def num_teams : ℕ := 12

noncomputable def regular_season_games_each : ℕ := 4

noncomputable def knockout_games_each : ℕ := 2

-- Calculate total number of games
theorem total_games : (num_teams * (num_teams - 1) / 2) * regular_season_games_each + 
                      (num_teams * knockout_games_each / 2) = 276 :=
by
  -- This is the statement to be proven
  sorry

end total_games_l169_169275


namespace delta_delta_delta_45_l169_169903

def delta (P : ℚ) : ℚ := (2 / 3) * P + 2

theorem delta_delta_delta_45 :
  delta (delta (delta 45)) = 158 / 9 :=
by sorry

end delta_delta_delta_45_l169_169903


namespace volleyball_tournament_l169_169550

theorem volleyball_tournament (n : ℕ) (h : n * (n - 1) / 2 = 28) : n = 8 :=
sorry

end volleyball_tournament_l169_169550


namespace total_savings_correct_l169_169261

-- Define the savings of Sam, Victory and Alex according to the given conditions
def sam_savings : ℕ := 1200
def victory_savings : ℕ := sam_savings - 200
def alex_savings : ℕ := 2 * victory_savings

-- Define the total savings
def total_savings : ℕ := sam_savings + victory_savings + alex_savings

-- The theorem to prove the total savings
theorem total_savings_correct : total_savings = 4200 :=
by
  sorry

end total_savings_correct_l169_169261


namespace probability_red_ball_l169_169041

variable (num_white_balls : ℕ) (num_red_balls : ℕ)
variable (total_balls : ℕ := num_white_balls + num_red_balls)
variable (favorable_outcomes : ℕ := num_red_balls)
variable (probability_of_red : ℚ := favorable_outcomes / total_balls)

theorem probability_red_ball
  (h1 : num_white_balls = 3)
  (h2 : num_red_balls = 7) :
  probability_of_red = 7 / 10 := by
  sorry

end probability_red_ball_l169_169041


namespace largest_circle_area_l169_169485

theorem largest_circle_area (x y : ℝ) (h1 : x + y = 30) (h2 : x * y = 200) :
  ∃ r : ℝ, (2 * π * r = 60) ∧ (π * r ^ 2 = 900 / π) := 
sorry

end largest_circle_area_l169_169485


namespace floor_sqrt_120_eq_10_l169_169909

theorem floor_sqrt_120_eq_10 :
  (√120).to_floor = 10 := by
  have h1 : √100 = 10 := by norm_num
  have h2 : √121 = 11 := by norm_num
  have h : 100 < 120 ∧ 120 < 121 := by norm_num
  have sqrt_120 : 10 < √120 ∧ √120 < 11 :=
    by exact ⟨real.sqrt_lt' 120 121 h.2, real.sqrt_lt'' 100 120 h.1⟩
  sorry

end floor_sqrt_120_eq_10_l169_169909


namespace no_nat_numbers_m_n_satisfy_eq_l169_169343

theorem no_nat_numbers_m_n_satisfy_eq (m n : ℕ) : ¬ (m^2 = n^2 + 2014) := sorry

end no_nat_numbers_m_n_satisfy_eq_l169_169343


namespace james_carrot_sticks_l169_169814

theorem james_carrot_sticks (total_carrots : ℕ) (after_dinner_carrots : ℕ) (before_dinner_carrots : ℕ) 
  (h1 : total_carrots = 37) (h2 : after_dinner_carrots = 15) :
  before_dinner_carrots = total_carrots - after_dinner_carrots :=
by
suffices h : 37 - 15 = 22 by
  rw [← h1, ← h2]
  exact h
apply rfl

end james_carrot_sticks_l169_169814


namespace jesse_remaining_pages_l169_169064

theorem jesse_remaining_pages (pages_read : ℕ)
  (h1 : pages_read = 83)
  (h2 : pages_read = (1 / 3 : ℝ) * total_pages)
  : pages_remaining = 166 :=
  by 
    -- Here we would build the proof, skipped with sorry
    sorry

end jesse_remaining_pages_l169_169064


namespace y_power_x_equals_49_l169_169029

theorem y_power_x_equals_49 (x y : ℝ) (h : |x - 2| = -(y + 7)^2) : y ^ x = 49 := by
  sorry

end y_power_x_equals_49_l169_169029


namespace solve_inner_parentheses_l169_169611

theorem solve_inner_parentheses (x : ℝ) : 
  45 - (28 - (37 - (15 - x))) = 57 ↔ x = 18 := by
  sorry

end solve_inner_parentheses_l169_169611


namespace common_difference_l169_169224

-- Definitions
variable (a₁ d : ℝ) -- First term and common difference of the arithmetic sequence

-- Conditions
def mean_nine_terms (a₁ d : ℝ) : Prop :=
  (1 / 2 * (2 * a₁ + 8 * d)) = 10

def mean_ten_terms (a₁ d : ℝ) : Prop :=
  (1 / 2 * (2 * a₁ + 9 * d)) = 13

-- Theorem to prove the common difference is 6
theorem common_difference (a₁ d : ℝ) :
  mean_nine_terms a₁ d → 
  mean_ten_terms a₁ d → 
  d = 6 := by
  intros
  sorry

end common_difference_l169_169224


namespace total_get_well_cards_l169_169976

-- Definitions for the number of cards received in each place
def cardsInHospital : ℕ := 403
def cardsAtHome : ℕ := 287

-- Theorem statement:
theorem total_get_well_cards : cardsInHospital + cardsAtHome = 690 := by
  sorry

end total_get_well_cards_l169_169976


namespace full_price_ticket_revenue_l169_169136

-- Given conditions
variable {f d p : ℕ}
variable (h1 : f + d = 160)
variable (h2 : f * p + d * (2 * p / 3) = 2800)

-- Goal: Prove the full-price ticket revenue is 1680.
theorem full_price_ticket_revenue : f * p = 1680 :=
sorry

end full_price_ticket_revenue_l169_169136


namespace maria_money_left_l169_169071

def ticket_cost : ℕ := 300
def hotel_cost : ℕ := ticket_cost / 2
def transportation_cost : ℕ := 80
def num_days : ℕ := 5
def avg_meal_cost_per_day : ℕ := 40
def tourist_tax_rate : ℚ := 0.10
def starting_amount : ℕ := 760

def total_meal_cost : ℕ := num_days * avg_meal_cost_per_day
def expenses_subject_to_tax := hotel_cost + transportation_cost
def tourist_tax := tourist_tax_rate * expenses_subject_to_tax
def total_expenses := ticket_cost + hotel_cost + transportation_cost + total_meal_cost + tourist_tax
def money_left := starting_amount - total_expenses

theorem maria_money_left : money_left = 7 := by
  sorry

end maria_money_left_l169_169071


namespace number_of_uncracked_seashells_l169_169116

theorem number_of_uncracked_seashells (toms_seashells freds_seashells cracked_seashells : ℕ) 
  (h_tom : toms_seashells = 15) 
  (h_fred : freds_seashells = 43) 
  (h_cracked : cracked_seashells = 29) : 
  toms_seashells + freds_seashells - cracked_seashells = 29 :=
by
  sorry

end number_of_uncracked_seashells_l169_169116


namespace simplify_P_eq_l169_169167

noncomputable def P (x y : ℝ) : ℝ := (x^2 - y^2) / (x * y) - (x * y - y^2) / (x * y - x^2)

theorem simplify_P_eq (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hxy: x ≠ y) : P x y = x / y := 
by
  -- Insert proof here
  sorry

end simplify_P_eq_l169_169167


namespace keith_attended_games_l169_169256

-- Definitions from the conditions
def total_games : ℕ := 20
def missed_games : ℕ := 9

-- The statement to prove
theorem keith_attended_games : (total_games - missed_games) = 11 :=
by
  sorry

end keith_attended_games_l169_169256


namespace simplified_equation_equivalent_l169_169967

theorem simplified_equation_equivalent  (x : ℝ) :
    (x / 0.3 = 1 + (1.2 - 0.3 * x) / 0.2) ↔ (10 * x / 3 = 1 + (12 - 3 * x) / 2) :=
by sorry

end simplified_equation_equivalent_l169_169967


namespace turtle_reaches_waterhole_in_28_minutes_l169_169740

-- Definitions
constant x : ℝ
constant turtle_speed : ℝ := 1 / 30
constant lion1_time_to_waterhole : ℝ := 5
constant turtle_time_to_waterhole : ℝ := 30

-- Speeds of the Lion Cubs
def lion1_speed := x
def lion2_speed := 1.5 * x

-- Time for the lion cubs to meet
def meeting_time := lion1_time_to_waterhole / (1 + lion2_speed / lion1_speed)

-- Distance traveled by the turtle in the meeting time
def turtle_distance_covered := turtle_speed * meeting_time

-- Remaining distance for the turtle
def remaining_turtle_distance := 1 - turtle_distance_covered

-- Time for the turtle to cover the remaining distance
def turtle_remaining_time := remaining_turtle_distance * 30

-- Prove that the turtle takes 28 minutes after the meeting to reach the waterhole
theorem turtle_reaches_waterhole_in_28_minutes : turtle_remaining_time = 28 :=
by
  -- Placeholder for the actual proof
  sorry

end turtle_reaches_waterhole_in_28_minutes_l169_169740


namespace melanie_picked_plums_l169_169235

variable (picked_plums : ℕ)
variable (given_plums : ℕ := 3)
variable (total_plums : ℕ := 10)

theorem melanie_picked_plums :
  picked_plums + given_plums = total_plums → picked_plums = 7 := by
  sorry

end melanie_picked_plums_l169_169235


namespace isosceles_triangle_base_length_l169_169521

theorem isosceles_triangle_base_length (P Q : ℕ) (x y : ℕ) (hP : P = 15) (hQ : Q = 12) (hPerimeter : 2 * x + y = 27) 
      (hCondition : (y = P ∧ (1 / 2) * x + x = P) ∨ (y = Q ∧ (1 / 2) * x + x = Q)) : 
  y = 7 ∨ y = 11 :=
sorry

end isosceles_triangle_base_length_l169_169521


namespace zoe_total_money_l169_169868

def numberOfPeople : ℕ := 6
def sodaCostPerBottle : ℝ := 0.5
def pizzaCostPerSlice : ℝ := 1.0

theorem zoe_total_money :
  numberOfPeople * sodaCostPerBottle + numberOfPeople * pizzaCostPerSlice = 9 := 
by
  sorry

end zoe_total_money_l169_169868


namespace other_root_is_seven_thirds_l169_169835

theorem other_root_is_seven_thirds {m : ℝ} (h : ∃ r : ℝ, 3 * r * r + m * r - 7 = 0 ∧ r = -1) : 
  ∃ r' : ℝ, r' ≠ -1 ∧ 3 * r' * r' + m * r' - 7 = 0 ∧ r' = 7 / 3 :=
by
  sorry

end other_root_is_seven_thirds_l169_169835


namespace length_more_than_breadth_l169_169103

theorem length_more_than_breadth (b x : ℝ) (h1 : b + x = 61) (h2 : 26.50 * (4 * b + 2 * x) = 5300) : x = 22 :=
by
  sorry

end length_more_than_breadth_l169_169103


namespace work_done_by_student_l169_169130

theorem work_done_by_student
  (M : ℝ)  -- mass of the student
  (m : ℝ)  -- mass of the stone
  (h : ℝ)  -- height from which the stone is thrown
  (L : ℝ)  -- distance on the ice where the stone lands
  (g : ℝ)  -- acceleration due to gravity
  (t : ℝ := Real.sqrt (2 * h / g))  -- time it takes for the stone to hit the ice derived from free fall equation
  (Vk : ℝ := L / t)  -- initial speed of the stone derived from horizontal motion
  (Vu : ℝ := m / M * Vk)  -- initial speed of the student derived from conservation of momentum
  : (1/2 * m * Vk^2 + (1/2) * M * Vu^2) = 126.74 :=
by
  sorry

end work_done_by_student_l169_169130


namespace distance_between_cities_l169_169101

theorem distance_between_cities:
    ∃ (x y : ℝ),
    (x = 135) ∧
    (y = 175) ∧
    (7 / 9 * x = 105) ∧
    (x + 7 / 9 * x + y = 415) ∧
    (x = 27 / 35 * y) :=
by
  sorry

end distance_between_cities_l169_169101


namespace number_of_solutions_l169_169507

theorem number_of_solutions (x y : ℕ) : (3 * x + 2 * y = 1001) → ∃! (n : ℕ), n = 167 := by
  sorry

end number_of_solutions_l169_169507


namespace altitude_inequality_l169_169919

theorem altitude_inequality
  (a b m_a m_b : ℝ)
  (h1 : a > b)
  (h2 : a * m_a = b * m_b) :
  a^2010 + m_a^2010 ≥ b^2010 + m_b^2010 :=
sorry

end altitude_inequality_l169_169919


namespace paul_bags_on_saturday_l169_169702

-- Definitions and Conditions
def total_cans : ℕ := 72
def cans_per_bag : ℕ := 8
def extra_bags : ℕ := 3

-- Statement of the problem
theorem paul_bags_on_saturday (S : ℕ) :
  S * cans_per_bag = total_cans - (extra_bags * cans_per_bag) →
  S = 6 :=
sorry

end paul_bags_on_saturday_l169_169702


namespace sum_of_non_solutions_l169_169418

noncomputable def A : ℚ := 3
noncomputable def B : ℚ := -45 / 13
noncomputable def C : ℚ := -70 / 13

def non_solution_sum : ℚ := -9 - 70 / 13

theorem sum_of_non_solutions :
  non_solution_sum = -187 / 13 := by
sorry

end sum_of_non_solutions_l169_169418


namespace oranges_in_buckets_l169_169255

theorem oranges_in_buckets :
  ∀ (x : ℕ),
  (22 + x + (x - 11) = 89) →
  (x - 22 = 17) :=
by
  intro x h
  sorry

end oranges_in_buckets_l169_169255


namespace min_value_expression_l169_169506

noncomputable def expression (x : ℝ) : ℝ :=
  (15 - x) * (8 - x) * (15 + x) * (8 + x)

theorem min_value_expression : ∃ x : ℝ, expression x = -6480.25 :=
sorry

end min_value_expression_l169_169506


namespace largest_abs_val_among_2_3_neg3_neg4_l169_169626

def abs_val (a : Int) : Nat := a.natAbs

theorem largest_abs_val_among_2_3_neg3_neg4 : 
  ∀ (x : Int), x ∈ [2, 3, -3, -4] → abs_val x ≤ abs_val (-4) := by
  sorry

end largest_abs_val_among_2_3_neg3_neg4_l169_169626


namespace cost_per_gallon_is_45_l169_169816

variable (totalArea coverage cost_jason cost_jeremy dollars_per_gallon : ℕ)

-- Conditions
def total_area := 1600
def coverage_per_gallon := 400
def num_coats := 2
def contribution_jason := 180
def contribution_jeremy := 180

-- Gallons needed calculation
def gallons_per_coat := total_area / coverage_per_gallon
def total_gallons := gallons_per_coat * num_coats

-- Total cost calculation
def total_cost := contribution_jason + contribution_jeremy

-- Cost per gallon calculation
def cost_per_gallon := total_cost / total_gallons

-- Proof statement
theorem cost_per_gallon_is_45 : cost_per_gallon = 45 :=
by
  sorry

end cost_per_gallon_is_45_l169_169816


namespace find_b_l169_169844

theorem find_b (a b c : ℕ) (h₁ : 1 < a) (h₂ : 1 < b) (h₃ : 1 < c):
  (∀ N : ℝ, N ≠ 1 → (N^(3/a) * N^(2/(ab)) * N^(1/(abc)) = N^(39/48))) → b = 4 :=
  by
  sorry

end find_b_l169_169844


namespace no_nat_solutions_m_sq_eq_n_sq_plus_2014_l169_169352

theorem no_nat_solutions_m_sq_eq_n_sq_plus_2014 :
  ¬ ∃ (m n : ℕ), m ^ 2 = n ^ 2 + 2014 := 
sorry

end no_nat_solutions_m_sq_eq_n_sq_plus_2014_l169_169352


namespace equidistant_point_quadrants_l169_169205

theorem equidistant_point_quadrants :
  ∀ (x y : ℝ), 3 * x + 5 * y = 15 → (|x| = |y| → (x > 0 → y > 0 ∧ x = y ∧ y = x) ∧ (x < 0 → y > 0 ∧ x = -y ∧ -x = y)) := 
by
  sorry

end equidistant_point_quadrants_l169_169205


namespace train_speed_is_117_l169_169885

noncomputable def train_speed (train_length : ℝ) (crossing_time : ℝ) (man_speed_kmph : ℝ) : ℝ :=
  let man_speed_mps := man_speed_kmph * 1000 / 3600
  let relative_speed := train_length / crossing_time
  (relative_speed - man_speed_mps) * 3.6

theorem train_speed_is_117 :
  train_speed 300 9 3 = 117 :=
by
  -- We leave the proof as sorry since only the statement is needed
  sorry

end train_speed_is_117_l169_169885


namespace total_amount_paid_l169_169557

-- Definitions of the conditions
def cost_earbuds : ℝ := 200
def tax_rate : ℝ := 0.15

-- Statement to prove
theorem total_amount_paid : (cost_earbuds + (cost_earbuds * tax_rate)) = 230 := sorry

end total_amount_paid_l169_169557


namespace total_dots_is_78_l169_169628

-- Define the conditions as Lean definitions
def ladybugs_monday : ℕ := 8
def ladybugs_tuesday : ℕ := 5
def dots_per_ladybug : ℕ := 6

-- Define the total number of ladybugs
def total_ladybugs : ℕ := ladybugs_monday + ladybugs_tuesday

-- Define the total number of dots
def total_dots : ℕ := total_ladybugs * dots_per_ladybug

-- Theorem stating the problem to solve
theorem total_dots_is_78 : total_dots = 78 := by
  sorry

end total_dots_is_78_l169_169628


namespace problem1_problem2_l169_169496

theorem problem1 : 
  (5 / 7 : ℚ) * (-14 / 3) / (5 / 3) = -2 := 
by 
  sorry

theorem problem2 : 
  (-15 / 7 : ℚ) / (-6 / 5) * (-7 / 5) = -5 / 2 := 
by 
  sorry

end problem1_problem2_l169_169496


namespace no_nat_numbers_m_n_satisfy_eq_l169_169342

theorem no_nat_numbers_m_n_satisfy_eq (m n : ℕ) : ¬ (m^2 = n^2 + 2014) := sorry

end no_nat_numbers_m_n_satisfy_eq_l169_169342


namespace area_of_BDOE_l169_169980

namespace Geometry

noncomputable def areaQuadrilateralBDOE (AE CD AB BC AC : ℝ) : ℝ :=
  if AE = 2 ∧ CD = 11 ∧ AB = 8 ∧ BC = 8 ∧ AC = 6 then
    189 * Real.sqrt 55 / 88
  else
    0

theorem area_of_BDOE :
  areaQuadrilateralBDOE 2 11 8 8 6 = 189 * Real.sqrt 55 / 88 :=
by 
  sorry

end Geometry

end area_of_BDOE_l169_169980


namespace choose_15_4_l169_169549

/-- The number of ways to choose 4 captains from a team of 15 people is 1365. -/
theorem choose_15_4 : nat.choose 15 4 = 1365 := by
  sorry

end choose_15_4_l169_169549


namespace gcd_of_17420_23826_36654_l169_169650

theorem gcd_of_17420_23826_36654 : Nat.gcd (Nat.gcd 17420 23826) 36654 = 2 := 
by 
  sorry

end gcd_of_17420_23826_36654_l169_169650


namespace sequence_fifth_term_l169_169380

theorem sequence_fifth_term (a : ℕ → ℕ) (h₁ : a 1 = 1) (h₂ : a 2 = 2)
    (h₃ : ∀ n > 2, a n = a (n-1) + a (n-2)) : a 5 = 8 :=
sorry

end sequence_fifth_term_l169_169380


namespace product_roots_positive_real_part_l169_169800

open Complex

theorem product_roots_positive_real_part :
    (∃ (roots : Fin 6 → ℂ),
       (∀ k, roots k ^ 6 = -64) ∧
       (∀ k, (roots k).re > 0 → (roots 0).re > 0 ∧ (roots 0).im > 0 ∧
                               (roots 1).re > 0 ∧ (roots 1).im < 0) ∧
       (roots 0 * roots 1 = 4)
    ) :=
sorry

end product_roots_positive_real_part_l169_169800


namespace minimum_distance_PQ_l169_169924

open Real

noncomputable def minimum_distance (t : ℝ) : ℝ := 
  (|t - 1|) / (sqrt (1 + t ^ 2))

theorem minimum_distance_PQ :
  let t := sqrt 2 / 2
  let x_P := 2
  let y_P := 0
  let x_Q := -1 + t
  let y_Q := 2 + t
  let d := minimum_distance (x_Q - y_Q + 3)
  (d - 2) = (5 * sqrt 2) / 2 - 2 :=
sorry

end minimum_distance_PQ_l169_169924


namespace proof_ab_value_l169_169798

theorem proof_ab_value (a b : ℝ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 29) : a * b = 10 := 
by
  sorry

end proof_ab_value_l169_169798


namespace area_of_triangle_ABC_equation_of_circumcircle_l169_169685

-- Define points A, B, and C
structure Point where
  x : ℝ
  y : ℝ

def A : Point := { x := 1, y := 2 }
def B : Point := { x := 1, y := 3 }
def C : Point := { x := 3, y := 6 }

-- Theorem to prove the area of triangle ABC
theorem area_of_triangle_ABC : 
  let base := |B.y - A.y|
  let height := |C.x - A.x|
  (1/2) * base * height = 1 := sorry

-- Theorem to prove the equation of the circumcircle of triangle ABC
theorem equation_of_circumcircle : 
  let D := -10
  let E := -5
  let F := 15
  ∀ (x y : ℝ), (x - 5)^2 + (y - 5/2)^2 = 65/4 ↔ 
                x^2 + y^2 + D * x + E * y + F = 0 := sorry

end area_of_triangle_ABC_equation_of_circumcircle_l169_169685


namespace problem_given_conditions_l169_169515

noncomputable def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

theorem problem_given_conditions {S a : ℕ → ℝ} (a : ℕ → ℝ) :
  (arithmetic_sequence a (-3)) →
  (S 13 = -26) →
  (a 9 = 4) →
  (∀ n, a n = 31 - 3 * n) ∧ 
  (∀ n, ∑ k in Finset.range n, a (2 * k + 1) = -3 * n^2 + 34 * n) :=
begin
  intros seq sum_cond a9_cond,
  split,
  {
    intro n,
    -- Proof of the general term of the sequence
    sorry,
  },
  {
    intro n,
    -- Proof of the sum of series
    sorry,
  }
end

end problem_given_conditions_l169_169515


namespace max_bicycle_distance_l169_169745

-- Define the properties of the tires
def front_tire_duration : ℕ := 5000
def rear_tire_duration : ℕ := 3000

-- Define the maximum distance the bicycle can travel
def max_distance : ℕ := 3750

-- The main statement to be proven (proof is not required)
theorem max_bicycle_distance 
  (swap_usage : ∀ (d1 d2 : ℕ), d1 + d2 <= front_tire_duration + rear_tire_duration) : 
  ∃ (x : ℕ), x = max_distance := 
sorry

end max_bicycle_distance_l169_169745


namespace emails_difference_l169_169413

theorem emails_difference
  (emails_morning : ℕ)
  (emails_afternoon : ℕ)
  (h_morning : emails_morning = 10)
  (h_afternoon : emails_afternoon = 3)
  : emails_morning - emails_afternoon = 7 := by
  sorry

end emails_difference_l169_169413


namespace infinite_cube_volume_sum_l169_169712

noncomputable def sum_of_volumes_of_infinite_cubes (a : ℝ) : ℝ :=
  ∑' n, (((a / (3 ^ n))^3))

theorem infinite_cube_volume_sum (a : ℝ) : sum_of_volumes_of_infinite_cubes a = (27 / 26) * a^3 :=
sorry

end infinite_cube_volume_sum_l169_169712


namespace max_three_digit_sum_l169_169211

theorem max_three_digit_sum (A B C : ℕ) (hA : A < 10) (hB : B < 10) (hC : C < 10) (h1 : A ≠ B) (h2 : B ≠ C) (h3 : C ≠ A) :
  110 * A + 10 * B + 3 * C ≤ 981 :=
sorry

end max_three_digit_sum_l169_169211


namespace problem_min_a2_area_l169_169687

noncomputable def area (a b c : ℝ) (A B C : ℝ) : ℝ := 
  0.5 * b * c * Real.sin A

noncomputable def min_a2_area (a b c : ℝ) (A B C : ℝ): ℝ := 
  let S := area a b c A B C
  a^2 / S

theorem problem_min_a2_area :
  ∀ (a b c A B C : ℝ), 
    a > 0 → b > 0 → c > 0 → 
    A + B + C = Real.pi →
    a / Real.sin A = b / Real.sin B ∧ a / Real.sin A = c / Real.sin C →
    b * Real.cos C + c * Real.cos B = 3 * a * Real.cos A →
    min_a2_area a b c A B C ≥ 2 * Real.sqrt 2 :=
by
  sorry

end problem_min_a2_area_l169_169687


namespace baseball_league_games_l169_169409

theorem baseball_league_games (n m : ℕ) (h : 3 * n + 4 * m = 76) (h1 : n > 2 * m) (h2 : m > 4) : n = 16 :=
by 
  sorry

end baseball_league_games_l169_169409


namespace cricket_bat_cost_l169_169152

variable (CP_A : ℝ) (CP_B : ℝ) (CP_C : ℝ)

-- Conditions
def CP_B_def : Prop := CP_B = 1.20 * CP_A
def CP_C_def : Prop := CP_C = 1.25 * CP_B
def CP_C_val : Prop := CP_C = 234

-- Theorem statement
theorem cricket_bat_cost (h1 : CP_B_def CP_A CP_B) (h2 : CP_C_def CP_B CP_C) (h3 : CP_C_val CP_C) : CP_A = 156 :=by
  sorry

end cricket_bat_cost_l169_169152


namespace polygon_sides_l169_169751

/-- 
A regular polygon with interior angles of 160 degrees has 18 sides.
-/
theorem polygon_sides (n : ℕ) (h : ∀ (i : ℕ), i < n → (interior_angle : ℝ) = 160) : n = 18 := 
by
  have angle_sum : 180 * (n - 2) = 160 * n := 
    by sorry
  have eq_sides : n = 18 := 
    by sorry
  exact eq_sides

end polygon_sides_l169_169751


namespace find_n_expansion_l169_169510

theorem find_n_expansion : 
  (∃ n : ℕ, 4^n + 2^n = 1056) → n = 5 :=
by sorry

end find_n_expansion_l169_169510


namespace prism_faces_even_or_odd_l169_169865

theorem prism_faces_even_or_odd (n : ℕ) (hn : 3 ≤ n) : ¬ (2 + n) % 2 = 1 :=
by
  sorry

end prism_faces_even_or_odd_l169_169865


namespace problem_statement_l169_169194

noncomputable def proposition_p (x : ℝ) : Prop := ∃ x0 : ℝ, x0 - 2 > 0
noncomputable def proposition_q (x : ℝ) : Prop := ∀ x : ℝ, (2:ℝ)^x > x^2

theorem problem_statement : ∃ (p q : Prop), (∃ x0 : ℝ, x0 - 2 > 0) ∧ (¬ (∀ x : ℝ, (2:ℝ)^x > x^2)) :=
by
  sorry

end problem_statement_l169_169194


namespace time_difference_l169_169175

-- Definitions
def time_chinese : ℕ := 5
def time_english : ℕ := 7

-- Statement to prove
theorem time_difference : time_english - time_chinese = 2 := by
  -- Proof goes here
  sorry

end time_difference_l169_169175


namespace find_a7_l169_169222

variable {a : ℕ → ℕ}  -- Define the geometric sequence as a function from natural numbers to natural numbers.
variable (h_geo_seq : ∀ (n k : ℕ), a n ^ 2 = a (n - k) * a (n + k)) -- property of geometric sequences
variable (h_a3 : a 3 = 2) -- given a₃ = 2
variable (h_a5 : a 5 = 8) -- given a₅ = 8

theorem find_a7 : a 7 = 32 :=
by
  sorry

end find_a7_l169_169222


namespace big_al_bananas_l169_169761

/-- Big Al ate 140 bananas from May 1 through May 6. Each day he ate five more bananas than on the previous day. On May 4, Big Al did not eat any bananas due to fasting. Prove that Big Al ate 38 bananas on May 6. -/
theorem big_al_bananas : 
  ∃ a : ℕ, (a + (a + 5) + (a + 10) + 0 + (a + 15) + (a + 20) = 140) ∧ ((a + 20) = 38) :=
by sorry

end big_al_bananas_l169_169761


namespace minimum_value_of_f_on_interval_l169_169400

noncomputable def f (x : ℝ) : ℝ := - (1 / 2) * x^2 + Real.log x

theorem minimum_value_of_f_on_interval :
  (∀ x ∈ (Set.Icc (1 / Real.exp 1) (Real.exp 1)), f x ≥ f (Real.exp 1)) ∧
  ∃ x ∈ (Set.Icc (1 / Real.exp 1) (Real.exp 1)), f x = f (Real.exp 1) := 
by
  sorry

end minimum_value_of_f_on_interval_l169_169400


namespace persons_in_boat_l169_169711

theorem persons_in_boat (W1 W2 new_person_weight : ℝ) (n : ℕ)
  (hW1 : W1 = 55)
  (h_new_person : new_person_weight = 50)
  (hW2 : W2 = W1 - 5) :
  (n * W1 + new_person_weight) / (n + 1) = W2 → false :=
by
  intros h_eq
  sorry

end persons_in_boat_l169_169711


namespace no_nat_m_n_square_diff_2014_l169_169338

theorem no_nat_m_n_square_diff_2014 :
  ¬ ∃ (m n : ℕ), m^2 = n^2 + 2014 :=
by
  sorry

end no_nat_m_n_square_diff_2014_l169_169338


namespace no_nat_solutions_for_m2_eq_n2_plus_2014_l169_169362

theorem no_nat_solutions_for_m2_eq_n2_plus_2014 :
  ∀ m n : ℕ, ¬(m^2 = n^2 + 2014) := by
sorry

end no_nat_solutions_for_m2_eq_n2_plus_2014_l169_169362


namespace jan_uses_24_gallons_for_plates_and_clothes_l169_169108

theorem jan_uses_24_gallons_for_plates_and_clothes :
  (65 - (2 * 7 + (2 * 7 - 11))) / 2 = 24 :=
by sorry

end jan_uses_24_gallons_for_plates_and_clothes_l169_169108


namespace no_nat_m_n_square_diff_2014_l169_169332

theorem no_nat_m_n_square_diff_2014 :
  ¬ ∃ (m n : ℕ), m^2 = n^2 + 2014 :=
by
  sorry

end no_nat_m_n_square_diff_2014_l169_169332


namespace min_power_for_84_to_divide_336_l169_169031

theorem min_power_for_84_to_divide_336 : 
  ∃ n : ℕ, (∀ m : ℕ, 84^m % 336 = 0 → m ≥ n) ∧ n = 2 := 
sorry

end min_power_for_84_to_divide_336_l169_169031


namespace large_ball_radius_final_radius_l169_169252

-- Define the radius of small balls
def small_ball_radius : ℝ := 0.5

-- Number of small balls
def small_ball_count : ℕ := 12

-- Volume of one small ball using the formula V = (4 / 3) * pi * r^3
def small_ball_volume : ℝ := (4 / 3) * Real.pi * (small_ball_radius ^ 3)

-- Total volume of all small balls
def total_volume : ℝ := small_ball_count * small_ball_volume

-- The formula for the volume of the large ball with radius R
def large_ball_volume (R : ℝ) : ℝ := (4 / 3) * Real.pi * (R ^ 3)

-- The radius of the larger ball is R such that the total volume is equal to the volume of the large ball
theorem large_ball_radius :
  ∃ R : ℝ, large_ball_volume R = total_volume ∧ R = (3 / 2) ^ (1 / 3) := 
by
  existsi (3 / 2) ^ (1 / 3)
  split
  · -- Proof that the volume formula holds
    unfold large_ball_volume total_volume small_ball_volume
    rw [mul_comm (_ : ℝ)]
    norm_num
    ring -- In general you should show the equality holds but we use ring due to large calculation.
  · -- Proof that R matches the desired radius
    rfl -- Since we chose exactly (3 / 2)^(1/3)

-- The desired theorem statement without proof
theorem final_radius : 
  ∃ R : ℝ, (4 / 3) * Real.pi * (R ^ 3) = 12 * (4 / 3) * Real.pi * (0.5 ^ 3) 
    ∧ R = (3 / 2) ^ (1 / 3) := sorry

end large_ball_radius_final_radius_l169_169252


namespace distinct_collections_proof_l169_169574

noncomputable def distinct_collections_count : ℕ := 240

theorem distinct_collections_proof : distinct_collections_count = 240 := by
  sorry

end distinct_collections_proof_l169_169574


namespace sequence_terms_l169_169941

theorem sequence_terms (S : ℕ → ℤ) (a : ℕ → ℤ) 
  (hS : ∀ n, S n = 3 ^ n - 2) :
  (a 1 = 1) ∧ (∀ n, n ≥ 2 → a n = 2 * 3 ^ (n - 1)) := by
  sorry

end sequence_terms_l169_169941


namespace robe_initial_savings_l169_169985

noncomputable def initial_savings (repair_fee corner_light_cost brake_disk_cost tires_cost remaining_savings : ℕ) : ℕ :=
  remaining_savings + repair_fee + corner_light_cost + 2 * brake_disk_cost + tires_cost

theorem robe_initial_savings :
  let R := 10
  let corner_light := 2 * R
  let brake_disk := 3 * corner_light
  let tires := corner_light + 2 * brake_disk
  let remaining := 480
  initial_savings R corner_light brake_disk tires remaining = 770 :=
by
  sorry

end robe_initial_savings_l169_169985


namespace avg_price_six_toys_l169_169639

def avg_price_five_toys : ℝ := 10
def price_sixth_toy : ℝ := 16
def total_toys : ℕ := 5 + 1

theorem avg_price_six_toys (avg_price_five_toys price_sixth_toy : ℝ) (total_toys : ℕ) :
  (avg_price_five_toys * 5 + price_sixth_toy) / total_toys = 11 := by
  sorry

end avg_price_six_toys_l169_169639


namespace james_money_left_l169_169689

-- Define the initial conditions
def ticket1_cost : ℕ := 150
def ticket2_cost : ℕ := 150
def ticket3_cost : ℕ := ticket1_cost / 3
def total_money : ℕ := 500
def roommate_share : ℕ := 2

-- Define and prove the theorem
theorem james_money_left : 
  let total_ticket_cost := ticket1_cost + ticket2_cost + ticket3_cost in
  let james_cost := total_ticket_cost / roommate_share in
  total_money - james_cost = 325 :=
by 
  let total_ticket_cost := ticket1_cost + ticket2_cost + ticket3_cost
  let james_cost := total_ticket_cost / roommate_share
  exact eq.refl 325

end james_money_left_l169_169689


namespace max_value_expression_l169_169005

theorem max_value_expression (x y : ℝ) : 
  (2 * x + 3 * y + 4) / Real.sqrt (x^2 + y^2 + 2) ≤ Real.sqrt 29 :=
by
  exact sorry

end max_value_expression_l169_169005


namespace jesse_remaining_pages_l169_169065

theorem jesse_remaining_pages (pages_read : ℕ)
  (h1 : pages_read = 83)
  (h2 : pages_read = (1 / 3 : ℝ) * total_pages)
  : pages_remaining = 166 :=
  by 
    -- Here we would build the proof, skipped with sorry
    sorry

end jesse_remaining_pages_l169_169065


namespace option_D_not_necessarily_true_l169_169212

variable {a b c : ℝ}

theorem option_D_not_necessarily_true 
  (h1 : c < b)
  (h2 : b < a)
  (h3 : a * c < 0) : ¬((c * b^2 < a * b^2) ↔ (b ≠ 0 ∨ b = 0 ∧ (c * b^2 < a * b^2))) := 
sorry

end option_D_not_necessarily_true_l169_169212


namespace sum_of_squares_of_medians_triangle_13_14_15_l169_169468

noncomputable def sum_of_squares_of_medians (a b c : ℝ) : ℝ :=
  (3 / 4) * (a^2 + b^2 + c^2)

theorem sum_of_squares_of_medians_triangle_13_14_15 :
  sum_of_squares_of_medians 13 14 15 = 442.5 :=
by
  -- By calculation using the definition of sum_of_squares_of_medians
  -- and substituting the given side lengths.
  -- Detailed proof steps are omitted
  sorry

end sum_of_squares_of_medians_triangle_13_14_15_l169_169468


namespace store_credit_percentage_l169_169456

theorem store_credit_percentage (SN NES cash_given change_back game_value : ℕ) (P : ℚ)
  (hSN : SN = 150)
  (hNES : NES = 160)
  (hcash_given : cash_given = 80)
  (hchange_back : change_back = 10)
  (hgame_value : game_value = 30)
  (hP_def : NES = P * SN + (cash_given - change_back) + game_value) :
  P = 0.4 :=
  sorry

end store_credit_percentage_l169_169456


namespace quadratic_real_roots_l169_169784

-- Define the quadratic equation
def quadratic_eq (a x : ℝ) : ℝ :=
  (a - 1) * x^2 - 2 * x + 1

-- Define the discriminant of the quadratic equation
def discriminant (a : ℝ) : ℝ :=
  4 - 4 * (a - 1)

-- The main theorem stating the needed proof problem
theorem quadratic_real_roots (a : ℝ) : (∃ x : ℝ, quadratic_eq a x = 0) ↔ a ≤ 2 := by
  -- Proof will be inserted here
  sorry

end quadratic_real_roots_l169_169784


namespace like_terms_exponents_l169_169680

theorem like_terms_exponents (m n : ℤ) 
  (h1 : 3 = m - 2) 
  (h2 : n + 1 = 2) : m - n = 4 := 
by
  sorry

end like_terms_exponents_l169_169680


namespace part_a_cube_edge_length_part_b_cube_edge_length_l169_169161

-- Part (a)
theorem part_a_cube_edge_length (small_cubes : ℕ) (edge_length_original : ℤ) :
  small_cubes = 512 → edge_length_original^3 = small_cubes → edge_length_original = 8 :=
by
  intros h1 h2
  sorry

-- Part (b)
theorem part_b_cube_edge_length (small_cubes_internal : ℕ) (edge_length_inner : ℤ) (edge_length_original : ℤ) :
  small_cubes_internal = 512 →
  edge_length_inner^3 = small_cubes_internal → 
  edge_length_original = edge_length_inner + 2 →
  edge_length_original = 10 :=
by
  intros h1 h2 h3
  sorry

end part_a_cube_edge_length_part_b_cube_edge_length_l169_169161


namespace sqrt_equiv_c_d_l169_169495

noncomputable def c : ℤ := 3
noncomputable def d : ℤ := 375

theorem sqrt_equiv_c_d : ∀ (x y : ℤ), x = 3^5 ∧ y = 5^3 → (∃ c d : ℤ, (c = 3 ∧ d = 375 ∧ x * y = c^4 * d))
    ∧ c + d = 378 := by sorry

end sqrt_equiv_c_d_l169_169495


namespace set_union_example_l169_169935

open Set

/-- Given sets A = {1, 2, 3} and B = {-1, 1}, prove that A ∪ B = {-1, 1, 2, 3} -/
theorem set_union_example : 
  let A := ({1, 2, 3} : Set ℤ)
  let B := ({-1, 1} : Set ℤ)
  A ∪ B = ({-1, 1, 2, 3} : Set ℤ) :=
by
  let A := ({1, 2, 3} : Set ℤ)
  let B := ({-1, 1} : Set ℤ)
  show A ∪ B = ({-1, 1, 2, 3} : Set ℤ)
  -- Proof to be provided here
  sorry

end set_union_example_l169_169935


namespace parabola_vertex_l169_169449

theorem parabola_vertex (y x : ℝ) (h : y = x^2 - 6 * x + 1) : 
  ∃ v_x v_y, (v_x, v_y) = (3, -8) :=
by 
  sorry

end parabola_vertex_l169_169449


namespace choose_4_from_15_l169_169541

theorem choose_4_from_15 : Nat.choose 15 4 = 1365 := by
  sorry

end choose_4_from_15_l169_169541


namespace valid_tree_arrangements_l169_169260

-- Define the types of trees
inductive TreeType
| Birch
| Oak

-- Define the condition that each tree must be adjacent to a tree of the other type
def isValidArrangement (trees : List TreeType) : Prop :=
  ∀ (i : ℕ), i < trees.length - 1 → trees.nthLe i sorry ≠ trees.nthLe (i + 1) sorry

-- Define the main problem
theorem valid_tree_arrangements : ∃ (ways : Nat), ways = 16 ∧
  ∃ (arrangements : List (List TreeType)), arrangements.length = ways ∧
    ∀ arrangement ∈ arrangements, arrangement.length = 7 ∧ isValidArrangement arrangement :=
sorry

end valid_tree_arrangements_l169_169260


namespace Sams_age_is_10_l169_169568

theorem Sams_age_is_10 (S M : ℕ) (h1 : M = S + 7) (h2 : S + M = 27) : S = 10 := 
by
  sorry

end Sams_age_is_10_l169_169568


namespace find_min_max_A_l169_169144

-- Define a 9-digit number B
def is_9_digit (B : ℕ) : Prop := B ≥ 100000000 ∧ B < 1000000000

-- Define a function that checks if a number is coprime with 24
def coprime_with_24 (B : ℕ) : Prop := Nat.gcd B 24 = 1

-- Define the transformation from B to A
def transform (B : ℕ) : ℕ := let b := B % 10 in b * 100000000 + (B / 10)

-- Lean 4 statement for the problem
theorem find_min_max_A :
  (∃ (B : ℕ), is_9_digit B ∧ coprime_with_24 B ∧ B > 666666666 ∧ transform B = 999999998) ∧
  (∃ (B : ℕ), is_9_digit B ∧ coprime_with_24 B ∧ B > 666666666 ∧ transform B = 166666667) :=
  by
    sorry -- Proof is omitted

end find_min_max_A_l169_169144


namespace negation_of_proposition_l169_169655

noncomputable def original_proposition :=
  ∀ a b : ℝ, (a * b = 0) → (a = 0)

theorem negation_of_proposition :
  ¬ original_proposition ↔ ∃ a b : ℝ, (a * b = 0) ∧ (a ≠ 0) :=
by
  sorry

end negation_of_proposition_l169_169655


namespace bernardo_larger_probability_l169_169760

-- Mathematical definitions
def bernardo_set : Finset ℕ := {1,2,3,4,5,6,7,8,10}
def silvia_set : Finset ℕ := {1,2,3,4,5,6}

-- Probability calculation function (you need to define the detailed implementation)
noncomputable def probability_bernardo_gt_silvia : ℚ := sorry

-- The proof statement
theorem bernardo_larger_probability : 
  probability_bernardo_gt_silvia = 13 / 20 :=
sorry

end bernardo_larger_probability_l169_169760


namespace tank_fraction_full_l169_169207

theorem tank_fraction_full 
  (initial_fraction : ℚ)
  (full_capacity : ℚ)
  (added_water : ℚ)
  (initial_fraction_eq : initial_fraction = 3/4)
  (full_capacity_eq : full_capacity = 40)
  (added_water_eq : added_water = 5) :
  ((initial_fraction * full_capacity + added_water) / full_capacity) = 7/8 :=
by 
  sorry

end tank_fraction_full_l169_169207


namespace carter_siblings_oldest_age_l169_169990

theorem carter_siblings_oldest_age
    (avg_age : ℕ)
    (sibling1 : ℕ)
    (sibling2 : ℕ)
    (sibling3 : ℕ)
    (sibling4 : ℕ) :
    avg_age = 9 →
    sibling1 = 5 →
    sibling2 = 8 →
    sibling3 = 7 →
    ((sibling1 + sibling2 + sibling3 + sibling4) / 4) = avg_age →
    sibling4 = 16 := by
  intros
  sorry

end carter_siblings_oldest_age_l169_169990


namespace olympiad_not_possible_l169_169057

theorem olympiad_not_possible (x : ℕ) (y : ℕ) (h1 : x + y = 1000) (h2 : y = x + 43) : false := by
  sorry

end olympiad_not_possible_l169_169057


namespace number_of_red_items_l169_169723

-- Define the mathematics problem
theorem number_of_red_items (R : ℕ) : 
  (23 + 1) + (11 + 1) + R = 66 → 
  R = 30 := 
by 
  intro h
  sorry

end number_of_red_items_l169_169723


namespace inverse_proportionality_l169_169843

theorem inverse_proportionality (a b c k a1 a2 b1 b2 c1 c2 : ℝ)
    (h1 : a * b * c = k)
    (h2 : a1 / a2 = 3 / 4)
    (h3 : b1 = 2 * b2)
    (h4 : c1 ≠ 0 ∧ c2 ≠ 0) :
    c1 / c2 = 2 / 3 :=
sorry

end inverse_proportionality_l169_169843


namespace coefficient_of_linear_term_l169_169098

def polynomial (x : ℝ) := x^2 - 2 * x - 3

theorem coefficient_of_linear_term : (∀ x : ℝ, polynomial x = x^2 - 2 * x - 3) → -2 = -2 := by
  intro h
  sorry

end coefficient_of_linear_term_l169_169098


namespace minimum_photos_l169_169047

-- Given conditions:
-- 1. There are exactly three monuments.
-- 2. A group of 42 tourists visited the city.
-- 3. Each tourist took at most one photo of each of the three monuments.
-- 4. Any two tourists together had photos of all three monuments.

theorem minimum_photos (n m : ℕ) (h₁ : n = 42) (h₂ : m = 3) 
  (photographs : ℕ → ℕ → Bool) -- whether tourist i took a photo of monument j
  (h₃ : ∀ i : ℕ, i < n → ∀ j : ℕ, j < m → photographs i j = tt ∨ photographs i j = ff) -- each tourist took at most one photo of each monument
  (h₄ : ∀ i₁ i₂ : ℕ, i₁ < n → i₂ < n → i₁ ≠ i₂ → ∀ j : ℕ, j < m → photographs i₁ j = tt ∨ photographs i₂ j = tt) -- any two tourists together had photos of all three monuments
  : ∃ min_photos : ℕ, min_photos = 123 :=
by
  sorry

end minimum_photos_l169_169047


namespace find_y_l169_169213

theorem find_y (x y : ℝ) (h1 : x = 2) (h2 : x^(3 * y) = 8) : y = 1 := by
  sorry

end find_y_l169_169213


namespace range_of_m_l169_169789

open Real

theorem range_of_m 
    (a : ℝ) (ha_pos : a > 0) (ha_ne_one : a ≠ 1) 
    (m : ℝ)
    (h : m * (a + 1/a) / sqrt 2 > 1) : 
    m ≥ sqrt 2 / 2 :=
sorry

end range_of_m_l169_169789


namespace no_nat_solutions_m2_eq_n2_plus_2014_l169_169312

theorem no_nat_solutions_m2_eq_n2_plus_2014 :
  ¬ ∃ (m n : ℕ), m^2 = n^2 + 2014 :=
by
  sorry

end no_nat_solutions_m2_eq_n2_plus_2014_l169_169312


namespace volume_rotation_l169_169273

theorem volume_rotation
  (f : ℝ → ℝ)
  (g : ℝ → ℝ)
  (a b : ℝ)
  (h₁ : ∀ (x : ℝ), f x = x^3)
  (h₂ : ∀ (x : ℝ), g x = x^(1/2))
  (h₃ : a = 0)
  (h₄ : b = 1):
  ∫ x in a..b, π * ((g x)^2 - (f x)^2) = 5 * π / 14 :=
by
  sorry

end volume_rotation_l169_169273


namespace floor_sqrt_120_l169_169906

theorem floor_sqrt_120 : (Int.floor (Real.sqrt 120)) = 10 := by
  have h1 : 10^2 < 120 := by norm_num
  have h2 : 120 < 11^2 := by norm_num
  -- Additional steps to show that Int.floor (Real.sqrt 120) = 10
  sorry

end floor_sqrt_120_l169_169906


namespace rectangle_area_l169_169438

theorem rectangle_area (side_of_square := 45)
  (radius_of_circle := side_of_square)
  (length_of_rectangle := (2/5 : ℚ) * radius_of_circle)
  (breadth_of_rectangle := 10) :
  breadth_of_rectangle * length_of_rectangle = 180 := 
by
  sorry

end rectangle_area_l169_169438


namespace largest_three_digit_multiple_of_six_with_sum_fifteen_l169_169264

theorem largest_three_digit_multiple_of_six_with_sum_fifteen : 
  ∃ (n : ℕ), (100 ≤ n ∧ n < 1000) ∧ (n % 6 = 0) ∧ (Nat.digits 10 n).sum = 15 ∧ 
  ∀ (m : ℕ), (100 ≤ m ∧ m < 1000) ∧ (m % 6 = 0) ∧ (Nat.digits 10 m).sum = 15 → m ≤ n :=
  sorry

end largest_three_digit_multiple_of_six_with_sum_fifteen_l169_169264


namespace circumscribed_quadrilateral_l169_169819

open EuclideanGeometry

theorem circumscribed_quadrilateral (ABCD : Type*)
  [h : has_incircle ABCD]
  (K L M N : Point)
  (K1 L1 M1 N1 : Point)
  (hK : ext_angle_bisector_eq K DAB ABC)
  (hL : ext_angle_bisector_eq L ABC BCD)
  (hM : ext_angle_bisector_eq M BCD CDA)
  (hN : ext_angle_bisector_eq N CDA DAB)
  (hK1 : orthocenter_eq K1 ABK)
  (hL1 : orthocenter_eq L1 BCL)
  (hM1 : orthocenter_eq M1 CDM)
  (hN1 : orthocenter_eq N1 DAN) :
  is_parallelogram K1 L1 M1 N1 :=
sorry

end circumscribed_quadrilateral_l169_169819


namespace root_power_division_l169_169586

noncomputable def root4 (a : ℝ) : ℝ := a^(1/4)
noncomputable def root6 (a : ℝ) : ℝ := a^(1/6)

theorem root_power_division : 
  (root4 7) / (root6 7) = 7^(1/12) :=
by sorry

end root_power_division_l169_169586


namespace sufficient_condition_above_2c_l169_169754

theorem sufficient_condition_above_2c (a b c : ℝ) (h1 : a > c) (h2 : b > c) : a + b > 2 * c :=
by
  sorry

end sufficient_condition_above_2c_l169_169754


namespace value_of_a_m_minus_3n_l169_169953

theorem value_of_a_m_minus_3n (a : ℝ) (m n : ℝ) (h1 : a^m = 8) (h2 : a^n = 2) : a^(m - 3 * n) = 1 :=
sorry

end value_of_a_m_minus_3n_l169_169953


namespace john_paid_after_tax_l169_169555

-- Definitions based on problem conditions
def original_cost : ℝ := 200
def tax_rate : ℝ := 0.15

-- Definition of the tax amount
def tax_amount : ℝ := tax_rate * original_cost

-- Definition of the total amount paid
def total_amount_paid : ℝ := original_cost + tax_amount

-- Theorem statement for the proof
theorem john_paid_after_tax : total_amount_paid = 230 := by
  sorry

end john_paid_after_tax_l169_169555


namespace at_least_two_consecutive_heads_probability_l169_169747

theorem at_least_two_consecutive_heads_probability :
  let outcomes := ["HHH", "HHT", "HTH", "HTT", "THH", "THT", "TTH", "TTT"]
  let favorable_outcomes := ["HHH", "HHT", "THH"]
  let total_outcomes := outcomes.length
  let num_favorable := favorable_outcomes.length
  (num_favorable / total_outcomes : ℚ) = 1 / 2 :=
by sorry

end at_least_two_consecutive_heads_probability_l169_169747


namespace opposite_face_of_x_l169_169499

theorem opposite_face_of_x 
    (A D F B E x : Prop) 
    (h1 : x → (A ∧ D ∧ F))
    (h2 : x → B)
    (h3 : E → D ∧ ¬x) : B := 
sorry

end opposite_face_of_x_l169_169499


namespace binomial_coeff_arithmetic_seq_l169_169802

theorem binomial_coeff_arithmetic_seq (n : ℕ) (x : ℝ) (h : ∀ (a b c : ℝ), a = 1 ∧ b = n/2 ∧ c = n*(n-1)/8 → (b - a) = (c - b)) : n = 8 :=
sorry

end binomial_coeff_arithmetic_seq_l169_169802


namespace part_a_part_b_part_c_l169_169475

-- Definitions for the problem
def hard_problem_ratio_a := 2 / 3
def unsolved_problem_ratio_a := 2 / 3
def well_performing_students_ratio_a := 2 / 3

def hard_problem_ratio_b := 3 / 4
def unsolved_problem_ratio_b := 3 / 4
def well_performing_students_ratio_b := 3 / 4

def hard_problem_ratio_c := 7 / 10
def unsolved_problem_ratio_c := 7 / 10
def well_performing_students_ratio_c := 7 / 10

-- Theorems to prove
theorem part_a : 
  ∃ (hard_problem_ratio_a unsolved_problem_ratio_a well_performing_students_ratio_a : ℚ),
  hard_problem_ratio_a == 2 / 3 ∧
  unsolved_problem_ratio_a == 2 / 3 ∧
  well_performing_students_ratio_a == 2 / 3 →
  (True) := sorry

theorem part_b : 
  ∀ (hard_problem_ratio_b : ℚ),
  hard_problem_ratio_b == 3 / 4 →
  (False) := sorry

theorem part_c : 
  ∀ (hard_problem_ratio_c : ℚ),
  hard_problem_ratio_c == 7 / 10 →
  (False) := sorry

end part_a_part_b_part_c_l169_169475


namespace firefighters_time_to_extinguish_fire_l169_169139

theorem firefighters_time_to_extinguish_fire (gallons_per_minute_per_hose : ℕ) (total_gallons : ℕ) (number_of_firefighters : ℕ)
  (H1 : gallons_per_minute_per_hose = 20)
  (H2 : total_gallons = 4000)
  (H3 : number_of_firefighters = 5): 
  (total_gallons / (gallons_per_minute_per_hose * number_of_firefighters)) = 40 := 
by 
  sorry

end firefighters_time_to_extinguish_fire_l169_169139


namespace units_digit_17_pow_2007_l169_169734

theorem units_digit_17_pow_2007 :
  (17 ^ 2007) % 10 = 3 := 
sorry

end units_digit_17_pow_2007_l169_169734


namespace average_death_rate_l169_169805

def birth_rate := 4 -- people every 2 seconds
def net_increase_per_day := 43200 -- people

def seconds_per_day := 86400 -- 24 * 60 * 60

def net_increase_per_second := net_increase_per_day / seconds_per_day -- people per second

def death_rate := (birth_rate / 2) - net_increase_per_second -- people per second

theorem average_death_rate :
  death_rate * 2 = 3 := by
  -- proof is omitted
  sorry

end average_death_rate_l169_169805


namespace zachary_pushups_l169_169742

variable {P : ℕ}
variable {C : ℕ}

theorem zachary_pushups :
  C = 58 → C = P + 12 → P = 46 :=
by 
  intros hC1 hC2
  rw [hC2] at hC1
  linarith

end zachary_pushups_l169_169742


namespace concentric_circle_area_ratio_l169_169109

theorem concentric_circle_area_ratio (r R : ℝ) (h_ratio : (π * R^2) / (π * r^2) = 16 / 3) :
  R - r = 1.309 * r :=
by
  sorry

end concentric_circle_area_ratio_l169_169109


namespace no_nat_numbers_m_n_satisfy_eq_l169_169346

theorem no_nat_numbers_m_n_satisfy_eq (m n : ℕ) : ¬ (m^2 = n^2 + 2014) := sorry

end no_nat_numbers_m_n_satisfy_eq_l169_169346


namespace candy_bar_cost_l169_169759

variable (C : ℕ)

theorem candy_bar_cost
  (soft_drink_cost : ℕ)
  (num_candy_bars : ℕ)
  (total_spent : ℕ)
  (h1 : soft_drink_cost = 2)
  (h2 : num_candy_bars = 5)
  (h3 : total_spent = 27) :
  num_candy_bars * C + soft_drink_cost = total_spent → C = 5 := by
  sorry

end candy_bar_cost_l169_169759


namespace increase_in_average_age_l169_169096

variable (A : ℝ)
variable (A_increase : ℝ)
variable (orig_age_sum : ℝ)
variable (new_age_sum : ℝ)

def original_total_age (A : ℝ) := 8 * A
def new_total_age (A : ℝ) := original_total_age A - 20 - 22 + 29 + 29

theorem increase_in_average_age (A : ℝ) (orig_age_sum := original_total_age A) (new_age_sum := new_total_age A) : 
  (new_age_sum / 8) = (A + 2) := 
by
  unfold new_total_age
  unfold original_total_age
  sorry

end increase_in_average_age_l169_169096


namespace parcel_cost_l169_169583

theorem parcel_cost (P : ℤ) (hP : P ≥ 1) : 
  (P ≤ 5 → C = 15 + 4 * (P - 1)) ∧ (P > 5 → C = 15 + 4 * (P - 1) - 10) :=
sorry

end parcel_cost_l169_169583


namespace no_valid_n_exists_l169_169397

theorem no_valid_n_exists :
  ¬ ∃ n : ℕ, 219 ≤ n ∧ n ≤ 2019 ∧ ∃ x y : ℕ, 
    1 ≤ x ∧ x < n ∧ n < y ∧ (∀ k : ℕ, k ≤ n → k ≠ x ∧ k ≠ x+1 → y % k = 0) := 
by {
  sorry
}

end no_valid_n_exists_l169_169397


namespace marble_remainder_l169_169268

theorem marble_remainder
  (r p : ℕ)
  (h_r : r % 5 = 2)
  (h_p : p % 5 = 4) :
  (r + p) % 5 = 1 :=
by
  sorry

end marble_remainder_l169_169268


namespace algebraic_expression_value_l169_169290

theorem algebraic_expression_value (x : ℝ) (h : x^2 + x - 1 = 0) : x^3 + 2 * x^2 - 7 = -6 :=
by
  sorry

end algebraic_expression_value_l169_169290


namespace angle_C_eq_pi_over_3_l169_169682

theorem angle_C_eq_pi_over_3 (a b c A B C : ℝ)
  (h : (a + c) * (Real.sin A - Real.sin C) = b * (Real.sin A - Real.sin B)) :
  C = Real.pi / 3 :=
sorry

end angle_C_eq_pi_over_3_l169_169682


namespace gear_rotations_l169_169258

-- Definitions from the conditions
def gearA_teeth : ℕ := 12
def gearB_teeth : ℕ := 54

-- The main problem: prove that gear A needs 9 rotations and gear B needs 2 rotations
theorem gear_rotations :
  ∃ x y : ℕ, 12 * x = 54 * y ∧ x = 9 ∧ y = 2 := by
  sorry

end gear_rotations_l169_169258


namespace sufficient_not_necessary_perpendicular_l169_169099

theorem sufficient_not_necessary_perpendicular (a : ℝ) :
  (∀ x y : ℝ, (a + 2) * x + 3 * a * y + 1 = 0 ∧
              (a - 2) * x + (a + 2) * y - 3 = 0 → false) ↔ a = -2 :=
sorry

end sufficient_not_necessary_perpendicular_l169_169099


namespace detour_distance_l169_169703

-- Definitions based on conditions:
def D_black : ℕ := sorry -- The original distance along the black route
def D_black_C : ℕ := sorry -- The distance from C to B along the black route
def D_red : ℕ := sorry -- The distance from C to B along the red route

-- Extra distance due to detour calculation
def D_extra := D_red - D_black_C

-- Prove that the extra distance is 14 km
theorem detour_distance : D_extra = 14 := by
  sorry

end detour_distance_l169_169703


namespace previous_salary_l169_169833

theorem previous_salary (P : ℝ) (h : 1.05 * P = 2100) : P = 2000 :=
by
  sorry

end previous_salary_l169_169833


namespace total_dots_is_78_l169_169627

-- Define the conditions as Lean definitions
def ladybugs_monday : ℕ := 8
def ladybugs_tuesday : ℕ := 5
def dots_per_ladybug : ℕ := 6

-- Define the total number of ladybugs
def total_ladybugs : ℕ := ladybugs_monday + ladybugs_tuesday

-- Define the total number of dots
def total_dots : ℕ := total_ladybugs * dots_per_ladybug

-- Theorem stating the problem to solve
theorem total_dots_is_78 : total_dots = 78 := by
  sorry

end total_dots_is_78_l169_169627


namespace joe_lowest_dropped_score_l169_169068

theorem joe_lowest_dropped_score (A B C D : ℕ) 
  (hmean_before : (A + B + C + D) / 4 = 35)
  (hmean_after : (A + B + C) / 3 = 40)
  (hdrop : D = min A (min B (min C D))) :
  D = 20 :=
by sorry

end joe_lowest_dropped_score_l169_169068


namespace divides_mn_minus_one_l169_169421

theorem divides_mn_minus_one (m n p : ℕ) (hp : p.Prime) (h1 : m < n) (h2 : n < p) 
    (hm2 : p ∣ m^2 + 1) (hn2 : p ∣ n^2 + 1) : p ∣ m * n - 1 :=
by
  sorry

end divides_mn_minus_one_l169_169421


namespace n_ge_2_pow_k_add_1_minus_1_find_n_good_l169_169124

open Nat

-- Define n-goodness

def n_good (k n : ℕ) : Prop :=
  ∃ (t : Tournament n), ∃ (v : Fin n), ∀ (u : Fin n), v ≠ u → u ∉ t.losses v

-- Statement 1: For a tournament with n players, prove that n >= 2^(k+1) - 1 for some player to have lost all k's matches
theorem n_ge_2_pow_k_add_1_minus_1 (k : ℕ) : ∃ n, n ≥ 2^(k+1) - 1 :=
  sorry

-- Statement 2: Find all n such that 2 is n-good and prove n >= 7
theorem find_n_good (n : ℕ) : (n_good 2 n ↔ n ≥ 7) :=
  sorry

end n_ge_2_pow_k_add_1_minus_1_find_n_good_l169_169124


namespace inequality_solution_range_of_a_l169_169204

def f (x a : ℝ) : ℝ := |x - 3| - |x - a|

-- Statement 1
theorem inequality_solution (x : ℝ) : f x 2 ≤ -1 / 2 ↔ x ≥ 11 / 4 :=
by
  sorry

-- Statement 2
theorem range_of_a (a : ℝ) : (∃ x : ℝ, f x a ≥ a) ↔ a ≤ 3 / 2 :=
by
  sorry

end inequality_solution_range_of_a_l169_169204


namespace cover_black_squares_with_L_shape_l169_169921

-- Define a function to check if a number is odd
def is_odd (n : ℕ) : Prop := n % 2 = 1

-- Define the main theorem
theorem cover_black_squares_with_L_shape (n : ℕ) (h_odd : is_odd n) (h_corner_black : ∀i j, (i = 0 ∨ i = n - 1) ∧ (j = 0 ∨ j = n - 1) → (i + j) % 2 = 1) : n ≥ 7 :=
sorry

end cover_black_squares_with_L_shape_l169_169921


namespace calum_disco_ball_budget_l169_169292

-- Defining the conditions
def n_d : ℕ := 4  -- Number of disco balls
def n_f : ℕ := 10  -- Number of food boxes
def p_f : ℕ := 25  -- Price per food box in dollars
def B : ℕ := 330  -- Total budget in dollars

-- Defining the expected result
def p_d : ℕ := 20  -- Cost per disco ball in dollars

-- Proof statement (no proof, just the statement)
theorem calum_disco_ball_budget :
  (10 * p_f + 4 * p_d = B) → (p_d = 20) :=
by
  sorry

end calum_disco_ball_budget_l169_169292


namespace quadratic_inequality_solution_l169_169433

theorem quadratic_inequality_solution (x : ℝ) : 
  3 * x^2 - 8 * x - 3 > 0 ↔ (x < -1/3 ∨ x > 3) :=
by
  sorry

end quadratic_inequality_solution_l169_169433


namespace no_positive_a_inequality_holds_l169_169365

theorem no_positive_a_inequality_holds :
  ¬ ∃ (a : ℝ), (0 < a) ∧ (∀ (x : ℝ), |cos x| + |cos (a * x)| > sin x + sin (a * x)) :=
sorry

end no_positive_a_inequality_holds_l169_169365


namespace necessary_but_not_sufficient_l169_169234

def M : Set ℝ := {x | -2 < x ∧ x < 3}
def P : Set ℝ := {x | x ≤ -1}

theorem necessary_but_not_sufficient :
  (∀ x, x ∈ M ∩ P → x ∈ M ∪ P) ∧ (∃ x, x ∈ M ∪ P ∧ x ∉ M ∩ P) :=
by
  sorry

end necessary_but_not_sufficient_l169_169234


namespace decreases_as_x_increases_graph_passes_through_origin_l169_169931

-- Proof Problem 1: Show that y decreases as x increases if and only if k > 2
theorem decreases_as_x_increases (k : ℝ) : (∀ x1 x2 : ℝ, (x1 < x2) → ((2 - k) * x1 - k^2 + 4) > ((2 - k) * x2 - k^2 + 4)) ↔ (k > 2) := 
  sorry

-- Proof Problem 2: Show that the graph passes through the origin if and only if k = -2
theorem graph_passes_through_origin (k : ℝ) : ((2 - k) * 0 - k^2 + 4 = 0) ↔ (k = -2) :=
  sorry

end decreases_as_x_increases_graph_passes_through_origin_l169_169931


namespace timesToFillBottlePerWeek_l169_169569

noncomputable def waterConsumptionPerDay : ℕ := 4 * 5
noncomputable def waterConsumptionPerWeek : ℕ := 7 * waterConsumptionPerDay
noncomputable def bottleCapacity : ℕ := 35

theorem timesToFillBottlePerWeek : 
  waterConsumptionPerWeek / bottleCapacity = 4 := 
by
  sorry

end timesToFillBottlePerWeek_l169_169569


namespace value_of_m2_plus_3n2_l169_169405

noncomputable def real_numbers_with_condition (m n : ℝ) : Prop :=
  (m^2 + 3*n^2)^2 - 4*(m^2 + 3*n^2) - 12 = 0

theorem value_of_m2_plus_3n2 (m n : ℝ) (h : real_numbers_with_condition m n) : m^2 + 3*n^2 = 6 :=
by
  sorry

end value_of_m2_plus_3n2_l169_169405


namespace houses_with_white_mailboxes_l169_169142

theorem houses_with_white_mailboxes (total_mail : ℕ) (total_houses : ℕ) (red_mailboxes : ℕ) (mail_per_house : ℕ)
    (h1 : total_mail = 48) (h2 : total_houses = 8) (h3 : red_mailboxes = 3) (h4 : mail_per_house = 6) :
  total_houses - red_mailboxes = 5 :=
by
  sorry

end houses_with_white_mailboxes_l169_169142


namespace unique_positive_real_solution_of_polynomial_l169_169398

theorem unique_positive_real_solution_of_polynomial :
  ∃! x : ℝ, x > 0 ∧ (x^11 + 8 * x^10 + 15 * x^9 + 1000 * x^8 - 1200 * x^7 = 0) :=
by
  sorry

end unique_positive_real_solution_of_polynomial_l169_169398


namespace not_factorable_l169_169237

open Polynomial

def P (x y : ℝ) := x ^ 200 * y ^ 200 + 1

theorem not_factorable (f : Polynomial ℝ) (g : Polynomial ℝ) :
  ¬(P x y = f * g) := sorry

end not_factorable_l169_169237


namespace purely_imaginary_complex_number_l169_169533

theorem purely_imaginary_complex_number (a : ℝ) :
  (∃ b : ℝ, (a^2 - 3 * a + 2) = 0 ∧ a ≠ 1) → a = 2 :=
by
  sorry

end purely_imaginary_complex_number_l169_169533


namespace max_area_right_triangle_l169_169513

def right_triangle_max_area (l : ℝ) (p : ℝ) (h : ℝ) : ℝ :=
  l + p + h

noncomputable def maximal_area (x y : ℝ) : ℝ :=
  (1/2) * x * y

theorem max_area_right_triangle (x y : ℝ) (h : ℝ) (hp : h = Real.sqrt (x^2 + y^2)) (hp2: x + y + h = 60) :
  maximal_area 30 30 = 450 :=
by
  sorry

end max_area_right_triangle_l169_169513


namespace smallest_N_l169_169155

theorem smallest_N (l m n : ℕ) (N : ℕ) (h_block : N = l * m * n)
  (h_invisible : (l - 1) * (m - 1) * (n - 1) = 120) :
  N = 216 :=
sorry

end smallest_N_l169_169155


namespace inequality_am_gm_l169_169198

theorem inequality_am_gm 
  (a b c d : ℝ) 
  (h_nonneg_a : 0 ≤ a) 
  (h_nonneg_b : 0 ≤ b) 
  (h_nonneg_c : 0 ≤ c) 
  (h_nonneg_d : 0 ≤ d) 
  (h_sum : a * b + b * c + c * d + d * a = 1) : 
  (a^3 / (b + c + d) + b^3 / (c + d + a) + c^3 / (d + a + b) + d^3 / (a + b + c)) ≥ 1 / 3 :=
by
  sorry


end inequality_am_gm_l169_169198


namespace find_min_max_A_l169_169145

-- Define a 9-digit number B
def is_9_digit (B : ℕ) : Prop := B ≥ 100000000 ∧ B < 1000000000

-- Define a function that checks if a number is coprime with 24
def coprime_with_24 (B : ℕ) : Prop := Nat.gcd B 24 = 1

-- Define the transformation from B to A
def transform (B : ℕ) : ℕ := let b := B % 10 in b * 100000000 + (B / 10)

-- Lean 4 statement for the problem
theorem find_min_max_A :
  (∃ (B : ℕ), is_9_digit B ∧ coprime_with_24 B ∧ B > 666666666 ∧ transform B = 999999998) ∧
  (∃ (B : ℕ), is_9_digit B ∧ coprime_with_24 B ∧ B > 666666666 ∧ transform B = 166666667) :=
  by
    sorry -- Proof is omitted

end find_min_max_A_l169_169145


namespace van_helsing_removed_percentage_l169_169597

theorem van_helsing_removed_percentage :
  ∀ (V W : ℕ), 
  (5 * V / 2 + 10 * 8 = 105) →
  (W = 4 * V) →
  8 / W * 100 = 20 := 
by
  sorry

end van_helsing_removed_percentage_l169_169597


namespace sum_of_distinct_selections_is_34_l169_169753

-- Define a 4x4 grid filled sequentially from 1 to 16
def grid : List (List ℕ) := [
  [1, 2, 3, 4],
  [5, 6, 7, 8],
  [9, 10, 11, 12],
  [13, 14, 15, 16]
]

-- Define a type for selections from the grid ensuring distinct rows and columns.
structure Selection where
  row : ℕ
  col : ℕ
  h_row : row < 4
  h_col : col < 4

-- Define the sum of any selection of 4 numbers from distinct rows and columns in the grid.
def sum_of_selection (selections : List Selection) : ℕ :=
  if h : List.length selections = 4 then
    List.sum (List.map (λ sel => (grid.get! sel.row).get! sel.col) selections)
  else 0

-- The main theorem
theorem sum_of_distinct_selections_is_34 (selections : List Selection) 
  (h_distinct_rows : List.Nodup (List.map (λ sel => sel.row) selections))
  (h_distinct_cols : List.Nodup (List.map (λ sel => sel.col) selections)) :
  sum_of_selection selections = 34 :=
by
  -- Proof is omitted
  sorry

end sum_of_distinct_selections_is_34_l169_169753


namespace max_expression_value_l169_169441

theorem max_expression_value (a b c d : ℝ) 
  (h1 : -6.5 ≤ a ∧ a ≤ 6.5) 
  (h2 : -6.5 ≤ b ∧ b ≤ 6.5) 
  (h3 : -6.5 ≤ c ∧ c ≤ 6.5) 
  (h4 : -6.5 ≤ d ∧ d ≤ 6.5) : 
  a + 2*b + c + 2*d - a*b - b*c - c*d - d*a ≤ 182 :=
sorry

end max_expression_value_l169_169441


namespace symmetric_point_coordinates_l169_169582

theorem symmetric_point_coordinates (M N : ℝ × ℝ) (x y : ℝ) 
  (hM : M = (-2, 1)) 
  (hN_symmetry : N = (M.1, -M.2)) : N = (-2, -1) :=
by
  sorry

end symmetric_point_coordinates_l169_169582


namespace exactly_one_divisible_by_5_l169_169012

def a (n : ℕ) : ℕ := 2^(2*n + 1) - 2^(n + 1) + 1
def b (n : ℕ) : ℕ := 2^(2*n + 1) + 2^(n + 1) + 1

theorem exactly_one_divisible_by_5 (n : ℕ) (hn : 0 < n) : (a n % 5 = 0 ∧ b n % 5 ≠ 0) ∨ (a n % 5 ≠ 0 ∧ b n % 5 = 0) :=
  sorry

end exactly_one_divisible_by_5_l169_169012


namespace chord_central_angle_l169_169482

-- Given that a chord divides the circumference of a circle in the ratio 5:7
-- Prove that the central angle opposite this chord can be either 75° or 105°
theorem chord_central_angle (x : ℝ) (h : 5 * x + 7 * x = 180) :
  5 * x = 75 ∨ 7 * x = 105 :=
sorry

end chord_central_angle_l169_169482


namespace fraction_add_eq_l169_169673

theorem fraction_add_eq (x y : ℝ) (hx : y / x = 3 / 7) : (x + y) / x = 10 / 7 :=
by
  sorry

end fraction_add_eq_l169_169673


namespace geometric_sequence_a4_value_l169_169531

theorem geometric_sequence_a4_value 
  (a : ℕ → ℝ) 
  (q : ℝ) 
  (h_pos : ∀ n, 0 < a n) 
  (h_geom : ∀ n, a (n + 1) = a n * q) 
  (h1 : a 1 + (2 / 3) * a 2 = 3) 
  (h2 : a 4^2 = (1 / 9) * a 3 * a 7) 
  :
  a 4 = 27 :=
sorry

end geometric_sequence_a4_value_l169_169531


namespace curves_intersect_at_three_points_l169_169606

theorem curves_intersect_at_three_points (b : ℝ) :
  (∃ x y : ℝ, x^2 + y^2 = b^2 ∧ y = 2 * x^2 - b) ∧ 
  (∀ x₁ y₁ x₂ y₂ x₃ y₃ : ℝ,
    (x₁ ≠ x₂ ∨ y₁ ≠ y₂) ∧ (x₁ ≠ x₃ ∨ y₁ ≠ y₃) ∧ (x₂ ≠ x₃ ∨ y₂ ≠ y₃) ∧
    (x₁^2 + y₁^2 = b^2) ∧ (x₂^2 + y₂^2 = b^2) ∧ (x₃^2 + y₃^2 = b^2) ∧
    (y₁ = 2 * x₁^2 - b) ∧ (y₂ = 2 * x₂^2 - b) ∧ (y₃ = 2 * x₃^2 - b)) ↔ b > 1 / 4 :=
by
  sorry

end curves_intersect_at_three_points_l169_169606


namespace trig_identity_l169_169069

theorem trig_identity (θ : ℝ) (h : cos (2 * θ) = 1 / 5) : sin θ ^ 6 + cos θ ^ 6 = 7 / 25 :=
by
  sorry

end trig_identity_l169_169069


namespace Jhon_payment_per_day_l169_169416

theorem Jhon_payment_per_day
  (total_days : ℕ)
  (present_days : ℕ)
  (absent_pay : ℝ)
  (total_pay : ℝ)
  (Jhon_present_days : total_days = 60)
  (Jhon_presence : present_days = 35)
  (Jhon_absent_payment : absent_pay = 3.0)
  (Jhon_total_payment : total_pay = 170) :
  ∃ (P : ℝ), 
    P = 2.71 ∧ 
    total_pay = (present_days * P + (total_days - present_days) * absent_pay) := 
sorry

end Jhon_payment_per_day_l169_169416


namespace find_radius_l169_169847

theorem find_radius (AB EO : ℝ) (AE BE : ℝ) (h1 : AB = AE + BE) (h2 : AE = 2 * BE) (h3 : EO = 7) :
  ∃ R : ℝ, R = 11 := by
  sorry

end find_radius_l169_169847


namespace value_of_a_m_minus_3n_l169_169954

theorem value_of_a_m_minus_3n (a : ℝ) (m n : ℝ) (h1 : a^m = 8) (h2 : a^n = 2) : a^(m - 3 * n) = 1 :=
sorry

end value_of_a_m_minus_3n_l169_169954


namespace max_area_of_right_angled_isosceles_triangle_l169_169444

theorem max_area_of_right_angled_isosceles_triangle (a b : ℝ) (h₁ : a = 12) (h₂ : b = 15) :
  ∃ A : ℝ, A = 72 ∧ 
  (∀ (x : ℝ), x ≤ min a b → (1 / 2) * x^2 ≤ A) :=
by
  use 72
  sorry

end max_area_of_right_angled_isosceles_triangle_l169_169444


namespace numbers_left_on_blackboard_l169_169851

theorem numbers_left_on_blackboard (n11 n12 n13 n14 n15 : ℕ)
    (h_n11 : n11 = 11) (h_n12 : n12 = 12) (h_n13 : n13 = 13) (h_n14 : n14 = 14) (h_n15 : n15 = 15)
    (total_numbers : n11 + n12 + n13 + n14 + n15 = 65) :
  ∃ (remaining1 remaining2 : ℕ), remaining1 = 12 ∧ remaining2 = 14 := 
sorry

end numbers_left_on_blackboard_l169_169851


namespace range_a_of_tangents_coincide_l169_169943

theorem range_a_of_tangents_coincide (x1 x2 : ℝ) (h1 : x1 < 0) (h2 : 0 < x2) (a : ℝ)
  (h3 : -1 / (x2 ^ 2) = 2 * x1 + 1) (h4 : x1 ^ 2 = -a) :
  1/4 < a ∧ a < 1 :=
by
  sorry 

end range_a_of_tangents_coincide_l169_169943


namespace no_solution_exists_l169_169306

theorem no_solution_exists (m n : ℕ) : ¬ (m^2 = n^2 + 2014) :=
by
  sorry

end no_solution_exists_l169_169306


namespace altitude_length_l169_169218

theorem altitude_length 
    {A B C : Type*} [MetricSpace A] [MetricSpace B] [MetricSpace C] 
    (AB BC AC : ℝ) (hAC : 𝕜) 
    (h₀ : AB = 8)
    (h₁ : BC = 7)
    (h₂ : AC = 5) :
  h = (5 * Real.sqrt 3) / 2 :=
sorry

end altitude_length_l169_169218


namespace kho_kho_only_l169_169744

variable (K H B : ℕ)

theorem kho_kho_only :
  (K + B = 10) ∧ (H + 5 = H + B) ∧ (B = 5) ∧ (K + H + B = 45) → H = 35 :=
by
  intros h
  sorry

end kho_kho_only_l169_169744


namespace negation_proposition_l169_169716

theorem negation_proposition : 
  (¬ ∃ x_0 : ℝ, 2 * x_0 - 3 > 1) ↔ (∀ x : ℝ, 2 * x - 3 ≤ 1) :=
by
  sorry

end negation_proposition_l169_169716


namespace arithmetic_sequence_sum_l169_169267

theorem arithmetic_sequence_sum :
  let a₁ := 3
  let d := 4
  let n := 30
  let aₙ := a₁ + (n - 1) * d
  let Sₙ := (n / 2) * (a₁ + aₙ)
  Sₙ = 1830 :=
by
  intros
  let a₁ := 3
  let d := 4
  let n := 30
  let aₙ := a₁ + (n - 1) * d
  let Sₙ := (n / 2) * (a₁ + aₙ)
  sorry

end arithmetic_sequence_sum_l169_169267


namespace solve_for_a_and_b_range_of_f_when_x_lt_zero_l169_169526

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := (1 + a * (2 ^ x)) / (2 ^ x + b)

theorem solve_for_a_and_b (a b : ℝ) :
  f a b 1 = 3 ∧
  f a b (-1) = -3 →
  a = 1 ∧ b = -1 :=
by
  sorry

theorem range_of_f_when_x_lt_zero (x : ℝ) :
  ∀ x < 0, f 1 (-1) x < -1 :=
by 
  sorry

end solve_for_a_and_b_range_of_f_when_x_lt_zero_l169_169526


namespace sum_lucky_numbers_divisible_by_2002_l169_169134

-- Define a structure for six-digit numbers
structure LuckyNumber where
  a b c d e f : ℕ
  h₁ : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ d ≠ e ∧ d ≠ f ∧ e ≠ f
  h₂ : a + b + c = d + e + f

theorem sum_lucky_numbers_divisible_by_2002 : 
  let S := Finset.univ.filter (λ N : LuckyNumber, True)
  (S.sum (λ N, 100000 * N.a + 10000 * N.b + 1000 * N.c + 100 * N.d + 10 * N.e + N.f)) % 2002 = 0 :=
sorry

end sum_lucky_numbers_divisible_by_2002_l169_169134


namespace min_photographs_42_tourists_3_monuments_l169_169044

noncomputable def min_photos_taken (num_tourists : ℕ) (num_monuments : ℕ) : ℕ :=
  if num_tourists = 42 ∧ num_monuments = 3 then 123 else 0

-- Main statement:
theorem min_photographs_42_tourists_3_monuments : 
  (∀ (num_tourists num_monuments : ℕ), 
    num_tourists = 42 ∧ num_monuments = 3 → min_photos_taken num_tourists num_monuments = 123)
  := by
    sorry

end min_photographs_42_tourists_3_monuments_l169_169044


namespace product_has_correct_sign_and_units_digit_l169_169640

noncomputable def product_negative_integers_divisible_by_3_less_than_198 : ℤ :=
  sorry

theorem product_has_correct_sign_and_units_digit :
  product_negative_integers_divisible_by_3_less_than_198 < 0 ∧
  product_negative_integers_divisible_by_3_less_than_198 % 10 = 6 :=
by
  sorry

end product_has_correct_sign_and_units_digit_l169_169640


namespace garden_perimeter_is_48_l169_169487

def square_garden_perimeter (pond_area garden_remaining_area : ℕ) : ℕ :=
  let garden_area := pond_area + garden_remaining_area
  let side_length := Int.natAbs (Int.sqrt garden_area)
  4 * side_length

theorem garden_perimeter_is_48 :
  square_garden_perimeter 20 124 = 48 :=
  by
  sorry

end garden_perimeter_is_48_l169_169487


namespace Rob_has_three_dimes_l169_169577

theorem Rob_has_three_dimes (quarters dimes nickels pennies : ℕ) 
                            (val_quarters val_nickels val_pennies : ℚ)
                            (total_amount : ℚ) :
  quarters = 7 →
  nickels = 5 →
  pennies = 12 →
  val_quarters = 0.25 →
  val_nickels = 0.05 →
  val_pennies = 0.01 →
  total_amount = 2.42 →
  (7 * 0.25 + 5 * 0.05 + 12 * 0.01 + dimes * 0.10 = total_amount) →
  dimes = 3 :=
by sorry

end Rob_has_three_dimes_l169_169577


namespace min_value_l169_169500

noncomputable def f (x : ℝ) : ℝ := 2017 * x + Real.sin (x / 2018) + (2019 ^ x - 1) / (2019 ^ x + 1)

theorem min_value (a b : ℝ) (ha : 0 < a) (hb : 0 < b)
  (h : f (2 * a) + f (b - 4) = 0) :
  2 * a + b = 4 → (1 / a + 2 / b) = 2 :=
by sorry

end min_value_l169_169500


namespace container_weight_l169_169478

-- Definition of the problem conditions
def weight_of_copper_bar : ℕ := 90
def weight_of_steel_bar := weight_of_copper_bar + 20
def weight_of_tin_bar := weight_of_steel_bar / 2

-- Formal statement to be proven
theorem container_weight (n : ℕ) (h1 : weight_of_steel_bar = 2 * weight_of_tin_bar)
  (h2 : weight_of_steel_bar = weight_of_copper_bar + 20)
  (h3 : weight_of_copper_bar = 90) :
  20 * (weight_of_copper_bar + weight_of_steel_bar + weight_of_tin_bar) = 5100 := 
by sorry

end container_weight_l169_169478


namespace randy_blocks_l169_169083

theorem randy_blocks (total_blocks house_blocks diff_blocks tower_blocks : ℕ) 
  (h_total : total_blocks = 90)
  (h_house : house_blocks = 89)
  (h_diff : house_blocks = tower_blocks + diff_blocks)
  (h_diff_value : diff_blocks = 26) :
  tower_blocks = 63 :=
by
  -- sorry is placed here to skip the proof.
  sorry

end randy_blocks_l169_169083


namespace no_naturals_satisfy_m_squared_eq_n_squared_plus_2014_l169_169331

theorem no_naturals_satisfy_m_squared_eq_n_squared_plus_2014 :
  ∀ (m n : ℕ), ¬ (m^2 = n^2 + 2014) :=
by
  intro m n
  sorry

end no_naturals_satisfy_m_squared_eq_n_squared_plus_2014_l169_169331


namespace oshea_large_planters_l169_169836

theorem oshea_large_planters {total_seeds small_planter_capacity num_small_planters large_planter_capacity : ℕ} 
  (h1 : total_seeds = 200)
  (h2 : small_planter_capacity = 4)
  (h3 : num_small_planters = 30)
  (h4 : large_planter_capacity = 20) :
  (total_seeds - num_small_planters * small_planter_capacity) / large_planter_capacity = 4 :=
by
  sorry

end oshea_large_planters_l169_169836


namespace min_value_inequality_l169_169379

theorem min_value_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_sum : x + y + z = 1) : 
  (1 / x + 4 / y + 9 / z ≥ 36) ∧ 
  ((1 / x + 4 / y + 9 / z = 36) ↔ (x = 1 / 6 ∧ y = 1 / 3 ∧ z = 1 / 2)) :=
by
  sorry

end min_value_inequality_l169_169379


namespace sameTypeTerm_l169_169489

variable (a b : ℝ) -- Assume a and b are real numbers 

-- Definitions for each term in the conditions
def term1 : ℝ := 2 * a * b^2
def term2 : ℝ := -a^2 * b
def term3 : ℝ := -2 * a * b
def term4 : ℝ := 5 * a^2

-- The term we are comparing against
def compareTerm : ℝ := 3 * a^2 * b

-- The condition we want to prove
theorem sameTypeTerm : term2 = compareTerm :=
  sorry


end sameTypeTerm_l169_169489


namespace school_should_purchase_bookshelves_l169_169092

theorem school_should_purchase_bookshelves
  (x : ℕ)
  (h₁ : x ≥ 20)
  (cost_A : ℕ := 20 * 300 + 100 * (x - 20))
  (cost_B : ℕ := (20 * 300 + 100 * x) * 80 / 100)
  (h₂ : cost_A = cost_B) : x = 40 :=
by sorry

end school_should_purchase_bookshelves_l169_169092


namespace length_after_haircut_l169_169688

-- Definitions
def original_length : ℕ := 18
def cut_length : ℕ := 9

-- Target statement to prove
theorem length_after_haircut : original_length - cut_length = 9 :=
by
  -- Simplification and proof
  sorry

end length_after_haircut_l169_169688


namespace intersecting_rectangles_shaded_area_l169_169594

theorem intersecting_rectangles_shaded_area 
  (a_w : ℕ) (a_l : ℕ) (b_w : ℕ) (b_l : ℕ) (c_w : ℕ) (c_l : ℕ)
  (overlap_ab_w : ℕ) (overlap_ab_h : ℕ)
  (overlap_ac_w : ℕ) (overlap_ac_h : ℕ)
  (overlap_bc_w : ℕ) (overlap_bc_h : ℕ)
  (triple_overlap_w : ℕ) (triple_overlap_h : ℕ) :
  a_w = 4 → a_l = 12 →
  b_w = 5 → b_l = 10 →
  c_w = 3 → c_l = 6 →
  overlap_ab_w = 4 → overlap_ab_h = 5 →
  overlap_ac_w = 3 → overlap_ac_h = 4 →
  overlap_bc_w = 3 → overlap_bc_h = 3 →
  triple_overlap_w = 3 → triple_overlap_h = 3 →
  ((a_w * a_l) + (b_w * b_l) + (c_w * c_l)) - 
  ((overlap_ab_w * overlap_ab_h) + (overlap_ac_w * overlap_ac_h) + (overlap_bc_w * overlap_bc_h)) + 
  (triple_overlap_w * triple_overlap_h) = 84 :=
by 
  sorry

end intersecting_rectangles_shaded_area_l169_169594


namespace angle_ABC_is_83_l169_169656

-- Define a structure for the quadrilateral ABCD 
structure Quadrilateral (A B C D : Type) :=
  (angle_BAC : ℝ) -- Measure in degrees
  (angle_CAD : ℝ) -- Measure in degrees
  (angle_ACD : ℝ) -- Measure in degrees
  (side_AB : ℝ) -- Lengths of sides
  (side_AD : ℝ)
  (side_AC : ℝ)

-- Define the conditions from the problem
variable {A B C D : Type}
variable (quad : Quadrilateral A B C D)
variable (h1 : quad.angle_BAC = 60)
variable (h2 : quad.angle_CAD = 60)
variable (h3 : quad.angle_ACD = 23)
variable (h4 : quad.side_AB + quad.side_AD = quad.side_AC)

-- State the theorem to be proved
theorem angle_ABC_is_83 : quad.angle_ACD = 23 → quad.angle_CAD = 60 → 
                           quad.angle_BAC = 60 → quad.side_AB + quad.side_AD = quad.side_AC → 
                           ∃ angle_ABC : ℝ, angle_ABC = 83 := by
  sorry

end angle_ABC_is_83_l169_169656


namespace roots_polynomial_pq_sum_l169_169230

theorem roots_polynomial_pq_sum :
  ∀ p q : ℝ, 
  (∀ x : ℝ, (x - 1) * (x - 2) * (x - 3) * (x - 4) = x^4 - 10 * x^3 + p * x^2 - q * x + 24) 
  → p + q = 85 :=
by 
  sorry

end roots_polynomial_pq_sum_l169_169230


namespace man_older_than_son_l169_169278

theorem man_older_than_son (S M : ℕ) (hS : S = 27) (hM : M + 2 = 2 * (S + 2)) : M - S = 29 := 
by {
  sorry
}

end man_older_than_son_l169_169278


namespace correct_calculated_value_l169_169607

theorem correct_calculated_value (n : ℕ) (h : n + 9 = 30) : n + 7 = 28 :=
by
  sorry

end correct_calculated_value_l169_169607


namespace orthocenter_of_triangle_l169_169942

theorem orthocenter_of_triangle (A : ℝ × ℝ) (x y : ℝ) 
  (h₁ : x + y = 0) (h₂ : 2 * x - 3 * y + 1 = 0) : 
  A = (1, 2) → (x, y) = (-1 / 5, 1 / 5) :=
by
  sorry

end orthocenter_of_triangle_l169_169942


namespace union_M_N_intersection_complementM_N_l169_169668

open Set  -- Open the Set namespace for convenient notation.

noncomputable def funcDomain : Set ℝ := {x : ℝ | -1 ≤ x ∧ x < 2}
noncomputable def setN : Set ℝ := {x : ℝ | 0 < x ∧ x < 3}
noncomputable def complementFuncDomain : Set ℝ := {x : ℝ | x < -1 ∨ x ≥ 2}

theorem union_M_N :
  (funcDomain ∪ setN) = {x : ℝ | -1 ≤ x ∧ x < 3} :=
by
  sorry

theorem intersection_complementM_N :
  (complementFuncDomain ∩ setN) = {x : ℝ | 2 ≤ x ∧ x < 3} :=
by
  sorry

end union_M_N_intersection_complementM_N_l169_169668


namespace lottery_problem_l169_169804

theorem lottery_problem (n : ℕ) (hn : n ≥ 5) :
  (∑ i in finset.Ico 1 n, if (i ≤ n - 2) then combinations (n - 2) 3 else 0) =
  combinations n 3 :=
sorry

end lottery_problem_l169_169804


namespace parabola_opens_upward_l169_169100

theorem parabola_opens_upward (a : ℝ) (b : ℝ) (h : a > 0) : ∀ x : ℝ, 3*x^2 + 2 = a*x^2 + b → a = 3 ∧ b = 2 → ∀ x : ℝ, 3 * x^2 + 2 ≤ a * x^2 + b := 
by
  sorry

end parabola_opens_upward_l169_169100


namespace units_digit_of_17_pow_2007_l169_169735

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_of_17_pow_2007 : units_digit (17 ^ 2007) = 3 := by
  have h : ∀ n, units_digit (17 ^ n) = units_digit (7 ^ n) := by
    intro n
    sorry  -- Same units digit logic for powers of 17 as for powers of 7.
  have pattern : units_digit (7 ^ 1) = 7 ∧ 
                 units_digit (7 ^ 2) = 9 ∧ 
                 units_digit (7 ^ 3) = 3 ∧ 
                 units_digit (7 ^ 4) = 1 := by
    sorry  -- Units digit pattern for powers of 7.
  have mod_cycle : 2007 % 4 = 3 := by
    sorry  -- Calculation of 2007 mod 4.
  have result : units_digit (7 ^ 2007) = units_digit (7 ^ 3) := by
    rw [← mod_eq_of_lt (by norm_num : 2007 % 4 < 4), mod_cycle]
    exact (and.left (and.right (and.right pattern)))  -- Extract units digit of 7^3 from pattern.
  rw [h]
  exact result

end units_digit_of_17_pow_2007_l169_169735


namespace relationship_m_n_l169_169112

theorem relationship_m_n (m n : ℕ) (h : 10 / (m + 10 + n) = (m + n) / (m + 10 + n)) : m + n = 10 := 
by sorry

end relationship_m_n_l169_169112


namespace base6_base5_subtraction_in_base10_l169_169299

def base6_to_nat (n : ℕ) : ℕ :=
  3 * 6^2 + 2 * 6^1 + 5 * 6^0

def base5_to_nat (n : ℕ) : ℕ :=
  2 * 5^2 + 3 * 5^1 + 1 * 5^0

theorem base6_base5_subtraction_in_base10 : base6_to_nat 325 - base5_to_nat 231 = 59 := by
  sorry

end base6_base5_subtraction_in_base10_l169_169299


namespace no_nat_solutions_for_m2_eq_n2_plus_2014_l169_169361

theorem no_nat_solutions_for_m2_eq_n2_plus_2014 :
  ∀ m n : ℕ, ¬(m^2 = n^2 + 2014) := by
sorry

end no_nat_solutions_for_m2_eq_n2_plus_2014_l169_169361


namespace product_of_repeating_decimals_l169_169894

theorem product_of_repeating_decimals :
  let x := (4 / 9 : ℚ)
  let y := (7 / 9 : ℚ)
  x * y = 28 / 81 :=
by
  sorry

end product_of_repeating_decimals_l169_169894


namespace slower_train_speed_l169_169855

theorem slower_train_speed
  (v : ℝ)  -- The speed of the slower train
  (faster_train_speed : ℝ := 46)  -- The speed of the faster train
  (train_length : ℝ := 37.5)  -- The length of each train in meters
  (time_to_pass : ℝ := 27)  -- Time taken to pass in seconds
  (kms_to_ms : ℝ := 1000 / 3600)  -- Conversion factor from km/hr to m/s
  (relative_distance : ℝ := 2 * train_length)  -- Distance covered when passing

  (h : relative_distance = (faster_train_speed - v) * kms_to_ms * time_to_pass) :
  v = 36 :=
by
  -- The proof should be placed here
  sorry

end slower_train_speed_l169_169855


namespace road_construction_problem_l169_169616

theorem road_construction_problem (x : ℝ) (h₁ : x > 0) :
    1200 / x - 1200 / (1.20 * x) = 2 :=
by
  sorry

end road_construction_problem_l169_169616


namespace container_weight_l169_169479

-- Definition of the problem conditions
def weight_of_copper_bar : ℕ := 90
def weight_of_steel_bar := weight_of_copper_bar + 20
def weight_of_tin_bar := weight_of_steel_bar / 2

-- Formal statement to be proven
theorem container_weight (n : ℕ) (h1 : weight_of_steel_bar = 2 * weight_of_tin_bar)
  (h2 : weight_of_steel_bar = weight_of_copper_bar + 20)
  (h3 : weight_of_copper_bar = 90) :
  20 * (weight_of_copper_bar + weight_of_steel_bar + weight_of_tin_bar) = 5100 := 
by sorry

end container_weight_l169_169479


namespace first_character_more_lines_than_second_l169_169415

theorem first_character_more_lines_than_second :
  let x := 2
  let second_character_lines := 3 * x + 6
  20 - second_character_lines = 8 := by
  sorry

end first_character_more_lines_than_second_l169_169415


namespace isosceles_trapezoid_problem_l169_169822

variable (AB CD AD BC : ℝ)
variable (x : ℝ)

noncomputable def p_squared (AB CD AD BC : ℝ) (x : ℝ) : ℝ :=
  if AB = 100 ∧ CD = 25 ∧ AD = x ∧ BC = x then 1875 else 0

theorem isosceles_trapezoid_problem (h₁ : AB = 100)
                                    (h₂ : CD = 25)
                                    (h₃ : AD = x)
                                    (h₄ : BC = x) :
  p_squared AB CD AD BC x = 1875 := by
  sorry

end isosceles_trapezoid_problem_l169_169822


namespace linear_coefficient_l169_169097

def polynomial := (x : ℝ) -> x^2 - 2*x - 3

theorem linear_coefficient (x : ℝ) : (polynomial x) = x^2 - 2*x - 3 → -2 :=
by
  sorry

end linear_coefficient_l169_169097


namespace fractionSpentOnMachinery_l169_169277

-- Given conditions
def companyCapital (C : ℝ) : Prop := 
  ∃ remainingCapital, remainingCapital = 0.675 * C ∧ 
  ∃ rawMaterial, rawMaterial = (1/4) * C ∧ 
  ∃ remainingAfterRaw, remainingAfterRaw = (3/4) * C ∧ 
  ∃ spentOnMachinery, spentOnMachinery = remainingAfterRaw - remainingCapital

-- Question translated to Lean statement
theorem fractionSpentOnMachinery (C : ℝ) (h : companyCapital C) : 
  ∃ remainingAfterRaw spentOnMachinery,
    spentOnMachinery / remainingAfterRaw = 1/10 :=
by 
  sorry

end fractionSpentOnMachinery_l169_169277


namespace probability_area_less_than_circumference_l169_169878

theorem probability_area_less_than_circumference :
  let probability (d : ℕ) := if d = 2 then (1 / 100 : ℚ)
                             else if d = 3 then (1 / 50 : ℚ)
                             else 0
  let sum_prob (d_s : List ℚ) := d_s.foldl (· + ·) 0
  let outcomes : List ℕ := List.range' 2 19 -- dice sum range from 2 to 20
  let valid_outcomes : List ℕ := outcomes.filter (· < 4)
  sum_prob (valid_outcomes.map probability) = (3 / 100 : ℚ) :=
by
  sorry

end probability_area_less_than_circumference_l169_169878


namespace sum_of_ages_l169_169614

theorem sum_of_ages (P K : ℕ) (h1 : P - 7 = 3 * (K - 7)) (h2 : P + 2 = 2 * (K + 2)) : P + K = 50 :=
by
  sorry

end sum_of_ages_l169_169614


namespace bmws_sold_l169_169615

-- Definitions stated by the problem:
def total_cars : ℕ := 300
def percentage_mercedes : ℝ := 0.20
def percentage_toyota : ℝ := 0.25
def percentage_nissan : ℝ := 0.10
def percentage_bmws : ℝ := 1 - (percentage_mercedes + percentage_toyota + percentage_nissan)

-- Statement to prove:
theorem bmws_sold : (total_cars : ℝ) * percentage_bmws = 135 := by
  sorry

end bmws_sold_l169_169615


namespace canal_cross_section_area_l169_169584

/-- Definitions of the conditions -/
def top_width : Real := 6
def bottom_width : Real := 4
def depth : Real := 257.25

/-- Proof statement -/
theorem canal_cross_section_area : 
  (1 / 2) * (top_width + bottom_width) * depth = 1286.25 :=
by
  sorry

end canal_cross_section_area_l169_169584


namespace mariana_socks_probability_l169_169696

open Finset

theorem mariana_socks_probability :
  let colors := ({0, 1, 2, 3, 4} : Finset ℕ) in
  let pairs := colors.powerset.filter(λ s, s.card = 3) in
  let favorable_outcomes := pairs.card * (Finset.card (erase (singleton 2))) * 1 * 1 * 2 in
  let total_combinations := choose 10 5 in
  (favorable_outcomes : ℚ / total_combinations : ℚ) = 5 / 21 :=
by
  sorry

end mariana_socks_probability_l169_169696


namespace paul_number_proof_l169_169837

theorem paul_number_proof (a b : ℕ) (h₀ : 0 ≤ a ∧ a ≤ 9) (h₁ : 0 ≤ b ∧ b ≤ 9) (h₂ : a - b = 7) :
  (10 * a + b = 81) ∨ (10 * a + b = 92) :=
  sorry

end paul_number_proof_l169_169837


namespace inscribed_circle_radii_rel_l169_169038

theorem inscribed_circle_radii_rel {a b c r r1 r2 : ℝ} :
  (a^2 + b^2 = c^2) ∧
  (r1 = (a / c) * r) ∧
  (r2 = (b / c) * r) →
  r^2 = r1^2 + r2^2 :=
by 
  sorry

end inscribed_circle_radii_rel_l169_169038


namespace regular_ngon_on_parallel_lines_l169_169657

theorem regular_ngon_on_parallel_lines (n : ℕ) : 
  (∃ f : ℝ → ℝ, (∀ m : ℕ, ∃ k : ℕ, f (m * (360 / n)) = k * (360 / n))) ↔
  n = 3 ∨ n = 4 ∨ n = 6 := 
sorry

end regular_ngon_on_parallel_lines_l169_169657


namespace principal_amount_l169_169993

theorem principal_amount (P : ℝ) (r t : ℝ) (d : ℝ) 
  (h1 : r = 7)
  (h2 : t = 2)
  (h3 : d = 49)
  (h4 : P * ((1 + r / 100) ^ t - 1) - P * (r * t / 100) = d) :
  P = 10000 :=
by sorry

end principal_amount_l169_169993


namespace beautiful_fold_through_F_l169_169075

noncomputable def beautiful_fold_probability := 
  let ABCD : set (ℝ × ℝ) := { p | (0 ≤ p.fst ∧ p.fst ≤ 1) ∧ (0 ≤ p.snd ∧ p.snd ≤ 1) } in
  let diagonals : set (ℝ × ℝ) := { p | p.fst = p.snd ∨ p.fst + p.snd = 1 } in
  calc
    _ = (measure_theory.measure (diagonals ∩ ABCD)) / 
          (measure_theory.measure ABCD) : sorry

theorem beautiful_fold_through_F :
  beautiful_fold_probability = 1/2 := sorry

end beautiful_fold_through_F_l169_169075


namespace no_nat_m_n_eq_m_squared_eq_n_squared_plus_2014_l169_169322

theorem no_nat_m_n_eq_m_squared_eq_n_squared_plus_2014 :
  ¬ ∃ (m n : ℕ), m^2 = n^2 + 2014 :=
by 
  sorry

end no_nat_m_n_eq_m_squared_eq_n_squared_plus_2014_l169_169322


namespace anna_candy_division_l169_169756

theorem anna_candy_division : 
  ∀ (total_candies friends : ℕ), 
  total_candies = 30 → 
  friends = 4 → 
  ∃ (candies_to_remove : ℕ), 
  candies_to_remove = 2 ∧ 
  (total_candies - candies_to_remove) % friends = 0 := 
by
  sorry

end anna_candy_division_l169_169756


namespace no_nat_solutions_m_sq_eq_n_sq_plus_2014_l169_169355

theorem no_nat_solutions_m_sq_eq_n_sq_plus_2014 :
  ¬ ∃ (m n : ℕ), m ^ 2 = n ^ 2 + 2014 := 
sorry

end no_nat_solutions_m_sq_eq_n_sq_plus_2014_l169_169355


namespace hyperbola_eccentricity_l169_169520

-- Define the context/conditions
noncomputable def hyperbola_vertex_to_asymptote_distance (a b e : ℝ) : Prop :=
  (2 = b / e)

noncomputable def hyperbola_focus_to_asymptote_distance (a b e : ℝ) : Prop :=
  (6 = b)

-- Define the main theorem to prove the eccentricity
theorem hyperbola_eccentricity (a b e : ℝ) (h1 : hyperbola_vertex_to_asymptote_distance a b e) (h2 : hyperbola_focus_to_asymptote_distance a b e) : 
  e = 3 := 
sorry 

end hyperbola_eccentricity_l169_169520


namespace sum_areas_of_eight_disks_l169_169367

noncomputable def eight_disks_sum_areas (C_radius disk_count : ℝ) 
  (cover_C : ℝ) (no_overlap : ℝ) (tangent_neighbors : ℝ) : ℕ :=
  let r := (2 - Real.sqrt 2)
  let area_one_disk := Real.pi * r^2
  let total_area := disk_count * area_one_disk
  let a := 48
  let b := 32
  let c := 2
  a + b + c

theorem sum_areas_of_eight_disks : eight_disks_sum_areas 1 8 1 1 1 = 82 :=
  by
  -- sorry is used to skip the proof
  sorry

end sum_areas_of_eight_disks_l169_169367


namespace central_angle_of_sector_l169_169522

theorem central_angle_of_sector (r A : ℝ) (h₁ : r = 4) (h₂ : A = 4) :
  (1 / 2) * r^2 * (1 / 4) = A :=
by
  sorry

end central_angle_of_sector_l169_169522


namespace school_children_count_l169_169699

-- Define the conditions
variable (A P C B G : ℕ)
variable (A_eq : A = 160)
variable (kids_absent : ∀ (present kids absent children : ℕ), present = kids - absent → absent = 160)
variable (bananas_received : ∀ (two_per child kids : ℕ), (2 * kids) + (2 * 160) = 2 * 6400 + (4 * (6400 / 160)))
variable (boys_girls : B = 3 * G)

-- State the theorem
theorem school_children_count (C : ℕ) (A P B G : ℕ) 
  (A_eq : A = 160)
  (kids_absent : P = C - A)
  (bananas_received : (2 * P) + (2 * A) = 2 * P + (4 * (P / A)))
  (boys_girls : B = 3 * G)
  (total_bananas : 2 * P + 4 * (P / A) = 12960) :
  C = 6560 := 
sorry

end school_children_count_l169_169699


namespace ratio_of_areas_l169_169458

theorem ratio_of_areas (Q : Point) (r1 r2 : ℝ) (h : r1 < r2)
  (arc_length_smaller : ℝ) (arc_length_larger : ℝ)
  (h_arc_smaller : arc_length_smaller = (60 / 360) * (2 * r1 * π))
  (h_arc_larger : arc_length_larger = (30 / 360) * (2 * r2 * π))
  (h_equal_arcs : arc_length_smaller = arc_length_larger) :
  (π * r1^2) / (π * r2^2) = 1/4 :=
by
  sorry

end ratio_of_areas_l169_169458


namespace vector_combination_l169_169390

open Complex

def z1 : ℂ := -1 + I
def z2 : ℂ := 1 + I
def z3 : ℂ := 1 + 4 * I

def A : ℝ × ℝ := (-1, 1)
def B : ℝ × ℝ := (1, 1)
def C : ℝ × ℝ := (1, 4)

def OA : ℝ × ℝ := A
def OB : ℝ × ℝ := B
def OC : ℝ × ℝ := C

def x : ℝ := sorry
def y : ℝ := sorry

theorem vector_combination (hx : OC = ( - x + y, x + y )) : 
    x + y = 4 :=
by
    sorry

end vector_combination_l169_169390


namespace candy_system_of_equations_l169_169890

-- Definitions based on conditions
def candy_weight := 100
def candy_price1 := 36
def candy_price2 := 20
def mixed_candy_price := 28

theorem candy_system_of_equations (x y: ℝ):
  (x + y = candy_weight) ∧ (candy_price1 * x + candy_price2 * y = mixed_candy_price * candy_weight) :=
sorry

end candy_system_of_equations_l169_169890


namespace range_of_m_l169_169021

-- Definitions according to the problem conditions
def p (x : ℝ) : Prop := (-2 ≤ x ∧ x ≤ 10)
def q (x : ℝ) (m : ℝ) : Prop := (1 - m ≤ x ∧ x ≤ 1 + m) ∧ m > 0

-- Rephrasing the problem statement in Lean
theorem range_of_m (x : ℝ) (m : ℝ) :
  (∀ x, p x → q x m) → m ≥ 9 :=
sorry

end range_of_m_l169_169021


namespace ln_of_gt_of_pos_l169_169025

variable {a b : ℝ}

theorem ln_of_gt_of_pos (h1 : a > b) (h2 : b > 0) : Real.log a > Real.log b :=
sorry

end ln_of_gt_of_pos_l169_169025


namespace simplify_first_expression_simplify_second_expression_l169_169987

theorem simplify_first_expression (x y : ℝ) : 3 * x - 2 * y + 1 + 3 * y - 2 * x - 5 = x + y - 4 :=
sorry

theorem simplify_second_expression (x : ℝ) : (2 * x ^ 4 - 5 * x ^ 2 - 4 * x + 3) - (3 * x ^ 3 - 5 * x ^ 2 - 4 * x) = 2 * x ^ 4 - 3 * x ^ 3 + 3 :=
sorry

end simplify_first_expression_simplify_second_expression_l169_169987


namespace rabbits_to_hamsters_l169_169999

theorem rabbits_to_hamsters (rabbits hamsters : ℕ) (h_ratio : 3 * hamsters = 4 * rabbits) (h_rabbits : rabbits = 18) : hamsters = 24 :=
by
  sorry

end rabbits_to_hamsters_l169_169999


namespace min_value_fraction_sum_l169_169518

theorem min_value_fraction_sum (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 4) :
  (∃ x : ℝ, x = (1 / a + 4 / b) ∧ x = 9 / 4) :=
by
  sorry

end min_value_fraction_sum_l169_169518


namespace triangle_classification_l169_169776

theorem triangle_classification (a b c : ℕ) (h : a + b + c = 12) :
((
  (a = b ∨ b = c ∨ a = c)  -- Isosceles
  ∨ (a^2 + b^2 = c^2 ∨ b^2 + c^2 = a^2 ∨ c^2 + a^2 = b^2)  -- Right-angled
  ∨ (a = b ∧ b = c)  -- Equilateral
)) :=
sorry

end triangle_classification_l169_169776


namespace quadratic_real_roots_range_l169_169786

theorem quadratic_real_roots_range (a : ℝ) : 
  (∃ x : ℝ, (a - 1) * x^2 - 2 * x + 1 = 0) ↔ (a ≤ 2) :=
by
-- Proof outline:
-- Case 1: when a = 1, the equation simplifies to -2x + 1 = 0, which has a real solution x = 1/2.
-- Case 2: when a ≠ 1, the quadratic equation has real roots if the discriminant 8 - 4a ≥ 0, i.e., 2 ≥ a.
sorry

end quadratic_real_roots_range_l169_169786


namespace total_cost_of_returned_packets_l169_169567

/--
  Martin bought 10 packets of milk with varying prices.
  The average price (arithmetic mean) of all the packets is 25¢.
  If Martin returned three packets to the retailer, and the average price of the remaining packets was 20¢,
  then the total cost, in cents, of the three returned milk packets is 110¢.
-/
theorem total_cost_of_returned_packets 
  (T10 : ℕ) (T7 : ℕ) (average_price_10 : T10 / 10 = 25)
  (average_price_7 : T7 / 7 = 20) :
  (T10 - T7 = 110) := 
sorry

end total_cost_of_returned_packets_l169_169567


namespace find_f_of_7_l169_169663

variable {f : ℝ → ℝ}

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def is_periodic_function (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

theorem find_f_of_7 (h1 : is_odd_function f)
                    (h2 : is_periodic_function f 4)
                    (h3 : ∀ x, 0 < x ∧ x < 2 → f x = 2 * x^2) :
  f 7 = -2 := 
by
  sorry

end find_f_of_7_l169_169663


namespace range_of_m_l169_169524

theorem range_of_m (m x1 x2 y1 y2 : ℝ) (h1 : y1 = (1 + 2 * m) / x1) (h2 : y2 = (1 + 2 * m) / x2)
    (hx : x1 < 0 ∧ 0 < x2) (hy : y1 < y2) : m > -1 / 2 :=
sorry

end range_of_m_l169_169524


namespace minimum_value_of_a_l169_169199

theorem minimum_value_of_a (x : ℝ) (a : ℝ) (hx : 0 ≤ x) (hx2 : x ≤ 20) (ha : 0 < a) (h : (20 - x) / 4 + a / 2 * Real.sqrt x ≥ 5) : 
  a ≥ Real.sqrt 5 := 
sorry

end minimum_value_of_a_l169_169199


namespace number_of_red_balls_l169_169964

-- Definitions and conditions
def ratio_white_red (w : ℕ) (r : ℕ) : Prop := (w : ℤ) * 3 = 5 * (r : ℤ)
def white_balls : ℕ := 15

-- The theorem to prove
theorem number_of_red_balls (r : ℕ) (h : ratio_white_red white_balls r) : r = 9 :=
by
  sorry

end number_of_red_balls_l169_169964


namespace sunflower_seeds_contest_l169_169159

theorem sunflower_seeds_contest 
  (first_player_seeds : ℕ) (second_player_seeds : ℕ) (total_seeds : ℕ) 
  (third_player_seeds : ℕ) (third_more : ℕ) 
  (h1 : first_player_seeds = 78) 
  (h2 : second_player_seeds = 53) 
  (h3 : total_seeds = 214) 
  (h4 : first_player_seeds + second_player_seeds + third_player_seeds = total_seeds) 
  (h5 : third_more = third_player_seeds - second_player_seeds) : 
  third_more = 30 :=
by
  sorry

end sunflower_seeds_contest_l169_169159


namespace simplify_expression_l169_169707

-- Define the statement we want to prove
theorem simplify_expression (s : ℕ) : (105 * s - 63 * s) = 42 * s :=
  by
    -- Placeholder for the proof
    sorry

end simplify_expression_l169_169707


namespace burger_cost_l169_169806

theorem burger_cost
  (B P : ℝ)
  (h₁ : P = 2 * B)
  (h₂ : P + 3 * B = 45) :
  B = 9 := by
  sorry

end burger_cost_l169_169806


namespace original_rice_amount_l169_169852

theorem original_rice_amount (x : ℝ) 
  (h1 : (x / 2) - 3 = 18) : 
  x = 42 :=
sorry

end original_rice_amount_l169_169852


namespace no_positive_a_for_inequality_l169_169366

theorem no_positive_a_for_inequality (a : ℝ) (h : 0 < a) : 
  ¬ ∀ x : ℝ, |Real.cos x| + |Real.cos (a * x)| > Real.sin x + Real.sin (a * x) := by
  sorry

end no_positive_a_for_inequality_l169_169366


namespace regular_polygon_exterior_angle_l169_169402

theorem regular_polygon_exterior_angle (n : ℕ) (h : n > 2) (h_exterior : 36 = 360 / n) : n = 10 :=
sorry

end regular_polygon_exterior_angle_l169_169402


namespace minimum_value_of_k_l169_169949

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * x + b
noncomputable def g (a c : ℝ) (x : ℝ) : ℝ := a * x + c
noncomputable def h (a b c : ℝ) (x : ℝ) : ℝ := (f a b x)^2 + 8 * (g a c x)
noncomputable def k (a b c : ℝ) (x : ℝ) : ℝ := (g a c x)^2 + 8 * (f a b x)

theorem minimum_value_of_k:
  ∃ a b c : ℝ, a ≠ 0 ∧ (∀ x : ℝ, h a b c x ≥ -29) → (∃ x : ℝ, k a b c x = -3) := sorry

end minimum_value_of_k_l169_169949


namespace LCM_of_fractions_l169_169856

noncomputable def LCM (a b : Rat) : Rat :=
  a * b / (gcd a.num b.num / gcd a.den b.den : Int)

theorem LCM_of_fractions (x : ℤ) (h : x ≠ 0) :
  LCM (1 / (4 * x : ℚ)) (LCM (1 / (6 * x : ℚ)) (1 / (9 * x : ℚ))) = 1 / (36 * x) :=
by
  sorry

end LCM_of_fractions_l169_169856


namespace no_nat_solutions_m2_eq_n2_plus_2014_l169_169309

theorem no_nat_solutions_m2_eq_n2_plus_2014 :
  ¬ ∃ (m n : ℕ), m^2 = n^2 + 2014 :=
by
  sorry

end no_nat_solutions_m2_eq_n2_plus_2014_l169_169309


namespace floor_sqrt_120_l169_169910

theorem floor_sqrt_120 : (⌊Real.sqrt 120⌋ = 10) :=
by
  -- Conditions from the problem
  have h1: 10^2 = 100 := rfl
  have h2: 11^2 = 121 := rfl
  have h3: 10 < Real.sqrt 120 := sorry
  have h4: Real.sqrt 120 < 11 := sorry
  -- Proof goal
  sorry

end floor_sqrt_120_l169_169910


namespace tommy_expected_value_score_l169_169117

/-- Tommy's 25-question true-false test setup -/
def tommy_test : {n : ℕ} → distr (vector bool n) :=
λ n, distr.uniform (vector bool n) 

/-- Streak points calculation -/
def streak_points (qs : vector bool 25) : ℕ :=
qs.to_list.foldl (λ ⟨streak, total⟩ q, 
  if q then (streak + 1, total + (streak + 1)) 
  else (0, total)) (0, 0)).2

/-- Expected value of Tommy's score -/
def expected_value_of_tommy_score : ℕ :=
25 * 2

/-- Prove that the expected value of Tommy’s score is 50 -/
theorem tommy_expected_value_score:
  ∑ (q : vector bool 25), (tommy_test 25).pdf q * (streak_points q) = expected_value_of_tommy_score :=
sorry

end tommy_expected_value_score_l169_169117


namespace problem_statement_l169_169952

theorem problem_statement
  (a b : ℝ)
  (ha : a = Real.sqrt 2 + 1)
  (hb : b = Real.sqrt 2 - 1) :
  a^2 - a * b + b^2 = 5 :=
sorry

end problem_statement_l169_169952


namespace perfect_cubes_not_divisible_by_10_l169_169648

-- Definitions based on conditions
def is_divisible_by_10 (n : ℕ) : Prop := 10 ∣ n
def is_perfect_cube (n : ℕ) : Prop := ∃ (k : ℕ), n = k ^ 3
def erase_last_three_digits (n : ℕ) : ℕ := n / 1000

-- Main statement
theorem perfect_cubes_not_divisible_by_10 (x : ℕ) :
  is_perfect_cube x ∧ ¬ is_divisible_by_10 x ∧ is_perfect_cube (erase_last_three_digits x) →
  x = 1331 ∨ x = 1728 :=
by
  sorry

end perfect_cubes_not_divisible_by_10_l169_169648


namespace man_completion_time_l169_169743

theorem man_completion_time (w_time : ℕ) (efficiency_increase : ℚ) (m_time : ℕ) :
  w_time = 40 → efficiency_increase = 1.25 → m_time = (w_time : ℚ) / efficiency_increase → m_time = 32 :=
by
  sorry

end man_completion_time_l169_169743


namespace number_of_students_l169_169570

def candiesPerStudent : ℕ := 2
def totalCandies : ℕ := 18
def expectedStudents : ℕ := 9

theorem number_of_students :
  totalCandies / candiesPerStudent = expectedStudents :=
sorry

end number_of_students_l169_169570


namespace find_m_n_l169_169926

theorem find_m_n (x : ℝ) (m n : ℝ) 
  (h : (2 * x - 5) * (x + m) = 2 * x^2 - 3 * x + n) :
  m = 1 ∧ n = -5 :=
by
  have h_expand : (2 * x - 5) * (x + m) = 2 * x^2 + (2 * m - 5) * x - 5 * m := by
    ring
  rw [h_expand] at h
  have coeff_eq1 : 2 * m - 5 = -3 := by sorry
  have coeff_eq2 : -5 * m = n := by sorry
  have m_sol : m = 1 := by
    linarith [coeff_eq1]
  have n_sol : n = -5 := by
    rw [m_sol] at coeff_eq2
    linarith
  exact ⟨m_sol, n_sol⟩

end find_m_n_l169_169926


namespace darla_total_payment_l169_169899

-- Definitions of the conditions
def rate_per_watt : ℕ := 4
def energy_usage : ℕ := 300
def late_fee : ℕ := 150

-- Definition of the expected total cost
def expected_total_cost : ℕ := 1350

-- Theorem stating the problem
theorem darla_total_payment :
  rate_per_watt * energy_usage + late_fee = expected_total_cost := 
by 
  sorry

end darla_total_payment_l169_169899


namespace find_b_l169_169250

theorem find_b (b p : ℚ) :
  (∀ x : ℚ, (2 * x^3 + b * x + 7 = (x^2 + p * x + 1) * (2 * x + 7))) →
  b = -45 / 2 :=
sorry

end find_b_l169_169250


namespace no_nat_solutions_for_m2_eq_n2_plus_2014_l169_169359

theorem no_nat_solutions_for_m2_eq_n2_plus_2014 :
  ∀ m n : ℕ, ¬(m^2 = n^2 + 2014) := by
sorry

end no_nat_solutions_for_m2_eq_n2_plus_2014_l169_169359


namespace problem1_problem2_l169_169788

noncomputable def h (x a : ℝ) : ℝ := (x - a) * Real.exp x + a
noncomputable def f (x b : ℝ) : ℝ := x^2 - 2 * b * x - 3 * Real.exp 1 + Real.exp 1 + 15 / 2

theorem problem1 (a : ℝ) :
  ∃ c, ∀ x ∈ Set.Icc (-1:ℝ) (1:ℝ), h x a ≥ c :=
by
  sorry

theorem problem2 (b : ℝ) :
  (∀ x1 ∈ Set.Icc (-1:ℝ) (1:ℝ), ∃ x2 ∈ Set.Icc (1:ℝ) (2:ℝ), h x1 3 ≥ f x2 b) →
  b ≥ 17 / 8 :=
by
  sorry

end problem1_problem2_l169_169788


namespace factor_polynomial_l169_169772

theorem factor_polynomial (a b : ℕ) : 
  2 * a^3 - 3 * a^2 * b - 3 * a * b^2 + 2 * b^3 = (a + b) * (a - 2 * b) * (2 * a - b) :=
by sorry

end factor_polynomial_l169_169772


namespace minimum_value_l169_169229

-- Define the expression E(a, b, c)
def E (a b c : ℝ) : ℝ := a^2 + 8 * a * b + 24 * b^2 + 16 * b * c + 6 * c^2

-- State the minimum value theorem
theorem minimum_value (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  E a b c = 18 :=
sorry

end minimum_value_l169_169229


namespace maximum_radius_l169_169670

open Set Real

-- Definitions of sets M, N, and D_r.
def M : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.snd ≥ 1 / 4 * p.fst^2}

def N : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.snd ≤ -1 / 4 * p.fst^2 + p.fst + 7}

def D_r (x₀ y₀ r : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.fst - x₀)^2 + (p.snd - y₀)^2 ≤ r^2}

-- Theorem statement for the largest r
theorem maximum_radius {x₀ y₀ : ℝ} (H : D_r x₀ y₀ r ⊆ M ∩ N) :
  r = sqrt ((25 - 5 * sqrt 5) / 2) :=
sorry

end maximum_radius_l169_169670


namespace simplify_div_l169_169431

theorem simplify_div : (27 * 10^12) / (9 * 10^4) = 3 * 10^8 := 
by
  sorry

end simplify_div_l169_169431


namespace range_of_a_l169_169662

theorem range_of_a (a : ℝ) (p q : Prop) 
    (h₀ : (p ↔ (3 - 2 * a > 1))) 
    (h₁ : (q ↔ (-2 < a ∧ a < 2))) 
    (h₂ : (p ∨ q)) 
    (h₃ : ¬ (p ∧ q)) : 
    a ≤ -2 ∨ 1 ≤ a ∧ a < 2 :=
by
  sorry

end range_of_a_l169_169662


namespace Jesse_pages_left_to_read_l169_169067

def pages_read := [10, 15, 27, 12, 19]
def total_pages_read := pages_read.sum
def fraction_read : ℚ := 1 / 3
def total_pages : ℚ := total_pages_read / fraction_read
def pages_left_to_read : ℚ := total_pages - total_pages_read

theorem Jesse_pages_left_to_read :
  pages_left_to_read = 166 := by
  sorry

end Jesse_pages_left_to_read_l169_169067


namespace virginia_taught_fewer_years_l169_169729

-- Definitions based on conditions
variable (V A D : ℕ)

-- Dennis has taught for 34 years
axiom h1 : D = 34

-- Virginia has taught for 9 more years than Adrienne
axiom h2 : V = A + 9

-- Combined total of years taught is 75
axiom h3 : V + A + D = 75

-- Proof statement: Virginia has taught for 9 fewer years than Dennis
theorem virginia_taught_fewer_years : D - V = 9 :=
  sorry

end virginia_taught_fewer_years_l169_169729


namespace turban_price_l169_169023

theorem turban_price (T : ℝ) (total_salary : ℝ) (received_salary : ℝ)
  (cond1 : total_salary = 90 + T)
  (cond2 : received_salary = 65 + T)
  (cond3 : received_salary = (3 / 4) * total_salary) :
  T = 10 :=
by
  sorry

end turban_price_l169_169023


namespace bicyclist_speed_remainder_l169_169979

theorem bicyclist_speed_remainder (total_distance first_distance remainder_distance first_speed avg_speed remainder_speed time_total time_first time_remainder : ℝ) 
  (H1 : total_distance = 350)
  (H2 : first_distance = 200)
  (H3 : remainder_distance = total_distance - first_distance)
  (H4 : first_speed = 20)
  (H5 : avg_speed = 17.5)
  (H6 : time_total = total_distance / avg_speed)
  (H7 : time_first = first_distance / first_speed)
  (H8 : time_remainder = time_total - time_first)
  (H9 : remainder_speed = remainder_distance / time_remainder) :
  remainder_speed = 15 := 
sorry

end bicyclist_speed_remainder_l169_169979


namespace even_function_order_l169_169197

noncomputable def f (m : ℝ) (x : ℝ) := (m - 1) * x^2 + 6 * m * x + 2

theorem even_function_order (m : ℝ) (h_even : ∀ x : ℝ, f m (-x) = f m x) : 
  m = 0 ∧ f m (-2) < f m 1 ∧ f m 1 < f m 0 := by
sorry

end even_function_order_l169_169197


namespace no_nat_solutions_for_m2_eq_n2_plus_2014_l169_169357

theorem no_nat_solutions_for_m2_eq_n2_plus_2014 :
  ∀ m n : ℕ, ¬(m^2 = n^2 + 2014) := by
sorry

end no_nat_solutions_for_m2_eq_n2_plus_2014_l169_169357


namespace arithmetic_geometric_progression_l169_169445

-- Define the arithmetic progression terms
def u (a d : ℝ) (n : ℕ) : ℝ := a + (n - 1) * d

-- Define the property that the squares of the 12th, 13th, and 15th terms form a geometric progression
def geometric_progression (a d : ℝ) : Prop :=
  let u12 := u a d 12
  let u13 := u a d 13
  let u15 := u a d 15
  (u13^2 / u12^2 = u15^2 / u13^2)

-- The main statement
theorem arithmetic_geometric_progression (a d : ℝ) (h : geometric_progression a d) :
  d = 0 ∨ 4 * ((a + 11 * d)^2) = (a + 12 *d)^2 * (a + 14 * d)^2 / (a + 12 * d)^2 ∨ (a + 11 * d) * ((a + 11 * d) - 2 *d) = 0 :=
sorry

end arithmetic_geometric_progression_l169_169445


namespace least_positive_integer_l169_169731

theorem least_positive_integer (k : ℕ) (h : (528 + k) % 5 = 0) : k = 2 :=
sorry

end least_positive_integer_l169_169731


namespace units_digit_17_pow_2007_l169_169737

theorem units_digit_17_pow_2007 : (17^2007) % 10 = 3 :=
by sorry

end units_digit_17_pow_2007_l169_169737


namespace ln_of_gt_of_pos_l169_169026

variable {a b : ℝ}

theorem ln_of_gt_of_pos (h1 : a > b) (h2 : b > 0) : Real.log a > Real.log b :=
sorry

end ln_of_gt_of_pos_l169_169026


namespace problem_part1_problem_part2_l169_169937

variable (α : Real)
variable (h : Real.tan α = 1 / 2)

theorem problem_part1 : 
  (2 * Real.cos α - 3 * Real.sin α) / (3 * Real.cos α + 4 * Real.sin α) = 1 / 10 := sorry

theorem problem_part2 : 
  Real.sin α ^ 2 - 3 * Real.sin α * Real.cos α + 4 * Real.cos α ^ 2 = 11 / 5 := sorry

end problem_part1_problem_part2_l169_169937


namespace line_shift_up_l169_169247

theorem line_shift_up (x y : ℝ) (k : ℝ) (h : y = -2 * x - 4) : 
    y + k = -2 * x - 1 := by
  sorry

end line_shift_up_l169_169247


namespace possible_dimensions_of_plot_l169_169622

theorem possible_dimensions_of_plot (x : ℕ) :
  (∃ a b : ℕ, a < 10 ∧ b < 10 ∧ 1000 * a + 100 * a + 10 * b + b = x * (x + 1)) →
  x = 33 ∨ x = 66 ∨ x = 99 :=
sorry

end possible_dimensions_of_plot_l169_169622


namespace algebra_expression_value_l169_169386

theorem algebra_expression_value (a : ℝ) (h : a^2 - 4 * a - 6 = 0) : a^2 - 4 * a + 3 = 9 :=
by
  sorry

end algebra_expression_value_l169_169386


namespace james_carrot_sticks_l169_169815

theorem james_carrot_sticks (x : ℕ) (h : x + 15 = 37) : x = 22 :=
by {
  sorry
}

end james_carrot_sticks_l169_169815


namespace square_presses_exceed_1000_l169_169135

theorem square_presses_exceed_1000:
  ∃ n : ℕ, (n = 3) ∧ (3 ^ (2^n) > 1000) :=
by
  sorry

end square_presses_exceed_1000_l169_169135


namespace no_solution_exists_l169_169305

theorem no_solution_exists (m n : ℕ) : ¬ (m^2 = n^2 + 2014) :=
by
  sorry

end no_solution_exists_l169_169305


namespace no_nat_numbers_m_n_satisfy_eq_l169_169341

theorem no_nat_numbers_m_n_satisfy_eq (m n : ℕ) : ¬ (m^2 = n^2 + 2014) := sorry

end no_nat_numbers_m_n_satisfy_eq_l169_169341


namespace conditional_prob_B_given_A_l169_169701

namespace ConditionalProbability

def questions_total := 5
def science_questions := 3
def arts_questions := 2
def draw_first := 1
def draw_second := 2

def event_A (q : ℕ) : Prop := q = draw_first → science_questions
def event_B (q : ℕ) : Prop := q = draw_second → science_questions

theorem conditional_prob_B_given_A :
  P(event_B | event_A) = 1 / 2 := sorry

end ConditionalProbability

end conditional_prob_B_given_A_l169_169701


namespace solve_problem_l169_169796

noncomputable def proof_problem (x y : ℝ) : Prop :=
  (0.65 * x > 26) ∧ (0.40 * y < -3) ∧ ((x - y)^2 ≥ 100) 
  → (x > 40) ∧ (y < -7.5)

theorem solve_problem (x y : ℝ) (h : proof_problem x y) : (x > 40) ∧ (y < -7.5) := 
sorry

end solve_problem_l169_169796


namespace sandbag_weight_proof_l169_169454

-- Define all given conditions
def bag_capacity : ℝ := 250
def fill_percentage : ℝ := 0.80
def material_weight_multiplier : ℝ := 1.40 -- since 40% heavier means 1 + 0.40
def empty_bag_weight : ℝ := 0

-- Using these definitions, form the goal to prove
theorem sandbag_weight_proof : 
  (fill_percentage * bag_capacity * material_weight_multiplier) + empty_bag_weight = 280 :=
by
  sorry

end sandbag_weight_proof_l169_169454


namespace train_length_l169_169141

noncomputable def jogger_speed_kmh : ℝ := 9
noncomputable def train_speed_kmh : ℝ := 45
noncomputable def head_start : ℝ := 270
noncomputable def passing_time : ℝ := 39

noncomputable def kmh_to_ms (speed: ℝ) : ℝ := speed * (1000 / 3600)

theorem train_length (l : ℝ) 
  (v_j := kmh_to_ms jogger_speed_kmh)
  (v_t := kmh_to_ms train_speed_kmh)
  (d_h := head_start)
  (t := passing_time) :
  l = 120 :=
by 
  sorry

end train_length_l169_169141


namespace geom_seq_a4_a5_a6_value_l169_169037

theorem geom_seq_a4_a5_a6_value (a : ℕ → ℝ) (h_geom : ∃ r, 0 < r ∧ ∀ n, a (n + 1) = r * a n)
  (h_roots : ∃ x y, x * y = 16 ∧ x + y = 10 ∧ a 1 = x ∧ a 9 = y) :
  a 4 * a 5 * a 6 = 64 :=
by
  sorry

end geom_seq_a4_a5_a6_value_l169_169037


namespace problem_solution_l169_169739

theorem problem_solution : (6 * 7 * 8 * 9 * 10) / (6 + 7 + 8 + 9 + 10) = 756 := by
  sorry

end problem_solution_l169_169739


namespace total_number_of_flowers_l169_169893

theorem total_number_of_flowers : 
  let red_roses := 1491
  let yellow_carnations := 3025
  let white_roses := 1768
  let purple_tulips := 2150
  let pink_daisies := 3500
  let blue_irises := 2973
  let orange_marigolds := 4234
  red_roses + yellow_carnations + white_roses + purple_tulips + pink_daisies + blue_irises + orange_marigolds = 19141 :=
by 
  sorry

end total_number_of_flowers_l169_169893


namespace fundraiser_total_money_l169_169653

def fundraiser_money : ℝ :=
  let brownies_students := 70
  let brownies_each := 20
  let brownies_price := 1.50
  let cookies_students := 40
  let cookies_each := 30
  let cookies_price := 2.25
  let donuts_students := 35
  let donuts_each := 18
  let donuts_price := 3.00
  let cupcakes_students := 25
  let cupcakes_each := 12
  let cupcakes_price := 2.50
  let total_brownies := brownies_students * brownies_each
  let total_cookies := cookies_students * cookies_each
  let total_donuts := donuts_students * donuts_each
  let total_cupcakes := cupcakes_students * cupcakes_each
  let money_brownies := total_brownies * brownies_price
  let money_cookies := total_cookies * cookies_price
  let money_donuts := total_donuts * donuts_price
  let money_cupcakes := total_cupcakes * cupcakes_price
  money_brownies + money_cookies + money_donuts + money_cupcakes

theorem fundraiser_total_money : fundraiser_money = 7440 := sorry

end fundraiser_total_money_l169_169653


namespace sum_of_squares_l169_169446

theorem sum_of_squares (x y : ℝ) (h1 : x + y = 21) (h2 : x * y = 43) : x^2 + y^2 = 355 :=
sorry

end sum_of_squares_l169_169446


namespace sphere_surface_area_from_box_l169_169248

/--
Given a rectangular box with length = 2, width = 2, and height = 1,
prove that if all vertices of the rectangular box lie on the surface of a sphere,
then the surface area of the sphere is 9π.
--/
theorem sphere_surface_area_from_box :
  let length := 2
  let width := 2
  let height := 1
  ∃ (r : ℝ), ∀ (d := Real.sqrt (length^2 + width^2 + height^2)),
  r = d / 2 → 4 * Real.pi * r^2 = 9 * Real.pi :=
by
  sorry

end sphere_surface_area_from_box_l169_169248


namespace intersection_correct_l169_169233

open Set

def M := {x : ℝ | x^2 + x - 6 < 0}
def N := {x : ℝ | 1 ≤ x ∧ x ≤ 3}
def intersection := (M ∩ N) = {x : ℝ | 1 ≤ x ∧ x < 2}

theorem intersection_correct : intersection := by
  sorry

end intersection_correct_l169_169233


namespace no_naturals_satisfy_m_squared_eq_n_squared_plus_2014_l169_169327

theorem no_naturals_satisfy_m_squared_eq_n_squared_plus_2014 :
  ∀ (m n : ℕ), ¬ (m^2 = n^2 + 2014) :=
by
  intro m n
  sorry

end no_naturals_satisfy_m_squared_eq_n_squared_plus_2014_l169_169327


namespace spelling_bee_initial_students_l169_169808

theorem spelling_bee_initial_students (x : ℕ) 
    (h1 : (2 / 3) * x = 2 / 3 * x)
    (h2 : (3 / 4) * ((1 / 3) * x) = 3 / 4 * (1 / 3 * x))
    (h3 : (1 / 3) * x * (1 / 4) = 30) : 
  x = 120 :=
sorry

end spelling_bee_initial_students_l169_169808


namespace solution_set_of_inequality_l169_169589

theorem solution_set_of_inequality (x : ℝ) : x * (1 - x) > 0 ↔ 0 < x ∧ x < 1 :=
by sorry

end solution_set_of_inequality_l169_169589


namespace unattainable_y_value_l169_169509

theorem unattainable_y_value :
  ∀ (y x : ℝ), (y = (1 - x) / (2 * x^2 + 3 * x + 4)) → (∀ x, 2 * x^2 + 3 * x + 4 ≠ 0) → y ≠ 0 :=
by
  intros y x h1 h2
  -- Proof to be provided
  sorry

end unattainable_y_value_l169_169509


namespace greatest_integer_c_not_in_range_l169_169118

theorem greatest_integer_c_not_in_range :
  ∃ c : ℤ, (¬ ∃ x : ℝ, x^2 + (c:ℝ)*x + 18 = -6) ∧ (∀ c' : ℤ, c' > c → (∃ x : ℝ, x^2 + (c':ℝ)*x + 18 = -6)) :=
sorry

end greatest_integer_c_not_in_range_l169_169118


namespace main_theorem_l169_169830

variables {m n : ℕ} {x : ℝ}
variables {a : ℕ → ℕ}
noncomputable def relatively_prime (a : ℕ → ℕ) (n : ℕ) : Prop :=
∀ i j, i ≠ j → i < n → j < n → Nat.gcd (a i) (a j) = 1

noncomputable def distinct (a : ℕ → ℕ) (n : ℕ) : Prop :=
∀ i j, i ≠ j → i < n → j < n → a i ≠ a j

theorem main_theorem (hm : 1 < m) (hn : 1 < n) (hge : m ≥ n)
  (hrel_prime : relatively_prime a n)
  (hdistinct : distinct a n)
  (hbound : ∀ i, i < n → a i ≤ m)
  : ∃ i, i < n ∧ ‖a i * x‖ ≥ (2 / (m * (m + 1))) * ‖x‖ := 
sorry

end main_theorem_l169_169830


namespace solve_for_x_l169_169579

theorem solve_for_x (x : ℝ) (h₁ : x ≠ -3) :
  (7 * x^2 - 3) / (x + 3) - 3 / (x + 3) = 1 / (x + 3) ↔ x = 1 ∨ x = -1 := 
sorry

end solve_for_x_l169_169579


namespace final_state_probability_l169_169538

-- Initial conditions
def initial_state := (3, 1, 1)

-- Definition of a state transition given the new rule
structure state :=
  (raashan : ℕ)
  (sylvia : ℕ)
  (ted : ℕ)

def transition (s : state) : state :=
  sorry -- Placeholder for the transition function

-- Define the event of interest
def exact_state (s : state) : Prop :=
  s.raashan = 2 ∧ s.sylvia = 2 ∧ s.ted = 2

-- Define the probability calculation after 2023 transitions
noncomputable def probability_final_state : ℚ :=
  sorry -- Placeholder for the probabilistic calculation

-- The theorem to prove
theorem final_state_probability : probability_final_state = 1 / 6 :=
  sorry

end final_state_probability_l169_169538


namespace floor_sqrt_120_eq_10_l169_169908

theorem floor_sqrt_120_eq_10 :
  (√120).to_floor = 10 := by
  have h1 : √100 = 10 := by norm_num
  have h2 : √121 = 11 := by norm_num
  have h : 100 < 120 ∧ 120 < 121 := by norm_num
  have sqrt_120 : 10 < √120 ∧ √120 < 11 :=
    by exact ⟨real.sqrt_lt' 120 121 h.2, real.sqrt_lt'' 100 120 h.1⟩
  sorry

end floor_sqrt_120_eq_10_l169_169908


namespace both_students_given_correct_l169_169460

open ProbabilityTheory

variables (P_A P_B : ℝ)

-- Define the conditions from part a)
def student_a_correct := P_A = 3 / 5
def student_b_correct := P_B = 1 / 3

-- Define the event that both students correctly answer
def both_students_correct := P_A * P_B

-- Define the event that the question is answered correctly
def question_answered_correctly := (P_A * (1 - P_B)) + ((1 - P_A) * P_B) + (P_A * P_B)

-- Define the conditional probability we need to prove
theorem both_students_given_correct (hA : student_a_correct P_A) (hB : student_b_correct P_B) :
  both_students_correct P_A P_B / question_answered_correctly P_A P_B = 3 / 11 := 
sorry

end both_students_given_correct_l169_169460


namespace no_solution_fractional_eq_l169_169432

   theorem no_solution_fractional_eq : 
     ¬ ∃ x : ℚ, (3 * x / (x - 5) + 15 / (5 - x) = 1) := by
     sorry
   
end no_solution_fractional_eq_l169_169432


namespace largest_n_for_triangle_property_l169_169767

-- Define the triangle property for a set
def triangle_property (S : Set ℕ) : Prop :=
  ∀ {a b c : ℕ}, a ∈ S → b ∈ S → c ∈ S → a < b → b < c → a + b > c

-- Define the smallest subset that violates the triangle property
def violating_subset : Set ℕ := {5, 6, 11, 17, 28, 45, 73, 118, 191, 309}

-- Define the set of consecutive integers from 5 to n
def consecutive_integers (n : ℕ) : Set ℕ := {x : ℕ | 5 ≤ x ∧ x ≤ n}

-- The theorem we want to prove
theorem largest_n_for_triangle_property : ∀ (S : Set ℕ), S = consecutive_integers 308 → triangle_property S := sorry

end largest_n_for_triangle_property_l169_169767


namespace zoe_takes_correct_amount_of_money_l169_169869

def numberOfPeople : ℕ := 6
def costPerSoda : ℝ := 0.5
def costPerPizza : ℝ := 1.0

def totalCost : ℝ := (numberOfPeople * costPerSoda) + (numberOfPeople * costPerPizza)

theorem zoe_takes_correct_amount_of_money : totalCost = 9 := sorry

end zoe_takes_correct_amount_of_money_l169_169869


namespace range_of_b_l169_169182

theorem range_of_b (b : ℝ) : (∃ x : ℝ, |x - 2| + |x - 5| < b) → b > 3 :=
by 
-- This is where the proof would go.
sorry

end range_of_b_l169_169182


namespace fox_initial_coins_l169_169595

theorem fox_initial_coins :
  ∃ x : ℤ, x - 10 = 0 ∧ 2 * (x - 10) - 50 = 0 ∧ 2 * (2 * (x - 10) - 50) - 50 = 0 ∧
  2 * (2 * (2 * (x - 10) - 50) - 50) - 50 = 0 ∧ 2 * (2 * (2 * (2 * (x - 10) - 50) - 50) - 50) - 50 = 0 ∧
  x = 56 := 
by
  -- we skip the proof here
  sorry

end fox_initial_coins_l169_169595


namespace probability_particle_at_23_l169_169148

noncomputable def probability_at_point : ℚ :=
  let prob_move := (1 / 2 : ℚ)
  let num_moves := 5
  let num_rights := 2
  ∑ k in finset.range (num_moves + 1), if k = num_rights then (nat.choose num_moves k) * prob_move ^ num_moves else 0

theorem probability_particle_at_23 :
  probability_at_point = (nat.choose 5 2) * (1/2 : ℚ) ^ 5 :=
by
  sorry

end probability_particle_at_23_l169_169148


namespace union_is_real_l169_169936

-- Definitions of sets A and B
def setA : Set ℝ := {x | x^2 - x - 2 ≥ 0}
def setB : Set ℝ := {x | x > -1}

-- Theorem to prove
theorem union_is_real :
  setA ∪ setB = Set.univ :=
by
  sorry

end union_is_real_l169_169936


namespace problem_proof_l169_169801

-- Formalizing the conditions of the problem
variable {a : ℕ → ℝ}  -- Define the arithmetic sequence
variable (d : ℝ)      -- Common difference of the arithmetic sequence
variable (a₅ a₆ a₇ : ℝ)  -- Specific terms in the sequence

-- The condition given in the problem
axiom cond1 : a 5 + a 6 + a 7 = 15

-- A definition for an arithmetic sequence
noncomputable def is_arithmetic_seq (a : ℕ → ℝ) (d : ℝ) : Prop :=
∀ n : ℕ, a (n + 1) = a n + d

-- Using the axiom to deduce that a₆ = 5
axiom prop_arithmetic : is_arithmetic_seq a d

-- We want to prove that sum of terms from a₃ to a₉ = 35
theorem problem_proof : a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 = 35 :=
by sorry

end problem_proof_l169_169801


namespace least_N_bench_sections_l169_169154

-- First, define the problem conditions
def bench_capacity_adult (N : ℕ) : ℕ := 7 * N
def bench_capacity_child (N : ℕ) : ℕ := 11 * N

-- Define the problem statement to be proven
theorem least_N_bench_sections :
  ∃ N : ℕ, (N > 0) ∧ (bench_capacity_adult N = bench_capacity_child N → N = 77) :=
sorry

end least_N_bench_sections_l169_169154


namespace center_of_circle_tangent_to_parallel_lines_l169_169137

-- Define the line equations
def line1 (x y : ℝ) : Prop := 3 * x - 4 * y = 40
def line2 (x y : ℝ) : Prop := 3 * x - 4 * y = -20
def line3 (x y : ℝ) : Prop := x - 2 * y = 0

-- The proof problem
theorem center_of_circle_tangent_to_parallel_lines
  (x y : ℝ)
  (h1 : line1 x y → false)
  (h2 : line2 x y → false)
  (h3 : line3 x y) :
  x = 10 ∧ y = 5 := by
  sorry

end center_of_circle_tangent_to_parallel_lines_l169_169137


namespace greatest_value_of_x_l169_169505

theorem greatest_value_of_x (x : ℝ) : 
  (∃ (M : ℝ), (∀ y : ℝ, (y ^ 2 - 14 * y + 45 <= 0) → y <= M) ∧ (M ^ 2 - 14 * M + 45 <= 0)) ↔ M = 9 :=
by
  sorry

end greatest_value_of_x_l169_169505


namespace bike_ride_time_l169_169757

theorem bike_ride_time (y : ℚ) : 
  let speed_fast := 25
  let speed_slow := 10
  let total_distance := 170
  let total_time := 10
  (speed_fast * y + speed_slow * (total_time - y) = total_distance) 
  → y = 14 / 3 := 
by 
  sorry

end bike_ride_time_l169_169757


namespace conic_sections_ab_value_l169_169897

theorem conic_sections_ab_value
  (a b : ℝ)
  (h1 : b^2 - a^2 = 25)
  (h2 : a^2 + b^2 = 64) :
  |a * b| = Real.sqrt 868.5 :=
by
  -- Proof will be filled in later
  sorry

end conic_sections_ab_value_l169_169897


namespace find_sum_of_coordinates_of_other_endpoint_l169_169442

theorem find_sum_of_coordinates_of_other_endpoint :
  ∃ (x y : ℤ), (7, -5) = (10 + x / 2, 4 + y / 2) ∧ x + y = -10 :=
by
  sorry

end find_sum_of_coordinates_of_other_endpoint_l169_169442


namespace store_loss_90_l169_169882

theorem store_loss_90 (x y : ℝ) (h1 : x * (1 + 0.12) = 3080) (h2 : y * (1 - 0.12) = 3080) :
  2 * 3080 - x - y = -90 :=
by
  sorry

end store_loss_90_l169_169882


namespace value_of_f_of_g_l169_169377

def f (x : ℝ) : ℝ := 2 * x + 4
def g (x : ℝ) : ℝ := x^2 - 9

theorem value_of_f_of_g : f (g 3) = 4 :=
by
  -- The proof would go here. Since we are only defining the statement, we can leave this as 'sorry'.
  sorry

end value_of_f_of_g_l169_169377


namespace avg_of_first_three_groups_prob_of_inspection_l169_169434
  
-- Define the given frequency distribution as constants
def freq_40_50 : ℝ := 0.04
def freq_50_60 : ℝ := 0.06
def freq_60_70 : ℝ := 0.22
def freq_70_80 : ℝ := 0.28
def freq_80_90 : ℝ := 0.22
def freq_90_100 : ℝ := 0.18

-- Calculate the midpoint values for the first three groups
def mid_40_50 : ℝ := 45
def mid_50_60 : ℝ := 55
def mid_60_70 : ℝ := 65

-- Define the probabilities interpreted from the distributions
def prob_poor : ℝ := freq_40_50 + freq_50_60
def prob_avg : ℝ := freq_60_70 + freq_70_80
def prob_good : ℝ := freq_80_90 + freq_90_100

-- Define the main theorem for the average score of the first three groups
theorem avg_of_first_three_groups :
  (mid_40_50 * freq_40_50 + mid_50_60 * freq_50_60 + mid_60_70 * freq_60_70) /
  (freq_40_50 + freq_50_60 + freq_60_70) = 60.625 := 
by { sorry }

-- Define the theorem for the probability of inspection
theorem prob_of_inspection :
  1 - (3 * (prob_good * prob_avg * prob_avg) + 3 * (prob_avg * prob_avg * prob_good) + (prob_good * prob_good * prob_good)) = 0.396 :=
by { sorry }

end avg_of_first_three_groups_prob_of_inspection_l169_169434


namespace power_subtraction_l169_169956

variable {a m n : ℝ}

theorem power_subtraction (hm : a^m = 8) (hn : a^n = 2) : a^(m - 3 * n) = 1 := by
  sorry

end power_subtraction_l169_169956


namespace max_third_side_l169_169095

open Real

variables {A B C : ℝ} {a b c : ℝ} 

theorem max_third_side (h : cos (4 * A) + cos (4 * B) + cos (4 * C) = 1) 
                       (ha : a = 8) (hb : b = 15) : c = 17 :=
 by
  sorry 

end max_third_side_l169_169095


namespace arithmetic_mean_of_roots_l169_169989

-- Definitions corresponding to the conditions
def quadratic_eqn (a b c : ℝ) (x : ℝ) : Prop := a * x^2 + b * x + c = 0

-- The term statement for the quadratic equation mean
theorem arithmetic_mean_of_roots : 
  ∀ (a b c : ℝ), a = 1 → b = 4 → c = 1 → (∃ (x1 x2 : ℝ), quadratic_eqn a b c x1 ∧ quadratic_eqn a b c x2 ∧ -4 / 2 = -2) :=
by
  -- skip the proof
  sorry

end arithmetic_mean_of_roots_l169_169989


namespace least_value_of_N_l169_169863

theorem least_value_of_N : ∃ (N : ℕ), (N % 6 = 5) ∧ (N % 5 = 4) ∧ (N % 4 = 3) ∧ (N % 3 = 2) ∧ (N % 2 = 1) ∧ N = 59 :=
by
  sorry

end least_value_of_N_l169_169863


namespace relationship_y1_y2_y3_l169_169010

-- Define the quadratic function
def quadratic_function (x : ℝ) : ℝ := -3 * x^2 + 2

-- Define the points and their coordinates
structure Point :=
  (x : ℝ)
  (y : ℝ)

def A : Point := ⟨-1, quadratic_function (-1)⟩
def B : Point := ⟨1, quadratic_function 1⟩
def C : Point := ⟨2, quadratic_function 2⟩

-- Prove the relationship between y1, y2, and y3
theorem relationship_y1_y2_y3 :
  A.y = B.y ∧ A.y > C.y :=
by
  sorry

end relationship_y1_y2_y3_l169_169010


namespace dan_total_marbles_l169_169297

theorem dan_total_marbles (violet_marbles : ℕ) (red_marbles : ℕ) (h₁ : violet_marbles = 64) (h₂ : red_marbles = 14) : violet_marbles + red_marbles = 78 :=
sorry

end dan_total_marbles_l169_169297


namespace square_section_dimensions_l169_169282

theorem square_section_dimensions (x length : ℕ) :
  (250 ≤ x^2 + x * length ∧ x^2 + x * length ≤ 300) ∧ (25 ≤ length ∧ length ≤ 30) →
  (x = 7 ∨ x = 8) :=
  by
    sorry

end square_section_dimensions_l169_169282


namespace LCM_of_fractions_l169_169858

theorem LCM_of_fractions (x : ℕ) (h : x > 0) : 
  lcm (1 / (4 * x)) (lcm (1 / (6 * x)) (1 / (9 * x))) = 1 / (36 * x) :=
by
  sorry

end LCM_of_fractions_l169_169858


namespace ratio_equality_l169_169950

theorem ratio_equality (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z)
  (h4 : x ≠ y) (h5 : y ≠ z) (h6 : z ≠ x)
  (h7 : (y + 1) / (x + z) = (x + y + 2) / (z + 1))
  (h8 : (x + 1) / y = (y + 1) / (x + z)) :
  (x + 1) / y = 1 :=
by
  sorry

end ratio_equality_l169_169950


namespace first_term_geometric_sequence_l169_169714

theorem first_term_geometric_sequence (a5 a6 : ℚ) (h1 : a5 = 48) (h2 : a6 = 64) : 
  ∃ a : ℚ, a = 243 / 16 :=
by
  sorry

end first_term_geometric_sequence_l169_169714


namespace correct_statement_d_l169_169471

theorem correct_statement_d : 
  (∃ x : ℝ, 2^x < x^2) ↔ ¬(∀ x : ℝ, 2^x ≥ x^2) :=
by
  sorry

end correct_statement_d_l169_169471


namespace fraction_inequality_l169_169532

theorem fraction_inequality (a b m : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : m > 0) : 
  (b / a) < (b + m) / (a + m) := 
sorry

end fraction_inequality_l169_169532


namespace marie_lost_erasers_l169_169072

def initialErasers : ℕ := 95
def finalErasers : ℕ := 53

theorem marie_lost_erasers : initialErasers - finalErasers = 42 := by
  sorry

end marie_lost_erasers_l169_169072


namespace intersection_with_y_axis_l169_169992

theorem intersection_with_y_axis :
  ∃ (y : ℝ), (y = -x^2 + 3*x - 4) ∧ (x = 0) ∧ (y = -4) := 
by
  sorry

end intersection_with_y_axis_l169_169992


namespace rational_mul_example_l169_169283

theorem rational_mul_example : ((19 + 15 / 16) * (-8)) = (-159 - 1 / 2) :=
by
  sorry

end rational_mul_example_l169_169283


namespace vector_equation_l169_169396

noncomputable def vec_a : (ℝ × ℝ) := (1, -1)
noncomputable def vec_b : (ℝ × ℝ) := (2, 1)
noncomputable def vec_c : (ℝ × ℝ) := (-2, 1)

theorem vector_equation (x y : ℝ) 
  (h : vec_c = (x * vec_a.1 + y * vec_b.1, x * vec_a.2 + y * vec_b.2)) : 
  x - y = -1 := 
by { sorry }

end vector_equation_l169_169396


namespace no_nat_m_n_eq_m_squared_eq_n_squared_plus_2014_l169_169318

theorem no_nat_m_n_eq_m_squared_eq_n_squared_plus_2014 :
  ¬ ∃ (m n : ℕ), m^2 = n^2 + 2014 :=
by 
  sorry

end no_nat_m_n_eq_m_squared_eq_n_squared_plus_2014_l169_169318


namespace compute_inverse_10_mod_1729_l169_169165

def inverse_of_10_mod_1729 : ℕ :=
  1537

theorem compute_inverse_10_mod_1729 :
  (10 * inverse_of_10_mod_1729) % 1729 = 1 :=
by
  sorry

end compute_inverse_10_mod_1729_l169_169165


namespace sin_14pi_div_3_eq_sqrt3_div_2_l169_169914

theorem sin_14pi_div_3_eq_sqrt3_div_2 : Real.sin (14 * Real.pi / 3) = Real.sqrt 3 / 2 := by
  sorry

end sin_14pi_div_3_eq_sqrt3_div_2_l169_169914


namespace faye_age_l169_169503

def ages (C D E F : ℕ) :=
  D = E - 2 ∧
  E = C + 3 ∧
  F = C + 4 ∧
  D = 15

theorem faye_age (C D E F : ℕ) (h : ages C D E F) : F = 18 :=
by
  unfold ages at h
  sorry

end faye_age_l169_169503


namespace range_of_m_l169_169393

noncomputable def inequality_has_solutions (x m : ℝ) :=
  |x + 2| - |x + 3| > m

theorem range_of_m (m : ℝ) :
  (∃ x : ℝ, inequality_has_solutions x m) → m < 1 :=
by
  sorry

end range_of_m_l169_169393


namespace average_temperature_l169_169114

theorem average_temperature (t1 t2 t3 : ℤ) (h1 : t1 = -14) (h2 : t2 = -8) (h3 : t3 = 1) :
  (t1 + t2 + t3) / 3 = -7 :=
by
  sorry

end average_temperature_l169_169114


namespace non_zero_real_solution_l169_169862

theorem non_zero_real_solution (x : ℝ) (hx : x ≠ 0) (h : (3 * x)^5 = (9 * x)^4) : x = 27 :=
sorry

end non_zero_real_solution_l169_169862


namespace initial_workers_l169_169888

theorem initial_workers (M : ℝ) :
  let totalLength : ℝ := 15
  let totalDays : ℝ := 300
  let completedLength : ℝ := 2.5
  let completedDays : ℝ := 100
  let remainingLength : ℝ := totalLength - completedLength
  let remainingDays : ℝ := totalDays - completedDays
  let extraMen : ℝ := 60
  let rateWithM : ℝ := completedLength / completedDays
  let newRate : ℝ := remainingLength / remainingDays
  let newM : ℝ := M + extraMen
  (rateWithM * M = newRate * newM) → M = 100 :=
by
  intros h
  sorry

end initial_workers_l169_169888


namespace suff_but_not_nec_l169_169516

-- Definition of proposition p
def p (m : ℝ) : Prop := m = -1

-- Definition of proposition q
def q (m : ℝ) : Prop := 
  let line1 := fun (x y : ℝ) => x - y = 0
  let line2 := fun (x y : ℝ) => x + (m^2) * y = 0
  ∀ (x1 y1 x2 y2 : ℝ), line1 x1 y1 → line2 x2 y2 → (x1 = x2 → y1 = -y2)

-- The proof problem
theorem suff_but_not_nec (m : ℝ) : p m → q m ∧ (q m → m = -1 ∨ m = 1) :=
sorry

end suff_but_not_nec_l169_169516


namespace no_naturals_satisfy_m_squared_eq_n_squared_plus_2014_l169_169325

theorem no_naturals_satisfy_m_squared_eq_n_squared_plus_2014 :
  ∀ (m n : ℕ), ¬ (m^2 = n^2 + 2014) :=
by
  intro m n
  sorry

end no_naturals_satisfy_m_squared_eq_n_squared_plus_2014_l169_169325


namespace values_of_y_satisfy_quadratic_l169_169171

theorem values_of_y_satisfy_quadratic :
  (∃ (x y : ℝ), 3 * x^2 + 4 * x + 7 * y + 2 = 0 ∧ 3 * x + 2 * y + 4 = 0) →
  (∃ (y : ℝ), 4 * y^2 + 29 * y + 6 = 0) :=
by sorry

end values_of_y_satisfy_quadratic_l169_169171


namespace problem_1_problem_2_l169_169944

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^2 / 6 + 1 / x - a * Real.log x

theorem problem_1 (a : ℝ) (h : 0 < a) :
  (∀ x : ℝ, 0 < x ∧ x ≤ 3 → f x a ≤ f 3 a) → a ≥ 8 / 3 :=
sorry

theorem problem_2 (a : ℝ) (h1 : 0 < a) (x0 : ℝ) :
  (∃! t : ℝ, 0 < t ∧ f t a = 0) → Real.log x0 = (x0^3 + 6) / (2 * (x0^3 - 3)) :=
sorry

end problem_1_problem_2_l169_169944


namespace range_of_m_is_leq_3_l169_169791

noncomputable def is_range_of_m (m : ℝ) : Prop :=
  ∀ x : ℝ, 5^x + 3 > m

theorem range_of_m_is_leq_3 (m : ℝ) : is_range_of_m m ↔ m ≤ 3 :=
by
  sorry

end range_of_m_is_leq_3_l169_169791


namespace right_triangle_inequality_l169_169829

variable (a b c : ℝ)

theorem right_triangle_inequality
  (h1 : b < a) -- shorter leg is less than longer leg
  (h2 : c = Real.sqrt (a^2 + b^2)) -- hypotenuse from Pythagorean theorem
  : a + b / 2 > c ∧ c > (8 / 9) * (a + b / 2) := 
sorry

end right_triangle_inequality_l169_169829


namespace find_x_value_l169_169966

theorem find_x_value (x : ℝ) (h : 3 * x + 6 * x + x + 2 * x = 360) : x = 30 :=
by sorry

end find_x_value_l169_169966


namespace quadratic_function_characterization_l169_169779

variable (f : ℝ → ℝ)

def quadratic_function_satisfies_conditions (f : ℝ → ℝ) : Prop :=
  (f 0 = 2) ∧ (∀ x, f (x + 1) - f x = 2 * x - 1)

theorem quadratic_function_characterization
  (hf : quadratic_function_satisfies_conditions f) : 
  (∀ x, f x = x^2 - 2 * x + 2) ∧ 
  (f (-1) = 5) ∧ 
  (f 1 = 1) ∧ 
  (f 2 = 2) := by
sorry

end quadratic_function_characterization_l169_169779


namespace sequence_general_term_l169_169948

theorem sequence_general_term
  (a : ℕ → ℝ)
  (h₁ : a 1 = 1 / 2)
  (h₂ : ∀ n, a (n + 1) = 3 * a n + 7) :
  ∀ n, a n = 4 * 3^(n - 1) - 7 / 2 :=
by
  sorry

end sequence_general_term_l169_169948


namespace triangle_area_of_ellipse_l169_169017

noncomputable def ellipse_area : ℝ :=
  let a : ℝ := 2
  let b : ℝ := sqrt 3
  let c : ℝ := sqrt (a * a - b * b)
  let F1F2 : ℝ := 2 * c
  let angle : ℝ := π / 3 -- 60 degrees in radians
  let PF1_length : ℝ := 4
  let PF1_PF2_length : ℝ := PF1_length / 2 * 2
  let sin_angle : ℝ := real.sin angle in
  1 / 2 * PF1_length * PF1_length / 2 * sin_angle

theorem triangle_area_of_ellipse :
  ∀ F1 F2 P : EuclideanGeometry.Point 2,
    ∃ a b : ℝ, 
    a = 2 ∧ b = sqrt 3 ∧ 
    c = sqrt (a * a - b * b) ∧
    Ellipse ((0, 0), 4, 3) P ∧
    F1F2 = 2 * c ∧ 
    angle = π / 3 ->
    area_of_triangle F1 F2 P = sqrt 3 :=
by
  sorry

end triangle_area_of_ellipse_l169_169017


namespace quadratic_real_roots_range_l169_169785

theorem quadratic_real_roots_range (a : ℝ) : 
  (∃ x : ℝ, (a - 1) * x^2 - 2 * x + 1 = 0) ↔ (a ≤ 2) :=
by
-- Proof outline:
-- Case 1: when a = 1, the equation simplifies to -2x + 1 = 0, which has a real solution x = 1/2.
-- Case 2: when a ≠ 1, the quadratic equation has real roots if the discriminant 8 - 4a ≥ 0, i.e., 2 ≥ a.
sorry

end quadratic_real_roots_range_l169_169785


namespace tan_fraction_identity_l169_169374

theorem tan_fraction_identity (x : ℝ) 
  (h : Real.tan (x + Real.pi / 4) = 2) : 
  Real.tan x / Real.tan (2 * x) = 4 / 9 := 
by 
  sorry

end tan_fraction_identity_l169_169374


namespace largest_number_is_sqrt_7_l169_169593

noncomputable def largest_root (d e f : ℝ) : ℝ :=
if d ≥ e ∧ d ≥ f then d else if e ≥ d ∧ e ≥ f then e else f

theorem largest_number_is_sqrt_7 :
  ∃ (d e f : ℝ), (d + e + f = 3) ∧ (d * e + d * f + e * f = -14) ∧ (d * e * f = 21) ∧ (largest_root d e f = Real.sqrt 7) :=
sorry

end largest_number_is_sqrt_7_l169_169593


namespace every_positive_integer_displayable_l169_169480

-- Definitions based on the conditions of the problem
def flip_switch_up (n : ℕ) : ℕ := n + 1
def flip_switch_down (n : ℕ) : ℕ := n - 1
def press_red_button (n : ℕ) : ℕ := n * 3
def press_yellow_button (n : ℕ) : ℕ := if n % 3 = 0 then n / 3 else n
def press_green_button (n : ℕ) : ℕ := n * 5
def press_blue_button (n : ℕ) : ℕ := if n % 5 = 0 then n / 5 else n

-- Prove that every positive integer can appear on the calculator display
theorem every_positive_integer_displayable : ∀ n : ℕ, n > 0 → 
  ∃ m : ℕ, m = n ∧
    (m = flip_switch_up m ∨ m = flip_switch_down m ∨ 
     m = press_red_button m ∨ m = press_yellow_button m ∨ 
     m = press_green_button m ∨ m = press_blue_button m) := 
sorry

end every_positive_integer_displayable_l169_169480


namespace no_solution_exists_l169_169300

theorem no_solution_exists (m n : ℕ) : ¬ (m^2 = n^2 + 2014) :=
by
  sorry

end no_solution_exists_l169_169300


namespace no_nat_solutions_m2_eq_n2_plus_2014_l169_169308

theorem no_nat_solutions_m2_eq_n2_plus_2014 :
  ¬ ∃ (m n : ℕ), m^2 = n^2 + 2014 :=
by
  sorry

end no_nat_solutions_m2_eq_n2_plus_2014_l169_169308


namespace wendy_tooth_extraction_cost_eq_290_l169_169598

def dentist_cleaning_cost : ℕ := 70
def dentist_filling_cost : ℕ := 120
def wendy_dentist_bill : ℕ := 5 * dentist_filling_cost
def wendy_cleaning_and_fillings_cost : ℕ := dentist_cleaning_cost + 2 * dentist_filling_cost
def wendy_tooth_extraction_cost : ℕ := wendy_dentist_bill - wendy_cleaning_and_fillings_cost

theorem wendy_tooth_extraction_cost_eq_290 : wendy_tooth_extraction_cost = 290 := by
  sorry

end wendy_tooth_extraction_cost_eq_290_l169_169598


namespace initial_money_is_10_l169_169225

-- Definition for the initial amount of money
def initial_money (X : ℝ) : Prop :=
  let spent_on_cupcakes := (1 / 5) * X
  let remaining_after_cupcakes := X - spent_on_cupcakes
  let spent_on_milkshake := 5
  let remaining_after_milkshake := remaining_after_cupcakes - spent_on_milkshake
  remaining_after_milkshake = 3

-- The statement proving that Ivan initially had $10
theorem initial_money_is_10 (X : ℝ) (h : initial_money X) : X = 10 :=
by sorry

end initial_money_is_10_l169_169225


namespace equilateral_triangle_fixed_area_equilateral_triangle_max_area_l169_169215

theorem equilateral_triangle_fixed_area (a b c : ℝ) (Δ : ℝ) (s : ℝ) (R : ℝ) (is_triangle : a + b > c ∧ a + c > b ∧ b + c > a) :
  (s = (a + b + c) / 2 ∧ Δ = Real.sqrt (s * (s - a) * (s - b) * (s - c))) →
  (a * b * c = minimized ∨ a + b + c = minimized ∨ a^2 + b^2 + c^2 = minimized ∨ R = minimized) →
    (a = b ∧ b = c) :=
by
  sorry

theorem equilateral_triangle_max_area (a b c : ℝ) (Δ : ℝ) (s : ℝ) (R : ℝ) (is_triangle : a + b > c ∧ a + c > b ∧ b + c > a) :
  (s = (a + b + c) / 2 ∧ Δ = Real.sqrt (s * (s - a) * (s - b) * (s - c))) →
  (a * b * c = fixed ∨ a + b + c = fixed ∨ a^2 + b^2 + c^2 = fixed ∨ R = fixed) →
  (Δ = maximized) →
    (a = b ∧ b = c) :=
by
  sorry

end equilateral_triangle_fixed_area_equilateral_triangle_max_area_l169_169215


namespace part_a_l169_169272

theorem part_a (a b c : ℝ) : 
  (∀ n : ℝ, (n + 2)^2 = a * (n + 1)^2 + b * n^2 + c * (n - 1)^2) ↔ (a = 3 ∧ b = -3 ∧ c = 1) :=
by 
  sorry

end part_a_l169_169272


namespace solve_quadratic_equation_l169_169094

theorem solve_quadratic_equation (x : ℝ) : x^2 - 4*x + 3 = 0 ↔ (x = 1 ∨ x = 3) := 
by 
  sorry

end solve_quadratic_equation_l169_169094


namespace find_rate_per_kg_mangoes_l169_169634

-- Definitions based on the conditions
def rate_per_kg_grapes : ℕ := 70
def quantity_grapes : ℕ := 8
def total_payment : ℕ := 1000
def quantity_mangoes : ℕ := 8

-- Proposition stating what we want to prove
theorem find_rate_per_kg_mangoes (r : ℕ) (H : total_payment = (rate_per_kg_grapes * quantity_grapes) + (r * quantity_mangoes)) : r = 55 := sorry

end find_rate_per_kg_mangoes_l169_169634


namespace total_players_l169_169055

def num_teams : Nat := 35
def players_per_team : Nat := 23

theorem total_players :
  num_teams * players_per_team = 805 :=
by
  sorry

end total_players_l169_169055


namespace tutors_work_together_again_in_360_days_l169_169173

theorem tutors_work_together_again_in_360_days :
  Nat.lcm (Nat.lcm (Nat.lcm 5 6) 8) 9 = 360 :=
by
  sorry

end tutors_work_together_again_in_360_days_l169_169173


namespace prime_divisor_form_l169_169704

theorem prime_divisor_form (n : ℕ) (q : ℕ) (hq : (2^(2^n) + 1) % q = 0) (prime_q : Nat.Prime q) :
  ∃ k : ℕ, q = 2^(n+1) * k + 1 :=
sorry

end prime_divisor_form_l169_169704


namespace miles_driven_before_gas_stop_l169_169076

def total_distance : ℕ := 78
def distance_left : ℕ := 46

theorem miles_driven_before_gas_stop : total_distance - distance_left = 32 := by
  sorry

end miles_driven_before_gas_stop_l169_169076


namespace smallest_sum_l169_169383

theorem smallest_sum (x y : ℕ) (hx : x > 0) (hy : y > 0) (hxy : x ≠ y) 
  (h : (1/x + 1/y = 1/10)) : x + y = 49 := 
sorry

end smallest_sum_l169_169383


namespace plants_per_row_l169_169874

theorem plants_per_row (P : ℕ) (rows : ℕ) (yield_per_plant : ℕ) (total_yield : ℕ) 
  (h1 : rows = 30)
  (h2 : yield_per_plant = 20)
  (h3 : total_yield = 6000)
  (h4 : rows * yield_per_plant * P = total_yield) : 
  P = 10 :=
by 
  sorry

end plants_per_row_l169_169874


namespace man_completes_in_9_days_l169_169877

-- Definitions of the work rates and the conditions given
def M : ℚ := sorry
def W : ℚ := 1 / 6
def B : ℚ := 1 / 18
def combined_rate : ℚ := 1 / 3

-- Statement that the man alone can complete the work in 9 days
theorem man_completes_in_9_days
  (h_combined : M + W + B = combined_rate) : 1 / M = 9 :=
  sorry

end man_completes_in_9_days_l169_169877


namespace quadratic_inequality_solution_set_l169_169120

theorem quadratic_inequality_solution_set (a : ℝ) (h : a < -2) : 
  { x : ℝ | ax^2 + (a - 2)*x - 2 ≥ 0 } = { x : ℝ | -1 ≤ x ∧ x ≤ 2/a } := 
by
  sorry

end quadratic_inequality_solution_set_l169_169120


namespace parallel_line_slope_l169_169601

theorem parallel_line_slope (x y : ℝ) :
  (∃ k b : ℝ, 3 * x + 6 * y = k * x + b) ∧ (∃ a b, y = a * x + b) ∧ 3 * x + 6 * y = -24 → 
  ∃ m : ℝ, m = -1/2 :=
by
  sorry

end parallel_line_slope_l169_169601


namespace minimum_photos_l169_169053

variable (Tourist : Type) (Monument : Type) (TakePhoto : Tourist → Monument → Prop)

theorem minimum_photos (h_tourists : ∀ t1 t2 : Tourist, t1 ≠ t2 → ∃ m1 m2 m3 : Monument, TakePhoto t1 m1 ∧ TakePhoto t1 m2 ∧ TakePhoto t2 m3 ∧ 
                       ¬ ((TakePhoto t1 m3 ∧ TakePhoto t2 m1) ∨ (TakePhoto t1 m3 ∧ TakePhoto t2 m2) ∨ (TakePhoto t1 m1 ∧ TakePhoto t2 m2))) 
                       (h_total_tourists : ∃ S : set Tourist, S.card = 42) : 
  ∃ n : ℕ, n = 123 ∧ ∀ t ∈ h_total_tourists.1, ∃ (m1 m2 m3 : Monument), (TakePhoto t m1 ∨ TakePhoto t m2 ∨ TakePhoto t m3) :=
sorry

end minimum_photos_l169_169053


namespace no_nat_m_n_eq_m_squared_eq_n_squared_plus_2014_l169_169316

theorem no_nat_m_n_eq_m_squared_eq_n_squared_plus_2014 :
  ¬ ∃ (m n : ℕ), m^2 = n^2 + 2014 :=
by 
  sorry

end no_nat_m_n_eq_m_squared_eq_n_squared_plus_2014_l169_169316


namespace anthony_pencils_total_l169_169158

def pencils_initial : Nat := 9
def pencils_kathryn : Nat := 56
def pencils_greg : Nat := 84
def pencils_maria : Nat := 138

theorem anthony_pencils_total : 
  pencils_initial + pencils_kathryn + pencils_greg + pencils_maria = 287 := 
by
  sorry

end anthony_pencils_total_l169_169158


namespace minimum_photos_l169_169054

variable (Tourist : Type) (Monument : Type) (TakePhoto : Tourist → Monument → Prop)

theorem minimum_photos (h_tourists : ∀ t1 t2 : Tourist, t1 ≠ t2 → ∃ m1 m2 m3 : Monument, TakePhoto t1 m1 ∧ TakePhoto t1 m2 ∧ TakePhoto t2 m3 ∧ 
                       ¬ ((TakePhoto t1 m3 ∧ TakePhoto t2 m1) ∨ (TakePhoto t1 m3 ∧ TakePhoto t2 m2) ∨ (TakePhoto t1 m1 ∧ TakePhoto t2 m2))) 
                       (h_total_tourists : ∃ S : set Tourist, S.card = 42) : 
  ∃ n : ℕ, n = 123 ∧ ∀ t ∈ h_total_tourists.1, ∃ (m1 m2 m3 : Monument), (TakePhoto t m1 ∨ TakePhoto t m2 ∨ TakePhoto t m3) :=
sorry

end minimum_photos_l169_169054


namespace decimal_to_binary_51_l169_169898

theorem decimal_to_binary_51 : (51 : ℕ) = 0b110011 := by sorry

end decimal_to_binary_51_l169_169898


namespace negation_is_false_l169_169439

-- Define even numbers
def even (n : ℕ) : Prop := ∃ k, n = 2 * k

-- Define the original proposition P
def P (a b : ℕ) : Prop := even a ∧ even b → even (a + b)

-- The negation of the proposition P
def notP (a b : ℕ) : Prop := ¬(even a ∧ even b → even (a + b))

-- The theorem to prove
theorem negation_is_false : ∀ a b : ℕ, ¬notP a b :=
by
  sorry

end negation_is_false_l169_169439


namespace find_x_l169_169664

variable (x : ℝ)

def length := 4 * x
def width := x + 3

def area := length x * width x
def perimeter := 2 * length x + 2 * width x

theorem find_x (h : area x = 3 * perimeter x) : x = 5.342 := by
  sorry

end find_x_l169_169664


namespace slope_range_l169_169517

variables (x y k : ℝ)

theorem slope_range :
  (2 ≤ x ∧ x ≤ 3) ∧ (y = -2 * x + 8) ∧ (k = -3 * y / (2 * x)) →
  -3 ≤ k ∧ k ≤ -1 :=
by
  sorry

end slope_range_l169_169517


namespace simplify_and_evaluate_l169_169090

theorem simplify_and_evaluate : 
    ∀ (a b : ℤ), a = 1 → b = -1 → 
    ((2 * a^2 * b - 2 * a * b^2 - b^3) / b - (a + b) * (a - b) = 3) := 
by
  intros a b ha hb
  sorry

end simplify_and_evaluate_l169_169090


namespace compute_expression_l169_169825

noncomputable def roots_exist (P : Polynomial ℝ) (α β γ : ℝ) : Prop :=
  P = Polynomial.C (-13) + Polynomial.X * (Polynomial.C 11 + Polynomial.X * (Polynomial.C (-7) + Polynomial.X))

theorem compute_expression (α β γ : ℝ) (h : roots_exist (Polynomial.X^3 - 7 * Polynomial.X^2 + 11 * Polynomial.X - 13) α β γ) :
  (α ≠ 0) → (β ≠ 0) → (γ ≠ 0) → (α^2 * β^2 + β^2 * γ^2 + γ^2 * α^2 = -61) :=
  sorry

end compute_expression_l169_169825


namespace no_solution_exists_l169_169302

theorem no_solution_exists (m n : ℕ) : ¬ (m^2 = n^2 + 2014) :=
by
  sorry

end no_solution_exists_l169_169302


namespace larger_of_two_numbers_l169_169854

theorem larger_of_two_numbers (x y : ℝ) (h1 : x + y = 50) (h2 : x - y = 8) : max x y = 29 :=
by
  sorry

end larger_of_two_numbers_l169_169854


namespace probability_both_correct_given_any_correct_l169_169462

-- Defining the probabilities
def P_A : ℚ := 3 / 5
def P_B : ℚ := 1 / 3

-- Defining the events and their products
def P_AnotB : ℚ := P_A * (1 - P_B)
def P_notAB : ℚ := (1 - P_A) * P_B
def P_AB : ℚ := P_A * P_B

-- Calculated Probability of C
def P_C : ℚ := P_AnotB + P_notAB + P_AB

-- The proof statement
theorem probability_both_correct_given_any_correct : (P_AB / P_C) = 3 / 11 :=
by
  sorry

end probability_both_correct_given_any_correct_l169_169462


namespace no_nat_numbers_m_n_satisfy_eq_l169_169344

theorem no_nat_numbers_m_n_satisfy_eq (m n : ℕ) : ¬ (m^2 = n^2 + 2014) := sorry

end no_nat_numbers_m_n_satisfy_eq_l169_169344


namespace triangle_area_proof_l169_169826

def vector2 := ℝ × ℝ

def a : vector2 := (6, 3)
def b : vector2 := (-4, 5)

noncomputable def det (u v : vector2) : ℝ := u.1 * v.2 - u.2 * v.1

noncomputable def parallelogram_area (u v : vector2) : ℝ := |det u v|

noncomputable def triangle_area (u v : vector2) : ℝ := parallelogram_area u v / 2

theorem triangle_area_proof : triangle_area a b = 21 := 
by 
  sorry

end triangle_area_proof_l169_169826


namespace trig_solution_l169_169242

noncomputable def solve_trig_system (x y : ℝ) : Prop :=
  (3 * Real.cos x + 4 * Real.sin x = -1.4) ∧ 
  (13 * Real.cos x - 41 * Real.cos y = -45) ∧ 
  (13 * Real.sin x + 41 * Real.sin y = 3)

theorem trig_solution :
  solve_trig_system (112.64 * Real.pi / 180) (347.32 * Real.pi / 180) ∧ 
  solve_trig_system (239.75 * Real.pi / 180) (20.31 * Real.pi / 180) :=
by {
    repeat { sorry }
  }

end trig_solution_l169_169242


namespace sum_of_coefficients_polynomial_expansion_l169_169210

theorem sum_of_coefficients_polynomial_expansion :
  let polynomial := (2 * (1 : ℤ) + 3)^5
  ∃ b_5 b_4 b_3 b_2 b_1 b_0 : ℤ,
  polynomial = b_5 * 1^5 + b_4 * 1^4 + b_3 * 1^3 + b_2 * 1^2 + b_1 * 1 + b_0 ∧
  (b_5 + b_4 + b_3 + b_2 + b_1 + b_0) = 3125 :=
by
  sorry

end sum_of_coefficients_polynomial_expansion_l169_169210


namespace x_plus_y_eq_3012_plus_pi_div_2_l169_169938

theorem x_plus_y_eq_3012_plus_pi_div_2
  (x y : ℝ)
  (h1 : x + Real.cos y = 3012)
  (h2 : x + 3012 * Real.sin y = 3010)
  (h3 : 0 ≤ y ∧ y ≤ Real.pi) :
  x + y = 3012 + Real.pi / 2 :=
sorry

end x_plus_y_eq_3012_plus_pi_div_2_l169_169938


namespace solution_set_contains_0_and_2_l169_169803

theorem solution_set_contains_0_and_2 (k : ℝ) : 
  ∀ x, ((1 + k^2) * x ≤ k^4 + 4) → (x = 0 ∨ x = 2) :=
by {
  sorry -- Proof is omitted
}

end solution_set_contains_0_and_2_l169_169803


namespace regular_polygon_sides_160_l169_169750

theorem regular_polygon_sides_160 (n : ℕ) 
  (h1 : n ≥ 3) 
  (h2 : ∀ i, 1 ≤ i ∧ i ≤ n → (interior_angle : ℝ) = 160) : 
  n = 18 :=
by
  sorry

end regular_polygon_sides_160_l169_169750


namespace polynomial_identity_and_sum_of_squares_l169_169027

theorem polynomial_identity_and_sum_of_squares :
  ∃ (p q r s t u : ℤ), (∀ (x : ℤ), 512 * x^3 + 64 = (p * x^2 + q * x + r) * (s * x^2 + t * x + u)) ∧
    p^2 + q^2 + r^2 + s^2 + t^2 + u^2 = 5472 :=
sorry

end polynomial_identity_and_sum_of_squares_l169_169027


namespace number_of_zeros_of_g_is_4_l169_169391

noncomputable def f (x : ℝ) : ℝ := 
  if x < 0 then x + 1/x else Real.log x

noncomputable def g (x : ℝ) : ℝ := 
  f (f x + 2) + 2

theorem number_of_zeros_of_g_is_4 : 
  ∃ S : Finset ℝ, S.card = 4 ∧ ∀ x ∈ S, g x = 0 :=
sorry

end number_of_zeros_of_g_is_4_l169_169391


namespace planes_contain_at_least_three_midpoints_l169_169209

-- Define the cube structure and edge midpoints
structure Cube where
  edges : Fin 12

def midpoints (c : Cube) : Set (Fin 12) := { e | true }

-- Define the total planes considering the constraints
noncomputable def planes : ℕ := 4 + 18 + 56

-- The proof goal
theorem planes_contain_at_least_three_midpoints :
  planes = 81 := by
  sorry

end planes_contain_at_least_three_midpoints_l169_169209


namespace max_value_of_expression_l169_169420

theorem max_value_of_expression
  (x y z : ℝ)
  (h : 9 * x^2 + 4 * y^2 + 25 * z^2 = 1) :
  11 * x + 3 * y + 8 * z ≤ 3.1925 :=
sorry

end max_value_of_expression_l169_169420


namespace no_nat_solutions_m_sq_eq_n_sq_plus_2014_l169_169353

theorem no_nat_solutions_m_sq_eq_n_sq_plus_2014 :
  ¬ ∃ (m n : ℕ), m ^ 2 = n ^ 2 + 2014 := 
sorry

end no_nat_solutions_m_sq_eq_n_sq_plus_2014_l169_169353


namespace no_naturals_satisfy_m_squared_eq_n_squared_plus_2014_l169_169330

theorem no_naturals_satisfy_m_squared_eq_n_squared_plus_2014 :
  ∀ (m n : ℕ), ¬ (m^2 = n^2 + 2014) :=
by
  intro m n
  sorry

end no_naturals_satisfy_m_squared_eq_n_squared_plus_2014_l169_169330


namespace augustus_makes_3_milkshakes_l169_169493

def augMilkshakePerHour (A : ℕ) (Luna : ℕ) (hours : ℕ) (totalMilkshakes : ℕ) : Prop :=
  (A + Luna) * hours = totalMilkshakes

theorem augustus_makes_3_milkshakes :
  augMilkshakePerHour 3 7 8 80 :=
by
  -- We assume the proof here
  sorry

end augustus_makes_3_milkshakes_l169_169493


namespace no_nat_m_n_eq_m_squared_eq_n_squared_plus_2014_l169_169320

theorem no_nat_m_n_eq_m_squared_eq_n_squared_plus_2014 :
  ¬ ∃ (m n : ℕ), m^2 = n^2 + 2014 :=
by 
  sorry

end no_nat_m_n_eq_m_squared_eq_n_squared_plus_2014_l169_169320


namespace darla_total_payment_l169_169901

-- Define the cost per watt, total watts used, and late fee
def cost_per_watt : ℝ := 4
def total_watts : ℝ := 300
def late_fee : ℝ := 150

-- Define the total cost of electricity
def electricity_cost : ℝ := cost_per_watt * total_watts

-- Define the total amount Darla needs to pay
def total_amount : ℝ := electricity_cost + late_fee

-- The theorem to prove the total amount equals $1350
theorem darla_total_payment : total_amount = 1350 := by
  sorry

end darla_total_payment_l169_169901


namespace polygon_sides_l169_169752

/-- 
A regular polygon with interior angles of 160 degrees has 18 sides.
-/
theorem polygon_sides (n : ℕ) (h : ∀ (i : ℕ), i < n → (interior_angle : ℝ) = 160) : n = 18 := 
by
  have angle_sum : 180 * (n - 2) = 160 * n := 
    by sorry
  have eq_sides : n = 18 := 
    by sorry
  exact eq_sides

end polygon_sides_l169_169752


namespace color_coat_drying_time_l169_169414

theorem color_coat_drying_time : ∀ (x : ℕ), 2 + 2 * x + 5 = 13 → x = 3 :=
by
  intro x
  intro h
  sorry

end color_coat_drying_time_l169_169414


namespace bc_over_ad_eq_50_point_4_l169_169824

theorem bc_over_ad_eq_50_point_4 :
  let B := (2, 2, 5)
  let S (r : ℝ) (B : ℝ × ℝ × ℝ) := {p | dist p B ≤ r }
  let d := (20 : ℝ)
  let c := (48 : ℝ)
  let b := (28 * Real.pi : ℝ)
  let a := ((4 * Real.pi) / 3 : ℝ)
  let bc := b * c
  let ad := a * d
  bc / ad = 50.4 := by
    sorry

end bc_over_ad_eq_50_point_4_l169_169824


namespace expected_balls_in_original_pos_after_two_transpositions_l169_169091

theorem expected_balls_in_original_pos_after_two_transpositions :
  ∃ (n : ℚ), n = 3.2 := 
sorry

end expected_balls_in_original_pos_after_two_transpositions_l169_169091


namespace area_of_triangle_ADE_l169_169411

theorem area_of_triangle_ADE (A B C D E : Type) (AB BC AC : ℝ) (AD AE : ℝ)
  (h1 : AB = 8) (h2 : BC = 13) (h3 : AC = 15) (h4 : AD = 3) (h5 : AE = 11) :
  let s := (AB + BC + AC) / 2
  let area_ABC := Real.sqrt (s * (s - AB) * (s - BC) * (s - AC))
  let sinA := 2 * area_ABC / (AB * AC)
  let area_ADE := (1 / 2) * AD * AE * sinA
  area_ADE = (33 * Real.sqrt 3) / 4 :=
by 
  have s := (8 + 13 + 15) / 2
  have area_ABC := Real.sqrt (s * (s - 8) * (s - 13) * (s - 15))
  have sinA := 2 * area_ABC / (8 * 15)
  have area_ADE := (1 / 2) * 3 * 11 * sinA
  sorry

end area_of_triangle_ADE_l169_169411


namespace sum_of_natural_numbers_l169_169162

theorem sum_of_natural_numbers (n : ℕ) (h : n * (n + 1) = 812) : n = 28 := by
  sorry

end sum_of_natural_numbers_l169_169162


namespace find_ratio_l169_169660

variable {a : ℕ → ℝ}

-- Conditions
def is_geometric_sequence (a : ℕ → ℝ) : Prop := 
  ∀ n : ℕ, a n > 0 ∧ a (n+1) / a n = a 1 / a 0

def forms_arithmetic_sequence (a1 a3_half a2_times_two : ℝ) : Prop :=
  a3_half = (a1 + a2_times_two) / 2

theorem find_ratio (a : ℕ → ℝ) (h_geom : is_geometric_sequence a)
  (h_arith : forms_arithmetic_sequence (a 1) (1/2 * a 3) (2 * a 2)) :
  (a 8 + a 9) / (a 6 + a 7) = 3 + 2 * Real.sqrt 2 :=
sorry

end find_ratio_l169_169660


namespace geometric_sequence_ratio_l169_169968

variable {α : Type*} [Field α]

def geometric_sequence (a_1 q : α) (n : ℕ) : α :=
  a_1 * q ^ (n - 1)

theorem geometric_sequence_ratio (a1 q a4 a14 a5 a13 : α)
  (h_seq : ∀ n, geometric_sequence a1 q (n + 1) = a_5) 
  (h0 : geometric_sequence a1 q 5 * geometric_sequence a1 q 13 = 6) 
  (h1 : geometric_sequence a1 q 4 + geometric_sequence a1 q 14 = 5) :
  (∃ (k : α), k = 2 / 3 ∨ k = 3 / 2) → 
  geometric_sequence a1 q 80 / geometric_sequence a1 q 90 = k :=
by
  sorry

end geometric_sequence_ratio_l169_169968


namespace shadow_area_greatest_integer_l169_169281

theorem shadow_area_greatest_integer (x : ℝ)
  (h1 : ∀ (a : ℝ), a = 1)
  (h2 : ∀ (b : ℝ), b = 48)
  (h3 : ∀ (c: ℝ), x = 1 / 6):
  ⌊1000 * x⌋ = 166 := 
by sorry

end shadow_area_greatest_integer_l169_169281


namespace point_in_shaded_region_l169_169081

open Set

-- Define the circles
def Circle1 : set ℝ := {p : ℝ × ℝ | (p.1 - x1)^2 + (p.2 - y1)^2 = r1^2}
def Circle2 : set ℝ := {p : ℝ × ℝ | (p.1 - x2)^2 + (p.2 - y2)^2 = r2^2}

-- Define the external tangents to both circles
def ext_tangents (C1 C2 : set ℝ) : set (set ℝ) := sorry 

-- Define the shaded regions determined by external tangents but excluding the tangent lines themselves
def shaded_region (C1 C2 : set ℝ) : set ℝ := { p : ℝ × ℝ | ∃ l ∈ ext_tangents C1 C2, p ∉ l }

-- The point \( M \) that satisfies the given condition
def locus_of_M (C1 C2 : set ℝ) : set ℝ := shaded_region C1 C2

-- Main theorem statement
theorem point_in_shaded_region (C1 C2 : set ℝ) (C1_non_overlap_C2 : disjoint C1 C2) :
  ∀ M : ℝ × ℝ, M ∈ locus_of_M C1 C2 → 
  (∀ l : set ℝ, l ∈ {l : set ℝ | ∃ p ∈ shaded_region C1 C2, p ∉ l} → ∃ p : ℝ × ℝ, p ∈ Circle1 ∨ p ∈ Circle2) := 
sorry

end point_in_shaded_region_l169_169081


namespace determine_q_l169_169638

theorem determine_q (q : ℝ → ℝ) :
  (∀ x, q x = 0 ↔ x = 3 ∨ x = -3) ∧
  (∀ k : ℝ, k < 3) ∧ -- indicating degree considerations for asymptotes
  (q 2 = 18) →
  q = (fun x => (-18 / 5) * x ^ 2 + 162 / 5) :=
by
  sorry

end determine_q_l169_169638


namespace no_nat_solutions_for_m2_eq_n2_plus_2014_l169_169358

theorem no_nat_solutions_for_m2_eq_n2_plus_2014 :
  ∀ m n : ℕ, ¬(m^2 = n^2 + 2014) := by
sorry

end no_nat_solutions_for_m2_eq_n2_plus_2014_l169_169358


namespace tan_alpha_eq_neg2_sin2a_plus_1_over_1_plus_sin2a_plus_cos2a_eq_neg1_over_2_l169_169185

variable (α : ℝ)
variable (h : (2 * Real.sin α + 3 * Real.cos α) / (Real.sin α - 2 * Real.cos α) = 1 / 4)

theorem tan_alpha_eq_neg2 : Real.tan α = -2 :=
  sorry

theorem sin2a_plus_1_over_1_plus_sin2a_plus_cos2a_eq_neg1_over_2 :
  (Real.sin (2 * α) + 1) / (1 + Real.sin (2 * α) + Real.cos (2 * α)) = -1 / 2 :=
  sorry

end tan_alpha_eq_neg2_sin2a_plus_1_over_1_plus_sin2a_plus_cos2a_eq_neg1_over_2_l169_169185


namespace find_m_n_diff_l169_169667

theorem find_m_n_diff (a : ℝ) (n m: ℝ) (h_a_pos : 0 < a) (h_a_ne_one : a ≠ 1)
  (h_pass : a^(2 * m - 6) + n = 2) :
  m - n = 2 :=
sorry

end find_m_n_diff_l169_169667


namespace sum_of_differences_of_7_in_657932657_l169_169465

theorem sum_of_differences_of_7_in_657932657 :
  let numeral := 657932657
  let face_value (d : Nat) := d
  let local_value (d : Nat) (pos : Nat) := d * 10 ^ pos
  let indices_of_7 := [6, 0]
  let differences := indices_of_7.map (fun pos => local_value 7 pos - face_value 7)
  differences.sum = 6999993 :=
by
  sorry

end sum_of_differences_of_7_in_657932657_l169_169465


namespace initial_sentences_today_l169_169972

-- Definitions of the given conditions
def typing_rate : ℕ := 6
def initial_typing_time : ℕ := 20
def additional_typing_time : ℕ := 15
def erased_sentences : ℕ := 40
def post_meeting_typing_time : ℕ := 18
def total_sentences_end_of_day : ℕ := 536

def sentences_typed_before_break := initial_typing_time * typing_rate
def sentences_typed_after_break := additional_typing_time * typing_rate
def sentences_typed_post_meeting := post_meeting_typing_time * typing_rate
def sentences_today := sentences_typed_before_break + sentences_typed_after_break - erased_sentences + sentences_typed_post_meeting

theorem initial_sentences_today : total_sentences_end_of_day - sentences_today = 258 := by
  -- proof here
  sorry

end initial_sentences_today_l169_169972


namespace blake_initial_amount_l169_169633

noncomputable def initial_amount_given (amount: ℕ) := 3 * amount / 2

theorem blake_initial_amount (h_given_amount_b_to_c: ℕ) (c_to_b_transfer: ℕ) (h_c_to_b: c_to_b_transfer = 30000) :
  initial_amount_given c_to_b_transfer = h_given_amount_b_to_c → h_given_amount_b_to_c = 20000 :=
by
  intro h
  have calc_initial := calc 
    initial_amount_given 30000
      = 3 * 30000 / 2 : by rfl
      ... = 90000 / 2 : by norm_num
      ... = 45000 : by norm_num
  contradiction

end blake_initial_amount_l169_169633


namespace max_donation_amount_l169_169453

theorem max_donation_amount (x : ℝ) : 
  (500 * x + 1500 * (x / 2) = 0.4 * 3750000) → x = 1200 :=
by 
  sorry

end max_donation_amount_l169_169453


namespace B_time_to_complete_work_l169_169873

theorem B_time_to_complete_work :
  (A1 B1 C1 : ℚ) (h1 : A1 = 1/5) (h2 : B1 + C1 = 1/3) (h3 : A1 + C1 = 1/2) :
  1/B1 = 30 :=
by
  sorry

end B_time_to_complete_work_l169_169873


namespace ConeCannotHaveSquarePlanView_l169_169106

def PlanViewIsSquare (solid : Type) : Prop :=
  -- Placeholder to denote the property that the plan view of a solid is a square
  sorry

def IsCone (solid : Type) : Prop :=
  -- Placeholder to denote the property that the solid is a cone
  sorry

theorem ConeCannotHaveSquarePlanView (solid : Type) :
  (PlanViewIsSquare solid) → ¬ (IsCone solid) :=
sorry

end ConeCannotHaveSquarePlanView_l169_169106


namespace zoe_total_money_l169_169867

def numberOfPeople : ℕ := 6
def sodaCostPerBottle : ℝ := 0.5
def pizzaCostPerSlice : ℝ := 1.0

theorem zoe_total_money :
  numberOfPeople * sodaCostPerBottle + numberOfPeople * pizzaCostPerSlice = 9 := 
by
  sorry

end zoe_total_money_l169_169867


namespace value_of_expression_l169_169861

theorem value_of_expression : 2 - (-2 : ℝ) ^ (-2 : ℝ) = 7 / 4 := 
by 
  sorry

end value_of_expression_l169_169861


namespace percentage_of_candidates_selected_in_State_A_is_6_l169_169219

-- Definitions based on conditions
def candidates_appeared : ℕ := 8400
def candidates_selected_B : ℕ := (7 * candidates_appeared) / 100 -- 7% of 8400
def extra_candidates_selected : ℕ := 84
def candidates_selected_A : ℕ := candidates_selected_B - extra_candidates_selected

-- Definition based on the goal proof
def percentage_selected_A : ℕ := (candidates_selected_A * 100) / candidates_appeared

-- The theorem we need to prove
theorem percentage_of_candidates_selected_in_State_A_is_6 :
  percentage_selected_A = 6 :=
by
  sorry

end percentage_of_candidates_selected_in_State_A_is_6_l169_169219


namespace find_constant_term_of_polynomial_with_negative_integer_roots_l169_169695

theorem find_constant_term_of_polynomial_with_negative_integer_roots
  (p q r s : ℝ) (t1 t2 t3 t4 : ℝ)
  (h_roots : ∀ {x : ℝ}, x^4 + p*x^3 + q*x^2 + r*x + s = (x + t1)*(x + t2)*(x + t3)*(x + t4))
  (h_neg_int_roots : ∀ {i : ℕ}, i < 4 → t1 = i ∨ t2 = i ∨ t3 = i ∨ t4 = i)
  (h_sum_coeffs : p + q + r + s = 168) :
  s = 144 :=
by
  sorry

end find_constant_term_of_polynomial_with_negative_integer_roots_l169_169695


namespace larger_of_two_numbers_l169_169102

-- Define necessary conditions
def hcf : ℕ := 23
def factor1 : ℕ := 11
def factor2 : ℕ := 12
def lcm : ℕ := hcf * factor1 * factor2

-- Define the problem statement in Lean
theorem larger_of_two_numbers : ∃ (a b : ℕ), a = hcf * factor1 ∧ b = hcf * factor2 ∧ max a b = 276 := by
  sorry

end larger_of_two_numbers_l169_169102


namespace evariste_stairs_l169_169853

def num_ways (n : ℕ) : ℕ :=
  if n = 0 then 1
  else if n = 1 then 1
  else num_ways (n - 1) + num_ways (n - 2)

theorem evariste_stairs (n : ℕ) : num_ways n = u_n :=
  sorry

end evariste_stairs_l169_169853


namespace julia_account_balance_l169_169469

noncomputable def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r) ^ n

theorem julia_account_balance :
  let P := 1500
  let r := 0.04
  let n := 21
  let A := compound_interest P r n
in A ≈ 3046.28 :=
by sorry

end julia_account_balance_l169_169469


namespace no_nat_solutions_m2_eq_n2_plus_2014_l169_169313

theorem no_nat_solutions_m2_eq_n2_plus_2014 :
  ¬ ∃ (m n : ℕ), m^2 = n^2 + 2014 :=
by
  sorry

end no_nat_solutions_m2_eq_n2_plus_2014_l169_169313


namespace not_true_B_l169_169769

def star (x y : ℝ) : ℝ := x^2 - 2*x*y + y^2

theorem not_true_B (x y : ℝ) : 2 * star x y ≠ star (2 * x) (2 * y) := by
  sorry

end not_true_B_l169_169769


namespace sequence_geometric_l169_169811

theorem sequence_geometric (a : ℕ → ℝ) (r : ℝ) (h1 : ∀ n, a (n + 1) = r * a n) (h2 : a 4 = 2) : a 2 * a 6 = 4 :=
by
  sorry

end sequence_geometric_l169_169811


namespace binomial_square_l169_169918

variable (c : ℝ)

theorem binomial_square (h : ∃ a : ℝ, (x^2 - 164 * x + c) = (x + a)^2) : c = 6724 := sorry

end binomial_square_l169_169918


namespace charlie_third_week_data_l169_169571

theorem charlie_third_week_data (d3 : ℕ) : 
  let data_plan := 8
  let cost_per_GB := 10
  let extra_charge := 120
  let week1 := 2
  let week2 := 3
  let week4 := 10
  let total_extra_GB := extra_charge / cost_per_GB
  let total_data := week1 + week2 + week4 + d3
  let overage_GB := total_data - data_plan
  overage_GB = total_extra_GB -> d3 = 5 := 
by
  let data_plan := 8
  let cost_per_GB := 10
  let extra_charge := 120
  let week1 := 2
  let week2 := 3
  let week4 := 10
  let total_extra_GB := extra_charge / cost_per_GB
  let total_data := week1 + week2 + week4 + d3
  let overage_GB := total_data - data_plan
  have : overage_GB = total_extra_GB := sorry
  have : d3 = 5 := sorry
  sorry

end charlie_third_week_data_l169_169571


namespace parabola_focus_coordinates_l169_169915

theorem parabola_focus_coordinates (x y : ℝ) (h : y = 2 * x^2) : (0, 1 / 8) = (0, 1 / 8) :=
by
  sorry

end parabola_focus_coordinates_l169_169915


namespace quadratic_real_roots_condition_l169_169184

theorem quadratic_real_roots_condition (a : ℝ) :
  (∃ x : ℝ, (a - 5) * x^2 - 4 * x - 1 = 0) ↔ (a ≥ 1 ∧ a ≠ 5) :=
by
  sorry

end quadratic_real_roots_condition_l169_169184


namespace price_difference_l169_169610

noncomputable def original_price (discounted_price : ℝ) : ℝ :=
  discounted_price / 0.85

noncomputable def final_price (discounted_price : ℝ) : ℝ :=
  discounted_price * 1.25

theorem price_difference (discounted_price : ℝ) (h : discounted_price = 71.4) : 
  (final_price discounted_price) - (original_price discounted_price) = 5.25 := 
by
  sorry

end price_difference_l169_169610


namespace nate_pages_left_to_read_l169_169077

-- Define the constants and conditions
def total_pages : ℕ := 400
def percentage_read : ℕ := 20

-- Calculate the number of pages already read
def pages_read := total_pages * percentage_read / 100

-- Calculate the number of pages left
def pages_left := total_pages - pages_read

-- Statement to prove
theorem nate_pages_left_to_read : pages_left = 320 :=
by {
  unfold pages_read,
  unfold pages_left,
  simp,
  sorry -- The proof will be filled in based on the calculations in the solution.
}

end nate_pages_left_to_read_l169_169077


namespace zander_stickers_l169_169866

/-- Zander starts with 100 stickers, Andrew receives 1/5 of Zander's total, 
    and Bill receives 3/10 of the remaining stickers. Prove that the total 
    number of stickers given to Andrew and Bill is 44. -/
theorem zander_stickers :
  let total_stickers := 100
  let andrew_fraction := 1 / 5
  let remaining_stickers := total_stickers - (total_stickers * andrew_fraction)
  let bill_fraction := 3 / 10
  (total_stickers * andrew_fraction) + (remaining_stickers * bill_fraction) = 44 := 
by
  sorry

end zander_stickers_l169_169866


namespace Isabel_initial_flowers_l169_169552

-- Constants for conditions
def b := 7  -- Number of bouquets after wilting
def fw := 10  -- Number of wilted flowers
def n := 8  -- Number of flowers in each bouquet

-- Theorem statement
theorem Isabel_initial_flowers (h1 : b = 7) (h2 : fw = 10) (h3 : n = 8) : 
  (b * n + fw = 66) := by
  sorry

end Isabel_initial_flowers_l169_169552


namespace no_such_sequence_exists_l169_169110

theorem no_such_sequence_exists (a : ℕ → ℝ) :
  (∀ i, 1 ≤ i ∧ i ≤ 13 → a i + a (i + 1) + a (i + 2) > 0) →
  (∀ i, 1 ≤ i ∧ i ≤ 12 → a i + a (i + 1) + a (i + 2) + a (i + 3) < 0) →
  False :=
by
  sorry

end no_such_sequence_exists_l169_169110


namespace cost_of_green_pill_l169_169494

-- Let the cost of a green pill be g and the cost of a pink pill be p
variables (g p : ℕ)
-- Beth takes two green pills and one pink pill each day
-- A green pill costs twice as much as a pink pill
-- The total cost for the pills over three weeks (21 days) is $945

theorem cost_of_green_pill : 
  (2 * g + p) * 21 = 945 ∧ g = 2 * p → g = 18 :=
by
  sorry

end cost_of_green_pill_l169_169494


namespace probability_xi_leq_0_l169_169523

noncomputable def xi : ℝ → ℝ := sorry -- Define the random variable ξ, which is normally distributed.

axiom normal_dist {μ δ : ℝ} (Hδ_pos : δ > 0) : 
  ∀ x : ℝ, xi x ∼ NormalDist.mk μ δ

axiom probability_condition : (ProbabilityMassFunc xi).probability_event (λ x, x ≤ 4) = 0.84

theorem probability_xi_leq_0 (μ δ : ℝ) (Hδ_pos : δ > 0) (Hμ : μ = 2) :
  (ProbabilityMassFunc xi).probability_event (λ x, x ≤ 0) = 0.16 := by
begin
  -- main proof using the given conditions
  have H1 : (ProbabilityMassFunc xi).probability_event (λ x, x ≤ 4) = 0.84,
  from probability_condition,
  have H2 : (ProbabilityMassFunc xi).probability_event (λ x, x ≥ 4) = 1 - 0.84,
  from by rw [sub_self],
  have symmetry_property : (ProbabilityMassFunc xi).probability_event (λ x, x ≤ 0) = 
                          (ProbabilityMassFunc xi).probability_event (λ x, x ≥ 4),
  from sorry,
  rw [H2, symmetry_property],
  exact 0.16,
end

end probability_xi_leq_0_l169_169523


namespace rectangle_length_l169_169880

theorem rectangle_length (L W : ℝ) 
  (h1 : L + W = 23) 
  (h2 : L^2 + W^2 = 289) : 
  L = 15 :=
by 
  sorry

end rectangle_length_l169_169880


namespace largest_three_digit_multiple_of_6_sum_15_l169_169266

-- Statement of the problem in Lean
theorem largest_three_digit_multiple_of_6_sum_15 : 
  ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 6 = 0 ∧ (Nat.digits 10 n).sum = 15 ∧ 
  ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ m % 6 = 0 ∧ (Nat.digits 10 m).sum = 15 → m ≤ n :=
by
  sorry -- proof not required

end largest_three_digit_multiple_of_6_sum_15_l169_169266


namespace tan_225_eq_1_l169_169849

theorem tan_225_eq_1 : Real.tan (225 * Real.pi / 180) = 1 := by
  sorry

end tan_225_eq_1_l169_169849


namespace distance_from_point_to_origin_l169_169410

theorem distance_from_point_to_origin (x y : ℝ) (h : x = -3 ∧ y = 4) : 
  (Real.sqrt (x^2 + y^2)) = 5 := by
  sorry

end distance_from_point_to_origin_l169_169410


namespace LCM_of_fractions_l169_169859

theorem LCM_of_fractions (x : ℕ) (h : x > 0) : 
  lcm (1 / (4 * x)) (lcm (1 / (6 * x)) (1 / (9 * x))) = 1 / (36 * x) :=
by
  sorry

end LCM_of_fractions_l169_169859


namespace smallest_sum_l169_169384

theorem smallest_sum (x y : ℕ) (hx : x > 0) (hy : y > 0) (hxy : x ≠ y) 
  (h : (1/x + 1/y = 1/10)) : x + y = 49 := 
sorry

end smallest_sum_l169_169384


namespace generate_13121_not_generate_12131_l169_169718

theorem generate_13121 : ∃ n m : ℕ, 13121 + 1 = 2^n * 3^m := by
  sorry

theorem not_generate_12131 : ¬∃ n m : ℕ, 12131 + 1 = 2^n * 3^m := by
  sorry

end generate_13121_not_generate_12131_l169_169718


namespace jason_initial_cards_l169_169060

-- Conditions
def cards_given_away : ℕ := 9
def cards_left : ℕ := 4

-- Theorem to prove
theorem jason_initial_cards : cards_given_away + cards_left = 13 :=
by
  sorry

end jason_initial_cards_l169_169060


namespace largest_three_digit_multiple_of_6_sum_15_l169_169265

-- Statement of the problem in Lean
theorem largest_three_digit_multiple_of_6_sum_15 : 
  ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 6 = 0 ∧ (Nat.digits 10 n).sum = 15 ∧ 
  ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ m % 6 = 0 ∧ (Nat.digits 10 m).sum = 15 → m ≤ n :=
by
  sorry -- proof not required

end largest_three_digit_multiple_of_6_sum_15_l169_169265


namespace workers_in_first_group_l169_169962

theorem workers_in_first_group
  (W D : ℕ)
  (h1 : 6 * W * D = 9450)
  (h2 : 95 * D = 9975) :
  W = 15 := 
sorry

end workers_in_first_group_l169_169962


namespace complex_modulus_squared_l169_169564

theorem complex_modulus_squared (z : ℂ) (h : z^2 + Complex.abs z^2 = 4 + 6 * Complex.I) : Complex.abs z^2 = 13 / 2 :=
by
  sorry

end complex_modulus_squared_l169_169564


namespace smallest_b_in_AP_l169_169228

theorem smallest_b_in_AP (a b c : ℝ) (d : ℝ) (ha : a = b - d) (hc : c = b + d) (habc : a * b * c = 125) (hpos : 0 < a ∧ 0 < b ∧ 0 < c) : 
    b = 5 :=
by
  -- Proof needed here
  sorry

end smallest_b_in_AP_l169_169228


namespace largest_A_form_B_moving_last_digit_smallest_A_form_B_moving_last_digit_l169_169146

theorem largest_A_form_B_moving_last_digit (B : Nat) (h0 : Nat.gcd B 24 = 1) (h1 : B > 666666666) (h2 : B < 1000000000) :
  let A := 10^8 * (B % 10) + (B / 10)
  A ≤ 999999998 :=
sorry

theorem smallest_A_form_B_moving_last_digit (B : Nat) (h0 : Nat.gcd B 24 = 1) (h1 : B > 666666666) (h2 : B < 1000000000) :
  let A := 10^8 * (B % 10) + (B / 10)
  A ≥ 166666667 :=
sorry

end largest_A_form_B_moving_last_digit_smallest_A_form_B_moving_last_digit_l169_169146


namespace range_of_a_l169_169947

theorem range_of_a (a b : ℝ) :
  (∀ x : ℝ, (a * x^2 + b * x + 1 < 2)) ∧ (a - b + 1 = 1) → (-4 < a ∧ a ≤ 0) :=
by
  sorry

end range_of_a_l169_169947


namespace a3_eq_5_l169_169809

-- Define the geometric sequence and its properties
variables {a : ℕ → ℝ} {q : ℝ}

-- Assumptions
def geom_seq (a : ℕ → ℝ) (q : ℝ) := ∀ n : ℕ, a (n + 1) = a 1 * (q ^ n)
axiom a1_pos : a 1 > 0
axiom a2a4_eq_25 : a 2 * a 4 = 25
axiom geom : geom_seq a q

-- Statement to prove
theorem a3_eq_5 : a 3 = 5 :=
by sorry

end a3_eq_5_l169_169809


namespace james_remaining_balance_l169_169690

theorem james_remaining_balance 
  (initial_balance : ℕ := 500) 
  (ticket_1_2_cost : ℕ := 150)
  (ticket_3_cost : ℕ := ticket_1_2_cost / 3)
  (total_cost : ℕ := 2 * ticket_1_2_cost + ticket_3_cost)
  (roommate_share : ℕ := total_cost / 2) :
  initial_balance - roommate_share = 325 := 
by 
  -- By not considering the solution steps, we skip to the proof.
  sorry

end james_remaining_balance_l169_169690


namespace range_of_f_l169_169018

-- Define the function f
def f (x : ℕ) : ℤ := x^2 - 2 * x

-- Define the domain
def domain : Finset ℕ := {0, 1, 2, 3}

-- Define the expected range
def expected_range : Finset ℤ := {-1, 0, 3}

-- State the theorem
theorem range_of_f : (domain.image f) = expected_range := by
  sorry

end range_of_f_l169_169018


namespace variance_transformation_l169_169388

theorem variance_transformation (x : ℕ → ℝ) (n : ℕ)
  (h : var (finset.range (n + 1)).image x = 3) :
  var (finset.range (n + 1)).image (λ i, 2 * x i + 4) = 12 :=
by 
  sorry

end variance_transformation_l169_169388


namespace function_is_decreasing_l169_169203

noncomputable def f (a b x : ℝ) : ℝ := a * x^2 + b * x - 2

theorem function_is_decreasing (a b : ℝ) (f_even : ∀ x : ℝ, f a b x = f a b (-x))
  (domain_condition : 1 + a + 2 = 0) :
  ∀ x y : ℝ, 1 ≤ x → x < y → y ≤ 2 → f a 0 x > f a 0 y :=
by
  sorry

end function_is_decreasing_l169_169203


namespace IntervalForKTriangleLengths_l169_169929

noncomputable def f (x k : ℝ) := (x^4 + k * x^2 + 1) / (x^4 + x^2 + 1)

theorem IntervalForKTriangleLengths (k : ℝ) :
  (∀ (x : ℝ), 1 ≤ f x k ∧
              (k ≥ 1 → f x k ≤ (k + 2) / 3) ∧ 
              (k < 1 → f x k ≥ (k + 2) / 3)) →
  (∀ (a b c : ℝ), (f a k < f b k + f c k) ∧ 
                  (f b k < f a k + f c k) ∧ 
                  (f c k < f a k + f b k)) ↔ (-1/2 < k ∧ k < 4) :=
by sorry

#check f
#check IntervalForKTriangleLengths

end IntervalForKTriangleLengths_l169_169929


namespace roots_quad_eq_l169_169795

-- The problem statement in Lean 4
theorem roots_quad_eq (a r s : ℝ)
  (h1 : r + s = a + 1)
  (h2 : r * s = a) :
  (r - s) ^ 2 = a ^ 2 - 2 * a + 1 :=
sorry

end roots_quad_eq_l169_169795


namespace find_other_endpoint_l169_169249

theorem find_other_endpoint (x1 y1 x2 y2 xm ym : ℝ)
  (midpoint_formula_x : xm = (x1 + x2) / 2)
  (midpoint_formula_y : ym = (y1 + y2) / 2)
  (h_midpoint : xm = -3 ∧ ym = 2)
  (h_endpoint : x1 = -7 ∧ y1 = 6) :
  x2 = 1 ∧ y2 = -2 := 
sorry

end find_other_endpoint_l169_169249


namespace polynomial_solution_l169_169178

noncomputable def is_solution (P : ℝ → ℝ) : Prop :=
  (P 0 = 0) ∧ (∀ x : ℝ, P x = (P (x + 1) + P (x - 1)) / 2)

theorem polynomial_solution (P : ℝ → ℝ) : is_solution P → ∃ a : ℝ, ∀ x : ℝ, P x = a * x :=
  sorry

end polynomial_solution_l169_169178


namespace temperature_on_friday_l169_169271

theorem temperature_on_friday 
  (M T W Th F : ℝ)
  (h1 : (M + T + W + Th) / 4 = 48)
  (h2 : (T + W + Th + F) / 4 = 46)
  (h3 : M = 42) : 
  F = 34 :=
by
  sorry

end temperature_on_friday_l169_169271


namespace words_with_mistakes_percentage_l169_169285

theorem words_with_mistakes_percentage (n x : ℕ) 
  (h1 : (x - 1 : ℝ) / n = 0.24)
  (h2 : (x - 1 : ℝ) / (n - 1) = 0.25) :
  (x : ℝ) / n * 100 = 28 := 
by 
  sorry

end words_with_mistakes_percentage_l169_169285


namespace arnel_kept_fifty_pencils_l169_169492

theorem arnel_kept_fifty_pencils
    (num_boxes : ℕ) (pencils_each_box : ℕ) (friends : ℕ) (pencils_each_friend : ℕ) (total_pencils : ℕ)
    (boxes_pencils : ℕ) (friends_pencils : ℕ) :
    num_boxes = 10 →
    pencils_each_box = 5 →
    friends = 5 →
    pencils_each_friend = 8 →
    friends_pencils = friends * pencils_each_friend →
    boxes_pencils = num_boxes * pencils_each_box →
    total_pencils = boxes_pencils + friends_pencils →
    (total_pencils - friends_pencils) = 50 :=
by
    sorry

end arnel_kept_fifty_pencils_l169_169492


namespace min_socks_no_conditions_l169_169036

theorem min_socks_no_conditions (m n : Nat) (h : (m * (m - 1) = 2 * (m + n) * (m + n - 1))) : 
  m + n ≥ 4 := sorry

end min_socks_no_conditions_l169_169036


namespace average_pages_per_hour_l169_169151

theorem average_pages_per_hour 
  (P : ℕ) (H : ℕ) (hP : P = 30000) (hH : H = 150) : 
  P / H = 200 := 
by 
  sorry

end average_pages_per_hour_l169_169151


namespace abdul_largest_number_l169_169625

theorem abdul_largest_number {a b c d : ℕ} 
  (h1 : a + (b + c + d) / 3 = 17)
  (h2 : b + (a + c + d) / 3 = 21)
  (h3 : c + (a + b + d) / 3 = 23)
  (h4 : d + (a + b + c) / 3 = 29) :
  d = 21 :=
by sorry

end abdul_largest_number_l169_169625


namespace cone_surface_area_volume_ineq_l169_169254

theorem cone_surface_area_volume_ineq
  (A V r a m : ℝ)
  (hA : A = π * r * (r + a))
  (hV : V = (1/3) * π * r^2 * m)
  (hPythagoras : a^2 = r^2 + m^2) :
  A^3 ≥ 72 * π * V^2 := 
by
  sorry

end cone_surface_area_volume_ineq_l169_169254


namespace find_A_find_b_c_l169_169511

variable (a b c A B C : ℝ)
hypothesis (h1 : ∀ {A B C : ℕ} (a b c : ℝ), a, b, c are the sides opposite to angles A, B, C in  \triangle ABC respectively)
hypothesis (h2 : ∀ (a b c A B C: ℝ), c = a * Real.sin C - c * Real.cos A)

theorem find_A (h1) (h2)  : A = Real.pi / 2 := by
  sorry

variable (area : ℝ)
hypothesis (h3 : area = 2)
hypothesis (h4 : ∀ (a : ℕ),  a = 2 ) 

theorem find_b_c (h1) (h2) (h3) (h4)  : (b = 2) ∧ (c = 2) := by
  sorry

end find_A_find_b_c_l169_169511


namespace sum_of_odd_function_at_points_l169_169407

def is_odd_function (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = -f x

theorem sum_of_odd_function_at_points (f : ℝ → ℝ) (h : is_odd_function f) : 
  f (-2) + f (-1) + f 0 + f 1 + f 2 = 0 :=
by
  sorry

end sum_of_odd_function_at_points_l169_169407


namespace parabola_equation_maximum_area_of_triangle_l169_169684

-- Definitions of the conditions
def parabola_eq (x y : ℝ) (p : ℝ) : Prop := x^2 = 2 * p * y ∧ p > 0
def distances_equal (AO AF : ℝ) : Prop := AO = 3 / 2 ∧ AF = 3 / 2
def line_eq (x k b y : ℝ) : Prop := y = k * x + b
def midpoint_y (y1 y2 : ℝ) : Prop := (y1 + y2) / 2 = 1

-- Part (I)
theorem parabola_equation (p : ℝ) (x y AO AF : ℝ) (h1 : parabola_eq x y p)
  (h2 : distances_equal AO AF) :
  x^2 = 4 * y :=
sorry

-- Part (II)
theorem maximum_area_of_triangle (p k b AO AF x1 y1 x2 y2 : ℝ)
  (h1 : parabola_eq x1 y1 p) (h2 : parabola_eq x2 y2 p)
  (h3 : distances_equal AO AF) (h4 : line_eq x1 k b y1) 
  (h5 : line_eq x2 k b y2) (h6 : midpoint_y y1 y2)
  : ∃ (area : ℝ), area = 2 :=
sorry

end parabola_equation_maximum_area_of_triangle_l169_169684


namespace derek_percentage_difference_l169_169613

-- Definitions and assumptions based on conditions
def average_score_first_test (A : ℝ) : ℝ := A

def derek_score_first_test (D1 : ℝ) (A : ℝ) : Prop := D1 = 0.5 * A

def derek_score_second_test (D2 : ℝ) (D1 : ℝ) : Prop := D2 = 1.5 * D1

-- Theorem statement
theorem derek_percentage_difference (A D1 D2 : ℝ)
  (h1 : derek_score_first_test D1 A)
  (h2 : derek_score_second_test D2 D1) :
  (A - D2) / A * 100 = 25 :=
by
  -- Placeholder for the proof
  sorry

end derek_percentage_difference_l169_169613


namespace positive_difference_of_perimeters_l169_169592

theorem positive_difference_of_perimeters :
  let length1 := 6
  let width1 := 1
  let length2 := 3
  let width2 := 2
  let perimeter1 := 2 * (length1 + width1)
  let perimeter2 := 2 * (length2 + width2)
  (perimeter1 - perimeter2) = 4 :=
by
  let length1 := 6
  let width1 := 1
  let length2 := 3
  let width2 := 2
  let perimeter1 := 2 * (length1 + width1)
  let perimeter2 := 2 * (length2 + width2)
  show (perimeter1 - perimeter2) = 4
  sorry

end positive_difference_of_perimeters_l169_169592


namespace rons_pick_times_l169_169085

def total_members(couples single_people : ℕ) : ℕ := couples * 2 + single_people + 2

def times_rons_pick(total_members : ℕ) : ℕ := 52 / total_members

theorem rons_pick_times
    (couples single_people : ℕ)
    (h_couples : couples = 3)
    (h_single_people : single_people = 5) :
    times_rons_pick (total_members couples single_people) = 4 :=
by
  have h_total_members : total_members couples single_people = 13 := by
    simp [total_members, h_couples, h_single_people]
  simp [times_rons_pick, h_total_members]
  sorry

end rons_pick_times_l169_169085


namespace no_nat_m_n_eq_m_squared_eq_n_squared_plus_2014_l169_169323

theorem no_nat_m_n_eq_m_squared_eq_n_squared_plus_2014 :
  ¬ ∃ (m n : ℕ), m^2 = n^2 + 2014 :=
by 
  sorry

end no_nat_m_n_eq_m_squared_eq_n_squared_plus_2014_l169_169323


namespace units_digit_17_pow_2007_l169_169733

theorem units_digit_17_pow_2007 :
  (17 ^ 2007) % 10 = 3 := 
sorry

end units_digit_17_pow_2007_l169_169733


namespace mason_internet_speed_l169_169074

-- Definitions based on the conditions
def total_data : ℕ := 880
def downloaded_data : ℕ := 310
def remaining_time : ℕ := 190

-- Statement: The speed of Mason's Internet connection after it slows down
theorem mason_internet_speed :
  (total_data - downloaded_data) / remaining_time = 3 :=
by
  sorry

end mason_internet_speed_l169_169074


namespace fibonacci_inequality_l169_169253

def Fibonacci (n : ℕ) : ℕ :=
  match n with
  | 0     => 0
  | 1     => 1
  | 2     => 2
  | n + 2 => Fibonacci (n + 1) + Fibonacci n

theorem fibonacci_inequality (n : ℕ) (h : n > 0) : 
  Real.sqrt (Fibonacci (n+1)) > 1 + 1 / Real.sqrt (Fibonacci n) := 
sorry

end fibonacci_inequality_l169_169253


namespace no_nat_numbers_m_n_satisfy_eq_l169_169340

theorem no_nat_numbers_m_n_satisfy_eq (m n : ℕ) : ¬ (m^2 = n^2 + 2014) := sorry

end no_nat_numbers_m_n_satisfy_eq_l169_169340


namespace part_a_l169_169131

theorem part_a (x y : ℝ) (hx : x ≠ 1) (hy : y ≠ 1) (hxy : x * y ≠ 1) :
  (x * y) / (1 - x * y) = x / (1 - x) + y / (1 - y) :=
sorry

end part_a_l169_169131


namespace evening_temperature_is_correct_l169_169631

-- Define the temperatures at noon and in the evening
def T_noon : ℤ := 3
def T_evening : ℤ := -2

-- State the theorem to prove
theorem evening_temperature_is_correct : T_evening = -2 := by
  sorry

end evening_temperature_is_correct_l169_169631


namespace area_of_pentagon_m_n_l169_169176

noncomputable def m : ℤ := 12
noncomputable def n : ℤ := 11

theorem area_of_pentagon_m_n :
  let pentagon_area := (Real.sqrt m) + (Real.sqrt n)
  m + n = 23 :=
by
  have m_pos : m > 0 := by sorry
  have n_pos : n > 0 := by sorry
  sorry

end area_of_pentagon_m_n_l169_169176


namespace trader_loss_percent_l169_169959

theorem trader_loss_percent :
  let SP1 : ℝ := 404415
  let SP2 : ℝ := 404415
  let gain_percent : ℝ := 15 / 100
  let loss_percent : ℝ := 15 / 100
  let CP1 : ℝ := SP1 / (1 + gain_percent)
  let CP2 : ℝ := SP2 / (1 - loss_percent)
  let TCP : ℝ := CP1 + CP2
  let TSP : ℝ := SP1 + SP2
  let overall_loss : ℝ := TSP - TCP
  let overall_loss_percent : ℝ := (overall_loss / TCP) * 100
  overall_loss_percent = -2.25 := 
sorry

end trader_loss_percent_l169_169959


namespace negation_of_universal_statement_l169_169717

theorem negation_of_universal_statement :
  (¬ ∀ x : ℝ, x^4 - x^3 + x^2 + 5 ≤ 0) ↔ (∃ x : ℝ, x^4 - x^3 + x^2 + 5 > 0) :=
by sorry

end negation_of_universal_statement_l169_169717


namespace midpoint_concurrence_l169_169658

variables {V : Type*} [inner_product_space ℝ V] 
variables (A B C P : V)

def L : V := C + B - P
def M : V := A + C - P
def N : V := B + A - P

theorem midpoint_concurrence :
  ∃ S : V, 
    S = (A + L) / 2 ∧
    S = (B + M) / 2 ∧
    S = (C + N) / 2 :=
begin
  use (A + B + C - P) / 2,
  split,
  { rw [L, add_comm C B, ←add_assoc, add_sub_cancel], },
  split,
  { rw [M, add_comm A C, add_assoc, symm (sub_add_cancel)], },
  { rw [N, add_comm A B, add_assoc, symm (add_sub_cancel)] }
end

end midpoint_concurrence_l169_169658


namespace no_nat_solutions_m2_eq_n2_plus_2014_l169_169315

theorem no_nat_solutions_m2_eq_n2_plus_2014 :
  ¬ ∃ (m n : ℕ), m^2 = n^2 + 2014 :=
by
  sorry

end no_nat_solutions_m2_eq_n2_plus_2014_l169_169315


namespace angle_conversion_l169_169501

theorem angle_conversion : (1 : ℝ) * (π / 180) * (-225) = - (5 * π / 4) :=
by
  sorry

end angle_conversion_l169_169501


namespace constant_term_of_binomial_expansion_l169_169512

theorem constant_term_of_binomial_expansion :
  let a := ∫ x in (0:ℝ)..(2:ℝ), (1 - 2*x) in
  (a = ∫ x in (0:ℝ)..(2:ℝ), (1 - 2*x)) → 
  (let expr := (1/2 * x^2 + a / x) ^ 6 in
    ∃ c : ℤ, natDegreePolynomialxnTerm expr 0 = 60) :=
by { sorry }

end constant_term_of_binomial_expansion_l169_169512


namespace inequality_correct_transformation_l169_169269

-- Definitions of the conditions
variables (a b : ℝ)

-- The equivalent proof problem
theorem inequality_correct_transformation (h : a > b) : -a < -b :=
by sorry

end inequality_correct_transformation_l169_169269


namespace total_games_proof_l169_169807

def num_teams : ℕ := 20
def num_games_per_team_regular_season : ℕ := 38
def total_regular_season_games : ℕ := num_teams * (num_games_per_team_regular_season / 2)
def num_games_per_team_mid_season : ℕ := 3
def total_mid_season_games : ℕ := num_teams * num_games_per_team_mid_season
def quarter_finals_teams : ℕ := 8
def quarter_finals_matchups : ℕ := quarter_finals_teams / 2
def quarter_finals_games : ℕ := quarter_finals_matchups * 2
def semi_finals_teams : ℕ := quarter_finals_matchups
def semi_finals_matchups : ℕ := semi_finals_teams / 2
def semi_finals_games : ℕ := semi_finals_matchups * 2
def final_teams : ℕ := semi_finals_matchups
def final_games : ℕ := final_teams * 2
def total_playoff_games : ℕ := quarter_finals_games + semi_finals_games + final_games

def total_season_games : ℕ := total_regular_season_games + total_mid_season_games + total_playoff_games

theorem total_games_proof : total_season_games = 454 := by
  -- The actual proof will go here
  sorry

end total_games_proof_l169_169807


namespace find_a_and_lambda_find_inverse_of_A_l169_169200

open Matrix

def A (a : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![1, a], ![-1, 4]]

def v : Vector (Fin 2) ℝ := ![1, 1]

noncomputable def λvalue (a : ℝ) : ℝ :=
  (A a).mulVec v 0 -- Extract the λ value

theorem find_a_and_lambda : 
  ∀ (a : ℝ), A a.mulVec v = (λvalue a) • v → a = 2 ∧ λvalue 2 = 3 :=
by sorry

def A_2 : Matrix (Fin 2) (Fin 2) ℝ := A 2

noncomputable def A_inv : Matrix (Fin 2) (Fin 2) ℝ := inverse (A 2)

theorem find_inverse_of_A : A_inv = ![
  ![(2 / 3), (-1 / 3)],
  ![(1 / 6), (1 / 6)]
] :=
by sorry

end find_a_and_lambda_find_inverse_of_A_l169_169200


namespace identify_random_event_l169_169157

-- Definitions of the events
def event1 : String := "Tossing a coin twice in a row and getting heads both times"
def event2 : String := "Opposite charges attract each other"
def event3 : String := "Water freezes at 1 ℃ under standard atmospheric pressure"

-- Statements about the type of events
def is_random_event (e : String) : Prop := 
  e = event1 -- We are directly identifying event1 as the random event here.

theorem identify_random_event : is_random_event event1 :=
by
  sorry

end identify_random_event_l169_169157


namespace solution_set_a1_range_of_a_l169_169378

def f (x a : ℝ) : ℝ := abs (x - a) * abs (x + abs (x - 2)) * abs (x - a)

theorem solution_set_a1 (x : ℝ) : f x 1 < 0 ↔ x < 1 :=
by
  sorry

theorem range_of_a (a : ℝ) : (∀ x, x < 1 → f x a < 0) ↔ 1 ≤ a :=
by
  sorry

end solution_set_a1_range_of_a_l169_169378


namespace joe_eats_at_least_two_different_fruits_l169_169892

namespace JoeFruitProblem

-- Define the probability space and events
noncomputable def prob_at_least_two_different_fruits : ℚ := sorry

theorem joe_eats_at_least_two_different_fruits
  : prob_at_least_two_different_fruits = 63 / 64 := 
sorry

end JoeFruitProblem

end joe_eats_at_least_two_different_fruits_l169_169892


namespace corrected_observations_mean_l169_169715

noncomputable def corrected_mean (mean incorrect correct: ℚ) (n: ℕ) : ℚ :=
  let S_incorrect := mean * n
  let Difference := correct - incorrect
  let S_corrected := S_incorrect + Difference
  S_corrected / n

theorem corrected_observations_mean:
  corrected_mean 36 23 34 50 = 36.22 := by
  sorry

end corrected_observations_mean_l169_169715


namespace divides_polynomial_difference_l169_169419

def P (a b c d x : ℤ) : ℤ := a * x^3 + b * x^2 + c * x + d

theorem divides_polynomial_difference (a b c d x y : ℤ) (hxneqy : x ≠ y) :
  (x - y) ∣ (P a b c d x - P a b c d y) :=
by
  sorry

end divides_polynomial_difference_l169_169419


namespace polynomial_solution_l169_169179

noncomputable def P (x : ℝ) : ℝ := sorry

theorem polynomial_solution (P : ℝ → ℝ) (h1 : P 0 = 0)
  (h2 : ∀ x : ℝ, P x = (P (x + 1) + P (x - 1)) / 2) :
  ∃ (a : ℝ), ∀ x : ℝ, P x = a * x :=
begin
  sorry
end

end polynomial_solution_l169_169179


namespace rectangle_perimeter_eq_l169_169766

noncomputable def rectangle_perimeter (z w : ℕ) : ℕ :=
  let longer_side := w
  let shorter_side := (z - w) / 2
  2 * longer_side + 2 * shorter_side

theorem rectangle_perimeter_eq (z w : ℕ) : rectangle_perimeter z w = w + z := by
  sorry

end rectangle_perimeter_eq_l169_169766


namespace price_comparison_2010_l169_169443

def X_initial : ℝ := 4.20
def Y_initial : ℝ := 6.30
def r_X : ℝ := 0.45
def r_Y : ℝ := 0.20
def n : ℕ := 9

theorem price_comparison_2010: 
  X_initial + r_X * n > Y_initial + r_Y * n := by
  sorry

end price_comparison_2010_l169_169443


namespace complex_addition_l169_169823

namespace ComplexProof

def B := (3 : ℂ) + (2 * Complex.I)
def Q := (-5 : ℂ)
def R := (2 * Complex.I)
def T := (3 : ℂ) + (5 * Complex.I)

theorem complex_addition :
  B - Q + R + T = (1 : ℂ) + (9 * Complex.I) := 
by
  sorry

end ComplexProof

end complex_addition_l169_169823


namespace darla_total_payment_l169_169902

-- Define the cost per watt, total watts used, and late fee
def cost_per_watt : ℝ := 4
def total_watts : ℝ := 300
def late_fee : ℝ := 150

-- Define the total cost of electricity
def electricity_cost : ℝ := cost_per_watt * total_watts

-- Define the total amount Darla needs to pay
def total_amount : ℝ := electricity_cost + late_fee

-- The theorem to prove the total amount equals $1350
theorem darla_total_payment : total_amount = 1350 := by
  sorry

end darla_total_payment_l169_169902


namespace not_right_triangle_l169_169223

theorem not_right_triangle (a b c : ℝ) (h : a / b = 1 / 2 ∧ b / c = 2 / 3) :
  ¬(a^2 = b^2 + c^2) :=
by sorry

end not_right_triangle_l169_169223


namespace marked_price_l169_169140

theorem marked_price (initial_price : ℝ) (discount_percent : ℝ) (profit_margin_percent : ℝ) (final_discount_percent : ℝ) (marked_price : ℝ) :
  initial_price = 40 → 
  discount_percent = 0.25 → 
  profit_margin_percent = 0.50 → 
  final_discount_percent = 0.10 → 
  marked_price = 50 := by
  sorry

end marked_price_l169_169140


namespace abs_gt_1_not_sufficient_nor_necessary_l169_169477

theorem abs_gt_1_not_sufficient_nor_necessary (a : ℝ) :
  ¬((|a| > 1) → (a > 0)) ∧ ¬((a > 0) → (|a| > 1)) :=
by
  sorry

end abs_gt_1_not_sufficient_nor_necessary_l169_169477


namespace avg_megabyte_usage_per_hour_l169_169619

theorem avg_megabyte_usage_per_hour (megabytes : ℕ) (days : ℕ) (hours : ℕ) (avg_mbps : ℕ)
  (h1 : megabytes = 27000)
  (h2 : days = 15)
  (h3 : hours = days * 24)
  (h4 : avg_mbps = megabytes / hours) : 
  avg_mbps = 75 := by
  sorry

end avg_megabyte_usage_per_hour_l169_169619


namespace average_last_two_numbers_l169_169435

theorem average_last_two_numbers (a b c d e f g : ℝ) 
  (h1 : (a + b + c + d + e + f + g) / 7 = 63) 
  (h2 : (a + b + c) / 3 = 58) 
  (h3 : (d + e) / 2 = 70) :
  ((f + g) / 2) = 63.5 := 
sorry

end average_last_two_numbers_l169_169435


namespace problem_solution_l169_169370

open Set

theorem problem_solution (x : ℝ) :
  (x ∈ {y : ℝ | (2 / (y + 2) + 4 / (y + 8) ≥ 1)} ↔ x ∈ Ioo (-8 : ℝ) (-2 : ℝ)) :=
sorry

end problem_solution_l169_169370


namespace calum_spend_per_disco_ball_l169_169294

def calum_budget := 330
def food_cost_per_box := 25
def number_of_food_boxes := 10
def number_of_disco_balls := 4

theorem calum_spend_per_disco_ball : (calum_budget - food_cost_per_box * number_of_food_boxes) / number_of_disco_balls = 20 :=
by
  sorry

end calum_spend_per_disco_ball_l169_169294


namespace generating_function_solution_count_l169_169232

noncomputable theory

def a_n (n k : ℕ) : ℕ :=
  ((Polynomial.X : Polynomial ℚ)^(n + k - 1)).coeff n

def F (x : ℚ) (k : ℕ) : ℚ :=
  (1 - x) ^ -k

theorem generating_function (k : ℕ) : 
  generate_function (λ n, a_n n k) = λ x, F x k :=
sorry

theorem solution_count (n k : ℕ) :
  a_n n k = nat.choose (n + k - 1) n :=
sorry

end generating_function_solution_count_l169_169232


namespace most_likely_units_digit_l169_169172

theorem most_likely_units_digit :
  ∃ m n : Fin 11, ∀ (M N : Fin 11), (∃ k : Nat, k * 11 + M + N = m + n) → 
    (m + n) % 10 = 0 :=
by
  sorry

end most_likely_units_digit_l169_169172


namespace train_passing_time_l169_169412

theorem train_passing_time 
  (length_train : ℕ) 
  (speed_train_kmph : ℕ) 
  (time_to_pass : ℕ)
  (h1 : length_train = 60)
  (h2 : speed_train_kmph = 54)
  (h3 : time_to_pass = 4) :
  time_to_pass = length_train * 18 / (speed_train_kmph * 5) := by
  sorry

end train_passing_time_l169_169412


namespace compare_triangle_operations_l169_169770

def tri_op (a b : ℤ) : ℤ := a * b - a - b + 1

theorem compare_triangle_operations : tri_op (-3) 4 = tri_op 4 (-3) :=
by
  unfold tri_op
  sorry

end compare_triangle_operations_l169_169770


namespace isosceles_triangle_perimeter_l169_169722

theorem isosceles_triangle_perimeter (a b : ℕ)
  (h_eqn : ∀ x : ℕ, (x - 4) * (x - 2) = 0 → x = 4 ∨ x = 2)
  (h_isosceles : ∃ a b : ℕ, (a = 4 ∧ b = 2) ∨ (a = 2 ∧ b = 4) ∨ (a = 4 ∧ b = 4)) :
  a + a + b = 10 :=
by
  sorry

end isosceles_triangle_perimeter_l169_169722


namespace ratio_cereal_A_to_B_l169_169125

-- Definitions translated from conditions
def sugar_percentage_A : ℕ := 10
def sugar_percentage_B : ℕ := 2
def desired_sugar_percentage : ℕ := 6

-- The theorem based on the question and correct answer
theorem ratio_cereal_A_to_B :
  let difference_A := sugar_percentage_A - desired_sugar_percentage
  let difference_B := desired_sugar_percentage - sugar_percentage_B
  difference_A = 4 ∧ difference_B = 4 → 
  difference_B / difference_A = 1 :=
by
  intros
  sorry

end ratio_cereal_A_to_B_l169_169125


namespace solve_quadratic_equation_l169_169093

theorem solve_quadratic_equation (x : ℝ) : x^2 - 4*x + 3 = 0 ↔ (x = 1 ∨ x = 3) := 
by 
  sorry

end solve_quadratic_equation_l169_169093


namespace probability_jqk_3_13_l169_169006

def probability_jack_queen_king (total_cards jacks queens kings : ℕ) : ℚ :=
  (jacks + queens + kings) / total_cards

theorem probability_jqk_3_13 :
  probability_jack_queen_king 52 4 4 4 = 3 / 13 := by
  sorry

end probability_jqk_3_13_l169_169006


namespace limes_given_l169_169166

theorem limes_given (original_limes now_limes : ℕ) (h1 : original_limes = 9) (h2 : now_limes = 5) : (original_limes - now_limes = 4) := 
by
  sorry

end limes_given_l169_169166


namespace george_boxes_l169_169007

-- Define the problem conditions and the question's expected outcome.
def total_blocks : ℕ := 12
def blocks_per_box : ℕ := 6
def expected_num_boxes : ℕ := 2

-- The proof statement that needs to be proved: George has the expected number of boxes.
theorem george_boxes : total_blocks / blocks_per_box = expected_num_boxes := 
  sorry

end george_boxes_l169_169007


namespace average_temperature_l169_169922

def temperatures : List ℝ := [-36, 13, -15, -10]

theorem average_temperature : (List.sum temperatures) / (temperatures.length) = -12 := by
  sorry

end average_temperature_l169_169922


namespace no_nat_m_n_eq_m_squared_eq_n_squared_plus_2014_l169_169319

theorem no_nat_m_n_eq_m_squared_eq_n_squared_plus_2014 :
  ¬ ∃ (m n : ℕ), m^2 = n^2 + 2014 :=
by 
  sorry

end no_nat_m_n_eq_m_squared_eq_n_squared_plus_2014_l169_169319


namespace square_perimeter_increase_l169_169588

theorem square_perimeter_increase (s : ℝ) : (4 * (s + 2) - 4 * s) = 8 := 
by
  sorry

end square_perimeter_increase_l169_169588


namespace range_of_a_l169_169514

section
  variable {x a : ℝ}

  -- Define set A
  def setA : Set ℝ := { x | x^2 - 4*x + 3 < 0 }

  -- Define set B
  def setB (a : ℝ) : Set ℝ := 
    { x | (2*x + a ≤ 0) ∧ (x^2 - 2*(a + 7)*x + 5 ≤ 0)}

  -- The proof problem statement
  theorem range_of_a (a : ℝ) : 
    (setA ⊆ setB a) ↔ (-4 ≤ a ∧ a ≤ -2) :=
  sorry
end

end range_of_a_l169_169514


namespace function_identity_l169_169694

variables {R : Type*} [LinearOrderedField R]

-- Define real-valued functions f, g, h
variables (f g h : R → R)

-- Define function composition and multiplication
def comp (f g : R → R) (x : R) := f (g x)
def mul (f g : R → R) (x : R) := f x * g x

-- The statement to prove
theorem function_identity (x : R) : 
  comp (mul f g) h x = mul (comp f h) (comp g h) x :=
sorry

end function_identity_l169_169694


namespace find_corresponding_element_l169_169970

def f (p : ℝ × ℝ) : ℝ × ℝ := (p.1 - p.2, p.1 + p.2)

theorem find_corresponding_element :
  f (-1, 2) = (-3, 1) :=
by
  sorry

end find_corresponding_element_l169_169970


namespace weight_ratio_l169_169799

noncomputable def students_weight : ℕ := 79
noncomputable def siblings_total_weight : ℕ := 116

theorem weight_ratio (S W : ℕ) (h1 : siblings_total_weight = S + W) (h2 : students_weight = S):
  (S - 5) / (siblings_total_weight - S) = 2 :=
by
  sorry

end weight_ratio_l169_169799


namespace probability_of_four_of_same_value_l169_169001

-- Define the conditions
def total_ways_to_draw_6_cards : ℕ := Nat.factorial 52 / (Nat.factorial 6 * Nat.factorial (52 - 6))
def ways_to_choose_4_of_same_value : ℕ := 13 * (Nat.factorial 48 / (Nat.factorial 2 * Nat.factorial (48 - 2)))

-- Define the probability calculation
def probability : ℚ :=
  (ways_to_choose_4_of_same_value : ℚ) / (total_ways_to_draw_6_cards : ℚ)

-- Prove that the probability equals 3/4165
theorem probability_of_four_of_same_value :
  probability = 3 / 4165 :=
sorry

end probability_of_four_of_same_value_l169_169001


namespace no_nat_solutions_for_m2_eq_n2_plus_2014_l169_169360

theorem no_nat_solutions_for_m2_eq_n2_plus_2014 :
  ∀ m n : ℕ, ¬(m^2 = n^2 + 2014) := by
sorry

end no_nat_solutions_for_m2_eq_n2_plus_2014_l169_169360


namespace alloy_problem_solution_l169_169551

theorem alloy_problem_solution (x y k n : ℝ) (H_weight : k * 4 * x + n * 3 * y = 10)
    (H_ratio : (kx + ny)/(k * 3 * x + n * 2 * y) = 3/7) :
    k * 4 * x = 4 :=
by
  -- Proof to be provided
  sorry

end alloy_problem_solution_l169_169551


namespace number_of_people_entered_l169_169279

-- Define the total number of placards
def total_placards : ℕ := 5682

-- Define the number of placards each person takes
def placards_per_person : ℕ := 2

-- The Lean theorem to prove the number of people who entered the stadium
theorem number_of_people_entered : total_placards / placards_per_person = 2841 :=
by
  -- Proof will be inserted here
  sorry

end number_of_people_entered_l169_169279


namespace intercept_sum_l169_169617

theorem intercept_sum {x y : ℝ} 
  (h : y - 3 = -3 * (x - 5)) 
  (hx : x = 6) 
  (hy : y = 18) 
  (intercept_sum_eq : x + y = 24) : 
  x + y = 24 :=
by
  sorry

end intercept_sum_l169_169617


namespace ratio_lcm_gcf_eq_55_l169_169732

theorem ratio_lcm_gcf_eq_55 : 
  ∀ (a b : ℕ), a = 210 → b = 462 →
  (Nat.lcm a b / Nat.gcd a b) = 55 :=
by
  intros a b ha hb
  rw [ha, hb]
  sorry

end ratio_lcm_gcf_eq_55_l169_169732


namespace chebyshev_substitution_even_chebyshev_substitution_odd_l169_169605

def T (n : ℕ) (x : ℝ) : ℝ := sorry -- Chebyshev polynomial of the first kind
def U (n : ℕ) (x : ℝ) : ℝ := sorry -- Chebyshev polynomial of the second kind

theorem chebyshev_substitution_even (k : ℕ) (α : ℝ) :
  T (2 * k) (Real.sin α) = (-1)^k * Real.cos ((2 * k) * α) ∧
  U ((2 * k) - 1) (Real.sin α) = (-1)^(k + 1) * (Real.sin ((2 * k) * α) / Real.cos α) :=
by
  sorry

theorem chebyshev_substitution_odd (k : ℕ) (α : ℝ) :
  T (2 * k + 1) (Real.sin α) = (-1)^k * Real.sin ((2 * k + 1) * α) ∧
  U (2 * k) (Real.sin α) = (-1)^k * (Real.cos ((2 * k + 1) * α) / Real.cos α) :=
by
  sorry

end chebyshev_substitution_even_chebyshev_substitution_odd_l169_169605


namespace Sandwiches_count_l169_169087

-- Define the number of toppings and the number of choices for the patty
def num_toppings : Nat := 10
def num_choices_per_topping : Nat := 2
def num_patties : Nat := 3

-- Define the theorem to prove the total number of sandwiches
theorem Sandwiches_count : (num_choices_per_topping ^ num_toppings) * num_patties = 3072 :=
by
  sorry

end Sandwiches_count_l169_169087


namespace ice_cream_cost_l169_169002

variable {x F M : ℤ}

theorem ice_cream_cost (h1 : F = x - 7) (h2 : M = x - 1) (h3 : F + M < x) : x = 7 :=
by
  sorry

end ice_cream_cost_l169_169002


namespace gnollish_valid_sentence_count_is_48_l169_169845

-- Define the problem parameters
def gnollish_words : List String := ["word1", "word2", "splargh", "glumph", "kreeg"]

def valid_sentence_count : Nat :=
  let total_sentences := 4 * 4 * 4
  let invalid_sentences :=
    4 +         -- (word) splargh glumph
    4 +         -- splargh glumph (word)
    4 +         -- (word) splargh kreeg
    4           -- splargh kreeg (word)
  total_sentences - invalid_sentences

-- Prove that the number of valid 3-word sentences is 48
theorem gnollish_valid_sentence_count_is_48 : valid_sentence_count = 48 := by
  sorry

end gnollish_valid_sentence_count_is_48_l169_169845


namespace chess_group_players_l169_169111

theorem chess_group_players (n : ℕ) (h : n * (n - 1) / 2 = 28) : n = 8 :=
by {
  sorry
}

end chess_group_players_l169_169111


namespace jillian_largest_apartment_l169_169491

noncomputable def largest_apartment_size (budget : ℝ) (rate : ℝ) : ℝ :=
  budget / rate

theorem jillian_largest_apartment : largest_apartment_size 720 1.20 = 600 := by
  sorry

end jillian_largest_apartment_l169_169491


namespace fuel_needed_to_empty_l169_169483

theorem fuel_needed_to_empty (x : ℝ) 
  (h1 : (3/4) * x - (1/3) * x = 15) :
  (1/3) * x = 12 :=
by 
-- Proving the result
sorry

end fuel_needed_to_empty_l169_169483


namespace range_of_x_satisfying_inequality_l169_169016

noncomputable def f : ℝ → ℝ := sorry -- f is some even and monotonically increasing function

theorem range_of_x_satisfying_inequality :
  (∀ x, f (-x) = f x) ∧ (∀ x y, 0 ≤ x → x ≤ y → f x ≤ f y) → {x : ℝ | f x < f 1} = {x : ℝ | -1 < x ∧ x < 1} :=
by
  intro h
  sorry

end range_of_x_satisfying_inequality_l169_169016


namespace regular_polygon_sides_160_l169_169749

theorem regular_polygon_sides_160 (n : ℕ) 
  (h1 : n ≥ 3) 
  (h2 : ∀ i, 1 ≤ i ∧ i ≤ n → (interior_angle : ℝ) = 160) : 
  n = 18 :=
by
  sorry

end regular_polygon_sides_160_l169_169749


namespace max_value_f_min_value_a_l169_169525

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x - 4 * Real.pi / 3) + 2 * (Real.cos x)^2

theorem max_value_f :
  ∀ x, f x ≤ 2 ∧ (∃ k : ℤ, x = k * Real.pi - Real.pi / 6) → f x = 2 :=
by { sorry }

variables {A B C a b c : ℝ}

noncomputable def f' (x : ℝ) : ℝ := Real.cos (2 * x +  Real.pi / 3) + 1

theorem min_value_a
  (h1 : f' (B + C) = 3/2)
  (h2 : b + c = 2)
  (h3 : A + B + C = Real.pi)
  (h4 : Real.cos A = 1/2) :
  ∃ a, ∀ b c, a^2 = b^2 + c^2 - 2 * b * c * Real.cos A ∧ a ≥ 1 :=
by { sorry }

end max_value_f_min_value_a_l169_169525


namespace floor_sqrt_120_eq_10_l169_169912

theorem floor_sqrt_120_eq_10 : ⌊Real.sqrt 120⌋ = 10 := by
  -- Here, we note that we are given:
  -- 100 < 120 < 121 and the square root of it lies between 10 and 11
  sorry

end floor_sqrt_120_eq_10_l169_169912


namespace larger_to_smaller_ratio_l169_169447

theorem larger_to_smaller_ratio (x y : ℝ) (h1 : 0 < y) (h2 : y < x) (h3 : x + y = 7 * (x - y)) :
  x / y = 4 / 3 :=
by
  sorry

end larger_to_smaller_ratio_l169_169447


namespace consecutive_days_sum_l169_169033

theorem consecutive_days_sum (x : ℕ) (h : 3 * x + 3 = 33) : x = 10 ∧ x + 1 = 11 ∧ x + 2 = 12 :=
by {
  sorry
}

end consecutive_days_sum_l169_169033


namespace find_integer_pairs_l169_169504

theorem find_integer_pairs (x y : ℤ) (h : x^3 - y^3 = 2 * x * y + 8) : 
  (x = 0 ∧ y = -2) ∨ (x = 2 ∧ y = 0) := 
by {
  sorry
}

end find_integer_pairs_l169_169504


namespace tori_passing_question_l169_169457

def arithmetic_questions : ℕ := 20
def algebra_questions : ℕ := 40
def geometry_questions : ℕ := 40
def total_questions : ℕ := arithmetic_questions + algebra_questions + geometry_questions
def arithmetic_correct_pct : ℕ := 80
def algebra_correct_pct : ℕ := 50
def geometry_correct_pct : ℕ := 70
def passing_grade_pct : ℕ := 65

theorem tori_passing_question (questions_needed_to_pass : ℕ) (arithmetic_correct : ℕ) (algebra_correct : ℕ) (geometry_correct : ℕ) : 
  questions_needed_to_pass = 1 :=
by
  let arithmetic_correct : ℕ := (arithmetic_correct_pct * arithmetic_questions / 100)
  let algebra_correct : ℕ := (algebra_correct_pct * algebra_questions / 100)
  let geometry_correct : ℕ := (geometry_correct_pct * geometry_questions / 100)
  let total_correct : ℕ := arithmetic_correct + algebra_correct + geometry_correct
  let passing_grade : ℕ := (passing_grade_pct * total_questions / 100)
  let questions_needed_to_pass : ℕ := passing_grade - total_correct
  exact sorry

end tori_passing_question_l169_169457


namespace find_c_value_l169_169969

def finds_c (c : ℝ) : Prop :=
  6 * (-(c / 6)) + 9 * (-(c / 9)) + c = 0 ∧ (-(c / 6) + -(c / 9) = 30)

theorem find_c_value : ∃ c : ℝ, finds_c c ∧ c = -108 :=
by
  use -108
  sorry

end find_c_value_l169_169969


namespace find_a_and_b_l169_169621

open Function

theorem find_a_and_b (a b : ℚ) (k : ℚ)  (hA : (6 : ℚ) = k * (-3))
    (hB : (a : ℚ) = k * 2)
    (hC : (-1 : ℚ) = k * b) : 
    a = -4 ∧ b = 1 / 2 :=
by
  sorry

end find_a_and_b_l169_169621


namespace validity_of_D_l169_169782

def binary_op (a b : ℕ) : ℕ := a^(b + 1)

theorem validity_of_D (a b n : ℕ) (ha : 0 < a) (hb : 0 < b) (hn : 0 < n) :
  binary_op (a^n) b = (binary_op a b)^n := 
by
  sorry

end validity_of_D_l169_169782


namespace no_nat_m_n_eq_m_squared_eq_n_squared_plus_2014_l169_169317

theorem no_nat_m_n_eq_m_squared_eq_n_squared_plus_2014 :
  ¬ ∃ (m n : ℕ), m^2 = n^2 + 2014 :=
by 
  sorry

end no_nat_m_n_eq_m_squared_eq_n_squared_plus_2014_l169_169317


namespace graph_transformation_l169_169875

theorem graph_transformation (a b c : ℝ) (h1 : c = 1) (h2 : a + b + c = -2) (h3 : a - b + c = 2) :
  (∀ x, cx^2 + 2 * bx + a = (x - 2)^2 - 5) := 
sorry

end graph_transformation_l169_169875


namespace chess_club_officers_l169_169425

/-- The Chess Club with 24 members needs to choose 3 officers: president,
    secretary, and treasurer. Each person can hold at most one office. 
    Alice and Bob will only serve together as officers. Prove that 
    the number of ways to choose the officers is 9372. -/
theorem chess_club_officers : 
  let members := 24
  let num_officers := 3
  let alice_and_bob_together := true
  ∃ n : ℕ, n = 9372 := sorry

end chess_club_officers_l169_169425


namespace max_sundays_in_84_days_l169_169600

-- Define constants
def days_in_week : ℕ := 7
def total_days : ℕ := 84

-- Theorem statement
theorem max_sundays_in_84_days : (total_days / days_in_week) = 12 :=
by sorry

end max_sundays_in_84_days_l169_169600


namespace triangle_area_inradius_l169_169719

theorem triangle_area_inradius
  (perimeter : ℝ) (inradius : ℝ) (area : ℝ)
  (h1 : perimeter = 35)
  (h2 : inradius = 4.5)
  (h3 : area = inradius * (perimeter / 2)) :
  area = 78.75 := by
  sorry

end triangle_area_inradius_l169_169719


namespace min_value_sin_cos_l169_169652

open Real

theorem min_value_sin_cos : ∀ x : ℝ, 
  ∃ (y : ℝ), (∀ x, y ≤ sin x ^ 6 + (5 / 3) * cos x ^ 6) ∧ y = 5 / 8 :=
by
  sorry

end min_value_sin_cos_l169_169652


namespace cannot_determine_shape_l169_169661

noncomputable def is_point (P : ℝ × ℝ) := ∃ x y : ℝ, P = (x, y)

def rectangle_points : (ℝ × ℝ) → Prop :=
λ A, (A = (0, 0)) ∨ (A = (0, 4)) ∨ (A = (6, 4)) ∨ (A = (6, 0))

def line_eq_from_A_45 (x : ℝ) : ℝ := x
def line_eq_from_A_75 (x : ℝ) : ℝ := real.tan (75 * real.pi / 180) * x

def line_eq_from_B_neg45 (x : ℝ) : ℝ := 4 - x
def line_eq_from_B_neg75 (x : ℝ) : ℝ := 4 - (real.tan (75 * real.pi / 180) * x)

def intersect (f g : ℝ → ℝ) : (ℝ × ℝ) :=
  let x := (4 - (real.tan (75 * real.pi / 180))) / (1 + real.tan (75 * real.pi / 180)) in
  (x, f x)

theorem cannot_determine_shape :
  ¬ ∃ P1 P2 : ℝ × ℝ,
    is_point P1 ∧ is_point P2 ∧
    intersect line_eq_from_A_45 line_eq_from_B_neg45 = P1 ∧
    intersect line_eq_from_A_75 line_eq_from_B_neg75 = P2 ∧
    ∀ P, ¬rectangle_points P :=
sorry

end cannot_determine_shape_l169_169661


namespace inverse_89_mod_90_l169_169645

theorem inverse_89_mod_90 : (89 * 89) % 90 = 1 := 
by
  -- Mathematical proof is skipped
  sorry

end inverse_89_mod_90_l169_169645


namespace similar_polygons_area_sum_l169_169793

theorem similar_polygons_area_sum (a b c k : ℝ) (t' t'' T : ℝ)
    (h₁ : t' = k * a^2)
    (h₂ : t'' = k * b^2)
    (h₃ : T = t' + t''):
    c^2 = a^2 + b^2 := 
by 
  sorry

end similar_polygons_area_sum_l169_169793


namespace total_profit_l169_169156

-- Define the variables for the subscriptions and profits
variables {A B C : ℕ} -- Subscription amounts
variables {profit : ℕ} -- Total profit

-- Given conditions
def conditions (A B C : ℕ) (profit : ℕ) :=
  50000 = A + B + C ∧
  A = B + 4000 ∧
  B = C + 5000 ∧
  A * profit = 29400 * 50000

-- Statement of the theorem
theorem total_profit (A B C : ℕ) (profit : ℕ) (h : conditions A B C profit) :
  profit = 70000 :=
sorry

end total_profit_l169_169156


namespace domain_of_log_function_l169_169585

open Real

noncomputable def domain_of_function : Set ℝ :=
  {x | x > 2 ∨ x < -1}

theorem domain_of_log_function :
  ∀ x : ℝ, (x^2 - x - 2 > 0) ↔ (x > 2 ∨ x < -1) :=
by
  intro x
  exact sorry

end domain_of_log_function_l169_169585


namespace complement_intersection_l169_169395

-- Define the universal set U and sets A and B.
def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def A : Set ℕ := {1, 4, 6}
def B : Set ℕ := {4, 5, 7}

-- Define the complements of A and B in U.
def C_UA : Set ℕ := U \ A
def C_UB : Set ℕ := U \ B

-- The proof problem: Prove that the intersection of the complements of A and B 
-- in the universal set U equals {2, 3, 8}.
theorem complement_intersection :
  (C_UA ∩ C_UB = {2, 3, 8}) := by
  sorry

end complement_intersection_l169_169395


namespace mowing_time_l169_169239

/-- 
Rena uses a mower to trim her "L"-shaped lawn which consists of two rectangular sections 
sharing one $50$-foot side. One section is $120$-foot by $50$-foot and the other is $70$-foot by 
$50$-foot. The mower has a swath width of $35$ inches with overlaps by $5$ inches. 
Rena walks at the rate of $4000$ feet per hour. 
Prove that it takes 0.95 hours for Rena to mow the entire lawn.
-/
theorem mowing_time 
  (length1 length2 width mower_swath overlap : ℝ) 
  (Rena_speed : ℝ) (effective_swath : ℝ) (total_area total_strips total_distance : ℝ)
  (h1 : length1 = 120)
  (h2 : length2 = 70)
  (h3 : width = 50)
  (h4 : mower_swath = 35 / 12)
  (h5 : overlap = 5 / 12)
  (h6 : effective_swath = mower_swath - overlap)
  (h7 : Rena_speed = 4000)
  (h8 : total_area = length1 * width + length2 * width)
  (h9 : total_strips = (length1 + length2) / effective_swath)
  (h10 : total_distance = total_strips * width) : 
  (total_distance / Rena_speed = 0.95) :=
by sorry

end mowing_time_l169_169239


namespace trigonometric_value_l169_169188

theorem trigonometric_value (α : ℝ) (h : Real.tan α = 2) :
  (2 * Real.sin α ^ 2 + 1) / Real.cos (2 * (α - Real.pi / 4)) = 13 / 4 := 
sorry

end trigonometric_value_l169_169188


namespace prime_quadruples_l169_169168

def is_prime (n : ℕ) : Prop := ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem prime_quadruples {p₁ p₂ p₃ p₄ : ℕ} (prime_p₁ : is_prime p₁) (prime_p₂ : is_prime p₂) (prime_p₃ : is_prime p₃) (prime_p₄ : is_prime p₄)
  (h1 : p₁ < p₂) (h2 : p₂ < p₃) (h3 : p₃ < p₄) (eq_condition : p₁ * p₂ + p₂ * p₃ + p₃ * p₄ + p₄ * p₁ = 882) :
  (p₁ = 2 ∧ p₂ = 5 ∧ p₃ = 19 ∧ p₄ = 37) ∨
  (p₁ = 2 ∧ p₂ = 11 ∧ p₃ = 19 ∧ p₄ = 31) ∨
  (p₁ = 2 ∧ p₂ = 13 ∧ p₃ = 19 ∧ p₄ = 29) :=
sorry

end prime_quadruples_l169_169168


namespace weekly_sales_correct_l169_169236

open Real

noncomputable def cost_left_handed_mouse (cost_normal_mouse : ℝ) : ℝ :=
  cost_normal_mouse * 1.3

noncomputable def cost_left_handed_keyboard (cost_normal_keyboard : ℝ) : ℝ :=
  cost_normal_keyboard * 1.2

noncomputable def cost_left_handed_scissors (cost_normal_scissors : ℝ) : ℝ :=
  cost_normal_scissors * 1.5

noncomputable def daily_sales_mouse (cost_normal_mouse : ℝ) : ℝ :=
  25 * cost_left_handed_mouse cost_normal_mouse

noncomputable def daily_sales_keyboard (cost_normal_keyboard : ℝ) : ℝ :=
  10 * cost_left_handed_keyboard cost_normal_keyboard

noncomputable def daily_sales_scissors (cost_normal_scissors : ℝ) : ℝ :=
  15 * cost_left_handed_scissors cost_normal_scissors

noncomputable def bundle_price (cost_normal_mouse : ℝ) (cost_normal_keyboard : ℝ) (cost_normal_scissors : ℝ) : ℝ :=
  (cost_left_handed_mouse cost_normal_mouse + cost_left_handed_keyboard cost_normal_keyboard + cost_left_handed_scissors cost_normal_scissors) * 0.9

noncomputable def daily_sales_bundle (cost_normal_mouse : ℝ) (cost_normal_keyboard : ℝ) (cost_normal_scissors : ℝ) : ℝ :=
  5 * bundle_price cost_normal_mouse cost_normal_keyboard cost_normal_scissors

noncomputable def weekly_sales (cost_normal_mouse : ℝ) (cost_normal_keyboard : ℝ) (cost_normal_scissors : ℝ) : ℝ :=
  3 * (daily_sales_mouse cost_normal_mouse + daily_sales_keyboard cost_normal_keyboard + daily_sales_scissors cost_normal_scissors + daily_sales_bundle cost_normal_mouse cost_normal_keyboard cost_normal_scissors) +
  1.5 * (daily_sales_mouse cost_normal_mouse + daily_sales_keyboard cost_normal_keyboard + daily_sales_scissors cost_normal_scissors + daily_sales_bundle cost_normal_mouse cost_normal_keyboard cost_normal_scissors)

theorem weekly_sales_correct :
  weekly_sales 120 80 30 = 29922.25 := sorry

end weekly_sales_correct_l169_169236


namespace john_paid_after_tax_l169_169556

-- Definitions based on problem conditions
def original_cost : ℝ := 200
def tax_rate : ℝ := 0.15

-- Definition of the tax amount
def tax_amount : ℝ := tax_rate * original_cost

-- Definition of the total amount paid
def total_amount_paid : ℝ := original_cost + tax_amount

-- Theorem statement for the proof
theorem john_paid_after_tax : total_amount_paid = 230 := by
  sorry

end john_paid_after_tax_l169_169556


namespace rectangular_field_area_l169_169871

noncomputable def a : ℝ := 14
noncomputable def c : ℝ := 17
noncomputable def b := Real.sqrt (c^2 - a^2)
noncomputable def area := a * b

theorem rectangular_field_area : area = 14 * Real.sqrt 93 := by
  sorry

end rectangular_field_area_l169_169871


namespace min_photos_l169_169049

def num_monuments := 3
def num_tourists := 42

def took_photo_of (T M : Type) := T → M → Prop
def photo_coverage (T M : Type) (f : took_photo_of T M) :=
  ∀ (t1 t2 : T) (m : M), f t1 m ∨ f t2 m

def minimal_photos (T M : Type) [Fintype T] [Fintype M] [DecidableEq M]
  (f : took_photo_of T M) :=
  ∑ m, card { t : T | f t m }

theorem min_photos {T : Type} [Fintype T] [DecidableEq T] (M : fin num_monuments) (f : took_photo_of T M)
  (hT : card T = num_tourists)
  (h_cov : photo_coverage T M f) :
  minimal_photos T M f ≥ 123 :=
sorry

end min_photos_l169_169049


namespace min_q_of_abs_poly_eq_three_l169_169777

theorem min_q_of_abs_poly_eq_three (p q : ℝ) :
  (∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ (|x1^2 + p * x1 + q| = 3) ∧ (|x2^2 + p * x2 + q| = 3) ∧ (|x3^2 + p * x3 + q| = 3)) →
  q = -3 :=
sorry

end min_q_of_abs_poly_eq_three_l169_169777


namespace num_intersecting_chords_on_circle_l169_169573

theorem num_intersecting_chords_on_circle (points : Fin 20 → Prop) : 
  ∃ num_chords : ℕ, num_chords = 156180 :=
by
  sorry

end num_intersecting_chords_on_circle_l169_169573


namespace car_fuel_efficiency_in_city_l169_169473

theorem car_fuel_efficiency_in_city 
    (H C T : ℝ) 
    (h1 : H * T = 462) 
    (h2 : (H - 15) * T = 336) : 
    C = 40 :=
by 
    sorry

end car_fuel_efficiency_in_city_l169_169473


namespace cost_of_gas_per_gallon_l169_169063

-- Definitions based on the conditions
def hours_driven_1 : ℕ := 2
def speed_1 : ℕ := 60
def hours_driven_2 : ℕ := 3
def speed_2 : ℕ := 50
def mileage_per_gallon : ℕ := 30
def total_gas_cost : ℕ := 18

-- An assumption to simplify handling dollars and gallons
noncomputable def cost_per_gallon : ℕ := total_gas_cost / (speed_1 * hours_driven_1 + speed_2 * hours_driven_2) * mileage_per_gallon

theorem cost_of_gas_per_gallon :
  cost_per_gallon = 2 := by
sorry

end cost_of_gas_per_gallon_l169_169063


namespace books_total_correct_l169_169428

-- Define the constants for the number of books obtained each day
def books_day1 : ℕ := 54
def books_day2_total : ℕ := 23
def books_day2_kept : ℕ := 12
def books_day3_multiplier : ℕ := 3

-- Calculate the total number of books obtained each day
def books_day3 := books_day3_multiplier * books_day2_total
def total_books := books_day1 + books_day2_kept + books_day3

-- The theorem to prove
theorem books_total_correct : total_books = 135 := by
  sorry

end books_total_correct_l169_169428


namespace eccentricity_of_hyperbola_l169_169916

-- Define the conditions
def hyperbola_eccentricity (a b : ℝ) := sqrt (1 + (b^2 / a^2))

theorem eccentricity_of_hyperbola :
  (hyperbola_eccentricity (sqrt 2) (sqrt 3)) = sqrt 5 / 2 :=
by
  sorry

end eccentricity_of_hyperbola_l169_169916


namespace no_nat_solutions_for_m2_eq_n2_plus_2014_l169_169363

theorem no_nat_solutions_for_m2_eq_n2_plus_2014 :
  ∀ m n : ℕ, ¬(m^2 = n^2 + 2014) := by
sorry

end no_nat_solutions_for_m2_eq_n2_plus_2014_l169_169363


namespace choose_4_captains_from_15_l169_169542

def num_ways_to_choose_captains : ℕ := 15.choose 4

theorem choose_4_captains_from_15 : num_ways_to_choose_captains = 1365 := by
  sorry

end choose_4_captains_from_15_l169_169542


namespace ab_value_l169_169881

theorem ab_value (a b : ℝ) (h1 : a + b = 4) (h2 : a^2 + b^2 = 18) : a * b = -1 :=
by sorry

end ab_value_l169_169881


namespace balloons_lost_is_correct_l169_169691

def original_balloons : ℕ := 8
def current_balloons : ℕ := 6
def lost_balloons : ℕ := original_balloons - current_balloons

theorem balloons_lost_is_correct : lost_balloons = 2 := by
  sorry

end balloons_lost_is_correct_l169_169691


namespace horses_added_l169_169058

-- Define the problem parameters and conditions.
def horses_initial := 3
def water_per_horse_drinking_per_day := 5
def water_per_horse_bathing_per_day := 2
def days := 28
def total_water := 1568

-- Define the assumption based on the given problem.
def total_water_per_horse_per_day := water_per_horse_drinking_per_day + water_per_horse_bathing_per_day
def total_water_initial_horses := horses_initial * total_water_per_horse_per_day * days
def water_for_new_horses := total_water - total_water_initial_horses
def daily_water_consumption_new_horses := water_for_new_horses / days
def number_of_new_horses := daily_water_consumption_new_horses / total_water_per_horse_per_day

-- The theorem to prove number of horses added.
theorem horses_added : number_of_new_horses = 5 := 
  by {
    -- This is where you would put the proof steps.
    sorry -- skipping the proof for now
  }

end horses_added_l169_169058


namespace find_constant_t_l169_169371

theorem find_constant_t :
  (exists t : ℚ,
  ∀ x : ℚ,
    (5 * x ^ 2 - 6 * x + 7) * (4 * x ^ 2 + t * x + 10) =
      20 * x ^ 4 - 48 * x ^ 3 + 114 * x ^ 2 - 102 * x + 70) :=
sorry

end find_constant_t_l169_169371


namespace solve_for_m_l169_169928

open Real

theorem solve_for_m (a b m : ℝ)
  (h1 : (1/2)^a = m)
  (h2 : 3^b = m)
  (h3 : 1/a - 1/b = 2) :
  m = sqrt 6 / 6 := 
  sorry

end solve_for_m_l169_169928


namespace average_temperature_l169_169923

theorem average_temperature :
  let temp1 := -36
  let temp2 := 13
  let temp3 := -15
  let temp4 := -10
  (temp1 + temp2 + temp3 + temp4) / 4 = -12 :=
by
  unfold temp1 temp2 temp3 temp4
  calc
    (-36 + 13 + -15 + -10) / 4
      = (-48) / 4 : by norm_num
      ... = -12 : by norm_num

end average_temperature_l169_169923


namespace modular_inverse_l169_169643

/-- Define the number 89 -/
def a : ℕ := 89

/-- Define the modulus 90 -/
def n : ℕ := 90

/-- The condition given in the problem -/
lemma pow_mod (h : a ≡ -1 [MOD n]) : (a * a) % n = 1 % n := by 
  sorry

/-- The main statement to prove the modular inverse -/
theorem modular_inverse (h : a ≡ -1 [MOD n]) : (a * a) % n = 1 % n → a ≡ a⁻¹ [MOD n] := by
  intro h1
  have h2 : a⁻¹ % n = a % n := by 
    sorry
  exact h2

end modular_inverse_l169_169643


namespace stream_speed_l169_169474

theorem stream_speed (v : ℝ) (h1 : 36 > 0) (h2 : 80 > 0) (h3 : 40 > 0) (t_down : 80 / (36 + v) = 40 / (36 - v)) : v = 12 := 
by
  sorry

end stream_speed_l169_169474


namespace trigonometric_identity_tangent_line_l169_169681

theorem trigonometric_identity_tangent_line 
  (α : ℝ) 
  (h_tan : Real.tan α = 4) 
  : Real.cos α ^ 2 - Real.sin (2 * α) = - 7 / 17 := 
by sorry

end trigonometric_identity_tangent_line_l169_169681


namespace no_nat_solutions_m_sq_eq_n_sq_plus_2014_l169_169348

theorem no_nat_solutions_m_sq_eq_n_sq_plus_2014 :
  ¬ ∃ (m n : ℕ), m ^ 2 = n ^ 2 + 2014 := 
sorry

end no_nat_solutions_m_sq_eq_n_sq_plus_2014_l169_169348


namespace total_property_price_l169_169997

theorem total_property_price :
  let price_per_sqft : ℝ := 98
  let house_sqft : ℝ := 2400
  let barn_sqft : ℝ := 1000
  let house_price : ℝ := house_sqft * price_per_sqft
  let barn_price : ℝ := barn_sqft * price_per_sqft
  let total_price : ℝ := house_price + barn_price
  total_price = 333200 := by
  sorry

end total_property_price_l169_169997


namespace probability_valid_pairs_is_correct_l169_169430

open Finset

def valid_pairs : Finset (ℕ × ℕ) :=
  { (1, 3), (1, 5), (2, 4), (2, 6) }

def all_pairs : Finset (ℕ × ℕ) :=
  (range 6).product (range 6) \ { (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6) }

noncomputable def probability_valid_pairs : ℚ :=
  (4 : ℚ) / 15

theorem probability_valid_pairs_is_correct :
  (all_pairs.card = 15) ∧ 
  (valid_pairs.card = 4) ∧ 
  (valid_pairs.card : ℚ) / (all_pairs.card : ℚ) = probability_valid_pairs :=
by
  sorry

end probability_valid_pairs_is_correct_l169_169430


namespace calum_disco_ball_budget_l169_169291

-- Defining the conditions
def n_d : ℕ := 4  -- Number of disco balls
def n_f : ℕ := 10  -- Number of food boxes
def p_f : ℕ := 25  -- Price per food box in dollars
def B : ℕ := 330  -- Total budget in dollars

-- Defining the expected result
def p_d : ℕ := 20  -- Cost per disco ball in dollars

-- Proof statement (no proof, just the statement)
theorem calum_disco_ball_budget :
  (10 * p_f + 4 * p_d = B) → (p_d = 20) :=
by
  sorry

end calum_disco_ball_budget_l169_169291


namespace sample_size_correct_l169_169387

-- Definitions derived from conditions in a)
def total_employees : ℕ := 120
def male_employees : ℕ := 90
def sampled_male_employees : ℕ := 18

-- Theorem stating the mathematically equivalent proof problem
theorem sample_size_correct : 
  ∃ (sample_size : ℕ), sample_size = (total_employees * (sampled_male_employees / male_employees)) :=
sorry

end sample_size_correct_l169_169387


namespace trees_to_plant_l169_169608

def road_length : ℕ := 156
def interval : ℕ := 6
def trees_needed (road_length interval : ℕ) := road_length / interval + 1

theorem trees_to_plant : trees_needed road_length interval = 27 := by
  sorry

end trees_to_plant_l169_169608


namespace points_four_units_away_l169_169700

theorem points_four_units_away (x : ℤ) : (x - (-1) = 4 ∨ x - (-1) = -4) ↔ (x = 3 ∨ x = -5) :=
by
  sorry

end points_four_units_away_l169_169700


namespace find_cos_alpha_l169_169659

noncomputable def cos_alpha_satisfies_condition (α : ℝ) : Prop :=
  (1 - Real.cos α) / Real.sin α = 3

theorem find_cos_alpha (α : ℝ) 
  (h : cos_alpha_satisfies_condition α) : Real.cos α = -4 / 5 :=
by
  sorry

end find_cos_alpha_l169_169659


namespace product_of_digits_of_nondivisible_by_5_number_is_30_l169_169755

-- Define the four-digit numbers
def numbers : List ℕ := [4825, 4835, 4845, 4855, 4865]

-- Define units and tens digit function
def units_digit (n : ℕ) := n % 10
def tens_digit (n : ℕ) := (n / 10) % 10

-- Assertion that 4865 is the number that is not divisible by 5
def not_divisible_by_5 (n : ℕ) : Prop := ¬ (units_digit n = 5 ∨ units_digit n = 0)

-- Lean 4 statement to prove the product of units and tens digit of the number not divisible by 5 is 30
theorem product_of_digits_of_nondivisible_by_5_number_is_30 :
  ∃ n ∈ numbers, not_divisible_by_5 n ∧ (units_digit n) * (tens_digit n) = 30 :=
by
  sorry

end product_of_digits_of_nondivisible_by_5_number_is_30_l169_169755


namespace converse_false_inverse_false_l169_169020

-- Definitions of the conditions
def is_rhombus (Q : Type) : Prop := -- definition of a rhombus
  sorry

def is_parallelogram (Q : Type) : Prop := -- definition of a parallelogram
  sorry

variable {Q : Type}

-- Initial statement: If a quadrilateral is a rhombus, then it is a parallelogram.
axiom initial_statement : is_rhombus Q → is_parallelogram Q

-- Goals: Prove both the converse and inverse are false
theorem converse_false : ¬ ((is_parallelogram Q) → (is_rhombus Q)) :=
sorry

theorem inverse_false : ¬ (¬ (is_rhombus Q) → ¬ (is_parallelogram Q)) :=
    sorry

end converse_false_inverse_false_l169_169020


namespace rectangle_width_decrease_l169_169535

theorem rectangle_width_decrease (a b : ℝ) (p x : ℝ) 
  (hp : p ≥ 0) (hx : x ≥ 0)
  (area_eq : a * b = (a * (1 + p / 100)) * (b * (1 - x / 100))) :
  x = (100 * p) / (100 + p) := 
by
  sorry

end rectangle_width_decrease_l169_169535


namespace part_a_part_b_l169_169705

-- Let γ and δ represent acute angles, γ < δ implies γ - sin γ < δ - sin δ 
theorem part_a (alpha beta : ℝ) (h_alpha : 0 < alpha) (h_alpha2 : alpha < π/2) 
  (h_beta : 0 < beta) (h_beta2 : beta < π/2) (h : alpha < beta) : 
  alpha - Real.sin alpha < beta - Real.sin beta := sorry

-- Let γ and δ represent acute angles, γ < δ implies tan γ - γ < tan δ - δ 
theorem part_b (alpha beta : ℝ) (h_alpha : 0 < alpha) (h_alpha2 : alpha < π/2) 
  (h_beta : 0 < beta) (h_beta2 : beta < π/2) (h : alpha < beta) : 
  Real.tan alpha - alpha < Real.tan beta - beta := sorry

end part_a_part_b_l169_169705


namespace max_cookies_Andy_eats_l169_169889

theorem max_cookies_Andy_eats (cookies_total : ℕ) (h_cookies_total : cookies_total = 30) 
  (exists_pos_a : ∃ a : ℕ, a > 0 ∧ 3 * a = 30 - a ∧ (∃ k : ℕ, 3 * a = k ∧ ∃ m : ℕ, a = m)) 
  : ∃ max_a : ℕ, max_a ≤ 7 ∧ 3 * max_a < cookies_total ∧ 3 * max_a ∣ cookies_total ∧ max_a = 6 :=
by
  sorry

end max_cookies_Andy_eats_l169_169889


namespace trains_meet_at_9am_l169_169127

-- Definitions of conditions
def distance_AB : ℝ := 65
def speed_train_A : ℝ := 20
def speed_train_B : ℝ := 25
def start_time_train_A : ℝ := 7
def start_time_train_B : ℝ := 8

-- This function calculates the meeting time of the two trains
noncomputable def meeting_time (distance_AB : ℝ) (speed_train_A : ℝ) (speed_train_B : ℝ) 
    (start_time_train_A : ℝ) (start_time_train_B : ℝ) : ℝ :=
  let distance_train_A := speed_train_A * (start_time_train_B - start_time_train_A)
  let remaining_distance := distance_AB - distance_train_A
  let relative_speed := speed_train_A + speed_train_B
  start_time_train_B + remaining_distance / relative_speed

-- Theorem stating the time when the two trains meet
theorem trains_meet_at_9am :
    meeting_time distance_AB speed_train_A speed_train_B start_time_train_A start_time_train_B = 9 := sorry

end trains_meet_at_9am_l169_169127


namespace inverse_mod_l169_169164

theorem inverse_mod (a b n : ℕ) (h : (a * b) % n = 1) : b % n = a⁻¹ % n := sorry

example : ∃ x : ℕ, (10 * x) % 1729 = 1 ∧ x < 1729 :=
by
  use 1585
  have h₁ : (10 * 1585) % 1729 = 1 := by norm_num
  exact ⟨h₁, by norm_num⟩

end inverse_mod_l169_169164


namespace sampling_is_systematic_l169_169618

-- Defining the conditions
def mock_exam (rooms students_per_room seat_selected: ℕ) : Prop :=
  rooms = 80 ∧ students_per_room = 30 ∧ seat_selected = 15

-- Theorem statement
theorem sampling_is_systematic 
  (rooms students_per_room seat_selected: ℕ)
  (h: mock_exam rooms students_per_room seat_selected) : 
  sampling_method = "Systematic sampling" :=
sorry

end sampling_is_systematic_l169_169618


namespace find_X_plus_Y_in_base_8_l169_169003

theorem find_X_plus_Y_in_base_8 (X Y : ℕ) (h1 : 3 * 8^2 + X * 8 + Y + 5 * 8 + 2 = 4 * 8^2 + X * 8 + 3) : X + Y = 1 :=
sorry

end find_X_plus_Y_in_base_8_l169_169003


namespace speed_comparison_l169_169280

theorem speed_comparison (v v2 : ℝ) (h1 : v2 > 0) (h2 : v = 5 * v2) : v = 5 * v2 :=
by
  exact h2 

end speed_comparison_l169_169280


namespace range_of_x_range_of_a_l169_169565

variable {x a : ℝ}

def p (x a : ℝ) : Prop := x^2 - 4 * a * x + 3 * a^2 < 0
def q (x : ℝ) : Prop := (x - 3) / (x - 2) < 0

theorem range_of_x (h1 : a = 1) (h2 : p x 1 ∧ q x) : 2 < x ∧ x < 3 := by
  sorry

theorem range_of_a (h : ∀ x, p x a → q x) : 1 ≤ a ∧ a ≤ 2 := by
  sorry

end range_of_x_range_of_a_l169_169565


namespace quadratic_intersects_x_axis_iff_l169_169679

theorem quadratic_intersects_x_axis_iff (m : ℝ) : 
  (∃ x : ℝ, x^2 + 2 * x - m = 0) ↔ m ≥ -1 := 
by
  sorry

end quadratic_intersects_x_axis_iff_l169_169679


namespace no_nat_solutions_for_m2_eq_n2_plus_2014_l169_169356

theorem no_nat_solutions_for_m2_eq_n2_plus_2014 :
  ∀ m n : ℕ, ¬(m^2 = n^2 + 2014) := by
sorry

end no_nat_solutions_for_m2_eq_n2_plus_2014_l169_169356


namespace calum_spend_per_disco_ball_l169_169293

def calum_budget := 330
def food_cost_per_box := 25
def number_of_food_boxes := 10
def number_of_disco_balls := 4

theorem calum_spend_per_disco_ball : (calum_budget - food_cost_per_box * number_of_food_boxes) / number_of_disco_balls = 20 :=
by
  sorry

end calum_spend_per_disco_ball_l169_169293


namespace relationship_of_y_values_l169_169951

def parabola_y (x : ℝ) (c : ℝ) : ℝ :=
  2 * (x + 1)^2 + c

theorem relationship_of_y_values (c : ℝ) (y1 y2 y3 : ℝ) :
  y1 = parabola_y (-2) c →
  y2 = parabola_y 1 c →
  y3 = parabola_y 2 c →
  y3 > y2 ∧ y2 > y1 :=
by
  intros h1 h2 h3
  sorry

end relationship_of_y_values_l169_169951


namespace krishan_money_l169_169251

theorem krishan_money (R G K : ℕ) 
  (h_ratio1 : R * 17 = G * 7) 
  (h_ratio2 : G * 17 = K * 7) 
  (h_R : R = 735) : 
  K = 4335 := 
sorry

end krishan_money_l169_169251


namespace eggs_divided_l169_169840

theorem eggs_divided (boxes : ℝ) (eggs_per_box : ℝ) (total_eggs : ℝ) :
  boxes = 2.0 → eggs_per_box = 1.5 → total_eggs = boxes * eggs_per_box → total_eggs = 3.0 :=
by
  intros
  sorry

end eggs_divided_l169_169840


namespace pages_left_to_read_l169_169080

-- Define the conditions
def total_pages : ℕ := 400
def percent_read : ℚ := 20 / 100
def pages_read := total_pages * percent_read

-- Define the question as a theorem
theorem pages_left_to_read (total_pages : ℕ) (percent_read : ℚ) (pages_read : ℚ) : ℚ :=
total_pages - pages_read

-- Assert the correct answer
example : pages_left_to_read total_pages percent_read pages_read = 320 := 
by
  sorry

end pages_left_to_read_l169_169080


namespace no_nat_m_n_square_diff_2014_l169_169333

theorem no_nat_m_n_square_diff_2014 :
  ¬ ∃ (m n : ℕ), m^2 = n^2 + 2014 :=
by
  sorry

end no_nat_m_n_square_diff_2014_l169_169333


namespace no_nat_solutions_m2_eq_n2_plus_2014_l169_169314

theorem no_nat_solutions_m2_eq_n2_plus_2014 :
  ¬ ∃ (m n : ℕ), m^2 = n^2 + 2014 :=
by
  sorry

end no_nat_solutions_m2_eq_n2_plus_2014_l169_169314


namespace total_marbles_l169_169746

/-- A craftsman makes 35 jars. This is exactly 2.5 times the number of clay pots he made.
If each jar has 5 marbles and each clay pot has four times as many marbles as the jars plus an additional 3 marbles, 
prove that the total number of marbles is 497. -/
theorem total_marbles (number_of_jars : ℕ) (number_of_clay_pots : ℕ) (marbles_in_jar : ℕ) (marbles_in_clay_pot : ℕ) :
  number_of_jars = 35 →
  (number_of_jars : ℝ) = 2.5 * number_of_clay_pots →
  marbles_in_jar = 5 →
  marbles_in_clay_pot = 4 * marbles_in_jar + 3 →
  (number_of_jars * marbles_in_jar + number_of_clay_pots * marbles_in_clay_pot) = 497 :=
by 
  sorry

end total_marbles_l169_169746


namespace expression_eval_l169_169286

theorem expression_eval : (-4)^7 / 4^5 + 5^3 * 2 - 7^2 = 185 := by
  sorry

end expression_eval_l169_169286


namespace jackson_hermit_crabs_l169_169059

theorem jackson_hermit_crabs (H : ℕ) (total_souvenirs : ℕ) 
  (h1 : total_souvenirs = H + 3 * H + 6 * H) 
  (h2 : total_souvenirs = 450) : H = 45 :=
by {
  sorry
}

end jackson_hermit_crabs_l169_169059


namespace total_books_l169_169578

def sam_books := 110
def joan_books := 102
def tom_books := 125
def alice_books := 97

theorem total_books : sam_books + joan_books + tom_books + alice_books = 434 :=
by
  sorry

end total_books_l169_169578


namespace find_a_l169_169960

theorem find_a (a : ℝ) : 
  (∃ (r : ℕ), r = 3 ∧ 
  ((-1)^r * (Nat.choose 5 r : ℝ) * a^(5 - r) = -40)) ↔ a = 2 ∨ a = -2 :=
by
    sorry

end find_a_l169_169960


namespace no_infinite_arithmetic_progression_l169_169821

open Classical

variable {R : Type*} [LinearOrderedField R]

noncomputable def f (x : R) : R := sorry

theorem no_infinite_arithmetic_progression
  (f_strict_inc : ∀ x y : R, 0 < x ∧ 0 < y → x < y → f x < f y)
  (f_convex : ∀ x y : R, 0 < x ∧ 0 < y → f ((x + y) / 2) < (f x + f y) / 2) :
  ∀ a : ℕ → R, (∀ n : ℕ, a n = f n) → ¬(∃ d : R, ∀ k : ℕ, a (k + 1) - a k = d) :=
sorry

end no_infinite_arithmetic_progression_l169_169821


namespace unique_solution_of_equation_l169_169028

theorem unique_solution_of_equation (x y : ℝ) (h : |x + 2| + (y - 1)^2 = 0) : x = -2 ∧ y = 1 :=
by
  sorry

end unique_solution_of_equation_l169_169028


namespace cos_half_pi_plus_alpha_correct_l169_169187

noncomputable def cos_half_pi_plus_alpha
  (α : ℝ)
  (h1 : Real.tan α = -3/4)
  (h2 : α ∈ Set.Ioo (3 * Real.pi / 2) (2 * Real.pi)) : Real :=
  Real.cos (Real.pi / 2 + α)

theorem cos_half_pi_plus_alpha_correct
  (α : ℝ)
  (h1 : Real.tan α = -3/4)
  (h2 : α ∈ Set.Ioo (3 * Real.pi / 2) (2 * Real.pi)) :
  cos_half_pi_plus_alpha α h1 h2 = 3/5 := by
  sorry

end cos_half_pi_plus_alpha_correct_l169_169187


namespace roots_modulus_less_than_one_l169_169553

theorem roots_modulus_less_than_one
  (A B C D : ℝ)
  (h1 : ∀ x, x^2 + A * x + B = 0 → |x| < 1)
  (h2 : ∀ x, x^2 + C * x + D = 0 → |x| < 1) :
  ∀ x, x^2 + (A + C) / 2 * x + (B + D) / 2 = 0 → |x| < 1 :=
by
  sorry

end roots_modulus_less_than_one_l169_169553


namespace initial_bananas_each_child_l169_169832

-- Define the variables and conditions.
def total_children : ℕ := 320
def absent_children : ℕ := 160
def present_children := total_children - absent_children
def extra_bananas : ℕ := 2

-- We are to prove the initial number of bananas each child was supposed to get.
theorem initial_bananas_each_child (B : ℕ) (x : ℕ) :
  B = total_children * x ∧ B = present_children * (x + extra_bananas) → x = 2 :=
by
  sorry

end initial_bananas_each_child_l169_169832


namespace inequality_proof_l169_169930

-- Define the context of non-negative real numbers and sum to 1
variable {x y z : ℝ}
variable (h_nonneg : 0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z)
variable (h_sum : x + y + z = 1)

-- State the theorem to be proved
theorem inequality_proof (h_nonneg : 0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z) (h_sum : x + y + z = 1) :
    0 ≤ x * y + y * z + z * x - 2 * x * y * z ∧ x * y + y * z + z * x - 2 * x * y * z ≤ 7 / 27 :=
    sorry

end inequality_proof_l169_169930


namespace negation_proposition_l169_169587

theorem negation_proposition : ¬(∀ x : ℝ, x > 0 → x ≥ 1) ↔ ∃ x : ℝ, x > 0 ∧ x < 1 := 
by
  sorry

end negation_proposition_l169_169587


namespace inverse_of_B_cubed_l169_169376

variable (B : Matrix (Fin 2) (Fin 2) ℝ)
def B_inv := Matrix.of ![![3, -2], ![0, -1]]
noncomputable def B_cubed_inv := ((B_inv) 3)^3

theorem inverse_of_B_cubed :
  B_inv = Matrix.of ![![27, -24], ![0, -1]] :=
by
  sorry

end inverse_of_B_cubed_l169_169376


namespace range_of_a_l169_169385

variable (A B : Set ℝ)
variable (a : ℝ)

def setA : Set ℝ := {x | x < -1 ∨ x ≥ 1}
def setB (a : ℝ) : Set ℝ := {x | x ≤ 2 * a ∨ x ≥ a + 1}

theorem range_of_a (a : ℝ) :
  (compl (setB a) ⊆ setA) ↔ (a ≤ -2 ∨ (1 / 2 ≤ a ∧ a < 1)) :=
by
  sorry

end range_of_a_l169_169385


namespace part1_solution_part2_solution_l169_169392

section Part1

noncomputable def f (x : ℝ) : ℝ := x^2 + x - 6

theorem part1_solution (x : ℝ) : f x > 0 ↔ x < -3 ∨ x > 2 :=
sorry

end Part1

section Part2

variables (a : ℝ) (ha : a < 0)
noncomputable def g (x : ℝ) : ℝ := a*x^2 + (3 - 2*a)*x - 6

theorem part2_solution (x : ℝ) :
  if h1 : a < -3/2 then g x < 0 ↔ x < -3/a ∨ x > 2
  else if h2 : a = -3/2 then g x < 0 ↔ x ≠ 2
  else -3/2 < a ∧ a < 0 → g x < 0 ↔ x < 2 ∨ x > -3/a :=
sorry

end Part2

end part1_solution_part2_solution_l169_169392


namespace find_english_score_l169_169697

-- Define the scores
def M : ℕ := 82
def K : ℕ := M + 5
variable (E : ℕ)

-- The average score condition
axiom avg_condition : (K + E + M) / 3 = 89

-- Our goal is to prove that E = 98
theorem find_english_score : E = 98 :=
by
  -- The proof will go here
  sorry

end find_english_score_l169_169697


namespace vector_sum_l169_169671

-- Define the vectors a and b according to the conditions.
def a : (ℝ × ℝ) := (2, 1)
def b : (ℝ × ℝ) := (-3, 4)

-- Prove that the vector sum a + b is (-1, 5).
theorem vector_sum : (a.1 + b.1, a.2 + b.2) = (-1, 5) :=
by
  -- include the proof later
  sorry

end vector_sum_l169_169671


namespace minimum_photos_l169_169048

-- Given conditions:
-- 1. There are exactly three monuments.
-- 2. A group of 42 tourists visited the city.
-- 3. Each tourist took at most one photo of each of the three monuments.
-- 4. Any two tourists together had photos of all three monuments.

theorem minimum_photos (n m : ℕ) (h₁ : n = 42) (h₂ : m = 3) 
  (photographs : ℕ → ℕ → Bool) -- whether tourist i took a photo of monument j
  (h₃ : ∀ i : ℕ, i < n → ∀ j : ℕ, j < m → photographs i j = tt ∨ photographs i j = ff) -- each tourist took at most one photo of each monument
  (h₄ : ∀ i₁ i₂ : ℕ, i₁ < n → i₂ < n → i₁ ≠ i₂ → ∀ j : ℕ, j < m → photographs i₁ j = tt ∨ photographs i₂ j = tt) -- any two tourists together had photos of all three monuments
  : ∃ min_photos : ℕ, min_photos = 123 :=
by
  sorry

end minimum_photos_l169_169048


namespace bobby_finishes_candies_in_weeks_l169_169160

def total_candies (packets: Nat) (candies_per_packet: Nat) : Nat := packets * candies_per_packet

def candies_eaten_per_week (candies_per_day_mon_fri: Nat) (days_mon_fri: Nat) (candies_per_day_weekend: Nat) (days_weekend: Nat) : Nat :=
  (candies_per_day_mon_fri * days_mon_fri) + (candies_per_day_weekend * days_weekend)

theorem bobby_finishes_candies_in_weeks :
  let packets := 2
  let candies_per_packet := 18
  let candies_per_day_mon_fri := 2
  let days_mon_fri := 5
  let candies_per_day_weekend := 1
  let days_weekend := 2

  total_candies packets candies_per_packet / candies_eaten_per_week candies_per_day_mon_fri days_mon_fri candies_per_day_weekend days_weekend = 3 :=
by
  sorry

end bobby_finishes_candies_in_weeks_l169_169160


namespace all_rationals_in_A_l169_169429

noncomputable def f (n : ℕ) : ℚ := (n-1)/(n+2)

def A : Set ℚ := { q | ∃ (s : Finset ℕ), q = s.sum f }

theorem all_rationals_in_A : A = Set.univ :=
by
  sorry

end all_rationals_in_A_l169_169429


namespace calculate_value_l169_169014

theorem calculate_value 
  (a : Int) (b : Int) (c : Real) (d : Real)
  (h1 : a = -1)
  (h2 : b = 2)
  (h3 : c * d = 1) :
  a + b - c * d = 0 := 
by
  sorry

end calculate_value_l169_169014


namespace choose_4_captains_from_15_l169_169543

def num_ways_to_choose_captains : ℕ := 15.choose 4

theorem choose_4_captains_from_15 : num_ways_to_choose_captains = 1365 := by
  sorry

end choose_4_captains_from_15_l169_169543


namespace problem1_solutionset_problem2_minvalue_l169_169787

noncomputable def f (x : ℝ) : ℝ := 45 * abs (2 * x - 1)
noncomputable def g (x : ℝ) : ℝ := f x + f (x - 1)

theorem problem1_solutionset :
  {x : ℝ | 0 < x ∧ x < 2 / 3} = {x : ℝ | f x + abs (x + 1) < 2} :=
by
  sorry

theorem problem2_minvalue (a : ℝ) (m n : ℝ) (h : m + n = a ∧ m > 0 ∧ n > 0) :
  a = 2 → (4 / m + 1 / n) ≥ 9 / 2 :=
by
  sorry

end problem1_solutionset_problem2_minvalue_l169_169787


namespace blake_initial_amount_l169_169632

theorem blake_initial_amount (X : ℝ) (h1 : X > 0) (h2 : 3 * X / 2 = 30000) : X = 20000 :=
sorry

end blake_initial_amount_l169_169632


namespace iris_total_spending_l169_169056

theorem iris_total_spending :
  ∀ (price_jacket price_shorts price_pants : ℕ), 
  price_jacket = 10 → 
  price_shorts = 6 → 
  price_pants = 12 → 
  (3 * price_jacket + 2 * price_shorts + 4 * price_pants) = 90 :=
by
  intros price_jacket price_shorts price_pants
  sorry

end iris_total_spending_l169_169056


namespace evaluate_g_f_l169_169214

def f (a b : ℤ) : ℤ × ℤ := (-a, b)

def g (m n : ℤ) : ℤ × ℤ := (m, -n)

theorem evaluate_g_f : g (f 2 (-3)).1 (f 2 (-3)).2 = (-2, 3) := by
  sorry

end evaluate_g_f_l169_169214


namespace magnitude_of_z_l169_169389

noncomputable def z : ℂ := Complex.I * (3 + 4 * Complex.I)

theorem magnitude_of_z : Complex.abs z = 5 := by
  sorry

end magnitude_of_z_l169_169389


namespace find_x_l169_169470

theorem find_x (x : ℝ) (h : 3 * x = (20 - x) + 20) : x = 10 :=
sorry

end find_x_l169_169470


namespace tourist_total_value_l169_169488

theorem tourist_total_value
    (tax_rate : ℝ)
    (V : ℝ)
    (tax_paid : ℝ)
    (exempt_amount : ℝ) :
    exempt_amount = 600 ∧
    tax_rate = 0.07 ∧
    tax_paid = 78.4 →
    (tax_rate * (V - exempt_amount) = tax_paid) →
    V = 1720 :=
by
  intros h1 h2
  have h_exempt : exempt_amount = 600 := h1.left
  have h_tax_rate : tax_rate = 0.07 := h1.right.left
  have h_tax_paid : tax_paid = 78.4 := h1.right.right
  sorry

end tourist_total_value_l169_169488


namespace more_oranges_than_apples_l169_169227

-- Definitions based on conditions
def apples : ℕ := 14
def oranges : ℕ := 2 * 12  -- 2 dozen oranges

-- Statement to prove
theorem more_oranges_than_apples : oranges - apples = 10 := by
  sorry

end more_oranges_than_apples_l169_169227


namespace turtle_reaches_waterhole_28_minutes_after_meeting_l169_169741

theorem turtle_reaches_waterhole_28_minutes_after_meeting (x : ℝ) (distance_lion1 : ℝ := 5 * x) 
  (speed_lion2 : ℝ := 1.5 * x) (distance_turtle : ℝ := 30) (speed_turtle : ℝ := 1/30) : 
  ∃ t_meeting : ℝ, t_meeting = 2 ∧ (distance_turtle - speed_turtle * t_meeting) / speed_turtle = 28 :=
by 
  sorry

end turtle_reaches_waterhole_28_minutes_after_meeting_l169_169741


namespace no_solution_exists_l169_169304

theorem no_solution_exists (m n : ℕ) : ¬ (m^2 = n^2 + 2014) :=
by
  sorry

end no_solution_exists_l169_169304


namespace general_formula_for_sequence_l169_169932

noncomputable def S := ℕ → ℚ
noncomputable def a := ℕ → ℚ

theorem general_formula_for_sequence (a : a) (S : S) (h1 : a 1 = 2)
  (h2 : ∀ n : ℕ, S (n + 1) = (2 / 3) * a (n + 1) + 1 / 3) :
  ∀ n : ℕ, a n = 
  if n = 1 then 2 
  else -5 * (-2)^(n-2) := 
by 
  sorry

end general_formula_for_sequence_l169_169932


namespace find_E_l169_169030

variable (x E x1 x2 : ℝ)

/-- Given conditions as assumptions: -/
axiom h1 : (x + 3)^2 / E = 2
axiom h2 : x1 - x2 = 14

/-- Prove the required expression for E in terms of x: -/
theorem find_E : E = (x + 3)^2 / 2 := sorry

end find_E_l169_169030


namespace min_photographs_42_tourists_3_monuments_l169_169043

noncomputable def min_photos_taken (num_tourists : ℕ) (num_monuments : ℕ) : ℕ :=
  if num_tourists = 42 ∧ num_monuments = 3 then 123 else 0

-- Main statement:
theorem min_photographs_42_tourists_3_monuments : 
  (∀ (num_tourists num_monuments : ℕ), 
    num_tourists = 42 ∧ num_monuments = 3 → min_photos_taken num_tourists num_monuments = 123)
  := by
    sorry

end min_photographs_42_tourists_3_monuments_l169_169043


namespace annual_income_correct_l169_169126

def investment (amount : ℕ) := 6800
def dividend_rate (rate : ℕ) := 20
def stock_price (price : ℕ) := 136
def face_value : ℕ := 100
def calculate_annual_income (amount rate price value : ℕ) : ℕ := 
  let shares := amount / price
  let annual_income_per_share := value * rate / 100
  shares * annual_income_per_share

theorem annual_income_correct : calculate_annual_income (investment 6800) (dividend_rate 20) (stock_price 136) face_value = 1000 :=
by
  sorry

end annual_income_correct_l169_169126


namespace value_of_expression_l169_169238

variable (a b c : ℝ)

theorem value_of_expression (h1 : a ≠ 1 ∧ b ≠ 1 ∧ c ≠ 1)
                            (h2 : abc = 1)
                            (h3 : a^2 + b^2 + c^2 - ((1 / (a^2)) + (1 / (b^2)) + (1 / (c^2))) = 8 * (a + b + c) - 8 * (ab + bc + ca)) :
                            (1 / (a - 1)) + (1 / (b - 1)) + (1 / (c - 1)) = -3/2 :=
by
  sorry

end value_of_expression_l169_169238


namespace A_roster_method_l169_169669

open Set

def A : Set ℤ := {x : ℤ | (∃ (n : ℤ), n > 0 ∧ 6 / (5 - x) = n) }

theorem A_roster_method :
  A = {-1, 2, 3, 4} :=
  sorry

end A_roster_method_l169_169669


namespace no_elimination_method_l169_169604

theorem no_elimination_method
  (x y : ℤ)
  (h1 : x + 3 * y = 4)
  (h2 : 2 * x - y = 1) :
  ¬ (∀ z : ℤ, z = x + 3 * y - 3 * (2 * x - y)) →
  ∃ x y : ℤ, x + 3 * y - 3 * (2 * x - y) ≠ 0 := sorry

end no_elimination_method_l169_169604


namespace bag_weight_l169_169455

variable total_capacity : ℝ
variable fill_percentage : ℝ
variable additional_weight_factor : ℝ

-- Given conditions
axiom h1 : total_capacity = 250
axiom h2 : fill_percentage = 0.8
axiom h3 : additional_weight_factor = 0.4

-- Prove the weight of the bag
theorem bag_weight : 
  total_capacity * fill_percentage * (1 + additional_weight_factor) = 280 := by
  sorry

end bag_weight_l169_169455


namespace trigonometric_identity_example_l169_169612

open Real Real.Angle

theorem trigonometric_identity_example :
  sin (34 * π / 180) * sin (26 * π / 180) - cos (34 * π / 180) * cos (26 * π / 180) = - 1 / 2 :=
by
  -- Convert degrees to radians
  -- Proof goes here
  sorry

end trigonometric_identity_example_l169_169612


namespace minimum_photos_taken_l169_169045

theorem minimum_photos_taken (A B C : Type) [finite A] [finite B] [finite C] 
  (tourists : fin 42) 
  (photo : tourists → fin 3 → Prop)
  (h1 : ∀ t : tourists, ∀ m : fin 3, photo t m ∨ ¬photo t m) 
  (h2 : ∀ t1 t2 : tourists, ∃ m : fin 3, photo t1 m ∨ photo t2 m) :
  ∃(n : ℕ), n = 123 ∧ ∀ (photo_count : nat), photo_count = ∑ t in fin_range 42, ∑ m in fin_range 3, if photo t m then 1 else 0 → photo_count ≥ n :=
sorry

end minimum_photos_taken_l169_169045


namespace exist_n_exactly_3_rainy_days_l169_169961

-- Define the binomial coefficient
def binomial_coeff (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Define the binomial probability
def binomial_prob (n k : ℕ) (p : ℝ) : ℝ :=
  (binomial_coeff n k : ℝ) * p^k * (1 - p)^(n - k)

theorem exist_n_exactly_3_rainy_days (p : ℝ) (k : ℕ) (prob : ℝ) :
  p = 0.5 → k = 3 → prob = 0.25 →
  ∃ n : ℕ, binomial_prob n k p = prob :=
by
  intros h1 h2 h3
  sorry

end exist_n_exactly_3_rainy_days_l169_169961


namespace unique_real_solution_l169_169654

theorem unique_real_solution :
  ∃! x : ℝ, -((x + 2) ^ 2) ≥ 0 :=
sorry

end unique_real_solution_l169_169654


namespace is_odd_function_l169_169170

def f (x : ℝ) : ℝ := x^3 - x

theorem is_odd_function : ∀ x : ℝ, f (-x) = -f x :=
by
  intro x
  sorry

end is_odd_function_l169_169170


namespace polygon_exterior_angle_l169_169404

theorem polygon_exterior_angle (n : ℕ) (h : 36 = 360 / n) : n = 10 :=
sorry

end polygon_exterior_angle_l169_169404


namespace find_ϕ_l169_169201

noncomputable def f (ω ϕ : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + ϕ)

noncomputable def g (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x)

theorem find_ϕ (ω ϕ : ℝ) (h1 : 0 < ω) (h2 : abs ϕ < Real.pi / 2) (h3 : ∀ x : ℝ, f ω ϕ (x + Real.pi / 6) = g ω x) 
  (h4 : 2 * Real.pi / ω = Real.pi) : ϕ = Real.pi / 3 :=
by sorry

end find_ϕ_l169_169201


namespace percent_decrease_l169_169692

-- Definitions based on conditions
def originalPrice : ℝ := 100
def salePrice : ℝ := 10

-- The percentage decrease is the main statement to prove
theorem percent_decrease : ((originalPrice - salePrice) / originalPrice) * 100 = 90 := 
by
  -- Placeholder for proof
  sorry

end percent_decrease_l169_169692


namespace washing_water_use_l169_169107

variable (gallons_collected : Nat)
variable (gallons_per_car : Nat)
variable (num_cars : Nat)
variable (less_gallons_plants : Nat)

-- Here are the conditions provided in the problem
def initial_gallons_collected := gallons_collected = 65
def water_used_per_car := gallons_per_car = 7
def number_of_cars := num_cars = 2
def less_gallons_for_plants := less_gallons_plants = 11

-- Calculate total water used for cars
def total_water_cars := num_cars * gallons_per_car
-- Calculate water used for plants
def water_plants := total_water_cars - less_gallons_plants
-- Calculate total used for cars and plants
def total_water_cars_plants := total_water_cars + water_plants
-- Calculate remaining water
def remaining_water := gallons_collected - total_water_cars_plants
-- Calculate water used to wash plates and clothes
def water_plates_clothes := remaining_water / 2

-- The theorem to prove the problem statement
theorem washing_water_use (hg : initial_gallons_collected) (hwc : water_used_per_car) (hnc : number_of_cars) (hlp : less_gallons_for_plants) :
  water_plates_clothes = 24 :=
by
  -- Given conditions from the problem
  unfold initial_gallons_collected water_used_per_car number_of_cars less_gallons_for_plants at hg hwc hnc hlp
  -- Definition lies outside the immediate scope of main code, thus accuracy over proof is ensured
  sorry

end washing_water_use_l169_169107


namespace gumballs_difference_l169_169762

variable (x y : ℕ)

def total_gumballs := 16 + 12 + 20 + x + y
def avg_gumballs (T : ℕ) := T / 5

theorem gumballs_difference (h1 : 18 <= avg_gumballs (total_gumballs x y)) 
                            (h2 : avg_gumballs (total_gumballs x y) <= 27) : (87 - 42) = 45 := by
  sorry

end gumballs_difference_l169_169762


namespace simplify_and_evaluate_l169_169089

theorem simplify_and_evaluate (x : ℝ) (h₁ : x ≠ 0) (h₂ : x = 2) : 
  (1 + 1 / x) / ((x^2 - 1) / x) = 1 := 
by 
  sorry

end simplify_and_evaluate_l169_169089


namespace intersection_A_complement_B_l169_169423

-- Definitions of sets A and B and their complement in the universal set R, which is the real numbers.
def A : Set ℝ := {-1, 0, 1, 2, 3}
def B : Set ℝ := {x | x^2 - 2 * x > 0}
def complement_R_B : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}

-- The proof statement verifying the intersection of set A with the complement of set B.
theorem intersection_A_complement_B : A ∩ complement_R_B = {0, 1, 2} := by
  sorry

end intersection_A_complement_B_l169_169423


namespace sum_p_q_eq_21_l169_169529

theorem sum_p_q_eq_21 (p q : ℤ) :
  {x | x^2 + 6 * x - q = 0} ∩ {x | x^2 - p * x + 6 = 0} = {2} → p + q = 21 :=
by
  sorry

end sum_p_q_eq_21_l169_169529


namespace turtle_minimum_distance_l169_169886

/-- 
Given a turtle starting at the origin (0,0), crawling at a speed of 5 m/hour,
and turning 90 degrees at the end of each hour, prove that after 11 hours,
the minimum distance from the origin it could be is 5 meters.
-/
theorem turtle_minimum_distance :
  let speed := 5
  let hours := 11
  let distance (n : ℕ) := n * speed
  in ∃ (final_position : ℤ × ℤ),
      final_position = (5, 0) ∨ final_position = (-5, 0) ∨ final_position = (0, 5) ∨ final_position = (0, -5) :=
  sorry

end turtle_minimum_distance_l169_169886


namespace Jesse_pages_left_to_read_l169_169066

def pages_read := [10, 15, 27, 12, 19]
def total_pages_read := pages_read.sum
def fraction_read : ℚ := 1 / 3
def total_pages : ℚ := total_pages_read / fraction_read
def pages_left_to_read : ℚ := total_pages - total_pages_read

theorem Jesse_pages_left_to_read :
  pages_left_to_read = 166 := by
  sorry

end Jesse_pages_left_to_read_l169_169066


namespace graduates_continued_second_degree_l169_169686

noncomputable theory

def number_of_graduates : ℕ := 73
def graduates_found_job : ℕ := 32
def graduates_did_both : ℕ := 13
def graduates_did_neither : ℕ := 9

theorem graduates_continued_second_degree :
  let total := number_of_graduates in
  let job := graduates_found_job in
  let both := graduates_did_both in
  let neither := graduates_did_neither in
  ∃ s : ℕ, total - neither = 64 ∧ job - both = 19 ∧ 64 - 19 - both = 32 ∧ s = 32 + both ∧ s = 45 :=
begin
  sorry
end

end graduates_continued_second_degree_l169_169686


namespace arithmetic_sequence_sum_l169_169940

variable (S : ℕ → ℕ)   -- S is a function that gives the sum of the first k*n terms

theorem arithmetic_sequence_sum
  (n : ℕ)
  (h1 : S n = 45)
  (h2 : S (2 * n) = 60) :
  S (3 * n) = 65 := sorry

end arithmetic_sequence_sum_l169_169940


namespace Robert_can_read_one_book_l169_169838

def reading_speed : ℕ := 100 -- pages per hour
def book_length : ℕ := 350 -- pages
def available_time : ℕ := 5 -- hours

theorem Robert_can_read_one_book :
  (available_time * reading_speed) >= book_length ∧ 
  (available_time * reading_speed) < 2 * book_length :=
by {
  -- The proof steps are omitted as instructed.
  sorry
}

end Robert_can_read_one_book_l169_169838


namespace max_x_for_integer_fraction_l169_169466

theorem max_x_for_integer_fraction (x : ℤ) (h : ∃ k : ℤ, x^2 + 2 * x + 11 = k * (x - 3)) : x ≤ 29 :=
by {
    -- This is where the proof would be,
    -- but we skip the proof per the instructions.
    sorry
}

end max_x_for_integer_fraction_l169_169466


namespace ratio_HC_JE_l169_169082

noncomputable def A : ℝ := 0
noncomputable def B : ℝ := 1
noncomputable def C : ℝ := B + 2
noncomputable def D : ℝ := C + 1
noncomputable def E : ℝ := D + 1
noncomputable def F : ℝ := E + 2

variable (G H J K : ℝ × ℝ)
variable (parallel_AG_HC parallel_AG_JE parallel_AG_KB : Prop)

-- Conditions
axiom points_on_line : A < B ∧ B < C ∧ C < D ∧ D < E ∧ E < F
axiom AB : B - A = 1
axiom BC : C - B = 2
axiom CD : D - C = 1
axiom DE : E - D = 1
axiom EF : F - E = 2
axiom G_off_AF : G.2 ≠ 0
axiom H_on_GD : H.1 = G.1 ∧ H.2 = D
axiom J_on_GF : J.1 = G.1 ∧ J.2 = F
axiom K_on_GB : K.1 = G.1 ∧ K.2 = B
axiom parallel_hc_je_kb_ag : parallel_AG_HC ∧ parallel_AG_JE ∧ parallel_AG_KB ∧ (G.2 / 1) = (K.2 / (K.1 - G.1))

-- Task: Prove the ratio HC/JE = 7/8
theorem ratio_HC_JE : (H.2 - C) / (J.2 - E) = 7 / 8 :=
sorry

end ratio_HC_JE_l169_169082


namespace least_number_l169_169651

theorem least_number (n : ℕ) (h1 : n % 31 = 3) (h2 : n % 9 = 3) : n = 282 :=
sorry

end least_number_l169_169651


namespace set_difference_NM_l169_169768

open Set

def setDifference (A B : Set ℕ) : Set ℕ := {x | x ∈ A ∧ x ∉ B}

theorem set_difference_NM :
  let M := {1, 2, 3, 4, 5}
  let N := {1, 2, 3, 7}
  setDifference N M = {7} :=
by
  sorry

end set_difference_NM_l169_169768


namespace tax_percentage_l169_169296

theorem tax_percentage (car_price tax_paid first_tier_price : ℝ) (first_tier_tax_rate : ℝ) (tax_second_tier : ℝ) :
  car_price = 30000 ∧
  tax_paid = 5500 ∧
  first_tier_price = 10000 ∧
  first_tier_tax_rate = 0.25 ∧
  tax_second_tier = 0.15
  → (tax_second_tier) = 0.15 :=
by
  intros h
  rcases h with ⟨h1, h2, h3, h4, h5⟩
  sorry

end tax_percentage_l169_169296


namespace min_value_of_m_l169_169677

theorem min_value_of_m (a b c : ℝ) (h1 : a + b + c = 1) (h2 : a * b + b * c + c * a = -1) (h3 : a * b * c = -m) : 
    m = - (min (-a ^ 3 + a ^ 2 + a ) (- (1 / 27))) := 
sorry

end min_value_of_m_l169_169677


namespace largest_positive_real_root_bound_l169_169298

theorem largest_positive_real_root_bound (b0 b1 b2 : ℝ)
  (h_b0 : abs b0 ≤ 1) (h_b1 : abs b1 ≤ 1) (h_b2 : abs b2 ≤ 1) :
  ∃ r : ℝ, r > 0 ∧ r^3 + b2 * r^2 + b1 * r + b0 = 0 ∧ 1.5 < r ∧ r < 2 := 
sorry

end largest_positive_real_root_bound_l169_169298


namespace parabola_vertex_point_sum_l169_169177

theorem parabola_vertex_point_sum (a b c : ℚ) 
  (h1 : ∃ (a b c : ℚ), ∀ x : ℚ, (y = a * x ^ 2 + b * x + c) = (y = - (1 / 3) * (x - 5) ^ 2 + 3)) 
  (h2 : ∀ x : ℚ, ((x = 2) ∧ (y = 0)) → (0 = a * 2 ^ 2 + b * 2 + c)) :
  a + b + c = -7 / 3 := 
sorry

end parabola_vertex_point_sum_l169_169177


namespace work_done_by_student_l169_169129

theorem work_done_by_student
  (M : ℝ)  -- mass of the student
  (m : ℝ)  -- mass of the stone
  (h : ℝ)  -- height from which the stone is thrown
  (L : ℝ)  -- distance on the ice where the stone lands
  (g : ℝ)  -- acceleration due to gravity
  (t : ℝ := Real.sqrt (2 * h / g))  -- time it takes for the stone to hit the ice derived from free fall equation
  (Vk : ℝ := L / t)  -- initial speed of the stone derived from horizontal motion
  (Vu : ℝ := m / M * Vk)  -- initial speed of the student derived from conservation of momentum
  : (1/2 * m * Vk^2 + (1/2) * M * Vu^2) = 126.74 :=
by
  sorry

end work_done_by_student_l169_169129


namespace pounds_in_one_ton_is_2600_l169_169575

variable (pounds_in_one_ton : ℕ)
variable (ounces_in_one_pound : ℕ := 16)
variable (packets : ℕ := 2080)
variable (weight_per_packet_pounds : ℕ := 16)
variable (weight_per_packet_ounces : ℕ := 4)
variable (gunny_bag_capacity_tons : ℕ := 13)

theorem pounds_in_one_ton_is_2600 :
  (packets * (weight_per_packet_pounds + weight_per_packet_ounces / ounces_in_one_pound)) = (gunny_bag_capacity_tons * pounds_in_one_ton) →
  pounds_in_one_ton = 2600 :=
sorry

end pounds_in_one_ton_is_2600_l169_169575


namespace skating_rink_visitors_by_noon_l169_169624

-- Defining the initial conditions
def initial_visitors : ℕ := 264
def visitors_left : ℕ := 134
def visitors_arrived : ℕ := 150

-- Theorem to prove the number of people at the skating rink by noon
theorem skating_rink_visitors_by_noon : initial_visitors - visitors_left + visitors_arrived = 280 := 
by 
  sorry

end skating_rink_visitors_by_noon_l169_169624


namespace range_of_a_l169_169206

noncomputable def A (a : ℝ) : Set ℝ := {x | a - 1 < x ∧ x < 2 * a + 1}
noncomputable def B : Set ℝ := {x | 0 < x ∧ x < 1}

theorem range_of_a (a : ℝ) (h : A a ∩ B = ∅) : a ≤ -1 / 2 ∨ a ≥ 2 :=
  sorry

end range_of_a_l169_169206


namespace find_expression_value_l169_169008

theorem find_expression_value (a b : ℝ)
  (h1 : a^2 - a - 3 = 0)
  (h2 : b^2 - b - 3 = 0) :
  2 * a^3 + b^2 + 3 * a^2 - 11 * a - b + 5 = 23 :=
  sorry

end find_expression_value_l169_169008


namespace average_temperature_bucyrus_l169_169113

theorem average_temperature_bucyrus : 
  let t1 := -14
  let t2 := -8
  let t3 := 1 
  (t1 + t2 + t3) / 3 = -7 := 
  by
  rw [show t1 + t2 + t3 = -21 by norm_num]
  rw [show -21 / 3 = -7 by norm_num]
  sorry

end average_temperature_bucyrus_l169_169113


namespace solve_system_unique_solution_l169_169774

theorem solve_system_unique_solution:
  ∃! (x y : ℚ), 3 * x - 4 * y = -7 ∧ 4 * x + 5 * y = 23 ∧ x = 57 / 31 ∧ y = 97 / 31 := by
  sorry

end solve_system_unique_solution_l169_169774


namespace find_root_equation_l169_169917

theorem find_root_equation : ∃ x : ℤ, x - (5 / (x - 4)) = 2 - (5 / (x - 4)) ∧ x = 2 :=
by
  sorry

end find_root_equation_l169_169917


namespace hugo_probability_l169_169220

noncomputable def P_hugo_first_roll_seven_given_win (P_Hugo_wins : ℚ) (P_first_roll_seven : ℚ)
  (P_all_others_roll_less_than_seven : ℚ) : ℚ :=
(P_first_roll_seven * P_all_others_roll_less_than_seven) / P_Hugo_wins

theorem hugo_probability :
  let P_Hugo_wins := (1 : ℚ) / 4
  let P_first_roll_seven := (1 : ℚ) / 8
  let P_all_others_roll_less_than_seven := (27 : ℚ) / 64
  P_hugo_first_roll_seven_given_win P_Hugo_wins P_first_roll_seven P_all_others_roll_less_than_seven = (27 : ℚ) / 128 :=
by
  sorry

end hugo_probability_l169_169220


namespace tan_11pi_over_6_l169_169647

theorem tan_11pi_over_6 :
  Real.tan (11 * Real.pi / 6) = -1 / Real.sqrt 3 :=
by
  sorry

end tan_11pi_over_6_l169_169647


namespace sam_dimes_example_l169_169839

theorem sam_dimes_example (x y : ℕ) (h₁ : x = 9) (h₂ : y = 7) : x + y = 16 :=
by 
  sorry

end sam_dimes_example_l169_169839


namespace rectangle_area_l169_169534

theorem rectangle_area
  (L B : ℕ)
  (h1 : L - B = 23)
  (h2 : 2 * L + 2 * B = 186) : L * B = 2030 :=
sorry

end rectangle_area_l169_169534


namespace alicia_read_more_books_than_ian_l169_169905

def books_read : List Nat := [3, 5, 8, 6, 7, 4, 2, 1]

def alicia_books (books : List Nat) : Nat :=
  books.maximum?.getD 0

def ian_books (books : List Nat) : Nat :=
  books.minimum?.getD 0

theorem alicia_read_more_books_than_ian :
  alicia_books books_read - ian_books books_read = 7 :=
by
  -- By reviewing the given list of books read [3, 5, 8, 6, 7, 4, 2, 1]
  -- We find that alicia_books books_read = 8 and ian_books books_read = 1
  -- Thus, 8 - 1 = 7
  sorry

end alicia_read_more_books_than_ian_l169_169905


namespace k_polygonal_intersects_fermat_l169_169191

theorem k_polygonal_intersects_fermat (k : ℕ) (n m : ℕ) (h1: k > 2) 
  (h2 : ∃ n m, (k - 2) * n * (n - 1) / 2 + n = 2 ^ (2 ^ m) + 1) : 
  k = 3 ∨ k = 5 :=
  sorry

end k_polygonal_intersects_fermat_l169_169191


namespace find_cupcakes_l169_169424

def total_students : ℕ := 20
def treats_per_student : ℕ := 4
def cookies : ℕ := 20
def brownies : ℕ := 35
def total_treats : ℕ := total_students * treats_per_student
def cupcakes : ℕ := total_treats - (cookies + brownies)

theorem find_cupcakes : cupcakes = 25 := by
  sorry

end find_cupcakes_l169_169424


namespace ratio_of_tshirts_l169_169566

def spending_on_tshirts (Lisa_tshirts Carly_tshirts Lisa_jeans Lisa_coats Carly_jeans Carly_coats : ℝ) : Prop :=
  Lisa_tshirts = 40 ∧
  Lisa_jeans = Lisa_tshirts / 2 ∧
  Lisa_coats = 2 * Lisa_tshirts ∧
  Carly_jeans = 3 * Lisa_jeans ∧
  Carly_coats = Lisa_coats / 4 ∧
  Lisa_tshirts + Lisa_jeans + Lisa_coats + Carly_tshirts + Carly_jeans + Carly_coats = 230

theorem ratio_of_tshirts 
  (Lisa_tshirts Carly_tshirts Lisa_jeans Lisa_coats Carly_jeans Carly_coats : ℝ)
  (h : spending_on_tshirts Lisa_tshirts Carly_tshirts Lisa_jeans Lisa_coats Carly_jeans Carly_coats)
  : Carly_tshirts / Lisa_tshirts = 1 / 4 := 
sorry

end ratio_of_tshirts_l169_169566


namespace minimum_value_f_maximum_value_f_l169_169190

-- Problem 1: Minimum value of f(x) = 12/x + 3x for x > 0
theorem minimum_value_f (x : ℝ) (h : x > 0) : 
  (12 / x + 3 * x) ≥ 12 :=
sorry

-- Problem 2: Maximum value of f(x) = x(1 - 3x) for 0 < x < 1/3
theorem maximum_value_f (x : ℝ) (h1 : 0 < x) (h2 : x < 1 / 3) :
  x * (1 - 3 * x) ≤ 1 / 12 :=
sorry

end minimum_value_f_maximum_value_f_l169_169190


namespace remainder_modulo_l169_169231

theorem remainder_modulo (y : ℕ) (hy : 5 * y ≡ 1 [MOD 17]) : (7 + y) % 17 = 14 :=
sorry

end remainder_modulo_l169_169231


namespace RouteB_quicker_than_RouteA_l169_169977

def RouteA_segment1_time : ℚ := 4 / 40 -- time in hours
def RouteA_segment2_time : ℚ := 4 / 20 -- time in hours
def RouteA_total_time : ℚ := RouteA_segment1_time + RouteA_segment2_time -- total time in hours

def RouteB_segment1_time : ℚ := 6 / 35 -- time in hours
def RouteB_segment2_time : ℚ := 1 / 15 -- time in hours
def RouteB_total_time : ℚ := RouteB_segment1_time + RouteB_segment2_time -- total time in hours

def time_difference_minutes : ℚ := (RouteA_total_time - RouteB_total_time) * 60 -- difference in minutes

theorem RouteB_quicker_than_RouteA : time_difference_minutes = 3.71 := by
  sorry

end RouteB_quicker_than_RouteA_l169_169977


namespace no_nat_m_n_square_diff_2014_l169_169334

theorem no_nat_m_n_square_diff_2014 :
  ¬ ∃ (m n : ℕ), m^2 = n^2 + 2014 :=
by
  sorry

end no_nat_m_n_square_diff_2014_l169_169334


namespace max_elevation_l169_169879

def particle_elevation (t : ℝ) : ℝ := 200 * t - 20 * t^2 + 50

theorem max_elevation : ∃ t : ℝ, particle_elevation t = 550 :=
by {
  sorry
}

end max_elevation_l169_169879


namespace fraction_relation_l169_169846

theorem fraction_relation (n d : ℕ) (h1 : (n + 1 : ℚ) / (d + 1) = 3 / 5) (h2 : (n : ℚ) / d = 5 / 9) :
  ∃ k : ℚ, d = k * 2 * n ∧ k = 9 / 10 :=
by
  sorry

end fraction_relation_l169_169846


namespace length_of_first_leg_of_triangle_l169_169623

theorem length_of_first_leg_of_triangle 
  (a b c : ℝ) 
  (h1 : b = 8) 
  (h2 : c = 10) 
  (h3 : c^2 = a^2 + b^2) : 
  a = 6 :=
by
  sorry

end length_of_first_leg_of_triangle_l169_169623


namespace cos_lt_sin3_div_x3_l169_169982

open Real

theorem cos_lt_sin3_div_x3 (x : ℝ) (h1 : 0 < x) (h2 : x < pi / 2) : 
  cos x < (sin x / x) ^ 3 := 
  sorry

end cos_lt_sin3_div_x3_l169_169982


namespace choose_4_from_15_is_1365_l169_169546

theorem choose_4_from_15_is_1365 : nat.choose 15 4 = 1365 :=
by
  sorry

end choose_4_from_15_is_1365_l169_169546


namespace minimum_score_for_advanced_course_l169_169257

theorem minimum_score_for_advanced_course (q1 q2 q3 q4 : ℕ) (H1 : q1 = 88) (H2 : q2 = 84) (H3 : q3 = 82) :
  (q1 + q2 + q3 + q4) / 4 ≥ 85 → q4 = 86 := by
  sorry

end minimum_score_for_advanced_course_l169_169257


namespace certain_number_is_4_l169_169958

theorem certain_number_is_4 (x y C : ℝ) (h1 : 2 * x - y = C) (h2 : 6 * x - 3 * y = 12) : C = 4 :=
by
  -- Proof goes here
  sorry

end certain_number_is_4_l169_169958


namespace no_naturals_satisfy_m_squared_eq_n_squared_plus_2014_l169_169329

theorem no_naturals_satisfy_m_squared_eq_n_squared_plus_2014 :
  ∀ (m n : ℕ), ¬ (m^2 = n^2 + 2014) :=
by
  intro m n
  sorry

end no_naturals_satisfy_m_squared_eq_n_squared_plus_2014_l169_169329


namespace simplify_expression_and_evaluate_at_zero_l169_169708

theorem simplify_expression_and_evaluate_at_zero :
  ((2 * (0 : ℝ) - 1) / (0 + 1) - 0 + 1) / ((0 - 2) / ((0 ^ 2) + 2 * 0 + 1)) = 0 :=
by
  -- proof omitted
  sorry

end simplify_expression_and_evaluate_at_zero_l169_169708


namespace perpendicular_slope_l169_169373

-- Define the line equation and the result we want to prove about its perpendicular slope
def line_eq (x y : ℝ) := 5 * x - 2 * y = 10

theorem perpendicular_slope : ∀ (m : ℝ), 
  (∀ (x y : ℝ), line_eq x y → y = (5 / 2) * x - 5) →
  m = -(2 / 5) :=
by
  intros m H
  -- Additional logical steps would go here
  sorry

end perpendicular_slope_l169_169373


namespace find_N_l169_169771

theorem find_N (N : ℕ) :
  (∃ (f : ℕ × ℕ × ℕ × ℕ × ℕ → ℕ), (∀ (x y z w t : ℕ), f (x, y, z, w, t) = (a + b + c + d + 2)^N ∧ x > 0 ∧ y > 0 ∧ z > 0 ∧ w > 0 ∧ x + y + z + w + t = N)
   →  ∑ n, (f = 715)) ↔ N = 12 := sorry

end find_N_l169_169771


namespace max_n_value_l169_169831

def S (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (Finset.range n).sum a

theorem max_n_value (a : ℕ → ℝ) (h1 : ∀ n : ℕ, 1 ≤ n → (2 * (n + 0.5) = a n + a (n + 1))) 
  (h2 : S a 63 = 2020) (h3 : a 2 < 3) : 63 ∈ { n : ℕ | S a n = 2020 } :=
sorry

end max_n_value_l169_169831


namespace men_absent_l169_169484

theorem men_absent (x : ℕ) :
  let original_men := 42
  let original_days := 17
  let remaining_days := 21 
  let total_work := original_men * original_days
  let remaining_men_work := (original_men - x) * remaining_days 
  total_work = remaining_men_work →
  x = 8 :=
by
  intros
  let total_work := 42 * 17
  let remaining_men_work := (42 - x) * 21
  have h : total_work = remaining_men_work := ‹total_work = remaining_men_work›
  sorry

end men_absent_l169_169484


namespace correct_calculation_l169_169122

theorem correct_calculation (m n : ℝ) :
  3 * m^2 * n - 3 * m^2 * n = 0 ∧
  ¬ (3 * m^2 - 2 * m^2 = 1) ∧
  ¬ (3 * m^2 + 2 * m^2 = 5 * m^4) ∧
  ¬ (3 * m + 2 * n = 5 * m * n) := by
  sorry

end correct_calculation_l169_169122


namespace number_of_red_balls_l169_169040

-- Initial conditions
def num_black_balls : ℕ := 7
def num_white_balls : ℕ := 5
def freq_red_ball : ℝ := 0.4

-- Proving the number of red balls
theorem number_of_red_balls (total_balls : ℕ) (num_red_balls : ℕ) :
  total_balls = num_black_balls + num_white_balls + num_red_balls ∧
  (num_red_balls : ℝ) / total_balls = freq_red_ball →
  num_red_balls = 8 :=
by
  sorry

end number_of_red_balls_l169_169040


namespace f_x_when_x_negative_l169_169781

-- Define the properties of the function
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def f_definition (f : ℝ → ℝ) : Prop :=
  ∀ x, 0 < x → f x = x * (1 + x)

-- The theorem we want to prove
theorem f_x_when_x_negative (f : ℝ → ℝ) 
  (h1: odd_function f)
  (h2: f_definition f) : 
  ∀ x, x < 0 → f x = -x * (1 - x) :=
by
  sorry

end f_x_when_x_negative_l169_169781


namespace total_dots_l169_169629

def ladybugs_monday : ℕ := 8
def ladybugs_tuesday : ℕ := 5
def dots_per_ladybug : ℕ := 6

theorem total_dots :
  (ladybugs_monday + ladybugs_tuesday) * dots_per_ladybug = 78 :=
by
  sorry

end total_dots_l169_169629


namespace circles_are_externally_tangent_l169_169394

-- Conditions given in the problem
def r1 (r2 : ℝ) : Prop := ∃ r1 : ℝ, r1 * r2 = 10 ∧ r1 + r2 = 7
def distance := 7

-- The positional relationship proof problem statement
theorem circles_are_externally_tangent (r1 r2 : ℝ) (h : r1 * r2 = 10 ∧ r1 + r2 = 7) (d : ℝ) (h_d : d = distance) : 
  d = r1 + r2 :=
sorry

end circles_are_externally_tangent_l169_169394


namespace find_local_min_l169_169813

def z (x y : ℝ) : ℝ := x^2 + 2 * y^2 - 2 * x * y - x - 2 * y

theorem find_local_min: ∃ (x y : ℝ), x = 2 ∧ y = 3/2 ∧ ∀ ⦃h : ℝ⦄, h ≠ 0 → z (2 + h) (3/2 + h) > z 2 (3/2) :=
by
  sorry

end find_local_min_l169_169813


namespace no_naturals_satisfy_m_squared_eq_n_squared_plus_2014_l169_169324

theorem no_naturals_satisfy_m_squared_eq_n_squared_plus_2014 :
  ∀ (m n : ℕ), ¬ (m^2 = n^2 + 2014) :=
by
  intro m n
  sorry

end no_naturals_satisfy_m_squared_eq_n_squared_plus_2014_l169_169324


namespace least_integer_value_y_l169_169730

theorem least_integer_value_y (y : ℤ) (h : abs (3 * y - 4) ≤ 25) : y = -7 :=
sorry

end least_integer_value_y_l169_169730


namespace floor_sqrt_120_l169_169911

theorem floor_sqrt_120 : (⌊Real.sqrt 120⌋ = 10) :=
by
  -- Conditions from the problem
  have h1: 10^2 = 100 := rfl
  have h2: 11^2 = 121 := rfl
  have h3: 10 < Real.sqrt 120 := sorry
  have h4: Real.sqrt 120 < 11 := sorry
  -- Proof goal
  sorry

end floor_sqrt_120_l169_169911


namespace solve_abs_inequality_l169_169243

theorem solve_abs_inequality (x : ℝ) (h : 1 < |x - 1| ∧ |x - 1| < 4) : (-3 < x ∧ x < 0) ∨ (2 < x ∧ x < 5) :=
by
  sorry

end solve_abs_inequality_l169_169243


namespace least_lcm_of_x_and_z_l169_169848

theorem least_lcm_of_x_and_z (x y z : ℕ) (h₁ : Nat.lcm x y = 20) (h₂ : Nat.lcm y z = 28) : 
  ∃ l, l = Nat.lcm x z ∧ l = 35 := 
sorry

end least_lcm_of_x_and_z_l169_169848


namespace no_naturals_satisfy_m_squared_eq_n_squared_plus_2014_l169_169326

theorem no_naturals_satisfy_m_squared_eq_n_squared_plus_2014 :
  ∀ (m n : ℕ), ¬ (m^2 = n^2 + 2014) :=
by
  intro m n
  sorry

end no_naturals_satisfy_m_squared_eq_n_squared_plus_2014_l169_169326


namespace no_nat_m_n_eq_m_squared_eq_n_squared_plus_2014_l169_169321

theorem no_nat_m_n_eq_m_squared_eq_n_squared_plus_2014 :
  ¬ ∃ (m n : ℕ), m^2 = n^2 + 2014 :=
by 
  sorry

end no_nat_m_n_eq_m_squared_eq_n_squared_plus_2014_l169_169321


namespace natural_solution_unique_l169_169773

theorem natural_solution_unique (n : ℕ) (h : (2 * n - 1) / n^5 = 3 - 2 / n) : n = 1 := by
  sorry

end natural_solution_unique_l169_169773


namespace weight_of_replaced_person_l169_169991

theorem weight_of_replaced_person 
  (avg_increase : ℝ) (new_person_weight : ℝ) (n : ℕ) (original_weight : ℝ) 
  (h1 : avg_increase = 2.5)
  (h2 : new_person_weight = 95)
  (h3 : n = 8)
  (h4 : original_weight = new_person_weight - n * avg_increase) : 
  original_weight = 75 := 
by
  sorry

end weight_of_replaced_person_l169_169991


namespace weight_of_rod_l169_169408

theorem weight_of_rod (w₆ : ℝ) (h₁ : w₆ = 6.1) : 
  w₆ / 6 * 12 = 12.2 := by
  sorry

end weight_of_rod_l169_169408


namespace nine_values_of_x_l169_169208

theorem nine_values_of_x : ∃! (n : ℕ), ∃! (xs : Finset ℕ), xs.card = n ∧ 
  (∀ x ∈ xs, 3 * x < 100 ∧ 4 * x ≥ 100) ∧ 
  (xs.image (λ x => x)).val = ({25, 26, 27, 28, 29, 30, 31, 32, 33} : Finset ℕ).val :=
sorry

end nine_values_of_x_l169_169208


namespace no_nat_solutions_m2_eq_n2_plus_2014_l169_169311

theorem no_nat_solutions_m2_eq_n2_plus_2014 :
  ¬ ∃ (m n : ℕ), m^2 = n^2 + 2014 :=
by
  sorry

end no_nat_solutions_m2_eq_n2_plus_2014_l169_169311


namespace count_jianzhan_count_gift_boxes_l169_169132

-- Definitions based on given conditions
def firewood_red_clay : Int := 90
def firewood_white_clay : Int := 60
def electric_red_clay : Int := 75
def electric_white_clay : Int := 75
def total_red_clay : Int := 1530
def total_white_clay : Int := 1170

-- Proof problem 1: Number of "firewood firing" and "electric firing" Jianzhan produced
theorem count_jianzhan (x y : Int) (hx : firewood_red_clay * x + electric_red_clay * y = total_red_clay)
  (hy : firewood_white_clay * x + electric_white_clay * y = total_white_clay) : 
  x = 12 ∧ y = 6 :=
sorry

-- Definitions based on given conditions for Part 2
def total_jianzhan : Int := 18
def box_a_capacity : Int := 2
def box_b_capacity : Int := 6

-- Proof problem 2: Number of purchasing plans for gift boxes
theorem count_gift_boxes (m n : Int) (h : box_a_capacity * m + box_b_capacity * n = total_jianzhan) : 
  ∃ s : Finset (Int × Int), s.card = 4 ∧ ∀ (p : Int × Int), p ∈ s ↔ (p = (9, 0) ∨ p = (6, 1) ∨ p = (3, 2) ∨ p = (0, 3)) :=
sorry

end count_jianzhan_count_gift_boxes_l169_169132


namespace distinct_ratios_zero_l169_169070

theorem distinct_ratios_zero (p q r : ℝ) (hpq : p ≠ q) (hqr : q ≠ r) (hrp : r ≠ p) 
  (h : p / (q - r) + q / (r - p) + r / (p - q) = 3) :
  p / (q - r)^2 + q / (r - p)^2 + r / (p - q)^2 = 0 :=
sorry

end distinct_ratios_zero_l169_169070


namespace total_people_in_line_l169_169728

theorem total_people_in_line (n_front n_behind : ℕ) (hfront : n_front = 11) (hbehind : n_behind = 12) : n_front + n_behind + 1 = 24 := by
  sorry

end total_people_in_line_l169_169728


namespace totalStudents_l169_169842

-- Define the number of seats per ride
def seatsPerRide : ℕ := 15

-- Define the number of empty seats per ride
def emptySeatsPerRide : ℕ := 3

-- Define the number of rides taken
def ridesTaken : ℕ := 18

-- Define the number of students per ride
def studentsPerRide (seats : ℕ) (empty : ℕ) : ℕ := seats - empty

-- Calculate the total number of students
theorem totalStudents : studentsPerRide seatsPerRide emptySeatsPerRide * ridesTaken = 216 :=
by
  sorry

end totalStudents_l169_169842


namespace no_nat_solutions_m_sq_eq_n_sq_plus_2014_l169_169351

theorem no_nat_solutions_m_sq_eq_n_sq_plus_2014 :
  ¬ ∃ (m n : ℕ), m ^ 2 = n ^ 2 + 2014 := 
sorry

end no_nat_solutions_m_sq_eq_n_sq_plus_2014_l169_169351


namespace pears_left_l169_169817

theorem pears_left (jason_pears : ℕ) (keith_pears : ℕ) (mike_ate : ℕ) 
  (h1 : jason_pears = 46) 
  (h2 : keith_pears = 47) 
  (h3 : mike_ate = 12) : 
  jason_pears + keith_pears - mike_ate = 81 := 
by 
  sorry

end pears_left_l169_169817


namespace ratio_of_ages_in_two_years_l169_169748

theorem ratio_of_ages_in_two_years (S M : ℕ) 
  (h1 : M = S + 37) 
  (h2 : S = 35) : 
  (M + 2) / (S + 2) = 2 := 
by 
  -- We skip the proof steps as instructed
  sorry

end ratio_of_ages_in_two_years_l169_169748


namespace differential_equation_solution_exists_l169_169181

open Real

noncomputable def general_solution (x : ℝ) (C1 C2 : ℝ) : ℝ :=
  C1 + C2 * exp (-x) + (x^3) / 3

noncomputable def particular_solution (x : ℝ) : ℝ :=
  -2 + 3 * exp (-x) + (x^3) / 3

theorem differential_equation_solution_exists :
  ∃ (y : ℝ → ℝ), (∀ x, y x = -2 + 3 * exp (-x) + (x^3) / 3) ∧
  (∀ y, deriv y = λ x, (-3 * exp (-x) + x^2) ∧ y 0 = 1 ∧ deriv y 0 = -3) :=
by
  sorry

end differential_equation_solution_exists_l169_169181


namespace a_neg_half_not_bounded_a_bounded_range_l169_169904

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  1 + a * (1/3)^x + (1/9)^x

theorem a_neg_half_not_bounded (a : ℝ) :
  a = -1/2 → ¬(∃ M > 0, ∀ x < 0, |f x a| ≤ M) :=
by
  sorry

theorem a_bounded_range (a : ℝ) : 
  (∀ x ≥ 0, |f x a| ≤ 4) → -6 ≤ a ∧ a ≤ 2 :=
by
  sorry

end a_neg_half_not_bounded_a_bounded_range_l169_169904


namespace no_nat_m_n_square_diff_2014_l169_169337

theorem no_nat_m_n_square_diff_2014 :
  ¬ ∃ (m n : ℕ), m^2 = n^2 + 2014 :=
by
  sorry

end no_nat_m_n_square_diff_2014_l169_169337


namespace necessary_but_not_sufficient_condition_l169_169189

variable (p q : Prop)

theorem necessary_but_not_sufficient_condition (hpq : p ∨ q) (h : p ∧ q) : p ∧ q ↔ (p ∨ q) := by
  sorry

end necessary_but_not_sufficient_condition_l169_169189


namespace no_quadratic_polynomials_f_g_l169_169983

theorem no_quadratic_polynomials_f_g (f g : ℝ → ℝ) 
  (hf : ∃ a b c, ∀ x, f x = a * x^2 + b * x + c)
  (hg : ∃ d e h, ∀ x, g x = d * x^2 + e * x + h) : 
  ¬ (∀ x, f (g x) = x^4 - 3 * x^3 + 3 * x^2 - x) :=
by
  sorry

end no_quadratic_polynomials_f_g_l169_169983


namespace find_x_l169_169508

theorem find_x 
  (x : ℝ)
  (h : 3.5 * ((3.6 * 0.48 * 2.50) / (0.12 * x * 0.5)) = 2800.0000000000005) : 
  x = 0.225 := 
sorry

end find_x_l169_169508


namespace population_at_300pm_l169_169891

namespace BacteriaProblem

def initial_population : ℕ := 50
def time_increments_to_220pm : ℕ := 4   -- 4 increments of 5 mins each till 2:20 p.m.
def time_increments_to_300pm : ℕ := 2   -- 2 increments of 10 mins each till 3:00 p.m.

def growth_factor_before_220pm : ℕ := 3
def growth_factor_after_220pm : ℕ := 2

theorem population_at_300pm :
  initial_population * growth_factor_before_220pm^time_increments_to_220pm *
  growth_factor_after_220pm^time_increments_to_300pm = 16200 :=
by
  sorry

end BacteriaProblem

end population_at_300pm_l169_169891


namespace smaller_circle_radius_l169_169039

theorem smaller_circle_radius
  (R : ℝ) (r : ℝ)
  (h1 : R = 12)
  (h2 : 7 = 7) -- This is trivial and just emphasizes the arrangement of seven congruent smaller circles
  (h3 : 4 * (2 * r) = 2 * R) : r = 3 := by
  sorry

end smaller_circle_radius_l169_169039


namespace digit_sum_2001_not_perfect_square_l169_169497

theorem digit_sum_2001_not_perfect_square (n : ℕ) (h : (n.digits 10).sum = 2001) : ¬ ∃ k : ℕ, n = k * k := 
sorry

end digit_sum_2001_not_perfect_square_l169_169497


namespace units_digit_17_pow_2007_l169_169738

theorem units_digit_17_pow_2007 : (17^2007) % 10 = 3 :=
by sorry

end units_digit_17_pow_2007_l169_169738


namespace range_of_a_in_triangle_l169_169963

open Real

noncomputable def law_of_sines_triangle (A B C : ℝ) (a b c : ℝ) :=
  sin A / a = sin B / b ∧ sin B / b = sin C / c

theorem range_of_a_in_triangle (b : ℝ) (B : ℝ) (a : ℝ) (h1 : b = 2) (h2 : B = pi / 4) (h3 : true) :
  2 < a ∧ a < 2 * sqrt 2 :=
by
  sorry

end range_of_a_in_triangle_l169_169963


namespace range_of_m_l169_169011

theorem range_of_m (m : ℝ) (x : ℝ) (h₁ : x^2 - 8*x - 20 ≤ 0) 
  (h₂ : (x - 1 - m) * (x - 1 + m) ≤ 0) (h₃ : 0 < m) : 
  m ≤ 3 := sorry

end range_of_m_l169_169011


namespace min_omega_value_l169_169939

noncomputable def f (ω φ x : ℝ) : ℝ := Real.sin (ω * x + φ)

theorem min_omega_value (ω : ℝ) (φ : ℝ) (h_ω_pos : ω > 0)
  (h_even : ∀ x : ℝ, f ω φ x = f ω φ (-x))
  (h_symmetry : f ω φ 1 = 0 ∧ ∀ x : ℝ, f ω φ (1 + x) = - f ω φ (1 - x)) :
  ω = Real.pi / 2 :=
by
  sorry

end min_omega_value_l169_169939


namespace line_log_intersection_l169_169876

theorem line_log_intersection (a b : ℤ) (k : ℝ)
  (h₁ : k = a + Real.sqrt b)
  (h₂ : k > 0)
  (h₃ : Real.log k / Real.log 2 - Real.log (k + 2) / Real.log 2 = 1
    ∨ Real.log (k + 2) / Real.log 2 - Real.log k / Real.log 2 = 1) :
  a + b = 2 :=
sorry

end line_log_intersection_l169_169876


namespace both_students_given_correct_l169_169459

open ProbabilityTheory

variables (P_A P_B : ℝ)

-- Define the conditions from part a)
def student_a_correct := P_A = 3 / 5
def student_b_correct := P_B = 1 / 3

-- Define the event that both students correctly answer
def both_students_correct := P_A * P_B

-- Define the event that the question is answered correctly
def question_answered_correctly := (P_A * (1 - P_B)) + ((1 - P_A) * P_B) + (P_A * P_B)

-- Define the conditional probability we need to prove
theorem both_students_given_correct (hA : student_a_correct P_A) (hB : student_b_correct P_B) :
  both_students_correct P_A P_B / question_answered_correctly P_A P_B = 3 / 11 := 
sorry

end both_students_given_correct_l169_169459


namespace sub_number_l169_169440

theorem sub_number : 600 - 333 = 267 := by
  sorry

end sub_number_l169_169440


namespace min_value_a_b_l169_169196

open Real

theorem min_value_a_b {a b : ℝ} (ha : 0 < a) (hb : 0 < b)
  (parallel : a * (1 - b) - b * (a - 4) = 0) : a + b = 9 / 2 :=
by sorry

end min_value_a_b_l169_169196


namespace children_in_school_l169_169427

theorem children_in_school (C B : ℕ) (h1 : B = 2 * C) (h2 : B = 4 * (C - 370)) : C = 740 :=
by
  sorry

end children_in_school_l169_169427


namespace f_is_even_f_range_l169_169945

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (|x| + 2) / (1 - |x|)

-- Prove that f(x) is an even function
theorem f_is_even : ∀ x : ℝ, f (-x) = f x := by
  sorry

-- Prove the range of f(x) is (-∞, -1) ∪ [2, +∞)
theorem f_range : ∀ y : ℝ, ∃ x : ℝ, y = f x ↔ y ≥ 2 ∨ y < -1 := by
  sorry

end f_is_even_f_range_l169_169945


namespace hyperbola_a_solution_l169_169527

noncomputable def hyperbola_a_value (a : ℝ) : Prop :=
  (a > 0) ∧ (∀ x y : ℝ, (x^2 / a^2) - (y^2 / 2) = 1) ∧ (∃ e : ℝ, e = 2)

theorem hyperbola_a_solution : ∃ a : ℝ, hyperbola_a_value a ∧ a = (Real.sqrt 6) / 3 :=
  by
    sorry

end hyperbola_a_solution_l169_169527


namespace linear_function_quadrant_l169_169946

theorem linear_function_quadrant (x y : ℝ) (h : y = -3 * x + 2) :
  ¬ (x > 0 ∧ y > 0) :=
by
  sorry

end linear_function_quadrant_l169_169946


namespace average_snowfall_per_hour_l169_169536

theorem average_snowfall_per_hour (total_snowfall : ℕ) (hours_per_week : ℕ) (total_snowfall_eq : total_snowfall = 210) (hours_per_week_eq : hours_per_week = 7 * 24) : 
  total_snowfall / hours_per_week = 5 / 4 :=
by
  -- skip the proof
  sorry

end average_snowfall_per_hour_l169_169536


namespace area_of_set_A_l169_169519

noncomputable def area_of_A : ℝ :=
  π

theorem area_of_set_A :
  (∀ (a b : ℝ), ¬ ∃ x : ℝ, (x ^ 2 - 2 * a * x + 1 = 2 * b * (a - x))) →
  (set.univ ∩ {p : ℝ × ℝ | (p.1^2 + (p.2)^2 < 1)}).measure = π :=
by
  intro h
  sorry

end area_of_set_A_l169_169519


namespace min_photos_l169_169050

def num_monuments := 3
def num_tourists := 42

def took_photo_of (T M : Type) := T → M → Prop
def photo_coverage (T M : Type) (f : took_photo_of T M) :=
  ∀ (t1 t2 : T) (m : M), f t1 m ∨ f t2 m

def minimal_photos (T M : Type) [Fintype T] [Fintype M] [DecidableEq M]
  (f : took_photo_of T M) :=
  ∑ m, card { t : T | f t m }

theorem min_photos {T : Type} [Fintype T] [DecidableEq T] (M : fin num_monuments) (f : took_photo_of T M)
  (hT : card T = num_tourists)
  (h_cov : photo_coverage T M f) :
  minimal_photos T M f ≥ 123 :=
sorry

end min_photos_l169_169050


namespace sum_s_h_e_base_three_l169_169024

def distinct_non_zero_digits (S H E : ℕ) : Prop :=
  S ≠ 0 ∧ H ≠ 0 ∧ E ≠ 0 ∧ S < 3 ∧ H < 3 ∧ E < 3 ∧ S ≠ H ∧ H ≠ E ∧ S ≠ E

def base_three_addition (S H E : ℕ) :=
  (S + H * 3 + E * 9) + (H + E * 3) == (H * 3 + S * 9 + S*27)

theorem sum_s_h_e_base_three (S H E : ℕ) (h1 : distinct_non_zero_digits S H E) (h2 : base_three_addition S H E) :
  (S + H + E = 5) := by sorry

end sum_s_h_e_base_three_l169_169024


namespace no_positive_a_exists_l169_169364

theorem no_positive_a_exists :
  ¬ ∃ (a : ℝ), (0 < a) ∧ ∀ (x : ℝ), |cos x| + |cos (a * x)| > sin x + sin (a * x) :=
by
  sorry

end no_positive_a_exists_l169_169364


namespace ellipse_major_minor_axis_ratio_l169_169666

theorem ellipse_major_minor_axis_ratio
  (a b : ℝ)
  (h₀ : a = 2 * b):
  2 * a = 4 * b :=
by
  sorry

end ellipse_major_minor_axis_ratio_l169_169666


namespace nate_pages_left_to_read_l169_169078

-- Define the constants and conditions
def total_pages : ℕ := 400
def percentage_read : ℕ := 20

-- Calculate the number of pages already read
def pages_read := total_pages * percentage_read / 100

-- Calculate the number of pages left
def pages_left := total_pages - pages_read

-- Statement to prove
theorem nate_pages_left_to_read : pages_left = 320 :=
by {
  unfold pages_read,
  unfold pages_left,
  simp,
  sorry -- The proof will be filled in based on the calculations in the solution.
}

end nate_pages_left_to_read_l169_169078


namespace no_nat_m_n_square_diff_2014_l169_169339

theorem no_nat_m_n_square_diff_2014 :
  ¬ ∃ (m n : ℕ), m^2 = n^2 + 2014 :=
by
  sorry

end no_nat_m_n_square_diff_2014_l169_169339


namespace floor_sqrt_120_l169_169907

theorem floor_sqrt_120 : (Int.floor (Real.sqrt 120)) = 10 := by
  have h1 : 10^2 < 120 := by norm_num
  have h2 : 120 < 11^2 := by norm_num
  -- Additional steps to show that Int.floor (Real.sqrt 120) = 10
  sorry

end floor_sqrt_120_l169_169907


namespace line_passes_through_point_l169_169995

theorem line_passes_through_point :
  ∀ (m : ℝ), (∃ y : ℝ, y - 2 = m * (-1) + m) :=
by
  intros m
  use 2
  sorry

end line_passes_through_point_l169_169995


namespace no_real_solutions_for_identical_lines_l169_169828

theorem no_real_solutions_for_identical_lines :
  ¬∃ (a d : ℝ), (∀ x y : ℝ, 5 * x + a * y + d = 0 ↔ 2 * d * x - 3 * y + 8 = 0) :=
by
  sorry

end no_real_solutions_for_identical_lines_l169_169828


namespace hanoi_moves_correct_l169_169725

def hanoi_moves (n : ℕ) : ℕ :=
  if n = 0 then 0 else 2 * hanoi_moves (n - 1) + 1

theorem hanoi_moves_correct (n : ℕ) : hanoi_moves n = 2^n - 1 := by
  sorry

end hanoi_moves_correct_l169_169725


namespace remainder_of_power_mod_l169_169637

theorem remainder_of_power_mod (a n p : ℕ) (h_prime : Nat.Prime p) (h_a : a < p) :
  (3 : ℕ)^2024 % 17 = 13 :=
by
  sorry

end remainder_of_power_mod_l169_169637


namespace paint_left_after_two_coats_l169_169121

theorem paint_left_after_two_coats :
  let initial_paint := 3 -- liters
  let first_coat_paint := initial_paint / 2
  let paint_after_first_coat := initial_paint - first_coat_paint
  let second_coat_paint := (2 / 3) * paint_after_first_coat
  let paint_after_second_coat := paint_after_first_coat - second_coat_paint
  (paint_after_second_coat * 1000) = 500 := by
  sorry

end paint_left_after_two_coats_l169_169121


namespace largest_A_form_B_moving_last_digit_smallest_A_form_B_moving_last_digit_l169_169147

theorem largest_A_form_B_moving_last_digit (B : Nat) (h0 : Nat.gcd B 24 = 1) (h1 : B > 666666666) (h2 : B < 1000000000) :
  let A := 10^8 * (B % 10) + (B / 10)
  A ≤ 999999998 :=
sorry

theorem smallest_A_form_B_moving_last_digit (B : Nat) (h0 : Nat.gcd B 24 = 1) (h1 : B > 666666666) (h2 : B < 1000000000) :
  let A := 10^8 * (B % 10) + (B / 10)
  A ≥ 166666667 :=
sorry

end largest_A_form_B_moving_last_digit_smallest_A_form_B_moving_last_digit_l169_169147


namespace greatest_possible_value_l169_169864

theorem greatest_possible_value :
  ∃ (N P M : ℕ), (M < 10) ∧ (N < 10) ∧ (P < 10) ∧ (M * (111 * M) = N * 1000 + P * 100 + M * 10 + M)
                ∧ (N * 1000 + P * 100 + M * 10 + M = 3996) :=
by
  sorry

end greatest_possible_value_l169_169864


namespace solve_olympics_problem_max_large_sets_l169_169581

-- Definitions based on the conditions
variables (x y : ℝ)

-- Condition 1: 2 small sets cost $20 less than 1 large set
def condition1 : Prop := y - 2 * x = 20

-- Condition 2: 3 small sets and 2 large sets cost $390
def condition2 : Prop := 3 * x + 2 * y = 390

-- Finding unit prices
def unit_prices : Prop := x = 50 ∧ y = 120

-- Condition 3: Budget constraint for purchasing sets
def budget_constraint (m : ℕ) : Prop := m ≤ 7

-- Prove unit prices and purchasing constraints
theorem solve_olympics_problem :
  condition1 x y ∧ condition2 x y → unit_prices x y :=
by
  sorry

theorem max_large_sets :
  budget_constraint 7 :=
by
  sorry

end solve_olympics_problem_max_large_sets_l169_169581


namespace minimum_photos_l169_169052

theorem minimum_photos (M N T : Type) [DecidableEq M] [DecidableEq N] [DecidableEq T] 
  (monuments : Finset M) (city : N) (tourists : Finset T) 
  (photos : T → Finset M) :
  monuments.card = 3 → 
  tourists.card = 42 → 
  (∀ t : T, (photos t).card ≤ 3) → 
  (∀ t₁ t₂ : T, t₁ ≠ t₂ → (photos t₁ ∪ photos t₂) = monuments) → 
  ∑ t in tourists, (photos t).card ≥ 123 := 
by
  intros h_monuments_card h_tourists_card h_photos_card h_photos_union
  sorry

end minimum_photos_l169_169052


namespace theta_in_first_quadrant_l169_169186

noncomputable def quadrant_of_theta (theta : ℝ) (h1 : Real.sin (Real.pi + theta) < 0) (h2 : Real.cos (Real.pi - theta) < 0) : ℕ :=
  if 0 < Real.sin theta ∧ 0 < Real.cos theta then 1 else sorry

theorem theta_in_first_quadrant (theta : ℝ) (h1 : Real.sin (Real.pi + theta) < 0) (h2 : Real.cos (Real.pi - theta) < 0) :
  quadrant_of_theta theta h1 h2 = 1 :=
by
  sorry

end theta_in_first_quadrant_l169_169186


namespace no_nat_m_n_square_diff_2014_l169_169336

theorem no_nat_m_n_square_diff_2014 :
  ¬ ∃ (m n : ℕ), m^2 = n^2 + 2014 :=
by
  sorry

end no_nat_m_n_square_diff_2014_l169_169336


namespace calc_result_l169_169635

theorem calc_result : 75 * 1313 - 25 * 1313 = 65750 := 
by 
  sorry

end calc_result_l169_169635


namespace proof_problem_l169_169202

-- Given function f
def f (x : ℝ) : ℝ := sin (x + π / 4) ^ 2

-- Define a and b as per problem statement
def a : ℝ := f (real.log 5)
def b : ℝ := f (real.log (1 / 5))

-- Proof statement encapsulating the questions from the problem
theorem proof_problem
  (h1 : a = f (real.log 5))
  (h2 : b = f (real.log (1 / 5))) :
  a + b = 1 ∧ a - b = sin (2 * real.log 5) :=
sorry

end proof_problem_l169_169202


namespace no_positive_integer_n_exists_l169_169971

theorem no_positive_integer_n_exists {n : ℕ} (hn : n > 0) :
  ¬ ((∃ k, 5 * 10^(k - 1) ≤ 2^n ∧ 2^n < 6 * 10^(k - 1)) ∧
     (∃ m, 2 * 10^(m - 1) ≤ 5^n ∧ 5^n < 3 * 10^(m - 1))) :=
sorry

end no_positive_integer_n_exists_l169_169971


namespace modular_inverse_l169_169644

/-- Define the number 89 -/
def a : ℕ := 89

/-- Define the modulus 90 -/
def n : ℕ := 90

/-- The condition given in the problem -/
lemma pow_mod (h : a ≡ -1 [MOD n]) : (a * a) % n = 1 % n := by 
  sorry

/-- The main statement to prove the modular inverse -/
theorem modular_inverse (h : a ≡ -1 [MOD n]) : (a * a) % n = 1 % n → a ≡ a⁻¹ [MOD n] := by
  intro h1
  have h2 : a⁻¹ % n = a % n := by 
    sorry
  exact h2

end modular_inverse_l169_169644


namespace minimum_value_expression_l169_169920

theorem minimum_value_expression :
  ∀ (r s t : ℝ), (1 ≤ r ∧ r ≤ s ∧ s ≤ t ∧ t ≤ 4) →
  (r - 1) ^ 2 + (s / r - 1) ^ 2 + (t / s - 1) ^ 2 + (4 / t - 1) ^ 2 = 4 * (Real.sqrt 2 - 1) ^ 2 := 
sorry

end minimum_value_expression_l169_169920


namespace parabola_focus_coincides_hyperbola_focus_l169_169406

theorem parabola_focus_coincides_hyperbola_focus (p : ℝ) : 
  (∀ x y : ℝ, y^2 = 2 * p * x -> (3,0) = (3,0)) → 
  (∀ x y : ℝ, x^2 / 6 - y^2 / 3 = 1 -> x = 3) → 
  p = 6 :=
by
  sorry

end parabola_focus_coincides_hyperbola_focus_l169_169406


namespace class_B_more_uniform_l169_169295

def x_A : ℝ := 80
def x_B : ℝ := 80
def S2_A : ℝ := 240
def S2_B : ℝ := 180

theorem class_B_more_uniform (h1 : x_A = 80) (h2 : x_B = 80) (h3 : S2_A = 240) (h4 : S2_B = 180) : 
  S2_B < S2_A :=
by {
  exact sorry
}

end class_B_more_uniform_l169_169295


namespace no_nat_solutions_m_sq_eq_n_sq_plus_2014_l169_169349

theorem no_nat_solutions_m_sq_eq_n_sq_plus_2014 :
  ¬ ∃ (m n : ℕ), m ^ 2 = n ^ 2 + 2014 := 
sorry

end no_nat_solutions_m_sq_eq_n_sq_plus_2014_l169_169349


namespace smallest_integer_n_l169_169602

theorem smallest_integer_n (n : ℤ) (h : n^2 - 9 * n + 20 > 0) : n ≥ 6 := 
sorry

end smallest_integer_n_l169_169602


namespace limit_of_sqrt_function_l169_169287

open Real

theorem limit_of_sqrt_function :
  tendsto (fun x => sqrt (x * (2 + sin(1 / x)) + 4 * cos x)) (𝓝 0) (𝓝 2) :=
begin
  sorry
end

end limit_of_sqrt_function_l169_169287


namespace water_overflowed_calculation_l169_169850

/-- The water supply rate is 200 kilograms per hour. -/
def water_supply_rate : ℕ := 200

/-- The water tank capacity is 4000 kilograms. -/
def tank_capacity : ℕ := 4000

/-- The water runs for 24 hours. -/
def running_time : ℕ := 24

/-- Calculation for the kilograms of water that overflowed. -/
theorem water_overflowed_calculation :
  water_supply_rate * running_time - tank_capacity = 800 :=
by
  -- calculation skipped
  sorry

end water_overflowed_calculation_l169_169850


namespace trajectory_of_N_l169_169422

variables {x y x₀ y₀ : ℝ}

def F : ℝ × ℝ := (1, 0)

def M (x₀ : ℝ) : ℝ × ℝ := (x₀, 0)
def P (y₀ : ℝ) : ℝ × ℝ := (0, y₀)
def N (x y : ℝ) : ℝ × ℝ := (x, y)

def PM (x₀ y₀ : ℝ) : ℝ × ℝ := (x₀, -y₀)
def PF (y₀ : ℝ) : ℝ × ℝ := (1, -y₀)

def perpendicular (v1 v2 : ℝ × ℝ) := v1.fst * v2.fst + v1.snd * v2.snd = 0

def MN_eq_2MP (x y x₀ y₀ : ℝ) := ((x - x₀), y) = (2 * (-x₀), 2 * y₀)

theorem trajectory_of_N (h1 : perpendicular (PM x₀ y₀) (PF y₀))
  (h2 : MN_eq_2MP x y x₀ y₀) :
  y^2 = 4*x :=
by
  sorry

end trajectory_of_N_l169_169422


namespace hyperbola_asymptote_l169_169790

theorem hyperbola_asymptote (a : ℝ) (h : a > 0) : 
  (∀ x y : ℝ, (x^2 - y^2 / a^2) = 1 → (y = 2*x ∨ y = -2*x)) → a = 2 :=
by
  intro h_asymptote
  sorry

end hyperbola_asymptote_l169_169790


namespace darla_total_payment_l169_169900

-- Definitions of the conditions
def rate_per_watt : ℕ := 4
def energy_usage : ℕ := 300
def late_fee : ℕ := 150

-- Definition of the expected total cost
def expected_total_cost : ℕ := 1350

-- Theorem stating the problem
theorem darla_total_payment :
  rate_per_watt * energy_usage + late_fee = expected_total_cost := 
by 
  sorry

end darla_total_payment_l169_169900


namespace parabola_tangent_line_l169_169216

theorem parabola_tangent_line (a b : ℝ) :
  (∀ x : ℝ, ax^2 + b * x + 2 = 2 * x + 3 → a = -1 ∧ b = 4) :=
sorry

end parabola_tangent_line_l169_169216


namespace area_of_given_triangle_l169_169463

def point := (ℝ × ℝ)

def area_of_triangle (A B C : point) : ℝ :=
  0.5 * (B.1 - A.1) * (C.2 - A.2)

theorem area_of_given_triangle :
  area_of_triangle (0, 0) (4, 0) (4, 6) = 12.0 :=
by 
  sorry

end area_of_given_triangle_l169_169463


namespace quadratic_has_two_distinct_real_roots_l169_169270

theorem quadratic_has_two_distinct_real_roots :
  let a := (1 : ℝ)
  let b := (-5 : ℝ)
  let c := (-1 : ℝ)
  b^2 - 4 * a * c > 0 :=
by
  sorry

end quadratic_has_two_distinct_real_roots_l169_169270


namespace probability_at_least_one_vowel_l169_169088

noncomputable def set1 : finset char := {'a', 'b', 'c', 'd', 'e'}
noncomputable def set2 : finset char := {'k', 'l', 'm', 'n', 'o', 'p'}
noncomputable def set3 : finset char := {'r', 's', 't', 'u', 'v'}
noncomputable def set4 : finset char := {'w', 'x', 'y', 'z', 'i'}

def vowels : set char := {'a', 'e', 'i', 'o', 'u'}

def probability_of_picking_at_least_one_vowel : ℚ :=
  let p_set1 := (3 / 5 : ℚ)
  let p_set2 := (1 : ℚ)
  let p_set3 := (1 : ℚ)
  let p_set4 := (4 / 5 : ℚ)
  let p_no_vowel := p_set1 * p_set2 * p_set3 * p_set4
  (1 - p_no_vowel)

theorem probability_at_least_one_vowel :
  probability_of_picking_at_least_one_vowel = 13 / 25 :=
sorry

end probability_at_least_one_vowel_l169_169088


namespace find_n_l169_169375

theorem find_n (n : ℕ) : (256 : ℝ) ^ (1 / 4 : ℝ) = 4 ^ n → 256 = (4 ^ 4 : ℝ) → n = 1 :=
by
  intros h₁ h₂
  sorry

end find_n_l169_169375


namespace total_bad_carrots_and_tomatoes_l169_169259

theorem total_bad_carrots_and_tomatoes 
  (vanessa_carrots : ℕ := 17)
  (vanessa_tomatoes : ℕ := 12)
  (mother_carrots : ℕ := 14)
  (mother_tomatoes : ℕ := 22)
  (brother_carrots : ℕ := 6)
  (brother_tomatoes : ℕ := 8)
  (good_carrots : ℕ := 28)
  (good_tomatoes : ℕ := 35) :
  (vanessa_carrots + mother_carrots + brother_carrots - good_carrots) + 
  (vanessa_tomatoes + mother_tomatoes + brother_tomatoes - good_tomatoes) = 16 := 
by
  sorry

end total_bad_carrots_and_tomatoes_l169_169259


namespace total_team_players_l169_169758

-- Conditions
def team_percent_boys : ℚ := 0.6
def team_percent_girls := 1 - team_percent_boys
def junior_girls_count : ℕ := 10
def total_girls := junior_girls_count * 2
def girl_percentage_as_decimal := team_percent_girls

-- Problem
theorem total_team_players : (total_girls : ℚ) / girl_percentage_as_decimal = 50 := 
by 
    sorry

end total_team_players_l169_169758


namespace four_op_two_l169_169561

def op (a b : ℝ) : ℝ := 2 * a + 5 * b

theorem four_op_two : op 4 2 = 18 := by
  sorry

end four_op_two_l169_169561


namespace johnny_marble_choice_l169_169226

/-- Johnny has 9 different colored marbles and always chooses 1 specific red marble.
    Prove that the number of ways to choose four marbles from his bag is 56. -/
theorem johnny_marble_choice : (Nat.choose 8 3) = 56 := 
by
  sorry

end johnny_marble_choice_l169_169226


namespace product_greater_than_sum_l169_169827

variable {a b : ℝ}

theorem product_greater_than_sum (ha : a > 2) (hb : b > 2) : a * b > a + b := 
  sorry

end product_greater_than_sum_l169_169827


namespace abs_sum_lt_abs_diff_l169_169934

theorem abs_sum_lt_abs_diff (a b : ℝ) (h : a * b < 0) : |a + b| < |a - b| := 
sorry

end abs_sum_lt_abs_diff_l169_169934


namespace geometric_sequence_common_ratio_l169_169765

theorem geometric_sequence_common_ratio (a : ℕ → ℝ) (r : ℝ) (h_geometric : ∀ n, a (n + 1) = r * a n)
  (h_relation : ∀ n, a n = (1 / 2) * (a (n + 1) + a (n + 2))) (h_positive : ∀ n, a n > 0) : r = 1 :=
sorry

end geometric_sequence_common_ratio_l169_169765


namespace meaningful_sqrt_range_l169_169217

theorem meaningful_sqrt_range (x : ℝ) (h : 2 * x - 3 ≥ 0) : x ≥ 3 / 2 :=
sorry

end meaningful_sqrt_range_l169_169217


namespace minimum_value_a_plus_3b_plus_9c_l169_169563

open Real

theorem minimum_value_a_plus_3b_plus_9c (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a * b * c = 27) :
  a + 3 * b + 9 * c ≥ 27 :=
sorry

end minimum_value_a_plus_3b_plus_9c_l169_169563


namespace power_subtraction_l169_169955

variable {a m n : ℝ}

theorem power_subtraction (hm : a^m = 8) (hn : a^n = 2) : a^(m - 3 * n) = 1 := by
  sorry

end power_subtraction_l169_169955


namespace problem_arithmetic_sequence_l169_169013

noncomputable def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
∀ n : ℕ, a (n + 1) = a n + d

theorem problem_arithmetic_sequence (a : ℕ → ℝ) (d a2 a8 : ℝ) :
  arithmetic_sequence a d →
  (a 2 + a 3 + a 4 + a 5 + a 6 = 450) →
  (a 1 + a 7 = 2 * a 4) →
  (a 2 + a 6 = 2 * a 4) →
  (a 2 + a 8 = 180) :=
by
  sorry

end problem_arithmetic_sequence_l169_169013


namespace same_roots_condition_l169_169464

-- Definition of quadratic equations with coefficients a1, b1, c1 and a2, b2, c2
variables (a1 b1 c1 a2 b2 c2 : ℝ)

-- The condition we need to prove
theorem same_roots_condition :
  (a1 ≠ 0 ∧ a2 ≠ 0) → 
  (a1 / a2 = b1 / b2 ∧ b1 / b2 = c1 / c2) 
    ↔ 
  ∀ x : ℝ, (a1 * x^2 + b1 * x + c1 = 0 ↔ a2 * x^2 + b2 * x + c2 = 0) :=
sorry

end same_roots_condition_l169_169464


namespace cartons_being_considered_l169_169481

-- Definitions based on conditions
def packs_per_box : ℕ := 10
def boxes_per_carton : ℕ := 12
def price_per_pack : ℕ := 1
def total_cost : ℕ := 1440

-- Calculate total cost per carton
def cost_per_carton : ℕ := boxes_per_carton * packs_per_box * price_per_pack

-- Formulate the main theorem
theorem cartons_being_considered : (total_cost / cost_per_carton) = 12 :=
by
  -- The relevant steps would go here, but we're only providing the statement
  sorry

end cartons_being_considered_l169_169481


namespace irrational_lattice_point_exists_l169_169019

theorem irrational_lattice_point_exists (k : ℝ) (h_irrational : ¬ ∃ q r : ℚ, q / r = k)
  (ε : ℝ) (h_pos : ε > 0) : ∃ m n : ℤ, |m * k - n| < ε :=
by
  sorry

end irrational_lattice_point_exists_l169_169019


namespace find_area_of_triangle_l169_169035

noncomputable def triangle_area (a b: ℝ) (cosC: ℝ) : ℝ :=
  let sinC := Real.sqrt (1 - cosC^2)
  0.5 * a * b * sinC

theorem find_area_of_triangle :
  ∀ (a b cosC : ℝ), a = 3 * Real.sqrt 2 → b = 2 * Real.sqrt 3 → cosC = 1 / 3 →
  triangle_area a b cosC = 4 * Real.sqrt 3 :=
by
  intros a b cosC ha hb hcosC
  rw [ha, hb, hcosC]
  sorry

end find_area_of_triangle_l169_169035


namespace no_nat_numbers_m_n_satisfy_eq_l169_169347

theorem no_nat_numbers_m_n_satisfy_eq (m n : ℕ) : ¬ (m^2 = n^2 + 2014) := sorry

end no_nat_numbers_m_n_satisfy_eq_l169_169347


namespace necessarily_negative_expression_l169_169975

theorem necessarily_negative_expression
  (x y z : ℝ)
  (hx : 0 < x ∧ x < 1)
  (hy : -1 < y ∧ y < 0)
  (hz : 0 < z ∧ z < 1)
  : y - z < 0 :=
sorry

end necessarily_negative_expression_l169_169975


namespace perimeter_of_square_36_l169_169245

variable (a s P : ℕ)

def is_square_area : Prop := a = s * s
def is_square_perimeter : Prop := P = 4 * s
def condition : Prop := 5 * a = 10 * P + 45

theorem perimeter_of_square_36 (h1 : is_square_area a s) (h2 : is_square_perimeter P s) (h3 : condition a P) : P = 36 := 
by
  sorry

end perimeter_of_square_36_l169_169245


namespace polygon_intersections_inside_circle_l169_169984

noncomputable def number_of_polygon_intersections
    (polygonSides: List Nat) : Nat :=
  let pairs := [(4,5), (4,7), (4,9), (5,7), (5,9), (7,9)]
  pairs.foldl (λ acc (p1, p2) => acc + 2 * min p1 p2) 0

theorem polygon_intersections_inside_circle :
  number_of_polygon_intersections [4, 5, 7, 9] = 58 :=
by
  sorry

end polygon_intersections_inside_circle_l169_169984


namespace find_value_of_a2004_b2004_l169_169015

-- Given Definitions and Conditions
def a : ℝ := sorry
def b : ℝ := sorry
def A : Set ℝ := {a, a^2, a * b}
def B : Set ℝ := {1, a, b}

-- The theorem statement
theorem find_value_of_a2004_b2004 (h : A = B) : a ^ 2004 + b ^ 2004 = 1 :=
sorry

end find_value_of_a2004_b2004_l169_169015


namespace recover_original_sequence_l169_169539

theorem recover_original_sequence :
  ∃ (a d : ℤ),
    [a, a + d, a + 2 * d, a + 3 * d, a + 4 * d, a + 5 * d] = [113, 125, 137, 149, 161, 173] :=
by
  sorry

end recover_original_sequence_l169_169539


namespace num_three_digit_integers_divisible_by_12_l169_169672

theorem num_three_digit_integers_divisible_by_12 : 
  (∃ (count : ℕ), count = 3 ∧ 
    (∀ (n : ℕ), 100 ≤ n ∧ n ≤ 999 → 
      (∀ d : ℕ, d ∈ [n / 100, (n / 10) % 10, n % 10] → 4 < d) ∧ 
      n % 12 = 0 → 
      count = count + 1)) := 
sorry

end num_three_digit_integers_divisible_by_12_l169_169672


namespace sufficient_but_not_necessary_l169_169927

def p (x : ℝ) : Prop := |x - 4| > 2
def q (x : ℝ) : Prop := x > 1

theorem sufficient_but_not_necessary (x : ℝ) :
  (∀ x, 2 ≤ x ∧ x ≤ 6 → x > 1) ∧ ¬(∀ x, x > 1 → 2 ≤ x ∧ x ≤ 6) :=
  sorry

end sufficient_but_not_necessary_l169_169927


namespace difference_between_second_and_third_levels_l169_169620

def total_parking_spots : ℕ := 400
def first_level_open_spots : ℕ := 58
def second_level_open_spots : ℕ := first_level_open_spots + 2
def fourth_level_open_spots : ℕ := 31
def total_full_spots : ℕ := 186

def total_open_spots : ℕ := total_parking_spots - total_full_spots

def third_level_open_spots : ℕ := 
  total_open_spots - (first_level_open_spots + second_level_open_spots + fourth_level_open_spots)

def difference_open_spots : ℕ := third_level_open_spots - second_level_open_spots

theorem difference_between_second_and_third_levels : difference_open_spots = 5 :=
sorry

end difference_between_second_and_third_levels_l169_169620


namespace polynomial_m_n_values_l169_169810

theorem polynomial_m_n_values :
  ∀ (m n : ℝ), ((x - 1) * (x + m) = x^2 - n * x - 6) → (m = 6 ∧ n = -5) := 
by
  intros m n h
  sorry

end polynomial_m_n_values_l169_169810


namespace max_min_values_l169_169004

noncomputable def y (x : ℝ) : ℝ :=
  3 - 4 * Real.sin x - 4 * (Real.cos x)^2

theorem max_min_values :
  (∀ k : ℤ, y (- (Real.pi/2) + 2 * k * Real.pi) = 7) ∧
  (∀ k : ℤ, y (Real.pi/6 + 2 * k * Real.pi) = -2) ∧
  (∀ k : ℤ, y (5 * Real.pi/6 + 2 * k * Real.pi) = -2) := by
  sorry

end max_min_values_l169_169004


namespace rectangle_cut_l169_169502

def dimensions_ratio (x y : ℕ) : Prop := ∃ (r : ℚ), x = r * y

theorem rectangle_cut (k m n : ℕ) (hk : ℝ) (hm : ℝ) (hn : ℝ) 
  (h1 : k + m + n = 10) 
  (h2 : k * 9 / 10 = hk)
  (h3 : m * 9 / 10 = hm)
  (h4 : n * 9 / 10 = hn)
  (h5 : hk + hm + hn = 9) :
  ∃ (k' m' n' : ℕ), 
    dimensions_ratio k k' ∧ 
    dimensions_ratio m m' ∧
    dimensions_ratio n n' ∧
    k ≠ m ∧ m ≠ n ∧ k ≠ n :=
sorry

end rectangle_cut_l169_169502


namespace calculate_expression_l169_169895

theorem calculate_expression :
  ( (5^1010)^2 - (5^1008)^2) / ( (5^1009)^2 - (5^1007)^2) = 25 := 
by
  sorry

end calculate_expression_l169_169895


namespace count_elements_in_T_l169_169559

theorem count_elements_in_T :
  let T := {n: ℕ | 1 < n ∧ ∃ k, (10^20 - 1) = n * k }
  (∀ (n : ℕ), n ∈ T ↔ 1 < n ∧ (10^20 - 1) % n = 0) →
  Nat.Prime 999001 →
  ∃ D, D = (Nat.divisors  (10^20 - 1)).card →
  T.card = D - 1 :=
by
  sorry

end count_elements_in_T_l169_169559


namespace time_to_be_d_miles_apart_l169_169073

def mary_walk_rate := 4 -- Mary's walking rate in miles per hour
def sharon_walk_rate := 6 -- Sharon's walking rate in miles per hour
def time_to_be_3_miles_apart := 0.3 -- Time in hours to be 3 miles apart
def initial_distance := 3 -- They are 3 miles apart after 0.3 hours

theorem time_to_be_d_miles_apart (d: ℝ) : ∀ t: ℝ,
  (mary_walk_rate + sharon_walk_rate) * t = d ↔ 
  t = d / (mary_walk_rate + sharon_walk_rate) :=
by
  intros
  sorry

end time_to_be_d_miles_apart_l169_169073


namespace edward_lives_left_l169_169472

theorem edward_lives_left : 
  let initial_lives := 50
  let stage1_loss := 18
  let stage1_gain := 7
  let stage2_loss := 10
  let stage2_gain := 5
  let stage3_loss := 13
  let stage3_gain := 2
  let final_lives := initial_lives - stage1_loss + stage1_gain - stage2_loss + stage2_gain - stage3_loss + stage3_gain
  final_lives = 23 :=
by
  sorry

end edward_lives_left_l169_169472


namespace units_digit_of_17_pow_2007_l169_169736

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_of_17_pow_2007 : units_digit (17 ^ 2007) = 3 := by
  have h : ∀ n, units_digit (17 ^ n) = units_digit (7 ^ n) := by
    intro n
    sorry  -- Same units digit logic for powers of 17 as for powers of 7.
  have pattern : units_digit (7 ^ 1) = 7 ∧ 
                 units_digit (7 ^ 2) = 9 ∧ 
                 units_digit (7 ^ 3) = 3 ∧ 
                 units_digit (7 ^ 4) = 1 := by
    sorry  -- Units digit pattern for powers of 7.
  have mod_cycle : 2007 % 4 = 3 := by
    sorry  -- Calculation of 2007 mod 4.
  have result : units_digit (7 ^ 2007) = units_digit (7 ^ 3) := by
    rw [← mod_eq_of_lt (by norm_num : 2007 % 4 < 4), mod_cycle]
    exact (and.left (and.right (and.right pattern)))  -- Extract units digit of 7^3 from pattern.
  rw [h]
  exact result

end units_digit_of_17_pow_2007_l169_169736


namespace function_satisfies_conditions_l169_169369

-- Define the functional equation condition
def functional_eq (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (f x + x * y) = f x * f (y + 1)

-- Lean statement for the proof problem
theorem function_satisfies_conditions (f : ℝ → ℝ) (h : functional_eq f) :
  (∀ x : ℝ, f x = 0) ∨ (∀ x : ℝ, f x = 1) ∨ (∀ x : ℝ, f x = x) :=
sorry

end function_satisfies_conditions_l169_169369


namespace derivative_of_y_l169_169372

variable (a b c x : ℝ)

def y : ℝ := (x - a) * (x - b) * (x - c)

theorem derivative_of_y :
  deriv (fun x:ℝ => (x - a) * (x - b) * (x - c)) x = 3 * x^2 - 2 * (a + b + c) * x + (a * b + a * c + b * c) :=
by
  sorry

end derivative_of_y_l169_169372


namespace no_nat_numbers_m_n_satisfy_eq_l169_169345

theorem no_nat_numbers_m_n_satisfy_eq (m n : ℕ) : ¬ (m^2 = n^2 + 2014) := sorry

end no_nat_numbers_m_n_satisfy_eq_l169_169345


namespace inequality_solution_l169_169476

theorem inequality_solution (x : ℝ) : 
  -1 < x ∧ x < 0 ∨ 0 < x ∧ x < 1 ∨ 3 ≤ x ∧ x < 4 → 
  (x + 6 ≥ 0) ∧ (x + 1 > 0) ∧ (5 - x > 0) ∧ (x ≠ 0) ∧ (x ≠ 1) ∧ (x ≠ 4) ∧
  ( (x - 3) / ((x - 1) * (4 - x)) ≥ 0 ) :=
sorry

end inequality_solution_l169_169476


namespace probability_both_correct_given_any_correct_l169_169461

-- Defining the probabilities
def P_A : ℚ := 3 / 5
def P_B : ℚ := 1 / 3

-- Defining the events and their products
def P_AnotB : ℚ := P_A * (1 - P_B)
def P_notAB : ℚ := (1 - P_A) * P_B
def P_AB : ℚ := P_A * P_B

-- Calculated Probability of C
def P_C : ℚ := P_AnotB + P_notAB + P_AB

-- The proof statement
theorem probability_both_correct_given_any_correct : (P_AB / P_C) = 3 / 11 :=
by
  sorry

end probability_both_correct_given_any_correct_l169_169461


namespace student_weekly_allowance_l169_169883

theorem student_weekly_allowance (A : ℝ) :
  (3 / 4) * (1 / 3) * ((2 / 5) * A + 4) - 2 = 0 ↔ A = 100/3 := sorry

end student_weekly_allowance_l169_169883


namespace diff_baseball_soccer_l169_169084

variable (totalBalls soccerBalls basketballs tennisBalls baseballs volleyballs : ℕ)

axiom h1 : totalBalls = 145
axiom h2 : soccerBalls = 20
axiom h3 : basketballs = soccerBalls + 5
axiom h4 : tennisBalls = 2 * soccerBalls
axiom h5 : baseballs > soccerBalls
axiom h6 : volleyballs = 30

theorem diff_baseball_soccer : baseballs - soccerBalls = 10 :=
  by {
    sorry
  }

end diff_baseball_soccer_l169_169084


namespace pages_left_to_read_l169_169079

-- Define the conditions
def total_pages : ℕ := 400
def percent_read : ℚ := 20 / 100
def pages_read := total_pages * percent_read

-- Define the question as a theorem
theorem pages_left_to_read (total_pages : ℕ) (percent_read : ℚ) (pages_read : ℚ) : ℚ :=
total_pages - pages_read

-- Assert the correct answer
example : pages_left_to_read total_pages percent_read pages_read = 320 := 
by
  sorry

end pages_left_to_read_l169_169079


namespace turtle_minimum_distance_l169_169887

theorem turtle_minimum_distance 
  (constant_speed : ℝ)
  (turn_angle : ℝ)
  (total_time : ℕ) :
  constant_speed = 5 →
  turn_angle = 90 →
  total_time = 11 →
  ∃ (final_position : ℝ × ℝ), 
    (final_position = (5, 0) ∨ final_position = (-5, 0) ∨ final_position = (0, 5) ∨ final_position = (0, -5)) ∧
    dist final_position (0, 0) = 5 :=
by
  intros
  sorry

end turtle_minimum_distance_l169_169887


namespace jason_initial_cards_l169_169061

-- Conditions
def cards_given_away : ℕ := 9
def cards_left : ℕ := 4

-- Theorem to prove
theorem jason_initial_cards : cards_given_away + cards_left = 13 :=
by
  sorry

end jason_initial_cards_l169_169061


namespace floor_sqrt_120_eq_10_l169_169913

theorem floor_sqrt_120_eq_10 : ⌊Real.sqrt 120⌋ = 10 := by
  -- Here, we note that we are given:
  -- 100 < 120 < 121 and the square root of it lies between 10 and 11
  sorry

end floor_sqrt_120_eq_10_l169_169913


namespace complex_number_location_second_quadrant_l169_169678

theorem complex_number_location_second_quadrant (z : ℂ) (h : z / (1 + I) = I) : z.re < 0 ∧ z.im > 0 :=
by sorry

end complex_number_location_second_quadrant_l169_169678


namespace regular_polygon_exterior_angle_l169_169401

theorem regular_polygon_exterior_angle (n : ℕ) (h : n > 2) (h_exterior : 36 = 360 / n) : n = 10 :=
sorry

end regular_polygon_exterior_angle_l169_169401


namespace A_fraction_simplification_l169_169797

noncomputable def A : ℚ := 
  ((3/8) * (13/5)) / ((5/2) * (6/5)) +
  ((5/8) * (8/5)) / (3 * (6/5) * (25/6)) +
  (20/3) * (3/25) +
  28 +
  (1 / 9) / 7 +
  (1/5) / (9 * 22)

theorem A_fraction_simplification :
  let num := 1901
  let denom := 3360
  (A = num / denom) :=
sorry

end A_fraction_simplification_l169_169797


namespace no_nat_m_n_square_diff_2014_l169_169335

theorem no_nat_m_n_square_diff_2014 :
  ¬ ∃ (m n : ℕ), m^2 = n^2 + 2014 :=
by
  sorry

end no_nat_m_n_square_diff_2014_l169_169335


namespace no_solution_exists_l169_169303

theorem no_solution_exists (m n : ℕ) : ¬ (m^2 = n^2 + 2014) :=
by
  sorry

end no_solution_exists_l169_169303


namespace triangle_third_side_length_l169_169221

theorem triangle_third_side_length
  (AC BC : ℝ)
  (h_a h_b h_c : ℝ)
  (half_sum_heights_eq : (h_a + h_b) / 2 = h_c) :
  AC = 6 → BC = 3 → AB = 4 :=
by
  sorry

end triangle_third_side_length_l169_169221


namespace final_volume_solution_l169_169554

variables (V2 V12 V_final : ℝ)

-- Given conditions
def V2_percent_solution (V2 : ℝ) := true
def V12_percent_solution (V12 : ℝ) := V12 = 18
def mixture_equation (V2 V12 V_final : ℝ) := 0.02 * V2 + 0.12 * V12 = 0.05 * V_final
def total_volume (V2 V12 V_final : ℝ) := V_final = V2 + V12

theorem final_volume_solution (V2 V_final : ℝ) (hV2: V2_percent_solution V2)
    (hV12 : V12_percent_solution V12) (h_mix : mixture_equation V2 V12 V_final)
    (h_total : total_volume V2 V12 V_final) : V_final = 60 :=
sorry

end final_volume_solution_l169_169554


namespace smallest_a_l169_169105

theorem smallest_a (a b : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 96 * a^2 = b^3) : a = 12 :=
by
  sorry

end smallest_a_l169_169105


namespace least_number_of_shoes_needed_on_island_l169_169426

def number_of_inhabitants : ℕ := 10000
def percentage_one_legged : ℕ := 5
def shoes_needed (N : ℕ) : ℕ :=
  let one_legged := (percentage_one_legged * N) / 100
  let two_legged := N - one_legged
  let barefooted_two_legged := two_legged / 2
  let shoes_for_one_legged := one_legged
  let shoes_for_two_legged := (two_legged - barefooted_two_legged) * 2
  shoes_for_one_legged + shoes_for_two_legged

theorem least_number_of_shoes_needed_on_island :
  shoes_needed number_of_inhabitants = 10000 :=
sorry

end least_number_of_shoes_needed_on_island_l169_169426


namespace latest_time_for_60_degrees_l169_169034

def temperature_at_time (t : ℝ) : ℝ :=
  -2 * t^2 + 16 * t + 40

theorem latest_time_for_60_degrees (t : ℝ) :
  temperature_at_time t = 60 → t = 5 :=
sorry

end latest_time_for_60_degrees_l169_169034


namespace simplify_fraction_l169_169986

theorem simplify_fraction (a b c : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) (h₃ : c ≠ 0) : 
  (8 * a^4 * b^2 * c) / (4 * a^3 * b) = 2 * a * b * c :=
by
  sorry

end simplify_fraction_l169_169986


namespace inverse_89_mod_90_l169_169642

theorem inverse_89_mod_90 : (89 * 89) % 90 = 1 := 
by
  sorry -- proof goes here

end inverse_89_mod_90_l169_169642


namespace part_I_part_II_l169_169241

noncomputable def f (x a : ℝ) : ℝ := |x - a| - |2 * x - 1|

theorem part_I (a : ℝ) (x : ℝ) (h : a = 2) :
    f x a + 3 ≥ 0 ↔ -4 ≤ x ∧ x ≤ 2 := 
by
    -- problem restatement
    sorry

theorem part_II (a : ℝ) (h : ∀ x : ℝ, 1 ≤ x ∧ x ≤ 3 → f x a ≤ 3) :
    -3 ≤ a ∧ a ≤ 5 := 
by
    -- problem restatement
    sorry

end part_I_part_II_l169_169241


namespace polygon_exterior_angle_l169_169403

theorem polygon_exterior_angle (n : ℕ) (h : 36 = 360 / n) : n = 10 :=
sorry

end polygon_exterior_angle_l169_169403


namespace no_nat_solutions_m_sq_eq_n_sq_plus_2014_l169_169354

theorem no_nat_solutions_m_sq_eq_n_sq_plus_2014 :
  ¬ ∃ (m n : ℕ), m ^ 2 = n ^ 2 + 2014 := 
sorry

end no_nat_solutions_m_sq_eq_n_sq_plus_2014_l169_169354


namespace student_game_incorrect_statement_l169_169580

theorem student_game_incorrect_statement (a : ℚ) : ¬ (∀ a : ℚ, -a - 2 < 0) :=
by
  -- skip the proof for now
  sorry

end student_game_incorrect_statement_l169_169580


namespace sum_of_variables_l169_169675

theorem sum_of_variables (a b c : ℝ) (h1 : a * b = 36) (h2 : a * c = 72) (h3 : b * c = 108) (ha : a = 2 * Real.sqrt 6) (hb : b = 3 * Real.sqrt 6) (hc : c = 6 * Real.sqrt 6) : 
  a + b + c = 11 * Real.sqrt 6 :=
by
  sorry

end sum_of_variables_l169_169675


namespace katie_five_dollar_bills_l169_169818

theorem katie_five_dollar_bills (x y : ℕ) (h1 : x + y = 12) (h2 : 5 * x + 10 * y = 80) : x = 8 :=
by
  sorry

end katie_five_dollar_bills_l169_169818


namespace tan_ratio_l169_169417

theorem tan_ratio (a b : ℝ) (ha : 0 < a ∧ a < π/2) (hb : 0 < b ∧ b < π/2)
  (h1 : Real.sin (a + b) = 5/8) (h2 : Real.sin (a - b) = 3/8) :
  (Real.tan a) / (Real.tan b) = 4 :=
by
  sorry

end tan_ratio_l169_169417


namespace choose_15_4_l169_169548

/-- The number of ways to choose 4 captains from a team of 15 people is 1365. -/
theorem choose_15_4 : nat.choose 15 4 = 1365 := by
  sorry

end choose_15_4_l169_169548


namespace grunters_win_all_five_l169_169244

theorem grunters_win_all_five (p : ℚ) (games : ℕ) (win_prob : ℚ) :
  games = 5 ∧ win_prob = 3 / 5 → 
  p = (win_prob) ^ games ∧ p = 243 / 3125 := 
by
  intros h
  cases h
  sorry

end grunters_win_all_five_l169_169244


namespace calc_expr_l169_169636

noncomputable def expr_val : ℝ :=
  Real.sqrt 4 - |(-(1 / 4 : ℝ))| + (Real.pi - 2)^0 + 2^(-2 : ℝ)

theorem calc_expr : expr_val = 3 := by
  sorry

end calc_expr_l169_169636


namespace sin_eq_one_fifth_l169_169032

noncomputable def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem sin_eq_one_fifth (ϕ : ℝ)
  (h : binomial_coefficient 5 3 * (Real.cos ϕ)^2 = 4) :
  Real.sin (2 * ϕ - π / 2) = 1 / 5 := sorry

end sin_eq_one_fifth_l169_169032


namespace largest_non_sum_l169_169119

theorem largest_non_sum (n : ℕ) : 
  ¬ (∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ b ∣ 2 ∧ n = 36 * a + b) ↔ n = 104 :=
by
  sorry

end largest_non_sum_l169_169119


namespace czakler_inequality_l169_169820

variable {a b : ℕ} (ha : a > 0) (hb : b > 0)
variable {c : ℝ} (hc : c > 0)

theorem czakler_inequality (h : (a + 1 : ℝ) / (b + c) = b / a) : c ≥ 1 := by
  sorry

end czakler_inequality_l169_169820


namespace result_of_fractions_mult_l169_169860

theorem result_of_fractions_mult (a b c d : ℚ) (x : ℕ) :
  a = 3 / 4 →
  b = 1 / 2 →
  c = 2 / 5 →
  d = 5100 →
  a * b * c * d = 765 := by
  sorry

end result_of_fractions_mult_l169_169860


namespace find_f1_l169_169665

def periodic (f : ℝ → ℝ) (p : ℝ) := ∀ x, f (x + p) = f x
def odd (f : ℝ → ℝ) := ∀ x, f (-x) = -f x

theorem find_f1 (f : ℝ → ℝ)
  (h_periodic : periodic f 2)
  (h_odd : odd f) :
  f 1 = 0 :=
sorry

end find_f1_l169_169665


namespace inequality_solution_m_range_l169_169133

def f (x : ℝ) : ℝ := abs (x - 2)
def g (x : ℝ) (m : ℝ) : ℝ := -abs (x + 3) + m

theorem inequality_solution (a : ℝ) :
  (∀ x : ℝ, a = 1 → f x + a - 1 > 0 ↔ x ≠ 2) ∧
  (a > 1 → ∀ x : ℝ, f x + a - 1 > 0 ↔ True) ∧
  (a < 1 → ∀ x : ℝ, f x + a - 1 > 0 ↔ x < a + 1 ∨ x > 3 - a) :=
by
  sorry

theorem m_range (m : ℝ) : (∀ x : ℝ, f x ≥ g x m) → m < 5 :=
by
  sorry

end inequality_solution_m_range_l169_169133


namespace matrix_power_four_l169_169763

noncomputable def matrixA : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![3 * Real.sqrt 2 / 2, -3 / 2], ![3 / 2, 3 * Real.sqrt 2 / 2]]

theorem matrix_power_four :
  matrixA ^ 4 = ![![ -81, 0], ![0, -81]] :=
by sorry

end matrix_power_four_l169_169763


namespace sequence_sum_l169_169528

-- Assume the sum of first n terms of the sequence {a_n} is given by S_n = n^2 + n + 1
def S (n : ℕ) : ℕ := n^2 + n + 1

-- The sequence a_8 + a_9 + a_10 + a_11 + a_12 is what we want to prove equals 100.
theorem sequence_sum : S 12 - S 7 = 100 :=
by
  sorry

end sequence_sum_l169_169528


namespace sara_walking_distance_l169_169436

noncomputable def circle_area := 616
noncomputable def pi_estimate := (22: ℚ) / 7
noncomputable def extra_distance := 3

theorem sara_walking_distance (r : ℚ) (radius_pos : 0 < r) : 
  pi_estimate * r^2 = circle_area →
  2 * pi_estimate * r + extra_distance = 91 :=
by
  intros h
  sorry

end sara_walking_distance_l169_169436


namespace evaluate_expression_l169_169174

theorem evaluate_expression : (↑7 ^ (1/4) / ↑7 ^ (1/6)) = (↑7 ^ (1/12)) :=
by
  sorry

end evaluate_expression_l169_169174


namespace taxi_fare_ride_distance_l169_169590

theorem taxi_fare_ride_distance (fare_first: ℝ) (first_mile: ℝ) (additional_fare_rate: ℝ) (additional_distance: ℝ) (total_amount: ℝ) (tip: ℝ) (x: ℝ) :
  fare_first = 3.00 ∧ first_mile = 0.75 ∧ additional_fare_rate = 0.25 ∧ additional_distance = 0.1 ∧ total_amount = 15 ∧ tip = 3 ∧
  (total_amount - tip) = fare_first + additional_fare_rate * (x - first_mile) / additional_distance → x = 4.35 :=
by
  intros
  sorry

end taxi_fare_ride_distance_l169_169590


namespace cattle_train_speed_is_56_l169_169276

variable (v : ℝ)

def cattle_train_speed :=
  let cattle_distance_until_diesel_starts := 6 * v
  let diesel_speed := v - 33
  let diesel_distance := 12 * diesel_speed
  let cattle_additional_distance := 12 * v
  let total_distance := cattle_distance_until_diesel_starts + diesel_distance + cattle_additional_distance
  total_distance = 1284

theorem cattle_train_speed_is_56 (h : cattle_train_speed v) : v = 56 :=
  sorry

end cattle_train_speed_is_56_l169_169276


namespace find_params_l169_169180

theorem find_params (a b c : ℝ) :
    (∀ x : ℝ, x = 2 ∨ x = -2 → x^5 + 4 * x^4 + a * x = b * x^2 + 4 * c) 
    → a = 16 ∧ b = 48 ∧ c = -32 :=
by
  sorry

end find_params_l169_169180


namespace Elberta_has_23_dollars_l169_169794

theorem Elberta_has_23_dollars (GrannySmith_has : ℕ := 72)
    (Anjou_has : ℕ := GrannySmith_has / 4)
    (Elberta_has : ℕ := Anjou_has + 5) : Elberta_has = 23 :=
by
  sorry

end Elberta_has_23_dollars_l169_169794


namespace no_nat_solutions_m2_eq_n2_plus_2014_l169_169310

theorem no_nat_solutions_m2_eq_n2_plus_2014 :
  ¬ ∃ (m n : ℕ), m^2 = n^2 + 2014 :=
by
  sorry

end no_nat_solutions_m2_eq_n2_plus_2014_l169_169310


namespace container_weight_l169_169143

noncomputable def weight_in_pounds : ℝ := 57 + 3/8
noncomputable def weight_in_ounces : ℝ := weight_in_pounds * 16
noncomputable def number_of_containers : ℝ := 7
noncomputable def ounces_per_container : ℝ := weight_in_ounces / number_of_containers

theorem container_weight :
  ounces_per_container = 131.142857 :=
by sorry

end container_weight_l169_169143


namespace quadratic_has_solutions_l169_169596

theorem quadratic_has_solutions :
  (1 + Real.sqrt 2)^2 - 2 * (1 + Real.sqrt 2) - 1 = 0 ∧ 
  (1 - Real.sqrt 2)^2 - 2 * (1 - Real.sqrt 2) - 1 = 0 :=
by
  sorry

end quadratic_has_solutions_l169_169596


namespace number_of_ways_to_choose_4_captains_from_15_l169_169544

def choose_captains (n r : ℕ) : ℕ :=
  Nat.choose n r

theorem number_of_ways_to_choose_4_captains_from_15 :
  choose_captains 15 4 = 1365 := by
  sorry

end number_of_ways_to_choose_4_captains_from_15_l169_169544


namespace find_a_l169_169792

noncomputable def set_A (a : ℝ) : Set ℝ := {a + 2, 2 * a^2 + a}

theorem find_a (a : ℝ) (h : 3 ∈ set_A a) : a = -3 / 2 :=
by
  sorry

end find_a_l169_169792


namespace total_cost_of_lollipops_l169_169240

/-- Given Sarah bought 12 lollipops and shared one-quarter of them, 
    and Julie reimbursed Sarah 75 cents for the shared lollipops,
    Prove that the total cost of the lollipops in dollars is $3. --/
theorem total_cost_of_lollipops 
(Sarah_lollipops : ℕ) 
(shared_fraction : ℚ) 
(Julie_paid : ℚ) 
(total_lollipops_cost : ℚ)
(h1 : Sarah_lollipops = 12) 
(h2 : shared_fraction = 1/4) 
(h3 : Julie_paid = 75 / 100) 
(h4 : total_lollipops_cost = 
        ((Julie_paid / (Sarah_lollipops * shared_fraction)) * Sarah_lollipops / 100)) :
total_lollipops_cost = 3 := 
sorry

end total_cost_of_lollipops_l169_169240


namespace totalCandy_l169_169183

-- Define the number of pieces of candy each person had
def TaquonCandy : ℕ := 171
def MackCandy : ℕ := 171
def JafariCandy : ℕ := 76

-- Prove that the total number of pieces of candy they had together is 418
theorem totalCandy : TaquonCandy + MackCandy + JafariCandy = 418 := by
  sorry

end totalCandy_l169_169183


namespace four_op_two_l169_169562

def op (a b : ℝ) : ℝ := 2 * a + 5 * b

theorem four_op_two : op 4 2 = 18 := by
  sorry

end four_op_two_l169_169562


namespace matrix_power_eq_l169_169764

noncomputable def rotation_matrix (θ : ℝ) : Matrix 2 2 ℝ :=
  ![![Real.cos θ, -Real.sin θ], ![Real.sin θ, Real.cos θ]]

theorem matrix_power_eq :
  (3 • (rotation_matrix (Real.pi / 4))) ^ 4 =
    81 • ![![(-1 : ℝ), 0], ![0, -1]] :=
by sorry

end matrix_power_eq_l169_169764


namespace only_one_way_to_center_l169_169841

def is_center {n : ℕ} (grid_size n : ℕ) (coord : ℕ × ℕ) : Prop :=
  coord = (grid_size / 2 + 1, grid_size / 2 + 1)

def count_ways_to_center : ℕ :=
  if h : (1 <= 3 ∧ 3 <= 5) then 1 else 0

theorem only_one_way_to_center : count_ways_to_center = 1 := by
  sorry

end only_one_way_to_center_l169_169841


namespace two_pow_n_add_two_gt_n_sq_l169_169896

open Nat

theorem two_pow_n_add_two_gt_n_sq (n : ℕ) (h : n > 0) : 2^n + 2 > n^2 :=
by
  sorry

end two_pow_n_add_two_gt_n_sq_l169_169896


namespace purchasing_methods_l169_169138

theorem purchasing_methods :
  ∃ (s : Finset (ℕ × ℕ)), s.card = 7 ∧
    ∀ (x y : ℕ), (x, y) ∈ s ↔ 60 * x + 70 * y ≤ 500 ∧ 3 ≤ x ∧ 2 ≤ y :=
sorry

end purchasing_methods_l169_169138


namespace equilateral_triangle_l169_169812

theorem equilateral_triangle (a b c : ℝ) (h1 : a^4 = b^4 + c^4 - b^2 * c^2) (h2 : b^4 = a^4 + c^4 - a^2 * c^2) : 
  a = b ∧ b = c ∧ c = a :=
by sorry

end equilateral_triangle_l169_169812


namespace fraction_sum_ratio_l169_169399

theorem fraction_sum_ratio :
  let A := (Finset.range 1002).sum (λ k => 1 / ((2 * k + 1) * (2 * k + 2)))
  let B := (Finset.range 1002).sum (λ k => 1 / ((1003 + k) * (2004 - k)))
  (A / B) = (3007 / 2) :=
by
  sorry

end fraction_sum_ratio_l169_169399


namespace product_of_three_numbers_summing_to_eleven_l169_169452

def numbers : List ℕ := [2, 3, 4, 6]

theorem product_of_three_numbers_summing_to_eleven : 
  ∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a ∈ numbers ∧ b ∈ numbers ∧ c ∈ numbers ∧ a + b + c = 11 ∧ a * b * c = 36 := 
by
  sorry

end product_of_three_numbers_summing_to_eleven_l169_169452


namespace articles_produced_l169_169530

theorem articles_produced (a b c p q r : Nat) (h : a * b * c = abc) : p * q * r = pqr := sorry

end articles_produced_l169_169530


namespace min_value_fraction_l169_169022

theorem min_value_fraction (m n : ℝ) (hm : m > 0) (hn : n > 0) (h : 2 * m + n = 1) : 
  ∀ x, (x = (1 / m + 8 / n)) → x ≥ 18 :=
by
  sorry

end min_value_fraction_l169_169022


namespace right_triangle_leg_lengths_l169_169965

theorem right_triangle_leg_lengths (a b c : ℕ) (h : a ^ 2 + b ^ 2 = c ^ 2) (h1: c = 17) (h2: a + (c - b) = 17) (h3: b + (c - a) = 17) : a = 8 ∧ b = 15 :=
by {
  sorry
}

end right_triangle_leg_lengths_l169_169965


namespace minimize_payment_l169_169724

theorem minimize_payment :
  ∀ (bd_A td_A bd_B td_B bd_C td_C : ℕ),
    bd_A = 42 → td_A = 36 →
    bd_B = 48 → td_B = 41 →
    bd_C = 54 → td_C = 47 →
    ∃ (S : ℕ), S = 36 ∧ 
      (S = bd_A - (bd_A - td_A)) ∧
      (S < bd_B - (bd_B - td_B)) ∧
      (S < bd_C - (bd_C - td_C)) := 
by {
  sorry
}

end minimize_payment_l169_169724


namespace total_amount_paid_l169_169558

-- Definitions of the conditions
def cost_earbuds : ℝ := 200
def tax_rate : ℝ := 0.15

-- Statement to prove
theorem total_amount_paid : (cost_earbuds + (cost_earbuds * tax_rate)) = 230 := sorry

end total_amount_paid_l169_169558


namespace no_naturals_satisfy_m_squared_eq_n_squared_plus_2014_l169_169328

theorem no_naturals_satisfy_m_squared_eq_n_squared_plus_2014 :
  ∀ (m n : ℕ), ¬ (m^2 = n^2 + 2014) :=
by
  intro m n
  sorry

end no_naturals_satisfy_m_squared_eq_n_squared_plus_2014_l169_169328


namespace AndrewAge_l169_169284

noncomputable def AndrewAgeProof : Prop :=
  ∃ (a g : ℕ), g = 10 * a ∧ g - a = 45 ∧ a = 5

-- Proof is not required, so we use sorry to skip the proof.
theorem AndrewAge : AndrewAgeProof := by
  sorry

end AndrewAge_l169_169284


namespace zoe_takes_correct_amount_of_money_l169_169870

def numberOfPeople : ℕ := 6
def costPerSoda : ℝ := 0.5
def costPerPizza : ℝ := 1.0

def totalCost : ℝ := (numberOfPeople * costPerSoda) + (numberOfPeople * costPerPizza)

theorem zoe_takes_correct_amount_of_money : totalCost = 9 := sorry

end zoe_takes_correct_amount_of_money_l169_169870


namespace vertical_asymptotes_sum_l169_169994

theorem vertical_asymptotes_sum : 
  (∀ x : ℝ, 4 * x^2 + 7 * x + 3 = 0 → x = -3 / 4 ∨ x = -1) →
  (-3 / 4) + (-1) = -7 / 4 :=
by
  intro h
  sorry

end vertical_asymptotes_sum_l169_169994


namespace inequality_proof_l169_169957

theorem inequality_proof (x a : ℝ) (h1 : x > a) (h2 : a > 0) : x^2 > ax ∧ ax > a^2 :=
by
  sorry

end inequality_proof_l169_169957


namespace cards_divisible_by_100_l169_169009

open Nat

-- Define the problem statement
theorem cards_divisible_by_100 :
  let cards := Finset.range 5000
  let valid_pairs := cards.filter (λ n, ∃ m ∈ cards, (n + m) % 100 = 0)
  valid_pairs.card = 124950 :=
by
  sorry

end cards_divisible_by_100_l169_169009


namespace calc_expression_l169_169288

theorem calc_expression : (113^2 - 104^2) / 9 = 217 := by
  sorry

end calc_expression_l169_169288


namespace wendy_tooth_extraction_cost_eq_290_l169_169599

def dentist_cleaning_cost : ℕ := 70
def dentist_filling_cost : ℕ := 120
def wendy_dentist_bill : ℕ := 5 * dentist_filling_cost
def wendy_cleaning_and_fillings_cost : ℕ := dentist_cleaning_cost + 2 * dentist_filling_cost
def wendy_tooth_extraction_cost : ℕ := wendy_dentist_bill - wendy_cleaning_and_fillings_cost

theorem wendy_tooth_extraction_cost_eq_290 : wendy_tooth_extraction_cost = 290 := by
  sorry

end wendy_tooth_extraction_cost_eq_290_l169_169599


namespace Ron_book_picking_times_l169_169086

theorem Ron_book_picking_times (couples members : ℕ) (weeks people : ℕ) (Ron wife picks_per_year : ℕ) 
  (h1 : couples = 3) 
  (h2 : members = 5) 
  (h3 : Ron = 1) 
  (h4 : wife = 1) 
  (h5 : weeks = 52) 
  (h6 : people = 2 * couples + members + Ron + wife) 
  (h7 : picks_per_year = weeks / people) 
  : picks_per_year = 4 :=
by
  -- Definition steps can be added here if needed, currently immediate from conditions h1 to h7
  sorry

end Ron_book_picking_times_l169_169086


namespace square_b_perimeter_l169_169988

theorem square_b_perimeter (a b : ℝ) 
  (ha : a^2 = 65) 
  (prob : (65 - b^2) / 65 = 0.7538461538461538) : 
  4 * b = 16 :=
by 
  sorry

end square_b_perimeter_l169_169988


namespace power_point_relative_to_circle_l169_169576

noncomputable def circle_power (a b R x1 y1 : ℝ) : ℝ :=
  (x1 - a) ^ 2 + (y1 - b) ^ 2 - R ^ 2

theorem power_point_relative_to_circle (a b R x1 y1 : ℝ) :
  (x1 - a) ^ 2 + (y1 - b) ^ 2 - R ^ 2 = circle_power a b R x1 y1 := by
  unfold circle_power
  sorry

end power_point_relative_to_circle_l169_169576


namespace deepak_present_age_l169_169721

-- Let R be Rahul's current age and D be Deepak's current age
variables (R D : ℕ)

-- Given conditions
def ratio_condition : Prop := (4 : ℚ) / 3 = (R : ℚ) / D
def rahul_future_age_condition : Prop := R + 6 = 50

-- Prove Deepak's present age D is 33 years
theorem deepak_present_age : ratio_condition R D ∧ rahul_future_age_condition R → D = 33 := 
sorry

end deepak_present_age_l169_169721


namespace identity_1_over_n_n_plus_1_sum_series_1_over_k_k_plus_1_sum_series_1_over_3k_minus_2_3k_plus_1_l169_169698

-- Question 1: Prove the given identity for 1/(n(n+1))
theorem identity_1_over_n_n_plus_1 (n : ℕ) (hn : n ≠ 0) : 
  (1 : ℚ) / (n * (n + 1)) = (1 : ℚ) / n - (1 : ℚ) / (n + 1) :=
by
  sorry

-- Question 2: Prove the sum of series 1/k(k+1) from k=1 to k=2021
theorem sum_series_1_over_k_k_plus_1 : 
  (Finset.range 2021).sum (λ k => (1 : ℚ) / (k+1) / (k+2)) = 2021 / 2022 :=
by
  sorry

-- Question 3: Prove the sum of series 1/(3k-2)(3k+1) from k=1 to k=673
theorem sum_series_1_over_3k_minus_2_3k_plus_1 : 
  (Finset.range 673).sum (λ k => (1 : ℚ) / ((3 * k + 1 - 2) * (3 * k + 1))) = 674 / 2023 :=
by
  sorry

end identity_1_over_n_n_plus_1_sum_series_1_over_k_k_plus_1_sum_series_1_over_3k_minus_2_3k_plus_1_l169_169698


namespace black_cards_taken_out_l169_169709

theorem black_cards_taken_out (initial_black : ℕ) (remaining_black : ℕ) (total_cards : ℕ) (black_cards_per_deck : ℕ) :
  total_cards = 52 → black_cards_per_deck = 26 →
  initial_black = black_cards_per_deck → remaining_black = 22 →
  initial_black - remaining_black = 4 := by
  intros
  sorry

end black_cards_taken_out_l169_169709


namespace choose_4_from_15_l169_169540

theorem choose_4_from_15 : Nat.choose 15 4 = 1365 := by
  sorry

end choose_4_from_15_l169_169540


namespace inverse_89_mod_90_l169_169641

theorem inverse_89_mod_90 : (89 * 89) % 90 = 1 := 
by
  sorry -- proof goes here

end inverse_89_mod_90_l169_169641


namespace no_solution_exists_l169_169301

theorem no_solution_exists (m n : ℕ) : ¬ (m^2 = n^2 + 2014) :=
by
  sorry

end no_solution_exists_l169_169301


namespace prove_relationship_l169_169467

noncomputable def relationship_x_y_z (x y z : ℝ) (t : ℝ) : Prop :=
  (x / Real.sin t) = (y / Real.sin (2 * t)) ∧ (x / Real.sin t) = (z / Real.sin (3 * t))

theorem prove_relationship (x y z t : ℝ) (h : relationship_x_y_z x y z t) : x^2 - y^2 + x * z = 0 :=
by
  sorry

end prove_relationship_l169_169467


namespace tan_alpha_cos2alpha_plus_2sin2alpha_l169_169674

theorem tan_alpha_cos2alpha_plus_2sin2alpha (α : ℝ) (h : Real.tan α = 3 / 4) : 
  Real.cos α ^ 2 + 2 * Real.sin (2 * α) = 64 / 25 :=
by
  sorry

end tan_alpha_cos2alpha_plus_2sin2alpha_l169_169674


namespace percentage_decrease_in_speed_l169_169149

variable (S : ℝ) (S' : ℝ) (T T' : ℝ)

noncomputable def percentageDecrease (originalSpeed decreasedSpeed : ℝ) : ℝ :=
  ((originalSpeed - decreasedSpeed) / originalSpeed) * 100

theorem percentage_decrease_in_speed :
  T = 40 ∧ T' = 50 ∧ S' = (4 / 5) * S →
  percentageDecrease S S' = 20 :=
by sorry

end percentage_decrease_in_speed_l169_169149


namespace sum_symmetry_l169_169437

-- Definitions of minimum and maximum faces for dice in the problem
def min_face := 2
def max_face := 7
def num_dice := 8

-- Definitions of the minimum and maximum sum outcomes
def min_sum := num_dice * min_face
def max_sum := num_dice * max_face

-- Definition of the average value for symmetry
def avg_sum := (min_sum + max_sum) / 2

-- Definition of the probability symmetry theorem
theorem sum_symmetry (S : ℕ) : 
  (min_face <= S) ∧ (S <= max_face * num_dice) → 
  ∃ T, T = 2 * avg_sum - S ∧ T = 52 :=
by
  sorry

end sum_symmetry_l169_169437


namespace initial_shipment_robot_rascals_l169_169884

theorem initial_shipment_robot_rascals 
(T : ℝ) 
(h1 : (0.7 * T = 168)) : 
  T = 240 :=
sorry

end initial_shipment_robot_rascals_l169_169884


namespace sum_of_brothers_ages_l169_169727

theorem sum_of_brothers_ages (Bill Eric: ℕ) 
  (h1: 4 = Bill - Eric) 
  (h2: Bill = 16) : 
  Bill + Eric = 28 := 
by 
  sorry

end sum_of_brothers_ages_l169_169727


namespace one_over_nine_inv_half_eq_three_l169_169289

theorem one_over_nine_inv_half_eq_three : (1 / 9 : ℝ) ^ (-1 / 2 : ℝ) = 3 := 
by
  sorry

end one_over_nine_inv_half_eq_three_l169_169289


namespace product_units_tens_not_divisible_by_5_l169_169490

-- Define the list of four-digit numbers
def numbers : List ℕ := [4750, 4760, 4775, 4785, 4790]

-- Define a function to check if a number is divisible by 5
def divisible_by_5 (n : ℕ) : Prop := n % 5 = 0

-- Define a function to extract the units digit of a number
def units_digit (n : ℕ) : ℕ := n % 10

-- Define a function to extract the tens digit of a number
def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

-- Statement: The product of the units digit and the tens digit of the number
-- that is not divisible by 5 in the list is 0
theorem product_units_tens_not_divisible_by_5 : 
  ∃ n ∈ numbers, ¬divisible_by_5 n ∧ (units_digit n * tens_digit n = 0) :=
by sorry

end product_units_tens_not_divisible_by_5_l169_169490


namespace alan_must_eat_more_l169_169973

theorem alan_must_eat_more (
  kevin_eats_total : ℕ,
  kevin_time : ℕ,
  alan_eats_rate : ℕ
) (h_kevin_eats_total : kevin_eats_total = 64) 
  (h_kevin_time : kevin_time = 8)
  (h_alan_eats_rate : alan_eats_rate = 5)
  (kevin_rate_gt_alan_rate : (kevin_eats_total / kevin_time) > alan_eats_rate) :
  ∃ wings_more_per_minute : ℕ, wings_more_per_minute = 4 :=
by
  sorry

end alan_must_eat_more_l169_169973


namespace female_voters_percentage_is_correct_l169_169537

def percentage_of_population_that_are_female_voters
  (female_percentage : ℝ)
  (voter_percentage_of_females : ℝ) : ℝ :=
  female_percentage * voter_percentage_of_females * 100

theorem female_voters_percentage_is_correct :
  percentage_of_population_that_are_female_voters 0.52 0.4 = 20.8 := by
  sorry

end female_voters_percentage_is_correct_l169_169537


namespace no_solution_exists_l169_169307

theorem no_solution_exists (m n : ℕ) : ¬ (m^2 = n^2 + 2014) :=
by
  sorry

end no_solution_exists_l169_169307


namespace geo_seq_second_term_l169_169448

theorem geo_seq_second_term (b r : Real) 
  (h1 : 280 * r = b) 
  (h2 : b * r = 90 / 56) 
  (h3 : b > 0) 
  : b = 15 * Real.sqrt 2 := 
by 
  sorry

end geo_seq_second_term_l169_169448


namespace alan_more_wings_per_minute_to_beat_record_l169_169974

-- Define relevant parameters and conditions
def kevin_wings := 64
def time_minutes := 8
def alan_rate := 5

-- Theorem: Alan must eat 3 more wings per minute to beat Kevin's record
theorem alan_more_wings_per_minute_to_beat_record : 
  (kevin_wings > alan_rate * time_minutes) → ((kevin_wings - (alan_rate * time_minutes)) / time_minutes = 3) :=
by
  sorry

end alan_more_wings_per_minute_to_beat_record_l169_169974


namespace no_nat_solutions_m_sq_eq_n_sq_plus_2014_l169_169350

theorem no_nat_solutions_m_sq_eq_n_sq_plus_2014 :
  ¬ ∃ (m n : ℕ), m ^ 2 = n ^ 2 + 2014 := 
sorry

end no_nat_solutions_m_sq_eq_n_sq_plus_2014_l169_169350


namespace probability_of_perfect_square_sum_l169_169726

def two_dice_probability_of_perfect_square_sum : ℚ :=
  let totalOutcomes := 12 * 12
  let perfectSquareOutcomes := 3 + 8 + 9 -- ways to get sums 4, 9, and 16
  (perfectSquareOutcomes : ℚ) / (totalOutcomes : ℚ)

theorem probability_of_perfect_square_sum :
  two_dice_probability_of_perfect_square_sum = 5 / 36 :=
by
  sorry

end probability_of_perfect_square_sum_l169_169726


namespace height_of_triangle_is_5_l169_169246

def base : ℝ := 4
def area : ℝ := 10

theorem height_of_triangle_is_5 :
  ∃ (height : ℝ), (base * height) / 2 = area ∧ height = 5 :=
by
  sorry

end height_of_triangle_is_5_l169_169246


namespace incorrect_operation_l169_169123

theorem incorrect_operation 
    (x y : ℝ) :
    (x - y) / (x + y) = (y - x) / (y + x) ↔ False := 
by 
  sorry

end incorrect_operation_l169_169123


namespace largest_factor_of_form_l169_169274

theorem largest_factor_of_form (n : ℕ) (h : n % 10 = 4) : 120 ∣ n * (n + 1) * (n + 2) :=
sorry

end largest_factor_of_form_l169_169274


namespace sum_of_consecutive_ints_product_eq_336_l169_169775

def consecutive_ints_sum (a b c : ℤ) : Prop :=
  b = a + 1 ∧ c = b + 1

theorem sum_of_consecutive_ints_product_eq_336 (a b c : ℤ) (h1 : consecutive_ints_sum a b c) (h2 : a * b * c = 336) :
  a + b + c = 21 :=
sorry

end sum_of_consecutive_ints_product_eq_336_l169_169775


namespace quadratic_real_roots_l169_169783

-- Define the quadratic equation
def quadratic_eq (a x : ℝ) : ℝ :=
  (a - 1) * x^2 - 2 * x + 1

-- Define the discriminant of the quadratic equation
def discriminant (a : ℝ) : ℝ :=
  4 - 4 * (a - 1)

-- The main theorem stating the needed proof problem
theorem quadratic_real_roots (a : ℝ) : (∃ x : ℝ, quadratic_eq a x = 0) ↔ a ≤ 2 := by
  -- Proof will be inserted here
  sorry

end quadratic_real_roots_l169_169783


namespace angle_C_is_pi_div_three_l169_169683

-- Definitions for the problem
variables {α : Type} [linear_ordered_field α] {a b c : α} {A B C : ℝ}

-- Assuming we have a proof that (a + c) * (sin A - sin C) = b * (sin A - sin B)
axiom equation : (a + c) * (Real.sin A - Real.sin C) = b * (Real.sin A - Real.sin B)

-- Prove that angle C is π / 3, given the conditions
theorem angle_C_is_pi_div_three (h : (a + c) * (Real.sin A - Real.sin C) = b * (Real.sin A - Real.sin B)) :
  C = π / 3 :=
by 
  sorry -- skip the proof

end angle_C_is_pi_div_three_l169_169683


namespace general_admission_price_l169_169486

theorem general_admission_price :
  ∃ x : ℝ,
    ∃ G V : ℕ,
      VIP_price = 45 ∧ Total_tickets_sold = 320 ∧ Total_revenue = 7500 ∧ VIP_tickets_less = 276 ∧
      G + V = Total_tickets_sold ∧ V = G - VIP_tickets_less ∧ 45 * V + x * G = Total_revenue ∧ x = 21.85 :=
sorry

end general_admission_price_l169_169486


namespace neither_necessary_nor_sufficient_condition_l169_169192

def red_balls := 5
def yellow_balls := 3
def white_balls := 2
def total_balls := red_balls + yellow_balls + white_balls

def event_A_occurs := ∃ (r : ℕ) (y : ℕ), (r ≤ red_balls) ∧ (y ≤ yellow_balls) ∧ (r = 1) ∧ (y = 1)
def event_B_occurs := ∃ (x y : ℕ), (x ≤ total_balls) ∧ (y ≤ total_balls) ∧ (x ≠ y)

theorem neither_necessary_nor_sufficient_condition :
  ¬(¬event_A_occurs → ¬event_B_occurs) ∧ ¬(¬event_B_occurs → ¬event_A_occurs) := 
sorry

end neither_necessary_nor_sufficient_condition_l169_169192


namespace smallest_sum_xy_min_45_l169_169382

theorem smallest_sum_xy_min_45 (x y : ℕ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x ≠ y) (h4 : 1 / (x : ℝ) + 1 / (y : ℝ) = 1 / 10) :
  x + y = 45 :=
by
  sorry

end smallest_sum_xy_min_45_l169_169382


namespace product_between_21st_and_24th_multiple_of_3_l169_169104

theorem product_between_21st_and_24th_multiple_of_3 : 
  (66 * 69 = 4554) :=
by
  sorry

end product_between_21st_and_24th_multiple_of_3_l169_169104


namespace tan_alpha_plus_pi_over_3_sin_cos_ratio_l169_169195

theorem tan_alpha_plus_pi_over_3
  (α : ℝ)
  (h : Real.tan (α / 2) = 3) :
  Real.tan (α + Real.pi / 3) = (48 - 25 * Real.sqrt 3) / 11 := 
sorry

theorem sin_cos_ratio
  (α : ℝ)
  (h : Real.tan (α / 2) = 3) :
  (Real.sin α + 2 * Real.cos α) / (3 * Real.sin α - 2 * Real.cos α) = -5 / 17 :=
sorry

end tan_alpha_plus_pi_over_3_sin_cos_ratio_l169_169195


namespace number_of_red_parrots_l169_169572

-- Defining the conditions from a)
def fraction_yellow_parrots : ℚ := 2 / 3
def total_birds : ℕ := 120

-- Stating the theorem we want to prove
theorem number_of_red_parrots (H1 : fraction_yellow_parrots = 2 / 3) (H2 : total_birds = 120) : 
  (1 - fraction_yellow_parrots) * total_birds = 40 := 
by 
  sorry

end number_of_red_parrots_l169_169572


namespace right_triangle_inequality_l169_169981

theorem right_triangle_inequality (a b c : ℝ) (h : a^2 + b^2 = c^2) :
  a^4 + b^4 < c^4 :=
by
  sorry

end right_triangle_inequality_l169_169981


namespace minimum_photos_l169_169051

theorem minimum_photos (M N T : Type) [DecidableEq M] [DecidableEq N] [DecidableEq T] 
  (monuments : Finset M) (city : N) (tourists : Finset T) 
  (photos : T → Finset M) :
  monuments.card = 3 → 
  tourists.card = 42 → 
  (∀ t : T, (photos t).card ≤ 3) → 
  (∀ t₁ t₂ : T, t₁ ≠ t₂ → (photos t₁ ∪ photos t₂) = monuments) → 
  ∑ t in tourists, (photos t).card ≥ 123 := 
by
  intros h_monuments_card h_tourists_card h_photos_card h_photos_union
  sorry

end minimum_photos_l169_169051


namespace total_books_on_shelves_l169_169451

theorem total_books_on_shelves (shelves books_per_shelf : ℕ) (h_shelves : shelves = 350) (h_books_per_shelf : books_per_shelf = 25) :
  shelves * books_per_shelf = 8750 :=
by {
  sorry
}

end total_books_on_shelves_l169_169451


namespace label_sum_l169_169262

theorem label_sum (n : ℕ) : 
  (∃ S : ℕ → ℕ, S 1 = 2 ∧ (∀ k, k > 1 → (S (k + 1) = 2 * S k)) ∧ S n = 2 * 3 ^ (n - 1)) := 
sorry

end label_sum_l169_169262


namespace probability_Q_eq_i_l169_169591

noncomputable def vertices : set ℂ := 
  { complex.sqrt 2 * complex.I, -complex.sqrt 2 * complex.I, complex.sqrt 2, -complex.sqrt 2,
    (1 + complex.I) / complex.sqrt 8, (-1 + complex.I) / complex.sqrt 8, 
    (1 - complex.I) / complex.sqrt 8, (-1 - complex.I) / complex.sqrt 8 }

def selected_vertices : ℕ → ℂ
| k := if h : k < 16 then classical.some (classical.some_spec $ finset.exists_mem vertices.to_finset) else 0

def Q : ℂ := ∏ k in finset.range 16, selected_vertices k

theorem probability_Q_eq_i :
  ∃ c d q : ℕ, nat.prime q ∧ ¬ q ∣ c ∧ Q = complex.I →
  ∃ c d q : ℕ, nat.prime q ∧ ¬ q ∣ c ∧ (c + d + q = 6452) ∧ 
  (∃ (p : ℚ), p = 6435 / 32768) :=
begin
  sorry,
end

end probability_Q_eq_i_l169_169591


namespace ratio_y_share_to_total_l169_169150

theorem ratio_y_share_to_total
  (total_profit : ℝ)
  (diff_share : ℝ)
  (h_total : total_profit = 800)
  (h_diff : diff_share = 160) :
  ∃ (a b : ℝ), (b / (a + b) = 2 / 5) ∧ (|a - b| = (a + b) / 5) :=
by
  sorry

end ratio_y_share_to_total_l169_169150


namespace range_of_independent_variable_l169_169998

theorem range_of_independent_variable (x : ℝ) :
  (sqrt (x - 1)).nonneg → x ≥ 1 :=
by
  sorry

end range_of_independent_variable_l169_169998


namespace inverse_89_mod_90_l169_169646

theorem inverse_89_mod_90 : (89 * 89) % 90 = 1 := 
by
  -- Mathematical proof is skipped
  sorry

end inverse_89_mod_90_l169_169646


namespace compute_product_l169_169498

theorem compute_product : (100 - 5) * (100 + 5) = 9975 := by
  sorry

end compute_product_l169_169498


namespace paths_from_A_to_B_l169_169872

def path_count_A_to_B : Nat :=
  let red_to_blue_ways := [2, 3]  -- 2 ways to first blue, 3 ways to second blue
  let blue_to_green_ways_first := 4 * 2  -- Each of the 2 green arrows from first blue, 4 ways each
  let blue_to_green_ways_second := 5 * 2 -- Each of the 2 green arrows from second blue, 5 ways each
  let green_to_B_ways_first := 2 * blue_to_green_ways_first  -- Each of the first green, 2 ways each
  let green_to_B_ways_second := 3 * blue_to_green_ways_second  -- Each of the second green, 3 ways each
  green_to_B_ways_first + green_to_B_ways_second  -- Total paths from green arrows to B

theorem paths_from_A_to_B : path_count_A_to_B = 46 := by
  sorry

end paths_from_A_to_B_l169_169872


namespace f_shift_l169_169693

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x^2 - 4 * x + 2

-- Define the main theorem
theorem f_shift (x h : ℝ) : f (x + h) - f x = h * (6 * x + 3 * h - 4) :=
by
  sorry

end f_shift_l169_169693


namespace lines_coplanar_iff_k_eq_neg2_l169_169834

noncomputable def line1 (s k : ℝ) : ℝ × ℝ × ℝ :=
(2 + s, 4 - k * s, 2 + k * s)

noncomputable def line2 (t : ℝ) : ℝ × ℝ × ℝ :=
(t, 2 + 2 * t, 3 - t)

theorem lines_coplanar_iff_k_eq_neg2 :
  (∃ s t : ℝ, line1 s k = line2 t) → k = -2 :=
by
  sorry

end lines_coplanar_iff_k_eq_neg2_l169_169834


namespace rank_abc_l169_169560

noncomputable def a := 6 ^ 0.4
noncomputable def b := real.log 0.5 / real.log 0.4
noncomputable def c := real.log 0.4 / real.log 8

theorem rank_abc : c < b ∧ b < a :=
by
  -- proof steps go here
  sorry

end rank_abc_l169_169560


namespace ferris_wheel_cost_l169_169609

theorem ferris_wheel_cost (roller_coaster_cost log_ride_cost zach_initial_tickets zach_additional_tickets total_tickets ferris_wheel_cost : ℕ) 
  (h1 : roller_coaster_cost = 7)
  (h2 : log_ride_cost = 1)
  (h3 : zach_initial_tickets = 1)
  (h4 : zach_additional_tickets = 9)
  (h5 : total_tickets = zach_initial_tickets + zach_additional_tickets)
  (h6 : total_tickets - (roller_coaster_cost + log_ride_cost) = ferris_wheel_cost) :
  ferris_wheel_cost = 2 := 
by
  sorry

end ferris_wheel_cost_l169_169609


namespace triangle_height_BF_l169_169925

theorem triangle_height_BF 
  (B A C E F : Point)
  (BE_circumcenter: is_circumcenter_on (triangle B A C) (ray B E))
  (cond_1: intersection (line_segment B E) (line_segment A C) = some E)
  (cond_2: AF_FE_val : (length (line_segment A F)) * (length (line_segment F E)) = 5)
  (cond_3: cot_ratio : cot (angle E B C) / cot (angle B E C) = 3 / 4)
  : length (line_segment B F) = 1.94 := 
by
  sorry

end triangle_height_BF_l169_169925


namespace LCM_of_fractions_l169_169857

noncomputable def LCM (a b : Rat) : Rat :=
  a * b / (gcd a.num b.num / gcd a.den b.den : Int)

theorem LCM_of_fractions (x : ℤ) (h : x ≠ 0) :
  LCM (1 / (4 * x : ℚ)) (LCM (1 / (6 * x : ℚ)) (1 / (9 * x : ℚ))) = 1 / (36 * x) :=
by
  sorry

end LCM_of_fractions_l169_169857


namespace equal_distribution_l169_169368

variables (Emani Howard : ℕ)

-- Emani has $30 more than Howard
axiom emani_condition : Emani = Howard + 30

-- Emani has $150
axiom emani_has_money : Emani = 150

theorem equal_distribution : (Emani + Howard) / 2 = 135 :=
by
  sorry

end equal_distribution_l169_169368


namespace exists_increasing_infinite_sequence_of_perfect_squares_divisible_by_13_power_l169_169000

open Nat

theorem exists_increasing_infinite_sequence_of_perfect_squares_divisible_by_13_power :
  ∃ (a : ℕ → ℕ), (∀ k : ℕ, (∃ b : ℕ, a k = b ^ 2)) ∧ (StrictMono a) ∧ (∀ k : ℕ, 13^k ∣ (a k + 1)) :=
sorry

end exists_increasing_infinite_sequence_of_perfect_squares_divisible_by_13_power_l169_169000


namespace largest_three_digit_multiple_of_six_with_sum_fifteen_l169_169263

theorem largest_three_digit_multiple_of_six_with_sum_fifteen : 
  ∃ (n : ℕ), (100 ≤ n ∧ n < 1000) ∧ (n % 6 = 0) ∧ (Nat.digits 10 n).sum = 15 ∧ 
  ∀ (m : ℕ), (100 ≤ m ∧ m < 1000) ∧ (m % 6 = 0) ∧ (Nat.digits 10 m).sum = 15 → m ≤ n :=
  sorry

end largest_three_digit_multiple_of_six_with_sum_fifteen_l169_169263


namespace total_property_value_l169_169720

-- Define the given conditions
def price_per_sq_ft_condo := 98
def price_per_sq_ft_barn := 84
def price_per_sq_ft_detached := 102
def price_per_sq_ft_garage := 60
def sq_ft_condo := 2400
def sq_ft_barn := 1200
def sq_ft_detached := 3500
def sq_ft_garage := 480

-- Main statement to prove the total value of the property
theorem total_property_value :
  (price_per_sq_ft_condo * sq_ft_condo + 
   price_per_sq_ft_barn * sq_ft_barn + 
   price_per_sq_ft_detached * sq_ft_detached + 
   price_per_sq_ft_garage * sq_ft_garage = 721800) :=
by
  -- Placeholder for the actual proof
  sorry

end total_property_value_l169_169720


namespace total_dots_l169_169630

def ladybugs_monday : ℕ := 8
def ladybugs_tuesday : ℕ := 5
def dots_per_ladybug : ℕ := 6

theorem total_dots :
  (ladybugs_monday + ladybugs_tuesday) * dots_per_ladybug = 78 :=
by
  sorry

end total_dots_l169_169630


namespace sum_ninth_power_l169_169978

theorem sum_ninth_power (a b : ℝ) (h1 : a + b = 1) (h2 : a^2 + b^2 = 3) 
                        (h3 : a^3 + b^3 = 4) (h4 : a^4 + b^4 = 7)
                        (h5 : a^5 + b^5 = 11)
                        (h_ind : ∀ n, n ≥ 3 → a^n + b^n = a^(n-1) + b^(n-1) + a^(n-2) + b^(n-2)) :
  a^9 + b^9 = 76 :=
by
  sorry

end sum_ninth_power_l169_169978
