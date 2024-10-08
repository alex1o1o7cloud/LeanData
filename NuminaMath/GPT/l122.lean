import Mathlib

namespace necessary_but_not_sufficient_condition_l122_122315

variable {x : ℝ}

theorem necessary_but_not_sufficient_condition 
    (h : -1 ≤ x ∧ x < 2) : 
    (-1 ≤ x ∧ x < 3) ∧ ¬(((-1 ≤ x ∧ x < 3) → (-1 ≤ x ∧ x < 2))) :=
by
  sorry

end necessary_but_not_sufficient_condition_l122_122315


namespace volleyball_club_girls_l122_122168

theorem volleyball_club_girls (B G : ℕ) (h1 : B + G = 32) (h2 : (1 / 3 : ℝ) * G + ↑B = 20) : G = 18 := 
by
  sorry

end volleyball_club_girls_l122_122168


namespace fraction_undefined_l122_122348

theorem fraction_undefined (x : ℝ) : (x + 1 = 0) ↔ (x = -1) := 
  sorry

end fraction_undefined_l122_122348


namespace apples_not_sold_correct_l122_122895

-- Define the constants and conditions
def boxes_ordered_per_week : ℕ := 10
def apples_per_box : ℕ := 300
def fraction_sold : ℚ := 3 / 4

-- Define the total number of apples ordered in a week
def total_apples_ordered : ℕ := boxes_ordered_per_week * apples_per_box

-- Define the total number of apples sold in a week
def apples_sold : ℚ := fraction_sold * total_apples_ordered

-- Define the total number of apples not sold in a week
def apples_not_sold : ℚ := total_apples_ordered - apples_sold

-- Lean statement to prove the total number of apples not sold is 750
theorem apples_not_sold_correct :
  apples_not_sold = 750 := 
sorry

end apples_not_sold_correct_l122_122895


namespace selling_price_correct_l122_122201

namespace Shopkeeper

def costPrice : ℝ := 1500
def profitPercentage : ℝ := 20
def expectedSellingPrice : ℝ := 1800

theorem selling_price_correct
  (cp : ℝ := costPrice)
  (pp : ℝ := profitPercentage) :
  cp * (1 + pp / 100) = expectedSellingPrice :=
by
  sorry

end Shopkeeper

end selling_price_correct_l122_122201


namespace infinite_sum_eq_3_over_8_l122_122934

theorem infinite_sum_eq_3_over_8 :
  ∑' n : ℕ, (n : ℝ) / (n^4 + 4) = 3 / 8 :=
sorry

end infinite_sum_eq_3_over_8_l122_122934


namespace halfway_point_l122_122622

theorem halfway_point (x1 x2 : ℚ) (h1 : x1 = 1 / 6) (h2 : x2 = 5 / 6) : 
  (x1 + x2) / 2 = 1 / 2 :=
by
  sorry

end halfway_point_l122_122622


namespace sufficient_but_not_necessary_condition_for_q_l122_122875

def proposition_p (a : ℝ) := (1 / a) > (1 / 4)
def proposition_q (a : ℝ) := ∀ x : ℝ, (a * x^2 + a * x + 1) > 0

theorem sufficient_but_not_necessary_condition_for_q (a : ℝ) :
  proposition_p a → proposition_q a → (∃ a : ℝ, 0 < a ∧ a < 4) ∧ (∃ a : ℝ, 0 < a ∧ a < 4 ∧ ¬ proposition_p a) 
  := sorry

end sufficient_but_not_necessary_condition_for_q_l122_122875


namespace shift_parabola_two_units_right_l122_122731

-- Define the original parabola
def original_parabola (x : ℝ) : ℝ := x^2

-- Define the shift function
def shift (f : ℝ → ℝ) (h : ℝ) (x : ℝ) : ℝ := f (x - h)

-- Define the new parabola equation after shifting 2 units to the right
def shifted_parabola (x : ℝ) : ℝ := (x - 2)^2

-- The theorem stating that shifting the original parabola 2 units to the right equals the new parabola equation
theorem shift_parabola_two_units_right :
  ∀ x : ℝ, shift original_parabola 2 x = shifted_parabola x :=
by
  intros
  sorry

end shift_parabola_two_units_right_l122_122731


namespace rice_containers_l122_122996

theorem rice_containers (total_weight_pounds : ℚ) (weight_per_container_ounces : ℚ) (pound_to_ounces : ℚ) : 
  total_weight_pounds = 29/4 → 
  weight_per_container_ounces = 29 → 
  pound_to_ounces = 16 → 
  (total_weight_pounds * pound_to_ounces) / weight_per_container_ounces = 4 := 
by
  intros h1 h2 h3 
  rw [h1, h2, h3]
  sorry

end rice_containers_l122_122996


namespace other_x_intercept_of_parabola_l122_122828

theorem other_x_intercept_of_parabola (a b c : ℝ) :
  (∃ x : ℝ, y = a * x ^ 2 + b * x + c) ∧ (2, 10) ∈ {p | ∃ x : ℝ, p = (x, a * x ^ 2 + b * x + c)} ∧ (1, 0) ∈ {p | ∃ x : ℝ, p = (x, a * x ^ 2 + b * x + c)}
  → ∃ x : ℝ, x = 3 ∧ (x, 0) ∈ {p | ∃ x : ℝ, p = (x, a * x ^ 2 + b * x + c)} :=
by
  sorry

end other_x_intercept_of_parabola_l122_122828


namespace compare_neg_fractions_l122_122036

theorem compare_neg_fractions : - (1 : ℝ) / 3 < - (1 : ℝ) / 4 :=
  sorry

end compare_neg_fractions_l122_122036


namespace avg_diff_l122_122836

theorem avg_diff (n : ℕ) (m : ℝ) (mistake : ℝ) (true_value : ℝ)
   (h_n : n = 30) (h_mistake : mistake = 15) (h_true_value : true_value = 105) 
   (h_m : m = true_value - mistake) : 
   (m / n) = 3 := 
by
  sorry

end avg_diff_l122_122836


namespace range_of_k_l122_122570

noncomputable def f (x : ℝ) (k : ℝ) : ℝ := x^(-k^2 + k + 2)

theorem range_of_k (k : ℝ) : (∃ k, (f 2 k < f 3 k)) ↔ (-1 < k) ∧ (k < 2) :=
by
  sorry

end range_of_k_l122_122570


namespace probability_of_odd_numbers_l122_122111

def binomial (n k : ℕ) : ℕ :=
  Nat.choose n k

def probability_of_5_odd_numbers_in_7_rolls : ℚ :=
  (binomial 7 5) * (1 / 2)^5 * (1 / 2)^(7-5)

theorem probability_of_odd_numbers :
  probability_of_5_odd_numbers_in_7_rolls = 21 / 128 :=
by
  sorry

end probability_of_odd_numbers_l122_122111


namespace num_bicycles_eq_20_l122_122937

-- Definitions based on conditions
def num_cars : ℕ := 10
def num_motorcycles : ℕ := 5
def total_wheels : ℕ := 90
def wheels_per_bicycle : ℕ := 2
def wheels_per_car : ℕ := 4
def wheels_per_motorcycle : ℕ := 2

-- Statement to prove
theorem num_bicycles_eq_20 (B : ℕ) 
  (h_wheels_from_bicycles : wheels_per_bicycle * B = 2 * B)
  (h_wheels_from_cars : num_cars * wheels_per_car = 40)
  (h_wheels_from_motorcycles : num_motorcycles * wheels_per_motorcycle = 10)
  (h_total_wheels : wheels_per_bicycle * B + 40 + 10 = total_wheels) :
  B = 20 :=
sorry

end num_bicycles_eq_20_l122_122937


namespace xiao_hua_seat_correct_l122_122332

-- Define the classroom setup
def classroom : Type := ℤ × ℤ

-- Define the total number of rows and columns in the classroom.
def total_rows : ℤ := 7
def total_columns : ℤ := 8

-- Define the position of Xiao Ming's seat.
def xiao_ming_seat : classroom := (3, 7)

-- Define the position of Xiao Hua's seat.
def xiao_hua_seat : classroom := (5, 2)

-- Prove that Xiao Hua's seat is designated as (5, 2)
theorem xiao_hua_seat_correct : xiao_hua_seat = (5, 2) := by
  -- The proof would go here
  sorry

end xiao_hua_seat_correct_l122_122332


namespace probability_of_sum_leq_10_l122_122414

open Nat

-- Define the three dice roll outcomes
def dice_outcomes := {n : ℕ | 1 ≤ n ∧ n ≤ 6}

-- Define the total number of outcomes when rolling three dice
def total_outcomes : ℕ := 6 ^ 3

-- Count the number of valid outcomes where the sum of three dice is less than or equal to 10
def count_valid_outcomes : ℕ := 75  -- This is determined through combinatorial calculations or software

-- Define the desired probability
def desired_probability := (count_valid_outcomes : ℚ) / total_outcomes

-- Prove that the desired probability equals 25/72
theorem probability_of_sum_leq_10 :
  desired_probability = 25 / 72 :=
by sorry

end probability_of_sum_leq_10_l122_122414


namespace number_of_pairs_l122_122306

theorem number_of_pairs (n : ℕ) (h : n ≥ 3) : 
  ∃ a : ℕ, a = (n-2) * 2^(n-1) + 1 :=
by
  sorry

end number_of_pairs_l122_122306


namespace chess_tournament_points_l122_122456

theorem chess_tournament_points (boys girls : ℕ) (total_points : ℝ) 
  (total_matches : ℕ)
  (matches_among_boys points_among_boys : ℕ)
  (matches_among_girls points_among_girls : ℕ)
  (matches_between points_between : ℕ)
  (total_players : ℕ := boys + girls)
  (H1 : boys = 9) (H2 : girls = 3) (H3 : total_players = 12)
  (H4 : total_matches = total_players * (total_players - 1) / 2) 
  (H5 : total_points = total_matches) 
  (H6 : matches_among_boys = boys * (boys - 1) / 2) 
  (H7 : points_among_boys = matches_among_boys)
  (H8 : matches_among_girls = girls * (girls - 1) / 2) 
  (H9 : points_among_girls = matches_among_girls) 
  (H10 : matches_between = boys * girls) 
  (H11 : points_between = matches_between) :
  ¬ ∃ (P_B P_G : ℝ) (x : ℝ),
    P_B = points_among_boys + x ∧
    P_G = points_among_girls + (points_between - x) ∧
    P_B = P_G := by
  sorry

end chess_tournament_points_l122_122456


namespace abs_diff_31st_terms_l122_122439

/-- Sequence C is an arithmetic sequence with a starting term 100 and a common difference 15. --/
def seqC (n : ℕ) : ℤ :=
  100 + 15 * (n - 1)

/-- Sequence D is an arithmetic sequence with a starting term 100 and a common difference -20. --/
def seqD (n : ℕ) : ℤ :=
  100 - 20 * (n - 1)

/-- Absolute value of the difference between the 31st terms of sequences C and D is 1050. --/
theorem abs_diff_31st_terms : |seqC 31 - seqD 31| = 1050 := by
  sorry

end abs_diff_31st_terms_l122_122439


namespace relationship_between_x_plus_one_and_ex_l122_122404

theorem relationship_between_x_plus_one_and_ex (x : ℝ) : x + 1 ≤ Real.exp x :=
sorry

end relationship_between_x_plus_one_and_ex_l122_122404


namespace value_of_fraction_l122_122596

theorem value_of_fraction (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0)
  (h : (b / c = 2005) ∧ (c / b = 2005)) : (b + c) / (a + b) = 2005 :=
by
  sorry

end value_of_fraction_l122_122596


namespace initial_fee_l122_122443

theorem initial_fee (initial_fee : ℝ) : 
  (∀ (distance_charge_per_segment travel_total_charge : ℝ), 
    distance_charge_per_segment = 0.35 → 
    3.6 / 0.4 * distance_charge_per_segment + initial_fee = travel_total_charge → 
    travel_total_charge = 5.20)
    → initial_fee = 2.05 :=
by
  intro h
  specialize h 0.35 5.20
  sorry

end initial_fee_l122_122443


namespace find_m_when_circle_tangent_to_line_l122_122589

theorem find_m_when_circle_tangent_to_line 
    (m : ℝ)
    (circle_eq : (x y : ℝ) → (x - 1)^2 + (y - 1)^2 = 4 * m)
    (line_eq : (x y : ℝ) → x + y = 2 * m) :
    (m = 2 + Real.sqrt 3) ∨ (m = 2 - Real.sqrt 3) :=
sorry

end find_m_when_circle_tangent_to_line_l122_122589


namespace percentage_increase_twice_l122_122004

theorem percentage_increase_twice {P : ℝ} (x : ℝ) :
  (P * (1 + x)^2) = (P * (1 + 0.6900000000000001)) →
  x = 0.30 :=
by
  sorry

end percentage_increase_twice_l122_122004


namespace matthew_younger_than_freddy_l122_122220

variables (M R F : ℕ)

-- Define the conditions
def sum_of_ages : Prop := M + R + F = 35
def matthew_older_than_rebecca : Prop := M = R + 2
def freddy_age : Prop := F = 15

-- Prove the statement "Matthew is 4 years younger than Freddy."
theorem matthew_younger_than_freddy (h1 : sum_of_ages M R F) (h2 : matthew_older_than_rebecca M R) (h3 : freddy_age F) :
    F - M = 4 := by
  sorry

end matthew_younger_than_freddy_l122_122220


namespace race_time_l122_122314

theorem race_time (t_A t_B : ℝ) (v_A v_B : ℝ)
  (h1 : t_B = t_A + 7)
  (h2 : v_A * t_A = 80)
  (h3 : v_B * t_B = 80)
  (h4 : v_A * (t_A + 7) = 136) :
  t_A = 10 :=
by
  sorry

end race_time_l122_122314


namespace rectangle_area_l122_122270

-- Define the given dimensions
def length : ℝ := 1.5
def width : ℝ := 0.75
def expected_area : ℝ := 1.125

-- State the problem
theorem rectangle_area (l w : ℝ) (h_l : l = length) (h_w : w = width) : l * w = expected_area :=
by sorry

end rectangle_area_l122_122270


namespace cuboid_length_l122_122702

theorem cuboid_length (A b h : ℝ) (A_eq : A = 2400) (b_eq : b = 10) (h_eq : h = 16) :
    ∃ l : ℝ, 2 * (l * b + b * h + h * l) = A ∧ l = 40 := by
  sorry

end cuboid_length_l122_122702


namespace ball_distribution_l122_122538

theorem ball_distribution (basketballs volleyballs classes balls : ℕ) 
  (h1 : basketballs = 2) 
  (h2 : volleyballs = 3) 
  (h3 : classes = 4) 
  (h4 : balls = 4) :
  (classes.choose 3) + (classes.choose 2) = 10 :=
by
  sorry

end ball_distribution_l122_122538


namespace mia_days_not_worked_l122_122561

theorem mia_days_not_worked :
  ∃ (y : ℤ), (∃ (x : ℤ), 
  x + y = 30 ∧ 80 * x - 40 * y = 1600) ∧ y = 20 :=
by
  sorry

end mia_days_not_worked_l122_122561


namespace statue_selling_price_l122_122231

/-- Problem conditions -/
def original_cost : ℤ := 550
def profit_percentage : ℝ := 0.20

/-- Proof problem statement -/
theorem statue_selling_price : original_cost + profit_percentage * original_cost = 660 := by
  sorry

end statue_selling_price_l122_122231


namespace quadratic_equation_completes_to_square_l122_122626

theorem quadratic_equation_completes_to_square :
  ∀ x : ℝ, x^2 + 4 * x + 2 = 0 → (x + 2)^2 = 2 :=
by
  intro x
  intro h
  sorry

end quadratic_equation_completes_to_square_l122_122626


namespace intersection_sum_zero_l122_122206

-- Definitions from conditions:
def lineA (x : ℝ) : ℝ := -x
def lineB (x : ℝ) : ℝ := 5 * x - 10

-- Declaration of the theorem:
theorem intersection_sum_zero : ∃ a b : ℝ, lineA a = b ∧ lineB a = b ∧ a + b = 0 := sorry

end intersection_sum_zero_l122_122206


namespace complement_union_example_l122_122966

open Set

variable (I : Set ℕ) (A : Set ℕ) (B : Set ℕ)

noncomputable def complement (U : Set ℕ) (S : Set ℕ) : Set ℕ := {x ∈ U | x ∉ S}

theorem complement_union_example
    (hI : I = {0, 1, 2, 3, 4})
    (hA : A = {0, 1, 2, 3})
    (hB : B = {2, 3, 4}) :
    (complement I A) ∪ (complement I B) = {0, 1, 4} := by
  sorry

end complement_union_example_l122_122966


namespace max_k_value_l122_122481

open Real

theorem max_k_value (x y k : ℝ)
  (h_pos_x : 0 < x)
  (h_pos_y : 0 < y)
  (h_pos_k : 0 < k)
  (h_eq : 6 = k^2 * (x^2 / y^2 + y^2 / x^2) + k * (x / y + y / x)) :
  k ≤ 3 / 2 :=
by
  sorry

end max_k_value_l122_122481


namespace warehouseGoodsDecreased_initialTonnage_totalLoadingFees_l122_122318

noncomputable def netChange (tonnages : List Int) : Int :=
  List.sum tonnages

noncomputable def initialGoods (finalGoods : Int) (change : Int) : Int :=
  finalGoods + change

noncomputable def totalFees (tonnages : List Int) (feePerTon : Int) : Int :=
  feePerTon * List.sum (tonnages.map (Int.natAbs))

theorem warehouseGoodsDecreased 
  (tonnages : List Int) (finalGoods : Int) (feePerTon : Int)
  (h1 : tonnages = [21, -32, -16, 35, -38, -20]) 
  (h2 : finalGoods = 580)
  (h3 : feePerTon = 4) : 
  netChange tonnages < 0 := by
  sorry

theorem initialTonnage 
  (tonnages : List Int) (finalGoods : Int) (change : Int)
  (h1 : tonnages = [21, -32, -16, 35, -38, -20])
  (h2 : finalGoods = 580)
  (h3 : change = netChange tonnages) : 
  initialGoods finalGoods change = 630 := by
  sorry

theorem totalLoadingFees 
  (tonnages : List Int) (feePerTon : Int)
  (h1 : tonnages = [21, -32, -16, 35, -38, -20])
  (h2 : feePerTon = 4) : 
  totalFees tonnages feePerTon = 648 := by
  sorry

end warehouseGoodsDecreased_initialTonnage_totalLoadingFees_l122_122318


namespace number_of_people_l122_122844

variable (P M : ℕ)

-- Conditions
def cond1 : Prop := (500 = P * M)
def cond2 : Prop := (500 = (P + 5) * (M - 2))

-- Goal
theorem number_of_people (h1 : cond1 P M) (h2 : cond2 P M) : P = 33 :=
sorry

end number_of_people_l122_122844


namespace area_EPHQ_l122_122219

theorem area_EPHQ {EFGH : Type} 
  (rectangle_EFGH : EFGH) 
  (length_EF : Real) (width_EG : Real) 
  (P_point : Real) (Q_point : Real) 
  (area_EFGH : Real) 
  (area_EFP : Real) 
  (area_EHQ : Real) : 
  length_EF = 12 → width_EG = 6 → P_point = 4 → Q_point = 3 → 
  area_EFGH = length_EF * width_EG →
  area_EFP = (1 / 2) * width_EG * P_point →
  area_EHQ = (1 / 2) * length_EF * Q_point → 
  (area_EFGH - area_EFP - area_EHQ) = 42 := 
by 
  intros h1 h2 h3 h4 h5 h6 h7 
  sorry

end area_EPHQ_l122_122219


namespace arrangement_of_chairs_and_stools_l122_122746

theorem arrangement_of_chairs_and_stools :
  (Nat.choose 10 3) = 120 :=
by
  -- Proof goes here
  sorry

end arrangement_of_chairs_and_stools_l122_122746


namespace hyperbola_asymptotes_l122_122963

def hyperbola (x y : ℝ) : Prop := (x^2 / 8) - (y^2 / 2) = 1

theorem hyperbola_asymptotes (x y : ℝ) :
  hyperbola x y → (y = (1/2) * x ∨ y = - (1/2) * x) :=
by
  sorry

end hyperbola_asymptotes_l122_122963


namespace power_inequality_l122_122189

theorem power_inequality (a b c d : ℝ) (ha : 0 < a) (hab : a ≤ b) (hbc : b ≤ c) (hcd : c ≤ d) :
  a^b * b^c * c^d * d^a ≥ b^a * c^b * d^c * a^d := 
sorry

end power_inequality_l122_122189


namespace multiplication_identity_l122_122395

theorem multiplication_identity : 5 ^ 29 * 4 ^ 15 = 2 * 10 ^ 29 := by
  sorry

end multiplication_identity_l122_122395


namespace card_selection_ways_l122_122388

theorem card_selection_ways (deck_size : ℕ) (suits : ℕ) (cards_per_suit : ℕ) (total_cards_chosen : ℕ)
  (repeated_suit_count : ℕ) (distinct_suits_count : ℕ) (distinct_ranks_count : ℕ) 
  (correct_answer : ℕ) :
  deck_size = 52 ∧ suits = 4 ∧ cards_per_suit = 13 ∧ total_cards_chosen = 5 ∧ 
  repeated_suit_count = 2 ∧ distinct_suits_count = 3 ∧ distinct_ranks_count = 11 ∧ 
  correct_answer = 414384 :=
by 
  -- Sorry is used to skip actual proof steps, according to the instructions.
  sorry

end card_selection_ways_l122_122388


namespace mean_of_set_l122_122268

theorem mean_of_set (x y : ℝ) 
  (h : (28 + x + 50 + 78 + 104) / 5 = 62) : 
  (48 + 62 + 98 + y + x) / 5 = (258 + y) / 5 :=
by
  -- we would now proceed to prove this according to lean's proof tactics.
  sorry

end mean_of_set_l122_122268


namespace range_of_a_ineq_l122_122514

noncomputable def range_of_a (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ < 1 ∧ 1 < x₂ ∧ x₁ * x₁ + (a * a - 1) * x₁ + (a - 2) = 0 ∧
                x₂ * x₂ + (a * a - 1) * x₂ + (a - 2) = 0

theorem range_of_a_ineq (a : ℝ) : (∃ x₁ x₂ : ℝ, x₁ < 1 ∧ 1 < x₂ ∧
    x₁^2 + (a^2 - 1) * x₁ + (a - 2) = 0 ∧
    x₂^2 + (a^2 - 1) * x₂ + (a - 2) = 0) → -2 < a ∧ a < 1 :=
sorry

end range_of_a_ineq_l122_122514


namespace population_of_village_l122_122736

-- Define the given condition
def total_population (P : ℝ) : Prop :=
  0.4 * P = 23040

-- The theorem to prove that the total population is 57600
theorem population_of_village : ∃ P : ℝ, total_population P ∧ P = 57600 :=
by
  sorry

end population_of_village_l122_122736


namespace complement_of_union_is_neg3_l122_122752

open Set

variable (U A B : Set Int)

def complement_union (U A B : Set Int) : Set Int :=
  U \ (A ∪ B)

theorem complement_of_union_is_neg3 (U A B : Set Int) (hU : U = {-3, -2, -1, 0, 1, 2, 3, 4, 5, 6})
  (hA : A = {-1, 0, 1, 2, 3}) (hB : B = {-2, 3, 4, 5, 6}) :
  complement_union U A B = {-3} :=
by
  sorry

end complement_of_union_is_neg3_l122_122752


namespace cody_tickets_l122_122309

theorem cody_tickets (initial_tickets : ℕ) (spent_tickets : ℕ) (won_tickets : ℕ) : 
  initial_tickets = 49 ∧ spent_tickets = 25 ∧ won_tickets = 6 → 
  initial_tickets - spent_tickets + won_tickets = 30 :=
by sorry

end cody_tickets_l122_122309


namespace fraction_simplifies_to_two_l122_122952

theorem fraction_simplifies_to_two :
  (2 + 4 + 6 + 8 + 10 + 12 + 14 + 16 + 18 + 20) / (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10) = 2 := by
  sorry

end fraction_simplifies_to_two_l122_122952


namespace num_users_in_china_in_2022_l122_122819

def num_users_scientific (n : ℝ) : Prop :=
  n = 1.067 * 10^9

theorem num_users_in_china_in_2022 :
  num_users_scientific 1.067e9 :=
by
  sorry

end num_users_in_china_in_2022_l122_122819


namespace sum_of_repeating_decimals_l122_122214

-- Defining the given repeating decimals as fractions
def rep_decimal1 : ℚ := 2 / 9
def rep_decimal2 : ℚ := 2 / 99
def rep_decimal3 : ℚ := 2 / 9999

-- Stating the theorem to prove the given sum equals the correct answer
theorem sum_of_repeating_decimals :
  rep_decimal1 + rep_decimal2 + rep_decimal3 = 224422 / 9999 :=
by
  sorry

end sum_of_repeating_decimals_l122_122214


namespace fiveLetterWordsWithAtLeastOneVowel_l122_122881

-- Definitions for the given conditions
def letters : List Char := ['A', 'B', 'C', 'D', 'E', 'F']
def vowels : List Char := ['A', 'E']
def consonants : List Char := ['B', 'C', 'D', 'F']

-- Total number of 5-letter words with no restrictions
def totalWords := 6^5

-- Total number of 5-letter words containing no vowels
def noVowelWords := 4^5

-- Prove that the number of 5-letter words with at least one vowel is 6752
theorem fiveLetterWordsWithAtLeastOneVowel : (totalWords - noVowelWords) = 6752 := by
  sorry

end fiveLetterWordsWithAtLeastOneVowel_l122_122881


namespace solve_first_equation_solve_second_equation_l122_122032

-- Statement for the first equation
theorem solve_first_equation : ∀ x : ℝ, x^2 - 3*x - 4 = 0 ↔ x = 4 ∨ x = -1 := by
  sorry

-- Statement for the second equation
theorem solve_second_equation : ∀ x : ℝ, x * (x - 2) = 1 ↔ x = 1 + Real.sqrt 2 ∨ x = 1 - Real.sqrt 2 := by
  sorry

end solve_first_equation_solve_second_equation_l122_122032


namespace calculation_of_expression_l122_122740

theorem calculation_of_expression
  (w x y z : ℕ)
  (h : 2^w * 3^x * 5^y * 7^z = 13230) :
  3 * w + 2 * x + 6 * y + 4 * z = 23 :=
sorry

end calculation_of_expression_l122_122740


namespace min_value_y_l122_122045

theorem min_value_y (x : ℝ) (hx : x > 3) : 
  ∃ y, (∀ x > 3, y = min_value) ∧ min_value = 5 :=
by 
  sorry

end min_value_y_l122_122045


namespace philip_school_trip_days_l122_122138

-- Define the distances for the trips
def school_trip_one_way_miles : ℝ := 2.5
def market_trip_one_way_miles : ℝ := 2

-- Define the number of times he makes the trips in a day and in a week
def school_round_trips_per_day : ℕ := 2
def market_round_trips_per_week : ℕ := 1

-- Define the total mileage in a week
def weekly_mileage : ℕ := 44

-- Define the equation based on the given conditions
def weekly_school_trip_distance (d : ℕ) : ℝ :=
  (school_trip_one_way_miles * 2 * school_round_trips_per_day) * d

def weekly_market_trip_distance : ℝ :=
  (market_trip_one_way_miles * 2) * market_round_trips_per_week

-- Define the main theorem to be proved
theorem philip_school_trip_days :
  ∃ d : ℕ, weekly_school_trip_distance d + weekly_market_trip_distance = weekly_mileage ∧ d = 4 :=
by
  sorry

end philip_school_trip_days_l122_122138


namespace inner_rectangle_length_l122_122358

def inner_rect_width : ℕ := 2

def second_rect_area (x : ℕ) : ℕ := 6 * (x + 4)

def largest_rect_area (x : ℕ) : ℕ := 10 * (x + 8)

def shaded_area_1 (x : ℕ) : ℕ := second_rect_area x - 2 * x

def shaded_area_2 (x : ℕ) : ℕ := largest_rect_area x - second_rect_area x

def in_arithmetic_progression (a b c : ℕ) : Prop := b - a = c - b

theorem inner_rectangle_length (x : ℕ) :
  in_arithmetic_progression (2 * x) (shaded_area_1 x) (shaded_area_2 x) → x = 4 := by
  intros
  sorry

end inner_rectangle_length_l122_122358


namespace peter_money_left_l122_122572

variable (soda_cost : ℝ) (money_brought : ℝ) (soda_ounces : ℝ)

theorem peter_money_left (h1 : soda_cost = 0.25) (h2 : money_brought = 2) (h3 : soda_ounces = 6) : 
    money_brought - soda_ounces * soda_cost = 0.50 := 
by 
  sorry

end peter_money_left_l122_122572


namespace mad_hatter_must_secure_at_least_70_percent_l122_122405

theorem mad_hatter_must_secure_at_least_70_percent :
  ∀ (N : ℕ) (uM uH uD : ℝ) (α : ℝ),
    uM = 0.2 ∧ uH = 0.25 ∧ uD = 0.3 → 
    uM + α * 0.25 ≥ 0.25 + (1 - α) * 0.25 ∧
    uM + α * 0.25 ≥ 0.3 + (1 - α) * 0.25 →
    α ≥ 0.7 :=
by
  intros N uM uH uD α h hx
  sorry 

end mad_hatter_must_secure_at_least_70_percent_l122_122405


namespace weight_of_dry_grapes_l122_122532

def fresh_grapes : ℝ := 10 -- weight of fresh grapes in kg
def fresh_water_content : ℝ := 0.90 -- fresh grapes contain 90% water by weight
def dried_water_content : ℝ := 0.20 -- dried grapes contain 20% water by weight

theorem weight_of_dry_grapes : 
  (fresh_grapes * (1 - fresh_water_content)) / (1 - dried_water_content) = 1.25 := 
by 
  sorry

end weight_of_dry_grapes_l122_122532


namespace arithmetic_sequence_sum_l122_122156

theorem arithmetic_sequence_sum (a : ℕ → ℤ) (S : ℕ → ℤ) (h_seq : ∀ n, a (n + 1) - a n = a 2 - a 1)
  (h_a3 : a 3 = 5) (h_a5 : a 5 = 9) :
  S 7 = 49 :=
sorry

end arithmetic_sequence_sum_l122_122156


namespace factorize1_factorize2_l122_122936

-- Part 1: Prove the factorization of xy - 1 - x + y
theorem factorize1 (x y : ℝ) : (x * y - 1 - x + y) = (y - 1) * (x + 1) :=
  sorry

-- Part 2: Prove the factorization of (a^2 + b^2)^2 - 4a^2b^2
theorem factorize2 (a b : ℝ) : (a^2 + b^2)^2 - 4 * a^2 * b^2 = (a + b)^2 * (a - b)^2 :=
  sorry

end factorize1_factorize2_l122_122936


namespace find_a7_l122_122100

section GeometricSequence

variables {a : ℕ → ℝ} (q : ℝ) (a1 : ℝ)

-- a_n is defined as a geometric sequence
def geometric_sequence (a : ℕ → ℝ) (a1 q : ℝ) : Prop :=
∀ n, a n = a1 * q^(n-1)

-- Given conditions
axiom h1 : geometric_sequence a a1 q
axiom h2 : a 2 * a 4 * a 5 = a 3 * a 6
axiom h3 : a 9 * a 10 = -8

--The proof goal
theorem find_a7 : a 7 = -2 :=
sorry

end GeometricSequence

end find_a7_l122_122100


namespace purely_imaginary_complex_l122_122305

theorem purely_imaginary_complex :
  ∀ (x y : ℤ), (x - 4) ≠ 0 → (y^2 - 3*y - 4) ≠ 0 → (∃ (z : ℂ), z = ⟨0, x^2 + 3*x - 4⟩) → 
    (x = 4 ∧ y ≠ 4 ∧ y ≠ -1) :=
by
  intro x y hx hy hz
  sorry

end purely_imaginary_complex_l122_122305


namespace diameter_is_10sqrt6_l122_122483

noncomputable def radius (A : ℝ) (hA : A = 150 * Real.pi) : ℝ :=
  Real.sqrt (A / Real.pi)

noncomputable def diameter (A : ℝ) (hA : A = 150 * Real.pi) : ℝ :=
  2 * radius A hA

theorem diameter_is_10sqrt6 (A : ℝ) (hA : A = 150 * Real.pi) :
  diameter A hA = 10 * Real.sqrt 6 :=
  sorry

end diameter_is_10sqrt6_l122_122483


namespace option_C_correct_l122_122460

variables {Line Plane : Type}
variables (m n : Line) (α β : Plane)

-- Definitions for parallel and perpendicular relationships
def parallel (l : Line) (p : Plane) : Prop := sorry
def perpendicular (l : Line) (p : Plane) : Prop := sorry
def line_parallel (l₁ l₂ : Line) : Prop := sorry

-- Theorem statement based on problem c) translation
theorem option_C_correct (H1 : line_parallel m n) (H2 : perpendicular m α) : perpendicular n α :=
sorry

end option_C_correct_l122_122460


namespace sufficient_but_not_necessary_condition_l122_122408

theorem sufficient_but_not_necessary_condition (a b : ℝ) (h₀ : b > a) (h₁ : a > 0) :
  (1 / (a ^ 2) > 1 / (b ^ 2)) :=
by
  sorry

end sufficient_but_not_necessary_condition_l122_122408


namespace robin_extra_drinks_l122_122151

-- Conditions
def initial_sodas : ℕ := 22
def initial_energy_drinks : ℕ := 15
def initial_smoothies : ℕ := 12
def drank_sodas : ℕ := 6
def drank_energy_drinks : ℕ := 9
def drank_smoothies : ℕ := 2

-- Total drinks bought
def total_drinks_bought : ℕ :=
  initial_sodas + initial_energy_drinks + initial_smoothies
  
-- Total drinks consumed
def total_drinks_consumed : ℕ :=
  drank_sodas + drank_energy_drinks + drank_smoothies

-- Number of extra drinks
def extra_drinks : ℕ :=
  total_drinks_bought - total_drinks_consumed

-- Theorem to prove
theorem robin_extra_drinks : extra_drinks = 32 :=
  by
  -- skipping the proof
  sorry

end robin_extra_drinks_l122_122151


namespace cos_thm_l122_122748

variable (θ : ℝ)

-- Conditions
def condition1 : Prop := 3 * Real.sin (2 * θ) = 4 * Real.tan θ
def condition2 : Prop := ∀ k : ℤ, θ ≠ k * Real.pi

-- Prove that cos 2θ = 1/3 given the conditions
theorem cos_thm (h1 : condition1 θ) (h2 : condition2 θ) : Real.cos (2 * θ) = 1 / 3 :=
by
  sorry

end cos_thm_l122_122748


namespace binom_30_3_eq_4060_l122_122515

theorem binom_30_3_eq_4060 : Nat.choose 30 3 = 4060 := by
  sorry

end binom_30_3_eq_4060_l122_122515


namespace find_magnitude_a_l122_122491

noncomputable def vector_a (m : ℝ) : ℝ × ℝ := (1, 2 * m)
def vector_b (m : ℝ) : ℝ × ℝ := (m + 1, 1)
def vector_c (m : ℝ) : ℝ × ℝ := (2, m)
def vector_add (v1 v2 : ℝ × ℝ) : ℝ × ℝ := (v1.1 + v2.1, v1.2 + v2.2)
def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2
noncomputable def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

theorem find_magnitude_a (m : ℝ) (h : dot_product (vector_add (vector_a m) (vector_c m)) (vector_b m) = 0) :
  magnitude (vector_a (-1 / 2)) = Real.sqrt 2 :=
by
  sorry

end find_magnitude_a_l122_122491


namespace quadratic_unique_solution_l122_122286

theorem quadratic_unique_solution (k : ℝ) (x : ℝ) :
  (16 ^ 2 - 4 * 2 * k * 4 = 0) → (k = 8 ∧ x = -1 / 2) :=
by
  sorry

end quadratic_unique_solution_l122_122286


namespace casey_savings_l122_122803

-- Define the constants given in the problem conditions
def wage_employee_1 : ℝ := 20
def wage_employee_2 : ℝ := 22
def subsidy : ℝ := 6
def hours_per_week : ℝ := 40

-- Define the weekly cost of each employee
def weekly_cost_employee_1 := wage_employee_1 * hours_per_week
def weekly_cost_employee_2 := (wage_employee_2 - subsidy) * hours_per_week

-- Define the savings by hiring the cheaper employee
def savings := weekly_cost_employee_1 - weekly_cost_employee_2

-- Theorem stating the expected savings
theorem casey_savings : savings = 160 := by
  -- Proof is not included
  sorry

end casey_savings_l122_122803


namespace reading_time_difference_l122_122368

theorem reading_time_difference (xanthia_speed molly_speed book_length : ℕ)
  (hx : xanthia_speed = 120) (hm : molly_speed = 60) (hb : book_length = 300) :
  (book_length / molly_speed - book_length / xanthia_speed) * 60 = 150 :=
by
  -- We acknowledge the proof here would use the given values
  sorry

end reading_time_difference_l122_122368


namespace count_arithmetic_progressions_22_1000_l122_122948

def num_increasing_arithmetic_progressions (n k max_val : ℕ) : ℕ :=
  -- This is a stub for the arithmetic sequence counting function.
  sorry

theorem count_arithmetic_progressions_22_1000 :
  num_increasing_arithmetic_progressions 22 22 1000 = 23312 :=
sorry

end count_arithmetic_progressions_22_1000_l122_122948


namespace geometric_sequence_ratio_l122_122549

theorem geometric_sequence_ratio (a1 : ℕ) (S : ℕ → ℕ) (r : ℤ) (h1 : r = -2) (h2 : ∀ n, S n = a1 * (1 - r ^ n) / (1 - r)) :
  S 4 / S 2 = 5 :=
by
  -- Placeholder for proof steps
  sorry

end geometric_sequence_ratio_l122_122549


namespace find_a_of_extremum_l122_122071

theorem find_a_of_extremum (a b : ℝ) (f : ℝ → ℝ) (f' : ℝ → ℝ)
  (h1 : f x = x^3 + a*x^2 + b*x + a^2)
  (h2 : f' x = 3*x^2 + 2*a*x + b)
  (h3 : f' 1 = 0)
  (h4 : f 1 = 10) : a = 4 := by
  sorry

end find_a_of_extremum_l122_122071


namespace speed_of_train_l122_122782

open Real

-- Define the conditions as given in the problem
def length_of_bridge : ℝ := 650
def length_of_train : ℝ := 200
def time_to_pass_bridge : ℝ := 17

-- Define the problem statement which needs to be proved
theorem speed_of_train : (length_of_bridge + length_of_train) / time_to_pass_bridge = 50 :=
by
  sorry

end speed_of_train_l122_122782


namespace inequality_proof_l122_122475

theorem inequality_proof
  (x y z : ℝ)
  (hx : 0 < x)
  (hy : 0 < y)
  (hz : 0 < z)
  (hxyz : x * y * z = 1) :
  x^2 + y^2 + z^2 + x * y + y * z + z * x ≥ 2 * (Real.sqrt x + Real.sqrt y + Real.sqrt z) :=
by
  sorry

end inequality_proof_l122_122475


namespace min_value_geometric_sequence_l122_122072

-- Definitions based on conditions
noncomputable def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) := ∀ n : ℕ, a (n + 1) = a n * q

-- We need to state the problem using the above definitions
theorem min_value_geometric_sequence 
  (a : ℕ → ℝ) (q : ℝ) (s t : ℕ) 
  (h_seq : is_geometric_sequence a q) 
  (h_q : q ≠ 1) 
  (h_st : a s * a t = (a 5) ^ 2) 
  (h_s_pos : s > 0) 
  (h_t_pos : t > 0) 
  : 4 / s + 1 / (4 * t) = 5 / 8 := sorry

end min_value_geometric_sequence_l122_122072


namespace integers_square_less_than_three_times_l122_122316

theorem integers_square_less_than_three_times (x : ℤ) : x^2 < 3 * x ↔ x = 1 ∨ x = 2 :=
by
  sorry

end integers_square_less_than_three_times_l122_122316


namespace bus_driver_regular_rate_l122_122811

theorem bus_driver_regular_rate (hours := 60) (total_pay := 1200) (regular_hours := 40) (overtime_rate_factor := 1.75) :
  ∃ R : ℝ, 40 * R + 20 * (1.75 * R) = 1200 ∧ R = 16 := 
by
  sorry

end bus_driver_regular_rate_l122_122811


namespace correct_location_l122_122676

-- Define the possible options
inductive Location
| A : Location
| B : Location
| C : Location
| D : Location

-- Define the conditions
def option_A : Prop := ¬(∃ d, d ≠ "right")
def option_B : Prop := ¬(∃ d, d ≠ 900)
def option_C : Prop := ¬(∃ d, d ≠ "west")
def option_D : Prop := (∃ d₁ d₂, d₁ = "west" ∧ d₂ = 900)

-- The objective is to prove that option D is the correct description of the location
theorem correct_location : ∃ l, l = Location.D → 
  (option_A ∧ option_B ∧ option_C ∧ option_D) :=
by
  sorry

end correct_location_l122_122676


namespace example_calculation_l122_122517

theorem example_calculation (a : ℝ) : (a^2)^3 = a^6 :=
by 
  sorry

end example_calculation_l122_122517


namespace monkey_bananas_max_l122_122208

noncomputable def max_bananas_home : ℕ :=
  let total_bananas := 100
  let distance := 50
  let carry_capacity := 50
  let consumption_rate := 1
  let distance_each_way := distance / 2
  let bananas_eaten_each_way := distance_each_way * consumption_rate
  let bananas_left_midway := total_bananas / 2 - bananas_eaten_each_way
  let bananas_picked_midway := bananas_left_midway * 2
  let bananas_left_home := bananas_picked_midway - distance_each_way * consumption_rate
  bananas_left_home

theorem monkey_bananas_max : max_bananas_home = 25 :=
  sorry

end monkey_bananas_max_l122_122208


namespace simplify_expression_l122_122829

variable (y : ℝ)

theorem simplify_expression : (3 * y^4)^2 = 9 * y^8 :=
by 
  sorry

end simplify_expression_l122_122829


namespace exists_monomial_l122_122101

variables (x y : ℕ) -- Define x and y as natural numbers

theorem exists_monomial :
  ∃ (c : ℕ) (e_x e_y : ℕ), c = 3 ∧ e_x + e_y = 3 ∧ (c * x ^ e_x * y ^ e_y) = (3 * x ^ e_x * y ^ e_y) :=
by
  sorry

end exists_monomial_l122_122101


namespace min_expression_value_l122_122321

noncomputable def expression (x y : ℝ) : ℝ := 2*x^2 + 2*y^2 - 8*x + 6*y + 25

theorem min_expression_value : ∃ (x y : ℝ), expression x y = 12.5 :=
by
  sorry

end min_expression_value_l122_122321


namespace maximize_c_l122_122192

theorem maximize_c (c d e : ℤ) (h1 : 5 * c + (d - 12)^2 + e^3 = 235) (h2 : c < d) : c ≤ 22 :=
sorry

end maximize_c_l122_122192


namespace quadratic_root_relationship_l122_122654

theorem quadratic_root_relationship (a b c : ℝ) (α β : ℝ)
  (h1 : a ≠ 0)
  (h2 : α + β = -b / a)
  (h3 : α * β = c / a)
  (h4 : β = 3 * α) : 
  3 * b^2 = 16 * a * c :=
sorry

end quadratic_root_relationship_l122_122654


namespace greatest_power_of_two_factor_l122_122493

theorem greatest_power_of_two_factor (a b c d : ℕ) (h1 : a = 10) (h2 : b = 1006) (h3 : c = 6) (h4 : d = 503) :
  ∃ k : ℕ, 2^k ∣ (a^b - c^d) ∧ ∀ j : ℕ, 2^j ∣ (a^b - c^d) → j ≤ 503 :=
sorry

end greatest_power_of_two_factor_l122_122493


namespace least_subtract_divisible_l122_122470

theorem least_subtract_divisible:
  ∃ n : ℕ, n = 31 ∧ (13603 - n) % 87 = 0 :=
by
  sorry

end least_subtract_divisible_l122_122470


namespace cost_per_bag_l122_122038

theorem cost_per_bag
  (friends : ℕ)
  (payment_per_friend : ℕ)
  (total_bags : ℕ)
  (total_cost : ℕ)
  (h1 : friends = 3)
  (h2 : payment_per_friend = 5)
  (h3 : total_bags = 5)
  (h4 : total_cost = friends * payment_per_friend) :
  total_cost / total_bags = 3 :=
by {
  sorry
}

end cost_per_bag_l122_122038


namespace motorcycle_speed_for_10_minute_prior_arrival_l122_122454

noncomputable def distance_from_home_to_station (x : ℝ) : Prop :=
  x / 30 + 15 / 60 = x / 18 - 15 / 60

noncomputable def speed_to_arrive_10_minutes_before_departure (x : ℝ) (v : ℝ) : Prop :=
  v = x / (1 - 10 / 60)

theorem motorcycle_speed_for_10_minute_prior_arrival :
  (∀ x : ℝ, distance_from_home_to_station x) →
  (∃ x : ℝ, 
    ∃ v : ℝ, speed_to_arrive_10_minutes_before_departure x v ∧ v = 27) :=
by 
  intro h
  exists 22.5
  exists 27
  unfold distance_from_home_to_station at h
  unfold speed_to_arrive_10_minutes_before_departure
  sorry

end motorcycle_speed_for_10_minute_prior_arrival_l122_122454


namespace no_common_interior_points_l122_122877

open Metric

-- Define the distance conditions for two convex polygons F1 and F2
variables {F1 F2 : Set (EuclideanSpace ℝ (Fin 2))}

-- F1 is a convex polygon
def is_convex (S : Set (EuclideanSpace ℝ (Fin 2))) : Prop :=
  ∀ {x y : EuclideanSpace ℝ (Fin 2)} {a b : ℝ},
    x ∈ S → y ∈ S → 0 ≤ a → 0 ≤ b → a + b = 1 → a • x + b • y ∈ S

-- Conditions provided in the problem
def condition1 (F : Set (EuclideanSpace ℝ (Fin 2))) : Prop :=
  ∀ {x y : EuclideanSpace ℝ (Fin 2)}, x ∈ F → y ∈ F → dist x y ≤ 1

def condition2 (F1 F2 : Set (EuclideanSpace ℝ (Fin 2))) : Prop :=
  ∀ {x : EuclideanSpace ℝ (Fin 2)} {y : EuclideanSpace ℝ (Fin 2)}, x ∈ F1 → y ∈ F2 → dist x y > 1 / Real.sqrt 2

-- The theorem to prove
theorem no_common_interior_points (h1 : is_convex F1) (h2 : is_convex F2) 
  (h3 : condition1 F1) (h4 : condition1 F2) (h5 : condition2 F1 F2) :
  ∀ p ∈ interior F1, ∀ q ∈ interior F2, p ≠ q :=
sorry

end no_common_interior_points_l122_122877


namespace smallest_nonnegative_a_l122_122938

open Real

theorem smallest_nonnegative_a (a b : ℝ) (h_b : b = π / 4)
(sin_eq : ∀ (x : ℤ), sin (a * x + b) = sin (17 * x)) : 
a = 17 - π / 4 := by 
  sorry

end smallest_nonnegative_a_l122_122938


namespace surface_area_change_l122_122066

noncomputable def original_surface_area (l w h : ℝ) : ℝ :=
  2 * (l * w + l * h + w * h)

noncomputable def new_surface_area (l w h c : ℝ) : ℝ :=
  original_surface_area l w h - 
  (3 * (c * c)) + 
  (2 * c * c)

theorem surface_area_change (l w h c : ℝ) (hl : l = 5) (hw : w = 4) (hh : h = 3) (hc : c = 2) :
  new_surface_area l w h c = original_surface_area l w h - 8 :=
by 
  sorry

end surface_area_change_l122_122066


namespace mono_increasing_necessary_not_sufficient_problem_statement_l122_122124

-- Define the function
def f (x : ℝ) (m : ℝ) : ℝ := x^3 + 2*x^2 + m*x + 1

-- Define the first condition of p: f(x) is monotonically increasing in (-∞, +∞)
def is_monotonically_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x ≤ f y

-- Define the second condition q: m > 4/3
def m_gt_4_over_3 (m : ℝ) : Prop := m > 4/3

-- State the theorem: 
theorem mono_increasing_necessary_not_sufficient (m : ℝ):
  is_monotonically_increasing (f x) → m_gt_4_over_3 m → 
  (is_monotonically_increasing (f x) ↔ m ≥ 4/3) ∧ (¬ is_monotonically_increasing (f x) → m > 4/3) := 
by
  sorry

-- Main theorem tying the conditions to the conclusion
theorem problem_statement (m : ℝ):
  is_monotonically_increasing (f x) → m_gt_4_over_3 m → 
  (is_monotonically_increasing (f x) ↔ m ≥ 4/3) ∧ (¬ is_monotonically_increasing (f x) → m > 4/3) :=
  by sorry

end mono_increasing_necessary_not_sufficient_problem_statement_l122_122124


namespace find_angle_D_l122_122583

theorem find_angle_D (A B C D E F : ℝ) (hA : A = 50) (hB : B = 35) (hC : C = 40) 
  (triangle_sum1 : A + B + C + E + F = 180) (triangle_sum2 : D + E + F = 180) : 
  D = 125 :=
by
  -- Only adding a comment, proof omitted for the purpose of this task
  sorry

end find_angle_D_l122_122583


namespace normal_time_to_finish_bs_l122_122730

theorem normal_time_to_finish_bs (P : ℕ) (H1 : P = 5) (H2 : ∀ total_time, total_time = 6 → total_time = (3 / 4) * (P + B)) : B = (8 - P) :=
by sorry

end normal_time_to_finish_bs_l122_122730


namespace range_of_a3_l122_122747

open Real

def convex_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, (a n + a (n + 2)) / 2 ≤ a (n + 1)

def sequence_condition (a : ℕ → ℝ) (b : ℕ → ℝ) : Prop :=
  ∀ n, 1 ≤ n → n < 10 → abs (a n - b n) ≤ 20

def b (n : ℕ) : ℝ := n^2 - 6 * n + 10

theorem range_of_a3 (a : ℕ → ℝ) :
  convex_sequence a →
  a 1 = 1 →
  a 10 = 28 →
  sequence_condition a b →
  7 ≤ a 3 ∧ a 3 ≤ 19 :=
sorry

end range_of_a3_l122_122747


namespace binary_to_decimal_is_1023_l122_122055

-- Define the binary number 1111111111 in terms of its decimal representation
def binary_to_decimal : ℕ :=
  (1 * 2^9 + 1 * 2^8 + 1 * 2^7 + 1 * 2^6 + 1 * 2^5 + 1 * 2^4 + 1 * 2^3 + 1 * 2^2 + 1 * 2^1 + 1 * 2^0)

-- The theorem statement
theorem binary_to_decimal_is_1023 : binary_to_decimal = 1023 :=
by
  sorry

end binary_to_decimal_is_1023_l122_122055


namespace atomic_weight_Ba_l122_122222

-- Definitions for conditions
def atomic_weight_O : ℕ := 16
def molecular_weight_compound : ℕ := 153

-- Theorem statement
theorem atomic_weight_Ba : ∃ bw, molecular_weight_compound = bw + atomic_weight_O ∧ bw = 137 :=
by {
  -- Skip the proof
  sorry
}

end atomic_weight_Ba_l122_122222


namespace num_three_digit_numbers_l122_122177

theorem num_three_digit_numbers (a b c : ℕ) :
  a ≠ 0 →
  b = (a + c) / 2 →
  c = a - b →
  ∃ n1 n2 n3 : ℕ, 
    (n1 = 100 * 3 + 10 * 2 + 1) ∧
    (n2 = 100 * 9 + 10 * 6 + 3) ∧
    (n3 = 100 * 6 + 10 * 4 + 2) ∧ 
    3 = 3 := 
sorry  

end num_three_digit_numbers_l122_122177


namespace ambulance_ride_cost_l122_122328

-- Define the conditions as per the given problem.
def totalBill : ℝ := 5000
def medicationPercentage : ℝ := 0.5
def overnightStayPercentage : ℝ := 0.25
def foodCost : ℝ := 175

-- Define the question to be proved.
theorem ambulance_ride_cost :
  let medicationCost := totalBill * medicationPercentage
  let remainingAfterMedication := totalBill - medicationCost
  let overnightStayCost := remainingAfterMedication * overnightStayPercentage
  let remainingAfterOvernight := remainingAfterMedication - overnightStayCost
  let remainingAfterFood := remainingAfterOvernight - foodCost
  remainingAfterFood = 1700 :=
by
  -- Proof can be completed here
  sorry

end ambulance_ride_cost_l122_122328


namespace difference_sum_even_odd_1000_l122_122362

open Nat

def sum_first_n_odd (n : ℕ) : ℕ :=
  n * n

def sum_first_n_even (n : ℕ) : ℕ :=
  n * (n + 1)

theorem difference_sum_even_odd_1000 :
  sum_first_n_even 1000 - sum_first_n_odd 1000 = 1000 :=
by
  sorry

end difference_sum_even_odd_1000_l122_122362


namespace find_angle_ACB_l122_122645

-- Definitions corresponding to the conditions
def angleABD : ℝ := 145
def angleBAC : ℝ := 105
def supplementary (a b : ℝ) : Prop := a + b = 180
def triangleAngleSum (a b c : ℝ) : Prop := a + b + c = 180

theorem find_angle_ACB :
  ∃ (angleACB : ℝ), 
    supplementary angleABD angleABC ∧
    triangleAngleSum angleBAC angleABC angleACB ∧
    angleACB = 40 := 
sorry

end find_angle_ACB_l122_122645


namespace landA_area_and_ratio_l122_122924

/-
  a = 3, b = 5, c = 6
  p = 1/2 * (a + b + c)
  S = sqrt(p * (p - a) * (p - b) * (p - c))
  S_A = 2 * sqrt(14)
  S_B = 3/2 * sqrt(14)
  S_A / S_B = 4 / 3
-/
theorem landA_area_and_ratio :
  let a := 3
  let b := 5
  let c := 6
  let p := (a + b + c) / 2
  let S_A := Real.sqrt (p * (p - a) * (p - b) * (p - c))
  let S_B := 3 / 2 * Real.sqrt 14
  S_A = 2 * Real.sqrt 14 ∧ S_A / S_B = 4 / 3 :=
by
  sorry

end landA_area_and_ratio_l122_122924


namespace units_produced_today_l122_122324

theorem units_produced_today (n : ℕ) (P : ℕ) (T : ℕ) 
  (h1 : n = 14)
  (h2 : P = 60 * n)
  (h3 : (P + T) / (n + 1) = 62) : 
  T = 90 :=
by
  sorry

end units_produced_today_l122_122324


namespace find_multiplier_value_l122_122697

def number : ℤ := 18
def increase : ℤ := 198

theorem find_multiplier_value (x : ℤ) (h : number * x = number + increase) : x = 12 :=
by
  sorry

end find_multiplier_value_l122_122697


namespace Jack_hands_in_l122_122706

def num_hundred_bills := 2
def num_fifty_bills := 1
def num_twenty_bills := 5
def num_ten_bills := 3
def num_five_bills := 7
def num_one_bills := 27
def to_leave_in_till := 300

def total_money_in_notes : Nat :=
  (num_hundred_bills * 100) +
  (num_fifty_bills * 50) +
  (num_twenty_bills * 20) +
  (num_ten_bills * 10) +
  (num_five_bills * 5) +
  (num_one_bills * 1)

def money_to_hand_in := total_money_in_notes - to_leave_in_till

theorem Jack_hands_in : money_to_hand_in = 142 := by
  sorry

end Jack_hands_in_l122_122706


namespace factor_x10_minus_1296_l122_122120

theorem factor_x10_minus_1296 (x : ℝ) : (x^10 - 1296) = (x^5 + 36) * (x^5 - 36) :=
  by
  sorry

end factor_x10_minus_1296_l122_122120


namespace coin_count_l122_122385

theorem coin_count (x y : ℕ) 
  (h1 : x + y = 12) 
  (h2 : 5 * x + 10 * y = 90) :
  x = 6 ∧ y = 6 := 
sorry

end coin_count_l122_122385


namespace number_of_roses_cut_l122_122070

-- Let's define the initial and final conditions
def initial_roses : ℕ := 6
def final_roses : ℕ := 16

-- Define the number of roses Mary cut from her garden
def roses_cut := final_roses - initial_roses

-- Now, we state the theorem we aim to prove
theorem number_of_roses_cut : roses_cut = 10 :=
by
  -- Proof goes here
  sorry

end number_of_roses_cut_l122_122070


namespace matrix_inverse_eq_scaling_l122_122558

variable (d k : ℚ)

def B : Matrix (Fin 3) (Fin 3) ℚ := ![
  ![1, 2, 3],
  ![4, 5, d],
  ![6, 7, 8]
]

theorem matrix_inverse_eq_scaling :
  (B d)⁻¹ = k • (B d) →
  d = 13/9 ∧ k = -329/52 :=
by
  sorry

end matrix_inverse_eq_scaling_l122_122558


namespace triangle_area_l122_122536

theorem triangle_area (l1 l2 : ℝ → ℝ → Prop)
  (h1 : ∀ x y, l1 x y ↔ 3 * x - y + 12 = 0)
  (h2 : ∀ x y, l2 x y ↔ 3 * x + 2 * y - 6 = 0) :
  ∃ A : ℝ, A = 9 :=
by
  sorry

end triangle_area_l122_122536


namespace base_5_to_decimal_l122_122837

theorem base_5_to_decimal : 
  let b5 := [1, 2, 3, 4] -- base-5 number 1234 in list form
  let decimal := 194
  (b5[0] * 5^3 + b5[1] * 5^2 + b5[2] * 5^1 + b5[3] * 5^0) = decimal :=
by
  -- Proof details go here
  sorry

end base_5_to_decimal_l122_122837


namespace tetrahedron_volume_l122_122814

-- Definition of the required constants and variables
variables {S1 S2 S3 S4 r : ℝ}

-- The volume formula we need to prove
theorem tetrahedron_volume :
  (V = 1/3 * (S1 + S2 + S3 + S4) * r) :=
sorry

end tetrahedron_volume_l122_122814


namespace sum_abc_l122_122771

theorem sum_abc (A B C : ℕ) (hposA : 0 < A) (hposB : 0 < B) (hposC : 0 < C) (hgcd : Nat.gcd A (Nat.gcd B C) = 1)
  (hlog : A * Real.log 5 / Real.log 100 + B * Real.log 2 / Real.log 100 = C) : A + B + C = 5 :=
sorry

end sum_abc_l122_122771


namespace solution_set_inequality_l122_122667

variable (f : ℝ → ℝ)

-- Conditions
def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)
def is_decreasing_on_non_neg (f : ℝ → ℝ) : Prop := ∀ x y, 0 ≤ x → x ≤ y → f y ≤ f x
def f_neg_half_eq_zero (f : ℝ → ℝ) : Prop := f (-1/2) = 0

-- Problem statement
theorem solution_set_inequality (f : ℝ → ℝ) 
  (hf_even : is_even_function f) 
  (hf_decreasing : is_decreasing_on_non_neg f) 
  (hf_neg_half_zero : f_neg_half_eq_zero f) : 
  {x : ℝ | f (Real.logb (1/4) x) < 0} = {x | x > 2} ∪ {x | 0 < x ∧ x < 1/2} :=
  sorry

end solution_set_inequality_l122_122667


namespace stingrays_count_l122_122865

theorem stingrays_count (Sh S : ℕ) (h1 : Sh = 2 * S) (h2 : S + Sh = 84) : S = 28 :=
by
  -- Proof will be filled here
  sorry

end stingrays_count_l122_122865


namespace visitors_not_ill_l122_122772

theorem visitors_not_ill (total_visitors : ℕ) (ill_percentage : ℕ) (fall_ill : ℕ) : 
  total_visitors = 500 → 
  ill_percentage = 40 → 
  fall_ill = (ill_percentage * total_visitors) / 100 →
  total_visitors - fall_ill = 300 :=
by
  intros h1 h2 h3
  sorry

end visitors_not_ill_l122_122772


namespace angles_equal_l122_122922

theorem angles_equal (α θ γ : Real) (hα : 0 < α ∧ α < π / 2) (hθ : 0 < θ ∧ θ < π / 2) (hγ : 0 < γ ∧ γ < π / 2)
  (h : Real.sin (α + γ) * Real.tan α = Real.sin (θ + γ) * Real.tan θ) : α = θ :=
by
  sorry

end angles_equal_l122_122922


namespace baking_dish_to_recipe_book_ratio_is_2_l122_122254

-- Definitions of costs
def cost_recipe_book : ℕ := 6
def cost_ingredient : ℕ := 3
def num_ingredients : ℕ := 5
def cost_apron : ℕ := cost_recipe_book + 1
def total_spent : ℕ := 40

-- Definition to calculate the total cost excluding the baking dish
def cost_excluding_baking_dish : ℕ :=
  cost_recipe_book + cost_apron + cost_ingredient * num_ingredients

-- Definition of cost of baking dish
def cost_baking_dish : ℕ := total_spent - cost_excluding_baking_dish

-- Definition of the ratio
def ratio_baking_dish_to_recipe_book : ℕ := cost_baking_dish / cost_recipe_book

-- Theorem stating that the ratio is 2
theorem baking_dish_to_recipe_book_ratio_is_2 :
  ratio_baking_dish_to_recipe_book = 2 :=
sorry

end baking_dish_to_recipe_book_ratio_is_2_l122_122254


namespace square_side_length_in_right_triangle_l122_122020

theorem square_side_length_in_right_triangle
  (AC BC : ℝ)
  (h1 : AC = 3)
  (h2 : BC = 7)
  (right_triangle : ∃ A B C : ℝ × ℝ, A = (3, 0) ∧ B = (0, 7) ∧ C = (0, 0) ∧ (A.1 - C.1)^2 + (A.2 - C.2)^2 = AC^2 ∧ (B.1 - C.1)^2 + (B.2 - C.2)^2 = BC^2 ∧ (A.1 - B.1)^2 + (A.2 - B.2)^2 = AC^2 + BC^2) :
  ∃ s : ℝ, s = 2.1 :=
by
  -- Proof goes here
  sorry

end square_side_length_in_right_triangle_l122_122020


namespace positive_solution_sqrt_eq_l122_122977

theorem positive_solution_sqrt_eq (y : ℝ) (hy_pos : 0 < y) : 
    (∃ a, a = y ∧ a^2 = y * a) ∧ (∃ b, b = y ∧ b^2 = y + b) ∧ y = 2 :=
by 
  sorry

end positive_solution_sqrt_eq_l122_122977


namespace correct_calculation_l122_122611

theorem correct_calculation (a b c d : ℤ) (h1 : a = -1) (h2 : b = -3) (h3 : c = 3) (h4 : d = -3) :
  a * b = c :=
by 
  rw [h1, h2]
  exact h3.symm

end correct_calculation_l122_122611


namespace all_three_digits_same_two_digits_same_all_digits_different_l122_122853

theorem all_three_digits_same (a : ℕ) (h1 : a < 10) (h2 : 3 * a = 24) : a = 8 :=
by sorry

theorem two_digits_same (a b : ℕ) (h1 : a < 10) (h2 : b < 10) (h3 : 2 * a + b = 24 ∨ a + 2 * b = 24) : 
  (a = 9 ∧ b = 6) ∨ (a = 6 ∧ b = 9) :=
by sorry

theorem all_digits_different (a b c : ℕ) (h1 : a < 10) (h2 : b < 10) (h3 : c < 10)
  (h4 : a ≠ b) (h5 : a ≠ c) (h6 : b ≠ c) (h7 : a + b + c = 24) :
  (a, b, c) = (7, 8, 9) ∨ (a, b, c) = (7, 9, 8) ∨ (a, b, c) = (8, 7, 9) ∨ (a, b, c) = (8, 9, 7) ∨ (a, b, c) = (9, 7, 8) ∨ (a, b, c) = (9, 8, 7) :=
by sorry

end all_three_digits_same_two_digits_same_all_digits_different_l122_122853


namespace car_and_cyclist_speeds_and_meeting_point_l122_122304

/-- 
(1) Distance between points $A$ and $B$ is $80 \mathrm{~km}$.
(2) After one hour, the distance between them reduces to $24 \mathrm{~km}$.
(3) The cyclist takes a 1-hour rest but they meet $90$ minutes after their departure.
-/
def initial_distance : ℝ := 80 -- km
def distance_after_one_hour : ℝ := 24 -- km apart after 1 hour
def cyclist_rest_duration : ℝ := 1 -- hour
def meeting_time : ℝ := 1.5 -- hours (90 minutes after departure)

def car_speed : ℝ := 40 -- km/hr
def cyclist_speed : ℝ := 16 -- km/hr

theorem car_and_cyclist_speeds_and_meeting_point :
  initial_distance = 80 → 
  distance_after_one_hour = 24 → 
  cyclist_rest_duration = 1 → 
  meeting_time = 1.5 → 
  car_speed = 40 ∧ cyclist_speed = 16 ∧ meeting_point_from_A = 60 ∧ meeting_point_from_B = 20 :=
by
  sorry

end car_and_cyclist_speeds_and_meeting_point_l122_122304


namespace john_votes_l122_122336

theorem john_votes (J : ℝ) (total_votes : ℝ) (third_candidate_votes : ℝ) (james_votes : ℝ) 
  (h1 : total_votes = 1150) 
  (h2 : third_candidate_votes = J + 150) 
  (h3 : james_votes = 0.70 * (total_votes - J - third_candidate_votes)) 
  (h4 : total_votes = J + james_votes + third_candidate_votes) : 
  J = 500 := 
by 
  rw [h1, h2, h3] at h4 
  sorry

end john_votes_l122_122336


namespace quadratic_inequality_solution_set_l122_122473

theorem quadratic_inequality_solution_set (a b : ℝ) :
  (∀ x : ℝ, (2 < x ∧ x < 3) → (ax^2 + 5*x + b > 0)) →
  ∃ x : ℝ, (-1/2 < x ∧ x < -1/3) :=
sorry

end quadratic_inequality_solution_set_l122_122473


namespace part1_part2_l122_122217

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then 2^(-x)
  else (Real.log x / Real.log 3 - 1) * (Real.log x / Real.log 3 - 2)

theorem part1 : f (Real.log 3 / Real.log 2 - Real.log 2 / Real.log 2) = 2 / 3 := by
  sorry

theorem part2 : ∃ x : ℝ, f x = -1 / 4 := by
  sorry

end part1_part2_l122_122217


namespace distance_from_origin_to_point_on_parabola_l122_122799

theorem distance_from_origin_to_point_on_parabola
  (y x : ℝ)
  (focus : ℝ × ℝ := (4, 0))
  (on_parabola : y^2 = 8 * x)
  (distance_to_focus : Real.sqrt ((x - 4)^2 + y^2) = 4) :
  Real.sqrt (x^2 + y^2) = 2 * Real.sqrt 5 :=
by
  sorry

end distance_from_origin_to_point_on_parabola_l122_122799


namespace expectation_S_tau_eq_varliminf_ratio_S_tau_l122_122434

noncomputable def xi : ℕ → ℝ := sorry
noncomputable def tau : ℝ := sorry

-- Statement (a)
theorem expectation_S_tau_eq (ES_tau : ℝ := sorry) (E_tau : ℝ := sorry) (E_xi1 : ℝ := sorry) :
  ES_tau = E_tau * E_xi1 := sorry

-- Statement (b)
theorem varliminf_ratio_S_tau (liminf_val : ℝ := sorry) (E_tau : ℝ := sorry) :
  (liminf_val = E_tau) := sorry

end expectation_S_tau_eq_varliminf_ratio_S_tau_l122_122434


namespace exists_diff_shape_and_color_l122_122353

variable (Pitcher : Type) 
variable (shape color : Pitcher → Prop)
variable (exists_diff_shape : ∃ (A B : Pitcher), shape A ≠ shape B)
variable (exists_diff_color : ∃ (A B : Pitcher), color A ≠ color B)

theorem exists_diff_shape_and_color : ∃ (A B : Pitcher), shape A ≠ shape B ∧ color A ≠ color B :=
  sorry

end exists_diff_shape_and_color_l122_122353


namespace general_term_formula_l122_122968

variable (a S : ℕ → ℚ)

-- Condition 1: The sum of the first n terms of the sequence {a_n} is S_n
def sum_first_n_terms (n : ℕ) : ℚ := S n

-- Condition 2: a_n = 3S_n - 2
def a_n (n : ℕ) : Prop := a n = 3 * S n - 2

theorem general_term_formula (n : ℕ) (h1 : a 1 = 1)
  (h2 : ∀ k, k ≥ 2 → a (k) = - (1/2) * a (k - 1) ) : 
  a n = (-1/2)^(n-1) :=
sorry

end general_term_formula_l122_122968


namespace tetrahedron_cube_volume_ratio_l122_122497

theorem tetrahedron_cube_volume_ratio (s : ℝ) (h_s : s > 0):
    let V_cube := s ^ 3
    let a := s * Real.sqrt 3
    let V_tetrahedron := (Real.sqrt 2 / 12) * a ^ 3
    (V_tetrahedron / V_cube) = (Real.sqrt 6 / 4) := by
    sorry

end tetrahedron_cube_volume_ratio_l122_122497


namespace problem_EF_fraction_of_GH_l122_122176

theorem problem_EF_fraction_of_GH (E F G H : Type) 
  (GE EH GH GF FH EF : ℝ) 
  (h1 : GE = 3 * EH) 
  (h2 : GF = 8 * FH)
  (h3 : GH = GE + EH)
  (h4 : GH = GF + FH) : 
  EF = 5 / 36 * GH :=
by
  sorry

end problem_EF_fraction_of_GH_l122_122176


namespace trig_identity_proof_l122_122760

theorem trig_identity_proof :
  (1 - 1 / (Real.cos (Real.pi / 6))) *
  (1 + 1 / (Real.sin (Real.pi / 3))) *
  (1 - 1 / (Real.sin (Real.pi / 6))) *
  (1 + 1 / (Real.cos (Real.pi / 3))) = 3 :=
by sorry

end trig_identity_proof_l122_122760


namespace sum_of_nine_l122_122433

theorem sum_of_nine (S : ℕ → ℕ) (a : ℕ → ℕ) (h₀ : ∀ (n : ℕ), S n = n * (a 1 + a n) / 2)
(h₁ : S 3 = 30) (h₂ : S 6 = 100) : S 9 = 240 := 
sorry

end sum_of_nine_l122_122433


namespace count_ball_distribution_l122_122394

theorem count_ball_distribution (A B C D : ℕ) (balls : ℕ) :
  (A + B > C + D ∧ A + B + C + D = balls) → 
  (balls = 30) →
  (∃ n, n = 2600) :=
by
  intro h_ball_dist h_balls
  sorry

end count_ball_distribution_l122_122394


namespace mimi_spent_on_clothes_l122_122230

theorem mimi_spent_on_clothes : 
  let A := 800
  let N := 2 * A
  let S := 4 * A
  let P := 1 / 2 * N
  let total_spending := 10000
  let total_sneaker_spending := A + N + S + P
  let amount_spent_on_clothes := total_spending - total_sneaker_spending
  amount_spent_on_clothes = 3600 := 
by
  sorry

end mimi_spent_on_clothes_l122_122230


namespace relation_P_Q_l122_122576

def P : Set ℝ := {x | x ≠ 0}
def Q : Set ℝ := {x | x > 0}
def complement_P : Set ℝ := {0}

theorem relation_P_Q : Q ∩ complement_P = ∅ := 
by sorry

end relation_P_Q_l122_122576


namespace cannot_make_62_cents_with_five_coins_l122_122695

theorem cannot_make_62_cents_with_five_coins :
  ∀ (p n d q : ℕ), p + n + d + q = 5 ∧ q ≤ 1 →
  1 * p + 5 * n + 10 * d + 25 * q ≠ 62 := by
  intro p n d q h
  sorry

end cannot_make_62_cents_with_five_coins_l122_122695


namespace largest_multiple_of_8_smaller_than_neg_80_l122_122805

theorem largest_multiple_of_8_smaller_than_neg_80 :
  ∃ n : ℤ, (8 ∣ n) ∧ n < -80 ∧ ∀ m : ℤ, (8 ∣ m ∧ m < -80 → m ≤ n) :=
sorry

end largest_multiple_of_8_smaller_than_neg_80_l122_122805


namespace polygon_proof_l122_122698

-- Define the conditions and the final proof problem.
theorem polygon_proof 
  (interior_angle : ℝ) 
  (side_length : ℝ) 
  (h1 : interior_angle = 160) 
  (h2 : side_length = 4) 
  : ∃ n : ℕ, ∃ P : ℝ, (interior_angle = 180 * (n - 2) / n) ∧ (P = n * side_length) ∧ (n = 18) ∧ (P = 72) :=
by
  sorry

end polygon_proof_l122_122698


namespace g_of_negative_8_l122_122438

def f (x : ℝ) : ℝ := 4 * x - 9
def g (y : ℝ) : ℝ := y^2 + 6 * y - 7

theorem g_of_negative_8 : g (-8) = -87 / 16 :=
by
  -- Proof goes here
  sorry

end g_of_negative_8_l122_122438


namespace common_point_exists_l122_122479

theorem common_point_exists (a b c : ℝ) :
  ∃ x y : ℝ, y = a * x ^ 2 - b * x + c ∧ y = b * x ^ 2 - c * x + a ∧ y = c * x ^ 2 - a * x + b :=
  sorry

end common_point_exists_l122_122479


namespace iterate_g_eq_2_l122_122873

def g (n : ℕ) : ℕ :=
if n % 2 = 1 then n^2 - 2*n + 2 else 2*n

theorem iterate_g_eq_2 {n : ℕ} (hn : 1 ≤ n ∧ n ≤ 100): 
  (∃ m : ℕ, (Nat.iterate g m n) = 2) ↔ n = 1 :=
by
sorry

end iterate_g_eq_2_l122_122873


namespace trigonometric_identity_l122_122209

theorem trigonometric_identity (theta : ℝ) (h : Real.cos ((5 * Real.pi)/12 - theta) = 1/3) :
  Real.sin ((Real.pi)/12 + theta) = 1/3 :=
by
  sorry

end trigonometric_identity_l122_122209


namespace magnitude_of_complex_l122_122099

def complex_number := Complex.mk 2 3 -- Define the complex number 2+3i

theorem magnitude_of_complex : Complex.abs complex_number = Real.sqrt 13 := by
  sorry

end magnitude_of_complex_l122_122099


namespace arithmetic_sequence_geometric_condition_l122_122525

theorem arithmetic_sequence_geometric_condition :
  ∃ d : ℝ, d ≠ 0 ∧ (∀ (a_n : ℕ → ℝ), (a_n 1 = 1) ∧ 
    (a_n 3 = a_n 1 + 2 * d) ∧ (a_n 13 = a_n 1 + 12 * d) ∧ 
    (a_n 3 ^ 2 = a_n 1 * a_n 13) ↔ d = 2) :=
by 
  sorry

end arithmetic_sequence_geometric_condition_l122_122525


namespace walking_time_12_hours_l122_122956

theorem walking_time_12_hours :
  ∀ t : ℝ, 
  (∀ (v1 v2 : ℝ), 
  v1 = 7 ∧ v2 = 3 →
  120 = (v1 + v2) * t) →
  t = 12 := 
by
  intros t h
  specialize h 7 3 ⟨rfl, rfl⟩
  sorry

end walking_time_12_hours_l122_122956


namespace heather_blocks_l122_122170

theorem heather_blocks (initial_blocks : ℕ) (shared_blocks : ℕ) (remaining_blocks : ℕ) :
  initial_blocks = 86 → shared_blocks = 41 → remaining_blocks = initial_blocks - shared_blocks → remaining_blocks = 45 :=
by
  sorry

end heather_blocks_l122_122170


namespace matrix_addition_correct_l122_122464

def matrixA : Matrix (Fin 2) (Fin 2) ℤ := fun i j =>
  if i = 0 then
    if j = 0 then 4 else -2
  else
    if j = 0 then -3 else 5

def matrixB : Matrix (Fin 2) (Fin 2) ℤ := fun i j =>
  if i = 0 then
    if j = 0 then -6 else 0
  else
    if j = 0 then 7 else -8

def resultMatrix : Matrix (Fin 2) (Fin 2) ℤ := fun i j =>
  if i = 0 then
    if j = 0 then -2 else -2
  else
    if j = 0 then 4 else -3

theorem matrix_addition_correct :
  matrixA + matrixB = resultMatrix :=
by
  sorry

end matrix_addition_correct_l122_122464


namespace asymptotic_lines_of_hyperbola_l122_122540

open Real

-- Given: Hyperbola equation
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 1

-- To Prove: Asymptotic lines equation
theorem asymptotic_lines_of_hyperbola : 
  ∀ x y : ℝ, hyperbola x y → (y = x ∨ y = -x) :=
by
  intros x y h
  sorry

end asymptotic_lines_of_hyperbola_l122_122540


namespace intersection_points_l122_122298

theorem intersection_points : 
  (∃ x : ℝ, y = -2 * x + 4 ∧ y = 0 ∧ (x, y) = (2, 0)) ∧
  (∃ y : ℝ, y = -2 * 0 + 4 ∧ (0, y) = (0, 4)) :=
by
  sorry

end intersection_points_l122_122298


namespace unique_ordered_triple_lcm_l122_122386

theorem unique_ordered_triple_lcm:
  ∃! (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c), 
    Nat.lcm a b = 2100 ∧ Nat.lcm b c = 3150 ∧ Nat.lcm c a = 4200 :=
by
  sorry

end unique_ordered_triple_lcm_l122_122386


namespace determine_parabola_coefficients_l122_122150

noncomputable def parabola_coefficients (a b c : ℚ) : Prop :=
  ∀ (x y : ℚ), 
      (y = a * x^2 + b * x + c) ∧
      (
        ((4, 5) = (x, y)) ∧
        ((2, 3) = (x, y))
      )

theorem determine_parabola_coefficients :
  parabola_coefficients (-1/2) 4 (-3) :=
by
  sorry

end determine_parabola_coefficients_l122_122150


namespace exists_x_inequality_l122_122489

theorem exists_x_inequality (a : ℝ) : 
  (∃ x : ℝ, x^2 - 3 * a * x + 9 < 0) ↔ a < -2 ∨ a > 2 :=
by
  sorry

end exists_x_inequality_l122_122489


namespace selling_price_of_cycle_l122_122193

theorem selling_price_of_cycle (cost_price : ℕ) (gain_percent : ℕ) (cost_price_eq : cost_price = 1500) (gain_percent_eq : gain_percent = 8) :
  ∃ selling_price : ℕ, selling_price = 1620 := 
by
  sorry

end selling_price_of_cycle_l122_122193


namespace hcf_of_three_numbers_l122_122252

theorem hcf_of_three_numbers (a b c : ℕ) (h1 : Nat.lcm a (Nat.lcm b c) = 45600) (h2 : a * b * c = 109183500000) :
  Nat.gcd a (Nat.gcd b c) = 2393750 := by
  sorry

end hcf_of_three_numbers_l122_122252


namespace playground_area_l122_122959

theorem playground_area
  (w l : ℕ)
  (h₁ : l = 2 * w + 25)
  (h₂ : 2 * (l + w) = 650) :
  w * l = 22500 := 
sorry

end playground_area_l122_122959


namespace age_of_15th_student_l122_122319

theorem age_of_15th_student
  (avg_age_15_students : ℕ)
  (total_students : ℕ)
  (avg_age_5_students : ℕ)
  (students_5 : ℕ)
  (avg_age_9_students : ℕ)
  (students_9 : ℕ)
  (total_age_15_students_eq : avg_age_15_students * total_students = 225)
  (total_age_5_students_eq : avg_age_5_students * students_5 = 70)
  (total_age_9_students_eq : avg_age_9_students * students_9 = 144) :
  (avg_age_15_students * total_students - (avg_age_5_students * students_5 + avg_age_9_students * students_9) = 11) :=
by
  sorry

end age_of_15th_student_l122_122319


namespace jonah_total_raisins_l122_122060

-- Define the amounts of yellow and black raisins added
def yellow_raisins : ℝ := 0.3
def black_raisins : ℝ := 0.4

-- The main statement to be proved
theorem jonah_total_raisins : yellow_raisins + black_raisins = 0.7 :=
by 
  sorry

end jonah_total_raisins_l122_122060


namespace truck_weight_l122_122629

theorem truck_weight (T R : ℝ) (h1 : T + R = 7000) (h2 : R = 0.5 * T - 200) : T = 4800 :=
by sorry

end truck_weight_l122_122629


namespace zoe_remaining_pictures_l122_122052

-- Definitions for the problem conditions
def monday_pictures := 24
def tuesday_pictures := 37
def wednesday_pictures := 50
def thursday_pictures := 33
def friday_pictures := 44

def rate_first := 4
def rate_second := 5
def rate_third := 6
def rate_fourth := 3
def rate_fifth := 7

def days_colored (start_day : ℕ) (end_day := 6) := end_day - start_day

def remaining_pictures (total_pictures : ℕ) (rate_per_day : ℕ) (days : ℕ) : ℕ :=
  total_pictures - (rate_per_day * days)

-- Main theorem statement
theorem zoe_remaining_pictures : 
  remaining_pictures monday_pictures rate_first (days_colored 1) +
  remaining_pictures tuesday_pictures rate_second (days_colored 2) +
  remaining_pictures wednesday_pictures rate_third (days_colored 3) +
  remaining_pictures thursday_pictures rate_fourth (days_colored 4) +
  remaining_pictures friday_pictures rate_fifth (days_colored 5) = 117 :=
  sorry

end zoe_remaining_pictures_l122_122052


namespace height_of_triangle_l122_122612

-- Define the dimensions of the rectangle
variable (l w : ℝ)

-- Assume the base of the triangle is equal to the length of the rectangle
-- We need to prove that the height of the triangle h = 2w

theorem height_of_triangle (h : ℝ) (hl_eq_length : l > 0) (hw_eq_width : w > 0) :
  (l * w) = (1 / 2) * l * h → h = 2 * w :=
by
  sorry

end height_of_triangle_l122_122612


namespace number_of_multiples_of_15_l122_122567

theorem number_of_multiples_of_15 (a b : ℕ) (h₁ : a = 15) (h₂ : b = 305) : 
  ∃ n : ℕ, n = 20 ∧ ∀ k, (1 ≤ k ∧ k ≤ n) → (15 * k) ≥ a ∧ (15 * k) ≤ b := by
  sorry

end number_of_multiples_of_15_l122_122567


namespace book_pairs_count_l122_122259

theorem book_pairs_count :
  let mystery_count := 3
  let fantasy_count := 4
  let biography_count := 3
  mystery_count * fantasy_count + mystery_count * biography_count + fantasy_count * biography_count = 33 :=
by 
  sorry

end book_pairs_count_l122_122259


namespace problem_l122_122858

theorem problem (x : ℕ) (h : 2^x + 2^x + 2^x = 256) : x * (x + 1) = 72 :=
sorry

end problem_l122_122858


namespace ruth_started_with_89_apples_l122_122898

theorem ruth_started_with_89_apples 
  (initial_apples : ℕ)
  (shared_apples : ℕ)
  (remaining_apples : ℕ)
  (h1 : shared_apples = 5)
  (h2 : remaining_apples = 84)
  (h3 : remaining_apples = initial_apples - shared_apples) : 
  initial_apples = 89 :=
by
  sorry

end ruth_started_with_89_apples_l122_122898


namespace find_values_of_a_b_l122_122411

variable (a b : ℤ)

def A : Set ℤ := {1, b, a + b}
def B : Set ℤ := {a - b, a * b}
def common_set : Set ℤ := {-1, 0}

theorem find_values_of_a_b (h : A a b ∩ B a b = common_set) : (a, b) = (-1, 0) := by
  sorry

end find_values_of_a_b_l122_122411


namespace problem_statement_l122_122279

-- Given the conditions and the goal
theorem problem_statement (x y z : ℝ) 
  (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (hxyz_sum : x + y + z = 1) :
  (2 * x^2 / (y + z)) + (2 * y^2 / (z + x)) + (2 * z^2 / (x + y)) ≥ 1 :=
by
  sorry

end problem_statement_l122_122279


namespace sequence_bound_l122_122058

theorem sequence_bound (a b c : ℕ → ℝ) :
  (a 0 = 1) ∧ (b 0 = 0) ∧ (c 0 = 0) ∧
  (∀ n, n ≥ 1 → a n = a (n-1) + c (n-1) / n) ∧
  (∀ n, n ≥ 1 → b n = b (n-1) + a (n-1) / n) ∧
  (∀ n, n ≥ 1 → c n = c (n-1) + b (n-1) / n) →
  ∀ n, n ≥ 1 → |a n - (n + 1) / 3| < 2 / Real.sqrt (3 * n) :=
by sorry

end sequence_bound_l122_122058


namespace simplify_fraction_l122_122902

theorem simplify_fraction (b : ℕ) (hb : b = 2) : (15 * b ^ 4) / (45 * b ^ 3) = 2 / 3 :=
by
  sorry

end simplify_fraction_l122_122902


namespace aqua_park_earnings_l122_122496

/-- Define the costs and groups of visitors. --/
def admission_fee : ℕ := 12
def tour_fee : ℕ := 6
def group1_size : ℕ := 10
def group2_size : ℕ := 5

/-- Define the total earnings of the aqua park. --/
def total_earnings : ℕ := (admission_fee + tour_fee) * group1_size + admission_fee * group2_size

/-- Prove that the total earnings are $240. --/
theorem aqua_park_earnings : total_earnings = 240 :=
by
  -- proof steps would go here
  sorry

end aqua_park_earnings_l122_122496


namespace min_avg_score_less_than_record_l122_122075

theorem min_avg_score_less_than_record
  (old_record_avg : ℝ := 287.5)
  (players : ℕ := 6)
  (rounds : ℕ := 12)
  (total_points_11_rounds : ℝ := 19350.5)
  (bonus_points_9_rounds : ℕ := 300) :
  ∀ final_round_avg : ℝ, (final_round_avg = (old_record_avg * players * rounds - total_points_11_rounds + bonus_points_9_rounds) / players) →
  old_record_avg - final_round_avg = 12.5833 :=
by {
  sorry
}

end min_avg_score_less_than_record_l122_122075


namespace part1_part2_l122_122603

variable {R : Type} [LinearOrderedField R]

def f (x : R) : R := abs (x - 2) + 2
def g (m : R) (x : R) : R := m * abs x

theorem part1 (x : R) : f x > 5 ↔ x < -1 ∨ x > 5 := by
  sorry

theorem part2 (m : R) : (∀ x : R, f x ≥ g m x) → m ∈ Set.Iic (1 : R) := by
  sorry

end part1_part2_l122_122603


namespace last_10_digits_repeat_periodically_l122_122221

theorem last_10_digits_repeat_periodically :
  ∃ (p : ℕ) (n₀ : ℕ), p = 4 * 10^9 ∧ n₀ = 10 ∧ 
  ∀ n, (2^(n + p) % 10^10 = 2^n % 10^10) :=
by sorry

end last_10_digits_repeat_periodically_l122_122221


namespace jenna_round_trip_pay_l122_122653

theorem jenna_round_trip_pay :
  let pay_per_mile := 0.40
  let one_way_miles := 400
  let round_trip_miles := 2 * one_way_miles
  let total_pay := round_trip_miles * pay_per_mile
  total_pay = 320 := 
by
  sorry

end jenna_round_trip_pay_l122_122653


namespace profit_at_end_of_first_year_l122_122714

theorem profit_at_end_of_first_year :
  let total_amount := 50000
  let part1 := 30000
  let interest_rate1 := 0.10
  let part2 := total_amount - part1
  let interest_rate2 := 0.20
  let time_period := 1
  let interest1 := part1 * interest_rate1 * time_period
  let interest2 := part2 * interest_rate2 * time_period
  let total_profit := interest1 + interest2
  total_profit = 7000 := 
by 
  sorry

end profit_at_end_of_first_year_l122_122714


namespace power_function_half_l122_122773

theorem power_function_half (f : ℝ → ℝ) (h : ∀ x : ℝ, f x = x ^ (1/2)) (hx : f 4 = 2) : 
  f (1/2) = (Real.sqrt 2) / 2 :=
by sorry

end power_function_half_l122_122773


namespace scientific_notation_1300000_l122_122498

theorem scientific_notation_1300000 :
  1300000 = 1.3 * 10^6 :=
sorry

end scientific_notation_1300000_l122_122498


namespace polynomial_solution_l122_122633

noncomputable def polynomial_form (P : ℝ → ℝ) : Prop :=
∀ (x y z : ℝ), x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ (2 * x * y * z = x + y + z) →
(P x / (y * z) + P y / (z * x) + P z / (x * y) = P (x - y) + P (y - z) + P (z - x))

theorem polynomial_solution (P : ℝ → ℝ) : polynomial_form P → ∃ c : ℝ, ∀ x : ℝ, P x = c * (x ^ 2 + 3) := 
by 
  sorry

end polynomial_solution_l122_122633


namespace hot_dogs_sold_next_innings_l122_122085

-- Defining the conditions
variables (total_initial hot_dogs_sold_first_innings hot_dogs_left : ℕ)

-- Given conditions that need to hold true
axiom initial_count : total_initial = 91
axiom first_innings_sold : hot_dogs_sold_first_innings = 19
axiom remaining_hot_dogs : hot_dogs_left = 45

-- Prove the number of hot dogs sold during the next three innings is 27
theorem hot_dogs_sold_next_innings : total_initial - (hot_dogs_sold_first_innings + hot_dogs_left) = 27 :=
by
  sorry

end hot_dogs_sold_next_innings_l122_122085


namespace geometric_series_sum_l122_122282

theorem geometric_series_sum :
  let a := -3
  let r := -2
  let n := 9
  let term := a * r^(n-1)
  let Sn := (a * (r^n - 1)) / (r - 1)
  term = -768 → Sn = 514 := by
  intros a r n term Sn h_term
  sorry

end geometric_series_sum_l122_122282


namespace constant_term_in_expansion_l122_122484

theorem constant_term_in_expansion :
  let f := (x - (2 / x^2))
  let expansion := f^9
  ∃ c: ℤ, expansion = c ∧ c = -672 :=
sorry

end constant_term_in_expansion_l122_122484


namespace find_m_l122_122139

theorem find_m (a b m : ℝ) :
  (∀ x : ℝ, (x^2 - b * x + b^2) / (a * x^2 - b^2) = (m - 1) / (m + 1) → (∀ y : ℝ, x = y ∧ x = -y)) →
  c = b^2 →
  m = (a - 1) / (a + 1) :=
by
  sorry

end find_m_l122_122139


namespace intersection_of_A_and_B_l122_122867

def A : Set ℝ := {x : ℝ | -1 < x ∧ x ≤ 2}
def B : Set ℝ := {-1, 0, 1}
def intersection : Set ℝ := {0, 1}

theorem intersection_of_A_and_B : A ∩ B = intersection := 
by sorry

end intersection_of_A_and_B_l122_122867


namespace derivative_f_intervals_of_monotonicity_extrema_l122_122636

noncomputable def f (x : ℝ) := (x + 1)^2 * (x - 1)

theorem derivative_f (x : ℝ) : deriv f x = 3 * x^2 + 2 * x - 1 := sorry

theorem intervals_of_monotonicity :
  (∀ x, x < -1 → deriv f x > 0) ∧
  (∀ x, -1 < x ∧ x < -1/3 → deriv f x < 0) ∧
  (∀ x, x > -1/3 → deriv f x > 0) := sorry

theorem extrema :
  f (-1) = 0 ∧
  f (-1/3) = -(32 / 27) := sorry

end derivative_f_intervals_of_monotonicity_extrema_l122_122636


namespace minimum_spend_on_boxes_l122_122915

noncomputable def box_length : ℕ := 20
noncomputable def box_width : ℕ := 20
noncomputable def box_height : ℕ := 12
noncomputable def cost_per_box : ℝ := 0.40
noncomputable def total_volume : ℕ := 2400000

theorem minimum_spend_on_boxes : 
  (total_volume / (box_length * box_width * box_height)) * cost_per_box = 200 :=
by
  sorry

end minimum_spend_on_boxes_l122_122915


namespace find_q_l122_122804

-- Define the conditions and the statement to prove
theorem find_q (p q : ℝ) (hp1 : p > 1) (hq1 : q > 1) 
  (h1 : 1 / p + 1 / q = 3 / 2)
  (h2 : p * q = 9) : q = 6 := 
sorry

end find_q_l122_122804


namespace positive_number_square_roots_l122_122302

theorem positive_number_square_roots (a : ℝ) 
  (h1 : (2 * a - 1) ^ 2 = (a - 2) ^ 2) 
  (h2 : ∃ b : ℝ, b > 0 ∧ ((2 * a - 1) = b ∨ (a - 2) = b)) : 
  ∃ n : ℝ, n = 1 :=
by
  sorry

end positive_number_square_roots_l122_122302


namespace polar_to_rectangular_l122_122148

theorem polar_to_rectangular : 
  ∀ (r θ : ℝ), r = 2 ∧ θ = 2 * Real.pi / 3 → 
  (r * Real.cos θ, r * Real.sin θ) = (-1, Real.sqrt 3) := by
  sorry

end polar_to_rectangular_l122_122148


namespace c_share_is_160_l122_122658

theorem c_share_is_160 (a b c : ℕ) (total : ℕ) (h1 : 4 * a = 5 * b) (h2 : 5 * b = 10 * c) (h_total : a + b + c = 880) : c = 160 :=
by
  sorry

end c_share_is_160_l122_122658


namespace beaker_water_division_l122_122917

-- Given conditions
variable (buckets : ℕ) (bucket_capacity : ℕ) (remaining_water : ℝ)
  (total_buckets : ℕ := 2) (capacity : ℕ := 120) (remaining : ℝ := 2.4)

-- Theorem statement
theorem beaker_water_division (h1 : buckets = total_buckets)
                             (h2 : bucket_capacity = capacity)
                             (h3 : remaining_water = remaining) :
                             (total_water : ℝ := buckets * bucket_capacity + remaining_water ) → 
                             (water_per_beaker : ℝ := total_water / 3) →
                             water_per_beaker = 80.8 :=
by
  -- Skipping the proof steps here, will use sorry
  sorry

end beaker_water_division_l122_122917


namespace one_element_in_A_inter_B_range_m_l122_122197

theorem one_element_in_A_inter_B_range_m (m : ℝ) :
  let A := {p : ℝ × ℝ | ∃ x, p.1 = x ∧ p.2 = -x^2 + m * x - 1}
  let B := {p : ℝ × ℝ | ∃ x, p.1 = x ∧ p.2 = 3 - x ∧ 0 ≤ x ∧ x ≤ 3}
  (∃! p, p ∈ A ∧ p ∈ B) → (m = 3 ∨ m > 10 / 3) :=
by
  sorry

end one_element_in_A_inter_B_range_m_l122_122197


namespace swan_populations_after_10_years_l122_122914

noncomputable def swan_population_rita (R : ℝ) : ℝ :=
  480 * (1 - R / 100) ^ 10

noncomputable def swan_population_sarah (S : ℝ) : ℝ :=
  640 * (1 - S / 100) ^ 10

noncomputable def swan_population_tom (T : ℝ) : ℝ :=
  800 * (1 - T / 100) ^ 10

theorem swan_populations_after_10_years 
  (R S T : ℝ) :
  swan_population_rita R = 480 * (1 - R / 100) ^ 10 ∧
  swan_population_sarah S = 640 * (1 - S / 100) ^ 10 ∧
  swan_population_tom T = 800 * (1 - T / 100) ^ 10 := 
by sorry

end swan_populations_after_10_years_l122_122914


namespace oranges_ratio_l122_122250

theorem oranges_ratio (initial_oranges_kgs : ℕ) (additional_oranges_kgs : ℕ) (total_oranges_three_weeks : ℕ) :
  initial_oranges_kgs = 10 →
  additional_oranges_kgs = 5 →
  total_oranges_three_weeks = 75 →
  (2 * (total_oranges_three_weeks - (initial_oranges_kgs + additional_oranges_kgs)) / 2) / (initial_oranges_kgs + additional_oranges_kgs) = 2 :=
by
  intros h_initial h_additional h_total
  sorry

end oranges_ratio_l122_122250


namespace find_multiple_l122_122767

-- Definitions of the divisor, original number, and remainders given in the problem conditions.
def D : ℕ := 367
def remainder₁ : ℕ := 241
def remainder₂ : ℕ := 115

-- Statement of the problem.
theorem find_multiple (N m k l : ℕ) :
  (N = k * D + remainder₁) →
  (m * N = l * D + remainder₂) →
  ∃ m, m > 0 ∧ 241 * m - 115 % 367 = 0 ∧ m = 2 :=
by
  sorry

end find_multiple_l122_122767


namespace smallest_value_of_a_b_l122_122366

theorem smallest_value_of_a_b :
  ∃ (a b : ℤ), (∀ x : ℤ, ((x^2 + a*x + 20) = 0 ∨ (x^2 + 17*x + b) = 0) → x < 0) ∧ a + b = -5 :=
sorry

end smallest_value_of_a_b_l122_122366


namespace first_pump_time_l122_122754

-- Definitions for the conditions provided
def newer_model_rate := 1 / 6
def combined_rate := 1 / 3.6
def time_for_first_pump : ℝ := 9

-- The theorem to be proven
theorem first_pump_time (T : ℝ) (h1 : 1 / 6 + 1 / T = 1 / 3.6) : T = 9 :=
sorry

end first_pump_time_l122_122754


namespace find_volume_of_sphere_l122_122401

noncomputable def volume_of_sphere (AB BC AA1 : ℝ) (hAB : AB = 2) (hBC : BC = 2) (hAA1 : AA1 = 2 * Real.sqrt 2) : ℝ :=
  let diagonal := Real.sqrt (AB^2 + BC^2 + AA1^2)
  let radius := diagonal / 2
  (4 * Real.pi * radius^3) / 3

theorem find_volume_of_sphere : volume_of_sphere 2 2 (2 * Real.sqrt 2) (by rfl) (by rfl) (by rfl) = (32 * Real.pi) / 3 :=
by
  sorry

end find_volume_of_sphere_l122_122401


namespace car_speed_first_hour_l122_122613

theorem car_speed_first_hour (x : ℝ) (h1 : (x + 75) / 2 = 82.5) : x = 90 :=
sorry

end car_speed_first_hour_l122_122613


namespace add_multiply_round_l122_122928

theorem add_multiply_round :
  let a := 73.5891
  let b := 24.376
  let c := (a + b) * 2
  (Float.round (c * 100) / 100) = 195.93 :=
by
  sorry

end add_multiply_round_l122_122928


namespace trigonometric_sum_l122_122130

theorem trigonometric_sum (θ : ℝ) (h_tan_θ : Real.tan θ = 5 / 12) (h_range : π ≤ θ ∧ θ ≤ 3 * π / 2) : 
  Real.cos θ + Real.sin θ = -17 / 13 :=
by
  sorry

end trigonometric_sum_l122_122130


namespace coronavirus_case_ratio_l122_122621

theorem coronavirus_case_ratio (n_first_wave_cases : ℕ) (total_second_wave_cases : ℕ) (n_days : ℕ) 
  (h1 : n_first_wave_cases = 300) (h2 : total_second_wave_cases = 21000) (h3 : n_days = 14) :
  (total_second_wave_cases / n_days) / n_first_wave_cases = 5 :=
by sorry

end coronavirus_case_ratio_l122_122621


namespace monthly_cost_per_iguana_l122_122643

theorem monthly_cost_per_iguana
  (gecko_cost snake_cost annual_cost : ℕ)
  (monthly_cost_per_iguana : ℕ)
  (gecko_count iguana_count snake_count : ℕ)
  (annual_cost_eq : annual_cost = 1140)
  (gecko_count_eq : gecko_count = 3)
  (iguana_count_eq : iguana_count = 2)
  (snake_count_eq : snake_count = 4)
  (gecko_cost_eq : gecko_cost = 15)
  (snake_cost_eq : snake_cost = 10)
  (total_annual_cost_eq : gecko_count * gecko_cost + iguana_count * monthly_cost_per_iguana * 12 + snake_count * snake_cost * 12 = annual_cost) :
  monthly_cost_per_iguana = 5 :=
by
  sorry

end monthly_cost_per_iguana_l122_122643


namespace race_distance_l122_122988

theorem race_distance {d x y z : ℝ} :
  (d / x = (d - 25) / y) →
  (d / y = (d - 15) / z) →
  (d / x = (d - 37) / z) →
  d = 125 :=
by
  intros h1 h2 h3
  -- Insert proof here
  sorry

end race_distance_l122_122988


namespace range_log_div_pow3_div3_l122_122043

noncomputable def log_div (x y : ℝ) : ℝ := Real.log (x / y)
noncomputable def log_div_pow3 (x y : ℝ) : ℝ := Real.log (x^3 / y^(1/2))
noncomputable def log_div_pow3_div3 (x y : ℝ) : ℝ := Real.log (x^3 / (3 * y))

theorem range_log_div_pow3_div3 
  (x y : ℝ) 
  (h1 : 1 ≤ log_div x y ∧ log_div x y ≤ 2)
  (h2 : 2 ≤ log_div_pow3 x y ∧ log_div_pow3 x y ≤ 3) 
  : Real.log (x^3 / (3 * y)) ∈ Set.Icc (26/15 : ℝ) 3 :=
sorry

end range_log_div_pow3_div3_l122_122043


namespace sharpened_off_length_l122_122705

-- Define the conditions
def original_length : ℤ := 31
def length_after_sharpening : ℤ := 14

-- Define the theorem to prove the length sharpened off is 17 inches
theorem sharpened_off_length : original_length - length_after_sharpening = 17 := sorry

end sharpened_off_length_l122_122705


namespace repeated_digit_in_mod_sequence_l122_122110

theorem repeated_digit_in_mod_sequence : 
  ∃ (x y : ℕ), x ≠ y ∧ (2^1970 % 9 = 4) ∧ 
  (∀ n : ℕ, n < 10 → n = 2^1970 % 9 → n = x ∨ n = y) :=
sorry

end repeated_digit_in_mod_sequence_l122_122110


namespace roses_cut_l122_122568

def initial_roses : ℕ := 6
def new_roses : ℕ := 16

theorem roses_cut : new_roses - initial_roses = 10 := by
  sorry

end roses_cut_l122_122568


namespace augmented_matrix_solution_l122_122830

theorem augmented_matrix_solution :
  ∀ (m n : ℝ),
  (∃ (x y : ℝ), (m * x = 6 ∧ 3 * y = n) ∧ (x = -3 ∧ y = 4)) →
  m + n = 10 :=
by
  intros m n h
  sorry

end augmented_matrix_solution_l122_122830


namespace percentage_of_same_grade_is_48_l122_122909

def students_with_same_grade (grades : ℕ × ℕ → ℕ) : ℕ :=
  grades (0, 0) + grades (1, 1) + grades (2, 2) + grades (3, 3) + grades (4, 4)

theorem percentage_of_same_grade_is_48
  (grades : ℕ × ℕ → ℕ)
  (h : grades (0, 0) = 3 ∧ grades (1, 1) = 6 ∧ grades (2, 2) = 8 ∧ grades (3, 3) = 4 ∧ grades (4, 4) = 3)
  (total_students : ℕ) (h_students : total_students = 50) :
  (students_with_same_grade grades / 50 : ℚ) * 100 = 48 :=
by
  sorry

end percentage_of_same_grade_is_48_l122_122909


namespace bags_of_white_flour_l122_122646

theorem bags_of_white_flour (total_flour wheat_flour : ℝ) (h1 : total_flour = 0.3) (h2 : wheat_flour = 0.2) : 
  total_flour - wheat_flour = 0.1 :=
by
  sorry

end bags_of_white_flour_l122_122646


namespace find_k_for_quadratic_has_one_real_root_l122_122604

theorem find_k_for_quadratic_has_one_real_root (k : ℝ) : 
  (∃ x : ℝ, (3 * x - 4) * (x + 6) = -53 + k * x) ↔ (k = 14 + 2 * Real.sqrt 87 ∨ k = 14 - 2 * Real.sqrt 87) :=
sorry

end find_k_for_quadratic_has_one_real_root_l122_122604


namespace find_digit_A_l122_122575

theorem find_digit_A (A : ℕ) (h1 : 0 ≤ A ∧ A ≤ 9) (h2 : (2 + A + 3 + A) % 9 = 0) : A = 2 :=
by
  sorry

end find_digit_A_l122_122575


namespace cars_with_both_features_l122_122118

theorem cars_with_both_features (T P_s P_w N B : ℕ)
  (hT : T = 65) 
  (hPs : P_s = 45) 
  (hPw : P_w = 25) 
  (hN : N = 12) 
  (h_equation : P_s + P_w - B + N = T) :
  B = 17 :=
by
  sorry

end cars_with_both_features_l122_122118


namespace number_of_children_is_30_l122_122122

-- Informal statements
def total_guests := 80
def men := 40
def women := men / 2
def adults := men + women
def children := total_guests - adults
def children_after_adding_10 := children + 10

-- Formal proof statement
theorem number_of_children_is_30 :
  children_after_adding_10 = 30 := by
  sorry

end number_of_children_is_30_l122_122122


namespace arithmetic_sequence_function_positive_l122_122325

def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = - (f x)

def is_monotonically_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_function_positive
  {f : ℝ → ℝ} {a : ℕ → ℝ}
  (hf_odd : is_odd f)
  (hf_mono : is_monotonically_increasing f)
  (ha_arith : is_arithmetic_sequence a)
  (ha3_pos : a 3 > 0) : 
  f (a 1) + f (a 3) + f (a 5) > 0 := 
sorry

end arithmetic_sequence_function_positive_l122_122325


namespace initial_number_is_correct_l122_122236

theorem initial_number_is_correct (x : ℝ) (h : 8 * x - 4 = 2.625) : x = 0.828125 :=
by
  sorry

end initial_number_is_correct_l122_122236


namespace part1_part2_l122_122069

-- Proof for part 1
theorem part1 (x : ℤ) : (x - 1 ∣ x - 3 ↔ (x = -1 ∨ x = 0 ∨ x = 2 ∨ x = 3)) :=
by sorry

-- Proof for part 2
theorem part2 (x : ℤ) : (x + 2 ∣ x^2 + 3 ↔ (x = -9 ∨ x = -3 ∨ x = -1 ∨ x = 5)) :=
by sorry

end part1_part2_l122_122069


namespace longer_subsegment_of_YZ_l122_122173

/-- In triangle XYZ with sides in the ratio 3:4:5, and side YZ being 12 cm.
    The angle bisector XW divides side YZ into segments YW and ZW.
    Prove that the length of ZW is 48/7 cm. --/
theorem longer_subsegment_of_YZ (YZ : ℝ) (hYZ : YZ = 12)
    (XY XZ : ℝ) (hRatio : XY / XZ = 3 / 4) : 
    ∃ ZW : ℝ, ZW = 48 / 7 :=
by
  -- We would provide proof here
  sorry

end longer_subsegment_of_YZ_l122_122173


namespace largest_possible_radius_tangent_circle_l122_122762

theorem largest_possible_radius_tangent_circle :
  ∃ (r : ℝ), 0 < r ∧
    (∀ x y, (x - r)^2 + (y - r)^2 = r^2 → 
    ((x = 9 ∧ y = 2) → (r = 17))) :=
by
  sorry

end largest_possible_radius_tangent_circle_l122_122762


namespace cylinder_surface_area_l122_122941

noncomputable def surface_area_of_cylinder (r l : ℝ) : ℝ :=
  2 * Real.pi * r * (r + l)

theorem cylinder_surface_area (r : ℝ) (h_radius : r = 1) (l : ℝ) (h_length : l = 2 * r) :
  surface_area_of_cylinder r l = 6 * Real.pi := by
  -- Using the given conditions and definition, we need to prove the surface area is 6π
  sorry

end cylinder_surface_area_l122_122941


namespace alice_met_tweedledee_l122_122354

noncomputable def brother_statement (day : ℕ) : Prop :=
  sorry -- Define the exact logical structure of the statement "I am lying today, and my name is Tweedledum" here

theorem alice_met_tweedledee (day : ℕ) : brother_statement day → (∃ (b : String), b = "Tweedledee") :=
by
  sorry -- provide the proof here

end alice_met_tweedledee_l122_122354


namespace largest_x_value_l122_122376

theorem largest_x_value : ∃ x : ℝ, (x / 7 + 3 / (7 * x) = 1) ∧ (∀ y : ℝ, (y / 7 + 3 / (7 * y) = 1) → y ≤ (7 + Real.sqrt 37) / 2) :=
by
  -- (Proof of the theorem is omitted for this task)
  sorry

end largest_x_value_l122_122376


namespace total_pieces_equiv_231_l122_122841

-- Define the arithmetic progression for rods.
def rods_arithmetic_sequence : ℕ → ℕ
| 0 => 0
| n + 1 => 3 * (n + 1)

-- Define the sum of the first 10 terms of the sequence.
def rods_total (n : ℕ) : ℕ :=
  let a := 3
  let d := 3
  n / 2 * (2 * a + (n - 1) * d)

def rods_count : ℕ :=
  rods_total 10

-- Define the 11th triangular number for connectors.
def triangular_number (n : ℕ) : ℕ :=
  n * (n + 1) / 2

def connectors_count : ℕ :=
  triangular_number 11

-- Define the total number of pieces.
def total_pieces : ℕ :=
  rods_count + connectors_count

-- The theorem we aim to prove.
theorem total_pieces_equiv_231 : total_pieces = 231 := by
  sorry

end total_pieces_equiv_231_l122_122841


namespace total_digits_written_total_digit_1_appearances_digit_at_position_2016_l122_122090

-- Problem 1
theorem total_digits_written : 
  let digits_1_to_9 := 9
  let digits_10_to_99 := 90 * 2
  let digits_100_to_999 := 900 * 3
  digits_1_to_9 + digits_10_to_99 + digits_100_to_999 = 2889 := 
by
  sorry

-- Problem 2
theorem total_digit_1_appearances : 
  let digit_1_as_1_digit := 1
  let digit_1_as_2_digits := 10 + 9
  let digit_1_as_3_digits := 100 + 9 * 10 + 9 * 10
  digit_1_as_1_digit + digit_1_as_2_digits + digit_1_as_3_digits = 300 := 
by
  sorry

-- Problem 3
theorem digit_at_position_2016 : 
  let position_1_to_99 := 9 + 90 * 2
  let remaining_positions := 2016 - position_1_to_99
  let three_digit_positions := remaining_positions / 3
  let specific_number := 100 + three_digit_positions - 1
  specific_number % 10 = 8 := 
by
  sorry

end total_digits_written_total_digit_1_appearances_digit_at_position_2016_l122_122090


namespace meaningful_expression_range_l122_122631

theorem meaningful_expression_range (x : ℝ) :
  (x + 2 ≥ 0) ∧ (x - 1 ≠ 0) ↔ (x ≥ -2) ∧ (x ≠ 1) :=
by
  sorry

end meaningful_expression_range_l122_122631


namespace parabola_sum_l122_122253

variables (a b c x y : ℝ)

noncomputable def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem parabola_sum (h1 : ∀ x, quadratic a b c x = -(x - 3)^2 + 4)
    (h2 : quadratic a b c 1 = 0)
    (h3 : quadratic a b c 5 = 0) :
    a + b + c = 0 :=
by
  -- We assume quadratic(a, b, c, x) = a * x^2 + b * x + c
  -- We assume quadratic(a, b, c, 1) = 0 and quadratic(a, b, c, 5) = 0
  -- We need to prove a + b + c = 0
  sorry

end parabola_sum_l122_122253


namespace infinite_series_sum_l122_122109

theorem infinite_series_sum
  (a b : ℝ)
  (h1 : (∑' n : ℕ, a / (b ^ (n + 1))) = 4) :
  (∑' n : ℕ, a / ((a + b) ^ (n + 1))) = 4 / 5 := 
sorry

end infinite_series_sum_l122_122109


namespace not_hexagonal_pyramid_l122_122333

-- Definition of the pyramid with slant height, base radius, and height
structure Pyramid where
  r : ℝ  -- Side length of the base equilateral triangle
  h : ℝ  -- Height of the pyramid
  l : ℝ  -- Slant height (lateral edge)
  hypo : h^2 + (r / 2)^2 = l^2

-- The theorem to prove a pyramid with all edges equal cannot be hexagonal
theorem not_hexagonal_pyramid (p : Pyramid) : p.l ≠ p.r :=
sorry

end not_hexagonal_pyramid_l122_122333


namespace solution_to_quadratic_solution_to_cubic_l122_122280

-- Problem 1: x^2 = 4
theorem solution_to_quadratic (x : ℝ) : x^2 = 4 -> x = 2 ∨ x = -2 := by
  sorry

-- Problem 2: 64x^3 + 27 = 0
theorem solution_to_cubic (x : ℝ) : 64 * x^3 + 27 = 0 -> x = -3 / 4 := by
  sorry

end solution_to_quadratic_solution_to_cubic_l122_122280


namespace h_eq_20_at_y_eq_4_l122_122261

noncomputable def k (y : ℝ) : ℝ := 40 / (y + 5)

noncomputable def h (y : ℝ) : ℝ := 4 * (k⁻¹ y)

theorem h_eq_20_at_y_eq_4 : h 4 = 20 := 
by 
  -- Insert proof here
  sorry

end h_eq_20_at_y_eq_4_l122_122261


namespace number_of_smaller_cubes_l122_122009

theorem number_of_smaller_cubes (edge : ℕ) (N : ℕ) (h_edge : edge = 5)
  (h_divisors : ∃ (a b c : ℕ), a + b + c = N ∧ a * 1^3 + b * 2^3 + c * 3^3 = edge^3 ∧ a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) :
  N = 22 :=
by
  sorry

end number_of_smaller_cubes_l122_122009


namespace solve_mt_eq_l122_122635

theorem solve_mt_eq (m n : ℤ) (hm : m ≠ 0) (hn : n ≠ 0) :
  (m^2 + n) * (m + n^2) = (m - n)^3 →
  (m = -1 ∧ n = -1) ∨ (m = 8 ∧ n = -10) ∨ (m = 9 ∧ n = -6) ∨ (m = 9 ∧ n = -21) :=
by
  sorry

end solve_mt_eq_l122_122635


namespace percentage_of_loss_is_10_l122_122573

-- Definitions based on conditions
def cost_price : ℝ := 1800
def selling_price : ℝ := 1620
def loss : ℝ := cost_price - selling_price

-- The goal: prove the percentage of loss equals 10%
theorem percentage_of_loss_is_10 :
  (loss / cost_price) * 100 = 10 := by
  sorry

end percentage_of_loss_is_10_l122_122573


namespace sum_ac_equals_seven_l122_122446

theorem sum_ac_equals_seven 
  (a b c d : ℝ)
  (h1 : ab + bc + cd + da = 42)
  (h2 : b + d = 6) :
  a + c = 7 := 
sorry

end sum_ac_equals_seven_l122_122446


namespace geometric_sequence_b_value_l122_122180

theorem geometric_sequence_b_value (b : ℝ) (h1 : 25 * b = b^2) (h2 : b * (1 / 4) = b / 4) :
  b = 5 / 2 :=
sorry

end geometric_sequence_b_value_l122_122180


namespace range_of_a_l122_122119

-- Given definitions from the problem
def p (a : ℝ) : Prop :=
  (4 - 4 * a) > 0

def q (a : ℝ) : Prop :=
  (a - 3) * (a + 1) < 0

-- The theorem we want to prove
theorem range_of_a (a : ℝ) : ¬ (p a ∨ q a) ↔ a ≥ 3 := 
by sorry

end range_of_a_l122_122119


namespace min_value_exists_max_value_exists_l122_122495

noncomputable def y (x : ℝ) : ℝ := 3 - 4 * Real.sin x - 4 * (Real.cos x)^2

theorem min_value_exists :
  (∃ k : ℤ, y (π / 6 + 2 * k * π) = -2) ∧ (∃ k : ℤ, y (5 * π / 6 + 2 * k * π) = -2) :=
by 
  sorry

theorem max_value_exists :
  ∃ k : ℤ, y (-π / 2 + 2 * k * π) = 7 :=
by 
  sorry

end min_value_exists_max_value_exists_l122_122495


namespace find_X_l122_122882

theorem find_X (X : ℚ) (h : (1/3 : ℚ) * (1/4 : ℚ) * X = (1/4 : ℚ) * (1/6 : ℚ) * 120) : X = 60 := 
sorry

end find_X_l122_122882


namespace find_TS_l122_122624

-- Definitions of the conditions as given:
def PQ : ℝ := 25
def PS : ℝ := 25
def QR : ℝ := 15
def RS : ℝ := 15
def PT : ℝ := 15
def ST_parallel_QR : Prop := true  -- ST is parallel to QR (used as a given fact)

-- Main statement in Lean:
theorem find_TS (h1 : PQ = 25) (h2 : PS = 25) (h3 : QR = 15) (h4 : RS = 15) (h5 : PT = 15)
               (h6 : ST_parallel_QR) : TS = 24 :=
by
  sorry

end find_TS_l122_122624


namespace initial_jelly_beans_l122_122137

theorem initial_jelly_beans (total_children : ℕ) (percentage : ℕ) (jelly_per_child : ℕ) (remaining_jelly : ℕ) :
  (percentage = 80) → (total_children = 40) → (jelly_per_child = 2) → (remaining_jelly = 36) →
  (total_children * percentage / 100 * jelly_per_child + remaining_jelly = 100) :=
by
  intros h1 h2 h3 h4
  sorry

end initial_jelly_beans_l122_122137


namespace sum_of_terms_arithmetic_sequence_l122_122692

variable {S : ℕ → ℕ}
variable {k : ℕ}

-- Given conditions
axiom S_k : S k = 2
axiom S_3k : S (3 * k) = 18

-- The statement to prove
theorem sum_of_terms_arithmetic_sequence : S (4 * k) = 32 := by
  sorry

end sum_of_terms_arithmetic_sequence_l122_122692


namespace total_cats_l122_122942

def initial_siamese_cats : Float := 13.0
def initial_house_cats : Float := 5.0
def added_cats : Float := 10.0

theorem total_cats : initial_siamese_cats + initial_house_cats + added_cats = 28.0 := by
  sorry

end total_cats_l122_122942


namespace square_of_cube_of_third_smallest_prime_l122_122584

-- Definition of the third smallest prime number
def third_smallest_prime : Nat := 5

-- Definition of the cube of a number
def cube (n : Nat) : Nat := n ^ 3

-- Definition of the square of a number
def square (n : Nat) : Nat := n ^ 2

-- Theorem stating that the square of the cube of the third smallest prime number is 15625
theorem square_of_cube_of_third_smallest_prime : 
  square (cube third_smallest_prime) = 15625 := by 
  sorry

end square_of_cube_of_third_smallest_prime_l122_122584


namespace power_function_solution_l122_122685

theorem power_function_solution (m : ℝ) 
    (h1 : m^2 - 3 * m + 3 = 1) 
    (h2 : m - 1 ≠ 0) : m = 2 := 
by
  sorry

end power_function_solution_l122_122685


namespace croissants_for_breakfast_l122_122713

def total_items (C : ℕ) : Prop :=
  C + 18 + 30 = 110

theorem croissants_for_breakfast (C : ℕ) (h : total_items C) : C = 62 :=
by {
  -- The proof might be here, but since it's not required:
  sorry
}

end croissants_for_breakfast_l122_122713


namespace smaller_solution_of_quadratic_eq_l122_122926

theorem smaller_solution_of_quadratic_eq : 
  (exists x y : ℝ, x < y ∧ x^2 - 13 * x + 36 = 0 ∧ y^2 - 13 * y + 36 = 0 ∧ x = 4) :=
by sorry

end smaller_solution_of_quadratic_eq_l122_122926


namespace ryan_hours_difference_l122_122940

theorem ryan_hours_difference :
  let hours_english := 6
  let hours_chinese := 7
  hours_chinese - hours_english = 1 := 
by
  -- this is where the proof steps would go
  sorry

end ryan_hours_difference_l122_122940


namespace vector_perpendicular_to_plane_l122_122939

theorem vector_perpendicular_to_plane
  (a b c d : ℝ)
  (x1 y1 z1 x2 y2 z2 : ℝ)
  (h1 : a * x1 + b * y1 + c * z1 + d = 0)
  (h2 : a * x2 + b * y2 + c * z2 + d = 0) :
  a * (x1 - x2) + b * (y1 - y2) + c * (z1 - z2) = 0 :=
sorry

end vector_perpendicular_to_plane_l122_122939


namespace votes_for_eliot_l122_122507

theorem votes_for_eliot (randy_votes : ℕ) (shaun_votes : ℕ) (eliot_votes : ℕ)
  (h_randy : randy_votes = 16)
  (h_shaun : shaun_votes = 5 * randy_votes)
  (h_eliot : eliot_votes = 2 * shaun_votes) :
  eliot_votes = 160 :=
by
  sorry

end votes_for_eliot_l122_122507


namespace probability_of_3_black_face_cards_l122_122675

-- Definitions based on conditions
def total_cards : ℕ := 36
def total_black_face_cards : ℕ := 8
def total_other_cards : ℕ := total_cards - total_black_face_cards
def draw_cards : ℕ := 6
def draw_black_face_cards : ℕ := 3
def draw_other_cards := draw_cards - draw_black_face_cards

-- Calculation using combinations
noncomputable def combination (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

noncomputable def total_combinations : ℕ := combination total_cards draw_cards
noncomputable def favorable_combinations : ℕ := combination total_black_face_cards draw_black_face_cards * combination total_other_cards draw_other_cards

-- Calculating probability
noncomputable def probability : ℚ := favorable_combinations / total_combinations

-- The theorem to be proved
theorem probability_of_3_black_face_cards : probability = 11466 / 121737 := by
  -- proof
  sorry

end probability_of_3_black_face_cards_l122_122675


namespace negation_example_l122_122733

theorem negation_example : ¬(∀ x : ℝ, x > 1 → x^2 > 1) ↔ ∃ x : ℝ, x > 1 ∧ x^2 ≤ 1 := by
  sorry

end negation_example_l122_122733


namespace find_set_C_l122_122967

def A : Set ℝ := {x | x^2 - 3 * x + 2 = 0}
def B (a : ℝ) : Set ℝ := {x | a * x - 2 = 0}
def C : Set ℝ := {a | B a ⊆ A}

theorem find_set_C : C = {0, 1, 2} :=
by
  sorry

end find_set_C_l122_122967


namespace group_left_to_clean_is_third_group_l122_122437

-- Definition of group sizes
def group1 := 7
def group2 := 10
def group3 := 16
def group4 := 18

-- Definitions and conditions
def total_students := group1 + group2 + group3 + group4
def lecture_factor := 4
def english_students := 7  -- From solution: must be 7 students attending the English lecture
def math_students := lecture_factor * english_students

-- Hypothesis of the students allocating to lectures
def students_attending_lectures := english_students + math_students
def students_left_to_clean := total_students - students_attending_lectures

-- The statement to be proved in Lean
theorem group_left_to_clean_is_third_group
  (h : students_left_to_clean = group3) :
  students_left_to_clean = 16 :=
sorry

end group_left_to_clean_is_third_group_l122_122437


namespace f_strictly_increasing_on_l122_122550

-- Define the function
def f (x : ℝ) : ℝ := x^2 * (2 - x)

-- Define the derivative of the function
def f' (x : ℝ) : ℝ := -3 * x^2 + 4 * x

-- Define the property that the function is strictly increasing on an interval
def strictly_increasing_on (a b : ℝ) (f : ℝ → ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f x < f y

-- State the theorem
theorem f_strictly_increasing_on : strictly_increasing_on 0 (4/3) f :=
sorry

end f_strictly_increasing_on_l122_122550


namespace find_a4_and_s5_l122_122556

def geometric_sequence (a : ℕ → ℚ) (q : ℚ) : Prop :=
  ∀ n, a (n + 1) = a n * q

variable (a : ℕ → ℚ) (q : ℚ)

axiom condition_1 : a 1 + a 3 = 10
axiom condition_2 : a 4 + a 6 = 1 / 4

theorem find_a4_and_s5 (h_geom : geometric_sequence a q) :
  a 4 = 1 ∧ (a 1 * (1 - q^5) / (1 - q)) = 31 / 2 :=
by
  sorry

end find_a4_and_s5_l122_122556


namespace maximum_distance_l122_122586

noncomputable def point_distance (x1 y1 x2 y2 : ℝ) : ℝ := 
  Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

def square_side_length := 2

def distance_condition (u v w : ℝ) : Prop := 
  u^2 + v^2 = 2 * w^2

theorem maximum_distance 
  (x y : ℝ) 
  (h1 : point_distance x y 0 0 = u) 
  (h2 : point_distance x y 2 0 = v) 
  (h3 : point_distance x y 2 2 = w)
  (h4 : distance_condition u v w) :
  ∃ (d : ℝ), d = point_distance x y 0 2 ∧ d = 2 * Real.sqrt 5 := sorry

end maximum_distance_l122_122586


namespace line_equation_l122_122732

theorem line_equation (a : ℝ) (P : ℝ × ℝ) (hx : P = (5, 6)) 
                      (cond : (a ≠ 0) ∧ (2 * a = 17)) : 
  ∃ (m b : ℝ), - (m * (0 : ℝ) + b) = a ∧ (- m * 17 / 2 + b) = 6 ∧ 
               (x + 2 * y - 17 =  0) := sorry

end line_equation_l122_122732


namespace positive_difference_l122_122445

theorem positive_difference (x y : ℝ) (h1 : x + y = 40) (h2 : 3 * y - 4 * x = 10) :
  |y - x| = 60 / 7 := 
sorry

end positive_difference_l122_122445


namespace total_cars_l122_122307

-- Conditions
def initial_cars : ℕ := 150
def uncle_cars : ℕ := 5
def grandpa_cars : ℕ := 2 * uncle_cars
def dad_cars : ℕ := 10
def mum_cars : ℕ := dad_cars + 5
def auntie_cars : ℕ := 6

-- Proof statement (theorem)
theorem total_cars : initial_cars + (grandpa_cars + dad_cars + mum_cars + auntie_cars + uncle_cars) = 196 :=
by
  sorry

end total_cars_l122_122307


namespace part_one_part_two_l122_122140

noncomputable def f (x a: ℝ) : ℝ := abs (x - 1) + abs (x + a)
noncomputable def g (a : ℝ) : ℝ := a^2 - a - 2

theorem part_one (x : ℝ) : f x 3 > g 3 + 2 ↔ x < -4 ∨ x > 2 := by
  sorry

theorem part_two (a : ℝ) :
  (∀ x : ℝ, -a ≤ x ∧ x ≤ 1 → f x a ≤ g a) ↔ a ≥ 3 := by
  sorry

end part_one_part_two_l122_122140


namespace jordan_rectangle_width_l122_122256

theorem jordan_rectangle_width
  (carol_length : ℕ) (carol_width : ℕ) (jordan_length : ℕ) (jordan_width : ℕ)
  (h_carol_dims : carol_length = 12) (h_carol_dims2 : carol_width = 15)
  (h_jordan_length : jordan_length = 6)
  (h_area_eq : carol_length * carol_width = jordan_length * jordan_width) :
  jordan_width = 30 := 
sorry

end jordan_rectangle_width_l122_122256


namespace probability_sum_odd_correct_l122_122674

noncomputable def probability_sum_odd : ℚ :=
  let total_ways := 10
  let ways_sum_odd := 6
  ways_sum_odd / total_ways

theorem probability_sum_odd_correct :
  probability_sum_odd = 3 / 5 :=
by
  unfold probability_sum_odd
  rfl

end probability_sum_odd_correct_l122_122674


namespace people_at_first_table_l122_122492

theorem people_at_first_table (N x : ℕ) 
  (h1 : 20 < N) 
  (h2 : N < 50)
  (h3 : (N - x) % 42 = 0)
  (h4 : N % 8 = 7) : 
  x = 5 :=
sorry

end people_at_first_table_l122_122492


namespace div_power_n_minus_one_l122_122598

theorem div_power_n_minus_one (n : ℕ) (hn : n > 0) (h : n ∣ (2^n - 1)) : n = 1 := by
  sorry

end div_power_n_minus_one_l122_122598


namespace solve_equation_l122_122131

theorem solve_equation :
  ∀ y : ℤ, 4 * (y - 1) = 1 - 3 * (y - 3) → y = 2 :=
by
  intros y h
  sorry

end solve_equation_l122_122131


namespace value_of_a_l122_122380

theorem value_of_a (a : ℝ) (h1 : a < 0) (h2 : |a| = 3) : a = -3 := 
by
  sorry

end value_of_a_l122_122380


namespace greatest_x_lcm_l122_122216

theorem greatest_x_lcm (x : ℕ) (h1 : Nat.lcm x 15 = Nat.lcm 90 15) (h2 : Nat.lcm x 18 = Nat.lcm 90 18) : x = 90 := 
sorry

end greatest_x_lcm_l122_122216


namespace sum_of_maximum_and_minimum_of_u_l122_122704

theorem sum_of_maximum_and_minimum_of_u :
  ∀ (x y z : ℝ),
    0 ≤ x → 0 ≤ y → 0 ≤ z →
    3 * x + 2 * y + z = 5 →
    2 * x + y - 3 * z = 1 →
    3 * x + y - 7 * z = 3 * z - 2 →
    (-5 : ℝ) / 7 + (-1 : ℝ) / 11 = -62 / 77 :=
by
  sorry

end sum_of_maximum_and_minimum_of_u_l122_122704


namespace y_range_l122_122097

theorem y_range (x y : ℝ) (h1 : 4 * x + y = 1) (h2 : -1 < x) (h3 : x ≤ 2) : -7 ≤ y ∧ y < -3 := 
by
  sorry

end y_range_l122_122097


namespace sin_pi_over_4_plus_alpha_l122_122962

open Real

theorem sin_pi_over_4_plus_alpha
  (α : ℝ)
  (hα : 0 < α ∧ α < π)
  (h_tan : tan (α - π / 4) = 1 / 3) :
  sin (π / 4 + α) = 3 * sqrt 10 / 10 :=
sorry

end sin_pi_over_4_plus_alpha_l122_122962


namespace packs_of_yellow_bouncy_balls_l122_122505

/-- Maggie bought 4 packs of red bouncy balls, some packs of yellow bouncy balls (denoted as Y), and 4 packs of green bouncy balls. -/
theorem packs_of_yellow_bouncy_balls (Y : ℕ) : 
  (4 + Y + 4) * 10 = 160 -> Y = 8 := 
by 
  sorry

end packs_of_yellow_bouncy_balls_l122_122505


namespace sum_of_angles_l122_122999

theorem sum_of_angles (θ₁ θ₂ θ₃ θ₄ : ℝ)
  (h₁ : θ₁ = 67.5) (h₂ : θ₂ = 157.5) (h₃ : θ₃ = 247.5) (h₄ : θ₄ = 337.5) :
  θ₁ + θ₂ + θ₃ + θ₄ = 810 :=
by
  -- These parameters are used only to align with provided conditions
  let r₁ := 1
  let r₂ := r₁
  let r₃ := r₁
  let r₄ := r₁
  have z₁ := r₁ * (Complex.cos θ₁ + Complex.sin θ₁ * Complex.I)
  have z₂ := r₂ * (Complex.cos θ₂ + Complex.sin θ₂ * Complex.I)
  have z₃ := r₃ * (Complex.cos θ₃ + Complex.sin θ₃ * Complex.I)
  have z₄ := r₄ * (Complex.cos θ₄ + Complex.sin θ₄ * Complex.I)
  sorry

end sum_of_angles_l122_122999


namespace election_percentage_l122_122609

-- Define the total number of votes (V), winner's votes, and the vote difference
def total_votes (V : ℕ) : Prop := V = 1944 + (1944 - 288)

-- Define the percentage calculation from the problem
def percentage_of_votes (votes_received total_votes : ℕ) : ℕ := (votes_received * 100) / total_votes

-- State the core theorem to prove the winner received 54 percent of the total votes
theorem election_percentage (V : ℕ) (h : total_votes V) : percentage_of_votes 1944 V = 54 := by
  sorry

end election_percentage_l122_122609


namespace coeff_sum_eq_neg_two_l122_122127

theorem coeff_sum_eq_neg_two (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℝ) :
  (∀ x : ℝ, (x^10 + x^4 + 1) = a + a₁ * (x+1) + a₂ * (x+1)^2 + a₃ * (x+1)^3 + a₄ * (x+1)^4 
   + a₅ * (x+1)^5 + a₆ * (x+1)^6 + a₇ * (x+1)^7 + a₈ * (x+1)^8 + a₉ * (x+1)^9 + a₁₀ * (x+1)^10) 
  → (a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ + a₁₀ = -2) := 
by sorry

end coeff_sum_eq_neg_two_l122_122127


namespace determine_number_of_solutions_l122_122136

noncomputable def num_solutions_eq : Prop :=
  let f (x : ℝ) := (3 * x ^ 2 - 15 * x) / (x ^ 2 - 7 * x + 10)
  let g (x : ℝ) := x - 4
  ∃ S : Finset ℝ, 
    (∀ x ∈ S, (x ≠ 2 ∧ x ≠ 5) ∧ f x = g x) ∧
    S.card = 2

theorem determine_number_of_solutions : num_solutions_eq :=
  by
  sorry

end determine_number_of_solutions_l122_122136


namespace diff_between_largest_and_smallest_fraction_l122_122488

theorem diff_between_largest_and_smallest_fraction : 
  let f1 := (3 : ℚ) / 4
  let f2 := (7 : ℚ) / 8
  let f3 := (13 : ℚ) / 16
  let f4 := (1 : ℚ) / 2
  let largest := max f1 (max f2 (max f3 f4))
  let smallest := min f1 (min f2 (min f3 f4))
  largest - smallest = (3 : ℚ) / 8 :=
by
  sorry

end diff_between_largest_and_smallest_fraction_l122_122488


namespace Jovana_shells_l122_122616

theorem Jovana_shells (initial_shells : ℕ) (added_shells : ℕ) (total_shells : ℕ) :
  initial_shells = 5 → added_shells = 12 → total_shells = initial_shells + added_shells → total_shells = 17 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end Jovana_shells_l122_122616


namespace value_of_b_l122_122553

noncomputable def problem (a1 a2 a3 a4 a5 : ℤ) (b : ℤ) :=
  (a1 ≠ a2) ∧ (a1 ≠ a3) ∧ (a1 ≠ a4) ∧ (a1 ≠ a5) ∧
  (a2 ≠ a3) ∧ (a2 ≠ a4) ∧ (a2 ≠ a5) ∧
  (a3 ≠ a4) ∧ (a3 ≠ a5) ∧
  (a4 ≠ a5) ∧
  (a1 + a2 + a3 + a4 + a5 = 9) ∧
  ((b - a1) * (b - a2) * (b - a3) * (b - a4) * (b - a5) = 2009) ∧
  (∃ b : ℤ, b = 10)

theorem value_of_b (a1 a2 a3 a4 a5 : ℤ) (b : ℤ) :
  problem a1 a2 a3 a4 a5 b → b = 10 :=
  sorry

end value_of_b_l122_122553


namespace initial_girls_l122_122152

theorem initial_girls (G : ℕ) (h : G + 682 = 1414) : G = 732 := 
by
  sorry

end initial_girls_l122_122152


namespace fred_limes_l122_122272

theorem fred_limes (limes_total : ℕ) (alyssa_limes : ℕ) (nancy_limes : ℕ) (fred_limes : ℕ)
  (h_total : limes_total = 103)
  (h_alyssa : alyssa_limes = 32)
  (h_nancy : nancy_limes = 35)
  (h_fred : fred_limes = limes_total - (alyssa_limes + nancy_limes)) :
  fred_limes = 36 :=
by
  sorry

end fred_limes_l122_122272


namespace problem_statement_l122_122526

variable (a b c d : ℝ)

-- Definitions for the conditions
def condition1 := a + b + c + d = 100
def condition2 := (a / (b + c + d)) + (b / (a + c + d)) + (c / (a + b + d)) + (d / (a + b + c)) = 95

-- The theorem which needs to be proved
theorem problem_statement (h1 : condition1 a b c d) (h2 : condition2 a b c d) :
  (1 / (b + c + d)) + (1 / (a + c + d)) + (1 / (a + b + d)) + (1 / (a + b + c)) = 99 / 100 := by
  sorry

end problem_statement_l122_122526


namespace calculate_f_ff_f60_l122_122006

def f (N : ℝ) : ℝ := 0.3 * N + 2

theorem calculate_f_ff_f60 : f (f (f 60)) = 4.4 := by
  sorry

end calculate_f_ff_f60_l122_122006


namespace simplify_expression_l122_122293

theorem simplify_expression :
  ((5 * 10^7) / (2 * 10^2)) + (4 * 10^5) = 650000 := 
by
  sorry

end simplify_expression_l122_122293


namespace normal_price_of_article_l122_122426

theorem normal_price_of_article 
  (final_price : ℝ)
  (discount1 : ℝ) 
  (discount2 : ℝ) 
  (P : ℝ)
  (h : final_price = 108) 
  (h1 : discount1 = 0.10) 
  (h2 : discount2 = 0.20)
  (h_eq : (1 - discount1) * (1 - discount2) * P = final_price) :
  P = 150 := by
  sorry

end normal_price_of_article_l122_122426


namespace jason_books_is_21_l122_122776

def keith_books : ℕ := 20
def total_books : ℕ := 41

theorem jason_books_is_21 (jason_books : ℕ) : 
  jason_books + keith_books = total_books → 
  jason_books = 21 := 
by 
  intro h
  sorry

end jason_books_is_21_l122_122776


namespace find_p_l122_122744

theorem find_p (p q : ℚ) (h1 : 5 * p + 6 * q = 10) (h2 : 6 * p + 5 * q = 17) : p = 52 / 11 :=
by
  sorry

end find_p_l122_122744


namespace points_description_l122_122322

noncomputable def clubsuit (a b : ℝ) : ℝ := a^3 * b - a * b^3

theorem points_description (x y : ℝ) : 
  (clubsuit x y = clubsuit y x) ↔ (x = 0) ∨ (y = 0) ∨ (x = y) ∨ (x + y = 0) := 
by 
  sorry

end points_description_l122_122322


namespace find_expression_value_l122_122413

theorem find_expression_value (m: ℝ) (h: m^2 - 2 * m - 1 = 0) : 
  (m - 1)^2 - (m - 3) * (m + 3) - (m - 1) * (m - 3) = 6 := 
by 
  sorry

end find_expression_value_l122_122413


namespace real_part_of_one_over_one_minus_z_l122_122212

open Complex

noncomputable def real_part_fraction {z : ℂ} (hz1 : norm z = 1) (hz2 : ¬(z.im = 0)) : ℝ :=
  re (1 / (1 - z))

theorem real_part_of_one_over_one_minus_z (z : ℂ) (hz1 : norm z = 1) (hz2 : ¬(z.im = 0)) :
  real_part_fraction hz1 hz2 = 1 / 2 :=
by
  sorry

end real_part_of_one_over_one_minus_z_l122_122212


namespace sqrt_three_irrational_l122_122608

-- Define what it means for a number to be rational
def is_rational (x : ℝ) : Prop := ∃ (p q : ℤ), q ≠ 0 ∧ x = p / q

-- Define what it means for a number to be irrational
def is_irrational (x : ℝ) : Prop := ¬ is_rational x

-- State that sqrt(3) is irrational
theorem sqrt_three_irrational : is_irrational (Real.sqrt 3) :=
sorry

end sqrt_three_irrational_l122_122608


namespace trajectory_of_midpoint_l122_122065

theorem trajectory_of_midpoint (Q : ℝ × ℝ) (P : ℝ × ℝ) (N : ℝ × ℝ)
  (h1 : Q.1^2 - Q.2^2 = 1)
  (h2 : N = (2 * P.1 - Q.1, 2 * P.2 - Q.2))
  (h3 : N.1 + N.2 = 2)
  (h4 : (P.2 - Q.2) / (P.1 - Q.1) = 1) :
  2 * P.1^2 - 2 * P.2^2 - 2 * P.1 + 2 * P.2 - 1 = 0 :=
  sorry

end trajectory_of_midpoint_l122_122065


namespace range_of_a_l122_122419

noncomputable def f : ℝ → ℝ := sorry

def is_even (f : ℝ → ℝ) := ∀ x : ℝ, f x = f (-x)
def is_monotone_on_nonneg (f : ℝ → ℝ) := ∀ ⦃x y : ℝ⦄, 0 ≤ x → 0 ≤ y → x < y → f x < f y

axiom even_f : is_even f
axiom monotone_f : is_monotone_on_nonneg f

theorem range_of_a (a : ℝ) (h : f a ≥ f 3) : a ≤ -3 ∨ a ≥ 3 :=
by
  sorry

end range_of_a_l122_122419


namespace keith_turnips_l122_122833

theorem keith_turnips (Alyssa_turnips Keith_turnips : ℕ) 
  (total_turnips : Alyssa_turnips + Keith_turnips = 15) 
  (alyssa_grew : Alyssa_turnips = 9) : Keith_turnips = 6 :=
by
  sorry

end keith_turnips_l122_122833


namespace negation_of_proposition_l122_122375

theorem negation_of_proposition (a b : ℝ) : ¬ (a > b ∧ a - 1 > b - 1) ↔ a ≤ b ∨ a - 1 ≤ b - 1 :=
by sorry

end negation_of_proposition_l122_122375


namespace bus_trip_distance_l122_122487

theorem bus_trip_distance
  (D S : ℕ) (H1 : S = 55)
  (H2 : D / S - 1 = D / (S + 5))
  : D = 660 :=
sorry

end bus_trip_distance_l122_122487


namespace mary_travel_time_l122_122383

noncomputable def ambulance_speed : ℝ := 60
noncomputable def don_speed : ℝ := 30
noncomputable def don_time : ℝ := 0.5

theorem mary_travel_time : (don_speed * don_time) / ambulance_speed * 60 = 15 := by
  sorry

end mary_travel_time_l122_122383


namespace gina_college_expenses_l122_122134

theorem gina_college_expenses
  (credits : ℕ)
  (cost_per_credit : ℕ)
  (num_textbooks : ℕ)
  (cost_per_textbook : ℕ)
  (facilities_fee : ℕ)
  (H_credits : credits = 14)
  (H_cost_per_credit : cost_per_credit = 450)
  (H_num_textbooks : num_textbooks = 5)
  (H_cost_per_textbook : cost_per_textbook = 120)
  (H_facilities_fee : facilities_fee = 200)
  : (credits * cost_per_credit) + (num_textbooks * cost_per_textbook) + facilities_fee = 7100 := by
  sorry

end gina_college_expenses_l122_122134


namespace is_triangle_inequality_set_B_valid_triangle_set_A_not_triangle_set_C_not_triangle_set_D_not_triangle_l122_122719

theorem is_triangle_inequality (a b c: ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem set_B_valid_triangle :
  is_triangle_inequality 5 5 6 := by
  sorry

theorem set_A_not_triangle :
  ¬ is_triangle_inequality 7 4 2 := by
  sorry

theorem set_C_not_triangle :
  ¬ is_triangle_inequality 3 4 8 := by
  sorry

theorem set_D_not_triangle :
  ¬ is_triangle_inequality 2 3 5 := by
  sorry

end is_triangle_inequality_set_B_valid_triangle_set_A_not_triangle_set_C_not_triangle_set_D_not_triangle_l122_122719


namespace unsold_books_l122_122121

-- Definitions from conditions
def books_total : ℕ := 150
def books_sold : ℕ := (2 / 3) * books_total
def book_price : ℕ := 5
def total_received : ℕ := 500

-- Proof statement
theorem unsold_books :
  (books_sold * book_price = total_received) →
  (books_total - books_sold = 50) :=
by
  sorry

end unsold_books_l122_122121


namespace number_of_satisfying_ns_l122_122442

noncomputable def a_n (n : ℕ) : ℕ := (n-1)*(2*n-1)

def b_n (n : ℕ) : ℕ := 2^n * n

def condition (n : ℕ) : Prop := b_n n ≤ 2019 * a_n n

theorem number_of_satisfying_ns : 
  ∃ n : ℕ, n = 14 ∧ ∀ k : ℕ, (1 ≤ k ∧ k ≤ 14) → condition k := 
by
  sorry

end number_of_satisfying_ns_l122_122442


namespace arithmetic_sequence_sum_l122_122123

theorem arithmetic_sequence_sum (a : ℕ → ℤ) (S : ℕ → ℤ)
  (hS : ∀ n, S n = n * (a 1 + a n) / 2)
  (h : a 3 = 20 - a 6) : S 8 = 80 :=
sorry

end arithmetic_sequence_sum_l122_122123


namespace determine_fake_coin_weight_l122_122541

theorem determine_fake_coin_weight
  (coins : Fin 25 → ℤ) 
  (fake_coin : Fin 25) 
  (all_same_weight : ∀ (i j : Fin 25), i ≠ fake_coin → j ≠ fake_coin → coins i = coins j)
  (fake_diff_weight : ∃ (x : Fin 25), (coins x ≠ coins fake_coin)) :
  ∃ (is_heavy : Bool), 
    (is_heavy = true ↔ coins fake_coin > coins (Fin.ofNat 0)) ∨ 
    (is_heavy = false ↔ coins fake_coin < coins (Fin.ofNat 0)) :=
  sorry

end determine_fake_coin_weight_l122_122541


namespace percentage_of_students_70_79_l122_122764

-- Defining basic conditions
def students_in_range_90_100 := 5
def students_in_range_80_89 := 9
def students_in_range_70_79 := 7
def students_in_range_60_69 := 4
def students_below_60 := 3

-- Total number of students
def total_students := students_in_range_90_100 + students_in_range_80_89 + students_in_range_70_79 + students_in_range_60_69 + students_below_60

-- Percentage of students in the 70%-79% range
def percent_students_70_79 := (students_in_range_70_79 / total_students) * 100

theorem percentage_of_students_70_79 : percent_students_70_79 = 25 := by
  sorry

end percentage_of_students_70_79_l122_122764


namespace volume_in_cubic_yards_l122_122245

-- Definition: A box with a specific volume in cubic feet.
def volume_in_cubic_feet (v : ℝ) : Prop :=
  v = 200

-- Definition: Conversion factor from cubic feet to cubic yards.
def cubic_feet_per_cubic_yard : ℝ := 27

-- Theorem: The volume of the box in cubic yards given the volume in cubic feet.
theorem volume_in_cubic_yards (v_cubic_feet : ℝ) 
    (h : volume_in_cubic_feet v_cubic_feet) : 
    v_cubic_feet / cubic_feet_per_cubic_yard = 200 / 27 :=
  by
    rw [h]
    sorry

end volume_in_cubic_yards_l122_122245


namespace incorrect_relation_when_agtb_l122_122818

theorem incorrect_relation_when_agtb (a b : ℝ) (c : ℝ) (h : a > b) : c = 0 → ¬ (a * c^2 > b * c^2) :=
by
  -- Not providing the proof here as specified in the instructions.
  sorry

end incorrect_relation_when_agtb_l122_122818


namespace product_of_numbers_l122_122779

theorem product_of_numbers (x y : ℝ) (h1 : x + y = 60) (h2 : x - y = 10) : x * y = 875 :=
sorry

end product_of_numbers_l122_122779


namespace total_students_in_class_l122_122311

theorem total_students_in_class 
  (b : ℕ)
  (boys_jelly_beans : ℕ := b * b)
  (girls_jelly_beans : ℕ := (b + 1) * (b + 1))
  (total_jelly_beans : ℕ := 432) 
  (condition : boys_jelly_beans + girls_jelly_beans = total_jelly_beans) :
  (b + b + 1 = 29) :=
sorry

end total_students_in_class_l122_122311


namespace min_a_4_l122_122017

theorem min_a_4 (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 9 * x + y = x * y) : 
  4 * x + y ≥ 25 :=
sorry

end min_a_4_l122_122017


namespace bag_of_potatoes_weight_l122_122226

variable (W : ℝ)

-- Define the condition given in the problem.
def condition : Prop := W = 12 / (W / 2)

-- Define the statement we want to prove.
theorem bag_of_potatoes_weight : condition W → W = 24 := by
  intro h
  sorry

end bag_of_potatoes_weight_l122_122226


namespace find_g7_l122_122600

namespace ProofProblem

variable (g : ℝ → ℝ)
variable (h1 : ∀ x y : ℝ, g (x + y) = g x + g y)
variable (h2 : g 6 = 8)

theorem find_g7 : g 7 = 28 / 3 := by
  sorry

end ProofProblem

end find_g7_l122_122600


namespace complement_intersection_l122_122211

def U : Set Nat := {1, 2, 3, 4, 5}
def A : Set Nat := {1, 2, 3}
def B : Set Nat := {2, 3, 4}

theorem complement_intersection :
  U \ (A ∩ B) = {1, 4, 5} := by
    sorry

end complement_intersection_l122_122211


namespace jill_spent_30_percent_on_food_l122_122599

variables (T F : ℝ)

theorem jill_spent_30_percent_on_food
  (h1 : 0.04 * T = 0.016 * T + 0.024 * T)
  (h2 : 0.40 + 0.30 + F = 1) :
  F = 0.30 :=
by 
  sorry

end jill_spent_30_percent_on_food_l122_122599


namespace monotonic_decreasing_interval_l122_122480

noncomputable def function_y (x : ℝ) : ℝ := x^3 - 3*x^2 - 9*x + 5

theorem monotonic_decreasing_interval : 
  ∀ x, -1 < x ∧ x < 3 →  (deriv function_y x < 0) :=
by
  sorry

end monotonic_decreasing_interval_l122_122480


namespace min_value_of_a_l122_122554

theorem min_value_of_a 
  {f : ℕ → ℝ} 
  (h : ∀ x : ℕ, 0 < x → f x = (x^2 + a * x + 11) / (x + 1)) 
  (ineq : ∀ x : ℕ, 0 < x → f x ≥ 3) : a ≥ -8 / 3 :=
sorry

end min_value_of_a_l122_122554


namespace solve_equation_l122_122329

theorem solve_equation :
  (∃ x : ℝ, (x^2 + 3*x + 5) / (x^2 + 5*x + 6) = x + 3) → (x = -1) :=
by
  sorry

end solve_equation_l122_122329


namespace smallest_possible_average_l122_122766

def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

def proper_digits (n : ℕ) : Prop :=
  ∀ (d : ℕ), d ∈ n.digits 10 → d = 0 ∨ d = 4 ∨ d = 8

theorem smallest_possible_average :
  ∃ n : ℕ, (n + 2) - n = 2 ∧ (sum_of_digits n + sum_of_digits (n + 2)) % 4 = 0 ∧ (∀ (d : ℕ), d ∈ n.digits 10 → d = 0 ∨ d = 4 ∨ d = 8) ∧ ∀ (d : ℕ), d ∈ (n + 2).digits 10 → d = 0 ∨ d = 4 ∨ d = 8 
  ∧ (n + (n + 2)) / 2 = 249 :=
sorry

end smallest_possible_average_l122_122766


namespace intersecting_lines_fixed_point_l122_122931

variable (p a b : ℝ)
variable (h1 : a ≠ 0)
variable (h2 : b ≠ 0)
variable (h3 : b^2 ≠ 2 * p * a)

def parabola (M : ℝ × ℝ) : Prop := M.2^2 = 2 * p * M.1

def fixed_points (A B : ℝ × ℝ) : Prop :=
  A = (a, b) ∧ B = (-a, 0)

def intersect_parabola (M1 M2 M : ℝ × ℝ) : Prop :=
  parabola p M ∧ parabola p M1 ∧ parabola p M2 ∧ M ≠ M1 ∧ M ≠ M2

theorem intersecting_lines_fixed_point (M M1 M2 : ℝ × ℝ)
  (hP : parabola p M) 
  (hA : (a, b) ≠ M) 
  (hB : (-a, 0) ≠ M) 
  (h_intersect : intersect_parabola p M1 M2 M) :
  ∃ C : ℝ × ℝ, C = (a, 2 * p * a / b) :=
sorry

end intersecting_lines_fixed_point_l122_122931


namespace cosine_evaluation_l122_122723

variable (α : ℝ)

theorem cosine_evaluation
  (h : Real.sin (Real.pi / 6 + α) = 1 / 3) :
  Real.cos (Real.pi / 3 - α) = 1 / 3 :=
sorry

end cosine_evaluation_l122_122723


namespace cube_sum_from_square_l122_122601

noncomputable def a_plus_inv_a_squared_eq_5 (a : ℝ) : Prop :=
  (a + 1/a) ^ 2 = 5

theorem cube_sum_from_square (a : ℝ) (h : a_plus_inv_a_squared_eq_5 a) :
  a^3 + (1/a)^3 = 2 * Real.sqrt 5 ∨ a^3 + (1/a)^3 = -2 * Real.sqrt 5 :=
by
  sorry

end cube_sum_from_square_l122_122601


namespace road_signs_at_first_intersection_l122_122205

theorem road_signs_at_first_intersection (x : ℕ) 
    (h1 : x + (x + x / 4) + 2 * (x + x / 4) + (2 * (x + x / 4) - 20) = 270) : 
    x = 40 := 
sorry

end road_signs_at_first_intersection_l122_122205


namespace polygonal_chain_max_length_not_exceed_200_l122_122860

-- Define the size of the board
def board_size : ℕ := 15

-- Define the concept of a polygonal chain length on a symmetric board
def polygonal_chain_length (n : ℕ) : ℕ := sorry -- length function yet to be defined

-- Define the maximum length constant to be compared with
def max_length : ℕ := 200

-- Define the theorem statement including all conditions and constraints
theorem polygonal_chain_max_length_not_exceed_200 :
  ∃ (n : ℕ), n = board_size ∧ 
             (∀ (length : ℕ),
             length = polygonal_chain_length n →
             length ≤ max_length) :=
sorry

end polygonal_chain_max_length_not_exceed_200_l122_122860


namespace xy_expr_value_l122_122855

variable (x y : ℝ)

-- Conditions
def cond1 : Prop := x - y = 2
def cond2 : Prop := x * y = 3

-- Statement to prove
theorem xy_expr_value (h1 : cond1 x y) (h2 : cond2 x y) : x * y^2 - x^2 * y = -6 :=
by
  sorry

end xy_expr_value_l122_122855


namespace ratio_arithmetic_sequence_triangle_l122_122292

theorem ratio_arithmetic_sequence_triangle (a b c : ℝ) 
  (h_triangle : a^2 + b^2 = c^2)
  (h_arith_seq : ∃ d, b = a + d ∧ c = a + 2 * d) :
  a / b = 3 / 4 ∧ b / c = 4 / 5 :=
by
  sorry

end ratio_arithmetic_sequence_triangle_l122_122292


namespace ratio_of_c_and_d_l122_122073

theorem ratio_of_c_and_d 
  (x y c d : ℝ)
  (h₁ : 4 * x - 2 * y = c)
  (h₂ : 6 * y - 12 * x = d) 
  (h₃ : d ≠ 0) : 
  c / d = -1 / 3 :=
by
  sorry

end ratio_of_c_and_d_l122_122073


namespace number_in_pattern_l122_122701

theorem number_in_pattern (m n : ℕ) (h : 8 * m - 5 = 2023) (hn : n = 5) : m + n = 258 :=
by
  sorry

end number_in_pattern_l122_122701


namespace problem_statement_l122_122724

-- Given: x, y, z are real numbers such that x < 0 and x < y < z
variables {x y z : ℝ} 

-- Conditions
axiom h1 : x < 0
axiom h2 : x < y
axiom h3 : y < z

-- Statement to prove: x + y < y + z
theorem problem_statement : x + y < y + z :=
by {
  sorry
}

end problem_statement_l122_122724


namespace jason_needs_87_guppies_per_day_l122_122037

def guppies_needed_per_day (moray_eel_guppies : Nat)
  (betta_fish_number : Nat) (betta_fish_guppies : Nat)
  (angelfish_number : Nat) (angelfish_guppies : Nat)
  (lionfish_number : Nat) (lionfish_guppies : Nat) : Nat :=
  moray_eel_guppies +
  betta_fish_number * betta_fish_guppies +
  angelfish_number * angelfish_guppies +
  lionfish_number * lionfish_guppies

theorem jason_needs_87_guppies_per_day :
  guppies_needed_per_day 20 5 7 3 4 2 10 = 87 := by
  sorry

end jason_needs_87_guppies_per_day_l122_122037


namespace students_in_line_l122_122108

theorem students_in_line (n : ℕ) (h : 1 ≤ n ∧ n ≤ 130) : 
  n = 3 ∨ n = 43 ∨ n = 129 :=
by
  sorry

end students_in_line_l122_122108


namespace area_of_large_rectangle_l122_122499

-- Define the given areas for the sub-shapes
def shaded_square_area : ℝ := 4
def bottom_rectangle_area : ℝ := 2
def right_rectangle_area : ℝ := 6

-- Prove the total area of the large rectangle EFGH is 12 square inches
theorem area_of_large_rectangle : shaded_square_area + bottom_rectangle_area + right_rectangle_area = 12 := 
by 
sorry

end area_of_large_rectangle_l122_122499


namespace arithmetic_sequence_sum_l122_122957

theorem arithmetic_sequence_sum :
  let first_term := 1
  let common_diff := 2
  let last_term := 33
  let n := (last_term + 1) / common_diff
  (n * (first_term + last_term)) / 2 = 289 :=
by
  sorry

end arithmetic_sequence_sum_l122_122957


namespace expand_polynomial_l122_122114

theorem expand_polynomial : 
  ∀ (x : ℝ), (5 * x - 3) * (2 * x^2 + 4 * x + 1) = 10 * x^3 + 14 * x^2 - 7 * x - 3 :=
by
  intro x
  sorry

end expand_polynomial_l122_122114


namespace max_omega_value_l122_122440

noncomputable def f (ω φ x : ℝ) := Real.sin (ω * x + φ)

def center_of_symmetry (ω φ : ℝ) := 
  ∃ n : ℤ, ω * (-Real.pi / 4) + φ = n * Real.pi

def extremum_point (ω φ : ℝ) :=
  ∃ n' : ℤ, ω * (Real.pi / 4) + φ = n' * Real.pi + Real.pi / 2

def monotonic_in_interval (f : ℝ → ℝ) (a b : ℝ) :=
  ∀ x y, a < x ∧ x < b → a < y ∧ y < b → x ≤ y → f x ≤ f y

theorem max_omega_value (ω : ℝ) (φ : ℝ) : 
  (ω > 0) →
  (|φ| ≤ Real.pi / 2) →
  center_of_symmetry ω φ →
  extremum_point ω φ →
  monotonic_in_interval (f ω φ) (5 * Real.pi / 18) (2 * Real.pi / 5) →
  ω = 5 :=
by
  sorry

end max_omega_value_l122_122440


namespace find_x_l122_122450

theorem find_x (x : ℝ) : (x + 3 * x + 1000 + 3000) / 4 = 2018 → x = 1018 :=
by 
  intro h
  sorry

end find_x_l122_122450


namespace prob_score_at_most_7_l122_122973

-- Definitions based on the conditions
def prob_10_ring : ℝ := 0.15
def prob_9_ring : ℝ := 0.35
def prob_8_ring : ℝ := 0.2
def prob_7_ring : ℝ := 0.1

-- Define the event of scoring no more than 7
def score_at_most_7 := prob_7_ring

-- Theorem statement
theorem prob_score_at_most_7 : score_at_most_7 = 0.1 := by 
  -- proof goes here
  sorry

end prob_score_at_most_7_l122_122973


namespace sum_arithmetic_seq_nine_terms_l122_122738

theorem sum_arithmetic_seq_nine_terms
  (a_n : ℕ → ℝ)
  (S_n : ℕ → ℝ)
  (k : ℝ)
  (h1 : ∀ n, a_n n = k * n + 4 - 5 * k)
  (h2 : ∀ n, S_n n = (n / 2) * (a_n 1 + a_n n))
  : S_n 9 = 36 :=
sorry

end sum_arithmetic_seq_nine_terms_l122_122738


namespace mark_card_sum_l122_122378

/--
Mark has seven green cards numbered 1 through 7 and five red cards numbered 2 through 6.
He arranges the cards such that colors alternate and the sum of each pair of neighboring cards forms a prime.
Prove that the sum of the numbers on the last three cards in his stack is 16.
-/
theorem mark_card_sum {green_cards : Fin 7 → ℕ} {red_cards : Fin 5 → ℕ}
  (h_green_numbered : ∀ i, 1 ≤ green_cards i ∧ green_cards i ≤ 7)
  (h_red_numbered : ∀ i, 2 ≤ red_cards i ∧ red_cards i ≤ 6)
  (h_alternate : ∀ i, i < 6 → (∃ j k, green_cards j + red_cards k = prime) ∨ (red_cards j + green_cards k = prime)) :
  ∃ s, s = 16 := sorry

end mark_card_sum_l122_122378


namespace general_term_of_sequence_l122_122312

-- Definition of arithmetic sequence with positive common difference
def is_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) := ∀ n : ℕ, a (n + 1) = a n + d

-- Given conditions
variables {a : ℕ → ℤ} {d : ℤ}
axiom positive_common_difference : d > 0
axiom cond1 : a 3 * a 4 = 117
axiom cond2 : a 2 + a 5 = 22

-- Target statement to prove
theorem general_term_of_sequence : is_arithmetic_sequence a d → a n = 4 * n - 3 :=
by sorry

end general_term_of_sequence_l122_122312


namespace no_real_solution_l122_122886

theorem no_real_solution :
  ¬ ∃ x : ℝ, (3 * x ^ 2 / (x - 2) - (5 * x + 4) / 4 + (10 - 9 * x) / (x - 2) + 2 = 0) :=
sorry

end no_real_solution_l122_122886


namespace find_function_expression_l122_122079

noncomputable def f (a b x : ℝ) : ℝ := 2 ^ (a * x + b)

theorem find_function_expression
  (a b : ℝ)
  (h1 : f a b 1 = 2)
  (h2 : ∃ g : ℝ → ℝ, (∀ x y : ℝ, f (-a) (-b) x = y ↔ f a b y = x) ∧ g (f a b 1) = 1) :
  ∃ (a b : ℝ), f a b x = 2 ^ (-x + 2) :=
by
  sorry

end find_function_expression_l122_122079


namespace xyz_product_condition_l122_122580

theorem xyz_product_condition (x y z : ℝ) (h : x^2 + y^2 = x * y * (z + 1 / z)) : 
  x = y * z ∨ y = x * z :=
sorry

end xyz_product_condition_l122_122580


namespace train_stoppage_time_l122_122051

-- Definitions from conditions
def speed_without_stoppages := 60 -- kmph
def speed_with_stoppages := 36 -- kmph

-- Main statement to prove
theorem train_stoppage_time : (60 - 36) / 60 * 60 = 24 := by
  sorry

end train_stoppage_time_l122_122051


namespace sin_product_l122_122839

theorem sin_product :
  (Real.sin (12 * Real.pi / 180)) * 
  (Real.sin (36 * Real.pi / 180)) *
  (Real.sin (72 * Real.pi / 180)) *
  (Real.sin (84 * Real.pi / 180)) = 1 / 16 := 
by
  sorry

end sin_product_l122_122839


namespace magnitude_of_two_a_minus_b_l122_122202

namespace VectorMagnitude

-- Define the vectors a and b
def a : ℝ × ℝ := (2, 1)
def b : ℝ × ℝ := (3, -2)

-- Function to calculate the magnitude of a vector
noncomputable def magnitude (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

-- Vector operation 2a - b
def two_a_minus_b : ℝ × ℝ := (2 * a.1 - b.1, 2 * a.2 - b.2)

-- The theorem to prove
theorem magnitude_of_two_a_minus_b : magnitude two_a_minus_b = Real.sqrt 17 := by
  sorry

end VectorMagnitude

end magnitude_of_two_a_minus_b_l122_122202


namespace parallel_lines_slope_l122_122614

theorem parallel_lines_slope (a : ℝ) (h : ∀ x y : ℝ, (x + a * y + 6 = 0) → ((a - 2) * x + 3 * y + 2 * a = 0)) : a = -1 :=
by
  sorry

end parallel_lines_slope_l122_122614


namespace ratio_of_coefficients_l122_122610

theorem ratio_of_coefficients (x y c d : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hd : d ≠ 0)
  (H1 : 8 * x - 6 * y = c) (H2 : 12 * y - 18 * x = d) :
  c / d = -4 / 9 := 
by {
  sorry
}

end ratio_of_coefficients_l122_122610


namespace least_rice_l122_122002

variable (o r : ℝ)

-- Conditions
def condition_1 : Prop := o ≥ 8 + r / 2
def condition_2 : Prop := o ≤ 3 * r

-- The main theorem we want to prove
theorem least_rice (h1 : condition_1 o r) (h2 : condition_2 o r) : r ≥ 4 :=
sorry

end least_rice_l122_122002


namespace friendly_number_pair_a_equals_negative_three_fourths_l122_122317

theorem friendly_number_pair_a_equals_negative_three_fourths (a : ℚ) (h : (a / 2) + (3 / 4) = (a + 3) / 6) : 
  a = -3 / 4 :=
sorry

end friendly_number_pair_a_equals_negative_three_fourths_l122_122317


namespace largest_integer_remainder_condition_l122_122590

theorem largest_integer_remainder_condition (number : ℤ) (h1 : number < 100) (h2 : number % 7 = 4) :
  number = 95 := sorry

end largest_integer_remainder_condition_l122_122590


namespace find_n_l122_122430

theorem find_n (x n : ℤ) (k m : ℤ) (h1 : x = 82*k + 5) (h2 : x + n = 41*m + 22) : n = 5 := by
  sorry

end find_n_l122_122430


namespace highest_score_of_D_l122_122620

theorem highest_score_of_D
  (a b c d : ℕ)
  (h1 : a + b = c + d)
  (h2 : b + d > a + c)
  (h3 : a > b + c) :
  d > a :=
by
  sorry

end highest_score_of_D_l122_122620


namespace lines_through_same_quadrants_l122_122885

theorem lines_through_same_quadrants (k b : ℝ) (hk : k ≠ 0):
    ∃ n, n ≥ 7 ∧ ∀ (f : Fin n → ℝ × ℝ), ∃ (i j : Fin n), i ≠ j ∧ 
    ((f i).1 > 0 ∧ (f j).1 > 0 ∧ (f i).2 > 0 ∧ (f j).2 > 0 ∨
     (f i).1 > 0 ∧ (f j).1 > 0 ∧ (f i).2 < 0 ∧ (f j).2 < 0 ∨
     (f i).1 > 0 ∧ (f j).1 > 0 ∧ (f i).2 = 0 ∧ (f j).2 = 0 ∨
     (f i).1 < 0 ∧ (f j).1 < 0 ∧ (f i).2 > 0 ∧ (f j).2 > 0 ∨
     (f i).1 < 0 ∧ (f j).1 < 0 ∧ (f i).2 < 0 ∧ (f j).2 < 0 ∨
     (f i).1 < 0 ∧ (f j).1 < 0 ∧ (f i).2 = 0 ∧ (f j).2 = 0) :=
by sorry

end lines_through_same_quadrants_l122_122885


namespace factorize_a_cubed_minus_a_l122_122485

theorem factorize_a_cubed_minus_a (a : ℝ) : a^3 - a = a * (a + 1) * (a - 1) :=
by
  sorry

end factorize_a_cubed_minus_a_l122_122485


namespace find_value_of_a3_plus_a5_l122_122218

variable {a : ℕ → ℝ}
variable {r : ℝ}

noncomputable def geometric_seq (a : ℕ → ℝ) (r : ℝ) : Prop :=
∀ n, a (n + 1) = a n * r

theorem find_value_of_a3_plus_a5 (h_geom : geometric_seq a r) (h_pos: ∀ n, 0 < a n)
  (h_eq: a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 25) :
  a 3 + a 5 = 5 := by
  sorry

end find_value_of_a3_plus_a5_l122_122218


namespace price_reduction_percentage_l122_122826

theorem price_reduction_percentage (original_price new_price : ℕ) 
  (h_original : original_price = 250) 
  (h_new : new_price = 200) : 
  (original_price - new_price) * 100 / original_price = 20 := 
by 
  -- include the proof when needed
  sorry

end price_reduction_percentage_l122_122826


namespace lunch_break_duration_l122_122910

theorem lunch_break_duration (m a : ℝ) (L : ℝ) :
  (9 - L) * (m + a) = 0.6 → 
  (7 - L) * a = 0.3 → 
  (5 - L) * m = 0.1 → 
  L = 42 / 60 :=
by sorry

end lunch_break_duration_l122_122910


namespace value_of_y_l122_122978

variable (y : ℚ)

def first_boy_marbles : ℚ := 4 * y + 2
def second_boy_marbles : ℚ := 2 * y
def third_boy_marbles : ℚ := y + 3
def total_marbles : ℚ := 31

theorem value_of_y (h : first_boy_marbles y + second_boy_marbles y + third_boy_marbles y = total_marbles) :
  y = 26 / 7 :=
by
  sorry

end value_of_y_l122_122978


namespace find_a_l122_122452

-- Definitions given in the conditions
def f (x : ℝ) : ℝ := x^2 - 2
def g (x : ℝ) : ℝ := x^2 + 6

-- The main theorem to show
theorem find_a (a : ℝ) (h₀ : a > 0) (h₁ : f (g a) = 18) : a = Real.sqrt 14 := sorry

end find_a_l122_122452


namespace ratio_of_segments_intersecting_chords_l122_122765

open Real

variables (EQ FQ HQ GQ : ℝ)

theorem ratio_of_segments_intersecting_chords 
  (h1 : EQ = 5) 
  (h2 : GQ = 7) 
  (h3 : EQ * FQ = GQ * HQ) : 
  FQ / HQ = 7 / 5 :=
by
  sorry

end ratio_of_segments_intersecting_chords_l122_122765


namespace original_price_l122_122431

theorem original_price (P : ℝ) (h₁ : 0.30 * P = 46) : P = 153.33 :=
  sorry

end original_price_l122_122431


namespace solve_for_a_l122_122901

theorem solve_for_a (f : ℝ → ℝ) (a : ℝ)
  (h1 : ∀ x, f (x^3) = Real.log x / Real.log a)
  (h2 : f 8 = 1) :
  a = 2 :=
sorry

end solve_for_a_l122_122901


namespace extremum_at_x_1_max_integer_k_l122_122546

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (x - 1) * Real.log x - (a + 1) * x

theorem extremum_at_x_1 (a : ℝ) : (∀ x : ℝ, 0 < x → ((Real.log x - 1 / x - a = 0) ↔ x = 1))
  → a = -1 ∧
  (∀ x : ℝ, 0 < x → (Real.log x - 1 / x + 1) < 0 → f x (-1) < f 1 (-1) ∧
  (Real.log x - 1 / x + 1) > 0 → f 1 (-1) < f x (-1)) :=
sorry

theorem max_integer_k (k : ℤ) :
  (∀ x : ℝ, 0 < x → (f x 1 > k))
  → k ≤ -4 :=
sorry

end extremum_at_x_1_max_integer_k_l122_122546


namespace calculate_remaining_area_l122_122074

/-- In a rectangular plot of land ABCD, where AB = 20 meters and BC = 12 meters, 
    a triangular garden ABE is installed where AE = 15 meters and BE intersects AE at a perpendicular angle, 
    the area of the remaining part of the land which is not occupied by the garden is 150 square meters. -/
theorem calculate_remaining_area 
  (AB BC AE : ℝ) 
  (hAB : AB = 20) 
  (hBC : BC = 12) 
  (hAE : AE = 15)
  (h_perpendicular : true) : -- BE ⊥ AE implying right triangle ABE
  ∃ area_remaining : ℝ, area_remaining = 150 :=
by
  sorry

end calculate_remaining_area_l122_122074


namespace total_snowfall_l122_122476

theorem total_snowfall (morning afternoon : ℝ) (h1 : morning = 0.125) (h2 : afternoon = 0.5) :
  morning + afternoon = 0.625 := by
  sorry

end total_snowfall_l122_122476


namespace cone_height_correct_l122_122455

noncomputable def height_of_cone (R1 R2 R3 base_radius : ℝ) : ℝ :=
  if R1 = 20 ∧ R2 = 40 ∧ R3 = 40 ∧ base_radius = 21 then 28 else 0

theorem cone_height_correct :
  height_of_cone 20 40 40 21 = 28 :=
by sorry

end cone_height_correct_l122_122455


namespace quadratic_has_real_roots_l122_122198

theorem quadratic_has_real_roots (k : ℝ) (h : k > 0) : ∃ x : ℝ, x^2 + 2 * x - k = 0 :=
by
  sorry

end quadratic_has_real_roots_l122_122198


namespace committee_count_with_president_l122_122986

-- Define the conditions
def total_people : ℕ := 12
def committee_size : ℕ := 5
def remaining_people : ℕ := 11
def president_inclusion : ℕ := 1

-- Define the calculation of binomial coefficient
noncomputable def binomial (n k : ℕ) : ℕ :=
  if h : k ≤ n then (Nat.choose n k) else 0

-- State the problem in Lean 4
theorem committee_count_with_president : 
  binomial remaining_people (committee_size - president_inclusion) = 330 :=
sorry

end committee_count_with_president_l122_122986


namespace man_l122_122407

-- Conditions
def speed_with_current : ℝ := 18
def speed_of_current : ℝ := 3.4

-- Problem statement
theorem man's_speed_against_current :
  (speed_with_current - speed_of_current - speed_of_current) = 11.2 := 
by
  sorry

end man_l122_122407


namespace Polly_lunch_time_l122_122571

-- Define the conditions
def breakfast_time_per_day := 20
def total_days_in_week := 7
def dinner_time_4_days := 10
def remaining_days_in_week := 3
def remaining_dinner_time_per_day := 30
def total_cooking_time := 305

-- Define the total time Polly spends cooking breakfast in a week
def total_breakfast_time := breakfast_time_per_day * total_days_in_week

-- Define the total time Polly spends cooking dinner in a week
def total_dinner_time := (dinner_time_4_days * 4) + (remaining_dinner_time_per_day * remaining_days_in_week)

-- Define the time Polly spends cooking lunch in a week
def lunch_time := total_cooking_time - (total_breakfast_time + total_dinner_time)

-- The theorem to prove Polly's lunch time
theorem Polly_lunch_time : lunch_time = 35 :=
by
  sorry

end Polly_lunch_time_l122_122571


namespace exists_initial_value_l122_122233

theorem exists_initial_value (x : ℤ) : ∃ y : ℤ, x + 49 = y^2 :=
sorry

end exists_initial_value_l122_122233


namespace Jane_possible_numbers_l122_122258

def is_factor (a b : ℕ) : Prop := b % a = 0
def in_range (n : ℕ) : Prop := 500 ≤ n ∧ n ≤ 4000

def Jane_number (m : ℕ) : Prop :=
  is_factor 180 m ∧
  is_factor 42 m ∧
  in_range m

theorem Jane_possible_numbers :
  Jane_number 1260 ∧ Jane_number 2520 ∧ Jane_number 3780 :=
by
  sorry

end Jane_possible_numbers_l122_122258


namespace value_of_x_plus_y_l122_122619

theorem value_of_x_plus_y (x y : ℝ) (h : (x + 1)^2 + |y - 6| = 0) : x + y = 5 :=
by
sorry

end value_of_x_plus_y_l122_122619


namespace simplify_expression_l122_122591

-- Define the given conditions
def pow_2_5 : ℕ := 32
def pow_4_4 : ℕ := 256
def pow_2_2 : ℕ := 4
def pow_neg_2_3 : ℤ := -8

-- State the theorem to prove
theorem simplify_expression : 
  (pow_2_5 + pow_4_4) * (pow_2_2 - pow_neg_2_3)^8 = 123876479488 := 
by
  sorry

end simplify_expression_l122_122591


namespace least_possible_value_of_a_plus_b_l122_122927

theorem least_possible_value_of_a_plus_b : ∃ (a b : ℕ), a > 0 ∧ b > 0 ∧
  Nat.gcd (a + b) 330 = 1 ∧
  b ∣ a^a ∧ 
  ∀ k : ℕ, b^3 ∣ a^a → (k ∣ a → k = 1) ∧
  a + b = 392 :=
by
  sorry

end least_possible_value_of_a_plus_b_l122_122927


namespace exit_forest_strategy_l122_122053

/-- A strategy ensuring the parachutist will exit the forest with a path length of less than 2.5l -/
theorem exit_forest_strategy (l : Real) : 
  ∃ (path_length : Real), path_length < 2.5 * l :=
by
  use 2.278 * l
  sorry

end exit_forest_strategy_l122_122053


namespace part1_m_n_part2_k_l122_122028

-- Definitions of vectors a, b, and c
def veca : ℝ × ℝ := (3, 2)
def vecb : ℝ × ℝ := (-1, 2)
def vecc : ℝ × ℝ := (4, 1)

-- Part (1)
theorem part1_m_n : 
  ∃ (m n : ℝ), (-m + 4 * n = 3) ∧ (2 * m + n = 2) :=
sorry

-- Part (2)
theorem part2_k : 
  ∃ (k : ℝ), (3 + 4 * k) * 2 - (-5) * (2 + k) = 0 :=
sorry

end part1_m_n_part2_k_l122_122028


namespace contest_correct_answers_l122_122903

/-- 
In a mathematics contest with ten problems, a student gains 
5 points for a correct answer and loses 2 points for an 
incorrect answer. If Olivia answered every problem 
and her score was 29, how many correct answers did she have?
-/
theorem contest_correct_answers (c w : ℕ) (h1 : c + w = 10) (h2 : 5 * c - 2 * w = 29) : c = 7 :=
by 
  sorry

end contest_correct_answers_l122_122903


namespace factor_polynomial_l122_122953

theorem factor_polynomial (x y : ℝ) : 2 * x^2 - 2 * y^2 = 2 * (x + y) * (x - y) :=
by
  sorry

end factor_polynomial_l122_122953


namespace problem_solution_l122_122184

noncomputable def f (x : ℝ) : ℝ :=
  if x < 1 then 1 + Real.log (2 - x) / Real.log 2 else 2 ^ (x - 1)

theorem problem_solution : f (-2) + f (Real.log 12 / Real.log 2) = 9 := by
  sorry

end problem_solution_l122_122184


namespace scientific_calculator_ratio_l122_122648

theorem scientific_calculator_ratio (total : ℕ) (basic_cost : ℕ) (change : ℕ) (sci_ratio : ℕ → ℕ) (graph_ratio : ℕ → ℕ) : 
  total = 100 →
  basic_cost = 8 →
  sci_ratio basic_cost = 8 * x →
  graph_ratio (sci_ratio basic_cost) = 3 * sci_ratio basic_cost →
  change = 28 →
  8 + (8 * x) + (24 * x) = 72 →
  x = 2 :=
by
  sorry

end scientific_calculator_ratio_l122_122648


namespace magic_square_sum_l122_122393

theorem magic_square_sum (S a b c d e : ℤ) (h1 : x + 15 + 100 = S)
                        (h2 : 23 + d + e = S)
                        (h3 : x + a + 23 = S)
                        (h4 : a = 92)
                        (h5 : 92 + b + d = x + 15 + 100)
                        (h6 : b = 0)
                        (h7 : d = 100) : x = 77 :=
by {
  sorry
}

end magic_square_sum_l122_122393


namespace tan_monotonic_increasing_interval_l122_122892

noncomputable def monotonic_increasing_interval (k : ℤ) : Set ℝ :=
  {x : ℝ | 2 * k * Real.pi - (5 * Real.pi) / 3 < x ∧ x < 2 * k * Real.pi + Real.pi / 3 }

theorem tan_monotonic_increasing_interval (k : ℤ) :
  ∀ x : ℝ, (y = Real.tan ((x / 2) + (Real.pi / 3))) → 
           x ∈ monotonic_increasing_interval k :=
sorry

end tan_monotonic_increasing_interval_l122_122892


namespace cos_value_given_sin_condition_l122_122950

open Real

theorem cos_value_given_sin_condition (x : ℝ) (h : sin (x + π / 12) = -1/4) : 
  cos (5 * π / 6 - 2 * x) = -7 / 8 :=
sorry -- Proof steps are omitted.

end cos_value_given_sin_condition_l122_122950


namespace gain_percentage_is_five_percent_l122_122896

variables (CP SP New_SP Loss Loss_Percentage Gain Gain_Percentage : ℝ)
variables (H1 : Loss_Percentage = 10)
variables (H2 : CP = 933.33)
variables (H3 : Loss = (Loss_Percentage / 100) * CP)
variables (H4 : SP = CP - Loss)
variables (H5 : New_SP = SP + 140)
variables (H6 : Gain = New_SP - CP)
variables (H7 : Gain_Percentage = (Gain / CP) * 100)

theorem gain_percentage_is_five_percent :
  Gain_Percentage = 5 :=
by
  -- Proof goes here
  sorry

end gain_percentage_is_five_percent_l122_122896


namespace initial_money_l122_122359

-- Define the conditions
variable (M : ℝ)
variable (h : (1 / 3) * M = 50)

-- Define the theorem to be proved
theorem initial_money : M = 150 := 
by
  sorry

end initial_money_l122_122359


namespace solution_inequality_equivalence_l122_122424

-- Define the inequality to be proved
def inequality (x : ℝ) : Prop :=
  (x + 1 / 2) * (3 / 2 - x) ≥ 0

-- Define the set of solutions such that -1/2 ≤ x ≤ 3/2
def solution_set (x : ℝ) : Prop :=
  -1 / 2 ≤ x ∧ x ≤ 3 / 2

-- The statement to be proved: the solution set of the inequality is {x | -1/2 ≤ x ≤ 3/2}
theorem solution_inequality_equivalence :
  {x : ℝ | inequality x} = {x : ℝ | solution_set x} :=
by 
  sorry

end solution_inequality_equivalence_l122_122424


namespace machine_production_in_10_seconds_l122_122851

def items_per_minute : ℕ := 150
def seconds_per_minute : ℕ := 60
def production_rate_per_second : ℚ := items_per_minute / seconds_per_minute
def production_time_in_seconds : ℕ := 10
def expected_production_in_ten_seconds : ℚ := 25

theorem machine_production_in_10_seconds :
  (production_rate_per_second * production_time_in_seconds) = expected_production_in_ten_seconds :=
sorry

end machine_production_in_10_seconds_l122_122851


namespace arithmetic_sequence_value_l122_122264

theorem arithmetic_sequence_value :
  ∀ (a_n : ℕ → ℤ) (d : ℤ),
    (∀ n : ℕ, a_n n = a_n 0 + ↑n * d) →
    a_n 2 = 4 →
    a_n 4 = 8 →
    a_n 10 = 20 :=
by
  intros a_n d h_arith h_a3 h_a5
  --
  sorry

end arithmetic_sequence_value_l122_122264


namespace number_of_students_l122_122397

theorem number_of_students (n : ℕ) (h1 : n < 40) (h2 : n % 7 = 3) (h3 : n % 6 = 1) : n = 31 := 
by
  sorry

end number_of_students_l122_122397


namespace sum_of_fractions_l122_122185

theorem sum_of_fractions :
  (3 / 20 : ℝ) +  (7 / 200) + (8 / 2000) + (3 / 20000) = 0.1892 :=
by 
  sorry

end sum_of_fractions_l122_122185


namespace wheel_sum_even_and_greater_than_10_l122_122521

-- Definitions based on conditions
def prob_even_A : ℚ := 3 / 8
def prob_odd_A : ℚ := 5 / 8
def prob_even_B : ℚ := 1 / 4
def prob_odd_B : ℚ := 3 / 4

-- Event probabilities from solution steps
def prob_both_even : ℚ := prob_even_A * prob_even_B
def prob_both_odd : ℚ := prob_odd_A * prob_odd_B
def prob_even_sum : ℚ := prob_both_even + prob_both_odd
def prob_even_sum_greater_10 : ℚ := 1 / 3

-- Compute final probability
def final_probability : ℚ := prob_even_sum * prob_even_sum_greater_10

-- The statement that needs proving
theorem wheel_sum_even_and_greater_than_10 : final_probability = 3 / 16 := by
  sorry

end wheel_sum_even_and_greater_than_10_l122_122521


namespace smallest_D_for_inequality_l122_122552

theorem smallest_D_for_inequality :
  ∃ D : ℝ, (∀ x y z : ℝ, 2 * x^2 + 3 * y^2 + z^2 + 3 ≥ D * (x + y + z)) ∧ 
           D = -Real.sqrt (72 / 11) :=
by
  sorry

end smallest_D_for_inequality_l122_122552


namespace pills_first_day_l122_122597

theorem pills_first_day (P : ℕ) 
  (h1 : P + (P + 2) + (P + 4) + (P + 6) + (P + 8) + (P + 10) + (P + 12) = 49) : 
  P = 1 :=
by sorry

end pills_first_day_l122_122597


namespace coconut_grove_l122_122987

theorem coconut_grove (x : ℕ) :
  (60 * (x + 1) + 120 * x + 180 * (x - 1)) = 300 * x → x = 2 :=
by
  intro h
  -- We can leave the proof part to prove this later.
  sorry

end coconut_grove_l122_122987


namespace susie_earnings_l122_122115

-- Define the constants and conditions
def price_per_slice : ℕ := 3
def price_per_whole_pizza : ℕ := 15
def slices_sold : ℕ := 24
def whole_pizzas_sold : ℕ := 3

-- Calculate earnings from slices and whole pizzas
def earnings_from_slices : ℕ := slices_sold * price_per_slice
def earnings_from_whole_pizzas : ℕ := whole_pizzas_sold * price_per_whole_pizza
def total_earnings : ℕ := earnings_from_slices + earnings_from_whole_pizzas

-- Prove that the total earnings are $117
theorem susie_earnings : total_earnings = 117 := by
  sorry

end susie_earnings_l122_122115


namespace binom_150_1_eq_150_l122_122267

/-- Definition of factorial -/
def fact : ℕ → ℕ
| 0 => 1
| (n + 1) => (n + 1) * fact n

/-- Definition of binomial coefficient -/
def binom (n k : ℕ) : ℕ :=
fact n / (fact k * fact (n - k))

/-- Theorem stating the specific binomial coefficient calculation -/
theorem binom_150_1_eq_150 : binom 150 1 = 150 := by
  sorry

end binom_150_1_eq_150_l122_122267


namespace substring_012_appears_148_times_l122_122207

noncomputable def count_substring_012_in_base_3_concat (n : ℕ) : ℕ :=
  -- The function that counts the "012" substrings in the concatenated base-3 representations
  sorry

theorem substring_012_appears_148_times :
  count_substring_012_in_base_3_concat 728 = 148 :=
  sorry

end substring_012_appears_148_times_l122_122207


namespace problem_water_percentage_l122_122011

noncomputable def percentage_water_in_mixture 
  (volA volB volC volD : ℕ) 
  (pctA pctB pctC pctD : ℝ) : ℝ :=
  let total_volume := volA + volB + volC + volD
  let total_solution := volA * pctA + volB * pctB + volC * pctC + volD * pctD
  let total_water := total_volume - total_solution
  (total_water / total_volume) * 100

theorem problem_water_percentage :
  percentage_water_in_mixture 100 90 60 50 0.25 0.3 0.4 0.2 = 71.33 :=
by
  -- proof goes here
  sorry

end problem_water_percentage_l122_122011


namespace greatest_divisor_remainders_l122_122709

theorem greatest_divisor_remainders (x : ℕ) (h1 : 1255 % x = 8) (h2 : 1490 % x = 11) : x = 29 :=
by
  -- The proof steps would go here, but for now, we use sorry.
  sorry

end greatest_divisor_remainders_l122_122709


namespace correct_sum_l122_122951

theorem correct_sum (x y : ℕ) (h1 : x > y) (h2 : x - y = 4) (h3 : x * y = 98) : x + y = 18 := 
by
  sorry

end correct_sum_l122_122951


namespace roots_sum_one_imp_b_eq_neg_a_l122_122337

theorem roots_sum_one_imp_b_eq_neg_a (a b c : ℝ) (h : a ≠ 0) 
  (hr : ∀ (r s : ℝ), r + s = 1 → (r * s = c / a) → a * (r^2 + (b/a) * r + c/a) = 0) : b = -a :=
sorry

end roots_sum_one_imp_b_eq_neg_a_l122_122337


namespace age_of_sisters_l122_122331

theorem age_of_sisters (a b : ℕ) (h1 : 10 * a - 9 * b = 89) 
  (h2 : 10 = 10) : a = 17 ∧ b = 9 :=
by sorry

end age_of_sisters_l122_122331


namespace problem_lean_version_l122_122390

theorem problem_lean_version (n : ℕ) : 
  (n > 0) ∧ (6^n - 1 ∣ 7^n - 1) ↔ ∃ k : ℕ, n = 4 * k :=
by
  sorry

end problem_lean_version_l122_122390


namespace remaining_pages_after_a_week_l122_122022

-- Define the conditions
def total_pages : Nat := 381
def pages_read_initial : Nat := 149
def pages_per_day : Nat := 20
def days : Nat := 7

-- Define the final statement to prove
theorem remaining_pages_after_a_week :
  let pages_left_initial := total_pages - pages_read_initial
  let pages_read_week := pages_per_day * days
  let pages_remaining := pages_left_initial - pages_read_week
  pages_remaining = 92 := by
  sorry

end remaining_pages_after_a_week_l122_122022


namespace solve_for_y_l122_122042

theorem solve_for_y (x y : ℤ) (h1 : x - y = 16) (h2 : x + y = 10) : y = -3 :=
sorry

end solve_for_y_l122_122042


namespace trigonometric_identity_l122_122564

theorem trigonometric_identity :
  (3 / (Real.sin (20 * Real.pi / 180))^2) - 
  (1 / (Real.cos (20 * Real.pi / 180))^2) + 
  64 * (Real.sin (20 * Real.pi / 180))^2 = 32 :=
by sorry

end trigonometric_identity_l122_122564


namespace problem1_problem2_l122_122178

theorem problem1 (n : ℕ) : 2 ≤ (1 + 1 / n) ^ n ∧ (1 + 1 / n) ^ n < 3 :=
sorry

theorem problem2 (n : ℕ) : (n / 3) ^ n < n! :=
sorry

end problem1_problem2_l122_122178


namespace zero_of_sum_of_squares_eq_zero_l122_122402

theorem zero_of_sum_of_squares_eq_zero (a b : ℝ) (h : a^2 + b^2 = 0) : a = 0 ∧ b = 0 :=
by
  sorry

end zero_of_sum_of_squares_eq_zero_l122_122402


namespace sqrt_0_1681_eq_0_41_l122_122793

theorem sqrt_0_1681_eq_0_41 (h : Real.sqrt 16.81 = 4.1) : Real.sqrt 0.1681 = 0.41 := by 
  sorry

end sqrt_0_1681_eq_0_41_l122_122793


namespace Cindy_crayons_l122_122757

variable (K : ℕ) -- Karen's crayons
variable (C : ℕ) -- Cindy's crayons

-- Given conditions
def Karen_has_639_crayons : Prop := K = 639
def Karen_has_135_more_crayons_than_Cindy : Prop := K = C + 135

-- The proof problem: showing Cindy's crayons
theorem Cindy_crayons (h1 : Karen_has_639_crayons K) (h2 : Karen_has_135_more_crayons_than_Cindy K C) : C = 504 :=
by
  sorry

end Cindy_crayons_l122_122757


namespace b_investment_l122_122628

theorem b_investment (A_invest C_invest total_profit A_profit x : ℝ) 
(h1 : A_invest = 2400) 
(h2 : C_invest = 9600) 
(h3 : total_profit = 9000) 
(h4 : A_profit = 1125)
(h5 : x = (8100000 / 1125)) : 
x = 7200 := by
  rw [h5]
  sorry

end b_investment_l122_122628


namespace largest_n_l122_122777

theorem largest_n : ∃ (n : ℕ), n < 1000 ∧ (∃ (m : ℕ), lcm m n = 3 * m * gcd m n) ∧ (∀ k, k < 1000 ∧ (∃ (m' : ℕ), lcm m' k = 3 * m' * gcd m' k) → k ≤ 972) := sorry

end largest_n_l122_122777


namespace rounding_sum_eq_one_third_probability_l122_122334

noncomputable def rounding_sum_probability : ℝ :=
  (λ (total : ℝ) => 
    let round := (λ (x : ℝ) => if x < 0.5 then 0 else if x < 1.5 then 1 else if x < 2.5 then 2 else 3)
    let interval := (λ (start : ℝ) (end_ : ℝ) => end_ - start)
    let sum_conditions := [((0.5,1.5), 3), ((1.5,2.5), 2)]
    let total_length := 3

    let valid_intervals := sum_conditions.map (λ p => interval (p.fst.fst) (p.fst.snd))
    let total_valid_interval := List.sum valid_intervals
    total_valid_interval / total_length
  ) 3

theorem rounding_sum_eq_one_third_probability : rounding_sum_probability = 2 / 3 := by sorry

end rounding_sum_eq_one_third_probability_l122_122334


namespace least_integer_square_eq_double_plus_64_l122_122735

theorem least_integer_square_eq_double_plus_64 :
  ∃ x : ℤ, x^2 = 2 * x + 64 ∧ ∀ y : ℤ, y^2 = 2 * y + 64 → y ≥ x → x = -8 :=
by
  sorry

end least_integer_square_eq_double_plus_64_l122_122735


namespace contrapositive_iff_l122_122199

theorem contrapositive_iff (a b : ℤ) : (a > b → a - 5 > b - 5) ↔ (a - 5 ≤ b - 5 → a ≤ b) :=
by sorry

end contrapositive_iff_l122_122199


namespace ratio_of_segments_of_hypotenuse_l122_122737

theorem ratio_of_segments_of_hypotenuse
  (a b c r s : ℝ) (h_right_triangle : a^2 + b^2 = c^2)
  (h_ratio : a / b = 2 / 5)
  (h_r : r = (a^2) / c) 
  (h_s : s = (b^2) / c) : 
  r / s = 4 / 25 := sorry

end ratio_of_segments_of_hypotenuse_l122_122737


namespace distance_between_feet_of_perpendiculars_eq_area_over_radius_l122_122923
noncomputable def area (ABC : Type) : ℝ := sorry
noncomputable def circumradius (ABC : Type) : ℝ := sorry

theorem distance_between_feet_of_perpendiculars_eq_area_over_radius
  (ABC : Type)
  (area_ABC : ℝ)
  (R : ℝ)
  (h_area : area ABC = area_ABC)
  (h_radius : circumradius ABC = R) :
  ∃ (m : ℝ), m = area_ABC / R := sorry

end distance_between_feet_of_perpendiculars_eq_area_over_radius_l122_122923


namespace initial_nickels_l122_122655

theorem initial_nickels (N : ℕ) (h1 : N + 9 + 2 = 18) : N = 7 :=
by sorry

end initial_nickels_l122_122655


namespace city_map_representation_l122_122741

-- Given conditions
def scale (x : ℕ) : ℕ := x * 6
def cm_represents_km(cm : ℕ) : ℕ := scale cm
def fifteen_cm := 15
def ninety_km := 90

-- Given condition: 15 centimeters represents 90 kilometers
axiom representation : cm_represents_km fifteen_cm = ninety_km

-- Proof statement: A 20-centimeter length represents 120 kilometers
def twenty_cm := 20
def correct_answer := 120

theorem city_map_representation : cm_represents_km twenty_cm = correct_answer := by
  sorry

end city_map_representation_l122_122741


namespace total_lives_l122_122080

/-- Suppose there are initially 4 players, then 5 more players join. Each player has 3 lives.
    Prove that the total number of lives is equal to 27. -/
theorem total_lives (initial_players : ℕ) (additional_players : ℕ) (lives_per_player : ℕ) 
  (h_initial : initial_players = 4) (h_additional : additional_players = 5) (h_lives : lives_per_player = 3) : 
  initial_players + additional_players = 9 ∧ 
  (initial_players + additional_players) * lives_per_player = 27 :=
by
  sorry

end total_lives_l122_122080


namespace box_surface_area_correct_l122_122918

-- Define the dimensions of the original cardboard.
def original_length : ℕ := 25
def original_width : ℕ := 40

-- Define the size of the squares removed from each corner.
def square_side : ℕ := 8

-- Define the surface area function.
def surface_area (length width : ℕ) (square_side : ℕ) : ℕ :=
  let area_remaining := (length * width) - 4 * (square_side * square_side)
  area_remaining

-- The theorem statement to prove
theorem box_surface_area_correct : surface_area original_length original_width square_side = 744 :=
by
  sorry

end box_surface_area_correct_l122_122918


namespace inequality_proof_l122_122965

theorem inequality_proof (x y : ℝ) (hx : x > 1) (hy : y > 1) : 
  (x^2 / (y - 1) + y^2 / (x - 1) ≥ 8) :=
  sorry

end inequality_proof_l122_122965


namespace alex_ahead_of_max_after_even_l122_122921

theorem alex_ahead_of_max_after_even (x : ℕ) (h1 : x - 200 + 170 + 440 = 1110) : x = 300 :=
sorry

end alex_ahead_of_max_after_even_l122_122921


namespace number_of_pies_is_correct_l122_122976

def weight_of_apples : ℕ := 120
def weight_for_applesauce (w : ℕ) : ℕ := w / 2
def weight_for_pies (w wholly_app : ℕ) : ℕ := w - wholly_app
def pies (weight_per_pie total_weight : ℕ) : ℕ := total_weight / weight_per_pie

theorem number_of_pies_is_correct :
  pies 4 (weight_for_pies weight_of_apples (weight_for_applesauce weight_of_apples)) = 15 :=
by
  sorry

end number_of_pies_is_correct_l122_122976


namespace toppings_combination_l122_122947

-- Define the combination function
def combination (n k : ℕ) : ℕ := n.choose k

theorem toppings_combination :
  combination 9 3 = 84 := by
  sorry

end toppings_combination_l122_122947


namespace tangent_line_to_curve_at_P_l122_122294

noncomputable def tangent_line_at_point (x y : ℝ) := 4 * x - y - 2 = 0

theorem tangent_line_to_curve_at_P :
  (∃ (b: ℝ), ∀ (x: ℝ), b = 2 * 1^2 → tangent_line_at_point 1 2)
:= 
by
  sorry

end tangent_line_to_curve_at_P_l122_122294


namespace spaghetti_cost_l122_122016

theorem spaghetti_cost (hamburger_cost french_fry_cost soda_cost spaghetti_cost split_payment friends : ℝ) 
(hamburger_count : ℕ) (french_fry_count : ℕ) (soda_count : ℕ) (friend_count : ℕ)
(h_split_payment : split_payment * friend_count = 25)
(h_hamburger_cost : hamburger_cost = 3 * hamburger_count)
(h_french_fry_cost : french_fry_cost = 1.20 * french_fry_count)
(h_soda_cost : soda_cost = 0.5 * soda_count)
(h_total_order_cost : hamburger_cost + french_fry_cost + soda_cost + spaghetti_cost = split_payment * friend_count) :
spaghetti_cost = 2.70 :=
by {
  sorry
}

end spaghetti_cost_l122_122016


namespace time_for_train_to_pass_pole_l122_122049

-- Definitions based on conditions
def train_length_meters : ℕ := 160
def train_speed_kmph : ℕ := 72

-- The calculated speed in m/s
def train_speed_mps : ℕ := train_speed_kmph * 1000 / 3600

-- The calculation of time taken to pass the pole
def time_to_pass_pole : ℕ := train_length_meters / train_speed_mps

-- The theorem statement
theorem time_for_train_to_pass_pole : time_to_pass_pole = 8 := sorry

end time_for_train_to_pass_pole_l122_122049


namespace parabolas_intersect_diff_l122_122671

theorem parabolas_intersect_diff (a b c d : ℝ) (h1 : c ≥ a)
  (h2 : b = 3 * a^2 - 6 * a + 3)
  (h3 : d = 3 * c^2 - 6 * c + 3)
  (h4 : b = -2 * a^2 - 4 * a + 6)
  (h5 : d = -2 * c^2 - 4 * c + 6) :
  c - a = 1.6 :=
sorry

end parabolas_intersect_diff_l122_122671


namespace shirt_tie_combinations_l122_122617

noncomputable def shirts : ℕ := 8
noncomputable def ties : ℕ := 7
noncomputable def forbidden_combinations : ℕ := 2

theorem shirt_tie_combinations :
  shirts * ties - forbidden_combinations = 54 := by
  sorry

end shirt_tie_combinations_l122_122617


namespace quadrilateral_area_lt_one_l122_122897

theorem quadrilateral_area_lt_one 
  (a b c d : ℝ) 
  (h_a : a < 1) 
  (h_b : b < 1) 
  (h_c : c < 1) 
  (h_d : d < 1) 
  (h_pos_a : 0 ≤ a)
  (h_pos_b : 0 ≤ b)
  (h_pos_c : 0 ≤ c)
  (h_pos_d : 0 ≤ d) :
  ∃ (area : ℝ), area < 1 :=
by
  sorry

end quadrilateral_area_lt_one_l122_122897


namespace value_of_f_2012_l122_122916

def odd_function (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = -f x

noncomputable def f : ℝ → ℝ := sorry

axiom odd_fn : odd_function f
axiom f_at_2 : f 2 = 0
axiom functional_eq : ∀ x : ℝ, f (x + 4) = f x + f 4

theorem value_of_f_2012 : f 2012 = 0 :=
by
  sorry

end value_of_f_2012_l122_122916


namespace problem_true_propositions_l122_122852

-- Definitions
def is_square (q : ℕ) : Prop := q = 4
def is_trapezoid (q : ℕ) : Prop := q ≠ 4
def is_parallelogram (q : ℕ) : Prop := q = 2

-- Propositions
def prop_negation (p : Prop) : Prop := ¬ p
def prop_contrapositive (p q : Prop) : Prop := ¬ q → ¬ p
def prop_inverse (p q : Prop) : Prop := p → q

-- True propositions
theorem problem_true_propositions (a b c : ℕ) (h1 : ¬ (is_square 4)) (h2 : ¬ (is_parallelogram 3)) (h3 : ¬ (a * c^2 > b * c^2 → a > b)) : 
    (prop_negation (is_square 4) ∧ prop_contrapositive (is_trapezoid 3) (is_parallelogram 3)) ∧ ¬ prop_inverse (a * c^2 > b * c^2) (a > b) := 
by
    sorry

end problem_true_propositions_l122_122852


namespace pizzeria_large_pizzas_sold_l122_122092

theorem pizzeria_large_pizzas_sold (price_small price_large total_earnings num_small_pizzas num_large_pizzas : ℕ):
  price_small = 2 →
  price_large = 8 →
  total_earnings = 40 →
  num_small_pizzas = 8 →
  total_earnings = price_small * num_small_pizzas + price_large * num_large_pizzas →
  num_large_pizzas = 3 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end pizzeria_large_pizzas_sold_l122_122092


namespace part1_part2_l122_122639

def A (x : ℝ) (a : ℝ) : Prop := x^2 - 4 * a * x + 3 * a^2 < 0
def B (x : ℝ) : Prop := (x - 3) * (2 - x) ≥ 0

theorem part1 (a : ℝ) (ha1: a = 1) :
  ∀ x, (A x 1 ∧ B x) ↔ (2 ≤ x ∧ x < 3) :=
sorry

theorem part2 (a : ℝ) (ha1: a = 1) :
  ∀ x, (A x 1 ∨ B x) ↔ (1 < x ∧ x ≤ 3) :=
sorry

end part1_part2_l122_122639


namespace total_circle_area_within_triangle_l122_122024

-- Define the sides of the triangle
def triangle_sides : Prop := ∃ (a b c : ℝ), a = 3 ∧ b = 4 ∧ c = 5

-- Define the radii and center of the circles at each vertex of the triangle
def circle_centers_and_radii : Prop := ∃ (r : ℝ) (A B C : ℝ × ℝ), r = 1

-- The formal statement that we need to prove:
theorem total_circle_area_within_triangle :
  triangle_sides ∧ circle_centers_and_radii → 
  (total_area_of_circles_within_triangle = π / 2) := sorry

end total_circle_area_within_triangle_l122_122024


namespace solution_set_of_absolute_inequality_l122_122673

theorem solution_set_of_absolute_inequality :
  {x : ℝ | |2 * x - 1| < 1} = {x : ℝ | 0 < x ∧ x < 1} :=
by
  sorry

end solution_set_of_absolute_inequality_l122_122673


namespace largest_two_digit_n_l122_122707

theorem largest_two_digit_n (x : ℕ) (n : ℕ) (hx : x < 10) (hx_nonzero : 0 < x)
  (hn : n = 12 * x * x) (hn_two_digit : n < 100) : n = 48 :=
by sorry

end largest_two_digit_n_l122_122707


namespace projectile_reaches_30m_at_2_seconds_l122_122374

theorem projectile_reaches_30m_at_2_seconds:
  ∀ t : ℝ, -5 * t^2 + 25 * t = 30 → t = 2 ∨ t = 3 :=
by
  sorry

end projectile_reaches_30m_at_2_seconds_l122_122374


namespace train_passes_bridge_in_128_seconds_l122_122365

/-- A proof problem regarding a train passing a bridge -/
theorem train_passes_bridge_in_128_seconds 
  (train_length : ℕ) 
  (train_speed_kmh : ℕ) 
  (bridge_length : ℕ) 
  (conversion_factor : ℚ) 
  (time_to_pass : ℚ) :
  train_length = 1200 →
  train_speed_kmh = 90 →
  bridge_length = 2000 →
  conversion_factor = (5 / 18) →
  time_to_pass = (train_length + bridge_length) / (train_speed_kmh * conversion_factor) →
  time_to_pass = 128 := 
by
  -- We are skipping the proof itself
  sorry

end train_passes_bridge_in_128_seconds_l122_122365


namespace k_plus_a_equals_three_halves_l122_122689

theorem k_plus_a_equals_three_halves :
  ∃ (k a : ℝ), (2 = k * 4 ^ a) ∧ (k + a = 3 / 2) :=
sorry

end k_plus_a_equals_three_halves_l122_122689


namespace collapsing_fraction_l122_122412

-- Define the total number of homes on Gotham St as a variable.
variable (T : ℕ)

/-- Fraction of homes on Gotham Street that are termite-ridden. -/
def fraction_termite_ridden (T : ℕ) : ℚ := 1 / 3

/-- Fraction of homes on Gotham Street that are termite-ridden but not collapsing. -/
def fraction_termite_not_collapsing (T : ℕ) : ℚ := 1 / 10

/-- Fraction of termite-ridden homes that are collapsing. -/
theorem collapsing_fraction :
  (fraction_termite_ridden T - fraction_termite_not_collapsing T) = 7 / 30 :=
by
  sorry

end collapsing_fraction_l122_122412


namespace jason_bought_correct_dozens_l122_122743

-- Given conditions
def cupcakes_per_cousin : Nat := 3
def cousins : Nat := 16
def cupcakes_per_dozen : Nat := 12

-- Calculated value
def total_cupcakes : Nat := cupcakes_per_cousin * cousins
def dozens_of_cupcakes_bought : Nat := total_cupcakes / cupcakes_per_dozen

-- Theorem statement
theorem jason_bought_correct_dozens : dozens_of_cupcakes_bought = 4 := by
  -- Proof omitted
  sorry

end jason_bought_correct_dozens_l122_122743


namespace value_of_n_l122_122749

theorem value_of_n (n : ℕ) (h1 : 0 < n) (h2 : n < Real.sqrt 65) (h3 : Real.sqrt 65 < n + 1) : n = 8 := 
sorry

end value_of_n_l122_122749


namespace victor_won_games_l122_122768

theorem victor_won_games (V : ℕ) (ratio_victor_friend : 9 * 20 = 5 * V) : V = 36 :=
sorry

end victor_won_games_l122_122768


namespace how_many_fewer_girls_l122_122356

def total_students : ℕ := 27
def girls : ℕ := 11
def boys : ℕ := total_students - girls
def fewer_girls_than_boys : ℕ := boys - girls

theorem how_many_fewer_girls :
  fewer_girls_than_boys = 5 :=
sorry

end how_many_fewer_girls_l122_122356


namespace red_balls_estimation_l122_122248

noncomputable def numberOfRedBalls (x : ℕ) : ℝ := x / (x + 3)

theorem red_balls_estimation {x : ℕ} (h : numberOfRedBalls x = 0.85) : x = 17 :=
by
  sorry

end red_balls_estimation_l122_122248


namespace train_travel_distance_l122_122377

theorem train_travel_distance (m : ℝ) (h : 3 * 60 * 1 = m) : m = 180 :=
by
  sorry

end train_travel_distance_l122_122377


namespace train_speed_is_60_kmph_l122_122854

-- Define the distance and time
def train_length : ℕ := 400
def bridge_length : ℕ := 800
def time_to_pass_bridge : ℕ := 72

-- Define the distances and calculations
def total_distance : ℕ := train_length + bridge_length
def speed_m_per_s : ℚ := total_distance / time_to_pass_bridge
def speed_km_per_h : ℚ := speed_m_per_s * 3.6

-- State and prove the theorem
theorem train_speed_is_60_kmph : speed_km_per_h = 60 := by
  sorry

end train_speed_is_60_kmph_l122_122854


namespace dimes_count_l122_122039

def num_dimes (total_in_cents : ℤ) (value_quarter value_dime value_nickel : ℤ) (num_each : ℤ) : Prop :=
  total_in_cents = num_each * (value_quarter + value_dime + value_nickel)

theorem dimes_count (num_each : ℤ) :
  num_dimes 440 25 10 5 num_each → num_each = 11 :=
by sorry

end dimes_count_l122_122039


namespace triangle_existence_l122_122283

theorem triangle_existence 
  (h_a h_b m_a : ℝ) :
  (m_a ≥ h_a) → 
  ((h_a > 1/2 * h_b ∧ m_a > h_a → true ∨ false) ∧ 
  (m_a = h_a → true ∨ false) ∧ 
  (h_a ≤ 1/2 * h_b ∧ 1/2 * h_b < m_a → true ∨ false) ∧ 
  (h_a ≤ 1/2 * h_b ∧ 1/2 * h_b = m_a → false ∨ true) ∧ 
  (1/2 * h_b > m_a → false)) :=
by
  intro
  sorry

end triangle_existence_l122_122283


namespace min_guesses_correct_l122_122634

noncomputable def min_guesses (n k : ℕ) (h : n > k) : ℕ :=
  if n = 2 * k then 2 else 1

theorem min_guesses_correct (n k : ℕ) (h : n > k) :
  min_guesses n k h = if n = 2 * k then 2 else 1 :=
by
  sorry

end min_guesses_correct_l122_122634


namespace hypotenuse_length_l122_122878

theorem hypotenuse_length (a b c : ℝ) (h_right_angled : c^2 = a^2 + b^2) (h_sum_of_squares : a^2 + b^2 + c^2 = 2500) : c = 25 * Real.sqrt 2 :=
by
  sorry

end hypotenuse_length_l122_122878


namespace markup_amount_l122_122582

def purchase_price : ℝ := 48
def overhead_percentage : ℝ := 0.35
def net_profit : ℝ := 18

def overhead : ℝ := purchase_price * overhead_percentage
def total_cost : ℝ := purchase_price + overhead
def selling_price : ℝ := total_cost + net_profit
def markup : ℝ := selling_price - purchase_price

theorem markup_amount : markup = 34.80 := by
  sorry

end markup_amount_l122_122582


namespace equation_of_line_l122_122534

theorem equation_of_line 
  (slope : ℝ)
  (a1 b1 c1 a2 b2 c2 : ℝ)
  (h_slope : slope = 2)
  (h_line1 : a1 = 3 ∧ b1 = 4 ∧ c1 = -5)
  (h_line2 : a2 = 3 ∧ b2 = -4 ∧ c2 = -13) 
  : ∃ (a b c : ℝ), (a = 2 ∧ b = -1 ∧ c = -7) ∧ 
    (∀ x y : ℝ, (a1 * x + b1 * y + c1 = 0) ∧ (a2 * x + b2 * y + c2 = 0) → (a * x + b * y + c = 0)) :=
by
  sorry

end equation_of_line_l122_122534


namespace min_value_of_z_l122_122664

theorem min_value_of_z : ∀ x : ℝ, ∃ z : ℝ, z = x^2 + 16 * x + 20 ∧ (∀ y : ℝ, y = x^2 + 16 * x + 20 → z ≤ y) → z = -44 := 
by
  sorry

end min_value_of_z_l122_122664


namespace sqrt_of_9_eq_pm_3_l122_122652

theorem sqrt_of_9_eq_pm_3 : (∃ x : ℤ, x * x = 9) → (∃ x : ℤ, x = 3 ∨ x = -3) :=
by
  sorry

end sqrt_of_9_eq_pm_3_l122_122652


namespace solutions_to_cube_eq_27_l122_122144

theorem solutions_to_cube_eq_27 (z : ℂ) : 
  (z^3 = 27) ↔ (z = 3 ∨ z = (Complex.mk (-3 / 2) (3 * Real.sqrt 3 / 2)) ∨ z = (Complex.mk (-3 / 2) (-3 * Real.sqrt 3 / 2))) :=
by sorry

end solutions_to_cube_eq_27_l122_122144


namespace find_speed_of_goods_train_l122_122033

variable (v : ℕ) -- Speed of the goods train in km/h

theorem find_speed_of_goods_train
  (h1 : 0 < v) 
  (h2 : 6 * v + 4 * 90 = 10 * v) :
  v = 36 :=
by
  sorry

end find_speed_of_goods_train_l122_122033


namespace num_possible_bases_l122_122530

theorem num_possible_bases (b : ℕ) (h1 : b ≥ 2) (h2 : b^3 ≤ 256) (h3 : 256 < b^4) : ∃ n : ℕ, n = 2 :=
by
  sorry

end num_possible_bases_l122_122530


namespace prob_3_tails_in_8_flips_l122_122102

def unfair_coin_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (Nat.choose n k) * (p^k) * ((1 - p)^(n - k))

def probability_of_3_tails : ℚ :=
  unfair_coin_probability 8 3 (2/3)

theorem prob_3_tails_in_8_flips :
  probability_of_3_tails = 448 / 6561 :=
by
  sorry

end prob_3_tails_in_8_flips_l122_122102


namespace last_two_digits_of_7_pow_2023_l122_122191

theorem last_two_digits_of_7_pow_2023 : (7 ^ 2023) % 100 = 43 := by
  sorry

end last_two_digits_of_7_pow_2023_l122_122191


namespace man_speed_was_5_kmph_l122_122721

theorem man_speed_was_5_kmph (time_in_minutes : ℕ) (distance_in_km : ℝ)
  (h_time : time_in_minutes = 30)
  (h_distance : distance_in_km = 2.5) :
  (distance_in_km / (time_in_minutes / 60 : ℝ) = 5) :=
by
  sorry

end man_speed_was_5_kmph_l122_122721


namespace estimate_greater_than_exact_l122_122759

namespace NasreenRounding

variables (a b c d a' b' c' d' : ℕ)

-- Conditions: a, b, c, and d are large positive integers.
-- Definitions for rounding up and down
def round_up (n : ℕ) : ℕ := n + 1  -- Simplified model for rounding up
def round_down (n : ℕ) : ℕ := n - 1  -- Simplified model for rounding down

-- Conditions: a', b', c', and d' are the rounded values of a, b, c, and d respectively.
variable (h_round_a_up : a' = round_up a)
variable (h_round_b_down : b' = round_down b)
variable (h_round_c_down : c' = round_down c)
variable (h_round_d_down : d' = round_down d)

-- Question: Show that the estimate is greater than the original
theorem estimate_greater_than_exact :
  (a' / b' - c' * d') > (a / b - c * d) :=
sorry

end NasreenRounding

end estimate_greater_than_exact_l122_122759


namespace correct_calculation_l122_122808

theorem correct_calculation :
  (∃ (x y : ℝ), 5 * x + 2 * y ≠ 7 * x * y) ∧
  (∃ (x : ℝ), 3 * x - 2 * x ≠ 1) ∧
  (∃ (x : ℝ), x^2 + x^5 ≠ x^7) →
  (∀ (x y : ℝ), 3 * x^2 * y - 4 * y * x^2 = -x^2 * y) :=
by
  sorry

end correct_calculation_l122_122808


namespace max_value_of_a_l122_122278

theorem max_value_of_a
  (a b c : ℝ)
  (h1 : a + b + c = 7)
  (h2 : a * b + a * c + b * c = 12) :
  a ≤ (7 + Real.sqrt 46) / 3 :=
sorry

example 
  (a b c : ℝ)
  (h1 : a + b + c = 7)
  (h2 : a * b + a * c + b * c = 12) : 
  (7 - Real.sqrt 46) / 3 ≤ a :=
sorry

end max_value_of_a_l122_122278


namespace iron_needed_for_hydrogen_l122_122502

-- Conditions of the problem
def reaction (Fe H₂SO₄ FeSO₄ H₂ : ℕ) : Prop :=
  Fe + H₂SO₄ = FeSO₄ + H₂

-- Given data
def balanced_equation : Prop :=
  reaction 1 1 1 1
 
def produced_hydrogen : ℕ := 2
def produced_from_sulfuric_acid : ℕ := 2
def needed_iron : ℕ := 2

-- Problem statement to be proved
theorem iron_needed_for_hydrogen (H₂SO₄ H₂ : ℕ) (h1 : produced_hydrogen = H₂) (h2 : produced_from_sulfuric_acid = H₂SO₄) (balanced_eq : balanced_equation) :
  needed_iron = 2 := by
sorry

end iron_needed_for_hydrogen_l122_122502


namespace cannot_determine_right_triangle_l122_122351

def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

def two_angles_complementary (α β : ℝ) : Prop :=
  α + β = 90

def exterior_angle_is_right (γ : ℝ) : Prop :=
  γ = 90

theorem cannot_determine_right_triangle :
  ¬ (∃ (a b c : ℝ), a = 1 ∧ b = 1 ∧ c = 2 ∧ is_right_triangle a b c) :=
by sorry

end cannot_determine_right_triangle_l122_122351


namespace second_pipe_fill_time_l122_122271

theorem second_pipe_fill_time
  (rate1: ℝ) (rate_outlet: ℝ) (combined_time: ℝ)
  (h1: rate1 = 1 / 18)
  (h2: rate_outlet = 1 / 45)
  (h_combined: combined_time = 0.05):
  ∃ (x: ℝ), (1 / x) = 60 :=
by
  sorry

end second_pipe_fill_time_l122_122271


namespace matilda_jellybeans_l122_122827

/-- Suppose Matilda has half as many jellybeans as Matt.
    Suppose Matt has ten times as many jellybeans as Steve.
    Suppose Steve has 84 jellybeans.
    Then Matilda has 420 jellybeans. -/
theorem matilda_jellybeans
    (matilda_jellybeans : ℕ)
    (matt_jellybeans : ℕ)
    (steve_jellybeans : ℕ)
    (h1 : matilda_jellybeans = matt_jellybeans / 2)
    (h2 : matt_jellybeans = 10 * steve_jellybeans)
    (h3 : steve_jellybeans = 84) : matilda_jellybeans = 420 := 
sorry

end matilda_jellybeans_l122_122827


namespace simplify_trig_identity_l122_122444

theorem simplify_trig_identity (α β : ℝ) : 
  (Real.cos (α + β) * Real.cos β + Real.sin (α + β) * Real.sin β) = Real.cos α :=
by
  sorry

end simplify_trig_identity_l122_122444


namespace proposition_check_l122_122670

variable (P : ℕ → Prop)

theorem proposition_check 
  (h : ∀ k : ℕ, ¬ P (k + 1) → ¬ P k)
  (h2012 : P 2012) : P 2013 :=
by
  sorry

end proposition_check_l122_122670


namespace third_consecutive_odd_integer_l122_122651

theorem third_consecutive_odd_integer (x : ℤ) (h : 3 * x = 2 * (x + 4) + 3) : x + 4 = 15 :=
sorry

end third_consecutive_odd_integer_l122_122651


namespace expression_for_f_when_x_lt_0_l122_122223

noncomputable section

variable (f : ℝ → ℝ)

theorem expression_for_f_when_x_lt_0
  (hf_neg : ∀ x : ℝ, f (-x) = -f x)
  (hf_pos : ∀ x : ℝ, x > 0 → f x = x * abs (x - 2)) :
  ∀ x : ℝ, x < 0 → f x = x * abs (x + 2) :=
by
  sorry

end expression_for_f_when_x_lt_0_l122_122223


namespace least_positive_integer_n_l122_122857

theorem least_positive_integer_n (n : ℕ) (hn : n = 10) :
  (2:ℝ)^(1 / 5 * (n * (n + 1) / 2)) > 1000 :=
by
  sorry

end least_positive_integer_n_l122_122857


namespace percentage_le_29_l122_122516

def sample_size : ℕ := 100
def freq_17_19 : ℕ := 1
def freq_19_21 : ℕ := 1
def freq_21_23 : ℕ := 3
def freq_23_25 : ℕ := 3
def freq_25_27 : ℕ := 18
def freq_27_29 : ℕ := 16
def freq_29_31 : ℕ := 28
def freq_31_33 : ℕ := 30

theorem percentage_le_29 : (freq_17_19 + freq_19_21 + freq_21_23 + freq_23_25 + freq_25_27 + freq_27_29) * 100 / sample_size = 42 :=
by
  sorry

end percentage_le_29_l122_122516


namespace overtime_hours_l122_122469

theorem overtime_hours (regular_rate: ℝ) (regular_hours: ℝ) (total_payment: ℝ) (overtime_rate_multiplier: ℝ) (overtime_hours: ℝ):
  regular_rate = 3 → regular_hours = 40 → total_payment = 198 → overtime_rate_multiplier = 2 → 
  overtime_hours = (total_payment - (regular_rate * regular_hours)) / (regular_rate * overtime_rate_multiplier) →
  overtime_hours = 13 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end overtime_hours_l122_122469


namespace garden_perimeter_l122_122308

theorem garden_perimeter (w l : ℕ) (garden_width : ℕ) (garden_perimeter : ℕ)
  (garden_area playground_length playground_width : ℕ)
  (h1 : garden_width = 16)
  (h2 : playground_length = 16)
  (h3 : garden_area = 16 * l)
  (h4 : playground_area = w * playground_length)
  (h5 : garden_area = playground_area)
  (h6 : garden_perimeter = 2 * l + 2 * garden_width)
  (h7 : garden_perimeter = 56):
  l = 12 :=
by
  sorry

end garden_perimeter_l122_122308


namespace mina_crafts_total_l122_122174

theorem mina_crafts_total :
  let a₁ := 3
  let d := 4
  let n := 10
  let crafts_sold_on_day (d: ℕ) := a₁ + (d - 1) * d
  let S (n: ℕ) := (n * (2 * a₁ + (n - 1) * d)) / 2
  S n = 210 :=
by
  sorry

end mina_crafts_total_l122_122174


namespace value_of_m_l122_122246

theorem value_of_m (m : ℝ) (h : ∀ x : ℝ, 0 < x → x < 2 → - (1 / 2) * x^2 + 2 * x ≤ m * x) :
  m = 1 :=
sorry

end value_of_m_l122_122246


namespace triangle_is_isosceles_l122_122417

theorem triangle_is_isosceles 
  (A B C : ℝ)
  (h : 0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π)
  (h_triangle : A + B + C = π)
  (h_condition : (Real.sin B) * (Real.sin C) = (Real.cos (A / 2)) ^ 2) :
  (B = C) :=
sorry

end triangle_is_isosceles_l122_122417


namespace correct_statement_l122_122946

theorem correct_statement (a b : ℚ) :
  (|a| = b → a = b) ∧ (|a| > |b| → a > b) ∧ (|a| > b → |a| > |b|) ∧ (|a| = b → a^2 = (-b)^2) ↔ 
  (true ∧ false ∧ false ∧ true) :=
by
  sorry

end correct_statement_l122_122946


namespace percentage_of_profits_to_revenues_l122_122588

theorem percentage_of_profits_to_revenues (R P : ℝ) (h1 : 0.7 * R = R - 0.3 * R) (h2 : 0.105 * R = 0.15 * (0.7 * R)) (h3 : 0.105 * R = 1.0499999999999999 * P) :
  (P / R) * 100 = 10 :=
by
  sorry

end percentage_of_profits_to_revenues_l122_122588


namespace every_positive_integer_has_good_multiple_l122_122795

def is_good (n : ℕ) : Prop :=
  ∃ (D : Finset ℕ), (D.sum id = n) ∧ (1 ∈ D) ∧ (∀ d ∈ D, d ∣ n)

theorem every_positive_integer_has_good_multiple (n : ℕ) (hn : n > 0) : ∃ m : ℕ, (m % n = 0) ∧ is_good m :=
  sorry

end every_positive_integer_has_good_multiple_l122_122795


namespace day_after_60_days_is_monday_l122_122606

theorem day_after_60_days_is_monday
    (birthday_is_thursday : ∃ d : ℕ, d % 7 = 0) :
    ∃ d : ℕ, (d + 60) % 7 = 4 :=
by
  -- Proof steps are omitted here
  sorry

end day_after_60_days_is_monday_l122_122606


namespace platform_length_l122_122257

/-- Mathematical proof problem:
The problem is to prove that given the train's length, time taken to cross a signal pole and 
time taken to cross a platform, the length of the platform is 525 meters.
-/
theorem platform_length 
    (train_length : ℕ) (time_pole : ℕ) (time_platform : ℕ) (P : ℕ) 
    (h_train_length : train_length = 450) (h_time_pole : time_pole = 18) 
    (h_time_platform : time_platform = 39) (h_P : P = 525) : 
    P = 525 := 
  sorry

end platform_length_l122_122257


namespace cupcakes_total_l122_122866

theorem cupcakes_total (initially_made : ℕ) (sold : ℕ) (newly_made : ℕ) (initially_made_eq : initially_made = 42) (sold_eq : sold = 22) (newly_made_eq : newly_made = 39) : initially_made - sold + newly_made = 59 :=
by
  sorry

end cupcakes_total_l122_122866


namespace oranges_per_child_l122_122929

theorem oranges_per_child (children oranges : ℕ) (h1 : children = 4) (h2 : oranges = 12) : oranges / children = 3 := by
  sorry

end oranges_per_child_l122_122929


namespace range_of_a_l122_122229

noncomputable def f (a x : ℝ) : ℝ := a * x ^ 2 - 1

def is_fixed_point (a x : ℝ) : Prop := f a x = x

def is_stable_point (a x : ℝ) : Prop := f a (f a x) = x

def are_equal_sets (a : ℝ) : Prop :=
  {x : ℝ | is_fixed_point a x} = {x : ℝ | is_stable_point a x}

theorem range_of_a (a : ℝ) (h : are_equal_sets a) : - (1 / 4) ≤ a ∧ a ≤ 3 / 4 := 
by
  sorry

end range_of_a_l122_122229


namespace people_own_pets_at_least_l122_122391

-- Definitions based on given conditions
def people_owning_only_dogs : ℕ := 15
def people_owning_only_cats : ℕ := 10
def people_owning_only_cats_and_dogs : ℕ := 5
def people_owning_cats_dogs_snakes : ℕ := 3
def total_snakes : ℕ := 59

-- Theorem statement to prove the total number of people owning pets
theorem people_own_pets_at_least : 
  people_owning_only_dogs + people_owning_only_cats + people_owning_only_cats_and_dogs + people_owning_cats_dogs_snakes ≥ 33 :=
by {
  -- Proof steps will go here
  sorry
}

end people_own_pets_at_least_l122_122391


namespace part1_part2_part3_l122_122581

noncomputable def A : Set ℝ := { x | x ≥ 1 ∨ x ≤ -3 }
noncomputable def B : Set ℝ := { x | -4 < x ∧ x < 0 }
noncomputable def C : Set ℝ := { x | x ≤ -4 ∨ x ≥ 0 }

theorem part1 : A ∩ B = { x | -4 < x ∧ x ≤ -3 } := 
by { sorry }

theorem part2 : A ∪ B = { x | x < 0 ∨ x ≥ 1 } := 
by { sorry }

theorem part3 : A ∪ C = { x | x ≤ -3 ∨ x ≥ 0 } := 
by { sorry }

end part1_part2_part3_l122_122581


namespace intersection_complement_N_M_eq_singleton_two_l122_122812

def M : Set ℝ := {y | y ≥ 2}
def N : Set ℝ := {x | x > 2}
def C_R_N : Set ℝ := {x | x ≤ 2}

theorem intersection_complement_N_M_eq_singleton_two :
  (C_R_N ∩ M = {2}) :=
by
  sorry

end intersection_complement_N_M_eq_singleton_two_l122_122812


namespace chess_tournament_participants_l122_122703

theorem chess_tournament_participants (n : ℕ) (h : n * (n - 1) / 2 = 120) : n = 16 :=
sorry

end chess_tournament_participants_l122_122703


namespace box_height_l122_122068

variables (length width : ℕ) (cube_volume cubes total_volume : ℕ)
variable (height : ℕ)

theorem box_height :
  length = 12 →
  width = 16 →
  cube_volume = 3 →
  cubes = 384 →
  total_volume = cubes * cube_volume →
  total_volume = length * width * height →
  height = 6 :=
by
  intros
  sorry

end box_height_l122_122068


namespace allowable_rectangular_formations_count_l122_122512

theorem allowable_rectangular_formations_count (s t f : ℕ) 
  (h1 : s * t = 240)
  (h2 : Nat.Prime s)
  (h3 : 8 ≤ t ∧ t ≤ 30)
  (h4 : f ≤ 8)
  : f = 0 :=
sorry

end allowable_rectangular_formations_count_l122_122512


namespace correct_range_of_x_l122_122494

variable {x : ℝ}

noncomputable def isosceles_triangle (x y : ℝ) : Prop :=
  let perimeter := 2 * y + x
  let relationship := y = - (1/2) * x + 8
  perimeter = 16 ∧ relationship

theorem correct_range_of_x (x y : ℝ) (h : isosceles_triangle x y) : 0 < x ∧ x < 8 :=
by
  -- The proof of the theorem is omitted
  sorry

end correct_range_of_x_l122_122494


namespace school_election_votes_l122_122912

theorem school_election_votes (E S R L : ℕ)
  (h1 : E = 2 * S)
  (h2 : E = 4 * R)
  (h3 : S = 5 * R)
  (h4 : S = 3 * L)
  (h5 : R = 16) :
  E = 64 ∧ S = 80 ∧ R = 16 ∧ L = 27 := by
  sorry

end school_election_votes_l122_122912


namespace proof_l122_122389

noncomputable def problem_statement (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ (∀ x : ℝ, |x + a| + |x - b| + c ≥ 4)

theorem proof (a b c : ℝ) (h : problem_statement a b c) :
  a + b + c = 4 ∧ (∀ x : ℝ, 1 / a + 4 / b + 9 / c ≥ 9) :=
by
  sorry

end proof_l122_122389


namespace min_distance_origin_to_line_l122_122513

theorem min_distance_origin_to_line 
  (x y : ℝ) 
  (h : x + y = 4) : 
  ∃ P : ℝ, P = 2 * Real.sqrt 2 ∧ 
    (∀ Q : ℝ, Q = Real.sqrt (x^2 + y^2) → P ≤ Q) :=
by
  sorry

end min_distance_origin_to_line_l122_122513


namespace minimize_y_l122_122964

theorem minimize_y (a b : ℝ) : 
  ∃ x, x = (a + b) / 2 ∧ ∀ x', ((x' - a)^3 + (x' - b)^3) ≥ ((x - a)^3 + (x - b)^3) :=
sorry

end minimize_y_l122_122964


namespace sum_of_M_l122_122416

theorem sum_of_M (x y z w M : ℕ) (hxw : w = x + y + z) (hM : M = x * y * z * w) (hM_cond : M = 12 * (x + y + z + w)) :
  ∃ sum_M, sum_M = 2208 :=
by 
  sorry

end sum_of_M_l122_122416


namespace find_x_l122_122832

noncomputable def positive_real (a : ℝ) := 0 < a

theorem find_x (x y : ℝ) (h1 : positive_real x) (h2 : positive_real y)
  (h3 : 6 * x^3 + 12 * x^2 * y = 2 * x^4 + 3 * x^3 * y)
  (h4 : x + y = 3) : x = 2 :=
by
  sorry

end find_x_l122_122832


namespace speed_including_stoppages_l122_122453

-- Definitions
def speed_excluding_stoppages : ℤ := 50 -- kmph
def stoppage_time_per_hour : ℕ := 24 -- minutes

-- Theorem to prove the speed of the train including stoppages
theorem speed_including_stoppages (h1 : speed_excluding_stoppages = 50)
                                  (h2 : stoppage_time_per_hour = 24) :
  ∃ s : ℤ, s = 30 := 
sorry

end speed_including_stoppages_l122_122453


namespace determinant_difference_l122_122432

namespace MatrixDeterminantProblem

open Matrix

variables {R : Type*} [CommRing R]

theorem determinant_difference (a b c d : R) 
  (h : det ![![a, b], ![c, d]] = 15) :
  det ![![3 * a, 3 * b], ![3 * c, 3 * d]] - 
  det ![![3 * b, 3 * a], ![3 * d, 3 * c]] = 270 := 
by
  sorry

end MatrixDeterminantProblem

end determinant_difference_l122_122432


namespace minimal_perimeter_triangle_l122_122087

noncomputable def cos_P : ℚ := 3 / 5
noncomputable def cos_Q : ℚ := 24 / 25
noncomputable def cos_R : ℚ := -1 / 5

theorem minimal_perimeter_triangle
  (P Q R : ℝ) (a b c : ℕ)
  (h0 : a^2 + b^2 + c^2 - 2 * a * b * cos_P - 2 * b * c * cos_Q - 2 * c * a * cos_R = 0)
  (h1 : cos_P^2 + (1 - cos_P^2) = 1)
  (h2 : cos_Q^2 + (1 - cos_Q^2) = 1)
  (h3 : cos_R^2 + (1 - cos_R^2) = 1) :
  a + b + c = 47 :=
sorry

end minimal_perimeter_triangle_l122_122087


namespace repeating_decimal_as_fraction_l122_122542

theorem repeating_decimal_as_fraction :
  ∃ x : ℝ, x = 7.45 ∧ (100 * x - x = 738) → x = 82 / 11 :=
by
  sorry

end repeating_decimal_as_fraction_l122_122542


namespace angle_B_range_l122_122330

def range_of_angle_B (a b c : ℝ) (A B C : ℝ) : Prop :=
  (0 < B ∧ B ≤ Real.pi / 3)

theorem angle_B_range
  (a b c A B C : ℝ)
  (h1 : b^2 = a * c)
  (h2 : A + B + C = π)
  (h3 : a > 0)
  (h4 : b > 0)
  (h5 : c > 0)
  (h6 : a + b > c)
  (h7 : a + c > b)
  (h8 : b + c > a) :
  range_of_angle_B a b c A B C :=
sorry

end angle_B_range_l122_122330


namespace fractional_eq_a_range_l122_122739

theorem fractional_eq_a_range (a : ℝ) :
  (∃ x : ℝ, (a / (x + 2) = 1 - 3 / (x + 2)) ∧ (x < 0)) ↔ (a < -1 ∧ a ≠ -3) := by
  sorry

end fractional_eq_a_range_l122_122739


namespace scientific_notation_of_100000000_l122_122891

theorem scientific_notation_of_100000000 :
  100000000 = 1 * 10^8 :=
sorry

end scientific_notation_of_100000000_l122_122891


namespace muffins_sold_in_afternoon_l122_122030

variable (total_muffins : ℕ)
variable (morning_muffins : ℕ)
variable (remaining_muffins : ℕ)

theorem muffins_sold_in_afternoon 
  (h1 : total_muffins = 20) 
  (h2 : morning_muffins = 12) 
  (h3 : remaining_muffins = 4) : 
  (total_muffins - remaining_muffins - morning_muffins) = 4 := 
by
  sorry

end muffins_sold_in_afternoon_l122_122030


namespace erasers_left_in_the_box_l122_122225

-- Conditions expressed as definitions
def E0 : ℕ := 320
def E1 : ℕ := E0 - 67
def E2 : ℕ := E1 - 126
def E3 : ℕ := E2 + 30

-- Proof problem statement
theorem erasers_left_in_the_box : E3 = 157 := 
by sorry

end erasers_left_in_the_box_l122_122225


namespace problem1_problem2_problem3_l122_122227

-- Problem 1
theorem problem1 : -2.8 + (-3.6) + 3 - (-3.6) = 0.2 := 
by
  sorry

-- Problem 2
theorem problem2 : (-4) ^ 2010 * (-0.25) ^ 2009 + (-12) * (1 / 3 - 3 / 4 + 5 / 6) = -9 := 
by
  sorry

-- Problem 3
theorem problem3 : 13 * (16/60 : ℝ) * 5 - 19 * (12/60 : ℝ) / 6 = 13 * (8/60 : ℝ) + 50 := 
by
  sorry

end problem1_problem2_problem3_l122_122227


namespace number_of_valid_pairs_l122_122158

theorem number_of_valid_pairs :
  (∃ (count : ℕ), count = 280 ∧
    (∃ (m n : ℕ),
      1 ≤ m ∧ m ≤ 2899 ∧
      5^n < 2^m ∧ 2^m < 2^(m+3) ∧ 2^(m+3) < 5^(n+1))) :=
sorry

end number_of_valid_pairs_l122_122158


namespace oliver_shirts_not_washed_l122_122792

theorem oliver_shirts_not_washed :
  let short_sleeve_shirts := 39
  let long_sleeve_shirts := 47
  let total_shirts := short_sleeve_shirts + long_sleeve_shirts
  let washed_shirts := 20
  let not_washed_shirts := total_shirts - washed_shirts
  not_washed_shirts = 66 := by
  sorry

end oliver_shirts_not_washed_l122_122792


namespace birds_joined_l122_122003

variable (initialBirds : ℕ) (totalBirds : ℕ)

theorem birds_joined (h1 : initialBirds = 2) (h2 : totalBirds = 6) : (totalBirds - initialBirds) = 4 :=
by
  sorry

end birds_joined_l122_122003


namespace floor_equation_solution_l122_122023

theorem floor_equation_solution (a b : ℕ) (a_pos : 0 < a) (b_pos : 0 < b) :
  (⌊ (a^2 : ℝ) / b ⌋ + ⌊ (b^2 : ℝ) / a ⌋ = ⌊ (a^2 + b^2 : ℝ) / (a * b) ⌋ + a * b) ↔
    (∃ n : ℕ, a = n ∧ b = n^2 + 1) ∨ (∃ n : ℕ, a = n^2 + 1 ∧ b = n) :=
sorry

end floor_equation_solution_l122_122023


namespace pq_necessary_not_sufficient_l122_122187

theorem pq_necessary_not_sufficient (p q : Prop) : (p ∨ q) → (p ∧ q) ↔ false :=
by sorry

end pq_necessary_not_sufficient_l122_122187


namespace binomial_expansion_sum_l122_122295

theorem binomial_expansion_sum (a : ℝ) (a₁ a₂ a₃ a₄ a₅ : ℝ)
  (h₁ : (a * x - 1)^5 = a + a₁ * x + a₂ * x^2 + a₃ * x^3 + a₄ * x^4 + a₅ * x^5)
  (h₂ : a₃ = 80) :
  a + a₁ + a₂ + a₃ + a₄ + a₅ = 1 :=
sorry

end binomial_expansion_sum_l122_122295


namespace difference_of_squares_divisibility_l122_122537

theorem difference_of_squares_divisibility (a b : ℤ) :
  ∃ m : ℤ, (2 * a + 3) ^ 2 - (2 * b + 1) ^ 2 = 8 * m ∧ 
           ¬∃ n : ℤ, (2 * a + 3) ^ 2 - (2 * b + 1) ^ 2 = 16 * n :=
by
  sorry

end difference_of_squares_divisibility_l122_122537


namespace trains_same_distance_at_meeting_l122_122960

theorem trains_same_distance_at_meeting
  (d v : ℝ) (h_d : 0 < d) (h_v : 0 < v) :
  ∃ t : ℝ, v * t + v * (t - 1) = d ∧ 
  v * t = (d + v) / 2 ∧ 
  d - (v * (t - 1)) = (d + v) / 2 :=
by
  sorry

end trains_same_distance_at_meeting_l122_122960


namespace ratio_cost_to_marked_price_l122_122908

theorem ratio_cost_to_marked_price (p : ℝ) (hp : p > 0) :
  let selling_price := (3 / 4) * p
  let cost_price := (5 / 6) * selling_price
  cost_price / p = 5 / 8 :=
by 
  sorry

end ratio_cost_to_marked_price_l122_122908


namespace k_is_perfect_square_l122_122029

theorem k_is_perfect_square (m n : ℕ) (h_pos_m : 0 < m) (h_pos_n : 0 < n) (k : ℕ)
  (h_k : k = (m + n)^2 / (4 * m * (m - n)^2 + 4)) 
  (h_int_k : k * (4 * m * (m - n)^2 + 4) = (m + n)^2) :
  ∃ x : ℕ, k = x^2 := 
sorry

end k_is_perfect_square_l122_122029


namespace find_number_l122_122310

theorem find_number (x : ℝ) (h : x / 14.5 = 171) : x = 2479.5 :=
by
  sorry

end find_number_l122_122310


namespace fraction_addition_solution_is_six_l122_122822

theorem fraction_addition_solution_is_six :
  (1 / 9) + (1 / 18) = 1 / 6 := 
sorry

end fraction_addition_solution_is_six_l122_122822


namespace radius_of_sphere_inscribed_in_box_l122_122093

theorem radius_of_sphere_inscribed_in_box (a b c s : ℝ)
  (h1 : a + b + c = 42)
  (h2 : 2 * (a * b + b * c + c * a) = 576)
  (h3 : (2 * s)^2 = a^2 + b^2 + c^2) :
  s = 3 * Real.sqrt 33 :=
by sorry

end radius_of_sphere_inscribed_in_box_l122_122093


namespace workers_time_together_l122_122913

theorem workers_time_together (T : ℝ) (h1 : ∀ t : ℝ, (T + 8) = t → 1 / t = 1 / (T + 8))
                                (h2 : ∀ t : ℝ, (T + 4.5) = t → 1 / t = 1 / (T + 4.5))
                                (h3 : 1 / (T + 8) + 1 / (T + 4.5) = 1 / T) : T = 6 :=
sorry

end workers_time_together_l122_122913


namespace find_larger_number_l122_122847

-- Define the conditions
variables (L S : ℕ)
axiom condition1 : L - S = 1365
axiom condition2 : L = 6 * S + 35

-- State the theorem
theorem find_larger_number : L = 1631 :=
by
  sorry

end find_larger_number_l122_122847


namespace neg_existence_of_ge_impl_universal_lt_l122_122047

theorem neg_existence_of_ge_impl_universal_lt : (¬ ∃ x : ℕ, x^2 ≥ x) ↔ ∀ x : ℕ, x^2 < x := 
sorry

end neg_existence_of_ge_impl_universal_lt_l122_122047


namespace ball_travel_distance_l122_122708

theorem ball_travel_distance 
    (initial_height : ℕ)
    (half : ℕ → ℕ)
    (num_bounces : ℕ)
    (height_after_bounce : ℕ → ℕ)
    (total_distance : ℕ) :
    initial_height = 16 ∧ 
    (∀ n, half n = n / 2) ∧ 
    num_bounces = 4 ∧ 
    (height_after_bounce 0 = initial_height) ∧
    (∀ n, height_after_bounce (n + 1) = half (height_after_bounce n))
→ total_distance = 46 :=
by
  sorry

end ball_travel_distance_l122_122708


namespace circle_radius_l122_122472

theorem circle_radius {s r : ℝ} (h : 4 * s = π * r^2) (h1 : 2 * r = s * Real.sqrt 2) : 
  r = 4 * Real.sqrt 2 / π :=
by
  sorry

end circle_radius_l122_122472


namespace length_of_PQ_l122_122490

theorem length_of_PQ (R P Q : ℝ × ℝ) (hR : R = (10, 8))
(hP_line1 : ∃ p : ℝ, P = (p, 24 * p / 7))
(hQ_line2 : ∃ q : ℝ, Q = (q, 5 * q / 13))
(h_mid : ∃ (p q : ℝ), R = ((p + q) / 2, (24 * p / 14 + 5 * q / 26) / 2))
(answer_eq : ∃ (a b : ℕ), PQ_length = a / b ∧ a.gcd b = 1 ∧ a + b = 4925) : 
∃ a b : ℕ, a + b = 4925 := sorry

end length_of_PQ_l122_122490


namespace elena_deductions_in_cents_l122_122508

-- Definitions based on the conditions
def cents_per_dollar : ℕ := 100
def hourly_wage_in_dollars : ℕ := 25
def hourly_wage_in_cents : ℕ := hourly_wage_in_dollars * cents_per_dollar
def tax_rate : ℚ := 0.02
def health_benefit_rate : ℚ := 0.015

-- The problem to prove
theorem elena_deductions_in_cents:
  (tax_rate * hourly_wage_in_cents) + (health_benefit_rate * hourly_wage_in_cents) = 87.5 := 
by
  sorry

end elena_deductions_in_cents_l122_122508


namespace average_side_length_of_squares_l122_122059

theorem average_side_length_of_squares (a1 a2 a3 a4 : ℕ) 
(h1 : a1 = 36) (h2 : a2 = 64) (h3 : a3 = 100) (h4 : a4 = 144) :
(Real.sqrt a1 + Real.sqrt a2 + Real.sqrt a3 + Real.sqrt a4) / 4 = 9 := 
by
  sorry

end average_side_length_of_squares_l122_122059


namespace product_mnp_l122_122862

theorem product_mnp (m n p : ℕ) (b x z c : ℂ) (h1 : b^8 * x * z - b^7 * z - b^6 * x = b^5 * (c^5 - 1)) 
  (h2 : (b^m * x - b^n) * (b^p * z - b^3) = b^5 * c^5) : m * n * p = 30 :=
sorry

end product_mnp_l122_122862


namespace second_grade_students_sampled_l122_122509

-- Definitions corresponding to conditions in a)
def total_students := 2000
def mountain_climbing_fraction := 2 / 5
def running_ratios := (2, 3, 5)
def sample_size := 200

-- Calculation of total running participants based on ratio
def total_running_students :=
  total_students * (1 - mountain_climbing_fraction)

def a := 2 * (total_running_students / (2 + 3 + 5))
def b := 3 * (total_running_students / (2 + 3 + 5))
def c := 5 * (total_running_students / (2 + 3 + 5))

def running_sample_size := sample_size * (3 / 5) --since the ratio is 3:5

-- The statement to prove
theorem second_grade_students_sampled : running_sample_size * (3 / (2+3+5)) = 36 :=
by
  sorry

end second_grade_students_sampled_l122_122509


namespace percy_bound_longer_martha_step_l122_122323

theorem percy_bound_longer_martha_step (steps_per_gap_martha: ℕ) (bounds_per_gap_percy: ℕ)
  (gaps: ℕ) (total_distance: ℕ) 
  (step_length_martha: ℝ) (bound_length_percy: ℝ) :
  steps_per_gap_martha = 50 →
  bounds_per_gap_percy = 15 →
  gaps = 50 →
  total_distance = 10560 →
  step_length_martha = total_distance / (steps_per_gap_martha * gaps) →
  bound_length_percy = total_distance / (bounds_per_gap_percy * gaps) →
  (bound_length_percy - step_length_martha) = 10 :=
by
  sorry

end percy_bound_longer_martha_step_l122_122323


namespace exponential_function_example_l122_122650

def is_exponential_function (f : ℝ → ℝ) : Prop :=
  ∃ a > 0, a ≠ 1 ∧ ∀ x, f x = a ^ x

theorem exponential_function_example : is_exponential_function (fun x => 3 ^ x) :=
by
  sorry

end exponential_function_example_l122_122650


namespace nominal_rate_of_interest_l122_122578

theorem nominal_rate_of_interest
  (EAR : ℝ)
  (n : ℕ)
  (h_EAR : EAR = 0.0609)
  (h_n : n = 2) :
  ∃ i : ℝ, (1 + i / n)^n - 1 = EAR ∧ i = 0.059 := 
by 
  sorry

end nominal_rate_of_interest_l122_122578


namespace area_ratio_l122_122823

-- Define the problem conditions
def Square (s : ℝ) := s > 0
def Rectangle (longer shorter : ℝ) := longer = 1.2 * shorter ∧ shorter = 0.8 * shorter

-- Define a function to calculate the area of square
def area_square (s : ℝ) : ℝ := s * s

-- Define a function to calculate the area of rectangle
def area_rectangle (longer shorter : ℝ) : ℝ := longer * shorter

-- State the proof problem
theorem area_ratio (s : ℝ) (h_square : Square s) :
  let longer := 1.2 * s
  let shorter := 0.8 * s
  area_rectangle longer shorter / area_square s = 24 / 25 :=
by
  sorry

end area_ratio_l122_122823


namespace five_colored_flags_l122_122696

def num_different_flags (colors total_stripes : ℕ) : ℕ :=
  Nat.choose colors total_stripes * Nat.factorial total_stripes

theorem five_colored_flags : num_different_flags 11 5 = 55440 := by
  sorry

end five_colored_flags_l122_122696


namespace probability_of_X_eq_4_l122_122262

noncomputable def probability_X_eq_4 : ℝ :=
  let total_balls := 12
  let new_balls := 9
  let old_balls := 3
  let draw := 3
  -- Number of ways to choose 2 old balls from 3
  let choose_old := Nat.choose old_balls 2
  -- Number of ways to choose 1 new ball from 9
  let choose_new := Nat.choose new_balls 1
  -- Total number of ways to choose 3 balls from 12
  let total_ways := Nat.choose total_balls draw
  -- Probability calculation
  (choose_old * choose_new) / total_ways

theorem probability_of_X_eq_4 : probability_X_eq_4 = 27 / 220 := by
  sorry

end probability_of_X_eq_4_l122_122262


namespace simplify_expression_l122_122421

theorem simplify_expression :
  (1 * 2 * a * 3 * a^2 * 4 * a^3 * 5 * a^4 * 6 * a^5) = 720 * a^15 :=
by
  sorry

end simplify_expression_l122_122421


namespace annual_interest_rate_l122_122021

variable (P : ℝ) (t : ℝ)
variable (h1 : t = 25)
variable (h2 : ∀ r : ℝ, P * 2 = P * (1 + r * t))

theorem annual_interest_rate : ∃ r : ℝ, P * 2 = P * (1 + r * t) ∧ r = 0.04 := by
  sorry

end annual_interest_rate_l122_122021


namespace four_ping_pong_four_shuttlecocks_cost_l122_122770

theorem four_ping_pong_four_shuttlecocks_cost
  (x y : ℝ)
  (h1 : 3 * x + 2 * y = 15.5)
  (h2 : 2 * x + 3 * y = 17) :
  4 * x + 4 * y = 26 :=
sorry

end four_ping_pong_four_shuttlecocks_cost_l122_122770


namespace range_of_f_l122_122232

noncomputable def f : ℝ → ℝ := sorry -- Define f appropriately

theorem range_of_f : Set.range f = {y : ℝ | 0 < y} :=
sorry

end range_of_f_l122_122232


namespace total_pupils_correct_l122_122347

-- Definitions of the number of girls and boys in each school
def girlsA := 542
def boysA := 387
def girlsB := 713
def boysB := 489
def girlsC := 628
def boysC := 361

-- Total pupils in each school
def pupilsA := girlsA + boysA
def pupilsB := girlsB + boysB
def pupilsC := girlsC + boysC

-- Total pupils across all schools
def total_pupils := pupilsA + pupilsB + pupilsC

-- The proof statement (no proof provided, hence sorry)
theorem total_pupils_correct : total_pupils = 3120 := by sorry

end total_pupils_correct_l122_122347


namespace num_valid_n_l122_122930

theorem num_valid_n (n q r : ℤ) (h₁ : 10000 ≤ n) (h₂ : n ≤ 99999)
  (h₃ : n = 50 * q + r) (h₄ : 200 ≤ q) (h₅ : q ≤ 1999)
  (h₆ : 0 ≤ r) (h₇ : r < 50) :
  (∃ (count : ℤ), count = 14400) := by
  sorry

end num_valid_n_l122_122930


namespace fraction_evaluation_l122_122831

theorem fraction_evaluation : (1 - (1 / 4)) / (1 - (1 / 3)) = (9 / 8) :=
by
  sorry

end fraction_evaluation_l122_122831


namespace sequence_length_l122_122465

theorem sequence_length (a d n : ℕ) (h1 : a = 3) (h2 : d = 5) (h3: 3 + (n-1) * d = 3008) : n = 602 := 
by
  sorry

end sequence_length_l122_122465


namespace jangshe_clothing_cost_l122_122425

theorem jangshe_clothing_cost
  (total_spent : ℝ)
  (untaxed_piece1 : ℝ)
  (untaxed_piece2 : ℝ)
  (total_pieces : ℕ)
  (remaining_pieces : ℕ)
  (remaining_pieces_price : ℝ)
  (sales_tax : ℝ)
  (price_multiple_of_five : ℝ) :
  total_spent = 610 ∧
  untaxed_piece1 = 49 ∧
  untaxed_piece2 = 81 ∧
  total_pieces = 7 ∧
  remaining_pieces = 5 ∧
  sales_tax = 0.10 ∧
  (∃ k : ℕ, remaining_pieces_price = k * 5) →
  remaining_pieces_price / remaining_pieces = 87 :=
by
  sorry

end jangshe_clothing_cost_l122_122425


namespace hexagon_side_lengths_l122_122459

open Nat

/-- Define two sides AB and BC of a hexagon with their given lengths -/
structure Hexagon :=
  (AB BC AD BE CF DE: ℕ)
  (distinct_lengths : AB ≠ BC ∧ (AB = 7 ∧ BC = 8))
  (total_perimeter : AB + BC + AD + BE + CF + DE = 46)

-- Define a theorem to prove the number of sides measuring 8 units
theorem hexagon_side_lengths (h: Hexagon) :
  ∃ (n : ℕ), n = 4 ∧ n * 8 + (6 - n) * 7 = 46 :=
by
  -- Assume the proof here
  sorry

end hexagon_side_lengths_l122_122459


namespace prime_root_condition_l122_122203

theorem prime_root_condition (p : ℕ) (hp : Nat.Prime p) :
  (∃ x y : ℤ, x ≠ y ∧ (x^2 + 2 * p * x - 240 * p = 0) ∧ (y^2 + 2 * p * y - 240 * p = 0) ∧ x*y = -240*p) → p = 5 :=
by sorry

end prime_root_condition_l122_122203


namespace proof_problem_l122_122565

open Set Real

def M : Set ℝ := { x : ℝ | ∃ y : ℝ, y = log (1 - 2 / x) }
def N : Set ℝ := { x : ℝ | ∃ y : ℝ, y = sqrt (x - 1) }

theorem proof_problem : N ∩ (U \ M) = Icc 1 2 := by
  sorry

end proof_problem_l122_122565


namespace remainder_of_f_x10_mod_f_l122_122371

def f (x : ℤ) : ℤ := x^4 + x^3 + x^2 + x + 1

theorem remainder_of_f_x10_mod_f (x : ℤ) : (f (x ^ 10)) % (f x) = 5 :=
by
  sorry

end remainder_of_f_x10_mod_f_l122_122371


namespace find_a4_l122_122888

theorem find_a4 (a : ℕ → ℕ) 
  (h1 : ∀ n, (a n + 1) / (a (n + 1) + 1) = 1 / 2) 
  (h2 : a 2 = 2) : 
  a 4 = 11 :=
sorry

end find_a4_l122_122888


namespace pencil_ratio_l122_122067

theorem pencil_ratio (B G : ℕ) (h1 : ∀ (n : ℕ), n = 20) 
  (h2 : ∀ (n : ℕ), n = 40) 
  (h3 : ∀ (n : ℕ), n = 160) 
  (h4 : G = 20 + B)
  (h5 : B + 20 + G + 40 = 160) : 
  (B / 20) = 4 := 
  by sorry

end pencil_ratio_l122_122067


namespace dot_product_two_a_plus_b_with_a_l122_122303

-- Define vector a
def a : ℝ × ℝ := (2, -1)

-- Define vector b
def b : ℝ × ℝ := (-1, 2)

-- Define the scalar multiplication of vector a by 2
def two_a : ℝ × ℝ := (2 * a.1, 2 * a.2)

-- Define the vector addition of 2a and b
def two_a_plus_b : ℝ × ℝ := (two_a.1 + b.1, two_a.2 + b.2)

-- Define dot product of two vectors
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

-- Prove that the dot product of (2 * a + b) and a equals 6
theorem dot_product_two_a_plus_b_with_a :
  dot_product two_a_plus_b a = 6 :=
by
  sorry

end dot_product_two_a_plus_b_with_a_l122_122303


namespace digit_A_value_l122_122463

theorem digit_A_value :
  ∃ (A : ℕ), A < 10 ∧ (45 % A = 0) ∧ (172 * 10 + A * 10 + 6) % 8 = 0 ∧
    ∀ (B : ℕ), B < 10 ∧ (45 % B = 0) ∧ (172 * 10 + B * 10 + 6) % 8 = 0 → B = A := sorry

end digit_A_value_l122_122463


namespace symmetric_point_coordinates_l122_122410

theorem symmetric_point_coordinates :
  ∀ (M N : ℝ × ℝ), M = (3, -4) ∧ M.fst = -N.fst ∧ M.snd = N.snd → N = (-3, -4) :=
by
  intro M N h
  sorry

end symmetric_point_coordinates_l122_122410


namespace area_of_equilateral_triangle_with_inscribed_circle_l122_122422

theorem area_of_equilateral_triangle_with_inscribed_circle 
  (r : ℝ) (A : ℝ) (area_circle_eq : A = 9 * Real.pi)
  (DEF_equilateral : ∀ {a b c : ℝ}, a = b ∧ b = c): 
  ∃ area_def : ℝ, area_def = 27 * Real.sqrt 3 :=
by
  -- proof omitted
  sorry

end area_of_equilateral_triangle_with_inscribed_circle_l122_122422


namespace algebraic_expression_value_l122_122889

variable (a : ℝ)

theorem algebraic_expression_value (h : a = Real.sqrt 2) :
  (a / (a - 1)^2) / (1 + 1 / (a - 1)) = Real.sqrt 2 + 1 :=
by
  sorry

end algebraic_expression_value_l122_122889


namespace find_x_find_y_find_p_q_r_l122_122845

-- Condition: The number on the line connecting two circles is the sum of the two numbers in the circles.

-- For part (a):
theorem find_x (a b : ℝ) (x : ℝ) (h1 : a + 4 = 13) (h2 : a + b = 10) (h3 : b + 4 = x) : x = 5 :=
by {
  -- Proof can be filled in here to show x = 5 by solving the equations.
  sorry
}

-- For part (b):
theorem find_y (w y : ℝ) (h1 : 3 * w + w = y) (h2 : 6 * w = 48) : y = 32 := 
by {
  -- Proof can be filled in here to show y = 32 by solving the equations.
  sorry
}

-- For part (c):
theorem find_p_q_r (p q r : ℝ) (h1 : p + r = 3) (h2 : p + q = 18) (h3 : q + r = 13) : p = 4 ∧ q = 14 ∧ r = -1 :=
by {
  -- Proof can be filled in here to show p = 4, q = 14, r = -1 by solving the equations.
  sorry
}

end find_x_find_y_find_p_q_r_l122_122845


namespace perpendicular_vectors_l122_122699

theorem perpendicular_vectors (b : ℝ) :
  (5 * b - 12 = 0) → b = 12 / 5 :=
by
  intro h
  sorry

end perpendicular_vectors_l122_122699


namespace actual_cost_of_article_l122_122448

theorem actual_cost_of_article (x : ℝ) (h : 0.60 * x = 1050) : x = 1750 := by
  sorry

end actual_cost_of_article_l122_122448


namespace leah_probability_of_seeing_change_l122_122062

open Set

-- Define the length of each color interval
def green_duration := 45
def yellow_duration := 5
def red_duration := 35

-- Total cycle duration
def total_cycle_duration := green_duration + yellow_duration + red_duration

-- Leah's viewing intervals
def change_intervals : Set (ℕ × ℕ) :=
  {(40, 45), (45, 50), (80, 85)}

-- Probability calculation
def favorable_time := 15
def probability_of_change := (favorable_time : ℚ) / (total_cycle_duration : ℚ)

theorem leah_probability_of_seeing_change : probability_of_change = 3 / 17 :=
by
  -- We use sorry here as we are only required to state the theorem without proof.
  sorry

end leah_probability_of_seeing_change_l122_122062


namespace total_vegetarian_is_33_l122_122373

-- Definitions of the quantities involved
def only_vegetarian : Nat := 19
def both_vegetarian_non_vegetarian : Nat := 12
def vegan_strictly_vegetarian : Nat := 3
def vegan_non_vegetarian : Nat := 2

-- The total number of people consuming vegetarian dishes
def total_vegetarian_consumers : Nat := only_vegetarian + both_vegetarian_non_vegetarian + vegan_non_vegetarian

-- Prove the number of people consuming vegetarian dishes
theorem total_vegetarian_is_33 :
  total_vegetarian_consumers = 33 :=
sorry

end total_vegetarian_is_33_l122_122373


namespace friend_reading_time_l122_122095

def my_reading_time : ℕ := 120  -- It takes me 120 minutes to read the novella

def speed_ratio : ℕ := 3  -- My friend reads three times as fast as I do

theorem friend_reading_time : my_reading_time / speed_ratio = 40 := by
  -- Proof
  sorry

end friend_reading_time_l122_122095


namespace inequality_l122_122418
-- Import the necessary libraries from Mathlib

-- Define the theorem statement
theorem inequality (a b c : ℝ) : a^2 + b^2 + c^2 ≥ a * b + b * c + c * a := 
by
  sorry

end inequality_l122_122418


namespace balanced_number_example_l122_122146

/--
A number is balanced if it is a three-digit number, all digits are different,
and it equals the sum of all possible two-digit numbers composed from its different digits.
-/
def isBalanced (n : ℕ) : Prop :=
  (n / 100 ≠ (n / 10) % 10) ∧ (n / 100 ≠ n % 10) ∧ ((n / 10) % 10 ≠ n % 10) ∧
  (n = (10 * (n / 100) + (n / 10) % 10) + (10 * (n / 100) + n % 10) +
    (10 * ((n / 10) % 10) + n / 100) + (10 * ((n / 10) % 10) + n % 10) +
    (10 * (n % 10) + n / 100) + (10 * (n % 10) + ((n / 10) % 10)))

theorem balanced_number_example : isBalanced 132 :=
  sorry

end balanced_number_example_l122_122146


namespace math_problem_l122_122238

noncomputable def problem_statement (a b c d : ℕ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  (6 - a) * (6 - b) * (6 - c) * (6 - d) = 9

theorem math_problem
  (a b c d : ℕ)
  (h1 : a ≠ b)
  (h2 : a ≠ c)
  (h3 : a ≠ d)
  (h4 : b ≠ c)
  (h5 : b ≠ d)
  (h6 : c ≠ d)
  (h7 : (6 - a) * (6 - b) * (6 - c) * (6 - d) = 9) :
  a + b + c + d = 24 :=
sorry

end math_problem_l122_122238


namespace factor_polynomial_l122_122745

theorem factor_polynomial (y : ℝ) : 
  y^6 - 64 = (y - 2) * (y + 2) * (y^2 + 2 * y + 4) * (y^2 - 2 * y + 4) :=
by
  sorry

end factor_polynomial_l122_122745


namespace cubic_yards_to_cubic_feet_l122_122112

theorem cubic_yards_to_cubic_feet (yards_to_feet: 1 = 3): 6 * 27 = 162 := by
  -- We know from the setup that:
  -- 1 cubic yard = 27 cubic feet
  -- Hence,
  -- 6 cubic yards = 6 * 27 = 162 cubic feet
  sorry

end cubic_yards_to_cubic_feet_l122_122112


namespace simplify_fraction_l122_122349

theorem simplify_fraction (i : ℂ) (h : i^2 = -1) : 
  (2 - i) / (1 + 4 * i) = -2 / 17 - (9 / 17) * i :=
by
  sorry

end simplify_fraction_l122_122349


namespace gcd_of_polynomials_l122_122899

theorem gcd_of_polynomials (b : ℤ) (h : ∃ k : ℤ, b = 2 * 5959 * k) :
  Int.gcd (4 * b^2 + 73 * b + 156) (4 * b + 15) = 1 :=
by
  sorry

end gcd_of_polynomials_l122_122899


namespace initial_students_count_eq_16_l122_122769

variable (n T : ℕ)
variable (h1 : (T:ℝ) / n = 62.5)
variable (h2 : ((T - 70):ℝ) / (n - 1) = 62.0)

theorem initial_students_count_eq_16 :
  n = 16 :=
by
  sorry

end initial_students_count_eq_16_l122_122769


namespace eval_gg3_l122_122551

def g (x : ℕ) : ℕ := 3 * x^2 + 3 * x - 2

theorem eval_gg3 : g (g 3) = 3568 :=
by 
  sorry

end eval_gg3_l122_122551


namespace cone_sphere_ratio_l122_122637

theorem cone_sphere_ratio (r h : ℝ) (π_pos : 0 < π) (r_pos : 0 < r) :
  (1/3) * π * r^2 * h = (1/3) * (4/3) * π * r^3 → h / r = 4/3 :=
by
  sorry

end cone_sphere_ratio_l122_122637


namespace range_of_f_l122_122662

open Set

noncomputable def f (x : ℝ) : ℝ := x + Real.sqrt (2 * x - 1)

theorem range_of_f : range f = Ici (1 / 2) :=
by
  sorry

end range_of_f_l122_122662


namespace solve_for_x_l122_122008

theorem solve_for_x (x : ℝ) : 7 * (4 * x + 3) - 5 = -3 * (2 - 5 * x) ↔ x = -22 / 13 := 
by 
  sorry

end solve_for_x_l122_122008


namespace quadratic_expression_l122_122905

-- Definitions of roots and their properties
def quadratic_roots (r s : ℚ) : Prop :=
  (r + s = 5 / 3) ∧ (r * s = -8 / 3)

theorem quadratic_expression (r s : ℚ) (h : quadratic_roots r s) :
  (9 * r^2 - 9 * s^2) / (r - s) = 15 :=
by
  sorry

end quadratic_expression_l122_122905


namespace cost_per_rug_proof_l122_122666

noncomputable def cost_per_rug (price_sold : ℝ) (number_rugs : ℕ) (profit : ℝ) : ℝ :=
  let total_revenue := number_rugs * price_sold
  let total_cost := total_revenue - profit
  total_cost / number_rugs

theorem cost_per_rug_proof : cost_per_rug 60 20 400 = 40 :=
by
  -- Lean will need the proof steps here, which are skipped
  -- The solution steps illustrate how Lean would derive this in a proof
  sorry

end cost_per_rug_proof_l122_122666


namespace original_circle_area_l122_122625

theorem original_circle_area (A : ℝ) (h1 : ∃ sector_area : ℝ, sector_area = 5) (h2 : A / 64 = 5) : A = 320 := 
by sorry

end original_circle_area_l122_122625


namespace triangle_inequality_l122_122076

variable (R r e f : ℝ)

theorem triangle_inequality (h1 : ∃ (A B C : ℝ × ℝ), true)
                            (h2 : true) :
  R^2 - e^2 ≥ 4 * (r^2 - f^2) :=
by sorry

end triangle_inequality_l122_122076


namespace am_gm_inequality_l122_122785

theorem am_gm_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  (a + b) * (b + c) * (c + a) ≥ 8 * a * b * c :=
by 
  sorry

end am_gm_inequality_l122_122785


namespace triangle_CD_length_l122_122165

noncomputable def triangle_AB_values : ℝ := 4024
noncomputable def triangle_AC_values : ℝ := 4024
noncomputable def triangle_BC_values : ℝ := 2012
noncomputable def CD_value : ℝ := 504.5

theorem triangle_CD_length 
  (AB AC : ℝ)
  (BC : ℝ)
  (CD : ℝ)
  (h1 : AB = triangle_AB_values)
  (h2 : AC = triangle_AC_values)
  (h3 : BC = triangle_BC_values) :
  CD = CD_value := by
  sorry

end triangle_CD_length_l122_122165


namespace find_y_of_x_pow_l122_122893

theorem find_y_of_x_pow (x y : ℝ) (h1 : x = 2) (h2 : x^(3*y - 1) = 8) : y = 4 / 3 :=
by
  -- skipping proof
  sorry

end find_y_of_x_pow_l122_122893


namespace find_a_l122_122352

variable {x : ℝ} {a b : ℝ}

def setA : Set ℝ := {x | Real.log x / Real.log 2 > 1}
def setB (a : ℝ) : Set ℝ := {x | x < a}
def setIntersection (b : ℝ) : Set ℝ := {x | b < x ∧ x < 2 * b + 3}

theorem find_a (h : setA ∩ setB a = setIntersection b) : a = 7 := 
by
  sorry

end find_a_l122_122352


namespace expression_evaluation_l122_122693

theorem expression_evaluation :
  2 - 3 * (-4) + 5 - (-6) * 7 = 61 :=
sorry

end expression_evaluation_l122_122693


namespace digit_a_solution_l122_122084

theorem digit_a_solution :
  ∃ a : ℕ, a000 + a998 + a999 = 22997 → a = 7 :=
sorry

end digit_a_solution_l122_122084


namespace divide_0_24_by_0_004_l122_122166

theorem divide_0_24_by_0_004 : 0.24 / 0.004 = 60 := by
  sorry

end divide_0_24_by_0_004_l122_122166


namespace part1_l122_122089

theorem part1 (a b c : ℤ) (h : a + b + c = 0) : a^3 + a^2 * c - a * b * c + b^2 * c + b^3 = 0 := 
sorry

end part1_l122_122089


namespace petya_vasya_meet_at_lantern_64_l122_122780

-- Define the total number of lanterns and intervals
def total_lanterns : ℕ := 100
def total_intervals : ℕ := total_lanterns - 1

-- Define the positions of Petya and Vasya at a given time
def petya_initial : ℕ := 1
def vasya_initial : ℕ := 100
def petya_position : ℕ := 22
def vasya_position : ℕ := 88

-- Define the number of intervals covered by Petya and Vasya
def petya_intervals_covered : ℕ := petya_position - petya_initial
def vasya_intervals_covered : ℕ := vasya_initial - vasya_position

-- Define the combined intervals covered
def combined_intervals_covered : ℕ := petya_intervals_covered + vasya_intervals_covered

-- Define the interval after which Petya and Vasya will meet
def meeting_intervals : ℕ := total_intervals - combined_intervals_covered

-- Define the final meeting point according to Petya's travel
def meeting_lantern : ℕ := petya_initial + (meeting_intervals / 2)

theorem petya_vasya_meet_at_lantern_64 : meeting_lantern = 64 := by {
  -- Proof goes here
  sorry
}

end petya_vasya_meet_at_lantern_64_l122_122780


namespace rounding_effect_l122_122400

/-- Given positive integers x, y, and z, and rounding scenarios, the
  approximation of x/y - z is necessarily less than its exact value
  when z is rounded up and x and y are rounded down. -/
theorem rounding_effect (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) 
(RoundXDown RoundYDown RoundZUp : ℕ → ℕ) 
(HRoundXDown : ∀ a, RoundXDown a ≤ a)
(HRoundYDown : ∀ a, RoundYDown a ≤ a)
(HRoundZUp : ∀ a, a ≤ RoundZUp a) :
  (RoundXDown x) / (RoundYDown y) - (RoundZUp z) < x / y - z :=
sorry

end rounding_effect_l122_122400


namespace travel_time_between_resorts_l122_122026

theorem travel_time_between_resorts
  (num_cars : ℕ)
  (car_interval : ℕ)
  (opposing_encounter_time : ℕ)
  (travel_time : ℕ) :
  num_cars = 80 →
  car_interval = 15 →
  (opposing_encounter_time * 2 * car_interval / travel_time) = num_cars →
  travel_time = 20 :=
by
  sorry

end travel_time_between_resorts_l122_122026


namespace maximal_cards_taken_l122_122656

theorem maximal_cards_taken (cards : Finset ℕ) (h_cards : ∀ n, n ∈ cards ↔ 1 ≤ n ∧ n ≤ 100)
                            (andriy_cards nick_cards : Finset ℕ)
                            (h_card_count : andriy_cards.card = nick_cards.card)
                            (h_card_relation : ∀ n, n ∈ andriy_cards → (2 * n + 2) ∈ nick_cards) :
                            andriy_cards.card + nick_cards.card ≤ 50 := 
sorry

end maximal_cards_taken_l122_122656


namespace trader_gain_l122_122056

-- Conditions
def cost_price (pen : Type) : ℕ → ℝ := sorry -- Type to represent the cost price of a pen
def selling_price (pen : Type) : ℕ → ℝ := sorry -- Type to represent the selling price of a pen
def gain_percentage : ℝ := 0.40 -- 40% gain

-- Statement of the problem to prove
theorem trader_gain (C : ℝ) (N : ℕ) : 
  (100 : ℕ) * C * gain_percentage = N * C → 
  N = 40 :=
by
  sorry

end trader_gain_l122_122056


namespace find_a_and_solve_inequality_l122_122920

theorem find_a_and_solve_inequality :
  (∀ x : ℝ, |x^2 - 4 * x + a| + |x - 3| ≤ 5 → x ≤ 3) →
  a = 8 :=
by
  sorry

end find_a_and_solve_inequality_l122_122920


namespace union_of_intervals_l122_122700

theorem union_of_intervals :
  let M := {x : ℝ | x^2 - 3 * x - 4 ≤ 0}
  let N := {x : ℝ | x^2 - 16 ≤ 0}
  M ∪ N = {x : ℝ | -4 ≤ x ∧ x ≤ 4} :=
by
  sorry

end union_of_intervals_l122_122700


namespace sum_squares_divisible_by_4_iff_even_l122_122361

theorem sum_squares_divisible_by_4_iff_even (a b c : ℕ) (ha : a % 2 = 0) (hb : b % 2 = 0) (hc : c % 2 = 0) : 
(a^2 + b^2 + c^2) % 4 = 0 ↔ 
  (a % 2 = 0 ∧ b % 2 = 0 ∧ c % 2 = 0) :=
sorry

end sum_squares_divisible_by_4_iff_even_l122_122361


namespace find_function_l122_122301

theorem find_function (f : ℕ → ℕ) (k : ℕ) :
  (∀ n : ℕ, f n < f (n + 1)) →
  (∀ n : ℕ, f (f n) = n + 2 * k) →
  ∀ n : ℕ, f n = n + k := 
by
  intro h1 h2
  sorry

end find_function_l122_122301


namespace rational_root_neg_one_third_l122_122821

def P (x : ℚ) : ℚ := 3 * x^5 - 4 * x^3 - 7 * x^2 + 2 * x + 1

theorem rational_root_neg_one_third : P (-1/3) = 0 :=
by
  have : (-1/3 : ℚ) ≠ 0 := by norm_num
  sorry

end rational_root_neg_one_third_l122_122821


namespace original_price_l122_122669

variables (p q d : ℝ)


theorem original_price (x : ℝ) (h : x * (1 + p / 100) * (1 - q / 100) = d) :
  x = 100 * d / (100 + p - q - p * q / 100) := 
sorry

end original_price_l122_122669


namespace dog_adult_weight_l122_122711

theorem dog_adult_weight 
  (w7 : ℕ) (w7_eq : w7 = 6)
  (w9 : ℕ) (w9_eq : w9 = 2 * w7)
  (w3m : ℕ) (w3m_eq : w3m = 2 * w9)
  (w5m : ℕ) (w5m_eq : w5m = 2 * w3m)
  (w1y : ℕ) (w1y_eq : w1y = w5m + 30) :
  w1y = 78 := by
  -- Proof is not required, so we leave it with sorry.
  sorry

end dog_adult_weight_l122_122711


namespace min_rows_for_students_l122_122791

def min_rows (total_students seats_per_row max_students_per_school : ℕ) : ℕ :=
  total_students / seats_per_row + if total_students % seats_per_row == 0 then 0 else 1

theorem min_rows_for_students :
  ∀ (total_students seats_per_row max_students_per_school : ℕ),
  (total_students = 2016) →
  (seats_per_row = 168) →
  (max_students_per_school = 40) →
  min_rows total_students seats_per_row max_students_per_school = 15 :=
by
  intros total_students seats_per_row max_students_per_school h1 h2 h3
  -- We write down the proof outline to show that 15 is the required minimum
  sorry

end min_rows_for_students_l122_122791


namespace sin_cos_pow_eq_l122_122326

theorem sin_cos_pow_eq (sin cos : ℝ → ℝ) (x : ℝ) (h₀ : sin x + cos x = -1) (n : ℕ) : 
  sin x ^ n + cos x ^ n = (-1) ^ n :=
by
  sorry

end sin_cos_pow_eq_l122_122326


namespace sales_on_second_street_l122_122817

noncomputable def commission_per_system : ℕ := 25
noncomputable def total_commission : ℕ := 175
noncomputable def total_systems_sold : ℕ := total_commission / commission_per_system

def first_street_sales (S : ℕ) : ℕ := S
def second_street_sales (S : ℕ) : ℕ := 2 * S
def third_street_sales : ℕ := 0
def fourth_street_sales : ℕ := 1

def total_sales (S : ℕ) : ℕ := first_street_sales S + second_street_sales S + third_street_sales + fourth_street_sales

theorem sales_on_second_street (S : ℕ) : total_sales S = total_systems_sold → second_street_sales S = 4 := by
  sorry

end sales_on_second_street_l122_122817


namespace problem_statement_l122_122985

noncomputable def log_three_four : ℝ := Real.log 4 / Real.log 3
noncomputable def a : ℝ := Real.log (log_three_four) / Real.log (3/4)
noncomputable def b : ℝ := Real.rpow (3/4 : ℝ) 0.5
noncomputable def c : ℝ := Real.rpow (4/3 : ℝ) 0.5

theorem problem_statement : a < b ∧ b < c :=
by
  sorry

end problem_statement_l122_122985


namespace exponent_of_two_gives_n_l122_122734

theorem exponent_of_two_gives_n (x: ℝ) (n: ℝ) (b: ℝ)
  (h1: n = 2 ^ x)
  (h2: n ^ b = 8)
  (h3: b = 12) : x = 3 / 12 :=
by
  sorry

end exponent_of_two_gives_n_l122_122734


namespace problem1_problem2_l122_122243

-- Define the variables for the two problems
variables (a b x : ℝ)

-- The first problem
theorem problem1 :
  (a + 2 * b) ^ 2 - 4 * b * (a + b) = a ^ 2 :=
by 
  -- Proof goes here
  sorry

-- The second problem
theorem problem2 :
  ((x ^ 2 - 2 * x) / (x ^ 2 - 4 * x + 4) + 1 / (2 - x)) / ((x - 1) / (x ^ 2 - 4)) = x + 2 :=
by 
  -- Proof goes here
  sorry

end problem1_problem2_l122_122243


namespace range_of_a_l122_122155

theorem range_of_a (a : ℝ) : (∀ x : ℝ, |x + 2| + |x - 1| ≥ a) ↔ a ≤ 3 := by
  sorry

end range_of_a_l122_122155


namespace PatriciaHighlightFilmTheorem_l122_122241

def PatriciaHighlightFilmProblem : Prop :=
  let point_guard_seconds := 130
  let shooting_guard_seconds := 145
  let small_forward_seconds := 85
  let power_forward_seconds := 60
  let center_seconds := 180
  let total_seconds := point_guard_seconds + shooting_guard_seconds + small_forward_seconds + power_forward_seconds + center_seconds
  let num_players := 5
  let average_seconds := total_seconds / num_players
  let average_minutes := average_seconds / 60
  average_minutes = 2

theorem PatriciaHighlightFilmTheorem : PatriciaHighlightFilmProblem :=
  by
    -- Proof goes here
    sorry

end PatriciaHighlightFilmTheorem_l122_122241


namespace sufficient_condition_l122_122605

theorem sufficient_condition (a b c : ℤ) : (a = c + 1) → (b = a - 1) → a * (a - b) + b * (b - c) + c * (c - a) = 2 :=
by
  intros h1 h2
  sorry

end sufficient_condition_l122_122605


namespace calculate_expression_l122_122086

theorem calculate_expression : (1000^2) / (252^2 - 248^2) = 500 := sorry

end calculate_expression_l122_122086


namespace inequality_proof_l122_122379

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a * b * c = 1) :
  1 / (1 + a + b) + 1 / (1 + b + c) + 1 / (1 + c + a) ≤ 1 := 
by 
  sorry

end inequality_proof_l122_122379


namespace original_pencils_l122_122145

-- Define the conditions
def pencils_added : ℕ := 30
def total_pencils_now : ℕ := 71

-- Define the theorem to prove the original number of pencils
theorem original_pencils (original_pencils : ℕ) :
  total_pencils_now = original_pencils + pencils_added → original_pencils = 41 :=
by
  intros h
  sorry

end original_pencils_l122_122145


namespace tan_of_x_is_3_l122_122850

theorem tan_of_x_is_3 (x : ℝ) (h : Real.tan x = 3) (hx : Real.cos x ≠ 0) : 
  (Real.sin x + 3 * Real.cos x) / (2 * Real.sin x - 3 * Real.cos x) = 2 :=
by
  sorry

end tan_of_x_is_3_l122_122850


namespace average_food_per_week_l122_122343

-- Definitions based on conditions
def food_first_dog := 13
def food_second_dog := 2 * food_first_dog
def food_third_dog := 6
def number_of_dogs := 3

-- Statement of the proof problem
theorem average_food_per_week : 
  (food_first_dog + food_second_dog + food_third_dog) / number_of_dogs = 15 := 
by sorry

end average_food_per_week_l122_122343


namespace solve_for_x_l122_122363

def delta (x : ℝ) : ℝ := 4 * x + 5
def phi (x : ℝ) : ℝ := 6 * x + 3

theorem solve_for_x (x : ℝ) (h : delta (phi x) = -1) : x = - 3 / 4 :=
by
  sorry

end solve_for_x_l122_122363


namespace monomial_sum_l122_122518

theorem monomial_sum (m n : ℤ) (h1 : n = 2) (h2 : m + 2 = 1) : m + n = 1 := by
  sorry

end monomial_sum_l122_122518


namespace positive_integer_expression_l122_122441

theorem positive_integer_expression (q : ℕ) (h : q > 0) : 
  ((∃ k : ℕ, k > 0 ∧ (5 * q + 18) = k * (3 * q - 8)) ↔ q = 3 ∨ q = 4 ∨ q = 5 ∨ q = 12) := 
sorry

end positive_integer_expression_l122_122441


namespace joe_first_lift_is_400_mike_first_lift_is_450_lisa_second_lift_is_250_l122_122355

-- Defining the weights of Joe's lifts
variable (J1 J2 : ℕ)

-- Conditions for Joe
def joe_conditions : Prop :=
  (J1 + J2 = 900) ∧ (2 * J1 = J2 + 300)

-- Defining the weights of Mike's lifts
variable (M1 M2 : ℕ)

-- Conditions for Mike  
def mike_conditions : Prop :=
  (M1 + M2 = 1100) ∧ (M2 = M1 + 200)

-- Defining the weights of Lisa's lifts
variable (L1 L2 : ℕ)

-- Conditions for Lisa  
def lisa_conditions : Prop :=
  (L1 + L2 = 1000) ∧ (L1 = 3 * L2)

-- Proof statements
theorem joe_first_lift_is_400 (h : joe_conditions J1 J2) : J1 = 400 :=
by
  sorry

theorem mike_first_lift_is_450 (h : mike_conditions M1 M2) : M1 = 450 :=
by
  sorry

theorem lisa_second_lift_is_250 (h : lisa_conditions L1 L2) : L2 = 250 :=
by
  sorry

end joe_first_lift_is_400_mike_first_lift_is_450_lisa_second_lift_is_250_l122_122355


namespace find_ratio_of_constants_l122_122126

theorem find_ratio_of_constants (x y c d : ℚ) (hx : x ≠ 0) (hy : y ≠ 0) (hd : d ≠ 0) 
  (h₁ : 8 * x - 6 * y = c) (h₂ : 12 * y - 18 * x = d) : c / d = -4 / 9 := 
sorry

end find_ratio_of_constants_l122_122126


namespace rectangle_area_increase_l122_122990

theorem rectangle_area_increase
  (l w : ℝ)
  (h₀ : l > 0) -- original length is positive
  (h₁ : w > 0) -- original width is positive
  (length_increase : l' = 1.3 * l) -- new length after increase
  (width_increase : w' = 1.15 * w) -- new width after increase
  (new_area : A' = l' * w') -- new area after increase
  (original_area : A = l * w) -- original area
  :
  ((A' / A) * 100 - 100) = 49.5 := by
  sorry

end rectangle_area_increase_l122_122990


namespace doughnuts_per_box_l122_122618

theorem doughnuts_per_box (total_doughnuts : ℕ) (boxes : ℕ) (h_doughnuts : total_doughnuts = 48) (h_boxes : boxes = 4) : 
  total_doughnuts / boxes = 12 :=
by
  -- This is a placeholder for the proof
  sorry

end doughnuts_per_box_l122_122618


namespace apple_crisps_calculation_l122_122722

theorem apple_crisps_calculation (apples crisps : ℕ) (h : crisps = 3 ∧ apples = 12) : 
  (36 / apples) * crisps = 9 := by
  sorry

end apple_crisps_calculation_l122_122722


namespace common_difference_arithmetic_sequence_l122_122755

theorem common_difference_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ)
  (h1 : a 5 = 10) (h2 : a 12 = 31) : d = 3 :=
by
  sorry

end common_difference_arithmetic_sequence_l122_122755


namespace trigonometric_identity_l122_122954

noncomputable def special_operation (a b : ℝ) : ℝ := a^2 - a * b - b^2

theorem trigonometric_identity :
  special_operation (Real.sin (Real.pi / 12)) (Real.cos (Real.pi / 12))
  = - (1 + 2 * Real.sqrt 3) / 4 :=
by
  sorry

end trigonometric_identity_l122_122954


namespace max_value_k_l122_122683

theorem max_value_k (x y : ℝ) (k : ℝ) (h₁ : x^2 + y^2 = 1) (h₂ : ∀ x y, x^2 + y^2 = 1 → x + y - k ≥ 0) : 
  k ≤ -Real.sqrt 2 :=
sorry

end max_value_k_l122_122683


namespace count_noncongruent_triangles_l122_122806

theorem count_noncongruent_triangles :
  ∃ (n : ℕ), n = 13 ∧
  ∀ (a b c : ℕ), a < b ∧ b < c ∧ a + b > c ∧ a + b + c < 20 ∧ ¬(a * a + b * b = c * c)
  → n = 13 := by {
  sorry
}

end count_noncongruent_triangles_l122_122806


namespace triangle_inequality_lt_l122_122679

theorem triangle_inequality_lt {a b c : ℝ} (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h1 : a < b + c) (h2 : b < a + c) (h3 : c < a + b) : a^2 + b^2 + c^2 < 2 * (a*b + b*c + c*a) := 
sorry

end triangle_inequality_lt_l122_122679


namespace solution_set_of_inequality_l122_122911

theorem solution_set_of_inequality :
  { x : ℝ | ∃ (h : x ≠ 1), 1 / (x - 1) ≥ -1 } = { x : ℝ | x ≤ 0 ∨ 1 < x } :=
by sorry

end solution_set_of_inequality_l122_122911


namespace directrix_of_parabola_l122_122228

theorem directrix_of_parabola (p m : ℝ) (hp : p > 0)
  (hM_on_parabola : (4, m).fst ^ 2 = 2 * p * (4, m).snd)
  (hM_to_focus : dist (4, m) (p / 2, 0) = 6) :
  -p/2 = -2 :=
sorry

end directrix_of_parabola_l122_122228


namespace ratio_of_speeds_l122_122859

theorem ratio_of_speeds
  (speed_of_tractor : ℝ)
  (speed_of_bike : ℝ)
  (speed_of_car : ℝ)
  (h1 : speed_of_tractor = 575 / 25)
  (h2 : speed_of_car = 331.2 / 4)
  (h3 : speed_of_bike = 2 * speed_of_tractor) :
  speed_of_car / speed_of_bike = 1.8 :=
by
  sorry

end ratio_of_speeds_l122_122859


namespace sum_of_ages_l122_122809

variables (P M Mo : ℕ)

theorem sum_of_ages (h1 : 5 * P = 3 * M)
                    (h2 : 5 * M = 3 * Mo)
                    (h3 : Mo - P = 32) :
  P + M + Mo = 98 :=
by
  sorry

end sum_of_ages_l122_122809


namespace tank_filled_fraction_l122_122856

noncomputable def initial_quantity (total_capacity : ℕ) := (3 / 4 : ℚ) * total_capacity

noncomputable def final_quantity (initial : ℚ) (additional : ℚ) := initial + additional

noncomputable def fraction_of_capacity (quantity : ℚ) (total_capacity : ℕ) := quantity / total_capacity

theorem tank_filled_fraction (total_capacity : ℕ) (additional_gas : ℚ)
  (initial_fraction : ℚ) (final_fraction : ℚ) :
  initial_fraction = initial_quantity total_capacity →
  final_fraction = fraction_of_capacity (final_quantity initial_fraction additional_gas) total_capacity →
  total_capacity = 42 →
  additional_gas = 7 →
  initial_fraction = 31.5 →
  final_fraction = (833 / 909 : ℚ) :=
by
  sorry

end tank_filled_fraction_l122_122856


namespace latus_rectum_of_parabola_l122_122350

theorem latus_rectum_of_parabola (x : ℝ) :
  (∀ x, y = (-1 / 4 : ℝ) * x^2) → y = (-1 / 2 : ℝ) :=
sorry

end latus_rectum_of_parabola_l122_122350


namespace distinguishes_conditional_from_sequential_l122_122237

variable (C P S I D : Prop)

-- Conditions
def conditional_structure_includes_processing_box  : Prop := C = P
def conditional_structure_includes_start_end_box   : Prop := C = S
def conditional_structure_includes_io_box          : Prop := C = I
def conditional_structure_includes_decision_box    : Prop := C = D
def sequential_structure_excludes_decision_box     : Prop := ¬S = D

-- Proof problem statement
theorem distinguishes_conditional_from_sequential : C → S → I → D → P → 
    (conditional_structure_includes_processing_box C P) ∧ 
    (conditional_structure_includes_start_end_box C S) ∧ 
    (conditional_structure_includes_io_box C I) ∧ 
    (conditional_structure_includes_decision_box C D) ∧ 
    sequential_structure_excludes_decision_box S D → 
    (D = true) :=
by sorry

end distinguishes_conditional_from_sequential_l122_122237


namespace gcd_547_323_l122_122774

theorem gcd_547_323 : Nat.gcd 547 323 = 1 := 
by
  sorry

end gcd_547_323_l122_122774


namespace totalCups_l122_122813

-- Let's state our definitions based on the conditions:
def servingsPerBox : ℕ := 9
def cupsPerServing : ℕ := 2

-- Our goal is to prove the following statement.
theorem totalCups (hServings: servingsPerBox = 9) (hCups: cupsPerServing = 2) : servingsPerBox * cupsPerServing = 18 := by
  -- The detailed proof will go here.
  sorry

end totalCups_l122_122813


namespace average_spring_headcount_average_fall_headcount_l122_122761

namespace AverageHeadcount

def springHeadcounts := [10900, 10500, 10700, 11300]
def fallHeadcounts := [11700, 11500, 11600, 11300]

def averageHeadcount (counts : List ℕ) : ℕ :=
  counts.sum / counts.length

theorem average_spring_headcount :
  averageHeadcount springHeadcounts = 10850 := by
  sorry

theorem average_fall_headcount :
  averageHeadcount fallHeadcounts = 11525 := by
  sorry

end AverageHeadcount

end average_spring_headcount_average_fall_headcount_l122_122761


namespace prob1_part1_prob1_part2_l122_122544

noncomputable def U : Set ℝ := Set.univ
noncomputable def A : Set ℝ := {x | -2 < x ∧ x < 5}
noncomputable def B (a : ℝ) : Set ℝ := {x | 2 - a < x ∧ x < 1 + 2 * a}

theorem prob1_part1 (a : ℝ) (ha : a = 3) :
  A ∪ B a = {x | -2 < x ∧ x < 7} ∧ A ∩ B a = {x | -1 < x ∧ x < 5} :=
by {
  sorry
}

theorem prob1_part2 (h : ∀ x, x ∈ A → x ∈ B a) :
  ∀ a : ℝ, a ≤ 2 :=
by {
  sorry
}

end prob1_part1_prob1_part2_l122_122544


namespace camilla_jellybeans_l122_122820

theorem camilla_jellybeans (b c : ℕ) (h1 : b = 3 * c) (h2 : b - 20 = 4 * (c - 20)) :
  b = 180 :=
by
  -- Proof steps would go here
  sorry

end camilla_jellybeans_l122_122820


namespace base_b_square_l122_122210

theorem base_b_square (b : ℕ) (h : b > 2) : ∃ k : ℕ, 121 = k ^ 2 :=
by
  sorry

end base_b_square_l122_122210


namespace cost_ratio_l122_122649

theorem cost_ratio (S J M : ℝ) (h1 : S = 4) (h2 : M = 0.75 * (S + J)) (h3 : S + J + M = 21) : J / S = 2 :=
by
  sorry

end cost_ratio_l122_122649


namespace root_equation_l122_122447

variables (m : ℝ)

theorem root_equation {m : ℝ} (h : m^2 - 2 * m - 3 = 0) : m^2 - 2 * m + 2023 = 2026 :=
by {
  sorry 
}

end root_equation_l122_122447


namespace chloe_fifth_test_score_l122_122627

theorem chloe_fifth_test_score (a1 a2 a3 a4 a5 : ℕ)
  (h1 : a1 = 84) (h2 : a2 = 87) (h3 : a3 = 78) (h4 : a4 = 90)
  (h_avg : (a1 + a2 + a3 + a4 + a5) / 5 ≥ 85) : 
  a5 ≥ 86 :=
by
  sorry

end chloe_fifth_test_score_l122_122627


namespace PQ_value_l122_122786

theorem PQ_value (DE DF EF : ℕ) (CF : ℝ) (P Q : ℝ) 
  (h1 : DE = 996)
  (h2 : DF = 995)
  (h3 : EF = 994)
  (hCF :  CF = (995^2 - 4) / 1990)
  (hP : P = (1492.5 - EF))
  (hQ : Q = (s - DF)) :
  PQ = 1 ∧ m + n = 2 :=
by
  sorry

end PQ_value_l122_122786


namespace sum_of_intersections_l122_122602

theorem sum_of_intersections :
  (∃ x1 y1 x2 y2 x3 y3 x4 y4, 
    y1 = (x1 - 1)^2 ∧ y2 = (x2 - 1)^2 ∧ y3 = (x3 - 1)^2 ∧ y4 = (x4 - 1)^2 ∧
    x1 - 2 = (y1 + 1)^2 ∧ x2 - 2 = (y2 + 1)^2 ∧ x3 - 2 = (y3 + 1)^2 ∧ x4 - 2 = (y4 + 1)^2 ∧
    (x1 + x2 + x3 + x4 + y1 + y2 + y3 + y4) = 2) :=
sorry

end sum_of_intersections_l122_122602


namespace Katrina_sold_in_morning_l122_122367

theorem Katrina_sold_in_morning :
  ∃ M : ℕ, (120 - 57 - 16 - 11) = M := sorry

end Katrina_sold_in_morning_l122_122367


namespace total_seats_l122_122557

theorem total_seats (KA_pos : ℕ) (SL_pos : ℕ) (h1 : KA_pos = 10) (h2 : SL_pos = 29) (h3 : SL_pos = KA_pos + (KA_pos * 2 - 1) / 2):
  let total_positions := 2 * (SL_pos - KA_pos - 1) + 2
  total_positions = 38 :=
by
  sorry

end total_seats_l122_122557


namespace find_random_discount_l122_122224

theorem find_random_discount
  (initial_price : ℝ) (final_price : ℝ) (autumn_discount : ℝ) (loyalty_discount : ℝ) (random_discount : ℝ) :
  initial_price = 230 ∧ final_price = 69 ∧ autumn_discount = 0.25 ∧ loyalty_discount = 0.20 ∧ 
  final_price = initial_price * (1 - autumn_discount) * (1 - loyalty_discount) * (1 - random_discount / 100) →
  random_discount = 50 :=
by
  intros h
  sorry

end find_random_discount_l122_122224


namespace hannahs_grapes_per_day_l122_122798

-- Definitions based on conditions
def oranges_per_day : ℕ := 20
def days : ℕ := 30
def total_fruits : ℕ := 1800
def total_oranges : ℕ := oranges_per_day * days

-- The math proof problem to be targeted
theorem hannahs_grapes_per_day : 
  (total_fruits - total_oranges) / days = 40 := 
by
  -- Proof to be filled in here
  sorry

end hannahs_grapes_per_day_l122_122798


namespace prob_white_given_popped_l122_122044

-- Definitions for given conditions:
def P_white : ℚ := 1 / 2
def P_yellow : ℚ := 1 / 4
def P_blue : ℚ := 1 / 4

def P_popped_given_white : ℚ := 1 / 3
def P_popped_given_yellow : ℚ := 3 / 4
def P_popped_given_blue : ℚ := 2 / 3

-- Calculations derived from conditions:
def P_white_popped : ℚ := P_white * P_popped_given_white
def P_yellow_popped : ℚ := P_yellow * P_popped_given_yellow
def P_blue_popped : ℚ := P_blue * P_popped_given_blue

def P_popped : ℚ := P_white_popped + P_yellow_popped + P_blue_popped

-- Main theorem to be proved:
theorem prob_white_given_popped : (P_white_popped / P_popped) = 2 / 11 :=
by sorry

end prob_white_given_popped_l122_122044


namespace percentage_of_total_is_sixty_l122_122429

def num_boys := 600
def diff_boys_girls := 400
def num_girls := num_boys + diff_boys_girls
def total_people := num_boys + num_girls
def target_number := 960
def target_percentage := (target_number / total_people) * 100

theorem percentage_of_total_is_sixty :
  target_percentage = 60 := by
  sorry

end percentage_of_total_is_sixty_l122_122429


namespace sequence_fifth_term_l122_122300

theorem sequence_fifth_term (a b c : ℕ) :
  (a = (2 + b) / 3) →
  (b = (a + 34) / 3) →
  (34 = (b + c) / 3) →
  c = 89 :=
by
  intros ha hb hc
  sorry

end sequence_fifth_term_l122_122300


namespace cars_produced_in_europe_l122_122007

theorem cars_produced_in_europe (total_cars : ℕ) (cars_in_north_america : ℕ) (cars_in_europe : ℕ) :
  total_cars = 6755 → cars_in_north_america = 3884 → cars_in_europe = total_cars - cars_in_north_america → cars_in_europe = 2871 :=
by
  -- necessary calculations and logical steps
  sorry

end cars_produced_in_europe_l122_122007


namespace number_of_ways_to_form_team_l122_122751

-- Defining the conditions
def total_employees : ℕ := 15
def num_men : ℕ := 10
def num_women : ℕ := 5
def team_size : ℕ := 6
def men_in_team : ℕ := 4
def women_in_team : ℕ := 2

-- Using binomial coefficient to represent combinations
noncomputable def choose (n k : ℕ) : ℕ := Nat.choose n k

-- The theorem to be proved
theorem number_of_ways_to_form_team :
  (choose num_men men_in_team) * (choose num_women women_in_team) = 
  choose 10 4 * choose 5 2 :=
by
  sorry

end number_of_ways_to_form_team_l122_122751


namespace permutation_sum_inequality_l122_122974

noncomputable def permutations (n : ℕ) : List (List ℚ) :=
  List.permutations ((List.range (n+1)).map (fun i => if i = 0 then (1 : ℚ) else (1 : ℚ) / i))

theorem permutation_sum_inequality (n : ℕ) (a b : Fin n → ℚ)
  (ha : ∃ p : List ℚ, p ∈ permutations n ∧ ∀ i, a i = p.get? i) 
  (hb : ∃ q : List ℚ, q ∈ permutations n ∧ ∀ i, b i = q.get? i)
  (h_sum : ∀ i j : Fin n, i ≤ j → a i + b i ≥ a j + b j) 
  (m : Fin n) :
  a m + b m ≤ 4 / (m + 1) :=
sorry

end permutation_sum_inequality_l122_122974


namespace part_I_part_II_l122_122281

noncomputable
def x₀ : ℝ := 2

noncomputable
def f (x m : ℝ) : ℝ := |x - m| + |x + 1/m| - x₀

theorem part_I (x : ℝ) : |x + 3| - 2 * x - 1 < 0 ↔ x > 2 :=
by sorry

theorem part_II (m : ℝ) (h : m > 0) :
  (∃ x : ℝ, f x m = 0) → m = 1 :=
by sorry

end part_I_part_II_l122_122281


namespace remainder_x_101_div_x2_plus1_x_plus1_l122_122943

theorem remainder_x_101_div_x2_plus1_x_plus1 : 
  (x^101) % ((x^2 + 1) * (x + 1)) = x :=
by
  sorry

end remainder_x_101_div_x2_plus1_x_plus1_l122_122943


namespace servings_per_day_l122_122035

-- Conditions
def week_servings := 21
def days_per_week := 7

-- Question and Answer
theorem servings_per_day : week_servings / days_per_week = 3 := 
by
  sorry

end servings_per_day_l122_122035


namespace rational_numbers_cubic_sum_l122_122159

theorem rational_numbers_cubic_sum
  (a b c : ℚ)
  (h1 : a - b + c = 3)
  (h2 : a^2 + b^2 + c^2 = 3) :
  a^3 + b^3 + c^3 = 1 :=
by
  sorry

end rational_numbers_cubic_sum_l122_122159


namespace area_percent_of_smaller_rectangle_l122_122775

-- Definitions of the main geometric elements and assumptions
def larger_rectangle (w h : ℝ) : Prop := (w > 0) ∧ (h > 0)
def radius_of_circle (w h r : ℝ) : Prop := r = Real.sqrt (w^2 + h^2)
def inscribed_smaller_rectangle (w h x y : ℝ) : Prop := 
  (0 < x) ∧ (x < 1) ∧ (0 < y) ∧ (y < 1) ∧
  ((h + 2 * y * h)^2 + (x * w)^2 = w^2 + h^2)

-- Prove the area percentage relationship
theorem area_percent_of_smaller_rectangle 
  (w h x y : ℝ) 
  (hw : w > 0) (hh : h > 0)
  (hcirc : radius_of_circle w h (Real.sqrt (w^2 + h^2)))
  (hsmall_rect : inscribed_smaller_rectangle w h x y) :
  (4 * x * y) / (4.0 * 1.0) * 100 = 8.33 := sorry

end area_percent_of_smaller_rectangle_l122_122775


namespace number_of_red_yarns_l122_122104

-- Definitions
def scarves_per_yarn : Nat := 3
def blue_yarns : Nat := 6
def yellow_yarns : Nat := 4
def total_scarves : Nat := 36

-- Theorem
theorem number_of_red_yarns (R : Nat) (H1 : scarves_per_yarn * blue_yarns + scarves_per_yarn * yellow_yarns + scarves_per_yarn * R = total_scarves) :
  R = 2 :=
by
  sorry

end number_of_red_yarns_l122_122104


namespace Jenny_recycling_l122_122884

theorem Jenny_recycling:
  let bottle_weight := 6
  let can_weight := 2
  let glass_jar_weight := 8
  let max_weight := 100
  let num_cans := 20
  let bottle_value := 10
  let can_value := 3
  let glass_jar_value := 12
  let total_money := (num_cans * can_value) + (7 * glass_jar_value) + (0 * bottle_value)
  total_money = 144 ∧ num_cans = 20 ∧ glass_jars = 7 ∧ bottles = 0 := by sorry

end Jenny_recycling_l122_122884


namespace intersection_points_count_l122_122449

theorem intersection_points_count:
  let line1 := { p : ℝ × ℝ | ∃ x y : ℝ, 4 * y - 3 * x = 2 ∧ (p.1 = x ∧ p.2 = y) }
  let line2 := { p : ℝ × ℝ | ∃ x y : ℝ, x + 3 * y = 3 ∧ (p.1 = x ∧ p.2 = y) }
  let line3 := { p : ℝ × ℝ | ∃ x y : ℝ, 6 * x - 8 * y = 6 ∧ (p.1 = x ∧ p.2 = y) }
  ∃! p1 p2 : ℝ × ℝ, p1 ∈ line1 ∧ p1 ∈ line2 ∧ p2 ∈ line2 ∧ p2 ∈ line3 :=
by
  sorry

end intersection_points_count_l122_122449


namespace solution_eq_c_l122_122665

variables (x : ℝ) (a : ℝ) 

def p := ∃ x0 : ℝ, (0 < x0) ∧ (3^x0 + x0 = 2016)
def q := ∃ a : ℝ, (0 < a) ∧ (∀ x : ℝ, (|x| - a * x) = (|(x)| - a * (-x)))

theorem solution_eq_c : p ∧ ¬q :=
by {
  sorry -- proof placeholder
}

end solution_eq_c_l122_122665


namespace problem_statement_l122_122142

theorem problem_statement (a b : ℕ → ℕ) (h1 : a 1 = 1) (h2 : a 2 = 2) 
  (h3 : ∀ n : ℕ, a (n + 2) = a n)
  (h_b : ∀ n : ℕ, b (n + 1) - b n = a n)
  (h_repeat : ∀ k : ℕ, ∃ m : ℕ, (b (2 * m) / a m) = k)
  : b 1 = 2 :=
sorry

end problem_statement_l122_122142


namespace pizza_slices_l122_122574

theorem pizza_slices (total_slices pepperoni_slices mushroom_slices : ℕ) 
  (h_total : total_slices = 24)
  (h_pepperoni : pepperoni_slices = 15)
  (h_mushrooms : mushroom_slices = 16)
  (h_at_least_one : total_slices = pepperoni_slices + mushroom_slices - both_slices)
  : both_slices = 7 :=
by
  have h1 : total_slices = 24 := h_total
  have h2 : pepperoni_slices = 15 := h_pepperoni
  have h3 : mushroom_slices = 16 := h_mushrooms
  have h4 : total_slices = 24 := by sorry
  sorry

end pizza_slices_l122_122574


namespace kite_area_l122_122344

theorem kite_area (EF GH : ℝ) (FG EH : ℕ) (h1 : FG * FG + EH * EH = 25) : EF * GH = 12 :=
by
  sorry

end kite_area_l122_122344


namespace max_g_at_8_l122_122998

noncomputable def g : ℝ → ℝ :=
  sorry -- We define g here abstractly, with nonnegative coefficients

axiom g_nonneg_coeffs : ∀ x, 0 ≤ g x
axiom g_at_4 : g 4 = 16
axiom g_at_16 : g 16 = 256

theorem max_g_at_8 : g 8 ≤ 64 :=
by sorry

end max_g_at_8_l122_122998


namespace Amy_homework_time_l122_122276

def mathProblems : Nat := 18
def spellingProblems : Nat := 6
def problemsPerHour : Nat := 4
def totalProblems : Nat := mathProblems + spellingProblems
def totalHours : Nat := totalProblems / problemsPerHour

theorem Amy_homework_time :
  totalHours = 6 := by
  sorry

end Amy_homework_time_l122_122276


namespace problem_part1_29_13_problem_part2_mn_problem_part3_k_36_problem_part4_min_val_l122_122906

def is_perfect_number (n : ℕ) : Prop :=
  ∃ a b : ℤ, n = a^2 + b^2

theorem problem_part1_29_13 : is_perfect_number 29 ∧ is_perfect_number 13 := by
  sorry

theorem problem_part2_mn : 
  ∃ m n : ℤ, (∀ a : ℤ, a^2 - 4 * a + 8 = (a - m)^2 + n^2) ∧ (m * n = 4 ∨ m * n = -4) := by
  sorry

theorem problem_part3_k_36 (a b : ℤ) : 
  ∃ k : ℤ, (∀ k : ℤ, a^2 + 4*a*b + 5*b^2 - 12*b + k = (a + 2*b)^2 + (b-6)^2) ∧ k = 36 := by
  sorry

theorem problem_part4_min_val (a b : ℝ) : 
  (∀ (a b : ℝ), -a^2 + 5*a + b - 7 = 0 → ∃ a' b', (a + b = (a'-2)^2 + 3) ∧ a' + b' = 3) := by
  sorry

end problem_part1_29_13_problem_part2_mn_problem_part3_k_36_problem_part4_min_val_l122_122906


namespace maximum_n_Sn_pos_l122_122468

def arithmetic_sequence := ℕ → ℝ

noncomputable def sum_first_n_terms (a : arithmetic_sequence) (n : ℕ) : ℝ :=
  (n / 2) * (2 * a 1 + (n - 1) * (a 2 - a 1))

axiom a1_eq : ∀ (a : arithmetic_sequence), (a 1) = 2 * (a 2) + (a 4)

axiom S5_eq_5 : ∀ (a : arithmetic_sequence), sum_first_n_terms a 5 = 5

theorem maximum_n_Sn_pos : ∀ (a : arithmetic_sequence), (∃ (n : ℕ), n < 6 ∧ sum_first_n_terms a n > 0) → n = 5 :=
  sorry

end maximum_n_Sn_pos_l122_122468


namespace payment_first_trip_payment_second_trip_l122_122861

-- Define conditions and questions
variables {x y : ℝ}

-- Conditions: discounts and expenditure
def discount_1st_trip (x : ℝ) := 0.9 * x
def discount_2nd_trip (y : ℝ) := 300 * 0.9 + (y - 300) * 0.8

def combined_discount (x y : ℝ) := 300 * 0.9 + (x + y - 300) * 0.8

-- Given conditions as equations
axiom eq1 : discount_1st_trip x + discount_2nd_trip y - combined_discount x y = 19
axiom eq2 : x + y - (discount_1st_trip x + discount_2nd_trip y) = 67

-- The proof statements
theorem payment_first_trip : discount_1st_trip 190 = 171 := by sorry

theorem payment_second_trip : discount_2nd_trip 390 = 342 := by sorry

end payment_first_trip_payment_second_trip_l122_122861


namespace total_sheep_l122_122384

theorem total_sheep (n : ℕ) 
  (h1 : 3 ∣ n)
  (h2 : 5 ∣ n)
  (h3 : 6 ∣ n)
  (h4 : 8 ∣ n)
  (h5 : n * 7 / 40 = 12) : 
  n = 68 :=
by
  sorry

end total_sheep_l122_122384


namespace neg_power_identity_l122_122266

variable (m : ℝ)

theorem neg_power_identity : (-m^2)^3 = -m^6 :=
sorry

end neg_power_identity_l122_122266


namespace length_of_other_train_l122_122462

variable (L : ℝ)

theorem length_of_other_train
    (train1_length : ℝ := 260)
    (train1_speed_kmh : ℝ := 120)
    (train2_speed_kmh : ℝ := 80)
    (time_to_cross : ℝ := 9)
    (train1_speed : ℝ := train1_speed_kmh * 1000 / 3600)
    (train2_speed : ℝ := train2_speed_kmh * 1000 / 3600)
    (relative_speed : ℝ := train1_speed + train2_speed)
    (total_distance : ℝ := relative_speed * time_to_cross)
    (other_train_length : ℝ := total_distance - train1_length) :
    L = other_train_length := by
  sorry

end length_of_other_train_l122_122462


namespace arithmetic_sequence_term_l122_122992

theorem arithmetic_sequence_term :
  ∀ a : ℕ → ℕ, (a 1 = 1) → (∀ n : ℕ, a (n + 1) - a n = 2) → (a 6 = 11) :=
by
  intros a h1 hrec
  sorry

end arithmetic_sequence_term_l122_122992


namespace arrange_chairs_and_stools_l122_122153

-- Definition of the mathematical entities based on the conditions
def num_ways_to_arrange (women men : ℕ) : ℕ :=
  let total := women + men
  (total.factorial) / (women.factorial * men.factorial)

-- Prove that the arrangement yields the correct number of ways
theorem arrange_chairs_and_stools :
  num_ways_to_arrange 7 3 = 120 := by
  -- The specific definitions and steps are not to be included in the Lean statement;
  -- hence, adding a placeholder for the proof.
  sorry

end arrange_chairs_and_stools_l122_122153


namespace bryden_receives_22_50_dollars_l122_122369

-- Define the face value of a regular quarter
def face_value_regular : ℝ := 0.25

-- Define the number of regular quarters Bryden has
def num_regular_quarters : ℕ := 4

-- Define the face value of the special quarter
def face_value_special : ℝ := face_value_regular * 2

-- The collector pays 15 times the face value for regular quarters
def multiplier : ℝ := 15

-- Calculate the total face value of all quarters
def total_face_value : ℝ := (num_regular_quarters * face_value_regular) + face_value_special

-- Calculate the total amount Bryden will receive
def total_amount_received : ℝ := multiplier * total_face_value

-- Prove that the total amount Bryden will receive is $22.50
theorem bryden_receives_22_50_dollars : total_amount_received = 22.50 :=
by
  sorry

end bryden_receives_22_50_dollars_l122_122369


namespace polynomial_value_l122_122290

noncomputable def polynomial_spec (p : ℝ) : Prop :=
  p^3 - 5 * p + 1 = 0

theorem polynomial_value (p : ℝ) (h : polynomial_spec p) : 
  p^4 - 3 * p^3 - 5 * p^2 + 16 * p + 2015 = 2018 := 
by
  sorry

end polynomial_value_l122_122290


namespace find_coordinates_of_P_l122_122592

noncomputable def pointP_minimizes_dot_product : Prop :=
  let OA := (2, 2)
  let OB := (4, 1)
  let AP x := (x - 2, -2)
  let BP x := (x - 4, -1)
  let dot_product x := (AP x).1 * (BP x).1 + (AP x).2 * (BP x).2
  ∃ x, (dot_product x = (x - 3) ^ 2 + 1) ∧ (∀ y, dot_product y ≥ dot_product x) ∧ (x = 3)

theorem find_coordinates_of_P : pointP_minimizes_dot_product :=
  sorry

end find_coordinates_of_P_l122_122592


namespace five_digit_number_unique_nonzero_l122_122163

theorem five_digit_number_unique_nonzero (a b c d e : ℕ) (h1 : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e) (h2 : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ e ≠ 0) (h3 : (100 * a + 10 * b + c) * 7 = 100 * c + 10 * d + e) : a = 1 ∧ b = 2 ∧ c = 9 ∧ d = 4 ∧ e = 6 :=
by
  sorry

end five_digit_number_unique_nonzero_l122_122163


namespace ones_digit_of_9_pow_47_l122_122623

theorem ones_digit_of_9_pow_47 : (9 ^ 47) % 10 = 9 := 
by
  sorry

end ones_digit_of_9_pow_47_l122_122623


namespace workman_problem_l122_122527

theorem workman_problem (A B : ℝ) (h1 : A = B / 2) (h2 : (A + B) * 10 = 1) : B = 1 / 15 := by
  sorry

end workman_problem_l122_122527


namespace students_passed_l122_122423

noncomputable def total_students : ℕ := 360
noncomputable def bombed : ℕ := (5 * total_students) / 12
noncomputable def not_bombed : ℕ := total_students - bombed
noncomputable def no_show : ℕ := (7 * not_bombed) / 15
noncomputable def remaining_after_no_show : ℕ := not_bombed - no_show
noncomputable def less_than_D : ℕ := 45
noncomputable def remaining_after_less_than_D : ℕ := remaining_after_no_show - less_than_D
noncomputable def technical_issues : ℕ := remaining_after_less_than_D / 8
noncomputable def passed_students : ℕ := remaining_after_less_than_D - technical_issues

theorem students_passed : passed_students = 59 := by
  sorry

end students_passed_l122_122423


namespace fan_rotation_is_not_translation_l122_122503

def phenomenon := Type

def is_translation (p : phenomenon) : Prop := sorry

axiom elevator_translation : phenomenon
axiom drawer_translation : phenomenon
axiom fan_rotation : phenomenon
axiom car_translation : phenomenon

axiom elevator_is_translation : is_translation elevator_translation
axiom drawer_is_translation : is_translation drawer_translation
axiom car_is_translation : is_translation car_translation

theorem fan_rotation_is_not_translation : ¬ is_translation fan_rotation := sorry

end fan_rotation_is_not_translation_l122_122503


namespace price_of_item_a_l122_122160

theorem price_of_item_a : 
  let coins_1000 := 7
  let coins_100 := 4
  let coins_10 := 5
  let price_1000 := coins_1000 * 1000
  let price_100 := coins_100 * 100
  let price_10 := coins_10 * 10
  let total_price := price_1000 + price_100 + price_10
  total_price = 7450 := by
    sorry

end price_of_item_a_l122_122160


namespace range_of_m_local_odd_function_l122_122194

def is_local_odd_function (f : ℝ → ℝ) : Prop :=
  ∃ x : ℝ, f (-x) = -f x

noncomputable def f (x m : ℝ) : ℝ :=
  9^x - m * 3^x - 3

theorem range_of_m_local_odd_function :
  (∀ m : ℝ, is_local_odd_function (λ x => f x m) ↔ m ∈ Set.Ici (-2)) :=
by
  sorry

end range_of_m_local_odd_function_l122_122194


namespace tan_alpha_l122_122981

theorem tan_alpha (α : ℝ) (h1 : Real.sin (Real.pi - α) = 1 / 3) (h2 : Real.sin (2 * α) > 0) : 
  Real.tan α = Real.sqrt 2 / 4 :=
by 
  sorry

end tan_alpha_l122_122981


namespace david_more_pushups_than_zachary_l122_122103

-- Definitions based on conditions
def david_pushups : ℕ := 37
def zachary_pushups : ℕ := 7

-- Theorem statement proving the answer
theorem david_more_pushups_than_zachary : david_pushups - zachary_pushups = 30 := by
  sorry

end david_more_pushups_than_zachary_l122_122103


namespace box_volume_correct_l122_122871

variables (length width height : ℕ)

def volume_of_box (length width height : ℕ) : ℕ :=
  length * width * height

theorem box_volume_correct :
  volume_of_box 20 15 10 = 3000 :=
by
  -- This is where the proof would go
  sorry 

end box_volume_correct_l122_122871


namespace polar_coordinates_of_point_l122_122091

open Real

theorem polar_coordinates_of_point :
  ∃ r θ : ℝ, r = 4 ∧ θ = 5 * π / 3 ∧ r > 0 ∧ 0 ≤ θ ∧ θ < 2 * π ∧ 
           (∃ x y : ℝ, x = 2 ∧ y = -2 * sqrt 3 ∧ x = r * cos θ ∧ y = r * sin θ) :=
sorry

end polar_coordinates_of_point_l122_122091


namespace total_cost_is_correct_l122_122010

def bus_ride_cost : ℝ := 1.75
def train_ride_cost : ℝ := bus_ride_cost + 6.35
def total_cost : ℝ := bus_ride_cost + train_ride_cost

theorem total_cost_is_correct : total_cost = 9.85 :=
by
  -- proof here
  sorry

end total_cost_is_correct_l122_122010


namespace largest_club_size_is_four_l122_122018

variable {Player : Type} -- Assume Player is a type

-- Definition of the lesson-taking relation
variable (takes_lessons_from : Player → Player → Prop)

-- Club conditions
def club_conditions (A B C : Player) : Prop :=
  (takes_lessons_from A B ∧ ¬takes_lessons_from B C ∧ ¬takes_lessons_from C A) ∨ 
  (¬takes_lessons_from A B ∧ takes_lessons_from B C ∧ ¬takes_lessons_from C A) ∨ 
  (¬takes_lessons_from A B ∧ ¬takes_lessons_from B C ∧ takes_lessons_from C A)

theorem largest_club_size_is_four :
  ∀ (club : Finset Player),
  (∀ (A B C : Player), A ≠ B → B ≠ C → C ≠ A → A ∈ club → B ∈ club → C ∈ club → club_conditions takes_lessons_from A B C) →
  club.card ≤ 4 :=
sorry

end largest_club_size_is_four_l122_122018


namespace number_of_pictures_deleted_l122_122835

-- Definitions based on the conditions
def total_files_deleted : ℕ := 17
def songs_deleted : ℕ := 8
def text_files_deleted : ℕ := 7

-- The question rewritten as a Lean theorem statement
theorem number_of_pictures_deleted : 
  (total_files_deleted - songs_deleted - text_files_deleted) = 2 := 
by
  sorry

end number_of_pictures_deleted_l122_122835


namespace ellipse_parametric_form_l122_122186

theorem ellipse_parametric_form :
  (∃ A B C D E F : ℤ,
    ((∀ t : ℝ, (3 * (Real.sin t - 2)) / (3 - Real.cos t) = x ∧ 
     (2 * (Real.cos t - 4)) / (3 - Real.cos t) = y) → 
    (A * x^2 + B * x * y + C * y^2 + D * x + E * y + F = 0)) ∧
    Int.gcd (Int.natAbs A) (Int.gcd (Int.natAbs B) (Int.gcd (Int.natAbs C) (Int.gcd (Int.natAbs D) (Int.gcd (Int.natAbs E) (Int.natAbs F))))) = 1 ∧
    (Int.natAbs A + Int.natAbs B + Int.natAbs C + Int.natAbs D + Int.natAbs E + Int.natAbs F = 1846)) := 
sorry

end ellipse_parametric_form_l122_122186


namespace polygon_interior_angle_l122_122083

theorem polygon_interior_angle (n : ℕ) (h1 : ∀ (i : ℕ), i < n → (n - 2) * 180 / n = 140): n = 9 := 
sorry

end polygon_interior_angle_l122_122083


namespace consecutive_numbers_average_l122_122046

theorem consecutive_numbers_average (a b c d e f g : ℕ)
  (h1 : (a + b + c + d + e + f + g) / 7 = 9)
  (h2 : 2 * a = g) : 
  7 = 7 :=
by sorry

end consecutive_numbers_average_l122_122046


namespace find_a1_over_d_l122_122796

variable {a : ℕ → ℝ} (d : ℝ)

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n : ℕ, a (n + 1) = a n + d

theorem find_a1_over_d 
  (d_ne_zero : d ≠ 0) 
  (seq : arithmetic_sequence a d) 
  (h : a 2021 = a 20 + a 21) : 
  a 1 / d = 1981 :=
by 
  sorry

end find_a1_over_d_l122_122796


namespace symmetric_function_value_l122_122758

theorem symmetric_function_value (f : ℝ → ℝ)
  (h : ∀ x, f (2^(x-2)) = x) : f 8 = 5 :=
sorry

end symmetric_function_value_l122_122758


namespace community_group_loss_l122_122143

def cookies_bought : ℕ := 800
def cost_per_4_cookies : ℚ := 3 -- dollars per 4 cookies
def sell_per_3_cookies : ℚ := 2 -- dollars per 3 cookies

def cost_per_cookie : ℚ := cost_per_4_cookies / 4
def sell_per_cookie : ℚ := sell_per_3_cookies / 3

def total_cost (n : ℕ) (cost_per_cookie : ℚ) : ℚ := n * cost_per_cookie
def total_revenue (n : ℕ) (sell_per_cookie : ℚ) : ℚ := n * sell_per_cookie

def loss (n : ℕ) (cost_per_cookie sell_per_cookie : ℚ) : ℚ := 
  total_cost n cost_per_cookie - total_revenue n sell_per_cookie

theorem community_group_loss : loss cookies_bought cost_per_cookie sell_per_cookie = 64 := by
  sorry

end community_group_loss_l122_122143


namespace prime_in_range_l122_122682

theorem prime_in_range (p: ℕ) (h_prime: Nat.Prime p) (h_int_roots: ∃ a b: ℤ, a ≠ b ∧ a + b = -p ∧ a * b = -520 * p) : 11 < p ∧ p ≤ 21 := 
by
  sorry

end prime_in_range_l122_122682


namespace num_right_angle_triangles_l122_122842

-- Step d): Lean 4 statement
theorem num_right_angle_triangles {C : ℝ × ℝ} (hC : C.2 = 0) :
  (C = (-2, 0) ∨ C = (4, 0) ∨ C = (1, 0)) ↔ ∃ A B : ℝ × ℝ,
  (A = (-2, 3)) ∧ (B = (4, 3)) ∧ 
  (A.2 = B.2) ∧ (A.1 ≠ B.1) ∧ 
  (((C.1-A.1)*(B.1-A.1) + (C.2-A.2)*(B.2-A.2) = 0) ∨ 
   ((C.1-B.1)*(A.1-B.1) + (C.2-B.2)*(A.2-B.2) = 0)) :=
sorry

end num_right_angle_triangles_l122_122842


namespace more_stable_shooting_performance_l122_122466

theorem more_stable_shooting_performance :
  ∀ (SA2 SB2 : ℝ), SA2 = 1.9 → SB2 = 3 → (SA2 < SB2) → "A" = "Athlete with more stable shooting performance" :=
by
  intros SA2 SB2 h1 h2 h3
  sorry

end more_stable_shooting_performance_l122_122466


namespace pentagonal_grid_toothpicks_l122_122890

theorem pentagonal_grid_toothpicks :
  ∀ (base toothpicks per sides toothpicks per joint : ℕ),
    base = 10 → 
    sides = 4 → 
    toothpicks_per_side = 8 → 
    joints = 5 → 
    toothpicks_per_joint = 1 → 
    (base + sides * toothpicks_per_side + joints * toothpicks_per_joint = 47) :=
by
  intros base sides toothpicks_per_side joints toothpicks_per_joint
  sorry

end pentagonal_grid_toothpicks_l122_122890


namespace bell_ratio_l122_122175

theorem bell_ratio :
  ∃ (B3 B2 : ℕ), 
  B2 = 2 * 50 ∧ 
  50 + B2 + B3 = 550 ∧ 
  (B3 / B2 = 4) := 
sorry

end bell_ratio_l122_122175


namespace bag_cost_is_10_l122_122980

def timothy_initial_money : ℝ := 50
def tshirt_cost : ℝ := 8
def keychain_cost : ℝ := 2
def keychains_per_set : ℝ := 3
def number_of_tshirts : ℝ := 2
def number_of_bags : ℝ := 2
def number_of_keychains : ℝ := 21

noncomputable def cost_of_each_bag : ℝ :=
  let cost_of_tshirts := number_of_tshirts * tshirt_cost
  let remaining_money_after_tshirts := timothy_initial_money - cost_of_tshirts
  let cost_of_keychains := (number_of_keychains / keychains_per_set) * keychain_cost
  let remaining_money_after_keychains := remaining_money_after_tshirts - cost_of_keychains
  remaining_money_after_keychains / number_of_bags

theorem bag_cost_is_10 :
  cost_of_each_bag = 10 := by
  sorry

end bag_cost_is_10_l122_122980


namespace select_student_based_on_variance_l122_122840

-- Define the scores for students A and B
def scoresA : List ℚ := [12.1, 12.1, 12.0, 11.9, 11.8, 12.1]
def scoresB : List ℚ := [12.2, 12.0, 11.8, 12.0, 12.3, 11.7]

-- Define the function to calculate the mean of a list of rational numbers
def mean (scores : List ℚ) : ℚ := (scores.foldr (· + ·) 0) / scores.length

-- Define the function to calculate the variance of a list of rational numbers
def variance (scores : List ℚ) : ℚ :=
  let m := mean scores
  (scores.foldr (λ x acc => acc + (x - m) ^ 2) 0) / scores.length

-- Prove that the variance of student A's scores is less than the variance of student B's scores
theorem select_student_based_on_variance :
  variance scoresA < variance scoresB := by
  sorry

end select_student_based_on_variance_l122_122840


namespace area_of_triangle_8_9_9_l122_122863

noncomputable def triangle_area (a b c : ℕ) : Real :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem area_of_triangle_8_9_9 : triangle_area 8 9 9 = 4 * Real.sqrt 65 :=
by
  sorry

end area_of_triangle_8_9_9_l122_122863


namespace point_on_xaxis_y_coord_zero_l122_122398

theorem point_on_xaxis_y_coord_zero (m : ℝ) (h : (3, m).snd = 0) : m = 0 :=
by 
  -- proof goes here
  sorry

end point_on_xaxis_y_coord_zero_l122_122398


namespace bride_older_than_groom_l122_122787

-- Define the ages of the bride and groom
variables (B G : ℕ)

-- Given conditions
def groom_age : Prop := G = 83
def total_age : Prop := B + G = 185

-- Theorem to prove how much older the bride is than the groom
theorem bride_older_than_groom (h1 : groom_age G) (h2 : total_age B G) : B - G = 19 :=
sorry

end bride_older_than_groom_l122_122787


namespace sequence_is_geometric_not_arithmetic_l122_122054

def is_arithmetic_sequence (a b c : ℕ) : Prop :=
  b - a = c - b

def is_geometric_sequence (a b c : ℕ) : Prop :=
  b / a = c / b

theorem sequence_is_geometric_not_arithmetic :
  ∀ (a₁ a₂ an : ℕ), a₁ = 3 ∧ a₂ = 9 ∧ an = 729 →
    ¬ is_arithmetic_sequence a₁ a₂ an ∧ is_geometric_sequence a₁ a₂ an :=
by
  intros a₁ a₂ an h
  sorry

end sequence_is_geometric_not_arithmetic_l122_122054


namespace corner_cell_revisit_l122_122240

theorem corner_cell_revisit
    (M N : ℕ)
    (hM : M = 101)
    (hN : N = 200)
    (initial_position : ℕ × ℕ)
    (h_initial : initial_position = (0, 0) ∨ initial_position = (0, 200) ∨ initial_position = (101, 0) ∨ initial_position = (101, 200)) :
    ∃ final_position : ℕ × ℕ, 
      final_position = initial_position ∧ (final_position = (0, 0) ∨ final_position = (0, 200) ∨ final_position = (101, 0) ∨ final_position = (101, 200)) :=
by
  sorry

end corner_cell_revisit_l122_122240


namespace middle_marble_radius_l122_122615

theorem middle_marble_radius (r_1 r_5 : ℝ) (h1 : r_1 = 8) (h5 : r_5 = 18) : 
  ∃ r_3 : ℝ, r_3 = 12 :=
by
  let r_3 := Real.sqrt (r_1 * r_5)
  have h : r_3 = 12 := sorry
  exact ⟨r_3, h⟩

end middle_marble_radius_l122_122615


namespace arrange_students_l122_122559

theorem arrange_students 
  (students : Fin 6 → Type) 
  (A B : Type) 
  (h1 : ∃ i j, students i = A ∧ students j = B ∧ (i = j + 1 ∨ j = i + 1)) : 
  (∃ (n : ℕ), n = 240) := 
sorry

end arrange_students_l122_122559


namespace rectangle_perimeter_l122_122958

theorem rectangle_perimeter (s : ℕ) (ABCD_area : 4 * s * s = 400) :
  2 * (2 * s + 2 * s) = 80 :=
by
  -- Skipping the proof
  sorry

end rectangle_perimeter_l122_122958


namespace solution_set_of_inequality_l122_122783

theorem solution_set_of_inequality :
  { x : ℝ | |x^2 - 3 * x| > 4 } = { x : ℝ | x < -1 ∨ x > 4 } :=
sorry

end solution_set_of_inequality_l122_122783


namespace simplify_and_evaluate_fraction_l122_122640

theorem simplify_and_evaluate_fraction (x : ℤ) (hx : x = 5) :
  ((2 * x + 1) / (x - 1) - 1) / ((x + 2) / (x^2 - 2 * x + 1)) = 4 :=
by
  rw [hx]
  sorry

end simplify_and_evaluate_fraction_l122_122640


namespace todd_money_left_l122_122984

-- Define the initial amount of money Todd has
def initial_amount : ℕ := 20

-- Define the number of candy bars Todd buys
def number_of_candy_bars : ℕ := 4

-- Define the cost per candy bar
def cost_per_candy_bar : ℕ := 2

-- Define the total cost of the candy bars
def total_cost : ℕ := number_of_candy_bars * cost_per_candy_bar

-- Define the final amount of money Todd has left
def final_amount : ℕ := initial_amount - total_cost

-- The statement to be proven in Lean
theorem todd_money_left : final_amount = 12 := by
  -- The proof is omitted
  sorry

end todd_money_left_l122_122984


namespace magnitude_of_Z_l122_122135

-- Define the complex number Z
def Z : ℂ := 3 - 4 * Complex.I

-- Define the theorem to prove the magnitude of Z
theorem magnitude_of_Z : Complex.abs Z = 5 := by
  sorry

end magnitude_of_Z_l122_122135


namespace vikki_worked_42_hours_l122_122681

-- Defining the conditions
def hourly_pay_rate : ℝ := 10
def tax_deduction : ℝ := 0.20 * hourly_pay_rate
def insurance_deduction : ℝ := 0.05 * hourly_pay_rate
def union_dues : ℝ := 5
def take_home_pay : ℝ := 310

-- Equation derived from the given conditions
def total_hours_worked (h : ℝ) : Prop :=
  hourly_pay_rate * h - (tax_deduction * h + insurance_deduction * h + union_dues) = take_home_pay

-- Prove that Vikki worked for 42 hours given the conditions
theorem vikki_worked_42_hours : total_hours_worked 42 := by
  sorry

end vikki_worked_42_hours_l122_122681


namespace initial_blue_balls_l122_122181

theorem initial_blue_balls (B : ℕ) 
  (h1 : 18 - 3 = 15) 
  (h2 : (B - 3) / 15 = 1 / 5) : 
  B = 6 :=
by sorry

end initial_blue_balls_l122_122181


namespace second_group_product_number_l122_122800

theorem second_group_product_number (a₀ : ℕ) (h₀ : 0 ≤ a₀ ∧ a₀ < 20)
  (h₁ : 4 * 20 + a₀ = 94) : 1 * 20 + a₀ = 34 :=
by
  sorry

end second_group_product_number_l122_122800


namespace nate_age_is_14_l122_122474

def nate_current_age (N : ℕ) : Prop :=
  ∃ E : ℕ, E = N / 2 ∧ N - E = 7

theorem nate_age_is_14 : nate_current_age 14 :=
by {
  sorry
}

end nate_age_is_14_l122_122474


namespace days_in_month_l122_122725

theorem days_in_month
  (monthly_production : ℕ)
  (production_per_half_hour : ℚ)
  (hours_per_day : ℕ)
  (daily_production : ℚ)
  (days_in_month : ℚ) :
  monthly_production = 8400 ∧
  production_per_half_hour = 6.25 ∧
  hours_per_day = 24 ∧
  daily_production = production_per_half_hour * 2 * hours_per_day ∧
  days_in_month = monthly_production / daily_production
  → days_in_month = 28 :=
by
  sorry

end days_in_month_l122_122725


namespace algebraic_simplification_evaluate_expression_for_x2_evaluate_expression_for_x_neg2_l122_122720

theorem algebraic_simplification (x : ℤ) (h1 : -3 < x) (h2 : x < 3) (h3 : x ≠ 0) (h4 : x ≠ 1) (h5 : x ≠ -1) :
  (x - (x / (x + 1))) / (1 + (1 / (x^2 - 1))) = x - 1 :=
sorry

theorem evaluate_expression_for_x2 (h1 : -3 < 2) (h2 : 2 < 3) (h3 : 2 ≠ 0) (h4 : 2 ≠ 1) (h5 : 2 ≠ -1) :
  (2 - (2 / (2 + 1))) / (1 + (1 / (2^2 - 1))) = 1 :=
sorry

theorem evaluate_expression_for_x_neg2 (h1 : -3 < -2) (h2 : -2 < 3) (h3 : -2 ≠ 0) (h4 : -2 ≠ 1) (h5 : -2 ≠ -1) :
  (-2 - (-2 / (-2 + 1))) / (1 + (1 / ((-2)^2 - 1))) = -3 :=
sorry

end algebraic_simplification_evaluate_expression_for_x2_evaluate_expression_for_x_neg2_l122_122720


namespace triangle_properties_l122_122995

theorem triangle_properties (A B C : ℝ) (a b c : ℝ) 
  (angle_A : A = 30) (angle_B : B = 45) (side_a : a = Real.sqrt 2) :
  b = 2 ∧ (1 / 2) * a * b * Real.sin (105 * Real.pi / 180) = (Real.sqrt 3 + 1) / 2 := by
sorry

end triangle_properties_l122_122995


namespace chord_intersection_probability_l122_122273

theorem chord_intersection_probability
  (points : Finset Point)
  (hp : points.card = 2000)
  (A B C D E : Point)
  (hA : A ∈ points)
  (hB : B ∈ points)
  (hC : C ∈ points)
  (hD : D ∈ points)
  (hE : E ∈ points)
  (distinct : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ C ≠ D ∧ C ≠ E ∧ D ≠ E)
  : probability_chord_intersection := by
    sorry

end chord_intersection_probability_l122_122273


namespace cone_volume_l122_122504

theorem cone_volume (l : ℝ) (θ : ℝ) (h r V : ℝ)
  (h_l : l = 5)
  (h_θ : θ = (8 * Real.pi) / 5)
  (h_arc_length : 2 * Real.pi * r = l * θ)
  (h_radius: r = 4)
  (h_height : h = Real.sqrt (l^2 - r^2))
  (h_volume_eq : V = (1 / 3) * Real.pi * r^2 * h) :
  V = 16 * Real.pi :=
by
  -- proof goes here
  sorry

end cone_volume_l122_122504


namespace expression_of_quadratic_function_coordinates_of_vertex_l122_122688

def quadratic_function_through_points (a b : ℝ) : Prop :=
  (0 = a * (-3)^2 + b * (-3) + 3) ∧ (-5 = a * 2^2 + b * 2 + 3)

theorem expression_of_quadratic_function :
  ∃ a b : ℝ, quadratic_function_through_points a b ∧ ∀ x : ℝ, -x^2 - 2 * x + 3 = a * x^2 + b * x + 3 :=
by
  sorry

theorem coordinates_of_vertex :
  - (1 : ℝ) * (1 : ℝ) = (-1) / (2 * (-1)) ∧ 4 = -(1 - (-1) + 3) + 4 :=
by
  sorry

end expression_of_quadratic_function_coordinates_of_vertex_l122_122688


namespace least_n_condition_l122_122019

-- Define the conditions and the question in Lean 4
def jackson_position (n : ℕ) : ℕ := sorry  -- Defining the position of Jackson after n steps

def expected_value (n : ℕ) : ℝ := sorry  -- Defining the expected value E_n

theorem least_n_condition : ∃ n : ℕ, (1 / expected_value n > 2017) ∧ (∀ m < n, 1 / expected_value m ≤ 2017) ∧ n = 13446 :=
by {
  -- Jackson starts at position 1
  -- The conditions described in the problem will be formulated here
  -- We need to show that the least n such that 1 / E_n > 2017 is 13446
  sorry
}

end least_n_condition_l122_122019


namespace field_length_is_112_l122_122657

-- Define the conditions
def is_pond_side_length : ℕ := 8
def pond_area : ℕ := is_pond_side_length * is_pond_side_length
def pond_to_field_area_ratio : ℚ := 1 / 98

-- Define the field properties
def field_area (w l : ℕ) : ℕ := w * l

-- Expressing the condition given length is double the width
def length_double_width (w l : ℕ) : Prop := l = 2 * w

-- Equating the areas based on the ratio given
def area_condition (w l : ℕ) : Prop := pond_area = pond_to_field_area_ratio * field_area w l

-- The main theorem
theorem field_length_is_112 : ∃ w l, length_double_width w l ∧ area_condition w l ∧ l = 112 := by
  sorry

end field_length_is_112_l122_122657


namespace unique_arrangements_MOON_l122_122427

theorem unique_arrangements_MOON : 
  let M := 1
  let O := 2
  let N := 1
  let total_letters := 4
  (Nat.factorial total_letters / (Nat.factorial O)) = 12 :=
by
  sorry

end unique_arrangements_MOON_l122_122427


namespace cindy_first_to_get_five_l122_122500

def probability_of_five : ℚ := 1 / 6

def anne_turn (p: ℚ) : ℚ := 1 - p
def cindy_turn (p: ℚ) : ℚ := p
def none_get_five (p: ℚ) : ℚ := (1 - p)^3

theorem cindy_first_to_get_five : 
    (∑' n, (anne_turn probability_of_five * none_get_five probability_of_five ^ n) * 
                cindy_turn probability_of_five) = 30 / 91 := by 
    sorry

end cindy_first_to_get_five_l122_122500


namespace total_balls_l122_122753

def black_balls : ℕ := 8
def white_balls : ℕ := 6 * black_balls
theorem total_balls : white_balls + black_balls = 56 := 
by 
  sorry

end total_balls_l122_122753


namespace probability_red_buttons_l122_122005

/-- 
Initial condition: Jar A contains 6 red buttons and 10 blue buttons.
Carla removes the same number of red buttons as blue buttons from Jar A and places them in Jar B.
Jar A's state after action: Jar A retains 3/4 of its original number of buttons.
Question: What is the probability that both selected buttons are red? Express your answer as a common fraction.
-/
theorem probability_red_buttons :
  let initial_red_a := 6
  let initial_blue_a := 10
  let total_buttons_a := initial_red_a + initial_blue_a
  
  -- Jar A after removing buttons
  let retained_fraction := 3 / 4
  let remaining_buttons_a := retained_fraction * total_buttons_a
  let removed_buttons := total_buttons_a - remaining_buttons_a
  let removed_red_buttons := removed_buttons / 2
  let removed_blue_buttons := removed_buttons / 2
  
  -- Remaining red and blue buttons in Jar A
  let remaining_red_a := initial_red_a - removed_red_buttons
  let remaining_blue_a := initial_blue_a - removed_blue_buttons

  -- Total remaining buttons in Jar A
  let total_remaining_a := remaining_red_a + remaining_blue_a

  -- Jar B contains the removed buttons
  let total_buttons_b := removed_buttons
  
  -- Probability calculations
  let probability_red_a := remaining_red_a / total_remaining_a
  let probability_red_b := removed_red_buttons / total_buttons_b

  -- Combined probability of selecting red button from both jars
  probability_red_a * probability_red_b = 1 / 6 :=
by
  sorry

end probability_red_buttons_l122_122005


namespace smallest_integer_CC4_DD6_rep_l122_122686

-- Lean 4 Statement
theorem smallest_integer_CC4_DD6_rep (C D : ℕ) (hC : C < 4) (hD : D < 6) :
  (5 * C = 7 * D) → (5 * C = 35 ∧ 7 * D = 35) :=
by
  sorry

end smallest_integer_CC4_DD6_rep_l122_122686


namespace frog_climb_time_l122_122341

-- Define the problem as an assertion within Lean.
theorem frog_climb_time 
  (well_depth : ℕ) (climb_up : ℕ) (slide_down : ℕ) (time_per_meter: ℕ) (climb_start_time : ℕ) 
  (time_to_slide_multiplier: ℚ)
  (time_to_second_position: ℕ) 
  (final_distance: ℕ) 
  (total_time: ℕ)
  (h_start : well_depth = 12)
  (h_climb_up: climb_up = 3)
  (h_slide_down : slide_down = 1)
  (h_time_per_meter : time_per_meter = 1)
  (h_time_to_slide_multiplier: time_to_slide_multiplier = 1/3)
  (h_time_to_second_position : climb_start_time = 8 * 60 /\ time_to_second_position = 8 * 60 + 17)
  (h_final_distance : final_distance = 3)
  (h_total_time: total_time = 22) :
  
  ∃ (t: ℕ), 
    t = total_time := 
by
  sorry

end frog_climb_time_l122_122341


namespace divides_14_pow_n_minus_27_for_all_natural_numbers_l122_122790

theorem divides_14_pow_n_minus_27_for_all_natural_numbers :
  ∀ n : ℕ, 13 ∣ 14^n - 27 :=
by sorry

end divides_14_pow_n_minus_27_for_all_natural_numbers_l122_122790


namespace lassis_from_mangoes_l122_122050

theorem lassis_from_mangoes (m l m' : ℕ) (h : m' = 18) (hlm : l / m = 8 / 3) : l / m' = 48 / 18 :=
by
  sorry

end lassis_from_mangoes_l122_122050


namespace light_stripes_total_area_l122_122133

theorem light_stripes_total_area (x : ℝ) (h : 45 * x = 135) :
  2 * x + 4 * x + 6 * x + 8 * x = 60 := 
sorry

end light_stripes_total_area_l122_122133


namespace value_at_minus_two_l122_122869

def f (x : ℝ) : ℝ := x^2 + 3 * x - 5

theorem value_at_minus_two : f (-2) = -7 := by
  sorry

end value_at_minus_two_l122_122869


namespace triangle_inequality_l122_122345

theorem triangle_inequality (a b c : ℕ) : 
    a + b > c ∧ a + c > b ∧ b + c > a ↔ 
    (a, b, c) = (2, 3, 4) ∨ (a, b, c) = (3, 4, 7) ∨ (a, b, c) = (4, 6, 2) ∨ (a, b, c) = (7, 10, 2)
    → (a + b > c ∧ a + c > b ∧ b + c > a ↔ (a, b, c) = (2, 3, 4)) ∧
      (a + b = c ∨ a + c = b ∨ b + c = a         ↔ (a, b, c) = (3, 4, 7)) ∧
      (a + b = c ∨ a + c = b ∨ b + c = a        ↔ (a, b, c) = (4, 6, 2)) ∧
      (a + b < c ∨ a + c < b ∨ b + c < a        ↔ (a, b, c) = (7, 10, 2)) :=
sorry

end triangle_inequality_l122_122345


namespace joan_kittens_total_l122_122539

-- Definition of the initial conditions
def joan_original_kittens : ℕ := 8
def neighbor_original_kittens : ℕ := 6
def joan_gave_away : ℕ := 2
def neighbor_gave_away : ℕ := 4
def joan_adopted_from_neighbor : ℕ := 3

-- The final number of kittens Joan has
def joan_final_kittens : ℕ :=
  let joan_remaining := joan_original_kittens - joan_gave_away
  let neighbor_remaining := neighbor_original_kittens - neighbor_gave_away
  let adopted := min joan_adopted_from_neighbor neighbor_remaining
  joan_remaining + adopted

theorem joan_kittens_total : joan_final_kittens = 8 := 
by 
  -- Lean proof would go here, but adding sorry for now
  sorry

end joan_kittens_total_l122_122539


namespace square_side_length_l122_122467

theorem square_side_length (A : ℝ) (side : ℝ) (h₁ : A = side^2) (h₂ : A = 12) : side = 2 * Real.sqrt 3 := 
by
  sorry

end square_side_length_l122_122467


namespace problem_condition_holds_l122_122247

theorem problem_condition_holds (x y : ℝ) (h₁ : x + 0.35 * y - (x + y) = 200) : y = -307.69 :=
sorry

end problem_condition_holds_l122_122247


namespace max_a_l122_122545

-- Define the conditions
def line_equation (m : ℚ) (x : ℤ) : ℚ := m * x + 3

def no_lattice_points (m : ℚ) : Prop :=
  ∀ (x : ℤ), 1 ≤ x ∧ x ≤ 50 → ¬ ∃ (y : ℤ), line_equation m x = y

def m_range (m a : ℚ) : Prop := (2 : ℚ) / 5 < m ∧ m < a

-- Define the problem statement
theorem max_a (a : ℚ) : (a = 22 / 51) ↔ (∃ m, no_lattice_points m ∧ m_range m a) :=
by 
  sorry

end max_a_l122_122545


namespace total_points_correct_l122_122933

-- Define the number of teams
def num_teams : ℕ := 16

-- Define the number of draws
def num_draws : ℕ := 30

-- Define the scoring system
def points_for_win : ℕ := 3
def points_for_draw : ℕ := 1
def loss_deduction_threshold : ℕ := 3
def points_deduction_per_threshold : ℕ := 1

-- Define the total number of games
def total_games : ℕ := num_teams * (num_teams - 1) / 2

-- Define the number of wins (non-draw games)
def num_wins : ℕ := total_games - num_draws

-- Define the total points from wins
def total_points_from_wins : ℕ := num_wins * points_for_win

-- Define the total points from draws
def total_points_from_draws : ℕ := num_draws * points_for_draw * 2

-- Define the total points (as no team lost more than twice, no deductions apply)
def total_points : ℕ := total_points_from_wins + total_points_from_draws

theorem total_points_correct :
  total_points = 330 := by
  sorry

end total_points_correct_l122_122933


namespace find_c_plus_d_l122_122949

noncomputable def g (c d : ℝ) (x : ℝ) : ℝ :=
if x < 3 then c * x + d else 10 - 2 * x

theorem find_c_plus_d (c d : ℝ) (h : ∀ x, g c d (g c d x) = x) : c + d = 4.5 :=
sorry

end find_c_plus_d_l122_122949


namespace jen_profit_is_960_l122_122872

def buying_price : ℕ := 80
def selling_price : ℕ := 100
def num_candy_bars_bought : ℕ := 50
def num_candy_bars_sold : ℕ := 48

def profit_per_candy_bar := selling_price - buying_price
def total_profit := profit_per_candy_bar * num_candy_bars_sold

theorem jen_profit_is_960 : total_profit = 960 := by
  sorry

end jen_profit_is_960_l122_122872


namespace final_toy_count_correct_l122_122501

def initial_toy_count : ℝ := 5.3
def tuesday_toys_left (initial: ℝ) : ℝ := initial * 0.605
def tuesday_new_toys : ℝ := 3.6
def wednesday_toys_left (tuesday_total: ℝ) : ℝ := tuesday_total * 0.498
def wednesday_new_toys : ℝ := 2.4
def thursday_toys_left (wednesday_total: ℝ) : ℝ := wednesday_total * 0.692
def thursday_new_toys : ℝ := 4.5

def total_toys (initial: ℝ) : ℝ :=
  let after_tuesday := tuesday_toys_left initial + tuesday_new_toys
  let after_wednesday := wednesday_toys_left after_tuesday + wednesday_new_toys
  let after_thursday := thursday_toys_left after_wednesday + thursday_new_toys
  after_thursday

def toys_lost_tuesday (initial: ℝ) (left: ℝ) : ℝ := initial - left
def toys_lost_wednesday (tuesday_total: ℝ) (left: ℝ) : ℝ := tuesday_total - left
def toys_lost_thursday (wednesday_total: ℝ) (left: ℝ) : ℝ := wednesday_total - left
def total_lost_toys (initial: ℝ) : ℝ :=
  let tuesday_left := tuesday_toys_left initial
  let tuesday_total := tuesday_left + tuesday_new_toys
  let wednesday_left := wednesday_toys_left tuesday_total
  let wednesday_total := wednesday_left + wednesday_new_toys
  let thursday_left := thursday_toys_left wednesday_total
  let lost_tuesday := toys_lost_tuesday initial tuesday_left
  let lost_wednesday := toys_lost_wednesday tuesday_total wednesday_left
  let lost_thursday := toys_lost_thursday wednesday_total thursday_left
  lost_tuesday + lost_wednesday + lost_thursday

def final_toy_count (initial: ℝ) : ℝ :=
  let current_toys := total_toys initial
  let lost_toys := total_lost_toys initial
  current_toys + lost_toys

theorem final_toy_count_correct :
  final_toy_count initial_toy_count = 15.8 := sorry

end final_toy_count_correct_l122_122501


namespace edward_cards_l122_122183

noncomputable def num_cards_each_binder : ℝ := (7496.5 + 27.7) / 23
noncomputable def num_cards_fewer_binder : ℝ := num_cards_each_binder - 27.7

theorem edward_cards : 
  (⌊num_cards_each_binder + 0.5⌋ = 327) ∧ (⌊num_cards_fewer_binder + 0.5⌋ = 299) :=
by
  sorry

end edward_cards_l122_122183


namespace frequencies_of_first_class_products_confidence_in_difference_of_quality_l122_122887

theorem frequencies_of_first_class_products 
  (aA bA aB bB : ℕ) 
  (total_A total_B total_products : ℕ) 
  (hA : aA = 150) (hB : bA = 50) (hC : aB = 120) (hD : bB = 80) 
  (hE : total_A = 200) (hF : total_B = 200) (h_total : total_products = total_A + total_B) : 
  aA / total_A = 3 / 4 ∧ aB / total_B = 3 / 5 := 
by
  sorry

theorem confidence_in_difference_of_quality 
  (n a b c d : ℕ) 
  (h_n : n = 400) (hA : a = 150) (hB : b = 50) (hC : c = 120) (hD : d = 80) 
  (K2 : ℝ) 
  (hK2 : K2 = n * ((a * d - b * c) ^ 2) / ((a + b) * (c + d) * (a + c) * (b + d))) : 
  K2 > 6.635 ∧ K2 < 10.828 := 
by
  sorry

end frequencies_of_first_class_products_confidence_in_difference_of_quality_l122_122887


namespace find_m_l122_122132

-- Definitions of the given vectors a, b, and c
def vec_a (m : ℝ) : ℝ × ℝ := (1, m)
def vec_b : ℝ × ℝ := (2, 5)
def vec_c (m : ℝ) : ℝ × ℝ := (m, 3)

-- Definition of vector addition and subtraction
def vec_add (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 + v.1, u.2 + v.2)
def vec_sub (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 - v.1, u.2 - v.2)

-- Parallel vectors condition: the ratio of their components must be equal
def parallel (u v : ℝ × ℝ) : Prop :=
  u.1 * v.2 = u.2 * v.1

-- The main theorem stating the desired result
theorem find_m (m : ℝ) :
  parallel (vec_add (vec_a m) (vec_c m)) (vec_sub (vec_a m) vec_b) ↔ 
  m = (3 + Real.sqrt 17) / 2 ∨ m = (3 - Real.sqrt 17) / 2 :=
by
  sorry

end find_m_l122_122132


namespace locus_of_M_l122_122188

theorem locus_of_M (k : ℝ) (A B M : ℝ × ℝ) (hA : A.1 ≥ 0 ∧ A.2 = 0) (hB : B.2 ≥ 0 ∧ B.1 = 0) (h_sum : A.1 + B.2 = k) :
    ∃ (M : ℝ × ℝ), (M.1 - k / 2)^2 + (M.2 - k / 2)^2 = k^2 / 2 :=
by
  sorry

end locus_of_M_l122_122188


namespace ellipse_slope_ratio_l122_122979

theorem ellipse_slope_ratio (a b x1 y1 x2 y2 c k1 k2 : ℝ) (h1 : a > b) (h2 : b > 0)
  (h3 : c = a / 2) (h4 : a = 2) (h5 : c = 1) (h6 : b = Real.sqrt 3) 
  (h7 : 3 * x1 ^ 2 + 4 * y1 ^ 2 = 12 * c ^ 2) 
  (h8 : 3 * x2 ^ 2 + 4 * y2 ^ 2 = 12 * c ^ 2) 
  (h9 : x1 = y1 - c) (h10 : x2 = y2 - c)
  (h11 : y1^2 = 9 / 4)
  (h12 : y1 = -3 / 2 ∨ y1 = 3 / 2) 
  (h13 : k1 = -3 / 2) 
  (h14 : k2 = -1 / 2) :
  k1 / k2 = 3 := 
  sorry

end ellipse_slope_ratio_l122_122979


namespace hyperbola_equation_l122_122880

-- Lean 4 statement
theorem hyperbola_equation (a b : ℝ) (hpos_a : a > 0) (hpos_b : b > 0)
    (length_imag_axis : 2 * b = 2)
    (asymptote : ∃ (k : ℝ), ∀ x : ℝ, y = k * x ↔ y = (1 / 2) * x) :
  (x y : ℝ) → (x^2 / a^2) - (y^2 / b^2) = 1 ↔ (x^2 / 4) - (y^2 / 1) = 1 :=
by 
  intros
  sorry

end hyperbola_equation_l122_122880


namespace maximum_value_of_a_l122_122726

theorem maximum_value_of_a (a : ℝ) : 
  (∀ x : ℝ, x^2 + |2 * x - 6| ≥ a) ↔ a ≤ 5 :=
by
  sorry

end maximum_value_of_a_l122_122726


namespace point_B_represent_l122_122715

-- Given conditions
def point_A := -2
def units_moved := 4

-- Lean statement to prove
theorem point_B_represent : 
  ∃ B : ℤ, (B = point_A - units_moved) ∨ (B = point_A + units_moved) := by
    sorry

end point_B_represent_l122_122715


namespace find_width_of_lot_l122_122531

noncomputable def volume_of_rectangular_prism (l w h : ℝ) : ℝ := l * w * h

theorem find_width_of_lot
  (l h v : ℝ)
  (h_len : l = 40)
  (h_height : h = 2)
  (h_volume : v = 1600)
  : ∃ w : ℝ, volume_of_rectangular_prism l w h = v ∧ w = 20 := by
  use 20
  simp [volume_of_rectangular_prism, h_len, h_height, h_volume]
  sorry

end find_width_of_lot_l122_122531


namespace different_language_classes_probability_l122_122190

theorem different_language_classes_probability :
  let total_students := 40
  let french_students := 28
  let spanish_students := 26
  let german_students := 15
  let french_and_spanish_students := 10
  let french_and_german_students := 6
  let spanish_and_german_students := 8
  let all_three_languages_students := 3
  let total_pairs := Nat.choose total_students 2
  let french_only := french_students - (french_and_spanish_students + french_and_german_students - all_three_languages_students) - all_three_languages_students
  let spanish_only := spanish_students - (french_and_spanish_students + spanish_and_german_students - all_three_languages_students) - all_three_languages_students
  let german_only := german_students - (french_and_german_students + spanish_and_german_students - all_three_languages_students) - all_three_languages_students
  let french_only_pairs := Nat.choose french_only 2
  let spanish_only_pairs := Nat.choose spanish_only 2
  let german_only_pairs := Nat.choose german_only 2
  let single_language_pairs := french_only_pairs + spanish_only_pairs + german_only_pairs
  let different_classes_probability := 1 - (single_language_pairs / total_pairs)
  different_classes_probability = (34 / 39) :=
by
  sorry

end different_language_classes_probability_l122_122190


namespace jane_mean_score_l122_122167

-- Define Jane's scores as a list
def jane_scores : List ℕ := [95, 88, 94, 86, 92, 91]

-- Define the total number of quizzes
def total_quizzes : ℕ := 6

-- Define the sum of Jane's scores
def sum_scores : ℕ := 95 + 88 + 94 + 86 + 92 + 91

-- Define the mean score calculation
def mean_score : ℕ := sum_scores / total_quizzes

-- The theorem to state Jane's mean score
theorem jane_mean_score : mean_score = 91 := by
  -- This theorem statement correctly reflects the mathematical problem provided.
  sorry

end jane_mean_score_l122_122167


namespace probability_queen_then_diamond_l122_122098

-- Define a standard deck of 52 cards
def deck := List.range 52

-- Define a function to check if a card is a Queen
def is_queen (card : ℕ) : Prop :=
card % 13 = 10

-- Define a function to check if a card is a Diamond. Here assuming index for diamond starts at 0 and ends at 12
def is_diamond (card : ℕ) : Prop :=
card / 13 = 0

-- The main theorem statement
theorem probability_queen_then_diamond : 
  let prob := 1 / 52 * 12 / 51 + 3 / 52 * 13 / 51
  prob = 52 / 221 :=
by
  sorry

end probability_queen_then_diamond_l122_122098


namespace initial_boys_l122_122728

-- Define the initial condition
def initial_girls : ℕ := 18
def additional_girls : ℕ := 7
def quitting_boys : ℕ := 4
def total_children_after_changes : ℕ := 36

-- Define the initial number of boys
variable (B : ℕ)

-- State the main theorem
theorem initial_boys (h : 25 + (B - 4) = 36) : B = 15 :=
by
  sorry

end initial_boys_l122_122728


namespace functional_equation_solution_form_l122_122945

noncomputable def functional_equation_problem (f : ℝ → ℝ) :=
  ∀ x y : ℝ, (x + y) * (f x - f y) = (x - y) * f (x + y)

theorem functional_equation_solution_form :
  (∀ f : ℝ → ℝ, (functional_equation_problem f) → (∃ a b : ℝ, ∀ x : ℝ, f x = a * x ^ 2 + b * x)) :=
by 
  sorry

end functional_equation_solution_form_l122_122945


namespace sum_of_remainders_mod_15_l122_122900

theorem sum_of_remainders_mod_15 (a b c : ℕ) (h1 : a % 15 = 11) (h2 : b % 15 = 13) (h3 : c % 15 = 14) :
  (a + b + c) % 15 = 8 :=
by
  sorry

end sum_of_remainders_mod_15_l122_122900


namespace boys_at_park_l122_122750

theorem boys_at_park (girls parents groups people_per_group : ℕ) 
  (h_girls : girls = 14) 
  (h_parents : parents = 50)
  (h_groups : groups = 3) 
  (h_people_per_group : people_per_group = 25) : 
  (groups * people_per_group) - (girls + parents) = 11 := 
by 
  -- Not providing the proof, only the statement
  sorry

end boys_at_park_l122_122750


namespace pos_real_unique_solution_l122_122061

theorem pos_real_unique_solution (x : ℝ) (hx_pos : 0 < x) (h : (x - 3) / 8 = 5 / (x - 8)) : x = 16 :=
sorry

end pos_real_unique_solution_l122_122061


namespace find_missing_number_l122_122172

theorem find_missing_number (x : ℤ) (h : (4 + 3) + (8 - x - 1) = 11) : x = 3 :=
sorry

end find_missing_number_l122_122172


namespace train_arrival_day_l122_122239

-- Definitions for the start time and journey duration
def start_time : ℕ := 0  -- early morning (0 hours) on Tuesday
def journey_duration : ℕ := 28  -- 28 hours

-- Proving the arrival time
theorem train_arrival_day (start_time journey_duration : ℕ) :
  journey_duration == 28 → 
  start_time == 0 → 
  (journey_duration / 24, journey_duration % 24) == (1, 4) → 
  true := 
by
  intros
  sorry

end train_arrival_day_l122_122239


namespace interval_of_x₀_l122_122116

-- Definition of the problem
variable (x₀ : ℝ)

-- Conditions
def condition_1 := x₀ > 0 ∧ x₀ < Real.pi
def condition_2 := Real.sin x₀ + Real.cos x₀ = 2 / 3

-- Proof problem statement
theorem interval_of_x₀ 
  (h1 : condition_1 x₀)
  (h2 : condition_2 x₀) : 
  x₀ > 7 * Real.pi / 12 ∧ x₀ < 3 * Real.pi / 4 := 
sorry

end interval_of_x₀_l122_122116


namespace best_model_is_A_l122_122339

-- Definitions of the models and their R^2 values
def ModelA_R_squared : ℝ := 0.95
def ModelB_R_squared : ℝ := 0.81
def ModelC_R_squared : ℝ := 0.50
def ModelD_R_squared : ℝ := 0.32

-- Definition stating that the best fitting model is the one with the highest R^2 value
def best_fitting_model (R_squared_A R_squared_B R_squared_C R_squared_D: ℝ) : Prop :=
  R_squared_A > R_squared_B ∧ R_squared_A > R_squared_C ∧ R_squared_A > R_squared_D

-- Proof statement
theorem best_model_is_A : best_fitting_model ModelA_R_squared ModelB_R_squared ModelC_R_squared ModelD_R_squared :=
by
  -- Skipping the proof logic
  sorry

end best_model_is_A_l122_122339


namespace train_second_speed_20_l122_122078

variable (x v: ℕ)

theorem train_second_speed_20 
  (h1 : (x / 40) + (2 * x / v) = (6 * x / 48)) : 
  v = 20 := by 
  sorry

end train_second_speed_20_l122_122078


namespace problem_may_not_be_equal_l122_122048

-- Define the four pairs of expressions
def expr_A (a b : ℕ) := (a + b) = (b + a)
def expr_B (a : ℕ) := (3 * a) = (a + a + a)
def expr_C (a b : ℕ) := (3 * (a + b)) ≠ (3 * a + b)
def expr_D (a : ℕ) := (a ^ 3) = (a * a * a)

-- State the theorem stating that the expression in condition C may not be equal
theorem problem_may_not_be_equal (a b : ℕ) : (3 * (a + b)) ≠ (3 * a + b) :=
by
  sorry

end problem_may_not_be_equal_l122_122048


namespace rose_paid_after_discount_l122_122543

-- Define the conditions as given in the problem statement
def original_price : ℕ := 10
def discount_rate : ℕ := 10

-- Define the theorem that needs to be proved
theorem rose_paid_after_discount : 
  original_price - (original_price * discount_rate / 100) = 9 :=
by
  -- Here we skip the proof with sorry
  sorry

end rose_paid_after_discount_l122_122543


namespace find_y_value_l122_122824
-- Import the necessary Lean library

-- Define the conditions and the target theorem
theorem find_y_value (h : 6 * y + 3 * y + y + 4 * y = 360) : y = 180 / 7 :=
by
  sorry

end find_y_value_l122_122824


namespace find_function_l122_122001

noncomputable def f (x : ℝ) : ℝ := sorry 

theorem find_function (f : ℝ → ℝ)
  (cond : ∀ x y z : ℝ, x + y + z = 0 → f (x^3) + (f y)^3 + (f z)^3 = 3 * x * y * z) :
  ∀ x : ℝ, f x = x :=
by
  sorry

end find_function_l122_122001


namespace total_surface_area_of_cube_l122_122041

theorem total_surface_area_of_cube (edge_sum : ℕ) (h_edge_sum : edge_sum = 180) :
  ∃ (S : ℕ), S = 1350 := 
by
  sorry

end total_surface_area_of_cube_l122_122041


namespace bridge_length_l122_122263

-- Definitions based on conditions
def Lt : ℕ := 148
def Skm : ℕ := 45
def T : ℕ := 30

-- Conversion from km/h to m/s
def conversion_factor : ℕ := 1000 / 3600
def Sm : ℝ := Skm * conversion_factor

-- Calculation of distance traveled in 30 seconds
def distance : ℝ := Sm * T

-- The length of the bridge
def L_bridge : ℝ := distance - Lt

theorem bridge_length : L_bridge = 227 := sorry

end bridge_length_l122_122263


namespace suff_condition_not_necc_condition_l122_122594

variable (x : ℝ)

def A : Prop := 0 < x ∧ x < 5
def B : Prop := |x - 2| < 3

theorem suff_condition : A x → B x := by
  sorry

theorem not_necc_condition : B x → ¬ A x := by
  sorry

end suff_condition_not_necc_condition_l122_122594


namespace integer_solutions_of_inequality_l122_122285

theorem integer_solutions_of_inequality (x : ℤ) : 
  (-4 < 1 - 3 * (x: ℤ) ∧ 1 - 3 * (x: ℤ) ≤ 4) ↔ (x = -1 ∨ x = 0 ∨ x = 1) := 
by 
  sorry

end integer_solutions_of_inequality_l122_122285


namespace soccer_ball_purchase_l122_122510

theorem soccer_ball_purchase (wholesale_price retail_price profit remaining_balls final_profit : ℕ)
  (h1 : wholesale_price = 30)
  (h2 : retail_price = 45)
  (h3 : profit = retail_price - wholesale_price)
  (h4 : remaining_balls = 30)
  (h5 : final_profit = 1500) :
  ∃ (initial_balls : ℕ), (initial_balls - remaining_balls) * profit = final_profit ∧ initial_balls = 130 :=
by
  sorry

end soccer_ball_purchase_l122_122510


namespace intersecting_lines_l122_122403

theorem intersecting_lines (a b : ℝ) (h1 : 1 = 1 / 4 * 2 + a) (h2 : 2 = 1 / 4 * 1 + b) : 
  a + b = 9 / 4 := 
sorry

end intersecting_lines_l122_122403


namespace wade_customers_sunday_l122_122249

theorem wade_customers_sunday :
  let tips_per_customer := 2
  let customers_friday := 28
  let customers_saturday := 3 * customers_friday
  let total_tips := 296
  let tips_friday := customers_friday * tips_per_customer
  let tips_saturday := customers_saturday * tips_per_customer
  let tips_fri_sat := tips_friday + tips_saturday
  let tips_sunday := total_tips - tips_fri_sat
  let customers_sunday := tips_sunday / tips_per_customer
  customers_sunday = 36 :=
by
  let tips_per_customer := 2
  let customers_friday := 28
  let customers_saturday := 3 * customers_friday
  let total_tips := 296
  let tips_friday := customers_friday * tips_per_customer
  let tips_saturday := customers_saturday * tips_per_customer
  let tips_fri_sat := tips_friday + tips_saturday
  let tips_sunday := total_tips - tips_fri_sat
  let customers_sunday := tips_sunday / tips_per_customer
  have h : customers_sunday = 36 := by sorry
  exact h

end wade_customers_sunday_l122_122249


namespace painting_time_l122_122632

noncomputable def bob_rate : ℕ := 120 / 8
noncomputable def alice_rate : ℕ := 150 / 10
noncomputable def combined_rate : ℕ := bob_rate + alice_rate
noncomputable def total_area : ℕ := 120 + 150
noncomputable def working_time : ℕ := total_area / combined_rate
noncomputable def lunch_break : ℕ := 1
noncomputable def total_time : ℕ := working_time + lunch_break

theorem painting_time : total_time = 10 := by
  -- Proof skipped
  sorry

end painting_time_l122_122632


namespace greatest_product_sum_300_l122_122297

theorem greatest_product_sum_300 : ∃ x y : ℤ, x + y = 300 ∧ x * y = 22500 :=
by
  use 150, 150
  constructor
  · exact rfl
  · exact rfl

end greatest_product_sum_300_l122_122297


namespace bananas_to_oranges_cost_l122_122932

noncomputable def cost_equivalence (bananas apples oranges : ℕ) : Prop :=
  (5 * bananas = 3 * apples) ∧
  (8 * apples = 5 * oranges)

theorem bananas_to_oranges_cost (bananas apples oranges : ℕ) 
  (h : cost_equivalence bananas apples oranges) :
  oranges = 9 :=
by sorry

end bananas_to_oranges_cost_l122_122932


namespace reema_loan_period_l122_122063

theorem reema_loan_period (P SI : ℕ) (R : ℚ) (h1 : P = 1200) (h2 : SI = 432) (h3 : R = 6) : 
  ∃ T : ℕ, SI = (P * R * T) / 100 ∧ T = 6 :=
by
  sorry

end reema_loan_period_l122_122063


namespace circumference_of_jack_head_l122_122471

theorem circumference_of_jack_head (J C : ℝ) (h1 : (2 / 3) * C = 10) (h2 : (1 / 2) * J + 9 = 15) :
  J = 12 :=
by
  sorry

end circumference_of_jack_head_l122_122471


namespace opposite_of_3_is_neg3_l122_122569

theorem opposite_of_3_is_neg3 : forall (n : ℤ), n = 3 -> -n = -3 :=
by
  sorry

end opposite_of_3_is_neg3_l122_122569


namespace Bruno_wants_2_5_dozens_l122_122547

theorem Bruno_wants_2_5_dozens (total_pens : ℕ) (dozen_pens : ℕ) (h_total_pens : total_pens = 30) (h_dozen_pens : dozen_pens = 12) : (total_pens / dozen_pens : ℚ) = 2.5 :=
by 
  sorry

end Bruno_wants_2_5_dozens_l122_122547


namespace radius_range_of_circle_l122_122710

theorem radius_range_of_circle (r : ℝ) :
  (∃ (x y : ℝ), (x - 3)^2 + (y + 5)^2 = r^2 ∧ 
    (∃ a b : ℝ, 4*a - 3*b - 2 = 0 ∧ ∃ c d : ℝ, 4*c - 3*d - 2 = 0 ∧ 
      (a - x)^2 + (b - y)^2 = 1 ∧ (c - x)^2 + (d - y)^2 = 1 ∧
       a ≠ c ∧ b ≠ d)) ↔ 4 < r ∧ r < 6 :=
by
  sorry

end radius_range_of_circle_l122_122710


namespace son_time_to_complete_job_l122_122457

theorem son_time_to_complete_job (M S : ℝ) (hM : M = 1 / 5) (hMS : M + S = 1 / 4) : S = 1 / 20 → 1 / S = 20 :=
by
  sorry

end son_time_to_complete_job_l122_122457


namespace two_students_cover_all_questions_l122_122781

-- Define the main properties
variables (students : Finset ℕ) (questions : Finset ℕ)
variable (solves : ℕ → ℕ → Prop)

-- Assume the given conditions
axiom total_students : students.card = 8
axiom total_questions : questions.card = 8
axiom each_question_solved_by_min_5_students : ∀ q, q ∈ questions → 
(∃ student_set : Finset ℕ, student_set.card ≥ 5 ∧ ∀ s ∈ student_set, solves s q)

-- The theorem to be proven
theorem two_students_cover_all_questions :
  ∃ s1 s2 : ℕ, s1 ∈ students ∧ s2 ∈ students ∧ s1 ≠ s2 ∧ 
  ∀ q ∈ questions, solves s1 q ∨ solves s2 q :=
sorry -- proof to be written

end two_students_cover_all_questions_l122_122781


namespace mittens_pairing_possible_l122_122661

/--
In a kindergarten's lost and found basket, there are 30 mittens: 
10 blue, 10 green, 10 red, 15 right-hand, and 15 left-hand. 

Prove that it is always possible to create matching pairs of one right-hand 
and one left-hand mitten of the same color for 5 children.
-/
theorem mittens_pairing_possible : 
  (∃ (right_blue left_blue right_green left_green right_red left_red : ℕ), 
    right_blue + left_blue + right_green + left_green + right_red + left_red = 30 ∧
    right_blue ≤ 10 ∧ left_blue ≤ 10 ∧
    right_green ≤ 10 ∧ left_green ≤ 10 ∧
    right_red ≤ 10 ∧ left_red ≤ 10 ∧
    right_blue + right_green + right_red = 15 ∧
    left_blue + left_green + left_red = 15) →
  (∃ right_blue left_blue right_green left_green right_red left_red,
    min right_blue left_blue + 
    min right_green left_green + 
    min right_red left_red ≥ 5) :=
sorry

end mittens_pairing_possible_l122_122661


namespace multiplication_result_l122_122161

theorem multiplication_result :
  121 * 54 = 6534 := by
  sorry

end multiplication_result_l122_122161


namespace Tabitha_age_proof_l122_122989

variable (Tabitha_age current_hair_colors: ℕ)
variable (Adds_new_color_per_year: ℕ)
variable (initial_hair_colors: ℕ)
variable (years_passed: ℕ)

theorem Tabitha_age_proof (h1: Adds_new_color_per_year = 1)
                          (h2: initial_hair_colors = 2)
                          (h3: ∀ years_passed, Tabitha_age  = 15 + years_passed)
                          (h4: Adds_new_color_per_year  = 1 )
                          (h5: current_hair_colors =  8 - 3)
                          (h6: current_hair_colors  =  initial_hair_colors + 3)
                          : Tabitha_age = 18 := 
by {
  sorry  -- Proof omitted
}

end Tabitha_age_proof_l122_122989


namespace common_difference_minimum_sum_value_l122_122523

variable {α : Type}
variables (a : ℕ → ℤ) (d : ℤ)
variables (S : ℕ → ℚ)

-- Conditions: Arithmetic sequence property and specific initial values
def is_arithmetic_sequence (d : ℤ) : Prop :=
  ∀ n, a n = a 1 + (n - 1) * d

axiom a1_eq_neg3 : a 1 = -3
axiom condition : 11 * a 5 = 5 * a 8 - 13

-- Define the sum of the first n terms of the arithmetic sequence
def sum_of_arithmetic_sequence (n : ℕ) (a : ℕ → ℤ) (d : ℤ) : ℚ :=
  (↑n / 2) * (2 * a 1 + ↑((n - 1) * d))

-- Prove the common difference and the minimum sum value
theorem common_difference : d = 31 / 9 :=
sorry

theorem minimum_sum_value : S 1 = -2401 / 840 :=
sorry

end common_difference_minimum_sum_value_l122_122523


namespace average_speed_of_car_l122_122969

theorem average_speed_of_car 
  (speed_first_hour : ℕ)
  (speed_second_hour : ℕ)
  (total_time : ℕ)
  (h1 : speed_first_hour = 90)
  (h2 : speed_second_hour = 40)
  (h3 : total_time = 2) : 
  (speed_first_hour + speed_second_hour) / total_time = 65 := 
by
  sorry

end average_speed_of_car_l122_122969


namespace evenFunctionExists_l122_122027

-- Definitions based on conditions
def isEvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

def passesThroughPoints (f : ℝ → ℝ) (points : List (ℝ × ℝ)) : Prop :=
  ∀ p ∈ points, f p.1 = p.2

-- Example function
def exampleEvenFunction (x : ℝ) : ℝ := x^2 * (x - 3) * (x + 1)

-- Points to pass through
def givenPoints : List (ℝ × ℝ) := [(-1, 0), (0.5, 2.5), (3, 0)]

-- Theorem to be proven
theorem evenFunctionExists : 
  isEvenFunction exampleEvenFunction ∧ passesThroughPoints exampleEvenFunction givenPoints :=
by
  sorry

end evenFunctionExists_l122_122027


namespace exists_almost_square_divides_2010_l122_122971

noncomputable def almost_square (a b : ℕ) : Prop :=
  (a = b + 1 ∨ b = a + 1) ∧ a * b = 2010

theorem exists_almost_square_divides_2010 :
  ∃ (a b : ℕ), almost_square a b :=
sorry

end exists_almost_square_divides_2010_l122_122971


namespace difference_in_distances_l122_122955

noncomputable def circumference (r : ℝ) : ℝ := 2 * Real.pi * r

noncomputable def distance_covered (r : ℝ) (revolutions : ℕ) : ℝ :=
  circumference r * revolutions

theorem difference_in_distances :
  let r1 := 22.4
  let r2 := 34.2
  let revolutions := 400
  let D1 := distance_covered r1 revolutions
  let D2 := distance_covered r2 revolutions
  D2 - D1 = 29628 :=
by
  sorry

end difference_in_distances_l122_122955


namespace wall_width_is_correct_l122_122482

-- Definitions based on the conditions
def brick_length : ℝ := 25  -- in cm
def brick_height : ℝ := 11.25  -- in cm
def brick_width : ℝ := 6  -- in cm
def num_bricks : ℝ := 5600
def wall_length : ℝ := 700  -- 7 m in cm
def wall_height : ℝ := 600  -- 6 m in cm
def total_volume : ℝ := num_bricks * (brick_length * brick_height * brick_width)

-- Prove that the inferred width of the wall is correct
theorem wall_width_is_correct : (total_volume / (wall_length * wall_height)) = 22.5 := by
  sorry

end wall_width_is_correct_l122_122482


namespace simplify_sqrt_l122_122694

theorem simplify_sqrt (x : ℝ) (h : x = (Real.sqrt 3) + 1) : Real.sqrt (x^2) = Real.sqrt 3 + 1 :=
by
  -- This will serve as the placeholder for the proof.
  sorry

end simplify_sqrt_l122_122694


namespace pythagorean_triple_third_number_l122_122025

theorem pythagorean_triple_third_number (x : ℕ) (h1 : x^2 + 8^2 = 17^2) : x = 15 :=
sorry

end pythagorean_triple_third_number_l122_122025


namespace sum_x1_x2_eq_five_l122_122235

theorem sum_x1_x2_eq_five {x1 x2 : ℝ} 
  (h1 : 2^x1 = 5 - x1)
  (h2 : x2 + Real.log x2 / Real.log 2 = 5) : 
  x1 + x2 = 5 := 
sorry

end sum_x1_x2_eq_five_l122_122235


namespace semicircle_inequality_l122_122420

open Real

theorem semicircle_inequality {A B C D E : ℝ} (h : A^2 + B^2 + C^2 + D^2 + E^2 = 1):
  (A - B)^2 + (B - C)^2 + (C - D)^2 + (D - E)^2 + (A - B) * (B - C) * (C - D) + (B - C) * (C - D) * (D - E) < 4 :=
by
  -- proof omitted
  sorry

end semicircle_inequality_l122_122420


namespace florida_north_dakota_license_plate_difference_l122_122094

theorem florida_north_dakota_license_plate_difference :
  let florida_license_plates := 26^3 * 10^3
  let north_dakota_license_plates := 26^3 * 10^3
  florida_license_plates = north_dakota_license_plates :=
by
  let florida_license_plates := 26^3 * 10^3
  let north_dakota_license_plates := 26^3 * 10^3
  show florida_license_plates = north_dakota_license_plates
  sorry

end florida_north_dakota_license_plate_difference_l122_122094


namespace cousins_arrangement_l122_122563

def number_of_arrangements (cousins rooms : ℕ) (min_empty_rooms : ℕ) : ℕ := sorry

theorem cousins_arrangement : number_of_arrangements 5 4 1 = 56 := 
by sorry

end cousins_arrangement_l122_122563


namespace sufficient_not_necessary_l122_122834

theorem sufficient_not_necessary (p q: Prop) :
  ¬ (p ∨ q) → ¬ p ∧ (¬ p → ¬(¬ p ∧ ¬ q)) := sorry

end sufficient_not_necessary_l122_122834


namespace metallic_sheet_length_l122_122579

theorem metallic_sheet_length (w : ℝ) (s : ℝ) (v : ℝ) (L : ℝ) 
  (h_w : w = 38) 
  (h_s : s = 8) 
  (h_v : v = 5632) 
  (h_volume : (L - 2 * s) * (w - 2 * s) * s = v) : 
  L = 48 :=
by
  -- To complete the proof, follow the mathematical steps:
  -- (L - 2 * s) * (w - 2 * s) * s = v
  -- (L - 2 * 8) * (38 - 2 * 8) * 8 = 5632
  -- Simplify and solve for L
  sorry

end metallic_sheet_length_l122_122579


namespace find_constant_term_l122_122141

-- Definitions based on conditions:
def sum_of_coeffs (n : ℕ) : ℕ := 4 ^ n
def sum_of_binom_coeffs (n : ℕ) : ℕ := 2 ^ n
def P_plus_Q_equals (n : ℕ) : Prop := sum_of_coeffs n + sum_of_binom_coeffs n = 272

-- Constant term in the binomial expansion:
def constant_term (n r : ℕ) : ℕ := Nat.choose n r * (3 ^ (n - r))

-- The proof statement
theorem find_constant_term : 
  ∃ n r : ℕ, P_plus_Q_equals n ∧ n = 4 ∧ r = 1 ∧ constant_term n r = 108 :=
by {
  sorry
}

end find_constant_term_l122_122141


namespace sufficient_but_not_necessary_l122_122641

theorem sufficient_but_not_necessary (x : ℝ) :
  (x > 3 → x^2 > 4) ∧ ¬(x^2 > 4 → x > 3) :=
by sorry

end sufficient_but_not_necessary_l122_122641


namespace radical_axis_of_non_concentric_circles_l122_122296

theorem radical_axis_of_non_concentric_circles 
  {a R1 R2 : ℝ} (a_pos : a ≠ 0) (R1_pos : R1 > 0) (R2_pos : R2 > 0) :
  ∃ (x : ℝ), ∀ (y : ℝ), 
  ((x + a)^2 + y^2 - R1^2 = (x - a)^2 + y^2 - R2^2) ↔ x = (R2^2 - R1^2) / (4 * a) :=
by sorry

end radical_axis_of_non_concentric_circles_l122_122296


namespace simplify_polynomial_l122_122034

variable (p : ℝ)

theorem simplify_polynomial :
  (7 * p ^ 5 - 4 * p ^ 3 + 8 * p ^ 2 - 5 * p + 3) + (- p ^ 5 + 3 * p ^ 3 - 7 * p ^ 2 + 6 * p + 2) =
  6 * p ^ 5 - p ^ 3 + p ^ 2 + p + 5 :=
by
  sorry

end simplify_polynomial_l122_122034


namespace part1_sales_increase_part2_price_reduction_l122_122382

-- Part 1: If the price is reduced by 4 yuan, the new average daily sales will be 28 items.
theorem part1_sales_increase (initial_sales : ℕ) (increase_per_yuan : ℕ) (reduction : ℕ) :
  initial_sales = 20 → increase_per_yuan = 2 → reduction = 4 →
  initial_sales + increase_per_yuan * reduction = 28 :=
by sorry

-- Part 2: By how much should the price of each item be reduced for a daily profit of 1050 yuan.
theorem part2_price_reduction (initial_sales : ℕ) (increase_per_yuan : ℕ) (initial_profit : ℕ) 
  (target_profit : ℕ) (min_profit_per_item : ℕ) (x : ℕ) :
  initial_sales = 20 → increase_per_yuan = 2 → initial_profit = 40 → target_profit = 1050 
  → min_profit_per_item = 25 → (40 - x) * (20 + 2 * x) = 1050 → (40 - x) ≥ 25 → x = 5 :=
by sorry

end part1_sales_increase_part2_price_reduction_l122_122382


namespace gold_bars_per_row_l122_122784

theorem gold_bars_per_row 
  (total_worth : ℝ)
  (total_rows : ℕ)
  (value_per_bar : ℝ)
  (h_total_worth : total_worth = 1600000)
  (h_total_rows : total_rows = 4)
  (h_value_per_bar : value_per_bar = 40000) :
  total_worth / value_per_bar / total_rows = 10 :=
by
  sorry

end gold_bars_per_row_l122_122784


namespace abs_diff_squares_l122_122077

theorem abs_diff_squares (a b : ℝ) (ha : a = 105) (hb : b = 103) : |a^2 - b^2| = 416 :=
by
  sorry

end abs_diff_squares_l122_122077


namespace find_t_l122_122125

theorem find_t (t : ℝ) (h : (1 / (t+3) + 3 * t / (t+3) - 4 / (t+3)) = 5) : t = -9 :=
by
  sorry

end find_t_l122_122125


namespace rectangle_area_increase_l122_122717

theorem rectangle_area_increase (a b : ℝ) :
  let new_length := (1 + 1/4) * a
  let new_width := (1 + 1/5) * b
  let original_area := a * b
  let new_area := new_length * new_width
  let area_increase := new_area - original_area
  (area_increase / original_area) = 1/2 := 
by
  sorry

end rectangle_area_increase_l122_122717


namespace election_candidate_a_votes_l122_122409

theorem election_candidate_a_votes :
  let total_votes : ℕ := 560000
  let invalid_percentage : ℚ := 15 / 100
  let candidate_a_percentage : ℚ := 70 / 100
  let total_valid_votes := total_votes * (1 - invalid_percentage)
  let candidate_a_votes := total_valid_votes * candidate_a_percentage
  candidate_a_votes = 333200 :=
by
  let total_votes : ℕ := 560000
  let invalid_percentage : ℚ := 15 / 100
  let candidate_a_percentage : ℚ := 70 / 100
  let total_valid_votes := total_votes * (1 - invalid_percentage)
  let candidate_a_votes := total_valid_votes * candidate_a_percentage
  show candidate_a_votes = 333200
  sorry

end election_candidate_a_votes_l122_122409


namespace solve_problem_l122_122807

open Classical

-- Definition of the problem conditions
def problem_conditions (x y : ℝ) : Prop :=
  5 * y^2 + 3 * y + 2 = 2 * (10 * x^2 + 3 * y + 3) ∧ y = 3 * x + 1

-- Definition of the quadratic solution considering the quadratic formula
def quadratic_solution (x : ℝ) : Prop :=
  x = (-21 + Real.sqrt 641) / 50 ∨ x = (-21 - Real.sqrt 641) / 50

-- Main theorem statement
theorem solve_problem :
  ∃ x y : ℝ, problem_conditions x y ∧ quadratic_solution x :=
by
  sorry

end solve_problem_l122_122807


namespace equivalent_problem_l122_122801

variable (a b : ℤ)

def condition1 : Prop :=
  a * (-2)^3 + b * (-2) - 7 = 9

def condition2 : Prop :=
  8 * a + 2 * b - 7 = -23

theorem equivalent_problem (h : condition1 a b) : condition2 a b :=
sorry

end equivalent_problem_l122_122801


namespace loaves_of_bread_can_bake_l122_122106

def total_flour_in_cupboard := 200
def total_flour_on_counter := 100
def total_flour_in_pantry := 100
def flour_per_loaf := 200

theorem loaves_of_bread_can_bake :
  (total_flour_in_cupboard + total_flour_on_counter + total_flour_in_pantry) / flour_per_loaf = 2 := by
  sorry

end loaves_of_bread_can_bake_l122_122106


namespace inequality_proof_l122_122678

theorem inequality_proof (x y : ℝ) (n : ℕ) (hx : 0 < x) (hy : 0 < y) (hn : 0 < n):
  x^n / (1 + x^2) + y^n / (1 + y^2) ≤ (x^n + y^n) / (1 + x * y) :=
by
  sorry

end inequality_proof_l122_122678


namespace labourer_saving_after_debt_clearance_l122_122894

variable (averageExpenditureFirst6Months : ℕ)
variable (monthlyIncome : ℕ)
variable (reducedMonthlyExpensesNext4Months : ℕ)

theorem labourer_saving_after_debt_clearance (h1 : averageExpenditureFirst6Months = 90)
                                              (h2 : monthlyIncome = 81)
                                              (h3 : reducedMonthlyExpensesNext4Months = 60) :
    (monthlyIncome * 4) - ((reducedMonthlyExpensesNext4Months * 4) + 
    ((averageExpenditureFirst6Months * 6) - (monthlyIncome * 6))) = 30 := by
  sorry

end labourer_saving_after_debt_clearance_l122_122894


namespace probability_two_white_balls_is_4_over_15_l122_122994

-- Define the conditions of the problem
def total_balls : ℕ := 15
def white_balls_initial : ℕ := 8
def black_balls : ℕ := 7
def balls_drawn : ℕ := 2 -- Note: Even though not explicitly required, it's part of the context

-- Calculate the probability of drawing two white balls without replacement
noncomputable def probability_two_white_balls : ℚ :=
  (white_balls_initial / total_balls) * ((white_balls_initial - 1) / (total_balls - 1))

-- The theorem to prove
theorem probability_two_white_balls_is_4_over_15 :
  probability_two_white_balls = 4 / 15 := by
  sorry

end probability_two_white_balls_is_4_over_15_l122_122994


namespace part1_part2_min_part2_max_part3_l122_122370

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x - 2 * a / x - 3 * Real.log x

noncomputable def f' (a : ℝ) (x : ℝ) : ℝ := a + 2 * a / (x^2) - 3 / x

theorem part1 (a : ℝ) : f' a 1 = 0 -> a = 1 := sorry

noncomputable def f1 (x : ℝ) : ℝ := x - 2 / x - 3 * Real.log x

noncomputable def f1' (x : ℝ) : ℝ := 1 + 2 / (x^2) - 3 / x

theorem part2_min (h_a : 1 = 1) : 
    ∀ (x : ℝ), (1 ≤ x) ∧ (x ≤ Real.exp 1) -> 
    (f1 2 <= f1 x) := sorry

theorem part2_max (h_a : 1 = 1) : 
    ∀ (x : ℝ), (1 ≤ x) ∧ (x ≤ Real.exp 1) ->
    (f1 x <= f1 1) := sorry

theorem part3 (a : ℝ) : 
    (∀ (x : ℝ), x > 0 -> f' a x ≥ 0) -> a ≥ (3 * Real.sqrt 2) / 4 := sorry

end part1_part2_min_part2_max_part3_l122_122370


namespace chinese_character_symmetry_l122_122000

-- Definitions of the characters and their symmetry properties
def is_symmetric (ch : String) : Prop :=
  ch = "喜"

-- Hypotheses (conditions)
def option_A := "喜"
def option_B := "欢"
def option_C := "数"
def option_D := "学"

-- Lean statement to prove the symmetry
theorem chinese_character_symmetry :
  is_symmetric option_A ∧ 
  ¬ is_symmetric option_B ∧ 
  ¬ is_symmetric option_C ∧ 
  ¬ is_symmetric option_D :=
by
  sorry

end chinese_character_symmetry_l122_122000


namespace largest_solution_of_equation_l122_122275

theorem largest_solution_of_equation :
  let eq := λ x : ℝ => x^4 - 50 * x^2 + 625
  ∃ x : ℝ, eq x = 0 ∧ ∀ y : ℝ, eq y = 0 → y ≤ x :=
sorry

end largest_solution_of_equation_l122_122275


namespace value_of_expression_l122_122788

theorem value_of_expression (b : ℚ) (h : b = 1/3) : (3 * b⁻¹ + (b⁻¹ / 3)) / b = 30 :=
by
  rw [h]
  sorry

end value_of_expression_l122_122788


namespace point_P_in_fourth_quadrant_l122_122478

def point_in_fourth_quadrant (x y : ℝ) : Prop :=
  x > 0 ∧ y < 0

theorem point_P_in_fourth_quadrant (m : ℝ) : point_in_fourth_quadrant (1 + m^2) (-1) :=
by
  sorry

end point_P_in_fourth_quadrant_l122_122478


namespace shorter_piece_length_l122_122511

-- Definitions according to conditions in a)
variables (x : ℝ) (total_length : ℝ := 140)
variables (ratio : ℝ := 5 / 2)

-- Statement to be proved
theorem shorter_piece_length : x + ratio * x = total_length → x = 40 := 
by
  intros h
  sorry

end shorter_piece_length_l122_122511


namespace find_C_coordinates_l122_122169

variables {A B M L C : ℝ × ℝ}

def is_midpoint (M A B : ℝ × ℝ) : Prop :=
  M.1 = (A.1 + B.1) / 2 ∧ M.2 = (A.2 + B.2) / 2

def on_line_bisector (L B : ℝ × ℝ) : Prop :=
  B.1 = 6  -- Vertical line through B

theorem find_C_coordinates
  (A := (2, 8))
  (M := (4, 11))
  (L := (6, 6))
  (hM : is_midpoint M A B)
  (hL : on_line_bisector L B) :
  C = (6, 14) :=
sorry

end find_C_coordinates_l122_122169


namespace max_triangle_side_l122_122566

-- Definitions of conditions
def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def has_perimeter (a b c : ℕ) (p : ℕ) : Prop :=
  a + b + c = p

def different_integers (a b c : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c

-- The main theorem to prove
theorem max_triangle_side (a b c : ℕ) (h_triangle : is_triangle a b c)
                         (h_perimeter : has_perimeter a b c 24)
                         (h_diff : different_integers a b c) :
  c ≤ 11 :=
sorry

end max_triangle_side_l122_122566


namespace base_case_of_interior_angle_sum_l122_122204

-- Definitions consistent with conditions: A convex polygon with at least n sides where n >= 3.
def convex_polygon (n : ℕ) : Prop := n ≥ 3

-- Proposition: If w the sum of angles for convex polygons, we start checking from n = 3.
theorem base_case_of_interior_angle_sum (n : ℕ) (h : convex_polygon n) :
  n = 3 := 
by
  sorry

end base_case_of_interior_angle_sum_l122_122204


namespace vertex_angle_of_obtuse_isosceles_triangle_l122_122587

noncomputable def isosceles_obtuse_triangle (a b h : ℝ) (φ : ℝ) : Prop :=
  a^2 = 2 * b * h ∧
  b = 2 * a * Real.cos ((180 - φ) / 2) ∧
  h = a * Real.sin ((180 - φ) / 2) ∧
  90 < φ ∧ φ < 180

theorem vertex_angle_of_obtuse_isosceles_triangle (a b h : ℝ) (φ : ℝ) :
  isosceles_obtuse_triangle a b h φ → φ = 150 :=
by
  sorry

end vertex_angle_of_obtuse_isosceles_triangle_l122_122587


namespace total_carrot_sticks_l122_122129

-- Define the number of carrot sticks James ate before and after dinner
def carrot_sticks_before_dinner : Nat := 22
def carrot_sticks_after_dinner : Nat := 15

-- Prove that the total number of carrot sticks James ate is 37
theorem total_carrot_sticks : carrot_sticks_before_dinner + carrot_sticks_after_dinner = 37 :=
  by sorry

end total_carrot_sticks_l122_122129


namespace solution_set_of_inequality_l122_122081

theorem solution_set_of_inequality :
  { x : ℝ | x * (x - 1) ≤ 0 } = { x : ℝ | 0 ≤ x ∧ x ≤ 1 } :=
by
sorry

end solution_set_of_inequality_l122_122081


namespace combined_length_of_straight_parts_l122_122461

noncomputable def length_of_straight_parts (R : ℝ) (p : ℝ) : ℝ := p * R

theorem combined_length_of_straight_parts :
  ∀ (R : ℝ) (p : ℝ), R = 80 ∧ p = 0.25 → length_of_straight_parts R p = 20 :=
by
  intros R p h
  cases' h with hR hp
  rw [hR, hp]
  simp [length_of_straight_parts]
  sorry

end combined_length_of_straight_parts_l122_122461


namespace statement_II_must_be_true_l122_122677

-- Define the set of all creatures
variable (Creature : Type)

-- Define properties for being a dragon, mystical, and fire-breathing
variable (Dragon Mystical FireBreathing : Creature → Prop)

-- Given conditions
-- All dragons breathe fire
axiom all_dragons_breathe_fire : ∀ c, Dragon c → FireBreathing c
-- Some mystical creatures are dragons
axiom some_mystical_creatures_are_dragons : ∃ c, Mystical c ∧ Dragon c

-- Questions to prove (we will only formalize the must be true statement)
-- Statement II: Some fire-breathing creatures are mystical creatures

theorem statement_II_must_be_true : ∃ c, FireBreathing c ∧ Mystical c :=
by
  sorry

end statement_II_must_be_true_l122_122677


namespace fish_caught_together_l122_122012

theorem fish_caught_together (Blaines_fish Keiths_fish : ℕ) 
  (h1 : Blaines_fish = 5) 
  (h2 : Keiths_fish = 2 * Blaines_fish) : 
  Blaines_fish + Keiths_fish = 15 := 
by 
  sorry

end fish_caught_together_l122_122012


namespace total_fencing_cost_l122_122396

theorem total_fencing_cost
  (park_is_square : true)
  (cost_per_side : ℕ)
  (h1 : cost_per_side = 43) :
  4 * cost_per_side = 172 :=
by
  sorry

end total_fencing_cost_l122_122396


namespace union_M_N_l122_122015

def M := {x : ℝ | -2 < x ∧ x < -1}
def N := {x : ℝ | (1 / 2 : ℝ)^x ≤ 4}

theorem union_M_N :
  M ∪ N = {x : ℝ | x ≥ -2} :=
sorry

end union_M_N_l122_122015


namespace tickets_total_l122_122289

theorem tickets_total (T : ℝ) (h1 : T / 2 + (T / 2) / 4 = 3600) : T = 5760 :=
by
  sorry

end tickets_total_l122_122289


namespace win_percentage_of_people_with_envelopes_l122_122849

theorem win_percentage_of_people_with_envelopes (total_people : ℕ) (percent_with_envelopes : ℝ) (winners : ℕ) (num_with_envelopes : ℕ) : 
  total_people = 100 ∧ percent_with_envelopes = 0.40 ∧ num_with_envelopes = total_people * percent_with_envelopes ∧ winners = 8 → 
    (winners / num_with_envelopes) * 100 = 20 :=
by
  intros
  sorry

end win_percentage_of_people_with_envelopes_l122_122849


namespace freds_total_marbles_l122_122691

theorem freds_total_marbles :
  let red := 38
  let green := red / 2
  let dark_blue := 6
  red + green + dark_blue = 63 := by
  sorry

end freds_total_marbles_l122_122691


namespace product_of_primes_l122_122607

theorem product_of_primes :
  (7 * 97 * 89) = 60431 :=
by
  sorry

end product_of_primes_l122_122607


namespace total_employees_in_buses_l122_122288

-- Define the capacity of each bus
def capacity : ℕ := 150

-- Define the fill percentages of each bus
def fill_percentage_bus1 : ℚ := 60 / 100
def fill_percentage_bus2 : ℚ := 70 / 100

-- Calculate the number of passengers in each bus
def passengers_bus1 : ℚ := fill_percentage_bus1 * capacity
def passengers_bus2 : ℚ := fill_percentage_bus2 * capacity

-- Calculate the total number of passengers
def total_passengers : ℚ := passengers_bus1 + passengers_bus2

-- The proof statement
theorem total_employees_in_buses : total_passengers = 195 :=
by
  sorry

end total_employees_in_buses_l122_122288


namespace evaluate_expression_l122_122810

theorem evaluate_expression :
  let a := 5 ^ 1001
  let b := 6 ^ 1002
  (a + b) ^ 2 - (a - b) ^ 2 = 24 * 30 ^ 1001 :=
by
  sorry

end evaluate_expression_l122_122810


namespace f_bounds_l122_122727

-- Define the function f with the given properties
def f : ℝ → ℝ :=
sorry 

-- Specify the conditions on f
axiom f_0 : f 0 = 0
axiom f_1 : f 1 = 1
axiom f_ratio (x y z : ℝ) (h1 : 0 ≤ x) (h2 : x < y) (h3 : y < z) (h4 : z ≤ 1) 
  (h5 : z - y = y - x) : 1/2 ≤ (f z - f y) / (f y - f x) ∧ (f z - f y) / (f y - f x) ≤ 2

-- State the theorem to be proven
theorem f_bounds : 1 / 7 ≤ f (1 / 3) ∧ f (1 / 3) ≤ 4 / 7 :=
sorry

end f_bounds_l122_122727


namespace find_y_l122_122458

open Classical

theorem find_y (a b c x y : ℚ)
  (h1 : a / b = 5 / 4)
  (h2 : b / c = 3 / x)
  (h3 : a / c = y / 4) :
  y = 15 / x :=
sorry

end find_y_l122_122458


namespace sufficient_but_not_necessary_l122_122528

theorem sufficient_but_not_necessary (a b : ℝ) : 
  (a > |b|) → (a^3 > b^3) ∧ ¬((a^3 > b^3) → (a > |b|)) :=
by
  sorry

end sufficient_but_not_necessary_l122_122528


namespace find_a_for_square_binomial_l122_122260

theorem find_a_for_square_binomial (a : ℚ) (h: ∃ (b : ℚ), ∀ (x : ℚ), 9 * x^2 + 21 * x + a = (3 * x + b)^2) : a = 49 / 4 := 
by 
  sorry

end find_a_for_square_binomial_l122_122260


namespace greatest_abs_solution_l122_122213

theorem greatest_abs_solution :
  (∃ x : ℝ, x^2 + 18 * x + 81 = 0 ∧ ∀ y : ℝ, y^2 + 18 * y + 81 = 0 → |x| ≥ |y| ∧ |x| = 9) :=
sorry

end greatest_abs_solution_l122_122213


namespace max_students_can_distribute_equally_l122_122265

-- Define the given numbers of pens and pencils
def pens : ℕ := 1001
def pencils : ℕ := 910

-- State the problem in Lean 4 as a theorem
theorem max_students_can_distribute_equally :
  Nat.gcd pens pencils = 91 :=
sorry

end max_students_can_distribute_equally_l122_122265


namespace sea_creatures_lost_l122_122961

theorem sea_creatures_lost (sea_stars : ℕ) (seashells : ℕ) (snails : ℕ) (items_left : ℕ)
  (h1 : sea_stars = 34) (h2 : seashells = 21) (h3 : snails = 29) (h4 : items_left = 59) :
  sea_stars + seashells + snails - items_left = 25 :=
by
  sorry

end sea_creatures_lost_l122_122961


namespace jonas_socks_solution_l122_122149

theorem jonas_socks_solution (p_s p_h n_p n_t n : ℕ) (h_ps : p_s = 20) (h_ph : p_h = 5) (h_np : n_p = 10) (h_nt : n_t = 10) :
  2 * (p_s * 2 + p_h * 2 + n_p + n_t) = 2 * (p_s * 2 + p_h * 2 + n_p + n_t + n * 2) :=
by
  -- skipping the proof part
  sorry

end jonas_socks_solution_l122_122149


namespace vec_c_is_linear_comb_of_a_b_l122_122562

structure Vec2 :=
  (x : ℝ)
  (y : ℝ)

def a := Vec2.mk 1 2
def b := Vec2.mk (-2) 3
def c := Vec2.mk 4 1

theorem vec_c_is_linear_comb_of_a_b : c = Vec2.mk (2 * a.x - b.x) (2 * a.y - b.y) :=
  by
    sorry

end vec_c_is_linear_comb_of_a_b_l122_122562


namespace trigonometric_inequality_equality_conditions_l122_122327

theorem trigonometric_inequality
  (α β : ℝ)
  (hα : 0 < α ∧ α < π / 2)
  (hβ : 0 < β ∧ β < π / 2) :
  (1 / (Real.cos α)^2 + 1 / ((Real.sin α)^2 * (Real.sin β)^2 * (Real.cos β)^2)) ≥ 9 :=
sorry

theorem equality_conditions
  (α β : ℝ)
  (hα : α = Real.arctan (Real.sqrt 2))
  (hβ : β = π / 4) :
  (1 / (Real.cos α)^2 + 1 / ((Real.sin α)^2 * (Real.sin β)^2 * (Real.cos β)^2)) = 9 :=
sorry

end trigonometric_inequality_equality_conditions_l122_122327


namespace triangle_area_l122_122506

open Real

def line1 (x y : ℝ) : Prop := y = 6
def line2 (x y : ℝ) : Prop := y = 2 + x
def line3 (x y : ℝ) : Prop := y = 2 - x

def is_vertex (x y : ℝ) (l1 l2 : ℝ → ℝ → Prop) : Prop := l1 x y ∧ l2 x y

def vertices (v1 v2 v3 : ℝ × ℝ) : Prop :=
  is_vertex v1.1 v1.2 line1 line2 ∧
  is_vertex v2.1 v2.2 line1 line3 ∧
  is_vertex v3.1 v3.2 line2 line3

def area_triangle (v1 v2 v3 : ℝ × ℝ) : ℝ :=
  0.5 * abs ((v1.1 * v2.2 + v2.1 * v3.2 + v3.1 * v1.2) -
             (v2.1 * v1.2 + v3.1 * v2.2 + v1.1 * v3.2))

theorem triangle_area : vertices (4, 6) (-4, 6) (0, 2) → area_triangle (4, 6) (-4, 6) (0, 2) = 8 :=
by
  sorry

end triangle_area_l122_122506


namespace bill_new_win_percentage_l122_122993

theorem bill_new_win_percentage :
  ∀ (initial_games : ℕ) (initial_win_percentage : ℚ) (additional_games : ℕ) (losses_in_additional_games : ℕ),
  initial_games = 200 →
  initial_win_percentage = 0.63 →
  additional_games = 100 →
  losses_in_additional_games = 43 →
  ((initial_win_percentage * initial_games + (additional_games - losses_in_additional_games)) / (initial_games + additional_games)) * 100 = 61 := 
by
  intros initial_games initial_win_percentage additional_games losses_in_additional_games h1 h2 h3 h4
  sorry

end bill_new_win_percentage_l122_122993


namespace find_b_if_lines_parallel_l122_122520

-- Definitions of the line equations and parallel condition
def first_line (x y : ℝ) (b : ℝ) : Prop := 3 * y - b = -9 * x + 1
def second_line (x y : ℝ) (b : ℝ) : Prop := 2 * y + 8 = (b - 3) * x - 2

-- Definition of parallel lines (their slopes are equal)
def parallel_lines (m1 m2 : ℝ) : Prop := m1 = m2

-- Given conditions and the conclusion to prove
theorem find_b_if_lines_parallel :
  ∃ b : ℝ, (∀ x y : ℝ, first_line x y b → ∃ m1 : ℝ, ∀ x y : ℝ, ∃ c : ℝ, y = m1 * x + c) ∧ 
           (∀ x y : ℝ, second_line x y b → ∃ m2 : ℝ, ∀ x y : ℝ, ∃ c : ℝ, y = m2 * x + c) ∧ 
           parallel_lines (-3) ((b - 3) / 2) →
           b = -3 :=
by {
  sorry
}

end find_b_if_lines_parallel_l122_122520


namespace winning_margin_l122_122876

theorem winning_margin (total_votes : ℝ) (winning_votes : ℝ) (winning_percent : ℝ) (losing_percent : ℝ) 
  (win_votes_eq: winning_votes = winning_percent * total_votes)
  (perc_eq: winning_percent + losing_percent = 1)
  (win_votes_given: winning_votes = 550)
  (winning_percent_given: winning_percent = 0.55)
  (losing_percent_given: losing_percent = 0.45) :
  winning_votes - (losing_percent * total_votes) = 100 := 
by
  sorry

end winning_margin_l122_122876


namespace maintain_constant_chromosomes_l122_122088

-- Definitions
def meiosis_reduces_chromosomes (original_chromosomes : ℕ) : ℕ := original_chromosomes / 2

def fertilization_restores_chromosomes (half_chromosomes : ℕ) : ℕ := half_chromosomes * 2

-- The proof problem
theorem maintain_constant_chromosomes (original_chromosomes : ℕ) (somatic_chromosomes : ℕ) :
  meiosis_reduces_chromosomes original_chromosomes = somatic_chromosomes / 2 ∧
  fertilization_restores_chromosomes (meiosis_reduces_chromosomes original_chromosomes) = somatic_chromosomes :=
sorry

end maintain_constant_chromosomes_l122_122088


namespace perfect_square_trinomial_l122_122313

theorem perfect_square_trinomial (m : ℝ) (h : ∃ a : ℝ, x^2 + 2 * x + m = (x + a)^2) : m = 1 := 
sorry

end perfect_square_trinomial_l122_122313


namespace find_plane_equation_l122_122593

def point := ℝ × ℝ × ℝ

def plane_equation (A B C D : ℝ) (x y z : ℝ) : Prop :=
  A * x + B * y + C * z + D = 0

def points := (0, 3, -1) :: (4, 7, 1) :: (2, 5, 0) :: []

def correct_plane_equation : Prop :=
  ∃ A B C D : ℝ, plane_equation A B C D = fun x y z => A * x + B * y + C * z + D = 0 ∧ 
  (A, B, C, D) = (0, 1, -2, -5) ∧ ∀ x y z, (x, y, z) ∈ points → plane_equation A B C D x y z

theorem find_plane_equation : correct_plane_equation :=
sorry

end find_plane_equation_l122_122593


namespace batsman_average_increase_l122_122843

theorem batsman_average_increase
  (A : ℕ)
  (h_average_after_17th : (16 * A + 90) / 17 = 42) :
  42 - A = 3 :=
by
  sorry

end batsman_average_increase_l122_122843


namespace cathy_initial_money_l122_122357

-- Definitions of the conditions
def moneyFromDad : Int := 25
def moneyFromMom : Int := 2 * moneyFromDad
def totalMoneyReceived : Int := moneyFromDad + moneyFromMom
def currentMoney : Int := 87

-- Theorem stating the proof problem
theorem cathy_initial_money (initialMoney : Int) :
  initialMoney + totalMoneyReceived = currentMoney → initialMoney = 12 :=
by
  sorry

end cathy_initial_money_l122_122357


namespace average_percentage_decrease_l122_122729

theorem average_percentage_decrease (x : ℝ) : 60 * (1 - x) * (1 - x) = 48.6 → x = 0.1 :=
by sorry

end average_percentage_decrease_l122_122729


namespace problem1_problem2_l122_122874

-- Definitions for conditions
def p (x : ℝ) : Prop := x^2 - 7 * x + 10 < 0
def q (x : ℝ) (m : ℝ) : Prop := x^2 - 4 * m * x + 3 * m^2 < 0

-- Problem 1: For m = 4, p ∧ q implies 4 < x < 5
theorem problem1 (x : ℝ) (h : 4 < x ∧ x < 5) : 
  p x ∧ q x 4 :=
sorry

-- Problem 2: ∃ m, m > 0, m ≤ 2, and 3m ≥ 5 implies (5/3 ≤ m ≤ 2)
theorem problem2 (m : ℝ) (h1 : m > 0) (h2 : m ≤ 2) (h3 : 3 * m ≥ 5) : 
  5 / 3 ≤ m ∧ m ≤ 2 :=
sorry

end problem1_problem2_l122_122874


namespace closure_of_M_is_closed_interval_l122_122346

noncomputable def U : Set ℝ := Set.univ

noncomputable def M : Set ℝ := {a | a^2 - 2 * a > 0}

theorem closure_of_M_is_closed_interval :
  closure M = {a | 0 ≤ a ∧ a ≤ 2} :=
by
  sorry

end closure_of_M_is_closed_interval_l122_122346


namespace seating_arrangements_l122_122415

-- Definitions for conditions
def num_parents : ℕ := 2
def num_children : ℕ := 3
def num_front_seats : ℕ := 2
def num_back_seats : ℕ := 3
def num_family_members : ℕ := num_parents + num_children

-- The statement we need to prove
theorem seating_arrangements : 
  (num_parents * -- choices for driver
  (num_family_members - 1) * -- choices for the front passenger
  (num_back_seats.factorial)) = 48 := -- arrangements for the back seats
by
  sorry

end seating_arrangements_l122_122415


namespace number_of_elements_in_set_l122_122815

-- We define the conditions in terms of Lean definitions.
variable (n : ℕ) (S : ℕ)

-- Define the initial wrong average condition
def wrong_avg_condition : Prop := (S + 26) / n = 18

-- Define the corrected average condition
def correct_avg_condition : Prop := (S + 36) / n = 19

-- The main theorem to be proved
theorem number_of_elements_in_set (h1 : wrong_avg_condition n S) (h2 : correct_avg_condition n S) : n = 10 := 
sorry

end number_of_elements_in_set_l122_122815


namespace necessary_and_sufficient_condition_l122_122716

theorem necessary_and_sufficient_condition 
  (a : ℕ) 
  (A B : ℝ) 
  (x y z : ℤ) 
  (h1 : (x^2 + y^2 + z^2 : ℝ) = (B * ↑a)^2) 
  (h2 : (x^2 * (A * x^2 + B * y^2) + y^2 * (A * y^2 + B * z^2) + z^2 * (A * z^2 + B * x^2) : ℝ) = (1 / 4) * (2 * A + B) * (B * (↑a)^4)) :
  B = 2 * A :=
by
  sorry

end necessary_and_sufficient_condition_l122_122716


namespace microphotonics_budget_allocation_l122_122712

theorem microphotonics_budget_allocation
    (home_electronics : ℕ)
    (food_additives : ℕ)
    (gen_mod_microorg : ℕ)
    (ind_lubricants : ℕ)
    (basic_astrophysics_degrees : ℕ)
    (full_circle_degrees : ℕ := 360)
    (total_budget_percentage : ℕ := 100)
    (basic_astrophysics_percentage : ℕ) :
  home_electronics = 24 →
  food_additives = 15 →
  gen_mod_microorg = 19 →
  ind_lubricants = 8 →
  basic_astrophysics_degrees = 72 →
  basic_astrophysics_percentage = (basic_astrophysics_degrees * total_budget_percentage) / full_circle_degrees →
  (total_budget_percentage -
    (home_electronics + food_additives + gen_mod_microorg + ind_lubricants + basic_astrophysics_percentage)) = 14 :=
by
  intros he fa gmm il bad bp
  sorry

end microphotonics_budget_allocation_l122_122712


namespace PetyaColorsAll64Cells_l122_122040

-- Assuming a type for representing cell coordinates
structure Cell where
  row : ℕ
  col : ℕ

def isColored (c : Cell) : Prop := true  -- All cells are colored
def LShapedFigures : Set (Set Cell) := sorry  -- Define what constitutes an L-shaped figure

theorem PetyaColorsAll64Cells :
  (∀ tilesVector ∈ LShapedFigures, ¬∀ cell ∈ tilesVector, isColored cell) → (∀ c : Cell, c.row < 8 ∧ c.col < 8 ∧ isColored c) := sorry

end PetyaColorsAll64Cells_l122_122040


namespace sqrt_number_is_169_l122_122687

theorem sqrt_number_is_169 (a b : ℝ) 
  (h : a^2 + b^2 + (4 * a - 6 * b + 13) = 0) : 
  (a^2 + b^2)^2 = 169 :=
sorry

end sqrt_number_is_169_l122_122687


namespace outfits_count_l122_122524

theorem outfits_count (shirts ties : ℕ) (h_shirts : shirts = 7) (h_ties : ties = 6) : 
  (shirts * (ties + 1) = 49) :=
by
  sorry

end outfits_count_l122_122524


namespace coordinates_of_AC_l122_122428

structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def vector_sub (p1 p2 : Point3D) : Point3D :=
  { x := p1.x - p2.x,
    y := p1.y - p2.y,
    z := p1.z - p2.z }

def scalar_mult (k : ℝ) (v : Point3D) : Point3D :=
  { x := k * v.x,
    y := k * v.y,
    z := k * v.z }

noncomputable def A : Point3D := { x := 1, y := 2, z := 3 }
noncomputable def B : Point3D := { x := 4, y := 5, z := 9 }

theorem coordinates_of_AC : vector_sub B A = { x := 3, y := 3, z := 6 } →
  scalar_mult (1 / 3) (vector_sub B A) = { x := 1, y := 1, z := 2 } :=
by
  sorry

end coordinates_of_AC_l122_122428


namespace f_when_x_lt_4_l122_122904

noncomputable def f : ℝ → ℝ := sorry

theorem f_when_x_lt_4 (x : ℝ) (h1 : ∀ y : ℝ, y > 4 → f y = 2^(y-1)) (h2 : ∀ y : ℝ, f (4-y) = f (4+y)) (hx : x < 4) : f x = 2^(7-x) :=
by
  sorry

end f_when_x_lt_4_l122_122904


namespace remainder_mod_17_zero_l122_122057

theorem remainder_mod_17_zero :
  let x1 := 2002 + 3
  let x2 := 2003 + 3
  let x3 := 2004 + 3
  let x4 := 2005 + 3
  let x5 := 2006 + 3
  let x6 := 2007 + 3
  ( (x1 % 17) * (x2 % 17) * (x3 % 17) * (x4 % 17) * (x5 % 17) * (x6 % 17) ) % 17 = 0 :=
by
  let x1 := 2002 + 3
  let x2 := 2003 + 3
  let x3 := 2004 + 3
  let x4 := 2005 + 3
  let x5 := 2006 + 3
  let x6 := 2007 + 3
  sorry

end remainder_mod_17_zero_l122_122057


namespace part_a_l122_122291

theorem part_a (students : Fin 64 → Fin 8 × Fin 8 × Fin 8) :
  ∃ (A B : Fin 64), (students A).1 ≥ (students B).1 ∧ (students A).2.1 ≥ (students B).2.1 ∧ (students A).2.2 ≥ (students B).2.2 :=
sorry

end part_a_l122_122291


namespace power_calculation_l122_122340

noncomputable def a : ℕ := 3 ^ 1006
noncomputable def b : ℕ := 7 ^ 1007
noncomputable def lhs : ℕ := (a + b)^2 - (a - b)^2
noncomputable def rhs : ℕ := 42 * (10 ^ 1007)

theorem power_calculation : lhs = rhs := by
  sorry

end power_calculation_l122_122340


namespace pirate_islands_probability_l122_122644

open Finset

/-- There are 7 islands.
There is a 1/5 chance of finding an island with treasure only (no traps).
There is a 1/10 chance of finding an island with treasure and traps.
There is a 1/10 chance of finding an island with traps only (no treasure).
There is a 3/5 chance of finding an island with neither treasure nor traps.
We want to prove that the probability of finding exactly 3 islands
with treasure only and the remaining 4 islands with neither treasure
nor traps is 81/2225. -/
theorem pirate_islands_probability :
  (Nat.choose 7 3 : ℚ) * ((1/5)^3) * ((3/5)^4) = 81 / 2225 :=
by
  /- Here goes the proof -/
  sorry

end pirate_islands_probability_l122_122644


namespace range_of_x_minus_y_l122_122320

variable (x y : ℝ)
variable (h1 : 2 < x) (h2 : x < 4) (h3 : -1 < y) (h4 : y < 3)

theorem range_of_x_minus_y : -1 < x - y ∧ x - y < 5 := 
by {
  sorry
}

end range_of_x_minus_y_l122_122320


namespace second_plan_minutes_included_l122_122756

theorem second_plan_minutes_included 
  (monthly_fee1 : ℝ := 50) 
  (limit1 : ℝ := 500) 
  (cost_per_minute1 : ℝ := 0.35) 
  (monthly_fee2 : ℝ := 75) 
  (cost_per_minute2 : ℝ := 0.45) 
  (M : ℝ) 
  (usage : ℝ := 2500)
  (cost1 := monthly_fee1 + cost_per_minute1 * (usage - limit1))
  (cost2 := monthly_fee2 + cost_per_minute2 * (usage - M))
  (equal_costs : cost1 = cost2) : 
  M = 1000 := 
by
  sorry 

end second_plan_minutes_included_l122_122756


namespace common_solutions_for_y_l122_122096

theorem common_solutions_for_y (x y : ℝ) :
  (x^2 + y^2 = 16) ∧ (x^2 - 3 * y = 12) ↔ (y = -4 ∨ y = 1) :=
by
  sorry

end common_solutions_for_y_l122_122096


namespace sum_of_solutions_l122_122387

theorem sum_of_solutions (x : ℝ) (h : x + 16 / x = 12) : x = 8 ∨ x = 4 → 8 + 4 = 12 := by
  sorry

end sum_of_solutions_l122_122387


namespace rectangle_area_l122_122522

theorem rectangle_area (area_square : ℝ) 
  (width_rectangle : ℝ) (length_rectangle : ℝ)
  (h1 : area_square = 16)
  (h2 : width_rectangle^2 = area_square)
  (h3 : length_rectangle = 3 * width_rectangle) :
  width_rectangle * length_rectangle = 48 := by sorry

end rectangle_area_l122_122522


namespace vertex_of_given_function_l122_122338

-- Definition of the given quadratic function
def given_function (x : ℝ) : ℝ := 2 * (x - 4) ^ 2 + 5

-- Definition of the vertex coordinates
def vertex_coordinates : ℝ × ℝ := (4, 5)

-- Theorem stating the vertex coordinates of the function
theorem vertex_of_given_function : (0, given_function 4) = vertex_coordinates :=
by 
  -- Placeholder for the proof
  sorry

end vertex_of_given_function_l122_122338


namespace find_f2_l122_122164

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := x^5 + a * x^3 + b * x - 8

theorem find_f2 (a b : ℝ) (h : f (-2) a b = 10) : f 2 a b = -26 := 
by
  sorry

end find_f2_l122_122164


namespace greendale_high_school_points_l122_122105

-- Define the conditions
def roosevelt_points_first_game : ℕ := 30
def roosevelt_points_second_game : ℕ := roosevelt_points_first_game / 2
def roosevelt_points_third_game : ℕ := roosevelt_points_second_game * 3
def roosevelt_total_points : ℕ := roosevelt_points_first_game + roosevelt_points_second_game + roosevelt_points_third_game
def roosevelt_total_with_bonus : ℕ := roosevelt_total_points + 50
def difference : ℕ := 10

-- Define the assertion for Greendale's points
def greendale_points : ℕ := roosevelt_total_with_bonus - difference

-- The theorem to be proved
theorem greendale_high_school_points : greendale_points = 130 :=
by
  -- Proof should be here, but we add sorry to skip it as per the instructions
  sorry

end greendale_high_school_points_l122_122105


namespace find_n_l122_122436

theorem find_n (n m : ℕ) (h : m = 4) (eq1 : (1/5)^m * (1/4)^n = 1/(10^4)) : n = 2 :=
by
  sorry

end find_n_l122_122436


namespace balls_in_boxes_l122_122660

theorem balls_in_boxes (n m : Nat) (h : n = 6) (k : m = 2) : (m ^ n) = 64 := by
  sorry

end balls_in_boxes_l122_122660


namespace opposite_of_negative_five_l122_122399

theorem opposite_of_negative_five : ∀ x : Int, -5 + x = 0 → x = 5 :=
by
  intros x h
  sorry

end opposite_of_negative_five_l122_122399


namespace tiffany_lives_after_game_l122_122154

/-- Tiffany's initial number of lives -/
def initial_lives : ℕ := 43

/-- Lives Tiffany loses in the hard part of the game -/
def lost_lives : ℕ := 14

/-- Lives Tiffany gains in the next level -/
def gained_lives : ℕ := 27

/-- Calculate the total lives Tiffany has after losing and gaining lives -/
def total_lives : ℕ := (initial_lives - lost_lives) + gained_lives

-- Prove that the total number of lives Tiffany has is 56
theorem tiffany_lives_after_game : total_lives = 56 := by
  -- This is where the proof would go
  sorry

end tiffany_lives_after_game_l122_122154


namespace range_of_a_l122_122242

noncomputable def A (a : ℝ) := {x : ℝ | a < x ∧ x < 2 * a + 1}
def B := {x : ℝ | abs (x - 1) > 2}

theorem range_of_a (a : ℝ) (h : A a ⊆ B) : a ≤ -1 ∨ a ≥ 3 := by
  sorry

end range_of_a_l122_122242


namespace no_valid_partition_of_nat_l122_122868

-- Definitions of the sets A, B, and C as nonempty subsets of positive integers
variable (A B C : Set ℕ)

-- Definition to capture the key condition in the problem
def valid_partition (A B C : Set ℕ) : Prop :=
  (∀ x ∈ A, ∀ y ∈ B, (x^2 - x * y + y^2) ∈ C) 

-- The main theorem stating that such a partition is impossible
theorem no_valid_partition_of_nat : 
  (∃ A B C : Set ℕ, A ≠ ∅ ∧ B ≠ ∅ ∧ C ≠ ∅ ∧ (∀ x ∈ A, ∀ y ∈ B, (x^2 - x * y + y^2) ∈ C)) → False :=
by
  sorry

end no_valid_partition_of_nat_l122_122868


namespace find_integer_n_l122_122195

theorem find_integer_n (n : ℤ) (h : (⌊n^2 / 4⌋ - (⌊n / 2⌋)^2) = 3) : n = 7 :=
sorry

end find_integer_n_l122_122195


namespace tourist_group_people_count_l122_122171

def large_room_people := 3
def small_room_people := 2
def small_rooms_rented := 1
def people_in_small_room := small_rooms_rented * small_room_people

theorem tourist_group_people_count : 
  ∀ x : ℕ, x ≥ 1 ∧ (x + small_rooms_rented) = (people_in_small_room + x * large_room_people) → 
  (people_in_small_room + x * large_room_people) = 5 := 
  by
  sorry

end tourist_group_people_count_l122_122171


namespace func_eq_condition_l122_122342

variable (a : ℝ)

theorem func_eq_condition (f : ℝ → ℝ) :
  (∀ x : ℝ, f (Real.sin x) + a * f (Real.cos x) = Real.cos (2 * x)) ↔ a ∈ (Set.univ \ {1} : Set ℝ) :=
by
  sorry

end func_eq_condition_l122_122342


namespace trajectory_of_Q_existence_of_M_l122_122435

-- Define the circles C1 and C2
def C1 (x y : ℝ) : Prop := (x + 2) ^ 2 + y ^ 2 = 81 / 16
def C2 (x y : ℝ) : Prop := (x - 2) ^ 2 + y ^ 2 = 1 / 16

-- Define the conditions about circle Q
def is_tangent_to_both (Q : ℝ → ℝ → Prop) : Prop :=
  ∃ r : ℝ, (∀ x y : ℝ, Q x y → (x + 2)^2 + y^2 = (r + 9/4)^2) ∧ (∀ x y : ℝ, Q x y → (x - 2)^2 + y^2 = (r + 1/4)^2)

-- Prove the trajectory of the center of Q
theorem trajectory_of_Q (Q : ℝ → ℝ → Prop) (h : is_tangent_to_both Q) :
  ∀ x y : ℝ, Q x y ↔ (x^2 - y^2 / 3 = 1 ∧ x ≥ 1) :=
sorry

-- Prove the existence and coordinates of M
theorem existence_of_M (M : ℝ) (Q : ℝ → ℝ → Prop) (h : is_tangent_to_both Q) :
  ∃ x y : ℝ, (x, y) = (-1, 0) ∧ (∀ x0 y0 : ℝ, Q x0 y0 → ((-y0 / (x0 - 2) = 2 * (y0 / (x0 - M)) / (1 - (y0 / (x0 - M))^2)) ↔ M = -1)) :=
sorry

end trajectory_of_Q_existence_of_M_l122_122435


namespace natasha_time_to_top_l122_122082

theorem natasha_time_to_top (T : ℝ) 
  (descent_time : ℝ) 
  (whole_journey_avg_speed : ℝ) 
  (climbing_speed : ℝ) 
  (desc_time_condition : descent_time = 2) 
  (whole_journey_avg_speed_condition : whole_journey_avg_speed = 3.5) 
  (climbing_speed_condition : climbing_speed = 2.625) 
  (distance_to_top : ℝ := climbing_speed * T) 
  (avg_speed_condition : whole_journey_avg_speed = 2 * distance_to_top / (T + descent_time)) :
  T = 4 := by
  sorry

end natasha_time_to_top_l122_122082


namespace flight_duration_l122_122789

theorem flight_duration (h m : ℕ) (H1 : 11 * 60 + 7 < 14 * 60 + 45) (H2 : 0 < m) (H3 : m < 60) :
  h + m = 41 := 
sorry

end flight_duration_l122_122789


namespace bus_speed_excluding_stoppages_l122_122797

theorem bus_speed_excluding_stoppages (v : ℝ) (stoppage_time : ℝ) (speed_incl_stoppages : ℝ) :
  stoppage_time = 15 / 60 ∧ speed_incl_stoppages = 48 → v = 64 :=
by
  intro h
  sorry

end bus_speed_excluding_stoppages_l122_122797


namespace solution_set_of_inequality_l122_122630

theorem solution_set_of_inequality (x : ℝ) :
  (3 * x + 5) / (x - 1) > x ↔ x < -1 ∨ (1 < x ∧ x < 5) :=
sorry

end solution_set_of_inequality_l122_122630


namespace factorial_division_l122_122870

open Nat

theorem factorial_division : 12! / 11! = 12 := sorry

end factorial_division_l122_122870


namespace fruits_in_box_l122_122274

theorem fruits_in_box (initial_persimmons : ℕ) (added_apples : ℕ) (total_fruits : ℕ) :
  initial_persimmons = 2 → added_apples = 7 → total_fruits = initial_persimmons + added_apples → total_fruits = 9 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end fruits_in_box_l122_122274


namespace range_of_a_l122_122925

noncomputable def f (x a : ℝ) : ℝ := x^2 - 2 * Real.exp 1 * x - Real.log x / x + a

theorem range_of_a (a : ℝ) :
  (∃ x > 0, f x a = 0) → a ≤ Real.exp 2 + 1 / Real.exp 1 := by
  sorry

end range_of_a_l122_122925


namespace correct_operation_l122_122663

theorem correct_operation (x y m c d : ℝ) : (5 * x * y - 4 * x * y = x * y) :=
by sorry

end correct_operation_l122_122663


namespace sum_k_over_3_pow_k_eq_three_fourths_l122_122577

noncomputable def sum_k_over_3_pow_k : ℝ :=
  ∑' k : ℕ, (k + 1) / 3 ^ (k + 1)

theorem sum_k_over_3_pow_k_eq_three_fourths :
  sum_k_over_3_pow_k = 3 / 4 := sorry

end sum_k_over_3_pow_k_eq_three_fourths_l122_122577


namespace uniformity_comparison_l122_122287

theorem uniformity_comparison (S1 S2 : ℝ) (h1 : S1^2 = 13.2) (h2 : S2^2 = 26.26) : S1^2 < S2^2 :=
by {
  sorry
}

end uniformity_comparison_l122_122287


namespace value_of_half_plus_five_l122_122014

theorem value_of_half_plus_five (n : ℕ) (h₁ : n = 20) : (n / 2) + 5 = 15 := 
by {
  sorry
}

end value_of_half_plus_five_l122_122014


namespace perpendicular_lines_condition_perpendicular_lines_sufficient_not_necessary_l122_122477

-- Mathematical definitions and theorems required for the problem
theorem perpendicular_lines_condition (m : ℝ) :
  3 * m + m * (2 * m - 1) = 0 ↔ (m = 0 ∨ m = -1) :=
by sorry

-- Translate the specific problem into Lean
theorem perpendicular_lines_sufficient_not_necessary (m : ℝ) (h : 3 * m + m * (2 * m - 1) = 0) :
  m = -1 ∨ (m ≠ -1 ∧ 3 * m + m * (2 * m - 1) = 0) :=
by sorry

end perpendicular_lines_condition_perpendicular_lines_sufficient_not_necessary_l122_122477


namespace total_weight_of_fish_l122_122972

theorem total_weight_of_fish (fry : ℕ) (survival_rate : ℚ) 
  (first_catch : ℕ) (first_avg_weight : ℚ) 
  (second_catch : ℕ) (second_avg_weight : ℚ)
  (third_catch : ℕ) (third_avg_weight : ℚ)
  (total_weight : ℚ) :
  fry = 100000 ∧ 
  survival_rate = 0.95 ∧ 
  first_catch = 40 ∧ 
  first_avg_weight = 2.5 ∧ 
  second_catch = 25 ∧ 
  second_avg_weight = 2.2 ∧ 
  third_catch = 35 ∧ 
  third_avg_weight = 2.8 ∧ 
  total_weight = fry * survival_rate * 
    ((first_catch * first_avg_weight + 
      second_catch * second_avg_weight + 
      third_catch * third_avg_weight) / 100) / 10000 →
  total_weight = 24 :=
by
  sorry

end total_weight_of_fish_l122_122972


namespace fraction_inequality_l122_122013

variables (a b m : ℝ)

theorem fraction_inequality (h1 : a > b) (h2 : m > 0) : (b + m) / (a + m) > b / a :=
sorry

end fraction_inequality_l122_122013


namespace simple_interest_fraction_l122_122944

theorem simple_interest_fraction (P : ℝ) (R : ℝ) (T : ℝ) (SI : ℝ) (F : ℝ)
  (h1 : R = 5)
  (h2 : T = 4)
  (h3 : SI = (P * R * T) / 100)
  (h4 : SI = F * P) :
  F = 1/5 :=
by
  sorry

end simple_interest_fraction_l122_122944


namespace mn_minus_7_is_negative_one_l122_122638

def opp (x : Int) : Int := -x
def largest_negative_integer : Int := -1
def m := opp (-6)
def n := opp largest_negative_integer

theorem mn_minus_7_is_negative_one : m * n - 7 = -1 := by
  sorry

end mn_minus_7_is_negative_one_l122_122638


namespace three_distinct_solutions_no_solution_for_2009_l122_122548

-- Problem 1: Show that the equation has at least three distinct solutions if it has one
theorem three_distinct_solutions (n : ℕ) (hn : n > 0) :
  (∃ x y : ℤ, x^3 - 3*x*y^2 + y^3 = n) →
  (∃ (x1 y1 x2 y2 x3 y3 : ℤ), 
    x1^3 - 3*x1*y1^2 + y1^3 = n ∧ 
    x2^3 - 3*x2*y2^2 + y2^3 = n ∧ 
    x3^3 - 3*x3*y3^2 + y3^3 = n ∧ 
    (x1, y1) ≠ (x2, y2) ∧ 
    (x1, y1) ≠ (x3, y3) ∧ 
    (x2, y2) ≠ (x3, y3)) :=
sorry

-- Problem 2: Show that the equation has no solutions when n = 2009
theorem no_solution_for_2009 :
  ¬ ∃ x y : ℤ, x^3 - 3*x*y^2 + y^3 = 2009 :=
sorry

end three_distinct_solutions_no_solution_for_2009_l122_122548


namespace angie_age_l122_122680

variables (age : ℕ)

theorem angie_age (h : 2 * age + 4 = 20) : age = 8 :=
sorry

end angie_age_l122_122680


namespace range_of_m_l122_122659

theorem range_of_m (m : ℝ) (x y : ℝ)
  (h1 : x + y - 3 * m = 0)
  (h2 : 2 * x - y + 2 * m - 1 = 0)
  (h3 : x > 0)
  (h4 : y < 0) : 
  -1 < m ∧ m < 1/8 := 
sorry

end range_of_m_l122_122659


namespace nina_homework_total_l122_122595

-- Definitions based on conditions
def ruby_math_homework : Nat := 6
def ruby_reading_homework : Nat := 2
def nina_math_homework : Nat := 4 * ruby_math_homework
def nina_reading_homework : Nat := 8 * ruby_reading_homework
def nina_total_homework : Nat := nina_math_homework + nina_reading_homework

-- The theorem to prove
theorem nina_homework_total : nina_total_homework = 40 := by
  sorry

end nina_homework_total_l122_122595


namespace quadratic_increasing_for_x_geq_3_l122_122982

theorem quadratic_increasing_for_x_geq_3 (x : ℝ) : 
  x ≥ 3 → y = 2 * (x - 3)^2 - 1 → ∃ d > 0, ∀ p ≥ x, y ≤ 2 * (p - 3)^2 - 1 := sorry

end quadratic_increasing_for_x_geq_3_l122_122982


namespace plane_equation_l122_122182

theorem plane_equation (x y z : ℝ)
  (h₁ : ∃ t : ℝ, x = 2 * t + 1 ∧ y = -3 * t ∧ z = 3 - t)
  (h₂ : ∃ (t₁ t₂ : ℝ), 4 * t₁ + 5 * t₂ - 3 = 0 ∧ 2 * t₁ + t₂ + 2 * t₂ = 0) : 
  2*x - y + 7*z - 23 = 0 :=
sorry

end plane_equation_l122_122182


namespace age_of_youngest_child_l122_122672

theorem age_of_youngest_child (x : ℕ) :
  (x + (x + 3) + (x + 6) + (x + 9) + (x + 12) = 55) → x = 5 :=
by
  intro h
  sorry

end age_of_youngest_child_l122_122672


namespace floor_floor_3x_eq_floor_x_plus_1_l122_122244

theorem floor_floor_3x_eq_floor_x_plus_1 (x : ℝ) :
  (⌊⌊3 * x⌋ - 1⌋ = ⌊x + 1⌋) ↔ (2 / 3 ≤ x ∧ x < 4 / 3) :=
by
  sorry

end floor_floor_3x_eq_floor_x_plus_1_l122_122244


namespace distinct_real_c_f_ff_ff_five_l122_122642

def f (x : ℝ) : ℝ := x^2 + 2 * x

theorem distinct_real_c_f_ff_ff_five : 
  (∀ c : ℝ, f (f (f (f c))) = 5 → False) :=
by
  sorry

end distinct_real_c_f_ff_ff_five_l122_122642


namespace xy_value_x2_y2_value_l122_122794

noncomputable def x : ℝ := Real.sqrt 7 + Real.sqrt 3
noncomputable def y : ℝ := Real.sqrt 7 - Real.sqrt 3

theorem xy_value : x * y = 4 := by
  -- proof goes here
  sorry

theorem x2_y2_value : x^2 + y^2 = 20 := by
  -- proof goes here
  sorry

end xy_value_x2_y2_value_l122_122794


namespace harry_terry_difference_l122_122113

theorem harry_terry_difference :
  let H := 8 - (2 + 5)
  let T := 8 - 2 + 5
  H - T = -10 :=
by 
  sorry

end harry_terry_difference_l122_122113


namespace triangle_area_is_sqrt3_over_4_l122_122364

noncomputable def area_of_triangle (a b c : ℝ) (A B C : ℝ) : ℝ :=
  1/2 * b * c * Real.sin A

theorem triangle_area_is_sqrt3_over_4
  (a b c A B : ℝ)
  (h1 : A = Real.pi / 3)
  (h2 : b = 2 * a * Real.cos B)
  (h3 : c = 1)
  (h4 : B = Real.pi / 3)
  (h5 : a = 1)
  (h6 : b = 1) :
  area_of_triangle a b c A B (Real.pi - A - B) = Real.sqrt 3 / 4 :=
by
  sorry

end triangle_area_is_sqrt3_over_4_l122_122364


namespace range_of_f_on_interval_l122_122879

-- Definition of the function
def f (x : ℝ) : ℝ := x^2 - 2 * x + 2

-- Definition of the interval
def domain (x : ℝ) : Prop := -2 ≤ x ∧ x ≤ 2

-- The main statement
theorem range_of_f_on_interval : 
  ∀ y, (∃ x, domain x ∧ f x = y) ↔ (1 ≤ y ∧ y ≤ 10) :=
by
  sorry

end range_of_f_on_interval_l122_122879


namespace two_digit_numbers_l122_122406

theorem two_digit_numbers (n m : ℕ) (Hn : 1 ≤ n ∧ n ≤ 9) (Hm : n < m ∧ m ≤ 9) :
  ∃ (count : ℕ), count = 36 :=
by
  sorry

end two_digit_numbers_l122_122406


namespace students_first_day_l122_122519

-- Definitions based on conditions
def total_books : ℕ := 120
def books_per_student : ℕ := 5
def students_second_day : ℕ := 5
def students_third_day : ℕ := 6
def students_fourth_day : ℕ := 9

-- Main goal
theorem students_first_day (total_books_eq : total_books = 120)
                           (books_per_student_eq : books_per_student = 5)
                           (students_second_day_eq : students_second_day = 5)
                           (students_third_day_eq : students_third_day = 6)
                           (students_fourth_day_eq : students_fourth_day = 9) :
  let books_given_second_day := students_second_day * books_per_student
  let books_given_third_day := students_third_day * books_per_student
  let books_given_fourth_day := students_fourth_day * books_per_student
  let total_books_given_after_first_day := books_given_second_day + books_given_third_day + books_given_fourth_day
  let books_first_day := total_books - total_books_given_after_first_day
  let students_first_day := books_first_day / books_per_student
  students_first_day = 4 :=
by sorry

end students_first_day_l122_122519


namespace bisect_angle_BAX_l122_122802

-- Definitions and conditions
variables {A B C M X : Point}
variable (is_scalene_triangle : ScaleneTriangle A B C)
variable (is_midpoint : Midpoint M B C)
variable (is_parallel : Parallel (Line C X) (Line A B))
variable (angle_right : Angle AM X = 90)

-- The theorem statement to be proven
theorem bisect_angle_BAX (h1 : is_scalene_triangle)
                         (h2 : is_midpoint)
                         (h3 : is_parallel)
                         (h4 : angle_right) :
  Bisects (Line A M) (Angle B A X) :=
sorry

end bisect_angle_BAX_l122_122802


namespace total_coins_l122_122147

def piles_of_quarters : Nat := 5
def piles_of_dimes : Nat := 5
def coins_per_pile : Nat := 3

theorem total_coins :
  (piles_of_quarters * coins_per_pile) + (piles_of_dimes * coins_per_pile) = 30 := by
  sorry

end total_coins_l122_122147


namespace apples_in_basket_l122_122560

-- Definitions based on conditions
def total_apples : ℕ := 138
def apples_per_box : ℕ := 18

-- Problem: prove the number of apples in the basket
theorem apples_in_basket : (total_apples % apples_per_box) = 12 :=
by 
  -- Skip the proof part by adding sorry
  sorry

end apples_in_basket_l122_122560


namespace negation_of_p_is_false_l122_122486

def prop_p : Prop :=
  ∀ x : ℝ, 1 < x → (Real.log (x + 2) / Real.log 3 - 2 / 2^x) > 0

theorem negation_of_p_is_false : ¬(∃ x : ℝ, 1 < x ∧ (Real.log (x + 2) / Real.log 3 - 2 / 2^x) ≤ 0) :=
sorry

end negation_of_p_is_false_l122_122486


namespace calculate_perimeter_l122_122919

-- Definitions based on conditions
def num_posts : ℕ := 36
def post_width : ℕ := 2
def gap_width : ℕ := 4
def sides : ℕ := 4

-- Computations inferred from the conditions (not using solution steps directly)
def posts_per_side : ℕ := num_posts / sides
def gaps_per_side : ℕ := posts_per_side - 1
def side_length : ℕ := posts_per_side * post_width + gaps_per_side * gap_width

-- Theorem statement, proving the perimeter is 200 feet
theorem calculate_perimeter : 4 * side_length = 200 := by
  sorry

end calculate_perimeter_l122_122919


namespace mary_sugar_cups_l122_122277

theorem mary_sugar_cups (sugar_required : ℕ) (sugar_remaining : ℕ) (sugar_added : ℕ) (h1 : sugar_required = 11) (h2 : sugar_added = 1) : sugar_remaining = 10 :=
by
  -- Placeholder for the proof
  sorry

end mary_sugar_cups_l122_122277


namespace steak_amount_per_member_l122_122555

theorem steak_amount_per_member : 
  ∀ (num_members steaks_needed ounces_per_steak total_ounces each_amount : ℕ),
    num_members = 5 →
    steaks_needed = 4 →
    ounces_per_steak = 20 →
    total_ounces = steaks_needed * ounces_per_steak →
    each_amount = total_ounces / num_members →
    each_amount = 16 :=
by
  intros num_members steaks_needed ounces_per_steak total_ounces each_amount
  intro h_members h_steaks h_ounces_per_steak h_total_ounces h_each_amount
  sorry

end steak_amount_per_member_l122_122555


namespace cylinder_cone_volume_ratio_l122_122179

theorem cylinder_cone_volume_ratio (h r_cylinder r_cone : ℝ)
  (hcylinder_csa : π * r_cylinder^2 = π * r_cone^2 / 4):
  (π * r_cylinder^2 * h) / (1 / 3 * π * r_cone^2 * h) = 3 / 4 :=
by
  sorry

end cylinder_cone_volume_ratio_l122_122179


namespace brownie_to_bess_ratio_l122_122816

-- Define daily milk production
def bess_daily_milk : ℕ := 2
def daisy_daily_milk : ℕ := bess_daily_milk + 1

-- Calculate weekly milk production
def bess_weekly_milk : ℕ := bess_daily_milk * 7
def daisy_weekly_milk : ℕ := daisy_daily_milk * 7

-- Given total weekly milk production
def total_weekly_milk : ℕ := 77
def combined_bess_daisy_weekly_milk : ℕ := bess_weekly_milk + daisy_weekly_milk
def brownie_weekly_milk : ℕ := total_weekly_milk - combined_bess_daisy_weekly_milk

-- Main proof statement
theorem brownie_to_bess_ratio : brownie_weekly_milk / bess_weekly_milk = 3 :=
by
  -- Skip the proof
  sorry

end brownie_to_bess_ratio_l122_122816


namespace compute_expression_l122_122535

theorem compute_expression : (3 + 9)^2 + (3^2 + 9^2) = 234 := by
  sorry

end compute_expression_l122_122535


namespace find_m_l122_122838

noncomputable def quadratic_eq (x : ℝ) (m : ℝ) : ℝ := 2 * x^2 + 4 * x + m

theorem find_m (x₁ x₂ m : ℝ) 
  (h1 : quadratic_eq x₁ m = 0)
  (h2 : quadratic_eq x₂ m = 0)
  (h3 : 16 - 8 * m ≥ 0)
  (h4 : x₁^2 + x₂^2 + 2 * x₁ * x₂ - x₁^2 * x₂^2 = 0) 
  : m = -4 :=
sorry

end find_m_l122_122838


namespace rhombus_diagonals_l122_122778

theorem rhombus_diagonals (p d_sum : ℝ) (h₁ : p = 100) (h₂ : d_sum = 62) :
  ∃ d₁ d₂ : ℝ, (d₁ + d₂ = d_sum) ∧ (d₁^2 + d₂^2 = (p/4)^2 * 4) ∧ ((d₁ = 48 ∧ d₂ = 14) ∨ (d₁ = 14 ∧ d₂ = 48)) :=
by
  sorry

end rhombus_diagonals_l122_122778


namespace line_equation_l122_122647

noncomputable def arithmetic_sequence (n : ℕ) (a_1 d : ℝ) : ℝ :=
  a_1 + (n - 1) * d

theorem line_equation
  (a_2 a_4 a_5 : ℝ)
  (a_2_cond : a_2 = arithmetic_sequence 2 a_1 d)
  (a_4_cond : a_4 = arithmetic_sequence 4 a_1 d)
  (a_5_cond : a_5 = arithmetic_sequence 5 a_1 d)
  (sum_cond : a_2 + a_4 = 12)
  (a_5_val : a_5 = 10)
  (circle_eq : ∀ x y : ℝ, x^2 + y^2 - 2 * y = 0 ↔ (x - 0)^2 + (y - 1)^2 = 1)
  : ∃ (line : ℝ → ℝ → Prop), line x y ↔ (6 * x - y + 1 = 0) :=
by
  sorry

end line_equation_l122_122647


namespace katie_initial_candies_l122_122997

theorem katie_initial_candies (K : ℕ) (h1 : K + 23 - 8 = 23) : K = 8 :=
sorry

end katie_initial_candies_l122_122997


namespace staircase_problem_l122_122864

theorem staircase_problem :
  ∃ (n : ℕ), (n > 20) ∧ (n % 5 = 4) ∧ (n % 6 = 3) ∧ (n % 7 = 5) ∧ n = 159 :=
by sorry

end staircase_problem_l122_122864


namespace find_TU_square_l122_122935

-- Definitions
variables (P Q R S T U : ℝ × ℝ)
variable (side : ℝ)
variable (QT RU PT SU PQ : ℝ)

-- Setting the conditions
variables (side_eq_10 : side = 10)
variables (QT_eq_7 : QT = 7)
variables (RU_eq_7 : RU = 7)
variables (PT_eq_24 : PT = 24)
variables (SU_eq_24 : SU = 24)
variables (PQ_eq_10 : PQ = 10)

-- The theorem statement
theorem find_TU_square : TU^2 = 1150 :=
by
  -- Proof to be done here.
  sorry

end find_TU_square_l122_122935


namespace least_number_of_cookies_l122_122372

theorem least_number_of_cookies :
  ∃ x : ℕ, x % 6 = 4 ∧ x % 5 = 3 ∧ x % 8 = 6 ∧ x % 9 = 7 ∧ x = 208 :=
by
  sorry

end least_number_of_cookies_l122_122372


namespace total_sand_weight_is_34_l122_122451

-- Define the conditions
def eden_buckets : ℕ := 4
def mary_buckets : ℕ := eden_buckets + 3
def iris_buckets : ℕ := mary_buckets - 1
def weight_per_bucket : ℕ := 2

-- Define the total weight calculation
def total_buckets : ℕ := eden_buckets + mary_buckets + iris_buckets
def total_weight : ℕ := total_buckets * weight_per_bucket

-- The proof statement
theorem total_sand_weight_is_34 : total_weight = 34 := by
  sorry

end total_sand_weight_is_34_l122_122451


namespace black_area_remaining_after_changes_l122_122529

theorem black_area_remaining_after_changes :
  let initial_fraction_black := 1
  let change_factor := 8 / 9
  let num_changes := 4
  let final_fraction_black := (change_factor ^ num_changes)
  final_fraction_black = 4096 / 6561 :=
by
  sorry

end black_area_remaining_after_changes_l122_122529


namespace ratio_fifth_terms_l122_122533

-- Define the arithmetic sequences and their sums
variables {a b : ℕ → ℕ}
variables {S T : ℕ → ℕ}

-- Assume conditions of the problem
axiom sum_condition (n : ℕ) : S n = n * (a 1 + a n) / 2
axiom sum_condition2 (n : ℕ) : T n = n * (b 1 + b n) / 2
axiom ratio_condition : ∀ n, S n / T n = (2 * n - 3) / (3 * n - 2)

-- Prove the ratio of fifth terms a_5 / b_5
theorem ratio_fifth_terms : (a 5 : ℚ) / b 5 = 3 / 5 := by
  sorry

end ratio_fifth_terms_l122_122533


namespace max_value_of_expression_l122_122983

variable (a b c d : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c) (h₄ : 0 < d) (h₅ : a + b + c + d = 3)

theorem max_value_of_expression :
  3 * a^2 * b^3 * c * d^2 ≤ 177147 / 40353607 :=
sorry

end max_value_of_expression_l122_122983


namespace find_f_sqrt2_l122_122883

noncomputable def f : ℝ → ℝ := sorry

axiom f_domain : ∀ x, x > 0 → (∃ y, f y = x ∨ y = x)

axiom f_multiplicative : ∀ x y : ℝ, x > 0 → y > 0 → f (x * y) = f x + f y
axiom f_at_8 : f 8 = 3

-- Define the problem statement
theorem find_f_sqrt2 : f (Real.sqrt 2) = 1 / 2 := sorry

end find_f_sqrt2_l122_122883


namespace problem_statement_l122_122360

theorem problem_statement (a x : ℝ) (h_linear_eq : (a + 4) * x ^ |a + 3| + 8 = 0) : a^2 + a - 1 = 1 :=
sorry

end problem_statement_l122_122360


namespace solution_set_M_inequality_ab_l122_122031

def f (x : ℝ) : ℝ := |x - 1| + |x + 3|

theorem solution_set_M :
  {x | -3 ≤ x ∧ x ≤ 1} = { x : ℝ | f x ≤ 4 } :=
sorry

theorem inequality_ab
  (a b : ℝ) (h1 : -3 ≤ a ∧ a ≤ 1) (h2 : -3 ≤ b ∧ b ≤ 1) :
  (a^2 + 2 * a - 3) * (b^2 + 2 * b - 3) ≥ 0 :=
sorry

end solution_set_M_inequality_ab_l122_122031


namespace total_number_of_notes_l122_122335

theorem total_number_of_notes 
  (total_money : ℕ)
  (fifty_rupees_notes : ℕ)
  (five_hundred_rupees_notes : ℕ)
  (total_money_eq : total_money = 10350)
  (fifty_rupees_notes_eq : fifty_rupees_notes = 117)
  (money_eq : 50 * fifty_rupees_notes + 500 * five_hundred_rupees_notes = total_money) :
  fifty_rupees_notes + five_hundred_rupees_notes = 126 :=
by sorry

end total_number_of_notes_l122_122335


namespace planA_equals_planB_at_3_l122_122381

def planA_charge_for_first_9_minutes : ℝ := 0.24
def planA_charge (X: ℝ) (minutes: ℕ) : ℝ := if minutes <= 9 then X else X + 0.06 * (minutes - 9)
def planB_charge (minutes: ℕ) : ℝ := 0.08 * minutes

theorem planA_equals_planB_at_3 : planA_charge planA_charge_for_first_9_minutes 3 = planB_charge 3 :=
by sorry

end planA_equals_planB_at_3_l122_122381


namespace exists_xyz_l122_122907

noncomputable def sum_of_digits (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.sum

theorem exists_xyz :
  ∃ x y z : ℕ, x > 0 ∧ y > 0 ∧ z > 0 ∧
  (x + sum_of_digits x = y + sum_of_digits y ∧ y + sum_of_digits y = z + sum_of_digits z) :=
by {
  sorry
}

end exists_xyz_l122_122907


namespace jacket_price_is_48_l122_122585

-- Definitions according to the conditions
def jacket_problem (P S D : ℝ) : Prop :=
  S = P + 0.40 * S ∧
  D = 0.80 * S ∧
  16 = D - P

-- Statement of the theorem
theorem jacket_price_is_48 :
  ∃ P S D, jacket_problem P S D ∧ P = 48 :=
by
  sorry

end jacket_price_is_48_l122_122585


namespace correct_statement_l122_122269

def synthetic_method_is_direct : Prop := -- define the synthetic method
  True  -- We'll say True to assume it's a direct proof method. This is a simplification.

def analytic_method_is_direct : Prop := -- define the analytic method
  True  -- We'll say True to assume it's a direct proof method. This is a simplification.

theorem correct_statement : synthetic_method_is_direct ∧ analytic_method_is_direct → 
                             "Synthetic method and analytic method are direct proof methods" = "A" :=
by
  intros h
  cases h
  -- This is where you would provide the proof steps. We skip this with sorry.
  sorry

end correct_statement_l122_122269


namespace findPrincipalAmount_l122_122718

noncomputable def principalAmount (r : ℝ) (t : ℝ) (diff : ℝ) : ℝ :=
  let n := 2 -- compounded semi-annually
  let rate_per_period := (1 + r / n)
  let num_periods := n * t
  (diff / (rate_per_period^num_periods - 1 - r * t))

theorem findPrincipalAmount :
  let r := 0.05
  let t := 3
  let diff := 25
  abs (principalAmount r t diff - 2580.39) < 0.01 := 
by 
  sorry

end findPrincipalAmount_l122_122718


namespace polygon_sides_l122_122299

theorem polygon_sides (n : ℕ) : (n - 2) * 180 + 360 = 1980 → n = 11 :=
by sorry

end polygon_sides_l122_122299


namespace rest_area_milepost_l122_122157

theorem rest_area_milepost (milepost_first : ℕ) (milepost_seventh : ℕ) (h_first : milepost_first = 20) (h_seventh : milepost_seventh = 140) : 
  ∃ milepost_rest : ℕ, milepost_rest = (milepost_first + milepost_seventh) / 2 ∧ milepost_rest = 80 :=
by
  sorry

end rest_area_milepost_l122_122157


namespace simplify_f_value_of_f_l122_122684

noncomputable def f (α : ℝ) : ℝ :=
  (Real.sin (α - (5 * Real.pi) / 2) * Real.cos ((3 * Real.pi) / 2 + α) * Real.tan (Real.pi - α)) /
  (Real.tan (-α - Real.pi) * Real.sin (Real.pi - α))

theorem simplify_f (α : ℝ) : f α = -Real.cos α := by
  sorry

theorem value_of_f (α : ℝ)
  (h : Real.cos (α + (3 * Real.pi) / 2) = 1 / 5)
  (h2 : α > Real.pi / 2 ∧ α < Real.pi ) : 
  f α = 2 * Real.sqrt 6 / 5 := by
  sorry

end simplify_f_value_of_f_l122_122684


namespace sequence_formula_l122_122064

theorem sequence_formula (a : ℕ → ℕ) (h1 : a 1 = 1) (h2 : ∀ n > 1, a n - a (n - 1) = 2^(n-1)) : a n = 2^n - 1 := 
sorry

end sequence_formula_l122_122064


namespace number_of_hours_sold_l122_122234

def packs_per_hour_peak := 6
def packs_per_hour_low := 4
def price_per_pack := 60
def extra_revenue := 1800

def revenue_per_hour_peak := packs_per_hour_peak * price_per_pack
def revenue_per_hour_low := packs_per_hour_low * price_per_pack
def revenue_diff_per_hour := revenue_per_hour_peak - revenue_per_hour_low

theorem number_of_hours_sold (h : ℕ) 
  (h_eq : revenue_diff_per_hour * h = extra_revenue) : 
  h = 15 :=
by
  -- skip proof
  sorry

end number_of_hours_sold_l122_122234


namespace cost_of_berries_and_cheese_l122_122200

variables (b m l c : ℕ)

theorem cost_of_berries_and_cheese (h1 : b + m + l + c = 25)
                                  (h2 : m = 2 * l)
                                  (h3 : c = b + 2) : 
                                  b + c = 10 :=
by {
  -- proof omitted, this is just the statement
  sorry
}

end cost_of_berries_and_cheese_l122_122200


namespace customer_paid_l122_122970

theorem customer_paid (cost_price : ℕ) (markup_percent : ℕ) (selling_price : ℕ) : 
  cost_price = 6672 → markup_percent = 25 → selling_price = cost_price + (markup_percent * cost_price / 100) → selling_price = 8340 :=
by
  intros h_cost_price h_markup_percent h_selling_price
  rw [h_cost_price, h_markup_percent] at h_selling_price
  exact h_selling_price

end customer_paid_l122_122970


namespace tan_sum_identity_l122_122848

theorem tan_sum_identity
  (A B C : ℝ)
  (h1 : A + B + C = Real.pi)
  (h2 : Real.tan A + Real.tan B + Real.tan C = Real.tan A * Real.tan B * Real.tan C) :
  Real.tan A + Real.tan B + Real.tan C = Real.tan A * Real.tan B * Real.tan C := 
sorry

end tan_sum_identity_l122_122848


namespace find_point_M_l122_122846

/-- Define the function f(x) = x^3 + x - 2. -/
def f (x : ℝ) : ℝ := x^3 + x - 2

/-- Define the derivative of the function, f'(x). -/
def f' (x : ℝ) : ℝ := 3 * x^2 + 1

/-- Define the condition that the slope of the tangent line is perpendicular to y = -1/4x - 1. -/
def slope_perpendicular_condition (m : ℝ) : Prop := m = 4

/-- Main theorem: The coordinates of the point M are (1, 0) and (-1, -4). -/
theorem find_point_M : 
  ∃ (x₀ y₀ : ℝ), f x₀ = y₀ ∧ slope_perpendicular_condition (f' x₀) ∧ 
  ((x₀ = 1 ∧ y₀ = 0) ∨ (x₀ = -1 ∧ y₀ = -4)) := 
sorry

end find_point_M_l122_122846


namespace find_nat_numbers_l122_122255

theorem find_nat_numbers (a b : ℕ) (c : ℕ) (h : ∀ n : ℕ, a^n + b^n = c^(n+1)) : a = 2 ∧ b = 2 ∧ c = 2 :=
by
  sorry

end find_nat_numbers_l122_122255


namespace parabola_relationship_l122_122117

theorem parabola_relationship 
  (c : ℝ) (y1 y2 y3 : ℝ) 
  (h1 : y1 = 2*(-2 - 1)^2 + c) 
  (h2 : y2 = 2*(0 - 1)^2 + c) 
  (h3 : y3 = 2*((5:ℝ)/3 - 1)^2 + c):
  y1 > y2 ∧ y2 > y3 :=
by
  sorry

end parabola_relationship_l122_122117


namespace problem_min_value_l122_122196

theorem problem_min_value {a b c : ℝ} (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a^2 + a * b + a * c + b * c = 4) : 
  (2 * a + b + c) ≥ 4 := 
  sorry

end problem_min_value_l122_122196


namespace error_in_step_one_l122_122284

theorem error_in_step_one : 
  ∃ a b c d : ℝ, 
    (a * (x + 1) - b = c * (x - 2)) = (3 * (x + 1) - 6 = 2 * (x - 2)) → 
    a ≠ 3 ∨ b ≠ 6 ∨ c ≠ 2 := 
by
  sorry

end error_in_step_one_l122_122284


namespace polygon_interior_angle_sum_l122_122690

theorem polygon_interior_angle_sum (n : ℕ) (h : 180 * (n - 2) = 2340) :
  180 * (n - 2 + 3) = 2880 := by
  sorry

end polygon_interior_angle_sum_l122_122690


namespace computer_price_difference_l122_122742

-- Define the conditions as stated
def basic_computer_price := 1500
def total_price := 2500
def printer_price (P : ℕ) := basic_computer_price + P = total_price

def enhanced_computer_price (P E : ℕ) := P = (E + P) / 3

-- The theorem stating the proof problem
theorem computer_price_difference (P E : ℕ) 
  (h1 : printer_price P) 
  (h2 : enhanced_computer_price P E) : E - basic_computer_price = 500 :=
sorry

end computer_price_difference_l122_122742


namespace evaluate_fraction_l122_122825

theorem evaluate_fraction :
  (20 - 18 + 16 - 14 + 12 - 10 + 8 - 6 + 4 - 2) / (2 - 4 + 6 - 8 + 10 - 12 + 14 - 16 + 18) = 1 :=
by
  sorry

end evaluate_fraction_l122_122825


namespace simplify_fraction_l122_122975

theorem simplify_fraction (a : ℝ) (h1 : a ≠ 4) (h2 : a ≠ -4) : 
  (2 * a / (a^2 - 16) - 1 / (a - 4) = 1 / (a + 4)) := 
by 
  sorry 

end simplify_fraction_l122_122975


namespace inverse_proportionality_ratio_l122_122162

variable {x y k x1 x2 y1 y2 : ℝ}

theorem inverse_proportionality_ratio
  (h1 : x * y = k)
  (hx1 : x1 ≠ 0)
  (hx2 : x2 ≠ 0)
  (hy1 : y1 ≠ 0)
  (hy2 : y2 ≠ 0)
  (hx_ratio : x1 / x2 = 3 / 4)
  (hxy1 : x1 * y1 = k)
  (hxy2 : x2 * y2 = k) :
  y1 / y2 = 4 / 3 := by
  sorry

end inverse_proportionality_ratio_l122_122162


namespace marbles_in_jar_l122_122991

theorem marbles_in_jar (M : ℕ) (h1 : ∀ n : ℕ, n = 20 → ∀ m : ℕ, m = M / n → ∀ a b : ℕ, a = n + 2 → b = m - 1 → ∀ k : ℕ, k = M / a → k = b) : M = 220 :=
by 
  sorry

end marbles_in_jar_l122_122991


namespace travel_time_difference_is_58_minutes_l122_122668

-- Define the distances and speeds for Minnie
def minnie_uphill_distance := 15
def minnie_uphill_speed := 10
def minnie_downhill_distance := 25
def minnie_downhill_speed := 40
def minnie_flat_distance := 30
def minnie_flat_speed := 25

-- Define the distances and speeds for Penny
def penny_flat_distance := 30
def penny_flat_speed := 35
def penny_downhill_distance := 25
def penny_downhill_speed := 50
def penny_uphill_distance := 15
def penny_uphill_speed := 15

-- Calculate Minnie's total travel time in hours
def minnie_time := (minnie_uphill_distance / minnie_uphill_speed) + 
                   (minnie_downhill_distance / minnie_downhill_speed) + 
                   (minnie_flat_distance / minnie_flat_speed)

-- Calculate Penny's total travel time in hours
def penny_time := (penny_flat_distance / penny_flat_speed) + 
                  (penny_downhill_distance / penny_downhill_speed) +
                  (penny_uphill_distance / penny_uphill_speed)

-- Calculate difference in minutes
def time_difference_minutes := (minnie_time - penny_time) * 60

-- The proof statement
theorem travel_time_difference_is_58_minutes :
  time_difference_minutes = 58 := by
  sorry

end travel_time_difference_is_58_minutes_l122_122668


namespace daily_egg_count_per_female_emu_l122_122215

noncomputable def emus_per_pen : ℕ := 6
noncomputable def pens : ℕ := 4
noncomputable def total_eggs_per_week : ℕ := 84

theorem daily_egg_count_per_female_emu :
  (total_eggs_per_week / ((pens * emus_per_pen) / 2 * 7) = 1) :=
by
  sorry

end daily_egg_count_per_female_emu_l122_122215


namespace geometric_sequence_ratio_l122_122392

theorem geometric_sequence_ratio (a : ℕ → ℤ) (q : ℤ) (n : ℕ) (i : ℕ → ℕ) (ε : ℕ → ℤ) :
  (∀ (k : ℕ), 1 ≤ k ∧ k ≤ n → a k = a 1 * q ^ (k - 1)) ∧
  (∃ (k : ℕ), 1 ≤ k ∧ k ≤ n → ε k * a (i k) = 0) ∧
  (∀ m, 1 ≤ i m ∧ i m ≤ n) → q = -1 := 
sorry

end geometric_sequence_ratio_l122_122392


namespace calculate_expression_l122_122128

theorem calculate_expression :
  2⁻¹ + (3 - Real.pi)^0 + abs (2 * Real.sqrt 3 - Real.sqrt 2) + 2 * Real.cos (Real.pi / 4) - Real.sqrt 12 = 3 / 2 :=
sorry

end calculate_expression_l122_122128


namespace linear_inequality_solution_l122_122251

theorem linear_inequality_solution {x y m n : ℤ} 
  (h_table: (∀ x, if x = -2 then y = 3 
                else if x = -1 then y = 2 
                else if x = 0 then y = 1 
                else if x = 1 then y = 0 
                else if x = 2 then y = -1 
                else if x = 3 then y = -2 
                else true)) 
  (h_eq: m * x - n = y) : 
  x ≥ -1 :=
sorry

end linear_inequality_solution_l122_122251


namespace positive_root_condition_negative_root_condition_zero_root_condition_l122_122107

variable (a b c : ℝ)

-- Condition for a positive root
theorem positive_root_condition : 
  ((a > 0 ∧ b > c) ∨ (a < 0 ∧ b < c)) ↔ (∃ x : ℝ, x > 0 ∧ a * x = b - c) :=
sorry

-- Condition for a negative root
theorem negative_root_condition : 
  ((a > 0 ∧ b < c) ∨ (a < 0 ∧ b > c)) ↔ (∃ x : ℝ, x < 0 ∧ a * x = b - c) :=
sorry

-- Condition for a root equal to zero
theorem zero_root_condition : 
  (a ≠ 0 ∧ b = c) ↔ (∃ x : ℝ, x = 0 ∧ a * x = b - c) :=
sorry

end positive_root_condition_negative_root_condition_zero_root_condition_l122_122107


namespace difference_of_squares_l122_122763

theorem difference_of_squares (x y : ℕ) (h1 : x + y = 60) (h2 : x - y = 16) : x^2 - y^2 = 960 :=
by
  sorry

end difference_of_squares_l122_122763
