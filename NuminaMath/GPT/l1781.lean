import Mathlib

namespace NUMINAMATH_GPT_vertical_distance_l1781_178138

variable (storiesPerTrip tripsPerDay daysPerWeek feetPerStory : ℕ)

def totalVerticalDistance
  (storiesPerTrip tripsPerDay daysPerWeek feetPerStory : ℕ) : ℕ :=
  2 * storiesPerTrip * feetPerStory * tripsPerDay * daysPerWeek

theorem vertical_distance (h1 : storiesPerTrip = 5)
                          (h2 : tripsPerDay = 3)
                          (h3 : daysPerWeek = 7)
                          (h4 : feetPerStory = 10) :
  totalVerticalDistance storiesPerTrip tripsPerDay daysPerWeek feetPerStory = 2100 := by
  sorry

end NUMINAMATH_GPT_vertical_distance_l1781_178138


namespace NUMINAMATH_GPT_correct_truth_values_l1781_178145

open Real

def proposition_p : Prop := ∀ (a : ℝ), 0 < a → a^2 ≠ 0

def converse_p : Prop := ∀ (a : ℝ), a^2 ≠ 0 → 0 < a

def inverse_p : Prop := ∀ (a : ℝ), ¬(0 < a) → a^2 = 0

def contrapositive_p : Prop := ∀ (a : ℝ), a^2 = 0 → ¬(0 < a)

def negation_p : Prop := ∃ (a : ℝ), 0 < a ∧ a^2 = 0

theorem correct_truth_values : 
  (converse_p = False) ∧ 
  (inverse_p = False) ∧ 
  (contrapositive_p = True) ∧ 
  (negation_p = False) := by
  sorry

end NUMINAMATH_GPT_correct_truth_values_l1781_178145


namespace NUMINAMATH_GPT_find_seven_m_squared_minus_one_l1781_178155

theorem find_seven_m_squared_minus_one (m : ℝ)
  (h1 : ∃ x₁, 5 * m + 3 * x₁ = 1 + x₁)
  (h2 : ∃ x₂, 2 * x₂ + m = 3 * m)
  (h3 : ∀ x₁ x₂, (5 * m + 3 * x₁ = 1 + x₁) → (2 * x₂ + m = 3 * m) → x₁ = x₂ + 2) :
  7 * m^2 - 1 = 2 / 7 :=
by
  let m := -3/7
  sorry

end NUMINAMATH_GPT_find_seven_m_squared_minus_one_l1781_178155


namespace NUMINAMATH_GPT_find_a4_l1781_178162

variable (a : ℕ → ℝ)
variable (q : ℝ)
variable (T_7 : ℝ)

-- Conditions
axiom geom_seq (n : ℕ) : a (n + 1) = q * a n
axiom common_ratio_ne_one : q ≠ 1
axiom product_first_seven_terms : (a 1) * (a 2) * (a 3) * (a 4) * (a 5) * (a 6) * (a 7) = 128

-- Goal
theorem find_a4 : a 4 = 2 :=
sorry

end NUMINAMATH_GPT_find_a4_l1781_178162


namespace NUMINAMATH_GPT_find_r_values_l1781_178173

theorem find_r_values (r : ℝ) (h1 : r ≥ 8) (h2 : r ≤ 20) :
  16 ≤ (r - 4) ^ (3/2) ∧ (r - 4) ^ (3/2) ≤ 128 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_r_values_l1781_178173


namespace NUMINAMATH_GPT_circle_equation1_circle_equation2_l1781_178177

-- Definitions for the first question
def center1 : (ℝ × ℝ) := (2, -2)
def pointP : (ℝ × ℝ) := (6, 3)

-- Definitions for the second question
def pointA : (ℝ × ℝ) := (-4, -5)
def pointB : (ℝ × ℝ) := (6, -1)

-- Theorems we need to prove
theorem circle_equation1 : (x - 2)^2 + (y + 2)^2 = 41 :=
sorry

theorem circle_equation2 : (x - 1)^2 + (y + 3)^2 = 29 :=
sorry

end NUMINAMATH_GPT_circle_equation1_circle_equation2_l1781_178177


namespace NUMINAMATH_GPT_flower_pots_problem_l1781_178192

noncomputable def cost_of_largest_pot (x : ℝ) : ℝ := x + 5 * 0.15

theorem flower_pots_problem
  (x : ℝ)       -- cost of the smallest pot
  (total_cost : ℝ) -- total cost of all pots
  (h_total_cost : total_cost = 8.25)
  (h_price_relation : total_cost = 6 * x + (0.15 + 2 * 0.15 + 3 * 0.15 + 4 * 0.15 + 5 * 0.15)) :
  cost_of_largest_pot x = 1.75 :=
by
  sorry

end NUMINAMATH_GPT_flower_pots_problem_l1781_178192


namespace NUMINAMATH_GPT_range_of_a_l1781_178131

open Set

variable (a : ℝ)

noncomputable def I := univ ℝ
noncomputable def A := {x : ℝ | x ≤ a + 1}
noncomputable def B := {x : ℝ | x ≥ 1}
noncomputable def complement_B := {x : ℝ | x < 1}

theorem range_of_a (h : A a ⊆ complement_B) : a < 0 := sorry

end NUMINAMATH_GPT_range_of_a_l1781_178131


namespace NUMINAMATH_GPT_krishna_fraction_wins_l1781_178180

theorem krishna_fraction_wins (matches_total : ℕ) (callum_points : ℕ) (points_per_win : ℕ) (callum_wins : ℕ) :
  matches_total = 8 → callum_points = 20 → points_per_win = 10 → callum_wins = callum_points / points_per_win →
  (matches_total - callum_wins) / matches_total = 3 / 4 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_krishna_fraction_wins_l1781_178180


namespace NUMINAMATH_GPT_number_from_first_group_is_6_l1781_178165

-- Defining conditions
def num_students : Nat := 160
def sample_size : Nat := 20
def groups := List.range' 0 num_students (num_students / sample_size)

def num_from_group_16 (x : Nat) : Nat := 8 * 15 + x
def drawn_number_from_16 : Nat := 126

-- Main theorem
theorem number_from_first_group_is_6 : ∃ x : Nat, num_from_group_16 x = drawn_number_from_16 ∧ x = 6 := 
by
  sorry

end NUMINAMATH_GPT_number_from_first_group_is_6_l1781_178165


namespace NUMINAMATH_GPT_positive_solution_x_l1781_178119

theorem positive_solution_x (x y z : ℝ) (h1 : x * y = 8 - x - 4 * y) (h2 : y * z = 12 - 3 * y - 6 * z) (h3 : x * z = 40 - 5 * x - 2 * z) (hy : y = 3) (hz : z = -1) : x = 6 :=
by
  sorry

end NUMINAMATH_GPT_positive_solution_x_l1781_178119


namespace NUMINAMATH_GPT_meeting_time_final_time_statement_l1781_178111

-- Define the speeds and distance as given conditions
def brodie_speed : ℝ := 50
def ryan_speed : ℝ := 40
def initial_distance : ℝ := 120

-- Define what we know about their meeting time and validate it mathematically
theorem meeting_time :
  (initial_distance / (brodie_speed + ryan_speed)) = 4 / 3 := sorry

-- Assert the time in minutes for completeness
noncomputable def time_in_minutes : ℝ := ((4 / 3) * 60)

-- Assert final statement matching the answer in hours and minutes
theorem final_time_statement :
  time_in_minutes = 80 := sorry

end NUMINAMATH_GPT_meeting_time_final_time_statement_l1781_178111


namespace NUMINAMATH_GPT_haley_lives_gained_l1781_178140

-- Define the given conditions
def initial_lives : ℕ := 14
def lives_lost : ℕ := 4
def total_lives_after_gain : ℕ := 46

-- Define the goal: How many lives did Haley gain in the next level?
theorem haley_lives_gained : (total_lives_after_gain = initial_lives - lives_lost + lives_gained) → lives_gained = 36 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_haley_lives_gained_l1781_178140


namespace NUMINAMATH_GPT_sum_of_distances_minimized_l1781_178189

theorem sum_of_distances_minimized (x : ℝ) (h : 0 ≤ x ∧ x ≤ 50) : 
  abs (x - 0) + abs (x - 50) = 50 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_distances_minimized_l1781_178189


namespace NUMINAMATH_GPT_expression_meaningful_l1781_178185

theorem expression_meaningful (x : ℝ) : (∃ y, y = 4 / (x - 5)) ↔ x ≠ 5 :=
by
  sorry

end NUMINAMATH_GPT_expression_meaningful_l1781_178185


namespace NUMINAMATH_GPT_twenty_five_percent_M_eq_thirty_five_percent_1504_l1781_178123

theorem twenty_five_percent_M_eq_thirty_five_percent_1504 (M : ℝ) : 
  0.25 * M = 0.35 * 1504 → M = 2105.6 :=
by
  sorry

end NUMINAMATH_GPT_twenty_five_percent_M_eq_thirty_five_percent_1504_l1781_178123


namespace NUMINAMATH_GPT_bronze_status_families_count_l1781_178169

theorem bronze_status_families_count :
  ∃ B : ℕ, (B * 25) = (700 - (7 * 50 + 1 * 100)) ∧ B = 10 := 
sorry

end NUMINAMATH_GPT_bronze_status_families_count_l1781_178169


namespace NUMINAMATH_GPT_scientific_notation_of_0_0000007_l1781_178142

theorem scientific_notation_of_0_0000007 :
  0.0000007 = 7 * 10 ^ (-7) :=
  by
  sorry

end NUMINAMATH_GPT_scientific_notation_of_0_0000007_l1781_178142


namespace NUMINAMATH_GPT_largest_of_four_numbers_l1781_178136

theorem largest_of_four_numbers 
  (a b c d : ℝ) 
  (h1 : a + 5 = b^2 - 1) 
  (h2 : a + 5 = c^2 + 3) 
  (h3 : a + 5 = d - 4) 
  : d > max (max a b) c :=
sorry

end NUMINAMATH_GPT_largest_of_four_numbers_l1781_178136


namespace NUMINAMATH_GPT_recolor_possible_l1781_178105

theorem recolor_possible (cell_color : Fin 50 → Fin 50 → Fin 100)
  (H1 : ∀ i j, ∃ k l, cell_color i j = k ∧ cell_color i (j+1) = l ∧ k ≠ l ∧ j < 49)
  (H2 : ∀ i j, ∃ k l, cell_color i j = k ∧ cell_color (i+1) j = l ∧ k ≠ l ∧ i < 49) :
  ∃ c1 c2, (c1 ≠ c2) ∧
  ∀ i j, (cell_color i j = c1 → cell_color i j = c2 ∨ ∀ k l, (cell_color k l = c1 → cell_color k l ≠ c2)) :=
  by
  sorry

end NUMINAMATH_GPT_recolor_possible_l1781_178105


namespace NUMINAMATH_GPT_additional_cards_l1781_178107

theorem additional_cards (total_cards : ℕ) (complete_decks : ℕ) (cards_per_deck : ℕ) (num_decks : ℕ) 
  (h1 : total_cards = 160) (h2 : num_decks = 3) (h3 : cards_per_deck = 52) :
  total_cards - (num_decks * cards_per_deck) = 4 :=
by
  sorry

end NUMINAMATH_GPT_additional_cards_l1781_178107


namespace NUMINAMATH_GPT_minimum_value_x_plus_4_div_x_l1781_178199

theorem minimum_value_x_plus_4_div_x (x : ℝ) (hx : x > 0) : x + 4 / x ≥ 4 :=
sorry

end NUMINAMATH_GPT_minimum_value_x_plus_4_div_x_l1781_178199


namespace NUMINAMATH_GPT_calculate_expression_l1781_178128

theorem calculate_expression :
  ((2000000000000 - 1234567890123) * 3 = 2296296329631) :=
by 
  sorry

end NUMINAMATH_GPT_calculate_expression_l1781_178128


namespace NUMINAMATH_GPT_minimum_value_of_fraction_l1781_178150

variable {a b : ℝ}

theorem minimum_value_of_fraction (h1 : a > b) (h2 : a * b = 1) : 
  ∃ (c : ℝ), c = 2 * Real.sqrt 2 ∧ ∀ x > b, a * x = 1 -> 
  (x - b + 2 / (x - b) ≥ c) :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_of_fraction_l1781_178150


namespace NUMINAMATH_GPT_alex_silver_tokens_l1781_178143

theorem alex_silver_tokens :
  let R : Int -> Int -> Int := fun x y => 100 - 3 * x + 2 * y
  let B : Int -> Int -> Int := fun x y => 50 + 2 * x - 4 * y
  let x := 61
  let y := 42
  100 - 3 * x + 2 * y < 3 → 50 + 2 * x - 4 * y < 4 → x + y = 103 :=
by
  intro hR hB
  sorry

end NUMINAMATH_GPT_alex_silver_tokens_l1781_178143


namespace NUMINAMATH_GPT_circles_intersect_twice_l1781_178146

noncomputable def circle1 (x y : ℝ) : Prop :=
  (x - 3)^2 + y^2 = 9

noncomputable def circle2 (x y : ℝ) : Prop :=
  x^2 + (y - 1.5)^2 = 9 / 4

theorem circles_intersect_twice : 
  (∃ (p : ℝ × ℝ), circle1 p.1 p.2 ∧ circle2 p.1 p.2) ∧ 
  (∀ (p q : ℝ × ℝ), circle1 p.1 p.2 ∧ circle2 p.1 p.2 ∧ circle1 q.1 q.2 ∧ circle2 q.1 q.2 → (p = q ∨ p ≠ q)) →
  ∃ (p1 p2 : ℝ × ℝ), 
    p1 ≠ p2 ∧
    circle1 p1.1 p1.2 ∧ circle2 p1.1 p1.2 ∧
    circle1 p2.1 p2.2 ∧ circle2 p2.1 p2.2 := 
by {
  sorry
}

end NUMINAMATH_GPT_circles_intersect_twice_l1781_178146


namespace NUMINAMATH_GPT_alchemerion_age_problem_l1781_178151

theorem alchemerion_age_problem 
  (A S F : ℕ)
  (h1 : A = 3 * S)
  (h2 : F = 2 * A + 40)
  (h3 : A = 360) :
  A + S + F = 1240 :=
by 
  sorry

end NUMINAMATH_GPT_alchemerion_age_problem_l1781_178151


namespace NUMINAMATH_GPT_find_x_value_l1781_178127

-- Define the conditions and the proof problem as Lean 4 statement
theorem find_x_value 
  (k : ℚ)
  (h1 : ∀ (x y : ℚ), (2 * x - 3) / (2 * y + 10) = k)
  (h2 : (2 * 4 - 3) / (2 * 5 + 10) = k)
  : (∃ x : ℚ, (2 * x - 3) / (2 * 10 + 10) = k) ↔ x = 5.25 :=
by
  sorry

end NUMINAMATH_GPT_find_x_value_l1781_178127


namespace NUMINAMATH_GPT_unique_solution_linear_system_l1781_178149

theorem unique_solution_linear_system
  (a11 a22 a33 : ℝ) (a12 a13 a21 a23 a31 a32 : ℝ) 
  (x1 x2 x3 : ℝ) 
  (h1 : 0 < a11) (h2 : 0 < a22) (h3 : 0 < a33)
  (h4 : a12 < 0) (h5 : a13 < 0) (h6 : a21 < 0) (h7 : a23 < 0) (h8 : a31 < 0) (h9 : a32 < 0)
  (h10 : 0 < a11 + a12 + a13) (h11 : 0 < a21 + a22 + a23) (h12 : 0 < a31 + a32 + a33) :
  (a11 * x1 + a12 * x2 + a13 * x3 = 0) →
  (a21 * x1 + a22 * x2 + a23 * x3 = 0) →
  (a31 * x1 + a32 * x2 + a33 * x3 = 0) →
  x1 = 0 ∧ x2 = 0 ∧ x3 = 0 := by
  sorry

end NUMINAMATH_GPT_unique_solution_linear_system_l1781_178149


namespace NUMINAMATH_GPT_waiter_tables_l1781_178139

theorem waiter_tables (init_customers : ℕ) (left_customers : ℕ) (people_per_table : ℕ) (remaining_customers : ℕ) (num_tables : ℕ) :
  init_customers = 44 →
  left_customers = 12 →
  people_per_table = 8 →
  remaining_customers = init_customers - left_customers →
  num_tables = remaining_customers / people_per_table →
  num_tables = 4 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_waiter_tables_l1781_178139


namespace NUMINAMATH_GPT_art_performance_selection_l1781_178148

-- Definitions from the conditions
def total_students := 6
def singers := 3
def dancers := 2
def both := 1

-- Mathematical expression in Lean
noncomputable def ways_to_select (n k : ℕ) : ℕ := Nat.choose n k

theorem art_performance_selection 
    (total_students singers dancers both: ℕ) 
    (h1 : total_students = 6)
    (h2 : singers = 3)
    (h3 : dancers = 2)
    (h4 : both = 1) :
  (ways_to_select 4 2 * 3 - 1) = (Nat.choose 4 2 * 3 - 1) := 
sorry

end NUMINAMATH_GPT_art_performance_selection_l1781_178148


namespace NUMINAMATH_GPT_fraction_of_bikinis_or_trunks_l1781_178193

theorem fraction_of_bikinis_or_trunks (h_bikinis : Real := 0.38) (h_trunks : Real := 0.25) :
  h_bikinis + h_trunks = 0.63 :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_bikinis_or_trunks_l1781_178193


namespace NUMINAMATH_GPT_percent_area_square_in_rectangle_l1781_178102

theorem percent_area_square_in_rectangle
  (s : ℝ) 
  (w : ℝ) 
  (l : ℝ)
  (h1 : w = 3 * s) 
  (h2 : l = (9 / 2) * s) 
  : (s^2 / (l * w)) * 100 = 7.41 :=
by
  sorry

end NUMINAMATH_GPT_percent_area_square_in_rectangle_l1781_178102


namespace NUMINAMATH_GPT_remove_parentheses_l1781_178159

variable (a b c : ℝ)

theorem remove_parentheses :
  -3 * a - (2 * b - c) = -3 * a - 2 * b + c :=
by
  sorry

end NUMINAMATH_GPT_remove_parentheses_l1781_178159


namespace NUMINAMATH_GPT_sequence_existence_l1781_178157

theorem sequence_existence (n : ℕ) : 
  (∃ (x : ℕ → ℤ), 
    (∀ i j, 1 ≤ i ∧ i ≤ j ∧ j ≤ n ∧ i + j ≤ n ∧ ((x i - x j) % 3 = 0) → (x (i + j) + x i + x j + 1) % 3 = 0)) ↔ (n = 8) := 
by 
  sorry

end NUMINAMATH_GPT_sequence_existence_l1781_178157


namespace NUMINAMATH_GPT_expand_product_l1781_178103

theorem expand_product (x : ℝ) : 2 * (x - 3) * (x + 6) = 2 * x^2 + 6 * x - 36 :=
by sorry

end NUMINAMATH_GPT_expand_product_l1781_178103


namespace NUMINAMATH_GPT_series_value_l1781_178152

noncomputable def sum_series (a b c : ℝ) (h_positivity : 0 < c ∧ 0 < b ∧ 0 < a) (h_order : a > b ∧ b > c) : ℝ :=
∑' n : ℕ, (if h : n > 0 then
             1 / (((n - 1) * c - (n - 2) * b) * (n * c - (n - 1) * a))
           else 
             0)

theorem series_value (a b c : ℝ) (h_positivity : 0 < c ∧ 0 < b ∧ 0 < a) (h_order : a > b ∧ b > c) :
  sum_series a b c h_positivity h_order = 1 / ((c - a) * b) :=
by
  sorry

end NUMINAMATH_GPT_series_value_l1781_178152


namespace NUMINAMATH_GPT_colony_fungi_day_l1781_178186

theorem colony_fungi_day (n : ℕ): 
  (4 * 2^n > 150) = (n = 6) :=
sorry

end NUMINAMATH_GPT_colony_fungi_day_l1781_178186


namespace NUMINAMATH_GPT_subtraction_result_l1781_178176

noncomputable def division_value : ℝ := 1002 / 20.04

theorem subtraction_result : 2500 - division_value = 2450.0499 :=
by
  have division_eq : division_value = 49.9501 := by sorry
  rw [division_eq]
  norm_num

end NUMINAMATH_GPT_subtraction_result_l1781_178176


namespace NUMINAMATH_GPT_number_of_votes_for_winner_l1781_178179

-- Define the conditions
def total_votes : ℝ := 1000
def winner_percentage : ℝ := 0.55
def margin_of_victory : ℝ := 100

-- The statement to prove
theorem number_of_votes_for_winner :
  0.55 * total_votes = 550 :=
by
  -- We are supposed to provide the proof but it's skipped here
  sorry

end NUMINAMATH_GPT_number_of_votes_for_winner_l1781_178179


namespace NUMINAMATH_GPT_cherry_tomatoes_weight_l1781_178171

def kilogram_to_grams (kg : ℕ) : ℕ := kg * 1000

theorem cherry_tomatoes_weight (kg_tomatoes : ℕ) (extra_tomatoes_g : ℕ) : kg_tomatoes = 2 → extra_tomatoes_g = 560 → kilogram_to_grams kg_tomatoes + extra_tomatoes_g = 2560 :=
by
  intros h1 h2
  rw [h1, h2]
  sorry

end NUMINAMATH_GPT_cherry_tomatoes_weight_l1781_178171


namespace NUMINAMATH_GPT_complex_expression_evaluation_l1781_178101

-- Definition of the imaginary unit i with property i^2 = -1
def i : ℂ := Complex.I

-- Theorem stating that the given expression equals i
theorem complex_expression_evaluation : i * (1 - i) - 1 = i := by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_complex_expression_evaluation_l1781_178101


namespace NUMINAMATH_GPT_angle_B_possible_values_l1781_178158

theorem angle_B_possible_values
  (a b : ℝ) (A B : ℝ)
  (h_a : a = 2)
  (h_b : b = 2 * Real.sqrt 3)
  (h_A : A = Real.pi / 6) 
  (h_A_range : (0 : ℝ) < A ∧ A < Real.pi) :
  B = Real.pi / 3 ∨ B = 2 * Real.pi / 3 :=
  sorry

end NUMINAMATH_GPT_angle_B_possible_values_l1781_178158


namespace NUMINAMATH_GPT_three_digit_numbers_divisible_by_5_l1781_178161

theorem three_digit_numbers_divisible_by_5 : ∃ n : ℕ, n = 181 ∧ ∀ x : ℕ, (100 ≤ x ∧ x ≤ 999 ∧ x % 5 = 0) → ∃ k : ℕ, x = 100 + k * 5 ∧ k < n := sorry

end NUMINAMATH_GPT_three_digit_numbers_divisible_by_5_l1781_178161


namespace NUMINAMATH_GPT_find_principal_and_rate_l1781_178156

variables (P R : ℝ)

theorem find_principal_and_rate
  (h1 : 20 = P * R * 2 / 100)
  (h2 : 22 = P * ((1 + R / 100) ^ 2 - 1)) :
  P = 50 ∧ R = 20 :=
by
  sorry

end NUMINAMATH_GPT_find_principal_and_rate_l1781_178156


namespace NUMINAMATH_GPT_example_problem_l1781_178198

theorem example_problem
  (h1 : 0.25 < 1) 
  (h2 : 0.15 < 0.25) : 
  3.04 / 0.25 > 1 :=
by
  sorry

end NUMINAMATH_GPT_example_problem_l1781_178198


namespace NUMINAMATH_GPT_circle_probability_l1781_178174

noncomputable def problem_statement : Prop :=
  let outer_radius := 3
  let inner_radius := 1
  let pivotal_radius := 2
  let outer_area := Real.pi * outer_radius ^ 2
  let inner_area := Real.pi * pivotal_radius ^ 2
  let probability := inner_area / outer_area
  probability = 4 / 9

theorem circle_probability : problem_statement := sorry

end NUMINAMATH_GPT_circle_probability_l1781_178174


namespace NUMINAMATH_GPT_bread_slices_l1781_178116

theorem bread_slices (c : ℕ) (cost_each_slice_in_cents : ℕ)
  (total_paid_in_cents : ℕ) (change_in_cents : ℕ) (n : ℕ) (slices_per_loaf : ℕ) :
  c = 3 →
  cost_each_slice_in_cents = 40 →
  total_paid_in_cents = 2 * 2000 →
  change_in_cents = 1600 →
  total_paid_in_cents - change_in_cents = n * cost_each_slice_in_cents →
  n = c * slices_per_loaf →
  slices_per_loaf = 20 :=
by sorry

end NUMINAMATH_GPT_bread_slices_l1781_178116


namespace NUMINAMATH_GPT_ratio_AB_to_AD_l1781_178167

/-
In rectangle ABCD, 30% of its area overlaps with square EFGH. Square EFGH shares 40% of its area with rectangle ABCD. If AD equals one-tenth of the side length of square EFGH, what is AB/AD?
-/

theorem ratio_AB_to_AD (s x y : ℝ)
  (h1 : 0.3 * (x * y) = 0.4 * s^2)
  (h2 : y = s / 10):
  (x / y) = 400 / 3 :=
by
  sorry

end NUMINAMATH_GPT_ratio_AB_to_AD_l1781_178167


namespace NUMINAMATH_GPT_maximum_value_l1781_178134

theorem maximum_value (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) (h4 : x + y + z = 3) : 
  (x^2 - x * y + y^2) * (x^2 - x * z + z^2) * (y^2 - y * z + z^2) * (z^2 - z * x + x^2)  ≤ 1 :=
sorry

end NUMINAMATH_GPT_maximum_value_l1781_178134


namespace NUMINAMATH_GPT_count_integers_l1781_178106

theorem count_integers (n : ℤ) (h : -11 ≤ n ∧ n ≤ 11) : ∃ (s : Finset ℤ), s.card = 7 ∧ ∀ x ∈ s, (x - 1) * (x + 3) * (x + 7) < 0 :=
by
  sorry

end NUMINAMATH_GPT_count_integers_l1781_178106


namespace NUMINAMATH_GPT_prec_property_l1781_178125

noncomputable def prec (a b : ℕ) : Prop :=
  sorry -- The construction of the relation from the problem

axiom prec_total : ∀ a b : ℕ, (prec a b ∨ prec b a ∨ a = b)
axiom prec_trans : ∀ a b c : ℕ, (prec a b ∧ prec b c) → prec a c

theorem prec_property : ∀ a b c : ℕ, (prec a b ∧ prec b c) → 2 * b ≠ a + c :=
by
  sorry

end NUMINAMATH_GPT_prec_property_l1781_178125


namespace NUMINAMATH_GPT_angle_D_measure_l1781_178160

theorem angle_D_measure (B C E F D : ℝ) 
  (h₁ : B = 120)
  (h₂ : B + C = 180)
  (h₃ : E = 45)
  (h₄ : F = C) 
  (h₅ : D + E + F = 180) :
  D = 75 := sorry

end NUMINAMATH_GPT_angle_D_measure_l1781_178160


namespace NUMINAMATH_GPT_decimal_equivalent_of_one_quarter_l1781_178170

theorem decimal_equivalent_of_one_quarter:
  ( (1:ℚ) / (4:ℚ) )^1 = 0.25 := 
sorry

end NUMINAMATH_GPT_decimal_equivalent_of_one_quarter_l1781_178170


namespace NUMINAMATH_GPT_repeating_decimal_arithmetic_l1781_178182

def x : ℚ := 0.234 -- repeating decimal 0.234
def y : ℚ := 0.567 -- repeating decimal 0.567
def z : ℚ := 0.891 -- repeating decimal 0.891

theorem repeating_decimal_arithmetic :
  x - y + z = 186 / 333 := 
sorry

end NUMINAMATH_GPT_repeating_decimal_arithmetic_l1781_178182


namespace NUMINAMATH_GPT_number_halfway_between_l1781_178197

theorem number_halfway_between :
  ∃ x : ℚ, x = (1/12 + 1/14) / 2 ∧ x = 13 / 168 :=
sorry

end NUMINAMATH_GPT_number_halfway_between_l1781_178197


namespace NUMINAMATH_GPT_rectangle_dimensions_l1781_178115

theorem rectangle_dimensions (w l : ℕ) (h1 : l = 2 * w) (h2 : 2 * (w * l) = 2 * (2 * w + w)) :
  w = 6 ∧ l = 12 := 
by sorry

end NUMINAMATH_GPT_rectangle_dimensions_l1781_178115


namespace NUMINAMATH_GPT_total_depreciation_correct_residual_value_correct_sales_price_correct_l1781_178191

-- Definitions and conditions
def initial_cost := 500000
def max_capacity := 100000
def jul_bottles := 200
def aug_bottles := 15000
def sep_bottles := 12300

def depreciation_per_bottle := initial_cost / max_capacity

-- Part (a)
def total_depreciation_jul := jul_bottles * depreciation_per_bottle
def total_depreciation_aug := aug_bottles * depreciation_per_bottle
def total_depreciation_sep := sep_bottles * depreciation_per_bottle
def total_depreciation := total_depreciation_jul + total_depreciation_aug + total_depreciation_sep

theorem total_depreciation_correct :
  total_depreciation = 137500 := 
by sorry

-- Part (b)
def residual_value := initial_cost - total_depreciation

theorem residual_value_correct :
  residual_value = 362500 := 
by sorry

-- Part (c)
def desired_profit := 10000
def sales_price := residual_value + desired_profit

theorem sales_price_correct :
  sales_price = 372500 := 
by sorry

end NUMINAMATH_GPT_total_depreciation_correct_residual_value_correct_sales_price_correct_l1781_178191


namespace NUMINAMATH_GPT_cubic_eq_one_real_root_l1781_178121

/-- The equation x^3 - 4x^2 + 9x + c = 0 has exactly one real root for any real number c. -/
theorem cubic_eq_one_real_root (c : ℝ) : 
  ∃! x : ℝ, x^3 - 4 * x^2 + 9 * x + c = 0 :=
sorry

end NUMINAMATH_GPT_cubic_eq_one_real_root_l1781_178121


namespace NUMINAMATH_GPT_combined_weight_l1781_178196

theorem combined_weight (x y z : ℕ) (h1 : x + z = 78) (h2 : x + y = 69) (h3 : y + z = 137) : x + y + z = 142 :=
by
  -- Intermediate steps or any additional lemmas could go here
sorry

end NUMINAMATH_GPT_combined_weight_l1781_178196


namespace NUMINAMATH_GPT_eq_x4_inv_x4_l1781_178118

theorem eq_x4_inv_x4 (x : ℝ) (h : x^2 + (1 / x^2) = 2) : 
  x^4 + (1 / x^4) = 2 := 
by 
  sorry

end NUMINAMATH_GPT_eq_x4_inv_x4_l1781_178118


namespace NUMINAMATH_GPT_diff_reading_math_homework_l1781_178108

-- Define the conditions as given in the problem
def pages_math_homework : ℕ := 3
def pages_reading_homework : ℕ := 4

-- The statement to prove that Rachel had 1 more page of reading homework than math homework
theorem diff_reading_math_homework : pages_reading_homework - pages_math_homework = 1 := by
  sorry

end NUMINAMATH_GPT_diff_reading_math_homework_l1781_178108


namespace NUMINAMATH_GPT_problem_A_inter_complement_B_l1781_178114

noncomputable def A : Set ℝ := {x : ℝ | Real.log x / Real.log 2 < 2}
noncomputable def B : Set ℝ := {x : ℝ | (x - 2) / (x - 1) ≥ 0}
noncomputable def complement_B : Set ℝ := {x : ℝ | ¬((x - 2) / (x - 1) ≥ 0)}

theorem problem_A_inter_complement_B : 
  (A ∩ complement_B) = {x : ℝ | 1 < x ∧ x < 2} := by
  sorry

end NUMINAMATH_GPT_problem_A_inter_complement_B_l1781_178114


namespace NUMINAMATH_GPT_child_ticket_cost_l1781_178166

def cost_of_adult_ticket : ℕ := 22
def number_of_adults : ℕ := 2
def number_of_children : ℕ := 2
def total_family_cost : ℕ := 58
def cost_of_child_ticket : ℕ := 7

theorem child_ticket_cost :
  2 * cost_of_adult_ticket + number_of_children * cost_of_child_ticket = total_family_cost :=
by
  sorry

end NUMINAMATH_GPT_child_ticket_cost_l1781_178166


namespace NUMINAMATH_GPT_hours_buses_leave_each_day_l1781_178144

theorem hours_buses_leave_each_day
  (num_buses : ℕ)
  (num_days : ℕ)
  (buses_per_half_hour : ℕ)
  (h1 : num_buses = 120)
  (h2 : num_days = 5)
  (h3 : buses_per_half_hour = 2) :
  (num_buses / num_days) / buses_per_half_hour = 12 :=
by
  sorry

end NUMINAMATH_GPT_hours_buses_leave_each_day_l1781_178144


namespace NUMINAMATH_GPT_condition_a_gt_1_iff_a_gt_0_l1781_178135

theorem condition_a_gt_1_iff_a_gt_0 : ∀ (a : ℝ), (a > 1) ↔ (a > 0) :=
by 
  sorry

end NUMINAMATH_GPT_condition_a_gt_1_iff_a_gt_0_l1781_178135


namespace NUMINAMATH_GPT_constant_term_proof_l1781_178172

noncomputable def constant_term_in_binomial_expansion (c : ℚ) (x : ℚ) : ℚ :=
  if h : (c = (2 : ℚ) - (1 / (8 * x^3))∧ x ≠ 0) then 
    28
  else 
    0

theorem constant_term_proof : 
  constant_term_in_binomial_expansion ((2 : ℚ) - (1 / (8 * (1 : ℚ)^3))) 1 = 28 := 
by
  sorry

end NUMINAMATH_GPT_constant_term_proof_l1781_178172


namespace NUMINAMATH_GPT_max_principals_l1781_178110

theorem max_principals (n_years term_length max_principals: ℕ) 
  (h1 : n_years = 12) 
  (h2 : term_length = 4)
  (h3 : max_principals = 4): 
  (∃ p : ℕ, p = max_principals) :=
by
  sorry

end NUMINAMATH_GPT_max_principals_l1781_178110


namespace NUMINAMATH_GPT_point_on_angle_bisector_l1781_178164

theorem point_on_angle_bisector (a b : ℝ) (h : ∃ p : ℝ, (a, b) = (p, -p)) : a = -b :=
sorry

end NUMINAMATH_GPT_point_on_angle_bisector_l1781_178164


namespace NUMINAMATH_GPT_find_x_l1781_178184

theorem find_x (x : ℝ) :
  (x * 13.26 + x * 9.43 + x * 77.31 = 470) → (x = 4.7) :=
by
  sorry

end NUMINAMATH_GPT_find_x_l1781_178184


namespace NUMINAMATH_GPT_problem1_problem2_l1781_178104

theorem problem1 :
  Real.sqrt 27 - (Real.sqrt 2 * Real.sqrt 6) + 3 * Real.sqrt (1/3) = 2 * Real.sqrt 3 := 
  by sorry

theorem problem2 :
  (Real.sqrt 5 + Real.sqrt 2) * (Real.sqrt 5 - Real.sqrt 2) + (Real.sqrt 3 - 1)^2 = 7 - 2 * Real.sqrt 3 := 
  by sorry

end NUMINAMATH_GPT_problem1_problem2_l1781_178104


namespace NUMINAMATH_GPT_train_time_to_B_l1781_178163

theorem train_time_to_B (T : ℝ) (M : ℝ) :
  (∃ (D : ℝ), (T + 5) * (D + M) / T = 6 * M ∧ 2 * D = 5 * M) → T = 7 :=
by
  sorry

end NUMINAMATH_GPT_train_time_to_B_l1781_178163


namespace NUMINAMATH_GPT_coefficients_balance_l1781_178141

noncomputable def num_positive_coeffs (n : ℕ) : ℕ :=
  n + 1

noncomputable def num_negative_coeffs (n : ℕ) : ℕ :=
  n + 1

theorem coefficients_balance (n : ℕ) (h_odd: Odd n) (x : ℝ) :
  num_positive_coeffs n = num_negative_coeffs n :=
by
  sorry

end NUMINAMATH_GPT_coefficients_balance_l1781_178141


namespace NUMINAMATH_GPT_selling_price_of_article_l1781_178187

theorem selling_price_of_article (cost_price : ℕ) (gain_percent : ℕ) (profit : ℕ) (selling_price : ℕ) : 
  cost_price = 100 → gain_percent = 10 → profit = (gain_percent * cost_price) / 100 → selling_price = cost_price + profit → selling_price = 110 :=
by
  intros
  sorry

end NUMINAMATH_GPT_selling_price_of_article_l1781_178187


namespace NUMINAMATH_GPT_matrix_characteristic_eq_l1781_178113

noncomputable def B : Matrix (Fin 3) (Fin 3) ℚ :=
  ![![1, 2, 2], ![2, 1, 2], ![2, 2, 1]]

theorem matrix_characteristic_eq :
  ∃ (a b c : ℚ), a = -6 ∧ b = -12 ∧ c = -18 ∧ 
  (B ^ 3 + a • (B ^ 2) + b • B + c • (1 : Matrix (Fin 3) (Fin 3) ℚ) = 0) :=
by
  sorry

end NUMINAMATH_GPT_matrix_characteristic_eq_l1781_178113


namespace NUMINAMATH_GPT_total_number_of_players_l1781_178120

theorem total_number_of_players (n : ℕ) (h1 : n > 7) 
  (h2 : (4 * (n * (n - 1)) / 3 + 56 = (n + 8) * (n + 7) / 2)) : n + 8 = 50 :=
by
  sorry

end NUMINAMATH_GPT_total_number_of_players_l1781_178120


namespace NUMINAMATH_GPT_sufficient_condition_not_necessary_condition_l1781_178190

theorem sufficient_condition (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) : ab > 0 := by
  sorry

theorem not_necessary_condition (a b : ℝ) : ¬(a > 0 ∧ b > 0) → ab > 0 := by
  sorry

end NUMINAMATH_GPT_sufficient_condition_not_necessary_condition_l1781_178190


namespace NUMINAMATH_GPT_committee_probability_l1781_178175

theorem committee_probability :
  let total_committees := Nat.choose 30 6
  let boys_choose := Nat.choose 18 3
  let girls_choose := Nat.choose 12 3
  let specific_committees := boys_choose * girls_choose
  specific_committees / total_committees = 64 / 211 := 
by
  let total_committees := Nat.choose 30 6
  let boys_choose := Nat.choose 18 3
  let girls_choose := Nat.choose 12 3
  let specific_committees := boys_choose * girls_choose
  have h_total_committees : total_committees = 593775 := by sorry
  have h_boys_choose : boys_choose = 816 := by sorry
  have h_girls_choose : girls_choose = 220 := by sorry
  have h_specific_committees : specific_committees = 179520 := by sorry
  have h_probability : specific_committees / total_committees = 64 / 211 := by sorry
  exact h_probability

end NUMINAMATH_GPT_committee_probability_l1781_178175


namespace NUMINAMATH_GPT_range_of_a_for_real_roots_l1781_178147

theorem range_of_a_for_real_roots (a : ℝ) (h : a ≠ 0) :
  (∃ (x : ℝ), a*x^2 + 2*x + 1 = 0) ↔ a ≤ 1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_for_real_roots_l1781_178147


namespace NUMINAMATH_GPT_find_k_l1781_178126

def triangle_sides (a b c : ℕ) : Prop :=
a < b + c ∧ b < a + c ∧ c < a + b

def is_right_triangle (a b c : ℕ) : Prop :=
a * a + b * b = c * c

def angle_bisector_length (a b c l : ℕ) : Prop :=
∃ k : ℚ, l = k * Real.sqrt 2 ∧ k = 5 / 2

theorem find_k :
  ∀ (AB BC AC BD : ℕ),
  triangle_sides AB BC AC ∧ is_right_triangle AB BC AC ∧
  AB = 5 ∧ BC = 12 ∧ AC = 13 ∧ angle_bisector_length 5 12 13 BD →
  ∃ k : ℚ, BD = k * Real.sqrt 2 ∧ k = 5 / 2 := by
  sorry

end NUMINAMATH_GPT_find_k_l1781_178126


namespace NUMINAMATH_GPT_anderson_family_seating_l1781_178122

def anderson_family_seating_arrangements : Prop :=
  ∃ (family : Fin 5 → String),
    (family 0 = "Mr. Anderson" ∨ family 0 = "Mrs. Anderson") ∧
    (∀ (i : Fin 5), i ≠ 0 → family i ≠ family 0) ∧
    family 1 ≠ family 0 ∧ (family 1 = "Mrs. Anderson" ∨ family 1 = "Child 1" ∨ family 1 = "Child 2") ∧
    family 2 = "Child 3" ∧
    (family 3 ≠ family 0 ∧ family 3 ≠ family 1 ∧ family 3 ≠ family 2) ∧
    (family 4 ≠ family 0 ∧ family 4 ≠ family 1 ∧ family 4 ≠ family 2 ∧ family 4 ≠ family 3) ∧
    (family 3 = "Child 1" ∨ family 3 = "Child 2") ∧
    (family 4 = "Child 1" ∨ family 4 = "Child 2") ∧
    family 3 ≠ family 4 → 
    (2 * 3 * 2 = 12)

theorem anderson_family_seating : anderson_family_seating_arrangements := 
  sorry

end NUMINAMATH_GPT_anderson_family_seating_l1781_178122


namespace NUMINAMATH_GPT_football_team_total_players_l1781_178112

/-- The conditions are:
1. There are some players on a football team.
2. 46 are throwers.
3. All throwers are right-handed.
4. One third of the rest of the team are left-handed.
5. There are 62 right-handed players in total.
And we need to prove that the total number of players on the football team is 70. 
--/

theorem football_team_total_players (P : ℕ) 
  (h_throwers : P >= 46) 
  (h_total_right_handed : 62 = 46 + 2 * (P - 46) / 3)
  (h_remainder_left_handed : 1 * (P - 46) / 3 = (P - 46) / 3) :
  P = 70 :=
by
  sorry

end NUMINAMATH_GPT_football_team_total_players_l1781_178112


namespace NUMINAMATH_GPT_find_inverse_sum_l1781_178153

theorem find_inverse_sum (a b : ℝ) (h1 : a ≠ b) (h2 : a / b + a = b / a + b) : 1 / a + 1 / b = -1 :=
sorry

end NUMINAMATH_GPT_find_inverse_sum_l1781_178153


namespace NUMINAMATH_GPT_employees_after_reduction_l1781_178132

def reduction (original : Float) (percent : Float) : Float :=
  original - (percent * original)

theorem employees_after_reduction :
  reduction 243.75 0.20 = 195 := by
  sorry

end NUMINAMATH_GPT_employees_after_reduction_l1781_178132


namespace NUMINAMATH_GPT_mickey_horses_per_week_l1781_178195

def days_in_week : ℕ := 7

def horses_minnie_per_day : ℕ := days_in_week + 3

def horses_twice_minnie_per_day : ℕ := 2 * horses_minnie_per_day

def horses_mickey_per_day : ℕ := horses_twice_minnie_per_day - 6

def horses_mickey_per_week : ℕ := days_in_week * horses_mickey_per_day

theorem mickey_horses_per_week : horses_mickey_per_week = 98 := sorry

end NUMINAMATH_GPT_mickey_horses_per_week_l1781_178195


namespace NUMINAMATH_GPT_probability_of_bayonet_base_on_third_try_is_7_over_120_l1781_178130

noncomputable def probability_picking_bayonet_base_bulb_on_third_try : ℚ :=
  (3 / 10) * (2 / 9) * (7 / 8)

/-- Given a box containing 3 screw base bulbs and 7 bayonet base bulbs, all with the
same shape and power and placed with their bases down. An electrician takes one bulb
at a time without returning it. The probability that he gets a bayonet base bulb on his
third try is 7/120. -/
theorem probability_of_bayonet_base_on_third_try_is_7_over_120 :
  probability_picking_bayonet_base_bulb_on_third_try = 7 / 120 :=
by 
  sorry

end NUMINAMATH_GPT_probability_of_bayonet_base_on_third_try_is_7_over_120_l1781_178130


namespace NUMINAMATH_GPT_harry_sandy_meet_point_l1781_178109

theorem harry_sandy_meet_point :
  let H : ℝ × ℝ := (10, -3)
  let S : ℝ × ℝ := (2, 7)
  let t : ℝ := 2 / 3
  let meet_point : ℝ × ℝ := (H.1 + t * (S.1 - H.1), H.2 + t * (S.2 - H.2))
  meet_point = (14 / 3, 11 / 3) := 
by
  sorry

end NUMINAMATH_GPT_harry_sandy_meet_point_l1781_178109


namespace NUMINAMATH_GPT_y_pow_expression_l1781_178183

theorem y_pow_expression (y : ℝ) (h : y + 1/y = 3) : y^13 - 5 * y^9 + y^5 = 0 :=
sorry

end NUMINAMATH_GPT_y_pow_expression_l1781_178183


namespace NUMINAMATH_GPT_time_after_1876_minutes_l1781_178117

-- Define the structure for Time
structure Time where
  hour : Nat
  minute : Nat

-- Define a function to add minutes to a time
noncomputable def add_minutes (t : Time) (m : Nat) : Time :=
  let total_minutes := t.minute + m
  let additional_hours := total_minutes / 60
  let remaining_minutes := total_minutes % 60
  let new_hour := (t.hour + additional_hours) % 24
  { hour := new_hour, minute := remaining_minutes }

-- Definition of the starting time
def three_pm : Time := { hour := 15, minute := 0 }

-- The main theorem statement
theorem time_after_1876_minutes : add_minutes three_pm 1876 = { hour := 10, minute := 16 } :=
  sorry

end NUMINAMATH_GPT_time_after_1876_minutes_l1781_178117


namespace NUMINAMATH_GPT_A_neg10_3_eq_neg1320_l1781_178133

noncomputable def A (x : ℝ) (m : ℕ) : ℝ :=
  if m = 0 then 1 else x * A (x - 1) (m - 1)

theorem A_neg10_3_eq_neg1320 : A (-10) 3 = -1320 := 
by
  sorry

end NUMINAMATH_GPT_A_neg10_3_eq_neg1320_l1781_178133


namespace NUMINAMATH_GPT_q_true_given_not_p_and_p_or_q_l1781_178181

theorem q_true_given_not_p_and_p_or_q (p q : Prop) (hnp : ¬p) (hpq : p ∨ q) : q :=
by
  sorry

end NUMINAMATH_GPT_q_true_given_not_p_and_p_or_q_l1781_178181


namespace NUMINAMATH_GPT_range_of_a_for_two_unequal_roots_l1781_178100

theorem range_of_a_for_two_unequal_roots (a : ℝ) : 
  (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ a * Real.log x₁ = x₁ ∧ a * Real.log x₂ = x₂) ↔ a > Real.exp 1 :=
sorry

end NUMINAMATH_GPT_range_of_a_for_two_unequal_roots_l1781_178100


namespace NUMINAMATH_GPT_determine_c_l1781_178188

noncomputable def ab5c_decimal (a b c : ℕ) : ℕ :=
  729 * a + 81 * b + 45 + c

theorem determine_c (a b c : ℕ) (h₁ : a ≠ 0) (h₂ : ∃ k : ℕ, ab5c_decimal a b c = k^2) :
  c = 0 ∨ c = 7 :=
by
  sorry

end NUMINAMATH_GPT_determine_c_l1781_178188


namespace NUMINAMATH_GPT_factoring_expression_l1781_178168

theorem factoring_expression (a b c x y : ℝ) :
  -a * (x - y) - b * (y - x) + c * (x - y) = -(x - y) * (a + b - c) :=
by
  sorry

end NUMINAMATH_GPT_factoring_expression_l1781_178168


namespace NUMINAMATH_GPT_result_when_decreased_by_5_and_divided_by_7_l1781_178129

theorem result_when_decreased_by_5_and_divided_by_7 (x y : ℤ)
  (h1 : (x - 5) / 7 = y)
  (h2 : (x - 6) / 8 = 6) :
  y = 7 :=
by
  sorry

end NUMINAMATH_GPT_result_when_decreased_by_5_and_divided_by_7_l1781_178129


namespace NUMINAMATH_GPT_correct_operation_l1781_178124

theorem correct_operation {a : ℝ} : (a ^ 6 / a ^ 2 = a ^ 4) :=
by sorry

end NUMINAMATH_GPT_correct_operation_l1781_178124


namespace NUMINAMATH_GPT_least_positive_t_l1781_178154

theorem least_positive_t (t : ℕ) (α : ℝ) (h1 : 0 < α ∧ α < π / 2)
  (h2 : π / 10 < α ∧ α ≤ π / 6) 
  (h3 : (3 * α)^2 = α * (π - 5 * α)) :
  t = 27 :=
by
  have hα : α = π / 14 := 
    by
      sorry
  sorry

end NUMINAMATH_GPT_least_positive_t_l1781_178154


namespace NUMINAMATH_GPT_value_of_a_l1781_178178

def P : Set ℝ := { x | x^2 ≤ 4 }
def M (a : ℝ) : Set ℝ := { a }

theorem value_of_a (a : ℝ) (h : P ∪ {a} = P) : a ∈ { x : ℝ | -2 ≤ x ∧ x ≤ 2 } := by
  sorry

end NUMINAMATH_GPT_value_of_a_l1781_178178


namespace NUMINAMATH_GPT_find_r_l1781_178194

theorem find_r (r s : ℝ)
  (h1 : ∀ α β : ℝ, (α + β = -r) ∧ (α * β = s) → 
         ∃ t : ℝ, (t^2 - (α^2 + β^2) * t + (α^2 * β^2) = 0) ∧ |α^2 - β^2| = 8)
  (h_sum : ∃ α β : ℝ, α + β = 10) :
  r = -10 := by
  sorry

end NUMINAMATH_GPT_find_r_l1781_178194


namespace NUMINAMATH_GPT_abs_diff_51st_terms_correct_l1781_178137

-- Definition of initial conditions for sequences A and C
def seqA_first_term : ℤ := 40
def seqA_common_difference : ℤ := 8

def seqC_first_term : ℤ := 40
def seqC_common_difference : ℤ := -5

-- Definition of the nth term function for an arithmetic sequence
def nth_term (a₁ d n : ℤ) : ℤ := a₁ + d * (n - 1)

-- 51st term of sequence A
def a_51 : ℤ := nth_term seqA_first_term seqA_common_difference 51

-- 51st term of sequence C
def c_51 : ℤ := nth_term seqC_first_term seqC_common_difference 51

-- Absolute value of the difference
def abs_diff_51st_terms : ℤ := Int.natAbs (a_51 - c_51)

-- The theorem to be proved
theorem abs_diff_51st_terms_correct : abs_diff_51st_terms = 650 := by
  sorry

end NUMINAMATH_GPT_abs_diff_51st_terms_correct_l1781_178137
