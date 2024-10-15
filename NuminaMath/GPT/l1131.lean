import Mathlib

namespace NUMINAMATH_GPT_minimum_value_l1131_113164

noncomputable def min_expression (a b : ℝ) : ℝ :=
  a^2 + b^2 + 1 / (a + b)^2 + 1 / (a^2 * b^2)

theorem minimum_value (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) : 
  ∃ (c : ℝ), c = 2 * Real.sqrt 2 + 3 ∧ min_expression a b ≥ c :=
by
  use 2 * Real.sqrt 2 + 3
  sorry

end NUMINAMATH_GPT_minimum_value_l1131_113164


namespace NUMINAMATH_GPT_jake_fewer_peaches_l1131_113135

theorem jake_fewer_peaches (steven_peaches : ℕ) (jake_peaches : ℕ) (h1 : steven_peaches = 19) (h2 : jake_peaches = 7) : steven_peaches - jake_peaches = 12 :=
sorry

end NUMINAMATH_GPT_jake_fewer_peaches_l1131_113135


namespace NUMINAMATH_GPT_select_team_with_smaller_variance_l1131_113133

theorem select_team_with_smaller_variance 
    (variance_A variance_B : ℝ)
    (hA : variance_A = 1.5)
    (hB : variance_B = 2.8)
    : variance_A < variance_B → "Team A" = "Team A" :=
by
  intros h
  sorry

end NUMINAMATH_GPT_select_team_with_smaller_variance_l1131_113133


namespace NUMINAMATH_GPT_profit_percent_is_25_l1131_113127

-- Define the cost price (CP) and selling price (SP) based on the given ratio.
def CP (x : ℝ) := 4 * x
def SP (x : ℝ) := 5 * x

-- Calculate the profit percent based on the given conditions.
noncomputable def profitPercent (x : ℝ) := ((SP x - CP x) / CP x) * 100

-- Prove that the profit percent is 25% given the ratio of CP to SP is 4:5.
theorem profit_percent_is_25 (x : ℝ) : profitPercent x = 25 := by
  sorry

end NUMINAMATH_GPT_profit_percent_is_25_l1131_113127


namespace NUMINAMATH_GPT_football_defense_stats_l1131_113174

/-- Given:
1. Team 1 has an average of 1.5 goals conceded per match.
2. Team 1 has a standard deviation of 1.1 for the total number of goals conceded throughout the year.
3. Team 2 has an average of 2.1 goals conceded per match.
4. Team 2 has a standard deviation of 0.4 for the total number of goals conceded throughout the year.

Prove:
There are exactly 3 correct statements out of the 4 listed statements. -/
theorem football_defense_stats
  (avg_goals_team1 : ℝ := 1.5)
  (std_dev_team1 : ℝ := 1.1)
  (avg_goals_team2 : ℝ := 2.1)
  (std_dev_team2 : ℝ := 0.4) :
  ∃ correct_statements : ℕ, correct_statements = 3 := 
by
  sorry

end NUMINAMATH_GPT_football_defense_stats_l1131_113174


namespace NUMINAMATH_GPT_rate_for_gravelling_roads_l1131_113114

variable (length breadth width cost : ℕ)
variable (rate per_square_meter : ℕ)

def total_area_parallel_length : ℕ := length * width
def total_area_parallel_breadth : ℕ := (breadth * width) - (width * width)
def total_area : ℕ := total_area_parallel_length length width + total_area_parallel_breadth breadth width

def rate_per_square_meter := cost / total_area length breadth width

theorem rate_for_gravelling_roads :
  (length = 70) →
  (breadth = 30) →
  (width = 5) →
  (cost = 1900) →
  rate_per_square_meter length breadth width cost = 4 := by
  intros; exact sorry

end NUMINAMATH_GPT_rate_for_gravelling_roads_l1131_113114


namespace NUMINAMATH_GPT_milk_water_ratio_l1131_113102

theorem milk_water_ratio
  (vessel1_milk_ratio : ℚ)
  (vessel1_water_ratio : ℚ)
  (vessel2_milk_ratio : ℚ)
  (vessel2_water_ratio : ℚ)
  (equal_mixture_units  : ℚ)
  (h1 : vessel1_milk_ratio / vessel1_water_ratio = 4 / 1)
  (h2 : vessel2_milk_ratio / vessel2_water_ratio = 7 / 3)
  :
  (vessel1_milk_ratio + vessel2_milk_ratio) / 
  (vessel1_water_ratio + vessel2_water_ratio) = 11 / 4 :=
by
  sorry

end NUMINAMATH_GPT_milk_water_ratio_l1131_113102


namespace NUMINAMATH_GPT_triangle_angles_inequality_l1131_113181

theorem triangle_angles_inequality (A B C : ℝ) (h1 : A + B + C = Real.pi) (h2 : 0 < A) (h3 : 0 < B) (h4 : 0 < C) 
(h5 : A < Real.pi) (h6 : B < Real.pi) (h7 : C < Real.pi) : 
  A * Real.cos B + Real.sin A * Real.sin C > 0 := 
by 
  sorry

end NUMINAMATH_GPT_triangle_angles_inequality_l1131_113181


namespace NUMINAMATH_GPT_time_to_paint_remaining_rooms_l1131_113165

-- Definitions for the conditions
def total_rooms : ℕ := 11
def time_per_room : ℕ := 7
def painted_rooms : ℕ := 2

-- Statement of the problem
theorem time_to_paint_remaining_rooms : 
  total_rooms - painted_rooms = 9 →
  (total_rooms - painted_rooms) * time_per_room = 63 := 
by 
  intros h1
  sorry

end NUMINAMATH_GPT_time_to_paint_remaining_rooms_l1131_113165


namespace NUMINAMATH_GPT_tyler_bought_10_erasers_l1131_113107

/--
Given that Tyler initially has $100, buys 8 scissors for $5 each, buys some erasers for $4 each,
and has $20 remaining after these purchases, prove that he bought 10 erasers.
-/
theorem tyler_bought_10_erasers : ∀ (initial_money scissors_cost erasers_cost remaining_money : ℕ), 
  initial_money = 100 →
  scissors_cost = 5 →
  erasers_cost = 4 →
  remaining_money = 20 →
  ∃ (scissors_count erasers_count : ℕ),
    scissors_count = 8 ∧ 
    initial_money - scissors_count * scissors_cost - erasers_count * erasers_cost = remaining_money ∧ 
    erasers_count = 10 :=
by
  intros
  sorry

end NUMINAMATH_GPT_tyler_bought_10_erasers_l1131_113107


namespace NUMINAMATH_GPT_possible_k_values_l1131_113142

variables (p q r s k : ℂ) (h1 : p ≠ 0) (h2 : q ≠ 0) (h3 : r ≠ 0) (h4 : s ≠ 0)
          (h5 : p * q = r * s)
          (h6 : p * k ^ 3 + q * k ^ 2 + r * k + s = 0)
          (h7 : q * k ^ 3 + r * k ^ 2 + s * k + p = 0)

noncomputable def roots_of_unity := {k : ℂ | k ^ 4 = 1}

theorem possible_k_values : k ∈ roots_of_unity :=
by {
  sorry
}

end NUMINAMATH_GPT_possible_k_values_l1131_113142


namespace NUMINAMATH_GPT_abs_expression_eq_6500_l1131_113176

def given_expression (x : ℝ) : ℝ := 
  abs (abs x - x - abs x + 500) - x

theorem abs_expression_eq_6500 (x : ℝ) (h : x = -3000) : given_expression x = 6500 := by
  sorry

end NUMINAMATH_GPT_abs_expression_eq_6500_l1131_113176


namespace NUMINAMATH_GPT_solve_for_y_l1131_113178

theorem solve_for_y (x y : ℝ) (h1 : x * y = 1) (h2 : x / y = 36) (h3 : 0 < x) (h4 : 0 < y) : 
  y = 1 / 6 := 
sorry

end NUMINAMATH_GPT_solve_for_y_l1131_113178


namespace NUMINAMATH_GPT_monthly_revenue_l1131_113197

variable (R : ℝ) -- The monthly revenue

-- Conditions
def after_taxes (R : ℝ) : ℝ := R * 0.90
def after_marketing (R : ℝ) : ℝ := (after_taxes R) * 0.95
def after_operational_costs (R : ℝ) : ℝ := (after_marketing R) * 0.80
def total_employee_wages (R : ℝ) : ℝ := (after_operational_costs R) * 0.15

-- Number of employees and their wages
def number_of_employees : ℝ := 10
def wage_per_employee : ℝ := 4104
def total_wages : ℝ := number_of_employees * wage_per_employee

-- Proof problem
theorem monthly_revenue : R = 400000 ↔ total_employee_wages R = total_wages := by
  sorry

end NUMINAMATH_GPT_monthly_revenue_l1131_113197


namespace NUMINAMATH_GPT_pair_with_gcf_20_l1131_113132

theorem pair_with_gcf_20 (a b : ℕ) (h1 : a = 20) (h2 : b = 40) : Nat.gcd a b = 20 := by
  rw [h1, h2]
  sorry

end NUMINAMATH_GPT_pair_with_gcf_20_l1131_113132


namespace NUMINAMATH_GPT_solve_for_x_l1131_113111

theorem solve_for_x : ∀ (x : ℝ), (4 * x^2 - 3 * x + 2) / (x + 2) = 4 * x - 5 → x = 2 :=
by
  intros x h
  sorry

end NUMINAMATH_GPT_solve_for_x_l1131_113111


namespace NUMINAMATH_GPT_damage_in_dollars_l1131_113116

noncomputable def euros_to_dollars (euros : ℝ) : ℝ := euros * (1 / 0.9)

theorem damage_in_dollars :
  euros_to_dollars 45000000 = 49995000 :=
by
  -- This is where the proof would go
  sorry

end NUMINAMATH_GPT_damage_in_dollars_l1131_113116


namespace NUMINAMATH_GPT_TimTotalRunHoursPerWeek_l1131_113187

def TimUsedToRunTimesPerWeek : ℕ := 3
def TimAddedExtraDaysPerWeek : ℕ := 2
def MorningRunHours : ℕ := 1
def EveningRunHours : ℕ := 1

theorem TimTotalRunHoursPerWeek :
  (TimUsedToRunTimesPerWeek + TimAddedExtraDaysPerWeek) * (MorningRunHours + EveningRunHours) = 10 :=
by
  sorry

end NUMINAMATH_GPT_TimTotalRunHoursPerWeek_l1131_113187


namespace NUMINAMATH_GPT_soldiers_count_l1131_113138

-- Statements of conditions and proofs
theorem soldiers_count (n : ℕ) (s : ℕ) :
  (n * n + 30 = s) →
  ((n + 1) * (n + 1) - 50 = s) →
  s = 1975 :=
by
  intros h1 h2
  -- We know from h1 and h2 that there should be a unique solution for s and n that satisfies both
  -- conditions. Our goal is to show that s must be 1975.

  -- Initialize the proof structure
  sorry

end NUMINAMATH_GPT_soldiers_count_l1131_113138


namespace NUMINAMATH_GPT_mike_total_hours_l1131_113156

-- Define the number of hours Mike worked each day.
def hours_per_day : ℕ := 3

-- Define the number of days Mike worked.
def days : ℕ := 5

-- Define the total number of hours Mike worked.
def total_hours : ℕ := hours_per_day * days

-- State and prove that the total hours Mike worked is 15.
theorem mike_total_hours : total_hours = 15 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_mike_total_hours_l1131_113156


namespace NUMINAMATH_GPT_find_f_of_13_l1131_113131

def f : ℤ → ℤ := sorry  -- We define f as a function from integers to integers

theorem find_f_of_13 : 
  (∀ x, f (-x) = -f x) ∧ 
  (∀ x k : ℤ, f (x + 4 * k) = f x) ∧ 
  (f (-1) = 2) → 
  f 13 = -2 := 
by 
  sorry

end NUMINAMATH_GPT_find_f_of_13_l1131_113131


namespace NUMINAMATH_GPT_particular_solutions_of_diff_eq_l1131_113162

variable {x y : ℝ}

theorem particular_solutions_of_diff_eq
  (h₁ : ∀ C : ℝ, x^2 = C * (y - C))
  (h₂ : x > 0) :
  (y = 2 * x ∨ y = -2 * x) ↔ (x * (y')^2 - 2 * y * y' + 4 * x = 0) := 
sorry

end NUMINAMATH_GPT_particular_solutions_of_diff_eq_l1131_113162


namespace NUMINAMATH_GPT_smallest_integer_solution_l1131_113150

theorem smallest_integer_solution (y : ℤ) : (10 - 5 * y < 5) → y = 2 := by
  sorry

end NUMINAMATH_GPT_smallest_integer_solution_l1131_113150


namespace NUMINAMATH_GPT_find_x_l1131_113112

theorem find_x (x y : ℤ) (hx : x > y) (hy : y > 0) (h : x + y + x * y = 143) : x = 15 :=
by sorry

end NUMINAMATH_GPT_find_x_l1131_113112


namespace NUMINAMATH_GPT_mutually_exclusive_events_l1131_113139

/-- A group consists of 3 boys and 2 girls. Two students are to be randomly selected to participate in a speech competition. -/
def num_boys : ℕ := 3
def num_girls : ℕ := 2
def total_selected : ℕ := 2

/-- Possible events under consideration:
  A*: Exactly one boy is selected or exactly two girls are selected -/
def is_boy (s : ℕ) (boys : ℕ) : Prop := s ≤ boys 
def is_girl (s : ℕ) (girls : ℕ) : Prop := s ≤ girls
def one_boy_selected (selected : ℕ) (boys : ℕ) := selected = 1 ∧ is_boy selected boys
def two_girls_selected (selected : ℕ) (girls : ℕ) := selected = 2 ∧ is_girl selected girls

theorem mutually_exclusive_events 
  (selected_boy : ℕ) (selected_girl : ℕ) :
  one_boy_selected selected_boy num_boys ∧ selected_boy + selected_girl = total_selected 
  ∧ two_girls_selected selected_girl num_girls 
  → (one_boy_selected selected_boy num_boys ∨ two_girls_selected selected_girl num_girls) :=
by
  sorry

end NUMINAMATH_GPT_mutually_exclusive_events_l1131_113139


namespace NUMINAMATH_GPT_fraction_of_journey_asleep_l1131_113106

theorem fraction_of_journey_asleep (x y : ℝ) (hx : x > 0) (hy : y = x / 3) :
  y / x = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_journey_asleep_l1131_113106


namespace NUMINAMATH_GPT_pesticide_residue_comparison_l1131_113145

noncomputable def f (x : ℝ) : ℝ := 1 / (1 + x^2)

theorem pesticide_residue_comparison (a : ℝ) (ha : a > 0) :
  (f a = (1 / (1 + a^2))) ∧ 
  (if a = 2 * Real.sqrt 2 then f a = 16 / (4 + a^2)^2 else 
   if a > 2 * Real.sqrt 2 then f a > 16 / (4 + a^2)^2 else 
   f a < 16 / (4 + a^2)^2) ∧
  (f 0 = 1) ∧ 
  (f 1 = 1 / 2) := sorry

end NUMINAMATH_GPT_pesticide_residue_comparison_l1131_113145


namespace NUMINAMATH_GPT_contestant_final_score_l1131_113115

theorem contestant_final_score 
    (content_score : ℕ)
    (delivery_score : ℕ)
    (weight_content : ℕ)
    (weight_delivery : ℕ)
    (h1 : content_score = 90)
    (h2 : delivery_score = 85)
    (h3 : weight_content = 6)
    (h4 : weight_delivery = 4) : 
    (content_score * weight_content + delivery_score * weight_delivery) / (weight_content + weight_delivery) = 88 := 
sorry

end NUMINAMATH_GPT_contestant_final_score_l1131_113115


namespace NUMINAMATH_GPT_bellas_score_l1131_113159

theorem bellas_score (sum_19 : ℕ) (sum_20 : ℕ) (avg_19 : ℕ) (avg_20 : ℕ) (n_19 : ℕ) (n_20 : ℕ) :
  avg_19 = 82 → avg_20 = 85 → n_19 = 19 → n_20 = 20 → sum_19 = n_19 * avg_19 → sum_20 = n_20 * avg_20 →
  sum_20 - sum_19 = 142 := 
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end NUMINAMATH_GPT_bellas_score_l1131_113159


namespace NUMINAMATH_GPT_product_in_A_l1131_113121

def A : Set ℤ := { z | ∃ a b : ℤ, z = a^2 + 4 * a * b + b^2 }

theorem product_in_A (x y : ℤ) (hx : x ∈ A) (hy : y ∈ A) : x * y ∈ A := 
by
  sorry

end NUMINAMATH_GPT_product_in_A_l1131_113121


namespace NUMINAMATH_GPT_distinct_rational_numbers_count_l1131_113192

theorem distinct_rational_numbers_count :
  ∃ N : ℕ, 
    (N = 49) ∧
    ∀ (k : ℚ), |k| < 50 →
      (∃ x : ℤ, x^2 - k * x + 18 = 0) →
        ∃ m: ℤ, k = 2 * m ∧ |m| < 25 :=
sorry

end NUMINAMATH_GPT_distinct_rational_numbers_count_l1131_113192


namespace NUMINAMATH_GPT_combinations_with_common_subjects_l1131_113128

-- Conditions and known facts
def subjects : Finset String := {"politics", "history", "geography", "physics", "chemistry", "biology", "technology"}
def personA_must_choose : Finset String := {"physics", "politics"}
def personB_cannot_choose : String := "technology"
def total_combinations : Nat := Nat.choose 7 3
def valid_combinations : Nat := Nat.choose 5 1 * Nat.choose 6 3
def non_common_subject_combinations : Nat := 4 + 4

-- We need to prove this statement
theorem combinations_with_common_subjects : valid_combinations - non_common_subject_combinations = 92 := by
  sorry

end NUMINAMATH_GPT_combinations_with_common_subjects_l1131_113128


namespace NUMINAMATH_GPT_monotonic_increase_interval_l1131_113146

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem monotonic_increase_interval : ∀ x : ℝ, 0 < x ∧ x < Real.exp 1 → 0 < (Real.log x) / x :=
by sorry

end NUMINAMATH_GPT_monotonic_increase_interval_l1131_113146


namespace NUMINAMATH_GPT_yoojeong_rabbits_l1131_113157

theorem yoojeong_rabbits :
  ∀ (R C : ℕ), 
  let minyoung_dogs := 9
  let minyoung_cats := 3
  let minyoung_rabbits := 5
  let minyoung_total := minyoung_dogs + minyoung_cats + minyoung_rabbits
  let yoojeong_total := minyoung_total + 2
  let yoojeong_dogs := 7
  let yoojeong_cats := R - 2
  yoojeong_total = yoojeong_dogs + (R - 2) + R → 
  R = 7 :=
by
  intros R C minyoung_dogs minyoung_cats minyoung_rabbits minyoung_total yoojeong_total yoojeong_dogs yoojeong_cats
  have h1 : minyoung_total = 9 + 3 + 5 := rfl
  have h2 : yoojeong_total = minyoung_total + 2 := by sorry
  have h3 : yoojeong_dogs = 7 := rfl
  have h4 : yoojeong_cats = R - 2 := by sorry
  sorry

end NUMINAMATH_GPT_yoojeong_rabbits_l1131_113157


namespace NUMINAMATH_GPT_mathematicians_correctness_l1131_113185

theorem mathematicians_correctness :
  (2 / 5 + 3 / 8) / (5 + 8) = 5 / 13 ∧
  (4 / 10 + 3 / 8) / (10 + 8) = 7 / 18 ∧
  (3 / 8 < 2 / 5 ∧ 2 / 5 < 17 / 40) → false :=
by 
  sorry

end NUMINAMATH_GPT_mathematicians_correctness_l1131_113185


namespace NUMINAMATH_GPT_range_of_a_l1131_113141

noncomputable def f (a x : ℝ) : ℝ := x^3 + 3 * a * x^2 + 3 * (a + 2) * x + 1

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, deriv (f a) x ≥ 0) ↔ -1 ≤ a ∧ a ≤ 2 :=
sorry

end NUMINAMATH_GPT_range_of_a_l1131_113141


namespace NUMINAMATH_GPT_probability_A_to_B_in_8_moves_l1131_113168

-- Define vertices
inductive Vertex : Type
| A | B | C | D | E | F

open Vertex

-- Define the probability of ending up at Vertex B after 8 moves starting from Vertex A
noncomputable def probability_at_B_after_8_moves : ℚ :=
  let prob := (3 : ℚ) / 16
  prob

-- Theorem statement
theorem probability_A_to_B_in_8_moves :
  (probability_at_B_after_8_moves = (3 : ℚ) / 16) :=
by
  -- Proof to be provided
  sorry

end NUMINAMATH_GPT_probability_A_to_B_in_8_moves_l1131_113168


namespace NUMINAMATH_GPT_number_of_floors_l1131_113194

def hours_per_room : ℕ := 6
def hourly_rate : ℕ := 15
def total_earnings : ℕ := 3600
def rooms_per_floor : ℕ := 10

theorem number_of_floors : 
  (total_earnings / hourly_rate / hours_per_room) / rooms_per_floor = 4 := by
  sorry

end NUMINAMATH_GPT_number_of_floors_l1131_113194


namespace NUMINAMATH_GPT_remainder_when_3_pow_305_div_13_l1131_113147

theorem remainder_when_3_pow_305_div_13 :
  (3 ^ 305) % 13 = 9 := 
by {
  sorry
}

end NUMINAMATH_GPT_remainder_when_3_pow_305_div_13_l1131_113147


namespace NUMINAMATH_GPT_moles_NaClO4_formed_l1131_113151

-- Condition: Balanced chemical reaction
def reaction : Prop := ∀ (NaOH HClO4 NaClO4 H2O : ℕ), NaOH + HClO4 = NaClO4 + H2O

-- Given: 3 moles of NaOH and 3 moles of HClO4
def initial_moles_NaOH : ℕ := 3
def initial_moles_HClO4 : ℕ := 3

-- Question: number of moles of NaClO4 formed
def final_moles_NaClO4 : ℕ := 3

-- Proof Problem: Given the balanced chemical reaction and initial moles, prove the final moles of NaClO4
theorem moles_NaClO4_formed : reaction → initial_moles_NaOH = 3 → initial_moles_HClO4 = 3 → final_moles_NaClO4 = 3 :=
by
  intros
  sorry

end NUMINAMATH_GPT_moles_NaClO4_formed_l1131_113151


namespace NUMINAMATH_GPT_angle_C_in_triangle_l1131_113119

theorem angle_C_in_triangle (A B C : ℝ) (h1 : A + B = 80) (h2 : A + B + C = 180) : C = 100 := 
by 
  sorry

end NUMINAMATH_GPT_angle_C_in_triangle_l1131_113119


namespace NUMINAMATH_GPT_grocery_store_total_bottles_l1131_113144

def total_bottles (regular_soda : Nat) (diet_soda : Nat) : Nat :=
  regular_soda + diet_soda

theorem grocery_store_total_bottles :
 (total_bottles 9 8 = 17) :=
 by
   sorry

end NUMINAMATH_GPT_grocery_store_total_bottles_l1131_113144


namespace NUMINAMATH_GPT_maximum_value_of_expression_l1131_113188

noncomputable def max_value (x y z w : ℝ) : ℝ := 2 * x + 3 * y + 5 * z - 4 * w

theorem maximum_value_of_expression 
  (x y z w : ℝ)
  (h : 9 * x^2 + 4 * y^2 + 25 * z^2 + 16 * w^2 = 4) : 
  max_value x y z w ≤ 6 * Real.sqrt 6 :=
sorry

end NUMINAMATH_GPT_maximum_value_of_expression_l1131_113188


namespace NUMINAMATH_GPT_max_students_distribution_l1131_113169

theorem max_students_distribution (pens toys : ℕ) (h_pens : pens = 451) (h_toys : toys = 410) :
  Nat.gcd pens toys = 41 :=
by
  sorry

end NUMINAMATH_GPT_max_students_distribution_l1131_113169


namespace NUMINAMATH_GPT_catering_budget_total_l1131_113189

theorem catering_budget_total 
  (total_guests : ℕ)
  (guests_want_chicken guests_want_steak : ℕ)
  (cost_steak cost_chicken : ℕ) 
  (H1 : total_guests = 80)
  (H2 : guests_want_steak = 3 * guests_want_chicken)
  (H3 : cost_steak = 25)
  (H4 : cost_chicken = 18)
  (H5 : guests_want_chicken + guests_want_steak = 80) :
  (guests_want_chicken * cost_chicken + guests_want_steak * cost_steak = 1860) := 
by
  sorry

end NUMINAMATH_GPT_catering_budget_total_l1131_113189


namespace NUMINAMATH_GPT_find_p_l1131_113140

theorem find_p 
  (p q x y : ℤ)
  (h1 : p * x + q * y = 8)
  (h2 : 3 * x - q * y = 38)
  (hx : x = 2)
  (hy : y = -4) : 
  p = 20 := 
by 
  subst hx
  subst hy
  sorry

end NUMINAMATH_GPT_find_p_l1131_113140


namespace NUMINAMATH_GPT_trains_crossing_time_l1131_113152

theorem trains_crossing_time :
  let length_first_train := 500
  let length_second_train := 800
  let speed_first_train := 80 * (5/18 : ℚ)  -- convert km/hr to m/s
  let speed_second_train := 100 * (5/18 : ℚ)  -- convert km/hr to m/s
  let relative_speed := speed_first_train + speed_second_train
  let total_distance := length_first_train + length_second_train
  let time_taken := total_distance / relative_speed
  time_taken = 26 :=
by
  sorry

end NUMINAMATH_GPT_trains_crossing_time_l1131_113152


namespace NUMINAMATH_GPT_rectangle_perimeter_l1131_113163

theorem rectangle_perimeter (a b : ℕ) (h1 : a ≠ b) (h2 : a * b = 2 * (2 * a + 2 * b)) : 2 * (a + b) = 36 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_perimeter_l1131_113163


namespace NUMINAMATH_GPT_ab_ac_bc_all_real_l1131_113160

theorem ab_ac_bc_all_real (a b c : ℝ) (h : a + b + c = 1) : ∃ x : ℝ, ab + ac + bc = x := by
  sorry

end NUMINAMATH_GPT_ab_ac_bc_all_real_l1131_113160


namespace NUMINAMATH_GPT_second_store_earns_at_least_72000_more_l1131_113184

-- Conditions as definitions in Lean.
def discount_price := 900000 -- 10% discount on 1 million yuan.
def full_price := 1000000 -- Full price for 1 million yuan without discount.

-- Prize calculation for the second department store.
def prize_first := 1000 * 5
def prize_second := 500 * 10
def prize_third := 200 * 20
def prize_fourth := 100 * 40
def prize_fifth := 10 * 1000

def total_prizes := prize_first + prize_second + prize_third + prize_fourth + prize_fifth

def second_store_net_income := full_price - total_prizes -- Net income after subtracting prizes.

-- The proof problem statement.
theorem second_store_earns_at_least_72000_more :
  second_store_net_income - discount_price >= 72000 := sorry

end NUMINAMATH_GPT_second_store_earns_at_least_72000_more_l1131_113184


namespace NUMINAMATH_GPT_solve_for_a_l1131_113101

theorem solve_for_a (a x : ℤ) (h : x + 2 * a = -3) (hx : x = 1) : a = -2 := by
  sorry

end NUMINAMATH_GPT_solve_for_a_l1131_113101


namespace NUMINAMATH_GPT_depth_of_grass_sheet_l1131_113113

-- Given conditions
def playground_area : ℝ := 5900
def grass_cost_per_cubic_meter : ℝ := 2.80
def total_cost : ℝ := 165.2

-- Variable to solve for
variable (d : ℝ)

-- Theorem statement
theorem depth_of_grass_sheet
  (h : total_cost = (playground_area * d) * grass_cost_per_cubic_meter) :
  d = 0.01 :=
by
  sorry

end NUMINAMATH_GPT_depth_of_grass_sheet_l1131_113113


namespace NUMINAMATH_GPT_fraction_values_l1131_113123

theorem fraction_values (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : 2 * x^2 + 2 * y^2 = 5 * x * y) :
  ∃ k ∈ ({3, -3} : Set ℝ), (x + y) / (x - y) = k :=
by
  sorry

end NUMINAMATH_GPT_fraction_values_l1131_113123


namespace NUMINAMATH_GPT_delta_eq_bullet_l1131_113158

-- Definitions of all variables involved
variables (Δ Θ σ : ℕ)

-- Condition 1: Δ + Δ = σ
def cond1 : Prop := Δ + Δ = σ

-- Condition 2: σ + Δ = Θ
def cond2 : Prop := σ + Δ = Θ

-- Condition 3: Θ = 3Δ
def cond3 : Prop := Θ = 3 * Δ

-- The proof problem
theorem delta_eq_bullet (Δ Θ σ : ℕ) (h1 : Δ + Δ = σ) (h2 : σ + Δ = Θ) (h3 : Θ = 3 * Δ) : 3 * Δ = Θ :=
by
  -- Simply restate the conditions and ensure the proof
  sorry

end NUMINAMATH_GPT_delta_eq_bullet_l1131_113158


namespace NUMINAMATH_GPT_tyson_one_point_count_l1131_113198

def tyson_three_points := 3 * 15
def tyson_two_points := 2 * 12
def total_points := 75
def points_from_three_and_two := tyson_three_points + tyson_two_points

theorem tyson_one_point_count :
  ∃ n : ℕ, n % 2 = 0 ∧ (n = total_points - points_from_three_and_two) :=
sorry

end NUMINAMATH_GPT_tyson_one_point_count_l1131_113198


namespace NUMINAMATH_GPT_geometric_series_first_term_l1131_113117

theorem geometric_series_first_term
  (a r : ℝ)
  (h1 : a / (1 - r) = 30)
  (h2 : a^2 / (1 - r^2) = 90)
  (hrange : |r| < 1) :
  a = 60 / 11 :=
by 
  sorry

end NUMINAMATH_GPT_geometric_series_first_term_l1131_113117


namespace NUMINAMATH_GPT_simplified_fraction_l1131_113130

noncomputable def simplify_and_rationalize (a b c d e f : ℝ) : ℝ :=
  (Real.sqrt a / Real.sqrt b) * (Real.sqrt c / Real.sqrt d) * (Real.sqrt e / Real.sqrt f)

theorem simplified_fraction :
  simplify_and_rationalize 3 7 5 9 6 8 = Real.sqrt 35 / 14 :=
by
  sorry

end NUMINAMATH_GPT_simplified_fraction_l1131_113130


namespace NUMINAMATH_GPT_union_complement_set_l1131_113193

open Set

variable (U : Set ℕ) (M : Set ℕ) (N : Set ℕ)

def complement_in_U (U N : Set ℕ) : Set ℕ :=
  U \ N

theorem union_complement_set :
  U = {0, 1, 2, 4, 6, 8} →
  M = {0, 4, 6} →
  N = {0, 1, 6} →
  M ∪ (complement_in_U U N) = {0, 2, 4, 6, 8} :=
by
  intros
  rw [complement_in_U, union_comm]
  sorry

end NUMINAMATH_GPT_union_complement_set_l1131_113193


namespace NUMINAMATH_GPT_club_membership_l1131_113180

theorem club_membership:
  (∃ (committee : ℕ → Prop) (member_assign : (ℕ × ℕ) → ℕ → Prop),
    (∀ i, i < 5 → ∃! m, member_assign (i, m) 2) ∧
    (∀ i j, i < 5 ∧ j < 5 ∧ i ≠ j → ∃! m, m < 10 ∧ member_assign (i, j) m)
  ) → 
  ∃ n, n = 10 :=
by
  sorry

end NUMINAMATH_GPT_club_membership_l1131_113180


namespace NUMINAMATH_GPT_sin_tan_relation_l1131_113155

theorem sin_tan_relation (θ : ℝ) (h : Real.tan θ = 2) : 
  Real.sin θ * Real.sin (3 * Real.pi / 2 + θ) = -(2 / 5) := 
sorry

end NUMINAMATH_GPT_sin_tan_relation_l1131_113155


namespace NUMINAMATH_GPT_cosine_expression_rewrite_l1131_113134

theorem cosine_expression_rewrite (x : ℝ) :
  ∃ a b c d : ℕ, 
    a * (Real.cos (b * x) * Real.cos (c * x) * Real.cos (d * x)) = 
    Real.cos (2 * x) + Real.cos (6 * x) + Real.cos (14 * x) + Real.cos (18 * x) 
    ∧ a + b + c + d = 22 := sorry

end NUMINAMATH_GPT_cosine_expression_rewrite_l1131_113134


namespace NUMINAMATH_GPT_sum_placed_on_SI_l1131_113100

theorem sum_placed_on_SI :
  let P₁ := 4000
  let r₁ := 0.10
  let t₁ := 2
  let CI := P₁ * ((1 + r₁)^t₁ - 1)

  let SI := (1 / 2 * CI : ℝ)
  let r₂ := 0.08
  let t₂ := 3
  let P₂ := SI / (r₂ * t₂)

  P₂ = 1750 :=
by
  sorry

end NUMINAMATH_GPT_sum_placed_on_SI_l1131_113100


namespace NUMINAMATH_GPT_total_bouncy_balls_l1131_113126

def red_packs := 4
def yellow_packs := 8
def green_packs := 4
def balls_per_pack := 10

theorem total_bouncy_balls:
  (red_packs * balls_per_pack + yellow_packs * balls_per_pack + green_packs * balls_per_pack) = 160 :=
by 
  sorry

end NUMINAMATH_GPT_total_bouncy_balls_l1131_113126


namespace NUMINAMATH_GPT_fabric_sales_fraction_l1131_113186

def total_sales := 36
def stationery_sales := 15
def jewelry_sales := total_sales / 4
def fabric_sales := total_sales - jewelry_sales - stationery_sales

theorem fabric_sales_fraction:
  (fabric_sales : ℝ) / total_sales = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_fabric_sales_fraction_l1131_113186


namespace NUMINAMATH_GPT_eval_expression_l1131_113122

theorem eval_expression (a b c : ℕ) (h₀ : a = 3) (h₁ : b = 2) (h₂ : c = 1) : 
  (a^3 + b^2 + c)^2 - (a^3 + b^2 - c)^2 = 124 :=
by
  sorry

end NUMINAMATH_GPT_eval_expression_l1131_113122


namespace NUMINAMATH_GPT_sqrt_mixed_number_simplify_l1131_113190

open Real

theorem sqrt_mixed_number_simplify :
  sqrt (8 + 9 / 16) = sqrt 137 / 4 :=
by 
  sorry

end NUMINAMATH_GPT_sqrt_mixed_number_simplify_l1131_113190


namespace NUMINAMATH_GPT_extreme_point_a_zero_l1131_113154

noncomputable def f (a x : ℝ) : ℝ := a * x^3 + x^2 - (a + 2) * x + 1
def f_prime (a x : ℝ) : ℝ := 3 * a * x^2 + 2 * x - (a + 2)

theorem extreme_point_a_zero (a : ℝ) (h : f_prime a 1 = 0) : a = 0 :=
by
  sorry

end NUMINAMATH_GPT_extreme_point_a_zero_l1131_113154


namespace NUMINAMATH_GPT_initial_performers_count_l1131_113120

theorem initial_performers_count (n : ℕ)
    (h1 : ∃ rows, 8 * rows = n)
    (h2 : ∃ (m : ℕ), n + 16 = m ∧ ∃ s, s * s = m)
    (h3 : ∃ (k : ℕ), n + 1 = k ∧ ∃ t, t * t = k) : 
    n = 48 := 
sorry

end NUMINAMATH_GPT_initial_performers_count_l1131_113120


namespace NUMINAMATH_GPT_pieces_by_first_team_correct_l1131_113171

-- Define the number of pieces required.
def total_pieces : ℕ := 500

-- Define the number of pieces made by the second team.
def pieces_by_second_team : ℕ := 131

-- Define the number of pieces made by the third team.
def pieces_by_third_team : ℕ := 180

-- Define the number of pieces made by the first team.
def pieces_by_first_team : ℕ := total_pieces - (pieces_by_second_team + pieces_by_third_team)

-- Statement to prove
theorem pieces_by_first_team_correct : pieces_by_first_team = 189 := 
by 
  -- Proof to be filled in
  sorry

end NUMINAMATH_GPT_pieces_by_first_team_correct_l1131_113171


namespace NUMINAMATH_GPT_inequality_problem_l1131_113149

theorem inequality_problem (a b c d : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d)
    (h_sum : a + b + c + d = 4) : 
    a^2 * b * c + b^2 * d * a + c^2 * d * a + d^2 * b * c ≤ 4 := 
sorry

end NUMINAMATH_GPT_inequality_problem_l1131_113149


namespace NUMINAMATH_GPT_find_n_l1131_113182

theorem find_n (n : ℕ) (h1 : 0 ≤ n) (h2 : n ≤ 14) : n ≡ 14567 [MOD 15] → n = 2 := 
by
  sorry

end NUMINAMATH_GPT_find_n_l1131_113182


namespace NUMINAMATH_GPT_log_a_plus_b_eq_zero_l1131_113183

open Complex

noncomputable def a_b_expression : ℂ := (⟨2, 1⟩ / ⟨1, 1⟩ : ℂ)

noncomputable def a : ℝ := a_b_expression.re

noncomputable def b : ℝ := a_b_expression.im

theorem log_a_plus_b_eq_zero : log (a + b) = 0 := by
  sorry

end NUMINAMATH_GPT_log_a_plus_b_eq_zero_l1131_113183


namespace NUMINAMATH_GPT_value_of_frac_l1131_113172

theorem value_of_frac (x y z w : ℕ) 
  (hz : z = 5 * w) 
  (hy : y = 3 * z) 
  (hx : x = 4 * y) : 
  x * z / (y * w) = 20 := 
  sorry

end NUMINAMATH_GPT_value_of_frac_l1131_113172


namespace NUMINAMATH_GPT_remainder_2001_to_2005_mod_19_l1131_113173

theorem remainder_2001_to_2005_mod_19 :
  (2001 * 2002 * 2003 * 2004 * 2005) % 19 = 11 :=
by
  -- Use modular arithmetic properties to convert each factor
  have h2001 : 2001 % 19 = 6 := by sorry
  have h2002 : 2002 % 19 = 7 := by sorry
  have h2003 : 2003 % 19 = 8 := by sorry
  have h2004 : 2004 % 19 = 9 := by sorry
  have h2005 : 2005 % 19 = 10 := by sorry

  -- Compute the product modulo 19
  have h_prod : (6 * 7 * 8 * 9 * 10) % 19 = 11 := by sorry

  -- Combining these results
  have h_final : ((2001 * 2002 * 2003 * 2004 * 2005) % 19) = (6 * 7 * 8 * 9 * 10) % 19 := by sorry
  exact Eq.trans h_final h_prod

end NUMINAMATH_GPT_remainder_2001_to_2005_mod_19_l1131_113173


namespace NUMINAMATH_GPT_min_square_sum_l1131_113143

theorem min_square_sum (a b : ℝ) (h : a + b = 3) : a^2 + b^2 ≥ 9 / 2 :=
by 
  sorry

end NUMINAMATH_GPT_min_square_sum_l1131_113143


namespace NUMINAMATH_GPT_min_dwarfs_l1131_113118

theorem min_dwarfs (chairs : Fin 30 → Prop) 
  (h1 : ∀ i : Fin 30, chairs i ∨ chairs ((i + 1) % 30) ∨ chairs ((i + 2) % 30)) :
  ∃ S : Finset (Fin 30), (S.card = 10) ∧ (∀ i : Fin 30, i ∈ S) :=
sorry

end NUMINAMATH_GPT_min_dwarfs_l1131_113118


namespace NUMINAMATH_GPT_chromium_percentage_l1131_113103

theorem chromium_percentage (c1 c2 : ℝ) (w1 w2 : ℝ) (percentage1 percentage2 : ℝ) : 
  percentage1 = 0.1 → 
  percentage2 = 0.08 → 
  w1 = 15 → 
  w2 = 35 → 
  (c1 = percentage1 * w1) → 
  (c2 = percentage2 * w2) → 
  (c1 + c2 = 4.3) → 
  ((w1 + w2) = 50) →
  ((c1 + c2) / (w1 + w2) * 100 = 8.6) := 
by 
  sorry

end NUMINAMATH_GPT_chromium_percentage_l1131_113103


namespace NUMINAMATH_GPT_expression_equivalence_l1131_113105

-- Define the initial expression
def expr (w : ℝ) : ℝ := 3 * w + 4 - 2 * w^2 - 5 * w - 6 + w^2 + 7 * w + 8 - 3 * w^2

-- Define the simplified expression
def simplified_expr (w : ℝ) : ℝ := 5 * w - 4 * w^2 + 6

-- Theorem stating the equivalence
theorem expression_equivalence (w : ℝ) : expr w = simplified_expr w :=
by
  -- we would normally simplify and prove here, but we state the theorem and skip the proof for now.
  sorry

end NUMINAMATH_GPT_expression_equivalence_l1131_113105


namespace NUMINAMATH_GPT_goldfish_count_equal_in_6_months_l1131_113125

def initial_goldfish_brent : ℕ := 3
def initial_goldfish_gretel : ℕ := 243

def goldfish_brent (n : ℕ) : ℕ := initial_goldfish_brent * 4^n
def goldfish_gretel (n : ℕ) : ℕ := initial_goldfish_gretel * 3^n

theorem goldfish_count_equal_in_6_months : 
  (∃ n : ℕ, goldfish_brent n = goldfish_gretel n) ↔ n = 6 :=
by
  sorry

end NUMINAMATH_GPT_goldfish_count_equal_in_6_months_l1131_113125


namespace NUMINAMATH_GPT_cos_R_in_triangle_PQR_l1131_113136

theorem cos_R_in_triangle_PQR
  (P Q R : ℝ) (hP : P = 90) (hQ : Real.sin Q = 3/5)
  (h_sum : P + Q + R = 180) (h_PQ_comp : P + Q = 90) :
  Real.cos R = 3 / 5 := 
sorry

end NUMINAMATH_GPT_cos_R_in_triangle_PQR_l1131_113136


namespace NUMINAMATH_GPT_number_of_people_in_group_l1131_113179

theorem number_of_people_in_group (P : ℕ) : 
  (∃ (P : ℕ), 0 < P ∧ (364 / P - 1 = 364 / (P + 2))) → P = 26 :=
by
  sorry

end NUMINAMATH_GPT_number_of_people_in_group_l1131_113179


namespace NUMINAMATH_GPT_minimum_bailing_rate_is_seven_l1131_113124

noncomputable def minimum_bailing_rate (shore_distance : ℝ) (paddling_speed : ℝ) 
                                       (water_intake_rate : ℝ) (max_capacity : ℝ) : ℝ := 
  let time_to_shore := shore_distance / paddling_speed
  let intake_total := water_intake_rate * time_to_shore
  let required_rate := (intake_total - max_capacity) / time_to_shore
  required_rate

theorem minimum_bailing_rate_is_seven 
  (shore_distance : ℝ) (paddling_speed : ℝ) (water_intake_rate : ℝ) (max_capacity : ℝ) :
  shore_distance = 2 →
  paddling_speed = 3 →
  water_intake_rate = 8 →
  max_capacity = 40 →
  minimum_bailing_rate shore_distance paddling_speed water_intake_rate max_capacity = 7 :=
by 
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end NUMINAMATH_GPT_minimum_bailing_rate_is_seven_l1131_113124


namespace NUMINAMATH_GPT_total_spent_amount_l1131_113166

-- Define the conditions
def spent_relation (B D : ℝ) : Prop := D = 0.75 * B
def payment_difference (B D : ℝ) : Prop := B = D + 12.50

-- Define the theorem to prove
theorem total_spent_amount (B D : ℝ) 
  (h1 : spent_relation B D) 
  (h2 : payment_difference B D) : 
  B + D = 87.50 :=
sorry

end NUMINAMATH_GPT_total_spent_amount_l1131_113166


namespace NUMINAMATH_GPT_train_distance_30_minutes_l1131_113175

theorem train_distance_30_minutes (h : ∀ (t : ℝ), 0 < t → (1 / 2) * t = 1 / 2 * t) : 
  (1 / 2) * 30 = 15 :=
by
  sorry

end NUMINAMATH_GPT_train_distance_30_minutes_l1131_113175


namespace NUMINAMATH_GPT_tens_digit_17_pow_1993_l1131_113129

theorem tens_digit_17_pow_1993 :
  (17 ^ 1993) % 100 / 10 = 3 := by
  sorry

end NUMINAMATH_GPT_tens_digit_17_pow_1993_l1131_113129


namespace NUMINAMATH_GPT_vertex_of_parabola_l1131_113161

theorem vertex_of_parabola :
  ∃ (x y : ℝ), (∀ x : ℝ, y = x^2 - 12 * x + 9) → (x, y) = (6, -27) :=
sorry

end NUMINAMATH_GPT_vertex_of_parabola_l1131_113161


namespace NUMINAMATH_GPT_range_of_a_l1131_113199

noncomputable def f (a x : ℝ) : ℝ := x^3 + x^2 - a * x - 4
noncomputable def f_derivative (a x : ℝ) : ℝ := 3 * x^2 + 2 * x - a

def has_exactly_one_extremum_in_interval (a : ℝ) : Prop :=
  (f_derivative a (-1)) * (f_derivative a 1) < 0

theorem range_of_a (a : ℝ) :
  has_exactly_one_extremum_in_interval a ↔ (1 < a ∧ a < 5) :=
sorry

end NUMINAMATH_GPT_range_of_a_l1131_113199


namespace NUMINAMATH_GPT_find_n_l1131_113167

theorem find_n :
  ∃ (n : ℤ), 50 ≤ n ∧ n ≤ 120 ∧ (n % 8 = 0) ∧ (n % 7 = 5) ∧ (n % 6 = 3) ∧ n = 208 := 
by {
  sorry
}

end NUMINAMATH_GPT_find_n_l1131_113167


namespace NUMINAMATH_GPT_double_rooms_booked_l1131_113170

theorem double_rooms_booked (S D : ℕ) 
(rooms_booked : S + D = 260) 
(single_room_cost : 35 * S + 60 * D = 14000) : 
D = 196 := 
sorry

end NUMINAMATH_GPT_double_rooms_booked_l1131_113170


namespace NUMINAMATH_GPT_fraction_simplification_l1131_113104

theorem fraction_simplification :
  (1^2 + 1) * (2^2 + 1) * (3^2 + 1) / ((2^2 - 1) * (3^2 - 1) * (4^2 - 1)) = 5 / 18 :=
by
  sorry

end NUMINAMATH_GPT_fraction_simplification_l1131_113104


namespace NUMINAMATH_GPT_blue_pill_cost_l1131_113177

theorem blue_pill_cost (y : ℕ) :
  -- Conditions
  (∀ t d : ℕ, t = 21 → 
     d = 14 → 
     (735 - d * 2 = t * ((2 * y) + (y + 2)) / t) →
     2 * y + (y + 2) = 35) →
  -- Conclusion
  y = 11 :=
by
  sorry

end NUMINAMATH_GPT_blue_pill_cost_l1131_113177


namespace NUMINAMATH_GPT_weight_of_new_person_l1131_113191

variable (avg_increase : ℝ) (n_persons : ℕ) (weight_replaced : ℝ)

theorem weight_of_new_person (h1 : avg_increase = 3.5) (h2 : n_persons = 8) (h3 : weight_replaced = 65) :
  let total_weight_increase := n_persons * avg_increase
  let weight_new := weight_replaced + total_weight_increase
  weight_new = 93 := by
  sorry

end NUMINAMATH_GPT_weight_of_new_person_l1131_113191


namespace NUMINAMATH_GPT_smallest_prime_perimeter_l1131_113153

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_scalene_triangle (a b c : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a + b > c ∧ a + c > b ∧ b + c > a

def is_prime_perimeter_scalene_triangle (a b c : ℕ) : Prop :=
  is_prime a ∧ is_prime b ∧ is_prime c ∧
  is_scalene_triangle a b c ∧ is_prime (a + b + c)

theorem smallest_prime_perimeter (a b c : ℕ) :
  (a = 5 ∧ a < b ∧ a < c ∧ is_prime_perimeter_scalene_triangle a b c) →
  (a + b + c = 23) :=
by
  sorry

end NUMINAMATH_GPT_smallest_prime_perimeter_l1131_113153


namespace NUMINAMATH_GPT_find_c_value_l1131_113148

theorem find_c_value (A B C : ℝ) (S1_area S2_area : ℝ) (b : ℝ) :
  S1_area = 40 * b + 1 →
  S2_area = 40 * b →
  ∃ c, AC + CB = c ∧ c = 462 :=
by
  intro hS1 hS2
  sorry

end NUMINAMATH_GPT_find_c_value_l1131_113148


namespace NUMINAMATH_GPT_brooke_total_jumping_jacks_l1131_113108

def sj1 : Nat := 20
def sj2 : Nat := 36
def sj3 : Nat := 40
def sj4 : Nat := 50
def Brooke_jumping_jacks : Nat := 3 * (sj1 + sj2 + sj3 + sj4)

theorem brooke_total_jumping_jacks : Brooke_jumping_jacks = 438 := by
  sorry

end NUMINAMATH_GPT_brooke_total_jumping_jacks_l1131_113108


namespace NUMINAMATH_GPT_count_selection_4_balls_count_selection_5_balls_score_at_least_7_points_l1131_113196

-- Setup the basic context
def Pocket := Finset (Fin 11)

-- The pocket contains 4 red balls and 7 white balls
def red_balls : Finset (Fin 11) := {0, 1, 2, 3}
def white_balls : Finset (Fin 11) := {4, 5, 6, 7, 8, 9, 10}

-- Question 1
theorem count_selection_4_balls :
  (red_balls.card.choose 4) + (red_balls.card.choose 3 * white_balls.card.choose 1) +
  (red_balls.card.choose 2 * white_balls.card.choose 2) = 115 := 
sorry

-- Question 2
theorem count_selection_5_balls_score_at_least_7_points :
  (red_balls.card.choose 2 * white_balls.card.choose 3) +
  (red_balls.card.choose 3 * white_balls.card.choose 2) +
  (red_balls.card.choose 4 * white_balls.card.choose 1) = 301 := 
sorry

end NUMINAMATH_GPT_count_selection_4_balls_count_selection_5_balls_score_at_least_7_points_l1131_113196


namespace NUMINAMATH_GPT_exp_sum_is_neg_one_l1131_113110

noncomputable def sumExpExpressions : ℂ :=
  (Complex.exp (Real.pi * Complex.I / 7) +
   Complex.exp (2 * Real.pi * Complex.I / 7) +
   Complex.exp (3 * Real.pi * Complex.I / 7) +
   Complex.exp (4 * Real.pi * Complex.I / 7) +
   Complex.exp (5 * Real.pi * Complex.I / 7) +
   Complex.exp (6 * Real.pi * Complex.I / 7) +
   Complex.exp (2 * Real.pi * Complex.I / 9) +
   Complex.exp (4 * Real.pi * Complex.I / 9) +
   Complex.exp (6 * Real.pi * Complex.I / 9) +
   Complex.exp (8 * Real.pi * Complex.I / 9) +
   Complex.exp (10 * Real.pi * Complex.I / 9) +
   Complex.exp (12 * Real.pi * Complex.I / 9) +
   Complex.exp (14 * Real.pi * Complex.I / 9) +
   Complex.exp (16 * Real.pi * Complex.I / 9))

theorem exp_sum_is_neg_one : sumExpExpressions = -1 := by
  sorry

end NUMINAMATH_GPT_exp_sum_is_neg_one_l1131_113110


namespace NUMINAMATH_GPT_min_value_expr_l1131_113109

theorem min_value_expr (x : ℝ) (hx : x > 0) : 4 * x + 1 / x^2 ≥ 5 :=
by
  sorry

end NUMINAMATH_GPT_min_value_expr_l1131_113109


namespace NUMINAMATH_GPT_system_of_equations_m_value_l1131_113137

theorem system_of_equations_m_value {x y m : ℝ} 
  (h1 : 2 * x + y = 4)
  (h2 : x + 2 * y = m)
  (h3 : x + y = 1) : m = -1 := 
sorry

end NUMINAMATH_GPT_system_of_equations_m_value_l1131_113137


namespace NUMINAMATH_GPT_perpendicular_line_through_point_l1131_113195

open Real

def line (a b c x y : ℝ) : Prop := a * x + b * y + c = 0

theorem perpendicular_line_through_point (x y : ℝ) (c : ℝ) :
  (line 2 1 (-5) x y) → (x = 3) ∧ (y = 0) → (line 1 (-2) 3 x y) := by
sorry

end NUMINAMATH_GPT_perpendicular_line_through_point_l1131_113195
