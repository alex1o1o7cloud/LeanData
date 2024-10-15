import Mathlib

namespace NUMINAMATH_GPT_coats_collected_in_total_l235_23528

def high_school_coats : Nat := 6922
def elementary_school_coats : Nat := 2515
def total_coats : Nat := 9437

theorem coats_collected_in_total : 
  high_school_coats + elementary_school_coats = total_coats := 
  by
  sorry

end NUMINAMATH_GPT_coats_collected_in_total_l235_23528


namespace NUMINAMATH_GPT_number_proportion_l235_23506

theorem number_proportion (number : ℚ) :
  (number : ℚ) / 12 = 9 / 360 →
  number = 0.3 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_number_proportion_l235_23506


namespace NUMINAMATH_GPT_find_a_l235_23594

noncomputable def polynomial (a : ℝ) : ℝ → ℝ := λ x => a * x^2 + (a - 3) * x + 1

-- This is a statement without the actual computation or proof.
theorem find_a (a : ℝ) :
  (∀ x : ℝ, polynomial a x = 0 → (∃! x, polynomial a x = 0)) ↔ a = 0 ∨ a = 1 ∨ a = 9 :=
sorry

end NUMINAMATH_GPT_find_a_l235_23594


namespace NUMINAMATH_GPT_problem_l235_23557

-- Definitions of the function g and its values at specific points
def g (d e f : ℝ) (x : ℝ) : ℝ := d * x^2 + e * x + f

-- Conditions given in the problem
theorem problem (d e f : ℝ)
  (h0 : g d e f 0 = 8)
  (h1 : g d e f 1 = 5) :
  d + e + 2 * f = 13 :=
by
  sorry

end NUMINAMATH_GPT_problem_l235_23557


namespace NUMINAMATH_GPT_men_count_in_first_group_is_20_l235_23518

noncomputable def men_needed_to_build_fountain (work1 : ℝ) (days1 : ℕ) (length1 : ℝ) (workers2 : ℕ) (days2 : ℕ) (length2 : ℝ) (work_per_man_per_day2 : ℝ) : ℕ :=
  let work_per_day2 := length2 / days2
  let work_per_man_per_day2 := work_per_day2 / workers2
  let total_work1 := length1 / days1
  Nat.floor (total_work1 / work_per_man_per_day2)

theorem men_count_in_first_group_is_20 :
  men_needed_to_build_fountain 56 6 56 35 3 49 (49 / (35 * 3)) = 20 :=
by
  sorry

end NUMINAMATH_GPT_men_count_in_first_group_is_20_l235_23518


namespace NUMINAMATH_GPT_similar_triangle_leg_length_l235_23539

theorem similar_triangle_leg_length (a b c : ℝ) (h0 : a = 12) (h1 : b = 9) (h2 : c = 7.5) :
  ∃ y : ℝ, ((12 / 7.5) = (9 / y) → y = 5.625) :=
by
  use 5.625
  intro h
  linarith

end NUMINAMATH_GPT_similar_triangle_leg_length_l235_23539


namespace NUMINAMATH_GPT_reaction_requires_two_moles_of_HNO3_l235_23541

def nitric_acid_reaction (HNO3 NaHCO3 NaNO3 CO2 H2O : ℕ) 
  (reaction : HNO3 + NaHCO3 = NaNO3 + CO2 + H2O)
  (n_NaHCO3 : ℕ) : ℕ :=
  if n_NaHCO3 = 2 then 2 else sorry

theorem reaction_requires_two_moles_of_HNO3
  (HNO3 NaHCO3 NaNO3 CO2 H2O : ℕ) 
  (reaction : HNO3 + NaHCO3 = NaNO3 + CO2 + H2O)
  (n_NaHCO3 : ℕ) :
  n_NaHCO3 = 2 → nitric_acid_reaction HNO3 NaHCO3 NaNO3 CO2 H2O reaction n_NaHCO3 = 2 :=
by sorry

end NUMINAMATH_GPT_reaction_requires_two_moles_of_HNO3_l235_23541


namespace NUMINAMATH_GPT_algebra_expression_value_l235_23568

theorem algebra_expression_value (a b : ℝ) (h : (30^3) * a + 30 * b - 7 = 9) :
  (-30^3) * a + (-30) * b + 2 = -14 := 
by
  sorry

end NUMINAMATH_GPT_algebra_expression_value_l235_23568


namespace NUMINAMATH_GPT_system_consistent_and_solution_l235_23511

theorem system_consistent_and_solution (a x : ℝ) : 
  (a = -10 ∧ x = -1/3) ∨ (a = -8 ∧ x = -1) ∨ (a = 4 ∧ x = -2) ↔ 
  3 * x^2 - x - a - 10 = 0 ∧ (a + 4) * x + a + 12 = 0 := by
  sorry

end NUMINAMATH_GPT_system_consistent_and_solution_l235_23511


namespace NUMINAMATH_GPT_shot_put_distance_l235_23547

theorem shot_put_distance :
  (∃ x : ℝ, (y = - 1 / 12 * x^2 + 2 / 3 * x + 5 / 3) ∧ y = 0) ↔ x = 10 := 
by
  sorry

end NUMINAMATH_GPT_shot_put_distance_l235_23547


namespace NUMINAMATH_GPT_winner_more_votes_l235_23540

variable (totalStudents : ℕ) (votingPercentage : ℤ) (winnerPercentage : ℤ) (loserPercentage : ℤ)

theorem winner_more_votes
    (h1 : totalStudents = 2000)
    (h2 : votingPercentage = 25)
    (h3 : winnerPercentage = 55)
    (h4 : loserPercentage = 100 - winnerPercentage)
    (h5 : votingStudents = votingPercentage * totalStudents / 100)
    (h6 : winnerVotes = winnerPercentage * votingStudents / 100)
    (h7 : loserVotes = loserPercentage * votingStudents / 100)
    : winnerVotes - loserVotes = 50 := by
  sorry

end NUMINAMATH_GPT_winner_more_votes_l235_23540


namespace NUMINAMATH_GPT_speed_in_still_water_l235_23582

theorem speed_in_still_water (upstream downstream : ℝ) 
  (h_up : upstream = 25) 
  (h_down : downstream = 45) : 
  (upstream + downstream) / 2 = 35 := 
by 
  -- Proof will go here
  sorry

end NUMINAMATH_GPT_speed_in_still_water_l235_23582


namespace NUMINAMATH_GPT_coneCannotBeQuadrilateral_l235_23516

-- Define types for our geometric solids
inductive Solid
| Cylinder
| Cone
| FrustumCone
| Prism

-- Define a predicate for whether the cross-section can be a quadrilateral
def canBeQuadrilateral (s : Solid) : Prop :=
  match s with
  | Solid.Cylinder => true
  | Solid.Cone => false
  | Solid.FrustumCone => true
  | Solid.Prism => true

-- The theorem we need to prove
theorem coneCannotBeQuadrilateral : canBeQuadrilateral Solid.Cone = false := by
  sorry

end NUMINAMATH_GPT_coneCannotBeQuadrilateral_l235_23516


namespace NUMINAMATH_GPT_determine_g_2023_l235_23527

noncomputable def g (x : ℕ) : ℝ := sorry

axiom g_pos (x : ℕ) (hx : x > 0) : g x > 0

axiom g_property (x y : ℕ) (h1 : x > 2 * y) (h2 : 0 < y) : 
  g (x - y) = Real.sqrt (g (x / y) + 3)

theorem determine_g_2023 : g 2023 = (1 + Real.sqrt 13) / 2 :=
by
  sorry

end NUMINAMATH_GPT_determine_g_2023_l235_23527


namespace NUMINAMATH_GPT_portia_high_school_students_l235_23554

variables (P L M : ℕ)
axiom h1 : P = 4 * L
axiom h2 : P = 2 * M
axiom h3 : P + L + M = 4800

theorem portia_high_school_students : P = 2740 :=
by sorry

end NUMINAMATH_GPT_portia_high_school_students_l235_23554


namespace NUMINAMATH_GPT_exponent_proof_l235_23530

theorem exponent_proof (m : ℝ) : (243 : ℝ) = (3 : ℝ)^5 → (243 : ℝ)^(1/3) = (3 : ℝ)^m → m = 5/3 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_exponent_proof_l235_23530


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_l235_23564

variables (A B : Prop)

theorem necessary_but_not_sufficient 
  (h1 : ¬ B → ¬ A)  -- Condition: ¬ B → ¬ A is true
  (h2 : ¬ (¬ A → ¬ B))  -- Condition: ¬ A → ¬ B is false
  : (A → B) ∧ ¬ (B → A) := -- Conclusion: A → B and not (B → A)
by
  -- Proof is not required, so we place sorry
  sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_l235_23564


namespace NUMINAMATH_GPT_identify_A_B_l235_23556

variable {Person : Type}
variable (isTruthful isLiar : Person → Prop)
variable (isBoy isGirl : Person → Prop)

variables (A B : Person)

-- Conditions
axiom truthful_or_liar : ∀ x : Person, isTruthful x ∨ isLiar x
axiom boy_or_girl : ∀ x : Person, isBoy x ∨ isGirl x
axiom not_both_truthful_and_liar : ∀ x : Person, ¬(isTruthful x ∧ isLiar x)
axiom not_both_boy_and_girl : ∀ x : Person, ¬(isBoy x ∧ isGirl x)

-- Statements made by A and B
axiom A_statement : isTruthful A → isLiar B 
axiom B_statement : isBoy B → isGirl A 

-- Goal: prove the identities of A and B
theorem identify_A_B : isTruthful A ∧ isBoy A ∧ isLiar B ∧ isBoy B :=
by {
  sorry
}

end NUMINAMATH_GPT_identify_A_B_l235_23556


namespace NUMINAMATH_GPT_circle_area_l235_23581

-- Define the conditions of the problem
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 + 6*x - 4*y + 9 = 0

-- State the proof problem
theorem circle_area : (∀ (x y : ℝ), circle_equation x y) → (∀ r : ℝ, r = 2 → π * r^2 = 4 * π) :=
by
  sorry

end NUMINAMATH_GPT_circle_area_l235_23581


namespace NUMINAMATH_GPT_points_satisfy_l235_23524

theorem points_satisfy (x y : ℝ) : 
  (y^2 - y = x^2 - x) ↔ (y = x ∨ y = 1 - x) :=
by sorry

end NUMINAMATH_GPT_points_satisfy_l235_23524


namespace NUMINAMATH_GPT_find_smaller_number_l235_23538

theorem find_smaller_number (x y : ℕ) (h1 : x + y = 24) (h2 : 7 * x = 5 * y) : x = 10 :=
sorry

end NUMINAMATH_GPT_find_smaller_number_l235_23538


namespace NUMINAMATH_GPT_constant_term_correct_l235_23529

variable (x : ℝ)

noncomputable def constant_term_expansion : ℝ :=
  let term := λ (r : ℕ) => (Nat.choose 9 r) * (-2)^r * x^((9 - 9 * r) / 2)
  term 1

theorem constant_term_correct : 
  constant_term_expansion x = -18 :=
sorry

end NUMINAMATH_GPT_constant_term_correct_l235_23529


namespace NUMINAMATH_GPT_find_roots_l235_23517

theorem find_roots : 
  ∀ x : ℝ, (x^2 - 5*x + 6) * (x - 3) * (x + 2) = 0 ↔ (x = -2 ∨ x = 2 ∨ x = 3) := by
  sorry

end NUMINAMATH_GPT_find_roots_l235_23517


namespace NUMINAMATH_GPT_g_eval_1000_l235_23598

def g (n : ℕ) : ℕ := sorry
axiom g_comp (n : ℕ) : g (g n) = 2 * n
axiom g_form (n : ℕ) : g (3 * n + 1) = 3 * n + 2

theorem g_eval_1000 : g 1000 = 1008 :=
by
  sorry

end NUMINAMATH_GPT_g_eval_1000_l235_23598


namespace NUMINAMATH_GPT_value_of_expression_l235_23590

theorem value_of_expression : 1 + 2 / (1 + 2 / (2 * 2)) = 7 / 3 := 
by 
  -- proof to be filled in
  sorry

end NUMINAMATH_GPT_value_of_expression_l235_23590


namespace NUMINAMATH_GPT_min_rows_needed_l235_23526

-- Define the basic conditions
def total_students := 2016
def seats_per_row := 168
def max_students_per_school := 40

-- Define the minimum number of rows required to accommodate all conditions
noncomputable def min_required_rows (students : ℕ) (seats : ℕ) (max_per_school : ℕ) : ℕ := 15

-- Lean theorem asserting the truth of the above definition under given conditions
theorem min_rows_needed : min_required_rows total_students seats_per_row max_students_per_school = 15 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_min_rows_needed_l235_23526


namespace NUMINAMATH_GPT_exists_polynomial_distinct_powers_of_2_l235_23502

open Polynomial

variable (n : ℕ) (hn : n > 0)

theorem exists_polynomial_distinct_powers_of_2 :
  ∃ P : Polynomial ℤ, P.degree = n ∧ (∃ (k : Fin (n + 1) → ℕ), ∀ i j : Fin (n + 1), i ≠ j → 2 ^ k i ≠ 2 ^ k j ∧ (∀ i, P.eval i.val = 2 ^ k i)) :=
sorry

end NUMINAMATH_GPT_exists_polynomial_distinct_powers_of_2_l235_23502


namespace NUMINAMATH_GPT_xiao_li_profit_l235_23503

noncomputable def original_price_per_share : ℝ := 21 / 1.05
noncomputable def closing_price_first_day : ℝ := original_price_per_share * 0.94
noncomputable def selling_price_second_day : ℝ := closing_price_first_day * 1.10
noncomputable def total_profit : ℝ := (selling_price_second_day - 21) * 5000

theorem xiao_li_profit :
  total_profit = 600 := sorry

end NUMINAMATH_GPT_xiao_li_profit_l235_23503


namespace NUMINAMATH_GPT_number_of_sad_children_l235_23532

-- Definitions of the given conditions
def total_children : ℕ := 60
def happy_children : ℕ := 30
def neither_happy_nor_sad_children : ℕ := 20

-- The main statement to be proved
theorem number_of_sad_children : 
  total_children - happy_children - neither_happy_nor_sad_children = 10 := 
by 
  sorry

end NUMINAMATH_GPT_number_of_sad_children_l235_23532


namespace NUMINAMATH_GPT_a_2018_value_l235_23560

theorem a_2018_value (S a : ℕ -> ℕ) (h₁ : S 1 = a 1) (h₂ : a 1 = 1) (h₃ : ∀ n : ℕ, n > 0 -> S (n + 1) = 3 * S n) :
  a 2018 = 2 * 3 ^ 2016 :=
sorry

end NUMINAMATH_GPT_a_2018_value_l235_23560


namespace NUMINAMATH_GPT_result_of_y_minus_3x_l235_23583

theorem result_of_y_minus_3x (x y : ℝ) (h1 : x + y = 8) (h2 : y - x = 7.5) : y - 3 * x = 7 :=
sorry

end NUMINAMATH_GPT_result_of_y_minus_3x_l235_23583


namespace NUMINAMATH_GPT_find_cost_price_l235_23566

-- Define the known data
def cost_price_80kg (C : ℝ) := 80 * C
def cost_price_20kg := 20 * 20
def selling_price_mixed := 2000
def total_cost_price_mixed (C : ℝ) := cost_price_80kg C + cost_price_20kg

-- Using the condition for 25% profit
def selling_price_of_mixed (C : ℝ) := 1.25 * total_cost_price_mixed C

-- The main theorem
theorem find_cost_price (C : ℝ) : selling_price_of_mixed C = selling_price_mixed → C = 15 :=
by
  sorry

end NUMINAMATH_GPT_find_cost_price_l235_23566


namespace NUMINAMATH_GPT_no_real_solutions_l235_23571

theorem no_real_solutions (x : ℝ) (h_nonzero : x ≠ 0) (h_pos : 0 < x):
  (x^10 + 1) * (x^8 + x^6 + x^4 + x^2 + 1) ≠ 12 * x^9 :=
by
-- Proof will go here.
sorry

end NUMINAMATH_GPT_no_real_solutions_l235_23571


namespace NUMINAMATH_GPT_find_fourth_number_l235_23519

theorem find_fourth_number : 
  ∃ (x : ℝ), (217 + 2.017 + 0.217 + x = 221.2357) ∧ (x = 2.0017) :=
by
  sorry

end NUMINAMATH_GPT_find_fourth_number_l235_23519


namespace NUMINAMATH_GPT_polynomial_coeff_sum_eq_neg_two_l235_23562

/-- If (1 - 2 * x) ^ 9 = a₉ * x ^ 9 + a₈ * x ^ 8 + ... + a₂ * x ^ 2 + a₁ * x + a₀, 
then a₁ + a₂ + ... + a₈ + a₉ = -2. -/
theorem polynomial_coeff_sum_eq_neg_two 
  (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ : ℤ) 
  (h : (1 - 2 * x) ^ 9 = a₉ * x ^ 9 + a₈ * x ^ 8 + a₇ * x ^ 7 + a₆ * x ^ 6 + a₅ * x ^ 5 + a₄ * x ^ 4 + a₃ * x ^ 3 + a₂ * x ^ 2 + a₁ * x + a₀) : 
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ = -2 :=
by sorry

end NUMINAMATH_GPT_polynomial_coeff_sum_eq_neg_two_l235_23562


namespace NUMINAMATH_GPT_triangle_DEF_area_l235_23536

noncomputable def point := (ℝ × ℝ)

def D : point := (-2, 2)
def E : point := (8, 2)
def F : point := (6, -4)

def area_of_triangle (A B C : point) : ℝ :=
  0.5 * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

theorem triangle_DEF_area : area_of_triangle D E F = 30 := by
  sorry

end NUMINAMATH_GPT_triangle_DEF_area_l235_23536


namespace NUMINAMATH_GPT_max_value_of_z_l235_23534

theorem max_value_of_z 
    (x y : ℝ) 
    (h1 : |2 * x + y + 1| ≤ |x + 2 * y + 2|)
    (h2 : -1 ≤ y ∧ y ≤ 1) : 
    2 * x + y ≤ 5 := 
sorry

end NUMINAMATH_GPT_max_value_of_z_l235_23534


namespace NUMINAMATH_GPT_households_used_both_brands_l235_23573

/-- 
A marketing firm determined that, of 160 households surveyed, 80 used neither brand A nor brand B soap.
60 used only brand A soap and for every household that used both brands of soap, 3 used only brand B soap.
--/
theorem households_used_both_brands (X: ℕ) (H: 4*X + 140 = 160): X = 5 :=
by
  sorry

end NUMINAMATH_GPT_households_used_both_brands_l235_23573


namespace NUMINAMATH_GPT_ellipse_condition_necessary_but_not_sufficient_l235_23561

-- Define the conditions and proof statement in Lean 4
theorem ellipse_condition (m : ℝ) (h₁ : 2 < m) (h₂ : m < 6) : 
  (6 - m ≠ m - 2) -> 
  (∃ x y : ℝ, (x^2) / (m - 2) + (y^2) / (6 - m)= 1) :=
by
  sorry

theorem necessary_but_not_sufficient : (2 < m ∧ m < 6) ↔ (2 < m ∧ m < 6 ∧ m ≠ 4) :=
by
  sorry

end NUMINAMATH_GPT_ellipse_condition_necessary_but_not_sufficient_l235_23561


namespace NUMINAMATH_GPT_edge_length_of_cube_l235_23535

theorem edge_length_of_cube (V : ℝ) (e : ℝ) (h1 : V = 2744) (h2 : V = e^3) : e = 14 := 
by 
  sorry

end NUMINAMATH_GPT_edge_length_of_cube_l235_23535


namespace NUMINAMATH_GPT_x_lt_1_nec_not_suff_l235_23555

theorem x_lt_1_nec_not_suff (x : ℝ) : (x < 1 → x^2 < 1) ∧ (¬(x < 1) → x^2 < 1) := 
by {
  sorry
}

end NUMINAMATH_GPT_x_lt_1_nec_not_suff_l235_23555


namespace NUMINAMATH_GPT_minimum_a_l235_23544

theorem minimum_a (a : ℝ) (h : a > 0) :
  (∀ (N : ℝ × ℝ), (N.1 - a)^2 + (N.2 + a - 3)^2 = 1 → 
   dist (N.1, N.2) (0, 0) ≥ 2) → a ≥ 3 :=
by
  sorry

end NUMINAMATH_GPT_minimum_a_l235_23544


namespace NUMINAMATH_GPT_total_songs_l235_23514

-- Define the number of albums Faye bought and the number of songs per album
def country_albums : ℕ := 2
def pop_albums : ℕ := 3
def songs_per_album : ℕ := 6

-- Define the total number of albums Faye bought
def total_albums : ℕ := country_albums + pop_albums

-- Prove that the total number of songs Faye bought is 30
theorem total_songs : total_albums * songs_per_album = 30 := by
  sorry

end NUMINAMATH_GPT_total_songs_l235_23514


namespace NUMINAMATH_GPT_find_p_l235_23563

theorem find_p (p : ℝ) : 
  (Nat.choose 5 3) * p^3 = 80 → p = 2 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_p_l235_23563


namespace NUMINAMATH_GPT_no_intersection_l235_23501

def M := { p : ℝ × ℝ | p.1^2 + p.2^2 = 1 }
def N (a : ℝ) := { p : ℝ × ℝ | abs (p.1 - 1) + abs (p.2 - 1) = a }

theorem no_intersection (a : ℝ) : M ∩ (N a) = ∅ ↔ a ∈ (Set.Ioo (2-Real.sqrt 2) (2+Real.sqrt 2)) := 
by 
  sorry

end NUMINAMATH_GPT_no_intersection_l235_23501


namespace NUMINAMATH_GPT_range_of_x_l235_23578

theorem range_of_x (f : ℝ → ℝ) (h_even : ∀ x, f x = f (-x)) (h_increasing : ∀ {a b : ℝ}, a ≤ b → b ≤ 0 → f a ≤ f b) :
  (∀ x : ℝ, f (2^(2*x^2 - x - 1)) ≥ f (-4)) → ∀ x, x ∈ Set.Icc (-1 : ℝ) (3/2 : ℝ) :=
by 
  sorry

end NUMINAMATH_GPT_range_of_x_l235_23578


namespace NUMINAMATH_GPT_number_of_blue_crayons_given_to_Becky_l235_23513

-- Definitions based on the conditions
def initial_green_crayons : ℕ := 5
def initial_blue_crayons : ℕ := 8
def given_out_green_crayons : ℕ := 3
def total_crayons_left : ℕ := 9

-- Statement of the problem and expected proof
theorem number_of_blue_crayons_given_to_Becky (initial_green_crayons initial_blue_crayons given_out_green_crayons total_crayons_left : ℕ) : 
  initial_green_crayons = 5 →
  initial_blue_crayons = 8 →
  given_out_green_crayons = 3 →
  total_crayons_left = 9 →
  ∃ num_blue_crayons_given_to_Becky, num_blue_crayons_given_to_Becky = 1 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_number_of_blue_crayons_given_to_Becky_l235_23513


namespace NUMINAMATH_GPT_fraction_equality_l235_23509

theorem fraction_equality (a b : ℝ) (h : a ≠ b) (h_eq : a / b + (a + 5 * b) / (b + 5 * a) = 2) : a / b = 3 / 5 :=
by
  sorry

end NUMINAMATH_GPT_fraction_equality_l235_23509


namespace NUMINAMATH_GPT_jerry_cut_pine_trees_l235_23525

theorem jerry_cut_pine_trees (P : ℕ)
  (h1 : 3 * 60 = 180)
  (h2 : 4 * 100 = 400)
  (h3 : 80 * P + 180 + 400 = 1220) :
  P = 8 :=
by {
  sorry -- Proof not required as per the instructions
}

end NUMINAMATH_GPT_jerry_cut_pine_trees_l235_23525


namespace NUMINAMATH_GPT_inscribed_sphere_radius_eq_l235_23520

-- Define the parameters for the right cone
structure RightCone where
  base_radius : ℝ
  height : ℝ

-- Given the right cone conditions
def givenCone : RightCone := { base_radius := 15, height := 40 }

-- Define the properties for inscribed sphere
def inscribedSphereRadius (c : RightCone) : ℝ := sorry

-- The theorem statement for the radius of the inscribed sphere
theorem inscribed_sphere_radius_eq (c : RightCone) : ∃ (b d : ℝ), 
  inscribedSphereRadius c = b * Real.sqrt d - b ∧ (b + d = 14) :=
by
  use 5, 9
  sorry

end NUMINAMATH_GPT_inscribed_sphere_radius_eq_l235_23520


namespace NUMINAMATH_GPT_translation_m_n_l235_23565

theorem translation_m_n (m n : ℤ) (P Q : ℤ × ℤ) (hP : P = (-1, -3)) (hQ : Q = (-2, 0))
(hx : P.1 - m = Q.1) (hy : P.2 + n = Q.2) :
  m + n = 4 :=
by
  sorry

end NUMINAMATH_GPT_translation_m_n_l235_23565


namespace NUMINAMATH_GPT_fraction_of_p_l235_23549

theorem fraction_of_p (p q r f : ℝ) (hp : p = 49) (hqr : p = (2 * f * 49) + 35) : f = 1/7 :=
sorry

end NUMINAMATH_GPT_fraction_of_p_l235_23549


namespace NUMINAMATH_GPT_julia_height_in_cm_l235_23543

def height_in_feet : ℕ := 5
def height_in_inches : ℕ := 4
def feet_to_inches : ℕ := 12
def inch_to_cm : ℝ := 2.54

theorem julia_height_in_cm : (height_in_feet * feet_to_inches + height_in_inches) * inch_to_cm = 162.6 :=
sorry

end NUMINAMATH_GPT_julia_height_in_cm_l235_23543


namespace NUMINAMATH_GPT_slope_OA_l235_23588

-- Definitions for the given conditions
def ellipse (a b : ℝ) := {P : ℝ × ℝ | (P.1^2) / a^2 + (P.2^2) / b^2 = 1}

def C1 := ellipse 2 1  -- ∑(x^2 / 4 + y^2 = 1)
def C2 := ellipse 2 4  -- ∑(y^2 / 16 + x^2 / 4 = 1)

variable {P₁ P₂ : ℝ × ℝ}  -- Points A and B
variable (h1 : P₁ ∈ C1)
variable (h2 : P₂ ∈ C2)
variable (h_rel : P₂.1 = 2 * P₁.1 ∧ P₂.2 = 2 * P₁.2)  -- ∑(x₂ = 2x₁, y₂ = 2y₁)

-- Proof that the slope of ray OA is ±1
theorem slope_OA : ∃ (m : ℝ), (m = 1 ∨ m = -1) :=
sorry

end NUMINAMATH_GPT_slope_OA_l235_23588


namespace NUMINAMATH_GPT_marble_count_l235_23591

-- Definitions from conditions
variable (M P : ℕ)
def condition1 : Prop := M = 26 * P
def condition2 : Prop := M = 28 * (P - 1)

-- Theorem to be proved
theorem marble_count (h1 : condition1 M P) (h2 : condition2 M P) : M = 364 := 
by
  sorry

end NUMINAMATH_GPT_marble_count_l235_23591


namespace NUMINAMATH_GPT_james_owns_145_l235_23537

theorem james_owns_145 (total : ℝ) (diff : ℝ) (james_and_ali : total = 250) (james_more_than_ali : diff = 40):
  ∃ (james ali : ℝ), ali + diff = james ∧ ali + james = total ∧ james = 145 :=
by
  sorry

end NUMINAMATH_GPT_james_owns_145_l235_23537


namespace NUMINAMATH_GPT_max_value_of_largest_integer_l235_23587

theorem max_value_of_largest_integer (a1 a2 a3 a4 a5 a6 a7 : ℕ) (h1 : a1 + a2 + a3 + a4 + a5 + a6 + a7 = 560) (h2 : a7 - a1 = 20) : a7 ≤ 21 :=
sorry

end NUMINAMATH_GPT_max_value_of_largest_integer_l235_23587


namespace NUMINAMATH_GPT_age_ratio_in_years_l235_23510

variable (s d x : ℕ)

theorem age_ratio_in_years (h1 : s - 3 = 2 * (d - 3)) (h2 : s - 7 = 3 * (d - 7)) (hx : (s + x) = 3 * (d + x) / 2) : x = 5 := sorry

end NUMINAMATH_GPT_age_ratio_in_years_l235_23510


namespace NUMINAMATH_GPT_ship_with_highest_no_car_round_trip_percentage_l235_23508

theorem ship_with_highest_no_car_round_trip_percentage
    (pA : ℝ)
    (cA_r : ℝ)
    (pB : ℝ)
    (cB_r : ℝ)
    (pC : ℝ)
    (cC_r : ℝ)
    (hA : pA = 0.30)
    (hA_car : cA_r = 0.25)
    (hB : pB = 0.50)
    (hB_car : cB_r = 0.15)
    (hC : pC = 0.20)
    (hC_car : cC_r = 0.35) :
    let percentA := pA - (cA_r * pA)
    let percentB := pB - (cB_r * pB)
    let percentC := pC - (cC_r * pC)
    percentB > percentA ∧ percentB > percentC :=
by
  sorry

end NUMINAMATH_GPT_ship_with_highest_no_car_round_trip_percentage_l235_23508


namespace NUMINAMATH_GPT_inequality_holds_l235_23552

theorem inequality_holds (c : ℝ) : (∀ x : ℝ, 3 * Real.sin x - 4 * Real.cos x + c > 0) → c > 5 := by sorry

end NUMINAMATH_GPT_inequality_holds_l235_23552


namespace NUMINAMATH_GPT_determine_a_and_theta_l235_23589

noncomputable def f (a θ : ℝ) (x : ℝ) : ℝ := 2 * a * Real.sin (2 * x + θ)

theorem determine_a_and_theta :
  (∃ a θ : ℝ, 0 < θ ∧ θ < π ∧ a ≠ 0 ∧ (∀ x ∈ Set.Icc (-2 : ℝ) (2 : ℝ), f a θ x ∈ Set.Icc (-2 : ℝ) 2) ∧ 
  (∀ (x1 x2 : ℝ), x1 ∈ Set.Icc (-5 * π / 12) (π / 12) → x2 ∈ Set.Icc (-5 * π / 12) (π / 12) → x1 < x2 → f a θ x1 > f a θ x2)) →
  (a = -1) ∧ (θ = π / 3) :=
sorry

end NUMINAMATH_GPT_determine_a_and_theta_l235_23589


namespace NUMINAMATH_GPT_find_larger_number_l235_23512

theorem find_larger_number (L S : ℕ) (h1 : L - S = 1325) (h2 : L = 5 * S + 5) : L = 1655 :=
sorry

end NUMINAMATH_GPT_find_larger_number_l235_23512


namespace NUMINAMATH_GPT_seven_circle_divisors_exists_non_adjacent_divisors_l235_23576

theorem seven_circle_divisors_exists_non_adjacent_divisors (a : Fin 7 → ℕ)
  (h_adj : ∀ i : Fin 7, a i ∣ a (i + 1) % 7 ∨ a (i + 1) % 7 ∣ a i) :
  ∃ (i j : Fin 7), i ≠ j ∧ j ≠ i + 1 % 7 ∧ j ≠ i + 6 % 7 ∧ (a i ∣ a j ∨ a j ∣ a i) :=
by
  sorry

end NUMINAMATH_GPT_seven_circle_divisors_exists_non_adjacent_divisors_l235_23576


namespace NUMINAMATH_GPT_maximum_height_l235_23504

noncomputable def h (t : ℝ) : ℝ :=
  -20 * t ^ 2 + 100 * t + 30

theorem maximum_height : 
  ∃ t : ℝ, h t = 155 ∧ ∀ t' : ℝ, h t' ≤ 155 := 
sorry

end NUMINAMATH_GPT_maximum_height_l235_23504


namespace NUMINAMATH_GPT_floor_area_l235_23533

theorem floor_area (length_feet : ℝ) (width_feet : ℝ) (feet_to_meters : ℝ) 
  (h_length : length_feet = 15) (h_width : width_feet = 10) (h_conversion : feet_to_meters = 0.3048) :
  let length_meters := length_feet * feet_to_meters
  let width_meters := width_feet * feet_to_meters
  let area_meters := length_meters * width_meters
  area_meters = 13.93 := 
by
  sorry

end NUMINAMATH_GPT_floor_area_l235_23533


namespace NUMINAMATH_GPT_increased_sales_type_B_l235_23592

-- Definitions for sales equations
def store_A_sales (x y : ℝ) : Prop :=
  60 * x + 15 * y = 3600

def store_B_sales (x y : ℝ) : Prop :=
  40 * x + 60 * y = 4400

-- Definition for the price of clothing items
def price_A (x : ℝ) : Prop :=
  x = 50

def price_B (y : ℝ) : Prop :=
  y = 40

-- Definition for the increased sales in May for type A
def may_sales_A (x : ℝ) : Prop :=
  100 * x * 1.2 = 6000

-- Definition to prove percentage increase for type B sales in May
noncomputable def percentage_increase_B (x y : ℝ) : ℝ :=
  ((4500 - (100 * y * 0.4)) / (100 * y * 0.4)) * 100

theorem increased_sales_type_B (x y : ℝ)
  (h1 : store_A_sales x y)
  (h2 : store_B_sales x y)
  (hA : price_A x)
  (hB : price_B y)
  (hMayA : may_sales_A x) :
  percentage_increase_B x y = 50 :=
sorry

end NUMINAMATH_GPT_increased_sales_type_B_l235_23592


namespace NUMINAMATH_GPT_min_value_nS_n_l235_23548

theorem min_value_nS_n (a : ℕ → ℝ) (S : ℕ → ℝ) (m : ℕ) 
  (h2 : m ≥ 2)
  (h3 : S (m - 1) = -2)
  (h4 : S m = 0)
  (h5 : S (m + 1) = 3) :
  ∃ n : ℕ, n * S n = -9 :=
sorry

end NUMINAMATH_GPT_min_value_nS_n_l235_23548


namespace NUMINAMATH_GPT_inclination_angle_of_y_axis_l235_23546

theorem inclination_angle_of_y_axis : 
  ∀ (l : ℝ), l = 90 :=
sorry

end NUMINAMATH_GPT_inclination_angle_of_y_axis_l235_23546


namespace NUMINAMATH_GPT_min_visible_sum_of_prism_faces_l235_23522

theorem min_visible_sum_of_prism_faces :
  let corners := 8
  let edges := 8
  let face_centers := 12
  let min_corner_sum := 6 -- Each corner dice can show 1, 2, and 3
  let min_edge_sum := 3    -- Each edge dice can show 1 and 2
  let min_face_center_sum := 1 -- Each face center dice can show 1
  let total_sum := corners * min_corner_sum + edges * min_edge_sum + face_centers * min_face_center_sum
  total_sum = 84 := 
by
  -- The proof is omitted
  sorry

end NUMINAMATH_GPT_min_visible_sum_of_prism_faces_l235_23522


namespace NUMINAMATH_GPT_valid_operation_l235_23531

theorem valid_operation :
  ∀ x : ℝ, x^2 + x^3 ≠ x^5 ∧
  ∀ a b : ℝ, (a - b)^2 ≠ a^2 - b^2 ∧
  ∀ m : ℝ, (|m| = m ↔ m ≥ 0) :=
by
  sorry

end NUMINAMATH_GPT_valid_operation_l235_23531


namespace NUMINAMATH_GPT_tax_rate_for_remaining_l235_23558

variable (total_earnings deductions first_tax_rate total_tax taxed_amount remaining_taxable_income rem_tax_rate : ℝ)

def taxable_income (total_earnings deductions : ℝ) := total_earnings - deductions

def tax_on_first_portion (portion tax_rate : ℝ) := portion * tax_rate

def remaining_taxable (total_taxable first_portion : ℝ) := total_taxable - first_portion

def total_tax_payable (tax_first tax_remaining : ℝ) := tax_first + tax_remaining

theorem tax_rate_for_remaining :
  total_earnings = 100000 ∧ 
  deductions = 30000 ∧ 
  first_tax_rate = 0.10 ∧
  total_tax = 12000 ∧
  tax_on_first_portion 20000 first_tax_rate = 2000 ∧
  taxed_amount = 2000 ∧
  remaining_taxable_income = taxable_income total_earnings deductions - 20000 ∧
  total_tax_payable taxed_amount (remaining_taxable_income * rem_tax_rate) = total_tax →
  rem_tax_rate = 0.20 := 
sorry

end NUMINAMATH_GPT_tax_rate_for_remaining_l235_23558


namespace NUMINAMATH_GPT_honey_production_l235_23570

-- Define the conditions:
def bees : ℕ := 60
def days : ℕ := 60
def honey_per_bee : ℕ := 1

-- Statement to prove:
theorem honey_production (bees_eq : 60 = bees) (days_eq : 60 = days) (honey_per_bee_eq : 1 = honey_per_bee) :
  bees * honey_per_bee = 60 := by
  sorry

end NUMINAMATH_GPT_honey_production_l235_23570


namespace NUMINAMATH_GPT_distance_midpoint_to_origin_l235_23567

variables {a b c d m k l n : ℝ}

theorem distance_midpoint_to_origin (h1 : b = m * a + k) (h2 : d = m * c + k) (h3 : n = -1 / m) :
  dist (0, 0) ( ((a + c) / 2), ((m * (a + c) + 2 * k) / 2) ) = (1 / 2) * Real.sqrt ((1 + m^2) * (a + c)^2 + 4 * k^2 + 4 * m * (a + c) * k) :=
by
  sorry

end NUMINAMATH_GPT_distance_midpoint_to_origin_l235_23567


namespace NUMINAMATH_GPT_sum_of_units_digits_eq_0_l235_23542

-- Units digit function definition
def units_digit (n : ℕ) : ℕ :=
  n % 10

-- Problem statement in Lean 
theorem sum_of_units_digits_eq_0 :
  units_digit (units_digit (17 * 34) + units_digit (19 * 28)) = 0 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_units_digits_eq_0_l235_23542


namespace NUMINAMATH_GPT_find_number_l235_23521

theorem find_number (N : ℕ) (k : ℕ) (Q : ℕ)
  (h1 : N = 9 * k)
  (h2 : Q = 25 * 9 + 7)
  (h3 : N / 9 = Q) :
  N = 2088 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l235_23521


namespace NUMINAMATH_GPT_total_dog_food_amount_l235_23577

def initial_dog_food : ℝ := 15
def first_purchase : ℝ := 15
def second_purchase : ℝ := 10

theorem total_dog_food_amount : initial_dog_food + first_purchase + second_purchase = 40 := 
by 
  sorry

end NUMINAMATH_GPT_total_dog_food_amount_l235_23577


namespace NUMINAMATH_GPT_percentage_increase_of_kim_l235_23580

variables (S P K : ℝ)
variables (h1 : S = 0.80 * P) (h2 : S + P = 1.80) (h3 : K = 1.12)

theorem percentage_increase_of_kim (hK : K = 1.12) (hS : S = 0.80 * P) (hSP : S + P = 1.80) :
  ((K - S) / S * 100) = 40 :=
sorry

end NUMINAMATH_GPT_percentage_increase_of_kim_l235_23580


namespace NUMINAMATH_GPT_tournament_key_player_l235_23574

theorem tournament_key_player (n : ℕ) (plays : Fin n → Fin n → Bool) (wins : ∀ i j, plays i j → ¬plays j i) :
  ∃ X, ∀ (Y : Fin n), Y ≠ X → (plays X Y ∨ ∃ Z, plays X Z ∧ plays Z Y) :=
by
  sorry

end NUMINAMATH_GPT_tournament_key_player_l235_23574


namespace NUMINAMATH_GPT_probability_of_odd_product_is_zero_l235_23586

-- Define the spinners
def spinnerC : List ℕ := [1, 3, 5, 7]
def spinnerD : List ℕ := [2, 4, 6]

-- Define the condition that the odds and evens have a specific product property
axiom odd_times_even_is_even {a b : ℕ} (ha : a % 2 = 1) (hb : b % 2 = 0) : (a * b) % 2 = 0

-- Define the probability of getting an odd product
noncomputable def probability_odd_product : ℕ :=
  if ∃ a ∈ spinnerC, ∃ b ∈ spinnerD, (a * b) % 2 = 1 then 1 else 0

-- Main theorem
theorem probability_of_odd_product_is_zero : probability_odd_product = 0 := by
  sorry

end NUMINAMATH_GPT_probability_of_odd_product_is_zero_l235_23586


namespace NUMINAMATH_GPT_problem_statement_l235_23597

theorem problem_statement (x : ℕ) (h : x = 2016) : (x^2 - x) - (x^2 - 2 * x + 1) = 2015 := by
  sorry

end NUMINAMATH_GPT_problem_statement_l235_23597


namespace NUMINAMATH_GPT_final_quarters_l235_23507

-- Define the initial conditions and transactions
def initial_quarters : ℕ := 760
def first_spent : ℕ := 418
def second_spent : ℕ := 192

-- Define the final amount of quarters Sally should have
theorem final_quarters (initial_quarters first_spent second_spent : ℕ) : initial_quarters - first_spent - second_spent = 150 :=
by
  sorry

end NUMINAMATH_GPT_final_quarters_l235_23507


namespace NUMINAMATH_GPT_smallest_divisible_12_13_14_l235_23515

theorem smallest_divisible_12_13_14 :
  ∃ n : ℕ, n > 0 ∧ (n % 12 = 0) ∧ (n % 13 = 0) ∧ (n % 14 = 0) ∧ n = 1092 := by
  sorry

end NUMINAMATH_GPT_smallest_divisible_12_13_14_l235_23515


namespace NUMINAMATH_GPT_sweater_cost_l235_23584

theorem sweater_cost (S : ℚ) (M : ℚ) (C : ℚ) (h1 : S = 80) (h2 : M = 3 / 4 * 80) (h3 : C = S - M) : C = 20 := by
  sorry

end NUMINAMATH_GPT_sweater_cost_l235_23584


namespace NUMINAMATH_GPT_number_of_exclusive_students_l235_23595

-- Definitions from the conditions
def S_both : ℕ := 16
def S_alg : ℕ := 36
def S_geo_only : ℕ := 15

-- Theorem to prove the number of students taking algebra or geometry but not both
theorem number_of_exclusive_students : (S_alg - S_both) + S_geo_only = 35 :=
by
  sorry

end NUMINAMATH_GPT_number_of_exclusive_students_l235_23595


namespace NUMINAMATH_GPT_uncle_jerry_total_tomatoes_l235_23553

def tomatoes_reaped_yesterday : ℕ := 120
def tomatoes_reaped_more_today : ℕ := 50

theorem uncle_jerry_total_tomatoes : 
  tomatoes_reaped_yesterday + (tomatoes_reaped_yesterday + tomatoes_reaped_more_today) = 290 :=
by 
  sorry

end NUMINAMATH_GPT_uncle_jerry_total_tomatoes_l235_23553


namespace NUMINAMATH_GPT_number_of_teachers_l235_23500

theorem number_of_teachers (total_people : ℕ) (sampled_individuals : ℕ) (sampled_students : ℕ) 
    (school_total : total_people = 2400) 
    (sample_total : sampled_individuals = 160) 
    (sample_students : sampled_students = 150) : 
    ∃ teachers : ℕ, teachers = 150 := 
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_number_of_teachers_l235_23500


namespace NUMINAMATH_GPT_isosceles_triangle_base_angle_l235_23523

theorem isosceles_triangle_base_angle (a b h θ : ℝ)
  (h1 : a^2 = 4 * b^2 * h)
  (h_b : b = 2 * a * Real.cos θ)
  (h_h : h = a * Real.sin θ) :
  θ = Real.arccos (1/4) :=
by
  sorry

end NUMINAMATH_GPT_isosceles_triangle_base_angle_l235_23523


namespace NUMINAMATH_GPT_ratio_of_Frederick_to_Tyson_l235_23575

-- Definitions of the ages based on given conditions
def Kyle : Nat := 25
def Tyson : Nat := 20
def Julian : Nat := Kyle - 5
def Frederick : Nat := Julian + 20

-- The ratio of Frederick's age to Tyson's age
def ratio : Nat × Nat := (Frederick / Nat.gcd Frederick Tyson, Tyson / Nat.gcd Frederick Tyson)

-- Proving the ratio is 2:1
theorem ratio_of_Frederick_to_Tyson : ratio = (2, 1) := by
  sorry

end NUMINAMATH_GPT_ratio_of_Frederick_to_Tyson_l235_23575


namespace NUMINAMATH_GPT_inequality_solution_set_l235_23579

noncomputable def solution_set := {x : ℝ | x^2 + 2 * x - 3 ≥ 0}

theorem inequality_solution_set :
  (solution_set = {x : ℝ | x ≤ -3 ∨ x ≥ 1}) :=
sorry

end NUMINAMATH_GPT_inequality_solution_set_l235_23579


namespace NUMINAMATH_GPT_stone_breadth_5_l235_23585

theorem stone_breadth_5 (hall_length_m hall_breadth_m stone_length_dm num_stones b₁ b₂ : ℝ) 
  (h1 : hall_length_m = 36) 
  (h2 : hall_breadth_m = 15) 
  (h3 : stone_length_dm = 3) 
  (h4 : num_stones = 3600)
  (h5 : hall_length_m * 10 * hall_breadth_m * 10 = 54000)
  (h6 : stone_length_dm * b₁ * num_stones = hall_length_m * 10 * hall_breadth_m * 10) :
  b₂ = 5 := 
  sorry

end NUMINAMATH_GPT_stone_breadth_5_l235_23585


namespace NUMINAMATH_GPT_b4_minus_a4_l235_23593

-- Given quadratic equation and specified root, prove the difference of fourth powers.
theorem b4_minus_a4 (a b : ℝ) (h_root : (a^2 - b^2)^2 = x) (h_equation : x^2 + 4 * a^2 * b^2 * x = 4) : b^4 - a^4 = 2 ∨ b^4 - a^4 = -2 :=
sorry

end NUMINAMATH_GPT_b4_minus_a4_l235_23593


namespace NUMINAMATH_GPT_unaccounted_bottles_l235_23559

theorem unaccounted_bottles :
  let total_bottles := 254
  let football_bottles := 11 * 6
  let soccer_bottles := 53
  let lacrosse_bottles := football_bottles + 12
  let rugby_bottles := 49
  let team_bottles := football_bottles + soccer_bottles + lacrosse_bottles + rugby_bottles
  total_bottles - team_bottles = 8 :=
by
  rfl

end NUMINAMATH_GPT_unaccounted_bottles_l235_23559


namespace NUMINAMATH_GPT_largest_divisor_of_product_of_5_consecutive_integers_l235_23545

/-- What is the largest integer that must divide the product of any 5 consecutive integers? -/
theorem largest_divisor_of_product_of_5_consecutive_integers :
  ∀ n : ℤ, ∃ d : ℤ, d = 24 ∧ d ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  sorry

end NUMINAMATH_GPT_largest_divisor_of_product_of_5_consecutive_integers_l235_23545


namespace NUMINAMATH_GPT_gcd_36_n_eq_12_l235_23550

theorem gcd_36_n_eq_12 (n : ℕ) (h1 : 80 ≤ n) (h2 : n ≤ 100) (h3 : Int.gcd 36 n = 12) : n = 84 ∨ n = 96 :=
by
  sorry

end NUMINAMATH_GPT_gcd_36_n_eq_12_l235_23550


namespace NUMINAMATH_GPT_rate_of_current_l235_23505

theorem rate_of_current (c : ℝ) : 
  (∀ t : ℝ, t = 0.4 → ∀ d : ℝ, d = 9.6 → ∀ b : ℝ, b = 20 →
  d = (b + c) * t → c = 4) :=
sorry

end NUMINAMATH_GPT_rate_of_current_l235_23505


namespace NUMINAMATH_GPT_range_of_a_l235_23596

def is_odd_function (f : ℝ → ℝ) := 
  ∀ x : ℝ, f (-x) = - f x

noncomputable def f (x : ℝ) :=
  if x ≥ 0 then x^2 + 2*x else -(x^2 + 2*(-x))

theorem range_of_a (a : ℝ) (h_odd : is_odd_function f) 
(hf_pos : ∀ x : ℝ, x ≥ 0 → f x = x^2 + 2*x) : 
  f (2 - a^2) > f a → -2 < a ∧ a < 1 :=
sorry

end NUMINAMATH_GPT_range_of_a_l235_23596


namespace NUMINAMATH_GPT_together_finish_work_in_10_days_l235_23599

theorem together_finish_work_in_10_days (x_days y_days : ℕ) (hx : x_days = 15) (hy : y_days = 30) :
  let x_rate := 1 / (x_days : ℚ)
  let y_rate := 1 / (y_days : ℚ)
  let combined_rate := x_rate + y_rate
  let total_days := 1 / combined_rate
  total_days = 10 :=
by
  sorry

end NUMINAMATH_GPT_together_finish_work_in_10_days_l235_23599


namespace NUMINAMATH_GPT_find_second_number_in_denominator_l235_23569

theorem find_second_number_in_denominator :
  (0.625 * 0.0729 * 28.9) / (0.0017 * x * 8.1) = 382.5 → x = 0.24847 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_second_number_in_denominator_l235_23569


namespace NUMINAMATH_GPT_ratio_of_siblings_l235_23551

/-- Let's define the sibling relationships and prove the ratio of Janet's to Masud's siblings is 3 to 1. -/
theorem ratio_of_siblings (masud_siblings : ℕ) (carlos_siblings janet_siblings : ℕ)
  (h1 : masud_siblings = 60)
  (h2 : carlos_siblings = 3 * masud_siblings / 4)
  (h3 : janet_siblings = carlos_siblings + 135) 
  (h4 : janet_siblings < some_mul * masud_siblings) : 
  janet_siblings / masud_siblings = 3 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_siblings_l235_23551


namespace NUMINAMATH_GPT_plane_through_intersection_l235_23572

def plane1 (x y z : ℝ) : Prop := x + y + 5 * z - 1 = 0
def plane2 (x y z : ℝ) : Prop := 2 * x + 3 * y - z + 2 = 0
def pointM (x y z : ℝ) : Prop := (x, y, z) = (3, 2, 1)

theorem plane_through_intersection (x y z : ℝ) :
  plane1 x y z ∧ plane2 x y z ∧ pointM x y z → 5 * x + 14 * y - 74 * z + 31 = 0 := by
  intro h
  sorry

end NUMINAMATH_GPT_plane_through_intersection_l235_23572
