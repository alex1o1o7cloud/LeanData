import Mathlib

namespace fg_eq_gf_condition_l1355_135576

/-- Definitions of the functions f and g --/
def f (m n c x : ℝ) : ℝ := m * x + n + c
def g (p q c x : ℝ) : ℝ := p * x + q + c

/-- The main theorem stating the equivalence of the condition for f(g(x)) = g(f(x)) --/
theorem fg_eq_gf_condition (m n p q c x : ℝ) :
  f m n c (g p q c x) = g p q c (f m n c x) ↔ n * (1 - p) - q * (1 - m) + c * (m - p) = 0 := by
  sorry

end fg_eq_gf_condition_l1355_135576


namespace sin_value_l1355_135534

theorem sin_value (x : ℝ) (h : Real.sin (x + π / 3) = Real.sqrt 3 / 3) :
  Real.sin (2 * π / 3 - x) = Real.sqrt 3 / 3 :=
by
  sorry

end sin_value_l1355_135534


namespace garden_area_increase_l1355_135510

noncomputable def original_garden_length : ℝ := 60
noncomputable def original_garden_width : ℝ := 20
noncomputable def original_garden_area : ℝ := original_garden_length * original_garden_width
noncomputable def original_garden_perimeter : ℝ := 2 * (original_garden_length + original_garden_width)

noncomputable def circle_radius : ℝ := original_garden_perimeter / (2 * Real.pi)
noncomputable def circle_area : ℝ := Real.pi * (circle_radius ^ 2)

noncomputable def area_increase : ℝ := circle_area - original_garden_area

theorem garden_area_increase :
  area_increase = (6400 / Real.pi) - 1200 :=
by 
  sorry -- proof goes here

end garden_area_increase_l1355_135510


namespace reeya_third_subject_score_l1355_135507

theorem reeya_third_subject_score
  (score1 score2 score4 : ℕ)
  (avg_score : ℕ)
  (num_subjects : ℕ)
  (total_score : ℕ)
  (score3 : ℕ) :
  score1 = 65 →
  score2 = 67 →
  score4 = 85 →
  avg_score = 75 →
  num_subjects = 4 →
  total_score = avg_score * num_subjects →
  score1 + score2 + score3 + score4 = total_score →
  score3 = 83 :=
by
  intros h1 h2 h4 h5 h6 h7 h8
  sorry

end reeya_third_subject_score_l1355_135507


namespace salary_january_l1355_135598

theorem salary_january
  (J F M A May : ℝ)
  (h1 : (J + F + M + A) / 4 = 8000)
  (h2 : (F + M + A + May) / 4 = 8600)
  (h3 : May = 6500) :
  J = 4100 :=
by 
  sorry

end salary_january_l1355_135598


namespace simplify_expression_l1355_135532

theorem simplify_expression (a c d x y z : ℝ) :
  (cx * (a^3 * x^3 + 3 * a^3 * y^3 + c^3 * z^3) + dz * (a^3 * x^3 + 3 * c^3 * x^3 + c^3 * z^3)) / (cx + dz) =
  a^3 * x^3 + c^3 * z^3 + (3 * cx * a^3 * y^3 / (cx + dz)) + (3 * dz * c^3 * x^3 / (cx + dz)) :=
by
  sorry

end simplify_expression_l1355_135532


namespace stickers_after_birthday_l1355_135545

-- Definitions based on conditions
def initial_stickers : Nat := 39
def birthday_stickers : Nat := 22

-- Theorem stating the problem we aim to prove
theorem stickers_after_birthday : initial_stickers + birthday_stickers = 61 :=
by 
  sorry

end stickers_after_birthday_l1355_135545


namespace points_difference_l1355_135502

-- Define the given data
def points_per_touchdown : ℕ := 7
def brayden_gavin_touchdowns : ℕ := 7
def cole_freddy_touchdowns : ℕ := 9

-- Define the theorem to prove the difference in points
theorem points_difference :
  (points_per_touchdown * cole_freddy_touchdowns) - 
  (points_per_touchdown * brayden_gavin_touchdowns) = 14 :=
  by sorry

end points_difference_l1355_135502


namespace evaluate_expression_l1355_135578

def x : ℚ := 1 / 4
def y : ℚ := 1 / 3
def z : ℚ := 12

theorem evaluate_expression : x^3 * y^4 * z = 1 / 432 := 
by
  sorry

end evaluate_expression_l1355_135578


namespace number_of_chickens_free_ranging_l1355_135554

-- Defining the conditions
def chickens_in_coop : ℕ := 14
def chickens_in_run (coop_chickens : ℕ) : ℕ := 2 * coop_chickens
def chickens_free_ranging (run_chickens : ℕ) : ℕ := 2 * run_chickens - 4

-- Proving the number of chickens free ranging
theorem number_of_chickens_free_ranging : chickens_free_ranging (chickens_in_run chickens_in_coop) = 52 := by
  -- Lean will be able to infer
  sorry  -- proof is not required

end number_of_chickens_free_ranging_l1355_135554


namespace plane_equation_of_points_l1355_135556

theorem plane_equation_of_points :
  ∃ A B C D : ℤ, A > 0 ∧ Int.gcd (Int.gcd (Int.gcd A (Int.natAbs B)) (Int.natAbs C)) (Int.natAbs D) = 1 ∧
  ∀ x y z : ℤ, (15 * x + 7 * y + 17 * z - 26 = 0) ↔
  (A * x + B * y + C * z + D = 0) :=
by
  sorry

end plane_equation_of_points_l1355_135556


namespace salmon_trip_l1355_135558

theorem salmon_trip (male_female_sum : 712261 + 259378 = 971639) : 
  712261 + 259378 = 971639 := 
by 
  exact male_female_sum

end salmon_trip_l1355_135558


namespace complement_correct_l1355_135525

open Set

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 3, 5}

theorem complement_correct : (U \ A) = {2, 4} := by
  sorry

end complement_correct_l1355_135525


namespace ratio_of_areas_l1355_135580

theorem ratio_of_areas (b : ℝ) (h1 : 0 < b) (h2 : b < 4) 
  (h3 : (9 : ℝ) / 25 = (4 - b) / b * (4 : ℝ)) : b = 2.5 := 
sorry

end ratio_of_areas_l1355_135580


namespace simplify_and_evaluate_expression_l1355_135573

theorem simplify_and_evaluate_expression (x : ℝ) (h : x = 1/2) : x^2 * (x - 1) - x * (x^2 + x - 1) = 0 := by
  sorry

end simplify_and_evaluate_expression_l1355_135573


namespace cone_cylinder_volume_ratio_l1355_135579

theorem cone_cylinder_volume_ratio (h_cyl r_cyl: ℝ) (h_cone: ℝ) :
  h_cyl = 10 → r_cyl = 5 → h_cone = 5 →
  (1/3 * (Real.pi * r_cyl^2 * h_cone)) / (Real.pi * r_cyl^2 * h_cyl) = 1/6 :=
by
  intros h_cyl_eq r_cyl_eq h_cone_eq
  rw [h_cyl_eq, r_cyl_eq, h_cone_eq]
  sorry

end cone_cylinder_volume_ratio_l1355_135579


namespace even_function_a_value_l1355_135511

def f (x a : ℝ) : ℝ := (x + 1) * (x - a)

theorem even_function_a_value (a : ℝ) (h : ∀ x : ℝ, f x a = f (-x) a) : a = 1 := by
  sorry

end even_function_a_value_l1355_135511


namespace A_inter_complement_B_eq_l1355_135519

-- Define set A
def set_A : Set ℝ := {x | -3 < x ∧ x < 6}

-- Define set B
def set_B : Set ℝ := {x | 2 < x ∧ x < 7}

-- Define the complement of set B in the real numbers
def complement_B : Set ℝ := {x | x ≤ 2 ∨ x ≥ 7}

-- Define the intersection of set A with the complement of set B
def A_inter_complement_B : Set ℝ := set_A ∩ complement_B

-- Stating the theorem to prove
theorem A_inter_complement_B_eq : A_inter_complement_B = {x | -3 < x ∧ x ≤ 2} :=
by
  -- Proof goes here
  sorry

end A_inter_complement_B_eq_l1355_135519


namespace find_m_value_l1355_135513

theorem find_m_value (x m : ℝ)
  (h1 : -3 * x = -5 * x + 4)
  (h2 : m^x - 9 = 0) :
  m = 3 ∨ m = -3 := 
sorry

end find_m_value_l1355_135513


namespace grid_values_equal_l1355_135562

theorem grid_values_equal (f : ℤ × ℤ → ℕ) (h : ∀ (i j : ℤ), 
  f (i, j) = 1 / 4 * (f (i + 1, j) + f (i - 1, j) + f (i, j + 1) + f (i, j - 1))) :
  ∀ (i j i' j' : ℤ), f (i, j) = f (i', j') :=
by
  sorry

end grid_values_equal_l1355_135562


namespace retailer_profit_percentage_l1355_135552

theorem retailer_profit_percentage
  (cost_price : ℝ)
  (marked_percent : ℝ)
  (discount_percent : ℝ)
  (selling_price : ℝ)
  (marked_price : ℝ)
  (profit_percent : ℝ) :
  marked_percent = 60 →
  discount_percent = 25 →
  marked_price = cost_price * (1 + marked_percent / 100) →
  selling_price = marked_price * (1 - discount_percent / 100) →
  profit_percent = ((selling_price - cost_price) / cost_price) * 100 →
  profit_percent = 20 :=
by
  sorry

end retailer_profit_percentage_l1355_135552


namespace cats_weigh_more_by_5_kg_l1355_135584

def puppies_weight (num_puppies : ℕ) (weight_per_puppy : ℝ) : ℝ :=
  num_puppies * weight_per_puppy

def cats_weight (num_cats : ℕ) (weight_per_cat : ℝ) : ℝ :=
  num_cats * weight_per_cat

theorem cats_weigh_more_by_5_kg :
  puppies_weight 4 7.5  = 30 ∧ cats_weight 14 2.5 = 35 → (cats_weight 14 2.5 - puppies_weight 4 7.5 = 5) := 
by
  intro h
  sorry

end cats_weigh_more_by_5_kg_l1355_135584


namespace how_many_children_l1355_135530

-- Definitions based on conditions
def total_spectators : ℕ := 10000
def men : ℕ := 7000
def others : ℕ := total_spectators - men -- women + children
def children_per_woman : ℕ := 5

-- Variables
variable (W C : ℕ)

-- Conditions as Lean equalities
def condition1 : W + C = others := by sorry
def condition2 : C = children_per_woman * W := by sorry

-- Theorem statement to prove the number of children
theorem how_many_children (h1 : W + C = others) (h2 : C = children_per_woman * W) : C = 2500 :=
by sorry

end how_many_children_l1355_135530


namespace consecutive_tree_distance_l1355_135514

theorem consecutive_tree_distance (yard_length : ℕ) (num_trees : ℕ) (distance : ℚ)
  (h1 : yard_length = 520) 
  (h2 : num_trees = 40) :
  distance = yard_length / (num_trees - 1) :=
by
  -- Proof steps would go here
  sorry

end consecutive_tree_distance_l1355_135514


namespace tangent_line_problem_l1355_135587

theorem tangent_line_problem 
  (x1 x2 : ℝ)
  (h1 : (1 / x1) = Real.exp x2)
  (h2 : Real.log x1 - 1 = Real.exp x2 * (1 - x2)) :
  (2 / (x1 - 1) + x2 = -1) :=
by 
  sorry

end tangent_line_problem_l1355_135587


namespace combined_area_of_four_removed_triangles_l1355_135593

noncomputable def combined_area_of_removed_triangles (s x y: ℝ) : Prop :=
  x + y = s ∧ s - 2 * x = 15 ∧ s - 2 * y = 9 ∧
  4 * (1 / 2 * x * y) = 67.5

-- Statement of the problem
theorem combined_area_of_four_removed_triangles (s x y: ℝ) :
  combined_area_of_removed_triangles s x y :=
  by
    sorry

end combined_area_of_four_removed_triangles_l1355_135593


namespace neil_baked_cookies_l1355_135531

theorem neil_baked_cookies (total_cookies : ℕ) (given_to_friend : ℕ) (cookies_left : ℕ)
    (h1 : given_to_friend = (2 / 5) * total_cookies)
    (h2 : cookies_left = (3 / 5) * total_cookies)
    (h3 : cookies_left = 12) : total_cookies = 20 :=
by
  sorry

end neil_baked_cookies_l1355_135531


namespace positive_difference_sum_even_odd_l1355_135565

theorem positive_difference_sum_even_odd :
  let sum_first_n_even (n : ℕ) := 2 * (n * (n + 1)) / 2
  let sum_first_n_odd (n : ℕ) := n * n
  let sum_30_even := sum_first_n_even 30
  let sum_25_odd := sum_first_n_odd 25
  sum_30_even - sum_25_odd = 305 :=
by
  let sum_first_n_even (n : ℕ) := 2 * (n * (n + 1)) / 2
  let sum_first_n_odd (n : ℕ) := n * n
  let sum_30_even := sum_first_n_even 30
  let sum_25_odd := sum_first_n_odd 25
  show sum_30_even - sum_25_odd = 305
  sorry

end positive_difference_sum_even_odd_l1355_135565


namespace tangent_line_circle_p_l1355_135574

theorem tangent_line_circle_p (p : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 + 6 * x + 8 = 0 → (x = -p/2 ∨ y = 0)) → 
  (p = 4 ∨ p = 8) :=
by
  sorry

end tangent_line_circle_p_l1355_135574


namespace angle_A_condition_area_range_condition_l1355_135585

/-- Given a triangle ABC with sides opposite to internal angles A, B, and C labeled as a, b, and c respectively. 
Given the condition a * cos C + sqrt 3 * a * sin C = b + c.
Prove that angle A = π / 3.
-/
theorem angle_A_condition
  (a b c : ℝ) (C : ℝ) (h : a * Real.cos C + Real.sqrt 3 * a * Real.sin C = b + c) :
  A = Real.pi / 3 := sorry
  
/-- Given an acute triangle ABC with b = 2 and angle A = π / 3,
find the range of possible values for the area of the triangle ABC.
-/
theorem area_range_condition
  (a c : ℝ) (A : ℝ) (b : ℝ) (C B : ℝ)
  (h1 : b = 2)
  (h2 : A = Real.pi / 3)
  (h3 : 0 < B) (h4 : B < Real.pi / 2)
  (h5 : 0 < C) (h6 : C < Real.pi / 2)
  (h7 : A + C = 2 * Real.pi / 3) :
  Real.sqrt 3 / 2 < (1 / 2) * a * b * Real.sin C ∧
  (1 / 2) * a * b * Real.sin C < 2 * Real.sqrt 3 := sorry

end angle_A_condition_area_range_condition_l1355_135585


namespace line_passes_through_3_1_l1355_135503

open Classical

noncomputable def line_passes_through_fixed_point (m x y : ℝ) : Prop :=
  (2 * m + 1) * x + (m + 1) * y - 7 * m - 4 = 0

theorem line_passes_through_3_1 (m : ℝ) :
  line_passes_through_fixed_point m 3 1 :=
by
  sorry

end line_passes_through_3_1_l1355_135503


namespace grid_sum_21_proof_l1355_135543

-- Define the condition that the sum of the horizontal and vertical lines are 21
def valid_grid (nums : List ℕ) (x : ℕ) : Prop :=
  nums ≠ [] ∧ (((nums.sum + x) = 42) ∧ (21 + 21 = 42))

-- Define the main theorem to prove x = 7
theorem grid_sum_21_proof (nums : List ℕ) (h : valid_grid nums 7) : 7 ∈ nums :=
  sorry

end grid_sum_21_proof_l1355_135543


namespace seq_2016_2017_l1355_135590

-- Define the sequence a_n
def seq (n : ℕ) : ℚ := sorry

-- Given conditions
axiom a1_cond : seq 1 = 1/2
axiom a2_cond : seq 2 = 1/3
axiom seq_rec : ∀ n : ℕ, seq n * seq (n + 2) = 1

-- The main goal
theorem seq_2016_2017 : seq 2016 + seq 2017 = 7/2 := sorry

end seq_2016_2017_l1355_135590


namespace first_day_exceeding_100_paperclips_l1355_135529

def paperclips_day (k : ℕ) : ℕ := 3 * 2^k

theorem first_day_exceeding_100_paperclips :
  ∃ (k : ℕ), paperclips_day k > 100 ∧ k = 6 := by
  sorry

end first_day_exceeding_100_paperclips_l1355_135529


namespace statement_c_false_l1355_135539

theorem statement_c_false : ¬ ∃ (x y : ℝ), x^2 + y^2 < 0 := by
  sorry

end statement_c_false_l1355_135539


namespace number_of_real_solutions_l1355_135567

noncomputable def greatest_integer (x: ℝ) : ℤ :=
  ⌊x⌋

def equation (x: ℝ) :=
  4 * x^2 - 40 * (greatest_integer x : ℝ) + 51 = 0

theorem number_of_real_solutions : 
  ∃ (x1 x2 x3 x4: ℝ), 
  equation x1 ∧ equation x2 ∧ equation x3 ∧ equation x4 ∧ 
  x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧ x2 ≠ x3 ∧ x2 ≠ x4 ∧ x3 ≠ x4 := 
sorry

end number_of_real_solutions_l1355_135567


namespace percent_increase_bike_helmet_l1355_135536

theorem percent_increase_bike_helmet :
  let old_bike_cost := 160
  let old_helmet_cost := 40
  let bike_increase_rate := 0.05
  let helmet_increase_rate := 0.10
  let new_bike_cost := old_bike_cost * (1 + bike_increase_rate)
  let new_helmet_cost := old_helmet_cost * (1 + helmet_increase_rate)
  let old_total_cost := old_bike_cost + old_helmet_cost
  let new_total_cost := new_bike_cost + new_helmet_cost
  let increase_amount := new_total_cost - old_total_cost
  let percent_increase := (increase_amount / old_total_cost) * 100
  percent_increase = 6 :=
by
  sorry

end percent_increase_bike_helmet_l1355_135536


namespace rate_percent_simple_interest_l1355_135538

theorem rate_percent_simple_interest:
  ∀ (P SI T R : ℝ), SI = 400 → P = 1000 → T = 4 → (SI = P * R * T / 100) → R = 10 :=
by
  intros P SI T R h_si h_p h_t h_formula
  -- Proof skipped
  sorry

end rate_percent_simple_interest_l1355_135538


namespace soda_cost_proof_l1355_135596

-- Define the main facts about the weeds
def weeds_flower_bed : ℕ := 11
def weeds_vegetable_patch : ℕ := 14
def weeds_grass : ℕ := 32 / 2  -- Only half the weeds in the grass

-- Define the earning rate
def earning_per_weed : ℕ := 6

-- Define the total earnings and the remaining money conditions
def total_earnings : ℕ := (weeds_flower_bed + weeds_vegetable_patch + weeds_grass) * earning_per_weed
def remaining_money : ℕ := 147

-- Define the cost of the soda
def cost_of_soda : ℕ := total_earnings - remaining_money

-- Problem statement: Prove that the cost of the soda is 99 cents
theorem soda_cost_proof : cost_of_soda = 99 := by
  sorry

end soda_cost_proof_l1355_135596


namespace cards_sum_l1355_135575

theorem cards_sum (a b c d e f g h : ℕ) 
  (h1 : (a + b) * (c + d) * (e + f) * (g + h) = 330) :
  a + b + c + d + e + f + g + h = 21 :=
by
  sorry

end cards_sum_l1355_135575


namespace susan_gave_sean_8_apples_l1355_135540

theorem susan_gave_sean_8_apples (initial_apples total_apples apples_given : ℕ) 
  (h1 : initial_apples = 9)
  (h2 : total_apples = 17)
  (h3 : apples_given = total_apples - initial_apples) : 
  apples_given = 8 :=
by
  sorry

end susan_gave_sean_8_apples_l1355_135540


namespace chocolates_difference_l1355_135560

/-!
We are given that:
- Robert ate 10 chocolates
- Nickel ate 5 chocolates

We need to prove that Robert ate 5 more chocolates than Nickel.
-/

def robert_chocolates := 10
def nickel_chocolates := 5

theorem chocolates_difference : robert_chocolates - nickel_chocolates = 5 :=
by
  -- Proof is omitted as per instructions
  sorry

end chocolates_difference_l1355_135560


namespace inverse_of_f_at_neg2_l1355_135521

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x + 1

-- Define the property of the inverse function we need to prove
theorem inverse_of_f_at_neg2 : f (-(3/2)) = -2 :=
  by
    -- Placeholder for the proof
    sorry

end inverse_of_f_at_neg2_l1355_135521


namespace intersection_empty_condition_l1355_135518

-- Define the sets M and N under the given conditions
def M : Set (ℝ × ℝ) := { p | p.1^2 + 2 * p.2^2 = 3 }

def N (m b : ℝ) : Set (ℝ × ℝ) := { p | p.2 = m * p.1 + b }

-- The theorem that we need to prove based on the problem statement
theorem intersection_empty_condition (b : ℝ) :
  (∀ m : ℝ, M ∩ N m b = ∅) ↔ (b^2 > 6 * m^2 + 2) := sorry

end intersection_empty_condition_l1355_135518


namespace Eva_is_6_l1355_135581

def ages : Set ℕ := {2, 4, 6, 8, 10}

def conditions : Prop :=
  ∃ a b, a ∈ ages ∧ b ∈ ages ∧ a + b = 12 ∧
  b ≠ 2 ∧ b ≠ 10 ∧ a ≠ 2 ∧ a ≠ 10 ∧
  (∃ c d, c ∈ ages ∧ d ∈ ages ∧ c = 2 ∧ d = 10 ∧
           (∃ e, e ∈ ages ∧ e = 4 ∧
           ∃ eva, eva ∈ ages ∧ eva ≠ 2 ∧ eva ≠ 4 ∧ eva ≠ 8 ∧ eva ≠ 10 ∧ eva = 6))

theorem Eva_is_6 (h : conditions) : ∃ eva, eva ∈ ages ∧ eva = 6 := sorry

end Eva_is_6_l1355_135581


namespace initial_green_marbles_l1355_135506

theorem initial_green_marbles (m g' : ℕ) (h_m : m = 23) (h_g' : g' = 9) : (g' + m = 32) :=
by
  subst h_m
  subst h_g'
  rfl

end initial_green_marbles_l1355_135506


namespace union_A_B_l1355_135569

def setA : Set ℝ := {x | (x + 1) * (x - 2) < 0}
def setB : Set ℝ := {x | 1 < x ∧ x ≤ 3}

theorem union_A_B : setA ∪ setB = {x | -1 < x ∧ x ≤ 3} :=
by
  sorry

end union_A_B_l1355_135569


namespace sugar_amount_l1355_135528

variables (S F B : ℝ)

-- Conditions
def condition1 : Prop := S / F = 5 / 2
def condition2 : Prop := F / B = 10 / 1
def condition3 : Prop := F / (B + 60) = 8 / 1

-- Theorem to prove
theorem sugar_amount (h1 : condition1 S F) (h2 : condition2 F B) (h3 : condition3 F B) : S = 6000 :=
sorry

end sugar_amount_l1355_135528


namespace opposite_difference_five_times_l1355_135597

variable (a b : ℤ) -- Using integers for this example

theorem opposite_difference_five_times (a b : ℤ) : (-a - 5 * b) = -(a) - (5 * b) := 
by
  -- The proof details would be filled in here
  sorry

end opposite_difference_five_times_l1355_135597


namespace total_money_calculation_l1355_135546

theorem total_money_calculation (N50 N500 Total_money : ℕ) 
( h₁ : N50 = 37 ) 
( h₂ : N50 + N500 = 54 ) :
Total_money = N50 * 50 + N500 * 500 ↔ Total_money = 10350 := 
by 
  sorry

end total_money_calculation_l1355_135546


namespace rectangle_length_difference_l1355_135508

variable (s l w : ℝ)

-- Conditions
def condition1 : Prop := 2 * (l + w) = 4 * s + 4
def condition2 : Prop := w = s - 2

-- Theorem to prove
theorem rectangle_length_difference
  (s l w : ℝ)
  (h1 : condition1 s l w)
  (h2 : condition2 s w) : l = s + 4 :=
by
sorry

end rectangle_length_difference_l1355_135508


namespace product_of_equal_numbers_l1355_135501

theorem product_of_equal_numbers (a b c d : ℕ) (h1 : (a + b + c + d) / 4 = 20) (h2 : a = 12) (h3 : b = 22) 
(h4 : c = d) : c * d = 529 := 
by
  sorry

end product_of_equal_numbers_l1355_135501


namespace matrix_pow_sub_l1355_135515

open Matrix

noncomputable def B : Matrix (Fin 2) (Fin 2) ℚ := !![3, 4; 0, 2]

theorem matrix_pow_sub : 
  B^10 - 3 • B^9 = !![0, 4; 0, -1] := 
by
  sorry

end matrix_pow_sub_l1355_135515


namespace john_twice_james_l1355_135564

def john_age : ℕ := 39
def years_ago : ℕ := 3
def years_future : ℕ := 6
def age_difference : ℕ := 4

theorem john_twice_james {J : ℕ} (h : 39 - years_ago = 2 * (J + years_future)) : 
  (J + age_difference = 16) :=
by
  sorry  -- Proof steps here

end john_twice_james_l1355_135564


namespace sum_x_y_z_l1355_135551
open Real

theorem sum_x_y_z (a b : ℝ) (h1 : a / b = 98 / 63) (x y z : ℕ) (h2 : (sqrt a) / (sqrt b) = (x * sqrt y) / z) : x + y + z = 18 := 
by
  sorry

end sum_x_y_z_l1355_135551


namespace max_b_plus_c_triangle_l1355_135500

theorem max_b_plus_c_triangle (a b c : ℝ) (A : ℝ) 
  (h₁ : a = 4) (h₂ : A = Real.pi / 3) (h₃ : a^2 = b^2 + c^2 - 2 * b * c * Real.cos A) :
  b + c ≤ 8 :=
by
  -- sorry is added to skip the proof for now.
  sorry

end max_b_plus_c_triangle_l1355_135500


namespace fraction_of_earth_surface_inhabitable_l1355_135542

theorem fraction_of_earth_surface_inhabitable (f_land : ℚ) (f_inhabitable_land : ℚ)
  (h1 : f_land = 1 / 3)
  (h2 : f_inhabitable_land = 2 / 3) :
  f_land * f_inhabitable_land = 2 / 9 :=
by
  sorry

end fraction_of_earth_surface_inhabitable_l1355_135542


namespace scientific_notation_11580000_l1355_135549

theorem scientific_notation_11580000 :
  11580000 = 1.158 * 10^7 :=
sorry

end scientific_notation_11580000_l1355_135549


namespace weights_in_pile_l1355_135541

theorem weights_in_pile (a b c : ℕ) (h1 : a + b + c = 100) (h2 : a + 10 * b + 50 * c = 500) : 
  a = 60 ∧ b = 39 ∧ c = 1 :=
sorry

end weights_in_pile_l1355_135541


namespace terminal_zeros_of_product_l1355_135572

noncomputable def prime_factors (n : ℕ) : List (ℕ × ℕ) := sorry

theorem terminal_zeros_of_product (n m : ℕ) (hn : prime_factors n = [(2, 1), (5, 2)])
 (hm : prime_factors m = [(2, 3), (3, 2), (5, 1)]) : 
  (∃ k, n * m = 10^k) ∧ k = 3 :=
by {
  sorry
}

end terminal_zeros_of_product_l1355_135572


namespace total_toothpicks_correct_l1355_135535

def number_of_horizontal_toothpicks (height : ℕ) (width : ℕ) : ℕ :=
(height + 1) * width

def number_of_vertical_toothpicks (height : ℕ) (width : ℕ) : ℕ :=
(height) * (width + 1)

def total_toothpicks (height : ℕ) (width : ℕ) : ℕ :=
number_of_horizontal_toothpicks height width + number_of_vertical_toothpicks height width

theorem total_toothpicks_correct:
  total_toothpicks 30 15 = 945 :=
by
  sorry

end total_toothpicks_correct_l1355_135535


namespace second_half_takes_200_percent_longer_l1355_135544

noncomputable def time_take (distance : ℕ) (speed : ℕ) : ℕ := distance / speed

variable (total_distance : ℕ := 640)
variable (first_half_speed : ℕ := 80)
variable (average_speed : ℕ := 40)

theorem second_half_takes_200_percent_longer :
  let first_half_distance := total_distance / 2;
  let first_half_time := time_take first_half_distance first_half_speed;
  let total_time := time_take total_distance average_speed;
  let second_half_time := total_time - first_half_time;
  let time_increase := second_half_time - first_half_time;
  let percentage_increase := (time_increase * 100) / first_half_time;
  percentage_increase = 200 :=
by
  sorry

end second_half_takes_200_percent_longer_l1355_135544


namespace cistern_fill_time_l1355_135517

-- Define the problem conditions
def pipe_p_fill_time : ℕ := 10
def pipe_q_fill_time : ℕ := 15
def joint_filling_time : ℕ := 2
def remaining_fill_time : ℕ := 10 -- This is the answer we need to prove

-- Prove that the remaining fill time is equal to 10 minutes
theorem cistern_fill_time :
  (joint_filling_time * (1 / pipe_p_fill_time + 1 / pipe_q_fill_time) + (remaining_fill_time / pipe_q_fill_time)) = 1 :=
sorry

end cistern_fill_time_l1355_135517


namespace determine_peter_and_liar_l1355_135537

structure Brothers where
  names : Fin 2 → String
  tells_truth : Fin 2 → Bool -- true if the brother tells the truth, false if lies
  (unique_truth_teller : ∃! (i : Fin 2), tells_truth i)
  (one_is_peter : ∃ (i : Fin 2), names i = "Péter")

theorem determine_peter_and_liar (B : Brothers) : 
  ∃ (peter liar : Fin 2), B.names peter = "Péter" ∧ B.tells_truth liar = false ∧
    ∀ (p q : Fin 2), B.names p = "Péter" → B.tells_truth q = false → p = peter ∧ q = liar :=
by
  sorry

end determine_peter_and_liar_l1355_135537


namespace deepak_speed_proof_l1355_135577

noncomputable def deepak_speed (circumference : ℝ) (meeting_time : ℝ) (wife_speed_kmh : ℝ) : ℝ :=
  let wife_speed_mpm := wife_speed_kmh * 1000 / 60
  let wife_distance := wife_speed_mpm * meeting_time
  let deepak_speed_mpm := ((circumference - wife_distance) / meeting_time)
  deepak_speed_mpm * 60 / 1000

theorem deepak_speed_proof :
  deepak_speed 726 5.28 3.75 = 4.5054 :=
by
  -- The functions and definitions used here come from the problem statement
  -- Conditions:
  -- circumference = 726
  -- meeting_time = 5.28 minutes
  -- wife_speed_kmh = 3.75 km/hr
  sorry

end deepak_speed_proof_l1355_135577


namespace abs_neg_three_l1355_135524

theorem abs_neg_three : abs (-3) = 3 :=
sorry

end abs_neg_three_l1355_135524


namespace exists_sequence_of_ten_numbers_l1355_135591

theorem exists_sequence_of_ten_numbers :
  ∃ a : Fin 10 → ℝ,
    (∀ i : Fin 6,    a i + a ⟨i.1 + 1, sorry⟩ + a ⟨i.1 + 2, sorry⟩ + a ⟨i.1 + 3, sorry⟩ + a ⟨i.1 + 4, sorry⟩ > 0) ∧
    (∀ j : Fin 4, a j + a ⟨j.1 + 1, sorry⟩ + a ⟨j.1 + 2, sorry⟩ + a ⟨j.1 + 3, sorry⟩ + a ⟨j.1 + 4, sorry⟩ + a ⟨j.1 + 5, sorry⟩ + a ⟨j.1 + 6, sorry⟩ < 0) :=
sorry

end exists_sequence_of_ten_numbers_l1355_135591


namespace fixed_monthly_fee_l1355_135527

theorem fixed_monthly_fee (x y : ℝ)
  (h1 : x + y = 15.80)
  (h2 : x + 3 * y = 28.62) :
  x = 9.39 :=
sorry

end fixed_monthly_fee_l1355_135527


namespace distance_AB_l1355_135592

open Classical Real

theorem distance_AB (A B F : ℝ × ℝ) (hA : A.2 ^ 2 = 4 * A.1) (B_def : B = (3, 0)) (F_def : F = (1, 0)) (AF_eq_BF : dist A F = dist B F) : dist A B = 2 * Real.sqrt 2 := by
  sorry

end distance_AB_l1355_135592


namespace jake_split_shots_l1355_135561

theorem jake_split_shots (shot_volume : ℝ) (purity : ℝ) (alcohol_consumed : ℝ) 
    (h1 : shot_volume = 1.5) (h2 : purity = 0.50) (h3 : alcohol_consumed = 3) : 
    2 * (alcohol_consumed / (purity * shot_volume)) = 8 :=
by
  sorry

end jake_split_shots_l1355_135561


namespace sugar_cone_count_l1355_135571

theorem sugar_cone_count (ratio_sugar_waffle : ℕ → ℕ → Prop) (sugar_waffle_ratio : ratio_sugar_waffle 5 4) 
(w : ℕ) (h_w : w = 36) : ∃ s : ℕ, ratio_sugar_waffle s w ∧ s = 45 :=
by
  sorry

end sugar_cone_count_l1355_135571


namespace genuine_items_count_l1355_135512

def total_purses : ℕ := 26
def total_handbags : ℕ := 24
def fake_purses : ℕ := total_purses / 2
def fake_handbags : ℕ := total_handbags / 4
def genuine_purses : ℕ := total_purses - fake_purses
def genuine_handbags : ℕ := total_handbags - fake_handbags

theorem genuine_items_count : genuine_purses + genuine_handbags = 31 := by
  sorry

end genuine_items_count_l1355_135512


namespace largest_value_of_x_not_defined_l1355_135595

noncomputable def quadratic_formula (a b c : ℝ) : (ℝ × ℝ) :=
  let discriminant := b*b - 4*a*c
  let sqrt_discriminant := Real.sqrt discriminant
  let x1 := (-b + sqrt_discriminant) / (2*a)
  let x2 := (-b - sqrt_discriminant) / (2*a)
  (x1, x2)

noncomputable def largest_root : ℝ :=
  let (x1, x2) := quadratic_formula 4 (-81) 49
  if x1 > x2 then x1 else x2

theorem largest_value_of_x_not_defined :
  largest_root = 19.6255 :=
by
  sorry

end largest_value_of_x_not_defined_l1355_135595


namespace find_number_of_flowers_l1355_135547
open Nat

theorem find_number_of_flowers (F : ℕ) (h_candles : choose 4 2 = 6) (h_groupings : 6 * choose F 8 = 54) : F = 9 :=
sorry

end find_number_of_flowers_l1355_135547


namespace solve_inequality_l1355_135570

theorem solve_inequality (x : ℝ) :
  (abs ((6 - x) / 4) < 3) ∧ (2 ≤ x) ↔ (2 ≤ x) ∧ (x < 18) := 
by
  sorry

end solve_inequality_l1355_135570


namespace total_valid_votes_l1355_135533

theorem total_valid_votes (V : ℝ) (h1 : 0.70 * V - 0.30 * V = 176) : V = 440 :=
by sorry

end total_valid_votes_l1355_135533


namespace quiz_score_of_dropped_student_l1355_135548

theorem quiz_score_of_dropped_student 
    (avg_all : ℝ) (num_all : ℕ) (new_avg_remaining : ℝ) (num_remaining : ℕ)
    (total_all : ℝ := num_all * avg_all) (total_remaining : ℝ := num_remaining * new_avg_remaining) :
    avg_all = 61.5 → num_all = 16 → new_avg_remaining = 64 → num_remaining = 15 → (total_all - total_remaining = 24) :=
by
  intros h_avg_all h_num_all h_new_avg_remaining h_num_remaining
  rw [h_avg_all, h_new_avg_remaining, h_num_all, h_num_remaining]
  sorry

end quiz_score_of_dropped_student_l1355_135548


namespace minimum_value_ineq_l1355_135555

variable (m n : ℝ)

noncomputable def minimum_value := (1 / (2 * m)) + (1 / n)

theorem minimum_value_ineq (h1 : m > 0) (h2 : n > 0) (h3 : m + 2 * n = 1) : minimum_value m n = 9 / 2 := 
sorry

end minimum_value_ineq_l1355_135555


namespace not_possible_to_color_plane_l1355_135520

theorem not_possible_to_color_plane :
  ¬ ∃ (color : ℕ → ℕ × ℕ → ℕ) (c : ℕ), 
    (c = 2016) ∧
    (∀ (A B C : ℕ × ℕ), (A ≠ B ∧ B ≠ C ∧ C ≠ A) → 
                        (color c A = color c B) ∨ (color c B = color c C) ∨ (color c C = color c A)) :=
by
  sorry

end not_possible_to_color_plane_l1355_135520


namespace geometric_seq_not_sufficient_necessary_l1355_135553

theorem geometric_seq_not_sufficient_necessary (a_n : ℕ → ℝ) (q : ℝ) 
  (h1 : ∀ n, a_n (n+1) = a_n n * q) : 
  ¬ ((∃ q > 1, ∀ n, a_n (n+1) > a_n n) ∧ (∀ q > 1, ∀ n, a_n (n+1) > a_n n)) := 
sorry

end geometric_seq_not_sufficient_necessary_l1355_135553


namespace find_fourth_number_in_proportion_l1355_135588

-- Define the given conditions
def x : ℝ := 0.39999999999999997
def proportion (y : ℝ) := 0.60 / x = 6 / y

-- State the theorem to be proven
theorem find_fourth_number_in_proportion :
  proportion y → y = 4 :=
by
  intro h
  sorry

end find_fourth_number_in_proportion_l1355_135588


namespace solve_for_x_l1355_135589

theorem solve_for_x (x : ℝ) (h : (x / 5) + 3 = 4) : x = 5 :=
by
  sorry

end solve_for_x_l1355_135589


namespace lcm_factor_l1355_135583

-- Define the variables and conditions
variables (A B H L x : ℕ)
variable (hcf_23 : Nat.gcd A B = 23)
variable (larger_number_391 : A = 391)
variable (lcm_hcf_mult_factors : L = Nat.lcm A B)
variable (lcm_factors : L = 23 * x * 17)

-- The proof statement
theorem lcm_factor (hcf_23 : Nat.gcd A B = 23) (larger_number_391 : A = 391) (lcm_hcf_mult_factors : L = Nat.lcm A B) (lcm_factors : L = 23 * x * 17) :
  x = 17 :=
sorry

end lcm_factor_l1355_135583


namespace dot_product_focus_hyperbola_l1355_135526

-- Definitions related to the problem of the hyperbola
def hyperbola (x y : ℝ) : Prop := (x^2 / 3) - y^2 = 1

def is_focus (c : ℝ) (x y : ℝ) : Prop := (x = c ∧ y = 0) ∨ (x = -c ∧ y = 0)

-- Problem conditions
def point_on_hyperbola (p : ℝ × ℝ) : Prop := hyperbola p.1 p.2

def triangle_area (a b c : ℝ × ℝ) (area : ℝ) : Prop :=
  0.5 * (a.1 * (b.2 - c.2) + b.1 * (c.2 - a.2) + c.1 * (a.2 - b.2)) = area

def foci_of_hyperbola : (ℝ × ℝ) × (ℝ × ℝ) := ((2, 0), (-2, 0))

-- Main statement to prove
theorem dot_product_focus_hyperbola
  (m n : ℝ)
  (hP : point_on_hyperbola (m, n))
  (hArea : triangle_area (2, 0) (m, n) (-2, 0) 2) :
  ((-2 - m) * (2 - m) + (-n) * (-n)) = 3 :=
sorry

end dot_product_focus_hyperbola_l1355_135526


namespace candy_left_l1355_135559

-- Define the given conditions
def KatieCandy : ℕ := 8
def SisterCandy : ℕ := 23
def AteCandy : ℕ := 8

-- The theorem stating the total number of candy left
theorem candy_left (k : ℕ) (s : ℕ) (e : ℕ) (hk : k = KatieCandy) (hs : s = SisterCandy) (he : e = AteCandy) : 
  (k + s) - e = 23 :=
by
  -- (Proof will be inserted here, but we include a placeholder "sorry" for now)
  sorry

end candy_left_l1355_135559


namespace parabola_sequence_l1355_135522

theorem parabola_sequence (m: ℝ) (n: ℕ):
  (∀ t s: ℝ, t * s = -1/4) →
  (∀ x y: ℝ, y^2 = (1/(3^n)) * m * (x - (m / 4) * (1 - (1/(3^n))))) :=
sorry

end parabola_sequence_l1355_135522


namespace correct_statement_about_Digital_Earth_l1355_135504

-- Definitions of the statements
def statement_A : Prop :=
  "Digital Earth is a reflection of the real Earth through digital means" = "Correct statement about Digital Earth"

def statement_B : Prop :=
  "Digital Earth is an extension of GIS technology" = "Correct statement about Digital Earth"

def statement_C : Prop :=
  "Digital Earth can only achieve global information sharing through the internet" = "Correct statement about Digital Earth"

def statement_D : Prop :=
  "The core idea of Digital Earth is to use digital means to uniformly address Earth's issues" = "Correct statement about Digital Earth"

-- Theorem that needs to be proved 
theorem correct_statement_about_Digital_Earth : statement_C :=
by 
  sorry

end correct_statement_about_Digital_Earth_l1355_135504


namespace choose_7_starters_with_at_least_one_quadruplet_l1355_135563

-- Given conditions
variable (n : ℕ := 18) -- total players
variable (k : ℕ := 7)  -- number of starters
variable (q : ℕ := 4)  -- number of quadruplets

-- Lean statement
theorem choose_7_starters_with_at_least_one_quadruplet 
  (h : n = 18) 
  (h1 : k = 7) 
  (h2 : q = 4) :
  (Nat.choose 18 7 - Nat.choose 14 7) = 28392 :=
by
  sorry

end choose_7_starters_with_at_least_one_quadruplet_l1355_135563


namespace solve_inequality_l1355_135586

theorem solve_inequality (x : ℝ) : 2 * x + 6 > 5 * x - 3 → x < 3 :=
by
  -- Proof steps would go here
  sorry

end solve_inequality_l1355_135586


namespace sequence_term_l1355_135516

noncomputable def a_n (n : ℕ) : ℕ :=
  if n = 1 then 3 else 4 * n - 2

def S_n (n : ℕ) : ℕ :=
  2 * n^2 + 1

theorem sequence_term (n : ℕ) : a_n n = if n = 1 then S_n 1 else S_n n - S_n (n - 1) :=
by 
  sorry

end sequence_term_l1355_135516


namespace father_age_when_sum_100_l1355_135566

/-- Given the current ages of the mother and father, prove that the father's age will be 51 years old when the sum of their ages is 100. -/
theorem father_age_when_sum_100 (M F : ℕ) (hM : M = 42) (hF : F = 44) :
  ∃ X : ℕ, (M + X) + (F + X) = 100 ∧ F + X = 51 :=
by
  sorry

end father_age_when_sum_100_l1355_135566


namespace largest_p_plus_q_l1355_135599

-- All required conditions restated as Assumptions
def triangle {R : Type*} [LinearOrderedField R] (p q : R) : Prop :=
  let B : R × R := (10, 15)
  let C : R × R := (25, 15)
  let A : R × R := (p, q)
  let M : R × R := ((B.1 + C.1) / 2, (B.2 + C.2) / 2)
  let area : R := (1 / 2) * abs ((A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)))
  let median_slope : R := (A.2 - M.2) / (A.1 - M.1)
  area = 100 ∧ median_slope = -3

-- Statement to be proven
theorem largest_p_plus_q {R : Type*} [LinearOrderedField R] (p q : R) :
  triangle p q → p + q = 70 / 3 :=
by
  sorry

end largest_p_plus_q_l1355_135599


namespace otimes_calc_1_otimes_calc_2_otimes_calc_3_l1355_135523

def otimes (a b : Int) : Int :=
  a^2 - Int.natAbs b

theorem otimes_calc_1 : otimes (-2) 3 = 1 :=
by
  sorry

theorem otimes_calc_2 : otimes 5 (-4) = 21 :=
by
  sorry

theorem otimes_calc_3 : otimes (-3) (-1) = 8 :=
by
  sorry

end otimes_calc_1_otimes_calc_2_otimes_calc_3_l1355_135523


namespace track_width_eight_l1355_135568

theorem track_width_eight (r1 r2 : ℝ) (h : 2 * Real.pi * r1 - 2 * Real.pi * r2 = 16 * Real.pi) : r1 - r2 = 8 := 
sorry

end track_width_eight_l1355_135568


namespace problem_statement_l1355_135557

variable (x : ℝ)
def A := ({-3, x^2, x + 1} : Set ℝ)
def B := ({x - 3, 2 * x - 1, x^2 + 1} : Set ℝ)

theorem problem_statement (hx : A x ∩ B x = {-3}) : 
  x = -1 ∧ A x ∪ B x = ({-4, -3, 0, 1, 2} : Set ℝ) :=
by
  sorry

end problem_statement_l1355_135557


namespace sugar_percentage_in_new_solution_l1355_135505

open Real

noncomputable def original_volume : ℝ := 450
noncomputable def original_sugar_percentage : ℝ := 20 / 100
noncomputable def added_sugar : ℝ := 7.5
noncomputable def added_water : ℝ := 20
noncomputable def added_kola : ℝ := 8.1
noncomputable def added_flavoring : ℝ := 2.3

noncomputable def original_sugar_amount : ℝ := original_volume * original_sugar_percentage
noncomputable def total_sugar_amount : ℝ := original_sugar_amount + added_sugar
noncomputable def new_total_volume : ℝ := original_volume + added_water + added_kola + added_flavoring + added_sugar
noncomputable def new_sugar_percentage : ℝ := (total_sugar_amount / new_total_volume) * 100

theorem sugar_percentage_in_new_solution : abs (new_sugar_percentage - 19.97) < 0.01 := sorry

end sugar_percentage_in_new_solution_l1355_135505


namespace xiao_ming_completion_days_l1355_135582

/-
  Conditions:
  1. The total number of pages is 960.
  2. The planned number of days to finish the book is 20.
  3. Xiao Ming actually read 12 more pages per day than planned.

  Question:
  How many days did it actually take Xiao Ming to finish the book?

  Answer:
  The actual number of days to finish the book is 16 days.
-/

open Nat

theorem xiao_ming_completion_days :
  let total_pages := 960
  let planned_days := 20
  let additional_pages_per_day := 12
  let planned_pages_per_day := total_pages / planned_days
  let actual_pages_per_day := planned_pages_per_day + additional_pages_per_day
  let actual_days := total_pages / actual_pages_per_day
  actual_days = 16 :=
by
  let total_pages := 960
  let planned_days := 20
  let additional_pages_per_day := 12
  let planned_pages_per_day := total_pages / planned_days
  let actual_pages_per_day := planned_pages_per_day + additional_pages_per_day
  let actual_days := total_pages / actual_pages_per_day
  show actual_days = 16
  sorry

end xiao_ming_completion_days_l1355_135582


namespace probability_of_finding_last_defective_product_on_fourth_inspection_l1355_135594

theorem probability_of_finding_last_defective_product_on_fourth_inspection :
  let total_products := 6
  let qualified_products := 4
  let defective_products := 2
  let probability := (4 / 6) * (3 / 5) * (2 / 4) * (1 / 3) + (4 / 6) * (2 / 5) * (3 / 4) * (1 / 3) + (2 / 6) * (4 / 5) * (3 / 4) * (1 / 3)
  probability = 1 / 5 :=
by
  let total_products := 6
  let qualified_products := 4
  let defective_products := 2
  let probability := (4 / 6) * (3 / 5) * (2 / 4) * (1 / 3) + (4 / 6) * (2 / 5) * (3 / 4) * (1 / 3) + (2 / 6) * (4 / 5) * (3 / 4) * (1 / 3)
  have : probability = 1 / 5 := sorry
  exact this

end probability_of_finding_last_defective_product_on_fourth_inspection_l1355_135594


namespace sum_of_reciprocals_of_roots_l1355_135550

theorem sum_of_reciprocals_of_roots {r1 r2 : ℚ} (h1 : r1 + r2 = 15) (h2 : r1 * r2 = 6) :
  (1 / r1 + 1 / r2) = 5 / 2 := 
by sorry

end sum_of_reciprocals_of_roots_l1355_135550


namespace additional_money_needed_l1355_135509

-- Define the initial conditions as assumptions
def initial_bales : ℕ := 15
def previous_cost_per_bale : ℕ := 20
def multiplier : ℕ := 3
def new_cost_per_bale : ℕ := 27

-- Define the problem statement
theorem additional_money_needed :
  let initial_cost := initial_bales * previous_cost_per_bale 
  let new_bales := initial_bales * multiplier
  let new_cost := new_bales * new_cost_per_bale
  new_cost - initial_cost = 915 :=
by
  sorry

end additional_money_needed_l1355_135509
