import Mathlib

namespace number_of_odd_palindromes_l120_120111

def is_palindrome (n : ℕ) : Prop :=
  let d0 := n % 10
  let d1 := (n / 10) % 10
  let d2 := n / 100
  n < 1000 ∧ n >= 100 ∧ d0 = d2

def is_odd (n : ℕ) : Prop :=
  n % 2 = 1

theorem number_of_odd_palindromes : ∃ n : ℕ, is_palindrome n ∧ is_odd n → n = 50 :=
by
  sorry

end number_of_odd_palindromes_l120_120111


namespace problem_1_problem_2_l120_120782

noncomputable def vec_a (x : ℝ) : ℝ × ℝ := (Real.sin (2 * x), Real.cos (2 * x))
noncomputable def vec_b : ℝ × ℝ := (1 / 2, Real.sqrt 3 / 2)
noncomputable def f (x : ℝ) : ℝ := (vec_a x).fst * vec_b.fst + (vec_a x).snd * vec_b.snd + 2

theorem problem_1 (x : ℝ) : x ∈ Set.Icc (k * Real.pi - (5 / 12) * Real.pi) (k * Real.pi + (1 / 12) * Real.pi) → ∃ k : ℤ, ∀ x : ℝ, f (x) = Real.sin (2 * x + (1 / 3) * Real.pi) + 2 :=
sorry

theorem problem_2 (x : ℝ) : x ∈ Set.Icc (π / 6) (2 * π / 3) → f (π / 6) = (Real.sqrt 3 / 2) + 2 ∧ f (7 * π / 12) = 1 :=
sorry

end problem_1_problem_2_l120_120782


namespace lines_intersect_at_same_point_l120_120161

theorem lines_intersect_at_same_point (m k : ℝ) :
  (∃ x y : ℝ, y = 3 * x + 5 ∧ y = -4 * x + m ∧ y = 2 * x + k) ↔ k = (m + 30) / 7 :=
by {
  sorry -- proof not required, only statement.
}

end lines_intersect_at_same_point_l120_120161


namespace quiz_answer_key_count_l120_120407

theorem quiz_answer_key_count :
  ∃ n : ℕ, n = 480 ∧
  (∃ tf_count : ℕ, tf_count = 30 ∧
   (∃ mc_count : ℕ, mc_count = 16 ∧ 
    n = tf_count * mc_count)) :=
    sorry

end quiz_answer_key_count_l120_120407


namespace james_ate_eight_slices_l120_120665

-- Define the conditions
def num_pizzas := 2
def slices_per_pizza := 6
def fraction_james_ate := 2 / 3
def total_slices := num_pizzas * slices_per_pizza

-- Define the statement to prove
theorem james_ate_eight_slices : fraction_james_ate * total_slices = 8 :=
by
  sorry

end james_ate_eight_slices_l120_120665


namespace cone_base_circumference_l120_120935

theorem cone_base_circumference (V : ℝ) (h : ℝ) (C : ℝ) (r : ℝ) :
  V = 18 * Real.pi →
  h = 6 →
  (V = (1 / 3) * Real.pi * r^2 * h) →
  C = 2 * Real.pi * r →
  C = 6 * Real.pi :=
by
  intros h1 h2 h3 h4
  sorry

end cone_base_circumference_l120_120935


namespace coat_price_reduction_l120_120936

theorem coat_price_reduction (original_price reduction_amount : ℝ) (h : original_price = 500) (h_red : reduction_amount = 150) :
  ((reduction_amount / original_price) * 100) = 30 :=
by
  rw [h, h_red]
  norm_num

end coat_price_reduction_l120_120936


namespace village_population_l120_120916

-- Defining the variables and the condition
variable (P : ℝ) (h : 0.9 * P = 36000)

-- Statement of the theorem to prove
theorem village_population : P = 40000 :=
by sorry

end village_population_l120_120916


namespace weighted_average_salary_l120_120378

theorem weighted_average_salary :
  let num_managers := 9
  let salary_managers := 4500
  let num_associates := 18
  let salary_associates := 3500
  let num_lead_cashiers := 6
  let salary_lead_cashiers := 3000
  let num_sales_representatives := 45
  let salary_sales_representatives := 2500
  let total_salaries := 
    (num_managers * salary_managers) +
    (num_associates * salary_associates) +
    (num_lead_cashiers * salary_lead_cashiers) +
    (num_sales_representatives * salary_sales_representatives)
  let total_employees := 
    num_managers + num_associates + num_lead_cashiers + num_sales_representatives
  let weighted_avg_salary := total_salaries / total_employees
  weighted_avg_salary = 3000 := 
by
  sorry

end weighted_average_salary_l120_120378


namespace gena_encoded_numbers_unique_l120_120803

theorem gena_encoded_numbers_unique : 
  ∃ (B AN AX NO FF d : ℕ), (AN - B = d) ∧ (AX - AN = d) ∧ (NO - AX = d) ∧ (FF - NO = d) ∧ 
  [B, AN, AX, NO, FF] = [5, 12, 19, 26, 33] := sorry

end gena_encoded_numbers_unique_l120_120803


namespace student_scores_correct_answers_l120_120592

variable (c w : ℕ)

theorem student_scores_correct_answers :
  (c + w = 60) ∧ (4 * c - w = 130) → c = 38 :=
by
  intro h
  sorry

end student_scores_correct_answers_l120_120592


namespace bird_families_migration_l120_120721

theorem bird_families_migration 
  (total_families : ℕ)
  (africa_families : ℕ)
  (asia_families : ℕ)
  (south_america_families : ℕ)
  (africa_days : ℕ)
  (asia_days : ℕ)
  (south_america_days : ℕ)
  (migrated_families : ℕ)
  (remaining_families : ℕ)
  (total_migration_time : ℕ)
  (H1 : total_families = 200)
  (H2 : africa_families = 60)
  (H3 : asia_families = 95)
  (H4 : south_america_families = 30)
  (H5 : africa_days = 7)
  (H6 : asia_days = 14)
  (H7 : south_america_days = 10)
  (H8 : migrated_families = africa_families + asia_families + south_america_families)
  (H9 : remaining_families = total_families - migrated_families)
  (H10 : total_migration_time = 
          africa_families * africa_days + 
          asia_families * asia_days + 
          south_america_families * south_america_days) :
  remaining_families = 15 ∧ total_migration_time = 2050 :=
by
  sorry

end bird_families_migration_l120_120721


namespace photograph_perimeter_is_23_l120_120036

noncomputable def photograph_perimeter (w h m : ℝ) : ℝ :=
if (w + 4) * (h + 4) = m ∧ (w + 8) * (h + 8) = m + 94 then 2 * (w + h) else 0

theorem photograph_perimeter_is_23 (w h m : ℝ) 
    (h₁ : (w + 4) * (h + 4) = m) 
    (h₂ : (w + 8) * (h + 8) = m + 94) : 
    photograph_perimeter w h m = 23 := 
by 
  sorry

end photograph_perimeter_is_23_l120_120036


namespace find_value_l120_120569

def star (a b : ℝ) (x y : ℝ) : ℝ := a * x + b * y + 2010

variable (a b : ℝ)

axiom h1 : 3 * a + 5 * b = 1
axiom h2 : 4 * a + 9 * b = -1

theorem find_value : star a b 1 2 = 2010 := 
by 
  sorry

end find_value_l120_120569


namespace new_volume_increased_dimensions_l120_120252

theorem new_volume_increased_dimensions (l w h : ℝ) 
  (h_vol : l * w * h = 5000) 
  (h_sa : l * w + w * h + h * l = 900) 
  (h_sum_edges : l + w + h = 60) : 
  (l + 2) * (w + 2) * (h + 2) = 7048 := 
by 
  sorry

end new_volume_increased_dimensions_l120_120252


namespace unique_solution_exists_l120_120451

theorem unique_solution_exists (k : ℚ) (h : k ≠ 0) : 
  (∀ x : ℚ, (x + 3) / (kx - 2) = x → x = -2) ↔ k = -3 / 4 := 
by
  sorry

end unique_solution_exists_l120_120451


namespace find_a_n_l120_120335

theorem find_a_n (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h_sum : ∀ n : ℕ, S n = 1/2 * (a n + 1 / (a n))) :
  ∀ n : ℕ, a n = Real.sqrt ↑n - Real.sqrt (↑n - 1) :=
by
  sorry

end find_a_n_l120_120335


namespace room_width_l120_120768

theorem room_width (W : ℝ) (L : ℝ := 17) (veranda_width : ℝ := 2) (veranda_area : ℝ := 132) :
  (21 * (W + veranda_width) - L * W = veranda_area) → W = 12 :=
by
  -- setup of the problem
  have total_length := L + 2 * veranda_width
  have total_width := W + 2 * veranda_width
  have area_room_incl_veranda := total_length * total_width - (L * W)
  -- the statement is already provided in the form of the theorem to be proven
  sorry

end room_width_l120_120768


namespace three_digit_difference_divisible_by_9_l120_120518

theorem three_digit_difference_divisible_by_9 :
  ∀ (a b c : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 →
    (100 * a + 10 * b + c - (a + b + c)) % 9 = 0 :=
by
  intros a b c h
  sorry

end three_digit_difference_divisible_by_9_l120_120518


namespace hockey_cards_count_l120_120005

-- Define integer variables for the number of hockey, football and baseball cards
variables (H F B : ℕ)

-- Define the conditions given in the problem
def condition1 : Prop := F = 4 * H
def condition2 : Prop := B = F - 50
def condition3 : Prop := H > 0
def condition4 : Prop := H + F + B = 1750

-- The theorem to prove
theorem hockey_cards_count 
  (h1 : condition1 H F)
  (h2 : condition2 F B)
  (h3 : condition3 H)
  (h4 : condition4 H F B) : 
  H = 200 := by
sorry

end hockey_cards_count_l120_120005


namespace least_four_digit_palindrome_divisible_by_11_l120_120714

theorem least_four_digit_palindrome_divisible_by_11 : 
  ∃ (A B : ℕ), (A ≠ 0 ∧ A < 10 ∧ B < 10 ∧ 1000 * A + 100 * B + 10 * B + A = 1111 ∧ (2 * A - 2 * B) % 11 = 0) := 
by
  sorry

end least_four_digit_palindrome_divisible_by_11_l120_120714


namespace dragons_legs_l120_120347

theorem dragons_legs :
  ∃ (n : ℤ), ∀ (x y : ℤ), x + 3 * y = 26
                       → 40 * x + n * y = 298
                       → n = 14 :=
by
  sorry

end dragons_legs_l120_120347


namespace sum_of_midpoints_x_coordinates_l120_120708

theorem sum_of_midpoints_x_coordinates (a b c : ℝ) (h : a + b + c = 15) :
  (a + b) / 2 + (a + c) / 2 + (b + c) / 2 = 15 := by
  sorry

end sum_of_midpoints_x_coordinates_l120_120708


namespace solve_for_y_l120_120570

theorem solve_for_y (x y : ℝ) (h : 4 * x + y = 9) : y = 9 - 4 * x :=
by sorry

end solve_for_y_l120_120570


namespace ed_marbles_l120_120941

theorem ed_marbles (doug_initial_marbles : ℕ) (marbles_lost : ℕ) (ed_doug_difference : ℕ) 
  (h1 : doug_initial_marbles = 22) (h2 : marbles_lost = 3) (h3 : ed_doug_difference = 5) : 
  (doug_initial_marbles + ed_doug_difference) = 27 :=
by
  sorry

end ed_marbles_l120_120941


namespace pythagorean_triple_fits_l120_120107

theorem pythagorean_triple_fits 
  (k : ℤ) (n : ℤ) : 
  (∃ k, (n = 5 * k ∨ n = 12 * k ∨ n = 13 * k) ∧ 
      (n = 62 ∨ n = 96 ∨ n = 120 ∨ n = 91 ∨ n = 390)) ↔ 
      (n = 120 ∨ n = 91) := by 
  sorry

end pythagorean_triple_fits_l120_120107


namespace class_strength_l120_120029

/-- The average age of an adult class is 40 years.
    12 new students with an average age of 32 years join the class,
    therefore decreasing the average by 4 years.
    What was the original strength of the class? -/
theorem class_strength (x : ℕ) (h1 : ∃ (x : ℕ), ∀ (y : ℕ), y ≠ x → y = 40) 
                       (h2 : 12 ≥ 0) (h3 : 32 ≥ 0) (h4 : (x + 12) * 36 = 40 * x + 12 * 32) : 
  x = 12 := 
sorry

end class_strength_l120_120029


namespace remainder_sum_first_150_l120_120687

-- Definitions based on the conditions
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

-- Lean statement equivalent to the mathematical problem
theorem remainder_sum_first_150 :
  (sum_first_n 150) % 11250 = 75 :=
by 
sorry

end remainder_sum_first_150_l120_120687


namespace distribution_of_balls_l120_120911

theorem distribution_of_balls :
  ∃ (P : ℕ → ℕ → ℕ), P 6 4 = 9 := 
by
  sorry

end distribution_of_balls_l120_120911


namespace range_of_a_l120_120598

theorem range_of_a (A B : Set ℝ) (a : ℝ)
  (hA : A = {x | 2 * a + 1 ≤ x ∧ x ≤ 3 * a - 5})
  (hB : B = {x | 3 ≤ x ∧ x ≤ 22}) :
  A ⊆ (A ∩ B) ↔ (1 ≤ a ∧ a ≤ 9) :=
by
  sorry

end range_of_a_l120_120598


namespace parity_implies_even_sum_l120_120390

theorem parity_implies_even_sum (n m : ℤ) (h : Even (n^2 + m^2 + n * m)) : ¬Odd (n + m) :=
sorry

end parity_implies_even_sum_l120_120390


namespace find_number_l120_120647

theorem find_number (x : ℝ) : (1.12 * x) / 4.98 = 528.0642570281125 → x = 2350 :=
  by 
  sorry

end find_number_l120_120647


namespace ratio_melina_alma_age_l120_120974

theorem ratio_melina_alma_age
  (A M : ℕ)
  (alma_score : ℕ)
  (h1 : M = 60)
  (h2 : alma_score = 40)
  (h3 : A + M = 2 * alma_score)
  : M / A = 3 :=
by
  sorry

end ratio_melina_alma_age_l120_120974


namespace class_average_correct_l120_120426

def class_average_test_A : ℝ :=
  0.30 * 97 + 0.25 * 85 + 0.20 * 78 + 0.15 * 65 + 0.10 * 55

def class_average_test_B : ℝ :=
  0.30 * 93 + 0.25 * 80 + 0.20 * 75 + 0.15 * 70 + 0.10 * 60

theorem class_average_correct :
  round class_average_test_A = 81 ∧
  round class_average_test_B = 79 := 
by 
  sorry

end class_average_correct_l120_120426


namespace fraction_product_l120_120640

theorem fraction_product :
  (2 / 3) * (5 / 7) * (9 / 11) * (4 / 13) = 360 / 3003 := by
  sorry

end fraction_product_l120_120640


namespace val_4_at_6_l120_120724

def at_op (a b : ℤ) : ℤ := 2 * a - 4 * b

theorem val_4_at_6 : at_op 4 6 = -16 := by
  sorry

end val_4_at_6_l120_120724


namespace number_of_toys_gained_l120_120199

theorem number_of_toys_gained
  (num_toys : ℕ) (selling_price : ℕ) (cost_price_one_toy : ℕ)
  (total_cp := num_toys * cost_price_one_toy)
  (profit := selling_price - total_cp)
  (num_toys_equiv_to_profit := profit / cost_price_one_toy) :
  num_toys = 18 → selling_price = 23100 → cost_price_one_toy = 1100 → num_toys_equiv_to_profit = 3 :=
by
  intros h1 h2 h3
  -- Proof to be completed
  sorry

end number_of_toys_gained_l120_120199


namespace choir_population_l120_120557

theorem choir_population 
  (female_students : ℕ) 
  (male_students : ℕ) 
  (choir_multiple : ℕ) 
  (total_students_orchestra : ℕ := female_students + male_students)
  (total_students_choir : ℕ := choir_multiple * total_students_orchestra)
  (h_females : female_students = 18) 
  (h_males : male_students = 25) 
  (h_multiple : choir_multiple = 3) : 
  total_students_choir = 129 := 
by
  -- The proof of the theorem will be done here.
  sorry

end choir_population_l120_120557


namespace arrangement_possible_32_arrangement_possible_100_l120_120398

-- Problem (1)
theorem arrangement_possible_32 : 
  ∃ (f : Fin 32 → Fin 32), ∀ (a b : Fin 32), ∀ (i : Fin 32), 
    a < b → i < b → f i = (a + b) / 2 → False := 
sorry

-- Problem (2)
theorem arrangement_possible_100 : 
  ∃ (f : Fin 100 → Fin 100), ∀ (a b : Fin 100), ∀ (i : Fin 100),
    a < b → i < b → f i = (a + b) / 2 → False := 
sorry


end arrangement_possible_32_arrangement_possible_100_l120_120398


namespace restaurant_meal_cost_l120_120973

def cost_of_group_meal (total_people : Nat) (kids : Nat) (adult_meal_cost : Nat) : Nat :=
  let adults := total_people - kids
  adults * adult_meal_cost

theorem restaurant_meal_cost :
  cost_of_group_meal 9 2 2 = 14 := by
  sorry

end restaurant_meal_cost_l120_120973


namespace complement_union_is_correct_l120_120314

def U : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 4}

theorem complement_union_is_correct : (U \ A) ∪ B = {0, 2, 4} := by
  sorry

end complement_union_is_correct_l120_120314


namespace jovana_added_pounds_l120_120310

noncomputable def initial_amount : ℕ := 5
noncomputable def final_amount : ℕ := 28

theorem jovana_added_pounds : final_amount - initial_amount = 23 := by
  sorry

end jovana_added_pounds_l120_120310


namespace find_number_l120_120988

theorem find_number : ∃ x, x - 0.16 * x = 126 ↔ x = 150 :=
by 
  sorry

end find_number_l120_120988


namespace rectangular_prism_edge_properties_l120_120417

-- Define a rectangular prism and the concept of parallel and perpendicular pairs of edges.
structure RectangularPrism :=
  (vertices : Fin 8 → Fin 3 → ℝ)
  -- Additional necessary conditions on the structure could be added here.

-- Define the number of parallel edges in a rectangular prism
def number_of_parallel_edge_pairs (rp : RectangularPrism) : ℕ :=
  -- Formula or logic to count parallel edge pairs.
  8 -- Placeholder for actual logic computation, based on problem conditions.

-- Define the number of perpendicular edges in a rectangular prism
def number_of_perpendicular_edge_pairs (rp : RectangularPrism) : ℕ :=
  -- Formula or logic to count perpendicular edge pairs.
  20 -- Placeholder for actual logic computation, based on problem conditions.

-- Theorem that asserts the requirement based on conditions
theorem rectangular_prism_edge_properties (rp : RectangularPrism) :
  number_of_parallel_edge_pairs rp = 8 ∧ number_of_perpendicular_edge_pairs rp = 20 :=
  by
    -- Placeholder proof that establishes the theorem
    sorry

end rectangular_prism_edge_properties_l120_120417


namespace tan_identity_find_sum_l120_120509

-- Given conditions
def is_geometric_sequence (a b c : ℝ) : Prop := b^2 = a * c

-- Specific problem statements
theorem tan_identity (a b c : ℝ) (A B C : ℝ)
  (h_geometric : is_geometric_sequence a b c)
  (h_cosB : Real.cos B = 3 / 4) :
  1 / Real.tan A + 1 / Real.tan C = 4 / Real.sqrt 7 :=
sorry

theorem find_sum (a b c : ℝ)
  (h_dot_product : a * c * 3 / 4 = 3 / 2) :
  a + c = 3 :=
sorry

end tan_identity_find_sum_l120_120509


namespace Jung_age_is_26_l120_120009

-- Define the ages of Li, Zhang, and Jung
def Li : ℕ := 12
def Zhang : ℕ := 2 * Li
def Jung : ℕ := Zhang + 2

-- The goal is to prove Jung's age is 26 years
theorem Jung_age_is_26 : Jung = 26 :=
by
  -- Placeholder for the proof
  sorry

end Jung_age_is_26_l120_120009


namespace arithmetic_sequence_sum_l120_120680

theorem arithmetic_sequence_sum (S : ℕ → ℤ) (m : ℕ) 
  (h1 : S (m - 1) = -2) 
  (h2 : S m = 0) 
  (h3 : S (m + 1) = 3) : 
  m = 5 :=
by sorry

end arithmetic_sequence_sum_l120_120680


namespace makenna_garden_larger_by_132_l120_120147

-- Define the dimensions of Karl's garden
def length_karl : ℕ := 22
def width_karl : ℕ := 50

-- Define the dimensions of Makenna's garden including the walking path
def length_makenna_total : ℕ := 30
def width_makenna_total : ℕ := 46
def walking_path_width : ℕ := 1

-- Define the area calculation functions
def area (length : ℕ) (width : ℕ) : ℕ := length * width

-- Calculate the areas
def area_karl : ℕ := area length_karl width_karl
def area_makenna : ℕ := area (length_makenna_total - 2 * walking_path_width) (width_makenna_total - 2 * walking_path_width)

-- Define the theorem to prove
theorem makenna_garden_larger_by_132 :
  area_makenna = area_karl + 132 :=
by
  -- We skip the proof part
  sorry

end makenna_garden_larger_by_132_l120_120147


namespace triangle_value_a_l120_120498

theorem triangle_value_a (a : ℕ) (h1: a + 2 > 6) (h2: a + 6 > 2) (h3: 2 + 6 > a) : a = 7 :=
sorry

end triangle_value_a_l120_120498


namespace paint_cost_decrease_l120_120888

variables (C P : ℝ)
variable (cost_decrease_canvas : ℝ := 0.40)
variable (total_cost_decrease : ℝ := 0.56)
variable (paint_to_canvas_ratio : ℝ := 4)

theorem paint_cost_decrease (x : ℝ) : 
  P = 4 * C ∧ 
  P * (1 - x) + C * (1 - cost_decrease_canvas) = (1 - total_cost_decrease) * (P + C) → 
  x = 0.60 :=
by
  intro h
  sorry

end paint_cost_decrease_l120_120888


namespace solution_set_of_inequality_l120_120240

theorem solution_set_of_inequality :
  {x : ℝ | x^2 * (x - 4) ≥ 0} = {x : ℝ | x = 0 ∨ x ≥ 4} :=
by
  sorry

end solution_set_of_inequality_l120_120240


namespace max_points_of_intersection_l120_120856

-- Define the conditions
variable {α : Type*} [DecidableEq α]
variable (L : Fin 100 → α → α → Prop) -- Representation of the lines

-- Define property of being parallel
variable (are_parallel : ∀ {n : ℕ}, L (5 * n) = L (5 * n + 5))

-- Define property of passing through point B
variable (passes_through_B : ∀ {n : ℕ}, ∃ P B, L (5 * n - 4) P B)

-- Prove the stated result
theorem max_points_of_intersection : 
  ∃ max_intersections, max_intersections = 4571 :=
by {
  sorry
}

end max_points_of_intersection_l120_120856


namespace change_is_4_25_l120_120901

-- Define the conditions
def apple_cost : ℝ := 0.75
def amount_paid : ℝ := 5.00

-- State the theorem
theorem change_is_4_25 : amount_paid - apple_cost = 4.25 :=
by
  sorry

end change_is_4_25_l120_120901


namespace stamp_problem_l120_120464

/-- Define the context where we have stamps of 7, n, and (n + 2) cents, and 120 cents being the largest
    value that cannot be formed using these stamps -/
theorem stamp_problem (n : ℕ) (h : ∀ k, k > 120 → ∃ a b c, k = 7 * a + n * b + (n + 2) * c) (hn : ¬ ∃ a b c, 120 = 7 * a + n * b + (n + 2) * c) : n = 22 :=
sorry

end stamp_problem_l120_120464


namespace original_price_of_saree_is_400_l120_120416

-- Define the original price of the saree
variable (P : ℝ)

-- Define the sale price after successive discounts
def sale_price (P : ℝ) : ℝ := 0.80 * P * 0.95

-- We want to prove that the original price P is 400 given that the sale price is 304
theorem original_price_of_saree_is_400 (h : sale_price P = 304) : P = 400 :=
sorry

end original_price_of_saree_is_400_l120_120416


namespace solution_set_of_x_abs_x_lt_x_l120_120262

theorem solution_set_of_x_abs_x_lt_x :
  {x : ℝ | x * |x| < x} = {x : ℝ | 0 < x ∧ x < 1} ∪ {x : ℝ | x < -1} :=
by
  sorry

end solution_set_of_x_abs_x_lt_x_l120_120262


namespace average_speed_is_correct_l120_120511

-- Define the conditions
def initial_odometer : ℕ := 2552
def final_odometer : ℕ := 2882
def time_first_day : ℕ := 5
def time_second_day : ℕ := 7

-- Calculate total time and distance
def total_time : ℕ := time_first_day + time_second_day
def total_distance : ℕ := final_odometer - initial_odometer

-- Prove that the average speed is 27.5 miles per hour
theorem average_speed_is_correct : (total_distance : ℚ) / (total_time : ℚ) = 27.5 :=
by
  sorry

end average_speed_is_correct_l120_120511


namespace lunks_to_apples_l120_120785

theorem lunks_to_apples :
  (∀ (a b c d e f : ℕ), (7 * b = 4 * a) → (3 * d = 5 * c) → c = 24 → f * e = d → e = 27) :=
by sorry

end lunks_to_apples_l120_120785


namespace kay_weight_training_time_l120_120019

variables (total_minutes : ℕ) (aerobic_ratio weight_ratio : ℕ)
-- Conditions
def kay_exercise := total_minutes = 250
def ratio_cond := aerobic_ratio = 3 ∧ weight_ratio = 2
def total_ratio_parts := aerobic_ratio + weight_ratio

-- Question and proof goal
theorem kay_weight_training_time (h1 : kay_exercise total_minutes) (h2 : ratio_cond aerobic_ratio weight_ratio) :
  (total_minutes / total_ratio_parts * weight_ratio) = 100 :=
by
  sorry

end kay_weight_training_time_l120_120019


namespace smallest_prime_dividing_large_sum_is_5_l120_120855

-- Definitions based on the conditions
def large_sum : ℕ := 4^15 + 7^12

-- Prime number checking function
def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

-- Check for the smallest prime number dividing the sum
def smallest_prime_dividing_sum (n : ℕ) : ℕ := 
  if n % 2 = 0 then 2 
  else if n % 3 = 0 then 3 
  else if n % 5 = 0 then 5 
  else 2 -- Since 2 is a placeholder, theoretical logic checks can replace this branch

-- Final theorem to prove
theorem smallest_prime_dividing_large_sum_is_5 : smallest_prime_dividing_sum large_sum = 5 := 
  sorry

end smallest_prime_dividing_large_sum_is_5_l120_120855


namespace contribution_amount_l120_120611

theorem contribution_amount (x : ℝ) (S : ℝ) :
  (S = 10 * x) ∧ (S = 15 * (x - 100)) → x = 300 :=
by
  sorry

end contribution_amount_l120_120611


namespace strawberries_left_correct_l120_120248

-- Define the initial and given away amounts in kilograms and grams
def initial_strawberries_kg : Int := 3
def initial_strawberries_g : Int := 300
def given_strawberries_kg : Int := 1
def given_strawberries_g : Int := 900

-- Define the conversion from kilograms to grams
def kg_to_g (kg : Int) : Int := kg * 1000

-- Calculate the total strawberries initially and given away in grams
def total_initial_strawberries_g : Int :=
  (kg_to_g initial_strawberries_kg) + initial_strawberries_g

def total_given_strawberries_g : Int :=
  (kg_to_g given_strawberries_kg) + given_strawberries_g

-- The amount of strawberries left after giving some away
def strawberries_left : Int :=
  total_initial_strawberries_g - total_given_strawberries_g

-- The statement to prove
theorem strawberries_left_correct :
  strawberries_left = 1400 :=
by
  sorry

end strawberries_left_correct_l120_120248


namespace not_sum_three_nonzero_squares_l120_120830

-- To state that 8n - 1 is not the sum of three non-zero squares
theorem not_sum_three_nonzero_squares (n : ℕ) :
  ¬ (∃ a b c : ℕ, a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ 8 * n - 1 = a^2 + b^2 + c^2) := by
  sorry

end not_sum_three_nonzero_squares_l120_120830


namespace dora_knows_coin_position_l120_120881

-- Definitions
def R_is_dime_or_nickel (R : ℕ) (L : ℕ) : Prop := 
  (R = 10 ∧ L = 5) ∨ (R = 5 ∧ L = 10)

-- Theorem statement
theorem dora_knows_coin_position (R : ℕ) (L : ℕ) 
  (h : R_is_dime_or_nickel R L) :
  (3 * R + 2 * L) % 2 = 0 ↔ (R = 10 ∧ L = 5) :=
by
  sorry

end dora_knows_coin_position_l120_120881


namespace two_by_three_grid_count_l120_120877

noncomputable def valid2x3Grids : Nat :=
  let valid_grids : Nat := 9
  valid_grids

theorem two_by_three_grid_count : valid2x3Grids = 9 := by
  -- Skipping the proof steps, but stating the theorem.
  sorry

end two_by_three_grid_count_l120_120877


namespace hypotenuse_right_triangle_l120_120260

theorem hypotenuse_right_triangle (a b : ℕ) (h1 : a = 15) (h2 : b = 36) :
  ∃ c, c ^ 2 = a ^ 2 + b ^ 2 ∧ c = 39 :=
by
  sorry

end hypotenuse_right_triangle_l120_120260


namespace ryan_lamps_probability_l120_120012

theorem ryan_lamps_probability :
  let total_lamps := 8
  let red_lamps := 4
  let blue_lamps := 4
  let total_ways_to_arrange := Nat.choose total_lamps red_lamps
  let total_ways_to_turn_on := Nat.choose total_lamps 4
  let remaining_blue := blue_lamps - 1 -- Due to leftmost lamp being blue and off
  let remaining_red := red_lamps - 1 -- Due to rightmost lamp being red and on
  let remaining_red_after_middle := remaining_red - 1 -- Due to middle lamp being red and off
  let remaining_lamps := remaining_blue + remaining_red_after_middle
  let ways_to_assign_remaining_red := Nat.choose remaining_lamps remaining_red_after_middle
  let ways_to_turn_on_remaining_lamps := Nat.choose remaining_lamps 2
  let favorable_ways := ways_to_assign_remaining_red * ways_to_turn_on_remaining_lamps
  let total_possibilities := total_ways_to_arrange * total_ways_to_turn_on
  favorable_ways / total_possibilities = (10 / 490) := by
  sorry

end ryan_lamps_probability_l120_120012


namespace watched_commercials_eq_100_l120_120457

variable (x : ℕ) -- number of people who watched commercials
variable (s : ℕ := 27) -- number of subscribers
variable (rev_comm : ℝ := 0.50) -- revenue per commercial
variable (rev_sub : ℝ := 1.00) -- revenue per subscriber
variable (total_rev : ℝ := 77.00) -- total revenue

theorem watched_commercials_eq_100 (h : rev_comm * (x : ℝ) + rev_sub * (s : ℝ) = total_rev) : x = 100 := by
  sorry

end watched_commercials_eq_100_l120_120457


namespace probability_jerry_at_four_l120_120516

theorem probability_jerry_at_four :
  let total_flips := 8
  let coordinate := 4
  let total_possible_outcomes := 2 ^ total_flips
  let favorable_outcomes := Nat.choose total_flips (total_flips / 2 + coordinate / 2)
  let P := favorable_outcomes / total_possible_outcomes
  let a := 7
  let b := 64
  ∃ (a b : ℕ), Nat.gcd a b = 1 ∧ P = a / b ∧ a + b = 71
:= sorry

end probability_jerry_at_four_l120_120516


namespace plane_divides_pyramid_l120_120853

noncomputable def volume_of_parts (a h KL KK1: ℝ): ℝ × ℝ :=
  -- Define the pyramid and prism structure and the conditions
  let volume_total := (1/3) * (a^2) * h
  let volume_part1 := 512/15
  let volume_part2 := volume_total - volume_part1
  (⟨volume_part1, volume_part2⟩ : ℝ × ℝ)

theorem plane_divides_pyramid (a h KL KK1: ℝ) 
  (h₁ : a = 8 * Real.sqrt 2) 
  (h₂ : h = 4) 
  (h₃ : KL = 2) 
  (h₄ : KK1 = 1):
  volume_of_parts a h KL KK1 = (512/15, 2048/15) := 
by 
  sorry

end plane_divides_pyramid_l120_120853


namespace system_of_equations_solution_l120_120733

theorem system_of_equations_solution
  (x y z : ℤ)
  (h1 : x + y + z = 12)
  (h2 : 8 * x + 5 * y + 3 * z = 60) :
  (x = 0 ∧ y = 12 ∧ z = 0) ∨
  (x = 2 ∧ y = 7 ∧ z = 3) ∨
  (x = 4 ∧ y = 2 ∧ z = 6) :=
sorry

end system_of_equations_solution_l120_120733


namespace zmod_field_l120_120033

theorem zmod_field (p : ℕ) [Fact (Nat.Prime p)] : Field (ZMod p) :=
sorry

end zmod_field_l120_120033


namespace investment_amount_l120_120667

noncomputable def annual_income (investment : ℝ) (percent_stock : ℝ) (market_price : ℝ) : ℝ :=
  (investment * percent_stock / 100) / market_price * market_price

theorem investment_amount (annual_income_value : ℝ) (percent_stock : ℝ) (market_price : ℝ) (investment : ℝ) :
  annual_income investment percent_stock market_price = annual_income_value →
  investment = 6800 :=
by
  intros
  sorry

end investment_amount_l120_120667


namespace parabola_x_intercepts_l120_120829

theorem parabola_x_intercepts : 
  ∃! x : ℝ, ∃ y : ℝ, y = 0 ∧ x = -3 * y ^ 2 + 2 * y + 3 :=
by 
  sorry

end parabola_x_intercepts_l120_120829


namespace cost_of_camel_is_6000_l120_120008

noncomputable def cost_of_camel : ℕ := 6000

variables (C H O E : ℕ)
variables (cost_of_camel_rs cost_of_horses cost_of_oxen cost_of_elephants : ℕ)

-- Conditions
axiom cond1 : 10 * C = 24 * H
axiom cond2 : 16 * H = 4 * O
axiom cond3 : 6 * O = 4 * E
axiom cond4 : 10 * E = 150000

theorem cost_of_camel_is_6000
    (cond1 : 10 * C = 24 * H)
    (cond2 : 16 * H = 4 * O)
    (cond3 : 6 * O = 4 * E)
    (cond4 : 10 * E = 150000) :
  cost_of_camel = 6000 := 
sorry

end cost_of_camel_is_6000_l120_120008


namespace fraction_of_white_surface_area_l120_120748

/-- A cube has edges of 4 inches and is constructed using 64 smaller cubes, each with edges of 1 inch.
Out of these smaller cubes, 56 are white and 8 are black. The 8 black cubes fully cover one face of the larger cube.
Prove that the fraction of the surface area of the larger cube that is white is 5/6. -/
theorem fraction_of_white_surface_area 
  (total_cubes : ℕ := 64)
  (white_cubes : ℕ := 56)
  (black_cubes : ℕ := 8)
  (total_surface_area : ℕ := 96)
  (black_face_area : ℕ := 16)
  (white_surface_area : ℕ := 80) :
  white_surface_area / total_surface_area = 5 / 6 :=
sorry

end fraction_of_white_surface_area_l120_120748


namespace average_minutes_run_l120_120204

theorem average_minutes_run (t : ℕ) (t_pos : 0 < t) 
  (average_first_graders : ℕ := 8) 
  (average_second_graders : ℕ := 12) 
  (average_third_graders : ℕ := 16)
  (num_first_graders : ℕ := 9 * t)
  (num_second_graders : ℕ := 3 * t)
  (num_third_graders : ℕ := t) :
  (8 * 9 * t + 12 * 3 * t + 16 * t) / (9 * t + 3 * t + t) = 10 := 
by
  sorry

end average_minutes_run_l120_120204


namespace arithmetic_sequence_problem_l120_120245

theorem arithmetic_sequence_problem 
  (a : ℕ → ℚ) 
  (a1 : a 1 = 1 / 3) 
  (a2_a5 : a 2 + a 5 = 4) 
  (an : ∃ n, a n = 33) :
  ∃ n, a n = 33 ∧ n = 50 := 
by 
  sorry

end arithmetic_sequence_problem_l120_120245


namespace original_apples_l120_120267

-- Define the conditions
def sells_50_percent (initial remaining : ℕ) : Prop :=
  (initial / 2) = remaining

-- Define the goal
theorem original_apples (remaining : ℕ) (initial : ℕ) (h : sells_50_percent initial remaining) : initial = 10000 :=
by
  sorry

end original_apples_l120_120267


namespace intermediate_value_theorem_example_l120_120629

theorem intermediate_value_theorem_example (f : ℝ → ℝ) :
  f 2007 < 0 → f 2008 < 0 → f 2009 > 0 → ∃ x, 2007 < x ∧ x < 2008 ∧ f x = 0 :=
by
  sorry

end intermediate_value_theorem_example_l120_120629


namespace arithmetic_sequence_a20_l120_120615

theorem arithmetic_sequence_a20 (a : ℕ → ℝ) (d : ℝ) 
  (h1 : ∀ n, a (n + 1) = a n + d) 
  (h2 : a 1 + a 3 + a 5 = 18)
  (h3 : a 2 + a 4 + a 6 = 24) :
  a 20 = 40 :=
sorry

end arithmetic_sequence_a20_l120_120615


namespace min_value_frac_l120_120827

theorem min_value_frac (x y : ℝ) (h₁ : x + y = 1) (h₂ : x > 0) (h₃ : y > 0) : 
  ∃ c, (∀ (a b : ℝ), (a + b = 1) → (a > 0) → (b > 0) → (1/a + 4/b) ≥ c) ∧ c = 9 :=
by
  sorry

end min_value_frac_l120_120827


namespace percentage_decrease_in_selling_price_l120_120449

theorem percentage_decrease_in_selling_price (S M : ℝ) 
  (purchase_price : S = 240 + M)
  (markup_percentage : M = 0.25 * S)
  (gross_profit : S - 16 = 304) : 
  (320 - 304) / 320 * 100 = 5 := 
by
  sorry

end percentage_decrease_in_selling_price_l120_120449


namespace infinitely_many_primes_congruent_3_mod_4_l120_120980

def is_congruent_3_mod_4 (p : ℕ) : Prop :=
  p % 4 = 3

def is_prime (p : ℕ) : Prop :=
  Nat.Prime p

def S (p : ℕ) : Prop :=
  is_prime p ∧ is_congruent_3_mod_4 p

theorem infinitely_many_primes_congruent_3_mod_4 :
  ∀ n : ℕ, ∃ p : ℕ, p > n ∧ S p :=
sorry

end infinitely_many_primes_congruent_3_mod_4_l120_120980


namespace complex_number_imaginary_l120_120388

theorem complex_number_imaginary (x : ℝ) 
  (h1 : x^2 - 2*x - 3 = 0)
  (h2 : x + 1 ≠ 0) : x = 3 := sorry

end complex_number_imaginary_l120_120388


namespace convert_to_base_8_l120_120561

theorem convert_to_base_8 (n : ℕ) (hn : n = 3050) : 
  ∃ d1 d2 d3 d4 : ℕ, d1 = 5 ∧ d2 = 7 ∧ d3 = 5 ∧ d4 = 2 ∧ n = d1 * 8^3 + d2 * 8^2 + d3 * 8^1 + d4 * 8^0 :=
by 
  use 5, 7, 5, 2
  sorry

end convert_to_base_8_l120_120561


namespace car_speed_l120_120773

variable (v : ℝ)
variable (Distance : ℝ := 1)  -- distance in kilometers
variable (Speed_120 : ℝ := 120)  -- speed in kilometers per hour
variable (Time_120 : ℝ := Distance / Speed_120)  -- time in hours to travel 1 km at 120 km/h
variable (Time_120_sec : ℝ := Time_120 * 3600)  -- time in seconds to travel 1 km at 120 km/h
variable (Additional_time : ℝ := 2)  -- additional time in seconds
variable (Time_v_sec : ℝ := Time_120_sec + Additional_time)  -- time in seconds for unknown speed
variable (Time_v : ℝ := Time_v_sec / 3600)  -- time in hours for unknown speed

theorem car_speed (h : v = Distance / Time_v) : v = 112.5 :=
by
  -- The given proof steps will go here
  sorry

end car_speed_l120_120773


namespace compound_interest_rate_l120_120844

theorem compound_interest_rate
  (P : ℝ) (r : ℝ) :
  (3000 = P * (1 + r / 100)^3) →
  (3600 = P * (1 + r / 100)^4) →
  r = 20 :=
by
  sorry

end compound_interest_rate_l120_120844


namespace triangle_area_l120_120515

theorem triangle_area (a b c : ℕ) (h1 : a + b + c = 12) (h2 : a > 0) (h3 : b > 0) (h4 : c > 0) 
                      (h5 : a + b > c) (h6 : b + c > a) (h7 : c + a > b) :
  ∃ A : ℝ, A = 2 * Real.sqrt 6 ∧
    ∃ (h : 0 ≤ A), A = (Real.sqrt (A * 12 * (12 - a) * (12 - b) * (12 - c))) :=
sorry

end triangle_area_l120_120515


namespace orchid_bushes_planted_l120_120720

theorem orchid_bushes_planted (b1 b2 : ℕ) (h1 : b1 = 22) (h2 : b2 = 35) : b2 - b1 = 13 :=
by 
  sorry

end orchid_bushes_planted_l120_120720


namespace average_age_increase_by_one_l120_120697

-- Definitions based on the conditions.
def initial_average_age : ℕ := 14
def initial_students : ℕ := 10
def new_students_average_age : ℕ := 17
def new_students : ℕ := 5

-- Helper calculation for the total age of initial students.
def total_age_initial_students := initial_students * initial_average_age

-- Helper calculation for the total age of new students.
def total_age_new_students := new_students * new_students_average_age

-- Helper calculation for the total age of all students.
def total_age_all_students := total_age_initial_students + total_age_new_students

-- Helper calculation for the number of all students.
def total_students := initial_students + new_students

-- Calculate the new average age.
def new_average_age := total_age_all_students / total_students

-- The goal is to prove the increase in average age is 1 year.
theorem average_age_increase_by_one :
  new_average_age - initial_average_age = 1 :=
by
  -- Proof goes here
  sorry

end average_age_increase_by_one_l120_120697


namespace total_yellow_leaves_l120_120929

noncomputable def calculate_yellow_leaves (total : ℕ) (percent_brown : ℕ) (percent_green : ℕ) : ℕ :=
  let brown_leaves := (total * percent_brown + 50) / 100
  let green_leaves := (total * percent_green + 50) / 100
  total - (brown_leaves + green_leaves)

theorem total_yellow_leaves :
  let t_yellow := calculate_yellow_leaves 15 25 40
  let f_yellow := calculate_yellow_leaves 22 30 20
  let s_yellow := calculate_yellow_leaves 30 15 50
  t_yellow + f_yellow + s_yellow = 26 :=
by
  sorry

end total_yellow_leaves_l120_120929


namespace third_side_not_twelve_l120_120349

theorem third_side_not_twelve (x : ℕ) (h1 : x > 5) (h2 : x < 11) (h3 : x % 2 = 0) : x ≠ 12 :=
by
  -- The proof is omitted
  sorry

end third_side_not_twelve_l120_120349


namespace sqrt_meaningful_range_l120_120093

theorem sqrt_meaningful_range (x : ℝ) (h : 2 * x - 3 ≥ 0) : x ≥ 3 / 2 :=
sorry

end sqrt_meaningful_range_l120_120093


namespace find_last_three_digits_of_9_pow_107_l120_120170

theorem find_last_three_digits_of_9_pow_107 : (9 ^ 107) % 1000 = 969 := 
by 
  sorry

end find_last_three_digits_of_9_pow_107_l120_120170


namespace inequality_has_solutions_l120_120048

theorem inequality_has_solutions (a : ℝ) :
  (∃ x : ℝ, |x + 3| + |x - 1| < a^2 - 3 * a) ↔ (a < -1 ∨ 4 < a) := 
by
  sorry

end inequality_has_solutions_l120_120048


namespace parabola_distance_focus_P_l120_120519

noncomputable def distance_PF : ℝ := sorry

theorem parabola_distance_focus_P : ∀ (P : ℝ × ℝ) (F : ℝ × ℝ),
  P.2^2 = 4 * P.1 ∧ F = (1, 0) ∧ P.1 = 4 → distance_PF = 5 :=
by
  intros P F h
  sorry

end parabola_distance_focus_P_l120_120519


namespace required_run_rate_is_correct_l120_120813

-- Define the initial conditions
def run_rate_first_10_overs : ℝ := 3.2
def overs_first_10 : ℝ := 10
def target_runs : ℝ := 282
def remaining_overs : ℝ := 40

-- Given total runs in the first 10 overs
def total_runs_first_10_overs : ℝ := run_rate_first_10_overs * overs_first_10
-- Given runs needed in the remaining 40 overs
def runs_needed_remaining_overs : ℝ := target_runs - total_runs_first_10_overs

-- Lean statement to prove the required run rate in the remaining 40 overs
theorem required_run_rate_is_correct (h1 : run_rate_first_10_overs = 3.2)
                                     (h2 : overs_first_10 = 10)
                                     (h3 : target_runs = 282)
                                     (h4 : remaining_overs = 40) :
  (runs_needed_remaining_overs / remaining_overs) = 6.25 :=
by sorry


end required_run_rate_is_correct_l120_120813


namespace binomial_1300_2_eq_844350_l120_120362

theorem binomial_1300_2_eq_844350 : Nat.choose 1300 2 = 844350 := 
by
  sorry

end binomial_1300_2_eq_844350_l120_120362


namespace half_abs_sum_diff_squares_cubes_l120_120071

theorem half_abs_sum_diff_squares_cubes (a b : ℤ) (h1 : a = 21) (h2 : b = 15) :
  (|a^2 - b^2| + |a^3 - b^3|) / 2 = 3051 := by
  sorry

end half_abs_sum_diff_squares_cubes_l120_120071


namespace divisible_by_factorial_l120_120272

def f : ℕ → ℕ → ℕ
| 0, 0 => 1
| 0, _ => 0
| _, 0 => 0
| n + 1, k + 1 => (n + 1) * (f (n + 1) k + f n k)

theorem divisible_by_factorial (n k : ℕ) : n! ∣ f n k := by sorry

end divisible_by_factorial_l120_120272


namespace tan_half_angle_l120_120725

theorem tan_half_angle (p q : ℝ) (h_cos : Real.cos p + Real.cos q = 3 / 5) (h_sin : Real.sin p + Real.sin q = 1 / 5) : Real.tan ((p + q) / 2) = 1 / 3 :=
sorry

end tan_half_angle_l120_120725


namespace series_sum_l120_120455

noncomputable def series_term (n : ℕ) : ℝ :=
  (4 * n + 2) / ((6 * n - 5)^2 * (6 * n + 1)^2)

theorem series_sum :
  (∑' n : ℕ, series_term (n + 1)) = 1 / 6 :=
by
  sorry

end series_sum_l120_120455


namespace find_solutions_equation_l120_120183

theorem find_solutions_equation :
  {x : ℝ | 1 / (x^2 + 13 * x - 12) + 1 / (x^2 + 4 * x - 12) + 1 / (x^2 - 11 * x - 12) = 0}
  = {1, -12, 4, -3} :=
by
  sorry

end find_solutions_equation_l120_120183


namespace problem_statement_l120_120578

def f (x : ℝ) : ℝ := x^2 - 4*x + 4

theorem problem_statement : f (f (f (f (f (f 2))))) = 4 :=
by
  sorry

end problem_statement_l120_120578


namespace service_station_location_l120_120579

/-- The first exit is at milepost 35. -/
def first_exit_milepost : ℕ := 35

/-- The eighth exit is at milepost 275. -/
def eighth_exit_milepost : ℕ := 275

/-- The expected milepost of the service station built halfway between the first exit and the eighth exit is 155. -/
theorem service_station_location : (first_exit_milepost + (eighth_exit_milepost - first_exit_milepost) / 2) = 155 := by
  sorry

end service_station_location_l120_120579


namespace wall_width_l120_120076

theorem wall_width
  (w h l : ℝ)
  (h_eq : h = 6 * w)
  (l_eq : l = 7 * h)
  (volume_eq : w * h * l = 6804) :
  w = 3 :=
by
  sorry

end wall_width_l120_120076


namespace password_correct_l120_120957

-- conditions
def poly1 (x y : ℤ) : ℤ := x ^ 4 - y ^ 4
def factor1 (x y : ℤ) : ℤ := (x - y) * (x + y) * (x ^ 2 + y ^ 2)

def poly2 (x y : ℤ) : ℤ := x ^ 3 - x * y ^ 2
def factor2 (x y : ℤ) : ℤ := x * (x - y) * (x + y)

-- given values
def x := 18
def y := 5

-- goal
theorem password_correct : factor2 x y = 18 * 13 * 23 :=
by
  -- We setup the goal with the equivalent sequence of the password generation
  sorry

end password_correct_l120_120957


namespace probability_of_rectangle_area_greater_than_32_l120_120959

-- Definitions representing the problem conditions
def segment_length : ℝ := 12
def point_C (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ segment_length
def rectangle_area (x : ℝ) : ℝ := x * (segment_length - x)

-- The probability we need to prove. 
noncomputable def desired_probability : ℝ := 1 / 3

theorem probability_of_rectangle_area_greater_than_32 :
  (∀ x, point_C x → rectangle_area x > 32) → (desired_probability = 1 / 3) :=
by
  sorry

end probability_of_rectangle_area_greater_than_32_l120_120959


namespace symmetric_line_x_axis_l120_120376

theorem symmetric_line_x_axis (y : ℝ → ℝ) (x : ℝ) :
  (∀ x, y x = 2 * x + 1) → (∀ x, -y x = 2 * x + 1) → y x = -2 * x -1 :=
by
  intro h1 h2
  sorry

end symmetric_line_x_axis_l120_120376


namespace total_distance_traveled_in_12_hours_l120_120645

variable (n a1 d : ℕ) (u : ℕ → ℕ)

def arithmetic_seq_sum (n : ℕ) (a1 : ℕ) (d : ℕ) : ℕ :=
  n * a1 + (n * (n - 1) * d) / 2

theorem total_distance_traveled_in_12_hours :
  arithmetic_seq_sum 12 55 2 = 792 := by
  sorry

end total_distance_traveled_in_12_hours_l120_120645


namespace negation_is_all_odd_or_at_least_two_even_l120_120383

-- Define natural numbers a, b, and c.
variables {a b c : ℕ}

-- Define a predicate is_even which checks if a number is even.
def is_even (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k

-- Define the statement that exactly one of the natural numbers a, b, and c is even.
def exactly_one_even (a b c : ℕ) : Prop :=
  (is_even a ∨ is_even b ∨ is_even c) ∧
  ¬ (is_even a ∧ is_even b) ∧
  ¬ (is_even a ∧ is_even c) ∧
  ¬ (is_even b ∧ is_even c)

-- Define the negation of the statement that exactly one of the natural numbers a, b, and c is even.
def negation_of_exactly_one_even (a b c : ℕ) : Prop :=
  ¬ exactly_one_even a b c

-- State that the negation of exactly one even number among a, b, c is equivalent to all being odd or at least two being even.
theorem negation_is_all_odd_or_at_least_two_even :
  negation_of_exactly_one_even a b c ↔ (¬ is_even a ∧ ¬ is_even b ∧ ¬ is_even c) ∨ (is_even a ∧ is_even b) ∨ (is_even a ∧ is_even c) ∨ (is_even b ∧ is_even c) :=
sorry

end negation_is_all_odd_or_at_least_two_even_l120_120383


namespace cube_inequality_l120_120283

theorem cube_inequality {a b : ℝ} (h : a > b) : a^3 > b^3 :=
sorry

end cube_inequality_l120_120283


namespace rectangle_area_ratio_l120_120954

theorem rectangle_area_ratio (x d : ℝ) (h_ratio : 5 * x / (2 * x) = 5 / 2) (h_diag : d = 13) :
  ∃ k : ℝ, 10 * x^2 = k * d^2 ∧ k = 10 / 29 :=
by
  sorry

end rectangle_area_ratio_l120_120954


namespace water_added_to_solution_l120_120531

theorem water_added_to_solution :
  let initial_volume := 340
  let initial_sugar := 0.20 * initial_volume
  let added_sugar := 3.2
  let added_kola := 6.8
  let final_sugar := initial_sugar + added_sugar
  let final_percentage_sugar := 19.66850828729282 / 100
  let final_volume := final_sugar / final_percentage_sugar
  let added_water := final_volume - initial_volume - added_sugar - added_kola
  added_water = 12 :=
by
  sorry

end water_added_to_solution_l120_120531


namespace original_plan_trees_average_l120_120343

-- Definitions based on conditions
def original_trees_per_day (x : ℕ) := x
def increased_trees_per_day (x : ℕ) := x + 5
def time_to_plant_60_trees (x : ℕ) := 60 / (x + 5)
def time_to_plant_45_trees (x : ℕ) := 45 / x

-- The main theorem we need to prove
theorem original_plan_trees_average : ∃ x : ℕ, time_to_plant_60_trees x = time_to_plant_45_trees x ∧ x = 15 :=
by
  -- Placeholder for the proof
  sorry

end original_plan_trees_average_l120_120343


namespace product_of_five_consecutive_integers_not_square_l120_120573

theorem product_of_five_consecutive_integers_not_square (n : ℕ) :
  let P := n * (n + 1) * (n + 2) * (n + 3) * (n + 4)
  ∀ k : ℕ, P ≠ k^2 := 
sorry

end product_of_five_consecutive_integers_not_square_l120_120573


namespace pet_purchase_ways_l120_120435

-- Define the conditions
def number_of_puppies : Nat := 20
def number_of_kittens : Nat := 6
def number_of_hamsters : Nat := 8

def alice_choices : Nat := number_of_puppies

-- Define the problem statement in Lean
theorem pet_purchase_ways : 
  (number_of_puppies = 20) ∧ 
  (number_of_kittens = 6) ∧ 
  (number_of_hamsters = 8) → 
  (alice_choices * 2 * number_of_kittens * number_of_hamsters) = 1920 := 
by
  intros h
  sorry

end pet_purchase_ways_l120_120435


namespace work_completion_time_l120_120705

theorem work_completion_time (x : ℝ) (a_work_rate b_work_rate combined_work_rate : ℝ) :
  a_work_rate = 1 / 15 ∧
  b_work_rate = 1 / 20 ∧
  combined_work_rate = 1 / 7.2 ∧
  a_work_rate + b_work_rate + (1 / x) = combined_work_rate → 
  x = 45 := by
  sorry

end work_completion_time_l120_120705


namespace rational_division_example_l120_120837

theorem rational_division_example : (3 / 7) / 5 = 3 / 35 := by
  sorry

end rational_division_example_l120_120837


namespace smallest_n_for_divisibility_l120_120915

theorem smallest_n_for_divisibility (a₁ a₂ : ℕ) (n : ℕ) (h₁ : a₁ = 5 / 8) (h₂ : a₂ = 25) :
  (∃ n : ℕ, n ≥ 1 ∧ (a₁ * (40 ^ (n - 1)) % 2000000 = 0)) → (n = 7) :=
by
  sorry

end smallest_n_for_divisibility_l120_120915


namespace Juan_birth_year_proof_l120_120445

-- Let BTC_year(n) be the year of the nth BTC competition.
def BTC_year (n : ℕ) : ℕ :=
  1990 + (n - 1) * 2

-- Juan's birth year given his age and the BTC he participated in.
def Juan_birth_year (current_year : ℕ) (age : ℕ) : ℕ :=
  current_year - age

-- Main proof problem statement.
theorem Juan_birth_year_proof :
  (BTC_year 5 = 1998) →
  (Juan_birth_year 1998 14 = 1984) :=
by
  intros
  sorry

end Juan_birth_year_proof_l120_120445


namespace tori_original_height_l120_120050

-- Definitions for given conditions
def current_height : ℝ := 7.26
def height_gained : ℝ := 2.86

-- Theorem statement
theorem tori_original_height : current_height - height_gained = 4.40 :=
by sorry

end tori_original_height_l120_120050


namespace negation_of_at_most_three_l120_120919

theorem negation_of_at_most_three (x : ℕ) : ¬ (x ≤ 3) ↔ x > 3 :=
by sorry

end negation_of_at_most_three_l120_120919


namespace gcd_sixPn_n_minus_2_l120_120193

def nthSquarePyramidalNumber (n : ℕ) : ℕ :=
  (n * (n + 1) * (2 * n + 1)) / 6

def sixPn (n : ℕ) : ℕ :=
  n * (n + 1) * (2 * n + 1)

theorem gcd_sixPn_n_minus_2 (n : ℕ) (h_pos : 0 < n) : Int.gcd (sixPn n) (n - 2) ≤ 12 :=
by
  sorry

end gcd_sixPn_n_minus_2_l120_120193


namespace sum_last_two_digits_l120_120789

theorem sum_last_two_digits (n : ℕ) (h1 : n = 20) : (9^n + 11^n) % 100 = 1 :=
by
  sorry

end sum_last_two_digits_l120_120789


namespace living_room_curtain_length_l120_120783

theorem living_room_curtain_length :
  let length_bolt := 16
  let width_bolt := 12
  let area_bolt := length_bolt * width_bolt
  let area_left := 160
  let area_cut := area_bolt - area_left
  let length_bedroom := 2
  let width_bedroom := 4
  let area_bedroom := length_bedroom * width_bedroom
  let area_living_room := area_cut - area_bedroom
  let width_living_room := 4
  area_living_room / width_living_room = 6 :=
by
  sorry

end living_room_curtain_length_l120_120783


namespace num_three_digit_integers_sum_to_seven_l120_120154

open Nat

-- Define the three digits a, b, and c and their constraints
def digits_satisfy (a b c : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ a + b + c = 7

-- State the theorem we wish to prove
theorem num_three_digit_integers_sum_to_seven :
  (∃ n : ℕ, ∃ a b c : ℕ, digits_satisfy a b c ∧ a * 100 + b * 10 + c = n) →
  (∃ N : ℕ, N = 28) :=
sorry

end num_three_digit_integers_sum_to_seven_l120_120154


namespace number_of_sequences_l120_120778

-- Define the number of possible outcomes for a single coin flip
def coinFlipOutcomes : ℕ := 2

-- Define the number of flips
def numberOfFlips : ℕ := 8

-- Theorem statement: The number of distinct sequences when flipping a coin eight times is 256
theorem number_of_sequences (n : ℕ) (outcomes : ℕ) (h : outcomes = 2) (hn : n = 8) : outcomes ^ n = 256 := by
  sorry

end number_of_sequences_l120_120778


namespace percentage_relations_with_respect_to_z_l120_120305

variable (x y z w : ℝ)
variable (h1 : x = 1.30 * y)
variable (h2 : y = 0.50 * z)
variable (h3 : w = 2 * x)

theorem percentage_relations_with_respect_to_z : 
  x = 0.65 * z ∧ y = 0.50 * z ∧ w = 1.30 * z := by
  sorry

end percentage_relations_with_respect_to_z_l120_120305


namespace age_of_b_is_6_l120_120547

theorem age_of_b_is_6 (x : ℕ) (h1 : 5 * x / 3 * x = 5 / 3)
                         (h2 : (5 * x + 2) / (3 * x + 2) = 3 / 2) : 3 * x = 6 := 
by
  sorry

end age_of_b_is_6_l120_120547


namespace washingMachineCapacity_l120_120520

-- Definitions based on the problem's conditions
def numberOfShirts : ℕ := 2
def numberOfSweaters : ℕ := 33
def numberOfLoads : ℕ := 5

-- Statement we need to prove
theorem washingMachineCapacity : 
  (numberOfShirts + numberOfSweaters) / numberOfLoads = 7 := sorry

end washingMachineCapacity_l120_120520


namespace kangaroo_arrangement_count_l120_120168

theorem kangaroo_arrangement_count :
  let k := 8
  let tallest_at_ends := 2
  let middle := k - tallest_at_ends
  (tallest_at_ends * (middle.factorial)) = 1440 := by
  sorry

end kangaroo_arrangement_count_l120_120168


namespace find_missing_digit_divisibility_by_4_l120_120022

theorem find_missing_digit_divisibility_by_4 (x : ℕ) (h : x < 10) :
  (3280 + x) % 4 = 0 ↔ x = 0 ∨ x = 2 ∨ x = 4 ∨ x = 6 ∨ x = 8 :=
by
  sorry

end find_missing_digit_divisibility_by_4_l120_120022


namespace gcd_of_powers_of_two_l120_120938

def m : ℕ := 2^2100 - 1
def n : ℕ := 2^2000 - 1

theorem gcd_of_powers_of_two :
  Nat.gcd m n = 2^100 - 1 := sorry

end gcd_of_powers_of_two_l120_120938


namespace inequality_conditions_l120_120546

variable (a b : ℝ)

theorem inequality_conditions (ha : 1 / a < 1 / b) (hb : 1 / b < 0) : 
  (1 / (a + b) < 1 / (a * b)) ∧ ¬(a * - (1 / a) > b * - (1 / b)) := 
by 
  sorry

end inequality_conditions_l120_120546


namespace prime_odd_sum_l120_120732

theorem prime_odd_sum (x y : ℕ) (h_prime : Prime x) (h_odd : y % 2 = 1) (h_eq : x^2 + y = 2005) : x + y = 2003 :=
by
  sorry

end prime_odd_sum_l120_120732


namespace student_average_vs_true_average_l120_120953

theorem student_average_vs_true_average (w x y z : ℝ) (h : w < x ∧ x < y ∧ y < z) : 
  (2 * w + 2 * x + y + z) / 6 < (w + x + y + z) / 4 :=
by
  sorry

end student_average_vs_true_average_l120_120953


namespace intersection_of_A_and_B_l120_120816

noncomputable def A := {x : ℝ | Real.log x ≤ 0}
noncomputable def B := {x : ℝ | abs (x^2 - 1) ≤ 1}

theorem intersection_of_A_and_B : A ∩ B = A :=
sorry

end intersection_of_A_and_B_l120_120816


namespace aku_mother_packages_l120_120004

theorem aku_mother_packages
  (friends : Nat)
  (cookies_per_package : Nat)
  (cookies_per_child : Nat)
  (total_children : Nat)
  (birthday : Nat)
  (H_friends : friends = 4)
  (H_cookies_per_package : cookies_per_package = 25)
  (H_cookies_per_child : cookies_per_child = 15)
  (H_total_children : total_children = friends + 1)
  (H_birthday : birthday = 10) :
  (total_children * cookies_per_child) / cookies_per_package = 3 :=
by
  sorry

end aku_mother_packages_l120_120004


namespace vector_dot_product_l120_120544

open Matrix

section VectorDotProduct

variables (A : ℝ × ℝ) (B : ℝ × ℝ) (C : ℝ × ℝ)
variables (E : ℝ × ℝ) (F : ℝ × ℝ)

def vector_sub (P Q : ℝ × ℝ) : ℝ × ℝ := (P.1 - Q.1, P.2 - Q.2)
def vector_add (P Q : ℝ × ℝ) : ℝ × ℝ := (P.1 + Q.1, P.2 + Q.2)
def scalar_mul (k : ℝ) (P : ℝ × ℝ) : ℝ × ℝ := (k * P.1, k * P.2)
def dot_product (P Q : ℝ × ℝ) : ℝ := P.1 * Q.1 + P.2 * Q.2

axiom A_coord : A = (1, 2)
axiom B_coord : B = (2, -1)
axiom C_coord : C = (2, 2)
axiom E_is_trisection : vector_add (vector_sub B A) (scalar_mul (1/3) (vector_sub C B)) = E
axiom F_is_trisection : vector_add (vector_sub B A) (scalar_mul (2/3) (vector_sub C B)) = F

theorem vector_dot_product : dot_product (vector_sub E A) (vector_sub F A) = 3 := by
  sorry

end VectorDotProduct

end vector_dot_product_l120_120544


namespace white_tile_count_l120_120971

theorem white_tile_count (total_tiles yellow_tiles blue_tiles purple_tiles white_tiles : ℕ)
  (h_total : total_tiles = 20)
  (h_yellow : yellow_tiles = 3)
  (h_blue : blue_tiles = yellow_tiles + 1)
  (h_purple : purple_tiles = 6)
  (h_sum : total_tiles = yellow_tiles + blue_tiles + purple_tiles + white_tiles) :
  white_tiles = 7 :=
sorry

end white_tile_count_l120_120971


namespace volume_of_region_l120_120968

theorem volume_of_region :
  ∃ (V : ℝ), V = 9 ∧
  ∀ (x y z : ℝ), |x + y + z| + |x + y - z| + |x - y + z| + |-x + y + z| ≤ 6 :=
sorry

end volume_of_region_l120_120968


namespace tracy_initial_candies_l120_120821

theorem tracy_initial_candies (x : ℕ) (consumed_candies : ℕ) (remaining_candies_given_rachel : ℕ) (remaining_candies_given_monica : ℕ) (candies_eaten_by_tracy : ℕ) (candies_eaten_by_mom : ℕ) 
  (brother_candies_taken : ℕ) (final_candies : ℕ) (h_consume : consumed_candies = 2 / 5 * x) (h_remaining1 : remaining_candies_given_rachel = 1 / 3 * (3 / 5 * x)) 
  (h_remaining2 : remaining_candies_given_monica = 1 / 6 * (3 / 5 * x)) (h_left_after_friends : 3 / 5 * x - (remaining_candies_given_rachel + remaining_candies_given_monica) = 3 / 10 * x)
  (h_candies_left : 3 / 10 * x - (candies_eaten_by_tracy + candies_eaten_by_mom) = final_candies + brother_candies_taken) (h_eaten_tracy : candies_eaten_by_tracy = 10)
  (h_eaten_mom : candies_eaten_by_mom = 10) (h_final : final_candies = 6) (h_brother_bound : 2 ≤ brother_candies_taken ∧ brother_candies_taken ≤ 6) : x = 100 := 
by 
  sorry

end tracy_initial_candies_l120_120821


namespace find_a_l120_120928

theorem find_a {a : ℝ} (h : {x : ℝ | (1/2 : ℝ) < x ∧ x < 2} = {x : ℝ | 0 < ax^2 + 5 * x - 2}) : a = -2 :=
sorry

end find_a_l120_120928


namespace sequence_periodic_a_n_plus_2_eq_a_n_l120_120210

-- Definition of the sequence and conditions
noncomputable def seq (a : ℕ → ℤ) :=
  ∀ n : ℕ, ∃ α k : ℕ, a n = Int.ofNat (2^α) * k ∧ Int.gcd (Int.ofNat k) 2 = 1 ∧ a (n+1) = Int.ofNat (2^α) - k

-- Definition of periodic sequence
def periodic (a : ℕ → ℤ) (d : ℕ) :=
  ∀ n : ℕ, a (n + d) = a n

-- Proving the desired property
theorem sequence_periodic_a_n_plus_2_eq_a_n (a : ℕ → ℤ) (d : ℕ) (h_seq : seq a) (h_periodic : periodic a d) :
  ∀ n : ℕ, a (n + 2) = a n :=
sorry

end sequence_periodic_a_n_plus_2_eq_a_n_l120_120210


namespace paving_cost_l120_120525

variable (L : ℝ) (W : ℝ) (R : ℝ)

def area (L W : ℝ) := L * W
def cost (A R : ℝ) := A * R

theorem paving_cost (hL : L = 5) (hW : W = 4.75) (hR : R = 900) : cost (area L W) R = 21375 :=
by
  sorry

end paving_cost_l120_120525


namespace when_was_p_turned_off_l120_120972

noncomputable def pipe_p_rate := (1/12 : ℚ)  -- Pipe p rate
noncomputable def pipe_q_rate := (1/15 : ℚ)  -- Pipe q rate
noncomputable def combined_rate := (3/20 : ℚ) -- Combined rate of p and q when both are open
noncomputable def time_after_p_off := (1.5 : ℚ)  -- Time for q to fill alone after p is off
noncomputable def fill_cistern (t : ℚ) := combined_rate * t + pipe_q_rate * time_after_p_off

theorem when_was_p_turned_off (t : ℚ) : fill_cistern t = 1 ↔ t = 6 := sorry

end when_was_p_turned_off_l120_120972


namespace range_of_a_l120_120439

theorem range_of_a (x a : ℝ) : (∃ x : ℝ,  |x + 2| + |x - 3| ≤ |a - 1| ) ↔ (a ≤ -4 ∨ a ≥ 6) :=
by
  sorry

end range_of_a_l120_120439


namespace least_even_p_l120_120492

def is_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

theorem least_even_p 
  (p : ℕ) 
  (hp : 2 ∣ p) -- p is an even integer
  (h : is_square (300 * p)) -- 300 * p is the square of an integer
  : p = 3 := 
sorry

end least_even_p_l120_120492


namespace percentage_of_class_are_men_proof_l120_120042

/-- Definition of the problem using the conditions provided. -/
def percentage_of_class_are_men (W M : ℝ) : Prop :=
  -- Conditions based on the problem statement
  M + W = 100 ∧
  0.10 * W + 0.85 * M = 40

/-- The proof statement we need to show: Under the given conditions, the percentage of men (M) is 40. -/
theorem percentage_of_class_are_men_proof (W M : ℝ) :
  percentage_of_class_are_men W M → M = 40 :=
by
  sorry

end percentage_of_class_are_men_proof_l120_120042


namespace age_difference_l120_120736

theorem age_difference (p f : ℕ) (hp : p = 11) (hf : f = 42) : f - p = 31 :=
by
  sorry

end age_difference_l120_120736


namespace cubics_sum_div_abc_eq_three_l120_120159

theorem cubics_sum_div_abc_eq_three {a b c : ℝ} (h₁ : a ≠ 0) (h₂ : b ≠ 0) (h₃ : c ≠ 0) (h₄ : a + b + c = 0) :
  (a^3 + b^3 + c^3) / (a * b * c) = 3 :=
by
  sorry

end cubics_sum_div_abc_eq_three_l120_120159


namespace direction_vector_l1_l120_120129

theorem direction_vector_l1
  (m : ℝ)
  (l₁ : ∀ x y : ℝ, (m + 3) * x + 4 * y + 3 * m - 5 = 0)
  (l₂ : ∀ x y : ℝ, 2 * x + (m + 6) * y - 8 = 0)
  (h_perp : ((m + 3) * 2 = -4 * (m + 6)))
  : ∃ v : ℝ × ℝ, v = (-1, -1/2) :=
by
  sorry

end direction_vector_l1_l120_120129


namespace equi_partite_complex_number_a_l120_120145

-- A complex number z = 1 + (a-1)i
def z (a : ℝ) : ℂ := ⟨1, a - 1⟩

-- Definition of an equi-partite complex number
def is_equi_partite (z : ℂ) : Prop := z.re = z.im

-- The theorem to prove
theorem equi_partite_complex_number_a (a : ℝ) : is_equi_partite (z a) ↔ a = 2 := 
by
  sorry

end equi_partite_complex_number_a_l120_120145


namespace find_a_l120_120434

theorem find_a (k a : ℚ) (hk : 4 * k = 60) (ha : 15 * a - 5 = 60) : a = 13 / 3 :=
by
  sorry

end find_a_l120_120434


namespace sum_arithmetic_sequence_S12_l120_120549

variable {a : ℕ → ℝ} -- Arithmetic sequence a_n
variable {S : ℕ → ℝ} -- Sum of the first n terms S_n

-- Conditions given in the problem
axiom condition1 (n : ℕ) : S n = (n / 2) * (a 1 + a n)
axiom condition2 : a 4 + a 9 = 10

-- Proving that S 12 = 60 given the conditions
theorem sum_arithmetic_sequence_S12 : S 12 = 60 := by
  sorry

end sum_arithmetic_sequence_S12_l120_120549


namespace correct_word_is_any_l120_120755

def words : List String := ["other", "any", "none", "some"]

def is_correct_word (word : String) : Prop :=
  "Jane was asked a lot of questions, but she didn’t answer " ++ word ++ " of them." = 
    "Jane was asked a lot of questions, but she didn’t answer any of them."

theorem correct_word_is_any : is_correct_word "any" :=
by
  sorry

end correct_word_is_any_l120_120755


namespace right_angled_triangles_count_l120_120131

theorem right_angled_triangles_count : 
  ∃ n : ℕ, n = 12 ∧ ∀ (a b c : ℕ), (a = 2016^(1/2)) → (a^2 + b^2 = c^2) →
  (∃ (n k : ℕ), (c - b) = n ∧ (c + b) = k ∧ 2 ∣ n ∧ 2 ∣ k ∧ (n * k = 2016)) :=
by {
  sorry
}

end right_angled_triangles_count_l120_120131


namespace sum_of_squares_l120_120208

theorem sum_of_squares (x y : ℝ) : 2 * x^2 + 2 * y^2 = (x + y)^2 + (x - y)^2 := 
by
  sorry

end sum_of_squares_l120_120208


namespace sum_reciprocals_square_l120_120169

theorem sum_reciprocals_square (x y : ℕ) (h : x * y = 11) : (1 : ℚ) / (↑x ^ 2) + (1 : ℚ) / (↑y ^ 2) = 122 / 121 :=
by
  sorry

end sum_reciprocals_square_l120_120169


namespace euler_disproof_l120_120275

theorem euler_disproof :
  ∃ (n : ℕ), 0 < n ∧ (133^5 + 110^5 + 84^5 + 27^5 = n^5 ∧ n = 144) :=
by
  sorry

end euler_disproof_l120_120275


namespace mia_socks_l120_120374

-- Defining the number of each type of socks
variables {a b c : ℕ}

-- Conditions and constraints
def total_pairs (a b c : ℕ) : Prop := a + b + c = 15
def total_cost (a b c : ℕ) : Prop := 2 * a + 3 * b + 5 * c = 35
def at_least_one (a b c : ℕ) : Prop := a ≥ 1 ∧ b ≥ 1 ∧ c ≥ 1

-- Main theorem to prove the number of 2-dollar pairs of socks
theorem mia_socks : 
  ∀ (a b c : ℕ), 
  total_pairs a b c → 
  total_cost a b c → 
  at_least_one a b c → 
  a = 12 :=
by
  sorry

end mia_socks_l120_120374


namespace negation_of_abs_x_minus_2_lt_3_l120_120534

theorem negation_of_abs_x_minus_2_lt_3 :
  ¬ (∀ x : ℝ, |x - 2| < 3) ↔ ∃ x : ℝ, |x - 2| ≥ 3 :=
by
  sorry

end negation_of_abs_x_minus_2_lt_3_l120_120534


namespace complex_quadrant_l120_120237

theorem complex_quadrant (z : ℂ) (h : z = (↑0 + 1*I) / (1 + 1*I)) : z.re > 0 ∧ z.im > 0 := 
by
  sorry

end complex_quadrant_l120_120237


namespace problem_statement_l120_120480

theorem problem_statement (m : ℂ) (h : m^2 + m - 1 = 0) : m^3 + 2 * m^2 + 2005 = 2006 :=
  sorry

end problem_statement_l120_120480


namespace evaluate_dollar_l120_120758

variable {R : Type} [Field R]

def dollar (a b : R) : R := (a - b) ^ 2

theorem evaluate_dollar (x y : R) : dollar (2 * x + 3 * y) (3 * x - 4 * y) = x ^ 2 - 14 * x * y + 49 * y ^ 2 := by
  sorry

end evaluate_dollar_l120_120758


namespace find_a_plus_b_l120_120291

theorem find_a_plus_b (a b : ℝ) (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = x^2 + a * x + b) 
  (h2 : { x : ℝ | 0 ≤ f x ∧ f x ≤ 6 - x } = { x : ℝ | 2 ≤ x ∧ x ≤ 3 } ∪ {6}) 
  : a + b = 9 := 
sorry

end find_a_plus_b_l120_120291


namespace max_sum_arithmetic_sequence_terms_l120_120666

theorem max_sum_arithmetic_sequence_terms (d : ℝ) (a : ℕ → ℝ) (n : ℕ) 
  (h0 : ∀ n : ℕ, a (n + 1) = a n + d)
  (h1 : d < 0)
  (h2 : a 1 ^ 2 = a 11 ^ 2) : 
  (n = 5) ∨ (n = 6) :=
sorry

end max_sum_arithmetic_sequence_terms_l120_120666


namespace part1_solution_set_part2_range_of_a_l120_120486

noncomputable def f (x : ℝ) : ℝ := abs (4 * x - 1) - abs (x + 2)

-- Part 1: Prove the solution set of f(x) < 8 is -9 / 5 < x < 11 / 3
theorem part1_solution_set : {x : ℝ | f x < 8} = {x : ℝ | -9 / 5 < x ∧ x < 11 / 3} :=
sorry

-- Part 2: Prove the range of a such that the inequality has a solution
theorem part2_range_of_a (a : ℝ) : (∃ x : ℝ, f x + 5 * abs (x + 2) < a^2 - 8 * a) ↔ (a < -1 ∨ a > 9) :=
sorry

end part1_solution_set_part2_range_of_a_l120_120486


namespace total_fruit_punch_l120_120082

/-- Conditions -/
def orange_punch : ℝ := 4.5
def cherry_punch : ℝ := 2 * orange_punch
def apple_juice : ℝ := cherry_punch - 1.5
def pineapple_juice : ℝ := 3
def grape_punch : ℝ := 1.5 * apple_juice

/-- Proof that total fruit punch is 35.25 liters -/
theorem total_fruit_punch :
  orange_punch + cherry_punch + apple_juice + pineapple_juice + grape_punch = 35.25 := by
  sorry

end total_fruit_punch_l120_120082


namespace percentage_by_which_x_is_less_than_y_l120_120461

noncomputable def percentageLess (x y : ℝ) : ℝ :=
  ((y - x) / y) * 100

theorem percentage_by_which_x_is_less_than_y :
  ∀ (x y : ℝ),
  y = 125 + 0.10 * 125 →
  x = 123.75 →
  percentageLess x y = 10 :=
by
  intros x y h1 h2
  rw [h1, h2]
  unfold percentageLess
  sorry

end percentage_by_which_x_is_less_than_y_l120_120461


namespace turns_per_minute_l120_120139

theorem turns_per_minute (x : ℕ) (h₁ : x > 0) (h₂ : 60 / x = (60 / (x + 5)) + 2) :
  60 / x = 6 ∧ 60 / (x + 5) = 4 :=
by sorry

end turns_per_minute_l120_120139


namespace transformed_solution_equiv_l120_120701

noncomputable def quadratic_solution_set (f : ℝ → ℝ) : Set ℝ :=
  {x | f x > 0}

noncomputable def transformed_solution_set (f : ℝ → ℝ) : Set ℝ :=
  {x | f (10^x) > 0}

theorem transformed_solution_equiv (f : ℝ → ℝ) :
  quadratic_solution_set f = {x | x < -1 ∨ x > 1 / 2} →
  transformed_solution_set f = {x | x > -Real.log 2} :=
by sorry

end transformed_solution_equiv_l120_120701


namespace calculation_correct_l120_120189

theorem calculation_correct : 4 * 6 * 8 - 10 / 2 = 187 := by
  sorry

end calculation_correct_l120_120189


namespace mowing_time_l120_120628

theorem mowing_time (length width: ℝ) (swath_width_overlap_rate: ℝ)
                    (walking_speed: ℝ) (ft_per_inch: ℝ)
                    (length_eq: length = 100)
                    (width_eq: width = 120)
                    (swath_eq: swath_width_overlap_rate = 24)
                    (walking_eq: walking_speed = 4500)
                    (conversion_eq: ft_per_inch = 1/12) :
                    (length / walking_speed) * (width / (swath_width_overlap_rate * ft_per_inch)) = 1.33 :=
by
    rw [length_eq, width_eq, swath_eq, walking_eq, conversion_eq]
    exact sorry

end mowing_time_l120_120628


namespace larger_acute_angle_right_triangle_l120_120194

theorem larger_acute_angle_right_triangle (x : ℝ) (h1 : x > 0) (h2 : x + 5 * x = 90) : 5 * x = 75 := by
  sorry

end larger_acute_angle_right_triangle_l120_120194


namespace mark_less_than_kate_and_laura_l120_120365

theorem mark_less_than_kate_and_laura (K : ℝ) (h : K + 2 * K + 3 * K + 4.5 * K = 360) :
  let Pat := 2 * K
  let Mark := 3 * K
  let Laura := 4.5 * K
  let Combined := K + Laura
  Mark - Combined = -85.72 :=
sorry

end mark_less_than_kate_and_laura_l120_120365


namespace values_of_a_and_b_l120_120771

theorem values_of_a_and_b (a b : ℝ) :
  (∀ x : ℝ, x ≥ -1 → a * x^2 + b * x + a^2 - 1 ≤ 0) →
  a = 0 ∧ b = -1 :=
sorry

end values_of_a_and_b_l120_120771


namespace closest_fraction_l120_120153

theorem closest_fraction (n : ℤ) : 
  let frac1 := 37 / 57 
  let closest := 15 / 23
  n = 15 ∧ abs (851 - 57 * n) = min (abs (851 - 57 * 14)) (abs (851 - 57 * 15)) :=
by
  let frac1 := (37 : ℚ) / 57
  let closest := (15 : ℚ) / 23
  have h : 37 * 23 = 851 := by norm_num
  have denom : 57 * 23 = 1311 := by norm_num
  let num := 851
  sorry

end closest_fraction_l120_120153


namespace no_tiling_with_seven_sided_convex_l120_120757

noncomputable def Polygon := {n : ℕ // 3 ≤ n}

def convex (M : Polygon) : Prop := sorry

def tiles_plane (M : Polygon) : Prop := sorry

theorem no_tiling_with_seven_sided_convex (M : Polygon) (h_convex : convex M) (h_sides : 7 ≤ M.1) : ¬ tiles_plane M := sorry

end no_tiling_with_seven_sided_convex_l120_120757


namespace sum_of_ages_in_10_years_l120_120351

-- Define the initial conditions about Ann's and Tom's ages
def AnnCurrentAge : ℕ := 6
def TomCurrentAge : ℕ := 2 * AnnCurrentAge

-- Define their ages 10 years later
def AnnAgeIn10Years : ℕ := AnnCurrentAge + 10
def TomAgeIn10Years : ℕ := TomCurrentAge + 10

-- The proof statement
theorem sum_of_ages_in_10_years : AnnAgeIn10Years + TomAgeIn10Years = 38 := by
  sorry

end sum_of_ages_in_10_years_l120_120351


namespace owls_joined_l120_120932

theorem owls_joined (initial_owls : ℕ) (total_owls : ℕ) (join_owls : ℕ) 
  (h_initial : initial_owls = 3) (h_total : total_owls = 5) : join_owls = 2 :=
by {
  -- Sorry is used to skip the proof
  sorry
}

end owls_joined_l120_120932


namespace ticket_ratio_proof_l120_120723

-- Define the initial number of tickets Tate has.
def initial_tate_tickets : ℕ := 32

-- Define the additional tickets Tate buys.
def additional_tickets : ℕ := 2

-- Define the total tickets they have together.
def combined_tickets : ℕ := 51

-- Calculate Tate's total number of tickets after buying more tickets.
def total_tate_tickets := initial_tate_tickets + additional_tickets

-- Define the number of tickets Peyton has.
def peyton_tickets := combined_tickets - total_tate_tickets

-- Define the ratio of Peyton's tickets to Tate's tickets.
def tickets_ratio := peyton_tickets / total_tate_tickets

theorem ticket_ratio_proof : tickets_ratio = 1 / 2 :=
by
  unfold tickets_ratio peyton_tickets total_tate_tickets initial_tate_tickets additional_tickets
  norm_num
  sorry

end ticket_ratio_proof_l120_120723


namespace largest_difference_l120_120034

noncomputable def A := 3 * 2023^2024
noncomputable def B := 2023^2024
noncomputable def C := 2022 * 2023^2023
noncomputable def D := 3 * 2023^2023
noncomputable def E := 2023^2023
noncomputable def F := 2023^2022

theorem largest_difference :
  max (A - B) (max (B - C) (max (C - D) (max (D - E) (E - F)))) = A - B := 
sorry

end largest_difference_l120_120034


namespace volume_in_cubic_meters_l120_120850

noncomputable def mass_condition : ℝ := 100 -- mass in kg
noncomputable def volume_per_gram : ℝ := 10 -- volume in cubic centimeters per gram
noncomputable def volume_per_kg : ℝ := volume_per_gram * 1000 -- volume in cubic centimeters per kg
noncomputable def mass_in_kg : ℝ := mass_condition

theorem volume_in_cubic_meters (h : mass_in_kg = 100)
    (v_per_kg : volume_per_kg = volume_per_gram * 1000) :
  (mass_in_kg * volume_per_kg) / 1000000 = 1 := by
  sorry

end volume_in_cubic_meters_l120_120850


namespace smallest_x_l120_120955

theorem smallest_x (x : ℕ) (h₁ : x % 3 = 2) (h₂ : x % 4 = 3) (h₃ : x % 5 = 4) : x = 59 :=
by
  sorry

end smallest_x_l120_120955


namespace find_first_day_income_l120_120143

def income_4 (i2 i3 i4 i5 : ℕ) : ℕ := i2 + i3 + i4 + i5

def total_income_5 (average_income : ℕ) : ℕ := 5 * average_income

def income_1 (total : ℕ) (known : ℕ) : ℕ := total - known

theorem find_first_day_income (i2 i3 i4 i5 a income5 : ℕ) (h1 : income_4 i2 i3 i4 i5 = 1800)
  (h2 : a = 440)
  (h3 : total_income_5 a = income5)
  : income_1 income5 (income_4 i2 i3 i4 i5) = 400 := 
sorry

end find_first_day_income_l120_120143


namespace range_of_b_l120_120059

theorem range_of_b (a b : ℝ) : 
  (∀ x : ℝ, -3 < x ∧ x < 1 → (1 - a) * x^2 - 4 * x + 6 > 0) ∧
  (∀ x : ℝ, 3 * x^2 + b * x + 3 ≥ 0) →
  (-6 ≤ b ∧ b ≤ 6) :=
by
  sorry

end range_of_b_l120_120059


namespace geometric_series_sum_eq_4_div_3_l120_120448

theorem geometric_series_sum_eq_4_div_3 (a : ℝ) (r : ℝ) (h₀ : a = 1) (h₁ : r = 1 / 4) :
  ∑' n : ℕ, a * r^n = 4 / 3 := by
  sorry

end geometric_series_sum_eq_4_div_3_l120_120448


namespace range_of_ab_min_value_of_ab_plus_inv_ab_l120_120559

theorem range_of_ab (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 1) : 0 < a * b ∧ a * b ≤ 1 / 4 :=
sorry

theorem min_value_of_ab_plus_inv_ab (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 1) : 
  (∃ ab, ab = a * b ∧ ab + 1 / ab = 17 / 4) :=
sorry

end range_of_ab_min_value_of_ab_plus_inv_ab_l120_120559


namespace two_rooks_non_attacking_two_kings_non_attacking_two_bishops_non_attacking_two_knights_non_attacking_two_queens_non_attacking_l120_120556

noncomputable def rooks_non_attacking : Nat :=
  8 * 8 * 7 * 7 / 2

theorem two_rooks_non_attacking : rooks_non_attacking = 1568 := by
  sorry

noncomputable def kings_non_attacking : Nat :=
  (4 * 60 + 24 * 58 + 36 * 55 + 24 * 55 + 4 * 50) / 2

theorem two_kings_non_attacking : kings_non_attacking = 1806 := by
  sorry

noncomputable def bishops_non_attacking : Nat :=
  (28 * 25 + 20 * 54 + 12 * 52 + 4 * 50) / 2

theorem two_bishops_non_attacking : bishops_non_attacking = 1736 := by
  sorry

noncomputable def knights_non_attacking : Nat :=
  (4 * 61 + 8 * 60 + 20 * 59 + 16 * 57 + 15 * 55) / 2

theorem two_knights_non_attacking : knights_non_attacking = 1848 := by
  sorry

noncomputable def queens_non_attacking : Nat :=
  (28 * 42 + 20 * 40 + 12 * 38 + 4 * 36) / 2

theorem two_queens_non_attacking : queens_non_attacking = 1288 := by
  sorry

end two_rooks_non_attacking_two_kings_non_attacking_two_bishops_non_attacking_two_knights_non_attacking_two_queens_non_attacking_l120_120556


namespace first_position_remainder_one_l120_120026

theorem first_position_remainder_one (a : ℕ) (h1 : 1 ≤ a ∧ a ≤ 2023)
(h2 : ∀ b c d : ℕ, b = a ∧ c = a + 2 ∧ d = a + 4 → 
  b % 3 ≠ c % 3 ∧ c % 3 ≠ d % 3 ∧ d % 3 ≠ b % 3):
  a % 3 = 1 :=
sorry

end first_position_remainder_one_l120_120026


namespace max_a1_value_l120_120361

theorem max_a1_value (a : ℕ → ℕ) (h1 : ∀ n : ℕ, a (n+2) = a n + a (n+1))
    (h2 : ∀ n : ℕ, a n > 0) (h3 : a 5 = 60) : a 1 ≤ 11 :=
by 
  sorry

end max_a1_value_l120_120361


namespace problem1_problem2_l120_120989

def f (x : ℝ) := |x - 1| + |x + 2|

def T (a : ℝ) := -Real.sqrt 3 < a ∧ a < Real.sqrt 3

theorem problem1 (a : ℝ) : (∀ x : ℝ, f x > a^2) ↔ T a :=
by
  sorry

theorem problem2 (m n : ℝ) (h1 : T m) (h2 : T n) : Real.sqrt 3 * |m + n| < |m * n + 3| :=
by
  sorry

end problem1_problem2_l120_120989


namespace car_return_speed_l120_120465

noncomputable def round_trip_speed (d : ℝ) (r : ℝ) : ℝ :=
  let travel_time_to_B := d / 75
  let break_time := 1 / 2
  let travel_time_to_A := d / r
  let total_time := travel_time_to_B + travel_time_to_A + break_time
  let total_distance := 2 * d
  total_distance / total_time

theorem car_return_speed :
  let d := 150
  let avg_speed := 50
  round_trip_speed d 42.857 = avg_speed :=
by
  sorry

end car_return_speed_l120_120465


namespace root_of_quadratic_eq_is_two_l120_120539

theorem root_of_quadratic_eq_is_two (k : ℝ) : (2^2 - 3 * 2 + k = 0) → k = 2 :=
by
  intro h
  sorry

end root_of_quadratic_eq_is_two_l120_120539


namespace binary_predecessor_l120_120293

theorem binary_predecessor (N : ℕ) (hN : N = 0b11000) : 0b10111 + 1 = N := 
by
  sorry

end binary_predecessor_l120_120293


namespace eighteen_gon_vertex_number_l120_120671

theorem eighteen_gon_vertex_number (a b : ℕ) (P : ℕ) (h₁ : a = 20) (h₂ : b = 18) (h₃ : P = a + b) : P = 38 :=
by
  rw [h₁, h₂] at h₃
  exact h₃

end eighteen_gon_vertex_number_l120_120671


namespace reciprocal_of_sum_l120_120653

-- Define the fractions
def a := (1: ℚ) / 2
def b := (1: ℚ) / 3

-- Define their sum
def c := a + b

-- Define the expected reciprocal
def reciprocal := (6: ℚ) / 5

-- The theorem we want to prove:
theorem reciprocal_of_sum : (c⁻¹ = reciprocal) :=
by 
  sorry

end reciprocal_of_sum_l120_120653


namespace hyperbola_asymptotes_l120_120554

theorem hyperbola_asymptotes :
  ∀ x y : ℝ, x^2 - y^2 / 4 = 1 → (y = 2 * x ∨ y = -2 * x) :=
by
  intros x y h
  sorry

end hyperbola_asymptotes_l120_120554


namespace daisies_per_bouquet_l120_120070

def total_bouquets := 20
def rose_bouquets := 10
def roses_per_rose_bouquet := 12
def total_flowers_sold := 190

def total_roses_sold := rose_bouquets * roses_per_rose_bouquet
def daisy_bouquets := total_bouquets - rose_bouquets
def total_daisies_sold := total_flowers_sold - total_roses_sold

theorem daisies_per_bouquet :
  (total_daisies_sold / daisy_bouquets = 7) := sorry

end daisies_per_bouquet_l120_120070


namespace range_of_y_function_l120_120110

def range_of_function : Set ℝ :=
  {y : ℝ | ∃ (x : ℝ), x ≠ -2 ∧ y = (x^2 + 5*x + 6)/(x+2)}

theorem range_of_y_function :
  range_of_function = {y : ℝ | y ≠ 1} :=
by
  sorry

end range_of_y_function_l120_120110


namespace pizza_eating_group_l120_120716

theorem pizza_eating_group (x y : ℕ) (h1 : 6 * x + 2 * y ≥ 49) (h2 : 7 * x + 3 * y ≤ 59) : x = 8 ∧ y = 2 := by
  sorry

end pizza_eating_group_l120_120716


namespace perfect_square_trinomial_l120_120948

theorem perfect_square_trinomial (k : ℝ) :
  ∃ k, (∀ x, (4 * x^2 - 2 * k * x + 1) = (2 * x + 1)^2 ∨ (4 * x^2 - 2 * k * x + 1) = (2 * x - 1)^2) → 
  (k = 2 ∨ k = -2) := by
  sorry

end perfect_square_trinomial_l120_120948


namespace not_periodic_cos_add_cos_sqrt2_l120_120507

noncomputable def f (x : ℝ) : ℝ := Real.cos x + Real.cos (x * Real.sqrt 2)

theorem not_periodic_cos_add_cos_sqrt2 :
  ¬(∃ T : ℝ, T > 0 ∧ ∀ x : ℝ, f (x + T) = f x) :=
sorry

end not_periodic_cos_add_cos_sqrt2_l120_120507


namespace minimum_value_of_expression_l120_120750

noncomputable def expr (a b c : ℝ) : ℝ := 8 * a^3 + 27 * b^3 + 64 * c^3 + 27 / (8 * a * b * c)

theorem minimum_value_of_expression (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) :
  expr a b c ≥ 18 * Real.sqrt 3 := 
by
  sorry

end minimum_value_of_expression_l120_120750


namespace first_day_of_month_is_tuesday_l120_120633

theorem first_day_of_month_is_tuesday (day23_is_wednesday : (23 % 7 = 3)) : (1 % 7 = 2) :=
sorry

end first_day_of_month_is_tuesday_l120_120633


namespace PQRS_value_l120_120609

theorem PQRS_value :
  let P := (Real.sqrt 2011 + Real.sqrt 2010)
  let Q := (-Real.sqrt 2011 - Real.sqrt 2010)
  let R := (Real.sqrt 2011 - Real.sqrt 2010)
  let S := (Real.sqrt 2010 - Real.sqrt 2011)
  P * Q * R * S = -1 :=
by
  sorry

end PQRS_value_l120_120609


namespace fisher_eligibility_l120_120302

theorem fisher_eligibility (A1 A2 S : ℕ) (hA1 : A1 = 84) (hS : S = 82) :
  (S ≥ 80) → (A1 + A2 ≥ 170) → (A2 = 86) :=
by
  sorry

end fisher_eligibility_l120_120302


namespace known_number_l120_120328

theorem known_number (A B : ℕ) (h_hcf : 1 / (Nat.gcd A B) = 1 / 15) (h_lcm : 1 / Nat.lcm A B = 1 / 312) (h_B : B = 195) : A = 24 :=
by
  -- Skipping proof
  sorry

end known_number_l120_120328


namespace solve_for_a_l120_120961

theorem solve_for_a (x a : ℝ) (h : x = 3) (eq : 5 * x - a = 8) : a = 7 :=
by
  -- sorry to skip the proof as instructed
  sorry

end solve_for_a_l120_120961


namespace blueberry_jelly_amount_l120_120188

-- Definition of the conditions
def total_jelly : ℕ := 6310
def strawberry_jelly : ℕ := 1792

-- Formal statement of the problem
theorem blueberry_jelly_amount : 
  total_jelly - strawberry_jelly = 4518 :=
by
  sorry

end blueberry_jelly_amount_l120_120188


namespace rectangle_width_is_pi_l120_120845

theorem rectangle_width_is_pi (w : ℝ) (h1 : real_w ≠ 0)
    (h2 : ∀ w, ∃ length, length = 2 * w)
    (h3 : ∀ w, 2 * (length + w) = 6 * w)
    (h4 : 2 * (2 * w + w) = 6 * π) : 
    w = π :=
by {
  sorry -- The proof would go here.
}

end rectangle_width_is_pi_l120_120845


namespace inequality_l120_120001

theorem inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  1 / (a^3 + b^3 + a * b * c) + 1 / (b^3 + c^3 + a * b * c) + 1 / (c^3 + a^3 + a * b * c) ≤ 1 / (a * b * c) :=
sorry

end inequality_l120_120001


namespace price_of_adult_ticket_eq_32_l120_120217

theorem price_of_adult_ticket_eq_32 
  (num_adults : ℕ)
  (num_children : ℕ)
  (price_child_ticket : ℕ)
  (price_adult_ticket : ℕ)
  (total_collected : ℕ)
  (h1 : num_adults = 400)
  (h2 : num_children = 200)
  (h3 : price_adult_ticket = 2 * price_child_ticket)
  (h4 : total_collected = 16000)
  (h5 : total_collected = num_adults * price_adult_ticket + num_children * price_child_ticket)
  : price_adult_ticket = 32 := 
by
  sorry

end price_of_adult_ticket_eq_32_l120_120217


namespace find_J_l120_120894

variables (J S B : ℕ)

-- Conditions
def condition1 : Prop := J - 20 = 2 * S
def condition2 : Prop := B = J / 2
def condition3 : Prop := J + S + B = 330
def condition4 : Prop := (J - 20) + S + B = 318

-- Theorem to prove
theorem find_J (h1 : condition1 J S) (h2 : condition2 J B) (h3 : condition3 J S B) (h4 : condition4 J S B) :
  J = 170 :=
sorry

end find_J_l120_120894


namespace percentage_in_quarters_l120_120438

theorem percentage_in_quarters:
  let dimes : ℕ := 40
  let quarters : ℕ := 30
  let value_dimes : ℕ := dimes * 10
  let value_quarters : ℕ := quarters * 25
  let total_value : ℕ := value_dimes + value_quarters
  let percentage_quarters : ℚ := (value_quarters : ℚ) / total_value * 100
  percentage_quarters = 65.22 := sorry

end percentage_in_quarters_l120_120438


namespace smallest_prime_divides_l120_120627

theorem smallest_prime_divides (p : ℕ) (a : ℕ) 
  (h1 : Prime p) (h2 : p > 100) (h3 : a > 1) (h4 : p ∣ (a^89 - 1) / (a - 1)) :
  p = 179 := 
sorry

end smallest_prime_divides_l120_120627


namespace pairs_with_green_shirts_l120_120238

theorem pairs_with_green_shirts (red_shirts green_shirts total_pairs red_pairs : ℕ) 
    (h1 : red_shirts = 70) 
    (h2 : green_shirts = 58) 
    (h3 : total_pairs = 64) 
    (h4 : red_pairs = 34) 
    : (∃ green_pairs : ℕ, green_pairs = 28) := 
by 
    sorry

end pairs_with_green_shirts_l120_120238


namespace nonnegative_integer_pairs_solution_l120_120787

theorem nonnegative_integer_pairs_solution :
  ∀ (x y: ℕ), ((x * y + 2) ^ 2 = x^2 + y^2) ↔ (x = 0 ∧ y = 2) ∨ (x = 2 ∧ y = 0) :=
by
  sorry

end nonnegative_integer_pairs_solution_l120_120787


namespace arccos_half_eq_pi_div_3_l120_120514

theorem arccos_half_eq_pi_div_3 :
  Real.arccos (1 / 2) = Real.pi / 3 :=
sorry

end arccos_half_eq_pi_div_3_l120_120514


namespace sqrt_diff_approx_l120_120247

noncomputable def x : ℝ := Real.sqrt 50 - Real.sqrt 48

theorem sqrt_diff_approx : abs (x - 0.14) < 0.01 :=
by
  sorry

end sqrt_diff_approx_l120_120247


namespace alice_bob_age_difference_18_l120_120728

-- Define Alice's and Bob's ages with the given constraints
def is_odd (n : ℕ) : Prop := n % 2 = 1

def alice_age (a b : ℕ) : ℕ := 10 * a + b
def bob_age (a b : ℕ) : ℕ := 10 * b + a

theorem alice_bob_age_difference_18 (a b : ℕ) (ha : is_odd a) (hb : is_odd b)
  (h : alice_age a b + 7 = 3 * (bob_age a b + 7)) : alice_age a b - bob_age a b = 18 :=
sorry

end alice_bob_age_difference_18_l120_120728


namespace arithmetic_seq_a7_constant_l120_120997

variable {α : Type*} [AddCommGroup α] [Module ℤ α]

-- Definition of an arithmetic sequence
def is_arithmetic_seq (a : ℕ → α) : Prop :=
∃ d : α, ∀ n : ℕ, a (n + 1) = a n + d

-- Given arithmetic sequence {a_n}
variable (a : ℕ → α)
-- Given the property that a_2 + a_4 + a_{15} is a constant
variable (C : α)
variable (h : is_arithmetic_seq a)
variable (h_constant : a 2 + a 4 + a 15 = C)

-- Prove that a_7 is a constant
theorem arithmetic_seq_a7_constant (h : is_arithmetic_seq a) (h_constant : a 2 + a 4 + a 15 = C) : ∃ k : α, a 7 = k :=
by
  sorry

end arithmetic_seq_a7_constant_l120_120997


namespace distinct_real_roots_max_abs_gt_2_l120_120994

theorem distinct_real_roots_max_abs_gt_2 
  (r1 r2 r3 q : ℝ)
  (h_distinct : r1 ≠ r2 ∧ r2 ≠ r3 ∧ r1 ≠ r3)
  (h_sum : r1 + r2 + r3 = -q)
  (h_product : r1 * r2 * r3 = -9)
  (h_sum_prod : r1 * r2 + r2 * r3 + r3 * r1 = 6)
  (h_nonzero_discriminant : q^2 * 6^2 - 4 * 6^3 - 4 * q^3 * 9 - 27 * 9^2 + 18 * q * 6 * (-9) ≠ 0) :
  max (|r1|) (max (|r2|) (|r3|)) > 2 :=
sorry

end distinct_real_roots_max_abs_gt_2_l120_120994


namespace solve_equation_l120_120552

theorem solve_equation (x : ℝ) (h : 3 * x ≠ 0) (h2 : x + 2 ≠ 0) : (2 / (3 * x) = 1 / (x + 2)) ↔ x = 4 := by
  sorry

end solve_equation_l120_120552


namespace compare_abc_l120_120090

noncomputable def a : ℝ := (1 / 4) * Real.logb 2 3
noncomputable def b : ℝ := 1 / 2
noncomputable def c : ℝ := (1 / 2) * Real.logb 5 3

theorem compare_abc : c < a ∧ a < b := sorry

end compare_abc_l120_120090


namespace sports_lottery_systematic_sampling_l120_120319

-- Definition of the sports lottery condition
def is_first_prize_ticket (n : ℕ) : Prop := n % 1000 = 345

-- Statement of the proof problem
theorem sports_lottery_systematic_sampling :
  (∀ n : ℕ, 1 ≤ n ∧ n ≤ 100000 → is_first_prize_ticket n) →
  ∃ interval, (∀ segment_start : ℕ,  segment_start < 1000 → is_first_prize_ticket (segment_start + interval * 999))
  := by sorry

end sports_lottery_systematic_sampling_l120_120319


namespace cricket_average_increase_l120_120017

-- Define the conditions as variables
variables (innings_initial : ℕ) (average_initial : ℕ) (runs_next_innings : ℕ)
variables (runs_increase : ℕ)

-- Given conditions
def conditions := (innings_initial = 13) ∧ (average_initial = 22) ∧ (runs_next_innings = 92)

-- Target: Calculate the desired increase in average (runs_increase)
theorem cricket_average_increase (h : conditions innings_initial average_initial runs_next_innings) :
  runs_increase = 5 :=
  sorry

end cricket_average_increase_l120_120017


namespace tiles_count_l120_120340

variable (c r : ℕ)

-- given: r = 10
def initial_rows_eq : Prop := r = 10

-- assertion: number of tiles is conserved after rearrangement
def tiles_conserved : Prop := c * r = (c - 2) * (r + 4)

-- desired: total number of tiles is 70
def total_tiles : Prop := c * r = 70

theorem tiles_count (h1 : initial_rows_eq r) (h2 : tiles_conserved c r) : total_tiles c r :=
by
  subst h1
  sorry

end tiles_count_l120_120340


namespace find_number_l120_120123

theorem find_number (x : ℝ) (h : (5/3) * x = 45) : x = 27 :=
by
  sorry

end find_number_l120_120123


namespace carter_baseball_cards_l120_120485

theorem carter_baseball_cards (m c : ℕ) (h1 : m = 210) (h2 : m = c + 58) : c = 152 := 
by
  sorry

end carter_baseball_cards_l120_120485


namespace silver_coin_worth_l120_120386

theorem silver_coin_worth :
  ∀ (g : ℕ) (S : ℕ) (n_gold n_silver cash : ℕ), 
  g = 50 →
  n_gold = 3 →
  n_silver = 5 →
  cash = 30 →
  n_gold * g + n_silver * S + cash = 305 →
  S = 25 :=
by
  intros g S n_gold n_silver cash
  intros hg hng hnsi hcash htotal
  sorry

end silver_coin_worth_l120_120386


namespace problem_expression_value_l120_120478

theorem problem_expression_value {a b c k1 k2 : ℂ} 
  (h_root : ∀ x, x^3 - k1 * x - k2 = 0 → x = a ∨ x = b ∨ x = c) 
  (h_condition : k1 + k2 ≠ 1)
  (h_vieta1 : a + b + c = 0)
  (h_vieta2 : a * b + b * c + c * a = -k1)
  (h_vieta3 : a * b * c = k2) :
  (1 + a)/(1 - a) + (1 + b)/(1 - b) + (1 + c)/(1 - c) = 
  (3 + k1 + 3 * k2)/(1 - k1 - k2) :=
by
  sorry

end problem_expression_value_l120_120478


namespace employed_males_percentage_l120_120069

theorem employed_males_percentage (p_employed : ℝ) (p_employed_females : ℝ) : 
  (64 / 100) * (1 - 21.875 / 100) * 100 = 49.96 :=
by
  sorry

end employed_males_percentage_l120_120069


namespace range_of_x_for_positive_function_value_l120_120381

variable {R : Type*} [LinearOrderedField R]

def even_function (f : R → R) := ∀ x, f (-x) = f x

def monotonically_decreasing_on_nonnegatives (f : R → R) := ∀ x y, 0 ≤ x → 0 ≤ y → x ≤ y → f y ≤ f x

theorem range_of_x_for_positive_function_value (f : R → R)
  (hf_even : even_function f)
  (hf_monotonic : monotonically_decreasing_on_nonnegatives f)
  (hf_at_2 : f 2 = 0)
  (hf_positive : ∀ x, f (x - 1) > 0) :
  ∀ x, -1 < x ∧ x < 3 := sorry

end range_of_x_for_positive_function_value_l120_120381


namespace obtain_100_using_fewer_than_ten_threes_example1_obtain_100_using_fewer_than_ten_threes_example2_l120_120641

-- The main theorem states that 100 can be obtained using fewer than ten 3's.

theorem obtain_100_using_fewer_than_ten_threes_example1 :
  100 = (333 / 3) - (33 / 3) :=
by
  sorry

theorem obtain_100_using_fewer_than_ten_threes_example2 :
  100 = (33 * 3) + (3 / 3) :=
by
  sorry

end obtain_100_using_fewer_than_ten_threes_example1_obtain_100_using_fewer_than_ten_threes_example2_l120_120641


namespace pascal_current_speed_l120_120352

variable (v : ℝ)
variable (h₁ : v > 0) -- current speed is positive

-- Conditions
variable (h₂ : 96 / (v - 4) = 96 / (1.5 * v) + 16)

-- Proving the speed
theorem pascal_current_speed (h₁ : v > 0) (h₂ : 96 / (v - 4) = 96 / (1.5 * v) + 16) : v = 8 :=
sorry

end pascal_current_speed_l120_120352


namespace range_of_a_l120_120104

open Real

namespace PropositionProof

-- Define propositions p and q
def p (a x : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0
def q (x : ℝ) : Prop := (x^2 - x - 6 ≤ 0) ∨ (x^2 + 2*x - 8 > 0)

theorem range_of_a (a : ℝ) (h : a < 0) :
  (¬ ∀ x, ¬ p a x → ∀ x, ¬ q x) ↔ (a ≤ -4 ∨ -2/3 ≤ a ∧ a < 0) :=
sorry

end PropositionProof

end range_of_a_l120_120104


namespace Angelina_speeds_l120_120806

def distance_home_to_grocery := 960
def distance_grocery_to_gym := 480
def distance_gym_to_library := 720
def time_diff_grocery_to_gym := 40
def time_diff_gym_to_library := 20

noncomputable def initial_speed (v : ℝ) :=
  (distance_home_to_grocery : ℝ) = (v * (960 / v)) ∧
  (distance_grocery_to_gym : ℝ) = (2 * v * (240 / v)) ∧
  (distance_gym_to_library : ℝ) = (3 * v * (720 / v))

theorem Angelina_speeds (v : ℝ) :
  initial_speed v →
  v = 18 ∧ 2 * v = 36 ∧ 3 * v = 54 :=
by
  sorry

end Angelina_speeds_l120_120806


namespace cyclic_sum_inequality_l120_120484

open Real

theorem cyclic_sum_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a * b * c = 1) :
  1 / (1 + a + b) + 1 / (1 + b + c) + 1 / (1 + c + a) ≤ 1 := 
  sorry

end cyclic_sum_inequality_l120_120484


namespace sum_a6_a7_a8_l120_120092

-- Sequence definition and sum of the first n terms
def S (n : ℕ) : ℕ := n^2 + 3 * n

theorem sum_a6_a7_a8 : S 8 - S 5 = 48 :=
by
  -- Definition and proof details are skipped
  sorry

end sum_a6_a7_a8_l120_120092


namespace XY_sum_l120_120367

theorem XY_sum (A B C D X Y : ℕ) 
  (h1 : A + B + C + D = 22) 
  (h2 : X = A + B) 
  (h3 : Y = C + D) 
  : X + Y = 4 := 
  sorry

end XY_sum_l120_120367


namespace range_of_m_l120_120211

theorem range_of_m (m : ℝ) : (∃ (x y : ℝ), x^2 + y^2 - 2*x - 4*y + m = 0) → m < 5 :=
by
  sorry

end range_of_m_l120_120211


namespace polynomial_composite_l120_120137

theorem polynomial_composite (x : ℕ) : ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ 4 * x^3 + 6 * x^2 + 4 * x + 1 = a * b :=
by
  sorry

end polynomial_composite_l120_120137


namespace fraction_calls_by_team_B_l120_120370

-- Define the conditions
variables (A B C : ℝ)
axiom ratio_agents : A = (5 / 8) * B
axiom ratio_calls : ∀ (c : ℝ), c = (6 / 5) * C

-- Prove the fraction of the total calls processed by team B
theorem fraction_calls_by_team_B 
  (h1 : A = (5 / 8) * B)
  (h2 : ∀ (c : ℝ), c = (6 / 5) * C) :
  (B * C) / ((5 / 8) * B * (6 / 5) * C + B * C) = 4 / 7 :=
by {
  -- proof is omitted, so we use sorry
  sorry
}

end fraction_calls_by_team_B_l120_120370


namespace width_of_plot_is_correct_l120_120764

-- Definitions based on the given conditions
def cost_per_acre_per_month : ℝ := 60
def total_monthly_rent : ℝ := 600
def length_of_plot : ℝ := 360
def sq_feet_per_acre : ℝ := 43560

-- Theorems to be proved based on the conditions and the correct answer
theorem width_of_plot_is_correct :
  let number_of_acres := total_monthly_rent / cost_per_acre_per_month
  let total_sq_footage := number_of_acres * sq_feet_per_acre
  let width_of_plot := total_sq_footage / length_of_plot
  width_of_plot = 1210 :=
by 
  sorry

end width_of_plot_is_correct_l120_120764


namespace union_of_A_and_B_l120_120537

variable {α : Type*}

def A (x : ℝ) : Prop := x - 1 > 0
def B (x : ℝ) : Prop := 0 < x ∧ x ≤ 3

theorem union_of_A_and_B : ∀ x : ℝ, (A x ∨ B x) ↔ (0 < x) :=
by
  sorry

end union_of_A_and_B_l120_120537


namespace mean_of_added_numbers_l120_120490

noncomputable def mean (a : List ℚ) : ℚ :=
  (a.sum) / (a.length)

theorem mean_of_added_numbers 
  (sum_eight_numbers : ℚ)
  (sum_eleven_numbers : ℚ)
  (x y z : ℚ)
  (h_eight : sum_eight_numbers = 8 * 72)
  (h_eleven : sum_eleven_numbers = 11 * 85)
  (h_sum_added : x + y + z = sum_eleven_numbers - sum_eight_numbers) :
  (x + y + z) / 3 = 119 + 2/3 := 
sorry

end mean_of_added_numbers_l120_120490


namespace boat_speed_in_still_water_l120_120993

variables (V_b V_c V_w : ℝ)

-- Conditions from the problem
def speed_upstream (V_b V_c V_w : ℝ) : ℝ := V_b - V_c - V_w
def water_current_range (V_c : ℝ) : Prop := 2 ≤ V_c ∧ V_c ≤ 4
def wind_resistance_range (V_w : ℝ) : Prop := -1 ≤ V_w ∧ V_w ≤ 1
def upstream_speed : Prop := speed_upstream V_b 4 (2 - (-1)) + (2 - -1) = 4

-- Statement of the proof problem
theorem boat_speed_in_still_water :
  (∀ V_c V_w, water_current_range V_c → wind_resistance_range V_w → speed_upstream V_b V_c V_w = 4) → V_b = 7 :=
by
  sorry

end boat_speed_in_still_water_l120_120993


namespace ordered_triples_count_l120_120149

open Real

theorem ordered_triples_count :
  ∃ (S : Finset (ℝ × ℝ × ℝ)),
    (∀ (a b c : ℝ), (a, b, c) ∈ S ↔ (a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ ab = c ∧ bc = a + b ∧ ca = b)) ∧
    S.card = 2 := 
sorry

end ordered_triples_count_l120_120149


namespace greatest_third_term_of_arithmetic_sequence_l120_120150

theorem greatest_third_term_of_arithmetic_sequence (a d : ℕ) (h : 4 * a + 6 * d = 46) : a + 2 * d ≤ 15 :=
sorry

end greatest_third_term_of_arithmetic_sequence_l120_120150


namespace stratified_sampling_l120_120337

-- Definition of the given variables and conditions
def total_students_grade10 : ℕ := 30
def total_students_grade11 : ℕ := 40
def selected_students_grade11 : ℕ := 8

-- Implementation of the stratified sampling proportion requirement
theorem stratified_sampling (x : ℕ) (hx : (x : ℚ) / total_students_grade10 = (selected_students_grade11 : ℚ) / total_students_grade11) :
  x = 6 :=
by
  sorry

end stratified_sampling_l120_120337


namespace range_of_a_l120_120475

theorem range_of_a (a : ℝ) : (∀ x : ℝ, |x - 1| + |x + 1| > a) ↔ a < 2 := 
sorry

end range_of_a_l120_120475


namespace correct_statement_l120_120447

-- Define the conditions as assumptions

/-- Condition 1: To understand the service life of a batch of new energy batteries, a sampling survey can be used. -/
def condition1 : Prop := True

/-- Condition 2: If the probability of winning a lottery is 2%, then buying 50 of these lottery tickets at once will definitely win. -/
def condition2 : Prop := False

/-- Condition 3: If the average of two sets of data, A and B, is the same, SA^2=2.3, SB^2=4.24, then set B is more stable. -/
def condition3 : Prop := False

/-- Condition 4: Rolling a die with uniform density and getting a score of 0 is a certain event. -/
def condition4 : Prop := False

-- The main theorem to prove the correct statement is A
theorem correct_statement : condition1 = True ∧ condition2 = False ∧ condition3 = False ∧ condition4 = False :=
by
  constructor; repeat { try { exact True.intro }; try { exact False.elim (by sorry) } }

end correct_statement_l120_120447


namespace find_two_digit_number_l120_120213

theorem find_two_digit_number : ∃ (x y : ℕ), 1 ≤ x ∧ x ≤ 9 ∧ 0 ≤ y ∧ y ≤ 9 ∧ 10 * x + y = x^3 + y^2 ∧ 10 * x + y = 24 := by
  sorry

end find_two_digit_number_l120_120213


namespace margo_paired_with_irma_probability_l120_120835

theorem margo_paired_with_irma_probability :
  let n := 15
  let total_outcomes := n
  let favorable_outcomes := 1
  let probability := favorable_outcomes / total_outcomes
  probability = (1 / 15) :=
by
  let n := 15
  let total_outcomes := n
  let favorable_outcomes := 1
  let probability := favorable_outcomes / total_outcomes
  have h : probability = 1 / 15 := by
    -- skipping the proof details as per instructions
    sorry
  exact h

end margo_paired_with_irma_probability_l120_120835


namespace expand_expression_l120_120013

theorem expand_expression (x : ℝ) : (15 * x + 17 + 3) * (3 * x) = 45 * x^2 + 60 * x :=
by
  sorry

end expand_expression_l120_120013


namespace positive_slope_of_asymptote_l120_120477

-- Define the conditions
def is_hyperbola (x y : ℝ) : Prop :=
  abs (Real.sqrt ((x - 1) ^ 2 + (y + 2) ^ 2) - Real.sqrt ((x - 5) ^ 2 + (y + 2) ^ 2)) = 3

-- Prove the positive slope of the asymptote of the given hyperbola
theorem positive_slope_of_asymptote :
  (∀ x y : ℝ, is_hyperbola x y) → abs (Real.sqrt 7 / 3) = Real.sqrt 7 / 3 :=
by
  intros h
  -- Proof to be provided (proof steps from the provided solution would be used here usually)
  sorry

end positive_slope_of_asymptote_l120_120477


namespace trigonometric_identity_l120_120831

open Real

theorem trigonometric_identity :
  let cos_18 := (sqrt 5 + 1) / 4
  let sin_18 := (sqrt 5 - 1) / 4
  4 * cos_18 ^ 2 - 1 = 1 / (4 * sin_18 ^ 2) :=
by
  let cos_18 := (sqrt 5 + 1) / 4
  let sin_18 := (sqrt 5 - 1) / 4
  sorry

end trigonometric_identity_l120_120831


namespace quinn_frogs_caught_l120_120512

-- Defining the conditions
def Alster_frogs : Nat := 2

def Quinn_frogs (Alster_caught: Nat) : Nat := Alster_caught

def Bret_frogs (Quinn_caught: Nat) : Nat := 3 * Quinn_caught

-- Given that Bret caught 12 frogs, prove the amount Quinn caught
theorem quinn_frogs_caught (Bret_caught: Nat) (h1: Bret_caught = 12) : Quinn_frogs Alster_frogs = 4 :=
by
  sorry

end quinn_frogs_caught_l120_120512


namespace multiplication_of_positive_and_negative_l120_120822

theorem multiplication_of_positive_and_negative :
  9 * (-3) = -27 := by
  sorry

end multiplication_of_positive_and_negative_l120_120822


namespace find_a_plus_b_l120_120577

def star (a b : ℕ) : ℕ := a^b + a + b

theorem find_a_plus_b (a b : ℕ) (h2a : 2 ≤ a) (h2b : 2 ≤ b) (h_ab : star a b = 20) :
  a + b = 6 :=
sorry

end find_a_plus_b_l120_120577


namespace find_flat_fee_l120_120007

def flat_fee_exists (f n : ℝ) : Prop :=
  f + n = 120 ∧ f + 4 * n = 255

theorem find_flat_fee : ∃ f n, flat_fee_exists f n ∧ f = 75 := by
  sorry

end find_flat_fee_l120_120007


namespace hyperbola_condition_l120_120281

theorem hyperbola_condition (k : ℝ) : 
  (0 < k ∧ k < 1) ↔ ∀ x y : ℝ, (x^2 / (k - 1)) + (y^2 / (k + 2)) = 1 → 
  (k - 1 < 0 ∧ k + 2 > 0 ∨ k - 1 > 0 ∧ k + 2 < 0) := 
sorry

end hyperbola_condition_l120_120281


namespace blue_paper_side_length_l120_120025

theorem blue_paper_side_length (side_red : ℝ) (side_blue : ℝ) (same_area : side_red^2 = side_blue * x) (side_red_val : side_red = 5) (side_blue_val : side_blue = 4) : x = 6.25 :=
by
  sorry

end blue_paper_side_length_l120_120025


namespace range_f_when_a_1_range_of_a_values_l120_120913

noncomputable def f (x a : ℝ) : ℝ := |x - a| + |x + 4|

theorem range_f_when_a_1 : 
  (∀ x : ℝ, f x 1 ≥ 5) :=
sorry

theorem range_of_a_values :
  (∀ x, f x a ≥ 1) → (a ∈ Set.union (Set.Iic (-5)) (Set.Ici (-3))) :=
sorry

end range_f_when_a_1_range_of_a_values_l120_120913


namespace find_a_for_quadratic_l120_120674

theorem find_a_for_quadratic (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ a^2 * (x - 2) + a * (39 - 20 * x) + 20 = 0 ∧ a^2 * (y - 2) + a * (39 - 20 * y) + 20 = 0) ↔ a = 20 := 
sorry

end find_a_for_quadratic_l120_120674


namespace average_speed_first_part_l120_120499

noncomputable def speed_of_first_part (v : ℝ) : Prop :=
  let distance_first_part := 124
  let speed_second_part := 60
  let distance_second_part := 250 - distance_first_part
  let total_time := 5.2
  (distance_first_part / v) + (distance_second_part / speed_second_part) = total_time

theorem average_speed_first_part : speed_of_first_part 40 :=
  sorry

end average_speed_first_part_l120_120499


namespace horner_eval_v4_at_2_l120_120805

def f (x : ℝ) : ℝ := x^6 - 12*x^5 + 60*x^4 - 160*x^3 + 240*x^2 - 192*x + 64

theorem horner_eval_v4_at_2 : 
  let x := 2
  let v_0 := 1
  let v_1 := (v_0 * x) - 12 
  let v_2 := (v_1 * x) + 60 
  let v_3 := (v_2 * x) - 160 
  let v_4 := (v_3 * x) + 240 
  v_4 = 80 := 
by 
  sorry

end horner_eval_v4_at_2_l120_120805


namespace minimum_area_of_cyclic_quadrilateral_l120_120727

theorem minimum_area_of_cyclic_quadrilateral :
  ∀ (r1 r2 : ℝ), (r1 = 1) ∧ (r2 = 2) →
    ∃ (A : ℝ), A = 3 * Real.sqrt 3 ∧ 
    (∀ (q : ℝ) (circumscribed : q ≤ A),
      ∀ (p : Prop), (p = (∃ x y z w, 
        ∀ (cx : ℝ) (cy : ℝ) (cr : ℝ), 
          cr = r2 ∧ 
          (Real.sqrt ((x - cx)^2 + (y - cy)^2) = r2) ∧ 
          (Real.sqrt ((z - cx)^2 + (w - cy)^2) = r2) ∧ 
          (Real.sqrt ((x - cx)^2 + (w - cy)^2) = r1) ∧ 
          (Real.sqrt ((z - cx)^2 + (y - cy)^2) = r1)
      )) → q = A) :=
sorry

end minimum_area_of_cyclic_quadrilateral_l120_120727


namespace Victor_worked_hours_l120_120134

theorem Victor_worked_hours (h : ℕ) (pay_rate : ℕ) (total_earnings : ℕ) 
  (H1 : pay_rate = 6) 
  (H2 : total_earnings = 60) 
  (H3 : 2 * (pay_rate * h) = total_earnings): 
  h = 5 := 
by 
  sorry

end Victor_worked_hours_l120_120134


namespace intersection_nonempty_implies_a_gt_neg1_l120_120382

def A := {x : ℝ | -1 ≤ x ∧ x < 2}
def B (a : ℝ) := {x : ℝ | x < a}

theorem intersection_nonempty_implies_a_gt_neg1 (a : ℝ) : (A ∩ B a).Nonempty → a > -1 :=
by
  sorry

end intersection_nonempty_implies_a_gt_neg1_l120_120382


namespace smallest_sum_of_squares_l120_120752

theorem smallest_sum_of_squares (x y : ℤ) (h : x^2 - y^2 = 145) : 
  ∃ x y, x^2 - y^2 = 145 ∧ x^2 + y^2 = 433 :=
sorry

end smallest_sum_of_squares_l120_120752


namespace smallest_number_among_10_11_12_l120_120535

theorem smallest_number_among_10_11_12 : min (min 10 11) 12 = 10 :=
by sorry

end smallest_number_among_10_11_12_l120_120535


namespace marthas_bedroom_size_l120_120952

-- Define the variables and conditions
def total_square_footage := 300
def additional_square_footage := 60
def Martha := 120
def Jenny := Martha + additional_square_footage

-- The main theorem stating the requirement 
theorem marthas_bedroom_size : (Martha + (Martha + additional_square_footage) = total_square_footage) -> Martha = 120 :=
by 
  sorry

end marthas_bedroom_size_l120_120952


namespace number_of_bookshelves_l120_120163

def total_space : ℕ := 400
def reserved_space : ℕ := 160
def shelf_space : ℕ := 80

theorem number_of_bookshelves : (total_space - reserved_space) / shelf_space = 3 := by
  sorry

end number_of_bookshelves_l120_120163


namespace line_length_after_erasing_l120_120766

-- Definition of the initial length in meters and the erased length in centimeters
def initial_length_meters : ℝ := 1.5
def erased_length_centimeters : ℝ := 15.25

-- Conversion factor from meters to centimeters
def meters_to_centimeters (m : ℝ) : ℝ := m * 100

-- Definition of the initial length in centimeters
def initial_length_centimeters : ℝ := meters_to_centimeters initial_length_meters

-- Statement of the theorem
theorem line_length_after_erasing :
  initial_length_centimeters - erased_length_centimeters = 134.75 :=
by
  -- The proof would go here
  sorry

end line_length_after_erasing_l120_120766


namespace chandra_monsters_l120_120303

def monsters_day_1 : Nat := 2
def monsters_day_2 : Nat := monsters_day_1 * 3
def monsters_day_3 : Nat := monsters_day_2 * 4
def monsters_day_4 : Nat := monsters_day_3 * 5
def monsters_day_5 : Nat := monsters_day_4 * 6

def total_monsters : Nat := monsters_day_1 + monsters_day_2 + monsters_day_3 + monsters_day_4 + monsters_day_5

theorem chandra_monsters : total_monsters = 872 :=
by
  unfold total_monsters
  unfold monsters_day_1
  unfold monsters_day_2
  unfold monsters_day_3
  unfold monsters_day_4
  unfold monsters_day_5
  sorry

end chandra_monsters_l120_120303


namespace square_side_length_l120_120550

theorem square_side_length (A : ℝ) (h : A = 100) : ∃ s : ℝ, s * s = A ∧ s = 10 := by
  sorry

end square_side_length_l120_120550


namespace varphi_solution_l120_120574

noncomputable def varphi (x : ℝ) (m n : ℝ) : ℝ :=
  m * x + n / x

theorem varphi_solution :
  ∃ (m n : ℝ), (varphi 1 m n = 8) ∧ (varphi 16 m n = 16) ∧ (∀ x, varphi x m n = 3 * x + 5 / x) :=
sorry

end varphi_solution_l120_120574


namespace parabola_equation_l120_120003

theorem parabola_equation (A B : ℝ × ℝ) (x₁ x₂ y₁ y₂ p : ℝ) :
  A = (x₁, y₁) →
  B = (x₂, y₂) →
  x₁ + x₂ = (p + 8) / 2 →
  x₁ * x₂ = 4 →
  (x₁ - x₂)^2 + (y₁ - y₂)^2 = 45 →
  (y₁ = 2 * x₁ - 4) →
  (y₂ = 2 * x₂ - 4) →
  ((y₁^2 = 2 * p * x₁) ∧ (y₂^2 = 2 * p * x₂)) →
  (y₁^2 = 4 * x₁ ∨ y₂^2 = -36 * x₂) := 
by {
  sorry
}

end parabola_equation_l120_120003


namespace total_cups_l120_120286

variable (eggs : ℕ) (flour : ℕ)
variable (h : eggs = 60) (h1 : flour = eggs / 2)

theorem total_cups (eggs : ℕ) (flour : ℕ) (h : eggs = 60) (h1 : flour = eggs / 2) : 
  eggs + flour = 90 := 
by
  sorry

end total_cups_l120_120286


namespace inequality_solution_l120_120792

theorem inequality_solution (x : ℝ) :
  (x+3)/(x+4) > (4*x+5)/(3*x+10) ↔ x ∈ Set.Ioo (-4 : ℝ) (- (10 : ℝ) / 3) ∪ Set.Ioi 2 :=
by
  sorry

end inequality_solution_l120_120792


namespace profit_is_eight_dollars_l120_120739

-- Define the given quantities and costs
def total_bracelets : ℕ := 52
def bracelets_given_away : ℕ := 8
def cost_of_materials : ℝ := 3.00
def selling_price_per_bracelet : ℝ := 0.25

-- Define the number of bracelets sold
def bracelets_sold := total_bracelets - bracelets_given_away

-- Calculate the total money earned from selling the bracelets
def total_earnings := bracelets_sold * selling_price_per_bracelet

-- Calculate the profit made by Alice
def profit := total_earnings - cost_of_materials

-- Prove that the profit is $8.00
theorem profit_is_eight_dollars : profit = 8.00 := by
  sorry

end profit_is_eight_dollars_l120_120739


namespace highest_possible_relocation_preference_l120_120704

theorem highest_possible_relocation_preference
  (total_employees : ℕ)
  (relocated_to_X_percent : ℝ)
  (relocated_to_Y_percent : ℝ)
  (prefer_X_percent : ℝ)
  (prefer_Y_percent : ℝ)
  (htotal : total_employees = 200)
  (hrelocated_to_X_percent : relocated_to_X_percent = 0.30)
  (hrelocated_to_Y_percent : relocated_to_Y_percent = 0.70)
  (hprefer_X_percent : prefer_X_percent = 0.60)
  (hprefer_Y_percent : prefer_Y_percent = 0.40) :
  ∃ (max_relocated_with_preference : ℕ), max_relocated_with_preference = 140 :=
by
  sorry

end highest_possible_relocation_preference_l120_120704


namespace tan_ratio_l120_120254

theorem tan_ratio (a b : ℝ)
  (h1 : Real.cos (a + b) = 1 / 3)
  (h2 : Real.cos (a - b) = 1 / 2) :
  (Real.tan a) / (Real.tan b) = 5 :=
sorry

end tan_ratio_l120_120254


namespace runner_speed_ratio_l120_120823

theorem runner_speed_ratio (d s u v_f v_s : ℝ) (hs : s ≠ 0) (hu : u ≠ 0)
  (H1 : (v_f + v_s) * s = d) (H2 : (v_f - v_s) * u = v_s * u) :
  v_f / v_s = 2 :=
by
  sorry

end runner_speed_ratio_l120_120823


namespace g_odd_find_a_f_increasing_l120_120186

-- Problem (I): Prove that if g(x) = f(x) - a is an odd function, then a = 1, given f(x) = 1 - 2/x.
theorem g_odd_find_a (f : ℝ → ℝ) (g : ℝ → ℝ) (a : ℝ) : 
  (∀ x, f x = 1 - (2 / x)) → 
  (∀ x, g x = f x - a) → 
  (∀ x, g (-x) = - g x) → 
  a = 1 := 
  by
  intros h1 h2 h3
  sorry

-- Problem (II): Prove that f(x) is monotonically increasing on (0, +∞),
-- given f(x) = 1 - 2/x.

theorem f_increasing (f : ℝ → ℝ) : 
  (∀ x, f x = 1 - (2 / x)) → 
  ∀ (x1 x2 : ℝ), 0 < x1 → x1 < x2 → f x1 < f x2 := 
  by
  intros h1 x1 x2 hx1 hx12
  sorry

end g_odd_find_a_f_increasing_l120_120186


namespace range_of_a_l120_120346

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - x^2 + x + 2
noncomputable def g (x : ℝ) : ℝ := (Real.exp 1 * Real.log x) / x
noncomputable def h (x : ℝ) : ℝ := (x^2 - x - 2) / x^3

theorem range_of_a (a : ℝ) :
  (∀ x1 x2, 0 < x1 ∧ x1 ≤ 1 ∧ 0 < x2 ∧ x2 ≤ 1 → f a x1 ≥ g x2) ↔ a ≥ -2 :=
by
  sorry

end range_of_a_l120_120346


namespace number_of_possible_lists_l120_120320

theorem number_of_possible_lists : 
  let num_balls := 15
  let num_draws := 4
  (num_balls ^ num_draws) = 50625 := by
  sorry

end number_of_possible_lists_l120_120320


namespace field_width_calculation_l120_120883

theorem field_width_calculation (w : ℝ) (h_length : length = 24) (h_length_width_relation : length = 2 * w - 3) : w = 13.5 :=
by 
  sorry

end field_width_calculation_l120_120883


namespace scarves_per_box_l120_120703

theorem scarves_per_box (S : ℕ) 
  (boxes : ℕ := 8) 
  (mittens_per_box : ℕ := 6) 
  (total_clothing : ℕ := 80) 
  (total_mittens : ℕ := boxes * mittens_per_box) 
  (total_scarves : ℕ := total_clothing - total_mittens) 
  (scarves_per_box : ℕ := total_scarves / boxes) 
  : scarves_per_box = 4 := 
by 
  sorry

end scarves_per_box_l120_120703


namespace problem_statement_l120_120934

noncomputable def angle_between_vectors (a b : EuclideanSpace ℝ (Fin 3)) : ℝ :=
Real.arccos (inner a b / (‖a‖ * ‖b‖))

theorem problem_statement
  (a b : EuclideanSpace ℝ (Fin 3))
  (h_angle_ab : angle_between_vectors a b = Real.pi / 3)
  (h_norm_a : ‖a‖ = 2)
  (h_norm_b : ‖b‖ = 1) :
  angle_between_vectors a (a + 2 • b) = Real.pi / 6 :=
sorry

end problem_statement_l120_120934


namespace triplet_solution_l120_120814

theorem triplet_solution (a b c : ℕ) (h1 : a^2 + b^2 + c^2 = 2005) (h2 : a ≤ b) (h3 : b ≤ c) :
  (a = 24 ∧ b = 30 ∧ c = 23) ∨ 
  (a = 12 ∧ b = 30 ∧ c = 31) ∨
  (a = 18 ∧ b = 40 ∧ c = 9) ∨
  (a = 15 ∧ b = 22 ∧ c = 36) ∨
  (a = 12 ∧ b = 30 ∧ c = 31) :=
sorry

end triplet_solution_l120_120814


namespace compute_expression_l120_120253

theorem compute_expression :
    ( (2 / 3) * Real.sqrt 15 - Real.sqrt 20 ) / ( (1 / 3) * Real.sqrt 5 ) = 2 * Real.sqrt 3 - 6 :=
by
  sorry

end compute_expression_l120_120253


namespace rectangle_area_decrease_l120_120921

noncomputable def rectangle_area_change (L B : ℝ) (hL : L > 0) (hB : B > 0) : ℝ :=
  let L' := 1.10 * L
  let B' := 0.90 * B
  let A  := L * B
  let A' := L' * B'
  A'

theorem rectangle_area_decrease (L B : ℝ) (hL : L > 0) (hB : B > 0) :
  rectangle_area_change L B hL hB = 0.99 * (L * B) := by
  sorry

end rectangle_area_decrease_l120_120921


namespace import_tax_l120_120181

theorem import_tax (total_value : ℝ) (tax_rate : ℝ) (excess_limit : ℝ) (correct_tax : ℝ)
  (h1 : total_value = 2560) (h2 : tax_rate = 0.07) (h3 : excess_limit = 1000) : 
  correct_tax = tax_rate * (total_value - excess_limit) :=
by
  sorry

end import_tax_l120_120181


namespace problem1_problem2_l120_120415

theorem problem1 (x : ℝ) : (x + 4) ^ 2 - 5 * (x + 4) = 0 → x = -4 ∨ x = 1 :=
by
  sorry

theorem problem2 (x : ℝ) : x ^ 2 - 2 * x - 15 = 0 → x = -3 ∨ x = 5 :=
by
  sorry

end problem1_problem2_l120_120415


namespace women_more_than_men_l120_120176

theorem women_more_than_men 
(M W : ℕ) 
(h_ratio : (M:ℚ) / W = 5 / 9) 
(h_total : M + W = 14) :
W - M = 4 := 
by 
  sorry

end women_more_than_men_l120_120176


namespace inlet_pipe_filling_rate_l120_120670

def leak_rate (volume : ℕ) (time_hours : ℕ) : ℕ :=
  volume / time_hours

def net_emptying_rate (volume : ℕ) (time_hours : ℕ) : ℕ :=
  volume / time_hours

def inlet_rate_per_hour (net_rate : ℕ) (leak_rate : ℕ) : ℕ :=
  leak_rate - net_rate

def convert_to_minutes (rate_per_hour : ℕ) : ℕ :=
  rate_per_hour / 60

theorem inlet_pipe_filling_rate :
  let volume := 4320
  let time_to_empty_with_leak := 6
  let net_time_to_empty := 12
  let leak_rate := leak_rate volume time_to_empty_with_leak
  let net_rate := net_emptying_rate volume net_time_to_empty
  let fill_rate_per_hour := inlet_rate_per_hour net_rate leak_rate
  convert_to_minutes fill_rate_per_hour = 6 := by
    -- Proof ends with a placeholder 'sorry'
    sorry

end inlet_pipe_filling_rate_l120_120670


namespace calc_value_l120_120115

theorem calc_value (a b : ℝ) (h : b = 3 * a - 2) : 2 * b - 6 * a + 2 = -2 := 
by 
  sorry

end calc_value_l120_120115


namespace brenda_age_l120_120174

-- Define ages of Addison, Brenda, Carlos, and Janet
variables (A B C J : ℕ)

-- Formalize the conditions from the problem
def condition1 := A = 4 * B
def condition2 := C = 2 * B
def condition3 := A = J

-- State the theorem we aim to prove
theorem brenda_age (A B C J : ℕ) (h1 : condition1 A B)
                                (h2 : condition2 C B)
                                (h3 : condition3 A J) :
  B = J / 4 :=
sorry

end brenda_age_l120_120174


namespace total_wheels_in_parking_lot_l120_120661

theorem total_wheels_in_parking_lot :
  let cars := 5
  let trucks := 3
  let bikes := 2
  let three_wheelers := 4
  let wheels_per_car := 4
  let wheels_per_truck := 6
  let wheels_per_bike := 2
  let wheels_per_three_wheeler := 3
  (cars * wheels_per_car + trucks * wheels_per_truck + bikes * wheels_per_bike + three_wheelers * wheels_per_three_wheeler) = 54 := by
  sorry

end total_wheels_in_parking_lot_l120_120661


namespace max_page_number_with_given_fives_l120_120121

theorem max_page_number_with_given_fives (plenty_digit_except_five : ℕ → ℕ) 
  (H0 : ∀ d ≠ 5, ∀ n, plenty_digit_except_five d = n)
  (H5 : plenty_digit_except_five 5 = 30) : ∃ (n : ℕ), n = 154 :=
by {
  sorry
}

end max_page_number_with_given_fives_l120_120121


namespace side_length_uncovered_l120_120444

theorem side_length_uncovered (L W : ℝ) (h₁ : L * W = 50) (h₂ : 2 * W + L = 25) : L = 20 :=
by {
  sorry
}

end side_length_uncovered_l120_120444


namespace ratio_fenced_region_l120_120420

theorem ratio_fenced_region (L W : ℝ) (k : ℝ) 
  (area_eq : L * W = 200)
  (fence_eq : 2 * W + L = 40)
  (mult_eq : L = k * W) :
  k = 2 :=
by
  sorry

end ratio_fenced_region_l120_120420


namespace shadow_stretch_rate_is_5_feet_per_hour_l120_120192

-- Given conditions
def shadow_length_in_inches (hours_past_noon : ℕ) : ℕ := 360
def hours_past_noon : ℕ := 6

-- Convert inches to feet
def inches_to_feet (inches : ℕ) : ℕ := inches / 12

-- Calculate rate of increase of shadow length per hour
def rate_of_shadow_stretch_per_hour : ℕ := inches_to_feet (shadow_length_in_inches hours_past_noon) / hours_past_noon

theorem shadow_stretch_rate_is_5_feet_per_hour :
  rate_of_shadow_stretch_per_hour = 5 := by
  sorry

end shadow_stretch_rate_is_5_feet_per_hour_l120_120192


namespace correct_statements_B_and_C_l120_120030

variable {a b c : ℝ}

-- Definitions from the conditions
def conditionB (a b c : ℝ) : Prop := a > b ∧ b > 0 ∧ c < 0
def conclusionB (a b c : ℝ) : Prop := c / a^2 > c / b^2

def conditionC (a b c : ℝ) : Prop := c > a ∧ a > b ∧ b > 0
def conclusionC (a b c : ℝ) : Prop := a / (c - a) > b / (c - b)

theorem correct_statements_B_and_C (a b c : ℝ) : 
  (conditionB a b c → conclusionB a b c) ∧ 
  (conditionC a b c → conclusionC a b c) :=
by
  sorry

end correct_statements_B_and_C_l120_120030


namespace fraction_simplification_l120_120976

-- Definitions based on conditions and question
def lcm_462_42 : ℕ := 462
def prime_factors_462 : List ℕ := [2, 3, 7, 11]
def prime_factors_42 : List ℕ := [2, 3, 7]

-- Main theorem statement
theorem fraction_simplification :
  (1 / 462) + (17 / 42) = 94 / 231 :=
by
  sorry

end fraction_simplification_l120_120976


namespace teapot_volume_proof_l120_120197

noncomputable def arithmetic_sequence (a d : ℝ) (n : ℕ) : ℝ :=
  a + (n - 1) * d

theorem teapot_volume_proof (a d : ℝ)
  (h1 : arithmetic_sequence a d 1 + arithmetic_sequence a d 2 + arithmetic_sequence a d 3 = 0.5)
  (h2 : arithmetic_sequence a d 7 + arithmetic_sequence a d 8 + arithmetic_sequence a d 9 = 2.5) :
  arithmetic_sequence a d 5 = 0.5 :=
by {
  sorry
}

end teapot_volume_proof_l120_120197


namespace perimeter_of_large_rectangle_l120_120930

-- We are bringing in all necessary mathematical libraries, no specific submodules needed.
theorem perimeter_of_large_rectangle
  (small_rectangle_longest_side : ℝ)
  (number_of_small_rectangles : ℕ)
  (length_of_large_rectangle : ℝ)
  (height_of_large_rectangle : ℝ)
  (perimeter_of_large_rectangle : ℝ) :
  small_rectangle_longest_side = 10 ∧ number_of_small_rectangles = 9 →
  length_of_large_rectangle = 2 * small_rectangle_longest_side →
  height_of_large_rectangle = 5 * (small_rectangle_longest_side / 2) →
  perimeter_of_large_rectangle = 2 * (length_of_large_rectangle + height_of_large_rectangle) →
  perimeter_of_large_rectangle = 76 := by
  sorry

end perimeter_of_large_rectangle_l120_120930


namespace mixed_alcohol_solution_l120_120632

theorem mixed_alcohol_solution 
    (vol_x : ℝ) (vol_y : ℝ) (conc_x : ℝ) (conc_y : ℝ) (target_conc : ℝ) (vol_y_given : vol_y = 750) 
    (conc_x_given : conc_x = 0.10) (conc_y_given : conc_y = 0.30) (target_conc_given : target_conc = 0.25) : 
    vol_x = 250 → 
    (conc_x * vol_x + conc_y * vol_y) / (vol_x + vol_y) = target_conc :=
by
  intros h_x
  rw [vol_y_given, conc_x_given, conc_y_given, target_conc_given, h_x]
  sorry

end mixed_alcohol_solution_l120_120632


namespace rational_cos_terms_l120_120820

open Real

noncomputable def rational_sum (x : ℝ) (rS : ℚ) (rC : ℚ) :=
  let S := sin (64 * x) + sin (65 * x)
  let C := cos (64 * x) + cos (65 * x)
  S = rS ∧ C = rC

theorem rational_cos_terms (x : ℝ) (rS : ℚ) (rC : ℚ) :
  rational_sum x rS rC → (∃ q1 q2 : ℚ, cos (64 * x) = q1 ∧ cos (65 * x) = q2) :=
sorry

end rational_cos_terms_l120_120820


namespace exists_3x3_grid_l120_120917

theorem exists_3x3_grid : 
  ∃ (a₁₂ a₂₁ a₂₃ a₃₂ : ℕ), 
  a₁₂ ≠ a₂₁ ∧ a₁₂ ≠ a₂₃ ∧ a₁₂ ≠ a₃₂ ∧ 
  a₂₁ ≠ a₂₃ ∧ a₂₁ ≠ a₃₂ ∧ 
  a₂₃ ≠ a₃₂ ∧ 
  a₁₂ ≤ 25 ∧ a₂₁ ≤ 25 ∧ a₂₃ ≤ 25 ∧ a₃₂ ≤ 25 ∧ 
  a₁₂ > 0 ∧ a₂₁ > 0 ∧ a₂₃ > 0 ∧ a₃₂ > 0 ∧
  (∃ (a₁₁ a₁₃ a₃₁ a₃₃ a₂₂ : ℕ),
  a₁₁ ≤ 25 ∧ a₁₃ ≤ 25 ∧ a₃₁ ≤ 25 ∧ a₃₃ ≤ 25 ∧ a₂₂ ≤ 25 ∧
  a₁₁ > 0 ∧ a₁₃ > 0 ∧ a₃₁ > 0 ∧ a₃₃ > 0 ∧ a₂₂ > 0 ∧
  a₁₁ ≠ a₁₂ ∧ a₁₁ ≠ a₂₁ ∧ a₁₁ ≠ a₁₃ ∧ a₁₁ ≠ a₃₁ ∧ 
  a₁₃ ≠ a₃₃ ∧ a₁₃ ≠ a₂₃ ∧ a₂₁ ≠ a₃₁ ∧ a₃₁ ≠ a₃₂ ∧ 
  a₃₃ ≠ a₂₂ ∧ a₃₃ ≠ a₃₂ ∧ a₂₂ = 1 ∧
  (a₁₂ % a₂₂ = 0 ∨ a₂₂ % a₁₂ = 0) ∧
  (a₂₁ % a₂₂ = 0 ∨ a₂₂ % a₂₁ = 0) ∧
  (a₂₃ % a₂₂ = 0 ∨ a₂₂ % a₂₃ = 0) ∧
  (a₃₂ % a₂₂ = 0 ∨ a₂₂ % a₃₂ = 0) ∧
  (a₁₁ % a₁₂ = 0 ∨ a₁₂ % a₁₁ = 0) ∧
  (a₁₁ % a₂₁ = 0 ∨ a₂₁ % a₁₁ = 0) ∧
  (a₁₃ % a₁₂ = 0 ∨ a₁₂ % a₁₃ = 0) ∧
  (a₁₃ % a₂₃ = 0 ∨ a₂₃ % a₁₃ = 0) ∧
  (a₃₁ % a₂₁ = 0 ∨ a₂₁ % a₃₁ = 0) ∧
  (a₃₁ % a₃₂ = 0 ∨ a₃₂ % a₃₁ = 0) ∧
  (a₃₃ % a₂₃ = 0 ∨ a₂₃ % a₃₃ = 0) ∧
  (a₃₃ % a₃₂ = 0 ∨ a₃₂ % a₃₃ = 0)) 
  :=
sorry

end exists_3x3_grid_l120_120917


namespace number_of_teachers_l120_120643

theorem number_of_teachers
  (students : ℕ) (lessons_per_student_per_day : ℕ) (lessons_per_teacher_per_day : ℕ) (students_per_class : ℕ)
  (h1 : students = 1200)
  (h2 : lessons_per_student_per_day = 5)
  (h3 : lessons_per_teacher_per_day = 4)
  (h4 : students_per_class = 30) :
  ∃ teachers : ℕ, teachers = 50 :=
by
  have total_lessons : ℕ := lessons_per_student_per_day * students
  have classes : ℕ := total_lessons / students_per_class
  have teachers : ℕ := classes / lessons_per_teacher_per_day
  use teachers
  sorry

end number_of_teachers_l120_120643


namespace percentage_increase_in_llama_cost_l120_120268

def cost_of_goat : ℕ := 400
def number_of_goats : ℕ := 3
def total_cost : ℕ := 4800

def llamas_cost (x : ℕ) : Prop :=
  let total_cost_goats := number_of_goats * cost_of_goat
  let total_cost_llamas := total_cost - total_cost_goats
  let number_of_llamas := 2 * number_of_goats
  let cost_per_llama := total_cost_llamas / number_of_llamas
  let increase := cost_per_llama - cost_of_goat
  ((increase / cost_of_goat) * 100) = x

theorem percentage_increase_in_llama_cost :
  llamas_cost 50 :=
sorry

end percentage_increase_in_llama_cost_l120_120268


namespace find_q_l120_120595

theorem find_q (p q : ℝ) (h : ∀ x : ℝ, (x^2 + p * x + q) ≥ 1) : q = 1 + (p^2 / 4) :=
sorry

end find_q_l120_120595


namespace box_area_ratio_l120_120903

theorem box_area_ratio 
  (l w h : ℝ)
  (V : l * w * h = 5184)
  (A1 : w * h = (1/2) * l * w)
  (A2 : l * h = 288):
  (l * w) / (l * h) = 3 / 2 := 
by
  sorry

end box_area_ratio_l120_120903


namespace first_statement_second_statement_difference_between_statements_l120_120288

variable (A B C : Prop)

-- First statement: (A ∨ B) → C
theorem first_statement : (A ∨ B) → C :=
sorry

-- Second statement: (A ∧ B) → C
theorem second_statement : (A ∧ B) → C :=
sorry

-- Proof that shows the difference between the two statements
theorem difference_between_statements :
  ((A ∨ B) → C) ↔ ¬((A ∧ B) → C) :=
sorry

end first_statement_second_statement_difference_between_statements_l120_120288


namespace n_divisible_by_40_l120_120698

theorem n_divisible_by_40 {n : ℕ} (h_pos : 0 < n)
  (h1 : ∃ k1 : ℕ, 2 * n + 1 = k1 * k1)
  (h2 : ∃ k2 : ℕ, 3 * n + 1 = k2 * k2) :
  ∃ k : ℕ, n = 40 * k := 
sorry

end n_divisible_by_40_l120_120698


namespace same_color_combination_probability_l120_120682

-- Defining the number of each color candy 
def num_red : Nat := 12
def num_blue : Nat := 12
def num_green : Nat := 6

-- Terry and Mary each pick 3 candies at random
def total_pick : Nat := 3

-- The total number of candies in the jar
def total_candies : Nat := num_red + num_blue + num_green

-- Probability of Terry and Mary picking the same color combination
def probability_same_combination : ℚ := 2783 / 847525

-- The theorem statement
theorem same_color_combination_probability :
  let terry_picks_red := (num_red * (num_red - 1) * (num_red - 2)) / (total_candies * (total_candies - 1) * (total_candies - 2))
  let remaining_red := num_red - total_pick
  let mary_picks_red := (remaining_red * (remaining_red - 1) * (remaining_red - 2)) / (27 * 26 * 25)
  let combined_red := terry_picks_red * mary_picks_red

  let terry_picks_blue := (num_blue * (num_blue - 1) * (num_blue - 2)) / (total_candies * (total_candies - 1) * (total_candies - 2))
  let remaining_blue := num_blue - total_pick
  let mary_picks_blue := (remaining_blue * (remaining_blue - 1) * (remaining_blue - 2)) / (27 * 26 * 25)
  let combined_blue := terry_picks_blue * mary_picks_blue

  let terry_picks_green := (num_green * (num_green - 1) * (num_green - 2)) / (total_candies * (total_candies - 1) * (total_candies - 2))
  let remaining_green := num_green - total_pick
  let mary_picks_green := (remaining_green * (remaining_green - 1) * (remaining_green - 2)) / (27 * 26 * 25)
  let combined_green := terry_picks_green * mary_picks_green

  let total_probability := 2 * combined_red + 2 * combined_blue + combined_green
  total_probability = probability_same_combination := sorry

end same_color_combination_probability_l120_120682


namespace average_age_with_teacher_l120_120195

theorem average_age_with_teacher (A : ℕ) (h : 21 * 16 = 20 * A + 36) : A = 15 := by
  sorry

end average_age_with_teacher_l120_120195


namespace pure_acid_total_is_3_8_l120_120144

/-- Volume of Solution A in liters -/
def volume_A : ℝ := 8

/-- Concentration of Solution A (in decimals, i.e., 20% as 0.20) -/
def concentration_A : ℝ := 0.20

/-- Volume of Solution B in liters -/
def volume_B : ℝ := 5

/-- Concentration of Solution B (in decimals, i.e., 35% as 0.35) -/
def concentration_B : ℝ := 0.35

/-- Volume of Solution C in liters -/
def volume_C : ℝ := 3

/-- Concentration of Solution C (in decimals, i.e., 15% as 0.15) -/
def concentration_C : ℝ := 0.15

/-- Total amount of pure acid in the resulting mixture -/
def total_pure_acid : ℝ :=
  (volume_A * concentration_A) +
  (volume_B * concentration_B) +
  (volume_C * concentration_C)

theorem pure_acid_total_is_3_8 : total_pure_acid = 3.8 := by
  sorry

end pure_acid_total_is_3_8_l120_120144


namespace seeds_germinated_percentage_l120_120746

theorem seeds_germinated_percentage (n1 n2 : ℕ) (p1 p2 : ℝ) (h1 : n1 = 300) (h2 : n2 = 200) (h3 : p1 = 0.25) (h4 : p2 = 0.30) :
  ( (n1 * p1 + n2 * p2) / (n1 + n2) ) * 100 = 27 :=
by
  sorry

end seeds_germinated_percentage_l120_120746


namespace last_score_is_65_l120_120389

-- Define the scores and the problem conditions
def scores := [65, 72, 75, 80, 85, 88, 92]
def total_sum := 557
def remaining_sum (score : ℕ) : ℕ := total_sum - score

-- Define a property to check divisibility
def divisible_by (n d : ℕ) : Prop := n % d = 0

-- The main theorem statement
theorem last_score_is_65 :
  (∀ s ∈ scores, divisible_by (remaining_sum s) 6) ∧ divisible_by total_sum 7 ↔ scores = [65, 72, 75, 80, 85, 88, 92] :=
sorry

end last_score_is_65_l120_120389


namespace initial_tax_rate_l120_120249

theorem initial_tax_rate 
  (income : ℝ)
  (differential_savings : ℝ)
  (final_tax_rate : ℝ)
  (initial_tax_rate : ℝ) 
  (h1 : income = 42400) 
  (h2 : differential_savings = 4240) 
  (h3 : final_tax_rate = 32)
  (h4 : differential_savings = (initial_tax_rate / 100) * income - (final_tax_rate / 100) * income) :
  initial_tax_rate = 42 :=
sorry

end initial_tax_rate_l120_120249


namespace range_of_a_l120_120815

noncomputable def quadratic (a x : ℝ) : ℝ := a * x^2 + a * x + 1
noncomputable def quadratic_eq (x₀ a : ℝ) : Prop := x₀^2 - x₀ + a = 0

theorem range_of_a (a : ℝ) (p : ∀ x : ℝ, quadratic a x > 0) (q : ∃ x₀ : ℝ, quadratic_eq x₀ a) : 0 ≤ a ∧ a ≤ 1/4 :=
  sorry

end range_of_a_l120_120815


namespace inequality_holds_for_all_x_l120_120774

theorem inequality_holds_for_all_x (x : ℝ) : 3 * x^2 + 9 * x ≥ -12 := by
  sorry

end inequality_holds_for_all_x_l120_120774


namespace frank_sales_quota_l120_120603

theorem frank_sales_quota (x : ℕ) :
  (3 * x + 12 + 23 = 50) → x = 5 :=
by sorry

end frank_sales_quota_l120_120603


namespace minimum_value_expression_l120_120729

theorem minimum_value_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : 
  4 * x^4 + 16 * y^4 + 36 * z^4 + 9 / (x * y * z) ≥ 24 :=
by
  sorry

end minimum_value_expression_l120_120729


namespace probability_of_purple_probability_of_blue_or_purple_l120_120152

def total_jelly_beans : ℕ := 60
def purple_jelly_beans : ℕ := 5
def blue_jelly_beans : ℕ := 18

theorem probability_of_purple :
  (purple_jelly_beans : ℚ) / total_jelly_beans = 1 / 12 :=
by
  sorry
  
theorem probability_of_blue_or_purple :
  (blue_jelly_beans + purple_jelly_beans : ℚ) / total_jelly_beans = 23 / 60 :=
by
  sorry

end probability_of_purple_probability_of_blue_or_purple_l120_120152


namespace minimum_value_of_f_roots_sum_gt_2_l120_120201

noncomputable def f (x : ℝ) : ℝ := Real.log x + 1 / x

theorem minimum_value_of_f : ∃ x : ℝ, (∀ y : ℝ, f y ≥ f x) ∧ f 1 = 1 := by
  exists 1
  sorry

theorem roots_sum_gt_2 (a x₁ x₂ : ℝ) (h_f_x₁ : f x₁ = a) (h_f_x₂ : f x₂ = a) (h_x₁_lt_x₂ : x₁ < x₂) :
    x₁ + x₂ > 2 := by
  sorry

end minimum_value_of_f_roots_sum_gt_2_l120_120201


namespace largest_angle_of_triangle_l120_120187

theorem largest_angle_of_triangle (x : ℝ) 
  (h1 : 4 * x + 5 * x + 9 * x = 180) 
  (h2 : 4 * x > 40) : 
  9 * x = 90 := 
sorry

end largest_angle_of_triangle_l120_120187


namespace find_fourth_number_l120_120979

variable (x : ℝ)

theorem find_fourth_number
  (h : 3 + 33 + 333 + x = 399.6) :
  x = 30.6 :=
sorry

end find_fourth_number_l120_120979


namespace parabola_focus_directrix_l120_120571

noncomputable def parabola_distance_property (p : ℝ) (hp : 0 < p) : Prop :=
  let focus := (2 * p, 0)
  let directrix := -2 * p
  let distance := 4 * p
  p = distance / 4

-- Theorem: Given a parabola with equation y^2 = 8px (p > 0), p represents 1/4 of the distance from the focus to the directrix.
theorem parabola_focus_directrix (p : ℝ) (hp : 0 < p) : parabola_distance_property p hp :=
by
  sorry

end parabola_focus_directrix_l120_120571


namespace sheena_weeks_to_complete_l120_120198

/- Definitions -/
def time_per_dress : ℕ := 12
def number_of_dresses : ℕ := 5
def weekly_sewing_time : ℕ := 4

/- Theorem -/
theorem sheena_weeks_to_complete : (number_of_dresses * time_per_dress) / weekly_sewing_time = 15 := 
by 
  /- Proof is omitted -/
  sorry

end sheena_weeks_to_complete_l120_120198


namespace exactly_one_even_l120_120694

theorem exactly_one_even (a b c : ℕ) : 
  (∀ x, ¬ (a = x ∧ b = x ∧ c = x) ∧ 
  (a % 2 = 0 ∨ b % 2 = 0 ∨ c % 2 = 0) ∧ 
  ¬ (a % 2 = 0 ∧ b % 2 = 0) ∧ 
  ¬ (a % 2 = 0 ∧ c % 2 = 0) ∧ 
  ¬ (b % 2 = 0 ∧ c % 2 = 0)) :=
by
  sorry

end exactly_one_even_l120_120694


namespace smallest_positive_integer_solution_l120_120459

theorem smallest_positive_integer_solution :
  ∃ x : ℕ, 0 < x ∧ 5 * x ≡ 17 [MOD 34] ∧ (∀ y : ℕ, 0 < y ∧ 5 * y ≡ 17 [MOD 34] → x ≤ y) :=
sorry

end smallest_positive_integer_solution_l120_120459


namespace internal_diagonal_cubes_l120_120041

theorem internal_diagonal_cubes :
  let A := (120, 360, 400)
  let gcd_xy := gcd 120 360
  let gcd_yz := gcd 360 400
  let gcd_zx := gcd 400 120
  let gcd_xyz := gcd (gcd 120 360) 400
  let new_cubes := 120 + 360 + 400 - (gcd_xy + gcd_yz + gcd_zx) + gcd_xyz
  new_cubes = 720 :=
by
  -- Definitions
  let A := (120, 360, 400)
  let gcd_xy := Int.gcd 120 360
  let gcd_yz := Int.gcd 360 400
  let gcd_zx := Int.gcd 400 120
  let gcd_xyz := Int.gcd (Int.gcd 120 360) 400
  let new_cubes := 120 + 360 + 400 - (gcd_xy + gcd_yz + gcd_zx) + gcd_xyz

  -- Assertion
  exact Eq.refl new_cubes

end internal_diagonal_cubes_l120_120041


namespace largest_four_digit_number_l120_120904

theorem largest_four_digit_number :
  ∃ a b c d : ℕ, 
    9 < 1000 * a + 100 * b + 10 * c + d ∧ 
    1000 * a + 100 * b + 10 * c + d < 10000 ∧ 
    c = a + b ∧ 
    d = b + c ∧ 
    1000 * a + 100 * b + 10 * c + d = 9099 :=
by {
  sorry
}

end largest_four_digit_number_l120_120904


namespace altitude_eq_4r_l120_120802

variable (A B C D : Type) [DecidableEq A] [DecidableEq B] [DecidableEq C] [DecidableEq D]

-- We define the geometrical relations and constraints
def AC_eq_BC (A B C D : Type) [DecidableEq A] [DecidableEq B] [DecidableEq C] [DecidableEq D] (AC BC : ℝ) : Prop :=
AC = BC

def in_circle_radius_eq_r (A B C D : Type) [DecidableEq A] [DecidableEq B] [DecidableEq C] [DecidableEq D] (incircle_radius r : ℝ) : Prop :=
incircle_radius = r

def ex_circle_radius_eq_r (A B C D : Type) [DecidableEq A] [DecidableEq B] [DecidableEq C] [DecidableEq D] (excircle_radius r : ℝ) : Prop :=
excircle_radius = r

-- Main theorem to prove
theorem altitude_eq_4r 
  (A B C D : Type) [DecidableEq A] [DecidableEq B] [DecidableEq C] [DecidableEq D]
  (AC BC : ℝ) (r : ℝ)
  (h : ℝ)
  (H1 : AC_eq_BC A B C D AC BC)
  (H2 : in_circle_radius_eq_r A B C D r r)
  (H3 : ex_circle_radius_eq_r A B C D r r) :
  h = 4 * r :=
  sorry

end altitude_eq_4r_l120_120802


namespace kira_breakfast_time_l120_120762

theorem kira_breakfast_time :
  let fry_time_per_sausage := 5 -- minutes per sausage
  let scramble_time_per_egg := 4 -- minutes per egg
  let sausages := 3
  let eggs := 6
  let time_to_fry := sausages * fry_time_per_sausage
  let time_to_scramble := eggs * scramble_time_per_egg
  (time_to_fry + time_to_scramble) = 39 := 
by
  sorry

end kira_breakfast_time_l120_120762


namespace binom_60_3_eq_34220_l120_120808

theorem binom_60_3_eq_34220 : Nat.choose 60 3 = 34220 := by
  sorry

end binom_60_3_eq_34220_l120_120808


namespace false_proposition_is_C_l120_120339

theorem false_proposition_is_C : ¬ (∀ x : ℝ, x^3 > 0) :=
sorry

end false_proposition_is_C_l120_120339


namespace nesbitt_inequality_nesbitt_inequality_eq_l120_120859

variable {a b c : ℝ}

theorem nesbitt_inequality (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≥ (3 / 2) :=
sorry

theorem nesbitt_inequality_eq (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  ((a / (b + c)) + (b / (a + c)) + (c / (a + b)) = (3 / 2)) ↔ (a = b ∧ b = c) :=
sorry

end nesbitt_inequality_nesbitt_inequality_eq_l120_120859


namespace expansion_a0_value_l120_120483

theorem expansion_a0_value :
  ∃ (a_0 a_1 a_2 a_3 a_4 a_5 : ℝ), (∀ x : ℝ, (x+1)^5 = a_0 + a_1*(x-1) + a_2*(x-1)^2 + a_3*(x-1)^3 + a_4*(x-1)^4 + a_5*(x-1)^5) ∧ a_0 = 32 :=
  sorry

end expansion_a0_value_l120_120483


namespace dante_coconuts_l120_120786

theorem dante_coconuts (P : ℕ) (D : ℕ) (S : ℕ) (hP : P = 14) (hD : D = 3 * P) (hS : S = 10) :
  (D - S) = 32 :=
by
  sorry

end dante_coconuts_l120_120786


namespace poly_has_int_solution_iff_l120_120299

theorem poly_has_int_solution_iff (a : ℤ) : 
  (a > 0 ∧ (∃ x : ℤ, a * x^2 + 2 * (2 * a - 1) * x + 4 * a - 7 = 0)) ↔ (a = 1 ∨ a = 5) :=
by {
  sorry
}

end poly_has_int_solution_iff_l120_120299


namespace find_apartment_number_l120_120061

open Nat

def is_apartment_number (x a b : ℕ) : Prop :=
  x = 10 * a + b ∧ x = 17 * b

theorem find_apartment_number : ∃ x a b : ℕ, is_apartment_number x a b ∧ x = 85 :=
by
  sorry

end find_apartment_number_l120_120061


namespace min_value_of_y_l120_120590

variable {x k : ℝ}

theorem min_value_of_y (h₁ : ∀ x > 0, 0 < k) 
  (h₂ : ∀ x > 0, (x^2 + k / x) ≥ 3) : k = 2 :=
sorry

end min_value_of_y_l120_120590


namespace snowfall_on_friday_l120_120279

def snowstorm (snow_wednesday snow_thursday total_snow : ℝ) : ℝ :=
  total_snow - (snow_wednesday + snow_thursday)

theorem snowfall_on_friday :
  snowstorm 0.33 0.33 0.89 = 0.23 := 
by
  -- (Conditions)
  -- snow_wednesday = 0.33
  -- snow_thursday = 0.33
  -- total_snow = 0.89
  -- (Conclusion) snowstorm 0.33 0.33 0.89 = 0.23
  sorry

end snowfall_on_friday_l120_120279


namespace wall_length_l120_120759

theorem wall_length (s : ℕ) (d : ℕ) (w : ℕ) (L : ℝ) 
  (hs : s = 18) 
  (hd : d = 20) 
  (hw : w = 32)
  (hcombined : (s ^ 2 + Real.pi * ((d / 2) ^ 2)) = (1 / 2) * (w * L)) :
  L = 39.88 := 
sorry

end wall_length_l120_120759


namespace white_tiles_count_l120_120975

-- Definitions from conditions
def total_tiles : ℕ := 20
def yellow_tiles : ℕ := 3
def blue_tiles : ℕ := yellow_tiles + 1
def purple_tiles : ℕ := 6

-- We need to prove that number of white tiles is 7
theorem white_tiles_count : total_tiles - (yellow_tiles + blue_tiles + purple_tiles) = 7 := by
  -- Placeholder for the actual proof
  sorry

end white_tiles_count_l120_120975


namespace equal_distances_l120_120083

theorem equal_distances (c : ℝ) (distance : ℝ) :
  abs (2 - -4) = distance ∧ (abs (c - -4) = distance ∨ abs (c - 2) = distance) ↔ (c = -10 ∨ c = 8) :=
by
  sorry

end equal_distances_l120_120083


namespace work_completion_l120_120203

theorem work_completion (W : ℕ) (n : ℕ) (h1 : 0 < n) (H1 : 0 < W) :
  (∀ w : ℕ, w ≤ W / n) → 
  (∀ k : ℕ, k = (7 * n) / 10 → k * (3 * W) / (10 * n) ≥ W / 3) → 
  (∀ m : ℕ, m = (3 * n) / 10 → m * (7 * W) / (10 * n) ≥ W / 3) → 
  ∃ g1 g2 g3 : ℕ, g1 + g2 + g3 < W / 3 :=
by
  sorry

end work_completion_l120_120203


namespace bus_passenger_count_l120_120182

-- Definitions for conditions
def initial_passengers : ℕ := 0
def passengers_first_stop (initial : ℕ) : ℕ := initial + 7
def passengers_second_stop (after_first : ℕ) : ℕ := after_first - 3 + 5
def passengers_third_stop (after_second : ℕ) : ℕ := after_second - 2 + 4

-- Statement we want to prove
theorem bus_passenger_count : 
  passengers_third_stop (passengers_second_stop (passengers_first_stop initial_passengers)) = 11 :=
by
  -- proof would go here
  sorry

end bus_passenger_count_l120_120182


namespace eval_expression_l120_120986

theorem eval_expression : 4 * (8 - 3) - 7 = 13 := by
  sorry

end eval_expression_l120_120986


namespace perpendicular_condition_l120_120399

theorem perpendicular_condition (a : ℝ) :
  (2 * a * x + (a - 1) * y + 2 = 0) ∧ ((a + 1) * x + 3 * a * y + 3 = 0) →
  (a = 1/5 ↔ ∃ x y: ℝ, ((- (2 * a / (a - 1))) * (-(a + 1) / (3 * a)) = -1)) :=
by
  sorry

end perpendicular_condition_l120_120399


namespace least_value_of_fourth_integer_l120_120886

theorem least_value_of_fourth_integer :
  ∃ (A B C D : ℕ), 
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
    A + B + C + D = 64 ∧ 
    A = 3 * B ∧ B = C - 2 ∧ 
    D = 52 := sorry

end least_value_of_fourth_integer_l120_120886


namespace Carson_skipped_times_l120_120446

variable (length width total_circles actual_distance perimeter distance_skipped : ℕ)
variable (total_distance : ℕ)

def perimeter_calculation (length width : ℕ) : ℕ := 2 * (length + width)

def total_distance_calculation (total_circles perimeter : ℕ) : ℕ := total_circles * perimeter

def distance_skipped_calculation (total_distance actual_distance : ℕ) : ℕ := total_distance - actual_distance

def times_skipped_calculation (distance_skipped perimeter : ℕ) : ℕ := distance_skipped / perimeter

theorem Carson_skipped_times (h_length : length = 600) 
                             (h_width : width = 400) 
                             (h_total_circles : total_circles = 10) 
                             (h_actual_distance : actual_distance = 16000) 
                             (h_perimeter : perimeter = perimeter_calculation length width) 
                             (h_total_distance : total_distance = total_distance_calculation total_circles perimeter) 
                             (h_distance_skipped : distance_skipped = distance_skipped_calculation total_distance actual_distance) :
                             times_skipped_calculation distance_skipped perimeter = 2 := 
by
  simp [perimeter_calculation, total_distance_calculation, distance_skipped_calculation, times_skipped_calculation]
  sorry

end Carson_skipped_times_l120_120446


namespace find_blue_yarn_count_l120_120148

def scarves_per_yarn : ℕ := 3
def red_yarn_count : ℕ := 2
def yellow_yarn_count : ℕ := 4
def total_scarves : ℕ := 36

def scarves_from_red_and_yellow : ℕ :=
  red_yarn_count * scarves_per_yarn + yellow_yarn_count * scarves_per_yarn

def blue_scarves : ℕ :=
  total_scarves - scarves_from_red_and_yellow

def blue_yarn_count : ℕ :=
  blue_scarves / scarves_per_yarn

theorem find_blue_yarn_count :
  blue_yarn_count = 6 :=
by 
  sorry

end find_blue_yarn_count_l120_120148


namespace rosalina_gifts_l120_120039

theorem rosalina_gifts (Emilio_gifts Jorge_gifts Pedro_gifts : ℕ) 
  (hEmilio : Emilio_gifts = 11) 
  (hJorge : Jorge_gifts = 6) 
  (hPedro : Pedro_gifts = 4) : 
  Emilio_gifts + Jorge_gifts + Pedro_gifts = 21 :=
by
  sorry

end rosalina_gifts_l120_120039


namespace increase_by_one_unit_l120_120264

-- Define the regression equation
def regression_eq (x : ℝ) : ℝ := 2 + 3 * x

-- State the theorem
theorem increase_by_one_unit (x : ℝ) : regression_eq (x + 1) - regression_eq x = 3 := by
  sorry

end increase_by_one_unit_l120_120264


namespace Peter_can_guarantee_victory_l120_120067

structure Board :=
  (size : ℕ)
  (cells : Fin size × Fin size → Option Color)

inductive Player
  | Peter
  | Victor
deriving DecidableEq

inductive Color
  | Red
  | Green
  | White
deriving DecidableEq

structure Move :=
  (player : Player)
  (rectangle : Fin 2 × Fin 2)
  (position : Fin 7 × Fin 7)

def isValidMove (board : Board) (move : Move) : Prop := sorry

def applyMove (board : Board) (move : Move) : Board := sorry

def allCellsColored (board : Board) : Prop := sorry

theorem Peter_can_guarantee_victory :
  ∀ (initialBoard : Board),
    (∀ (move : Move), move.player = Player.Victor → isValidMove initialBoard move) →
    Player.Peter = Player.Peter →
    (∃ finalBoard : Board,
       allCellsColored finalBoard ∧ 
       ¬ (∃ (move : Move), move.player = Player.Victor ∧ isValidMove finalBoard move)) :=
sorry

end Peter_can_guarantee_victory_l120_120067


namespace units_digit_quotient_4_l120_120057

theorem units_digit_quotient_4 (n : ℕ) (h₁ : n ≥ 1) :
  (5^1994 + 6^1994) % 10 = 1 ∧ (5^1994 + 6^1994) % 7 = 5 → 
  (5^1994 + 6^1994) / 7 % 10 = 4 := 
sorry

end units_digit_quotient_4_l120_120057


namespace not_product_of_two_integers_l120_120594

theorem not_product_of_two_integers (n : ℕ) (hn : n > 0) :
  ∀ t k : ℕ, t * (t + k) = n^2 + n + 1 → k ≥ 2 * Nat.sqrt n :=
by
  sorry

end not_product_of_two_integers_l120_120594


namespace boys_in_school_l120_120696

theorem boys_in_school (B G1 G2 : ℕ) (h1 : G1 = 632) (h2 : G2 = G1 + 465) (h3 : G2 = B + 687) : B = 410 :=
by
  sorry

end boys_in_school_l120_120696


namespace proof_problem_l120_120677

noncomputable def problem (a b c d : ℝ) : Prop :=
(a + b + c = 3) ∧ 
(a + b + d = -1) ∧ 
(a + c + d = 8) ∧ 
(b + c + d = 0) ∧ 
(a * b + c * d = -127 / 9)

theorem proof_problem (a b c d : ℝ) : 
  (a + b + c = 3) → 
  (a + b + d = -1) →
  (a + c + d = 8) → 
  (b + c + d = 0) → 
  (a * b + c * d = -127 / 9) :=
by 
  intro h1 h2 h3 h4
  -- Proof is omitted, "sorry" indicates it is to be filled in
  admit

end proof_problem_l120_120677


namespace range_of_m_l120_120983

-- Definitions of propositions and their negations
def p (x : ℝ) : Prop := x + 2 ≥ 0 ∧ x - 10 ≤ 0
def q (x m : ℝ) : Prop := 1 - m ≤ x ∧ x ≤ 1 + m ∧ m > 0
def not_p (x : ℝ) : Prop := x < -2 ∨ x > 10
def not_q (x m : ℝ) : Prop := x < (1 - m) ∨ x > (1 + m) ∧ m > 0

-- Statement that \neg p is a necessary but not sufficient condition for \neg q
def necessary_but_not_sufficient (x m : ℝ) : Prop := 
  (∀ x, not_q x m → not_p x) ∧ ¬(∀ x, not_p x → not_q x m)

-- The main theorem to prove
theorem range_of_m (m : ℝ) : (∀ x, necessary_but_not_sufficient x m) ↔ 9 ≤ m :=
by
  sorry

end range_of_m_l120_120983


namespace fraction_interval_l120_120793

theorem fraction_interval :
  (5 / 24 > 1 / 6) ∧ (5 / 24 < 1 / 4) ∧
  (¬ (5 / 12 > 1 / 6 ∧ 5 / 12 < 1 / 4)) ∧
  (¬ (5 / 36 > 1 / 6 ∧ 5 / 36 < 1 / 4)) ∧
  (¬ (5 / 60 > 1 / 6 ∧ 5 / 60 < 1 / 4)) ∧
  (¬ (5 / 48 > 1 / 6 ∧ 5 / 48 < 1 / 4)) :=
by
  sorry

end fraction_interval_l120_120793


namespace price_per_book_sold_l120_120861

-- Definitions based on the given conditions
def total_books_before_sale : ℕ := 3 * 50
def books_sold : ℕ := 2 * 50
def total_amount_received : ℕ := 500

-- Target statement to be proved
theorem price_per_book_sold :
  (total_amount_received : ℚ) / books_sold = 5 :=
sorry

end price_per_book_sold_l120_120861


namespace number_of_pages_in_bible_l120_120221

-- Definitions based on conditions
def hours_per_day := 2
def pages_per_hour := 50
def weeks := 4
def days_per_week := 7

-- Hypotheses transformed into mathematical facts
def total_days := weeks * days_per_week
def total_hours := total_days * hours_per_day
def total_pages := total_hours * pages_per_hour

-- Theorem to prove the Bible length based on conditions
theorem number_of_pages_in_bible : total_pages = 2800 := 
by
  sorry

end number_of_pages_in_bible_l120_120221


namespace ratio_of_blue_to_red_l120_120118

variable (B : ℕ) -- Number of blue lights

def total_white := 59
def total_colored := total_white - 5
def red_lights := 12
def green_lights := 6

def total_bought := red_lights + green_lights + B

theorem ratio_of_blue_to_red (h : total_bought = total_colored) :
  B / red_lights = 3 :=
by
  sorry

end ratio_of_blue_to_red_l120_120118


namespace find_m_l120_120836

theorem find_m (
  x : ℚ 
) (m : ℚ) 
  (h1 : 4 * x + 2 * m = 3 * x + 1) 
  (h2 : 3 * x + 2 * m = 6 * x + 1) 
: m = 1/2 := 
  sorry

end find_m_l120_120836


namespace james_out_of_pocket_l120_120228

-- Definitions based on conditions
def old_car_value : ℝ := 20000
def old_car_sold_for : ℝ := 0.80 * old_car_value
def new_car_sticker_price : ℝ := 30000
def new_car_bought_for : ℝ := 0.90 * new_car_sticker_price

-- Question and proof statement
def amount_out_of_pocket : ℝ := new_car_bought_for - old_car_sold_for

theorem james_out_of_pocket : amount_out_of_pocket = 11000 := by
  sorry

end james_out_of_pocket_l120_120228


namespace current_speed_correct_l120_120265

noncomputable def speed_of_current : ℝ :=
  let rowing_speed_still_water := 10 -- speed of rowing in still water in kmph
  let distance_meters := 60 -- distance covered in meters
  let time_seconds := 17.998560115190788 -- time taken in seconds
  let distance_km := distance_meters / 1000 -- converting distance to kilometers
  let time_hours := time_seconds / 3600 -- converting time to hours
  let downstream_speed := distance_km / time_hours -- calculating downstream speed
  downstream_speed - rowing_speed_still_water -- calculating and returning the speed of the current

theorem current_speed_correct : speed_of_current = 2.00048 := by
  -- The proof is not provided in this statement as per the requirements.
  sorry

end current_speed_correct_l120_120265


namespace cuboid_volume_l120_120890

/-- Given a cuboid with edges 6 cm, 5 cm, and 6 cm, the volume of the cuboid
    is 180 cm³. -/
theorem cuboid_volume (a b c : ℕ) (h1 : a = 6) (h2 : b = 5) (h3 : c = 6) :
  a * b * c = 180 := by
  sorry

end cuboid_volume_l120_120890


namespace original_total_cost_l120_120848

-- Definitions based on the conditions
def price_jeans : ℝ := 14.50
def price_shirt : ℝ := 9.50
def price_jacket : ℝ := 21.00

def jeans_count : ℕ := 2
def shirts_count : ℕ := 4
def jackets_count : ℕ := 1

-- The proof statement
theorem original_total_cost :
  (jeans_count * price_jeans) + (shirts_count * price_shirt) + (jackets_count * price_jacket) = 88 := 
by
  sorry

end original_total_cost_l120_120848


namespace intersection_of_sets_l120_120620

def SetA : Set ℝ := {x | 0 < x ∧ x < 3}
def SetB : Set ℝ := {x | x > 2}
def SetC : Set ℝ := {x | 2 < x ∧ x < 3}

theorem intersection_of_sets :
  SetA ∩ SetB = SetC :=
by
  sorry

end intersection_of_sets_l120_120620


namespace root_one_value_of_m_real_roots_range_of_m_l120_120354

variables {m x : ℝ}

-- Part 1: Prove that if 1 is a root of 'mx^2 - 4x + 1 = 0', then m = 3
theorem root_one_value_of_m (h : m * 1^2 - 4 * 1 + 1 = 0) : m = 3 :=
  by sorry

-- Part 2: Prove that 'mx^2 - 4x + 1 = 0' has real roots iff 'm ≤ 4 ∧ m ≠ 0'
theorem real_roots_range_of_m : (∃ x : ℝ, m * x^2 - 4 * x + 1 = 0) ↔ (m ≤ 4 ∧ m ≠ 0) :=
  by sorry

end root_one_value_of_m_real_roots_range_of_m_l120_120354


namespace general_formula_for_a_n_a_n_greater_than_b_n_find_n_in_geometric_sequence_l120_120731

-- Defines the sequences and properties given in the problem
def sequences (a_n b_n S_n T_n : ℕ → ℕ) : Prop :=
  a_n 1 = 1 ∧ S_n 2 = 4 ∧ 
  (∀ n : ℕ, 3 * S_n (n + 1) = 2 * S_n n + S_n (n + 2) + a_n n)

-- (1) Prove the general formula for {a_n}
theorem general_formula_for_a_n
  (a_n b_n S_n T_n : ℕ → ℕ)
  (h : sequences a_n b_n S_n T_n) :
  ∀ n : ℕ, a_n n = 2 * n - 1 :=
sorry

-- (2) If {b_n} is an arithmetic sequence and ∀n ∈ ℕ, S_n > T_n, prove a_n > b_n
theorem a_n_greater_than_b_n
  (a_n b_n S_n T_n : ℕ → ℕ)
  (h : sequences a_n b_n S_n T_n)
  (arithmetic_b : ∃ d: ℕ, ∀ n: ℕ, b_n n = b_n 0 + n * d)
  (Sn_greater_Tn : ∀ (n : ℕ), S_n n > T_n n) :
  ∀ n : ℕ, a_n n > b_n n :=
sorry

-- (3) If {b_n} is a geometric sequence, find n such that (a_n + 2 * T_n) / (b_n + 2 * S_n) = a_k
theorem find_n_in_geometric_sequence
  (a_n b_n S_n T_n : ℕ → ℕ)
  (h : sequences a_n b_n S_n T_n)
  (geometric_b : ∃ r: ℕ, ∀ n: ℕ, b_n n = b_n 0 * r^n)
  (b1_eq_1 : b_n 1 = 1)
  (b2_eq_3 : b_n 2 = 3)
  (k : ℕ) :
  ∃ n : ℕ, (a_n n + 2 * T_n n) / (b_n n + 2 * S_n n) = a_n k := 
sorry

end general_formula_for_a_n_a_n_greater_than_b_n_find_n_in_geometric_sequence_l120_120731


namespace triangle_base_length_l120_120767

theorem triangle_base_length (height : ℝ) (area : ℝ) (base : ℝ) 
  (h_height : height = 6) (h_area : area = 9) 
  (h_formula : area = (1/2) * base * height) : 
  base = 3 :=
by
  sorry

end triangle_base_length_l120_120767


namespace area_of_one_postcard_is_150_cm2_l120_120466

/-- Define the conditions of the problem. -/
def perimeter_of_stitched_postcard : ℕ := 70
def vertical_length_of_postcard : ℕ := 15

/-- Definition stating that postcards are attached horizontally and do not overlap. 
    This logically implies that the horizontal length gets doubled and perimeter is 2V + 4H. -/
def attached_horizontally (V H : ℕ) (P : ℕ) : Prop :=
  2 * V + 4 * H = P

/-- Main theorem stating the question and the derived answer,
    proving that the area of one postcard is 150 square centimeters. -/
theorem area_of_one_postcard_is_150_cm2 :
  ∃ (H : ℕ), attached_horizontally vertical_length_of_postcard H perimeter_of_stitched_postcard ∧
  (vertical_length_of_postcard * H = 150) :=
by 
  sorry -- the proof is omitted

end area_of_one_postcard_is_150_cm2_l120_120466


namespace parallel_lines_coefficient_l120_120679

theorem parallel_lines_coefficient (a : ℝ) :
  (x + 2*a*y - 1 = 0) → (3*a - 1)*x - a*y - 1 = 0 → (a = 0 ∨ a = 1/6) :=
by
  sorry

end parallel_lines_coefficient_l120_120679


namespace tencent_technological_innovation_basis_tencent_innovative_development_analysis_l120_120702

-- Define the dialectical materialist basis conditions
variable (dialectical_negation essence_innovation development_perspective unity_of_opposites : Prop)

-- Define Tencent's emphasis on technological innovation
variable (tencent_innovation : Prop)

-- Define the relationship between Tencent's development and materialist view of development
variable (unity_of_things_developmental progressiveness_tortuosity quantitative_qualitative_changes : Prop)
variable (tencent_development : Prop)

-- Prove that Tencent's emphasis on technological innovation aligns with dialectical materialism
theorem tencent_technological_innovation_basis :
  dialectical_negation ∧ essence_innovation ∧ development_perspective ∧ unity_of_opposites → tencent_innovation :=
by sorry

-- Prove that Tencent's innovative development aligns with dialectical materialist view of development
theorem tencent_innovative_development_analysis :
  unity_of_things_developmental ∧ progressiveness_tortuosity ∧ quantitative_qualitative_changes → tencent_development :=
by sorry

end tencent_technological_innovation_basis_tencent_innovative_development_analysis_l120_120702


namespace mod_2021_2022_2023_2024_eq_zero_mod_7_l120_120875

theorem mod_2021_2022_2023_2024_eq_zero_mod_7 :
  (2021 * 2022 * 2023 * 2024) % 7 = 0 := by
  sorry

end mod_2021_2022_2023_2024_eq_zero_mod_7_l120_120875


namespace rice_in_each_container_ounces_l120_120521

-- Given conditions
def total_rice_pounds : ℚ := 25 / 4
def num_containers : ℕ := 4
def pounds_to_ounces : ℚ := 16

-- Problem statement: proving the amount of rice in each container in ounces
theorem rice_in_each_container_ounces :
  (total_rice_pounds / num_containers) * pounds_to_ounces = 25 :=
by sorry

end rice_in_each_container_ounces_l120_120521


namespace correct_statements_l120_120469

theorem correct_statements : 
  (∃ n : ℕ, 24 = 4 * n) ∧ (∃ n : ℕ, 180 = 9 * n) :=
by
  sorry

end correct_statements_l120_120469


namespace find_N_l120_120870

theorem find_N (a b c : ℤ) (N : ℤ)
  (h1 : a + b + c = 105)
  (h2 : a - 5 = N)
  (h3 : b + 10 = N)
  (h4 : 5 * c = N) : 
  N = 50 :=
by
  sorry

end find_N_l120_120870


namespace binom_20_4_plus_10_l120_120052

open Nat

noncomputable def binom (n k : ℕ) : ℕ := factorial n / (factorial k * factorial (n - k))

theorem binom_20_4_plus_10 :
  binom 20 4 + 10 = 4855 := by
  sorry

end binom_20_4_plus_10_l120_120052


namespace anie_days_to_complete_l120_120657

def normal_work_hours : ℕ := 10
def extra_hours : ℕ := 5
def total_project_hours : ℕ := 1500

theorem anie_days_to_complete :
  (total_project_hours / (normal_work_hours + extra_hours)) = 100 :=
by
  sorry

end anie_days_to_complete_l120_120657


namespace shortest_fence_length_l120_120136

-- We define the conditions given in the problem.
def triangle_side_length : ℕ := 50
def number_of_dotted_lines : ℕ := 13

-- We need to prove that the shortest total length of the fences required to protect all the cabbage from goats equals 650 meters.
theorem shortest_fence_length : number_of_dotted_lines * triangle_side_length = 650 :=
by
  -- The proof steps are omitted as per instructions.
  sorry

end shortest_fence_length_l120_120136


namespace calculate_expression_l120_120196

theorem calculate_expression : (23 + 12)^2 - (23 - 12)^2 = 1104 := by
  sorry

end calculate_expression_l120_120196


namespace fraction_inequality_l120_120684

theorem fraction_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
    (a / b) + (b / c) + (c / a) ≤ (a^2 / b^2) + (b^2 / c^2) + (c^2 / a^2) := 
by
  sorry

end fraction_inequality_l120_120684


namespace cos_neg_11_div_4_pi_eq_neg_sqrt_2_div_2_l120_120014

theorem cos_neg_11_div_4_pi_eq_neg_sqrt_2_div_2 : 
  Real.cos (- (11 / 4) * Real.pi) = - Real.sqrt 2 / 2 := 
sorry

end cos_neg_11_div_4_pi_eq_neg_sqrt_2_div_2_l120_120014


namespace percentage_sum_l120_120818

theorem percentage_sum (A B C : ℕ) (x y : ℕ)
  (hA : A = 120) (hB : B = 110) (hC : C = 100)
  (hAx : A = C * (1 + x / 100))
  (hBy : B = C * (1 + y / 100)) : x + y = 30 := 
by
  sorry

end percentage_sum_l120_120818


namespace solution_set_f_pos_min_a2_b2_c2_l120_120543

def f (x : ℝ) : ℝ := |2 * x + 3| - |x - 1|

theorem solution_set_f_pos : 
  { x : ℝ | f x > 0 } = { x : ℝ | x < -3 / 2 ∨ -2 / 3 < x } := 
sorry

theorem min_a2_b2_c2 (a b c : ℝ) (h : a + 2 * b + 3 * c = 5) : 
  a^2 + b^2 + c^2 ≥ 25 / 14 :=
sorry

end solution_set_f_pos_min_a2_b2_c2_l120_120543


namespace arithmetic_sum_eight_terms_l120_120433

theorem arithmetic_sum_eight_terms :
  ∀ (a d : ℤ) (n : ℕ), a = -3 → d = 6 → n = 8 → 
  (last_term = a + (n - 1) * d) →
  (last_term = 39) →
  (sum = (n * (a + last_term)) / 2) →
  sum = 144 :=
by
  intros a d n ha hd hn hlast_term hlast_term_value hsum
  sorry

end arithmetic_sum_eight_terms_l120_120433


namespace point_transformations_l120_120427

theorem point_transformations (a b : ℝ) (h : (a ≠ 2 ∨ b ≠ 3))
  (H1 : ∃ x y : ℝ, (x, y) = (2 - (b - 3), 3 + (a - 2)) ∧ (y, x) = (-4, 2)) :
  b - a = -6 :=
by
  sorry

end point_transformations_l120_120427


namespace jamies_shoes_cost_l120_120075

-- Define the costs of items and the total cost.
def cost_total : ℤ := 110
def cost_coat : ℤ := 40
def cost_one_pair_jeans : ℤ := 20

-- Define the number of pairs of jeans.
def num_pairs_jeans : ℕ := 2

-- Define the cost of Jamie's shoes (to be proved).
def cost_jamies_shoes : ℤ := cost_total - (cost_coat + num_pairs_jeans * cost_one_pair_jeans)

theorem jamies_shoes_cost : cost_jamies_shoes = 30 :=
by
  -- Insert proof here
  sorry

end jamies_shoes_cost_l120_120075


namespace red_socks_l120_120056

variable {R : ℕ}

theorem red_socks (h1 : 2 * R + R + 6 * R = 90) : R = 10 := 
by
  sorry

end red_socks_l120_120056


namespace dice_sum_to_11_l120_120372

/-- Define the conditions for the outcomes of the dice rolls -/
def valid_outcomes (x : Fin 5 → ℕ) : Prop :=
  (∀ i, 1 ≤ x i ∧ x i ≤ 6) ∧ (x 0 + x 1 + x 2 + x 3 + x 4 = 11)

/-- Prove that there are exactly 205 ways to achieve a sum of 11 with five different colored dice -/
theorem dice_sum_to_11 : 
  (∃ (s : Finset (Fin 5 → ℕ)), (∀ x ∈ s, valid_outcomes x) ∧ s.card = 205) :=
  by
    sorry

end dice_sum_to_11_l120_120372


namespace find_cd_l120_120658

theorem find_cd : 
  (∀ x : ℝ, (4 * x - 3) / (x^2 - 3 * x - 18) = ((7 / 3) / (x - 6)) + ((5 / 3) / (x + 3))) :=
by
  intro x
  have h : x^2 - 3 * x - 18 = (x - 6) * (x + 3) := by
    sorry
  rw [h]
  sorry

end find_cd_l120_120658


namespace range_of_a_l120_120366

-- Definition of sets A and B
def A : Set ℝ := { x | -1 < x ∧ x < 1 }
def B (a : ℝ) : Set ℝ := { x | x < a }

-- Condition of the union of A and B
theorem range_of_a (a : ℝ) : (A ∪ B a = { x | x < 1 }) ↔ -1 < a ∧ a ≤ 1 :=
by
  sorry

end range_of_a_l120_120366


namespace arithmetic_geometric_sequences_sequence_sum_first_terms_l120_120173

noncomputable def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

noncomputable def geometric_sequence (b : ℕ → ℤ) : Prop :=
  ∃ q : ℤ, ∀ n : ℕ, b (n + 1) = b n * q

noncomputable def sequence_sum (a : ℕ → ℤ) (S : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, S n = (n + 1) * a 1 + (n * (n + 1)) / 2

theorem arithmetic_geometric_sequences
  (a b S : ℕ → ℤ)
  (h1 : arithmetic_sequence a)
  (h2 : geometric_sequence b)
  (h3 : a 0 = 1)
  (h4 : b 0 = 1)
  (h5 : b 2 * S 2 = 36)
  (h6 : b 1 * S 1 = 8) :
  ((∀ n, a n = 2 * n + 1) ∧ (∀ n, b n = 2 ^ n)) ∨
  ((∀ n, a n = -(2 * n / 3) + 5 / 3) ∧ (∀ n, b n = 6 ^ n)) :=
sorry

theorem sequence_sum_first_terms
  (a : ℕ → ℤ)
  (h : ∀ n, a n = 2 * n + 1)
  (S : ℕ → ℤ)
  (T : ℕ → ℚ)
  (hS : sequence_sum a S)
  (n : ℕ) :
  T n = n / (2 * n + 1) :=
sorry

end arithmetic_geometric_sequences_sequence_sum_first_terms_l120_120173


namespace even_function_m_value_l120_120476

def f (x m : ℝ) : ℝ := (x - 2) * (x - m)

theorem even_function_m_value (m : ℝ) :
  (∀ x : ℝ, f x m = f (-x) m) → m = -2 := by
  sorry

end even_function_m_value_l120_120476


namespace correct_polynomial_l120_120862

noncomputable def p : Polynomial ℤ :=
  Polynomial.C 1 * Polynomial.X^6 - Polynomial.C 8 * Polynomial.X^4 - Polynomial.C 2 * Polynomial.X^3 + Polynomial.C 13 * Polynomial.X^2 - Polynomial.C 10 * Polynomial.X - Polynomial.C 1

theorem correct_polynomial (r t : ℝ) :
  (r^3 - r - 1 = 0) → (t = r + Real.sqrt 2) → Polynomial.aeval t p = 0 :=
by
  sorry

end correct_polynomial_l120_120862


namespace factorization_of_expression_l120_120024

noncomputable def factorized_form (x : ℝ) : ℝ :=
  (x + 5 / 2 + Real.sqrt 13 / 2) * (x + 5 / 2 - Real.sqrt 13 / 2)

theorem factorization_of_expression (x : ℝ) :
  x^2 - 5 * x + 3 = factorized_form x :=
by
  sorry

end factorization_of_expression_l120_120024


namespace fraction_half_way_l120_120825

theorem fraction_half_way (x y : ℚ) (h1 : x = 3 / 4) (h2 : y = 5 / 6) : (x + y) / 2 = 19 / 24 := by
  sorry

end fraction_half_way_l120_120825


namespace sequence_x_value_l120_120683

theorem sequence_x_value
  (z y x : ℤ)
  (h1 : z + (-2) = -1)
  (h2 : y + 1 = -2)
  (h3 : x + (-3) = 1) :
  x = 4 := 
sorry

end sequence_x_value_l120_120683


namespace even_function_a_equals_one_l120_120096

theorem even_function_a_equals_one 
  (a : ℝ) 
  (h : ∀ x : ℝ, 2^(-x) + a * 2^x = 2^x + a * 2^(-x)) : 
  a = 1 := 
by
  sorry

end even_function_a_equals_one_l120_120096


namespace ordinate_of_point_A_l120_120429

noncomputable def p : ℝ := 1 / 4
noncomputable def distance_to_focus (y₀ : ℝ) : ℝ := y₀ + p / 2

theorem ordinate_of_point_A :
  ∃ y₀ : ℝ, (distance_to_focus y₀ = 9 / 8) → y₀ = 1 :=
by
  -- Assume solution steps here
  sorry

end ordinate_of_point_A_l120_120429


namespace point_on_hyperbola_l120_120091

theorem point_on_hyperbola : 
  (∃ x y : ℝ, (x, y) = (3, -2) ∧ y = -6 / x) :=
by
  sorry

end point_on_hyperbola_l120_120091


namespace solve_equation_l120_120909

theorem solve_equation (x : ℝ) (h : x * (x - 3) = 10) : x = 5 ∨ x = -2 :=
by sorry

end solve_equation_l120_120909


namespace trigonometric_identity_l120_120631

theorem trigonometric_identity (α : ℝ) (h : Real.sin (π + α) = -1/3) : Real.sin (2 * α) / Real.cos α = 2 / 3 := by
  sorry

end trigonometric_identity_l120_120631


namespace find_intersection_l120_120655

noncomputable def intersection_of_lines : Prop :=
  ∃ (x y : ℚ), (5 * x - 3 * y = 15) ∧ (6 * x + 2 * y = 14) ∧ (x = 11 / 4) ∧ (y = -5 / 4)

theorem find_intersection : intersection_of_lines :=
  sorry

end find_intersection_l120_120655


namespace Calvin_mistake_correct_l120_120922

theorem Calvin_mistake_correct (a : ℕ) : 37 + 31 * a = 37 * 31 + a → a = 37 :=
sorry

end Calvin_mistake_correct_l120_120922


namespace rhind_papyrus_problem_l120_120331

theorem rhind_papyrus_problem 
  (a1 a2 a3 a4 a5 : ℚ)
  (h1 : a2 = a1 + d)
  (h2 : a3 = a1 + 2 * d)
  (h3 : a4 = a1 + 3 * d)
  (h4 : a5 = a1 + 4 * d)
  (h_sum : a1 + a2 + a3 + a4 + a5 = 60)
  (h_condition : (a4 + a5) / 2 = a1 + a2 + a3) :
  a1 = 4 / 3 :=
by
  sorry

end rhind_papyrus_problem_l120_120331


namespace original_students_l120_120453

theorem original_students (a b : ℕ) : 
  a + b = 92 ∧ a - 5 = 3 * (b + 5 - 32) → a = 45 ∧ b = 47 :=
by sorry

end original_students_l120_120453


namespace smallest_positive_x_for_palindrome_l120_120857

def is_palindrome (n : ℕ) : Prop :=
  let s := n.digits 10
  s = s.reverse

theorem smallest_positive_x_for_palindrome :
  ∃ x : ℕ, x > 0 ∧ is_palindrome (x + 1234) ∧ (∀ y : ℕ, y > 0 → is_palindrome (y + 1234) → x ≤ y) ∧ x = 97 := 
sorry

end smallest_positive_x_for_palindrome_l120_120857


namespace num_ordered_pairs_l120_120336

theorem num_ordered_pairs :
  ∃ n : ℕ, n = 49 ∧ ∀ (a b : ℕ), a + b = 50 → 0 < a ∧ 0 < b → (1 ≤ a ∧ a < 50) :=
by
  sorry

end num_ordered_pairs_l120_120336


namespace intersection_A_B_l120_120668

-- Define set A
def A : Set ℤ := {-1, 1, 2, 3, 4}

-- Define set B with the given condition
def B : Set ℤ := {x : ℤ | 1 ≤ x ∧ x < 3}

-- The main theorem statement showing the intersection of A and B
theorem intersection_A_B : A ∩ B = {1, 2} :=
    sorry -- Placeholder for the proof

end intersection_A_B_l120_120668


namespace puddle_base_area_l120_120321

theorem puddle_base_area (rate depth hours : ℝ) (A : ℝ) 
  (h1 : rate = 10) 
  (h2 : depth = 30) 
  (h3 : hours = 3) 
  (h4 : depth * A = rate * hours) : 
  A = 1 := 
by 
  sorry

end puddle_base_area_l120_120321


namespace rectangle_area_l120_120688

theorem rectangle_area (P l w : ℝ) (h1 : P = 60) (h2 : l / w = 3 / 2) (h3 : P = 2 * l + 2 * w) : l * w = 216 :=
by
  sorry

end rectangle_area_l120_120688


namespace union_sets_l120_120608

def setA : Set ℝ := { x | abs (x - 1) < 3 }
def setB : Set ℝ := { x | x^2 - 4 * x < 0 }

theorem union_sets :
  setA ∪ setB = { x : ℝ | -2 < x ∧ x < 4 } :=
sorry

end union_sets_l120_120608


namespace sum_consecutive_equals_prime_l120_120284

theorem sum_consecutive_equals_prime (m k p : ℕ) (h_prime : Nat.Prime p) :
  (∃ S, S = (m * (2 * k + m - 1)) / 2 ∧ S = p) →
  m = 1 ∨ m = 2 :=
sorry

end sum_consecutive_equals_prime_l120_120284


namespace find_a8_l120_120753

variable (a : ℕ → ℝ)
variable (S : ℕ → ℝ)

-- Conditions
def geom_sequence (a : ℕ → ℝ) (a1 q : ℝ) :=
  ∀ n, a n = a1 * q ^ n

def sum_geom_sequence (S a : ℕ → ℝ) (a1 q : ℝ) :=
  ∀ n, S n = a1 * (1 - q ^ (n + 1)) / (1 - q)

def arithmetic_sequence (S : ℕ → ℝ) :=
  S 9 = S 3 + S 6

def sum_a2_a5 (a : ℕ → ℝ) :=
  a 2 + a 5 = 4

theorem find_a8 (a : ℕ → ℝ) (S : ℕ → ℝ) (a1 q : ℝ)
  (hgeom_seq : geom_sequence a a1 q)
  (hsum_geom_seq : sum_geom_sequence S a a1 q)
  (harith_seq : arithmetic_sequence S)
  (hsum_a2_a5 : sum_a2_a5 a) :
  a 8 = 2 :=
sorry

end find_a8_l120_120753


namespace find_triples_l120_120892

theorem find_triples (a m n : ℕ) (h1 : a ≥ 2) (h2 : m ≥ 2) :
  a^n + 203 ∣ a^(m * n) + 1 → ∃ (k : ℕ), (k ≥ 1) := 
sorry

end find_triples_l120_120892


namespace problem_statement_l120_120506

theorem problem_statement :
  75 * ((4 + 1/3) - (5 + 1/4)) / ((3 + 1/2) + (2 + 1/5)) = -5/31 := 
by
  sorry

end problem_statement_l120_120506


namespace find_constant_t_l120_120294

theorem find_constant_t : ∃ t : ℝ, 
  (∀ x : ℝ, (3 * x^2 - 4 * x + 5) * (2 * x^2 + t * x + 8) = 6 * x^4 + (-26) * x^3 + 58 * x^2 + (-76) * x + 40) ↔ t = -6 :=
by {
  sorry
}

end find_constant_t_l120_120294


namespace find_angle_C_l120_120992

-- Given conditions
variable {A B C : ℝ}
variable (h_triangle : A + B + C = π)
variable (h_tanA : Real.tan A = 1/2)
variable (h_cosB : Real.cos B = 3 * Real.sqrt 10 / 10)

-- The proof statement
theorem find_angle_C :
  C = 3 * π / 4 := by
  sorry

end find_angle_C_l120_120992


namespace tangent_lines_through_P_l120_120063

noncomputable def curve_eq (x : ℝ) : ℝ := 1/3 * x^3 + 4/3

theorem tangent_lines_through_P (x y : ℝ) :
  ((4 * x - y - 4 = 0 ∨ y = x + 2) ∧ (curve_eq 2 = 4)) :=
by
  sorry

end tangent_lines_through_P_l120_120063


namespace operation_is_addition_l120_120770

theorem operation_is_addition : (5 + (-5) = 0) :=
by
  sorry

end operation_is_addition_l120_120770


namespace problem_statement_l120_120747

open Real

noncomputable def f (ω varphi : ℝ) (x : ℝ) := 2 * sin (ω * x + varphi)

theorem problem_statement (ω varphi : ℝ) (x1 x2 : ℝ) (hω_pos : ω > 0) (hvarphi_abs : abs varphi < π / 2)
    (hf0 : f ω varphi 0 = -1) (hmonotonic : ∀ x y, π / 18 < x ∧ x < y ∧ y < π / 3 → f ω varphi x < f ω varphi y)
    (hshift : ∀ x, f ω varphi (x + π) = f ω varphi x)
    (hx1x2_interval : -17 * π / 12 < x1 ∧ x1 < -2 * π / 3 ∧ -17 * π / 12 < x2 ∧ x2 < -2 * π / 3 ∧ x1 ≠ x2)
    (heq_fx : f ω varphi x1 = f ω varphi x2) :
    f ω varphi (x1 + x2) = -1 :=
sorry

end problem_statement_l120_120747


namespace rancher_cows_l120_120276

theorem rancher_cows (H C : ℕ) (h1 : C = 5 * H) (h2 : C + H = 168) : C = 140 := by
  sorry

end rancher_cows_l120_120276


namespace grain_milling_l120_120981

theorem grain_milling (W : ℝ) (h : 0.9 * W = 100) : W = 111.1 :=
sorry

end grain_milling_l120_120981


namespace smallest_b_for_factoring_l120_120301

theorem smallest_b_for_factoring :
  ∃ b : ℕ, b > 0 ∧
    (∀ r s : ℤ, r * s = 2016 → r + s ≠ b) ∧
    (∀ r s : ℤ, r * s = 2016 → r + s = b → b = 92) :=
sorry

end smallest_b_for_factoring_l120_120301


namespace inequality_always_true_l120_120072

theorem inequality_always_true (a : ℝ) :
  (∀ x : ℝ, (a - 2) * x^2 + 2 * (a - 2) * x - 4 < 0) ↔ (-2 < a ∧ a ≤ 2) :=
sorry

end inequality_always_true_l120_120072


namespace vincent_earnings_after_5_days_l120_120832

def fantasy_book_price : ℕ := 4
def daily_fantasy_books_sold : ℕ := 5
def literature_book_price : ℕ := fantasy_book_price / 2
def daily_literature_books_sold : ℕ := 8
def days : ℕ := 5

def daily_earnings : ℕ :=
  (fantasy_book_price * daily_fantasy_books_sold) +
  (literature_book_price * daily_literature_books_sold)

def total_earnings (d : ℕ) : ℕ :=
  daily_earnings * d

theorem vincent_earnings_after_5_days : total_earnings days = 180 := by
  sorry

end vincent_earnings_after_5_days_l120_120832


namespace unpainted_cubes_count_l120_120854

/- Definitions of the conditions -/
def total_cubes : ℕ := 6 * 6 * 6
def painted_faces_per_face : ℕ := 4
def total_faces : ℕ := 6
def painted_faces : ℕ := painted_faces_per_face * total_faces
def overlapped_painted_faces : ℕ := 4 -- Each center four squares on one face corresponds to a center square on the opposite face.
def unique_painted_cubes : ℕ := painted_faces / 2

/- Lean Theorem statement that corresponds to proving the question asked in the problem -/
theorem unpainted_cubes_count : 
  total_cubes - unique_painted_cubes = 208 :=
  by
    sorry

end unpainted_cubes_count_l120_120854


namespace factorize_poly1_min_value_poly2_l120_120910

-- Define the polynomials
def poly1 := fun (x : ℝ) => x^2 + 2 * x - 3
def factored_poly1 := fun (x : ℝ) => (x - 1) * (x + 3)

def poly2 := fun (x : ℝ) => x^2 + 4 * x + 5
def min_value := 1

-- State the theorems without providing proofs
theorem factorize_poly1 : ∀ x : ℝ, poly1 x = factored_poly1 x := 
by { sorry }

theorem min_value_poly2 : ∀ x : ℝ, poly2 x ≥ min_value := 
by { sorry }

end factorize_poly1_min_value_poly2_l120_120910


namespace manufacturing_cost_of_shoe_l120_120113

theorem manufacturing_cost_of_shoe
  (transportation_cost_per_shoe : ℝ)
  (selling_price_per_shoe : ℝ)
  (gain_percentage : ℝ)
  (manufacturing_cost : ℝ)
  (H1 : transportation_cost_per_shoe = 5)
  (H2 : selling_price_per_shoe = 282)
  (H3 : gain_percentage = 0.20)
  (H4 : selling_price_per_shoe = (manufacturing_cost + transportation_cost_per_shoe) * (1 + gain_percentage)) :
  manufacturing_cost = 230 :=
sorry

end manufacturing_cost_of_shoe_l120_120113


namespace diana_owes_amount_l120_120624

def principal : ℝ := 75
def rate : ℝ := 0.07
def time : ℝ := 1
def interest : ℝ := principal * rate * time
def total_amount_owed : ℝ := principal + interest

theorem diana_owes_amount :
  total_amount_owed = 80.25 :=
by
  sorry

end diana_owes_amount_l120_120624


namespace absolute_value_simplification_l120_120744

theorem absolute_value_simplification (a b : ℝ) (ha : a < 0) (hb : b > 0) : |a - b| + |b - a| = -2 * a + 2 * b := 
by 
  sorry

end absolute_value_simplification_l120_120744


namespace village_current_population_l120_120560

def initial_population : ℕ := 4675
def died_by_bombardment : ℕ := (5*initial_population + 99) / 100 -- Equivalent to rounding (5/100) * 4675
def remaining_after_bombardment : ℕ := initial_population - died_by_bombardment
def left_due_to_fear : ℕ := (20*remaining_after_bombardment + 99) / 100 -- Equivalent to rounding (20/100) * remaining
def current_population : ℕ := remaining_after_bombardment - left_due_to_fear

theorem village_current_population : current_population = 3553 := by
  sorry

end village_current_population_l120_120560


namespace gcd_1729_1314_l120_120140

theorem gcd_1729_1314 : Nat.gcd 1729 1314 = 1 :=
by
  sorry

end gcd_1729_1314_l120_120140


namespace fred_baseball_cards_l120_120219

theorem fred_baseball_cards :
  ∀ (fred_cards_initial melanie_bought : ℕ), fred_cards_initial = 5 → melanie_bought = 3 → fred_cards_initial - melanie_bought = 2 :=
by
  intros fred_cards_initial melanie_bought h1 h2
  sorry

end fred_baseball_cards_l120_120219


namespace find_higher_selling_price_l120_120621

-- Define the constants and initial conditions
def cost_price : ℕ := 200
def selling_price_1 : ℕ := 340
def gain_1 : ℕ := selling_price_1 - cost_price
def new_gain : ℕ := gain_1 + gain_1 * 5 / 100

-- Define the problem statement
theorem find_higher_selling_price : 
  ∀ P : ℕ, P = cost_price + new_gain → P = 347 :=
by
  intro P
  intro h
  sorry

end find_higher_selling_price_l120_120621


namespace parabola_range_l120_120542

theorem parabola_range (x : ℝ) (h : 0 < x ∧ x < 3) : 
  1 ≤ (x^2 - 4*x + 5) ∧ (x^2 - 4*x + 5) < 5 :=
sorry

end parabola_range_l120_120542


namespace parabolic_arch_height_l120_120038

/-- Define the properties of the parabolic arch -/
def parabolic_arch (a k x : ℝ) : ℝ := a * x^2 + k

/-- Define the conditions of the problem -/
def conditions (a k : ℝ) : Prop :=
  (parabolic_arch a k 25 = 0) ∧ (parabolic_arch a k 0 = 20)

theorem parabolic_arch_height (a k : ℝ) (condition_a_k : conditions a k) :
  parabolic_arch a k 10 = 16.8 :=
by
  unfold conditions at condition_a_k
  cases' condition_a_k with h1 h2
  sorry

end parabolic_arch_height_l120_120038


namespace min_value_of_n_l120_120066

theorem min_value_of_n 
  (n k : ℕ) 
  (h1 : 8 * n = 225 * k + 3)
  (h2 : k ≡ 5 [MOD 8]) : 
  n = 141 := 
  sorry

end min_value_of_n_l120_120066


namespace find_certain_number_l120_120625

theorem find_certain_number (h1 : 2994 / 14.5 = 171) (h2 : ∃ x : ℝ, x / 1.45 = 17.1) : ∃ x : ℝ, x = 24.795 :=
by
  sorry

end find_certain_number_l120_120625


namespace custom_operation_correct_l120_120296

def custom_operation (a b : ℤ) : ℤ := (a + b) * (a - b)

theorem custom_operation_correct : custom_operation 6 3 = 27 :=
by {
  sorry
}

end custom_operation_correct_l120_120296


namespace complement_union_eq_complement_l120_120963

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

end complement_union_eq_complement_l120_120963


namespace g_value_at_49_l120_120326

noncomputable def g (x : ℝ) : ℝ := sorry

theorem g_value_at_49 :
  (∀ x y : ℝ, 0 < x → 0 < y → x * g y - y * g x = g (x^2 / y)) →
  g 49 = 0 :=
by
  -- Assuming the given condition holds for all positive real numbers x and y
  intro h
  -- sorry placeholder represents the proof process
  sorry

end g_value_at_49_l120_120326


namespace compound_h_atoms_l120_120122

theorem compound_h_atoms 
  (weight_H : ℝ) (weight_C : ℝ) (weight_O : ℝ)
  (num_C : ℕ) (num_O : ℕ)
  (total_molecular_weight : ℝ)
  (atomic_weight_H : ℝ) (atomic_weight_C : ℝ) (atomic_weight_O : ℝ)
  (H_w_is_1 : atomic_weight_H = 1)
  (C_w_is_12 : atomic_weight_C = 12)
  (O_w_is_16 : atomic_weight_O = 16)
  (C_atoms_is_1 : num_C = 1)
  (O_atoms_is_3 : num_O = 3)
  (total_mw_is_62 : total_molecular_weight = 62)
  (mw_C : weight_C = num_C * atomic_weight_C)
  (mw_O : weight_O = num_O * atomic_weight_O)
  (mw_CO : weight_C + weight_O = 60)
  (H_weight_contrib : total_molecular_weight - (weight_C + weight_O) = weight_H)
  (H_atoms_calc : weight_H = 2 * atomic_weight_H) :
  2 = 2 :=
by 
  sorry

end compound_h_atoms_l120_120122


namespace sachin_younger_than_rahul_l120_120330

theorem sachin_younger_than_rahul
  (S R : ℝ)
  (h1 : S = 24.5)
  (h2 : S / R = 7 / 9) :
  R - S = 7 := 
by sorry

end sachin_younger_than_rahul_l120_120330


namespace line_parallel_to_y_axis_l120_120775

theorem line_parallel_to_y_axis (a b : ℝ) (h : ∀ x y : ℝ, a * x + b * y + 1 = 0 → b = 0):
  a ≠ 0 ∧ b = 0 :=
sorry

end line_parallel_to_y_axis_l120_120775


namespace value_of_a1_l120_120841

def seq (a : ℕ → ℚ) (a_8 : ℚ) : Prop :=
  ∀ n : ℕ, (a (n + 1) = 1 / (1 - a n)) ∧ a 8 = 2

theorem value_of_a1 (a : ℕ → ℚ) (h : seq a 2) : a 1 = 1 / 2 :=
  sorry

end value_of_a1_l120_120841


namespace john_reams_needed_l120_120311

theorem john_reams_needed 
  (pages_flash_fiction_weekly : ℕ := 20) 
  (pages_short_story_weekly : ℕ := 50) 
  (pages_novel_annual : ℕ := 1500) 
  (weeks_in_year : ℕ := 52) 
  (sheets_per_ream : ℕ := 500) 
  (sheets_flash_fiction_weekly : ℕ := 10)
  (sheets_short_story_weekly : ℕ := 25) :
  let sheets_flash_fiction_annual := sheets_flash_fiction_weekly * weeks_in_year
  let sheets_short_story_annual := sheets_short_story_weekly * weeks_in_year
  let total_sheets_annual := sheets_flash_fiction_annual + sheets_short_story_annual + pages_novel_annual
  let reams_needed := (total_sheets_annual + sheets_per_ream - 1) / sheets_per_ream
  reams_needed = 7 := 
by sorry

end john_reams_needed_l120_120311


namespace time_to_build_wall_l120_120504

theorem time_to_build_wall (t_A t_B t_C : ℝ) 
  (h1 : 1 / t_A + 1 / t_B = 1 / 25)
  (h2 : 1 / t_C = 1 / 35)
  (h3 : 1 / t_A = 1 / t_B + 1 / t_C) : t_B = 87.5 :=
by
  sorry

end time_to_build_wall_l120_120504


namespace total_cookies_l120_120065

def total_chocolate_chip_batches := 5
def cookies_per_chocolate_chip_batch := 8
def total_oatmeal_batches := 3
def cookies_per_oatmeal_batch := 7
def total_sugar_batches := 1
def cookies_per_sugar_batch := 10
def total_double_chocolate_batches := 1
def cookies_per_double_chocolate_batch := 6

theorem total_cookies : 
  (total_chocolate_chip_batches * cookies_per_chocolate_chip_batch) +
  (total_oatmeal_batches * cookies_per_oatmeal_batch) +
  (total_sugar_batches * cookies_per_sugar_batch) +
  (total_double_chocolate_batches * cookies_per_double_chocolate_batch) = 77 :=
by sorry

end total_cookies_l120_120065


namespace base_7_to_base_10_l120_120606

theorem base_7_to_base_10 :
  (3 * 7^2 + 2 * 7^1 + 1 * 7^0) = 162 :=
by
  sorry

end base_7_to_base_10_l120_120606


namespace john_gives_to_stud_owner_l120_120933

variable (initial_puppies : ℕ) (puppies_given_away : ℕ) (puppies_kept : ℕ) (price_per_puppy : ℕ) (profit : ℕ)

theorem john_gives_to_stud_owner
  (h1 : initial_puppies = 8)
  (h2 : puppies_given_away = initial_puppies / 2)
  (h3 : puppies_kept = 1)
  (h4 : price_per_puppy = 600)
  (h5 : profit = 1500) :
  let puppies_left_to_sell := initial_puppies - puppies_given_away - puppies_kept
  let total_sales := puppies_left_to_sell * price_per_puppy
  total_sales - profit = 300 :=
by
  intro puppies_left_to_sell
  intro total_sales
  sorry

end john_gives_to_stud_owner_l120_120933


namespace hundred_days_from_friday_is_sunday_l120_120425

/-- Given that today is Friday, determine that 100 days from now is Sunday. -/
theorem hundred_days_from_friday_is_sunday (today : ℕ) (days_in_week : ℕ := 7) 
(friday : ℕ := 0) (sunday : ℕ := 2) : (((today + 100) % days_in_week) = sunday) :=
sorry

end hundred_days_from_friday_is_sunday_l120_120425


namespace fraction_halfway_between_one_fourth_and_one_sixth_l120_120175

theorem fraction_halfway_between_one_fourth_and_one_sixth :
  (1/4 + 1/6) / 2 = 5 / 24 :=
by
  sorry

end fraction_halfway_between_one_fourth_and_one_sixth_l120_120175


namespace jack_last_10_shots_made_l120_120652

theorem jack_last_10_shots_made (initial_shots : ℕ) (initial_percentage : ℚ)
  (additional_shots : ℕ) (new_percentage : ℚ)
  (initial_successful_shots : initial_shots * initial_percentage = 18)
  (total_shots : initial_shots + additional_shots = 40)
  (total_successful_shots : (initial_shots + additional_shots) * new_percentage = 25) :
  ∃ x : ℕ, x = 7 := by
sorry

end jack_last_10_shots_made_l120_120652


namespace earnings_correct_l120_120926

def price_8inch : ℝ := 5
def price_12inch : ℝ := 2.5 * price_8inch
def price_16inch : ℝ := 3 * price_8inch
def price_20inch : ℝ := 4 * price_8inch
def price_24inch : ℝ := 5.5 * price_8inch

noncomputable def earnings_monday : ℝ :=
  3 * price_8inch + 2 * price_12inch + 1 * price_16inch + 2 * price_20inch + 1 * price_24inch

noncomputable def earnings_tuesday : ℝ :=
  5 * price_8inch + 1 * price_12inch + 4 * price_16inch + 2 * price_24inch

noncomputable def earnings_wednesday : ℝ :=
  4 * price_8inch + 3 * price_12inch + 3 * price_16inch + 1 * price_20inch

noncomputable def earnings_thursday : ℝ :=
  2 * price_8inch + 2 * price_12inch + 2 * price_16inch + 1 * price_20inch + 3 * price_24inch

noncomputable def earnings_friday : ℝ :=
  6 * price_8inch + 4 * price_12inch + 2 * price_16inch + 2 * price_20inch

noncomputable def earnings_saturday : ℝ :=
  1 * price_8inch + 3 * price_12inch + 3 * price_16inch + 4 * price_20inch + 2 * price_24inch

noncomputable def earnings_sunday : ℝ :=
  3 * price_8inch + 2 * price_12inch + 4 * price_16inch + 3 * price_20inch + 1 * price_24inch

noncomputable def total_earnings : ℝ :=
  earnings_monday + earnings_tuesday + earnings_wednesday + earnings_thursday + earnings_friday + earnings_saturday + earnings_sunday

theorem earnings_correct : total_earnings = 1025 := by
  -- proof goes here
  sorry

end earnings_correct_l120_120926


namespace sufficient_but_not_necessary_condition_l120_120467

theorem sufficient_but_not_necessary_condition
  (a b : ℝ) : (a - b) * a^2 < 0 → a < b :=
sorry

end sufficient_but_not_necessary_condition_l120_120467


namespace books_sold_in_january_l120_120555

theorem books_sold_in_january (J : ℕ) 
  (h_avg : (J + 16 + 17) / 3 = 16) : J = 15 :=
sorry

end books_sold_in_january_l120_120555


namespace combined_selling_price_l120_120087

theorem combined_selling_price (C_c : ℕ) (C_s : ℕ) (C_m : ℕ) (L_c L_s L_m : ℕ)
  (hc : C_c = 1600)
  (hs : C_s = 12000)
  (hm : C_m = 45000)
  (hlc : L_c = 15)
  (hls : L_s = 10)
  (hlm : L_m = 5) :
  85 * C_c / 100 + 90 * C_s / 100 + 95 * C_m / 100 = 54910 := by
  sorry

end combined_selling_price_l120_120087


namespace functional_equation_holds_l120_120166

def f (p q : ℕ) : ℝ :=
  if p = 0 ∨ q = 0 then 0 else (p * q : ℝ)

theorem functional_equation_holds (p q : ℕ) : 
  f p q = 
    if p = 0 ∨ q = 0 then 0 
    else 1 + (1 / 2) * f (p + 1) (q - 1) + (1 / 2) * f (p - 1) (q + 1) :=
  by 
    sorry

end functional_equation_holds_l120_120166


namespace angle_supplement_complement_l120_120142

theorem angle_supplement_complement (x : ℝ) 
  (hsupp : 180 - x = 4 * (90 - x)) : 
  x = 60 :=
by
  sorry

end angle_supplement_complement_l120_120142


namespace discount_rate_l120_120345

theorem discount_rate (cost_price marked_price desired_profit_margin selling_price : ℝ)
  (h1 : cost_price = 160)
  (h2 : marked_price = 240)
  (h3 : desired_profit_margin = 0.2)
  (h4 : selling_price = cost_price * (1 + desired_profit_margin)) :
  marked_price * (1 - ((marked_price - selling_price) / marked_price)) = selling_price :=
by
  sorry

end discount_rate_l120_120345


namespace max_marks_set_for_test_l120_120873

-- Define the conditions according to the problem statement
def passing_percentage : ℝ := 0.70
def student_marks : ℝ := 120
def marks_needed_to_pass : ℝ := 150
def passing_threshold (M : ℝ) : ℝ := passing_percentage * M

-- The maximum marks set for the test
theorem max_marks_set_for_test (M : ℝ) : M = 386 :=
by
  -- Given the conditions
  have h : passing_threshold M = student_marks + marks_needed_to_pass := sorry
  -- Solving for M
  sorry

end max_marks_set_for_test_l120_120873


namespace sufficient_but_not_necessary_condition_l120_120612

theorem sufficient_but_not_necessary_condition (x : ℝ) :
  (x > 1 → x^2 + x - 2 > 0) ∧ (∃ y, y < -2 ∧ y^2 + y - 2 > 0) :=
by
  sorry

end sufficient_but_not_necessary_condition_l120_120612


namespace problem_equiv_math_problem_l120_120442
-- Lean Statement for the proof problem

variable {x y z : ℝ}

theorem problem_equiv_math_problem (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) 
  (eq1 : x^2 + x * y + y^2 / 3 = 25) 
  (eq2 : y^2 / 3 + z^2 = 9) 
  (eq3 : z^2 + z * x + x^2 = 16) :
  x * y + 2 * y * z + 3 * z * x = 24 * Real.sqrt 3 :=
by
  sorry

end problem_equiv_math_problem_l120_120442


namespace extreme_values_a_4_find_a_minimum_minus_5_l120_120103

noncomputable def f (x a : ℝ) : ℝ := 2 * x^2 - a * x + 5

theorem extreme_values_a_4 :
  (∀ x, x ∈ Set.Icc (-1:ℝ) 2 -> f x 4 ≤ 11) ∧ (∃ x, x ∈ Set.Icc (-1:ℝ) 2 ∧ f x 4 = 11) ∧
  (∀ x, x ∈ Set.Icc (-1:ℝ) 2 -> f x 4 ≥ 3) ∧ (∃ x, x ∈ Set.Icc (-1:ℝ) 2 ∧ f x 4 = 3) :=
  sorry

theorem find_a_minimum_minus_5 :
  ∀ (a : ℝ), (∃ x, x ∈ Set.Icc (-1:ℝ) 2 ∧ f x a = -5) -> (a = -12 ∨ a = 9) :=
  sorry

end extreme_values_a_4_find_a_minimum_minus_5_l120_120103


namespace even_function_derivative_at_zero_l120_120777

-- Define an even function f and its differentiability at x = 0
def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x
def differentiable_at_zero (f : ℝ → ℝ) : Prop := DifferentiableAt ℝ f 0

-- The theorem to prove that f'(0) = 0
theorem even_function_derivative_at_zero
  (f : ℝ → ℝ)
  (hf_even : is_even_function f)
  (hf_diff : differentiable_at_zero f) :
  deriv f 0 = 0 := 
sorry

end even_function_derivative_at_zero_l120_120777


namespace base7_number_divisibility_l120_120309

theorem base7_number_divisibility (x : ℕ) (h : 0 ≤ x ∧ x ≤ 6) :
  (5 * 343 + 2 * 49 + x * 7 + 4) % 29 = 0 ↔ x = 6 := 
by
  sorry

end base7_number_divisibility_l120_120309


namespace kevin_total_cost_l120_120784

theorem kevin_total_cost :
  let muffin_cost := 0.75
  let juice_cost := 1.45
  let total_muffins := 3
  let cost_muffins := total_muffins * muffin_cost
  let total_cost := cost_muffins + juice_cost
  total_cost = 3.70 :=
by
  sorry

end kevin_total_cost_l120_120784


namespace problem_statement_l120_120190

variables {R : Type*} [LinearOrderedField R]

theorem problem_statement (a b c : R) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : c ≠ a)
  (h : (b - a) ^ 2 - 4 * (b - c) * (c - a) = 0) : (b - c) / (c - a) = -1 :=
sorry

end problem_statement_l120_120190


namespace simplify_correct_l120_120172

def simplify_polynomial (x : Real) : Real :=
  (12 * x^10 + 6 * x^9 + 3 * x^8) + (2 * x^11 + x^10 + 4 * x^9 + x^7 + 4 * x^4 + 7 * x + 9)

theorem simplify_correct (x : Real) :
  simplify_polynomial x = 2 * x^11 + 13 * x^10 + 10 * x^9 + 3 * x^8 + x^7 + 4 * x^4 + 7 * x + 9 :=
by
  sorry

end simplify_correct_l120_120172


namespace sufficient_but_not_necessary_condition_l120_120128

noncomputable def is_increasing_function (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x < f y

theorem sufficient_but_not_necessary_condition (a : ℝ) :
  (∀ x y : ℝ, x < y → (a - 1) * (a ^ x) < (a - 1) * (a ^ y) → a > 1) ∧
  (¬ (∀ c : ℝ, is_increasing_function (λ x => (c - 1) * (c ^ x)) → c > 1)) :=
sorry

end sufficient_but_not_necessary_condition_l120_120128


namespace original_soldiers_eq_136_l120_120209

-- Conditions
def original_soldiers (n : ℕ) : ℕ := 8 * n
def after_adding_120 (n : ℕ) : ℕ := original_soldiers n + 120
def after_removing_120 (n : ℕ) : ℕ := original_soldiers n - 120

-- Given that both after_adding_120 n and after_removing_120 n are perfect squares.
def is_square (m : ℕ) : Prop := ∃ k : ℕ, k * k = m

-- Theorem statement
theorem original_soldiers_eq_136 : ∃ n : ℕ, original_soldiers n = 136 ∧ 
                                   is_square (after_adding_120 n) ∧ 
                                   is_square (after_removing_120 n) :=
sorry

end original_soldiers_eq_136_l120_120209


namespace nth_equation_l120_120000

theorem nth_equation (n : ℕ) (h : 0 < n) : (- (n : ℤ)) * (n : ℝ) / (n + 1) = - (n : ℤ) + (n : ℝ) / (n + 1) :=
sorry

end nth_equation_l120_120000


namespace cost_price_of_cloths_l120_120055

-- Definitions based on conditions
def SP_A := 8500 / 85
def Profit_A := 15
def CP_A := SP_A - Profit_A

def SP_B := 10200 / 120
def Profit_B := 12
def CP_B := SP_B - Profit_B

def SP_C := 4200 / 60
def Profit_C := 10
def CP_C := SP_C - Profit_C

-- Theorem to prove the cost prices
theorem cost_price_of_cloths :
    CP_A = 85 ∧
    CP_B = 73 ∧
    CP_C = 60 :=
by
    sorry

end cost_price_of_cloths_l120_120055


namespace geometric_sequence_new_product_l120_120375

theorem geometric_sequence_new_product 
  (a r : ℝ) (n : ℕ) (h_even : n % 2 = 0)
  (P S S' : ℝ)
  (hP : P = a^n * r^(n * (n-1) / 2))
  (hS : S = a * (1 - r^n) / (1 - r))
  (hS' : S' = (1 - r^n) / (a * (1 - r))) :
  (2^n * a^n * r^(n * (n-1) / 2)) = (S * S')^(n / 2) :=
sorry

end geometric_sequence_new_product_l120_120375


namespace simplify_expression_l120_120607

variable (a b : ℚ)

theorem simplify_expression (h1 : b ≠ 1/2) (h2 : b ≠ 1) :
  (2 * a + 1) / (1 - b / (2 * b - 1)) = (2 * a + 1) * (2 * b - 1) / (b - 1) :=
by 
  sorry

end simplify_expression_l120_120607


namespace zachary_pushups_l120_120212

variable {P : ℕ}
variable {C : ℕ}

theorem zachary_pushups :
  C = 58 → C = P + 12 → P = 46 :=
by 
  intros hC1 hC2
  rw [hC2] at hC1
  linarith

end zachary_pushups_l120_120212


namespace inscribed_sphere_radius_in_regular_octahedron_l120_120257

theorem inscribed_sphere_radius_in_regular_octahedron (a : ℝ) (r : ℝ) 
  (h1 : a = 6)
  (h2 : let V := 72 * Real.sqrt 2; V = (1 / 3) * ((8 * (3 * Real.sqrt 3)) * r)) : 
  r = Real.sqrt 6 :=
by
  sorry

end inscribed_sphere_radius_in_regular_octahedron_l120_120257


namespace one_greater_one_smaller_l120_120743

theorem one_greater_one_smaller (a b : ℝ) (h : ( (1 + a * b) / (a + b) )^2 < 1) :
  (a > 1 ∧ -1 < b ∧ b < 1) ∨ (b > 1 ∧ -1 < a ∧ a < 1) ∨ (a < -1 ∧ -1 < b ∧ b < 1) ∨ (b < -1 ∧ -1 < a ∧ a < 1) :=
by
  sorry

end one_greater_one_smaller_l120_120743


namespace day_of_week_proof_l120_120462

/-- 
January 1, 1978, is a Sunday in the Gregorian calendar.
What day of the week is January 1, 2000, in the Gregorian calendar?
-/
def day_of_week_2000 := "Saturday"

theorem day_of_week_proof :
  let initial_year := 1978
  let target_year := 2000
  let initial_weekday := "Sunday"
  let years_between := target_year - initial_year -- 22 years
  let normal_days := years_between * 365 -- Normal days in these years
  let leap_years := 5 -- Number of leap years in the range
  let total_days := normal_days + leap_years -- Total days considering leap years
  let remainder_days := total_days % 7 -- days modulo 7
  initial_weekday = "Sunday" → remainder_days = 6 → 
  day_of_week_2000 = "Saturday" :=
by
  sorry

end day_of_week_proof_l120_120462


namespace roundTripAverageSpeed_l120_120180

noncomputable def averageSpeed (distAB distBC speedAB speedBC speedCB totalTime : ℝ) : ℝ :=
  let timeAB := distAB / speedAB
  let timeBC := distBC / speedBC
  let timeCB := distBC / speedCB
  let timeBA := totalTime - (timeAB + timeBC + timeCB)
  let totalDistance := 2 * (distAB + distBC)
  totalDistance / totalTime

theorem roundTripAverageSpeed :
  averageSpeed 150 230 80 88 100 9 = 84.44 :=
by
  -- The actual proof will go here, which is not required for this task.
  sorry

end roundTripAverageSpeed_l120_120180


namespace eq_has_exactly_one_real_root_l120_120015

theorem eq_has_exactly_one_real_root : ∀ x : ℝ, 2007 * x^3 + 2006 * x^2 + 2005 * x = 0 ↔ x = 0 :=
by
sorry

end eq_has_exactly_one_real_root_l120_120015


namespace find_second_remainder_l120_120430

theorem find_second_remainder (k m n r : ℕ) 
  (h1 : n = 12 * k + 56) 
  (h2 : n = 34 * m + r) 
  (h3 : (22 + r) % 12 = 10) : 
  r = 10 :=
sorry

end find_second_remainder_l120_120430


namespace x_can_be_any_sign_l120_120350

theorem x_can_be_any_sign
  (x y z w : ℤ)
  (h1 : (y - 1) * (w - 2) ≠ 0)
  (h2 : (x + 2)/(y - 1) < - (z + 3)/(w - 2)) :
  ∃ x : ℤ, True :=
by
  sorry

end x_can_be_any_sign_l120_120350


namespace truncated_pyramid_properties_l120_120481

noncomputable def truncatedPyramidSurfaceArea
  (a b c : ℝ) (theta m : ℝ) : ℝ :=
sorry -- Definition of the surface area function

noncomputable def truncatedPyramidVolume
  (a b c : ℝ) (theta m : ℝ) : ℝ :=
sorry -- Definition of the volume function

theorem truncated_pyramid_properties
  (a b c : ℝ) (theta m : ℝ)
  (h₀ : a = 148) 
  (h₁ : b = 156) 
  (h₂ : c = 208) 
  (h₃ : theta = 112.62) 
  (h₄ : m = 27) :
  (truncatedPyramidSurfaceArea a b c theta m = 74352) ∧
  (truncatedPyramidVolume a b c theta m = 395280) :=
by
  sorry -- The actual proof will go here

end truncated_pyramid_properties_l120_120481


namespace jeff_bought_from_chad_l120_120513

/-
  Eric has 4 ninja throwing stars.
  Chad has twice as many ninja throwing stars as Eric.
  Jeff now has 6 ninja throwing stars.
  Together, they have 16 ninja throwing stars.
  How many ninja throwing stars did Jeff buy from Chad?
-/

def eric_stars : ℕ := 4
def chad_stars : ℕ := 2 * eric_stars
def jeff_stars : ℕ := 6
def total_stars : ℕ := 16

theorem jeff_bought_from_chad (bought : ℕ) :
  chad_stars - bought + jeff_stars + eric_stars = total_stars → bought = 2 :=
by
  sorry

end jeff_bought_from_chad_l120_120513


namespace find_salary_l120_120622

theorem find_salary (x y : ℝ) (h1 : x + y = 2000) (h2 : 0.05 * x = 0.15 * y) : x = 1500 :=
sorry

end find_salary_l120_120622


namespace find_ratio_l120_120344

-- Given that the tangent of angle θ (inclination angle) is -2
def tan_theta (θ : Real) : Prop := Real.tan θ = -2

theorem find_ratio (θ : Real) (h : tan_theta θ) :
  (Real.sin θ + Real.cos θ) / (Real.sin θ - Real.cos θ) = 1 / 3 := by
  sorry

end find_ratio_l120_120344


namespace integer_solutions_range_l120_120596

theorem integer_solutions_range (a : ℝ) :
  (∀ x : ℤ, x^2 - x + a - a^2 < 0 → x + 2 * a > 1) ↔ 1 < a ∧ a ≤ 2 := sorry

end integer_solutions_range_l120_120596


namespace intersection_of_M_and_N_l120_120304

-- Define sets M and N
def M : Set ℤ := {x | -2 ≤ x ∧ x ≤ 2}
def N : Set ℤ := {0, 1, 2}

-- The theorem to be proven: M ∩ N = {0, 1, 2}
theorem intersection_of_M_and_N : M ∩ N = {0, 1, 2} :=
by
  sorry

end intersection_of_M_and_N_l120_120304


namespace eval_expression_l120_120496

theorem eval_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end eval_expression_l120_120496


namespace find_y_l120_120120

theorem find_y (x y : ℝ) (h1 : (100 + 200 + 300 + x) / 4 = 250) (h2 : (300 + 150 + 100 + x + y) / 5 = 200) : y = 50 :=
by
  sorry

end find_y_l120_120120


namespace simplify_expression_l120_120324

variable (y : ℝ)

theorem simplify_expression :
  4 * y^3 + 8 * y + 6 - (3 - 4 * y^3 - 8 * y) = 8 * y^3 + 16 * y + 3 :=
by
  sorry

end simplify_expression_l120_120324


namespace probability_red_ball_10th_draw_l120_120960

-- Definitions for conditions in the problem
def total_balls : ℕ := 10
def red_balls : ℕ := 2

-- Probability calculation function
def probability_of_red_ball (total : ℕ) (red : ℕ) : ℚ :=
  red / total

-- Theorem statement: Given the conditions, the probability of drawing a red ball on the 10th attempt is 1/5
theorem probability_red_ball_10th_draw :
  probability_of_red_ball total_balls red_balls = 1 / 5 :=
by
  sorry

end probability_red_ball_10th_draw_l120_120960


namespace determine_A_l120_120734

noncomputable def is_single_digit (n : ℕ) : Prop := n < 10

theorem determine_A (A B C : ℕ) (hABC : 3 * (100 * A + 10 * B + C) = 888)
  (hA_single_digit : is_single_digit A) (hB_single_digit : is_single_digit B) (hC_single_digit : is_single_digit C)
  (h_different : A ≠ B ∧ B ≠ C ∧ A ≠ C) : A = 2 := 
  sorry

end determine_A_l120_120734


namespace correct_option_is_B_l120_120924

-- Definitions and conditions based on the problem
def is_monomial (t : String) : Prop :=
  t = "1"

def coefficient (expr : String) : Int :=
  if expr = "x" then 1
  else if expr = "-3x" then -3
  else 0

def degree (term : String) : Int :=
  if term = "5x^2y" then 3
  else 0

-- Proof statement
theorem correct_option_is_B : 
  is_monomial "1" ∧ ¬ (coefficient "x" = 0) ∧ ¬ (coefficient "-3x" = 3) ∧ ¬ (degree "5x^2y" = 2) := 
by
  -- Proof steps will go here
  sorry

end correct_option_is_B_l120_120924


namespace abc_eq_1_l120_120437

theorem abc_eq_1 (a b c : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0)
(h4 : a ≠ b) (h5 : b ≠ c) (h6 : c ≠ a)
(h7 : a + 1 / b^2 = b + 1 / c^2) (h8 : b + 1 / c^2 = c + 1 / a^2) :
  |a * b * c| = 1 :=
sorry

end abc_eq_1_l120_120437


namespace fourth_hexagon_dots_l120_120289

def dots_in_hexagon (n : ℕ) : ℕ :=
  if n = 0 then 1
  else 1 + (12 * (n * (n + 1) / 2))

theorem fourth_hexagon_dots : dots_in_hexagon 4 = 85 :=
by
  unfold dots_in_hexagon
  norm_num
  sorry

end fourth_hexagon_dots_l120_120289


namespace total_cost_correct_l120_120642

def football_cost : ℝ := 5.71
def marbles_cost : ℝ := 6.59
def total_cost : ℝ := 12.30

theorem total_cost_correct : football_cost + marbles_cost = total_cost := 
by
  sorry

end total_cost_correct_l120_120642


namespace find_P_l120_120925

theorem find_P (P Q R S : ℕ) (h1: P ≠ Q) (h2: R ≠ S) (h3: P * Q = 72) (h4: R * S = 72) (h5: P - Q = R + S) :
  P = 18 := 
  sorry

end find_P_l120_120925


namespace PhenotypicallyNormalDaughterProbability_l120_120602

-- Definitions based on conditions
def HemophiliaSexLinkedRecessive := true
def PhenylketonuriaAutosomalRecessive := true
def CouplePhenotypicallyNormal := true
def SonWithBothHemophiliaPhenylketonuria := true

-- Definition of the problem
theorem PhenotypicallyNormalDaughterProbability
  (HemophiliaSexLinkedRecessive : Prop)
  (PhenylketonuriaAutosomalRecessive : Prop)
  (CouplePhenotypicallyNormal : Prop)
  (SonWithBothHemophiliaPhenylketonuria : Prop) :
  -- The correct answer from the solution
  ∃ p : ℚ, p = 3/4 :=
  sorry

end PhenotypicallyNormalDaughterProbability_l120_120602


namespace max_omega_l120_120020

open Real

-- Define the function f(x) = sin(ωx + φ)
noncomputable def f (ω φ x : ℝ) := sin (ω * x + φ)

-- ω > 0 and |φ| ≤ π / 2
def condition_omega_pos (ω : ℝ) := ω > 0
def condition_phi_bound (φ : ℝ) := abs φ ≤ π / 2

-- x = -π/4 is a zero of f(x)
def condition_zero (ω φ : ℝ) := f ω φ (-π/4) = 0

-- x = π/4 is the axis of symmetry for the graph of y = f(x)
def condition_symmetry (ω φ : ℝ) := 
  ∀ x : ℝ, f ω φ (π/4 - x) = f ω φ (π/4 + x)

-- f(x) is monotonic in the interval (π/18, 5π/36)
def condition_monotonic (ω φ : ℝ) := 
  ∀ x₁ x₂ : ℝ, π/18 < x₁ ∧ x₁ < x₂ ∧ x₂ < 5 * π / 36 
  → f ω φ x₁ ≤ f ω φ x₂

-- Prove that the maximum value of ω satisfying all the conditions is 9
theorem max_omega (ω : ℝ) (φ : ℝ)
  (h1 : condition_omega_pos ω)
  (h2 : condition_phi_bound φ)
  (h3 : condition_zero ω φ)
  (h4 : condition_symmetry ω φ)
  (h5 : condition_monotonic ω φ) :
  ω ≤ 9 :=
sorry

end max_omega_l120_120020


namespace speed_man_l120_120817

noncomputable def speedOfMan : ℝ := 
  let d := 437.535 / 1000  -- distance in kilometers
  let t := 25 / 3600      -- time in hours
  d / t                    -- speed in kilometers per hour

theorem speed_man : speedOfMan = 63 := by
  sorry

end speed_man_l120_120817


namespace burger_meal_cost_l120_120940

theorem burger_meal_cost 
  (x : ℝ) 
  (h : 5 * (x + 1) = 35) : 
  x = 6 := 
sorry

end burger_meal_cost_l120_120940


namespace complement_M_l120_120527

def U : Set ℝ := Set.univ
def M : Set ℝ := { x | x^2 - 4 ≤ 0 }

theorem complement_M (U M : Set ℝ) : (U \ M) = { x : ℝ | x < -2 ∨ x > 2 } :=
by
  sorry

end complement_M_l120_120527


namespace find_x_l120_120086

theorem find_x (x : ℝ) (h : (3 * x) / 7 = 15) : x = 35 :=
sorry

end find_x_l120_120086


namespace simplify_and_evaluate_l120_120202

theorem simplify_and_evaluate (m : ℝ) (h : m = Real.sqrt 2) :
  (1 + 1 / (m - 2)) / ((m^2 - m) / (m - 2)) = Real.sqrt 2 / 2 :=
by
  sorry

end simplify_and_evaluate_l120_120202


namespace convex_2k_vertices_l120_120436

theorem convex_2k_vertices (k : ℕ) (h1 : 2 ≤ k) (h2 : k ≤ 50)
    (P : Finset (EuclideanSpace ℝ (Fin 2)))
    (hP : P.card = 100) (M : Finset (EuclideanSpace ℝ (Fin 2)))
    (hM : M.card = k) : 
  ∃ V : Finset (EuclideanSpace ℝ (Fin 2)), V.card = 2 * k ∧ ∀ m ∈ M, m ∈ convexHull ℝ V :=
by
  sorry

end convex_2k_vertices_l120_120436


namespace digit_6_count_1_to_700_l120_120790

theorem digit_6_count_1_to_700 :
  let countNumbersWithDigit6 := 700 - (7 * 9 * 9)
  countNumbersWithDigit6 = 133 := 
by
  let countNumbersWithDigit6 := 700 - (7 * 9 * 9)
  show countNumbersWithDigit6 = 133
  sorry

end digit_6_count_1_to_700_l120_120790


namespace Ryanne_is_7_years_older_than_Hezekiah_l120_120297

theorem Ryanne_is_7_years_older_than_Hezekiah
  (H : ℕ) (R : ℕ)
  (h1 : H = 4)
  (h2 : R + H = 15) :
  R - H = 7 := by
  sorry

end Ryanne_is_7_years_older_than_Hezekiah_l120_120297


namespace parabola_tangent_perpendicular_m_eq_one_parabola_min_MF_NF_l120_120636

open Real

theorem parabola_tangent_perpendicular_m_eq_one (k : ℝ) (hk : k > 0) :
  (∃ x₁ x₂ y₁ y₂ : ℝ, (x₁^2 = 4 * y₁) ∧ (x₂^2 = 4 * y₂) ∧ (y₁ = k * x₁ + m) ∧ (y₂ = k * x₂ + m) ∧ ((x₁ / 2) * (x₂ / 2) = -1)) → m = 1 :=
sorry

theorem parabola_min_MF_NF (k : ℝ) (hk : k > 0) :
  (m = 2) → 
  (∃ x₁ x₂ y₁ y₂ : ℝ, (x₁^2 = 4 * y₁) ∧ (x₂^2 = 4 * y₂) ∧ (y₁ = k * x₁ + 2) ∧ (y₂ = k * x₂ + 2) ∧ |(y₁ + 1) * (y₂ + 1)| ≥ 9) :=
sorry

end parabola_tangent_perpendicular_m_eq_one_parabola_min_MF_NF_l120_120636


namespace unit_digit_of_six_consecutive_product_is_zero_l120_120649

theorem unit_digit_of_six_consecutive_product_is_zero (n : ℕ) (h : n > 0) :
  (n * (n + 1) * (n + 2) * (n + 3) * (n + 4) * (n + 5)) % 10 = 0 := 
by sorry

end unit_digit_of_six_consecutive_product_is_zero_l120_120649


namespace nicky_catchup_time_l120_120097

-- Definitions related to the problem
def head_start : ℕ := 12
def speed_cristina : ℕ := 5
def speed_nicky : ℕ := 3
def time_to_catchup : ℕ := 36
def nicky_runtime_before_catchup : ℕ := head_start + time_to_catchup

-- Theorem to prove the correct runtime for Nicky before Cristina catches up
theorem nicky_catchup_time : nicky_runtime_before_catchup = 48 := by
  sorry

end nicky_catchup_time_l120_120097


namespace max_showers_l120_120332

open Nat

variable (household water_limit water_for_drinking_and_cooking water_per_shower pool_length pool_width pool_height water_per_cubic_foot pool_leakage_rate days_in_july : ℕ)

def volume_of_pool (length width height: ℕ): ℕ :=
  length * width * height

def water_usage (drinking cooking pool leakage: ℕ): ℕ :=
  drinking + cooking + pool + leakage

theorem max_showers (h1: water_limit = 1000)
                    (h2: water_for_drinking_and_cooking = 100)
                    (h3: water_per_shower = 20)
                    (h4: pool_length = 10)
                    (h5: pool_width = 10)
                    (h6: pool_height = 6)
                    (h7: water_per_cubic_foot = 1)
                    (h8: pool_leakage_rate = 5)
                    (h9: days_in_july = 31) : 
  (water_limit - water_usage water_for_drinking_and_cooking
                                  (volume_of_pool pool_length pool_width pool_height) 
                                  ((pool_leakage_rate * days_in_july))) / water_per_shower = 7 := by
  sorry

end max_showers_l120_120332


namespace recommended_cooking_time_is_5_minutes_l120_120098

-- Define the conditions
def time_cooked := 45 -- seconds
def time_remaining := 255 -- seconds

-- Define the total cooking time in seconds
def total_time_seconds := time_cooked + time_remaining

-- Define the conversion from seconds to minutes
def to_minutes (seconds : ℕ) : ℕ := seconds / 60

-- The main theorem to prove
theorem recommended_cooking_time_is_5_minutes :
  to_minutes total_time_seconds = 5 :=
by
  sorry

end recommended_cooking_time_is_5_minutes_l120_120098


namespace probability_digit_9_in_3_over_11_is_zero_l120_120588

-- Define the repeating block of the fraction 3/11
def repeating_block_3_over_11 : List ℕ := [2, 7]

-- Define the function to count the occurrences of a digit in a list
def count_occurrences (digit : ℕ) (lst : List ℕ) : ℕ :=
  lst.count digit

-- Define the probability function
def probability_digit_9_in_3_over_11 : ℚ :=
  (count_occurrences 9 repeating_block_3_over_11) / repeating_block_3_over_11.length

-- Theorem statement
theorem probability_digit_9_in_3_over_11_is_zero : 
  probability_digit_9_in_3_over_11 = 0 := 
by 
  sorry

end probability_digit_9_in_3_over_11_is_zero_l120_120588


namespace correct_calculation_l120_120368

theorem correct_calculation (x : ℝ) : (2 * x^5) / (-x)^3 = -2 * x^2 :=
by sorry

end correct_calculation_l120_120368


namespace arithmetic_sequence_5_7_9_l120_120874

variable {a : ℕ → ℕ}

theorem arithmetic_sequence_5_7_9 (h : 13 * (a 7) = 39) : a 5 + a 7 + a 9 = 9 := 
sorry

end arithmetic_sequence_5_7_9_l120_120874


namespace range_of_a_l120_120551

theorem range_of_a (a : ℝ) (h : a ≤ 1) :
  (∃! n : ℕ, n = (2 - a) - a + 1) → -1 < a ∧ a ≤ 0 :=
by 
  sorry

end range_of_a_l120_120551


namespace inequality_example_l120_120177

variable (f : ℝ → ℝ)
variable (f' : ℝ → ℝ)
variable (der : ∀ x, deriv f x = f' x)

theorem inequality_example (h : ∀ x : ℝ, f x > f' x) :
  2023 * f (Real.log 2022) > 2022 * f (Real.log 2023)
:= sorry

end inequality_example_l120_120177


namespace hyperbola_range_m_l120_120634

theorem hyperbola_range_m (m : ℝ) : 
  (∃ x y : ℝ, (m + 2 > 0 ∧ m - 2 < 0) ∧ (x^2 / (m + 2) + y^2 / (m - 2) = 1)) ↔ (-2 < m ∧ m < 2) :=
by
  sorry

end hyperbola_range_m_l120_120634


namespace tangent_lengths_identity_l120_120899

theorem tangent_lengths_identity
  (a b c BC AC AB : ℝ)
  (sqrt_a sqrt_b sqrt_c : ℝ)
  (h1 : sqrt_a^2 = a)
  (h2 : sqrt_b^2 = b)
  (h3 : sqrt_c^2 = c) :
  a * BC + c * AB - b * AC = BC * AC * AB :=
sorry

end tangent_lengths_identity_l120_120899


namespace first_term_arithmetic_sequence_l120_120630

def T_n (a d : ℚ) (n : ℕ) := n * (2 * a + (n - 1) * d) / 2

theorem first_term_arithmetic_sequence (a : ℚ)
  (h_const_ratio : ∀ (n : ℕ), n > 0 → 
    (T_n a 5 (4 * n)) / (T_n a 5 n) = (T_n a 5 4 / T_n a 5 1)) : 
  a = -5/2 :=
by 
  sorry

end first_term_arithmetic_sequence_l120_120630


namespace bottom_row_bricks_l120_120054

theorem bottom_row_bricks (x : ℕ) 
    (h : x + (x - 1) + (x - 2) + (x - 3) + (x - 4) = 200) : x = 42 :=
sorry

end bottom_row_bricks_l120_120054


namespace smoking_lung_cancer_problem_l120_120432

-- Defining the confidence relationship
def smoking_related_to_lung_cancer (confidence: ℝ) := confidence > 0.99

-- Statement 4: Among 100 smokers, it is possible that not a single person has lung cancer.
def statement_4 (N: ℕ) (p: ℝ) := N = 100 ∧ 0 ≤ p ∧ p ≤ 1 ∧ p ^ 100 > 0

-- The main theorem statement in Lean 4
theorem smoking_lung_cancer_problem (confidence: ℝ) (N: ℕ) (p: ℝ) 
  (h1: smoking_related_to_lung_cancer confidence): 
  statement_4 N p :=
by
  sorry -- Proof goes here

end smoking_lung_cancer_problem_l120_120432


namespace bagel_spending_l120_120230

variable (B D : ℝ)

theorem bagel_spending (h1 : B - D = 12.50) (h2 : D = B * 0.75) : B + D = 87.50 := 
sorry

end bagel_spending_l120_120230


namespace sum_infinite_series_eq_l120_120842

theorem sum_infinite_series_eq : 
  (∑' n : ℕ, if n > 0 then ((3 * n - 2 : ℝ) / (n * (n + 1) * (n + 3))) else 0) = (7 / 12) :=
by
  sorry

end sum_infinite_series_eq_l120_120842


namespace value_of_a_l120_120045

def f (x : ℝ) : ℝ := x^2 + 2
def g (x : ℝ) : ℝ := x^2 + 2

theorem value_of_a (a : ℝ) (ha : a > 1) (h : f (g a) = 12) : 
  a = Real.sqrt (Real.sqrt 10 - 2) :=
by sorry

end value_of_a_l120_120045


namespace max_value_x_l120_120292

theorem max_value_x : ∃ x, x ^ 2 = 38 ∧ x = Real.sqrt 38 := by
  sorry

end max_value_x_l120_120292


namespace problem_solution_l120_120269

def count_valid_n : ℕ :=
  let count_mult_3 := (3000 / 3)
  let count_mult_6 := (3000 / 6)
  count_mult_3 - count_mult_6

theorem problem_solution : count_valid_n = 500 := 
sorry

end problem_solution_l120_120269


namespace count_two_digit_integers_l120_120487

def two_digit_integers_satisfying_condition : Nat :=
  let candidates := [(2, 9), (3, 8), (4, 7), (5, 6), (6, 5), (7, 4), (8, 3), (9, 2)]
  candidates.length

theorem count_two_digit_integers :
  two_digit_integers_satisfying_condition = 8 :=
by
  sorry

end count_two_digit_integers_l120_120487


namespace evaluate_expression_l120_120754

theorem evaluate_expression : 12 * ((1/3 : ℚ) + (1/4) + (1/6))⁻¹ = 16 := 
by 
  sorry

end evaluate_expression_l120_120754


namespace rented_movie_cost_l120_120401

def cost_of_tickets (c_ticket : ℝ) (n_tickets : ℕ) := c_ticket * n_tickets
def total_cost (cost_tickets cost_bought : ℝ) := cost_tickets + cost_bought
def remaining_cost (total_spent cost_so_far : ℝ) := total_spent - cost_so_far

theorem rented_movie_cost
  (c_ticket : ℝ)
  (n_tickets : ℕ)
  (c_bought : ℝ)
  (c_total : ℝ)
  (h1 : c_ticket = 10.62)
  (h2 : n_tickets = 2)
  (h3 : c_bought = 13.95)
  (h4 : c_total = 36.78) :
  remaining_cost c_total (total_cost (cost_of_tickets c_ticket n_tickets) c_bought) = 1.59 :=
by 
  sorry

end rented_movie_cost_l120_120401


namespace distinct_real_number_sum_and_square_sum_eq_l120_120510

theorem distinct_real_number_sum_and_square_sum_eq
  (a b c d : ℝ)
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h_sum : a + b + c + d = 3)
  (h_square_sum : a^2 + b^2 + c^2 + d^2 = 45) :
  (a^5 / (a - b) / (a - c) / (a - d)) + (b^5 / (b - a) / (b - c) / (b - d)) +
  (c^5 / (c - a) / (c - b) / (c - d)) + (d^5 / (d - a) / (d - b) / (d - c)) = -9 :=
by
  sorry

end distinct_real_number_sum_and_square_sum_eq_l120_120510


namespace zeros_of_f_l120_120565

-- Define the quadratic function
def f (x : ℝ) : ℝ := x^2 - 3 * x + 2

-- State the theorem about its roots
theorem zeros_of_f : ∃ x : ℝ, f x = 0 ↔ x = 1 ∨ x = 2 := by
  sorry

end zeros_of_f_l120_120565


namespace cost_price_per_meter_of_cloth_l120_120597

theorem cost_price_per_meter_of_cloth
  (meters : ℕ) (selling_price : ℕ) (profit_per_meter : ℕ) (total_profit : ℕ) (cost_price : ℕ)
  (meters_eq : meters = 80)
  (selling_price_eq : selling_price = 10000)
  (profit_per_meter_eq : profit_per_meter = 7)
  (total_profit_eq : total_profit = profit_per_meter * meters)
  (selling_price_calc : selling_price = cost_price + total_profit)
  (cost_price_calc : cost_price = selling_price - total_profit)
  : (selling_price - total_profit) / meters = 118 :=
by
  -- here we would provide the proof, but we skip it with sorry
  sorry

end cost_price_per_meter_of_cloth_l120_120597


namespace garden_length_80_l120_120811

-- Let the width of the garden be denoted by w and the length by l
-- Given conditions
def is_rectangular_garden (l w : ℝ) := l = 2 * w ∧ 2 * l + 2 * w = 240

-- We want to prove that the length of the garden is 80 yards
theorem garden_length_80 (w : ℝ) (h : is_rectangular_garden (2 * w) w) : 2 * w = 80 :=
by
  sorry

end garden_length_80_l120_120811


namespace initial_oranges_per_tree_l120_120949

theorem initial_oranges_per_tree (x : ℕ) (h1 : 8 * (5 * x - 2 * x) / 5 = 960) : x = 200 :=
sorry

end initial_oranges_per_tree_l120_120949


namespace probability_no_defective_pencils_l120_120040

theorem probability_no_defective_pencils : 
  let total_pencils := 9
  let defective_pencils := 2
  let chosen_pencils := 3
  let non_defective_pencils := total_pencils - defective_pencils
  let total_ways := Nat.choose total_pencils chosen_pencils
  let non_defective_ways := Nat.choose non_defective_pencils chosen_pencils
  let probability := non_defective_ways / total_ways
  probability = 5 / 12 := 
by
  sorry

end probability_no_defective_pencils_l120_120040


namespace range_of_m_l120_120761

theorem range_of_m (m : ℝ) : 
  (∀ x y : ℝ, (x^2)/(9 - m) + (y^2)/(m - 5) = 1 → 
  (∃ m, (7 < m ∧ m < 9))) := 
sorry

end range_of_m_l120_120761


namespace order_xyz_l120_120885

theorem order_xyz (x : ℝ) (h1 : 0.8 < x) (h2 : x < 0.9) :
  let y := x^x
  let z := x^(x^x)
  x < z ∧ z < y :=
by
  sorry

end order_xyz_l120_120885


namespace geometric_sequence_solution_l120_120266

variables (a : ℕ → ℝ) (q : ℝ)
-- Given conditions
def condition1 : Prop := abs (a 1) = 1
def condition2 : Prop := a 5 = -8 * a 2
def condition3 : Prop := a 5 > a 2
-- Proof statement
theorem geometric_sequence_solution :
  condition1 a → condition2 a → condition3 a → ∀ n, a n = (-2)^(n - 1) :=
sorry

end geometric_sequence_solution_l120_120266


namespace quadratic_has_at_most_two_solutions_l120_120656

theorem quadratic_has_at_most_two_solutions (a b c : ℝ) (h : a ≠ 0) :
  ¬(∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧
    a * x1^2 + b * x1 + c = 0 ∧ 
    a * x2^2 + b * x2 + c = 0 ∧ 
    a * x3^2 + b * x3 + c = 0) := 
by {
  sorry
}

end quadratic_has_at_most_two_solutions_l120_120656


namespace proof_problem_l120_120966

theorem proof_problem 
  (a1 a2 b2 : ℚ)
  (ha1 : a1 = -9 + (8/3))
  (ha2 : a2 = -9 + 2 * (8/3))
  (hb2 : b2 = -3) :
  b2 * (a1 + a2) = 30 :=
by
  sorry

end proof_problem_l120_120966


namespace grade_A_probability_l120_120851

theorem grade_A_probability
  (P_B : ℝ) (P_C : ℝ)
  (hB : P_B = 0.05)
  (hC : P_C = 0.03) :
  1 - P_B - P_C = 0.92 :=
by
  sorry

end grade_A_probability_l120_120851


namespace share_of_C_l120_120342

/-- Given the conditions:
  - Total investment is Rs. 120,000.
  - A's investment is Rs. 6,000 more than B's.
  - B's investment is Rs. 8,000 more than C's.
  - Profit distribution ratio among A, B, and C is 4:3:2.
  - Total profit is Rs. 50,000.
Prove that C's share of the profit is Rs. 11,111.11. -/
theorem share_of_C (total_investment : ℝ)
  (A_more_than_B : ℝ)
  (B_more_than_C : ℝ)
  (profit_distribution : ℝ)
  (total_profit : ℝ) :
  total_investment = 120000 →
  A_more_than_B = 6000 →
  B_more_than_C = 8000 →
  profit_distribution = 4 / 9 →
  total_profit = 50000 →
  ∃ (C_share : ℝ), C_share = 11111.11 :=
by
  sorry

end share_of_C_l120_120342


namespace find_A_l120_120223

theorem find_A (A a b : ℝ) (h1 : 3^a = A) (h2 : 5^b = A) (h3 : 1/a + 1/b = 2) : A = Real.sqrt 15 :=
by
  /- Proof omitted -/
  sorry

end find_A_l120_120223


namespace group_booking_cost_correct_l120_120073

-- Definitions based on the conditions of the problem
def weekday_rate_first_week : ℝ := 18.00
def weekend_rate_first_week : ℝ := 20.00
def weekday_rate_additional_weeks : ℝ := 11.00
def weekend_rate_additional_weeks : ℝ := 13.00
def security_deposit : ℝ := 50.00
def discount_rate : ℝ := 0.10
def group_size : ℝ := 5
def stay_duration : ℕ := 23

-- Computation of total cost
def total_cost (first_week_weekdays : ℕ) (first_week_weekends : ℕ) 
  (additional_week_weekdays : ℕ) (additional_week_weekends : ℕ) 
  (additional_days_weekdays : ℕ) : ℝ := 
  let cost_first_weekdays := first_week_weekdays * weekday_rate_first_week
  let cost_first_weekends := first_week_weekends * weekend_rate_first_week
  let cost_additional_weeks := 2 * (additional_week_weekdays * weekday_rate_additional_weeks + 
                                    additional_week_weekends * weekend_rate_additional_weeks)
  let cost_additional_days := additional_days_weekdays * weekday_rate_additional_weeks
  let total_before_deposit := cost_first_weekdays + cost_first_weekends + 
                              cost_additional_weeks + cost_additional_days
  let total_before_discount := total_before_deposit + security_deposit
  let total_discount := discount_rate * total_before_discount
  total_before_discount - total_discount

-- Proof setup
theorem group_booking_cost_correct :
  total_cost 5 2 5 2 2 = 327.60 :=
by
  -- Placeholder for the proof; steps not required for Lean statement
  sorry

end group_booking_cost_correct_l120_120073


namespace simplify_175_sub_57_sub_43_simplify_128_sub_64_sub_36_simplify_156_sub_49_sub_51_l120_120937

theorem simplify_175_sub_57_sub_43 : 175 - 57 - 43 = 75 :=
by
  sorry

theorem simplify_128_sub_64_sub_36 : 128 - 64 - 36 = 28 :=
by
  sorry

theorem simplify_156_sub_49_sub_51 : 156 - 49 - 51 = 56 :=
by
  sorry

end simplify_175_sub_57_sub_43_simplify_128_sub_64_sub_36_simplify_156_sub_49_sub_51_l120_120937


namespace prob_XYZ_wins_l120_120089

-- Define probabilities as given in the conditions
def P_X : ℚ := 1 / 4
def P_Y : ℚ := 1 / 8
def P_Z : ℚ := 1 / 12

-- Define the probability that one of X, Y, or Z wins, assuming events are mutually exclusive
def P_XYZ_wins : ℚ := P_X + P_Y + P_Z

theorem prob_XYZ_wins : P_XYZ_wins = 11 / 24 := by
  -- sorry is used to skip the proof
  sorry

end prob_XYZ_wins_l120_120089


namespace quadratic_inequality_solutions_l120_120132

theorem quadratic_inequality_solutions (k : ℝ) :
  (0 < k ∧ k < 16) ↔ ∃ x : ℝ, x^2 - 8*x + k < 0 :=
sorry

end quadratic_inequality_solutions_l120_120132


namespace nathalie_cake_fraction_l120_120255

theorem nathalie_cake_fraction
    (cake_weight : ℕ)
    (pierre_ate : ℕ)
    (double_what_nathalie_ate : pierre_ate = 2 * (pierre_ate / 2))
    (pierre_ate_correct : pierre_ate = 100) :
    (pierre_ate / 2) / cake_weight = 1 / 8 :=
by
  sorry

end nathalie_cake_fraction_l120_120255


namespace correct_definition_of_regression_independence_l120_120580

-- Definitions
def regression_analysis (X Y : Type) := ∃ r : X → Y, true -- Placeholder, ideal definition studies correlation
def independence_test (X Y : Type) := ∃ rel : X → Y → Prop, true -- Placeholder, ideal definition examines relationship

-- Theorem statement
theorem correct_definition_of_regression_independence (X Y : Type) :
  (∃ r : X → Y, true) ∧ (∃ rel : X → Y → Prop, true)
  → "Regression analysis studies the correlation between two variables, and independence tests examine whether there is some kind of relationship between two variables" = "C" :=
sorry

end correct_definition_of_regression_independence_l120_120580


namespace packs_of_chocolate_l120_120317

theorem packs_of_chocolate (t c k x : ℕ) (ht : t = 42) (hc : c = 4) (hk : k = 22) (hx : x = t - (c + k)) : x = 16 :=
by
  rw [ht, hc, hk] at hx
  simp at hx
  exact hx

end packs_of_chocolate_l120_120317


namespace sum_of_two_rel_prime_numbers_l120_120322

theorem sum_of_two_rel_prime_numbers (k : ℕ) : 
  (∃ a b : ℕ, a > 1 ∧ b > 1 ∧ Nat.gcd a b = 1 ∧ k = a + b) ↔ (k = 5 ∨ k ≥ 7) := sorry

end sum_of_two_rel_prime_numbers_l120_120322


namespace total_weight_full_l120_120619

theorem total_weight_full {x y p q : ℝ}
    (h1 : x + (3/4) * y = p)
    (h2 : x + (1/3) * y = q) :
    x + y = (8/5) * p - (3/5) * q :=
by
  sorry

end total_weight_full_l120_120619


namespace solve_inequality_l120_120146

theorem solve_inequality (x : ℝ) : (3 * x - 5) / 2 > 2 * x → x < -5 :=
by
  sorry

end solve_inequality_l120_120146


namespace number_of_ways_to_fold_cube_with_one_face_missing_l120_120797

-- Definitions:
-- The polygon is initially in the shape of a cross with 5 congruent squares.
-- One additional square can be attached to any of the 12 possible edge positions around this polygon.
-- Define what it means for the resulting figure to fold into a cube with one face missing.

-- Statement:
theorem number_of_ways_to_fold_cube_with_one_face_missing 
  (initial_squares : ℕ)
  (additional_positions : ℕ)
  (valid_folding_positions : ℕ) : 
  initial_squares = 5 ∧ additional_positions = 12 → valid_folding_positions = 8 :=
by
  sorry

end number_of_ways_to_fold_cube_with_one_face_missing_l120_120797


namespace quarters_given_by_mom_l120_120062

theorem quarters_given_by_mom :
  let dimes := 4
  let quarters := 4
  let nickels := 7
  let value_dimes := 0.10 * dimes
  let value_quarters := 0.25 * quarters
  let value_nickels := 0.05 * nickels
  let initial_total := value_dimes + value_quarters + value_nickels
  let final_total := 3.00
  let additional_amount := final_total - initial_total
  additional_amount / 0.25 = 5 :=
by
  sorry

end quarters_given_by_mom_l120_120062


namespace Carter_gave_Marcus_58_cards_l120_120738

-- Define the conditions as variables
def original_cards : ℕ := 210
def current_cards : ℕ := 268

-- Define the question as a function
def cards_given_by_carter (original current : ℕ) : ℕ := current - original

-- Statement that we need to prove
theorem Carter_gave_Marcus_58_cards : cards_given_by_carter original_cards current_cards = 58 :=
by
  -- Proof goes here
  sorry

end Carter_gave_Marcus_58_cards_l120_120738


namespace ff_two_eq_three_l120_120541

noncomputable def f (x : ℝ) : ℝ :=
  if x < 6 then x^3 else Real.log x / Real.log x

theorem ff_two_eq_three : f (f 2) = 3 := by
  sorry

end ff_two_eq_three_l120_120541


namespace smallest_n_l120_120074

theorem smallest_n (n : ℕ) (h1 : n % 6 = 4) (h2 : n % 7 = 3) (h3 : 15 < n) : n = 52 :=
by
  sorry

end smallest_n_l120_120074


namespace false_statement_divisibility_l120_120613

-- Definitions for the divisibility conditions
def divisible_by (a b : ℕ) : Prop := ∃ k, b = a * k

-- The problem statement
theorem false_statement_divisibility (N : ℕ) :
  (divisible_by 2 N ∧ divisible_by 4 N ∧ divisible_by 12 N ∧ ¬ divisible_by 24 N) →
  (¬ divisible_by 24 N) :=
by
  -- The proof will need to be filled in here
  sorry

end false_statement_divisibility_l120_120613


namespace quadratic_roots_k_relation_l120_120035

theorem quadratic_roots_k_relation (k a b k1 k2 : ℝ) 
    (h_eq : k * (a^2 - a) + 2 * a + 7 = 0)
    (h_eq_b : k * (b^2 - b) + 2 * b + 7 = 0)
    (h_ratio : a / b + b / a = 3)
    (h_k : k = k1 ∨ k = k2)
    (h_vieta_sum : k1 + k2 = 39)
    (h_vieta_product : k1 * k2 = 4) :
    k1 / k2 + k2 / k1 = 1513 / 4 := 
    sorry

end quadratic_roots_k_relation_l120_120035


namespace discount_difference_l120_120623

theorem discount_difference (p : ℝ) (single_discount first_discount second_discount : ℝ) :
    p = 12000 →
    single_discount = 0.45 →
    first_discount = 0.35 →
    second_discount = 0.10 →
    (p * (1 - single_discount) - p * (1 - first_discount) * (1 - second_discount) = 420) := by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end discount_difference_l120_120623


namespace find_x_l120_120473

def balanced (a b c d : ℝ) : Prop :=
  a + b + c + d = a^2 + b^2 + c^2 + d^2

theorem find_x (x : ℝ) : (∀ (a b c d : ℝ), balanced a b c d → (x - a) * (x - b) * (x - c) * (x - d) ≥ 0) ↔ x ≥ 3 / 2 := 
by
  sorry

end find_x_l120_120473


namespace find_b_plus_m_l120_120313

noncomputable def f (a b x : ℝ) : ℝ := Real.log (x + 1) / Real.log a + b 

variable (a b m : ℝ)
-- Conditions
axiom h1 : a > 0
axiom h2 : a ≠ 1
axiom h3 : f a b m = 3

theorem find_b_plus_m : b + m = 3 :=
sorry

end find_b_plus_m_l120_120313


namespace line_through_point_and_intersects_circle_with_chord_length_8_l120_120839

theorem line_through_point_and_intersects_circle_with_chord_length_8 :
  ∃ (l : ℝ → ℝ), (∀ (x : ℝ), l x = 0 ↔ x = 5) ∨ 
  (∀ (x y : ℝ), 7 * x + 24 * y = 35) ↔ 
  (∃ (x : ℝ), x = 5) ∨ 
  (∀ (x y : ℝ), 7 * x + 24 * y = 35) := 
by
  sorry

end line_through_point_and_intersects_circle_with_chord_length_8_l120_120839


namespace binomial_coefficient_30_3_l120_120524

theorem binomial_coefficient_30_3 :
  Nat.choose 30 3 = 4060 := 
by 
  sorry

end binomial_coefficient_30_3_l120_120524


namespace minimize_sum_l120_120788

noncomputable def objective_function (x : ℝ) : ℝ := x + x^2

theorem minimize_sum : ∃ x : ℝ, (objective_function x = x + x^2) ∧ (∀ y : ℝ, objective_function y ≥ objective_function (-1/2)) :=
by
  sorry

end minimize_sum_l120_120788


namespace condition_neither_sufficient_nor_necessary_l120_120809

variable (a b : ℝ)

theorem condition_neither_sufficient_nor_necessary 
    (h1 : ∃ a b : ℝ, a > b ∧ ¬(a^2 > b^2))
    (h2 : ∃ a b : ℝ, a^2 > b^2 ∧ ¬(a > b)) :
  ¬((a > b) ↔ (a^2 > b^2)) :=
sorry

end condition_neither_sufficient_nor_necessary_l120_120809


namespace actual_cost_of_article_l120_120990

theorem actual_cost_of_article (x : ℝ) (hx : 0.76 * x = 988) : x = 1300 :=
sorry

end actual_cost_of_article_l120_120990


namespace chord_intersects_inner_circle_probability_l120_120616

noncomputable def probability_of_chord_intersecting_inner_circle
  (radius_inner : ℝ) (radius_outer : ℝ)
  (chord_probability : ℝ) : Prop :=
  radius_inner = 3 ∧ radius_outer = 5 ∧ chord_probability = 0.205

theorem chord_intersects_inner_circle_probability :
  probability_of_chord_intersecting_inner_circle 3 5 0.205 :=
by {
  sorry
}

end chord_intersects_inner_circle_probability_l120_120616


namespace tiles_per_row_l120_120713

theorem tiles_per_row (area : ℝ) (tile_length : ℝ) (h1 : area = 256) (h2 : tile_length = 2/3) : 
  (16 * 12) / (8) = 24 :=
by {
  sorry
}

end tiles_per_row_l120_120713


namespace arithmetic_sequence_sum_l120_120068

variable {a : ℕ → ℝ}

-- Condition 1: Arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
∀ n : ℕ, ∃ d : ℝ, a (n + 1) = a n + d

-- Condition 2: Given property
def property (a : ℕ → ℝ) : Prop :=
a 7 + a 13 = 20

theorem arithmetic_sequence_sum (h_seq : is_arithmetic_sequence a) (h_prop : property a) :
  a 9 + a 10 + a 11 = 30 := 
sorry

end arithmetic_sequence_sum_l120_120068


namespace gift_spending_l120_120214

def total_amount : ℝ := 700.00
def wrapping_expenses : ℝ := 139.00
def amount_spent_on_gifts : ℝ := 700.00 - 139.00

theorem gift_spending :
  (total_amount - wrapping_expenses) = 561.00 :=
by
  sorry

end gift_spending_l120_120214


namespace frank_remaining_money_l120_120810

theorem frank_remaining_money
  (cheapest_lamp : ℕ)
  (most_expensive_factor : ℕ)
  (frank_money : ℕ)
  (cheapest_lamp_cost : cheapest_lamp = 20)
  (most_expensive_lamp_cost : most_expensive_factor = 3)
  (frank_current_money : frank_money = 90) :
  frank_money - (most_expensive_factor * cheapest_lamp) = 30 :=
by {
  sorry
}

end frank_remaining_money_l120_120810


namespace range_of_a_l120_120471

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, (e^x - a)^2 + x^2 - 2 * a * x + a^2 ≤ 1 / 2) ↔ a = 1 / 2 :=
by
  sorry

end range_of_a_l120_120471


namespace xiao_yun_age_l120_120373

theorem xiao_yun_age (x : ℕ) (h1 : ∀ x, x + 25 = Xiao_Yun_fathers_current_age)
                     (h2 : ∀ x, Xiao_Yun_fathers_age_in_5_years = 2 * (x+5) - 10) :
  x = 30 := by
  sorry

end xiao_yun_age_l120_120373


namespace correct_polynomial_and_result_l120_120200

theorem correct_polynomial_and_result :
  ∃ p q r : Polynomial ℝ,
    q = X^2 - 3 * X + 5 ∧
    p + q = 5 * X^2 - 2 * X + 4 ∧
    p = 4 * X^2 + X - 1 ∧
    r = p - q ∧
    r = 3 * X^2 + 4 * X - 6 :=
by {
  sorry
}

end correct_polynomial_and_result_l120_120200


namespace how_many_strawberries_did_paul_pick_l120_120108

-- Here, we will define the known quantities
def original_strawberries : Nat := 28
def total_strawberries : Nat := 63

-- The statement to prove
theorem how_many_strawberries_did_paul_pick : total_strawberries - original_strawberries = 35 :=
by
  unfold total_strawberries
  unfold original_strawberries
  calc
    63 - 28 = 35 := by norm_num

end how_many_strawberries_did_paul_pick_l120_120108


namespace prism_faces_l120_120710

theorem prism_faces (E : ℕ) (h : E = 18) : 
  ∃ F : ℕ, F = 8 :=
by
  have L : ℕ := E / 3
  have F : ℕ := L + 2
  use F
  sorry

end prism_faces_l120_120710


namespace expression_value_l120_120982

theorem expression_value :
  3 * ((18 + 7)^2 - (7^2 + 18^2)) = 756 := 
sorry

end expression_value_l120_120982


namespace remainder_a25_div_26_l120_120826

def concatenate_numbers (n : ℕ) : ℕ :=
  -- Placeholder function for concatenating numbers from 1 to n
  sorry

theorem remainder_a25_div_26 :
  let a_25 := concatenate_numbers 25
  a_25 % 26 = 13 :=
by sorry

end remainder_a25_div_26_l120_120826


namespace line_through_point_parallel_l120_120405

theorem line_through_point_parallel (x y : ℝ) : 
  (∃ c : ℝ, x - 2 * y + c = 0 ∧ ∃ p : ℝ × ℝ, p = (1, 0) ∧ x - 2 * p.2 + c = 0) → (x - 2 * y - 1 = 0) :=
by
  sorry

end line_through_point_parallel_l120_120405


namespace license_count_l120_120232

def num_licenses : ℕ :=
  let num_letters := 3
  let num_digits := 10
  let num_digit_slots := 6
  num_letters * num_digits ^ num_digit_slots

theorem license_count :
  num_licenses = 3000000 := by
  sorry

end license_count_l120_120232


namespace hyperbola_focal_length_l120_120396

-- Define the constants a^2 and b^2 based on the given hyperbola equation.
def a_squared : ℝ := 16
def b_squared : ℝ := 25

-- Define the constants a and b as the square roots of a^2 and b^2.
noncomputable def a : ℝ := Real.sqrt a_squared
noncomputable def b : ℝ := Real.sqrt b_squared

-- Define the constant c based on the relation c^2 = a^2 + b^2.
noncomputable def c : ℝ := Real.sqrt (a_squared + b_squared)

-- The focal length of the hyperbola is 2c.
noncomputable def focal_length : ℝ := 2 * c

-- The theorem that captures the statement of the problem.
theorem hyperbola_focal_length : focal_length = 2 * Real.sqrt 41 := by
  -- Proof omitted.
  sorry

end hyperbola_focal_length_l120_120396


namespace evaluate_polynomial_at_two_l120_120246

def f (x : ℝ) : ℝ := x^5 + 2 * x^3 + 3 * x^2 + x + 1

theorem evaluate_polynomial_at_two : f 2 = 41 := by
  sorry

end evaluate_polynomial_at_two_l120_120246


namespace dvd_cost_packs_l120_120970

theorem dvd_cost_packs (cost_per_pack : ℕ) (number_of_packs : ℕ) (total_money : ℕ) :
  cost_per_pack = 12 → number_of_packs = 11 → total_money = (cost_per_pack * number_of_packs) → total_money = 132 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end dvd_cost_packs_l120_120970


namespace avg_salary_officers_l120_120998

-- Definitions of the given conditions
def avg_salary_employees := 120
def avg_salary_non_officers := 110
def num_officers := 15
def num_non_officers := 495

-- The statement to be proven
theorem avg_salary_officers : (15 * (15 * X) / (15 + 495)) = 450 :=
by
  sorry

end avg_salary_officers_l120_120998


namespace axis_of_symmetry_l120_120058

variables (a : ℝ) (x : ℝ)

def parabola := a * (x + 1) * (x - 3)

theorem axis_of_symmetry (h : a ≠ 0) : x = 1 := 
sorry

end axis_of_symmetry_l120_120058


namespace additional_birds_flew_up_l120_120064

-- Defining the conditions from the problem
def original_birds : ℕ := 179
def total_birds : ℕ := 217

-- Defining the question to be proved as a theorem
theorem additional_birds_flew_up : 
  total_birds - original_birds = 38 :=
by
  sorry

end additional_birds_flew_up_l120_120064


namespace can_cabinet_be_moved_out_through_door_l120_120908

/-
Definitions for the problem:
- Length, width, and height of the room
- Width, height, and depth of the cabinet
- Width and height of the door
-/

structure Dimensions where
  length : ℝ
  width : ℝ
  height : ℝ

def room : Dimensions := { length := 4, width := 2.5, height := 2.3 }
def cabinet : Dimensions := { length := 0.6, width := 1.8, height := 2.1 }
def door : Dimensions := { length := 0.8, height := 1.9, width := 0 }

theorem can_cabinet_be_moved_out_through_door : 
  (cabinet.length ≤ door.length ∧ cabinet.width ≤ door.height) ∨ 
  (cabinet.width ≤ door.length ∧ cabinet.length ≤ door.height) 
∧ 
cabinet.height ≤ room.height ∧ cabinet.width ≤ room.width ∧ 
cabinet.length ≤ room.length → True :=
by
  sorry

end can_cabinet_be_moved_out_through_door_l120_120908


namespace three_digit_max_l120_120112

theorem three_digit_max (n : ℕ) : 
  n % 9 = 1 ∧ n % 5 = 3 ∧ n % 7 = 2 ∧ 100 <= n ∧ n <= 999 → n = 793 :=
by
  sorry

end three_digit_max_l120_120112


namespace quadratic_function_min_value_l120_120695

noncomputable def f (a h k : ℝ) (x : ℝ) : ℝ :=
  a * (x - h) ^ 2 + k

theorem quadratic_function_min_value :
  ∀ (f : ℝ → ℝ) (n : ℕ),
  (f n = 13) ∧ (f (n + 1) = 13) ∧ (f (n + 2) = 35) →
  (∃ k, k = 2) :=
  sorry

end quadratic_function_min_value_l120_120695


namespace factorize_expression_l120_120572

theorem factorize_expression (x y : ℝ) : 
  (x + y)^2 - 14 * (x + y) + 49 = (x + y - 7)^2 := 
by
  sorry

end factorize_expression_l120_120572


namespace profit_without_discount_l120_120834

theorem profit_without_discount (CP SP_discount SP_without_discount : ℝ) (profit_discount profit_without_discount percent_discount : ℝ)
  (h1 : CP = 100) 
  (h2 : percent_discount = 0.05) 
  (h3 : profit_discount = 0.425) 
  (h4 : SP_discount = CP + profit_discount * CP) 
  (h5 : SP_discount = 142.5)
  (h6 : SP_without_discount = SP_discount / (1 - percent_discount)) : 
  profit_without_discount = ((SP_without_discount - CP) / CP) * 100 := 
by
  sorry

end profit_without_discount_l120_120834


namespace sprinted_further_than_jogged_l120_120400

def sprint_distance1 := 0.8932
def sprint_distance2 := 0.7773
def sprint_distance3 := 0.9539
def sprint_distance4 := 0.5417
def sprint_distance5 := 0.6843

def jog_distance1 := 0.7683
def jog_distance2 := 0.4231
def jog_distance3 := 0.5733
def jog_distance4 := 0.625
def jog_distance5 := 0.6549

def total_sprint_distance := sprint_distance1 + sprint_distance2 + sprint_distance3 + sprint_distance4 + sprint_distance5
def total_jog_distance := jog_distance1 + jog_distance2 + jog_distance3 + jog_distance4 + jog_distance5

theorem sprinted_further_than_jogged :
  total_sprint_distance - total_jog_distance = 0.8058 :=
by
  sorry

end sprinted_further_than_jogged_l120_120400


namespace valid_t_range_for_f_l120_120699

theorem valid_t_range_for_f :
  (∀ x : ℝ, |x + 1| + |x - t| ≥ 2015) ↔ t ∈ (Set.Iic (-2016) ∪ Set.Ici 2014) := 
sorry

end valid_t_range_for_f_l120_120699


namespace solve_equation_l120_120693

theorem solve_equation (x : ℝ) (h : -x^2 = (3 * x + 1) / (x + 3)) : x = -1 :=
sorry

end solve_equation_l120_120693


namespace evaluate_expression_l120_120171

noncomputable def expression := 
  (Real.sqrt 3 * Real.tan (Real.pi / 15) - 3) / 
  (4 * (Real.cos (Real.pi / 15))^2 * Real.sin (Real.pi / 15) - 2 * Real.sin (Real.pi / 15))

theorem evaluate_expression : expression = -4 * Real.sqrt 3 :=
  sorry

end evaluate_expression_l120_120171


namespace maximize_sequence_l120_120545

theorem maximize_sequence (n : ℕ) (an : ℕ → ℝ) (h : ∀ n, an n = (10/11)^n * (3 * n + 13)) : 
  (∃ n_max, (∀ m, an m ≤ an n_max) ∧ n_max = 6) :=
by
  sorry

end maximize_sequence_l120_120545


namespace remaining_insects_is_twenty_one_l120_120385

-- Define the initial counts of each type of insect
def spiders := 3
def ants := 12
def ladybugs := 8

-- Define the number of ladybugs that flew away
def ladybugs_flew_away := 2

-- Define the total initial number of insects
def total_insects_initial := spiders + ants + ladybugs

-- Define the total number of insects that remain after some ladybugs fly away
def total_insects_remaining := total_insects_initial - ladybugs_flew_away

-- Theorem statement: proving that the number of insects remaining is 21
theorem remaining_insects_is_twenty_one : total_insects_remaining = 21 := sorry

end remaining_insects_is_twenty_one_l120_120385


namespace total_marbles_l120_120479

theorem total_marbles (ratio_red_blue_green_yellow : ℕ → ℕ → ℕ → ℕ → Prop) (total : ℕ) :
  (∀ r b g y, ratio_red_blue_green_yellow r b g y ↔ r = 1 ∧ b = 5 ∧ g = 3 ∧ y = 2) →
  (∃ y, y = 20) →
  (total = y * 11 / 2) →
  total = 110 :=
by
  intros ratio_condition yellow_condition total_condition
  sorry

end total_marbles_l120_120479


namespace N_8_12_eq_288_l120_120582

-- Definitions for various polygonal numbers
def N3 (n : ℕ) : ℕ := n * (n + 1) / 2
def N4 (n : ℕ) : ℕ := n^2
def N5 (n : ℕ) : ℕ := 3 * n^2 / 2 - n / 2
def N6 (n : ℕ) : ℕ := 2 * n^2 - n

-- General definition conjectured
def N (n k : ℕ) : ℕ := (k - 2) * n^2 / 2 + (4 - k) * n / 2

-- The problem statement to prove N(8, 12) == 288
theorem N_8_12_eq_288 : N 8 12 = 288 := by
  -- We would need the proofs for the definitional equalities and calculation here
  sorry

end N_8_12_eq_288_l120_120582


namespace ripe_oranges_count_l120_120528

/-- They harvest 52 sacks of unripe oranges per day. -/
def unripe_oranges_per_day : ℕ := 52

/-- After 26 days of harvest, they will have 2080 sacks of oranges. -/
def total_oranges_after_26_days : ℕ := 2080

/-- Define the number of sacks of ripe oranges harvested per day. -/
def ripe_oranges_per_day (R : ℕ) : Prop :=
  26 * (R + unripe_oranges_per_day) = total_oranges_after_26_days

/-- Prove that they harvest 28 sacks of ripe oranges per day. -/
theorem ripe_oranges_count : ripe_oranges_per_day 28 :=
by {
  -- This is where the proof would go
  sorry
}

end ripe_oranges_count_l120_120528


namespace find_remainder_proof_l120_120207

def div_remainder_problem :=
  let number := 220050
  let sum := 555 + 445
  let difference := 555 - 445
  let quotient := 2 * difference
  let divisor := sum
  let quotient_correct := quotient = 220
  let division_formula := number = divisor * quotient + 50
  quotient_correct ∧ division_formula

theorem find_remainder_proof : div_remainder_problem := by
  sorry

end find_remainder_proof_l120_120207


namespace remainder_3249_div_82_eq_51_l120_120493

theorem remainder_3249_div_82_eq_51 : (3249 % 82) = 51 :=
by
  sorry

end remainder_3249_div_82_eq_51_l120_120493


namespace mode_of_gold_medals_is_8_l120_120234

def countries : List String := ["Norway", "Germany", "China", "USA", "Sweden", "Netherlands", "Austria"]

def gold_medals : List Nat := [16, 12, 9, 8, 8, 8, 7]

def mode (lst : List Nat) : Nat :=
  lst.foldr
    (fun (x : Nat) acc =>
      if lst.count x > lst.count acc then x else acc)
    lst.head!

theorem mode_of_gold_medals_is_8 :
  mode gold_medals = 8 :=
by sorry

end mode_of_gold_medals_is_8_l120_120234


namespace total_books_l120_120585

-- Define the number of books Tim has
def TimBooks : ℕ := 44

-- Define the number of books Sam has
def SamBooks : ℕ := 52

-- Statement to prove that the total number of books is 96
theorem total_books : TimBooks + SamBooks = 96 := by
  sorry

end total_books_l120_120585


namespace find_k_l120_120403

def system_of_equations (x y k : ℝ) : Prop :=
  x - y = k - 3 ∧
  3 * x + 5 * y = 2 * k + 8 ∧
  x + y = 2

theorem find_k (x y k : ℝ) (h : system_of_equations x y k) : k = 1 := 
sorry

end find_k_l120_120403


namespace total_nephews_proof_l120_120772

-- We declare the current number of nephews as unknown variables
variable (Alden_current Vihaan Shruti Nikhil : ℕ)

-- State the conditions as hypotheses
theorem total_nephews_proof
  (h1 : 70 = (1 / 3 : ℚ) * Alden_current)
  (h2 : Vihaan = Alden_current + 120)
  (h3 : Shruti = 2 * Vihaan)
  (h4 : Nikhil = Alden_current + Shruti - 40) :
  Alden_current + Vihaan + Shruti + Nikhil = 2030 := 
by
  sorry

end total_nephews_proof_l120_120772


namespace find_original_list_size_l120_120250

theorem find_original_list_size
  (n m : ℤ)
  (h1 : (m + 3) * (n + 1) = m * n + 20)
  (h2 : (m + 1) * (n + 2) = m * n + 22):
  n = 7 :=
sorry

end find_original_list_size_l120_120250


namespace fraction_power_rule_l120_120538

theorem fraction_power_rule :
  (5 / 6) ^ 4 = (625 : ℚ) / 1296 := 
by sorry

end fraction_power_rule_l120_120538


namespace initial_video_files_l120_120562

theorem initial_video_files (V : ℕ) (h1 : 26 + V - 48 = 14) : V = 36 := 
by
  sorry

end initial_video_files_l120_120562


namespace stacy_has_2_more_than_triple_steve_l120_120639

-- Definitions based on the given conditions
def skylar_berries : ℕ := 20
def steve_berries : ℕ := skylar_berries / 2
def stacy_berries : ℕ := 32

-- Statement to be proved
theorem stacy_has_2_more_than_triple_steve :
  stacy_berries = 3 * steve_berries + 2 := by
  sorry

end stacy_has_2_more_than_triple_steve_l120_120639


namespace probability_of_winning_fifth_game_championship_correct_overall_probability_of_winning_championship_correct_l120_120392

noncomputable def binomial (n k : ℕ) : ℝ :=
  (Nat.choose n k : ℝ)

noncomputable def probability_of_winning_fifth_game_championship : ℝ :=
  binomial 4 3 * 0.6^4 * 0.4

noncomputable def overall_probability_of_winning_championship : ℝ :=
  0.6^4 +
  binomial 4 3 * 0.6^4 * 0.4 +
  binomial 5 3 * 0.6^4 * 0.4^2 +
  binomial 6 3 * 0.6^4 * 0.4^3

theorem probability_of_winning_fifth_game_championship_correct :
  probability_of_winning_fifth_game_championship = 0.20736 := by
  sorry

theorem overall_probability_of_winning_championship_correct :
  overall_probability_of_winning_championship = 0.710208 := by
  sorry

end probability_of_winning_fifth_game_championship_correct_overall_probability_of_winning_championship_correct_l120_120392


namespace total_crayons_l120_120589

-- Definitions for conditions
def boxes : Nat := 7
def crayons_per_box : Nat := 5

-- Statement that needs to be proved
theorem total_crayons : boxes * crayons_per_box = 35 := by
  sorry

end total_crayons_l120_120589


namespace set_difference_A_B_l120_120102

-- Defining the sets A and B
def setA : Set ℝ := { x : ℝ | abs (4 * x - 1) > 9 }
def setB : Set ℝ := { x : ℝ | x >= 0 }

-- The theorem stating the result of set difference A - B
theorem set_difference_A_B : (setA \ setB) = { x : ℝ | x > 5/2 } :=
by
  -- Proof omitted
  sorry

end set_difference_A_B_l120_120102


namespace fifteenth_even_multiple_of_5_l120_120526

theorem fifteenth_even_multiple_of_5 : 15 * 2 * 5 = 150 := by
  sorry

end fifteenth_even_multiple_of_5_l120_120526


namespace part_1_conditions_part_2_min_value_l120_120505

theorem part_1_conditions
  (a b x : ℝ)
  (h1: 2 * a * x^2 - 8 * x - 3 * a^2 < 0)
  (h2: ∀ x, -1 < x -> x < b)
  : a = 2 ∧ b = 3 := sorry

theorem part_2_min_value
  (a b x y : ℝ)
  (h1: x > 0)
  (h2: y > 0)
  (h3: a = 2)
  (h4: b = 3)
  (h5: (a / x) + (b / y) = 1)
  : ∃ min_val : ℝ, min_val = 3 * x + 2 * y ∧ min_val = 24 := sorry

end part_1_conditions_part_2_min_value_l120_120505


namespace locus_eq_l120_120977

noncomputable def locus_of_centers (a b : ℝ) : Prop :=
  5 * a^2 + 9 * b^2 + 80 * a - 400 = 0

theorem locus_eq (a b : ℝ) :
  (∃ r : ℝ, (a^2 + b^2 = (r + 2)^2) ∧ ((a - 1)^2 + b^2 = (5 - r)^2)) →
  locus_of_centers a b :=
by
  intro h
  sorry

end locus_eq_l120_120977


namespace inequality_proof_l120_120648

noncomputable def inequality (x y z : ℝ) : Prop :=
  (x + y) / z + (y + z) / x + (z + x) / y ≥ 7

theorem inequality_proof (x y z : ℝ) (hx : x ≥ y + z) (hx_pos: 0 < x) (hy_pos: 0 < y) (hz_pos: 0 < z) :
  inequality x y z :=
by
  sorry

end inequality_proof_l120_120648


namespace hyperbola_no_common_point_l120_120164

theorem hyperbola_no_common_point (a b : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0) (y_line : ∀ x : ℝ, y = 2 * x) : 
  ∃ e : ℝ, e = (Real.sqrt (a^2 + b^2)) / a ∧ 1 < e ∧ e ≤ Real.sqrt 5 :=
by
  sorry

end hyperbola_no_common_point_l120_120164


namespace distance_between_foci_l120_120263

theorem distance_between_foci (a b : ℝ) (h₁ : a^2 = 18) (h₂ : b^2 = 2) :
  2 * (Real.sqrt (a^2 + b^2)) = 4 * Real.sqrt 5 :=
by
  sorry

end distance_between_foci_l120_120263


namespace necessary_sufficient_condition_geometric_sequence_l120_120659

noncomputable def an_geometric (a : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = r * a n

theorem necessary_sufficient_condition_geometric_sequence
  (a : ℕ → ℝ) (S : ℕ → ℝ) (p q : ℝ) (h_sum : ∀ n : ℕ, S (n + 1) = S n + a (n + 1))
  (h_eq : ∀ n : ℕ, a (n + 1) = p * S n + q) :
  (a 1 = q) ↔ (∃ r : ℝ, an_geometric a r) :=
sorry

end necessary_sufficient_condition_geometric_sequence_l120_120659


namespace max_difference_in_flour_masses_l120_120923

/--
Given three brands of flour with the following mass ranges:
1. Brand A: (48 ± 0.1) kg
2. Brand B: (48 ± 0.2) kg
3. Brand C: (48 ± 0.3) kg

Prove that the maximum difference in mass between any two bags of these different brands is 0.5 kg.
-/
theorem max_difference_in_flour_masses :
  (∀ (a b : ℝ), ((47.9 ≤ a ∧ a ≤ 48.1) ∧ (47.8 ≤ b ∧ b ≤ 48.2)) →
    |a - b| ≤ 0.5) ∧
  (∀ (a c : ℝ), ((47.9 ≤ a ∧ a ≤ 48.1) ∧ (47.7 ≤ c ∧ c ≤ 48.3)) →
    |a - c| ≤ 0.5) ∧
  (∀ (b c : ℝ), ((47.8 ≤ b ∧ b ≤ 48.2) ∧ (47.7 ≤ c ∧ c ≤ 48.3)) →
    |b - c| ≤ 0.5) := 
sorry

end max_difference_in_flour_masses_l120_120923


namespace maximum_value_of_f_minimum_value_of_f_l120_120517

-- Define the function f
def f (x y : ℝ) : ℝ := 3 * |x + y| + |4 * y + 9| + |7 * y - 3 * x - 18|

-- Define the condition
def condition (x y : ℝ) : Prop := x^2 + y^2 ≤ 5

-- State the maximum value theorem
theorem maximum_value_of_f (x y : ℝ) (h : condition x y) :
  ∃ (x y : ℝ), f x y = 27 + 6 * Real.sqrt 5 := sorry

-- State the minimum value theorem
theorem minimum_value_of_f (x y : ℝ) (h : condition x y) :
  ∃ (x y : ℝ), f x y = 27 - 3 * Real.sqrt 10 := sorry

end maximum_value_of_f_minimum_value_of_f_l120_120517


namespace scale_reading_l120_120421

theorem scale_reading (x : ℝ) (h₁ : 3.25 < x) (h₂ : x < 3.5) : x = 3.3 :=
sorry

end scale_reading_l120_120421


namespace polynomial_expansion_sum_l120_120357

theorem polynomial_expansion_sum :
  ∀ P Q R S : ℕ, ∀ x : ℕ, 
  (P = 4 ∧ Q = 10 ∧ R = 1 ∧ S = 21) → 
  ((x + 3) * (4 * x ^ 2 - 2 * x + 7) = P * x ^ 3 + Q * x ^ 2 + R * x + S) → 
  P + Q + R + S = 36 :=
by
  intros P Q R S x h1 h2
  sorry

end polynomial_expansion_sum_l120_120357


namespace dawn_bananas_l120_120133

-- Definitions of the given conditions
def total_bananas : ℕ := 200
def lydia_bananas : ℕ := 60
def donna_bananas : ℕ := 40

-- Proof that Dawn has 100 bananas
theorem dawn_bananas : (total_bananas - donna_bananas) - lydia_bananas = 100 := by
  sorry

end dawn_bananas_l120_120133


namespace stacy_days_to_finish_l120_120730

-- Definitions based on the conditions
def total_pages : ℕ := 81
def pages_per_day : ℕ := 27

-- The theorem statement
theorem stacy_days_to_finish : total_pages / pages_per_day = 3 := by
  -- the proof is omitted
  sorry

end stacy_days_to_finish_l120_120730


namespace not_all_less_than_two_l120_120418

theorem not_all_less_than_two {a b c : ℝ} (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ¬(a + 1/b < 2 ∧ b + 1/c < 2 ∧ c + 1/a < 2) :=
sorry

end not_all_less_than_two_l120_120418


namespace hyperbola_foci_distance_l120_120871

theorem hyperbola_foci_distance :
  let a := Real.sqrt 25
  let b := Real.sqrt 9
  let c := Real.sqrt (a ^ 2 + b ^ 2)
  let distance := 2 * c
  distance = 2 * Real.sqrt 34 :=
by
  let a := Real.sqrt 25
  let b := Real.sqrt 9
  let c := Real.sqrt (a ^ 2 + b ^ 2)
  let distance := 2 * c
  exact sorry

end hyperbola_foci_distance_l120_120871


namespace tan_of_log_conditions_l120_120969

theorem tan_of_log_conditions (x : ℝ) (h1 : 0 < x ∧ x < (Real.pi / 2))
  (h2 : Real.log (Real.sin (2 * x)) - Real.log (Real.sin x) = Real.log (1 / 2)) :
  Real.tan x = Real.sqrt 15 :=
sorry

end tan_of_log_conditions_l120_120969


namespace red_packet_grabbing_situations_l120_120463

-- Definitions based on the conditions
def numberOfPeople := 5
def numberOfPackets := 4
def packets := [2, 2, 3, 5]  -- 2-yuan, 2-yuan, 3-yuan, 5-yuan

-- Main theorem statement
theorem red_packet_grabbing_situations : 
  ∃ situations : ℕ, situations = 60 :=
by
  sorry

end red_packet_grabbing_situations_l120_120463


namespace milk_fraction_correct_l120_120601

def fraction_of_milk_in_coffee_cup (coffee_initial : ℕ) (milk_initial : ℕ) : ℚ :=
  let coffee_transferred := coffee_initial / 3
  let milk_cup_after_transfer := milk_initial + coffee_transferred
  let coffee_left := coffee_initial - coffee_transferred
  let total_mixed := milk_cup_after_transfer
  let transfer_back := total_mixed / 2
  let coffee_back := transfer_back * (coffee_transferred / total_mixed)
  let milk_back := transfer_back * (milk_initial / total_mixed)
  let coffee_final := coffee_left + coffee_back
  let milk_final := milk_back
  milk_final / (coffee_final + milk_final)

theorem milk_fraction_correct (coffee_initial : ℕ) (milk_initial : ℕ)
  (h_coffee : coffee_initial = 6) (h_milk : milk_initial = 3) :
  fraction_of_milk_in_coffee_cup coffee_initial milk_initial = 3 / 13 :=
by
  sorry

end milk_fraction_correct_l120_120601


namespace derivative_at_1_l120_120428

def f (x : ℝ) : ℝ := x^3 + x^2 - 2 * x - 2

def f_derivative (x : ℝ) : ℝ := 3*x^2 + 2*x - 2

theorem derivative_at_1 : f_derivative 1 = 3 := by
  sorry

end derivative_at_1_l120_120428


namespace ellipse_non_degenerate_l120_120846

noncomputable def non_degenerate_ellipse_condition (b : ℝ) : Prop := b > -13

theorem ellipse_non_degenerate (b : ℝ) :
  (∃ x y : ℝ, 4*x^2 + 9*y^2 - 16*x + 18*y + 12 = b) → non_degenerate_ellipse_condition b :=
by
  sorry

end ellipse_non_degenerate_l120_120846


namespace polynomial_coefficients_l120_120308

theorem polynomial_coefficients
  (x : ℝ)
  (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 : ℝ)
  (h : (x-3)^8 = a_0 + a_1 * (x-2) + a_2 * (x-2)^2 + a_3 * (x-2)^3 + 
                a_4 * (x-2)^4 + a_5 * (x-2)^5 + a_6 * (x-2)^6 + 
                a_7 * (x-2)^7 + a_8 * (x-2)^8) :
  (a_0 = 1) ∧ 
  (a_1 / 2 + a_2 / 2^2 + a_3 / 2^3 + a_4 / 2^4 + a_5 / 2^5 + 
   a_6 / 2^6 + a_7 / 2^7 + a_8 / 2^8 = -255 / 256) ∧ 
  (a_0 + a_2 + a_4 + a_6 + a_8 = 128) :=
by sorry

end polynomial_coefficients_l120_120308


namespace ratio_B_over_A_eq_one_l120_120205

theorem ratio_B_over_A_eq_one (A B : ℤ) (h : ∀ x : ℝ, x ≠ -3 → x ≠ 0 → x ≠ 3 → 
  (A : ℝ) / (x + 3) + (B : ℝ) / (x * (x - 3)) = (x^3 - 3*x^2 + 15*x - 9) / (x^3 + x^2 - 9*x)) :
  (B : ℝ) / (A : ℝ) = 1 :=
sorry

end ratio_B_over_A_eq_one_l120_120205


namespace div_floor_factorial_l120_120151

theorem div_floor_factorial (n q : ℕ) (hn : n ≥ 5) (hq : 2 ≤ q ∧ q ≤ n) :
  q - 1 ∣ (Nat.floor ((Nat.factorial (n - 1)) / q : ℚ)) :=
by
  sorry

end div_floor_factorial_l120_120151


namespace false_statement_B_l120_120500

theorem false_statement_B : ¬ ∀ α β : ℝ, (α < 90) ∧ (β < 90) → (α + β > 90) :=
by
  sorry

end false_statement_B_l120_120500


namespace playground_area_l120_120231

open Real

theorem playground_area (l w : ℝ) (h1 : 2*l + 2*w = 100) (h2 : l = 2*w) : l * w = 5000 / 9 :=
by
  sorry

end playground_area_l120_120231


namespace train_length_l120_120618

/-- Given a train that can cross an electric pole in 15 seconds and has a speed of 72 km/h, prove that the length of the train is 300 meters. -/
theorem train_length 
  (time_to_cross_pole : ℝ)
  (train_speed_kmh : ℝ)
  (h1 : time_to_cross_pole = 15)
  (h2 : train_speed_kmh = 72)
  : (train_speed_kmh * 1000 / 3600) * time_to_cross_pole = 300 := 
by
  -- Proof goes here
  sorry

end train_length_l120_120618


namespace A_n_is_integer_l120_120749

open Real

noncomputable def A_n (a b : ℕ) (θ : ℝ) (n : ℕ) : ℝ :=
  (a^2 + b^2)^n * sin (n * θ)

theorem A_n_is_integer (a b : ℕ) (h : a > b) (θ : ℝ) (hθ1 : 0 < θ) (hθ2 : θ < pi/2) (h_sin : sin θ = 2 * a * b / (a^2 + b^2)) :
  ∀ n : ℕ, ∃ k : ℤ, A_n a b θ n = k :=
by
  sorry

end A_n_is_integer_l120_120749


namespace amount_paid_l120_120088

theorem amount_paid (lemonade_price_per_cup sandwich_price_per_item change_received : ℝ) 
    (num_lemonades num_sandwiches : ℕ)
    (h1 : lemonade_price_per_cup = 2) 
    (h2 : sandwich_price_per_item = 2.50) 
    (h3 : change_received = 11) 
    (h4 : num_lemonades = 2) 
    (h5 : num_sandwiches = 2) : 
    (lemonade_price_per_cup * num_lemonades + sandwich_price_per_item * num_sandwiches + change_received = 20) :=
by
  sorry

end amount_paid_l120_120088


namespace shortest_distance_between_circles_zero_l120_120568

noncomputable def center_radius_circle1 : (ℝ × ℝ) × ℝ :=
  let c1 := (3, -5)
  let r1 := Real.sqrt 20
  (c1, r1)

noncomputable def center_radius_circle2 : (ℝ × ℝ) × ℝ :=
  let c2 := (-4, 1)
  let r2 := Real.sqrt 1
  (c2, r2)

theorem shortest_distance_between_circles_zero :
  let c1 := center_radius_circle1.1
  let r1 := center_radius_circle1.2
  let c2 := center_radius_circle2.1
  let r2 := center_radius_circle2.2
  let dist := Real.sqrt ((c1.1 - c2.1)^2 + (c1.2 - c2.2)^2)
  dist < r1 + r2 → 0 = 0 :=
by
  intros
  -- Add appropriate steps for the proof (skipping by using sorry for now)
  sorry

end shortest_distance_between_circles_zero_l120_120568


namespace hyperbola_center_l120_120876

theorem hyperbola_center (x1 y1 x2 y2 : ℝ) (f1 : x1 = 3) (f2 : y1 = -2) (f3 : x2 = 11) (f4 : y2 = 6) :
    (x1 + x2) / 2 = 7 ∧ (y1 + y2) / 2 = 2 :=
by
  sorry

end hyperbola_center_l120_120876


namespace brian_holds_breath_for_60_seconds_l120_120474

-- Definitions based on the problem conditions:
def initial_time : ℕ := 10
def after_first_week (t : ℕ) : ℕ := t * 2
def after_second_week (t : ℕ) : ℕ := t * 2
def after_final_week (t : ℕ) : ℕ := (t * 3) / 2

-- The Lean statement to prove:
theorem brian_holds_breath_for_60_seconds :
  after_final_week (after_second_week (after_first_week initial_time)) = 60 :=
by
  -- Proof steps would go here
  sorry

end brian_holds_breath_for_60_seconds_l120_120474


namespace rahim_sequence_final_value_l120_120402

theorem rahim_sequence_final_value :
  ∃ (a : ℕ) (b : ℕ), a ^ b = 5 ^ 16 :=
sorry

end rahim_sequence_final_value_l120_120402


namespace price_of_first_variety_l120_120962

theorem price_of_first_variety
  (P : ℝ)
  (H1 : 1 * P + 1 * 135 + 2 * 175.5 = 4 * 153) :
  P = 126 :=
by
  sorry

end price_of_first_variety_l120_120962


namespace rachel_brought_16_brownies_l120_120078

def total_brownies : ℕ := 40
def brownies_left_at_home : ℕ := 24

def brownies_brought_to_school : ℕ :=
  total_brownies - brownies_left_at_home

theorem rachel_brought_16_brownies :
  brownies_brought_to_school = 16 :=
by
  sorry

end rachel_brought_16_brownies_l120_120078


namespace max_profit_l120_120258

variables (x y : ℝ)

def profit (x y : ℝ) : ℝ := 50000 * x + 30000 * y

theorem max_profit :
  (3 * x + y ≤ 13) ∧ (2 * x + 3 * y ≤ 18) ∧ (x ≥ 0) ∧ (y ≥ 0) →
  (∃ x y, profit x y = 390000) :=
by
  sorry

end max_profit_l120_120258


namespace monotonic_intervals_range_of_k_l120_120178

noncomputable def f (x k : ℝ) : ℝ := (Real.exp x) / (x ^ 2) - k * (2 / x + Real.log x)
noncomputable def f' (x k : ℝ) : ℝ := (x - 2) * (Real.exp x - k * x) / (x^3)

theorem monotonic_intervals (k : ℝ) (h : k ≤ 0) :
  (∀ x : ℝ, 0 < x ∧ x < 2 → f' x k < 0) ∧ (∀ x : ℝ, x > 2 → f' x k > 0) := sorry

theorem range_of_k (k : ℝ) (h : e < k ∧ k < (e^2)/2) :
  ∃ x1 x2 : ℝ, 0 < x1 ∧ x1 < 2 ∧ 0 < x2 ∧ x2 < 2 ∧ 
    (f' x1 k = 0 ∧ f' x2 k = 0 ∧ x1 ≠ x2) := sorry

end monotonic_intervals_range_of_k_l120_120178


namespace expression_evaluation_l120_120893

theorem expression_evaluation :
  (8 / 4 - 3^2 + 4 * 5) = 13 :=
by sorry

end expression_evaluation_l120_120893


namespace bikes_in_parking_lot_l120_120006

theorem bikes_in_parking_lot (C : ℕ) (Total_Wheels : ℕ) (Wheels_per_car : ℕ) (Wheels_per_bike : ℕ) (h1 : C = 14) (h2 : Total_Wheels = 76) (h3 : Wheels_per_car = 4) (h4 : Wheels_per_bike = 2) : 
  ∃ B : ℕ, 4 * C + 2 * B = Total_Wheels ∧ B = 10 :=
by
  sorry

end bikes_in_parking_lot_l120_120006


namespace percentage_excess_calculation_l120_120686

theorem percentage_excess_calculation (A B : ℝ) (x : ℝ) 
  (h1 : (A * (1 + x / 100)) * (B * 0.95) = A * B * 1.007) : 
  x = 6.05 :=
by
  sorry

end percentage_excess_calculation_l120_120686


namespace arithmetic_and_geometric_mean_l120_120600

theorem arithmetic_and_geometric_mean (x y : ℝ) (h1: (x + y) / 2 = 20) (h2: Real.sqrt (x * y) = Real.sqrt 110) : x^2 + y^2 = 1380 :=
sorry

end arithmetic_and_geometric_mean_l120_120600


namespace ratio_of_women_working_in_retail_l120_120191

-- Define the population of Los Angeles
def population_LA : ℕ := 6000000

-- Define the proportion of women in Los Angeles
def half_population : ℕ := population_LA / 2

-- Define the number of women working in retail
def women_retail : ℕ := 1000000

-- Define the total number of women in Los Angeles
def total_women : ℕ := half_population

-- The statement to be proven:
theorem ratio_of_women_working_in_retail :
  (women_retail / total_women : ℚ) = 1 / 3 :=
by {
  -- The proof goes here
  sorry
}

end ratio_of_women_working_in_retail_l120_120191


namespace chrysler_building_floors_l120_120491

theorem chrysler_building_floors (L : ℕ) (Chrysler : ℕ) (h1 : Chrysler = L + 11)
  (h2 : L + Chrysler = 35) : Chrysler = 23 :=
by {
  --- proof would be here
  sorry
}

end chrysler_building_floors_l120_120491


namespace allergic_reaction_probability_is_50_percent_l120_120581

def can_have_allergic_reaction (choice : String) : Prop :=
  choice = "peanut_butter"

def percentage_of_allergic_reaction :=
  let total_peanut_butter := 40 + 30
  let total_cookies := 40 + 50 + 30 + 20
  (total_peanut_butter : Float) / (total_cookies : Float) * 100

theorem allergic_reaction_probability_is_50_percent :
  percentage_of_allergic_reaction = 50 := sorry

end allergic_reaction_probability_is_50_percent_l120_120581


namespace smallest_integer_value_l120_120567

theorem smallest_integer_value (x : ℤ) (h : 7 - 3 * x < 22) : x ≥ -4 := 
sorry

end smallest_integer_value_l120_120567


namespace simplify_to_ellipse_l120_120338

theorem simplify_to_ellipse (x y : ℝ) :
  (Real.sqrt ((x - 2)^2 + y^2) + Real.sqrt ((x + 2)^2 + y^2) = 10) →
  (x^2 / 25 + y^2 / 21 = 1) :=
by
  sorry

end simplify_to_ellipse_l120_120338


namespace median_length_of_right_triangle_l120_120898

noncomputable def length_of_median (a b c : ℕ) : ℝ := 
  if a * a + b * b = c * c then c / 2 else 0

theorem median_length_of_right_triangle :
  length_of_median 9 12 15 = 7.5 :=
by
  -- Insert the proof here
  sorry

end median_length_of_right_triangle_l120_120898


namespace remainder_correct_l120_120882

def dividend : ℕ := 165
def divisor : ℕ := 18
def quotient : ℕ := 9
def remainder : ℕ := 3

theorem remainder_correct {d q r : ℕ} (h1 : d = dividend) (h2 : q = quotient) (h3 : r = divisor * q) : d = 165 → q = 9 → 165 = 162 + remainder :=
by { sorry }

end remainder_correct_l120_120882


namespace area_of_region_between_semicircles_l120_120032

/-- Given a region between two semicircles with the same center and parallel diameters,
where the farthest distance between two points with a clear line of sight is 12 meters,
prove that the area of the region is 18π square meters. -/
theorem area_of_region_between_semicircles :
  ∃ (R r : ℝ), R > r ∧ (R - r = 6) ∧ 18 * Real.pi = (Real.pi / 2) * (R^2 - r^2) ∧ (R^2 - r^2 = 144) :=
sorry

end area_of_region_between_semicircles_l120_120032


namespace percentage_of_students_owning_cats_l120_120587

def total_students : ℕ := 500
def students_with_cats : ℕ := 75

theorem percentage_of_students_owning_cats (total_students students_with_cats : ℕ) (h_total: total_students = 500) (h_cats: students_with_cats = 75) :
  100 * (students_with_cats / total_students : ℝ) = 15 := by
  sorry

end percentage_of_students_owning_cats_l120_120587


namespace second_number_added_is_5_l120_120125

theorem second_number_added_is_5
  (x : ℕ) (h₁ : x = 3)
  (y : ℕ)
  (h₂ : (x + 1) * (x + 13) = (x + y) * (x + y)) :
  y = 5 :=
sorry

end second_number_added_is_5_l120_120125


namespace rhombus_area_2400_l120_120229

noncomputable def area_of_rhombus (x y : ℝ) : ℝ :=
  2 * x * y

theorem rhombus_area_2400 (x y : ℝ) 
  (hx : x = 15) 
  (hy : y = (16 / 3) * x) 
  (rx : 18.75 * 4 * x * y = x * y * (78.75)) 
  (ry : 50 * 4 * x * y = x * y * (200)) : 
  area_of_rhombus 15 80 = 2400 :=
by
  sorry

end rhombus_area_2400_l120_120229


namespace like_terms_monomials_l120_120769

theorem like_terms_monomials (a b : ℕ) : (5 * (m^8) * (n^6) = -(3/4) * (m^(2*a)) * (n^(2*b))) → (a = 4 ∧ b = 3) := by
  sorry

end like_terms_monomials_l120_120769


namespace odd_number_diff_squares_unique_l120_120828

theorem odd_number_diff_squares_unique (n : ℕ) (h : 0 < n) : 
  ∃! (x y : ℤ), (2 * n + 1) = x^2 - y^2 :=
by {
  sorry
}

end odd_number_diff_squares_unique_l120_120828


namespace store_owner_loss_percentage_l120_120318

theorem store_owner_loss_percentage :
  ∀ (initial_value : ℝ) (profit_margin : ℝ) (loss1 : ℝ) (loss2 : ℝ) (loss3 : ℝ) (tax_rate : ℝ),
    initial_value = 100 → profit_margin = 0.10 → loss1 = 0.20 → loss2 = 0.30 → loss3 = 0.25 → tax_rate = 0.12 →
      ((initial_value - initial_value * (1 - loss1) * (1 - loss2) * (1 - loss3)) / initial_value * 100) = 58 :=
by
  intros initial_value profit_margin loss1 loss2 loss3 tax_rate h_initial_value h_profit_margin h_loss1 h_loss2 h_loss3 h_tax_rate
  -- Variable assignments as per given conditions
  have h1 : initial_value = 100 := h_initial_value
  have h2 : profit_margin = 0.10 := h_profit_margin
  have h3 : loss1 = 0.20 := h_loss1
  have h4 : loss2 = 0.30 := h_loss2
  have h5 : loss3 = 0.25 := h_loss3
  have h6 : tax_rate = 0.12 := h_tax_rate
  
  sorry

end store_owner_loss_percentage_l120_120318


namespace remainder_when_x_squared_div_30_l120_120676

theorem remainder_when_x_squared_div_30 (x : ℤ) 
  (h1 : 5 * x ≡ 15 [ZMOD 30]) 
  (h2 : 7 * x ≡ 13 [ZMOD 30]) : 
  (x^2) % 30 = 21 := 
by 
  sorry

end remainder_when_x_squared_div_30_l120_120676


namespace Rachel_painting_time_l120_120472

noncomputable def Matt_time : ℕ := 12
noncomputable def Patty_time (Matt_time : ℕ) : ℕ := Matt_time / 3
noncomputable def Rachel_time (Patty_time : ℕ) : ℕ := 5 + 2 * Patty_time

theorem Rachel_painting_time : Rachel_time (Patty_time Matt_time) = 13 := by
  sorry

end Rachel_painting_time_l120_120472


namespace solve_floor_equation_l120_120927

theorem solve_floor_equation (x : ℝ) (h : ⌊x * ⌊x⌋⌋ = 20) : 5 ≤ x ∧ x < 5.25 := by
  sorry

end solve_floor_equation_l120_120927


namespace profit_percentage_is_20_l120_120858

noncomputable def selling_price : ℝ := 200
noncomputable def cost_price : ℝ := 166.67
noncomputable def profit : ℝ := selling_price - cost_price

theorem profit_percentage_is_20 :
  (profit / cost_price) * 100 = 20 := by
  sorry

end profit_percentage_is_20_l120_120858


namespace fraction_of_students_received_B_l120_120244

theorem fraction_of_students_received_B {total_students : ℝ}
  (fraction_A : ℝ)
  (fraction_A_or_B : ℝ)
  (h_fraction_A : fraction_A = 0.7)
  (h_fraction_A_or_B : fraction_A_or_B = 0.9) :
  fraction_A_or_B - fraction_A = 0.2 :=
by
  rw [h_fraction_A, h_fraction_A_or_B]
  sorry

end fraction_of_students_received_B_l120_120244


namespace geometric_sequence_analogy_l120_120864

variables {a_n b_n : ℕ → ℕ} {S T : ℕ → ℕ}

-- Conditions for the arithmetic sequence
def is_arithmetic_sequence_sum (S : ℕ → ℕ) :=
  S 8 - S 4 = 2 * (S 4) ∧ S 12 - S 8 = 2 * (S 8 - S 4)

-- Conditions for the geometric sequence
def is_geometric_sequence_product (T : ℕ → ℕ) :=
  (T 8 / T 4) = (T 4) ∧ (T 12 / T 8) = (T 8 / T 4)

-- Statement of the proof problem
theorem geometric_sequence_analogy
  (h_arithmetic : is_arithmetic_sequence_sum S)
  (h_geometric_nil : is_geometric_sequence_product T) :
  T 4 / T 4 = 1 ∧
  (T 8 / T 4) / (T 8 / T 4) = 1 ∧
  (T 12 / T 8) / (T 12 / T 8) = 1 := 
by
  sorry

end geometric_sequence_analogy_l120_120864


namespace find_number_l120_120700

theorem find_number (x : ℤ) (h : 4 * x = 28) : x = 7 :=
sorry

end find_number_l120_120700


namespace percentage_increase_l120_120939

-- defining the given values
def Z := 150
def total := 555
def x_from_y (Y : ℝ) := 1.25 * Y

-- defining the condition that x gets 25% more than y and z out of 555 is Rs. 150
def condition1 (X Y : ℝ) := X = x_from_y Y
def condition2 (X Y : ℝ) := X + Y + Z = total

-- theorem to prove
theorem percentage_increase (Y : ℝ) :
  condition1 (x_from_y Y) Y →
  condition2 (x_from_y Y) Y →
  ((Y - Z) / Z) * 100 = 20 :=
by
  sorry

end percentage_increase_l120_120939


namespace g_of_5_l120_120184

noncomputable def g : ℝ → ℝ := sorry

theorem g_of_5 :
  (∀ x y : ℝ, x * g y = y * g x) →
  g 20 = 30 →
  g 5 = 7.5 :=
by
  intros h1 h2
  sorry

end g_of_5_l120_120184


namespace count_coin_distributions_l120_120914

-- Mathematical conditions
def coin_denominations : Finset ℕ := {1, 2, 3, 5}
def number_of_boys : ℕ := 6

-- Theorem statement
theorem count_coin_distributions : (coin_denominations.card ^ number_of_boys) = 4096 :=
by
  sorry

end count_coin_distributions_l120_120914


namespace length_reduction_by_50_percent_l120_120329

variable (L B L' : ℝ)

def rectangle_dimension_change (L B : ℝ) (perc_area_change : ℝ) (new_breadth_factor : ℝ) : Prop :=
  let original_area := L * B
  let new_breadth := new_breadth_factor * B
  let new_area := L' * new_breadth
  let expected_new_area := (1 + perc_area_change) * original_area
  new_area = expected_new_area

theorem length_reduction_by_50_percent (L B : ℝ) (h1: rectangle_dimension_change L B L' 0.5 3) : 
  L' = 0.5 * L :=
by
  unfold rectangle_dimension_change at h1
  simp at h1
  sorry

end length_reduction_by_50_percent_l120_120329


namespace missing_digit_divisibility_by_13_l120_120185

theorem missing_digit_divisibility_by_13 (B : ℕ) (H : 0 ≤ B ∧ B ≤ 9) : 
  (13 ∣ (200 + 10 * B + 5)) ↔ B = 12 :=
by sorry

end missing_digit_divisibility_by_13_l120_120185


namespace total_number_of_questions_l120_120765

theorem total_number_of_questions (type_a_problems type_b_problems : ℕ) 
(time_spent_type_a time_spent_type_b : ℕ) 
(total_exam_time : ℕ) 
(h1 : type_a_problems = 50) 
(h2 : time_spent_type_a = 2 * time_spent_type_b) 
(h3 : time_spent_type_a * type_a_problems = 72) 
(h4 : total_exam_time = 180) :
type_a_problems + type_b_problems = 200 := 
by
  sorry

end total_number_of_questions_l120_120765


namespace sum_of_consecutive_integers_420_l120_120691

theorem sum_of_consecutive_integers_420 : 
  ∃ (k n : ℕ) (h1 : k ≥ 2) (h2 : k * n + k * (k - 1) / 2 = 420), 
  ∃ K : Finset ℕ, K.card = 6 ∧ (∀ x ∈ K, k = x) :=
by
  sorry

end sum_of_consecutive_integers_420_l120_120691


namespace cross_square_side_length_l120_120409

theorem cross_square_side_length (A : ℝ) (s : ℝ) (h1 : A = 810) 
(h2 : (2 * (s / 2)^2 + 2 * (s / 4)^2) = A) : s = 36 := by
  sorry

end cross_square_side_length_l120_120409


namespace right_triangle_even_or_odd_l120_120162

theorem right_triangle_even_or_odd (a b c : ℕ) (ha : Even a ∨ Odd a) (hb : Even b ∨ Odd b) (h : a^2 + b^2 = c^2) : 
  Even c ∨ (Even a ∧ Odd b) ∨ (Odd a ∧ Even b) :=
by
  sorry

end right_triangle_even_or_odd_l120_120162


namespace polynomial_remainder_l120_120847

theorem polynomial_remainder (x : ℤ) :
  let dividend := 3*x^3 - 2*x^2 - 23*x + 60
  let divisor := x - 4
  let quotient := 3*x^2 + 10*x + 17
  let remainder := 128
  dividend = divisor * quotient + remainder :=
by 
  -- proof steps would go here, but we use "sorry" as instructed
  sorry

end polynomial_remainder_l120_120847


namespace max_rectangle_area_l120_120692

theorem max_rectangle_area (x y : ℝ) (h : 2 * x + 2 * y = 60) : x * y ≤ 225 :=
sorry

end max_rectangle_area_l120_120692


namespace side_of_rhombus_l120_120895

variable (d : ℝ) (K : ℝ) 

-- Conditions
def shorter_diagonal := d
def longer_diagonal := 3 * d
def area_rhombus := K = (1 / 2) * d * (3 * d)

-- Proof Statement
theorem side_of_rhombus (h1 : K = (3 / 2) * d^2) : (∃ s : ℝ, s = Real.sqrt (5 * K / 3)) := 
  sorry

end side_of_rhombus_l120_120895


namespace sally_orange_balloons_l120_120085

def initial_orange_balloons : ℝ := 9.0
def found_orange_balloons : ℝ := 2.0

theorem sally_orange_balloons :
  initial_orange_balloons + found_orange_balloons = 11.0 := 
by
  sorry

end sally_orange_balloons_l120_120085


namespace order_numbers_l120_120411

theorem order_numbers (a b c : ℕ) (h1 : a = 8^10) (h2 : b = 4^15) (h3 : c = 2^31) : b = a ∧ a < c :=
by {
  sorry
}

end order_numbers_l120_120411


namespace ratio_expression_l120_120495

theorem ratio_expression (A B C : ℚ) (h : A / B = 3 / 1 ∧ B / C = 1 / 6) : 
  (4 * A + 3 * B) / (5 * C - 2 * A) = 5 / 8 :=
by sorry

end ratio_expression_l120_120495


namespace percentage_difference_l120_120206

theorem percentage_difference : (0.4 * 60 - (4/5 * 25)) = 4 := by
  sorry

end percentage_difference_l120_120206


namespace average_age_of_boys_l120_120644

theorem average_age_of_boys
  (N : ℕ) (G : ℕ) (A_G : ℕ) (A_S : ℚ) (B : ℕ)
  (hN : N = 652)
  (hG : G = 163)
  (hA_G : A_G = 11)
  (hA_S : A_S = 11.75)
  (hB : B = N - G) :
  (163 * 11 + 489 * x = 11.75 * 652) → x = 12 := by
  sorry

end average_age_of_boys_l120_120644


namespace find_ax5_by5_l120_120799

variable (a b x y : ℝ)

theorem find_ax5_by5 
  (h1 : a * x + b * y = 3)
  (h2 : a * x^2 + b * y^2 = 7)
  (h3 : a * x^3 + b * y^3 = 16)
  (h4 : a * x^4 + b * y^4 = 42) :
  a * x^5 + b * y^5 = 20 := 
sorry

end find_ax5_by5_l120_120799


namespace problem_solution_l120_120440

theorem problem_solution (a b : ℝ) (ha : |a| = 5) (hb : b = -3) :
  a + b = 2 ∨ a + b = -8 :=
by sorry

end problem_solution_l120_120440


namespace line_passes_through_point_l120_120323

-- We declare the variables for the real numbers a, b, and c
variables (a b c : ℝ)

-- We state the condition that a + b - c = 0
def condition1 : Prop := a + b - c = 0

-- We state the condition that not all of a, b, c are zero
def condition2 : Prop := ¬ (a = 0 ∧ b = 0 ∧ c = 0)

-- We state the theorem: the line ax + by + c = 0 passes through the point (-1, -1)
theorem line_passes_through_point (h1 : condition1 a b c) (h2 : condition2 a b c) :
  a * (-1) + b * (-1) + c = 0 := sorry

end line_passes_through_point_l120_120323


namespace skateboard_total_distance_is_3720_l120_120023

noncomputable def skateboard_distance : ℕ :=
  let a1 := 10
  let d := 9
  let n := 20
  let flat_time := 10
  let a_n := a1 + (n - 1) * d
  let ramp_distance := n * (a1 + a_n) / 2
  let flat_distance := a_n * flat_time
  ramp_distance + flat_distance

theorem skateboard_total_distance_is_3720 : skateboard_distance = 3720 := 
by
  sorry

end skateboard_total_distance_is_3720_l120_120023


namespace salary_reduction_percentage_l120_120872

theorem salary_reduction_percentage
  (S : ℝ) 
  (h : S * (1 - R / 100) = S / 1.388888888888889): R = 28 :=
sorry

end salary_reduction_percentage_l120_120872


namespace Janet_sold_six_action_figures_l120_120566

variable {x : ℕ}

theorem Janet_sold_six_action_figures
  (h₁ : 10 - x + 4 + 2 * (10 - x + 4) = 24) :
  x = 6 :=
by
  sorry

end Janet_sold_six_action_figures_l120_120566


namespace pears_picked_l120_120043

def Jason_pears : ℕ := 46
def Keith_pears : ℕ := 47
def Mike_pears : ℕ := 12
def total_pears : ℕ := 105

theorem pears_picked :
  Jason_pears + Keith_pears + Mike_pears = total_pears :=
by
  exact rfl

end pears_picked_l120_120043


namespace part1_part2_l120_120241

variable {x : ℝ}

/-- Prove that the range of the function f(x) = (sqrt(1+x) + sqrt(1-x) + 2) * (sqrt(1-x^2) + 1) for 0 ≤ x ≤ 1 is (0, 8]. -/
theorem part1 (hx : 0 ≤ x ∧ x ≤ 1) :
  0 < ((Real.sqrt (1 + x) + Real.sqrt (1 - x) + 2) * (Real.sqrt (1 - x^2) + 1)) ∧ 
  ((Real.sqrt (1 + x) + Real.sqrt (1 - x) + 2) * (Real.sqrt (1 - x^2) + 1)) ≤ 8 :=
sorry

/-- Prove that for 0 ≤ x ≤ 1, there exists a positive number β such that sqrt(1+x) + sqrt(1-x) ≤ 2 - x^2 / β, with the minimal β = 4. -/
theorem part2 (hx : 0 ≤ x ∧ x ≤ 1) :
  ∃ β : ℝ, β > 0 ∧ β = 4 ∧ (Real.sqrt (1 + x) + Real.sqrt (1 - x) ≤ 2 - x^2 / β) :=
sorry

end part1_part2_l120_120241


namespace hyperbola_foci_distance_l120_120798

theorem hyperbola_foci_distance :
  (∀ (x y : ℝ), (y = 2 * x + 3) ∨ (y = -2 * x + 7)) →
  (∃ (x y : ℝ), x = 4 ∧ y = 5 ∧ ((y = 2 * x + 3) ∨ (y = -2 * x + 7))) →
  (∃ h : ℝ, h = 6 * Real.sqrt 2) :=
by
  sorry

end hyperbola_foci_distance_l120_120798


namespace find_value_of_complex_fraction_l120_120650

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem find_value_of_complex_fraction :
  (1 - 2 * i) / (1 + i) = -1 / 2 - 3 / 2 * i := 
sorry

end find_value_of_complex_fraction_l120_120650


namespace proportion_correct_l120_120681

theorem proportion_correct {a b : ℝ} (h : 2 * a = 5 * b) : a / 5 = b / 2 :=
by {
  sorry
}

end proportion_correct_l120_120681


namespace factorize_expression_l120_120494

theorem factorize_expression (a x y : ℝ) :
  a * x^2 + 2 * a * x * y + a * y^2 = a * (x + y)^2 :=
by
  sorry

end factorize_expression_l120_120494


namespace liquid_in_cylinders_l120_120334

theorem liquid_in_cylinders (n : ℕ) (a : ℝ) (h1 : 2 ≤ n) :
  (∃ x : ℕ → ℝ, ∀ (k : ℕ), (1 ≤ k ∧ k ≤ n) → 
    (if k = 1 then 
      x k = a * n * (n - 2) / (n - 1) ^ 2 
    else if k = 2 then 
      x k = a * (n^2 - 2*n + 2) / (n - 1) ^ 2 
    else 
      x k = a)) :=
sorry

end liquid_in_cylinders_l120_120334


namespace num_perfect_square_factors_of_450_l120_120884

theorem num_perfect_square_factors_of_450 :
  ∃ n : ℕ, n = 4 ∧ ∀ d : ℕ, d ∣ 450 → (∃ k : ℕ, d = k * k) → d = 1 ∨ d = 25 ∨ d = 9 ∨ d = 225 :=
by
  sorry

end num_perfect_square_factors_of_450_l120_120884


namespace find_prime_triplets_l120_120031

def is_prime (n : ℕ) : Prop := Nat.Prime n

def valid_triplet (p q r : ℕ) : Prop :=
  is_prime p ∧ is_prime q ∧ is_prime r ∧ (p * (r + 1) = q * (r + 5))

theorem find_prime_triplets :
  { (p, q, r) | valid_triplet p q r } = {(3, 2, 7), (5, 3, 5), (7, 3, 2)} :=
by {
  sorry -- Proof is to be completed
}

end find_prime_triplets_l120_120031


namespace number_of_bad_carrots_l120_120897

-- Definitions for conditions
def olivia_picked : ℕ := 20
def mother_picked : ℕ := 14
def good_carrots : ℕ := 19

-- Sum of total carrots picked
def total_carrots : ℕ := olivia_picked + mother_picked

-- Theorem stating the number of bad carrots
theorem number_of_bad_carrots : total_carrots - good_carrots = 15 :=
by
  sorry

end number_of_bad_carrots_l120_120897


namespace bottles_per_case_l120_120706

theorem bottles_per_case (total_bottles_per_day : ℕ) (cases_required : ℕ) (bottles_per_case : ℕ)
  (h1 : total_bottles_per_day = 65000)
  (h2 : cases_required = 5000) :
  bottles_per_case = total_bottles_per_day / cases_required :=
by
  sorry

end bottles_per_case_l120_120706


namespace find_prime_pairs_l120_120084

open Nat

-- Define a function to check if a number is prime
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the problem as a theorem in Lean
theorem find_prime_pairs :
  ∀ (p n : ℕ), is_prime p ∧ n > 0 ∧ p^3 - 2*p^2 + p + 1 = 3^n ↔ (p = 2 ∧ n = 1) ∨ (p = 5 ∧ n = 4) :=
by
  sorry

end find_prime_pairs_l120_120084


namespace Sam_memorized_more_digits_l120_120678

variable (MinaDigits SamDigits CarlosDigits : ℕ)
variable (h1 : MinaDigits = 6 * CarlosDigits)
variable (h2 : MinaDigits = 24)
variable (h3 : SamDigits = 10)
 
theorem Sam_memorized_more_digits :
  SamDigits - CarlosDigits = 6 :=
by
  -- Let's unfold the statements and perform basic arithmetic.
  sorry

end Sam_memorized_more_digits_l120_120678


namespace quadrilateral_interior_angle_not_greater_90_l120_120951

-- Definition of the quadrilateral interior angle property
def quadrilateral_interior_angles := ∀ (a b c d : ℝ), (a + b + c + d = 360) → (a > 90 → b > 90 → c > 90 → d > 90 → false)

-- Proposition: There is at least one interior angle in a quadrilateral that is not greater than 90 degrees.
theorem quadrilateral_interior_angle_not_greater_90 :
  (∀ (a b c d : ℝ), (a + b + c + d = 360) → (a > 90 ∧ b > 90 ∧ c > 90 ∧ d > 90) → false) →
  (∃ (a b c d : ℝ), a + b + c + d = 360 ∧ (a ≤ 90 ∨ b ≤ 90 ∨ c ≤ 90 ∨ d ≤ 90)) :=
sorry

end quadrilateral_interior_angle_not_greater_90_l120_120951


namespace equilateral_triangle_l120_120717

namespace TriangleEquilateral

-- Define the structure of a triangle and given conditions
structure Triangle :=
  (A B C : ℝ)  -- vertices
  (angleA : ℝ) -- angle at vertex A
  (sideBC : ℝ) -- length of side BC
  (perimeter : ℝ)  -- perimeter of the triangle

-- Define the proof problem
theorem equilateral_triangle (T : Triangle) (h1 : T.angleA = 60)
  (h2 : T.sideBC = T.perimeter / 3) : 
  T.A = T.B ∧ T.B = T.C ∧ T.A = T.C ∧ T.A = T.B ∧ T.B = T.C ∧ T.A = T.C :=
  sorry

end TriangleEquilateral

end equilateral_triangle_l120_120717


namespace product_of_roots_l120_120458

theorem product_of_roots (x : ℝ) (h : x + 16 / x = 12) : (8 : ℝ) * (4 : ℝ) = 32 :=
by
  -- Your proof would go here
  sorry

end product_of_roots_l120_120458


namespace problem_statement_l120_120958

def f (x : ℝ) : ℝ := |x - 1| - |x|

theorem problem_statement : f (f (1/2)) = 1 :=
by
    sorry

end problem_statement_l120_120958


namespace ordering_PQR_l120_120946

noncomputable def P := Real.sqrt 2
noncomputable def Q := Real.sqrt 7 - Real.sqrt 3
noncomputable def R := Real.sqrt 6 - Real.sqrt 2

theorem ordering_PQR : P > R ∧ R > Q := by
  sorry

end ordering_PQR_l120_120946


namespace find_original_integer_l120_120763

theorem find_original_integer (a b c d : ℕ) 
    (h1 : (b + c + d) / 3 + 10 = 37) 
    (h2 : (a + c + d) / 3 + 10 = 31) 
    (h3 : (a + b + d) / 3 + 10 = 25) 
    (h4 : (a + b + c) / 3 + 10 = 19) : 
    d = 45 := 
    sorry

end find_original_integer_l120_120763


namespace line_slope_intercept_through_points_l120_120356

theorem line_slope_intercept_through_points (a b : ℝ) :
  (∀ x y : ℝ, (x, y) = (3, 7) ∨ (x, y) = (7, 19) → y = a * x + b) →
  a - b = 5 :=
by
  sorry

end line_slope_intercept_through_points_l120_120356


namespace distance_between_bakery_and_butcher_shop_l120_120412

variables (v1 v2 : ℝ) -- speeds of the butcher's and baker's son respectively
variables (x : ℝ) -- distance covered by the baker's son by the time they meet
variable (distance : ℝ) -- distance between the bakery and the butcher shop

-- Given conditions
def butcher_walks_500_more := x + 0.5
def butcher_time_left := 10 / 60
def baker_time_left := 22.5 / 60

-- Equivalent relationships
def v1_def := v1 = 6 * x
def v2_def := v2 = (8/3) * (x + 0.5)

-- Final proof problem
theorem distance_between_bakery_and_butcher_shop :
  (x + 0.5 + x) = 2.5 :=
sorry

end distance_between_bakery_and_butcher_shop_l120_120412


namespace number_is_209_given_base_value_is_100_l120_120735

theorem number_is_209_given_base_value_is_100 (n : ℝ) (base_value : ℝ) (H : base_value = 100) (percentage : ℝ) (H1 : percentage = 2.09) : n = 209 :=
by
  sorry

end number_is_209_given_base_value_is_100_l120_120735


namespace no_valid_k_values_l120_120353

open Nat

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

noncomputable def roots_are_primes (k : ℕ) : Prop :=
  ∃ p q : ℕ, is_prime p ∧ is_prime q ∧ p + q = 57 ∧ p * q = k

theorem no_valid_k_values : ∀ k : ℕ, ¬ roots_are_primes k := by
  sorry

end no_valid_k_values_l120_120353


namespace area_of_triangle_ABC_is_1_l120_120391

-- Define the vertices A, B, and C
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (2, 1)
def C : ℝ × ℝ := (2, 1)

-- Define the function to compute the area of the triangle given three vertices
def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  0.5 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

-- The main theorem to prove that the area of triangle ABC is 1
theorem area_of_triangle_ABC_is_1 : triangle_area A B C = 1 := 
by
  sorry

end area_of_triangle_ABC_is_1_l120_120391


namespace odd_even_shift_composition_l120_120100

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

def is_even_function_shifted (f : ℝ → ℝ) (shift : ℝ) : Prop :=
  ∀ x : ℝ, f (x + shift) = f (-x + shift)

theorem odd_even_shift_composition
  (f : ℝ → ℝ)
  (h_odd : is_odd_function f)
  (h_even_shift : is_even_function_shifted f 3)
  (h_f1 : f 1 = 1) :
  f 6 + f 11 = -1 := by
  sorry

end odd_even_shift_composition_l120_120100


namespace outlet_two_rate_l120_120651

/-- Definitions and conditions for the problem -/
def tank_volume_feet : ℝ := 20
def inlet_rate_cubic_inches_per_min : ℝ := 5
def outlet_one_rate_cubic_inches_per_min : ℝ := 9
def empty_time_minutes : ℝ := 2880
def cubic_feet_to_cubic_inches : ℝ := 1728
def tank_volume_cubic_inches := tank_volume_feet * cubic_feet_to_cubic_inches

/-- Statement to prove the rate of the other outlet pipe -/
theorem outlet_two_rate (x : ℝ) :
  tank_volume_cubic_inches / empty_time_minutes = outlet_one_rate_cubic_inches_per_min + x - inlet_rate_cubic_inches_per_min → 
  x = 8 :=
by
  sorry

end outlet_two_rate_l120_120651


namespace f_zero_eq_f_expression_alpha_value_l120_120470

noncomputable def f (ω x : ℝ) : ℝ :=
  3 * Real.sin (ω * x + Real.pi / 6)

theorem f_zero_eq (ω : ℝ) (hω : ω > 0) (h_period : (2 * Real.pi / ω) = Real.pi / 2) :
  f ω 0 = 3 / 2 :=
by
  sorry

theorem f_expression (ω : ℝ) (hω : ω > 0) (h_period : (2 * Real.pi / ω) = Real.pi / 2) :
  ∀ x : ℝ, f ω x = f 4 x :=
by
  sorry

theorem alpha_value (f_4 : ℝ → ℝ) (α : ℝ) (hα : α ∈ Set.Ioo 0 (Real.pi / 2))
  (h_f4 : ∀ x : ℝ, f_4 x = 3 * Real.sin (4 * x + Real.pi / 6)) (h_fα : f_4 (α / 2) = 3 / 2) :
  α = Real.pi / 3 :=
by
  sorry

end f_zero_eq_f_expression_alpha_value_l120_120470


namespace find_a1_a7_l120_120077

-- Definitions based on the problem conditions
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

def a_3_5_condition (a : ℕ → ℝ) : Prop :=
  a 3 + a 5 = -6

def a_2_6_condition (a : ℕ → ℝ) : Prop :=
  a 2 * a 6 = 8

-- The theorem we need to prove
theorem find_a1_a7 (a : ℕ → ℝ) (ha : is_geometric_sequence a) (h35 : a_3_5_condition a) (h26 : a_2_6_condition a) :
  a 1 + a 7 = -9 :=
sorry

end find_a1_a7_l120_120077


namespace sqrt_ab_equals_sqrt_2_l120_120488

theorem sqrt_ab_equals_sqrt_2 
  (a b : ℝ)
  (h1 : a ^ 2 = 16 / 25)
  (h2 : b ^ 3 = 125 / 8) : 
  Real.sqrt (a * b) = Real.sqrt 2 := 
by 
  -- proof will go here
  sorry

end sqrt_ab_equals_sqrt_2_l120_120488


namespace maximum_and_minimum_values_l120_120441

noncomputable def f (p q x : ℝ) : ℝ := x^3 - p * x^2 - q * x

theorem maximum_and_minimum_values
  (p q : ℝ)
  (h1 : f p q 1 = 0)
  (h2 : (deriv (f p q)) 1 = 0) :
  ∃ (max_val min_val : ℝ), max_val = 4 / 27 ∧ min_val = 0 := 
by {
  sorry
}

end maximum_and_minimum_values_l120_120441


namespace driver_average_speed_l120_120079

theorem driver_average_speed (v t : ℝ) (h1 : ∀ d : ℝ, d = v * t → (d / (v + 10)) = (3 / 4) * t) : v = 30 := by
  sorry

end driver_average_speed_l120_120079


namespace johns_total_spending_l120_120756

theorem johns_total_spending
    (online_phone_price : ℝ := 2000)
    (phone_price_increase : ℝ := 0.02)
    (phone_case_price : ℝ := 35)
    (screen_protector_price : ℝ := 15)
    (accessories_discount : ℝ := 0.05)
    (sales_tax : ℝ := 0.06) :
    let store_phone_price := online_phone_price * (1 + phone_price_increase)
    let regular_accessories_price := phone_case_price + screen_protector_price
    let discounted_accessories_price := regular_accessories_price * (1 - accessories_discount)
    let pre_tax_total := store_phone_price + discounted_accessories_price
    let total_spending := pre_tax_total * (1 + sales_tax)
    total_spending = 2212.75 :=
by
    sorry

end johns_total_spending_l120_120756


namespace binary_addition_correct_l120_120558

-- define the binary numbers as natural numbers using their binary representations
def bin_1010 : ℕ := 0b1010
def bin_10 : ℕ := 0b10
def bin_sum : ℕ := 0b1100

-- state the theorem that needs to be proved
theorem binary_addition_correct : bin_1010 + bin_10 = bin_sum := by
  sorry

end binary_addition_correct_l120_120558


namespace number_of_solutions_l120_120945

theorem number_of_solutions :
  ∃ (sols : Finset (ℝ × ℝ × ℝ × ℝ)), 
  (∀ (x y z w : ℝ), ((x, y, z, w) ∈ sols) ↔ (x = z + w + z * w * x ∧ y = w + x + w * x * y ∧ z = x + y + x * y * z ∧ w = y + z + y * z * w ∧ x * y + y * z + z * w + w * x = 2)) ∧ 
  sols.card = 5 :=
sorry

end number_of_solutions_l120_120945


namespace minimum_cans_needed_l120_120358

theorem minimum_cans_needed (h : ∀ c, c * 10 ≥ 120) : ∃ c, c = 12 :=
by
  sorry

end minimum_cans_needed_l120_120358


namespace value_of_5_l120_120218

def q' (q : ℤ) : ℤ := 3 * q - 3

theorem value_of_5'_prime : q' (q' 5) = 33 :=
by
  sorry

end value_of_5_l120_120218


namespace max_n_for_coloring_l120_120553

noncomputable def maximum_n : ℕ :=
  11

theorem max_n_for_coloring :
  ∃ n : ℕ, (n = maximum_n) ∧ ∀ k ∈ Finset.range n, 
  (∃ x y : ℕ, 1 ≤ x ∧ x ≤ 14 ∧ 1 ≤ y ∧ y ≤ 14 ∧ (x - y = k ∨ y - x = k) ∧ x ≠ y) ∧
  (∃ a b : ℕ, 1 ≤ a ∧ a ≤ 14 ∧ 1 ≤ b ∧ b ≤ 14 ∧ (a - b = k ∨ b - a = k) ∧ a ≠ b) :=
sorry

end max_n_for_coloring_l120_120553


namespace determine_sold_cakes_l120_120978

def initial_cakes := 121
def new_cakes := 170
def remaining_cakes := 186
def sold_cakes (S : ℕ) : Prop := initial_cakes - S + new_cakes = remaining_cakes

theorem determine_sold_cakes : ∃ S, sold_cakes S ∧ S = 105 :=
by
  use 105
  unfold sold_cakes
  simp
  sorry

end determine_sold_cakes_l120_120978


namespace total_space_compacted_l120_120795

-- Definitions according to the conditions
def num_cans : ℕ := 60
def space_per_can_before : ℝ := 30
def compaction_rate : ℝ := 0.20

-- Theorem statement
theorem total_space_compacted : num_cans * (space_per_can_before * compaction_rate) = 360 := by
  sorry

end total_space_compacted_l120_120795


namespace three_pow_2023_mod_17_l120_120325

theorem three_pow_2023_mod_17 : (3 ^ 2023) % 17 = 7 := by
  sorry

end three_pow_2023_mod_17_l120_120325


namespace find_c_d_l120_120956

theorem find_c_d (y c d : ℕ) (H1 : y = c + Real.sqrt d) (H2 : y^2 + 4 * y + 4 / y + 1 / (y^2) = 30) :
  c + d = 5 :=
sorry

end find_c_d_l120_120956


namespace exist_indices_with_non_decreasing_subsequences_l120_120393

theorem exist_indices_with_non_decreasing_subsequences
  (a b c : ℕ → ℕ) :
  (∀ n m : ℕ, n < m → ∃ p q : ℕ, q < p ∧ 
    a p ≥ a q ∧ 
    b p ≥ b q ∧ 
    c p ≥ c q) :=
  sorry

end exist_indices_with_non_decreasing_subsequences_l120_120393


namespace selling_price_correct_l120_120047

variable (CostPrice GainPercent : ℝ)
variables (Profit SellingPrice : ℝ)

noncomputable def calculateProfit : ℝ := (GainPercent / 100) * CostPrice

noncomputable def calculateSellingPrice : ℝ := CostPrice + calculateProfit CostPrice GainPercent

theorem selling_price_correct 
  (h1 : CostPrice = 900) 
  (h2 : GainPercent = 30)
  : calculateSellingPrice CostPrice GainPercent = 1170 := by
  sorry

end selling_price_correct_l120_120047


namespace find_max_number_l120_120985

noncomputable def increasing_sequence (a : ℕ → ℝ) := ∀ n m, n < m → a n < a m

noncomputable def arithmetic_progression (a : ℕ → ℝ) (d : ℝ) (n : ℕ) := 
  (a n + d = a (n+1)) ∧ (a (n+1) + d = a (n+2)) ∧ (a (n+2) + d = a (n+3))

noncomputable def geometric_progression (a : ℕ → ℝ) (r : ℝ) (n : ℕ) := 
  (a (n+1) = a n * r) ∧ (a (n+2) = a (n+1) * r) ∧ (a (n+3) = a (n+2) * r)

theorem find_max_number (a : ℕ → ℝ):
  increasing_sequence a → 
  (∃ n, arithmetic_progression a 4 n) →
  (∃ n, arithmetic_progression a 36 n) →
  (∃ n, geometric_progression a (a (n+1) / a n) n) →
  a 7 = 126 := sorry

end find_max_number_l120_120985


namespace triangle_area_l120_120984

namespace MathProof

theorem triangle_area (y_eq_6 y_eq_2_plus_x y_eq_2_minus_x : ℝ → ℝ)
  (h1 : ∀ x, y_eq_6 x = 6)
  (h2 : ∀ x, y_eq_2_plus_x x = 2 + x)
  (h3 : ∀ x, y_eq_2_minus_x x = 2 - x) :
  let a := (4, 6)
  let b := (-4, 6)
  let c := (0, 2)
  let base := dist a b
  let height := (6 - 2:ℝ)
  (1 / 2 * base * height = 16) := by
    sorry

end MathProof

end triangle_area_l120_120984


namespace find_m_l120_120315

theorem find_m (m : ℝ) (a a1 a2 a3 a4 a5 a6 : ℝ) 
  (h1 : (1 + m)^6 = a + a1 + a2 + a3 + a4 + a5 + a6) 
  (h2 : a1 + a2 + a3 + a4 + a5 + a6 = 63)
  (h3 : a = 1) : m = 1 ∨ m = -3 := 
by
  sorry

end find_m_l120_120315


namespace least_prime_factor_of_5pow6_minus_5pow4_l120_120626

def least_prime_factor (n : ℕ) : ℕ :=
  if h : n > 1 then (Nat.minFac n) else 0

theorem least_prime_factor_of_5pow6_minus_5pow4 : least_prime_factor (5^6 - 5^4) = 2 := by
  sorry

end least_prime_factor_of_5pow6_minus_5pow4_l120_120626


namespace equal_profits_at_20000_end_month_more_profit_50000_l120_120360

noncomputable section

-- Define the conditions
def profit_beginning_month (x : ℝ) : ℝ := 0.15 * x + 1.15 * x * 0.1
def profit_end_month (x : ℝ) : ℝ := 0.3 * x - 700

-- Proof Problem 1: Prove that at x = 20000, the profits are equal
theorem equal_profits_at_20000 : profit_beginning_month 20000 = profit_end_month 20000 :=
by
  sorry

-- Proof Problem 2: Prove that at x = 50000, selling at end of month yields more profit than selling at beginning of month
theorem end_month_more_profit_50000 : profit_end_month 50000 > profit_beginning_month 50000 :=
by
  sorry

end equal_profits_at_20000_end_month_more_profit_50000_l120_120360


namespace measure_of_angle_l120_120422

theorem measure_of_angle (x : ℝ) (h1 : 90 = x + (3 * x + 10)) : x = 20 :=
by
  sorry

end measure_of_angle_l120_120422


namespace smallest_n_l120_120021

theorem smallest_n (n : ℕ) (hn : n > 0) (h : 623 * n % 32 = 1319 * n % 32) : n = 4 :=
sorry

end smallest_n_l120_120021


namespace fraction_eq_l120_120840

theorem fraction_eq : (15.5 / (-0.75) : ℝ) = (-62 / 3) := 
by {
  sorry
}

end fraction_eq_l120_120840


namespace total_expenditure_now_l120_120635

-- Define the conditions in Lean
def original_student_count : ℕ := 100
def additional_students : ℕ := 25
def decrease_in_average_expenditure : ℤ := 10
def increase_in_total_expenditure : ℤ := 500

-- Let's denote the original average expenditure per student as A rupees
variable (A : ℤ)

-- Define the old and new expenditures
def original_total_expenditure := original_student_count * A
def new_average_expenditure := A - decrease_in_average_expenditure
def new_total_expenditure := (original_student_count + additional_students) * new_average_expenditure

-- The theorem to prove
theorem total_expenditure_now :
  new_total_expenditure A - original_total_expenditure A = increase_in_total_expenditure →
  new_total_expenditure A = 7500 :=
by
  sorry

end total_expenditure_now_l120_120635


namespace exponent_problem_l120_120126

theorem exponent_problem (a : ℝ) (m n : ℕ) (h1 : a ^ m = 3) (h2 : a ^ n = 2) : a ^ (m - 2 * n) = 3 / 4 := by
  sorry

end exponent_problem_l120_120126


namespace geometric_sequence_value_l120_120918

variable {α : Type*} [LinearOrderedField α] (a : ℕ → α)
variable (r : α)
variable (a_pos : ∀ n, a n > 0)
variable (h1 : a 1 = 2)
variable (h99 : a 99 = 8)
variable (geom_seq : ∀ n, a (n + 1) = r * a n)

theorem geometric_sequence_value :
  a 20 * a 50 * a 80 = 64 := by
  sorry

end geometric_sequence_value_l120_120918


namespace shaded_region_area_l120_120135

-- Conditions given in the problem
def diameter (d : ℝ) := d = 4
def length_feet (l : ℝ) := l = 2

-- Proof statement
theorem shaded_region_area (d l : ℝ) (h1 : diameter d) (h2 : length_feet l) : 
  (l * 12 / d * (d / 2)^2 * π = 24 * π) := by
  sorry

end shaded_region_area_l120_120135


namespace polynomial_sum_of_squares_l120_120002

theorem polynomial_sum_of_squares (P : Polynomial ℝ) (hP : ∀ x : ℝ, 0 < P.eval x) :
  ∃ (U V : Polynomial ℝ), P = U^2 + V^2 := 
by
  sorry

end polynomial_sum_of_squares_l120_120002


namespace fraction_problem_l120_120298

theorem fraction_problem (b : ℕ) (h₀ : 0 < b) (h₁ : (b : ℝ) / (b + 35) = 0.869) : b = 232 := 
by
  sorry

end fraction_problem_l120_120298


namespace geometric_seq_fourth_term_l120_120576

-- Define the conditions
def first_term (a1 : ℝ) : Prop := a1 = 512
def sixth_term (a1 r : ℝ) : Prop := a1 * r^5 = 32

-- Define the claim
def fourth_term (a1 r a4 : ℝ) : Prop := a4 = a1 * r^3

-- State the theorem
theorem geometric_seq_fourth_term :
  ∀ a1 r a4 : ℝ, first_term a1 → sixth_term a1 r → fourth_term a1 r a4 → a4 = 64 :=
by
  intros a1 r a4 h1 h2 h3
  rw [first_term, sixth_term, fourth_term] at *
  sorry

end geometric_seq_fourth_term_l120_120576


namespace unique_polynomial_P_l120_120709

open Polynomial

/-- The only polynomial P with real coefficients such that
    xP(y/x) + yP(x/y) = x + y for all nonzero real numbers x and y 
    is P(x) = x. --/
theorem unique_polynomial_P (P : ℝ[X]) (hP : ∀ x y : ℝ, x ≠ 0 → y ≠ 0 → x * P.eval (y / x) + y * P.eval (x / y) = x + y) :
P = Polynomial.C 1 * X :=
by sorry

end unique_polynomial_P_l120_120709


namespace find_a2023_l120_120584

theorem find_a2023
  (a : ℕ → ℚ)
  (h1 : a 1 = 1)
  (h2 : a 2 = 2/5)
  (h3 : a 3 = 1/4)
  (h_rule : ∀ n : ℕ, 0 < n → (1 / a n + 1 / a (n + 2) = 2 / a (n + 1))) :
  a 2023 = 1 / 3034 :=
by sorry

end find_a2023_l120_120584


namespace volunteer_selection_probability_l120_120905

theorem volunteer_selection_probability :
  ∀ (students total_students remaining_students selected_volunteers : ℕ),
    total_students = 2018 →
    remaining_students = total_students - 18 →
    selected_volunteers = 50 →
    (selected_volunteers : ℚ) / total_students = (25 : ℚ) / 1009 :=
by
  intros students total_students remaining_students selected_volunteers
  intros h1 h2 h3
  sorry

end volunteer_selection_probability_l120_120905


namespace simplify_expression_l120_120369

noncomputable def expr := (-1 : ℝ)^2023 + Real.sqrt 9 - Real.pi^0 + Real.sqrt (1 / 8) * Real.sqrt 32

theorem simplify_expression : expr = 3 := 
by sorry

end simplify_expression_l120_120369


namespace value_of_m_squared_plus_reciprocal_squared_l120_120363

theorem value_of_m_squared_plus_reciprocal_squared 
  (m : ℝ) 
  (h : m + 1/m = 10) :
  m^2 + 1/m^2 + 4 = 102 :=
by {
  sorry
}

end value_of_m_squared_plus_reciprocal_squared_l120_120363


namespace AM_GM_HM_inequality_l120_120741

theorem AM_GM_HM_inequality (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : a ≠ b) : 
  (a + b) / 2 > Real.sqrt (a * b) ∧ Real.sqrt (a * b) > (2 * a * b) / (a + b) := 
sorry

end AM_GM_HM_inequality_l120_120741


namespace simplify_expression_l120_120160

noncomputable def original_expression (x : ℝ) : ℝ :=
(x - 3 * x / (x + 1)) / ((x - 2) / (x^2 + 2 * x + 1))

theorem simplify_expression:
  ∀ (x : ℝ), -1 ≤ x ∧ x ≤ 2 ∧ x ≠ -1 ∧ x ≠ 2 → 
  (original_expression x = x^2 + x) ∧ 
  ((x = 1 → original_expression x = 2) ∧ (x = 0 → original_expression x = 0)) :=
by
  intros
  sorry

end simplify_expression_l120_120160


namespace find_value_of_n_l120_120889

def is_prime (p : ℕ) : Prop := Nat.Prime p

theorem find_value_of_n
  (a b c n : ℕ)
  (ha : is_prime a)
  (hb : is_prime b)
  (hc : is_prime c)
  (h1 : 2 * a + 3 * b = c)
  (h2 : 4 * a + c + 1 = 4 * b)
  (h3 : n = a * b * c)
  (h4 : n < 10000) :
  n = 1118 :=
by
  sorry

end find_value_of_n_l120_120889


namespace minimum_value_expression_l120_120395

noncomputable def minimum_expression (a b c : ℝ) : ℝ :=
  a^2 + 6 * a * b + 9 * b^2 + 3 * c^2

theorem minimum_value_expression (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a * b * c = 27) :
  minimum_expression a b c ≥ 126 :=
by
  sorry

end minimum_value_expression_l120_120395


namespace derivative_of_function_y_l120_120999

noncomputable def function_y (x : ℝ) : ℝ := (x^2) / (x + 3)

theorem derivative_of_function_y (x : ℝ) :
  deriv function_y x = (x^2 + 6 * x) / ((x + 3)^2) :=
by 
  -- sorry since the proof is not required
  sorry

end derivative_of_function_y_l120_120999


namespace zoo_people_l120_120685

def number_of_people (cars : ℝ) (people_per_car : ℝ) : ℝ :=
  cars * people_per_car

theorem zoo_people (h₁ : cars = 3.0) (h₂ : people_per_car = 63.0) :
  number_of_people cars people_per_car = 189.0 :=
by
  rw [h₁, h₂]
  -- multiply the numbers directly after substitution
  norm_num
  -- left this as a placeholder for now, can use calc or norm_num for final steps
  exact sorry

end zoo_people_l120_120685


namespace max_product_l120_120712

theorem max_product (x : ℤ) : x + (2000 - x) = 2000 → x * (2000 - x) ≤ 1000000 :=
by
  sorry

end max_product_l120_120712


namespace ratio_of_green_to_blue_l120_120226

-- Definitions of the areas and the circles
noncomputable def red_area : ℝ := Real.pi * (1 : ℝ) ^ 2
noncomputable def middle_area : ℝ := Real.pi * (2 : ℝ) ^ 2
noncomputable def large_area: ℝ := Real.pi * (3 : ℝ) ^ 2

noncomputable def blue_area : ℝ := middle_area - red_area
noncomputable def green_area : ℝ := large_area - middle_area

-- The proof that the ratio of the green area to the blue area is 5/3
theorem ratio_of_green_to_blue : green_area / blue_area = 5 / 3 := by
  sorry

end ratio_of_green_to_blue_l120_120226


namespace sum_of_squares_of_roots_l120_120489

theorem sum_of_squares_of_roots :
  let a := 5
  let b := -7
  let c := 2
  let x1 := (-b + (b^2 - 4*a*c)^(1/2)) / (2*a)
  let x2 := (-b - (b^2 - 4*a*c)^(1/2)) / (2*a)
  x1^2 + x2^2 = (b^2 - 2*a*c) / a^2 :=
by
  sorry

end sum_of_squares_of_roots_l120_120489


namespace john_must_solve_at_least_17_correct_l120_120967

theorem john_must_solve_at_least_17_correct :
  ∀ (x : ℕ), 25 = 20 + 5 → 7 * x - (20 - x) + 2 * 5 ≥ 120 → x ≥ 17 :=
by
  intros x h1 h2
  -- Remaining steps will be included in the proof
  sorry

end john_must_solve_at_least_17_correct_l120_120967


namespace part1_part2_l120_120080

variable (x y z : ℕ)

theorem part1 (h1 : 3 * x + 5 * y = 98) (h2 : 8 * x + 3 * y = 158) : x = 16 ∧ y = 10 :=
sorry

theorem part2 (hx : x = 16) (hy : y = 10) (hz : 16 * z + 10 * (40 - z) ≤ 550) : z ≤ 25 :=
sorry

end part1_part2_l120_120080


namespace evaluate_expression_l120_120942

theorem evaluate_expression (b : ℕ) (h : b = 2) : b^3 * b^4 = 128 :=
by
  -- sorry is used to skip the proof
  sorry

end evaluate_expression_l120_120942


namespace find_m_l120_120711

open Set

theorem find_m (m : ℝ) (A B : Set ℝ)
  (h1 : A = {-1, 3, 2 * m - 1})
  (h2 : B = {3, m})
  (h3 : B ⊆ A) : m = 1 ∨ m = -1 :=
by
  sorry

end find_m_l120_120711


namespace product_of_last_two_digits_div_by_6_and_sum_15_l120_120156

theorem product_of_last_two_digits_div_by_6_and_sum_15
  (n : ℕ)
  (h1 : n % 6 = 0)
  (A B : ℕ)
  (h2 : n % 100 = 10 * A + B)
  (h3 : A + B = 15)
  (h4 : B % 2 = 0) : 
  A * B = 54 := 
sorry

end product_of_last_two_digits_div_by_6_and_sum_15_l120_120156


namespace total_height_of_tower_l120_120220

theorem total_height_of_tower :
  let S₃₅ : ℕ := (35 * (35 + 1)) / 2
  let S₆₅ : ℕ := (65 * (65 + 1)) / 2
  S₃₅ + S₆₅ = 2775 :=
by
  let S₃₅ := (35 * (35 + 1)) / 2
  let S₆₅ := (65 * (65 + 1)) / 2
  sorry

end total_height_of_tower_l120_120220


namespace find_variable_value_l120_120801

axiom variable_property (x : ℝ) (h : 4 + 1 / x ≠ 0) : 5 / (4 + 1 / x) = 1 → x = 1

-- Given condition: 5 / (4 + 1 / x) = 1
-- Prove: x = 1
theorem find_variable_value (x : ℝ) (h : 4 + 1 / x ≠ 0) (h1 : 5 / (4 + 1 / x) = 1) : x = 1 :=
variable_property x h h1

end find_variable_value_l120_120801


namespace convert_to_scientific_notation_l120_120290

theorem convert_to_scientific_notation (H : 1 = 10^9) : 
  3600 * (10 : ℝ)^9 = 3.6 * (10 : ℝ)^12 :=
by
  sorry

end convert_to_scientific_notation_l120_120290


namespace compare_slopes_l120_120408

noncomputable def f (p q r x : ℝ) := x^3 + p * x^2 + q * x + r

noncomputable def s (p q x : ℝ) := 3 * x^2 + 2 * p * x + q

theorem compare_slopes (p q r a b c : ℝ) (hb : b ≠ 0) (ha : a ≠ c) 
  (hfa : f p q r a = 0) (hfc : f p q r c = 0) : a > c → s p q a > s p q c := 
by
  sorry

end compare_slopes_l120_120408


namespace division_remainder_l120_120119

-- let f(r) = r^15 + r + 1
def f (r : ℝ) : ℝ := r^15 + r + 1

-- let g(r) = r^2 - 1
def g (r : ℝ) : ℝ := r^2 - 1

-- remainder polynomial b(r)
def b (r : ℝ) : ℝ := r + 1

-- Lean statement to prove that polynomial division of f(r) by g(r) 
-- yields the remainder b(r)
theorem division_remainder (r : ℝ) : (f r) % (g r) = b r :=
  sorry

end division_remainder_l120_120119


namespace fixed_point_always_l120_120575

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := 2^x + Real.logb a (x + 1) + 3

theorem fixed_point_always (a : ℝ) (h : a > 0 ∧ a ≠ 1) : f 0 a = 4 :=
by
  sorry

end fixed_point_always_l120_120575


namespace series_sum_eq_l120_120109

noncomputable def series_sum : Real :=
  ∑' n : ℕ, (4 * (n + 1) + 1) / (((4 * (n + 1) - 1) ^ 3) * ((4 * (n + 1) + 3) ^ 3))

theorem series_sum_eq : series_sum = 1 / 5184 := sorry

end series_sum_eq_l120_120109


namespace paint_cost_is_200_l120_120251

-- Define the basic conditions and parameters
def side_length : ℕ := 5
def faces_of_cube : ℕ := 6
def area_per_face (side : ℕ) : ℕ := side * side
def total_surface_area (side : ℕ) (faces : ℕ) : ℕ := faces * area_per_face side
def coverage_per_kg : ℕ := 15
def cost_per_kg : ℕ := 20

-- Calculate total cost
def total_cost (side : ℕ) (faces : ℕ) (coverage : ℕ) (cost : ℕ) : ℕ :=
  let total_area := total_surface_area side faces
  let kgs_required := total_area / coverage
  kgs_required * cost

theorem paint_cost_is_200 :
  total_cost side_length faces_of_cube coverage_per_kg cost_per_kg = 200 :=
by
  sorry

end paint_cost_is_200_l120_120251


namespace factorization_a_minus_b_l120_120037

theorem factorization_a_minus_b (a b : ℤ) (h1 : 3 * b + a = -7) (h2 : a * b = -6) : a - b = 7 :=
sorry

end factorization_a_minus_b_l120_120037


namespace age_of_second_replaced_man_l120_120431

theorem age_of_second_replaced_man (avg_age_increase : ℕ) (avg_new_men_age : ℕ) (first_replaced_age : ℕ) (total_men : ℕ) (new_age_sum : ℕ) :
  avg_age_increase = 1 →
  avg_new_men_age = 34 →
  first_replaced_age = 21 →
  total_men = 12 →
  new_age_sum = 2 * avg_new_men_age →
  47 - (new_age_sum - (first_replaced_age + x)) = 12 →
  x = 35 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end age_of_second_replaced_man_l120_120431


namespace triangle_perimeter_l120_120669

theorem triangle_perimeter (a b : ℕ) (h1 : a = 2) (h2 : b = 3) (x : ℕ) 
  (x_odd : x % 2 = 1) (triangle_ineq : 1 < x ∧ x < 5) : a + b + x = 8 :=
by
  sorry

end triangle_perimeter_l120_120669


namespace f_derivative_at_1_intervals_of_monotonicity_l120_120419

def f (x : ℝ) := x^3 - 3 * x^2 + 10
def f' (x : ℝ) := 3 * x^2 - 6 * x

theorem f_derivative_at_1 : f' 1 = -3 := by
  sorry

theorem intervals_of_monotonicity :
  (∀ x : ℝ, x < 0 → f' x > 0) ∧
  (∀ x : ℝ, 0 < x ∧ x < 2 → f' x < 0) ∧
  (∀ x : ℝ, x > 2 → f' x > 0) := by
  sorry

end f_derivative_at_1_intervals_of_monotonicity_l120_120419


namespace solve_inequalities_solve_fruit_purchase_l120_120216

-- Part 1: Inequalities
theorem solve_inequalities {x : ℝ} : 
  (2 * x < 16) ∧ (3 * x > 2 * x + 3) → (3 < x ∧ x < 8) := by
  sorry

-- Part 2: Fruit Purchase
theorem solve_fruit_purchase {x y : ℝ} : 
  (x + y = 7) ∧ (5 * x + 8 * y = 41) → (x = 5 ∧ y = 2) := by
  sorry

end solve_inequalities_solve_fruit_purchase_l120_120216


namespace a9_value_l120_120866

-- Define the sequence
def seq (a : ℕ → ℚ) : Prop :=
  a 1 = 3 ∧ ∀ n, a (n+1) = 1 - (1 / a n)

-- State the theorem
theorem a9_value : ∃ a : ℕ → ℚ, seq a ∧ a 9 = -1/2 :=
by
  sorry

end a9_value_l120_120866


namespace binary_101_to_decimal_l120_120852

theorem binary_101_to_decimal : (1 * 2^2 + 0 * 2^1 + 1 * 2^0) = 5 := by
  sorry

end binary_101_to_decimal_l120_120852


namespace marble_problem_solution_l120_120646

noncomputable def probability_two_marbles (red_marble_initial white_marble_initial total_drawn : ℕ) : ℚ :=
  let total_initial := red_marble_initial + white_marble_initial
  let probability_first_white := (white_marble_initial : ℚ) / total_initial
  let red_marble_after_first_draw := red_marble_initial
  let total_after_first_draw := total_initial - 1
  let probability_second_red := (red_marble_after_first_draw : ℚ) / total_after_first_draw
  probability_first_white * probability_second_red

theorem marble_problem_solution :
  probability_two_marbles 4 6 2 = 4 / 15 := by
  sorry

end marble_problem_solution_l120_120646


namespace longer_side_length_l120_120737

theorem longer_side_length (x y : ℝ) (h1 : 2 * x + 2 * y = 60) (h2 : x * y = 221) : max x y = 17 :=
by
  sorry

end longer_side_length_l120_120737


namespace find_S15_l120_120987

-- Define the arithmetic progression series
variable {S : ℕ → ℕ}

-- Given conditions
axiom S5 : S 5 = 3
axiom S10 : S 10 = 12

-- We need to prove the final statement
theorem find_S15 : S 15 = 39 := 
by
  sorry

end find_S15_l120_120987


namespace sequence_general_term_l120_120891

theorem sequence_general_term (a : ℕ → ℕ) (h1 : a 1 = 1)
  (h2 : ∀ n : ℕ, 3 * (Finset.range (n + 1)).sum a = (n + 2) * a n) :
  ∀ n : ℕ, a n = n :=
by
  sorry

end sequence_general_term_l120_120891


namespace area_triangle_ABC_l120_120138

noncomputable def area_of_triangle_ABC : ℝ :=
  let base_AB : ℝ := 6 - 0
  let height_AB : ℝ := 2 - 0
  let base_BC : ℝ := 6 - 3
  let height_BC : ℝ := 8 - 0
  let base_CA : ℝ := 3 - 0
  let height_CA : ℝ := 8 - 2
  let area_ratio : ℝ := 1 / 2
  let area_I' : ℝ := area_ratio * base_AB * height_AB
  let area_II' : ℝ := area_ratio * 8 * 6
  let area_III' : ℝ := area_ratio * 8 * 3
  let total_small_triangles : ℝ := area_I' + area_II' + area_III'
  let total_area_rectangle : ℝ := 6 * 8
  total_area_rectangle - total_small_triangles

theorem area_triangle_ABC : area_of_triangle_ABC = 6 := 
by
  sorry

end area_triangle_ABC_l120_120138


namespace shirts_per_minute_l120_120503

theorem shirts_per_minute (total_shirts : ℕ) (total_minutes : ℕ) (h1 : total_shirts = 196) (h2 : total_minutes = 28) :
  total_shirts / total_minutes = 7 :=
by
  -- beginning of proof would go here
  sorry

end shirts_per_minute_l120_120503


namespace final_bicycle_price_l120_120751

-- Define conditions 
def original_price : ℝ := 200
def first_discount : ℝ := 0.20
def second_discount : ℝ := 0.25
def price_after_first_discount := original_price * (1 - first_discount)
def final_price := price_after_first_discount * (1 - second_discount)

-- Define the Lean statement to be proven
theorem final_bicycle_price :
  final_price = 120 :=
by
  -- Proof goes here
  sorry

end final_bicycle_price_l120_120751


namespace fraction_of_area_l120_120482

def larger_square_side : ℕ := 6
def shaded_square_side : ℕ := 2

def larger_square_area : ℕ := larger_square_side * larger_square_side
def shaded_square_area : ℕ := shaded_square_side * shaded_square_side

theorem fraction_of_area : (shaded_square_area : ℚ) / larger_square_area = 1 / 9 :=
by
  -- proof omitted
  sorry

end fraction_of_area_l120_120482


namespace probability_of_selecting_male_is_three_fifths_l120_120947

-- Define the number of male and female students
def num_male_students : ℕ := 6
def num_female_students : ℕ := 4

-- Define the total number of students
def total_students : ℕ := num_male_students + num_female_students

-- Define the probability of selecting a male student's ID
def probability_male_student : ℚ := num_male_students / total_students

-- Theorem: The probability of selecting a male student's ID is 3/5
theorem probability_of_selecting_male_is_three_fifths : probability_male_student = 3 / 5 :=
by
  -- Proof to be filled in
  sorry

end probability_of_selecting_male_is_three_fifths_l120_120947


namespace evaluate_x_squared_plus_y_squared_l120_120443

theorem evaluate_x_squared_plus_y_squared (x y : ℚ) (h1 : x + 2 * y = 20) (h2 : 3 * x + y = 19) : x^2 + y^2 = 401 / 5 :=
sorry

end evaluate_x_squared_plus_y_squared_l120_120443


namespace points_for_win_l120_120819

variable (W T : ℕ)

theorem points_for_win (W T : ℕ) (h1 : W * (T + 12) + T = 60) : W = 2 :=
by {
  sorry
}

end points_for_win_l120_120819


namespace number_of_students_l120_120387

theorem number_of_students (S : ℕ) (hS1 : S ≥ 2) (hS2 : S ≤ 80) 
                          (hO : ∀ n : ℕ, (n * S) % 120 = 0) : 
    S = 40 :=
sorry

end number_of_students_l120_120387


namespace haley_seeds_in_big_garden_l120_120274

def seeds_in_big_garden (total_seeds : ℕ) (small_gardens : ℕ) (seeds_per_small_garden : ℕ) : ℕ :=
  total_seeds - (small_gardens * seeds_per_small_garden)

theorem haley_seeds_in_big_garden :
  let total_seeds := 56
  let small_gardens := 7
  let seeds_per_small_garden := 3
  seeds_in_big_garden total_seeds small_gardens seeds_per_small_garden = 35 :=
by
  sorry

end haley_seeds_in_big_garden_l120_120274


namespace fraction_of_15_smaller_by_20_l120_120333

/-- Define 80% of 40 -/
def eighty_percent_of_40 : ℝ := 0.80 * 40

/-- Define the fraction of 15 that we are looking for -/
def fraction_of_15 (x : ℝ) : ℝ := x * 15

/-- Define the problem statement -/
theorem fraction_of_15_smaller_by_20 : ∃ x : ℝ, fraction_of_15 x = eighty_percent_of_40 - 20 ∧ x = 4 / 5 :=
by
  sorry

end fraction_of_15_smaller_by_20_l120_120333


namespace phi_value_l120_120604

open Real

noncomputable def f (x : ℝ) (φ : ℝ) : ℝ := sin (2 * x + φ)

theorem phi_value (φ : ℝ) (h : |φ| < π / 2) :
  (∀ x : ℝ, f (x + π / 3) φ = f (-(x + π / 3)) φ) → φ = -(π / 6) :=
by
  intro h'
  sorry

end phi_value_l120_120604


namespace total_amount_shared_l120_120906

-- Define the variables
variables (a b c : ℕ)

-- Define the conditions
axiom condition1 : a = (1 / 3 : ℝ) * (b + c)
axiom condition2 : b = (2 / 7 : ℝ) * (a + c)
axiom condition3 : a = b + 15

-- The proof statement
theorem total_amount_shared : a + b + c = 540 :=
by
  -- We assume these axioms are declared and noncontradictory
  sorry

end total_amount_shared_l120_120906


namespace integer_coefficient_equation_calculate_expression_l120_120878

noncomputable def a : ℝ := (Real.sqrt 5 - 1) / 2

theorem integer_coefficient_equation :
  a ^ 2 + a - 1 = 0 :=
sorry

theorem calculate_expression :
  a ^ 3 - 2 * a + 2015 = 2014 :=
sorry

end integer_coefficient_equation_calculate_expression_l120_120878


namespace arithmetic_seq_a4_l120_120224

theorem arithmetic_seq_a4 (a : ℕ → ℕ) 
  (h1 : a 1 = 2) 
  (h2 : a 2 = 4) 
  (h3 : a 3 = 6) : 
  a 4 = 8 :=
by
  sorry

end arithmetic_seq_a4_l120_120224


namespace value_of_a_l120_120995

open Set

theorem value_of_a (a : ℝ) (h : {1, 2} ∪ {x | x^2 - a * x + a - 1 = 0} = {1, 2}) : a = 3 :=
by
  sorry

end value_of_a_l120_120995


namespace divisibility_of_a81_l120_120287

theorem divisibility_of_a81 
  (p : ℕ) (hp : Nat.Prime p) (hp_gt2 : 2 < p)
  (a : ℕ → ℕ) (h_rec : ∀ n, n * a (n + 1) = (n + 1) * a n - (p / 2)^4) 
  (h_a1 : a 1 = 5) :
  16 ∣ a 81 := 
sorry

end divisibility_of_a81_l120_120287


namespace sum_of_ages_is_37_l120_120529

def maries_age : ℕ := 12
def marcos_age (M : ℕ) : ℕ := 2 * M + 1

theorem sum_of_ages_is_37 : maries_age + marcos_age maries_age = 37 := 
by
  -- Inserting the proof details
  sorry

end sum_of_ages_is_37_l120_120529


namespace arithmetic_sequence_sum_l120_120215

def f (x : ℝ) : ℝ := (x - 3) ^ 3 + x - 1

-- Define the arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d ≠ 0, ∀ n : ℕ, a (n + 1) = a n + d

-- Problem Statement
theorem arithmetic_sequence_sum 
  (a : ℕ → ℝ) 
  (h_arith : is_arithmetic_sequence a) 
  (h_f_sum : f (a 1) + f (a 2) + f (a 3) + f (a 4) + f (a 5) + f (a 6) + f (a 7) = 14) : 
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 = 21 :=
sorry

end arithmetic_sequence_sum_l120_120215


namespace part_1_part_2_l120_120306

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := x * Real.log x - m * x
noncomputable def f' (x : ℝ) (m : ℝ) : ℝ := Real.log x + 1 - m
noncomputable def h (x : ℝ) (a : ℝ) : ℝ := x * Real.log x - x - a * x^3
noncomputable def r (x : ℝ) : ℝ := (Real.log x - 1) / x^2
noncomputable def r' (x : ℝ) : ℝ := (3 - 2 * Real.log x) / x^3

theorem part_1 (x : ℝ) (m : ℝ) (h1 : f x m = -1) (h2 : f' x m = 0) :
  m = 1 ∧ (∀ y, y > 0 → y < x → f' y 1 < 0) ∧ (∀ y, y > x → f' y 1 > 0) :=
sorry

theorem part_2 (a : ℝ) :
  (a > 1 / (2 * Real.exp 3) → ∀ x, h x a ≠ 0) ∧
  (a ≤ 0 ∨ a = 1 / (2 * Real.exp 3) → ∃ x, h x a = 0 ∧ ∀ y, h y a = 0 → y = x) ∧
  (0 < a ∧ a < 1 / (2 * Real.exp 3) → ∃ x1 x2, x1 ≠ x2 ∧ h x1 a = 0 ∧ h x2 a = 0) :=
sorry

end part_1_part_2_l120_120306


namespace joohyeon_snack_count_l120_120586

theorem joohyeon_snack_count
  (c s : ℕ)
  (h1 : 300 * c + 500 * s = 3000)
  (h2 : c + s = 8) :
  s = 3 :=
sorry

end joohyeon_snack_count_l120_120586


namespace simplify_fractions_sum_l120_120583

theorem simplify_fractions_sum :
  (48 / 72) + (30 / 45) = 4 / 3 := 
by
  sorry

end simplify_fractions_sum_l120_120583


namespace find_a_l120_120663

-- Condition: Define a * b as 2a - b^2
def star (a b : ℝ) := 2 * a - b^2

-- Proof problem: Prove the value of a given the condition and that a * 7 = 16.
theorem find_a : ∃ a : ℝ, star a 7 = 16 ∧ a = 32.5 :=
by
  sorry

end find_a_l120_120663


namespace boat_speed_still_water_l120_120880

variable (V_b V_s t : ℝ)

-- Conditions given in the problem
axiom speedOfStream : V_s = 13
axiom timeRelation : ∀ t, (V_b + V_s) * t = 2 * (V_b - V_s) * t

-- The statement to be proved
theorem boat_speed_still_water : V_b = 39 :=
by
  sorry

end boat_speed_still_water_l120_120880


namespace intersection_points_l120_120637

-- Define the four line equations
def line1 (x y : ℝ) : Prop := 2 * y - 3 * x = 4
def line2 (x y : ℝ) : Prop := x + 3 * y = 3
def line3 (x y : ℝ) : Prop := 6 * x - 4 * y = 2
def line4 (x y : ℝ) : Prop := 5 * x - 15 * y = 15

-- State the theorem for intersection points
theorem intersection_points : 
  (line1 (18/11) (13/11) ∧ line2 (18/11) (13/11)) ∧ 
  (line2 (21/11) (8/11) ∧ line3 (21/11) (8/11)) :=
by
  sorry

end intersection_points_l120_120637


namespace row_number_sum_l120_120044

theorem row_number_sum (n : ℕ) (h : (2 * n - 1) ^ 2 = 2015 ^ 2) : n = 1008 :=
by
  sorry

end row_number_sum_l120_120044


namespace income_expenditure_ratio_l120_120410

theorem income_expenditure_ratio (I E S : ℝ) (hI : I = 10000) (hS : S = 2000) (hEq : S = I - E) : I / E = 5 / 4 :=
by {
  sorry
}

end income_expenditure_ratio_l120_120410


namespace amy_tips_calculation_l120_120867

theorem amy_tips_calculation 
  (hourly_wage : ℝ) (hours_worked : ℝ) (total_earnings : ℝ) 
  (h_wage : hourly_wage = 2)
  (h_hours : hours_worked = 7)
  (h_total : total_earnings = 23) : 
  total_earnings - (hourly_wage * hours_worked) = 9 := 
sorry

end amy_tips_calculation_l120_120867


namespace only_one_way_to_center_l120_120863

def is_center {n : ℕ} (grid_size n : ℕ) (coord : ℕ × ℕ) : Prop :=
  coord = (grid_size / 2 + 1, grid_size / 2 + 1)

def count_ways_to_center : ℕ :=
  if h : (1 <= 3 ∧ 3 <= 5) then 1 else 0

theorem only_one_way_to_center : count_ways_to_center = 1 := by
  sorry

end only_one_way_to_center_l120_120863


namespace find_a_pow_b_l120_120849

theorem find_a_pow_b (a b : ℝ) (h : (a - 2)^2 + |b + 1| = 0) : a^b = 1 / 2 := 
sorry

end find_a_pow_b_l120_120849


namespace polynomial_evaluation_qin_jiushao_l120_120259

theorem polynomial_evaluation_qin_jiushao :
  let x := 3
  let V0 := 7
  let V1 := V0 * x + 6
  let V2 := V1 * x + 5
  let V3 := V2 * x + 4
  let V4 := V3 * x + 3
  V4 = 789 :=
by
  -- placeholder for proof
  sorry

end polynomial_evaluation_qin_jiushao_l120_120259


namespace smallest_possible_a_plus_b_l120_120868

theorem smallest_possible_a_plus_b :
  ∃ (a b : ℕ), 0 < a ∧ 0 < b ∧ gcd (a + b) 330 = 1 ∧ (b ^ b ∣ a ^ a) ∧ ¬ (b ∣ a) ∧ (a + b = 507) := 
sorry

end smallest_possible_a_plus_b_l120_120868


namespace integer_condition_l120_120307

theorem integer_condition (p : ℕ) (h : p > 0) : 
  (∃ n : ℤ, (3 * (p: ℤ) + 25) = n * (2 * (p: ℤ) - 5)) ↔ (3 ≤ p ∧ p ≤ 35) :=
sorry

end integer_condition_l120_120307


namespace find_d_div_a_l120_120617
noncomputable def quad_to_square_form (x : ℝ) : ℝ :=
  x^2 + 1500 * x + 1800

theorem find_d_div_a : 
  ∃ (a d : ℝ), (∀ x : ℝ, quad_to_square_form x = (x + a)^2 + d) 
  ∧ a = 750 
  ∧ d = -560700 
  ∧ d / a = -560700 / 750 := 
sorry

end find_d_div_a_l120_120617


namespace hyperbola_eccentricity_correct_l120_120605

noncomputable def hyperbola_eccentricity (a b : ℝ) (h_a : a > 0) (h_b : b > 0)
  (h_asymptote : b / a = Real.tan (Real.pi / 6)) : ℝ :=
  Real.sqrt (1 + (b / a)^2)

theorem hyperbola_eccentricity_correct
  (a b : ℝ) (h_a : a > 0) (h_b : b > 0)
  (h_asymptote : b / a = Real.tan (Real.pi / 6)) :
  hyperbola_eccentricity a b h_a h_b h_asymptote = 2 * Real.sqrt 3 / 3 :=
by
  sorry

end hyperbola_eccentricity_correct_l120_120605


namespace husband_and_wife_age_l120_120726

theorem husband_and_wife_age (x y : ℕ) (h1 : 11 * x = 2 * (22 * y - 11 * x)) (h2 : 11 * x ≠ 0) (h3 : 11 * y ≠ 0) (h4 : 11 * (x + y) ≤ 99) : 
  x = 4 ∧ y = 3 :=
by
  sorry

end husband_and_wife_age_l120_120726


namespace num_integers_satisfy_l120_120452

theorem num_integers_satisfy : 
  ∃ n : ℕ, (n = 7 ∧ ∀ k : ℤ, (k > -5 ∧ k < 3) → (k = -4 ∨ k = -3 ∨ k = -2 ∨ k = -1 ∨ k = 0 ∨ k = 1 ∨ k = 2)) := 
sorry

end num_integers_satisfy_l120_120452


namespace find_f_4_l120_120130

noncomputable def f : ℝ → ℝ := sorry

axiom functional_equation : ∀ x : ℝ, f x + 3 * f (1 - x) = 4 * x ^ 2

theorem find_f_4 : f 4 = 5.5 :=
by
  sorry

end find_f_4_l120_120130


namespace ratios_of_square_areas_l120_120256

variable (x : ℝ)

def square_area (side_length : ℝ) : ℝ := side_length^2

theorem ratios_of_square_areas (hA : square_area x = x^2)
                               (hB : square_area (5 * x) = 25 * x^2)
                               (hC : square_area (2 * x) = 4 * x^2) :
  (square_area x / square_area (5 * x) = 1 / 25 ∧
   square_area (2 * x) / square_area (5 * x) = 4 / 25) := 
by {
  sorry
}

end ratios_of_square_areas_l120_120256


namespace calculate_x_l120_120379

def percentage (p : ℚ) (n : ℚ) := (p / 100) * n

theorem calculate_x : 
  (percentage 47 1442 - percentage 36 1412) + 65 = 234.42 := 
by 
  sorry

end calculate_x_l120_120379


namespace x_squared_plus_y_squared_l120_120243

theorem x_squared_plus_y_squared (x y : ℝ) (h1 : x - y = 18) (h2 : x * y = 16) : x^2 + y^2 = 356 := 
by
  sorry

end x_squared_plus_y_squared_l120_120243


namespace sum_X_Y_l120_120497

-- Define the variables and assumptions
variable (X Y : ℕ)

-- Hypotheses
axiom h1 : Y + 2 = X
axiom h2 : X + 5 = Y

-- Theorem statement
theorem sum_X_Y : X + Y = 12 := by
  sorry

end sum_X_Y_l120_120497


namespace circle_center_radius_l120_120105

theorem circle_center_radius :
  ∀ (x y : ℝ), (x + 1) ^ 2 + (y - 2) ^ 2 = 9 ↔ (x = -1 ∧ y = 2 ∧ ∃ r : ℝ, r = 3) :=
by
  sorry

end circle_center_radius_l120_120105


namespace prism_volume_eq_400_l120_120424

noncomputable def prism_volume (a b c : ℝ) : ℝ := a * b * c

theorem prism_volume_eq_400 
  (a b c : ℝ)
  (h1 : a * b = 40)
  (h2 : a * c = 50)
  (h3 : b * c = 80) :
  prism_volume a b c = 400 :=
by
  sorry

end prism_volume_eq_400_l120_120424


namespace CNY_share_correct_l120_120406

noncomputable def total_NWF : ℝ := 1388.01
noncomputable def deductions_method1 : List ℝ := [41.89, 2.77, 478.48, 554.91, 0.24]
noncomputable def previous_year_share_CNY : ℝ := 17.77
noncomputable def deductions_method2 : List (ℝ × String) := [(3.02, "EUR"), (0.2, "USD"), (34.47, "GBP"), (39.98, "others"), (0.02, "other")]

theorem CNY_share_correct :
  let CNY22 := total_NWF - (deductions_method1.foldl (λ a b => a + b) 0)
  let alpha22_CNY := (CNY22 / total_NWF) * 100
  let method2_result := 100 - (deductions_method2.foldl (λ a b => a + b.1) 0)
  alpha22_CNY = 22.31 ∧ method2_result = 22.31 := 
sorry

end CNY_share_correct_l120_120406


namespace log_base_change_l120_120278

theorem log_base_change (log_16_32 log_16_inv2: ℝ) : 
  (log_16_32 * log_16_inv2 = -5 / 16) :=
by
  sorry

end log_base_change_l120_120278


namespace find_x_solution_l120_120011

theorem find_x_solution (x : ℚ) : (∀ y : ℚ, 12 * x * y - 18 * y + 3 * x - 9 / 2 = 0) ↔ x = 3 / 2 :=
by
  sorry

end find_x_solution_l120_120011


namespace find_number_l120_120548

theorem find_number (x : ℝ) (h : 45 * 7 = 0.35 * x) : x = 900 :=
by
  -- Proof (skipped with sorry)
  sorry

end find_number_l120_120548


namespace binomial_variance_is_one_l120_120261

noncomputable def binomial_variance (n : ℕ) (p : ℚ) : ℚ := n * p * (1 - p)

theorem binomial_variance_is_one :
  binomial_variance 4 (1 / 2) = 1 := by
  sorry

end binomial_variance_is_one_l120_120261


namespace a_x1_x2_x13_eq_zero_l120_120523

theorem a_x1_x2_x13_eq_zero {a x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12 x13 : ℤ}
  (h1: a = (1 + x1) * (1 + x2) * (1 + x3) * (1 + x4) * (1 + x5) * (1 + x6) * (1 + x7) *
             (1 + x8) * (1 + x9) * (1 + x10) * (1 + x11) * (1 + x12) * (1 + x13))
  (h2: a = (1 - x1) * (1 - x2) * (1 - x3) * (1 - x4) * (1 - x5) * (1 - x6) * (1 - x7) *
             (1 - x8) * (1 - x9) * (1 - x10) * (1 - x11) * (1 - x12) * (1 - x13)) :
  a * x1 * x2 * x3 * x4 * x5 * x6 * x7 * x8 * x9 * x10 * x11 * x12 * x13 = 0 := by
  sorry

end a_x1_x2_x13_eq_zero_l120_120523


namespace quadratic_solution_linear_factor_solution_l120_120165

theorem quadratic_solution (x : ℝ) : (5 * x^2 + 2 * x - 1 = 0) ↔ (x = (-1 + Real.sqrt 6) / 5 ∨ x = (-1 - Real.sqrt 6) / 5) := by
  sorry

theorem linear_factor_solution (x : ℝ) : (x * (x - 3) - 4 * (3 - x) = 0) ↔ (x = 3 ∨ x = -4) := by
  sorry

end quadratic_solution_linear_factor_solution_l120_120165


namespace smallest_value_x_l120_120824

theorem smallest_value_x : 
  (∃ x : ℝ, ((5*x - 20)/(4*x - 5))^2 + ((5*x - 20)/(4*x - 5)) = 6 ∧ 
  (∀ y : ℝ, ((5*y - 20)/(4*y - 5))^2 + ((5*y - 20)/(4*y - 5)) = 6 → x ≤ y)) → 
  x = 35 / 17 :=
by 
  sorry

end smallest_value_x_l120_120824


namespace solution_exists_l120_120508

open Real

theorem solution_exists (x : ℝ) (h1 : x > 9) (h2 : sqrt (x - 3 * sqrt (x - 9)) + 3 = sqrt (x + 3 * sqrt (x - 9)) - 3) : x ≥ 18 :=
sorry

end solution_exists_l120_120508


namespace greatest_discarded_oranges_l120_120225

theorem greatest_discarded_oranges (n : ℕ) : n % 7 ≤ 6 := 
by 
  sorry

end greatest_discarded_oranges_l120_120225


namespace value_of_x_l120_120689

theorem value_of_x (x : ℕ) (M : Set ℕ) :
  M = {0, 1, 2} →
  M ∪ {x} = {0, 1, 2, 3} →
  x = 3 :=
by
  sorry

end value_of_x_l120_120689


namespace sum_of_triangles_l120_120745

def triangle (a b c : ℤ) : ℤ := a + b - c

theorem sum_of_triangles : triangle 1 3 4 + triangle 2 5 6 = 1 := by
  sorry

end sum_of_triangles_l120_120745


namespace smallest_k_values_l120_120460

def cos_squared_eq_one (k : ℕ) : Prop :=
  ∃ n : ℕ, k^2 + 49 = 180 * n

theorem smallest_k_values :
  ∃ (k1 k2 : ℕ), (cos_squared_eq_one k1) ∧ (cos_squared_eq_one k2) ∧
  (∀ k < k1, ¬ cos_squared_eq_one k) ∧ (∀ k < k2, ¬ cos_squared_eq_one k) ∧ 
  k1 = 31 ∧ k2 = 37 :=
by
  sorry

end smallest_k_values_l120_120460


namespace ratio_circumscribed_circle_area_triangle_area_l120_120794

open Real

theorem ratio_circumscribed_circle_area_triangle_area (h R : ℝ) (h_eq : R = h / 2) :
  let circle_area := π * R^2
  let triangle_area := (h^2) / 4
  (circle_area / triangle_area) = π :=
by
  sorry

end ratio_circumscribed_circle_area_triangle_area_l120_120794


namespace sum_of_series_l120_120950

noncomputable def seriesSum : ℝ := ∑' n : ℕ, (4 * (n + 1) + 1) / (3 ^ (n + 1))

theorem sum_of_series : seriesSum = 7 / 2 := by
  sorry

end sum_of_series_l120_120950


namespace Q_has_exactly_one_negative_root_l120_120468

def Q (x : ℝ) : ℝ := x^7 + 5 * x^5 + 5 * x^4 - 6 * x^3 - 2 * x^2 - 10 * x + 12

theorem Q_has_exactly_one_negative_root :
  ∃! r : ℝ, r < 0 ∧ Q r = 0 := sorry

end Q_has_exactly_one_negative_root_l120_120468


namespace intersection_of_sets_l120_120781

open Set

theorem intersection_of_sets (M N : Set ℕ) (hM : M = {1, 2, 3}) (hN : N = {2, 3, 4}) :
  M ∩ N = {2, 3} :=
by
  sorry

end intersection_of_sets_l120_120781


namespace min_value_expr_l120_120450

theorem min_value_expr (x y z : ℝ) (h : x > 0 ∧ y > 0 ∧ z > 0) (h_xyz : x * y * z = 1) : 
  x^2 + 4 * x * y + 9 * y^2 + 8 * y * z + 3 * z^2 ≥ 9^(10/9) :=
sorry

end min_value_expr_l120_120450


namespace smallest_even_consecutive_sum_l120_120776

theorem smallest_even_consecutive_sum (n : ℕ) (h_even : n % 2 = 0) (h_sum : n + (n + 2) + (n + 4) = 162) : n = 52 :=
sorry

end smallest_even_consecutive_sum_l120_120776


namespace value_of_f2_l120_120865

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := a * x + b * x + 3

theorem value_of_f2 (a b : ℝ) (h1 : f 1 a b = 7) (h2 : f 3 a b = 15) : f 2 a b = 11 :=
by
  sorry

end value_of_f2_l120_120865


namespace find_F2_l120_120227

-- Set up the conditions as definitions
def m : ℝ := 1 -- in kg
def R1 : ℝ := 0.5 -- in meters
def R2 : ℝ := 1 -- in meters
def F1 : ℝ := 1 -- in Newtons

-- Rotational inertia I formula
def I (R : ℝ) : ℝ := m * R^2

-- Equality of angular accelerations
def alpha_eq (F1 F2 R1 R2 : ℝ) : Prop :=
  (F1 * R1) / (I R1) = (F2 * R2) / (I R2)

-- The proof goal
theorem find_F2 (F2 : ℝ) : 
  alpha_eq F1 F2 R1 R2 → F2 = 2 :=
by
  sorry

end find_F2_l120_120227


namespace eldora_boxes_paper_clips_l120_120707

theorem eldora_boxes_paper_clips (x y : ℝ)
  (h1 : 1.85 * x + 7 * y = 55.40)
  (h2 : 1.85 * 12 + 10 * y = 61.70)
  (h3 : 1.85 = 1.85) : -- Given && Asserting the constant price of one box

  x = 15 :=
by
  sorry

end eldora_boxes_paper_clips_l120_120707


namespace harmonyNumbersWithFirstDigit2_l120_120053

def isHarmonyNumber (n : ℕ) : Prop :=
  let digits := [n / 1000 % 10, n / 100 % 10, n / 10 % 10, n % 10]
  digits.sum = 6

def startsWithDigit (d n : ℕ) : Prop :=
  n / 1000 = d

theorem harmonyNumbersWithFirstDigit2 :
  ∃ c : ℕ, c = 15 ∧ ∀ n : ℕ, (1000 ≤ n ∧ n < 10000) → isHarmonyNumber n → startsWithDigit 2 n → ∃ m : ℕ, m < c ∧ m = n :=
sorry

end harmonyNumbersWithFirstDigit2_l120_120053


namespace find_principal_l120_120912

variable (SI : ℝ) (R : ℝ) (T : ℝ) (P : ℝ)

theorem find_principal (h1 : SI = 4020.75) (h2 : R = 0.0875) (h3 : T = 5.5) (h4 : SI = P * R * T) : 
  P = 8355.00 :=
sorry

end find_principal_l120_120912


namespace find_a_l120_120638

noncomputable def givenConditions (a b c R : ℝ) : Prop :=
  (a^2 / (b * c) - c / b - b / c = Real.sqrt 3) ∧ (R = 3)

theorem find_a (a b c : ℝ) (R : ℝ) (h : givenConditions a b c R) : a = 3 :=
by
  sorry

end find_a_l120_120638


namespace find_kn_l120_120327

theorem find_kn (k n : ℕ) (h : k * n^2 - k * n - n^2 + n = 94) : k = 48 ∧ n = 2 := 
by 
  sorry

end find_kn_l120_120327


namespace boxes_filled_l120_120060

theorem boxes_filled (total_toys toys_per_box : ℕ) (h1 : toys_per_box = 8) (h2 : total_toys = 32) :
  total_toys / toys_per_box = 4 :=
by
  sorry

end boxes_filled_l120_120060


namespace gcd_ab_a2b2_eq_one_or_two_l120_120081

-- Definitions and conditions
def coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

-- Problem statement
theorem gcd_ab_a2b2_eq_one_or_two (a b : ℕ) (h : coprime a b) : 
  Nat.gcd (a + b) (a^2 + b^2) = 1 ∨ Nat.gcd (a + b) (a^2 + b^2) = 2 :=
by
  sorry

end gcd_ab_a2b2_eq_one_or_two_l120_120081


namespace Angela_insect_count_l120_120920

variables (Angela Jacob Dean : ℕ)
-- Conditions
def condition1 : Prop := Angela = Jacob / 2
def condition2 : Prop := Jacob = 5 * Dean
def condition3 : Prop := Dean = 30

-- Theorem statement proving Angela's insect count
theorem Angela_insect_count (h1 : condition1 Angela Jacob) (h2 : condition2 Jacob Dean) (h3 : condition3 Dean) : Angela = 75 :=
by
  sorry

end Angela_insect_count_l120_120920


namespace price_of_other_frisbees_l120_120804

theorem price_of_other_frisbees :
  ∃ F3 Fx Px : ℕ, F3 + Fx = 60 ∧ 3 * F3 + Px * Fx = 204 ∧ Fx ≥ 24 ∧ Px = 4 := 
by
  sorry

end price_of_other_frisbees_l120_120804


namespace probability_sunflower_seed_l120_120397

theorem probability_sunflower_seed :
  ∀ (sunflower_seeds green_bean_seeds pumpkin_seeds : ℕ),
  sunflower_seeds = 2 →
  green_bean_seeds = 3 →
  pumpkin_seeds = 4 →
  (sunflower_seeds + green_bean_seeds + pumpkin_seeds = 9) →
  (sunflower_seeds : ℚ) / (sunflower_seeds + green_bean_seeds + pumpkin_seeds) = 2 / 9 := 
by 
  intros sunflower_seeds green_bean_seeds pumpkin_seeds h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  rw [h1, h2, h3]
  sorry -- Proof omitted as per instructions.

end probability_sunflower_seed_l120_120397


namespace incorrect_expression_among_options_l120_120270

theorem incorrect_expression_among_options :
  ¬(0.75 ^ (-0.3) < 0.75 ^ (0.1)) :=
by
  sorry

end incorrect_expression_among_options_l120_120270


namespace greatest_possible_sum_of_roots_l120_120807

noncomputable def quadratic_roots (c b : ℝ) : Prop :=
  ∃ α β : ℝ, α ≠ β ∧ α + β = c ∧ α * β = b ∧ |α - β| = 1

theorem greatest_possible_sum_of_roots :
  ∃ (c : ℝ), ( ∃ b : ℝ, quadratic_roots c b) ∧
             ( ∀ (d : ℝ), ( ∃ b : ℝ, quadratic_roots d b) → d ≤ 11 ) ∧ c = 11 :=
sorry

end greatest_possible_sum_of_roots_l120_120807


namespace Jenny_reading_days_l120_120654

theorem Jenny_reading_days :
  let words_per_hour := 100
  let book1_words := 200
  let book2_words := 400
  let book3_words := 300
  let total_words := book1_words + book2_words + book3_words
  let total_hours := total_words / words_per_hour
  let minutes_per_day := 54
  let hours_per_day := minutes_per_day / 60
  total_hours / hours_per_day = 10 :=
by
  sorry

end Jenny_reading_days_l120_120654


namespace original_selling_price_l120_120380

theorem original_selling_price (CP SP_original SP_loss : ℝ)
  (h1 : SP_original = CP * 1.25)
  (h2 : SP_loss = CP * 0.85)
  (h3 : SP_loss = 544) : SP_original = 800 :=
by
  -- The proof goes here, but we are skipping it with sorry
  sorry

end original_selling_price_l120_120380


namespace sufficient_condition_for_product_l120_120690

-- Given conditions
def intersects_parabola_at_two_points (x1 y1 x2 y2 : ℝ) : Prop :=
  y1^2 = 4 * x1 ∧ y2^2 = 4 * x2 ∧ x1 ≠ x2

def line_through_focus (x y : ℝ) (k : ℝ) : Prop :=
  y = k * (x - 1)

noncomputable def line_l (k : ℝ) (x : ℝ) : ℝ :=
  k * (x - 1)

-- The theorem to prove
theorem sufficient_condition_for_product 
  (x1 y1 x2 y2 k : ℝ)
  (h1 : intersects_parabola_at_two_points x1 y1 x2 y2)
  (h2 : line_through_focus x1 y1 k)
  (h3 : line_through_focus x2 y2 k) :
  x1 * x2 = 1 :=
sorry

end sufficient_condition_for_product_l120_120690


namespace problem_l120_120051

theorem problem (k : ℕ) (hk : 0 < k) (n : ℕ) : 
  (∃ p : ℕ, n = 2 * 3 ^ (k - 1) * p ∧ 0 < p) ↔ 3^k ∣ (2^n - 1) := 
by 
  sorry

end problem_l120_120051


namespace beads_problem_l120_120295

theorem beads_problem :
  ∃ b : ℕ, (b % 6 = 5) ∧ (b % 8 = 3) ∧ (b % 9 = 7) ∧ (b = 179) :=
by
  sorry

end beads_problem_l120_120295


namespace estimation_correct_l120_120869

-- Definitions corresponding to conditions.
def total_population : ℕ := 10000
def surveyed_population : ℕ := 200
def aware_surveyed : ℕ := 125

-- The proportion step: 125/200 = x/10000
def proportion (aware surveyed total_pop : ℕ) : ℕ :=
  (aware * total_pop) / surveyed

-- Using this to define our main proof goal
def estimated_aware := proportion aware_surveyed surveyed_population total_population

-- Final proof statement
theorem estimation_correct :
  estimated_aware = 6250 :=
sorry

end estimation_correct_l120_120869


namespace soldiers_to_add_l120_120101

theorem soldiers_to_add (N : ℕ) (add : ℕ) 
    (h1 : N % 7 = 2)
    (h2 : N % 12 = 2)
    (h_add : add = 84 - N) :
    add = 82 :=
by
  sorry

end soldiers_to_add_l120_120101


namespace freezer_temperature_is_minus_12_l120_120285

theorem freezer_temperature_is_minus_12 (refrigeration_temp freezer_temp : ℤ) (h1 : refrigeration_temp = 5) (h2 : freezer_temp = -12) : freezer_temp = -12 :=
by sorry

end freezer_temperature_is_minus_12_l120_120285


namespace fraction_of_problems_solved_by_Andrey_l120_120394

theorem fraction_of_problems_solved_by_Andrey (N x : ℕ) 
  (h1 : 0 < N) 
  (h2 : x = N / 2)
  (Boris_solves : ∀ y : ℕ, y = N - x → y / 3 = (N - x) / 3)
  (remaining_problems : ∀ y : ℕ, y = (N - x) - (N - x) / 3 → y = 2 * (N - x) / 3) 
  (Viktor_solves : (2 * (N - x) / 3 = N / 3)) :
  x / N = 1 / 2 := 
by {
  sorry
}

end fraction_of_problems_solved_by_Andrey_l120_120394


namespace part_a_part_b_part_c_l120_120016

-- Defining a structure for the problem
structure Rectangle :=
(area : ℝ)

structure Figure :=
(area : ℝ)

-- Defining the conditions
variables (R : Rectangle) 
  (F1 F2 F3 F4 F5 : Figure)
  (overlap_area_pair : Figure → Figure → ℝ)
  (overlap_area_triple : Figure → Figure → Figure → ℝ)

-- Given conditions
axiom R_area : R.area = 1
axiom F1_area : F1.area = 0.5
axiom F2_area : F2.area = 0.5
axiom F3_area : F3.area = 0.5
axiom F4_area : F4.area = 0.5
axiom F5_area : F5.area = 0.5

-- Statements to prove
theorem part_a : ∃ (F1 F2 : Figure), overlap_area_pair F1 F2 ≥ 3 / 20 := sorry
theorem part_b : ∃ (F1 F2 : Figure), overlap_area_pair F1 F2 ≥ 1 / 5 := sorry
theorem part_c : ∃ (F1 F2 F3 : Figure), overlap_area_triple F1 F2 F3 ≥ 1 / 20 := sorry

end part_a_part_b_part_c_l120_120016


namespace solve_abs_eq_l120_120791

theorem solve_abs_eq (x : ℝ) : 
    (3 * x + 9 = abs (-20 + 4 * x)) ↔ 
    (x = 29) ∨ (x = 11 / 7) := 
by sorry

end solve_abs_eq_l120_120791


namespace arithmetic_sequence_geometric_subsequence_l120_120413

theorem arithmetic_sequence_geometric_subsequence :
  ∀ (a : ℕ → ℝ), (∀ n, a (n + 1) = a n + 2) ∧ (a 1 * a 3 = a 2 ^ 2) → a 2 = 4 :=
by
  intros a h
  sorry

end arithmetic_sequence_geometric_subsequence_l120_120413


namespace ratio_x_y_l120_120673

theorem ratio_x_y (x y : ℝ) (h1 : x * y = 9) (h2 : 0 < x) (h3 : 0 < y) (h4 : y = 0.5) : x / y = 36 :=
by
  sorry

end ratio_x_y_l120_120673


namespace vector_perpendicular_to_a_l120_120591

theorem vector_perpendicular_to_a :
  let a := (4, 3)
  let b := (3, -4)
  a.1 * b.1 + a.2 * b.2 = 0 := by
  let a := (4, 3)
  let b := (3, -4)
  sorry

end vector_perpendicular_to_a_l120_120591


namespace sport_formulation_water_l120_120796

theorem sport_formulation_water (corn_syrup_ounces : ℕ) (h_cs : corn_syrup_ounces = 3) : 
  ∃ water_ounces : ℕ, water_ounces = 45 :=
by
  -- The ratios for the "sport" formulation: Flavoring : Corn Syrup : Water = 1 : 4 : 60
  let flavoring_ratio := 1
  let corn_syrup_ratio := 4
  let water_ratio := 60
  -- The given corn syrup is 3 ounces which corresponds to corn_syrup_ratio parts
  have h_ratio : corn_syrup_ratio = 4 := rfl
  have h_flavoring_to_corn_syrup : flavoring_ratio / corn_syrup_ratio = 1 / 4 := by sorry
  have h_flavoring_to_water : flavoring_ratio / water_ratio = 1 / 60 := by sorry
  -- Set up the proportion
  have h_proportion : corn_syrup_ratio / corn_syrup_ounces = water_ratio / 45 := by sorry 
  -- Cross-multiply to solve for the water
  have h_cross_mul : 4 * 45 = 3 * 60 := by sorry
  exact ⟨45, rfl⟩

end sport_formulation_water_l120_120796


namespace total_animals_seen_correct_l120_120779

-- Define the number of beavers in the morning
def beavers_morning : ℕ := 35

-- Define the number of chipmunks in the morning
def chipmunks_morning : ℕ := 60

-- Define the number of beavers in the afternoon (tripled)
def beavers_afternoon : ℕ := 3 * beavers_morning

-- Define the number of chipmunks in the afternoon (decreased by 15)
def chipmunks_afternoon : ℕ := chipmunks_morning - 15

-- Calculate the total number of animals seen in the morning
def total_morning : ℕ := beavers_morning + chipmunks_morning

-- Calculate the total number of animals seen in the afternoon
def total_afternoon : ℕ := beavers_afternoon + chipmunks_afternoon

-- The total number of animals seen that day
def total_animals_seen : ℕ := total_morning + total_afternoon

theorem total_animals_seen_correct :
  total_animals_seen = 245 :=
by
  -- skipping the proof
  sorry

end total_animals_seen_correct_l120_120779


namespace triangle_area_l120_120280

-- Defining the rectangle dimensions
def length : ℝ := 35
def width : ℝ := 48

-- Defining the area of the right triangle formed by the diagonal of the rectangle
theorem triangle_area : (1 / 2) * length * width = 840 := by
  sorry

end triangle_area_l120_120280


namespace not_support_either_l120_120943

theorem not_support_either (total_attendance supporters_first supporters_second : ℕ) 
  (h1 : total_attendance = 50) 
  (h2 : supporters_first = 50 * 40 / 100) 
  (h3 : supporters_second = 50 * 34 / 100) : 
  total_attendance - (supporters_first + supporters_second) = 13 :=
by
  sorry

end not_support_either_l120_120943


namespace b_95_mod_49_l120_120896

-- Define the sequence b_n
def b (n : ℕ) : ℕ := 7^n + 9^n

-- Goal: Prove that the remainder when b 95 is divided by 49 is 28
theorem b_95_mod_49 : b 95 % 49 = 28 := 
by
  sorry

end b_95_mod_49_l120_120896


namespace find_m_l120_120371

def U : Set ℕ := {1, 2, 3, 4}
def compl_U_A : Set ℕ := {1, 4}

theorem find_m (m : ℕ) (A : Set ℕ) (hA : A = {x | x ^ 2 - 5 * x + m = 0 ∧ x ∈ U}) :
  compl_U_A = U \ A → m = 6 :=
by
  sorry

end find_m_l120_120371


namespace speed_difference_l120_120010

theorem speed_difference (distance : ℕ) (time_jordan time_alex : ℕ) (h_distance : distance = 12) (h_time_jordan : time_jordan = 10) (h_time_alex : time_alex = 15) :
  (distance / (time_jordan / 60) - distance / (time_alex / 60) = 24) := by
  -- Lean code to correctly parse and understand the natural numbers, division, and maintain the theorem structure.
  sorry

end speed_difference_l120_120010


namespace julies_birthday_day_of_week_l120_120563

theorem julies_birthday_day_of_week
    (fred_birthday_monday : Nat)
    (pat_birthday_before_fred : Nat)
    (julie_birthday_before_pat : Nat)
    (fred_birthday_after_pat : fred_birthday_monday - pat_birthday_before_fred = 37)
    (julie_birthday_before_pat_eq : pat_birthday_before_fred - julie_birthday_before_pat = 67)
    : (julie_birthday_before_pat - julie_birthday_before_pat % 7 + ((julie_birthday_before_pat % 7) - fred_birthday_monday % 7)) % 7 = 2 :=
by
  sorry

end julies_birthday_day_of_week_l120_120563


namespace simplify_expression_l120_120233

theorem simplify_expression (a b : ℝ) :
  ((3 * a^3 * b - 12 * a^2 * b^2 - 6 * a * b^3) / (-3 * a * b) - 4 * a * b) = (-a^2 + 2 * b^2) :=
by
  sorry

end simplify_expression_l120_120233


namespace range_of_m_l120_120094

theorem range_of_m (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 1 / (x + 1) + 4 / y = 1) (m : ℝ) :
  x + y / 4 > m^2 - 5 * m - 3 ↔ -1 < m ∧ m < 6 := sorry

end range_of_m_l120_120094


namespace billy_questions_third_hour_l120_120996

variable (x : ℝ)
variable (questions_in_first_hour : ℝ := x)
variable (questions_in_second_hour : ℝ := 1.5 * x)
variable (questions_in_third_hour : ℝ := 3 * x)
variable (total_questions_solved : ℝ := 242)

theorem billy_questions_third_hour (h : questions_in_first_hour + questions_in_second_hour + questions_in_third_hour = total_questions_solved) :
  questions_in_third_hour = 132 :=
by
  sorry

end billy_questions_third_hour_l120_120996


namespace option_c_same_function_l120_120907

theorem option_c_same_function :
  ∀ (x : ℝ), x ≠ 0 → (1 + (1 / x) = u ↔ u = 1 + (1 / (1 + 1 / x))) :=
by sorry

end option_c_same_function_l120_120907


namespace tan_pi_over_12_eq_l120_120018

theorem tan_pi_over_12_eq : Real.tan (Real.pi / 12) = 2 - Real.sqrt 3 :=
by
  sorry

end tan_pi_over_12_eq_l120_120018


namespace number_of_molecules_correct_l120_120660

-- Define Avogadro's number
def avogadros_number : ℝ := 6.022 * 10^23

-- Define the given number of molecules
def given_number_of_molecules : ℝ := 3 * 10^26

-- State the problem
theorem number_of_molecules_correct :
  (number_of_molecules = given_number_of_molecules) :=
by
  sorry

end number_of_molecules_correct_l120_120660


namespace evaluate_expression_l120_120454

theorem evaluate_expression : 
  ∀ (x y z : ℝ), 
  x = 2 → 
  y = -3 → 
  z = 1 → 
  x^2 + y^2 + z^2 + 2 * x * y - z^3 = 1 := by
  intros x y z hx hy hz
  rw [hx, hy, hz]
  sorry

end evaluate_expression_l120_120454


namespace total_number_of_balls_is_twelve_l120_120719

noncomputable def num_total_balls (a : ℕ) : Prop :=
(3 : ℚ) / a = (25 : ℚ) / 100

theorem total_number_of_balls_is_twelve : num_total_balls 12 :=
by sorry

end total_number_of_balls_is_twelve_l120_120719


namespace general_equation_of_curve_l120_120502

theorem general_equation_of_curve
  (t : ℝ) (ht : t > 0)
  (x : ℝ) (hx : x = (Real.sqrt t) - (1 / (Real.sqrt t)))
  (y : ℝ) (hy : y = 3 * (t + 1 / t) + 2) :
  x^2 = (y - 8) / 3 := by
  sorry

end general_equation_of_curve_l120_120502


namespace solve_equation_l120_120662

theorem solve_equation : 
  ∀ x : ℝ, (x - 3 ≠ 0) → (x + 6) / (x - 3) = 4 → x = 6 :=
by
  intros x h1 h2
  sorry

end solve_equation_l120_120662


namespace second_player_wins_l120_120501

-- Define the initial condition of the game
def initial_coins : Nat := 2016

-- Define the set of moves a player can make
def valid_moves : Finset Nat := {1, 2, 3}

-- Define the winning condition
def winning_player (coins : Nat) : String :=
  if coins % 4 = 0 then "second player"
  else "first player"

-- The theorem stating that second player has a winning strategy given the initial condition
theorem second_player_wins : winning_player initial_coins = "second player" :=
by
  sorry

end second_player_wins_l120_120501


namespace sum_first_nine_primes_l120_120364

theorem sum_first_nine_primes : 
  2 + 3 + 5 + 7 + 11 + 13 + 17 + 19 + 23 = 100 :=
by
  sorry

end sum_first_nine_primes_l120_120364


namespace train_length_l120_120533

theorem train_length (speed_kmh : ℝ) (time_sec : ℝ) (h_speed : speed_kmh = 30) (h_time : time_sec = 6) :
  ∃ length_meters : ℝ, abs (length_meters - 50) < 1 :=
by
  -- Converting speed from km/hr to m/s
  let speed_ms := speed_kmh * (1000 / 3600)
  
  -- Calculating length of the train using the distance formula
  let length_meters := speed_ms * time_sec

  use length_meters
  -- Proof would go here showing abs (length_meters - 50) < 1
  sorry

end train_length_l120_120533


namespace decreasing_power_function_has_specific_m_l120_120316

theorem decreasing_power_function_has_specific_m (m : ℝ) (x : ℝ) : 
  (∀ x > 0, (m^2 - m - 1) * x^(m^2 - 2 * m - 3) < 0) → 
  m = 2 :=
by
  sorry

end decreasing_power_function_has_specific_m_l120_120316


namespace determine_positive_integers_l120_120049

theorem determine_positive_integers (x y z : ℕ) (h : x^2 + y^2 - 15 = 2^z) :
  (x = 0 ∧ y = 4 ∧ z = 0) ∨ (x = 4 ∧ y = 0 ∧ z = 0) ∨
  (x = 1 ∧ y = 4 ∧ z = 1) ∨ (x = 4 ∧ y = 1 ∧ z = 1) :=
sorry

end determine_positive_integers_l120_120049


namespace sum_of_digits_l120_120532

theorem sum_of_digits (a b : ℕ) (h1 : 10 * a + b + 10 * b + a = 202) (h2 : a < 10) (h3 : b < 10) :
  a + b = 12 :=
sorry

end sum_of_digits_l120_120532


namespace find_m_l120_120384

theorem find_m (m : ℕ) (h : m * (Nat.factorial m) + 2 * (Nat.factorial m) = 5040) : m = 5 :=
by
  sorry

end find_m_l120_120384


namespace exactly_one_divisible_by_5_l120_120273

def a (n : ℕ) : ℕ := 2^(2*n + 1) - 2^(n + 1) + 1
def b (n : ℕ) : ℕ := 2^(2*n + 1) + 2^(n + 1) + 1

theorem exactly_one_divisible_by_5 (n : ℕ) (hn : 0 < n) : (a n % 5 = 0 ∧ b n % 5 ≠ 0) ∨ (a n % 5 ≠ 0 ∧ b n % 5 = 0) :=
  sorry

end exactly_one_divisible_by_5_l120_120273


namespace complete_the_square_l120_120155

theorem complete_the_square (a : ℝ) : a^2 + 4 * a - 5 = (a + 2)^2 - 9 :=
by sorry

end complete_the_square_l120_120155


namespace cost_of_new_shoes_l120_120282

theorem cost_of_new_shoes 
    (R : ℝ) 
    (L_r : ℝ) 
    (L_n : ℝ) 
    (increase_percent : ℝ) 
    (H_R : R = 13.50) 
    (H_L_r : L_r = 1) 
    (H_L_n : L_n = 2) 
    (H_inc_percent : increase_percent = 0.1852) : 
    2 * (R * (1 + increase_percent) / L_n) = 32.0004 := 
by
    sorry

end cost_of_new_shoes_l120_120282


namespace max_value_is_5_l120_120672

noncomputable def max_value (θ φ : ℝ) : ℝ :=
  3 * Real.sin θ * Real.cos φ + 2 * Real.sin φ ^ 2

theorem max_value_is_5 (θ φ : ℝ) (h1 : 0 ≤ θ) (h2 : θ ≤ Real.pi / 2) (h3 : 0 ≤ φ) (h4 : φ ≤ Real.pi / 2) :
  max_value θ φ ≤ 5 :=
sorry

end max_value_is_5_l120_120672


namespace functional_equation_solution_l120_120564

theorem functional_equation_solution (f : ℚ → ℚ) (h : ∀ x y : ℚ, f (x + y) = f x + f y) :
  ∃ a : ℚ, ∀ x : ℚ, f x = a * x :=
sorry

end functional_equation_solution_l120_120564


namespace find_r_l120_120614

theorem find_r (k r : ℝ) (h1 : 5 = k * 3^r) (h2 : 45 = k * 9^r) : r = 2 :=
sorry

end find_r_l120_120614


namespace students_not_taking_music_nor_art_l120_120099

theorem students_not_taking_music_nor_art (total_students music_students art_students both_students neither_students : ℕ) 
  (h_total : total_students = 500) 
  (h_music : music_students = 50) 
  (h_art : art_students = 20) 
  (h_both : both_students = 10) 
  (h_neither : neither_students = total_students - (music_students + art_students - both_students)) : 
  neither_students = 440 :=
by
  sorry

end students_not_taking_music_nor_art_l120_120099


namespace sum_first_100_terms_l120_120106

def a (n : ℕ) : ℤ := (-1) ^ (n + 1) * n

def S (n : ℕ) : ℤ := Finset.sum (Finset.range n) (λ i => a (i + 1))

theorem sum_first_100_terms : S 100 = -50 := 
by 
  sorry

end sum_first_100_terms_l120_120106


namespace correct_statements_l120_120900

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x - 13 / 4 * Real.pi)

theorem correct_statements :
    (f (Real.pi / 8) = 0) ∧ 
    (∀ x, 2 * Real.sin (2 * (x - 5 / 8 * Real.pi)) = f x) :=
by
  sorry

end correct_statements_l120_120900


namespace Alyssa_missed_games_l120_120675

theorem Alyssa_missed_games (total_games attended_games : ℕ) (h1 : total_games = 31) (h2 : attended_games = 13) : total_games - attended_games = 18 :=
by sorry

end Alyssa_missed_games_l120_120675


namespace problem_proof_l120_120095

noncomputable def problem : Prop :=
  ∀ x : ℝ, (x ≠ 2 ∧ (x-2)/(x-4) ≤ 3) ↔ (4 < x ∧ x < 5)

theorem problem_proof : problem := sorry

end problem_proof_l120_120095


namespace simplify_polynomial_l120_120028

variable (x : ℝ)

theorem simplify_polynomial : (2 * x^2 + 5 * x - 3) - (2 * x^2 + 9 * x - 6) = -4 * x + 3 :=
by
  sorry

end simplify_polynomial_l120_120028


namespace baker_cakes_remaining_l120_120239

theorem baker_cakes_remaining (initial_cakes: ℕ) (fraction_sold: ℚ) (sold_cakes: ℕ) (cakes_remaining: ℕ) :
  initial_cakes = 149 ∧ fraction_sold = 2/5 ∧ sold_cakes = 59 ∧ cakes_remaining = initial_cakes - sold_cakes → cakes_remaining = 90 :=
by
  sorry

end baker_cakes_remaining_l120_120239


namespace max_value_of_expr_l120_120158

noncomputable def max_expr (a b : ℝ) (h : a + b = 5) : ℝ :=
  a^4 * b + a^3 * b + a^2 * b + a * b + a * b^2 + a * b^3 + a * b^4

theorem max_value_of_expr (a b : ℝ) (h : a + b = 5) : max_expr a b h ≤ 6084 / 17 :=
sorry

end max_value_of_expr_l120_120158


namespace linear_system_solution_l120_120312

theorem linear_system_solution (x y : ℚ) (h1 : 3 * x + 2 * y = 7) (h2 : 6 * x - 5 * y = 4) :
  x = 43 / 27 ∧ y = 10 / 9 :=
sorry

end linear_system_solution_l120_120312


namespace parallel_lines_iff_a_eq_1_l120_120236

theorem parallel_lines_iff_a_eq_1 (a : ℝ) :
  (∀ x y : ℝ, ax + 2*y - 1 = 0 ↔ x + 2*y + 4 = 0) ↔ (a = 1) := 
sorry

end parallel_lines_iff_a_eq_1_l120_120236


namespace total_vases_l120_120536

theorem total_vases (vases_per_day : ℕ) (days : ℕ) (total_vases : ℕ) 
  (h1 : vases_per_day = 16) 
  (h2 : days = 16) 
  (h3 : total_vases = vases_per_day * days) : 
  total_vases = 256 := 
by 
  sorry

end total_vases_l120_120536


namespace line_not_tangent_if_only_one_common_point_l120_120117

theorem line_not_tangent_if_only_one_common_point (l p : ℝ) :
  (∃ y, y^2 = 2 * p * l) ∧ ¬ (∃ x : ℝ, y = l ∧ y^2 = 2 * p * x) := 
  sorry

end line_not_tangent_if_only_one_common_point_l120_120117


namespace max_length_of_each_piece_l120_120800

theorem max_length_of_each_piece (a b c d : ℕ) (h1 : a = 48) (h2 : b = 72) (h3 : c = 108) (h4 : d = 120) : Nat.gcd (Nat.gcd a b) (Nat.gcd c d) = 12 := by
  sorry

end max_length_of_each_piece_l120_120800


namespace least_possible_b_prime_l120_120114

theorem least_possible_b_prime :
  ∃ b a : ℕ, Nat.Prime a ∧ Nat.Prime b ∧ 2 * a + b = 180 ∧ a > b ∧ b = 2 :=
by
  sorry

end least_possible_b_prime_l120_120114


namespace count_decorations_l120_120414

/--
Define a function T(n) that determines the number of ways to decorate the window 
with n stripes according to the given conditions.
--/
def T : ℕ → ℕ
| 0       => 1 -- optional case for completeness
| 1       => 2
| 2       => 2
| (n + 1) => T n + T (n - 1)

theorem count_decorations : T 10 = 110 := by
  sorry

end count_decorations_l120_120414


namespace sum_single_digit_numbers_l120_120610

noncomputable def are_single_digit_distinct (a b c d : ℕ) : Prop :=
  a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

theorem sum_single_digit_numbers :
  ∀ (A B C D : ℕ),
  are_single_digit_distinct A B C D →
  1000 * A + B - (5000 + 10 * C + 9) = 1000 + 100 * D + 93 →
  A + B + C + D = 18 :=
by
  sorry

end sum_single_digit_numbers_l120_120610


namespace power_calculation_l120_120157

theorem power_calculation : 8^6 * 27^6 * 8^18 * 27^18 = 216^24 := by
  sorry

end power_calculation_l120_120157


namespace largest_expr_is_a_squared_plus_b_squared_l120_120404

noncomputable def largest_expression (a b : ℝ) (h₁ : a > 1) (h₂ : b > 1) (h₃ : a ≠ b) : Prop :=
  (a^2 + b^2 > a - b) ∧ (a^2 + b^2 > a + b) ∧ (a^2 + b^2 > 2 * a * b)

theorem largest_expr_is_a_squared_plus_b_squared (a b : ℝ) (h₁ : a > 1) (h₂ : b > 1) (h₃ : a ≠ b) : 
  largest_expression a b h₁ h₂ h₃ :=
by
  sorry

end largest_expr_is_a_squared_plus_b_squared_l120_120404


namespace area_of_triangle_is_24_l120_120931

open Real

-- Define the coordinates of the vertices
def A : ℝ × ℝ := (-2, 3)
def B : ℝ × ℝ := (6, 1)
def C : ℝ × ℝ := (10, 6)

-- Define the vectors from point C
def v : ℝ × ℝ := (A.1 - C.1, A.2 - C.2)
def w : ℝ × ℝ := (B.1 - C.1, B.2 - C.2)

-- Define the determinant for the parallelogram area
def parallelogram_area : ℝ :=
  abs (v.1 * w.2 - v.2 * w.1)

-- Prove the area of the triangle
theorem area_of_triangle_is_24 : (parallelogram_area / 2) = 24 := by
  sorry

end area_of_triangle_is_24_l120_120931


namespace test_point_selection_l120_120540

theorem test_point_selection (x_1 x_2 : ℝ)
    (interval_begin interval_end : ℝ) (h_interval : interval_begin = 2 ∧ interval_end = 4)
    (h_better_result : x_1 < x_2 ∨ x_1 > x_2)
    (h_test_points : (x_1 = interval_begin + 0.618 * (interval_end - interval_begin) ∧ 
                     x_2 = interval_begin + interval_end - x_1) ∨ 
                    (x_1 = interval_begin + interval_end - (interval_begin + 0.618 * (interval_end - interval_begin)) ∧ 
                     x_2 = interval_begin + 0.618 * (interval_end - interval_begin)))
  : ∃ x_3, x_3 = 3.528 ∨ x_3 = 2.472 := by
    sorry

end test_point_selection_l120_120540


namespace find_m_l120_120235

def A : Set ℕ := {1, 3}
def B (m : ℕ) : Set ℕ := {1, 2, m}

theorem find_m (m : ℕ) (h : A ⊆ B m) : m = 3 :=
sorry

end find_m_l120_120235


namespace age_of_new_person_l120_120780

theorem age_of_new_person (n : ℕ) (T A : ℕ) (h₁ : n = 10) (h₂ : T = 15 * n)
    (h₃ : (T + A) / (n + 1) = 17) : A = 37 := by
  sorry

end age_of_new_person_l120_120780


namespace div3_of_div9_l120_120355

theorem div3_of_div9 (u v : ℤ) (h : 9 ∣ (u^2 + u * v + v^2)) : 3 ∣ u ∧ 3 ∣ v :=
sorry

end div3_of_div9_l120_120355


namespace distance_from_tangency_to_tangent_theorem_l120_120522

noncomputable def distance_from_tangency_to_tangent (R r : ℝ) : ℝ :=
  2 * R * r / (R + r)

theorem distance_from_tangency_to_tangent_theorem (R r : ℝ) :
  ∃ d : ℝ, d = distance_from_tangency_to_tangent R r :=
by
  use 2 * R * r / (R + r)
  sorry

end distance_from_tangency_to_tangent_theorem_l120_120522


namespace lines_from_equation_l120_120838

-- Definitions for the conditions
def satisfies_equation (x y : ℝ) : Prop :=
  2 * x^2 + y^2 + 3 * x * y + 3 * x + y = 2

-- Equivalent Lean statement to the proof problem
theorem lines_from_equation :
  (∀ x y : ℝ, satisfies_equation x y → (y = -x - 2) ∨ (y = -2 * x + 1)) :=
by
  intros x y h
  sorry

end lines_from_equation_l120_120838


namespace slide_vs_slip_l120_120860

noncomputable def ladder : Type := sorry

def slide_distance (ladder : ladder) : ℝ := sorry
def slip_distance (ladder : ladder) : ℝ := sorry
def is_right_triangle (ladder : ladder) : Prop := sorry

theorem slide_vs_slip (l : ladder) (h : is_right_triangle l) : slip_distance l > slide_distance l :=
sorry

end slide_vs_slip_l120_120860


namespace min_value_of_xy_ratio_l120_120991

theorem min_value_of_xy_ratio :
  ∃ t : ℝ,
    (t = 2 ∨
    t = ((-1 + Real.sqrt 217) / 12) ∨
    t = ((-1 - Real.sqrt 217) / 12)) ∧
    min (min 2 ((-1 + Real.sqrt 217) / 12)) ((-1 - Real.sqrt 217) / 12) = -1.31 :=
sorry

end min_value_of_xy_ratio_l120_120991


namespace area_of_figure_enclosed_by_curve_l120_120348

theorem area_of_figure_enclosed_by_curve (θ : ℝ) : 
  ∃ (A : ℝ), A = 4 * Real.pi ∧ (∀ θ, (4 * Real.cos θ)^2 = (4 * Real.cos θ) * 4 * Real.cos θ) :=
sorry

end area_of_figure_enclosed_by_curve_l120_120348


namespace num_ordered_pairs_xy_eq_2200_l120_120423

/-- There are 24 ordered pairs (x, y) such that xy = 2200. -/
theorem num_ordered_pairs_xy_eq_2200 : 
  ∃ (n : ℕ), n = 24 ∧ (∃ divisors : Finset ℕ, 
    (∀ d ∈ divisors, 2200 % d = 0) ∧ 
    (divisors.card = 24)) := 
sorry

end num_ordered_pairs_xy_eq_2200_l120_120423


namespace instantaneous_velocity_at_1_l120_120965

noncomputable def particle_displacement (t : ℝ) : ℝ := t + Real.log t

theorem instantaneous_velocity_at_1 : 
  let v := fun t => deriv (particle_displacement) t
  v 1 = 2 :=
by
  sorry

end instantaneous_velocity_at_1_l120_120965


namespace calculate_seedlings_l120_120715

-- Define conditions
def condition_1 (x n : ℕ) : Prop :=
  x = 5 * n + 6

def condition_2 (x m : ℕ) : Prop :=
  x = 6 * m - 9

-- Define the main theorem based on these conditions
theorem calculate_seedlings (x : ℕ) : (∃ n, condition_1 x n) ∧ (∃ m, condition_2 x m) → x = 81 :=
by {
  sorry
}

end calculate_seedlings_l120_120715


namespace john_total_money_after_3_years_l120_120593

def principal : ℝ := 1000
def rate : ℝ := 0.1
def time : ℝ := 3

/-
  We need to prove that the total money after 3 years is $1300
-/
theorem john_total_money_after_3_years (principal : ℝ) (rate : ℝ) (time : ℝ) :
  principal + (principal * rate * time) = 1300 := by
  sorry

end john_total_money_after_3_years_l120_120593


namespace linear_system_substitution_correct_l120_120530

theorem linear_system_substitution_correct (x y : ℝ)
  (h1 : y = x - 1)
  (h2 : x + 2 * y = 7) :
  x + 2 * x - 2 = 7 :=
by
  sorry

end linear_system_substitution_correct_l120_120530


namespace area_shaded_smaller_dodecagon_area_in_circle_l120_120127

-- Part (a) statement
theorem area_shaded_smaller (dodecagon_area : ℝ) (shaded_area : ℝ) 
  (h : shaded_area = (1 / 12) * dodecagon_area) :
  shaded_area = dodecagon_area / 12 :=
sorry

-- Part (b) statement
theorem dodecagon_area_in_circle (r : ℝ) (A : ℝ) 
  (h : r = 1) (h' : A = (1 / 2) * 12 * r ^ 2 * Real.sin (2 * Real.pi / 12)) :
  A = 3 :=
sorry

end area_shaded_smaller_dodecagon_area_in_circle_l120_120127


namespace find_number_divided_by_6_l120_120833

theorem find_number_divided_by_6 (x : ℤ) (h : (x + 17) / 5 = 25) : x / 6 = 18 :=
by
  sorry

end find_number_divided_by_6_l120_120833


namespace total_fish_l120_120722

-- Defining the number of fish each person has, based on the conditions.
def billy_fish : ℕ := 10
def tony_fish : ℕ := 3 * billy_fish
def sarah_fish : ℕ := tony_fish + 5
def bobby_fish : ℕ := 2 * sarah_fish

-- The theorem stating the total number of fish.
theorem total_fish : billy_fish + tony_fish + sarah_fish + bobby_fish = 145 := by
  sorry

end total_fish_l120_120722


namespace robot_swap_eventually_non_swappable_l120_120879

theorem robot_swap_eventually_non_swappable (n : ℕ) (a : Fin n → ℕ) :
  ∃ t : ℕ, ∀ i : Fin (n - 1), ¬ (a (⟨i, sorry⟩ : Fin n) > a (⟨i + 1, sorry⟩ : Fin n)) ↔ n > 1 :=
sorry

end robot_swap_eventually_non_swappable_l120_120879


namespace functional_square_for_all_n_l120_120902

theorem functional_square_for_all_n (f : ℕ → ℕ) :
  (∀ m n : ℕ, ∃ k : ℕ, (f m + n) * (m + f n) = k ^ 2) ↔ ∃ c : ℕ, ∀ n : ℕ, f n = n + c := 
sorry

end functional_square_for_all_n_l120_120902


namespace school_distance_is_seven_l120_120760

-- Definitions based on conditions
def distance_to_school (x : ℝ) : Prop :=
  let monday_to_thursday_distance := 8 * x
  let friday_distance := 2 * x + 4
  let total_distance := monday_to_thursday_distance + friday_distance
  total_distance = 74

-- The problem statement to prove
theorem school_distance_is_seven : ∃ (x : ℝ), distance_to_school x ∧ x = 7 := 
by {
  sorry
}

end school_distance_is_seven_l120_120760


namespace initial_stock_decaf_percentage_l120_120664

-- Definitions as conditions of the problem
def initial_coffee_stock : ℕ := 400
def purchased_coffee_stock : ℕ := 100
def percentage_decaf_purchased : ℕ := 60
def total_percentage_decaf : ℕ := 32

/-- The proof problem statement -/
theorem initial_stock_decaf_percentage : 
  ∃ x : ℕ, x * initial_coffee_stock / 100 + percentage_decaf_purchased * purchased_coffee_stock / 100 = total_percentage_decaf * (initial_coffee_stock + purchased_coffee_stock) / 100 ∧ x = 25 :=
sorry

end initial_stock_decaf_percentage_l120_120664


namespace inequality_holds_l120_120046

theorem inequality_holds (x y : ℝ) (hx : x ≠ -1) (hy : y ≠ -1) (hxy : x * y = 1) :
  ( (2 + x) / (1 + x) )^2 + ( (2 + y) / (1 + y) )^2 ≥ 9 / 2 :=
by
  sorry

end inequality_holds_l120_120046


namespace inequality_hold_l120_120843

theorem inequality_hold (x y z : ℝ) :
  (x^2 + 2 * y^2 + 2 * z^2) / (x^2 + y * z) + 
  (y^2 + 2 * z^2 + 2 * x^2) / (y^2 + z * x) + 
  (z^2 + 2 * x^2 + 2 * y^2) / (z^2 + x * y) > 6 :=
by sorry

end inequality_hold_l120_120843


namespace carpet_rate_proof_l120_120742

noncomputable def carpet_rate (breadth_first : ℝ) (length_ratio : ℝ) (cost_second : ℝ) : ℝ :=
  let length_first := length_ratio * breadth_first
  let area_first := length_first * breadth_first
  let length_second := length_first * 1.4
  let breadth_second := breadth_first * 1.25
  let area_second := length_second * breadth_second 
  cost_second / area_second

theorem carpet_rate_proof : carpet_rate 6 1.44 4082.4 = 45 :=
by
  -- Here we provide the goal and state what needs to be proven.
  sorry

end carpet_rate_proof_l120_120742


namespace santana_brothers_birthday_l120_120456

theorem santana_brothers_birthday (b : ℕ) (oct : ℕ) (nov : ℕ) (dec : ℕ) (c_presents_diff : ℕ) :
  b = 7 → oct = 1 → nov = 1 → dec = 2 → c_presents_diff = 8 → (∃ M : ℕ, M = 3) :=
by
  sorry

end santana_brothers_birthday_l120_120456


namespace quality_of_algorithm_reflects_number_of_operations_l120_120300

-- Definitions
def speed_of_operation_is_important (c : Type) : Prop :=
  ∀ (c1 : c), true

-- Theorem stating that the number of operations within a unit of time is an important sign of the quality of an algorithm
theorem quality_of_algorithm_reflects_number_of_operations {c : Type} 
    (h_speed_important : speed_of_operation_is_important c) : 
  ∀ (a : Type) (q : a), true := 
sorry

end quality_of_algorithm_reflects_number_of_operations_l120_120300


namespace part1_part2_l120_120964

noncomputable def f (x : ℝ) : ℝ := abs (x + 20) - abs (16 - x)

theorem part1 (x : ℝ) : f x ≥ 0 ↔ x ≥ -2 := 
by sorry

theorem part2 (m : ℝ) (x_exists : ∃ x : ℝ, f x ≥ m) : m ≤ 36 := 
by sorry

end part1_part2_l120_120964


namespace intersection_A_B_l120_120718

namespace MathProof

open Set

def A := {y : ℝ | ∃ x : ℝ, y = x^2 - 2 * x}
def B := {y : ℝ | ∃ x : ℝ, y = -x^2 + 2 * x + 6}

theorem intersection_A_B : A ∩ B = Icc (-1 : ℝ) 7 :=
by
  sorry

end MathProof

end intersection_A_B_l120_120718


namespace division_by_reciprocal_l120_120887

theorem division_by_reciprocal :
  (10 / 3) / (1 / 5) = 50 / 3 := 
sorry

end division_by_reciprocal_l120_120887


namespace casey_nail_decorating_time_l120_120359

theorem casey_nail_decorating_time :
  let coat_application_time := 20
  let coat_drying_time := 20
  let pattern_time := 40
  let total_time := 3 * (coat_application_time + coat_drying_time) + pattern_time
  total_time = 160 :=
by
  let coat_application_time := 20
  let coat_drying_time := 20
  let pattern_time := 40
  let total_time := 3 * (coat_application_time + coat_drying_time) + pattern_time
  trivial

end casey_nail_decorating_time_l120_120359


namespace max_rectangle_area_l120_120242

theorem max_rectangle_area (l w : ℕ) (h_perimeter : 2 * l + 2 * w = 40) (h1 : l + w = 20) (hlw : l = 10 ∨ w = 10) : 
(l = 10 ∧ w = 10 ∧ l * w = 100) :=
by sorry

end max_rectangle_area_l120_120242


namespace chris_packed_percentage_l120_120944

theorem chris_packed_percentage (K C : ℕ) (h : K / (C : ℝ) = 2 / 3) :
  (C / (K + C : ℝ)) * 100 = 60 :=
by
  sorry

end chris_packed_percentage_l120_120944


namespace impossible_to_arrange_distinct_integers_in_grid_l120_120167

theorem impossible_to_arrange_distinct_integers_in_grid :
  ¬ ∃ (f : Fin 25 × Fin 41 → ℤ),
    (∀ i j, abs (f i - f j) ≤ 16 → (i ≠ j) → (i.1 = j.1 ∨ i.2 = j.2)) ∧
    (∃ i j, i ≠ j ∧ f i = f j) := 
sorry

end impossible_to_arrange_distinct_integers_in_grid_l120_120167


namespace sufficient_not_necessary_condition_l120_120599

variable (x : ℝ)

theorem sufficient_not_necessary_condition :
  (x > 2 → x > 1) ∧ (¬ (x > 1 → x > 2)) := by
sorry

end sufficient_not_necessary_condition_l120_120599


namespace parabola_properties_l120_120027

-- Define the parabola function as y = x^2 + px + q
def parabola (p q : ℝ) (x : ℝ) : ℝ := x^2 + p * x + q

-- Prove the properties of parabolas for varying p and q.
theorem parabola_properties (p q p' q' : ℝ) :
  (∀ x : ℝ, parabola p q x = x^2 + p * x + q) ∧
  (∀ x : ℝ, parabola p' q' x = x^2 + p' * x + q') →
  (∀ x : ℝ, ( ∃ k h : ℝ, parabola p q x = (x + h)^2 + k ) ∧ 
               ( ∃ k' h' : ℝ, parabola p' q' x = (x + h')^2 + k' ) ) ∧
  (∀ x : ℝ, h = -p / 2 ∧ k = q - p^2 / 4 ) ∧
  (∀ x : ℝ, h' = -p' / 2 ∧ k' = q' - p'^2 / 4 ) ∧
  (∀ x : ℝ, (h, k) ≠ (h', k') → parabola p q x ≠ parabola p' q' x) ∧
  (∀ x : ℝ, h = h' ∧ k = k' → parabola p q x = parabola p' q' x) :=
by
  sorry

end parabola_properties_l120_120027


namespace range_of_a_l120_120740

theorem range_of_a (a : ℝ) :
  (∃ (M : ℝ × ℝ), (M.1 - a)^2 + (M.2 - a + 2)^2 = 1 ∧
    (M.1)^2 + (M.2 - 2)^2 + (M.1)^2 + (M.2)^2 = 10) → 
  0 ≤ a ∧ a ≤ 3 := 
sorry

end range_of_a_l120_120740


namespace sufficient_but_not_necessary_condition_l120_120124

theorem sufficient_but_not_necessary_condition
  (a : ℝ) :
  (a = 2 → (a - 1) * (a - 2) = 0)
  ∧ (∃ a : ℝ, (a - 1) * (a - 2) = 0 ∧ a ≠ 2) :=
by
  sorry

end sufficient_but_not_necessary_condition_l120_120124


namespace sequences_count_l120_120341

theorem sequences_count (a_n b_n c_n : ℕ → ℕ) :
  (a_n 1 = 1) ∧ (b_n 1 = 1) ∧ (c_n 1 = 1) ∧ 
  (∀ n : ℕ, a_n (n + 1) = a_n n + b_n n) ∧ 
  (∀ n : ℕ, b_n (n + 1) = a_n n + b_n n + c_n n) ∧ 
  (∀ n : ℕ, c_n (n + 1) = b_n n + c_n n) → 
  ∀ n : ℕ, a_n n + b_n n + c_n n = 
            (1/2 * ((1 + Real.sqrt 2)^(n+1) + (1 - Real.sqrt 2)^(n+1))) :=
by
  intro h
  sorry

end sequences_count_l120_120341


namespace beverage_price_l120_120222

theorem beverage_price (P : ℝ) :
  (3 * 2.25 + 4 * P + 4 * 1.00) / 6 = 2.79 → P = 1.50 :=
by
  intro h -- Introduce the hypothesis.
  sorry  -- Proof is omitted.

end beverage_price_l120_120222


namespace inequality_proof_l120_120812

theorem inequality_proof (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) :
  x * (x - z) ^ 2 + y * (y - z) ^ 2 ≥ (x - z) * (y - z) * (x + y - z) :=
by
  sorry

end inequality_proof_l120_120812


namespace trip_attendees_trip_cost_savings_l120_120277

theorem trip_attendees (total_people : ℕ) (total_cost : ℕ) (adult_ticket : ℕ) 
(student_discount : ℕ) (group_discount : ℕ) (adults : ℕ) (students : ℕ) :
total_people = 130 → total_cost = 9600 → adult_ticket = 120 →
student_discount = 50 → group_discount = 40 → 
total_people = adults + students → 
total_cost = adults * adult_ticket + students * (adult_ticket * student_discount / 100) →
adults = 30 ∧ students = 100 :=
by sorry

theorem trip_cost_savings (total_people : ℕ) (individual_total_cost : ℕ) 
(group_total_cost : ℕ) (student_tickets : ℕ) (group_tickets : ℕ) 
(adult_ticket : ℕ) (student_discount : ℕ) (group_discount : ℕ) :
(total_people = 130) → (individual_total_cost = 7200 + 1800) → 
(group_total_cost = total_people * (adult_ticket * group_discount / 100)) →
(adult_ticket = 120) → (student_discount = 50) → (group_discount = 40) → 
(total_people = student_tickets + group_tickets) → (student_tickets = 30) → 
(group_tickets = 100) → (7200 + 1800 < 9360) → 
student_tickets = 30 ∧ group_tickets = 100 :=
by sorry

end trip_attendees_trip_cost_savings_l120_120277


namespace function_increasing_range_l120_120141

theorem function_increasing_range (a : ℝ) : 
    (∀ x : ℝ, x ≥ 4 → (2*x + 2*(a-1)) > 0) ↔ a ≥ -3 := 
by
  sorry

end function_increasing_range_l120_120141


namespace abs_diff_101st_term_l120_120116

theorem abs_diff_101st_term 
  (C D : ℕ → ℤ)
  (hC_start : C 0 = 20)
  (hD_start : D 0 = 20)
  (hC_diff : ∀ n, C (n + 1) = C n + 12)
  (hD_diff : ∀ n, D (n + 1) = D n - 6) :
  |C 100 - D 100| = 1800 :=
by
  sorry

end abs_diff_101st_term_l120_120116


namespace negation_of_there_exists_l120_120271

theorem negation_of_there_exists (x : ℝ) : ¬ (∃ x : ℝ, x^2 - x + 3 = 0) ↔ ∀ x : ℝ, x^2 - x + 3 ≠ 0 := by
  sorry

end negation_of_there_exists_l120_120271


namespace sqrt_diff_of_squares_l120_120377

theorem sqrt_diff_of_squares : (Real.sqrt 3 - 2) * (Real.sqrt 3 + 2) = -1 := by
  sorry

end sqrt_diff_of_squares_l120_120377


namespace intersecting_line_l120_120179

theorem intersecting_line {x y : ℝ} (h1 : x^2 + y^2 = 10) (h2 : (x - 1)^2 + (y - 3)^2 = 10) :
  x + 3 * y - 5 = 0 :=
sorry

end intersecting_line_l120_120179
