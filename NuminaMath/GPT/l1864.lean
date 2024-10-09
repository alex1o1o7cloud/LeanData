import Mathlib

namespace otimes_identity_l1864_186462

-- Define the operation ⊗
def otimes (k l : ℝ) : ℝ := k^2 - l^2

-- The goal is to show k ⊗ (k ⊗ k) = k^2 for any real number k
theorem otimes_identity (k : ℝ) : otimes k (otimes k k) = k^2 :=
by sorry

end otimes_identity_l1864_186462


namespace jerome_gave_to_meg_l1864_186410

theorem jerome_gave_to_meg (init_money half_money given_away meg bianca : ℝ) 
    (h1 : half_money = 43) 
    (h2 : init_money = 2 * half_money) 
    (h3 : 54 = init_money - given_away)
    (h4 : given_away = meg + bianca)
    (h5 : bianca = 3 * meg) : 
    meg = 8 :=
by
  sorry

end jerome_gave_to_meg_l1864_186410


namespace choosing_one_student_is_50_l1864_186456

-- Define the number of male students and female students
def num_male_students : Nat := 26
def num_female_students : Nat := 24

-- Define the total number of ways to choose one student
def total_ways_to_choose_one_student : Nat := num_male_students + num_female_students

-- Theorem statement proving the total number of ways to choose one student is 50
theorem choosing_one_student_is_50 : total_ways_to_choose_one_student = 50 := by
  sorry

end choosing_one_student_is_50_l1864_186456


namespace different_colors_of_roads_leading_out_l1864_186451

-- Define the city with intersections and streets
variables (n : ℕ) -- number of intersections
variables (c₁ c₂ c₃ : ℕ) -- number of external roads of each color

-- Conditions
axiom intersections_have_three_streets : ∀ (i : ℕ), i < n → (∀ (color : ℕ), color < 3 → exists (s : ℕ → ℕ), s color < n ∧ s color ≠ s ((color + 1) % 3) ∧ s color ≠ s ((color + 2) % 3))
axiom streets_colored_differently : ∀ (i : ℕ), i < n → (∀ (color1 color2 : ℕ), color1 < 3 → color2 < 3 → color1 ≠ color2 → exists (s1 s2 : ℕ → ℕ), s1 color1 < n ∧ s2 color2 < n ∧ s1 color1 ≠ s2 color2)

-- Problem Statement
theorem different_colors_of_roads_leading_out (h₁ : n % 2 = 0) (h₂ : c₁ + c₂ + c₃ = 3) : c₁ = 1 ∧ c₂ = 1 ∧ c₃ = 1 :=
by sorry

end different_colors_of_roads_leading_out_l1864_186451


namespace derivative_of_reciprocal_at_one_l1864_186474

noncomputable def f (x : ℝ) : ℝ := 1 / x

theorem derivative_of_reciprocal_at_one : (deriv f 1) = -1 :=
by {
    sorry
}

end derivative_of_reciprocal_at_one_l1864_186474


namespace edward_lives_left_l1864_186472

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

end edward_lives_left_l1864_186472


namespace necessary_but_not_sufficient_condition_for_increasing_geometric_sequence_l1864_186437

def is_increasing_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a n < a (n + 1)

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q, ∀ n, a (n + 1) = a n * q

theorem necessary_but_not_sufficient_condition_for_increasing_geometric_sequence
  (a : ℕ → ℝ)
  (h0 : a 0 > 0)
  (h_geom : is_geometric_sequence a) :
  (a 0^2 < a 1^2) ↔ (is_increasing_sequence a) ∧ ¬ (∀ n, a n > 0 → a (n + 1) > 0) :=
sorry

end necessary_but_not_sufficient_condition_for_increasing_geometric_sequence_l1864_186437


namespace num_integers_for_polynomial_negative_l1864_186464

open Int

theorem num_integers_for_polynomial_negative :
  ∃ (set_x : Finset ℤ), set_x.card = 12 ∧ ∀ x ∈ set_x, (x^4 - 65 * x^2 + 64) < 0 :=
by
  sorry

end num_integers_for_polynomial_negative_l1864_186464


namespace simplify_expression_l1864_186494

theorem simplify_expression (x y : ℝ) (h : (x + 2)^2 + abs (y - 1/2) = 0) :
  (x - 2*y)*(x + 2*y) - (x - 2*y)^2 = -6 :=
by
  -- Proof will be provided here
  sorry

end simplify_expression_l1864_186494


namespace max_product_of_two_integers_whose_sum_is_300_l1864_186481

theorem max_product_of_two_integers_whose_sum_is_300 :
  ∃ (a b : ℤ), a + b = 300 ∧ a * b = 22500 :=
by
  sorry

end max_product_of_two_integers_whose_sum_is_300_l1864_186481


namespace gcf_60_75_l1864_186421

theorem gcf_60_75 : Nat.gcd 60 75 = 15 := by
  sorry

end gcf_60_75_l1864_186421


namespace preferred_apples_percentage_l1864_186476

theorem preferred_apples_percentage (A B C O G : ℕ) (total freq_apples : ℕ)
  (hA : A = 70) (hB : B = 50) (hC: C = 30) (hO: O = 50) (hG: G = 40)
  (htotal : total = A + B + C + O + G)
  (hfa : freq_apples = A) :
  (freq_apples / total : ℚ) * 100 = 29 :=
by sorry

end preferred_apples_percentage_l1864_186476


namespace solution_set_inequality_l1864_186498

   theorem solution_set_inequality (a : ℝ) : (∀ x : ℝ, x^2 - 2*a*x + a > 0) ↔ (0 < a ∧ a < 1) :=
   sorry
   
end solution_set_inequality_l1864_186498


namespace points_on_opposite_sides_l1864_186463

theorem points_on_opposite_sides (a : ℝ) :
  (3 * 3 - 2 * 1 + a) * (3 * (-4) - 2 * 6 + a) < 0 ↔ -7 < a ∧ a < 24 :=
by sorry

end points_on_opposite_sides_l1864_186463


namespace roots_of_varying_signs_l1864_186453

theorem roots_of_varying_signs :
  (∃ x : ℝ, (4 * x^2 - 8 = 40 ∧ x != 0) ∧
           (∃ y : ℝ, (3 * y - 2)^2 = (y + 2)^2 ∧ y != 0) ∧
           (∃ z1 z2 : ℝ, z1 ≠ z2 ∧ (z1 = 0 ∨ z2 = 0) ∧ x^3 - 8 * x^2 + 13 * x + 10 = 0)) :=
sorry

end roots_of_varying_signs_l1864_186453


namespace inequality_reciprocal_l1864_186469

theorem inequality_reciprocal (a b c : ℝ) (h1 : a > b) (h2 : b > c) : 1 / (b - c) > 1 / (a - c) :=
sorry

end inequality_reciprocal_l1864_186469


namespace non_congruent_triangles_count_l1864_186430

-- Let there be 15 equally spaced points on a circle,
-- and considering triangles formed by connecting 3 of these points.
def num_non_congruent_triangles (n : Nat) : Nat :=
  (if n = 15 then 19 else 0)

theorem non_congruent_triangles_count :
  num_non_congruent_triangles 15 = 19 :=
by
  sorry

end non_congruent_triangles_count_l1864_186430


namespace faculty_after_reduction_is_correct_l1864_186478

-- Define the original number of faculty members
def original_faculty : ℝ := 253.25

-- Define the reduction percentage as a decimal
def reduction_percentage : ℝ := 0.23

-- Calculate the reduction amount
def reduction_amount : ℝ := original_faculty * reduction_percentage

-- Define the rounded reduction amount
def rounded_reduction_amount : ℝ := 58.25

-- Calculate the number of professors after the reduction
def professors_after_reduction : ℝ := original_faculty - rounded_reduction_amount

-- Statement to be proven: the number of professors after the reduction is 195
theorem faculty_after_reduction_is_correct : professors_after_reduction = 195 := by
  sorry

end faculty_after_reduction_is_correct_l1864_186478


namespace volume_of_displaced_water_square_of_displaced_water_volume_l1864_186484

-- Definitions for the conditions
def cube_side_length : ℝ := 10
def displaced_water_volume : ℝ := cube_side_length ^ 3
def displaced_water_volume_squared : ℝ := displaced_water_volume ^ 2

-- The Lean theorem statements proving the equivalence
theorem volume_of_displaced_water : displaced_water_volume = 1000 := by
  sorry

theorem square_of_displaced_water_volume : displaced_water_volume_squared = 1000000 := by
  sorry

end volume_of_displaced_water_square_of_displaced_water_volume_l1864_186484


namespace eq_x_in_terms_of_y_l1864_186479

theorem eq_x_in_terms_of_y (x y : ℝ) (h : 2 * x + y = 5) : x = (5 - y) / 2 := by
  sorry

end eq_x_in_terms_of_y_l1864_186479


namespace quadratic_real_equal_roots_l1864_186461

theorem quadratic_real_equal_roots (k : ℝ) :
  (∃ x : ℝ, 3 * x^2 - k * x + 2 * x + 15 = 0 ∧ ∀ y : ℝ, (3 * y^2 - k * y + 2 * y + 15 = 0 → y = x)) ↔
  (k = 6 * Real.sqrt 5 + 2 ∨ k = -6 * Real.sqrt 5 + 2) :=
by
  sorry

end quadratic_real_equal_roots_l1864_186461


namespace problem_solution_inf_problem_solution_prime_l1864_186448

-- Definitions based on the given conditions and problem statement
def is_solution_inf (m : ℕ) : Prop := 3^m ∣ 2^(3^m) + 1

def is_solution_prime (n : ℕ) : Prop := n.Prime ∧ n ∣ 2^n + 1

-- Lean statement for the math proof problem
theorem problem_solution_inf : ∀ m : ℕ, m ≥ 0 → is_solution_inf m := sorry

theorem problem_solution_prime : ∀ n : ℕ, n.Prime → is_solution_prime n → n = 3 := sorry

end problem_solution_inf_problem_solution_prime_l1864_186448


namespace millie_bracelets_left_l1864_186418

def millie_bracelets_initial : ℕ := 9
def millie_bracelets_lost : ℕ := 2

theorem millie_bracelets_left : millie_bracelets_initial - millie_bracelets_lost = 7 := 
by
  sorry

end millie_bracelets_left_l1864_186418


namespace find_x0_l1864_186482

-- Define a function f with domain [0, 3] and its inverse
variable {f : ℝ → ℝ}

-- Assume conditions for the inverse function
axiom f_inv_1 : ∀ x, 0 ≤ x ∧ x < 1 → 1 ≤ f x ∧ f x < 2
axiom f_inv_2 : ∀ x, 2 < x ∧ x ≤ 4 → 0 ≤ f x ∧ f x < 1

-- Domain condition
variables (x : ℝ) (hf_domain : 0 ≤ x ∧ x ≤ 3)

-- The main theorem
theorem find_x0 : (∃ x0: ℝ, f x0 = x0) → x = 2 :=
  sorry

end find_x0_l1864_186482


namespace probability_event_occurring_exactly_once_l1864_186493

theorem probability_event_occurring_exactly_once
  (P : ℝ)
  (h1 : ∀ n : ℕ, P ≥ 0 ∧ P ≤ 1) -- Probabilities are valid for all trials
  (h2 : (1 - (1 - P)^3) = 63 / 64) : -- Given condition for at least once
  (3 * P * (1 - P)^2 = 9 / 64) := 
by
  -- Here you would provide the proof steps using the conditions given.
  sorry

end probability_event_occurring_exactly_once_l1864_186493


namespace circle_radius_l1864_186450

theorem circle_radius (r : ℝ) (h : 3 * 2 * Real.pi * r = 2 * Real.pi * r^2) : r = 3 :=
by
  sorry

end circle_radius_l1864_186450


namespace A_B_work_together_finish_l1864_186432
noncomputable def work_rate_B := 1 / 12
noncomputable def work_rate_A := 2 * work_rate_B
noncomputable def combined_work_rate := work_rate_A + work_rate_B

theorem A_B_work_together_finish (hB: work_rate_B = 1/12) (hA: work_rate_A = 2 * work_rate_B) :
  (1 / combined_work_rate) = 4 :=
by
  -- Placeholder for the proof, we don't need to provide the proof steps
  sorry

end A_B_work_together_finish_l1864_186432


namespace students_more_than_pets_l1864_186473

-- Definition of given conditions
def num_students_per_classroom := 20
def num_rabbits_per_classroom := 2
def num_goldfish_per_classroom := 3
def num_classrooms := 5

-- Theorem stating the proof problem
theorem students_more_than_pets :
  let total_students := num_students_per_classroom * num_classrooms
  let total_pets := (num_rabbits_per_classroom + num_goldfish_per_classroom) * num_classrooms
  total_students - total_pets = 75 := by
  sorry

end students_more_than_pets_l1864_186473


namespace xy_plus_four_is_square_l1864_186424

theorem xy_plus_four_is_square (x y : ℕ) (h : ((1 / (x : ℝ)) + (1 / (y : ℝ)) + 1 / (x * y : ℝ)) = (1 / (x + 4 : ℝ) + 1 / (y - 4 : ℝ) + 1 / ((x + 4) * (y - 4) : ℝ))) : 
  ∃ (k : ℕ), xy + 4 = k^2 :=
by
  sorry

end xy_plus_four_is_square_l1864_186424


namespace student_weekly_allowance_l1864_186475

theorem student_weekly_allowance (A : ℝ) (h1 : (4 / 15) * A = 1) : A = 3.75 :=
by
  sorry

end student_weekly_allowance_l1864_186475


namespace fraction_eq_l1864_186444

def at_op (a b : ℝ) : ℝ := a * b - a * b^2
def hash_op (a b : ℝ) : ℝ := a^2 + b - a^2 * b

theorem fraction_eq :
  (at_op 8 3) / (hash_op 8 3) = 48 / 125 :=
by sorry

end fraction_eq_l1864_186444


namespace gibi_percentage_is_59_l1864_186489

-- Define the conditions
def max_score := 700
def avg_score := 490
def jigi_percent := 55
def mike_percent := 99
def lizzy_percent := 67

def jigi_score := (jigi_percent * max_score) / 100
def mike_score := (mike_percent * max_score) / 100
def lizzy_score := (lizzy_percent * max_score) / 100

def total_score := 4 * avg_score
def gibi_score := total_score - (jigi_score + mike_score + lizzy_score)

def gibi_percent := (gibi_score * 100) / max_score

-- The proof goal
theorem gibi_percentage_is_59 : gibi_percent = 59 := by
  sorry

end gibi_percentage_is_59_l1864_186489


namespace oranges_per_group_l1864_186427

theorem oranges_per_group (total_oranges groups : ℕ) (h1 : total_oranges = 384) (h2 : groups = 16) :
  total_oranges / groups = 24 := by
  sorry

end oranges_per_group_l1864_186427


namespace isosceles_triangle_angle_l1864_186443

theorem isosceles_triangle_angle
  (A B C : ℝ)
  (h1 : A = C)
  (h2 : B = 2 * A - 40)
  (h3 : A + B + C = 180) :
  B = 70 :=
by
  -- Proof omitted
  sorry

end isosceles_triangle_angle_l1864_186443


namespace clock_confusion_times_l1864_186401

-- Conditions translated into Lean definitions
def h_move : ℝ := 0.5  -- hour hand moves at 0.5 degrees per minute
def m_move : ℝ := 6.0  -- minute hand moves at 6 degrees per minute

-- Overlap condition formulated
def overlap_condition (n : ℕ) : Prop :=
  ∃ k : ℕ, k ≤ 10 ∧ 11 * (n : ℝ) = k * 360

-- The final theorem statement in Lean 4
theorem clock_confusion_times : 
  ∃ (count : ℕ), count = 132 ∧ 
    (∀ n < 144, (overlap_condition n → false)) :=
by
  -- Proof to be inserted here
  sorry

end clock_confusion_times_l1864_186401


namespace initial_number_of_persons_l1864_186413

theorem initial_number_of_persons (n : ℕ) 
  (w_increase : ∀ (k : ℕ), k = 4) 
  (old_weight new_weight : ℕ) 
  (h_old : old_weight = 58) 
  (h_new : new_weight = 106) 
  (h_difference : new_weight - old_weight = 48) 
  : n = 12 := 
by
  sorry

end initial_number_of_persons_l1864_186413


namespace bird_wings_l1864_186465

theorem bird_wings (P Pi C : ℕ) (h_total_money : 4 * 50 = 200)
  (h_total_cost : 30 * P + 20 * Pi + 15 * C = 200)
  (h_P_ge : P ≥ 1) (h_Pi_ge : Pi ≥ 1) (h_C_ge : C ≥ 1) :
  2 * (P + Pi + C) = 24 :=
sorry

end bird_wings_l1864_186465


namespace largest_n_for_perfect_square_l1864_186445

theorem largest_n_for_perfect_square :
  ∃ n : ℕ, 4 ^ 27 + 4 ^ 500 + 4 ^ n = k ^ 2 ∧ ∀ m : ℕ, 4 ^ 27 + 4 ^ 500 + 4 ^ m = l ^ 2 → m ≤ n  → n = 972 :=
sorry

end largest_n_for_perfect_square_l1864_186445


namespace necessary_condition_for_ellipse_l1864_186404

theorem necessary_condition_for_ellipse (m : ℝ) : 
  (5 - m > 0) → (m + 3 > 0) → (5 - m ≠ m + 3) → (-3 < m ∧ m < 5 ∧ m ≠ 1) :=
by sorry

end necessary_condition_for_ellipse_l1864_186404


namespace sum_of_powers_l1864_186477

theorem sum_of_powers : (-1: ℤ) ^ 2006 - (-1) ^ 2007 + 1 ^ 2008 + 1 ^ 2009 - 1 ^ 2010 = 3 := by
  sorry

end sum_of_powers_l1864_186477


namespace intersection_P_Q_range_a_l1864_186403

def set_P : Set ℝ := { x | 2 * x^2 - 3 * x + 1 ≤ 0 }
def set_Q (a : ℝ) : Set ℝ := { x | (x - a) * (x - a - 1) ≤ 0 }

theorem intersection_P_Q (a : ℝ) (h_a : a = 1) :
  set_P ∩ set_Q 1 = {1} :=
sorry

theorem range_a (a : ℝ) :
  (∀ x : ℝ, x ∈ set_P → x ∈ set_Q a) ↔ (0 ≤ a ∧ a ≤ 1/2) :=
sorry

end intersection_P_Q_range_a_l1864_186403


namespace kekai_garage_sale_l1864_186419

theorem kekai_garage_sale :
  let shirts := 5
  let shirt_price := 1
  let pants := 5
  let pant_price := 3
  let total_money := (shirts * shirt_price) + (pants * pant_price)
  let money_kept := total_money / 2
  money_kept = 10 :=
by
  sorry

end kekai_garage_sale_l1864_186419


namespace f_for_negative_x_l1864_186458

noncomputable def f (x : ℝ) : ℝ :=
if h : x > 0 then x * abs (x - 2) else 0  -- only assume the given case for x > 0

theorem f_for_negative_x (x : ℝ) (h : x < 0) : 
  f x = x * abs (x + 2) :=
by
  -- Sorry block to bypass the proof
  sorry

end f_for_negative_x_l1864_186458


namespace unique_integers_exist_l1864_186415

theorem unique_integers_exist (p : ℕ) (hp : p > 1) : 
  ∃ (a b c : ℤ), b^2 - 4*a*c = 1 - 4*p ∧ 0 < a ∧ a ≤ c ∧ -a ≤ b ∧ b < a :=
sorry

end unique_integers_exist_l1864_186415


namespace min_value_of_a_l1864_186426

theorem min_value_of_a (x y : ℝ) 
  (h1 : 0 < x) (h2 : 0 < y) 
  (h : ∀ x y, 0 < x → 0 < y → (x + y) * (1 / x + a / y) ≥ 9) :
  4 ≤ a :=
sorry

end min_value_of_a_l1864_186426


namespace find_a5_l1864_186454

variable {a : ℕ → ℝ} {q : ℝ}

def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop := 
  ∀ n, a (n+1) = a n * q

def is_increasing_sequence (a : ℕ → ℝ) : Prop := 
  ∀ n, a n < a (n+1)

def condition1 (a : ℕ → ℝ) : Prop :=
  a 5 ^ 2 = a 10

def condition2 (a : ℕ → ℝ) : Prop :=
  ∀ n, 2 * (a n + a (n+2)) = 5 * a (n+1)

theorem find_a5 (h1 : is_geometric_sequence a q) (h2 : is_increasing_sequence a) (h3 : condition1 a) (h4 : condition2 a) : 
  a 5 = 32 :=
sorry

end find_a5_l1864_186454


namespace factor_x10_minus_1024_l1864_186436

theorem factor_x10_minus_1024 (x : ℝ) : x^10 - 1024 = (x^5 + 32) * (x^5 - 32) :=
by
  sorry

end factor_x10_minus_1024_l1864_186436


namespace polynomial_value_l1864_186485

theorem polynomial_value (y : ℝ) (h : 4 * y^2 - 2 * y + 5 = 7) : 2 * y^2 - y + 1 = 2 :=
by
  sorry

end polynomial_value_l1864_186485


namespace new_foreign_students_l1864_186400

theorem new_foreign_students 
  (total_students : ℕ)
  (percent_foreign : ℕ)
  (foreign_students_next_sem : ℕ)
  (current_foreign_students : ℕ := total_students * percent_foreign / 100) : 
  total_students = 1800 → 
  percent_foreign = 30 → 
  foreign_students_next_sem = 740 → 
  foreign_students_next_sem - current_foreign_students = 200 :=
by
  intros
  sorry

end new_foreign_students_l1864_186400


namespace xy_solutions_l1864_186449

theorem xy_solutions : 
  ∀ (x y : ℕ), 0 < x → 0 < y →
  (xy ^ 2 + 7) ∣ (x^2 * y + x) →
  (x, y) = (7, 1) ∨ (x, y) = (14, 1) ∨ (x, y) = (35, 1) ∨ (x, y) = (7, 2) ∨ (∃ k : ℕ, x = 7 * k ∧ y = 7) :=
by
  sorry

end xy_solutions_l1864_186449


namespace correct_exponentiation_operation_l1864_186434

theorem correct_exponentiation_operation (a : ℝ) : (a^2)^3 = a^6 := 
by sorry

end correct_exponentiation_operation_l1864_186434


namespace carrie_first_day_miles_l1864_186483

theorem carrie_first_day_miles
  (x : ℕ)
  (h1 : ∀ y : ℕ, y = x + 124) -- Second day
  (h2 : ∀ y : ℕ, y = 159) -- Third day
  (h3 : ∀ y : ℕ, y = 189) -- Fourth day
  (h4 : ∀ z : ℕ, z = 106) -- Phone charge interval
  (h5 : ∀ n : ℕ, n = 7) -- Number of charges
  (h_total : 106 * 7 = x + (x + 124) + 159 + 189)
  : x = 135 :=
by sorry

end carrie_first_day_miles_l1864_186483


namespace bounded_region_area_l1864_186497

theorem bounded_region_area : 
  (∀ x y : ℝ, (y^2 + 4*x*y + 50*|x| = 500) → (x ≥ 0 ∧ y = 25 - 4*x) ∨ (x ≤ 0 ∧ y = -12.5 - 4*x)) →
  ∃ (A : ℝ), A = 156.25 :=
by
  sorry

end bounded_region_area_l1864_186497


namespace radha_profit_percentage_l1864_186402

theorem radha_profit_percentage (SP CP : ℝ) (hSP : SP = 144) (hCP : CP = 90) :
  ((SP - CP) / CP) * 100 = 60 := by
  sorry

end radha_profit_percentage_l1864_186402


namespace number_greater_by_l1864_186459

def question (a b : Int) : Int := a + b

theorem number_greater_by (a b : Int) : question a b = -11 :=
  by
    sorry

-- Use specific values from the provided problem:
example : question -5 -6 = -11 :=
  by
    sorry

end number_greater_by_l1864_186459


namespace sqrt_exp_sum_eq_eight_sqrt_two_l1864_186433

theorem sqrt_exp_sum_eq_eight_sqrt_two : 
  (Real.sqrt ((5 - 4 * Real.sqrt 2) ^ 2) + Real.sqrt ((5 + 4 * Real.sqrt 2) ^ 2) = 8 * Real.sqrt 2) :=
by
  sorry

end sqrt_exp_sum_eq_eight_sqrt_two_l1864_186433


namespace sqrt_9_minus_2_pow_0_plus_abs_neg1_l1864_186420

theorem sqrt_9_minus_2_pow_0_plus_abs_neg1 :
  (Real.sqrt 9 - 2^0 + abs (-1) = 3) :=
by
  -- Proof omitted for brevity
  sorry

end sqrt_9_minus_2_pow_0_plus_abs_neg1_l1864_186420


namespace driving_time_to_beach_l1864_186425

theorem driving_time_to_beach (total_trip_time : ℝ) (k : ℝ) (x : ℝ)
  (h1 : total_trip_time = 14)
  (h2 : k = 2.5)
  (h3 : total_trip_time = (2 * x) + (k * (2 * x))) :
  x = 2 := by 
  sorry

end driving_time_to_beach_l1864_186425


namespace thomas_payment_weeks_l1864_186411

theorem thomas_payment_weeks 
    (weekly_rate : ℕ) 
    (total_amount_paid : ℕ) 
    (h1 : weekly_rate = 4550) 
    (h2 : total_amount_paid = 19500) :
    (19500 / 4550 : ℕ) = 4 :=
by {
  sorry
}

end thomas_payment_weeks_l1864_186411


namespace two_integers_divide_2_pow_96_minus_1_l1864_186416

theorem two_integers_divide_2_pow_96_minus_1 : 
  ∃ a b : ℕ, (60 < a ∧ a < 70 ∧ 60 < b ∧ b < 70 ∧ a ≠ b ∧ a ∣ (2^96 - 1) ∧ b ∣ (2^96 - 1) ∧ a = 63 ∧ b = 65) := 
sorry

end two_integers_divide_2_pow_96_minus_1_l1864_186416


namespace find_y_l1864_186439

variable {x y : ℤ}
variables (h1 : y = 2 * x - 3) (h2 : x + y = 57)

theorem find_y : y = 37 :=
by {
    sorry
}

end find_y_l1864_186439


namespace beads_per_necklace_correct_l1864_186471
-- Importing the necessary library.

-- Defining the given number of necklaces and total beads.
def number_of_necklaces : ℕ := 11
def total_beads : ℕ := 308

-- Stating the proof goal as a theorem.
theorem beads_per_necklace_correct : (total_beads / number_of_necklaces) = 28 := 
by
  sorry

end beads_per_necklace_correct_l1864_186471


namespace total_population_l1864_186488

variables (b g t : ℕ)

theorem total_population (h1 : b = 4 * g) (h2 : g = 5 * t) : b + g + t = 26 * t :=
sorry

end total_population_l1864_186488


namespace gcd_sum_equality_l1864_186487

theorem gcd_sum_equality (n : ℕ) : 
  (Nat.gcd 6 n + Nat.gcd 8 (2 * n) = 10) ↔ 
  (∃ t : ℤ, n = 12 * t + 4 ∨ n = 12 * t + 6 ∨ n = 12 * t + 8) :=
by
  sorry

end gcd_sum_equality_l1864_186487


namespace maximum_value_f_zeros_l1864_186435

noncomputable def f (x : ℝ) (k : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 1 then k * x^2 + 2 * x - 1
  else if 1 < x then k * x + 1
  else 0

theorem maximum_value_f_zeros (k : ℝ) (x1 x2 : ℝ) :
  0 < k ∧ ∀ x, f x k = 0 ↔ x = x1 ∨ x = x2 → x1 ≠ x2 →
  x1 > 0 → x2 > 0 → -1 < k ∧ k < 0 →
  (x1 = -1 / k) ∧ (x2 = 1 / (1 + Real.sqrt (1 + k))) →
  ∃ y, (1 / x1) + (1 / x2) = y ∧ y = 9 / 4 := sorry

end maximum_value_f_zeros_l1864_186435


namespace number_corresponding_to_8_minutes_l1864_186422

theorem number_corresponding_to_8_minutes (x : ℕ) : 
  (12 / 6 = x / 480) → x = 960 :=
by
  sorry

end number_corresponding_to_8_minutes_l1864_186422


namespace simplify_expression_l1864_186442

def real_numbers (a b : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ a^3 + b^3 = a^2 + b^2

theorem simplify_expression (a b : ℝ) (h : real_numbers a b) :
  (a^2 / b + b^2 / a - 1 / (a * a * b * b)) = (a^4 + 2 * a * b + b^4 - 1) / (a * b) :=
by
  sorry

end simplify_expression_l1864_186442


namespace system_nonzero_solution_l1864_186446

-- Definition of the game setup and conditions
def initial_equations (a b c : ℤ) (x y z : ℤ) : Prop :=
  (a * x + b * y + c * z = 0) ∧
  (a * x + b * y + c * z = 0) ∧
  (a * x + b * y + c * z = 0)

-- The main proposition statement in Lean
theorem system_nonzero_solution :
  ∀ (a b c : ℤ), ∃ (x y z : ℤ), x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0 ∧ initial_equations a b c x y z :=
by
  sorry

end system_nonzero_solution_l1864_186446


namespace speed_of_second_train_l1864_186440

/-- 
Given:
1. A train leaves Mumbai at 9 am at a speed of 40 kmph.
2. After one hour, another train leaves Mumbai in the same direction at an unknown speed.
3. The two trains meet at a distance of 80 km from Mumbai.

Prove that the speed of the second train is 80 kmph.
-/
theorem speed_of_second_train (v : ℝ) :
  (∃ (distance_first : ℝ) (distance_meet : ℝ) (initial_speed_first : ℝ) (hours_later : ℤ),
    distance_first = 40 ∧ distance_meet = 80 ∧ initial_speed_first = 40 ∧ hours_later = 1 ∧
    v = distance_meet / (distance_meet / initial_speed_first - hours_later)) → v = 80 := by
  sorry

end speed_of_second_train_l1864_186440


namespace part1_part2_l1864_186468

section
variable (k : ℝ)

/-- Part 1: Range of k -/
def discriminant_eqn (k : ℝ) := (2 * k - 1) ^ 2 - 4 * (k ^ 2 - 1)

theorem part1 (h : discriminant_eqn k ≥ 0) : k ≤ 5 / 4 :=
by sorry

/-- Part 2: Value of k when x₁ and x₂ satisfy the given condition -/
def x1_x2_eqn (k x1 x2 : ℝ) := x1 ^ 2 + x2 ^ 2 = 16 + x1 * x2

def vieta (k : ℝ) (x1 x2 : ℝ) :=
  x1 + x2 = 1 - 2 * k ∧ x1 * x2 = k ^ 2 - 1

theorem part2 (x1 x2 : ℝ) (h1 : vieta k x1 x2) (h2 : x1_x2_eqn k x1 x2) : k = -2 :=
by sorry

end

end part1_part2_l1864_186468


namespace sad_employees_left_geq_cheerful_l1864_186412

-- Define the initial number of sad employees
def initial_sad_employees : Nat := 36

-- Define the final number of remaining employees after the game
def final_remaining_employees : Nat := 1

-- Define the total number of employees hit and out of the game
def employees_out : Nat := initial_sad_employees - final_remaining_employees

-- Define the number of cheerful employees who have left
def cheerful_employees_left := employees_out

-- Define the number of sad employees who have left
def sad_employees_left := employees_out

-- The theorem stating the problem proof
theorem sad_employees_left_geq_cheerful:
    sad_employees_left ≥ cheerful_employees_left :=
by
  -- Proof is omitted
  sorry

end sad_employees_left_geq_cheerful_l1864_186412


namespace prove_k_range_l1864_186457

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := a * x - b * Real.log x

theorem prove_k_range (a b k : ℝ) (h1 : a - b = 1) (h2 : f 1 a b = 2) :
  (∀ x ≥ 1, f x a b ≥ k * x) → k ≤ 2 - 1 / Real.exp 1 :=
by
  sorry

end prove_k_range_l1864_186457


namespace lights_ratio_l1864_186431

theorem lights_ratio (M S L : ℕ) (h1 : M = 12) (h2 : S = M + 10) (h3 : 118 = (S * 1) + (M * 2) + (L * 3)) :
  L = 24 ∧ L / M = 2 :=
by
  sorry

end lights_ratio_l1864_186431


namespace smallest_int_neither_prime_nor_square_no_prime_lt_70_l1864_186486

def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def has_no_prime_factor_less_than (n : ℕ) (k : ℕ) : Prop :=
  ∀ p : ℕ, is_prime p → p < k → ¬ p ∣ n

theorem smallest_int_neither_prime_nor_square_no_prime_lt_70
  (n : ℕ) : 
  n = 5183 ∧ ¬ is_prime n ∧ ¬ is_square n ∧ has_no_prime_factor_less_than n 70 ∧
  (∀ m : ℕ, 0 < m → m < 5183 →
    ¬ (¬ is_prime m ∧ ¬ is_square m ∧ has_no_prime_factor_less_than m 70)) :=
by sorry

end smallest_int_neither_prime_nor_square_no_prime_lt_70_l1864_186486


namespace find_y_l1864_186480

theorem find_y : ∃ y : ℝ, 1.5 * y - 10 = 35 ∧ y = 30 :=
by
  sorry

end find_y_l1864_186480


namespace coordinates_C_l1864_186467

theorem coordinates_C 
  (A : ℝ × ℝ) (B : ℝ × ℝ) (C : ℝ × ℝ) 
  (hA : A = (-1, 3)) 
  (hB : B = (11, 7))
  (hBC_AB : (C.1 - B.1, C.2 - B.2) = (2 / 3) • (B.1 - A.1, B.2 - A.2)) :
  C = (19, 29 / 3) :=
sorry

end coordinates_C_l1864_186467


namespace bus_speed_calculation_l1864_186447

noncomputable def bus_speed_excluding_stoppages : ℝ :=
  let effective_speed_with_stoppages := 50 -- kmph
  let stoppage_time_in_minutes := 13.125 -- minutes per hour
  let stoppage_time_in_hours := stoppage_time_in_minutes / 60 -- convert to hours
  let effective_moving_time := 1 - stoppage_time_in_hours -- effective moving time in one hour
  let bus_speed := (effective_speed_with_stoppages * 60) / (60 - stoppage_time_in_minutes) -- calculate bus speed
  bus_speed

theorem bus_speed_calculation : bus_speed_excluding_stoppages = 64 := by
  sorry

end bus_speed_calculation_l1864_186447


namespace min_value_of_quadratic_function_min_attained_at_negative_two_l1864_186496

def quadratic_function (x : ℝ) : ℝ := 3 * (x + 2)^2 - 5

theorem min_value_of_quadratic_function : ∀ x : ℝ, quadratic_function x ≥ -5 :=
by
  sorry

theorem min_attained_at_negative_two : quadratic_function (-2) = -5 :=
by
  sorry

end min_value_of_quadratic_function_min_attained_at_negative_two_l1864_186496


namespace values_of_m_zero_rain_l1864_186452

def f (x y : ℝ) : ℝ := abs (x^3 + 2*x^2*y - 5*x*y^2 - 6*y^3)

theorem values_of_m_zero_rain :
  {m : ℝ | ∀ x : ℝ, f x (m * x) = 0} = {-1, 1/2, -1/3} :=
sorry

end values_of_m_zero_rain_l1864_186452


namespace x_14_and_inverse_x_14_l1864_186408

theorem x_14_and_inverse_x_14 (x : ℂ) (h : x^2 + x + 1 = 0) : x^14 + x⁻¹^14 = -1 :=
by
  sorry

end x_14_and_inverse_x_14_l1864_186408


namespace radhika_total_games_l1864_186460

-- Define the conditions
def giftsOnChristmas := 12
def giftsOnBirthday := 8
def alreadyOwned := (giftsOnChristmas + giftsOnBirthday) / 2
def totalGifts := giftsOnChristmas + giftsOnBirthday
def expectedTotalGames := totalGifts + alreadyOwned

-- Define the proof statement
theorem radhika_total_games : 
  giftsOnChristmas = 12 ∧ giftsOnBirthday = 8 ∧ alreadyOwned = 10 
  ∧ totalGifts = 20 ∧ expectedTotalGames = 30 :=
by 
  sorry

end radhika_total_games_l1864_186460


namespace sugar_packs_l1864_186409

variable (totalSugar : ℕ) (packWeight : ℕ) (sugarLeft : ℕ)

noncomputable def numberOfPacks (totalSugar packWeight sugarLeft : ℕ) : ℕ :=
  (totalSugar - sugarLeft) / packWeight

theorem sugar_packs : numberOfPacks 3020 250 20 = 12 := by
  sorry

end sugar_packs_l1864_186409


namespace focus_parabola_l1864_186455

theorem focus_parabola (f : ℝ) (d : ℝ) (y : ℝ) :
  (∀ y, ((- (1 / 8) * y^2 - f) ^ 2 + y^2 = (- (1 / 8) * y^2 - d) ^ 2)) → 
  (d - f = 4) → 
  (f^2 = d^2) → 
  f = -2 :=
by
  sorry

end focus_parabola_l1864_186455


namespace min_value_problem_l1864_186492

noncomputable def min_value (a b c d e f : ℝ) := (2 / a) + (3 / b) + (9 / c) + (16 / d) + (25 / e) + (36 / f)

theorem min_value_problem 
  (a b c d e f : ℝ)
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : d > 0) (h5 : e > 0) (h6 : f > 0) 
  (h_sum : a + b + c + d + e + f = 10) : 
  min_value a b c d e f >= (329 + 38 * Real.sqrt 6) / 10 := 
sorry

end min_value_problem_l1864_186492


namespace power_division_l1864_186499

theorem power_division (a : ℝ) (h : a ≠ 0) : ((-a)^6) / (a^3) = a^3 := by
  sorry

end power_division_l1864_186499


namespace square_carpet_side_length_l1864_186406

theorem square_carpet_side_length (area : ℝ) (h : area = 10) :
  ∃ s : ℝ, s * s = area ∧ 3 < s ∧ s < 4 :=
by
  sorry

end square_carpet_side_length_l1864_186406


namespace shaded_squares_percentage_l1864_186414

theorem shaded_squares_percentage : 
  let grid_size := 6
  let total_squares := grid_size * grid_size
  let shaded_squares := total_squares / 2
  (shaded_squares / total_squares) * 100 = 50 :=
by
  /- Definitions and conditions -/
  let grid_size := 6
  let total_squares := grid_size * grid_size
  let shaded_squares := total_squares / 2

  /- Required proof statement -/
  have percentage_shaded : (shaded_squares / total_squares) * 100 = 50 := sorry

  /- Return the proof -/
  exact percentage_shaded

end shaded_squares_percentage_l1864_186414


namespace arithmetic_sum_of_11_terms_l1864_186491

variable {α : Type*} [LinearOrderedField α] (a : ℕ → α) (d : α)

def arithmetic_sequence (a : ℕ → α) (a₁ : α) (d : α) : Prop :=
∀ n, a n = a₁ + n * d

def sum_first_n_terms (a : ℕ → α) (n : ℕ) : α :=
(n + 1) * (a 0 + a n) / 2

theorem arithmetic_sum_of_11_terms
  (a₁ d : α)
  (a : ℕ → α)
  (h_seq : arithmetic_sequence a a₁ d)
  (h_cond : a 8 = (1 / 2) * a 11 + 3) :
  sum_first_n_terms a 10 = 66 := by
  sorry

end arithmetic_sum_of_11_terms_l1864_186491


namespace difference_in_earnings_in_currency_B_l1864_186438

-- Definitions based on conditions
def num_red_stamps : Nat := 30
def num_white_stamps : Nat := 80
def price_per_red_stamp_currency_A : Nat := 5
def price_per_white_stamp_currency_B : Nat := 50
def exchange_rate_A_to_B : Nat := 2

-- Theorem based on the question and correct answer
theorem difference_in_earnings_in_currency_B : 
  num_white_stamps * price_per_white_stamp_currency_B - 
  (num_red_stamps * price_per_red_stamp_currency_A * exchange_rate_A_to_B) = 3700 := 
  by
  sorry

end difference_in_earnings_in_currency_B_l1864_186438


namespace quadratic_integer_roots_l1864_186470

theorem quadratic_integer_roots (a b x : ℤ) :
  (∀ x₁ x₂ : ℤ, x₁ + x₂ = -b / a ∧ x₁ * x₂ = b / a → (x₁ = x₂ ∧ x₁ = -2 ∧ b = 4 * a) ∨ (x = -1 ∧ a = 0 ∧ b ≠ 0) ∨ (x = 0 ∧ a ≠ 0 ∧ b = 0)) :=
sorry

end quadratic_integer_roots_l1864_186470


namespace souvenir_cost_l1864_186417

def total_souvenirs : ℕ := 1000
def total_cost : ℝ := 220
def unknown_souvenirs : ℕ := 400
def known_cost : ℝ := 0.20

theorem souvenir_cost :
  ∃ x : ℝ, x = 0.25 ∧ total_cost = unknown_souvenirs * x + (total_souvenirs - unknown_souvenirs) * known_cost :=
by
  sorry

end souvenir_cost_l1864_186417


namespace claire_photos_l1864_186441

theorem claire_photos (L R C : ℕ) (h1 : L = R) (h2 : L = 3 * C) (h3 : R = C + 28) : C = 14 := by
  sorry

end claire_photos_l1864_186441


namespace solution_unique_l1864_186429

def is_solution (x : ℝ) : Prop :=
  ⌊x * ⌊x⌋⌋ = 48

theorem solution_unique (x : ℝ) : is_solution x → x = -48 / 7 :=
by
  intro h
  -- Proof goes here
  sorry

end solution_unique_l1864_186429


namespace largest_root_divisible_by_17_l1864_186423

theorem largest_root_divisible_by_17 (a : ℝ) (h : Polynomial.eval a (Polynomial.C 1 + Polynomial.C (-3) * Polynomial.X^2 + Polynomial.X^3) = 0) (root_large : ∀ x ∈ {b | Polynomial.eval b (Polynomial.C 1 + Polynomial.C (-3) * Polynomial.X^2 + Polynomial.X^3) = 0}, x ≤ a) :
  a^1788 % 17 = 0 ∧ a^1988 % 17 = 0 :=
by
  sorry

end largest_root_divisible_by_17_l1864_186423


namespace tan_A_in_right_triangle_l1864_186405

theorem tan_A_in_right_triangle (A B C : Type) [Inhabited A] [Inhabited B] [Inhabited C] (angle_A angle_B angle_C : ℝ) 
  (sin_B : ℚ) (tan_A : ℚ) :
  angle_C = 90 ∧ sin_B = 3 / 5 → tan_A = 4 / 3 := by
  sorry

end tan_A_in_right_triangle_l1864_186405


namespace rounds_played_l1864_186407

-- Define the given conditions as Lean constants
def totalPoints : ℝ := 378.5
def pointsPerRound : ℝ := 83.25

-- Define the goal as a Lean theorem
theorem rounds_played :
  Int.ceil (totalPoints / pointsPerRound) = 5 := 
by 
  sorry

end rounds_played_l1864_186407


namespace average_percent_increase_per_year_l1864_186466

-- Definitions and conditions
def initialPopulation : ℕ := 175000
def finalPopulation : ℕ := 297500
def numberOfYears : ℕ := 10

-- Statement to prove
theorem average_percent_increase_per_year : 
  ((finalPopulation - initialPopulation) / numberOfYears : ℚ) / initialPopulation * 100 = 7 := by
  sorry

end average_percent_increase_per_year_l1864_186466


namespace length_of_room_l1864_186428

theorem length_of_room (width : ℝ) (cost_per_sq_meter : ℝ) (total_cost : ℝ) (L : ℝ) 
  (h_width : width = 2.75)
  (h_cost_per_sq_meter : cost_per_sq_meter = 600)
  (h_total_cost : total_cost = 10725)
  (h_area_cost_eq : total_cost = L * width * cost_per_sq_meter) : 
  L = 6.5 :=
by 
  simp [h_width, h_cost_per_sq_meter, h_total_cost, h_area_cost_eq] at *
  sorry

end length_of_room_l1864_186428


namespace sum_of_squares_eq_2_l1864_186490

theorem sum_of_squares_eq_2 (a b : ℝ) 
  (h : (a^2 + b^2) * (a^2 + b^2 + 4) = 12) : a^2 + b^2 = 2 :=
by sorry

end sum_of_squares_eq_2_l1864_186490


namespace fraction_inequality_l1864_186495

-- Given the conditions
variables {c x y : ℝ} (h1 : c > x) (h2 : x > y) (h3 : y > 0)

-- Prove that \frac{x}{c-x} > \frac{y}{c-y}
theorem fraction_inequality (h4 : c > 0) : (x / (c - x)) > (y / (c - y)) :=
by {
  sorry  -- Proof to be completed
}

end fraction_inequality_l1864_186495
