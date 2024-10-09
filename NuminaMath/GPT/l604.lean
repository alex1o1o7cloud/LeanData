import Mathlib

namespace area_of_circle_l604_60468

def circleEquation (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 8*y = -9

theorem area_of_circle :
  (∃ (center : ℝ × ℝ) (radius : ℝ), radius = 4 ∧ ∀ (x y : ℝ), circleEquation x y ↔ (x - center.1)^2 + (y - center.2)^2 = radius^2) →
  (16 * Real.pi) = 16 * Real.pi := 
by 
  intro h
  have := h
  sorry

end area_of_circle_l604_60468


namespace circle_center_is_neg4_2_l604_60433

noncomputable def circle_center (x y : ℝ) : Prop :=
  x^2 + 8 * x + y^2 - 4 * y = 16

theorem circle_center_is_neg4_2 :
  ∃ (h k : ℝ), (h = -4 ∧ k = 2) ∧
  ∀ (x y : ℝ), circle_center x y ↔ (x + 4)^2 + (y - 2)^2 = 36 :=
by
  sorry

end circle_center_is_neg4_2_l604_60433


namespace fgf_of_3_l604_60420

-- Definitions of the functions f and g
def f (x : ℤ) : ℤ := 4 * x + 4
def g (x : ℤ) : ℤ := 5 * x + 2

-- The statement we need to prove
theorem fgf_of_3 : f (g (f 3)) = 332 := by
  sorry

end fgf_of_3_l604_60420


namespace range_of_a_l604_60452

variable (a x : ℝ)

def A : Set ℝ := {x | 1 ≤ x ∧ x ≤ 4}
def M (a : ℝ) : Set ℝ := if a = 2 then {2} else {x | 2 ≤ x ∧ x ≤ a}

theorem range_of_a (a : ℝ) (p : x ∈ M a) (h : a ≥ 2) (hpq : Set.Subset (M a) A) : 2 ≤ a ∧ a ≤ 4 :=
  sorry

end range_of_a_l604_60452


namespace average_beef_sales_l604_60485

theorem average_beef_sales 
  (thursday_sales : ℕ)
  (friday_sales : ℕ)
  (saturday_sales : ℕ)
  (h_thursday : thursday_sales = 210)
  (h_friday : friday_sales = 2 * thursday_sales)
  (h_saturday : saturday_sales = 150) :
  (thursday_sales + friday_sales + saturday_sales) / 3 = 260 :=
by sorry

end average_beef_sales_l604_60485


namespace range_of_a_l604_60445

def f (x a : ℝ) := |x - 2| + |x + a|

theorem range_of_a (a : ℝ) : (∀ x : ℝ, f x a ≥ 3) → a ≤ -5 ∨ a ≥ 1 :=
  sorry

end range_of_a_l604_60445


namespace negation_of_rectangular_parallelepipeds_have_12_edges_l604_60421

-- Define a structure for Rectangular Parallelepiped and the property of having edges
structure RectangularParallelepiped where
  hasEdges : ℕ → Prop

-- Problem statement
theorem negation_of_rectangular_parallelepipeds_have_12_edges :
  (∀ rect_p : RectangularParallelepiped, rect_p.hasEdges 12) →
  ∃ rect_p : RectangularParallelepiped, ¬ rect_p.hasEdges 12 := 
by
  sorry

end negation_of_rectangular_parallelepipeds_have_12_edges_l604_60421


namespace trigonometric_identity_l604_60476

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = 3) :
  2 * Real.sin α ^ 2 - Real.sin α * Real.cos α + Real.cos α ^ 2 = 8 / 5 := 
by
  sorry

end trigonometric_identity_l604_60476


namespace sin_squared_alpha_eq_one_add_sin_squared_beta_l604_60412

variable {α θ β : ℝ}

theorem sin_squared_alpha_eq_one_add_sin_squared_beta
  (h1 : Real.sin α = Real.sin θ + Real.cos θ)
  (h2 : Real.sin β ^ 2 = 2 * Real.sin θ * Real.cos θ) :
  Real.sin α ^ 2 = 1 + Real.sin β ^ 2 := 
sorry

end sin_squared_alpha_eq_one_add_sin_squared_beta_l604_60412


namespace first_class_seat_count_l604_60462

theorem first_class_seat_count :
  let seats_first_class := 10
  let seats_business_class := 30
  let seats_economy_class := 50
  let people_economy_class := seats_economy_class / 2
  let people_business_and_first := people_economy_class
  let unoccupied_business := 8
  let people_business_class := seats_business_class - unoccupied_business
  people_business_and_first - people_business_class = 3 := by
  sorry

end first_class_seat_count_l604_60462


namespace find_x_minus_y_l604_60430

theorem find_x_minus_y (x y : ℝ) (h1 : x + y = 8) (h2 : x^2 - y^2 = 12) : x - y = 3 / 2 :=
by
  sorry

end find_x_minus_y_l604_60430


namespace max_value_of_f_at_0_min_value_of_f_on_neg_inf_to_0_range_of_a_for_ineq_l604_60455

noncomputable def f (x : ℝ) : ℝ :=
  (x^2 + 5*x + 5) / Real.exp x

theorem max_value_of_f_at_0 :
  f 0 = 5 := by
  sorry

theorem min_value_of_f_on_neg_inf_to_0 :
  f (-3) = -Real.exp 3 := by
  sorry

theorem range_of_a_for_ineq :
  ∀ x : ℝ, x^2 + 5*x + 5 - a * Real.exp x ≥ 0 ↔ a ≤ -Real.exp 3 := by
  sorry

end max_value_of_f_at_0_min_value_of_f_on_neg_inf_to_0_range_of_a_for_ineq_l604_60455


namespace value_of_7_star_3_l604_60413

def star (a b : ℕ) : ℕ := 4 * a + 3 * b - a * b

theorem value_of_7_star_3 : star 7 3 = 16 :=
by
  -- Proof would go here
  sorry

end value_of_7_star_3_l604_60413


namespace haley_trees_initially_grew_l604_60417

-- Given conditions
def num_trees_died : ℕ := 2
def num_trees_survived : ℕ := num_trees_died + 7

-- Prove the total number of trees initially grown
theorem haley_trees_initially_grew : num_trees_died + num_trees_survived = 11 :=
by
  -- here we would provide the proof eventually
  sorry

end haley_trees_initially_grew_l604_60417


namespace second_pirate_gets_diamond_l604_60479

theorem second_pirate_gets_diamond (coins_bag1 coins_bag2 : ℕ) :
  (coins_bag1 ≤ 1 ∧ coins_bag2 ≤ 1) ∨ (coins_bag1 > 1 ∨ coins_bag2 > 1) →
  (∃ n k : ℕ, n % 2 = 0 → (coins_bag1 + n) = (coins_bag2 + k)) :=
sorry

end second_pirate_gets_diamond_l604_60479


namespace calculate_perimeter_l604_60489

noncomputable def length_square := 8
noncomputable def breadth_square := 8 -- since it's a square, length and breadth are the same
noncomputable def length_rectangle := 8
noncomputable def breadth_rectangle := 4

noncomputable def combined_length := length_square + length_rectangle
noncomputable def combined_breadth := breadth_square 

noncomputable def perimeter := 2 * (combined_length + combined_breadth)

theorem calculate_perimeter : 
  length_square = 8 ∧ 
  breadth_square = 8 ∧ 
  length_rectangle = 8 ∧ 
  breadth_rectangle = 4 ∧ 
  perimeter = 48 := 
by 
  sorry

end calculate_perimeter_l604_60489


namespace log_product_zero_l604_60422

theorem log_product_zero :
  (Real.log 3 / Real.log 2 + Real.log 27 / Real.log 2) *
  (Real.log 4 / Real.log 4 + Real.log (1 / 4) / Real.log 4) = 0 := by
  -- Place proof here
  sorry

end log_product_zero_l604_60422


namespace negation_of_proposition_l604_60469

theorem negation_of_proposition (a b : ℝ) : 
  ¬(a + b = 1 → a^2 + b^2 ≥ 1/2) ↔ (a + b ≠ 1 → a^2 + b^2 < 1/2) :=
by sorry

end negation_of_proposition_l604_60469


namespace karens_class_fund_l604_60456

noncomputable def ratio_of_bills (T W : ℕ) : ℕ × ℕ := (T / Nat.gcd T W, W / Nat.gcd T W)

theorem karens_class_fund (T W : ℕ) (hW : W = 3) (hfund : 10 * T + 20 * W = 120) :
  ratio_of_bills T W = (2, 1) :=
by
  sorry

end karens_class_fund_l604_60456


namespace total_percent_decrease_is_19_l604_60442

noncomputable def original_value : ℝ := 100
noncomputable def first_year_decrease : ℝ := 0.10
noncomputable def second_year_decrease : ℝ := 0.10
noncomputable def value_after_first_year : ℝ := original_value * (1 - first_year_decrease)
noncomputable def value_after_second_year : ℝ := value_after_first_year * (1 - second_year_decrease)
noncomputable def total_decrease_in_dollars : ℝ := original_value - value_after_second_year
noncomputable def total_percent_decrease : ℝ := (total_decrease_in_dollars / original_value) * 100

theorem total_percent_decrease_is_19 :
  total_percent_decrease = 19 := by
  sorry

end total_percent_decrease_is_19_l604_60442


namespace n_squared_plus_m_squared_odd_implies_n_plus_m_not_even_l604_60481

theorem n_squared_plus_m_squared_odd_implies_n_plus_m_not_even (n m : ℤ) (h : (n^2 + m^2) % 2 = 1) : (n + m) % 2 ≠ 0 := by
  sorry

end n_squared_plus_m_squared_odd_implies_n_plus_m_not_even_l604_60481


namespace true_proposition_is_D_l604_60415

open Real

theorem true_proposition_is_D :
  (∃ x_0 : ℝ, exp x_0 ≤ 0) = False ∧
  (∀ x : ℝ, 2 ^ x > x ^ 2) = False ∧
  (∀ a b : ℝ, a + b = 0 ↔ a / b = -1) = False ∧
  (∀ a b : ℝ, a > 1 ∧ b > 1 → a * b > 1) = True :=
by
    sorry

end true_proposition_is_D_l604_60415


namespace number_of_dolls_l604_60443

theorem number_of_dolls (total_toys : ℕ) (fraction_action_figures : ℚ) 
  (remaining_fraction_action_figures : fraction_action_figures = 1 / 4) 
  (remaining_fraction_dolls : 1 - fraction_action_figures = 3 / 4) 
  (total_toys_eq : total_toys = 24) : 
  (total_toys - total_toys * fraction_action_figures) = 18 := 
by 
  sorry

end number_of_dolls_l604_60443


namespace pupils_in_program_l604_60466

theorem pupils_in_program {total_people parents : ℕ} (h1 : total_people = 238) (h2 : parents = 61) :
  total_people - parents = 177 := by
  sorry

end pupils_in_program_l604_60466


namespace three_pow_forty_gt_four_pow_thirty_gt_five_pow_twenty_sixteen_pow_thirty_one_gt_eight_pow_forty_one_gt_four_pow_sixty_one_a_lt_b_l604_60483

-- Problem (1)
theorem three_pow_forty_gt_four_pow_thirty_gt_five_pow_twenty : 3^40 > 4^30 ∧ 4^30 > 5^20 := 
by
  sorry

-- Problem (2)
theorem sixteen_pow_thirty_one_gt_eight_pow_forty_one_gt_four_pow_sixty_one : 16^31 > 8^41 ∧ 8^41 > 4^61 :=
by 
  sorry

-- Problem (3)
theorem a_lt_b (a b : ℝ) (h1 : 1 < a) (h2 : 1 < b) (h3 : a^5 = 2) (h4 : b^7 = 3) : a < b :=
by
  sorry

end three_pow_forty_gt_four_pow_thirty_gt_five_pow_twenty_sixteen_pow_thirty_one_gt_eight_pow_forty_one_gt_four_pow_sixty_one_a_lt_b_l604_60483


namespace smallest_a_value_l604_60490

theorem smallest_a_value {a b c : ℝ} :
  (∃ (a b c : ℝ), (∀ x, (a * (x - 1/2)^2 - 5/4 = a * x^2 + b * x + c)) ∧ a > 0 ∧ ∃ n : ℤ, a + b + c = n)
  → (∃ (a : ℝ), a = 1) :=
by
  sorry

end smallest_a_value_l604_60490


namespace equation_solution_l604_60406

theorem equation_solution (x : ℝ) :
  (1 / x + 1 / (x + 2) - 1 / (x + 4) - 1 / (x + 6) + 1 / (x + 8) = 0) →
  (x = -4 - 2 * Real.sqrt 3) ∨ (x = 2 - 2 * Real.sqrt 3) := by
  sorry

end equation_solution_l604_60406


namespace cost_of_fencing_l604_60411

-- Define the conditions
def width_garden : ℕ := 12
def length_playground : ℕ := 16
def width_playground : ℕ := 12
def price_per_meter : ℕ := 15
def area_playground : ℕ := length_playground * width_playground
def area_garden : ℕ := area_playground
def length_garden : ℕ := area_garden / width_garden
def perimeter_garden : ℕ := 2 * (length_garden + width_garden)
def cost_fencing : ℕ := perimeter_garden * price_per_meter

-- State the theorem
theorem cost_of_fencing : cost_fencing = 840 := by
  sorry

end cost_of_fencing_l604_60411


namespace tom_gave_fred_balloons_l604_60494

variable (initial_balloons : ℕ) (remaining_balloons : ℕ)

def balloons_given (initial remaining : ℕ) : ℕ :=
  initial - remaining

theorem tom_gave_fred_balloons (h₀ : initial_balloons = 30) (h₁ : remaining_balloons = 14) :
  balloons_given initial_balloons remaining_balloons = 16 :=
by
  -- Here we are skipping the proof
  sorry

end tom_gave_fred_balloons_l604_60494


namespace factorial_quotient_l604_60436

/-- Prove that the quotient of the factorial of 4! divided by 4! simplifies to 23!. -/
theorem factorial_quotient : (Nat.factorial (Nat.factorial 4)) / (Nat.factorial 4) = Nat.factorial 23 := 
by
  sorry

end factorial_quotient_l604_60436


namespace number_of_paths_from_A_to_D_l604_60428

-- Definitions based on conditions
def paths_A_to_B : ℕ := 2
def paths_B_to_C : ℕ := 2
def paths_A_to_C : ℕ := 1
def paths_C_to_D : ℕ := 2
def paths_B_to_D : ℕ := 2

-- Theorem statement
theorem number_of_paths_from_A_to_D : 
  paths_A_to_B * paths_B_to_C * paths_C_to_D + 
  paths_A_to_C * paths_C_to_D + 
  paths_A_to_B * paths_B_to_D = 14 :=
by {
  -- proof steps will go here
  sorry
}

end number_of_paths_from_A_to_D_l604_60428


namespace perimeter_large_star_l604_60423

theorem perimeter_large_star (n m : ℕ) (P : ℕ)
  (triangle_perimeter : ℕ) (quad_perimeter : ℕ) (small_star_perimeter : ℕ)
  (hn : n = 5) (hm : m = 5)
  (h_triangle_perimeter : triangle_perimeter = 7)
  (h_quad_perimeter : quad_perimeter = 18)
  (h_small_star_perimeter : small_star_perimeter = 3) :
  m * quad_perimeter + small_star_perimeter = n * triangle_perimeter + P → P = 58 :=
by 
  -- Placeholder proof
  sorry

end perimeter_large_star_l604_60423


namespace question1_geometric_sequence_question2_minimum_term_l604_60426

theorem question1_geometric_sequence (a : ℕ → ℝ) (p : ℝ) (q : ℝ) :
  (∀ n : ℕ, a (n + 1) = a n + p * (3 ^ n) - n * q) →
  q = 0 →
  (a 1 = 1 / 2) →
  (∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a 1 * (r ^ n)) →
  (p = 0 ∨ p = 1) :=
by sorry

theorem question2_minimum_term (a : ℕ → ℝ) (p : ℝ) (q : ℝ) :
  (∀ n : ℕ, a (n + 1) = a n + p * (3 ^ n) - n * q) →
  p = 1 →
  (a 1 = 1 / 2) →
  (a 4 = min (min (a 1) (a 2)) (a 3)) →
  3 ≤ q ∧ q ≤ 27 / 4 :=
by sorry

end question1_geometric_sequence_question2_minimum_term_l604_60426


namespace cryptarithm_base_solution_l604_60438

theorem cryptarithm_base_solution :
  ∃ (K I T : ℕ) (d : ℕ), 
    O = 0 ∧
    2 * T = I ∧
    T + 1 = K ∧
    K + I = d ∧ 
    d = 7 ∧ 
    K ≠ I ∧ K ≠ T ∧ K ≠ O ∧
    I ≠ T ∧ I ≠ O ∧
    T ≠ O :=
sorry

end cryptarithm_base_solution_l604_60438


namespace max_n_m_sum_l604_60457

-- Definition of the function f
def f (x : ℝ) : ℝ := -x^2 + 4 * x

-- Statement of the problem
theorem max_n_m_sum {m n : ℝ} (h : n > m) (h_range : ∀ x, m ≤ x ∧ x ≤ n → -5 ≤ f x ∧ f x ≤ 4) : n + m = 7 :=
sorry

end max_n_m_sum_l604_60457


namespace lilies_per_centerpiece_correct_l604_60439

-- Definitions based on the conditions
def num_centerpieces : ℕ := 6
def roses_per_centerpiece : ℕ := 8
def cost_per_flower : ℕ := 15
def total_budget : ℕ := 2700

-- Definition of the number of orchids per centerpiece using given condition
def orchids_per_centerpiece : ℕ := 2 * roses_per_centerpiece

-- Definition of the total cost for roses and orchids before calculating lilies
def total_rose_cost : ℕ := num_centerpieces * roses_per_centerpiece * cost_per_flower
def total_orchid_cost : ℕ := num_centerpieces * orchids_per_centerpiece * cost_per_flower
def total_rose_and_orchid_cost : ℕ := total_rose_cost + total_orchid_cost

-- Definition for the remaining budget for lilies
def remaining_budget_for_lilies : ℕ := total_budget - total_rose_and_orchid_cost

-- Number of lilies in total and per centerpiece
def total_lilies : ℕ := remaining_budget_for_lilies / cost_per_flower
def lilies_per_centerpiece : ℕ := total_lilies / num_centerpieces

-- The proof statement we want to assert
theorem lilies_per_centerpiece_correct : lilies_per_centerpiece = 6 :=
by
  sorry

end lilies_per_centerpiece_correct_l604_60439


namespace inequality_sqrt_sum_ge_one_l604_60444

variable (a b c : ℝ)
variable (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c)
variable (prod_abc : a * b * c = 1)

theorem inequality_sqrt_sum_ge_one :
  (Real.sqrt (a / (8 + a)) + Real.sqrt (b / (8 + b)) + Real.sqrt (c / (8 + c)) ≥ 1) :=
by
  sorry

end inequality_sqrt_sum_ge_one_l604_60444


namespace total_sours_is_123_l604_60495

noncomputable def cherry_sours := 32
noncomputable def lemon_sours := 40 -- Derived from the ratio 4/5 = 32/x
noncomputable def orange_sours := 24 -- 25% of the total sours in the bag after adding them
noncomputable def grape_sours := 27 -- Derived from the ratio 3/2 = 40/y

theorem total_sours_is_123 :
  cherry_sours + lemon_sours + orange_sours + grape_sours = 123 :=
by
  sorry

end total_sours_is_123_l604_60495


namespace calculate_expression_l604_60461

theorem calculate_expression : (7^2 - 5^2)^3 = 13824 := by
  sorry

end calculate_expression_l604_60461


namespace parallel_vectors_x_value_l604_60419

-- Defining the vectors a and b
def a : ℝ × ℝ := (4, 2)
def b (x : ℝ) : ℝ × ℝ := (x, 3)

-- Condition for vectors a and b to be parallel
def are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 - a.2 * b.1 = 0

theorem parallel_vectors_x_value : ∃ x, are_parallel a (b x) ∧ x = 6 := by
  sorry

end parallel_vectors_x_value_l604_60419


namespace inequality_solution_l604_60472

theorem inequality_solution (x y : ℝ) : 
  (x^2 - 4 * x * y + 4 * x^2 < x^2) ↔ (x < y ∧ y < 3 * x ∧ x > 0) := 
sorry

end inequality_solution_l604_60472


namespace hypotenuse_length_l604_60488

def triangle_hypotenuse (x : ℝ) (h : ℝ) : Prop :=
  (3 * x - 3)^2 + x^2 = h^2 ∧
  (1 / 2) * x * (3 * x - 3) = 72

theorem hypotenuse_length :
  ∃ (x h : ℝ), triangle_hypotenuse x h ∧ h = Real.sqrt 505 :=
by
  sorry

end hypotenuse_length_l604_60488


namespace output_of_program_l604_60498

def loop_until (i S : ℕ) : ℕ :=
if i < 9 then S
else loop_until (i - 1) (S * i)

theorem output_of_program : loop_until 11 1 = 990 :=
sorry

end output_of_program_l604_60498


namespace height_of_second_triangle_l604_60451

theorem height_of_second_triangle
  (base1 : ℝ) (height1 : ℝ) (base2 : ℝ) (height2 : ℝ)
  (h_base1 : base1 = 15)
  (h_height1 : height1 = 12)
  (h_base2 : base2 = 20)
  (h_area_relation : (base2 * height2) / 2 = 2 * (base1 * height1) / 2) :
  height2 = 18 :=
sorry

end height_of_second_triangle_l604_60451


namespace problem_l604_60427

variable {w z : ℝ}

theorem problem (hw : w = 8) (hz : z = 3) (h : ∀ z w, z * (w^(1/3)) = 6) : w = 1 :=
by
  sorry

end problem_l604_60427


namespace solution_set_of_inequality_system_l604_60496

theorem solution_set_of_inequality_system (x : ℝ) :
  (3 * x - 1 ≥ x + 1) ∧ (x + 4 > 4 * x - 2) ↔ (1 ≤ x ∧ x < 2) := 
by
  sorry

end solution_set_of_inequality_system_l604_60496


namespace probability_both_A_B_selected_l604_60497

-- Define the problem with the necessary conditions.
def probability_AB_selected (total_students : ℕ) (select_students : ℕ) (A B : ℕ) : ℚ :=
  if A < total_students ∧ B < total_students ∧ total_students = 5 ∧ select_students = 3 then
    let total_ways := (Nat.factorial total_students) / ((Nat.factorial select_students) * (Nat.factorial (total_students - select_students)))
    let favorable_ways := (Nat.factorial (total_students - 2)) / ((Nat.factorial (select_students - 2)) * (Nat.factorial ((total_students - 2) - (select_students - 2))))
    favorable_ways / total_ways
  else 0

theorem probability_both_A_B_selected 
  (total_students : ℕ) (select_students : ℕ) (A B : ℕ) 
  (h1 : total_students = 5) (h2 : select_students = 3) 
  (h3 : A < total_students) (h4 : B < total_students) 
  : probability_AB_selected total_students select_students A B = 3 / 10 := 
by {
  -- Insert the delta and logic to prove the theorem here.
  sorry
}

end probability_both_A_B_selected_l604_60497


namespace greatest_number_divides_with_remainders_l604_60460

theorem greatest_number_divides_with_remainders (d : ℕ) :
  (1657 % d = 6) ∧ (2037 % d = 5) → d = 127 :=
by
  sorry

end greatest_number_divides_with_remainders_l604_60460


namespace max_rational_sums_is_1250_l604_60465

/-- We define a structure to represent the problem's conditions. -/
structure GridConfiguration where
  grid_rows : Nat
  grid_cols : Nat
  total_numbers : Nat
  rational_count : Nat
  irrational_count : Nat
  (h_grid : grid_rows = 50)
  (h_grid_col : grid_cols = 50)
  (h_total_numbers : total_numbers = 100)
  (h_rational_count : rational_count = 50)
  (h_irrational_count : irrational_count = 50)

/-- We define a function to calculate the number of rational sums in the grid. -/
def max_rational_sums (config : GridConfiguration) : Nat :=
  let x := config.rational_count / 2 -- rational numbers to the left
  let ni := 2 * x * x - 100 * x + 2500
  let rational_sums := 2500 - ni
  rational_sums

/-- The theorem stating the maximum number of rational sums is 1250. -/
theorem max_rational_sums_is_1250 (config : GridConfiguration) : max_rational_sums config = 1250 :=
  sorry

end max_rational_sums_is_1250_l604_60465


namespace jill_marathon_time_l604_60449

def jack_marathon_distance : ℝ := 42
def jack_marathon_time : ℝ := 6
def speed_ratio : ℝ := 0.7

theorem jill_marathon_time :
  ∃ t_jill : ℝ, (t_jill = jack_marathon_distance / (jack_marathon_distance / jack_marathon_time / speed_ratio)) ∧
  t_jill = 4.2 :=
by
  -- The proof goes here
  sorry

end jill_marathon_time_l604_60449


namespace number_of_paths_K_to_L_l604_60464

-- Definition of the problem structure
def K : Type := Unit
def A : Type := Unit
def R : Type := Unit
def L : Type := Unit

-- Defining the number of paths between each stage
def paths_from_K_to_A := 2
def paths_from_A_to_R := 4
def paths_from_R_to_L := 8

-- The main theorem stating the number of paths from K to L
theorem number_of_paths_K_to_L : paths_from_K_to_A * 2 * 2 = 8 := by 
  sorry

end number_of_paths_K_to_L_l604_60464


namespace weaving_increase_l604_60418

theorem weaving_increase (a₁ : ℕ) (S₃₀ : ℕ) (d : ℚ) (hₐ₁ : a₁ = 5) (hₛ₃₀ : S₃₀ = 390)
  (h_sum : S₃₀ = 30 * (a₁ + (a₁ + 29 * d)) / 2) : d = 16 / 29 :=
by {
  sorry
}

end weaving_increase_l604_60418


namespace hyperbola_sufficient_but_not_necessary_l604_60474

theorem hyperbola_sufficient_but_not_necessary :
  (∀ (C : Type) (x y : ℝ), C = {p : ℝ × ℝ | ((p.1)^2 / 16) - ((p.2)^2 / 9) = 1} →
  (∀ x, y = 3 * (x / 4) ∨ y = -3 * (x / 4)) →
  ∃ (C' : Type) (x' y' : ℝ), C' = {p : ℝ × ℝ | ((p.1)^2 / 64) - ((p.2)^2 / 36) = 1} ∧
  (∀ x', y' = 3 * (x' / 4) ∨ y' = -3 * (x' / 4))) :=
sorry

end hyperbola_sufficient_but_not_necessary_l604_60474


namespace simplify_trig_expression_l604_60486

open Real

theorem simplify_trig_expression (α : ℝ) : 
  (cos (2 * π + α) * tan (π + α)) / cos (π / 2 - α) = 1 := 
sorry

end simplify_trig_expression_l604_60486


namespace circle_radius_inscribed_l604_60429

noncomputable def a : ℝ := 6
noncomputable def b : ℝ := 12
noncomputable def c : ℝ := 18

noncomputable def r : ℝ :=
  let term1 := 1/a
  let term2 := 1/b
  let term3 := 1/c
  let sqrt_term := Real.sqrt ((1/(a * b)) + (1/(a * c)) + (1/(b * c)))
  1 / ((term1 + term2 + term3) + 2 * sqrt_term)

theorem circle_radius_inscribed :
  r = 36 / 17 := 
by
  sorry

end circle_radius_inscribed_l604_60429


namespace find_range_of_a_l604_60400

noncomputable def f (a : ℝ) : ℝ → ℝ :=
  λ x => a * (x - 2 * Real.exp 1) * Real.log x + 1

def range_of_a (a : ℝ) : Prop :=
  (a < 0 ∨ a > 1 / Real.exp 1)

theorem find_range_of_a (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f a x₁ = 0 ∧ f a x₂ = 0) ↔ range_of_a a := by
  sorry

end find_range_of_a_l604_60400


namespace train_passes_man_in_12_seconds_l604_60402

noncomputable def time_to_pass_man (train_length: ℝ) (train_speed_kmph: ℝ) (man_speed_kmph: ℝ) : ℝ :=
  let relative_speed_kmph := train_speed_kmph + man_speed_kmph
  let relative_speed_mps := relative_speed_kmph * (5 / 18)
  train_length / relative_speed_mps

theorem train_passes_man_in_12_seconds :
  time_to_pass_man 220 60 6 = 12 := by
 sorry

end train_passes_man_in_12_seconds_l604_60402


namespace total_distance_biked_two_days_l604_60431

def distance_yesterday : ℕ := 12
def distance_today : ℕ := (2 * distance_yesterday) - 3
def total_distance_biked : ℕ := distance_yesterday + distance_today

theorem total_distance_biked_two_days : total_distance_biked = 33 :=
by {
  -- Given distance_yesterday = 12
  -- distance_today calculated as (2 * distance_yesterday) - 3 = 21
  -- total_distance_biked = distance_yesterday + distance_today = 33
  sorry
}

end total_distance_biked_two_days_l604_60431


namespace find_tangent_line_l604_60408

def is_perpendicular (m1 m2 : ℝ) : Prop :=
  m1 * m2 = -1

def is_tangent_to_circle (a b c : ℝ) : Prop :=
  let d := abs c / (Real.sqrt (a^2 + b^2))
  d = 1

def in_first_quadrant (x y : ℝ) : Prop :=
  x > 0 ∧ y > 0

theorem find_tangent_line :
  ∀ (k b : ℝ),
    is_perpendicular k 1 →
    is_tangent_to_circle 1 1 b →
    ∃ (x y : ℝ), in_first_quadrant x y ∧ x + y - b = 0 →
    b = Real.sqrt 2 := sorry

end find_tangent_line_l604_60408


namespace find_x_from_average_l604_60435

theorem find_x_from_average :
  let sum_series := 5151
  let n := 102
  let known_average := 50 * (x + 1)
  (sum_series + x) / n = known_average → 
  x = 51 / 5099 :=
by
  intros
  sorry

end find_x_from_average_l604_60435


namespace smallest_number_starting_with_five_l604_60424

theorem smallest_number_starting_with_five :
  ∃ n : ℕ, ∃ m : ℕ, m = (5 * m + 5) / 4 ∧ 5 * n + m = 512820 ∧ m < 10^6 := sorry

end smallest_number_starting_with_five_l604_60424


namespace specimen_exchange_l604_60467

theorem specimen_exchange (x : ℕ) (h : x * (x - 1) = 110) : x * (x - 1) = 110 := by
  exact h

end specimen_exchange_l604_60467


namespace market_value_of_stock_l604_60434

variable (face_value : ℝ) (annual_dividend yield : ℝ)

-- Given conditions:
def stock_four_percent := annual_dividend = 0.04 * face_value
def stock_yield_five_percent := yield = 0.05

-- Problem statement:
theorem market_value_of_stock (face_value := 100) (annual_dividend := 4) (yield := 0.05) 
  (h1 : stock_four_percent face_value annual_dividend) 
  (h2 : stock_yield_five_percent yield) : 
  (4 / 0.05) * 100 = 80 :=
by
  sorry

end market_value_of_stock_l604_60434


namespace circles_disjoint_l604_60440

theorem circles_disjoint (a : ℝ) : ((x - 1)^2 + (y - 1)^2 = 4) ∧ (x^2 + (y - a)^2 = 1) → (a < 1 - 2 * Real.sqrt 2 ∨ a > 1 + 2 * Real.sqrt 2) :=
by sorry

end circles_disjoint_l604_60440


namespace hyperbola_focus_distance_l604_60473
open Real

theorem hyperbola_focus_distance
  (a b : ℝ)
  (ha : a = 5)
  (hb : b = 3)
  (hyperbola_eq : ∀ x y : ℝ, (x^2 / a^2 - y^2 / b^2 = 1) ↔ (∃ M : ℝ × ℝ, M = (x, y)))
  (M : ℝ × ℝ)
  (hM_on_hyperbola : ∃ x y : ℝ, M = (x, y) ∧ x^2 / a^2 - y^2 / b^2 = 1)
  (F1_pos : ℝ)
  (h_dist_F1 : dist M (F1_pos, 0) = 18) :
  (∃ (F2_dist : ℝ), (F2_dist = 8 ∨ F2_dist = 28) ∧ dist M (F2_dist, 0) = F2_dist) := 
sorry

end hyperbola_focus_distance_l604_60473


namespace nonempty_solution_set_iff_a_gt_2_l604_60441

theorem nonempty_solution_set_iff_a_gt_2 (a : ℝ) : 
  (∃ x : ℝ, |x - 3| + |x - 5| < a) ↔ a > 2 :=
sorry

end nonempty_solution_set_iff_a_gt_2_l604_60441


namespace sum_of_common_divisors_36_48_l604_60405

-- Definitions based on the conditions
def is_divisor (n d : ℕ) : Prop := d ∣ n

-- List of divisors for 36 and 48
def divisors_36 : List ℕ := [1, 2, 3, 4, 6, 9, 12, 18, 36]
def divisors_48 : List ℕ := [1, 2, 3, 4, 6, 8, 12, 16, 24, 48]

-- Definition of common divisors
def common_divisors_36_48 : List ℕ := [1, 2, 3, 4, 6, 12]

-- Sum of common divisors
def sum_common_divisors_36_48 := common_divisors_36_48.sum

-- The statement of the theorem
theorem sum_of_common_divisors_36_48 : sum_common_divisors_36_48 = 28 := by
  sorry

end sum_of_common_divisors_36_48_l604_60405


namespace point_coordinates_l604_60448

def point_in_second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0 

theorem point_coordinates (m : ℝ) 
  (h1 : point_in_second_quadrant (-m-1) (2*m+1))
  (h2 : |2*m + 1| = 5) : (-m-1, 2*m+1) = (-3, 5) :=
sorry

end point_coordinates_l604_60448


namespace number_of_older_females_l604_60471

theorem number_of_older_females (total_population : ℕ) (num_groups : ℕ) (one_group_population : ℕ) :
  total_population = 1000 → num_groups = 5 → total_population = num_groups * one_group_population →
  one_group_population = 200 :=
by
  intro h1 h2 h3
  sorry

end number_of_older_females_l604_60471


namespace original_amount_charged_l604_60403

variables (P : ℝ) (interest_rate : ℝ) (total_owed : ℝ)

theorem original_amount_charged :
  interest_rate = 0.09 →
  total_owed = 38.15 →
  (P + P * interest_rate = total_owed) →
  P = 35 :=
by
  intros h_interest_rate h_total_owed h_equation
  sorry

end original_amount_charged_l604_60403


namespace sqrt_meaningful_iff_l604_60410

theorem sqrt_meaningful_iff (x : ℝ) : (3 - x ≥ 0) ↔ (x ≤ 3) := by
  sorry

end sqrt_meaningful_iff_l604_60410


namespace range_of_k_l604_60459

noncomputable def operation (a b : ℝ) : ℝ := Real.sqrt (a * b) + a + b

theorem range_of_k (k : ℝ) (h : operation 1 (k^2) < 3) : -1 < k ∧ k < 1 :=
by
  sorry

end range_of_k_l604_60459


namespace harry_lost_sea_creatures_l604_60407

def initial_sea_stars := 34
def initial_seashells := 21
def initial_snails := 29
def initial_crabs := 17

def sea_stars_reproduced := 5
def seashells_reproduced := 3
def snails_reproduced := 4

def final_items := 105

def sea_stars_after_reproduction := initial_sea_stars + (sea_stars_reproduced * 2 - sea_stars_reproduced)
def seashells_after_reproduction := initial_seashells + (seashells_reproduced * 2 - seashells_reproduced)
def snails_after_reproduction := initial_snails + (snails_reproduced * 2 - snails_reproduced)
def crabs_after_reproduction := initial_crabs

def total_after_reproduction := sea_stars_after_reproduction + seashells_after_reproduction + snails_after_reproduction + crabs_after_reproduction

theorem harry_lost_sea_creatures : total_after_reproduction - final_items = 8 :=
by
  sorry

end harry_lost_sea_creatures_l604_60407


namespace num_distinct_values_for_sum_l604_60493

theorem num_distinct_values_for_sum (x y z : ℝ) 
  (h : (x^2 - 9)^2 + (y^2 - 4)^2 + (z^2 - 1)^2 = 0) :
  ∃ s : Finset ℝ, 
  (∀ x y z, (x^2 - 9)^2 + (y^2 - 4)^2 + (z^2 - 1)^2 = 0 → (x + y + z) ∈ s) ∧ 
  s.card = 7 :=
by sorry

end num_distinct_values_for_sum_l604_60493


namespace max_num_pieces_l604_60425

-- Definition of areas
def largeCake_area : ℕ := 21 * 21
def smallPiece_area : ℕ := 3 * 3

-- Problem Statement
theorem max_num_pieces : largeCake_area / smallPiece_area = 49 := by
  sorry

end max_num_pieces_l604_60425


namespace problem1_problem2_problem3_problem4_l604_60480

theorem problem1 : 25 - 9 + (-12) - (-7) = 4 := by
  sorry

theorem problem2 : (1 / 9) * (-2)^3 / ((2 / 3)^2) = -2 := by
  sorry

theorem problem3 : ((5 / 12) + (2 / 3) - (3 / 4)) * (-12) = -4 := by
  sorry

theorem problem4 : -(1^4) + (-2) / (-1/3) - |(-9)| = -4 := by
  sorry

end problem1_problem2_problem3_problem4_l604_60480


namespace total_mass_of_individuals_l604_60453

def boat_length : Float := 3.0
def boat_breadth : Float := 2.0
def initial_sink_depth : Float := 0.018
def density_of_water : Float := 1000.0
def mass_of_second_person : Float := 75.0

theorem total_mass_of_individuals :
  let V1 := boat_length * boat_breadth * initial_sink_depth
  let m1 := V1 * density_of_water
  let total_mass := m1 + mass_of_second_person
  total_mass = 183 :=
by
  sorry

end total_mass_of_individuals_l604_60453


namespace weight_of_mixture_l604_60450

variable (A B : ℝ)
variable (ratio_A_B : A / B = 9 / 11)
variable (consumed_A : A = 26.1)

theorem weight_of_mixture (A B : ℝ) (ratio_A_B : A / B = 9 / 11) (consumed_A : A = 26.1) : 
  A + B = 58 :=
sorry

end weight_of_mixture_l604_60450


namespace probability_of_rolling_2_4_6_on_8_sided_die_l604_60454

theorem probability_of_rolling_2_4_6_on_8_sided_die : 
  ∀ (ω : Fin 8), 
  (1 / 8) * (ite (ω = 1 ∨ ω = 3 ∨ ω = 5) 1 0) = 3 / 8 := 
by 
  sorry

end probability_of_rolling_2_4_6_on_8_sided_die_l604_60454


namespace Eugene_buys_four_t_shirts_l604_60437

noncomputable def t_shirt_price : ℝ := 20
noncomputable def pants_price : ℝ := 80
noncomputable def shoes_price : ℝ := 150
noncomputable def discount : ℝ := 0.10

noncomputable def discounted_t_shirt_price : ℝ := t_shirt_price - (t_shirt_price * discount)
noncomputable def discounted_pants_price : ℝ := pants_price - (pants_price * discount)
noncomputable def discounted_shoes_price : ℝ := shoes_price - (shoes_price * discount)

noncomputable def num_pants : ℝ := 3
noncomputable def num_shoes : ℝ := 2
noncomputable def total_paid : ℝ := 558

noncomputable def total_cost_of_pants_and_shoes : ℝ := (num_pants * discounted_pants_price) + (num_shoes * discounted_shoes_price)
noncomputable def remaining_cost_for_t_shirts : ℝ := total_paid - total_cost_of_pants_and_shoes

noncomputable def num_t_shirts : ℝ := remaining_cost_for_t_shirts / discounted_t_shirt_price

theorem Eugene_buys_four_t_shirts : num_t_shirts = 4 := by
  sorry

end Eugene_buys_four_t_shirts_l604_60437


namespace largest_number_le_1_1_from_set_l604_60409

def is_largest_le (n : ℚ) (l : List ℚ) (bound : ℚ) : Prop :=
  (n ∈ l ∧ n ≤ bound) ∧ ∀ m ∈ l, m ≤ bound → m ≤ n

theorem largest_number_le_1_1_from_set : 
  is_largest_le (9/10) [14/10, 9/10, 12/10, 5/10, 13/10] (11/10) :=
by 
  sorry

end largest_number_le_1_1_from_set_l604_60409


namespace final_selling_price_l604_60404

def actual_price : ℝ := 9356.725146198829
def price_after_first_discount (P : ℝ) : ℝ := P * 0.80
def price_after_second_discount (P1 : ℝ) : ℝ := P1 * 0.90
def price_after_third_discount (P2 : ℝ) : ℝ := P2 * 0.95

theorem final_selling_price :
  (price_after_third_discount (price_after_second_discount (price_after_first_discount actual_price))) = 6400 :=
by 
  -- Here we would need to provide the proof, but it is skipped with sorry
  sorry

end final_selling_price_l604_60404


namespace triangle_tan_A_and_area_l604_60470

theorem triangle_tan_A_and_area {A B C a b c : ℝ} (hB : B = Real.pi / 3)
  (h1 : (Real.cos A - 3 * Real.cos C) * b = (3 * c - a) * Real.cos B)
  (hb : b = Real.sqrt 14) : 
  ∃ tan_A : ℝ, tan_A = Real.sqrt 3 / 5 ∧  -- First part: the value of tan A
  ∃ S : ℝ, S = (3 * Real.sqrt 3) / 2 :=  -- Second part: the area of triangle ABC
by
  sorry

end triangle_tan_A_and_area_l604_60470


namespace transform_to_100_l604_60475

theorem transform_to_100 (a b c : ℤ) (h : Int.gcd (Int.gcd a b) c = 1) :
  ∃ f : (ℤ × ℤ × ℤ → ℤ × ℤ × ℤ), (∀ p : ℤ × ℤ × ℤ,
    ∃ q : ℕ, q ≤ 5 ∧ f^[q] p = (1, 0, 0)) :=
sorry

end transform_to_100_l604_60475


namespace right_triangle_inradius_l604_60463

theorem right_triangle_inradius (a b c : ℕ) (h : a = 6) (h2 : b = 8) (h3 : c = 10) :
  ((a^2 + b^2 = c^2) ∧ (1/2 * ↑a * ↑b = 24) ∧ ((a + b + c) / 2 = 12) ∧ (24 = 12 * 2)) :=
by 
  sorry

end right_triangle_inradius_l604_60463


namespace solution_set_x_plus_3_f_x_plus_4_l604_60416

variable {f : ℝ → ℝ}
variable {f' : ℝ → ℝ}

-- Given conditions
axiom even_f_x_plus_1 : ∀ x : ℝ, f (x + 1) = f (-x + 1)
axiom deriv_negative_f : ∀ x : ℝ, x > 1 → f' x < 0
axiom f_at_4_equals_zero : f 4 = 0

-- To prove
theorem solution_set_x_plus_3_f_x_plus_4 :
  {x : ℝ | (x + 3) * f (x + 4) < 0} = {x : ℝ | -6 < x ∧ x < -3} ∪ {x : ℝ | x > 0} := sorry

end solution_set_x_plus_3_f_x_plus_4_l604_60416


namespace distance_after_second_sign_l604_60458

-- Define the known conditions
def total_distance_ridden : ℕ := 1000
def distance_to_first_sign : ℕ := 350
def distance_between_signs : ℕ := 375

-- The distance Matt rode after passing the second sign
theorem distance_after_second_sign :
  total_distance_ridden - (distance_to_first_sign + distance_between_signs) = 275 := by
  sorry

end distance_after_second_sign_l604_60458


namespace calc1_l604_60484

theorem calc1 : (2 - Real.sqrt 3) ^ 0 - Real.sqrt 12 + Real.tan (Real.pi / 3) = 1 - Real.sqrt 3 :=
by
  sorry

end calc1_l604_60484


namespace circle_radius_l604_60492

theorem circle_radius (A : ℝ) (r : ℝ) (hA : A = 121 * Real.pi) (hArea : A = Real.pi * r^2) : r = 11 :=
by
  sorry

end circle_radius_l604_60492


namespace arithmetic_mean_multiplied_correct_l604_60477

-- Define the fractions involved
def frac1 : ℚ := 3 / 4
def frac2 : ℚ := 5 / 8

-- Define the arithmetic mean and the final multiplication result
def mean_and_multiply_result : ℚ := ( (frac1 + frac2) / 2 ) * 3

-- Statement to prove that the calculated result is equal to 33/16
theorem arithmetic_mean_multiplied_correct : mean_and_multiply_result = 33 / 16 := 
by 
  -- Skipping the proof with sorry for the statement only requirement
  sorry

end arithmetic_mean_multiplied_correct_l604_60477


namespace solution_l604_60414

def solve_for_x (x : ℝ) : Prop :=
  7 + 3.5 * x = 2.1 * x - 25

theorem solution (x : ℝ) (h : solve_for_x x) : x = -22.857 :=
by
  sorry

end solution_l604_60414


namespace problem_1_problem_2_l604_60491

def f (x a : ℝ) := |x + a| + |x + 3|
def g (x : ℝ) := |x - 1| + 2

theorem problem_1 : ∀ x : ℝ, |g x| < 3 ↔ 0 < x ∧ x < 2 := 
by
  sorry

theorem problem_2 : (∀ x1 : ℝ, ∃ x2 : ℝ, f x1 a = g x2) ↔ a ≥ 5 ∨ a ≤ 1 := 
by
  sorry

end problem_1_problem_2_l604_60491


namespace gain_percent_l604_60487

variable (C S : ℝ)

theorem gain_percent 
  (h : 81 * C = 45 * S) : ((4 / 5) * 100) = 80 := 
by 
  sorry

end gain_percent_l604_60487


namespace ellipse_semi_focal_distance_range_l604_60401

theorem ellipse_semi_focal_distance_range (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : b < a) (h_ellipse : a^2 = b^2 + c^2) :
  1 < (b + c) / a ∧ (b + c) / a ≤ Real.sqrt 2 := 
sorry

end ellipse_semi_focal_distance_range_l604_60401


namespace sweet_treats_distribution_l604_60447

-- Define the number of cookies, cupcakes, brownies, and students
def cookies : ℕ := 20
def cupcakes : ℕ := 25
def brownies : ℕ := 35
def students : ℕ := 20

-- Define the total number of sweet treats
def total_sweet_treats : ℕ := cookies + cupcakes + brownies

-- Define the number of sweet treats each student will receive
def sweet_treats_per_student : ℕ := total_sweet_treats / students

-- Prove that each student will receive 4 sweet treats
theorem sweet_treats_distribution : sweet_treats_per_student = 4 := 
by sorry

end sweet_treats_distribution_l604_60447


namespace sqrt_x_minus_2_range_l604_60446

theorem sqrt_x_minus_2_range (x : ℝ) : (↑0 ≤ (x - 2)) ↔ (x ≥ 2) := sorry

end sqrt_x_minus_2_range_l604_60446


namespace path_bound_l604_60482

/-- Definition of P_k: the number of non-intersecting paths of length k starting from point O on a grid 
    where each cell has side length 1. -/
def P_k (k : ℕ) : ℕ := sorry  -- This would normally be defined through some combinatorial method

/-- The main theorem stating the required proof statement. -/
theorem path_bound (k : ℕ) : (P_k k : ℝ) / (3^k : ℝ) < 2 := sorry

end path_bound_l604_60482


namespace cupcakes_for_children_l604_60499

-- Definitions for the conditions
def packs15 : Nat := 4
def packs10 : Nat := 4
def cupcakes_per_pack15 : Nat := 15
def cupcakes_per_pack10 : Nat := 10

-- Proposition to prove the total number of cupcakes is 100
theorem cupcakes_for_children :
  (packs15 * cupcakes_per_pack15) + (packs10 * cupcakes_per_pack10) = 100 := by
  sorry

end cupcakes_for_children_l604_60499


namespace max_area_of_triangle_ABC_l604_60432

-- Definitions for the problem conditions
def A : ℝ × ℝ := (-2, 0)
def B : ℝ × ℝ := (5, 4)
def parabola (x : ℝ) : ℝ := x^2 - 3 * x
def C (r : ℝ) : ℝ × ℝ := (r, parabola r)

-- Function to compute the Shoelace Theorem area of ABC
def shoelace_area (A B C : ℝ × ℝ) : ℝ :=
  0.5 * abs ((A.1 * B.2 + B.1 * C.2 + C.1 * A.2) - (A.2 * B.1 + B.2 * C.1 + C.2 * A.1))

-- Proof statement
theorem max_area_of_triangle_ABC : ∃ (r : ℝ), -2 ≤ r ∧ r ≤ 5 ∧ shoelace_area A B (C r) = 39 := 
  sorry

end max_area_of_triangle_ABC_l604_60432


namespace relatively_prime_subsequence_exists_l604_60478

theorem relatively_prime_subsequence_exists :
  ∃ (s : ℕ → ℕ), (∀ i j : ℕ, i ≠ j → Nat.gcd (2^(s i) - 3) (2^(s j) - 3) = 1) :=
by
  sorry

end relatively_prime_subsequence_exists_l604_60478
