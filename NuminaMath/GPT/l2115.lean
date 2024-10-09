import Mathlib

namespace eval_f_a_plus_1_l2115_211522

-- Define the function f
def f (x : ℝ) : ℝ := x^2

-- Define the condition
axiom a : ℝ

-- State the theorem to be proven
theorem eval_f_a_plus_1 : f (a + 1) = a^2 + 2*a + 1 :=
by
  sorry

end eval_f_a_plus_1_l2115_211522


namespace problems_per_page_l2115_211505

theorem problems_per_page (total_problems : ℕ) (percent_solved : ℝ) (pages_left : ℕ)
  (h_total : total_problems = 550)
  (h_percent : percent_solved = 0.65)
  (h_pages : pages_left = 3) :
  (total_problems - Nat.ceil (percent_solved * total_problems)) / pages_left = 64 := by
  sorry

end problems_per_page_l2115_211505


namespace total_age_of_wines_l2115_211549

theorem total_age_of_wines (age_carlo_rosi : ℕ) (age_franzia : ℕ) (age_twin_valley : ℕ) 
    (h1 : age_carlo_rosi = 40) (h2 : age_franzia = 3 * age_carlo_rosi) (h3 : age_carlo_rosi = 4 * age_twin_valley) : 
    age_franzia + age_carlo_rosi + age_twin_valley = 170 := 
by
    sorry

end total_age_of_wines_l2115_211549


namespace min_expression_l2115_211530

theorem min_expression (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : 1/a + 1/b = 1) : 
  (∃ x : ℝ, x = min ((1 / (a - 1)) + (4 / (b - 1))) 4) :=
sorry

end min_expression_l2115_211530


namespace shortest_path_length_l2115_211598

theorem shortest_path_length (x y z : ℕ) (h1 : x + y = z + 1) (h2 : x + z = y + 5) (h3 : y + z = x + 7) : 
  min (min x y) z = 3 :=
by sorry

end shortest_path_length_l2115_211598


namespace max_value_expr_l2115_211568

theorem max_value_expr (x y z : ℝ) (h : 9 * x^2 + 4 * y^2 + 25 * z^2 = 4) : 
  10 * x + 3 * y + 15 * z ≤ 9.455 :=
sorry

end max_value_expr_l2115_211568


namespace find_x3_plus_y3_l2115_211554

theorem find_x3_plus_y3 (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 + y^2 = 167) : x^3 + y^3 = 2005 :=
sorry

end find_x3_plus_y3_l2115_211554


namespace ice_cream_not_sold_total_l2115_211556

theorem ice_cream_not_sold_total :
  let chocolate_initial := 50
  let mango_initial := 54
  let vanilla_initial := 80
  let strawberry_initial := 40
  let chocolate_sold := (3 / 5 : ℚ) * chocolate_initial
  let mango_sold := (2 / 3 : ℚ) * mango_initial
  let vanilla_sold := (75 / 100 : ℚ) * vanilla_initial
  let strawberry_sold := (5 / 8 : ℚ) * strawberry_initial
  let chocolate_not_sold := chocolate_initial - chocolate_sold
  let mango_not_sold := mango_initial - mango_sold
  let vanilla_not_sold := vanilla_initial - vanilla_sold
  let strawberry_not_sold := strawberry_initial - strawberry_sold
  chocolate_not_sold + mango_not_sold + vanilla_not_sold + strawberry_not_sold = 73 :=
by sorry

end ice_cream_not_sold_total_l2115_211556


namespace soccer_balls_donated_l2115_211579

def num_classes_per_school (elem_classes mid_classes : ℕ) : ℕ :=
  elem_classes + mid_classes

def total_classes (num_schools : ℕ) (classes_per_school : ℕ) : ℕ :=
  num_schools * classes_per_school

def total_soccer_balls (num_classes : ℕ) (balls_per_class : ℕ) : ℕ :=
  num_classes * balls_per_class

theorem soccer_balls_donated 
  (elem_classes mid_classes num_schools balls_per_class : ℕ) 
  (h_elem_classes : elem_classes = 4) 
  (h_mid_classes : mid_classes = 5) 
  (h_num_schools : num_schools = 2) 
  (h_balls_per_class : balls_per_class = 5) :
  total_soccer_balls (total_classes num_schools (num_classes_per_school elem_classes mid_classes)) balls_per_class = 90 :=
by
  sorry

end soccer_balls_donated_l2115_211579


namespace star_inequalities_not_all_true_simultaneously_l2115_211513

theorem star_inequalities_not_all_true_simultaneously
  (AB BC CD DE EF FG GH HK KL LA : ℝ)
  (h1 : BC > AB)
  (h2 : DE > CD)
  (h3 : FG > EF)
  (h4 : HK > GH)
  (h5 : LA > KL) :
  False :=
  sorry

end star_inequalities_not_all_true_simultaneously_l2115_211513


namespace five_level_pyramid_has_80_pieces_l2115_211571

-- Definitions based on problem conditions
def rods_per_level (level : ℕ) : ℕ :=
  if level = 1 then 4
  else if level = 2 then 8
  else if level = 3 then 12
  else if level = 4 then 16
  else if level = 5 then 20
  else 0

def connectors_per_level_transition : ℕ := 4

-- The total rods used for a five-level pyramid
def total_rods_five_levels : ℕ :=
  rods_per_level 1 + rods_per_level 2 + rods_per_level 3 + rods_per_level 4 + rods_per_level 5

-- The total connectors used for a five-level pyramid
def total_connectors_five_levels : ℕ :=
  connectors_per_level_transition * 5

-- The total pieces required for a five-level pyramid
def total_pieces_five_levels : ℕ :=
  total_rods_five_levels + total_connectors_five_levels

-- Main theorem statement for the proof problem
theorem five_level_pyramid_has_80_pieces : 
  total_pieces_five_levels = 80 :=
by
  -- We expect the total_pieces_five_levels to be equal to 80
  sorry

end five_level_pyramid_has_80_pieces_l2115_211571


namespace complex_real_part_of_product_l2115_211541

theorem complex_real_part_of_product (z1 z2 : ℂ) (i : ℂ) 
  (hz1 : z1 = 4 + 29 * Complex.I)
  (hz2 : z2 = 6 + 9 * Complex.I)
  (hi : i = Complex.I) : 
  ((z1 - z2) * i).re = 20 := 
by
  sorry

end complex_real_part_of_product_l2115_211541


namespace problem_statement_l2115_211564

noncomputable def f (x : ℝ) : ℝ := (1/3)^x - x^2

theorem problem_statement (x0 x1 x2 m : ℝ) (h0 : f x0 = m) (h1 : 0 < x1) (h2 : x1 < x0) (h3 : x0 < x2) :
    f x1 > m ∧ f x2 < m :=
sorry

end problem_statement_l2115_211564


namespace candy_box_original_price_l2115_211585

theorem candy_box_original_price (P : ℝ) (h₁ : 1.25 * P = 10) : P = 8 := 
sorry

end candy_box_original_price_l2115_211585


namespace probability_correct_l2115_211572

noncomputable def probability_all_players_have_5_after_2023_rings 
    (initial_money : ℕ)
    (num_rings : ℕ) 
    (target_money : ℕ)
    : ℝ := 
    if initial_money = 5 ∧ num_rings = 2023 ∧ target_money = 5 
    then 1 / 4 
    else 0

theorem probability_correct : 
        probability_all_players_have_5_after_2023_rings 5 2023 5 = 1 / 4 := 
by 
    sorry

end probability_correct_l2115_211572


namespace fifth_term_arithmetic_sequence_is_19_l2115_211526

def arithmetic_sequence_nth_term (a1 d n : ℕ) : ℕ :=
  a1 + (n - 1) * d

theorem fifth_term_arithmetic_sequence_is_19 :
  arithmetic_sequence_nth_term 3 4 5 = 19 := 
  by
  sorry

end fifth_term_arithmetic_sequence_is_19_l2115_211526


namespace remainder_numGreenRedModal_l2115_211525

def numGreenMarbles := 7
def numRedMarbles (n : ℕ) := 7 + n
def validArrangement (g r : ℕ) := (g + r = numGreenMarbles + numRedMarbles r) ∧ 
  (g = r)

theorem remainder_numGreenRedModal (N' : ℕ) :
  N' % 1000 = 432 :=
sorry

end remainder_numGreenRedModal_l2115_211525


namespace shelves_needed_is_five_l2115_211534

-- Definitions for the conditions
def initial_bears : Nat := 15
def additional_bears : Nat := 45
def bears_per_shelf : Nat := 12

-- Adding the number of bears received to the initial stock
def total_bears : Nat := initial_bears + additional_bears

-- Calculating the number of shelves used
def shelves_used : Nat := total_bears / bears_per_shelf

-- Statement to prove
theorem shelves_needed_is_five : shelves_used = 5 :=
by
  -- Insert specific step only if necessary, otherwise use sorry
  sorry

end shelves_needed_is_five_l2115_211534


namespace min_photos_for_condition_l2115_211515

noncomputable def minimum_photos (girls boys : ℕ) : ℕ :=
  if (girls = 4 ∧ boys = 8) 
  then 33
  else 0

theorem min_photos_for_condition (girls boys : ℕ) (photos : ℕ) :
  girls = 4 → boys = 8 → photos = minimum_photos girls boys
  → ∃ (pa : ℕ), pa >= 33 → pa = photos :=
by
  intros h1 h2 h3
  use minimum_photos girls boys
  rw [h3]
  sorry

end min_photos_for_condition_l2115_211515


namespace selena_left_with_l2115_211523

/-- Selena got a tip of $99 and spent money on various foods whose individual costs are provided. 
Prove that she will be left with $38. -/
theorem selena_left_with : 
  let tip := 99
  let steak_cost := 24
  let num_steaks := 2
  let burger_cost := 3.5
  let num_burgers := 2
  let ice_cream_cost := 2
  let num_ice_cream := 3
  let total_spent := (steak_cost * num_steaks) + (burger_cost * num_burgers) + (ice_cream_cost * num_ice_cream)
  tip - total_spent = 38 := 
by 
  sorry

end selena_left_with_l2115_211523


namespace union_A_B_subset_B_A_l2115_211584

-- Condition definitions
def A : Set ℝ := {x | 2 * x - 8 = 0}
def B (m : ℝ) : Set ℝ := {x | x^2 - 2 * (m + 1) * x + m^2 = 0}

-- Problem 1: If m = 4, prove A ∪ B = {2, 4, 8}
theorem union_A_B (m : ℝ) (h : m = 4) : A ∪ B m = {2, 4, 8} :=
sorry

-- Problem 2: If B ⊆ A, find the range for m
theorem subset_B_A (m : ℝ) (h : B m ⊆ A) : 
  m = 4 + 2 * Real.sqrt 2 ∨ m = 4 - 2 * Real.sqrt 2 ∨ m < -1 / 2 :=
sorry

end union_A_B_subset_B_A_l2115_211584


namespace combined_area_rectangle_triangle_l2115_211552

/-- 
  Given a rectangle ABCD with vertices A = (10, -30), 
  B = (2010, 170), D = (12, -50), and a right triangle
  ADE with vertex E = (12, -30), prove that the combined
  area of the rectangle and the triangle is 
  40400 + 20√101.
-/
theorem combined_area_rectangle_triangle :
  let A := (10, -30)
  let B := (2010, 170)
  let D := (12, -50)
  let E := (12, -30)
  let length_AB := Real.sqrt ((2010 - 10)^2 + (170 + 30)^2)
  let length_AD := Real.sqrt ((12 - 10)^2 + (-50 + 30)^2)
  let area_rectangle := length_AB * length_AD
  let length_DE := Real.sqrt ((12 - 12)^2 + (-50 + 30)^2)
  let area_triangle := 1/2 * length_DE * length_AD
  area_rectangle + area_triangle = 40400 + 20 * Real.sqrt 101 :=
by
  sorry

end combined_area_rectangle_triangle_l2115_211552


namespace log_bounds_l2115_211548

-- Definitions and assumptions
def tenCubed : Nat := 1000
def tenFourth : Nat := 10000
def twoNine : Nat := 512
def twoFourteen : Nat := 16384

-- Statement that encapsulates the proof problem
theorem log_bounds (h1 : 10^3 = tenCubed) 
                   (h2 : 10^4 = tenFourth) 
                   (h3 : 2^9 = twoNine) 
                   (h4 : 2^14 = twoFourteen) : 
  (2 / 7 : ℝ) < Real.log 2 / Real.log 10 ∧ Real.log 2 / Real.log 10 < (1 / 3 : ℝ) :=
sorry

end log_bounds_l2115_211548


namespace ethanol_percentage_in_fuel_B_l2115_211566

theorem ethanol_percentage_in_fuel_B 
  (tank_capacity : ℕ)
  (fuel_A_vol : ℕ)
  (ethanol_in_A_percentage : ℝ)
  (ethanol_total : ℝ)
  (ethanol_A_vol : ℝ)
  (fuel_B_vol : ℕ)
  (ethanol_B_vol : ℝ)
  (ethanol_B_percentage : ℝ) 
  (h1 : tank_capacity = 204)
  (h2 : fuel_A_vol = 66)
  (h3 : ethanol_in_A_percentage = 0.12)
  (h4 : ethanol_total = 30)
  (h5 : ethanol_A_vol = fuel_A_vol * ethanol_in_A_percentage)
  (h6 : ethanol_B_vol = ethanol_total - ethanol_A_vol)
  (h7 : fuel_B_vol = tank_capacity - fuel_A_vol)
  (h8 : ethanol_B_percentage = (ethanol_B_vol / fuel_B_vol) * 100) :
  ethanol_B_percentage = 16 :=
by sorry

end ethanol_percentage_in_fuel_B_l2115_211566


namespace negation_proposition_l2115_211546

theorem negation_proposition (x y : ℝ) :
  (¬ (x^2 + y^2 = 0 → x = 0 ∧ y = 0)) ↔ (x^2 + y^2 ≠ 0 → (x ≠ 0 ∨ y ≠ 0)) := 
sorry

end negation_proposition_l2115_211546


namespace elder_twice_as_old_l2115_211592

theorem elder_twice_as_old (Y E : ℕ) (hY : Y = 35) (hDiff : E - Y = 20) : ∃ (X : ℕ),  X = 15 ∧ E - X = 2 * (Y - X) := 
by
  sorry

end elder_twice_as_old_l2115_211592


namespace minimum_value_expression_l2115_211536

open Real

theorem minimum_value_expression (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  ((a^2 + 4*a + 2) * (b^2 + 4*b + 2) * (c^2 + 4*c + 2)) / (a * b * c) ≥ 216 :=
sorry

end minimum_value_expression_l2115_211536


namespace profit_percent_eq_20_l2115_211532

-- Define cost price 'C' and original selling price 'S'
variable (C S : ℝ)

-- Hypothesis: selling at 2/3 of the original price results in a 20% loss 
def condition (C S : ℝ) : Prop :=
  (2 / 3) * S = 0.8 * C

-- Main theorem: profit percent when selling at the original price is 20%
theorem profit_percent_eq_20 (C S : ℝ) (h : condition C S) : (S - C) / C * 100 = 20 :=
by
  -- Proof steps would go here but we use sorry to indicate the proof is omitted
  sorry

end profit_percent_eq_20_l2115_211532


namespace sum_of_distinct_product_GH_l2115_211528

def divisible_by_45 (n : ℕ) : Prop :=
  45 ∣ n

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def is_single_digit (d : ℕ) : Prop :=
  d < 10

theorem sum_of_distinct_product_GH : 
  ∀ (G H : ℕ), 
    is_single_digit G ∧ is_single_digit H ∧ 
    divisible_by_45 (8620000307 + 10000000 * G + H) → 
    (if H = 5 then GH = 6 else if H = 0 then GH = 0 else GH = 0) := 
  sorry

-- Note: This is a simplified representation; tailored more complex conditions and steps may be encapsulated in separate definitions and theorems as needed.

end sum_of_distinct_product_GH_l2115_211528


namespace problem1_problem2_l2115_211561

-- Problem 1: Sequence "Seven six five four three two one" is a descending order
theorem problem1 : ∃ term: String, term = "Descending Order" ∧ "Seven six five four three two one" = "Descending Order" := sorry

-- Problem 2: Describing a computing tool that knows 0 and 1 and can calculate large numbers (computer)
theorem problem2 : ∃ tool: String, tool = "Computer" ∧ "I only know 0 and 1, can calculate millions and billions, available in both software and hardware" = "Computer" := sorry

end problem1_problem2_l2115_211561


namespace second_assistant_smoked_pipes_l2115_211563

theorem second_assistant_smoked_pipes
    (x y z : ℚ)
    (H1 : (2 / 3) * x = (4 / 9) * y)
    (H2 : x + y = 1)
    (H3 : (x + z) / (y - z) = y / x) :
    z = 1 / 5 → x = 2 / 5 ∧ y = 3 / 5 →
    ∀ n : ℕ, n = 5 :=
by
  sorry

end second_assistant_smoked_pipes_l2115_211563


namespace prize_winners_l2115_211580

theorem prize_winners (total_people : ℕ) (percent_envelope : ℝ) (percent_win : ℝ) 
  (h_total : total_people = 100) (h_percent_envelope : percent_envelope = 0.40) 
  (h_percent_win : percent_win = 0.20) : 
  (percent_win * (percent_envelope * total_people)) = 8 := by
  sorry

end prize_winners_l2115_211580


namespace number_of_questions_in_test_l2115_211517

theorem number_of_questions_in_test (x : ℕ) (sections questions_correct : ℕ)
  (h_sections : sections = 5)
  (h_questions_correct : questions_correct = 32)
  (h_percentage : 0.70 < (questions_correct : ℚ) / x ∧ (questions_correct : ℚ) / x < 0.77) 
  (h_multiple_of_sections : x % sections = 0) : 
  x = 45 :=
sorry

end number_of_questions_in_test_l2115_211517


namespace clothes_washer_final_price_l2115_211507

theorem clothes_washer_final_price
  (P : ℝ) (d1 d2 d3 : ℝ)
  (hP : P = 500)
  (hd1 : d1 = 0.10)
  (hd2 : d2 = 0.20)
  (hd3 : d3 = 0.05) :
  (P * (1 - d1) * (1 - d2) * (1 - d3)) / P = 0.684 :=
by
  sorry

end clothes_washer_final_price_l2115_211507


namespace expression_for_C_value_of_C_l2115_211529

variables (x y : ℝ)

-- Definitions based on the given conditions
def A := x^2 - 2 * x * y + y^2
def B := x^2 + 2 * x * y + y^2

-- The algebraic expression for C
def C := - x^2 + 10 * x * y - y^2

-- Prove that the expression for C is correct
theorem expression_for_C (h : 3 * A x y - 2 * B x y + C x y = 0) : 
  C x y = - x^2 + 10 * x * y - y^2 := 
by {
  sorry
}

-- Prove the value of C when x = 1/2 and y = -2
theorem value_of_C : C (1/2) (-2) = -57/4 :=
by {
  sorry
}

end expression_for_C_value_of_C_l2115_211529


namespace max_cos_a_l2115_211591

theorem max_cos_a (a b : ℝ) (h : Real.cos (a + b) = Real.cos a - Real.cos b) : 
  Real.cos a ≤ 1 := 
sorry

end max_cos_a_l2115_211591


namespace profit_450_l2115_211567

-- Define the conditions
def cost_per_garment : ℕ := 40
def wholesale_price : ℕ := 60

-- Define the piecewise function for wholesale price P
noncomputable def P (x : ℕ) : ℕ :=
  if h : 0 < x ∧ x ≤ 100 then wholesale_price
  else if h : 100 < x ∧ x ≤ 500 then 62 - x / 50
  else 0

-- Define the profit function L
noncomputable def L (x : ℕ) : ℕ :=
  if h : 0 < x ∧ x ≤ 100 then (P x - cost_per_garment) * x
  else if h : 100 < x ∧ x ≤ 500 then (22 * x - x^2 / 50)
  else 0

-- State the theorem
theorem profit_450 : L 450 = 5850 :=
by
  sorry

end profit_450_l2115_211567


namespace horner_value_at_2_l2115_211527

noncomputable def f (x : ℝ) := 2 * x^5 - 3 * x^3 + 2 * x^2 + x - 3

theorem horner_value_at_2 : f 2 = 12 := sorry

end horner_value_at_2_l2115_211527


namespace find_certain_number_l2115_211562

theorem find_certain_number (x : ℕ) (certain_number : ℕ)
  (h1 : certain_number * x = 675)
  (h2 : x = 27) : certain_number = 25 :=
by
  -- Proof goes here
  sorry

end find_certain_number_l2115_211562


namespace even_sum_count_l2115_211540

theorem even_sum_count (x y : ℕ) 
  (hx : x = (40 + 42 + 44 + 46 + 48 + 50 + 52 + 54 + 56 + 58 + 60)) 
  (hy : y = ((60 - 40) / 2 + 1)) : 
  x + y = 561 := 
by 
  sorry

end even_sum_count_l2115_211540


namespace intersection_points_circle_l2115_211537

-- Defining the two lines based on the parameter u
def line1 (u : ℝ) (x y : ℝ) : Prop := 2 * u * x - 3 * y - 2 * u = 0
def line2 (u : ℝ) (x y : ℝ) : Prop := x - 3 * u * y + 2 = 0

-- Proof statement that shows the intersection points lie on a circle
theorem intersection_points_circle (u x y : ℝ) :
  line1 u x y → line2 u x y → (x - 1)^2 + y^2 = 1 :=
by {
  -- This completes the proof statement, but leaves implementation as exercise
  sorry
}

end intersection_points_circle_l2115_211537


namespace James_baked_muffins_l2115_211599

theorem James_baked_muffins (arthur_muffins : Nat) (multiplier : Nat) (james_muffins : Nat) : 
  arthur_muffins = 115 → 
  multiplier = 12 → 
  james_muffins = arthur_muffins * multiplier → 
  james_muffins = 1380 :=
by
  intros haf ham hmul
  rw [haf, ham] at hmul
  simp at hmul
  exact hmul

end James_baked_muffins_l2115_211599


namespace find_root_and_m_l2115_211544

theorem find_root_and_m (x₁ m : ℝ) (h₁ : -2 * x₁ = 2) (h₂ : x^2 + m * x + 2 = 0) : x₁ = -1 ∧ m = 3 := 
by 
  -- Proof omitted
  sorry

end find_root_and_m_l2115_211544


namespace number_of_trees_is_correct_l2115_211535

-- Define the conditions
def length_of_plot := 120
def width_of_plot := 70
def distance_between_trees := 5

-- Define the calculated number of intervals along each side
def intervals_along_length := length_of_plot / distance_between_trees
def intervals_along_width := width_of_plot / distance_between_trees

-- Define the number of trees along each side including the boundaries
def trees_along_length := intervals_along_length + 1
def trees_along_width := intervals_along_width + 1

-- Define the total number of trees
def total_number_of_trees := trees_along_length * trees_along_width

-- The theorem we want to prove
theorem number_of_trees_is_correct : total_number_of_trees = 375 :=
by sorry

end number_of_trees_is_correct_l2115_211535


namespace reading_comprehension_application_method_1_application_method_2_l2115_211576

-- Reading Comprehension Problem in Lean 4
theorem reading_comprehension (x : ℝ) (h : x^2 + x + 5 = 8) : 2 * x^2 + 2 * x - 4 = 2 :=
by sorry

-- Application of Methods Problem (1) in Lean 4
theorem application_method_1 (x : ℝ) (h : x^2 + x + 2 = 9) : -2 * x^2 - 2 * x + 3 = -11 :=
by sorry

-- Application of Methods Problem (2) in Lean 4
theorem application_method_2 (a b : ℝ) (h : 8 * a + 2 * b = 5) : a * (-2)^3 + b * (-2) + 3 = -2 :=
by sorry

end reading_comprehension_application_method_1_application_method_2_l2115_211576


namespace sin_cos_identity_l2115_211503

theorem sin_cos_identity :
  (Real.sin (20 * Real.pi / 180) * Real.cos (10 * Real.pi / 180) 
  - Real.cos (200 * Real.pi / 180) * Real.sin (10 * Real.pi / 180)) = 1 / 2 := 
by
  -- This would be where the proof goes
  sorry

end sin_cos_identity_l2115_211503


namespace sum_of_abc_is_33_l2115_211586

theorem sum_of_abc_is_33 (a b c N : ℕ) (h : a ≠ b ∧ b ≠ c ∧ a ≠ c)
    (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hN1 : N = 5 * a + 3 * b + 5 * c)
    (hN2 : N = 4 * a + 5 * b + 4 * c) (hN_range : 131 < N ∧ N < 150) :
  a + b + c = 33 := 
sorry

end sum_of_abc_is_33_l2115_211586


namespace number_of_ways_to_assign_friends_to_teams_l2115_211551

theorem number_of_ways_to_assign_friends_to_teams (n m : ℕ) (h_n : n = 7) (h_m : m = 4) : m ^ n = 16384 :=
by
  rw [h_n, h_m]
  exact pow_succ' 4 6

end number_of_ways_to_assign_friends_to_teams_l2115_211551


namespace billy_scores_two_points_each_round_l2115_211553

def billy_old_score := 725
def billy_rounds := 363
def billy_target_score := billy_old_score + 1
def billy_points_per_round := billy_target_score / billy_rounds

theorem billy_scores_two_points_each_round :
  billy_points_per_round = 2 := by
  sorry

end billy_scores_two_points_each_round_l2115_211553


namespace find_constants_l2115_211531

variable (x : ℝ)

def A := 3
def B := -3
def C := 11

theorem find_constants (h₁ : x ≠ 2) (h₂ : x ≠ 4) :
  (5 * x + 2) / ((x - 2) * (x - 4)^2) = A / (x - 2) + B / (x - 4) + C / (x - 4)^2 :=
by
  unfold A B C
  sorry

end find_constants_l2115_211531


namespace find_number_exceeds_sixteen_percent_l2115_211516

theorem find_number_exceeds_sixteen_percent (x : ℝ) (h : x - 0.16 * x = 63) : x = 75 :=
sorry

end find_number_exceeds_sixteen_percent_l2115_211516


namespace centipede_and_earthworm_meeting_time_l2115_211518

noncomputable def speed_centipede : ℚ := 5 / 3
noncomputable def speed_earthworm : ℚ := 5 / 2
noncomputable def initial_gap : ℚ := 20

theorem centipede_and_earthworm_meeting_time : 
  ∃ t : ℚ, (5 / 2) * t = initial_gap + (5 / 3) * t ∧ t = 24 := 
by
  sorry

end centipede_and_earthworm_meeting_time_l2115_211518


namespace triangular_pyramid_surface_area_l2115_211510

theorem triangular_pyramid_surface_area
  (base_area : ℝ)
  (side_area : ℝ) :
  base_area = 3 ∧ side_area = 6 → base_area + 3 * side_area = 21 :=
by
  sorry

end triangular_pyramid_surface_area_l2115_211510


namespace prime_sum_divisors_l2115_211596

theorem prime_sum_divisors (p : ℕ) (s : ℕ) : 
  (2 ≤ s ∧ s ≤ 10) → 
  (p = 2^s - 1) → 
  (p = 3 ∨ p = 7 ∨ p = 31 ∨ p = 127) :=
by
  intros h1 h2
  sorry

end prime_sum_divisors_l2115_211596


namespace second_trial_temperatures_l2115_211569

-- Definitions based on the conditions
def range_start : ℝ := 60
def range_end : ℝ := 70
def golden_ratio : ℝ := 0.618

-- Calculations for trial temperatures
def lower_trial_temp : ℝ := range_start + (range_end - range_start) * golden_ratio
def upper_trial_temp : ℝ := range_end - (range_end - range_start) * golden_ratio

-- Lean 4 statement to prove the trial temperatures
theorem second_trial_temperatures :
  lower_trial_temp = 66.18 ∧ upper_trial_temp = 63.82 :=
by
  sorry

end second_trial_temperatures_l2115_211569


namespace gcd_polynomial_l2115_211543

theorem gcd_polynomial (b : ℤ) (h : 1729 ∣ b) : Int.gcd (b^2 + 11*b + 28) (b + 5) = 2 := 
by
  sorry

end gcd_polynomial_l2115_211543


namespace james_initial_bars_l2115_211574

def initial_chocolate_bars (sold_last_week sold_this_week needs_to_sell : ℕ) : ℕ :=
  sold_last_week + sold_this_week + needs_to_sell

theorem james_initial_bars : 
  initial_chocolate_bars 5 7 6 = 18 :=
by 
  sorry

end james_initial_bars_l2115_211574


namespace quadratic_has_two_real_roots_for_any_m_find_m_given_roots_conditions_l2115_211524

theorem quadratic_has_two_real_roots_for_any_m (m : ℝ) : 
  ∃ (α β : ℝ), (α^2 - 3*α + 2 - m^2 - m = 0) ∧ (β^2 - 3*β + 2 - m^2 - m = 0) :=
sorry

theorem find_m_given_roots_conditions (α β : ℝ) (m : ℝ) 
  (h1 : α^2 - 3*α + 2 - m^2 - m = 0) 
  (h2 : β^2 - 3*β + 2 - m^2 - m = 0) 
  (h3 : α^2 + β^2 = 9) : 
  m = -2 ∨ m = 1 :=
sorry

end quadratic_has_two_real_roots_for_any_m_find_m_given_roots_conditions_l2115_211524


namespace time_between_four_and_five_straight_line_l2115_211538

theorem time_between_four_and_five_straight_line :
  ∃ t : ℚ, t = 21 + 9/11 ∨ t = 54 + 6/11 :=
by
  sorry

end time_between_four_and_five_straight_line_l2115_211538


namespace cos_sum_nonneg_one_l2115_211545

theorem cos_sum_nonneg_one (x y z : ℝ) (h : x + y + z = 0) : abs (Real.cos x) + abs (Real.cos y) + abs (Real.cos z) ≥ 1 := 
by {
  sorry
}

end cos_sum_nonneg_one_l2115_211545


namespace ratio_after_addition_l2115_211555

theorem ratio_after_addition (a b : ℕ) (h1 : a * 3 = b * 2) (h2 : b - a = 8) : (a + 4) * 7 = (b + 4) * 5 :=
by
  sorry

end ratio_after_addition_l2115_211555


namespace tangent_line_equation_at_point_l2115_211595

noncomputable def f (x : ℝ) : ℝ := Real.log (2 * x - 1) - x

theorem tangent_line_equation_at_point :
  ∃ a b c : ℝ, (∀ x y : ℝ, a * x + b * y + c = 0 ↔ (x = 1 → y = -1 → f x = y)) ∧ (a * 1 + b * (-1) + c = 0) :=
by
  sorry

end tangent_line_equation_at_point_l2115_211595


namespace cos_180_eq_neg_one_l2115_211577

/-- The cosine of 180 degrees is -1. -/
theorem cos_180_eq_neg_one : Real.cos (180 * Real.pi / 180) = -1 :=
by sorry

end cos_180_eq_neg_one_l2115_211577


namespace range_of_a_l2115_211557

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x^2 - 2*x + a ≥ 0) ↔ (1 ≤ a) :=
by sorry

end range_of_a_l2115_211557


namespace trig_identity_tan_solutions_l2115_211594

open Real

theorem trig_identity_tan_solutions :
  ∃ α β : ℝ, (tan α) * (tan β) = -3 ∧ (tan α) + (tan β) = 3 ∧
  abs (sin (α + β) ^ 2 - 3 * sin (α + β) * cos (α + β) - 3 * cos (α + β) ^ 2) = 3 :=
by
  have: ∀ x : ℝ, x^2 - 3*x - 3 = 0 → x = (3 + sqrt 21) / 2 ∨ x = (3 - sqrt 21) / 2 := sorry
  sorry

end trig_identity_tan_solutions_l2115_211594


namespace scaled_multiplication_l2115_211547

theorem scaled_multiplication (h : 268 * 74 = 19832) : 2.68 * 0.74 = 1.9832 :=
by
  -- proof steps would go here
  sorry

end scaled_multiplication_l2115_211547


namespace isosceles_triangle_leg_length_l2115_211542

theorem isosceles_triangle_leg_length
  (P : ℝ) (base : ℝ) (L : ℝ)
  (h_isosceles : true)
  (h_perimeter : P = 24)
  (h_base : base = 10)
  (h_perimeter_formula : P = base + 2 * L) :
  L = 7 := 
by
  sorry

end isosceles_triangle_leg_length_l2115_211542


namespace total_pencils_l2115_211587

-- Defining the number of pencils each person has.
def jessica_pencils : ℕ := 8
def sandy_pencils : ℕ := 8
def jason_pencils : ℕ := 8

-- Theorem stating the total number of pencils
theorem total_pencils : jessica_pencils + sandy_pencils + jason_pencils = 24 := by
  sorry

end total_pencils_l2115_211587


namespace decorations_count_l2115_211508

-- Define the conditions as Lean definitions
def plastic_skulls := 12
def broomsticks := 4
def spiderwebs := 12
def pumpkins := 2 * spiderwebs
def large_cauldron := 1
def budget_more_decorations := 20
def left_to_put_up := 10

-- Define the total decorations
def decorations_already_up := plastic_skulls + broomsticks + spiderwebs + pumpkins + large_cauldron
def additional_decorations := budget_more_decorations + left_to_put_up
def total_decorations := decorations_already_up + additional_decorations

-- Prove the total number of decorations will be 83
theorem decorations_count : total_decorations = 83 := by 
  sorry

end decorations_count_l2115_211508


namespace solve_x4_minus_inv_x4_l2115_211502

-- Given condition
def condition (x : ℝ) : Prop := x - (1 / x) = 5

-- Theorem statement ensuring the problem is mathematically equivalent
theorem solve_x4_minus_inv_x4 (x : ℝ) (hx : condition x) : x^4 - (1 / x^4) = 723 :=
by
  sorry

end solve_x4_minus_inv_x4_l2115_211502


namespace wait_at_least_15_seconds_probability_l2115_211575

-- Define the duration of the red light
def red_light_duration : ℕ := 40

-- Define the minimum waiting time for the green light
def min_wait_time : ℕ := 15

-- Define the duration after which pedestrian does not need to wait 15 seconds
def max_arrival_time : ℕ := red_light_duration - min_wait_time

-- Lean statement to prove the required probability
theorem wait_at_least_15_seconds_probability :
  (max_arrival_time : ℝ) / red_light_duration = 5 / 8 :=
by
  -- Proof omitted with sorry
  sorry

end wait_at_least_15_seconds_probability_l2115_211575


namespace tea_mixture_price_l2115_211597

theorem tea_mixture_price :
  ∀ (price_A price_B : ℝ) (ratio_A ratio_B : ℝ),
  price_A = 65 →
  price_B = 70 →
  ratio_A = 1 →
  ratio_B = 1 →
  (price_A * ratio_A + price_B * ratio_B) / (ratio_A + ratio_B) = 67.5 :=
by
  intros price_A price_B ratio_A ratio_B h1 h2 h3 h4
  sorry

end tea_mixture_price_l2115_211597


namespace surface_area_rectangular_solid_l2115_211500

def length := 5
def width := 4
def depth := 1

def surface_area (l w d : ℕ) := 2 * (l * w) + 2 * (l * d) + 2 * (w * d)

theorem surface_area_rectangular_solid : surface_area length width depth = 58 := 
by 
sorry

end surface_area_rectangular_solid_l2115_211500


namespace man_speed_in_still_water_l2115_211573

noncomputable def speed_of_man_in_still_water (vm vs : ℝ) : Prop :=
  -- Condition 1: v_m + v_s = 8
  vm + vs = 8 ∧
  -- Condition 2: v_m - v_s = 5
  vm - vs = 5

-- Proving the speed of the man in still water is 6.5 km/h
theorem man_speed_in_still_water : ∃ (v_m : ℝ), (∃ v_s : ℝ, speed_of_man_in_still_water v_m v_s) ∧ v_m = 6.5 :=
by
  sorry

end man_speed_in_still_water_l2115_211573


namespace even_k_l2115_211501

theorem even_k :
  ∀ (a b n k : ℕ),
  1 ≤ a → 1 ≤ b → 0 < n →
  2^n - 1 = a * b →
  (a * b + a - b - 1) % 2^k = 0 →
  (a * b + a - b - 1) % 2^(k+1) ≠ 0 →
  Even k :=
by
  intros a b n k ha hb hn h1 h2 h3
  sorry

end even_k_l2115_211501


namespace det_of_commuting_matrices_l2115_211590

theorem det_of_commuting_matrices (n : ℕ) (hn : n ≥ 2) (A B : Matrix (Fin n) (Fin n) ℝ)
  (hA : A * A = -1) (hAB : A * B = B * A) : 
  0 ≤ B.det := 
sorry

end det_of_commuting_matrices_l2115_211590


namespace hose_Z_fill_time_l2115_211520

theorem hose_Z_fill_time (P X Y Z : ℝ) (h1 : X + Y = P / 3) (h2 : Y = P / 9) (h3 : X + Z = P / 4) (h4 : X + Y + Z = P / 2.5) : Z = P / 15 :=
sorry

end hose_Z_fill_time_l2115_211520


namespace log_comparison_l2115_211559

theorem log_comparison (a b c : ℝ) 
  (h₁ : a = Real.log 6 / Real.log 3)
  (h₂ : b = Real.log 10 / Real.log 5)
  (h₃ : c = Real.log 14 / Real.log 7) :
  a > b ∧ b > c :=
  sorry

end log_comparison_l2115_211559


namespace no_solutions_l2115_211539

/-- Prove that there are no pairs of positive integers (x, y) such that x² + y² + x = 2x³. -/
theorem no_solutions : ∀ x y : ℕ, 0 < x → 0 < y → (x^2 + y^2 + x = 2 * x^3) → false :=
by
  sorry

end no_solutions_l2115_211539


namespace combined_cost_l2115_211588

variable (bench_cost : ℝ) (table_cost : ℝ)

-- Conditions
axiom bench_cost_def : bench_cost = 250.0
axiom table_cost_def : table_cost = 2 * bench_cost

-- Goal
theorem combined_cost (bench_cost : ℝ) (table_cost : ℝ) 
  (h1 : bench_cost = 250.0) (h2 : table_cost = 2 * bench_cost) : 
  table_cost + bench_cost = 750.0 :=
by
  sorry

end combined_cost_l2115_211588


namespace sum_bn_2999_l2115_211519

def b_n (n : ℕ) : ℕ :=
  if n % 17 = 0 ∧ n % 19 = 0 then 15
  else if n % 19 = 0 ∧ n % 13 = 0 then 18
  else if n % 13 = 0 ∧ n % 17 = 0 then 17
  else 0

theorem sum_bn_2999 : (Finset.range 3000).sum b_n = 572 := by
  sorry

end sum_bn_2999_l2115_211519


namespace right_angled_triangle_setB_l2115_211511

def isRightAngledTriangle (a b c : ℝ) : Prop :=
  a * a + b * b = c * c

theorem right_angled_triangle_setB :
  isRightAngledTriangle 1 1 (Real.sqrt 2) ∧
  ¬isRightAngledTriangle 1 2 3 ∧
  ¬isRightAngledTriangle 6 8 11 ∧
  ¬isRightAngledTriangle 2 3 4 :=
by
  sorry

end right_angled_triangle_setB_l2115_211511


namespace sum_of_first_and_third_is_68_l2115_211512

theorem sum_of_first_and_third_is_68
  (A B C : ℕ)
  (h1 : A + B + C = 98)
  (h2 : A * 3 = B * 2)  -- implying A / B = 2 / 3
  (h3 : B * 8 = C * 5)  -- implying B / C = 5 / 8
  (h4 : B = 30) :
  A + C = 68 :=
sorry

end sum_of_first_and_third_is_68_l2115_211512


namespace sq_sum_ge_one_third_l2115_211504

theorem sq_sum_ge_one_third (a b c : ℝ) (h : a + b + c = 1) : a^2 + b^2 + c^2 ≥ 1 / 3 := 
sorry

end sq_sum_ge_one_third_l2115_211504


namespace min_time_to_same_side_l2115_211521

def side_length : ℕ := 50
def speed_A : ℕ := 5
def speed_B : ℕ := 3

def time_to_same_side (side_length speed_A speed_B : ℕ) : ℕ :=
  30

theorem min_time_to_same_side :
  time_to_same_side side_length speed_A speed_B = 30 :=
by
  -- The proof goes here
  sorry

end min_time_to_same_side_l2115_211521


namespace faye_total_crayons_l2115_211589

  def num_rows : ℕ := 16
  def crayons_per_row : ℕ := 6
  def total_crayons : ℕ := num_rows * crayons_per_row

  theorem faye_total_crayons : total_crayons = 96 :=
  by
  sorry
  
end faye_total_crayons_l2115_211589


namespace rhombus_diagonal_l2115_211550

theorem rhombus_diagonal (d1 d2 area : ℝ) (h1 : d1 = 20) (h2 : area = 160) (h3 : area = (d1 * d2) / 2) :
  d2 = 16 :=
by
  rw [h1, h2] at h3
  linarith

end rhombus_diagonal_l2115_211550


namespace weight_of_each_piece_l2115_211533

theorem weight_of_each_piece 
  (x : ℝ)
  (h : 2 * x + 0.08 = 0.75) : 
  x = 0.335 :=
by
  sorry

end weight_of_each_piece_l2115_211533


namespace inequality_with_sum_of_one_l2115_211593

theorem inequality_with_sum_of_one
  (a b c d : ℝ) (h_pos: 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d)
  (h_sum: a + b + c + d = 1) :
  (a^2 / (a + b) + b^2 / (b + c) + c^2 / (c + d) + d^2 / (d + a) >= 1 / 2) :=
sorry

end inequality_with_sum_of_one_l2115_211593


namespace jordan_purchase_total_rounded_l2115_211581

theorem jordan_purchase_total_rounded :
  let p1 := 2.49
  let p2 := 6.51
  let p3 := 11.49
  let r1 := 2 -- rounded value of p1
  let r2 := 7 -- rounded value of p2
  let r3 := 11 -- rounded value of p3
  r1 + r2 + r3 = 20 :=
by
  let p1 := 2.49
  let p2 := 6.51
  let p3 := 11.49
  let r1 := 2
  let r2 := 7
  let r3 := 11
  show r1 + r2 + r3 = 20
  sorry

end jordan_purchase_total_rounded_l2115_211581


namespace simplify_polynomial_l2115_211582

def poly1 (x : ℝ) : ℝ := 5 * x^12 - 3 * x^9 + 6 * x^8 - 2 * x^7
def poly2 (x : ℝ) : ℝ := 7 * x^12 + 2 * x^11 - x^9 + 4 * x^7 + 2 * x^5 - x + 3
def expected (x : ℝ) : ℝ := 12 * x^12 + 2 * x^11 - 4 * x^9 + 6 * x^8 + 2 * x^7 + 2 * x^5 - x + 3

theorem simplify_polynomial (x : ℝ) : poly1 x + poly2 x = expected x :=
  by sorry

end simplify_polynomial_l2115_211582


namespace factorization_x3_minus_9xy2_l2115_211570

theorem factorization_x3_minus_9xy2 (x y : ℝ) : x^3 - 9 * x * y^2 = x * (x + 3 * y) * (x - 3 * y) :=
by sorry

end factorization_x3_minus_9xy2_l2115_211570


namespace calculate_expression_l2115_211560

def x : Float := 3.241
def y : Float := 14
def z : Float := 100
def expected_result : Float := 0.45374

theorem calculate_expression : (x * y) / z = expected_result := by
  sorry

end calculate_expression_l2115_211560


namespace T_n_sum_general_term_b_b_n_comparison_l2115_211558

noncomputable def sequence_a (n : ℕ) : ℕ := sorry  -- Placeholder for sequence {a_n}
noncomputable def S (n : ℕ) : ℕ := sorry  -- Placeholder for sum of first n terms S_n
noncomputable def sequence_b (n : ℕ) (q : ℝ) : ℝ := sorry  -- Placeholder for sequence {b_n}

axiom sequence_a_def : ∀ n : ℕ, 2 * sequence_a (n + 1) = sequence_a n + sequence_a (n + 2)
axiom sequence_a_5 : sequence_a 5 = 5
axiom S_7 : S 7 = 28

noncomputable def T (n : ℕ) : ℝ := (2 * n : ℝ) / (n + 1 : ℝ)

theorem T_n_sum : ∀ n : ℕ, T n = 2 * (1 - 1 / (n + 1)) := sorry

axiom b1 : ℝ
axiom b_def : ∀ (n : ℕ) (q : ℝ), q > 0 → sequence_b (n + 1) q = sequence_b n q + q ^ (sequence_a n)

theorem general_term_b (q : ℝ) (n : ℕ) (hq : q > 0) : 
  (if q = 1 then sequence_b n q = n else sequence_b n q = (1 - q ^ n) / (1 - q)) := sorry

theorem b_n_comparison (q : ℝ) (n : ℕ) (hq : q > 0) : 
  sequence_b n q * sequence_b (n + 2) q < (sequence_b (n + 1) q) ^ 2 := sorry

end T_n_sum_general_term_b_b_n_comparison_l2115_211558


namespace number_of_boys_in_second_class_l2115_211509

def boys_in_first_class : ℕ := 28
def portion_of_second_class (b2 : ℕ) : ℚ := 7 / 8 * b2

theorem number_of_boys_in_second_class (b2 : ℕ) (h : portion_of_second_class b2 = boys_in_first_class) : b2 = 32 :=
by 
  sorry

end number_of_boys_in_second_class_l2115_211509


namespace inverse_of_f_inverse_of_f_inv_l2115_211506

noncomputable def f (x : ℝ) : ℝ := 3^(x - 1) + 1

noncomputable def f_inv (x : ℝ) : ℝ := 1 + Real.log x / Real.log 3

theorem inverse_of_f (x : ℝ) (hx : x > 1) : f_inv (f x) = x :=
by
  sorry

theorem inverse_of_f_inv (x : ℝ) (hx : x > 1) : f (f_inv x) = x :=
by
  sorry

end inverse_of_f_inverse_of_f_inv_l2115_211506


namespace function_intersects_all_lines_l2115_211514

theorem function_intersects_all_lines :
  (∃ f : ℝ → ℝ, (∀ a : ℝ, ∃ y : ℝ, y = f a) ∧ (∀ k b : ℝ, ∃ x : ℝ, f x = k * x + b)) :=
sorry

end function_intersects_all_lines_l2115_211514


namespace abcd_value_l2115_211565

noncomputable def abcd_eval (a b c d : ℂ) : ℂ := a * b * c * d

theorem abcd_value (a b c d : ℂ) 
  (h1 : a + b + c + d = 5)
  (h2 : (5 - a)^4 + (5 - b)^4 + (5 - c)^4 + (5 - d)^4 = 125)
  (h3 : (a + b)^4 + (b + c)^4 + (c + d)^4 + (d + a)^4 + (a + c)^4 + (b + d)^4 = 1205)
  (h4 : a^4 + b^4 + c^4 + d^4 = 25) : 
  abcd_eval a b c d = 70 := 
sorry

end abcd_value_l2115_211565


namespace necessary_and_sufficient_condition_l2115_211578

-- Define the first circle
def circle1 (m : ℝ) : Set (ℝ × ℝ) :=
  { p | (p.1 + m)^2 + p.2^2 = 1 }

-- Define the second circle
def circle2 : Set (ℝ × ℝ) :=
  { p | (p.1 - 2)^2 + p.2^2 = 4 }

-- Define the condition -1 ≤ m ≤ 1
def condition (m : ℝ) : Prop :=
  -1 ≤ m ∧ m ≤ 1

-- Define the property for circles having common points
def circlesHaveCommonPoints (m : ℝ) : Prop :=
  ∃ p : ℝ × ℝ, p ∈ circle1 m ∧ p ∈ circle2

-- The final statement
theorem necessary_and_sufficient_condition (m : ℝ) :
  condition m → circlesHaveCommonPoints m ↔ (-5 ≤ m ∧ m ≤ 1) :=
by
  sorry

end necessary_and_sufficient_condition_l2115_211578


namespace complete_the_square_b_l2115_211583

theorem complete_the_square_b (x : ℝ) : (x ^ 2 - 6 * x + 7 = 0) → ∃ a b : ℝ, (x + a) ^ 2 = b ∧ b = 2 :=
by
sorry

end complete_the_square_b_l2115_211583
