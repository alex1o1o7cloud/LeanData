import Mathlib

namespace students_not_taking_test_l516_51676

theorem students_not_taking_test (total_students students_q1 students_q2 students_both not_taken : ℕ)
  (h_total : total_students = 30)
  (h_q1 : students_q1 = 25)
  (h_q2 : students_q2 = 22)
  (h_both : students_both = 22)
  (h_not_taken : not_taken = total_students - students_q2) :
  not_taken = 8 := by
  sorry

end students_not_taking_test_l516_51676


namespace solve_system_l516_51653

noncomputable def system_solution (x y : ℝ) :=
  x + y = 20 ∧ x * y = 36

theorem solve_system :
  (system_solution 18 2) ∧ (system_solution 2 18) :=
  sorry

end solve_system_l516_51653


namespace problem_statement_l516_51602

noncomputable def f (x : ℝ) : ℝ :=
if 0 ≤ x ∧ x ≤ 1 then 2 - x else 2 - (x % 2)

theorem problem_statement : 
  (∀ x : ℝ, f (-x) = f x) →
  (∀ x : ℝ, f (x + 1) + f x = 3) →
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → f x = 2 - x) →
  f (-2007.5) = 1.5 :=
by sorry

end problem_statement_l516_51602


namespace correlations_are_1_3_4_l516_51621

def relation1 : Prop := ∃ (age wealth : ℝ), true
def relation2 : Prop := ∀ (point : ℝ × ℝ), ∃ (coords : ℝ × ℝ), coords = point
def relation3 : Prop := ∃ (yield : ℝ) (climate : ℝ), true
def relation4 : Prop := ∃ (diameter height : ℝ), true
def relation5 : Prop := ∃ (student : Type) (school : Type), true

theorem correlations_are_1_3_4 :
  (relation1 ∨ relation3 ∨ relation4) ∧ ¬ (relation2 ∨ relation5) :=
sorry

end correlations_are_1_3_4_l516_51621


namespace newspapers_on_sunday_l516_51674

theorem newspapers_on_sunday (papers_weekend : ℕ) (diff_papers : ℕ) 
  (h1 : papers_weekend = 110) 
  (h2 : diff_papers = 20) 
  (h3 : ∃ (S Su : ℕ), Su = S + diff_papers ∧ S + Su = papers_weekend) :
  ∃ Su, Su = 65 :=
by
  sorry

end newspapers_on_sunday_l516_51674


namespace joyce_apples_l516_51654

theorem joyce_apples (initial_apples given_apples remaining_apples : ℕ) (h1 : initial_apples = 75) (h2 : given_apples = 52) (h3 : remaining_apples = initial_apples - given_apples) : remaining_apples = 23 :=
by
  rw [h1, h2] at h3
  exact h3

end joyce_apples_l516_51654


namespace sum_of_fractions_decimal_equivalence_l516_51651

theorem sum_of_fractions :
  (2 / 15 : ℚ) + (4 / 20) + (5 / 45) = 4 / 9 := 
sorry

theorem decimal_equivalence :
  (4 / 9 : ℚ) = 0.444 := 
sorry

end sum_of_fractions_decimal_equivalence_l516_51651


namespace find_k_intersection_l516_51609

theorem find_k_intersection :
  ∃ (k : ℝ), 
  (∀ (x y : ℝ), y = 2 * x + 3 → y = k * x + 1 → (x = 1 ∧ y = 5) → k = 4) :=
sorry

end find_k_intersection_l516_51609


namespace harmonica_value_l516_51603

theorem harmonica_value (x : ℕ) (h1 : ∃ k : ℕ, ∃ r : ℕ, x = 12 * k + r ∧ r ≠ 0 
                                                   ∧ r ≠ 6 ∧ r ≠ 9 
                                                   ∧ r ≠ 10 ∧ r ≠ 11)
                         (h2 : ¬ (x * x % 12 = 0)) : 
                         4 = 4 :=
by 
  sorry

end harmonica_value_l516_51603


namespace points_2_units_away_l516_51661

theorem points_2_units_away : (∃ x : ℝ, (x = -3 ∨ x = 1) ∧ (abs (x - (-1)) = 2)) :=
by
  sorry

end points_2_units_away_l516_51661


namespace Xiaohong_wins_5_times_l516_51649

theorem Xiaohong_wins_5_times :
  ∃ W L : ℕ, (3 * W - 2 * L = 1) ∧ (W + L = 12) ∧ W = 5 :=
by
  sorry

end Xiaohong_wins_5_times_l516_51649


namespace distance_from_dormitory_to_city_l516_51684

theorem distance_from_dormitory_to_city (D : ℝ) 
  (h1 : D = (1/2) * D + (1/4) * D + 6) : D = 24 := 
  sorry

end distance_from_dormitory_to_city_l516_51684


namespace set_list_method_l516_51652

theorem set_list_method : 
  {x : ℝ | x^2 - 2 * x + 1 = 0} = {1} :=
sorry

end set_list_method_l516_51652


namespace sum_series_eq_eight_l516_51665

noncomputable def sum_series : ℝ := ∑' n : ℕ, (3 * (n + 1) + 2) / 2^(n + 1)

theorem sum_series_eq_eight : sum_series = 8 := 
 by
  sorry

end sum_series_eq_eight_l516_51665


namespace angle_perpendicular_sides_l516_51667

theorem angle_perpendicular_sides (α β : ℝ) (hα : α = 80) 
  (h_perp : ∀ {x y}, ((x = α → y = 180 - x) ∨ (y = 180 - α → x = y))) : 
  β = 80 ∨ β = 100 :=
by
  sorry

end angle_perpendicular_sides_l516_51667


namespace minimum_cuts_to_unit_cubes_l516_51697

def cubes := List (ℕ × ℕ × ℕ)

def cube_cut (c : cubes) (n : ℕ) (dim : ℕ) : cubes :=
  sorry -- Function body not required for the statement

theorem minimum_cuts_to_unit_cubes (c : cubes) (s : ℕ) (dim : ℕ) :
  c = [(4,4,4)] ∧ s = 64 ∧ dim = 3 →
  ∃ (n : ℕ), n = 9 ∧
    (∀ cuts : cubes, cube_cut cuts n dim = [(1,1,1)]) :=
sorry

end minimum_cuts_to_unit_cubes_l516_51697


namespace power_sum_l516_51680

theorem power_sum :
  (-3)^3 + (-3)^2 + (-3) + 3 + 3^2 + 3^3 = 18 :=
by
  sorry

end power_sum_l516_51680


namespace right_triangle_property_l516_51600

-- Variables representing the lengths of the sides and the height of the right triangle
variables (a b c h : ℝ)

-- Hypotheses from the conditions
-- 1. a and b are the lengths of the legs of the right triangle
-- 2. c is the length of the hypotenuse
-- 3. h is the height to the hypotenuse
-- Given equation: 1/2 * a * b = 1/2 * c * h
def given_equation (a b c h : ℝ) : Prop := (1 / 2) * a * b = (1 / 2) * c * h

-- The theorem to prove
theorem right_triangle_property (a b c h : ℝ) (h_eq : given_equation a b c h) : (1 / a^2 + 1 / b^2) = 1 / h^2 :=
sorry

end right_triangle_property_l516_51600


namespace pencils_and_pens_cost_l516_51604

theorem pencils_and_pens_cost (p q : ℝ)
  (h1 : 8 * p + 3 * q = 5.60)
  (h2 : 2 * p + 5 * q = 4.25) :
  3 * p + 4 * q = 9.68 :=
sorry

end pencils_and_pens_cost_l516_51604


namespace intersection_union_complement_union_l516_51648

open Set

variable (U : Set ℝ) (A B : Set ℝ)
variable [Inhabited (Set ℝ)]

noncomputable def setA : Set ℝ := { x : ℝ | abs (x - 2) > 1 }
noncomputable def setB : Set ℝ := { x : ℝ | x ≥ 0 }

theorem intersection (U : Set ℝ) : 
  (setA ∩ setB) = { x : ℝ | (0 < x ∧ x < 1) ∨ x > 3 } := 
  sorry

theorem union (U : Set ℝ) : 
  (setA ∪ setB) = univ := 
  sorry

theorem complement_union (U : Set ℝ) : 
  ((U \ setA) ∪ setB) = { x : ℝ | x ≥ 0 } := 
  sorry

end intersection_union_complement_union_l516_51648


namespace triangle_ABC_right_angle_l516_51642

def point := (ℝ × ℝ)
def line (P: point) := P.1 = 5 ∨ ∃ a: ℝ, P.1 - 5 = a * (P.2 + 2)
def parabola (P: point) := P.2 ^ 2 = 4 * P.1
def perpendicular_slopes (k1 k2: ℝ) := k1 * k2 = -1

theorem triangle_ABC_right_angle (A B C: point) (P: point) 
  (hA: A = (1, 2))
  (hP: P = (5, -2))
  (h_line: line B ∧ line C)
  (h_parabola: parabola B ∧ parabola C):
  (∃ k_AB k_AC: ℝ, perpendicular_slopes k_AB k_AC) →
  ∃k_AB k_AC: ℝ, k_AB * k_AC = -1 :=
by sorry

end triangle_ABC_right_angle_l516_51642


namespace power_sum_prime_eq_l516_51685

theorem power_sum_prime_eq (p a n : ℕ) (hp : p.Prime) (h_eq : 2^p + 3^p = a^n) : n = 1 :=
by sorry

end power_sum_prime_eq_l516_51685


namespace unique_solution_condition_l516_51659

-- Define p and q as real numbers
variables (p q : ℝ)

-- The Lean statement to prove a unique solution when q ≠ 4
theorem unique_solution_condition : (∀ x : ℝ, (4 * x - 7 + p = q * x + 2) ↔ (q ≠ 4)) :=
by
  sorry

end unique_solution_condition_l516_51659


namespace necessary_but_not_sufficient_condition_l516_51650

theorem necessary_but_not_sufficient_condition (x : ℝ) :
  (x^2 - 3 * x < 0) → (0 < x ∧ x < 4) :=
sorry

end necessary_but_not_sufficient_condition_l516_51650


namespace expression_value_l516_51668

theorem expression_value (a b : ℤ) (ha : a = -4) (hb : b = 3) : 
  -2 * a - b ^ 3 + 2 * a * b = -43 := by
  rw [ha, hb]
  sorry

end expression_value_l516_51668


namespace curve1_line_and_circle_curve2_two_points_l516_51626

-- Define the first condition: x(x^2 + y^2 - 4) = 0
def curve1 (x y : ℝ) : Prop := x * (x^2 + y^2 - 4) = 0

-- Define the second condition: x^2 + (x^2 + y^2 - 4)^2 = 0
def curve2 (x y : ℝ) : Prop := x^2 + (x^2 + y^2 - 4)^2 = 0

-- The corresponding theorem statements
theorem curve1_line_and_circle : ∀ x y : ℝ, curve1 x y ↔ (x = 0 ∨ (x^2 + y^2 = 4)) := 
sorry 

theorem curve2_two_points : ∀ x y : ℝ, curve2 x y ↔ (x = 0 ∧ (y = 2 ∨ y = -2)) := 
sorry 

end curve1_line_and_circle_curve2_two_points_l516_51626


namespace payback_duration_l516_51677

-- Define constants for the problem conditions
def C : ℝ := 25000
def R : ℝ := 4000
def E : ℝ := 1500

-- Formal statement to be proven
theorem payback_duration : C / (R - E) = 10 := 
by
  sorry

end payback_duration_l516_51677


namespace find_number_l516_51633

theorem find_number (x : ℕ) (h : x + 18 = 44) : x = 26 :=
by
  sorry

end find_number_l516_51633


namespace complete_the_square_l516_51627

theorem complete_the_square (z : ℤ) : 
    z^2 - 6*z + 17 = (z - 3)^2 + 8 :=
sorry

end complete_the_square_l516_51627


namespace range_of_m_l516_51681

theorem range_of_m
  (h : ∀ x : ℝ, (m / (2 * x - 4) = (1 - x) / (2 - x) - 2) → x > 0) :
  m < 6 ∧ m ≠ 2 :=
by
  sorry

end range_of_m_l516_51681


namespace students_called_back_l516_51644

theorem students_called_back (g b d t c : ℕ) (h1 : g = 9) (h2 : b = 14) (h3 : d = 21) (h4 : t = g + b) (h5 : c = t - d) : c = 2 := by 
  sorry

end students_called_back_l516_51644


namespace Elmo_books_count_l516_51692

-- Define the number of books each person has
def Stu_books : ℕ := 4
def Laura_books : ℕ := 2 * Stu_books
def Elmo_books : ℕ := 3 * Laura_books

-- The theorem we need to prove
theorem Elmo_books_count : Elmo_books = 24 := by
  -- this part is skipped since no proof is required
  sorry

end Elmo_books_count_l516_51692


namespace euston_carriages_l516_51687

-- Definitions of the conditions
def E (N : ℕ) : ℕ := N + 20
def No : ℕ := 100
def FS : ℕ := No + 20
def total_carriages (E N : ℕ) : ℕ := E + N + No + FS

theorem euston_carriages (N : ℕ) (h : total_carriages (E N) N = 460) : E N = 130 :=
by
  -- Proof goes here
  sorry

end euston_carriages_l516_51687


namespace simplify_fraction_rationalize_denominator_l516_51693

noncomputable def sqrt (x : ℝ) : ℝ := Real.sqrt x
noncomputable def fraction := 5 / (sqrt 125 + 3 * sqrt 45 + 4 * sqrt 20 + sqrt 75)

theorem simplify_fraction_rationalize_denominator :
  fraction = sqrt 5 / 27 :=
by
  sorry

end simplify_fraction_rationalize_denominator_l516_51693


namespace more_soccer_balls_than_basketballs_l516_51690

theorem more_soccer_balls_than_basketballs :
  let soccer_boxes := 8
  let basketball_boxes := 5
  let balls_per_box := 12
  soccer_boxes * balls_per_box - basketball_boxes * balls_per_box = 36 := by
  sorry

end more_soccer_balls_than_basketballs_l516_51690


namespace unique_root_condition_l516_51669

theorem unique_root_condition (a : ℝ) : 
  (∀ x : ℝ, x^3 + a*x^2 - 4*a*x + a^2 - 4 = 0 → ∃! x₀ : ℝ, x = x₀) ↔ a < 1 :=
by sorry

end unique_root_condition_l516_51669


namespace odd_f_l516_51640

noncomputable def f (x : ℝ) : ℝ :=
if x > 0 then x^2 - 2^x else if x < 0 then -x^2 + 2^(-x) else 0

theorem odd_f (x : ℝ) : (f (-x) = -f x) :=
by
  sorry

end odd_f_l516_51640


namespace max_value_of_quadratic_function_l516_51630

noncomputable def quadratic_function (x : ℝ) : ℝ := -5*x^2 + 25*x - 15

theorem max_value_of_quadratic_function : ∃ x : ℝ, quadratic_function x = 750 :=
by
-- maximum value
sorry

end max_value_of_quadratic_function_l516_51630


namespace martha_weight_l516_51624

theorem martha_weight :
  ∀ (Bridget_weight : ℕ) (difference : ℕ) (Martha_weight : ℕ),
  Bridget_weight = 39 → difference = 37 →
  Bridget_weight = Martha_weight + difference →
  Martha_weight = 2 :=
by
  intros Bridget_weight difference Martha_weight hBridget hDifference hRelation
  sorry

end martha_weight_l516_51624


namespace john_needs_one_plank_l516_51689

theorem john_needs_one_plank (total_nails : ℕ) (nails_per_plank : ℕ) (extra_nails : ℕ) (P : ℕ)
    (h1 : total_nails = 11)
    (h2 : nails_per_plank = 3)
    (h3 : extra_nails = 8)
    (h4 : total_nails = nails_per_plank * P + extra_nails) :
    P = 1 :=
by
    sorry

end john_needs_one_plank_l516_51689


namespace num_dogs_correct_l516_51694

-- Definitions based on conditions
def total_animals : ℕ := 17
def number_of_cats : ℕ := 8

-- Definition based on required proof
def number_of_dogs : ℕ := total_animals - number_of_cats

-- Proof statement
theorem num_dogs_correct : number_of_dogs = 9 :=
by
  sorry

end num_dogs_correct_l516_51694


namespace profit_margin_A_cost_price_B_units_purchased_l516_51691

variables (cost_price_A selling_price_A selling_price_B profit_margin_B total_units total_cost : ℕ)
variables (units_A units_B : ℕ)

-- Conditions
def condition1 : cost_price_A = 40 := sorry
def condition2 : selling_price_A = 60 := sorry
def condition3 : selling_price_B = 80 := sorry
def condition4 : profit_margin_B = 60 := sorry
def condition5 : total_units = 50 := sorry
def condition6 : total_cost = 2200 := sorry

-- Proof statements 
theorem profit_margin_A (h1 : cost_price_A = 40) (h2 : selling_price_A = 60) :
  (selling_price_A - cost_price_A) * 100 / cost_price_A = 50 :=
by sorry

theorem cost_price_B (h3 : selling_price_B = 80) (h4 : profit_margin_B = 60) :
  (selling_price_B * 100) / (100 + profit_margin_B) = 50 :=
by sorry

theorem units_purchased (h5 : 40 * units_A + 50 * units_B = 2200)
  (h6 : units_A + units_B = 50) :
  units_A = 30 ∧ units_B = 20 :=
by sorry


end profit_margin_A_cost_price_B_units_purchased_l516_51691


namespace hexagon_area_ratio_l516_51638

open Real

theorem hexagon_area_ratio (r s : ℝ) (h_eq_diam : s = r * sqrt 3) :
    (let a1 := (3 * sqrt 3 / 2) * ((3 * r / 4) ^ 2)
     let a2 := (3 * sqrt 3 / 2) * r^2
     a1 / a2 = 9 / 16) :=
by
  sorry

end hexagon_area_ratio_l516_51638


namespace amount_spent_on_marbles_l516_51628

/-- A theorem to determine the amount Mike spent on marbles. -/
theorem amount_spent_on_marbles 
  (total_amount : ℝ) 
  (cost_football : ℝ) 
  (cost_baseball : ℝ) 
  (total_amount_eq : total_amount = 20.52)
  (cost_football_eq : cost_football = 4.95)
  (cost_baseball_eq : cost_baseball = 6.52) :
  ∃ (cost_marbles : ℝ), cost_marbles = total_amount - (cost_football + cost_baseball) 
  ∧ cost_marbles = 9.05 := 
by
  sorry

end amount_spent_on_marbles_l516_51628


namespace range_of_numbers_l516_51607

theorem range_of_numbers (a b c : ℕ) (h_mean : (a + b + c) / 3 = 4) (h_median : b = 4) (h_smallest : a = 1) :
  c - a = 6 :=
sorry

end range_of_numbers_l516_51607


namespace arithmetic_sequence_a_m_n_zero_l516_51699

theorem arithmetic_sequence_a_m_n_zero
  (a : ℕ → ℕ)
  (m n : ℕ) 
  (hm : m > 0) (hn : n > 0)
  (h_ma_m : a m = n)
  (h_na_n : a n = m) : 
  a (m + n) = 0 :=
by 
  sorry

end arithmetic_sequence_a_m_n_zero_l516_51699


namespace inequality_solution_l516_51636

theorem inequality_solution :
  {x : ℝ | (3 * x - 8) * (x - 4) / (x - 1) ≥ 0 } = { x : ℝ | x < 1 } ∪ { x : ℝ | x ≥ 4 } :=
by {
  sorry
}

end inequality_solution_l516_51636


namespace eval_expression_l516_51637

theorem eval_expression : (Real.pi + 2023)^0 + 2 * Real.sin (45 * Real.pi / 180) - (1 / 2)^(-1 : ℤ) + abs (Real.sqrt 2 - 2) = 1 :=
by
  sorry

end eval_expression_l516_51637


namespace average_of_hidden_primes_l516_51660

theorem average_of_hidden_primes (p₁ p₂ : ℕ) (h₁ : Nat.Prime p₁) (h₂ : Nat.Prime p₂) (h₃ : p₁ + 37 = p₂ + 53) : 
  (p₁ + p₂) / 2 = 11 := 
by
  sorry

end average_of_hidden_primes_l516_51660


namespace area_of_walkways_l516_51605

-- Define the dimensions of the individual flower bed
def flower_bed_width : ℕ := 8
def flower_bed_height : ℕ := 3

-- Define the number of rows and columns of flower beds
def rows_of_beds : ℕ := 4
def cols_of_beds : ℕ := 3

-- Define the width of the walkways
def walkway_width : ℕ := 2

-- Calculate the total width and height of the garden including walkways
def total_width : ℕ := (cols_of_beds * flower_bed_width) + (cols_of_beds + 1) * walkway_width
def total_height : ℕ := (rows_of_beds * flower_bed_height) + (rows_of_beds + 1) * walkway_width

-- Calculate the area of the garden including walkways
def total_area : ℕ := total_width * total_height

-- Calculate the total area of all the flower beds
def total_beds_area : ℕ := (rows_of_beds * cols_of_beds) * (flower_bed_width * flower_bed_height)

-- Prove the area of walkways
theorem area_of_walkways : total_area - total_beds_area = 416 := by
  sorry

end area_of_walkways_l516_51605


namespace decreasing_function_implies_inequality_l516_51608

theorem decreasing_function_implies_inequality (k b : ℝ) (h : ∀ x : ℝ, (2 * k + 1) * x + b = (2 * k + 1) * x + b) :
  (∀ x1 x2 : ℝ, x1 < x2 → (2 * k + 1) * x1 + b > (2 * k + 1) * x2 + b) → k < -1/2 :=
by sorry

end decreasing_function_implies_inequality_l516_51608


namespace actual_time_l516_51641

variables (m_pos : ℕ) (h_pos : ℕ)

-- The mirrored positions
def minute_hand_in_mirror : ℕ := 10
def hour_hand_in_mirror : ℕ := 5

theorem actual_time (m_pos h_pos : ℕ) 
  (hm : m_pos = 2) 
  (hh : h_pos < 7 ∧ h_pos ≥ 6) : 
  m_pos = 10 ∧ h_pos < 7 ∧ h_pos ≥ 6 :=
sorry

end actual_time_l516_51641


namespace solve_real_equation_l516_51657

theorem solve_real_equation (x : ℝ) (h1 : x ≠ -1) (h2 : x ≠ -3) :
  (x ^ 3 + 3 * x ^ 2 - x) / (x ^ 2 + 4 * x + 3) + x = -7 ↔ x = -5 / 2 ∨ x = -4 := 
by
  sorry

end solve_real_equation_l516_51657


namespace find_PB_l516_51623

variables (P A B C D : Point) (PA PD PC PB : ℝ)
-- Assume P is interior to rectangle ABCD
-- Conditions
axiom hPA : PA = 3
axiom hPD : PD = 4
axiom hPC : PC = 5

-- The main statement to prove
theorem find_PB (P A B C D : Point) (PA PD PC PB : ℝ)
  (hPA : PA = 3) (hPD : PD = 4) (hPC : PC = 5) : PB = 3 * Real.sqrt 2 :=
by
  sorry

end find_PB_l516_51623


namespace distinct_arrangements_apple_l516_51647

theorem distinct_arrangements_apple : 
  let n := 5
  let freq_p := 2
  let freq_a := 1
  let freq_l := 1
  let freq_e := 1
  (Nat.factorial n) / (Nat.factorial freq_p * Nat.factorial freq_a * Nat.factorial freq_l * Nat.factorial freq_e) = 60 :=
by
  sorry

end distinct_arrangements_apple_l516_51647


namespace jovana_shells_l516_51663

theorem jovana_shells :
  let jovana_initial := 5
  let first_friend := 15
  let second_friend := 17
  jovana_initial + first_friend + second_friend = 37 := by
  sorry

end jovana_shells_l516_51663


namespace initial_boxes_l516_51678

theorem initial_boxes (x : ℕ) (h : x + 6 = 14) : x = 8 :=
by sorry

end initial_boxes_l516_51678


namespace empty_tank_time_l516_51606

-- Definitions based on problem conditions
def tank_full_fraction := 1 / 5
def pipeA_fill_time := 15
def pipeB_empty_time := 6

-- Derived definitions
def rate_of_pipeA := 1 / pipeA_fill_time
def rate_of_pipeB := 1 / pipeB_empty_time
def combined_rate := rate_of_pipeA - rate_of_pipeB 

-- The time to empty the tank when both pipes are open
def time_to_empty (initial_fraction : ℚ) (combined_rate : ℚ) : ℚ :=
  initial_fraction / -combined_rate

-- The main theorem to prove
theorem empty_tank_time
  (initial_fraction : ℚ := tank_full_fraction)
  (combined_rate : ℚ := combined_rate)
  (time : ℚ := time_to_empty initial_fraction combined_rate) :
  time = 2 :=
by
  sorry

end empty_tank_time_l516_51606


namespace program_output_l516_51635

theorem program_output (a : ℕ) (h : a = 3) : (if a < 10 then 2 * a else a * a) = 6 :=
by
  rw [h]
  norm_num

end program_output_l516_51635


namespace probability_of_first_four_cards_each_suit_l516_51618

noncomputable def probability_first_four_different_suits : ℚ := 3 / 32

theorem probability_of_first_four_cards_each_suit :
  let n := 52
  let k := 5
  let suits := 4
  (probability_first_four_different_suits = (3 / 32)) :=
by
  sorry

end probability_of_first_four_cards_each_suit_l516_51618


namespace intersection_A_B_range_m_l516_51610

-- Definitions for Sets A, B, and C
def SetA : Set ℝ := { x | -2 ≤ x ∧ x < 5 }
def SetB : Set ℝ := { x | 3 * x - 5 ≥ x - 1 }
def SetC (m : ℝ) : Set ℝ := { x | -x + m > 0 }

-- Problem 1: Prove \( A \cap B = \{ x \mid 2 \leq x < 5 \} \)
theorem intersection_A_B : SetA ∩ SetB = { x : ℝ | 2 ≤ x ∧ x < 5 } :=
by
  sorry

-- Problem 2: Prove \( m \in [5, +\infty) \) given \( A \cup C = C \)
theorem range_m (m : ℝ) : (SetA ∪ SetC m = SetC m) → m ∈ Set.Ici 5 :=
by
  sorry

end intersection_A_B_range_m_l516_51610


namespace find_cost_prices_l516_51601

-- These represent the given selling prices of the items.
def SP_computer_table : ℝ := 3600
def SP_office_chair : ℝ := 5000
def SP_bookshelf : ℝ := 1700

-- These represent the percentage markups and discounts as multipliers.
def markup_computer_table : ℝ := 1.20
def markup_office_chair : ℝ := 1.25
def discount_bookshelf : ℝ := 0.85

-- The problem requires us to find the cost prices. We will define these as variables.
variable (C O B : ℝ)

theorem find_cost_prices :
  (SP_computer_table = C * markup_computer_table) ∧
  (SP_office_chair = O * markup_office_chair) ∧
  (SP_bookshelf = B * discount_bookshelf) →
  (C = 3000) ∧ (O = 4000) ∧ (B = 2000) :=
by
  sorry

end find_cost_prices_l516_51601


namespace geometric_sequence_general_term_l516_51672

theorem geometric_sequence_general_term (n : ℕ) (a : ℕ → ℕ) (a1 : ℕ) (q : ℕ) 
  (h1 : a1 = 4) (h2 : q = 3) (h3 : ∀ n, a n = a1 * (q ^ (n - 1))) :
  a n = 4 * 3^(n - 1) := by
  sorry

end geometric_sequence_general_term_l516_51672


namespace combined_average_mark_l516_51629

theorem combined_average_mark 
  (n_A n_B n_C n_D n_E : ℕ) 
  (avg_A avg_B avg_C avg_D avg_E : ℕ)
  (students_A : n_A = 22) (students_B : n_B = 28)
  (students_C : n_C = 15) (students_D : n_D = 35)
  (students_E : n_E = 25)
  (avg_marks_A : avg_A = 40) (avg_marks_B : avg_B = 60)
  (avg_marks_C : avg_C = 55) (avg_marks_D : avg_D = 75)
  (avg_marks_E : avg_E = 50) : 
  (22 * 40 + 28 * 60 + 15 * 55 + 35 * 75 + 25 * 50) / (22 + 28 + 15 + 35 + 25) = 58.08 := 
  by 
    sorry

end combined_average_mark_l516_51629


namespace triangle_ABC_is_acute_l516_51655

noncomputable def arithmeticSeqTerm (a1 d : ℝ) (n : ℕ) : ℝ :=
  a1 + (n - 1) * d

noncomputable def geometricSeqTerm (a1 r : ℝ) (n : ℕ) : ℝ :=
  a1 * r^(n - 1)

def tanA_condition (a1 d : ℝ) :=
  arithmeticSeqTerm a1 d 3 = -4 ∧ arithmeticSeqTerm a1 d 7 = 4

def tanB_condition (a1 r : ℝ) :=
  geometricSeqTerm a1 r 3 = 1/3 ∧ geometricSeqTerm a1 r 6 = 9

theorem triangle_ABC_is_acute {A B : ℝ} (a1a da a1b rb : ℝ) 
  (hA : tanA_condition a1a da) 
  (hB : tanB_condition a1b rb) :
  0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ (A + B) < π :=
  sorry

end triangle_ABC_is_acute_l516_51655


namespace second_discount_is_5_percent_l516_51645

noncomputable def salePriceSecondDiscount (initialPrice finalPrice priceAfterFirstDiscount: ℝ) : ℝ :=
  (initialPrice - priceAfterFirstDiscount) + (priceAfterFirstDiscount - finalPrice)

noncomputable def secondDiscountPercentage (initialPrice finalPrice priceAfterFirstDiscount: ℝ) : ℝ :=
  (priceAfterFirstDiscount - finalPrice) / priceAfterFirstDiscount * 100

theorem second_discount_is_5_percent :
  ∀ (initialPrice finalPrice priceAfterFirstDiscount: ℝ),
    initialPrice = 600 ∧
    finalPrice = 456 ∧
    priceAfterFirstDiscount = initialPrice * 0.80 →
    secondDiscountPercentage initialPrice finalPrice priceAfterFirstDiscount = 5 :=
by
  intros
  sorry

end second_discount_is_5_percent_l516_51645


namespace determine_number_of_shelves_l516_51658

-- Define the total distance Karen bikes round trip
def total_distance : ℕ := 3200

-- Define the number of books per shelf
def books_per_shelf : ℕ := 400

-- Calculate the one-way distance from Karen's home to the library
def one_way_distance (total_distance : ℕ) : ℕ := total_distance / 2

-- Define the total number of books, which is the same as the one-way distance
def total_books (one_way_distance : ℕ) : ℕ := one_way_distance

-- Calculate the number of shelves
def number_of_shelves (total_books : ℕ) (books_per_shelf : ℕ) : ℕ :=
  total_books / books_per_shelf

theorem determine_number_of_shelves :
  number_of_shelves (total_books (one_way_distance total_distance)) books_per_shelf = 4 :=
by 
  -- the proof would go here
  sorry

end determine_number_of_shelves_l516_51658


namespace y_axis_symmetry_l516_51617

theorem y_axis_symmetry (x y : ℝ) (P : ℝ × ℝ) (hx : P = (-5, 3)) : 
  (P.1 = -5 ∧ P.2 = 3) → (P.1 * -1, P.2) = (5, 3) :=
by
  intro h
  rw [hx]
  simp [Neg.neg, h]
  sorry

end y_axis_symmetry_l516_51617


namespace find_larger_integer_l516_51620

-- Defining the problem statement with the given conditions
theorem find_larger_integer (x : ℕ) (h : (x + 6) * 2 = 4 * x) : 4 * x = 24 :=
sorry

end find_larger_integer_l516_51620


namespace transform_center_l516_51666

def point := (ℝ × ℝ)

def reflect_x_axis (p : point) : point :=
  (p.1, -p.2)

def translate_right (p : point) (d : ℝ) : point :=
  (p.1 + d, p.2)

theorem transform_center (C : point) (hx : C = (3, -4)) :
  translate_right (reflect_x_axis C) 3 = (6, 4) :=
by
  sorry

end transform_center_l516_51666


namespace sum_of_d_and_e_l516_51670

theorem sum_of_d_and_e (d e : ℤ) : 
  (∃ d e : ℤ, ∀ x : ℝ, x^2 - 24 * x + 50 = (x + d)^2 + e) → d + e = -106 :=
by
  sorry

end sum_of_d_and_e_l516_51670


namespace problem_statement_l516_51611

noncomputable def f : ℝ → ℝ := sorry  -- Define f as a noncomputable function to accommodate the problem constraints

variables (a : ℝ)

theorem problem_statement (periodic_f : ∀ x, f (x + 3) = f x)
    (odd_f : ∀ x, f (-x) = -f x)
    (ineq_f1 : f 1 < 1)
    (eq_f2 : f 2 = (2*a-1)/(a+1)) :
    a < -1 ∨ 0 < a :=
by
  sorry

end problem_statement_l516_51611


namespace sufficient_not_necessary_condition_l516_51634

theorem sufficient_not_necessary_condition (a : ℝ) :
  (∀ x : ℝ, (1 ≤ x ∧ x ≤ 2) → (x^2 - a ≤ 0)) → (a ≥ 5) :=
sorry

end sufficient_not_necessary_condition_l516_51634


namespace tesla_ratio_l516_51643

variables (s c e : ℕ)
variables (h1 : e = s + 10) (h2 : c = 6) (h3 : e = 13)

theorem tesla_ratio : s / c = 1 / 2 :=
by
  sorry

end tesla_ratio_l516_51643


namespace find_a_l516_51622

theorem find_a (a : ℝ) : 
  (∀ (i : ℂ), i^2 = -1 → (a * i / (2 - i) + 1 = 2 * i)) → a = 5 :=
by
  intro h
  sorry

end find_a_l516_51622


namespace non_empty_prime_subsets_count_l516_51664

-- Definition of the set S
def S : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

-- Definition of primes in S
def prime_subset_S : Set ℕ := {x ∈ S | Nat.Prime x}

-- The statement to prove
theorem non_empty_prime_subsets_count : 
  ∃ n, n = 15 ∧ ∀ T ⊆ prime_subset_S, T ≠ ∅ → ∃ m, n = 2^m - 1 := 
by
  sorry

end non_empty_prime_subsets_count_l516_51664


namespace solve_equation_l516_51679

theorem solve_equation (x : ℝ) : x * (x - 1) = 0 ↔ x = 0 ∨ x = 1 := 
by
  sorry  -- Placeholder for the proof

end solve_equation_l516_51679


namespace region_ratio_l516_51646

theorem region_ratio (side_length : ℝ) (s r : ℝ) 
  (h1 : side_length = 2)
  (h2 : s = (1 / 2) * (1 : ℝ) * (1 : ℝ))
  (h3 : r = (1 / 2) * (Real.sqrt 2) * (Real.sqrt 2)) :
  r / s = 2 :=
by
  sorry

end region_ratio_l516_51646


namespace completing_the_square_x_squared_plus_4x_plus_3_eq_0_l516_51625

theorem completing_the_square_x_squared_plus_4x_plus_3_eq_0 :
  (x : ℝ) → x^2 + 4 * x + 3 = 0 → (x + 2)^2 = 1 :=
by
  intros x h
  -- The actual proof will be provided here
  sorry

end completing_the_square_x_squared_plus_4x_plus_3_eq_0_l516_51625


namespace part1_solution_set_part2_range_of_a_l516_51675

-- Definitions of f and g as provided in the problem.
def f (x : ℝ) : ℝ := |x + 1| + |x - 2|
def g (x a : ℝ) : ℝ := |x + 1| - |x - a| + a

-- Problem 1: Prove the solution set for f(x) ≤ 5 is [-2, 3]
theorem part1_solution_set : { x : ℝ | f x ≤ 5 } = { x : ℝ | -2 ≤ x ∧ x ≤ 3 } :=
  sorry

-- Problem 2: Prove the range of a when f(x) ≥ g(x) always holds is (-∞, 1]
theorem part2_range_of_a (a : ℝ) (h : ∀ x : ℝ, f x ≥ g x a) : a ≤ 1 :=
  sorry

end part1_solution_set_part2_range_of_a_l516_51675


namespace range_of_m_l516_51696

noncomputable def set_A := { x : ℝ | x^2 + x - 6 = 0 }
noncomputable def set_B (m : ℝ) := { x : ℝ | m * x + 1 = 0 }

theorem range_of_m (m : ℝ) : set_A ∪ set_B m = set_A → m = 0 ∨ m = -1 / 2 ∨ m = 1 / 3 :=
by
  sorry

end range_of_m_l516_51696


namespace units_digit_n_l516_51614

theorem units_digit_n (m n : ℕ) (h1 : m * n = 31 ^ 6) (h2 : m % 10 = 9) : n % 10 = 2 := 
sorry

end units_digit_n_l516_51614


namespace linear_system_solution_l516_51683

theorem linear_system_solution (x y : ℝ) (h1 : x = 2) (h2 : y = -3) : x + y = -1 :=
by
  sorry

end linear_system_solution_l516_51683


namespace find_s_l516_51612

theorem find_s (x y : Real -> Real) : 
  (x 2 = 2 ∧ y 2 = 5) ∧ 
  (x 6 = 6 ∧ y 6 = 17) ∧ 
  (x 10 = 10 ∧ y 10 = 29) ∧ 
  (∀ x, y x = 3 * x - 1) -> 
  (y 34 = 101) := 
by 
  sorry

end find_s_l516_51612


namespace asymptotes_of_hyperbola_l516_51688

theorem asymptotes_of_hyperbola (a b x y : ℝ) (h : a = 5 ∧ b = 2) :
  (x^2 / 25 - y^2 / 4 = 1) → (y = (2 / 5) * x ∨ y = -(2 / 5) * x) :=
by
  sorry

end asymptotes_of_hyperbola_l516_51688


namespace sum_of_digits_of_N_l516_51673

theorem sum_of_digits_of_N (N : ℕ) (h : N * (N + 1) / 2 = 2016) : (6 + 3 = 9) :=
by
  sorry

end sum_of_digits_of_N_l516_51673


namespace solve_xyz_l516_51615

theorem solve_xyz (a b c x y z : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h1 : x + y + z = a + b + c)
  (h2 : 4 * x * y * z - (a^2 * x + b^2 * y + c^2 * z) = a * b * c) :
  (x, y, z) = ( (b + c) / 2, (c + a) / 2, (a + b) / 2 ) :=
sorry

end solve_xyz_l516_51615


namespace trisha_total_distance_l516_51631

theorem trisha_total_distance :
  let d1 := 0.1111111111111111
  let d2 := 0.1111111111111111
  let d3 := 0.6666666666666666
  d1 + d2 + d3 = 0.8888888888888888 := 
by
  sorry

end trisha_total_distance_l516_51631


namespace statement_two_statement_three_l516_51662

section
variables {R : Type*} [Field R]
variables (a b c p q : R)
noncomputable def f (x : R) := a * x^2 + b * x + c

-- Statement ②
theorem statement_two (hpq : f a b c p = f a b c q) (hpq_neq : p ≠ q) : 
  f a b c (p + q) = c :=
sorry

-- Statement ③
theorem statement_three (hf : f a b c (p + q) = c) (hpq_neq : p ≠ q) : 
  p + q = 0 ∨ f a b c p = f a b c q :=
sorry

end

end statement_two_statement_three_l516_51662


namespace number_of_blue_tiles_is_16_l516_51639

def length_of_floor : ℕ := 20
def breadth_of_floor : ℕ := 10
def tile_length : ℕ := 2

def total_tiles : ℕ := (length_of_floor / tile_length) * (breadth_of_floor / tile_length)

def black_tiles : ℕ :=
  let rows_length := 2 * (length_of_floor / tile_length)
  let rows_breadth := 2 * (breadth_of_floor / tile_length)
  (rows_length + rows_breadth) - 4

def remaining_tiles : ℕ := total_tiles - black_tiles
def white_tiles : ℕ := remaining_tiles / 3
def blue_tiles : ℕ := remaining_tiles - white_tiles

theorem number_of_blue_tiles_is_16 :
  blue_tiles = 16 :=
by
  sorry

end number_of_blue_tiles_is_16_l516_51639


namespace value_of_fg_neg_one_l516_51695

def f (x : ℝ) : ℝ := x - 2

def g (x : ℝ) : ℝ := x^2 + 4 * x + 3

theorem value_of_fg_neg_one : f (g (-1)) = -2 :=
by
  sorry

end value_of_fg_neg_one_l516_51695


namespace minimize_x_2y_l516_51613

noncomputable def minimum_value_x_2y (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 3 / (x + 2) + 3 / (y + 2) = 1) : ℝ :=
  x + 2 * y

theorem minimize_x_2y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 3 / (x + 2) + 3 / (y + 2) = 1) :
  minimum_value_x_2y x y hx hy h = 3 + 6 * Real.sqrt 2 :=
sorry

end minimize_x_2y_l516_51613


namespace decreasing_interval_ln_quadratic_l516_51632

theorem decreasing_interval_ln_quadratic :
  ∀ x : ℝ, (x < 1 ∨ x > 3) → (∀ a b : ℝ, (a ≤ b) → (a < 1 ∨ a > 3) → (b < 1 ∨ b > 3) → (a ≤ x ∧ x ≤ b → (x^2 - 4 * x + 3) ≥ (b^2 - 4 * b + 3))) :=
by
  sorry

end decreasing_interval_ln_quadratic_l516_51632


namespace equation_of_BC_area_of_triangle_l516_51698

section triangle_geometry

variables (x y : ℝ)

/-- Given equations of the altitudes and vertex A, the equation of side BC is 2x + 3y + 7 = 0 -/
theorem equation_of_BC (h1 : 2 * x - 3 * y + 1 = 0) (h2 : x + y = 0) (A : ℝ × ℝ) (hA : A = (1, 2)) :
  ∃ (a b c : ℝ), (a * x + b * y + c = 0) ∧ (a, b, c) = (2, 3, 7) := 
sorry

/-- Given equations of the altitudes and vertex A, the area of triangle ABC is 45/2 -/
theorem area_of_triangle (h1 : 2 * x - 3 * y + 1 = 0) (h2 : x + y = 0) (A : ℝ × ℝ) (hA : A = (1, 2)) :
  ∃ (area : ℝ), (area = (45 / 2)) := 
sorry

end triangle_geometry

end equation_of_BC_area_of_triangle_l516_51698


namespace find_lunch_break_duration_l516_51616

def lunch_break_duration : ℝ → ℝ → ℝ → ℝ
  | s, a, L => L

theorem find_lunch_break_duration (s a L : ℝ) :
  (8 - L) * (s + a) = 0.6 ∧ (6.4 - L) * a = 0.28 ∧ (9.6 - L) * s = 0.12 →
  lunch_break_duration s a L = 1 :=
  by
    sorry

end find_lunch_break_duration_l516_51616


namespace zack_initial_marbles_l516_51656

noncomputable def total_initial_marbles (x : ℕ) : ℕ :=
  81 * x + 27

theorem zack_initial_marbles :
  ∃ x : ℕ, total_initial_marbles x = 270 :=
by
  use 3
  sorry

end zack_initial_marbles_l516_51656


namespace complex_multiplication_l516_51682

theorem complex_multiplication : ∀ (i : ℂ), i^2 = -1 → i * (2 + 3 * i) = (-3 : ℂ) + 2 * i :=
by
  intros i hi
  sorry

end complex_multiplication_l516_51682


namespace sue_votes_correct_l516_51619

def total_votes : ℕ := 1000
def percentage_others : ℝ := 0.65
def sue_votes : ℕ := 350

theorem sue_votes_correct :
  sue_votes = (total_votes : ℝ) * (1 - percentage_others) :=
by
  sorry

end sue_votes_correct_l516_51619


namespace average_length_of_strings_l516_51671

theorem average_length_of_strings (l1 l2 l3 : ℝ) (hl1 : l1 = 2) (hl2 : l2 = 5) (hl3 : l3 = 3) : 
  (l1 + l2 + l3) / 3 = 10 / 3 :=
by
  rw [hl1, hl2, hl3]
  change (2 + 5 + 3) / 3 = 10 / 3
  sorry

end average_length_of_strings_l516_51671


namespace smallest_solution_of_quadratic_l516_51686

theorem smallest_solution_of_quadratic :
  ∃ x : ℝ, 6 * x^2 - 29 * x + 35 = 0 ∧ x = 7 / 3 :=
sorry

end smallest_solution_of_quadratic_l516_51686
