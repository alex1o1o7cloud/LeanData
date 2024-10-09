import Mathlib

namespace max_value_l1159_115908

theorem max_value (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : 9 * a^2 + 4 * b^2 + c^2 = 91) :
  a + 2 * b + 3 * c ≤ 30.333 :=
by
  sorry

end max_value_l1159_115908


namespace sum_is_five_or_negative_five_l1159_115986

theorem sum_is_five_or_negative_five (a b c d : ℤ) 
  (h1 : a ≠ b) (h2 : a ≠ c) (h3 : a ≠ d) 
  (h4 : b ≠ c) (h5 : b ≠ d) (h6 : c ≠ d)
  (h7 : a * b * c * d = 14) : 
  (a + b + c + d = 5) ∨ (a + b + c + d = -5) :=
by
  sorry

end sum_is_five_or_negative_five_l1159_115986


namespace debate_organizing_committees_count_l1159_115929

theorem debate_organizing_committees_count :
    ∃ (n : ℕ), n = 5 * (Nat.choose 8 4) * (Nat.choose 8 3)^4 ∧ n = 3442073600 :=
by
  sorry

end debate_organizing_committees_count_l1159_115929


namespace inequality_proof_l1159_115955

theorem inequality_proof (a b : ℝ) (h1 : a > 1) (h2 : b > 1) :
    (a^2 / (b - 1)) + (b^2 / (a - 1)) ≥ 8 := 
by
  sorry

end inequality_proof_l1159_115955


namespace perfect_square_trinomial_l1159_115982

theorem perfect_square_trinomial (m : ℝ) :
  (∃ a b : ℝ, (x : ℝ) → (x^2 + 2 * (m - 1) * x + 16) = (a * x + b)^2) → (m = 5 ∨ m = -3) :=
by
  sorry

end perfect_square_trinomial_l1159_115982


namespace union_sets_l1159_115945

def M : Set ℝ := {x | -1 ≤ x ∧ x < 3}
def N : Set ℝ := {x | 2 < x ∧ x ≤ 5}

theorem union_sets :
  M ∪ N = {x | -1 ≤ x ∧ x ≤ 5} := by
  sorry

end union_sets_l1159_115945


namespace erasers_in_each_box_l1159_115914

theorem erasers_in_each_box (boxes : ℕ) (price_per_eraser : ℚ) (total_money_made : ℚ) (total_erasers_sold : ℕ) (erasers_per_box : ℕ) :
  boxes = 48 → price_per_eraser = 0.75 → total_money_made = 864 → total_erasers_sold = 1152 → total_erasers_sold / boxes = erasers_per_box → erasers_per_box = 24 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end erasers_in_each_box_l1159_115914


namespace f_2021_value_l1159_115906

def A : Set ℚ := {x | x ≠ -1 ∧ x ≠ 0}

def f (x : ℚ) : ℝ := sorry -- Placeholder for function definition with its properties

axiom f_property : ∀ x ∈ A, f x + f (1 + 1 / x) = 1 / 2 * Real.log (|x|)

theorem f_2021_value : f 2021 = 1 / 2 * Real.log 2021 :=
by
  sorry

end f_2021_value_l1159_115906


namespace onion_to_carrot_ratio_l1159_115981

theorem onion_to_carrot_ratio (p c o g : ℕ) (h1 : 6 * p = c) (h2 : c = o) (h3 : g = 1 / 3 * o) (h4 : p = 2) (h5 : g = 8) : o / c = 1 / 1 :=
by
  sorry

end onion_to_carrot_ratio_l1159_115981


namespace solve_system_of_equations_l1159_115930

theorem solve_system_of_equations (x y_1 y_2 y_3: ℝ) (n : ℤ) (h1 : -3 ≤ n) (h2 : n ≤ 3)
  (h_eq1 : (1 - x^2) * y_1 = 2 * x)
  (h_eq2 : (1 - y_1^2) * y_2 = 2 * y_1)
  (h_eq3 : (1 - y_2^2) * y_3 = 2 * y_2)
  (h_eq4 : y_3 = x) :
  y_1 = Real.tan (2 * n * Real.pi / 7) ∧
  y_2 = Real.tan (4 * n * Real.pi / 7) ∧
  y_3 = Real.tan (n * Real.pi / 7) ∧
  x = Real.tan (n * Real.pi / 7) :=
sorry

end solve_system_of_equations_l1159_115930


namespace quadratic_roots_product_l1159_115954

theorem quadratic_roots_product :
  ∀ (x1 x2: ℝ), (x1^2 - 4 * x1 - 2 = 0 ∧ x2^2 - 4 * x2 - 2 = 0) → (x1 * x2 = -2) :=
by
  -- Assume x1 and x2 are roots of the quadratic equation
  intros x1 x2 h
  sorry

end quadratic_roots_product_l1159_115954


namespace positional_relationship_l1159_115927

-- Defining the concepts of parallelism, containment, and positional relationships
structure Line -- subtype for a Line
structure Plane -- subtype for a Plane

-- Definitions and Conditions
def is_parallel_to (l : Line) (p : Plane) : Prop := sorry  -- A line being parallel to a plane
def is_contained_in (l : Line) (p : Plane) : Prop := sorry  -- A line being contained within a plane
def are_skew (l₁ l₂ : Line) : Prop := sorry  -- Two lines being skew
def are_parallel (l₁ l₂ : Line) : Prop := sorry  -- Two lines being parallel

-- Given conditions
variables (a b : Line) (α : Plane)
axiom Ha : is_parallel_to a α
axiom Hb : is_contained_in b α

-- The theorem to be proved
theorem positional_relationship (a b : Line) (α : Plane) 
  (Ha : is_parallel_to a α) 
  (Hb : is_contained_in b α) : 
  (are_skew a b ∨ are_parallel a b) :=
sorry

end positional_relationship_l1159_115927


namespace part_I_min_value_part_II_a_range_l1159_115919

noncomputable def f (x a : ℝ) : ℝ := abs (2 * x - a) - abs (x + 3)

theorem part_I_min_value (x : ℝ) : f x 1 ≥ -7 / 2 :=
by sorry 

theorem part_II_a_range (x a : ℝ) (hx : 0 ≤ x) (hx' : x ≤ 3) (hf : f x a ≤ 4) : -4 ≤ a ∧ a ≤ 7 :=
by sorry

end part_I_min_value_part_II_a_range_l1159_115919


namespace combined_ages_l1159_115971

theorem combined_ages (h_age : ℕ) (diff : ℕ) (years_later : ℕ) (hurley_age : h_age = 14) 
                       (age_difference : diff = 20) (years_passed : years_later = 40) : 
                       h_age + diff + years_later * 2 = 128 := by
  sorry

end combined_ages_l1159_115971


namespace find_number_l1159_115942

theorem find_number (x : ℝ) : (8 * x = 0.4 * 900) -> x = 45 :=
by
  sorry

end find_number_l1159_115942


namespace tangent_line_slope_l1159_115987

theorem tangent_line_slope (h : ℝ → ℝ) (a : ℝ) (P : ℝ × ℝ) 
  (tangent_eq : ∀ x y, 2 * x + y + 1 = 0 ↔ (x, y) = (a, h a)) : 
  deriv h a < 0 :=
sorry

end tangent_line_slope_l1159_115987


namespace value_of_expression_l1159_115949

theorem value_of_expression (a b c d m : ℝ)
  (h1 : a = -b)
  (h2 : c * d = 1)
  (h3 : |m| = 5)
  : 2 * (a + b) - 3 * c * d + m = 2 ∨ 2 * (a + b) - 3 * c * d + m = -8 := by
  sorry

end value_of_expression_l1159_115949


namespace pond_length_l1159_115944

theorem pond_length (L W S : ℝ) (h1 : L = 2 * W) (h2 : L = 80) (h3 : S^2 = (1/50) * (L * W)) : S = 8 := 
by 
  -- Insert proof here 
  sorry

end pond_length_l1159_115944


namespace max_triangles_l1159_115990

theorem max_triangles (n : ℕ) (h : n = 10) : 
  ∃ T : ℕ, T = 150 :=
by
  sorry

end max_triangles_l1159_115990


namespace no_solution_for_99_l1159_115909

theorem no_solution_for_99 :
  ∃ n : ℕ, (¬ ∃ x y : ℕ, 0 < x ∧ 0 < y ∧ 9 * x + 11 * y = n) ∧
  (∀ m : ℕ, n < m → ∃ x y : ℕ, 0 < x ∧ 0 < y ∧ 9 * x + 11 * y = m) ∧
  n = 99 :=
by
  sorry

end no_solution_for_99_l1159_115909


namespace find_k_l1159_115934

theorem find_k (k : ℝ) (h : (3, 1) ∈ {(x, y) | y = k * x - 2} ∧ k ≠ 0) : k = 1 :=
by sorry

end find_k_l1159_115934


namespace find_triplet_l1159_115997

theorem find_triplet (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (x + y) ^ 2 + 3 * x + y + 1 = z ^ 2 → y = x ∧ z = 2 * x + 1 :=
by
  sorry

end find_triplet_l1159_115997


namespace consumer_installment_credit_value_l1159_115965

variable (consumer_installment_credit : ℝ) 

noncomputable def automobile_installment_credit := 0.36 * consumer_installment_credit

noncomputable def finance_company_credit := 35

theorem consumer_installment_credit_value :
  (∃ C : ℝ, automobile_installment_credit C = 0.36 * C ∧ finance_company_credit = (1 / 3) * automobile_installment_credit C) →
  consumer_installment_credit = 291.67 :=
by
  sorry

end consumer_installment_credit_value_l1159_115965


namespace negation_of_statement_equivalence_l1159_115998

-- Definitions of the math club and enjoyment of puzzles
def member_of_math_club (x : Type) : Prop := sorry
def enjoys_puzzles (x : Type) : Prop := sorry

-- Original statement: All members of the math club enjoy puzzles
def original_statement : Prop :=
∀ x, member_of_math_club x → enjoys_puzzles x

-- Negation of the original statement
def negated_statement : Prop :=
∃ x, member_of_math_club x ∧ ¬ enjoys_puzzles x

-- Proof problem statement
theorem negation_of_statement_equivalence :
  ¬ original_statement ↔ negated_statement :=
sorry

end negation_of_statement_equivalence_l1159_115998


namespace roots_of_polynomial_l1159_115951

noncomputable def polynomial : Polynomial ℤ := Polynomial.X^3 - 4 * Polynomial.X^2 - Polynomial.X + 4

theorem roots_of_polynomial :
  (Polynomial.X - 1) * (Polynomial.X + 1) * (Polynomial.X - 4) = polynomial :=
by
  sorry

end roots_of_polynomial_l1159_115951


namespace arith_seq_ratio_l1159_115935

variables {a₁ d : ℝ} (h₁ : d ≠ 0) (h₂ : (a₁ + 2*d)^2 ≠ a₁ * (a₁ + 8*d))

theorem arith_seq_ratio:
  (a₁ + 2*d) / (a₁ + 5*d) = 1 / 2 :=
sorry

end arith_seq_ratio_l1159_115935


namespace range_of_b_l1159_115933

theorem range_of_b (x b : ℝ) (hb : b > 0) : 
  (∃ x : ℝ, |x - 2| + |x + 1| < b) ↔ b > 3 :=
by
  sorry

end range_of_b_l1159_115933


namespace simplified_expression_l1159_115918

variable (m : ℝ) (h : m = Real.sqrt 3)

theorem simplified_expression : (m - (m + 9) / (m + 1)) / ((m^2 + 3 * m) / (m + 1)) = 1 - Real.sqrt 3 :=
by
  rw [h]
  sorry

end simplified_expression_l1159_115918


namespace width_decreased_by_28_6_percent_l1159_115973

theorem width_decreased_by_28_6_percent (L W : ℝ) (A : ℝ) 
    (hA : A = L * W) (hL : 1.4 * L * (W / 1.4) = A) :
    (1 - (W / 1.4 / W)) * 100 = 28.6 :=
by 
  sorry

end width_decreased_by_28_6_percent_l1159_115973


namespace parabola_equations_l1159_115953

theorem parabola_equations (x y : ℝ) (h₁ : (0, 0) = (0, 0)) (h₂ : (-2, 3) = (-2, 3)) :
  (x^2 = 4 / 3 * y) ∨ (y^2 = - 9 / 2 * x) :=
sorry

end parabola_equations_l1159_115953


namespace rectangles_greater_than_one_area_l1159_115988

theorem rectangles_greater_than_one_area (n : ℕ) (H : n = 5) : ∃ r, r = 84 :=
by
  sorry

end rectangles_greater_than_one_area_l1159_115988


namespace find_f_2013_l1159_115991

open Function

theorem find_f_2013 {f : ℝ → ℝ} (Hodd : ∀ x, f (-x) = -f x)
  (Hperiodic : ∀ x, f (x + 4) = f x)
  (Hf_neg1 : f (-1) = 2) :
  f 2013 = -2 := by
sorry

end find_f_2013_l1159_115991


namespace right_triangle_hypotenuse_l1159_115994

theorem right_triangle_hypotenuse (a b c : ℕ) (h1 : a^2 + b^2 = c^2) 
  (h2 : b = c - 1575) (h3 : b < 1991) : c = 1800 :=
sorry

end right_triangle_hypotenuse_l1159_115994


namespace Tim_scores_expected_value_l1159_115913

theorem Tim_scores_expected_value :
  let LAIMO := 15
  let FARML := 10
  let DOMO := 50
  let p := 1 / 3
  let expected_LAIMO := LAIMO * p
  let expected_FARML := FARML * p
  let expected_DOMO := DOMO * p
  expected_LAIMO + expected_FARML + expected_DOMO = 25 :=
by
  -- The Lean proof would go here
  sorry

end Tim_scores_expected_value_l1159_115913


namespace fraction_equality_l1159_115928

theorem fraction_equality (x : ℝ) : (5 + x) / (7 + x) = (3 + x) / (4 + x) → x = -1 :=
by
  sorry

end fraction_equality_l1159_115928


namespace brian_has_78_white_stones_l1159_115969

-- Given conditions
variables (W B : ℕ) (R Bl : ℕ)
variables (x : ℕ)
variables (total_stones : ℕ := 330)
variables (total_collection1 : ℕ := 100)
variables (total_collection3 : ℕ := 130)

-- Condition: First collection stones sum to 100
#check W + B = 100

-- Condition: Brian has more white stones than black ones
#check W > B

-- Condition: Ratio of red to blue stones is 3:2 in the third collection
#check R + Bl = 130
#check R = 3 * x
#check Bl = 2 * x

-- Condition: Total number of stones in all three collections is 330
#check total_stones = total_collection1 + total_collection1 + total_collection3

-- New collection's magnetic stones ratio condition
#check 2 * W / 78 = 2

-- Prove that Brian has 78 white stones
theorem brian_has_78_white_stones
  (h1 : W + B = 100)
  (h2 : W > B)
  (h3 : R + Bl = 130)
  (h4 : R = 3 * x)
  (h5 : Bl = 2 * x)
  (h6 : 2 * W / 78 = 2) :
  W = 78 :=
sorry

end brian_has_78_white_stones_l1159_115969


namespace simplify_fraction_l1159_115962

theorem simplify_fraction (x y : ℚ) (hx : x = 3) (hy : y = 2) : 
  (9 * x^3 * y^2) / (12 * x^2 * y^4) = 9 / 16 := by
  sorry

end simplify_fraction_l1159_115962


namespace total_cookies_l1159_115939

theorem total_cookies
  (num_bags : ℕ)
  (cookies_per_bag : ℕ)
  (h_num_bags : num_bags = 286)
  (h_cookies_per_bag : cookies_per_bag = 452) :
  num_bags * cookies_per_bag = 129272 :=
by
  sorry

end total_cookies_l1159_115939


namespace f_2_solutions_l1159_115940

theorem f_2_solutions : 
  ∀ (x y : ℤ), 
    (1 ≤ x) ∧ (0 ≤ y) ∧ (y ≤ (-x + 2)) → 
    (∃ (a b c : Int), 
      (a = 1 ∧ (b = 0 ∨ b = 1) ∨ 
       a = 2 ∧ b = 0) ∧ 
      a = x ∧ b = y ∨ 
      c = 3 → false) ∧ 
    (∃ n : ℕ, n = 3) := by
  sorry

end f_2_solutions_l1159_115940


namespace solution_set_of_inequality_l1159_115943

theorem solution_set_of_inequality (x : ℝ) : (x^2 - 2*x - 5 > 2*x) ↔ (x > 5 ∨ x < -1) :=
by sorry

end solution_set_of_inequality_l1159_115943


namespace bobby_position_after_100_turns_l1159_115968

def movement_pattern (start_pos : ℤ × ℤ) (n : ℕ) : (ℤ × ℤ) :=
  let x := start_pos.1 - ((2 * (n / 4 + 1) + 3 * (n / 4)) * ((n + 1) / 4))
  let y := start_pos.2 + ((2 * (n / 4 + 1) + 2 * (n / 4)) * ((n + 1) / 4))
  if n % 4 == 0 then (x, y)
  else if n % 4 == 1 then (x, y + 2 * ((n + 3) / 4) + 1)
  else if n % 4 == 2 then (x - 3 * ((n + 5) / 4), y + 2 * ((n + 3) / 4) + 1)
  else (x - 3 * ((n + 5) / 4) + 3, y + 2 * ((n + 3) / 4) - 2)

theorem bobby_position_after_100_turns :
  movement_pattern (10, -10) 100 = (-667, 640) :=
sorry

end bobby_position_after_100_turns_l1159_115968


namespace range_of_a_l1159_115959

noncomputable def p (a : ℝ) : Prop := ∀ x : ℝ, x^2 - 2*x > a

noncomputable def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0

theorem range_of_a (a : ℝ) (h1 : p a ∨ q a) (h2 : ¬ (p a ∧ q a)) :
  (a > -2 ∧ a < -1) ∨ (a ≥ 1) :=
by
  sorry

end range_of_a_l1159_115959


namespace milk_production_l1159_115967

variable (a b c d e : ℝ)

theorem milk_production (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) (he : 0 < e) :
  let rate_per_cow_per_day := b / (a * c)
  let production_per_day := d * rate_per_cow_per_day
  let total_production := production_per_day * e
  total_production = (b * d * e) / (a * c) :=
by
  sorry

end milk_production_l1159_115967


namespace mean_proportional_49_64_l1159_115985

theorem mean_proportional_49_64 : Real.sqrt (49 * 64) = 56 :=
by
  sorry

end mean_proportional_49_64_l1159_115985


namespace rectangle_length_35_l1159_115925

theorem rectangle_length_35
  (n_rectangles : ℕ) (area_abcd : ℝ) (rect_length_multiple : ℕ) (rect_width_multiple : ℕ) 
  (n_rectangles_eq : n_rectangles = 6)
  (area_abcd_eq : area_abcd = 4800)
  (rect_length_multiple_eq : rect_length_multiple = 3)
  (rect_width_multiple_eq : rect_width_multiple = 2) :
  ∃ y : ℝ, round y = 35 ∧ y^2 * (4/3) = area_abcd :=
by
  sorry


end rectangle_length_35_l1159_115925


namespace creative_sum_l1159_115992

def letterValue (ch : Char) : Int :=
  let n := (ch.toNat - 'a'.toNat + 1) % 12
  if n = 0 then 2
  else if n = 1 then 1
  else if n = 2 then 2
  else if n = 3 then 3
  else if n = 4 then 2
  else if n = 5 then 1
  else if n = 6 then 0
  else if n = 7 then -1
  else if n = 8 then -2
  else if n = 9 then -3
  else if n = 10 then -2
  else if n = 11 then -1
  else 0 -- this should never happen

def wordValue (word : String) : Int :=
  word.foldl (λ acc ch => acc + letterValue ch) 0

theorem creative_sum : wordValue "creative" = -2 :=
  by
    sorry

end creative_sum_l1159_115992


namespace Gage_skating_time_l1159_115915

theorem Gage_skating_time :
  let min_per_hr := 60
  let skating_6_days := 6 * (1 * min_per_hr + 20)
  let skating_4_days := 4 * (1 * min_per_hr + 35)
  let needed_total := 11 * 90
  let skating_10_days := skating_6_days + skating_4_days
  let minutes_on_eleventh_day := needed_total - skating_10_days
  minutes_on_eleventh_day = 130 :=
by
  sorry

end Gage_skating_time_l1159_115915


namespace student_marks_l1159_115999

theorem student_marks 
    (correct: ℕ) 
    (attempted: ℕ) 
    (marks_per_correct: ℕ) 
    (marks_per_incorrect: ℤ) 
    (correct_answers: correct = 27)
    (attempted_questions: attempted = 70)
    (marks_per_correct_condition: marks_per_correct = 3)
    (marks_per_incorrect_condition: marks_per_incorrect = -1): 
    (correct * marks_per_correct + (attempted - correct) * marks_per_incorrect) = 38 :=
by
    sorry

end student_marks_l1159_115999


namespace num_k_vals_l1159_115917

-- Definitions of the conditions
def div_by_7 (n k : ℕ) : Prop :=
  (2 * 3^(6*n) + k * 2^(3*n + 1) - 1) % 7 = 0

-- Main theorem statement
theorem num_k_vals : 
  ∃ (S : Finset ℕ), (∀ k ∈ S, k < 100 ∧ ∀ n, div_by_7 n k) ∧ S.card = 14 := 
by
  sorry

end num_k_vals_l1159_115917


namespace necessary_condition_l1159_115923

variables (a b : ℝ)

theorem necessary_condition (h : a > b) : a > b - 1 :=
sorry

end necessary_condition_l1159_115923


namespace friends_team_division_l1159_115952

theorem friends_team_division :
  let num_friends : ℕ := 8
  let num_teams : ℕ := 4
  let ways_to_divide := num_teams ^ num_friends
  ways_to_divide = 65536 :=
by
  sorry

end friends_team_division_l1159_115952


namespace wall_length_correct_l1159_115956

noncomputable def length_of_wall : ℝ :=
  let volume_of_one_brick := 25 * 11.25 * 6
  let total_volume_of_bricks := volume_of_one_brick * 6800
  let wall_width := 600
  let wall_height := 22.5
  total_volume_of_bricks / (wall_width * wall_height)

theorem wall_length_correct : length_of_wall = 850 := by
  sorry

end wall_length_correct_l1159_115956


namespace sum_of_four_digit_numbers_l1159_115964

open Nat

theorem sum_of_four_digit_numbers (s : Finset ℤ) :
  (∀ x, x ∈ s → (∃ k, x = 30 * k + 2) ∧ 1000 ≤ x ∧ x ≤ 9999) →
  s.sum id = 1652100 := by
  sorry

end sum_of_four_digit_numbers_l1159_115964


namespace triangle_isosceles_l1159_115974

theorem triangle_isosceles
  (A B C : ℝ) -- Angles of the triangle, A, B, and C
  (h1 : A = 2 * C) -- Condition 1: Angle A equals twice angle C
  (h2 : B = 2 * C) -- Condition 2: Angle B equals twice angle C
  (h3 : A + B + C = 180) -- Sum of angles in a triangle equals 180 degrees
  : A = B := -- Conclusion: with the conditions above, angles A and B are equal
by
  sorry

end triangle_isosceles_l1159_115974


namespace a_values_l1159_115947

def A (a : ℤ) : Set ℤ := {2, a^2 - a + 2, 1 - a}

theorem a_values (a : ℤ) (h : 4 ∈ A a) : a = 2 ∨ a = -3 :=
sorry

end a_values_l1159_115947


namespace circleAtBottomAfterRotation_l1159_115963

noncomputable def calculateFinalCirclePosition (initialPosition : String) (sides : ℕ) : String :=
  if (sides = 8) then (if initialPosition = "bottom" then "bottom" else "unknown") else "unknown"

theorem circleAtBottomAfterRotation :
  calculateFinalCirclePosition "bottom" 8 = "bottom" :=
by
  sorry

end circleAtBottomAfterRotation_l1159_115963


namespace chocolate_chip_cookie_price_l1159_115966

noncomputable def price_of_chocolate_chip_cookies :=
  let total_boxes := 1585
  let total_revenue := 1586.75
  let plain_boxes := 793.375
  let price_of_plain := 0.75
  let revenue_plain := plain_boxes * price_of_plain
  let choco_boxes := total_boxes - plain_boxes
  (993.71875 - revenue_plain) / choco_boxes

theorem chocolate_chip_cookie_price :
  price_of_chocolate_chip_cookies = 1.2525 :=
by sorry

end chocolate_chip_cookie_price_l1159_115966


namespace expression_indeterminate_l1159_115936

-- Given variables a, b, c, d which are real numbers
variables {a b c d : ℝ}

-- Statement asserting that the expression is indeterminate under given conditions
theorem expression_indeterminate
  (h : true) :
  ¬∃ k, (a^2 + b^2 - c^2 - 2 * b * d)/(a^2 + c^2 - b^2 - 2 * c * d) = k :=
sorry

end expression_indeterminate_l1159_115936


namespace not_q_is_false_l1159_115941

variable (n : ℤ)

-- Definition of the propositions
def p (n : ℤ) : Prop := 2 * n - 1 % 2 = 1 -- 2n - 1 is odd
def q (n : ℤ) : Prop := (2 * n + 1) % 2 = 0 -- 2n + 1 is even

-- Proof statement: Not q is false, meaning q is false
theorem not_q_is_false (n : ℤ) : ¬ q n = False := sorry

end not_q_is_false_l1159_115941


namespace N_divisible_by_9_l1159_115916

def sum_of_digits (n : ℕ) : ℕ := sorry

theorem N_divisible_by_9 (N : ℕ) (h : sum_of_digits N = sum_of_digits (5 * N)) : N % 9 = 0 := 
sorry

end N_divisible_by_9_l1159_115916


namespace kendra_bought_3_hats_l1159_115904

-- Define the price of a wooden toy
def price_of_toy : ℕ := 20

-- Define the price of a hat
def price_of_hat : ℕ := 10

-- Define the amount Kendra went to the shop with
def initial_amount : ℕ := 100

-- Define the number of wooden toys Kendra bought
def number_of_toys : ℕ := 2

-- Define the amount of change Kendra received
def change_received : ℕ := 30

-- Prove that Kendra bought 3 hats
theorem kendra_bought_3_hats : 
  initial_amount - change_received - (number_of_toys * price_of_toy) = 3 * price_of_hat := by
  sorry

end kendra_bought_3_hats_l1159_115904


namespace notebook_cost_l1159_115905

theorem notebook_cost (n p : ℝ) (h1 : n + p = 2.40) (h2 : n = 2 + p) : n = 2.20 := by
  sorry

end notebook_cost_l1159_115905


namespace ball_min_bounces_reach_target_height_l1159_115984

noncomputable def minimum_bounces (initial_height : ℝ) (ratio : ℝ) (target_height : ℝ) : ℕ :=
  Nat.ceil (Real.log (target_height / initial_height) / Real.log ratio)

theorem ball_min_bounces_reach_target_height :
  minimum_bounces 20 (2 / 3) 2 = 6 :=
by
  -- This is where the proof would go, but we use sorry to skip it
  sorry

end ball_min_bounces_reach_target_height_l1159_115984


namespace carter_stretching_legs_frequency_l1159_115910

-- Given conditions
def tripDuration : ℤ := 14 * 60 -- in minutes
def foodStops : ℤ := 2
def gasStops : ℤ := 3
def pitStopDuration : ℤ := 20 -- in minutes
def totalTripDuration : ℤ := 18 * 60 -- in minutes

-- Prove that Carter stops to stretch his legs every 2 hours
theorem carter_stretching_legs_frequency :
  ∃ (stretchingStops : ℤ), (totalTripDuration - tripDuration = (foodStops + gasStops + stretchingStops) * pitStopDuration) ∧
    (stretchingStops * pitStopDuration = totalTripDuration - (tripDuration + (foodStops + gasStops) * pitStopDuration)) ∧
    (14 / stretchingStops = 2) :=
by sorry

end carter_stretching_legs_frequency_l1159_115910


namespace problem1_problem2_l1159_115932

-- Definition of the function
def f (a x : ℝ) := x^2 + a * x + 3

-- Problem statement 1: Prove that if f(x) ≥ a for all x ∈ ℜ, then a ≤ 3.
theorem problem1 (a : ℝ) : (∀ x : ℝ, f a x ≥ a) → a ≤ 3 := sorry

-- Problem statement 2: Prove that if f(x) ≥ a for all x ∈ [-2, 2], then -6 ≤ a ≤ 2.
theorem problem2 (a : ℝ) : (∀ x : ℝ, -2 ≤ x ∧ x ≤ 2 → f a x ≥ a) → -6 ≤ a ∧ a ≤ 2 := sorry

end problem1_problem2_l1159_115932


namespace area_of_circumscribed_circle_l1159_115937

noncomputable def circumradius_of_equilateral_triangle (s : ℝ) : ℝ :=
  s / Real.sqrt 3

noncomputable def area_of_circle (r : ℝ) : ℝ :=
  Real.pi * r^2

theorem area_of_circumscribed_circle (s : ℝ) (h : s = 12) : area_of_circle (circumradius_of_equilateral_triangle s) = 48 * Real.pi := by
  sorry

end area_of_circumscribed_circle_l1159_115937


namespace sum_arithmetic_series_remainder_l1159_115938

theorem sum_arithmetic_series_remainder :
  let a := 2
  let l := 12
  let d := 1
  let n := (l - a) / d + 1
  let S := n / 2 * (a + l)
  S % 9 = 5 :=
by
  let a := 2
  let l := 12
  let d := 1
  let n := (l - a) / d + 1
  let S := n / 2 * (a + l)
  show S % 9 = 5
  sorry

end sum_arithmetic_series_remainder_l1159_115938


namespace solve_quadratic_l1159_115979

theorem solve_quadratic {x : ℚ} (h1 : x > 0) (h2 : 3 * x ^ 2 + 11 * x - 20 = 0) : x = 4 / 3 :=
sorry

end solve_quadratic_l1159_115979


namespace inradius_of_regular_tetrahedron_l1159_115961

theorem inradius_of_regular_tetrahedron (h r : ℝ) (S : ℝ) 
  (h_eq: 4 * (1/3) * S * r = (1/3) * S * h) : r = (1/4) * h :=
sorry

end inradius_of_regular_tetrahedron_l1159_115961


namespace oscar_leap_vs_elmer_stride_l1159_115957

theorem oscar_leap_vs_elmer_stride :
  ∀ (num_poles : ℕ) (distance : ℝ) (elmer_strides_per_gap : ℕ) (oscar_leaps_per_gap : ℕ)
    (elmer_stride_time_mult : ℕ) (total_distance_poles : ℕ)
    (elmer_total_strides : ℕ) (oscar_total_leaps : ℕ) (elmer_stride_length : ℝ)
    (oscar_leap_length : ℝ) (expected_diff : ℝ),
    num_poles = 81 →
    distance = 10560 →
    elmer_strides_per_gap = 60 →
    oscar_leaps_per_gap = 15 →
    elmer_stride_time_mult = 2 →
    total_distance_poles = 2 →
    elmer_total_strides = elmer_strides_per_gap * (num_poles - 1) →
    oscar_total_leaps = oscar_leaps_per_gap * (num_poles - 1) →
    elmer_stride_length = distance / elmer_total_strides →
    oscar_leap_length = distance / oscar_total_leaps →
    expected_diff = oscar_leap_length - elmer_stride_length →
    expected_diff = 6.6
:= sorry

end oscar_leap_vs_elmer_stride_l1159_115957


namespace total_miles_traveled_l1159_115907

noncomputable def initial_fee : ℝ := 2.0
noncomputable def charge_per_2_5_mile : ℝ := 0.35
noncomputable def total_charge : ℝ := 5.15

theorem total_miles_traveled :
  ∃ (miles : ℝ), total_charge = initial_fee + (charge_per_2_5_mile * miles * (5 / 2)) ∧ miles = 3.6 :=
by
  sorry

end total_miles_traveled_l1159_115907


namespace abc_inequality_l1159_115902

open Real

noncomputable def posReal (x : ℝ) : Prop := x > 0

theorem abc_inequality (a b c : ℝ) 
  (hCond1 : posReal a) 
  (hCond2 : posReal b) 
  (hCond3 : posReal c) 
  (hCond4 : a * b * c = 1) : 
  (a - 1 + 1 / b) * (b - 1 + 1 / c) * (c - 1 + 1 / a) ≤ 1 :=
by
  sorry

end abc_inequality_l1159_115902


namespace martin_crayons_l1159_115931

theorem martin_crayons : (8 * 7 = 56) := by
  sorry

end martin_crayons_l1159_115931


namespace mild_numbers_with_mild_squares_count_l1159_115972

def is_mild (n : ℕ) : Prop :=
  ∀ d ∈ n.digits 3, d = 0 ∨ d = 1

theorem mild_numbers_with_mild_squares_count :
  ∃ count : ℕ, count = 7 ∧ (∀ n : ℕ, 1 ≤ n ∧ n ≤ 1000 → is_mild n → is_mild (n * n)) → count = 7 := by
  sorry

end mild_numbers_with_mild_squares_count_l1159_115972


namespace pablo_days_to_complete_puzzles_l1159_115920

-- Define the given conditions 
def puzzle_pieces_300 := 300
def puzzle_pieces_500 := 500
def puzzles_300 := 8
def puzzles_500 := 5
def rate_per_hour := 100
def max_hours_per_day := 7

-- Calculate total number of pieces
def total_pieces_300 := puzzles_300 * puzzle_pieces_300
def total_pieces_500 := puzzles_500 * puzzle_pieces_500
def total_pieces := total_pieces_300 + total_pieces_500

-- Calculate the number of pieces Pablo can put together per day
def pieces_per_day := max_hours_per_day * rate_per_hour

-- Calculate the number of days required for Pablo to complete all puzzles
def days_to_complete := total_pieces / pieces_per_day

-- Proposition to prove
theorem pablo_days_to_complete_puzzles : days_to_complete = 7 := sorry

end pablo_days_to_complete_puzzles_l1159_115920


namespace problem1_problem2_l1159_115960

-- Problem 1
theorem problem1 :
  (Real.sqrt 48 / Real.sqrt 3 - Real.sqrt (1 / 2) * Real.sqrt 48 + Real.sqrt 54 = 4 + Real.sqrt 6) :=
by
  sorry

-- Problem 2
theorem problem2 :
  (Real.sqrt 8 + Real.sqrt 32 - Real.sqrt 2 = 5 * Real.sqrt 2) :=
by
  sorry

end problem1_problem2_l1159_115960


namespace wallet_amount_l1159_115901

-- Definitions of given conditions
def num_toys := 28
def cost_per_toy := 10
def num_teddy_bears := 20
def cost_per_teddy_bear := 15

-- Calculation of total costs
def total_cost_of_toys := num_toys * cost_per_toy
def total_cost_of_teddy_bears := num_teddy_bears * cost_per_teddy_bear

-- Total amount of money in Louise's wallet
def total_cost := total_cost_of_toys + total_cost_of_teddy_bears

-- Proof that the total cost is $580
theorem wallet_amount : total_cost = 580 :=
by
  -- Skipping the proof for now
  sorry

end wallet_amount_l1159_115901


namespace find_semi_perimeter_l1159_115948

noncomputable def semi_perimeter_of_rectangle (a b : ℝ) (h₁ : a * b = 4024) (h₂ : a = 2 * b) : ℝ :=
  (a + b) / 2

theorem find_semi_perimeter (a b : ℝ) (h₁ : a * b = 4024) (h₂ : a = 2 * b) : semi_perimeter_of_rectangle a b h₁ h₂ = (3 / 2) * Real.sqrt 2012 :=
  sorry

end find_semi_perimeter_l1159_115948


namespace speed_of_current_l1159_115976

theorem speed_of_current (m c : ℝ) (h1 : m + c = 18) (h2 : m - c = 11.2) : c = 3.4 :=
by
  sorry

end speed_of_current_l1159_115976


namespace count_ways_to_exhaust_black_matches_l1159_115946

theorem count_ways_to_exhaust_black_matches 
  (n r g : ℕ) 
  (h_r_le_n : r ≤ n) 
  (h_g_le_n : g ≤ n) 
  (h_r_ge_0 : 0 ≤ r) 
  (h_g_ge_0 : 0 ≤ g) 
  (h_n_ge_0 : 0 < n) :
  ∃ ways : ℕ, ways = (Nat.factorial (3 * n - r - g - 1)) / (Nat.factorial (n - 1) * Nat.factorial (n - r) * Nat.factorial (n - g)) :=
by
  sorry

end count_ways_to_exhaust_black_matches_l1159_115946


namespace original_polygon_sides_l1159_115996

theorem original_polygon_sides {n : ℕ} 
    (hn : (n - 2) * 180 = 1080) : n = 7 ∨ n = 8 ∨ n = 9 :=
sorry

end original_polygon_sides_l1159_115996


namespace symmetric_point_product_l1159_115950

theorem symmetric_point_product (x y : ℤ) (h1 : (2008, y) = (-x, -1)) : x * y = -2008 :=
by {
  sorry
}

end symmetric_point_product_l1159_115950


namespace henry_books_l1159_115903

def initial_books := 99
def boxes := 3
def books_per_box := 15
def room_books := 21
def coffee_table_books := 4
def kitchen_books := 18
def picked_books := 12

theorem henry_books :
  (initial_books - (boxes * books_per_box + room_books + coffee_table_books + kitchen_books) + picked_books) = 23 :=
by
  sorry

end henry_books_l1159_115903


namespace triangle_side_s_l1159_115993

/-- The sides of a triangle have lengths 8, 13, and s where s is a whole number.
    What is the smallest possible value of s?
    We need to show that the minimum possible value of s such that 8 + s > 13,
    s < 21, and 13 + s > 8 is s = 6. -/
theorem triangle_side_s (s : ℕ) : 
  (8 + s > 13) ∧ (8 + 13 > s) ∧ (13 + s > 8) → s = 6 :=
by
  sorry

end triangle_side_s_l1159_115993


namespace inequality_ln_x_lt_x_lt_exp_x_l1159_115970

noncomputable def f (x a : ℝ) : ℝ := Real.exp x - a * x - 1
noncomputable def g (x : ℝ) : ℝ := Real.log x - x

theorem inequality_ln_x_lt_x_lt_exp_x (x : ℝ) (h : x > 0) : Real.log x < x ∧ x < Real.exp x := by
  -- We need to supply the proof here
  sorry

end inequality_ln_x_lt_x_lt_exp_x_l1159_115970


namespace solve_for_y_l1159_115980

theorem solve_for_y (x y : ℤ) (h1 : x + y = 290) (h2 : x - y = 200) : y = 45 := 
by 
  sorry

end solve_for_y_l1159_115980


namespace taxi_cost_per_mile_l1159_115911

variable (x : ℝ)

-- Mike's total cost
def Mike_total_cost := 2.50 + 36 * x

-- Annie's total cost
def Annie_total_cost := 2.50 + 5.00 + 16 * x

-- The primary theorem to prove
theorem taxi_cost_per_mile : Mike_total_cost x = Annie_total_cost x → x = 0.25 := by
  sorry

end taxi_cost_per_mile_l1159_115911


namespace pool_water_amount_correct_l1159_115989

noncomputable def water_in_pool_after_ten_hours : ℝ :=
  let h1 := 8
  let h2_3 := 10 * 2
  let h4_5 := 14 * 2
  let h6 := 12
  let h7 := 12 - 8
  let h8 := 12 - 18
  let h9 := 12 - 24
  let h10 := 6
  h1 + h2_3 + h4_5 + h6 + h7 + h8 + h9 + h10

theorem pool_water_amount_correct :
  water_in_pool_after_ten_hours = 60 := 
sorry

end pool_water_amount_correct_l1159_115989


namespace isosceles_triangle_perimeter_l1159_115926

theorem isosceles_triangle_perimeter (a b c : ℝ) 
  (h1 : a = 4 ∨ b = 4 ∨ c = 4) 
  (h2 : a = 8 ∨ b = 8 ∨ c = 8) 
  (isosceles : a = b ∨ b = c ∨ a = c) : 
  a + b + c = 20 :=
by
  sorry

end isosceles_triangle_perimeter_l1159_115926


namespace min_value_x_plus_reciprocal_min_value_x_plus_reciprocal_equality_at_one_l1159_115978

theorem min_value_x_plus_reciprocal (x : ℝ) (h : x > 0) : x + 1 / x ≥ 2 :=
by
  sorry

theorem min_value_x_plus_reciprocal_equality_at_one : (1 : ℝ) + 1 / 1 = 2 :=
by
  norm_num

end min_value_x_plus_reciprocal_min_value_x_plus_reciprocal_equality_at_one_l1159_115978


namespace middle_card_number_is_6_l1159_115921

noncomputable def middle_card_number : ℕ :=
  6

theorem middle_card_number_is_6 (a b c : ℕ) (h1 : a < b) (h2 : b < c) (h3 : a + b + c = 17)
  (casey_cannot_determine : ∀ (x : ℕ), (a = x) → ∃ (y z : ℕ), y ≠ z ∧ a + y + z = 17 ∧ a < y ∧ y < z)
  (tracy_cannot_determine : ∀ (x : ℕ), (c = x) → ∃ (y z : ℕ), y ≠ z ∧ y + z + c = 17 ∧ y < z ∧ z < c)
  (stacy_cannot_determine : ∀ (x : ℕ), (b = x) → ∃ (y z : ℕ), y ≠ z ∧ y + b + z = 17 ∧ y < b ∧ b < z) : 
  b = middle_card_number :=
sorry

end middle_card_number_is_6_l1159_115921


namespace sum_of_factors_of_120_is_37_l1159_115912

theorem sum_of_factors_of_120_is_37 :
  ∃ a b c d e : ℤ, (a * b = 120) ∧ (b = a + 1) ∧ (c * d * e = 120) ∧ (d = c + 1) ∧ (e = d + 1) ∧ (a + b + c + d + e = 37) :=
by
  sorry

end sum_of_factors_of_120_is_37_l1159_115912


namespace gcd_288_123_l1159_115900

-- Define the conditions
def cond1 : 288 = 2 * 123 + 42 := by sorry
def cond2 : 123 = 2 * 42 + 39 := by sorry
def cond3 : 42 = 39 + 3 := by sorry
def cond4 : 39 = 13 * 3 := by sorry

-- Prove that GCD of 288 and 123 is 3
theorem gcd_288_123 : Nat.gcd 288 123 = 3 := by
  sorry

end gcd_288_123_l1159_115900


namespace total_of_three_new_observations_l1159_115958

theorem total_of_three_new_observations (avg9 : ℕ) (num9 : ℕ) 
(new_obs : ℕ) (new_avg_diff : ℕ) (new_num : ℕ) 
(total9 : ℕ) (new_avg : ℕ) (total12 : ℕ) : 
avg9 = 15 ∧ num9 = 9 ∧ new_obs = 3 ∧ new_avg_diff = 2 ∧
new_num = num9 + new_obs ∧ new_avg = avg9 - new_avg_diff ∧
total9 = num9 * avg9 ∧ total9 + 3 * (new_avg) = total12 → 
total12 - total9 = 21 := by sorry

end total_of_three_new_observations_l1159_115958


namespace total_candles_used_l1159_115975

def cakes_baked : ℕ := 8
def cakes_given_away : ℕ := 2
def remaining_cakes : ℕ := cakes_baked - cakes_given_away
def candles_per_cake : ℕ := 6

theorem total_candles_used : remaining_cakes * candles_per_cake = 36 :=
by
  -- proof omitted
  sorry

end total_candles_used_l1159_115975


namespace power_increase_fourfold_l1159_115983

theorem power_increase_fourfold 
    (F v : ℝ)
    (k : ℝ)
    (R : ℝ := k * v)
    (P_initial : ℝ := F * v)
    (v' : ℝ := 2 * v)
    (F' : ℝ := 2 * F)
    (R' : ℝ := k * v')
    (P_final : ℝ := F' * v') :
    P_final = 4 * P_initial := 
by
  sorry

end power_increase_fourfold_l1159_115983


namespace workshop_workers_l1159_115924

theorem workshop_workers (W N: ℕ) 
  (h1: 8000 * W = 70000 + 6000 * N) 
  (h2: W = 7 + N) : 
  W = 14 := 
  by 
    sorry

end workshop_workers_l1159_115924


namespace henry_geography_math_score_l1159_115995

variable (G M : ℕ)

theorem henry_geography_math_score (E : ℕ) (H : ℕ) (total_score : ℕ) 
  (hE : E = 66) 
  (hH : H = (G + M + E) / 3)
  (hTotal : G + M + E + H = total_score) 
  (htotal_score : total_score = 248) :
  G + M = 120 := 
by
  sorry

end henry_geography_math_score_l1159_115995


namespace sum_and_round_to_nearest_ten_l1159_115922

/-- A function to round a number to the nearest ten -/
def round_to_nearest_ten (n : ℕ) : ℕ :=
  if n % 10 < 5 then n - n % 10 else n + 10 - n % 10

/-- The sum of 54 and 29 rounded to the nearest ten is 80 -/
theorem sum_and_round_to_nearest_ten : round_to_nearest_ten (54 + 29) = 80 :=
by
  sorry

end sum_and_round_to_nearest_ten_l1159_115922


namespace evaluate_nested_fraction_l1159_115977

-- We start by defining the complex nested fraction
def nested_fraction : Rat :=
  1 / (3 - (1 / (3 - (1 / (3 - 1 / 3)))))

-- We assert that the value of the nested fraction is 8/21 
theorem evaluate_nested_fraction : nested_fraction = 8 / 21 := by
  sorry

end evaluate_nested_fraction_l1159_115977
