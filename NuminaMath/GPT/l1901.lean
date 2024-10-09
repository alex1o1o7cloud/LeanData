import Mathlib

namespace average_after_adding_ten_l1901_190149

theorem average_after_adding_ten (avg initial_sum new_mean : ℕ) (n : ℕ) (h1 : n = 15) (h2 : avg = 40) (h3 : initial_sum = n * avg) (h4 : new_mean = (initial_sum + n * 10) / n) : new_mean = 50 := 
by
  sorry

end average_after_adding_ten_l1901_190149


namespace typing_speed_in_6_minutes_l1901_190192

theorem typing_speed_in_6_minutes (total_chars : ℕ) (chars_first_minute : ℕ) (chars_last_minute : ℕ) (chars_other_minutes : ℕ) :
  total_chars = 2098 →
  chars_first_minute = 112 →
  chars_last_minute = 97 →
  chars_other_minutes = 1889 →
  (1889 / 6 : ℝ) < 315 → 
  ¬(∀ n, 1 ≤ n ∧ n ≤ 14 - 6 + 1 → chars_other_minutes / 6 ≥ 946) :=
by
  -- Given that analyzing the content, 
  -- proof is skipped here, replace this line with the actual proof.
  sorry

end typing_speed_in_6_minutes_l1901_190192


namespace simplify_expression_l1901_190189

variable (a b : ℤ)

theorem simplify_expression : (a - b) - (3 * (a + b)) - b = a - 8 * b := 
by sorry

end simplify_expression_l1901_190189


namespace car_distance_travelled_l1901_190125

theorem car_distance_travelled (time_hours : ℝ) (time_minutes : ℝ) (time_seconds : ℝ)
    (actual_speed : ℝ) (reduced_speed : ℝ) (distance : ℝ) :
    time_hours = 1 → 
    time_minutes = 40 →
    time_seconds = 48 →
    actual_speed = 34.99999999999999 → 
    reduced_speed = (5 / 7) * actual_speed → 
    distance = reduced_speed * ((time_hours + time_minutes / 60 + time_seconds / 3600) : ℝ) →
    distance = 42 := sorry

end car_distance_travelled_l1901_190125


namespace gcd_of_ropes_l1901_190130

theorem gcd_of_ropes : Nat.gcd (Nat.gcd 45 75) 90 = 15 := 
by
  sorry

end gcd_of_ropes_l1901_190130


namespace like_terms_exponents_l1901_190141

theorem like_terms_exponents (n m : ℕ) (h1 : n + 2 = 3) (h2 : 2 * m - 1 = 3) : n = 1 ∧ m = 2 :=
by sorry

end like_terms_exponents_l1901_190141


namespace det_matrix_A_l1901_190136

def matrix_A : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![8, 4], ![-2, 3]]

def determinant_2x2 (A : Matrix (Fin 2) (Fin 2) ℤ) : ℤ :=
  A 0 0 * A 1 1 - A 0 1 * A 1 0

theorem det_matrix_A : determinant_2x2 matrix_A = 32 := by
  sorry

end det_matrix_A_l1901_190136


namespace min_quality_inspection_machines_l1901_190197

theorem min_quality_inspection_machines (z x : ℕ) :
  (z + 30 * x) / 30 = 1 →
  (z + 10 * x) / 10 = 2 →
  (z + 5 * x) / 5 ≥ 4 :=
by
  intros h1 h2
  sorry

end min_quality_inspection_machines_l1901_190197


namespace Jan_is_6_inches_taller_than_Bill_l1901_190150

theorem Jan_is_6_inches_taller_than_Bill :
  ∀ (Cary Bill Jan : ℕ),
    Cary = 72 →
    Bill = Cary / 2 →
    Jan = 42 →
    Jan - Bill = 6 :=
by
  intros
  sorry

end Jan_is_6_inches_taller_than_Bill_l1901_190150


namespace sandy_carrots_l1901_190199

-- Definitions and conditions
def total_carrots : ℕ := 14
def mary_carrots : ℕ := 6

-- Proof statement
theorem sandy_carrots : (total_carrots - mary_carrots) = 8 :=
by
  -- sorry is used to bypass the actual proof steps
  sorry

end sandy_carrots_l1901_190199


namespace chicken_feathers_after_crossing_l1901_190118

def cars_dodged : ℕ := 23
def initial_feathers : ℕ := 5263
def feathers_lost : ℕ := 2 * cars_dodged
def final_feathers : ℕ := initial_feathers - feathers_lost

theorem chicken_feathers_after_crossing :
  final_feathers = 5217 := by
sorry

end chicken_feathers_after_crossing_l1901_190118


namespace point_in_fourth_quadrant_l1901_190113

-- Define the point (2, -3)
structure Point where
  x : ℤ
  y : ℤ

def A : Point := { x := 2, y := -3 }

-- Define what it means for a point to be in a specific quadrant
def inFirstQuadrant (P : Point) : Prop :=
  P.x > 0 ∧ P.y > 0

def inSecondQuadrant (P : Point) : Prop :=
  P.x < 0 ∧ P.y > 0

def inThirdQuadrant (P : Point) : Prop :=
  P.x < 0 ∧ P.y < 0

def inFourthQuadrant (P : Point) : Prop :=
  P.x > 0 ∧ P.y < 0

-- Define the theorem to prove that the point A lies in the fourth quadrant
theorem point_in_fourth_quadrant : inFourthQuadrant A :=
  sorry

end point_in_fourth_quadrant_l1901_190113


namespace choose_3_of_9_colors_l1901_190151

-- Define the combination function
noncomputable def combination (n k : ℕ) := n.choose k

-- Noncomputable because factorial and combination require division.
noncomputable def combination_9_3 := combination 9 3

-- State the theorem we are proving
theorem choose_3_of_9_colors : combination_9_3 = 84 :=
by
  -- Proof skipped
  sorry

end choose_3_of_9_colors_l1901_190151


namespace notebook_ratio_l1901_190185

theorem notebook_ratio (C N : ℕ) (h1 : ∀ k, N = k / C)
  (h2 : ∃ k, N = k / (C / 2) ∧ 16 = k / (C / 2))
  (h3 : C * N = 512) : (N : ℚ) / C = 1 / 8 := 
by
  sorry

end notebook_ratio_l1901_190185


namespace sufficient_but_not_necessary_condition_l1901_190177

theorem sufficient_but_not_necessary_condition (x y : ℝ) :
  (x^2 + y^2 ≤ 1) → ((x - 1)^2 + y^2 ≤ 4) ∧ ¬ ((x - 1)^2 + y^2 ≤ 4 → x^2 + y^2 ≤ 1) :=
by sorry

end sufficient_but_not_necessary_condition_l1901_190177


namespace largest_integer_remainder_l1901_190127

theorem largest_integer_remainder :
  ∃ (a : ℤ), a < 61 ∧ a % 6 = 5 ∧ ∀ b : ℤ, b < 61 ∧ b % 6 = 5 → b ≤ a :=
by
  sorry

end largest_integer_remainder_l1901_190127


namespace price_reduction_for_2100_yuan_price_reduction_for_max_profit_l1901_190164

-- Condition definitions based on the problem statement
def units_sold (x : ℝ) : ℝ := 30 + 2 * x
def profit_per_unit (x : ℝ) : ℝ := 50 - x
def daily_profit (x : ℝ) : ℝ := profit_per_unit x * units_sold x

-- Statement to prove the price reduction for achieving a daily profit of 2100 yuan
theorem price_reduction_for_2100_yuan : ∃ x : ℝ, daily_profit x = 2100 ∧ x = 20 :=
  sorry

-- Statement to prove the price reduction to maximize the daily profit
theorem price_reduction_for_max_profit : ∀ x : ℝ, ∃ y : ℝ, (∀ z : ℝ, daily_profit z ≤ y) ∧ x = 17.5 :=
  sorry

end price_reduction_for_2100_yuan_price_reduction_for_max_profit_l1901_190164


namespace expression_value_l1901_190142

theorem expression_value : (25 + 15)^2 - (25^2 + 15^2) = 750 := by
  sorry

end expression_value_l1901_190142


namespace remainder_of_8x_minus_5_l1901_190111

theorem remainder_of_8x_minus_5 (x : ℕ) (h : x % 15 = 7) : (8 * x - 5) % 15 = 6 :=
by
  sorry

end remainder_of_8x_minus_5_l1901_190111


namespace blue_beads_l1901_190188

-- Variables to denote the number of blue, red, white, and silver beads
variables (B R W S : ℕ)

-- Conditions derived from the problem statement
def conditions : Prop :=
  (R = 2 * B) ∧
  (W = B + R) ∧
  (S = 10) ∧
  (B + R + W + S = 40)

-- The theorem to prove
theorem blue_beads (B R W S : ℕ) (h : conditions B R W S) : B = 5 :=
by
  sorry

end blue_beads_l1901_190188


namespace students_in_school_at_least_225_l1901_190191

-- Conditions as definitions
def students_in_band := 85
def students_in_sports := 200
def students_in_both := 60
def students_in_either := 225

-- The proof statement
theorem students_in_school_at_least_225 :
  students_in_band + students_in_sports - students_in_both = students_in_either :=
by
  -- This statement will just assert the correctness as per given information in the problem
  sorry

end students_in_school_at_least_225_l1901_190191


namespace author_earnings_calculation_l1901_190116

open Real

namespace AuthorEarnings

def paperCoverCopies  : ℕ := 32000
def paperCoverPrice   : ℝ := 0.20
def paperCoverPercent : ℝ := 0.06

def hardCoverCopies   : ℕ := 15000
def hardCoverPrice    : ℝ := 0.40
def hardCoverPercent  : ℝ := 0.12

def total_earnings_paper_cover : ℝ := paperCoverCopies * paperCoverPrice
def earnings_paper_cover : ℝ := total_earnings_paper_cover * paperCoverPercent

def total_earnings_hard_cover : ℝ := hardCoverCopies * hardCoverPrice
def earnings_hard_cover : ℝ := total_earnings_hard_cover * hardCoverPercent

def author_total_earnings : ℝ := earnings_paper_cover + earnings_hard_cover

theorem author_earnings_calculation : author_total_earnings = 1104 := by
  sorry

end AuthorEarnings

end author_earnings_calculation_l1901_190116


namespace probability_all_girls_is_correct_l1901_190103

noncomputable def probability_all_girls : ℚ :=
  let total_members := 15
  let boys := 7
  let girls := 8
  let choose_3_from_15 := Nat.choose total_members 3
  let choose_3_from_8 := Nat.choose girls 3
  choose_3_from_8 / choose_3_from_15

theorem probability_all_girls_is_correct : 
  probability_all_girls = 8 / 65 := by
sorry

end probability_all_girls_is_correct_l1901_190103


namespace nat_solution_unique_l1901_190175

theorem nat_solution_unique (x y : ℕ) (h : x + y = x * y) : (x, y) = (2, 2) :=
sorry

end nat_solution_unique_l1901_190175


namespace playground_perimeter_l1901_190137

theorem playground_perimeter (x y : ℝ) 
  (h1 : x^2 + y^2 = 289) 
  (h2 : x * y = 120) : 
  2 * (x + y) = 46 :=
by 
  sorry

end playground_perimeter_l1901_190137


namespace find_age_of_15th_person_l1901_190140

-- Define the conditions given in the problem
def total_age_of_18_persons (avg_18 : ℕ) (num_18 : ℕ) : ℕ := avg_18 * num_18
def total_age_of_5_persons (avg_5 : ℕ) (num_5 : ℕ) : ℕ := avg_5 * num_5
def total_age_of_9_persons (avg_9 : ℕ) (num_9 : ℕ) : ℕ := avg_9 * num_9

-- Define the overall question which is the age of the 15th person
def age_of_15th_person (total_18 : ℕ) (total_5 : ℕ) (total_9 : ℕ) : ℕ :=
  total_18 - total_5 - total_9

-- Statement of the theorem to prove
theorem find_age_of_15th_person :
  let avg_18 := 15
  let num_18 := 18
  let avg_5 := 14
  let num_5 := 5
  let avg_9 := 16
  let num_9 := 9
  let total_18 := total_age_of_18_persons avg_18 num_18 
  let total_5 := total_age_of_5_persons avg_5 num_5
  let total_9 := total_age_of_9_persons avg_9 num_9
  age_of_15th_person total_18 total_5 total_9 = 56 :=
by
  -- Definitions for the total ages
  let avg_18 := 15
  let num_18 := 18
  let avg_5 := 14
  let num_5 := 5
  let avg_9 := 16
  let num_9 := 9
  let total_18 := total_age_of_18_persons avg_18 num_18 
  let total_5 := total_age_of_5_persons avg_5 num_5
  let total_9 := total_age_of_9_persons avg_9 num_9
  
  -- Goal: compute the age of the 15th person
  let answer := age_of_15th_person total_18 total_5 total_9

  -- Prove that the computed age is equal to 56
  show answer = 56
  sorry

end find_age_of_15th_person_l1901_190140


namespace line_tangent_to_parabola_l1901_190153

theorem line_tangent_to_parabola (k : ℝ) :
  (∀ (x y : ℝ), y^2 = 16 * x → 4 * x + 3 * y + k = 0) → k = 9 :=
by
  sorry

end line_tangent_to_parabola_l1901_190153


namespace compare_abc_l1901_190162

noncomputable def a : ℝ := (1 / 6) ^ (1 / 2)
noncomputable def b : ℝ := Real.log 1 / 3 / Real.log 6
noncomputable def c : ℝ := Real.log 1 / 7 / Real.log (1 / 6)

theorem compare_abc : c > a ∧ a > b := by
  sorry

end compare_abc_l1901_190162


namespace quadrants_cos_sin_identity_l1901_190146

theorem quadrants_cos_sin_identity (α : ℝ) 
  (h1 : π < α ∧ α < 2 * π)  -- α in the fourth quadrant
  (h2 : Real.cos α = 3 / 5) :
  (1 + Real.sqrt 2 * Real.cos (2 * α - π / 4)) / 
  (Real.sin (α + π / 2)) = -2 / 5 :=
by
  sorry

end quadrants_cos_sin_identity_l1901_190146


namespace not_prime_41_squared_plus_41_plus_41_l1901_190102

def is_prime (n : ℕ) : Prop := ∀ m k : ℕ, m * k = n → m = 1 ∨ k = 1

theorem not_prime_41_squared_plus_41_plus_41 :
  ¬ is_prime (41^2 + 41 + 41) :=
by {
  sorry
}

end not_prime_41_squared_plus_41_plus_41_l1901_190102


namespace problem_statement_l1901_190135

theorem problem_statement (a b c : ℤ) (h : c = b + 2) : 
  (a - (b + c)) - ((a + c) - b) = 0 :=
by
  sorry

end problem_statement_l1901_190135


namespace cost_of_materials_l1901_190121

theorem cost_of_materials (initial_bracelets given_away : ℕ) (sell_price profit : ℝ)
  (h1 : initial_bracelets = 52) 
  (h2 : given_away = 8) 
  (h3 : sell_price = 0.25) 
  (h4 : profit = 8) :
  let remaining_bracelets := initial_bracelets - given_away
  let total_revenue := remaining_bracelets * sell_price
  let cost_of_materials := total_revenue - profit
  cost_of_materials = 3 := 
by
  sorry

end cost_of_materials_l1901_190121


namespace mary_initial_flour_l1901_190138

theorem mary_initial_flour (F_total F_add F_initial : ℕ) 
  (h_total : F_total = 9)
  (h_add : F_add = 6)
  (h_initial : F_initial = F_total - F_add) :
  F_initial = 3 :=
sorry

end mary_initial_flour_l1901_190138


namespace impossible_to_convince_logical_jury_of_innocence_if_guilty_l1901_190105

theorem impossible_to_convince_logical_jury_of_innocence_if_guilty :
  (guilty : Prop) →
  (jury_is_logical : Prop) →
  guilty →
  (∀ statement : Prop, (logical_deduction : Prop) → (logical_deduction → ¬guilty)) →
  False :=
by
  intro guilty jury_is_logical guilty_premise logical_argument
  sorry

end impossible_to_convince_logical_jury_of_innocence_if_guilty_l1901_190105


namespace max_parabola_ratio_l1901_190109

noncomputable def parabola_max_ratio (x y : ℝ) : ℝ :=
  let O : ℝ × ℝ := (0, 0)
  let F : ℝ × ℝ := (1, 0)
  let M : ℝ × ℝ := (x, y)
  
  let MO : ℝ := Real.sqrt (x^2 + y^2)
  let MF : ℝ := Real.sqrt ((x - 1)^2 + y^2)
  
  MO / MF

theorem max_parabola_ratio :
  ∃ x y : ℝ, y^2 = 4 * x ∧ parabola_max_ratio x y = 2 * Real.sqrt 3 / 3 :=
sorry

end max_parabola_ratio_l1901_190109


namespace quadratic_inequality_solution_l1901_190139

theorem quadratic_inequality_solution : ∀ x : ℝ, -8 * x^2 + 4 * x - 7 ≤ 0 :=
by
  sorry

end quadratic_inequality_solution_l1901_190139


namespace transformed_equation_solutions_l1901_190119

theorem transformed_equation_solutions :
  (∀ x : ℝ, x^2 + 2 * x - 3 = 0 → (x = 1 ∨ x = -3)) →
  (∀ x : ℝ, (x + 3)^2 + 2 * (x + 3) - 3 = 0 → (x = -2 ∨ x = -6)) :=
by
  intro h
  sorry

end transformed_equation_solutions_l1901_190119


namespace distinct_real_roots_range_of_m_l1901_190100

theorem distinct_real_roots_range_of_m (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (x₁^2 + x₁ - m = 0) ∧ (x₂^2 + x₂ - m = 0)) → m > -1/4 := 
sorry

end distinct_real_roots_range_of_m_l1901_190100


namespace ellipse_properties_l1901_190169

theorem ellipse_properties :
  (∀ x y: ℝ, (x^2)/100 + (y^2)/36 = 1) →
  ∃ a b c e : ℝ, 
  a = 10 ∧ 
  b = 6 ∧ 
  c = 8 ∧ 
  2 * a = 20 ∧ 
  e = 4 / 5 :=
by
  intros
  sorry

end ellipse_properties_l1901_190169


namespace solve_for_x_y_l1901_190152

noncomputable def x_y_2018_sum (x y : ℝ) : ℝ := x^2018 + y^2018

theorem solve_for_x_y (A B : Set ℝ) (x y : ℝ)
  (hA : A = {x, x * y, x + y})
  (hB : B = {0, |x|, y}) 
  (h : A = B) :
  x_y_2018_sum x y = 2 := 
by
  sorry

end solve_for_x_y_l1901_190152


namespace num_possible_pairs_l1901_190110

theorem num_possible_pairs (a b : ℕ) (h1 : b > a) (h2 : (a - 8) * (b - 8) = 32) : 
    (∃ n, n = 3) :=
by { sorry }

end num_possible_pairs_l1901_190110


namespace sum_of_digits_l1901_190198

-- Conditions setup
variables (a b c d : ℕ)
variables (h1 : a + c = 10) 
variables (h2 : b + c = 9) 
variables (h3 : a + d = 10)
variables (distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)

theorem sum_of_digits : a + b + c + d = 19 :=
sorry

end sum_of_digits_l1901_190198


namespace arithmetic_sequence_sum_ratio_l1901_190129

theorem arithmetic_sequence_sum_ratio (a_n : ℕ → ℕ) (S : ℕ → ℕ) 
  (hS : ∀ n, S n = n * a_n 1 + n * (n - 1) / 2 * (a_n 2 - a_n 1)) 
  (h1 : S 6 / S 3 = 4) : S 9 / S 6 = 9 / 4 := 
by 
  sorry

end arithmetic_sequence_sum_ratio_l1901_190129


namespace min_sum_ab_l1901_190194

theorem min_sum_ab {a b : ℤ} (h : a * b = 36) : a + b ≥ -37 := sorry

end min_sum_ab_l1901_190194


namespace smallest_n_l1901_190144

/--
Each of \( 2020 \) boxes in a line contains 2 red marbles, 
and for \( 1 \le k \le 2020 \), the box in the \( k \)-th 
position also contains \( k \) white marbles. 

Let \( Q(n) \) be the probability that James stops after 
drawing exactly \( n \) marbles. Prove that the smallest 
value of \( n \) for which \( Q(n) < \frac{1}{2020} \) 
is 31.
-/
theorem smallest_n (Q : ℕ → ℚ) (hQ : ∀ n, Q n = (2 : ℚ) / ((n + 1) * (n + 2)))
  : ∃ n, Q n < 1/2020 ∧ ∀ m < n, Q m ≥ 1/2020 := by
  sorry

end smallest_n_l1901_190144


namespace books_taken_out_on_Tuesday_l1901_190186

theorem books_taken_out_on_Tuesday (T : ℕ) (initial_books : ℕ) (returned_books : ℕ) (withdrawn_books : ℕ) (final_books : ℕ) :
  initial_books = 250 ∧
  returned_books = 35 ∧
  withdrawn_books = 15 ∧
  final_books = 150 →
  T = 120 :=
by
  sorry

end books_taken_out_on_Tuesday_l1901_190186


namespace trigonometric_relationship_l1901_190124

noncomputable def α : ℝ := Real.cos 4
noncomputable def b : ℝ := Real.cos (4 * Real.pi / 5)
noncomputable def c : ℝ := Real.sin (7 * Real.pi / 6)

theorem trigonometric_relationship : b < α ∧ α < c := 
by
  sorry

end trigonometric_relationship_l1901_190124


namespace line_equation_passing_through_points_l1901_190112

theorem line_equation_passing_through_points 
  (a₁ b₁ a₂ b₂ : ℝ)
  (h1 : 2 * a₁ + 3 * b₁ + 1 = 0)
  (h2 : 2 * a₂ + 3 * b₂ + 1 = 0)
  (h3 : ∀ (x y : ℝ), (x, y) = (2, 3) → a₁ * x + b₁ * y + 1 = 0 ∧ a₂ * x + b₂ * y + 1 = 0) :
  (∀ (x y : ℝ), (2 * x + 3 * y + 1 = 0) ↔ 
                (a₁ = x ∧ b₁ = y) ∨ (a₂ = x ∧ b₂ = y)) :=
by
  sorry

end line_equation_passing_through_points_l1901_190112


namespace percentage_discount_proof_l1901_190171

noncomputable def ticket_price : ℝ := 25
noncomputable def price_to_pay : ℝ := 18.75
noncomputable def discount_amount : ℝ := ticket_price - price_to_pay
noncomputable def percentage_discount : ℝ := (discount_amount / ticket_price) * 100

theorem percentage_discount_proof : percentage_discount = 25 := by
  sorry

end percentage_discount_proof_l1901_190171


namespace third_smallest_triangular_square_l1901_190173

theorem third_smallest_triangular_square :
  ∃ n : ℕ, n = 1225 ∧ 
           (∃ x y : ℕ, y^2 - 8 * x^2 = 1 ∧ 
                        y = 99 ∧ x = 35) :=
by
  sorry

end third_smallest_triangular_square_l1901_190173


namespace apples_purchased_by_danny_l1901_190156

theorem apples_purchased_by_danny (pinky_apples : ℕ) (total_apples : ℕ) (danny_apples : ℕ) :
  pinky_apples = 36 → total_apples = 109 → danny_apples = total_apples - pinky_apples → danny_apples = 73 :=
by 
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end apples_purchased_by_danny_l1901_190156


namespace distinct_permutations_mathematics_l1901_190166

theorem distinct_permutations_mathematics : 
  let n := 11
  let freqM := 2
  let freqA := 2
  let freqT := 2
  (n.factorial / (freqM.factorial * freqA.factorial * freqT.factorial)) = 4989600 :=
by
  let n := 11
  let freqM := 2
  let freqA := 2
  let freqT := 2
  sorry

end distinct_permutations_mathematics_l1901_190166


namespace solve_quadratic_l1901_190196

theorem solve_quadratic (x : ℝ) : x^2 - 2*x = 0 ↔ (x = 0 ∨ x = 2) :=
by
  sorry

end solve_quadratic_l1901_190196


namespace combined_stripes_eq_22_l1901_190178

def stripes_olga_per_shoe : ℕ := 3
def shoes_per_person : ℕ := 2
def stripes_olga_total : ℕ := stripes_olga_per_shoe * shoes_per_person

def stripes_rick_per_shoe : ℕ := stripes_olga_per_shoe - 1
def stripes_rick_total : ℕ := stripes_rick_per_shoe * shoes_per_person

def stripes_hortense_per_shoe : ℕ := stripes_olga_per_shoe * 2
def stripes_hortense_total : ℕ := stripes_hortense_per_shoe * shoes_per_person

def total_stripes : ℕ := stripes_olga_total + stripes_rick_total + stripes_hortense_total

theorem combined_stripes_eq_22 : total_stripes = 22 := by
  sorry

end combined_stripes_eq_22_l1901_190178


namespace arithmetic_mean_of_p_and_q_l1901_190170

variable (p q r : ℝ)

theorem arithmetic_mean_of_p_and_q
  (h1 : (p + q) / 2 = 10)
  (h2 : (q + r) / 2 = 22)
  (h3 : r - p = 24) :
  (p + q) / 2 = 10 :=
by
  sorry

end arithmetic_mean_of_p_and_q_l1901_190170


namespace calvin_weeks_buying_chips_l1901_190120

variable (daily_spending : ℝ := 0.50)
variable (days_per_week : ℝ := 5)
variable (total_spending : ℝ := 10)
variable (spending_per_week := daily_spending * days_per_week)

theorem calvin_weeks_buying_chips :
  total_spending / spending_per_week = 4 := by
  sorry

end calvin_weeks_buying_chips_l1901_190120


namespace simplify_fraction_expression_l1901_190159

variable (d : ℝ)

theorem simplify_fraction_expression :
  (6 + 5 * d) / 11 + 3 = (39 + 5 * d) / 11 := 
by
  sorry

end simplify_fraction_expression_l1901_190159


namespace orange_profit_loss_l1901_190106

variable (C : ℝ) -- Cost price of one orange in rupees

-- Conditions as hypotheses
theorem orange_profit_loss :
  (1 / 16 - C) / C * 100 = 4 :=
by
  have h1 : 1.28 * C = 1 / 12 := sorry
  have h2 : C = 1 / (12 * 1.28) := sorry
  have h3 : C = 1 / 15.36 := sorry
  have h4 : (1/16 - C) = 1 / 384 := sorry
  -- Proof of main statement here
  sorry

end orange_profit_loss_l1901_190106


namespace find_A_l1901_190168

theorem find_A (A : ℝ) (h : (12 + 3) * (12 - A) = 120) : A = 4 :=
by sorry

end find_A_l1901_190168


namespace calculate_taxes_l1901_190143

def gross_pay : ℝ := 4500
def tax_rate_1 : ℝ := 0.10
def tax_rate_2 : ℝ := 0.15
def tax_rate_3 : ℝ := 0.20
def income_bracket_1 : ℝ := 1500
def income_bracket_2 : ℝ := 2000
def income_bracket_remaining : ℝ := gross_pay - income_bracket_1 - income_bracket_2
def standard_deduction : ℝ := 100

theorem calculate_taxes :
  let tax_1 := tax_rate_1 * income_bracket_1
  let tax_2 := tax_rate_2 * income_bracket_2
  let tax_3 := tax_rate_3 * income_bracket_remaining
  let total_tax := tax_1 + tax_2 + tax_3
  let tax_after_deduction := total_tax - standard_deduction
  tax_after_deduction = 550 :=
by
  sorry

end calculate_taxes_l1901_190143


namespace combined_salary_l1901_190122

theorem combined_salary (S_B : ℝ) (S_A : ℝ) (h1 : S_B = 8000) (h2 : 0.20 * S_A = 0.15 * S_B) : 
S_A + S_B = 14000 :=
by {
  sorry
}

end combined_salary_l1901_190122


namespace find_function_l1901_190183

-- Let f be a differentiable function over all real numbers
variable (f : ℝ → ℝ)
variable (k : ℝ)

-- Condition: f is differentiable over (-∞, ∞)
variable (h_diff : differentiable ℝ f)

-- Condition: f(0) = 1
variable (h_init : f 0 = 1)

-- Condition: for any x1, x2 in ℝ, f(x1 + x2) ≥ f(x1) f(x2)
variable (h_ineq : ∀ x1 x2 : ℝ, f (x1 + x2) ≥ f x1 * f x2)

-- We aim to prove: f(x) = e^(kx)
theorem find_function : ∃ k : ℝ, ∀ x : ℝ, f x = Real.exp (k * x) :=
sorry

end find_function_l1901_190183


namespace find_hyperbola_focus_l1901_190160

theorem find_hyperbola_focus : ∃ (x y : ℝ), 
  2 * x ^ 2 - 3 * y ^ 2 + 8 * x - 12 * y - 8 = 0 
  → (x, y) = (-2 + (Real.sqrt 30)/3, -2) :=
by
  sorry

end find_hyperbola_focus_l1901_190160


namespace other_solution_quadratic_l1901_190174

theorem other_solution_quadratic (h : (49 : ℚ) * (5 / 7)^2 - 88 * (5 / 7) + 40 = 0) : 
  ∃ x : ℚ, x ≠ 5 / 7 ∧ (49 * x^2 - 88 * x + 40 = 0) ∧ x = 8 / 7 :=
by
  sorry

end other_solution_quadratic_l1901_190174


namespace least_integer_square_l1901_190184

theorem least_integer_square (x : ℤ) : x^2 = 2 * x + 72 → x = -6 := 
by
  intro h
  sorry

end least_integer_square_l1901_190184


namespace sum_of_g1_l1901_190193

-- Define the main conditions
variable {g : ℝ → ℝ}
variable (h_nonconst : ∀ a b : ℝ, a ≠ b → g a ≠ g b)
axiom main_condition : ∀ x : ℝ, x ≠ 0 → g (x - 1) + g x + g (x + 1) = (g x) ^ 2 / (2025 * x)

-- Define the goal
theorem sum_of_g1 :
  g 1 = 6075 :=
sorry

end sum_of_g1_l1901_190193


namespace forgot_to_mow_l1901_190145

-- Definitions
def earning_per_lawn : ℕ := 9
def lawns_to_mow : ℕ := 12
def actual_earning : ℕ := 36

-- Statement to prove
theorem forgot_to_mow : (lawns_to_mow - (actual_earning / earning_per_lawn)) = 8 := by
  sorry

end forgot_to_mow_l1901_190145


namespace compound_interest_years_l1901_190117

-- Definitions for the given conditions
def principal : ℝ := 1200
def rate : ℝ := 0.20
def compound_interest : ℝ := 873.60
def compounded_yearly : ℝ := 1

-- Calculate the future value from principal and compound interest
def future_value : ℝ := principal + compound_interest

-- Statement of the problem: Prove that the number of years t was 3 given the conditions
theorem compound_interest_years :
  ∃ (t : ℝ), future_value = principal * (1 + rate / compounded_yearly)^(compounded_yearly * t) := sorry

end compound_interest_years_l1901_190117


namespace find_numbers_l1901_190163

theorem find_numbers (a b c d : ℕ)
  (h1 : a + b + c = 21)
  (h2 : a + b + d = 28)
  (h3 : a + c + d = 29)
  (h4 : b + c + d = 30) : 
  a = 6 ∧ b = 7 ∧ c = 8 ∧ d = 15 :=
sorry

end find_numbers_l1901_190163


namespace max_value_of_z_l1901_190157

variable (x y z : ℝ)

def condition1 : Prop := 2 * x + y ≤ 4
def condition2 : Prop := x ≤ y
def condition3 : Prop := x ≥ 1 / 2
def objective_function : ℝ := 2 * x - y

theorem max_value_of_z :
  (∀ x y, condition1 x y ∧ condition2 x y ∧ condition3 x → z = objective_function x y) →
  z ≤ 4 / 3 :=
sorry

end max_value_of_z_l1901_190157


namespace initial_percentage_alcohol_l1901_190108

-- Define the initial conditions
variables (P : ℚ) -- percentage of alcohol in the initial solution
variables (V1 V2 : ℚ) -- volumes of the initial solution and added alcohol
variables (C2 : ℚ) -- concentration of the resulting solution

-- Given the initial conditions and additional parameters
def initial_solution_volume : ℚ := 6
def added_alcohol_volume : ℚ := 1.8
def final_solution_volume : ℚ := initial_solution_volume + added_alcohol_volume
def final_solution_concentration : ℚ := 0.5 -- 50%

-- The amount of alcohol initially = (P / 100) * V1
-- New amount of alcohol after adding pure alcohol
-- This should equal to the final concentration of the new volume

theorem initial_percentage_alcohol : 
  (P / 100 * initial_solution_volume) + added_alcohol_volume = final_solution_concentration * final_solution_volume → 
  P = 35 :=
sorry

end initial_percentage_alcohol_l1901_190108


namespace num_of_friends_donated_same_l1901_190148

def total_clothing_donated_by_adam (pants jumpers pajama_sets t_shirts : ℕ) : ℕ :=
  pants + jumpers + 2 * pajama_sets + t_shirts

def clothing_kept_by_adam (initial_donation : ℕ) : ℕ :=
  initial_donation / 2

def clothing_donated_by_friends (total_donated keeping friends_donation : ℕ) : ℕ :=
  total_donated - keeping

def num_friends (friends_donation adam_initial_donation : ℕ) : ℕ :=
  friends_donation / adam_initial_donation

theorem num_of_friends_donated_same (pants jumpers pajama_sets t_shirts total_donated : ℕ)
  (initial_donation := total_clothing_donated_by_adam pants jumpers pajama_sets t_shirts)
  (keeping := clothing_kept_by_adam initial_donation)
  (friends_donation := clothing_donated_by_friends total_donated keeping initial_donation)
  (friends := num_friends friends_donation initial_donation)
  (hp : pants = 4)
  (hj : jumpers = 4)
  (hps : pajama_sets = 4)
  (ht : t_shirts = 20)
  (htotal : total_donated = 126) :
  friends = 3 :=
by
  sorry

end num_of_friends_donated_same_l1901_190148


namespace throwing_skips_l1901_190158

theorem throwing_skips :
  ∃ x y : ℕ, 
  y > x ∧ 
  (∃ z : ℕ, z = 2 * y ∧ 
  (∃ w : ℕ, w = z - 3 ∧ 
  (∃ u : ℕ, u = w + 1 ∧ u = 8))) ∧ 
  x + y + 2 * y + (2 * y - 3) + (2 * y - 2) = 33 ∧ 
  y - x = 2 :=
sorry

end throwing_skips_l1901_190158


namespace number_of_factors_of_x_l1901_190133

theorem number_of_factors_of_x (a b c : ℕ) (h1 : Nat.Prime a) (h2 : Nat.Prime b) (h3 : Nat.Prime c) (h4 : a < b) (h5 : b < c) (h6 : ¬ a = b) (h7 : ¬ b = c) (h8 : ¬ a = c) :
  let x := 2^2 * a^3 * b^2 * c^4
  let num_factors := (2 + 1) * (3 + 1) * (2 + 1) * (4 + 1)
  num_factors = 180 := by
sorry

end number_of_factors_of_x_l1901_190133


namespace ratio_of_costs_l1901_190167

-- Definitions based on conditions
def old_car_cost : ℕ := 1800
def new_car_cost : ℕ := 1800 + 2000

-- Theorem stating the desired proof
theorem ratio_of_costs :
  (new_car_cost / old_car_cost : ℚ) = 19 / 9 :=
by
  sorry

end ratio_of_costs_l1901_190167


namespace inequality_always_true_l1901_190195

theorem inequality_always_true (a b c d : ℝ) (h1 : a > b) (h2 : b > c) (h3 : c > d) : a + c > b + d :=
by sorry

end inequality_always_true_l1901_190195


namespace original_decimal_l1901_190182

theorem original_decimal (x : ℝ) (h : 1000 * x / 100 = 12.5) : x = 1.25 :=
by
  sorry

end original_decimal_l1901_190182


namespace women_with_fair_hair_percentage_l1901_190155

-- Define the conditions
variables {E : ℝ} (hE : E > 0)

def percent_factor : ℝ := 100

def employees_have_fair_hair (E : ℝ) : ℝ := 0.80 * E
def fair_hair_women (E : ℝ) : ℝ := 0.40 * (employees_have_fair_hair E)

-- Define the target proof statement
theorem women_with_fair_hair_percentage
  (h1 : E > 0)
  (h2 : employees_have_fair_hair E = 0.80 * E)
  (h3 : fair_hair_women E = 0.40 * (employees_have_fair_hair E)):
  (fair_hair_women E / E) * percent_factor = 32 := 
sorry

end women_with_fair_hair_percentage_l1901_190155


namespace alpha_less_than_60_degrees_l1901_190161

theorem alpha_less_than_60_degrees
  (R r : ℝ)
  (b c : ℝ)
  (α : ℝ)
  (h1 : b * c = 8 * R * r) :
  α < 60 := sorry

end alpha_less_than_60_degrees_l1901_190161


namespace range_of_m_l1901_190115

theorem range_of_m (m x : ℝ) (h : (x + m) / 3 - (2 * x - 1) / 2 = m) (hx : x ≤ 0) : m ≥ 3 / 4 := 
sorry

end range_of_m_l1901_190115


namespace original_price_of_computer_l1901_190134

theorem original_price_of_computer
  (P : ℝ)
  (h1 : 1.30 * P = 351)
  (h2 : 2 * P = 540) :
  P = 270 :=
by
  sorry

end original_price_of_computer_l1901_190134


namespace constant_k_for_linear_function_l1901_190123

theorem constant_k_for_linear_function (k : ℝ) (h : ∀ (x : ℝ), y = x^(k-1) + 2 → y = a * x + b) : k = 2 :=
sorry

end constant_k_for_linear_function_l1901_190123


namespace find_m_n_and_sqrt_l1901_190187

-- definitions based on conditions
def condition_1 (m : ℤ) : Prop := m + 3 = 1
def condition_2 (n : ℤ) : Prop := 2 * n - 12 = 64

-- the proof problem statement
theorem find_m_n_and_sqrt (m n : ℤ) (h1 : condition_1 m) (h2 : condition_2 n) : 
  m = -2 ∧ n = 38 ∧ Int.sqrt (m + n) = 6 := 
sorry

end find_m_n_and_sqrt_l1901_190187


namespace ella_distance_from_start_l1901_190131

noncomputable def compute_distance (m1 : ℝ) (f1 f2 m_to_f : ℝ) : ℝ :=
  let f1' := m1 * m_to_f
  let total_west := f1' + f2
  let distance_in_feet := Real.sqrt (f1^2 + total_west^2)
  distance_in_feet / m_to_f

theorem ella_distance_from_start :
  let starting_west := 10
  let first_north := 30
  let second_west := 40
  let meter_to_feet := 3.28084 
  compute_distance starting_west first_north second_west meter_to_feet = 24.01 := sorry

end ella_distance_from_start_l1901_190131


namespace number_of_ordered_pairs_l1901_190154

theorem number_of_ordered_pairs : 
  ∃ n, n = 325 ∧ ∀ (a b : ℤ), 
    1 ≤ a ∧ a ≤ 50 ∧ a % 2 = 1 ∧ 
    0 ≤ b ∧ b % 2 = 0 ∧ 
    ∃ r s : ℤ, r + s = -a ∧ r * s = b :=
sorry

end number_of_ordered_pairs_l1901_190154


namespace num_of_dogs_l1901_190179

theorem num_of_dogs (num_puppies : ℕ) (dog_food_per_meal : ℕ) (dog_meals_per_day : ℕ) (total_food : ℕ)
  (h1 : num_puppies = 4)
  (h2 : dog_food_per_meal = 4)
  (h3 : dog_meals_per_day = 3)
  (h4 : total_food = 108)
  : ∃ (D : ℕ), num_puppies * (dog_food_per_meal / 2) * (dog_meals_per_day * 3) + D * (dog_food_per_meal * dog_meals_per_day) = total_food ∧ D = 3 :=
by
  sorry

end num_of_dogs_l1901_190179


namespace number_of_possible_values_of_r_eq_894_l1901_190147

noncomputable def r_possible_values : ℕ :=
  let lower_bound := 0.3125
  let upper_bound := 0.4018
  let min_r := 3125  -- equivalent to the lowest four-digit decimal ≥ 0.3125
  let max_r := 4018  -- equivalent to the highest four-digit decimal ≤ 0.4018
  1 + max_r - min_r  -- total number of possible values

theorem number_of_possible_values_of_r_eq_894 :
  r_possible_values = 894 :=
by
  sorry

end number_of_possible_values_of_r_eq_894_l1901_190147


namespace a_3_eq_5_l1901_190132

variable (a : ℕ → ℕ) -- Defines the arithmetic sequence
variable (S : ℕ → ℕ) -- The sum of the first n terms of the sequence

-- Condition: S_5 = 25
axiom S_5_eq_25 : S 5 = 25

-- Define what it means for S to be the sum of the first n terms of the arithmetic sequence
axiom sum_arith_seq : ∀ n, S n = n * (a 1 + a n) / 2

theorem a_3_eq_5 : a 3 = 5 :=
by
  -- Proof is skipped using sorry
  sorry

end a_3_eq_5_l1901_190132


namespace horses_tiles_equation_l1901_190107

-- Conditions from the problem
def total_horses (x y : ℕ) : Prop := x + y = 100
def total_tiles (x y : ℕ) : Prop := 3 * x + (1 / 3 : ℚ) * y = 100

-- The statement to prove
theorem horses_tiles_equation (x y : ℕ) :
  total_horses x y ∧ total_tiles x y ↔ 
  (x + y = 100 ∧ (3 * x + (1 / 3 : ℚ) * y = 100)) :=
by
  sorry

end horses_tiles_equation_l1901_190107


namespace linear_dependent_iff_38_div_3_l1901_190104

theorem linear_dependent_iff_38_div_3 (k : ℚ) :
  k = 38 / 3 ↔ ∃ (α β γ : ℚ), α ≠ 0 ∨ β ≠ 0 ∨ γ ≠ 0 ∧
    α * 1 + β * 4 + γ * 7 = 0 ∧
    α * 2 + β * 5 + γ * 8 = 0 ∧
    α * 3 + β * k + γ * 9 = 0 :=
by
  sorry

end linear_dependent_iff_38_div_3_l1901_190104


namespace combined_6th_grade_percent_is_15_l1901_190180

-- Definitions
def annville_students := 100
def cleona_students := 200

def percent_6th_annville := 11
def percent_6th_cleona := 17

def total_students := annville_students + cleona_students
def total_6th_students := (percent_6th_annville * annville_students / 100) + (percent_6th_cleona * cleona_students / 100)

def percent_6th_combined := (total_6th_students * 100) / total_students

-- Theorem statement
theorem combined_6th_grade_percent_is_15 : percent_6th_combined = 15 :=
by
  sorry

end combined_6th_grade_percent_is_15_l1901_190180


namespace purely_imaginary_iff_real_iff_second_quadrant_iff_l1901_190126

def Z (m : ℝ) : ℂ := ⟨m^2 - 2 * m - 3, m^2 + 3 * m + 2⟩

theorem purely_imaginary_iff (m : ℝ) : (Z m).re = 0 ∧ (Z m).im ≠ 0 ↔ m = 3 :=
by sorry

theorem real_iff (m : ℝ) : (Z m).im = 0 ↔ m = -1 ∨ m = -2 :=
by sorry

theorem second_quadrant_iff (m : ℝ) : (Z m).re < 0 ∧ (Z m).im > 0 ↔ -1 < m ∧ m < 3 :=
by sorry

end purely_imaginary_iff_real_iff_second_quadrant_iff_l1901_190126


namespace max_light_window_l1901_190172

noncomputable def max_window_light : Prop :=
  ∃ (x : ℝ), (4 - 2 * x) / 3 * x = -2 / 3 * (x - 1) ^ 2 + 2 / 3 ∧ x = 1 ∧ (4 - 2 * x) / 3 = 2 / 3

theorem max_light_window : max_window_light :=
by
  sorry

end max_light_window_l1901_190172


namespace quotient_of_sum_l1901_190101

theorem quotient_of_sum (a b c x y z : ℝ)
  (h1 : a^2 + b^2 + c^2 = 25)
  (h2 : x^2 + y^2 + z^2 = 36)
  (h3 : a * x + b * y + c * z = 30) :
  (a + b + c) / (x + y + z) = 5 / 6 :=
by
  sorry

end quotient_of_sum_l1901_190101


namespace total_feet_in_garden_l1901_190128

theorem total_feet_in_garden (num_dogs num_ducks feet_per_dog feet_per_duck : ℕ)
  (h1 : num_dogs = 6) (h2 : num_ducks = 2)
  (h3 : feet_per_dog = 4) (h4 : feet_per_duck = 2) :
  num_dogs * feet_per_dog + num_ducks * feet_per_duck = 28 :=
by
  sorry

end total_feet_in_garden_l1901_190128


namespace count_integers_M_3_k_l1901_190165

theorem count_integers_M_3_k (M : ℕ) (hM : M < 500) :
  (∃ k : ℕ, k ≥ 1 ∧ ∃ m : ℕ, m ≥ 1 ∧ M = 2 * k * (m + k - 1)) ∧
  (∃ k1 k2 k3 k4 : ℕ, k1 ≠ k2 ∧ k1 ≠ k3 ∧ k1 ≠ k4 ∧
    k2 ≠ k3 ∧ k2 ≠ k4 ∧ k3 ≠ k4 ∧
    (M / 2 = (k1 + k2 + k3 + k4) ∨ M / 2 = (k1 * k2 * k3 * k4))) →
  (∃ n : ℕ, n = 6) :=
by
  sorry

end count_integers_M_3_k_l1901_190165


namespace eq_4_double_prime_l1901_190190

-- Define the function f such that f(q) = 3q - 3
def f (q : ℕ) : ℕ := 3 * q - 3

-- Theorem statement to show that f(f(4)) = 24
theorem eq_4_double_prime : f (f 4) = 24 := by
  sorry

end eq_4_double_prime_l1901_190190


namespace negation_of_forall_statement_l1901_190114

theorem negation_of_forall_statement :
  ¬ (∀ x : ℝ, x^2 + 2 * x ≥ 0) ↔ ∃ x : ℝ, x^2 + 2 * x < 0 := 
by
  sorry

end negation_of_forall_statement_l1901_190114


namespace bob_total_miles_l1901_190181

def total_miles_day1 (T : ℝ) := 0.20 * T
def remaining_miles_day1 (T : ℝ) := T - total_miles_day1 T
def total_miles_day2 (T : ℝ) := 0.50 * remaining_miles_day1 T
def remaining_miles_day2 (T : ℝ) := remaining_miles_day1 T - total_miles_day2 T
def total_miles_day3 (T : ℝ) := 28

theorem bob_total_miles (T : ℝ) (h : total_miles_day3 T = remaining_miles_day2 T) : T = 70 :=
by
  sorry

end bob_total_miles_l1901_190181


namespace prove_if_alpha_parallel_beta_and_a_perpendicular_beta_then_a_perpendicular_alpha_l1901_190176

-- Definitions of the entities involved
variables {L : Type} -- All lines
variables {P : Type} -- All planes

-- Relations
variables (perpendicular : L → P → Prop)
variables (parallel : P → P → Prop)

-- Conditions
variables (a b : L)
variables (α β : P)

-- Statements we want to prove
theorem prove_if_alpha_parallel_beta_and_a_perpendicular_beta_then_a_perpendicular_alpha
  (H1 : parallel α β) 
  (H2 : perpendicular a β) : 
  perpendicular a α :=
  sorry

end prove_if_alpha_parallel_beta_and_a_perpendicular_beta_then_a_perpendicular_alpha_l1901_190176
