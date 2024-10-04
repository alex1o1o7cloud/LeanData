import Mathlib

namespace no_seating_in_four_consecutive_seats_l74_74601

theorem no_seating_in_four_consecutive_seats :
  let total_arrangements := Nat.factorial 10
  let grouped_arrangements := Nat.factorial 7 * Nat.factorial 4
  let acceptable_arrangements := total_arrangements - grouped_arrangements
  acceptable_arrangements = 3507840 :=
by
  sorry

end no_seating_in_four_consecutive_seats_l74_74601


namespace arithmetic_sequence_term_difference_l74_74980

theorem arithmetic_sequence_term_difference :
  let a : ℕ := 3
  let d : ℕ := 6
  let t1 := a + 1499 * d
  let t2 := a + 1503 * d
  t2 - t1 = 24 :=
    by
    sorry

end arithmetic_sequence_term_difference_l74_74980


namespace odd_positive_integer_minus_twenty_l74_74505

theorem odd_positive_integer_minus_twenty (x : ℕ) (h : x = 53) : (2 * x - 1) - 20 = 85 := by
  subst h
  rfl

end odd_positive_integer_minus_twenty_l74_74505


namespace cookies_in_jar_l74_74652

noncomputable def C : ℕ := sorry

theorem cookies_in_jar (h : C - 1 = (C + 5) / 2) : C = 7 := by
  sorry

end cookies_in_jar_l74_74652


namespace peter_total_spent_l74_74212

/-
Peter bought a scooter for a certain sum of money. He spent 5% of the cost on the first round of repairs, another 10% on the second round of repairs, and 7% on the third round of repairs. After this, he had to pay a 12% tax on the original cost. Also, he offered a 15% holiday discount on the scooter's selling price. Despite the discount, he still managed to make a profit of $2000. How much did he spend in total, including repairs, tax, and discount if his profit percentage was 30%?
-/

noncomputable def total_spent (C S P : ℝ) : Prop :=
    (0.3 * C = P) ∧
    (0.85 * S = 1.34 * C + P) ∧
    (C = 2000 / 0.3) ∧
    (1.34 * C = 8933.33)

theorem peter_total_spent
  (C S P : ℝ)
  (h1 : 0.3 * C = P)
  (h2 : 0.85 * S = 1.34 * C + P)
  (h3 : C = 2000 / 0.3)
  : 1.34 * C = 8933.33 := by 
  sorry

end peter_total_spent_l74_74212


namespace sum_of_four_consecutive_integers_divisible_by_two_l74_74090

theorem sum_of_four_consecutive_integers_divisible_by_two (n : ℤ) : 
  2 ∣ ((n-1) + n + (n+1) + (n+2)) :=
by
  sorry

end sum_of_four_consecutive_integers_divisible_by_two_l74_74090


namespace brick_length_is_50_l74_74391

theorem brick_length_is_50
  (x : ℝ)
  (brick_volume_eq : x * 11.25 * 6 * 3200 = 800 * 600 * 22.5) :
  x = 50 :=
by
  sorry

end brick_length_is_50_l74_74391


namespace integral_result_l74_74558

noncomputable def integral_problem : ℝ :=
  ∫ x in (1:ℝ)..3, 2 * x - 1 / x^2

theorem integral_result : integral_problem = 8 := by
  sorry

end integral_result_l74_74558


namespace min_value_proof_l74_74791

noncomputable def min_value (t c : ℝ) :=
  (t^2 + c^2 - 2 * t * c + 2 * c^2) / 2

theorem min_value_proof (a b t c : ℝ) (h : a + b = t) :
  (a^2 + (b + c)^2) ≥ min_value t c :=
by
  sorry

end min_value_proof_l74_74791


namespace cricket_player_average_l74_74392

theorem cricket_player_average (A : ℕ)
  (H1 : 10 * A + 62 = 11 * (A + 4)) : A = 18 :=
by {
  sorry -- The proof itself
}

end cricket_player_average_l74_74392


namespace jason_initial_cards_l74_74332

theorem jason_initial_cards (cards_given_away cards_left : ℕ) (h1 : cards_given_away = 9) (h2 : cards_left = 4) :
  cards_given_away + cards_left = 13 :=
sorry

end jason_initial_cards_l74_74332


namespace problem_statement_l74_74204

theorem problem_statement (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_abc : abc = 1) :
  (1 / (a^3 * (b + c))) + (1 / (b^3 * (c + a))) + (1 / (c^3 * (a + b))) ≥ (3 / 2) := 
  sorry

end problem_statement_l74_74204


namespace total_weight_of_remaining_eggs_is_correct_l74_74471

-- Define the initial conditions and the question as Lean definitions
def total_eggs : Nat := 12
def weight_per_egg : Nat := 10
def num_boxes : Nat := 4
def melted_boxes : Nat := 1

-- Calculate the total weight of the eggs
def total_weight : Nat := total_eggs * weight_per_egg

-- Calculate the number of eggs per box
def eggs_per_box : Nat := total_eggs / num_boxes

-- Calculate the weight per box
def weight_per_box : Nat := eggs_per_box * weight_per_egg

-- Calculate the number of remaining boxes after one is tossed out
def remaining_boxes : Nat := num_boxes - melted_boxes

-- Calculate the total weight of the remaining chocolate eggs
def remaining_weight : Nat := remaining_boxes * weight_per_box

-- The proof task
theorem total_weight_of_remaining_eggs_is_correct : remaining_weight = 90 := by
  sorry

end total_weight_of_remaining_eggs_is_correct_l74_74471


namespace weight_loss_clothes_percentage_l74_74259

theorem weight_loss_clothes_percentage (W : ℝ) : 
  let initial_weight := W
  let weight_after_loss := 0.89 * initial_weight
  let final_weight_with_clothes := 0.9078 * initial_weight
  let added_weight_percentage := (final_weight_with_clothes / weight_after_loss - 1) * 100
  added_weight_percentage = 2 :=
by
  sorry

end weight_loss_clothes_percentage_l74_74259


namespace problem_statement_l74_74740
-- Broader import to bring in necessary library components.

-- Definition of the equation that needs to be satisfied by the points.
def satisfies_equation (x y : ℝ) : Prop := 2 * x^2 + y^2 + 3 * x * y + 3 * x + y = 2

-- Definitions of the two lines that form the solution set.
def line1 (x y : ℝ) : Prop := y = -x - 2
def line2 (x y : ℝ) : Prop := y = -2 * x + 1

-- Prove that the set of points that satisfy the given equation is the union of the two lines.
theorem problem_statement (x y : ℝ) : satisfies_equation x y ↔ line1 x y ∨ line2 x y :=
sorry

end problem_statement_l74_74740


namespace second_grade_students_sampled_l74_74270

-- Definitions corresponding to conditions in a)
def total_students := 2000
def mountain_climbing_fraction := 2 / 5
def running_ratios := (2, 3, 5)
def sample_size := 200

-- Calculation of total running participants based on ratio
def total_running_students :=
  total_students * (1 - mountain_climbing_fraction)

def a := 2 * (total_running_students / (2 + 3 + 5))
def b := 3 * (total_running_students / (2 + 3 + 5))
def c := 5 * (total_running_students / (2 + 3 + 5))

def running_sample_size := sample_size * (3 / 5) --since the ratio is 3:5

-- The statement to prove
theorem second_grade_students_sampled : running_sample_size * (3 / (2+3+5)) = 36 :=
by
  sorry

end second_grade_students_sampled_l74_74270


namespace diane_postage_problem_l74_74290

-- Definition of stamps
def stamps : List (ℕ × ℕ) := [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9)]

-- Define a function to compute the number of arrangements that sums to a target value
def arrangements_sum_to (target : ℕ) (stamps : List (ℕ × ℕ)) : ℕ :=
  sorry -- Implementation detail is skipped

-- The main theorem to prove
theorem diane_postage_problem :
  arrangements_sum_to 15 stamps = 271 :=
by sorry

end diane_postage_problem_l74_74290


namespace length_of_wall_l74_74316

-- Define the dimensions of a brick
def brick_length : ℝ := 40
def brick_width : ℝ := 11.25
def brick_height : ℝ := 6

-- Define the dimensions of the wall
def wall_height : ℝ := 600
def wall_width : ℝ := 22.5

-- Define the required number of bricks
def required_bricks : ℝ := 4000

-- Calculate the volume of a single brick
def volume_brick : ℝ := brick_length * brick_width * brick_height

-- Calculate the volume of the wall
def volume_wall (length : ℝ) : ℝ := length * wall_height * wall_width

-- The theorem to prove
theorem length_of_wall : ∃ (L : ℝ), required_bricks * volume_brick = volume_wall L → L = 800 :=
sorry

end length_of_wall_l74_74316


namespace intersection_A_B_range_of_a_l74_74425

-- Problem 1: Prove the intersection of A and B when a = 4
theorem intersection_A_B (a : ℝ) (h : a = 4) :
  { x : ℝ | 5 ≤ x ∧ x ≤ 7 } ∩ { x : ℝ | x ≤ 3 ∨ 5 < x} = {6, 7} :=
by sorry

-- Problem 2: Prove the range of values for a such that A ⊆ B
theorem range_of_a :
  { a : ℝ | (a < 2) ∨ (a > 4) } :=
by sorry

end intersection_A_B_range_of_a_l74_74425


namespace solve_equation_l74_74736

theorem solve_equation (x y : ℝ) (h : 2 * x^2 + y^2 + 3 * x * y + 3 * x + y = 2):
  y = -x - 2 ∨ y = -2 * x + 1 :=
sorry


end solve_equation_l74_74736


namespace jay_more_points_than_tobee_l74_74192

-- Declare variables.
variables (x J S : ℕ)

-- Given conditions
def Tobee_points := 4
def Jay_points := Tobee_points + x -- Jay_score is 4 + x
def Sean_points := (Tobee_points + Jay_points) - 2 -- Sean_score is 4 + Jay - 2

-- The total score condition
def total_score_condition := Tobee_points + Jay_points + Sean_points = 26

-- The main statement to be proven
theorem jay_more_points_than_tobee (h : total_score_condition) : J - Tobee_points = 6 :=
sorry

end jay_more_points_than_tobee_l74_74192


namespace expression_for_f_l74_74908

theorem expression_for_f {f : ℤ → ℤ} (h : ∀ x, f (x + 1) = 3 * x + 4) : ∀ x, f x = 3 * x + 1 :=
by
  sorry

end expression_for_f_l74_74908


namespace f_x_plus_1_l74_74174

-- Given function definition
def f (x : ℝ) := x^2

-- Statement to prove
theorem f_x_plus_1 (x : ℝ) : f (x + 1) = x^2 + 2 * x + 1 := 
by
  rw [f]
  -- This simplifies to:
  -- (x + 1)^2 = x^2 + 2 * x + 1
  sorry

end f_x_plus_1_l74_74174


namespace probability_of_sum_20_is_correct_l74_74962

noncomputable def probability_sum_20 : ℚ :=
  let total_outcomes := 12 * 12
  let favorable_outcomes := 5
  favorable_outcomes / total_outcomes

theorem probability_of_sum_20_is_correct :
  probability_sum_20 = 5 / 144 :=
by
  sorry

end probability_of_sum_20_is_correct_l74_74962


namespace problem_statement_l74_74742
-- Broader import to bring in necessary library components.

-- Definition of the equation that needs to be satisfied by the points.
def satisfies_equation (x y : ℝ) : Prop := 2 * x^2 + y^2 + 3 * x * y + 3 * x + y = 2

-- Definitions of the two lines that form the solution set.
def line1 (x y : ℝ) : Prop := y = -x - 2
def line2 (x y : ℝ) : Prop := y = -2 * x + 1

-- Prove that the set of points that satisfy the given equation is the union of the two lines.
theorem problem_statement (x y : ℝ) : satisfies_equation x y ↔ line1 x y ∨ line2 x y :=
sorry

end problem_statement_l74_74742


namespace incorrect_conclusions_l74_74422

theorem incorrect_conclusions
  (h1 : ∃ (y x : ℝ), (¬∃ a b : ℝ, a < 0 ∧ y = a * x + b) ∧ ∃ a b : ℝ, y = 2.347 * x - 6.423)
  (h2 : ∃ (y x : ℝ), (∃ a b : ℝ, a < 0 ∧ y = a * x + b) ∧ y = -3.476 * x + 5.648)
  (h3 : ∃ (y x : ℝ), (∃ a b : ℝ, a > 0 ∧ y = a * x + b) ∧ y = 5.437 * x + 8.493)
  (h4 : ∃ (y x : ℝ), (¬∃ a b : ℝ, a > 0 ∧ y = a * x + b) ∧ y = -4.326 * x - 4.578) :
  (∃ (y x : ℝ), y = 2.347 * x - 6.423 ∧ (¬∃ a b : ℝ, a < 0 ∧ y = a * x + b)) ∧
  (∃ (y x : ℝ), y = -4.326 * x - 4.578 ∧ (¬∃ a b : ℝ, a > 0 ∧ y = a * x + b)) :=
by {
  sorry
}

end incorrect_conclusions_l74_74422


namespace simplify_fraction_l74_74803

theorem simplify_fraction :
  (4^5 + 4^3) / (4^4 - 4^2) = 68 / 15 :=
by sorry

end simplify_fraction_l74_74803


namespace range_of_a_l74_74917

theorem range_of_a (a : ℝ) : (∀ x : ℝ, |x + 2| + |x - 1| ≥ a) ↔ a ≤ 3 := by
  sorry

end range_of_a_l74_74917


namespace space_left_each_side_l74_74532

theorem space_left_each_side (wall_width : ℕ) (picture_width : ℕ)
  (picture_centered : wall_width = 2 * ((wall_width - picture_width) / 2) + picture_width) :
  (wall_width - picture_width) / 2 = 9 :=
by
  have h : wall_width = 25 := sorry
  have h2 : picture_width = 7 := sorry
  exact sorry

end space_left_each_side_l74_74532


namespace total_rainfall_l74_74281

theorem total_rainfall
  (monday : ℝ)
  (tuesday : ℝ)
  (wednesday : ℝ)
  (h_monday : monday = 0.17)
  (h_tuesday : tuesday = 0.42)
  (h_wednesday : wednesday = 0.08) :
  monday + tuesday + wednesday = 0.67 :=
by
  sorry

end total_rainfall_l74_74281


namespace triangle_ineq_l74_74627

theorem triangle_ineq
  (a b c : ℝ)
  (triangle_sides : 0 < a ∧ 0 < b ∧ 0 < c)
  (triangle_ineq : a + b + c = 1) :
  a^2 + b^2 + c^2 + 4 * a * b * c < 1 / 2 :=
by
  sorry

end triangle_ineq_l74_74627


namespace calculate_expression_l74_74705

noncomputable def expr1 : ℝ := (Real.sqrt 2 + Real.sqrt 3) * (Real.sqrt 2 - Real.sqrt 3)
noncomputable def expr2 : ℝ := (2 * Real.sqrt 2 - 1) ^ 2
noncomputable def combined_expr : ℝ := expr1 + expr2

-- We need to prove the main statement
theorem calculate_expression : combined_expr = 8 - 4 * Real.sqrt 2 :=
by
  sorry

end calculate_expression_l74_74705


namespace calculate_expression_l74_74408

theorem calculate_expression : (Real.sqrt 4) + abs (3 - Real.pi) + (1 / 3)⁻¹ = 2 + Real.pi :=
by 
  sorry

end calculate_expression_l74_74408


namespace work_problem_l74_74267

theorem work_problem (A B C : ℝ) (hB : B = 3) (h1 : 1 / B + 1 / C = 1 / 2) (h2 : 1 / A + 1 / C = 1 / 2) : A = 3 := by
  sorry

end work_problem_l74_74267


namespace probability_double_head_l74_74499

/-- There are three types of coins:
  - Coin 1: one head and one tail
  - Coin 2: two heads
  - Coin 3: two tails
  A coin is randomly selected and flipped, resulting in heads. 
  Prove that the probability that the coin is Coin 2 (the double head coin)
  is 2/3. -/
theorem probability_double_head (h : true) : 
  let Coin1 : ℕ := 1,
      Coin2 : ℕ := 2,
      Coin3 : ℕ := 0,
      totalHeads : ℕ := Coin1 + Coin2 + Coin3
  in 
  let p_Coin1 := 1 / 3,
      p_Coin2 := 1 / 3,
      p_Coin3 := 1 / 3,
      p_Heads_given_Coin1 := 1 / 2,
      p_Heads_given_Coin2 := 1,
      p_Heads_given_Coin3 := 0,
      p_Heads := (p_Heads_given_Coin1 * p_Coin1) + (p_Heads_given_Coin2 * p_Coin2) + (p_Heads_given_Coin3 * p_Coin3)
  in (p_Heads_given_Coin2 * p_Coin2) / p_Heads = 2 / 3 :=
by
  sorry

end probability_double_head_l74_74499


namespace notebook_price_l74_74769

theorem notebook_price (students_buying_notebooks n c : ℕ) (total_students : ℕ := 36) (total_cost : ℕ := 990) :
  students_buying_notebooks > 18 ∧ c > n ∧ students_buying_notebooks * n * c = total_cost → c = 15 :=
by
  sorry

end notebook_price_l74_74769


namespace remainder_4x_div_9_l74_74987

theorem remainder_4x_div_9 (x : ℕ) (k : ℤ) (h : x = 9 * k + 5) : (4 * x) % 9 = 2 := 
by sorry

end remainder_4x_div_9_l74_74987


namespace diamond_expression_l74_74882

-- Define the diamond operation
def diamond (a b : ℚ) : ℚ := a - 1 / b

-- Declare the main theorem
theorem diamond_expression :
  (diamond (diamond 2 3) 4) - (diamond 2 (diamond 3 4)) = -29 / 132 := 
by
  sorry

end diamond_expression_l74_74882


namespace factorial_square_gt_power_l74_74215

theorem factorial_square_gt_power {n : ℕ} (h : n > 2) : (n! * n!) > n^n :=
sorry

end factorial_square_gt_power_l74_74215


namespace problem_solution_l74_74914

theorem problem_solution (a b c d e : ℤ) (h : (x - 3)^4 = ax^4 + bx^3 + cx^2 + dx + e) :
  b + c + d + e = 15 :=
by
  sorry

end problem_solution_l74_74914


namespace factorization_of_x_squared_minus_4_l74_74983

theorem factorization_of_x_squared_minus_4 (x : ℝ) : x^2 - 4 = (x - 2) * (x + 2) :=
by
  sorry

end factorization_of_x_squared_minus_4_l74_74983


namespace region_relation_l74_74990

theorem region_relation (A B C : ℝ)
  (a b c : ℝ) (h1 : a = 15) (h2 : b = 36) (h3 : c = 39)
  (h_triangle : a^2 + b^2 = c^2)
  (h_right_triangle : true) -- Since the triangle is already confirmed as right-angle
  (h_A : A = (π * (c / 2)^2 / 2 - 270) / 2)
  (h_B : B = (π * (c / 2)^2 / 2 - 270) / 2)
  (h_C : C = π * (c / 2)^2 / 2) :
  A + B + 270 = C :=
by
  sorry

end region_relation_l74_74990


namespace classics_books_l74_74226

theorem classics_books (authors : ℕ) (books_per_author : ℕ) (h_authors : authors = 6) (h_books_per_author : books_per_author = 33) :
  authors * books_per_author = 198 := 
by { rw [h_authors, h_books_per_author], norm_num }

end classics_books_l74_74226


namespace gcd_72_108_l74_74113

theorem gcd_72_108 : Nat.gcd 72 108 = 36 :=
by
  sorry

end gcd_72_108_l74_74113


namespace jane_not_finish_probability_l74_74964

theorem jane_not_finish_probability :
  (1 : ℚ) - (5 / 8) = (3 / 8) := by
  sorry

end jane_not_finish_probability_l74_74964


namespace find_pairs_l74_74201

theorem find_pairs (a b : ℕ) (h1: a > 0) (h2: b > 0) (q r : ℕ)
  (h3: a^2 + b^2 = q * (a + b) + r) (h4: q^2 + r = 1977) : 
  (a = 50 ∧ b = 37) ∨ (a = 37 ∧ b = 50) :=
sorry

end find_pairs_l74_74201


namespace total_weight_of_remaining_chocolate_eggs_l74_74469

-- Define the conditions as required:
def marie_makes_12_chocolate_eggs : ℕ := 12
def weight_of_each_egg : ℕ := 10
def number_of_boxes : ℕ := 4
def boxes_discarded : ℕ := 1

-- Define the proof statement
theorem total_weight_of_remaining_chocolate_eggs :
  let eggs_per_box := marie_makes_12_chocolate_eggs / number_of_boxes,
      remaining_boxes := number_of_boxes - boxes_discarded,
      remaining_eggs := remaining_boxes * eggs_per_box,
      total_weight := remaining_eggs * weight_of_each_egg
  in total_weight = 90 := by
  sorry

end total_weight_of_remaining_chocolate_eggs_l74_74469


namespace handshake_count_l74_74498

-- Defining the conditions
def number_of_companies : ℕ := 5
def representatives_per_company : ℕ := 5
def total_participants : ℕ := number_of_companies * representatives_per_company

-- Defining the number of handshakes each person makes
def handshakes_per_person : ℕ := total_participants - 1 - (representatives_per_company - 1)

-- Defining the total number of handshakes
def total_handshakes : ℕ := (total_participants * handshakes_per_person) / 2

theorem handshake_count :
  total_handshakes = 250 :=
by
  sorry

end handshake_count_l74_74498


namespace bookmarks_per_day_l74_74416

theorem bookmarks_per_day (pages_now : ℕ) (pages_end_march : ℕ) (days_in_march : ℕ) (pages_added : ℕ) (pages_per_day : ℕ)
  (h1 : pages_now = 400)
  (h2 : pages_end_march = 1330)
  (h3 : days_in_march = 31)
  (h4 : pages_added = pages_end_march - pages_now)
  (h5 : pages_per_day = pages_added / days_in_march) :
  pages_per_day = 30 := sorry

end bookmarks_per_day_l74_74416


namespace coin_toss_sequences_count_l74_74314

theorem coin_toss_sequences_count :
  ∃ (seq : List (List Bool)), 
    seq.length = 16 ∧
    (seq.count (== [tt, tt]) = 5) ∧
    (seq.count (== [tt, ff]) = 4) ∧
    (seq.count (== [ff, tt]) = 3) ∧
    (seq.count (== [ff, ff]) = 3) → 
    560 := 
by
  sorry

end coin_toss_sequences_count_l74_74314


namespace arithmetic_sequence_sum_l74_74613

theorem arithmetic_sequence_sum (a_n : ℕ → ℤ) (S_n : ℕ → ℤ) (m : ℕ) 
  (h1 : S_n m = 0) (h2 : S_n (m - 1) = -2) (h3 : S_n (m + 1) = 3) :
  m = 5 :=
sorry

end arithmetic_sequence_sum_l74_74613


namespace john_jury_duty_days_l74_74787

-- Definitions based on the given conditions
variable (jury_selection_days : ℕ) (trial_multiple : ℕ) (deliberation_days : ℕ) (hours_per_day_deliberation : ℕ)
variable (total_days_jury_duty : ℕ)

-- Conditions
def condition1 : Prop := jury_selection_days = 2
def condition2 : Prop := trial_multiple = 4
def condition3 : Prop := deliberation_days = 6
def condition4 : Prop := hours_per_day_deliberation = 16
def correct_answer : Prop := total_days_jury_duty = 19

-- Total days calculation
def total_days_calc : ℕ :=
  let trial_days := jury_selection_days * trial_multiple
  let total_deliberation_hours := deliberation_days * 24
  let actual_deliberation_days := total_deliberation_hours / hours_per_day_deliberation
  jury_selection_days + trial_days + actual_deliberation_days

-- Statement we need to prove
theorem john_jury_duty_days : condition1 ∧ condition2 ∧ condition3 ∧ condition4 → total_days_calc = total_days_jury_duty :=
by {
  intros h,
  cases h with h1 h_rest,
  cases h_rest with h2 h_rest1,
  cases h_rest1 with h3 h4,
  rw [condition1, condition2, condition3, condition4] at h1 h2 h3 h4,
  simp [h1, h2, h3, h4],
  sorry
} -- Proof omitted

end john_jury_duty_days_l74_74787


namespace age_relation_l74_74732

variable (x y z : ℕ)

theorem age_relation (h1 : x > y) : (z > y) ↔ (∃ w, w > 0 ∧ y + z > 2 * x) :=
sorry

end age_relation_l74_74732


namespace minimum_value_inequality_l74_74569

theorem minimum_value_inequality (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
    (h : x + 2 * y + 3 * z = 1) :
  (16 / x^3 + 81 / (8 * y^3) + 1 / (27 * z^3)) ≥ 1296 := sorry

end minimum_value_inequality_l74_74569


namespace max_single_player_salary_l74_74772

theorem max_single_player_salary (n : ℕ) (m : ℕ) (T : ℕ) (n_pos : n = 18) (m_pos : m = 20000) (T_pos : T = 800000) :
  ∃ x : ℕ, (∀ y : ℕ, y ≤ x → y ≤ 460000) ∧ (17 * m + x ≤ T) :=
by
  sorry

end max_single_player_salary_l74_74772


namespace pentagon_square_ratio_l74_74540

theorem pentagon_square_ratio (p s : ℕ) 
  (h1 : 5 * p = 20) (h2 : 4 * s = 20) : p / s = 4 / 5 :=
by sorry

end pentagon_square_ratio_l74_74540


namespace equation_solution_l74_74750

theorem equation_solution:
  ∀ (x y : ℝ), 
  2 * x^2 + y^2 + 3 * x * y + 3 * x + y = 2 ↔ 
  (y = -x - 2 ∨ y = -2 * x + 1) := 
by
  sorry

end equation_solution_l74_74750


namespace cookies_in_jar_l74_74658

noncomputable def number_of_cookies_in_jar : ℕ := sorry

theorem cookies_in_jar :
  (number_of_cookies_in_jar - 1) = (1 / 2 : ℝ) * (number_of_cookies_in_jar + 5) →
  number_of_cookies_in_jar = 7 :=
by
  sorry

end cookies_in_jar_l74_74658


namespace min_value_1abc_l74_74263

theorem min_value_1abc (a b c : ℕ) (h₁ : 0 ≤ a ∧ a ≤ 9) (h₂ : 0 ≤ b ∧ b ≤ 9) (h₃ : c = 0) 
    (h₄ : (1000 + 100 * a + 10 * b + c) % 2 = 0) 
    (h₅ : (1000 + 100 * a + 10 * b + c) % 3 = 0) 
    (h₆ : (1000 + 100 * a + 10 * b + c) % 5 = 0)
  : 1000 + 100 * a + 10 * b + c = 1020 :=
by
  sorry

end min_value_1abc_l74_74263


namespace sequence_term_position_l74_74364

theorem sequence_term_position (n : ℕ) (h : 2 * Real.sqrt 5 = Real.sqrt (3 * n - 1)) : n = 7 :=
sorry

end sequence_term_position_l74_74364


namespace sequence_is_arithmetic_l74_74571

-- Define a_n as a sequence in terms of n, where the formula is given.
def a_n (n : ℕ) : ℕ := 2 * n + 1

-- Theorem stating that the sequence is arithmetic with a common difference of 2.
theorem sequence_is_arithmetic : ∀ (n : ℕ), n > 0 → (a_n n) - (a_n (n - 1)) = 2 :=
by
  sorry

end sequence_is_arithmetic_l74_74571


namespace seven_pow_eight_mod_100_l74_74683

theorem seven_pow_eight_mod_100 :
  (7 ^ 8) % 100 = 1 := 
by {
  -- here can be the steps of the proof, but for now we use sorry
  sorry
}

end seven_pow_eight_mod_100_l74_74683


namespace eval_floor_neg_7_div_2_l74_74716

def floor (x : ℝ) : ℤ := 
  if h : ∃ z : ℤ, (z : ℝ) = x then h.some
  else ⌊x⌋

theorem eval_floor_neg_7_div_2 : floor (-7 / 2) = -4 := by
  -- Proof steps would go here
  sorry

end eval_floor_neg_7_div_2_l74_74716


namespace jack_jill_next_in_step_l74_74781

theorem jack_jill_next_in_step (stride_jack : ℕ) (stride_jill : ℕ) : 
  stride_jack = 64 → stride_jill = 56 → Nat.lcm stride_jack stride_jill = 448 :=
by
  intros h₁ h₂
  rw [h₁, h₂]
  sorry

end jack_jill_next_in_step_l74_74781


namespace greatest_integer_gcd_18_is_6_l74_74822

theorem greatest_integer_gcd_18_is_6 (n : ℕ) (h1 : n < 150) (h2 : Nat.gcd n 18 = 6) : n = 138 := 
sorry

end greatest_integer_gcd_18_is_6_l74_74822


namespace correct_statement_C_l74_74828

theorem correct_statement_C : ∀ y : ℝ, y^2 = 25 ↔ (y = 5 ∨ y = -5) := 
by sorry

end correct_statement_C_l74_74828


namespace hungarian_teams_probability_l74_74099

theorem hungarian_teams_probability :
  ∀ (total_teams hungarian_teams pairing_choices favorable_choices : ℕ)
  (all_pairings equally_likely : Prop),
  total_teams = 8 →
  hungarian_teams = 3 →
  pairing_choices = total_teams - 1 →
  favorable_choices = 5 * 4 * 3 →
  all_pairings = 7 * 5 * 3 ∧ equally_likely →
  (favorable_choices : ℚ) / (all_pairings : ℚ) = 4 / 7 := by
  intros total_teams hungarian_teams pairing_choices favorable_choices all_pairings equally_likely
  intros h_total h_hungarian h_pairing_choices h_favorable_choices h_all_eq
  rw [h_total, h_hungarian, h_pairing_choices, h_favorable_choices, h_all_eq.left]
  norm_num
  sorry

end hungarian_teams_probability_l74_74099


namespace ratio_of_sides_l74_74538

theorem ratio_of_sides (perimeter_pentagon perimeter_square : ℝ) (hp : perimeter_pentagon = 20) (hs : perimeter_square = 20) : (4:ℝ) / (5:ℝ) = (4:ℝ) / (5:ℝ) :=
by
  sorry

end ratio_of_sides_l74_74538


namespace fraction_of_acid_in_third_flask_l74_74817

def mass_of_acid_first_flask := 10
def mass_of_acid_second_flask := 20
def mass_of_acid_third_flask := 30
def mass_of_acid_first_flask_with_water (w : ℝ) := mass_of_acid_first_flask / (mass_of_acid_first_flask + w) = 1 / 20
def mass_of_acid_second_flask_with_water (W w : ℝ) := mass_of_acid_second_flask / (mass_of_acid_second_flask + (W - w)) = 7 / 30
def mass_of_acid_third_flask_with_water (W : ℝ) := mass_of_acid_third_flask / (mass_of_acid_third_flask + W)

theorem fraction_of_acid_in_third_flask (W w : ℝ) (h1 : mass_of_acid_first_flask_with_water w) (h2 : mass_of_acid_second_flask_with_water W w) :
  mass_of_acid_third_flask_with_water W = 21 / 200 :=
by
  sorry

end fraction_of_acid_in_third_flask_l74_74817


namespace washing_time_is_45_l74_74880

-- Definitions based on conditions
variables (x : ℕ) -- time to wash one load
axiom h1 : 2 * x + 75 = 165 -- total laundry time equation

-- The statement to prove: washing one load takes 45 minutes
theorem washing_time_is_45 : x = 45 :=
by
  sorry

end washing_time_is_45_l74_74880


namespace smallest_positive_even_integer_l74_74563

noncomputable def smallest_even_integer (n : ℕ) : ℕ := 
  if 2 * n > 0 ∧ (3^(n * (n + 1) / 8)) > 500 then n else 0

theorem smallest_positive_even_integer :
  smallest_even_integer 6 = 6 :=
by
  -- Skipping the proofs
  sorry

end smallest_positive_even_integer_l74_74563


namespace calculate_k_l74_74020

variable (A B C D k : ℕ)

def workers_time : Prop :=
  (1 / (A : ℚ) + 1 / (B : ℚ) + 1 / (C : ℚ) + 1 / (D : ℚ)) = 
  (1 / (A - 8 : ℚ)) ∧
  (1 / (A : ℚ) + 1 / (B : ℚ) + 1 / (C : ℚ) + 1 / (D : ℚ)) = 
  (1 / (B - 2 : ℚ)) ∧
  (1 / (A : ℚ) + 1 / (B : ℚ) + 1 / (C : ℚ) + 1 / (D : ℚ)) = 
  (3 / (C : ℚ))

theorem calculate_k (h : workers_time A B C D) : k = 16 :=
  sorry

end calculate_k_l74_74020


namespace solve_equation_l74_74737

theorem solve_equation (x y : ℝ) (h : 2 * x^2 + y^2 + 3 * x * y + 3 * x + y = 2):
  y = -x - 2 ∨ y = -2 * x + 1 :=
sorry


end solve_equation_l74_74737


namespace find_angle_KML_l74_74235

noncomputable def angle_KML (r : ℝ) (S_OKL : ℝ) (S_KLM : ℝ) : ℝ :=
  if r = 5 ∧ S_OKL = 12 ∧ S_KLM > 30 then
    real.arccos (3/5)
  else
    0 -- default value if conditions are not met

-- Formal statement in Lean 4
theorem find_angle_KML :
  ∀ (O K L M : Type) (radius : ℝ)
  (area_OKL : ℝ) (area_KLM : ℝ),
  (radius = 5) →
  (area_OKL = 12) →
  (area_KLM > 30) →
  angle_KML radius area_OKL area_KLM = real.arccos (3 / 5) :=
begin
  intros O K L M r S_OKL S_KLM hr hS_OKL hS_KLM,
  simp [angle_KML],
  split_ifs,
  { exact rfl },
  { sorry }, -- default value case skipped
end

end find_angle_KML_l74_74235


namespace average_rainfall_per_hour_eq_l74_74768

-- Define the conditions
def february_days_non_leap_year : ℕ := 28
def hours_per_day : ℕ := 24
def total_rainfall_in_inches : ℕ := 280
def total_hours_in_february : ℕ := february_days_non_leap_year * hours_per_day

-- Define the goal
theorem average_rainfall_per_hour_eq :
  total_rainfall_in_inches / total_hours_in_february = 5 / 12 :=
sorry

end average_rainfall_per_hour_eq_l74_74768


namespace evaluate_expression_l74_74156

theorem evaluate_expression : 
  (3 + Real.sqrt 3 + 1 / (3 + Real.sqrt 3) + 1 / (Real.sqrt 3 - 3) = 3 + 2 * Real.sqrt 3 / 3) :=
by
  sorry

end evaluate_expression_l74_74156


namespace cube_root_of_27_eq_3_l74_74076

theorem cube_root_of_27_eq_3 : real.cbrt 27 = 3 :=
by {
  have h : 27 = 3 ^ 3 := by norm_num,
  rw real.cbrt_eq_iff_pow_eq (by norm_num : 0 ≤ 27) h,
  norm_num,
  sorry
}

end cube_root_of_27_eq_3_l74_74076


namespace total_weight_of_remaining_chocolate_eggs_l74_74470

-- Define the conditions as required:
def marie_makes_12_chocolate_eggs : ℕ := 12
def weight_of_each_egg : ℕ := 10
def number_of_boxes : ℕ := 4
def boxes_discarded : ℕ := 1

-- Define the proof statement
theorem total_weight_of_remaining_chocolate_eggs :
  let eggs_per_box := marie_makes_12_chocolate_eggs / number_of_boxes,
      remaining_boxes := number_of_boxes - boxes_discarded,
      remaining_eggs := remaining_boxes * eggs_per_box,
      total_weight := remaining_eggs * weight_of_each_egg
  in total_weight = 90 := by
  sorry

end total_weight_of_remaining_chocolate_eggs_l74_74470


namespace desk_chair_production_l74_74848

theorem desk_chair_production (x : ℝ) (h₁ : x > 0) (h₂ : 540 / x - 540 / (x + 2) = 3) : 
  ∃ x, 540 / x - 540 / (x + 2) = 3 := 
by
  sorry

end desk_chair_production_l74_74848


namespace only_integer_solution_l74_74354

theorem only_integer_solution (a b c d : ℤ) (h : a^2 + b^2 = 3 * (c^2 + d^2)) : 
  a = 0 ∧ b = 0 ∧ c = 0 ∧ d = 0 := 
by
  sorry

end only_integer_solution_l74_74354


namespace slope_ratio_l74_74821

theorem slope_ratio (s t k b : ℝ) 
  (h1: b = -12 * s)
  (h2: b = k - 7) 
  (ht: t = (7 - k) / 7) 
  (hs: s = (7 - k) / 12): 
  s / t = 7 / 12 := 
  sorry

end slope_ratio_l74_74821


namespace fraction_sum_59_l74_74229

theorem fraction_sum_59 :
  ∃ (a b : ℕ), (0.84375 = (a : ℚ) / b) ∧ (Nat.gcd a b = 1) ∧ (a + b = 59) :=
sorry

end fraction_sum_59_l74_74229


namespace missing_digit_B_l74_74660

theorem missing_digit_B :
  ∃ B : ℕ, 0 ≤ B ∧ B ≤ 9 ∧ (200 + 10 * B + 5) % 13 = 0 := 
sorry

end missing_digit_B_l74_74660


namespace marathons_yards_l74_74695

theorem marathons_yards
  (miles_per_marathon : ℕ)
  (yards_per_marathon : ℕ)
  (miles_in_yard : ℕ)
  (marathons_run : ℕ)
  (total_miles : ℕ)
  (total_yards : ℕ)
  (y : ℕ) :
  miles_per_marathon = 30
  → yards_per_marathon = 520
  → miles_in_yard = 1760
  → marathons_run = 8
  → total_miles = (miles_per_marathon * marathons_run) + (yards_per_marathon * marathons_run) / miles_in_yard
  → total_yards = (yards_per_marathon * marathons_run) % miles_in_yard
  → y = 640 := 
by
  intros
  sorry

end marathons_yards_l74_74695


namespace problem_statement_l74_74167

-- Define the constants and variables
variables (x y z a b c : ℝ)

-- Define the conditions given in the problem
def condition1 : Prop := x / a + y / b + z / c = 4
def condition2 : Prop := a / x + b / y + c / z = 1

-- State the theorem that proves the question equals the correct answer
theorem problem_statement (h1 : condition1 x y z a b c) (h2 : condition2 x y z a b c) :
    x^2 / a^2 + y^2 / b^2 + z^2 / c^2 = 12 :=
sorry

end problem_statement_l74_74167


namespace digit_x_for_divisibility_by_29_l74_74809

-- Define the base 7 number 34x1_7 in decimal form
def base7_to_decimal (x : ℕ) : ℕ := 3 * 7^3 + 4 * 7^2 + x * 7 + 1

-- State the proof problem
theorem digit_x_for_divisibility_by_29 (x : ℕ) (h : base7_to_decimal x % 29 = 0) : x = 3 :=
by
  sorry

end digit_x_for_divisibility_by_29_l74_74809


namespace range_of_AB_l74_74187

variable (AB BC AC : ℝ)
variable (θ : ℝ)
variable (B : ℝ)

-- Conditions
axiom angle_condition : θ = 150
axiom length_condition : AC = 2

-- Theorem to prove
theorem range_of_AB (h_θ : θ = 150) (h_AC : AC = 2) : (0 < AB) ∧ (AB ≤ 4) :=
sorry

end range_of_AB_l74_74187


namespace remainder_of_sum_of_5_consecutive_numbers_mod_9_l74_74010

theorem remainder_of_sum_of_5_consecutive_numbers_mod_9 :
  (9154 + 9155 + 9156 + 9157 + 9158) % 9 = 1 :=
by
  sorry

end remainder_of_sum_of_5_consecutive_numbers_mod_9_l74_74010


namespace cookies_in_jar_l74_74659

noncomputable def number_of_cookies_in_jar : ℕ := sorry

theorem cookies_in_jar :
  (number_of_cookies_in_jar - 1) = (1 / 2 : ℝ) * (number_of_cookies_in_jar + 5) →
  number_of_cookies_in_jar = 7 :=
by
  sorry

end cookies_in_jar_l74_74659


namespace bradley_travel_time_l74_74282

theorem bradley_travel_time (T : ℕ) (h1 : T / 4 = 20) (h2 : T / 3 = 45) : T - 20 = 280 :=
by
  -- Placeholder for proof
  sorry

end bradley_travel_time_l74_74282


namespace triangle_angle_ABC_l74_74773

theorem triangle_angle_ABC
  (ABD CBD ABC : ℝ) 
  (h1 : ABD = 70)
  (h2 : ABD + CBD + ABC = 200)
  (h3 : CBD = 60) : ABC = 70 := 
sorry

end triangle_angle_ABC_l74_74773


namespace arithmetic_mean_reciprocals_first_four_primes_l74_74875

theorem arithmetic_mean_reciprocals_first_four_primes :
  (1/2 + 1/3 + 1/5 + 1/7) / 4 = 247 / 840 :=
by
  sorry

end arithmetic_mean_reciprocals_first_four_primes_l74_74875


namespace alpha_beta_store_ways_l74_74863

open Finset

theorem alpha_beta_store_ways:
  let oreo_flavors := 5
  let milk_flavors := 3
  let total_products := oreo_flavors + milk_flavors
  let alpha_choices (alpha_items: finset (fin total_products)) := alpha_items.card ≤ 3
  let beta_choices (beta_items: multiset (fin oreo_flavors)) := beta_items.card ≤ 3
  let total_ways := 
    (card (powerset_len 3 (range total_products))) + 
    (card (powerset_len 2 (range total_products)) * card (multiset.range oreo_flavors)) + 
    (card (powerset_len 1 (range total_products)) * (card (powerset_len 2 (range oreo_flavors)) + oreo_flavors)) + 
    (card (multiset.powerset_len 3 (range oreo_flavors)))
  in total_ways = 351 :=
by
  let oreo_flavors := 5
  let milk_flavors := 3
  let total_products := oreo_flavors + milk_flavors
  
  -- Define choices for Alpha and Beta, and their respective constraints
  let alpha_choices (alpha_items : finset (fin total_products)) := alpha_items.card ≤ 3
  let beta_choices (beta_items : multiset (fin oreo_flavors)) := beta_items.card ≤ 3
  
  -- Calculate total ways
  let total_ways :=
    (card (powerset_len 3 (range total_products))) + 
    (card (powerset_len 2 (range total_products)) * card (multiset.range oreo_flavors)) + 
    (card (powerset_len 1 (range total_products)) * (card (powerset_len 2 (range oreo_flavors)) + oreo_flavors)) + 
    (card (multiset.powerset_len 3 (range oreo_flavors)))
  
  -- Assert the final number of ways
  show total_ways = 351, from sorry

end alpha_beta_store_ways_l74_74863


namespace arithmetic_mean_reciprocals_primes_l74_74872

theorem arithmetic_mean_reciprocals_primes
  (p : Finset ℕ)
  (h_p : p = {2, 3, 5, 7})
  : (p.sum (λ x, 1 / ↑x) / 4) = (247 / 840) :=
by
  sorry

end arithmetic_mean_reciprocals_primes_l74_74872


namespace fishing_probability_correct_l74_74369

-- Definitions for probabilities
def P_sunny : ℝ := 0.3
def P_rainy : ℝ := 0.5
def P_cloudy : ℝ := 0.2

def P_fishing_given_sunny : ℝ := 0.7
def P_fishing_given_rainy : ℝ := 0.3
def P_fishing_given_cloudy : ℝ := 0.5

-- The total probability function
def P_fishing : ℝ :=
  P_sunny * P_fishing_given_sunny +
  P_rainy * P_fishing_given_rainy +
  P_cloudy * P_fishing_given_cloudy

theorem fishing_probability_correct : P_fishing = 0.46 :=
by 
  sorry -- Proof goes here

end fishing_probability_correct_l74_74369


namespace sin_alpha_beta_l74_74576

theorem sin_alpha_beta (a b c α β : ℝ) (h1 : 0 < α ∧ α < π) (h2 : 0 < β ∧ β < π)
    (h3 : a * Real.cos α + b * Real.sin α + c = 0) (h4 : a * Real.cos β + b * Real.sin β + c = 0) 
    (h5 : α ≠ β) : Real.sin (α + β) = (2 * a * b) / (a ^ 2 + b ^ 2) := 
sorry

end sin_alpha_beta_l74_74576


namespace smallest_common_multiple_l74_74507

theorem smallest_common_multiple (n : ℕ) (h1 : n > 0) (h2 : 8 ∣ n) (h3 : 6 ∣ n) : n = 24 :=
by sorry

end smallest_common_multiple_l74_74507


namespace twenty_percent_less_than_sixty_equals_one_third_more_than_what_number_l74_74663

theorem twenty_percent_less_than_sixty_equals_one_third_more_than_what_number :
  (4 / 3) * n = 48 → n = 36 :=
by
  intro h
  sorry

end twenty_percent_less_than_sixty_equals_one_third_more_than_what_number_l74_74663


namespace system1_solution_system2_solution_l74_74956

-- Statement for the Part 1 Equivalent Problem.
theorem system1_solution :
  ∀ (x y : ℤ),
    (x - 3 * y = -10) ∧ (x + y = 6) → (x = 2 ∧ y = 4) :=
by
  intros x y h
  rcases h with ⟨h1, h2⟩
  sorry

-- Statement for the Part 2 Equivalent Problem.
theorem system2_solution :
  ∀ (x y : ℚ),
    (x / 2 - (y - 1) / 3 = 1) ∧ (4 * x - y = 8) → (x = 12 / 5 ∧ y = 8 / 5) :=
by
  intros x y h
  rcases h with ⟨h1, h2⟩
  sorry

end system1_solution_system2_solution_l74_74956


namespace incorrect_equation_a_neq_b_l74_74372

theorem incorrect_equation_a_neq_b (a b : ℝ) (h : a ≠ b) : a - b ≠ b - a :=
  sorry

end incorrect_equation_a_neq_b_l74_74372


namespace units_digit_of_150_factorial_is_zero_l74_74248

-- Define the conditions for the problem
def is_units_digit_zero_of_factorial (n : ℕ) : Prop :=
  n = 150 → (Nat.factorial n % 10 = 0)

-- The statement of the proof problem
theorem units_digit_of_150_factorial_is_zero : is_units_digit_zero_of_factorial 150 :=
  sorry

end units_digit_of_150_factorial_is_zero_l74_74248


namespace rhombus_area_3cm_45deg_l74_74288

noncomputable def rhombusArea (a : ℝ) (theta : ℝ) : ℝ :=
  a * (a * Real.sin theta)

theorem rhombus_area_3cm_45deg :
  rhombusArea 3 (Real.pi / 4) = 9 * Real.sqrt 2 / 2 := 
by
  sorry

end rhombus_area_3cm_45deg_l74_74288


namespace sum_of_angles_eq_62_l74_74016

noncomputable def Φ (x : ℝ) : ℝ := Real.sin x
noncomputable def Ψ (x : ℝ) : ℝ := Real.cos x
def θ : List ℝ := [31, 30, 1, 0]

theorem sum_of_angles_eq_62 :
  θ.sum = 62 := by
  sorry

end sum_of_angles_eq_62_l74_74016


namespace expand_binomials_l74_74159

theorem expand_binomials (x : ℝ) : (3 * x + 4) * (2 * x + 7) = 6 * x^2 + 29 * x + 28 := 
by 
  sorry

end expand_binomials_l74_74159


namespace sqrt_14400_eq_120_l74_74352

theorem sqrt_14400_eq_120 : Real.sqrt 14400 = 120 :=
by
  sorry

end sqrt_14400_eq_120_l74_74352


namespace females_on_police_force_l74_74986

theorem females_on_police_force (H : ∀ (total_female_officers total_officers_on_duty female_officers_on_duty : ℕ), 
  total_officers_on_duty = 500 ∧ female_officers_on_duty = total_officers_on_duty / 2 ∧ female_officers_on_duty = total_female_officers / 4) :
  ∃ total_female_officers : ℕ, total_female_officers = 1000 := 
by {
  sorry
}

end females_on_police_force_l74_74986


namespace tangent_length_and_equations_l74_74577

theorem tangent_length_and_equations (P : ℝ × ℝ) (C : ℝ × ℝ) (r : ℝ) (x y : ℝ) :
  P = (2, 3) → C = (1, 1) → r = 1 →
  ((x - 1)^2 + (y - 1)^2 = 1) →
  (∃ L : ℝ, L = 2) ∧ 
  (∃ eq1 eq2 : (ℝ × ℝ) → Prop, eq1 = λ p, 3 * p.1 - 4 * p.2 + 6 = 0 ∧ eq2 = λ p, p.1 = 2) :=
begin
  intros,
  split,
  { use 2,
    -- Proof that the length of the tangent line is 2, skipped
    sorry },
  { use (λ p, 3 * p.1 - 4 * p.2 + 6 = 0),
    use (λ p, p.1 = 2),
    -- Proof that these are the equations of the tangent lines, skipped
    sorry }
end

end tangent_length_and_equations_l74_74577


namespace necessary_and_sufficient_condition_l74_74168

theorem necessary_and_sufficient_condition (a b : ℕ) (ha : a > 0) (hb : b > 0) :
  a + b > a * b ↔ (a = 1 ∨ b = 1) :=
sorry

end necessary_and_sufficient_condition_l74_74168


namespace length_of_first_video_l74_74937

theorem length_of_first_video
  (total_time : ℕ)
  (second_video_time : ℕ)
  (last_two_videos_time : ℕ)
  (first_video_time : ℕ)
  (total_seconds : total_time = 510)
  (second_seconds : second_video_time = 4 * 60 + 30)
  (last_videos_seconds : last_two_videos_time = 60 + 60)
  (total_watch_time : total_time = second_video_time + last_two_videos_time + first_video_time) :
  first_video_time = 120 :=
by
  sorry

end length_of_first_video_l74_74937


namespace captain_age_eq_your_age_l74_74318

-- Represent the conditions as assumptions
variables (your_age : ℕ) -- You, the captain, have an age as a natural number

-- Define the statement
theorem captain_age_eq_your_age (H_cap : ∀ captain, captain = your_age) : ∀ captain, captain = your_age := by
  sorry

end captain_age_eq_your_age_l74_74318


namespace solve_for_x_l74_74719

theorem solve_for_x (x : ℝ) (h1 : x ≠ -3) (h2 : (2 / (x + 3) + 3 * x / (x + 3) - 5 / (x + 3) = 5)) : x = -9 :=
by
  sorry

end solve_for_x_l74_74719


namespace find_x_l74_74340

def diamond (a b : ℝ) : ℝ := 3 * a * b - a + b

theorem find_x : ∃ x : ℝ, diamond 3 x = 24 ∧ x = 2.7 :=
by
  sorry

end find_x_l74_74340


namespace average_time_per_other_class_l74_74337

theorem average_time_per_other_class (school_hours : ℚ) (num_classes : ℕ) (hist_chem_hours : ℚ)
  (total_school_time_minutes : ℕ) (hist_chem_time_minutes : ℕ) (num_other_classes : ℕ)
  (other_classes_time_minutes : ℕ) (average_time_other_classes : ℕ) :
  school_hours = 7.5 →
  num_classes = 7 →
  hist_chem_hours = 1.5 →
  total_school_time_minutes = school_hours * 60 →
  hist_chem_time_minutes = hist_chem_hours * 60 →
  other_classes_time_minutes = total_school_time_minutes - hist_chem_time_minutes →
  num_other_classes = num_classes - 2 →
  average_time_other_classes = other_classes_time_minutes / num_other_classes →
  average_time_other_classes = 72 :=
by
  intros
  sorry

end average_time_per_other_class_l74_74337


namespace greatest_integer_value_l74_74975

theorem greatest_integer_value (x : ℤ) : ∃ x, (∀ y, (x^2 + 2 * x + 10) % (x - 3) = 0 → x ≥ y) → x = 28 :=
by
  sorry

end greatest_integer_value_l74_74975


namespace total_interest_at_tenth_year_l74_74682

-- Define the conditions for the simple interest problem
variables (P R T : ℝ)

-- Given conditions in the problem
def initial_condition : Prop := (P * R * 10) / 100 = 800
def trebled_principal_condition : Prop := (3 * P * R * 5) / 100 = 1200

-- Statement to prove
theorem total_interest_at_tenth_year (h1 : initial_condition P R) (h2 : trebled_principal_condition P R) :
  (800 + 1200) = 2000 := by
  sorry

end total_interest_at_tenth_year_l74_74682


namespace percentage_of_apples_is_50_l74_74389

-- Definitions based on the conditions
def initial_apples : ℕ := 10
def initial_oranges : ℕ := 23
def oranges_removed : ℕ := 13

-- Final percentage calculation after removing 13 oranges
def percentage_apples (apples oranges_removed : ℕ) :=
  let total_initial := initial_apples + initial_oranges
  let oranges_left := initial_oranges - oranges_removed
  let total_after_removal := initial_apples + oranges_left
  (initial_apples * 100) / total_after_removal

-- The theorem to be proved
theorem percentage_of_apples_is_50 : percentage_apples initial_apples oranges_removed = 50 := by
  sorry

end percentage_of_apples_is_50_l74_74389


namespace parallel_lines_l74_74024

theorem parallel_lines (a : ℝ) :
  (∀ x y : ℝ, x + a * y - (2 * a + 2) = 0 ∧ a * x + y - (a + 1) = 0 → (∀ x y : ℝ, (1 / a = a / 1) ∧ (1 / a ≠ (2 * -a - 2) / (1 * -a - 1)))) → a = 1 := by
sorry

end parallel_lines_l74_74024


namespace compute_value_l74_74701

variables {p q r : ℝ}

theorem compute_value (h1 : (p * q) / (p + r) + (q * r) / (q + p) + (r * p) / (r + q) = -7)
                      (h2 : (p * r) / (p + r) + (q * p) / (q + p) + (r * q) / (r + q) = 8) :
  (q / (p + q) + r / (q + r) + p / (r + p)) = 9 :=
sorry

end compute_value_l74_74701


namespace simplify_fraction_l74_74804

theorem simplify_fraction :
  (4^5 + 4^3) / (4^4 - 4^2) = 68 / 15 :=
by sorry

end simplify_fraction_l74_74804


namespace total_volume_of_cubes_l74_74152

theorem total_volume_of_cubes 
  (Carl_cubes : ℕ)
  (Carl_side_length : ℕ)
  (Kate_cubes : ℕ)
  (Kate_side_length : ℕ)
  (h1 : Carl_cubes = 8)
  (h2 : Carl_side_length = 2)
  (h3 : Kate_cubes = 3)
  (h4 : Kate_side_length = 3) :
  Carl_cubes * Carl_side_length ^ 3 + Kate_cubes * Kate_side_length ^ 3 = 145 :=
by
  sorry

end total_volume_of_cubes_l74_74152


namespace part_a_part_b_l74_74940

variable (p : ℕ → ℕ)
axiom primes_sequence : ∀ n, (∀ m < p n, m ∣ p n → m = 1 ∨ m = p n) ∧ p 1 = 2 ∧ p 2 = 3 ∧ p 3 = 5 ∧ p 4 = 7 ∧ p 5 = 11

theorem part_a (n : ℕ) (h : n ≥ 5) : p n > 2 * n := 
  by sorry

theorem part_b (n : ℕ) : p n > 3 * n ↔ n ≥ 12 := 
  by sorry

end part_a_part_b_l74_74940


namespace truck_travel_distance_l74_74856

theorem truck_travel_distance (b t : ℝ) (ht : t > 0) (ht30 : t + 30 > 0) : 
  let converted_feet := 4 * 60
  let time_half := converted_feet / 2
  let speed_first_half := b / 4
  let speed_second_half := b / 4
  let distance_first_half := speed_first_half * time_half / t
  let distance_second_half := speed_second_half * time_half / (t + 30)
  let total_distance_feet := distance_first_half + distance_second_half
  let result_yards := total_distance_feet / 3
  result_yards = (10 * b / t) + (10 * b / (t + 30))
:= by
  -- proof skipped
  sorry

end truck_travel_distance_l74_74856


namespace ratio_length_to_breadth_l74_74487

theorem ratio_length_to_breadth (b l : ℕ) (A : ℕ) (h1 : b = 30) (h2 : A = 2700) (h3 : A = l * b) :
  l / b = 3 :=
by sorry

end ratio_length_to_breadth_l74_74487


namespace rectangle_width_length_ratio_l74_74927

theorem rectangle_width_length_ratio (w : ℕ) (h : w + 10 = 15) : w / 10 = 1 / 2 :=
by sorry

end rectangle_width_length_ratio_l74_74927


namespace sum_is_45_l74_74225

noncomputable def sum_of_numbers (a b c : ℝ) : ℝ :=
  a + b + c

theorem sum_is_45 {a b c : ℝ} (h1 : ∃ a b c, (a ≤ b ∧ b ≤ c) ∧ b = 10)
  (h2 : (a + b + c) / 3 = a + 20)
  (h3 : (a + b + c) / 3 = c - 25) :
  sum_of_numbers a b c = 45 := 
sorry

end sum_is_45_l74_74225


namespace problem_statement_l74_74450

-- Define the problem conditions
def m : ℝ := 2 * Real.sin (Real.pi * 18 / 180)
def n : ℝ := 4 - m^2

-- Define the expression to be evaluated
def expr : ℝ := m * Real.sqrt n / (2 * Real.cos (Real.pi * 27 / 180)^2 - 1)

-- Prove the expression equals 2
theorem problem_statement : expr = 2 :=
by
  -- proof will be placed here
  sorry

end problem_statement_l74_74450


namespace units_digit_of_150_factorial_is_zero_l74_74250

theorem units_digit_of_150_factorial_is_zero : 
  ∃ k : ℕ, (150! = k * 10) :=
begin
  -- We need to prove that there exists a natural number k such that 150! is equal to k times 10
  sorry
end

end units_digit_of_150_factorial_is_zero_l74_74250


namespace total_days_on_jury_duty_l74_74783

-- Definitions based on conditions
def jurySelectionDays := 2
def juryDeliberationDays := 6
def deliberationHoursPerDay := 16
def trialDurationMultiplier := 4

-- Calculate the total number of hours spent in deliberation
def totalDeliberationHours := juryDeliberationDays * 24

-- Calculate the number of days spent in deliberation based on hours per day
def deliberationDays := totalDeliberationHours / deliberationHoursPerDay

-- Calculate the trial days based on trial duration multiplier
def trialDays := jurySelectionDays * trialDurationMultiplier

-- Calculate the total days on jury duty
def totalJuryDutyDays := jurySelectionDays + trialDays + deliberationDays

theorem total_days_on_jury_duty : totalJuryDutyDays = 19 := by
  sorry

end total_days_on_jury_duty_l74_74783


namespace cross_square_side_length_l74_74693

theorem cross_square_side_length (A : ℝ) (s : ℝ) (h1 : A = 810) 
(h2 : (2 * (s / 2)^2 + 2 * (s / 4)^2) = A) : s = 36 := by
  sorry

end cross_square_side_length_l74_74693


namespace ratio_as_percentage_l74_74323

theorem ratio_as_percentage (x : ℝ) (h : (x / 2) / (3 * x / 5) = 3 / 5) : 
  (3 / 5) * 100 = 60 := 
sorry

end ratio_as_percentage_l74_74323


namespace jake_balloons_bought_l74_74401

theorem jake_balloons_bought (B : ℕ) (h : 6 = (2 + B) + 1) : B = 3 :=
by
  -- proof omitted
  sorry

end jake_balloons_bought_l74_74401


namespace initially_had_8_l74_74479

-- Define the number of puppies given away
def given_away : ℕ := 4

-- Define the number of puppies still with Sandy
def still_has : ℕ := 4

-- Define the total number of puppies initially
def initially_had (x y : ℕ) : ℕ := x + y

-- Prove that the number of puppies Sandy's dog had initially equals 8
theorem initially_had_8 : initially_had given_away still_has = 8 :=
by sorry

end initially_had_8_l74_74479


namespace right_triangle_hypotenuse_l74_74976

theorem right_triangle_hypotenuse (a b : ℕ) (h₁ : a = 75) (h₂ : b = 100) : ∃ c, c = 125 ∧ c^2 = a^2 + b^2 :=
by
  sorry

end right_triangle_hypotenuse_l74_74976


namespace negation_of_p_l74_74025

open Real

-- Define the original proposition p
def p : Prop := ∀ x : ℝ, exp x > log x

-- Theorem stating that the negation of p is as described
theorem negation_of_p : ¬p ↔ ∃ x : ℝ, exp x ≤ log x :=
by
  sorry

end negation_of_p_l74_74025


namespace john_jury_duty_days_l74_74788

-- Definitions based on the given conditions
variable (jury_selection_days : ℕ) (trial_multiple : ℕ) (deliberation_days : ℕ) (hours_per_day_deliberation : ℕ)
variable (total_days_jury_duty : ℕ)

-- Conditions
def condition1 : Prop := jury_selection_days = 2
def condition2 : Prop := trial_multiple = 4
def condition3 : Prop := deliberation_days = 6
def condition4 : Prop := hours_per_day_deliberation = 16
def correct_answer : Prop := total_days_jury_duty = 19

-- Total days calculation
def total_days_calc : ℕ :=
  let trial_days := jury_selection_days * trial_multiple
  let total_deliberation_hours := deliberation_days * 24
  let actual_deliberation_days := total_deliberation_hours / hours_per_day_deliberation
  jury_selection_days + trial_days + actual_deliberation_days

-- Statement we need to prove
theorem john_jury_duty_days : condition1 ∧ condition2 ∧ condition3 ∧ condition4 → total_days_calc = total_days_jury_duty :=
by {
  intros h,
  cases h with h1 h_rest,
  cases h_rest with h2 h_rest1,
  cases h_rest1 with h3 h4,
  rw [condition1, condition2, condition3, condition4] at h1 h2 h3 h4,
  simp [h1, h2, h3, h4],
  sorry
} -- Proof omitted

end john_jury_duty_days_l74_74788


namespace practice_time_for_Friday_l74_74206

variables (M T W Th F : ℕ)

def conditions : Prop :=
  (M = 2 * T) ∧
  (T = W - 10) ∧
  (W = Th + 5) ∧
  (Th = 50) ∧
  (M + T + W + Th + F = 300)

theorem practice_time_for_Friday (h : conditions M T W Th F) : F = 60 :=
sorry

end practice_time_for_Friday_l74_74206


namespace chord_square_length_eq_512_l74_74409

open Real

/-
The conditions are:
1. The radii of two smaller circles are 4 and 8.
2. These circles are externally tangent to each other.
3. Both smaller circles are internally tangent to a larger circle with radius 12.
4. A common external tangent to the two smaller circles serves as a chord of the larger circle.
-/

noncomputable def radius_small1 : ℝ := 4
noncomputable def radius_small2 : ℝ := 8
noncomputable def radius_large : ℝ := 12

/-- Show that the square of the length of the chord formed by the common external tangent of two smaller circles 
which are externally tangent to each other and internally tangent to a larger circle is 512. -/
theorem chord_square_length_eq_512 : ∃ (PQ : ℝ), PQ^2 = 512 := by
  sorry

end chord_square_length_eq_512_l74_74409


namespace donna_soda_crates_l74_74714

def soda_crates (bridge_limit : ℕ) (truck_empty : ℕ) (crate_weight : ℕ) (dryer_weight : ℕ) (num_dryers : ℕ) (truck_loaded : ℕ) (produce_ratio : ℕ) : ℕ :=
  sorry

theorem donna_soda_crates :
  soda_crates 20000 12000 50 3000 3 24000 2 = 20 :=
sorry

end donna_soda_crates_l74_74714


namespace problem_1_problem_2_l74_74461

-- Definitions for problem (1)
def p (x a : ℝ) := x^2 - 4 * a * x + 3 * a^2 < 0 ∧ a > 0
def q (x : ℝ) := x^2 - x - 6 ≤ 0 ∧ x^2 + 3 * x - 10 > 0

-- Statement for problem (1)
theorem problem_1 (a : ℝ) (h : p 1 a ∧ q x) : 2 < x ∧ x < 3 :=
by 
  sorry

-- Definitions for problem (2)
def neg_p (x a : ℝ) := ¬ (x^2 - 4 * a * x + 3 * a^2 < 0 ∧ a > 0)
def neg_q (x : ℝ) := ¬ (x^2 - x - 6 ≤ 0 ∧ x^2 + 3 * x - 10 > 0)

-- Statement for problem (2)
theorem problem_2 (a : ℝ) (h : ∀ x, neg_p x a → neg_q x ∧ ¬ (neg_q x → neg_p x a)) : 1 < a ∧ a ≤ 2 :=
by 
  sorry

end problem_1_problem_2_l74_74461


namespace range_of_a_squared_plus_b_l74_74900

variable (a b : ℝ)

theorem range_of_a_squared_plus_b (h1 : a < -2) (h2 : b > 4) : ∃ y, y = a^2 + b ∧ 8 < y :=
by
  sorry

end range_of_a_squared_plus_b_l74_74900


namespace anne_speed_ratio_l74_74867

theorem anne_speed_ratio (B A A' : ℝ) (h_A : A = 1/12) (h_together_current : (B + A) * 4 = 1) (h_together_new : (B + A') * 3 = 1) :
  A' / A = 2 := 
by
  sorry

end anne_speed_ratio_l74_74867


namespace profit_percentage_approx_l74_74594

-- Define the cost price of the first item
def CP1 (S1 : ℚ) : ℚ := 0.81 * S1

-- Define the selling price of the second item as 10% less than the first
def S2 (S1 : ℚ) : ℚ := 0.90 * S1

-- Define the cost price of the second item as 81% of its selling price
def CP2 (S1 : ℚ) : ℚ := 0.81 * (S2 S1)

-- Define the total selling price before tax
def TSP (S1 : ℚ) : ℚ := S1 + S2 S1

-- Define the total amount received after a 5% tax
def TAR (S1 : ℚ) : ℚ := TSP S1 * 0.95

-- Define the total cost price of both items
def TCP (S1 : ℚ) : ℚ := CP1 S1 + CP2 S1

-- Define the profit
def P (S1 : ℚ) : ℚ := TAR S1 - TCP S1

-- Define the profit percentage
def ProfitPercentage (S1 : ℚ) : ℚ := (P S1 / TCP S1) * 100

-- Prove the profit percentage is approximately 17.28%
theorem profit_percentage_approx (S1 : ℚ) : abs (ProfitPercentage S1 - 17.28) < 0.01 :=
by
  sorry

end profit_percentage_approx_l74_74594


namespace reciprocal_of_one_twentieth_l74_74646

theorem reciprocal_of_one_twentieth : (1 / (1 / 20 : ℝ)) = 20 := 
by
  sorry

end reciprocal_of_one_twentieth_l74_74646


namespace problem_statement_l74_74813

variable {a : ℕ+ → ℝ} 

theorem problem_statement (h : ∀ n : ℕ+, (a n + a (n + 1)) / 2 < a n) :
  (∀ n : ℕ+, a (n + 1) < a n) ∧ -- Sequence is decreasing (original proposition)
  (∀ (a : ℕ+ → ℝ), (∀ n : ℕ+, a (n + 1) < a n) → (∀ n : ℕ+, (a n + a (n + 1)) / 2 < a n)) ∧ -- Inverse
  ((∀ n : ℕ+, (a n + a (n + 1)) / 2 < a n) → (∀ n : ℕ+, a (n + 1) < a n)) ∧ -- Converse
  ((∀ (a : ℕ+ → ℝ), (∀ n : ℕ+, a (n + 1) < a n) → (∀ n : ℕ+, (a n + a (n + 1)) / 2 < a n))) -- Contrapositive
:= by
  sorry

end problem_statement_l74_74813


namespace cadence_total_earnings_l74_74284

noncomputable def total_earnings (old_years : ℕ) (old_monthly : ℕ) (new_increment : ℤ) (extra_months : ℕ) : ℤ :=
  let old_months := old_years * 12
  let old_earnings := old_monthly * old_months
  let new_monthly := old_monthly + ((old_monthly * new_increment) / 100)
  let new_months := old_months + extra_months
  let new_earnings := new_monthly * new_months
  old_earnings + new_earnings

theorem cadence_total_earnings :
  total_earnings 3 5000 20 5 = 426000 :=
by
  sorry

end cadence_total_earnings_l74_74284


namespace problem_proof_l74_74234

theorem problem_proof (N : ℤ) (h : N / 5 = 4) : ((N - 10) * 3) - 18 = 12 :=
by
  -- proof goes here
  sorry

end problem_proof_l74_74234


namespace smallest_common_multiple_8_6_l74_74512

theorem smallest_common_multiple_8_6 : ∃ n : ℕ, n > 0 ∧ (8 ∣ n) ∧ (6 ∣ n) ∧ ∀ m : ℕ, m > 0 ∧ (8 ∣ m) ∧ (6 ∣ m) → n ≤ m :=
begin
  use 24,
  split,
  { exact zero_lt_24, },
  split,
  { exact dvd.intro 3 rfl, },
  split,
  { exact dvd.intro 4 rfl, },
  intros m hm h8 h6,
  -- actual proof here
  sorry
end

end smallest_common_multiple_8_6_l74_74512


namespace determine_remaining_sides_l74_74134

variables (A B C D E : Type)

def cyclic_quadrilateral (A B C D : Type) : Prop := sorry

def known_sides (AB CD : ℝ) : Prop := AB > 0 ∧ CD > 0

def known_ratio (m n : ℝ) : Prop := m > 0 ∧ n > 0

theorem determine_remaining_sides
  {A B C D : Type}
  (h_cyclic : cyclic_quadrilateral A B C D)
  (AB CD : ℝ) (h_sides : known_sides AB CD)
  (m n : ℝ) (h_ratio : known_ratio m n) :
  ∃ (BC AD : ℝ), BC / AD = m / n ∧ BC > 0 ∧ AD > 0 :=
sorry

end determine_remaining_sides_l74_74134


namespace simplification_evaluation_l74_74955

noncomputable def simplify_and_evaluate (x : ℤ) : ℚ :=
  (1 - 1 / (x - 1)) * ((x - 1) / ((x - 2) * (x - 2)))

theorem simplification_evaluation (x : ℤ) (h1 : x > 0) (h2 : 3 - x ≥ 0) : 
  simplify_and_evaluate x = 1 :=
by
  have h3 : x = 3 := sorry
  rw [simplify_and_evaluate, h3]
  simp [h3]
  sorry

end simplification_evaluation_l74_74955


namespace six_smallest_distinct_integers_l74_74893

theorem six_smallest_distinct_integers:
  ∃ (a b c d e f : ℕ), 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
    c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
    d ≠ e ∧ d ≠ f ∧
    e ≠ f ∧
    a * b * c * d * e = 999999 ∧ a = 3 ∧
    f = 37 ∨
    a * b * c * d * f = 999999 ∧ a = 3 ∧ e = 13 ∨ 
    a * b * d * f * e = 999999 ∧ c = 9 ∧ 
    a * c * d * e * f = 999999 ∧ b = 7 ∧ 
    b * c * d * e * f = 999999 ∧ a = 3 := 
sorry

end six_smallest_distinct_integers_l74_74893


namespace real_part_of_z_l74_74916

open Complex

theorem real_part_of_z (z : ℂ) (h : I * z = 1 + 2 * I) : z.re = 2 :=
sorry

end real_part_of_z_l74_74916


namespace candy_problem_l74_74466

theorem candy_problem (N S a : ℕ) (h1 : a = S - a - 7) (h2 : a > 1) : S = 21 := 
sorry

end candy_problem_l74_74466


namespace planA_charge_for_8_minutes_eq_48_cents_l74_74133

theorem planA_charge_for_8_minutes_eq_48_cents
  (X : ℝ)
  (hA : ∀ t : ℝ, t ≤ 8 → X = X)
  (hB : ∀ t : ℝ, 6 * 0.08 = 0.48)
  (hEqual : 6 * 0.08 = X) :
  X = 0.48 := by
  sorry

end planA_charge_for_8_minutes_eq_48_cents_l74_74133


namespace common_roots_product_l74_74079

theorem common_roots_product
  (p q r s : ℝ)
  (hpqrs1 : p + q + r = 0)
  (hpqrs2 : pqr = -20)
  (hpqrs3 : p + q + s = -4)
  (hpqrs4 : pqs = -80)
  : p * q = 20 :=
sorry

end common_roots_product_l74_74079


namespace prime_divides_expression_l74_74941

theorem prime_divides_expression (p : ℕ) (hp : Nat.Prime p) : ∃ n : ℕ, p ∣ (2^n + 3^n + 6^n - 1) := 
by
  sorry

end prime_divides_expression_l74_74941


namespace cone_tangent_min_lateral_area_l74_74494

/-- 
Given a cone with volume π / 6, prove that when the lateral area of the cone is minimized,
the tangent of the angle between the slant height and the base is sqrt(2).
-/
theorem cone_tangent_min_lateral_area :
  ∀ (r h l : ℝ), (π / 6 = (1 / 3) * π * r^2 * h) →
    (h = 1 / (2 * r^2)) →
    (l = Real.sqrt (r^2 + h^2)) →
    ((π * r * l) ≥ (3 / 4 * π)) →
    (r = Real.sqrt (2) / 2) →
    (h / r = Real.sqrt (2)) :=
by
  intro r h l V_cond h_cond l_def min_lateral_area r_val
  -- Proof steps go here (omitted as per the instruction)
  sorry

end cone_tangent_min_lateral_area_l74_74494


namespace minValue_is_9_minValue_achieves_9_l74_74942

noncomputable def minValue (x y : ℝ) : ℝ :=
  (x^2 + 1/(y^2)) * (1/(x^2) + 4 * y^2)

theorem minValue_is_9 (x y : ℝ) (hx : x > 0) (hy : y > 0) : minValue x y ≥ 9 :=
  sorry

theorem minValue_achieves_9 (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x * y = 1/2) : minValue x y = 9 :=
  sorry

end minValue_is_9_minValue_achieves_9_l74_74942


namespace units_digit_of_150_factorial_is_zero_l74_74246

theorem units_digit_of_150_factorial_is_zero : (Nat.factorial 150) % 10 = 0 := by
sorry

end units_digit_of_150_factorial_is_zero_l74_74246


namespace units_digit_factorial_150_l74_74251

theorem units_digit_factorial_150 : (nat.factorial 150) % 10 = 0 :=
sorry

end units_digit_factorial_150_l74_74251


namespace inscribed_quadrilateral_circle_eq_radius_l74_74570

noncomputable def inscribed_circle_condition (AB CD AD BC : ℝ) : Prop :=
  AB + CD = AD + BC

noncomputable def equal_radius_condition (r₁ r₂ r₃ r₄ : ℝ) : Prop :=
  r₁ = r₃ ∨ r₄ = r₂

theorem inscribed_quadrilateral_circle_eq_radius 
  (AB CD AD BC r₁ r₂ r₃ r₄ : ℝ)
  (h_inscribed_circle: inscribed_circle_condition AB CD AD BC)
  (h_four_circles: ∀ i, (i = 1 ∨ i = 2 ∨ i = 3 ∨ i = 4) → ∃ (r : ℝ), r = rᵢ): 
  equal_radius_condition r₁ r₂ r₃ r₄ :=
by {
  sorry
}

end inscribed_quadrilateral_circle_eq_radius_l74_74570


namespace square_root_of_25_is_5_and_minus_5_l74_74831

theorem square_root_of_25_is_5_and_minus_5 : ∃ y : ℝ, y^2 = 25 ∧ (y = 5 ∨ y = -5) :=
by
  have h1 : 5^2 = 25 := by norm_num
  have h2 : (-5)^2 = 25 := by norm_num
  use 5
  use -5
  split
  · exact h1
  · exact h2

end square_root_of_25_is_5_and_minus_5_l74_74831


namespace class_total_students_l74_74925

theorem class_total_students (x y : ℕ)
  (initial_absent : y = (1/6) * x)
  (after_sending_chalk : y = (1/5) * (x - 1)) :
  x + y = 7 :=
by
  sorry

end class_total_students_l74_74925


namespace pentagon_square_ratio_l74_74541

theorem pentagon_square_ratio (p s : ℕ) 
  (h1 : 5 * p = 20) (h2 : 4 * s = 20) : p / s = 4 / 5 :=
by sorry

end pentagon_square_ratio_l74_74541


namespace paintings_in_four_weeks_l74_74862

theorem paintings_in_four_weeks (hours_per_week : ℕ) (hours_per_painting : ℕ) (num_weeks : ℕ) :
  hours_per_week = 30 → hours_per_painting = 3 → num_weeks = 4 → 
  (hours_per_week / hours_per_painting) * num_weeks = 40 :=
by
  -- Sorry is used since we are not providing the proof
  sorry

end paintings_in_four_weeks_l74_74862


namespace find_m_l74_74178

-- Define the vectors a, b, and c
def a : ℝ × ℝ := (2, -1)
def b : ℝ × ℝ := (1, 0)
def c : ℝ × ℝ := (1, -2)

-- Define the condition that a is parallel to m * b - c
def is_parallel (a : ℝ × ℝ) (v : ℝ × ℝ) : Prop :=
  a.1 * v.2 = a.2 * v.1

-- The main theorem we want to prove
theorem find_m (m : ℝ) (h : is_parallel a (m * b.1 - c.1, m * b.2 - c.2)) : m = -3 :=
by {
  -- This will be filled in with the appropriate proof
  sorry
}

end find_m_l74_74178


namespace banana_permutations_l74_74584

open Function

def factorial : ℕ → ℕ
| 0     := 1
| (n + 1) := (n + 1) * factorial n

def number_of_permutations (total items1 items2 : ℕ) : ℕ :=
  factorial total / (factorial items1 * factorial items2 * factorial (total - items1 - items2))

theorem banana_permutations : number_of_permutations 6 3 2 = 60 :=
by {
  unfold number_of_permutations,
  simp,
  sorry
}


end banana_permutations_l74_74584


namespace total_weekly_airflow_l74_74088

-- Definitions from conditions
def fanA_airflow : ℝ := 10  -- liters per second
def fanA_time_per_day : ℝ := 10 * 60  -- converted to seconds (10 minutes * 60 seconds/minute)

def fanB_airflow : ℝ := 15  -- liters per second
def fanB_time_per_day : ℝ := 20 * 60  -- converted to seconds (20 minutes * 60 seconds/minute)

def fanC_airflow : ℝ := 25  -- liters per second
def fanC_time_per_day : ℝ := 30 * 60  -- converted to seconds (30 minutes * 60 seconds/minute)

def days_in_week : ℝ := 7

-- Theorem statement to be proven
theorem total_weekly_airflow : fanA_airflow * fanA_time_per_day * days_in_week +
                               fanB_airflow * fanB_time_per_day * days_in_week +
                               fanC_airflow * fanC_time_per_day * days_in_week = 483000 := 
by
  -- skip the proof
  sorry

end total_weekly_airflow_l74_74088


namespace complement_intersection_l74_74756

def A : Set ℝ := {x | x^2 - 5 * x - 6 ≤ 0}
def B : Set ℝ := {x | x > 7}

theorem complement_intersection :
  (Set.univ \ A) ∩ B = {x | x > 7} :=
by
  sorry

end complement_intersection_l74_74756


namespace farmer_farm_size_l74_74847

theorem farmer_farm_size 
  (sunflowers flax : ℕ)
  (h1 : flax = 80)
  (h2 : sunflowers = flax + 80) :
  (sunflowers + flax = 240) :=
by
  sorry

end farmer_farm_size_l74_74847


namespace random_point_between_R_S_l74_74213

theorem random_point_between_R_S {P Q R S : ℝ} (PQ PR RS : ℝ) (h1 : PQ = 4 * PR) (h2 : PQ = 8 * RS) :
  let PS := PR + RS
  let probability := RS / PQ
  probability = 5 / 8 :=
by
  let PS := PR + RS
  let probability := RS / PQ
  sorry

end random_point_between_R_S_l74_74213


namespace total_cost_for_tickets_l74_74520

-- Definitions given in conditions
def num_students : ℕ := 20
def num_teachers : ℕ := 3
def ticket_cost : ℕ := 5

-- Proof Statement 
theorem total_cost_for_tickets : num_students + num_teachers * ticket_cost = 115 := by
  sorry

end total_cost_for_tickets_l74_74520


namespace class_with_avg_40_students_l74_74815

theorem class_with_avg_40_students
  (x y : ℕ)
  (h : 40 * x + 60 * y = (380 * (x + y)) / 7) : x = 40 :=
sorry

end class_with_avg_40_students_l74_74815


namespace man_work_days_l74_74991

theorem man_work_days (M : ℕ) (h1 : (1 : ℝ)/M + (1 : ℝ)/10 = 1/5) : M = 10 :=
sorry

end man_work_days_l74_74991


namespace complement_of_A_in_U_l74_74031

open Set

def univeral_set : Set ℕ := { x | x + 1 ≤ 0 ∨ 0 ≤ x - 5 }

def A : Set ℕ := {1, 2, 4}

noncomputable def complement_U_A : Set ℕ := {0, 3}

theorem complement_of_A_in_U : (compl A ∩ univeral_set) = complement_U_A := 
by 
  sorry

end complement_of_A_in_U_l74_74031


namespace rate_of_stream_l74_74992

def effectiveSpeedDownstream (v : ℝ) : ℝ := 36 + v
def effectiveSpeedUpstream (v : ℝ) : ℝ := 36 - v

theorem rate_of_stream (v : ℝ) (hf1 : effectiveSpeedUpstream v = 3 * effectiveSpeedDownstream v) : v = 18 := by
  sorry

end rate_of_stream_l74_74992


namespace case_a_second_player_wins_case_b_first_player_wins_case_c_winner_based_on_cell_color_case_d_examples_l74_74665

-- Conditions for Case (a)
def corner_cell (board : Type) (cell : board) : Prop :=
  -- definition to determine if a cell is a corner cell
  sorry

theorem case_a_second_player_wins (board : Type) (starting_cell : board) (player : ℕ) :
  corner_cell board starting_cell → 
  player = 2 :=
by
  sorry
  
-- Conditions for Case (b)
def initial_setup_according_to_figure (board : Type) (starting_cell : board) : Prop :=
  -- definition to determine if a cell setup matches the figure
  sorry

theorem case_b_first_player_wins (board : Type) (starting_cell : board) (player : ℕ) :
  initial_setup_according_to_figure board starting_cell → 
  player = 1 :=
by
  sorry

-- Conditions for Case (c)
def black_cell (board : Type) (cell : board) : Prop :=
  -- definition to determine if a cell is black
  sorry

theorem case_c_winner_based_on_cell_color (board : Type) (starting_cell : board) (player : ℕ) :
  (black_cell board starting_cell → player = 1) ∧ (¬ black_cell board starting_cell → player = 2) :=
by
  sorry
  
-- Conditions for Case (d)
def same_starting_cell_two_games (board : Type) (starting_cell : board) : Prop :=
  -- definition for same starting cell but different outcomes in games
  sorry

theorem case_d_examples (board : Type) (starting_cell : board) (player1 player2 : ℕ) :
  (same_starting_cell_two_games board starting_cell → (player1 = 1 ∧ player2 = 2)) ∨ 
  (same_starting_cell_two_games board starting_cell → (player1 = 2 ∧ player2 = 1)) :=
by
  sorry

end case_a_second_player_wins_case_b_first_player_wins_case_c_winner_based_on_cell_color_case_d_examples_l74_74665


namespace vlad_taller_than_sister_l74_74370

theorem vlad_taller_than_sister : 
  ∀ (vlad_height sister_height : ℝ), 
  vlad_height = 190.5 → sister_height = 86.36 → vlad_height - sister_height = 104.14 :=
by
  intros vlad_height sister_height vlad_height_eq sister_height_eq
  rw [vlad_height_eq, sister_height_eq]
  sorry

end vlad_taller_than_sister_l74_74370


namespace Jon_needs_to_wash_20_pairs_of_pants_l74_74454

theorem Jon_needs_to_wash_20_pairs_of_pants
  (machine_capacity : ℕ)
  (shirts_per_pound : ℕ)
  (pants_per_pound : ℕ)
  (num_shirts : ℕ)
  (num_loads : ℕ)
  (total_pounds : ℕ)
  (weight_of_shirts : ℕ)
  (remaining_weight : ℕ)
  (num_pairs_of_pants : ℕ) :
  machine_capacity = 5 →
  shirts_per_pound = 4 →
  pants_per_pound = 2 →
  num_shirts = 20 →
  num_loads = 3 →
  total_pounds = num_loads * machine_capacity →
  weight_of_shirts = num_shirts / shirts_per_pound →
  remaining_weight = total_pounds - weight_of_shirts →
  num_pairs_of_pants = remaining_weight * pants_per_pound →
  num_pairs_of_pants = 20 :=
by
  intros _ _ _ _ _ _ _ _ _
  sorry

end Jon_needs_to_wash_20_pairs_of_pants_l74_74454


namespace company_percentage_increase_l74_74286

/-- Company P had 426.09 employees in January and 490 employees in December.
    Prove that the percentage increase in employees from January to December is 15%. --/
theorem company_percentage_increase :
  ∀ (employees_jan employees_dec : ℝ),
  employees_jan = 426.09 → 
  employees_dec = 490 → 
  ((employees_dec - employees_jan) / employees_jan) * 100 = 15 :=
by
  intros employees_jan employees_dec h_jan h_dec
  sorry

end company_percentage_increase_l74_74286


namespace area_of_field_l74_74140

-- Define the conditions: length, width, and total fencing
def length : ℕ := 40
def fencing : ℕ := 74

-- Define the property being proved: the area of the field
theorem area_of_field : ∃ (width : ℕ), 2 * width + length = fencing ∧ length * width = 680 :=
by
  -- Proof omitted
  sorry

end area_of_field_l74_74140


namespace certain_number_l74_74689

theorem certain_number (x y : ℝ) (h1 : 0.65 * x = 0.20 * y) (h2 : x = 210) : y = 682.5 :=
by
  sorry

end certain_number_l74_74689


namespace jane_journey_duration_l74_74929

noncomputable def hours_to_seconds (h : ℕ) : ℕ := h * 3600 + 30

theorem jane_journey_duration :
  ∃ (start_time end_time : ℕ), 
    (start_time > 10 * 3600) ∧ (start_time < 11 * 3600) ∧
    (end_time > 17 * 3600) ∧ (end_time < 18 * 3600) ∧
    end_time - start_time = hours_to_seconds 7 :=
by sorry

end jane_journey_duration_l74_74929


namespace john_total_jury_duty_days_l74_74785

-- Definitions based on the given conditions
def jury_selection_days : ℕ := 2
def trial_length_multiplier : ℕ := 4
def deliberation_days : ℕ := 6
def deliberation_hours_per_day : ℕ := 16
def hours_in_a_day : ℕ := 24

-- Define the total days John spends on jury duty
def total_days_on_jury_duty : ℕ :=
  jury_selection_days +
  (trial_length_multiplier * jury_selection_days) +
  (deliberation_days * deliberation_hours_per_day) / hours_in_a_day

-- The theorem to prove
theorem john_total_jury_duty_days : total_days_on_jury_duty = 14 := by
  sorry

end john_total_jury_duty_days_l74_74785


namespace max_value_of_a_b_c_l74_74427

theorem max_value_of_a_b_c (a b c : ℤ) (h1 : a + b = 2006) (h2 : c - a = 2005) (h3 : a < b) : 
  a + b + c = 5013 :=
sorry

end max_value_of_a_b_c_l74_74427


namespace biscuits_more_than_cookies_l74_74120

theorem biscuits_more_than_cookies :
  let morning_butter_cookies := 20
  let morning_biscuits := 40
  let afternoon_butter_cookies := 10
  let afternoon_biscuits := 20
  let total_butter_cookies := morning_butter_cookies + afternoon_butter_cookies
  let total_biscuits := morning_biscuits + afternoon_biscuits
  total_biscuits - total_butter_cookies = 30 :=
by
  sorry

end biscuits_more_than_cookies_l74_74120


namespace molecular_weight_correct_l74_74978

-- Atomic weights
def atomic_weight_Al : ℝ := 26.98
def atomic_weight_O : ℝ := 15.999
def atomic_weight_H : ℝ := 1.008

-- Number of each type of atom in the compound
def num_Al : ℕ := 1
def num_O : ℕ := 3
def num_H : ℕ := 3

-- Molecular weight calculation
def molecular_weight : ℝ :=
  (num_Al * atomic_weight_Al) + (num_O * atomic_weight_O) + (num_H * atomic_weight_H)

theorem molecular_weight_correct : molecular_weight = 78.001 := by
  sorry

end molecular_weight_correct_l74_74978


namespace remainder_of_130_div_k_l74_74896

theorem remainder_of_130_div_k (k a : ℕ) (hk : 90 = a * k^2 + 18) : 130 % k = 4 :=
sorry

end remainder_of_130_div_k_l74_74896


namespace audrey_not_dreaming_fraction_l74_74865

theorem audrey_not_dreaming_fraction :
  let cycle1_not_dreaming := 3 / 4
  let cycle2_not_dreaming := 5 / 7
  let cycle3_not_dreaming := 2 / 3
  let cycle4_not_dreaming := 4 / 7
  cycle1_not_dreaming + cycle2_not_dreaming + cycle3_not_dreaming + cycle4_not_dreaming = 227 / 84 :=
by
  let cycle1_not_dreaming := 3 / 4
  let cycle2_not_dreaming := 5 / 7
  let cycle3_not_dreaming := 2 / 3
  let cycle4_not_dreaming := 4 / 7
  sorry

end audrey_not_dreaming_fraction_l74_74865


namespace arithmetic_sequence_seventy_fifth_term_l74_74485

theorem arithmetic_sequence_seventy_fifth_term:
  ∀ (a₁ a₂ d : ℕ), a₁ = 3 → a₂ = 51 → a₂ = a₁ + 24 * d → (3 + 74 * d) = 151 := by
  sorry

end arithmetic_sequence_seventy_fifth_term_l74_74485


namespace find_y_from_equation_l74_74720

theorem find_y_from_equation (y : ℕ) 
  (h : (12 ^ 2) * (6 ^ 3) / y = 72) : 
  y = 432 :=
  sorry

end find_y_from_equation_l74_74720


namespace Jorge_Giuliana_cakes_l74_74455

theorem Jorge_Giuliana_cakes (C : ℕ) :
  (2 * 7 + 2 * C + 2 * 30 = 110) → (C = 18) :=
by
  sorry

end Jorge_Giuliana_cakes_l74_74455


namespace park_area_l74_74141

theorem park_area (l w : ℝ) (h1 : l + w = 40) (h2 : l = 3 * w) : l * w = 300 :=
by
  sorry

end park_area_l74_74141


namespace problem_part1_problem_part2_l74_74430

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := abs (a - x)

def setA (a : ℝ) : Set ℝ := {x | f a (2 * x - 3 / 2) > 2 * f a (x + 2) + 2}

theorem problem_part1 {a : ℝ} (h : a = 3 / 2) : setA a = {x | x < 0} := by
  sorry

theorem problem_part2 {a : ℝ} (h : a = 3 / 2) (x0 : ℝ) (hx0 : x0 ∈ setA a) (x : ℝ) : 
    f a (x0 * x) ≥ x0 * f a x + f a (a * x0) := by
  sorry

end problem_part1_problem_part2_l74_74430


namespace evaluate_expression_l74_74017

theorem evaluate_expression : 6 - 5 * (10 - (2 + 3)^2) * 2 = 306 := by
  sorry

end evaluate_expression_l74_74017


namespace handshake_count_l74_74497

-- Defining the conditions
def number_of_companies : ℕ := 5
def representatives_per_company : ℕ := 5
def total_participants : ℕ := number_of_companies * representatives_per_company

-- Defining the number of handshakes each person makes
def handshakes_per_person : ℕ := total_participants - 1 - (representatives_per_company - 1)

-- Defining the total number of handshakes
def total_handshakes : ℕ := (total_participants * handshakes_per_person) / 2

theorem handshake_count :
  total_handshakes = 250 :=
by
  sorry

end handshake_count_l74_74497


namespace simplify_and_evaluate_l74_74066

-- Define the expression
def expression (x : ℝ) := -(2 * x^2 + 3 * x) + 2 * (4 * x + x^2)

-- State the theorem
theorem simplify_and_evaluate : expression (-2) = -10 :=
by
  -- The proof goes here
  sorry

end simplify_and_evaluate_l74_74066


namespace coprime_pos_addition_l74_74208

noncomputable def X_log_Y_Z (X Y Z : ℕ) : Prop :=
  X * (Real.log 2 / Real.log 1000) + Y * (Real.log 3 / Real.log 1000) = Z

theorem coprime_pos_addition (X Y Z : ℕ) (h1 : Nat.Coprime X Y) (h2 : Nat.Coprime X Z) (h3 : Nat.Coprime Y Z)
  (hX : 0 < X) (hY : 0 < Y) (hZ : 0 < Z) (hXYZ : X_log_Y_Z X Y Z) : 
  X + Y + Z = 4 := 
sorry

end coprime_pos_addition_l74_74208


namespace one_fifth_of_five_times_nine_l74_74891

theorem one_fifth_of_five_times_nine (a b : ℕ) (h1 : a = 5) (h2 : b = 9) : (1 / 5 : ℚ) * (a * b) = 9 := by
  sorry

end one_fifth_of_five_times_nine_l74_74891


namespace find_smallest_m_l74_74824

theorem find_smallest_m : ∃ m : ℕ, m > 0 ∧ (790 * m ≡ 1430 * m [MOD 30]) ∧ ∀ n : ℕ, n > 0 ∧ (790 * n ≡ 1430 * n [MOD 30]) → m ≤ n :=
by
  sorry

end find_smallest_m_l74_74824


namespace rectangle_area_relation_l74_74534

theorem rectangle_area_relation (x y : ℝ) (h : x * y = 4) (hx : x > 0) : y = 4 / x := 
sorry

end rectangle_area_relation_l74_74534


namespace frank_worked_days_l74_74899

def total_hours : ℝ := 8.0
def hours_per_day : ℝ := 2.0

theorem frank_worked_days :
  (total_hours / hours_per_day = 4.0) :=
by sorry

end frank_worked_days_l74_74899


namespace isosceles_triangle_base_angle_l74_74590

theorem isosceles_triangle_base_angle (vertex_angle : ℝ) (h1 : vertex_angle = 110) :
  ∃ base_angle : ℝ, base_angle = 35 :=
by
  use 35
  sorry

end isosceles_triangle_base_angle_l74_74590


namespace positive_integer_solutions_l74_74642

theorem positive_integer_solutions (x : ℕ) (h : 2 * x + 9 ≥ 3 * (x + 2)) : x = 1 ∨ x = 2 ∨ x = 3 :=
by
  sorry

end positive_integer_solutions_l74_74642


namespace multiplication_simplify_l74_74154

theorem multiplication_simplify :
  12 * (1 / 8) * 32 = 48 := 
sorry

end multiplication_simplify_l74_74154


namespace joseph_savings_ratio_l74_74100

theorem joseph_savings_ratio
    (thomas_monthly_savings : ℕ)
    (thomas_years_saving : ℕ)
    (total_savings : ℕ)
    (joseph_total_savings_is_total_minus_thomas : total_savings = thomas_monthly_savings * 12 * thomas_years_saving + (total_savings - thomas_monthly_savings * 12 * thomas_years_saving))
    (thomas_saves_each_month : thomas_monthly_savings = 40)
    (years_saving : thomas_years_saving = 6)
    (total_amount : total_savings = 4608) :
    (total_savings - thomas_monthly_savings * 12 * thomas_years_saving) / (12 * thomas_years_saving) / thomas_monthly_savings = 3 / 5 :=
by
  sorry

end joseph_savings_ratio_l74_74100


namespace determine_range_of_b_l74_74579

noncomputable def f (b x : ℝ) : ℝ := (Real.log x + (x - b) ^ 2) / x
noncomputable def f'' (b x : ℝ) : ℝ := (2 * Real.log x - 2) / x ^ 3

theorem determine_range_of_b (b : ℝ) (h : ∃ x ∈ Set.Icc (1 / 2) 2, f b x > -x * f'' b x) :
  b < 9 / 4 :=
by
  sorry

end determine_range_of_b_l74_74579


namespace value_of_sum_l74_74760

theorem value_of_sum (a b : ℝ) (h : a^2 * b^2 + a^2 + b^2 + 1 - 2 * a * b = 2 * a * b) : a + b = 2 ∨ a + b = -2 :=
sorry

end value_of_sum_l74_74760


namespace first_term_of_geometric_series_l74_74647

-- Define the conditions and the question
theorem first_term_of_geometric_series (a r : ℝ) (h1 : a / (1 - r) = 30) (h2 : a^2 / (1 - r^2) = 180) : a = 10 :=
by sorry

end first_term_of_geometric_series_l74_74647


namespace standard_equation_of_parabola_l74_74595

theorem standard_equation_of_parabola (focus : ℝ × ℝ): 
  (focus.1 - 2 * focus.2 - 4 = 0) → 
  ((focus = (4, 0) → (∃ a : ℝ, ∀ x y : ℝ, y^2 = 4 * a * x)) ∨
   (focus = (0, -2) → (∃ b : ℝ, ∀ x y : ℝ, x^2 = 4 * b * y))) :=
by
  sorry

end standard_equation_of_parabola_l74_74595


namespace product_of_possible_values_l74_74703

theorem product_of_possible_values (N : ℤ) (M L : ℤ) 
(h1 : M = L + N)
(h2 : M - 3 = L + N - 3)
(h3 : L + 5 = L + 5)
(h4 : |(L + N - 3) - (L + 5)| = 4) :
N = 12 ∨ N = 4 → (12 * 4 = 48) :=
by sorry

end product_of_possible_values_l74_74703


namespace cookies_in_jar_l74_74653

noncomputable def C : ℕ := sorry

theorem cookies_in_jar (h : C - 1 = (C + 5) / 2) : C = 7 := by
  sorry

end cookies_in_jar_l74_74653


namespace max_stickers_one_student_l74_74545

def total_students : ℕ := 25
def mean_stickers : ℕ := 4
def total_stickers := total_students * mean_stickers
def minimum_stickers_per_student : ℕ := 1
def minimum_stickers_taken_by_24_students := (total_students - 1) * minimum_stickers_per_student

theorem max_stickers_one_student : 
  total_stickers - minimum_stickers_taken_by_24_students = 76 := by
  sorry

end max_stickers_one_student_l74_74545


namespace month_length_l74_74052

def treats_per_day : ℕ := 2
def cost_per_treat : ℝ := 0.1
def total_cost : ℝ := 6

theorem month_length : (total_cost / cost_per_treat) / treats_per_day = 30 := by
  sorry

end month_length_l74_74052


namespace simplify_and_evaluate_expr_l74_74628

theorem simplify_and_evaluate_expr (a b : ℕ) (h₁ : a = 2) (h₂ : b = 2023) : 
  (a + b)^2 + b * (a - b) - 3 * a * b = 4 := by
  sorry

end simplify_and_evaluate_expr_l74_74628


namespace sugar_in_first_combination_l74_74073

def cost_per_pound : ℝ := 0.45
def cost_combination_1 (S : ℝ) : ℝ := cost_per_pound * S + cost_per_pound * 16
def cost_combination_2 : ℝ := cost_per_pound * 30 + cost_per_pound * 25
def total_weight_combination_2 : ℕ := 30 + 25
def total_weight_combination_1 (S : ℕ) : ℕ := S + 16

theorem sugar_in_first_combination :
  ∀ (S : ℕ), cost_combination_1 S = 26 ∧ cost_combination_2 = 26 → total_weight_combination_1 S = total_weight_combination_2 → S = 39 :=
by sorry

end sugar_in_first_combination_l74_74073


namespace solve_quadratic_inequality_l74_74631

theorem solve_quadratic_inequality (a x : ℝ) :
  (x ^ 2 - (2 + a) * x + 2 * a < 0) ↔ 
  ((a < 2 ∧ a < x ∧ x < 2) ∨ (a = 2 ∧ false) ∨ 
   (a > 2 ∧ 2 < x ∧ x < a)) :=
by sorry

end solve_quadratic_inequality_l74_74631


namespace range_of_a_l74_74081

-- Defining the core problem conditions in Lean
def prop_p (a : ℝ) : Prop := ∃ x₀ : ℝ, a * x₀^2 + 2 * a * x₀ + 1 < 0

-- The original proposition p is false, thus we need to show the range of a is 0 ≤ a ≤ 1
theorem range_of_a (a : ℝ) : ¬ prop_p a → 0 ≤ a ∧ a ≤ 1 :=
by
  sorry

end range_of_a_l74_74081


namespace A_share_in_profit_l74_74857

-- Define the investments and profits
def A_investment : ℕ := 6300
def B_investment : ℕ := 4200
def C_investment : ℕ := 10500
def total_profit : ℕ := 12200

-- Define the total investment
def total_investment : ℕ := A_investment + B_investment + C_investment

-- Define A's ratio in the investment
def A_ratio : ℚ := A_investment / total_investment

-- Define A's share in the profit
def A_share : ℚ := total_profit * A_ratio

-- The theorem to prove
theorem A_share_in_profit : A_share = 3660 := by
  sorry

end A_share_in_profit_l74_74857


namespace find_positive_real_unique_solution_l74_74892

theorem find_positive_real_unique_solution (x : ℝ) (h : 0 < x ∧ (x - 6) / 16 = 6 / (x - 16)) : x = 22 :=
sorry

end find_positive_real_unique_solution_l74_74892


namespace Larry_wins_game_probability_l74_74790

noncomputable def winning_probability_Larry : ℚ :=
  ∑' n : ℕ, if n % 3 = 0 then (2 / 3) ^ (n / 3 * 3) * (1 / 3) else 0

theorem Larry_wins_game_probability : winning_probability_Larry = 9 / 19 :=
by
  sorry

end Larry_wins_game_probability_l74_74790


namespace negation_of_existence_l74_74057

theorem negation_of_existence :
  (¬ ∃ x₀ : ℝ, x₀^2 + 2*x₀ + 2 ≤ 0) ↔ (∀ x : ℝ, x^2 + 2*x + 2 > 0) :=
by
  sorry

end negation_of_existence_l74_74057


namespace gcf_72_108_l74_74108

def gcf (a b : ℕ) : ℕ := 
  Nat.gcd a b

theorem gcf_72_108 : gcf 72 108 = 36 := by
  sorry

end gcf_72_108_l74_74108


namespace correctness_statement_l74_74424

-- Given points A, B, C are on the specific parabola
variable (a c x1 x2 x3 y1 y2 y3 : ℝ)
variable (ha : a < 0) -- a < 0 since the parabola opens upwards
variable (hA : y1 = - (a / 4) * x1^2 + a * x1 + c)
variable (hB : y2 = a + c) -- B is the vertex
variable (hC : y3 = - (a / 4) * x3^2 + a * x3 + c)
variable (hOrder : y1 > y3 ∧ y3 ≥ y2)

theorem correctness_statement : abs (x1 - x2) > abs (x3 - x2) :=
sorry

end correctness_statement_l74_74424


namespace vacuum_cleaner_cost_l74_74606

-- Variables
variables (V : ℝ)

-- Conditions
def cost_of_dishwasher := 450
def coupon := 75
def total_spent := 625

-- The main theorem to prove
theorem vacuum_cleaner_cost : V + cost_of_dishwasher - coupon = total_spent → V = 250 :=
by
  -- Proof logic goes here
  sorry

end vacuum_cleaner_cost_l74_74606


namespace perimeter_of_triangle_is_13_l74_74810

-- Conditions
noncomputable def perimeter_of_triangle_with_two_sides_and_third_root_of_eq : ℝ :=
  let a := 3
  let b := 6
  let c1 := 2 -- One root of the equation x^2 - 6x + 8 = 0
  let c2 := 4 -- Another root of the equation x^2 - 6x + 8 = 0
  if a + b > c2 ∧ a + c2 > b ∧ b + c2 > a then
    a + b + c2
  else
    0 -- not possible to form a triangle with these sides

-- Assertion
theorem perimeter_of_triangle_is_13 :
  perimeter_of_triangle_with_two_sides_and_third_root_of_eq = 13 := 
sorry

end perimeter_of_triangle_is_13_l74_74810


namespace honey_nectar_relationship_l74_74526

-- Definitions representing the conditions
def nectarA_water_content (x : ℝ) := 0.7 * x
def nectarB_water_content (y : ℝ) := 0.5 * y
def final_honey_water_content := 0.3
def evaporation_loss (initial_content : ℝ) := 0.15 * initial_content

-- The system of equations to prove
theorem honey_nectar_relationship (x y : ℝ) :
  (x + y = 1) ∧ (0.595 * x + 0.425 * y = 0.3) :=
sorry

end honey_nectar_relationship_l74_74526


namespace mouse_shortest_path_on_cube_l74_74993

noncomputable def shortest_path_length (edge_length : ℝ) : ℝ :=
  2 * edge_length * Real.sqrt 2

theorem mouse_shortest_path_on_cube :
  shortest_path_length 2 = 4 * Real.sqrt 2 :=
by
  sorry

end mouse_shortest_path_on_cube_l74_74993


namespace identify_stolen_bag_with_two_weighings_l74_74396

-- Definition of the weights of the nine bags
def weights : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9]

-- Statement of the problem: Using two weighings on a balance scale without weights,
-- prove that it is possible to identify the specific bag from which the treasure was stolen.
theorem identify_stolen_bag_with_two_weighings (stolen_bag : {n // n < 9}) :
  ∃ (group1 group2 : List ℕ), group1 ≠ group2 ∧ (group1.sum = 11 ∨ group1.sum = 15) ∧ (group2.sum = 11 ∨ group2.sum = 15) →
  ∃ (b1 b2 b3 : ℕ), b1 ≠ b2 ∧ b1 ≠ b3 ∧ b2 ≠ b3 ∧ b1 + b2 + b3 = 6 ∧ (b1 + b2 = 11 ∨ b1 + b2 = 15) := sorry

end identify_stolen_bag_with_two_weighings_l74_74396


namespace coin_difference_l74_74049

noncomputable def max_value (p n d : ℕ) : ℕ := p + 5 * n + 10 * d
noncomputable def min_value (p n d : ℕ) : ℕ := p + 5 * n + 10 * d

theorem coin_difference (p n d : ℕ) (h₁ : p + n + d = 3030) (h₂ : 10 ≤ p) (h₃ : 10 ≤ n) (h₄ : 10 ≤ d) :
  max_value 10 10 3010 - min_value 3010 10 10 = 27000 := by
  sorry

end coin_difference_l74_74049


namespace average_movers_l74_74778

noncomputable def average_people_per_hour (total_people : ℕ) (total_hours : ℕ) : ℝ :=
  total_people / total_hours

theorem average_movers :
  average_people_per_hour 5000 168 = 29.76 :=
by
  sorry

end average_movers_l74_74778


namespace factor_difference_of_squares_l74_74161

theorem factor_difference_of_squares (t : ℝ) : 4 * t^2 - 81 = (2 * t - 9) * (2 * t + 9) := 
by
  sorry

end factor_difference_of_squares_l74_74161


namespace consecutive_product_neq_consecutive_even_product_l74_74011

open Nat

theorem consecutive_product_neq_consecutive_even_product :
  ∀ m n : ℕ, m * (m + 1) ≠ 4 * n * (n + 1) :=
by
  intros m n
  -- Proof is omitted, as per instructions.
  sorry

end consecutive_product_neq_consecutive_even_product_l74_74011


namespace sufficient_not_necessary_l74_74825

theorem sufficient_not_necessary (a b : ℝ) :
  (a^2 + b^2 = 0 → ab = 0) ∧ (ab = 0 → ¬(a^2 + b^2 = 0)) := 
by
  have h1 : (a^2 + b^2 = 0 → ab = 0) := sorry
  have h2 : (ab = 0 → ¬(a^2 + b^2 = 0)) := sorry
  exact ⟨h1, h2⟩

end sufficient_not_necessary_l74_74825


namespace solve_for_five_minus_a_l74_74441

theorem solve_for_five_minus_a (a b : ℤ) 
  (h1 : 5 + a = 6 - b)
  (h2 : 6 + b = 9 + a) : 
  5 - a = 6 := 
by 
  sorry

end solve_for_five_minus_a_l74_74441


namespace cars_between_15000_and_20000_l74_74403

theorem cars_between_15000_and_20000 
  (total_cars : ℕ)
  (less_than_15000_ratio : ℝ)
  (more_than_20000_ratio : ℝ)
  : less_than_15000_ratio = 0.15 → 
    more_than_20000_ratio = 0.40 → 
    total_cars = 3000 → 
    ∃ (cars_between : ℕ),
      cars_between = total_cars - (less_than_15000_ratio * total_cars + more_than_20000_ratio * total_cars) ∧ 
      cars_between = 1350 :=
by
  sorry

end cars_between_15000_and_20000_l74_74403


namespace ratio_of_sides_l74_74536

theorem ratio_of_sides (perimeter_pentagon perimeter_square : ℝ) (hp : perimeter_pentagon = 20) (hs : perimeter_square = 20) : (4:ℝ) / (5:ℝ) = (4:ℝ) / (5:ℝ) :=
by
  sorry

end ratio_of_sides_l74_74536


namespace Malou_third_quiz_score_l74_74060

theorem Malou_third_quiz_score (q1 q2 q3 : ℕ) (avg_score : ℕ) (total_quizzes : ℕ) : 
  q1 = 91 ∧ q2 = 90 ∧ avg_score = 91 ∧ total_quizzes = 3 → q3 = 92 :=
by
  intro h
  sorry

end Malou_third_quiz_score_l74_74060


namespace jane_paints_correct_area_l74_74782

def height_of_wall : ℕ := 10
def length_of_wall : ℕ := 15
def width_of_door : ℕ := 3
def height_of_door : ℕ := 5

def area_of_wall := height_of_wall * length_of_wall
def area_of_door := width_of_door * height_of_door
def area_to_be_painted := area_of_wall - area_of_door

theorem jane_paints_correct_area : area_to_be_painted = 135 := by
  sorry

end jane_paints_correct_area_l74_74782


namespace find_prime_pairs_l74_74162

open Nat

-- Define a function to check if a number is prime
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the problem as a theorem in Lean
theorem find_prime_pairs :
  ∀ (p n : ℕ), is_prime p ∧ n > 0 ∧ p^3 - 2*p^2 + p + 1 = 3^n ↔ (p = 2 ∧ n = 1) ∨ (p = 5 ∧ n = 4) :=
by
  sorry

end find_prime_pairs_l74_74162


namespace average_book_width_l74_74197

noncomputable def book_widths : List ℚ := [7, 3/4, 1.25, 3, 8, 2.5, 12]
def number_of_books : ℕ := 7
def total_sum_of_widths : ℚ := 34.5

theorem average_book_width :
  ((book_widths.sum) / number_of_books) = 241/49 :=
by
  sorry

end average_book_width_l74_74197


namespace logic_problem_l74_74726

variable (p q : Prop)

theorem logic_problem (h₁ : ¬ p) (h₂ : p ∨ q) : p = False ∧ q = True :=
by
  sorry

end logic_problem_l74_74726


namespace find_ordered_pair_l74_74562

theorem find_ordered_pair (x y : ℝ) :
  (2 * x + 3 * y = (6 - x) + (6 - 3 * y)) ∧ (x - 2 * y = (x - 2) - (y + 2)) ↔ (x = -4) ∧ (y = 4) := by
  sorry

end find_ordered_pair_l74_74562


namespace solve_phi_l74_74319

noncomputable def find_phi (phi : ℝ) : Prop :=
  2 * Real.cos phi - Real.sin phi = Real.sqrt 3 * Real.sin (20 / 180 * Real.pi)

theorem solve_phi (phi : ℝ) :
  find_phi phi ↔ (phi = 140 / 180 * Real.pi ∨ phi = 40 / 180 * Real.pi) :=
sorry

end solve_phi_l74_74319


namespace inequality_sqrt_sum_l74_74799

theorem inequality_sqrt_sum (a b c : ℝ) : 
  (Real.sqrt (a^2 + b^2 - a * b) + Real.sqrt (b^2 + c^2 - b * c)) ≥ Real.sqrt (a^2 + c^2 + a * c) :=
sorry

end inequality_sqrt_sum_l74_74799


namespace max_points_right_triangle_l74_74115

theorem max_points_right_triangle (n : ℕ) :
  (∀ (pts : Fin n → ℝ × ℝ), ∀ (i j k : Fin n), i ≠ j → j ≠ k → i ≠ k →
    let p1 := pts i
    let p2 := pts j
    let p3 := pts k
    let a := (p2.1 - p1.1)^2 + (p2.2 - p1.2)^2
    let b := (p3.1 - p2.1)^2 + (p3.2 - p2.2)^2
    let c := (p3.1 - p1.1)^2 + (p3.2 - p1.2)^2
    a + b = c ∨ b + c = a ∨ c + a = b) →
  n ≤ 4 :=
sorry

end max_points_right_triangle_l74_74115


namespace remainder_is_90_l74_74357

theorem remainder_is_90:
  let larger_number := 2982
  let smaller_number := 482
  let quotient := 6
  (larger_number - smaller_number = 2500) ∧ 
  (larger_number = quotient * smaller_number + r) →
  (r = 90) :=
by
  sorry

end remainder_is_90_l74_74357


namespace break_even_point_l74_74482

/-- Conditions of the problem -/
def fixed_costs : ℝ := 10410
def variable_cost_per_unit : ℝ := 2.65
def selling_price_per_unit : ℝ := 20

/-- The mathematically equivalent proof problem / statement -/
theorem break_even_point :
  fixed_costs / (selling_price_per_unit - variable_cost_per_unit) = 600 := 
by
  -- Proof to be filled in
  sorry

end break_even_point_l74_74482


namespace required_speed_l74_74210

noncomputable def distance_travelled_late (d: ℝ) (t: ℝ) : ℝ :=
  50 * (t + 1/12)

noncomputable def distance_travelled_early (d: ℝ) (t: ℝ) : ℝ :=
  70 * (t - 1/12)

theorem required_speed :
  ∃ (s: ℝ), s = 58 ∧ 
  (∀ (d t: ℝ), distance_travelled_late d t = d ∧ distance_travelled_early d t = d → 
  d / t = s) :=
by
  sorry

end required_speed_l74_74210


namespace f_at_one_f_extremes_l74_74028

noncomputable def f : ℝ → ℝ := sorry
axiom f_domain : ∀ x : ℝ, x > 0 → f x = f x
axiom f_multiplicative : ∀ x y : ℝ, x > 0 → y > 0 → f (x * y) = f x + f y
axiom f_positive : ∀ x : ℝ, x > 1 → f x > 0

theorem f_at_one : f 1 = 0 := sorry

theorem f_extremes (hf_sub_one_fifth : f (1 / 5) = -1) :
  ∃ c d : ℝ, (∀ x : ℝ, 1 / 25 ≤ x ∧ x ≤ 125 → c ≤ f x ∧ f x ≤ d) ∧
  c = -2 ∧ d = 3 := sorry

end f_at_one_f_extremes_l74_74028


namespace train_crossing_time_l74_74779

-- Definitions of the given problem conditions
def train_length : ℕ := 120  -- in meters.
def speed_kmph : ℕ := 144   -- in km/h.

-- Conversion factor
def km_per_hr_to_m_per_s (speed : ℕ) : ℚ :=
  speed * (1000 / 3600 : ℚ)

-- Speed in m/s
def train_speed : ℚ := km_per_hr_to_m_per_s speed_kmph

-- Time calculation
def time_to_cross_pole (length : ℕ) (speed : ℚ) : ℚ :=
  length / speed

-- The theorem we want to prove.
theorem train_crossing_time :
  time_to_cross_pole train_length train_speed = 3 := by 
  sorry

end train_crossing_time_l74_74779


namespace cannot_have_1970_minus_signs_in_grid_l74_74190

theorem cannot_have_1970_minus_signs_in_grid :
  ∀ (k l : ℕ), k ≤ 100 → l ≤ 100 → (k+l)*50 - k*l ≠ 985 :=
by
  intros k l hk hl
  sorry

end cannot_have_1970_minus_signs_in_grid_l74_74190


namespace initial_pokemon_cards_l74_74330

theorem initial_pokemon_cards (x : ℤ) (h : x - 9 = 4) : x = 13 :=
by
  sorry

end initial_pokemon_cards_l74_74330


namespace decryption_probabilities_l74_74504

-- Definitions based on conditions
variables {A B : Type}
variables (p1 p2 : ℝ)
variables (X Y : ℕ)

-- Assumptions and statement of proof
theorem decryption_probabilities (h1 : 0 < p1) (h2 : p1 < p2) (h3 : p2 < 1/2)
  (hX : X ~ (Binomial 3 p1)) (hY : Y ~ (Binomial 3 p2)) :
  (3 * p1 < 3 * p2 ∧ 3 * p1 * (1 - p1) < 3 * p2 * (1 - p2)) :=
by
  -- Insert formal proof steps here
  sorry

end decryption_probabilities_l74_74504


namespace find_least_positive_x_l74_74561

theorem find_least_positive_x :
  ∃ x : ℕ, 0 < x ∧ (x + 5713) % 15 = 1847 % 15 ∧ x = 4 :=
by
  sorry

end find_least_positive_x_l74_74561


namespace diamond_value_l74_74589

def diamond (x y : ℕ) : ℕ := 4 * x + 6 * y

theorem diamond_value : diamond 3 4 = 36 := by
  -- Given condition: x ♢ y = 4x + 6y
  -- To prove: (diamond 3 4) = 36
  sorry

end diamond_value_l74_74589


namespace min_val_of_3x_add_4y_l74_74593

theorem min_val_of_3x_add_4y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 3 * y = 5 * x * y) : 
  (3 * x + 4 * y ≥ 5) ∧ (3 * x + 4 * y = 5 → x + 4 * y = 3) := 
by
  sorry

end min_val_of_3x_add_4y_l74_74593


namespace bus_speed_including_stoppages_l74_74158

theorem bus_speed_including_stoppages 
  (speed_without_stoppages : ℕ) 
  (stoppage_time_per_hour : ℕ) 
  (correct_speed_including_stoppages : ℕ) :
  speed_without_stoppages = 54 →
  stoppage_time_per_hour = 10 →
  correct_speed_including_stoppages = 45 :=
by
sorry

end bus_speed_including_stoppages_l74_74158


namespace triangle_with_ratio_is_right_triangle_l74_74448

/-- If the ratio of the interior angles of a triangle is 1:2:3, then the triangle is a right triangle. -/
theorem triangle_with_ratio_is_right_triangle (x : ℝ) (h : x + 2*x + 3*x = 180) : 
  3*x = 90 :=
sorry

end triangle_with_ratio_is_right_triangle_l74_74448


namespace sin_sum_cos_product_tan_sum_tan_product_l74_74476

theorem sin_sum_cos_product
  (A B C : ℝ)
  (h : A + B + C = π) : 
  (Real.sin A + Real.sin B + Real.sin C = 4 * Real.cos (A / 2) * Real.cos (B / 2) * Real.cos (C / 2)) :=
sorry

theorem tan_sum_tan_product
  (A B C : ℝ)
  (h : A + B + C = π) :
  (Real.tan A + Real.tan B + Real.tan C = Real.tan A * Real.tan B * Real.tan C) := 
sorry

end sin_sum_cos_product_tan_sum_tan_product_l74_74476


namespace ratio_c_d_l74_74030

theorem ratio_c_d (x y c d : ℝ) 
  (h1 : 8 * x - 5 * y = c)
  (h2 : 10 * y - 12 * x = d)
  (hx : x ≠ 0)
  (hy : y ≠ 0)
  (hd : d ≠ 0) :
  c / d = -2 / 3 :=
sorry

end ratio_c_d_l74_74030


namespace inequality_example_l74_74988

theorem inequality_example (a b : ℝ) (h1 : a ≥ b) (h2 : b > 0) : 2 * a^3 - b^3 ≥ 2 * a * b^2 - a^2 * b := 
  sorry

end inequality_example_l74_74988


namespace find_excluded_number_l74_74261

-- Definition of the problem conditions
def avg (nums : List ℕ) : ℕ := (nums.sum / nums.length)

-- Problem condition: the average of 5 numbers is 27
def condition1 (nums : List ℕ) : Prop :=
  nums.length = 5 ∧ avg nums = 27

-- Problem condition: excluding one number, the average of remaining 4 numbers is 25
def condition2 (nums : List ℕ) (x : ℕ) : Prop :=
  let nums' := nums.filter (λ n => n ≠ x)
  nums.length = 5 ∧ nums'.length = 4 ∧ avg nums' = 25

-- Proof statement: finding the excluded number
theorem find_excluded_number (nums : List ℕ) (x : ℕ) (h1 : condition1 nums) (h2 : condition2 nums x) : x = 35 := 
by
  sorry

end find_excluded_number_l74_74261


namespace side_length_of_square_l74_74528

noncomputable def area_of_circle : ℝ := 3848.4510006474966
noncomputable def pi : ℝ := Real.pi

theorem side_length_of_square :
  ∃ s : ℝ, (∃ r : ℝ, area_of_circle = pi * r * r ∧ 2 * r = s) ∧ s = 70 := 
by
  sorry

end side_length_of_square_l74_74528


namespace sarah_score_is_122_l74_74953

-- Define the problem parameters and state the theorem
theorem sarah_score_is_122 (s g : ℝ)
  (h1 : s = g + 40)
  (h2 : (s + g) / 2 = 102) :
  s = 122 :=
by {
  -- Placeholder for the actual proof
  sorry
}

end sarah_score_is_122_l74_74953


namespace directrix_of_parabola_l74_74298

theorem directrix_of_parabola (a b c : ℝ) (parabola_eqn : ∀ x : ℝ, y = 3 * x^2 - 6 * x + 2)
  (vertex : ∃ h k : ℝ, h = 1 ∧ k = -1)
  : ∃ y : ℝ, y = -13 / 12 := 
sorry

end directrix_of_parabola_l74_74298


namespace equation_solution_l74_74748

theorem equation_solution:
  ∀ (x y : ℝ), 
  2 * x^2 + y^2 + 3 * x * y + 3 * x + y = 2 ↔ 
  (y = -x - 2 ∨ y = -2 * x + 1) := 
by
  sorry

end equation_solution_l74_74748


namespace cats_in_shelter_l74_74933

-- Define the initial conditions
def initial_cats := 20
def monday_addition := 2
def tuesday_addition := 1
def wednesday_subtraction := 3 * 2

-- Problem statement: Prove that the total number of cats after all events is 17
theorem cats_in_shelter : initial_cats + monday_addition + tuesday_addition - wednesday_subtraction = 17 :=
by
  sorry

end cats_in_shelter_l74_74933


namespace three_dice_product_probability_divisible_by_10_l74_74895

open ProbabilityTheory

def die_faces := {1, 2, 3, 4, 5, 6}

def is_divisible_by_10 (n : ℕ) : Prop :=
  (10 ∣ n)

theorem three_dice_product_probability_divisible_by_10 :
  (∑ outcome in
    (Finset.product₃ die_faces die_faces die_faces),
    if is_divisible_by_10 (outcome.1 * outcome.2 * outcome.3) then (1 : ℝ) else 0) / (6 * 6 * 6) = 2 / 3 := 
by sorry

end three_dice_product_probability_divisible_by_10_l74_74895


namespace complement_intersection_l74_74426

open Set

def A := {x : ℝ | x^2 - x - 12 ≤ 0}
def B := {x : ℝ | sqrt x ≤ 2}
def C := {x : ℝ | abs (x + 1) ≤ 2}

theorem complement_intersection (x : ℝ) : x ∈ A \ (B ∩ C) ↔ x ∈ ((Icc (-3 : ℝ) 0).Ioo ∪ (Ioo 1 4)) :=
by {
  sorry,
}

end complement_intersection_l74_74426


namespace g_ln_1_div_2017_l74_74029

open Real

-- Define the functions fulfilling the given conditions
variables (f g : ℝ → ℝ) (a : ℝ)

-- Define assumptions as required by the conditions
axiom f_property : ∀ m n : ℝ, f (m + n) = f m + f n - 1
axiom g_def : ∀ x : ℝ, g x = f x + a^x / (a^x + 1)
axiom a_property : a > 0 ∧ a ≠ 1
axiom g_ln_2017 : g (log 2017) = 2018

-- The theorem to prove
theorem g_ln_1_div_2017 : g (log (1 / 2017)) = -2015 := by
  sorry

end g_ln_1_div_2017_l74_74029


namespace sum_valid_primes_l74_74214

open Nat

theorem sum_valid_primes:
  ∃ (a b c : ℕ)
   (p : ℕ)
   [Fact (Prime p)],
  (a + b + c + 1) % p = 0 ∧
  (a^2 + b^2 + c^2 + 1) % p = 0 ∧
  (a^3 + b^3 + c^3 + 1) % p = 0 ∧
  (a^4 + b^4 + c^4 + 7459) % p = 0 ∧
  p < 1000 →
  p = 2 ∨ p = 3 ∨ p = 13 ∨ p = 41 →
  (2 + 3 + 13 + 41 = 59) := by
  sorry

end sum_valid_primes_l74_74214


namespace rain_on_tuesday_l74_74328

theorem rain_on_tuesday 
  (rain_monday : ℝ)
  (rain_less : ℝ) 
  (h1 : rain_monday = 0.9) 
  (h2 : rain_less = 0.7) : 
  (rain_monday - rain_less) = 0.2 :=
by
  sorry

end rain_on_tuesday_l74_74328


namespace max_elements_X_l74_74054

structure GameState where
  fire : Nat
  stone : Nat
  metal : Nat

def canCreateX (state : GameState) (x : Nat) : Bool :=
  state.metal >= x ∧ state.fire >= 2 * x ∧ state.stone >= 3 * x

def maxCreateX (state : GameState) : Nat :=
  if h : canCreateX state 14 then 14 else 0 -- we would need to show how to actually maximizing the value

theorem max_elements_X : maxCreateX ⟨50, 50, 0⟩ = 14 := 
by 
  -- Proof would go here, showing via the conditions given above
  -- We would need to show no more than 14 can be created given the initial resources
  sorry

end max_elements_X_l74_74054


namespace value_of_a_l74_74587

theorem value_of_a (a x : ℝ) (h : x = 4) (h_eq : x^2 - 3 * x = a^2) : a = 2 ∨ a = -2 :=
by
  -- The proof is omitted, but the theorem statement adheres to the problem conditions and expected result.
  sorry

end value_of_a_l74_74587


namespace hotel_room_mistake_l74_74231

theorem hotel_room_mistake (a b c : ℕ) (ha : 1 ≤ a ∧ a ≤ 9) (hb : 0 ≤ b ∧ b ≤ 9) (hc : 0 ≤ c ∧ c ≤ 9) :
  100 * a + 10 * b + c = (a + 1) * (b + 1) * c → false := by sorry

end hotel_room_mistake_l74_74231


namespace triangle_acute_angle_exists_l74_74423

theorem triangle_acute_angle_exists (a b c d e : ℝ)
  (h_abc : a + b > c ∧ a + c > b ∧ b + c > a)
  (h_abd : a + b > d ∧ a + d > b ∧ b + d > a)
  (h_abe : a + b > e ∧ a + e > b ∧ b + e > a)
  (h_acd : a + c > d ∧ a + d > c ∧ c + d > a)
  (h_ace : a + c > e ∧ a + e > c ∧ c + e > a)
  (h_ade : a + d > e ∧ a + e > d ∧ d + e > a)
  (h_bcd : b + c > d ∧ b + d > c ∧ c + d > b)
  (h_bce : b + c > e ∧ b + e > c ∧ c + e > b)
  (h_bde : b + d > e ∧ b + e > d ∧ d + e > b)
  (h_cde : c + d > e ∧ c + e > d ∧ d + e > c) :
  ∃ x y z, (x = a ∨ x = b ∨ x = c ∨ x = d ∨ x = e) ∧
           (y = a ∨ y = b ∨ y = c ∨ y = d ∨ y = e) ∧
           (z = a ∨ z = b ∨ z = c ∨ z = d ∨ z = e) ∧
           x ≠ y ∧ y ≠ z ∧ z ≠ x ∧
           x + y > z ∧ x + z > y ∧ y + z > x ∧
           (x * x + y * y > z * z ∧ y * y + z * z > x * x ∧ z * z + x * x > y * y) := 
sorry

end triangle_acute_angle_exists_l74_74423


namespace smallest_rectangles_to_cover_square_l74_74981

theorem smallest_rectangles_to_cover_square :
  ∃ n : ℕ, 
    (∃ a : ℕ, a = 3 * 4) ∧
    (∃ k : ℕ, k = lcm 3 4) ∧
    (∃ s : ℕ, s = k * k) ∧
    (s / a = n) ∧
    n = 12 :=
by
  sorry

end smallest_rectangles_to_cover_square_l74_74981


namespace cars_between_15000_and_20000_l74_74404

theorem cars_between_15000_and_20000 
  (total_cars : ℕ)
  (less_than_15000_ratio : ℝ)
  (more_than_20000_ratio : ℝ)
  : less_than_15000_ratio = 0.15 → 
    more_than_20000_ratio = 0.40 → 
    total_cars = 3000 → 
    ∃ (cars_between : ℕ),
      cars_between = total_cars - (less_than_15000_ratio * total_cars + more_than_20000_ratio * total_cars) ∧ 
      cars_between = 1350 :=
by
  sorry

end cars_between_15000_and_20000_l74_74404


namespace probability_heads_given_heads_l74_74501

-- Definitions based on the conditions
inductive Coin
| coin1 | coin2 | coin3

open Coin

def P (event : Set Coin) : ℝ := 
  if event = {coin1} then 1/3
  else if event = {coin2} then 1/3
  else if event = {coin3} then 1/3
  else 0

def event_heads (coin : Coin) : bool :=
  match coin with
  | coin1 => true
  | coin2 => true
  | coin3 => false

-- Probability of showing heads
def P_heads_given_coin (coin : Coin) : ℝ :=
  if event_heads coin then 1 else 0

-- Total probability of getting heads
def P_heads : ℝ := 
  P {coin1} * (if event_heads coin1 then 1 else 0) +
  P {coin2} * (if event_heads coin2 then 1 else 0) +
  P {coin3} * (if event_heads coin3 then 0 else 0)

-- Using Bayes' Theorem to find probability that the coin showing heads is coin2
def P_coin2_given_heads : ℝ :=
  (P_heads_given_coin coin2 * P {coin2}) / P_heads

-- Defining the theorem statement
theorem probability_heads_given_heads : P_coin2_given_heads = 2 / 3 := by
  sorry

end probability_heads_given_heads_l74_74501


namespace biology_marks_l74_74883

theorem biology_marks (english : ℕ) (math : ℕ) (physics : ℕ) (chemistry : ℕ) (average : ℕ) (biology : ℕ) 
  (h1 : english = 36) 
  (h2 : math = 35) 
  (h3 : physics = 42) 
  (h4 : chemistry = 57) 
  (h5 : average = 45) 
  (h6 : (english + math + physics + chemistry + biology) / 5 = average) : 
  biology = 55 := 
by
  sorry

end biology_marks_l74_74883


namespace correct_exponent_calculation_l74_74516

theorem correct_exponent_calculation (x : ℝ) : (-x^3)^4 = x^12 := 
by sorry

end correct_exponent_calculation_l74_74516


namespace cost_per_square_inch_l74_74346

def length : ℕ := 9
def width : ℕ := 12
def total_cost : ℕ := 432

theorem cost_per_square_inch :
  total_cost / ((length * width) / 2) = 8 := 
by 
  sorry

end cost_per_square_inch_l74_74346


namespace train_speed_initial_l74_74855

variable (x : ℝ)
variable (v : ℝ)
variable (average_speed : ℝ := 40 / 3)
variable (initial_distance : ℝ := x)
variable (initial_speed : ℝ := v)
variable (next_distance : ℝ := 4 * x)
variable (next_speed : ℝ := 20)

theorem train_speed_initial : 
  (5 * x) / ((x / v) + (x / 5)) = 40 / 3 → v = 40 / 7 :=
by
  -- Definition of average speed in the context of the problem
  let t1 := x / v
  let t2 := (4 * x) / 20
  let total_distance := 5 * x
  let total_time := t1 + t2
  have avg_speed_eq : total_distance / total_time = 40 / 3 := by sorry
  sorry

end train_speed_initial_l74_74855


namespace find_projection_l74_74717

noncomputable def a : ℝ × ℝ := (-3, 2)
noncomputable def b : ℝ × ℝ := (5, -1)
noncomputable def p : ℝ × ℝ := (21/73, 56/73)
noncomputable def d : ℝ × ℝ := (8, -3)

theorem find_projection :
  ∃ t : ℝ, (t * d.1 - a.1, t * d.2 + a.2) = p ∧
          (p.1 - a.1) * d.1 + (p.2 - a.2) * d.2 = 0 :=
by
  sorry

end find_projection_l74_74717


namespace right_handed_players_total_l74_74968

theorem right_handed_players_total
    (total_players : ℕ)
    (throwers : ℕ)
    (left_handed : ℕ)
    (right_handed : ℕ) :
    total_players = 150 →
    throwers = 60 →
    left_handed = (total_players - throwers) / 2 →
    right_handed = (total_players - throwers) / 2 →
    total_players - throwers = 2 * left_handed →
    left_handed + right_handed + throwers = total_players →
    ∀ throwers : ℕ, throwers = 60 →
    right_handed + throwers = 105 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end right_handed_players_total_l74_74968


namespace cube_divided_by_five_tetrahedrons_l74_74243

-- Define the minimum number of tetrahedrons needed to divide a cube
def min_tetrahedrons_to_divide_cube : ℕ := 5

-- State the theorem
theorem cube_divided_by_five_tetrahedrons : min_tetrahedrons_to_divide_cube = 5 :=
by
  -- The proof is skipped, as instructed
  sorry

end cube_divided_by_five_tetrahedrons_l74_74243


namespace number_of_students_in_third_group_l74_74224

-- Definitions based on given conditions
def students_group1 : ℕ := 9
def students_group2 : ℕ := 10
def tissues_per_box : ℕ := 40
def total_tissues : ℕ := 1200

-- Define the number of students in the third group as a variable
variable {x : ℕ}

-- Prove that the number of students in the third group is 11
theorem number_of_students_in_third_group (h : 360 + 400 + 40 * x = 1200) : x = 11 :=
by sorry

end number_of_students_in_third_group_l74_74224


namespace subset_condition_for_A_B_l74_74315

open Set

theorem subset_condition_for_A_B {a : ℝ} (A B : Set ℝ) 
  (hA : A = {x | abs (x - 2) < a}) 
  (hB : B = {x | x^2 - 2 * x - 3 < 0}) :
  B ⊆ A ↔ 3 ≤ a :=
  sorry

end subset_condition_for_A_B_l74_74315


namespace billboards_color_schemes_is_55_l74_74797

def adjacent_color_schemes (n : ℕ) : ℕ :=
  if h : n = 8 then 55 else 0

theorem billboards_color_schemes_is_55 :
  adjacent_color_schemes 8 = 55 :=
sorry

end billboards_color_schemes_is_55_l74_74797


namespace bouquet_cost_l74_74864

theorem bouquet_cost (c₁ : ℕ) (r₁ r₂ : ℕ) (c_discount : ℕ) (discount_percentage: ℕ) :
  (c₁ = 30) → (r₁ = 15) → (r₂ = 45) → (c_discount = 81) → (discount_percentage = 10) → 
  ((c₂ : ℕ) → (c₂ = (c₁ * r₂) / r₁) → (r₂ > 30) → 
  (c_discount = c₂ - (c₂ * discount_percentage / 100))) → 
  c_discount = 81 :=
by
  intros h1 h2 h3 h4 h5
  subst_vars
  sorry

end bouquet_cost_l74_74864


namespace geometric_prog_105_l74_74967

theorem geometric_prog_105 {a q : ℝ} 
  (h_sum : a + a * q + a * q^2 = 105) 
  (h_arith : a * q - a = (a * q^2 - 15) - a * q) :
  (a = 15 ∧ q = 2) ∨ (a = 60 ∧ q = 0.5) :=
by
  sorry

end geometric_prog_105_l74_74967


namespace p_or_q_then_p_and_q_is_false_l74_74585

theorem p_or_q_then_p_and_q_is_false (p q : Prop) (hpq : p ∨ q) : ¬(p ∧ q) :=
sorry

end p_or_q_then_p_and_q_is_false_l74_74585


namespace cadence_total_earnings_l74_74283

/-- Cadence's earning details over her employment period -/
def cadence_old_company_months := 3 * 12
def cadence_new_company_months := cadence_old_company_months + 5
def cadence_old_company_salary_per_month := 5000
def cadence_salary_increase_rate := 0.20
def cadence_new_company_salary_per_month := 
  cadence_old_company_salary_per_month * (1 + cadence_salary_increase_rate)

def cadence_old_company_earnings := 
  cadence_old_company_months * cadence_old_company_salary_per_month

def cadence_new_company_earnings := 
  cadence_new_company_months * cadence_new_company_salary_per_month
  
def total_earnings := 
  cadence_old_company_earnings + cadence_new_company_earnings

theorem cadence_total_earnings :
  total_earnings = 426000 := 
by
  sorry

end cadence_total_earnings_l74_74283


namespace construct_angle_approx_l74_74351
-- Use a broader import to bring in the entirety of the necessary library

-- Define the problem 
theorem construct_angle_approx (α : ℝ) (m : ℕ) (h : ∃ l : ℕ, (l : ℝ) / 2^m * 90 ≤ α ∧ α ≤ ((l+1) : ℝ) / 2^m * 90) :
  ∃ β : ℝ, β ∈ { β | ∃ l : ℕ, β = (l : ℝ) / 2^m * 90} ∧ |α - β| ≤ 90 / 2^m :=
sorry

end construct_angle_approx_l74_74351


namespace second_odd_integer_l74_74084

theorem second_odd_integer (n : ℤ) (h : (n - 2) + (n + 2) = 128) : n = 64 :=
by
  sorry

end second_odd_integer_l74_74084


namespace weight_of_new_person_l74_74355

theorem weight_of_new_person (avg_increase : ℝ) (num_persons : ℕ) (old_weight : ℝ) (new_weight : ℝ) :
  avg_increase = 2.5 → num_persons = 8 → old_weight = 60 → 
  new_weight = old_weight + num_persons * avg_increase → new_weight = 80 :=
  by
    intros
    sorry

end weight_of_new_person_l74_74355


namespace units_digit_factorial_150_zero_l74_74255

def units_digit (n : ℕ) : ℕ :=
  (nat.factorial n) % 10

theorem units_digit_factorial_150_zero :
  units_digit 150 = 0 :=
sorry

end units_digit_factorial_150_zero_l74_74255


namespace competition_participants_solved_all_three_l74_74771

theorem competition_participants_solved_all_three
  (p1 p2 p3 : ℕ → Prop)
  (total_participants : ℕ)
  (h1 : ∃ n, n = 85 * total_participants / 100 ∧ ∀ k, k < n → p1 k)
  (h2 : ∃ n, n = 80 * total_participants / 100 ∧ ∀ k, k < n → p2 k)
  (h3 : ∃ n, n = 75 * total_participants / 100 ∧ ∀ k, k < n → p3 k) :
  ∃ n, n ≥ 40 * total_participants / 100 ∧ ∀ k, k < n → p1 k ∧ p2 k ∧ p3 k :=
by
  sorry

end competition_participants_solved_all_three_l74_74771


namespace alan_has_5_20_cent_coins_l74_74399

theorem alan_has_5_20_cent_coins
  (a b c : ℕ)
  (h1 : a + b + c = 20)
  (h2 : ((400 - 15 * a - 10 * b) / 5) + 1 = 24) :
  c = 5 :=
by
  sorry

end alan_has_5_20_cent_coins_l74_74399


namespace exists_N_with_N_and_N2_ending_same_l74_74418

theorem exists_N_with_N_and_N2_ending_same : 
  ∃ (N : ℕ), (N > 0) ∧ (N % 100000 = (N*N) % 100000) ∧ (N / 10000 ≠ 0) := sorry

end exists_N_with_N_and_N2_ending_same_l74_74418


namespace symmetric_line_proof_l74_74299

-- Define the given lines
def line_l (x y : ℝ) : Prop := 3 * x - 4 * y + 5 = 0
def axis_of_symmetry (x y : ℝ) : Prop := x + y = 0

-- Define the final symmetric line to be proved
def symmetric_line (x y : ℝ) : Prop := 4 * x - 3 * y - 5 = 0

-- State the theorem
theorem symmetric_line_proof (x y : ℝ) : 
  (line_l (-y) (-x)) → 
  axis_of_symmetry x y → 
  symmetric_line x y := 
sorry

end symmetric_line_proof_l74_74299


namespace total_cost_ice_cream_l74_74062

noncomputable def price_Chocolate : ℝ := 2.50
noncomputable def price_Vanilla : ℝ := 2.00
noncomputable def price_Strawberry : ℝ := 2.25
noncomputable def price_Mint : ℝ := 2.20
noncomputable def price_WaffleCone : ℝ := 1.50
noncomputable def price_ChocolateChips : ℝ := 1.00
noncomputable def price_Fudge : ℝ := 1.25
noncomputable def price_WhippedCream : ℝ := 0.75

def scoops_Pierre : ℕ := 3  -- 2 scoops Chocolate + 1 scoop Mint
def scoops_Mother : ℕ := 4  -- 2 scoops Vanilla + 1 scoop Strawberry + 1 scoop Mint

noncomputable def price_Pierre_BeforeOffer : ℝ :=
  2 * price_Chocolate + price_Mint + price_WaffleCone + price_ChocolateChips

noncomputable def free_Pierre : ℝ := price_Mint -- Mint is the cheapest among Pierre's choices

noncomputable def price_Pierre_AfterOffer : ℝ := price_Pierre_BeforeOffer - free_Pierre

noncomputable def price_Mother_BeforeOffer : ℝ :=
  2 * price_Vanilla + price_Strawberry + price_Mint + price_WaffleCone + price_Fudge + price_WhippedCream

noncomputable def free_Mother : ℝ := price_Vanilla -- Vanilla is the cheapest among Mother's choices

noncomputable def price_Mother_AfterOffer : ℝ := price_Mother_BeforeOffer - free_Mother

noncomputable def total_BeforeDiscount : ℝ := price_Pierre_AfterOffer + price_Mother_AfterOffer

noncomputable def discount_Amount : ℝ := total_BeforeDiscount * 0.15

noncomputable def total_AfterDiscount : ℝ := total_BeforeDiscount - discount_Amount

theorem total_cost_ice_cream : total_AfterDiscount = 14.83 := by
  sorry


end total_cost_ice_cream_l74_74062


namespace like_terms_satisfy_conditions_l74_74572

theorem like_terms_satisfy_conditions (m n : ℤ) (h1 : m - 1 = n) (h2 : m + n = 3) :
  m = 2 ∧ n = 1 := by
  sorry

end like_terms_satisfy_conditions_l74_74572


namespace f_decreasing_on_negative_interval_and_min_value_l74_74039

noncomputable def f : ℝ → ℝ := sorry

-- Define the conditions
def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

def increasing_on_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x ≤ y → y ≤ b → f x ≤ f y

def minimum_value (f : ℝ → ℝ) (m : ℝ) : Prop :=
  ∀ x, f x ≥ m ∧ ∃ x0, f x0 = m

-- Given the conditions
variables (condition1 : even_function f)
          (condition2 : increasing_on_interval f 3 7)
          (condition3 : minimum_value f 2)

-- Prove that f is decreasing on [-7,-3] and minimum value is 2
theorem f_decreasing_on_negative_interval_and_min_value :
  ∀ x y, -7 ≤ x → x ≤ y → y ≤ -3 → f y ≤ f x ∧ minimum_value f 2 :=
sorry

end f_decreasing_on_negative_interval_and_min_value_l74_74039


namespace part1_part2_l74_74724

def A (x y : ℝ) : ℝ := 3 * x ^ 2 + 2 * x * y - 2 * x - 1
def B (x y : ℝ) : ℝ := - x ^ 2 + x * y - 1

theorem part1 (x y : ℝ) : A x y + 3 * B x y = 5 * x * y - 2 * x - 4 := by
  sorry

theorem part2 (y : ℝ) : (∀ x : ℝ, 5 * x * y - 2 * x - 4 = -4) → y = 2 / 5 := by
  sorry

end part1_part2_l74_74724


namespace set_inter_complement_l74_74460

variable (U : Set ℕ) (A : Set ℕ) (B : Set ℕ)

theorem set_inter_complement :
  U = {1, 2, 3, 4, 5, 6, 7} ∧ A = {1, 2, 3, 4} ∧ B = {3, 5, 6} →
  A ∩ (U \ B) = {1, 2, 4} :=
by
  sorry

end set_inter_complement_l74_74460


namespace probability_event_in_single_trial_l74_74188

theorem probability_event_in_single_trial (p : ℝ) 
  (h1 : 0 ≤ p ∧ p ≤ 1) 
  (h2 : (1 - p)^4 = 16 / 81) : 
  p = 1 / 3 :=
sorry

end probability_event_in_single_trial_l74_74188


namespace nonagon_perimeter_l74_74979

theorem nonagon_perimeter :
  (2 + 2 + 3 + 3 + 1 + 3 + 2 + 2 + 2 = 20) := by
  sorry

end nonagon_perimeter_l74_74979


namespace simplify_expression_l74_74566

theorem simplify_expression (x : ℝ) : x * (x * (x * (3 - x) - 5) + 12) + 2 = -x^4 + 3 * x^3 - 5 * x^2 + 12 * x + 2 := by
  sorry

end simplify_expression_l74_74566


namespace boat_speed_in_still_water_l74_74260

theorem boat_speed_in_still_water (V_b : ℝ) (D : ℝ) (V_s : ℝ) 
  (h1 : V_s = 3) 
  (h2 : D = (V_b + V_s) * 1) 
  (h3 : D = (V_b - V_s) * 1.5) : 
  V_b = 15 := 
by 
  sorry

end boat_speed_in_still_water_l74_74260


namespace prime_p_squared_plus_71_divisors_l74_74611

open Nat

def is_prime (n : ℕ) : Prop := Nat.Prime n

def num_distinct_divisors (n : ℕ) : ℕ :=
  (factors n).toFinset.card

theorem prime_p_squared_plus_71_divisors (p : ℕ) (hp : is_prime p) 
  (hdiv : num_distinct_divisors (p ^ 2 + 71) ≤ 10) : p = 2 ∨ p = 3 :=
sorry

end prime_p_squared_plus_71_divisors_l74_74611


namespace point_B_representation_l74_74623

theorem point_B_representation : 
  ∀ A B : ℤ, A = -3 → B = A + 7 → B = 4 := 
by
  intros A B hA hB
  rw hA at hB
  rw hB
  exact rfl

end point_B_representation_l74_74623


namespace projection_equal_p_l74_74118

open Real EuclideanSpace

noncomputable def vector1 : ℝ × ℝ := (-3, 4)
noncomputable def vector2 : ℝ × ℝ := (1, 6)
noncomputable def v : ℝ × ℝ := (4, 2)
noncomputable def p : ℝ × ℝ := (-2.2, 4.4)

theorem projection_equal_p (p_ortho : (p.1 * v.1 + p.2 * v.2) = 0) : p = (4 * (1 / 5) - 3, 2 * (1 / 5) + 4) :=
by
  sorry

end projection_equal_p_l74_74118


namespace reciprocal_self_eq_one_or_neg_one_l74_74037

theorem reciprocal_self_eq_one_or_neg_one (x : ℝ) (h : x = 1 / x) : x = 1 ∨ x = -1 := sorry

end reciprocal_self_eq_one_or_neg_one_l74_74037


namespace arithmetic_mean_reciprocals_first_four_primes_l74_74876

theorem arithmetic_mean_reciprocals_first_four_primes :
  (1/2 + 1/3 + 1/5 + 1/7) / 4 = 247 / 840 :=
by
  sorry

end arithmetic_mean_reciprocals_first_four_primes_l74_74876


namespace find_number_l74_74043

theorem find_number (N : ℕ) (h1 : N / 3 = 8) (h2 : N / 8 = 3) : N = 24 :=
by
  sorry

end find_number_l74_74043


namespace gcd_of_105_1001_2436_l74_74667

noncomputable def gcd_problem : ℕ :=
  Nat.gcd (Nat.gcd 105 1001) 2436

theorem gcd_of_105_1001_2436 : gcd_problem = 7 :=
by {
  sorry
}

end gcd_of_105_1001_2436_l74_74667


namespace negation_of_existential_statement_l74_74080

theorem negation_of_existential_statement {f : ℝ → ℝ} :
  (¬ ∃ x₀ : ℝ, f x₀ < 0) ↔ (∀ x : ℝ, f x ≥ 0) :=
by
  sorry

end negation_of_existential_statement_l74_74080


namespace student_selection_l74_74087

theorem student_selection (a b c : ℕ) (h₁ : a = 3) (h₂ : b = 5) (h₃ : c = 4) : a + b + c = 12 :=
by {
  sorry
}

end student_selection_l74_74087


namespace geometric_series_proof_l74_74607

-- Definitions used in conditions
def a (n : ℕ) (a : ℤ) : ℕ → ℤ := λ n, (∑ i in finset.range n, a ^ i)

-- Theorem statement
theorem geometric_series_proof (a : ℤ) (n1 n2 : ℕ) (h1 : n2 > n1) (h2 : ∀ (p : ℕ), p ∣ (n2 - n1) → prime p → p ∣ (a ^ p - 1)) : 
  ∃ k : ℤ, (a n2 a - a n1 a) / (n2 - n1) = k := 
sorry

end geometric_series_proof_l74_74607


namespace six_box_four_div_three_eight_box_two_div_four_l74_74063

def fills_middle_zero (d : Nat) : Prop :=
  d < 3

def fills_last_zero (d : Nat) : Prop :=
  (80 + d) % 4 = 0

theorem six_box_four_div_three {d : Nat} : fills_middle_zero d → ((600 + d * 10 + 4) / 3) % 100 / 10 = 0 :=
  sorry

theorem eight_box_two_div_four {d : Nat} : fills_last_zero d → ((800 + d * 10 + 2) / 4) % 10 = 0 :=
  sorry

end six_box_four_div_three_eight_box_two_div_four_l74_74063


namespace relay_race_time_reduction_l74_74565

theorem relay_race_time_reduction
    (T T1 T2 T3 T4 T5 : ℝ)
    (h1 : T1 = 0.1 * T)
    (h2 : T2 = 0.2 * T)
    (h3 : T3 = 0.24 * T)
    (h4 : T4 = 0.3 * T)
    (h5 : T5 = 0.16 * T) :
    ((T1 + T2 + T3 + T4 + T5) - (T1 + T2 + T3 + T4 + T5 / 2)) / (T1 + T2 + T3 + T4 + T5) = 0.08 :=
by
  sorry

end relay_race_time_reduction_l74_74565


namespace option_C_is_neither_even_nor_odd_l74_74860

noncomputable def f_A (x : ℝ) : ℝ := x^2 + |x|
noncomputable def f_B (x : ℝ) : ℝ := 2^x - 2^(-x)
noncomputable def f_C (x : ℝ) : ℝ := x^2 - 3^x
noncomputable def f_D (x : ℝ) : ℝ := 1/(x+1) + 1/(x-1)

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = - (f x)

theorem option_C_is_neither_even_nor_odd : ¬ is_even f_C ∧ ¬ is_odd f_C :=
by
  sorry

end option_C_is_neither_even_nor_odd_l74_74860


namespace cookies_in_jar_l74_74654

theorem cookies_in_jar (C : ℕ) (h : C - 1 = (C + 5) / 2) : C = 7 :=
by
  -- Proof goes here
  sorry

end cookies_in_jar_l74_74654


namespace three_digit_number_l74_74143

theorem three_digit_number (m : ℕ) : (300 * m + 10 * m + (m - 1)) = (311 * m - 1) :=
by 
  sorry

end three_digit_number_l74_74143


namespace ratio_of_sides_l74_74537

theorem ratio_of_sides (perimeter_pentagon perimeter_square : ℝ) (hp : perimeter_pentagon = 20) (hs : perimeter_square = 20) : (4:ℝ) / (5:ℝ) = (4:ℝ) / (5:ℝ) :=
by
  sorry

end ratio_of_sides_l74_74537


namespace sector_area_120_deg_radius_3_l74_74905

theorem sector_area_120_deg_radius_3 (r : ℝ) (theta_deg : ℝ) (theta_rad : ℝ) (A : ℝ)
  (h1 : r = 3)
  (h2 : theta_deg = 120)
  (h3 : theta_rad = (2 * Real.pi / 3))
  (h4 : A = (1 / 2) * theta_rad * r^2) :
  A = 3 * Real.pi :=
  sorry

end sector_area_120_deg_radius_3_l74_74905


namespace solution_set_f_x_leq_x_range_of_a_l74_74944

-- Definition of the function f
def f (x : ℝ) : ℝ := |2 * x - 7| + 1

-- Proof Problem for Question (1):
-- Given: f(x) = |2x - 7| + 1
-- Prove: The solution set of the inequality f(x) <= x is {x | 8/3 <= x <= 6}
theorem solution_set_f_x_leq_x :
  { x : ℝ | f x ≤ x } = { x : ℝ | 8 / 3 ≤ x ∧ x ≤ 6 } :=
sorry

-- Definition of the function g
def g (x : ℝ) : ℝ := f x - 2 * |x - 1|

-- Proof Problem for Question (2):
-- Given: f(x) = |2x - 7| + 1 and g(x) = f(x) - 2 * |x - 1|
-- Prove: If ∃ x, g(x) <= a, then a >= -4
theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, g x ≤ a) → a ≥ -4 :=
sorry

end solution_set_f_x_leq_x_range_of_a_l74_74944


namespace problem_solution_l74_74458

noncomputable def S : ℝ :=
  1 / (4 - Real.sqrt 15) + 1 / (Real.sqrt 15 - Real.sqrt 14) - 1 / (Real.sqrt 14 - Real.sqrt 13) + 1 / (Real.sqrt 13 - 3)

theorem problem_solution : S = 7 := 
by
  sorry

end problem_solution_l74_74458


namespace students_with_equal_scores_l74_74806

theorem students_with_equal_scores 
  (n : ℕ)
  (scores : Fin n → Fin (n - 1)): 
  ∃ i j : Fin n, i ≠ j ∧ scores i = scores j := 
by 
  sorry

end students_with_equal_scores_l74_74806


namespace equation_solution_l74_74751

theorem equation_solution:
  ∀ (x y : ℝ), 
  2 * x^2 + y^2 + 3 * x * y + 3 * x + y = 2 ↔ 
  (y = -x - 2 ∨ y = -2 * x + 1) := 
by
  sorry

end equation_solution_l74_74751


namespace cats_in_shelter_l74_74932

-- Define the initial conditions
def initial_cats := 20
def monday_addition := 2
def tuesday_addition := 1
def wednesday_subtraction := 3 * 2

-- Problem statement: Prove that the total number of cats after all events is 17
theorem cats_in_shelter : initial_cats + monday_addition + tuesday_addition - wednesday_subtraction = 17 :=
by
  sorry

end cats_in_shelter_l74_74932


namespace largest_number_with_two_moves_l74_74945

theorem largest_number_with_two_moves (n : Nat) (matches_limit : Nat) (initial_number : Nat)
  (h_n : initial_number = 1405) (h_limit: matches_limit = 2) : n = 7705 :=
by
  sorry

end largest_number_with_two_moves_l74_74945


namespace distance_is_absolute_value_l74_74473

noncomputable def distance_to_origin (x : ℝ) : ℝ := |x|

theorem distance_is_absolute_value (x : ℝ) : distance_to_origin x = |x| :=
by
  sorry

end distance_is_absolute_value_l74_74473


namespace snakes_in_breeding_ball_l74_74616

theorem snakes_in_breeding_ball (x : ℕ) (h : 3 * x + 12 = 36) : x = 8 :=
by sorry

end snakes_in_breeding_ball_l74_74616


namespace greatest_possible_value_l74_74371

theorem greatest_possible_value (x : ℝ) : 
  (∃ (k : ℝ), k = (5 * x - 25) / (4 * x - 5) ∧ k^2 + k = 20) → x ≤ 2 := 
sorry

end greatest_possible_value_l74_74371


namespace solve_for_x_l74_74218

theorem solve_for_x (y z x : ℝ) (h1 : 2 / 3 = y / 90) (h2 : 2 / 3 = (y + z) / 120) (h3 : 2 / 3 = (x - z) / 150) : x = 120 :=
by
  sorry

end solve_for_x_l74_74218


namespace total_students_in_middle_school_l74_74005

/-- Given that 20% of the students are in the band and there are 168 students in the band,
    prove that the total number of students in the middle school is 840. -/
theorem total_students_in_middle_school (total_students : ℕ) (band_students : ℕ) 
  (h1 : 20 ≤ 100)
  (h2 : band_students = 168)
  (h3 : band_students = 20 * total_students / 100) 
  : total_students = 840 :=
sorry

end total_students_in_middle_school_l74_74005


namespace fraction_eval_l74_74881

theorem fraction_eval : 
  ((12^4 + 324) * (24^4 + 324) * (36^4 + 324)) / 
  ((6^4 + 324) * (18^4 + 324) * (30^4 + 324)) = (84 / 35) :=
by
  sorry

end fraction_eval_l74_74881


namespace cube_surface_area_example_l74_74814

def cube_surface_area (V : ℝ) (S : ℝ) : Prop :=
  (∃ s : ℝ, s ^ 3 = V ∧ S = 6 * s ^ 2)

theorem cube_surface_area_example : cube_surface_area 8 24 :=
by
  sorry

end cube_surface_area_example_l74_74814


namespace find_number_l74_74664

theorem find_number (n : ℝ) : (4 / 3) * n = 48 → n = 36 :=
by
  intro h
  have h1 : n = 48 * (3 / 4) := by
    sorry
  exact h1

end find_number_l74_74664


namespace shorter_piece_length_l74_74379

-- Definitions for the conditions
def total_length : ℕ := 70
def ratio (short long : ℕ) : Prop := long = (5 * short) / 2

-- The proof problem statement
theorem shorter_piece_length (x : ℕ) (h1 : total_length = x + (5 * x) / 2) : x = 20 :=
sorry

end shorter_piece_length_l74_74379


namespace num_valid_pairs_l74_74754

theorem num_valid_pairs : ∃ (n : ℕ), n = 8 ∧ (∀ (a b : ℕ), 0 < a ∧ 0 < b ∧ a + b ≤ 150 ∧ ((a + 1 / b) / (1 / a + b) = 17) ↔ (a = 17 * b) ∧ b ≤ 8) :=
by
  sorry

end num_valid_pairs_l74_74754


namespace area_of_triangle_ABC_l74_74048

theorem area_of_triangle_ABC (A B C : ℝ) (a b c : ℝ) 
  (h1 : b = 2) (h2 : c = 3) (h3 : C = 2 * B): 
  ∃ S : ℝ, S = 1/2 * b * c * (Real.sin A) ∧ S = 15 * (Real.sqrt 7) / 16 :=
by
  sorry

end area_of_triangle_ABC_l74_74048


namespace pipe_q_fills_cistern_in_15_minutes_l74_74239

theorem pipe_q_fills_cistern_in_15_minutes :
  ∃ T : ℝ, 
    (1/12 * 2 + 1/T * 2 + 1/T * 10.5 = 1) → 
    T = 15 :=
by {
  -- Assume the conditions and derive T = 15
  sorry
}

end pipe_q_fills_cistern_in_15_minutes_l74_74239


namespace range_of_m_l74_74920

theorem range_of_m (m : ℝ) : 
  (∃ x : ℝ, |x - 1| + |x + m| ≤ 4) ↔ -5 ≤ m ∧ m ≤ 3 :=
by
  sorry

end range_of_m_l74_74920


namespace prime_factor_of_sum_of_four_consecutive_integers_l74_74097

-- Define four consecutive integers and their sum
def sum_four_consecutive_integers (n : ℤ) : ℤ := (n - 1) + n + (n + 1) + (n + 2)

-- The theorem states that 2 is a divisor of the sum of any four consecutive integers
theorem prime_factor_of_sum_of_four_consecutive_integers (n : ℤ) : 
  ∃ p : ℤ, Prime p ∧ p ∣ sum_four_consecutive_integers n :=
begin
  use 2,
  split,
  {
    apply Prime_two,
  },
  {
    unfold sum_four_consecutive_integers,
    norm_num,
    exact dvd.intro (2 * n + 1) rfl,
  },
end

end prime_factor_of_sum_of_four_consecutive_integers_l74_74097


namespace smallest_common_multiple_of_8_and_6_l74_74509

theorem smallest_common_multiple_of_8_and_6 : ∃ n : ℕ, n > 0 ∧ (8 ∣ n) ∧ (6 ∣ n) ∧ ∀ m : ℕ, (m > 0 ∧ (8 ∣ m) ∧ (6 ∣ m)) → n ≤ m :=
by
  sorry

end smallest_common_multiple_of_8_and_6_l74_74509


namespace solve_Diamond_l74_74183

theorem solve_Diamond :
  ∀ (Diamond : ℕ), (Diamond * 7 + 4 = Diamond * 8 + 1) → Diamond = 3 :=
by
  intros Diamond h
  sorry

end solve_Diamond_l74_74183


namespace kaleb_savings_l74_74053

theorem kaleb_savings (x : ℕ) (h : x + 25 = 8 * 8) : x = 39 := 
by
  sorry

end kaleb_savings_l74_74053


namespace ones_digit_power_sum_l74_74823

noncomputable def ones_digit_of_power_sum_is_5 : Prop :=
  (1^2010 + 2^2010 + 3^2010 + 4^2010 + 5^2010 + 6^2010 + 7^2010 + 8^2010 + 9^2010 + 10^2010) % 10 = 5

theorem ones_digit_power_sum : ones_digit_of_power_sum_is_5 :=
  sorry

end ones_digit_power_sum_l74_74823


namespace problem_l74_74056

theorem problem (p q r : ℂ)
  (h1 : p + q + r = 0)
  (h2 : p * q + q * r + r * p = -2)
  (h3 : p * q * r = 2)
  (hp : p ^ 3 = 2 * p + 2)
  (hq : q ^ 3 = 2 * q + 2)
  (hr : r ^ 3 = 2 * r + 2) :
  p * (q - r) ^ 2 + q * (r - p) ^ 2 + r * (p - q) ^ 2 = -18 := by
  sorry

end problem_l74_74056


namespace part1_part2_l74_74735

noncomputable def f (a x : ℝ) : ℝ := x^2 + (a+1)*x + a

theorem part1 (a : ℝ) :
  (∀ x : ℝ, 1 < x ∧ x < 2 → f a x < 0) → a ≤ -2 := sorry

theorem part2 (a x : ℝ) :
  f a x > 0 ↔
  (a > 1 ∧ (x < -a ∨ x > -1)) ∨
  (a = 1 ∧ x ≠ -1) ∨
  (a < 1 ∧ (x < -1 ∨ x > -a)) := sorry

end part1_part2_l74_74735


namespace john_books_per_day_l74_74935

theorem john_books_per_day (books_per_week := 2) (weeks := 6) (total_books := 48) :
  (total_books / (books_per_week * weeks) = 4) :=
by
  sorry

end john_books_per_day_l74_74935


namespace candy_problem_l74_74467

theorem candy_problem (N S a : ℕ) (h1 : a = S - a - 7) (h2 : a > 1) : S = 21 := 
sorry

end candy_problem_l74_74467


namespace distinct_convex_quadrilaterals_l74_74901

open Nat

noncomputable def combinations (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem distinct_convex_quadrilaterals (n : ℕ) (h : n > 4) 
  (no_three_collinear : ℕ → Prop) :
  ∃ k, k ≥ combinations n 5 / (n - 4) :=
by
  sorry

end distinct_convex_quadrilaterals_l74_74901


namespace university_committee_count_l74_74407

theorem university_committee_count :
  let departments := ["math", "stats", "cs"],
      males_per_dept := 3,
      females_per_dept := 3,
      total_males := 3 * 3,
      total_females := 3 * 3,
      total_committee_men := 4,
      total_committee_women := 2,
      total_committee_size := total_committee_men + total_committee_women,
      choose := Nat.choose in
  (departments.choose 2).length = 3
  ∧ (choose 3 2 = 3)
  ∧ (choose 3 1 = 3)
  ∧ (choose 3 3 = 1)
  ∧ ((3 * 3 * 3 * 3 * 2) + (3 * 9 * 9) = 891) :=
by
  sorry

end university_committee_count_l74_74407


namespace value_at_2_l74_74257

def f (x : ℝ) : ℝ := x^3 - 3 * x^2 + 2 * x

theorem value_at_2 : f 2 = 0 := by
  sorry

end value_at_2_l74_74257


namespace smallest_common_multiple_8_6_l74_74513

theorem smallest_common_multiple_8_6 : ∃ n : ℕ, n > 0 ∧ (8 ∣ n) ∧ (6 ∣ n) ∧ ∀ m : ℕ, m > 0 ∧ (8 ∣ m) ∧ (6 ∣ m) → n ≤ m :=
begin
  use 24,
  split,
  { exact zero_lt_24, },
  split,
  { exact dvd.intro 3 rfl, },
  split,
  { exact dvd.intro 4 rfl, },
  intros m hm h8 h6,
  -- actual proof here
  sorry
end

end smallest_common_multiple_8_6_l74_74513


namespace lesser_fraction_l74_74085

theorem lesser_fraction 
  (x y : ℚ)
  (h_sum : x + y = 13 / 14)
  (h_prod : x * y = 1 / 5) :
  min x y = 87 / 700 := sorry

end lesser_fraction_l74_74085


namespace right_rect_prism_volume_l74_74816

theorem right_rect_prism_volume (a b c : ℝ) 
  (h1 : a * b = 56) 
  (h2 : b * c = 63) 
  (h3 : a * c = 36) : 
  a * b * c = 504 := by
  sorry

end right_rect_prism_volume_l74_74816


namespace probability_multiple_of_7_condition_l74_74238

theorem probability_multiple_of_7_condition :
  ∃ (a b : ℕ), 1 ≤ a ∧ a ≤ 100 ∧ 1 ≤ b ∧ b ≤ 100 ∧ a ≠ b ∧ (ab + a + b + 1) % 7 = 0 → 
  (1295 / 4950 = 259 / 990) :=
sorry

end probability_multiple_of_7_condition_l74_74238


namespace route_Y_quicker_than_route_X_l74_74209

theorem route_Y_quicker_than_route_X :
    let dist_X := 9  -- distance of Route X in miles
    let speed_X := 45  -- speed of Route X in miles per hour
    let dist_Y := 8  -- total distance of Route Y in miles
    let normal_dist_Y := 6.5  -- normal speed distance of Route Y in miles
    let construction_dist_Y := 1.5  -- construction zone distance of Route Y in miles
    let normal_speed_Y := 50  -- normal speed of Route Y in miles per hour
    let construction_speed_Y := 25  -- construction zone speed of Route Y in miles per hour
    let time_X := (dist_X / speed_X) * 60  -- time for Route X in minutes
    let time_Y1 := (normal_dist_Y / normal_speed_Y) * 60  -- time for normal speed segment of Route Y in minutes
    let time_Y2 := (construction_dist_Y / construction_speed_Y) * 60  -- time for construction zone segment of Route Y in minutes
    let time_Y := time_Y1 + time_Y2  -- total time for Route Y in minutes
    time_X - time_Y = 0.6 :=  -- the difference in time between Route X and Route Y in minutes
by
  sorry

end route_Y_quicker_than_route_X_l74_74209


namespace vacation_cost_in_usd_l74_74884

theorem vacation_cost_in_usd :
  let n := 7
  let rent_per_person_eur := 65
  let transport_per_person_usd := 25
  let food_per_person_gbp := 50
  let activities_per_person_jpy := 2750
  let eur_to_usd := 1.20
  let gbp_to_usd := 1.40
  let jpy_to_usd := 0.009
  let total_rent_usd := n * rent_per_person_eur * eur_to_usd
  let total_transport_usd := n * transport_per_person_usd
  let total_food_usd := n * food_per_person_gbp * gbp_to_usd
  let total_activities_usd := n * activities_per_person_jpy * jpy_to_usd
  let total_cost_usd := total_rent_usd + total_transport_usd + total_food_usd + total_activities_usd
  total_cost_usd = 1384.25 := by
    sorry

end vacation_cost_in_usd_l74_74884


namespace find_f_six_minus_a_l74_74431

noncomputable def f (x : ℝ) : ℝ :=
if x ≤ 1 then 2^(x-2) - 2 else -Real.logb 2 (x + 1)

variable (a : ℝ)
axiom h : f a = -3

theorem find_f_six_minus_a : f (6 - a) = - 15 / 8 :=
by
  sorry

end find_f_six_minus_a_l74_74431


namespace range_of_a_l74_74834

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x^2 - 2 * x + 5 ≥ a^2 - 3 * a) → (-1 ≤ a ∧ a ≤ 4) :=
by
  sorry

end range_of_a_l74_74834


namespace olivia_money_left_l74_74948

-- Defining hourly wages
def wage_monday : ℕ := 10
def wage_wednesday : ℕ := 12
def wage_friday : ℕ := 14
def wage_saturday : ℕ := 20

-- Defining hours worked each day
def hours_monday : ℕ := 5
def hours_wednesday : ℕ := 4
def hours_friday : ℕ := 3
def hours_saturday : ℕ := 2

-- Defining business-related expenses and tax rate
def expenses : ℕ := 50
def tax_rate : ℝ := 0.15

-- Calculate total earnings
def total_earnings : ℕ :=
  (hours_monday * wage_monday) +
  (hours_wednesday * wage_wednesday) +
  (hours_friday * wage_friday) +
  (hours_saturday * wage_saturday)

-- Earnings after expenses
def earnings_after_expenses : ℕ :=
  total_earnings - expenses

-- Calculate tax amount
def tax_amount : ℝ :=
  tax_rate * (total_earnings : ℝ)

-- Final amount Olivia has left
def remaining_amount : ℝ :=
  (earnings_after_expenses : ℝ) - tax_amount

theorem olivia_money_left : remaining_amount = 103 := by
  sorry

end olivia_money_left_l74_74948


namespace tourist_groupings_l74_74241

theorem tourist_groupings :
  (∑ i in finset.range(7) \ finset.singleton(0), nat.choose 8 i) = 246 :=
by sorry

end tourist_groupings_l74_74241


namespace value_of_B_l74_74915

theorem value_of_B (x y : ℕ) (h1 : x > y) (h2 : y > 1) (h3 : x * y = x + y + 22) :
  (x / y) = 12 :=
sorry

end value_of_B_l74_74915


namespace sum_of_powers_eq_zero_l74_74943

theorem sum_of_powers_eq_zero
  (a b c : ℝ)
  (n : ℝ)
  (h1 : a + b + c = 0)
  (h2 : a^3 + b^3 + c^3 = 0) :
  a^(2* ⌊n⌋ + 1) + b^(2* ⌊n⌋ + 1) + c^(2* ⌊n⌋ + 1) = 0 := by
  sorry

end sum_of_powers_eq_zero_l74_74943


namespace asymptotes_of_hyperbola_l74_74078

-- Definition of hyperbola
def hyperbola_eq (x y : ℝ) : Prop :=
  (x^2 / 4) - y^2 = 1

-- The main theorem to prove
theorem asymptotes_of_hyperbola (x y : ℝ) :
  hyperbola_eq x y → (y = (1/2) * x ∨ y = -(1/2) * x) :=
by 
  sorry

end asymptotes_of_hyperbola_l74_74078


namespace ratio_a3_a2_l74_74320

theorem ratio_a3_a2 (a_0 a_1 a_2 a_3 a_4 a_5 : ℤ) (x : ℝ)
  (h : (1 - 2 * x)^5 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5) :
  a_3 / a_2 = -2 :=
sorry

end ratio_a3_a2_l74_74320


namespace cookies_in_jar_l74_74656

theorem cookies_in_jar (C : ℕ) (h : C - 1 = (C + 5) / 2) : C = 7 :=
by
  -- Proof goes here
  sorry

end cookies_in_jar_l74_74656


namespace classics_section_books_l74_74227

-- Define the number of authors
def num_authors : Nat := 6

-- Define the number of books per author
def books_per_author : Nat := 33

-- Define the total number of books
def total_books : Nat := num_authors * books_per_author

-- Prove that the total number of books is 198
theorem classics_section_books : total_books = 198 := by
  sorry

end classics_section_books_l74_74227


namespace calc_value_l74_74596

theorem calc_value (a b : ℝ) (h : b = 3 * a - 2) : 2 * b - 6 * a + 2 = -2 := 
by 
  sorry

end calc_value_l74_74596


namespace point_B_is_4_l74_74622

def point_A : ℤ := -3
def units_to_move : ℤ := 7
def point_B : ℤ := point_A + units_to_move

theorem point_B_is_4 : point_B = 4 :=
by
  sorry

end point_B_is_4_l74_74622


namespace value_of_x_squared_minus_y_squared_l74_74445

theorem value_of_x_squared_minus_y_squared (x y : ℝ) 
  (h₁ : x + y = 20) 
  (h₂ : x - y = 6) :
  x^2 - y^2 = 120 := 
by 
  sorry

end value_of_x_squared_minus_y_squared_l74_74445


namespace problem_l74_74761

theorem problem (x : ℝ) (h : x^2 + 5 * x - 990 = 0) : x^3 + 6 * x^2 - 985 * x + 1012 = 2002 :=
sorry

end problem_l74_74761


namespace start_page_day2_correct_l74_74388

variables (total_pages : ℕ) (percentage_read_day1 : ℝ) (start_page_day2 : ℕ)

theorem start_page_day2_correct
  (h1 : total_pages = 200)
  (h2 : percentage_read_day1 = 0.2)
  : start_page_day2 = total_pages * percentage_read_day1 + 1 :=
by
  sorry

end start_page_day2_correct_l74_74388


namespace find_c_solution_l74_74417

theorem find_c_solution {c : ℚ} 
  (h₁ : ∃ x : ℤ, 2 * (x : ℚ)^2 + 17 * x - 55 = 0 ∧ x = ⌊c⌋)
  (h₂ : ∃ x : ℚ, 6 * x^2 - 23 * x + 7 = 0 ∧ 0 ≤ x ∧ x < 1 ∧ x = c - ⌊c⌋) :
  c = -32 / 3 :=
by
  sorry

end find_c_solution_l74_74417


namespace solve_equation_l74_74738

theorem solve_equation (x y : ℝ) (h : 2 * x^2 + y^2 + 3 * x * y + 3 * x + y = 2):
  y = -x - 2 ∨ y = -2 * x + 1 :=
sorry


end solve_equation_l74_74738


namespace no_integer_solution_for_equation_l74_74951

theorem no_integer_solution_for_equation :
  ¬ ∃ (x y : ℤ), x^2 + 3 * x * y - 2 * y^2 = 122 :=
sorry

end no_integer_solution_for_equation_l74_74951


namespace total_packs_of_groceries_is_14_l74_74468

-- Define the number of packs of cookies
def packs_of_cookies : Nat := 2

-- Define the number of packs of cakes
def packs_of_cakes : Nat := 12

-- Define the total packs of groceries as the sum of packs of cookies and cakes
def total_packs_of_groceries : Nat := packs_of_cookies + packs_of_cakes

-- The theorem which states that the total packs of groceries is 14
theorem total_packs_of_groceries_is_14 : total_packs_of_groceries = 14 := by
  -- this is where the proof would go
  sorry

end total_packs_of_groceries_is_14_l74_74468


namespace order_of_6_l74_74184

def f (x : ℤ) : ℤ := (x^2) % 13

theorem order_of_6 :
  ∀ n : ℕ, (∀ k < n, f^[k] 6 ≠ 6) → f^[n] 6 = 6 → n = 72 :=
by
  sorry

end order_of_6_l74_74184


namespace hilary_big_toenails_count_l74_74752

def fit_toenails (total_capacity : ℕ) (big_toenail_space_ratio : ℕ) (current_regular : ℕ) (additional_regular : ℕ) : ℕ :=
  (total_capacity - (current_regular + additional_regular)) / big_toenail_space_ratio

theorem hilary_big_toenails_count :
  fit_toenails 100 2 40 20 = 10 :=
  by
    sorry

end hilary_big_toenails_count_l74_74752


namespace count_valid_n_l74_74317

theorem count_valid_n :
  let n_values := [50, 550, 1050, 1550, 2050]
  ( ∀ n : ℤ, (50 * ((n + 500) / 50) - 500 = n) ∧ (Int.floor (Real.sqrt (2 * n : ℝ)) = (n + 500) / 50) → n ∈ n_values ) ∧
  ((∀ n : ℤ, ∃ k : ℤ, (n = 50 * k - 500) ∧ (k = Int.floor (Real.sqrt (2 * (50 * k - 500) : ℝ))) ∧ 0 < n ) → n_values.length = 5) :=
by
  sorry

end count_valid_n_l74_74317


namespace h_at_8_l74_74205

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x + 2

noncomputable def h (x : ℝ) : ℝ :=
  let a := 1
  let b := 1
  let c := 2
  (1/2) * (x - a^3) * (x - b^3) * (x - c^3)

theorem h_at_8 : h 8 = 147 := 
by 
  sorry

end h_at_8_l74_74205


namespace probability_james_wins_l74_74195

open Classical

def outcomes : Finset (ℕ × ℕ) := 
  Finset.product (Finset.range 6).succ (Finset.range 6).succ

def winning_pairs (x y : ℕ) : Prop := 
  abs (x - y) ≤ 2

def winning_outcomes : Finset (ℕ × ℕ) := 
  outcomes.filter (λ p, winning_pairs p.1 p.2)

theorem probability_james_wins : 
  winning_outcomes.card / outcomes.card = 2 / 3 :=
sorry

end probability_james_wins_l74_74195


namespace contractor_work_done_l74_74268

def initial_people : ℕ := 10
def remaining_people : ℕ := 8
def total_days : ℕ := 100
def remaining_days : ℕ := 75
def fraction_done : ℚ := 1/4
def total_work : ℚ := 1

theorem contractor_work_done (x : ℕ) 
  (h1 : initial_people * x = fraction_done * total_work) 
  (h2 : remaining_people * remaining_days = (1 - fraction_done) * total_work) :
  x = 60 :=
by
  sorry

end contractor_work_done_l74_74268


namespace number_of_distinct_products_l74_74753

   -- We define the set S
   def S : Finset ℕ := {2, 3, 5, 11, 13}

   -- We define what it means to have a distinct product of two or more elements
   def distinctProducts (s : Finset ℕ) : Finset ℕ :=
     (s.powerset.filter (λ t => 2 ≤ t.card)).image (λ t => t.prod id)

   -- We state the theorem that there are exactly 26 distinct products
   theorem number_of_distinct_products : (distinctProducts S).card = 26 :=
   sorry
   
end number_of_distinct_products_l74_74753


namespace cylinder_volume_ratio_l74_74127

theorem cylinder_volume_ratio (r1 r2 V1 V2 : ℝ) (h1 : 2 * Real.pi * r1 = 6) (h2 : 2 * Real.pi * r2 = 10) (hV1 : V1 = Real.pi * r1^2 * 10) (hV2 : V2 = Real.pi * r2^2 * 6) :
  V1 < V2 → (V2 / V1) = 5 / 3 :=
by
  sorry

end cylinder_volume_ratio_l74_74127


namespace point_to_focus_distance_l74_74313

def parabola : Set (ℝ × ℝ) := { p | p.2^2 = 4 * p.1 }

def point_P : ℝ × ℝ := (3, 2) -- Since y^2 = 4*3 hence y = ±2 and we choose one of the (3, 2) or (3, -2)

def focus_F : ℝ × ℝ := (1, 0) -- Focus of y^2 = 4x is (1, 0)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem point_to_focus_distance : distance point_P focus_F = 4 := by
  sorry -- Proof goes here

end point_to_focus_distance_l74_74313


namespace candle_length_sum_l74_74012

theorem candle_length_sum (l s : ℕ) (x : ℤ) 
  (h1 : l = s + 32)
  (h2 : s = (5 * x)) 
  (h3 : l = (7 * (3 * x))) :
  l + s = 52 := 
sorry

end candle_length_sum_l74_74012


namespace age_condition_l74_74731

theorem age_condition (x y z : ℕ) (h1 : x > y) : 
  (z > y) ↔ (y + z > 2 * x) ∧ (∀ x y z, y + z > 2 * x → z > y) := sorry

end age_condition_l74_74731


namespace erased_number_is_30_l74_74997

-- Definitions based on conditions
def consecutiveNumbers (start n : ℕ) : List ℕ :=
  List.range' start n

def erase (l : List ℕ) (x : ℕ) : List ℕ :=
  List.filter (λ y => y ≠ x) l

def average (l : List ℕ) : ℚ :=
  l.sum / l.length

-- Statement to prove
theorem erased_number_is_30 :
  ∃ n x, average (erase (consecutiveNumbers 11 n) x) = 23 ∧ x = 30 := by
  sorry

end erased_number_is_30_l74_74997


namespace sum_of_lengths_of_legs_of_larger_triangle_l74_74240

theorem sum_of_lengths_of_legs_of_larger_triangle
  (area_small : ℝ) (area_large : ℝ) (hypo_small : ℝ)
  (h_area_small : area_small = 18) (h_area_large : area_large = 288) (h_hypo_small : hypo_small = 10) :
  ∃ (sum_legs_large : ℝ), sum_legs_large = 52 :=
by
  sorry

end sum_of_lengths_of_legs_of_larger_triangle_l74_74240


namespace solve_equation_l74_74067

noncomputable def equation (x : ℝ) : Prop :=
  2021 * x = 2022 * x ^ (2021 / 2022) - 1

theorem solve_equation : ∀ x : ℝ, equation x ↔ x = 1 :=
by
  intro x
  sorry

end solve_equation_l74_74067


namespace number_of_boxes_l74_74137

-- Define the given conditions
def total_chocolates : ℕ := 442
def chocolates_per_box : ℕ := 26

-- Prove the number of small boxes in the large box
theorem number_of_boxes : (total_chocolates / chocolates_per_box) = 17 := by
  sorry

end number_of_boxes_l74_74137


namespace monotonically_increasing_l74_74949

variable {R : Type} [LinearOrderedField R]

def f (x : R) : R := 3 * x + 1

theorem monotonically_increasing : ∀ x₁ x₂ : R, x₁ < x₂ → f x₁ < f x₂ :=
by
  intro x₁ x₂ h
 -- this is where the proof would go
  sorry

end monotonically_increasing_l74_74949


namespace largest_prime_factor_13231_l74_74163

-- Define the conditions
def is_prime (n : ℕ) : Prop := ∀ k : ℕ, k ∣ n → k = 1 ∨ k = n

-- State the problem as a theorem in Lean 4
theorem largest_prime_factor_13231 (H1 : 13231 = 121 * 109) 
    (H2 : is_prime 109)
    (H3 : 121 = 11^2) :
    ∃ p, is_prime p ∧ p ∣ 13231 ∧ ∀ q, is_prime q ∧ q ∣ 13231 → q ≤ p :=
by
  sorry

end largest_prime_factor_13231_l74_74163


namespace right_triangle_hypotenuse_l74_74395

theorem right_triangle_hypotenuse (a b c : ℝ) 
  (h₁ : a + b + c = 40) 
  (h₂ : a * b = 60) 
  (h₃ : a^2 + b^2 = c^2) : c = 18.5 := 
by 
  sorry

end right_triangle_hypotenuse_l74_74395


namespace christen_potatoes_and_total_time_l74_74032

-- Variables representing the given conditions
variables (homer_rate : ℕ) (christen_rate : ℕ) (initial_potatoes : ℕ) 
(homer_time_alone : ℕ) (total_time : ℕ)

-- Specific values for the given problem
def homerRate := 4
def christenRate := 6
def initialPotatoes := 60
def homerTimeAlone := 5

-- Function to calculate the number of potatoes peeled by Homer alone
def potatoesPeeledByHomerAlone :=
  homerRate * homerTimeAlone

-- Function to calculate the number of remaining potatoes
def remainingPotatoes :=
  initialPotatoes - potatoesPeeledByHomerAlone

-- Function to calculate the total peeling rate when Homer and Christen are working together
def combinedRate :=
  homerRate + christenRate

-- Function to calculate the time taken to peel the remaining potatoes
def timePeelingTogether :=
  remainingPotatoes / combinedRate

-- Function to calculate the total time spent peeling potatoes
def totalTime :=
  homerTimeAlone + timePeelingTogether

-- Function to calculate the number of potatoes peeled by Christen
def potatoesPeeledByChristen :=
  christenRate * timePeelingTogether

/- The theorem to be proven: Christen peeled 24 potatoes, and it took 9 minutes to peel all the potatoes. -/
theorem christen_potatoes_and_total_time :
  (potatoesPeeledByChristen = 24) ∧ (totalTime = 9) :=
by {
  sorry
}

end christen_potatoes_and_total_time_l74_74032


namespace projectile_height_35_l74_74995

noncomputable def projectile_height (t : ℝ) : ℝ := -4.9 * t^2 + 30 * t

theorem projectile_height_35 (t : ℝ) :
  projectile_height t = 35 ↔ t = 10/7 :=
by {
  sorry
}

end projectile_height_35_l74_74995


namespace rain_at_least_one_day_l74_74302

-- Define the probabilities
def P_A1 : ℝ := 0.30
def P_A2 : ℝ := 0.40
def P_A2_given_A1 : ℝ := 0.70

-- Define complementary probabilities
def P_not_A1 : ℝ := 1 - P_A1
def P_not_A2 : ℝ := 1 - P_A2
def P_not_A2_given_A1 : ℝ := 1 - P_A2_given_A1

-- Calculate probabilities of no rain on both days under different conditions
def P_no_rain_both_days_if_no_rain_first : ℝ := P_not_A1 * P_not_A2
def P_no_rain_both_days_if_rain_first : ℝ := P_A1 * P_not_A2_given_A1

-- Total probability of no rain on both days
def P_no_rain_both_days : ℝ := P_no_rain_both_days_if_no_rain_first + P_no_rain_both_days_if_rain_first

-- Probability of rain on at least one of the two days
def P_rain_one_or_more_days : ℝ := 1 - P_no_rain_both_days

-- Expressing the result as a percentage
def result_percentage : ℝ := P_rain_one_or_more_days * 100

-- Theorem statement
theorem rain_at_least_one_day : result_percentage = 49 := by
  -- We skip the proof
  sorry

end rain_at_least_one_day_l74_74302


namespace scale_model_height_is_correct_l74_74859

noncomputable def height_of_scale_model (h_real : ℝ) (V_real : ℝ) (V_scale : ℝ) : ℝ :=
  h_real / (V_real / V_scale)^(1/3:ℝ)

theorem scale_model_height_is_correct :
  height_of_scale_model 90 500000 0.2 = 0.66 :=
by
  sorry

end scale_model_height_is_correct_l74_74859


namespace perimeter_less_than_1_km_perimeter_less_than_1_km_zero_thickness_perimeter_to_area_ratio_l74_74837

section Problem

-- Definitions based on the problem conditions

-- Condition: Side length of each square is 1 cm
def side_length : ℝ := 1

-- Condition: Thickness of the nail for parts a) and b)
def nail_thickness_a := 0.1
def nail_thickness_b := 0

-- Given a perimeter P and area S, the perimeter cannot exceed certain thresholds based on problem analysis

theorem perimeter_less_than_1_km (P : ℝ) (S : ℝ) (r : ℝ) (h1 : 2 * S ≥ r * P) (h2 : r = 0.1) : P < 1000 * 100 :=
  sorry

theorem perimeter_less_than_1_km_zero_thickness (P : ℝ) (S : ℝ) (r : ℝ) (h1 : 2 * S ≥ r * P) (h2 : r = 0) : P < 1000 * 100 :=
  sorry

theorem perimeter_to_area_ratio (P : ℝ) (S : ℝ) (h : P / S ≤ 700) : P / S < 100000 :=
  sorry

end Problem

end perimeter_less_than_1_km_perimeter_less_than_1_km_zero_thickness_perimeter_to_area_ratio_l74_74837


namespace cookies_in_jar_l74_74651

noncomputable def C : ℕ := sorry

theorem cookies_in_jar (h : C - 1 = (C + 5) / 2) : C = 7 := by
  sorry

end cookies_in_jar_l74_74651


namespace expression_evaluation_l74_74706

theorem expression_evaluation :
  (-2: ℤ)^3 + ((36: ℚ) / (3: ℚ)^2 * (-1 / 2: ℚ)) + abs (-5: ℤ) = -5 :=
by
  sorry

end expression_evaluation_l74_74706


namespace pool_balls_pyramid_arrangement_l74_74326

/-- In how many distinguishable ways can 10 distinct pool balls be arranged in a pyramid
    (6 on the bottom, 3 in the middle, 1 on the top), assuming that all rotations of the pyramid are indistinguishable? -/
def pyramid_pool_balls_distinguishable_arrangements : Nat :=
  let total_arrangements := Nat.factorial 10
  let indistinguishable_rotations := 9
  total_arrangements / indistinguishable_rotations

theorem pool_balls_pyramid_arrangement :
  pyramid_pool_balls_distinguishable_arrangements = 403200 :=
by
  -- Proof will be added here
  sorry

end pool_balls_pyramid_arrangement_l74_74326


namespace probability_same_number_l74_74277

theorem probability_same_number (h1: ∀ n, 0 < n ∧ n < 300 → (n % 15 = 0 ↔ n % 20 = 0)) : 
  let total_combinations := 20 * 15 in
  let common_multiples := 5 in
  ((common_multiples : ℚ) / total_combinations = (1 : ℚ) / 60) :=
by
  sorry

end probability_same_number_l74_74277


namespace abc_value_l74_74610

theorem abc_value (a b c : ℂ) 
  (h1 : a * b + 5 * b = -20)
  (h2 : b * c + 5 * c = -20)
  (h3 : c * a + 5 * a = -20) : 
  a * b * c = -100 := 
by {
  sorry
}

end abc_value_l74_74610


namespace number_of_solutions_l74_74886

theorem number_of_solutions :
  ∃ (sols : Finset ℝ), 
    (∀ x, x ∈ sols → 0 ≤ x ∧ x ≤ 2 * Real.pi ∧ 2 * (Real.sin x)^3 - 5 * (Real.sin x)^2 + 2 * Real.sin x = 0) 
    ∧ Finset.card sols = 5 := 
by
  sorry

end number_of_solutions_l74_74886


namespace count_valid_numbers_l74_74711

open BigOperators

def is_valid_number (n : ℕ) : Prop :=
  n.to_string.length = 4 ∧
  (∀ i, n.to_string.nth i ∈ ['1', '2', '3']) ∧
  '1' ∈ n.to_string ∧ '2' ∈ n.to_string ∧ '3' ∈ n.to_string ∧
  ∀ i, i < 3 → n.to_string.nth i ≠ n.to_string.nth (i + 1)

theorem count_valid_numbers : 
  {n | is_valid_number n}.card = 66 := 
sorry

end count_valid_numbers_l74_74711


namespace selling_price_l74_74861

theorem selling_price (cost_price profit_percentage : ℝ) (h_cost : cost_price = 250) (h_profit : profit_percentage = 0.60) :
  cost_price + profit_percentage * cost_price = 400 := sorry

end selling_price_l74_74861


namespace units_digit_factorial_150_zero_l74_74256

def units_digit (n : ℕ) : ℕ :=
  (nat.factorial n) % 10

theorem units_digit_factorial_150_zero :
  units_digit 150 = 0 :=
sorry

end units_digit_factorial_150_zero_l74_74256


namespace sufficient_condition_of_necessary_condition_l74_74033

-- Define the necessary condition
def necessary_condition (A B : Prop) : Prop := A → B

-- The proof problem statement
theorem sufficient_condition_of_necessary_condition
  {A B : Prop} (h : necessary_condition A B) : necessary_condition A B :=
by
  exact h

end sufficient_condition_of_necessary_condition_l74_74033


namespace find_A_l74_74013

def spadesuit (A B : ℝ) : ℝ := 4 * A + 3 * B + 6

theorem find_A (A : ℝ) (h : spadesuit A 5 = 59) : A = 9.5 :=
by sorry

end find_A_l74_74013


namespace marla_errand_time_l74_74615

theorem marla_errand_time :
  let drive_time_one_way := 20
  let parent_teacher_night := 70
  let drive_time_total := drive_time_one_way * 2
  let total_time := drive_time_total + parent_teacher_night
  total_time = 110 := by
  let drive_time_one_way := 20
  let parent_teacher_night := 70
  let drive_time_total := drive_time_one_way * 2
  let total_time := drive_time_total + parent_teacher_night
  sorry

end marla_errand_time_l74_74615


namespace problem_statement_l74_74743
-- Broader import to bring in necessary library components.

-- Definition of the equation that needs to be satisfied by the points.
def satisfies_equation (x y : ℝ) : Prop := 2 * x^2 + y^2 + 3 * x * y + 3 * x + y = 2

-- Definitions of the two lines that form the solution set.
def line1 (x y : ℝ) : Prop := y = -x - 2
def line2 (x y : ℝ) : Prop := y = -2 * x + 1

-- Prove that the set of points that satisfy the given equation is the union of the two lines.
theorem problem_statement (x y : ℝ) : satisfies_equation x y ↔ line1 x y ∨ line2 x y :=
sorry

end problem_statement_l74_74743


namespace probability_at_least_one_pen_l74_74643

noncomputable def PAs  := 3/5
noncomputable def PBs  := 2/3
noncomputable def PABs := PAs * PBs

theorem probability_at_least_one_pen : PAs + PBs - PABs = 13 / 15 := by
  sorry

end probability_at_least_one_pen_l74_74643


namespace integer_roots_condition_l74_74264

theorem integer_roots_condition (n : ℕ) (hn : n > 0) :
  (∃ x : ℤ, x^2 - 4 * x + n = 0) ↔ (n = 3 ∨ n = 4) := 
by
  sorry

end integer_roots_condition_l74_74264


namespace blueberries_in_blue_box_l74_74841

theorem blueberries_in_blue_box (S B : ℕ) (h1 : S - B = 15) (h2 : S + B = 87) : B = 36 :=
by sorry

end blueberries_in_blue_box_l74_74841


namespace fish_stock_l74_74069

theorem fish_stock {
  initial_stock fish_sold new_stock : ℕ,
  spoil_fraction : ℚ,
  initial_stock = 200 → fish_sold = 50 → new_stock = 200 →
  spoil_fraction = 1 / 3 →
  ∃ (final_stock : ℕ), final_stock = initial_stock - fish_sold - ⌊(initial_stock - fish_sold) * spoil_fraction⌋ + new_stock ∧ final_stock = 300 :=
begin
  intros h_initial h_sold h_new h_spoil,
  use initial_stock - fish_sold - (⌊(initial_stock - fish_sold) * spoil_fraction⌋ : ℕ) + new_stock,
  split,
  {
    rw [h_initial, h_sold, h_new, h_spoil],
    norm_num,
  },
  exact ⟨300⟩,
end

end fish_stock_l74_74069


namespace cookies_in_jar_l74_74657

noncomputable def number_of_cookies_in_jar : ℕ := sorry

theorem cookies_in_jar :
  (number_of_cookies_in_jar - 1) = (1 / 2 : ℝ) * (number_of_cookies_in_jar + 5) →
  number_of_cookies_in_jar = 7 :=
by
  sorry

end cookies_in_jar_l74_74657


namespace average_visitors_per_day_in_month_l74_74694

theorem average_visitors_per_day_in_month (avg_visitors_sunday : ℕ) (avg_visitors_other_days : ℕ) (days_in_month : ℕ) (starts_sunday : Bool) :
  avg_visitors_sunday = 140 → avg_visitors_other_days = 80 → days_in_month = 30 → starts_sunday = true → 
  (∀ avg_visitors, avg_visitors = (4 * avg_visitors_sunday + 26 * avg_visitors_other_days) / days_in_month → avg_visitors = 88) :=
by
  intros h1 h2 h3 h4
  have total_visitors : ℕ := 4 * avg_visitors_sunday + 26 * avg_visitors_other_days
  have avg := total_visitors / days_in_month
  have visitors : ℕ := 2640
  sorry

end average_visitors_per_day_in_month_l74_74694


namespace seashells_total_l74_74620

theorem seashells_total :
  let monday := 5
  let tuesday := 7 - 3
  let wednesday := (2 * monday) / 2
  let thursday := 3 * 7
  monday + tuesday + wednesday + thursday = 35 :=
by
  sorry

end seashells_total_l74_74620


namespace arithmetic_mean_of_reciprocals_of_first_four_primes_l74_74878

theorem arithmetic_mean_of_reciprocals_of_first_four_primes :
  let primes := [2, 3, 5, 7] in
  let reciprocals := primes.map (λ p => (1 : ℚ) / p) in
  let sum_reciprocals := reciprocals.sum in
  let mean := sum_reciprocals / (primes.length : ℚ) in
  mean = 247 / 840 :=
by
  let primes := [2, 3, 5, 7]
  let reciprocals := primes.map (λ p => (1 : ℚ) / p)
  let sum_reciprocals := reciprocals.sum
  let mean := sum_reciprocals / (primes.length : ℚ)
  sorry

end arithmetic_mean_of_reciprocals_of_first_four_primes_l74_74878


namespace arithmetic_statement_not_basic_l74_74258

-- Define the basic algorithmic statements as a set
def basic_algorithmic_statements : Set String := 
  {"Input statement", "Output statement", "Assignment statement", "Conditional statement", "Loop statement"}

-- Define the arithmetic statement
def arithmetic_statement : String := "Arithmetic statement"

-- Prove that arithmetic statement is not a basic algorithmic statement
theorem arithmetic_statement_not_basic :
  arithmetic_statement ∉ basic_algorithmic_statements :=
sorry

end arithmetic_statement_not_basic_l74_74258


namespace gcd_72_108_l74_74114

theorem gcd_72_108 : Nat.gcd 72 108 = 36 :=
by
  sorry

end gcd_72_108_l74_74114


namespace candies_total_l74_74465

theorem candies_total (N a S : ℕ) (h1 : S = 2 * a + 7) (h2 : S = N * a) (h3 : a > 1) (h4 : N = 3) : S = 21 := 
sorry

end candies_total_l74_74465


namespace sufficient_conditions_for_x_sq_lt_one_l74_74002

theorem sufficient_conditions_for_x_sq_lt_one
  (x : ℝ) :
  (0 < x ∧ x < 1) ∨ (-1 < x ∧ x < 0) ∨ (-1 < x ∧ x < 1) → x^2 < 1 :=
by
  sorry

end sufficient_conditions_for_x_sq_lt_one_l74_74002


namespace largest_possible_b_l74_74649

theorem largest_possible_b (a b c : ℕ) (h1 : 1 < c) (h2 : c < b) (h3 : b < a) (h4 : a * b * c = 360) : b = 10 :=
sorry

end largest_possible_b_l74_74649


namespace clock_angle_8_30_l74_74223

theorem clock_angle_8_30 
  (angle_per_hour_mark : ℝ := 30)
  (angle_per_minute_mark : ℝ := 6)
  (hour_hand_angle_8 : ℝ := 8 * angle_per_hour_mark)
  (half_hour_movement : ℝ := 0.5 * angle_per_hour_mark)
  (hour_hand_angle_8_30 : ℝ := hour_hand_angle_8 + half_hour_movement)
  (minute_hand_angle_30 : ℝ := 30 * angle_per_minute_mark) :
  abs (hour_hand_angle_8_30 - minute_hand_angle_30) = 75 :=
by
  sorry

end clock_angle_8_30_l74_74223


namespace range_of_a_l74_74763

noncomputable def min_expr (x: ℝ) : ℝ := x + 2/(x - 2)

theorem range_of_a (a: ℝ) : 
  (∀ x > 2, a ≤ min_expr x) ↔ a ≤ 2 + 2 * Real.sqrt 2 := 
by
  sorry

end range_of_a_l74_74763


namespace children_got_on_the_bus_l74_74688

-- Definitions
def original_children : ℕ := 26
def current_children : ℕ := 64

-- Theorem stating the problem
theorem children_got_on_the_bus : (current_children - original_children = 38) :=
by {
  sorry
}

end children_got_on_the_bus_l74_74688


namespace prob_B_win_correct_l74_74666

-- Define the probabilities for player A winning and a draw
def prob_A_win : ℝ := 0.3
def prob_draw : ℝ := 0.4

-- Define the total probability of all outcomes
def total_prob : ℝ := 1

-- Define the probability of player B winning
def prob_B_win : ℝ := total_prob - prob_A_win - prob_draw

-- Proof problem: Prove that the probability of player B winning is 0.3
theorem prob_B_win_correct : prob_B_win = 0.3 :=
by
  -- The proof would go here, but we use sorry to skip it for now.
  sorry

end prob_B_win_correct_l74_74666


namespace selling_price_is_1260_l74_74478

-- Definitions based on conditions
def purchase_price : ℕ := 900
def repair_cost : ℕ := 300
def gain_percent : ℕ := 5 -- percentage as a natural number

-- Known variables
def total_cost : ℕ := purchase_price + repair_cost
def gain_amount : ℕ := (gain_percent * total_cost) / 100
def selling_price : ℕ := total_cost + gain_amount

-- The theorem we want to prove
theorem selling_price_is_1260 : selling_price = 1260 := by
  sorry

end selling_price_is_1260_l74_74478


namespace ratio_male_to_female_l74_74926

theorem ratio_male_to_female (total_members female_members : ℕ) (h_total : total_members = 18) (h_female : female_members = 6) :
  (total_members - female_members) / Nat.gcd (total_members - female_members) female_members = 2 ∧
  female_members / Nat.gcd (total_members - female_members) female_members = 1 :=
by
  sorry

end ratio_male_to_female_l74_74926


namespace correct_factorization_l74_74375

theorem correct_factorization {x y : ℝ} :
  (2 * x ^ 2 - 8 * y ^ 2 = 2 * (x + 2 * y) * (x - 2 * y)) ∧
  ¬(x ^ 2 + 3 * x * y + 9 * y ^ 2 = (x + 3 * y) ^ 2)
    ∧ ¬(2 * x ^ 2 - 4 * x * y + 9 * y ^ 2 = (2 * x - 3 * y) ^ 2)
    ∧ ¬(x * (x - y) + y * (y - x) = (x - y) * (x + y)) := 
by sorry

end correct_factorization_l74_74375


namespace geometric_series_sum_l74_74879

theorem geometric_series_sum :
  let a := 1
  let r := (1 : ℝ) / 5
  ∑' n : ℕ, a * r ^ n = 5 / 4 :=
by
  sorry

end geometric_series_sum_l74_74879


namespace trent_walks_to_bus_stop_l74_74503

theorem trent_walks_to_bus_stop (x : ℕ) (h1 : 2 * (x + 7) = 22) : x = 4 :=
sorry

end trent_walks_to_bus_stop_l74_74503


namespace symmetric_point_of_A_l74_74484

theorem symmetric_point_of_A (a b : ℝ) 
  (h1 : 2 * a - 4 * b + 9 = 0) 
  (h2 : ∃ t : ℝ, (a, b) = (1 - 4 * t, 4 + 2 * t)) : 
  (a, b) = (1, 4) :=
sorry

end symmetric_point_of_A_l74_74484


namespace max_vec_diff_magnitude_l74_74913

open Real

noncomputable def vec_a (θ : ℝ) : ℝ × ℝ := (1, sin θ)
noncomputable def vec_b (θ : ℝ) : ℝ × ℝ := (1, cos θ)

noncomputable def vec_diff_magnitude (θ : ℝ) : ℝ :=
  let a := vec_a θ
  let b := vec_b θ
  abs ((a.1 - b.1)^2 + (a.2 - b.2)^2)^(1/2)

theorem max_vec_diff_magnitude : ∀ θ : ℝ, vec_diff_magnitude θ ≤ sqrt 2 :=
by
  intro θ
  sorry

end max_vec_diff_magnitude_l74_74913


namespace equation_solution_l74_74749

theorem equation_solution:
  ∀ (x y : ℝ), 
  2 * x^2 + y^2 + 3 * x * y + 3 * x + y = 2 ↔ 
  (y = -x - 2 ∨ y = -2 * x + 1) := 
by
  sorry

end equation_solution_l74_74749


namespace a_eq_b_if_fraction_is_integer_l74_74890

theorem a_eq_b_if_fraction_is_integer (a b : ℕ) (h_pos_a : 1 ≤ a) (h_pos_b : 1 ≤ b) :
  ∃ k : ℕ, (a^4 + a^3 + 1) = k * (a^2 * b^2 + a * b^2 + 1) -> a = b :=
by
  sorry

end a_eq_b_if_fraction_is_integer_l74_74890


namespace smallest_common_multiple_8_6_l74_74511

theorem smallest_common_multiple_8_6 : 
  ∃ n : ℕ, n > 0 ∧ (n % 8 = 0) ∧ (n % 6 = 0) ∧ ∀ m : ℕ, m > 0 ∧ (m % 8 = 0) ∧ (m % 6 = 0) → m ≥ n :=
begin
  use 24,
  split,
  { norm_num }, -- 24 > 0
  split,
  { norm_num }, -- 24 % 8 = 0
  split,
  { norm_num }, -- 24 % 6 = 0
  { intros m hm,
    cases hm with hp8 hp6,
    norm_num at hp8 hp6,
    sorry -- Prove that 24 is the smallest such number
  }
end

end smallest_common_multiple_8_6_l74_74511


namespace find_b_l74_74279

theorem find_b (a b c d : ℝ) (h : ∃ k : ℝ, 2 * k = π ∧ k * (b / 2) = π) : b = 4 :=
by
  sorry

end find_b_l74_74279


namespace exists_infinitely_many_m_l74_74798

theorem exists_infinitely_many_m (k : ℕ) (hk : 0 < k) : 
  ∃ᶠ m in at_top, 3 ^ k ∣ m ^ 3 + 10 :=
sorry

end exists_infinitely_many_m_l74_74798


namespace closest_years_l74_74366

theorem closest_years (a b c d : ℕ) (h1 : 10 * a + b + 10 * c + d = 10 * b + c) :
  (a = 1 ∧ b = 8 ∧ c = 6 ∧ d = 8) ∨ (a = 2 ∧ b = 3 ∧ c = 0 ∧ d =7) ↔
  ((10 * 1 + 8 + 10 * 6 + 8 = 10 * 8 + 6) ∧ (10 * 2 + 3 + 10 * 0 + 7 = 10 * 3 + 0)) :=
sorry

end closest_years_l74_74366


namespace square_garden_perimeter_l74_74999

theorem square_garden_perimeter (A : ℝ) (s : ℝ) (N : ℝ) 
  (h1 : A = 9)
  (h2 : s^2 = A)
  (h3 : N = 4 * s) 
  : N = 12 := 
by
  sorry

end square_garden_perimeter_l74_74999


namespace walking_time_l74_74674

noncomputable def time_to_reach_destination (mr_harris_speed : ℝ) (mr_harris_time_to_store : ℝ) (your_speed : ℝ) (distance_factor : ℝ) : ℝ :=
  let store_distance := mr_harris_speed * mr_harris_time_to_store
  let your_destination_distance := distance_factor * store_distance
  your_destination_distance / your_speed

theorem walking_time (mr_harris_speed your_speed : ℝ) (mr_harris_time_to_store : ℝ) (distance_factor : ℝ) (h_speed : your_speed = 2 * mr_harris_speed) (h_time : mr_harris_time_to_store = 2) (h_factor : distance_factor = 3) :
  time_to_reach_destination mr_harris_speed mr_harris_time_to_store your_speed distance_factor = 3 :=
by
  rw [h_time, h_speed, h_factor]
  -- calculations based on given conditions
  sorry

end walking_time_l74_74674


namespace problem_statement_l74_74741
-- Broader import to bring in necessary library components.

-- Definition of the equation that needs to be satisfied by the points.
def satisfies_equation (x y : ℝ) : Prop := 2 * x^2 + y^2 + 3 * x * y + 3 * x + y = 2

-- Definitions of the two lines that form the solution set.
def line1 (x y : ℝ) : Prop := y = -x - 2
def line2 (x y : ℝ) : Prop := y = -2 * x + 1

-- Prove that the set of points that satisfy the given equation is the union of the two lines.
theorem problem_statement (x y : ℝ) : satisfies_equation x y ↔ line1 x y ∨ line2 x y :=
sorry

end problem_statement_l74_74741


namespace handshake_count_l74_74496

theorem handshake_count (num_companies : ℕ) (num_representatives : ℕ) 
  (total_handshakes : ℕ) (h1 : num_companies = 5) (h2 : num_representatives = 5)
  (h3 : total_handshakes = (num_companies * num_representatives * 
   (num_companies * num_representatives - 1 - (num_representatives - 1)) / 2)) :
  total_handshakes = 250 :=
by
  rw [h1, h2] at h3
  exact h3

end handshake_count_l74_74496


namespace union_of_sets_l74_74911

open Set

-- Define the sets A and B
def A : Set ℤ := {-2, 0}
def B : Set ℤ := {-2, 3}

-- Prove that the union of A and B equals {–2, 0, 3}
theorem union_of_sets : A ∪ B = {-2, 0, 3} := by
  sorry

end union_of_sets_l74_74911


namespace tangent_line_at_1_extreme_points_range_of_a_l74_74175

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.log x + a * (x ^ 2 - 3 * x + 2)

theorem tangent_line_at_1 (a : ℝ) (h : a = 0) :
  ∃ m b, ∀ x, f x a = m * x + b ∧ m = 1 ∧ b = -1 := sorry

theorem extreme_points (a : ℝ) :
  (0 < a ∧ a <= 8 / 9 → ∀ x, 0 < x → f x a = 0) ∧
  (a > 8 / 9 → ∃ x1 x2, x1 < x2 ∧ f x1 a = 0 ∧ f x2 a = 0 ∧
   (∀ x, 0 < x ∧ x < x1 → f x a = 0) ∧
   (∀ x, x1 < x ∧ x < x2 → f x a = 0) ∧
   (∀ x, x2 < x → f x a = 0)) ∧
  (a < 0 → ∃ x1 x2, x1 < 0 ∧ 0 < x2 ∧ f x1 a = 0 ∧ f x2 a = 0 ∧
   (∀ x, 0 < x ∧ x < x2 → f x a = 0) ∧
   (∀ x, x2 < x → f x a = 0)) := sorry

theorem range_of_a (a : ℝ) :
  (∀ x, 1 ≤ x → f x a >= 0) ↔ 0 ≤ a ∧ a ≤ 1 := sorry

end tangent_line_at_1_extreme_points_range_of_a_l74_74175


namespace integer_n_cubed_minus_3_n_squared_plus_n_plus_2_eq_power_of_5_l74_74888

theorem integer_n_cubed_minus_3_n_squared_plus_n_plus_2_eq_power_of_5 :
  ∀ n : ℤ, ∃ k : ℕ, (n^3 - 3 * n^2 + n + 2 = 5^k) ↔ n = 3 :=
by
  intro n
  exists sorry
  sorry

end integer_n_cubed_minus_3_n_squared_plus_n_plus_2_eq_power_of_5_l74_74888


namespace proof1_proof2_false_proof3_proof4_false_proof5_l74_74554

-- Condition 1
def condition1 (α β : ℝ) : Prop :=
  α + β = 7 * Real.pi / 4

theorem proof1 (α β : ℝ) (h : condition1 α β) : 
  (1 - Real.tan α) * (1 - Real.tan β) = 2 := 
sorry

-- Condition 2
def vector_a : ℝ × ℝ := (1, -2)
def vector_b (λ : ℝ) : ℝ × ℝ := (2, λ)

def condition2 (λ : ℝ) : Prop :=
  λ < 1

theorem proof2_false (λ : ℝ) : 
  (let a := vector_a 
   let b := vector_b λ 
   (a.1 * b.1 + a.2 * b.2) > 0) ↔ ¬ condition2 λ :=
sorry

-- Condition 3
structure Point (α : Type) :=
(x : α) (y : α)

def condition3 (O A B C : Point ℝ) (λ : ℝ) (P : Point ℝ) : Prop :=
  λ > 0 ∧ λ < Real.infinity ∧ 
  (P.x = O.x + λ * (B.x - A.x + C.x - A.x)) ∧ 
  (P.y = O.y + λ * (B.y - A.y + C.y - A.y))

theorem proof3 (O A B C P : Point ℝ) (λ : ℝ) (h : condition3 O A B C λ P) : 
  (let G := Point.mk ((A.x + B.x + C.x) / 3) ((A.y + B.y + C.y) / 3) in P = G) :=
sorry

-- Condition 4
def triangle (A B C : Point ℝ) : Prop := 
  A ≠ B ∧ B ≠ C ∧ A ≠ C

def condition4 (A B C : Point ℝ) (angle_A : ℝ) (a c : ℝ) : Prop :=
  angle_A = Real.pi / 3 ∧ a = 4 ∧ c = 3 * Real.sqrt 3 ∧ triangle A B C

theorem proof4_false (A B C : Point ℝ) (angle_A : ℝ) (a c : ℝ) (h : condition4 A B C angle_A a c) : 
  ¬ ∃ B C, triangle A B C :=
sorry

-- Condition 5
def condition5 (A B C : Point ℝ) (R : ℝ) : Prop :=
  ∃ a b, 2 * R * (Real.sin (A.x) ^ 2 - Real.sin (C.x) ^ 2) = (Real.sqrt 2 * a - b) * Real.sin (B.x)

theorem proof5 (A B C : Point ℝ) (R a b : ℝ) (h : condition5 A B C R) : 
  let max_area := (Real.sqrt 2 + 1) / 2 * R^2 in 
  ∃ S, S ≤ max_area :=
sorry

end proof1_proof2_false_proof3_proof4_false_proof5_l74_74554


namespace ratio_of_adults_to_children_l74_74960

-- Defining conditions as functions
def admission_fees_condition (a c : ℕ) : ℕ := 30 * a + 15 * c

-- Stating the problem
theorem ratio_of_adults_to_children (a c : ℕ) 
  (h1 : admission_fees_condition a c = 2250)
  (h2 : a ≥ 1) 
  (h3 : c ≥ 1) 
  : a / c = 2 := 
sorry

end ratio_of_adults_to_children_l74_74960


namespace rupert_candles_l74_74621

theorem rupert_candles (peter_candles : ℕ) (rupert_times_older : ℝ) (h1 : peter_candles = 10) (h2 : rupert_times_older = 3.5) :
    ∃ rupert_candles : ℕ, rupert_candles = peter_candles * rupert_times_older := 
by
  sorry

end rupert_candles_l74_74621


namespace laurie_shells_l74_74348

def alan_collected : ℕ := 48
def ben_collected (alan : ℕ) : ℕ := alan / 4
def laurie_collected (ben : ℕ) : ℕ := ben * 3

theorem laurie_shells (a : ℕ) (b : ℕ) (l : ℕ) (h1 : alan_collected = a)
  (h2 : ben_collected a = b) (h3 : laurie_collected b = l) : l = 36 := 
by
  sorry

end laurie_shells_l74_74348


namespace keaton_apple_earnings_l74_74336

theorem keaton_apple_earnings
  (orange_harvest_interval : ℕ)
  (orange_income_per_harvest : ℕ)
  (total_yearly_income : ℕ)
  (orange_harvests_per_year : ℕ)
  (orange_yearly_income : ℕ)
  (apple_yearly_income : ℕ) :
  orange_harvest_interval = 2 →
  orange_income_per_harvest = 50 →
  total_yearly_income = 420 →
  orange_harvests_per_year = 12 / orange_harvest_interval →
  orange_yearly_income = orange_harvests_per_year * orange_income_per_harvest →
  apple_yearly_income = total_yearly_income - orange_yearly_income →
  apple_yearly_income = 120 :=
by
  sorry

end keaton_apple_earnings_l74_74336


namespace angle_D_is_90_l74_74164

theorem angle_D_is_90 (A B C D : ℝ) (h1 : A + B = 180) (h2 : C = D) (h3 : A = 50) (h4 : B = 130) (h5 : C + D = 180) :
  D = 90 :=
by
  sorry

end angle_D_is_90_l74_74164


namespace larger_segment_of_triangle_l74_74145

theorem larger_segment_of_triangle (x y : ℝ) (h1 : 40^2 = x^2 + y^2) 
  (h2 : 90^2 = (100 - x)^2 + y^2) :
  100 - x = 82.5 :=
by {
  sorry
}

end larger_segment_of_triangle_l74_74145


namespace find_number_l74_74687

def problem (x : ℝ) : Prop :=
  0.25 * x = 130 + 190

theorem find_number (x : ℝ) (h : problem x) : x = 1280 :=
by 
  sorry

end find_number_l74_74687


namespace heartsuit_properties_l74_74287

def heartsuit (x y : ℝ) : ℝ := abs (x - y)

theorem heartsuit_properties (x y : ℝ) :
  (heartsuit x y ≥ 0) ∧ (heartsuit x y > 0 ↔ x ≠ y) := by
  -- Proof will go here 
  sorry

end heartsuit_properties_l74_74287


namespace polynomial_coefficients_equivalence_l74_74202

theorem polynomial_coefficients_equivalence
    {a0 a1 a2 a3 a4 a5 : ℤ}
    (h_poly : (2*x-1)^5 = a0 + a1*x + a2*x^2 + a3*x^3 + a4*x^4 + a5*x^5):
    (a0 + a1 + a2 + a3 + a4 + a5 = 1) ∧
    (|a0| + |a1| + |a2| + |a3| + |a4| + |a5| = 243) ∧
    (a1 + a3 + a5 = 122) ∧
    ((a0 + a2 + a4)^2 - (a1 + a3 + a5)^2 = -243) :=
    sorry

end polynomial_coefficients_equivalence_l74_74202


namespace units_digit_of_150_factorial_is_zero_l74_74249

theorem units_digit_of_150_factorial_is_zero : 
  ∃ k : ℕ, (150! = k * 10) :=
begin
  -- We need to prove that there exists a natural number k such that 150! is equal to k times 10
  sorry
end

end units_digit_of_150_factorial_is_zero_l74_74249


namespace dad_vacuum_time_l74_74603

theorem dad_vacuum_time (x : ℕ) (h1 : 2 * x + 5 = 27) (h2 : x + (2 * x + 5) = 38) :
  (2 * x + 5) = 27 := by
  sorry

end dad_vacuum_time_l74_74603


namespace problem_solution_set_l74_74429

-- Definitions and conditions according to the given problem
def odd_function_domain := {x : ℝ | x ≠ 0}
def function_condition1 (f : ℝ → ℝ) (x : ℝ) : Prop := x > 0 → deriv f x < (3 * f x) / x
def function_condition2 (f : ℝ → ℝ) : Prop := f 1 = 1 / 2
def function_condition3 (f : ℝ → ℝ) : Prop := ∀ x, f (2 * x) = 2 * f x

-- Main proof statement
theorem problem_solution_set (f : ℝ → ℝ)
  (odd_function : ∀ x, f (-x) = -f x)
  (dom : ∀ x, x ∈ odd_function_domain → f x ≠ 0)
  (cond1 : ∀ x, function_condition1 f x)
  (cond2 : function_condition2 f)
  (cond3 : function_condition3 f) :
  {x : ℝ | f x / (4 * x) < 2 * x^2} = {x : ℝ | x < -1 / 4} ∪ {x : ℝ | x > 1 / 4} :=
sorry

end problem_solution_set_l74_74429


namespace light_coloured_blocks_in_tower_l74_74365

theorem light_coloured_blocks_in_tower :
  let central_blocks := 4
  let outer_columns := 8
  let height_per_outer_column := 2
  let total_light_coloured_blocks := central_blocks + outer_columns * height_per_outer_column
  total_light_coloured_blocks = 20 :=
by
  let central_blocks := 4
  let outer_columns := 8
  let height_per_outer_column := 2
  let total_light_coloured_blocks := central_blocks + outer_columns * height_per_outer_column
  show total_light_coloured_blocks = 20
  sorry

end light_coloured_blocks_in_tower_l74_74365


namespace find_y_l74_74117

theorem find_y (x y : ℝ) (h1 : x + 2 * y = 12) (h2 : x = 6) : y = 3 :=
by
  sorry

end find_y_l74_74117


namespace tangent_parallel_to_given_line_l74_74433

theorem tangent_parallel_to_given_line (a : ℝ) : 
  let y := λ x : ℝ => x^2 + a / x
  let y' := λ x : ℝ => (deriv y) x
  y' 1 = 2 
  → a = 0 := by
  -- y'(1) is the derivative of y at x=1
  sorry

end tangent_parallel_to_given_line_l74_74433


namespace arithmetic_mean_reciprocal_primes_l74_74874

theorem arithmetic_mean_reciprocal_primes :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let r1 := (1 : ℚ) / p1
  let r2 := (1 : ℚ) / p2
  let r3 := (1 : ℚ) / p3
  let r4 := (1 : ℚ) / p4
  (r1 + r2 + r3 + r4) / 4 = 247 / 840 := by
sorry

end arithmetic_mean_reciprocal_primes_l74_74874


namespace sum_equals_one_l74_74704

noncomputable def sum_proof (x y z : ℝ) (h : x * y * z = 1) : ℝ :=
  (1 / (1 + x + x * y)) + (1 / (1 + y + y * z)) + (1 / (1 + z + z * x))

theorem sum_equals_one (x y z : ℝ) (h : x * y * z = 1) : 
  sum_proof x y z h = 1 := sorry

end sum_equals_one_l74_74704


namespace cost_of_each_pair_of_socks_eq_2_l74_74135

-- Definitions and conditions
def cost_of_shoes : ℤ := 74
def cost_of_bag : ℤ := 42
def paid_amount : ℤ := 118
def discount_rate : ℚ := 0.10

-- Given the conditions
def total_cost (x : ℚ) : ℚ := cost_of_shoes + 2 * x + cost_of_bag
def discount (x : ℚ) : ℚ := if total_cost x > 100 then discount_rate * (total_cost x - 100) else 0
def total_cost_after_discount (x : ℚ) : ℚ := total_cost x - discount x

-- Theorem to prove
theorem cost_of_each_pair_of_socks_eq_2 : 
  ∃ x : ℚ, total_cost_after_discount x = paid_amount ∧ 2 * x = 4 :=
by
  sorry

end cost_of_each_pair_of_socks_eq_2_l74_74135


namespace number_of_questions_in_exam_l74_74191

theorem number_of_questions_in_exam :
  ∀ (typeA : ℕ) (typeB : ℕ) (timeA : ℝ) (timeB : ℝ) (totalTime : ℝ),
    typeA = 100 →
    timeA = 1.2 →
    timeB = 0.6 →
    totalTime = 180 →
    120 = typeA * timeA →
    totalTime - 120 = typeB * timeB →
    typeA + typeB = 200 :=
by
  intros typeA typeB timeA timeB totalTime h_typeA h_timeA h_timeB h_totalTime h_timeA_calc h_remaining_time
  sorry

end number_of_questions_in_exam_l74_74191


namespace group_weight_problem_l74_74483

theorem group_weight_problem (n : ℕ) (avg_weight_increase : ℕ) (weight_diff : ℕ) (total_weight_increase : ℕ) 
  (h1 : avg_weight_increase = 3) (h2 : weight_diff = 75 - 45) (h3 : total_weight_increase = avg_weight_increase * n)
  (h4 : total_weight_increase = weight_diff) : n = 10 := by
  sorry

end group_weight_problem_l74_74483


namespace johnson_family_children_count_l74_74959

theorem johnson_family_children_count (m : ℕ) (x : ℕ) (xy : ℕ) :
  (m + 50 + xy = (2 + x) * 21) ∧ (2 * m + xy = 60) → x = 1 :=
by
  sorry

end johnson_family_children_count_l74_74959


namespace smallest_n_divisible_by_5_l74_74420

def is_not_divisible_by_5 (x : ℤ) : Prop :=
  ¬ (x % 5 = 0)

def avg_is_integer (xs : List ℤ) : Prop :=
  (List.sum xs) % 5 = 0

theorem smallest_n_divisible_by_5 (n : ℕ) (h1 : n > 1980)
  (h2 : ∀ x ∈ List.range n, is_not_divisible_by_5 x)
  : n = 1985 :=
by
  -- The proof would go here
  sorry

end smallest_n_divisible_by_5_l74_74420


namespace symmetric_point_of_P_l74_74451

-- Define a point in the Cartesian coordinate system
structure Point :=
  (x : ℤ)
  (y : ℤ)

-- Define central symmetry with respect to the origin
def symmetric (p : Point) : Point :=
  { x := -p.x, y := -p.y }

-- Given point P with coordinates (1, -2)
def P : Point := { x := 1, y := -2 }

-- The theorem to be proved: the symmetric point of P is (-1, 2)
theorem symmetric_point_of_P :
  symmetric P = { x := -1, y := 2 } :=
by
  -- Proof is omitted.
  sorry

end symmetric_point_of_P_l74_74451


namespace probability_even_sum_l74_74662

open Nat

def balls : List Nat := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

def even_sum_probability : ℚ :=
  let total_outcomes := 12 * 11
  let even_balls := balls.filter (λ n => n % 2 = 0)
  let odd_balls := balls.filter (λ n => n % 2 = 1)
  let even_outcomes := even_balls.length * (even_balls.length - 1)
  let odd_outcomes := odd_balls.length * (odd_balls.length - 1)
  let favorable_outcomes := even_outcomes + odd_outcomes
  (favorable_outcomes : ℚ) / (total_outcomes : ℚ)

theorem probability_even_sum :
  even_sum_probability = 5 / 11 := by
  sorry

end probability_even_sum_l74_74662


namespace jeff_cat_shelter_l74_74931

theorem jeff_cat_shelter :
  let initial_cats := 20
  let monday_cats := 2
  let tuesday_cats := 1
  let people_adopted := 3
  let cats_per_person := 2
  let total_cats := initial_cats + monday_cats + tuesday_cats
  let adopted_cats := people_adopted * cats_per_person
  total_cats - adopted_cats = 17 := 
by
  sorry

end jeff_cat_shelter_l74_74931


namespace graphs_symmetric_respect_to_x_equals_1_l74_74702

-- Define the function f
variable (f : ℝ → ℝ)

-- Define g(x) = f(x-1)
def g (x : ℝ) : ℝ := f (x - 1)

-- Define h(x) = f(1 - x)
def h (x : ℝ) : ℝ := f (1 - x)

-- The theorem that their graphs are symmetric with respect to the line x = 1
theorem graphs_symmetric_respect_to_x_equals_1 :
  ∀ x : ℝ, g f x = h f x ↔ f x = f (2 - x) :=
sorry

end graphs_symmetric_respect_to_x_equals_1_l74_74702


namespace ellipse_foci_coordinates_l74_74414

/-- Define the parameters for the ellipse. -/
def ellipse_eq (x y : ℝ) : Prop := (x^2) / 25 + (y^2) / 169 = 1

/-- Prove the coordinates of the foci of the given ellipse. -/
theorem ellipse_foci_coordinates :
  (∀ (x y : ℝ), ellipse_eq x y → False) →
  ∃ (c : ℝ), c = 12 ∧ 
  ((0, c) = (0, 12) ∧ (0, -c) = (0, -12)) := 
by
  sorry

end ellipse_foci_coordinates_l74_74414


namespace carpet_width_l74_74278

theorem carpet_width
  (carpet_percentage : ℝ)
  (living_room_area : ℝ)
  (carpet_length : ℝ) :
  carpet_percentage = 0.30 →
  living_room_area = 120 →
  carpet_length = 9 →
  carpet_percentage * living_room_area / carpet_length = 4 :=
by
  sorry

end carpet_width_l74_74278


namespace john_total_jury_duty_days_l74_74786

-- Definitions based on the given conditions
def jury_selection_days : ℕ := 2
def trial_length_multiplier : ℕ := 4
def deliberation_days : ℕ := 6
def deliberation_hours_per_day : ℕ := 16
def hours_in_a_day : ℕ := 24

-- Define the total days John spends on jury duty
def total_days_on_jury_duty : ℕ :=
  jury_selection_days +
  (trial_length_multiplier * jury_selection_days) +
  (deliberation_days * deliberation_hours_per_day) / hours_in_a_day

-- The theorem to prove
theorem john_total_jury_duty_days : total_days_on_jury_duty = 14 := by
  sorry

end john_total_jury_duty_days_l74_74786


namespace vector_b_norm_range_l74_74438

variable (a b : ℝ × ℝ)
variable (norm_a : ‖a‖ = 1)
variable (norm_sum : ‖a + b‖ = 2)

theorem vector_b_norm_range : 1 ≤ ‖b‖ ∧ ‖b‖ ≤ 3 :=
sorry

end vector_b_norm_range_l74_74438


namespace mike_spent_on_mower_blades_l74_74618

theorem mike_spent_on_mower_blades (x : ℝ) 
  (initial_money : ℝ := 101) 
  (cost_of_games : ℝ := 54) 
  (games : ℝ := 9) 
  (price_per_game : ℝ := 6) 
  (h1 : 101 - x = 54) :
  x = 47 := 
by
  sorry

end mike_spent_on_mower_blades_l74_74618


namespace arithmetic_mean_reciprocals_primes_l74_74870

theorem arithmetic_mean_reciprocals_primes : 
  let p := [2, 3, 5, 7] in 
  let reciprocals := p.map (λ n => 1 / (n : ℚ)) in
  (reciprocals.sum / reciprocals.length) = (247 / 840 : ℚ) :=
by
  sorry

end arithmetic_mean_reciprocals_primes_l74_74870


namespace sum_of_reciprocals_of_roots_l74_74564

theorem sum_of_reciprocals_of_roots (r1 r2 : ℝ) (h1 : r1 * r2 = 7) (h2 : r1 + r2 = 16) :
  (1 / r1) + (1 / r2) = 16 / 7 :=
by
  sorry

end sum_of_reciprocals_of_roots_l74_74564


namespace distance_to_first_museum_l74_74333

theorem distance_to_first_museum (x : ℝ) 
  (dist_second_museum : ℝ) 
  (total_distance : ℝ) 
  (h1 : dist_second_museum = 15) 
  (h2 : total_distance = 40) 
  (h3 : 2 * x + 2 * dist_second_museum = total_distance) : x = 5 :=
by 
  sorry

end distance_to_first_museum_l74_74333


namespace complex_integral_cosh_l74_74151

theorem complex_integral_cosh (R : ℝ) (hR : R = 2) :
  ∮ (|z|=R) (λ z, (cosh (I * z)) / (z^2 + 4 * z + 3)) = real.pi * I * real.cos 1 :=
by
  sorry

end complex_integral_cosh_l74_74151


namespace total_weight_of_remaining_eggs_is_correct_l74_74472

-- Define the initial conditions and the question as Lean definitions
def total_eggs : Nat := 12
def weight_per_egg : Nat := 10
def num_boxes : Nat := 4
def melted_boxes : Nat := 1

-- Calculate the total weight of the eggs
def total_weight : Nat := total_eggs * weight_per_egg

-- Calculate the number of eggs per box
def eggs_per_box : Nat := total_eggs / num_boxes

-- Calculate the weight per box
def weight_per_box : Nat := eggs_per_box * weight_per_egg

-- Calculate the number of remaining boxes after one is tossed out
def remaining_boxes : Nat := num_boxes - melted_boxes

-- Calculate the total weight of the remaining chocolate eggs
def remaining_weight : Nat := remaining_boxes * weight_per_box

-- The proof task
theorem total_weight_of_remaining_eggs_is_correct : remaining_weight = 90 := by
  sorry

end total_weight_of_remaining_eggs_is_correct_l74_74472


namespace smallest_positive_integer_l74_74514

theorem smallest_positive_integer (x : ℕ) (hx_pos : x > 0) (h : x < 15) : x = 1 :=
by
  sorry

end smallest_positive_integer_l74_74514


namespace negation_exists_x_squared_leq_abs_x_l74_74812

theorem negation_exists_x_squared_leq_abs_x :
  (¬ ∃ x : ℝ, x ∈ Set.Ioo (-1 : ℝ) (0 : ℝ) ∧ x^2 ≤ |x|) ↔ (∀ x : ℝ, x ∈ Set.Ioo (-1 : ℝ) (0 : ℝ) → x^2 > |x|) :=
by
  sorry

end negation_exists_x_squared_leq_abs_x_l74_74812


namespace simplify_fraction_lemma_l74_74708

noncomputable def simplify_fraction (a : ℝ) (h : a ≠ 5) : ℝ :=
  (a^2 - 5 * a) / (a - 5)

theorem simplify_fraction_lemma (a : ℝ) (h : a ≠ 5) : simplify_fraction a h = a := by
  sorry

end simplify_fraction_lemma_l74_74708


namespace arcsin_range_l74_74727

theorem arcsin_range (α : ℝ ) (x : ℝ ) (h₁ : x = Real.cos α) (h₂ : -Real.pi / 4 ≤ α ∧ α ≤ 3 * Real.pi / 4) : 
-Real.pi / 4 ≤ Real.arcsin x ∧ Real.arcsin x ≤ Real.pi / 2 :=
sorry

end arcsin_range_l74_74727


namespace distance_between_points_l74_74419

noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem distance_between_points :
  distance 2 5 5 1 = 5 :=
by
  sorry

end distance_between_points_l74_74419


namespace alexander_spends_total_amount_l74_74547

theorem alexander_spends_total_amount :
  (5 * 1) + (2 * 2) = 9 :=
by
  sorry

end alexander_spends_total_amount_l74_74547


namespace repaved_inches_before_today_l74_74692

theorem repaved_inches_before_today :
  let A := 4000
  let B := 3500
  let C := 2500
  let repaved_A := 0.70 * A
  let repaved_B := 0.60 * B
  let repaved_C := 0.80 * C
  let total_repaved_before := repaved_A + repaved_B + repaved_C
  let repaved_today := 950
  let new_total_repaved := total_repaved_before + repaved_today
  new_total_repaved - repaved_today = 6900 :=
by
  sorry

end repaved_inches_before_today_l74_74692


namespace roses_and_orchids_difference_l74_74969

theorem roses_and_orchids_difference :
  let roses_now := 11
  let orchids_now := 20
  orchids_now - roses_now = 9 := 
by
  sorry

end roses_and_orchids_difference_l74_74969


namespace P_subsetneq_M_l74_74207

def M := {x : ℝ | x > 1}
def P := {x : ℝ | x^2 - 6*x + 9 = 0}

theorem P_subsetneq_M : P ⊂ M := by
  sorry

end P_subsetneq_M_l74_74207


namespace proposition_and_implication_l74_74349

theorem proposition_and_implication
  (m : ℝ)
  (h1 : 5/4 * (m^2 + m) > 0)
  (h2 : 1 + 9 - 4 * (5/4 * (m^2 + m)) > 0)
  (h3 : m + 3/2 ≥ 0)
  (h4 : m - 1/2 ≤ 0) :
  (-3/2 ≤ m ∧ m < -1) ∨ (0 < m ∧ m ≤ 1/2) :=
sorry

end proposition_and_implication_l74_74349


namespace remainder_when_divided_by_44_l74_74696

theorem remainder_when_divided_by_44 (N : ℕ) (Q : ℕ) (R : ℕ)
  (h1 : N = 44 * 432 + R)
  (h2 : N = 31 * Q + 5) :
  R = 2 :=
sorry

end remainder_when_divided_by_44_l74_74696


namespace angle_B_is_pi_over_3_l74_74767

theorem angle_B_is_pi_over_3
  (A B C a b c : ℝ)
  (h1 : b * Real.cos B = (a * Real.cos C + c * Real.cos A) / 2)
  (h2 : 0 < B)
  (h3 : B < Real.pi)
  (h4 : 0 < A)
  (h5 : A < Real.pi)
  (h6 : 0 < C)
  (h7 : C < Real.pi) :
  B = Real.pi / 3 :=
by
  sorry

end angle_B_is_pi_over_3_l74_74767


namespace max_fraction_l74_74026

theorem max_fraction (x y : ℝ) (hx : -3 ≤ x ∧ x ≤ -1) (hy : 1 ≤ y ∧ y ≤ 3) : 
  ∃ k, k = (x + y) / x ∧ k ≤ -2 := 
sorry

end max_fraction_l74_74026


namespace sum_xyz_l74_74035

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem sum_xyz :
  (∀ x y z : ℝ,
  log_base 3 (log_base 4 (log_base 5 x)) = 0 ∧
  log_base 4 (log_base 5 (log_base 3 y)) = 0 ∧
  log_base 5 (log_base 3 (log_base 4 z)) = 0 →
  x + y + z = 932) := 
by
  sorry

end sum_xyz_l74_74035


namespace abs_nonneg_l74_74373

theorem abs_nonneg (a : ℝ) : 0 ≤ |a| :=
sorry

end abs_nonneg_l74_74373


namespace all_rationals_on_number_line_l74_74517

theorem all_rationals_on_number_line :
  ∀ q : ℚ, ∃ p : ℝ, p = ↑q :=
by
  sorry

end all_rationals_on_number_line_l74_74517


namespace number_of_5_letter_words_with_at_least_one_vowel_l74_74582

theorem number_of_5_letter_words_with_at_least_one_vowel :
  let letters := ['A', 'B', 'C', 'D', 'E', 'F', 'G']
  let vowels := ['A', 'E']
  ∃ n : ℕ, n = 7^5 - 5^5 ∧ n = 13682 :=
by
  sorry

end number_of_5_letter_words_with_at_least_one_vowel_l74_74582


namespace combined_weight_after_removal_l74_74522

theorem combined_weight_after_removal (weight_sugar weight_salt weight_removed : ℕ) 
                                       (h_sugar : weight_sugar = 16)
                                       (h_salt : weight_salt = 30)
                                       (h_removed : weight_removed = 4) : 
                                       (weight_sugar + weight_salt) - weight_removed = 42 :=
by {
  sorry
}

end combined_weight_after_removal_l74_74522


namespace correct_statement_C_l74_74829

theorem correct_statement_C : (∃ x : ℝ, x^2 = 25 ∧ (x = 5 ∨ x = -5)) :=
by
  sorry

end correct_statement_C_l74_74829


namespace gcf_72_108_l74_74110

def gcf (a b : ℕ) : ℕ := 
  Nat.gcd a b

theorem gcf_72_108 : gcf 72 108 = 36 := by
  sorry

end gcf_72_108_l74_74110


namespace simplify_fraction_l74_74802

theorem simplify_fraction:
  (4^5 + 4^3) / (4^4 - 4^2) = 68 / 15 := by
  sorry

end simplify_fraction_l74_74802


namespace determine_m_l74_74710

def f (x : ℝ) := 5 * x^2 + 3 * x + 7
def g (x : ℝ) (m : ℝ) := 2 * x^2 - m * x + 1

theorem determine_m (m : ℝ) : f 5 - g 5 m = 55 → m = -7 :=
by
  unfold f
  unfold g
  sorry

end determine_m_l74_74710


namespace motorcyclist_average_speed_l74_74851

theorem motorcyclist_average_speed :
  let distance_ab := 120
  let speed_ab := 45
  let distance_bc := 130
  let speed_bc := 60
  let distance_cd := 150
  let speed_cd := 50
  let time_ab := distance_ab / speed_ab
  let time_bc := distance_bc / speed_bc
  let time_cd := distance_cd / speed_cd
  (time_ab = time_bc + 2)
  → (time_cd = time_ab / 2)
  → avg_speed = (distance_ab + distance_bc + distance_cd) / (time_ab + time_bc + time_cd)
  → avg_speed = 2400 / 47 := sorry

end motorcyclist_average_speed_l74_74851


namespace solve_for_five_minus_a_l74_74442

theorem solve_for_five_minus_a (a b : ℤ) 
  (h1 : 5 + a = 6 - b)
  (h2 : 6 + b = 9 + a) : 
  5 - a = 6 := 
by 
  sorry

end solve_for_five_minus_a_l74_74442


namespace part1_proof_l74_74462

variable (α β t x1 x2 : ℝ)

-- Conditions
def quadratic_roots := 2 * α ^ 2 - t * α - 2 = 0 ∧ 2 * β ^ 2 - t * β - 2 = 0
def roots_relation := α + β = t / 2 ∧ α * β = -1
def points_in_interval := α < β ∧ α ≤ x1 ∧ x1 ≤ β ∧ α ≤ x2 ∧ x2 ≤ β ∧ x1 ≠ x2

-- Proof of Part 1
theorem part1_proof (h1 : quadratic_roots α β t) (h2 : roots_relation α β t)
                    (h3 : points_in_interval α β x1 x2) : 
                    4 * x1 * x2 - t * (x1 + x2) - 4 < 0 := 
sorry

end part1_proof_l74_74462


namespace max_xy_l74_74166

theorem max_xy (x y : ℝ) (hx_pos : 0 < x) (hy_pos : 0 < y) (h_eq : 2 * x + 3 * y = 6) : 
  xy ≤ (3/2) :=
sorry

end max_xy_l74_74166


namespace complement_intersection_l74_74463

open Set

variable (U M N : Set ℕ)
variable (H₁ : U = {1, 2, 3, 4, 5, 6})
variable (H₂ : M = {1, 2, 3, 5})
variable (H₃ : N = {1, 3, 4, 6})

theorem complement_intersection :
  (U \ (M ∩ N)) = {2, 4, 5, 6} :=
by
  sorry

end complement_intersection_l74_74463


namespace inequality_proof_l74_74344

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_cond : x ≥ y + z) :
  (x + y) / z + (y + z) / x + (z + x) / y ≥ 7 :=
by
  sorry

end inequality_proof_l74_74344


namespace solve_for_z_l74_74480

theorem solve_for_z (i : ℂ) (z : ℂ) (h : 3 - 5 * i * z = -2 + 5 * i * z) (h_i : i^2 = -1) :
  z = -i / 2 :=
by {
  sorry
}

end solve_for_z_l74_74480


namespace value_of_expression_l74_74669

theorem value_of_expression :
  (3^2 - 3) - (4^2 - 4) + (5^2 - 5) - (6^2 - 6) = -16 :=
by
  sorry

end value_of_expression_l74_74669


namespace sheets_in_backpack_l74_74614

-- Definitions for the conditions
def total_sheets := 91
def desk_sheets := 50

-- Theorem statement with the goal
theorem sheets_in_backpack (total_sheets : ℕ) (desk_sheets : ℕ) (h1 : total_sheets = 91) (h2 : desk_sheets = 50) : 
  ∃ backpack_sheets : ℕ, backpack_sheets = total_sheets - desk_sheets ∧ backpack_sheets = 41 :=
by
  -- The proof is omitted here
  sorry

end sheets_in_backpack_l74_74614


namespace age_of_new_person_l74_74681

theorem age_of_new_person (avg_age : ℝ) (x : ℝ) 
  (h1 : 10 * avg_age - (10 * (avg_age - 3)) = 42 - x) : 
  x = 12 := 
by
  sorry

end age_of_new_person_l74_74681


namespace absolute_value_inequality_solution_l74_74718

theorem absolute_value_inequality_solution (x : ℝ) : |2*x - 1| < 3 ↔ -1 < x ∧ x < 2 := 
sorry

end absolute_value_inequality_solution_l74_74718


namespace expression_value_l74_74982

theorem expression_value (a b : ℕ) (h₁ : a = 37) (h₂ : b = 12) : 
  (a + b)^2 - (a^2 + b^2) = 888 := by
  sorry

end expression_value_l74_74982


namespace fertilizer_production_l74_74393

theorem fertilizer_production (daily_production : ℕ) (days : ℕ) (total_production : ℕ) 
  (h1 : daily_production = 105) 
  (h2 : days = 24) 
  (h3 : total_production = daily_production * days) : 
  total_production = 2520 := 
  by 
  -- skipping the proof
  sorry

end fertilizer_production_l74_74393


namespace no_primes_in_Q_plus_m_l74_74608

def Q : ℕ := 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

def is_prime (n : ℕ) : Prop := Nat.Prime n

theorem no_primes_in_Q_plus_m (m : ℕ) (hm : 2 ≤ m ∧ m ≤ 32) : ¬is_prime (Q + m) :=
by
  sorry  -- Proof would be provided here

end no_primes_in_Q_plus_m_l74_74608


namespace age_condition_l74_74730

theorem age_condition (x y z : ℕ) (h1 : x > y) : 
  (z > y) ↔ (y + z > 2 * x) ∧ (∀ x y z, y + z > 2 * x → z > y) := sorry

end age_condition_l74_74730


namespace total_days_on_jury_duty_l74_74784

-- Definitions based on conditions
def jurySelectionDays := 2
def juryDeliberationDays := 6
def deliberationHoursPerDay := 16
def trialDurationMultiplier := 4

-- Calculate the total number of hours spent in deliberation
def totalDeliberationHours := juryDeliberationDays * 24

-- Calculate the number of days spent in deliberation based on hours per day
def deliberationDays := totalDeliberationHours / deliberationHoursPerDay

-- Calculate the trial days based on trial duration multiplier
def trialDays := jurySelectionDays * trialDurationMultiplier

-- Calculate the total days on jury duty
def totalJuryDutyDays := jurySelectionDays + trialDays + deliberationDays

theorem total_days_on_jury_duty : totalJuryDutyDays = 19 := by
  sorry

end total_days_on_jury_duty_l74_74784


namespace outOfPocketCost_l74_74502

noncomputable def visitCost : ℝ := 300
noncomputable def castCost : ℝ := 200
noncomputable def insuranceCoverage : ℝ := 0.60

theorem outOfPocketCost : (visitCost + castCost - (visitCost + castCost) * insuranceCoverage) = 200 := by
  sorry

end outOfPocketCost_l74_74502


namespace octagon_diagonal_ratio_l74_74194

theorem octagon_diagonal_ratio (P : ℝ → ℝ → Prop) (d1 d2 : ℝ) (h1 : P d1 d2) : d1 / d2 = Real.sqrt 2 / 2 :=
sorry

end octagon_diagonal_ratio_l74_74194


namespace rectangle_side_greater_than_twelve_l74_74518

theorem rectangle_side_greater_than_twelve (a b : ℕ) (h1 : a ≠ b) (h2 : a * b = 6 * (a + b)) : a > 12 ∨ b > 12 :=
sorry

end rectangle_side_greater_than_twelve_l74_74518


namespace brush_length_percentage_increase_l74_74709

-- Define the length of Carla's brush in inches
def carla_brush_length_in_inches : ℝ := 12

-- Define the conversion factor from inches to centimeters
def inch_to_cm : ℝ := 2.54

-- Define the length of Carmen's brush in centimeters
def carmen_brush_length_in_cm : ℝ := 45

-- Noncomputable definition to calculate the percentage increase
noncomputable def percentage_increase : ℝ :=
  let carla_brush_length_in_cm := carla_brush_length_in_inches * inch_to_cm
  (carmen_brush_length_in_cm - carla_brush_length_in_cm) / carla_brush_length_in_cm * 100

-- Statement to prove the percentage increase is 47.6%
theorem brush_length_percentage_increase :
  percentage_increase = 47.6 :=
sorry

end brush_length_percentage_increase_l74_74709


namespace original_selling_price_is_800_l74_74007

-- Let CP denote the cost price
variable (CP : ℝ)

-- Condition 1: Selling price with a profit of 25%
def selling_price_with_profit (CP : ℝ) : ℝ := 1.25 * CP

-- Condition 2: Selling price with a loss of 35%
def selling_price_with_loss (CP : ℝ) : ℝ := 0.65 * CP

-- Given selling price with loss is Rs. 416
axiom loss_price_is_416 : selling_price_with_loss CP = 416

-- We need to prove the original selling price (with profit) is Rs. 800
theorem original_selling_price_is_800 : selling_price_with_profit CP = 800 :=
by sorry

end original_selling_price_is_800_l74_74007


namespace correct_statement_C_l74_74830

theorem correct_statement_C : (∃ x : ℝ, x^2 = 25 ∧ (x = 5 ∨ x = -5)) :=
by
  sorry

end correct_statement_C_l74_74830


namespace elberta_amount_l74_74439

theorem elberta_amount (grannySmith_amount : ℝ) (Anjou_factor : ℝ) (extra_amount : ℝ) :
  grannySmith_amount = 45 →
  Anjou_factor = 1 / 4 →
  extra_amount = 4 →
  (extra_amount + Anjou_factor * grannySmith_amount) = 15.25 :=
by
  intros h_grannySmith h_AnjouFactor h_extraAmount
  sorry

end elberta_amount_l74_74439


namespace stella_profit_l74_74957

-- Definitions based on the conditions
def number_of_dolls := 6
def price_per_doll := 8
def number_of_clocks := 4
def price_per_clock := 25
def number_of_glasses := 8
def price_per_glass := 6
def number_of_vases := 3
def price_per_vase := 12
def number_of_postcards := 10
def price_per_postcard := 3
def cost_of_merchandise := 250

-- Calculations based on given problem and solution
def revenue_from_dolls := number_of_dolls * price_per_doll
def revenue_from_clocks := number_of_clocks * price_per_clock
def revenue_from_glasses := number_of_glasses * price_per_glass
def revenue_from_vases := number_of_vases * price_per_vase
def revenue_from_postcards := number_of_postcards * price_per_postcard
def total_revenue := revenue_from_dolls + revenue_from_clocks + revenue_from_glasses + revenue_from_vases + revenue_from_postcards
def profit := total_revenue - cost_of_merchandise

-- Main theorem statement
theorem stella_profit : profit = 12 := by
  sorry

end stella_profit_l74_74957


namespace smallest_four_digit_divisible_l74_74018

def distinct_nonzero_digits (n : ℕ) : Prop :=
  let digits := List.map (λ c, c.to_nat - '0'.to_nat) (n.digits 10)
  digits.length = digits.nodup.length ∧ 0 ∉ digits

def divisible_by_digits (n : ℕ) : Prop :=
  ∀ d ∈ n.digits 10, d ≠ 0 → n % (d.to_nat - '0'.to_nat) = 0

theorem smallest_four_digit_divisible :
  ∀ n ∈ range 1000 10000,
    distinct_nonzero_digits n → divisible_by_digits n → n = 1362 :=
by
  sorry

end smallest_four_digit_divisible_l74_74018


namespace solve_for_a_l74_74413

def op (a b : ℝ) : ℝ := 3 * a - 2 * b ^ 2

theorem solve_for_a (a : ℝ) : op a 3 = 15 → a = 11 :=
by
  intro h
  rw [op] at h
  sorry

end solve_for_a_l74_74413


namespace cylinder_volume_ratio_l74_74130

noncomputable def volume_of_cylinder (height : ℝ) (circumference : ℝ) : ℝ := 
  let r := circumference / (2 * Real.pi)
  Real.pi * r^2 * height

theorem cylinder_volume_ratio :
  let V1 := volume_of_cylinder 10 6
  let V2 := volume_of_cylinder 6 10
  max V1 V2 / min V1 V2 = 5 / 3 :=
by
  let V1 := volume_of_cylinder 10 6
  let V2 := volume_of_cylinder 6 10
  have hV1 : V1 = 90 / Real.pi := sorry
  have hV2 : V2 = 150 / Real.pi := sorry
  sorry

end cylinder_volume_ratio_l74_74130


namespace toad_difference_l74_74236

variables (Tim_toads Jim_toads Sarah_toads : ℕ)

theorem toad_difference (h1 : Tim_toads = 30) 
                        (h2 : Jim_toads > Tim_toads) 
                        (h3 : Sarah_toads = 2 * Jim_toads) 
                        (h4 : Sarah_toads = 100) :
  Jim_toads - Tim_toads = 20 :=
by
  -- The next lines are placeholders for the logical steps which need to be proven
  sorry

end toad_difference_l74_74236


namespace solve_equation_l74_74739

theorem solve_equation (x y : ℝ) (h : 2 * x^2 + y^2 + 3 * x * y + 3 * x + y = 2):
  y = -x - 2 ∨ y = -2 * x + 1 :=
sorry


end solve_equation_l74_74739


namespace simplify_and_evaluate_l74_74805

theorem simplify_and_evaluate (x : ℝ) (hx : x = Real.sqrt 2 + 1) :
  (x + 1) / x / (x - (1 + x^2) / (2 * x)) = Real.sqrt 2 :=
by
  sorry

end simplify_and_evaluate_l74_74805


namespace solution_n_value_l74_74638

open BigOperators

noncomputable def problem_statement (a b n : ℝ) : Prop :=
  ∃ (A B : ℝ), A = Real.log a ∧ B = Real.log b ∧
    (7 * A + 15 * B) - (4 * A + 9 * B) = (11 * A + 20 * B) - (7 * A + 15 * B) ∧
    (4 + 135) * B = Real.log (b^n)

theorem solution_n_value (a b : ℝ) (h_pos : a > 0) (h_pos_b : b > 0) :
  problem_statement a b 139 :=
by
  sorry

end solution_n_value_l74_74638


namespace solve_quadratic_l74_74481

theorem solve_quadratic : ∀ x, x^2 - 4 * x + 3 = 0 ↔ x = 3 ∨ x = 1 := 
by
  sorry

end solve_quadratic_l74_74481


namespace fraction_multiplication_l74_74008

theorem fraction_multiplication :
  (3 / 4) ^ 5 * (4 / 3) ^ 2 = 8 / 19 :=
by
  sorry

end fraction_multiplication_l74_74008


namespace no_solution_inequality_l74_74918

theorem no_solution_inequality (a : ℝ) : 
  (∀ x : ℝ, a * x^2 - |x + 1| + 2 * a ≥ 0) → a ≥ (Real.sqrt 3 + 1) / 4 :=
by
  intro h
  sorry

end no_solution_inequality_l74_74918


namespace sector_area_l74_74038

theorem sector_area (θ : ℝ) (r : ℝ) (hθ : θ = π / 3) (hr : r = 4) : 
  (1/2) * (r * θ) * r = 8 * π / 3 :=
by
  -- Implicitly use the given values of θ and r by substituting them in the expression.
  sorry

end sector_area_l74_74038


namespace simple_interest_rate_l74_74677

theorem simple_interest_rate (P : ℝ) (T : ℝ) (R : ℝ) (H : T = 4 ∧ (7 / 6) * P = P + (P * R * T / 100)) :
  R = 4.17 :=
by
  sorry

end simple_interest_rate_l74_74677


namespace molecular_weight_of_compound_l74_74116

theorem molecular_weight_of_compound :
  let Cu_atoms := 2
  let C_atoms := 3
  let O_atoms := 5
  let N_atoms := 1
  let atomic_weight_Cu := 63.546
  let atomic_weight_C := 12.011
  let atomic_weight_O := 15.999
  let atomic_weight_N := 14.007
  Cu_atoms * atomic_weight_Cu +
  C_atoms * atomic_weight_C +
  O_atoms * atomic_weight_O +
  N_atoms * atomic_weight_N = 257.127 :=
by
  sorry

end molecular_weight_of_compound_l74_74116


namespace wire_attachment_distance_l74_74211

theorem wire_attachment_distance :
  ∃ x : ℝ, 
    (∀ z y : ℝ, z = Real.sqrt (x ^ 2 + 3.6 ^ 2) ∧ y = Real.sqrt ((x + 5) ^ 2 + 3.6 ^ 2) →
      z + y = 13) ∧
    abs ((x : ℝ) - 2.7) < 0.01 := -- Assuming numerical closeness within a small epsilon for practical solutions.
sorry -- Proof not provided.

end wire_attachment_distance_l74_74211


namespace trailing_zeroes_base_81_l74_74181

theorem trailing_zeroes_base_81 (n : ℕ) : (n = 15) → num_trailing_zeroes_base 81 (factorial n) = 1 :=
by
  sorry

end trailing_zeroes_base_81_l74_74181


namespace gcf_72_108_l74_74103

theorem gcf_72_108 : Nat.gcd 72 108 = 36 := by
  sorry

end gcf_72_108_l74_74103


namespace range_of_a_l74_74624

def p (a : ℝ) : Prop := ∀ x : ℝ, x^2 + 2 * a * x + 4 > 0
def q (a : ℝ) : Prop := ∀ x : ℝ, (3 - 2 * a) ^ x > 0 -- using our characterization for 'increasing'

theorem range_of_a (a : ℝ) : (p a ∨ q a) ∧ ¬ (p a ∧ q a) ↔ (a ≤ -2 ∨ (1 ≤ a ∧ a < 2)) :=
by
  sorry

end range_of_a_l74_74624


namespace marble_count_l74_74770

theorem marble_count (p y v : ℝ) (h1 : y + v = 10) (h2 : p + v = 12) (h3 : p + y = 5) :
  p + y + v = 13.5 :=
sorry

end marble_count_l74_74770


namespace speed_of_car_first_hour_98_l74_74082

def car_speed_in_first_hour_is_98 (x : ℕ) : Prop :=
  (70 + x) / 2 = 84 → x = 98

theorem speed_of_car_first_hour_98 (x : ℕ) (h : car_speed_in_first_hour_is_98 x) : x = 98 :=
  by
  sorry

end speed_of_car_first_hour_98_l74_74082


namespace rotated_triangle_surface_area_l74_74776

theorem rotated_triangle_surface_area :
  ∀ (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C]
    (ACLength : ℝ) (BCLength : ℝ) (right_angle : ℝ -> ℝ -> ℝ -> Prop)
    (pi_def : Real) (surface_area : ℝ -> ℝ -> ℝ),
    (right_angle 90 0 90) → (ACLength = 3) → (BCLength = 4) →
    surface_area ACLength BCLength = 24 * pi_def  :=
by
  sorry

end rotated_triangle_surface_area_l74_74776


namespace largest_common_term_l74_74634

/-- The arithmetic progression sequence1 --/
def sequence1 (n : ℕ) : ℤ := 4 + 5 * n

/-- The arithmetic progression sequence2 --/
def sequence2 (n : ℕ) : ℤ := 5 + 8 * n

/-- The common term condition for sequence1 --/
def common_term_condition1 (a : ℤ) : Prop := ∃ n : ℕ, a = sequence1 n

/-- The common term condition for sequence2 --/
def common_term_condition2 (a : ℤ) : Prop := ∃ n : ℕ, a = sequence2 n

/-- The largest common term less than 1000 --/
def is_largest_common_term (a : ℤ) : Prop :=
  common_term_condition1 a ∧ common_term_condition2 a ∧ a < 1000 ∧
  ∀ b : ℤ, common_term_condition1 b ∧ common_term_condition2 b ∧ b < 1000 → b ≤ a

/-- Lean theorem statement --/
theorem largest_common_term :
  ∃ a : ℤ, is_largest_common_term a ∧ a = 989 :=
sorry

end largest_common_term_l74_74634


namespace find_a_plus_b_l74_74792

theorem find_a_plus_b (a b : ℝ) (f g : ℝ → ℝ) (h1 : ∀ x, f x = a * x + b) (h2 : ∀ x, g x = 3 * x - 4) 
(h3 : ∀ x, g (f x) = 4 * x + 5) : a + b = 13 / 3 :=
sorry

end find_a_plus_b_l74_74792


namespace jeff_cat_shelter_l74_74930

theorem jeff_cat_shelter :
  let initial_cats := 20
  let monday_cats := 2
  let tuesday_cats := 1
  let people_adopted := 3
  let cats_per_person := 2
  let total_cats := initial_cats + monday_cats + tuesday_cats
  let adopted_cats := people_adopted * cats_per_person
  total_cats - adopted_cats = 17 := 
by
  sorry

end jeff_cat_shelter_l74_74930


namespace bags_weight_after_removal_l74_74525

theorem bags_weight_after_removal (sugar_weight salt_weight weight_removed : ℕ) (h1 : sugar_weight = 16) (h2 : salt_weight = 30) (h3 : weight_removed = 4) :
  sugar_weight + salt_weight - weight_removed = 42 := by
  sorry

end bags_weight_after_removal_l74_74525


namespace prime_factors_of_four_consecutive_integers_sum_l74_74091

theorem prime_factors_of_four_consecutive_integers_sum : 
  ∃ p, Prime p ∧ ∀ n : ℤ, p ∣ ((n-2) + (n-1) + n + (n+1)) :=
by {
  use 2,
  split,
  { norm_num },
  { intro n,
    simp,
    exact dvd.intro (2 * n - 1) rfl }
}

end prime_factors_of_four_consecutive_integers_sum_l74_74091


namespace taxi_ride_cost_l74_74273

theorem taxi_ride_cost (initial_cost : ℝ) (cost_first_3_miles : ℝ) (rate_first_3_miles : ℝ) (rate_after_3_miles : ℝ) (total_miles : ℝ) (remaining_miles : ℝ) :
  initial_cost = 2.00 ∧ rate_first_3_miles = 0.30 ∧ rate_after_3_miles = 0.40 ∧ total_miles = 8 ∧ total_miles - 3 = remaining_miles →
  initial_cost + 3 * rate_first_3_miles + remaining_miles * rate_after_3_miles = 4.90 :=
sorry

end taxi_ride_cost_l74_74273


namespace waiter_initial_tables_l74_74274

theorem waiter_initial_tables
  (T : ℝ)
  (H1 : (T - 12.0) * 8.0 = 256) :
  T = 44.0 :=
sorry

end waiter_initial_tables_l74_74274


namespace plumber_fix_cost_toilet_l74_74852

noncomputable def fixCost_Sink : ℕ := 30
noncomputable def fixCost_Shower : ℕ := 40

theorem plumber_fix_cost_toilet
  (T : ℕ)
  (Earnings1 : ℕ := 3 * T + 3 * fixCost_Sink)
  (Earnings2 : ℕ := 2 * T + 5 * fixCost_Sink)
  (Earnings3 : ℕ := T + 2 * fixCost_Shower + 3 * fixCost_Sink)
  (MaxEarnings : ℕ := 250) :
  Earnings2 = MaxEarnings → T = 50 :=
by
  sorry

end plumber_fix_cost_toilet_l74_74852


namespace remaining_volume_of_cube_l74_74269

theorem remaining_volume_of_cube :
  let s := 6
  let r := 3
  let h := 6
  let V_cube := s^3
  let V_cylinder := Real.pi * (r^2) * h
  V_cube - V_cylinder = 216 - 54 * Real.pi :=
by
  sorry

end remaining_volume_of_cube_l74_74269


namespace sum_of_divisor_and_quotient_is_correct_l74_74672

theorem sum_of_divisor_and_quotient_is_correct (divisor quotient : ℕ)
  (h1 : 1000 ≤ divisor ∧ divisor < 10000) -- Divisor is a four-digit number.
  (h2 : quotient * divisor + remainder = original_number) -- Division condition (could be more specific)
  (h3 : remainder < divisor) -- Remainder condition
  (h4 : original_number = 82502) -- Given original number
  : divisor + quotient = 723 := 
sorry

end sum_of_divisor_and_quotient_is_correct_l74_74672


namespace expression_evaluation_l74_74707

theorem expression_evaluation :
  (-2: ℤ)^3 + ((36: ℚ) / (3: ℚ)^2 * (-1 / 2: ℚ)) + abs (-5: ℤ) = -5 :=
by
  sorry

end expression_evaluation_l74_74707


namespace seven_nat_sum_divisible_by_5_l74_74954

theorem seven_nat_sum_divisible_by_5 
  (a b c d e f g : ℕ)
  (h1 : (b + c + d + e + f + g) % 5 = 0)
  (h2 : (a + c + d + e + f + g) % 5 = 0)
  (h3 : (a + b + d + e + f + g) % 5 = 0)
  (h4 : (a + b + c + e + f + g) % 5 = 0)
  (h5 : (a + b + c + d + f + g) % 5 = 0)
  (h6 : (a + b + c + d + e + g) % 5 = 0)
  (h7 : (a + b + c + d + e + f) % 5 = 0)
  : 
  a % 5 = 0 ∧ b % 5 = 0 ∧ c % 5 = 0 ∧ d % 5 = 0 ∧ e % 5 = 0 ∧ f % 5 = 0 ∧ g % 5 = 0 :=
sorry

end seven_nat_sum_divisible_by_5_l74_74954


namespace prime_factors_of_four_consecutive_integers_sum_l74_74092

theorem prime_factors_of_four_consecutive_integers_sum : 
  ∃ p, Prime p ∧ ∀ n : ℤ, p ∣ ((n-2) + (n-1) + n + (n+1)) :=
by {
  use 2,
  split,
  { norm_num },
  { intro n,
    simp,
    exact dvd.intro (2 * n - 1) rfl }
}

end prime_factors_of_four_consecutive_integers_sum_l74_74092


namespace gcf_72_108_l74_74106

theorem gcf_72_108 : Nat.gcd 72 108 = 36 := by
  sorry

end gcf_72_108_l74_74106


namespace total_cost_of_items_l74_74961

theorem total_cost_of_items (m n : ℕ) : (8 * m + 5 * n) = 8 * m + 5 * n := 
by sorry

end total_cost_of_items_l74_74961


namespace sufficient_but_not_necessary_condition_l74_74989

theorem sufficient_but_not_necessary_condition (x : ℝ) :
  (x > 1 → |x| > 1) ∧ ¬ (|x| > 1 → x > 1) :=
by
  sorry

end sufficient_but_not_necessary_condition_l74_74989


namespace polynomial_roots_absolute_sum_l74_74303

theorem polynomial_roots_absolute_sum (p q r : ℤ) (h1 : p + q + r = 0) (h2 : p * q + q * r + r * p = -2027) :
  |p| + |q| + |r| = 98 := 
sorry

end polynomial_roots_absolute_sum_l74_74303


namespace max_tan_A_correct_l74_74600

noncomputable def max_tan_A {A B C : ℝ} (AB BC : ℝ) (hAB : AB = 25) (hBC : BC = 20) : Prop :=
  ∀ C, ∃ (A B: ℝ), AC = (Real.sqrt (AB^2 - BC^2)) ∧ (AC = 15) → Real.tan (A) = 4 / 3

theorem max_tan_A_correct {A B C : ℝ} : max_tan_A 25 20 :=
by { sorry }

end max_tan_A_correct_l74_74600


namespace fraction_value_l74_74755

theorem fraction_value (x : ℝ) (h : 1 - 6 / x + 9 / (x^2) = 0) : 2 / x = 2 / 3 :=
  sorry

end fraction_value_l74_74755


namespace points_scored_by_others_l74_74808

-- Define the conditions as hypothesis
variables (P_total P_Jessie : ℕ)
  (H1 : P_total = 311)
  (H2 : P_Jessie = 41)
  (H3 : ∀ P_Lisa P_Devin: ℕ, P_Lisa = P_Jessie ∧ P_Devin = P_Jessie)

-- Define what we need to prove
theorem points_scored_by_others (P_others : ℕ) :
  P_total = 311 → P_Jessie = 41 → 
  (∀ P_Lisa P_Devin: ℕ, P_Lisa = P_Jessie ∧ P_Devin = P_Jessie) → 
  P_others = 188 :=
by
  sorry

end points_scored_by_others_l74_74808


namespace cube_root_of_27_l74_74075

theorem cube_root_of_27 : ∃ x : ℝ, x ^ 3 = 27 ↔ ∃ y : ℝ, y = 3 := by
  sorry

end cube_root_of_27_l74_74075


namespace find_y_l74_74459

noncomputable def x : ℝ := (4 / 25)^(1 / 3)

theorem find_y (y : ℝ) (h1 : 0 < y) (h2 : y < x) (h3 : x^x = y^y) : y = (32 / 3125)^(1 / 3) :=
sorry

end find_y_l74_74459


namespace pentagon_square_ratio_l74_74542

theorem pentagon_square_ratio (p s : ℝ) (h₁ : 5 * p = 20) (h₂ : 4 * s = 20) : p / s = 4 / 5 := 
by 
  sorry

end pentagon_square_ratio_l74_74542


namespace rhombus_area_l74_74447

theorem rhombus_area (a b : ℝ) (h : (a - 1) ^ 2 + Real.sqrt (b - 4) = 0) : (1 / 2) * a * b = 2 := by
  sorry

end rhombus_area_l74_74447


namespace problem1_problem2_problem3_l74_74560

-- Problem 1
theorem problem1 (x y : ℝ) : 4 * x^2 - y^4 = (2 * x + y^2) * (2 * x - y^2) :=
by
  -- proof omitted
  sorry

-- Problem 2
theorem problem2 (x y : ℝ) : 8 * x^2 - 24 * x * y + 18 * y^2 = 2 * (2 * x - 3 * y)^2 :=
by
  -- proof omitted
  sorry

-- Problem 3
theorem problem3 (x y : ℝ) : (x - y) * (3 * x + 1) - 2 * (x^2 - y^2) - (y - x)^2 = (x - y) * (1 - y) :=
by
  -- proof omitted
  sorry

end problem1_problem2_problem3_l74_74560


namespace cybil_ronda_probability_l74_74712

theorem cybil_ronda_probability :
  let total_cards := 15
  let cybil_cards := 6
  let ronda_cards := 9
  let draw_cards := 3
  let total_combinations := Nat.choose total_cards draw_cards
  let no_cybil_combinations := Nat.choose ronda_cards draw_cards
  let no_ronda_combinations := Nat.choose cybil_cards draw_cards
  let p_no_cybil := no_cybil_combinations / total_combinations
  let p_no_ronda := no_ronda_combinations / total_combinations
  let p_at_least_one_each := 1 - p_no_cybil - p_no_ronda
  p_at_least_one_each = (351 / 455) :=
by
  unfold total_cards cybil_cards ronda_cards draw_cards total_combinations no_cybil_combinations no_ronda_combinations p_no_cybil p_no_ronda p_at_least_one_each
  norm_num
  sorry

end cybil_ronda_probability_l74_74712


namespace spend_amount_7_l74_74691

variable (x y z w : ℕ) (k : ℕ)

theorem spend_amount_7 
  (h1 : 10 * x + 15 * y + 25 * z + 40 * w = 100 * k)
  (h2 : x + y + z + w = 30)
  (h3 : (x = 5 ∨ x = 10) ∧ (y = 5 ∨ y = 10) ∧ (z = 5 ∨ z = 10) ∧ (w = 5 ∨ w = 10)) : 
  k = 7 := 
sorry

end spend_amount_7_l74_74691


namespace sum_of_four_consecutive_integers_divisible_by_two_l74_74089

theorem sum_of_four_consecutive_integers_divisible_by_two (n : ℤ) : 
  2 ∣ ((n-1) + n + (n+1) + (n+2)) :=
by
  sorry

end sum_of_four_consecutive_integers_divisible_by_two_l74_74089


namespace cylinder_volume_ratio_l74_74128

-- First define the problem: a 6 x 10 rectangle rolled to form two different cylinders
theorem cylinder_volume_ratio : 
  let r1 := 3 / Real.pi in
  let V1 := Real.pi * r1^2 * 10 in
  let r2 := 5 / Real.pi in
  let V2 := Real.pi * r2^2 * 6 in
  V2 / V1 = 5 / 3 :=
by
  -- The proof steps are omitted as the theorem states only.
  sorry

end cylinder_volume_ratio_l74_74128


namespace highway_length_proof_l74_74293

variable (L : ℝ) (v1 v2 : ℝ) (t : ℝ)

def highway_length : Prop :=
  v1 = 55 ∧ v2 = 35 ∧ t = 1 / 15 ∧ (L / v2 - L / v1 = t) ∧ L = 6.42

theorem highway_length_proof : highway_length L 55 35 (1 / 15) := by
  sorry

end highway_length_proof_l74_74293


namespace negative_values_count_l74_74567

theorem negative_values_count (n : ℕ) : (n < 13) → (n^2 < 150) → ∃ (k : ℕ), k = 12 :=
by
  sorry

end negative_values_count_l74_74567


namespace find_x_in_triangle_l74_74452

theorem find_x_in_triangle (y z : ℝ) (cos_Y_minus_Z : ℝ) (h1 : y = 7) (h2 : z = 6) (h3 : cos_Y_minus_Z = 1 / 2) : 
    ∃ x : ℝ, x = Real.sqrt 73 :=
by
  existsi Real.sqrt 73
  sorry

end find_x_in_triangle_l74_74452


namespace total_tickets_sold_correct_total_tickets_sold_is_21900_l74_74294

noncomputable def total_tickets_sold : ℕ := 5400 + 16500

theorem total_tickets_sold_correct :
    total_tickets_sold = 5400 + 5 * (16500 / 5) :=
by
  rw [Nat.div_mul_cancel]
  sorry

-- The following theorem states the main proof equivalence:
theorem total_tickets_sold_is_21900 :
    total_tickets_sold = 21900 :=
by
  sorry

end total_tickets_sold_correct_total_tickets_sold_is_21900_l74_74294


namespace maximize_f_at_1_5_l74_74670

noncomputable def f (x: ℝ) : ℝ := -3 * x^2 + 9 * x + 5

theorem maximize_f_at_1_5 : ∀ x: ℝ, f 1.5 ≥ f x := by
  sorry

end maximize_f_at_1_5_l74_74670


namespace tourist_group_people_count_l74_74699

def large_room_people := 3
def small_room_people := 2
def small_rooms_rented := 1
def people_in_small_room := small_rooms_rented * small_room_people

theorem tourist_group_people_count : 
  ∀ x : ℕ, x ≥ 1 ∧ (x + small_rooms_rented) = (people_in_small_room + x * large_room_people) → 
  (people_in_small_room + x * large_room_people) = 5 := 
  by
  sorry

end tourist_group_people_count_l74_74699


namespace marco_new_cards_l74_74793

theorem marco_new_cards (total_cards : ℕ) (fraction_duplicates : ℕ) (fraction_traded : ℕ) : 
  total_cards = 500 → 
  fraction_duplicates = 4 → 
  fraction_traded = 5 → 
  (total_cards / fraction_duplicates) / fraction_traded = 25 :=
by
  intro h1 h2 h3
  rw [h1, h2, h3]
  rw [Nat.div_div_eq_div_mul]
  norm_num

-- Since Lean requires explicit values, we use the following lemma to lead the theorem to concrete values:
lemma marco_new_cards_concrete : 
  (500 / 4) / 5 = 25 :=
marco_new_cards 500 4 5 rfl rfl rfl

end marco_new_cards_l74_74793


namespace ThreePowerTowerIsLarger_l74_74673

-- original power tower definitions
def A : ℕ := 3^(3^(3^3))
def B : ℕ := 2^(2^(2^(2^2)))

-- reduced forms given from the conditions
def reducedA : ℕ := 3^(3^27)
def reducedB : ℕ := 2^(2^16)

theorem ThreePowerTowerIsLarger : reducedA > reducedB := by
  sorry

end ThreePowerTowerIsLarger_l74_74673


namespace evaluate_fraction_l74_74157

theorem evaluate_fraction : (1 / (2 + (1 / (3 + (1 / 4))))) = 13 / 30 :=
by
  sorry

end evaluate_fraction_l74_74157


namespace matrix_power_50_l74_74200

open Matrix

-- Define the matrix A
def A : Matrix (Fin 2) (Fin 2) ℤ := ![(5 : ℤ), 2; -16, -6]

-- The target matrix we want to prove A^50 equals to
def target : Matrix (Fin 2) (Fin 2) ℤ := ![(-301 : ℤ), -100; 800, 299]

-- Prove that A^50 equals to the target matrix
theorem matrix_power_50 : A^50 = target := 
by {
  sorry
}

end matrix_power_50_l74_74200


namespace perpendicular_bisector_eq_l74_74285

-- Define the circles
def C1 (x y : ℝ) : Prop := x^2 + y^2 - 4 * x + 6 * y = 0
def C2 (x y : ℝ) : Prop := x^2 + y^2 - 6 * x = 0

-- Prove that the perpendicular bisector of line segment AB has the equation 3x - y - 9 = 0
theorem perpendicular_bisector_eq :
  (∀ x y : ℝ, C1 x y → C2 x y → 3 * x - y - 9 = 0) :=
by
  sorry

end perpendicular_bisector_eq_l74_74285


namespace mass_of_fourth_metal_l74_74849

theorem mass_of_fourth_metal 
  (m1 m2 m3 m4 : ℝ)
  (total_mass : m1 + m2 + m3 + m4 = 20)
  (h1 : m1 = 1.5 * m2)
  (h2 : m2 = 3/4 * m3)
  (h3 : m3 = 5/6 * m4) :
  m4 = 20 * (48 / 163) :=
sorry

end mass_of_fourth_metal_l74_74849


namespace tessa_initial_apples_l74_74220

-- Define conditions as variables
variable (initial_apples anita_gave : ℕ)
variable (apples_needed_for_pie : ℕ := 10)
variable (apples_additional_now_needed : ℕ := 1)

-- Define the current amount of apples Tessa has
noncomputable def current_apples :=
  apples_needed_for_pie - apples_additional_now_needed

-- Define the initial apples Tessa had before Anita gave her 5 apples
noncomputable def initial_apples_calculated :=
  current_apples - anita_gave

-- Lean statement to prove the initial number of apples Tessa had
theorem tessa_initial_apples (h_initial_apples : anita_gave = 5) : initial_apples_calculated = 4 :=
by
  -- Here is where the proof would go; we use sorry to indicate it's not provided
  sorry

end tessa_initial_apples_l74_74220


namespace min_value_2x_minus_y_l74_74322

open Real

theorem min_value_2x_minus_y : ∀ (x y : ℝ), |x| ≤ y ∧ y ≤ 2 → ∃ (c : ℝ), c = 2 * x - y ∧ ∀ z, z = 2 * x - y → z ≥ -6 := sorry

end min_value_2x_minus_y_l74_74322


namespace Nina_money_l74_74796

theorem Nina_money : ∃ (M : ℝ) (W : ℝ), M = 10 * W ∧ M = 14 * (W - 3) ∧ M = 105 :=
by
  sorry

end Nina_money_l74_74796


namespace exists_set_B_l74_74338

open Finset

theorem exists_set_B (A : Finset ℕ) (hA : ∀ x ∈ A, 0 < x) : 
  ∃ B : Finset ℕ, A ⊆ B ∧ (∏ x in B, x = ∑ x in B, x^2) :=
sorry

end exists_set_B_l74_74338


namespace cylinder_volume_ratio_l74_74129

-- First define the problem: a 6 x 10 rectangle rolled to form two different cylinders
theorem cylinder_volume_ratio : 
  let r1 := 3 / Real.pi in
  let V1 := Real.pi * r1^2 * 10 in
  let r2 := 5 / Real.pi in
  let V2 := Real.pi * r2^2 * 6 in
  V2 / V1 = 5 / 3 :=
by
  -- The proof steps are omitted as the theorem states only.
  sorry

end cylinder_volume_ratio_l74_74129


namespace irrational_pi_l74_74376

def is_irrational (x : ℝ) : Prop := ¬ ∃ a b : ℤ, b ≠ 0 ∧ x = a / b

theorem irrational_pi : is_irrational π := by
  sorry

end irrational_pi_l74_74376


namespace moles_of_NaOH_combined_l74_74300

-- Define the reaction conditions
variable (moles_NH4NO3 : ℕ) (moles_NaNO3 : ℕ)

-- Define a proof problem that asserts the number of moles of NaOH combined
theorem moles_of_NaOH_combined
  (h1 : moles_NH4NO3 = 3)  -- 3 moles of NH4NO3 are combined
  (h2 : moles_NaNO3 = 3)  -- 3 moles of NaNO3 are formed
  : ∃ moles_NaOH : ℕ, moles_NaOH = 3 :=
by {
  -- Proof skeleton to be filled
  sorry
}

end moles_of_NaOH_combined_l74_74300


namespace range_of_t_l74_74345

theorem range_of_t (x y : ℝ) (h1 : 1 ≤ x) (h2 : x ≤ y) (h3 : x + y > 1) (h4 : x + 1 > y) (h5 : y + 1 > x) :
    1 ≤ max (1 / x) (max (x / y) y) * min (1 / x) (min (x / y) y) ∧
    max (1 / x) (max (x / y) y) * min (1 / x) (min (x / y) y) < (1 + Real.sqrt 5) / 2 := 
sorry

end range_of_t_l74_74345


namespace square_root_of_25_is_5_and_minus_5_l74_74832

theorem square_root_of_25_is_5_and_minus_5 : ∃ y : ℝ, y^2 = 25 ∧ (y = 5 ∨ y = -5) :=
by
  have h1 : 5^2 = 25 := by norm_num
  have h2 : (-5)^2 = 25 := by norm_num
  use 5
  use -5
  split
  · exact h1
  · exact h2

end square_root_of_25_is_5_and_minus_5_l74_74832


namespace perpendicular_vectors_x_value_l74_74177

theorem perpendicular_vectors_x_value :
  let a := (4, 2)
  let b := (x, 3)
  a.1 * b.1 + a.2 * b.2 = 0 -> x = -3/2 :=
by
  intros
  sorry

end perpendicular_vectors_x_value_l74_74177


namespace paint_one_third_of_square_l74_74083

theorem paint_one_third_of_square (n k : ℕ) (h1 : n = 18) (h2 : k = 6) :
    Nat.choose n k = 18564 :=
by
  rw [h1, h2]
  sorry

end paint_one_third_of_square_l74_74083


namespace solve_for_a_and_b_l74_74444

theorem solve_for_a_and_b (a b : ℤ) (h1 : 5 + a = 6 - b) (h2 : 6 + b = 9 + a) : 5 - a = 6 := 
sorry

end solve_for_a_and_b_l74_74444


namespace correct_calculation_of_mistake_l74_74844

theorem correct_calculation_of_mistake (x : ℝ) (h : x - 48 = 52) : x + 48 = 148 :=
by
  sorry

end correct_calculation_of_mistake_l74_74844


namespace base_angle_isosceles_triangle_l74_74592

theorem base_angle_isosceles_triangle (T : Triangle) (h_iso : is_isosceles T) (h_angle : internal_angle T = 110) :
  base_angle T = 35 :=
sorry

end base_angle_isosceles_triangle_l74_74592


namespace deepak_present_age_l74_74123

theorem deepak_present_age 
  (x : ℕ) 
  (rahul_age_now : ℕ := 5 * x)
  (deepak_age_now : ℕ := 2 * x)
  (rahul_age_future : ℕ := rahul_age_now + 6) 
  (future_condition : rahul_age_future = 26) 
  : deepak_age_now = 8 :=
by
  sorry

end deepak_present_age_l74_74123


namespace units_digit_of_150_factorial_is_zero_l74_74247

-- Define the conditions for the problem
def is_units_digit_zero_of_factorial (n : ℕ) : Prop :=
  n = 150 → (Nat.factorial n % 10 = 0)

-- The statement of the proof problem
theorem units_digit_of_150_factorial_is_zero : is_units_digit_zero_of_factorial 150 :=
  sorry

end units_digit_of_150_factorial_is_zero_l74_74247


namespace evaluate_expression_l74_74559

theorem evaluate_expression (x : ℝ) (h1 : x ≠ -2) (h2 : x ≠ 3) :
  (3 * x ^ 2 - 2 * x + 1) / ((x + 2) * (x - 3)) - (x ^ 2 - 5 * x + 6) / ((x + 2) * (x - 3)) =
  (2 * x ^ 2 + 3 * x - 5) / ((x + 2) * (x - 3)) :=
by
  sorry

end evaluate_expression_l74_74559


namespace solve_equation_l74_74629

theorem solve_equation :
  ∀ x : ℝ, (101 * x ^ 2 - 18 * x + 1) ^ 2 - 121 * x ^ 2 * (101 * x ^ 2 - 18 * x + 1) + 2020 * x ^ 4 = 0 ↔ 
    x = 1 / 18 ∨ x = 1 / 9 :=
by
  intro x
  sorry

end solve_equation_l74_74629


namespace calculate_wheel_radii_l74_74777

theorem calculate_wheel_radii (rpmA rpmB : ℕ) (length : ℝ) (r R : ℝ) :
  rpmA = 1200 →
  rpmB = 1500 →
  length = 9 →
  (4 : ℝ) / 5 * r = R →
  2 * (R + r) = 9 →
  r = 2 ∧ R = 2.5 :=
by
  intros
  sorry

end calculate_wheel_radii_l74_74777


namespace triangle_isosceles_or_right_l74_74165

theorem triangle_isosceles_or_right (a b c : ℝ) (A B C : ℝ) 
  (h1 : a ≠ 0) (h2 : b ≠ 0) (triangle_abc : A + B + C = 180)
  (opposite_sides : ∀ {x y}, x ≠ y → x + y < 180) 
  (condition : a * Real.cos A = b * Real.cos B) :
  (A = B ∨ A + B = 90) :=
by {
  sorry
}

end triangle_isosceles_or_right_l74_74165


namespace problem_statement_l74_74150

theorem problem_statement :
  75 * ((4 + 1/3) - (5 + 1/4)) / ((3 + 1/2) + (2 + 1/5)) = -5/31 := 
by
  sorry

end problem_statement_l74_74150


namespace solve_eq1_solve_eq2_l74_74068

theorem solve_eq1 (x : ℝ) : (x^2 - 2 * x - 8 = 0) ↔ (x = 4 ∨ x = -2) :=
sorry

theorem solve_eq2 (x : ℝ) : (2 * x^2 - 4 * x + 1 = 0) ↔ (x = (2 + Real.sqrt 2) / 2 ∨ x = (2 - Real.sqrt 2) / 2) :=
sorry

end solve_eq1_solve_eq2_l74_74068


namespace correct_triangle_l74_74984

-- Define the conditions for the sides of each option
def sides_A := (1, 2, 3)
def sides_B := (3, 4, 5)
def sides_C := (3, 1, 1)
def sides_D := (3, 4, 7)

-- Conditions for forming a triangle
def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

-- Prove the problem statement
theorem correct_triangle : is_triangle 3 4 5 :=
by
  sorry

end correct_triangle_l74_74984


namespace work_completed_in_days_l74_74122

theorem work_completed_in_days (time_a time_b time_c : ℕ)
  (ha : time_a = 15) (hb : time_b = 20) (hc : time_c = 45) :
  let combined_work_rate := (1 / rat.of_int time_a) + 
                            (1 / rat.of_int time_b) + 
                            (1 / rat.of_int time_c) in
  let days_to_complete_work := 1 / combined_work_rate in
  days_to_complete_work = 7.2 := 
by
  sorry

end work_completed_in_days_l74_74122


namespace john_trip_time_l74_74334

theorem john_trip_time (normal_distance : ℕ) (normal_time : ℕ) (extra_distance : ℕ) 
  (double_extra_distance : ℕ) (same_speed : ℕ) 
  (h1: normal_distance = 150) 
  (h2: normal_time = 3) 
  (h3: extra_distance = 50)
  (h4: double_extra_distance = 2 * extra_distance)
  (h5: same_speed = normal_distance / normal_time) : 
  normal_time + double_extra_distance / same_speed = 5 :=
by 
  sorry

end john_trip_time_l74_74334


namespace solve_for_a_and_b_l74_74443

theorem solve_for_a_and_b (a b : ℤ) (h1 : 5 + a = 6 - b) (h2 : 6 + b = 9 + a) : 5 - a = 6 := 
sorry

end solve_for_a_and_b_l74_74443


namespace scientific_notation_example_l74_74556

theorem scientific_notation_example :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ |a| ∧ |a| < 10 ∧ 218000000 = a * 10 ^ n ∧ a = 2.18 ∧ n = 8 :=
by {
  -- statement of the problem conditions
  sorry
}

end scientific_notation_example_l74_74556


namespace part1_part2_axis_of_symmetry_part2_center_of_symmetry_l74_74179

noncomputable def m (x : ℝ) : ℝ × ℝ := (2 * (Real.cos x) ^ 2, Real.sin x)
noncomputable def n (x : ℝ) : ℝ × ℝ := (1, 2 * Real.cos x)

def perpendicular (u v : ℝ × ℝ) : Prop := u.1 * v.1 + u.2 * v.2 = 0

noncomputable def f (x : ℝ) : ℝ := (m x).1 * (n x).1 + (m x).2 * (n x).2

theorem part1 (x : ℝ) (h1 : 0 < x ∧ x < π) (h2 : perpendicular (m x) (n x)) :
  x = π / 2 ∨ x = 3 * π / 4 :=
sorry

theorem part2_axis_of_symmetry (k : ℤ) : 
  ∃ c : ℝ, ∀ x : ℝ, f x = f (2 * c - x) ∧ 
    ((2 * x + π / 4) = k * π + π / 2 → x = k * π / 2 + π / 8) :=
sorry

theorem part2_center_of_symmetry (k : ℤ) :
  ∃ x c : ℝ, f x = 1 ∧ ((2 * x + π / 4) = k * π → x = k * π / 2 - π / 8) :=
sorry

end part1_part2_axis_of_symmetry_part2_center_of_symmetry_l74_74179


namespace smallest_common_multiple_l74_74506

theorem smallest_common_multiple (n : ℕ) (h1 : n > 0) (h2 : 8 ∣ n) (h3 : 6 ∣ n) : n = 24 :=
by sorry

end smallest_common_multiple_l74_74506


namespace option_b_not_valid_l74_74311

theorem option_b_not_valid (a b c d : ℝ) (h_arith_seq : b - a = d ∧ c - b = d ∧ d ≠ 0) : 
  a^3 * b + b^3 * c + c^3 * a < a^4 + b^4 + c^4 :=
by sorry

end option_b_not_valid_l74_74311


namespace Emily_used_10_dimes_l74_74296

theorem Emily_used_10_dimes
  (p n d : ℕ)
  (h1 : p + n + d = 50)
  (h2 : p + 5 * n + 10 * d = 200) :
  d = 10 := by
  sorry

end Emily_used_10_dimes_l74_74296


namespace problem_l74_74173

noncomputable def fx (a b c : ℝ) (x : ℝ) : ℝ := a * x + b / x + c

theorem problem 
  (a b c : ℝ) 
  (h_odd : ∀ x, fx a b c x = -fx a b c (-x))
  (h_f1 : fx a b c 1 = 5 / 2)
  (h_f2 : fx a b c 2 = 17 / 4) :
  (a = 2) ∧ (b = 1 / 2) ∧ (c = 0) ∧ (∀ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 1 / 2 → fx a b c x₁ > fx a b c x₂) := 
sorry

end problem_l74_74173


namespace minimum_sum_of_dimensions_l74_74535

theorem minimum_sum_of_dimensions {a b c : ℕ} (h1 : a * b * c = 2310) (h2 : a > 0) (h3 : b > 0) (h4 : c > 0) :
  a + b + c ≥ 42 := 
sorry

end minimum_sum_of_dimensions_l74_74535


namespace necessary_and_sufficient_condition_for_parallel_lines_l74_74843

theorem necessary_and_sufficient_condition_for_parallel_lines (a l : ℝ) :
  (a = -1) ↔ (∀ x y : ℝ, ax + 3 * y + 3 = 0 → x + (a - 2) * y + l = 0) := 
sorry

end necessary_and_sufficient_condition_for_parallel_lines_l74_74843


namespace tom_has_9_balloons_l74_74972

-- Define Tom's and Sara's yellow balloon counts
variables (total_balloons saras_balloons toms_balloons : ℕ)

-- Given conditions
axiom total_balloons_def : total_balloons = 17
axiom saras_balloons_def : saras_balloons = 8
axiom toms_balloons_total : toms_balloons + saras_balloons = total_balloons

-- Theorem stating that Tom has 9 yellow balloons
theorem tom_has_9_balloons : toms_balloons = 9 := by
  sorry

end tom_has_9_balloons_l74_74972


namespace original_deck_card_count_l74_74846

variable (r b : ℕ)

theorem original_deck_card_count (h1 : r / (r + b) = 1 / 4) (h2 : r / (r + b + 6) = 1 / 6) : r + b = 12 :=
by
  -- The proof goes here
  sorry

end original_deck_card_count_l74_74846


namespace trig_identity_l74_74301

theorem trig_identity :
  (Real.sin (20 * Real.pi / 180) * Real.sin (50 * Real.pi / 180) + 
   Real.cos (20 * Real.pi / 180) * Real.sin (40 * Real.pi / 180)) = 
  (Real.sqrt 3 / 2) :=
by
  sorry

end trig_identity_l74_74301


namespace total_tiles_l74_74998

theorem total_tiles (s : ℕ) (h_black_tiles : 2 * s - 1 = 75) : s^2 = 1444 :=
by {
  sorry
}

end total_tiles_l74_74998


namespace abs_x_plus_abs_y_eq_one_area_l74_74071

theorem abs_x_plus_abs_y_eq_one_area : 
  (∃ (A : ℝ), ∀ (x y : ℝ), |x| + |y| = 1 → A = 2) :=
sorry

end abs_x_plus_abs_y_eq_one_area_l74_74071


namespace age_relation_l74_74733

variable (x y z : ℕ)

theorem age_relation (h1 : x > y) : (z > y) ↔ (∃ w, w > 0 ∧ y + z > 2 * x) :=
sorry

end age_relation_l74_74733


namespace probability_of_forming_CHORAL_is_correct_l74_74605

-- Definitions for selecting letters with given probabilities
def probability_select_C_A_L_from_CAMEL : ℚ :=
  1 / 10

def probability_select_H_O_R_from_SHRUB : ℚ :=
  1 / 10

def probability_select_G_from_GLOW : ℚ :=
  1 / 2

-- Calculating the total probability of selecting letters to form "CHORAL"
def probability_form_CHORAL : ℚ :=
  probability_select_C_A_L_from_CAMEL * 
  probability_select_H_O_R_from_SHRUB * 
  probability_select_G_from_GLOW

theorem probability_of_forming_CHORAL_is_correct :
  probability_form_CHORAL = 1 / 200 :=
by
  -- Statement to be proven here
  sorry

end probability_of_forming_CHORAL_is_correct_l74_74605


namespace min_abs_sum_l74_74361

theorem min_abs_sum (x : ℝ) : ∃ y : ℝ, y = min ((|x+1| + |x-2| + |x-3|)) 4 :=
sorry

end min_abs_sum_l74_74361


namespace points_on_equation_correct_l74_74745

theorem points_on_equation_correct (x y : ℝ) :
  (2 * x^2 + y^2 + 3 * x * y + 3 * x + y = 2) ↔ (y = -x - 2 ∨ y = -2 * x + 1) :=
by 
  sorry

end points_on_equation_correct_l74_74745


namespace not_divisible_by_121_l74_74230

theorem not_divisible_by_121 (n : ℤ) : ¬ ∃ t : ℤ, (n^2 + 3*n + 5) = 121 * t ∧ (n^2 - 3*n + 5) = 121 * t := sorry

end not_divisible_by_121_l74_74230


namespace ratio_Pat_Mark_l74_74474

-- Total hours charged by all three
def total_hours (P K M : ℕ) : Prop :=
  P + K + M = 144

-- Pat charged twice as much time as Kate
def pat_hours (P K : ℕ) : Prop :=
  P = 2 * K

-- Mark charged 80 hours more than Kate
def mark_hours (M K : ℕ) : Prop :=
  M = K + 80

-- The ratio of Pat's hours to Mark's hours
def ratio (P M : ℕ) : ℚ :=
  (P : ℚ) / (M : ℚ)

theorem ratio_Pat_Mark (P K M : ℕ)
  (h1 : total_hours P K M)
  (h2 : pat_hours P K)
  (h3 : mark_hours M K) :
  ratio P M = (1 : ℚ) / (3 : ℚ) :=
by
  sorry

end ratio_Pat_Mark_l74_74474


namespace impossible_event_abs_lt_zero_l74_74374

theorem impossible_event_abs_lt_zero (a : ℝ) : ¬ (|a| < 0) :=
sorry

end impossible_event_abs_lt_zero_l74_74374


namespace sum_even_squares_sum_odd_squares_l74_74014

open scoped BigOperators

def sumOfSquaresEven (n : ℕ) : ℕ :=
  ∑ i in Finset.range n, (2 * (i + 1))^2

def sumOfSquaresOdd (n : ℕ) : ℕ :=
  ∑ i in Finset.range n, (2 * i + 1)^2

theorem sum_even_squares (n : ℕ) :
  sumOfSquaresEven n = (2 * n * (n - 1) * (2 * n - 1)) / 3 := by
    sorry

theorem sum_odd_squares (n : ℕ) :
  sumOfSquaresOdd n = (n * (4 * n^2 - 1)) / 3 := by
    sorry

end sum_even_squares_sum_odd_squares_l74_74014


namespace quadratic_transformation_l74_74490

theorem quadratic_transformation (p q r : ℝ) :
  (∀ x : ℝ, p * x^2 + q * x + r = 3 * (x - 5)^2 + 15) →
  (∀ x : ℝ, 4 * p * x^2 + 4 * q * x + 4 * r = 12 * (x - 5)^2 + 60) :=
by
  intro h
  exact sorry

end quadratic_transformation_l74_74490


namespace correct_statement_C_l74_74827

theorem correct_statement_C : ∀ y : ℝ, y^2 = 25 ↔ (y = 5 ∨ y = -5) := 
by sorry

end correct_statement_C_l74_74827


namespace cost_of_pencil_l74_74305

theorem cost_of_pencil (x y : ℕ) (h1 : 4 * x + 3 * y = 224) (h2 : 2 * x + 5 * y = 154) : y = 12 := 
by
  sorry

end cost_of_pencil_l74_74305


namespace lindy_distance_traveled_l74_74680

/-- Jack and Christina are standing 240 feet apart on a level surface. 
Jack walks in a straight line toward Christina at a constant speed of 5 feet per second. 
Christina walks in a straight line toward Jack at a constant speed of 3 feet per second. 
Lindy runs at a constant speed of 9 feet per second from Christina to Jack, back to Christina, back to Jack, and so forth. 
The total distance Lindy travels when the three meet at one place is 270 feet. -/
theorem lindy_distance_traveled
    (initial_distance : ℝ)
    (jack_speed : ℝ)
    (christina_speed : ℝ)
    (lindy_speed : ℝ)
    (time_to_meet : ℝ)
    (total_distance_lindy : ℝ) :
    initial_distance = 240 ∧
    jack_speed = 5 ∧
    christina_speed = 3 ∧
    lindy_speed = 9 ∧
    time_to_meet = (initial_distance / (jack_speed + christina_speed)) ∧
    total_distance_lindy = lindy_speed * time_to_meet →
    total_distance_lindy = 270 :=
by
  sorry

end lindy_distance_traveled_l74_74680


namespace probability_sum_odd_l74_74557

section

variable {α : Type} {β : Type}

-- Define the probability measure
noncomputable def prob {Ω : Type} (ω: Ω → ℕ) (p : Ω): ℚ :=
(p.card (set_of ω) / (finset.card p)).to_rat

-- Define the sets of events
def wheel_A : finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def even_num (n : ℕ) : Prop := n % 2 = 0
def odd_num (n : ℕ) : Prop := ¬ even_num n

def wheel_B : finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def even_numbers_B : finset ℕ := {2, 4, 6, 8, 10, 12}
def odd_numbers_B : finset ℕ := {1, 3, 5, 7}

-- Proving the probability calculation
theorem probability_sum_odd :
  prob (even_num) wheel_A = 1 / 2 →
  prob (odd_num) wheel_A = 1 / 2 →
  prob (even_num) wheel_B = 3 / 4 →
  prob (odd_num) wheel_B = 1 / 4 →
  prob (λ n, even_num n ∨ odd_num n) (wheel_A ×ˢ wheel_B) = 1 / 2 :=
by
  sorry

end

end probability_sum_odd_l74_74557


namespace find_a9_l74_74609

variable (S : ℕ → ℤ) (a : ℕ → ℤ)
variable (d a1 : ℤ)

def arithmetic_seq (n : ℕ) : ℤ :=
  a1 + ↑n * d

def sum_arithmetic_seq (n : ℕ) : ℤ :=
  n * a1 + (n * (n - 1) / 2) * d

axiom h1 : sum_arithmetic_seq 8 = 4 * arithmetic_seq 3
axiom h2 : arithmetic_seq 7 = -2

theorem find_a9 : arithmetic_seq 9 = -6 :=
by
  sorry

end find_a9_l74_74609


namespace rectangle_diagonal_length_proof_parallel_l74_74684

-- Definition of a rectangle whose sides are parallel to the coordinate axes
structure RectangleParallel :=
  (a b : ℕ)
  (area_eq : a * b = 2018)
  (diagonal_length : ℕ)

-- Prove that the length of the diagonal of the given rectangle is sqrt(1018085)
def rectangle_diagonal_length_parallel : RectangleParallel → Prop :=
  fun r => r.diagonal_length = Int.sqrt (r.a * r.a + r.b * r.b)

theorem rectangle_diagonal_length_proof_parallel (r : RectangleParallel)
  (h1 : r.a * r.b = 2018)
  (h2 : r.a ≠ r.b)
  (h3 : r.diagonal_length = Int.sqrt (r.a * r.a + r.b * r.b)) :
  r.diagonal_length = Int.sqrt 1018085 := 
  sorry

end rectangle_diagonal_length_proof_parallel_l74_74684


namespace find_original_speed_l74_74546

theorem find_original_speed :
  ∀ (v T : ℝ), 
    (300 = 212 + 88) →
    (T + 2/3 = 212 / v + 88 / (v - 50)) →
    v = 110 :=
by
  intro v T h_dist h_trip
  sorry

end find_original_speed_l74_74546


namespace square_perimeter_l74_74139

-- First, declare the side length of the square (rectangle)
variable (s : ℝ)

-- State the conditions: the area is 484 cm^2 and it's a square
axiom area_condition : s^2 = 484
axiom is_square : ∀ (s : ℝ), s > 0

-- Define the perimeter of the square
def perimeter (s : ℝ) : ℝ := 4 * s

-- State the theorem: perimeter == 88 given the conditions
theorem square_perimeter : perimeter s = 88 :=
by 
  -- Prove the statement given the axiom 'area_condition'
  sorry

end square_perimeter_l74_74139


namespace calories_in_300g_lemonade_l74_74617

def lemonade_calories (lemon_juice_in_g : Nat) (sugar_in_g : Nat) (water_in_g : Nat) (lemon_juice_cal : Nat) (sugar_cal : Nat) : Nat :=
  (lemon_juice_in_g * lemon_juice_cal / 100) + (sugar_in_g * sugar_cal / 100)

def total_weight (lemon_juice_in_g : Nat) (sugar_in_g : Nat) (water_in_g : Nat) : Nat :=
  lemon_juice_in_g + sugar_in_g + water_in_g

theorem calories_in_300g_lemonade :
  (lemonade_calories 500 200 1000 30 400) * 300 / (total_weight 500 200 1000) = 168 := 
  by
    sorry

end calories_in_300g_lemonade_l74_74617


namespace sum_of_consecutive_odds_eq_power_l74_74363

theorem sum_of_consecutive_odds_eq_power (n : ℕ) (k : ℕ) (hn : n > 0) (hk : k ≥ 2) :
  ∃ a : ℤ, n * (2 * a + n) = n^k ∧
            (∀ i : ℕ, i < n → 2 * a + 2 * (i : ℤ) + 1 = 2 * a + 1 + 2 * i) :=
by
  sorry

end sum_of_consecutive_odds_eq_power_l74_74363


namespace nancy_history_books_l74_74795

/-- Nancy started with 46 books in total on the cart.
    She shelved 8 romance books and 4 poetry books from the top section.
    She shelved 5 Western novels and 6 biographies from the bottom section.
    Half the books on the bottom section were mystery books.
    Prove that Nancy shelved 12 history books.
-/
theorem nancy_history_books 
  (total_books : ℕ)
  (romance_books : ℕ)
  (poetry_books : ℕ)
  (western_novels : ℕ)
  (biographies : ℕ)
  (bottom_books_half_mystery : ℕ)
  (history_books : ℕ) :
  (total_books = 46) →
  (romance_books = 8) →
  (poetry_books = 4) →
  (western_novels = 5) →
  (biographies = 6) →
  (bottom_books_half_mystery = 11) →
  (history_books = total_books - ((romance_books + poetry_books) + (2 * (western_novels + biographies)))) →
  history_books = 12 :=
by
  intros
  sorry

end nancy_history_books_l74_74795


namespace cost_price_of_toy_l74_74531

theorem cost_price_of_toy 
  (cost_price : ℝ)
  (SP : ℝ := 120000)
  (num_toys : ℕ := 40)
  (profit_per_toy : ℝ := 500)
  (gain_per_toy : ℝ := cost_price + profit_per_toy)
  (total_gain : ℝ := 8 * cost_price + profit_per_toy * num_toys)
  (total_cost_price : ℝ := num_toys * cost_price)
  (SP_eq_cost_plus_gain : SP = total_cost_price + total_gain) :
  cost_price = 2083.33 :=
by
  sorry

end cost_price_of_toy_l74_74531


namespace solution_mix_percentage_l74_74217

theorem solution_mix_percentage
  (x y z : ℝ)
  (hx1 : x + y + z = 100)
  (hx2 : 0.40 * x + 0.50 * y + 0.30 * z = 46)
  (hx3 : z = 100 - x - y) :
  x = 40 ∧ y = 60 ∧ z = 0 :=
by
  sorry

end solution_mix_percentage_l74_74217


namespace fraction_six_power_l74_74342

theorem fraction_six_power (n : ℕ) (hyp : n = 6 ^ 2024) : n / 6 = 6 ^ 2023 :=
by sorry

end fraction_six_power_l74_74342


namespace ratio_of_magnets_given_away_l74_74858

-- Define the conditions of the problem
def initial_magnets_adam : ℕ := 18
def magnets_peter : ℕ := 24
def final_magnets_adam (x : ℕ) : ℕ := initial_magnets_adam - x
def half_magnets_peter : ℕ := magnets_peter / 2

-- The main statement to prove
theorem ratio_of_magnets_given_away (x : ℕ) (h : final_magnets_adam x = half_magnets_peter) :
    (x : ℚ) / initial_magnets_adam = 1 / 3 :=
by
  sorry -- This is where the proof would go

end ratio_of_magnets_given_away_l74_74858


namespace initial_sum_of_money_l74_74272

theorem initial_sum_of_money (A2 A7 : ℝ) (H1 : A2 = 520) (H2 : A7 = 820) :
  ∃ P : ℝ, P = 400 :=
by
  -- Proof starts here
  sorry

end initial_sum_of_money_l74_74272


namespace gcd_98_63_l74_74486

-- The statement of the problem in Lean 4
theorem gcd_98_63 : Nat.gcd 98 63 = 7 := by
  sorry

end gcd_98_63_l74_74486


namespace selection_methods_count_l74_74723

def binomial (n k : ℕ) : ℕ := Nat.choose n k

theorem selection_methods_count :
  let females := 8
  let males := 4
  (binomial females 2 * binomial males 1) + (binomial females 1 * binomial males 2) = 112 :=
by
  sorry

end selection_methods_count_l74_74723


namespace bug_total_distance_l74_74527

def total_distance_bug (start : ℤ) (pos1 : ℤ) (pos2 : ℤ) (pos3 : ℤ) : ℤ :=
  abs (pos1 - start) + abs (pos2 - pos1) + abs (pos3 - pos2)

theorem bug_total_distance :
  total_distance_bug 3 (-4) 6 2 = 21 :=
by
  -- We insert a sorry here to indicate the proof is skipped.
  sorry

end bug_total_distance_l74_74527


namespace sphere_surface_area_l74_74221

theorem sphere_surface_area (r : ℝ) (h : π * r^2 = 81 * π) : 4 * π * r^2 = 324 * π :=
  sorry

end sphere_surface_area_l74_74221


namespace time_to_destination_l74_74675

-- Variables and Conditions
variable (Harris_speed : ℝ)  -- Speed of Mr. Harris (in units of distance per hour)
variable (your_speed : ℝ)  -- Your speed (in units of distance per hour)
variable (time_Harris : ℝ) -- Time taken by Mr. Harris to reach the store (in hours)
variable (distance_ratio : ℝ) -- Ratio of your destination distance to the store distance

-- Hypotheses based on given conditions
hypothesis h1 : your_speed = 2 * Harris_speed
hypothesis h2 : time_Harris = 2
hypothesis h3 : distance_ratio = 3

-- Statement to prove
theorem time_to_destination : (3 / 2 : ℝ) = 3 := by 
    sorry

end time_to_destination_l74_74675


namespace units_digit_factorial_150_l74_74252

theorem units_digit_factorial_150 : (nat.factorial 150) % 10 = 0 :=
sorry

end units_digit_factorial_150_l74_74252


namespace problem1_problem2_problem3_l74_74552

-- Definition of operation T
def T (x y m n : ℚ) := (m * x + n * y) * (x + 2 * y)

-- Problem 1: Given T(1, -1) = 0 and T(0, 2) = 8, prove m = 1 and n = 1
theorem problem1 (m n : ℚ) (h1 : T 1 (-1) m n = 0) (h2 : T 0 2 m n = 8) : m = 1 ∧ n = 1 := by
  sorry

-- Problem 2: Given the system of inequalities in terms of p and knowing T(x, y) = (mx + ny)(x + 2y) with m = 1 and n = 1
--            has exactly 3 integer solutions, prove the range of values for a is 42 ≤ a < 54
theorem problem2 (a : ℚ) 
  (h1 : ∃ p : ℚ, T (2 * p) (2 - p) 1 1 > 4 ∧ T (4 * p) (3 - 2 * p) 1 1 ≤ a)
  (h2 : ∃! p : ℤ, -1 < p ∧ p ≤ (a - 18) / 12) : 42 ≤ a ∧ a < 54 := by
  sorry

-- Problem 3: Given T(x, y) = T(y, x) when x^2 ≠ y^2, prove m = 2n
theorem problem3 (m n : ℚ) 
  (h : ∀ x y : ℚ, x^2 ≠ y^2 → T x y m n = T y x m n) : m = 2 * n := by
  sorry

end problem1_problem2_problem3_l74_74552


namespace arithmetic_mean_reciprocal_primes_l74_74873

theorem arithmetic_mean_reciprocal_primes :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let r1 := (1 : ℚ) / p1
  let r2 := (1 : ℚ) / p2
  let r3 := (1 : ℚ) / p3
  let r4 := (1 : ℚ) / p4
  (r1 + r2 + r3 + r4) / 4 = 247 / 840 := by
sorry

end arithmetic_mean_reciprocal_primes_l74_74873


namespace arithmetic_mean_reciprocals_primes_l74_74871

theorem arithmetic_mean_reciprocals_primes
  (p : Finset ℕ)
  (h_p : p = {2, 3, 5, 7})
  : (p.sum (λ x, 1 / ↑x) / 4) = (247 / 840) :=
by
  sorry

end arithmetic_mean_reciprocals_primes_l74_74871


namespace price_on_hot_day_l74_74065

noncomputable def regular_price_P (P : ℝ) : Prop :=
  7 * 32 * (P - 0.75) + 3 * 32 * (1.25 * P - 0.75) = 450

theorem price_on_hot_day (P : ℝ) (h : regular_price_P P) : 1.25 * P = 2.50 :=
by sorry

end price_on_hot_day_l74_74065


namespace jon_percentage_increase_l74_74936

def initial_speed : ℝ := 80
def trainings : ℕ := 4
def weeks_per_training : ℕ := 4
def speed_increase_per_week : ℝ := 1

theorem jon_percentage_increase :
  let total_weeks := trainings * weeks_per_training
  let total_increase := total_weeks * speed_increase_per_week
  let final_speed := initial_speed + total_increase
  let percentage_increase := (total_increase / initial_speed) * 100
  percentage_increase = 20 :=
by
  sorry

end jon_percentage_increase_l74_74936


namespace compute_d_l74_74169

noncomputable def d_value (c : ℚ) : ℚ :=
let root1 := (3 : ℚ) + real.sqrt 2 in
let root2 := (3 : ℚ) - real.sqrt 2 in
let root3 := (-36 : ℚ) / (root1 * root2) in
root1 * root2 + (root1 * root3) + (root2 * root3)

theorem compute_d (c : ℚ) (d : ℚ) (h : polynomial.aeval (3 + real.sqrt 2) (X^3 + c * X^2 + d * X - 36) = 0 ∧ c ∈ ℚ ∧ d ∈ ℚ) : 
  d = -23 - (6/7) :=
by
  have eq1 : (3:ℚ) + real.sqrt 2 ≠ (3:ℚ) - real.sqrt 2 := by sorry
  have eq2 : 3 + sqrt 2 ≠ 3 - sqrt 2 := by apply eq1 
  calc
    d = d_value c : by sorry
       ... = -23 - (6/7) : by sorry

end compute_d_l74_74169


namespace cosine_of_tangent_line_at_e_l74_74580

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

theorem cosine_of_tangent_line_at_e :
  let θ := Real.arctan 2
  Real.cos θ = Real.sqrt (1 / 5) := by
  sorry

end cosine_of_tangent_line_at_e_l74_74580


namespace prize_distribution_l74_74193

/--
In a best-of-five competition where two players of equal level meet in the final, 
with a score of 2:1 after the first three games and the total prize money being 12,000 yuan, 
the prize awarded to the player who has won 2 games should be 9,000 yuan.
-/
theorem prize_distribution (prize_money : ℝ) 
  (A_wins : ℕ) (B_wins : ℕ) (prob_A : ℝ) (prob_B : ℝ) (total_games : ℕ) : 
  total_games = 5 → 
  prize_money = 12000 → 
  A_wins = 2 → 
  B_wins = 1 → 
  prob_A = 1/2 → 
  prob_B = 1/2 → 
  ∃ prize_for_A : ℝ, prize_for_A = 9000 :=
by
  intros
  sorry

end prize_distribution_l74_74193


namespace product_is_square_of_24975_l74_74009

theorem product_is_square_of_24975 : (500 * 49.95 * 4.995 * 5000 : ℝ) = (24975 : ℝ) ^ 2 :=
by {
  sorry
}

end product_is_square_of_24975_l74_74009


namespace food_cost_max_l74_74266

theorem food_cost_max (x : ℝ) (total_cost : ℝ) (tax_rate : ℝ) (tip_rate : ℝ) (max_total : ℝ) (food_cost_max : ℝ) :
  total_cost = x * (1 + tax_rate + tip_rate) →
  tax_rate = 0.07 →
  tip_rate = 0.15 →
  max_total = 50 →
  total_cost ≤ max_total →
  food_cost_max = 50 / 1.22 →
  x ≤ food_cost_max :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end food_cost_max_l74_74266


namespace simplify_fraction_l74_74801

theorem simplify_fraction:
  (4^5 + 4^3) / (4^4 - 4^2) = 68 / 15 := by
  sorry

end simplify_fraction_l74_74801


namespace units_produced_today_l74_74304

theorem units_produced_today (n : ℕ) (P : ℕ) (T : ℕ) 
  (h1 : n = 14)
  (h2 : P = 60 * n)
  (h3 : (P + T) / (n + 1) = 62) : 
  T = 90 :=
by
  sorry

end units_produced_today_l74_74304


namespace find_a_l74_74581

noncomputable def A : Set ℝ := {1, 2, 3, 4}
noncomputable def B (a : ℝ) : Set ℝ := { x | x ≤ a }

theorem find_a (a : ℝ) (h_union : A ∪ B a = Set.Iic 5) : a = 5 := by
  sorry

end find_a_l74_74581


namespace range_of_a_l74_74186

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, |x + 2| + |x - 1| > Real.logb 2 a) →
  0 < a ∧ a < 8 :=
by
  sorry

end range_of_a_l74_74186


namespace no_apples_info_l74_74394

theorem no_apples_info (r d : ℕ) (condition1 : r = 79) (condition2 : d = 53) (condition3 : r = d + 26) : 
  ∀ a : ℕ, (a = a) → false :=
by
  intro a h
  sorry

end no_apples_info_l74_74394


namespace Cheerful_snakes_not_Green_l74_74661

variables {Snake : Type} (snakes : Finset Snake)
variable (Cheerful Green CanSing CanMultiply : Snake → Prop)

-- Conditions
axiom Cheerful_impl_CanSing : ∀ s, Cheerful s → CanSing s
axiom Green_impl_not_CanMultiply : ∀ s, Green s → ¬ CanMultiply s
axiom not_CanMultiply_impl_not_CanSing : ∀ s, ¬ CanMultiply s → ¬ CanSing s

-- Question
theorem Cheerful_snakes_not_Green : ∀ s, Cheerful s → ¬ Green s :=
by sorry

end Cheerful_snakes_not_Green_l74_74661


namespace solve_for_y_l74_74671

theorem solve_for_y 
  (a b c d y : ℚ) 
  (h₀ : a ≠ b) 
  (h₁ : a ≠ 0) 
  (h₂ : c ≠ d) 
  (h₃ : (b + y) / (a + y) = d / c) : 
  y = (a * d - b * c) / (c - d) :=
by
  sorry

end solve_for_y_l74_74671


namespace fraction_sum_simplest_l74_74228

theorem fraction_sum_simplest (a b : ℕ) (h : 0.84375 = (a : ℝ) / b) (ha : a = 27) (hb : b = 32) : a + b = 59 :=
by
  rw [ha, hb]
  norm_num
  sorry

end fraction_sum_simplest_l74_74228


namespace quadratic_solution_downward_solution_minimum_solution_l74_74910

def is_quadratic (m : ℝ) : Prop :=
  m^2 + 3 * m - 2 = 2

def opens_downwards (m : ℝ) : Prop :=
  m + 3 < 0

def has_minimum (m : ℝ) : Prop :=
  m + 3 > 0

theorem quadratic_solution (m : ℝ) :
  is_quadratic m → (m = -4 ∨ m = 1) :=
sorry

theorem downward_solution (m : ℝ) :
  is_quadratic m → opens_downwards m → m = -4 :=
sorry

theorem minimum_solution (m : ℝ) :
  is_quadratic m → has_minimum m → m = 1 :=
sorry

end quadratic_solution_downward_solution_minimum_solution_l74_74910


namespace sequence_an_correct_l74_74047

noncomputable def seq_an (n : ℕ) : ℚ :=
if h : n = 1 then 1 else (1 / (2 * n - 1) - 1 / (2 * n - 3))

theorem sequence_an_correct (n : ℕ) (S : ℕ → ℚ)
  (h1 : S 1 = 1)
  (h2 : ∀ n ≥ 2, S n ^ 2 = seq_an n * (S n - 0.5)) :
  seq_an n = if n = 1 then 1 else (1 / (2 * n - 1) - 1 / (2 * n - 3)) :=
sorry

end sequence_an_correct_l74_74047


namespace triangle_ABC_AC_l74_74602

-- Defining the relevant points and lengths in the triangle
variables {A B C D : Type} 
variables (AB CD : ℝ)
variables (AD BC AC : ℝ)

-- Given constants
axiom hAB : AB = 3
axiom hCD : CD = Real.sqrt 3
axiom hAD_BC : AD = BC

-- The final theorem statement that needs to be proved
theorem triangle_ABC_AC :
  (AD = BC) ∧ (CD = Real.sqrt 3) ∧ (AB = 3) → AC = Real.sqrt 7 :=
by
  intros h
  sorry

end triangle_ABC_AC_l74_74602


namespace acid_concentration_in_third_flask_l74_74819

theorem acid_concentration_in_third_flask 
    (w : ℚ) (W : ℚ) (hw : 10 / (10 + w) = 1 / 20) (hW : 20 / (20 + (W - w)) = 7 / 30) 
    (W_total : W = 256.43) : 
    30 / (30 + W) = 21 / 200 := 
by 
  sorry

end acid_concentration_in_third_flask_l74_74819


namespace sum_of_present_ages_l74_74453

def Jed_age_future (current_Jed: ℕ) (years: ℕ) : ℕ := 
  current_Jed + years

def Matt_age (current_Jed: ℕ) : ℕ := 
  current_Jed - 10

def sum_ages (jed_age: ℕ) (matt_age: ℕ) : ℕ := 
  jed_age + matt_age

theorem sum_of_present_ages :
  ∃ jed_curr_age matt_curr_age : ℕ, 
  (Jed_age_future jed_curr_age 10 = 25) ∧ 
  (jed_curr_age = matt_curr_age + 10) ∧ 
  (sum_ages jed_curr_age matt_curr_age = 20) :=
sorry

end sum_of_present_ages_l74_74453


namespace characterize_functional_equation_l74_74297

def satisfies_condition (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x^2 - y^2) = x * f x - y * f y

theorem characterize_functional_equation (f : ℝ → ℝ) (h : satisfies_condition f) :
  ∃ c : ℝ, ∀ x : ℝ, f x = c * x :=
sorry

end characterize_functional_equation_l74_74297


namespace flat_rate_first_night_l74_74136

-- Definitions of conditions
def total_cost_sarah (f n : ℕ) := f + 3 * n = 210
def total_cost_mark (f n : ℕ) := f + 7 * n = 450

-- Main theorem to be proven
theorem flat_rate_first_night : 
  ∃ f n : ℕ, total_cost_sarah f n ∧ total_cost_mark f n ∧ f = 30 :=
by
  sorry

end flat_rate_first_night_l74_74136


namespace arithmetic_sequence_sum_l74_74575

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Given conditions
variables (a : ℕ → ℝ)
variable (h_arith : is_arithmetic_sequence a)
variable (h_sum : a 2 + a 3 + a 10 + a 11 = 48)

-- Goal
theorem arithmetic_sequence_sum : a 6 + a 7 = 24 :=
sorry

end arithmetic_sequence_sum_l74_74575


namespace range_of_a_l74_74909

noncomputable def function_with_extreme_at_zero_only (a b : ℝ) : Prop :=
∀ x : ℝ, x ≠ 0 → 4 * x^2 + 3 * a * x + 4 > 0

theorem range_of_a (a b : ℝ) (h : function_with_extreme_at_zero_only a b) : 
  -8 / 3 ≤ a ∧ a ≤ 8 / 3 :=
sorry

end range_of_a_l74_74909


namespace total_cost_price_is_584_l74_74800

-- Define the costs of individual items
def cost_watch : ℕ := 144
def cost_bracelet : ℕ := 250
def cost_necklace : ℕ := 190

-- The proof statement: the total cost price is 584
theorem total_cost_price_is_584 : cost_watch + cost_bracelet + cost_necklace = 584 :=
by
  -- We skip the proof steps here, assuming the above definitions are correct.
  sorry

end total_cost_price_is_584_l74_74800


namespace function_domain_l74_74644

theorem function_domain (x : ℝ) :
  (x - 3 > 0) ∧ (5 - x ≥ 0) ↔ (3 < x ∧ x ≤ 5) :=
by
  sorry

end function_domain_l74_74644


namespace inequality_transformations_l74_74759

theorem inequality_transformations (a b : ℝ) (h : a > b) :
  (3 * a > 3 * b) ∧ (a + 2 > b + 2) ∧ (-5 * a < -5 * b) :=
by
  sorry

end inequality_transformations_l74_74759


namespace time_taken_l74_74588

-- Define the function T which takes the number of cats, the number of rats, and returns the time in minutes
def T (n m : ℕ) : ℕ := if n = m then 4 else sorry

-- The theorem states that, given n cats and n rats, the time taken is 4 minutes
theorem time_taken (n : ℕ) : T n n = 4 :=
by simp [T]

end time_taken_l74_74588


namespace golden_section_MP_length_l74_74734

noncomputable def golden_ratio : ℝ := (Real.sqrt 5 + 1) / 2

theorem golden_section_MP_length (MN : ℝ) (hMN : MN = 2) (P : ℝ) 
  (hP : P > 0 ∧ P < MN ∧ P / (MN - P) = (MN - P) / P)
  (hMP_NP : MN - P < P) :
  P = Real.sqrt 5 - 1 :=
by
  sorry

end golden_section_MP_length_l74_74734


namespace smallest_common_multiple_8_6_l74_74510

theorem smallest_common_multiple_8_6 : 
  ∃ n : ℕ, n > 0 ∧ (n % 8 = 0) ∧ (n % 6 = 0) ∧ ∀ m : ℕ, m > 0 ∧ (m % 8 = 0) ∧ (m % 6 = 0) → m ≥ n :=
begin
  use 24,
  split,
  { norm_num }, -- 24 > 0
  split,
  { norm_num }, -- 24 % 8 = 0
  split,
  { norm_num }, -- 24 % 6 = 0
  { intros m hm,
    cases hm with hp8 hp6,
    norm_num at hp8 hp6,
    sorry -- Prove that 24 is the smallest such number
  }
end

end smallest_common_multiple_8_6_l74_74510


namespace circle_through_ABC_l74_74912

-- Define points A, B, and C
def A : ℝ × ℝ := (-1, 0)
def B : ℝ × ℝ := (3, 0)
def C : ℝ × ℝ := (1, 4)

-- Define the circle equation components to be proved
def circle_eqn (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 3*y - 3 = 0

-- The theorem statement that we need to prove
theorem circle_through_ABC : 
  ∃ (D E F : ℝ), (∀ x y, (x, y) = A ∨ (x, y) = B ∨ (x, y) = C → x^2 + y^2 + D*x + E*y + F = 0) 
  → circle_eqn x y :=
sorry

end circle_through_ABC_l74_74912


namespace units_digit_150_factorial_is_zero_l74_74254

theorem units_digit_150_factorial_is_zero :
  (nat.trailing_digits (nat.factorial 150) 1) = 0 :=
by sorry

end units_digit_150_factorial_is_zero_l74_74254


namespace textile_firm_looms_l74_74398

theorem textile_firm_looms
  (sales_val : ℝ)
  (manu_exp : ℝ)
  (estab_charges : ℝ)
  (profit_decrease : ℝ)
  (L : ℝ)
  (h_sales : sales_val = 500000)
  (h_manu_exp : manu_exp = 150000)
  (h_estab_charges : estab_charges = 75000)
  (h_profit_decrease : profit_decrease = 7000)
  (hem_equal_contrib : ∀ l : ℝ, l > 0 →
    (l = sales_val / (sales_val / L) - manu_exp / (manu_exp / L)))
  : L = 50 := 
by
  sorry

end textile_firm_looms_l74_74398


namespace root_interval_l74_74686

noncomputable def f (x : ℝ) : ℝ := Real.log (x + 1) + 2 * x - 1

theorem root_interval : ∃ m : ℝ, f m = 0 ∧ 0 < m ∧ m < 1 :=
by
  have h_decreasing : ∀ x y : ℝ, x < y → f x < f y :=
    sorry -- Proof that f is increasing on (-1, +∞)
  have h_f0 : f 0 = -1 := by
    sorry -- Calculation that f(0) = -1
  have h_f1 : f 1 = Real.log 2 + 1 := by
    sorry -- Calculation that f(1) = ln(2) + 1
  have h_exist_root : ∃ m : ℝ, f m = 0 ∧ 0 < m ∧ m < 1 :=
    by
      sorry -- Existence of a root in (0,1)
  exact h_exist_root

end root_interval_l74_74686


namespace percentage_decrease_l74_74045

theorem percentage_decrease (x y : ℝ) :
  let x' := 0.8 * x
  let y' := 0.7 * y
  let original_expr := x^2 * y^3
  let new_expr := (x')^2 * (y')^3
  let perc_decrease := (original_expr - new_expr) / original_expr * 100
  perc_decrease = 78.048 := by
  sorry

end percentage_decrease_l74_74045


namespace birthday_check_value_l74_74121

theorem birthday_check_value : 
  ∃ C : ℝ, (150 + C) / 4 = C ↔ C = 50 :=
by
  sorry

end birthday_check_value_l74_74121


namespace find_middle_number_l74_74492

namespace Problem

-- Define the three numbers x, y, z
variables (x y z : ℕ)

-- Given conditions from the problem
def condition1 (h1 : x + y = 18) := x + y = 18
def condition2 (h2 : x + z = 23) := x + z = 23
def condition3 (h3 : y + z = 27) := y + z = 27
def condition4 (h4 : x < y ∧ y < z) := x < y ∧ y < z

-- Statement to prove:
theorem find_middle_number (h1 : x + y = 18) (h2 : x + z = 23) (h3 : y + z = 27) (h4 : x < y ∧ y < z) : 
  y = 11 :=
by
  sorry

end Problem

end find_middle_number_l74_74492


namespace positive_integer_solutions_l74_74641

theorem positive_integer_solutions (x : ℕ) (h : 2 * x + 9 ≥ 3 * (x + 2)) : x = 1 ∨ x = 2 ∨ x = 3 :=
by
  sorry

end positive_integer_solutions_l74_74641


namespace length_of_square_side_l74_74966

-- Definitions based on conditions
def perimeter_of_triangle : ℝ := 46
def total_perimeter : ℝ := 78
def perimeter_of_square : ℝ := total_perimeter - perimeter_of_triangle

-- Lean statement for the problem
theorem length_of_square_side : perimeter_of_square / 4 = 8 := by
  sorry

end length_of_square_side_l74_74966


namespace remainder_of_2357916_div_8_l74_74244

theorem remainder_of_2357916_div_8 : (2357916 % 8) = 4 := by
  sorry

end remainder_of_2357916_div_8_l74_74244


namespace problem_statement_l74_74203

theorem problem_statement (a b c : ℝ) (h : a^6 + b^6 + c^6 = 3) : a^7 * b^2 + b^7 * c^2 + c^7 * a^2 ≤ 3 := 
  sorry

end problem_statement_l74_74203


namespace pentagon_square_ratio_l74_74543

theorem pentagon_square_ratio (p s : ℝ) (h₁ : 5 * p = 20) (h₂ : 4 * s = 20) : p / s = 4 / 5 := 
by 
  sorry

end pentagon_square_ratio_l74_74543


namespace number_of_spotted_blue_fish_l74_74368

def total_fish := 60
def blue_fish := total_fish / 3
def spotted_blue_fish := blue_fish / 2

theorem number_of_spotted_blue_fish : spotted_blue_fish = 10 :=
by
  -- Proof is omitted
  sorry

end number_of_spotted_blue_fish_l74_74368


namespace stripes_distance_l74_74854

theorem stripes_distance (d : ℝ) (L : ℝ) (c : ℝ) (y : ℝ) 
  (hd : d = 40) (hL : L = 50) (hc : c = 15)
  (h_ratio : y / d = c / L) : y = 12 :=
by
  rw [hd, hL, hc] at h_ratio
  sorry

end stripes_distance_l74_74854


namespace correct_equation_l74_74325

theorem correct_equation (x : ℕ) (h : x ≤ 26) :
    let a_parts := 2100
    let b_parts := 1200
    let total_workers := 26
    let a_rate := 30
    let b_rate := 20
    let type_a_time := (a_parts : ℚ) / (a_rate * x)
    let type_b_time := (b_parts : ℚ) / (b_rate * (total_workers - x))
    type_a_time = type_b_time :=
by
    sorry

end correct_equation_l74_74325


namespace fraction_not_covered_l74_74237

/--
Given that frame X has a diameter of 16 cm and frame Y has a diameter of 12 cm,
prove that the fraction of the surface of frame X that is not covered by frame Y is 7/16.
-/
theorem fraction_not_covered (dX dY : ℝ) (hX : dX = 16) (hY : dY = 12) : 
  let rX := dX / 2
  let rY := dY / 2
  let AX := Real.pi * rX^2
  let AY := Real.pi * rY^2
  let uncovered_area := AX - AY
  let fraction_not_covered := uncovered_area / AX
  fraction_not_covered = 7 / 16 :=
by
  sorry

end fraction_not_covered_l74_74237


namespace prime_factor_of_sum_of_four_consecutive_integers_l74_74098

-- Define four consecutive integers and their sum
def sum_four_consecutive_integers (n : ℤ) : ℤ := (n - 1) + n + (n + 1) + (n + 2)

-- The theorem states that 2 is a divisor of the sum of any four consecutive integers
theorem prime_factor_of_sum_of_four_consecutive_integers (n : ℤ) : 
  ∃ p : ℤ, Prime p ∧ p ∣ sum_four_consecutive_integers n :=
begin
  use 2,
  split,
  {
    apply Prime_two,
  },
  {
    unfold sum_four_consecutive_integers,
    norm_num,
    exact dvd.intro (2 * n + 1) rfl,
  },
end

end prime_factor_of_sum_of_four_consecutive_integers_l74_74098


namespace expression_of_f_f_increasing_on_interval_inequality_solution_l74_74432

noncomputable def f (x : ℝ) : ℝ := (x / (1 + x^2))

-- 1. Proving f(x) is the given function
theorem expression_of_f (x : ℝ) (h₁ : f x = (a*x + b) / (1 + x^2)) (h₂ : (∀ x, f (-x) = -f x)) (h₃ : f (1/2) = 2/5) :
  f x = x / (1 + x^2) :=
sorry

-- 2. Prove f(x) is increasing on (-1,1)
theorem f_increasing_on_interval {x₁ x₂ : ℝ} (h₁ : -1 < x₁ ∧ x₁ < 1) (h₂ : -1 < x₂ ∧ x₂ < 1) (h₃ : x₁ < x₂) :
  f x₁ < f x₂ :=
sorry

-- 3. Solve the inequality f(t-1) + f(t) < 0 on (0, 1/2)
theorem inequality_solution (t : ℝ) (h₁ : 0 < t) (h₂ : t < 1/2) :
  f (t - 1) + f t < 0 :=
sorry

end expression_of_f_f_increasing_on_interval_inequality_solution_l74_74432


namespace correct_sum_of_integers_l74_74101

theorem correct_sum_of_integers (x y : ℕ) (h1 : x - y = 4) (h2 : x * y = 192) : x + y = 28 := by
  sorry

end correct_sum_of_integers_l74_74101


namespace probability_event_A_occurrence_at_least_two_and_at_most_four_times_l74_74780

theorem probability_event_A_occurrence_at_least_two_and_at_most_four_times :
  let n := 2000
  let p := 0.001
  let lambda := n * p
  ∑ k in finset.range 5, if k >= 2 then (lambda^k / (k.factorial * Real.exp(lambda))) else 0 = 0.541 := 
sorry

end probability_event_A_occurrence_at_least_two_and_at_most_four_times_l74_74780


namespace probability_blue_given_popped_is_18_over_53_l74_74132

section PopcornProblem

/-- Representation of probabilities -/
def prob_white : ℚ := 1 / 2
def prob_yellow : ℚ := 1 / 4
def prob_blue : ℚ := 1 / 4

def pop_white_given_white : ℚ := 1 / 2
def pop_yellow_given_yellow : ℚ := 3 / 4
def pop_blue_given_blue : ℚ := 9 / 10

/-- Joint probabilities of kernel popping -/
def prob_white_popped : ℚ := prob_white * pop_white_given_white
def prob_yellow_popped : ℚ := prob_yellow * pop_yellow_given_yellow
def prob_blue_popped : ℚ := prob_blue * pop_blue_given_blue

/-- Total probability of popping -/
def prob_popped : ℚ := prob_white_popped + prob_yellow_popped + prob_blue_popped

/-- Conditional probability of being a blue kernel given that it popped -/
def prob_blue_given_popped : ℚ := prob_blue_popped / prob_popped

/-- The main theorem to prove the final probability -/
theorem probability_blue_given_popped_is_18_over_53 :
  prob_blue_given_popped = 18 / 53 :=
by sorry

end PopcornProblem

end probability_blue_given_popped_is_18_over_53_l74_74132


namespace side_salad_cost_l74_74061

theorem side_salad_cost (T S : ℝ)
  (h1 : T + S + 4 + 2 = 2 * T) 
  (h2 : (T + S + 4 + 2) + T = 24) : S = 2 :=
by
  sorry

end side_salad_cost_l74_74061


namespace Namjoon_walk_extra_l74_74946

-- Define the usual distance Namjoon walks to school
def usual_distance := 1.2

-- Define the distance Namjoon walked to the intermediate point
def intermediate_distance := 0.3

-- Define the total distance Namjoon walked today
def total_distance_today := (intermediate_distance * 2) + usual_distance

-- Define the extra distance walked today compared to usual
def extra_distance := total_distance_today - usual_distance

-- State the theorem to prove that the extra distance walked today is 0.6 km
theorem Namjoon_walk_extra : extra_distance = 0.6 := 
by
  sorry

end Namjoon_walk_extra_l74_74946


namespace molecular_weight_n2o_l74_74977

theorem molecular_weight_n2o (w : ℕ) (n : ℕ) (h : w = 352 ∧ n = 8) : (w / n = 44) :=
sorry

end molecular_weight_n2o_l74_74977


namespace december_revenue_times_average_l74_74036

variable (D : ℝ) -- December's revenue
variable (N : ℝ) -- November's revenue
variable (J : ℝ) -- January's revenue

-- Conditions
def revenue_in_november : N = (2/5) * D := by sorry
def revenue_in_january : J = (1/2) * N := by sorry

-- Statement to be proved
theorem december_revenue_times_average :
  D = (10/3) * ((N + J) / 2) :=
by sorry

end december_revenue_times_average_l74_74036


namespace initial_cakes_l74_74866

variable (friend_bought : Nat) (baker_has : Nat)

theorem initial_cakes (h1 : friend_bought = 140) (h2 : baker_has = 15) : 
  (friend_bought + baker_has = 155) := 
by
  sorry

end initial_cakes_l74_74866


namespace base_of_hill_depth_l74_74222

theorem base_of_hill_depth : 
  ∀ (H : ℕ), 
  (H = 900) → 
  (1 / 4 * H = 225) :=
by
  intros H h
  sorry

end base_of_hill_depth_l74_74222


namespace sin_double_angle_plus_pi_over_4_l74_74757

theorem sin_double_angle_plus_pi_over_4 (α : ℝ) 
  (h : Real.tan α = 3) : 
  Real.sin (2 * α + Real.pi / 4) = -Real.sqrt 2 / 10 := 
by 
  sorry

end sin_double_angle_plus_pi_over_4_l74_74757


namespace find_value_of_b_l74_74019

theorem find_value_of_b (x b : ℕ) 
    (h1 : 5 * (x + 8) = 5 * x + b + 33) : b = 7 :=
sorry

end find_value_of_b_l74_74019


namespace impossible_seed_germinate_without_water_l74_74119

-- Definitions for the conditions
def heats_up_when_conducting (conducts : Bool) : Prop := conducts
def determines_plane (non_collinear : Bool) : Prop := non_collinear
def germinates_without_water (germinates : Bool) : Prop := germinates
def wins_lottery_consecutively (wins_twice : Bool) : Prop := wins_twice

-- The fact that a seed germinates without water is impossible
theorem impossible_seed_germinate_without_water 
  (conducts : Bool) 
  (non_collinear : Bool) 
  (germinates : Bool) 
  (wins_twice : Bool) 
  (h1 : heats_up_when_conducting conducts) 
  (h2 : determines_plane non_collinear) 
  (h3 : ¬germinates_without_water germinates) 
  (h4 : wins_lottery_consecutively wins_twice) :
  ¬germinates_without_water true :=
sorry

end impossible_seed_germinate_without_water_l74_74119


namespace radius_range_l74_74171

-- Conditions:
-- r1 is the radius of circle O1
-- r2 is the radius of circle O2
-- d is the distance between centers of circles O1 and O2
-- PO1 is the distance from a point P on circle O2 to the center of circle O1

variables (r1 r2 d PO1 : ℝ)

-- Given r1 = 1, d = 5, PO1 = 2
axiom r1_def : r1 = 1
axiom d_def : d = 5
axiom PO1_def : PO1 = 2

-- To prove: 3 ≤ r2 ≤ 7
theorem radius_range (r2 : ℝ) (h : d = 5 ∧ r1 = 1 ∧ PO1 = 2 ∧ (∃ P : ℝ, P = r2)) : 3 ≤ r2 ∧ r2 ≤ 7 :=
by {
  sorry
}

end radius_range_l74_74171


namespace rectangle_square_ratio_l74_74064

theorem rectangle_square_ratio (l w s : ℝ) (h1 : 0.4 * l * w = 0.25 * s * s) : l / w = 15.625 :=
by
  sorry

end rectangle_square_ratio_l74_74064


namespace general_formula_l74_74435

-- Define the sequence term a_n
def sequence_term (n : ℕ) : ℚ :=
  if h : n = 0 then 1
  else (2 * n - 1 : ℚ) / (n * n)

-- State the theorem for the general formula of the nth term
theorem general_formula (n : ℕ) (hn : n ≠ 0) : 
  sequence_term n = (2 * n - 1 : ℚ) / (n * n) :=
by sorry

end general_formula_l74_74435


namespace sum_of_squares_l74_74586

theorem sum_of_squares (a b : ℝ) (h1 : (a + b)^2 = 11) (h2 : (a - b)^2 = 5) : a^2 + b^2 = 8 := 
sorry

end sum_of_squares_l74_74586


namespace p_true_of_and_not_p_false_l74_74919

variable {p q : Prop}

theorem p_true_of_and_not_p_false (h1 : ¬(p ∧ q)) (h2 : ¬¬p) : p :=
sorry

end p_true_of_and_not_p_false_l74_74919


namespace painter_earnings_l74_74397

noncomputable theory
open_locale big_operators

variables (n_s n_n : ℕ → ℕ) (digit_cost : ℕ → ℕ) (south_total north_total : ℕ)

def south_addresses : ℕ → ℕ :=
  λ n, 5 + (n - 1) * 7

def north_addresses : ℕ → ℕ :=
  λ n, 7 + (n - 1) * 8

def count_digits : ℕ → ℕ
| n := if n < 10 then 1 else if n < 100 then 2 else 3

def total_digit_cost (addr_fn : ℕ → ℕ) (n : ℕ) : ℕ :=
  (finset.range n).sum (λ i, count_digits (addr_fn (i + 1)))

theorem painter_earnings : 
  total_digit_cost south_addresses 25 + total_digit_cost north_addresses 25 = 125 :=
by sorry

end painter_earnings_l74_74397


namespace pocket_knife_worth_40_l74_74530

def value_of_pocket_knife (x : ℕ) (p : ℕ) (R : ℕ) : Prop :=
  p = 10 * x ∧
  R = 10 * x^2 ∧
  (∃ num_100_bills : ℕ, 2 * num_100_bills * 100 + 40 = R)

theorem pocket_knife_worth_40 (x : ℕ) (p : ℕ) (R : ℕ) :
  value_of_pocket_knife x p R → (∃ knife_value : ℕ, knife_value = 40) :=
by
  sorry

end pocket_knife_worth_40_l74_74530


namespace fair_share_of_bill_l74_74383

noncomputable def total_bill : Real := 139.00
noncomputable def tip_percent : Real := 0.10
noncomputable def num_people : Real := 6
noncomputable def expected_amount_per_person : Real := 25.48

theorem fair_share_of_bill :
  (total_bill + (tip_percent * total_bill)) / num_people = expected_amount_per_person :=
by
  sorry

end fair_share_of_bill_l74_74383


namespace find_original_expression_l74_74155

theorem find_original_expression (a b c X : ℤ) :
  X + (a * b - 2 * b * c + 3 * a * c) = 2 * b * c - 3 * a * c + 2 * a * b →
  X = 4 * b * c - 6 * a * c + a * b :=
by
  sorry

end find_original_expression_l74_74155


namespace correct_transformation_l74_74275

theorem correct_transformation (a b : ℝ) (hb : b ≠ 0) : 
  (a / b) = ((a + 2 * a) / (b + 2 * b)) :=
by 
  sorry

end correct_transformation_l74_74275


namespace find_carbon_atoms_l74_74529

variable (n : ℕ)
variable (molecular_weight : ℝ := 124.0)
variable (weight_Cu : ℝ := 63.55)
variable (weight_C : ℝ := 12.01)
variable (weight_O : ℝ := 16.00)
variable (num_Cu : ℕ := 1)
variable (num_O : ℕ := 3)

theorem find_carbon_atoms 
  (h : molecular_weight = (num_Cu * weight_Cu) + (n * weight_C) + (num_O * weight_O)) : 
  n = 1 :=
sorry

end find_carbon_atoms_l74_74529


namespace second_quadrant_necessary_not_sufficient_l74_74685

variable (α : ℝ)

def is_obtuse (α : ℝ) : Prop := 90 < α ∧ α < 180
def is_second_quadrant (α : ℝ) : Prop := 90 < α ∧ α < 180

theorem second_quadrant_necessary_not_sufficient : 
  (∀ α, is_obtuse α → is_second_quadrant α) ∧ ¬ (∀ α, is_second_quadrant α → is_obtuse α) := by
  sorry

end second_quadrant_necessary_not_sufficient_l74_74685


namespace min_value_product_l74_74668

theorem min_value_product (a : Fin 6 → ℕ) 
  (hperm : ∃ σ : Equiv.Perm (Fin 6), ∀ i, a i = σ i) :
  (∏ i, (a i - a ((i + 1) % 6)) / (a ((i + 2) % 6) - a ((i + 3) % 6))) = 1 := by  
  sorry

end min_value_product_l74_74668


namespace jessica_repay_l74_74934

theorem jessica_repay (P : ℝ) (r : ℝ) (n : ℝ) (x : ℕ)
  (hx : P = 20)
  (hr : r = 0.12)
  (hn : n = 3 * P) :
  x = 17 :=
sorry

end jessica_repay_l74_74934


namespace cars_cost_between_15000_and_20000_l74_74405

theorem cars_cost_between_15000_and_20000 (total_cars : ℕ) (p1 p2 : ℕ) :
    total_cars = 3000 → 
    p1 = 15 → 
    p2 = 40 → 
    (p1 * total_cars / 100 + p2 * total_cars / 100 + x = total_cars) → 
    x = 1350 :=
by
  intro h_total
  intro h_p1
  intro h_p2
  intro h_eq
  sorry

end cars_cost_between_15000_and_20000_l74_74405


namespace sum_abs_binom_coeff_l74_74903

theorem sum_abs_binom_coeff (a a1 a2 a3 a4 a5 a6 a7 : ℤ)
    (h : (1 - 2 * x) ^ 7 = a + a1 * x + a2 * x^2 + a3 * x^3 + a4 * x^4 + a5 * x^5 + a6 * x^6 + a7 * x^7) :
    |a1| + |a2| + |a3| + |a4| + |a5| + |a6| + |a7| = 3 ^ 7 - 1 := sorry

end sum_abs_binom_coeff_l74_74903


namespace sum_of_four_consecutive_integers_prime_factor_l74_74096

theorem sum_of_four_consecutive_integers_prime_factor (n : ℤ) : ∃ p : ℤ, Prime p ∧ p = 2 ∧ ∀ n : ℤ, p ∣ ((n - 1) + n + (n + 1) + (n + 2)) := 
by 
  sorry

end sum_of_four_consecutive_integers_prime_factor_l74_74096


namespace sum_of_four_consecutive_integers_prime_factor_l74_74095

theorem sum_of_four_consecutive_integers_prime_factor (n : ℤ) : ∃ p : ℤ, Prime p ∧ p = 2 ∧ ∀ n : ℤ, p ∣ ((n - 1) + n + (n + 1) + (n + 2)) := 
by 
  sorry

end sum_of_four_consecutive_integers_prime_factor_l74_74095


namespace victor_earnings_l74_74974

def hourly_wage := 6 -- dollars per hour
def hours_monday := 5 -- hours
def hours_tuesday := 5 -- hours

theorem victor_earnings : (hourly_wage * (hours_monday + hours_tuesday)) = 60 :=
by
  sorry

end victor_earnings_l74_74974


namespace cube_volume_given_surface_area_l74_74840

theorem cube_volume_given_surface_area (s : ℝ) (h₀ : 6 * s^2 = 150) : s^3 = 125 := by
  sorry

end cube_volume_given_surface_area_l74_74840


namespace square_area_inscribed_triangle_l74_74142

-- Definitions from the conditions of the problem
variable (EG : ℝ) (hF : ℝ)

-- Since EG = 12 inches and the altitude from F to EG is 7 inches
theorem square_area_inscribed_triangle 
(EG_eq : EG = 12) 
(hF_eq : hF = 7) :
  ∃ (AB : ℝ), AB ^ 2 = 36 :=
by 
  sorry

end square_area_inscribed_triangle_l74_74142


namespace solve_for_x_l74_74578

theorem solve_for_x :
  ∃ x : ℤ, (225 - 4209520 / ((1000795 + (250 + x) * 50) / 27)) = 113 ∧ x = 40 := 
by
  sorry

end solve_for_x_l74_74578


namespace age_problem_l74_74189

theorem age_problem 
  (x y z u : ℕ)
  (h1 : x + 6 = 3 * (y - u))
  (h2 : x = y + z - u)
  (h3: y = x - u) 
  (h4 : x + 19 = 2 * z):
  x = 69 ∧ y = 47 ∧ z = 44 :=
by
  sorry

end age_problem_l74_74189


namespace find_m_l74_74321

theorem find_m (m : ℝ) (h : 2 / m = (m + 1) / 3) : m = -3 := by
  sorry

end find_m_l74_74321


namespace binom_coefficient_largest_l74_74027

theorem binom_coefficient_largest (n : ℕ) (h : (n / 2) + 1 = 7) : n = 12 :=
by
  sorry

end binom_coefficient_largest_l74_74027


namespace lineup_count_l74_74410

theorem lineup_count (n k : ℕ) (h : n = 13) (k_eq : k = 4) : (n.choose k) = 715 := by
  sorry

end lineup_count_l74_74410


namespace arithmetic_square_root_of_16_l74_74072

theorem arithmetic_square_root_of_16 : ∃! (x : ℝ), x^2 = 16 ∧ x ≥ 0 :=
by
  sorry

end arithmetic_square_root_of_16_l74_74072


namespace Yuna_drank_most_l74_74604

noncomputable def Jimin_juice : ℝ := 0.7
noncomputable def Eunji_juice : ℝ := Jimin_juice - 1/10
noncomputable def Yoongi_juice : ℝ := 4/5
noncomputable def Yuna_juice : ℝ := Jimin_juice + 0.2

theorem Yuna_drank_most :
  Yuna_juice = max (max Jimin_juice Eunji_juice) (max Yoongi_juice Yuna_juice) :=
by
  sorry

end Yuna_drank_most_l74_74604


namespace sabrina_herbs_l74_74626

theorem sabrina_herbs (S V : ℕ) 
  (h1 : 2 * S = 12)
  (h2 : 12 + S + V = 29) :
  V - S = 5 := by
  sorry

end sabrina_herbs_l74_74626


namespace ratio_of_boys_l74_74449

theorem ratio_of_boys (p : ℝ) (h : p = (3/4) * (1 - p)) : 
  p = 3 / 7 := 
by 
  sorry

end ratio_of_boys_l74_74449


namespace largest_divisor_of_n4_minus_n_l74_74721

theorem largest_divisor_of_n4_minus_n (n : ℤ) (h : ∃ k : ℤ, n = 4 * k) : 4 ∣ (n^4 - n) :=
by sorry

end largest_divisor_of_n4_minus_n_l74_74721


namespace nonWhiteHomesWithoutFireplace_l74_74493

-- Definitions based on the conditions
def totalHomes : ℕ := 400
def whiteHomes (h : ℕ) : ℕ := h / 4
def nonWhiteHomes (h w : ℕ) : ℕ := h - w
def nonWhiteHomesWithFireplace (nh : ℕ) : ℕ := nh / 5

-- Theorem statement to prove the required result
theorem nonWhiteHomesWithoutFireplace : 
  let h := totalHomes
  let w := whiteHomes h
  let nh := nonWhiteHomes h w
  let nf := nonWhiteHomesWithFireplace nh
  nh - nf = 240 :=
by
  let h := totalHomes
  let w := whiteHomes h
  let nh := nonWhiteHomes h w
  let nf := nonWhiteHomesWithFireplace nh
  show nh - nf = 240
  sorry

end nonWhiteHomesWithoutFireplace_l74_74493


namespace measure_of_angle_Q_in_hexagon_l74_74553

theorem measure_of_angle_Q_in_hexagon :
  ∀ (Q : ℝ),
    (∃ (angles : List ℝ),
      angles = [134, 108, 122, 99, 87] ∧ angles.sum = 550) →
    180 * (6 - 2) - (134 + 108 + 122 + 99 + 87) = 170 → Q = 170 := by
  sorry

end measure_of_angle_Q_in_hexagon_l74_74553


namespace matrix_pow_A_50_l74_74199

def A : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![5, 2], ![-16, -6]]

theorem matrix_pow_A_50 :
  A ^ 50 = ![![301, 100], ![-800, -249]] :=
by
  sorry

end matrix_pow_A_50_l74_74199


namespace calories_in_250g_mixed_drink_l74_74196

def calories_in_mixed_drink (grams_cranberry : ℕ) (grams_honey : ℕ) (grams_water : ℕ)
  (calories_per_100g_cranberry : ℕ) (calories_per_100g_honey : ℕ) (calories_per_100g_water : ℕ)
  (total_grams : ℕ) (portion_grams : ℕ) : ℚ :=
  ((grams_cranberry * calories_per_100g_cranberry + grams_honey * calories_per_100g_honey + grams_water * calories_per_100g_water) : ℚ)
  / (total_grams * portion_grams)

theorem calories_in_250g_mixed_drink :
  calories_in_mixed_drink 150 50 300 30 304 0 100 250 = 98.5 := by
  -- The proof will involve arithmetic operations
  sorry

end calories_in_250g_mixed_drink_l74_74196


namespace semicircles_problem_l74_74411

-- Define the problem in Lean
theorem semicircles_problem 
  (D : ℝ) -- Diameter of the large semicircle
  (N : ℕ) -- Number of small semicircles
  (r : ℝ) -- Radius of each small semicircle
  (H1 : D = 2 * N * r) -- Combined diameter of small semicircles is equal to the large semicircle's diameter
  (H2 : (N * (π * r^2 / 2)) / ((π * (N * r)^2 / 2) - (N * (π * r^2 / 2))) = 1 / 10) -- Ratio of areas condition
  : N = 11 :=
   sorry -- Proof to be filled in later

end semicircles_problem_l74_74411


namespace cylinder_volume_ratio_l74_74131

noncomputable def volume_of_cylinder (height : ℝ) (circumference : ℝ) : ℝ := 
  let r := circumference / (2 * Real.pi)
  Real.pi * r^2 * height

theorem cylinder_volume_ratio :
  let V1 := volume_of_cylinder 10 6
  let V2 := volume_of_cylinder 6 10
  max V1 V2 / min V1 V2 = 5 / 3 :=
by
  let V1 := volume_of_cylinder 10 6
  let V2 := volume_of_cylinder 6 10
  have hV1 : V1 = 90 / Real.pi := sorry
  have hV2 : V2 = 150 / Real.pi := sorry
  sorry

end cylinder_volume_ratio_l74_74131


namespace smallest_value_of_y_square_l74_74612

-- Let's define the conditions
variable (EF GH y : ℝ)

-- The given conditions of the problem
def is_isosceles_trapezoid (EF GH y : ℝ) : Prop :=
  EF = 100 ∧ GH = 25 ∧ y > 0

def has_tangent_circle (EF GH y : ℝ) : Prop :=
  is_isosceles_trapezoid EF GH y ∧ 
  ∃ P : ℝ, P = EF / 2

-- Main proof statement
theorem smallest_value_of_y_square (EF GH y : ℝ)
  (h1 : is_isosceles_trapezoid EF GH y)
  (h2 : has_tangent_circle EF GH y) :
  y^2 = 1875 :=
  sorry

end smallest_value_of_y_square_l74_74612


namespace cookies_in_jar_l74_74655

theorem cookies_in_jar (C : ℕ) (h : C - 1 = (C + 5) / 2) : C = 7 :=
by
  -- Proof goes here
  sorry

end cookies_in_jar_l74_74655


namespace exists_mn_coprime_l74_74350

theorem exists_mn_coprime (a b : ℤ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_gcd : Int.gcd a b = 1) :
  ∃ (m n : ℕ), 1 ≤ m ∧ 1 ≤ n ∧ (a^m + b^n) % (a * b) = 1 % (a * b) :=
sorry

end exists_mn_coprime_l74_74350


namespace positive_integer_solutions_inequality_l74_74639

theorem positive_integer_solutions_inequality :
  {x : ℕ | 2 * x + 9 ≥ 3 * (x + 2)} = {1, 2, 3} :=
by
  sorry

end positive_integer_solutions_inequality_l74_74639


namespace ratio_flowers_l74_74146

theorem ratio_flowers (flowers_monday flowers_tuesday flowers_week total_flowers flowers_friday : ℕ)
    (h_monday : flowers_monday = 4)
    (h_tuesday : flowers_tuesday = 8)
    (h_total : total_flowers = 20)
    (h_week : total_flowers = flowers_monday + flowers_tuesday + flowers_friday) :
    flowers_friday / flowers_monday = 2 :=
by
  sorry

end ratio_flowers_l74_74146


namespace arithmetic_mean_reciprocals_primes_l74_74869

theorem arithmetic_mean_reciprocals_primes : 
  let p := [2, 3, 5, 7] in 
  let reciprocals := p.map (λ n => 1 / (n : ℚ)) in
  (reciprocals.sum / reciprocals.length) = (247 / 840 : ℚ) :=
by
  sorry

end arithmetic_mean_reciprocals_primes_l74_74869


namespace emily_final_lives_l74_74887

/-- Initial number of lives Emily had. --/
def initialLives : ℕ := 42

/-- Number of lives Emily lost in the hard part of the game. --/
def livesLost : ℕ := 25

/-- Number of lives Emily gained in the next level. --/
def livesGained : ℕ := 24

/-- Final number of lives Emily should have after the changes. --/
def finalLives : ℕ := (initialLives - livesLost) + livesGained

theorem emily_final_lives : finalLives = 41 := by
  /-
  Proof is omitted as per instructions.
  Prove that the final number of lives Emily has is 41.
  -/
  sorry

end emily_final_lives_l74_74887


namespace greatest_difference_is_124_l74_74676

-- Define the variables a, b, c, and x
variables (a b c x : ℕ)

-- Define the conditions of the problem
def conditions (a b c : ℕ) := 
  (4 * a = 2 * b) ∧ 
  (4 * a = c) ∧ 
  (a > 0) ∧ 
  (a < 10) ∧ 
  (b < 10) ∧ 
  (c < 10)

-- Define the value of a number given its digits
def number (a b c : ℕ) := 100 * a + 10 * b + c

-- Define the maximum and minimum values of x
def max_val (a : ℕ) := number a (2 * a) (4 * a)
def min_val (a : ℕ) := number a (2 * a) (4 * a)

-- Define the greatest difference
def greatest_difference := max_val 2 - min_val 1

-- Prove that the greatest difference is 124
theorem greatest_difference_is_124 : greatest_difference = 124 :=
by 
  unfold greatest_difference 
  unfold max_val 
  unfold min_val 
  unfold number 
  sorry

end greatest_difference_is_124_l74_74676


namespace number_of_true_propositions_l74_74003

-- Definitions based on conditions
def prop1 (x : ℝ) : Prop := x^2 - x + 1 > 0
def prop2 (x : ℝ) : Prop := x^2 + x - 6 < 0 → x ≤ 2
def prop3 (x : ℝ) : Prop := (x^2 - 5*x + 6 = 0) → x = 2

-- Main theorem
theorem number_of_true_propositions : 
  (∀ x : ℝ, prop1 x) ∧ (∀ x : ℝ, prop2 x) ∧ (∃ x : ℝ, ¬ prop3 x) → 
  2 = 2 :=
by sorry

end number_of_true_propositions_l74_74003


namespace smallest_number_l74_74004

theorem smallest_number (a b c d : ℤ) (h1 : a = -2) (h2 : b = 0) (h3 : c = -3) (h4 : d = 1) : 
  min (min a b) (min c d) = c :=
by
  -- Proof goes here
  sorry

end smallest_number_l74_74004


namespace last_four_digits_of_5_pow_2013_l74_74947

theorem last_four_digits_of_5_pow_2013 : (5 ^ 2013) % 10000 = 3125 :=
by
  sorry

end last_four_digits_of_5_pow_2013_l74_74947


namespace watch_cost_l74_74412

-- Definitions based on conditions
def initial_money : ℤ := 1
def money_from_david : ℤ := 12
def money_needed : ℤ := 7

-- Indicating the total money Evan has after receiving money from David
def total_money := initial_money + money_from_david

-- The cost of the watch based on total money Evan has and additional money needed
def cost_of_watch := total_money + money_needed

-- Proving the cost of the watch
theorem watch_cost : cost_of_watch = 20 := by
  -- We are skipping the proof steps here
  sorry

end watch_cost_l74_74412


namespace solve_system_of_equations_l74_74632

theorem solve_system_of_equations : ∃ (x y : ℝ), 4 * x + y = 6 ∧ 3 * x - y = 1 ∧ x = 1 ∧ y = 2 :=
by
  existsi (1 : ℝ)
  existsi (2 : ℝ)
  sorry

end solve_system_of_equations_l74_74632


namespace qin_jiushao_value_l74_74242

def polynomial (x : ℤ) : ℤ :=
  2 * x^5 + 5 * x^4 + 8 * x^3 + 7 * x^2 - 6 * x + 11

def step1 (x : ℤ) : ℤ := 2 * x + 5
def step2 (x : ℤ) (v : ℤ) : ℤ := v * x + 8
def step3 (x : ℤ) (v : ℤ) : ℤ := v * x + 7
def step_v3 (x : ℤ) (v : ℤ) : ℤ := v * x - 6

theorem qin_jiushao_value (x : ℤ) (v3 : ℤ) (h1 : x = 3) (h2 : v3 = 130) :
  step_v3 3 (step3 3 (step2 3 (step1 3))) = v3 :=
by {
  sorry
}

end qin_jiushao_value_l74_74242


namespace jerry_total_games_l74_74050

-- Conditions
def initial_games : ℕ := 7
def birthday_games : ℕ := 2

-- Statement
theorem jerry_total_games : initial_games + birthday_games = 9 := by sorry

end jerry_total_games_l74_74050


namespace total_shaded_area_l74_74853

theorem total_shaded_area (S T : ℝ) (h1 : 12 / S = 4) (h2 : S / T = 4) : 
  S^2 + 12 * T^2 = 15.75 :=
by 
  sorry

end total_shaded_area_l74_74853


namespace distance_to_x_axis_P_l74_74044

-- The coordinates of point P
def P : ℝ × ℝ := (3, -2)

-- The distance from point P to the x-axis
def distance_to_x_axis (point : ℝ × ℝ) : ℝ :=
  abs (point.snd)

theorem distance_to_x_axis_P : distance_to_x_axis P = 2 :=
by
  -- Use the provided point P and calculate the distance
  sorry

end distance_to_x_axis_P_l74_74044


namespace coin_problem_l74_74500

noncomputable def P_flip_heads_is_heads (coin : ℕ → bool → Prop) : ℚ :=
if coin 2 tt then 2 / 3 else 0

theorem coin_problem (coin : ℕ → bool → Prop)
  (h1 : ∀ coin_num, coin coin_num tt = (coin_num = 1 ∨ coin_num = 2))
  (h2 : ∀ coin_num, coin coin_num ff = (coin_num = 1 ∨ coin_num = 3)):
  P_flip_heads_is_heads coin = 2 / 3 :=
by
  sorry

end coin_problem_l74_74500


namespace brady_work_hours_l74_74149

theorem brady_work_hours (A : ℕ) :
    (A * 30 + 5 * 30 + 8 * 30 = 3 * 190) → 
    A = 6 :=
by sorry

end brady_work_hours_l74_74149


namespace final_fish_stock_l74_74070

def initial_stock : ℤ := 200 
def sold_fish : ℤ := 50 
def fraction_spoiled : ℚ := 1/3 
def new_stock : ℤ := 200 

theorem final_fish_stock : 
    initial_stock - sold_fish - (fraction_spoiled * (initial_stock - sold_fish)) + new_stock = 300 := 
by 
  sorry

end final_fish_stock_l74_74070


namespace tom_already_has_4_pounds_of_noodles_l74_74971

-- Define the conditions
def beef : ℕ := 10
def noodle_multiplier : ℕ := 2
def packages : ℕ := 8
def weight_per_package : ℕ := 2

-- Define the total noodles needed
def total_noodles_needed : ℕ := noodle_multiplier * beef

-- Define the total noodles bought
def total_noodles_bought : ℕ := packages * weight_per_package

-- Define the already owned noodles
def already_owned_noodles : ℕ := total_noodles_needed - total_noodles_bought

-- State the theorem to prove
theorem tom_already_has_4_pounds_of_noodles :
  already_owned_noodles = 4 :=
  sorry

end tom_already_has_4_pounds_of_noodles_l74_74971


namespace total_spent_l74_74198

theorem total_spent (puppy_cost dog_food_cost treats_cost_per_bag toys_cost crate_cost bed_cost collar_leash_cost bags_of_treats discount_rate : ℝ) :
  puppy_cost = 20 →
  dog_food_cost = 20 →
  treats_cost_per_bag = 2.5 →
  toys_cost = 15 →
  crate_cost = 20 →
  bed_cost = 20 →
  collar_leash_cost = 15 →
  bags_of_treats = 2 →
  discount_rate = 0.2 →
  (dog_food_cost + treats_cost_per_bag * bags_of_treats + toys_cost + crate_cost + bed_cost + collar_leash_cost) * (1 - discount_rate) + puppy_cost = 96 :=
by sorry

end total_spent_l74_74198


namespace quadratic_nonneg_range_l74_74598

theorem quadratic_nonneg_range (a : ℝ) :
  (¬ ∃ x : ℝ, x^2 + (a - 1) * x + 1 < 0) ↔ (-1 ≤ a ∧ a ≤ 3) :=
by
  sorry

end quadratic_nonneg_range_l74_74598


namespace combined_weight_after_removal_l74_74523

theorem combined_weight_after_removal (weight_sugar weight_salt weight_removed : ℕ) 
                                       (h_sugar : weight_sugar = 16)
                                       (h_salt : weight_salt = 30)
                                       (h_removed : weight_removed = 4) : 
                                       (weight_sugar + weight_salt) - weight_removed = 42 :=
by {
  sorry
}

end combined_weight_after_removal_l74_74523


namespace no_nat_solutions_l74_74477

theorem no_nat_solutions (x y z : ℕ) : x^2 + y^2 + z^2 ≠ 2 * x * y * z :=
sorry

end no_nat_solutions_l74_74477


namespace distance_light_travels_500_years_l74_74635

-- Define the given conditions
def distance_in_one_year_miles : ℝ := 5.87e12
def years_traveling : ℝ := 500
def miles_to_kilometers : ℝ := 1.60934

-- Define the expected distance in kilometers after 500 years
def expected_distance_in_kilometers : ℝ  := 4.723e15

-- State the theorem: the distance light travels in 500 years in kilometers
theorem distance_light_travels_500_years :
  (distance_in_one_year_miles * years_traveling * miles_to_kilometers) 
    = expected_distance_in_kilometers := 
by
  sorry

end distance_light_travels_500_years_l74_74635


namespace evaluate_floor_of_negative_seven_halves_l74_74715

def floor_of_negative_seven_halves : ℤ :=
  Int.floor (-7 / 2)

theorem evaluate_floor_of_negative_seven_halves :
  floor_of_negative_seven_halves = -4 :=
by
  sorry

end evaluate_floor_of_negative_seven_halves_l74_74715


namespace min_quality_inspection_machines_l74_74965

theorem min_quality_inspection_machines (z x : ℕ) :
  (z + 30 * x) / 30 = 1 →
  (z + 10 * x) / 10 = 2 →
  (z + 5 * x) / 5 ≥ 4 :=
by
  intros h1 h2
  sorry

end min_quality_inspection_machines_l74_74965


namespace parabola_focus_distance_l74_74906

open Real (sqrt)

theorem parabola_focus_distance {x y : ℝ} (h : y^2 = 8 * x) (hx : x + 2 = 4 ∨ x + 2 = 4) : 
  (sqrt (x^2 + y^2) = 2 * sqrt 5) :=
begin
  sorry
end

end parabola_focus_distance_l74_74906


namespace find_xyz_l74_74341

open Complex

-- Definitions of the variables and conditions
variables {a b c x y z : ℂ} (h_a_ne_zero : a ≠ 0) (h_b_ne_zero : b ≠ 0) (h_c_ne_zero : c ≠ 0)
  (h_x_ne_zero : x ≠ 0) (h_y_ne_zero : y ≠ 0) (h_z_ne_zero : z ≠ 0)
  (h1 : a = (b - c) * (x + 2))
  (h2 : b = (a - c) * (y + 2))
  (h3 : c = (a - b) * (z + 2))
  (h4 : x * y + x * z + y * z = 12)
  (h5 : x + y + z = 6)

-- Statement of the theorem
theorem find_xyz : x * y * z = 7 := 
by
  -- Proof steps to be filled in
  sorry

end find_xyz_l74_74341


namespace notification_probability_l74_74271

theorem notification_probability
  (num_students : ℕ)
  (num_notified_Li : ℕ)
  (num_notified_Zhang : ℕ)
  (prob_Li : ℚ)
  (prob_Zhang : ℚ)
  (h1 : num_students = 10)
  (h2 : num_notified_Li = 4)
  (h3 : num_notified_Zhang = 4)
  (h4 : prob_Li = (4 : ℚ) / 10)
  (h5 : prob_Zhang = (4 : ℚ) / 10) :
  prob_Li + prob_Zhang - prob_Li * prob_Zhang = (16 : ℚ) / 25 := 
by 
  sorry

end notification_probability_l74_74271


namespace area_of_shaded_region_l74_74327

noncomputable def r2 : ℝ := Real.sqrt 20
noncomputable def r1 : ℝ := 3 * r2

theorem area_of_shaded_region :
  let area := π * (r1 ^ 2) - π * (r2 ^ 2)
  area = 160 * π :=
by
  sorry

end area_of_shaded_region_l74_74327


namespace whale_tongue_weight_difference_l74_74637

noncomputable def tongue_weight_blue_whale_kg : ℝ := 2700
noncomputable def tongue_weight_fin_whale_kg : ℝ := 1800
noncomputable def kg_to_pounds : ℝ := 2.20462
noncomputable def ton_to_pounds : ℝ := 2000

noncomputable def tongue_weight_blue_whale_tons := (tongue_weight_blue_whale_kg * kg_to_pounds) / ton_to_pounds
noncomputable def tongue_weight_fin_whale_tons := (tongue_weight_fin_whale_kg * kg_to_pounds) / ton_to_pounds
noncomputable def weight_difference_tons := tongue_weight_blue_whale_tons - tongue_weight_fin_whale_tons

theorem whale_tongue_weight_difference :
  weight_difference_tons = 0.992079 :=
by
  sorry

end whale_tongue_weight_difference_l74_74637


namespace percentage_answered_first_correctly_l74_74845

variable (A B C D : ℝ)

-- Conditions translated to Lean
variable (hB : B = 0.65)
variable (hC : C = 0.20)
variable (hD : D = 0.60)

-- Statement to prove
theorem percentage_answered_first_correctly (hI : A + B - D = 1 - C) : A = 0.75 := by
  -- import conditions
  rw [hB, hC, hD] at hI
  -- solve the equation
  sorry

end percentage_answered_first_correctly_l74_74845


namespace unique_k_exists_l74_74583

theorem unique_k_exists (k : ℕ) (a b : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) :
  (a^2 + b^2 = k * a * b) ↔ k = 2 := sorry

end unique_k_exists_l74_74583


namespace find_a_for_quadratic_roots_l74_74358

theorem find_a_for_quadratic_roots :
  ∀ (a x₁ x₂ : ℝ), 
    (x₁ ≠ x₂) →
    (x₁ * x₁ + a * x₁ + 6 = 0) →
    (x₂ * x₂ + a * x₂ + 6 = 0) →
    (x₁ - (72 / (25 * x₂^3)) = x₂ - (72 / (25 * x₁^3))) →
    (a = 9 ∨ a = -9) :=
by
  sorry

end find_a_for_quadratic_roots_l74_74358


namespace race_head_start_l74_74836

-- This statement defines the problem in Lean 4
theorem race_head_start (Va Vb L H : ℝ) 
(h₀ : Va = 51 / 44 * Vb) 
(h₁ : L / Va = (L - H) / Vb) : 
H = 7 / 51 * L := 
sorry

end race_head_start_l74_74836


namespace simplify_to_ap_minus_b_l74_74216

noncomputable def simplify_expression (p : ℝ) : ℝ :=
  ((7*p + 3) - 3*p * 2) * 4 + (5 - 2 / 4) * (8*p - 12)

theorem simplify_to_ap_minus_b (p : ℝ) :
  simplify_expression p = 40 * p - 42 :=
by
  -- Proof steps would go here
  sorry

end simplify_to_ap_minus_b_l74_74216


namespace frank_sales_quota_l74_74898

theorem frank_sales_quota (x : ℕ) :
  (3 * x + 12 + 23 = 50) → x = 5 :=
by sorry

end frank_sales_quota_l74_74898


namespace compound_interest_l74_74765

theorem compound_interest (SI : ℝ) (R : ℝ) (T : ℝ) (P : ℝ) (CI : ℝ) :
  SI = 40 → R = 5 → T = 2 → SI = (P * R * T) / 100 → CI = P * ((1 + R / 100) ^ T - 1) → CI = 41 :=
by sorry

end compound_interest_l74_74765


namespace sequence_a_general_term_sequence_b_sum_of_first_n_terms_l74_74312

variable {n : ℕ}

def a (n : ℕ) : ℕ := 2 * n

def b (n : ℕ) : ℕ := 3^(n-1) + 2 * n

def T (n : ℕ) : ℕ := (3^n - 1) / 2 + n^2 + n

theorem sequence_a_general_term :
  (∀ n, a n = 2 * n) :=
by
  intro n
  sorry

theorem sequence_b_sum_of_first_n_terms :
  (∀ n, T n = (3^n - 1) / 2 + n^2 + n) :=
by
  intro n
  sorry

end sequence_a_general_term_sequence_b_sum_of_first_n_terms_l74_74312


namespace solution_set_l74_74232

theorem solution_set (x : ℝ) : 
  (-2 * x ≤ 6) ∧ (x + 1 < 0) ↔ (-3 ≤ x) ∧ (x < -1) := by
  sorry

end solution_set_l74_74232


namespace sum_adjacent_odd_l74_74124

/-
  Given 2020 natural numbers written in a circle, prove that the sum of any two adjacent numbers is odd.
-/

noncomputable def numbers_in_circle : Fin 2020 → ℕ := sorry

theorem sum_adjacent_odd (k : Fin 2020) :
  (numbers_in_circle k + numbers_in_circle (k + 1)) % 2 = 1 :=
sorry

end sum_adjacent_odd_l74_74124


namespace pages_allocation_correct_l74_74921

-- Define times per page for Alice, Bob, and Chandra
def t_A := 40
def t_B := 60
def t_C := 48

-- Define pages read by Alice, Bob, and Chandra
def pages_A := 295
def pages_B := 197
def pages_C := 420

-- Total pages in the novel
def total_pages := 912

-- Calculate the total time each one spends reading
def total_time_A := t_A * pages_A
def total_time_B := t_B * pages_B
def total_time_C := t_C * pages_C

-- Theorem: Prove the correct allocation of pages
theorem pages_allocation_correct : 
  total_pages = pages_A + pages_B + pages_C ∧
  total_time_A = total_time_B ∧
  total_time_B = total_time_C :=
by 
  -- Place end of proof here 
  sorry

end pages_allocation_correct_l74_74921


namespace find_x_l74_74446

theorem find_x (x : ℚ) (h : (3 * x - 7) / 4 = 15) : x = 67 / 3 :=
sorry

end find_x_l74_74446


namespace ralph_socks_problem_l74_74952

theorem ralph_socks_problem :
  ∃ x y z : ℕ, x + y + z = 10 ∧ x + 2 * y + 4 * z = 30 ∧ 1 ≤ x ∧ 1 ≤ y ∧ 1 ≤ z ∧ x = 2 :=
by
  sorry

end ralph_socks_problem_l74_74952


namespace total_amount_paid_correct_l74_74698

-- Definitions of wholesale costs, retail markups, and employee discounts
def wholesale_cost_video_recorder : ℝ := 200
def retail_markup_video_recorder : ℝ := 0.20
def employee_discount_video_recorder : ℝ := 0.30

def wholesale_cost_digital_camera : ℝ := 150
def retail_markup_digital_camera : ℝ := 0.25
def employee_discount_digital_camera : ℝ := 0.20

def wholesale_cost_smart_tv : ℝ := 800
def retail_markup_smart_tv : ℝ := 0.15
def employee_discount_smart_tv : ℝ := 0.25

-- Calculation of retail prices
def retail_price (wholesale_cost : ℝ) (markup : ℝ) : ℝ :=
  wholesale_cost * (1 + markup)

-- Calculation of employee prices
def employee_price (retail_price : ℝ) (discount : ℝ) : ℝ :=
  retail_price * (1 - discount)

-- Retail prices
def retail_price_video_recorder := retail_price wholesale_cost_video_recorder retail_markup_video_recorder
def retail_price_digital_camera := retail_price wholesale_cost_digital_camera retail_markup_digital_camera
def retail_price_smart_tv := retail_price wholesale_cost_smart_tv retail_markup_smart_tv

-- Employee prices
def employee_price_video_recorder := employee_price retail_price_video_recorder employee_discount_video_recorder
def employee_price_digital_camera := employee_price retail_price_digital_camera employee_discount_digital_camera
def employee_price_smart_tv := employee_price retail_price_smart_tv employee_discount_smart_tv

-- Total amount paid by the employee
def total_amount_paid := 
  employee_price_video_recorder 
  + employee_price_digital_camera 
  + employee_price_smart_tv

theorem total_amount_paid_correct :
  total_amount_paid = 1008 := 
  by 
    sorry

end total_amount_paid_correct_l74_74698


namespace loan_payment_difference_l74_74276

noncomputable def compounded_amount (P : ℝ) (r : ℝ) (n : ℝ) (t : ℝ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

noncomputable def simple_interest_amount (P : ℝ) (r : ℝ) (t : ℝ) : ℝ :=
  P + P * r * t

noncomputable def loan1_payment (P : ℝ) (r : ℝ) (n : ℝ) (t1 : ℝ) (t2 : ℝ) : ℝ :=
  let A1 := compounded_amount P r n t1
  let one_third_payment := A1 / 3
  let remaining := A1 - one_third_payment
  one_third_payment + compounded_amount remaining r n t2

noncomputable def loan2_payment (P : ℝ) (r : ℝ) (t : ℝ) : ℝ :=
  simple_interest_amount P r t

noncomputable def positive_difference (x y : ℝ) : ℝ :=
  if x > y then x - y else y - x

theorem loan_payment_difference: 
  ∀ P : ℝ, ∀ r1 r2 : ℝ, ∀ n : ℝ, ∀ t1 t2 : ℝ,
  P = 12000 → r1 = 0.08 → r2 = 0.09 → n = 12 → t1 = 7 → t2 = 8 →
  positive_difference 
    (loan2_payment P r2 (t1 + t2)) 
    (loan1_payment P r1 n t1 t2) = 2335 := 
by
  intros
  sorry

end loan_payment_difference_l74_74276


namespace value_of_e_is_91_l74_74678

noncomputable def value_of_e (a b c d e : ℤ) (k : ℤ) : Prop :=
  a % 2 = 1 ∧ b % 2 = 1 ∧ c % 2 = 1 ∧ d % 2 = 1 ∧ e % 2 = 1 ∧
  b = a + 2 * k ∧ c = a + 4 * k ∧ d = a + 6 * k ∧ e = a + 8 * k ∧
  a + c = 146 ∧ k > 0 ∧ 2 * k ≥ 4 ∧ k ≠ 2

theorem value_of_e_is_91 (a b c d e k : ℤ) (h : value_of_e a b c d e k) : e = 91 :=
  sorry

end value_of_e_is_91_l74_74678


namespace ellipse_foci_y_axis_range_l74_74172

theorem ellipse_foci_y_axis_range (k : ℝ) : 
  (2*k - 1 > 2 - k) → (2 - k > 0) → (1 < k ∧ k < 2) := 
by 
  intros h1 h2
  -- We use the assumptions to derive the target statement.
  sorry

end ellipse_foci_y_axis_range_l74_74172


namespace quadratic_distinct_roots_l74_74764

theorem quadratic_distinct_roots (m : ℝ) : 
  ((m - 2) * x ^ 2 + 2 * x + 1 = 0) → (m < 3 ∧ m ≠ 2) :=
by
  sorry

end quadratic_distinct_roots_l74_74764


namespace smallest_set_of_circular_handshakes_l74_74265

def circular_handshake_smallest_set (n : ℕ) : ℕ :=
  if h : n % 2 = 0 then n / 2 else (n / 2) + 1

theorem smallest_set_of_circular_handshakes :
  circular_handshake_smallest_set 36 = 18 :=
by
  sorry

end smallest_set_of_circular_handshakes_l74_74265


namespace smallest_b_value_l74_74055

variable {a b c d : ℝ}

-- Definitions based on conditions
def is_arithmetic_series (a b c : ℝ) (d : ℝ) : Prop :=
  a = b - d ∧ c = b + d

def abc_product (a b c : ℝ) : Prop :=
  a * b * c = 216

theorem smallest_b_value (a b c d : ℝ) 
  (pos_a : a > 0) (pos_b : b > 0) (pos_c : c > 0)
  (arith_series : is_arithmetic_series a b c d)
  (abc_216 : abc_product a b c) : 
  b ≥ 6 :=
by
  sorry

end smallest_b_value_l74_74055


namespace tan_proof_l74_74021

noncomputable def prove_tan_relation (α β : ℝ) : Prop :=
  2 * (Real.tan α) = 3 * (Real.tan β)

theorem tan_proof (α β : ℝ) (h : Real.tan (α - β) = (Real.sin (2*β)) / (5 - Real.cos (2*β))) : 
  prove_tan_relation α β :=
sorry

end tan_proof_l74_74021


namespace benny_seashells_l74_74148

-- Defining the conditions
def initial_seashells : ℕ := 66
def given_away_seashells : ℕ := 52

-- Statement of the proof problem
theorem benny_seashells : (initial_seashells - given_away_seashells) = 14 :=
by
  sorry

end benny_seashells_l74_74148


namespace value_of_number_l74_74125

theorem value_of_number (x : ℤ) (number : ℚ) (h₁ : x = 32) (h₂ : 35 - (23 - (15 - x)) = 12 * number / (1/2)) : number = -5/6 :=
by
  sorry

end value_of_number_l74_74125


namespace symmetric_points_l74_74475

theorem symmetric_points (a b : ℝ) (h1 : 2 * a + 1 = -1) (h2 : 4 = -(3 * b - 1)) :
  2 * a + b = -3 := 
sorry

end symmetric_points_l74_74475


namespace quadratic_equation_unique_solution_l74_74555

theorem quadratic_equation_unique_solution (p : ℚ) :
  (∃ x : ℚ, 3 * x^2 - 7 * x + p = 0) ∧ 
  ∀ y : ℚ, 3 * y^2 -7 * y + p ≠ 0 → ∀ z : ℚ, 3 * z^2 - 7 * z + p = 0 → y = z ↔ 
  p = 49 / 12 :=
by
  sorry

end quadratic_equation_unique_solution_l74_74555


namespace difference_between_numbers_l74_74648

theorem difference_between_numbers (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 190) : x - y = 19 :=
by sorry

end difference_between_numbers_l74_74648


namespace mark_owes_linda_l74_74058

-- Define the payment per room and the number of rooms painted
def payment_per_room := (13 : ℚ) / 3
def rooms_painted := (8 : ℚ) / 5

-- State the theorem and the proof
theorem mark_owes_linda : (payment_per_room * rooms_painted) = (104 : ℚ) / 15 := by
  sorry

end mark_owes_linda_l74_74058


namespace units_digit_150_factorial_is_zero_l74_74253

theorem units_digit_150_factorial_is_zero :
  (nat.trailing_digits (nat.factorial 150) 1) = 0 :=
by sorry

end units_digit_150_factorial_is_zero_l74_74253


namespace effect_of_dimension_changes_on_area_l74_74679

variable {L B : ℝ}  -- Original length and breadth

def original_area (L B : ℝ) : ℝ := L * B

def new_length (L : ℝ) : ℝ := 1.15 * L

def new_breadth (B : ℝ) : ℝ := 0.90 * B

def new_area (L B : ℝ) : ℝ := new_length L * new_breadth B

theorem effect_of_dimension_changes_on_area (L B : ℝ) :
  new_area L B = 1.035 * original_area L B :=
by
  sorry

end effect_of_dimension_changes_on_area_l74_74679


namespace geometric_sequence_property_l74_74309

noncomputable def S_n (n : ℕ) (a_n : ℕ → ℕ) : ℕ := 3 * 2^n - 3

noncomputable def a_n (n : ℕ) : ℕ := 3 * 2^(n-1)

noncomputable def b_n (n : ℕ) : ℕ := 2^(n-1)

noncomputable def T_n (n : ℕ) : ℕ := 2^n - 1

theorem geometric_sequence_property (n : ℕ) (hn : n ≥ 0) :
  T_n n < b_n (n+1) :=
by
  sorry

end geometric_sequence_property_l74_74309


namespace intersection_is_14_l74_74436

open Set

def A : Set ℕ := {1, 2, 3, 4}
def B : Set ℕ := {y | ∃ x ∈ A, y = 3 * x - 2}

theorem intersection_is_14 : A ∩ B = {1, 4} := 
by sorry

end intersection_is_14_l74_74436


namespace inequality_solution_set_quadratic_inequality_range_l74_74386

theorem inequality_solution_set (x : ℝ) :
  (9 / (x + 4) ≤ 2) ↔ (x < -4) ∨ (x ≥ 1 / 2) := 
sorry

theorem quadratic_inequality_range (k : ℝ) :
  (∀ x : ℝ, x^2 - 2*x + k^2 - 1 > 0) ↔ (k > Real.sqrt 2 ∨ k < -Real.sqrt 2) :=
sorry

end inequality_solution_set_quadratic_inequality_range_l74_74386


namespace xy_relationship_l74_74185

theorem xy_relationship (x y : ℤ) (h1 : 2 * x - y > x + 1) (h2 : x + 2 * y < 2 * y - 3) :
  x < -3 ∧ y < -4 ∧ x > y + 1 :=
sorry

end xy_relationship_l74_74185


namespace Xiao_Ming_vertical_height_increase_l74_74833

noncomputable def vertical_height_increase (slope_ratio_v slope_ratio_h : ℝ) (distance : ℝ) : ℝ :=
  let x := distance / (Real.sqrt (1 + (slope_ratio_h / slope_ratio_v)^2))
  x

theorem Xiao_Ming_vertical_height_increase
  (slope_ratio_v slope_ratio_h distance : ℝ)
  (h_ratio : slope_ratio_v = 1)
  (h_ratio2 : slope_ratio_h = 2.4)
  (h_distance : distance = 130) :
  vertical_height_increase slope_ratio_v slope_ratio_h distance = 50 :=
by
  unfold vertical_height_increase
  rw [h_ratio, h_ratio2, h_distance]
  sorry

end Xiao_Ming_vertical_height_increase_l74_74833


namespace pascal_triangle_ratio_l74_74042

theorem pascal_triangle_ratio (n r : ℕ) (hn1 : 5 * r = 2 * n - 3) (hn2 : 7 * r = 3 * n - 11) : n = 34 :=
by
  -- The proof steps will fill here eventually
  sorry

end pascal_triangle_ratio_l74_74042


namespace square_binomial_l74_74826

theorem square_binomial (x : ℝ) : (-x - 1) ^ 2 = x^2 + 2 * x + 1 :=
by
  sorry

end square_binomial_l74_74826


namespace islands_not_connected_by_bridges_for_infinitely_many_primes_l74_74421

open Nat

theorem islands_not_connected_by_bridges_for_infinitely_many_primes :
  ∃ᶠ p in at_top, ∃ n m : ℕ, n ≠ m ∧ ¬(p ∣ (n^2 - m + 1) * (m^2 - n + 1)) :=
sorry

end islands_not_connected_by_bridges_for_infinitely_many_primes_l74_74421


namespace intersection_of_A_and_B_l74_74339

def A : Set ℝ := {x | x^2 - 4*x + 3 < 0}
def B : Set ℝ := {x | x^2 - 6*x + 8 < 0}

theorem intersection_of_A_and_B : (A ∩ B) = {x | 2 < x ∧ x < 3} := by
  sorry

end intersection_of_A_and_B_l74_74339


namespace rectangle_width_length_ratio_l74_74923

theorem rectangle_width_length_ratio (w : ℕ) (h : ℕ) (P : ℕ) (H1 : h = 10) (H2 : P = 30) (H3 : 2 * w + 2 * h = P) :
  w / h = 1 / 2 :=
by
  sorry

end rectangle_width_length_ratio_l74_74923


namespace circle_inside_triangle_l74_74625

-- Define the problem conditions
def triangle_sides : ℕ × ℕ × ℕ := (3, 4, 5)
def circle_area : ℚ := 25 / 8

-- Define the problem statement
theorem circle_inside_triangle (a b c : ℕ) (area : ℚ)
    (h1 : (a, b, c) = triangle_sides)
    (h2 : area = circle_area) :
    ∃ r R : ℚ, R < r ∧ 2 * r = a + b - c ∧ R^2 = area / π := sorry

end circle_inside_triangle_l74_74625


namespace gcf_72_108_l74_74105

theorem gcf_72_108 : Nat.gcd 72 108 = 36 := by
  sorry

end gcf_72_108_l74_74105


namespace regression_line_passes_through_sample_mean_point_l74_74811

theorem regression_line_passes_through_sample_mean_point
  (a b : ℝ) (x y : ℝ)
  (hx : x = a + b*x) :
  y = a + b*x :=
by sorry

end regression_line_passes_through_sample_mean_point_l74_74811


namespace sin_60_eq_sqrt3_div_2_l74_74519

-- Problem statement translated to Lean
theorem sin_60_eq_sqrt3_div_2 : Real.sin (Real.pi / 3) = Real.sqrt 3 / 2 := 
by
  sorry

end sin_60_eq_sqrt3_div_2_l74_74519


namespace change_in_opinion_difference_l74_74006

theorem change_in_opinion_difference :
  let initially_liked_pct := 0.4;
  let initially_disliked_pct := 0.6;
  let finally_liked_pct := 0.8;
  let finally_disliked_pct := 0.2;
  let max_change := finally_liked_pct + (initially_disliked_pct - finally_disliked_pct);
  let min_change := finally_liked_pct - initially_liked_pct;
  max_change - min_change = 0.2 :=
by
  sorry

end change_in_opinion_difference_l74_74006


namespace units_digit_of_150_factorial_is_zero_l74_74245

theorem units_digit_of_150_factorial_is_zero : (Nat.factorial 150) % 10 = 0 := by
sorry

end units_digit_of_150_factorial_is_zero_l74_74245


namespace ratio_of_ages_in_two_years_l74_74138

theorem ratio_of_ages_in_two_years
  (S M : ℕ) 
  (h1 : M = S + 22)
  (h2 : S = 20)
  (h3 : ∃ k : ℕ, M + 2 = k * (S + 2)) :
  (M + 2) / (S + 2) = 2 :=
by
  sorry

end ratio_of_ages_in_two_years_l74_74138


namespace Ann_is_16_l74_74700

variable (A S : ℕ)

theorem Ann_is_16
  (h1 : A = S + 5)
  (h2 : A + S = 27) :
  A = 16 :=
by
  sorry

end Ann_is_16_l74_74700


namespace true_statements_l74_74985

theorem true_statements :
  (5 ∣ 25) ∧ (19 ∣ 209 ∧ ¬ (19 ∣ 63)) ∧ (30 ∣ 90) ∧ (14 ∣ 28 ∧ 14 ∣ 56) ∧ (9 ∣ 180) :=
by
  have A : 5 ∣ 25 := sorry
  have B1 : 19 ∣ 209 := sorry
  have B2 : ¬ (19 ∣ 63) := sorry
  have C : 30 ∣ 90 := sorry
  have D1 : 14 ∣ 28 := sorry
  have D2 : 14 ∣ 56 := sorry
  have E : 9 ∣ 180 := sorry
  exact ⟨A, ⟨B1, B2⟩, C, ⟨D1, D2⟩, E⟩

end true_statements_l74_74985


namespace marks_in_physics_l74_74000

-- Definitions of the variables
variables (P C M : ℕ)

-- Conditions
def condition1 : Prop := P + C + M = 210
def condition2 : Prop := P + M = 180
def condition3 : Prop := P + C = 140

-- The statement to prove
theorem marks_in_physics (h1 : condition1 P C M) (h2 : condition2 P M) (h3 : condition3 P C) : P = 110 :=
sorry

end marks_in_physics_l74_74000


namespace arithmetic_mean_of_reciprocals_of_first_four_primes_l74_74877

theorem arithmetic_mean_of_reciprocals_of_first_four_primes :
  let primes := [2, 3, 5, 7] in
  let reciprocals := primes.map (λ p => (1 : ℚ) / p) in
  let sum_reciprocals := reciprocals.sum in
  let mean := sum_reciprocals / (primes.length : ℚ) in
  mean = 247 / 840 :=
by
  let primes := [2, 3, 5, 7]
  let reciprocals := primes.map (λ p => (1 : ℚ) / p)
  let sum_reciprocals := reciprocals.sum
  let mean := sum_reciprocals / (primes.length : ℚ)
  sorry

end arithmetic_mean_of_reciprocals_of_first_four_primes_l74_74877


namespace trigonometric_identity_l74_74574

theorem trigonometric_identity
  (θ : ℝ)
  (h : (2 + (1 / (Real.sin θ) ^ 2)) / (1 + Real.sin θ) = 1) :
  (1 + Real.sin θ) * (2 + Real.cos θ) = 4 :=
sorry

end trigonometric_identity_l74_74574


namespace negation_of_universal_proposition_l74_74963

theorem negation_of_universal_proposition :
  (¬ (∀ x : ℝ, x^2 - 3 * x + 2 > 0)) ↔ (∃ x : ℝ, x^2 - 3 * x + 2 ≤ 0) :=
by
  sorry

end negation_of_universal_proposition_l74_74963


namespace discount_each_book_l74_74922

-- Definition of conditions
def original_price : ℝ := 5
def num_books : ℕ := 10
def total_paid : ℝ := 45

-- Theorem statement to prove the discount
theorem discount_each_book (d : ℝ) 
  (h1 : original_price * (num_books : ℝ) - d * (num_books : ℝ) = total_paid) : 
  d = 0.5 := 
sorry

end discount_each_book_l74_74922


namespace negation_of_universal_l74_74489

variable (P : ℝ → Prop)
def pos (x : ℝ) : Prop := x > 0
def gte_zero (x : ℝ) : Prop := x^2 - x ≥ 0
def lt_zero (x : ℝ) : Prop := x^2 - x < 0

theorem negation_of_universal :
  ¬ (∀ x, pos x → gte_zero x) ↔ ∃ x, pos x ∧ lt_zero x := by
  sorry

end negation_of_universal_l74_74489


namespace integer_solution_is_three_l74_74889

theorem integer_solution_is_three (n : ℤ) : (∃ k : ℤ, (n^3 - 3*n^2 + n + 2) = 5^k) ↔ n = 3 := 
by
  sorry

end integer_solution_is_three_l74_74889


namespace addition_of_decimals_l74_74001

theorem addition_of_decimals (a b : ℚ) (h1 : a = 7.56) (h2 : b = 4.29) : a + b = 11.85 :=
by
  -- The proof will be provided here
  sorry

end addition_of_decimals_l74_74001


namespace find_common_ratio_l74_74729

noncomputable def geom_series_common_ratio (q : ℝ) : Prop :=
  ∃ (a1 : ℝ), a1 > 0 ∧ (a1 * q^2 = 18) ∧ (a1 * (1 + q + q^2) = 26)

theorem find_common_ratio (q : ℝ) :
  geom_series_common_ratio q → q = 3 :=
sorry

end find_common_ratio_l74_74729


namespace find_b_c_d_l74_74550

def f (x : ℝ) := x^3 + 2 * x^2 + 3 * x + 4
def h (x : ℝ) := x^3 + 6 * x^2 - 8 * x + 16

theorem find_b_c_d :
  (∀ r : ℝ, f r = 0 → h (r^3) = 0) ∧ h (x : ℝ) = x^3 + 6 * x^2 - 8 * x + 16 :=
by 
  -- proof not required
  sorry

end find_b_c_d_l74_74550


namespace range_of_m_l74_74307

noncomputable def p (x : ℝ) : Prop := -2 ≤ x ∧ x ≤ 10
noncomputable def q (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 ≤ 0 ∧ m > 0

theorem range_of_m (m : ℝ) (h₀ : m > 0) (h₁ : ∀ x : ℝ, q x m → p x) : m ≥ 9 :=
sorry

end range_of_m_l74_74307


namespace candies_total_l74_74464

theorem candies_total (N a S : ℕ) (h1 : S = 2 * a + 7) (h2 : S = N * a) (h3 : a > 1) (h4 : N = 3) : S = 21 := 
sorry

end candies_total_l74_74464


namespace black_cars_count_l74_74645

theorem black_cars_count
    (r b : ℕ)
    (r_ratio : r = 33)
    (ratio_condition : r / b = 3 / 8) :
    b = 88 :=
by 
  sorry

end black_cars_count_l74_74645


namespace trueConverseB_l74_74377

noncomputable def conditionA : Prop :=
  ∀ (x y : ℝ), -- "Vertical angles are equal"
  sorry -- Placeholder for vertical angles equality

noncomputable def conditionB : Prop :=
  ∀ (l₁ l₂ : ℝ), -- "If the consecutive interior angles are supplementary, then the two lines are parallel."
  sorry -- Placeholder for supplementary angles imply parallel lines

noncomputable def conditionC : Prop :=
  ∀ (a b : ℝ), -- "If \(a = b\), then \(a^2 = b^2\)"
  a = b → a^2 = b^2

noncomputable def conditionD : Prop :=
  ∀ (a b : ℝ), -- "If \(a > 0\) and \(b > 0\), then \(a^2 + b^2 > 0\)"
  a > 0 ∧ b > 0 → a^2 + b^2 > 0

theorem trueConverseB (hB: conditionB) : -- Proposition (B) has a true converse
  ∀ (l₁ l₂ : ℝ), 
  (∃ (a1 a2 : ℝ), -- Placeholder for angles
  sorry) → (l₁ = l₂) := -- Placeholder for consecutive interior angles are supplementary
  sorry

end trueConverseB_l74_74377


namespace root_sum_value_l74_74343

theorem root_sum_value (r s t : ℝ) (h1: r + s + t = 24) (h2: r * s + s * t + t * r = 50) (h3: r * s * t = 24) :
  r / (1/r + s * t) + s / (1/s + t * r) + t / (1/t + r * s) = 19.04 :=
sorry

end root_sum_value_l74_74343


namespace perimeter_of_billboard_l74_74996
noncomputable def perimeter_billboard : ℝ :=
  let width := 8
  let area := 104
  let length := area / width
  let perimeter := 2 * (length + width)
  perimeter

theorem perimeter_of_billboard (width area : ℝ) (P : width = 8 ∧ area = 104) :
    perimeter_billboard = 42 :=
by
  sorry

end perimeter_of_billboard_l74_74996


namespace algebraic_comparison_l74_74548

theorem algebraic_comparison (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : 
  (a^2 / b + b^2 / a ≥ a + b) :=
by
  sorry

end algebraic_comparison_l74_74548


namespace Kaleb_second_half_points_l74_74046

theorem Kaleb_second_half_points (first_half_points total_points : ℕ) (h1 : first_half_points = 43) (h2 : total_points = 66) : total_points - first_half_points = 23 := by
  sorry

end Kaleb_second_half_points_l74_74046


namespace entrance_ticket_cost_l74_74521

theorem entrance_ticket_cost (num_students num_teachers ticket_cost : ℕ) (h1 : num_students = 20) (h2 : num_teachers = 3) (h3 : ticket_cost = 5) :
  (num_students + num_teachers) * ticket_cost = 115 :=
by
  rw [h1, h2, h3]
  norm_num
  sorry

end entrance_ticket_cost_l74_74521


namespace bags_weight_after_removal_l74_74524

theorem bags_weight_after_removal (sugar_weight salt_weight weight_removed : ℕ) (h1 : sugar_weight = 16) (h2 : salt_weight = 30) (h3 : weight_removed = 4) :
  sugar_weight + salt_weight - weight_removed = 42 := by
  sorry

end bags_weight_after_removal_l74_74524


namespace pilot_speed_outbound_l74_74533

theorem pilot_speed_outbound (v : ℝ) (d : ℝ) (s_return : ℝ) (t_total : ℝ) 
    (return_time : ℝ := d / s_return) 
    (outbound_time : ℝ := t_total - return_time) 
    (speed_outbound : ℝ := d / outbound_time) :
  d = 1500 → s_return = 500 → t_total = 8 → speed_outbound = 300 :=
by
  intros hd hs ht
  sorry

end pilot_speed_outbound_l74_74533


namespace shortest_chord_through_point_l74_74023

theorem shortest_chord_through_point 
  (P : ℝ × ℝ) (hx : P = (2, 1))
  (h_circle : ∀ x y : ℝ, (x - 1)^2 + y^2 = 4 → (x, y) ∈ {p : ℝ × ℝ | (p.fst - 1)^2 + p.snd^2 = 4}) :
  ∃ a b c : ℝ, a = 1 ∧ b = 1 ∧ c = -3 ∧ a * (P.1) + b * (P.2) + c = 0 := 
by
  -- proof skipped
  sorry

end shortest_chord_through_point_l74_74023


namespace economical_refuel_l74_74619

theorem economical_refuel (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x ≠ y) : 
  (x + y) / 2 > (2 * x * y) / (x + y) :=
sorry -- Proof omitted

end economical_refuel_l74_74619


namespace dorothy_profit_l74_74292

-- Define the conditions
def expense := 53
def number_of_doughnuts := 25
def price_per_doughnut := 3

-- Define revenue and profit calculations
def revenue := number_of_doughnuts * price_per_doughnut
def profit := revenue - expense

-- Prove the profit calculation
theorem dorothy_profit : profit = 22 := by
  sorry

end dorothy_profit_l74_74292


namespace value_of_expression_l74_74597

theorem value_of_expression 
  (a b : ℝ) 
  (h : b = 3 * a - 2) : 
  2 * b - 6 * a + 2 = -2 :=
by 
  intro h
  rw [h]
  linarith

end value_of_expression_l74_74597


namespace numbers_equal_l74_74897

theorem numbers_equal (a b c d : ℕ)
  (h1 : (a + b)^2 % (c * d) = 0)
  (h2 : (a + c)^2 % (b * d) = 0)
  (h3 : (a + d)^2 % (b * c) = 0)
  (h4 : (b + c)^2 % (a * d) = 0)
  (h5 : (b + d)^2 % (a * c) = 0)
  (h6 : (c + d)^2 % (a * b) = 0) :
  a = b ∨ b = c ∨ c = d ∨ a = c ∨ a = d ∨ b = d ∨ (a = b ∧ b = c) ∨ (b = c ∧ c = d) ∨ (a = b ∧ b = d) ∨ (a = c ∧ c = d) :=
sorry

end numbers_equal_l74_74897


namespace range_of_m_l74_74902

-- Definition of propositions p and q
def p (m : ℝ) : Prop := (2 * m - 3)^2 - 4 > 0
def q (m : ℝ) : Prop := m > 2

-- The main theorem stating the range of values for m
theorem range_of_m (m : ℝ) : (p m ∨ q m) ∧ ¬(p m ∧ q m) ↔ (m < 1/2 ∨ (2 < m ∧ m ≤ 5/2)) :=
by
  sorry

end range_of_m_l74_74902


namespace select_students_l74_74722

theorem select_students (n m : ℕ) (hn : n = 4) (hm : m = 3) :
  (nat.choose 4 3) * (nat.choose 3 1) + (nat.choose 4 2) * (nat.choose 3 2) + (nat.choose 4 1) * (nat.choose 3 3) = 34 :=
by 
  sorry

end select_students_l74_74722


namespace gcd_72_108_l74_74112

theorem gcd_72_108 : Nat.gcd 72 108 = 36 :=
by
  sorry

end gcd_72_108_l74_74112


namespace positive_integer_solutions_inequality_l74_74640

theorem positive_integer_solutions_inequality :
  {x : ℕ | 2 * x + 9 ≥ 3 * (x + 2)} = {1, 2, 3} :=
by
  sorry

end positive_integer_solutions_inequality_l74_74640


namespace total_bouncy_balls_l74_74059

-- Definitions based on the conditions of the problem
def packs_of_red := 4
def packs_of_yellow := 8
def packs_of_green := 4
def balls_per_pack := 10

-- Theorem stating the conclusion to be proven
theorem total_bouncy_balls :
  (packs_of_red + packs_of_yellow + packs_of_green) * balls_per_pack = 160 := 
by
  sorry

end total_bouncy_balls_l74_74059


namespace probability_three_primes_out_of_five_l74_74280

def probability_of_prime (p : ℚ) : Prop := ∃ k, k = 4 ∧ p = 4/10

def probability_of_not_prime (p : ℚ) : Prop := ∃ k, k = 6 ∧ p = 6/10

def combinations (n k : ℕ) : ℕ := Nat.choose n k

theorem probability_three_primes_out_of_five :
  ∀ p_prime p_not_prime : ℚ, 
  probability_of_prime p_prime →
  probability_of_not_prime p_not_prime →
  (combinations 5 3 * (p_prime^3 * p_not_prime^2) = 720/3125) :=
by
  intros p_prime p_not_prime h_prime h_not_prime
  sorry

end probability_three_primes_out_of_five_l74_74280


namespace gcf_72_108_l74_74107

def gcf (a b : ℕ) : ℕ := 
  Nat.gcd a b

theorem gcf_72_108 : gcf 72 108 = 36 := by
  sorry

end gcf_72_108_l74_74107


namespace find_sum_of_vars_l74_74434

-- Definitions of the quadratic polynomials
def quadratic1 (x : ℝ) : ℝ := 2 * x^2 + 8 * x + 11
def quadratic2 (y : ℝ) : ℝ := y^2 - 10 * y + 29
def quadratic3 (z : ℝ) : ℝ := 3 * z^2 - 18 * z + 32

-- Theorem statement
theorem find_sum_of_vars (x y z : ℝ) :
  quadratic1 x * quadratic2 y * quadratic3 z ≤ 60 → x + y - z = 0 :=
by 
-- here we would complete the proof steps
sorry

end find_sum_of_vars_l74_74434


namespace kate_money_left_l74_74456

def kate_savings_march := 27
def kate_savings_april := 13
def kate_savings_may := 28
def kate_expenditure_keyboard := 49
def kate_expenditure_mouse := 5

def total_savings := kate_savings_march + kate_savings_april + kate_savings_may
def total_expenditure := kate_expenditure_keyboard + kate_expenditure_mouse
def money_left := total_savings - total_expenditure

-- Prove that Kate has $14 left
theorem kate_money_left : money_left = 14 := 
by 
  sorry

end kate_money_left_l74_74456


namespace car_distance_traveled_l74_74390

theorem car_distance_traveled (d : ℝ)
  (h_avg_speed : 84.70588235294117 = 320 / ((d / 90) + (d / 80))) :
  d = 160 :=
by
  sorry

end car_distance_traveled_l74_74390


namespace min_value_of_vector_sum_l74_74308

noncomputable def min_vector_sum_magnitude (P Q: (ℝ×ℝ)) : ℝ :=
  let x := P.1
  let y := P.2
  let a := Q.1
  let b := Q.2
  Real.sqrt ((x + a)^2 + (y + b)^2)

theorem min_value_of_vector_sum :
  ∃ P Q, 
  (P.1 - 2)^2 + (P.2 - 2)^2 = 1 ∧ 
  Q.1 + Q.2 = 1 ∧ 
  min_vector_sum_magnitude P Q = (5 * Real.sqrt 2 - 2) / 2 :=
by
  sorry

end min_value_of_vector_sum_l74_74308


namespace ny_mets_fans_l74_74838

-- Let Y be the number of NY Yankees fans
-- Let M be the number of NY Mets fans
-- Let R be the number of Boston Red Sox fans
variables (Y M R : ℕ)

-- Given conditions
def ratio_Y_M : Prop := 3 * M = 2 * Y
def ratio_M_R : Prop := 4 * R = 5 * M
def total_fans : Prop := Y + M + R = 330

-- The theorem to prove
theorem ny_mets_fans (h1 : ratio_Y_M Y M) (h2 : ratio_M_R M R) (h3 : total_fans Y M R) : M = 88 :=
sorry

end ny_mets_fans_l74_74838


namespace eugene_initial_pencils_l74_74415

theorem eugene_initial_pencils (e given left : ℕ) (h1 : given = 6) (h2 : left = 45) (h3 : e = given + left) : e = 51 := by
  sorry

end eugene_initial_pencils_l74_74415


namespace gain_percent_calculation_l74_74835

variable (CP SP : ℝ)
variable (gain gain_percent : ℝ)

theorem gain_percent_calculation
  (h₁ : CP = 900) 
  (h₂ : SP = 1180)
  (h₃ : gain = SP - CP)
  (h₄ : gain_percent = (gain / CP) * 100) :
  gain_percent = 31.11 := by
sorry

end gain_percent_calculation_l74_74835


namespace cost_per_dvd_l74_74842

theorem cost_per_dvd (total_cost : ℝ) (num_dvds : ℕ) (cost_per_dvd : ℝ) :
  total_cost = 4.80 ∧ num_dvds = 4 → cost_per_dvd = 1.20 :=
by
  intro h
  sorry

end cost_per_dvd_l74_74842


namespace no_natural_numbers_for_squares_l74_74291

theorem no_natural_numbers_for_squares :
  ∀ x y : ℕ, ¬(∃ k m : ℕ, k^2 = x^2 + y ∧ m^2 = y^2 + x) :=
by sorry

end no_natural_numbers_for_squares_l74_74291


namespace work_duration_17_333_l74_74381

def work_done (rate: ℚ) (days: ℕ) : ℚ := rate * days

def combined_work_done (rate1: ℚ) (rate2: ℚ) (days: ℕ) : ℚ :=
  (rate1 + rate2) * days

def total_work_done (rate1: ℚ) (rate2: ℚ) (rate3: ℚ) (days: ℚ) : ℚ :=
  (rate1 + rate2 + rate3) * days

noncomputable def total_days_work_last (rate_p rate_q rate_r: ℚ) : ℚ :=
  have work_p := 8 * rate_p
  have work_pq := combined_work_done rate_p rate_q 4
  have remaining_work := 1 - (work_p + work_pq)
  have days_all_together := remaining_work / (rate_p + rate_q + rate_r)
  8 + 4 + days_all_together

theorem work_duration_17_333 (rate_p rate_q rate_r: ℚ) : total_days_work_last rate_p rate_q rate_r = 17.333 :=
  by 
  have hp := 1/40
  have hq := 1/24
  have hr := 1/30
  sorry -- proof omitted

end work_duration_17_333_l74_74381


namespace student_survey_l74_74380

-- Define the conditions given in the problem
theorem student_survey (S F : ℝ) (h1 : F = 25 + 65) (h2 : F = 0.45 * S) : S = 200 :=
by
  sorry

end student_survey_l74_74380


namespace allen_reading_days_l74_74402

theorem allen_reading_days (pages_per_day : ℕ) (total_pages : ℕ) 
  (h1 : pages_per_day = 10) (h2 : total_pages = 120) : 
  (total_pages / pages_per_day) = 12 := by
  sorry

end allen_reading_days_l74_74402


namespace range_of_a_minus_b_l74_74306

theorem range_of_a_minus_b (a b : ℝ) (ha : 0 < a ∧ a < 1) (hb : 2 < b ∧ b < 4) : -4 < a - b ∧ a - b < -1 :=
by
  sorry

end range_of_a_minus_b_l74_74306


namespace handshake_count_l74_74495

theorem handshake_count (num_companies : ℕ) (num_representatives : ℕ) 
  (total_handshakes : ℕ) (h1 : num_companies = 5) (h2 : num_representatives = 5)
  (h3 : total_handshakes = (num_companies * num_representatives * 
   (num_companies * num_representatives - 1 - (num_representatives - 1)) / 2)) :
  total_handshakes = 250 :=
by
  rw [h1, h2] at h3
  exact h3

end handshake_count_l74_74495


namespace cube_root_of_27_l74_74077

theorem cube_root_of_27 : ∃ x : ℝ, x^3 = 27 ∧ x = 3 :=
by
  use 3
  split
  { norm_num }
  { rfl }

end cube_root_of_27_l74_74077


namespace no_solution_in_positive_integers_l74_74950

theorem no_solution_in_positive_integers
    (x y : ℕ)
    (h : x > 0 ∧ y > 0) :
    x^2006 - 4 * y^2006 - 2006 ≠ 4 * y^2007 + 2007 * y :=
by
  sorry

end no_solution_in_positive_integers_l74_74950


namespace no_pos_int_mult_5005_in_form_l74_74180

theorem no_pos_int_mult_5005_in_form (i j : ℕ) (h₀ : 0 ≤ i) (h₁ : i < j) (h₂ : j ≤ 49) :
  ¬ ∃ k : ℕ, 5005 * k = 10^j - 10^i := by
  sorry

end no_pos_int_mult_5005_in_form_l74_74180


namespace neither_necessary_nor_sufficient_l74_74170

def p (x y : ℝ) : Prop := x > 1 ∧ y > 1
def q (x y : ℝ) : Prop := x + y > 3

theorem neither_necessary_nor_sufficient :
  ¬ (∀ x y, q x y → p x y) ∧ ¬ (∀ x y, p x y → q x y) :=
by
  sorry

end neither_necessary_nor_sufficient_l74_74170


namespace points_on_equation_correct_l74_74747

theorem points_on_equation_correct (x y : ℝ) :
  (2 * x^2 + y^2 + 3 * x * y + 3 * x + y = 2) ↔ (y = -x - 2 ∨ y = -2 * x + 1) :=
by 
  sorry

end points_on_equation_correct_l74_74747


namespace polynomial_value_at_2_l74_74102

def f (x : ℤ) : ℤ := 7 * x^7 + 6 * x^6 + 5 * x^5 + 4 * x^4 + 3 * x^3 + 2 * x^2 + x

theorem polynomial_value_at_2:
  f 2 = 1538 := by
  sorry

end polynomial_value_at_2_l74_74102


namespace gcf_72_108_l74_74104

theorem gcf_72_108 : Nat.gcd 72 108 = 36 := by
  sorry

end gcf_72_108_l74_74104


namespace prime_factor_of_sum_of_four_consecutive_integers_is_2_l74_74093

theorem prime_factor_of_sum_of_four_consecutive_integers_is_2 (n : ℤ) : 
  ∃ p : ℕ, prime p ∧ ∀ k : ℤ, (k-1) + k + (k+1) + (k+2) ∣ p :=
by
  -- Proof goes here.
  sorry

end prime_factor_of_sum_of_four_consecutive_integers_is_2_l74_74093


namespace Kylie_coins_left_l74_74789

-- Definitions based on given conditions
def piggyBank := 30
def brother := 26
def father := 2 * brother
def sofa := 15
def totalCoins := piggyBank + brother + father + sofa
def coinsGivenToLaura := totalCoins / 2
def coinsLeft := totalCoins - coinsGivenToLaura

-- Theorem statement
theorem Kylie_coins_left : coinsLeft = 62 := by sorry

end Kylie_coins_left_l74_74789


namespace boys_under_six_ratio_l74_74367

theorem boys_under_six_ratio (total_students : ℕ) (two_third_boys : (2/3 : ℚ) * total_students = 25) (boys_under_six : ℕ) (boys_under_six_eq : boys_under_six = 19) :
  boys_under_six / 25 = 19 / 25 :=
by
  sorry

end boys_under_six_ratio_l74_74367


namespace power_rule_for_fractions_calculate_fraction_l74_74868

theorem power_rule_for_fractions (a b : ℚ) (n : ℕ) : (a / b)^n = (a^n) / (b^n) := 
by sorry

theorem calculate_fraction (a b n : ℕ) (h : a = 3 ∧ b = 5 ∧ n = 3) : (a / b)^n = 27 / 125 :=
by
  obtain ⟨ha, hb, hn⟩ := h
  simp [ha, hb, hn, power_rule_for_fractions (3 : ℚ) (5 : ℚ) 3]

end power_rule_for_fractions_calculate_fraction_l74_74868


namespace jason_initial_cards_l74_74331

theorem jason_initial_cards (cards_given_away cards_left : ℕ) (h1 : cards_given_away = 9) (h2 : cards_left = 4) :
  cards_given_away + cards_left = 13 :=
sorry

end jason_initial_cards_l74_74331


namespace fraction_of_friends_l74_74924

variable (x y : ℕ) -- number of first-grade students and sixth-grade students

-- Conditions from the problem
def condition1 : Prop := ∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ a * x = b * y ∧ 1 / 3 = a / (a + b)
def condition2 : Prop := ∃ (c d : ℕ), c > 0 ∧ d > 0 ∧ c * y = d * x ∧ 2 / 5 = c / (c + d)

-- Theorem statement to prove that the fraction of students who are friends is 4/11
theorem fraction_of_friends (h1 : condition1 x y) (h2 : condition2 x y) :
  (1 / 3 : ℚ) * y + (2 / 5 : ℚ) * x / (x + y) = 4 / 11 :=
sorry

end fraction_of_friends_l74_74924


namespace distinct_roots_quadratic_l74_74359

theorem distinct_roots_quadratic (a x₁ x₂ : ℝ) (h₁ : x^2 + a*x + 8 = 0) 
  (h₂ : x₁ ≠ x₂) (h₃ : x₁ - 64 / (17 * x₂^3) = x₂ - 64 / (17 * x₁^3)) : 
  a = 12 ∨ a = -12 := 
sorry

end distinct_roots_quadratic_l74_74359


namespace pentagon_square_ratio_l74_74539

theorem pentagon_square_ratio (p s : ℕ) 
  (h1 : 5 * p = 20) (h2 : 4 * s = 20) : p / s = 4 / 5 :=
by sorry

end pentagon_square_ratio_l74_74539


namespace inequalities_hold_l74_74758

theorem inequalities_hold (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + b = 2) :
  a * b ≤ 1 ∧ a^2 + b^2 ≥ 2 ∧ (1 / a + 1 / b ≥ 2) :=
by
  sorry

end inequalities_hold_l74_74758


namespace find_y_given_z_25_l74_74034

theorem find_y_given_z_25 (k m x y z : ℝ) 
  (hk : y = k * x) 
  (hm : z = m * x)
  (hy5 : y = 10) 
  (hx5z15 : z = 15) 
  (hz25 : z = 25) : 
  y = 50 / 3 := 
  by sorry

end find_y_given_z_25_l74_74034


namespace sin_double_angle_given_cos_identity_l74_74725

theorem sin_double_angle_given_cos_identity (α : ℝ) 
  (h : Real.cos (α + π / 4) = Real.sqrt 2 / 4) : 
  Real.sin (2 * α) = 3 / 4 :=
by
  sorry

end sin_double_angle_given_cos_identity_l74_74725


namespace cars_cost_between_15000_and_20000_l74_74406

theorem cars_cost_between_15000_and_20000 (total_cars : ℕ) (p1 p2 : ℕ) :
    total_cars = 3000 → 
    p1 = 15 → 
    p2 = 40 → 
    (p1 * total_cars / 100 + p2 * total_cars / 100 + x = total_cars) → 
    x = 1350 :=
by
  intro h_total
  intro h_p1
  intro h_p2
  intro h_eq
  sorry

end cars_cost_between_15000_and_20000_l74_74406


namespace find_k_l74_74928

variables (m n k : ℤ)  -- Declaring m, n, k as integer variables.

theorem find_k (h1 : m = 2 * n + 5) (h2 : m + 2 = 2 * (n + k) + 5) : k = 1 :=
by
  sorry

end find_k_l74_74928


namespace cylinder_volume_ratio_l74_74126

theorem cylinder_volume_ratio (r1 r2 V1 V2 : ℝ) (h1 : 2 * Real.pi * r1 = 6) (h2 : 2 * Real.pi * r2 = 10) (hV1 : V1 = Real.pi * r1^2 * 10) (hV2 : V2 = Real.pi * r2^2 * 6) :
  V1 < V2 → (V2 / V1) = 5 / 3 :=
by
  sorry

end cylinder_volume_ratio_l74_74126


namespace find_incorrect_value_l74_74488

theorem find_incorrect_value (n : ℕ) (mean_initial mean_correct : ℕ) (wrongly_copied correct_value incorrect_value : ℕ) 
  (h1 : n = 30) 
  (h2 : mean_initial = 150) 
  (h3 : mean_correct = 151) 
  (h4 : correct_value = 165) 
  (h5 : n * mean_initial = 4500) 
  (h6 : n * mean_correct = 4530) 
  (h7 : n * mean_correct - n * mean_initial = 30) 
  (h8 : correct_value - (n * mean_correct - n * mean_initial) = incorrect_value) : 
  incorrect_value = 135 :=
by
  sorry

end find_incorrect_value_l74_74488


namespace solve_equation_l74_74630

theorem solve_equation (x : ℝ) : ((x-3)^2 + 4*x*(x-3) = 0) → (x = 3 ∨ x = 3/5) :=
by
  sorry

end solve_equation_l74_74630


namespace valid_divisors_of_196_l74_74650

theorem valid_divisors_of_196 : 
  ∃ d : Finset Nat, (∀ x ∈ d, 1 < x ∧ x < 196 ∧ 196 % x = 0) ∧ d.card = 7 := by
  sorry

end valid_divisors_of_196_l74_74650


namespace minimum_floor_sum_l74_74568

theorem minimum_floor_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ⌊a^2 + b^2 / c⌋ + ⌊b^2 + c^2 / a⌋ + ⌊c^2 + a^2 / b⌋ = 34 :=
sorry

end minimum_floor_sum_l74_74568


namespace power_function_increasing_l74_74040

   theorem power_function_increasing (a : ℝ) (h : ∀ x y : ℝ, 0 < x → 0 < y → x < y → x^a < y^a) : 0 < a :=
   by
   sorry
   
end power_function_increasing_l74_74040


namespace find_m_when_circle_tangent_to_line_l74_74289

theorem find_m_when_circle_tangent_to_line 
    (m : ℝ)
    (circle_eq : (x y : ℝ) → (x - 1)^2 + (y - 1)^2 = 4 * m)
    (line_eq : (x y : ℝ) → x + y = 2 * m) :
    (m = 2 + Real.sqrt 3) ∨ (m = 2 - Real.sqrt 3) :=
sorry

end find_m_when_circle_tangent_to_line_l74_74289


namespace weight_of_10m_l74_74766

-- Defining the proportional weight conditions
variable (weight_of_rod : ℝ → ℝ)

-- Conditional facts about the weight function
axiom weight_proportional : ∀ (length1 length2 : ℝ), length1 ≠ 0 → length2 ≠ 0 → 
  weight_of_rod length1 / length1 = weight_of_rod length2 / length2
axiom weight_of_6m : weight_of_rod 6 = 14.04

-- Theorem stating the weight of a 10m rod
theorem weight_of_10m : weight_of_rod 10 = 23.4 := 
sorry

end weight_of_10m_l74_74766


namespace prime_factor_of_sum_of_four_consecutive_integers_is_2_l74_74094

theorem prime_factor_of_sum_of_four_consecutive_integers_is_2 (n : ℤ) : 
  ∃ p : ℕ, prime p ∧ ∀ k : ℤ, (k-1) + k + (k+1) + (k+2) ∣ p :=
by
  -- Proof goes here.
  sorry

end prime_factor_of_sum_of_four_consecutive_integers_is_2_l74_74094


namespace zeroes_in_base_81_l74_74182

-- Definitions based on the conditions:
def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

-- Question: How many zeroes does 15! end with in base 81?
-- Lean 4 proof statement:
theorem zeroes_in_base_81 (n : ℕ) : n = 15 → Nat.factorial n = 
  (81 : ℕ) ^ k * m → k = 1 :=
by
  sorry

end zeroes_in_base_81_l74_74182


namespace election_total_votes_l74_74774

theorem election_total_votes
  (total_votes : ℕ)
  (votes_A : ℕ)
  (votes_B : ℕ)
  (votes_C : ℕ)
  (h1 : votes_A = 55 * total_votes / 100)
  (h2 : votes_B = 35 * total_votes / 100)
  (h3 : votes_C = total_votes - votes_A - votes_B)
  (h4 : votes_A = votes_B + 400) :
  total_votes = 2000 := by
  sorry

end election_total_votes_l74_74774


namespace number_of_ways_to_cut_pipe_l74_74762

theorem number_of_ways_to_cut_pipe : 
  (∃ (x y: ℕ), 2 * x + 3 * y = 15) ∧ 
  (∃! (x y: ℕ), 2 * x + 3 * y = 15) :=
by
  sorry

end number_of_ways_to_cut_pipe_l74_74762


namespace min_value_frac_l74_74573

theorem min_value_frac (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 2 * x + y = 2) : 
  (2 / x) + (1 / y) ≥ 9 / 2 :=
by
  sorry

end min_value_frac_l74_74573


namespace max_value_M_l74_74437

open Real

theorem max_value_M :
  ∃ M : ℝ, ∀ x y z u : ℝ, 0 < x ∧ 0 < y ∧ 0 < z ∧ 0 < u ∧ z ≥ y ∧ (x - 2 * y = z - 2 * u) ∧ (2 * y * z = u * x) →
  M ≤ z / y ∧ M = 6 + 4 * sqrt 2 := 
  sorry

end max_value_M_l74_74437


namespace pentagon_square_ratio_l74_74544

theorem pentagon_square_ratio (p s : ℝ) (h₁ : 5 * p = 20) (h₂ : 4 * s = 20) : p / s = 4 / 5 := 
by 
  sorry

end pentagon_square_ratio_l74_74544


namespace max_leap_years_in_200_years_l74_74775

theorem max_leap_years_in_200_years (leap_year_interval: ℕ) (span: ℕ) 
  (h1: leap_year_interval = 4) 
  (h2: span = 200) : 
  (span / leap_year_interval) = 50 := 
sorry

end max_leap_years_in_200_years_l74_74775


namespace arithmetic_sequence_sum_3m_l74_74233

variable {α : Type*} [LinearOrderedField α]

def arithmetic_sequence_sum (a₁ d : α) (n : ℕ) : α :=
  n • a₁ + (n * (n - 1) / 2) * d

theorem arithmetic_sequence_sum_3m (a₁ d : α) (m : ℕ) (h₁ : arithmetic_sequence_sum a₁ d m = 30) (h₂ : arithmetic_sequence_sum a₁ d (2 * m) = 100) :
  arithmetic_sequence_sum a₁ d (3 * m) = 210 :=
sorry

end arithmetic_sequence_sum_3m_l74_74233


namespace probability_of_multiple_of_3_l74_74387

-- For handling probability
noncomputable def probability {α : Type*} [Fintype α] (s : Finset α) : ℚ :=
  (s.card : ℚ) / (Fintype.card α)

-- Definitions to match conditions
def digits : Finset ℕ := {1, 2, 3, 4, 5}
def draws : Finset (Finset ℕ) := digits.powerset.filter (λ s, s.card = 4)

-- Function to determine if a set of digits sums to a multiple of 3
def sum_to_multiple_of_3 (s : Finset ℕ) : Prop :=
  s.sum id % 3 = 0

-- Set of draws whose sum of digits is a multiple of 3
def valid_draws := draws.filter sum_to_multiple_of_3

-- Total number of ways to order four digits from the draw
def total_permuted_draws : Finset (List ℕ) :=
  draws.bUnion (λ s, s.permutations)

-- Number of valid (i.e., multiple of 3) four-digit numbers
def valid_permuted_draws : Finset (List ℕ) :=
  valid_draws.bUnion (λ s, s.permutations)

-- The theorem to be proved
theorem probability_of_multiple_of_3 : probability valid_permuted_draws total_permuted_draws = 1 / 5 := by
  sorry

end probability_of_multiple_of_3_l74_74387


namespace find_smallest_y_l74_74994

noncomputable def x : ℕ := 5 * 15 * 35

def is_perfect_fourth_power (n : ℕ) : Prop :=
  ∃ m : ℕ, m ^ 4 = n

theorem find_smallest_y : ∃ y : ℕ, y > 0 ∧ is_perfect_fourth_power (x * y) ∧ y = 46485 := by
  sorry

end find_smallest_y_l74_74994


namespace acid_fraction_in_third_flask_correct_l74_74820

noncomputable def acid_concentration_in_third_flask 
  (w : ℝ) (W : ℝ) 
  (h1 : 10 / (10 + w) = 1 / 20)
  (h2 : 20 / (20 + (W - w)) = 7 / 30) 
  : ℝ := 
30 / (30 + W)

theorem acid_fraction_in_third_flask_correct
  (w : ℝ) (W : ℝ)
  (h1 : 10 / (10 + w) = 1 / 20)
  (h2 : 20 / (20 + (W - w)) = 7 / 30) :
  acid_concentration_in_third_flask w W h1 h2 = 21 / 200 :=
begin
  sorry
end

end acid_fraction_in_third_flask_correct_l74_74820


namespace isosceles_triangle_perimeter_l74_74147

-- Definitions and conditions
-- Define the lengths of the three sides of the triangle
def a : ℕ := 3
def b : ℕ := 8

-- Define that the triangle is isosceles
def is_isosceles_triangle := 
  (a = a) ∨ (b = b) ∨ (a = b)

-- Perimeter of the triangle
def perimeter (x y z : ℕ) := x + y + z

-- The theorem we need to prove
theorem isosceles_triangle_perimeter : is_isosceles_triangle → (a + b + b = 19) :=
by
  intro h
  sorry

end isosceles_triangle_perimeter_l74_74147


namespace problem_solution_l74_74385

theorem problem_solution :
  ((8 * 2.25 - 5 * 0.85) / 2.5 + (3 / 5 * 1.5 - 7 / 8 * 0.35) / 1.25) = 5.975 :=
by
  sorry

end problem_solution_l74_74385


namespace excircle_side_formula_l74_74160

theorem excircle_side_formula 
  (a b c r_a r_b r_c : ℝ)
  (h1 : r_c = Real.sqrt (r_a * r_b)) :
  c = (a^2 + b^2) / (a + b) :=
sorry

end excircle_side_formula_l74_74160


namespace evaluate_expression_l74_74262

theorem evaluate_expression (x : ℕ) (h : x = 3) : 5^3 - 2^x * 3 + 4^2 = 117 :=
by
  rw [h]
  sorry

end evaluate_expression_l74_74262


namespace isosceles_triangle_base_angles_l74_74591

theorem isosceles_triangle_base_angles (a b : ℝ) (h1 : a + b + b = 180)
  (h2 : a = 110) : b = 35 :=
by 
  sorry

end isosceles_triangle_base_angles_l74_74591


namespace blue_apples_l74_74440

theorem blue_apples (B : ℕ) (h : (12 / 5) * B = 12) : B = 5 :=
by
  sorry

end blue_apples_l74_74440


namespace largest_expr_l74_74938

noncomputable def A : ℝ := 2 * 1005 ^ 1006
noncomputable def B : ℝ := 1005 ^ 1006
noncomputable def C : ℝ := 1004 * 1005 ^ 1005
noncomputable def D : ℝ := 2 * 1005 ^ 1005
noncomputable def E : ℝ := 1005 ^ 1005
noncomputable def F : ℝ := 1005 ^ 1004

theorem largest_expr : A - B > B - C ∧ A - B > C - D ∧ A - B > D - E ∧ A - B > E - F :=
by
  sorry

end largest_expr_l74_74938


namespace gcf_72_108_l74_74109

def gcf (a b : ℕ) : ℕ := 
  Nat.gcd a b

theorem gcf_72_108 : gcf 72 108 = 36 := by
  sorry

end gcf_72_108_l74_74109


namespace total_coins_l74_74794

-- Define the number of stacks and the number of coins per stack
def stacks : ℕ := 5
def coins_per_stack : ℕ := 3

-- State the theorem to prove the total number of coins
theorem total_coins (s c : ℕ) (hs : s = stacks) (hc : c = coins_per_stack) : s * c = 15 :=
by
  -- Proof is omitted
  sorry

end total_coins_l74_74794


namespace points_on_equation_correct_l74_74746

theorem points_on_equation_correct (x y : ℝ) :
  (2 * x^2 + y^2 + 3 * x * y + 3 * x + y = 2) ↔ (y = -x - 2 ∨ y = -2 * x + 1) :=
by 
  sorry

end points_on_equation_correct_l74_74746


namespace john_total_spent_l74_74051

def silver_ounces : ℝ := 2.5
def silver_price_per_ounce : ℝ := 25
def gold_ounces : ℝ := 3.5
def gold_price_multiplier : ℝ := 60
def platinum_ounces : ℝ := 4.5
def platinum_price_per_ounce_gbp : ℝ := 80
def palladium_ounces : ℝ := 5.5
def palladium_price_per_ounce_eur : ℝ := 100

def usd_per_gbp_monday : ℝ := 1.3
def usd_per_gbp_friday : ℝ := 1.4
def usd_per_eur_wednesday : ℝ := 1.15
def usd_per_eur_saturday : ℝ := 1.2

def discount_rate : ℝ := 0.05
def tax_rate : ℝ := 0.08

def total_amount_john_spends_usd : ℝ := 
  (silver_ounces * silver_price_per_ounce * (1 - discount_rate)) + 
  (gold_ounces * (gold_price_multiplier * silver_price_per_ounce) * (1 - discount_rate)) + 
  (((platinum_ounces * platinum_price_per_ounce_gbp) * (1 + tax_rate)) * usd_per_gbp_monday) + 
  ((palladium_ounces * palladium_price_per_ounce_eur) * usd_per_eur_wednesday)

theorem john_total_spent : total_amount_john_spends_usd = 6184.815 := by
  sorry

end john_total_spent_l74_74051


namespace calculate_womans_haircut_cost_l74_74633

-- Define the necessary constants and conditions
def W : ℝ := sorry
def child_haircut_cost : ℝ := 36
def tip_percentage : ℝ := 0.20
def total_tip : ℝ := 24
def number_of_children : ℕ := 2

-- Helper function to calculate total cost before the tip
def total_cost_before_tip (W : ℝ) (number_of_children : ℕ) (child_haircut_cost : ℝ) : ℝ :=
  W + number_of_children * child_haircut_cost

-- Lean statement for the main theorem
theorem calculate_womans_haircut_cost (W : ℝ) (child_haircut_cost : ℝ) (tip_percentage : ℝ)
  (total_tip : ℝ) (number_of_children : ℕ) :
  (tip_percentage * total_cost_before_tip W number_of_children child_haircut_cost) = total_tip →
  W = 48 :=
by
  sorry

end calculate_womans_haircut_cost_l74_74633


namespace yellow_scores_l74_74970

theorem yellow_scores (W B : ℕ) 
  (h₁ : W / B = 7 / 6)
  (h₂ : (2 / 3 : ℚ) * (W - B) = 4) : 
  W + B = 78 :=
sorry

end yellow_scores_l74_74970


namespace train_speed_l74_74144

/-- Given: 
1. A train travels a distance of 80 km in 40 minutes. 
2. We need to prove that the speed of the train is 120 km/h.
-/
theorem train_speed (distance : ℝ) (time_minutes : ℝ) (time_hours : ℝ) (speed : ℝ) 
  (h_distance : distance = 80) 
  (h_time_minutes : time_minutes = 40) 
  (h_time_hours : time_hours = 40 / 60) 
  (h_speed : speed = distance / time_hours) : 
  speed = 120 :=
sorry

end train_speed_l74_74144


namespace zac_strawberries_l74_74335

theorem zac_strawberries (J M Z : ℕ) 
  (h1 : J + M + Z = 550) 
  (h2 : J + M = 350) 
  (h3 : M + Z = 250) : 
  Z = 200 :=
sorry

end zac_strawberries_l74_74335


namespace evaluate_f_at_3_l74_74904

noncomputable def f : ℝ → ℝ := sorry

axiom h_odd : ∀ x : ℝ, f (-x) = -f x 
axiom h_periodic : ∀ x : ℝ, f x = f (x + 4)
axiom h_def : ∀ x : ℝ, 0 < x ∧ x < 2 → f x = 2 * x^2

theorem evaluate_f_at_3 : f 3 = -2 := by
  sorry

end evaluate_f_at_3_l74_74904


namespace find_k_l74_74360

theorem find_k (k l : ℝ) (C : ℝ × ℝ) (OC : ℝ) (A B D : ℝ × ℝ)
  (hC_coords : C = (0, 3))
  (hl_val : l = 3)
  (line_eqn : ∀ x, y = k * x + l)
  (intersect_eqn : ∀ x, y = 1 / x)
  (hA_coords : A = (1 / 6, 6))
  (hD_coords : D = (1 / 6, 6))
  (dist_ABC : dist A B = dist B C)
  (dist_BCD : dist B C = dist C D)
  (OC_val : OC = 3) :
  k = 18 := 
sorry

end find_k_l74_74360


namespace intersection_locus_l74_74939

theorem intersection_locus
  (a b : ℝ) (a_gt_b : a > b) (b_gt_zero : b > 0)
  (ellipse_eq : ∀ x y : ℝ, (x^2)/(a^2) + (y^2)/(b^2) = 1) :
  ∃ (x y : ℝ), (x^2)/(a^2) - (y^2)/(b^2) = 1 :=
sorry

end intersection_locus_l74_74939


namespace minimum_yellow_marbles_l74_74400

theorem minimum_yellow_marbles :
  ∀ (n y : ℕ), 
  (3 ∣ n) ∧ (4 ∣ n) ∧ 
  (9 + y + 2 * y ≤ n) ∧ 
  (n = n / 3 + n / 4 + 9 + y + 2 * y) → 
  y = 4 :=
by
  sorry

end minimum_yellow_marbles_l74_74400


namespace find_original_number_l74_74356

theorem find_original_number (r : ℝ) (h : 1.15 * r - 0.7 * r = 40) : r = 88.88888888888889 :=
by
  sorry

end find_original_number_l74_74356


namespace points_on_equation_correct_l74_74744

theorem points_on_equation_correct (x y : ℝ) :
  (2 * x^2 + y^2 + 3 * x * y + 3 * x + y = 2) ↔ (y = -x - 2 ∨ y = -2 * x + 1) :=
by 
  sorry

end points_on_equation_correct_l74_74744


namespace odd_function_f_value_l74_74907

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then x^3 + x + 1 else x^3 + x - 1

theorem odd_function_f_value : 
  f 2 = 9 := by
  sorry

end odd_function_f_value_l74_74907


namespace negation_of_existential_l74_74362

theorem negation_of_existential (h : ∃ x_0 : ℝ, x_0^3 - x_0^2 + 1 ≤ 0) : 
  ∀ x : ℝ, x^3 - x^2 + 1 > 0 :=
sorry

end negation_of_existential_l74_74362


namespace simplify_expression1_simplify_expression2_l74_74353

-- Problem 1
theorem simplify_expression1 (a : ℝ) : 
  (a^2)^3 + 3 * a^4 * a^2 - a^8 / a^2 = 3 * a^6 :=
by sorry

-- Problem 2
theorem simplify_expression2 (x : ℝ) : 
  (x - 3) * (x + 4) - x * (x + 3) = -2 * x - 12 :=
by sorry

end simplify_expression1_simplify_expression2_l74_74353


namespace sum_of_permutations_of_1234567_l74_74894

theorem sum_of_permutations_of_1234567 : 
  let factorial_7 := 5040
  let sum_of_digits := 1 + 2 + 3 + 4 + 5 + 6 + 7
  let geometric_series_sum := (10 ^ 7 - 1) / (10 - 1)
  sum_of_digits * factorial_7 * geometric_series_sum = 22399997760 :=
by
  let factorial_7 := 5040
  let sum_of_digits := 1 + 2 + 3 + 4 + 5 + 6 + 7
  let geometric_series_sum := (10^7 - 1) / (10 - 1)
  sorry

end sum_of_permutations_of_1234567_l74_74894


namespace problem1_l74_74384

theorem problem1 (α : ℝ) (h : Real.tan (π / 4 + α) = 1 / 2) :
  (Real.sin (2 * α) - Real.cos α ^ 2) / (1 + Real.cos (2 * α)) = -5 / 6 := 
  sorry

end problem1_l74_74384


namespace statement_B_not_true_l74_74551

def op_star (x y : ℝ) := x^2 - 2*x*y + y^2

theorem statement_B_not_true (x y : ℝ) : 3 * (op_star x y) ≠ op_star (3 * x) (3 * y) :=
by
  have h1 : 3 * (op_star x y) = 3 * (x^2 - 2 * x * y + y^2) := rfl
  have h2 : op_star (3 * x) (3 * y) = (3 * x)^2 - 2 * (3 * x) * (3 * y) + (3 * y)^2 := rfl
  sorry

end statement_B_not_true_l74_74551


namespace cubic_sum_l74_74958

theorem cubic_sum (a b c : ℝ) (h1 : a + b + c = 0) (h2 : a * b + a * c + b * c = 1) (h3 : a * b * c = -2) : 
  a^3 + b^3 + c^3 = -6 :=
sorry

end cubic_sum_l74_74958


namespace range_of_k_l74_74428

theorem range_of_k 
  (h : ∀ x : ℝ, x = 1 → k^2 * x^2 - 6 * k * x + 8 ≥ 0) :
  k ≥ 4 ∨ k ≤ 2 := by
sorry

end range_of_k_l74_74428


namespace one_div_abs_z_eq_sqrt_two_l74_74022

open Complex

theorem one_div_abs_z_eq_sqrt_two (z : ℂ) (h : z = i / (1 - i)) : 1 / Complex.abs z = Real.sqrt 2 :=
by
  sorry

end one_div_abs_z_eq_sqrt_two_l74_74022


namespace sum_of_squares_l74_74728

theorem sum_of_squares (a b c : ℝ) (h1 : a + b + c = 5) (h2 : ab + bc + ac = 5) : a^2 + b^2 + c^2 = 15 :=
by sorry

end sum_of_squares_l74_74728


namespace more_blue_blocks_than_red_l74_74324

theorem more_blue_blocks_than_red 
  (red_blocks : ℕ) 
  (yellow_blocks : ℕ) 
  (blue_blocks : ℕ) 
  (total_blocks : ℕ) 
  (h_red : red_blocks = 18) 
  (h_yellow : yellow_blocks = red_blocks + 7) 
  (h_total : total_blocks = red_blocks + yellow_blocks + blue_blocks) 
  (h_total_given : total_blocks = 75) :
  blue_blocks - red_blocks = 14 :=
by sorry

end more_blue_blocks_than_red_l74_74324


namespace smallest_common_multiple_of_8_and_6_l74_74508

theorem smallest_common_multiple_of_8_and_6 : ∃ n : ℕ, n > 0 ∧ (8 ∣ n) ∧ (6 ∣ n) ∧ ∀ m : ℕ, (m > 0 ∧ (8 ∣ m) ∧ (6 ∣ m)) → n ≤ m :=
by
  sorry

end smallest_common_multiple_of_8_and_6_l74_74508


namespace number_of_designs_with_shaded_fraction_eq_three_fifths_l74_74295

theorem number_of_designs_with_shaded_fraction_eq_three_fifths :
  let designs := [ (3 / 8 : ℚ), (12 / 20 : ℚ), (2 / 3 : ℚ), (15 / 25 : ℚ), (4 / 8 : ℚ) ]
  in (designs.count (λ x, x = 3 / 5 : ℚ)) = 2 := by
  sorry

end number_of_designs_with_shaded_fraction_eq_three_fifths_l74_74295


namespace initial_pokemon_cards_l74_74329

theorem initial_pokemon_cards (x : ℤ) (h : x - 9 = 4) : x = 13 :=
by
  sorry

end initial_pokemon_cards_l74_74329


namespace concentration_flask3_l74_74818

open Real

noncomputable def proof_problem : Prop :=
  ∃ (W : ℝ) (w : ℝ), 
    let flask1_acid := 10 in
    let flask2_acid := 20 in
    let flask3_acid := 30 in
    let flask1_total := flask1_acid + w in
    let flask2_total := flask2_acid + (W - w) in
    let flask3_total := flask3_acid + W in
    (flask1_acid / flask1_total = 1 / 20) ∧
    (flask2_acid / flask2_total = 7 / 30) ∧
    (flask3_acid / flask3_total = 21 / 200)

theorem concentration_flask3 : proof_problem :=
begin
  sorry
end

end concentration_flask3_l74_74818


namespace company_KW_price_percentage_l74_74153

theorem company_KW_price_percentage
  (A B : ℝ)
  (h1 : ∀ P: ℝ, P = 1.9 * A)
  (h2 : ∀ P: ℝ, P = 2 * B) :
  Price = 131.034 / 100 * (A + B) := 
by
  sorry

end company_KW_price_percentage_l74_74153


namespace evaluate_expression_l74_74015

theorem evaluate_expression (x : ℝ) (h : x = -3) : (5 + x * (5 + x) - 5^2) / (x - 5 + x^2) = -26 :=
by
  rw [h]
  sorry

end evaluate_expression_l74_74015


namespace inequality_solution_l74_74713

theorem inequality_solution {x : ℝ} (h : -2 < (x^2 - 18*x + 24) / (x^2 - 4*x + 8) ∧ (x^2 - 18*x + 24) / (x^2 - 4*x + 8) < 2) : 
  x ∈ Set.Ioo (-2 : ℝ) (10 / 3) :=
by
  sorry

end inequality_solution_l74_74713


namespace Susan_roses_ratio_l74_74219

theorem Susan_roses_ratio (total_roses given_roses vase_roses remaining_roses : ℕ) 
  (H1 : total_roses = 3 * 12)
  (H2 : vase_roses = total_roses - given_roses)
  (H3 : remaining_roses = vase_roses * 2 / 3)
  (H4 : remaining_roses = 12) :
  given_roses / gcd given_roses total_roses = 1 ∧ total_roses / gcd given_roses total_roses = 2 :=
by
  sorry

end Susan_roses_ratio_l74_74219


namespace difference_of_numbers_l74_74973

theorem difference_of_numbers (a b : ℕ) (h1 : a = 2 * b) (h2 : (a + 4) / (b + 4) = 5 / 7) : a - b = 8 := 
by
  sorry

end difference_of_numbers_l74_74973


namespace gcd_of_polynomial_and_multiple_of_350_l74_74310

theorem gcd_of_polynomial_and_multiple_of_350 (b : ℕ) (hb : 350 ∣ b) :
  gcd (2 * b^3 + 3 * b^2 + 5 * b + 70) b = 70 := by
  sorry

end gcd_of_polynomial_and_multiple_of_350_l74_74310


namespace gcd_72_108_l74_74111

theorem gcd_72_108 : Nat.gcd 72 108 = 36 :=
by
  sorry

end gcd_72_108_l74_74111


namespace solution_l74_74885

noncomputable def f : ℝ → ℝ := sorry

axiom even_f : ∀ x : ℝ, f x = f (-x)

axiom periodic_f : ∀ x : ℝ, f (x - 3) = - f x

axiom increasing_f_on_interval : ∀ x1 x2 : ℝ, (0 ≤ x1 ∧ x1 ≤ 3 ∧ 0 ≤ x2 ∧ x2 ≤ 3 ∧ x1 ≠ x2) → (f x1 - f x2) / (x1 - x2) > 0

theorem solution : f 49 < f 64 ∧ f 64 < f 81 :=
by
  sorry

end solution_l74_74885


namespace intersection_of_A_and_B_l74_74176

noncomputable def A : Set ℝ := {x : ℝ | x^2 - 1 ≤ 0}
noncomputable def B : Set ℝ := {x : ℝ | 0 < x ∧ x ≤ 2}

theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | 0 < x ∧ x ≤ 1} :=
by
  sorry

end intersection_of_A_and_B_l74_74176


namespace candy_distribution_l74_74807

theorem candy_distribution (n : ℕ) (h1 : n > 0) (h2 : 100 % n = 0) (h3 : 99 % n = 0) : n = 11 :=
sorry

end candy_distribution_l74_74807


namespace least_number_to_add_l74_74515

theorem least_number_to_add (n : ℕ) (d : ℕ) (h1 : n = 907223) (h2 : d = 577) : (d - (n % d) = 518) := 
by
  rw [h1, h2]
  sorry

end least_number_to_add_l74_74515


namespace NewYearSeasonMarkup_is_25percent_l74_74697

variable (C N : ℝ)
variable (h1 : N >= 0)
variable (h2 : 0.92 * (1 + N) * 1.20 * C = 1.38 * C)

theorem NewYearSeasonMarkup_is_25percent : N = 0.25 :=
  by
  sorry

end NewYearSeasonMarkup_is_25percent_l74_74697


namespace fraction_equality_l74_74636

theorem fraction_equality : (2 + 4) / (1 + 2) = 2 := by
  sorry

end fraction_equality_l74_74636


namespace solution_set_inequality_l74_74041

theorem solution_set_inequality 
  (a b : ℝ)
  (h1 : ∀ x, a * x^2 + b * x + 3 > 0 ↔ -1 < x ∧ x < 1/2) :
  ((-1:ℝ) < x ∧ x < 2) ↔ 3 * x^2 + b * x + a < 0 :=
by 
  -- Write the proof here
  sorry

end solution_set_inequality_l74_74041


namespace right_handed_players_count_l74_74839

theorem right_handed_players_count (total_players throwers left_handed_non_throwers right_handed_non_throwers : ℕ) 
  (h1 : total_players = 70)
  (h2 : throwers = 46)
  (h3 : left_handed_non_throwers = (total_players - throwers) / 3)
  (h4 : right_handed_non_throwers = total_players - throwers - left_handed_non_throwers)
  (h5 : ∀ n, n = throwers + right_handed_non_throwers) : 
  (throwers + right_handed_non_throwers) = 62 := 
by 
  sorry

end right_handed_players_count_l74_74839


namespace peanuts_added_l74_74086

theorem peanuts_added (initial_peanuts final_peanuts added_peanuts : ℕ) 
(h1 : initial_peanuts = 10) 
(h2 : final_peanuts = 18) 
(h3 : final_peanuts = initial_peanuts + added_peanuts) : 
added_peanuts = 8 := 
by {
  sorry
}

end peanuts_added_l74_74086


namespace total_dogs_l74_74382

axiom brown_dogs : ℕ
axiom white_dogs : ℕ
axiom black_dogs : ℕ

theorem total_dogs (b w bl : ℕ) (h1 : b = 20) (h2 : w = 10) (h3 : bl = 15) : (b + w + bl) = 45 :=
by {
  sorry
}

end total_dogs_l74_74382


namespace sum_of_squares_of_roots_l74_74549

theorem sum_of_squares_of_roots :
  ∀ (p q r : ℚ), (3 * p^3 + 2 * p^2 - 5 * p - 8 = 0) ∧
                 (3 * q^3 + 2 * q^2 - 5 * q - 8 = 0) ∧
                 (3 * r^3 + 2 * r^2 - 5 * r - 8 = 0) →
                 p^2 + q^2 + r^2 = 34 / 9 := 
by
  sorry

end sum_of_squares_of_roots_l74_74549


namespace card_game_probability_l74_74690

theorem card_game_probability :
  let A_wins := 4;  -- number of heads needed for A to win all cards
  let B_wins := 4;  -- number of tails needed for B to win all cards
  let total_flips := 5;  -- exactly 5 flips
  (Nat.choose total_flips 1 + Nat.choose total_flips 1) / (2^total_flips) = 5 / 16 :=
by
  sorry

end card_game_probability_l74_74690


namespace ratio_accepted_to_rejected_l74_74599

-- Let n be the total number of eggs processed per day
def eggs_per_day := 400

-- Let accepted_per_batch be the number of accepted eggs per batch
def accepted_per_batch := 96

-- Let rejected_per_batch be the number of rejected eggs per batch
def rejected_per_batch := 4

-- On a particular day, 12 additional eggs were accepted
def additional_accepted_eggs := 12

-- Normalize definitions to make our statements clearer
def accepted_batches := eggs_per_day / (accepted_per_batch + rejected_per_batch)
def normally_accepted_eggs := accepted_per_batch * accepted_batches
def normally_rejected_eggs := rejected_per_batch * accepted_batches
def total_accepted_eggs := normally_accepted_eggs + additional_accepted_eggs
def total_rejected_eggs := eggs_per_day - total_accepted_eggs

theorem ratio_accepted_to_rejected :
  (total_accepted_eggs / gcd total_accepted_eggs total_rejected_eggs) = 99 ∧
  (total_rejected_eggs / gcd total_accepted_eggs total_rejected_eggs) = 1 :=
by
  sorry

end ratio_accepted_to_rejected_l74_74599


namespace combined_average_score_l74_74347

theorem combined_average_score (M A : ℝ) (m a : ℝ)
  (hM : M = 78) (hA : A = 85) (h_ratio : m = 2 * a / 3) :
  (78 * (2 * a / 3) + 85 * a) / ((2 * a / 3) + a) = 82 := by
  sorry

end combined_average_score_l74_74347


namespace fourth_metal_mass_l74_74850

noncomputable def alloy_mass_problem (m1 m2 m3 m4 : ℝ) : Prop :=
  m1 + m2 + m3 + m4 = 20 ∧
  m1 = 1.5 * m2 ∧
  m2 = (3 / 4) * m3 ∧
  m3 = (5 / 6) * m4

theorem fourth_metal_mass :
  ∃ (m4 : ℝ), (∃ (m1 m2 m3 : ℝ), alloy_mass_problem m1 m2 m3 m4) ∧ abs (m4 - 5.89) < 0.01 :=
begin
  sorry  -- Proof is skipped
end

end fourth_metal_mass_l74_74850


namespace fry_sausage_time_l74_74457

variable (time_per_sausage : ℕ)

noncomputable def time_for_sausages (sausages : ℕ) (tps : ℕ) : ℕ :=
  sausages * tps

noncomputable def time_for_eggs (eggs : ℕ) (minutes_per_egg : ℕ) : ℕ :=
  eggs * minutes_per_egg

noncomputable def total_time (time_sausages : ℕ) (time_eggs : ℕ) : ℕ :=
  time_sausages + time_eggs

theorem fry_sausage_time :
  let sausages := 3
  let eggs := 6
  let minutes_per_egg := 4
  let total_time_taken := 39
  total_time (time_for_sausages sausages time_per_sausage) (time_for_eggs eggs minutes_per_egg) = total_time_taken
  → time_per_sausage = 5 := by
  sorry

end fry_sausage_time_l74_74457


namespace cube_root_of_27_l74_74074

theorem cube_root_of_27 : 
  ∃ x : ℝ, x^3 = 27 ∧ x = 3 :=
begin
  sorry
end

end cube_root_of_27_l74_74074


namespace find_numbers_l74_74491

theorem find_numbers (x y : ℤ) (h1 : x + y = 18) (h2 : x - y = 24) : x = 21 ∧ y = -3 :=
by
  sorry

end find_numbers_l74_74491


namespace add_decimals_l74_74378

theorem add_decimals : 4.3 + 3.88 = 8.18 := 
sorry

end add_decimals_l74_74378
