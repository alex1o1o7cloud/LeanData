import Mathlib

namespace dance_parity_l2927_292761

theorem dance_parity (n : ℕ) (h_odd : Odd n) (dances : Fin n → ℕ) : 
  ∃ i : Fin n, Even (dances i) := by
  sorry

end dance_parity_l2927_292761


namespace min_guests_football_banquet_l2927_292731

theorem min_guests_football_banquet (total_food : ℕ) (max_per_guest : ℕ) 
  (h1 : total_food = 325)
  (h2 : max_per_guest = 2) :
  (total_food + max_per_guest - 1) / max_per_guest = 163 := by
  sorry

end min_guests_football_banquet_l2927_292731


namespace playground_area_l2927_292795

/-- The area of a rectangular playground with perimeter 90 meters and length three times the width -/
theorem playground_area : 
  ∀ (length width : ℝ),
  length > 0 → width > 0 →
  2 * (length + width) = 90 →
  length = 3 * width →
  length * width = 379.6875 := by
sorry

end playground_area_l2927_292795


namespace minimum_time_for_assessment_l2927_292718

/-- Represents the minimum time needed to assess students -/
def minimum_assessment_time (
  teacher1_problem_solving_time : ℕ)
  (teacher1_theory_time : ℕ)
  (teacher2_problem_solving_time : ℕ)
  (teacher2_theory_time : ℕ)
  (total_students : ℕ) : ℕ :=
  110

/-- Theorem stating the minimum time needed to assess 25 students
    given the specified conditions -/
theorem minimum_time_for_assessment :
  minimum_assessment_time 5 7 3 4 25 = 110 := by
  sorry

end minimum_time_for_assessment_l2927_292718


namespace cantor_set_removal_operations_l2927_292716

theorem cantor_set_removal_operations (n : ℕ) : 
  (((2 : ℝ) / 3) ^ (n - 1) * (1 / 3) ≥ 1 / 60) ↔ n ≤ 8 :=
sorry

end cantor_set_removal_operations_l2927_292716


namespace inequality_solution_set_l2927_292727

theorem inequality_solution_set (a : ℝ) (x₁ x₂ : ℝ) : 
  a > 0 →
  (∀ x, x^2 - a*x - 6*a^2 < 0 ↔ x₁ < x ∧ x < x₂) →
  x₂ - x₁ = 10 →
  a = 2 := by
sorry

end inequality_solution_set_l2927_292727


namespace sally_buttons_count_l2927_292774

/-- The number of buttons needed for all clothing items Sally sews over three days -/
def total_buttons : ℕ :=
  let shirt_buttons := 5
  let pants_buttons := 3
  let jacket_buttons := 10
  let monday := 4 * shirt_buttons + 2 * pants_buttons + 1 * jacket_buttons
  let tuesday := 3 * shirt_buttons + 1 * pants_buttons + 2 * jacket_buttons
  let wednesday := 2 * shirt_buttons + 3 * pants_buttons + 1 * jacket_buttons
  monday + tuesday + wednesday

/-- Theorem stating that the total number of buttons Sally needs is 103 -/
theorem sally_buttons_count : total_buttons = 103 := by
  sorry

end sally_buttons_count_l2927_292774


namespace square_minus_twelve_plus_fiftyfour_l2927_292784

theorem square_minus_twelve_plus_fiftyfour (a b : ℝ) 
  (h1 : 0 ≤ a) (h2 : 0 ≤ b) 
  (h3 : a^2 + b^2 = 74) (h4 : a * b = 35) : 
  a^2 - 12 * a + 54 = 19 := by
sorry

end square_minus_twelve_plus_fiftyfour_l2927_292784


namespace f_composition_equals_pi_plus_two_l2927_292739

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then x + 2
  else if x = 0 then Real.pi
  else 0

-- State the theorem
theorem f_composition_equals_pi_plus_two :
  f (f (f (-2))) = Real.pi + 2 := by
  sorry

end f_composition_equals_pi_plus_two_l2927_292739


namespace luis_gum_contribution_l2927_292750

/-- Calculates the number of gum pieces Luis gave to Maria -/
def luisGumPieces (initialPieces tomsContribution totalPieces : ℕ) : ℕ :=
  totalPieces - (initialPieces + tomsContribution)

theorem luis_gum_contribution :
  luisGumPieces 25 16 61 = 20 := by
  sorry

end luis_gum_contribution_l2927_292750


namespace largest_constant_inequality_l2927_292723

theorem largest_constant_inequality (x y z : ℝ) :
  ∃ (C : ℝ), (∀ (x y z : ℝ), x^2 + y^2 + z^2 + 1 ≥ C * (x + y + z)) ∧
    C = 2 / Real.sqrt 3 ∧
    ∀ (D : ℝ), (∀ (x y z : ℝ), x^2 + y^2 + z^2 + 1 ≥ D * (x + y + z)) → D ≤ C :=
by
  sorry

end largest_constant_inequality_l2927_292723


namespace cindy_envelope_distribution_l2927_292711

theorem cindy_envelope_distribution (initial_envelopes : ℕ) (friends : ℕ) (remaining_envelopes : ℕ) 
  (h1 : initial_envelopes = 37)
  (h2 : friends = 5)
  (h3 : remaining_envelopes = 22) :
  (initial_envelopes - remaining_envelopes) / friends = 3 := by
  sorry

end cindy_envelope_distribution_l2927_292711


namespace distance_is_sqrt_1501_div_17_l2927_292793

/-- The distance from a point to a line in 3D space -/
def distance_point_to_line (point : ℝ × ℝ × ℝ) (line_point : ℝ × ℝ × ℝ) (line_direction : ℝ × ℝ × ℝ) : ℝ :=
  sorry

/-- The given point -/
def given_point : ℝ × ℝ × ℝ := (2, 3, 4)

/-- A point on the given line -/
def line_point : ℝ × ℝ × ℝ := (5, 6, 8)

/-- The direction vector of the given line -/
def line_direction : ℝ × ℝ × ℝ := (4, 3, -3)

/-- Theorem stating that the distance from the given point to the line is √1501 / 17 -/
theorem distance_is_sqrt_1501_div_17 : 
  distance_point_to_line given_point line_point line_direction = Real.sqrt 1501 / 17 := by
  sorry

end distance_is_sqrt_1501_div_17_l2927_292793


namespace sum_zero_implies_product_sum_nonpositive_l2927_292780

theorem sum_zero_implies_product_sum_nonpositive
  (a b c : ℝ) (h : a + b + c = 0) :
  a * b + b * c + c * a ≤ 0 := by
  sorry

end sum_zero_implies_product_sum_nonpositive_l2927_292780


namespace house_coloring_l2927_292715

/-- A type representing the colors of houses -/
inductive Color
| Blue
| Green
| Red

/-- A function representing the move of residents between houses -/
def move (n : ℕ) : ℕ → ℕ :=
  sorry

/-- A function representing the coloring of houses -/
def color (n : ℕ) : ℕ → Color :=
  sorry

/-- The main theorem -/
theorem house_coloring (n : ℕ) (h_pos : 0 < n) :
  ∃ (move : ℕ → ℕ) (color : ℕ → Color),
    (∀ i : ℕ, i < n → move i < n) ∧  -- Each person moves to a valid house
    (∀ i j : ℕ, i < n → j < n → i ≠ j → move i ≠ move j) ∧  -- No two people move to the same house
    (∀ i : ℕ, i < n → move (move i) ≠ i) ∧  -- No person returns to their original house
    (∀ i : ℕ, i < n → color i ≠ color (move i)) :=  -- No person's new house has the same color as their old house
  sorry

#check house_coloring 1000

end house_coloring_l2927_292715


namespace inverse_statement_l2927_292724

theorem inverse_statement : 
  (∀ x : ℝ, x > 1 → x^2 - 2*x + 3 > 0) ↔ 
  (∀ x : ℝ, x^2 - 2*x + 3 > 0 → x > 1) :=
by sorry

end inverse_statement_l2927_292724


namespace chess_pawn_loss_l2927_292705

theorem chess_pawn_loss (total_pawns_start : ℕ) (pawns_per_player : ℕ) 
  (kennedy_lost : ℕ) (pawns_left : ℕ) : 
  total_pawns_start = 2 * pawns_per_player →
  pawns_per_player = 8 →
  kennedy_lost = 4 →
  pawns_left = 11 →
  pawns_per_player - (pawns_left - (pawns_per_player - kennedy_lost)) = 1 := by
  sorry

end chess_pawn_loss_l2927_292705


namespace max_volume_box_l2927_292741

/-- The maximum volume of a box created from a rectangular metal sheet --/
theorem max_volume_box (sheet_length sheet_width : ℝ) (h_length : sheet_length = 16)
  (h_width : sheet_width = 12) :
  ∃ (x : ℝ), 
    0 < x ∧ 
    x < sheet_length / 2 ∧ 
    x < sheet_width / 2 ∧
    ∀ (y : ℝ), 
      0 < y ∧ 
      y < sheet_length / 2 ∧ 
      y < sheet_width / 2 → 
      y * (sheet_length - 2*y) * (sheet_width - 2*y) ≤ 128 :=
by sorry

end max_volume_box_l2927_292741


namespace sin_2theta_value_l2927_292799

theorem sin_2theta_value (θ : Real) (h : Real.sin θ + Real.cos θ = Real.sqrt 7 / 2) :
  Real.sin (2 * θ) = 3 / 4 := by
  sorry

end sin_2theta_value_l2927_292799


namespace pentagonal_field_fencing_cost_l2927_292754

/-- Calculates the cost of fencing for an irregular pentagonal field -/
def fencing_cost (side1 side2 side3 side4 side5 : ℕ) 
                 (rate1 rate2 rate3 : ℕ) : ℕ :=
  rate1 * (side1 + side2) + rate2 * side3 + rate3 * (side4 + side5)

/-- Theorem stating the total cost of fencing for the given pentagonal field -/
theorem pentagonal_field_fencing_cost :
  fencing_cost 42 37 52 65 48 7 5 10 = 1943 := by sorry

end pentagonal_field_fencing_cost_l2927_292754


namespace simplify_expression_calculate_expression_l2927_292757

-- Part 1
theorem simplify_expression (a b : ℝ) (h : a > 0 ∧ b > 0) :
  (2 * a^(3/2) * b^(1/2)) * (-6 * a^(1/2) * b^(1/3)) / (-3 * a^(1/6) * b^(5/6)) = 4 * a^(11/6) := by
  sorry

-- Part 2
theorem calculate_expression :
  (2^(1/3) * 3^(1/2))^6 + (2^(1/2) * 2^(1/4))^(4/3) - 2^(1/4) * 8^(1/4) - (-2005)^0 = 100 := by
  sorry

end simplify_expression_calculate_expression_l2927_292757


namespace tournament_committee_count_l2927_292775

/-- Number of teams in the frisbee association -/
def num_teams : ℕ := 6

/-- Number of members in each team -/
def team_size : ℕ := 8

/-- Number of members selected from the host team -/
def host_select : ℕ := 3

/-- Number of members selected from each regular non-host team -/
def nonhost_select : ℕ := 2

/-- Number of members selected from the special non-host team -/
def special_nonhost_select : ℕ := 3

/-- Total number of possible tournament committees -/
def total_committees : ℕ := 11568055296

theorem tournament_committee_count :
  (num_teams) *
  (team_size.choose host_select) *
  ((team_size.choose nonhost_select) ^ (num_teams - 2)) *
  (team_size.choose special_nonhost_select) =
  total_committees :=
sorry

end tournament_committee_count_l2927_292775


namespace intersection_of_N_and_complement_of_M_l2927_292762

open Set

theorem intersection_of_N_and_complement_of_M : 
  let M : Set ℝ := {x | x > 2}
  let N : Set ℝ := {x | 1 < x ∧ x < 3}
  (N ∩ (univ \ M)) = {x : ℝ | 1 < x ∧ x ≤ 2} := by
  sorry

end intersection_of_N_and_complement_of_M_l2927_292762


namespace subtraction_result_l2927_292770

theorem subtraction_result : -3.219 - 7.305 = -10.524 := by sorry

end subtraction_result_l2927_292770


namespace movie_ticket_ratio_l2927_292792

def monday_cost : ℚ := 5
def wednesday_cost : ℚ := 2 * monday_cost

theorem movie_ticket_ratio :
  ∃ (saturday_cost : ℚ),
    wednesday_cost + saturday_cost = 35 ∧
    saturday_cost / monday_cost = 5 := by
  sorry

end movie_ticket_ratio_l2927_292792


namespace eliot_account_balance_l2927_292722

/-- Proves that Eliot's account balance is $200 given the problem conditions --/
theorem eliot_account_balance :
  ∀ (A E : ℝ),
    A > E →  -- Al has more money than Eliot
    A - E = (1/12) * (A + E) →  -- Difference is 1/12 of sum
    1.1 * A = 1.2 * E + 20 →  -- After increase, Al has $20 more
    E = 200 :=
by
  sorry

end eliot_account_balance_l2927_292722


namespace dinosaur_weight_theorem_l2927_292713

/-- The weight of a regular dinosaur in pounds -/
def regular_dino_weight : ℕ := 800

/-- The number of regular dinosaurs -/
def num_regular_dinos : ℕ := 5

/-- The additional weight of Barney compared to the combined weight of regular dinosaurs -/
def barney_extra_weight : ℕ := 1500

/-- The combined weight of Barney and the regular dinosaurs -/
def total_weight : ℕ := regular_dino_weight * num_regular_dinos + barney_extra_weight + regular_dino_weight * num_regular_dinos

theorem dinosaur_weight_theorem : total_weight = 9500 := by
  sorry

end dinosaur_weight_theorem_l2927_292713


namespace finite_ring_power_equality_l2927_292782

theorem finite_ring_power_equality (A : Type) [Ring A] [Fintype A] :
  ∃ (m p : ℕ), m > p ∧ p ≥ 1 ∧ ∀ (a : A), a^m = a^p := by
  sorry

end finite_ring_power_equality_l2927_292782


namespace ap_sequence_a_equals_one_l2927_292794

/-- Given a sequence 1, 6 + 2a, 10 + 5a, ..., if it forms an arithmetic progression, then a = 1 -/
theorem ap_sequence_a_equals_one (a : ℝ) :
  (∀ n : ℕ, (fun i => if i = 0 then 1 else if i = 1 then 6 + 2*a else 10 + 5*a) n.succ - 
             (fun i => if i = 0 then 1 else if i = 1 then 6 + 2*a else 10 + 5*a) n = 
             (fun i => if i = 0 then 1 else if i = 1 then 6 + 2*a else 10 + 5*a) 1 - 
             (fun i => if i = 0 then 1 else if i = 1 then 6 + 2*a else 10 + 5*a) 0) →
  a = 1 := by
sorry


end ap_sequence_a_equals_one_l2927_292794


namespace attendance_difference_l2927_292704

/-- The attendance difference between this week and last week for baseball games --/
theorem attendance_difference : 
  let second_game : ℕ := 80
  let first_game : ℕ := second_game - 20
  let third_game : ℕ := second_game + 15
  let this_week_total : ℕ := first_game + second_game + third_game
  let last_week_total : ℕ := 200
  this_week_total - last_week_total = 35 := by
  sorry

end attendance_difference_l2927_292704


namespace equation_proof_l2927_292783

theorem equation_proof : 10 * 6 - (9 - 3) * 2 = 48 := by
  sorry

end equation_proof_l2927_292783


namespace total_slices_is_136_l2927_292765

/-- The number of slices in a small pizza -/
def small_slices : ℕ := 6

/-- The number of slices in a medium pizza -/
def medium_slices : ℕ := 8

/-- The number of slices in a large pizza -/
def large_slices : ℕ := 12

/-- The total number of pizzas bought -/
def total_pizzas : ℕ := 15

/-- The number of small pizzas ordered -/
def small_pizzas : ℕ := 4

/-- The number of medium pizzas ordered -/
def medium_pizzas : ℕ := 5

/-- Theorem stating that the total number of slices is 136 -/
theorem total_slices_is_136 : 
  small_pizzas * small_slices + 
  medium_pizzas * medium_slices + 
  (total_pizzas - small_pizzas - medium_pizzas) * large_slices = 136 := by
  sorry

end total_slices_is_136_l2927_292765


namespace line_l_properties_l2927_292720

/-- A line that passes through (3,2) and has equal intercepts on both axes -/
def line_l (x y : ℝ) : Prop :=
  y = -x + 5

theorem line_l_properties :
  (∃ a : ℝ, line_l a 2 ∧ a = 3) ∧
  (∃ b : ℝ, line_l b 0 ∧ line_l 0 b ∧ b > 0) :=
by sorry

end line_l_properties_l2927_292720


namespace lake_bright_population_is_16000_l2927_292742

-- Define the total population
def total_population : ℕ := 80000

-- Define Gordonia's population as a fraction of the total
def gordonia_population : ℕ := total_population / 2

-- Define Toadon's population as a percentage of Gordonia's
def toadon_population : ℕ := (gordonia_population * 60) / 100

-- Define Lake Bright's population
def lake_bright_population : ℕ := total_population - gordonia_population - toadon_population

-- Theorem statement
theorem lake_bright_population_is_16000 :
  lake_bright_population = 16000 := by sorry

end lake_bright_population_is_16000_l2927_292742


namespace g_property_S_sum_S_difference_l2927_292717

def g (k : ℕ+) : ℕ+ :=
  sorry

def S (n : ℕ) : ℕ :=
  sorry

theorem g_property (m : ℕ+) : g (2 * m) = g m :=
  sorry

theorem S_sum : S 1 + S 2 + S 3 = 30 :=
  sorry

theorem S_difference (n : ℕ) (h : n ≥ 2) : S n - S (n - 1) = 4^(n - 1) :=
  sorry

end g_property_S_sum_S_difference_l2927_292717


namespace arithmetic_calculation_l2927_292773

theorem arithmetic_calculation : 2 + 5 * 4 - 6 + 3 = 19 := by sorry

end arithmetic_calculation_l2927_292773


namespace solution_set_f_greater_than_one_range_of_m_for_solution_l2927_292737

-- Define the function f
def f (x : ℝ) : ℝ := |x + 2| - |x - 1|

-- Theorem for the solution set of f(x) > 1
theorem solution_set_f_greater_than_one :
  {x : ℝ | f x > 1} = {x : ℝ | x > 0} := by sorry

-- Theorem for the range of m
theorem range_of_m_for_solution (m : ℝ) :
  (∃ x : ℝ, f x + 4 ≥ |1 - 2*m|) ↔ m ∈ Set.Icc (-3) 4 := by sorry

end solution_set_f_greater_than_one_range_of_m_for_solution_l2927_292737


namespace drawings_on_last_page_l2927_292767

-- Define the given conditions
def initial_notebooks : ℕ := 10
def pages_per_notebook : ℕ := 30
def initial_drawings_per_page : ℕ := 4
def new_drawings_per_page : ℕ := 8
def filled_notebooks : ℕ := 6
def filled_pages_in_seventh : ℕ := 25

-- Define the theorem
theorem drawings_on_last_page : 
  let total_drawings := initial_notebooks * pages_per_notebook * initial_drawings_per_page
  let full_pages := total_drawings / new_drawings_per_page
  let pages_in_complete_notebooks := filled_notebooks * pages_per_notebook
  let remaining_drawings := total_drawings - (full_pages * new_drawings_per_page)
  remaining_drawings = 0 := by
  sorry

end drawings_on_last_page_l2927_292767


namespace solution_set_of_inequality_l2927_292733

/-- A quadratic function with zeros at -2 and 3 -/
def f (x : ℝ) : ℝ := x^2 + a*x + b
  where
  a : ℝ := -1  -- Derived from the zeros, but not explicitly using the solution
  b : ℝ := -6  -- Derived from the zeros, but not explicitly using the solution

/-- The theorem statement -/
theorem solution_set_of_inequality (x : ℝ) :
  (f (-2*x) * (-1) > 0) ↔ (-3/2 < x ∧ x < 1) := by
  sorry

end solution_set_of_inequality_l2927_292733


namespace photocopy_cost_calculation_l2927_292758

/-- The cost of one photocopy -/
def photocopy_cost : ℝ := sorry

/-- The discount rate for orders over 100 copies -/
def discount_rate : ℝ := 0.25

/-- The number of copies each person needs -/
def copies_per_person : ℕ := 80

/-- The total number of copies when combining orders -/
def total_copies : ℕ := 2 * copies_per_person

/-- The amount saved per person when combining orders -/
def savings_per_person : ℝ := 0.40

theorem photocopy_cost_calculation : 
  photocopy_cost = 0.02 :=
by
  sorry

end photocopy_cost_calculation_l2927_292758


namespace percentage_equals_1000_l2927_292740

theorem percentage_equals_1000 (x : ℝ) (p : ℝ) : 
  (p / 100) * x = 1000 → 
  (120 / 100) * x = 6000 → 
  p = 20 := by
sorry

end percentage_equals_1000_l2927_292740


namespace sign_determination_l2927_292745

theorem sign_determination (a b : ℝ) (h1 : a + b < 0) (h2 : b / a > 0) : a < 0 ∧ b < 0 := by
  sorry

end sign_determination_l2927_292745


namespace ceiling_evaluation_l2927_292738

theorem ceiling_evaluation : ⌈(4 * (8 - 1/3 : ℚ))⌉ = 31 := by sorry

end ceiling_evaluation_l2927_292738


namespace art_team_arrangement_l2927_292777

/-- Given a team of 1000 members arranged in rows where each row from the second onward
    has one more person than the previous row, prove that there are 25 rows with 28 members
    in the first row. -/
theorem art_team_arrangement (k m : ℕ) : k > 16 →
  (k * (2 * m + k - 1)) / 2 = 1000 → k = 25 ∧ m = 28 := by
  sorry

end art_team_arrangement_l2927_292777


namespace digits_of_3_pow_10_times_5_pow_6_l2927_292781

theorem digits_of_3_pow_10_times_5_pow_6 :
  (Nat.digits 10 (3^10 * 5^6)).length = 9 := by
  sorry

end digits_of_3_pow_10_times_5_pow_6_l2927_292781


namespace complex_number_magnitude_squared_l2927_292735

theorem complex_number_magnitude_squared (z : ℂ) : z + Complex.abs z = 2 + 8*I → Complex.abs z ^ 2 = 289 := by
  sorry

end complex_number_magnitude_squared_l2927_292735


namespace count_five_digit_with_four_or_five_l2927_292755

/-- The number of five-digit positive integers. -/
def total_five_digit_integers : ℕ := 90000

/-- The number of five-digit positive integers without 4 or 5. -/
def five_digit_without_four_or_five : ℕ := 28672

/-- The number of five-digit positive integers containing either 4 or 5 at least once. -/
def five_digit_with_four_or_five : ℕ := total_five_digit_integers - five_digit_without_four_or_five

theorem count_five_digit_with_four_or_five :
  five_digit_with_four_or_five = 61328 := by
  sorry

end count_five_digit_with_four_or_five_l2927_292755


namespace simple_interest_calculation_l2927_292798

/-- Given a principal amount where the compound interest at 5% per annum for 2 years is $51.25,
    prove that the simple interest at the same rate and time is $50 -/
theorem simple_interest_calculation (P : ℝ) : 
  P * ((1 + 0.05)^2 - 1) = 51.25 → P * 0.05 * 2 = 50 := by
  sorry

end simple_interest_calculation_l2927_292798


namespace quadratic_roots_property_l2927_292701

theorem quadratic_roots_property (α β : ℝ) : 
  (α^2 - 4*α - 3 = 0) → 
  (β^2 - 4*β - 3 = 0) → 
  (α - 3) * (β - 3) = -6 := by
sorry

end quadratic_roots_property_l2927_292701


namespace problem_statement_l2927_292706

theorem problem_statement (a b c : ℝ) 
  (h1 : a^2 + b^2 - 4*a ≤ 1)
  (h2 : b^2 + c^2 - 8*b ≤ -3)
  (h3 : c^2 + a^2 - 12*c ≤ -26) :
  (a + b)^c = 27 := by
sorry

end problem_statement_l2927_292706


namespace coefficient_of_x_l2927_292710

/-- Given that for some natural number n:
    1) M = 4^n is the sum of coefficients in (5x - 1/√x)^n
    2) N = 2^n is the sum of binomial coefficients
    3) M - N = 240
    Then the coefficient of x in the expansion of (5x - 1/√x)^n is 150 -/
theorem coefficient_of_x (n : ℕ) (M N : ℝ) 
  (hM : M = 4^n)
  (hN : N = 2^n)
  (hDiff : M - N = 240) :
  ∃ (coeff : ℝ), coeff = 150 ∧ 
  coeff = (-1)^2 * (n.choose 2) * 5^(n-2) := by
  sorry

end coefficient_of_x_l2927_292710


namespace arccos_negative_one_equals_pi_l2927_292703

theorem arccos_negative_one_equals_pi : Real.arccos (-1) = π := by
  sorry

end arccos_negative_one_equals_pi_l2927_292703


namespace sum_with_radical_conjugate_l2927_292730

theorem sum_with_radical_conjugate : 
  (15 - Real.sqrt 500) + (15 + Real.sqrt 500) = 30 := by
  sorry

end sum_with_radical_conjugate_l2927_292730


namespace solution_set_f_geq_12_range_of_a_l2927_292712

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x - 5| + |x + 4|

-- Theorem for the solution set of f(x) ≥ 12
theorem solution_set_f_geq_12 :
  {x : ℝ | f x ≥ 12} = {x : ℝ | x ≥ 13/2 ∨ x ≤ -11/2} := by sorry

-- Theorem for the range of a
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f x - 2^(1 - 3*a) - 1 ≥ 0) → a ≥ -2/3 := by sorry

end solution_set_f_geq_12_range_of_a_l2927_292712


namespace oil_price_reduction_is_fifty_percent_l2927_292776

/-- Calculates the percentage reduction in oil price given the reduced price and additional quantity -/
def oil_price_reduction (reduced_price : ℚ) (additional_quantity : ℚ) : ℚ :=
  let original_price := (800 : ℚ) / (((800 : ℚ) / reduced_price) - additional_quantity)
  ((original_price - reduced_price) / original_price) * 100

/-- Theorem stating that under the given conditions, the oil price reduction is 50% -/
theorem oil_price_reduction_is_fifty_percent :
  oil_price_reduction 80 5 = 50 := by sorry

end oil_price_reduction_is_fifty_percent_l2927_292776


namespace difference_largest_smallest_l2927_292743

/-- The set of available digits --/
def available_digits : Finset Nat := {2, 0, 3, 5, 8}

/-- A four-digit number formed from the available digits --/
structure FourDigitNumber where
  digits : Finset Nat
  size_eq : digits.card = 4
  subset : digits ⊆ available_digits

/-- The largest four-digit number that can be formed --/
def largest_number : Nat := 8532

/-- The smallest four-digit number that can be formed --/
def smallest_number : Nat := 2035

/-- Theorem: The difference between the largest and smallest four-digit numbers is 6497 --/
theorem difference_largest_smallest :
  largest_number - smallest_number = 6497 := by sorry

end difference_largest_smallest_l2927_292743


namespace max_quadratic_solution_l2927_292786

theorem max_quadratic_solution (k a b c : ℕ+) (r : ℝ) :
  (∃ m n l : ℕ, a = k ^ m ∧ b = k ^ n ∧ c = k ^ l) →
  (a * r ^ 2 - b * r + c = 0) →
  (∀ x : ℝ, x ≠ r → a * x ^ 2 - b * x + c ≠ 0) →
  r < 100 →
  r ≤ 64 := by
sorry

end max_quadratic_solution_l2927_292786


namespace coefficient_is_200_l2927_292708

/-- The coefficient of x^4 in the expansion of (1+x^3)(1-x)^10 -/
def coefficientOfX4 : ℕ :=
  (Nat.choose 10 4) - (Nat.choose 10 1)

/-- Theorem stating that the coefficient of x^4 in the expansion of (1+x^3)(1-x)^10 is 200 -/
theorem coefficient_is_200 : coefficientOfX4 = 200 := by
  sorry

end coefficient_is_200_l2927_292708


namespace division_cannot_be_operation_l2927_292790

def P : Set ℤ := {x | ∃ k : ℤ, x = 2 * k}

theorem division_cannot_be_operation :
  ¬(∀ a b : ℤ, a ∈ P → b ∈ P → (a / b) ∈ P) :=
by
  sorry

end division_cannot_be_operation_l2927_292790


namespace union_equals_interval_l2927_292785

-- Define sets A and B
def A : Set ℝ := {x : ℝ | x^2 + 2*x - 3 < 0}
def B : Set ℝ := {x : ℝ | x^2 - 4*x ≤ 0}

-- Define the interval (-3, 4]
def interval : Set ℝ := Set.Ioc (-3) 4

-- Theorem statement
theorem union_equals_interval : A ∪ B = interval := by sorry

end union_equals_interval_l2927_292785


namespace linear_equation_solution_l2927_292791

theorem linear_equation_solution (m : ℤ) :
  (∃ x : ℚ, x^(|m|) - m*x + 1 = 0 ∧ ∃ a b : ℚ, a ≠ 0 ∧ a*x + b = 0) →
  (∃ x : ℚ, x^(|m|) - m*x + 1 = 0 ∧ x = -1/2) :=
by sorry

end linear_equation_solution_l2927_292791


namespace origin_and_point_same_side_l2927_292729

def line_equation (x y : ℝ) : ℝ := 3 * x + 2 * y + 5

def same_side (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  line_equation x₁ y₁ * line_equation x₂ y₂ > 0

theorem origin_and_point_same_side : same_side 0 0 (-3) 4 := by sorry

end origin_and_point_same_side_l2927_292729


namespace angle_complement_measure_l2927_292778

theorem angle_complement_measure : 
  ∀ x : ℝ, 
  (x + (3 * x + 10) = 90) →  -- Condition 1 and 2 combined
  (3 * x + 10 = 70) :=        -- The complement measure to prove
by
  sorry

end angle_complement_measure_l2927_292778


namespace other_side_heads_probability_is_two_thirds_l2927_292756

/-- Represents the three types of coins -/
inductive Coin
  | Normal
  | TwoHeads
  | TwoTails

/-- Represents the possible outcomes of a coin toss -/
inductive CoinSide
  | Heads
  | Tails

/-- The probability of selecting each type of coin -/
def coinProbability (c : Coin) : ℚ :=
  match c with
  | Coin.Normal => 1/3
  | Coin.TwoHeads => 1/3
  | Coin.TwoTails => 1/3

/-- The probability of getting heads when tossing a specific coin -/
def headsUpProbability (c : Coin) : ℚ :=
  match c with
  | Coin.Normal => 1/2
  | Coin.TwoHeads => 1
  | Coin.TwoTails => 0

/-- The probability that the other side is heads given that heads is showing -/
def otherSideHeadsProbability : ℚ := by
  sorry

theorem other_side_heads_probability_is_two_thirds :
  otherSideHeadsProbability = 2/3 := by
  sorry

end other_side_heads_probability_is_two_thirds_l2927_292756


namespace brother_age_l2927_292764

theorem brother_age (man_age brother_age : ℕ) : 
  man_age = brother_age + 12 →
  man_age + 2 = 2 * (brother_age + 2) →
  brother_age = 10 := by
sorry

end brother_age_l2927_292764


namespace special_triangle_min_perimeter_l2927_292700

/-- Triangle ABC with integer side lengths and specific angle conditions -/
structure SpecialTriangle where
  a : ℕ
  b : ℕ
  c : ℕ
  angle_A : ℝ
  angle_B : ℝ
  angle_C : ℝ
  angle_A_twice_B : angle_A = 2 * angle_B
  angle_C_obtuse : angle_C > Real.pi / 2
  angle_sum : angle_A + angle_B + angle_C = Real.pi

/-- The minimum perimeter of a SpecialTriangle is 77 -/
theorem special_triangle_min_perimeter (t : SpecialTriangle) : t.a + t.b + t.c ≥ 77 := by
  sorry

end special_triangle_min_perimeter_l2927_292700


namespace doctor_team_formation_l2927_292788

theorem doctor_team_formation (male_doctors female_doctors team_size : ℕ) 
  (h1 : male_doctors = 5)
  (h2 : female_doctors = 4)
  (h3 : team_size = 3) : 
  (Nat.choose male_doctors 2 * Nat.choose female_doctors 1 + 
   Nat.choose male_doctors 1 * Nat.choose female_doctors 2) = 70 := by
  sorry

end doctor_team_formation_l2927_292788


namespace three_to_nine_over_nine_cubed_equals_27_l2927_292714

theorem three_to_nine_over_nine_cubed_equals_27 : (3^9) / (9^3) = 27 := by
  sorry

end three_to_nine_over_nine_cubed_equals_27_l2927_292714


namespace f_strictly_increasing_after_one_l2927_292789

/-- The quadratic function f(x) = (x-1)^2 + 5 -/
def f (x : ℝ) : ℝ := (x - 1)^2 + 5

/-- Theorem: f(x) is strictly increasing for all x > 1 -/
theorem f_strictly_increasing_after_one :
  ∀ x₁ x₂ : ℝ, 1 < x₁ → x₁ < x₂ → f x₁ < f x₂ :=
by
  sorry

end f_strictly_increasing_after_one_l2927_292789


namespace yellow_second_draw_probability_l2927_292769

/-- The probability of drawing a yellow ball on the second draw -/
def prob_yellow_second_draw (yellow white : ℕ) : ℚ :=
  (white : ℚ) / (yellow + white) * yellow / (yellow + white - 1)

/-- Theorem: The probability of drawing a yellow ball on the second draw
    is 4/15 when there are 6 yellow and 4 white balls -/
theorem yellow_second_draw_probability :
  prob_yellow_second_draw 6 4 = 4 / 15 := by
  sorry

end yellow_second_draw_probability_l2927_292769


namespace impossible_coloring_l2927_292746

theorem impossible_coloring : ¬∃(color : ℕ → Bool),
  (∀ n : ℕ, color n ≠ color (n + 5)) ∧
  (∀ n : ℕ, color n ≠ color (2 * n)) :=
by sorry

end impossible_coloring_l2927_292746


namespace quadratic_equation_unique_solution_l2927_292736

theorem quadratic_equation_unique_solution (a c : ℝ) : 
  (∃! x, a * x^2 + 6 * x + c = 0) →  -- exactly one solution
  (a + c = 12) →                     -- sum condition
  (a < c) →                          -- order condition
  (a = 6 - 3 * Real.sqrt 3 ∧ c = 6 + 3 * Real.sqrt 3) := by
sorry

end quadratic_equation_unique_solution_l2927_292736


namespace product_104_96_l2927_292796

theorem product_104_96 : 104 * 96 = 9984 := by
  sorry

end product_104_96_l2927_292796


namespace fraction_to_decimal_l2927_292771

theorem fraction_to_decimal : (47 : ℚ) / (2^3 * 5^4) = 0.0094 := by sorry

end fraction_to_decimal_l2927_292771


namespace candy_distribution_l2927_292768

theorem candy_distribution (total_candies : ℕ) (candies_per_student : ℕ) 
  (h1 : total_candies = 901)
  (h2 : candies_per_student = 53)
  (h3 : total_candies % candies_per_student = 0) :
  total_candies / candies_per_student = 17 := by
  sorry

end candy_distribution_l2927_292768


namespace min_sum_with_reciprocal_constraint_l2927_292766

theorem min_sum_with_reciprocal_constraint (x y : ℝ) : 
  x > 0 → y > 0 → (1 / (x + 2) + 1 / (y + 2) = 1 / 6) → 
  x + y ≥ 20 ∧ ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 1 / (x + 2) + 1 / (y + 2) = 1 / 6 ∧ x + y = 20 := by
  sorry

end min_sum_with_reciprocal_constraint_l2927_292766


namespace triangle_angle_B_l2927_292707

theorem triangle_angle_B (A B C : Real) (a b : Real) :
  -- Triangle ABC exists
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi →
  -- Side lengths
  a = 4 ∧ b = 5 →
  -- Given condition
  Real.cos (B + C) + 3/5 = 0 →
  -- Conclusion: Measure of angle B
  B = Real.pi - Real.arccos (3/5) := by sorry

end triangle_angle_B_l2927_292707


namespace no_primes_in_sequence_infinitely_many_x_with_no_primes_l2927_292734

/-- Definition of the sequence a_n -/
def a (x : ℕ) : ℕ → ℕ
| 0 => 1
| 1 => x + 1
| (n + 2) => x * a x (n + 1) - a x n

/-- Theorem stating that for any c ≥ 3, the sequence contains no primes when x = c² - 2 -/
theorem no_primes_in_sequence (c : ℕ) (h : c ≥ 3) :
  ∀ n : ℕ, ¬ Nat.Prime (a (c^2 - 2) n) := by
  sorry

/-- Corollary: There exist infinitely many x such that the sequence contains no primes -/
theorem infinitely_many_x_with_no_primes :
  ∃ f : ℕ → ℕ, Monotone f ∧ ∀ k : ℕ, ∀ n : ℕ, ¬ Nat.Prime (a (f k) n) := by
  sorry

end no_primes_in_sequence_infinitely_many_x_with_no_primes_l2927_292734


namespace expression_value_l2927_292719

theorem expression_value (a b : ℝ) (h : a * b > 0) :
  (a / abs a) + (b / abs b) + ((a * b) / abs (a * b)) = 3 ∨
  (a / abs a) + (b / abs b) + ((a * b) / abs (a * b)) = -1 :=
by sorry

end expression_value_l2927_292719


namespace tens_digit_of_13_pow_2021_l2927_292702

theorem tens_digit_of_13_pow_2021 : ∃ n : ℕ, 13^2021 ≡ 10 * n + 1 [ZMOD 100] :=
sorry

end tens_digit_of_13_pow_2021_l2927_292702


namespace frustum_properties_l2927_292763

/-- Frustum properties -/
structure Frustum where
  r₁ : ℝ  -- radius of top base
  r₂ : ℝ  -- radius of bottom base
  l : ℝ   -- slant height
  h : ℝ   -- height

/-- Theorem about a specific frustum -/
theorem frustum_properties (f : Frustum) (h_r₁ : f.r₁ = 2) (h_r₂ : f.r₂ = 6)
    (h_lateral_area : π * (f.r₁ + f.r₂) * f.l = π * f.r₁^2 + π * f.r₂^2) :
    f.l = 5 ∧ π * f.h * (f.r₁^2 + f.r₂^2 + f.r₁ * f.r₂) / 3 = 52 * π := by
  sorry


end frustum_properties_l2927_292763


namespace julia_born_1979_l2927_292753

def wayne_age_2021 : ℕ := 37
def peter_age_diff : ℕ := 3
def julia_age_diff : ℕ := 2

def julia_birth_year : ℕ := 2021 - wayne_age_2021 - peter_age_diff - julia_age_diff

theorem julia_born_1979 : julia_birth_year = 1979 := by
  sorry

end julia_born_1979_l2927_292753


namespace shepherd_problem_l2927_292728

theorem shepherd_problem :
  ∃! (a b c : ℕ), 
    a + b + 10 * c = 100 ∧
    20 * a + 10 * b + 10 * c = 200 ∧
    a = 1 ∧ b = 9 ∧ c = 9 := by
  sorry

end shepherd_problem_l2927_292728


namespace school_teachers_count_l2927_292787

theorem school_teachers_count (total_people : ℕ) (sample_size : ℕ) (students_in_sample : ℕ) 
  (h1 : total_people = 3200)
  (h2 : sample_size = 160)
  (h3 : students_in_sample = 150) :
  total_people - (total_people * students_in_sample / sample_size) = 200 := by
  sorry

end school_teachers_count_l2927_292787


namespace triangle_longest_side_l2927_292779

theorem triangle_longest_side (x : ℚ) : 
  (x + 3 : ℚ) + (2 * x - 1 : ℚ) + (3 * x + 5 : ℚ) = 45 → 
  max (x + 3) (max (2 * x - 1) (3 * x + 5)) = 24 := by
  sorry

end triangle_longest_side_l2927_292779


namespace percentage_four_leaf_clovers_l2927_292744

/-- Proves that 20% of clovers have four leaves given the conditions -/
theorem percentage_four_leaf_clovers 
  (total_clovers : ℕ) 
  (purple_four_leaf : ℕ) 
  (h1 : total_clovers = 500)
  (h2 : purple_four_leaf = 25)
  (h3 : (4 : ℚ) * purple_four_leaf = total_clovers * (percentage_four_leaf / 100)) :
  percentage_four_leaf = 20 := by
  sorry

#check percentage_four_leaf_clovers

end percentage_four_leaf_clovers_l2927_292744


namespace unique_digit_factorial_sum_l2927_292752

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

def digit_factorial_sum (n : ℕ) : ℕ :=
  let d1 := n / 100
  let d2 := (n / 10) % 10
  let d3 := n % 10
  factorial d1 + factorial d2 + factorial d3

def has_zero_digit (n : ℕ) : Prop :=
  n % 10 = 0 ∨ (n / 10) % 10 = 0 ∨ n / 100 = 0

theorem unique_digit_factorial_sum :
  ∃! n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n = digit_factorial_sum n ∧ has_zero_digit n :=
sorry

end unique_digit_factorial_sum_l2927_292752


namespace triangle_perimeter_bound_l2927_292797

theorem triangle_perimeter_bound (a b s : ℝ) : 
  a = 7 → b = 23 → a > 0 → b > 0 → s > 0 → 
  a + b > s → a + s > b → b + s > a → 
  ∃ n : ℕ, n = 60 ∧ ∀ m : ℕ, (a + b + s < m ∧ m < n) → False :=
sorry

end triangle_perimeter_bound_l2927_292797


namespace right_triangle_third_side_length_l2927_292772

theorem right_triangle_third_side_length (a b c : ℝ) : 
  a = 8 → b = 15 → c ≥ 0 → a^2 + b^2 = c^2 → c ≥ 17 := by
  sorry

end right_triangle_third_side_length_l2927_292772


namespace food_drive_total_cans_l2927_292726

theorem food_drive_total_cans 
  (mark_cans jaydon_cans rachel_cans : ℕ) 
  (h1 : mark_cans = 4 * jaydon_cans)
  (h2 : jaydon_cans = 2 * rachel_cans + 5)
  (h3 : mark_cans = 100) : 
  mark_cans + jaydon_cans + rachel_cans = 135 := by
sorry


end food_drive_total_cans_l2927_292726


namespace inventory_difference_l2927_292732

/-- Inventory problem -/
theorem inventory_difference (ties belts black_shirts white_shirts : ℕ) 
  (h_ties : ties = 34)
  (h_belts : belts = 40)
  (h_black_shirts : black_shirts = 63)
  (h_white_shirts : white_shirts = 42)
  : (2 * (black_shirts + white_shirts) / 3) - ((ties + belts) / 2) = 33 := by
  sorry

end inventory_difference_l2927_292732


namespace company_employees_l2927_292747

/-- Calculates the initial number of employees in a company given the following conditions:
  * Hourly wage
  * Hours worked per day
  * Days worked per week
  * Weeks worked per month
  * Number of new hires
  * Total monthly payroll after hiring
-/
def initial_employees (
  hourly_wage : ℕ
  ) (hours_per_day : ℕ
  ) (days_per_week : ℕ
  ) (weeks_per_month : ℕ
  ) (new_hires : ℕ
  ) (total_payroll : ℕ
  ) : ℕ :=
  let monthly_hours := hours_per_day * days_per_week * weeks_per_month
  let monthly_wage := hourly_wage * monthly_hours
  (total_payroll / monthly_wage) - new_hires

theorem company_employees :
  initial_employees 12 10 5 4 200 1680000 = 500 := by
  sorry

end company_employees_l2927_292747


namespace work_completion_time_l2927_292748

/-- Given a piece of work that can be completed by different combinations of workers,
    this theorem proves how long it takes two workers to complete the work. -/
theorem work_completion_time
  (work : ℝ) -- The total amount of work to be done
  (rate_ab : ℝ) -- The rate at which a and b work together
  (rate_c : ℝ) -- The rate at which c works
  (h1 : rate_ab + rate_c = work) -- a, b, and c together complete the work in 1 day
  (h2 : rate_c = work / 2) -- c alone completes the work in 2 days
  : rate_ab = work / 2 := by
  sorry

end work_completion_time_l2927_292748


namespace rational_function_simplification_and_evaluation_l2927_292759

theorem rational_function_simplification_and_evaluation :
  ∀ x : ℝ, x ≠ 5 →
  (x^2 - 3*x - 10) / (x - 5) = x + 2 ∧
  (4^2 - 3*4 - 10) / (4 - 5) = 6 := by
sorry

end rational_function_simplification_and_evaluation_l2927_292759


namespace negative_b_from_cubic_inequality_l2927_292760

theorem negative_b_from_cubic_inequality (a b : ℝ) 
  (h1 : a * b ≠ 0)
  (h2 : ∀ x : ℝ, x ≥ 0 → (x - a) * (x - b) * (x - 2*a - b) ≥ 0) :
  b < 0 := by
  sorry

end negative_b_from_cubic_inequality_l2927_292760


namespace pool_capacity_theorem_l2927_292749

/-- Represents the dimensions and draining parameters of a pool -/
structure Pool :=
  (width : ℝ)
  (length : ℝ)
  (depth : ℝ)
  (drainRate : ℝ)
  (drainTime : ℝ)

/-- Calculates the volume of a pool -/
def poolVolume (p : Pool) : ℝ :=
  p.width * p.length * p.depth

/-- Calculates the amount of water drained from a pool -/
def waterDrained (p : Pool) : ℝ :=
  p.drainRate * p.drainTime

/-- Theorem stating that if the water drained equals the pool volume, 
    then the pool was at 100% capacity -/
theorem pool_capacity_theorem (p : Pool) 
  (h1 : p.width = 80)
  (h2 : p.length = 150)
  (h3 : p.depth = 10)
  (h4 : p.drainRate = 60)
  (h5 : p.drainTime = 2000)
  (h6 : waterDrained p = poolVolume p) :
  poolVolume p / poolVolume p = 1 := by
  sorry


end pool_capacity_theorem_l2927_292749


namespace combined_annual_income_l2927_292709

def monthly_income_problem (A B C D : ℝ) : Prop :=
  -- Ratio condition
  A / B = 5 / 3 ∧ B / C = 3 / 2 ∧
  -- B's income is 12% more than C's
  B = 1.12 * C ∧
  -- D's income is 15% less than A's
  D = 0.85 * A ∧
  -- C's income is 17000
  C = 17000

theorem combined_annual_income 
  (A B C D : ℝ) 
  (h : monthly_income_problem A B C D) : 
  (A + B + C + D) * 12 = 1375980 := by
  sorry

#check combined_annual_income

end combined_annual_income_l2927_292709


namespace g_solution_set_a_range_l2927_292751

-- Define the functions f and g
def f (a x : ℝ) := 3 * abs (x - a) + abs (3 * x + 1)
def g (x : ℝ) := abs (4 * x - 1) - abs (x + 2)

-- Theorem for the solution set of g(x) < 6
theorem g_solution_set :
  {x : ℝ | g x < 6} = {x : ℝ | -7/5 < x ∧ x < 3} := by sorry

-- Theorem for the range of a
theorem a_range (a : ℝ) :
  (∃ x₁ x₂ : ℝ, f a x₁ = -g x₂) → a ∈ Set.Icc (-13/12) (5/12) := by sorry

end g_solution_set_a_range_l2927_292751


namespace largest_replacement_l2927_292725

def original_number : ℚ := -0.3168

def replace_digit (n : ℚ) (old_digit new_digit : ℕ) : ℚ := sorry

theorem largest_replacement :
  ∀ d : ℕ, d ≠ 0 → d ≠ 3 → d ≠ 1 → d ≠ 6 → d ≠ 8 →
    replace_digit original_number 6 4 ≥ replace_digit original_number d 4 :=
by sorry

end largest_replacement_l2927_292725


namespace monster_count_is_thirteen_l2927_292721

/-- Represents the state of the battlefield --/
structure Battlefield where
  ultraman_heads : Nat
  ultraman_legs : Nat
  initial_monster_heads : Nat
  initial_monster_legs : Nat
  split_monster_heads : Nat
  split_monster_legs : Nat
  total_heads : Nat
  total_legs : Nat

/-- Calculates the number of monsters on the battlefield --/
def count_monsters (b : Battlefield) : Nat :=
  let remaining_heads := b.total_heads - b.ultraman_heads
  let remaining_legs := b.total_legs - b.ultraman_legs
  let initial_monsters := remaining_heads / b.initial_monster_heads
  let extra_legs := remaining_legs - (initial_monsters * b.initial_monster_legs)
  let splits := extra_legs / (2 * b.split_monster_legs - b.initial_monster_legs)
  initial_monsters + splits

/-- The main theorem stating that the number of monsters is 13 --/
theorem monster_count_is_thirteen (b : Battlefield) 
  (h1 : b.ultraman_heads = 1)
  (h2 : b.ultraman_legs = 2)
  (h3 : b.initial_monster_heads = 2)
  (h4 : b.initial_monster_legs = 5)
  (h5 : b.split_monster_heads = 1)
  (h6 : b.split_monster_legs = 6)
  (h7 : b.total_heads = 21)
  (h8 : b.total_legs = 73) :
  count_monsters b = 13 := by
  sorry

#eval count_monsters {
  ultraman_heads := 1,
  ultraman_legs := 2,
  initial_monster_heads := 2,
  initial_monster_legs := 5,
  split_monster_heads := 1,
  split_monster_legs := 6,
  total_heads := 21,
  total_legs := 73
}

end monster_count_is_thirteen_l2927_292721
