import Mathlib

namespace distance_from_A_to_D_l272_272671

theorem distance_from_A_to_D 
  (A B C D : Type)
  (east_of : B → A)
  (north_of : C → B)
  (distance_AC : Real)
  (angle_BAC : ℝ)
  (north_of_D : D → C)
  (distance_CD : Real) : 
  distance_AC = 5 * Real.sqrt 5 → 
  angle_BAC = 60 → 
  distance_CD = 15 → 
  ∃ (AD : Real), AD =
    Real.sqrt (
      (5 * Real.sqrt 15 / 2) ^ 2 + 
      (5 * Real.sqrt 5 / 2 + 15) ^ 2
    ) :=
by
  intros
  sorry


end distance_from_A_to_D_l272_272671


namespace triangle_condition_isosceles_or_right_l272_272950

theorem triangle_condition_isosceles_or_right {A B C : ℝ} {a b c : ℝ} 
  (h_triangle : A + B + C = π) (h_cos_eq : a * Real.cos A = b * Real.cos B) : 
  (A = B) ∨ (A + B = π / 2) :=
sorry

end triangle_condition_isosceles_or_right_l272_272950


namespace initial_marbles_l272_272564

theorem initial_marbles (M : ℕ) :
  (M - 5) % 3 = 0 ∧ M = 65 :=
by
  sorry

end initial_marbles_l272_272564


namespace circle_line_distance_difference_l272_272789

/-- We define the given circle and line and prove the difference between maximum and minimum distances
    from any point on the circle to the line is 5√2. -/
theorem circle_line_distance_difference :
  (∀ (x y : ℝ), x^2 + y^2 - 4*x - 4*y - 10 = 0) →
  (∀ (x y : ℝ), x + y - 8 = 0) →
  ∃ (d : ℝ), d = 5 * Real.sqrt 2 :=
by
  sorry

end circle_line_distance_difference_l272_272789


namespace necessarily_positive_y_plus_z_l272_272383

-- Given conditions
variables {x y z : ℝ}

-- Assert the conditions
axiom hx : 0 < x ∧ x < 1
axiom hy : -1 < y ∧ y < 0
axiom hz : 1 < z ∧ z < 2

-- Prove that y + z is necessarily positive
theorem necessarily_positive_y_plus_z : y + z > 0 :=
by
  sorry

end necessarily_positive_y_plus_z_l272_272383


namespace polynomial_simplification_l272_272384

theorem polynomial_simplification (x : ℤ) :
  (5 * x ^ 12 + 8 * x ^ 11 + 10 * x ^ 9) + (3 * x ^ 13 + 2 * x ^ 12 + x ^ 11 + 6 * x ^ 9 + 7 * x ^ 5 + 8 * x ^ 2 + 9) =
  3 * x ^ 13 + 7 * x ^ 12 + 9 * x ^ 11 + 16 * x ^ 9 + 7 * x ^ 5 + 8 * x ^ 2 + 9 :=
by
  sorry

end polynomial_simplification_l272_272384


namespace find_interest_rate_l272_272448

-- Define the given conditions
variables (P A t n CI : ℝ) (r : ℝ)

-- Suppose given conditions
variables (hP : P = 1200)
variables (hCI : CI = 240)
variables (hA : A = P + CI)
variables (ht : t = 1)
variables (hn : n = 1)

-- Define the statement to prove 
theorem find_interest_rate : (A = P * (1 + r / n)^(n * t)) → (r = 0.2) :=
by
  sorry

end find_interest_rate_l272_272448


namespace ratio_of_3_numbers_l272_272076

variable (A B C : ℕ)
variable (k : ℕ)

theorem ratio_of_3_numbers (h₁ : A = 5 * k) (h₂ : B = k) (h₃ : C = 4 * k) (h_sum : A + B + C = 1000) : C = 400 :=
  sorry

end ratio_of_3_numbers_l272_272076


namespace function_is_linear_l272_272786

noncomputable def f : ℕ → ℕ :=
  λ n => n + 1

axiom f_at_0 : f 0 = 1
axiom f_at_2016 : f 2016 = 2017
axiom f_equation : ∀ n : ℕ, f (f n) + f n = 2 * n + 3

theorem function_is_linear : ∀ n : ℕ, f n = n + 1 :=
by
  intro n
  sorry

end function_is_linear_l272_272786


namespace dice_sum_24_l272_272760

noncomputable def probability_of_sum_24 : ℚ :=
  let die_probability := (1 : ℚ) / 6
  in die_probability ^ 4

theorem dice_sum_24 :
  ∑ x in {x | x ∈ {1, 2, 3, 4, 5, 6} ∧ x = 6} = 24 → probability_of_sum_24 = 1 / 1296 :=
sorry

end dice_sum_24_l272_272760


namespace area_of_fig_between_x1_and_x2_l272_272225

noncomputable def area_under_curve_x2 (a b : ℝ) : ℝ :=
∫ x in a..b, x^2

theorem area_of_fig_between_x1_and_x2 :
  area_under_curve_x2 1 2 = 7 / 3 := by
  sorry

end area_of_fig_between_x1_and_x2_l272_272225


namespace Mr_Deane_filled_today_l272_272210

theorem Mr_Deane_filled_today :
  ∀ (x : ℝ),
    (25 * (1.4 - 0.4) + 1.4 * x = 39) →
    x = 10 :=
by
  intros x h
  sorry

end Mr_Deane_filled_today_l272_272210


namespace james_profit_l272_272360

theorem james_profit
  (tickets_bought : ℕ)
  (cost_per_ticket : ℕ)
  (percentage_winning : ℕ)
  (winning_tickets_percentage_5dollars : ℕ)
  (grand_prize : ℕ)
  (average_other_prizes : ℕ)
  (total_tickets : ℕ)
  (total_cost : ℕ)
  (winning_tickets : ℕ)
  (tickets_prize_5dollars : ℕ)
  (amount_won_5dollars : ℕ)
  (other_winning_tickets : ℕ)
  (other_tickets_prize : ℕ)
  (total_winning_amount : ℕ)
  (profit : ℕ) :

  tickets_bought = 200 →
  cost_per_ticket = 2 →
  percentage_winning = 20 →
  winning_tickets_percentage_5dollars = 80 →
  grand_prize = 5000 →
  average_other_prizes = 10 →
  total_tickets = tickets_bought →
  total_cost = total_tickets * cost_per_ticket →
  winning_tickets = (percentage_winning * total_tickets) / 100 →
  tickets_prize_5dollars = (winning_tickets_percentage_5dollars * winning_tickets) / 100 →
  amount_won_5dollars = tickets_prize_5dollars * 5 →
  other_winning_tickets = winning_tickets - 1 →
  other_tickets_prize = (other_winning_tickets - tickets_prize_5dollars) * average_other_prizes →
  total_winning_amount = amount_won_5dollars + grand_prize + other_tickets_prize →
  profit = total_winning_amount - total_cost →
  profit = 4830 := 
sorry

end james_profit_l272_272360


namespace min_breaks_for_square_12_can_form_square_15_l272_272458

-- Definitions and conditions for case n = 12
def stick_lengths_12 := (finset.range 12).map (λ i, i + 1)
def total_length_12 := stick_lengths_12.sum

-- Proof problem for n = 12
theorem min_breaks_for_square_12 : 
  ∃ min_breaks : ℕ, total_length_12 + min_breaks * 2 ∈ {k | k % 4 = 0} ∧ min_breaks = 2 :=
sorry

-- Definitions and conditions for case n = 15
def stick_lengths_15 := (finset.range 15).map (λ i, i + 1)
def total_length_15 := stick_lengths_15.sum

-- Proof problem for n = 15
theorem can_form_square_15 : 
  total_length_15 % 4 = 0 :=
sorry

end min_breaks_for_square_12_can_form_square_15_l272_272458


namespace red_car_speed_l272_272306

noncomputable def speed_blue : ℕ := 80
noncomputable def speed_green : ℕ := 8 * speed_blue
noncomputable def speed_red : ℕ := 2 * speed_green

theorem red_car_speed : speed_red = 1280 := by
  unfold speed_red
  unfold speed_green
  unfold speed_blue
  sorry

end red_car_speed_l272_272306


namespace length_of_bridge_l272_272108

theorem length_of_bridge
    (speed_kmh : Real)
    (time_minutes : Real)
    (speed_cond : speed_kmh = 5)
    (time_cond : time_minutes = 15) :
    let speed_mmin := speed_kmh * 1000 / 60
    let distance_m := speed_mmin * time_minutes
    distance_m = 1250 :=
by
    sorry

end length_of_bridge_l272_272108


namespace Lizzie_has_27_crayons_l272_272206

variable (Lizzie Bobbie Billie : ℕ)

axiom Billie_crayons : Billie = 18
axiom Bobbie_crayons : Bobbie = 3 * Billie
axiom Lizzie_crayons : Lizzie = Bobbie / 2

theorem Lizzie_has_27_crayons : Lizzie = 27 :=
by
  sorry

end Lizzie_has_27_crayons_l272_272206


namespace sqrt_expression_result_l272_272751

theorem sqrt_expression_result :
  (Real.sqrt (16 - 8 * Real.sqrt 3) - Real.sqrt (16 + 8 * Real.sqrt 3)) ^ 2 = 48 := 
sorry

end sqrt_expression_result_l272_272751


namespace total_stars_correct_l272_272999

-- Define the number of gold stars Shelby earned each day
def monday_stars : ℕ := 4
def tuesday_stars : ℕ := 7
def wednesday_stars : ℕ := 3
def thursday_stars : ℕ := 8
def friday_stars : ℕ := 2

-- Define the total number of gold stars
def total_stars : ℕ := monday_stars + tuesday_stars + wednesday_stars + thursday_stars + friday_stars

-- Prove that the total number of gold stars Shelby earned throughout the week is 24
theorem total_stars_correct : total_stars = 24 :=
by
  -- The proof goes here, using sorry to skip the proof
  sorry

end total_stars_correct_l272_272999


namespace arithmetic_sequence_a8_l272_272355

theorem arithmetic_sequence_a8 (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h1 : ∀ n, S n = n * (a 1 + a n) / 2)
  (h2 : S 15 = 90) :
  a 8 = 6 :=
by
  sorry

end arithmetic_sequence_a8_l272_272355


namespace find_b_l272_272738

theorem find_b
  (b : ℝ)
  (h1 : ∃ r : ℝ, 2 * r^2 + b * r - 65 = 0 ∧ r = 5)
  (h2 : 2 * 5^2 + b * 5 - 65 = 0) :
  b = 3 := by
  sorry

end find_b_l272_272738


namespace solve_inequalities_l272_272748

theorem solve_inequalities (x : ℝ) :
    ((x / 2 ≤ 3 + x) ∧ (3 + x < -3 * (1 + x))) ↔ (-6 ≤ x ∧ x < -3 / 2) :=
by
  sorry

end solve_inequalities_l272_272748


namespace original_number_l272_272272

theorem original_number (N y x : ℕ) 
  (h1: N + y = 54321)
  (h2: N = 10 * y + x)
  (h3: 11 * y + x = 54321)
  (h4: x = 54321 % 11)
  (hy: y = 4938) : 
  N = 49383 := 
  by 
  sorry

end original_number_l272_272272


namespace skates_cost_is_65_l272_272893

constant admission_cost : ℝ := 5
constant rental_cost_per_visit : ℝ := 2.50
constant visits_to_justify : ℕ := 26

noncomputable def new_skates_cost : ℝ :=
  rental_cost_per_visit * visits_to_justify

theorem skates_cost_is_65 : new_skates_cost = 65 := by
  unfold new_skates_cost
  calc
    rental_cost_per_visit * visits_to_justify
      = 2.50 * 26 : by sorry
    ... = 65 : by sorry

end skates_cost_is_65_l272_272893


namespace initial_nickels_l272_272527

theorem initial_nickels (quarters : ℕ) (initial_nickels : ℕ) (borrowed_nickels : ℕ) (current_nickels : ℕ) 
  (H1 : initial_nickels = 87) (H2 : borrowed_nickels = 75) (H3 : current_nickels = 12) : 
  initial_nickels = current_nickels + borrowed_nickels := 
by 
  -- proof steps go here
  sorry

end initial_nickels_l272_272527


namespace total_cost_verification_l272_272027

-- Conditions given in the problem
def holstein_cost : ℕ := 260
def jersey_cost : ℕ := 170
def num_hearts_on_card : ℕ := 4
def num_cards_in_deck : ℕ := 52
def cow_ratio_holstein : ℕ := 3
def cow_ratio_jersey : ℕ := 2
def sales_tax : ℝ := 0.05
def transport_cost_per_cow : ℕ := 20

def num_hearts_in_deck := num_cards_in_deck
def total_num_cows := 2 * num_hearts_in_deck
def total_parts_ratio := cow_ratio_holstein + cow_ratio_jersey

-- Total number of cows calculated 
def num_holstein_cows : ℕ := (cow_ratio_holstein * total_num_cows) / total_parts_ratio
def num_jersey_cows : ℕ := (cow_ratio_jersey * total_num_cows) / total_parts_ratio

-- Cost calculations
def holstein_total_cost := num_holstein_cows * holstein_cost
def jersey_total_cost := num_jersey_cows * jersey_cost
def total_cost_before_tax_and_transport := holstein_total_cost + jersey_total_cost
def total_sales_tax := total_cost_before_tax_and_transport * sales_tax
def total_transport_cost := total_num_cows * transport_cost_per_cow
def final_total_cost := total_cost_before_tax_and_transport + total_sales_tax + total_transport_cost

-- Lean statement to prove the result
theorem total_cost_verification : final_total_cost = 26324.50 := by sorry

end total_cost_verification_l272_272027


namespace minimum_value_A_l272_272838

theorem minimum_value_A (a b c : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_eq : a^2 * b + b^2 * c + c^2 * a = 3) : 
  (a^7 * b + b^7 * c + c^7 * a + a * b^3 + b * c^3 + c * a^3) ≥ 6 :=
by
  sorry

end minimum_value_A_l272_272838


namespace car_speed_decrease_l272_272388

theorem car_speed_decrease (d : ℝ) (speed_first : ℝ) (distance_fifth : ℝ) (time_interval : ℝ) :
  speed_first = 45 ∧ distance_fifth = 4.4 ∧ time_interval = 8 / 60 ∧ speed_first - 4 * d = distance_fifth / time_interval -> d = 3 :=
by
  intros h
  obtain ⟨_, _, _, h_eq⟩ := h
  sorry

end car_speed_decrease_l272_272388


namespace range_of_k_l272_272945

theorem range_of_k (k : ℝ) :
  (∃ a b c : ℝ, (a = 1) ∧ (b = -1) ∧ (c = -k) ∧ (b^2 - 4 * a * c > 0)) ↔ k > -1 / 4 :=
by
  sorry

end range_of_k_l272_272945


namespace ramu_repairs_cost_l272_272842

theorem ramu_repairs_cost :
  ∃ R : ℝ, 64900 - (42000 + R) = (29.8 / 100) * (42000 + R) :=
by
  use 8006.16
  sorry

end ramu_repairs_cost_l272_272842


namespace parabola_intercepts_sum_l272_272231

theorem parabola_intercepts_sum :
  let y_intercept := 4
  let x_intercept1 := (9 + Real.sqrt 33) / 6
  let x_intercept2 := (9 - Real.sqrt 33) / 6
  y_intercept + x_intercept1 + x_intercept2 = 7 :=
by
  let y_intercept := 4
  let x_intercept1 := (9 + Real.sqrt 33) / 6
  let x_intercept2 := (9 - Real.sqrt 33) / 6
  have sum_intercepts : y_intercept + x_intercept1 + x_intercept2 = 7 := by
        calc (4 : ℝ) + ((9 + Real.sqrt 33) / 6) + ((9 - Real.sqrt 33) / 6)
            = 4 + (18 / 6) : by
              rw [add_assoc, ← add_div, add_sub_cancel]
            ... = 4 + 3 : by norm_num
            ... = 7 : by norm_num
  exact sum_intercepts

end parabola_intercepts_sum_l272_272231


namespace number_of_friends_with_pears_l272_272834

-- Each friend either carries pears or oranges
def total_friends : Nat := 15
def friends_with_oranges : Nat := 6
def friends_with_pears : Nat := total_friends - friends_with_oranges

theorem number_of_friends_with_pears :
  friends_with_pears = 9 := by
  -- Proof steps would go here
  sorry

end number_of_friends_with_pears_l272_272834


namespace prove_p_or_q_l272_272782

-- Define propositions p and q
def p : Prop := ∃ n : ℕ, 0 = 2 * n
def q : Prop := ∃ m : ℕ, 3 = 2 * m

-- The Lean statement to prove
theorem prove_p_or_q : p ∨ q := by
  sorry

end prove_p_or_q_l272_272782


namespace integer_cube_less_than_triple_unique_l272_272702

theorem integer_cube_less_than_triple_unique (x : ℤ) (h : x^3 < 3*x) : x = 1 :=
sorry

end integer_cube_less_than_triple_unique_l272_272702


namespace point_of_tangency_l272_272450

def parabola1 (x y : ℝ) : Prop := y = x^2 + 15*x + 32
def parabola2 (x y : ℝ) : Prop := x = y^2 + 49*y + 593

theorem point_of_tangency :
  parabola1 (-7) (-24) ∧ parabola2 (-7) (-24) := by
  sorry

end point_of_tangency_l272_272450


namespace find_sample_size_l272_272115

theorem find_sample_size
  (teachers : ℕ := 200)
  (male_students : ℕ := 1200)
  (female_students : ℕ := 1000)
  (sampled_females : ℕ := 80)
  (total_people := teachers + male_students + female_students)
  (ratio : sampled_females / female_students = n / total_people)
  : n = 192 := 
by
  sorry

end find_sample_size_l272_272115


namespace plane_speed_in_still_air_l272_272291

theorem plane_speed_in_still_air (P W : ℝ) 
  (h1 : (P + W) * 3 = 900) 
  (h2 : (P - W) * 4 = 900) 
  : P = 262.5 :=
by
  sorry

end plane_speed_in_still_air_l272_272291


namespace part_I_part_II_l272_272617

def f (x a : ℝ) : ℝ := abs (3 * x + 2) - abs (2 * x + a)

theorem part_I (a : ℝ) : (∀ x : ℝ, f x a ≥ 0) ↔ a = 4 / 3 :=
by
  sorry

theorem part_II (a : ℝ) : (∃ x : ℝ, 1 ≤ x ∧ x ≤ 2 ∧ f x a ≤ 0) ↔ (3 ≤ a ∨ a ≤ -7) :=
by
  sorry

end part_I_part_II_l272_272617


namespace correct_operation_l272_272560

variable {x y : ℝ}

theorem correct_operation :
  (2 * x^2 + 4 * x^2 = 6 * x^2) → 
  (x * x^3 = x^4) → 
  ((x^3)^2 = x^6) →
  ((xy)^5 = x^5 * y^5) →
  ((x^3)^2 = x^6) := 
by 
  intros h1 h2 h3 h4
  exact h3

end correct_operation_l272_272560


namespace integer_solution_count_l272_272501

theorem integer_solution_count :
  (∃ x : ℤ, -4 * x ≥ x + 9 ∧ -3 * x ≤ 15 ∧ -5 * x ≥ 3 * x + 24) ↔
  (∃ n : ℕ, n = 3) :=
by
  sorry

end integer_solution_count_l272_272501


namespace arithmetic_sequence_sum_l272_272017

theorem arithmetic_sequence_sum (a b c : ℤ)
  (h1 : ∃ d : ℤ, a = 3 + d)
  (h2 : ∃ d : ℤ, b = 3 + 2 * d)
  (h3 : ∃ d : ℤ, c = 3 + 3 * d)
  (h4 : 3 + 3 * (c - 3) = 15) : a + b + c = 27 :=
by 
  sorry

end arithmetic_sequence_sum_l272_272017


namespace elaineExpenseChanges_l272_272198

noncomputable def elaineIncomeLastYear : ℝ := 20000 + 5000
noncomputable def elaineExpensesLastYearRent := 0.10 * elaineIncomeLastYear
noncomputable def elaineExpensesLastYearGroceries := 0.20 * elaineIncomeLastYear
noncomputable def elaineExpensesLastYearHealthcare := 0.15 * elaineIncomeLastYear
noncomputable def elaineTotalExpensesLastYear := elaineExpensesLastYearRent + elaineExpensesLastYearGroceries + elaineExpensesLastYearHealthcare
noncomputable def elaineSavingsLastYear := elaineIncomeLastYear - elaineTotalExpensesLastYear

noncomputable def elaineIncomeThisYear : ℝ := 23000 + 10000
noncomputable def elaineExpensesThisYearRent := 0.30 * elaineIncomeThisYear
noncomputable def elaineExpensesThisYearGroceries := 0.25 * elaineIncomeThisYear
noncomputable def elaineExpensesThisYearHealthcare := (0.15 * elaineIncomeThisYear) * 1.10
noncomputable def elaineTotalExpensesThisYear := elaineExpensesThisYearRent + elaineExpensesThisYearGroceries + elaineExpensesThisYearHealthcare
noncomputable def elaineSavingsThisYear := elaineIncomeThisYear - elaineTotalExpensesThisYear

theorem elaineExpenseChanges :
  ( ((elaineExpensesThisYearRent - elaineExpensesLastYearRent) / elaineExpensesLastYearRent) * 100 = 296)
  ∧ ( ((elaineExpensesThisYearGroceries - elaineExpensesLastYearGroceries) / elaineExpensesLastYearGroceries) * 100 = 65)
  ∧ ( ((elaineExpensesThisYearHealthcare - elaineExpensesLastYearHealthcare) / elaineExpensesLastYearHealthcare) * 100 = 45.2)
  ∧ ( (elaineSavingsLastYear / elaineIncomeLastYear) * 100 = 55)
  ∧ ( (elaineSavingsThisYear / elaineIncomeThisYear) * 100 = 28.5)
  ∧ ( (elaineTotalExpensesLastYear / elaineIncomeLastYear) = 0.45 )
  ∧ ( (elaineTotalExpensesThisYear / elaineIncomeThisYear) = 0.715 )
  ∧ ( (elaineSavingsLastYear - elaineSavingsThisYear) = 4345 ∧ ( (55 - ((elaineSavingsThisYear / elaineIncomeThisYear) * 100)) = 26.5 ))
:= by sorry

end elaineExpenseChanges_l272_272198


namespace cone_height_l272_272478

-- Definitions given in the problem
def slant_height : ℝ := 13
def lateral_area : ℝ := 65 * Real.pi

-- Definition of the radius as derived from the given conditions
def radius : ℝ := lateral_area / (Real.pi * slant_height) -- This simplifies to 5

-- Using the Pythagorean theorem to express the height
def height : ℝ := Real.sqrt (slant_height^2 - radius^2)

-- The statement to prove
theorem cone_height : height = 12 := by
  sorry

end cone_height_l272_272478


namespace sum_of_first_15_even_integers_l272_272252

theorem sum_of_first_15_even_integers : 
  let a := 2 in
  let d := 2 in
  let n := 15 in
  let S := (n / 2) * (a + (a + (n - 1) * d)) in
  S = 240 :=
by
  sorry

end sum_of_first_15_even_integers_l272_272252


namespace cards_arrangement_count_l272_272990

theorem cards_arrangement_count : 
  let cards := [1, 2, 3, 4, 5, 6, 7] in
  let valid_arrangements := 
    {arrangement | ∃ removed, 
      removed ∈ cards ∧ 
      (∀ remaining, 
        remaining = cards.erase removed → 
        (sorted remaining ∨ sorted (remaining.reverse))) } in
  valid_arrangements.card = 26 :=
sorry

end cards_arrangement_count_l272_272990


namespace time_to_meet_l272_272397

variable (distance : ℕ)
variable (speed1 speed2 time : ℕ)

-- Given conditions
def distanceAB := 480
def speedPassengerCar := 65
def speedCargoTruck := 55

-- Sum of the speeds of the two vehicles
def sumSpeeds := speedPassengerCar + speedCargoTruck

-- Prove that the time it takes for the two vehicles to meet is 4 hours
theorem time_to_meet : sumSpeeds * time = distanceAB → time = 4 :=
by
  sorry

end time_to_meet_l272_272397


namespace multiplication_is_correct_l272_272877

theorem multiplication_is_correct : 209 * 209 = 43681 := sorry

end multiplication_is_correct_l272_272877


namespace cos_A_sin_B_eq_l272_272222

theorem cos_A_sin_B_eq (A B : ℝ) (hA1 : 0 < A) (hA2 : A < π / 2) (hB1 : 0 < B) (hB2 : B < π / 2)
    (h : (4 + (Real.tan A)^2) * (5 + (Real.tan B)^2) = Real.sqrt 320 * Real.tan A * Real.tan B) :
    Real.cos A * Real.sin B = 1 / Real.sqrt 6 := sorry

end cos_A_sin_B_eq_l272_272222


namespace polynomial_geometric_roots_k_value_l272_272234

theorem polynomial_geometric_roots_k_value 
    (j k : ℝ)
    (h : ∃ (a r : ℝ), a ≠ 0 ∧ r ≠ 0 ∧ 
      (∀ u v : ℝ, (u = a ∨ u = a * r ∨ u = a * r^2 ∨ u = a * r^3) →
        (v = a ∨ v = a * r ∨ v = a * r^2 ∨ v = a * r^3) →
        u ≠ v) ∧ 
      (a + a * r + a * r^2 + a * r^3 = 0) ∧
      (a^4 * r^6 = 900)) :
  k = -900 :=
sorry

end polynomial_geometric_roots_k_value_l272_272234


namespace greatest_remainder_le_11_l272_272497

noncomputable def greatest_remainder (x : ℕ) : ℕ := x % 12

theorem greatest_remainder_le_11 (x : ℕ) (h : x % 12 ≠ 0) : greatest_remainder x = 11 :=
by
  sorry

end greatest_remainder_le_11_l272_272497


namespace reduced_admission_price_is_less_l272_272511

-- Defining the conditions
def regular_admission_cost : ℕ := 8
def total_people : ℕ := 2 + 3 + 1
def total_cost_before_6pm : ℕ := 30
def cost_per_person_before_6pm : ℕ := total_cost_before_6pm / total_people

-- Stating the theorem
theorem reduced_admission_price_is_less :
  (regular_admission_cost - cost_per_person_before_6pm) = 3 :=
by
  sorry -- Proof to be filled

end reduced_admission_price_is_less_l272_272511


namespace task_completion_days_l272_272733

theorem task_completion_days (a b c d : ℝ) 
    (h1 : 1/a + 1/b = 1/8)
    (h2 : 1/b + 1/c = 1/6)
    (h3 : 1/c + 1/d = 1/12) :
    1/a + 1/d = 1/24 :=
by
  sorry

end task_completion_days_l272_272733


namespace largest_triangle_angle_l272_272074

theorem largest_triangle_angle (y : ℝ) (h1 : 45 + 60 + y = 180) : y = 75 :=
by { sorry }

end largest_triangle_angle_l272_272074


namespace cube_less_than_triple_l272_272699

theorem cube_less_than_triple : ∀ x : ℤ, (x^3 < 3*x) ↔ (x = 1 ∨ x = -2) :=
by
  sorry

end cube_less_than_triple_l272_272699


namespace tangent_line_value_l272_272646

theorem tangent_line_value {a : ℝ} (h : a > 0) : 
  (∀ θ ρ, (ρ * (Real.cos θ + Real.sin θ) = a) → (ρ = 2 * Real.cos θ)) → 
  a = 1 + Real.sqrt 2 :=
sorry

end tangent_line_value_l272_272646


namespace centroid_of_triangle_l272_272128

theorem centroid_of_triangle :
  let A := (2, 8)
  let B := (6, 2)
  let C := (0, 4)
  let centroid := ( (A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3 )
  centroid = (8 / 3, 14 / 3) := 
by
  sorry

end centroid_of_triangle_l272_272128


namespace strictly_positive_integers_equal_l272_272661

theorem strictly_positive_integers_equal 
  (a b : ℤ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (h : (4 * a * b - 1) ∣ (4 * a^2 - 1)^2) : 
  a = b :=
sorry

end strictly_positive_integers_equal_l272_272661


namespace min_breaks_12_no_breaks_15_l272_272464

-- Define the function to sum the first n natural numbers
def sum_natural (n : ℕ) : ℕ := n * (n + 1) / 2

-- The main theorem for n = 12
theorem min_breaks_12 : ∀ (n = 12), (∑ i in finset.range (n + 1), i % 4 ≠ 0) → 2 := 
by sorry

-- The main theorem for n = 15
theorem no_breaks_15 : ∀ (n = 15), (∑ i in finset.range (n + 1), i % 4 = 0) → 0 := 
by sorry

end min_breaks_12_no_breaks_15_l272_272464


namespace boys_in_other_communities_l272_272513

def percentage_of_other_communities (p_M p_H p_S : ℕ) : ℕ :=
  100 - (p_M + p_H + p_S)

def number_of_boys_other_communities (total_boys : ℕ) (percentage_other : ℕ) : ℕ :=
  (percentage_other * total_boys) / 100

theorem boys_in_other_communities (N p_M p_H p_S : ℕ) (hN : N = 650) (hpM : p_M = 44) (hpH : p_H = 28) (hpS : p_S = 10) :
  number_of_boys_other_communities N (percentage_of_other_communities p_M p_H p_S) = 117 :=
by
  -- Steps to prove the theorem would go here
  sorry

end boys_in_other_communities_l272_272513


namespace problem_l272_272798

def f (x : ℤ) : ℤ := 7 * x - 3

theorem problem : f (f (f 3)) = 858 := by
  sorry

end problem_l272_272798


namespace sin_sum_of_roots_l272_272165

theorem sin_sum_of_roots (x1 x2 m : ℝ) (hx1 : 0 ≤ x1 ∧ x1 ≤ π) (hx2 : 0 ≤ x2 ∧ x2 ≤ π)
    (hroot1 : 2 * Real.sin x1 + Real.cos x1 = m) (hroot2 : 2 * Real.sin x2 + Real.cos x2 = m) :
    Real.sin (x1 + x2) = 4 / 5 := 
sorry

end sin_sum_of_roots_l272_272165


namespace parabola_directrix_is_x_eq_1_l272_272141

noncomputable def parabola_directrix (y : ℝ) : ℝ :=
  -1 / 4 * y^2

theorem parabola_directrix_is_x_eq_1 :
  ∀ x y, x = parabola_directrix y → x = 1 :=
by
  sorry

end parabola_directrix_is_x_eq_1_l272_272141


namespace min_birthdays_on_wednesday_l272_272711

theorem min_birthdays_on_wednesday (n x w: ℕ) (h_n : n = 61) 
  (h_ineq : w > x) (h_sum : 6 * x + w = n) : w ≥ 13 :=
by
  sorry

end min_birthdays_on_wednesday_l272_272711


namespace product_terms_l272_272031

variable (a_n : ℕ → ℝ)
variable (r : ℝ)

-- a1 = 1 and a10 = 3
axiom geom_seq  (h : ∀ n, a_n (n + 1) = r * a_n n) : a_n 1 = 1 → a_n 10 = 3

theorem product_terms :
  (∀ n, a_n (n + 1) = r * a_n n) → a_n 1 = 1 → a_n 10 = 3 → 
  a_n 2 * a_n 3 * a_n 4 * a_n 5 * a_n 6 * a_n 7 * a_n 8 * a_n 9 = 81 :=
by
  intros h1 h2 h3
  sorry

end product_terms_l272_272031


namespace equation_of_line_passing_through_point_with_slope_l272_272538

theorem equation_of_line_passing_through_point_with_slope :
  ∃ (l : ℝ → ℝ), l 0 = -1 ∧ ∀ (x y : ℝ), y = l x ↔ y + 1 = 2 * x :=
sorry

end equation_of_line_passing_through_point_with_slope_l272_272538


namespace gamma_delta_purchases_l272_272423

open Finset

-- Defining the problem context and proof
theorem gamma_delta_purchases :
  let cookies := 6
  let milk := 4
  let gamma_choices := cookies + milk
  let delta_choices := cookies
  ∑ i in (range 4), 
    ((choose gamma_choices 3 - i) * (if i = 0 then 1 else choose delta_choices i)) +
    ∑ j in (range 0..3), 
      if j = 2 then choose delta_choices 2 else
      if j = 1 then delta_choices else
      if j = 0 then choose (delta_choices - 1) 1 * delta_choices else choose delta_choices 3
  = 656 := by
  sorry

end gamma_delta_purchases_l272_272423


namespace odd_number_divides_3n_plus_1_l272_272127

theorem odd_number_divides_3n_plus_1 (n : ℕ) (h₁ : n % 2 = 1) (h₂ : n ∣ 3^n + 1) : n = 1 :=
by
  sorry

end odd_number_divides_3n_plus_1_l272_272127


namespace zack_marbles_number_l272_272566

-- Define the conditions as Lean definitions
def zack_initial_marbles (x : ℕ) :=
  (∃ k : ℕ, x = 3 * k + 5) ∧ (3 * 20 + 5 = 65)

-- State the theorem using the conditions
theorem zack_marbles_number : ∃ x : ℕ, zack_initial_marbles x ∧ x = 65 :=
by
  sorry

end zack_marbles_number_l272_272566


namespace lines_parallel_or_coincident_l272_272471

/-- Given lines l₁ and l₂ with certain properties,
    prove that they are either parallel or coincident. -/
theorem lines_parallel_or_coincident
  (P Q : ℝ × ℝ)
  (hP : P = (-2, -1))
  (hQ : Q = (3, -6))
  (h_slope1 : ∀ θ, θ = 135 → Real.tan (θ * (Real.pi / 180)) = -1)
  (h_slope2 : (Q.2 - P.2) / (Q.1 - P.1) = -1) : 
  true :=
by sorry

end lines_parallel_or_coincident_l272_272471


namespace find_m_l272_272927

theorem find_m 
  (m : ℝ)
  (h_pos : 0 < m)
  (asymptote_twice_angle : ∃ l : ℝ, l = 3 ∧ (x - l * y = 0 ∧ m * x^2 - y^2 = m)) :
  m = 3 :=
by
  sorry

end find_m_l272_272927


namespace cookies_total_is_60_l272_272825

def Mona_cookies : ℕ := 20
def Jasmine_cookies : ℕ := Mona_cookies - 5
def Rachel_cookies : ℕ := Jasmine_cookies + 10
def Total_cookies : ℕ := Mona_cookies + Jasmine_cookies + Rachel_cookies

theorem cookies_total_is_60 : Total_cookies = 60 := by
  sorry

end cookies_total_is_60_l272_272825


namespace acceptable_outfits_l272_272622

-- Definitions based on the given conditions
def shirts : Nat := 8
def pants : Nat := 5
def hats : Nat := 7
def pant_colors : List String := ["red", "black", "blue", "gray", "green"]
def shirt_colors : List String := ["red", "black", "blue", "gray", "green", "purple", "white"]
def hat_colors : List String := ["red", "black", "blue", "gray", "green", "purple", "white"]

-- Axiom that ensures distinct colors for pants, shirts, and hats.
axiom distinct_colors : ∀ color ∈ pant_colors, color ∈ shirt_colors ∧ color ∈ hat_colors

-- Problem statement
theorem acceptable_outfits : 
  let total_outfits := shirts * pants * hats
  let monochrome_outfits := List.length pant_colors
  let acceptable_outfits := total_outfits - monochrome_outfits
  acceptable_outfits = 275 :=
by
  sorry

end acceptable_outfits_l272_272622


namespace union_sets_l272_272342

def M : Set ℕ := {0, 1, 3}
def N : Set ℕ := {x | ∃ a ∈ M, x = 3 * a}

theorem union_sets : M ∪ N = {0, 1, 3, 9} :=
by
  sorry

end union_sets_l272_272342


namespace directrix_eqn_of_parabola_l272_272147

theorem directrix_eqn_of_parabola : 
  ∀ y : ℝ, x = - (1 / 4 : ℝ) * y ^ 2 → x = 1 :=
by
  sorry

end directrix_eqn_of_parabola_l272_272147


namespace time_for_B_alone_l272_272712

theorem time_for_B_alone (r_A r_B r_C : ℚ)
  (h1 : r_A + r_B = 1/3)
  (h2 : r_B + r_C = 2/7)
  (h3 : r_A + r_C = 1/4) :
  1/r_B = 168/31 :=
by
  sorry

end time_for_B_alone_l272_272712


namespace tom_paid_1145_l272_272859

-- Define the quantities
def quantity_apples : ℕ := 8
def rate_apples : ℕ := 70
def quantity_mangoes : ℕ := 9
def rate_mangoes : ℕ := 65

-- Calculate costs
def cost_apples : ℕ := quantity_apples * rate_apples
def cost_mangoes : ℕ := quantity_mangoes * rate_mangoes

-- Calculate the total amount paid
def total_amount_paid : ℕ := cost_apples + cost_mangoes

-- The theorem to prove
theorem tom_paid_1145 :
  total_amount_paid = 1145 :=
by sorry

end tom_paid_1145_l272_272859


namespace smallest_possible_b_l272_272658

theorem smallest_possible_b
  (a c b : ℤ)
  (h1 : a < c)
  (h2 : c < b)
  (h3 : c = (a + b) / 2)
  (h4 : b^2 / c = a) :
  b = 2 :=
sorry

end smallest_possible_b_l272_272658


namespace total_weight_moved_l272_272038

-- Define the given conditions as Lean definitions
def weight_per_rep : ℕ := 15
def number_of_reps : ℕ := 10
def number_of_sets : ℕ := 3

-- Define the theorem to prove total weight moved is 450 pounds
theorem total_weight_moved : weight_per_rep * number_of_reps * number_of_sets = 450 := by
  sorry

end total_weight_moved_l272_272038


namespace dice_sum_24_probability_l272_272766

noncomputable def probability_sum_24 : ℚ :=
  let prob_single_six := (1 : ℚ) / 6 in
  prob_single_six ^ 4

theorem dice_sum_24_probability :
  probability_sum_24 = 1 / 1296 :=
by
  sorry

end dice_sum_24_probability_l272_272766


namespace ice_cream_cone_cost_l272_272593

theorem ice_cream_cone_cost (total_sales : ℝ) (free_cones_given : ℕ) (cost_per_cone : ℝ) 
  (customers_per_group : ℕ) (cones_sold_per_group : ℕ) 
  (h1 : total_sales = 100)
  (h2: free_cones_given = 10)
  (h3: customers_per_group = 6)
  (h4: cones_sold_per_group = 5) :
  cost_per_cone = 2 := sorry

end ice_cream_cone_cost_l272_272593


namespace journey_speed_l272_272414

theorem journey_speed (v : ℝ) 
  (h1 : 3 * v + 60 * 2 = 240)
  (h2 : 3 + 2 = 5) :
  v = 40 :=
by
  sorry

end journey_speed_l272_272414


namespace probability_at_least_one_head_l272_272445

theorem probability_at_least_one_head :
  let p_tails : ℚ := 1 / 2
  let p_four_tails : ℚ := p_tails ^ 4
  let p_at_least_one_head : ℚ := 1 - p_four_tails
  p_at_least_one_head = 15 / 16 := by
  sorry

end probability_at_least_one_head_l272_272445


namespace add_fractions_add_fractions_as_mixed_l272_272296

theorem add_fractions : (3 / 4) + (5 / 6) + (4 / 3) = (35 / 12) := sorry

theorem add_fractions_as_mixed : (3 / 4) + (5 / 6) + (4 / 3) = 2 + 11 / 12 := sorry

end add_fractions_add_fractions_as_mixed_l272_272296


namespace last_digit_expr_is_4_l272_272573

-- Definitions for last digits.
def last_digit (n : ℕ) : ℕ := n % 10

def a : ℕ := 287
def b : ℕ := 269

def expr := (a * a) + (b * b) - (2 * a * b)

-- Conjecture stating that the last digit of the given expression is 4.
theorem last_digit_expr_is_4 : last_digit expr = 4 := 
by sorry

end last_digit_expr_is_4_l272_272573


namespace rectangular_field_area_l272_272402

noncomputable def length (c : ℚ) : ℚ := 3 * c / 2
noncomputable def width (c : ℚ) : ℚ := 4 * c / 2
noncomputable def area (c : ℚ) : ℚ := (length c) * (width c)
noncomputable def field_area (c1 : ℚ) (c2 : ℚ) : ℚ :=
  let l := length c1
  let w := width c1
  if 25 * c2 = 101.5 * 100 then
    area c1
  else
    0

theorem rectangular_field_area :
  ∃ (c : ℚ), field_area c 25 = 10092 := by
  sorry

end rectangular_field_area_l272_272402


namespace prob_sum_24_four_dice_l272_272768

section
open ProbabilityTheory

/-- Define the event E24 as the event where the sum of numbers on the top faces of four six-sided dice is 24 -/
def E24 : Event (StdGen) :=
eventOfFun {ω | ∑ i in range 4, (ω.gen_uniform int (6-1)) + 1 = 24}

/-- Probability that the sum of the numbers on top faces of four six-sided dice is 24 is 1/1296. -/
theorem prob_sum_24_four_dice : ⋆{ P(E24) = 1/1296 } := sorry

end

end prob_sum_24_four_dice_l272_272768


namespace work_rate_ab_together_l272_272568

-- Define A, B, and C as the work rates of individuals
variables (A B C : ℝ)

-- We are given the following conditions:
-- 1. a, b, and c together can finish the job in 11 days
-- 2. c alone can finish the job in 41.25 days

-- Given these conditions, we aim to prove that a and b together can finish the job in 15 days
theorem work_rate_ab_together
  (h1 : A + B + C = 1 / 11)
  (h2 : C = 1 / 41.25) :
  1 / (A + B) = 15 :=
by
  sorry

end work_rate_ab_together_l272_272568


namespace sqrt_fraction_fact_l272_272452

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem sqrt_fraction_fact :
  Real.sqrt (factorial 9 / 210 : ℝ) = 24 * Real.sqrt 3 := by
  sorry

end sqrt_fraction_fact_l272_272452


namespace determine_xyz_l272_272597

theorem determine_xyz (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0)
  (h4 : x + 1/y = 5) (h5 : y + 1/z = 2) (h6 : z + 1/x = 8/3) :
  x * y * z = 8 + 3 * Real.sqrt 7 :=
by
  sorry

end determine_xyz_l272_272597


namespace range_of_k_l272_272340

theorem range_of_k (k : ℝ) :
  (∃ x : ℝ, (k - 1) * x^2 + 2 * x - 1 = 0) ↔ (k ≥ 0 ∧ k ≠ 1) :=
by
  sorry

end range_of_k_l272_272340


namespace median_song_length_l272_272404

-- Define the list of song lengths in seconds
def song_lengths : List ℕ := [32, 43, 58, 65, 70, 72, 75, 80, 145, 150, 175, 180, 195, 210, 215, 225, 250, 252]

-- Define the statement that the median length of the songs is 147.5 seconds
theorem median_song_length : ∃ median : ℕ, median = 147 ∧ (median : ℚ) + 0.5 = 147.5 := by
  sorry

end median_song_length_l272_272404


namespace flour_needed_l272_272886

theorem flour_needed (cookies : ℕ) (flour : ℕ) (k : ℕ) (f_whole_wheat f_all_purpose : ℕ) 
  (h : cookies = 45) (h1 : flour = 3) (h2 : k = 90) (h3 : (k / 2) = 45) 
  (h4 : f_all_purpose = (flour * (k / cookies)) / 2) 
  (h5 : f_whole_wheat = (flour * (k / cookies)) / 2) : 
  f_all_purpose = 3 ∧ f_whole_wheat = 3 := 
by
  sorry

end flour_needed_l272_272886


namespace probability_point_within_circle_l272_272803

def fair_dice_outcomes : Finset (ℕ × ℕ) :=
  Finset.pi (Finset.range 6) (Finset.range 6)

def circle (x y : ℕ) : Prop :=
  x^2 + y^2 < 17

theorem probability_point_within_circle :
  (∑ p in fair_dice_outcomes, if circle p.1 p.2 then 1 else 0 : ℝ) / (Finset.card fair_dice_outcomes : ℝ) = 2 / 9 :=
by
  -- proof omitted
  sorry

end probability_point_within_circle_l272_272803


namespace pegs_arrangement_count_l272_272844

def num_ways_to_arrange_pegs : ℕ :=
  (nat.factorial 6) * (nat.factorial 5) * (nat.factorial 4) * (nat.factorial 3) * (nat.factorial 2)

theorem pegs_arrangement_count :
  num_ways_to_arrange_pegs = 12441600 :=
by
  unfold num_ways_to_arrange_pegs
  rw [nat.factorial, nat.factorial, nat.factorial, nat.factorial, nat.factorial]
  norm_num
  sorry

end pegs_arrangement_count_l272_272844


namespace directrix_of_parabola_l272_272132

-- Define the problem conditions
def parabola (x y : ℝ) : Prop :=
  x = -1/4 * y^2

-- State the theorem to prove that the directrix of the given parabola is x = 1
theorem directrix_of_parabola :
  (∀ (x y : ℝ), parabola x y → true) → 
  let d := 1 in
  ∀ (f : ℝ), (f = -d ∧ f^2 = d^2 ∧ d - f = 2) → true :=
by
  sorry

end directrix_of_parabola_l272_272132


namespace critics_voted_same_actor_actress_l272_272264

theorem critics_voted_same_actor_actress :
  ∃ (critic1 critic2 : ℕ) 
  (actor_vote1 actor_vote2 actress_vote1 actress_vote2 : ℕ),
  1 ≤ critic1 ∧ critic1 ≤ 3366 ∧
  1 ≤ critic2 ∧ critic2 ≤ 3366 ∧
  (critic1 ≠ critic2) ∧
  ∃ (vote_count : Fin 100 → ℕ) 
  (actor actress : Fin 3366 → Fin 100),
  (∀ n : Fin 100, ∃ act : Fin 100, vote_count act = n + 1) ∧
  actor critic1 = actor_vote1 ∧ actress critic1 = actress_vote1 ∧
  actor critic2 = actor_vote2 ∧ actress critic2 = actress_vote2 ∧
  actor_vote1 = actor_vote2 ∧ actress_vote1 = actress_vote2 :=
by
  -- Proof omitted
  sorry

end critics_voted_same_actor_actress_l272_272264


namespace positive_integer_solution_l272_272692

theorem positive_integer_solution (n : ℕ) (h1 : n + 2009 ∣ n^2 + 2009) (h2 : n + 2010 ∣ n^2 + 2010) : n = 1 := 
by
  -- The proof would go here.
  sorry

end positive_integer_solution_l272_272692


namespace john_pack_count_l272_272364

-- Defining the conditions
def utensilsInPack : Nat := 30
def knivesInPack : Nat := utensilsInPack / 3
def forksInPack : Nat := utensilsInPack / 3
def spoonsInPack : Nat := utensilsInPack / 3
def requiredKnivesRatio : Nat := 2
def requiredForksRatio : Nat := 3
def requiredSpoonsRatio : Nat := 5
def minimumSpoons : Nat := 50

-- Proving the solution
theorem john_pack_count : 
  ∃ packs : Nat, 
    (packs * spoonsInPack >= minimumSpoons) ∧
    (packs * foonsInPack / packs * knivesInPack = requiredForksRatio / requiredKnivesRatio) ∧
    (packs * spoonsInPack / packs * forksInPack = requiredForksRatio / requiredSpoonsRatio) ∧
    (packs * spoonsInPack / packs * knivesInPack = requiredSpoonsRatio / requiredKnivesRatio) ∧
    packs = 5 :=
sorry

end john_pack_count_l272_272364


namespace exists_multiple_of_power_of_two_non_zero_digits_l272_272672

open Nat

theorem exists_multiple_of_power_of_two_non_zero_digits (k : ℕ) (h : 0 < k) : 
  ∃ m : ℕ, (2^k ∣ m) ∧ (∀ d ∈ digits 10 m, d ≠ 0) :=
sorry

end exists_multiple_of_power_of_two_non_zero_digits_l272_272672


namespace find_point_on_parabola_l272_272884

open Real

theorem find_point_on_parabola :
  ∃ (x y : ℝ), 
  (0 ≤ x ∧ 0 ≤ y) ∧
  (x^2 = 8 * y) ∧
  sqrt (x^2 + (y - 2)^2) = 120 ∧
  (x = 2 * sqrt 236 ∧ y = 118) :=
by
  sorry

end find_point_on_parabola_l272_272884


namespace dice_sum_probability_l272_272758

theorem dice_sum_probability :
  let D := finset.range 1 7  -- outcomes of a fair six-sided die
  (∃! d1 d2 d3 d4 ∈ D, d1 + d2 + d3 + d4 = 24) ->
  (probability(space, {ω ∈ space | (ω 1 = 6) ∧ (ω 2 = 6) ∧ (ω 3 = 6) ∧ (ω 4 = 6)}) = 1/1296) :=
sorry

end dice_sum_probability_l272_272758


namespace max_t_squared_value_l272_272576

noncomputable def max_t_squared (R : ℝ) : ℝ :=
  let PR_QR_sq_sum := 4 * R^2
  let max_PR_QR_prod := 2 * R^2
  PR_QR_sq_sum + 2 * max_PR_QR_prod

theorem max_t_squared_value (R : ℝ) : max_t_squared R = 8 * R^2 :=
  sorry

end max_t_squared_value_l272_272576


namespace bag_of_food_costs_two_dollars_l272_272375

theorem bag_of_food_costs_two_dollars
  (cost_puppy : ℕ)
  (total_cost : ℕ)
  (daily_food : ℚ)
  (bag_food_quantity : ℚ)
  (weeks : ℕ)
  (h1 : cost_puppy = 10)
  (h2 : total_cost = 14)
  (h3 : daily_food = 1/3)
  (h4 : bag_food_quantity = 3.5)
  (h5 : weeks = 3) :
  (total_cost - cost_puppy) / (21 * daily_food / bag_food_quantity) = 2 := 
  by sorry

end bag_of_food_costs_two_dollars_l272_272375


namespace decreasing_condition_log_sum_condition_l272_272791

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := x * Real.log x - (1/2) * m * x^2 - x

def f_prime (x : ℝ) (m : ℝ) : ℝ := Real.log x - m * x

theorem decreasing_condition (m : ℝ) : (∀ x : ℝ, 0 < x → f_prime x m ≤ 0) ↔ m ≥ 1 / Real.exp 1 :=
by
  sorry

theorem log_sum_condition (x1 x2 m : ℝ) (hx : 0 < x1 ∧ x1 < x2 ∧ 0 < x2) (extreme_pts: f_prime x1 m = 0 ∧ f_prime x2 m = 0) :
  Real.log x1 + Real.log x2 > 2 :=
by
  sorry

end decreasing_condition_log_sum_condition_l272_272791


namespace polynomial_perfect_square_l272_272633

theorem polynomial_perfect_square (k : ℝ) 
  (h : ∃ a : ℝ, x^2 + 8*x + k = (x + a)^2) : 
  k = 16 :=
by
  sorry

end polynomial_perfect_square_l272_272633


namespace reciprocal_opposite_abs_val_l272_272852

theorem reciprocal_opposite_abs_val (a : ℚ) (h : a = -1 - 2/7) :
    (1 / a = -7/9) ∧ (-a = 1 + 2/7) ∧ (|a| = 1 + 2/7) := 
sorry

end reciprocal_opposite_abs_val_l272_272852


namespace john_profit_proof_l272_272812

-- Define the conditions
variables 
  (parts_cost : ℝ := 800)
  (selling_price_multiplier : ℝ := 1.4)
  (monthly_build_quantity : ℝ := 60)
  (monthly_rent : ℝ := 5000)
  (monthly_extra_expenses : ℝ := 3000)

-- Define the computed variables based on conditions
def selling_price_per_computer := parts_cost * selling_price_multiplier
def total_revenue := monthly_build_quantity * selling_price_per_computer
def total_cost_of_components := monthly_build_quantity * parts_cost
def total_expenses := monthly_rent + monthly_extra_expenses
def profit_per_month := total_revenue - total_cost_of_components - total_expenses

-- The theorem statement of the proof
theorem john_profit_proof : profit_per_month = 11200 := 
by
  sorry

end john_profit_proof_l272_272812


namespace larger_integer_exists_l272_272554

theorem larger_integer_exists (a b : ℤ) (h1 : a - b = 8) (h2 : a * b = 272) : a = 17 :=
sorry

end larger_integer_exists_l272_272554


namespace bugs_meet_on_diagonal_l272_272642

noncomputable def isosceles_trapezoid (A B C D : Type) : Prop :=
  ∃ (AB CD : ℝ), (AB > CD) ∧ (AB = AB) ∧ (CD = CD)

noncomputable def same_speeds (speed1 speed2 : ℝ) : Prop :=
  speed1 = speed2

noncomputable def opposite_directions (path1 path2 : ℝ → ℝ) (diagonal_length : ℝ) : Prop :=
  ∀ t, path1 t = diagonal_length - path2 t

noncomputable def bugs_meet (A B C D : Type) (path1 path2 : ℝ → ℝ) (T : ℝ) : Prop :=
  ∃ t ≤ T, path1 t = path2 t

theorem bugs_meet_on_diagonal :
  ∀ (A B C D : Type) (speed : ℝ) (path1 path2 : ℝ → ℝ) (diagonal_length cycle_period : ℝ),
  isosceles_trapezoid A B C D →
  same_speeds speed speed →
  (∀ t, 0 ≤ t → t ≤ cycle_period) →
  opposite_directions path1 path2 diagonal_length →
  bugs_meet A B C D path1 path2 cycle_period :=
by
  intros
  sorry

end bugs_meet_on_diagonal_l272_272642


namespace minimum_value_A_l272_272839

theorem minimum_value_A (a b c : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_eq : a^2 * b + b^2 * c + c^2 * a = 3) : 
  (a^7 * b + b^7 * c + c^7 * a + a * b^3 + b * c^3 + c * a^3) ≥ 6 :=
by
  sorry

end minimum_value_A_l272_272839


namespace draw_4_balls_in_order_ways_l272_272101

theorem draw_4_balls_in_order_ways : 
  ∀ (balls : Finset ℕ), 
  balls.card = 15 → 
  finset.permutations (balls).card 4 = 32760 := 
by 
  sorry

end draw_4_balls_in_order_ways_l272_272101


namespace original_five_digit_number_l272_272279

theorem original_five_digit_number :
  ∃ N y x : ℕ, (N = 10 * y + x) ∧ (N + y = 54321) ∧ (N = 49383) :=
by
  -- The proof script goes here
  sorry

end original_five_digit_number_l272_272279


namespace p_sq_plus_q_sq_l272_272974

theorem p_sq_plus_q_sq (p q : ℝ) (h1 : p * q = 12) (h2 : p + q = 8) : p^2 + q^2 = 40 :=
by
  sorry

end p_sq_plus_q_sq_l272_272974


namespace cookies_with_five_cups_of_flour_l272_272523

-- Define the conditions
def initial_cookies : ℕ := 24
def initial_flour : ℕ := 3
def additional_flour : ℕ := 5

-- State the problem
theorem cookies_with_five_cups_of_flour :
  (initial_cookies / initial_flour) * additional_flour = 40 :=
by
  -- Placeholder for proof
  sorry

end cookies_with_five_cups_of_flour_l272_272523


namespace bus_speed_calculation_l272_272320

noncomputable def bus_speed_excluding_stoppages : ℝ :=
  let effective_speed_with_stoppages := 50 -- kmph
  let stoppage_time_in_minutes := 13.125 -- minutes per hour
  let stoppage_time_in_hours := stoppage_time_in_minutes / 60 -- convert to hours
  let effective_moving_time := 1 - stoppage_time_in_hours -- effective moving time in one hour
  let bus_speed := (effective_speed_with_stoppages * 60) / (60 - stoppage_time_in_minutes) -- calculate bus speed
  bus_speed

theorem bus_speed_calculation : bus_speed_excluding_stoppages = 64 := by
  sorry

end bus_speed_calculation_l272_272320


namespace expected_number_of_digits_is_1_55_l272_272655

def probability_one_digit : ℚ := 9 / 20
def probability_two_digits : ℚ := 1 / 2
def probability_twenty : ℚ := 1 / 20
def expected_digits : ℚ := (1 * probability_one_digit) + (2 * probability_two_digits) + (2 * probability_twenty)

theorem expected_number_of_digits_is_1_55 :
  expected_digits = 1.55 :=
sorry

end expected_number_of_digits_is_1_55_l272_272655


namespace find_c_plus_inv_b_l272_272817

theorem find_c_plus_inv_b (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c)
  (h1 : a * b * c = 1) (h2 : a + 1 / c = 7) (h3 : b + 1 / a = 35) :
  c + 1 / b = 11 / 61 :=
by
  sorry

end find_c_plus_inv_b_l272_272817


namespace fraction_of_earth_surface_humans_can_inhabit_l272_272178

theorem fraction_of_earth_surface_humans_can_inhabit :
  (1 / 3) * (2 / 3) = (2 / 9) :=
by
  sorry

end fraction_of_earth_surface_humans_can_inhabit_l272_272178


namespace ermias_balls_more_is_5_l272_272428

-- Define the conditions
def time_per_ball : ℕ := 20
def alexia_balls : ℕ := 20
def total_time : ℕ := 900

-- Define Ermias's balls
def ermias_balls_more (x : ℕ) : ℕ := alexia_balls + x

-- Alexia's total inflation time
def alexia_total_time : ℕ := alexia_balls * time_per_ball

-- Ermias's total inflation time given x more balls than Alexia
def ermias_total_time (x : ℕ) : ℕ := (ermias_balls_more x) * time_per_ball

-- Total time taken by both Alexia and Ermias
def combined_time (x : ℕ) : ℕ := alexia_total_time + ermias_total_time x

-- Proven that Ermias inflated 5 more balls than Alexia given the total time condition
theorem ermias_balls_more_is_5 : (∃ x : ℕ, combined_time x = total_time) := 
by {
  sorry
}

end ermias_balls_more_is_5_l272_272428


namespace solution_set_inequality_l272_272469

theorem solution_set_inequality (a x : ℝ) (h : a > 0) :
  (∀ x, (a + 1 ≤ x ∧ x ≤ a + 3) ↔ (|((2 * x - 3 - 2 * a) / (x - a))| ≤ 1)) := 
sorry

end solution_set_inequality_l272_272469


namespace count_four_digit_increasing_odd_l272_272795

open Nat

-- Define the problem constraints
def increasing_order_digits {a b c d : ℕ} (h : 0 ≤ a ∧ a < b ∧ b < c ∧ c < d ∧ d ≤ 9) : Prop := true

def odd_last_digit {d : ℕ} (h : d % 2 = 1) : Prop := true

-- State the problem
theorem count_four_digit_increasing_odd :
  ∃ n : ℕ, (n = 130) ∧ 
  (∀ (a b c d : ℕ), (increasing_order_digits (and.intro (le_refl 0) (and.intro (lt_add_one_iff.mpr (lt_add_one_iff.mpr (lt_add_one_iff.mpr (lt_add_one_iff.mpr (le_refl 0))))) (and.intro (nat.le_succ _) (and.intro (nat.le_succ _) (nat.le_succ _))))))) →
  (odd_last_digit (and.intro (lt_add_one_iff.mpr (lt_add_one_iff.mpr (lt_add_one_iff.mpr (lt_add_one_iff.mpr (le_refl 0))))) (nat.le_succ _))) → n = 130) :=
begin
  use 130,
  split,
  { refl },
  { intros a b c d ho hi, sorry }
end


end count_four_digit_increasing_odd_l272_272795


namespace school_total_payment_l272_272651

def num_classes : ℕ := 4
def students_per_class : ℕ := 40
def chaperones_per_class : ℕ := 5
def student_fee : ℝ := 5.50
def adult_fee : ℝ := 6.50

def total_students : ℕ := num_classes * students_per_class
def total_adults : ℕ := num_classes * chaperones_per_class

def total_student_cost : ℝ := total_students * student_fee
def total_adult_cost : ℝ := total_adults * adult_fee

def total_cost : ℝ := total_student_cost + total_adult_cost

theorem school_total_payment : total_cost = 1010.0 := by
  sorry

end school_total_payment_l272_272651


namespace probability_divisor_of_12_l272_272721

open Probability

def divisors_of_12 := {1, 2, 3, 4, 6}

theorem probability_divisor_of_12 :
  ∃ (fair_die_roll : ProbabilityMeasure (Fin 6)), 
    P (fun x => x.val + 1 ∈ divisors_of_12) = 5 / 6 := 
by
  sorry

end probability_divisor_of_12_l272_272721


namespace original_number_l272_272271

theorem original_number (N y x : ℕ) 
  (h1: N + y = 54321)
  (h2: N = 10 * y + x)
  (h3: 11 * y + x = 54321)
  (h4: x = 54321 % 11)
  (hy: y = 4938) : 
  N = 49383 := 
  by 
  sorry

end original_number_l272_272271


namespace asha_wins_probability_l272_272509

variable (p_lose p_tie p_win : ℚ)

theorem asha_wins_probability 
  (h_lose : p_lose = 3 / 7) 
  (h_tie : p_tie = 1 / 7) 
  (h_total : p_win + p_lose + p_tie = 1) : 
  p_win = 3 / 7 := by
  sorry

end asha_wins_probability_l272_272509


namespace max_marks_400_l272_272429

theorem max_marks_400 {M : ℝ} (h1 : 0.35 * M = 140) : M = 400 :=
by 
-- skipping the proof using sorry
sorry

end max_marks_400_l272_272429


namespace fixed_point_1_3_l272_272848

noncomputable def fixed_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) : Prop :=
  (f (1) = 3) where f x := a^(x-1) + 2

theorem fixed_point_1_3 (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) : fixed_point a h1 h2 :=
by
  unfold fixed_point
  sorry

end fixed_point_1_3_l272_272848


namespace nonneg_integer_solutions_l272_272322

open Nat

theorem nonneg_integer_solutions (x y : ℕ) : 
  (x! + 2^y = (x + 1)! ) ↔ ((x = 1 ∧ y = 0) ∨ (x = 2 ∧ y = 1)) := by
  sorry

end nonneg_integer_solutions_l272_272322


namespace jack_marathon_time_l272_272517

noncomputable def marathon_distance : ℝ := 42
noncomputable def jill_time : ℝ := 4.2
noncomputable def speed_ratio : ℝ := 0.7636363636363637

noncomputable def jill_speed : ℝ := marathon_distance / jill_time
noncomputable def jack_speed : ℝ := speed_ratio * jill_speed
noncomputable def jack_time : ℝ := marathon_distance / jack_speed

theorem jack_marathon_time : jack_time = 5.5 := sorry

end jack_marathon_time_l272_272517


namespace zack_marbles_number_l272_272567

-- Define the conditions as Lean definitions
def zack_initial_marbles (x : ℕ) :=
  (∃ k : ℕ, x = 3 * k + 5) ∧ (3 * 20 + 5 = 65)

-- State the theorem using the conditions
theorem zack_marbles_number : ∃ x : ℕ, zack_initial_marbles x ∧ x = 65 :=
by
  sorry

end zack_marbles_number_l272_272567


namespace drawn_from_grade12_correct_l272_272105

-- Variables for the conditions
variable (total_students : ℕ) (sample_size : ℕ) (grade10_students : ℕ) 
          (grade11_students : ℕ) (grade12_students : ℕ) (drawn_from_grade12 : ℕ)

-- Conditions
def conditions : Prop :=
  total_students = 2400 ∧
  sample_size = 120 ∧
  grade10_students = 820 ∧
  grade11_students = 780 ∧
  grade12_students = total_students - grade10_students - grade11_students ∧
  drawn_from_grade12 = (grade12_students * sample_size) / total_students

-- Theorem to prove
theorem drawn_from_grade12_correct : conditions total_students sample_size grade10_students grade11_students grade12_students drawn_from_grade12 → drawn_from_grade12 = 40 :=
by
  intro h
  rcases h with ⟨h1, h2, h3, h4, h5, h6⟩
  sorry

end drawn_from_grade12_correct_l272_272105


namespace find_a_l272_272044

theorem find_a (a b c : ℂ) (ha : a.re = a) (h1 : a + b + c = 5) (h2 : a * b + b * c + c * a = 7) (h3 : a * b * c = 6) : a = 1 :=
by
  sorry

end find_a_l272_272044


namespace perpendicular_vecs_l272_272010

open Real

def vec_a : ℝ × ℝ := (1, 3)
def vec_b : ℝ × ℝ := (3, 4)
def lambda := 1 / 2

theorem perpendicular_vecs : 
  let vec_diff := (vec_a.1 - lambda * vec_b.1, vec_a.2 - lambda * vec_b.2)
  let vec_opp := (vec_b.1 - vec_a.1, vec_b.2 - vec_a.2)
  (vec_diff.1 * vec_opp.1 + vec_diff.2 * vec_opp.2) = 0 := 
by 
  let vec_diff := (vec_a.1 - lambda * vec_b.1, vec_a.2 - lambda * vec_b.2)
  let vec_opp := (vec_b.1 - vec_a.1, vec_b.2 - vec_a.2)
  show (vec_diff.1 * vec_opp.1 + vec_diff.2 * vec_opp.2) = 0
  sorry

end perpendicular_vecs_l272_272010


namespace percent_errors_l272_272102

theorem percent_errors (S : ℝ) (hS : S > 0) (Sm : ℝ) (hSm : Sm = 1.25 * S) :
  let P := 4 * S
  let Pm := 4 * Sm
  let A := S^2
  let Am := Sm^2
  let D := S * Real.sqrt 2
  let Dm := Sm * Real.sqrt 2
  let E_P := ((Pm - P) / P) * 100
  let E_A := ((Am - A) / A) * 100
  let E_D := ((Dm - D) / D) * 100
  E_P = 25 ∧ E_A = 56.25 ∧ E_D = 25 :=
by
  sorry

end percent_errors_l272_272102


namespace directrix_of_given_parabola_l272_272142

noncomputable def parabola_directrix (y : ℝ) : Prop :=
  let focus_x : ℝ := -1
  let point_on_parabola : ℝ × ℝ := (-1 / 4 * y^2, y)
  let PF_sq := (point_on_parabola.1 - focus_x)^2 + (point_on_parabola.2)^2
  let PQ_sq := (point_on_parabola.1 - 1)^2
  PF_sq = PQ_sq

theorem directrix_of_given_parabola : parabola_directrix y :=
sorry

end directrix_of_given_parabola_l272_272142


namespace minimum_value_range_l272_272181

noncomputable def f (a x : ℝ) : ℝ := abs (3 * x - 1) + a * x + 2

theorem minimum_value_range (a : ℝ) :
  (-3 ≤ a ∧ a ≤ 3) ↔ ∃ m, ∀ x, f a x ≥ m := sorry

end minimum_value_range_l272_272181


namespace find_rates_l272_272373

theorem find_rates
  (d b p t_p t_b t_w: ℕ)
  (rp rb rw: ℚ)
  (h1: d = b + 10)
  (h2: b = 3 * p)
  (h3: p = 50)
  (h4: t_p = 4)
  (h5: t_b = 2)
  (h6: t_w = 5)
  (h7: rp = p / t_p)
  (h8: rb = b / t_b)
  (h9: rw = d / t_w):
  rp = 12.5 ∧ rb = 75 ∧ rw = 32 := by
  sorry

end find_rates_l272_272373


namespace train_length_l272_272433

theorem train_length 
  (t1 t2 : ℕ) 
  (d2 : ℕ) 
  (V L : ℝ) 
  (h1 : t1 = 11)
  (h2 : t2 = 22)
  (h3 : d2 = 120)
  (h4 : V = L / t1)
  (h5 : V = (L + d2) / t2) : 
  L = 120 := 
by 
  sorry

end train_length_l272_272433


namespace polynomial_coefficients_sum_l272_272064

theorem polynomial_coefficients_sum :
  let a := -15
  let b := 69
  let c := -81
  let d := 27
  10 * a + 5 * b + 2 * c + d = 60 :=
by
  let a := -15
  let b := 69
  let c := -81
  let d := 27
  sorry

end polynomial_coefficients_sum_l272_272064


namespace minimum_sticks_broken_n12_can_form_square_n15_l272_272466

-- Define the total length function
def total_length (n : ℕ) : ℕ := (n * (n + 1)) / 2

-- For n = 12, prove that at least 2 sticks need to be broken to form a square
theorem minimum_sticks_broken_n12 : ∀ (n : ℕ), n = 12 → total_length n % 4 ≠ 0 → 2 = 2 := 
by 
  intros n h1 h2
  sorry

-- For n = 15, prove that a square can be directly formed
theorem can_form_square_n15 : ∀ (n : ℕ), n = 15 → total_length n % 4 = 0 := 
by 
  intros n h1
  sorry

end minimum_sticks_broken_n12_can_form_square_n15_l272_272466


namespace paving_cost_l272_272872

theorem paving_cost (l w r : ℝ) (h_l : l = 5.5) (h_w : w = 4) (h_r : r = 700) :
  l * w * r = 15400 :=
by sorry

end paving_cost_l272_272872


namespace calculate_geometric_sequence_sum_l272_272205

def geometric_sequence (a₁ r : ℤ) (n : ℕ) : ℤ :=
  a₁ * r^n

theorem calculate_geometric_sequence_sum :
  let a₁ := 1
  let r := -2
  let a₂ := geometric_sequence a₁ r 1
  let a₃ := geometric_sequence a₁ r 2
  let a₄ := geometric_sequence a₁ r 3
  a₁ + |a₂| + a₃ + |a₄| = 15 :=
by
  sorry

end calculate_geometric_sequence_sum_l272_272205


namespace simplify_expression_l272_272256

noncomputable def a : ℝ := 2 * Real.sqrt 12 - 4 * Real.sqrt 27 + 3 * Real.sqrt 75 + 7 * Real.sqrt 8 - 3 * Real.sqrt 18
noncomputable def b : ℝ := 4 * Real.sqrt 48 - 3 * Real.sqrt 27 - 5 * Real.sqrt 18 + 2 * Real.sqrt 50

theorem simplify_expression : a * b = 97 := by
  sorry

end simplify_expression_l272_272256


namespace trapezoid_perimeter_l272_272029

-- Define the properties of the isosceles trapezoid
structure IsoscelesTrapezoid (A B C D : Type) :=
  (AB CD BC DA : ℝ)
  (AB_parallel_CD : AB = CD)
  (BC_eq_DA : BC = 13)
  (DA_eq_BC : DA = 13)
  (sum_AB_CD : AB + CD = 24)

-- Define the problem's conditions as Lean definitions
def trapezoidABCD : IsoscelesTrapezoid ℝ ℝ ℝ ℝ :=
{
  AB := 12,
  CD := 12,
  BC := 13,
  DA := 13,
  AB_parallel_CD := by sorry,
  BC_eq_DA := by sorry,
  DA_eq_BC := by sorry,
  sum_AB_CD := by sorry,
}

-- State the theorem we want to prove
theorem trapezoid_perimeter (trapezoid : IsoscelesTrapezoid ℝ ℝ ℝ ℝ) : 
  trapezoid.AB + trapezoid.BC + trapezoid.CD + trapezoid.DA = 50 :=
by sorry

end trapezoid_perimeter_l272_272029


namespace min_age_of_youngest_person_l272_272083

theorem min_age_of_youngest_person
  {a b c d e : ℕ}
  (h_sum : a + b + c + d + e = 256)
  (h_order : a ≤ b ∧ b ≤ c ∧ c ≤ d ∧ d ≤ e)
  (h_diff : 2 ≤ (b - a) ∧ (b - a) ≤ 10 ∧ 
            2 ≤ (c - b) ∧ (c - b) ≤ 10 ∧ 
            2 ≤ (d - c) ∧ (d - c) ≤ 10 ∧ 
            2 ≤ (e - d) ∧ (e - d) ≤ 10) : 
  a = 32 :=
sorry

end min_age_of_youngest_person_l272_272083


namespace find_term_number_l272_272882

variable {α : ℝ} (b : ℕ → ℝ) (q : ℝ)

namespace GeometricProgression

noncomputable def geometric_progression (b : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ (n : ℕ), b (n + 1) = b n * q

noncomputable def satisfies_conditions (α : ℝ) (b : ℕ → ℝ) : Prop :=
  b 25 = 2 * Real.tan α ∧ b 31 = 2 * Real.sin α

theorem find_term_number (α : ℝ) (b : ℕ → ℝ) (q : ℝ) (hb : geometric_progression b q) (hc : satisfies_conditions α b) :
  ∃ n, b n = Real.sin (2 * α) ∧ n = 37 :=
sorry

end GeometricProgression

end find_term_number_l272_272882


namespace prime_between_40_50_largest_prime_lt_100_l272_272728

def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

def between (n m k : ℕ) : Prop := n < k ∧ k < m

theorem prime_between_40_50 :
  {x : ℕ | between 40 50 x ∧ isPrime x} = {41, 43, 47} :=
sorry

theorem largest_prime_lt_100 :
  ∃ p : ℕ, isPrime p ∧ p < 100 ∧ ∀ q : ℕ, isPrime q ∧ q < 100 → q ≤ p :=
sorry

end prime_between_40_50_largest_prime_lt_100_l272_272728


namespace cash_sales_amount_l272_272390

-- Definitions for conditions
def total_sales : ℕ := 80
def credit_sales : ℕ := (2 * total_sales) / 5

-- Statement of the proof problem
theorem cash_sales_amount :
  ∃ cash_sales : ℕ, cash_sales = total_sales - credit_sales ∧ cash_sales = 48 :=
by
  sorry

end cash_sales_amount_l272_272390


namespace compute_a_d_sum_l272_272659

variables {a1 a2 a3 d1 d2 d3 : ℝ}

theorem compute_a_d_sum
  (h : ∀ x : ℝ, x^6 + x^5 + x^4 + x^3 + x^2 + x + 1 = (x^2 + a1 * x + d1) * (x^2 + a2 * x + d2) * (x^2 + a3 * x + d3)) :
  a1 * d1 + a2 * d2 + a3 * d3 = 1 :=
  sorry

end compute_a_d_sum_l272_272659


namespace ellipse_eq_range_m_l272_272944

theorem ellipse_eq_range_m (m : ℝ) : 
  (∃ x y : ℝ, (x^2 / (m - 1) + y^2 / (3 - m) = 1)) ↔ (1 < m ∧ m < 2) ∨ (2 < m ∧ m < 3) :=
sorry

end ellipse_eq_range_m_l272_272944


namespace prob_divisor_of_12_l272_272723

theorem prob_divisor_of_12 :
  (∃ d : Finset ℕ, d = {1, 2, 3, 4, 6}) → (∃ s : Finset ℕ, s = {1, 2, 3, 4, 5, 6}) →
  let favorable := 5
  let total := 6
  favorable / total = (5 : ℚ / 6 ) := sorry

end prob_divisor_of_12_l272_272723


namespace expression_evaluation_l272_272503

theorem expression_evaluation (x : ℝ) (h : 2 * x - 7 = 8 * x - 1) : 5 * (x - 3) = -20 :=
by
  sorry

end expression_evaluation_l272_272503


namespace right_triangle_area_is_integer_l272_272073

theorem right_triangle_area_is_integer (a b : ℕ) (h1 : ∃ (A : ℕ), A = (1 / 2 : ℚ) * ↑a * ↑b) : (a % 2 = 0) ∨ (b % 2 = 0) :=
sorry

end right_triangle_area_is_integer_l272_272073


namespace problem1_l272_272262

theorem problem1 (α : Real) (h : Real.tan (Real.pi / 4 + α) = 2) : 
  (Real.sin α + 3 * Real.cos α) / (Real.sin α - Real.cos α) = -5 := 
sorry

end problem1_l272_272262


namespace grade_above_B_l272_272379

theorem grade_above_B (total_students : ℕ) (percentage_below_B : ℕ) (students_above_B : ℕ) :
  total_students = 60 ∧ percentage_below_B = 40 ∧ students_above_B = total_students * (100 - percentage_below_B) / 100 →
  students_above_B = 36 :=
by
  sorry

end grade_above_B_l272_272379


namespace distance_between_consecutive_trees_l272_272186

-- Define the conditions as separate definitions
def num_trees : ℕ := 57
def yard_length : ℝ := 720
def spaces_between_trees := num_trees - 1

-- Define the target statement to prove
theorem distance_between_consecutive_trees :
  yard_length / spaces_between_trees = 12.857142857 := sorry

end distance_between_consecutive_trees_l272_272186


namespace solve_for_x_l272_272536

def star (a b : ℝ) : ℝ := 3 * a - b

theorem solve_for_x :
  ∃ x : ℝ, star 2 (star 5 x) = 1 ∧ x = 10 := by
  sorry

end solve_for_x_l272_272536


namespace least_odd_prime_factor_2027_l272_272754

-- Definitions for the conditions
def is_prime (p : ℕ) : Prop := Nat.Prime p
def order_divides (a n p : ℕ) : Prop := a ^ n % p = 1

-- Define lean function to denote the problem.
theorem least_odd_prime_factor_2027 :
  ∀ p : ℕ, 
  is_prime p → 
  order_divides 2027 12 p ∧ ¬ order_divides 2027 6 p → 
  p ≡ 1 [MOD 12] → 
  2027^6 + 1 % p = 0 → 
  p = 37 :=
by
  -- skipping proof steps
  sorry

end least_odd_prime_factor_2027_l272_272754


namespace mart_income_percentage_of_juan_l272_272376

theorem mart_income_percentage_of_juan
  (J T M : ℝ)
  (h1 : T = 0.60 * J)
  (h2 : M = 1.60 * T) :
  M = 0.96 * J :=
by 
  sorry

end mart_income_percentage_of_juan_l272_272376


namespace palindromes_between_300_800_l272_272013

def palindrome_count (l u : ℕ) : ℕ :=
  (u / 100 - l / 100 + 1) * 10

theorem palindromes_between_300_800 : palindrome_count 300 800 = 50 :=
by
  sorry

end palindromes_between_300_800_l272_272013


namespace simplify_sqrt_eight_l272_272675

theorem simplify_sqrt_eight : Real.sqrt 8 = 2 * Real.sqrt 2 :=
by
  -- Given that 8 can be factored into 4 * 2 and the property sqrt(a * b) = sqrt(a) * sqrt(b)
  sorry

end simplify_sqrt_eight_l272_272675


namespace no_triangle_sides_exist_l272_272316

theorem no_triangle_sides_exist (x y z : ℝ) (h_triangle_sides : x > 0 ∧ y > 0 ∧ z > 0)
  (h_triangle_inequality : x < y + z ∧ y < x + z ∧ z < x + y) :
  x^3 + y^3 + z^3 ≠ (x + y) * (y + z) * (z + x) :=
sorry

end no_triangle_sides_exist_l272_272316


namespace total_chips_is_90_l272_272691

theorem total_chips_is_90
  (viv_vanilla : ℕ)
  (sus_choco : ℕ)
  (viv_choco_more : ℕ)
  (sus_vanilla_ratio : ℚ)
  (viv_choco : ℕ)
  (sus_vanilla : ℕ)
  (total_choco : ℕ)
  (total_vanilla : ℕ)
  (total_chips : ℕ) :
  viv_vanilla = 20 →
  sus_choco = 25 →
  viv_choco_more = 5 →
  sus_vanilla_ratio = 3 / 4 →
  viv_choco = sus_choco + viv_choco_more →
  sus_vanilla = (sus_vanilla_ratio * viv_vanilla) →
  total_choco = viv_choco + sus_choco →
  total_vanilla = viv_vanilla + sus_vanilla →
  total_chips = total_choco + total_vanilla →
  total_chips = 90 :=
by
  intros
  sorry

end total_chips_is_90_l272_272691


namespace range_of_a_l272_272025

theorem range_of_a (a : ℝ) (h : a - 2 * 1 + 4 > 0) : a > -2 :=
by
  -- proof is not required
  sorry

end range_of_a_l272_272025


namespace theta_in_fourth_quadrant_l272_272777

theorem theta_in_fourth_quadrant (θ : ℝ) (h1 : Real.cos θ > 0) (h2 : Real.tan (θ + Real.pi / 4) = 1 / 3) : 
  (θ > 3 * Real.pi / 2) ∧ (θ < 2 * Real.pi) :=
sorry

end theta_in_fourth_quadrant_l272_272777


namespace volume_eq_three_times_other_two_l272_272802

-- declare the given ratio of the radii
def r1 : ℝ := 1
def r2 : ℝ := 2
def r3 : ℝ := 3

-- calculate the volumes based on the given radii
noncomputable def V (r : ℝ) : ℝ := (4/3) * Real.pi * r^3

-- defining the volumes of the three spheres
noncomputable def V1 : ℝ := V r1
noncomputable def V2 : ℝ := V r2
noncomputable def V3 : ℝ := V r3

theorem volume_eq_three_times_other_two : V3 = 3 * (V1 + V2) := 
by
  sorry

end volume_eq_three_times_other_two_l272_272802


namespace number_of_lawns_mowed_l272_272813

noncomputable def ChargePerLawn : ℕ := 33
noncomputable def TotalTips : ℕ := 30
noncomputable def TotalEarnings : ℕ := 558

theorem number_of_lawns_mowed (L : ℕ) 
  (h1 : ChargePerLawn * L + TotalTips = TotalEarnings) : L = 16 := 
by
  sorry

end number_of_lawns_mowed_l272_272813


namespace amateur_definition_l272_272710
-- Import necessary libraries

-- Define the meaning of "amateur" and state that it is "amateurish" or "non-professional"
def meaning_of_amateur : String :=
  "amateurish or non-professional"

-- The main statement asserting that the meaning of "amateur" is indeed "amateurish" or "non-professional"
theorem amateur_definition : meaning_of_amateur = "amateurish or non-professional" :=
by
  -- The proof is trivial and assumed to be correct
  sorry

end amateur_definition_l272_272710


namespace problem1_problem2_l272_272167

noncomputable def f (x : ℝ) : ℝ := x^2 + 2 / x

def is_increasing_on (f : ℝ → ℝ) (domain : Set ℝ) : Prop :=
  ∀ x₁ x₂, x₁ ∈ domain → x₂ ∈ domain → x₁ < x₂ → f x₁ < f x₂

theorem problem1 : is_increasing_on f {x | 1 ≤ x} := 
by sorry

def is_decreasing (g : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → g x₁ > g x₂

theorem problem2 (g : ℝ → ℝ) (h_decreasing : is_decreasing g)
  (h_inequality : ∀ x : ℝ, 1 ≤ x → g (x^3 + 2) < g ((a^2 - 2 * a) * x)) :
  -1 < a ∧ a < 3 :=
by sorry

end problem1_problem2_l272_272167


namespace calculation_error_l272_272731

theorem calculation_error (x y : ℕ) : (25 * x + 5 * y) = 25 * x + 5 * y :=
by
  sorry

end calculation_error_l272_272731


namespace minimum_a_plus_2b_no_a_b_such_that_l272_272605

noncomputable def minimum_value (a b : ℝ) :=
  a + 2 * b

theorem minimum_a_plus_2b (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 2 * a * b = a + 2 * b + 3) : 
  minimum_value a b ≥ 6 :=
sorry

theorem no_a_b_such_that (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 2 * a * b = a + 2 * b + 3) : 
  a^2 + 4 * b^2 ≠ 17 :=
sorry

end minimum_a_plus_2b_no_a_b_such_that_l272_272605


namespace points_on_same_circle_l272_272180
open Real

theorem points_on_same_circle (m : ℝ) :
  ∃ D E F, 
  (2^2 + 1^2 + 2 * D + 1 * E + F = 0) ∧
  (4^2 + 2^2 + 4 * D + 2 * E + F = 0) ∧
  (3^2 + 4^2 + 3 * D + 4 * E + F = 0) ∧
  (1^2 + m^2 + 1 * D + m * E + F = 0) →
  (m = 2 ∨ m = 3) := 
sorry

end points_on_same_circle_l272_272180


namespace least_gumballs_to_get_four_same_color_l272_272726

theorem least_gumballs_to_get_four_same_color
  (R W B : ℕ)
  (hR : R = 9)
  (hW : W = 7)
  (hB : B = 8) : 
  ∃ n, n = 10 ∧ (∀ m < n, ∀ r w b : ℕ, r + w + b = m → r < 4 ∧ w < 4 ∧ b < 4) ∧ 
  (∀ r w b : ℕ, r + w + b = n → r = 4 ∨ w = 4 ∨ b = 4) :=
sorry

end least_gumballs_to_get_four_same_color_l272_272726


namespace simplify_fraction_l272_272016

variable (x y : ℝ)
variable (h1 : x ≠ 0)
variable (h2 : y ≠ 0)
variable (h3 : x - y^2 ≠ 0)

theorem simplify_fraction :
  (y^2 - 1/x) / (x - y^2) = (x * y^2 - 1) / (x^2 - x * y^2) :=
by
  sorry

end simplify_fraction_l272_272016


namespace total_cookies_l272_272832

def mona_cookies : ℕ := 20
def jasmine_cookies : ℕ := mona_cookies - 5
def rachel_cookies : ℕ := jasmine_cookies + 10

theorem total_cookies : mona_cookies + jasmine_cookies + rachel_cookies = 60 := 
by
  have h1 : jasmine_cookies = 15 := by sorry
  have h2 : rachel_cookies = 25 := by sorry
  have h3 : mona_cookies = 20 := by sorry
  sorry

end total_cookies_l272_272832


namespace min_sticks_to_break_n12_l272_272462

theorem min_sticks_to_break_n12 : 
  let sticks := (Finset.range 12).map (λ x => x + 1)
  let total_length := sticks.sum
  total_length % 4 ≠ 0 → 
  (∃ k, k < 3 ∧ 
    ∃ broken_sticks: Finset Nat, 
      (∀ s ∈ broken_sticks, s < 12 ∧ s > 0) ∧ broken_sticks.card = k ∧ 
        sticks.sum + (broken_sticks.sum / 2) % 4 = 0) :=
sorry

end min_sticks_to_break_n12_l272_272462


namespace divisible_by_five_l272_272862

theorem divisible_by_five (a b : ℕ) (h : 5 ∣ (a * b)) : (5 ∣ a) ∨ (5 ∣ b) :=
sorry

end divisible_by_five_l272_272862


namespace pool_drain_rate_l272_272679

-- Define the dimensions and other conditions
def poolLength : ℝ := 150
def poolWidth : ℝ := 40
def poolDepth : ℝ := 10
def poolCapacityPercent : ℝ := 0.80
def drainTime : ℕ := 800

-- Define the problem statement
theorem pool_drain_rate :
  let fullVolume := poolLength * poolWidth * poolDepth
  let volumeAt80Percent := fullVolume * poolCapacityPercent
  let drainRate := volumeAt80Percent / drainTime
  drainRate = 60 :=
by
  sorry

end pool_drain_rate_l272_272679


namespace dice_sum_24_probability_l272_272765

noncomputable def probability_sum_24 : ℚ :=
  let prob_single_six := (1 : ℚ) / 6 in
  prob_single_six ^ 4

theorem dice_sum_24_probability :
  probability_sum_24 = 1 / 1296 :=
by
  sorry

end dice_sum_24_probability_l272_272765


namespace snail_distance_l272_272843

def speed_A : ℝ := 10
def speed_B : ℝ := 15
def time_difference : ℝ := 0.5

theorem snail_distance : 
  ∃ (D : ℝ) (t_A t_B : ℝ), 
    D = speed_A * t_A ∧ 
    D = speed_B * t_B ∧
    t_A = t_B + time_difference ∧ 
    D = 15 := 
by
  sorry

end snail_distance_l272_272843


namespace cone_height_l272_272483

theorem cone_height (l : ℝ) (A : ℝ) (h : ℝ) (r : ℝ) 
  (h_slant_height : l = 13)
  (h_lateral_area : A = 65 * π)
  (h_radius : r = 5)
  (h_height_formula : h = Real.sqrt (l^2 - r^2)) : 
  h = 12 := 
by 
  sorry

end cone_height_l272_272483


namespace Ivan_returns_alive_Ivan_takes_princesses_l272_272656

theorem Ivan_returns_alive (Tsarevnas Koscheis: Finset ℕ) (five_girls: Finset ℕ) 
  (cond1: Tsarevnas.card = 3) (cond2: Koscheis.card = 2) (cond3: five_girls.card = 5)
  (cond4: Tsarevnas ∪ Koscheis = five_girls)
  (cond5: ∀ g ∈ five_girls, ∃ t ∈ Tsarevnas, t ≠ g ∧ ∃ k ∈ Koscheis, k ≠ g)
  (cond6: ∀ girl : ℕ, girl ∈ five_girls → 
          ∃ truth_count : ℕ, 
          (truth_count = (if girl ∈ Tsarevnas then 2 else 3))): 
  ∃ princesses : Finset ℕ, princesses.card = 3 ∧ princesses ⊆ Tsarevnas ∧ ∀ k ∈ Koscheis, k ∉ princesses :=
sorry

theorem Ivan_takes_princesses (Tsarevnas Koscheis: Finset ℕ) (five_girls: Finset ℕ) 
  (cond1: Tsarevnas.card = 3) (cond2: Koscheis.card = 2) (cond3: five_girls.card = 5)
  (cond4: Tsarevnas ∪ Koscheis = five_girls)
  (cond5: ∀ g ∈ five_girls, ∃ t ∈ Tsarevnas, t ≠ g ∧ ∃ k ∈ Koscheis, k ≠ g)
  (cond6 and cond7: ∀ girl1 girl2 girl3 : ℕ, girl1 ≠ girl2 → girl2 ≠ girl3 → girl1 ∈ Tsarevnas → girl2 ∈ Tsarevnas → girl3 ∈ Tsarevnas → 
          ∃ (eldest middle youngest : ℕ), 
              (eldest ∈ Tsarevnas ∧ middle ∈ Tsarevnas ∧ youngest ∈ Tsarevnas) 
          ∧
              (eldest ≠ middle ∧ eldest ≠ youngest ∧ middle ≠ youngest)
          ∧
              (∀ k ∈ Koscheis, k ≠ eldest ∧ k ≠ middle ∧ k ≠ youngest)
  ):
  ∃ princesses : Finset ℕ, 
          princesses.card = 3 ∧ princesses ⊆ Tsarevnas ∧ 
          (∃ eldest ,∃ middle,∃ youngest : ℕ, eldest ∈ princesses ∧ middle ∈ princesses ∧ youngest ∈ princesses ∧ 
                 eldest ≠ middle ∧ eldest ≠ youngest ∧ middle ≠ youngest)
:=
sorry

end Ivan_returns_alive_Ivan_takes_princesses_l272_272656


namespace coinsSold_l272_272119

-- Given conditions
def initialCoins : Nat := 250
def additionalCoins : Nat := 75
def coinsToKeep : Nat := 135

-- Theorem to prove
theorem coinsSold : (initialCoins + additionalCoins - coinsToKeep) = 190 := 
by
  -- Proof omitted 
  sorry

end coinsSold_l272_272119


namespace sum_of_first_15_even_positive_integers_l272_272251

theorem sum_of_first_15_even_positive_integers :
  let a := 2
  let l := 30
  let n := 15
  let S := (a + l) / 2 * n
  S = 240 := by
  sorry

end sum_of_first_15_even_positive_integers_l272_272251


namespace inverse_proportion_relation_l272_272331

theorem inverse_proportion_relation :
  let y1 := 3 / (-2)
  let y2 := 3 / (-1)
  let y3 := 3 / 1
  y2 < y1 ∧ y1 < y3 :=
by
  -- Variable definitions according to conditions
  let y1 := 3 / (-2)
  let y2 := 3 / (-1)
  let y3 := 3 / 1
  -- Proof steps go here (not required for the statement)
  -- Since proof steps are omitted, we use sorry to indicate it
  sorry

end inverse_proportion_relation_l272_272331


namespace balls_sum_l272_272421

theorem balls_sum (m n : ℕ) (h₁ : ∀ a, a ∈ ({m, 8, n} : Finset ℕ)) -- condition: balls are identical except for color
  (h₂ : (8 : ℝ) / (m + 8 + n) = (m + n : ℝ) / (m + 8 + n)) : m + n = 8 :=
sorry

end balls_sum_l272_272421


namespace intersection_point_is_correct_l272_272905

def line1 (x y : ℝ) := x - 2 * y + 7 = 0
def line2 (x y : ℝ) := 2 * x + y - 1 = 0

theorem intersection_point_is_correct : line1 (-1) 3 ∧ line2 (-1) 3 :=
by
  sorry

end intersection_point_is_correct_l272_272905


namespace problem1_problem2_l272_272916

-- Define the given sets A and B
def setA (a : ℝ) : Set ℝ := { x | a - 4 < x ∧ x < a + 4 }
def setB : Set ℝ := { x | x < -1 ∨ x > 5 }

-- Problem 1: Prove A ∩ B = { x | -3 < x ∧ x < -1 } when a = 1
theorem problem1 (a : ℝ) (h : a = 1) : 
  (setA a ∩ setB) = { x : ℝ | -3 < x ∧ x < -1 } := sorry

-- Problem 2: Prove range of a given A ∪ B = ℝ is (1, 3)
theorem problem2 (a : ℝ) : 
  (forall x : ℝ, x ∈ (setA a ∪ setB)) ↔ (1 < a ∧ a < 3) := sorry

end problem1_problem2_l272_272916


namespace correct_operations_l272_272118

theorem correct_operations : 
  (∀ x y : ℝ, x^2 + x^4 ≠ x^6) ∧
  (∀ x y : ℝ, 2*x + 4*y ≠ 6*x*y) ∧
  (∀ x : ℝ, x^6 / x^3 = x^3) ∧
  (∀ x : ℝ, (x^3)^2 = x^6) :=
by 
  sorry

end correct_operations_l272_272118


namespace largest_divisor_of_n_l272_272943

theorem largest_divisor_of_n (n : ℕ) (h_pos : 0 < n) (h_div : 360 ∣ n^2) : 60 ∣ n := 
sorry

end largest_divisor_of_n_l272_272943


namespace garage_has_18_wheels_l272_272724

namespace Garage

def bike_wheels_per_bike : ℕ := 2
def bikes_assembled : ℕ := 9

theorem garage_has_18_wheels
  (b : ℕ := bikes_assembled) 
  (w : ℕ := bike_wheels_per_bike) :
  b * w = 18 :=
by
  sorry

end Garage

end garage_has_18_wheels_l272_272724


namespace dice_sum_24_l272_272763

noncomputable def probability_of_sum_24 : ℚ :=
  let die_probability := (1 : ℚ) / 6
  in die_probability ^ 4

theorem dice_sum_24 :
  ∑ x in {x | x ∈ {1, 2, 3, 4, 5, 6} ∧ x = 6} = 24 → probability_of_sum_24 = 1 / 1296 :=
sorry

end dice_sum_24_l272_272763


namespace problem_statement_l272_272247

noncomputable def repeating_decimal_to_fraction (n : ℕ) : ℚ :=
  -- Conversion function for repeating two-digit decimals to fractions
  n / 99

theorem problem_statement :
  (repeating_decimal_to_fraction 63) / (repeating_decimal_to_fraction 21) = 3 :=
by
  -- expected simplification and steps skipped
  sorry

end problem_statement_l272_272247


namespace ratio_of_areas_of_concentric_circles_eq_9_over_4_l272_272861

theorem ratio_of_areas_of_concentric_circles_eq_9_over_4
  (C1 C2 : ℝ)
  (h1 : ∃ Q : ℝ, true) -- Existence of point Q
  (h2 : (30 / 360) * C1 = (45 / 360) * C2) -- Arcs formed by 30-degree and 45-degree angles are equal in length
  : (π * (C1 / (2 * π))^2) / (π * (C2 / (2 * π))^2) = 9 / 4 :=
by
  sorry

end ratio_of_areas_of_concentric_circles_eq_9_over_4_l272_272861


namespace base_form_exists_l272_272034

-- Definitions for three-digit number and its reverse in base g
def N (a b c g : ℕ) : ℕ := a * g^2 + b * g + c
def N_reverse (a b c g : ℕ) : ℕ := c * g^2 + b * g + a

-- The problem statement in Lean
theorem base_form_exists (a b c g : ℕ) (h₁ : 0 ≤ a) (h₂ : 0 ≤ b) (h₃ : 0 ≤ c) (h₄ : 0 < g)
    (h₅ : N a b c g = 2 * N_reverse a b c g) : ∃ k : ℕ, g = 3 * k + 2 ∧ k > 0 :=
by
  sorry

end base_form_exists_l272_272034


namespace prob_A_and_B_truth_l272_272799

-- Define the probabilities
def prob_A_truth := 0.70
def prob_B_truth := 0.60

-- State the theorem
theorem prob_A_and_B_truth : prob_A_truth * prob_B_truth = 0.42 :=
by
  sorry

end prob_A_and_B_truth_l272_272799


namespace probability_of_drawing_diamond_or_ace_l272_272585

-- Define the number of diamonds
def numDiamonds : ℕ := 13

-- Define the number of other Aces
def numOtherAces : ℕ := 3

-- Define the total number of cards in the deck
def totalCards : ℕ := 52

-- Define the number of desirable outcomes (either diamonds or Aces)
def numDesirableOutcomes : ℕ := numDiamonds + numOtherAces

-- Define the probability of drawing a diamond or an Ace
def desiredProbability : ℚ := numDesirableOutcomes / totalCards

theorem probability_of_drawing_diamond_or_ace :
  desiredProbability = 4 / 13 :=
by
  sorry

end probability_of_drawing_diamond_or_ace_l272_272585


namespace tom_tim_typing_ratio_l272_272093

theorem tom_tim_typing_ratio (T M : ℝ) (h1 : T + M = 12) (h2 : T + 1.3 * M = 15) : M / T = 5 :=
by
  sorry

end tom_tim_typing_ratio_l272_272093


namespace prob_sum_24_four_dice_l272_272771

section
open ProbabilityTheory

/-- Define the event E24 as the event where the sum of numbers on the top faces of four six-sided dice is 24 -/
def E24 : Event (StdGen) :=
eventOfFun {ω | ∑ i in range 4, (ω.gen_uniform int (6-1)) + 1 = 24}

/-- Probability that the sum of the numbers on top faces of four six-sided dice is 24 is 1/1296. -/
theorem prob_sum_24_four_dice : ⋆{ P(E24) = 1/1296 } := sorry

end

end prob_sum_24_four_dice_l272_272771


namespace Dana_Colin_relationship_l272_272650

variable (C : ℝ) -- Let C be the number of cards Colin has.

def Ben_cards (C : ℝ) : ℝ := 1.20 * C -- Ben has 20% more cards than Colin
def Dana_cards (C : ℝ) : ℝ := 1.40 * Ben_cards C + Ben_cards C -- Dana has 40% more cards than Ben

theorem Dana_Colin_relationship : Dana_cards C = 1.68 * C := by
  sorry

end Dana_Colin_relationship_l272_272650


namespace other_number_is_7_l272_272317

-- Given conditions
variable (a b : ℤ)
variable (h1 : 2 * a + 3 * b = 110)
variable (h2 : a = 32 ∨ b = 32)

-- The proof goal
theorem other_number_is_7 : (a = 7 ∧ b = 32) ∨ (a = 32 ∧ b = 7) :=
by
  sorry

end other_number_is_7_l272_272317


namespace infer_correct_l272_272166

theorem infer_correct (a b c : ℝ) (h1: c < b) (h2: b < a) (h3: a + b + c = 0) :
  (c * b^2 ≤ ab^2) ∧ (ab > ac) :=
by
  sorry

end infer_correct_l272_272166


namespace solve_for_x_l272_272385

theorem solve_for_x (x : ℚ) (h : x > 0) (hx : 3 * x^2 + 7 * x - 20 = 0) : x = 5 / 3 :=
by 
  sorry

end solve_for_x_l272_272385


namespace abs_inequality_solution_l272_272062

theorem abs_inequality_solution :
  {x : ℝ | |x - 2| + |x + 3| < 7} = {x : ℝ | -4 < x ∧ x < 3} :=
sorry

end abs_inequality_solution_l272_272062


namespace no_common_real_solution_l272_272749

theorem no_common_real_solution :
  ¬ ∃ (x y : ℝ), (x^2 - 6 * x + y + 9 = 0) ∧ (x^2 + 4 * y + 5 = 0) :=
by
  sorry

end no_common_real_solution_l272_272749


namespace right_triangle_third_side_l272_272434

theorem right_triangle_third_side (a b c : ℝ) (h1 : a = 8) (h2 : b = 15) : c = Real.sqrt (b^2 - a^2) :=
by
  rw [h1, h2]
  sorry

end right_triangle_third_side_l272_272434


namespace interest_rate_per_annum_l272_272910

noncomputable def principal : ℝ := 933.3333333333334
noncomputable def amount : ℝ := 1120
noncomputable def time : ℝ := 4

theorem interest_rate_per_annum (P A T : ℝ) (hP : P = principal) (hA : A = amount) (hT : T = time) :
  ∃ R : ℝ, R = 1.25 :=
sorry

end interest_rate_per_annum_l272_272910


namespace circle_center_coordinates_l272_272874

theorem circle_center_coordinates (b c p q : ℝ) 
    (h_circle_eq : ∀ x y : ℝ, x^2 + y^2 - 2 * p * x - 2 * q * y + 2 * q - 1 = 0) 
    (h_quad_roots : ∀ x : ℝ, x^2 + b * x + c = 0) 
    (h_condition : b^2 - 4 * c ≥ 0) : 
    (p = -b / 2) ∧ (q = (1 + c) / 2) := 
sorry

end circle_center_coordinates_l272_272874


namespace focus_of_parabola_l272_272393

theorem focus_of_parabola (x y : ℝ) : (y^2 + 4 * x = 0) → (x = -1 ∧ y = 0) :=
by sorry

end focus_of_parabola_l272_272393


namespace rectangle_area_l272_272092

theorem rectangle_area (b l : ℝ) (h1 : l = 3 * b) (h2 : 2 * (l + b) = 40) : l * b = 75 := by
  sorry

end rectangle_area_l272_272092


namespace find_m_value_l272_272626

theorem find_m_value :
  ∃ m : ℤ, 3 * 2^2000 - 5 * 2^1999 + 4 * 2^1998 - 2^1997 = m * 2^1997 ∧ m = 11 :=
by
  -- The proof would follow here.
  sorry

end find_m_value_l272_272626


namespace count_divisible_by_90_four_digit_numbers_l272_272933

theorem count_divisible_by_90_four_digit_numbers :
  ∃ (n : ℕ), (n = 10) ∧ (∀ (x : ℕ), 1000 ≤ x ∧ x < 10000 ∧ x % 90 = 0 ∧ x % 100 = 90 → (x = 1890 ∨ x = 2790 ∨ x = 3690 ∨ x = 4590 ∨ x = 5490 ∨ x = 6390 ∨ x = 7290 ∨ x = 8190 ∨ x = 9090 ∨ x = 9990)) :=
by
  sorry

end count_divisible_by_90_four_digit_numbers_l272_272933


namespace gcd_g_150_151_l272_272046

def g (x : ℤ) : ℤ := x^2 - 2*x + 3020

theorem gcd_g_150_151 : Int.gcd (g 150) (g 151) = 1 :=
  by
  sorry

end gcd_g_150_151_l272_272046


namespace relationship_y₁_y₂_y₃_l272_272332

variables (y₁ y₂ y₃ : ℝ)

def inverse_proportion (x : ℝ) : ℝ := 3 / x

-- Given points A(-2, y₁), B(-1, y₂), C(1, y₃)
-- and y₁ = inverse_proportion(-2), y₂ = inverse_proportion(-1), y₃ = inverse_proportion(1)
theorem relationship_y₁_y₂_y₃ : 
  let y₁ := inverse_proportion (-2),
      y₂ := inverse_proportion (-1),
      y₃ := inverse_proportion (1) in
  y₂ < y₁ ∧ y₁ < y₃ :=
by
  sorry

end relationship_y₁_y₂_y₃_l272_272332


namespace convert_base_3_to_5_l272_272441

def base_3_to_base_10 (n : Nat) : Nat :=
  let digits := [2, 0, 1, 2, 1]
  digits.foldr (λ d acc => acc * 3 + d) 0

def base_10_to_base_5 (n : Nat) : List Nat :=
  let rec go n acc :=
    if n = 0 then acc else go (n / 5) ((n % 5) :: acc)
  go n []

theorem convert_base_3_to_5 (n : Nat) (h : n = 20121) : 
  base_10_to_base_5 (base_3_to_base_10 n) = [1, 2, 0, 3] :=
by
  sorry

end convert_base_3_to_5_l272_272441


namespace spaces_per_row_l272_272381

theorem spaces_per_row 
  (kind_of_tomatoes : ℕ)
  (tomatoes_per_kind : ℕ)
  (kind_of_cucumbers : ℕ)
  (cucumbers_per_kind : ℕ)
  (potatoes : ℕ)
  (rows : ℕ)
  (additional_spaces : ℕ)
  (h1 : kind_of_tomatoes = 3)
  (h2 : tomatoes_per_kind = 5)
  (h3 : kind_of_cucumbers = 5)
  (h4 : cucumbers_per_kind = 4)
  (h5 : potatoes = 30)
  (h6 : rows = 10)
  (h7 : additional_spaces = 85) :
  (kind_of_tomatoes * tomatoes_per_kind + kind_of_cucumbers * cucumbers_per_kind + potatoes + additional_spaces) / rows = 15 :=
by
  sorry

end spaces_per_row_l272_272381


namespace tom_shirts_total_cost_l272_272552

theorem tom_shirts_total_cost 
  (num_tshirts_per_fandom : ℕ)
  (num_fandoms : ℕ)
  (cost_per_shirt : ℕ)
  (discount_rate : ℚ)
  (tax_rate : ℚ)
  (total_shirts : ℕ := num_tshirts_per_fandom * num_fandoms)
  (discount_per_shirt : ℚ := (cost_per_shirt : ℚ) * discount_rate)
  (cost_per_shirt_after_discount : ℚ := (cost_per_shirt : ℚ) - discount_per_shirt)
  (total_cost_before_tax : ℚ := (total_shirts * cost_per_shirt_after_discount))
  (tax_added : ℚ := total_cost_before_tax * tax_rate)
  (total_amount_paid : ℚ := total_cost_before_tax + tax_added)
  (h1 : num_tshirts_per_fandom = 5)
  (h2 : num_fandoms = 4)
  (h3 : cost_per_shirt = 15) 
  (h4 : discount_rate = 0.2)
  (h5 : tax_rate = 0.1)
  : total_amount_paid = 264 := 
by 
  sorry

end tom_shirts_total_cost_l272_272552


namespace share_per_person_l272_272871

-- Defining the total cost and number of people
def total_cost : ℝ := 12100
def num_people : ℝ := 11

-- The theorem stating that each person's share is $1,100.00
theorem share_per_person : total_cost / num_people = 1100 := by
  sorry

end share_per_person_l272_272871


namespace clea_escalator_time_standing_l272_272120

noncomputable def escalator_time (c : ℕ) : ℝ :=
  let s := (7 * c) / 5
  let d := 72 * c
  let t := d / s
  t

theorem clea_escalator_time_standing (c : ℕ) (h1 : 72 * c = 72 * c) (h2 : 30 * (c + (7 * c) / 5) = 72 * c): escalator_time c = 51 :=
by
  sorry

end clea_escalator_time_standing_l272_272120


namespace average_playtime_in_minutes_l272_272524

noncomputable def lena_playtime_hours : ℝ := 3.5
noncomputable def lena_playtime_minutes : ℝ := lena_playtime_hours * 60
noncomputable def brother_playtime_minutes : ℝ := 1.2 * lena_playtime_minutes + 17
noncomputable def sister_playtime_minutes : ℝ := 1.5 * brother_playtime_minutes

theorem average_playtime_in_minutes :
  (lena_playtime_minutes + brother_playtime_minutes + sister_playtime_minutes) / 3 = 294.17 :=
by
  sorry

end average_playtime_in_minutes_l272_272524


namespace parabola_directrix_is_x_eq_1_l272_272139

noncomputable def parabola_directrix (y : ℝ) : ℝ :=
  -1 / 4 * y^2

theorem parabola_directrix_is_x_eq_1 :
  ∀ x y, x = parabola_directrix y → x = 1 :=
by
  sorry

end parabola_directrix_is_x_eq_1_l272_272139


namespace dice_sum_probability_l272_272757

theorem dice_sum_probability :
  let D := finset.range 1 7  -- outcomes of a fair six-sided die
  (∃! d1 d2 d3 d4 ∈ D, d1 + d2 + d3 + d4 = 24) ->
  (probability(space, {ω ∈ space | (ω 1 = 6) ∧ (ω 2 = 6) ∧ (ω 3 = 6) ∧ (ω 4 = 6)}) = 1/1296) :=
sorry

end dice_sum_probability_l272_272757


namespace polynomial_factors_sum_l272_272625

theorem polynomial_factors_sum (a b : ℝ) 
  (h : ∃ c : ℝ, (∀ x: ℝ, x^3 + a * x^2 + b * x + 8 = (x + 1) * (x + 2) * (x + c))) : 
  a + b = 21 :=
sorry

end polynomial_factors_sum_l272_272625


namespace max_items_per_cycle_l272_272737

theorem max_items_per_cycle (shirts : Nat) (pants : Nat) (sweaters : Nat) (jeans : Nat)
  (cycle_time : Nat) (total_time : Nat) 
  (h_shirts : shirts = 18)
  (h_pants : pants = 12)
  (h_sweaters : sweaters = 17)
  (h_jeans : jeans = 13)
  (h_cycle_time : cycle_time = 45)
  (h_total_time : total_time = 3 * 60) :
  (shirts + pants + sweaters + jeans) / (total_time / cycle_time) = 15 :=
by
  -- We will provide the proof here
  sorry

end max_items_per_cycle_l272_272737


namespace average_one_half_one_fourth_one_eighth_l272_272750

theorem average_one_half_one_fourth_one_eighth : 
  ((1 / 2.0 + 1 / 4.0 + 1 / 8.0) / 3.0) = 7 / 24 := 
by sorry

end average_one_half_one_fourth_one_eighth_l272_272750


namespace equal_contribution_expense_split_l272_272154

theorem equal_contribution_expense_split (Mitch_expense Jam_expense Jay_expense Jordan_expense total_expense each_contribution : ℕ)
  (hmitch : Mitch_expense = 4 * 7)
  (hjam : Jam_expense = (2 * 15) / 10 + 4) -- note: 1.5 dollar per box interpreted as 15/10 to avoid float in Lean
  (hjay : Jay_expense = 3 * 3)
  (hjordan : Jordan_expense = 4 * 2)
  (htotal : total_expense = Mitch_expense + Jam_expense + Jay_expense + Jordan_expense)
  (hequal_split : each_contribution = total_expense / 4) :
  each_contribution = 13 :=
by
  sorry

end equal_contribution_expense_split_l272_272154


namespace ball_drawing_ways_l272_272097

theorem ball_drawing_ways :
  let n := 15
  let k := 4
  ∑(n-k+1 to n) = 32760 :=
by
  sorry

end ball_drawing_ways_l272_272097


namespace original_number_l272_272274

theorem original_number (N y x : ℕ) 
  (h1: N + y = 54321)
  (h2: N = 10 * y + x)
  (h3: 11 * y + x = 54321)
  (h4: x = 54321 % 11)
  (hy: y = 4938) : 
  N = 49383 := 
  by 
  sorry

end original_number_l272_272274


namespace repeating_decimal_sum_l272_272900

theorem repeating_decimal_sum (x : ℚ) (hx : x = 0.417) :
  let num := 46
  let denom := 111
  let sum := num + denom
  sum = 157 :=
by
  sorry

end repeating_decimal_sum_l272_272900


namespace ratio_first_term_l272_272422

theorem ratio_first_term (x : ℝ) (h1 : 60 / 100 = x / 25) : x = 15 := 
sorry

end ratio_first_term_l272_272422


namespace coefficient_of_x_squared_in_expansion_l272_272293

noncomputable def polynomial_expansion (p : Polynomial ℚ) (n : ℕ) : Polynomial ℚ :=
Polynomial.X ^ n + (-3) * Polynomial.C 1 * Polynomial.X ^ (n/2) + Polynomial.X * Polynomial.C 2 * Polynomial.X

theorem coefficient_of_x_squared_in_expansion : 
  (polynomial_expansion (Polynomial.X ^ 2 - 3 * Polynomial.X + Polynomial.C 2) 4).coeff 2 = 248 :=
sorry

end coefficient_of_x_squared_in_expansion_l272_272293


namespace find_a2_l272_272043

def arithmetic_sequence (a : ℕ → ℤ) (a1 : ℤ) (d : ℤ) : Prop :=
  a 1 = a1 ∧ ∀ n : ℕ, a (n + 1) = a n + d 

def sum_arithmetic_sequence (S : ℕ → ℤ) (a : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, S n = n * (a 1 + a n) / 2

theorem find_a2 (a : ℕ → ℤ) (S : ℕ → ℤ) (a1 : ℤ) (d : ℤ) 
  (h1 : arithmetic_sequence a a1 d)
  (h2 : sum_arithmetic_sequence S a)
  (h3 : a1 = -2010)
  (h4 : (S 2010) / 2010 - (S 2008) / 2008 = 2) :
  a 2 = -2008 :=
sorry

end find_a2_l272_272043


namespace directrix_eqn_of_parabola_l272_272146

theorem directrix_eqn_of_parabola : 
  ∀ y : ℝ, x = - (1 / 4 : ℝ) * y ^ 2 → x = 1 :=
by
  sorry

end directrix_eqn_of_parabola_l272_272146


namespace simplify_expression_l272_272790

theorem simplify_expression (x y : ℝ) (h : x ≠ 0 ∧ y ≠ 0) :
  ((x^3 + 2) / x * (y^3 + 2) / y) - ((x^3 - 2) / y * (y^3 - 2) / x) = 4 * (x^2 / y + y^2 / x) :=
by sorry

end simplify_expression_l272_272790


namespace find_b_l272_272015

noncomputable def a_and_b_integers_and_factor (a b : ℤ) : Prop :=
  ∀ (x : ℝ), (x^2 - x - 1) * (a*x^3 + b*x^2 - x + 1) = 0

theorem find_b (a b : ℤ) (h : a_and_b_integers_and_factor a b) : b = -1 :=
by 
  sorry

end find_b_l272_272015


namespace Drew_age_is_12_l272_272515

def Sam_age_current : ℕ := 46
def Sam_age_in_five_years : ℕ := Sam_age_current + 5

def Drew_age_now (D : ℕ) : Prop :=
  Sam_age_in_five_years = 3 * (D + 5)

theorem Drew_age_is_12 (D : ℕ) (h : Drew_age_now D) : D = 12 :=
by
  sorry

end Drew_age_is_12_l272_272515


namespace necessary_but_not_sufficient_condition_l272_272953

variable {a : ℕ → ℝ}
variable {q : ℝ}

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

theorem necessary_but_not_sufficient_condition
    (a1_pos : a 1 > 0)
    (geo_seq : geometric_sequence a q)
    (a3_lt_a6 : a 3 < a 6) :
  (a 1 < a 3) ↔ ∃ k : ℝ, k > 1 ∧ a 1 * k^2 < a 1 * k^5 :=
by
  sorry

end necessary_but_not_sufficient_condition_l272_272953


namespace count_divisible_by_90_four_digit_numbers_l272_272934

theorem count_divisible_by_90_four_digit_numbers :
  ∃ (n : ℕ), (n = 10) ∧ (∀ (x : ℕ), 1000 ≤ x ∧ x < 10000 ∧ x % 90 = 0 ∧ x % 100 = 90 → (x = 1890 ∨ x = 2790 ∨ x = 3690 ∨ x = 4590 ∨ x = 5490 ∨ x = 6390 ∨ x = 7290 ∨ x = 8190 ∨ x = 9090 ∨ x = 9990)) :=
by
  sorry

end count_divisible_by_90_four_digit_numbers_l272_272934


namespace Jerry_paid_more_last_month_l272_272968

def Debt_total : ℕ := 50
def Debt_remaining : ℕ := 23
def Paid_2_months_ago : ℕ := 12
def Paid_last_month : ℕ := 27 - Paid_2_months_ago

theorem Jerry_paid_more_last_month :
  Paid_last_month - Paid_2_months_ago = 3 :=
by
  -- Calculation for Paid_last_month
  have h : Paid_last_month = 27 - 12 := by rfl
  -- Compute the difference
  have diff : 15 - 12 = 3 := by rfl
  exact diff

end Jerry_paid_more_last_month_l272_272968


namespace sqrt_inequality_l272_272634

theorem sqrt_inequality (x : ℝ) (h : ∀ r : ℝ, r = 2 * x - 1 → r ≥ 0) : x ≥ 1 / 2 :=
sorry

end sqrt_inequality_l272_272634


namespace eval_expression_l272_272240

theorem eval_expression : 3 ^ 2 - (4 * 2) = 1 :=
by
  sorry

end eval_expression_l272_272240


namespace pure_alcohol_to_add_l272_272265

-- Variables and known values
variables (x : ℝ) -- amount of pure alcohol added
def initial_volume : ℝ := 6 -- initial solution volume in liters
def initial_concentration : ℝ := 0.35 -- initial alcohol concentration
def target_concentration : ℝ := 0.50 -- target alcohol concentration

-- Conditions
def initial_pure_alcohol : ℝ := initial_volume * initial_concentration

-- Statement of the problem
theorem pure_alcohol_to_add :
  (2.1 + x) / (initial_volume + x) = target_concentration ↔ x = 1.8 :=
by
  sorry

end pure_alcohol_to_add_l272_272265


namespace work_completion_days_l272_272117

theorem work_completion_days (D : ℕ) (W : ℕ) :
  (D : ℕ) = 6 :=
by 
  -- define constants and given conditions
  let original_men := 10
  let additional_men := 10
  let early_days := 3

  -- define the premise
  -- work done with original men in original days
  have work_done_original : W = (original_men * D) := sorry
  -- work done with additional men in reduced days
  have work_done_with_additional : W = ((original_men + additional_men) * (D - early_days)) := sorry

  -- prove the equality from the condition
  have eq : original_men * D = (original_men + additional_men) * (D - early_days) := sorry

  -- simplify to solve for D
  have solution : D = 6 := sorry

  exact solution

end work_completion_days_l272_272117


namespace sector_area_l272_272545

/--
The area of a sector with radius 6cm and central angle 15° is (3 * π / 2) cm².
-/
theorem sector_area (R : ℝ) (θ : ℝ) (h_radius : R = 6) (h_angle : θ = 15) :
    (S : ℝ) = (3 * Real.pi / 2) := by
  sorry

end sector_area_l272_272545


namespace arrangement_count_is_74_l272_272996

def count_valid_arrangements : Nat :=
  74

-- Lean statement for the proof
theorem arrangement_count_is_74 :
  let seven_cards := list.range' 1 7 in
  ∃ seq : list Nat, 
    (seq.length = 7) ∧ 
    (∀ n, list.erase seq n = list.range' 1 6 ∨ 
          (list.reverse (list.erase seq n) = list.range' 1 6)) ∧
    (count_valid_arrangements = 74) :=
by
  let seven_cards := list.range' 1 7
  existsi seven_cards
  split
  -- Provide the conditions here for Lean to handle
  sorry

end arrangement_count_is_74_l272_272996


namespace bruce_total_payment_l272_272868

-- Define the conditions
def quantity_grapes : Nat := 7
def rate_grapes : Nat := 70
def quantity_mangoes : Nat := 9
def rate_mangoes : Nat := 55

-- Define the calculation for total amount paid
def total_amount_paid : Nat :=
  (quantity_grapes * rate_grapes) + (quantity_mangoes * rate_mangoes)

-- Proof statement
theorem bruce_total_payment : total_amount_paid = 985 :=
by
  -- Proof steps would go here
  sorry

end bruce_total_payment_l272_272868


namespace infinite_primes_of_form_4n_plus_3_l272_272530

theorem infinite_primes_of_form_4n_plus_3 :
  ∀ (S : Finset ℕ), (∀ p ∈ S, Prime p ∧ p % 4 = 3) →
  ∃ q, Prime q ∧ q % 4 = 3 ∧ q ∉ S :=
by 
  sorry

end infinite_primes_of_form_4n_plus_3_l272_272530


namespace sum_even_probability_l272_272529

def probability_even_sum_of_wheels : ℚ :=
  let prob_wheel1_odd := 3 / 5
  let prob_wheel1_even := 2 / 5
  let prob_wheel2_odd := 2 / 3
  let prob_wheel2_even := 1 / 3
  (prob_wheel1_odd * prob_wheel2_odd) + (prob_wheel1_even * prob_wheel2_even)

theorem sum_even_probability :
  probability_even_sum_of_wheels = 8 / 15 :=
by
  -- Goal statement with calculations showed in the equivalent problem
  sorry

end sum_even_probability_l272_272529


namespace exponentiation_division_l272_272302

variable (a b : ℝ)

theorem exponentiation_division (a b : ℝ) : ((2 * a) / b) ^ 4 = (16 * a ^ 4) / (b ^ 4) := by
  sorry

end exponentiation_division_l272_272302


namespace largest_independent_amount_l272_272357

theorem largest_independent_amount (n : ℕ) :
  ∃ s, ¬∃ a b c d e f g h i j : ℕ, s = a * (3^n) + b * (3^(n-1) * 5) + c * (3^(n-2) * 5^2) + d * (3^(n-3) * 5^3) + 
        e * (3^(n-4) * 5^4) + f * (3^(n-5) * 5^5) + g * (3^(n-6) * 5^6) + h * (3^(n-7) * 5^7) + i * (3^(n-8) * 5^8) + 
        j * (5^n) := (5^(n+1)) - 2 * (3^(n+1)) :=
sorry

end largest_independent_amount_l272_272357


namespace triangle_cosine_sine_inequality_l272_272350

theorem triangle_cosine_sine_inequality (A B C : ℝ) (h : A + B + C = Real.pi) 
  (hA : 0 < A) (hB : 0 < B) (hC : 0 < C) 
  (hA_lt_pi : A < Real.pi)
  (hB_lt_pi : B < Real.pi)
  (hC_lt_pi : C < Real.pi) :
  Real.cos A * (Real.sin B + Real.sin C) ≥ -2 * Real.sqrt 6 / 9 := 
by
  sorry

end triangle_cosine_sine_inequality_l272_272350


namespace smallest_positive_integer_satisfying_condition_l272_272249

-- Define the condition
def isConditionSatisfied (n : ℕ) : Prop :=
  (Real.sqrt n - Real.sqrt (n - 1) < 0.01) ∧ n > 0

-- State the theorem
theorem smallest_positive_integer_satisfying_condition :
  ∃ n : ℕ, isConditionSatisfied n ∧ (∀ m : ℕ, isConditionSatisfied m → n ≤ m) ∧ n = 2501 :=
by
  sorry

end smallest_positive_integer_satisfying_condition_l272_272249


namespace part_one_part_two_part_three_l272_272486

def f(x : ℝ) := x^2 - 1
def g(a x : ℝ) := a * |x - 1|

-- (I)
theorem part_one (a : ℝ) : 
  ((∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ |f x₁| = g a x₁ ∧ |f x₂| = g a x₂) ↔ (a = 0 ∨ a = 2)) :=
sorry

-- (II)
theorem part_two (a : ℝ) : 
  (∀ x : ℝ, f x ≥ g a x) ↔ (a <= -2) :=
sorry

-- (III)
def G(a x : ℝ) := |f x| + g a x

theorem part_three (a : ℝ) (h : a < 0) : 
  (∀ x ∈ [-2, 2], G a x ≤ if a <= -3 then 0 else 3 + a) :=
sorry

end part_one_part_two_part_three_l272_272486


namespace skateboarded_one_way_distance_l272_272969

-- Define the total skateboarded distance and the walked distance.
def total_skateboarded : ℕ := 24
def walked_distance : ℕ := 4

-- Define the proof theorem.
theorem skateboarded_one_way_distance : 
    (total_skateboarded - walked_distance) / 2 = 10 := 
by sorry

end skateboarded_one_way_distance_l272_272969


namespace arithmetic_sequence_sum_l272_272238

theorem arithmetic_sequence_sum (S : ℕ → ℝ) (S_10_eq : S 10 = 20) (S_20_eq : S 20 = 15) :
  S 30 = -15 :=
by
  sorry

end arithmetic_sequence_sum_l272_272238


namespace intersection_point_for_m_l272_272161

variable (n : ℕ) (x_0 y_0 : ℕ)
variable (h₁ : n ≥ 2)
variable (h₂ : y_0 ^ 2 = n * x_0 - 1)
variable (h₃ : y_0 = x_0)

theorem intersection_point_for_m (m : ℕ) (hm : 0 < m) : ∃ k : ℕ, k ≥ 2 ∧ (y_0 ^ m = x_0 ^ m) ∧ (y_0 ^ m) ^ 2 = k * (x_0 ^ m) - 1 :=
by
  sorry

end intersection_point_for_m_l272_272161


namespace sufficient_but_not_necessary_l272_272624

theorem sufficient_but_not_necessary (a b : ℝ) (h : a > b ∧ b > 0) : a^2 > b^2 ∧ ¬ (a^2 > b^2 → a > b ∧ b > 0) :=
by
  sorry

end sufficient_but_not_necessary_l272_272624


namespace each_shopper_will_receive_amount_l272_272359

/-- Definitions of the given conditions -/
def isabella_has_more_than_sam : ℕ := 45
def isabella_has_more_than_giselle : ℕ := 15
def giselle_money : ℕ := 120

/-- Calculation based on the provided conditions -/
def isabella_money : ℕ := giselle_money + isabella_has_more_than_giselle
def sam_money : ℕ := isabella_money - isabella_has_more_than_sam
def total_money : ℕ := isabella_money + sam_money + giselle_money

/-- The total amount each shopper will receive when the donation is shared equally -/
def money_each_shopper_receives : ℕ := total_money / 3

/-- Main theorem to prove the statement derived from the problem -/
theorem each_shopper_will_receive_amount :
  money_each_shopper_receives = 115 := by
  sorry

end each_shopper_will_receive_amount_l272_272359


namespace cos_C_equal_two_thirds_l272_272964

variable {A B C : ℝ}
variable {a b c : ℝ}

-- Define the conditions
def condition1 : a > 0 ∧ b > 0 ∧ c > 0 := sorry
def condition2 : (a / b) + (b / a) = 4 * Real.cos C := sorry
def condition3 : Real.cos (A - B) = 1 / 6 := sorry

-- Statement to prove
theorem cos_C_equal_two_thirds 
  (h1: a > 0 ∧ b > 0 ∧ c > 0) 
  (h2: (a / b) + (b / a) = 4 * Real.cos C) 
  (h3: Real.cos (A - B) = 1 / 6) 
  : Real.cos C = 2 / 3 :=
  sorry

end cos_C_equal_two_thirds_l272_272964


namespace no_int_solutions_l272_272058

theorem no_int_solutions (c x y : ℤ) (h1 : 0 < c) (h2 : c % 2 = 1) : x ^ 2 - y ^ 3 ≠ (2 * c) ^ 3 - 1 :=
sorry

end no_int_solutions_l272_272058


namespace repeating_decimal_division_l272_272246

theorem repeating_decimal_division :
  (let r : ℚ := 1/99 in
   let x : ℚ := 63 * r in
   let y : ℚ := 21 * r in
   x / y = 3) :=
by {
  sorry
}

end repeating_decimal_division_l272_272246


namespace intersection_point_of_lines_l272_272596

noncomputable def line1 (x : ℝ) : ℝ := 3 * x - 4

noncomputable def line2 (x : ℝ) : ℝ := -1 / 3 * x + 10 / 3

def point : ℝ × ℝ := (4, 2)

theorem intersection_point_of_lines :
  ∃ (x y : ℝ), line1 x = y ∧ line2 x = y ∧ (x, y) = (2.2, 2.6) :=
by
  sorry

end intersection_point_of_lines_l272_272596


namespace algebraic_expression_value_l272_272158

noncomputable def a : ℝ := 1 + Real.sqrt 2
noncomputable def b : ℝ := 1 - Real.sqrt 2

theorem algebraic_expression_value :
  let a := 1 + Real.sqrt 2
  let b := 1 - Real.sqrt 2
  a^2 - a * b + b^2 = 7 := by
  sorry

end algebraic_expression_value_l272_272158


namespace problem1_problem2_l272_272170

def A := { x : ℝ | -2 < x ∧ x ≤ 4 }
def B := { x : ℝ | 2 - x < 1 }
def U := ℝ
def complement_B := { x : ℝ | x ≤ 1 }

theorem problem1 : { x : ℝ | 1 < x ∧ x ≤ 4 } = { x : ℝ | x ∈ A ∧ x ∈ B } := 
by sorry

theorem problem2 : { x : ℝ | x ≤ 4 } = { x : ℝ | x ∈ A ∨ x ∈ complement_B } := 
by sorry

end problem1_problem2_l272_272170


namespace abs_f_x_minus_f_a_lt_l272_272053

variable {R : Type*} [LinearOrderedField R]

def f (x : R) (c : R) := x ^ 2 - x + c

theorem abs_f_x_minus_f_a_lt (x a c : R) (h : abs (x - a) < 1) : 
  abs (f x c - f a c) < 2 * (abs a + 1) :=
by
  sorry

end abs_f_x_minus_f_a_lt_l272_272053


namespace sale_in_first_month_l272_272725

theorem sale_in_first_month
  (s2 : ℕ)
  (s3 : ℕ)
  (s4 : ℕ)
  (s5 : ℕ)
  (s6 : ℕ)
  (required_total_sales : ℕ)
  (average_sales : ℕ)
  : (required_total_sales = 39000) → 
    (average_sales = 6500) → 
    (s2 = 6927) →
    (s3 = 6855) →
    (s4 = 7230) →
    (s5 = 6562) →
    (s6 = 4991) →
    s2 + s3 + s4 + s5 + s6 = 32565 →
    required_total_sales - (s2 + s3 + s4 + s5 + s6) = 6435 :=
by
  intros
  sorry

end sale_in_first_month_l272_272725


namespace count_divisible_by_90_l272_272936

theorem count_divisible_by_90 : 
  ∃ n, n = 10 ∧ (∀ k, 1000 ≤ k ∧ k < 10000 ∧ k % 100 = 90 ∧ k % 90 = 0 → n = 10) :=
begin
  sorry
end

end count_divisible_by_90_l272_272936


namespace infinite_geometric_series_correct_l272_272902

noncomputable def infinite_geometric_series_sum : ℚ :=
  let a : ℚ := 5 / 3
  let r : ℚ := -9 / 20
  a / (1 - r)

theorem infinite_geometric_series_correct : infinite_geometric_series_sum = 100 / 87 := 
by
  sorry

end infinite_geometric_series_correct_l272_272902


namespace prime_gt_3_divides_exp_l272_272818

theorem prime_gt_3_divides_exp (p : ℕ) (hprime : Nat.Prime p) (hgt3 : p > 3) :
  42 * p ∣ 3^p - 2^p - 1 :=
sorry

end prime_gt_3_divides_exp_l272_272818


namespace simplify_expression_l272_272534

theorem simplify_expression (n : ℕ) :
  (2^(n+5) - 3 * 2^n) / (3 * 2^(n+3)) = 29 / 24 :=
by
  sorry

end simplify_expression_l272_272534


namespace find_number_l272_272109

theorem find_number {x : ℤ} (h : x + 5 = 6) : x = 1 :=
sorry

end find_number_l272_272109


namespace Olivia_score_l272_272985

theorem Olivia_score 
  (n : ℕ) (m : ℕ) (average20 : ℕ) (average21 : ℕ)
  (h_n : n = 20) (h_m : m = 21) (h_avg20 : average20 = 85) (h_avg21 : average21 = 86)
  : ∃ (scoreOlivia : ℕ), scoreOlivia = m * average21 - n * average20 :=
by
  sorry

end Olivia_score_l272_272985


namespace num_distinct_integers_formed_l272_272500

theorem num_distinct_integers_formed (digits : Multiset ℕ) (h : digits = {2, 2, 3, 3, 3}) : 
  Multiset.card (Multiset.powerset digits).attach = 10 := 
by {
  sorry
}

end num_distinct_integers_formed_l272_272500


namespace ch_sub_ch_add_sh_sub_sh_add_l272_272338

noncomputable def sh (x : ℝ) : ℝ := (Real.exp x - Real.exp (-x)) / 2
noncomputable def ch (x : ℝ) : ℝ := (Real.exp x + Real.exp (-x)) / 2

theorem ch_sub (x y : ℝ) : ch (x - y) = ch x * ch y - sh x * sh y := sorry
theorem ch_add (x y : ℝ) : ch (x + y) = ch x * ch y + sh x * sh y := sorry
theorem sh_sub (x y : ℝ) : sh (x - y) = sh x * ch y - ch x * sh y := sorry
theorem sh_add (x y : ℝ) : sh (x + y) = sh x * ch y + ch x * sh y := sorry

end ch_sub_ch_add_sh_sub_sh_add_l272_272338


namespace minimum_value_y_l272_272693

theorem minimum_value_y (x : ℝ) (h : x ≥ 1) : 5*x^2 - 8*x + 20 ≥ 13 :=
by {
  sorry
}

end minimum_value_y_l272_272693


namespace rectangle_length_twice_breadth_l272_272873

theorem rectangle_length_twice_breadth
  (b : ℝ) 
  (l : ℝ)
  (h1 : l = 2 * b)
  (h2 : (l - 5) * (b + 4) = l * b + 75) :
  l = 190 / 3 :=
sorry

end rectangle_length_twice_breadth_l272_272873


namespace green_competition_l272_272510

theorem green_competition {x : ℕ} (h : 0 ≤ x ∧ x ≤ 25) : 
  5 * x - (25 - x) ≥ 85 :=
by
  sorry

end green_competition_l272_272510


namespace necessarily_positive_y_plus_z_l272_272382

-- Given conditions
variables {x y z : ℝ}

-- Assert the conditions
axiom hx : 0 < x ∧ x < 1
axiom hy : -1 < y ∧ y < 0
axiom hz : 1 < z ∧ z < 2

-- Prove that y + z is necessarily positive
theorem necessarily_positive_y_plus_z : y + z > 0 :=
by
  sorry

end necessarily_positive_y_plus_z_l272_272382


namespace find_original_number_l272_272280

open Int

theorem find_original_number (N y x : ℕ) (h1 : N = 10 * y + x) (h2 : N + y = 54321) (h3 : x = 54321 % 11) (h4 : y = 4938) : N = 49383 := by
  sorry

end find_original_number_l272_272280


namespace area_of_lit_plot_l272_272089

noncomputable def litArea (r : ℝ) : ℝ := (π * r^2) / 4

theorem area_of_lit_plot :
  let radius := 21 in
  litArea 21 = 110.25 * π :=
by
  sorry

end area_of_lit_plot_l272_272089


namespace determine_valid_m_l272_272337

-- The function given in the problem
def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 + x + m + 2

-- The range of values for m
def valid_m (m : ℝ) : Prop := -1/4 ≤ m ∧ m ≤ 0

-- The condition that f is increasing on (-∞, 2)
def increasing_on_interval (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x y : ℝ, x < y → y < a → f x ≤ f y

-- The main statement we want to prove
theorem determine_valid_m (m : ℝ) :
  increasing_on_interval (f m) 2 ↔ valid_m m :=
sorry

end determine_valid_m_l272_272337


namespace train_problem_l272_272260

variables (x : ℝ) (p q : ℝ)
variables (speed_p speed_q : ℝ) (dist_diff : ℝ)

theorem train_problem
  (speed_p : speed_p = 50)
  (speed_q : speed_q = 40)
  (dist_diff : ∀ x, x = 500 → p = 50 * x ∧ q = 40 * (500 - 100)) :
  p + q = 900 :=
by
sorry

end train_problem_l272_272260


namespace cats_weigh_more_than_puppies_l272_272012

noncomputable def weight_puppy_A : ℝ := 6.5
noncomputable def weight_puppy_B : ℝ := 7.2
noncomputable def weight_puppy_C : ℝ := 8
noncomputable def weight_puppy_D : ℝ := 9.5
noncomputable def weight_cat : ℝ := 2.8
noncomputable def num_cats : ℕ := 16

theorem cats_weigh_more_than_puppies :
  (num_cats * weight_cat) - (weight_puppy_A + weight_puppy_B + weight_puppy_C + weight_puppy_D) = 13.6 :=
by
  sorry

end cats_weigh_more_than_puppies_l272_272012


namespace lcm_of_18_and_24_l272_272753

noncomputable def lcm_18_24 : ℕ :=
  Nat.lcm 18 24

theorem lcm_of_18_and_24 : lcm_18_24 = 72 :=
by
  sorry

end lcm_of_18_and_24_l272_272753


namespace absolute_value_condition_l272_272537

theorem absolute_value_condition (x : ℝ) : |x + 1| + |x - 2| ≤ 5 ↔ -2 ≤ x ∧ x ≤ 3 := sorry

end absolute_value_condition_l272_272537


namespace find_cone_height_l272_272474

noncomputable def cone_height (A l : ℝ) : ℝ := 
  let r := A / (l * Real.pi) in
  Real.sqrt (l^2 - r^2)

theorem find_cone_height : cone_height (65 * Real.pi) 13 = 12 := by
  let r := 5
  have h_eq : cone_height (65 * Real.pi) 13 = Real.sqrt (13^2 - r^2) := by 
    unfold cone_height
    sorry -- This step would carry out the necessary substeps.
  calc
    cone_height (65 * Real.pi) 13 = Real.sqrt (13^2 - r^2) : by exact h_eq
                         ... = Real.sqrt 144 : by norm_num
                         ... = 12 : by norm_num

end find_cone_height_l272_272474


namespace abs_inequality_proof_by_contradiction_l272_272408

theorem abs_inequality_proof_by_contradiction (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  |a| > |b| :=
by
  let h := |a| ≤ |b|
  sorry

end abs_inequality_proof_by_contradiction_l272_272408


namespace correct_speed_to_reach_on_time_l272_272984

theorem correct_speed_to_reach_on_time
  (d : ℝ)
  (t : ℝ)
  (h1 : d = 50 * (t + 1 / 12))
  (h2 : d = 70 * (t - 1 / 12)) :
  d / t = 58 := 
by
  sorry

end correct_speed_to_reach_on_time_l272_272984


namespace cards_arrangement_count_l272_272989

theorem cards_arrangement_count : 
  let cards := [1, 2, 3, 4, 5, 6, 7] in
  let valid_arrangements := 
    {arrangement | ∃ removed, 
      removed ∈ cards ∧ 
      (∀ remaining, 
        remaining = cards.erase removed → 
        (sorted remaining ∨ sorted (remaining.reverse))) } in
  valid_arrangements.card = 26 :=
sorry

end cards_arrangement_count_l272_272989


namespace carousel_rotation_time_l272_272106

-- Definitions and Conditions
variables (a v U x : ℝ)

-- Conditions given in the problem
def condition1 : Prop := (U * a - v * a = 2 * Real.pi)
def condition2 : Prop := (v * a = U * (x - a / 2))

-- Statement to prove
theorem carousel_rotation_time :
  condition1 a v U ∧ condition2 a v U x → x = 2 * a / 3 :=
by
  intro h
  have c1 := h.1
  have c2 := h.2
  sorry

end carousel_rotation_time_l272_272106


namespace complete_square_solution_l272_272220

theorem complete_square_solution (x : ℝ) :
  x^2 - 8 * x + 6 = 0 → (x - 4)^2 = 10 :=
by
  intro h
  -- Proof would go here
  sorry

end complete_square_solution_l272_272220


namespace trig_identity_l272_272918

theorem trig_identity (α : ℝ) (h : Real.tan α = 2 / 3) : 
  Real.sin (2 * α) - Real.cos (π - 2 * α) = 17 / 13 :=
by
  sorry

end trig_identity_l272_272918


namespace find_x_l272_272804

theorem find_x (x : ℤ) :
  3 < x ∧ x < 10 →
  5 < x ∧ x < 18 →
  -2 < x ∧ x < 9 →
  0 < x ∧ x < 8 →
  x + 1 < 9 →
  x = 6 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end find_x_l272_272804


namespace x_not_4_17_percent_less_than_z_x_is_8_0032_percent_less_than_z_l272_272507

def y_is_60_percent_greater_than_x (x y : ℝ) : Prop :=
  y = 1.60 * x

def z_is_40_percent_less_than_y (y z : ℝ) : Prop :=
  z = 0.60 * y

theorem x_not_4_17_percent_less_than_z (x y z : ℝ) (h1 : y_is_60_percent_greater_than_x x y) (h2 : z_is_40_percent_less_than_y y z) : 
  x ≠ 0.9583 * z :=
by {
  sorry
}

theorem x_is_8_0032_percent_less_than_z (x y z : ℝ) (h1 : y_is_60_percent_greater_than_x x y) (h2 : z_is_40_percent_less_than_y y z) : 
  x = 0.919968 * z :=
by {
  sorry
}

end x_not_4_17_percent_less_than_z_x_is_8_0032_percent_less_than_z_l272_272507


namespace base_8_subtraction_l272_272449

theorem base_8_subtraction : 
  let x := 0o1234   -- 1234 in base 8
  let y := 0o765    -- 765 in base 8
  let result := 0o225 -- 225 in base 8
  x - y = result := by sorry

end base_8_subtraction_l272_272449


namespace total_cost_is_67_15_l272_272235

noncomputable def calculate_total_cost : ℝ :=
  let caramel_cost := 3
  let candy_bar_cost := 2 * caramel_cost
  let cotton_candy_cost := (candy_bar_cost * 4) / 2
  let chocolate_bar_cost := candy_bar_cost + caramel_cost
  let lollipop_cost := candy_bar_cost / 3

  let candy_bar_total := 6 * candy_bar_cost
  let caramel_total := 3 * caramel_cost
  let cotton_candy_total := 1 * cotton_candy_cost
  let chocolate_bar_total := 2 * chocolate_bar_cost
  let lollipop_total := 2 * lollipop_cost

  let discounted_candy_bar_total := candy_bar_total * 0.9
  let discounted_caramel_total := caramel_total * 0.85
  let discounted_cotton_candy_total := cotton_candy_total * 0.8
  let discounted_chocolate_bar_total := chocolate_bar_total * 0.75
  let discounted_lollipop_total := lollipop_total -- No additional discount

  discounted_candy_bar_total +
  discounted_caramel_total +
  discounted_cotton_candy_total +
  discounted_chocolate_bar_total +
  discounted_lollipop_total

theorem total_cost_is_67_15 : calculate_total_cost = 67.15 := by
  sorry

end total_cost_is_67_15_l272_272235


namespace correct_quotient_l272_272185

theorem correct_quotient (x : ℕ) (hx1: x = 12 * 63)
                          (hx2 : x = 18 * 112) 
                          (hx3 : x = 24 * 84)
                          (hdiv21 : x % 21 = 0)
                          (hdiv27 : x % 27 = 0)
                          (hdiv36 : x % 36 = 0) :
    x / 21 = 96 :=
by sorry

end correct_quotient_l272_272185


namespace smallest_B_l272_272516

-- Definitions and conditions
def known_digit_sum : Nat := 4 + 8 + 3 + 9 + 4 + 2
def divisible_by_3 (n : Nat) : Bool := n % 3 = 0

-- Statement to prove
theorem smallest_B (B : Nat) (h : B < 10) (hdiv : divisible_by_3 (B + known_digit_sum)) : B = 0 :=
sorry

end smallest_B_l272_272516


namespace min_boxes_eliminated_l272_272028

theorem min_boxes_eliminated :
  ∃ (x : ℕ), 30 - x ≥ 0 ∧
  (7 : ℚ) / (30 - x : ℚ) ≥ 2 / 3 ∧
  x ≥ 20 := by
sorry

end min_boxes_eliminated_l272_272028


namespace find_common_chord_l272_272614

-- Define the two circles
def C1 (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 8*y - 8 = 0
def C2 (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 4*y - 2 = 0

-- The common chord is the line we need to prove
def CommonChord (x y : ℝ) : Prop := x + 2*y - 1 = 0

-- The theorem stating that the common chord is the line x + 2*y - 1 = 0
theorem find_common_chord (x y : ℝ) (p : C1 x y ∧ C2 x y) : CommonChord x y :=
sorry

end find_common_chord_l272_272614


namespace seats_per_bus_l272_272401

theorem seats_per_bus (students : ℕ) (buses : ℕ) (h1 : students = 111) (h2 : buses = 37) : students / buses = 3 := by
  sorry

end seats_per_bus_l272_272401


namespace dice_sum_24_l272_272762

noncomputable def probability_of_sum_24 : ℚ :=
  let die_probability := (1 : ℚ) / 6
  in die_probability ^ 4

theorem dice_sum_24 :
  ∑ x in {x | x ∈ {1, 2, 3, 4, 5, 6} ∧ x = 6} = 24 → probability_of_sum_24 = 1 / 1296 :=
sorry

end dice_sum_24_l272_272762


namespace parabola_directrix_is_x_eq_1_l272_272138

noncomputable def parabola_directrix (y : ℝ) : ℝ :=
  -1 / 4 * y^2

theorem parabola_directrix_is_x_eq_1 :
  ∀ x y, x = parabola_directrix y → x = 1 :=
by
  sorry

end parabola_directrix_is_x_eq_1_l272_272138


namespace solution_set_inequality1_solution_set_inequality2_l272_272708

def inequality1 (x : ℝ) : Prop := (2 * x + 1) / (3 - x) ≥ 0
def inequality2 (x : ℝ) : Prop := (2 * x + 1) / (x - 3) ≤ 0

theorem solution_set_inequality1 : {x : ℝ | (-1 / 2 : ℝ) <= x ∧ x < 3} = {x : ℝ | inequality1 x} :=
sorry

theorem solution_set_inequality2 : {x : ℝ | (-1 / 2 : ℝ) <= x ∧ x < 3} = {x : ℝ | inequality2 x} :=
sorry

end solution_set_inequality1_solution_set_inequality2_l272_272708


namespace max_S_R_squared_l272_272806

theorem max_S_R_squared (a b c : ℝ) (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) :
  (∃ a b c, DA = a ∧ DB = b ∧ DC = c ∧ S = 2 * (a * b + b * c + c * a) ∧
  R = (Real.sqrt (a^2 + b^2 + c^2)) / 2 ∧ (∃ max_val, max_val = (2 / 3) * (3 + Real.sqrt 3))) :=
sorry

end max_S_R_squared_l272_272806


namespace investment_ratio_l272_272890

theorem investment_ratio (A_invest B_invest C_invest : ℝ) (F : ℝ) (total_profit B_share : ℝ)
  (h1 : A_invest = 3 * B_invest)
  (h2 : B_invest = F * C_invest)
  (h3 : total_profit = 7700)
  (h4 : B_share = 1400)
  (h5 : (B_invest / (A_invest + B_invest + C_invest)) * total_profit = B_share) :
  (B_invest / C_invest) = 2 / 3 := 
by
  sorry

end investment_ratio_l272_272890


namespace M_is_subset_of_N_l272_272171

theorem M_is_subset_of_N : 
  ∀ (x y : ℝ), (|x| + |y| < 1) → 
    (Real.sqrt ((x - 1/2)^2 + (y + 1/2)^2) + Real.sqrt ((x + 1/2)^2 + (y - 1/2)^2) < 2 * Real.sqrt 2) :=
by
  intro x y h
  sorry

end M_is_subset_of_N_l272_272171


namespace medium_pizza_slices_l272_272888

theorem medium_pizza_slices (M : ℕ) 
  (small_pizza_slices : ℕ := 6)
  (large_pizza_slices : ℕ := 12)
  (total_pizzas : ℕ := 15)
  (small_pizzas : ℕ := 4)
  (medium_pizzas : ℕ := 5)
  (total_slices : ℕ := 136) :
  (small_pizzas * small_pizza_slices) + (medium_pizzas * M) + ((total_pizzas - small_pizzas - medium_pizzas) * large_pizza_slices) = total_slices → 
  M = 8 :=
by
  intro h
  sorry

end medium_pizza_slices_l272_272888


namespace work_rate_sum_l272_272709

theorem work_rate_sum (A B : ℝ) (W : ℝ) (h1 : (A + B) * 4 = W) (h2 : A * 8 = W) : (A + B) * 4 = W :=
by
  -- placeholder for actual proof
  sorry

end work_rate_sum_l272_272709


namespace part_I_part_II_l272_272607

noncomputable def f (x a b : ℝ) : ℝ := Real.exp x - a * x^2 - 2 * x + b

theorem part_I (a : ℝ) (h : a > 0) :
  ∃ x : ℝ, (Real.exp x - 2 * a * x - 2) < 0 :=
by sorry

theorem part_II (a : ℝ) (h1 : a > 0) (h2 : ∀ x : ℝ, f x a b > 0) :
  ∃ (b_min : ℤ), (b_min = 0) ∧ ∀ b' : ℤ, b' ≥ b_min → ∀ x : ℝ, f x a b' > 0 :=
by sorry

end part_I_part_II_l272_272607


namespace cos2_plus_sin2_given_tan_l272_272613

noncomputable def problem_cos2_plus_sin2_given_tan : Prop :=
  ∀ (α : ℝ), Real.tan α = 2 → Real.cos α ^ 2 + Real.sin (2 * α) = 1

-- Proof is omitted
theorem cos2_plus_sin2_given_tan : problem_cos2_plus_sin2_given_tan := sorry

end cos2_plus_sin2_given_tan_l272_272613


namespace hcf_of_given_numbers_l272_272849

def hcf (x y : ℕ) : ℕ := Nat.gcd x y

theorem hcf_of_given_numbers :
  ∃ (A B : ℕ), A = 33 ∧ A * B = 363 ∧ hcf A B = 11 := 
by
  sorry

end hcf_of_given_numbers_l272_272849


namespace focus_of_parabola_l272_272396

theorem focus_of_parabola (x y : ℝ) (h : y^2 + 4 * x = 0) : (x, y) = (-1, 0) := sorry

end focus_of_parabola_l272_272396


namespace sum_of_two_integers_l272_272400

theorem sum_of_two_integers (a b : ℕ) (h₁ : a * b + a + b = 135) (h₂ : Nat.gcd a b = 1) (h₃ : a < 30) (h₄ : b < 30) : a + b = 23 :=
sorry

end sum_of_two_integers_l272_272400


namespace largest_n_for_sine_cosine_inequality_l272_272907

theorem largest_n_for_sine_cosine_inequality :
  ∃ n : ℕ, (∀ x : ℝ, (Real.sin x) ^ n + (Real.cos x) ^ n ≥ 2 / n) ∧ ∀ m > n, ∃ x : ℝ, (Real.sin x) ^ m + (Real.cos x) ^ m < 2 / m :=
begin
  use 4,
  split,
  { intro x,
    have h1 : (Real.sin x) ^ 4 + (Real.cos x) ^ 4 ≥ 1 / 2,
    { -- Proof using QM-AM and other inequalities
      sorry
    },
    linarith,
  },
  { intros m hm,
    -- Proof that for m > 4 the inequality fails
    sorry
  }
end

end largest_n_for_sine_cosine_inequality_l272_272907


namespace count_odd_numbers_300_600_l272_272621

theorem count_odd_numbers_300_600 : ∃ n : ℕ, n = 149 ∧ ∀ k : ℕ, (301 ≤ k ∧ k < 600 ∧ k % 2 = 1) ↔ (301 ≤ k ∧ k < 600 ∧ k % 2 = 1 ∧ k - 301 < n * 2) :=
by {
  sorry
}

end count_odd_numbers_300_600_l272_272621


namespace dates_relation_l272_272056

def melanie_data_set : set ℕ :=
  { x | (x >= 1 ∧ x <= 28) ∨ (x = 29 ∧ x <= 29) ∨ (x = 30 ∧ x <= 30) ∨ (x = 31 ∧ x <= 31)}

noncomputable def median (s : set ℕ) : ℝ := sorry -- Median calculation
noncomputable def mean (s : set ℕ) : ℝ := sorry -- Mean calculation
noncomputable def modes_median (s : set ℕ) : ℝ := sorry -- Median of modes calculation

theorem dates_relation : 
  let s := melanie_data_set in
  d < mean s < median s :=
sorry

end dates_relation_l272_272056


namespace range_of_a_l272_272183

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, ¬ (ax^2 - ax + 1 = 0)) ↔ (0 ≤ a ∧ a < 4) :=
sorry

end range_of_a_l272_272183


namespace directrix_of_given_parabola_l272_272145

noncomputable def parabola_directrix (y : ℝ) : Prop :=
  let focus_x : ℝ := -1
  let point_on_parabola : ℝ × ℝ := (-1 / 4 * y^2, y)
  let PF_sq := (point_on_parabola.1 - focus_x)^2 + (point_on_parabola.2)^2
  let PQ_sq := (point_on_parabola.1 - 1)^2
  PF_sq = PQ_sq

theorem directrix_of_given_parabola : parabola_directrix y :=
sorry

end directrix_of_given_parabola_l272_272145


namespace total_chips_l272_272690

-- Definitions of the given conditions
def Viviana_chocolate_chips (Susana_chocolate_chips : ℕ) := Susana_chocolate_chips + 5
def Susana_vanilla_chips (Viviana_vanilla_chips : ℕ) := 3 / 4 * Viviana_vanilla_chips
def Viviana_vanilla_chips := 20
def Susana_chocolate_chips := 25

-- The statement to prove the total number of chips
theorem total_chips :
  let Viviana_choco := Viviana_chocolate_chips Susana_chocolate_chips,
      Susana_vani := Susana_vanilla_chips Viviana_vanilla_chips,
      total := Viviana_choco + Viviana_vanilla_chips + Susana_chocolate_chips + Susana_vani
  in total = 90 :=
by
  sorry

end total_chips_l272_272690


namespace min_value_expression_l272_272610

theorem min_value_expression (a : ℝ) (h : a > 2) : a + 4 / (a - 2) ≥ 6 :=
by
  sorry

end min_value_expression_l272_272610


namespace gcd_problem_l272_272590

def a := 47^11 + 1
def b := 47^11 + 47^3 + 1

theorem gcd_problem : Nat.gcd a b = 1 := 
by
  sorry

end gcd_problem_l272_272590


namespace feifei_reaches_school_at_828_l272_272300

-- Definitions for all conditions
def start_time : Nat := 8 * 60 + 10  -- Feifei starts walking at 8:10 AM in minutes since midnight
def dog_delay : Nat := 3             -- Dog starts chasing after 3 minutes
def catch_up_200m_time : ℕ := 1      -- Time for dog to catch Feifei at 200 meters
def catch_up_400m_time : ℕ := 4      -- Time for dog to catch Feifei at 400 meters
def school_distance : ℕ := 800       -- Distance from home to school
def feifei_speed : ℕ := 2            -- assumed speed of Feifei where distance covered uniformly
def dog_speed : ℕ := 6               -- dog speed is three times Feifei's speed
def catch_times := [200, 400, 800]   -- Distances (in meters) where dog catches Feifei

-- Derived condition:
def total_travel_time : ℕ := 
  let time_for_200m := catch_up_200m_time + catch_up_200m_time;
  let time_for_400m_and_back := 2* catch_up_400m_time ;
  (time_for_200m + time_for_400m_and_back + (school_distance - 400))

-- The statement we wish to prove:
theorem feifei_reaches_school_at_828 : 
  (start_time + total_travel_time - dog_delay/2) % 60 = 28 :=
sorry

end feifei_reaches_school_at_828_l272_272300


namespace min_breaks_12_no_breaks_15_l272_272465

-- Define the function to sum the first n natural numbers
def sum_natural (n : ℕ) : ℕ := n * (n + 1) / 2

-- The main theorem for n = 12
theorem min_breaks_12 : ∀ (n = 12), (∑ i in finset.range (n + 1), i % 4 ≠ 0) → 2 := 
by sorry

-- The main theorem for n = 15
theorem no_breaks_15 : ∀ (n = 15), (∑ i in finset.range (n + 1), i % 4 = 0) → 0 := 
by sorry

end min_breaks_12_no_breaks_15_l272_272465


namespace initial_friends_online_l272_272406

theorem initial_friends_online (F : ℕ) 
  (h1 : 8 + F = 13) 
  (h2 : 6 * F = 30) : 
  F = 5 :=
by
  sorry

end initial_friends_online_l272_272406


namespace sin_cos_half_angle_sum_l272_272328

theorem sin_cos_half_angle_sum 
  (θ : ℝ)
  (hcos : Real.cos θ = -7/25) 
  (hθ : θ ∈ Set.Ioo (-Real.pi) 0) : 
  Real.sin (θ/2) + Real.cos (θ/2) = -1/5 := 
sorry

end sin_cos_half_angle_sum_l272_272328


namespace fifteenth_term_is_correct_l272_272742

-- Define the initial conditions of the arithmetic sequence
def firstTerm : ℕ := 4
def secondTerm : ℕ := 9

-- Calculate the common difference
def commonDifference : ℕ := secondTerm - firstTerm

-- Define the nth term formula of the arithmetic sequence
def nthTerm (a d n : ℕ) : ℕ := a + (n - 1) * d

-- The main statement: proving that the 15th term of the given sequence is 74
theorem fifteenth_term_is_correct : nthTerm firstTerm commonDifference 15 = 74 :=
by
  sorry

end fifteenth_term_is_correct_l272_272742


namespace x_squared_minus_y_squared_l272_272176

theorem x_squared_minus_y_squared (x y : ℚ) 
  (h1 : x + y = 9 / 13) (h2 : x - y = 5 / 13) : x^2 - y^2 = 45 / 169 := 
by 
  -- proof omitted 
  sorry

end x_squared_minus_y_squared_l272_272176


namespace complex_division_l272_272261

theorem complex_division :
  (⟨5, -1⟩ : ℂ) / (⟨1, -1⟩ : ℂ) = (⟨3, 2⟩ : ℂ) :=
sorry

end complex_division_l272_272261


namespace problem_solution_l272_272018

theorem problem_solution (x : ℝ) (N : ℝ) (h1 : 625 ^ (-x) + N ^ (-2 * x) + 5 ^ (-4 * x) = 11) (h2 : x = 0.25) :
  N = 25 / 2809 :=
by
  sorry

end problem_solution_l272_272018


namespace find_principal_l272_272869

-- Conditions as definitions
def amount : ℝ := 1120
def rate : ℝ := 0.05
def time : ℝ := 2

-- Required to add noncomputable due to the use of division and real numbers
noncomputable def principal : ℝ := amount / (1 + rate * time)

-- The main theorem statement which needs to be proved
theorem find_principal :
  principal = 1018.18 :=
sorry  -- Proof is not required; it is left as sorry

end find_principal_l272_272869


namespace integer_cube_less_than_triple_unique_l272_272701

theorem integer_cube_less_than_triple_unique (x : ℤ) (h : x^3 < 3*x) : x = 1 :=
sorry

end integer_cube_less_than_triple_unique_l272_272701


namespace inequality_lemma_l272_272938

theorem inequality_lemma (a b c d : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d) (h5 : a * b * c * d = 1) :
  (1 / (b * c + c * d + d * a - 1)) +
  (1 / (a * b + c * d + d * a - 1)) +
  (1 / (a * b + b * c + d * a - 1)) +
  (1 / (a * b + b * c + c * d - 1)) ≤ 2 :=
sorry

end inequality_lemma_l272_272938


namespace determine_a_range_l272_272533

noncomputable def single_element_intersection (a : ℝ) : Prop :=
  let A := {p : ℝ × ℝ | ∃ x : ℝ, p = (x, a * x + 1)}
  let B := {p : ℝ × ℝ | ∃ x : ℝ, p = (x, |x|)}
  (∃ p : ℝ × ℝ, p ∈ A ∧ p ∈ B) ∧ 
  ∀ p₁ p₂ : ℝ × ℝ, p₁ ∈ A ∧ p₁ ∈ B → p₂ ∈ A ∧ p₂ ∈ B → p₁ = p₂

theorem determine_a_range : 
  ∀ a : ℝ, single_element_intersection a ↔ a ∈ Set.Iic (-1) ∨ a ∈ Set.Ici 1 :=
sorry

end determine_a_range_l272_272533


namespace problem_l272_272175

theorem problem (a : ℝ) (h : a^2 - 2 * a - 1 = 0) : -3 * a^2 + 6 * a + 5 = 2 := by
  sorry

end problem_l272_272175


namespace product_of_possible_values_of_x_l272_272645

theorem product_of_possible_values_of_x :
  (∃ x, |x - 7| - 3 = -2) → ∃ y z, |y - 7| - 3 = -2 ∧ |z - 7| - 3 = -2 ∧ y * z = 48 :=
by
  sorry

end product_of_possible_values_of_x_l272_272645


namespace proportion_solution_l272_272627

theorem proportion_solution (x : ℝ) (h : x / 6 = 4 / 0.39999999999999997) : x = 60 := sorry

end proportion_solution_l272_272627


namespace point_C_coordinates_line_MN_equation_area_triangle_ABC_l272_272349

-- Define the points A and B
def A : ℝ × ℝ := (5, -2)
def B : ℝ × ℝ := (7, 3)

-- Let C be an unknown point that we need to determine
variables (x y : ℝ)

-- Define the conditions given in the problem
axiom midpoint_M : (x + 5) / 2 = 0 ∧ (y + 3) / 2 = 0 -- Midpoint M lies on the y-axis
axiom midpoint_N : (x + 7) / 2 = 1 ∧ (y + 3) / 2 = 0 -- Midpoint N lies on the x-axis

-- The problem consists of proving three assertions
theorem point_C_coordinates :
  ∃ (x y : ℝ), (x, y) = (-5, -3) :=
by
  sorry

theorem line_MN_equation :
  ∃ (a b c : ℝ), a = 5 ∧ b = -2 ∧ c = -5 :=
by
  sorry

theorem area_triangle_ABC :
  ∃ (S : ℝ), S = 841 / 20 :=
by
  sorry

end point_C_coordinates_line_MN_equation_area_triangle_ABC_l272_272349


namespace find_x_coord_of_N_l272_272473

theorem find_x_coord_of_N
  (M N : ℝ × ℝ)
  (hM : M = (3, -5))
  (hN : N = (x, 2))
  (parallel : M.1 = N.1) :
  x = 3 :=
sorry

end find_x_coord_of_N_l272_272473


namespace second_alloy_amount_l272_272188

theorem second_alloy_amount (x : ℝ) : 
  (0.10 * 15 + 0.08 * x = 0.086 * (15 + x)) → 
  x = 35 := by 
sorry

end second_alloy_amount_l272_272188


namespace directrix_eqn_of_parabola_l272_272149

theorem directrix_eqn_of_parabola : 
  ∀ y : ℝ, x = - (1 / 4 : ℝ) * y ^ 2 → x = 1 :=
by
  sorry

end directrix_eqn_of_parabola_l272_272149


namespace parabola_directrix_is_x_eq_1_l272_272140

noncomputable def parabola_directrix (y : ℝ) : ℝ :=
  -1 / 4 * y^2

theorem parabola_directrix_is_x_eq_1 :
  ∀ x y, x = parabola_directrix y → x = 1 :=
by
  sorry

end parabola_directrix_is_x_eq_1_l272_272140


namespace grace_charges_for_pulling_weeds_l272_272493

theorem grace_charges_for_pulling_weeds :
  (∃ (W : ℕ ), 63 * 6 + 9 * W + 10 * 9 = 567 → W = 11) :=
by
  use 11
  intro h
  sorry

end grace_charges_for_pulling_weeds_l272_272493


namespace population_multiple_of_18_l272_272399

theorem population_multiple_of_18
  (a b c P : ℕ)
  (ha : P = a^2)
  (hb : P + 200 = b^2 + 1)
  (hc : b^2 + 301 = c^2) :
  ∃ k, P = 18 * k := 
sorry

end population_multiple_of_18_l272_272399


namespace count_irreducible_fractions_l272_272343

theorem count_irreducible_fractions (a b : ℕ) :
  let num := 2015
  let lower_bound := 2015 * 2016
  let upper_bound := 2015 ^ 2 
  (∀ (d : ℕ), lower_bound < d ∧ d ≤ upper_bound ∧ Int.gcd num d = 1) → 
  b = 1440 :=
by
  sorry

end count_irreducible_fractions_l272_272343


namespace water_bottle_capacity_l272_272435

theorem water_bottle_capacity :
  (20 * 250 + 13 * 600) / 1000 = 12.8 := 
by
  sorry

end water_bottle_capacity_l272_272435


namespace focus_of_parabola_l272_272395

theorem focus_of_parabola (x y : ℝ) (h : y^2 + 4 * x = 0) : (x, y) = (-1, 0) := sorry

end focus_of_parabola_l272_272395


namespace periodic_sequences_zero_at_two_l272_272330

variable {R : Type*} [AddGroup R]

def seq_a (a b : ℕ → R) (n : ℕ) : Prop := a (n + 1) = a n + b n
def seq_b (b c : ℕ → R) (n : ℕ) : Prop := b (n + 1) = b n + c n
def seq_c (c d : ℕ → R) (n : ℕ) : Prop := c (n + 1) = c n + d n
def seq_d (d a : ℕ → R) (n : ℕ) : Prop := d (n + 1) = d n + a n

theorem periodic_sequences_zero_at_two
  (a b c d : ℕ → R)
  (k m : ℕ)
  (hk : 1 ≤ k)
  (hm : 1 ≤ m)
  (ha : ∀ n, seq_a a b n)
  (hb : ∀ n, seq_b b c n)
  (hc : ∀ n, seq_c c d n)
  (hd : ∀ n, seq_d d a n)
  (kra : a (k + m) = a m)
  (krb : b (k + m) = b m)
  (krc : c (k + m) = c m)
  (krd : d (k + m) = d m) :
  a 2 = 0 ∧ b 2 = 0 ∧ c 2 = 0 ∧ d 2 = 0 := sorry

end periodic_sequences_zero_at_two_l272_272330


namespace problem1_problem2_l272_272369

noncomputable def f (x a b : ℝ) := |x + a^2| + |x - b^2|

theorem problem1 (a b x : ℝ) (h : a^2 + b^2 - 2 * a + 2 * b + 2 = 0) :
  f x a b >= 3 ↔ x <= -0.5 ∨ x >= 1.5 :=
sorry

theorem problem2 (a b x : ℝ) (h : a + b = 4) :
  f x a b >= 8 :=
sorry

end problem1_problem2_l272_272369


namespace percent_research_and_development_is_9_l272_272103

-- Define given percentages
def percent_transportation := 20
def percent_utilities := 5
def percent_equipment := 4
def percent_supplies := 2

-- Define degree representation and calculate percent for salaries
def degrees_in_circle := 360
def degrees_salaries := 216
def percent_salaries := (degrees_salaries * 100) / degrees_in_circle

-- Define the total percentage representation
def total_percent := 100
def known_percent := percent_transportation + percent_utilities + percent_equipment + percent_supplies + percent_salaries

-- Calculate the percent for research and development
def percent_research_and_development := total_percent - known_percent

-- Theorem statement
theorem percent_research_and_development_is_9 : percent_research_and_development = 9 :=
by 
  -- Placeholder for actual proof
  sorry

end percent_research_and_development_is_9_l272_272103


namespace maximum_height_l272_272292

noncomputable def h (t : ℝ) : ℝ :=
  -20 * t ^ 2 + 100 * t + 30

theorem maximum_height : 
  ∃ t : ℝ, h t = 155 ∧ ∀ t' : ℝ, h t' ≤ 155 := 
sorry

end maximum_height_l272_272292


namespace arrange_numbers_l272_272327

variable {a : ℝ}

theorem arrange_numbers (h1 : -1 < a) (h2 : a < 0) : (1 / a < a) ∧ (a < a ^ 2) ∧ (a ^ 2 < |a|) :=
by 
  sorry

end arrange_numbers_l272_272327


namespace tangent_line_at_slope_two_l272_272080

noncomputable def curve (x : ℝ) : ℝ := Real.log x + x + 1

theorem tangent_line_at_slope_two :
  ∃ (x₀ y₀ : ℝ), (deriv curve x₀ = 2) ∧ (curve x₀ = y₀) ∧ (∀ x, (2 * (x - x₀) + y₀) = (2 * x)) :=
by 
  sorry

end tangent_line_at_slope_two_l272_272080


namespace real_part_0_or_3_complex_part_not_0_or_3_purely_imaginary_at_2_no_second_quadrant_l272_272603

def z (m : ℝ) : ℂ := (m^2 - 5 * m + 6 : ℝ) + (m^2 - 3 * m : ℝ) * Complex.I

theorem real_part_0_or_3 (m : ℝ) : (m^2 - 3 * m = 0) ↔ (m = 0 ∨ m = 3) := sorry

theorem complex_part_not_0_or_3 (m : ℝ) : (m^2 - 3 * m ≠ 0) ↔ (m ≠ 0 ∧ m ≠ 3) := sorry

theorem purely_imaginary_at_2 (m : ℝ) : (m^2 - 5 * m + 6 = 0) ∧ (m^2 - 3 * m ≠ 0) ↔ (m = 2) := sorry

theorem no_second_quadrant (m : ℝ) : ¬(m^2 - 5 * m + 6 < 0 ∧ m^2 - 3 * m > 0) := sorry

end real_part_0_or_3_complex_part_not_0_or_3_purely_imaginary_at_2_no_second_quadrant_l272_272603


namespace determineFinalCounts_l272_272666

structure FruitCounts where
  plums : ℕ
  oranges : ℕ
  apples : ℕ
  pears : ℕ
  cherries : ℕ

def initialCounts : FruitCounts :=
  { plums := 10, oranges := 8, apples := 12, pears := 6, cherries := 0 }

def givenAway : FruitCounts :=
  { plums := 4, oranges := 3, apples := 5, pears := 0, cherries := 0 }

def receivedFromSam : FruitCounts :=
  { plums := 2, oranges := 0, apples := 0, pears := 1, cherries := 0 }

def receivedFromBrother : FruitCounts :=
  { plums := 0, oranges := 1, apples := 2, pears := 0, cherries := 0 }

def receivedFromNeighbor : FruitCounts :=
  { plums := 0, oranges := 0, apples := 0, pears := 3, cherries := 2 }

def finalCounts (initial given receivedSam receivedBrother receivedNeighbor : FruitCounts) : FruitCounts :=
  { plums := initial.plums - given.plums + receivedSam.plums,
    oranges := initial.oranges - given.oranges + receivedBrother.oranges,
    apples := initial.apples - given.apples + receivedBrother.apples,
    pears := initial.pears - given.pears + receivedSam.pears + receivedNeighbor.pears,
    cherries := initial.cherries - given.cherries + receivedNeighbor.cherries }

theorem determineFinalCounts :
  finalCounts initialCounts givenAway receivedFromSam receivedFromBrother receivedFromNeighbor =
  { plums := 8, oranges := 6, apples := 9, pears := 10, cherries := 2 } :=
by
  sorry

end determineFinalCounts_l272_272666


namespace inscribed_squares_equilateral_triangle_l272_272904

theorem inscribed_squares_equilateral_triangle (a b c h_a h_b h_c : ℝ) 
  (h1 : a * h_a / (a + h_a) = b * h_b / (b + h_b))
  (h2 : b * h_b / (b + h_b) = c * h_c / (c + h_c)) :
  a = b ∧ b = c ∧ h_a = h_b ∧ h_b = h_c :=
sorry

end inscribed_squares_equilateral_triangle_l272_272904


namespace cone_radius_l272_272075

noncomputable def radius_of_cone (V : ℝ) (h : ℝ) : ℝ := 
  3 / Real.sqrt (Real.pi)

theorem cone_radius :
  ∀ (V h : ℝ), V = 12 → h = 4 → radius_of_cone V h = 3 / Real.sqrt (Real.pi) :=
by
  intros V h hV hv
  sorry

end cone_radius_l272_272075


namespace inequality_problem_l272_272612

variables {a b c d : ℝ}

theorem inequality_problem (h₁ : a > 0) (h₂ : b > 0) (h₃ : c > 0) (h₄ : d > 0) :
  (a^3 + b^3 + c^3) / (a + b + c) + (b^3 + c^3 + d^3) / (b + c + d) +
  (c^3 + d^3 + a^3) / (c + d + a) + (d^3 + a^3 + b^3) / (d + a + b) ≥ a^2 + b^2 + c^2 + d^2 := 
by
  sorry

end inequality_problem_l272_272612


namespace find_value_of_d_l272_272921

theorem find_value_of_d
  (a b c d : ℕ) 
  (h1 : 0 < a) 
  (h2 : a < b) 
  (h3 : b < c) 
  (h4 : c < d) 
  (h5 : ab + bc + ac = abc) 
  (h6 : abc = d) : 
  d = 36 := 
sorry

end find_value_of_d_l272_272921


namespace value_of_f_neg_one_l272_272941

noncomputable def f : ℝ → ℝ := sorry

theorem value_of_f_neg_one (f_def : ∀ x, f (Real.tan x) = Real.sin (2 * x)) : f (-1) = -1 := 
by
sorry

end value_of_f_neg_one_l272_272941


namespace four_digit_div_90_count_l272_272932

theorem four_digit_div_90_count :
  ∃ n : ℕ, n = 10 ∧ (∀ (ab : ℕ), 10 ≤ ab ∧ ab ≤ 99 → ab % 9 = 0 → 
  (10 * ab + 90) % 90 = 0 ∧ 1000 ≤ 10 * ab + 90 ∧ 10 * ab + 90 < 10000) :=
sorry

end four_digit_div_90_count_l272_272932


namespace solution_set_inequality1_solution_set_inequality2_l272_272707

def inequality1 (x : ℝ) : Prop := (2 * x + 1) / (3 - x) ≥ 0
def inequality2 (x : ℝ) : Prop := (2 * x + 1) / (x - 3) ≤ 0

theorem solution_set_inequality1 : {x : ℝ | (-1 / 2 : ℝ) <= x ∧ x < 3} = {x : ℝ | inequality1 x} :=
sorry

theorem solution_set_inequality2 : {x : ℝ | (-1 / 2 : ℝ) <= x ∧ x < 3} = {x : ℝ | inequality2 x} :=
sorry

end solution_set_inequality1_solution_set_inequality2_l272_272707


namespace tangent_line_equation_l272_272081

noncomputable def curve (x : ℝ) : ℝ :=
  Real.log x + x + 1

theorem tangent_line_equation : ∃ x y : ℝ, derivative curve x = 2 ∧ curve x = y ∧ y = 2 * x := 
begin
  sorry
end

end tangent_line_equation_l272_272081


namespace directrix_of_parabola_l272_272130

-- Define the problem conditions
def parabola (x y : ℝ) : Prop :=
  x = -1/4 * y^2

-- State the theorem to prove that the directrix of the given parabola is x = 1
theorem directrix_of_parabola :
  (∀ (x y : ℝ), parabola x y → true) → 
  let d := 1 in
  ∀ (f : ℝ), (f = -d ∧ f^2 = d^2 ∧ d - f = 2) → true :=
by
  sorry

end directrix_of_parabola_l272_272130


namespace sum_algebra_values_l272_272850

def alphabet_value (n : ℕ) : ℤ :=
  match n % 8 with
  | 1 => 3
  | 2 => 1
  | 3 => 0
  | 4 => -1
  | 5 => -3
  | 6 => -1
  | 7 => 0
  | _ => 1

theorem sum_algebra_values : 
  alphabet_value 1 + 
  alphabet_value 12 + 
  alphabet_value 7 +
  alphabet_value 5 +
  alphabet_value 2 +
  alphabet_value 18 +
  alphabet_value 1 
  = 5 := by
  sorry

end sum_algebra_values_l272_272850


namespace diving_competition_scores_l272_272351

theorem diving_competition_scores (A B C D E : ℝ) (hA : 1 ≤ A ∧ A ≤ 10)
  (hB : 1 ≤ B ∧ B ≤ 10) (hC : 1 ≤ C ∧ C ≤ 10) (hD : 1 ≤ D ∧ D ≤ 10) 
  (hE : 1 ≤ E ∧ E ≤ 10) (degree_of_difficulty : ℝ) (h_diff : degree_of_difficulty = 3.2)
  (point_value : ℝ) (h_point_value : point_value = 79.36) :
  A = max A (max B (max C (max D E))) →
  E = min A (min B (min C (min D E))) →
  (B + C + D) = (point_value / degree_of_difficulty) :=
by sorry

end diving_competition_scores_l272_272351


namespace battery_usage_minutes_l272_272986

theorem battery_usage_minutes (initial_battery final_battery : ℝ) (initial_minutes : ℝ) (rate_of_usage : ℝ) :
  initial_battery - final_battery = rate_of_usage * initial_minutes →
  initial_battery = 100 →
  final_battery = 68 →
  initial_minutes = 60 →
  rate_of_usage = 8 / 15 →
  ∃ additional_minutes : ℝ, additional_minutes = 127.5 :=
by
  intros
  sorry

end battery_usage_minutes_l272_272986


namespace solution_correct_l272_272153

noncomputable def quadratic_inequality_solution (x : ℝ) : Prop :=
  x^2 - 36 * x + 320 ≤ 16

theorem solution_correct (x : ℝ) : quadratic_inequality_solution x ↔ 16 ≤ x ∧ x ≤ 19 :=
by sorry

end solution_correct_l272_272153


namespace distinct_meals_l272_272895

-- Define the conditions
def number_of_entrees : ℕ := 4
def number_of_drinks : ℕ := 3
def number_of_desserts : ℕ := 2

-- Define the main theorem
theorem distinct_meals : number_of_entrees * number_of_drinks * number_of_desserts = 24 := 
by
  -- sorry is used to skip the proof
  sorry

end distinct_meals_l272_272895


namespace parabola_directrix_l272_272136

theorem parabola_directrix : 
  ∃ d : ℝ, (∀ (y : ℝ), x = - (1 / 4) * y^2 -> 
  ( - (1 / 4) * y^2 = d)) ∧ d = 1 :=
by
  sorry

end parabola_directrix_l272_272136


namespace modulus_of_complex_number_l272_272123

theorem modulus_of_complex_number : abs (3 - 4 * complex.i) = 5 := 
by
  sorry

end modulus_of_complex_number_l272_272123


namespace electric_fan_wattage_l272_272970

theorem electric_fan_wattage (hours_per_day : ℕ) (energy_per_month : ℝ) (days_per_month : ℕ) 
  (h1 : hours_per_day = 8) (h2 : energy_per_month = 18) (h3 : days_per_month = 30) : 
  (energy_per_month * 1000) / (days_per_month * hours_per_day) = 75 := 
by { 
  -- Placeholder for the proof
  sorry 
}

end electric_fan_wattage_l272_272970


namespace problem_statement_l272_272940

theorem problem_statement (a b : ℝ) (h : a > b) : -2 * a < -2 * b := by
  sorry

end problem_statement_l272_272940


namespace mascot_sales_growth_rate_equation_l272_272541

-- Define the conditions
def march_sales : ℝ := 100000
def may_sales : ℝ := 115000
def growth_rate (x : ℝ) : Prop := x > 0

-- Define the equation to be proven
theorem mascot_sales_growth_rate_equation (x : ℝ) (h : growth_rate x) :
    10 * (1 + x) ^ 2 = 11.5 :=
sorry

end mascot_sales_growth_rate_equation_l272_272541


namespace largest_positive_integer_n_l272_272906

theorem largest_positive_integer_n :
  ∃ n : ℕ, (∀ x : ℝ, (Real.sin x) ^ n + (Real.cos x) ^ n ≥ 2 / n) ∧ (∀ m : ℕ, m > n → ∃ x : ℝ, (Real.sin x) ^ m + (Real.cos x) ^ m < 2 / m) :=
sorry

end largest_positive_integer_n_l272_272906


namespace volunteers_correct_l272_272267

-- Definitions of given conditions and the required result
def sheets_per_member : ℕ := 10
def cookies_per_sheet : ℕ := 16
def total_cookies : ℕ := 16000

-- Number of members who volunteered
def members : ℕ := total_cookies / (sheets_per_member * cookies_per_sheet)

-- Proof statement
theorem volunteers_correct :
  members = 100 :=
sorry

end volunteers_correct_l272_272267


namespace ratio_3_2_l272_272358

theorem ratio_3_2 (m n : ℕ) (h1 : m + n = 300) (h2 : m > 100) (h3 : n > 100) : m / n = 3 / 2 := by
  sorry

end ratio_3_2_l272_272358


namespace directrix_of_parabola_l272_272131

-- Define the problem conditions
def parabola (x y : ℝ) : Prop :=
  x = -1/4 * y^2

-- State the theorem to prove that the directrix of the given parabola is x = 1
theorem directrix_of_parabola :
  (∀ (x y : ℝ), parabola x y → true) → 
  let d := 1 in
  ∀ (f : ℝ), (f = -d ∧ f^2 = d^2 ∧ d - f = 2) → true :=
by
  sorry

end directrix_of_parabola_l272_272131


namespace calculate_total_houses_built_l272_272740

theorem calculate_total_houses_built :
  let initial_houses := 1426
  let final_houses := 2000
  let rate_a := 25
  let time_a := 6
  let rate_b := 15
  let time_b := 9
  let rate_c := 30
  let time_c := 4
  let total_houses_built := (rate_a * time_a) + (rate_b * time_b) + (rate_c * time_c)
  total_houses_built = 405 :=
by
  sorry

end calculate_total_houses_built_l272_272740


namespace unicorn_tether_l272_272295

theorem unicorn_tether (a b c : ℕ) (h_c_prime : Prime c) :
  (∃ (a b c : ℕ), c = 1 ∧ (25 - 15 = 10 ∧ 10^2 + 10^2 = 15^2 ∧ 
  a = 10 ∧ b = 125) ∧ a + b + c = 136) :=
  sorry

end unicorn_tether_l272_272295


namespace inequality_proof_l272_272609

variable (a b : ℝ)

theorem inequality_proof (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : a + b = 1) :
  (1 / a) + (1 / b) + (1 / (a * b)) ≥ 8 :=
  by
    sorry

end inequality_proof_l272_272609


namespace find_values_of_pqr_l272_272604

def A (p : ℝ) := {x : ℝ | x^2 + p * x - 2 = 0}
def B (q r : ℝ) := {x : ℝ | x^2 + q * x + r = 0}
def A_union_B (p q r : ℝ) := A p ∪ B q r = {-2, 1, 5}
def A_intersect_B (p q r : ℝ) := A p ∩ B q r = {-2}

theorem find_values_of_pqr (p q r : ℝ) :
  A_union_B p q r → A_intersect_B p q r → p = -1 ∧ q = -3 ∧ r = -10 :=
by
  sorry

end find_values_of_pqr_l272_272604


namespace problem_statement_l272_272345

theorem problem_statement (x y : ℕ) (hx : x = 3) (hy : y = 4) :
  (x^3 + 3 * y^2) / 7 = 75 / 7 :=
by 
  -- proof goes here
  sorry

end problem_statement_l272_272345


namespace max_intersections_l272_272125

-- Define the conditions
def num_points_x : ℕ := 15
def num_points_y : ℕ := 10

-- Define the binomial coefficient function
def binom (n k : ℕ) : ℕ :=
  Nat.choose n k

-- Define the problem statement
theorem max_intersections (I : ℕ) :
  (15 : ℕ) == num_points_x →
  (10 : ℕ) == num_points_y →
  (I = binom 15 2 * binom 10 2) →
  I = 4725 := by
  -- We add sorry to skip the proof
  sorry

end max_intersections_l272_272125


namespace cone_height_l272_272480

theorem cone_height (slant_height r : ℝ) (lateral_area : ℝ) (h : ℝ) 
  (h_slant : slant_height = 13) 
  (h_area : lateral_area = 65 * Real.pi) 
  (h_lateral_area : lateral_area = Real.pi * r * slant_height) 
  (h_radius : r = 5) :
  h = Real.sqrt (slant_height ^ 2 - r ^ 2) :=
by
  -- Definitions and conditions
  have h_slant_height : slant_height = 13 := h_slant
  have h_lateral_area_value : lateral_area = 65 * Real.pi := h_area
  have h_lateral_surface_area : lateral_area = Real.pi * r * slant_height := h_lateral_area
  have h_radius_5 : r = 5 := h_radius
  sorry -- Proof is omitted

end cone_height_l272_272480


namespace algebraic_expression_value_l272_272344

theorem algebraic_expression_value (m n : ℤ) (h : n - m = 2):
  (m^2 - n^2) / m * (2 * m / (m + n)) = -4 :=
sorry

end algebraic_expression_value_l272_272344


namespace intersection_of_M_and_N_l272_272783

def M : Set ℤ := {0, 1}
def N : Set ℤ := {-1, 0}

theorem intersection_of_M_and_N : M ∩ N = {0} := by
  sorry

end intersection_of_M_and_N_l272_272783


namespace rectangular_coords_transformation_l272_272583

noncomputable def sphericalToRectangular (ρ θ φ : ℝ) : ℝ × ℝ × ℝ :=
(ρ * Real.sin φ * Real.cos θ, ρ * Real.sin φ * Real.sin θ, ρ * Real.cos φ)

theorem rectangular_coords_transformation :
  let ρ := Real.sqrt (2 ^ 2 + (-3) ^ 2 + 6 ^ 2)
  let φ := Real.arccos (6 / ρ)
  let θ := Real.arctan (-3 / 2)
  sphericalToRectangular ρ (Real.pi + θ) φ = (-2, 3, 6) :=
by
  sorry

end rectangular_coords_transformation_l272_272583


namespace ratio_final_to_original_l272_272348

-- Given conditions
variable (d : ℝ)
variable (h1 : 364 = d * 1.30)

-- Problem statement
theorem ratio_final_to_original : (364 / d) = 1.3 := 
by sorry

end ratio_final_to_original_l272_272348


namespace max_value_expression_l272_272816

theorem max_value_expression (k : ℕ) (a b c : ℝ) (h : k > 0) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) (habc : a + b + c = 3 * k) :
  a^(3 * k - 1) * b + b^(3 * k - 1) * c + c^(3 * k - 1) * a + k^2 * a^k * b^k * c^k ≤ (3 * k - 1)^(3 * k - 1) :=
sorry

end max_value_expression_l272_272816


namespace triangle_angle_C_right_l272_272961

theorem triangle_angle_C_right {a b c A B C : ℝ}
  (h1 : a / Real.sin B + b / Real.sin A = 2 * c) 
  (h2 : a / Real.sin A = b / Real.sin B) 
  (h3 : b / Real.sin B = c / Real.sin C) : 
  C = Real.pi / 2 :=
by sorry

end triangle_angle_C_right_l272_272961


namespace minimum_cuts_for_10_pieces_l272_272747

theorem minimum_cuts_for_10_pieces :
  ∃ n : ℕ, (n * (n + 1)) / 2 ≥ 10 ∧ ∀ m < n, (m * (m + 1)) / 2 < 10 := sorry

end minimum_cuts_for_10_pieces_l272_272747


namespace min_breaks_for_square_12_can_form_square_15_l272_272459

-- Definitions and conditions for case n = 12
def stick_lengths_12 := (finset.range 12).map (λ i, i + 1)
def total_length_12 := stick_lengths_12.sum

-- Proof problem for n = 12
theorem min_breaks_for_square_12 : 
  ∃ min_breaks : ℕ, total_length_12 + min_breaks * 2 ∈ {k | k % 4 = 0} ∧ min_breaks = 2 :=
sorry

-- Definitions and conditions for case n = 15
def stick_lengths_15 := (finset.range 15).map (λ i, i + 1)
def total_length_15 := stick_lengths_15.sum

-- Proof problem for n = 15
theorem can_form_square_15 : 
  total_length_15 % 4 = 0 :=
sorry

end min_breaks_for_square_12_can_form_square_15_l272_272459


namespace gen_term_seq_l272_272160

open Nat

def seq (a : ℕ → ℕ) : Prop := 
a 1 = 1 ∧ (∀ n : ℕ, n ≠ 0 → a (n + 1) = 2 * a n - 3)

theorem gen_term_seq (a : ℕ → ℕ) (h : seq a) : ∀ n : ℕ, a n = 3 - 2^n :=
by
  sorry

end gen_term_seq_l272_272160


namespace geometric_sequence_log_sum_l272_272032

noncomputable def log_base_three (x : ℝ) : ℝ := Real.log x / Real.log 3

theorem geometric_sequence_log_sum (a : ℕ → ℝ) (h1 : ∀ n, a n > 0)
  (h2 : ∃ r, ∀ n, a (n + 1) = a n * r)
  (h3 : a 6 * a 7 = 9) :
  log_base_three (a 1) + log_base_three (a 2) + log_base_three (a 3) +
  log_base_three (a 4) + log_base_three (a 5) + log_base_three (a 6) +
  log_base_three (a 7) + log_base_three (a 8) + log_base_three (a 9) +
  log_base_three (a 10) + log_base_three (a 11) + log_base_three (a 12) = 12 :=
  sorry

end geometric_sequence_log_sum_l272_272032


namespace shaded_region_perimeter_l272_272805

theorem shaded_region_perimeter (r : ℝ) (θ : ℝ) (h₁ : r = 2) (h₂ : θ = 90) : 
  (2 * r + (2 * π * r * (1 - θ / 180))) = π + 4 := 
by sorry

end shaded_region_perimeter_l272_272805


namespace debby_deletion_l272_272254

theorem debby_deletion :
  ∀ (zoo_pics museum_pics remaining_pics deleted_pics : ℕ),
    zoo_pics = 24 →
    museum_pics = 12 →
    remaining_pics = 22 →
    deleted_pics = zoo_pics + museum_pics - remaining_pics →
    deleted_pics = 14 :=
by
  intros zoo_pics museum_pics remaining_pics deleted_pics h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  exact h4

end debby_deletion_l272_272254


namespace algebraic_identity_neg_exponents_l272_272599

theorem algebraic_identity_neg_exponents (x y z : ℂ) :
  (x + y + z)⁻¹ * (x⁻¹ + y⁻¹ + z⁻¹) = (y * z + x * z + x * y) * x⁻¹ * y⁻¹ * z⁻¹ * (x + y + z)⁻¹ :=
by
  sorry

end algebraic_identity_neg_exponents_l272_272599


namespace prob_high_quality_product_distribution_mean_variance_l272_272636

-- Definitions used in the problem
def QualityIndexDist : ProbabilityDistribution ℝ := normal 64 10 

def HighQualityRange : Set ℝ := {x | 54 ≤ x ∧ x ≤ 84}

def GivenProbabilities : Prop := 
  P(μ - σ ≤ X ∧ X ≤ μ + σ) = 0.6827 ∧
  P(μ - 2σ ≤ X ∧ X ≤ μ + 2σ) = 0.9545 ∧
  P(μ - 3σ ≤ X ∧ X ≤ μ + 3σ) = 0.9973

-- Proof Statements as Lean Theorems
theorem prob_high_quality_product : GivenProbabilities → 
  P(λ x, QualityIndexDist.pdf x, HighQualityRange) = 0.82 :=
sorry

theorem distribution_mean_variance (X : binom 5 0.82) : 
  (X.mean, X.variance) = (4.1, 0.738) :=
sorry


end prob_high_quality_product_distribution_mean_variance_l272_272636


namespace tan_alpha_plus_pi_div_4_l272_272164

theorem tan_alpha_plus_pi_div_4 (α : ℝ) (hcos : Real.cos α = 3 / 5) (h0 : 0 < α) (hpi : α < Real.pi) :
  Real.tan (α + Real.pi / 4) = -7 :=
by
  sorry

end tan_alpha_plus_pi_div_4_l272_272164


namespace value_of_J_l272_272163

-- Given conditions
variables (Y J : ℤ)

-- Condition definitions
axiom condition1 : 150 < Y ∧ Y < 300
axiom condition2 : Y = J^2 * J^3
axiom condition3 : ∃ n : ℤ, Y = n^3

-- Goal: Value of J
theorem value_of_J : J = 3 :=
by { sorry }  -- Proof omitted

end value_of_J_l272_272163


namespace die_roll_divisor_of_12_prob_l272_272719

def fair_die_probability_divisor_of_12 : Prop :=
  let favorable_outcomes := {1, 2, 3, 4, 6}
  let total_outcomes := 6
  let probability := favorable_outcomes.size / total_outcomes
  probability = 5 / 6

theorem die_roll_divisor_of_12_prob:
  fair_die_probability_divisor_of_12 :=
by
  sorry

end die_roll_divisor_of_12_prob_l272_272719


namespace solve_inequalities_l272_272403

theorem solve_inequalities (x : ℝ) : (x + 1 > 0 ∧ x - 3 < 2) ↔ (-1 < x ∧ x < 5) :=
by sorry

end solve_inequalities_l272_272403


namespace problem_divisible_by_900_l272_272227

theorem problem_divisible_by_900 (X : ℕ) (a b c d : ℕ) 
  (h1 : 1000 <= X)
  (h2 : X < 10000)
  (h3 : X = 1000 * a + 100 * b + 10 * c + d)
  (h4 : d ≠ 0)
  (h5 : (X + (1000 * a + 100 * c + 10 * b + d)) % 900 = 0)
  : X % 90 = 45 := 
sorry

end problem_divisible_by_900_l272_272227


namespace unclaimed_candy_fraction_l272_272892

-- Definitions for the shares taken by each person.
def al_share (x : ℕ) : ℚ := 3 / 7 * x
def bert_share (x : ℕ) : ℚ := 2 / 7 * (x - al_share x)
def carl_share (x : ℕ) : ℚ := 1 / 7 * ((x - al_share x) - bert_share x)
def dana_share (x : ℕ) : ℚ := 1 / 7 * (((x - al_share x) - bert_share x) - carl_share x)

-- The amount of candy that goes unclaimed.
def remaining_candy (x : ℕ) : ℚ := x - (al_share x + bert_share x + carl_share x + dana_share x)

-- The theorem we want to prove.
theorem unclaimed_candy_fraction (x : ℕ) : remaining_candy x / x = 584 / 2401 :=
by
  sorry

end unclaimed_candy_fraction_l272_272892


namespace angle_sum_triangle_l272_272963

theorem angle_sum_triangle (A B C : ℝ) 
  (hA : A = 20)
  (hC : C = 90) :
  B = 70 := 
by
  -- In a triangle the sum of angles is 180 degrees
  have h_sum : A + B + C = 180 := sorry
  -- Substitute the given angles A and C
  rw [hA, hC] at h_sum
  -- Simplify the equation to find B
  have hB : 20 + B + 90 = 180 := sorry
  linarith

end angle_sum_triangle_l272_272963


namespace coffee_per_cup_for_weak_l272_272678

-- Defining the conditions
def weak_coffee_cups : ℕ := 12
def strong_coffee_cups : ℕ := 12
def total_coffee_tbsp : ℕ := 36
def weak_increase_factor : ℕ := 1
def strong_increase_factor : ℕ := 2

-- The theorem stating the problem
theorem coffee_per_cup_for_weak :
  ∃ W : ℝ, (weak_coffee_cups * W + strong_coffee_cups * (strong_increase_factor * W) = total_coffee_tbsp) ∧ (W = 1) :=
  sorry

end coffee_per_cup_for_weak_l272_272678


namespace fraction_greater_than_decimal_l272_272124

theorem fraction_greater_than_decimal :
  (1 / 4 : ℝ) > (24999999 / (10^8 : ℝ)) + (1 / (4 * (10^8 : ℝ))) :=
by
  sorry

end fraction_greater_than_decimal_l272_272124


namespace abs_sum_binom_expansion_l272_272335

theorem abs_sum_binom_expansion :
  let a := (1 - 3 * x)^9,
  let sum_absolutes := ∑ k in Finset.range 10, |(Nat.choose 9 k) * (-3)^k|,
  sum_absolutes = 4^9 :=
by
  sorry

end abs_sum_binom_expansion_l272_272335


namespace teacher_age_l272_272574

theorem teacher_age (avg_age_students : ℕ) (num_students : ℕ) (avg_age_with_teacher : ℕ) (num_total : ℕ) 
  (h1 : avg_age_students = 14) (h2 : num_students = 50) (h3 : avg_age_with_teacher = 15) (h4 : num_total = 51) :
  ∃ (teacher_age : ℕ), teacher_age = 65 :=
by sorry

end teacher_age_l272_272574


namespace normal_distribution_prob_eq_l272_272801

open ProbabilityTheory

noncomputable def xi := Normal (-2) 4

theorem normal_distribution_prob_eq :
  ℙ (xi ∈ Ioc (-4) (-2)) = ℙ (xi ∈ Ioc (-2) 0) :=
sorry

end normal_distribution_prob_eq_l272_272801


namespace number_of_teachers_in_school_l272_272187

-- Definitions based on provided conditions
def number_of_girls : ℕ := 315
def number_of_boys : ℕ := 309
def total_number_of_people : ℕ := 1396

-- Proof goal: Number of teachers in the school
theorem number_of_teachers_in_school : 
  total_number_of_people - (number_of_girls + number_of_boys) = 772 :=
by
  sorry

end number_of_teachers_in_school_l272_272187


namespace abc_le_sqrt2_div_4_l272_272005

variable {a b c : ℝ}
variable (ha_pos : 0 < a) (hb_pos : 0 < b) (hc_pos : 0 < c)
variable (h : (a^2 / (1 + a^2)) + (b^2 / (1 + b^2)) + (c^2 / (1 + c^2)) = 1)

theorem abc_le_sqrt2_div_4 (ha_pos : 0 < a) (hb_pos : 0 < b) (hc_pos : 0 < c)
  (h : (a^2 / (1 + a^2)) + (b^2 / (1 + b^2)) + (c^2 / (1 + c^2)) = 1) :
  a * b * c ≤ (Real.sqrt 2) / 4 := 
sorry

end abc_le_sqrt2_div_4_l272_272005


namespace find_d_l272_272949

theorem find_d (y d : ℝ) (hy : y > 0) (h : (8 * y) / 20 + (3 * y) / d = 0.7 * y) : d = 10 :=
by
  sorry

end find_d_l272_272949


namespace cube_as_difference_of_squares_l272_272875

theorem cube_as_difference_of_squares (a : ℕ) : 
  a^3 = (a * (a + 1) / 2)^2 - (a * (a - 1) / 2)^2 := 
by 
  -- The proof portion would go here, but since we only need the statement:
  sorry

end cube_as_difference_of_squares_l272_272875


namespace advantageous_bank_l272_272553

variable (C : ℝ) (p n : ℝ)

noncomputable def semiAnnualCompounding (p : ℝ) (n : ℝ) : ℝ :=
  (1 + p / (2 * 100)) ^ n

noncomputable def monthlyCompounding (p : ℝ) (n : ℝ) : ℝ :=
  (1 + p / (12 * 100)) ^ (6 * n)

theorem advantageous_bank (p n : ℝ) :
  monthlyCompounding p n - semiAnnualCompounding p n > 0 := sorry

#check advantageous_bank

end advantageous_bank_l272_272553


namespace second_crew_tractors_l272_272321

theorem second_crew_tractors
    (total_acres : ℕ)
    (days : ℕ)
    (first_crew_days : ℕ)
    (first_crew_tractors : ℕ)
    (acres_per_tractor_per_day : ℕ)
    (remaining_days : ℕ)
    (remaining_acres_after_first_crew : ℕ)
    (second_crew_acres_per_tractor : ℕ) :
    total_acres = 1700 → days = 5 → first_crew_days = 2 → first_crew_tractors = 2 → 
    acres_per_tractor_per_day = 68 → remaining_days = 3 → 
    remaining_acres_after_first_crew = total_acres - (first_crew_tractors * acres_per_tractor_per_day * first_crew_days) → 
    second_crew_acres_per_tractor = acres_per_tractor_per_day * remaining_days → 
    (remaining_acres_after_first_crew / second_crew_acres_per_tractor = 7) := 
by
  sorry

end second_crew_tractors_l272_272321


namespace triangle_acute_of_angles_sum_gt_90_l272_272703

theorem triangle_acute_of_angles_sum_gt_90 
  (α β γ : ℝ) 
  (h₁ : α + β + γ = 180) 
  (h₂ : α + β > 90) 
  (h₃ : α + γ > 90) 
  (h₄ : β + γ > 90) 
  : α < 90 ∧ β < 90 ∧ γ < 90 :=
sorry

end triangle_acute_of_angles_sum_gt_90_l272_272703


namespace garden_length_l272_272572

theorem garden_length :
  ∀ (w : ℝ) (l : ℝ),
  (l = 2 * w) →
  (2 * l + 2 * w = 150) →
  l = 50 :=
by
  intros w l h1 h2
  sorry

end garden_length_l272_272572


namespace evaluate_expression_l272_272896

theorem evaluate_expression : 6 / (-1 / 2 + 1 / 3) = -36 := 
by
  sorry

end evaluate_expression_l272_272896


namespace parabola_intercepts_sum_l272_272229

noncomputable def y_intercept (f : ℝ → ℝ) : ℝ := f 0

noncomputable def x_intercepts_of_parabola (a b c : ℝ) : (ℝ × ℝ) :=
let Δ := b ^ 2 - 4 * a * c in
(
  (-b + real.sqrt Δ) / (2 * a),
  (-b - real.sqrt Δ) / (2 * a)
)

theorem parabola_intercepts_sum :
  let f := λ x : ℝ, 3 * x^2 - 9 * x + 4 in
  let (e, f) := x_intercepts_of_parabola 3 (-9) 4 in
  y_intercept f + e + f = 19 / 3 :=
by
  sorry

end parabola_intercepts_sum_l272_272229


namespace minimum_sticks_broken_n12_can_form_square_n15_l272_272467

-- Define the total length function
def total_length (n : ℕ) : ℕ := (n * (n + 1)) / 2

-- For n = 12, prove that at least 2 sticks need to be broken to form a square
theorem minimum_sticks_broken_n12 : ∀ (n : ℕ), n = 12 → total_length n % 4 ≠ 0 → 2 = 2 := 
by 
  intros n h1 h2
  sorry

-- For n = 15, prove that a square can be directly formed
theorem can_form_square_n15 : ∀ (n : ℕ), n = 15 → total_length n % 4 = 0 := 
by 
  intros n h1
  sorry

end minimum_sticks_broken_n12_can_form_square_n15_l272_272467


namespace problem_1_problem_2_l272_272172

open Set

variables {U : Type*} [TopologicalSpace U] (a x : ℝ)

def M : Set ℝ := { x | -2 ≤ x ∧ x ≤ 5 }
def N (a : ℝ) : Set ℝ := { x | a + 1 ≤ x ∧ x ≤ 2 * a + 1 }

noncomputable def complement_N (a : ℝ) : Set ℝ := { x | x < a + 1 ∨ 2 * a + 1 < x }

theorem problem_1 (h : a = 2) :
  M ∩ (complement_N a) = { x | -2 ≤ x ∧ x < 3 } :=
sorry

theorem problem_2 (h : M ∪ N a = M) :
  a ≤ 2 :=
sorry

end problem_1_problem_2_l272_272172


namespace proof_m_plus_n_l272_272157

variable (m n : ℚ) -- Defining m and n as rational numbers (ℚ)
-- Conditions from the problem:
axiom condition1 : 2 * m + 5 * n + 8 = 1
axiom condition2 : m - n - 3 = 1

-- Proof statement (theorem) that needs to be established:
theorem proof_m_plus_n : m + n = -2/7 :=
by
-- Since the proof is not required, we use "sorry" to placeholder the proof.
sorry

end proof_m_plus_n_l272_272157


namespace mouse_lives_correct_l272_272581

def cat_lives : ℕ := 9
def dog_lives : ℕ := cat_lives - 3
def mouse_lives : ℕ := dog_lives + 7

theorem mouse_lives_correct : mouse_lives = 13 :=
by
  sorry

end mouse_lives_correct_l272_272581


namespace jack_morning_emails_l272_272966

theorem jack_morning_emails (x : ℕ) (aft_mails eve_mails total_morn_eve : ℕ) (h1: aft_mails = 4) (h2: eve_mails = 8) (h3: total_morn_eve = 11) :
  x = total_morn_eve - eve_mails :=
by 
  sorry

end jack_morning_emails_l272_272966


namespace bernardo_prob_larger_l272_272438

def set_Bernardo := {1, 2, 3, 4, 5, 6, 7, 8, 9}
def set_Silvia := {1, 2, 3, 4, 5, 6, 7, 8, 10}

def selection_probability (set_B: Finset ℕ) (set_S: Finset ℕ) : ℚ :=
  let total_B := (set_B.card).choose 3
  let total_S := (set_S.card).choose 3
  let valid_selections := -- Number of valid ways Bernardo's number is larger
    (finset.powersetLen 3 set_B).sum
        (λ b, (finset.powersetLen 3 set_S).count
          (λ s, descending_array_to_num b > descending_array_to_num s))
  (valid_selections : ℚ) / (total_B * total_S)

theorem bernardo_prob_larger :
  selection_probability set_Bernardo set_Silvia = 83 / 168 :=
begin
  sorry
end

end bernardo_prob_larger_l272_272438


namespace gcd_exponent_min_speed_for_meeting_game_probability_difference_l272_272715

-- Problem p4
theorem gcd_exponent (a b : ℕ) (h1 : a = 6) (h2 : b = 9) (h3 : gcd a b = 3) : gcd (2^a - 1) (2^b - 1) = 7 := by
  sorry

-- Problem p5
theorem min_speed_for_meeting (v_S s : ℚ) (h : v_S = 1/2) : ∀ (s : ℚ), (s - v_S) ≥ 1 → s = 3/2 := by
  sorry

-- Problem p6
theorem game_probability_difference (N : ℕ) (p : ℚ) (h1 : N = 1) (h2 : p = 5/16) : N + p = 21/16 := by
  sorry

end gcd_exponent_min_speed_for_meeting_game_probability_difference_l272_272715


namespace statue_of_liberty_model_height_l272_272587

theorem statue_of_liberty_model_height :
  let scale_ratio : Int := 30
  let actual_height : Int := 305
  round (actual_height / scale_ratio) = 10 := by
  sorry

end statue_of_liberty_model_height_l272_272587


namespace min_value_of_f_l272_272156

def f (x y : ℝ) : ℝ := x^3 + y^3 + x^2 * y + x * y^2 - 3 * (x^2 + y^2 + x * y) + 3 * (x + y)

theorem min_value_of_f : ∀ x y : ℝ, x ≥ 1/2 → y ≥ 1/2 → f x y ≥ 1
    := by
      intros x y hx hy
      -- Rest of the proof would go here
      sorry

end min_value_of_f_l272_272156


namespace decrease_in_combined_area_l272_272929

theorem decrease_in_combined_area (r1 r2 r3 : ℝ) :
    let π := Real.pi
    let A_original := π * (r1 ^ 2) + π * (r2 ^ 2) + π * (r3 ^ 2)
    let r1' := r1 * 0.5
    let r2' := r2 * 0.5
    let r3' := r3 * 0.5
    let A_new := π * (r1' ^ 2) + π * (r2' ^ 2) + π * (r3' ^ 2)
    let Decrease := A_original - A_new
    Decrease = 0.75 * π * (r1 ^ 2) + 0.75 * π * (r2 ^ 2) + 0.75 * π * (r3 ^ 2) :=
by
  sorry

end decrease_in_combined_area_l272_272929


namespace original_number_of_workers_l272_272867

theorem original_number_of_workers (W A : ℕ)
  (h1 : W * 75 = A)
  (h2 : (W + 10) * 65 = A) :
  W = 65 :=
by
  sorry

end original_number_of_workers_l272_272867


namespace trigonometric_problem_l272_272470

theorem trigonometric_problem (θ : ℝ) (h : Real.tan θ = 2) :
  (3 * Real.sin θ - 2 * Real.cos θ) / (Real.sin θ + 3 * Real.cos θ) = 4 / 5 := by
  sorry

end trigonometric_problem_l272_272470


namespace square_possible_n12_square_possible_n15_l272_272460

-- Define the nature of the problem with condition n = 12
def min_sticks_to_break_for_square_n12 : ℕ :=
  let n := 12
  let total_length := (n * (n + 1)) / 2
  if total_length % 4 = 0 then 0 else 2

-- Define the nature of the problem with condition n = 15
def min_sticks_to_break_for_square_n15 : ℕ :=
  let n := 15
  let total_length := (n * (n + 1)) / 2
  if total_length % 4 = 0 then 0 else 2

-- Statement of the problems in Lean 4 language
theorem square_possible_n12 : min_sticks_to_break_for_square_n12 = 2 := by
  sorry

theorem square_possible_n15 : min_sticks_to_break_for_square_n15 = 0 := by
  sorry

end square_possible_n12_square_possible_n15_l272_272460


namespace find_divisor_l272_272418

theorem find_divisor (d : ℕ) : ((23 = (d * 7) + 2) → d = 3) :=
by
  sorry

end find_divisor_l272_272418


namespace bert_money_left_l272_272589

theorem bert_money_left
  (initial_amount : ℝ)
  (spent_hardware_store_fraction : ℝ)
  (amount_spent_dry_cleaners : ℝ)
  (spent_grocery_store_fraction : ℝ)
  (final_amount : ℝ) :
  initial_amount = 44 →
  spent_hardware_store_fraction = 1/4 →
  amount_spent_dry_cleaners = 9 →
  spent_grocery_store_fraction = 1/2 →
  final_amount = initial_amount - (spent_hardware_store_fraction * initial_amount) - amount_spent_dry_cleaners - (spent_grocery_store_fraction * (initial_amount - (spent_hardware_store_fraction * initial_amount) - amount_spent_dry_cleaners)) →
  final_amount = 12 :=
by
  sorry

end bert_money_left_l272_272589


namespace problem_integer_and_decimal_parts_eq_2_l272_272630

theorem problem_integer_and_decimal_parts_eq_2 :
  let x := 3
  let y := 2 - Real.sqrt 3
  2 * x^3 - (y^3 + 1 / y^3) = 2 :=
by
  sorry

end problem_integer_and_decimal_parts_eq_2_l272_272630


namespace Marge_savings_l272_272664

theorem Marge_savings
  (lottery_winnings : ℝ)
  (taxes_paid : ℝ)
  (student_loan_payment : ℝ)
  (amount_after_taxes : ℝ)
  (amount_after_student_loans : ℝ)
  (fun_money : ℝ)
  (investment : ℝ)
  (savings : ℝ)
  (h_win : lottery_winnings = 12006)
  (h_tax : taxes_paid = lottery_winnings / 2)
  (h_after_tax : amount_after_taxes = lottery_winnings - taxes_paid)
  (h_loans : student_loan_payment = amount_after_taxes / 3)
  (h_after_loans : amount_after_student_loans = amount_after_taxes - student_loan_payment)
  (h_fun : fun_money = 2802)
  (h_savings_investment : amount_after_student_loans - fun_money = savings + investment)
  (h_investment : investment = savings / 5)
  (h_left : amount_after_student_loans - fun_money = 1200) :
  savings = 1000 :=
by
  sorry

end Marge_savings_l272_272664


namespace solution_set_of_inequality_l272_272547

theorem solution_set_of_inequality (x : ℝ) (h : x ≠ 2) :
  (1 / (x - 2) > -2) ↔ (x < 3 / 2 ∨ x > 2) :=
by sorry

end solution_set_of_inequality_l272_272547


namespace original_number_l272_272270

theorem original_number (N y x : ℕ) 
  (h1: N + y = 54321)
  (h2: N = 10 * y + x)
  (h3: 11 * y + x = 54321)
  (h4: x = 54321 % 11)
  (hy: y = 4938) : 
  N = 49383 := 
  by 
  sorry

end original_number_l272_272270


namespace solve_linear_function_l272_272780

theorem solve_linear_function :
  (∀ (x y : ℤ), (x = -3 ∧ y = -4) ∨ (x = -2 ∧ y = -2) ∨ (x = -1 ∧ y = 0) ∨ 
                      (x = 0 ∧ y = 2) ∨ (x = 1 ∧ y = 4) ∨ (x = 2 ∧ y = 6) →
   ∃ (a b : ℤ), y = a * x + b ∧ a * 1 + b = 4) :=
sorry

end solve_linear_function_l272_272780


namespace train_lengths_l272_272735

noncomputable def train_problem : Prop :=
  let speed_T1_mps := 54 * (5/18)
  let speed_T2_mps := 72 * (5/18)
  let L_T1 := speed_T1_mps * 20
  let L_p := (speed_T1_mps * 44) - L_T1
  let L_T2 := speed_T2_mps * 16
  (L_p = 360) ∧ (L_T1 = 300) ∧ (L_T2 = 320)

theorem train_lengths : train_problem := sorry

end train_lengths_l272_272735


namespace problem_l272_272947

theorem problem (a : ℝ) :
  (∀ x : ℝ, (x > 1 ↔ (x - 1 > 0 ∧ 2 * x - a > 0))) → a ≤ 2 :=
by
  sorry

end problem_l272_272947


namespace regions_first_two_sets_regions_all_sets_l272_272367

-- Definitions for the problem
def triangle_regions_first_two_sets (n : ℕ) : ℕ :=
  (n + 1) * (n + 1)

def triangle_regions_all_sets (n : ℕ) : ℕ :=
  3 * n * n + 3 * n + 1

-- Proof Problem 1: Given n points on AB and AC, prove the regions are (n + 1)^2
theorem regions_first_two_sets (n : ℕ) :
  (n * (n + 1) + (n + 1)) = (n + 1) * (n + 1) :=
by sorry

-- Proof Problem 2: Given n points on AB, AC, and BC, prove the regions are 3n^2 + 3n + 1
theorem regions_all_sets (n : ℕ) :
  ((n + 1) * (n + 1) + n * (2 * n + 1)) = 3 * n * n + 3 * n + 1 :=
by sorry

end regions_first_two_sets_regions_all_sets_l272_272367


namespace train_speed_l272_272785

theorem train_speed (length_bridge : ℕ) (time_total : ℕ) (time_on_bridge : ℕ) (speed_of_train : ℕ) 
  (h1 : length_bridge = 800)
  (h2 : time_total = 60)
  (h3 : time_on_bridge = 40)
  (h4 : length_bridge + (time_total - time_on_bridge) * speed_of_train = time_total * speed_of_train) :
  speed_of_train = 20 := sorry

end train_speed_l272_272785


namespace problem_y_values_l272_272979

theorem problem_y_values (x : ℝ) (h : x^2 + 9 * (x / (x - 3))^2 = 54) :
  ∃ y : ℝ, (y = (x - 3)^2 * (x + 4) / (3 * x - 4)) ∧ (y = 7.5 ∨ y = 4.5) := by
sorry

end problem_y_values_l272_272979


namespace total_gas_cost_l272_272913

theorem total_gas_cost 
  (x : ℝ)
  (cost_per_person_initial : ℝ := x / 5)
  (cost_per_person_new : ℝ := x / 8)
  (cost_difference : cost_per_person_initial - cost_per_person_new = 15) :
  x = 200 :=
sorry

end total_gas_cost_l272_272913


namespace cookies_total_is_60_l272_272826

def Mona_cookies : ℕ := 20
def Jasmine_cookies : ℕ := Mona_cookies - 5
def Rachel_cookies : ℕ := Jasmine_cookies + 10
def Total_cookies : ℕ := Mona_cookies + Jasmine_cookies + Rachel_cookies

theorem cookies_total_is_60 : Total_cookies = 60 := by
  sorry

end cookies_total_is_60_l272_272826


namespace curve_intersects_itself_l272_272298

theorem curve_intersects_itself :
  ∃ t₁ t₂ : ℝ, t₁ ≠ t₂ ∧ (t₁^2 - 3, t₁^3 - 6 * t₁ + 4) = (3, 4) ∧ (t₂^2 - 3, t₂^3 - 6 * t₂ + 4) = (3, 4) :=
sorry

end curve_intersects_itself_l272_272298


namespace cylinder_ratio_l272_272778

theorem cylinder_ratio
  (V : ℝ) (R H : ℝ)
  (hV : V = 1000)
  (hVolume : π * R^2 * H = V) :
  H / R = 1 :=
by
  sorry

end cylinder_ratio_l272_272778


namespace unique_lcm_condition_l272_272199

theorem unique_lcm_condition (a : Fin 2000 → ℕ) (h_distinct: ∀ i j, i ≠ j → a i ≠ a j)
  (h_ordered: ∀ i j, i ≤ j → a i ≤ a j)
  (h_bound: ∀ i, 1 ≤ a i ∧ a i < 4000)
  (h_lcm: ∀ i j, i ≠ j → Nat.lcm (a i) (a j) ≥ 4000)
  : a 0 ≥ 1334 :=
by
  sorry

end unique_lcm_condition_l272_272199


namespace polynomial_simplification_l272_272151

theorem polynomial_simplification (x : ℝ) : 
  (x * (x * (2 - x) - 4) + 10) + 1 = -x^4 + 2 * x^3 - 4 * x^2 + 10 * x + 1 :=
by
  sorry

end polynomial_simplification_l272_272151


namespace quadratic_has_two_distinct_real_roots_l272_272946

theorem quadratic_has_two_distinct_real_roots (m : ℝ) : 
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x1^2 - 2 * x1 + m = 0) ∧ (x2^2 - 2 * x2 + m = 0)) ↔ (m < 1) :=
by
  sorry

end quadratic_has_two_distinct_real_roots_l272_272946


namespace find_original_number_l272_272283

open Int

theorem find_original_number (N y x : ℕ) (h1 : N = 10 * y + x) (h2 : N + y = 54321) (h3 : x = 54321 % 11) (h4 : y = 4938) : N = 49383 := by
  sorry

end find_original_number_l272_272283


namespace min_value_expression_l272_272662

open Real

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (∃ (y : ℝ), y = x * sqrt 2 ∧ ∀ (u : ℝ), ∀ (hu : u > 0), 
     sqrt ((x^2 + u^2) * (4 * x^2 + u^2)) / (x * u) ≥ 3 * sqrt 2) := 
sorry

end min_value_expression_l272_272662


namespace parabola_equation_l272_272788

theorem parabola_equation
  (axis_of_symmetry : ∀ x y : ℝ, x = 1)
  (focus : ∀ x y : ℝ, x = -1 ∧ y = 0) :
  ∀ y x : ℝ, y^2 = -4*x := 
sorry

end parabola_equation_l272_272788


namespace find_cone_height_l272_272475

noncomputable def cone_height (A l : ℝ) : ℝ := 
  let r := A / (l * Real.pi) in
  Real.sqrt (l^2 - r^2)

theorem find_cone_height : cone_height (65 * Real.pi) 13 = 12 := by
  let r := 5
  have h_eq : cone_height (65 * Real.pi) 13 = Real.sqrt (13^2 - r^2) := by 
    unfold cone_height
    sorry -- This step would carry out the necessary substeps.
  calc
    cone_height (65 * Real.pi) 13 = Real.sqrt (13^2 - r^2) : by exact h_eq
                         ... = Real.sqrt 144 : by norm_num
                         ... = 12 : by norm_num

end find_cone_height_l272_272475


namespace initial_bones_count_l272_272425

theorem initial_bones_count (B : ℕ) (h1 : B + 8 = 23) : B = 15 :=
sorry

end initial_bones_count_l272_272425


namespace fifth_pyTriple_is_correct_l272_272667

-- Definitions based on conditions from part (a)
def pyTriple (n : ℕ) : ℕ × ℕ × ℕ :=
  let a := 2 * n + 1
  let b := 2 * n * (n + 1)
  let c := b + 1
  (a, b, c)

-- Question: Prove that the 5th Pythagorean triple is (11, 60, 61)
theorem fifth_pyTriple_is_correct : pyTriple 5 = (11, 60, 61) :=
  by
    -- Skip the proof
    sorry

end fifth_pyTriple_is_correct_l272_272667


namespace sum_of_factors_eq_12_l272_272084

-- Define the polynomial for n = 1
def poly (x : ℤ) : ℤ := x^5 + x + 1

-- Define the two factors when x = 2
def factor1 (x : ℤ) : ℤ := x^3 - x^2 + 1
def factor2 (x : ℤ) : ℤ := x^2 + x + 1

-- State the sum of the two factors at x = 2 equals 12
theorem sum_of_factors_eq_12 (x : ℤ) (h : x = 2) : factor1 x + factor2 x = 12 :=
by {
  sorry
}

end sum_of_factors_eq_12_l272_272084


namespace least_positive_t_l272_272745

theorem least_positive_t (t : ℕ) (α : ℝ) (h1 : 0 < α ∧ α < π / 2)
  (h2 : π / 10 < α ∧ α ≤ π / 6) 
  (h3 : (3 * α)^2 = α * (π - 5 * α)) :
  t = 27 :=
by
  have hα : α = π / 14 := 
    by
      sorry
  sorry

end least_positive_t_l272_272745


namespace cone_height_l272_272482

theorem cone_height (l : ℝ) (A : ℝ) (h : ℝ) (r : ℝ) 
  (h_slant_height : l = 13)
  (h_lateral_area : A = 65 * π)
  (h_radius : r = 5)
  (h_height_formula : h = Real.sqrt (l^2 - r^2)) : 
  h = 12 := 
by 
  sorry

end cone_height_l272_272482


namespace alex_original_seat_l272_272676

-- We define a type for seats
inductive Seat where
  | s1 | s2 | s3 | s4 | s5 | s6
  deriving DecidableEq, Inhabited

open Seat

-- Define the initial conditions and movements
def initial_seats : (Fin 6 → Seat) := ![s1, s2, s3, s4, s5, s6]

def move_bella (s : Seat) : Seat :=
  match s with
  | s1 => s2
  | s2 => s3
  | s3 => s4
  | s4 => s5
  | s5 => s6
  | s6 => s1  -- invalid movement for the problem context, can handle separately

def move_coral (s : Seat) : Seat :=
  match s with
  | s1 => s6  -- two seats left from s1 wraps around to s6
  | s2 => s1
  | s3 => s2
  | s4 => s3
  | s5 => s4
  | s6 => s5

-- Dan and Eve switch seats among themselves
def switch_dan_eve (s : Seat) : Seat :=
  match s with
  | s3 => s4
  | s4 => s3
  | _ => s  -- all other positions remain the same

def move_finn (s : Seat) : Seat :=
  match s with
  | s1 => s2
  | s2 => s3
  | s3 => s4
  | s4 => s5
  | s5 => s6
  | s6 => s1  -- invalid movement for the problem context, can handle separately

-- Define the final seat for Alex
def alex_final_seat : Seat := s6  -- Alex returns to one end seat

-- Define a theorem for the proof of Alex's original seat being Seat.s1
theorem alex_original_seat :
  ∃ (original_seat : Seat), original_seat = s1 :=
  sorry

end alex_original_seat_l272_272676


namespace tom_total_spent_correct_l272_272858

-- Definitions for discount calculations
def original_price_skateboard : ℝ := 9.46
def discount_rate_skateboard : ℝ := 0.10
def discounted_price_skateboard : ℝ := original_price_skateboard * (1 - discount_rate_skateboard)

def original_price_marbles : ℝ := 9.56
def discount_rate_marbles : ℝ := 0.10
def discounted_price_marbles : ℝ := original_price_marbles * (1 - discount_rate_marbles)

def price_shorts : ℝ := 14.50

def original_price_action_figures : ℝ := 12.60
def discount_rate_action_figures : ℝ := 0.20
def discounted_price_action_figures : ℝ := original_price_action_figures * (1 - discount_rate_action_figures)

-- Total for all discounted items
def total_discounted_items : ℝ := 
  discounted_price_skateboard + discounted_price_marbles + price_shorts + discounted_price_action_figures

-- Currency conversion for video game
def price_video_game_eur : ℝ := 20.50
def exchange_rate_eur_to_usd : ℝ := 1.12
def price_video_game_usd : ℝ := price_video_game_eur * exchange_rate_eur_to_usd

-- Total amount spent including the video game
def total_spent : ℝ := total_discounted_items + price_video_game_usd

-- Lean proof statement
theorem tom_total_spent_correct :
  total_spent = 64.658 :=
by {
  -- This is a placeholder "by sorry" which means the proof is missing.
  sorry
}

end tom_total_spent_correct_l272_272858


namespace num_valid_pairs_equals_four_l272_272584

theorem num_valid_pairs_equals_four 
  (a b : ℕ) (ha : a > 0) (hb : b > 0) (hba : b > a)
  (hcond : a * b = 3 * (a - 4) * (b - 4)) :
  ∃! (s : Finset (ℕ × ℕ)), s.card = 4 ∧ 
    ∀ (p : ℕ × ℕ), p ∈ s → p.1 > 0 ∧ p.2 > 0 ∧ p.2 > p.1 ∧
      p.1 * p.2 = 3 * (p.1 - 4) * (p.2 - 4) := sorry

end num_valid_pairs_equals_four_l272_272584


namespace max_S_over_R_squared_l272_272807

theorem max_S_over_R_squared (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  let S := 2 * (a * b + b * c + c * a)
  let R := (sqrt (a^2 + b^2 + c^2)) / 2
  (S / R^2) ≤ (2 / 3) * (3 + sqrt 3) :=
by sorry

end max_S_over_R_squared_l272_272807


namespace find_solutions_l272_272485

noncomputable def f (x : ℝ) : ℝ := 2 * x ^ 3 - 9 * x ^ 2 + 6

theorem find_solutions :
  ∃ x1 x2 x3 : ℝ, f x1 = Real.sqrt 2 ∧ f x2 = Real.sqrt 2 ∧ f x3 = Real.sqrt 2 ∧ x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 :=
sorry

end find_solutions_l272_272485


namespace triangle_length_AX_l272_272647

theorem triangle_length_AX (A B C X : Type*) (AB AC BC AX XB : ℝ)
  (hAB : AB = 70) (hAC : AC = 42) (hBC : BC = 56)
  (h_bisect : ∃ (k : ℝ), AX = 3 * k ∧ XB = 4 * k) :
  AX = 30 := 
by
  sorry

end triangle_length_AX_l272_272647


namespace find_N_l272_272885

theorem find_N (x y : ℕ) (N : ℕ) (h1 : N = x * (x + 9)) (h2 : N = y * (y + 6)) : 
  N = 112 :=
  sorry

end find_N_l272_272885


namespace value_of_x_l272_272716

theorem value_of_x (x : ℝ) (h : 0.5 * x - (1 / 3) * x = 110) : x = 660 :=
sorry

end value_of_x_l272_272716


namespace total_cookies_l272_272831

def mona_cookies : ℕ := 20
def jasmine_cookies : ℕ := mona_cookies - 5
def rachel_cookies : ℕ := jasmine_cookies + 10

theorem total_cookies : mona_cookies + jasmine_cookies + rachel_cookies = 60 := 
by
  have h1 : jasmine_cookies = 15 := by sorry
  have h2 : rachel_cookies = 25 := by sorry
  have h3 : mona_cookies = 20 := by sorry
  sorry

end total_cookies_l272_272831


namespace draw_4_balls_ordered_l272_272099

theorem draw_4_balls_ordered : 
  ∀ (n m : ℕ), (n = 15) ∧ (m = 4) → (∏ i in finset.range(m), n - i) = 32760 :=
by
  intros n m h
  rcases h with ⟨hn, hm⟩
  rw [hn, hm]
  norm_num
  sorry

end draw_4_balls_ordered_l272_272099


namespace A_eq_B_l272_272366

noncomputable def A := Real.sqrt 5 + Real.sqrt (22 + 2 * Real.sqrt 5)
noncomputable def B := Real.sqrt (11 + 2 * Real.sqrt 29) 
                      + Real.sqrt (16 - 2 * Real.sqrt 29 
                                   + 2 * Real.sqrt (55 - 10 * Real.sqrt 29))

theorem A_eq_B : A = B := 
  sorry

end A_eq_B_l272_272366


namespace probability_of_24_is_1_div_1296_l272_272774

def sum_of_dice_is_24 (d1 d2 d3 d4 : ℕ) : Prop :=
  d1 + d2 + d3 + d4 = 24

def probability_of_six (d : ℕ) : Rat :=
  if d = 6 then 1 / 6 else 0

def probability_of_sum_24 (d1 d2 d3 d4 : ℕ) : Rat :=
  (probability_of_six d1) * (probability_of_six d2) * (probability_of_six d3) * (probability_of_six d4)

theorem probability_of_24_is_1_div_1296 :
  (probability_of_sum_24 6 6 6 6) = 1 / 1296 :=
by
  sorry

end probability_of_24_is_1_div_1296_l272_272774


namespace solve_equation_l272_272535

theorem solve_equation (x : ℝ) (h : x ≠ 2) :
  x^2 = (4*x^2 + 4) / (x - 2) ↔ (x = 1 + Real.sqrt 2 ∨ x = 1 - Real.sqrt 2 ∨ x = 4) :=
by
  sorry

end solve_equation_l272_272535


namespace number_subtracted_l272_272179

theorem number_subtracted (t k x : ℝ) (h1 : t = (5 / 9) * (k - x)) (h2 : t = 105) (h3 : k = 221) : x = 32 :=
by
  sorry

end number_subtracted_l272_272179


namespace number_of_solutions_l272_272014

-- Define the relevant trigonometric equation
def trig_equation (x : ℝ) : Prop := (Real.cos x)^2 + 3 * (Real.sin x)^2 = 1

-- Define the range for x
def in_range (x : ℝ) : Prop := -20 < x ∧ x < 100

-- Define the predicate that x satisfies both the trig equation and the range condition
def satisfies_conditions (x : ℝ) : Prop := trig_equation x ∧ in_range x

-- The final theorem statement (proof is omitted)
theorem number_of_solutions : 
  ∃ (count : ℕ), count = 38 ∧ ∀ (x : ℝ), satisfies_conditions x ↔ x = k * Real.pi ∧ -20 < k * Real.pi ∧ k * Real.pi < 100 := sorry

end number_of_solutions_l272_272014


namespace solve_for_k_l272_272024

noncomputable def f (k x : ℝ) : ℝ := k * x^2 + (k-1) * x + 2

theorem solve_for_k (k : ℝ) :
  (∀ x : ℝ, f k x = f k (-x)) ↔ k = 1 :=
by
  sorry

end solve_for_k_l272_272024


namespace rectangle_area_ratio_l272_272591

-- Define points in complex plane or as tuples (for 2D geometry)
structure Point where
  x : ℝ
  y : ℝ

-- Rectangle vertices
def A : Point := {x := 0, y := 0}
def B : Point := {x := 1, y := 0}
def C : Point := {x := 1, y := 2}
def D : Point := {x := 0, y := 2}

-- Centroid of triangle BCD
def E : Point := {x := 1.0, y := 1.333}

-- Point F such that DF = 1/4 * DA
def F : Point := {x := 1.5, y := 0}

-- Calculate areas of triangles and quadrilateral
noncomputable def area_triangle (P Q R : Point) : ℝ :=
  0.5 * abs ((Q.x - P.x) * (R.y - P.y) - (Q.y - P.y) * (R.x - P.x))

noncomputable def area_rectangle : ℝ :=
  2.0  -- Area of rectangle ABCD (1 * 2)

noncomputable def problem_statement : Prop :=
  let area_DFE := area_triangle D F E
  let area_ABEF := area_rectangle - area_triangle A B F - area_triangle D A F
  area_DFE / area_ABEF = 1 / 10.5

theorem rectangle_area_ratio :
  problem_statement :=
by
  sorry

end rectangle_area_ratio_l272_272591


namespace tan_product_l272_272744

theorem tan_product : 
(1 + Real.tan (Real.pi / 60)) * (1 + Real.tan (Real.pi / 30)) * (1 + Real.tan (Real.pi / 20)) * (1 + Real.tan (Real.pi / 15)) * (1 + Real.tan (Real.pi / 12)) * (1 + Real.tan (Real.pi / 10)) * (1 + Real.tan (Real.pi / 9)) * (1 + Real.tan (Real.pi / 6)) = 2^8 :=
by
  sorry 

end tan_product_l272_272744


namespace solve_for_x_l272_272386

theorem solve_for_x (x : ℚ) (h : x > 0) (hx : 3 * x^2 + 7 * x - 20 = 0) : x = 5 / 3 :=
by 
  sorry

end solve_for_x_l272_272386


namespace part_a_ellipse_and_lines_l272_272558

theorem part_a_ellipse_and_lines (x y : ℝ) : 
  (4 * x^2 + 8 * y^2 + 8 * y * abs y = 1) ↔ 
  ((y ≥ 0 ∧ (x^2 / (1/4) + y^2 / (1/16)) = 1) ∨ 
  (y < 0 ∧ ((x = 1/2) ∨ (x = -1/2)))) := 
sorry

end part_a_ellipse_and_lines_l272_272558


namespace find_large_number_l272_272258

theorem find_large_number (L S : ℕ) (h1 : L - S = 1311) (h2 : L = 11 * S + 11) : L = 1441 :=
sorry

end find_large_number_l272_272258


namespace minimize_travel_time_l272_272736

theorem minimize_travel_time
  (a b c d : ℝ)
  (v₁ v₂ v₃ v₄ : ℝ)
  (h1 : a > b)
  (h2 : b > c)
  (h3 : c > d)
  (h4 : v₁ > v₂)
  (h5 : v₂ > v₃)
  (h6 : v₃ > v₄) : 
  (a / v₁ + b / v₂ + c / v₃ + d / v₄) ≤ (a / v₁ + b / v₄ + c / v₃ + d / v₂) :=
sorry

end minimize_travel_time_l272_272736


namespace area_of_CDE_in_isosceles_triangle_l272_272956

noncomputable def isosceles_triangle_area (b : ℝ) (s : ℝ) (area : ℝ) : Prop :=
  area = (1 / 2) * b * s

noncomputable def cot (α : ℝ) : ℝ := 1 / Real.tan α

noncomputable def isosceles_triangle_vertex_angle (b : ℝ) (area : ℝ) (θ : ℝ) : Prop :=
  area = (b^2 / 4) * cot (θ / 2)

theorem area_of_CDE_in_isosceles_triangle (b θ area : ℝ) (hb : b = 3 * (2 * b / 3)) (hθ : θ = 100) (ha : area = 30) :
  ∃ CDE_area, CDE_area = area / 9 ∧ CDE_area = 10 / 3 :=
by
  sorry

end area_of_CDE_in_isosceles_triangle_l272_272956


namespace dice_sum_probability_l272_272759

theorem dice_sum_probability :
  let D := finset.range 1 7  -- outcomes of a fair six-sided die
  (∃! d1 d2 d3 d4 ∈ D, d1 + d2 + d3 + d4 = 24) ->
  (probability(space, {ω ∈ space | (ω 1 = 6) ∧ (ω 2 = 6) ∧ (ω 3 = 6) ∧ (ω 4 = 6)}) = 1/1296) :=
sorry

end dice_sum_probability_l272_272759


namespace valid_passwords_l272_272586

theorem valid_passwords (total_passwords restricted_passwords : Nat) 
  (h_total : total_passwords = 10^4)
  (h_restricted : restricted_passwords = 8) : 
  total_passwords - restricted_passwords = 9992 := by
  sorry

end valid_passwords_l272_272586


namespace ratio_a7_b7_l272_272398

variable {α : Type*}
variables {a_n b_n : ℕ → α} [AddGroup α] [Field α]
variables {S_n T_n : ℕ → α}

-- Define the sum of the first n terms for sequences a_n and b_n
def sum_of_first_terms_a (n : ℕ) := S_n n = (n * (a_n n + a_n (n-1))) / 2
def sum_of_first_terms_b (n : ℕ) := T_n n = (n * (b_n n + b_n (n-1))) / 2

-- Given condition about the ratio of sums
axiom ratio_condition (n : ℕ) : S_n n / T_n n = (3 * n - 2) / (2 * n + 1)

-- The statement to be proved
theorem ratio_a7_b7 : (a_n 7 / b_n 7) = (37 / 27) := sorry

end ratio_a7_b7_l272_272398


namespace cadence_worked_longer_by_5_months_l272_272741

-- Definitions
def months_old_company : ℕ := 36

def salary_old_company : ℕ := 5000

def salary_new_company : ℕ := 6000

def total_earnings : ℕ := 426000

-- Prove that Cadence worked 5 months longer at her new company
theorem cadence_worked_longer_by_5_months :
  ∃ x : ℕ, 
  total_earnings = salary_old_company * months_old_company + 
                  salary_new_company * (months_old_company + x)
  ∧ x = 5 :=
by {
  sorry
}

end cadence_worked_longer_by_5_months_l272_272741


namespace roots_in_ap_difference_one_l272_272077

theorem roots_in_ap_difference_one :
  ∀ (r1 r2 r3 : ℝ), 
    64 * r1^3 - 144 * r1^2 + 92 * r1 - 15 = 0 ∧
    64 * r2^3 - 144 * r2^2 + 92 * r2 - 15 = 0 ∧
    64 * r3^3 - 144 * r3^2 + 92 * r3 - 15 = 0 ∧
    (r2 - r1 = r3 - r2) →
    max (max r1 r2) r3 - min (min r1 r2) r3 = 1 := 
by
  intros r1 r2 r3 h
  sorry

end roots_in_ap_difference_one_l272_272077


namespace S10_equals_21_l272_272341

variable {a : ℕ → ℝ}

-- The given conditions
def initial_condition : a 1 = 3 := by sorry
def recursive_relation (n : ℕ) (h : 2 ≤ n) : a (n - 1) + a n + a (n + 1) = 6 := by sorry

-- Definition of the partial sum
def S (n : ℕ) : ℝ := (Finset.range n).sum a

-- The proof goal
theorem S10_equals_21 : S 10 = 21 := by sorry

end S10_equals_21_l272_272341


namespace range_of_a_max_area_of_triangle_l272_272339

variable (p a : ℝ) (h : p > 0)

def parabola_eq (x y : ℝ) := y ^ 2 = 2 * p * x
def line_eq (x y : ℝ) := y = x - a
def intersects_parabola (A B : ℝ × ℝ) := parabola_eq p A.fst A.snd ∧ line_eq a A.fst A.snd ∧ parabola_eq p B.fst B.snd ∧ line_eq a B.fst B.snd
def ab_length_le_2p (A B : ℝ × ℝ) := (Real.sqrt ((A.fst - B.fst)^2 + (A.snd - B.snd)^2) ≤ 2 * p)

theorem range_of_a
  (A B : ℝ × ℝ)
  (h_intersects : intersects_parabola a p A B)
  (h_ab_length : ab_length_le_2p p A B) :
  - p / 2 < a ∧ a ≤ - p / 4 := sorry

theorem max_area_of_triangle
  (A B : ℝ × ℝ) (N : ℝ × ℝ)
  (h_intersects : intersects_parabola a p A B)
  (h_ab_length : ab_length_le_2p p A B)
  (h_N : N.snd = 0) :
  ∃ (S : ℝ), S = Real.sqrt 2 * p^2 := sorry

end range_of_a_max_area_of_triangle_l272_272339


namespace storks_more_than_birds_l272_272263

-- Definitions based on given conditions
def initial_birds : ℕ := 3
def added_birds : ℕ := 2
def total_birds : ℕ := initial_birds + added_birds
def storks : ℕ := 6

-- Statement to prove the correct answer
theorem storks_more_than_birds : (storks - total_birds = 1) :=
by
  sorry

end storks_more_than_birds_l272_272263


namespace geometric_sequence_reciprocals_sum_l272_272779

theorem geometric_sequence_reciprocals_sum :
  ∃ (a : ℕ → ℝ) (q : ℝ), 
    (a 1 = 2) ∧ 
    (a 1 + a 3 + a 5 = 14) ∧ 
    (∀ n : ℕ, a (n + 1) = a n * q) → 
      (1 / a 1 + 1 / a 3 + 1 / a 5 = 7 / 8) :=
sorry

end geometric_sequence_reciprocals_sum_l272_272779


namespace choose_bar_chart_for_comparisons_l272_272687

/-- 
To easily compare the quantities of various items, one should choose a bar chart 
based on the characteristics of statistical charts.
-/
theorem choose_bar_chart_for_comparisons 
  (chart_type: Type) 
  (is_bar_chart: chart_type → Prop)
  (is_ideal_chart_for_comparison: chart_type → Prop)
  (bar_chart_ideal: ∀ c, is_bar_chart c → is_ideal_chart_for_comparison c) 
  (comparison_chart : chart_type) 
  (h: is_bar_chart comparison_chart): 
  is_ideal_chart_for_comparison comparison_chart := 
by
  exact bar_chart_ideal comparison_chart h

end choose_bar_chart_for_comparisons_l272_272687


namespace smallest_n_cube_mod_500_ends_in_388_l272_272451

theorem smallest_n_cube_mod_500_ends_in_388 :
  ∃ n : ℕ, 0 < n ∧ n^3 % 500 = 388 ∧ ∀ m : ℕ, 0 < m ∧ m^3 % 500 = 388 → n ≤ m :=
sorry

end smallest_n_cube_mod_500_ends_in_388_l272_272451


namespace evaluate_expression_l272_272319

theorem evaluate_expression : 
  ( (5 ^ 2014) ^ 2 - (5 ^ 2012) ^ 2 ) / ( (5 ^ 2013) ^ 2 - (5 ^ 2011) ^ 2 ) = 25 := 
by sorry

end evaluate_expression_l272_272319


namespace quadratic_inequality_solution_l272_272602

theorem quadratic_inequality_solution (y : ℝ) : 
  (y^2 - 9 * y + 14 ≤ 0) ↔ (2 ≤ y ∧ y ≤ 7) :=
sorry

end quadratic_inequality_solution_l272_272602


namespace probability_of_24_is_1_div_1296_l272_272775

def sum_of_dice_is_24 (d1 d2 d3 d4 : ℕ) : Prop :=
  d1 + d2 + d3 + d4 = 24

def probability_of_six (d : ℕ) : Rat :=
  if d = 6 then 1 / 6 else 0

def probability_of_sum_24 (d1 d2 d3 d4 : ℕ) : Rat :=
  (probability_of_six d1) * (probability_of_six d2) * (probability_of_six d3) * (probability_of_six d4)

theorem probability_of_24_is_1_div_1296 :
  (probability_of_sum_24 6 6 6 6) = 1 / 1296 :=
by
  sorry

end probability_of_24_is_1_div_1296_l272_272775


namespace boys_and_girls_in_class_l272_272957

theorem boys_and_girls_in_class (m d : ℕ)
  (A : (m - 1 = 10 ∧ (d - 1 = 14 ∨ d - 1 = 10 + 4 ∨ d - 1 = 10 - 4)) ∨ 
       (m - 1 = 14 - 4 ∧ (d - 1 = 14 ∨ d - 1 = 10 + 4 ∨ d - 1 = 10 - 4)))
  (B : (m - 1 = 13 ∧ (d - 1 = 11 ∨ d - 1 = 11 + 4 ∨ d - 1 = 11 - 4)) ∨ 
       (m - 1 = 11 - 4 ∧ (d - 1 = 11 ∨ d - 1 = 11 + 4 ∨ d - 1 = 11 - 4)))
  (C : (m - 1 = 13 ∧ (d - 1 = 19 ∨ d - 1 = 19 + 4 ∨ d - 1 = 19 - 4)) ∨ 
       (m - 1 = 19 - 4 ∧ (d - 1 = 19 ∨ d - 1 = 19 + 4 ∨ d - 1 = 19 - 4))) : 
  m = 14 ∧ d = 15 := 
sorry

end boys_and_girls_in_class_l272_272957


namespace domain_of_composite_function_l272_272168

theorem domain_of_composite_function :
  ∀ (f : ℝ → ℝ), (∀ x, -1 ≤ x ∧ x ≤ 3 → ∃ y, f x = y) →
  (∀ x, 0 ≤ x ∧ x ≤ 2 → ∃ y, f (2*x - 1) = y) :=
by
  intros f domain_f x hx
  sorry

end domain_of_composite_function_l272_272168


namespace pow_congruence_modulus_p_squared_l272_272978

theorem pow_congruence_modulus_p_squared (p : ℕ) (a b : ℤ) (hp : Nat.Prime p) (h : a ≡ b [ZMOD p]) : a^p ≡ b^p [ZMOD p^2] :=
sorry

end pow_congruence_modulus_p_squared_l272_272978


namespace all_good_rational_are_integers_l272_272730

def good_rational (x : ℚ) (α : ℝ) (N : ℕ) : Prop :=
  x > 1 ∧ ∀ n : ℕ, n ≥ N → |(x^n - (x^n).floor) - α| ≤ 1 / (2 * (x.num.natAbs + x.denom.natAbs))

theorem all_good_rational_are_integers (x : ℚ) :
  (∃ α : ℝ, ∃ N : ℕ, good_rational x α N) → ∃ k : ℤ, x = k ∧ k > 1 :=
by
  sorry

end all_good_rational_are_integers_l272_272730


namespace find_balloons_given_to_Fred_l272_272532

variable (x : ℝ)
variable (Sam_initial_balance : ℝ := 46.0)
variable (Dan_balance : ℝ := 16.0)
variable (total_balance : ℝ := 52.0)

theorem find_balloons_given_to_Fred
  (h : Sam_initial_balance - x + Dan_balance = total_balance) :
  x = 10.0 :=
by
  sorry

end find_balloons_given_to_Fred_l272_272532


namespace number_of_factors_in_224_l272_272269

def smallest_is_half_largest (n1 n2 : ℕ) : Prop :=
  n1 * 2 = n2

theorem number_of_factors_in_224 :
  ∃ n1 n2 n3 : ℕ, n1 * n2 * n3 = 224 ∧ smallest_is_half_largest (min n1 (min n2 n3)) (max n1 (max n2 n3)) ∧
    (if h : n1 < n2 ∧ n1 < n3 then
      if h2 : n2 < n3 then 
        smallest_is_half_largest n1 n3 
        else 
        smallest_is_half_largest n1 n2 
    else if h : n2 < n1 ∧ n2 < n3 then 
      if h2 : n1 < n3 then 
        smallest_is_half_largest n2 n3 
        else 
        smallest_is_half_largest n2 n1 
    else 
      if h2 : n1 < n2 then 
        smallest_is_half_largest n3 n2 
        else 
        smallest_is_half_largest n3 n1) = true ∧ 
    (if h : n1 < n2 ∧ n1 < n3 then
       if h2 : n2 < n3 then 
         n1 * n2 * n3 
         else 
         n1 * n3 * n2 
     else if h : n2 < n1 ∧ n2 < n3 then 
       if h2 : n1 < n3 then 
         n2 * n1 * n3
         else 
         n2 * n3 * n1 
     else 
       if h2 : n1 < n2 then 
         n3 * n1 * n2 
         else 
         n3 * n2 * n1) = 224 := sorry

end number_of_factors_in_224_l272_272269


namespace complex_sum_real_imag_l272_272911

theorem complex_sum_real_imag : 
  (Complex.re ((Complex.I / (1 + Complex.I)) - (1 / (2 * Complex.I))) + 
  Complex.im ((Complex.I / (1 + Complex.I)) - (1 / (2 * Complex.I)))) = 3/2 := 
by sorry

end complex_sum_real_imag_l272_272911


namespace total_penalty_kicks_l272_272224

theorem total_penalty_kicks (total_players : ℕ) (goalies : ℕ) (hoop_challenges : ℕ)
  (h_total : total_players = 25) (h_goalies : goalies = 5) (h_hoop_challenges : hoop_challenges = 10) :
  (goalies * (total_players - 1)) = 120 :=
by
  sorry

end total_penalty_kicks_l272_272224


namespace min_value_geometric_sequence_l272_272638

-- Definitions based on conditions
noncomputable def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) := ∀ n : ℕ, a (n + 1) = a n * q

-- We need to state the problem using the above definitions
theorem min_value_geometric_sequence 
  (a : ℕ → ℝ) (q : ℝ) (s t : ℕ) 
  (h_seq : is_geometric_sequence a q) 
  (h_q : q ≠ 1) 
  (h_st : a s * a t = (a 5) ^ 2) 
  (h_s_pos : s > 0) 
  (h_t_pos : t > 0) 
  : 4 / s + 1 / (4 * t) = 5 / 8 := sorry

end min_value_geometric_sequence_l272_272638


namespace converse_not_true_prop_B_l272_272200

noncomputable def line_in_plane (b : Type) (α : Type) : Prop := sorry
noncomputable def perp_line_plane (b : Type) (β : Type) : Prop := sorry
noncomputable def perp_planes (α : Type) (β : Type) : Prop := sorry
noncomputable def parallel_planes (α : Type) (β : Type) : Prop := sorry

variables (a b c : Type) (α β : Type)

theorem converse_not_true_prop_B :
  (line_in_plane b α) → (perp_planes α β) → ¬ (perp_line_plane b β) :=
sorry

end converse_not_true_prop_B_l272_272200


namespace total_weight_moved_l272_272039

-- Define the given conditions as Lean definitions
def weight_per_rep : ℕ := 15
def number_of_reps : ℕ := 10
def number_of_sets : ℕ := 3

-- Define the theorem to prove total weight moved is 450 pounds
theorem total_weight_moved : weight_per_rep * number_of_reps * number_of_sets = 450 := by
  sorry

end total_weight_moved_l272_272039


namespace total_cookies_l272_272829

def MonaCookies : ℕ := 20
def JasmineCookies : ℕ := MonaCookies - 5
def RachelCookies : ℕ := JasmineCookies + 10

theorem total_cookies : MonaCookies + JasmineCookies + RachelCookies = 60 := by
  -- Since we don't need to provide the solution steps, we simply use sorry.
  sorry

end total_cookies_l272_272829


namespace complementary_events_A_B_l272_272746

def is_odd (n : ℕ) : Prop := n % 2 = 1
def is_even (n : ℕ) : Prop := n % 2 = 0
def is_multiple_of_3 (n : ℕ) : Prop := n % 3 = 0

def A (n : ℕ) : Prop := is_odd n
def B (n : ℕ) : Prop := is_even n
def C (n : ℕ) : Prop := is_multiple_of_3 n

theorem complementary_events_A_B :
  (∀ n, A n → ¬ B n) ∧ (∀ n, B n → ¬ A n) ∧ (∀ n, A n ∨ B n) :=
  sorry

end complementary_events_A_B_l272_272746


namespace max_cards_possible_l272_272111

-- Define the dimensions for the cardboard and the card.
def cardboard_length : ℕ := 48
def cardboard_width : ℕ := 36
def card_length : ℕ := 16
def card_width : ℕ := 12

-- State the theorem to prove the maximum number of cards.
theorem max_cards_possible : (cardboard_length / card_length) * (cardboard_width / card_width) = 9 :=
by
  sorry -- Skip the proof, as only the statement is required.

end max_cards_possible_l272_272111


namespace smallest_n_l272_272559

theorem smallest_n (n : ℕ) : 
  (n % 6 = 2) ∧ (n % 7 = 3) ∧ (n % 8 = 4) → n = 8 :=
  by sorry

end smallest_n_l272_272559


namespace range_of_x_if_cos2_gt_sin2_l272_272174

theorem range_of_x_if_cos2_gt_sin2 (x : ℝ) (h1 : x ∈ Set.Icc 0 Real.pi) (h2 : Real.cos x ^ 2 > Real.sin x ^ 2) :
  x ∈ Set.Ico 0 (Real.pi / 4) ∪ Set.Ioc (3 * Real.pi / 4) Real.pi :=
by
  sorry

end range_of_x_if_cos2_gt_sin2_l272_272174


namespace all_Xanths_are_Yelps_and_Wicks_l272_272637

-- Definitions for Zorbs, Yelps, Xanths, and Wicks
variable {U : Type} (Zorb Yelp Xanth Wick : U → Prop)

-- Conditions from the problem
axiom all_Zorbs_are_Yelps : ∀ u, Zorb u → Yelp u
axiom all_Xanths_are_Zorbs : ∀ u, Xanth u → Zorb u
axiom all_Xanths_are_Wicks : ∀ u, Xanth u → Wick u

-- The goal is to prove that all Xanths are Yelps and are Wicks
theorem all_Xanths_are_Yelps_and_Wicks : ∀ u, Xanth u → Yelp u ∧ Wick u := sorry

end all_Xanths_are_Yelps_and_Wicks_l272_272637


namespace price_increase_percentage_l272_272040

variables
  (coffees_daily_before : ℕ := 4)
  (price_per_coffee_before : ℝ := 2)
  (coffees_daily_after : ℕ := 2)
  (price_increase_savings : ℝ := 2)
  (spending_before := coffees_daily_before * price_per_coffee_before)
  (spending_after := spending_before - price_increase_savings)
  (price_per_coffee_after := spending_after / coffees_daily_after)

theorem price_increase_percentage :
  ((price_per_coffee_after - price_per_coffee_before) / price_per_coffee_before) * 100 = 50 :=
by
  sorry

end price_increase_percentage_l272_272040


namespace simplify_polynomial_l272_272217

theorem simplify_polynomial :
  (2 * x^6 + x^5 + 3 * x^4 + 7 * x^2 + 2 * x + 25) - (x^6 + 2 * x^5 + x^4 + x^3 + 8 * x^2 + 15) = 
  (x^6 - x^5 + 2 * x^4 - x^3 - x^2 + 2 * x + 10) :=
by
  sorry

end simplify_polynomial_l272_272217


namespace cash_sales_is_48_l272_272391

variable (total_sales : ℝ) (credit_fraction : ℝ) (cash_sales : ℝ)

-- Conditions: Total sales were $80, 2/5 of the total sales were credit sales
def problem_conditions := total_sales = 80 ∧ credit_fraction = 2/5 ∧ cash_sales = (1 - credit_fraction) * total_sales

-- Question: Prove that the amount of cash sales Mr. Brandon made is $48.
theorem cash_sales_is_48 (h : problem_conditions total_sales credit_fraction cash_sales) : 
  cash_sales = 48 :=
by
  sorry

end cash_sales_is_48_l272_272391


namespace sin_cos_sixth_power_sum_l272_272657

theorem sin_cos_sixth_power_sum (θ : ℝ) (h : Real.sin (2 * θ) = Real.sqrt 2 / 2) : 
  (Real.sin θ)^6 + (Real.cos θ)^6 = 5 / 8 :=
by
  sorry

end sin_cos_sixth_power_sum_l272_272657


namespace problem1_problem2_l272_272303

-- Problem 1
theorem problem1 : 
  (-2.8) - (-3.6) + (-1.5) - (3.6) = -4.3 := 
by 
  sorry

-- Problem 2
theorem problem2 :
  (- (5 / 6 : ℚ) + (1 / 3 : ℚ) - (3 / 4 : ℚ)) * (-24) = 30 := 
by 
  sorry

end problem1_problem2_l272_272303


namespace solution_set_of_quadratic_inequality_l272_272237

theorem solution_set_of_quadratic_inequality (x : ℝ) :
  (x - 2) * (x + 2) < 5 ↔ -3 < x ∧ x < 3 :=
by 
  sorry

end solution_set_of_quadratic_inequality_l272_272237


namespace johns_haircut_tip_percentage_l272_272652

noncomputable def percent_of_tip (annual_spending : ℝ) (haircut_cost : ℝ) (haircut_frequency : ℕ) : ℝ := 
  ((annual_spending / haircut_frequency - haircut_cost) / haircut_cost) * 100

theorem johns_haircut_tip_percentage : 
  let hair_growth_rate : ℝ := 1.5
  let initial_length : ℝ := 6
  let max_length : ℝ := 9
  let haircut_cost : ℝ := 45
  let annual_spending : ℝ := 324
  let months_in_year : ℕ := 12
  let growth_period := 2 -- months it takes for hair to grow 3 inches
  let haircuts_per_year := months_in_year / growth_period -- number of haircuts per year
  percent_of_tip annual_spending haircut_cost haircuts_per_year = 20 := by
  sorry

end johns_haircut_tip_percentage_l272_272652


namespace functional_equation_solution_l272_272368

theorem functional_equation_solution :
  ∃ f : ℝ → ℝ,
  (f 1 = 1 ∧ (∀ x y : ℝ, f (x * y + f x) = x * f y + f x)) ∧ f (1/2) = 1/2 :=
by
  sorry

end functional_equation_solution_l272_272368


namespace cone_height_l272_272476

theorem cone_height (l : ℝ) (LA : ℝ) (h : ℝ) (r : ℝ) (h_eq : h = sqrt (l^2 - r^2))
  (LA_eq : LA = π * r * l) (l_val : l = 13) (LA_val : LA = 65 * π) : h = 12 :=
by
  -- substitution of the values of l and LA
  have l_13 := l_val,
  have LA_65π := LA_val,
  
  -- solve for r from LA = π * r * l
  have r_val : r = LA / (π * l), sorry,

  -- then use the Pythagorean theorem to solve for h
  have h_12 : h = sqrt (l^2 - r^2), sorry,

  -- final conclusion: h must be equal to 12
  exact sorry

end cone_height_l272_272476


namespace geometric_sequence_properties_l272_272810

noncomputable def geometric_sequence_sum (r a1 : ℝ) : Prop :=
  a1 * (r^3 + r^4) = 27 ∨ a1 * (r^3 + r^4) = -27

theorem geometric_sequence_properties (a1 r : ℝ) (h1 : a1 + a1 * r = 1) (h2 : a1 * r^2 + a1 * r^3 = 9) :
  geometric_sequence_sum r a1 :=
sorry

end geometric_sequence_properties_l272_272810


namespace best_model_l272_272512

theorem best_model (R1 R2 R3 R4 : ℝ) (h1 : R1 = 0.55) (h2 : R2 = 0.65) (h3 : R3 = 0.79) (h4 : R4 = 0.95) :
  R4 > R3 ∧ R4 > R2 ∧ R4 > R1 :=
by {
  sorry
}

end best_model_l272_272512


namespace tax_deduction_is_correct_l272_272304

-- Define the hourly wage and tax rate
def hourly_wage_dollars : ℝ := 25
def tax_rate : ℝ := 0.021

-- Define the conversion from dollars to cents
def dollars_to_cents (dollars : ℝ) : ℝ := dollars * 100

-- Calculate the hourly wage in cents
def hourly_wage_cents : ℝ := dollars_to_cents hourly_wage_dollars

-- Calculate the tax deducted in cents per hour
def tax_deduction_cents (wage : ℝ) (rate : ℝ) : ℝ := rate * wage

-- State the theorem that needs to be proven
theorem tax_deduction_is_correct :
  tax_deduction_cents hourly_wage_cents tax_rate = 52.5 :=
by
  sorry

end tax_deduction_is_correct_l272_272304


namespace income_expenditure_ratio_l272_272232

theorem income_expenditure_ratio (I E S : ℕ) (hI : I = 19000) (hS : S = 11400) (hRel : S = I - E) :
  I / E = 95 / 38 :=
by
  sorry

end income_expenditure_ratio_l272_272232


namespace heptagon_angle_sum_l272_272191

theorem heptagon_angle_sum 
  (angle_A angle_B angle_C angle_D angle_E angle_F angle_G : ℝ) 
  (h : angle_A + angle_B + angle_C + angle_D + angle_E + angle_F + angle_G = 540) :
  angle_A + angle_B + angle_C + angle_D + angle_E + angle_F + angle_G = 540 :=
by
  sorry

end heptagon_angle_sum_l272_272191


namespace unique_integer_cube_triple_l272_272695

theorem unique_integer_cube_triple (x : ℤ) (h : x^3 < 3 * x) : x = 1 := 
sorry

end unique_integer_cube_triple_l272_272695


namespace g_675_eq_42_l272_272050

theorem g_675_eq_42 
  (g : ℕ → ℕ) 
  (h_mul : ∀ x y : ℕ, x > 0 → y > 0 → g (x * y) = g x + g y) 
  (h_g15 : g 15 = 18) 
  (h_g45 : g 45 = 24) : g 675 = 42 :=
by
  sorry

end g_675_eq_42_l272_272050


namespace complement_intersection_eq_complement_l272_272928

open Set

theorem complement_intersection_eq_complement (U A B : Set ℕ) (hU : U = {1, 2, 3, 4}) (hA : A = {1, 2}) (hB : B = {2, 4}) :
  (U \ (A ∩ B)) = {1, 3, 4} :=
by
  sorry

end complement_intersection_eq_complement_l272_272928


namespace single_discount_equivalence_l272_272110

noncomputable def original_price : ℝ := 50
noncomputable def discount1 : ℝ := 0.15
noncomputable def discount2 : ℝ := 0.10
noncomputable def apply_discount (price : ℝ) (discount : ℝ) : ℝ :=
  price * (1 - discount)
noncomputable def effective_discount_price := 
  apply_discount (apply_discount original_price discount1) discount2
noncomputable def effective_discount :=
  (original_price - effective_discount_price) / original_price

theorem single_discount_equivalence :
  effective_discount = 0.235 := by
  sorry

end single_discount_equivalence_l272_272110


namespace rank_identity_l272_272041

theorem rank_identity (n p : ℕ) (A : Matrix (Fin n) (Fin n) ℝ) 
  (h1: 2 ≤ n) (h2: 2 ≤ p) (h3: A^(p+1) = A) : 
  Matrix.rank A + Matrix.rank (1 - A^p) = n := 
  sorry

end rank_identity_l272_272041


namespace kaylin_age_32_l272_272196

-- Defining the ages of the individuals as variables
variables (Kaylin Sarah Eli Freyja Alfred Olivia : ℝ)

-- Defining the given conditions
def conditions : Prop := 
  (Kaylin = Sarah - 5) ∧
  (Sarah = 2 * Eli) ∧
  (Eli = Freyja + 9) ∧
  (Freyja = 2.5 * Alfred) ∧
  (Alfred = (3/4) * Olivia) ∧
  (Freyja = 9.5)

-- Main statement to prove
theorem kaylin_age_32 (h : conditions Kaylin Sarah Eli Freyja Alfred Olivia) : Kaylin = 32 :=
by
  sorry

end kaylin_age_32_l272_272196


namespace complete_square_solution_l272_272219

theorem complete_square_solution (x : ℝ) :
  x^2 - 8 * x + 6 = 0 → (x - 4)^2 = 10 :=
by
  intro h
  -- Proof would go here
  sorry

end complete_square_solution_l272_272219


namespace print_pages_500_l272_272193

theorem print_pages_500 (cost_per_page cents total_dollars) : 
  cost_per_page = 3 → 
  total_dollars = 15 → 
  cents = 100 * total_dollars → 
  (cents / cost_per_page) = 500 :=
by 
  intros h1 h2 h3
  sorry

end print_pages_500_l272_272193


namespace draw_4_balls_in_order_l272_272096

theorem draw_4_balls_in_order :
  let choices : list ℕ := [15, 14, 13, 12] in
  (choices.foldr (λ x acc => x * acc) 1) = 32760 :=
by
  sorry

end draw_4_balls_in_order_l272_272096


namespace prove_R_value_l272_272526

noncomputable def geometric_series (Q : ℕ) : ℕ :=
  (2^(Q + 1) - 1)

noncomputable def R (F : ℕ) : ℝ :=
  Real.sqrt (Real.log (1 + F) / Real.log 2)

theorem prove_R_value :
  let F := geometric_series 120
  R F = 11 :=
by
  sorry

end prove_R_value_l272_272526


namespace seahorse_penguin_ratio_l272_272301

theorem seahorse_penguin_ratio :
  ∃ S P : ℕ, S = 70 ∧ P = S + 85 ∧ Nat.gcd 70 (S + 85) = 5 ∧ 70 / Nat.gcd 70 (S + 85) = 14 ∧ (S + 85) / Nat.gcd 70 (S + 85) = 31 :=
by
  sorry

end seahorse_penguin_ratio_l272_272301


namespace number_of_valid_arrangements_l272_272991

def is_ascending (l : List ℕ) : Prop :=
  ∀ i j, i < j → l.nth i ≤ l.nth j

def is_descending (l : List ℕ) : Prop :=
  ∀ i j, i < j → l.nth i ≥ l.nth j

def remove_one_is_ordered (l : List ℕ) : Prop :=
  ∃ (i : ℕ), (is_ascending (l.removeNth i) ∨ is_descending (l.removeNth i))

def valid_arrangements_count (cards : List ℕ) : ℕ :=
  -- counting the number of valid arrangements
  if (cards.length = 7
        ∧ ∀ i, i ∈ cards → 1 ≤ i ∧ i ≤ 7 ∧ (remove_one_is_ordered cards)) then 4 else 0

theorem number_of_valid_arrangements :
  valid_arrangements_count [1,2,3,4,5,6,7] = 4 :=
by sorry

end number_of_valid_arrangements_l272_272991


namespace factor_tree_value_l272_272353

theorem factor_tree_value :
  ∀ (X Y Z F G : ℕ),
  X = Y * Z → 
  Y = 7 * F → 
  F = 2 * 5 → 
  Z = 11 * G → 
  G = 7 * 3 → 
  X = 16170 := 
by
  intros X Y Z F G
  sorry

end factor_tree_value_l272_272353


namespace triangle_inequality_l272_272371

theorem triangle_inequality (a b c : ℝ) (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a) :
  a^2 * (b + c - a) + b^2 * (c + a - b) + c^2 * (a + b - c) ≤ 3 * a * b * c :=
by
  sorry

end triangle_inequality_l272_272371


namespace arithmetic_expression_value_l272_272150

theorem arithmetic_expression_value :
  (19 + 43 / 151) * 151 = 2910 :=
by {
  sorry
}

end arithmetic_expression_value_l272_272150


namespace value_of_squares_l272_272977

-- Define the conditions
variables (p q : ℝ)

-- State the theorem with the given conditions and the proof goal
theorem value_of_squares (h1 : p * q = 12) (h2 : p + q = 8) : p ^ 2 + q ^ 2 = 40 :=
sorry

end value_of_squares_l272_272977


namespace find_sets_l272_272619

open Set

noncomputable def U := ℝ
def A := {x : ℝ | Real.log x / Real.log 2 <= 2}
def B := {x : ℝ | x ≥ 1}

theorem find_sets (x : ℝ) :
  (A = {x : ℝ | -1 ≤ x ∧ x < 3}) ∧
  (B = {x : ℝ | -2 < x ∧ x ≤ 3}) ∧
  (compl A ∩ B = {x : ℝ | (-2 < x ∧ x < -1) ∨ x = 3}) :=
  sorry

end find_sets_l272_272619


namespace ratio_of_share_l272_272266

/-- A certain amount of money is divided amongst a, b, and c. 
The share of a is $122, and the total amount of money is $366. 
Prove that the ratio of a's share to the combined share of b and c is 1 / 2. -/
theorem ratio_of_share (a b c : ℝ) (total share_a : ℝ) (h1 : a + b + c = total) 
  (h2 : total = 366) (h3 : share_a = 122) : share_a / (total - share_a) = 1 / 2 := by
  sorry

end ratio_of_share_l272_272266


namespace congruence_is_sufficient_but_not_necessary_for_equal_area_l272_272162

-- Definition of conditions
def Congruent (Δ1 Δ2 : Type) : Prop := sorry -- Definition of congruent triangles
def EqualArea (Δ1 Δ2 : Type) : Prop := sorry -- Definition of triangles with equal area

-- Theorem statement
theorem congruence_is_sufficient_but_not_necessary_for_equal_area 
  (Δ1 Δ2 : Type) :
  (Congruent Δ1 Δ2 → EqualArea Δ1 Δ2) ∧ (¬ (EqualArea Δ1 Δ2 → Congruent Δ1 Δ2)) :=
sorry

end congruence_is_sufficient_but_not_necessary_for_equal_area_l272_272162


namespace lines_concurrent_on_OI_l272_272980

theorem lines_concurrent_on_OI
  (ABC : Triangle)
  (I : Point)
  (O : Point)
  (ΓA ΓB ΓC : Circle)
  (hI : I = incenter ABC)
  (hO : O = circumcenter ABC)
  (hΓA : ∀ (B C : Point), B ≠ C → B ∈ ΓA ∧ C ∈ ΓA ∧ is_tangent_to ΓA (incircle ABC))
  (hΓB : ∀ (A C : Point), A ≠ C → A ∈ ΓB ∧ C ∈ ΓB ∧ is_tangent_to ΓB (incircle ABC))
  (hΓC : ∀ (A B : Point), A ≠ B → A ∈ ΓC ∧ B ∈ ΓC ∧ is_tangent_to ΓC (incircle ABC))
  (A' B' C' : Point)
  (hAB : ∀ (B C : Point), B ≠ C → (B ∈ ΓB ∧ C ∈ ΓB ∧ B ∈ ΓC) ∧ C ∈ ΓC ∧ (A ∈ {A, A'} ∧ A' ∈ ΓB ∩ ΓC))
  (hBC : ∀ (A C : Point), A ≠ C → (A ∈ ΓA ∧ C ∈ ΓA ∧ A ∈ ΓC) ∧ C ∈ ΓC ∧ (B ∈ {B, B'} ∧ B' ∈ ΓA ∩ ΓC))
  (hCA : ∀ (A B : Point), A ≠ B → (A ∈ ΓA ∧ B ∈ ΓA ∧ A ∈ ΓB) ∧ B ∈ ΓB ∧ (C ∈ {C, C'} ∧ C' ∈ ΓA ∩ ΓB)) :
  ∃ (Q : Point), is_concurrent (AA' BB' CC') ∧ Q ∈ OI := sorry

end lines_concurrent_on_OI_l272_272980


namespace fraction_of_area_l272_272987

def larger_square_side : ℕ := 6
def shaded_square_side : ℕ := 2

def larger_square_area : ℕ := larger_square_side * larger_square_side
def shaded_square_area : ℕ := shaded_square_side * shaded_square_side

theorem fraction_of_area : (shaded_square_area : ℚ) / larger_square_area = 1 / 9 :=
by
  -- proof omitted
  sorry

end fraction_of_area_l272_272987


namespace no_convex_quad_with_given_areas_l272_272649

theorem no_convex_quad_with_given_areas :
  ¬ ∃ (A B C D M : Type) 
    (T_MAB T_MBC T_MDA T_MDC : ℕ) 
    (H1 : T_MAB = 1) 
    (H2 : T_MBC = 2)
    (H3 : T_MDA = 3) 
    (H4 : T_MDC = 4),
    true :=
by {
  sorry
}

end no_convex_quad_with_given_areas_l272_272649


namespace marathon_winner_average_speed_l272_272426

-- Define the conditions of the problem
def marathon_distance : ℝ := 42
def start_time : ℝ := 11 + 30 / 60
def end_time : ℝ := 13 + 45 / 60

-- Calculate the total running time
def running_time : ℝ := end_time - start_time

-- Calculate the average speed
def average_speed : ℝ := marathon_distance / running_time

-- The theorem we want to prove
theorem marathon_winner_average_speed : average_speed = 18.6 := by
  -- The proof goes here
  sorry

end marathon_winner_average_speed_l272_272426


namespace quadrilateral_area_l272_272531

theorem quadrilateral_area (EF FG EH HG : ℕ) (hEFH : EF * EF + FG * FG = 25)
(hEHG : EH * EH + HG * HG = 25) (h_distinct : EF ≠ EH ∧ FG ≠ HG) 
(h_greater_one : EF > 1 ∧ FG > 1 ∧ EH > 1 ∧ HG > 1) :
  (EF * FG) / 2 + (EH * HG) / 2 = 12 := 
sorry

end quadrilateral_area_l272_272531


namespace no_three_digit_number_exists_l272_272315

theorem no_three_digit_number_exists (a b c : ℕ) (h₁ : 0 ≤ a ∧ a < 10) (h₂ : 0 ≤ b ∧ b < 10) (h₃ : 0 ≤ c ∧ c < 10) (h₄ : a ≠ 0) :
  ¬ ∃ k : ℕ, k^2 = 99 * (a - c) :=
by
  sorry

end no_three_digit_number_exists_l272_272315


namespace farm_produce_weeks_l272_272665

def eggs_needed_per_week (saly_eggs ben_eggs ked_eggs : ℕ) : ℕ :=
  saly_eggs + ben_eggs + ked_eggs

def number_of_weeks (total_eggs : ℕ) (weekly_eggs : ℕ) : ℕ :=
  total_eggs / weekly_eggs

theorem farm_produce_weeks :
  let saly_eggs := 10
  let ben_eggs := 14
  let ked_eggs := 14 / 2
  let total_eggs := 124
  let weekly_eggs := eggs_needed_per_week saly_eggs ben_eggs ked_eggs
  number_of_weeks total_eggs weekly_eggs = 4 :=
by
  sorry 

end farm_produce_weeks_l272_272665


namespace find_power_of_4_l272_272453

theorem find_power_of_4 (x : Nat) : 
  (2 * x + 5 + 2 = 29) -> 
  (x = 11) :=
by
  sorry

end find_power_of_4_l272_272453


namespace sum_of_squares_of_real_solutions_l272_272912

theorem sum_of_squares_of_real_solutions (x : ℝ) (h : x ^ 64 = 16 ^ 16) : 
  (x = 2 ∨ x = -2) → (x ^ 2 + (-x) ^ 2) = 8 :=
by
  sorry

end sum_of_squares_of_real_solutions_l272_272912


namespace ceil_sqrt_250_eq_16_l272_272598

theorem ceil_sqrt_250_eq_16 : ⌈Real.sqrt 250⌉ = 16 :=
by
  have h1 : (15 : ℝ) < Real.sqrt 250 := sorry
  have h2 : Real.sqrt 250 < 16 := sorry
  exact sorry

end ceil_sqrt_250_eq_16_l272_272598


namespace degree_monomial_equal_four_l272_272069

def degree_of_monomial (a b : ℝ) := 
  (3 + 1)

theorem degree_monomial_equal_four (a b : ℝ) 
  (h : a^3 * b = (2/3) * a^3 * b) : 
  degree_of_monomial a b = 4 :=
by sorry

end degree_monomial_equal_four_l272_272069


namespace socks_pair_count_l272_272352

theorem socks_pair_count :
  let white := 5
  let brown := 5
  let blue := 3
  let green := 2
  (white * brown) + (white * blue) + (white * green) + (brown * blue) + (brown * green) + (blue * green) = 81 :=
by
  intros
  sorry

end socks_pair_count_l272_272352


namespace Michaela_needs_20_oranges_l272_272824

variable (M : ℕ)
variable (C : ℕ)

theorem Michaela_needs_20_oranges 
  (h1 : C = 2 * M)
  (h2 : M + C = 60):
  M = 20 :=
by 
  sorry

end Michaela_needs_20_oranges_l272_272824


namespace decrease_hours_by_13_percent_l272_272571

theorem decrease_hours_by_13_percent (W H : ℝ) (hW_pos : W > 0) (hH_pos : H > 0) :
  let W_new := 1.15 * W
  let H_new := H / 1.15
  let income_decrease_percentage := (1 - H_new / H) * 100
  abs (income_decrease_percentage - 13.04) < 0.01 := 
by
  sorry

end decrease_hours_by_13_percent_l272_272571


namespace prove_correct_y_l272_272090

noncomputable def find_larger_y (x y : ℕ) : Prop :=
  y - x = 1365 ∧ y = 6 * x + 15

noncomputable def correct_y : ℕ := 1635

theorem prove_correct_y (x y : ℕ) (h : find_larger_y x y) : y = correct_y :=
by
  sorry

end prove_correct_y_l272_272090


namespace cube_less_than_triple_l272_272697

theorem cube_less_than_triple : ∀ x : ℤ, (x^3 < 3*x) ↔ (x = 1 ∨ x = -2) :=
by
  sorry

end cube_less_than_triple_l272_272697


namespace inequality_holds_l272_272447

variable {a b c r : ℝ}
variable (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c)

/-- 
To prove that the inequality r (ab + bc + ca) + (3 - r) (1/a + 1/b + 1/c) ≥ 9 
is true for all r satisfying 0 < r < 3 and for arbitrary positive reals a, b, c. 
-/
theorem inequality_holds (h : 0 < r ∧ r < 3) : 
  r * (a * b + b * c + c * a) + (3 - r) * (1 / a + 1 / b + 1 / c) ≥ 9 := by
  sorry

end inequality_holds_l272_272447


namespace student_divisor_l272_272952

theorem student_divisor (x : ℕ) : (24 * x = 42 * 36) → x = 63 := 
by
  intro h
  sorry

end student_divisor_l272_272952


namespace popton_school_bus_total_toes_l272_272057

-- Define the number of toes per hand for each race
def toes_per_hand_hoopit : ℕ := 3
def toes_per_hand_neglart : ℕ := 2
def toes_per_hand_zentorian : ℕ := 4

-- Define the number of hands for each race
def hands_per_hoopit : ℕ := 4
def hands_per_neglart : ℕ := 5
def hands_per_zentorian : ℕ := 6

-- Define the number of students from each race on the bus
def num_hoopits : ℕ := 7
def num_neglarts : ℕ := 8
def num_zentorians : ℕ := 5

-- Calculate the total number of toes on the bus
def total_toes_on_bus : ℕ :=
  num_hoopits * (toes_per_hand_hoopit * hands_per_hoopit) +
  num_neglarts * (toes_per_hand_neglart * hands_per_neglart) +
  num_zentorians * (toes_per_hand_zentorian * hands_per_zentorian)

-- Theorem stating the number of toes on the bus
theorem popton_school_bus_total_toes : total_toes_on_bus = 284 :=
by
  sorry

end popton_school_bus_total_toes_l272_272057


namespace induction_proof_l272_272215

-- Given conditions and definitions
def plane_parts (n : ℕ) : ℕ := 1 + n * (n + 1) / 2

-- The induction hypothesis for k ≥ 2
def induction_step (k : ℕ) (h : 2 ≤ k) : Prop :=
  plane_parts (k + 1) - plane_parts k = k + 1

-- The complete statement we want to prove
theorem induction_proof (k : ℕ) (h : 2 ≤ k) : induction_step k h := by
  sorry

end induction_proof_l272_272215


namespace additional_charge_per_international_letter_l272_272310

-- Definitions based on conditions
def standard_postage_per_letter : ℕ := 108
def num_international_letters : ℕ := 2
def total_cost : ℕ := 460
def num_letters : ℕ := 4

-- Theorem stating the question
theorem additional_charge_per_international_letter :
  (total_cost - (num_letters * standard_postage_per_letter)) / num_international_letters = 14 :=
by
  sorry

end additional_charge_per_international_letter_l272_272310


namespace find_x_l272_272006

theorem find_x (x : ℤ) (A : Set ℤ) (B : Set ℤ) (hA : A = {1, 4, x}) (hB : B = {1, 2 * x, x ^ 2}) (hinter : A ∩ B = {4, 1}) : x = -2 :=
sorry

end find_x_l272_272006


namespace inequality_solution_l272_272548

theorem inequality_solution (x : ℝ) :
  (x - 2 > 1) ∧ (-2 * x ≤ 4) ↔ (x > 3) :=
by
  sorry

end inequality_solution_l272_272548


namespace range_of_values_l272_272615

variable {f : ℝ → ℝ}

-- Conditions and given data
def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x) = f (-x)

def is_monotone_on_nonneg (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x → x ≤ y → f (x) ≤ f (y)

def condition (f : ℝ → ℝ) (a : ℝ) : Prop :=
  f (Real.log a / Real.log 2) + f (-Real.log a / Real.log 2) ≤ 2 * f (1)

-- The goal
theorem range_of_values (h1 : is_even f) (h2 : is_monotone_on_nonneg f) (a : ℝ) (h3 : condition f a) :
  1/2 ≤ a ∧ a ≤ 2 :=
sorry

end range_of_values_l272_272615


namespace equiv_proof_problem_l272_272815

theorem equiv_proof_problem (b c : ℝ) (h1 : b ≠ 1 ∨ c ≠ 1) (h2 : ∃ n : ℝ, b = 1 + n ∧ c = 1 + 2 * n) (h3 : b * 1 = c * c) : 
  100 * (b - c) = 75 := 
by sorry

end equiv_proof_problem_l272_272815


namespace sequence_solution_exists_l272_272049

noncomputable def math_problem (a : ℕ → ℝ) : Prop :=
  ∀ n < 1990, a n > 0 ∧ a 1990 < 0

theorem sequence_solution_exists {a0 c : ℝ} (h_a0 : a0 > 0) (h_c : c > 0) :
  ∃ (a : ℕ → ℝ),
    a 0 = a0 ∧
    (∀ n, a (n + 1) = (a n + c) / (1 - a n * c)) ∧
    math_problem a :=
by
  sorry

end sequence_solution_exists_l272_272049


namespace cards_arrangement_count_is_10_l272_272997

-- Define the problem in Lean statement terms
def valid_arrangements_count : ℕ :=
  -- number of arrangements of seven cards where one card can be removed 
  -- leaving the remaining six cards in either ascending or descending order
  10

-- Theorem stating that the number of valid arrangements is 10
theorem cards_arrangement_count_is_10 : valid_arrangements_count = 10 :=
by
  -- Proof is omitted (the explanation above corresponds to the omitted proof details)
  sorry

end cards_arrangement_count_is_10_l272_272997


namespace evaluate_s_squared_plus_c_squared_l272_272047

variable {x y : ℝ}

theorem evaluate_s_squared_plus_c_squared (r : ℝ) (h_r_def : r = Real.sqrt (x^2 + y^2))
                                          (s : ℝ) (h_s_def : s = y / r)
                                          (c : ℝ) (h_c_def : c = x / r) :
  s^2 + c^2 = 1 :=
sorry

end evaluate_s_squared_plus_c_squared_l272_272047


namespace line_through_PQ_l272_272787

theorem line_through_PQ (x y : ℝ) (P Q : ℝ × ℝ)
  (hP : P = (3, 2)) (hQ : Q = (1, 4))
  (h_line : ∀ t, (x, y) = (1 - t) • P + t • Q):
  y = x - 2 :=
by
  have h1 : P = ((3 : ℝ), (2 : ℝ)) := hP
  have h2 : Q = ((1 : ℝ), (4 : ℝ)) := hQ
  sorry

end line_through_PQ_l272_272787


namespace ages_correct_in_2018_l272_272960

-- Define the initial ages in the year 2000
def age_marianne_2000 : ℕ := 20
def age_bella_2000 : ℕ := 8
def age_carmen_2000 : ℕ := 15

-- Define the birth year of Elli
def birth_year_elli : ℕ := 2003

-- Define the target year when Bella turns 18
def year_bella_turns_18 : ℕ := 2000 + 18

-- Define the ages to be proven
def age_marianne_2018 : ℕ := 30
def age_carmen_2018 : ℕ := 33
def age_elli_2018 : ℕ := 15

theorem ages_correct_in_2018 :
  age_marianne_2018 = age_marianne_2000 + (year_bella_turns_18 - 2000) ∧
  age_carmen_2018 = age_carmen_2000 + (year_bella_turns_18 - 2000) ∧
  age_elli_2018 = year_bella_turns_18 - birth_year_elli :=
by 
  -- The proof would go here
  sorry

end ages_correct_in_2018_l272_272960


namespace sum_third_three_l272_272030

variables {a : ℕ → ℤ}

-- Define the properties of the arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∀ n, a (n + 2) - a (n + 1) = a (n + 1) - a n

-- Given conditions
axiom sum_first_three : a 1 + a 2 + a 3 = 9
axiom sum_second_three : a 4 + a 5 + a 6 = 27
axiom arithmetic_seq : is_arithmetic_sequence a

-- The proof goal
theorem sum_third_three : a 7 + a 8 + a 9 = 45 :=
by
  sorry  -- Proof is omitted here

end sum_third_three_l272_272030


namespace dice_sum_24_probability_l272_272767

noncomputable def probability_sum_24 : ℚ :=
  let prob_single_six := (1 : ℚ) / 6 in
  prob_single_six ^ 4

theorem dice_sum_24_probability :
  probability_sum_24 = 1 / 1296 :=
by
  sorry

end dice_sum_24_probability_l272_272767


namespace fourth_term_of_geometric_sequence_l272_272226

noncomputable def geometric_sequence (a r : ℝ) (n : ℕ) :=
  a * r ^ (n - 1)

theorem fourth_term_of_geometric_sequence 
  (a : ℝ) (r : ℝ) (ar5_eq : a * r ^ 5 = 32) 
  (a_eq : a = 81) :
  geometric_sequence a r 4 = 24 := 
by 
  sorry

end fourth_term_of_geometric_sequence_l272_272226


namespace find_m_value_l272_272491

def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

theorem find_m_value (m : ℝ) 
  (h : dot_product (2 * m - 1, 3) (1, -1) = 2) : 
  m = 3 := by
  sorry

end find_m_value_l272_272491


namespace noah_holidays_l272_272211

theorem noah_holidays (holidays_per_month : ℕ) (months_in_year : ℕ) (holidays_total : ℕ) 
  (h1 : holidays_per_month = 3) (h2 : months_in_year = 12) (h3 : holidays_total = holidays_per_month * months_in_year) : 
  holidays_total = 36 := 
by
  sorry

end noah_holidays_l272_272211


namespace second_largest_div_second_smallest_l272_272411

theorem second_largest_div_second_smallest : 
  let a := 10
  let b := 11
  let c := 12
  ∃ second_smallest second_largest, 
    second_smallest = b ∧ second_largest = b ∧ second_largest / second_smallest = 1 := 
by
  let a := 10
  let b := 11
  let c := 12
  use b
  use b
  exact ⟨rfl, rfl, rfl⟩

end second_largest_div_second_smallest_l272_272411


namespace ratio_of_radii_l272_272959

theorem ratio_of_radii (a b c : ℝ) (h1 : π * c^2 - π * a^2 = 4 * π * a^2) (h2 : π * b^2 = (π * a^2 + π * c^2) / 2) :
  a / c = 1 / Real.sqrt 5 := by
  sorry

end ratio_of_radii_l272_272959


namespace rectangle_area_l272_272680

theorem rectangle_area (y : ℝ) (h : y > 0) 
    (h_area : ∃ (E F G H : ℝ × ℝ), 
        E = (0, 0) ∧ 
        F = (0, 5) ∧ 
        G = (y, 5) ∧ 
        H = (y, 0) ∧ 
        5 * y = 45) : 
    y = 9 := 
by
    sorry

end rectangle_area_l272_272680


namespace mary_initial_triangles_l272_272209

theorem mary_initial_triangles (s t : ℕ) (h1 : s + t = 10) (h2 : 4 * s + 3 * t = 36) : t = 4 :=
by
  sorry

end mary_initial_triangles_l272_272209


namespace min_students_l272_272951

theorem min_students (b g : ℕ) (hb : 1 ≤ b) (hg : 1 ≤ g)
    (h1 : b = (4/3) * g) 
    (h2 : (1/2) * b = 2 * ((1/3) * g)) 
    : b + g = 7 :=
by sorry

end min_students_l272_272951


namespace find_x_from_expression_l272_272489

theorem find_x_from_expression
  (y : ℚ)
  (h1 : y = -3/2)
  (h2 : -2 * (x : ℚ) - y^2 = 0.25) : 
  x = -5/4 := 
by 
  sorry

end find_x_from_expression_l272_272489


namespace total_mangoes_calculation_l272_272855

-- Define conditions as constants
def boxes : ℕ := 36
def dozen_to_mangoes : ℕ := 12
def dozens_per_box : ℕ := 10

-- Define the expected correct answer for the total mangoes
def expected_total_mangoes : ℕ := 4320

-- Lean statement to prove
theorem total_mangoes_calculation :
  dozens_per_box * dozen_to_mangoes * boxes = expected_total_mangoes :=
by sorry

end total_mangoes_calculation_l272_272855


namespace train_still_there_when_susan_arrives_l272_272734

-- Define the conditions and primary question
def time_between_1_and_2 (t : ℝ) : Prop := 0 ≤ t ∧ t ≤ 60

def train_arrival := {t : ℝ // time_between_1_and_2 t}
def susan_arrival := {t : ℝ // time_between_1_and_2 t}

def train_present (train : train_arrival) (susan : susan_arrival) : Prop :=
  susan.val ≥ train.val ∧ susan.val ≤ (train.val + 30)

-- Define the probability calculation
noncomputable def probability_train_present : ℝ :=
  (30 * 30 + (30 * (60 - 30) * 2) / 2) / (60 * 60)

theorem train_still_there_when_susan_arrives :
  probability_train_present = 1 / 2 :=
sorry

end train_still_there_when_susan_arrives_l272_272734


namespace international_postage_surcharge_l272_272307

theorem international_postage_surcharge 
  (n_letters : ℕ) 
  (std_postage_per_letter : ℚ) 
  (n_international : ℕ) 
  (total_cost : ℚ) 
  (cents_per_dollar : ℚ) 
  (std_total_cost : ℚ) 
  : 
  n_letters = 4 →
  std_postage_per_letter = 108 / 100 →
  n_international = 2 →
  total_cost = 460 / 100 →
  cents_per_dollar = 100 →
  std_total_cost = n_letters * std_postage_per_letter →
  (total_cost - std_total_cost) / n_international * cents_per_dollar = 14 := 
sorry

end international_postage_surcharge_l272_272307


namespace number_of_ways_to_draw_balls_l272_272100

def draw_balls : ℕ :=
  let balls := 15
  let draws := 4
  balls * (balls - 1) * (balls - 2) * (balls - 3)

theorem number_of_ways_to_draw_balls
  : draw_balls = 32760 :=
by
  sorry

end number_of_ways_to_draw_balls_l272_272100


namespace skate_cost_l272_272894

/- Define the initial conditions as Lean definitions -/
def admission_cost : ℕ := 5
def rental_cost : ℕ := 250 / 100  -- 2.50 dollars in cents for integer representation
def visits : ℕ := 26

/- Define the cost calculation as a Lean definition -/
def total_rental_cost (rental_cost : ℕ) (visits : ℕ) : ℕ := rental_cost * visits

/- Statement of the problem in Lean proof form -/
theorem skate_cost (C : ℕ) (h : total_rental_cost rental_cost visits = C) : C = 65 :=
by
  sorry

end skate_cost_l272_272894


namespace stratified_sampling_l272_272104

theorem stratified_sampling :
  let total_employees := 150
  let middle_managers := 30
  let senior_managers := 10
  let selected_employees := 30
  let selection_probability := selected_employees / total_employees
  let selected_middle_managers := middle_managers * selection_probability
  let selected_senior_managers := senior_managers * selection_probability
  selected_middle_managers = 6 ∧ selected_senior_managers = 2 :=
by
  sorry

end stratified_sampling_l272_272104


namespace count_divisible_by_90_l272_272935

theorem count_divisible_by_90 : 
  ∃ n, n = 10 ∧ (∀ k, 1000 ≤ k ∧ k < 10000 ∧ k % 100 = 90 ∧ k % 90 = 0 → n = 10) :=
begin
  sorry
end

end count_divisible_by_90_l272_272935


namespace measure_of_angle_B_l272_272033

theorem measure_of_angle_B (a b c R : ℝ) (A B C : ℝ)
  (h1 : a = 2 * R * Real.sin A)
  (h2 : b = 2 * R * Real.sin B)
  (h3 : c = 2 * R * Real.sin C)
  (h4 : 2 * R * (Real.sin A ^ 2 - Real.sin B ^ 2) = (Real.sqrt 2 * a - c) * Real.sin C) :
  B = Real.pi / 4 :=
by
  sorry

end measure_of_angle_B_l272_272033


namespace original_five_digit_number_l272_272278

theorem original_five_digit_number :
  ∃ N y x : ℕ, (N = 10 * y + x) ∧ (N + y = 54321) ∧ (N = 49383) :=
by
  -- The proof script goes here
  sorry

end original_five_digit_number_l272_272278


namespace prob_score_5_points_is_three_over_eight_l272_272184

noncomputable def probability_of_scoring_5_points : ℚ :=
  let total_events := 2^3
  let favorable_events := 3 -- Calculated from combinatorial logic.
  favorable_events / total_events

theorem prob_score_5_points_is_three_over_eight :
  probability_of_scoring_5_points = 3 / 8 :=
by
  sorry

end prob_score_5_points_is_three_over_eight_l272_272184


namespace max_candy_leftover_l272_272496

theorem max_candy_leftover (x : ℕ) : (∃ k : ℕ, x = 12 * k + 11) → (x % 12 = 11) :=
by
  sorry

end max_candy_leftover_l272_272496


namespace extreme_value_at_x_eq_one_l272_272228

noncomputable def f (x a b: ℝ) : ℝ := x^3 - a * x^2 + b * x + a^2
noncomputable def f_prime (x a b: ℝ) : ℝ := 3 * x^2 - 2 * a * x + b

theorem extreme_value_at_x_eq_one (a b : ℝ) (h_prime : f_prime 1 a b = 0) (h_value : f 1 a b = 10) : a = -4 :=
by 
  sorry -- proof goes here

end extreme_value_at_x_eq_one_l272_272228


namespace number_of_valid_arrangements_l272_272994

open Finset

-- We define the condition that a list is sorted in ascending order
def is_ascending (l : List ℕ) : Prop :=
  l = List.sort (≤) l

-- We define the condition that a list is sorted in descending order
def is_descending (l : List ℕ) : Prop :=
  l = List.sort (≥) l

def cards := Finset.range 7
def arrangements := cards.to_list.permutations

-- Define the function to check if a list of numbers (cards) 
-- can have one element removed to form an ascending or descending list
def valid_arrangement (l : List ℕ) : Prop :=
  ∃ (x : ℕ), (l.erase x).is_ascending ∨ (l.erase x).is_descending

-- Define the final theorem
theorem number_of_valid_arrangements : finset.card (arrangements.filter valid_arrangement) = 72 :=
by
  sorry

end number_of_valid_arrangements_l272_272994


namespace value_of_k_l272_272021

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

noncomputable def f (k x : ℝ) : ℝ := k * x^2 + (k - 1) * x + 2

theorem value_of_k (k : ℝ) :
  is_even_function (f k) → k = 1 :=
by {
  sorry
}

end value_of_k_l272_272021


namespace integer_cube_less_than_triple_unique_l272_272700

theorem integer_cube_less_than_triple_unique (x : ℤ) (h : x^3 < 3*x) : x = 1 :=
sorry

end integer_cube_less_than_triple_unique_l272_272700


namespace jeremy_sticker_distribution_l272_272363

def number_of_ways_to_distribute_stickers (total_stickers sheets : ℕ) : ℕ :=
  (Nat.choose (total_stickers - 1) (sheets - 1))

theorem jeremy_sticker_distribution : number_of_ways_to_distribute_stickers 10 3 = 36 :=
by
  sorry

end jeremy_sticker_distribution_l272_272363


namespace negation_proposition_l272_272233

theorem negation_proposition :
  (¬ (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 - 3 * x + 2 ≤ 0)) =
  (∃ x : ℝ, 1 ≤ x ∧ x ≤ 2 ∧ x^2 - 3 * x + 2 > 0) := 
sorry

end negation_proposition_l272_272233


namespace ball_bounces_height_l272_272879

theorem ball_bounces_height : ∃ k : ℕ, ∀ n ≥ k, 800 * (2 / 3: ℝ) ^ n < 10 :=
by
  sorry

end ball_bounces_height_l272_272879


namespace intersection_points_eq_one_l272_272122

-- Definitions for the equations of the circles
def circle1 (x y : ℝ) : ℝ := x^2 + (y - 3)^2
def circle2 (x y : ℝ) : ℝ := x^2 + (y + 2)^2

-- The proof problem statement
theorem intersection_points_eq_one : 
  ∃ p : ℝ × ℝ, (circle1 p.1 p.2 = 9) ∧ (circle2 p.1 p.2 = 4) ∧
  (∀ q : ℝ × ℝ, (circle1 q.1 q.2 = 9) ∧ (circle2 q.1 q.2 = 4) → q = p) :=
sorry

end intersection_points_eq_one_l272_272122


namespace max_distance_increases_l272_272669

noncomputable def largest_n_for_rearrangement (C : ℕ) (marked_points : ℕ) : ℕ :=
  670

theorem max_distance_increases (C : ℕ) (marked_points : ℕ) (n : ℕ) (dist : ℕ → ℕ → ℕ) :
  ∀ i j, i < marked_points → j < marked_points →
    dist i j ≤ n → 
    (∃ rearrangement : ℕ → ℕ, 
    ∀ i j, i < marked_points → j < marked_points → 
      dist (rearrangement i) (rearrangement j) > dist i j) → 
    n ≤ largest_n_for_rearrangement C marked_points := 
by
  sorry

end max_distance_increases_l272_272669


namespace probability_of_rolling_prime_is_half_l272_272704

def is_prime (n : ℕ) : Prop :=
  n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7

def total_outcomes : ℕ := 8

def successful_outcomes : ℕ := 4 -- prime numbers between 1 and 8 are 2, 3, 5, and 7

def probability_of_rolling_prime : ℚ :=
  successful_outcomes / total_outcomes

theorem probability_of_rolling_prime_is_half : probability_of_rolling_prime = 1 / 2 :=
  sorry

end probability_of_rolling_prime_is_half_l272_272704


namespace value_of_5_S_3_l272_272311

def operation_S (a b : ℝ) : ℝ := 4 * a + 6 * b - 2 * a * b

theorem value_of_5_S_3 : operation_S 5 3 = 8 :=
by
  sorry

end value_of_5_S_3_l272_272311


namespace value_of_y_l272_272796

theorem value_of_y (x y z : ℕ) (h1 : 3 * x = 3 / 4 * y) (h2 : x + z = 24) (h3 : z = 8) : y = 64 :=
by
  -- Proof omitted
  sorry

end value_of_y_l272_272796


namespace chameleons_cannot_all_turn_to_single_color_l272_272212

theorem chameleons_cannot_all_turn_to_single_color
  (W : ℕ) (B : ℕ)
  (hW : W = 20)
  (hB : B = 25)
  (h_interaction: ∀ t : ℕ, ∃ W' B' : ℕ,
    W' + B' = W + B ∧
    (W - B) % 3 = (W' - B') % 3) :
  ∀ t : ℕ, (W - B) % 3 ≠ 0 :=
by
  sorry

end chameleons_cannot_all_turn_to_single_color_l272_272212


namespace proof_l272_272255

-- Define the equation and its conditions
def equation (x m : ℤ) : Prop := (3 * x - 1) / 2 + m = 3

-- Part 1: Prove that for m = 5, the corresponding x must be 1
def part1 : Prop :=
  ∃ x : ℤ, equation x 5 ∧ x = 1

-- Part 2: Prove that if the equation has a positive integer solution, the positive integer m must be 2
def part2 : Prop :=
  ∃ m x : ℤ, m > 0 ∧ x > 0 ∧ equation x m ∧ m = 2

theorem proof : part1 ∧ part2 :=
  by
    sorry

end proof_l272_272255


namespace p_sq_plus_q_sq_l272_272975

theorem p_sq_plus_q_sq (p q : ℝ) (h1 : p * q = 12) (h2 : p + q = 8) : p^2 + q^2 = 40 :=
by
  sorry

end p_sq_plus_q_sq_l272_272975


namespace cooler_capacity_l272_272519

theorem cooler_capacity (C : ℝ) (h1 : 3.25 * C = 325) : C = 100 :=
sorry

end cooler_capacity_l272_272519


namespace min_value_x_y_l272_272052

open Real

theorem min_value_x_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  (x + 1/y) * (x + 1/y - 1024) + (y + 1/x) * (y + 1/x - 1024) ≥ -524288 :=
by 
  sorry

end min_value_x_y_l272_272052


namespace repeating_decimal_product_l272_272743

theorem repeating_decimal_product :
  let x : ℚ := 456 / 999
  in (x * 11 = 1672 / 333) :=
by
  sorry

end repeating_decimal_product_l272_272743


namespace theta_plus_2phi_eq_pi_div_4_l272_272925

noncomputable def theta (θ : ℝ) (φ : ℝ) : Prop := 
  ((Real.tan θ = 5 / 12) ∧ 
   (Real.sin φ = 1 / 2) ∧ 
   (0 < θ ∧ θ < Real.pi / 2) ∧ 
   (0 < φ ∧ φ < Real.pi / 2)  )

theorem theta_plus_2phi_eq_pi_div_4 (θ φ : ℝ) (h : theta θ φ) : 
    θ + 2 * φ = Real.pi / 4 :=
by 
  sorry

end theta_plus_2phi_eq_pi_div_4_l272_272925


namespace cross_product_scaled_v_and_w_l272_272129

-- Assume the vectors and their scalar multiple
def v : ℝ × ℝ × ℝ := (3, 1, 4)
def w : ℝ × ℝ × ℝ := (-2, 2, -3)
def v_scaled : ℝ × ℝ × ℝ := (6, 2, 8)

-- Define the cross product function
def cross_product (a b : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (a.2.1 * b.2.2 - a.2.2 * b.2.1,
   a.1 * b.2.2 - a.2.2 * b.1,
   a.1 * b.2.1 - a.2.1 * b.1)

theorem cross_product_scaled_v_and_w :
  cross_product v_scaled w = (-22, -2, 16) :=
by
  sorry

end cross_product_scaled_v_and_w_l272_272129


namespace common_ratio_of_arithmetic_sequence_l272_272472

variable {α : Type} [LinearOrderedField α]

def is_arithmetic_sequence (a : ℕ → α) : Prop :=
  ∃ d : α, ∀ n : ℕ, a (n + 1) = a n + d

theorem common_ratio_of_arithmetic_sequence (a : ℕ → α) (q : α)
  (h1 : is_arithmetic_sequence a)
  (h2 : ∀ n : ℕ, 2 * (a n + a (n + 2)) = 5 * a (n + 1))
  (h3 : a 1 > 0)
  (h4 : ∀ n : ℕ, a n < a (n + 1)) :
  q = 2 := 
sorry

end common_ratio_of_arithmetic_sequence_l272_272472


namespace min_value_of_expression_l272_272660

theorem min_value_of_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x + y + z = 5) : 
  (1/x + 4/y + 9/z) >= 36/5 :=
sorry

end min_value_of_expression_l272_272660


namespace find_b_l272_272784

-- Define the quadratic equation
def quadratic_eq (b : ℝ) (x : ℝ) : ℝ :=
  x^2 + b * x - 15

-- Prove that b = 49/8 given -8 is a solution to the quadratic equation
theorem find_b (b : ℝ) : quadratic_eq b (-8) = 0 -> b = 49 / 8 :=
by
  intro h
  sorry

end find_b_l272_272784


namespace directrix_of_given_parabola_l272_272143

noncomputable def parabola_directrix (y : ℝ) : Prop :=
  let focus_x : ℝ := -1
  let point_on_parabola : ℝ × ℝ := (-1 / 4 * y^2, y)
  let PF_sq := (point_on_parabola.1 - focus_x)^2 + (point_on_parabola.2)^2
  let PQ_sq := (point_on_parabola.1 - 1)^2
  PF_sq = PQ_sq

theorem directrix_of_given_parabola : parabola_directrix y :=
sorry

end directrix_of_given_parabola_l272_272143


namespace complement_U_A_intersection_A_B_complement_U_intersection_A_B_complement_A_intersection_B_l272_272009

open Set -- Open the Set namespace for convenience

-- Define the universal set U, and sets A and B
def U : Set ℝ := univ
def A : Set ℝ := {x | -2 < x ∧ x < 3}
def B : Set ℝ := {x | -3 < x ∧ x ≤ 3}

-- Proof statements
theorem complement_U_A : U \ A = {x | x ≥ 3 ∨ x ≤ -2} :=
by sorry

theorem intersection_A_B : A ∩ B = {x | -2 < x ∧ x < 3} :=
by sorry

theorem complement_U_intersection_A_B : U \ (A ∩ B) = {x | x ≥ 3 ∨ x ≤ -2} :=
by sorry

theorem complement_A_intersection_B : (U \ A) ∩ B = {x | (-3 < x ∧ x ≤ -2) ∨ x = 3} :=
by sorry

end complement_U_A_intersection_A_B_complement_U_intersection_A_B_complement_A_intersection_B_l272_272009


namespace operation_on_b_l272_272068

theorem operation_on_b (t b0 b1 : ℝ) (h : t * b1^4 = 16 * t * b0^4) : b1 = 2 * b0 :=
by
  sorry

end operation_on_b_l272_272068


namespace marlon_keeps_4_lollipops_l272_272208

def initial_lollipops : ℕ := 42
def fraction_given_to_emily : ℚ := 2 / 3
def lollipops_given_to_lou : ℕ := 10

theorem marlon_keeps_4_lollipops :
  let lollipops_given_to_emily := fraction_given_to_emily * initial_lollipops
  let lollipops_after_emily := initial_lollipops - lollipops_given_to_emily
  let marlon_keeps := lollipops_after_emily - lollipops_given_to_lou
  marlon_keeps = 4 :=
by
  sorry

end marlon_keeps_4_lollipops_l272_272208


namespace graph_contains_k_star_or_k_matching_l272_272370

open SimpleGraph

/-- Let G be a graph and k be a positive integer. If G has strictly more than 2(k-1)^2 edges, 
then G contains a k-star or a k-matching. -/
theorem graph_contains_k_star_or_k_matching 
  (G : SimpleGraph V) (k : ℕ)
  (h1 : 0 < k)
  (h2 : G.edge_finset.card > 2 * (k-1)^2) :
  ∃ (S : Finset (Sym2 V)), 
    ((∃ v : V, S.card = k ∧ ∀ e ∈ S, v ∈ e) ∨ (S.card = k ∧ ∀ e f ∈ S, e ≠ f ∧ Sym2.card e ∩ f = 0)) :=
sorry

end graph_contains_k_star_or_k_matching_l272_272370


namespace class_tree_total_l272_272095

theorem class_tree_total
  (trees_A : ℕ)
  (trees_B : ℕ)
  (hA : trees_A = 8)
  (hB : trees_B = 7)
  : trees_A + trees_B = 15 := 
by
  sorry

end class_tree_total_l272_272095


namespace y1_greater_than_y2_l272_272182

-- Define the function and points
def parabola (x : ℝ) (m : ℝ) : ℝ := x^2 - 4 * x + m

-- Define the points A and B on the parabola
def A_y1 (m : ℝ) : ℝ := parabola 0 m
def B_y2 (m : ℝ) : ℝ := parabola 1 m

-- Theorem statement
theorem y1_greater_than_y2 (m : ℝ) : A_y1 m > B_y2 m := 
  sorry

end y1_greater_than_y2_l272_272182


namespace mr_caiden_payment_l272_272378

theorem mr_caiden_payment (total_feet_needed : ℕ) (cost_per_foot : ℕ) (free_feet_supplied : ℕ) : 
  total_feet_needed = 300 → cost_per_foot = 8 → free_feet_supplied = 250 → 
  (total_feet_needed - free_feet_supplied) * cost_per_foot = 400 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num

end mr_caiden_payment_l272_272378


namespace find_a_value_l272_272631

theorem find_a_value
    (a : ℝ)
    (line : ∀ (x y : ℝ), 3 * x + y + a = 0)
    (circle : ∀ (x y : ℝ), x^2 + y^2 + 2 * x - 4 * y = 0) :
    a = 1 := sorry

end find_a_value_l272_272631


namespace pizza_slices_l272_272878

theorem pizza_slices (total_slices pepperoni_slices mushroom_slices : ℕ) 
  (h_total : total_slices = 24)
  (h_pepperoni : pepperoni_slices = 15)
  (h_mushrooms : mushroom_slices = 16)
  (h_at_least_one : total_slices = pepperoni_slices + mushroom_slices - both_slices)
  : both_slices = 7 :=
by
  have h1 : total_slices = 24 := h_total
  have h2 : pepperoni_slices = 15 := h_pepperoni
  have h3 : mushroom_slices = 16 := h_mushrooms
  have h4 : total_slices = 24 := by sorry
  sorry

end pizza_slices_l272_272878


namespace greatest_remainder_le_11_l272_272499

noncomputable def greatest_remainder (x : ℕ) : ℕ := x % 12

theorem greatest_remainder_le_11 (x : ℕ) (h : x % 12 ≠ 0) : greatest_remainder x = 11 :=
by
  sorry

end greatest_remainder_le_11_l272_272499


namespace sin_cos_pi_minus_two_alpha_l272_272917

theorem sin_cos_pi_minus_two_alpha (α : ℝ) (h : tan α = 2 / 3) : 
  sin (2 * α) - cos (π - 2 * α) = 17 / 13 :=
by
  sorry

end sin_cos_pi_minus_two_alpha_l272_272917


namespace find_original_number_l272_272282

open Int

theorem find_original_number (N y x : ℕ) (h1 : N = 10 * y + x) (h2 : N + y = 54321) (h3 : x = 54321 % 11) (h4 : y = 4938) : N = 49383 := by
  sorry

end find_original_number_l272_272282


namespace dice_sum_24_l272_272761

noncomputable def probability_of_sum_24 : ℚ :=
  let die_probability := (1 : ℚ) / 6
  in die_probability ^ 4

theorem dice_sum_24 :
  ∑ x in {x | x ∈ {1, 2, 3, 4, 5, 6} ∧ x = 6} = 24 → probability_of_sum_24 = 1 / 1296 :=
sorry

end dice_sum_24_l272_272761


namespace number_of_valid_arrangements_l272_272992

def is_ascending (l : List ℕ) : Prop :=
  ∀ i j, i < j → l.nth i ≤ l.nth j

def is_descending (l : List ℕ) : Prop :=
  ∀ i j, i < j → l.nth i ≥ l.nth j

def remove_one_is_ordered (l : List ℕ) : Prop :=
  ∃ (i : ℕ), (is_ascending (l.removeNth i) ∨ is_descending (l.removeNth i))

def valid_arrangements_count (cards : List ℕ) : ℕ :=
  -- counting the number of valid arrangements
  if (cards.length = 7
        ∧ ∀ i, i ∈ cards → 1 ≤ i ∧ i ≤ 7 ∧ (remove_one_is_ordered cards)) then 4 else 0

theorem number_of_valid_arrangements :
  valid_arrangements_count [1,2,3,4,5,6,7] = 4 :=
by sorry

end number_of_valid_arrangements_l272_272992


namespace possible_values_of_m_l272_272924

-- Defining sets A and B based on the given conditions
def set_A : Set ℝ := { x | x^2 - 2 * x - 3 = 0 }
def set_B (m : ℝ) : Set ℝ := { x | m * x + 1 = 0 }

-- The main theorem statement
theorem possible_values_of_m (m : ℝ) :
  (set_A ∪ set_B m = set_A) ↔ (m = 0 ∨ m = -1 / 3 ∨ m = 1) := by
  sorry

end possible_values_of_m_l272_272924


namespace parabola_intercept_sum_l272_272230

theorem parabola_intercept_sum : 
  let d := 4
  let e := (9 + Real.sqrt 33) / 6
  let f := (9 - Real.sqrt 33) / 6
  d + e + f = 7 :=
by 
  sorry

end parabola_intercept_sum_l272_272230


namespace find_x_l272_272169

variables {a b : EuclideanSpace ℝ (Fin 2)} {x : ℝ}

theorem find_x (h1 : ‖a + b‖ = 1) (h2 : ‖a - b‖ = x) (h3 : inner a b = -(3 / 8) * x) : x = 2 ∨ x = -(1 / 2) :=
sorry

end find_x_l272_272169


namespace expected_fixed_balls_after_swaps_l272_272856

/-- 
The expected number of balls occupying their original positions after Chris, Silva, and Alex 
each make one swap in a circular arrangement of six balls is 2.0. 
-/
theorem expected_fixed_balls_after_swaps : 
  let balls := {1, 2, 3, 4, 5, 6} 
  let swap := λ (b : Fin 6) (i j: Fin 6), if j == b then i else if i == b then j else b 
  ∀ chris_swap silva_swap alex_swap : Fin 6 → Fin 6,
  expected_value (λ b, if swap (swap (swap b chris_swap) silva_swap) alex_swap = b then 1 else 0) = 2 := 
by sorry

end expected_fixed_balls_after_swaps_l272_272856


namespace Jean_had_41_candies_at_first_l272_272362

-- Let total_candies be the initial number of candies Jean had
variable (total_candies : ℕ)
-- Jean gave 18 pieces to a friend
def given_away := 18
-- Jean ate 7 pieces
def eaten := 7
-- Jean has 16 pieces left now
def remaining := 16

-- Calculate the total number of candies initially
def candy_initial (total_candies given_away eaten remaining : ℕ) : Prop :=
  total_candies = remaining + (given_away + eaten)

-- Prove that Jean had 41 pieces of candy initially
theorem Jean_had_41_candies_at_first : candy_initial 41 given_away eaten remaining :=
by
  -- Skipping the proof for now
  sorry

end Jean_had_41_candies_at_first_l272_272362


namespace rectangular_prism_volume_l272_272020

theorem rectangular_prism_volume :
  ∀ (l w h : ℕ), 
  l = 2 * w → 
  w = 2 * h → 
  4 * (l + w + h) = 56 → 
  l * w * h = 64 := 
by
  intros l w h h_l_eq_2w h_w_eq_2h h_edge_len_eq_56
  sorry -- proof not provided

end rectangular_prism_volume_l272_272020


namespace arithmetic_geom_seq_l272_272611

variable {a_n : ℕ → ℝ}
variable {d a_1 : ℝ}
variable (h_seq : ∀ n, a_n n = a_1 + (n-1) * d)
variable (d_ne_zero : d ≠ 0)
variable (a_1_ne_zero : a_1 ≠ 0)
variable (geo_seq : (a_1 + d)^2 = a_1 * (a_1 + 3 * d))

theorem arithmetic_geom_seq :
  (a_1 + a_n 14) / a_n 3 = 5 := by
  sorry

end arithmetic_geom_seq_l272_272611


namespace race_distance_l272_272713

theorem race_distance (d v_A v_B v_C : ℝ) (h1 : d / v_A = (d - 20) / v_B)
  (h2 : d / v_B = (d - 10) / v_C) (h3 : d / v_A = (d - 28) / v_C) : d = 100 :=
by
  sorry

end race_distance_l272_272713


namespace number_of_numbers_in_last_group_l272_272241

theorem number_of_numbers_in_last_group :
  ∃ n : ℕ, (60 * 13) = (57 * 6) + 50 + (61 * n) ∧ n = 6 :=
sorry

end number_of_numbers_in_last_group_l272_272241


namespace floor_div_add_floor_div_succ_eq_l272_272821

theorem floor_div_add_floor_div_succ_eq (n : ℤ) : 
  (⌊(n : ℝ)/2⌋ + ⌊(n + 1 : ℝ)/2⌋ : ℤ) = n := 
sorry

end floor_div_add_floor_div_succ_eq_l272_272821


namespace parabola_directrix_l272_272134

theorem parabola_directrix : 
  ∃ d : ℝ, (∀ (y : ℝ), x = - (1 / 4) * y^2 -> 
  ( - (1 / 4) * y^2 = d)) ∧ d = 1 :=
by
  sorry

end parabola_directrix_l272_272134


namespace mark_spends_47_l272_272086

def apple_price : ℕ := 2
def apple_quantity : ℕ := 4
def bread_price : ℕ := 3
def bread_quantity : ℕ := 5
def cheese_price : ℕ := 6
def cheese_quantity : ℕ := 3
def cereal_price : ℕ := 5
def cereal_quantity : ℕ := 4
def coupon : ℕ := 10

def calculate_total_cost (apple_price apple_quantity bread_price bread_quantity cheese_price cheese_quantity cereal_price cereal_quantity coupon : ℕ) : ℕ :=
  let apples_cost := apple_price * (apple_quantity / 2)  -- Apply buy-one-get-one-free
  let bread_cost := bread_price * bread_quantity
  let cheese_cost := cheese_price * cheese_quantity
  let cereal_cost := cereal_price * cereal_quantity
  let subtotal := apples_cost + bread_cost + cheese_cost + cereal_cost
  let total_cost := if subtotal > 50 then subtotal - coupon else subtotal
  total_cost

theorem mark_spends_47 : calculate_total_cost apple_price apple_quantity bread_price bread_quantity cheese_price cheese_quantity cereal_price cereal_quantity coupon = 47 :=
  sorry

end mark_spends_47_l272_272086


namespace find_angle_ACE_l272_272919

-- Definitions of points and intersection properties
variables (A B C D E P Q O : Point) (ABCDE : ConvexPentagon A B C D E)
variable (H1 : Intersect BE AC P) 
variable (H2 : Intersect CE AD Q) 
variable (H3 : Intersect AD BE O)

-- Definitions of triangle properties
variables (T1 : IsoscelesTriangleAtVertex B P A 40) -- ∠BPA = 40°
variables (T2 : IsoscelesTriangleAtVertex D Q E 40) -- ∠DQE = 40°
variables (T3 : IsoscelesTriangle P A O)
variables (T4 : IsoscelesTriangle E Q O)

-- The goal statement
theorem find_angle_ACE : (angle ACE = 120 ∨ angle ACE = 75) := 
sorry

end find_angle_ACE_l272_272919


namespace length_of_shorter_angle_trisector_l272_272923

theorem length_of_shorter_angle_trisector (BC AC : ℝ) (h1 : BC = 3) (h2 : AC = 4) :
  let AB := Real.sqrt (BC^2 + AC^2)
  let x := 2 * (12 / (4 * Real.sqrt 3 + 3))
  let PC := 2 * x
  AB = 5 ∧ PC = (32 * Real.sqrt 3 - 24) / 13 :=
by
  sorry

end length_of_shorter_angle_trisector_l272_272923


namespace sequence_event_equivalence_l272_272203

open Set Filter

variables {Ω : Type*} {A : ℕ → Set Ω} -- Sequence of events A_n from Ω

theorem sequence_event_equivalence (A : ℕ → Set Ω) : 
  limsup (λ n, A n) \ liminf (λ n, A n) =
  limsup (λ n, A n \ A (n + 1)) ∧
  limsup (λ n, A (n + 1) \ A n) ∧
  limsup (λ n, symmetricDifference (A n) (A (n + 1))) :=
sorry

end sequence_event_equivalence_l272_272203


namespace first_storm_duration_l272_272555

theorem first_storm_duration
  (x y : ℕ)
  (h1 : 30 * x + 15 * y = 975)
  (h2 : x + y = 45) :
  x = 20 :=
by sorry

end first_storm_duration_l272_272555


namespace find_D_l272_272643

noncomputable def Point : Type := ℝ × ℝ

-- Given points A, B, and C
def A : Point := (-2, 0)
def B : Point := (6, 8)
def C : Point := (8, 6)

-- Condition: AB parallel to DC and AD parallel to BC, which means it is a parallelogram
def is_parallelogram (A B C D : Point) : Prop :=
  ((B.1 - A.1, B.2 - A.2) = (D.1 - C.1, D.2 - C.2)) ∧
  ((C.1 - B.1, C.2 - B.2) = (D.1 - A.1, D.2 - A.2))

-- Proves that with given A, B, and C, D should be (0, -2)
theorem find_D : ∃ D : Point, is_parallelogram A B C D ∧ D = (0, -2) :=
  by sorry

end find_D_l272_272643


namespace find_original_number_l272_272281

open Int

theorem find_original_number (N y x : ℕ) (h1 : N = 10 * y + x) (h2 : N + y = 54321) (h3 : x = 54321 % 11) (h4 : y = 4938) : N = 49383 := by
  sorry

end find_original_number_l272_272281


namespace certain_number_is_166_l272_272439

theorem certain_number_is_166 :
  ∃ x : ℕ, x - 78 =  (4 - 30) + 114 ∧ x = 166 := by
  sorry

end certain_number_is_166_l272_272439


namespace natural_numbers_solution_l272_272126

theorem natural_numbers_solution (a : ℕ) :
  ∃ k n : ℕ, k = 3 * a - 2 ∧ n = 2 * a - 1 ∧ (7 * k + 15 * n - 1) % (3 * k + 4 * n) = 0 :=
sorry

end natural_numbers_solution_l272_272126


namespace triangular_weight_l272_272079

theorem triangular_weight (c t : ℝ) (h1 : c + t = 3 * c) (h2 : 4 * c + t = t + c + 90) : t = 60 := 
by sorry

end triangular_weight_l272_272079


namespace initial_men_checking_exam_papers_l272_272221

theorem initial_men_checking_exam_papers :
  ∀ (M : ℕ),
  (M * 8 * 5 = (1/2 : ℝ) * (2 * 20 * 8)) → M = 4 :=
by
  sorry

end initial_men_checking_exam_papers_l272_272221


namespace largest_possible_sum_l272_272820

theorem largest_possible_sum (a b : ℤ) (h : a^2 - b^2 = 144) : a + b ≤ 72 :=
sorry

end largest_possible_sum_l272_272820


namespace minimum_socks_to_guarantee_20_pairs_l272_272881

-- Definitions and conditions
def red_socks := 120
def green_socks := 100
def blue_socks := 80
def black_socks := 50
def number_of_pairs := 20

-- Statement
theorem minimum_socks_to_guarantee_20_pairs 
  (red_socks green_socks blue_socks black_socks number_of_pairs: ℕ) 
  (h1: red_socks = 120) 
  (h2: green_socks = 100) 
  (h3: blue_socks = 80) 
  (h4: black_socks = 50) 
  (h5: number_of_pairs = 20) : 
  ∃ min_socks, min_socks = 43 := 
by 
  sorry

end minimum_socks_to_guarantee_20_pairs_l272_272881


namespace total_weight_proof_l272_272036
-- Import the entire math library

-- Assume the conditions as given variables
variables (w r s : ℕ)
-- Assign values to the given conditions
def weight_per_rep := 15
def reps_per_set := 10
def number_of_sets := 3

-- Calculate total weight moved
def total_weight_moved := w * r * s

-- The theorem to prove the total weight moved
theorem total_weight_proof : total_weight_moved weight_per_rep reps_per_set number_of_sets = 450 :=
by
  -- Provide the expected result directly, proving the statement
  sorry

end total_weight_proof_l272_272036


namespace probability_divisor_of_12_l272_272720

open Probability

def divisors_of_12 := {1, 2, 3, 4, 6}

theorem probability_divisor_of_12 :
  ∃ (fair_die_roll : ProbabilityMeasure (Fin 6)), 
    P (fun x => x.val + 1 ∈ divisors_of_12) = 5 / 6 := 
by
  sorry

end probability_divisor_of_12_l272_272720


namespace Kira_breakfast_time_l272_272520

theorem Kira_breakfast_time :
  let sausages := 3
  let eggs := 6
  let time_per_sausage := 5
  let time_per_egg := 4
  (sausages * time_per_sausage + eggs * time_per_egg) = 39 :=
by
  sorry

end Kira_breakfast_time_l272_272520


namespace differentiable_function_inequality_l272_272914

theorem differentiable_function_inequality (f : ℝ → ℝ) (h_diff : Differentiable ℝ f) 
  (h_cond : ∀ x : ℝ, (x - 1) * (deriv f x) ≥ 0) : 
  f 0 + f 2 ≥ 2 * (f 1) :=
sorry

end differentiable_function_inequality_l272_272914


namespace original_five_digit_number_l272_272275

theorem original_five_digit_number :
  ∃ N y x : ℕ, (N = 10 * y + x) ∧ (N + y = 54321) ∧ (N = 49383) :=
by
  -- The proof script goes here
  sorry

end original_five_digit_number_l272_272275


namespace total_cookies_l272_272833

def mona_cookies : ℕ := 20
def jasmine_cookies : ℕ := mona_cookies - 5
def rachel_cookies : ℕ := jasmine_cookies + 10

theorem total_cookies : mona_cookies + jasmine_cookies + rachel_cookies = 60 := 
by
  have h1 : jasmine_cookies = 15 := by sorry
  have h2 : rachel_cookies = 25 := by sorry
  have h3 : mona_cookies = 20 := by sorry
  sorry

end total_cookies_l272_272833


namespace speed_of_the_stream_l272_272085

theorem speed_of_the_stream (d v_s : ℝ) :
  (∀ (t_up t_down : ℝ), t_up = d / (57 - v_s) ∧ t_down = d / (57 + v_s) ∧ t_up = 2 * t_down) →
  v_s = 19 := by
  sorry

end speed_of_the_stream_l272_272085


namespace initial_pages_l272_272542

/-
Given:
1. Sammy uses 25% of the pages for his science project.
2. Sammy uses another 10 pages for his math homework.
3. There are 80 pages remaining in the pad.

Prove that the initial number of pages in the pad (P) is 120.
-/

theorem initial_pages (P : ℝ) (h1 : P * 0.25 + 10 + 80 = P) : 
  P = 120 :=
by 
  sorry

end initial_pages_l272_272542


namespace square_of_positive_difference_l272_272681

theorem square_of_positive_difference {y : ℝ}
  (h : (45 + y) / 2 = 50) :
  (|y - 45|)^2 = 100 :=
by
  sorry

end square_of_positive_difference_l272_272681


namespace jayden_half_of_ernesto_in_some_years_l272_272189

theorem jayden_half_of_ernesto_in_some_years :
  ∃ x : ℕ, (4 + x = (1 : ℝ) / 2 * (11 + x)) ∧ x = 3 := by
  sorry

end jayden_half_of_ernesto_in_some_years_l272_272189


namespace a_4_is_11_l272_272192

def S (n : ℕ) : ℕ := 2 * n^2 - 3 * n

def a (n : ℕ) : ℕ := S n - S (n - 1)

theorem a_4_is_11 : a 4 = 11 := by
  sorry

end a_4_is_11_l272_272192


namespace find_original_number_l272_272288

-- Variables and assumptions
variables (N y x : ℕ)

-- Conditions
def crossing_out_condition : Prop := N = 10 * y + x
def sum_condition : Prop := N + y = 54321
def remainder_condition : Prop := x = 54321 % 11
def y_value : Prop := y = 4938

-- The final proof problem statement
theorem find_original_number (h1 : crossing_out_condition N y x) (h2 : sum_condition N y) 
  (h3 : remainder_condition x) (h4 : y_value y) : N = 49383 :=
sorry

end find_original_number_l272_272288


namespace count_odd_numbers_300_600_l272_272620

theorem count_odd_numbers_300_600 : ∃ n : ℕ, n = 149 ∧ ∀ k : ℕ, (301 ≤ k ∧ k < 600 ∧ k % 2 = 1) ↔ (301 ≤ k ∧ k < 600 ∧ k % 2 = 1 ∧ k - 301 < n * 2) :=
by {
  sorry
}

end count_odd_numbers_300_600_l272_272620


namespace average_discount_rate_l272_272424

theorem average_discount_rate
  (bag_marked_price : ℝ) (bag_sold_price : ℝ)
  (shoes_marked_price : ℝ) (shoes_sold_price : ℝ)
  (jacket_marked_price : ℝ) (jacket_sold_price : ℝ)
  (h_bag : bag_marked_price = 80) (h_bag_sold : bag_sold_price = 68)
  (h_shoes : shoes_marked_price = 120) (h_shoes_sold : shoes_sold_price = 96)
  (h_jacket : jacket_marked_price = 150) (h_jacket_sold : jacket_sold_price = 135) : 
  (15 : ℝ) =
  (((bag_marked_price - bag_sold_price) / bag_marked_price * 100) + 
   ((shoes_marked_price - shoes_sold_price) / shoes_marked_price * 100) + 
   ((jacket_marked_price - jacket_sold_price) / jacket_marked_price * 100)) / 3 :=
by {
  sorry
}

end average_discount_rate_l272_272424


namespace volunteer_distribution_l272_272444

theorem volunteer_distribution :
  let students := 5
  let projects := 4
  let combinations := Nat.choose students 2
  let permutations := Nat.factorial projects
  combinations * permutations = 240 := 
by
  sorry

end volunteer_distribution_l272_272444


namespace ducks_drinking_l272_272578

theorem ducks_drinking (total_d : ℕ) (drank_before : ℕ) (drank_after : ℕ) :
  total_d = 20 → drank_before = 11 → drank_after = total_d - (drank_before + 1) → drank_after = 8 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  norm_num at h3
  exact h3

end ducks_drinking_l272_272578


namespace client_dropped_off_phones_l272_272654

def initial_phones : ℕ := 15
def repaired_phones : ℕ := 3
def coworker_phones : ℕ := 9

theorem client_dropped_off_phones (x : ℕ) : 
  initial_phones - repaired_phones + x = 2 * coworker_phones → x = 6 :=
by
  sorry

end client_dropped_off_phones_l272_272654


namespace original_five_digit_number_l272_272277

theorem original_five_digit_number :
  ∃ N y x : ℕ, (N = 10 * y + x) ∧ (N + y = 54321) ∧ (N = 49383) :=
by
  -- The proof script goes here
  sorry

end original_five_digit_number_l272_272277


namespace square_possible_n12_square_possible_n15_l272_272461

-- Define the nature of the problem with condition n = 12
def min_sticks_to_break_for_square_n12 : ℕ :=
  let n := 12
  let total_length := (n * (n + 1)) / 2
  if total_length % 4 = 0 then 0 else 2

-- Define the nature of the problem with condition n = 15
def min_sticks_to_break_for_square_n15 : ℕ :=
  let n := 15
  let total_length := (n * (n + 1)) / 2
  if total_length % 4 = 0 then 0 else 2

-- Statement of the problems in Lean 4 language
theorem square_possible_n12 : min_sticks_to_break_for_square_n12 = 2 := by
  sorry

theorem square_possible_n15 : min_sticks_to_break_for_square_n15 = 0 := by
  sorry

end square_possible_n12_square_possible_n15_l272_272461


namespace cubes_even_sum_even_l272_272202

theorem cubes_even_sum_even (p q : ℕ) (h : Even (p^3 - q^3)) : Even (p + q) := sorry

end cubes_even_sum_even_l272_272202


namespace first_group_hours_per_day_l272_272876

theorem first_group_hours_per_day :
  ∃ H : ℕ, 
    (39 * 12 * H = 30 * 26 * 3) ∧
    H = 5 :=
by sorry

end first_group_hours_per_day_l272_272876


namespace total_time_equiv_7_75_l272_272035

def acclimation_period : ℝ := 1
def learning_basics : ℝ := 2
def research_time_without_sabbatical : ℝ := learning_basics + 0.75 * learning_basics
def sabbatical : ℝ := 0.5
def research_time_with_sabbatical : ℝ := research_time_without_sabbatical + sabbatical
def dissertation_without_conference : ℝ := 0.5 * acclimation_period
def conference : ℝ := 0.25
def dissertation_with_conference : ℝ := dissertation_without_conference + conference
def total_time : ℝ := acclimation_period + learning_basics + research_time_with_sabbatical + dissertation_with_conference

theorem total_time_equiv_7_75 : total_time = 7.75 := by
  sorry

end total_time_equiv_7_75_l272_272035


namespace total_number_of_animals_l272_272521

-- Definitions for the number of each type of animal
def cats : ℕ := 645
def dogs : ℕ := 567
def rabbits : ℕ := 316
def reptiles : ℕ := 120

-- The statement to prove
theorem total_number_of_animals :
  cats + dogs + rabbits + reptiles = 1648 := by
  sorry

end total_number_of_animals_l272_272521


namespace greatest_remainder_le_11_l272_272498

noncomputable def greatest_remainder (x : ℕ) : ℕ := x % 12

theorem greatest_remainder_le_11 (x : ℕ) (h : x % 12 ≠ 0) : greatest_remainder x = 11 :=
by
  sorry

end greatest_remainder_le_11_l272_272498


namespace prob_at_least_one_l272_272244

-- Defining the probabilities of the alarms going off on time
def prob_A : ℝ := 0.80
def prob_B : ℝ := 0.90

-- Define the complementary event (neither alarm goes off on time)
def prob_neither : ℝ := (1 - prob_A) * (1 - prob_B)

-- The main theorem statement we need to prove
theorem prob_at_least_one : 1 - prob_neither = 0.98 :=
by
  sorry

end prob_at_least_one_l272_272244


namespace max_x_real_nums_l272_272048

theorem max_x_real_nums (x y z : ℝ) (h₁ : x + y + z = 6) (h₂ : x * y + x * z + y * z = 10) : x ≤ 2 :=
sorry

end max_x_real_nums_l272_272048


namespace age_of_b_l272_272415

variable {a b c d Y : ℝ}

-- Conditions
def condition1 (a b : ℝ) := a = b + 2
def condition2 (b c : ℝ) := b = 2 * c
def condition3 (a d : ℝ) := d = a / 2
def condition4 (a b c d Y : ℝ) := a + b + c + d = Y

-- Theorem to prove
theorem age_of_b (a b c d Y : ℝ) 
  (h1 : condition1 a b) 
  (h2 : condition2 b c) 
  (h3 : condition3 a d) 
  (h4 : condition4 a b c d Y) : 
  b = Y / 3 - 1 := 
sorry

end age_of_b_l272_272415


namespace cube_less_than_triple_l272_272698

theorem cube_less_than_triple : ∀ x : ℤ, (x^3 < 3*x) ↔ (x = 1 ∨ x = -2) :=
by
  sorry

end cube_less_than_triple_l272_272698


namespace non_divisible_l272_272988

theorem non_divisible (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  ¬ ∃ k : ℤ, x^2 + y^2 + z^2 = k * 3 * (x * y + y * z + z * x) :=
by sorry

end non_divisible_l272_272988


namespace sqrt_eq_pm_four_l272_272853

theorem sqrt_eq_pm_four (a : ℤ) : (a * a = 16) ↔ (a = 4 ∨ a = -4) :=
by sorry

end sqrt_eq_pm_four_l272_272853


namespace find_m_l272_272354

noncomputable def parametric_x (t m : ℝ) := (√3/2) * t + m
noncomputable def parametric_y (t : ℝ) := (1/2) * t

noncomputable def circle_eq (x y : ℝ) := (x - 2)^2 + y^2 = 4

noncomputable def line_eq (x y m : ℝ) := x - √3 * y - m = 0

def is_tangent (m : ℝ) : Prop :=
  let dist := abs ((2 - m) / (Real.sqrt ((√3/2)^2 + (1/2)^2))) in
  dist = 2

theorem find_m (m: ℝ) (t : ℝ):
  (∀ t, line_eq (parametric_x t m) (parametric_y t) m) →
  (circle_eq (parametric_x t m) (parametric_y t)) →
  is_tangent m :=
sorry

end find_m_l272_272354


namespace disinfectant_usage_l272_272427

theorem disinfectant_usage (x : ℝ) (hx1 : 0 < x) (hx2 : 120 / x / 2 = 120 / (x + 4)) : x = 4 :=
by
  sorry

end disinfectant_usage_l272_272427


namespace eight_S_three_l272_272594

def custom_operation_S (a b : ℤ) : ℤ := 4 * a + 6 * b + 3

theorem eight_S_three : custom_operation_S 8 3 = 53 := by
  sorry

end eight_S_three_l272_272594


namespace sum_tens_ones_digit_l272_272409

theorem sum_tens_ones_digit (a : ℕ) (b : ℕ) (n : ℕ) (h : a - b = 3) :
  let d := (3^n)
  let ones_digit := d % 10
  let tens_digit := (d / 10) % 10
  ones_digit + tens_digit = 9 :=
by 
  let d := 3^17
  let ones_digit := d % 10
  let tens_digit := (d / 10) % 10
  sorry

end sum_tens_ones_digit_l272_272409


namespace tourists_number_l272_272239

theorem tourists_number (m : ℕ) (k l : ℤ) (n : ℕ) (hn : n = 23) (hm1 : 2 * m ≡ 1 [MOD n]) (hm2 : 3 * m ≡ 13 [MOD n]) (hn_gt_13 : n > 13) : n = 23 := 
by
  sorry

end tourists_number_l272_272239


namespace smallest_difference_l272_272857

theorem smallest_difference (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_prod : a * b * c = 362880) (h_order : a < b ∧ b < c) : c - a = 92 := 
sorry

end smallest_difference_l272_272857


namespace solve_for_k_l272_272023

noncomputable def f (k x : ℝ) : ℝ := k * x^2 + (k-1) * x + 2

theorem solve_for_k (k : ℝ) :
  (∀ x : ℝ, f k x = f k (-x)) ↔ k = 1 :=
by
  sorry

end solve_for_k_l272_272023


namespace marco_strawberries_weight_l272_272055

theorem marco_strawberries_weight 
  (m : ℕ) 
  (total_weight : ℕ := 40) 
  (dad_weight : ℕ := 32) 
  (h : total_weight = m + dad_weight) : 
  m = 8 := 
sorry

end marco_strawberries_weight_l272_272055


namespace value_of_a_l272_272981

noncomputable def A : Set ℝ := {x | x^2 - x - 2 < 0}
noncomputable def B (a : ℝ) : Set ℝ := {x | a < x ∧ x < a + 5}

theorem value_of_a (a : ℝ) (h : A ⊆ B a) : -3 ≤ a ∧ a ≤ -1 :=
by
  sorry

end value_of_a_l272_272981


namespace tetrahedron_volume_formula_l272_272673

-- Definitions used directly in the conditions
variable (a b d : ℝ) (φ : ℝ)

-- Tetrahedron volume formula theorem statement
theorem tetrahedron_volume_formula 
  (ha_pos : 0 < a) 
  (hb_pos : 0 < b) 
  (hd_pos : 0 < d) 
  (hφ_pos : 0 < φ) 
  (hφ_le_pi : φ ≤ Real.pi) :
  (∀ V : ℝ, V = 1 / 6 * a * b * d * Real.sin φ) :=
sorry

end tetrahedron_volume_formula_l272_272673


namespace machine_value_after_2_years_l272_272107

section
def initial_value : ℝ := 1200
def depreciation_rate_year1 : ℝ := 0.10
def depreciation_rate_year2 : ℝ := 0.12
def repair_rate : ℝ := 0.03
def major_overhaul_rate : ℝ := 0.15

theorem machine_value_after_2_years :
  let value_after_repairs_2 := (initial_value * (1 - depreciation_rate_year1) + initial_value * repair_rate) * (1 - depreciation_rate_year2 + repair_rate)
  (value_after_repairs_2 * (1 - major_overhaul_rate)) = 863.23 := 
by
  -- proof here
  sorry
end

end machine_value_after_2_years_l272_272107


namespace average_speed_last_segment_l272_272061

theorem average_speed_last_segment
  (total_distance : ℕ)
  (total_time_minutes : ℕ)
  (avg_speed_first_segment : ℕ)
  (avg_speed_second_segment : ℕ)
  (expected_avg_speed_last_segment : ℕ) :
  total_distance = 96 →
  total_time_minutes = 90 →
  avg_speed_first_segment = 60 →
  avg_speed_second_segment = 65 →
  expected_avg_speed_last_segment = 67 →
  (3 * (avg_speed_first_segment + avg_speed_second_segment + expected_avg_speed_last_segment) = (total_distance * 2)) :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  have total_time_hours := 1.5
  have overall_avg_speed := 96 / 1.5
  have overall_avg_speed_value : overall_avg_speed = 64 := by linarith [total_time_hours]
  have avg_calc : (60 + 65 + 67) / 3 = 64 := by linarith
  sorry

end average_speed_last_segment_l272_272061


namespace sum_of_adjacents_to_15_l272_272543

-- Definitions of the conditions
def divisorsOf225 : Set ℕ := {3, 5, 9, 15, 25, 45, 75, 225}

-- Definition of the adjacency relationship
def isAdjacent (x y : ℕ) (s : Set ℕ) : Prop :=
  x ∈ s ∧ y ∈ s ∧ Nat.gcd x y > 1

-- Problem statement in Lean 4
theorem sum_of_adjacents_to_15 :
  ∃ x y : ℕ, isAdjacent 15 x divisorsOf225 ∧ isAdjacent 15 y divisorsOf225 ∧ x + y = 120 :=
by
  sorry

end sum_of_adjacents_to_15_l272_272543


namespace value_of_k_l272_272022

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

noncomputable def f (k x : ℝ) : ℝ := k * x^2 + (k - 1) * x + 2

theorem value_of_k (k : ℝ) :
  is_even_function (f k) → k = 1 :=
by {
  sorry
}

end value_of_k_l272_272022


namespace value_of_t_for_x_equals_y_l272_272416

theorem value_of_t_for_x_equals_y (t : ℝ) (h1 : x = 1 - 4 * t) (h2 : y = 2 * t - 2) : 
    t = 1 / 2 → x = y :=
by 
  intro ht
  rw [ht] at h1 h2
  sorry

end value_of_t_for_x_equals_y_l272_272416


namespace cube_surface_area_correct_l272_272268

def edge_length : ℝ := 11

def cube_surface_area (e : ℝ) : ℝ := 6 * e^2

theorem cube_surface_area_correct : cube_surface_area edge_length = 726 := by
  sorry

end cube_surface_area_correct_l272_272268


namespace slope_divides_polygon_area_l272_272808

structure Point where
  x : ℝ
  y : ℝ

noncomputable def polygon_vertices : List Point :=
  [⟨0, 0⟩, ⟨0, 4⟩, ⟨4, 4⟩, ⟨4, 2⟩, ⟨7, 2⟩, ⟨7, 0⟩]

-- Define the area calculation and conditions needed 
noncomputable def area_of_polygon (vertices : List Point) : ℝ :=
  -- Assuming here that a function exists to calculate the area given the vertices
  sorry

def line_through_origin (slope : ℝ) (x : ℝ) : Point :=
  ⟨x, slope * x⟩

theorem slope_divides_polygon_area :
  let line := line_through_origin (2 / 7)
  ∀ x : ℝ, ∃ (G : Point), 
  polygon_vertices = [⟨0, 0⟩, ⟨0, 4⟩, ⟨4, 4⟩, ⟨4, 2⟩, ⟨7, 2⟩, ⟨7, 0⟩] →
  area_of_polygon polygon_vertices / 2 = 
  area_of_polygon [⟨0, 0⟩, line x, G] :=
sorry

end slope_divides_polygon_area_l272_272808


namespace area_converted_2018_l272_272955

theorem area_converted_2018 :
  let a₁ := 8 -- initial area in ten thousand hectares
  let q := 1.1 -- common ratio
  let a₆ := a₁ * q^5 -- area converted in 2018
  a₆ = 8 * 1.1^5 :=
sorry

end area_converted_2018_l272_272955


namespace percent_only_cats_l272_272639

def total_students := 500
def total_cats := 120
def total_dogs := 200
def both_cats_and_dogs := 40
def only_cats := total_cats - both_cats_and_dogs

theorem percent_only_cats:
  (only_cats : ℕ) / (total_students : ℕ) * 100 = 16 := 
by 
  sorry

end percent_only_cats_l272_272639


namespace find_original_number_l272_272285

-- Variables and assumptions
variables (N y x : ℕ)

-- Conditions
def crossing_out_condition : Prop := N = 10 * y + x
def sum_condition : Prop := N + y = 54321
def remainder_condition : Prop := x = 54321 % 11
def y_value : Prop := y = 4938

-- The final proof problem statement
theorem find_original_number (h1 : crossing_out_condition N y x) (h2 : sum_condition N y) 
  (h3 : remainder_condition x) (h4 : y_value y) : N = 49383 :=
sorry

end find_original_number_l272_272285


namespace unique_integer_cube_triple_l272_272694

theorem unique_integer_cube_triple (x : ℤ) (h : x^3 < 3 * x) : x = 1 := 
sorry

end unique_integer_cube_triple_l272_272694


namespace greatest_number_of_dimes_l272_272983

theorem greatest_number_of_dimes (total_value : ℝ) (num_dimes : ℕ) (num_nickels : ℕ) 
  (h_same_num : num_dimes = num_nickels) (h_total_value : total_value = 4.80) 
  (h_value_calculation : 0.10 * num_dimes + 0.05 * num_nickels = total_value) :
  num_dimes = 32 :=
by
  sorry

end greatest_number_of_dimes_l272_272983


namespace arrangement_count_is_74_l272_272995

def count_valid_arrangements : Nat :=
  74

-- Lean statement for the proof
theorem arrangement_count_is_74 :
  let seven_cards := list.range' 1 7 in
  ∃ seq : list Nat, 
    (seq.length = 7) ∧ 
    (∀ n, list.erase seq n = list.range' 1 6 ∨ 
          (list.reverse (list.erase seq n) = list.range' 1 6)) ∧
    (count_valid_arrangements = 74) :=
by
  let seven_cards := list.range' 1 7
  existsi seven_cards
  split
  -- Provide the conditions here for Lean to handle
  sorry

end arrangement_count_is_74_l272_272995


namespace mary_balloon_count_l272_272835

theorem mary_balloon_count (n m : ℕ) (hn : n = 7) (hm : m = 4 * n) : m = 28 :=
by
  sorry

end mary_balloon_count_l272_272835


namespace lewis_total_earnings_l272_272822

def Weekly_earnings : ℕ := 92
def Number_of_weeks : ℕ := 5

theorem lewis_total_earnings : Weekly_earnings * Number_of_weeks = 460 := by
  sorry

end lewis_total_earnings_l272_272822


namespace minimize_travel_expense_l272_272236

noncomputable def travel_cost_A (x : ℕ) : ℝ := 2000 * x * 0.75
noncomputable def travel_cost_B (x : ℕ) : ℝ := 2000 * (x - 1) * 0.8

theorem minimize_travel_expense (x : ℕ) (h1 : 10 ≤ x) (h2 : x ≤ 25) :
  (10 ≤ x ∧ x ≤ 15 → travel_cost_B x < travel_cost_A x) ∧
  (x = 16 → travel_cost_A x = travel_cost_B x) ∧
  (17 ≤ x ∧ x ≤ 25 → travel_cost_A x < travel_cost_B x) :=
by
  sorry

end minimize_travel_expense_l272_272236


namespace correct_growth_rate_l272_272540

noncomputable def growth_rate_eq (x : ℝ) : Prop :=
  10 * (1 + x)^2 = 11.5

axiom initial_sales_volume : ℝ := 10
axiom final_sales_volume : ℝ := 11.5
axiom monthly_growth_rate (x : ℝ) : x > 0

theorem correct_growth_rate (x : ℝ) (hx : monthly_growth_rate x) :
  growth_rate_eq x :=
-- sorry
by
  have h1 : initial_sales_volume = 10 := rfl
  have h2 : final_sales_volume = 11.5 := rfl
  rw [h1, h2]
  sorry

end correct_growth_rate_l272_272540


namespace number_of_true_propositions_l272_272488

-- Define the original condition
def original_proposition (a b : ℝ) : Prop := (a + b = 1) → (a * b ≤ 1 / 4)

-- Define contrapositive
def contrapositive (a b : ℝ) : Prop := (a * b > 1 / 4) → (a + b ≠ 1)

-- Define inverse
def inverse (a b : ℝ) : Prop := (a * b ≤ 1 / 4) → (a + b = 1)

-- Define converse
def converse (a b : ℝ) : Prop := (a + b ≠ 1) → (a * b > 1 / 4)

-- State the problem
theorem number_of_true_propositions (a b : ℝ) :
  (original_proposition a b ∧ contrapositive a b ∧ ¬inverse a b ∧ ¬converse a b) → 
  (∃ n : ℕ, n = 1) :=
by sorry

end number_of_true_propositions_l272_272488


namespace waiter_earnings_l272_272437

theorem waiter_earnings (total_customers tipping_customers no_tip_customers tips_each : ℕ) (h1 : total_customers = 7) (h2 : no_tip_customers = 4) (h3 : tips_each = 9) (h4 : tipping_customers = total_customers - no_tip_customers) :
  tipping_customers * tips_each = 27 :=
by sorry

end waiter_earnings_l272_272437


namespace overall_sale_price_per_kg_l272_272116

-- Defining the quantities and prices
def tea_A_quantity : ℝ := 80
def tea_A_cost_per_kg : ℝ := 15
def tea_B_quantity : ℝ := 20
def tea_B_cost_per_kg : ℝ := 20
def tea_C_quantity : ℝ := 50
def tea_C_cost_per_kg : ℝ := 25
def tea_D_quantity : ℝ := 40
def tea_D_cost_per_kg : ℝ := 30

-- Defining the profit percentages
def tea_A_profit_percentage : ℝ := 0.30
def tea_B_profit_percentage : ℝ := 0.25
def tea_C_profit_percentage : ℝ := 0.20
def tea_D_profit_percentage : ℝ := 0.15

-- Desired sale price per kg
theorem overall_sale_price_per_kg : 
  (tea_A_quantity * tea_A_cost_per_kg * (1 + tea_A_profit_percentage) +
   tea_B_quantity * tea_B_cost_per_kg * (1 + tea_B_profit_percentage) +
   tea_C_quantity * tea_C_cost_per_kg * (1 + tea_C_profit_percentage) +
   tea_D_quantity * tea_D_cost_per_kg * (1 + tea_D_profit_percentage)) / 
  (tea_A_quantity + tea_B_quantity + tea_C_quantity + tea_D_quantity) = 26 := 
by
  sorry

end overall_sale_price_per_kg_l272_272116


namespace knowledge_competition_score_l272_272640

theorem knowledge_competition_score (x : ℕ) (hx : x ≤ 20) : 5 * x - (20 - x) ≥ 88 :=
  sorry

end knowledge_competition_score_l272_272640


namespace cookies_with_five_cups_of_flour_l272_272522

-- Define the conditions
def initial_cookies : ℕ := 24
def initial_flour : ℕ := 3
def additional_flour : ℕ := 5

-- State the problem
theorem cookies_with_five_cups_of_flour :
  (initial_cookies / initial_flour) * additional_flour = 40 :=
by
  -- Placeholder for proof
  sorry

end cookies_with_five_cups_of_flour_l272_272522


namespace find_integer_for_perfect_square_l272_272446

theorem find_integer_for_perfect_square :
  ∃ (n : ℤ), ∃ (m : ℤ), n^2 + 20 * n + 11 = m^2 ∧ n = 35 := by
  sorry

end find_integer_for_perfect_square_l272_272446


namespace fraction_arithmetic_l272_272864

theorem fraction_arithmetic :
  (3 / 4) / (5 / 8) + (1 / 8) = 53 / 40 :=
by
  sorry

end fraction_arithmetic_l272_272864


namespace ramesh_share_correct_l272_272065

-- Define basic conditions
def suresh_investment := 24000
def ramesh_investment := 40000
def total_profit := 19000

-- Define Ramesh's share calculation
def ramesh_share : ℤ :=
  let ratio_ramesh := ramesh_investment / (suresh_investment + ramesh_investment)
  ratio_ramesh * total_profit

-- Proof statement
theorem ramesh_share_correct : ramesh_share = 11875 := by
  sorry

end ramesh_share_correct_l272_272065


namespace unique_integer_cube_triple_l272_272696

theorem unique_integer_cube_triple (x : ℤ) (h : x^3 < 3 * x) : x = 1 := 
sorry

end unique_integer_cube_triple_l272_272696


namespace dice_sum_probability_l272_272756

theorem dice_sum_probability :
  let D := finset.range 1 7  -- outcomes of a fair six-sided die
  (∃! d1 d2 d3 d4 ∈ D, d1 + d2 + d3 + d4 = 24) ->
  (probability(space, {ω ∈ space | (ω 1 = 6) ∧ (ω 2 = 6) ∧ (ω 3 = 6) ∧ (ω 4 = 6)}) = 1/1296) :=
sorry

end dice_sum_probability_l272_272756


namespace work_problem_l272_272094

theorem work_problem (W : ℝ) (d : ℝ) :
  (1 / 40) * d * W + (28 / 35) * W = W → d = 8 :=
by
  intro h
  sorry

end work_problem_l272_272094


namespace intensity_on_Thursday_l272_272294

-- Step a) - Definitions from Conditions
def inversely_proportional (i b k : ℕ) : Prop := i * b = k

-- Translation of the proof problem
theorem intensity_on_Thursday (k b : ℕ) (h₁ : k = 24) (h₂ : b = 3) : ∃ i, inversely_proportional i b k ∧ i = 8 := 
by
  sorry

end intensity_on_Thursday_l272_272294


namespace max_volume_for_open_top_box_l272_272314

noncomputable def volume (a x : ℝ) : ℝ := x * (a - 2 * x)^2

theorem max_volume_for_open_top_box (a : ℝ) (ha : 0 < a) :
  ∃ x : ℝ, x = a / 6 ∧ 0 < x ∧ x < a / 2 ∧ volume a x = (2 * a^3 / 27) :=
begin
  -- Prove the statement here
  sorry
end

end max_volume_for_open_top_box_l272_272314


namespace die_roll_divisor_of_12_prob_l272_272718

def fair_die_probability_divisor_of_12 : Prop :=
  let favorable_outcomes := {1, 2, 3, 4, 6}
  let total_outcomes := 6
  let probability := favorable_outcomes.size / total_outcomes
  probability = 5 / 6

theorem die_roll_divisor_of_12_prob:
  fair_die_probability_divisor_of_12 :=
by
  sorry

end die_roll_divisor_of_12_prob_l272_272718


namespace number_of_sandwiches_l272_272070

-- Definitions based on conditions
def breads : Nat := 5
def meats : Nat := 7
def cheeses : Nat := 6
def total_sandwiches : Nat := breads * meats * cheeses
def turkey_mozzarella_exclusions : Nat := breads
def rye_beef_exclusions : Nat := cheeses

-- The proof problem statement
theorem number_of_sandwiches (total_sandwiches := 210) 
  (turkey_mozzarella_exclusions := 5) 
  (rye_beef_exclusions := 6) : 
  total_sandwiches - turkey_mozzarella_exclusions - rye_beef_exclusions = 199 := 
by sorry

end number_of_sandwiches_l272_272070


namespace probability_divisible_by_4_l272_272412

-- Definitions and assumptions
def is_divisible_by_4 (n : ℕ) : Prop :=
  n % 4 = 0

def fair_8_sided_die : Finset ℕ :=
  {n | n ∈ (Finset.range 9) \ {0}}

-- Main statement
theorem probability_divisible_by_4 :
  let P := (finset.filter (λ a, is_divisible_by_4 a) fair_8_sided_die).card.toRat / fair_8_sided_die.card.toRat in
  let P2 := P * P in
  (P2 = (1 / 16 : ℚ)) :=
sorry

end probability_divisible_by_4_l272_272412


namespace find_range_a_l272_272002

-- Define the proposition p
def p (m : ℝ) : Prop :=
1 < m ∧ m < 3 / 2

-- Define the proposition q
def q (m a : ℝ) : Prop :=
(m - a) * (m - (a + 1)) < 0

-- Define the sufficient but not necessary condition
def sufficient (a : ℝ) : Prop :=
(a ≤ 1) ∧ (3 / 2 ≤ a + 1)

theorem find_range_a (a : ℝ) :
  (∀ m, p m → q m a) → sufficient a → (1 / 2 ≤ a ∧ a ≤ 1) :=
sorry

end find_range_a_l272_272002


namespace find_cost_of_chocolate_l272_272059

theorem find_cost_of_chocolate
  (C : ℕ)
  (h1 : 5 * C + 10 = 90 - 55)
  (h2 : 5 * 2 = 10)
  (h3 : 55 = 90 - (5 * C + 10)):
  C = 5 :=
by
  sorry

end find_cost_of_chocolate_l272_272059


namespace range_of_x_l272_272336

noncomputable def f (x : ℝ) : ℝ := Real.log x + 2^x

theorem range_of_x (x : ℝ) (h : f (x^2 + 2) < f (3 * x)) : 1 < x ∧ x < 2 :=
by sorry

end range_of_x_l272_272336


namespace new_home_fraction_l272_272930

variable {M H G : ℚ} -- Use ℚ (rational numbers)

def library_fraction (H : ℚ) (G : ℚ) (M : ℚ) : ℚ :=
  (1 / 3 * H + 2 / 5 * G + 1 / 2 * M) / M

theorem new_home_fraction (H_eq : H = 1 / 2 * M) (G_eq : G = 3 * H) :
  library_fraction H G M = 29 / 30 :=
by
  sorry

end new_home_fraction_l272_272930


namespace cone_height_l272_272481

theorem cone_height (slant_height r : ℝ) (lateral_area : ℝ) (h : ℝ) 
  (h_slant : slant_height = 13) 
  (h_area : lateral_area = 65 * Real.pi) 
  (h_lateral_area : lateral_area = Real.pi * r * slant_height) 
  (h_radius : r = 5) :
  h = Real.sqrt (slant_height ^ 2 - r ^ 2) :=
by
  -- Definitions and conditions
  have h_slant_height : slant_height = 13 := h_slant
  have h_lateral_area_value : lateral_area = 65 * Real.pi := h_area
  have h_lateral_surface_area : lateral_area = Real.pi * r * slant_height := h_lateral_area
  have h_radius_5 : r = 5 := h_radius
  sorry -- Proof is omitted

end cone_height_l272_272481


namespace complex_division_l272_272001

-- Define i as the imaginary unit
def i : Complex := Complex.I

-- Define the problem statement to prove that 2i / (1 - i) equals -1 + i
theorem complex_division : (2 * i) / (1 - i) = -1 + i :=
by
  -- Since we are focusing on the statement, we use sorry to skip the proof
  sorry

end complex_division_l272_272001


namespace area_of_square_STUV_l272_272549

-- Defining the conditions
variable (C L : ℝ)
variable (h1 : 2 * (C + L) = 40)

-- The goal is to prove the area of the square STUV
theorem area_of_square_STUV : (C + L) * (C + L) = 400 :=
by
  sorry

end area_of_square_STUV_l272_272549


namespace inscribed_circle_radius_l272_272684

theorem inscribed_circle_radius :
  ∀ (a b c : ℝ), a = 3 → b = 6 → c = 18 → (∃ (r : ℝ), (1 / r) = (1 / a) + (1 / b) + (1 / c) + 4 * Real.sqrt ((1 / (a * b)) + (1 / (a * c)) + (1 / (b * c))) ∧ r = 9 / (5 + 6 * Real.sqrt 3)) :=
by
  intros a b c h₁ h₂ h₃
  sorry

end inscribed_circle_radius_l272_272684


namespace cone_slant_height_l272_272616

theorem cone_slant_height (r l : ℝ) (h1 : r = 1)
  (h2 : 2 * r * Real.pi = (1 / 2) * 2 * l * Real.pi) :
  l = 2 :=
by
  -- Proof steps go here
  sorry

end cone_slant_height_l272_272616


namespace total_laces_needed_l272_272901

variable (x : ℕ) -- Eva has x pairs of shoes
def long_laces_per_pair : ℕ := 3
def short_laces_per_pair : ℕ := 3
def laces_per_pair : ℕ := long_laces_per_pair + short_laces_per_pair

theorem total_laces_needed : 6 * x = 6 * x :=
by
  have h : laces_per_pair = 6 := rfl
  sorry

end total_laces_needed_l272_272901


namespace smallest_possible_value_l272_272682

/-
Given:
1. m and n are positive integers.
2. gcd of m and n is (x + 5).
3. lcm of m and n is x * (x + 5).
4. m = 60.
5. x is a positive integer.

Prove:
The smallest possible value of n is 100.
-/

theorem smallest_possible_value 
  (m n x : ℕ) 
  (h1 : m = 60) 
  (h2 : x > 0) 
  (h3 : Nat.gcd m n = x + 5) 
  (h4 : Nat.lcm m n = x * (x + 5)) : 
  n = 100 := 
by 
  sorry

end smallest_possible_value_l272_272682


namespace greene_family_amusement_park_spending_l272_272845

def spent_on_admission : ℝ := 45
def original_ticket_cost : ℝ := 50
def spent_less_than_original_cost_on_food_and_beverages : ℝ := 13
def spent_on_souvenir_Mr_Greene : ℝ := 15
def spent_on_souvenir_Mrs_Greene : ℝ := 2 * spent_on_souvenir_Mr_Greene
def cost_per_game : ℝ := 9
def number_of_children : ℝ := 3
def spent_on_transportation : ℝ := 25
def tax_rate : ℝ := 0.08

def food_and_beverages_cost : ℝ := original_ticket_cost - spent_less_than_original_cost_on_food_and_beverages
def games_cost : ℝ := number_of_children * cost_per_game
def taxable_amount : ℝ := food_and_beverages_cost + spent_on_souvenir_Mr_Greene + spent_on_souvenir_Mrs_Greene + games_cost
def tax : ℝ := tax_rate * taxable_amount
def total_expenditure : ℝ := spent_on_admission + food_and_beverages_cost + spent_on_souvenir_Mr_Greene + spent_on_souvenir_Mrs_Greene + games_cost + spent_on_transportation + tax

theorem greene_family_amusement_park_spending : total_expenditure = 187.72 :=
by {
  sorry
}

end greene_family_amusement_park_spending_l272_272845


namespace integer_values_between_fractions_l272_272004

theorem integer_values_between_fractions :
  let a := 4 / (Real.sqrt 3 + Real.sqrt 2)
  let b := 4 / (Real.sqrt 5 - Real.sqrt 3)
  ((⌊b⌋ - ⌈a⌉) + 1) = 6 :=
by sorry

end integer_values_between_fractions_l272_272004


namespace directrix_eqn_of_parabola_l272_272148

theorem directrix_eqn_of_parabola : 
  ∀ y : ℝ, x = - (1 / 4 : ℝ) * y ^ 2 → x = 1 :=
by
  sorry

end directrix_eqn_of_parabola_l272_272148


namespace prob_not_green_is_six_over_eleven_l272_272632

-- Define the odds for pulling a green marble
def odds_green : ℕ × ℕ := (5, 6)

-- Define the total number of events as the sum of both parts of the odds
def total_events : ℕ := odds_green.1 + odds_green.2

-- Define the probability of not pulling a green marble
def probability_not_green : ℚ := odds_green.2 / total_events

-- State the theorem
theorem prob_not_green_is_six_over_eleven : probability_not_green = 6 / 11 := by
  -- Proof goes here
  sorry

end prob_not_green_is_six_over_eleven_l272_272632


namespace prob_divisor_of_12_l272_272722

theorem prob_divisor_of_12 :
  (∃ d : Finset ℕ, d = {1, 2, 3, 4, 6}) → (∃ s : Finset ℕ, s = {1, 2, 3, 4, 5, 6}) →
  let favorable := 5
  let total := 6
  favorable / total = (5 : ℚ / 6 ) := sorry

end prob_divisor_of_12_l272_272722


namespace range_of_m_perimeter_of_isosceles_triangle_l272_272648

-- Define the variables for the lengths of the sides and the range of m
variables (AB BC AC : ℝ) (m : ℝ)

-- Conditions given in the problem
def triangle_conditions (AB BC : ℝ) (AC : ℝ) (m : ℝ) : Prop :=
  AB = 17 ∧ BC = 8 ∧ AC = 2 * m - 1

-- Proof that the range for m is between 5 and 13
theorem range_of_m (AB BC : ℝ) (m : ℝ) (h : triangle_conditions AB BC (2 * m - 1) m) : 
  5 < m ∧ m < 13 :=
by
  sorry

-- Proof that the perimeter is 42 when triangle is isosceles with given conditions
theorem perimeter_of_isosceles_triangle (AB BC AC : ℝ) (h : triangle_conditions AB BC AC 0) : 
  (AB = AC ∨ BC = AC) → (2 * AB + BC = 42) :=
by
  sorry

end range_of_m_perimeter_of_isosceles_triangle_l272_272648


namespace solve_sum_of_digits_eq_2018_l272_272051

def s (n : ℕ) : ℕ := (Nat.digits 10 n).sum

theorem solve_sum_of_digits_eq_2018 : ∃ n : ℕ, n + s n = 2018 := by
  sorry

end solve_sum_of_digits_eq_2018_l272_272051


namespace find_savings_l272_272417

-- Define the problem statement
def income_expenditure_problem (income expenditure : ℝ) (ratio : ℝ) : Prop :=
  (income / ratio = expenditure) ∧ (income = 20000)

-- Define the theorem for savings
theorem find_savings (income expenditure : ℝ) (ratio : ℝ) (h_ratio : ratio = 4 / 5) (h_income : income = 20000) : 
  income_expenditure_problem income expenditure ratio → income - expenditure = 4000 :=
by
  sorry

end find_savings_l272_272417


namespace inequality_solution_l272_272323

noncomputable def f (x : ℝ) : ℝ :=
  (2 / (x + 2)) + (4 / (x + 8))

theorem inequality_solution {x : ℝ} :
  f x ≥ 1/2 ↔ ((-8 < x ∧ x ≤ -4) ∨ (-2 ≤ x ∧ x ≤ 2)) :=
sorry

end inequality_solution_l272_272323


namespace highest_y_coordinate_l272_272071

theorem highest_y_coordinate (x y : ℝ) (h : (x^2 / 49 + (y-3)^2 / 25 = 0)) : y = 3 :=
by
  sorry

end highest_y_coordinate_l272_272071


namespace intersection_M_N_l272_272007

def M (x : ℝ) : Prop := Real.log x / Real.log 2 ≥ 0
def N (x : ℝ) : Prop := -2 ≤ x ∧ x ≤ 2

theorem intersection_M_N :
  {x : ℝ | M x} ∩ {x | N x} = {x | 1 ≤ x ∧ x ≤ 2} :=
by
  sorry

end intersection_M_N_l272_272007


namespace maximum_value_of_objective_function_l272_272490

variables (x y : ℝ)

def objective_function (x y : ℝ) := 3 * x + 2 * y

theorem maximum_value_of_objective_function : 
  (∀ x y, (x ≥ 0 ∧ y ≥ 0 ∧ x + y ≤ 4) → objective_function x y ≤ 12) 
  ∧ 
  (∃ x y, x ≥ 0 ∧ y ≥ 0 ∧ x + y ≤ 4 ∧ objective_function x y = 12) :=
sorry

end maximum_value_of_objective_function_l272_272490


namespace find_original_number_l272_272284

open Int

theorem find_original_number (N y x : ℕ) (h1 : N = 10 * y + x) (h2 : N + y = 54321) (h3 : x = 54321 % 11) (h4 : y = 4938) : N = 49383 := by
  sorry

end find_original_number_l272_272284


namespace sam_last_30_minutes_speed_l272_272060

/-- 
Given the total distance of 96 miles driven in 1.5 hours, 
with the first 30 minutes at an average speed of 60 mph, 
and the second 30 minutes at an average speed of 65 mph,
we need to show that the average speed during the last 30 minutes was 67 mph.
-/
theorem sam_last_30_minutes_speed (total_distance : ℤ) (time1 time2 : ℤ) (speed1 speed2 speed_last segment_time : ℤ)
  (h_total_distance : total_distance = 96)
  (h_total_time : time1 + time2 + segment_time = 90)
  (h_segment_time : segment_time = 30)
  (convert_time1 : time1 = 30)
  (convert_time2 : time2 = 30)
  (h_speed1 : speed1 = 60)
  (h_speed2 : speed2 = 65)
  (h_average_speed : ((60 + 65 + speed_last) / 3) = 64) :
  speed_last = 67 := 
sorry

end sam_last_30_minutes_speed_l272_272060


namespace min_sticks_to_break_n12_l272_272463

theorem min_sticks_to_break_n12 : 
  let sticks := (Finset.range 12).map (λ x => x + 1)
  let total_length := sticks.sum
  total_length % 4 ≠ 0 → 
  (∃ k, k < 3 ∧ 
    ∃ broken_sticks: Finset Nat, 
      (∀ s ∈ broken_sticks, s < 12 ∧ s > 0) ∧ broken_sticks.card = k ∧ 
        sticks.sum + (broken_sticks.sum / 2) % 4 = 0) :=
sorry

end min_sticks_to_break_n12_l272_272463


namespace geometric_seq_seventh_term_l272_272883

theorem geometric_seq_seventh_term (a r : ℕ) (r_pos : r > 0) (first_term : a = 3)
    (fifth_term : a * r^4 = 243) : a * r^6 = 2187 := by
  sorry

end geometric_seq_seventh_term_l272_272883


namespace max_cubes_submerged_l272_272088

noncomputable def cylinder_radius (diameter: ℝ) : ℝ := diameter / 2

noncomputable def water_volume (radius height: ℝ) : ℝ := Real.pi * radius^2 * height

noncomputable def cube_volume (edge: ℝ) : ℝ := edge^3

noncomputable def height_of_cubes (edge n: ℝ) : ℝ := edge * n

theorem max_cubes_submerged (diameter height water_height edge: ℝ) 
  (h1: diameter = 2.9)
  (h2: water_height = 4)
  (h3: edge = 2):
  ∃ max_n: ℝ, max_n = 5 := 
  sorry

end max_cubes_submerged_l272_272088


namespace lucy_total_cost_for_lamp_and_table_l272_272663

noncomputable def original_price_lamp : ℝ := 200 / 1.2

noncomputable def table_price : ℝ := 2 * original_price_lamp

noncomputable def total_cost_paid (lamp_cost discounted_price table_price: ℝ) :=
  lamp_cost + table_price

theorem lucy_total_cost_for_lamp_and_table :
  total_cost_paid 20 (original_price_lamp * 0.6) table_price = 353.34 :=
by
  let lamp_original_price := original_price_lamp
  have h1 : original_price_lamp * (0.6 * (1 / 5)) = 20 := by sorry
  have h2 : table_price = 2 * original_price_lamp := by sorry
  have h3 : total_cost_paid 20 (original_price_lamp * 0.6) table_price = 20 + table_price := by sorry
  have h4 : table_price = 2 * (200 / 1.2) := by sorry
  have h5 : 20 + table_price = 353.34 := by sorry
  exact h5

end lucy_total_cost_for_lamp_and_table_l272_272663


namespace andrea_average_distance_per_day_l272_272588

theorem andrea_average_distance_per_day
  (total_distance : ℕ := 168)
  (fraction_completed : ℚ := 3/7)
  (total_days : ℕ := 6)
  (days_completed : ℕ := 3) :
  (total_distance * (1 - fraction_completed) / (total_days - days_completed)) = 32 :=
by sorry

end andrea_average_distance_per_day_l272_272588


namespace inversely_proportional_decrease_l272_272387

theorem inversely_proportional_decrease :
  ∀ {x y q c : ℝ}, 
  0 < x ∧ 0 < y ∧ 0 < c ∧ 0 < q →
  (x * y = c) →
  (((1 + q / 100) * x) * ((100 / (100 + q)) * y) = c) →
  ((y - (100 / (100 + q)) * y) / y) * 100 = 100 * q / (100 + q) :=
by
  intros x y q c hb hxy hxy'
  sorry

end inversely_proportional_decrease_l272_272387


namespace test_takers_percent_correct_l272_272177

theorem test_takers_percent_correct 
  (n : Set ℕ → ℝ) 
  (A B : Set ℕ) 
  (hB : n B = 0.75) 
  (hAB : n (A ∩ B) = 0.60) 
  (hneither : n (Set.univ \ (A ∪ B)) = 0.05) 
  : n A = 0.80 := by
  sorry

end test_takers_percent_correct_l272_272177


namespace total_pokemon_cards_l272_272216

def pokemon_cards (sam dan tom keith : Nat) : Nat :=
  sam + dan + tom + keith

theorem total_pokemon_cards :
  pokemon_cards 14 14 14 14 = 56 := by
  sorry

end total_pokemon_cards_l272_272216


namespace find_b_exists_l272_272755

theorem find_b_exists (N : ℕ) (hN : N ≠ 1) : ∃ (a c d : ℕ), a > 1 ∧ c > 1 ∧ d > 1 ∧
  (N : ℝ) ^ (1/a + 1/(a*4) + 1/(a*4*c) + 1/(a*4*c*d)) = (N : ℝ) ^ (37/48) :=
by
  sorry

end find_b_exists_l272_272755


namespace visibility_beach_to_hill_visibility_ferry_to_tree_l272_272965

noncomputable def altitude_lake : ℝ := 104
noncomputable def altitude_hill_tree : ℝ := 154
noncomputable def map_distance_1 : ℝ := 70 / 100 -- Convert cm to meters
noncomputable def map_distance_2 : ℝ := 38.5 / 100 -- Convert cm to meters
noncomputable def map_scale : ℝ := 95000
noncomputable def earth_circumference : ℝ := 40000000 -- Convert km to meters

noncomputable def earth_radius : ℝ := earth_circumference / (2 * Real.pi)

noncomputable def visible_distance (height : ℝ) : ℝ :=
  Real.sqrt (2 * earth_radius * height)

noncomputable def actual_distance_1 : ℝ := map_distance_1 * map_scale
noncomputable def actual_distance_2 : ℝ := map_distance_2 * map_scale

theorem visibility_beach_to_hill :
  actual_distance_1 > visible_distance (altitude_hill_tree - altitude_lake) :=
by
  sorry

theorem visibility_ferry_to_tree :
  actual_distance_2 > visible_distance (altitude_hill_tree - altitude_lake) :=
by
  sorry

end visibility_beach_to_hill_visibility_ferry_to_tree_l272_272965


namespace round_robin_teams_l272_272688

theorem round_robin_teams (x : ℕ) (h : x * (x - 1) / 2 = 15) : x = 6 :=
sorry

end round_robin_teams_l272_272688


namespace induction_first_step_l272_272689

theorem induction_first_step (n : ℕ) (h₁ : n > 1) : 
  1 + 1/2 + 1/3 < 2 := 
sorry

end induction_first_step_l272_272689


namespace probability_product_divisible_by_4_gt_half_l272_272454

theorem probability_product_divisible_by_4_gt_half :
  let n := 2023
  let even_count := n / 2
  let four_div_count := n / 4
  let select_five := 5
  (true) ∧ (even_count = 1012) ∧ (four_div_count = 505)
  → 0.5 < (1 - ((2023 - even_count) / 2023) * ((2022 - (even_count - 1)) / 2022) * ((2021 - (even_count - 2)) / 2021) * ((2020 - (even_count - 3)) / 2020) * ((2019 - (even_count - 4)) / 2019)) :=
by
  sorry

end probability_product_divisible_by_4_gt_half_l272_272454


namespace max_ab_value_1_half_l272_272800

theorem max_ab_value_1_half 
  (a b : ℝ) 
  (h_pos_a : 0 < a)
  (h_pos_b : 0 < b)
  (h_eq : a + 2 * b = 1) :
  a = 1 / 2 → ab = 1 / 8 :=
sorry

end max_ab_value_1_half_l272_272800


namespace problem_statement_l272_272063

-- Define a multiple of 6 and a multiple of 9
variables (a b : ℤ)
variable (ha : ∃ k, a = 6 * k)
variable (hb : ∃ k, b = 9 * k)

-- Prove that a + b is a multiple of 3
theorem problem_statement : 
  (∃ k, a + b = 3 * k) ∧ 
  ¬((∀ m n, a = 6 * m ∧ b = 9 * n → (a + b = odd))) ∧ 
  ¬(∃ k, a + b = 6 * k) ∧ 
  ¬(∃ k, a + b = 9 * k) :=
by
  sorry

end problem_statement_l272_272063


namespace quadratic_equation_root_and_coef_l272_272019

theorem quadratic_equation_root_and_coef (k x : ℤ) (h1 : x^2 - 3 * x + k = 0)
  (root4 : x = 4) : (x = 4 ∧ k = -4 ∧ ∀ y, y ≠ 4 → y^2 - 3 * y + k = 0 → y = -1) :=
by {
  sorry
}

end quadratic_equation_root_and_coef_l272_272019


namespace max_marks_400_l272_272430

theorem max_marks_400 {M : ℝ} (h : 0.45 * M = 150 + 30) : M = 400 := 
by
  sorry

end max_marks_400_l272_272430


namespace total_cookies_l272_272828

def MonaCookies : ℕ := 20
def JasmineCookies : ℕ := MonaCookies - 5
def RachelCookies : ℕ := JasmineCookies + 10

theorem total_cookies : MonaCookies + JasmineCookies + RachelCookies = 60 := by
  -- Since we don't need to provide the solution steps, we simply use sorry.
  sorry

end total_cookies_l272_272828


namespace tens_digit_of_9_to_2023_l272_272557

theorem tens_digit_of_9_to_2023 :
  (9^2023 % 100) / 10 % 10 = 8 :=
sorry

end tens_digit_of_9_to_2023_l272_272557


namespace water_tank_capacity_l272_272410

-- Define the variables and conditions
variables (T : ℝ) (h : 0.35 * T = 36)

-- State the theorem
theorem water_tank_capacity : T = 103 :=
by
  -- Placeholder for proof
  sorry

end water_tank_capacity_l272_272410


namespace set_intersection_l272_272982

def S : Set ℝ := {x | x^2 - 5 * x + 6 ≥ 0}
def T : Set ℝ := {x | x > 1}
def result : Set ℝ := {x | x ≥ 3 ∨ (1 < x ∧ x ≤ 2)}

theorem set_intersection (x : ℝ) : x ∈ (S ∩ T) ↔ x ∈ result := by
  sorry

end set_intersection_l272_272982


namespace largest_angle_in_pentagon_l272_272641

theorem largest_angle_in_pentagon (A B C D E : ℝ) 
  (hA : A = 70) 
  (hB : B = 120) 
  (hCD : C = D) 
  (hE : E = 3 * C - 30) 
  (sum_angles : A + B + C + D + E = 540) :
  E = 198 := 
by 
  sorry

end largest_angle_in_pentagon_l272_272641


namespace fuel_capacity_ratio_l272_272361

noncomputable def oldCost : ℝ := 200
noncomputable def newCost : ℝ := 480
noncomputable def priceIncreaseFactor : ℝ := 1.20

theorem fuel_capacity_ratio (C C_new : ℝ) (h1 : newCost = C_new * oldCost * priceIncreaseFactor / C) : 
  C_new / C = 2 :=
sorry

end fuel_capacity_ratio_l272_272361


namespace total_bill_l272_272846

theorem total_bill (total_people : ℕ) (children : ℕ) (adult_cost : ℕ) (child_cost : ℕ)
  (h : total_people = 201) (hc : children = 161) (ha : adult_cost = 8) (hc_cost : child_cost = 4) :
  (201 - 161) * 8 + 161 * 4 = 964 :=
by
  rw [←h, ←hc, ←ha, ←hc_cost]
  sorry

end total_bill_l272_272846


namespace intersection_points_count_l272_272592

theorem intersection_points_count (B : ℝ) (hB : 0 < B) :
  ∃ p : ℕ, p = 4 ∧ (∀ x y : ℝ, (y = B * x^2 ∧ y^2 + 4 * y - 2 = x^2 + 5 * y) ↔ p = 4) := by
sorry

end intersection_points_count_l272_272592


namespace cubic_sum_identity_l272_272623

theorem cubic_sum_identity (a b c : ℝ) (h1 : a + b + c = 13) (h2 : ab + ac + bc = 40) : 
  a^3 + b^3 + c^3 - 3 * a * b * c = 637 :=
by
  sorry

end cubic_sum_identity_l272_272623


namespace prob_sum_24_four_dice_l272_272770

section
open ProbabilityTheory

/-- Define the event E24 as the event where the sum of numbers on the top faces of four six-sided dice is 24 -/
def E24 : Event (StdGen) :=
eventOfFun {ω | ∑ i in range 4, (ω.gen_uniform int (6-1)) + 1 = 24}

/-- Probability that the sum of the numbers on top faces of four six-sided dice is 24 is 1/1296. -/
theorem prob_sum_24_four_dice : ⋆{ P(E24) = 1/1296 } := sorry

end

end prob_sum_24_four_dice_l272_272770


namespace complex_division_product_l272_272333

theorem complex_division_product
  (i : ℂ)
  (h_exp: i * i = -1)
  (a b : ℝ)
  (h_div: (1 + 7 * i) / (2 - i) = a + b * i)
  : a * b = -3 := 
sorry

end complex_division_product_l272_272333


namespace cesaro_lupu_real_analysis_l272_272814

noncomputable def proof_problem (a b c x y z : ℝ) : Prop :=
  (0 < a ∧ a < 1) ∧ (0 < b ∧ b < 1) ∧ (0 < c ∧ c < 1) ∧
  (0 < x) ∧ (0 < y) ∧ (0 < z) ∧
  (a^x = b * c) ∧ (b^y = c * a) ∧ (c^z = a * b) →
  (1 / (2 + x) + 1 / (2 + y) + 1 / (2 + z) ≤ 3 / 4)

theorem cesaro_lupu_real_analysis (a b c x y z : ℝ) :
  proof_problem a b c x y z :=
by sorry

end cesaro_lupu_real_analysis_l272_272814


namespace tangent_line_eqn_l272_272082

noncomputable def curve (x : ℝ) : ℝ := Real.log x + x + 1
def derive (x : ℝ) : ℝ := (1 / x) + 1

theorem tangent_line_eqn :
  (∀ m x₀ y₀ : ℝ, (derive x₀ = m) → (y₀ = curve x₀) → (∀ x : ℝ, y₀ + m * (x - x₀) = 2 * x)) :=
by
  sorry

end tangent_line_eqn_l272_272082


namespace initial_southwards_distance_l272_272528

-- Define a structure that outlines the journey details
structure Journey :=
  (southwards : ℕ) 
  (westwards1 : ℕ := 10)
  (northwards : ℕ := 20)
  (westwards2 : ℕ := 20) 
  (home_distance : ℕ := 30)

-- Main theorem statement without proof
theorem initial_southwards_distance (j : Journey) : j.southwards + j.northwards = j.home_distance → j.southwards = 10 := by
  intro h
  sorry

end initial_southwards_distance_l272_272528


namespace cost_price_article_l272_272570
-- Importing the required library

-- Definition of the problem
theorem cost_price_article
  (C S C_new S_new : ℝ)
  (h1 : S = 1.05 * C)
  (h2 : C_new = 0.95 * C)
  (h3 : S_new = S - 1)
  (h4 : S_new = 1.045 * C) :
  C = 200 :=
by
  -- The proof is omitted
  sorry

end cost_price_article_l272_272570


namespace correct_option_C_correct_option_D_l272_272705

-- definitions representing the conditions
def A_inequality (x : ℝ) : Prop := (2 * x + 1) / (3 - x) ≤ 0
def B_inequality (x : ℝ) : Prop := (2 * x + 1) * (3 - x) ≥ 0
def C_inequality (x : ℝ) : Prop := (2 * x + 1) / (3 - x) ≥ 0
def D_inequality (x : ℝ) : Prop := (2 * x + 1) / (x - 3) ≤ 0
def solution_set (x : ℝ) : Prop := (-1 / 2 ≤ x ∧ x < 3)

-- proving that option C is equivalent to the solution set
theorem correct_option_C : ∀ x : ℝ, C_inequality x ↔ solution_set x :=
by sorry

-- proving that option D is equivalent to the solution set
theorem correct_option_D : ∀ x : ℝ, D_inequality x ↔ solution_set x :=
by sorry

end correct_option_C_correct_option_D_l272_272705


namespace number_of_valid_arrangements_l272_272993

open Finset

-- We define the condition that a list is sorted in ascending order
def is_ascending (l : List ℕ) : Prop :=
  l = List.sort (≤) l

-- We define the condition that a list is sorted in descending order
def is_descending (l : List ℕ) : Prop :=
  l = List.sort (≥) l

def cards := Finset.range 7
def arrangements := cards.to_list.permutations

-- Define the function to check if a list of numbers (cards) 
-- can have one element removed to form an ascending or descending list
def valid_arrangement (l : List ℕ) : Prop :=
  ∃ (x : ℕ), (l.erase x).is_ascending ∨ (l.erase x).is_descending

-- Define the final theorem
theorem number_of_valid_arrangements : finset.card (arrangements.filter valid_arrangement) = 72 :=
by
  sorry

end number_of_valid_arrangements_l272_272993


namespace intersection_is_one_l272_272618

def M : Set ℝ := {x | x - 1 = 0}
def N : Set ℝ := {x | x^2 - 3 * x + 2 = 0}

theorem intersection_is_one : M ∩ N = {1} :=
by
  sorry

end intersection_is_one_l272_272618


namespace inequality_am_gm_l272_272214

theorem inequality_am_gm (x : ℝ) (hx : x > 0) : x + 1/x ≥ 2 := by
  sorry

end inequality_am_gm_l272_272214


namespace negation_of_universal_sin_pos_l272_272683

theorem negation_of_universal_sin_pos :
  ¬ (∀ x : ℝ, Real.sin x > 0) ↔ ∃ x : ℝ, Real.sin x ≤ 0 :=
by sorry

end negation_of_universal_sin_pos_l272_272683


namespace squirrel_population_difference_l272_272405

theorem squirrel_population_difference :
  ∀ (total_population scotland_population rest_uk_population : ℕ), 
  scotland_population = 120000 →
  120000 = 75 * total_population / 100 →
  rest_uk_population = total_population - scotland_population →
  scotland_population - rest_uk_population = 80000 :=
by
  intros total_population scotland_population rest_uk_population h1 h2 h3
  sorry

end squirrel_population_difference_l272_272405


namespace nth_monomial_is_correct_l272_272887

-- conditions
def coefficient (n : ℕ) : ℕ := 2 * n - 1
def exponent (n : ℕ) : ℕ := n
def monomial (n : ℕ) : ℕ × ℕ := (coefficient n, exponent n)

-- theorem to prove the nth monomial
theorem nth_monomial_is_correct (n : ℕ) : monomial n = (2 * n - 1, n) := 
by 
    sorry

end nth_monomial_is_correct_l272_272887


namespace least_subtracted_12702_is_26_l272_272420

theorem least_subtracted_12702_is_26 : 12702 % 99 = 26 :=
by
  sorry

end least_subtracted_12702_is_26_l272_272420


namespace max_sum_product_l272_272045

theorem max_sum_product (a b c d : ℝ) (h_nonneg: a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0) (h_sum: a + b + c + d = 200) : 
  ab + bc + cd + da ≤ 10000 := 
sorry

end max_sum_product_l272_272045


namespace number_wall_top_block_value_l272_272958

theorem number_wall_top_block_value (a b c d : ℕ) 
    (h1 : a = 8) (h2 : b = 5) (h3 : c = 3) (h4 : d = 2) : 
    (a + b + (b + c) + (c + d) = 34) :=
by
  sorry

end number_wall_top_block_value_l272_272958


namespace det_B_squared_minus_3B_l272_272972

open Matrix
open Real

variable {α : Type*} [Fintype α] {n : ℕ}
variable [DecidableEq α]

noncomputable def B : Matrix (Fin 2) (Fin 2) ℝ := ![
  ![2, 4],
  ![1, 3]
]

theorem det_B_squared_minus_3B : det (B * B - 3 • B) = -8 := sorry

end det_B_squared_minus_3B_l272_272972


namespace directrix_of_given_parabola_l272_272144

noncomputable def parabola_directrix (y : ℝ) : Prop :=
  let focus_x : ℝ := -1
  let point_on_parabola : ℝ × ℝ := (-1 / 4 * y^2, y)
  let PF_sq := (point_on_parabola.1 - focus_x)^2 + (point_on_parabola.2)^2
  let PQ_sq := (point_on_parabola.1 - 1)^2
  PF_sq = PQ_sq

theorem directrix_of_given_parabola : parabola_directrix y :=
sorry

end directrix_of_given_parabola_l272_272144


namespace coin_flip_probability_l272_272346

-- Define the required variables and conditions
def n : ℕ := 3 -- number of flips
def k : ℕ := 2 -- number of desired heads
def p : ℝ := 0.5 -- probability of getting heads

-- Binomial coefficient calculation
def binomial_coeff (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Statement of the problem
theorem coin_flip_probability :
  (binomial_coeff n k) * (p ^ k) * ((1 - p) ^ (n - k)) = 0.375 :=
by
  -- Proof goes here
  sorry

end coin_flip_probability_l272_272346


namespace cone_height_l272_272477

theorem cone_height (l : ℝ) (LA : ℝ) (h : ℝ) (r : ℝ) (h_eq : h = sqrt (l^2 - r^2))
  (LA_eq : LA = π * r * l) (l_val : l = 13) (LA_val : LA = 65 * π) : h = 12 :=
by
  -- substitution of the values of l and LA
  have l_13 := l_val,
  have LA_65π := LA_val,
  
  -- solve for r from LA = π * r * l
  have r_val : r = LA / (π * l), sorry,

  -- then use the Pythagorean theorem to solve for h
  have h_12 : h = sqrt (l^2 - r^2), sorry,

  -- final conclusion: h must be equal to 12
  exact sorry

end cone_height_l272_272477


namespace parabola_directrix_l272_272135

theorem parabola_directrix : 
  ∃ d : ℝ, (∀ (y : ℝ), x = - (1 / 4) * y^2 -> 
  ( - (1 / 4) * y^2 = d)) ∧ d = 1 :=
by
  sorry

end parabola_directrix_l272_272135


namespace total_sum_is_2696_l272_272595

def numbers := (100, 4900)

def harmonic_mean (a b : ℕ) : ℕ :=
  2 * a * b / (a + b)

def arithmetic_mean (a b : ℕ) : ℕ :=
  (a + b) / 2

theorem total_sum_is_2696 : 
  harmonic_mean numbers.1 numbers.2 + arithmetic_mean numbers.1 numbers.2 = 2696 :=
by
  sorry

end total_sum_is_2696_l272_272595


namespace sqrt_12_lt_4_l272_272305

theorem sqrt_12_lt_4 : Real.sqrt 12 < 4 := sorry

end sqrt_12_lt_4_l272_272305


namespace central_angle_radian_measure_l272_272544

-- Define the unit circle radius
def unit_circle_radius : ℝ := 1

-- Given an arc of length 1
def arc_length : ℝ := 1

-- Problem Statement: Prove that the radian measure of the central angle α is 1
theorem central_angle_radian_measure :
  ∀ (r : ℝ) (l : ℝ), r = unit_circle_radius → l = arc_length → |l / r| = 1 :=
by
  intros r l hr hl
  rw [hr, hl]
  sorry

end central_angle_radian_measure_l272_272544


namespace three_numbers_equal_l272_272245

theorem three_numbers_equal {a b c d : ℕ} 
  (h : ∀ {x y z w : ℕ}, (x = a ∨ x = b ∨ x = c ∨ x = d) ∧ (y = a ∨ y = b ∨ y = c ∨ y = d) ∧
                  (z = a ∨ z = b ∨ z = c ∨ z = d) ∧ (w = a ∨ w = b ∨ w = c ∨ w = d) → x * y + z * w = x * z + y * w) :
  a = b ∨ a = c ∨ a = d ∨ b = c ∨ b = d ∨ c = d :=
sorry

end three_numbers_equal_l272_272245


namespace total_interest_correct_l272_272374

-- Initial conditions
def initial_investment : ℝ := 1200
def annual_interest_rate : ℝ := 0.08
def additional_deposit : ℝ := 500
def first_period : ℕ := 2
def second_period : ℕ := 2

-- Calculate the accumulated value after the first period
def first_accumulated_value : ℝ := initial_investment * (1 + annual_interest_rate)^first_period

-- Calculate the new principal after additional deposit
def new_principal := first_accumulated_value + additional_deposit

-- Calculate the accumulated value after the second period
def final_value := new_principal * (1 + annual_interest_rate)^second_period

-- Calculate the total interest earned after 4 years
def total_interest_earned := final_value - initial_investment - additional_deposit

-- Final theorem statement to be proven
theorem total_interest_correct : total_interest_earned = 515.26 :=
by sorry

end total_interest_correct_l272_272374


namespace range_of_a_for_increasing_function_l272_272072

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (6 - a) * x - 2 * a else a ^ x

theorem range_of_a_for_increasing_function (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x ≤ f a y) ↔ (3/2 ≤ a ∧ a < 6) := sorry

end range_of_a_for_increasing_function_l272_272072


namespace value_of_otimes_l272_272442

variable (a b : ℚ)

/-- Define the operation ⊗ -/
def otimes (x y : ℚ) : ℚ := a^2 * x + b * y - 3

/-- Given conditions -/
axiom condition1 : otimes a b 1 (-3) = 2 

/-- Target proof -/
theorem value_of_otimes : otimes a b 2 (-6) = 7 :=
by
  sorry

end value_of_otimes_l272_272442


namespace cost_of_door_tickets_l272_272436

theorem cost_of_door_tickets (x : ℕ) 
  (advanced_purchase_cost : ℕ)
  (total_tickets : ℕ)
  (total_revenue : ℕ)
  (advanced_tickets_sold : ℕ)
  (total_revenue_advanced : ℕ := advanced_tickets_sold * advanced_purchase_cost)
  (door_tickets_sold : ℕ := total_tickets - advanced_tickets_sold) : 
  advanced_purchase_cost = 8 ∧
  total_tickets = 140 ∧
  total_revenue = 1720 ∧
  advanced_tickets_sold = 100 →
  door_tickets_sold * x + total_revenue_advanced = total_revenue →
  x = 23 := 
by
  intros h1 h2
  sorry

end cost_of_door_tickets_l272_272436


namespace solution_range_of_a_l272_272008

theorem solution_range_of_a (a : ℝ) (x y : ℝ) :
  3 * x + y = 1 + a → x + 3 * y = 3 → x + y < 2 → a < 4 :=
by
  sorry

end solution_range_of_a_l272_272008


namespace intersection_M_N_l272_272042

def M := {m : ℤ | -3 < m ∧ m < 2}
def N := {x : ℤ | x * (x - 1) = 0}

theorem intersection_M_N : M ∩ N = {0, 1} := sorry

end intersection_M_N_l272_272042


namespace min_cards_for_certain_event_l272_272155

-- Let's define the deck configuration
structure DeckConfig where
  spades : ℕ
  clubs : ℕ
  hearts : ℕ
  total : ℕ

-- Define the given condition of the deck
def givenDeck : DeckConfig := { spades := 5, clubs := 4, hearts := 6, total := 15 }

-- Predicate to check if m cards drawn guarantees all three suits are present
def is_certain_event (m : ℕ) (deck : DeckConfig) : Prop :=
  m >= deck.spades + deck.hearts + 1

-- The main theorem to prove the minimum number of cards m
theorem min_cards_for_certain_event : ∀ m, is_certain_event m givenDeck ↔ m = 12 :=
by
  sorry

end min_cards_for_certain_event_l272_272155


namespace international_postage_surcharge_l272_272308

theorem international_postage_surcharge 
  (n_letters : ℕ) 
  (std_postage_per_letter : ℚ) 
  (n_international : ℕ) 
  (total_cost : ℚ) 
  (cents_per_dollar : ℚ) 
  (std_total_cost : ℚ) 
  : 
  n_letters = 4 →
  std_postage_per_letter = 108 / 100 →
  n_international = 2 →
  total_cost = 460 / 100 →
  cents_per_dollar = 100 →
  std_total_cost = n_letters * std_postage_per_letter →
  (total_cost - std_total_cost) / n_international * cents_per_dollar = 14 := 
sorry

end international_postage_surcharge_l272_272308


namespace Jordan_Lee_debt_equal_l272_272971

theorem Jordan_Lee_debt_equal (initial_debt_jordan : ℝ) (additional_debt_jordan : ℝ)
  (rate_jordan : ℝ) (initial_debt_lee : ℝ) (rate_lee : ℝ) :
  initial_debt_jordan + additional_debt_jordan + (initial_debt_jordan + additional_debt_jordan) * rate_jordan * 33.333333333333336 
  = initial_debt_lee + initial_debt_lee * rate_lee * 33.333333333333336 :=
by
  let t := 33.333333333333336
  have rate_jordan := 0.12
  have rate_lee := 0.08
  have initial_debt_jordan := 200
  have additional_debt_jordan := 20
  have initial_debt_lee := 300
  sorry

end Jordan_Lee_debt_equal_l272_272971


namespace union_sets_l272_272000

-- Define the sets A and B based on the given conditions
def set_A : Set ℝ := {x | abs (x - 1) < 2}
def set_B : Set ℝ := {x | Real.log x / Real.log 2 < 3}

-- Problem statement: Prove that the union of sets A and B is {x | -1 < x < 9}
theorem union_sets : (set_A ∪ set_B) = {x | -1 < x ∧ x < 9} :=
by
  sorry

end union_sets_l272_272000


namespace initial_balance_l272_272860

theorem initial_balance (B : ℝ) (payment : ℝ) (new_balance : ℝ)
  (h1 : payment = 50) (h2 : new_balance = 120) (h3 : B - payment = new_balance) :
  B = 170 :=
by
  rw [h1, h2] at h3
  linarith

end initial_balance_l272_272860


namespace average_selling_price_is_86_l272_272290

def selling_prices := [82, 86, 90, 85, 87, 85, 86, 82, 90, 87, 85, 86, 82, 86, 87, 90]

def average (prices : List Nat) : Nat :=
  (prices.sum) / prices.length

theorem average_selling_price_is_86 :
  average selling_prices = 86 :=
by
  sorry

end average_selling_price_is_86_l272_272290


namespace calculate_PC_l272_272635
noncomputable def ratio (a b : ℝ) : ℝ := a / b

theorem calculate_PC (AB BC CA PC PA : ℝ) (h1: AB = 6) (h2: BC = 10) (h3: CA = 8)
  (h4: ratio PC PA = ratio 8 6)
  (h5: ratio PA (PC + 10) = ratio 6 10) :
  PC = 40 :=
sorry

end calculate_PC_l272_272635


namespace slope_of_line_l272_272793

-- Define the parabola C
def parabola (x y : ℝ) : Prop := y^2 = 4 * x

-- Define the focus F of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define the line l intersecting the parabola C at points A and B
def line (k x : ℝ) : ℝ := k * (x - 1)

-- Condition based on the intersection and the given relationship 2 * (BF) = FA
def intersection_condition (k : ℝ) (A B : ℝ × ℝ) : Prop :=
  ∃ x1 x2 y1 y2,
    A = (x1, y1) ∧ B = (x2, y2) ∧
    parabola x1 y1 ∧ parabola x2 y2 ∧
    (y1 = line k x1) ∧ (y2 = line k x2) ∧
    2 * (dist (x2, y2) focus) = dist focus (x1, y1)

-- The main theorem to be proven
theorem slope_of_line (k : ℝ) (A B : ℝ × ℝ) :
  intersection_condition k A B → k = 2 * Real.sqrt 2 :=
sorry

end slope_of_line_l272_272793


namespace percentage_in_first_subject_l272_272732

theorem percentage_in_first_subject (P : ℝ) (H1 : 80 = 80) (H2 : 75 = 75) (H3 : (P + 80 + 75) / 3 = 75) : P = 70 :=
by
  sorry

end percentage_in_first_subject_l272_272732


namespace range_a_empty_intersection_range_a_sufficient_condition_l272_272334

noncomputable def A (x : ℝ) : Prop := -10 < x ∧ x < 2
noncomputable def B (x : ℝ) (a : ℝ) : Prop := x ≥ 1 + a ∨ x ≤ 1 - a
noncomputable def A_inter_B_empty (a : ℝ) : Prop := ∀ x : ℝ, A x → ¬ B x a
noncomputable def neg_p (x : ℝ) : Prop := x ≥ 2 ∨ x ≤ -10
noncomputable def neg_p_implies_q (a : ℝ) : Prop := ∀ x : ℝ, neg_p x → B x a

theorem range_a_empty_intersection : (∀ x : ℝ, A x → ¬ B x 11) → 11 ≤ a := by
  sorry

theorem range_a_sufficient_condition : (∀ x : ℝ, neg_p x → B x 1) → 0 < a ∧ a ≤ 1 := by
  sorry

end range_a_empty_intersection_range_a_sufficient_condition_l272_272334


namespace next_bell_ring_time_l272_272880

theorem next_bell_ring_time :
  let church_interval := 15
  let school_interval := 20
  let daycare_interval := 25
  let lcm_intervals := Nat.lcm church_interval (Nat.lcm school_interval daycare_interval)
  lcm_intervals = 300 →
  "05:00" := by
  sorry

end next_bell_ring_time_l272_272880


namespace largest_n_divisible_l272_272248

theorem largest_n_divisible : ∃ n : ℕ, (∀ k : ℕ, (k^3 + 150) % (k + 5) = 0 → k ≤ n) ∧ n = 20 := 
by
  sorry

end largest_n_divisible_l272_272248


namespace sum_S5_l272_272078

-- Geometric sequence definitions and conditions
noncomputable def geometric_sequence (a : ℝ) (r : ℝ) (n : ℕ) : ℝ := a * r^n

noncomputable def sum_of_geometric_sequence (a : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r^n) / (1 - r)

variables (a r : ℝ)

-- Given conditions translated into Lean:
-- a2 * a3 = 2 * a1
def condition1 := (geometric_sequence a r 1) * (geometric_sequence a r 2) = 2 * a

-- Arithmetic mean of a4 and 2 * a7 is 5/4
def condition2 := (geometric_sequence a r 3 + 2 * geometric_sequence a r 6) / 2 = 5 / 4

-- The final goal proving that S5 = 31
theorem sum_S5 (h1 : condition1 a r) (h2 : condition2 a r) : sum_of_geometric_sequence a r 5 = 31 := by
  apply sorry

end sum_S5_l272_272078


namespace negation_of_p_implies_a_gt_one_half_l272_272487

-- Define the proposition p
def p (a : ℝ) : Prop := ∃ x : ℝ, a * x^2 + x + 1 / 2 ≤ 0

-- Define the statement that negation of p implies a > 1/2
theorem negation_of_p_implies_a_gt_one_half (a : ℝ) (h : ¬ p a) : a > 1 / 2 :=
by
  sorry

end negation_of_p_implies_a_gt_one_half_l272_272487


namespace perfect_square_formula_l272_272413

theorem perfect_square_formula (x y : ℝ) :
  ¬∃ a b : ℝ, (x^2 + (1/4)*x + (1/4)) = (a + b)^2 ∧
  ¬∃ c d : ℝ, (x^2 + 2*x*y - y^2) = (c + d)^2 ∧
  ¬∃ e f : ℝ, (x^2 + x*y + y^2) = (e + f)^2 ∧
  ∃ g h : ℝ, (4*x^2 + 4*x + 1) = (g + h)^2 :=
sorry

end perfect_square_formula_l272_272413


namespace compare_f_values_l272_272468

variable (a : ℝ) (f : ℝ → ℝ) (m n : ℝ)

theorem compare_f_values (h_a : 0 < a ∧ a < 1)
    (h_f : ∀ x > 0, f (Real.logb a x) = a * (x^2 - 1) / (x * (a^2 - 1)))
    (h_mn : m > n ∧ n > 0 ∧ m > 0) :
    f (1 / n) > f (1 / m) := by 
  sorry

end compare_f_values_l272_272468


namespace find_original_number_l272_272289

-- Variables and assumptions
variables (N y x : ℕ)

-- Conditions
def crossing_out_condition : Prop := N = 10 * y + x
def sum_condition : Prop := N + y = 54321
def remainder_condition : Prop := x = 54321 % 11
def y_value : Prop := y = 4938

-- The final proof problem statement
theorem find_original_number (h1 : crossing_out_condition N y x) (h2 : sum_condition N y) 
  (h3 : remainder_condition x) (h4 : y_value y) : N = 49383 :=
sorry

end find_original_number_l272_272289


namespace jacob_age_in_X_years_l272_272967

-- Definitions of the conditions
variable (J M X : ℕ)

theorem jacob_age_in_X_years
  (h1 : J = M - 14)
  (h2 : M + 9 = 2 * (J + 9))
  (h3 : J = 5) :
  J + X = 5 + X :=
by
  sorry

end jacob_age_in_X_years_l272_272967


namespace range_of_a_range_of_m_l272_272792

def f (x : ℝ) : ℝ := |2 * x + 1| + |2 * x - 3|

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, f x < |1 - 2 * a|) ↔ a ∈ (Set.Iic (-3/2) ∪ Set.Ici (5/2)) := by sorry

theorem range_of_m (m : ℝ) : 
  (∃ t : ℝ, t^2 - 2 * Real.sqrt 6 * t + f m = 0) ↔ m ∈ (Set.Icc (-1) 2) := by sorry

end range_of_a_range_of_m_l272_272792


namespace length_second_platform_l272_272432

-- Define the conditions
def length_train : ℕ := 100
def time_platform1 : ℕ := 15
def length_platform1 : ℕ := 350
def time_platform2 : ℕ := 20

-- Prove the length of the second platform is 500m
theorem length_second_platform : ∀ (speed_train : ℚ), 
  speed_train = (length_train + length_platform1) / time_platform1 →
  (speed_train = (length_train + L) / time_platform2) → 
  L = 500 :=
by 
  intro speed_train h1 h2
  sorry

end length_second_platform_l272_272432


namespace least_positive_integer_property_l272_272326

theorem least_positive_integer_property : 
  ∃ (n d : ℕ) (p : ℕ) (h₁ : 1 ≤ d) (h₂ : d ≤ 9) (h₃ : p ≥ 2), 
  (10^p * d = 24 * n) ∧ (∃ k : ℕ, (n = 100 * 10^(p-2) / 3) ∧ (900 = 8 * 10^p + 100 / 3 * 10^(p-2))) := sorry

end least_positive_integer_property_l272_272326


namespace inequality_proof_l272_272506

theorem inequality_proof (a b : ℤ) (ha : a > 0) (hb : b > 0) : a + b ≤ 1 + a * b :=
by
  sorry

end inequality_proof_l272_272506


namespace number_of_different_teams_l272_272954

namespace DoctorTeam

-- Conditions
variables (total_doctors pediatricians surgeons general_practitioners : ℕ)
          (team_size : ℕ) (at_least_one_pead pediatrician_choice surgeon_choice general_practitioner_choice other_choices : ℕ)

-- Define specific numbers as per the problem
def conditions := 
  total_doctors = 25 ∧
  pediatricians = 5 ∧
  surgeons = 10 ∧
  general_practitioners = 10 ∧
  team_size = 5 ∧
  at_least_one_pead = 1 ∧ 
  pediatrician_choice = (choose pediatricians at_least_one_pead).val ∧
  surgeon_choice = (choose surgeons at_least_one_pead).val ∧
  general_practitioner_choice = (choose general_practitioners at_least_one_pead).val ∧
  other_choices = (choose (total_doctors - (at_least_one_pead * 3)) (team_size - (at_least_one_pead * 3))).val

-- The proof problem
theorem number_of_different_teams {total_doctors pediatricians surgeons general_practitioners team_size at_least_one_pead pediatrician_choice surgeon_choice general_practitioner_choice other_choices : ℕ} :
  conditions total_doctors pediatricians surgeons general_practitioners team_size at_least_one_pead pediatrician_choice surgeon_choice general_practitioner_choice other_choices →
  (pediatrician_choice * surgeon_choice * general_practitioner_choice * other_choices) = 115500 :=
by
  sorry

end DoctorTeam

end number_of_different_teams_l272_272954


namespace black_female_pigeons_more_than_males_l272_272727

theorem black_female_pigeons_more_than_males:
  let total_pigeons := 70
  let black_pigeons := total_pigeons / 2
  let black_male_percentage := 20 / 100
  let black_male_pigeons := black_pigeons * black_male_percentage
  let black_female_pigeons := black_pigeons - black_male_pigeons
  black_female_pigeons - black_male_pigeons = 21 := by
{
  let total_pigeons := 70
  let black_pigeons := total_pigeons / 2
  let black_male_percentage := 20 / 100
  let black_male_pigeons := black_pigeons * black_male_percentage
  let black_female_pigeons := black_pigeons - black_male_pigeons
  show black_female_pigeons - black_male_pigeons = 21
  sorry
}

end black_female_pigeons_more_than_males_l272_272727


namespace initial_pigs_l272_272242

theorem initial_pigs (x : ℕ) (h : x + 86 = 150) : x = 64 :=
by
  sorry

end initial_pigs_l272_272242


namespace find_c_l272_272942

-- Define the polynomial f(x)
def f (c : ℚ) (x : ℚ) : ℚ := 2 * c * x^3 + 14 * x^2 - 6 * c * x + 25

-- State the problem in Lean 4
theorem find_c (c : ℚ) : (∀ x : ℚ, f c x = 0 ↔ x = (-5)) → c = 75 / 44 := 
by sorry

end find_c_l272_272942


namespace sum_of_squares_l272_272948

theorem sum_of_squares (x y : ℝ) (h1 : x + y = 18) (h2 : x * y = 52) : x^2 + y^2 = 220 :=
sorry

end sum_of_squares_l272_272948


namespace simplify_expression_l272_272091

theorem simplify_expression :
  (3 / 4 : ℚ) * 60 - (8 / 5 : ℚ) * 60 + x = 12 → x = 63 :=
by
  intro h
  sorry

end simplify_expression_l272_272091


namespace perpendicular_vectors_l272_272173

/-- If vectors a = (1, 2) and b = (x, 4) are perpendicular, then x = -8. -/
theorem perpendicular_vectors (x : ℝ) (a b : ℝ × ℝ) 
  (ha : a = (1, 2)) (hb : b = (x, 4)) (h_perp : a.1 * b.1 + a.2 * b.2 = 0) : x = -8 :=
by {
  sorry
}

end perpendicular_vectors_l272_272173


namespace total_weight_proof_l272_272037
-- Import the entire math library

-- Assume the conditions as given variables
variables (w r s : ℕ)
-- Assign values to the given conditions
def weight_per_rep := 15
def reps_per_set := 10
def number_of_sets := 3

-- Calculate total weight moved
def total_weight_moved := w * r * s

-- The theorem to prove the total weight moved
theorem total_weight_proof : total_weight_moved weight_per_rep reps_per_set number_of_sets = 450 :=
by
  -- Provide the expected result directly, proving the statement
  sorry

end total_weight_proof_l272_272037


namespace pages_to_read_l272_272823

variable (E P_Science P_Civics P_Chinese Total : ℕ)
variable (h_Science : P_Science = 16)
variable (h_Civics : P_Civics = 8)
variable (h_Chinese : P_Chinese = 12)
variable (h_Total : Total = 14)

theorem pages_to_read :
  (E / 4) + (P_Science / 4) + (P_Civics / 4) + (P_Chinese / 4) = Total → 
  E = 20 := by
  sorry

end pages_to_read_l272_272823


namespace find_coordinates_of_C_l272_272809

structure Point where
  x : ℝ
  y : ℝ

def parallelogram (A B C D : Point) : Prop :=
  (B.x - A.x = C.x - D.x ∧ B.y - A.y = C.y - D.y) ∧
  (D.x - A.x = C.x - B.x ∧ D.y - A.y = C.y - B.y)

def A : Point := ⟨2, 3⟩
def B : Point := ⟨7, 3⟩
def D : Point := ⟨3, 7⟩
def C : Point := ⟨8, 7⟩

theorem find_coordinates_of_C :
  parallelogram A B C D → C = ⟨8, 7⟩ :=
by
  intro h
  have h₁ := h.1.1
  have h₂ := h.1.2
  have h₃ := h.2.1
  have h₄ := h.2.2
  sorry

end find_coordinates_of_C_l272_272809


namespace angle_CDB_45_degrees_l272_272889

theorem angle_CDB_45_degrees
  (α β γ δ : ℝ)
  (triangle_isosceles_right : α = β)
  (triangle_angle_BCD : γ = 90)
  (square_angle_DCE : δ = 90)
  (triangle_angle_ABC : α = β)
  (isosceles_triangle_angle : α + β + γ = 180)
  (isosceles_triangle_right : α = 45)
  (isosceles_triangle_sum : α + α + 90 = 180)
  (square_geometry : δ = 90) :
  γ + δ = 180 →  180 - (γ + α) = 45 :=
by
  sorry

end angle_CDB_45_degrees_l272_272889


namespace emily_total_beads_l272_272318

-- Let's define the given conditions
def necklaces : ℕ := 11
def beads_per_necklace : ℕ := 28

-- The statement to prove
theorem emily_total_beads : (necklaces * beads_per_necklace) = 308 := by
  sorry

end emily_total_beads_l272_272318


namespace Pam_has_740_fruits_l272_272213

/-
Define the given conditions.
-/
def Gerald_apple_bags : ℕ := 5
def apples_per_Gerald_bag : ℕ := 30
def Gerald_orange_bags : ℕ := 4
def oranges_per_Gerald_bag : ℕ := 25

def Pam_apple_bags : ℕ := 6
def apples_per_Pam_bag : ℕ := 3 * apples_per_Gerald_bag
def Pam_orange_bags : ℕ := 4
def oranges_per_Pam_bag : ℕ := 2 * oranges_per_Gerald_bag

/-
Proving the total number of apples and oranges Pam has.
-/
def total_fruits_Pam : ℕ :=
    Pam_apple_bags * apples_per_Pam_bag + Pam_orange_bags * oranges_per_Pam_bag

theorem Pam_has_740_fruits : total_fruits_Pam = 740 := by
  sorry

end Pam_has_740_fruits_l272_272213


namespace double_average_l272_272419

theorem double_average (n : ℕ) (initial_avg : ℝ) (new_avg : ℝ) (h1 : n = 25) (h2 : initial_avg = 70) (h3 : new_avg * n = 2 * (initial_avg * n)) : new_avg = 140 :=
sorry

end double_average_l272_272419


namespace ten_percent_markup_and_markdown_l272_272114

theorem ten_percent_markup_and_markdown (x : ℝ) (hx : x > 0) : 0.99 * x < x :=
by 
  sorry

end ten_percent_markup_and_markdown_l272_272114


namespace min_val_expression_l272_272841

theorem min_val_expression (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a^2 * b + b^2 * c + c^2 * a = 3) : 
  a^7 * b + b^7 * c + c^7 * a + a * b^3 + b * c^3 + c * a^3 ≥ 6 :=
sorry

end min_val_expression_l272_272841


namespace p_is_sufficient_not_necessary_for_q_l272_272608

-- Definitions for conditions p and q
def p (x : ℝ) := x^2 - x - 20 > 0
def q (x : ℝ) := 1 - x^2 < 0

-- The main statement
theorem p_is_sufficient_not_necessary_for_q:
  (∀ x, p x → q x) ∧ ¬(∀ x, q x → p x) :=
by
  sorry

end p_is_sufficient_not_necessary_for_q_l272_272608


namespace number_of_ways_to_draw_4_from_15_l272_272098

theorem number_of_ways_to_draw_4_from_15 : 
  let n := 4
  let k := 15
  (∏ i in finset.range n, (k - i)) = 32760 := 
by
  sorry

end number_of_ways_to_draw_4_from_15_l272_272098


namespace inequality_always_holds_l272_272455

theorem inequality_always_holds (m : ℝ) : (-6 < m ∧ m ≤ 0) ↔ ∀ x : ℝ, 2 * m * x^2 + m * x - 3 / 4 < 0 := 
sorry

end inequality_always_holds_l272_272455


namespace minimum_negative_factors_l272_272505

theorem minimum_negative_factors (a b c d : ℝ) (h1 : a * b * c * d < 0) (h2 : a + b = 0) (h3 : c * d > 0) : 
    (∃ x ∈ [a, b, c, d], x < 0) :=
by
  sorry

end minimum_negative_factors_l272_272505


namespace circles_intersect_l272_272407

theorem circles_intersect (m c : ℝ) (h1 : (1:ℝ) = (5 + (-m))) (h2 : (3:ℝ) = (5 + (c - (-2)))) :
  m + c = 3 :=
sorry

end circles_intersect_l272_272407


namespace find_xy_l272_272299

-- Defining the initial conditions
variable (x y : ℕ)

-- Defining the rectangular prism dimensions and the volume equation
def prism_volume_original : ℕ := 15 * 5 * 4 -- Volume = 300
def remaining_volume : ℕ := 120

-- The main theorem statement to prove the conditions and their solution
theorem find_xy (h1 : prism_volume_original - 5 * y * x = remaining_volume)
    (h2 : x < 4) 
    (h3 : y < 15) : 
    x = 3 ∧ y = 12 := sorry

end find_xy_l272_272299


namespace simplify_expression_l272_272674

variable {x y : ℝ}
variable (h : x * y ≠ 0)

theorem simplify_expression (h : x * y ≠ 0) :
  ((x^3 + 1) / x) * ((y^2 + 1) / y) - ((x^2 - 1) / y) * ((y^3 - 1) / x) =
  (x^3*y^2 - x^2*y^3 + x^3 + x^2 + y^2 + y^3) / (x*y) :=
by sorry

end simplify_expression_l272_272674


namespace range_of_m_l272_272347

theorem range_of_m (m : ℝ) : (∀ x : ℝ, x^2 - m * x + m > 0) ↔ 0 < m ∧ m < 4 :=
by sorry

end range_of_m_l272_272347


namespace production_average_l272_272456

theorem production_average (n : ℕ) (P : ℕ) (hP : P = n * 50)
  (h1 : (P + 95) / (n + 1) = 55) : n = 8 :=
by
  -- skipping the proof
  sorry

end production_average_l272_272456


namespace arithmetic_sequence_a8_l272_272190

theorem arithmetic_sequence_a8 (a : ℕ → ℤ) (h_arith : ∀ n m, a (n + 1) - a n = a m + 1 - a m) 
  (h1 : a 2 = 3) (h2 : a 5 = 12) : a 8 = 21 := 
by 
  sorry

end arithmetic_sequence_a8_l272_272190


namespace four_digit_div_90_count_l272_272931

theorem four_digit_div_90_count :
  ∃ n : ℕ, n = 10 ∧ (∀ (ab : ℕ), 10 ≤ ab ∧ ab ≤ 99 → ab % 9 = 0 → 
  (10 * ab + 90) % 90 = 0 ∧ 1000 ≤ 10 * ab + 90 ∧ 10 * ab + 90 < 10000) :=
sorry

end four_digit_div_90_count_l272_272931


namespace undefined_expression_l272_272915

theorem undefined_expression (a : ℝ) : (a = 3 ∨ a = -3) ↔ (a^2 - 9 = 0) := 
by
  sorry

end undefined_expression_l272_272915


namespace parallel_vectors_implies_x_value_l272_272794

variable (x : ℝ)

def vec_a : ℝ × ℝ := (1, 2)
def vec_b (x : ℝ) : ℝ × ℝ := (x, 1)
def scalar_mul (c : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (c * v.1, c * v.2)
def vec_add (v1 v2 : ℝ × ℝ) : ℝ × ℝ := (v1.1 + v2.1, v1.2 + v2.2)
def vec_sub (v1 v2 : ℝ × ℝ) : ℝ × ℝ := (v1.1 - v2.1, v1.2 - v2.2)

theorem parallel_vectors_implies_x_value :
  (∃ k : ℝ, vec_add vec_a (scalar_mul 2 (vec_b x)) = scalar_mul k (vec_sub (scalar_mul 2 vec_a) (scalar_mul 2 (vec_b x)))) →
  x = 1 / 2 :=
by
  sorry

end parallel_vectors_implies_x_value_l272_272794


namespace selling_price_of_bracelet_l272_272195

theorem selling_price_of_bracelet (x : ℝ) 
  (cost_per_bracelet : ℝ) 
  (num_bracelets : ℕ) 
  (box_of_cookies_cost : ℝ) 
  (money_left_after_buying_cookies : ℝ) 
  (total_revenue : ℝ) 
  (total_cost_of_supplies : ℝ) :
  cost_per_bracelet = 1 →
  num_bracelets = 12 →
  box_of_cookies_cost = 3 →
  money_left_after_buying_cookies = 3 →
  total_cost_of_supplies = cost_per_bracelet * num_bracelets →
  total_revenue = 9 →
  x = total_revenue / num_bracelets :=
by
  intros h1 h2 h3 h4 h5 h6
  -- Placeholder for the actual proof
  sorry

end selling_price_of_bracelet_l272_272195


namespace cookies_total_is_60_l272_272827

def Mona_cookies : ℕ := 20
def Jasmine_cookies : ℕ := Mona_cookies - 5
def Rachel_cookies : ℕ := Jasmine_cookies + 10
def Total_cookies : ℕ := Mona_cookies + Jasmine_cookies + Rachel_cookies

theorem cookies_total_is_60 : Total_cookies = 60 := by
  sorry

end cookies_total_is_60_l272_272827


namespace largest_indecomposable_amount_l272_272356

theorem largest_indecomposable_amount (n : ℕ) : 
  ∀ s, ¬(∃ k : ℕ, s = k * 5 ^ (n + 1) - 2 * 3 ^ (n + 1)) → 
       ¬(∃ (m : ℕ), m < 5 ∧ ∃ (r : ℕ), s = 5 * r + m * 3) :=
by
  intro s h_decomposable
  sorry

end largest_indecomposable_amount_l272_272356


namespace valid_digit_distribution_l272_272556

theorem valid_digit_distribution (n : ℕ) : 
  (∃ (d1 d2 d5 others : ℕ), 
    d1 = n / 2 ∧
    d2 = n / 5 ∧
    d5 = n / 5 ∧
    others = n / 10 ∧
    d1 + d2 + d5 + others = n) :=
by
  sorry

end valid_digit_distribution_l272_272556


namespace focus_of_parabola_l272_272394

theorem focus_of_parabola (x y : ℝ) : (y^2 + 4 * x = 0) → (x = -1 ∧ y = 0) :=
by sorry

end focus_of_parabola_l272_272394


namespace minimum_value_l272_272372

open Classical

variable {a b c : ℝ}

theorem minimum_value (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c) (h : a + b + c = 4) :
  36 ≤ (9 / a) + (16 / b) + (25 / c) :=
sorry

end minimum_value_l272_272372


namespace inequality_proof_l272_272204

variable (a b c d : ℝ) (h_nonneg_a : 0 ≤ a) (h_nonneg_b : 0 ≤ b) (h_nonneg_c : 0 ≤ c) (h_nonneg_d : 0 ≤ d)
variable (h_sum : a + b + c + d = 1)

theorem inequality_proof :
  a * b * c + b * c * d + c * d * a + d * a * b ≤ (1 / 27) + (176 / 27) * a * b * c * d :=
by
  sorry

end inequality_proof_l272_272204


namespace amusement_park_ticket_cost_l272_272194

/-- Jeremie is going to an amusement park with 3 friends. 
    The cost of a set of snacks is $5. 
    The total cost for everyone to go to the amusement park and buy snacks is $92.
    Prove that the cost of one ticket is $18.
-/
theorem amusement_park_ticket_cost 
  (number_of_people : ℕ)
  (snack_cost_per_person : ℕ)
  (total_cost : ℕ)
  (ticket_cost : ℕ) :
  number_of_people = 4 → 
  snack_cost_per_person = 5 → 
  total_cost = 92 → 
  ticket_cost = 18 :=
by
  intros h1 h2 h3
  sorry

end amusement_park_ticket_cost_l272_272194


namespace solve_inequality_l272_272606

noncomputable def solution_set (a b : ℝ) (x : ℝ) : Prop :=
x < -1 / b ∨ x > 1 / a

theorem solve_inequality (a b : ℝ) (x : ℝ)
  (h_a : a > 0) (h_b : b > 0) :
  (-b < 1 / x ∧ 1 / x < a) ↔ solution_set a b x :=
by
  sorry

end solve_inequality_l272_272606


namespace additional_charge_per_international_letter_l272_272309

-- Definitions based on conditions
def standard_postage_per_letter : ℕ := 108
def num_international_letters : ℕ := 2
def total_cost : ℕ := 460
def num_letters : ℕ := 4

-- Theorem stating the question
theorem additional_charge_per_international_letter :
  (total_cost - (num_letters * standard_postage_per_letter)) / num_international_letters = 14 :=
by
  sorry

end additional_charge_per_international_letter_l272_272309


namespace Mr_Caiden_payment_l272_272377

-- Defining the conditions as variables and constants
def total_roofing_needed : ℕ := 300
def cost_per_foot : ℕ := 8
def free_roofing : ℕ := 250

-- Define the remaining roofing needed and the total cost
def remaining_roofing : ℕ := total_roofing_needed - free_roofing
def total_cost : ℕ := remaining_roofing * cost_per_foot

-- The proof statement: 
theorem Mr_Caiden_payment : total_cost = 400 := 
by
  -- Proof omitted
  sorry

end Mr_Caiden_payment_l272_272377


namespace condition_for_a_b_complex_l272_272551

theorem condition_for_a_b_complex (a b : ℂ) (h1 : a ≠ 0) (h2 : 2 * a + b ≠ 0) :
  (2 * a + b) / a = b / (2 * a + b) → 
  (∃ z : ℂ, a = z ∨ b = z) ∨ 
  ((∃ z1 : ℂ, a = z1) ∧ (∃ z2 : ℂ, b = z2)) :=
sorry

end condition_for_a_b_complex_l272_272551


namespace number_divisible_by_20p_l272_272973

noncomputable def floor_expr (p : ℕ) : ℤ :=
  Int.floor ((2 + Real.sqrt 5) ^ p - 2 ^ (p + 1))

theorem number_divisible_by_20p (p : ℕ) (hp : Nat.Prime p ∧ p % 2 = 1) :
  ∃ k : ℤ, floor_expr p = k * 20 * p :=
by
  sorry

end number_divisible_by_20p_l272_272973


namespace total_games_in_single_elimination_tournament_l272_272431

def single_elimination_tournament_games (teams : ℕ) : ℕ :=
teams - 1

theorem total_games_in_single_elimination_tournament :
  single_elimination_tournament_games 23 = 22 :=
by
  sorry

end total_games_in_single_elimination_tournament_l272_272431


namespace simplify_expression_l272_272253

theorem simplify_expression : ( (144^2 - 12^2) / (120^2 - 18^2) * ((120 - 18) * (120 + 18)) / ((144 - 12) * (144 + 12)) ) = 1 :=
by
  sorry

end simplify_expression_l272_272253


namespace average_runs_in_30_matches_l272_272575

theorem average_runs_in_30_matches 
  (avg1 : ℕ) (matches1 : ℕ) (avg2 : ℕ) (matches2 : ℕ) (total_matches : ℕ)
  (h1 : avg1 = 40) (h2 : matches1 = 20) (h3 : avg2 = 13) (h4 : matches2 = 10) (h5 : total_matches = 30) :
  ((avg1 * matches1 + avg2 * matches2) / total_matches) = 31 := by
  sorry

end average_runs_in_30_matches_l272_272575


namespace solve_problem_l272_272250

noncomputable def smallest_positive_integer : ℕ :=
  Inf {n : ℕ | 0 < n ∧ (Real.sqrt n - Real.sqrt (n - 1) < 0.01)}

theorem solve_problem : smallest_positive_integer = 2501 :=
begin
  sorry
end

end solve_problem_l272_272250


namespace initial_girls_count_l272_272677

variable (p : ℕ) -- total number of people initially in the group
variable (girls_initial : ℕ) -- number of girls initially in the group
variable (girls_after : ℕ) -- number of girls after the change
variable (total_after : ℕ) -- total number of people after the change

/--
Initially, 50% of the group are girls. 
Later, five girls leave and five boys arrive, leading to 40% of the group now being girls.
--/
theorem initial_girls_count :
  (girls_initial = p / 2) →
  (total_after = p) →
  (girls_after = girls_initial - 5) →
  (girls_after = 2 * total_after / 5) →
  girls_initial = 25 :=
by
  intros h1 h2 h3 h4
  sorry

end initial_girls_count_l272_272677


namespace reciprocal_of_sum_is_correct_l272_272685

def reciprocal (r : ℚ) : ℚ := 1 / r

theorem reciprocal_of_sum_is_correct :
  reciprocal ((1 : ℚ) / 4 + (1 : ℚ) / 6) = 12 / 5 :=
by
  -- The proof is to be filled in here
  sorry

end reciprocal_of_sum_is_correct_l272_272685


namespace inequality_first_inequality_second_l272_272218

theorem inequality_first (x : ℝ) : 4 * x - 2 < 1 - 2 * x → x < 1 / 2 := 
sorry

theorem inequality_second (x : ℝ) : (3 - 2 * x ≥ x - 6) ∧ ((3 * x + 1) / 2 < 2 * x) → 1 < x ∧ x ≤ 3 :=
sorry

end inequality_first_inequality_second_l272_272218


namespace solution_set_l272_272629

variable {f : ℝ → ℝ}

-- Define that f is an odd function
def odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Define that f is decreasing on positive reals
def decreasing_on_pos_reals (f : ℝ → ℝ) : Prop := ∀ x y, 0 < x → x < y → f y < f x

-- Given conditions
axiom f_odd : odd_function f
axiom f_decreasing : decreasing_on_pos_reals f
axiom f_at_two_zero : f 2 = 0

-- Main theorem statement
theorem solution_set : { x : ℝ | (x - 1) * f (x - 1) > 0 } = { x | x < -1 } ∪ { x | x > 3 } :=
sorry

end solution_set_l272_272629


namespace actual_area_of_region_l272_272837

-- Problem Definitions
def map_scale : ℕ := 300000
def map_area_cm_squared : ℕ := 24

-- The actual area calculation should be 216 km²
theorem actual_area_of_region :
  let scale_factor_distance := map_scale
  let scale_factor_area := scale_factor_distance ^ 2
  let actual_area_cm_squared := map_area_cm_squared * scale_factor_area
  let actual_area_km_squared := actual_area_cm_squared / 10^10
  actual_area_km_squared = 216 := 
by
  sorry

end actual_area_of_region_l272_272837


namespace einstein_birth_weekday_l272_272847

-- Defining the reference day of the week for 31 May 2006
def reference_date := 31
def reference_month := 5
def reference_year := 2006
def reference_weekday := 3  -- Wednesday

-- Defining Albert Einstein's birth date
def einstein_birth_day := 14
def einstein_birth_month := 3
def einstein_birth_year := 1879

-- Defining the calculation of weekday
def weekday_from_reference(reference_day reference_weekday einstein_birth_day einstein_birth_month einstein_birth_year : Nat) : Nat :=
  let days_from_reference_to_birth := 46464  -- Total days calculated in solution
  (reference_weekday - (days_from_reference_to_birth % 7) + 7) % 7

-- Stating the theorem
theorem einstein_birth_weekday : weekday_from_reference reference_day reference_weekday einstein_birth_day einstein_birth_month einstein_birth_year = 5 :=
by
  -- Proof omitted
  sorry

end einstein_birth_weekday_l272_272847


namespace xy_sum_correct_l272_272851

theorem xy_sum_correct (x y : ℝ) 
  (h : (4 + 10 + 16 + 24) / 4 = (14 + x + y) / 3) : 
  x + y = 26.5 :=
by
  sorry

end xy_sum_correct_l272_272851


namespace correct_option_C_correct_option_D_l272_272706

-- definitions representing the conditions
def A_inequality (x : ℝ) : Prop := (2 * x + 1) / (3 - x) ≤ 0
def B_inequality (x : ℝ) : Prop := (2 * x + 1) * (3 - x) ≥ 0
def C_inequality (x : ℝ) : Prop := (2 * x + 1) / (3 - x) ≥ 0
def D_inequality (x : ℝ) : Prop := (2 * x + 1) / (x - 3) ≤ 0
def solution_set (x : ℝ) : Prop := (-1 / 2 ≤ x ∧ x < 3)

-- proving that option C is equivalent to the solution set
theorem correct_option_C : ∀ x : ℝ, C_inequality x ↔ solution_set x :=
by sorry

-- proving that option D is equivalent to the solution set
theorem correct_option_D : ∀ x : ℝ, D_inequality x ↔ solution_set x :=
by sorry

end correct_option_C_correct_option_D_l272_272706


namespace lcm_15_18_l272_272908

theorem lcm_15_18 : Nat.lcm 15 18 = 90 := by
  sorry

end lcm_15_18_l272_272908


namespace Jason_saturday_hours_l272_272518

theorem Jason_saturday_hours (x y : ℕ) 
  (h1 : 4 * x + 6 * y = 88)
  (h2 : x + y = 18) : 
  y = 8 :=
sorry

end Jason_saturday_hours_l272_272518


namespace difference_between_max_and_min_l272_272601

noncomputable def maxThree (a b c : ℝ) : ℝ :=
  max a (max b c)

noncomputable def minThree (a b c : ℝ) : ℝ :=
  min a (min b c)

theorem difference_between_max_and_min :
  maxThree 0.12 0.23 0.22 - minThree 0.12 0.23 0.22 = 0.11 :=
by
  sorry

end difference_between_max_and_min_l272_272601


namespace cash_sales_amount_l272_272389

-- Definitions for conditions
def total_sales : ℕ := 80
def credit_sales : ℕ := (2 * total_sales) / 5

-- Statement of the proof problem
theorem cash_sales_amount :
  ∃ cash_sales : ℕ, cash_sales = total_sales - credit_sales ∧ cash_sales = 48 :=
by
  sorry

end cash_sales_amount_l272_272389


namespace equation_solution_l272_272443

theorem equation_solution (x : ℝ) :
  (1 / x + 1 / (x + 2) - 1 / (x + 4) - 1 / (x + 6) + 1 / (x + 8) = 0) →
  (x = -4 - 2 * Real.sqrt 3) ∨ (x = 2 - 2 * Real.sqrt 3) := by
  sorry

end equation_solution_l272_272443


namespace tournament_total_players_l272_272514

/--
In a tournament involving n players:
- Each player scored half of all their points in matches against participants who took the last three places.
- Each game results in 1 point.
- Total points from matches among the last three (bad) players = 3.
- The number of games between good and bad players = 3n - 9.
- Total points good players scored from bad players = 3n - 12.
- Games among good players total to (n-3)(n-4)/2 resulting points.
Prove that the total number of participants in the tournament is 9.
-/
theorem tournament_total_players (n : ℕ) :
  3 * (n - 4) = (n - 3) * (n - 4) / 2 → 
  n = 9 :=
by
  intros h
  sorry

end tournament_total_players_l272_272514


namespace smores_cost_calculation_l272_272670

variable (people : ℕ) (s'mores_per_person : ℕ) (s'mores_per_set : ℕ) (cost_per_set : ℕ)

theorem smores_cost_calculation
  (h1 : s'mores_per_person = 3)
  (h2 : people = 8)
  (h3 : s'mores_per_set = 4)
  (h4 : cost_per_set = 3):
  (people * s'mores_per_person / s'mores_per_set) * cost_per_set = 18 := 
by
  sorry

end smores_cost_calculation_l272_272670


namespace tan_theta_cos_sin_id_l272_272159

theorem tan_theta_cos_sin_id (θ : ℝ) (h : Real.tan θ = 3) :
  (1 + Real.cos θ) / Real.sin θ + Real.sin θ / (1 - Real.cos θ) =
  (17 * (Real.sqrt 10 + 1)) / 24 :=
by
  sorry

end tan_theta_cos_sin_id_l272_272159


namespace probability_of_24_is_1_div_1296_l272_272773

def sum_of_dice_is_24 (d1 d2 d3 d4 : ℕ) : Prop :=
  d1 + d2 + d3 + d4 = 24

def probability_of_six (d : ℕ) : Rat :=
  if d = 6 then 1 / 6 else 0

def probability_of_sum_24 (d1 d2 d3 d4 : ℕ) : Rat :=
  (probability_of_six d1) * (probability_of_six d2) * (probability_of_six d3) * (probability_of_six d4)

theorem probability_of_24_is_1_div_1296 :
  (probability_of_sum_24 6 6 6 6) = 1 / 1296 :=
by
  sorry

end probability_of_24_is_1_div_1296_l272_272773


namespace min_area_triangle_AOB_l272_272003

noncomputable def f (x : ℝ) : ℝ := x^2
noncomputable def g (x : ℝ) : ℝ := -1/x

theorem min_area_triangle_AOB :
  (∃ u v : ℝ, u > 0 ∧ v > 0 ∧
               ∠ ((0, 0): ℝ × ℝ) (u, f u) (v, g v) = real.pi / 3 ∧
               S (u, v) = 1.1465) := 
begin
  -- Definitions to compute area
  let A := λ u : ℝ, (u, f u),
  let B := λ v : ℝ, (v, g v),
  let S := λ (u v : ℝ), (1 / 2) * abs (u * (-1 / v) - v * u^2),
  
  -- Real part of the problem statement, including the proof that will derive the answer
  use 0.411797,
  -- Some example value for u and v that you would use to test/debug
  use (u : ℝ),
  use (v : ℝ),
  split,
  { sorry },
  split,
  { sorry },
  split,
  { sorry },
  { 
    have h_area : S 0.411797 (some_v_value) = 1.1465, 
    { sorry },
    rw h_area,
  }
end

end min_area_triangle_AOB_l272_272003


namespace domain_all_real_iff_l272_272752

theorem domain_all_real_iff (k : ℝ) :
  (∀ x : ℝ, -3 * x ^ 2 - x + k ≠ 0 ) ↔ k < -1 / 12 :=
by
  sorry

end domain_all_real_iff_l272_272752


namespace max_candy_leftover_l272_272494

theorem max_candy_leftover (x : ℕ) : (∃ k : ℕ, x = 12 * k + 11) → (x % 12 = 11) :=
by
  sorry

end max_candy_leftover_l272_272494


namespace directrix_of_parabola_l272_272133

-- Define the problem conditions
def parabola (x y : ℝ) : Prop :=
  x = -1/4 * y^2

-- State the theorem to prove that the directrix of the given parabola is x = 1
theorem directrix_of_parabola :
  (∀ (x y : ℝ), parabola x y → true) → 
  let d := 1 in
  ∀ (f : ℝ), (f = -d ∧ f^2 = d^2 ∧ d - f = 2) → true :=
by
  sorry

end directrix_of_parabola_l272_272133


namespace tadpoles_more_than_fish_l272_272243

def fish_initial : ℕ := 100
def tadpoles_initial := 4 * fish_initial
def snails_initial : ℕ := 150
def fish_caught : ℕ := 12
def tadpoles_to_frogs := (2 * tadpoles_initial) / 3
def snails_crawled_away : ℕ := 20

theorem tadpoles_more_than_fish :
  let fish_now : ℕ := fish_initial - fish_caught
  let tadpoles_now : ℕ := tadpoles_initial - tadpoles_to_frogs
  fish_now < tadpoles_now ∧ tadpoles_now - fish_now = 46 :=
by
  sorry

end tadpoles_more_than_fish_l272_272243


namespace triangle_area_correct_l272_272502
noncomputable def area_of_triangle_intercepts : ℝ :=
  let f (x : ℝ) : ℝ := (x - 3) ^ 2 * (x + 2)
  let x1 := 3
  let x2 := -2
  let y_intercept := f 0
  let base := x1 - x2
  let height := y_intercept
  1 / 2 * base * height

theorem triangle_area_correct :
  area_of_triangle_intercepts = 45 :=
by
  sorry

end triangle_area_correct_l272_272502


namespace prob_sum_24_four_dice_l272_272769

section
open ProbabilityTheory

/-- Define the event E24 as the event where the sum of numbers on the top faces of four six-sided dice is 24 -/
def E24 : Event (StdGen) :=
eventOfFun {ω | ∑ i in range 4, (ω.gen_uniform int (6-1)) + 1 = 24}

/-- Probability that the sum of the numbers on top faces of four six-sided dice is 24 is 1/1296. -/
theorem prob_sum_24_four_dice : ⋆{ P(E24) = 1/1296 } := sorry

end

end prob_sum_24_four_dice_l272_272769


namespace time_for_plastic_foam_drift_l272_272580

def boat_speed_in_still_water : ℝ := sorry
def speed_of_water_flow : ℝ := sorry
def distance_between_docks : ℝ := sorry

theorem time_for_plastic_foam_drift (x y s t : ℝ) 
(hx : 6 * (x + y) = s)
(hy : 8 * (x - y) = s)
(t_eq : t = s / y) : 
t = 48 := 
sorry

end time_for_plastic_foam_drift_l272_272580


namespace cash_sales_is_48_l272_272392

variable (total_sales : ℝ) (credit_fraction : ℝ) (cash_sales : ℝ)

-- Conditions: Total sales were $80, 2/5 of the total sales were credit sales
def problem_conditions := total_sales = 80 ∧ credit_fraction = 2/5 ∧ cash_sales = (1 - credit_fraction) * total_sales

-- Question: Prove that the amount of cash sales Mr. Brandon made is $48.
theorem cash_sales_is_48 (h : problem_conditions total_sales credit_fraction cash_sales) : 
  cash_sales = 48 :=
by
  sorry

end cash_sales_is_48_l272_272392


namespace coronavirus_diameter_in_meters_l272_272776

theorem coronavirus_diameter_in_meters (n : ℕ) (h₁ : 1 = (10 : ℤ) ^ 9) (h₂ : n = 125) :
  (n * 10 ^ (-9 : ℤ) : ℝ) = 1.25 * 10 ^ (-7 : ℤ) :=
by
  sorry

end coronavirus_diameter_in_meters_l272_272776


namespace determine_8_genuine_coins_l272_272866

-- Assume there are 11 coins and one may be counterfeit.
variable (coins : Fin 11 → ℝ)
variable (is_counterfeit : Fin 11 → Prop)
variable (genuine_weight : ℝ)
variable (balance : (Fin 11 → ℝ) → (Fin 11 → ℝ) → Prop)

-- The weight of genuine coins.
axiom genuine_coins_weight : ∀ i, ¬ is_counterfeit i → coins i = genuine_weight

-- The statement of the mathematical problem in Lean 4.
theorem determine_8_genuine_coins :
  ∃ (genuine_set : Finset (Fin 11)), genuine_set.card ≥ 8 ∧ ∀ i ∈ genuine_set, ¬ is_counterfeit i :=
sorry

end determine_8_genuine_coins_l272_272866


namespace sum_of_ages_is_correct_l272_272550

-- Define the present ages of A, B, and C
def present_age_A : ℕ := 11

-- Define the ratio conditions from 3 years ago
def three_years_ago_ratio (A B C : ℕ) : Prop :=
  B - 3 = 2 * (A - 3) ∧ C - 3 = 3 * (A - 3)

-- The statement we want to prove
theorem sum_of_ages_is_correct {A B C : ℕ} (hA : A = 11)
  (h_ratio : three_years_ago_ratio A B C) :
  A + B + C = 57 :=
by
  -- The proof part will be handled here
  sorry

end sum_of_ages_is_correct_l272_272550


namespace range_of_m_l272_272628

theorem range_of_m (m : ℝ) (h : ∃ x : ℝ, x < 0 ∧ mx^2 + 2*x + 1 = 0) : m ∈ Set.Iic 1 :=
sorry

end range_of_m_l272_272628


namespace square_of_any_real_number_not_always_greater_than_zero_l272_272113

theorem square_of_any_real_number_not_always_greater_than_zero (a : ℝ) : 
    (∀ x : ℝ, x^2 ≥ 0) ∧ (exists x : ℝ, x = 0 ∧ x^2 = 0) :=
by {
  sorry
}

end square_of_any_real_number_not_always_greater_than_zero_l272_272113


namespace influenza_probability_l272_272739

theorem influenza_probability :
  let flu_rate_A := 0.06
  let flu_rate_B := 0.05
  let flu_rate_C := 0.04
  let population_ratio_A := 6
  let population_ratio_B := 5
  let population_ratio_C := 4
  (population_ratio_A * flu_rate_A + population_ratio_B * flu_rate_B + population_ratio_C * flu_rate_C) / 
  (population_ratio_A + population_ratio_B + population_ratio_C) = 77 / 1500 :=
by
  sorry

end influenza_probability_l272_272739


namespace average_age_union_l272_272714

open Real

variables {a b c d A B C D : ℝ}

theorem average_age_union (h1 : A / a = 40)
                         (h2 : B / b = 30)
                         (h3 : C / c = 45)
                         (h4 : D / d = 35)
                         (h5 : (A + B) / (a + b) = 37)
                         (h6 : (A + C) / (a + c) = 42)
                         (h7 : (A + D) / (a + d) = 39)
                         (h8 : (B + C) / (b + c) = 40)
                         (h9 : (B + D) / (b + d) = 37)
                         (h10 : (C + D) / (c + d) = 43) : 
  (A + B + C + D) / (a + b + c + d) = 44.5 := 
sorry

end average_age_union_l272_272714


namespace base4_to_base10_conversion_l272_272899

theorem base4_to_base10_conversion :
  2 * 4^4 + 0 * 4^3 + 3 * 4^2 + 1 * 4^1 + 2 * 4^0 = 566 :=
by
  sorry

end base4_to_base10_conversion_l272_272899


namespace min_val_expression_l272_272840

theorem min_val_expression (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a^2 * b + b^2 * c + c^2 * a = 3) : 
  a^7 * b + b^7 * c + c^7 * a + a * b^3 + b * c^3 + c * a^3 ≥ 6 :=
sorry

end min_val_expression_l272_272840


namespace point_symmetric_to_line_l272_272067

-- Define the problem statement
theorem point_symmetric_to_line (M : ℝ × ℝ) (l : ℝ × ℝ) (N : ℝ × ℝ) :
  M = (1, 4) →
  l = (1, -1) →
  (∃ a b, N = (a, b) ∧ a + b = 5 ∧ a - b = 1) →
  N = (3, 2) :=
by
  sorry

end point_symmetric_to_line_l272_272067


namespace t_shirts_per_package_l272_272561

theorem t_shirts_per_package (total_t_shirts : ℕ) (total_packages : ℕ) (h1 : total_t_shirts = 39) (h2 : total_packages = 3) : total_t_shirts / total_packages = 13 :=
by {
  sorry
}

end t_shirts_per_package_l272_272561


namespace baskets_and_remainder_l272_272686

-- Define the initial conditions
def cucumbers : ℕ := 216
def basket_capacity : ℕ := 23

-- Define the expected calculations
def expected_baskets : ℕ := cucumbers / basket_capacity
def expected_remainder : ℕ := cucumbers % basket_capacity

-- Theorem to prove the output values
theorem baskets_and_remainder :
  expected_baskets = 9 ∧ expected_remainder = 9 := by
  sorry

end baskets_and_remainder_l272_272686


namespace real_solution_exists_l272_272903

theorem real_solution_exists (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ 3) :
  (x^3 - 4*x^2) / (x^2 - 5*x + 6) - x = 9 → x = 9/2 :=
by sorry

end real_solution_exists_l272_272903


namespace minimum_races_to_determine_top_five_fastest_horses_l272_272259

-- Defining the conditions
def max_horses_per_race : ℕ := 3
def total_horses : ℕ := 50

-- The main statement to prove the minimum number of races y
theorem minimum_races_to_determine_top_five_fastest_horses (y : ℕ) :
  y = 19 :=
sorry

end minimum_races_to_determine_top_five_fastest_horses_l272_272259


namespace mayo_bottles_count_l272_272365

theorem mayo_bottles_count
  (ketchup_ratio mayo_ratio : ℕ) 
  (ratio_multiplier ketchup_bottles : ℕ)
  (h_ratio_eq : 3 = ketchup_ratio)
  (h_mayo_ratio_eq : 2 = mayo_ratio)
  (h_ketchup_bottles_eq : 6 = ketchup_bottles)
  (h_ratio_condition : ketchup_bottles * mayo_ratio = ketchup_ratio * ratio_multiplier) :
  ratio_multiplier = 4 := 
by 
  sorry

end mayo_bottles_count_l272_272365


namespace focus_parabola_l272_272325

theorem focus_parabola (f : ℝ) (d : ℝ) (y : ℝ) :
  (∀ y, ((- (1 / 8) * y^2 - f) ^ 2 + y^2 = (- (1 / 8) * y^2 - d) ^ 2)) → 
  (d - f = 4) → 
  (f^2 = d^2) → 
  f = -2 :=
by
  sorry

end focus_parabola_l272_272325


namespace Jason_earned_60_dollars_l272_272197

-- Define initial and final amounts of money
variable (Jason_initial Jason_final : ℕ)

-- State the assumption about Jason's initial and final amounts of money
variable (h_initial : Jason_initial = 3) (h_final : Jason_final = 63)

-- Define the amount of money Jason earned
def Jason_earn := Jason_final - Jason_initial

-- Prove that Jason earned 60 dollars by delivering newspapers
theorem Jason_earned_60_dollars : Jason_earn Jason_initial Jason_final = 60 := by
  sorry

end Jason_earned_60_dollars_l272_272197


namespace p_implies_q_l272_272781

theorem p_implies_q (x : ℝ) (h : |5 * x - 1| > 4) : x^2 - (3/2) * x + (1/2) > 0 := sorry

end p_implies_q_l272_272781


namespace cell_division_50_closest_to_10_15_l272_272717

theorem cell_division_50_closest_to_10_15 :
  10^14 < 2^50 ∧ 2^50 < 10^16 :=
sorry

end cell_division_50_closest_to_10_15_l272_272717


namespace m_value_for_arithmetic_seq_min_value_t_smallest_T_periodic_seq_l272_272920

-- Defining the sequence condition
def seq_condition (a : ℕ → ℝ) (m : ℝ) : Prop :=
  ∀ n ≥ 2, a n ^ 2 - a (n + 1) * a (n - 1) = m * (a 2 - a 1) ^ 2

-- (1) Value of m for an arithmetic sequence with a non-zero common difference
theorem m_value_for_arithmetic_seq {a : ℕ → ℝ} (d : ℝ) (h_nonzero : d ≠ 0) :
  (∀ n, a (n + 1) = a n + d) → seq_condition a 1 :=
by
  sorry

-- (2) Minimum value of t given specific conditions
theorem min_value_t {t p : ℝ} (a : ℕ → ℝ) (h_p : 3 ≤ p ∧ p ≤ 5) :
  a 1 = 1 ∧ a 2 = 2 ∧ a 3 = 4 ∧ (∀ n, t * a n + p ≥ n) → t = 1 / 32 :=
by
  sorry

-- (3) Smallest value of T for non-constant periodic sequence
theorem smallest_T_periodic_seq {a : ℕ → ℝ} {m : ℝ} (h_m_nonzero : m ≠ 0) :
  seq_condition a m → (∀ n, a (n + T) = a n) → (∃ T' > 0, ∀ T'', T'' > 0 → T'' = 3) :=
by
  sorry

end m_value_for_arithmetic_seq_min_value_t_smallest_T_periodic_seq_l272_272920


namespace abby_damon_weight_l272_272891

theorem abby_damon_weight (a' b' c' d' : ℕ) (h1 : a' + b' = 265) (h2 : b' + c' = 250) (h3 : c' + d' = 280) :
  a' + d' = 295 :=
  sorry -- Proof goes here

end abby_damon_weight_l272_272891


namespace functional_eq_solve_l272_272312

theorem functional_eq_solve (f : ℝ → ℝ) (h : ∀ x y : ℝ, f (2*x + f y) = x + y + f x) : 
  ∀ x : ℝ, f x = x := 
sorry

end functional_eq_solve_l272_272312


namespace trigonometric_identity_l272_272939

theorem trigonometric_identity (α : ℝ) (h : Real.sin (α + (Real.pi / 3)) = 3 / 5) :
  Real.cos ((Real.pi / 6) - α) = 3 / 5 :=
by
  sorry

end trigonometric_identity_l272_272939


namespace sum_smallest_numbers_eq_six_l272_272854

theorem sum_smallest_numbers_eq_six :
  let smallest_natural := 0
  let smallest_prime := 2
  let smallest_composite := 4
  smallest_natural + smallest_prime + smallest_composite = 6 := by
  sorry

end sum_smallest_numbers_eq_six_l272_272854


namespace functional_equation_solution_l272_272313

theorem functional_equation_solution :
  ∀ f : ℝ → ℝ, (∀ x y : ℝ, f(2 * x + f(y)) = x + y + f(x)) → (∀ x : ℝ, f(x) = x) :=
by
  intros f H
  sorry

end functional_equation_solution_l272_272313


namespace cards_arrangement_count_is_10_l272_272998

-- Define the problem in Lean statement terms
def valid_arrangements_count : ℕ :=
  -- number of arrangements of seven cards where one card can be removed 
  -- leaving the remaining six cards in either ascending or descending order
  10

-- Theorem stating that the number of valid arrangements is 10
theorem cards_arrangement_count_is_10 : valid_arrangements_count = 10 :=
by
  -- Proof is omitted (the explanation above corresponds to the omitted proof details)
  sorry

end cards_arrangement_count_is_10_l272_272998


namespace smallest_percentage_all_correct_l272_272582

theorem smallest_percentage_all_correct (p1 p2 p3 : ℝ) 
  (h1 : p1 = 0.9) 
  (h2 : p2 = 0.8)
  (h3 : p3 = 0.7) :
  ∃ x, x = 0.4 ∧ (x ≤ 1 - ((1 - p1) + (1 - p2) + (1 - p3))) :=
by 
  sorry

end smallest_percentage_all_correct_l272_272582


namespace frac_eq_three_l272_272011

theorem frac_eq_three (a b c : ℝ) 
  (h₁ : a / b = 4 / 3) (h₂ : (a + c) / (b - c) = 5 / 2) : 
  (3 * a + 2 * b) / (3 * a - 2 * b) = 3 :=
by
  sorry

end frac_eq_three_l272_272011


namespace sin_cos_15_eq_1_over_4_l272_272440

theorem sin_cos_15_eq_1_over_4 : (Real.sin (15 * Real.pi / 180)) * (Real.cos (15 * Real.pi / 180)) = 1 / 4 := 
by
  sorry

end sin_cos_15_eq_1_over_4_l272_272440


namespace angle_B_in_triangle_ABC_l272_272962

theorem angle_B_in_triangle_ABC (A B C : Prop) (angle_A : ℕ) (angle_C : ℕ) 
  (hA : angle_A = 20) (hC : angle_C = 90) : angle B = 70 :=
by
  sorry

end angle_B_in_triangle_ABC_l272_272962


namespace nick_coin_collection_l272_272836

theorem nick_coin_collection
  (total_coins : ℕ)
  (quarters_coins : ℕ)
  (dimes_coins : ℕ)
  (nickels_coins : ℕ)
  (state_quarters : ℕ)
  (pa_state_quarters : ℕ)
  (roosevelt_dimes : ℕ)
  (h_total : total_coins = 50)
  (h_quarters : quarters_coins = total_coins * 3 / 10)
  (h_dimes : dimes_coins = total_coins * 40 / 100)
  (h_nickels : nickels_coins = total_coins - (quarters_coins + dimes_coins))
  (h_state_quarters : state_quarters = quarters_coins * 2 / 5)
  (h_pa_state_quarters : pa_state_quarters = state_quarters * 3 / 8)
  (h_roosevelt_dimes : roosevelt_dimes = dimes_coins * 75 / 100) :
  pa_state_quarters = 2 ∧ roosevelt_dimes = 15 ∧ nickels_coins = 15 :=
by
  sorry

end nick_coin_collection_l272_272836


namespace sequence_increasing_l272_272539

noncomputable def a (n : ℕ) : ℚ := (2 * n) / (2 * n + 1)

theorem sequence_increasing (n : ℕ) (hn : 0 < n) : a n < a (n + 1) :=
by
  -- Proof to be provided
  sorry

end sequence_increasing_l272_272539


namespace quadratic_equation_with_given_root_l272_272121

theorem quadratic_equation_with_given_root : 
  ∃ p q : ℤ, (∀ x : ℝ, x^2 + (p : ℝ) * x + (q : ℝ) = 0 ↔ x = 2 - Real.sqrt 7 ∨ x = 2 + Real.sqrt 7) 
  ∧ (p = -4) ∧ (q = -3) :=
by
  sorry

end quadratic_equation_with_given_root_l272_272121


namespace initial_marbles_l272_272563

theorem initial_marbles (M : ℕ) :
  (M - 5) % 3 = 0 ∧ M = 65 :=
by
  sorry

end initial_marbles_l272_272563


namespace find_B_plus_C_l272_272504

-- Define the arithmetic translations for base 8 numbers
def base8_to_dec (a b c : ℕ) : ℕ := 8^2 * a + 8 * b + c

def condition1 (A B C : ℕ) : Prop :=
  A ≠ B ∧ B ≠ C ∧ C ≠ A ∧ 1 ≤ A ∧ A ≤ 7 ∧ 1 ≤ B ∧ B ≤ 7 ∧ 1 ≤ C ∧ C ≤ 7

-- Define the main condition in the problem
def condition2 (A B C : ℕ) : Prop :=
  base8_to_dec A B C + base8_to_dec B C A + base8_to_dec C A B = 8^3 * A + 8^2 * A + 8 * A

-- The main statement to be proven
theorem find_B_plus_C (A B C : ℕ) (h1 : condition1 A B C) (h2 : condition2 A B C) : B + C = 7 :=
sorry

end find_B_plus_C_l272_272504


namespace original_five_digit_number_l272_272276

theorem original_five_digit_number :
  ∃ N y x : ℕ, (N = 10 * y + x) ∧ (N + y = 54321) ∧ (N = 49383) :=
by
  -- The proof script goes here
  sorry

end original_five_digit_number_l272_272276


namespace major_premise_wrong_l272_272112

-- Definitions of the given conditions in Lean
def is_parallel_to_plane (line : Type) (plane : Type) : Prop := sorry -- Provide an appropriate definition
def contains_line (plane : Type) (line : Type) : Prop := sorry -- Provide an appropriate definition
def is_parallel_to_line (line1 : Type) (line2 : Type) : Prop := sorry -- Provide an appropriate definition

-- Given conditions
variables (b α a : Type)
variable (H1 : ¬ contains_line α b)  -- Line b is not contained in plane α
variable (H2 : contains_line α a)    -- Line a is contained in plane α
variable (H3 : is_parallel_to_plane b α) -- Line b is parallel to plane α

-- Proposition to prove: The major premise is wrong
theorem major_premise_wrong : ¬(∀ (a b : Type), is_parallel_to_plane b α → contains_line α a → is_parallel_to_line b a) :=
by
  sorry

end major_premise_wrong_l272_272112


namespace batsman_average_after_17th_inning_l272_272569

theorem batsman_average_after_17th_inning (A : ℝ) :
  (16 * A + 87) / 17 = A + 3 → A + 3 = 39 :=
by
  intro h
  sorry

end batsman_average_after_17th_inning_l272_272569


namespace remainder_div_polynomial_l272_272546

theorem remainder_div_polynomial :
  ∀ (x : ℝ), 
  ∃ (Q : ℝ → ℝ) (R : ℝ → ℝ), 
    R x = (3^101 - 2^101) * x + (2^101 - 2 * 3^101) ∧
    x^101 = (x^2 - 5 * x + 6) * Q x + R x :=
by
  sorry

end remainder_div_polynomial_l272_272546


namespace value_of_squares_l272_272976

-- Define the conditions
variables (p q : ℝ)

-- State the theorem with the given conditions and the proof goal
theorem value_of_squares (h1 : p * q = 12) (h2 : p + q = 8) : p ^ 2 + q ^ 2 = 40 :=
sorry

end value_of_squares_l272_272976


namespace average_snowfall_per_minute_l272_272026

def total_snowfall := 550
def days_in_december := 31
def hours_per_day := 24
def minutes_per_hour := 60

theorem average_snowfall_per_minute :
  (total_snowfall : ℝ) / (days_in_december * hours_per_day * minutes_per_hour) = 550 / (31 * 24 * 60) :=
by
  sorry

end average_snowfall_per_minute_l272_272026


namespace pie_shop_earnings_l272_272729

-- Define the conditions
def price_per_slice : ℕ := 3
def slices_per_pie : ℕ := 10
def number_of_pies : ℕ := 6

-- Calculate the total slices
def total_slices : ℕ := number_of_pies * slices_per_pie

-- Calculate the total earnings
def total_earnings : ℕ := total_slices * price_per_slice

-- State the theorem
theorem pie_shop_earnings : total_earnings = 180 :=
by
  -- Proof can be skipped with a sorry
  sorry

end pie_shop_earnings_l272_272729


namespace john_safety_percentage_l272_272653

def bench_max_weight : ℕ := 1000
def john_weight : ℕ := 250
def weight_on_bar : ℕ := 550
def total_weight := john_weight + weight_on_bar
def percentage_of_max_weight := (total_weight * 100) / bench_max_weight
def percentage_under_max_weight := 100 - percentage_of_max_weight

theorem john_safety_percentage : percentage_under_max_weight = 20 := by
  sorry

end john_safety_percentage_l272_272653


namespace percentage_of_600_equals_150_is_25_l272_272909

theorem percentage_of_600_equals_150_is_25 : (150 / 600 * 100) = 25 := by
  sorry

end percentage_of_600_equals_150_is_25_l272_272909


namespace find_original_number_l272_272287

-- Variables and assumptions
variables (N y x : ℕ)

-- Conditions
def crossing_out_condition : Prop := N = 10 * y + x
def sum_condition : Prop := N + y = 54321
def remainder_condition : Prop := x = 54321 % 11
def y_value : Prop := y = 4938

-- The final proof problem statement
theorem find_original_number (h1 : crossing_out_condition N y x) (h2 : sum_condition N y) 
  (h3 : remainder_condition x) (h4 : y_value y) : N = 49383 :=
sorry

end find_original_number_l272_272287


namespace total_cookies_l272_272830

def MonaCookies : ℕ := 20
def JasmineCookies : ℕ := MonaCookies - 5
def RachelCookies : ℕ := JasmineCookies + 10

theorem total_cookies : MonaCookies + JasmineCookies + RachelCookies = 60 := by
  -- Since we don't need to provide the solution steps, we simply use sorry.
  sorry

end total_cookies_l272_272830


namespace not_perfect_square_9n_squared_minus_9n_plus_9_l272_272201

theorem not_perfect_square_9n_squared_minus_9n_plus_9
  (n : ℕ) (h : n > 1) : ¬ (∃ k : ℕ, 9 * n^2 - 9 * n + 9 = k * k) := sorry

end not_perfect_square_9n_squared_minus_9n_plus_9_l272_272201


namespace cap_to_sunglasses_prob_l272_272668

-- Define the conditions
def num_people_wearing_sunglasses : ℕ := 60
def num_people_wearing_caps : ℕ := 40
def prob_sunglasses_and_caps : ℚ := 1 / 3

-- Define the statement to prove
theorem cap_to_sunglasses_prob : 
  (num_people_wearing_sunglasses * prob_sunglasses_and_caps) / num_people_wearing_caps = 1 / 2 :=
by
  sorry

end cap_to_sunglasses_prob_l272_272668


namespace count_ordered_pairs_l272_272152

theorem count_ordered_pairs : 
  ∃ n : ℕ, n = 136 ∧ 
  ∀ a b : ℝ, 
    (∃ x y : ℤ, a * x + b * y = 2 ∧ (x : ℝ)^2 + (y : ℝ)^2 = 65) → n = 136 := 
sorry

end count_ordered_pairs_l272_272152


namespace max_similar_triangles_five_points_l272_272380

-- Let P be a finite set of points on a plane with exactly 5 elements.
def max_similar_triangles(P : Finset (ℝ × ℝ)) : ℕ :=
  if h : P.card = 5 then
    8
  else
    0 -- This is irrelevant for the problem statement, but we need to define it.

-- The main theorem statement
theorem max_similar_triangles_five_points {P : Finset (ℝ × ℝ)} (h : P.card = 5) :
  max_similar_triangles P = 8 :=
sorry

end max_similar_triangles_five_points_l272_272380


namespace cone_height_l272_272479

-- Definitions given in the problem
def slant_height : ℝ := 13
def lateral_area : ℝ := 65 * Real.pi

-- Definition of the radius as derived from the given conditions
def radius : ℝ := lateral_area / (Real.pi * slant_height) -- This simplifies to 5

-- Using the Pythagorean theorem to express the height
def height : ℝ := Real.sqrt (slant_height^2 - radius^2)

-- The statement to prove
theorem cone_height : height = 12 := by
  sorry

end cone_height_l272_272479


namespace find_quaterns_l272_272600

theorem find_quaterns {
  x y z w : ℝ
} : 
  (x + y = z^2 + w^2 + 6 * z * w) → 
  (x + z = y^2 + w^2 + 6 * y * w) → 
  (x + w = y^2 + z^2 + 6 * y * z) → 
  (y + z = x^2 + w^2 + 6 * x * w) → 
  (y + w = x^2 + z^2 + 6 * x * z) → 
  (z + w = x^2 + y^2 + 6 * x * y) → 
  ( (x, y, z, w) = (0, 0, 0, 0) 
    ∨ (x, y, z, w) = (1/4, 1/4, 1/4, 1/4) 
    ∨ (x, y, z, w) = (-1/4, -1/4, 3/4, -1/4) 
    ∨ (x, y, z, w) = (-1/2, -1/2, 5/2, -1/2)
  ) :=
  sorry

end find_quaterns_l272_272600


namespace Lizzie_has_27_crayons_l272_272207

variable (Lizzie Bobbie Billie : ℕ)

axiom Billie_crayons : Billie = 18
axiom Bobbie_crayons : Bobbie = 3 * Billie
axiom Lizzie_crayons : Lizzie = Bobbie / 2

theorem Lizzie_has_27_crayons : Lizzie = 27 :=
by
  sorry

end Lizzie_has_27_crayons_l272_272207


namespace find_y_l272_272865

theorem find_y (x y : ℝ) (h1 : x + 2 * y = 10) (h2 : x = 4) : y = 3 := 
by sorry

end find_y_l272_272865


namespace tammy_haircuts_l272_272223

theorem tammy_haircuts (total_haircuts free_haircuts haircuts_to_next_free : ℕ) 
(h1 : free_haircuts = 5) 
(h2 : haircuts_to_next_free = 5) 
(h3 : total_haircuts = 79) : 
(haircuts_to_next_free = 5) :=
by {
  sorry
}

end tammy_haircuts_l272_272223


namespace complex_magnitude_problem_l272_272819

open Complex

theorem complex_magnitude_problem
  (z w : ℂ)
  (hz : abs z = 1)
  (hw : abs w = 2)
  (hzw : abs (z + w) = 3) :
  abs ((1 / z) + (1 / w)) = 3 / 2 :=
by {
  sorry
}

end complex_magnitude_problem_l272_272819


namespace vector_operation_l272_272492

variables {α : Type*} [AddCommGroup α] [Module ℝ α]
variables (a b : α)

theorem vector_operation :
  (1 / 2 : ℝ) • (2 • a - 4 • b) + 2 • b = a :=
by sorry

end vector_operation_l272_272492


namespace FO_gt_DI_l272_272644

-- Definitions and conditions
variables (F I D O : Type) [MetricSpace F] [MetricSpace I] [MetricSpace D] [MetricSpace O]
variables (FI DO DI FO : ℝ) (angle_FIO angle_DIO : ℝ)
variable (convex_FIDO : ConvexQuadrilateral F I D O)

-- Conditions
axiom FI_DO_equal : FI = DO
axiom FI_DO_gt_DI : FI > DI
axiom angles_equal : angle_FIO = angle_DIO

-- Goal
theorem FO_gt_DI : FO > DI :=
sorry

end FO_gt_DI_l272_272644


namespace total_students_l272_272870

theorem total_students (boys girls : ℕ) (h_ratio : boys / girls = 8 / 5) (h_girls : girls = 120) : boys + girls = 312 :=
by
  sorry

end total_students_l272_272870


namespace original_number_l272_272273

theorem original_number (N y x : ℕ) 
  (h1: N + y = 54321)
  (h2: N = 10 * y + x)
  (h3: 11 * y + x = 54321)
  (h4: x = 54321 % 11)
  (hy: y = 4938) : 
  N = 49383 := 
  by 
  sorry

end original_number_l272_272273


namespace inscribed_square_area_l272_272898

def isosceles_right_triangle (a b c : ℝ) : Prop :=
  a = b ∧ c = (a ^ 2 + b ^ 2) ^ (1 / 2)

def square_area (s : ℝ) : ℝ := s * s

theorem inscribed_square_area
  (a b c : ℝ) (s₁ s₂ : ℝ)
  (ha : a = 16 * 2) -- Leg lengths equal to 2 * 16 cm
  (hb : b = 16 * 2)
  (hc : c = 32 * Real.sqrt 2) -- Hypotenuse of the triangle
  (hiso : isosceles_right_triangle a b c)
  (harea₁ : square_area 16 = 256) -- Given square area
  (hS : s₂ = 16 * Real.sqrt 2 - 8) -- Side length of the new square
  : square_area s₂ = 576 - 256 * Real.sqrt 2 := sorry

end inscribed_square_area_l272_272898


namespace dice_sum_24_probability_l272_272764

noncomputable def probability_sum_24 : ℚ :=
  let prob_single_six := (1 : ℚ) / 6 in
  prob_single_six ^ 4

theorem dice_sum_24_probability :
  probability_sum_24 = 1 / 1296 :=
by
  sorry

end dice_sum_24_probability_l272_272764


namespace students_preferring_windows_is_correct_l272_272579

-- Define the total number of students surveyed
def total_students : ℕ := 210

-- Define the number of students preferring Mac
def students_preferring_mac : ℕ := 60

-- Define the number of students preferring both Mac and Windows equally
def students_preferring_both : ℕ := students_preferring_mac / 3

-- Define the number of students with no preference
def students_no_preference : ℕ := 90

-- Calculate the total number of students with a preference
def students_with_preference : ℕ := total_students - students_no_preference

-- Calculate the number of students preferring Windows
def students_preferring_windows : ℕ := students_with_preference - (students_preferring_mac + students_preferring_both)

-- State the theorem to prove that the number of students preferring Windows is 40
theorem students_preferring_windows_is_correct : students_preferring_windows = 40 :=
by
  -- calculations based on definitions
  unfold students_preferring_windows students_with_preference students_preferring_mac students_preferring_both students_no_preference total_students
  sorry

end students_preferring_windows_is_correct_l272_272579


namespace find_a_l272_272926

noncomputable def f (x a : ℝ) := x^2 + (a + 1) * x + (a + 2)

def g (x a : ℝ) := (a + 1) * x
def h (x a : ℝ) := x^2 + a + 2

def p (a : ℝ) := ∀ x ≥ (a + 1)^2, f x a ≤ x
def q (a : ℝ) := ∀ x, g x a < 0

theorem find_a : 
  (¬p a) → (p a ∨ q a) → a ≥ -1 := sorry

end find_a_l272_272926


namespace remainder_is_zero_l272_272863

def remainder_when_multiplied_then_subtracted (a b : ℕ) : ℕ :=
  (a * b - 8) % 8

theorem remainder_is_zero : remainder_when_multiplied_then_subtracted 104 106 = 0 := by
  sorry

end remainder_is_zero_l272_272863


namespace max_candy_leftover_l272_272495

theorem max_candy_leftover (x : ℕ) : (∃ k : ℕ, x = 12 * k + 11) → (x % 12 = 11) :=
by
  sorry

end max_candy_leftover_l272_272495


namespace triangle_inequality_proof_l272_272329

theorem triangle_inequality_proof (a b c : ℝ) (h : a + b > c) : a^3 + b^3 + 3 * a * b * c > c^3 :=
by sorry

end triangle_inequality_proof_l272_272329


namespace intersection_complement_l272_272054

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def P : Set ℕ := {1, 2, 3, 4}
def Q : Set ℕ := {3, 4, 5}

theorem intersection_complement : P ∩ (U \ Q) = {1, 2} :=
by
  sorry

end intersection_complement_l272_272054


namespace probability_of_same_color_is_correct_l272_272508

-- Define the parameters for balls in the bag
def green_balls : ℕ := 8
def red_balls : ℕ := 6
def blue_balls : ℕ := 1
def total_balls : ℕ := green_balls + red_balls + blue_balls

-- Define the probabilities of drawing each color
def prob_green : ℚ := green_balls / total_balls
def prob_red : ℚ := red_balls / total_balls
def prob_blue : ℚ := blue_balls / total_balls

-- Define the probability of drawing two balls of the same color
def prob_same_color : ℚ :=
  prob_green^2 + prob_red^2 + prob_blue^2

theorem probability_of_same_color_is_correct :
  prob_same_color = 101 / 225 :=
by
  sorry

end probability_of_same_color_is_correct_l272_272508


namespace find_min_n_l272_272297

theorem find_min_n (n k : ℕ) (h : 14 * n = k^2) : n = 14 := sorry

end find_min_n_l272_272297


namespace zack_marbles_number_l272_272565

-- Define the conditions as Lean definitions
def zack_initial_marbles (x : ℕ) :=
  (∃ k : ℕ, x = 3 * k + 5) ∧ (3 * 20 + 5 = 65)

-- State the theorem using the conditions
theorem zack_marbles_number : ∃ x : ℕ, zack_initial_marbles x ∧ x = 65 :=
by
  sorry

end zack_marbles_number_l272_272565


namespace probability_of_24_is_1_div_1296_l272_272772

def sum_of_dice_is_24 (d1 d2 d3 d4 : ℕ) : Prop :=
  d1 + d2 + d3 + d4 = 24

def probability_of_six (d : ℕ) : Rat :=
  if d = 6 then 1 / 6 else 0

def probability_of_sum_24 (d1 d2 d3 d4 : ℕ) : Rat :=
  (probability_of_six d1) * (probability_of_six d2) * (probability_of_six d3) * (probability_of_six d4)

theorem probability_of_24_is_1_div_1296 :
  (probability_of_sum_24 6 6 6 6) = 1 / 1296 :=
by
  sorry

end probability_of_24_is_1_div_1296_l272_272772


namespace dog_nails_per_foot_l272_272897

-- Definitions from conditions
def number_of_dogs := 4
def number_of_parrots := 8
def total_nails_to_cut := 113
def parrots_claws := 8

-- Derived calculations from the solution but only involving given conditions
def dogs_claws (nails_per_foot : ℕ) := 16 * nails_per_foot
def parrots_total_claws := number_of_parrots * parrots_claws

-- The main theorem to prove the number of nails per dog foot
theorem dog_nails_per_foot :
  ∃ x : ℚ, 16 * x + parrots_total_claws = total_nails_to_cut :=
by {
  -- Directly state the expected answer
  use 3.0625,
  -- Placeholder for proof
  sorry
}

end dog_nails_per_foot_l272_272897


namespace license_plate_count_correct_l272_272066

def rotokas_letters : Finset Char := {'A', 'E', 'G', 'I', 'K', 'O', 'P', 'R', 'S', 'T', 'U'}

def valid_license_plate_count : ℕ :=
  let first_letter_choices := 2 -- Letters A or E
  let last_letter_fixed := 1 -- Fixed as P
  let remaining_letters := rotokas_letters.erase 'V' -- Exclude V
  let second_letter_choices := (remaining_letters.erase 'P').card - 1 -- Exclude P and first letter
  let third_letter_choices := second_letter_choices - 1
  let fourth_letter_choices := third_letter_choices - 1
  2 * 9 * 8 * 7

theorem license_plate_count_correct :
  valid_license_plate_count = 1008 := by
  sorry

end license_plate_count_correct_l272_272066


namespace find_second_sum_l272_272257

theorem find_second_sum (x : ℝ) (h : 24 * x / 100 = (2730 - x) * 15 / 100) : 2730 - x = 1680 := by
  sorry

end find_second_sum_l272_272257


namespace initial_marbles_l272_272562

theorem initial_marbles (M : ℕ) :
  (M - 5) % 3 = 0 ∧ M = 65 :=
by
  sorry

end initial_marbles_l272_272562


namespace range_of_t_l272_272922

theorem range_of_t (a b : ℝ) 
  (h1 : a^2 + a * b + b^2 = 1) 
  (h2 : ∃ t : ℝ, t = a * b - a^2 - b^2) : 
  ∀ t, t = a * b - a^2 - b^2 → -3 ≤ t ∧ t ≤ -1/3 :=
by sorry

end range_of_t_l272_272922


namespace count_valid_numbers_l272_272937

theorem count_valid_numbers :
  {n : ℕ | n >= 100 ∧ n < 1000 ∧ (∀ d ∈ [n / 100, (n % 100) / 10, n % 10], d > 6) ∧ (n % 12 = 0)}.card = 1 :=
by
  sorry

end count_valid_numbers_l272_272937


namespace jame_initial_gold_bars_l272_272811

theorem jame_initial_gold_bars (X : ℝ) (h1 : X * 0.1 + 0.5 * (X * 0.9) = 0.5 * (X * 0.9) - 27) :
  X = 60 :=
by
-- Placeholder for proof
sorry

end jame_initial_gold_bars_l272_272811


namespace incorrect_option_C_l272_272457

theorem incorrect_option_C (a b : ℝ) (h1 : a > b) (h2 : b > a + b) : ¬ (ab > (a + b)^2) :=
by {
  sorry
}

end incorrect_option_C_l272_272457


namespace inequality_solution_l272_272324

theorem inequality_solution (x : ℝ) : 
  (3 / 20 + abs (2 * x - 5 / 40) < 9 / 40) → (1 / 40 < x ∧ x < 1 / 10) :=
by
  sorry

end inequality_solution_l272_272324


namespace negative_to_zero_power_l272_272797

theorem negative_to_zero_power (a : ℝ) (h : a ≠ 0) : (-a) ^ 0 = 1 :=
by
  sorry

end negative_to_zero_power_l272_272797


namespace school_raised_amount_correct_l272_272087

def school_fundraising : Prop :=
  let mrsJohnson := 2300
  let mrsSutton := mrsJohnson / 2
  let missRollin := mrsSutton * 8
  let topThreeTotal := missRollin * 3
  let mrEdward := missRollin * 0.75
  let msAndrea := mrEdward * 1.5
  let totalRaised := mrsJohnson + mrsSutton + missRollin + mrEdward + msAndrea
  let adminFee := totalRaised * 0.02
  let maintenanceExpense := totalRaised * 0.05
  let totalDeductions := adminFee + maintenanceExpense
  let finalAmount := totalRaised - totalDeductions
  finalAmount = 28737

theorem school_raised_amount_correct : school_fundraising := 
by 
  sorry

end school_raised_amount_correct_l272_272087


namespace ferris_wheel_seat_capacity_l272_272577

-- Define the given conditions
def people := 16
def seats := 4

-- Define the problem and the proof goal
theorem ferris_wheel_seat_capacity : people / seats = 4 := by
  sorry

end ferris_wheel_seat_capacity_l272_272577


namespace minimum_value_of_expression_l272_272525

theorem minimum_value_of_expression (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a + b + c = 6) :
  (9 / a + 16 / b + 25 / c) = 24 :=
sorry

end minimum_value_of_expression_l272_272525


namespace find_original_number_l272_272286

-- Variables and assumptions
variables (N y x : ℕ)

-- Conditions
def crossing_out_condition : Prop := N = 10 * y + x
def sum_condition : Prop := N + y = 54321
def remainder_condition : Prop := x = 54321 % 11
def y_value : Prop := y = 4938

-- The final proof problem statement
theorem find_original_number (h1 : crossing_out_condition N y x) (h2 : sum_condition N y) 
  (h3 : remainder_condition x) (h4 : y_value y) : N = 49383 :=
sorry

end find_original_number_l272_272286


namespace parabola_directrix_l272_272137

theorem parabola_directrix : 
  ∃ d : ℝ, (∀ (y : ℝ), x = - (1 / 4) * y^2 -> 
  ( - (1 / 4) * y^2 = d)) ∧ d = 1 :=
by
  sorry

end parabola_directrix_l272_272137


namespace find_k_l272_272484

-- Definitions of conditions
def equation1 (x k : ℝ) : Prop := x^2 + k*x + 10 = 0
def equation2 (x k : ℝ) : Prop := x^2 - k*x + 10 = 0
def roots_relation (a b k : ℝ) : Prop :=
  equation1 a k ∧ 
  equation1 b k ∧ 
  equation2 (a + 3) k ∧
  equation2 (b + 3) k

-- Statement to be proven
theorem find_k (a b k : ℝ) (h : roots_relation a b k) : k = 3 :=
sorry

end find_k_l272_272484
