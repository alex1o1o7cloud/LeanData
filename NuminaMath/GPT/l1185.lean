import Mathlib

namespace prime_factorization_of_expression_l1185_118583

theorem prime_factorization_of_expression :
  2 * 3 * 5 * 7 - 1 = 11 * 19 :=
sorry

end prime_factorization_of_expression_l1185_118583


namespace tiffany_total_score_l1185_118538

-- Definitions based on conditions
def points_per_treasure : ℕ := 6
def treasures_first_level : ℕ := 3
def treasures_second_level : ℕ := 5

-- The statement we want to prove
theorem tiffany_total_score : (points_per_treasure * treasures_first_level) + (points_per_treasure * treasures_second_level) = 48 := by
  sorry

end tiffany_total_score_l1185_118538


namespace center_of_circle_l1185_118505

theorem center_of_circle :
  ∀ (x y : ℝ), (x^2 - 8 * x + y^2 - 4 * y = 16) → (x, y) = (4, 2) :=
by
  sorry

end center_of_circle_l1185_118505


namespace characters_per_day_l1185_118548

-- Definitions based on conditions
def chars_total_older : ℕ := 8000
def chars_total_younger : ℕ := 6000
def chars_per_day_diff : ℕ := 100

-- Define the main theorem
theorem characters_per_day (x : ℕ) :
  chars_total_older / x = chars_total_younger / (x - chars_per_day_diff) := 
sorry

end characters_per_day_l1185_118548


namespace people_got_on_at_third_stop_l1185_118524

theorem people_got_on_at_third_stop
  (initial : ℕ)
  (got_off_first : ℕ)
  (got_off_second : ℕ)
  (got_on_second : ℕ)
  (got_off_third : ℕ)
  (people_after_third : ℕ) :
  initial = 50 →
  got_off_first = 15 →
  got_off_second = 8 →
  got_on_second = 2 →
  got_off_third = 4 →
  people_after_third = 28 →
  ∃ got_on_third : ℕ, got_on_third = 3 :=
by
  sorry

end people_got_on_at_third_stop_l1185_118524


namespace gcd_of_180_and_450_l1185_118537

theorem gcd_of_180_and_450 : Int.gcd 180 450 = 90 := 
  sorry

end gcd_of_180_and_450_l1185_118537


namespace jason_flame_time_l1185_118511

-- Define firing interval and flame duration
def firing_interval := 15
def flame_duration := 5

-- Define the function to calculate seconds per minute
def seconds_per_minute (interval : ℕ) (duration : ℕ) : ℕ :=
  (60 / interval) * duration

-- Theorem to state the problem
theorem jason_flame_time : 
  seconds_per_minute firing_interval flame_duration = 20 := 
by
  sorry

end jason_flame_time_l1185_118511


namespace largest_factor_and_smallest_multiple_of_18_l1185_118565

theorem largest_factor_and_smallest_multiple_of_18 :
  (∃ x, (x ∈ {d : ℕ | d ∣ 18}) ∧ (∀ y, y ∈ {d : ℕ | d ∣ 18} → y ≤ x) ∧ x = 18)
  ∧ (∃ y, (y ∈ {m : ℕ | 18 ∣ m}) ∧ (∀ z, z ∈ {m : ℕ | 18 ∣ m} → y ≤ z) ∧ y = 18) :=
by
  sorry

end largest_factor_and_smallest_multiple_of_18_l1185_118565


namespace last_part_length_l1185_118520

-- Definitions of the conditions
def total_length : ℝ := 74.5
def part1_length : ℝ := 15.5
def part2_length : ℝ := 21.5
def part3_length : ℝ := 21.5

-- Theorem statement to prove the length of the last part of the race
theorem last_part_length :
  (total_length - (part1_length + part2_length + part3_length)) = 16 := 
  by 
    sorry

end last_part_length_l1185_118520


namespace subtraction_result_l1185_118508

theorem subtraction_result: (3.75 - 1.4 = 2.35) :=
by
  sorry

end subtraction_result_l1185_118508


namespace product_decrease_l1185_118544

variable (a b : ℤ)

theorem product_decrease : (a - 3) * (b + 3) - a * b = 900 → a - b = 303 → a * b - (a + 3) * (b - 3) = 918 :=
by
    intros h1 h2
    sorry

end product_decrease_l1185_118544


namespace trapezoid_diagonal_intersection_l1185_118535

theorem trapezoid_diagonal_intersection (PQ RS PR : ℝ) (h1 : PQ = 3 * RS) (h2 : PR = 15) :
  ∃ RT : ℝ, RT = 15 / 4 :=
by
  have RT := 15 / 4
  use RT
  sorry

end trapezoid_diagonal_intersection_l1185_118535


namespace pure_imaginary_real_zero_l1185_118561

theorem pure_imaginary_real_zero (a : ℝ) (i : ℂ) (hi : i^2 = -1) (h : a * i = 0 + a * i) : a = 0 := by
  sorry

end pure_imaginary_real_zero_l1185_118561


namespace find_numbers_l1185_118549

theorem find_numbers :
  ∃ (x y z : ℕ), x = y + 75 ∧ 
                 (x * y = z + 1000) ∧
                 (z = 227 * y + 113) ∧
                 (x = 234) ∧ 
                 (y = 159) := by
  sorry

end find_numbers_l1185_118549


namespace complete_sets_characterization_l1185_118546

-- Definition of a complete set
def complete_set (A : Set ℕ) : Prop :=
  ∀ {a b : ℕ}, (a + b ∈ A) → (a * b ∈ A)

-- Theorem stating that the complete sets of natural numbers are exactly
-- {1}, {1, 2}, {1, 2, 3}, {1, 2, 3, 4}, ℕ.
theorem complete_sets_characterization :
  ∀ (A : Set ℕ), complete_set A ↔ (A = {1} ∨ A = {1, 2} ∨ A = {1, 2, 3} ∨ A = {1, 2, 3, 4} ∨ A = Set.univ) :=
sorry

end complete_sets_characterization_l1185_118546


namespace five_pow_sum_of_squares_l1185_118522

theorem five_pow_sum_of_squares (n : ℕ) : ∃ a b : ℕ, 5^n = a^2 + b^2 := 
sorry

end five_pow_sum_of_squares_l1185_118522


namespace smallest_n_identity_matrix_l1185_118567

noncomputable def rotation_45_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![
    ![Real.cos (Real.pi / 4), -Real.sin (Real.pi / 4)],
    ![Real.sin (Real.pi / 4), Real.cos (Real.pi / 4)]
  ]

theorem smallest_n_identity_matrix : ∃ n : ℕ, n > 0 ∧ (rotation_45_matrix ^ n = 1) ∧ ∀ m : ℕ, m > 0 → (rotation_45_matrix ^ m = 1 → n ≤ m) := sorry

end smallest_n_identity_matrix_l1185_118567


namespace parabola_range_proof_l1185_118555

noncomputable def parabola_range (a : ℝ) : Prop := 
  (-2 ≤ a ∧ a < 3) → 
  ∃ b : ℝ, b = a^2 + 2*a + 4 ∧ (3 ≤ b ∧ b < 19)

theorem parabola_range_proof (a : ℝ) (h : -2 ≤ a ∧ a < 3) : 
  ∃ b : ℝ, b = a^2 + 2*a + 4 ∧ (3 ≤ b ∧ b < 19) :=
sorry

end parabola_range_proof_l1185_118555


namespace range_of_a_l1185_118556

noncomputable def f (a x : ℝ) : ℝ := (a^2 - 2*a - 3)*x^2 + (a - 3)*x + 1

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, ∃ y : ℝ, f a x = y) ∧ 
  (∀ y : ℝ, ∃ x : ℝ, f a x = y) ↔ a = -1 := 
by
  sorry

end range_of_a_l1185_118556


namespace certain_number_is_166_l1185_118560

theorem certain_number_is_166 :
  ∃ x : ℕ, x - 78 =  (4 - 30) + 114 ∧ x = 166 := by
  sorry

end certain_number_is_166_l1185_118560


namespace direct_proportion_function_l1185_118584

-- Define the conditions for the problem
def condition1 (m : ℝ) : Prop := m ^ 2 - 1 = 0
def condition2 (m : ℝ) : Prop := m - 1 ≠ 0

-- The main theorem we need to prove
theorem direct_proportion_function (m : ℝ) (h1 : condition1 m) (h2 : condition2 m) : m = -1 :=
by
  sorry

end direct_proportion_function_l1185_118584


namespace number_of_2_dollar_socks_l1185_118551

-- Given conditions
def total_pairs (a b c : ℕ) := a + b + c = 15
def total_cost (a b c : ℕ) := 2 * a + 4 * b + 5 * c = 41
def min_each_pair (a b c : ℕ) := a ≥ 1 ∧ b ≥ 1 ∧ c ≥ 1

-- To be proved
theorem number_of_2_dollar_socks (a b c : ℕ) (h1 : total_pairs a b c) (h2 : total_cost a b c) (h3 : min_each_pair a b c) : 
  a = 11 := 
  sorry

end number_of_2_dollar_socks_l1185_118551


namespace probability_personA_not_personB_l1185_118515

theorem probability_personA_not_personB :
  let n := Nat.choose 5 3
  let m := Nat.choose 1 1 * Nat.choose 3 2
  (m / n : ℚ) = 3 / 10 :=
by
  -- Proof omitted
  sorry

end probability_personA_not_personB_l1185_118515


namespace ABC_three_digit_number_l1185_118572

theorem ABC_three_digit_number : 
    ∃ (A B C : ℕ), 
    A ≠ B ∧ B ≠ C ∧ A ≠ C ∧ 
    3 * C % 10 = 8 ∧ 
    3 * B + 1 % 10 = 8 ∧ 
    3 * A + 2 = 8 ∧ 
    100 * A + 10 * B + C = 296 := 
by
  sorry

end ABC_three_digit_number_l1185_118572


namespace food_needed_for_vacation_l1185_118574

-- Define the conditions
def daily_food_per_dog := 250 -- in grams
def number_of_dogs := 4
def number_of_days := 14

-- Define the proof problem
theorem food_needed_for_vacation :
  (daily_food_per_dog * number_of_dogs * number_of_days / 1000) = 14 :=
by
  sorry

end food_needed_for_vacation_l1185_118574


namespace determine_parabola_equation_l1185_118568

-- Define the conditions
def focus_on_line (focus : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, focus = (k - 2, k / 2 - 1)

-- Define the result equations
def is_standard_equation (eq : ℝ → ℝ → Prop) : Prop :=
  (∀ x y : ℝ, eq x y → x^2 = 4 * y) ∨ (∀ x y : ℝ, eq x y → y^2 = -8 * x)

-- Define the theorem stating that given the condition,
-- the standard equation is one of the two forms
theorem determine_parabola_equation (focus : ℝ × ℝ) (H : focus_on_line focus) :
  ∃ eq : ℝ → ℝ → Prop, is_standard_equation eq :=
sorry

end determine_parabola_equation_l1185_118568


namespace cost_per_set_l1185_118595

variable (C : ℝ)

theorem cost_per_set :
  let total_manufacturing_cost := 10000 + 500 * C
  let revenue := 500 * 50
  let profit := revenue - total_manufacturing_cost
  profit = 5000 → C = 20 := 
by
  sorry

end cost_per_set_l1185_118595


namespace number_of_ah_tribe_residents_l1185_118587

theorem number_of_ah_tribe_residents 
  (P A U : Nat) 
  (H1 : 16 < P) 
  (H2 : P ≤ 17) 
  (H3 : A + U = P) 
  (H4 : U = 2) : 
  A = 15 := 
by
  sorry

end number_of_ah_tribe_residents_l1185_118587


namespace value_of_a_sum_l1185_118590

theorem value_of_a_sum (a_7 a_6 a_5 a_4 a_3 a_2 a_1 a : ℝ) :
  (∀ x : ℝ, (3 * x - 1)^7 = a_7 * x^7 + a_6 * x^6 + a_5 * x^5 + a_4 * x^4 + a_3 * x^3 + a_2 * x^2 + a_1 * x + a) →
  a + a_1 + a_2 + a_3 + a_4 + a_5 + a_6 + a_7 = 128 := 
by
  sorry

end value_of_a_sum_l1185_118590


namespace table_tennis_matches_l1185_118539

theorem table_tennis_matches (n : ℕ) :
  ∃ x : ℕ, 3 * 2 - x + n * (n - 1) / 2 = 50 ∧ x = 1 :=
by
  sorry

end table_tennis_matches_l1185_118539


namespace opposite_meaning_for_option_C_l1185_118598

def opposite_meaning (a b : Int) : Bool :=
  (a > 0 ∧ b < 0) ∨ (a < 0 ∧ b > 0)

theorem opposite_meaning_for_option_C :
  (opposite_meaning 300 (-500)) ∧ 
  ¬ (opposite_meaning 5 (-5)) ∧ 
  ¬ (opposite_meaning 180 90) ∧ 
  ¬ (opposite_meaning 1 (-1)) :=
by
  unfold opposite_meaning
  sorry

end opposite_meaning_for_option_C_l1185_118598


namespace binary_multiplication_l1185_118596

theorem binary_multiplication : (10101 : ℕ) * (101 : ℕ) = 1101001 :=
by sorry

end binary_multiplication_l1185_118596


namespace inequality_solution_l1185_118509

theorem inequality_solution {x : ℝ} (h : 2 * x + 1 > x + 2) : x > 1 :=
by
  sorry

end inequality_solution_l1185_118509


namespace hawksbill_to_green_turtle_ratio_l1185_118519

theorem hawksbill_to_green_turtle_ratio (total_turtles : ℕ) (green_turtles : ℕ) (hawksbill_turtles : ℕ) (h1 : green_turtles = 800) (h2 : total_turtles = 3200) (h3 : hawksbill_turtles = total_turtles - green_turtles) :
  hawksbill_turtles / green_turtles = 3 :=
by {
  sorry
}

end hawksbill_to_green_turtle_ratio_l1185_118519


namespace kataleya_total_amount_paid_l1185_118502

/-- A store offers a $2 discount for every $10 purchase on any item in the store.
Kataleya went to the store and bought 400 peaches sold at forty cents each.
Prove that the total amount of money she paid at the store for the fruits is $128. -/
theorem kataleya_total_amount_paid : 
  let price_per_peach : ℝ := 0.40
  let number_of_peaches : ℝ := 400 
  let total_cost : ℝ := number_of_peaches * price_per_peach
  let discount_per_10_dollars : ℝ := 2
  let number_of_discounts := total_cost / 10
  let total_discount := number_of_discounts * discount_per_10_dollars
  let amount_paid := total_cost - total_discount
  amount_paid = 128 :=
by
  sorry

end kataleya_total_amount_paid_l1185_118502


namespace senior_citizen_tickets_l1185_118506

theorem senior_citizen_tickets (A S : ℕ) 
  (h1 : A + S = 510) 
  (h2 : 21 * A + 15 * S = 8748) : 
  S = 327 :=
by 
  -- Proof steps are omitted as instructed
  sorry

end senior_citizen_tickets_l1185_118506


namespace lattice_points_in_region_l1185_118530

theorem lattice_points_in_region :
  ∃ n : ℕ, n = 12 ∧ 
  ( ∀ x y : ℤ, (y = x ∨ y = -x ∨ y = -x^2 + 4) → n = 12) :=
by
  sorry

end lattice_points_in_region_l1185_118530


namespace volume_of_prism_l1185_118586

theorem volume_of_prism (x y z : ℝ) (h1 : x * y = 60)
                                     (h2 : y * z = 75)
                                     (h3 : x * z = 100) :
  x * y * z = 671 :=
by
  sorry

end volume_of_prism_l1185_118586


namespace ben_daily_spending_l1185_118591

variable (S : ℕ)

def daily_savings (S : ℕ) : ℕ := 50 - S

def total_savings (S : ℕ) : ℕ := 7 * daily_savings S

def final_amount (S : ℕ) : ℕ := 2 * total_savings S + 10

theorem ben_daily_spending :
  final_amount 15 = 500 :=
by
  unfold final_amount
  unfold total_savings
  unfold daily_savings
  sorry

end ben_daily_spending_l1185_118591


namespace no_representation_of_form_eight_k_plus_3_or_5_l1185_118554

theorem no_representation_of_form_eight_k_plus_3_or_5 (k : ℤ) :
  ∀ x y : ℤ, (8 * k + 3 ≠ x^2 - 2 * y^2) ∧ (8 * k + 5 ≠ x^2 - 2 * y^2) :=
by sorry

end no_representation_of_form_eight_k_plus_3_or_5_l1185_118554


namespace ratio_of_perimeters_l1185_118563

theorem ratio_of_perimeters (s1 s2 : ℝ) (h : s1^2 / s2^2 = 16 / 81) : 
    s1 / s2 = 4 / 9 :=
by
  -- Proof goes here
  sorry

end ratio_of_perimeters_l1185_118563


namespace cube_distance_l1185_118580

-- The Lean 4 statement
theorem cube_distance (side_length : ℝ) (h1 h2 h3 : ℝ) (r s t : ℕ) 
  (h1_eq : h1 = 18) (h2_eq : h2 = 20) (h3_eq : h3 = 22) (side_length_eq : side_length = 15) :
  r = 57 ∧ s = 597 ∧ t = 3 ∧ r + s + t = 657 :=
by
  sorry

end cube_distance_l1185_118580


namespace problem1_problem2_l1185_118512

-- Define the given angle
def given_angle (α : ℝ) : Prop := α = 2010

-- Define the theorem for the first problem
theorem problem1 (α : ℝ) (k : ℤ) (β : ℝ) (h₁ : given_angle α) 
  (h₂ : 0 ≤ β ∧ β < 360) (h₃ : α = k * 360 + β) : 
  -- Assert that α is in the third quadrant
  (190 ≤ β ∧ β < 270 → true) :=
sorry

-- Define the theorem for the second problem
theorem problem2 (α : ℝ) (θ : ℝ) (h₁ : given_angle α)
  (h₂ : -360 ≤ θ ∧ θ < 720)
  (h₃ : ∃ k : ℤ, θ = α + k * 360) : 
  θ = -150 ∨ θ = 210 ∨ θ = 570 :=
sorry

end problem1_problem2_l1185_118512


namespace monthly_average_growth_rate_eq_l1185_118503

theorem monthly_average_growth_rate_eq (x : ℝ) :
  16 * (1 + x)^2 = 25 :=
sorry

end monthly_average_growth_rate_eq_l1185_118503


namespace jill_arrives_before_jack_l1185_118540

def pool_distance : ℝ := 2
def jill_speed : ℝ := 12
def jack_speed : ℝ := 4
def hours_to_minutes (hours : ℝ) : ℝ := hours * 60

theorem jill_arrives_before_jack
    (d : ℝ) (v_jill : ℝ) (v_jack : ℝ) (convert : ℝ → ℝ)
    (h_d : d = pool_distance)
    (h_vj : v_jill = jill_speed)
    (h_vk : v_jack = jack_speed)
    (h_convert : convert = hours_to_minutes) :
  convert (d / v_jack) - convert (d / v_jill) = 20 := by
  sorry

end jill_arrives_before_jack_l1185_118540


namespace kaylee_more_boxes_to_sell_l1185_118588

-- Definitions for the conditions
def total_needed_boxes : ℕ := 33
def sold_to_aunt : ℕ := 12
def sold_to_mother : ℕ := 5
def sold_to_neighbor : ℕ := 4

-- Target proof goal
theorem kaylee_more_boxes_to_sell :
  total_needed_boxes - (sold_to_aunt + sold_to_mother + sold_to_neighbor) = 12 :=
sorry

end kaylee_more_boxes_to_sell_l1185_118588


namespace series_proof_l1185_118532

noncomputable def series_sum (a b : ℝ) : ℝ :=
  ∑' (n : ℕ), a / (b ^ (n + 1))

noncomputable def transformed_series_sum (a b : ℝ) : ℝ :=
  ∑' (n : ℕ), a / ((a + 2 * b) ^ (n + 1))

theorem series_proof (a b : ℝ)
  (h1 : series_sum a b = 7)
  (h2 : a = 7 * (b - 1)) :
  transformed_series_sum a b = 7 * (b - 1) / (9 * b - 8) :=
by sorry

end series_proof_l1185_118532


namespace find_number_l1185_118562

theorem find_number (x : ℕ) (h : (9 * x) / 3 = 27) : x = 9 :=
by
  sorry

end find_number_l1185_118562


namespace evaluate_f_l1185_118527

def f (x : ℝ) : ℝ := x^2 + 4*x - 3

theorem evaluate_f (x : ℝ) : f (x + 1) = x^2 + 6*x + 2 :=
by 
  -- The proof is omitted
  sorry

end evaluate_f_l1185_118527


namespace total_cakes_served_l1185_118594

-- Define the conditions
def cakes_lunch_today := 5
def cakes_dinner_today := 6
def cakes_yesterday := 3

-- Define the theorem we want to prove
theorem total_cakes_served : (cakes_lunch_today + cakes_dinner_today + cakes_yesterday) = 14 :=
by
  -- The proof is not required, so we use sorry to skip it
  sorry

end total_cakes_served_l1185_118594


namespace ink_length_figure_4_ink_length_difference_9_8_ink_length_figure_100_l1185_118553

-- Define the ink length of a figure
def ink_length (n : ℕ) : ℕ := 5 * n

-- Part (a): Determine the ink length of Figure 4.
theorem ink_length_figure_4 : ink_length 4 = 20 := by
  sorry

-- Part (b): Determine the difference between the ink length of Figure 9 and the ink length of Figure 8.
theorem ink_length_difference_9_8 : ink_length 9 - ink_length 8 = 5 := by
  sorry

-- Part (c): Determine the ink length of Figure 100.
theorem ink_length_figure_100 : ink_length 100 = 500 := by
  sorry

end ink_length_figure_4_ink_length_difference_9_8_ink_length_figure_100_l1185_118553


namespace sequence_formula_l1185_118501

theorem sequence_formula (a : ℕ → ℚ) (h₁ : a 1 = 1) (h_recurrence : ∀ n : ℕ, 2 * n * a n + 1 = (n + 1) * a n) :
  ∀ n : ℕ, a n = n / 2^(n - 1) :=
sorry

end sequence_formula_l1185_118501


namespace product_evaluation_l1185_118523

noncomputable def product_term (n : ℕ) : ℚ :=
  1 - (1 / (n * n))

noncomputable def product_expression : ℚ :=
  10 * 71 * (product_term 2) * (product_term 3) * (product_term 4) * (product_term 5) *
  (product_term 6) * (product_term 7) * (product_term 8) * (product_term 9) * (product_term 10)

theorem product_evaluation : product_expression = 71 := by
  sorry

end product_evaluation_l1185_118523


namespace tradesman_gain_l1185_118579

-- Let's define a structure representing the tradesman's buying and selling operation.
structure Trade where
  true_value : ℝ
  defraud_rate : ℝ
  buy_price : ℕ
  sell_price : ℕ

theorem tradesman_gain (T : Trade) (H1 : T.defraud_rate = 0.2) (H2 : T.true_value = 100)
  (H3 : T.buy_price = T.true_value * (1 - T.defraud_rate))
  (H4 : T.sell_price = T.true_value * (1 + T.defraud_rate)) :
  ((T.sell_price - T.buy_price) / T.buy_price) * 100 = 50 := 
by
  sorry

end tradesman_gain_l1185_118579


namespace range_of_x_l1185_118504

theorem range_of_x (S : ℕ → ℕ) (a : ℕ → ℕ) (x : ℕ) :
  (∀ n, n ≥ 2 → S (n - 1) + S n = 2 * n^2 + 1) →
  S 0 = 0 →
  a 1 = x →
  (∀ n, a n ≤ a (n + 1)) →
  2 < x ∧ x < 3 := 
sorry

end range_of_x_l1185_118504


namespace diameter_inscribed_circle_l1185_118570

noncomputable def diameter_of_circle (r : ℝ) : ℝ :=
2 * r

theorem diameter_inscribed_circle (r : ℝ) (h : 8 * r = π * r ^ 2) : diameter_of_circle r = 16 / π := by
  sorry

end diameter_inscribed_circle_l1185_118570


namespace wilson_sledding_l1185_118593

variable (T : ℕ)

theorem wilson_sledding :
  (4 * T) + 6 = 14 → T = 2 :=
by
  intros h
  sorry

end wilson_sledding_l1185_118593


namespace find_q_minus_p_l1185_118526

theorem find_q_minus_p (p q : ℕ) (h1 : 0 < p) (h2 : 0 < q) 
  (h3 : 6 * q < 11 * p) (h4 : 9 * p < 5 * q) (h_min : ∀ r : ℕ, r > 0 → (6:ℚ)/11 < (p:ℚ)/r → (p:ℚ)/r < (5:ℚ)/9 → q ≤ r) :
  q - p = 9 :=
sorry

end find_q_minus_p_l1185_118526


namespace slope_at_two_l1185_118533

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := x^3 + a * x^2 + b * x + a^2
noncomputable def f' (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := 3 * x^2 + 2 * a * x + b

theorem slope_at_two (a b : ℝ) (h1 : f' 1 a b = 0) (h2 : f 1 a b = 10) :
  f' 2 4 (-11) = 17 :=
sorry

end slope_at_two_l1185_118533


namespace evaluate_expression_l1185_118599

def f (x : ℝ) : ℝ := 3 * x^2 + 5 * x - 7

theorem evaluate_expression : 3 * f 2 - 2 * f (-2) = 55 := by
  sorry

end evaluate_expression_l1185_118599


namespace width_of_property_l1185_118577

theorem width_of_property (W : ℝ) 
  (h1 : ∃ w l, (w = W / 8) ∧ (l = 2250 / 10) ∧ (w * l = 28125)) : W = 1000 :=
by
  -- Formal proof here
  sorry

end width_of_property_l1185_118577


namespace proposition_only_A_l1185_118566

def is_proposition (statement : String) : Prop := sorry

def statement_A : String := "Red beans grow in the southern country"
def statement_B : String := "They sprout several branches in spring"
def statement_C : String := "I hope you pick more"
def statement_D : String := "For these beans symbolize longing"

theorem proposition_only_A :
  is_proposition statement_A ∧
  ¬is_proposition statement_B ∧
  ¬is_proposition statement_C ∧
  ¬is_proposition statement_D := 
sorry

end proposition_only_A_l1185_118566


namespace probability_of_red_ball_l1185_118547

noncomputable def total_balls : Nat := 4 + 2
noncomputable def red_balls : Nat := 2

theorem probability_of_red_ball :
  (red_balls : ℚ) / (total_balls : ℚ) = 1 / 3 :=
sorry

end probability_of_red_ball_l1185_118547


namespace product_of_roots_quadratic_l1185_118592

noncomputable def product_of_roots (a b c : ℝ) : ℝ :=
  let x1 := (-b + Real.sqrt (b^2 - 4 * a * c)) / (2 * a)
  let x2 := (-b - Real.sqrt (b^2 - 4 * a * c)) / (2 * a)
  x1 * x2

theorem product_of_roots_quadratic :
  (product_of_roots 1 3 (-5)) = -5 :=
by
  sorry

end product_of_roots_quadratic_l1185_118592


namespace arithmetic_sequence_8th_term_l1185_118585

theorem arithmetic_sequence_8th_term (a d : ℤ) 
  (h1 : a + 3 * d = 23)
  (h2 : a + 5 * d = 47) : 
  a + 7 * d = 71 := 
by 
  sorry

end arithmetic_sequence_8th_term_l1185_118585


namespace distinct_roots_polynomial_l1185_118573

theorem distinct_roots_polynomial (a b : ℂ) (h₁ : a ≠ b) (h₂: a^3 + 3*a^2 + a + 1 = 0) (h₃: b^3 + 3*b^2 + b + 1 = 0) :
  a^2 * b + a * b^2 + 3 * a * b = 1 :=
sorry

end distinct_roots_polynomial_l1185_118573


namespace cost_of_1500_pencils_l1185_118597

theorem cost_of_1500_pencils (cost_per_box : ℕ) (pencils_per_box : ℕ) (num_pencils : ℕ) :
  cost_per_box = 30 → pencils_per_box = 100 → num_pencils = 1500 → 
  (num_pencils * (cost_per_box / pencils_per_box) = 450) :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  simp
  sorry

end cost_of_1500_pencils_l1185_118597


namespace integral_cos8_0_2pi_l1185_118525

noncomputable def definite_integral_cos8 (a b : ℝ) : ℝ :=
  ∫ x in a..b, (Real.cos (x / 4)) ^ 8

theorem integral_cos8_0_2pi :
  definite_integral_cos8 0 (2 * Real.pi) = (35 * Real.pi) / 64 :=
by
  sorry

end integral_cos8_0_2pi_l1185_118525


namespace largest_ordered_pair_exists_l1185_118543

-- Define the condition for ordered pairs (a, b)
def ordered_pair_condition (a b : ℤ) : Prop :=
  1 ≤ a ∧ a ≤ b ∧ b ≤ 100 ∧ ∃ (k : ℤ), (a + b) * (a + b + 1) = k * a * b

-- Define the specific ordered pair to be checked
def specific_pair (a b : ℤ) : Prop :=
  a = 35 ∧ b = 90

-- The main statement to be proven
theorem largest_ordered_pair_exists : specific_pair 35 90 ∧ ordered_pair_condition 35 90 :=
by
  sorry

end largest_ordered_pair_exists_l1185_118543


namespace lucy_total_journey_l1185_118545

-- Define the length of Lucy's journey
def lucy_journey (x : ℝ) : Prop :=
  (1 / 4) * x + 25 + (1 / 6) * x = x

-- State the theorem
theorem lucy_total_journey : ∃ x : ℝ, lucy_journey x ∧ x = 300 / 7 := by
  sorry

end lucy_total_journey_l1185_118545


namespace solve_for_x_l1185_118516

theorem solve_for_x (x : ℤ) (h_eq : (7 * x - 5) / (x - 2) = 2 / (x - 2)) (h_cond : x ≠ 2) : x = 1 := by
  sorry

end solve_for_x_l1185_118516


namespace abc_zero_l1185_118550

theorem abc_zero {a b c : ℝ} 
(h1 : (a + b) * (b + c) * (c + a) = a * b * c)
(h2 : (a^3 + b^3) * (b^3 + c^3) * (c^3 + a^3) = a^3 * b^3 * c^3) : 
a * b * c = 0 := 
by sorry

end abc_zero_l1185_118550


namespace non_neg_integers_l1185_118569

open Nat

theorem non_neg_integers (n : ℕ) :
  (∃ x y k : ℕ, x.gcd y = 1 ∧ k ≥ 2 ∧ 3^n = x^k + y^k) ↔ (n = 0 ∨ n = 1 ∨ n = 2) := by
  sorry

end non_neg_integers_l1185_118569


namespace correct_average_l1185_118581

theorem correct_average (S' : ℝ) (a a' b b' c c' : ℝ) (n : ℕ) 
  (incorrect_avg : S' / n = 22) 
  (a_eq : a = 52) (a'_eq : a' = 32)
  (b_eq : b = 47) (b'_eq : b' = 27) 
  (c_eq : c = 68) (c'_eq : c' = 45)
  (n_eq : n = 12) 
  : ((S' - (a' + b' + c') + (a + b + c)) / 12 = 27.25) := 
by
  sorry

end correct_average_l1185_118581


namespace general_term_formula_l1185_118558

-- Define the problem parameters
variables (a : ℤ)

-- Definitions based on the conditions
def first_term : ℤ := a - 1
def second_term : ℤ := a + 1
def third_term : ℤ := 2 * a + 3

-- Define the theorem to prove the general term formula
theorem general_term_formula :
  2 * (first_term a + 1) = first_term a + third_term a → a = 0 →
  ∀ n : ℕ, a_n = 2 * n - 3 := 
by
  intro h1 h2
  sorry

end general_term_formula_l1185_118558


namespace four_digit_numbers_using_0_and_9_l1185_118557

theorem four_digit_numbers_using_0_and_9 :
  {n : ℕ | 1000 ≤ n ∧ n < 10000 ∧ ∀ d, d ∈ Nat.digits 10 n → (d = 0 ∨ d = 9)} = {9000, 9009, 9090, 9099, 9900, 9909, 9990, 9999} :=
by
  sorry

end four_digit_numbers_using_0_and_9_l1185_118557


namespace initial_amount_of_liquid_A_l1185_118514

theorem initial_amount_of_liquid_A (A B : ℝ) (initial_ratio : A = 4 * B) (removed_mixture : ℝ) (new_ratio : (A - (4/5) * removed_mixture) = (2 / 3) * ((B - (1/5) * removed_mixture) + removed_mixture)) :
  A = 16 := 
  sorry

end initial_amount_of_liquid_A_l1185_118514


namespace inequality_proof_l1185_118529

noncomputable def a : Real := (1 / 3) ^ Real.pi
noncomputable def b : Real := (1 / 3) ^ (1 / 2 : Real)
noncomputable def c : Real := Real.pi ^ (1 / 2 : Real)

theorem inequality_proof : a < b ∧ b < c :=
by
  -- Proof will be provided here
  sorry

end inequality_proof_l1185_118529


namespace no_primes_of_form_2pow5m_plus_2powm_plus_1_l1185_118589

theorem no_primes_of_form_2pow5m_plus_2powm_plus_1 {m : ℕ} (hm : m > 0) : ¬ (Prime (2^(5*m) + 2^m + 1)) :=
by
  sorry

end no_primes_of_form_2pow5m_plus_2powm_plus_1_l1185_118589


namespace inequality_holds_for_all_m_l1185_118575

theorem inequality_holds_for_all_m (m : ℝ) (h1 : ∀ (x : ℝ), x^2 - 8 * x + 20 > 0)
  (h2 : m < -1/2) : ∀ (x : ℝ), (x ^ 2 - 8 * x + 20) / (m * x ^ 2 + 2 * (m + 1) * x + 9 * m + 4) < 0 :=
by
  sorry

end inequality_holds_for_all_m_l1185_118575


namespace find_positive_m_has_exactly_single_solution_l1185_118507

theorem find_positive_m_has_exactly_single_solution :
  ∃ m : ℝ, 0 < m ∧ (∀ x : ℝ, 16 * x^2 + m * x + 4 = 0 → x = 16) :=
sorry

end find_positive_m_has_exactly_single_solution_l1185_118507


namespace at_most_one_zero_l1185_118559

-- Definition of the polynomial f(x)
def f (n : ℤ) (x : ℝ) : ℝ :=
  x^4 - 1994 * x^3 + (1993 + n) * x^2 - 11 * x + n

-- The target theorem statement
theorem at_most_one_zero (n : ℤ) : ∃! x : ℝ, f n x = 0 :=
by
  sorry

end at_most_one_zero_l1185_118559


namespace production_difference_l1185_118578

variables (p h : ℕ)

def first_day_production := p * h

def second_day_production := (p + 5) * (h - 3)

-- Given condition
axiom p_eq_3h : p = 3 * h

theorem production_difference : first_day_production p h - second_day_production p h = 4 * h + 15 :=
by
  sorry

end production_difference_l1185_118578


namespace number_description_l1185_118517

theorem number_description :
  4 * 10000 + 3 * 1000 + 7 * 100 + 5 * 10 + 2 + 8 / 10 + 4 / 100 = 43752.84 :=
by
  sorry

end number_description_l1185_118517


namespace boys_number_l1185_118564

variable (M W B : ℕ)

-- Conditions
axiom h1 : M = W
axiom h2 : W = B
axiom h3 : M * 8 = 120

theorem boys_number :
  B = 15 := by
  sorry

end boys_number_l1185_118564


namespace min_value_x1_x2_l1185_118518

theorem min_value_x1_x2 (a x_1 x_2 : ℝ) (h_a_pos : 0 < a) (h_sol_set : x_1 + x_2 = 4 * a) (h_prod_set : x_1 * x_2 = 3 * a^2) : 
  x_1 + x_2 + a / (x_1 * x_2) = 4 * a + 1 / (3 * a) :=
sorry

end min_value_x1_x2_l1185_118518


namespace problem_solution_l1185_118528

theorem problem_solution (n : ℤ) : 
  (1 / (n + 2) + 3 / (n + 2) + 2 * n / (n + 2) = 4) → (n = -2) :=
by
  intro h
  sorry

end problem_solution_l1185_118528


namespace triangle_area_ratio_l1185_118521

theorem triangle_area_ratio (x y : ℝ) (n m : ℕ) (hn : n > 0) (hm : m > 0) :
  let A_area := (1/2) * (y/n) * (x/2)
  let B_area := (1/2) * (x/m) * (y/2)
  A_area / B_area = m / n := by
  sorry

end triangle_area_ratio_l1185_118521


namespace ratio_of_pieces_l1185_118531

theorem ratio_of_pieces (total_length shorter_piece longer_piece : ℕ) 
    (h1 : total_length = 6) (h2 : shorter_piece = 2)
    (h3 : longer_piece = total_length - shorter_piece) :
    ((longer_piece : ℚ) / (shorter_piece : ℚ)) = 2 :=
by
    sorry

end ratio_of_pieces_l1185_118531


namespace strawberry_unit_prices_l1185_118513

theorem strawberry_unit_prices (x y : ℝ) (h1 : x = 1.5 * y) (h2 : 2 * x - 2 * y = 10) : x = 15 ∧ y = 10 :=
by
  sorry

end strawberry_unit_prices_l1185_118513


namespace triangle_subsegment_length_l1185_118510

theorem triangle_subsegment_length (DF DE EF DG GF : ℚ)
  (h_ratio : ∃ x : ℚ, DF = 3 * x ∧ DE = 4 * x ∧ EF = 5 * x)
  (h_EF_len : EF = 20)
  (h_angle_bisector : DG + GF = DE ∧ DG / GF = DE / DF) :
  DF < DE ∧ DE < EF →
  min DG GF = 48 / 7 :=
by
  sorry

end triangle_subsegment_length_l1185_118510


namespace more_apples_than_pears_l1185_118541

-- Definitions based on conditions
def total_fruits : ℕ := 85
def apples : ℕ := 48

-- Statement to prove
theorem more_apples_than_pears : (apples - (total_fruits - apples)) = 11 := by
  -- proof steps
  sorry

end more_apples_than_pears_l1185_118541


namespace number_of_monkeys_l1185_118571

theorem number_of_monkeys (N : ℕ)
  (h1 : N * 1 * 8 = 8)
  (h2 : 3 * 1 * 8 = 3 * 8) :
  N = 8 :=
sorry

end number_of_monkeys_l1185_118571


namespace find_heaviest_or_lightest_l1185_118576

theorem find_heaviest_or_lightest (stones : Fin 10 → ℝ)
  (h_distinct: ∀ i j : Fin 10, i ≠ j → stones i ≠ stones j)
  (h_pairwise_sums_distinct : ∀ i j k l : Fin 10, 
    i ≠ j → k ≠ l → stones i + stones j ≠ stones k + stones l) :
  (∃ i : Fin 10, ∀ j : Fin 10, stones i ≥ stones j) ∨ 
  (∃ i : Fin 10, ∀ j : Fin 10, stones i ≤ stones j) :=
sorry

end find_heaviest_or_lightest_l1185_118576


namespace difference_in_pencil_buyers_l1185_118552

theorem difference_in_pencil_buyers :
  ∀ (cost_per_pencil : ℕ) (total_cost_eighth_graders : ℕ) (total_cost_fifth_graders : ℕ), 
  cost_per_pencil = 13 →
  total_cost_eighth_graders = 234 →
  total_cost_fifth_graders = 325 →
  (total_cost_fifth_graders / cost_per_pencil) - (total_cost_eighth_graders / cost_per_pencil) = 7 :=
by
  intros cost_per_pencil total_cost_eighth_graders total_cost_fifth_graders 
         hcpe htc8 htc5
  sorry

end difference_in_pencil_buyers_l1185_118552


namespace jane_earnings_l1185_118500

def age_of_child (jane_start_age : ℕ) (child_factor : ℕ) : ℕ :=
  jane_start_age / child_factor

def babysit_rate (age : ℕ) : ℕ :=
  if age < 2 then 5
  else if age <= 5 then 7
  else 8

def amount_earned (hours rate : ℕ) : ℕ := 
  hours * rate

def total_earnings (earnings : List ℕ) : ℕ :=
  earnings.foldl (·+·) 0

theorem jane_earnings
  (jane_start_age : ℕ := 18)
  (child_A_hours : ℕ := 50)
  (child_B_hours : ℕ := 90)
  (child_C_hours : ℕ := 130)
  (child_D_hours : ℕ := 70) :
  let child_A_age := age_of_child jane_start_age 2
  let child_B_age := child_A_age - 2
  let child_C_age := child_B_age + 3
  let child_D_age := child_C_age
  let earnings_A := amount_earned child_A_hours (babysit_rate child_A_age)
  let earnings_B := amount_earned child_B_hours (babysit_rate child_B_age)
  let earnings_C := amount_earned child_C_hours (babysit_rate child_C_age)
  let earnings_D := amount_earned child_D_hours (babysit_rate child_D_age)
  total_earnings [earnings_A, earnings_B, earnings_C, earnings_D] = 2720 :=
by
  sorry

end jane_earnings_l1185_118500


namespace fraction_lost_l1185_118534

-- Definitions of the given conditions
def initial_pencils : ℕ := 30
def lost_pencils_initially : ℕ := 6
def current_pencils : ℕ := 16

-- Statement of the proof problem
theorem fraction_lost (initial_pencils lost_pencils_initially current_pencils : ℕ) :
  let remaining_pencils := initial_pencils - lost_pencils_initially
  let lost_remaining_pencils := remaining_pencils - current_pencils
  (lost_remaining_pencils : ℚ) / remaining_pencils = 1 / 3 :=
by
  let remaining_pencils := initial_pencils - lost_pencils_initially
  let lost_remaining_pencils := remaining_pencils - current_pencils
  sorry

end fraction_lost_l1185_118534


namespace regular_polygon_sides_l1185_118542

theorem regular_polygon_sides (n : ℕ) (h : n > 0) (h_exterior_angle : 360 / n = 10) : n = 36 :=
by sorry

end regular_polygon_sides_l1185_118542


namespace exists_integer_n_tangent_l1185_118536
open Real

noncomputable def degree_to_radian (d : ℝ) : ℝ :=
  d * (π / 180)

theorem exists_integer_n_tangent :
  ∃ (n : ℤ), -90 < (n : ℝ) ∧ (n : ℝ) < 90 ∧ tan (degree_to_radian (n : ℝ)) = tan (degree_to_radian 345) ∧ n = -15 :=
by
  sorry

end exists_integer_n_tangent_l1185_118536


namespace junk_mail_per_red_or_white_house_l1185_118582

noncomputable def pieces_per_house (total_pieces : ℕ) (total_houses : ℕ) : ℕ := 
  total_pieces / total_houses

noncomputable def total_pieces_for_type (pieces_per_house : ℕ) (houses_of_type : ℕ) : ℕ := 
  pieces_per_house * houses_of_type

noncomputable def total_pieces_for_red_or_white 
  (total_pieces : ℕ)
  (total_houses : ℕ)
  (white_houses : ℕ)
  (red_houses : ℕ) : ℕ :=
  let pieces_per_house := pieces_per_house total_pieces total_houses
  let pieces_for_white := total_pieces_for_type pieces_per_house white_houses
  let pieces_for_red := total_pieces_for_type pieces_per_house red_houses
  pieces_for_white + pieces_for_red

theorem junk_mail_per_red_or_white_house :
  ∀ (total_pieces : ℕ) (total_houses : ℕ) (white_houses : ℕ) (red_houses : ℕ),
    total_pieces = 48 →
    total_houses = 8 →
    white_houses = 2 →
    red_houses = 3 →
    total_pieces_for_red_or_white total_pieces total_houses white_houses red_houses / (white_houses + red_houses) = 6 :=
by
  intros
  sorry

end junk_mail_per_red_or_white_house_l1185_118582
