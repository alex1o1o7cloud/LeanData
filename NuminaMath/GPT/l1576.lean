import Mathlib

namespace gcd_lcm_identity_l1576_157653

def GCD (a b : ℕ) : ℕ := Nat.gcd a b
def LCM (a b : ℕ) : ℕ := a * (b / GCD a b)

theorem gcd_lcm_identity (a b c : ℕ) :
    (LCM a (LCM b c))^2 / (LCM a b * LCM b c * LCM c a) = (GCD a (GCD b c))^2 / (GCD a b * GCD b c * GCD c a) :=
by
  sorry

end gcd_lcm_identity_l1576_157653


namespace largest_integer_mod_l1576_157647

theorem largest_integer_mod (a : ℕ) (h₁ : a < 100) (h₂ : a % 5 = 2) : a = 97 :=
by sorry

end largest_integer_mod_l1576_157647


namespace yield_and_fertilization_correlated_l1576_157672

-- Define the variables and conditions
def yield_of_crops : Type := sorry
def fertilization : Type := sorry

-- State the condition
def yield_depends_on_fertilization (Y : yield_of_crops) (F : fertilization) : Prop :=
  -- The yield of crops depends entirely on fertilization
  sorry

-- State the theorem with the given condition and the conclusion
theorem yield_and_fertilization_correlated {Y : yield_of_crops} {F : fertilization} :
  yield_depends_on_fertilization Y F → sorry := 
  -- There is a correlation between the yield of crops and fertilization
  sorry

end yield_and_fertilization_correlated_l1576_157672


namespace man_l1576_157660

-- Define the speeds and values given in the problem conditions
def man_speed_with_current : ℝ := 15
def speed_of_current : ℝ := 2.5
def man_speed_against_current : ℝ := 10

-- Define the man's speed in still water as a variable
def man_speed_in_still_water : ℝ := man_speed_with_current - speed_of_current

-- The theorem we need to prove
theorem man's_speed_against_current_is_correct :
  (man_speed_in_still_water - speed_of_current = man_speed_against_current) :=
by
  -- Placeholder for proof
  sorry

end man_l1576_157660


namespace horizontal_asymptote_value_l1576_157699

theorem horizontal_asymptote_value :
  ∀ (x : ℝ),
  ((8 * x^4 + 6 * x^3 + 7 * x^2 + 2 * x + 4) / 
  (2 * x^4 + 5 * x^3 + 3 * x^2 + x + 6)) = (4 : ℝ) :=
by sorry

end horizontal_asymptote_value_l1576_157699


namespace least_possible_integer_for_friends_statements_l1576_157669

theorem least_possible_integer_for_friends_statements 
    (M : Nat)
    (statement_divisible_by : Nat → Prop)
    (h1 : ∀ n, 1 ≤ n ∧ n ≤ 30 → statement_divisible_by n = (M % n = 0))
    (h2 : ∃ m, 1 ≤ m ∧ m < 30 ∧ (statement_divisible_by m = false ∧ 
                                    statement_divisible_by (m + 1) = false)) :
    M = 12252240 :=
by
  sorry

end least_possible_integer_for_friends_statements_l1576_157669


namespace mathematicians_contemporaries_probability_l1576_157657

noncomputable def probability_contemporaries : ℚ :=
  let overlap_area : ℚ := 129600
  let total_area : ℚ := 360000
  overlap_area / total_area

theorem mathematicians_contemporaries_probability :
  probability_contemporaries = 18 / 25 :=
by
  sorry

end mathematicians_contemporaries_probability_l1576_157657


namespace bucket_capacity_l1576_157620

theorem bucket_capacity (x : ℝ) (h1 : 24 * x = 36 * 9) : x = 13.5 :=
by 
  sorry

end bucket_capacity_l1576_157620


namespace marie_erasers_l1576_157635

theorem marie_erasers (initial_erasers : ℕ) (lost_erasers : ℕ) (final_erasers : ℕ) :
  initial_erasers = 95 → lost_erasers = 42 → final_erasers = initial_erasers - lost_erasers → final_erasers = 53 :=
by
  intros h_initial h_lost h_final
  rw [h_initial, h_lost] at h_final
  exact h_final

end marie_erasers_l1576_157635


namespace find_p_minus_q_l1576_157654

theorem find_p_minus_q (x y p q : ℤ) (hx : x ≠ 0) (hy : y ≠ 0) (hp : p ≠ 0) (hq : q ≠ 0)
  (h1 : 3 / (x * p) = 8) (h2 : 5 / (y * q) = 18)
  (hminX : ∀ x', x' ≠ 0 → 3 / (x' * 3) ≠ 1 / 8)
  (hminY : ∀ y', y' ≠ 0 → 5 / (y' * 5) ≠ 1 / 18) :
  p - q = 0 :=
sorry

end find_p_minus_q_l1576_157654


namespace jaime_saves_enough_l1576_157648

-- Definitions of the conditions
def weekly_savings : ℕ := 50
def bi_weekly_expense : ℕ := 46
def target_savings : ℕ := 135

-- The proof goal
theorem jaime_saves_enough : ∃ weeks : ℕ, 2 * ((weeks * weekly_savings - bi_weekly_expense) / 2) = target_savings := 
sorry

end jaime_saves_enough_l1576_157648


namespace sum_of_series_l1576_157698

theorem sum_of_series :
  (∑' n : ℕ, (3^n) / (3^(3^n) + 1)) = 1 / 2 :=
sorry

end sum_of_series_l1576_157698


namespace negation_of_exist_prop_l1576_157630

theorem negation_of_exist_prop :
  (¬ ∃ x : ℝ, x^2 - x + 2 > 0) ↔ (∀ x : ℝ, x^2 - x + 2 ≤ 0) :=
by {
  sorry
}

end negation_of_exist_prop_l1576_157630


namespace john_speed_above_limit_l1576_157683

def distance : ℝ := 150
def time : ℝ := 2
def speed_limit : ℝ := 60

theorem john_speed_above_limit :
  distance / time - speed_limit = 15 :=
by
  sorry

end john_speed_above_limit_l1576_157683


namespace arithmetic_sequence_sum_l1576_157617

variable (a : ℕ → ℤ)

def arithmetic_sequence_condition_1 := a 5 = 3
def arithmetic_sequence_condition_2 := a 6 = -2

theorem arithmetic_sequence_sum :
  arithmetic_sequence_condition_1 a →
  arithmetic_sequence_condition_2 a →
  a 3 + a 4 + a 5 + a 6 + a 7 + a 8 = 3 :=
by
  intros h1 h2
  sorry

end arithmetic_sequence_sum_l1576_157617


namespace marla_adds_blue_paint_l1576_157665

variable (M B : ℝ)

theorem marla_adds_blue_paint :
  (20 = 0.10 * M) ∧ (B = 0.70 * M) → B = 140 := 
by 
  sorry

end marla_adds_blue_paint_l1576_157665


namespace min_balloon_count_l1576_157678

theorem min_balloon_count 
(R B : ℕ) (burst_red burst_blue : ℕ) 
(h1 : R = 7 * B) 
(h2 : burst_red = burst_blue / 3) 
(h3 : burst_red ≥ 1) :
R + B = 24 :=
by 
    sorry

end min_balloon_count_l1576_157678


namespace find_a_plus_b_l1576_157641

noncomputable def vector_a : ℝ × ℝ := (-1, 2)
noncomputable def vector_b (m : ℝ) : ℝ × ℝ := (m, 1)

def parallel_condition (a b : ℝ × ℝ) : Prop :=
  (a.1 * b.2 - a.2 * b.1 = 0)

theorem find_a_plus_b (m : ℝ) (h_parallel: 
  parallel_condition (⟨vector_a.1 + 2 * (vector_b m).1, vector_a.2 + 2 * (vector_b m).2⟩)
                     (⟨2 * vector_a.1 - (vector_b m).1, 2 * vector_a.2 - (vector_b m).2⟩)) :
  vector_a + vector_b (-1/2) = (-3/2, 3) := 
by
  sorry

end find_a_plus_b_l1576_157641


namespace prove_total_bill_is_correct_l1576_157687

noncomputable def totalCostAfterDiscounts : ℝ :=
  let adultsMealsCost := 8 * 12
  let teenagersMealsCost := 4 * 10
  let childrenMealsCost := 3 * 7
  let adultsSodasCost := 8 * 3.5
  let teenagersSodasCost := 4 * 3.5
  let childrenSodasCost := 3 * 1.8
  let appetizersCost := 4 * 8
  let dessertsCost := 5 * 5

  let subtotal := adultsMealsCost + teenagersMealsCost + childrenMealsCost +
                  adultsSodasCost + teenagersSodasCost + childrenSodasCost +
                  appetizersCost + dessertsCost

  let discountAdultsMeals := 0.10 * adultsMealsCost
  let discountDesserts := 5
  let discountChildrenMealsAndSodas := 0.15 * (childrenMealsCost + childrenSodasCost)

  let adjustedSubtotal := subtotal - discountAdultsMeals - discountDesserts - discountChildrenMealsAndSodas

  let additionalDiscount := if subtotal > 200 then 0.05 * adjustedSubtotal else 0
  let total := adjustedSubtotal - additionalDiscount

  total

theorem prove_total_bill_is_correct : totalCostAfterDiscounts = 230.70 :=
by sorry

end prove_total_bill_is_correct_l1576_157687


namespace find_amount_after_two_years_l1576_157676

noncomputable def initial_value : ℝ := 64000
noncomputable def yearly_increase (amount : ℝ) : ℝ := amount / 9
noncomputable def amount_after_year (amount : ℝ) : ℝ := amount + yearly_increase amount
noncomputable def amount_after_two_years : ℝ := amount_after_year (amount_after_year initial_value)

theorem find_amount_after_two_years : amount_after_two_years = 79012.34 :=
by
  sorry

end find_amount_after_two_years_l1576_157676


namespace largest_prime_factor_of_12321_l1576_157655

-- Definitions based on the given conditions
def n := 12321
def a := 111
def p₁ := 3
def p₂ := 37

-- Given conditions as hypotheses
theorem largest_prime_factor_of_12321 (h1 : n = a^2) (h2 : a = p₁ * p₂) (hp₁_prime : Prime p₁) (hp₂_prime : Prime p₂) :
  p₂ = 37 ∧ ∀ p, Prime p → p ∣ n → p ≤ 37 := 
by 
  sorry

end largest_prime_factor_of_12321_l1576_157655


namespace sets_equal_l1576_157644

theorem sets_equal :
  {u : ℤ | ∃ (m n l : ℤ), u = 12 * m + 8 * n + 4 * l} =
  {u : ℤ | ∃ (p q r : ℤ), u = 20 * p + 16 * q + 12 * r} := 
sorry

end sets_equal_l1576_157644


namespace average_books_per_month_l1576_157679

-- Definitions based on the conditions
def books_sold_january : ℕ := 15
def books_sold_february : ℕ := 16
def books_sold_march : ℕ := 17
def total_books_sold : ℕ := books_sold_january + books_sold_february + books_sold_march
def number_of_months : ℕ := 3

-- The theorem we need to prove
theorem average_books_per_month : total_books_sold / number_of_months = 16 :=
by
  sorry

end average_books_per_month_l1576_157679


namespace students_in_class_l1576_157603

def total_eggs : Nat := 56
def eggs_per_student : Nat := 8
def num_students : Nat := 7

theorem students_in_class :
  total_eggs / eggs_per_student = num_students :=
by
  sorry

end students_in_class_l1576_157603


namespace train_speed_kmph_l1576_157639

def length_of_train : ℝ := 120
def time_to_cross_bridge : ℝ := 17.39860811135109
def length_of_bridge : ℝ := 170

theorem train_speed_kmph : 
  (length_of_train + length_of_bridge) / time_to_cross_bridge * 3.6 = 60 := 
by
  sorry

end train_speed_kmph_l1576_157639


namespace ratio_y_to_x_l1576_157613

-- Define the setup as given in the conditions
variables (c x y : ℝ)

-- Condition 1: Selling price x results in a loss of 20%
def condition1 : Prop := x = 0.80 * c

-- Condition 2: Selling price y results in a profit of 25%
def condition2 : Prop := y = 1.25 * c

-- Theorem: Prove the ratio of y to x is 25/16 given the conditions
theorem ratio_y_to_x (c : ℝ) (h1 : condition1 c x) (h2 : condition2 c y) : y / x = 25 / 16 := 
sorry

end ratio_y_to_x_l1576_157613


namespace time_to_cross_pole_correct_l1576_157686

-- Definitions of the conditions
def trainSpeed_kmh : ℝ := 120 -- km/hr
def trainLength_m : ℝ := 300 -- meters

-- Assumed conversions
def kmToMeters : ℝ := 1000 -- meters in a km
def hoursToSeconds : ℝ := 3600 -- seconds in an hour

-- Conversion of speed from km/hr to m/s
noncomputable def trainSpeed_ms := (trainSpeed_kmh * kmToMeters) / hoursToSeconds

-- Time to cross the pole
noncomputable def timeToCrossPole := trainLength_m / trainSpeed_ms

-- The theorem stating the proof problem
theorem time_to_cross_pole_correct : timeToCrossPole = 9 := by
  sorry

end time_to_cross_pole_correct_l1576_157686


namespace cody_books_second_week_l1576_157604

noncomputable def total_books := 54
noncomputable def books_first_week := 6
noncomputable def books_weeks_after_second := 9
noncomputable def total_weeks := 7

theorem cody_books_second_week :
  let b2 := total_books - (books_first_week + books_weeks_after_second * (total_weeks - 2))
  b2 = 3 :=
by
  sorry

end cody_books_second_week_l1576_157604


namespace power_function_odd_f_m_plus_1_l1576_157611

noncomputable def f (x : ℝ) (m : ℝ) := x^(2 + m)

theorem power_function_odd_f_m_plus_1 (m : ℝ) (h_odd : ∀ x : ℝ, f (-x) m = -f x m)
  (h_domain : -1 ≤ m) : f (m + 1) m = 1 := by
  sorry

end power_function_odd_f_m_plus_1_l1576_157611


namespace apples_handout_l1576_157637

theorem apples_handout {total_apples pies_needed pies_count handed_out : ℕ}
  (h1 : total_apples = 51)
  (h2 : pies_needed = 5)
  (h3 : pies_count = 2)
  (han : handed_out = total_apples - (pies_needed * pies_count)) :
  handed_out = 41 :=
by {
  sorry
}

end apples_handout_l1576_157637


namespace sqrt_47_minus_2_range_l1576_157602

theorem sqrt_47_minus_2_range (h : 6 < Real.sqrt 47 ∧ Real.sqrt 47 < 7) : 4 < Real.sqrt 47 - 2 ∧ Real.sqrt 47 - 2 < 5 := by
  sorry

end sqrt_47_minus_2_range_l1576_157602


namespace triangle_angle_identity_l1576_157685

def triangle_angles_arithmetic_sequence (A B C : ℝ) : Prop :=
  A + C = 2 * B

def sum_of_triangle_angles (A B C : ℝ) : Prop :=
  A + B + C = 180

def angle_B_is_60 (B : ℝ) : Prop :=
  B = 60

theorem triangle_angle_identity (A B C a b c : ℝ)
  (h1 : triangle_angles_arithmetic_sequence A B C)
  (h2 : sum_of_triangle_angles A B C)
  (h3 : angle_B_is_60 B) : 
  1 / (a + b) + 1 / (b + c) = 3 / (a + b + c) :=
by 
  sorry

end triangle_angle_identity_l1576_157685


namespace polynomial_abs_sum_l1576_157692

theorem polynomial_abs_sum (a_0 a_1 a_2 a_3 a_4 a_5 : ℤ)
  (h : (2*X - 1)^5 = a_5 * X^5 + a_4 * X^4 + a_3 * X^3 + a_2 * X^2 + a_1 * X + a_0) :
  |a_0| + |a_1| + |a_2| + |a_3| + |a_4| + |a_5| = 243 :=
by
  sorry

end polynomial_abs_sum_l1576_157692


namespace shirts_production_l1576_157626

-- Definitions
def constant_rate (r : ℕ) : Prop := ∀ n : ℕ, 8 * n * r = 160 * n

theorem shirts_production (r : ℕ) (h : constant_rate r) : 16 * r = 32 :=
by sorry

end shirts_production_l1576_157626


namespace area_of_isosceles_trapezoid_l1576_157624

def isIsoscelesTrapezoid (a b c h : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a ∧ h ^ 2 = c ^ 2 - ((b - a) / 2) ^ 2

theorem area_of_isosceles_trapezoid :
  ∀ (a b c : ℝ), 
    a = 8 → b = 14 → c = 5 →
    ∃ h: ℝ, isIsoscelesTrapezoid a b c h ∧ ((a + b) / 2 * h = 44) :=
by
  intros a b c ha hb hc
  sorry

end area_of_isosceles_trapezoid_l1576_157624


namespace bella_bakes_most_cookies_per_batch_l1576_157643

theorem bella_bakes_most_cookies_per_batch (V : ℝ) :
  let alex_cookies := V / 9
  let bella_cookies := V / 7
  let carlo_cookies := V / 8
  let dana_cookies := V / 10
  alex_cookies < bella_cookies ∧ carlo_cookies < bella_cookies ∧ dana_cookies < bella_cookies :=
sorry

end bella_bakes_most_cookies_per_batch_l1576_157643


namespace average_last_4_matches_l1576_157623

theorem average_last_4_matches (avg_10_matches avg_6_matches : ℝ) (matches_10 matches_6 matches_4 : ℕ) :
  avg_10_matches = 38.9 →
  avg_6_matches = 41 →
  matches_10 = 10 →
  matches_6 = 6 →
  matches_4 = 4 →
  (avg_10_matches * matches_10 - avg_6_matches * matches_6) / matches_4 = 35.75 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end average_last_4_matches_l1576_157623


namespace B_catches_up_with_A_l1576_157605

theorem B_catches_up_with_A :
  let d := 140
  let vA := 10
  let vB := 20
  let tA := d / vA
  let tB := d / vB
  tA - tB = 7 := 
by
  -- Definitions
  let d := 140
  let vA := 10
  let vB := 20
  let tA := d / vA
  let tB := d / vB
  -- Goal
  show tA - tB = 7
  sorry

end B_catches_up_with_A_l1576_157605


namespace minimum_n_l1576_157670

-- Assume the sequence a_n is defined as part of an arithmetic sequence
def arithmetic_sequence (a d : ℕ) (n : ℕ) : ℕ := a + n * d

-- Define S_n as the sum of the first n terms in the sequence
def sum_arithmetic_sequence (a d n : ℕ) : ℕ := n * a + (n * (n - 1)) / 2 * d

-- Given conditions
def a1 := 2
def d := 1  -- Derived from the condition a1 + a4 = a5

-- Problem Statement
theorem minimum_n (n : ℕ) :
  (sum_arithmetic_sequence a1 d n > 32) ↔ n = 6 :=
sorry

end minimum_n_l1576_157670


namespace minimum_ticket_cost_l1576_157600

theorem minimum_ticket_cost :
  let num_people := 12
  let num_adults := 8
  let num_children := 4
  let adult_ticket_cost := 100
  let child_ticket_cost := 50
  let group_ticket_cost := 70
  num_people = num_adults + num_children →
  (num_people >= 10) →
  ∃ (cost : ℕ), cost = min (num_adults * adult_ticket_cost + num_children * child_ticket_cost) (group_ticket_cost * num_people) ∧
  cost = min (group_ticket_cost * 10 + child_ticket_cost * (num_people - 10)) (group_ticket_cost * num_people) →
  cost = 800 :=
by
  intro h1 h2
  sorry

end minimum_ticket_cost_l1576_157600


namespace geo_seq_a3_equals_one_l1576_157615

theorem geo_seq_a3_equals_one (a : ℕ → ℝ) (h_pos : ∀ n, 0 < a n) (h_T5 : a 1 * a 2 * a 3 * a 4 * a 5 = 1) : a 3 = 1 :=
sorry

end geo_seq_a3_equals_one_l1576_157615


namespace inequality_l1576_157684

variable (a b c : ℝ)

noncomputable def condition (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a * b * c = 1 / 8

theorem inequality (h : condition a b c) : 
  a^2 + b^2 + c^2 + a^2 * b^2 + a^2 * c^2 + b^2 * c^2 ≥ 15 / 16 :=
sorry

end inequality_l1576_157684


namespace final_amount_after_bets_l1576_157608

theorem final_amount_after_bets :
  let initial_amount := 128
  let num_bets := 8
  let num_wins := 4
  let num_losses := 4
  let bonus_per_win_after_loss := 10
  let win_multiplier := 3 / 2
  let loss_multiplier := 1 / 2
  ∃ final_amount : ℝ,
    (final_amount =
      initial_amount * (win_multiplier ^ num_wins) * (loss_multiplier ^ num_losses) + 2 * bonus_per_win_after_loss) ∧
    final_amount = 60.5 :=
sorry

end final_amount_after_bets_l1576_157608


namespace exists_special_function_l1576_157667

theorem exists_special_function : ∃ (s : ℚ → ℤ), (∀ x y : ℚ, x ≠ y → (x * y = 1 ∨ x + y = 0 ∨ x + y = 1) → s x * s y = -1) ∧ (∀ x : ℚ, s x = 1 ∨ s x = -1) :=
by
  sorry

end exists_special_function_l1576_157667


namespace proof_problem_l1576_157673

noncomputable def a {α : Type*} [LinearOrderedField α] : α := sorry
noncomputable def b {α : Type*} [LinearOrderedField α] : α := sorry
noncomputable def c {α : Type*} [LinearOrderedField α] : α := sorry
noncomputable def d {α : Type*} [LinearOrderedField α] : α := sorry

theorem proof_problem (a b c d : ℝ) (hpos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) 
(hprod : a * b * c * d = 1) : 
a^5 + b^5 + c^5 + d^5 ≥ a + b + c + d :=
sorry

end proof_problem_l1576_157673


namespace sum_of_coefficients_is_1_l1576_157601

-- Given conditions:
def polynomial_expansion (x y : ℤ) := (x - 2 * y) ^ 18

-- Proof statement:
theorem sum_of_coefficients_is_1 : (polynomial_expansion 1 1) = 1 := by
  -- The proof itself is omitted as per the instruction
  sorry

end sum_of_coefficients_is_1_l1576_157601


namespace sqrt_two_between_one_and_two_l1576_157659

theorem sqrt_two_between_one_and_two : 1 < Real.sqrt 2 ∧ Real.sqrt 2 < 2 := 
by
  -- sorry placeholder
  sorry

end sqrt_two_between_one_and_two_l1576_157659


namespace probability_red_or_black_probability_red_black_or_white_l1576_157618

theorem probability_red_or_black (total_balls red_balls black_balls : ℕ) : 
  total_balls = 12 → red_balls = 5 → black_balls = 4 → 
  (red_balls + black_balls) / total_balls = 3 / 4 :=
by
  intros
  sorry

theorem probability_red_black_or_white (total_balls red_balls black_balls white_balls : ℕ) :
  total_balls = 12 → red_balls = 5 → black_balls = 4 → white_balls = 2 → 
  (red_balls + black_balls + white_balls) / total_balls = 11 / 12 :=
by
  intros
  sorry

end probability_red_or_black_probability_red_black_or_white_l1576_157618


namespace like_terms_sum_l1576_157661

theorem like_terms_sum (n m : ℕ) 
  (h1 : n + 1 = 3) 
  (h2 : m - 1 = 3) : 
  m + n = 6 := 
  sorry

end like_terms_sum_l1576_157661


namespace product_gt_one_l1576_157640

theorem product_gt_one 
  (m : ℚ) (b : ℚ)
  (hm : m = 3 / 4)
  (hb : b = 5 / 2) :
  m * b > 1 := 
by
  sorry

end product_gt_one_l1576_157640


namespace square_difference_l1576_157628

theorem square_difference :
  153^2 - 147^2 = 1800 :=
by
  sorry

end square_difference_l1576_157628


namespace length_width_percentage_change_l1576_157664

variables (L W : ℝ) (x : ℝ)
noncomputable def area_change_percent : ℝ :=
  (L * (1 + x / 100) * W * (1 - x / 100) - L * W) / (L * W) * 100

theorem length_width_percentage_change (h : area_change_percent L W x = 4) :
  x = 20 :=
by
  sorry

end length_width_percentage_change_l1576_157664


namespace smallest_real_solution_l1576_157619

theorem smallest_real_solution (x : ℝ) : 
  (x * |x| = 3 * x + 4) → x = 4 :=
by {
  sorry -- Proof omitted as per the instructions
}

end smallest_real_solution_l1576_157619


namespace jugglers_balls_needed_l1576_157658

theorem jugglers_balls_needed (juggler_count balls_per_juggler : ℕ)
  (h_juggler_count : juggler_count = 378)
  (h_balls_per_juggler : balls_per_juggler = 6) :
  juggler_count * balls_per_juggler = 2268 :=
by
  -- This is where the proof would go.
  sorry

end jugglers_balls_needed_l1576_157658


namespace total_money_shared_l1576_157696

-- Conditions
def ken_share : ℕ := 1750
def tony_share : ℕ := 2 * ken_share

-- Question and proof to be demonstrated
theorem total_money_shared : ken_share + tony_share = 5250 :=
by sorry

end total_money_shared_l1576_157696


namespace gcd_of_powers_of_two_l1576_157610

noncomputable def m := 2^2048 - 1
noncomputable def n := 2^2035 - 1

theorem gcd_of_powers_of_two : Int.gcd m n = 8191 := by
  sorry

end gcd_of_powers_of_two_l1576_157610


namespace quadratic_real_roots_l1576_157636

theorem quadratic_real_roots (b : ℝ) : 
  (∃ x : ℝ, x^2 + b * x + 25 = 0) ↔ b ∈ Set.Iic (-10) ∪ Set.Ici 10 := 
by 
  sorry

end quadratic_real_roots_l1576_157636


namespace sum_of_base_8_digits_888_l1576_157690

def base_8_representation (n : ℕ) : ℕ := 
  let d0 := n % 8
  let n  := n / 8
  let d1 := n % 8
  let n  := n / 8
  let d2 := n % 8
  let n  := n / 8
  let d3 := n % 8
  d3 * 1000 + d2 * 100 + d1 * 10 + d0

def sum_of_digits (n : ℕ) : ℕ :=
  n / 1000 + (n % 1000) / 100 + (n % 100) / 10 + (n % 10)

theorem sum_of_base_8_digits_888 : 
  sum_of_digits (base_8_representation 888) = 13 :=
by
  sorry

end sum_of_base_8_digits_888_l1576_157690


namespace valve_solution_l1576_157668

noncomputable def valve_problem : Prop :=
  ∀ (x y z : ℝ),
  (1 / (x + y + z) = 2) →
  (1 / (x + z) = 4) →
  (1 / (y + z) = 3) →
  (1 / (x + y) = 2.4)

theorem valve_solution : valve_problem :=
by
  -- proof omitted
  intros x y z h1 h2 h3
  sorry

end valve_solution_l1576_157668


namespace negation_of_universal_l1576_157675

theorem negation_of_universal :
  (¬ (∀ x : ℝ, x > 0 → x^2 + x ≥ 0)) ↔ (∃ x_0 : ℝ, x_0 > 0 ∧ x_0^2 + x_0 < 0) :=
by
  sorry

end negation_of_universal_l1576_157675


namespace neq_is_necessary_but_not_sufficient_l1576_157629

theorem neq_is_necessary_but_not_sufficient (a b : ℝ) : (a ≠ b) → ¬ (∀ a b : ℝ, (a ≠ b) → (a / b + b / a > 2)) ∧ (∀ a b : ℝ, (a / b + b / a > 2) → (a ≠ b)) :=
by {
    sorry
}

end neq_is_necessary_but_not_sufficient_l1576_157629


namespace least_positive_integer_addition_l1576_157632

theorem least_positive_integer_addition (k : ℕ) (h₀ : 525 + k % 5 = 0) (h₁ : 0 < k) : k = 5 := 
by
  sorry

end least_positive_integer_addition_l1576_157632


namespace basketballs_count_l1576_157652

theorem basketballs_count (x : ℕ) : 
  let num_volleyballs := x
  let num_basketballs := 2 * x
  let num_soccer_balls := x - 8
  num_volleyballs + num_basketballs + num_soccer_balls = 100 →
  num_basketballs = 54 :=
by
  intros h
  sorry

end basketballs_count_l1576_157652


namespace calculate_abc_over_def_l1576_157656

theorem calculate_abc_over_def
  (a b c d e f : ℚ)
  (h1 : a / b = 1 / 3)
  (h2 : b / c = 2)
  (h3 : c / d = 1 / 2)
  (h4 : d / e = 3)
  (h5 : e / f = 1 / 2) :
  (a * b * c) / (d * e * f) = 1 / 2 :=
by
  sorry

end calculate_abc_over_def_l1576_157656


namespace problem_mod_1000_l1576_157680

noncomputable def M : ℕ := Nat.choose 18 9

theorem problem_mod_1000 : M % 1000 = 620 := by
  sorry

end problem_mod_1000_l1576_157680


namespace shifted_function_is_correct_l1576_157693

-- Define the original function
def original_function (x : ℝ) : ℝ := -2 * x

-- Define the shifted function
def shifted_function (x : ℝ) : ℝ := original_function (x - 3)

-- State the theorem to be proven
theorem shifted_function_is_correct :
  ∀ x : ℝ, shifted_function x = -2 * x + 6 :=
by
  sorry

end shifted_function_is_correct_l1576_157693


namespace perimeter_of_large_square_l1576_157694

theorem perimeter_of_large_square (squares : List ℕ) (h : squares = [1, 1, 2, 3, 5, 8, 13]) : 2 * (21 + 13) = 68 := by
  sorry

end perimeter_of_large_square_l1576_157694


namespace largest_r_satisfying_condition_l1576_157651

theorem largest_r_satisfying_condition :
  ∃ M : ℕ, ∀ (a : ℕ → ℕ) (r : ℝ) (h : ∀ n : ℕ, a n ≤ a (n + 2) ∧ a (n + 2) ≤ Real.sqrt (a n ^ 2 + r * a (n + 1))),
  (∀ n : ℕ, n ≥ M → a (n + 2) = a n) → r = 2 := 
by
  sorry

end largest_r_satisfying_condition_l1576_157651


namespace ratio_alisha_to_todd_is_two_to_one_l1576_157649

-- Definitions
def total_gumballs : ℕ := 45
def todd_gumballs : ℕ := 4
def bobby_gumballs (A : ℕ) : ℕ := 4 * A - 5
def remaining_gumballs : ℕ := 6

-- Condition stating Hector's gumball distribution
def hector_gumballs_distribution (A : ℕ) : Prop :=
  todd_gumballs + A + bobby_gumballs A + remaining_gumballs = total_gumballs

-- Definition for the ratio of the gumballs given to Alisha to Todd
def ratio_alisha_todd (A : ℕ) : ℕ × ℕ :=
  (A / 4, todd_gumballs / 4)

-- Theorem stating the problem
theorem ratio_alisha_to_todd_is_two_to_one : ∃ (A : ℕ), hector_gumballs_distribution A → ratio_alisha_todd A = (2, 1) :=
sorry

end ratio_alisha_to_todd_is_two_to_one_l1576_157649


namespace interest_calculation_correct_l1576_157622

-- Define the principal amounts and their respective interest rates
def principal1 : ℝ := 3000
def rate1 : ℝ := 0.08
def principal2 : ℝ := 8000 - principal1
def rate2 : ℝ := 0.05

-- Calculate interest for one year
def interest1 : ℝ := principal1 * rate1 * 1
def interest2 : ℝ := principal2 * rate2 * 1

-- Define the total interest
def total_interest : ℝ := interest1 + interest2

-- Prove that the total interest calculated is $490
theorem interest_calculation_correct : total_interest = 490 := by
  sorry

end interest_calculation_correct_l1576_157622


namespace some_employee_not_team_leader_l1576_157677

variables (Employee : Type) (isTeamLeader : Employee → Prop) (meetsDeadline : Employee → Prop)

-- Conditions
axiom some_employee_not_meets_deadlines : ∃ e : Employee, ¬ meetsDeadline e
axiom all_team_leaders_meet_deadlines : ∀ e : Employee, isTeamLeader e → meetsDeadline e

-- Theorem to prove
theorem some_employee_not_team_leader : ∃ e : Employee, ¬ isTeamLeader e :=
sorry

end some_employee_not_team_leader_l1576_157677


namespace porter_previous_painting_price_l1576_157646

variable (P : ℝ)

-- Conditions
def condition1 : Prop := 3.5 * P - 1000 = 49000

-- Correct Answer
def answer : ℝ := 14285.71

-- Theorem stating that the answer holds given the conditions
theorem porter_previous_painting_price (h : condition1 P) : P = answer :=
sorry

end porter_previous_painting_price_l1576_157646


namespace rope_length_before_folding_l1576_157674

theorem rope_length_before_folding (L : ℝ) (h : L / 4 = 10) : L = 40 :=
by
  sorry

end rope_length_before_folding_l1576_157674


namespace opposite_of_neg_two_is_two_l1576_157688

theorem opposite_of_neg_two_is_two (a : ℤ) (h : a = -2) : a + 2 = 0 := by
  rw [h]
  norm_num

end opposite_of_neg_two_is_two_l1576_157688


namespace simplify_expression_l1576_157634

theorem simplify_expression (x : ℝ) : (2 * x + 20) + (150 * x + 25) = 152 * x + 45 :=
by
  sorry

end simplify_expression_l1576_157634


namespace leadership_board_stabilizes_l1576_157606

theorem leadership_board_stabilizes :
  ∃ n : ℕ, 2 ^ n - 1 ≤ 2020 ∧ 2020 < 2 ^ (n + 1) - 1 := by
  sorry

end leadership_board_stabilizes_l1576_157606


namespace money_problem_l1576_157627

theorem money_problem
  (A B C : ℕ)
  (h1 : A + B + C = 450)
  (h2 : B + C = 350)
  (h3 : C = 100) :
  A + C = 200 :=
by
  sorry

end money_problem_l1576_157627


namespace number_of_BA3_in_sample_l1576_157625

-- Definitions for the conditions
def strains_BA1 : Nat := 60
def strains_BA2 : Nat := 20
def strains_BA3 : Nat := 40
def total_sample_size : Nat := 30

def total_strains : Nat := strains_BA1 + strains_BA2 + strains_BA3

-- Theorem statement translating to the equivalent proof problem
theorem number_of_BA3_in_sample :
  total_sample_size * strains_BA3 / total_strains = 10 :=
by
  sorry

end number_of_BA3_in_sample_l1576_157625


namespace inequality_inequality_l1576_157612

open Real

theorem inequality_inequality (n : ℕ) (k : ℝ) (hn : 0 < n) (hk : 0 < k) : 
  1 - 1/k ≤ n * (k^(1 / n) - 1) ∧ n * (k^(1 / n) - 1) ≤ k - 1 := 
  sorry

end inequality_inequality_l1576_157612


namespace red_candies_count_l1576_157642

def total_candies : ℕ := 3409
def blue_candies : ℕ := 3264

theorem red_candies_count : total_candies - blue_candies = 145 := by
  sorry

end red_candies_count_l1576_157642


namespace set_clock_correctly_l1576_157691

noncomputable def correct_clock_time
  (T_depart T_arrive T_depart_friend T_return : ℕ) 
  (T_visit := T_depart_friend - T_arrive) 
  (T_return_err := T_return - T_depart) 
  (T_total_travel := T_return_err - T_visit) 
  (T_travel_oneway := T_total_travel / 2) : ℕ :=
  T_depart + T_visit + T_travel_oneway

theorem set_clock_correctly 
  (T_depart T_arrive T_depart_friend T_return : ℕ)
  (h1 : T_depart ≤ T_return) -- The clock runs without accounting for the time away
  (h2 : T_arrive ≤ T_depart_friend) -- The friend's times are correct
  (h3 : T_return ≠ T_depart) -- The man was away for some non-zero duration
: 
  (correct_clock_time T_depart T_arrive T_depart_friend T_return) = 
  (T_depart + (T_depart_friend - T_arrive) + ((T_return - T_depart - (T_depart_friend - T_arrive)) / 2)) :=
sorry

end set_clock_correctly_l1576_157691


namespace sam_current_dimes_l1576_157621

def original_dimes : ℕ := 8
def sister_borrowed : ℕ := 4
def friend_borrowed : ℕ := 2
def sister_returned : ℕ := 2
def friend_returned : ℕ := 1

theorem sam_current_dimes : 
  (original_dimes - sister_borrowed - friend_borrowed + sister_returned + friend_returned = 5) :=
by
  sorry

end sam_current_dimes_l1576_157621


namespace range_of_m_l1576_157666

noncomputable def point := (ℝ × ℝ)
noncomputable def P : point := (-1, 1)
noncomputable def Q : point := (2, 2)
noncomputable def M : point := (0, -1)
noncomputable def line_eq (m : ℝ) := ∀ p : point, p.1 + m * p.2 + m = 0

theorem range_of_m (m : ℝ) (l : line_eq m) : -3 < m ∧ m < -2/3 := 
by
  sorry

end range_of_m_l1576_157666


namespace total_books_in_class_l1576_157633

theorem total_books_in_class (Tables : ℕ) (BooksPerTable : ℕ) (TotalBooks : ℕ) 
  (h1 : Tables = 500)
  (h2 : BooksPerTable = (2 * Tables) / 5)
  (h3 : TotalBooks = Tables * BooksPerTable) :
  TotalBooks = 100000 := 
sorry

end total_books_in_class_l1576_157633


namespace expression_equality_l1576_157671

theorem expression_equality :
  (5 + 2) * (5^2 + 2^2) * (5^4 + 2^4) * (5^8 + 2^8) * (5^16 + 2^16) * (5^32 + 2^32) * (5^64 + 2^64) = 5^128 - 2^128 := 
  sorry

end expression_equality_l1576_157671


namespace part_I_part_I_correct_interval_part_II_min_value_l1576_157697

noncomputable def f (x : ℝ) : ℝ := |2 * x + 1| - |x - 4|

theorem part_I : ∀ x : ℝ, (f x > 2) ↔ ( x < -7 ∨ (5 / 3 < x ∧ x < 4) ∨ x ≥ 4) := sorry

theorem part_I_correct_interval : ∀ x : ℝ, (f x > 2) → (x < -7 ∨ (5 / 3 < x ∧ x < 4) ∨ x ≥ 4) := sorry

theorem part_II_min_value : ∀ x : ℝ, ∃ y : ℝ, y = f x ∧ ∀ x : ℝ, f x ≥ y := 
sorry

end part_I_part_I_correct_interval_part_II_min_value_l1576_157697


namespace problem_statement_l1576_157638

variable (a b c d : ℝ)

noncomputable def circle_condition_1 : Prop := a = (1 : ℝ) / a
noncomputable def circle_condition_2 : Prop := b = (1 : ℝ) / b
noncomputable def circle_condition_3 : Prop := c = (1 : ℝ) / c
noncomputable def circle_condition_4 : Prop := d = (1 : ℝ) / d

theorem problem_statement (h1 : circle_condition_1 a)
                          (h2 : circle_condition_2 b)
                          (h3 : circle_condition_3 c)
                          (h4 : circle_condition_4 d) :
    2 * (a^2 + b^2 + c^2 + d^2) = (a + b + c + d)^2 := 
by
  sorry

end problem_statement_l1576_157638


namespace parabola_equation_l1576_157681

theorem parabola_equation (a : ℝ) :
  (∀ x, (x + 1) * (x - 3) = 0 ↔ x = -1 ∨ x = 3) →
  (∀ y, y = a * (0 + 1) * (0 - 3) → y = 3) →
  a = -1 → 
  (∀ x, y = a * (x + 1) * (x - 3) → y = -x^2 + 2 * x + 3) :=
by
  intros h₁ h₂ ha
  sorry

end parabola_equation_l1576_157681


namespace find_interval_solution_l1576_157607

def interval_solution : Set ℝ := {x | 2 < x / (3 * x - 7) ∧ x / (3 * x - 7) <= 7}

theorem find_interval_solution (x : ℝ) :
  x ∈ interval_solution ↔
  x ∈ Set.Ioc (49 / 20 : ℝ) (14 / 5 : ℝ) := 
sorry

end find_interval_solution_l1576_157607


namespace heather_blocks_l1576_157631

theorem heather_blocks (x : ℝ) (h1 : x + 41 = 127) : x = 86 := by
  sorry

end heather_blocks_l1576_157631


namespace find_h_plus_k_l1576_157614

theorem find_h_plus_k (h k : ℝ) :
  (∀ (x y : ℝ),
    (x - 3) ^ 2 + (y + 4) ^ 2 = 49) → 
  h = 3 ∧ k = -4 → 
  h + k = -1 :=
by
  sorry

end find_h_plus_k_l1576_157614


namespace trajectory_of_moving_point_l1576_157616

theorem trajectory_of_moving_point (P : ℝ × ℝ) (F1 : ℝ × ℝ) (F2 : ℝ × ℝ)
  (hF1 : F1 = (-2, 0)) (hF2 : F2 = (2, 0))
  (h_arith_mean : dist F1 F2 = (dist P F1 + dist P F2) / 2) :
  ∃ a b : ℝ, a = 4 ∧ b^2 = 12 ∧ (P.1^2 / a^2 + P.2^2 / b^2 = 1) :=
sorry

end trajectory_of_moving_point_l1576_157616


namespace distribute_neg3_l1576_157695

theorem distribute_neg3 (x y : ℝ) : -3 * (x - x * y) = -3 * x + 3 * x * y :=
by sorry

end distribute_neg3_l1576_157695


namespace function_equality_l1576_157650

theorem function_equality (f : ℕ → ℕ) 
  (h1 : ∀ n : ℕ, f n < f (n + 1) )
  (h2 : f 2 = 2)
  (h3 : ∀ m n : ℕ, f (m * n) = f m * f n) : 
  ∀ n : ℕ, f n = n :=
by
  sorry

end function_equality_l1576_157650


namespace boys_left_hand_to_girl_l1576_157663

-- Definitions based on the given conditions
def num_boys : ℕ := 40
def num_girls : ℕ := 28
def boys_right_hand_to_girl : ℕ := 18

-- Statement to prove
theorem boys_left_hand_to_girl : (num_boys - (num_boys - boys_right_hand_to_girl)) = boys_right_hand_to_girl := by
  sorry

end boys_left_hand_to_girl_l1576_157663


namespace total_questions_l1576_157682

theorem total_questions (qmc : ℕ) (qtotal : ℕ) (h1 : 10 = qmc) (h2 : qmc = (20 / 100) * qtotal) : qtotal = 50 :=
sorry

end total_questions_l1576_157682


namespace vertical_angles_equal_l1576_157609

-- Define what it means for two angles to be vertical angles.
def are_vertical_angles (α β : ℝ) : Prop :=
  ∃ (γ δ : ℝ), α + γ = 180 ∧ β + δ = 180 ∧ γ = β ∧ δ = α

-- The theorem statement:
theorem vertical_angles_equal (α β : ℝ) : are_vertical_angles α β → α = β := 
  sorry

end vertical_angles_equal_l1576_157609


namespace interval_1_5_frequency_is_0_70_l1576_157689

-- Define the intervals and corresponding frequencies
def intervals : List (ℤ × ℤ) := [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)]

def frequencies : List ℕ := [1, 1, 2, 3, 1, 2]

-- Sample capacity
def sample_capacity : ℕ := 10

-- Calculate the frequency of the sample in the interval [1,5)
noncomputable def frequency_in_interval_1_5 : ℝ := (frequencies.take 4).sum / sample_capacity

-- Prove that the frequency in the interval [1,5) is 0.70
theorem interval_1_5_frequency_is_0_70 : frequency_in_interval_1_5 = 0.70 := by
  sorry

end interval_1_5_frequency_is_0_70_l1576_157689


namespace remainder_of_f_when_divided_by_x_plus_2_l1576_157662

def f (x : ℝ) : ℝ := x^4 - 6 * x^3 + 11 * x^2 + 8 * x - 20

theorem remainder_of_f_when_divided_by_x_plus_2 : f (-2) = 72 := by
  sorry

end remainder_of_f_when_divided_by_x_plus_2_l1576_157662


namespace eccentricity_of_ellipse_l1576_157645

theorem eccentricity_of_ellipse :
  ∀ (x y : ℝ), (x^2) / 25 + (y^2) / 16 = 1 → 
  (∃ (e : ℝ), e = 3 / 5) :=
by
  sorry

end eccentricity_of_ellipse_l1576_157645
